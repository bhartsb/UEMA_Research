# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Layer 3, the episode segmenter: each effector's nearby primitive events grouped into time windows.

Temporal segmentation, not image segmentation (addendum section 22). Since detector v6 (2026-09-24)
the body is six EFFECTORS, each with its own episodes, which run side by side and overlap:

  hand_L, hand_R   the arm, the hand and its fingers and wrist (the arm's work is the hand's)
  gaze             the eyes and the head: fixations, head turns, fast head spins, and contacts
                   on the head or face (the head is the eyes' platform: moving it is a gaze act)
  trunk            the trunk and hips, posture changes, weight shifts, and the whole-body fast
                   events (a head jolt, an impact, a loss of support: the fast lane has no part)
  leg_L, leg_R     the thigh, shank and foot

A primitive belongs to an effector by its parts (effectors_of): a contact to the effector of
each infant part it joins (the left foot touching the right foot is an act of both legs); a
curl, digit, wrist, approach, hand view or lift to its hand; a drop to the hand it left; a
fixation, eye movement or head turn to gaze; a posture change or weight shift to the trunk. An
OBJECT MOVING belongs to the effectors whose parts touched it in the 100 ms before it began (the
hand that pushed it, the foot that kicked the mobile); an object that moved with no part on it
was moved by the world (a critter, the caregiver's toy, the mobile swinging on its own) and
opens no episode, as the caregiver's own events do not: episodes are the infant's acts.

Each effector's episode opens at a salient primitive of that effector and absorbs every salient
primitive of the same effector that starts within GAP_S of the episode's current end; it closes
after GAP_S of that effector's quiet, at a break in the record, and at the end of its BLOCK: the
record is cut into absolute blocks of MAX_S (step // 200), and no episode crosses a block
boundary. The blocks are what lets the digester finalize a window under continuous activity:
effectors whose episodes overlap without pause leave no step outside every episode, but every
block boundary is one (daemon._process never cuts through an episode).

Some primitives RIDE (rides()): finger, wrist, closing and head turns that continue a movement,
and two parts of one hand touching each other are tokens of the episode their start falls in, but
open and extend none. Turns in progress cannot mark an act's start (some turn of a hand is in
progress 99.8% of the time), so with every turn opening episodes a hand's episodes never paused and
ran to the block's end in production and babble alike. A head turn that STARTS FROM REST
(head_turns_from_rest) is an act onset and opens a gaze episode, holding it open only to its own
start; since detector v9 so is a hand turn that starts from its own joint's rest and rises within
the act timescale (hand_turns_from_rest), for its hand's episodes: until v8 such turns could not be
seen, because each turn began where the previous one ended, and a hand moving only its fingers
opened no episode. The effector's quiet is then the quiet of its acts: a hand touching something
other than itself, approaching, lifting, dropping or moving an object, coming into view, or a
finger, closing or wrist movement starting from rest; the gaze fixating or the head starting to
turn; the trunk shifting weight or changing posture; a leg touching.

CROSS-EFFECTOR PATTERNS ("the left foot touches the right foot, then the head turns") are PAIR
units: two fresh episodes of different effectors of one body whose starts fall within PAIR_S,
the earlier one leading (same start: the effector listed first leads). A pair's tokens are the
two episodes' opening tokens. PAIR_S is GAP_S: two acts closer than that are one episode when
one effector makes them, so two effectors' acts that close are the same moment's pattern - the
scale at which the global episodes of detector v5 joined acts of different effectors. A FRESH
episode starts at least GAP_S after its block began (after_cap False): one that starts sooner may
only continue activity the block boundary cut, which would pair every busy effector with every
other at each boundary.

The novelty miner counts effector episodes and pair units as its units (novelty.py). Before v6,
episodes were global: any salient primitive kept one episode open, so under continuous activity
every episode ran to its 10 s cap (production 2026-09-24: 88 of 96 and 77 of 77 episodes at the
cap; random-order babble 96 of 104), and novelty counted 10 s windows that contain a token pair.

SALIENT means the infant's acts and what they do to the world (salient() below). The body lying
on and brushing its surroundings is background: contacts with the floor, walls, panels or ceiling
open nothing, nor do eye movements (the gaze's acts are its fixations).

The returned list runs LATEST START FIRST: the digester's single pass over it (daemon._process)
then finds a cut inside no episode even where episodes overlap.
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Set

import numpy as np

from .features import KIND_SELF, Track

GAP_S = 0.5
MAX_S = 10.0          # the block: no episode crosses an absolute multiple of 200 steps
STEP_S = 0.05
PAIR_S = GAP_S        # a cross-effector pair: two fresh episodes starting within 0.5 s
TOUCH_BEFORE_S = 0.1  # an object moving belongs to the parts touching it in the 100 ms before

EFFECTORS = ("hand_L", "hand_R", "gaze", "trunk", "leg_L", "leg_R")
PAIR = "pair"


def token(p: dict) -> str:
    """A primitive as a short, comparable symbol (the novelty miner's alphabet)."""
    t = p["type"]
    if t == "contact":
        a, b = sorted([p["a_class"], p["b_class"]])
        return "touch:%s~%s" % (a, b)
    if t == "curl":
        return "curl:%s:%s" % (p["side"], p["direction"])
    if t == "digit":
        return "digit:%s:%s:%s" % (p["side"], p["digit"], p["direction"])
    if t == "wrist":
        return "wrist:%s:%s:%s" % (p["side"], p["kind"], "+" if p["change_deg"] > 0 else "-")
    if t == "hand_view":
        return "hand_view:%s%s" % (p["side"], ":fovea" if p["in_fovea"] else "")
    if t == "fixation":
        return "look:%s" % p["target_class"]
    if t == "eye_move":
        return "eye_move:%s" % ("world" if (p["world_saccade"] or p["world_reflex"]) else "learner")
    if t == "head_turn":
        return "head_turn"
    if t == "object_motion":
        return "object_moves:%s" % p["object"]
    if t in ("lift", "drop"):
        return "%s:%s" % (t, p["object"])
    if t == "support_shift":
        return "support_shift"
    if t == "posture":
        return "posture:%s>%s" % (p["frm"], p["to"])
    if t == "transient":
        return "transient:%s" % p["kind"]
    if t == "approach":
        return "approach:%s:%s" % (p["side"], p["object"])
    if t == "caregiver":
        return "caregiver:%s" % p["kind"]
    return t


BACKGROUND_CLASSES = ("floor", "wall", "art", "ceiling", "static")
APPROACH_SALIENT_CLOSED_M = 0.05     # a hand closing 5 cm on an object...
APPROACH_SALIENT_END_M = 0.20        # ...and ending within 20 cm of it


def salient(p: dict) -> bool:
    t = p["type"]
    if t == "contact":
        if p.get("open_start", False):
            return False
        return p["a_class"] not in BACKGROUND_CLASSES and p["b_class"] not in BACKGROUND_CLASSES
    if t == "fixation":
        return p["duration_s"] >= 0.1 and p["target_class"] not in BACKGROUND_CLASSES
    if t == "eye_move":
        return False                 # the gaze's acts are its fixations; saccades come every ~0.3 s
    if t == "approach":
        # a moving hand gets closer to several of the six toys at once: only an approach that
        # ends near its object is an act toward it
        return p["closed_m"] >= APPROACH_SALIENT_CLOSED_M and p["end_m"] <= APPROACH_SALIENT_END_M
    return True


# ====================================================================== effectors
def part_effector(name: str) -> Optional[str]:
    """The effector of one infant part, by its anatomical group or a detection's part name:
    'finger_L_index', 'hand_R_palm', 'arm_L_upper', 'hand_L' -> the hand; 'foot_R', 'leg_R_lower'
    -> the leg; 'head', 'eye', 'eye_L', 'face', 'mouth' -> gaze; 'trunk', 'root' -> trunk."""
    n = str(name or "")
    for s in ("L", "R"):
        if n.startswith(("hand_" + s, "finger_" + s, "arm_" + s)):
            return "hand_" + s
        if n.startswith(("leg_" + s, "foot_" + s)):
            return "leg_" + s
    if n.startswith(("head", "eye", "face", "mouth")):
        return "gaze"
    if n.startswith(("trunk", "root")):
        return "trunk"
    return None


def _entity_effector(tr: Track, e: int) -> Optional[str]:
    e = int(e)
    if not 0 <= e < tr.ent.n or int(tr.ent.kind[e]) != KIND_SELF:
        return None
    return part_effector(tr.ent.group[e])


def _touchers(tr: Track) -> Dict[int, tuple]:
    """Per touched entity that is not the infant: (the record of every contact row joining it
    with an infant part, that part's effector), for 'which effector touched this object just
    before it moved'."""
    if not len(tr.c_row):
        return {}
    eff_of = np.asarray([_entity_effector(tr, e) or "" for e in range(tr.ent.n)], dtype=object)
    a = tr.c_e[:, 0].astype(np.int64)
    b = tr.c_e[:, 1].astype(np.int64)
    self_a, self_b = tr.ent.is_self[a], tr.ent.is_self[b]
    obj = np.r_[b[self_a & ~self_b], a[self_b & ~self_a]]
    part = np.r_[a[self_a & ~self_b], b[self_b & ~self_a]]
    rows = np.r_[tr.c_row[self_a & ~self_b], tr.c_row[self_b & ~self_a]].astype(np.int64)
    effs = eff_of[part] if part.size else np.zeros(0, dtype=object)
    keep = effs != ""
    obj, rows, effs = obj[keep], rows[keep], effs[keep]
    return {int(o): (rows[obj == o], effs[obj == o]) for o in np.unique(obj)}


def effectors_of(tr: Track, p: dict, touch: Optional[Dict[int, tuple]] = None) -> Set[str]:
    """The effectors a primitive belongs to (the module doc's rules); empty for the world's own
    events: an object that moved with no part on it, and the caregiver's arrivals, departures
    and touches (the touched part's own contact with the adult is that part's primitive)."""
    t = p["type"]
    if t == "contact":
        return {x for x in (_entity_effector(tr, p["a"]), _entity_effector(tr, p["b"])) if x}
    if t in ("curl", "digit", "wrist", "approach", "hand_view"):
        return {"hand_" + p["side"]}
    if t in ("lift", "drop"):
        return {"hand_" + s for s in str(p.get("hands") or "").split("+") if s in ("L", "R")}
    if t in ("fixation", "eye_move", "head_turn"):
        return {"gaze"}
    if t in ("posture", "support_shift"):
        return {"trunk"}
    if t == "transient":
        return {"gaze"} if p.get("kind") == "head_spin" else {"trunk"}
    if t == "object_motion":
        if touch is None:
            touch = _touchers(tr)
        ent = int(tr.ent.objects[int(p["k"])]["entity"])
        if ent not in touch:
            return set()
        rows, effs = touch[ent]
        w = int(round(TOUCH_BEFORE_S / STEP_S))
        sel = (rows >= p["i0"] - w) & (rows <= p["i0"])
        return set(effs[sel].tolist())
    return set()                     # the caregiver's events: the world's, not the infant's


# ====================================================================== segmentation
def _block_ends(blk: np.ndarray) -> np.ndarray:
    """[N] the last index of each index's run of equal block numbers."""
    n = len(blk)
    out = np.arange(n)
    for i in range(n - 2, -1, -1):
        if blk[i + 1] == blk[i]:
            out[i] = out[i + 1]
    return out


TURN_TYPES = ("digit", "wrist", "curl", "head_turn")


def head_turns_from_rest(tr: Track, gap: int) -> Set[int]:
    """The first record of every head turn that STARTS FROM REST: no head turn in the `gap`
    records before it. The head turns are found again on the whole track (the digester hands
    segment() only the primitives after its finalized line), so a turn right after that line is
    judged against the turns before it, as in any other window. (Head turns leave gaps where the
    head is still. The hand's turns: hand_turns_from_rest.)"""
    from . import primitives as P
    out: Set[int] = set()
    last_end = None
    for p in sorted(P.head_turns(tr), key=lambda p: (p["i0"], p["i1"])):
        quiet = last_end is None or p["i0"] - last_end > gap
        # a turn at the track's first records has no known past: it continues, not starts
        if quiet and p["i0"] > gap and bool(np.all(tr.continuous[p["i0"] - gap + 1:p["i0"] + 1])):
            out.add(int(p["i0"]))
        last_end = p["i1"] if last_end is None else max(last_end, p["i1"])
    return out


HAND_TURNS = ("digit", "curl", "wrist")


def hand_turn_key(p: dict) -> tuple:
    """A hand turn's series (one digit, the hand's closing, one wrist axis) and its first record."""
    return (p["type"], p["side"], p.get("digit") or p.get("kind") or "", int(p["i0"]))


def hand_turns_from_rest(tr: Track, gap: int) -> Set[tuple]:
    """The hand turns (a digit, the hand's closing, a wrist axis) that START FROM REST - no turn of
    the SAME joint series in the `gap` records before, in an unbroken record - and rise within the
    act timescale (primitives.ACT_MAX_RISE_S): act onsets of the hand, as a head turn from rest is
    one of the gaze (detector v9; OBSERVER_DESIGN.md section 9, "a hand moving only its fingers opens
    no episode"). Keys as hand_turn_key. Found again on the whole track, as head turns are.

    Why per joint series, measured 2026-09-25 with v9's turn starts (a turn starts where its movement
    starts, primitives._run_start) on production segment 39, a segment-40 run and random-order babble:
    some turn of the hand is in progress 99.8% of the time (the digits' turns overlap one another), and
    a turn after 0.5 s of the whole hand's rest came 1-3 times in about 9,000 turns per run; one digit
    is inside its own turns 60-61% of the time, and 29-36% of its turns start after 0.5 s of its own
    rest, 97% of those rising within 2 s: about 50 onsets per hand per minute in production and in
    babble alike. A limp body's 26 digit turns from rest all rose over more than 2 s: drifts, not
    onsets. Before v9 every turn began where the previous one ended (0.0-0.1% from rest: v8's turns
    tiled the record), so none could mark an act's start."""
    from . import primitives as P
    out: Set[tuple] = set()
    series: Dict[tuple, List[dict]] = {}
    for p in P.curls(tr):
        series.setdefault(hand_turn_key(p)[:3], []).append(p)
    for key, turns in series.items():
        side = "LR".index(key[1])
        if key[0] == "digit":
            x = tr.curl[:, side, ("index", "middle", "ring", "little", "thumb").index(key[2])]
        elif key[0] == "curl":
            x = tr.closure[:, side]
        else:
            x = tr.wrist[:, side, ("rotation", "flexion", "deviation").index(key[2])]
        last_end = None
        for p in sorted(turns, key=lambda p: (p["i0"], p["i1"])):
            quiet = last_end is None or p["i0"] - last_end > gap
            if (quiet and p["i0"] > gap and bool(np.all(tr.continuous[p["i0"] - gap + 1:p["i0"] + 1]))
                    and P.rise_s(tr, x, p["i0"], p["i1"], p["change_deg"]) <= P.ACT_MAX_RISE_S):
                out.add(hand_turn_key(p))
            last_end = p["i1"] if last_end is None else max(last_end, p["i1"])
    return out


def rides(tr: Track, p: dict, eff: str, head_rest: Optional[Set[int]] = None,
          hand_rest: Optional[Set[tuple]] = None) -> bool:
    """A primitive that RIDES in its effector's episodes: a token of the episode its start falls
    in, but it opens none and holds none open.

    Turns in progress cannot mark when an act begins or ends: until v9 finger, wrist and closing
    turns tiled the record (measured 2026-09-24, 4,800 body-steps each of production segment 39 and
    random-order babble: a hand inside a finger turn 99-100% of the steps, a wrist turn 97-100%, a
    closing or opening turn 87-99%; the head inside a head turn 80% in production, 98% in babble), and
    with every turn opening episodes, each hand's episodes covered 99-100% of the steps and ran to the
    block's end. A head turn that STARTS FROM REST (head_turns_from_rest: 5% of production's head
    turns, 1% of babble's) is an act onset: it opens a gaze episode, holding it open only to its own
    start. Since v9 so is a hand turn from its own joint's rest (hand_turns_from_rest), for the
    hand's episodes. Two parts of ONE HAND touching ride too: 73% (production) and 74% (babble) of a
    hand's touches are the hand meeting itself as it closes."""
    if p["type"] == "head_turn":
        return head_rest is None or int(p["i0"]) not in head_rest
    if p["type"] in HAND_TURNS:
        return hand_rest is None or hand_turn_key(p) not in hand_rest
    if p["type"] in TURN_TYPES:
        return True
    if p["type"] == "contact" and eff in ("hand_L", "hand_R"):
        return _entity_effector(tr, p["a"]) == eff and _entity_effector(tr, p["b"]) == eff
    return False


def segment(tr: Track, prims: List[dict], run: str, body: int) -> List[dict]:
    gap = int(round(GAP_S / STEP_S))
    block = int(round(MAX_S / STEP_S))
    pair_w = int(round(PAIR_S / STEP_S))
    blk = np.asarray(tr.step, dtype=np.int64) // block
    blk_end = _block_ends(blk)
    touch = _touchers(tr)
    head_rest = head_turns_from_rest(tr, gap) if any(p["type"] == "head_turn" for p in prims) else set()
    hand_rest = hand_turns_from_rest(tr, gap) if any(p["type"] in HAND_TURNS for p in prims) else set()
    openers: Dict[str, List[dict]] = {e: [] for e in EFFECTORS}
    riders: Dict[str, List[dict]] = {e: [] for e in EFFECTORS}
    for p in prims:
        if not salient(p):
            continue
        for eff in effectors_of(tr, p, touch):
            (riders if rides(tr, p, eff, head_rest, hand_rest) else openers)[eff].append(p)
    order = lambda p: (p["i0"], p["i1"], p["type"])
    out: List[dict] = []
    for eff in EFFECTORS:
        eps: List[dict] = []
        cur = None
        for p in sorted(openers[eff], key=order):
            if cur is not None:
                broken = p["i0"] > cur["i1"] and not bool(np.all(tr.continuous[cur["i1"] + 1:p["i0"] + 1]))
                if p["i0"] > cur["i1"] + gap or broken or blk[p["i0"]] != cur["blk"]:
                    eps.append(cur)
                    cur = None
            if cur is None:
                cur = dict(effector=eff, body=int(body), pid=int(tr.pid), i0=int(p["i0"]), i1=int(p["i0"]),
                           blk=int(blk[p["i0"]]), opening=token(p), _prims=[],
                           after_cap=bool(int(tr.step[p["i0"]]) % block < gap))
            cur["_prims"].append(p)
            # a turn from rest opens, but its own extent (a slow drift can last seconds) holds
            # nothing open
            end = p["i0"] if p["type"] in TURN_TYPES else p["i1"]
            cur["i1"] = int(max(cur["i1"], min(end, blk_end[cur["i0"]])))
        if cur is not None:
            eps.append(cur)
        # riders join the episode their start falls in
        starts = [e["i0"] for e in eps]
        for p in riders[eff]:
            k = bisect.bisect_right(starts, p["i0"]) - 1
            if k >= 0 and p["i0"] <= eps[k]["i1"]:
                eps[k]["_prims"].append(p)
        for c in eps:
            c["tokens"] = [token(p) for p in sorted(c.pop("_prims"), key=order)]
            c.pop("blk", None)
            _finish(tr, c, run, body)
            out.append(c)
    out.extend(pairs(tr, out, run, body, pair_w))
    out.sort(key=lambda e: (-e["i0"], e["id"]))
    return out


def _finish(tr: Track, c: dict, run: str, body: int) -> None:
    """The fields every unit carries (effector episodes and pairs)."""
    i0, i1 = c["i0"], c["i1"]
    authors = tr.author[i0:i1 + 1]
    eff = c["effector"]
    held = np.asarray(tr.toy_held[i0:i1 + 1])
    # the scripted toy's hand-off moves the toy with a hand: that hand's acts are the script's
    if eff == "hand_L":
        toy = bool(np.isin(held, (1, 3)).any())
    elif eff == "hand_R":
        toy = bool(np.isin(held, (2, 3)).any())
    else:
        toy = bool(c.get("toy_held", False))
    tag = eff if eff != PAIR else "%s>%s:s%d" % (c["lead"], c["follow"], int(tr.step[c["follow_i0"]]))
    c.update(step0=int(tr.step[i0]), step1=int(tr.step[i1]), t0=float(tr.sim_time[i0]),
             duration_s=float(tr.sim_time[i1] - tr.sim_time[i0]),
             id="%s:b%d:p%d:s%d:%s" % (run, body, tr.pid, int(tr.step[i0]), tag),
             authors={int(a): int(n) for a, n in zip(*np.unique(authors, return_counts=True))},
             caregiver_present=bool(tr.caregiver_present[i0:i1 + 1].any()),
             caregiver_touch=bool(tr.caregiver_contact[i0:i1 + 1].any()),
             live=bool(tr.live[i0]), world_episode=int(tr.episode[i0]), toy_held=toy)


def pairs(tr: Track, eps: List[dict], run: str, body: int, pair_w: int) -> List[dict]:
    """Cross-effector PAIR units: two fresh episodes of different effectors whose starts fall
    within pair_w records, in one continuous stretch (module doc)."""
    rank = {e: k for k, e in enumerate(EFFECTORS)}
    fresh = sorted((e for e in eps if e.get("effector") in rank and not e.get("after_cap") and e.get("opening")),
                   key=lambda e: (e["i0"], rank[e["effector"]]))
    out = []
    seen = set()
    for a_i, a in enumerate(fresh):
        for b in fresh[a_i + 1:]:
            if b["i0"] - a["i0"] > pair_w:
                break
            if b["effector"] == a["effector"]:
                continue
            if not bool(np.all(tr.continuous[a["i0"] + 1:b["i0"] + 1])):
                continue
            if a["opening"] == b["opening"]:
                continue                     # one act of two effectors (a foot touching a foot)
            key = (a["i0"], b["i0"], a["opening"], b["opening"])
            if key in seen:
                continue                     # that act opened both legs: one pattern, counted once
            seen.add(key)
            c = dict(effector=PAIR, lead=a["effector"], follow=b["effector"], body=int(body), pid=int(tr.pid),
                     i0=int(a["i0"]), i1=int(max(a["i1"], b["i1"])), follow_i0=int(b["i0"]),
                     tokens=[a["opening"], b["opening"]], members=[a["id"], b["id"]],
                     lag_s=float(tr.sim_time[b["i0"]] - tr.sim_time[a["i0"]]),
                     after_cap=False, toy_held=bool(a.get("toy_held") or b.get("toy_held")))
            _finish(tr, c, run, body)
            c.pop("follow_i0", None)
            out.append(c)
    return out


def attach(episodes: List[dict], detections: List[dict]) -> None:
    """Each detection names its episode, and each episode lists the capabilities detected in it.

    A detection belongs to the effectors of its parts (every effector when it names none, as the
    gaze and several object rows do); each of those effectors' episodes that holds the
    detection's first or last step lists its capability. The detection's `episode` (its replay)
    is the first of them in EFFECTORS order, or None. A pair unit lists its two episodes'
    capabilities."""
    for e in episodes:
        e.setdefault("caps", [])
    by_eff: Dict[str, List[dict]] = {}
    for e in episodes:
        if e.get("effector") in EFFECTORS:
            by_eff.setdefault(e["effector"], []).append(e)
    starts = {}
    for eff, es in by_eff.items():
        es.sort(key=lambda e: e["i0"])               # one effector's episodes never overlap
        starts[eff] = [e["i0"] for e in es]

    def holding(eff, i):
        es = by_eff[eff]
        k = bisect.bisect_right(starts[eff], i) - 1
        return es[k] if k >= 0 and es[k]["i0"] <= i <= es[k]["i1"] else None

    for d in detections:
        want = {x for x in (part_effector(p) for p in (d.get("parts") or [])) if x} or set(EFFECTORS)
        host = None
        for eff in EFFECTORS:
            if eff not in want or eff not in by_eff:
                continue
            first, last = holding(eff, d["i0"]), holding(eff, d["i1"])
            for e in (first, last):
                if e is not None and d["cap"] not in e["caps"]:
                    e["caps"].append(d["cap"])
            if host is None:
                host = first or last
        d["episode"] = host["id"] if host else None
    by_id = {e["id"]: e for e in episodes if e.get("effector") != PAIR}
    for e in episodes:
        if e.get("effector") == PAIR:
            caps = []
            for m in e.get("members") or []:
                for c in (by_id.get(m) or {}).get("caps", []):
                    if c not in caps:
                        caps.append(c)
            e["caps"] = caps
