# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Layer 4, the capability detectors: the addendum's 53 (section 26) plus the finger rows.

Every detector returns MEASUREMENTS, not only a label (addendum section 22): the steps it spans,
the parts and objects involved, a confidence (how far past its defining thresholds the event
went: 0.5 at a threshold, 1.0 at twice the margin), continuous measures a progress plot can use
before a milestone is reached, and a repetition key (what counts as "the same thing again").
Prediction and intention evidence are attached afterwards by evidence.py; provenance (who wrote
the order in force, whether the caregiver or a script was involved) is attached here, because a
probe-forced or caregiver-carried act must never be read as the infant's own.

Thresholds are the instrument's definitions, validated by the scripted worlds (validate.py) and
given chance rates by babble-only physics (baseline.py). They never reach the learner.

NOT OBSERVABLE IN THIS WORLD (the crib, 2026-09-24): P11 climb onto a low platform and P12 back
down legs first (no platform or edge), O09 put in a container and O10 take out (no container).
Their detectors exist and return nothing; catalog.py states why, so an empty row is never read
as "not yet developed". The play-area zones that would expose them are a scene option
(OBSERVER_DESIGN.md, section 29 decision).

The toy is a mocap body: contact cannot move it, and with --toy-hand-off follow it FOLLOWS a hand
that two of its parts have touched for 0.1 s (caregiver.py). A toy hold or carry is therefore the
script's; detectors mark it `scripted`, and the tracker never counts a scripted act.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .features import (KIND_ART, KIND_CAREGIVER, KIND_CEILING, KIND_FLOOR, KIND_OBJECT, KIND_SELF,
                       KIND_STATIC, KIND_WALL, Track)
from .kinematics import DIGITS, FINGERS, SIDES
from . import primitives as P

STEP_S = 0.05

# ---- hand and fingers
H01_WINDOW_S = (0.05, 0.30)      # "finger flexion increases within 50-300 ms" of the touch
H01_MIN_CLOSE_DEG = 10.0
H01_PRE_CLOSE_DEG = 5.0          # already closing this much in the 100 ms before the touch: not "after"
HOLD_MIN_S = 0.20                # a hold lasts at least 200 ms of continuous contact
HOLD_MIN_CLOSURE_DEG = 15.0
HOLD_MAX_DRIFT_M = 0.02          # object-to-palm distance stays within 2 cm of its start
HOLD_MOVED_M = 0.01              # ...while the object moved with the palm at least 1 cm OFF ITS SUPPORT
                                 # (v8; or was opposed)
HOLD_CLOSE_ON_DEG = 10.0         # ...a grip (_grip) whose hand closed at least this much on the object
HOLD_FOREARM_HOLDS = True        # support by CONTACT (_support), used only where the physics cannot be
                                 # read (a track without recorded normals, an object of unknown mass, the
                                 # world's mocap toys): the hands and the forearms are the holding limbs,
                                 # the rest of the body is support. Where the physics can be read, support
                                 # is the upward force everything but the gripping hand puts on the object
                                 # (_grip), and a forearm is support exactly as far as it carries the object
                                 # (2026-09-25 evening: the forearm rule had been chosen on scripted boxes
                                 # sitting 4-5.5 mm inside the forearm, which no physical grasp makes)
RELEASE_OPEN_DEG = 10.0          # the hand opens this much in the 300 ms before letting go
ESCAPE_MPS = 0.10                # an object leaving the palm faster than this escaped
CYCLE_MIN_DEG = 15.0
CYCLE_MAX_S = 3.0
ISOLATION_OTHERS_MAX_DEG = 5.0
# THE ACT TIMESCALE (the catalog's 250 ms - 2 s for H05, H07, H08, HX1, HX2): the act's movement takes at
# most this long (until the review of 2026-09-25 the turn's 10-90% rise, _rise_s; since, the turn's fastest
# stretch moving the row's full line, _line_s: WHERE THE TIMESCALE IS MEASURED, below). HX1 since v8 (limp hands' 6-70 s drifts made 40 acts); H05,
# H07, H08 and HX2 since v9. Measured 2026-09-25 with v9's turn starts (agents/
# observer_detectors_2026-09-25.md): every limp H05 act rose over 8.5-16.5 s (the fingers settling
# after a reset or a switch to zero commands), a limp HX2 act over 102.8 s; babble's acts rise
# fast (random orders: H05 median 0.35 s, 1.6% over 2 s; H07 2.4%; H08 0%; HX2 0.1%) and so do
# production's (H05 median 0.35 s, 2.7% over 2 s; H07 11.7%; H08 2.7%; HX2 4.6%).
ACT_MAX_RISE_S = P.ACT_MAX_RISE_S
HX1_MAX_RISE_S = ACT_MAX_RISE_S
# WHERE THE TIMESCALE IS MEASURED (detector v9; the independent review of 2026-09-25, "the act timescale is
# measured over the whole run"). A turn starts where its monotone approach starts (primitives._run_start) and
# ends at its extreme, so its 10-90% rise counted a slow lead-in or tail beyond 10% of the change into the
# act's time: the rule removed 3-5 in 100 of production's spreading and palm-shaping acts (HX1, HX2) against
# at most 1 in 1,000 of babble's, and 3-6 in 10 of those removals held a movement of the full line within
# 2 s (segment 39, body 0: a 23-degree finger straightening with a 2.55 s rise, inside it the finger moved
# its full 15 degrees in 0.60 s). The same whole-turn reading let a flick UNDER the line count when a slow
# creep afterwards carried the turn past it (production, segment 40: a 15.2-degree spread whose 10-90% rise
# took 0.35 s, the last degrees creeping for 3 s). The act is a movement of the row's full line within the
# catalog's 2 s: ACT_LINE_SHARE of the row's line, over the turn's fastest stretch (primitives.line_s,
# measure `line_s`). `rise_s`, the whole turn's 10-90% rise, is still reported.
ACT_LINE_SHARE = 1.0
ISOLATION_MIN_RATIO = 3.0
OPPOSE_NEAR_M = 0.015
OPPOSE_APPROACH_M = 0.01
OPPOSE_SWIVEL_DEG = 10.0
WRIST_FOREARM_MAX_DEG = 10.0
WRIST_ROT_MAX_TRANSLATION_M = 0.02
SLIP_MPS = 0.03
SLIP_STOP_MPS = 0.01
GRIP_RISE = 0.30                 # force up 30 % within 300 ms of the slip
# ---- self and body
MIDLINE_M = 0.06
MIDLINE_TOLERANCE_M = 0.02
B02_CONVERGE_M = 0.01            # the two fingertips closed at least 1 cm in the 250 ms before
CONVERGE_M = 0.02
FACE_APPROACH_S = 0.25
FACE_APPROACH_MPS = 0.05         # the palm's mean speed over those 250 ms
MOUTH_MIN_S = 0.10
VIEW_ARM_SPEED_MPS = 0.10
VIEW_HEAD_MAX_DEG = 10.0
# ---- gaze
G01_HEAD_MAX_DEG = 2.0
G02_MIN_S = 0.15
G03_AWAY_S = (0.3, 5.0)
G03_CUE_MPS = 0.05
G04_MOVING_MPS = 0.05
G04_MIN_S = 0.5
G04_ON_FRACTION = 0.8
G04_GAZE_TURN_DEG = 5.0          # the gaze itself turned at least this much while it followed
G05_HEAD_DEG = 5.0
G05_ECC_DROP_DEG = 3.0
# ---- reach and grasp
REACH_MAX_START_M = 0.45
REACH_MIN_CLOSED_M = 0.05
REACH_MIN_FRACTION = 0.30
REACH_OWN_SHARE = 0.5            # of the closing the hand's own movement must account for
GAZE_LEAD_S = 1.0
OPPOSITION_MIN_DEG = 120.0         # v9: one exclusion inside the grip (_grip): two surfaces of one hand
                                  # whose pushes on the object are less than this far apart do not oppose
                                  # (contact normals; cos = -0.5: half of one push met head-on). Adjacent
                                  # faces of a box read 90 (edge contacts up to 107.6) in the kept grasp
                                  # cases, opposite faces about 180. Babble made 123 thumb co-contact
                                  # records in 120,000 body-steps; the rebuild (normals from the stored
                                  # states) could judge 35 of them (28%): 0-100 degrees, none at 100 or more;
                                  # an independent review placed every one of the 91 on a box or the ball
                                  # on the object's faces: same face or adjacent faces, none opposite
                                  # (2026-09-25; the first v2 recordings measure every row). v8's 90 was a
                                  # POSITION angle at the object's centre, which ran through most grasps
                                  # this hand can make on the 6 cm boxes (83-126 degrees) and missed a real
                                  # pinch of the toy's pad (30-44 degrees)
GRAVITY_MPS2 = 9.81
GRIP_OTHERS_MAX_WEIGHT = 0.5      # v9 grip: everything but the gripping hand (the floor, other objects, the
                                  # chest and the rest of the body, both forearms, the other hand, the
                                  # caregiver) pushes the object UP with less than half its weight: its
                                  # support does not come mainly from them (the operator's review, adopted
                                  # by the coordinator on the evening of 2026-09-25)
GRIP_SLIP_MPS = SLIP_MPS          # ...and it moves with the palm: the object-to-palm distance changes at most
                                  # this fast over the record's own burst (the slip line of H09 and R07; _slip_rate)
OPPOSING_SURFACES = (("thumb", "fingers"), ("thumb", "palm"), ("fingers", "palm"))
DEFAULT_FRICTION = 1.0            # MuJoCo's default sliding friction, for a track read without its model
CARRY_MIN_M = 0.05
PLACE_MAX_MPS = 0.03
PLACE_STABLE_S = 0.5
PLACE_STABLE_MPS = 0.05
# ---- posture and locomotion
P01_WINDOW_S = 2.0
P01_MOVING_PALM_MPS = 0.10
P01_RATIO = 0.5
PUSH_FORCE_FRACTION = 0.30
PUSH_EXTEND_DEG = 10.0
PUSH_DISPLACE_M = 0.01
# A PUSH IS A SUSTAINED FORCE (detector v9, 2026-09-25; agents/observer_detectors_2026-09-25.md, P05).
# The floor force a limb carries flickers from record to record (a planted foot's contact alternates
# between 0 and 37 N from one 10 ms frame to the next, and the recorder keeps each record's largest
# frame), and until v9 P05 compared the two single records at its window's ends: 90-98 of every 100 P05
# acts in babble, and 91-94 in production, rested on one record. Measured on the kept recordings
# (scratchpad p05_runs.py: every record whose rise over the 0.5 s window meets the line, grouped into
# runs of consecutive such records), a rise WITHOUT a push - the limb extending under 5 degrees - lasts
# one record in 85-90 of 100 runs and three or more in 3-4 of 100 (production segments 39 and 40,
# saved-world babble; the reset worlds' bodies dropping onto the floor at the start add longer ones).
# So the rise must hold over PUSH_HOLD_RECORDS consecutive records at each end: the force of every one
# of the window's last 3 records stands the line above the largest of the 3 records ending at its start
# (the flicker runs both ways: a dropout at the start inflates a rise as a spike at the end does). Three
# records put force in three bursts in a row, at least 70 ms of it.
PUSH_HOLD_RECORDS = 3
WEIGHT_SHARE_CHANGE = 0.20
TRUNK_LIFT_M = 0.03
TRUNK_LIFT_MIN_S = 0.25
LIMB_SUPPORT_SHARE = 0.5
SIT_MIN_S = 1.0
TRANSLATE_M = 0.10
TRANSLATE_MAX_S = 10.0
TRANSLATE_STRAIGHT = 0.6
CRAWL_MIN_CYCLES = 2
CRAWL_MIN_M = 0.05
# ---- object play
PUSH_DIR_MAX_DEG = 60.0
PULL_MIN_M = 0.02
BANG_MIN_IMPACTS = 2
BANG_WINDOW_S = 2.0
BANG_MIN_FORCE_N = 1.0
SHAKE_REVERSALS = 2
SHAKE_WINDOW_S = 1.0
SHAKE_AMP_M = 0.01
SPIN_RADPS = 1.0
SPIN_MIN_S = 0.3
TRANSFER_MIN_S = 0.2
TWO_HAND_MIN_S = 0.3
STILL_HAND_MPS = 0.03
ACTIVE_HAND_MPS = 0.05
STACK_STABLE_S = 1.0

# Acts made of the body touching an object. When the object is one the world moves (the toy the
# caregiver's script carries, the flying critters) and it was moving as the act began, the WORLD
# may have made the contact: such an act is marked world_moved and never counted (tracker.py),
# like one the caregiver's own body touched. Before 2026-09-24 a toy carried into a limp hand
# counted as the infant's "repeat contact": the limp baseline made 13 per 1,000 steps.
OBJECT_CONTACT_CAPS = {"H01", "H02", "H03", "H09", "R02", "R04", "R05", "R06", "R07", "R08", "HX3",
                       "O01", "O02", "O03", "O04", "O05", "O06", "O07", "O08", "O11"}

# The recorder's v2 field groups (observer_record.FIELDS) each v9 definition reads. A recording
# without them (observer_world_v1) measures these detectors only in part (H02 by the carried branch)
# or not at all (R04), so a babble summary made from one names them unmeasured (baseline.summarize)
# instead of counting a partial rate as chance.
# (O05, shaking, walks the hand-object contact intervals _holds lists but never reads `held`: it does
# not depend on opposition or support, so it is measured on v1 recordings as before.)
NEEDS_FIELDS = {cap: ("contact_normals",) for cap in ("R04", "H02", "H03", "H09", "R06", "R07")}

AUTHOR_NAMES = {0: "unknown", 1: "planner", 2: "reflex", 3: "reflex_noise", 4: "self_choice",
                5: "acting_snapshot", 6: "external_rest", 7: "probe", 8: "scripted",
                9: "random_actions", 10: "limp", 11: "yoked"}
MODEL_SELECTED = {"planner", "self_choice", "acting_snapshot"}
SPONTANEOUS = {"reflex", "reflex_noise"}


def provenance_class(author: str) -> str:
    """The addendum's provenance categories (section 36)."""
    if author in MODEL_SELECTED:
        return "planner-selected"
    if author in SPONTANEOUS:
        return "spontaneous"
    if author == "external_rest":
        return "external-rest"
    if author in ("probe", "scripted"):
        return "probe-forced"
    if author in ("random_actions", "limp", "yoked"):
        return "baseline"
    return "unknown"


def _conf(*margins: float) -> float:
    """0.5 at a threshold, 1.0 at twice the threshold's margin; the weakest criterion decides."""
    if not margins:
        return 1.0
    return float(min(max(0.0, min(1.0, 0.5 + 0.5 * m)) for m in margins))


def _over(value: float, threshold: float) -> float:
    return (value - threshold) / abs(threshold) if threshold else 1.0


def _under(value: float, threshold: float) -> float:
    return (threshold - value) / abs(threshold) if threshold else 1.0


class Context:
    def __init__(self, tr: Track, prims: List[dict]):
        self.tr = tr
        self.prims = prims
        self.by_type: Dict[str, List[dict]] = defaultdict(list)
        for p in prims:
            self.by_type[p["type"]].append(p)
        self.sup = P.support(tr)
        self.posture = P.posture_state(tr)
        self._pairs: Dict[tuple, dict] = {}
        self.others_than = {}

    def pair(self, key, mask_a, mask_b) -> dict:
        if key not in self._pairs:
            self._pairs[key] = self.tr.pair_series(mask_a, mask_b)
        return self._pairs[key]

    def hand_obj(self, side: str, k: int) -> dict:
        tr = self.tr
        return self.pair(("hand", side, k), tr.ent.hand[side], tr.object_mask(k))

    def part_obj(self, name: str, mask: np.ndarray, k: int) -> dict:
        return self.pair(("part", name, k), mask, self.tr.object_mask(k))

    def obj_floor(self, k: int) -> dict:
        return self.pair(("floor", k), self.tr.object_mask(k), self.tr.ent.floor)

    def obj_other_objects(self, k: int) -> dict:
        tr = self.tr
        m = np.zeros(tr.ent.n, dtype=bool)
        for kk, o in enumerate(tr.ent.objects):
            if kk != k and o["kind"] not in ("caregiver",):
                m[int(o["entity"])] = True
        return self.pair(("objobj", k), tr.object_mask(k), m)

    def det(self, cap: str, i0: int, i1: int, *, sub: str = "", key: str = "", measures=None,
            confidence: float = 1.0, objects=(), parts=(), scripted: bool = False, **extra) -> dict:
        tr = self.tr
        i0, i1 = int(max(0, i0)), int(min(tr.n - 1, i1))
        author = AUTHOR_NAMES.get(int(tr.author[i0]), "unknown")
        authors = sorted({AUTHOR_NAMES.get(int(a), "unknown") for a in tr.author[i0:i1 + 1]})
        cg_touch = bool(tr.caregiver_contact[max(0, i0 - 2):i1 + 1].any())
        d = dict(cap=cap, sub=sub, key=key or sub or cap, i0=i0, i1=i1, step0=int(tr.step[i0]),
                 step1=int(tr.step[i1]), t0=float(tr.sim_time[i0]),
                 duration_s=float(tr.sim_time[i1] - tr.sim_time[i0]),
                 measures={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                           for k, v in (measures or {}).items()},
                 confidence=float(confidence), objects=list(objects), parts=list(parts),
                 author=author, authors=authors, provenance=provenance_class(author),
                 order_seq=int(tr.order_seq[i0]), latched_at_start=bool(tr.latched[i0]),
                 caregiver_present=bool(tr.caregiver_present[i0:i1 + 1].any()),
                 caregiver_touch=cg_touch, rest=bool(tr.rest[i0:i1 + 1].any()),
                 scripted=bool(scripted), posture=str(self.posture[i0]), live=bool(tr.live[i0]))
        if cg_touch and d["provenance"] != "probe-forced":
            d["provenance_note"] = "caregiver touched the infant during the act"
        d["world_moved"] = bool(cap in OBJECT_CONTACT_CAPS and self.world_moved(objects, i0, i1))
        if d["world_moved"]:
            d["provenance_note"] = "an object the world moves (the caregiver's toy, a critter) was moving into the act"
        d.update(extra)
        return d

    def world_moved(self, names, i0: int, i1: int) -> bool:
        """One of the named objects is moved by the world (not a free body) and moved at least
        OBJ_MOVE_MPS in the two steps before the act or during it."""
        tr = self.tr
        for k, o in enumerate(tr.ent.objects):
            if o["free"] or o["kind"] in ("caregiver", "decoration"):
                continue
            if o["name"] not in names and tr.ent.label(int(o["entity"])) not in names:
                continue
            speed = tr.object_body_speed(k) if o["kind"] == "mobile" else tr.obj_speed[:, k]
            if np.nan_to_num(speed[max(0, i0 - 2):i1 + 1]).max() >= P.OBJ_MOVE_MPS:
                return True
        return False


# ====================================================================== helpers
def _hand_onsets(ctx: Context, side: str, exclude_self_hand: bool = True) -> List[dict]:
    """Contact primitives that begin on a hand of this side against anything that is not that
    hand itself; the first step of each is the touch."""
    tr = ctx.tr
    hand = tr.ent.hand[side]
    out = []
    for c in ctx.by_type["contact"]:
        a, b = c["a"], c["b"]
        if c["open_start"]:
            continue
        on_hand = hand[a] or hand[b]
        if not on_hand:
            continue
        other = b if hand[a] else a
        if exclude_self_hand and hand[other]:
            continue
        out.append(dict(c, other=int(other)))
    return out


def _intervals(mask, cont):
    return P._intervals(np.asarray(mask, dtype=bool), cont)


def _rel_speed(tr: Track, side: int, k: int) -> np.ndarray:
    return np.linalg.norm(tr.obj_vel[:, k] - tr.palm_vel[:, side], axis=-1)


def _obj_palm_dist(tr: Track, side: int, k: int) -> np.ndarray:
    return np.linalg.norm(tr.obj_pos[:, k] - tr.palm[:, side], axis=-1)


def _object_ok(o: dict) -> bool:
    return o["kind"] not in ("caregiver", "decoration")


def _toy_scripted(tr: Track, k: int, side: int, i0: int, i1: int) -> bool:
    o = tr.ent.objects[k]
    return (not o["free"] and o["kind"] == "ball"
            and bool((tr.toy_held[i0:i1 + 1] == (1 if side == 0 else 2)).any()))


# ====================================================================== hand and fingers
def H01(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    w0, w1 = int(round(H01_WINDOW_S[0] / STEP_S)), int(round(H01_WINDOW_S[1] / STEP_S))
    for si, side in enumerate(SIDES):
        for c in _hand_onsets(ctx, side):
            i = c["i0"]
            if i < 2 or i + w0 >= tr.n:
                continue
            c0 = tr.closure[i, si]
            pre = c0 - tr.closure[i - 2, si]
            seg = tr.closure[i + w0:min(tr.n, i + w1 + 1), si]
            if not np.isfinite(seg).any() or not tr.continuous[i - 1:i + w1 + 1].all():
                continue
            rise = float(np.nanmax(seg) - c0)
            if rise < H01_MIN_CLOSE_DEG or pre >= H01_PRE_CLOSE_DEG:
                continue
            half = c0 + rise / 2.0
            j = i + int(np.argmax(tr.closure[i:i + w1 + 1, si] >= half))
            peak = i + w0 + int(np.nanargmax(seg))
            out.append(ctx.det("H01", i, peak, sub=side, key="%s:%s" % (side, c["a_class"] if tr.ent.hand[side][c["b"]] else c["b_class"]),
                               measures=dict(latency_ms=1000.0 * tr.seconds(i, j), closure_increase_deg=rise,
                                             touch_force_n=c["peak_force_n"], closing_before_touch_deg=pre),
                               confidence=_conf(_over(rise, H01_MIN_CLOSE_DEG), _under(pre, H01_PRE_CLOSE_DEG)),
                               objects=[tr.ent.label(c["other"])], parts=["hand_" + side],
                               touched_kind=P.entity_class(tr, c["other"])))
    return out


def _support(ctx: Context, k: int, side: str = "") -> np.ndarray:
    """Per record: object k is supported by something other than the hand holding it.

    CONTACT, everywhere: the floor, another object, or (v9) any part of the infant's own body except the
    two hands and, while HOLD_FOREARM_HOLDS, the two forearms. Before v9 the infant's body was not
    support: a box lying on the belly or a thigh, moved by the body under it while a hand touched it,
    read as carried.

    And where the physics can be read (_grip: recorded normals, a free object of known mass), FORCE as
    well: everything but that hand - the forearms and the other hand included - pushes the object up with
    at least GRIP_OTHERS_MAX_WEIGHT of its weight. A forearm is then support exactly as far as it carries
    the object, and a box squeezed between two palms, each pushing sideways, rests on neither. Force
    alone is not enough: a limp hand under a box that still touches the floor carries most of its
    weight, and read by force alone the arm sliding it along was a carry in all 8 reset-world limp bodies
    (measured 2026-09-25 night: limp H02 0 -> 8); by contact the floor still holds it."""
    tr = ctx.tr
    body = tr.ent.is_self & ~tr.ent.hand["L"] & ~tr.ent.hand["R"]
    if HOLD_FOREARM_HOLDS:
        body = body & ~np.asarray([g in ("arm_L_lower", "arm_R_lower") for g in tr.ent.group])
    contact = (ctx.obj_floor(k)["on"] | ctx.obj_other_objects(k)["on"]
               | ctx.pair(("objbody", k, bool(HOLD_FOREARM_HOLDS)), tr.object_mask(k), body)["on"])
    if side:
        g = _grip(ctx, side, k)
        if g["physics"]:
            return contact | g["supported"]
    return contact


def _pair_rows(tr: Track, mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    """The contact rows joining an entity of mask_a with one of mask_b: the record (`row`), the first
    and last physics frame of the record's burst in which the pair touched (`f0`, `f1`: c_frame; the
    whole burst on a made-up track without it), its mean normal force in newtons (`f`), and its normal
    turned to point from the mask_a part into the mask_b part (`n`; NaN without recorded normals)."""
    e = np.asarray(tr.c_e, dtype=np.int64).reshape(-1, 2)
    a = mask_a[e[:, 0]] & mask_b[e[:, 1]]
    b = mask_b[e[:, 0]] & mask_a[e[:, 1]]
    sel = np.nonzero(a | b)[0]
    fr = getattr(tr, "c_frame", None)
    if fr is not None and len(fr):
        fr = np.asarray(fr, dtype=np.int64).reshape(-1, 2)[sel]
        f0, f1 = fr[:, 0], fr[:, 1]
    else:
        f0, f1 = np.zeros(len(sel), dtype=np.int64), np.full(len(sel), 127, dtype=np.int64)
    nm = getattr(tr, "c_normal", None)
    if nm is None or not len(nm):
        n = np.full((len(sel), 3), np.nan)
    else:
        nm = np.asarray(nm, dtype=np.float64).reshape(-1, 3)[sel]
        n = np.where(a[sel][:, None], nm, -nm)
    return dict(row=np.asarray(tr.c_row, dtype=np.int64)[sel], f0=f0, f1=f1,
                f=np.asarray(tr.c_fmean, dtype=np.float64)[sel], n=n)


def _ranges(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """np.concatenate([np.arange(l, h) for l, h in zip(lo, hi)]) without the loop."""
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    return np.repeat(lo - first, cnt) + np.arange(tot)


def _pushes_together(ra: dict, rb: dict, n: int):
    """What two contact sets on one object did in the SAME frames. The recorder gathers a record's
    contacts over the five physics frames of its burst, so a thumb that touched in frames 0-1 and a
    finger in frames 3-4 never touched together (an independent review of 2026-09-25 found 30 of 147
    kept babble and limp co-contact records so). Only rows of each side whose frame range overlaps a row
    of the other side's count. Returns per record: together [N] (the two touched in overlapping frames),
    and each side's push over those rows - the force [N, 3] (mean normal force times normal, summed)
    and the direction [N, 3] (the same with every force at least 1e-6 N, as pair_series weights)."""
    together = np.zeros(n, dtype=bool)
    fa, fb, da, db = (np.zeros((n, 3)) for _ in range(4))
    if not len(ra["row"]) or not len(rb["row"]):
        return together, fa, fb, da, db
    oa, ob = np.argsort(ra["row"], kind="stable"), np.argsort(rb["row"], kind="stable")
    A = {key: v[oa] for key, v in ra.items()}
    B = {key: v[ob] for key, v in rb.items()}
    lo = np.searchsorted(B["row"], A["row"], "left")
    hi = np.searchsorted(B["row"], A["row"], "right")
    ai = np.repeat(np.arange(len(A["row"])), hi - lo)
    bi = _ranges(lo, hi)
    if not len(ai):
        return together, fa, fb, da, db
    ov = (A["f0"][ai] <= B["f1"][bi]) & (B["f0"][bi] <= A["f1"][ai])
    ai, bi = ai[ov], bi[ov]
    if not len(ai):
        return together, fa, fb, da, db
    together[A["row"][ai]] = True
    for S, idx, force, direction in ((A, np.unique(ai), fa, da), (B, np.unique(bi), fb, db)):
        idx = idx[np.isfinite(S["n"][idx]).all(axis=1)]
        np.add.at(force, S["row"][idx], S["f"][idx, None] * S["n"][idx])
        np.add.at(direction, S["row"][idx], np.maximum(S["f"][idx], 1e-6)[:, None] * S["n"][idx])
    return together, fa, fb, da, db


def _friction(tr: Track, side: str, k: int) -> float:
    """The largest sliding friction among this hand's and object k's geoms in the track's model (MuJoCo
    gives a contact the larger of its two geoms' friction); DEFAULT_FRICTION without a model."""
    m = getattr(getattr(tr, "kin", None), "m", None)
    if m is None:
        return DEFAULT_FRICTION
    try:
        ents = tr.header["entities"]
        hand = tr.ent.hand[side]
        bodies = {int(b) for e in ents if 0 <= int(e.get("code", -1)) < len(hand) and hand[int(e["code"])]
                  for b in (e.get("bodies") or [])}
        geoms = [int(g) for g in (tr.ent.objects[k].get("geoms") or [])]
        geoms += [g for g in range(int(m.ngeom)) if int(m.geom_bodyid[g]) in bodies]
        if not geoms:
            return DEFAULT_FRICTION
        return float(np.max(np.asarray(m.geom_friction)[geoms, 0]))
    except Exception:
        return DEFAULT_FRICTION


def _acceleration(tr: Track, k: int) -> np.ndarray:
    """[N] m/s^2: object k's acceleration from its recorded positions, the larger of the estimates into
    and out of each record (0 where neither exists)."""
    a = np.full(tr.n, np.nan)
    if tr.n > 2:
        v = np.asarray(tr.obj_vel[:, k], dtype=np.float64)
        dv = np.linalg.norm(np.diff(v, axis=0), axis=1) / np.asarray(tr.dt[1:], dtype=np.float64)
        a = np.fmax(np.concatenate(([np.nan], dv)), np.concatenate((dv, [np.nan])))
    return np.nan_to_num(a, nan=0.0)


def _slip_rate(tr: Track, si: int, k: int) -> np.ndarray:
    """[N] m/s: how fast the object-to-palm distance changed over each record's OWN burst - from the
    previous record's last frame to this record's last frame, the frames the record gathered (NaN for the
    first record and one that follows a break: its burst has no start to compare with, and no record
    grips unmeasured). A distance, so a held object turning with the hand does not slip.
    Before 2026-09-26 the larger of the change into and OUT OF the record was taken. The change out of a
    record is the NEXT record's burst, so a grip whose object dropped away in the record after it read
    as slipping: grasp_lift_no_retention's one lifted record moved 0.0001 m/s from the palm over its own
    burst and 0.49 m/s over the next one's (the second review's finding 2, measured 2026-09-26), and a
    lift the hand could not keep could never grip. The coordinator's instruction: slip must not count the
    drop in the record after release."""
    d = np.linalg.norm(tr.obj_pos[:, k] - tr.palm[:, si], axis=-1)
    if tr.n < 2:
        return np.full(tr.n, np.nan)
    r = np.abs(np.diff(d)) / np.asarray(tr.dt[1:], dtype=np.float64)      # NaN across a break (dt NaN)
    return np.concatenate(([np.nan], r))


def _grip(ctx: Context, side: str, k: int) -> dict:
    """Per record, whether this hand GRIPS object k (v9, the physics of a grasp; the operator's review
    adopted by the coordinator on the evening of 2026-09-25). A record grips when
      * two surfaces of the hand - the thumb and the fingers, the thumb and the palm, or the fingers
        and the palm (a palmar grasp) - touch the object in the SAME frames (_pushes_together), and
      * their pushes OPPOSE: at least OPPOSITION_MIN_DEG apart (one exclusion: adjacent faces read 90),
      * they BALANCE: what the stronger push exceeds the weaker by along the pair's axis,
        |F_a - F_b| sin(angle / 2), is no more than the object's weight and acceleration,
        m (g + |a|), and the pair's own friction, mu cos(angle / 2) (F_a + F_b), can take (180 degrees
        apart friction takes none; a hand pushing an object against something is not balanced),
      * the object is RETAINED AGAINST GRAVITY by the hand: everything else (the floor, other objects,
        the chest and the rest of the body, both forearms, the other hand, the caregiver) pushes it up
        with less than GRIP_OTHERS_MAX_WEIGHT of its weight - its support does not come mainly from the
        chest, a forearm or the environment (`supported` is the opposite, and _support's force), and
      * it SLIPS little: the object-to-palm distance changes at most GRIP_SLIP_MPS over the record's own
        burst (_slip_rate: the next record's drop is that record's, never this one's).
    The physics needs the recorder's contact normals (observer_world_v2) and a free object of known
    mass (`physics`); without them no record grips. Per record also: `angle` (the widest pair's, NaN
    where no pair touched together), `pair_deg` [N, 3] and `pair_use` [N, 3] (the imbalance over what
    can take it: 1 at the line) per OPPOSING_SURFACES pair, `pair_force` [N, 3, 2], `others_up` (N),
    `weight_n`, `slip_mps`.

    Before v9 the angle was measured at the object's CENTRE between the thumb's and the other parts'
    mean contact POINTS, any frames of the record: on the world's 6 cm boxes the grasps this hand can
    make read 83-126 degrees, so the 90-degree line ran through them, and a real pinch of the toy's thin
    pad read 30-44 (agents/observer_validation.md, defect 6). By normals the kept scripted grasp cases
    are the thumb on one face and the ring and middle fingers on ADJACENT faces (90 degrees; edge
    contacts up to 107.6): docs/gain_of_function_v4/agents/observer_recorder_2026-09-25.md."""
    key = ("grip", side, k)
    cache = ctx.__dict__.setdefault("_grip_cache", {})
    if key in cache:
        return cache[key]
    tr = ctx.tr
    n = tr.n
    si = SIDES.index(side)
    o = tr.ent.objects[k]
    obj = tr.object_mask(k)
    npair = len(OPPOSING_SURFACES)
    mass = float(o.get("mass_kg") or 0.0)
    physics = bool(getattr(tr, "has_normals", False) and o.get("free") and mass > 0.0)
    g = dict(physics=physics, together=np.zeros(n, dtype=bool), pair_together=np.zeros((n, npair), dtype=bool),
             angle=np.full(n, np.nan), pair_deg=np.full((n, npair), np.nan), pair_use=np.full((n, npair), np.nan),
             pair_force=np.full((n, npair, 2), np.nan), others_up=np.full(n, np.nan),
             weight_n=mass * GRAVITY_MPS2 if physics else float("nan"), slip_mps=np.full(n, np.nan),
             supported=np.zeros(n, dtype=bool), grip=np.zeros(n, dtype=bool))
    surface = dict(thumb=tr.ent.thumb[side], fingers=tr.ent.fingers[side], palm=tr.ent.palm[side])
    rows = {name: _pair_rows(tr, mask, obj) for name, mask in surface.items()}
    acc = _acceleration(tr, k) if physics else None
    mu = _friction(tr, side, k) if physics else DEFAULT_FRICTION
    for p, (sa, sb) in enumerate(OPPOSING_SURFACES):
        tog, fa, fb, da, db = _pushes_together(rows[sa], rows[sb], n)
        g["together"] |= tog
        g["pair_together"][:, p] = tog
        if not physics:
            continue
        la, lb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
        ok = tog & (la > 0) & (lb > 0)
        if not ok.any():
            continue
        cos = np.sum(da[ok] * db[ok], axis=1) / (la[ok] * lb[ok])
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        na, nb = np.linalg.norm(fa[ok], axis=1), np.linalg.norm(fb[ok], axis=1)
        half = np.radians(ang) / 2.0
        excess = np.abs(na - nb) * np.sin(half)
        room = mass * (GRAVITY_MPS2 + acc[ok]) + mu * np.cos(half) * (na + nb)
        g["pair_deg"][ok, p] = ang
        g["pair_use"][ok, p] = excess / np.maximum(room, 1e-12)
        g["pair_force"][ok, p, 0], g["pair_force"][ok, p, 1] = na, nb
    if physics:
        with np.errstate(invalid="ignore"):
            g["angle"] = np.fmax.reduce(g["pair_deg"], axis=1)
            pair_ok = ((g["pair_deg"] >= OPPOSITION_MIN_DEG) & (g["pair_use"] <= 1.0)).any(axis=1)
        others = _pair_rows(tr, ~tr.ent.hand[side] & ~obj, obj)
        up = np.zeros(n)
        fin = np.isfinite(others["n"]).all(axis=1)
        np.add.at(up, others["row"][fin], np.maximum(0.0, others["f"][fin] * others["n"][fin, 2]))
        g["others_up"] = up
        # a touching row without a normal cannot say how much it carries: its record counts as supported
        # (no grip, no carry). A v2 recording gives every row a normal; normals REBUILT for measuring give
        # none to a pair no longer touching at a record's last frame, and read as no support they made
        # limp bodies' boxes sliding along the floor "carried" (measured 2026-09-25 night: limp H02 0 -> 8)
        blind = np.zeros(n, dtype=bool)
        blind[others["row"][~fin]] = True
        g["supported"] = (up >= GRIP_OTHERS_MAX_WEIGHT * g["weight_n"]) | blind
        g["slip_mps"] = _slip_rate(tr, si, k)
        with np.errstate(invalid="ignore"):
            g["grip"] = pair_ok & ~g["supported"] & (g["slip_mps"] <= GRIP_SLIP_MPS)
    cache[key] = g
    return g


def _carried_m(tr: Track, k: int, support: np.ndarray, i0: int, i1: int) -> float:
    """How far object k moved while OFF ITS SUPPORT during records i0..i1 (_support: what pushes it up
    with half its weight or more, besides the holding hand; by contact where the physics cannot be
    read): in each run of records off support, its largest net distance from where the run began (the
    record it left its support, or i0 when it starts off support). Sliding along the floor under a hand
    is no carrying; a net distance, not a summed path, keeps a held object's small shake out."""
    best = 0.0
    j = i0
    while j <= i1:
        if support[j]:
            j += 1
            continue
        r0 = j
        while j <= i1 and not support[j]:
            j += 1
        origin = r0 - 1 if r0 > i0 else r0
        d = np.linalg.norm(tr.obj_pos[r0:j, k] - tr.obj_pos[origin, k], axis=-1)
        if np.isfinite(d).any():
            best = max(best, float(np.nanmax(d)))
    return best


def _holds(ctx: Context) -> List[dict]:
    """Hand-object contact intervals of HOLD_MIN_S with the hand at least somewhat closed and the
    object staying with the palm: the shared basis of H02, H03, H09, R06, R07.

    The hand CONTROLS the object when it carried the object HOLD_MOVED_M off its support (v8; v9:
    support is what pushes the object up with half its weight or more besides the holding hand -
    the floor, other objects, the chest, a forearm, the other hand - _support), or the hand GRIPPED
    it (v9: _grip, R04's physics: two surfaces of the hand opposing and balanced in the same frames,
    the object retained against gravity by the hand and not slipping, in at least one record) and
    closed HOLD_CLOSE_ON_DEG on it. Before v8 any 1 cm of movement counted: in all 8 bodies of the
    limp baseline (no command) the resting right hand lay on the light box while the arm slid, the box
    slid 2.8 cm along the floor with it, and each made one 48.7 s "hold" and "move while holding"
    (2026-09-24, docs/gain_of_function_v4/agents/observer_validation.md, defect 1). Before v9
    "opposed" was thumb and fingers touching the object in the same record, whatever side each was
    on (a palm and thumb both on a box's top face were "opposed": defect 7). On a track without
    recorded normals (observer_world_v1) nothing grips and support is by contact: only the carried
    branch can hold."""
    tr = ctx.tr
    if hasattr(ctx, "_holds_cache"):
        return ctx._holds_cache
    out = []
    nmin = int(round(HOLD_MIN_S / STEP_S))
    for k, o in enumerate(tr.ent.objects):
        if not _object_ok(o):
            continue
        for si, side in enumerate(SIDES):
            s = ctx.hand_obj(side, k)
            if not s["on"].any():
                continue
            dist = _obj_palm_dist(tr, si, k)
            g = _grip(ctx, side, k)
            support = _support(ctx, k, side)
            for i0, i1 in _intervals(s["on"], tr.continuous):
                if i1 - i0 + 1 < nmin:
                    continue
                clos = float(np.nanmean(tr.closure[i0:i1 + 1, si]))
                drift = float(np.nanmax(np.abs(dist[i0:i1 + 1] - dist[i0])))
                f = s["force"][i0:i1 + 1]
                # the hand CONTROLS the object: it carried it off its support, or gripped it; a hand
                # resting on a box, or sliding it along the floor, is contact
                moved = _carried_m(tr, k, support, i0, i1)
                seg = g["angle"][i0:i1 + 1]
                opp_max = float(np.nanmax(seg)) if np.isfinite(seg).any() else float("nan")
                grips = int(g["grip"][i0:i1 + 1].sum())
                opposed = grips > 0
                # a grip counts without moving the object only if the hand CLOSED on it (a limp hand
                # lying against a box held it for 48 s, 2026-09-24 limp baseline)
                closed_on = float(np.nanmax(tr.closure[i0:i1 + 1, si]) - tr.closure[i0, si]) \
                    if np.isfinite(tr.closure[i0:i1 + 1, si]).any() else 0.0
                out.append(dict(k=k, side=side, si=si, i0=i0, i1=i1, closure=clos, drift=drift,
                                force_mean=float(f.mean()), force_cv=float(f.std() / max(1e-9, f.mean())),
                                moved_m=moved, opposed=opposed, opposition_deg=opp_max, grip_records=grips,
                                physics=bool(g["physics"]), closed_on_deg=closed_on,
                                held=(clos >= HOLD_MIN_CLOSURE_DEG and drift <= HOLD_MAX_DRIFT_M
                                      and (moved >= HOLD_MOVED_M or (opposed and closed_on >= HOLD_CLOSE_ON_DEG))),
                                scripted=_toy_scripted(tr, k, si, i0, i1)))
    ctx._holds_cache = out
    return out


def H02(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for h in _holds(ctx):
        if not h["held"]:
            continue
        o = tr.ent.objects[h["k"]]
        out.append(ctx.det("H02", h["i0"], h["i1"], sub=h["side"], key="%s:%s" % (h["side"], o["kind"]),
                           measures=dict(hold_s=tr.seconds(h["i0"], h["i1"]), slip_m=h["drift"],
                                         force_cv=h["force_cv"], closure_deg=h["closure"],
                                         carried_cm=100.0 * h["moved_m"], opposed=h["opposed"],
                                         opposition_deg=h["opposition_deg"], grip_records=h["grip_records"],
                                         physics_read=h["physics"]),
                           confidence=_conf(_over(tr.seconds(h["i0"], h["i1"]), HOLD_MIN_S),
                                            _under(h["drift"], HOLD_MAX_DRIFT_M),
                                            _over(h["closure"], HOLD_MIN_CLOSURE_DEG)),
                           objects=[o["name"]], parts=["hand_" + h["side"]], scripted=h["scripted"]))
    return out


def _gravity(tr: Track) -> np.ndarray:
    """The world's gravity (m/s2) from the track's compiled model; MuJoCo's default without one."""
    m = getattr(getattr(tr, "kin", None), "m", None)
    g = np.asarray(getattr(getattr(m, "opt", None), "gravity", (0.0, 0.0, -9.81)), dtype=np.float64)
    return g if np.linalg.norm(g) > 0 else np.array([0.0, 0.0, -9.81])


def _last_touch_s(tr: Track, side: str, k: int, i1: int) -> float:
    """The simulated time of the last frame of record i1 in which a part of this hand touched object k.
    The recorder stores each contact's first and last frame inside the record's burst (c_frame) and keeps
    the record's time at the burst's last frame, so frame f of a burst of F frames lies F - 1 - f frames
    before the record's time. A track without that column (a made-up test track) is read as touching up
    to the record's FIRST frame, the earliest the contact can have ended: the allowance is then largest."""
    frames = getattr(tr, "frames", None)
    F = int(frames[i1]) if frames is not None and np.size(frames) > i1 and int(frames[i1]) > 0 else 5
    frame_s = float(tr.header.get("env_dt_s") or STEP_S / F)
    last = 0
    c_frame = getattr(tr, "c_frame", None)
    if c_frame is not None and np.size(c_frame):
        hand, obj = tr.ent.hand[side], tr.object_mask(k)
        for j in tr.contacts_at(i1):
            a, b = int(tr.c_e[j, 0]), int(tr.c_e[j, 1])
            if (hand[a] and obj[b]) or (hand[b] and obj[a]):
                last = max(last, int(c_frame[j, 1]))
    return float(tr.sim_time[i1]) - (F - 1 - min(last, F - 1)) * frame_s


def _escape_mps(tr: Track, k: int, i1: int, t_touch: Optional[float] = None) -> float:
    """How fast object k leaves in the two records after its last hand-contact record i1, in a way
    gravity does not explain: its speed across gravity, any speed against gravity, and any fall faster
    than the free fall gravity explains. NaN when a record is missing.

    What gravity explains (detector v9; the independent review of 2026-09-25, "H03's escape tolerates a
    downward throw 8 times faster than a sideways one"): a record's velocity is the AVERAGE over the record
    (the change of the recorded position over its 50 ms), so it is compared with the average speed a fall
    from rest reaches over the same record, the fall starting at the last frame a part of the hand touched
    the object (t_touch, _last_touch_s): g x (the record's middle - t_touch). Until then the allowance was
    the speed a fall reaches at the END of the record, starting a whole record before the last contact
    record's end: 0.98 m/s over the first record after, where a fall from rest in the last contact record
    averages 0.25-0.64 m/s. So a box thrown down up to 0.8 m/s faster than a free fall read as let go,
    while sideways the line is 0.10 m/s. Now the same ESCAPE_MPS line holds along gravity, give or take
    the one frame (10 ms: 0.098 m/s of free fall) in which the touch ended: a box thrown down escapes from
    0.10-0.20 m/s past free fall, sideways from 0.10 m/s."""
    g = _gravity(tr)
    gmag = float(np.linalg.norm(g))
    down = g / gmag
    if t_touch is None:
        t_touch = float(tr.sim_time[i1]) - 4.0 * float(tr.header.get("env_dt_s") or STEP_S / 5)
    best = 0.0
    for i in range(i1 + 1, min(tr.n, i1 + 3)):
        v = np.asarray(tr.obj_vel[i, k], dtype=np.float64)
        if not np.isfinite(v).all():
            return float("nan")
        along = float(np.dot(v, down))                      # positive: falling
        middle = 0.5 * float(tr.sim_time[i - 1] + tr.sim_time[i])
        free = gmag * max(0.0, middle - t_touch)            # a fall from rest at t_touch, averaged over the record
        excess = along - min(max(along, 0.0), free)         # against gravity, or faster than free fall
        across = v - along * down
        best = max(best, float(math.sqrt(float(np.dot(across, across)) + excess * excess)))
    return best


def H03(ctx: Context) -> List[dict]:
    """Controlled release: a hold ends after the hand opened RELEASE_OPEN_DEG in its last 300 ms,
    and the object is not escaping: in the two records after contact ends it leaves at under
    ESCAPE_MPS in a way gravity does not explain (_escape_mps, v9).

    Before v8 its speed relative to the palm was read, so a hand that let go and withdrew at 0.2 m/s
    from a box standing still read as the box escaping (scripted case grasp_release_then_withdraw,
    2026-09-24). v8 read the object's own speed, so a box let go in the air read as escaping because
    it fell (OBSERVER_DESIGN.md section 9: 'H03 counts a release into the air as an escape'): a
    release 3 cm above the floor falls at 0.25 and 0.35 m/s in the two records after, over the 0.10
    line. v9 removes what gravity explains and keeps the rest: a box pulled out sideways or upward,
    or flung down, still escapes."""
    tr = ctx.tr
    out = []
    w = int(round(0.3 / STEP_S))
    for h in _holds(ctx):
        if not h["held"] or h["i1"] >= tr.n - 2:
            continue
        i1, si, k = h["i1"], h["si"], h["k"]
        a = max(h["i0"], i1 - w)
        opened = float(np.nanmax(tr.closure[a:i1 + 1, si]) - tr.closure[min(tr.n - 1, i1 + 1), si])
        speed = float(np.nanmax(tr.obj_speed[i1 + 1:min(tr.n, i1 + 3), k]))
        escape = _escape_mps(tr, k, i1, _last_touch_s(tr, h["side"], k, i1))
        if opened < RELEASE_OPEN_DEG or not (escape < ESCAPE_MPS):
            continue
        start = a + int(np.nanargmax(tr.closure[a:i1 + 1, si]))
        after = ctx.hand_obj(h["side"], k)["on"][i1 + 1:min(tr.n, i1 + 1 + w)]
        o = tr.ent.objects[k]
        out.append(ctx.det("H03", start, min(tr.n - 1, i1 + 1), sub=h["side"], key="%s:%s" % (h["side"], o["kind"]),
                           measures=dict(release_latency_ms=1000.0 * tr.seconds(start, min(tr.n - 1, i1 + 1)),
                                         opening_deg=opened, object_speed_mps=speed, escape_mps=escape,
                                         residual_contact_steps=int(after.sum())),
                           confidence=_conf(_over(opened, RELEASE_OPEN_DEG), _under(escape, ESCAPE_MPS)),
                           objects=[o["name"]], parts=["hand_" + h["side"]], scripted=h["scripted"]))
    return out


def H04(ctx: Context) -> List[dict]:
    """Open-close cycle: three or more closing and opening turns of CYCLE_MIN_DEG, alternating,
    each FOLLOWING the one before - the next closing or opening turn of that hand (no smaller one
    between) in an unbroken record - within CYCLE_MAX_S, with no touch starting on the hand.

    Until v8 'following' was 'starts within 2 records of the previous turn's end', which held for
    any two consecutive turns because turns tiled the record. Since v9 a turn starts where its own
    movement starts (primitives._turns), after any pause at the previous extreme; the pause lies
    inside the cycle and CYCLE_MAX_S bounds it, as it did when the pause was the next turn's start."""
    tr = ctx.tr
    out = []
    for si, side in enumerate(SIDES):
        every = [c for c in ctx.by_type["curl"] if c["side"] == side]
        order = {id(c): n for n, c in enumerate(every)}
        turns = [c for c in every if abs(c["change_deg"]) >= CYCLE_MIN_DEG]
        onsets = [c["i0"] for c in _hand_onsets(ctx, side)]

        def follows(p, q) -> bool:
            return (order[id(q)] == order[id(p)] + 1 and q["i0"] >= p["i1"]
                    and bool(np.all(tr.continuous[p["i1"] + 1:q["i0"] + 1])))
        j = 0
        while j + 2 < len(turns):
            a, b, c = turns[j], turns[j + 1], turns[j + 2]
            alternating = (np.sign(a["change_deg"]) != np.sign(b["change_deg"])
                           and np.sign(b["change_deg"]) != np.sign(c["change_deg"]))
            contiguous = follows(a, b) and follows(b, c)
            span = tr.seconds(a["i0"], c["i1"])
            passive = any(a["i0"] <= i <= c["i1"] for i in onsets)
            if alternating and contiguous and span <= CYCLE_MAX_S and not passive:
                amps = [abs(x["change_deg"]) for x in (a, b, c)]
                k = j + 3
                while (k < len(turns) and follows(turns[k - 1], turns[k])
                       and np.sign(turns[k]["change_deg"]) != np.sign(turns[k - 1]["change_deg"])
                       and not any(turns[k]["i0"] <= i <= turns[k]["i1"] for i in onsets)):
                    amps.append(abs(turns[k]["change_deg"]))
                    k += 1
                last = turns[k - 1]
                # the fingers' own motors drove it (MOTOR_ON): a limp hand's fingers settling are not
                peak = _command_peak(tr, _flex_adrs(tr, side, FINGERS), a["i0"], last["i1"])
                work = _motor_work(tr, _flex_adrs(tr, side, FINGERS), a["i0"], last["i1"])
                if _passive(peak) or _against(work):
                    j = k
                    continue
                out.append(ctx.det("H04", a["i0"], last["i1"], sub=side, key=side,
                                   measures=dict(half_cycles=len(amps), amplitude_deg=float(np.mean(amps)),
                                                 amplitude_cv=float(np.std(amps) / max(1e-9, np.mean(amps))),
                                                 period_s=2.0 * tr.seconds(a["i0"], last["i1"]) / len(amps),
                                                 command_peak=peak, motor_work=work),
                                   confidence=_conf(_over(min(amps), CYCLE_MIN_DEG)),
                                   parts=["hand_" + side],
                                   open_end=any(t.get("open_end", False) for t in turns[j:k])))
                j = k
            else:
                j += 1
    return out


# A HAND OR WRIST ACT IS DRIVEN BY ITS OWN MOTORS (detector v9; OBSERVER_DESIGN.md section 9, "a limp
# body still makes one passive finger fan per body as the arm settles"). Every MIMo hand and wrist
# joint has its own motor, and the recorder stores each record's motor commands (ctrl, -1..1). The
# rows built on joint turns (H04, H05, H07, H08, HX1, HX2) count a movement only when the motors of
# the joints that made it carried a command during it (at least MOTOR_ON). Measured 2026-09-25
# (agents/observer_detectors_2026-09-25.md): all 17 of those rows' acts in the two limp recordings
# moved with exactly zero command on their joints - the reset world's fingers swinging to their
# range limits one after another as the limp hand settled (the 8 fans of 30-32 degrees), and the
# saved worlds' hands relaxing when their commands went to zero at the first record; the weakest
# command on any act in babble or production was 0.0028 (random actuator commands; random orders
# 0.0085, production 0.027). A kinematic definition cannot tell these apart (observer_validation.md);
# the commands can. Scripted validation cases stipulate commands for the joints they move
# (validate.ScriptWorld.record, motors="drive") and zero for a passive movement (motors="off").
MOTOR_ON = 0.001


def _motor_of_qpos(tr: Track) -> Dict[int, int]:
    """qpos address -> the motor (an actuator with a joint transmission) that drives that joint, from
    the track's compiled model; {} for a track without one (a made-up test track)."""
    kin = getattr(tr, "kin", None)
    m = getattr(kin, "m", None)
    if m is None:
        return {}
    cache = getattr(kin, "_motor_of_qpos", None)
    if cache is None:
        cache = {}
        for u in range(int(m.nu)):
            if int(m.actuator_trntype[u]) == 0:          # mjTRN_JOINT
                cache[int(m.jnt_qposadr[int(m.actuator_trnid[u][0])])] = u
        kin._motor_of_qpos = cache
    return cache


def _command_peak(tr: Track, qadrs, i0: int, i1: int) -> float:
    """The largest |command| the motors of the joints at these qpos addresses received over records
    i0..i1; NaN when the track holds no commands or none of the joints has a motor (then no gate)."""
    mp = _motor_of_qpos(tr)
    us = [mp[int(a)] for a in qadrs if int(a) in mp]
    ctrl = getattr(tr, "ctrl", None)
    if not us or ctrl is None or np.size(ctrl) == 0:
        return float("nan")
    c = np.abs(np.asarray(ctrl[i0:i1 + 1], dtype=np.float64)[:, us])
    return float(np.nanmax(c)) if c.size and np.isfinite(c).any() else float("nan")


def _passive(peak: float) -> bool:
    """The joints moved with their motors silent (a finite peak command under MOTOR_ON)."""
    return bool(np.isfinite(peak) and peak < MOTOR_ON)


# THE MOTORS DROVE THE MOVEMENT (detector v9, 2026-09-25, the coordinator's decision of 8:40 PM;
# agents/observer_detectors_2026-09-25.md, "The motor-work rule"). A command on the moving joints (MOTOR_ON)
# does not say the motors MADE the movement: in production 36-43 of every 100 palm-shaping acts (HX2), 11-15
# of the spreading acts (HX1) and 2-4 of the finger isolations (H05) moved their joints AGAINST the joints'
# own commands (babble: 18, 8 and 7) - moved by a contact, by gravity, by the joints' springs or by the arm's
# motion while the motors resisted. A hand or wrist act now counts only when the motors' work on its joints
# over the act is positive (_motor_work: each record's command torque times the joint's turn in that record,
# summed). The cost, named: a finger that relaxes a command against its spring (the command falling while the
# spring returns the joint) does negative work too, and is no longer counted as that movement.
def _motor_gains(tr: Track) -> Dict[int, float]:
    """motor index -> gear x gain (the torque per unit of command) from the track's compiled model."""
    kin = getattr(tr, "kin", None)
    m = getattr(kin, "m", None)
    if m is None:
        return {}
    cache = getattr(kin, "_motor_gains", None)
    if cache is None:
        cache = {u: float(m.actuator_gear[u][0]) * float(m.actuator_gainprm[u][0]) for u in range(int(m.nu))}
        kin._motor_gains = cache
    return cache


def _motor_work(tr: Track, qadrs, i0: int, i1: int) -> float:
    """The work the recorded motor commands did on the joints at these qpos addresses over records i0..i1
    (N m x rad): for each record r after i0, the command of record r (the one applied during its burst)
    times gear x gain, times the joint's turn from record r - 1 to r, summed over the joints. Positive: the
    motors pushed the way the joints went. NaN when the track holds no commands or none of the joints has a
    motor (then no gate)."""
    mp = _motor_of_qpos(tr)
    gains = _motor_gains(tr)
    ctrl = getattr(tr, "ctrl", None)
    pairs = [(int(a), mp[int(a)]) for a in qadrs if int(a) in mp]
    if not pairs or ctrl is None or np.size(ctrl) == 0 or i1 <= i0:
        return float("nan")
    work = 0.0
    for a, u in pairs:
        dq = np.diff(np.asarray(tr.qpos[i0:i1 + 1, a], dtype=np.float64))
        c = np.asarray(ctrl[i0 + 1:i1 + 1, u], dtype=np.float64)
        ok = np.isfinite(dq) & np.isfinite(c)
        work += float(np.sum(gains.get(u, 1.0) * c[ok] * dq[ok]))
    return work


def _against(work: float) -> bool:
    """The joints moved against their own motors (a finite motor work that is not positive)."""
    return bool(np.isfinite(work) and work <= 0.0)


def _joint_adr(tr: Track, name: str) -> Optional[int]:
    for j in tr.header.get("joints", []):
        if j["name"].split(":")[-1] == name:
            return int(j["qpos"])
    return None


def _flex_adrs(tr: Track, side: str, digits) -> List[int]:
    kin = getattr(tr, "kin", None)
    flex = getattr(kin, "flex_q", None) or {}
    return [a for dg in digits for a in flex.get((side, dg), [])]


def _wrist_adrs(tr: Track, side: str, kind: str) -> List[int]:
    kin = getattr(tr, "kin", None)
    if kin is None or not hasattr(kin, "wrist_q"):
        return []
    return [a for role, a in kin.wrist_q.get(side, {}).items() if kin.wrist_kind.get(side, {}).get(role) == kind]


def _rise_s(tr: Track, x: np.ndarray, i0: int, i1: int, ch: float) -> float:
    """A turn's 10-90% rise in seconds (primitives.rise_s: HX1's rise since v8)."""
    return P.rise_s(tr, x, i0, i1, ch)


def _line_s(tr: Track, x: np.ndarray, i0: int, i1: int, ch: float, line: float) -> float:
    """The act timescale's measure: the seconds the turn's fastest stretch took to move ACT_LINE_SHARE of the
    row's line (primitives.line_s; WHERE THE TIMESCALE IS MEASURED, above)."""
    return P.line_s(tr, x, i0, i1, ch, ACT_LINE_SHARE * float(line))


def others_excursion(tr: Track, si: int, di: int, i0: int, i1: int) -> Dict[str, float]:
    """How far each of the other four digits of the hand moved INSIDE records i0..i1: its largest |flexion -
    flexion at i0|, degrees, by digit name (observer_attribution_v1, 2026-10-08)."""
    out = {}
    for x in range(5):
        if x == di:
            continue
        seg = np.asarray(tr.curl[i0:i1 + 1, si, x], dtype=np.float64)
        dev = np.abs(seg - seg[0])
        out[DIGITS[x]] = float(np.nanmax(dev)) if np.isfinite(dev).any() else float("nan")
    return out


def H05(ctx: Context) -> List[dict]:
    """Individual digit movement with the others' END positions unchanged: the finger rows (section 32). One digit's
    turn of at least primitives.DIGIT_MIN_DEG whose fastest stretch moves that line within ACT_MAX_RISE_S
    (v9, _line_s), driven by that digit's own motors (MOTOR_ON, v9), while the other four digits' flexion at the act's
    last record differs from their flexion at its first record by at most ISOLATION_OTHERS_MAX_DEG on average.

    WHAT "ALONE" MEASURES (observer_attribution_v1, 2026-10-08; observer source review F3,
    D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/observer/REVIEW.md). The rule above compares the other digits at the act's
    two ends only, so a digit that moved and came back inside the act counts as still: on the 47 left-index acts of
    seg41c_0841 another digit moved 15 degrees or more inside 19 of them. The rule is unchanged (which acts are found,
    and their golden detections, stay as they were); each act now also reports how far each other digit moved inside it
    (`others_max_excursion_deg`, by digit; their mean `others_mean_excursion_deg`) and `alone_inside`: that mean at
    most ISOLATION_OTHERS_MAX_DEG, the same line applied to the movement inside the act. The report counts it apart."""
    tr = ctx.tr
    out = []
    for d in ctx.by_type["digit"]:
        si = SIDES.index(d["side"])
        di = DIGITS.index(d["digit"])
        i0, i1 = d["i0"], d["i1"]
        others = [x for x in range(5) if x != di]
        ch_o = np.abs(tr.curl[i1, si, others] - tr.curl[i0, si, others])
        mean_o = float(np.nanmean(ch_o))
        ratio = abs(d["change_deg"]) / max(mean_o, 1.0)
        if mean_o <= ISOLATION_OTHERS_MAX_DEG and ratio >= ISOLATION_MIN_RATIO:
            rise = _rise_s(tr, tr.curl[:, si, di], i0, i1, d["change_deg"])
            line = _line_s(tr, tr.curl[:, si, di], i0, i1, d["change_deg"], P.DIGIT_MIN_DEG)
            if line > ACT_MAX_RISE_S:
                continue
            peak = _command_peak(tr, _flex_adrs(tr, d["side"], [d["digit"]]), i0, i1)
            work = _motor_work(tr, _flex_adrs(tr, d["side"], [d["digit"]]), i0, i1)
            if _passive(peak) or _against(work):
                continue
            exc = others_excursion(tr, si, di, i0, i1)
            vals = [v for v in exc.values() if np.isfinite(v)]
            mean_exc = float(np.mean(vals)) if vals else float("nan")
            out.append(ctx.det("H05", i0, i1, sub="%s:%s" % (d["side"], d["digit"]),
                               key="%s:%s:%s" % (d["side"], d["digit"], d["direction"]),
                               measures=dict(isolation_ratio=ratio, change_deg=d["change_deg"],
                                             others_mean_deg=mean_o, rise_s=rise, line_s=line, command_peak=peak,
                                             motor_work=work, others_max_excursion_deg=exc,
                                             others_mean_excursion_deg=mean_exc),
                               confidence=_conf(_under(mean_o, ISOLATION_OTHERS_MAX_DEG),
                                                _over(ratio, ISOLATION_MIN_RATIO)),
                               parts=["finger_%s_%s" % (d["side"], d["digit"])], direction=d["direction"],
                               alone_inside=(bool(mean_exc <= ISOLATION_OTHERS_MAX_DEG) if np.isfinite(mean_exc)
                                             else None),
                               open_end=bool(d.get("open_end", False))))
    return out


def H06(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for si, side in enumerate(SIDES):
        th = tr.tips[:, si, 4]
        dists = np.stack([np.linalg.norm(tr.tips[:, si, f] - th, axis=-1) for f in range(4)], axis=1)
        dmin = np.nanmin(dists, axis=1)
        near = dmin <= OPPOSE_NEAR_M
        for i0, i1 in _intervals(near, tr.continuous):
            a = max(0, i0 - 10)
            if not tr.continuous[a + 1:i0 + 1].all():
                continue
            approach = float(dmin[a] - dmin[i0])
            swivel = float(tr.swivel[i0, si] - np.nanmin(tr.swivel[a:i0 + 1, si]))
            if approach < OPPOSE_APPROACH_M or swivel < OPPOSE_SWIVEL_DEG:
                continue
            f = int(np.nanargmin(dists[i0]))
            v = np.diff(dmin[a:i0 + 1])
            reversals = int(np.sum(np.diff(np.sign(v[np.abs(v) > 1e-4])) != 0)) if v.size > 2 else 0
            out.append(ctx.det("H06", a, i1, sub="%s:%s" % (side, FINGERS[f]), key="%s:%s" % (side, FINGERS[f]),
                               measures=dict(min_distance_cm=100.0 * float(np.nanmin(dmin[i0:i1 + 1])),
                                             approach_cm=100.0 * approach, swivel_increase_deg=swivel,
                                             approach_reversals=reversals),
                               confidence=_conf(_over(approach, OPPOSE_APPROACH_M), _over(swivel, OPPOSE_SWIVEL_DEG)),
                               parts=["finger_%s_thumb" % side, "finger_%s_%s" % (side, FINGERS[f])]))
    return out


def _wrist(ctx: Context, cap: str, kind: str) -> List[dict]:
    """H07 and H08: a wrist turn of at least primitives.WRIST_MIN_DEG whose fastest stretch moves that line
    within ACT_MAX_RISE_S (_line_s)
    and driven by that wrist joint's own motor (MOTOR_ON; both v9), with the forearm turning at
    most WRIST_FOREARM_MAX_DEG (flexion) or the palm moving at most WRIST_ROT_MAX_TRANSLATION_M
    (rotation) over it."""
    tr = ctx.tr
    out = []
    k_axis = ("rotation", "flexion", "deviation").index(kind)
    for w in ctx.by_type["wrist"]:
        if w["kind"] != kind:
            continue
        si = SIDES.index(w["side"])
        i0, i1 = w["i0"], w["i1"]
        rise = _rise_s(tr, tr.wrist[:, si, k_axis], i0, i1, w["change_deg"])
        line = _line_s(tr, tr.wrist[:, si, k_axis], i0, i1, w["change_deg"], P.WRIST_MIN_DEG)
        if line > ACT_MAX_RISE_S:
            continue
        peak = _command_peak(tr, _wrist_adrs(tr, w["side"], kind), i0, i1)
        work = _motor_work(tr, _wrist_adrs(tr, w["side"], kind), i0, i1)
        if _passive(peak) or _against(work):
            continue
        if kind == "flexion":
            f0, f1 = tr.forearm[i0, si], tr.forearm[i1, si]
            fore = float(np.degrees(np.arccos(np.clip(np.dot(f0, f1), -1, 1)))) if np.isfinite(f0).all() and np.isfinite(f1).all() else float("nan")
            if not (fore <= WRIST_FOREARM_MAX_DEG):
                continue
            seg = tr.wrist[i0:min(tr.n, i1 + 4), si, 1]
            overshoot = float(abs(np.nanmax(seg) - tr.wrist[i1, si, 1])) if w["change_deg"] > 0 else float(abs(np.nanmin(seg) - tr.wrist[i1, si, 1]))
            out.append(ctx.det(cap, i0, i1, sub=w["side"], key="%s:%s" % (w["side"], "flex" if w["change_deg"] > 0 else "extend"),
                               measures=dict(range_deg=abs(w["change_deg"]), forearm_turn_deg=fore, overshoot_deg=overshoot,
                                             rise_s=rise, line_s=line, command_peak=peak, motor_work=work),
                               confidence=_conf(_over(abs(w["change_deg"]), P.WRIST_MIN_DEG), _under(fore, WRIST_FOREARM_MAX_DEG)),
                               parts=["hand_%s_palm" % w["side"]], open_end=bool(w.get("open_end", False))))
        else:
            trans = float(np.linalg.norm(tr.palm[i1, si] - tr.palm[i0, si]))
            if not (trans <= WRIST_ROT_MAX_TRANSLATION_M):
                continue
            sp = np.diff(tr.wrist[i0:i1 + 1, si, 0])
            out.append(ctx.det(cap, i0, i1, sub=w["side"], key="%s:%s" % (w["side"], "in" if w["change_deg"] > 0 else "out"),
                               measures=dict(angle_deg=abs(w["change_deg"]), palm_translation_cm=100.0 * trans,
                                             smoothness=float(1.0 - np.mean(np.diff(np.sign(sp)) != 0)) if sp.size > 1 else 1.0,
                                             rise_s=rise, line_s=line, command_peak=peak, motor_work=work),
                               confidence=_conf(_over(abs(w["change_deg"]), P.WRIST_MIN_DEG),
                                                _under(trans, WRIST_ROT_MAX_TRANSLATION_M)),
                               parts=["hand_%s_palm" % w["side"]], open_end=bool(w.get("open_end", False))))
    return out


def H07(ctx):
    return _wrist(ctx, "H07", "flexion")


def H08(ctx):
    return _wrist(ctx, "H08", "rotation")


def _slips(ctx: Context, h: dict):
    """(slip start, recovered index or None) within a hold of a free object."""
    tr = ctx.tr
    if not tr.ent.objects[h["k"]]["free"]:
        return []
    rel = _rel_speed(tr, h["si"], h["k"])
    out = []
    i = h["i0"] + 1
    while i <= h["i1"]:
        if rel[i] >= SLIP_MPS:
            j = i
            while j <= h["i1"] and not (rel[j] < SLIP_STOP_MPS):
                j += 1
            out.append((i, j if j <= h["i1"] else None))
            i = j + 1
        else:
            i += 1
    return out


def H09(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    w = int(round(0.3 / STEP_S))
    for h in _holds(ctx):
        if not h["held"]:
            continue
        f = ctx.hand_obj(h["side"], h["k"])["force"]
        for s0, s1 in _slips(ctx, h):
            pre = float(np.mean(f[max(h["i0"], s0 - 4):s0])) if s0 > h["i0"] else float(f[s0])
            peak = float(np.max(f[s0:min(tr.n, s0 + w + 1)]))
            if pre <= 0 or peak < pre * (1 + GRIP_RISE) or s1 is None:
                continue
            relaxed = float(np.mean(f[s1:min(h["i1"] + 1, s1 + 4)])) if s1 is not None else float("nan")
            out.append(ctx.det("H09", s0, s1, sub=h["side"], key=h["side"],
                               measures=dict(force_rise=peak / pre - 1.0, recovery_ms=1000.0 * tr.seconds(s0, s1),
                                             excess_force_ratio=peak / max(1e-9, relaxed)),
                               confidence=_conf(_over(peak / pre - 1.0, GRIP_RISE)),
                               objects=[tr.ent.objects[h["k"]]["name"]], parts=["hand_" + h["side"]]))
    return out


# ====================================================================== self and body
def _self_contacts(ctx: Context, mask_a, mask_b) -> List[dict]:
    out = []
    for c in ctx.by_type["contact"]:
        if c["open_start"] or not c["self_self"]:
            continue
        a, b = c["a"], c["b"]
        if (mask_a[a] and mask_b[b]) or (mask_a[b] and mask_b[a]):
            out.append(c)
    return out


def B01(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for c in _self_contacts(ctx, tr.ent.hand["L"], tr.ent.hand["R"]):
        i = c["i0"]
        pos = np.asarray(c["pos"])
        lat = float(abs(np.dot(pos - tr.trunk_pos[i], tr.trunk_lat[i]))) if np.isfinite(tr.trunk_lat[i]).all() else float("nan")
        a = max(0, i - 10)
        dpp = np.linalg.norm(tr.palm[:, 0] - tr.palm[:, 1], axis=-1)
        conv = float(dpp[a] - dpp[i])
        if not (lat <= MIDLINE_M) or conv < CONVERGE_M:
            continue
        out.append(ctx.det("B01", a, c["i1"], key="hands",
                           measures=dict(midline_offset_cm=100.0 * lat, converged_cm=100.0 * conv,
                                         contact_s=c["duration_s"] + STEP_S, force_n=c["peak_force_n"]),
                           confidence=_conf(_under(lat, MIDLINE_M), _over(conv, CONVERGE_M)),
                           parts=["hand_L", "hand_R"], onset_index=i, contact_entities=[c["a"], c["b"]]))
    return out


def B02(ctx: Context) -> List[dict]:
    tr = ctx.tr
    tips = np.zeros(tr.ent.n, dtype=bool)
    for m in tr.ent.fingertip.values():
        tips |= m
    out = []
    neighbours = {("index", "middle"), ("middle", "ring"), ("ring", "little")}
    for c in _self_contacts(ctx, tips, tips):
        i = c["i0"]
        sa, sb = tr.ent.side_of_hand(c["a"]), tr.ent.side_of_hand(c["b"])
        da, db = tr.ent.group[c["a"]].split("_")[-1], tr.ent.group[c["b"]].split("_")[-1]
        if sa == sb and (tuple(sorted((da, db), key=DIGITS.index)) in neighbours):
            continue                  # neighbouring fingertips of one hand touch when fingers move
        ta = tr.tips[:, SIDES.index(sa), DIGITS.index(da)]
        tb = tr.tips[:, SIDES.index(sb), DIGITS.index(db)]
        dd = np.linalg.norm(ta - tb, axis=-1)
        a0 = max(0, i - 5)
        converge = float(dd[a0] - dd[i]) if np.isfinite(dd[a0]) and np.isfinite(dd[i]) else 0.0
        if converge < B02_CONVERGE_M:
            continue
        rows = tr.contacts_at(i)
        sides = {tr.ent.side_of_hand(c["a"]), tr.ent.side_of_hand(c["b"])}
        collateral = 0
        for j in rows:
            e1, e2 = int(tr.c_e[j, 0]), int(tr.c_e[j, 1])
            if {e1, e2} == {c["a"], c["b"]}:
                continue
            if tr.ent.is_self[e1] and tr.ent.is_self[e2] and (
                    tr.ent.side_of_hand(e1) in sides or tr.ent.side_of_hand(e2) in sides):
                collateral += 1
        if collateral > 1:
            continue
        la, lb = sorted([tr.ent.group[c["a"]], tr.ent.group[c["b"]]])
        out.append(ctx.det("B02", i, c["i1"], sub="%s+%s" % (la, lb), key="%s+%s" % (la, lb),
                           measures=dict(collateral_contacts=collateral, contact_s=c["duration_s"] + STEP_S,
                                         force_n=c["peak_force_n"], converged_cm=100.0 * converge),
                           confidence=_conf(_under(collateral, 1.0) if collateral else 1.0),
                           parts=[la, lb]))
    return out


def _hand_head(ctx: Context):
    tr = ctx.tr
    for side in SIDES:
        for c in _self_contacts(ctx, tr.ent.hand[side], tr.ent.head):
            yield side, c


def _zone(ctx: Context, i: int, pos) -> str:
    tr = ctx.tr
    hb = tr.kin.head
    if hb is None or not np.isfinite(tr.head_R[i]).all():
        return "head"
    local = tr.head_R[i].T @ (np.asarray(pos) - tr.xpos[i, hb])
    return tr.kin.face_zone(local)


def B03(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    w = int(round(FACE_APPROACH_S / STEP_S))
    for side, c in _hand_head(ctx):
        i = c["i0"]
        zone = _zone(ctx, i, c["pos"])
        if zone not in ("face", "mouth"):
            continue
        si = SIDES.index(side)
        sp = tr.palm_speed[max(0, i - w):i + 1, si]
        # the hand approaches the face: a head turning into a resting hand is not this act
        if not (np.isfinite(sp).any() and float(np.nanmean(sp)) >= FACE_APPROACH_MPS):
            continue
        out.append(ctx.det("B03", max(0, i - w), c["i1"], sub=side, key="%s:%s" % (side, zone),
                           measures=dict(approach_speed_mps=float(np.nanmean(sp)) if np.isfinite(sp).any() else float("nan"),
                                         force_n=c["peak_force_n"], contact_s=c["duration_s"] + STEP_S),
                           confidence=1.0, parts=["hand_" + side, zone], zone=zone, onset_index=i))
    return out


def B04(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    nmin = max(1, int(round(MOUTH_MIN_S / STEP_S)))
    from .kinematics import MOUTH_RADIUS_M
    for si, side in enumerate(SIDES):
        pts = np.concatenate([tr.palm[:, si:si + 1], tr.tips[:, si]], axis=1)
        dm = np.nanmin(np.linalg.norm(pts - tr.mouth[:, None, :], axis=-1), axis=1)
        inside = dm <= MOUTH_RADIUS_M
        touching = np.zeros(tr.n, dtype=bool)
        for s, c in _hand_head(ctx):
            if s == side and _zone(ctx, c["i0"], c["pos"]) == "mouth":
                touching[c["i0"]:c["i1"] + 1] = True
        for i0, i1 in _intervals(inside | touching, tr.continuous):
            if i1 - i0 + 1 < nmin + 1:
                continue
            out.append(ctx.det("B04", i0, i1, sub=side, key=side,
                               measures=dict(min_distance_cm=100.0 * float(np.nanmin(dm[i0:i1 + 1])),
                                             duration_s=tr.seconds(i0, i1),
                                             contact_steps=int(touching[i0:i1 + 1].sum())),
                               confidence=_conf(_under(float(np.nanmin(dm[i0:i1 + 1])), MOUTH_RADIUS_M)),
                               parts=["hand_" + side, "mouth"]))
    return out


def B05(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for si, side in enumerate(SIDES):
        other = SIDES[1 - si]
        limb = tr.ent.arm[other] | tr.ent.leg[other]
        for c in _self_contacts(ctx, tr.ent.hand[side], limb):
            i = c["i0"]
            target = c["b_class"] if tr.ent.hand[side][c["a"]] else c["a_class"]
            cross = float(np.dot(tr.palm[i, si] - tr.trunk_pos[i], tr.trunk_lat[i])) if np.isfinite(tr.trunk_lat[i]).all() else float("nan")
            # the reach crosses the body's midline: the trunk's left axis is positive toward the
            # left, so a left hand must be at or past the midline (within 2 cm) on the right
            past = (cross <= MIDLINE_TOLERANCE_M) if side == "L" else (cross >= -MIDLINE_TOLERANCE_M)
            if not (np.isfinite(cross) and past):
                continue
            out.append(ctx.det("B05", i, c["i1"], sub="%s>%s" % (side, target), key="%s>%s" % (side, target.split("_")[0]),
                               measures=dict(palm_lateral_cm=100.0 * cross, force_n=c["peak_force_n"],
                                             contact_s=c["duration_s"] + STEP_S),
                               confidence=1.0, parts=["hand_" + side, target]))
    return out


def B06(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    w = int(round(0.25 / STEP_S))
    for v in ctx.by_type["hand_view"]:
        i = v["i0"]
        if i == 0 or not tr.continuous[i]:
            continue
        si = SIDES.index(v["side"])
        sp = tr.palm_speed[max(1, i - w):i + 1, si]
        head = float(np.nansum(tr.head_ang_speed[max(1, i - w):i + 1] * tr.dt[max(1, i - w):i + 1]))
        speed = float(np.nanmax(sp)) if np.isfinite(sp).any() else 0.0
        if speed < VIEW_ARM_SPEED_MPS or head > VIEW_HEAD_MAX_DEG:
            continue
        out.append(ctx.det("B06", max(0, i - w), v["i1"], sub=v["side"], key=v["side"],
                           measures=dict(arm_speed_mps=speed, head_turn_deg=head, min_angle_deg=v["min_angle_deg"],
                                         in_fovea=bool(v["in_fovea"]), seen_by_ray=bool(v["ray_hit"]),
                                         view_s=v["duration_s"] + STEP_S),
                           confidence=_conf(_over(speed, VIEW_ARM_SPEED_MPS), _under(head, VIEW_HEAD_MAX_DEG)),
                           parts=["hand_" + v["side"]], entry_index=i))
    return out


# ====================================================================== gaze
def G01(ctx: Context) -> List[dict]:
    out = []
    for e in ctx.by_type["eye_move"]:
        head = e["head_change_deg"]
        if not (np.isfinite(head) and head <= G01_HEAD_MAX_DEG):
            continue
        mover = "world_saccade" if e["world_saccade"] else "world_reflex" if e["world_reflex"] else "learner"
        out.append(ctx.det("G01", e["i0"], e["i1"], sub="LR"[e["eye"]], key=mover,
                           measures=dict(change_deg=e["change_deg"], head_change_deg=head),
                           confidence=_conf(_over(e["change_deg"], P.EYE_MIN_DEG), _under(head, G01_HEAD_MAX_DEG)),
                           parts=["eye_" + "LR"[e["eye"]]], eye_mover=mover,
                           scripted=mover != "learner"))
    return out


def _salient(tr: Track, entity: int) -> bool:
    k = int(tr.ent.kind[entity])
    return k in (KIND_SELF, KIND_OBJECT, KIND_CAREGIVER, KIND_ART)


def _fixation_measures(f: dict) -> dict:
    """The foveal bundle's readings of one fixation (primitives.fixations, detector v7), and since v9
    `inner_share`, the share of the 17 inner rays (centre and the rings at 2 and 6 degrees) on the
    target: None on a track recorded without them."""
    return dict(ray_share=f.get("ray_share"), aim_error_deg=f.get("error_deg"),
                distance_cm=None if f.get("distance_m") is None else 100.0 * f["distance_m"],
                target_switch=bool(f.get("switch")), inner_share=f.get("inner_share"))


def G02(ctx: Context) -> List[dict]:
    """Brief fixation: the left eye's fixation (its foveal bundle on one target: the centre ray's
    thing, primitives.fixations) on an object, a body part, the caregiver or a painted motif, for
    at least G02_MIN_S. Only the left eye counts, so one look with both eyes is one act."""
    tr = ctx.tr
    out = []
    for f in ctx.by_type["fixation"]:
        if f["eye"] != 0 or not _salient(tr, f["entity"]) or f["duration_s"] < G02_MIN_S:
            continue
        out.append(ctx.det("G02", f["i0"], f["i1"], sub=f["target_class"], key=f["target_class"],
                           measures=dict(duration_s=f["duration_s"] + STEP_S, jitter_deg=f["jitter_deg"],
                                         **_fixation_measures(f)),
                           confidence=_conf(_over(f["duration_s"], G02_MIN_S)),
                           objects=[f["target_label"]], target=f["target"],
                           previous_target=f.get("previous_target")))
    return out


def G03(ctx: Context) -> List[dict]:
    """Return fixation: a fixation of G02's kind, then the eye LOOKS AWAY - for at least one step
    none of its nine foveal rays is on the target (detector v6; before, any gap between two
    fixations counted, so a gaze that stayed on the target with a minority of its rays for 0.3 s
    'returned') - then fixates the same target again within G03_AWAY_S, with no external cue for
    the target in between (it did not move, the adult did not arrive)."""
    tr = ctx.tr
    out = []
    fx = [f for f in ctx.by_type["fixation"] if f["eye"] == 0 and _salient(tr, f["entity"])
          and f["duration_s"] >= G02_MIN_S]
    rays = {}
    for a_i, a in enumerate(fx):
        for b in fx[a_i + 1:]:
            gap = tr.seconds(a["i1"], b["i0"])
            if gap > G03_AWAY_S[1]:
                break
            if b["target"] != a["target"]:
                continue
            if gap < G03_AWAY_S[0] or not tr.continuous[a["i1"] + 1:b["i0"] + 1].all():
                break
            if a["target"] not in rays:
                rays[a["target"]] = tr.rays_on(a["target"], 0)
            away = int(np.sum(rays[a["target"]][a["i1"] + 1:b["i0"]] == 0))
            if away < 1:
                break                        # the target never left the fovea: no return
            # an external cue for the target during the gap: it moved, or the adult arrived
            k = tr.ent.object_of_entity.get(a["entity"])
            moved = bool(k is not None and np.nan_to_num(tr.obj_speed[a["i1"]:b["i0"] + 1, k]).max() >= G03_CUE_MPS)
            arrived = any(e["kind"] == "arrive" and a["i1"] <= e["i0"] <= b["i0"] for e in ctx.by_type["caregiver"])
            if not moved and not arrived:
                out.append(ctx.det("G03", a["i0"], b["i1"], sub=a["target_class"], key=a["target_class"],
                                   measures=dict(return_latency_s=gap, first_fixation_s=a["duration_s"] + STEP_S,
                                                 away_s=away * STEP_S, ray_share=b.get("ray_share"),
                                                 aim_error_deg=b.get("error_deg"),
                                                 inner_share=b.get("inner_share")),
                                   confidence=1.0, objects=[a["target_label"]]))
            break
    return out


def G04(ctx: Context) -> List[dict]:
    """Tracking: while an object moves for G04_MIN_S, the angle between the left eye's centre ray
    and the object stays within the fovea's half-angle for G04_ON_FRACTION of the time, and the
    gaze itself turns with it. The addendum defines tracking by this angular error: a moving toy
    can stay inside the fovea while the centre ray lags or leads it. A MAJORITY of the foveal rays
    on the object is possible only within about 15 cm of the eye (the rays' ring sits 12 degrees
    out), so the bundle is only reported beside it (`ray_share`: the mean share of the nine rays
    on the object). The mobile's point is its hanging parts (Track.target_point); before v6 it
    was the ceiling mount it hangs from."""
    tr = ctx.tr
    out = []
    half_fovea = 0.5 * float(tr.header.get("fovea_fovy_deg", 24.0))
    nmin = int(round(G04_MIN_S / STEP_S))
    for k, o in enumerate(tr.ent.objects):
        if o["kind"] in ("decoration",) or not (tr.object_flags[:, k] & 1).any():
            continue
        speed = tr.object_body_speed(k) if o["kind"] == "mobile" else tr.obj_speed[:, k]
        moving = (np.nan_to_num(speed) >= G04_MOVING_MPS) & ((tr.object_flags[:, k] & 1) > 0)
        if not moving.any():
            continue
        err = tr.gaze_error_deg(int(o["entity"]), 0)
        share = tr.rays_on(int(o["entity"]), 0) / float(tr.ray_ent.shape[2])
        for i0, i1 in _intervals(moving, tr.continuous):
            if i1 - i0 + 1 < nmin:
                continue
            on = err[i0:i1 + 1] <= half_fovea
            frac = float(np.mean(on))
            if frac < G04_ON_FRACTION:
                continue
            # tracking is the gaze MOVING with the object: a still eye that a slow object stays
            # inside is no tracking (a limp body "tracked" 24 times in 20,000 steps, 2026-09-24)
            f0, f1 = tr.eye_fwd[i0, 0], tr.eye_fwd[i1, 0]
            turned = float(np.degrees(np.arccos(np.clip(np.dot(f0, f1), -1, 1)))) if np.isfinite(f0).all() and np.isfinite(f1).all() else 0.0
            if turned < G04_GAZE_TURN_DEG:
                continue
            out.append(ctx.det("G04", i0, i1, sub=o["kind"], key=o["kind"],
                               measures=dict(mean_error_deg=float(np.nanmean(err[i0:i1 + 1])), on_target_fraction=frac,
                                             object_speed_mps=float(np.nanmean(speed[i0:i1 + 1])),
                                             duration_s=tr.seconds(i0, i1),
                                             ray_share=float(np.mean(share[i0:i1 + 1]))),
                               confidence=_conf(_over(frac, G04_ON_FRACTION)), objects=[o["name"]]))
    return out


def G05(ctx: Context) -> List[dict]:
    """Eye-head coordination: during one left-eye fixation of at least 0.45 s (the foveal bundle
    held on the target the whole time, primitives.fixations), the head turns at least
    G05_HEAD_DEG and the eye's eccentricity in its orbit falls at least G05_ECC_DROP_DEG."""
    tr = ctx.tr
    out = []
    ecc = np.linalg.norm(tr.eye_angles[:, 0], axis=-1)
    for f in ctx.by_type["fixation"]:
        if f["eye"] != 0 or not _salient(tr, f["entity"]) or f["duration_s"] < 0.45:
            continue
        i0, i1 = f["i0"], f["i1"]
        head = float(np.nansum(tr.head_ang_speed[i0 + 1:i1 + 1] * tr.dt[i0 + 1:i1 + 1]))
        drop = float(ecc[i0] - np.nanmin(ecc[i0:i1 + 1]))
        if head < G05_HEAD_DEG or drop < G05_ECC_DROP_DEG:
            continue
        out.append(ctx.det("G05", i0, i1, sub=f["target_class"], key=f["target_class"],
                           measures=dict(head_turn_deg=head, eccentricity_drop_deg=drop,
                                         fixation_s=f["duration_s"] + STEP_S, **_fixation_measures(f)),
                           confidence=_conf(_over(head, G05_HEAD_DEG), _over(drop, G05_ECC_DROP_DEG)),
                           objects=[f["target_label"]]))
    return out


# ====================================================================== reach and grasp
def R01(ctx: Context) -> List[dict]:
    """A hand closing on an object. One arm movement shortens the hand's distance to several of
    the crib's six toys at once; it is a move TOWARD the one it closes on most (the largest
    share of its starting distance, then the nearest at the end), so overlapping approaches by
    the same hand keep only that object (before 2026-09-24 each toy counted: 7.8 per 100
    production steps)."""
    tr = ctx.tr
    cands = []
    for a in ctx.by_type["approach"]:
        o = tr.ent.objects[a["k"]]
        if o["kind"] in ("caregiver", "decoration") or a["start_m"] > REACH_MAX_START_M:
            continue
        frac = a["closed_m"] / max(1e-9, a["start_m"])
        if a["closed_m"] < REACH_MIN_CLOSED_M and frac < REACH_MIN_FRACTION:
            continue
        # the hand must do at least half the closing: its end measured to where the object was
        # at the start (a toy carried toward a still hand is the world's approach, not a reach)
        si = SIDES.index(a["side"])
        own = a["start_m"] - float(np.linalg.norm(tr.palm[a["i1"], si] - tr.obj_pos[a["i0"], a["k"]]))
        if not own >= REACH_OWN_SHARE * a["closed_m"]:
            continue
        a = dict(a, own_closed_m=own)
        cands.append((a, o, frac))
    kept = []
    for a, o, frac in sorted(cands, key=lambda c: (-c[2], c[0]["end_m"])):
        if any(b["side"] == a["side"] and b["i0"] <= a["i1"] and a["i0"] <= b["i1"] for b, _, _ in kept):
            continue
        kept.append((a, o, frac))
    out = []
    for a, o, frac in sorted(kept, key=lambda c: c[0]["i0"]):
        out.append(ctx.det("R01", a["i0"], a["i1"], sub=a["side"], key="%s:%s" % (a["side"], o["kind"]),
                           measures=dict(closed_cm=100.0 * a["closed_m"], closed_fraction=frac,
                                         end_distance_cm=100.0 * a["end_m"], path_efficiency=a["path_efficiency"],
                                         hand_closed_cm=100.0 * a["own_closed_m"]),
                           confidence=_conf(max(_over(a["closed_m"], REACH_MIN_CLOSED_M), _over(frac, REACH_MIN_FRACTION))),
                           objects=[o["name"]], parts=["hand_" + a["side"]], k=a["k"]))
    return out


def R02(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    approaches = ctx.by_type["approach"]
    for si, side in enumerate(SIDES):
        for c in _hand_onsets(ctx, side):
            k = tr.ent.object_of_entity.get(c["other"])
            if k is None or not _object_ok(tr.ent.objects[k]):
                continue
            o = tr.ent.objects[k]
            # the hand made the contact: it closed on the object itself just before (at least half
            # of the closing its own movement, measured to where the object was). Without this a
            # limp body "repeated contact" 262 times in 20,000 steps (2026-09-24 baseline)
            pre = []
            for a in approaches:
                # the touch came during the approach or within 200 ms after it ended
                if a["k"] == k and a["side"] == side and a["i0"] < c["i0"] <= a["i1"] + 4:
                    # Later movement cannot explain an earlier touch. The record resolves
                    # contact to a 50 ms burst; never borrow evidence from a later burst.
                    end = min(a["i1"], c["i0"])
                    closed = a["start_m"] - float(np.linalg.norm(tr.palm[end, si] - tr.obj_pos[end, k]))
                    own = a["start_m"] - float(np.linalg.norm(tr.palm[end, si] - tr.obj_pos[a["i0"], k]))
                    if closed >= P.APPROACH_MIN_M and own >= REACH_OWN_SHARE * closed:
                        pre.append(a)
            if not pre:
                continue
            ttc = tr.seconds(pre[-1]["i0"], c["i0"])
            # the act begins at the approach; its touch starts at the contact (the evidence join
            # judges the touch prediction there: evidence.Joiner._onset_index)
            out.append(ctx.det("R02", pre[-1]["i0"], max(c["i1"], pre[-1]["i0"]), sub=o["kind"], key=o["name"],
                               measures=dict(time_to_contact_s=ttc, force_n=c["peak_force_n"],
                                             after_approach=True),
                               confidence=1.0, objects=[o["name"]], parts=["hand_" + side], k=k,
                               onset_index=c["i0"]))
    return out


def R03(ctx: Context, r01: List[dict]) -> List[dict]:
    """Visually guided reach: a reach (R01) whose object an eye fixated - its foveal bundle on it,
    the centre ray's thing (detector v7) - within GAZE_LEAD_S before the reach began or during it."""
    tr = ctx.tr
    out = []
    lead = int(round(GAZE_LEAD_S / STEP_S))
    for r in r01:
        e = int(tr.ent.objects[r["k"]]["entity"])
        fx = [f for f in ctx.by_type["fixation"] if f["entity"] == e and f["i1"] >= r["i0"] - lead and f["i0"] <= r["i1"]]
        if not fx:
            continue
        f = fx[0]
        out.append(ctx.det("R03", min(f["i0"], r["i0"]), r["i1"], sub=r["sub"], key=r["key"],
                           measures=dict(gaze_lead_s=tr.seconds(f["i0"], r["i0"]) if f["i0"] <= r["i0"] else -tr.seconds(r["i0"], f["i0"]),
                                         closed_cm=r["measures"]["closed_cm"], path_efficiency=r["measures"]["path_efficiency"],
                                         ray_share=f.get("ray_share"), aim_error_deg=f.get("error_deg"),
                                         inner_share=f.get("inner_share")),
                           confidence=r["confidence"], objects=r["objects"], parts=r["parts"]))
    return out


def R04(ctx: Context) -> List[dict]:
    """Opposing grasp (v9, physics): two surfaces of one hand - the thumb and the fingers, the thumb and
    the palm, or the fingers and the palm (a palmar grasp) - touch a free object in the same frames, and
    in at least one record of that contact the hand GRIPS it (_grip): the two pushes oppose (at least
    OPPOSITION_MIN_DEG apart, one exclusion among the rest) and balance, the object is retained against
    gravity by the hand (everything else pushes it up with less than half its weight: its support does
    not come mainly from the chest, a forearm or the environment), and it moves with the palm.
    Measures: the widest angle of a gripping record and the widest per surface pair over the contact,
    the share of the contact's records that grip, and at the widest gripping record the share of the
    weight others carried, how much of what can take the pushes' difference it used (1 = at the line),
    the slip and the weaker push over the stronger. Needs the recorder's contact normals
    (observer_world_v2): a track without them measures no R04. The world's mocap toys are held by the
    world, not the hand, and are left out (a pinch of the toy's pad is to be a grasp subtype of its own,
    the coordinator's decision of the evening of 2026-09-25)."""
    tr = ctx.tr
    out = []
    if not getattr(tr, "has_normals", False):
        return out
    names = ["%s_%s_deg" % pair for pair in OPPOSING_SURFACES]
    for k, o in enumerate(tr.ent.objects):
        if not _object_ok(o) or not o.get("free"):
            continue
        for si, side in enumerate(SIDES):
            g = _grip(ctx, side, k)
            if not g["physics"] or not g["grip"].any():
                continue
            for i0, i1 in _intervals(g["together"], tr.continuous):
                grip = g["grip"][i0:i1 + 1]
                if not grip.any():
                    continue
                ang = np.where(grip, g["angle"][i0:i1 + 1], -np.inf)
                j = i0 + int(np.argmax(ang))                   # the widest gripping record
                use = np.where((g["pair_deg"][j] >= OPPOSITION_MIN_DEG) & (g["pair_use"][j] <= 1.0),
                               g["pair_deg"][j], -np.inf)
                p = int(np.argmax(use))                        # its gripping pair
                fa, fb = g["pair_force"][j, p]
                seg = g["angle"][i0:i1 + 1]
                fin = np.isfinite(seg)
                per_pair = {nm: (float(np.nanmax(g["pair_deg"][i0:i1 + 1, q]))
                                 if np.isfinite(g["pair_deg"][i0:i1 + 1, q]).any() else float("nan"))
                            for q, nm in enumerate(names)}
                best = float(g["angle"][j])
                others = float(g["others_up"][j] / max(1e-12, g["weight_n"]))
                used = float(g["pair_use"][j, p])
                slip = float(g["slip_mps"][j])
                out.append(ctx.det("R04", i0, i1, sub="%s:%s" % (side, o["kind"]), key="%s:%s" % (side, o["kind"]),
                                   measures=dict(opposition_deg=best,
                                                 opposed_share=float(np.mean(seg[fin] >= OPPOSITION_MIN_DEG)),
                                                 grip_share=float(np.mean(grip)),
                                                 others_weight_share=others, balance_used=used, slip_mps=slip,
                                                 force_balance=float(min(fa, fb) / max(1e-9, max(fa, fb))),
                                                 duration_s=tr.seconds(i0, i1) + STEP_S, **per_pair),
                                   confidence=_conf(_over(best, OPPOSITION_MIN_DEG), _under(used, 1.0),
                                                    _under(others, GRIP_OTHERS_MAX_WEIGHT),
                                                    _under(slip, GRIP_SLIP_MPS)),
                                   objects=[o["name"]],
                                   parts=(["finger_%s_thumb" % side] if "thumb" in OPPOSING_SURFACES[p] else [])
                                   + ["hand_" + side],
                                   grip_pair="-".join(OPPOSING_SURFACES[p]),
                                   scripted=_toy_scripted(tr, k, si, i0, i1)))
    return out


def R05(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for l in ctx.by_type["lift"]:
        out.append(ctx.det("R05", l["i0"], l["i1"], sub=l["object"], key=l["object"],
                           measures=dict(height_cm=100.0 * l["height_m"], retained_s=l["duration_s"] + STEP_S),
                           confidence=_conf(_over(l["height_m"], P.LIFT_MIN_M)), objects=[l["object"]],
                           parts=["hand_" + h for h in l["hands"].split("+") if h]))
    return out


def R06(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for h in _holds(ctx):
        if not h["held"] or not tr.ent.objects[h["k"]]["free"]:
            continue
        p = tr.palm[h["i0"]:h["i1"] + 1, h["si"]]
        path = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=-1))) if len(p) > 1 else 0.0
        if path < CARRY_MIN_M:
            continue
        out.append(ctx.det("R06", h["i0"], h["i1"], sub=h["side"], key="%s:%s" % (h["side"], tr.ent.objects[h["k"]]["kind"]),
                           measures=dict(path_cm=100.0 * path, slip_cm=100.0 * h["drift"], retained_s=tr.seconds(h["i0"], h["i1"])),
                           confidence=_conf(_over(path, CARRY_MIN_M)), objects=[tr.ent.objects[h["k"]]["name"]],
                           parts=["hand_" + h["side"]]))
    return out


def R07(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for h in _holds(ctx):
        if not h["held"]:
            continue
        for s0, s1 in _slips(ctx, h):
            if s1 is None or tr.seconds(s0, s1) > 0.5:
                continue
            out.append(ctx.det("R07", s0, s1, sub=h["side"], key=h["side"],
                               measures=dict(recovery_ms=1000.0 * tr.seconds(s0, s1),
                                             peak_slip_mps=float(np.nanmax(_rel_speed(tr, h["si"], h["k"])[s0:s1 + 1]))),
                               confidence=1.0, objects=[tr.ent.objects[h["k"]]["name"]], parts=["hand_" + h["side"]]))
    return out


def _releases(ctx: Context):
    """(side index, object index, last contact index) for every end of hand contact with a
    free object, with the object's support at that moment."""
    tr = ctx.tr
    for k, o in enumerate(tr.ent.objects):
        if not o["free"]:
            continue
        for si, side in enumerate(SIDES):
            on = ctx.hand_obj(side, k)["on"]
            for i0, i1 in _intervals(on, tr.continuous):
                if i1 >= tr.n - 2:
                    continue
                yield si, side, k, i0, i1


def R08(ctx: Context) -> List[dict]:
    """Placed release: a lifted free object whose hand contact ends while it rests on the floor or
    another object, at or under PLACE_MAX_MPS both in the LAST CONTACT RECORD (v8) and in the record
    after, then stays still PLACE_STABLE_S. Before v8 only the record after was read: a box that
    slipped out of the hand and fell 3 cm at 0.6 m/s within one 50 ms record had landed by then
    (0.0 m/s) and read as placed (scripted case grasp_lift_no_retention, 2026-09-24; defect 3)."""
    tr = ctx.tr
    out = []
    w = int(round(PLACE_STABLE_S / STEP_S))
    lifts = ctx.by_type["lift"]
    for si, side, k, i0, i1 in _releases(ctx):
        j = i1 + 1
        # "released after it gains stable support": the hand carried it off its support first
        if not any(l["k"] == k and l["i0"] <= i1 and l["i1"] >= i0 for l in lifts):
            continue
        floor = ctx.obj_floor(k)["on"][i1:j + 1].any()
        onobj = ctx.obj_other_objects(k)["on"][i1:j + 1].any()
        v = float(np.nan_to_num(tr.obj_speed[j, k], nan=1.0))
        v_last = float(np.nan_to_num(tr.obj_speed[i1, k], nan=1.0))
        if not (floor or onobj) or v > PLACE_MAX_MPS or v_last > PLACE_MAX_MPS:
            continue
        after = tr.obj_speed[j:min(tr.n, j + w), k]
        if not tr.continuous[j:min(tr.n, j + w)].all() or np.nanmax(after) > PLACE_STABLE_MPS:
            continue
        o = tr.ent.objects[k]
        disp = float(np.linalg.norm(tr.obj_pos[min(tr.n - 1, j + w - 1), k] - tr.obj_pos[j, k]))
        out.append(ctx.det("R08", i0, min(tr.n - 1, j + w - 1), sub=side, key="%s:%s" % (side, "object" if onobj else "floor"),
                           measures=dict(speed_at_release_mps=v, speed_in_last_contact_mps=v_last,
                                         settle_displacement_cm=100.0 * disp, held_s=tr.seconds(i0, i1)),
                           confidence=_conf(_under(max(v, v_last), PLACE_MAX_MPS)), objects=[o["name"]],
                           parts=["hand_" + side],
                           onto="object" if onobj else "floor"))
    return out


# ====================================================================== posture and locomotion
def P01(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    w = int(round(P01_WINDOW_S / STEP_S))
    # windows on absolute steps (multiples of w), so a window never shifts with where the
    # digester's window began (a shifted grid re-emitted overlapping acts)
    i = 1
    while i < tr.n and int(tr.step[i]) % w:
        i += 1
    while i + w <= tr.n:
        seg = slice(i, i + w)
        if not tr.continuous[seg].all():
            i += w
            continue
        palm = float(np.nanmean(tr.palm_speed[seg]))
        head = float(np.sqrt(np.nanmean(tr.head_ang_speed[seg] ** 2)))
        trunk = float(np.sqrt(np.nanmean(tr.trunk_ang_speed[seg] ** 2)))
        moving = palm >= P01_MOVING_PALM_MPS
        if moving and trunk > 1.0:
            ratio = head / trunk
            if ratio <= P01_RATIO:
                out.append(ctx.det("P01", i, i + w - 1, key="head",
                                   measures=dict(head_rms_degps=head, trunk_rms_degps=trunk, ratio=ratio, palm_speed_mps=palm),
                                   confidence=_conf(_under(ratio, P01_RATIO), _over(palm, P01_MOVING_PALM_MPS)),
                                   parts=["head"]))
        i += w
    return out


def _external_push(ctx: Context, i0: int, i1: int) -> bool:
    """Something outside the infant moved it: the adult touched it, or a scripted or mounted
    object (the toy the script carries, the mobile) touched it while moving. The infant's own
    limb striking the floor is not outside help: before 2026-09-24 any impact counted, and the
    impact line marked a third of production frames, so no roll could ever count."""
    tr = ctx.tr
    if tr.caregiver_contact[max(0, i0 - 2):i1 + 1].any():
        return True
    for k, o in enumerate(tr.ent.objects):
        if o["free"] or o["kind"] == "caregiver":
            continue
        touched = ctx.pair(("selfobj", k), tr.ent.is_self, tr.object_mask(k))["on"][max(0, i0 - 2):i1 + 1]
        if not touched.any():
            continue
        speed = tr.object_body_speed(k) if o["kind"] == "mobile" else tr.obj_speed[:, k]
        if np.nan_to_num(speed[max(0, i0 - 2):i1 + 1]).max() >= P.OBJ_MOVE_MPS:
            return True
    return False


# A ROLL IS DRIVEN BY THE BODY'S OWN MOTORS (detector v9; the independent review of 2026-09-25: P02's scripted
# positive was a limp body turned by a torque from outside, and P02 counted it). The catalog's rolls are the
# infant's own: P02 "without external push", P03 "through self-generated motion", P04 "through a self-generated
# sequence". _external_push catches the caregiver and the world's mounted objects; a roll that nothing of the
# body drove (a limp body, or one turned by any other outside force) passed. As for the hand rows (MOTOR_ON,
# _motor_work), a roll now counts only when the body's motors carried a command during it and their work over
# it was positive: every motor's recorded command x gear x gain x its joint's turn, summed over the roll and
# over all the body's motors (_body_drove). The whole body is summed because a roll is made by the trunk, the
# legs, the arms and the head together. The cost, named: a roll whose motors on balance resisted it (a body
# tipped over by gravity from a side it was holding) no longer counts.
def _body_drove(tr: Track, i0: int, i1: int):
    """(largest |command|, work) of all the body's motors over records i0..i1 (_command_peak, _motor_work)."""
    adrs = sorted(_motor_of_qpos(tr))
    return _command_peak(tr, adrs, i0, i1), _motor_work(tr, adrs, i0, i1)


def _postures(ctx: Context, cap: str, frm: str, to: str) -> List[dict]:
    tr = ctx.tr
    out = []
    for p in ctx.by_type["posture"]:
        if p["frm"] != frm or p["to"] != to:
            continue
        if _external_push(ctx, p["i0"], p["i1"]):
            continue
        peak, work = _body_drove(tr, p["i0"], p["i1"])
        if _passive(peak) or _against(work):
            continue
        out.append(ctx.det(cap, p["i0"], p["i1"], sub=p["side_down"], key="%s-down" % p["side_down"],
                           measures=dict(roll_change_deg=p["roll_change_deg"], duration_s=p["duration_s"],
                                         command_peak=peak, motor_work=work),
                           confidence=1.0, parts=["trunk"]))
    return out


def P02(ctx):
    return _postures(ctx, "P02", "back", "side")


def P03(ctx):
    return _postures(ctx, "P03", "side", "back")


def P04(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    ps = ctx.by_type["posture"]
    for idx, p in enumerate(ps):
        if p["to"] != "belly":
            continue
        start = None
        for q in reversed(ps[:idx + 1]):
            if q["frm"] == "back" and tr.seconds(q["i0"], p["i1"]) <= 5.0:
                start = q
                break
        if start is None or _external_push(ctx, start["i0"], p["i1"]):
            continue
        peak, work = _body_drove(tr, start["i0"], p["i1"])
        if _passive(peak) or _against(work):
            continue
        out.append(ctx.det("P04", start["i0"], p["i1"], sub=p["side_down"], key="%s-down" % p["side_down"],
                           measures=dict(duration_s=tr.seconds(start["i0"], p["i1"]), command_peak=peak, motor_work=work),
                           confidence=1.0, parts=["trunk"]))
    return out


def P05(ctx: Context) -> List[dict]:
    """Push against support (catalog: hand or foot support force rises with limb extension and produces
    body displacement): over a 0.5 s window, the limb group's floor force RISES by PUSH_FORCE_FRACTION of
    the body's weight and holds (PUSH_HOLD_RECORDS, v9: the least of the window's last 3 records against
    the most of the 3 ending at its start), the elbow or knee extends PUSH_EXTEND_DEG and the centre of
    mass moves PUSH_DISPLACE_M."""
    tr = ctx.tr
    out = []
    wgt = float(tr.header.get("weight_n", 30.0))
    w = int(round(0.5 / STEP_S))
    k = PUSH_HOLD_RECORDS
    for side in SIDES:
        for limb, joint, parts in (("arm", "%s_elbow", ("hand_" + side, "arm_" + side)),
                                   ("leg", "%s_knee", ("leg_" + side,))):
            f = sum(ctx.sup[p] for p in parts if p in ctx.sup)
            ang = tr.limb.get(joint % ("left" if side == "L" else "right"))
            if ang is None:
                continue
            i = k - 1
            while i + w < tr.n:
                if not tr.continuous[i - k + 2:i + w + 1].all():
                    i += 1
                    continue
                rise = float(np.min(f[i + w - k + 1:i + w + 1]) - np.max(f[i - k + 1:i + 1]))
                extend = float(abs(ang[i]) - abs(ang[i + w]))        # toward straight
                disp = float(np.linalg.norm(tr.com[i + w] - tr.com[i]))
                if rise >= PUSH_FORCE_FRACTION * wgt and extend >= PUSH_EXTEND_DEG and disp >= PUSH_DISPLACE_M:
                    out.append(ctx.det("P05", i, i + w, sub="%s:%s" % (side, limb), key="%s:%s" % (side, limb),
                                       measures=dict(force_rise_bodyweights=rise / wgt, extension_deg=extend,
                                                     displacement_cm=100.0 * disp),
                                       confidence=_conf(_over(rise, PUSH_FORCE_FRACTION * wgt), _over(extend, PUSH_EXTEND_DEG),
                                                        _over(disp, PUSH_DISPLACE_M)),
                                       parts=list(parts)))
                    i += w
                else:
                    i += 1
    return out


def _median5(x: np.ndarray) -> np.ndarray:
    """Each value replaced by the median of the five records around it (NaN where none)."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - 2):i + 3]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = float(np.median(w))
    return out


def P06(ctx: Context) -> List[dict]:
    """Weight shift: the centre of mass moves over the floor (a support_shift primitive: 2 cm
    within 1 s, half of it kept for 250 ms) while the limbs' left-right share of the floor
    support moves the same way by WEIGHT_SHARE_CHANGE. The share is a five-record median, so one
    step of contact flicker cannot make it; the mass moving AWAY from the side that took the
    weight is no controlled shift and is not counted."""
    tr = ctx.tr
    out = []
    left = ctx.sup["hand_L"] + ctx.sup["arm_L"] + ctx.sup["leg_L"]
    right = ctx.sup["hand_R"] + ctx.sup["arm_R"] + ctx.sup["leg_R"]
    tot = np.maximum(ctx.sup["total"], 1e-9)
    share = _median5((left - right) / tot)
    for s in ctx.by_type["support_shift"]:
        ch = float(share[s["i1"]] - share[s["i0"]])
        if not np.isfinite(ch) or abs(ch) < WEIGHT_SHARE_CHANGE:
            continue
        lat = s.get("lateral_m", float("nan"))
        if np.isfinite(lat) and lat * ch < 0:
            continue
        out.append(ctx.det("P06", s["i0"], s["i1"], sub="toward_%s" % ("L" if ch > 0 else "R"),
                           key="toward_%s" % ("L" if ch > 0 else "R"),
                           measures=dict(shift_cm=100.0 * s["shift_m"], share_change=ch,
                                         sideways_cm=100.0 * lat if np.isfinite(lat) else float("nan")),
                           confidence=_conf(_over(abs(ch), WEIGHT_SHARE_CHANGE), _over(s["shift_m"], P.COM_SHIFT_M)),
                           parts=["trunk"]))
    return out


def P07(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    lying = np.isin(ctx.posture, ["back", "belly"])
    if not lying.any():
        return out
    base = float(np.nanmedian(tr.trunk_pos[lying, 2]))
    limbs = sum(ctx.sup[k] for k in ("hand_L", "hand_R", "arm_L", "arm_R", "leg_L", "leg_R"))
    share = limbs / np.maximum(ctx.sup["total"], 1e-9)
    up = (tr.trunk_pos[:, 2] - base >= TRUNK_LIFT_M) & (share >= LIMB_SUPPORT_SHARE)
    nmin = int(round(TRUNK_LIFT_MIN_S / STEP_S))
    for i0, i1 in _intervals(up, tr.continuous):
        if i1 - i0 + 1 < nmin or tr.caregiver_contact[i0:i1 + 1].any():
            continue
        h = float(np.nanmax(tr.trunk_pos[i0:i1 + 1, 2]) - base)
        out.append(ctx.det("P07", i0, i1, key=str(ctx.posture[i0]),
                           measures=dict(height_cm=100.0 * h, duration_s=tr.seconds(i0, i1) + STEP_S,
                                         limb_share=float(np.nanmean(share[i0:i1 + 1]))),
                           confidence=_conf(_over(h, TRUNK_LIFT_M)), parts=["trunk"]))
    return out


def P08(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    up = ctx.posture == "upright"
    nmin = int(round(SIT_MIN_S / STEP_S))
    for i0, i1 in _intervals(up, tr.continuous):
        if i1 - i0 + 1 < nmin or tr.caregiver_contact[i0:i1 + 1].any():
            continue
        out.append(ctx.det("P08", i0, i1, key="sit",
                           measures=dict(duration_s=tr.seconds(i0, i1) + STEP_S,
                                         max_tilt_deg=float(np.nanmax(tr.sit_deg[i0:i1 + 1]))),
                           confidence=_conf(_over(tr.seconds(i0, i1) + STEP_S, SIT_MIN_S)), parts=["trunk"]))
    return out


def _translations(ctx: Context):
    tr = ctx.tr
    wmax = int(round(TRANSLATE_MAX_S / STEP_S))
    i = 0
    while i < tr.n - 1 and int(tr.step[i]) % 10:        # starts on absolute steps, as P01
        i += 1
    while i < tr.n - 1:
        best = None
        for j in range(i + 1, min(tr.n, i + wmax + 1)):
            if not tr.continuous[j]:
                break
            net = float(np.linalg.norm(tr.com[j, :2] - tr.com[i, :2]))
            if net >= TRANSLATE_M:
                path = float(np.sum(np.linalg.norm(np.diff(tr.com[i:j + 1, :2], axis=0), axis=-1)))
                straight = net / max(path, 1e-9)
                if straight >= TRANSLATE_STRAIGHT:
                    best = (j, net, straight)
                    break
        if best:
            yield i, best[0], best[1], best[2]
            i = best[0] + 1
        else:
            i += 10


def P09(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for i0, i1, net, straight in _translations(ctx):
        if tr.caregiver_contact[i0:i1 + 1].any():
            continue
        d = tr.com[i1, :2] - tr.com[i0, :2]
        out.append(ctx.det("P09", i0, i1, key="translate",
                           measures=dict(distance_cm=100.0 * net, straightness=straight,
                                         speed_cmps=100.0 * net / max(1e-9, tr.seconds(i0, i1))),
                           confidence=_conf(_over(net, TRANSLATE_M), _over(straight, TRANSLATE_STRAIGHT)),
                           parts=["trunk"], heading_deg=float(math.degrees(math.atan2(d[1], d[0])))))
    return out


def P10(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    limbs = {"hand_L": tr.ent.hand["L"] | tr.ent.arm["L"], "hand_R": tr.ent.hand["R"] | tr.ent.arm["R"],
             "leg_L": tr.ent.leg["L"], "leg_R": tr.ent.leg["R"]}
    contact = {k: ctx.pair(("support", k), m, tr.ent.floor)["on"] for k, m in limbs.items()}
    for i0, i1, net, straight in _translations(ctx):
        if net < CRAWL_MIN_M:
            continue
        lifts = {}
        for k, on in contact.items():
            seg = on[i0:i1 + 1].astype(int)
            lifts[k] = int(np.sum(np.diff(seg) == -1))       # a limb leaving the floor, then (maybe) back
        alt = min(lifts["hand_L"] + lifts["leg_R"], lifts["hand_R"] + lifts["leg_L"])
        cycles = min(sum(lifts.values()) // 2, alt)
        if cycles < CRAWL_MIN_CYCLES:
            continue
        out.append(ctx.det("P10", i0, i1, key="crawl",
                           measures=dict(cycles=cycles, distance_cm=100.0 * net,
                                         speed_cmps=100.0 * net / max(1e-9, tr.seconds(i0, i1))),
                           confidence=_conf(_over(cycles, CRAWL_MIN_CYCLES)), parts=["trunk", "hand_L", "hand_R", "leg_L", "leg_R"]))
    return out


def P11(ctx):
    return []          # not observable here: the crib has no platform (catalog.py)


def P12(ctx):
    return []          # not observable here: no platform edge to descend from


# ====================================================================== object play
def _hand_contact_any(ctx: Context, k: int) -> np.ndarray:
    return ctx.hand_obj("L", k)["on"] | ctx.hand_obj("R", k)["on"]


def O01(ctx: Context) -> List[dict]:
    """Push: an object starts moving while a hand touches it, within PUSH_DIR_MAX_DEG of the
    palm's direction. The palm's direction is its velocity in the record before the object moved
    (a hand striking or pushing on arrival); when the palm was still then (under
    primitives.OBJ_MOVE_MPS), in the object's first moving record (v8): a hand resting on an
    object and pushing it from there starts with it. Before v8 that push was skipped (scripted case
    grasp_push_resting_hand, 2026-09-24; defect 5)."""
    tr = ctx.tr
    out = []
    for m in ctx.by_type["object_motion"]:
        k = m["k"]
        i0 = m["i0"]
        touch = _hand_contact_any(ctx, k)[max(0, i0 - 2):i0 + 1].any()
        if not touch:
            continue
        side = 0 if ctx.hand_obj("L", k)["on"][max(0, i0 - 2):i0 + 1].any() else 1
        pv = tr.palm_vel[max(0, i0 - 1), side]
        if not (np.isfinite(pv).all() and np.linalg.norm(pv) >= P.OBJ_MOVE_MPS):
            pv = tr.palm_vel[i0, side]
        d = np.asarray(m["direction"])
        if not (np.isfinite(pv).all() and np.linalg.norm(pv) > 1e-6 and np.linalg.norm(d) > 1e-6):
            continue
        ang = float(np.degrees(np.arccos(np.clip(np.dot(pv, d) / (np.linalg.norm(pv) * np.linalg.norm(d)), -1, 1))))
        if ang > PUSH_DIR_MAX_DEG:
            continue
        out.append(ctx.det("O01", max(0, i0 - 2), m["i1"], sub=m["object"], key=tr.ent.objects[k]["kind"],
                           measures=dict(displacement_cm=100.0 * m["displacement_m"], direction_error_deg=ang),
                           confidence=_conf(_under(ang, PUSH_DIR_MAX_DEG)), objects=[m["object"]], parts=["hand_" + SIDES[side]]))
    return out


def O02(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    for k, o in enumerate(tr.ent.objects):
        if not o["free"]:
            continue
        on = _hand_contact_any(ctx, k)
        d = np.linalg.norm(tr.obj_pos[:, k] - tr.trunk_pos, axis=-1)
        for i0, i1 in _intervals(on, tr.continuous):
            closer = float(d[i0] - d[i1])
            if closer < PULL_MIN_M:
                continue
            out.append(ctx.det("O02", i0, i1, sub=o["name"], key=o["kind"],
                               measures=dict(pulled_cm=100.0 * closer, contact_s=tr.seconds(i0, i1)),
                               confidence=_conf(_over(closer, PULL_MIN_M)), objects=[o["name"]]))
    return out


def O03(ctx: Context) -> List[dict]:
    out = []
    for d in ctx.by_type["drop"]:
        if not d["from_hand"]:
            continue
        out.append(ctx.det("O03", d["i0"], d["i1"], sub=d["object"], key=d["object"],
                           measures=dict(fall_cm=100.0 * d["fall_m"]), confidence=1.0, objects=[d["object"]]))
    return out


def O04(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    w = int(round(BANG_WINDOW_S / STEP_S))
    for k, o in enumerate(tr.ent.objects):
        if not o["free"]:
            continue
        held = _hand_contact_any(ctx, k)
        hits = []
        for c in ctx.by_type["contact"]:
            if c["open_start"] or c["peak_force_n"] < BANG_MIN_FORCE_N:
                continue
            ent = int(o["entity"])
            if ent not in (c["a"], c["b"]):
                continue
            other = c["b"] if c["a"] == ent else c["a"]
            if tr.ent.is_self[other] or not held[c["i0"]]:
                continue
            hits.append(c)
        hits.sort(key=lambda c: c["i0"])
        j = 0
        while j < len(hits):
            grp = [hits[j]]
            while j + len(grp) < len(hits) and hits[j + len(grp)]["i0"] - grp[0]["i0"] <= w:
                grp.append(hits[j + len(grp)])
            if len(grp) >= BANG_MIN_IMPACTS:
                gaps = np.diff([g["t0"] for g in grp])
                out.append(ctx.det("O04", grp[0]["i0"], grp[-1]["i1"], sub=o["name"], key=o["kind"],
                                   measures=dict(impacts=len(grp), interval_cv=float(np.std(gaps) / max(1e-9, np.mean(gaps))) if gaps.size else 0.0,
                                                 peak_force_n=float(max(g["peak_force_n"] for g in grp))),
                                   confidence=_conf(_over(len(grp), BANG_MIN_IMPACTS)), objects=[o["name"]]))
                j += len(grp)
            else:
                j += 1
    return out


def O05(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    w = int(round(SHAKE_WINDOW_S / STEP_S))
    for h in _holds(ctx):
        k = h["k"]
        if not tr.ent.objects[k]["free"] or h["i1"] - h["i0"] < 4:
            continue
        v = tr.obj_vel[h["i0"]:h["i1"] + 1, k]
        if not np.isfinite(v).all():
            v = np.nan_to_num(v)
        axis = np.linalg.svd(v - v.mean(0), full_matrices=False)[2][0]
        x = (tr.obj_pos[h["i0"]:h["i1"] + 1, k] - tr.obj_pos[h["i0"], k]) @ axis
        turns = P._turns(x, np.ones(len(x), dtype=bool), SHAKE_AMP_M)
        for a in range(len(turns) - SHAKE_REVERSALS):
            grp = turns[a:a + SHAKE_REVERSALS + 1]
            if (grp[-1][1] - grp[0][0]) <= w:
                amp = float(np.mean([abs(t[2]) for t in grp]))
                i0, i1 = h["i0"] + grp[0][0], h["i0"] + grp[-1][1]
                out.append(ctx.det("O05", i0, i1, sub=tr.ent.objects[k]["name"], key=tr.ent.objects[k]["kind"],
                                   measures=dict(amplitude_cm=100.0 * amp, frequency_hz=len(grp) / 2.0 / max(1e-9, tr.seconds(i0, i1))),
                                   confidence=_conf(_over(amp, SHAKE_AMP_M)), objects=[tr.ent.objects[k]["name"]],
                                   parts=["hand_" + h["side"]]))
                break
    return out


def O06(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    nmin = int(round(SPIN_MIN_S / STEP_S))
    for k, o in enumerate(tr.ent.objects):
        if not (o["free"] or o["kind"] == "mobile"):
            continue
        spin = tr.object_spin(k)
        touch = _hand_contact_any(ctx, k)
        for i0, i1 in _intervals(np.nan_to_num(spin) >= SPIN_RADPS, tr.continuous):
            if i1 - i0 + 1 < nmin or not touch[max(0, i0 - 10):i0 + 1].any():
                continue
            out.append(ctx.det("O06", i0, i1, sub=o["name"], key=o["kind"],
                               measures=dict(spin_radps=float(np.nanmean(spin[i0:i1 + 1])), duration_s=tr.seconds(i0, i1) + STEP_S),
                               confidence=_conf(_over(float(np.nanmean(spin[i0:i1 + 1])), SPIN_RADPS)), objects=[o["name"]]))
    return out


def hands_touch_together(tr: Track, k: int) -> np.ndarray:
    """Records with overlapping recorded contact-frame ranges for the two hands.

    Per-record contact booleans alone also include disjoint touches in one burst.
    The legacy recorder retains first/last frames, so this does not resolve gaps
    inside one contact pair's range.
    """
    obj = tr.object_mask(k)
    left = _pair_rows(tr, tr.ent.hand["L"], obj)
    right = _pair_rows(tr, tr.ent.hand["R"], obj)
    return _pushes_together(left, right, tr.n)[0]


def O07(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    nmin = int(round(TRANSFER_MIN_S / STEP_S))
    for k, o in enumerate(tr.ent.objects):
        if not o["free"]:
            continue
        L, R = ctx.hand_obj("L", k)["on"], ctx.hand_obj("R", k)["on"]
        fl = ctx.obj_floor(k)["on"]
        together = hands_touch_together(tr, k)
        # Disjoint touches inside one burst are neither a single-hand hold nor
        # evidence that the two hands overlapped.
        state = np.where(together, 3, np.where(L & ~R, 1, np.where(R & ~L, 2, 0)))
        i = 0
        while i < tr.n:
            if state[i] not in (1, 2):
                i += 1
                continue
            a = state[i]
            j = i + 1                  # the run's first record belongs to it whatever came before
            while j < tr.n and state[j] == a and tr.continuous[j]:
                j += 1
            if j - i < nmin or j >= tr.n or state[j] != 3 or not tr.continuous[j]:
                i = j                  # j > i: every pass advances
                continue
            m = j
            while m < tr.n and state[m] == 3 and tr.continuous[m]:
                m += 1
            b = 3 - a
            e = m
            while e < tr.n and state[e] == b and tr.continuous[e]:
                e += 1
            if e - m >= nmin and not fl[i:e].any():
                out.append(ctx.det("O07", i, e - 1, sub=o["name"], key="%s>%s" % ("LR"[a - 1], "LR"[b - 1]),
                                   measures=dict(overlap_s=tr.seconds(j, m - 1) + STEP_S), confidence=1.0,
                                   objects=[o["name"]], parts=["hand_L", "hand_R"]))
            i = max(e, i + 1)
    return out


def O08(ctx: Context) -> List[dict]:
    tr = ctx.tr
    out = []
    nmin = int(round(TWO_HAND_MIN_S / STEP_S))
    for k, o in enumerate(tr.ent.objects):
        if not o["free"]:
            continue
        both = hands_touch_together(tr, k)
        for i0, i1 in _intervals(both, tr.continuous):
            if i1 - i0 + 1 < nmin:
                continue
            sL = float(np.nanmean(tr.palm_speed[i0:i1 + 1, 0]))
            sR = float(np.nanmean(tr.palm_speed[i0:i1 + 1, 1]))
            still, active = (0, 1) if sL < sR else (1, 0)
            s_still, s_act = min(sL, sR), max(sL, sR)
            turn = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(tr.obj_quat[i0, k], tr.obj_quat[i1, k])))))))
            if s_still > STILL_HAND_MPS or (s_act < ACTIVE_HAND_MPS and turn < 20.0):
                continue
            out.append(ctx.det("O08", i0, i1, sub=o["name"], key="%s-holds" % SIDES[still],
                               measures=dict(still_hand_mps=s_still, active_hand_mps=s_act, object_turn_deg=turn,
                                             duration_s=tr.seconds(i0, i1) + STEP_S),
                               confidence=_conf(_under(s_still, STILL_HAND_MPS)), objects=[o["name"]],
                               parts=["hand_L", "hand_R"]))
    return out


def O09(ctx):
    return []          # not observable here: no container in the crib


def O10(ctx):
    return []


def O11(ctx: Context, r08: List[dict]) -> List[dict]:
    tr = ctx.tr
    out = []
    w = int(round(STACK_STABLE_S / STEP_S))
    for r in r08:
        if r.get("onto") != "object":
            continue
        k = tr.ent.object_of_entity.get(next((int(o["entity"]) for o in tr.ent.objects if o["name"] == r["objects"][0]), -1))
        if k is None:
            continue
        j = r["i1"]
        on = ctx.obj_other_objects(k)["on"][j:min(tr.n, j + w)]
        if len(on) < w or not on.all():
            continue
        out.append(ctx.det("O11", r["i0"], min(tr.n - 1, j + w - 1), sub=r["objects"][0], key="stack",
                           measures=dict(stable_s=STACK_STABLE_S, speed_at_release_mps=r["measures"]["speed_at_release_mps"]),
                           confidence=1.0, objects=r["objects"]))
    return out


# ====================================================================== the finger rows beyond the catalog
def HX1(ctx: Context) -> List[dict]:
    """Spread and gather (the dashboard's 'Spread and gather' hand action), v8: the FAN of the
    fingers - the index finger's sideways base angle minus the little finger's, the sum of the three
    neighbour differences - changing by at least DIGIT_MIN_DEG, the turn's fastest stretch moving those
    DIGIT_MIN_DEG within HX1_MAX_RISE_S (v9, _line_s; v8 read the whole turn's 10-90% rise, so a slow
    lead-in dropped a fast spread and a slow creep carried a smaller flick past the line). On both hands a positive side angle moves each fingertip toward the thumb
    (measured on this hand, 0.68-0.77 cm per 10 degrees), so the fan grows as the fingers spread,
    and all four tilting one way together leaves it unchanged.

    Before v8 the measure was the sum of the four |side angles| with no time bound: a sweep of all
    four fingers toward the thumb read as a spread, and limp hands drifting 25-62 degrees over 6-70 s
    made 40 acts in the 20,000 body-steps of the limp baseline (2026-09-24; defect 2).

    v9: the two side joints' own motors drove the fan (MOTOR_ON). Under v8 each of the 8 limp bodies
    of the reset world made one fan of 30-32 degrees 4-6 s after the reset (rise 0.8-0.9 s, no
    contact anywhere on the hand): the limp fingers' side joints swung one after another to their
    25-degree range limits as the hand settled, with every motor command zero."""
    tr = ctx.tr
    out = []
    adr = {j["name"].split(":")[-1]: int(j["qpos"]) for j in tr.header["joints"]}
    for si, side in enumerate(SIDES):
        pre = "right_" if side == "R" else "left_"
        if pre + "ff_side" not in adr or pre + "lf_side" not in adr:
            continue
        fan = np.degrees(tr.qpos[:, adr[pre + "ff_side"]] - tr.qpos[:, adr[pre + "lf_side"]])
        for i0, i1, ch, settled in P.turns_confirmed(fan, tr.continuous, P.DIGIT_MIN_DEG):
            # the act's time: the fastest stretch that moves the fan the full line (_line_s), so neither
            # stillness nor a slow creep at either end of the turn is part of the movement; the whole
            # turn's 10-90% rise (v8's measure) is still reported
            rise = _rise_s(tr, fan, i0, i1, ch)
            line = _line_s(tr, fan, i0, i1, ch, P.DIGIT_MIN_DEG)          # the fastest 15 degrees (v9)
            if line > HX1_MAX_RISE_S:
                continue
            peak = _command_peak(tr, [adr[pre + "ff_side"], adr[pre + "lf_side"]], i0, i1)
            work = _motor_work(tr, [adr[pre + "ff_side"], adr[pre + "lf_side"]], i0, i1)
            if _passive(peak) or _against(work):
                continue
            out.append(ctx.det("HX1", i0, i1, sub=side, key="%s:%s" % (side, "spread" if ch > 0 else "gather"),
                               measures=dict(change_deg=ch, fan_deg=abs(ch), rise_s=rise, line_s=line, command_peak=peak,
                                             motor_work=work),
                               confidence=_conf(_over(abs(ch), P.DIGIT_MIN_DEG), _under(line, HX1_MAX_RISE_S)),
                               parts=["hand_" + side], open_end=bool(settled < 0)))
    return out


def HX2(ctx: Context) -> List[dict]:
    """Shape the palm: the little-finger metacarpal's cupping hinge (the dashboard's 'Shape the
    palm'), a change of at least 15 degrees whose fastest stretch moves 15 degrees within ACT_MAX_RISE_S
    (_line_s) and driven by the hinge's
    own motor (MOTOR_ON; both v9)."""
    tr = ctx.tr
    out = []
    for si, side in enumerate(SIDES):
        adrs = [int(j["qpos"]) for j in tr.header["joints"]
                if j["name"].split(":")[-1] == ("right_lf_meta" if side == "R" else "left_lf_meta")]
        if not adrs:
            continue
        x = np.degrees(tr.qpos[:, adrs[0]])
        for i0, i1, ch, settled in P.turns_confirmed(x, tr.continuous, 15.0):
            rise = _rise_s(tr, x, i0, i1, ch)
            line = _line_s(tr, x, i0, i1, ch, 15.0)
            if line > ACT_MAX_RISE_S:
                continue
            peak = _command_peak(tr, [adrs[0]], i0, i1)
            work = _motor_work(tr, [adrs[0]], i0, i1)
            if _passive(peak) or _against(work):
                continue
            out.append(ctx.det("HX2", i0, i1, sub=side, key="%s:%s" % (side, "cup" if ch > 0 else "flatten"),
                               measures=dict(change_deg=ch, amount_deg=abs(ch), rise_s=rise, line_s=line, command_peak=peak,
                                             motor_work=work),
                               confidence=_conf(_over(abs(ch), 15.0)),
                               parts=["hand_" + side], open_end=bool(settled < 0)))
    return out


def HX3(ctx: Context) -> List[dict]:
    """Touch with one fingertip: a single fingertip's contact onset with an object or surface
    while no other part of that hand touches it (the dashboard's 'Touch, press and point')."""
    tr = ctx.tr
    out = []
    for si, side in enumerate(SIDES):
        for digit in DIGITS:
            tip = tr.ent.fingertip[(side, digit)]
            for c in ctx.by_type["contact"]:
                if c["open_start"]:
                    continue
                a, b = c["a"], c["b"]
                if not (tip[a] or tip[b]):
                    continue
                other = b if tip[a] else a
                if tr.ent.is_self[other] and tr.ent.hand[side][other]:
                    continue
                i = c["i0"]
                rows = tr.contacts_at(i)
                others = 0
                for j in rows:
                    e1, e2 = int(tr.c_e[j, 0]), int(tr.c_e[j, 1])
                    if other in (e1, e2):
                        part = e2 if e1 == other else e1
                        if tr.ent.hand[side][part] and not tip[part]:
                            others += 1
                if others:
                    continue
                out.append(ctx.det("HX3", i, c["i1"], sub="%s:%s" % (side, digit), key="%s:%s" % (side, digit),
                                   measures=dict(force_n=c["peak_force_n"], contact_s=c["duration_s"] + STEP_S),
                                   confidence=1.0, objects=[tr.ent.label(other)], parts=["finger_%s_%s" % (side, digit)],
                                   touched_kind=P.entity_class(tr, other)))
    return out


SIMPLE = [H01, H02, H03, H04, H05, H06, H07, H08, H09, B01, B02, B03, B04, B05, B06, G01, G02, G03,
          G04, G05, R02, R04, R05, R06, R07, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11,
          P12, O01, O02, O03, O04, O05, O06, O07, O08, O09, O10, HX1, HX2, HX3]


def run_all(tr: Track, prims: List[dict]) -> List[dict]:
    """Every detector over one track. R01 feeds R03; R08 feeds O11. O12 comes from novelty.py."""
    ctx = Context(tr, prims)
    out: List[dict] = []
    for fn in SIMPLE:
        out.extend(fn(ctx))
    r01 = R01(ctx)
    out.extend(r01)
    out.extend(R03(ctx, r01))
    r08 = R08(ctx)
    out.extend(r08)
    out.extend(O11(ctx, r08))
    return dedupe(out)


def dedupe(dets: List[dict]) -> List[dict]:
    """One act counts once: detections of one capability with the same sub-row and repetition
    key that begin on the same step (three parts of one hand touching the toy, one reach toward
    a row of boxes) keep the most confident; the others are dropped."""
    best: Dict[tuple, dict] = {}
    for d in dets:
        k = (d["cap"], d.get("sub", ""), d.get("key", ""), d["i0"])
        if k not in best or d["confidence"] > best[k]["confidence"]:
            best[k] = d
    return sorted(best.values(), key=lambda d: (d["i0"], d["cap"]))


ALL_IDS = ["H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "B01", "B02", "B03", "B04",
           "B05", "B06", "G01", "G02", "G03", "G04", "G05", "R01", "R02", "R03", "R04", "R05", "R06",
           "R07", "R08", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11",
           "P12", "O01", "O02", "O03", "O04", "O05", "O06", "O07", "O08", "O09", "O10", "O11", "O12"]
EXTRA_IDS = ["HX1", "HX2", "HX3"]
