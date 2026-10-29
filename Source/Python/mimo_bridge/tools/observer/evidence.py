# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The intent/prediction join (addendum section 34): what the learner expected, and whether its
planner chose the act, attached to each externally detected act.

God-state never flows the other way: this reads the learner's own records (the model lane and
causal_chain.jsonl) and joins them to detections by body and step. Four records per act, each
with its control, so a value is never read without the line it is judged against:

  credit       WHICH ORDER AN ACT IS CREDITED TO, for every act except an act of the hands alone: the
               order in force at the act's PEAK motion, not at its first step (movement lags its order).
               The peak is the world record within the act where the act's own parts moved most: the
               learner's handles for those parts (the same handles the efference verdict reads), their
               positions' change since the record before in centimetres, or for the eyes the change of
               the viewing direction in degrees. The world recorder stamps every record with the order
               in force (order_seq) and its author. Intention, the efference verdict and the planner-
               versus-babble rate use the credited order and its author.
               AN ACT OF THE HANDS ALONE (is_hand_act: every learner handle of its parts is a wrist, thumb or
               finger handle) IS CREDITED TO NO SINGLE ORDER (observer_attribution_v1, the operator's
               instruction of 7:37 AM PDT, Oct 8, 2026): command_intervals lists every order in force during
               it, the movement made DURING each order's records, and the orders of the records before it.
               "During" is not "caused by": a joint that keeps moving after its order changed may move under
               an earlier order's force or by mechanical coupling, so each interval also says whether that
               order's own motors pushed the way the joints went and whether its hand joint goal asked for
               that direction, and nothing more is concluded. Its writer class (credit.provenance) is the
               orders' class when every order in force during it shares one (planner-selected,
               spontaneous), 'mixed' otherwise; its intention verdict needs every one of those orders to be
               a decisive planner choice that reached the body and did not ask the other way; each of those
               orders is one unit of its movement prediction (below)
  prediction   Was the consequence in the learner's prediction BEFORE execution?
               - touch acts: the contact head's probabilities FOR the onset record (formed from
                 the record before and the order in force), against each patch's base rate of
                 starting a touch; predicted = some skin patch of the touching part that was
                 untouched at the record before has at least twice its base rate (ratio >= 2).
                 Judged only where the world record shows a touch starting on those patches.
                 The same verdict is scored at MATCHED NO-TOUCH MOMENTS (same body, patches and
                 bit, no touch starting, one, two and three seconds before and after the onset at
                 the onset's place in its order: contact_nulls) so the tracker can test onsets
                 against the verdict's measured chance rate instead of one half
               - movement acts: the efference expectation formed when the CREDITED order began,
                 against two controls for the handles the act moved: the same expectation
                 computed with ANOTHER body's order (the shuffled control), and PERSISTENCE, the
                 movement the previous order achieved (a body that "keeps doing what it did" - a
                 learner that only knows the body follows orders beats the shuffled control for
                 free, not this one); predicted = the own-order expectation was closer to what
                 the body did than both. For an act of the hands alone EVERY order in force during it
                 is judged, over that order's whole hold (movement_prediction_intervals), and each order
                 is one unit of the Predicted stage, the act's orders one cluster (tracker.
                 cluster_binom_above): the act-level conjunction's chance falls with its number of orders
               - object-effect acts: the next-moment prediction error (touch and vision) averaged
                 over the act and the two records after it (a push's consequence comes after its
                 first step), against the same body's median over the 2 s before; predicted = no
                 higher than that median (the act was not a surprise)
  intention    Did the learner's intention process select it? The credited order was written by
               the planner (or the qualified self-choice actor), the planner's evaluator ranked
               the chosen plan 'best' by more than its measured tie band (causal_chain.py: 0.0734
               on H84, so a 'tied' choice is an arbitrary pick among equals), and the order
               reached the body. The planner record must be the credited order's own (a latch
               without a record would otherwise lend the act the previous order's verdict). An act of
               the hands alone needs this of EVERY order in force during it, and none of them may have
               asked the other way (intention_intervals)
  wellness     the grounded wellness channels of that body at the act's first and last records:
               internal evidence shown under the behaviour, never in place of it; beside it the same
               change over stretches of the act's length that end in the 3 s before it starts, one a
               second (wellness_references: the references the page shows it against)

MATCHED MOMENTS ARE PLACED BY THEIR DISTANCE FROM THE ACT, NOT BY THE DIGESTION WINDOW (analysis v5;
independent review of 2026-09-25, finding 1). Until then both controls took one moment a second from the
steps the digestion window finalized and gave each to the nearest act of that window, so a moment could
lie as far from its act as the window was long: the live digester finalizes 100 to 200 steps a window,
a batch digest 2,000. On segment 40's 9:33 AM run, digested both ways with the same code, the touch
verdict passed at the no-touch moments of "whole-hand close after touch" 65 times in 100 in live-sized
windows and 35 in 100 in batch windows; the stretches without the act of "wrist rotation" changed
wellness by +0.0042 against +0.0138. Now each act's moments are fixed by the act alone: the touch
verdict's at 1, 2 and 3 s (MATCH_REACH_STEPS) before and after its onset, each moment given to the
earliest act of the same row and skin patches that has it among its own (the acts before it, which
every window reads in its look-back); the wellness references' stretches end in the 3 s before the act
starts, so they read no record after the act and no act the window has not finished. Both at the act's
own place in its order (order_phase).

THE LEARNER'S LANE IS ONE RECORD BEHIND THE WORLD'S (TRAINER_LANE_LAG). Loop step t of the trainer
first lets the learner read the frames of the burst run at step t - 1 (world record t - 1), then
latches the order and runs burst t (world record t). So the model lane's per-step records at lane
step t - the contact head's probabilities, the next-moment errors, the grounded wellness - describe
world record t - 1, and an efference record written at lane step t graded the order that drove
world records [began, t - 1). Measured 2026-09-24 on segment 39 (2,075 steps x 8 bodies, read-only):
every efference record's order began at `began` and the next order at t - 1 (3,304 of 3,304), every
latch's order was in force at the latch's own world record (3,304 of 3,304), and the contact head's
probabilities at lane step t follow the patch states of world record t - 2 - the input it was given,
the record before the one it predicts - more closely than any other record (area under the curve
0.996, against 0.949 for record t - 1 and 0.927 for record t). Before observer_analysis_v4 the join
read lane step s for world record s: the contact verdict judged the prediction for the record BEFORE
the touch, and the efference window ran one record into the next order.
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detectors import (AUTHOR_NAMES, MODEL_SELECTED, MOTOR_ON, OBJECT_CONTACT_CAPS, STEP_S, _motor_of_qpos,
                        provenance_class)
from .kinematics import FINGERS, SIDES

CONTACT_CAPS = {"H01", "B01", "B02", "B03", "B04", "B05", "R02", "HX3"}
SELF_TOUCH_CAPS = {"B01", "B02", "B03", "B04", "B05"}
MOVE_CAPS = {"H04", "H05", "H06", "H07", "H08", "HX1", "HX2", "R01", "R03", "B06", "P02", "P03",
             "P04", "P05", "P06", "P07", "P08", "P09", "P10", "G01", "G02", "G03", "G04", "G05"}
CONTACT_LIFT = 2.0
SURPRISE_WINDOW = 40
# The model lane's step t describes world record t - TRAINER_LANE_LAG (module doc). A lane built in
# world-record numbering (the tests' synthetic lanes) has lag 0; the daemon joins production lanes
# with this constant.
TRAINER_LANE_LAG = 1
# Matched moments, one a second: NULL_EVERY_STEPS apart, the occasion gap (tracker.OCCASION_GAP_STEPS,
# 1 s): the head's probabilities and the wellness channels change over a few records, so consecutive
# records would be the same observation counted again.
NULL_EVERY_STEPS = 20
# How far a matched moment lies from its act (module doc, MATCHED MOMENTS): at most 3 s, the digester's
# held-back end (daemon.MARGIN, 60 steps), so the window that finalizes an act has recorded every moment
# after it; the look-back (daemon.LOOKBACK_CHUNKS, 1,000 steps) holds every one before it.
MATCH_REACH_STEPS = 60
# An act longer than this gets no wellness reference: its stretches would start before the 1,000 steps
# of look-back every window reads before the acts it finalizes (the lineage's longest act to 7:26 PM
# PDT on 2026-09-25: 790 steps, palm shaping; the median of every row 1 to 58 steps)
REFERENCE_MAX_STEPS = 800
BITS = ("any", "self", "world")
# AN ACT OF THE HANDS ALONE IS CREDITED TO NO SINGLE ORDER (observer_attribution_v1, 2026-10-08; the operator's
# instruction of 7:37 AM PDT, Oct 8, 2026, relayed by the coordinator; observer source review F2,
# D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/observer/REVIEW.md). From analysis v5 to this change a hand act was credited to the
# order in force one record before the record of its largest WORLD-frame motion (the lag rule: at an order boundary
# the hands' change of direction is split between the new order's first two records, the new order's side winning at
# 34 of 100 boundaries in babble and 39 in production). On the 47 left-index acts of seg41c_0841 (control run 1,
# resumed from checkpoint 481,800) that record was another record than the record of the finger's largest bend in 45,
# and the credited order another order in 34. In body 5's act at model steps 483,793-483,802 the index finger bent
# 20.0 degrees under three orders: 73.5 in 100 of the bend during order 96760 (the motor controller with exploration
# noise), 21.7 during 96761 (the planner, whose joint goals asked the knuckle and middle joint to straighten) and 4.8
# during 96759 (the planner); the act was credited to 96761. Now each record's movement belongs to the order its burst
# ran under (command_intervals), a movement that continues after its order changed is reported as made DURING the next
# order beside that order's own motors and goal, and no order is said to have caused it. Acts of other parts, or of
# hands together with other parts, keep the peak's order (credit).
ATTRIBUTION_VERSION = "observer_attribution_v1"
HAND_HANDLES = ("wrist_", "thumb_", "index_", "middle_", "ring_", "pinky_")
LAGGING_HANDLES = HAND_HANDLES          # analysis v5's name for them (the lag rule itself is gone)
# The records before an act whose orders are listed as its history: two holds of the 5 model steps an order is held.
# A reporting window, never a gate.
HISTORY_RECORDS = 10
# The writers as the notes of an act of the hands alone name them (the record fields keep the code names, AUTHOR_NAMES);
# the text this module adds for those acts says "command" for what the code calls an order.
WRITER_WORDS = {"planner": "the planner", "reflex": "the motor controller",
                "reflex_noise": "the motor controller with exploration noise (reflex_noise)",
                "self_choice": "the self-choice actor (self_choice)",
                "acting_snapshot": "the acting snapshot (acting_snapshot)"}


def writer_words(name) -> str:
    return WRITER_WORDS.get(str(name), "the %s writer" % name)

PART_HANDLES = {
    "trunk": ["lower_trunk", "chest"], "head": ["head"],
}


def handles_for(parts: List[str], cap: str) -> List[str]:
    out: List[str] = []
    for p in parts:
        bits = p.split("_")
        if p.startswith("finger_"):
            side, digit = bits[1], bits[2]
            out.append({"little": "pinky"}.get(digit, digit) + "_" + side)
        elif p.startswith("hand_") and p.endswith("_palm"):
            out.append("wrist_" + bits[1])
        elif p.startswith("hand_"):
            s = bits[1]
            out += ["wrist_" + s, "thumb_" + s, "index_" + s, "middle_" + s, "ring_" + s, "pinky_" + s]
        elif p.startswith("arm_"):
            out += ["upperarm_" + bits[1], "lowerarm_" + bits[1]]
        elif p.startswith("leg_"):
            out += ["thigh_" + bits[1], "calf_" + bits[1], "foot_" + bits[1]]
        elif p.startswith("eye"):
            out.append("gaze")
        elif p in PART_HANDLES:
            out += PART_HANDLES[p]
    if cap.startswith("G") and "gaze" not in out:
        out.append("gaze")
    seen = []
    for h in out:
        if h not in seen:
            seen.append(h)
    return seen


def order_phase(tr) -> np.ndarray:
    """Each record's place in its order: steps since the order's first record (0 there). -1 in the
    track's first order, which may have begun before the first record read (its place is not known
    from this window), so a window starting mid-order never matches on a guessed place. 0 everywhere
    for a track without order numbers."""
    seq = getattr(tr, "order_seq", None)
    if seq is None:
        return np.zeros(tr.n, dtype=np.int64)
    seq = np.asarray(seq, dtype=np.int64)
    steps = np.asarray(tr.step, dtype=np.int64)
    idx = np.arange(tr.n)
    first = np.r_[False, seq[1:] != seq[:-1]]
    start = np.maximum.accumulate(np.where(first, idx, -1))       # the index where each order began
    return np.where(start >= 0, steps - steps[np.maximum(start, 0)], -1)


def _index_of(steps: np.ndarray, step: int) -> int:
    """The record of a world step in a track, or -1 when the track has none."""
    i = int(np.searchsorted(steps, step))
    return i if i < steps.size and int(steps[i]) == int(step) else -1


def _eyes_for(parts: List[str]) -> List[int]:
    """Which eyes an act's gaze motion is read from: the one it names (eye_L, eye_R), else both."""
    named = [0 if p == "eye_L" else 1 for p in parts if p in ("eye_L", "eye_R")]
    return named or [0, 1]


class Joiner:
    def __init__(self, lane, chain: List[dict], model_handles: List[str], world_header: dict,
                 lane_lag: int = 0):
        self.lane = lane
        self.lane_lag = int(lane_lag)
        self.handles = list(model_handles or [])
        self.patch_bodies = list(world_header.get("patch_bodies") or [])
        self.entities = world_header["entities"]
        self.handle_body = dict(zip(world_header.get("handle_names") or [],
                                    world_header.get("handle_bodies") or []))
        # the infant's own body groups: an act's touched_kind is one of these for an own part
        # (primitives.entity_class names the group, never the word 'self')
        self.self_groups = {e.get("group") for e in self.entities if e.get("kind") == "self"}
        latches = sorted((r for r in chain if r.get("type") == "latch"), key=lambda r: int(r.get("step", -1)))
        self.latch_steps = [int(r["step"]) for r in latches]
        self.latches = latches
        self.eff_steps = sorted(lane.eff) if lane is not None else []
        self._states_of = None

    # ------------------------------------------------------------------ credit
    def _motion(self, d: dict, tr, i0: int, i1: int) -> Tuple[Optional[np.ndarray], str]:
        """Per world record i0..i1, how far the act's own parts moved since the record before:
        the root sum square of its handles' position changes in cm, or the eyes' change of
        viewing direction in degrees for an act of the eyes alone. (None, why) when unmeasured."""
        hs = handles_for(d.get("parts") or [], d["cap"])
        bodies = [int(self.handle_body[h]) for h in hs if h != "gaze" and h in self.handle_body]
        cont = np.asarray(getattr(tr, "continuous", np.ones(tr.n, dtype=bool)), dtype=bool)
        idx = np.arange(i0, i1 + 1)
        ok = (idx >= 1) & cont[idx]            # a record whose predecessor is the moment before
        step = np.full(idx.size, np.nan)
        if bodies and getattr(tr, "xpos", None) is not None:
            if ok.any():
                now = np.asarray(tr.xpos[idx[ok]][:, bodies, :], dtype=np.float64)
                before = np.asarray(tr.xpos[idx[ok] - 1][:, bodies, :], dtype=np.float64)
                step[ok] = 100.0 * np.sqrt(np.sum((now - before) ** 2, axis=(1, 2)))
            return step, "cm"
        if "gaze" in hs and getattr(tr, "eye_fwd", None) is not None:
            if ok.any():
                eyes = _eyes_for(d.get("parts") or [])
                u = np.asarray(tr.eye_fwd[idx[ok] - 1][:, eyes, :], dtype=np.float64)
                v = np.asarray(tr.eye_fwd[idx[ok]][:, eyes, :], dtype=np.float64)
                nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.clip(np.sum(u * v, axis=-1) / (nu * nv), -1.0, 1.0)
                    ang = np.degrees(np.arccos(c))                  # [records, eyes]
                ang = np.where(np.isfinite(ang), ang, -np.inf).max(axis=1)
                step[ok] = np.where(np.isfinite(ang), ang, np.nan)
            return step, "deg"
        return None, "no learner handle or eye moves these parts"

    @staticmethod
    def is_hand_act(d: dict) -> bool:
        """An act of the hands alone: every learner handle of its parts is a wrist, thumb or finger handle
        (HAND_HANDLES). Such an act is credited to no single order (module doc, credit; command_intervals)."""
        hs = [h for h in handles_for(d.get("parts") or [], d["cap"]) if h != "gaze"]
        return bool(hs) and all(h.startswith(HAND_HANDLES) for h in hs)

    lags = is_hand_act                     # analysis v5's name for the same test

    def credit(self, d: dict, tr) -> dict:
        """The order an act is credited to: the one in force at the record of its largest motion (module
        doc). Falls back to the first record, saying why, when the motion is unmeasured. `credit_step` is the
        record whose order is credited, `peak_step` the peak's. An act of the hands alone gets no single order:
        the summary of its command_intervals (interval_credit)."""
        if self.is_hand_act(d):
            return self.interval_credit(self.command_intervals(d, tr))
        i0 = int(d.get("i0", 0))
        i1 = int(d.get("i1", i0))
        i1 = max(i0, min(i1, tr.n - 1))
        order = getattr(tr, "order_seq", None)
        author = getattr(tr, "author", None)
        first_seq = int(order[i0]) if order is not None else int(d.get("order_seq", -1))
        first_author = (AUTHOR_NAMES.get(int(author[i0]), "unknown") if author is not None
                        else d.get("author", "unknown"))
        motion, unit = self._motion(d, tr, i0, i1)
        peak, note, size = i0, None, None
        if motion is None:
            note = unit
        elif not np.isfinite(motion).any():
            note = "the parts' motion is not measured at these records"
        else:
            j = int(np.nanargmax(motion))
            peak, size = i0 + j, float(motion[j])
        at = peak
        seq = int(order[at]) if order is not None else first_seq
        who = AUTHOR_NAMES.get(int(author[at]), "unknown") if author is not None else first_author
        out = dict(peak_step=int(tr.step[peak]), credit_step=int(tr.step[at]), lag_records=0, order_seq=seq,
                   author=who, provenance=provenance_class(who),
                   first_step=int(tr.step[i0]), first_order_seq=first_seq, first_author=first_author,
                   moved=bool(seq != first_seq), peak_motion=size, motion_unit=unit if motion is not None else None,
                   measured=note is None)
        if note:
            out["note"] = note + "; credited to the order at the act's first record"
        return out

    # ------------------------------------------------------------------ the orders of an act of the hands
    @staticmethod
    def act_measure(d: dict, tr) -> Optional[dict]:
        """An act's own joint-space measure in degrees, as a weighted sum of joint angles: measure(record) = sum
        over its joints of weight x degrees(qpos[record, address]). H05 a digit's flexion (the mean of its flexion
        hinges, kinematics.curl_deg); H04 and H01 the hand's closure (the mean of the four fingers' flexions); HX1 the
        fan (the index finger's side angle minus the little finger's); HX2 the little-finger metacarpal's cupping
        hinge; H07 the wrist's flexion (signed as kinematics.wrist_deg signs it); H08 the wrist's rotation. None for
        any other act or a track without the joints; its movement is then its parts' world-frame motion (_motion)."""
        kin = getattr(tr, "kin", None)
        if kin is None or getattr(tr, "qpos", None) is None:
            return None
        cap, sub = d["cap"], str(d.get("sub") or "")
        side = sub.split(":")[0]
        flex = getattr(kin, "flex_q", None) or {}
        joints, name = None, None
        if cap == "H05" and ":" in sub:
            digit = sub.split(":", 1)[1]
            adrs = list(flex.get((side, digit)) or [])
            if adrs:
                joints, name = [(int(a), 1.0 / len(adrs)) for a in adrs], "%s %s flexion" % (side, digit)
        elif cap in ("H04", "H01") and side in SIDES:
            fingers = [list(flex.get((side, f)) or []) for f in FINGERS]
            fingers = [f for f in fingers if f]
            if fingers:
                joints = [(int(a), 1.0 / (len(fingers) * len(f))) for f in fingers for a in f]
                name = "%s hand closure" % side
        elif cap in ("HX1", "HX2") and side in SIDES:
            adr = {j["name"].split(":")[-1]: int(j["qpos"]) for j in ((getattr(tr, "header", None) or {}).get("joints") or [])}
            pre = "left_" if side == "L" else "right_"
            if cap == "HX1" and pre + "ff_side" in adr and pre + "lf_side" in adr:
                joints = [(adr[pre + "ff_side"], 1.0), (adr[pre + "lf_side"], -1.0)]
                name = "%s fan (index side angle minus little-finger side angle)" % side
            elif cap == "HX2" and pre + "lf_meta" in adr:
                joints, name = [(adr[pre + "lf_meta"], 1.0)], "%s palm cupping hinge" % side
        elif cap in ("H07", "H08") and side in SIDES:
            kind = "flexion" if cap == "H07" else "rotation"
            wq = (getattr(kin, "wrist_q", None) or {}).get(side, {})
            wk = (getattr(kin, "wrist_kind", None) or {}).get(side, {})
            adrs = [a for role, a in wq.items() if wk.get(role) == kind]
            if adrs:
                sign = float((getattr(kin, "wrist_sign", None) or {}).get(side, 1.0)) if kind == "flexion" else 1.0
                joints, name = [(int(a), sign) for a in adrs], "%s wrist %s" % (side, kind)
        if not joints:
            return None
        return dict(name=name, unit="deg", joints=joints)

    def command_intervals(self, d: dict, tr) -> dict:
        """THE ORDERS OF AN ACT, ONE INTERVAL EACH (observer_attribution_v1; module doc, credit). The act's movement
        is the change from its first record i0 to its last i1; the change from record r - 1 to r is made during the
        burst of record r, under the order in force at r. So:
          intervals  one per order in force at records i0 + 1 .. i1 (an act of one record: at i0, the record its own
                     burst made), in time order: the order (`command`), its writer
                     and provenance class, the model step it was latched at (None when it began before the records
                     read), the first and last model step of its records inside the act, the act measure's change and
                     path during them (degrees; for an act without a joint measure its parts' world-frame path, cm),
                     their share of the act's change and path, the parts' world-frame path (cm), whether the order's
                     own motors pushed the way each joint went (the share of the joint path, `motors_same_way_share`:
                     a coincidence, never a cause), the measure the order's hand joint goal asked for against the
                     measure where its interval began, and `asked`: 'same', 'opposite' or None (no goal recorded);
                     and over the order's WHOLE HOLD (all its consecutive records, inside the act or not: the window
                     its efference record grades) `hold_records`, `hold_whole` (read whole), `hold_change` (from the
                     record before its first record to its last), `hold_asked_change` (its goal minus the measure
                     there) and `hold_asked`.
          history    the orders of the HISTORY_RECORDS records up to and including i0 (an act of one record: up to
                     i0 - 1), the same fields: a movement that continues into the act may come from their forces.
          writer_class   the provenance class every interval shares, else 'mixed'; excluded_class the first probe,
                     baseline or scheduled-rest class at records i0 .. i1, if any (such an act is never counted).
        Movement during an interval is not movement caused by its order: a joint keeps moving after its order changed,
        under earlier forces or mechanical coupling (`note`)."""
        i0 = int(d.get("i0", 0))
        i1 = max(i0, min(int(d.get("i1", i0)), tr.n - 1))
        seq = getattr(tr, "order_seq", None)
        auth = getattr(tr, "author", None)
        steps = np.asarray(tr.step, dtype=np.int64)
        cont = np.asarray(getattr(tr, "continuous", np.ones(tr.n, dtype=bool)), dtype=bool)
        out = dict(version=ATTRIBUTION_VERSION, measure=None, unit=None, act_change=None, act_path=None, intervals=[],
                   history=[], writer_class=None, writer=None, excluded_class=None, n_commands=0,
                   note=("movement listed under a command was made DURING that command's records; it is not said to "
                         "be caused by that command: a joint keeps moving after its command changed, under earlier "
                         "forces or mechanical coupling"))
        if seq is None or auth is None:
            out["note"] = "the records carry no command numbers: no interval measured"
            return out
        seq = np.asarray(seq, dtype=np.int64)
        auth = np.asarray(auth, dtype=np.int64)
        meas = self.act_measure(d, tr)
        series = None
        if meas is not None:
            adrs = [a for a, _ in meas["joints"]]
            wts = np.asarray([w for _, w in meas["joints"]], dtype=np.float64)
            series = np.degrees(np.asarray(tr.qpos, dtype=np.float64)[:, adrs]) @ wts
            out.update(measure=meas["name"], unit="deg", act_change=float(series[i1] - series[i0]))
        motion, munit = self._motion(d, tr, i0, i1)
        if meas is None:
            out.update(measure="world-frame motion of the act's handles", unit=munit if motion is not None else None)
        mp = _motor_of_qpos(tr) if meas is not None else {}
        ctrl = getattr(tr, "ctrl", None)
        goal = getattr(tr, "order_hand_goal", None)
        kin = getattr(tr, "kin", None)
        gq = kin.hand_goal_qpos() if (meas is not None and goal is not None and kin is not None
                                      and hasattr(kin, "hand_goal_qpos")) else None
        gidx = {int(a): k for k, a in enumerate(gq)} if gq else {}

        def goal_measure(r: int) -> Optional[float]:
            if meas is None or goal is None or not gidx:
                return None
            ks = [gidx.get(int(a)) for a, _ in meas["joints"]]
            if any(k is None or k >= goal.shape[1] for k in ks):
                return None
            vals = np.asarray([goal[r, k] for k in ks], dtype=np.float64)
            if not np.isfinite(vals).all():
                return None
            return float(np.degrees(vals) @ np.asarray([w for _, w in meas["joints"]], dtype=np.float64))

        def motors(rs: List[int]) -> Optional[float]:
            if meas is None or ctrl is None or np.size(ctrl) == 0 or not mp:
                return None
            tot = same = 0.0
            for a, w in meas["joints"]:
                u = mp.get(int(a))
                if u is None:
                    continue
                for r in rs:
                    dq = float(np.degrees(tr.qpos[r, a] - tr.qpos[r - 1, a]))
                    c = float(ctrl[r, u])
                    if not (np.isfinite(dq) and np.isfinite(c)):
                        continue
                    tot += abs(w * dq)
                    if abs(c) >= MOTOR_ON and c * dq > 0:
                        same += abs(w * dq)
            return (same / tot) if tot > 0 else None

        def runs(records: List[int]) -> List[tuple]:
            """Consecutive records under one command and one writer: [(command, author, [records])]."""
            groups: List[tuple] = []
            for r in records:
                key = (int(seq[r]), int(auth[r]))
                if groups and groups[-1][0] == key[0] and groups[-1][1] == key[1] and r == groups[-1][2][-1] + 1:
                    groups[-1][2].append(r)
                else:
                    groups.append((key[0], key[1], [r]))
            return groups

        def hold_of(rs: List[int]) -> tuple:
            """THE COMMAND'S WHOLE HOLD around these records: the consecutive records under the same command and writer
            that contain them, within the records read, and whether it is read whole (the records just before its
            first and just after its last are read, consecutive, and under other commands)."""
            a, b = rs[0], rs[-1]
            key = (int(seq[a]), int(auth[a]))
            while a >= 1 and (int(seq[a - 1]), int(auth[a - 1])) == key and steps[a - 1] == steps[a] - 1:
                a -= 1
            while b + 1 < tr.n and (int(seq[b + 1]), int(auth[b + 1])) == key and steps[b + 1] == steps[b] + 1:
                b += 1
            whole = bool(a >= 1 and steps[a - 1] == steps[a] - 1 and int(seq[a - 1]) != key[0]
                         and b + 1 < tr.n and steps[b + 1] == steps[b] + 1 and int(seq[b + 1]) != key[0])
            return a, b, whole

        def describe(order: int, author_code: int, rs: List[int]) -> dict:
            writer = AUTHOR_NAMES.get(int(author_code), "unknown")
            first = int(np.argmax(seq == order))
            lat = getattr(tr, "latched", None)
            began_before = first == 0 and int(seq[0]) == order and not (lat is not None and bool(np.asarray(lat)[0]))
            latched = None if began_before else int(steps[first])
            moving = [r for r in rs if r >= 1 and cont[r]]
            e = dict(command=int(order), writer=writer, provenance=provenance_class(writer), latched_step=latched,
                     records=[int(steps[rs[0]]), int(steps[rs[-1]])], n_records=len(rs))
            # THE HOLD WINDOW (independent review of 4342f9194, fix 2): the command's efference record grades its WHOLE
            # hold, the change from the record before its first record to its last (the lane's achieved movement
            # matched the world's over exactly those records, offset -1, on all 47 left-index acts of control run 1:
            # defect_checks.json M1); "achieved" reads the same window, so both stages read one window per command
            ha, hb, whole = hold_of(rs)
            e.update(hold_records=[int(steps[ha]), int(steps[hb])], hold_whole=whole, hold_change=None,
                     hold_asked_change=None, hold_asked=None)
            if series is not None and whole and bool(np.all(cont[ha:hb + 1])):
                start = float(series[ha - 1])
                e["hold_change"] = float(series[hb] - start)
                gh = goal_measure(ha)
                if gh is not None:
                    e["hold_asked_change"] = float(gh - start)
                    if e["hold_change"] != 0.0 and gh != start:
                        e["hold_asked"] = "same" if (gh - start) * e["hold_change"] > 0 else "opposite"
            if series is not None and moving:
                ch = np.asarray([series[r] - series[r - 1] for r in moving], dtype=np.float64)
                e.update(moved_during=float(np.nansum(ch)), path_during=float(np.nansum(np.abs(ch))),
                         measure_at_start=float(series[moving[0] - 1]))
                g = goal_measure(moving[0])
                e["goal_measure"] = g
                if g is None or not np.isfinite(e["moved_during"]) or e["moved_during"] == 0.0:
                    e["asked"] = None
                else:
                    want = g - e["measure_at_start"]
                    e["asked"] = None if want == 0.0 else ("same" if want * e["moved_during"] > 0 else "opposite")
                e["motors_same_way_share"] = motors(moving)
            if motion is not None:
                w = [motion[r - i0] for r in rs if i0 <= r <= i1 and np.isfinite(motion[r - i0])]
                e["world_path"] = float(np.sum(w)) if w else None
                if series is None:
                    e["path_during"] = e["world_path"]
            return e

        # an act of one record (a touch at its onset record: HX3, B02) holds no change between its records; the record
        # itself was made by its own burst, under the command in force at it, which is its one interval
        act_records = list(range(i0 + 1, i1 + 1)) if i1 > i0 else [i0]
        ivs = [describe(o, a, rs) for o, a, rs in runs(act_records)]
        tot_path = sum(float(e.get("path_during") or 0.0) for e in ivs)
        for e in ivs:
            if out["act_change"] not in (None, 0.0) and e.get("moved_during") is not None:
                e["share_of_act_change"] = float(e["moved_during"] / out["act_change"])
            if tot_path > 0 and e.get("path_during") is not None:
                e["share_of_act_path"] = float(e["path_during"] / tot_path)
        last = i0 if i1 > i0 else i0 - 1          # the history ends where the act's own records begin
        hist = [describe(o, a, rs) for o, a, rs in runs(list(range(max(0, last - HISTORY_RECORDS + 1), last + 1)))]
        classes = []
        for e in ivs:
            if e["provenance"] not in classes:
                classes.append(e["provenance"])
        excluded = None
        for r in range(i0, i1 + 1):
            c = provenance_class(AUTHOR_NAMES.get(int(auth[r]), "unknown"))
            if c in ("probe-forced", "baseline", "external-rest"):
                excluded = c
                break
        writers = sorted({e["writer"] for e in ivs})
        out.update(act_path=tot_path, intervals=ivs, history=hist, n_commands=len(ivs),
                   writer_class=(classes[0] if len(classes) == 1 else ("mixed" if classes else None)),
                   writer=(writers[0] if len(writers) == 1 else ("mixed" if writers else None)),
                   excluded_class=excluded)
        return out

    @staticmethod
    def interval_credit(att: dict) -> dict:
        """The credit record of an act of the hands alone: no single order (rule 'command_intervals'), its writer
        class (or the excluded class that keeps it from being counted), and the orders in force during it."""
        return dict(rule="command_intervals", version=att.get("version"),
                    provenance=att.get("excluded_class") or att.get("writer_class") or "unknown",
                    author=att.get("writer") or "unknown",
                    commands=[e["command"] for e in att.get("intervals") or []],
                    n_commands=int(att.get("n_commands") or 0), measured=bool(att.get("intervals")), moved=None,
                    note="credited to no single command: every command in force during the act is listed in "
                         "evidence.attribution, with the movement made during its records")

    # ------------------------------------------------------------------ skin patches
    def _patches(self, d: dict, tr) -> List[int]:
        ents = d.get("contact_entities")
        bodies = set()
        if ents:
            for e in ents:
                bodies.update(self.entities[int(e)].get("bodies") or [])
        else:
            for p in d.get("parts") or []:
                for e in self.entities:
                    if e["kind"] != "self":
                        continue
                    g = e.get("group", "")
                    if (p.startswith("hand_") and len(p) == 6 and (g.startswith("hand_" + p[5]) or g.startswith("finger_" + p[5]))) \
                            or g == p:
                        bodies.update(e.get("bodies") or [])
        return [i for i, b in enumerate(self.patch_bodies) if int(b) in bodies]

    def patch_states(self, tr) -> Optional[np.ndarray]:
        """[n, P, 3] bool per world record of the track: anything / an own part / an outside thing
        touching each skin patch (the contact head's three bits), from the world recorder's
        contacts (every contact of the record's burst: the union the learner is graded on).
        None for a track without a contact table."""
        if self._states_of is not None and self._states_of[0] is tr:
            return self._states_of[1]
        if getattr(tr, "c_row", None) is None or not self.patch_bodies:
            return None
        patch_of_body = {int(b): i for i, b in enumerate(self.patch_bodies)}
        ent_patch = np.full(len(self.entities), -1, dtype=np.int64)
        ent_self = np.zeros(len(self.entities), dtype=bool)
        for k, e in enumerate(self.entities):
            if e.get("kind") == "self":
                ent_self[k] = True
                for b in e.get("bodies") or []:
                    if int(b) in patch_of_body:
                        ent_patch[k] = patch_of_body[int(b)]
        st = np.zeros((tr.n, len(self.patch_bodies), 3), dtype=bool)
        rows = np.asarray(tr.c_row, dtype=np.int64)
        es = np.asarray(tr.c_e, dtype=np.int64).reshape(-1, 2)
        for side in (0, 1):
            e, o = es[:, side], es[:, 1 - side]
            ok = (e >= 0) & (e < ent_patch.size) & (o >= 0) & (o < ent_self.size)
            p = np.where(ok, ent_patch[np.clip(e, 0, ent_patch.size - 1)], -1)
            ok &= p >= 0
            r, pp = rows[ok], p[ok]
            own = ent_self[o[ok]]
            st[r, pp, 0] = True
            st[r[own], pp[own], 1] = True
            st[r[~own], pp[~own], 2] = True
        self._states_of = (tr, st)
        return st

    def _bit(self, d: dict) -> int:
        """The contact head's bit an act is judged on: an own part for the self-touch rows and for
        any touch of the infant's own body, an outside thing for a touch of anything else (a row
        made of touching an object, detectors.OBJECT_CONTACT_CAPS, when the act does not name what
        it touched), and 'anything' otherwise. (Before observer_analysis_v4 a touch of an own part
        was judged on the outside bit: touched_kind names the part's group, never the word 'self';
        and a repeat contact, which names no touched kind, on 'anything'.)"""
        if d["cap"] in SELF_TOUCH_CAPS:
            return 1
        tk = d.get("touched_kind")
        if tk is None:
            return 2 if d["cap"] in OBJECT_CONTACT_CAPS else 0
        return 1 if (tk == "self" or tk in self.self_groups) else 2

    def contact_key(self, d: dict, tr) -> Optional[tuple]:
        """The key a touch act's matched no-touch moments are scored and shared under: its row, its
        touching part's skin patches and the contact head's bit (contact_nulls); None for an act of
        another row or of a part without skin patches."""
        if d["cap"] not in CONTACT_CAPS:
            return None
        patches = self._patches(d, tr)
        return (d["cap"], tuple(patches), self._bit(d)) if patches else None

    def matched_horizon(self, d: dict, tr) -> Optional[int]:
        """The last world step an act's matched moments read (contact_nulls: MATCH_REACH_STEPS after a
        touch's onset), or None for an act whose matched moments all lie before it. The digester
        finalizes an act only once its track and the learner's lane reach this step."""
        if self.contact_key(d, tr) is None:
            return None
        return int(tr.step[min(self._onset_index(d, tr), tr.n - 1)]) + MATCH_REACH_STEPS

    def _onset_index(self, d: dict, tr) -> int:
        """The record the touch starts at: the detector's onset_index, else for a repeat contact
        (R02, whose act begins at the approach) its time to contact after the first record, else
        the act's first record."""
        if d.get("onset_index") is not None:
            return int(d["onset_index"])
        i0 = int(d.get("i0", 0))
        ttc = (d.get("measures") or {}).get("time_to_contact_s")
        if d["cap"] == "R02" and ttc is not None and np.isfinite(float(ttc)):
            i1 = int(d.get("i1", i0))
            st = getattr(tr, "sim_time", None)
            if st is not None:
                for j in range(i0, min(i1, tr.n - 1) + 1):
                    if float(st[j]) - float(st[i0]) >= float(ttc) - 1e-6:
                        return j
            return min(i1, i0 + int(round(float(ttc) / STEP_S)))
        return i0

    def _contact_verdict(self, rec: dict, body: int, patches: List[int], bit: int,
                         eligible: Optional[np.ndarray]) -> Optional[dict]:
        """The verdict on one lane record: the largest ratio of probability to base rate of starting
        over the patches that could start a touch (untouched at the record before); None when no
        such patch has a base rate."""
        q = np.asarray(rec["q"], dtype=np.float64)[body, :, bit]
        base = np.asarray(rec["base"], dtype=np.float64)[bit]
        ps = [p for p in patches if (eligible is None or eligible[p]) and np.isfinite(base[p]) and np.isfinite(q[p])]
        if not ps:
            return None
        ratios = q[ps] / np.maximum(base[ps], 1e-6)
        j = int(np.argmax(ratios))
        return dict(patch=int(ps[j]), probability=float(q[ps[j]]), base_rate=float(base[ps[j]]),
                    ratio=float(ratios[j]), verdict=bool(ratios[j] >= CONTACT_LIFT), judged_patches=len(ps))

    def _lane_contact(self, step: int, body: int) -> Optional[dict]:
        """The lane's contact record FOR world record `step`, if the body's report was valid."""
        rec = self.lane.contact.get(int(step) + self.lane_lag)
        if rec is None:
            return None
        valid = rec.get("valid")
        if valid is not None and not bool(np.asarray(valid).reshape(-1)[body]):
            return None
        return rec

    # ------------------------------------------------------------------ prediction
    def contact_prediction(self, d: dict, body: int, tr) -> Optional[dict]:
        if self.lane is None:
            return None
        onset_i = self._onset_index(d, tr)
        step = int(tr.step[onset_i])
        rec = self._lane_contact(step, body)
        if rec is None:
            return dict(kind="contact_head", verdict=None, onset_step=step,
                        note="no valid contact-head record for the onset record")
        patches = self._patches(d, tr)
        if not patches:
            return dict(kind="contact_head", verdict=None, onset_step=step, note="no skin patch on the touching part")
        bit = self._bit(d)
        states = self.patch_states(tr)
        eligible = None
        starting = None
        if states is not None:
            cont = getattr(tr, "continuous", None)
            if onset_i < 1 or (cont is not None and not bool(cont[onset_i])):
                return dict(kind="contact_head", verdict=None, onset_step=step, bit=BITS[bit],
                            note="the record before the onset is not the moment before (a gap or a new world episode)")
            eligible = ~states[onset_i - 1, :, bit]
            starting = [p for p in patches if eligible[p] and states[onset_i, p, bit]]
            if not starting:
                return dict(kind="contact_head", verdict=None, onset_step=step, bit=BITS[bit],
                            note="the world record shows no touch of this kind starting on these skin patches here")
        v = self._contact_verdict(rec, body, patches, bit, eligible)
        if v is None:
            # the contact head reports no base rate until a patch has seen enough touches
            return dict(kind="contact_head", verdict=None, onset_step=step, bit=BITS[bit],
                        note="no measured base rate for the untouched skin patches of this part yet")
        out = dict(kind="contact_head", bit=BITS[bit], onset_step=step, lane_step=step + self.lane_lag,
                   probability=v["probability"], base_rate=v["base_rate"], ratio=v["ratio"], line=CONTACT_LIFT,
                   verdict=v["verdict"], judged_patches=v["judged_patches"], patches=len(patches),
                   starting_patches=None if starting is None else len(starting))
        out["_key"] = self.contact_key(d, tr)
        return out

    def movement_prediction(self, d: dict, body: int, tr, credit: Optional[dict] = None) -> Optional[dict]:
        if self.lane is None or not self.eff_steps:
            return None
        at = int(credit.get("credit_step", credit["peak_step"])) if credit else int(d["step0"])
        k = bisect.bisect_right(self.eff_steps, at)
        rec, s = None, None
        for s_ in self.eff_steps[k:k + 3]:
            r = self.lane.eff[s_]
            # the graded order drove world records [began, s - lane_lag): module doc
            if r["began"] <= at < s_ - self.lane_lag:
                rec, s = r, s_
                break
        if rec is None or "sh_cm" not in rec:
            return dict(kind="efference", verdict=None, note="no efference record for the credited order")
        hs = [self.handles.index(h) for h in handles_for(d.get("parts") or [], d["cap"]) if h in self.handles]
        if not hs:
            return dict(kind="efference", verdict=None, note="no learner handle moves these parts")
        exp, ach, sh = (np.asarray(rec[k2], dtype=np.float64)[body, hs] for k2 in ("exp_cm", "ach_cm", "sh_cm"))
        e_own = float(np.linalg.norm(exp - ach))
        e_sh = float(np.linalg.norm(sh - ach))
        # persistence: the previous order's achieved movement, the same body and handles
        prev = None
        i = self.eff_steps.index(s) if s in self.eff_steps else -1
        if i > 0:
            r0 = self.lane.eff[self.eff_steps[i - 1]]
            if int(r0.get("began", -1)) < int(rec["began"]):
                prev = np.asarray(r0["ach_cm"], dtype=np.float64)[body, hs]
        if prev is None or not np.isfinite(prev).all():
            return dict(kind="efference", verdict=None, own_order_error_cm=e_own, shuffled_order_error_cm=e_sh,
                        note="no previous order to compare with (the persistence control)")
        e_keep = float(np.linalg.norm(prev - ach))
        return dict(kind="efference", handles=[self.handles[h] for h in hs], own_order_error_cm=e_own,
                    shuffled_order_error_cm=e_sh, persistence_error_cm=e_keep,
                    achieved_cm=float(np.linalg.norm(ach)),
                    verdict=bool(e_own < e_sh and e_own < e_keep), order_began=int(rec["began"]))

    def movement_prediction_intervals(self, d: dict, body: int, tr, att: dict) -> Optional[dict]:
        """The movement verdicts of an act of the hands alone (observer_attribution_v1): the efference record of EVERY
        command in force during it (movement_prediction at each interval's first record: the record of that command's
        whole hold, `window`), each expectation formed where its own command began. EACH COMMAND IS ONE UNIT of the
        Predicted stage (tracker rules v6; independent review of 4342f9194, blocker 2): `commands_judged` counts those
        with a record and `commands_predicted` those whose expectation beat both controls; the tracker tests the units
        against one half with the act's commands as one cluster (tracker.cluster_binom_above). `verdict`, every command
        passing (None when one has no record), is a reading of the act beside the stage, never the stage: its chance
        under no prediction skill falls with the number of commands. The records are the efference head's world-frame
        displacements (observer source review F1: for a finger most of that is the hand being carried), `frame`."""
        if self.lane is None or not self.eff_steps:
            return None
        ivs = att.get("intervals") or []
        if not ivs:
            return dict(kind="efference", verdict=None, commands_judged=0, commands_predicted=0,
                        note="no command interval measured for the act")
        rows = []
        for e in ivs:
            m = self.movement_prediction(d, body, tr, dict(credit_step=e["records"][0], peak_step=e["records"][0]))
            rows.append(dict(command=e["command"], window=e.get("hold_records"), **{k: (m or {}).get(k) for k in (
                "verdict", "own_order_error_cm", "shuffled_order_error_cm", "persistence_error_cm", "achieved_cm",
                "order_began", "note")}))
        verdicts = [r["verdict"] for r in rows]
        judged = [v for v in verdicts if v is not None]
        verdict = None if any(v is None for v in verdicts) else bool(all(verdicts))
        return dict(kind="efference", verdict=verdict, frame="world", intervals=rows,
                    commands_judged=len(judged), commands_predicted=int(sum(1 for v in judged if v)),
                    rule="each command in force during the act is one unit of the Predicted stage: its own "
                         "expectation over the command's whole hold, formed where the command began, beat both controls "
                         "(another body's command; the previous command's movement). `verdict` is every command "
                         "passing: a reading of the act beside the stage, never the stage",
                    note=None if verdict is not None else "a command of the act has no efference record")

    def surprise(self, d: dict, body: int, tr) -> Optional[dict]:
        if self.lane is None or not self.lane.rows:
            return None
        senses = list(self.lane.senses or [])
        idx = [senses.index(s) for s in ("tactile", "vision") if s in senses]
        if not idx:
            return None
        s0 = int(d["step0"])
        lag = self.lane_lag
        rows = self.lane.rows
        during = [rows[s + lag]["next_pe_body"][body, idx] for s in range(s0, int(d.get("step1", s0)) + 3)
                  if s + lag in rows]
        if not during:
            return dict(kind="surprise", verdict=None, note="no learner record at the act")
        before = [rows[s + lag]["next_pe_body"][body, idx] for s in range(s0 - SURPRISE_WINDOW, s0)
                  if s + lag in rows]
        if len(before) < 5:
            return dict(kind="surprise", verdict=None, note="too few learner records before the act")
        med = np.nanmedian(np.asarray(before, dtype=np.float64), axis=0)
        now = np.nanmean(np.asarray(during, dtype=np.float64), axis=0)
        if not (np.isfinite(med).all() and np.isfinite(now).all()):
            return dict(kind="surprise", verdict=None, note="error not measured at the act")
        ratio = float(np.sum(now) / max(float(np.sum(med)), 1e-9))
        return dict(kind="surprise", error_ratio_to_recent_median=ratio, line=1.0, verdict=bool(ratio <= 1.0))

    # ------------------------------------------------------------------ the contact verdict's chance
    def contact_nulls(self, dets: List[dict], body: int, tr, all_dets: Optional[List[dict]] = None) -> None:
        """Score each judged touch act's verdict at its matched no-touch moments (module doc, MATCHED
        MOMENTS) and add the counts to its prediction record: null_moments, null_passed.

        An act's moments are the world records 1, 2 and 3 s (NULL_EVERY_STEPS apart, up to
        MATCH_REACH_STEPS) before and after its onset at the onset's own place in its order
        (order_phase), each given to the EARLIEST act of the same key (contact_key: the row, the
        touching part's skin patches and the head's bit) that has it among its own, over every act
        of that key the window found (all_dets; ties in onset by row, sub-row and steps). So the
        tracker can add up the moments of any set of acts without counting one twice, and an act's
        moments depend only on the acts before it, which every window reads in its look-back. A moment
        counts where it is live and continuous, the lane has a valid contact record for this body and,
        for the key's skin patches and bit, some patch was untouched at the record before and none of
        those starts a touch now. The verdict there is the act's own: the largest probability-to-base-
        rate ratio over those untouched patches, at least CONTACT_LIFT."""
        if self.lane is None:
            return
        states = self.patch_states(tr)
        judged = [d for d in dets if isinstance(((d.get("evidence") or {}).get("prediction") or {}).get("_key"), tuple)]
        if states is None or not judged:
            for d in judged:
                d["evidence"]["prediction"].pop("_key", None)
            return
        live = np.asarray(getattr(tr, "live", np.ones(tr.n, dtype=bool)), dtype=bool)
        cont = np.asarray(getattr(tr, "continuous", np.ones(tr.n, dtype=bool)), dtype=bool)
        steps = np.asarray(tr.step, dtype=np.int64)
        phase = order_phase(tr)
        offsets = [j * NULL_EVERY_STEPS for j in range(-(MATCH_REACH_STEPS // NULL_EVERY_STEPS),
                                                      MATCH_REACH_STEPS // NULL_EVERY_STEPS + 1) if j]
        mine = {id(d) for d in judged}
        pool = list(all_dets) if all_dets is not None else list(dets)
        known = {id(d) for d in pool}
        pool += [d for d in judged if id(d) not in known]
        by_key: Dict[tuple, List[tuple]] = {}
        for d in pool:
            key = d["evidence"]["prediction"]["_key"] if id(d) in mine else self.contact_key(d, tr)
            if key is None:
                continue
            onset = int(tr.step[min(self._onset_index(d, tr), tr.n - 1)])
            by_key.setdefault(key, []).append((onset, str(d.get("sub") or ""), int(d["step0"]), int(d["step1"]), d))
        for key, acts in by_key.items():
            _, patches, bit = key[0], list(key[1]), key[2]
            acts.sort(key=lambda a: a[:4])
            claimed = set()
            for onset, _s, _a, _b, a in acts:
                io = _index_of(steps, onset)
                ph = int(phase[io]) if io >= 0 else -1
                own = []
                for off in offsets:
                    m = _index_of(steps, onset + off)
                    if m < 0 or ph < 0 or int(phase[m]) != ph or m in claimed:
                        continue
                    claimed.add(m)
                    own.append(m)
                if id(a) not in mine:
                    continue                          # an act another window judges: its moments stay its own
                n = k = 0
                for m in own:
                    if m < 1 or not (live[m] and cont[m]):
                        continue
                    eligible = ~states[m - 1, :, bit]
                    if any(eligible[p] and states[m, p, bit] for p in patches):
                        continue                      # a touch starts here: not a no-touch moment
                    rec = self._lane_contact(int(steps[m]), body)
                    if rec is None:
                        continue
                    v = self._contact_verdict(rec, body, patches, bit, eligible)
                    if v is None:
                        continue
                    n += 1
                    k += int(v["verdict"])
                p = a["evidence"]["prediction"]
                p.pop("_key", None)
                p.update(null_moments=int(n), null_passed=int(k), null_every_steps=NULL_EVERY_STEPS,
                         null_reach_steps=MATCH_REACH_STEPS)

    # ------------------------------------------------------------------ intention
    def intention(self, d: dict, body: int, credit: Optional[dict] = None, noun: str = "order") -> Optional[dict]:
        """`noun` names what the notes call the order: 'order' for the credited order of an act (the notes as
        before), 'command' for one interval of an act of the hands alone (intention_intervals)."""
        it = "the credited order" if noun == "order" else "this " + noun
        author = credit["author"] if credit else d.get("author", "unknown")
        seq = int(credit["order_seq"]) if credit else int(d.get("order_seq", -2))
        at = int(credit.get("credit_step", credit["peak_step"])) if credit else int(d["step0"])
        if not self.latches:
            return dict(author=author, verdict=None if author in MODEL_SELECTED else False,
                        note="no planner records")
        k = bisect.bisect_right(self.latch_steps, at) - 1
        if k < 0:
            return dict(author=author, verdict=None if author in MODEL_SELECTED else False,
                        note="no planner record before the act")
        rec = self.latches[k]
        plan = rec.get("plan") or {}
        gaps = plan.get("pred_wellness_gaps") or {}
        outcome = (gaps.get("outcome") or [None] * (body + 1))
        outcome = outcome[body] if body < len(outcome) else None
        gap = (gaps.get("gap_to_best_alt") or [None] * (body + 1))
        gap = gap[body] if body < len(gap) else None
        reached = bool(rec.get("reached_body", False))
        rec_seq = rec.get("order_seq")
        same_order = rec_seq is not None and int(rec_seq) == seq
        chosen = author in MODEL_SELECTED
        if chosen and not same_order:
            return dict(author=author, latch_step=int(rec.get("step", -1)), same_order=False, verdict=None,
                        order_seq=seq, note="the planner left no record for %s" % it)
        verdict = bool(chosen and outcome == "best" and reached)
        note = ("the planner wrote the %s, its evaluator ranked it best by more than the tie band, "
                "and it reached the body" % noun if verdict else
                "the planner wrote the %s but could not tell its candidates apart (tied within the band)" % noun
                if chosen and outcome == "tied" else
                "%s was not the planner's" % it if not chosen else
                "the planner's choice did not reach the body" if not reached else
                "the planner ranked another candidate higher")
        return dict(author=author, latch_step=int(rec.get("step", -1)), same_order=same_order, order_seq=seq,
                    outcome=outcome, gap_to_best_alternative=gap, tie_band=gaps.get("tie_band"),
                    reached_body=reached, verdict=verdict, note=note)

    def planner_record(self, seq: int, at: int) -> Optional[dict]:
        """The planner's latch record of order `seq`: the latest latch at or before world step `at`, when it is that
        order's own (orders change only at latches, and every latch leaves a record); None otherwise."""
        k = bisect.bisect_right(self.latch_steps, int(at)) - 1
        if k < 0:
            return None
        rec = self.latches[k]
        return rec if rec.get("order_seq") is not None and int(rec["order_seq"]) == int(seq) else None

    def selection(self, body: int, e: dict) -> dict:
        """THE PLANNER'S OWN SELECTION OF ONE ORDER, AS RECORDED BEFORE EXECUTION (observer_attribution_v1; the
        operator's addition of Oct 8, 2026): its writer; for a planner order, the planner search's criterion, each
        candidate's predicted distance to the goal entry's tokens (`goal_distance`, 0 = the predicted next tokens equal
        the goal's), for the chosen order and every alternative recorded with it; `rank` (1 = the lowest distance of
        those recorded) and `margin` (the lowest alternative's distance minus the chosen's): reported, NEVER JUDGED, since
        no noise band of this distance has been measured (independent review of 4342f9194, fix 3: a band of zero called
        96761 nearest by 0.00097 on distances near 0.60; no band is invented here); the hand stage's own
        cost before and after its search (`hand_cost_first`, `hand_cost_selected`: the frozen fit's predicted
        finger-pose error after 250 ms, mean squared over the finger handles in units of 1.80 cm and 11 degrees); the
        planner's predicted change of each sense (root-mean-square token change); and, kept apart, the approximate
        wellness evaluator's outcome the decisive test reads ('tied' within its band of 0.0734). A selection is never
        read as intention."""
        out = dict(command=e["command"], writer=e["writer"], planner=e["writer"] in MODEL_SELECTED)
        if not out["planner"]:
            out["note"] = "written by %s, not by the planner" % writer_words(e["writer"])
            return out
        rec = self.planner_record(e["command"], e["records"][0])
        if rec is None:
            out["note"] = "the planner left no latch record for this command"
            return out
        plan = rec.get("plan") or {}

        def at_body(v):
            return v[body] if isinstance(v, list) and body < len(v) else None
        gd = plan.get("goal_distance") or {}
        alts = [a for a in (plan.get("alternatives") or []) if a in gd]
        chosen = at_body(gd.get("chosen"))
        alt = {a: at_body(gd[a]) for a in alts}
        hand = plan.get("hand_optimizer") or {}
        gaps = plan.get("pred_wellness_gaps") or {}
        out.update(latch_step=int(rec.get("step", -1)), reached_body=rec.get("reached_body"),
                   goal_distance=dict(chosen=chosen, **alt),
                   hand_cost_first=at_body(hand.get("first")), hand_cost_selected=at_body(hand.get("selected")),
                   hand_stage_valid=at_body(hand.get("valid")),
                   predicted_change={k: at_body(v) for k, v in (plan.get("pred_change") or {}).items()},
                   wellness_outcome=at_body(gaps.get("outcome")), wellness_tie_band=gaps.get("tie_band"))
        vals = [v for v in alt.values() if v is not None]
        if chosen is None or not vals:
            out["note"] = "the latch record holds no goal distances for this body"
        else:
            out["margin"] = float(min(vals) - chosen)
            out["rank"] = 1 + sum(1 for v in vals if v < chosen)
        return out

    def four_stages(self, d: dict, body: int, att: dict, pred: Optional[dict]) -> dict:
        """THE FOUR STAGES OF AN ACT OF THE HANDS ALONE, EACH ON ITS OWN EVIDENCE (observer_attribution_v1; the
        operator's addition of Oct 8, 2026). None of them is the decisive test of 'selected intentionally'
        (intention_intervals, unchanged with its tie band), and none of them reads a planner-written order as intentional:
          planner_selected   NO when a command in force during the act was not the planner's; when every one was,
                             NOT JUDGED: the planner's own goal distances, with each chosen command's rank and margin,
                             are reported, but no noise band of that distance has been measured, so whether the chosen
                             command was nearer the goal than its alternatives is not judged (review fix 3)
          predicted          each command's own efference expectation against both controls over the command's whole
                             hold (evidence.prediction; world frame): the counts of commands judged and predicted, and
                             every command passing as the act's reading
          achieved           the act's measure moved the way each command's hand joint goal asked over the command's
                             WHOLE HOLD (from the record before its first record to its last: the window its efference
                             record grades, so predicted and achieved read one window per command; review fix 2)
          improved           not readable from one act: it needs the same finger's acts across checkpoints or occasions
        Each verdict is True, False, or None where its evidence is missing or, for planner selected, not judged."""
        ivs = att.get("intervals") or []
        sel = [self.selection(body, e) for e in ivs]
        if not sel:
            planner_v, planner_why = None, "no command interval measured for the act"
        elif any(not s["planner"] for s in sel):
            planner_v, planner_why = False, "a command in force during the act was not the planner's"
        else:
            planner_v, planner_why = None, ("not judged: every command in force during the act was the planner's, and "
                                            "no noise band of the planner's goal distance has been measured, so the "
                                            "chosen command's rank and margin are reported, not judged")
        ach = []
        for e in ivs:
            want, made = e.get("hold_asked_change"), e.get("hold_change")
            ach.append(dict(command=e["command"], window=e.get("hold_records"), window_whole=e.get("hold_whole"),
                            asked_change=want, change_made=made, asked=e.get("hold_asked"),
                            share_of_asked=(float(made / want) if (want not in (None, 0.0) and made is not None) else None)))
        asked = [a["asked"] for a in ach]
        achieved_v = (False if "opposite" in asked else (None if (not asked or any(x is None for x in asked)) else True))
        return dict(
            planner_selected=dict(verdict=planner_v, why=planner_why, commands=sel,
                                  planner_commands=sum(1 for s in sel if s["planner"]), commands_in_act=len(sel),
                                  rule="no when a command in force during the act was not the planner's; otherwise not "
                                       "judged until the goal distance's noise band is measured; a selection, never "
                                       "read as intention"),
            predicted=dict(verdict=None if pred is None else pred.get("verdict"),
                           kind=None if pred is None else pred.get("kind"),
                           frame=None if pred is None else pred.get("frame"),
                           commands_judged=None if pred is None else pred.get("commands_judged"),
                           commands_predicted=None if pred is None else pred.get("commands_predicted"),
                           rule="each command's own expectation over its whole hold beat both controls "
                                "(evidence.prediction); the verdict is every command passing, a reading of the act"),
            achieved=dict(verdict=achieved_v, commands=ach,
                          rule="the act's measure moved the way each command's hand joint goal asked, over the "
                               "command's whole hold (the window its efference record grades), command by command"),
            improved=dict(verdict=None, note="one act cannot show improvement with experience: it needs this finger's "
                                             "acts compared across checkpoints or occasions"))

    def intention_intervals(self, d: dict, body: int, att: dict) -> Optional[dict]:
        """The intention verdict of an act of the hands alone (observer_attribution_v1): True only when EVERY order in
        force during it was the planner's (or the qualified self-choice actor's) decisive choice that reached the body
        (intention, per order) and none of them asked the other way (its hand joint goal against the movement made
        during it, command_intervals `asked`); False when any order fails either; None when an order's planner record
        is missing and none failed. One order is never enough for the act: its other orders made part of the movement."""
        ivs = att.get("intervals") or []
        if not ivs:
            return dict(author=att.get("writer"), verdict=None, note="no command interval measured for the act")
        rows, reasons = [], []
        for e in ivs:
            r = self.intention(d, body, dict(author=e["writer"], order_seq=e["command"], credit_step=e["records"][0],
                                             peak_step=e["records"][0]), noun="command")
            note = (r.get("note") if e["writer"] in MODEL_SELECTED
                    else "written by %s, not by the planner" % writer_words(e["writer"]))
            row = dict(command=e["command"], writer=e["writer"], verdict=r.get("verdict"), outcome=r.get("outcome"),
                       reached_body=r.get("reached_body") if e["writer"] in MODEL_SELECTED else None,
                       asked=e.get("asked"), note=note)
            if e.get("asked") == "opposite":
                row["verdict"] = False
                reasons.append("command %d's hand joint goal asked for the other direction" % e["command"])
            elif r.get("verdict") is False:
                reasons.append("command %d: %s" % (e["command"], note))
            rows.append(row)
        verdicts = [r["verdict"] for r in rows]
        verdict = False if any(v is False for v in verdicts) else (None if any(v is None for v in verdicts) else True)
        note = ("every command in force during the act was a decisive planner choice that reached the body and asked "
                "for this direction" if verdict else
                "; ".join(reasons) if verdict is False else "a planner command of the act left no planner record")
        return dict(author=att.get("writer"), writer_class=att.get("writer_class"), verdict=verdict, intervals=rows,
                    rule="every command in force during the act, never one command for the whole act", note=note)

    def _wellness_change(self, s0: int, s1: int, body: int) -> Optional[dict]:
        """The grounded wellness channels' change of one body from world record s0 to s1 (the lane's
        records s0 + lag and s1 + lag), per channel and summed; None without both records."""
        if self.lane is None:
            return None
        lag = self.lane_lag
        a, b = self.lane.rows.get(int(s0) + lag), self.lane.rows.get(int(s1) + lag)
        if not a or not b or "wellness" not in a or "wellness" not in b:
            return None
        ch = {k: float(b["wellness"][k][body] - a["wellness"][k][body])
              for k in a["wellness"] if k in b["wellness"]
              and np.isfinite(a["wellness"][k][body]) and np.isfinite(b["wellness"][k][body])}
        return dict(change=ch, total_change=float(sum(ch.values())) if ch else None)

    def wellness(self, d: dict, body: int) -> Optional[dict]:
        return self._wellness_change(int(d["step0"]), int(d["step1"]), body)

    # ------------------------------------------------------------------ the wellness change's reference
    def wellness_references(self, dets: List[dict], body: int, tr, all_dets: Optional[List[dict]] = None) -> None:
        """THE REFERENCES FOR THE WELLNESS CHANGE UNDER A BEHAVIOUR (OBSERVER_DESIGN.md section 9, entry 41;
        analysis v5): the same grounded wellness change over stretches of the act's length on the same
        body that END 1 to MATCH_REACH_STEPS steps (up to 3 s) BEFORE THE ACT STARTS, one a second
        (NULL_EVERY_STEPS apart: three stretches), each starting at the act's own place in its order
        (order_phase), over continuous live records. Two sets:
          ALL      every such stretch, whatever happens in it (other acts of the same row included):
                   all_n, all_sum, all_sq
          WITHOUT  the stretches that overlap no act of that row on this body (all_dets: every act the
                   window found): reference_n, reference_sum, reference_sq
        Both, not the second alone (the coordinator's decision of 2026-09-25, from the reading rules).
        BEFORE THE ACT, NOT AROUND IT (independent review of 2026-09-25, finding 1; module doc, MATCHED
        MOMENTS): a stretch after the act would carry the act's own consequences for wellness, and would
        need acts the window has not finished; before it, every act a stretch can overlap started before
        the act, so each act's references depend on the act alone, not on the digestion window. An act
        longer than REFERENCE_MAX_STEPS gets none (its stretches would start before the window's look-back).
        THE ACT'S OWN PLACE IN ITS ORDER (2026-09-25, 7:30 PM). The summed wellness moves in a sawtooth
        inside every order: the one-step change is -0.012 at the record after the latch, +0.019 at the
        next and about -0.003 at the other three (segment 40's 9:33 AM run; segment 39's 6:04 PM run the
        same), so a stretch's change depends on where in an order it starts. Stretches starting at an
        order's first record changed by -0.0125, +0.0066, +0.0059, +0.0023, +0.0009 for lengths 1-5 (the
        same pattern for 6-10), stretches starting at every step by about 0 (+0.0000 to +0.0005). The
        first references started only at steps that are multiples of 20, always an order's first record,
        and read +0.0106 to +0.017 per row on the lineage at every moment and without the act alike. So
        each stretch starts a whole number of seconds before the act and at the act's own place in its
        order."""
        if self.lane is None or not dets:
            return
        live = np.asarray(getattr(tr, "live", np.ones(tr.n, dtype=bool)), dtype=bool)
        cont = np.asarray(getattr(tr, "continuous", np.ones(tr.n, dtype=bool)), dtype=bool)
        steps = np.asarray(tr.step, dtype=np.int64)
        phase = order_phase(tr)
        spans: Dict[str, list] = {}
        for d in list(all_dets if all_dets is not None else []) + list(dets):
            spans.setdefault(d["cap"], []).append((int(d["step0"]), int(d["step1"])))
        # an act [a0, a1] overlaps a stretch [s0, s1] when a0 <= s1 and a1 >= s0: among the acts starting
        # at or before s1 (a prefix of the sorted starts), the latest end decides
        over = {}
        for cap, sp in spans.items():
            sp = np.asarray(sorted(set(sp)), dtype=np.int64).reshape(-1, 2)
            over[cap] = (sp[:, 0], np.maximum.accumulate(sp[:, 1]))
        for d in dets:
            s, e = int(d["step0"]), int(d["step1"])
            length = e - s
            i0 = _index_of(steps, s)
            ph = int(phase[i0]) if i0 >= 0 else -1
            got = dict(n=0, sum=0.0, sq=0.0, n_all=0, sum_all=0.0, sq_all=0.0)
            if 0 < length <= REFERENCE_MAX_STEPS and ph >= 0:
                starts, ends = over[d["cap"]]
                # stretches ending 1 to MATCH_REACH_STEPS before the act: 20 j in (length, length + reach]
                for j in range(length // NULL_EVERY_STEPS + 1, (length + MATCH_REACH_STEPS) // NULL_EVERY_STEPS + 1):
                    m = _index_of(steps, s - j * NULL_EVERY_STEPS)
                    if m < 0 or int(phase[m]) != ph:
                        continue
                    end = m + length
                    if end >= tr.n or int(steps[end]) != int(steps[m]) + length:
                        continue                          # a gap in the record inside the stretch
                    if not live[m:end + 1].all() or not cont[m + 1:end + 1].all():
                        continue
                    s0, s1 = int(steps[m]), int(steps[end])
                    w = self._wellness_change(s0, s1, body)
                    if not w or w.get("total_change") is None:
                        continue
                    x = float(w["total_change"])
                    got["n_all"] += 1                     # ALL: every stretch of that length
                    got["sum_all"] += x
                    got["sq_all"] += x * x
                    k = int(np.searchsorted(starts, s1, side="right"))
                    if k > 0 and int(ends[k - 1]) >= s0:
                        continue                          # an act of this row is there
                    got["n"] += 1
                    got["sum"] += x
                    got["sq"] += x * x
            ev = d.setdefault("evidence", {})
            wl = ev.get("wellness") or dict(change=None, total_change=None)
            wl.update(reference_n=got["n"], reference_sum=got["sum"], reference_sq=got["sq"],
                      all_n=got["n_all"], all_sum=got["sum_all"], all_sq=got["sq_all"],
                      reference_every_steps=NULL_EVERY_STEPS, reference_reach_steps=MATCH_REACH_STEPS)
            ev["wellness"] = wl

    def attach(self, dets: List[dict], body: int, tr, all_dets: Optional[List[dict]] = None) -> None:
        """Every record of the join for each act; all_dets, every act the window found (finalized or
        not), decides which act a matched no-touch moment belongs to (contact_nulls) and keeps the
        wellness references' stretches without the act off the row's acts (wellness_references).

        An act of the hands alone (is_hand_act) gets `evidence.attribution` (command_intervals), a credit naming no
        single order (interval_credit), its movement verdict over every order (movement_prediction_intervals) and its
        intention verdict over every order (intention_intervals); its `author` and `provenance` become the act's
        writer and writer class over those orders ('mixed' when they differ), and the first record's, which they were
        until observer_attribution_v1, are kept as `first_author` and `first_provenance`."""
        for d in dets:
            cap = d["cap"]
            if self.is_hand_act(d):
                att = self.command_intervals(d, tr)
                credit = self.interval_credit(att)
                if cap in CONTACT_CAPS:
                    pred = self.contact_prediction(d, body, tr)
                elif cap in MOVE_CAPS:
                    pred = self.movement_prediction_intervals(d, body, tr, att)
                else:
                    pred = self.surprise(d, body, tr)
                d["evidence"] = dict(prediction=pred, intention=self.intention_intervals(d, body, att),
                                     wellness=self.wellness(d, body), credit=credit, attribution=att,
                                     stages=self.four_stages(d, body, att, pred))
                d["first_author"], d["first_provenance"] = d.get("author"), d.get("provenance")
                if att.get("intervals"):
                    d["author"], d["provenance"] = credit["author"], credit["provenance"]
                continue
            credit = self.credit(d, tr)
            if cap in CONTACT_CAPS:
                pred = self.contact_prediction(d, body, tr)
            elif cap in MOVE_CAPS:
                pred = self.movement_prediction(d, body, tr, credit)
            else:
                pred = self.surprise(d, body, tr)
            d["evidence"] = dict(prediction=pred, intention=self.intention(d, body, credit),
                                 wellness=self.wellness(d, body), credit=credit)
        self.contact_nulls(dets, body, tr, all_dets)
        self.wellness_references(dets, body, tr, all_dets)
