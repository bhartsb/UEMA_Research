# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Layer 2, primitive events: the observer's alphabet (addendum v4 sections 22 and 34).

Each primitive is a dict with its type, the track indices i0..i1 (inclusive) and steps it spans,
and measurements. The thresholds below DEFINE the events; they are the measuring instrument's
settings, validated by the scripted worlds (validate.py) and given chance rates by the babble
baselines (baseline.py). None of them is a goal for the learner: nothing here reaches it.

  contact        one entity pair touching over consecutive steps (a break of up to
                 CONTACT_MERGE_STEPS is the same touch): parts, kinds, force, place
  curl           a hand closing or opening by at least CURL_MIN_DEG (mean of four fingers)
  digit          one digit bending or straightening by at least DIGIT_MIN_DEG
  wrist          a wrist rotation / flexion / deviation of at least WRIST_MIN_DEG
  hand_view      a hand inside an eye's field of view (and inside the fovea)
  fixation       an eye's foveal bundle staying on one target (object, body part, painted motif
                 cell: the centre ray's thing, else the bundle's majority) for FIX_MIN_STEPS
                 records, with its dwell, share of rays, aim error, distance and the change from
                 the previous target
  eye_move       an eye-in-head change of at least EYE_MIN_DEG between steps, with whether the
                 world's saccade generator or reflex moved it
  head_turn      the head turning at least HEAD_MIN_DEG within HEAD_WINDOW_S
  object_motion  a free object or the mobile moving (speed or spin)
  lift / drop    a free object leaving its support in a hand's contact / falling out of it
  support_shift  the body's centre of mass moving at least COM_SHIFT_M over the floor within
                 COM_SHIFT_MAX_S and staying shifted for COM_HOLD_S
  posture        the trunk's roll or sit state changing (back, side, belly; lying, upright)
  transient      a 10 ms-resolution event: head jolt, spin, impact (a contact force RISING),
                 loss of support (outside force staying low)
  approach       a hand's distance to an object falling by at least APPROACH_MIN_M
  caregiver      the adult touching the infant, arriving or leaving (an external cue)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .features import (KIND_ART, KIND_CAREGIVER, KIND_CEILING, KIND_FLOOR, KIND_OBJECT, KIND_SELF,
                       KIND_STATIC, KIND_WALL, Track)
from .kinematics import DIGITS, SIDES

CURL_MIN_DEG = 10.0          # a hand closing: four-finger mean curl changes this much between turns
DIGIT_MIN_DEG = 15.0         # one digit's own bend
WRIST_MIN_DEG = 20.0
EYE_MIN_DEG = 5.0            # a saccade-sized eye-in-head change between two 50 ms steps
HEAD_MIN_DEG = 10.0
HEAD_WINDOW_S = 0.5
# A fixation holds the foveal bundle on one target (Track.foveal_target) for at least 100 ms: two
# consecutive 50 ms records. Measured 2026-09-24 on 35,800 production body-steps (segments 38 and
# 39), centre-ray holds on objects, body parts, the caregiver and the panels: of 864 one-record
# holds, 214 (25%) were entered and left with the gaze turning 5 degrees or more per step - the
# ray swept across the target and never paused. Two-record holds were still (under 5 degrees
# between their records) in 414 of 620 (67%); a longer minimum buys no stillness (3 records: 320
# of 523, 61%; 4: 259 of 453, 57%), because a target as large as the caregiver keeps the ray while
# the gaze moves across it. One record carries no dwell at all; two are the shortest with one.
FIX_MIN_STEPS = 2
FIX_STEP_S = 0.05
OBJ_MOVE_MPS = 0.02          # an object moving at 2 cm/s or more
OBJ_SPIN_RADPS = 0.5
LIFT_MIN_M = 0.01            # a free object 1 cm above its resting height
DROP_SPEED_MPS = 0.2
APPROACH_MIN_M = 0.03
APPROACH_MAX_S = 2.0
# A break in one pair's contact of at most this many steps (100 ms) is the same touch. Measured
# 2026-09-24 on H84 (2,000 production body-steps): 28 % of re-contacts of the same two parts came
# after a break of one or two steps, and 36 % of contact intervals lasted a single step.
CONTACT_MERGE_STEPS = 2
# The support centre is the wrong signal for a weight shift: the force-weighted centre of the
# floor contacts moved 2.8 cm at the median between two 50 ms steps (production, 4,000 body-steps)
# because limb contacts carry large, flickering forces. The centre of mass moved 0.35 cm at the
# median over five steps (99.9th percentile 1.7 cm), so a 2 cm move of it is a real shift.
COM_SHIFT_M = 0.02
COM_SHIFT_MAX_S = 1.0
COM_HOLD_S = 0.25            # ...and at least half the shift is still there 250 ms later
G = 9.81
# Fast-lane events, each at a level the resting and moving body does not reach by itself
# (measured 2026-09-24 per 10 ms frame: 4,000 production body-steps, 20,000 random-order babble):
JOLT_MS2 = 19.6              # head acceleration 2 g away from gravity: 0.13 % of production frames,
                             # 0.36 % of babble frames (the old 5 m/s2 line: 6.9 % of production)
SPIN_RADPS = 3.0             # head angular speed: no production frame reached it, 0.1 % of babble
IMPACT_RISE_FRACTION = 1.0   # the largest single contact force RISING by a body weight within one
                             # frame; the level is no impact: the largest RESTING contact carries
                             # 0.44 body weights at the median, so the old "above half a body
                             # weight" line marked 34 % of production frames
SUPPORT_LOSS_FRACTION = 0.25  # force from outside the body below a quarter of the body weight...
SUPPORT_LOSS_FRAMES = 3       # ...for 30 ms, right after it carried at least three quarters: the
SUPPORT_HELD_FRACTION = 0.75  # between-frame swings (a half-weight drop in 7 % of frames) are not

# posture bands of the trunk's roll (degrees from lying on the back, kinematics.py)
BACK_MAX_DEG = 30.0
SIDE_MIN_DEG, SIDE_MAX_DEG = 60.0, 120.0
BELLY_MIN_DEG = 150.0
UPRIGHT_MAX_DEG = 45.0       # the trunk's head-ward axis within 45 degrees of vertical


def _intervals(mask: np.ndarray, cont: np.ndarray) -> List[tuple]:
    """(i0, i1) runs of True, broken where the track is not continuous."""
    out = []
    i0 = None
    for i, v in enumerate(mask):
        if v and i0 is not None and not cont[i]:
            out.append((i0, i - 1))
            i0 = None
        if v and i0 is None:
            i0 = i
        elif not v and i0 is not None:
            out.append((i0, i - 1))
            i0 = None
    if i0 is not None:
        out.append((i0, len(mask) - 1))
    return out


# WHERE A TURN STARTS (detector v9, 2026-09-25; OBSERVER_DESIGN.md section 9, "primitives._turns
# starts a turn at the FIRST low"). Until v8 a turn began at the lowest record of the whole stretch
# since the previous turn's peak (np.argmin: the global low, and the first of equal lows), while its
# own comment said the last low. Measured on production (segment 39 and two finished segment-40
# runs, 60,000 body-steps; docs/gain_of_function_v4/agents/observer_detectors_2026-09-25.md): exact
# flat lows never occur (a recorded joint always leaves its start value in the next record), but
# the fingers are seldom still: in 1.7% of the 20,093 finger turns the last record within 0.5
# degrees of the start level came more than 0.5 s after the recorded start (3.4% at 1 degree),
# because the finger bent 5-14 degrees and came back before the movement that reached the peak.
# Example: segment 39, body 0, left index from step 391,393: 27.2 -> 38.3 -> 33.1 -> 41.2 -> 27.6
# -> 44.6 degrees, recorded as one 17.4-degree bend starting 2.85 s before the bend to 44.6 began.
# A run now starts where the movement that reaches its extreme starts (_run_start). A resting
# joint's record-to-record change is small beside TURN_REST_FRACTION of a turn's line: limp bodies
# after 20 s moved a finger at most 0.28 degrees and a wrist 0.10 degrees in one 50 ms record
# (99.9th percentiles 0.21 and 0.043), against 0.75 degrees for a 15-degree finger turn, 0.5 for
# the hand's 10-degree closing and 1.0 for a 20-degree wrist turn.
TURN_REST_FRACTION = 0.05
# The act timescale: the catalog's 250 ms - 2 s for the finger, wrist, spread and cupping rows. The
# detectors ask a turn's fastest stretch to move the row's full line within it (line_s; detectors.
# ACT_MAX_RISE_S, "WHERE THE TIMESCALE IS MEASURED"); the segmenter's hand onsets ask the whole turn from
# rest to rise 10-90% within it (rise_s; segmenter.hand_turns_from_rest): an onset marks where the turn
# starts, so a turn whose slow lead-in starts it is no act onset there.
ACT_MAX_RISE_S = 2.0
# WHEN A TURN IS KNOWN (detector v9, 2026-09-25: the statistics track's window test, 7 of 7,927 acts of
# segment 40's 9:33 AM run found only when the run was digested in windows, whatever the look-back;
# agents/observer_detectors_2026-09-25.md, item 3b). A run's extreme, and with it the run's start
# (_run_start reads back from the peak), is known only once the series turns back by the run's line,
# stops, or the record breaks: until then later records can still carry the run further. A window
# ending inside such a run gave the digester a provisional turn, and acts built on it were final.
# A turn is now CONFIRMED by a reversal of at least its line, by a break in the record, or by
# TURN_HOLD_RECORDS records (ACT_MAX_RISE_S, the longest act's rise) in which the series never goes
# TURN_REST_FRACTION of the line past the level it last reached - it may pause, or move back by less
# than the line. The next run then begins once the series moves the line from its lowest (highest)
# record after that extreme, so a movement from a lower point after the pause is found as its own
# run. A turn not yet confirmed where the record ends is OPEN (turns_confirmed's `settled` -1; the
# primitive's `open_end`); the digester does not finalize an act built on an open turn until a later
# window confirms it, or the record is closed. A run stays open only while it keeps advancing (at
# least TURN_REST_FRACTION of its line every TURN_HOLD_RECORDS), so the wait is bounded.
TURN_HOLD_RECORDS = int(round(ACT_MAX_RISE_S / 0.05))       # records are 50 ms (5 frames of 10 ms)


def rise_s(tr: Track, x: np.ndarray, i0: int, i1: int, ch: float) -> float:
    """A turn's 10-90% rise in seconds: from the last record within 10% of its change of its start
    to the first within 10% of its extreme (HX1's rise since detector v8)."""
    seg = np.sign(ch) * (np.asarray(x[i0:i1 + 1], dtype=np.float64) - float(x[i0]))
    j90 = int(np.argmax(seg >= 0.9 * abs(ch)))
    low = np.nonzero(seg[:j90 + 1] <= 0.1 * abs(ch))[0]
    j10 = int(low[-1]) if low.size else 0
    return tr.seconds(i0 + j10, i0 + j90)


def line_s(tr: Track, x: np.ndarray, i0: int, i1: int, ch: float, line: float) -> float:
    """The shortest stretch of the turn i0..i1 over which the series moves `line` its way (the sign of
    ch), in seconds; infinity when no stretch does. The act timescale's measure (detectors.ACT_MAX_RISE_S,
    ACT_LINE_SHARE): it finds the fast movement wherever it lies inside a turn, so a slow lead-in or tail
    that the turn's start and extreme include does not lengthen it. For each record b the latest earlier
    record a with a value at most x[b] - line is one of the records lower than every record after it (a
    stack of rising values), found by bisection: O(n log n)."""
    y = np.sign(ch) * np.asarray(x[i0:i1 + 1], dtype=np.float64)
    best, stack = float("inf"), []
    for b in range(len(y)):
        v = y[b] - float(line)
        lo, hi = 0, len(stack)
        while lo < hi:
            mid = (lo + hi) // 2
            if y[stack[mid]] <= v:
                lo = mid + 1
            else:
                hi = mid
        if lo:
            best = min(best, tr.seconds(i0 + stack[lo - 1], i0 + b))
        while stack and y[stack[-1]] >= y[b]:
            stack.pop()
        stack.append(b)
    return best


def _run_start(y: np.ndarray, min_change: float, rest: float) -> int:
    """Index where the movement that reaches y's last element (a rise to its peak) starts: the
    latest record from which the rise to the peak still covers min_change, moved back to the low
    it rose from (a reversal of at most `rest` does not end the low), then to that low's LAST
    record within `rest` of it: where the joint leaves the low for good. The rise from the
    returned record to the peak is at least min_change."""
    s = int(np.nonzero(y[:-1] <= y[-1] - min_change)[0][-1])
    lo = s
    j = s - 1
    while j >= 0 and y[j] <= y[lo] + rest:
        if y[j] < y[lo]:
            lo = j
        j -= 1
    within = np.nonzero(y[lo:s + 1] <= y[lo] + rest)[0]
    return lo + int(within[-1])


def turns_confirmed(x: np.ndarray, cont: np.ndarray, min_change: float,
                    hold: int = TURN_HOLD_RECORDS) -> List[tuple]:
    """Runs of a series that rise or fall by at least min_change, found with hysteresis (a
    reversal smaller than min_change does not end a run): (i0, i1, change, settled), change =
    x[i1] - x[i0]. Breaks at discontinuities and NaN.

    i1 is the run's extreme (its first record at the peak of a rise, the trough of a fall), where
    the search for the next run begins. i0 is the LAST low before that peak (the last high before
    that trough): where the movement that reaches the extreme starts (_run_start; detector v9, the
    note above). Runs therefore no longer tile the record: a pause, or back-and-forth movement
    smaller than min_change, lies between one run's extreme and the next run's start.

    settled is the record that confirmed the run (WHEN A TURN IS KNOWN, above): the first record
    min_change back from the extreme, the `hold`-th record without progress (a record more than
    TURN_REST_FRACTION of min_change past the level the run last reached), or the last record before
    a break; -1 when the series ends first (an OPEN run, which later records may still carry
    further). A run confirmed within x is the same run in any longer series that begins with x."""
    out = []
    n = len(x)
    rest = TURN_REST_FRACTION * float(min_change)
    i = 0
    while i < n:
        if not np.isfinite(x[i]):
            i += 1
            continue
        j0 = i
        lo_i = hi_i = i
        direction = 0
        level, last = float(x[i]), i          # the run's progress: the level it last reached, and where
        settled = -1
        k = i + 1
        while k < n and np.isfinite(x[k]) and cont[k]:
            if x[k] > x[hi_i]:
                hi_i = k
            if x[k] < x[lo_i]:
                lo_i = k
            if direction == 0:
                # a run begins once the series moves min_change from its lowest (highest) record
                # since the search began: after a reversal that is the same as from the previous
                # extreme, where the search begins; after a hold, or where the record begins, the
                # series may first have moved the other way by less than the line
                if x[k] - x[lo_i] >= min_change:
                    direction, level, last = 1, float(x[k]), k
                elif x[hi_i] - x[k] >= min_change:
                    direction, level, last = -1, float(x[k]), k
            elif direction == 1:
                if x[k] > level + rest:
                    level, last = float(x[k]), k
                if x[hi_i] - x[k] >= min_change:
                    settled = k
                    break
            else:
                if x[k] < level - rest:
                    level, last = float(x[k]), k
                if x[k] - x[lo_i] >= min_change:
                    settled = k
                    break
            if direction != 0 and k - last >= hold:
                settled = k
                break
            k += 1
        if direction != 0 and settled < 0 and k < n:
            settled = k - 1                   # the record breaks: the run ends with it
        if direction == 1:
            start = j0 + _run_start(np.asarray(x[j0:hi_i + 1], dtype=np.float64), min_change, rest)
            out.append((start, hi_i, float(x[hi_i] - x[start]), settled))
            i = hi_i
        elif direction == -1:
            start = j0 + _run_start(-np.asarray(x[j0:lo_i + 1], dtype=np.float64), min_change, rest)
            out.append((start, lo_i, float(x[lo_i] - x[start]), settled))
            i = lo_i
        else:
            i = k if k > i else i + 1
    return out


def _turns(x: np.ndarray, cont: np.ndarray, min_change: float) -> List[tuple]:
    """turns_confirmed's runs as (i0, i1, change), confirmed or open alike."""
    return [(a, b, c) for a, b, c, _ in turns_confirmed(x, cont, min_change)]


def _ev(tr: Track, type_: str, i0: int, i1: int, **kw) -> dict:
    e = dict(type=type_, i0=int(i0), i1=int(i1), step0=int(tr.step[i0]), step1=int(tr.step[i1]),
             t0=float(tr.sim_time[i0]), duration_s=float(tr.sim_time[i1] - tr.sim_time[i0]))
    e.update(kw)
    return e


def entity_class(tr: Track, e: int) -> str:
    """A short class for an entity: its body group for the infant, else its kind / object kind."""
    e = int(e)
    k = int(tr.ent.kind[e])
    if k == KIND_SELF:
        return tr.ent.group[e]
    if k in (KIND_OBJECT, KIND_CAREGIVER):
        return tr.ent.group[e]
    return ["self", "object", "caregiver", "floor", "wall", "art", "ceiling", "static"][k]


def merge_breaks(on: np.ndarray, cont: np.ndarray, gap: int) -> np.ndarray:
    """on with every break of at most `gap` steps between two True runs filled, where the record
    is continuous across the break."""
    out = np.asarray(on, dtype=bool).copy()
    idx = np.nonzero(out)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1 and bool(np.all(cont[a + 1:b + 1])):
            out[a + 1:b] = True
    return out


def contacts(tr: Track) -> List[dict]:
    """Every entity pair's contact intervals; a break of up to CONTACT_MERGE_STEPS is one touch."""
    out = []
    pairs = {}
    for j in range(len(tr.c_row)):
        a, b = int(tr.c_e[j, 0]), int(tr.c_e[j, 1])
        pairs.setdefault((a, b), []).append(j)
    for (a, b), js in pairs.items():
        on = np.zeros(tr.n, dtype=bool)
        fmax = np.zeros(tr.n)
        pos = np.full((tr.n, 3), np.nan)
        for j in js:
            r = int(tr.c_row[j])
            on[r] = True
            fmax[r] = max(fmax[r], float(tr.c_fmax[j]))
            pos[r] = tr.c_pos[j]
        joined = merge_breaks(on, tr.continuous, CONTACT_MERGE_STEPS)
        for i0, i1 in _intervals(joined, tr.continuous):
            touching = on[i0:i1 + 1]
            out.append(_ev(tr, "contact", i0, i1, a=a, b=b, a_class=entity_class(tr, a),
                           b_class=entity_class(tr, b),
                           self_self=bool(tr.ent.is_self[a] and tr.ent.is_self[b]),
                           peak_force_n=float(fmax[i0:i1 + 1].max()),
                           mean_force_n=float(fmax[i0:i1 + 1][touching].mean()),
                           pos=[float(v) for v in pos[i0]],
                           breaks=int((~touching).sum()),
                           open_start=bool(i0 == 0), open_end=bool(i1 == tr.n - 1)))
    return out


def curls(tr: Track) -> List[dict]:
    """The hands' turns: closing (curl), one digit (digit), one wrist axis (wrist). `open_end`: the
    turn was not yet confirmed where the track ends (turns_confirmed)."""
    out = []
    for si, side in enumerate(SIDES):
        for i0, i1, ch, settled in turns_confirmed(tr.closure[:, si], tr.continuous, CURL_MIN_DEG):
            out.append(_ev(tr, "curl", i0, i1, side=side, change_deg=ch,
                           direction="close" if ch > 0 else "open",
                           start_deg=float(tr.closure[i0, si]), end_deg=float(tr.closure[i1, si]),
                           open_end=bool(settled < 0)))
        for di, digit in enumerate(DIGITS):
            for i0, i1, ch, settled in turns_confirmed(tr.curl[:, si, di], tr.continuous, DIGIT_MIN_DEG):
                out.append(_ev(tr, "digit", i0, i1, side=side, digit=digit, change_deg=ch,
                               direction="bend" if ch > 0 else "straighten", open_end=bool(settled < 0)))
        for ki, kind in enumerate(("rotation", "flexion", "deviation")):
            for i0, i1, ch, settled in turns_confirmed(tr.wrist[:, si, ki], tr.continuous, WRIST_MIN_DEG):
                out.append(_ev(tr, "wrist", i0, i1, side=side, kind=kind, change_deg=ch,
                               open_end=bool(settled < 0)))
    return out


def hand_views(tr: Track) -> List[dict]:
    """Intervals with a hand's palm or any fingertip inside an eye's field of view."""
    out = []
    half_wide = 0.5 * float(tr.header.get("wide_fovy_deg", 60.0))
    half_fovea = 0.5 * float(tr.header.get("fovea_fovy_deg", 24.0))
    for si, side in enumerate(SIDES):
        pts = np.concatenate([tr.palm[:, si:si + 1], tr.tips[:, si]], axis=1)      # [N, 6, 3]
        best = np.full(tr.n, np.inf)
        for eye in (0, 1):
            for p in range(pts.shape[1]):
                a = tr.angle_from_gaze(pts[:, p], eye)
                best = np.fmin(best, a)
        # a ray actually landing on this hand is the direct proof it was in sight
        on_ray = np.zeros(tr.n, dtype=bool)
        hand = tr.ent.hand[side]
        ents = tr.ray_ent.reshape(tr.n, -1)
        for i in range(tr.n):
            e = ents[i][ents[i] >= 0]
            on_ray[i] = bool(hand[e].any()) if e.size else False
        for i0, i1 in _intervals(best <= half_wide, tr.continuous):
            seg = best[i0:i1 + 1]
            out.append(_ev(tr, "hand_view", i0, i1, side=side, min_angle_deg=float(seg.min()),
                           in_fovea=bool((seg <= half_fovea).any()),
                           ray_hit=bool(on_ray[i0:i1 + 1].any())))
    return out


def fixations(tr: Track) -> List[dict]:
    """An eye's fixation: its foveal bundle on the same target - the centre ray's thing (an object,
    a body part, the adult, one painted panel's motif cell), or, where the centre ray lands on a
    background surface or nothing, the thing a majority of the 9 rays land on (Track.foveal_target,
    detector v7) - for at least FIX_MIN_STEPS consecutive records. Each carries: the target;
    `ray_share`, the mean share of the 9 rays on it (1 of 9 for a toy at reaching distance: the
    ring sits 12 degrees out); `error_deg`, the median angle between the centre ray and the target's point
    (Track.target_point; None for a surface); `dwell_ms`, its records x 50 ms; `jitter_deg`, how
    far the eye's axis turned per step while it held; `distance_m`, the centre ray's median
    distance to it; and the TARGET CHANGE from this eye's previous fixation: `switch` (another
    target), `previous_target` and `since_previous_s` (both None after a break in the record).
    `inner_share` (observer_world_v2, 2026-09-25): the mean share of the 17 inner rays - the centre
    ray and the rings at 2 and 6 degrees - on the target, how much of the fovea's middle it filled
    (None on a track recorded without the inner rings). Reported beside the rule, not used by it: a
    majority of the inner rays as the rule made holds less still (54% of 2,306 production holds
    with a 5-degree jump, against 49% of 2,462 under the centre ray) and kept 33% of the centre ray's
    looks at objects 20-40 cm away (docs/gain_of_function_v4/agents/observer_recorder_2026-09-25.md)."""
    out = []
    for eye in (0, 1):
        tgt, cnt, R = tr.foveal_target(eye)
        centre = tr.bundle_targets(eye)[:, 0]
        inner_on: Dict[int, Optional[np.ndarray]] = {}
        errors: Dict[int, np.ndarray] = {}
        prev = None
        i = 0
        while i < tr.n:
            j = i
            while j + 1 < tr.n and tgt[j + 1] == tgt[i] and tr.continuous[j + 1]:
                j += 1
            if tgt[i] >= 0 and j - i + 1 >= FIX_MIN_STEPS:
                t = int(tgt[i])
                e = tr.target_entity(t)
                # jitter: how much the eye's axis wandered while it stayed on the target
                f = tr.eye_fwd[i:j + 1, eye]
                ang = np.degrees(np.arccos(np.clip(np.sum(f[1:] * f[:-1], axis=-1), -1, 1))) if j > i else np.zeros(1)
                if t not in errors:
                    errors[t] = tr.gaze_error_deg(t, eye)
                err = errors[t][i:j + 1]
                on_c = (centre[i:j + 1] == t) | (tr.ray_ent[i:j + 1, eye, 0].astype(np.int64) == t)
                dist = np.asarray(tr.ray_dist[i:j + 1, eye, 0], dtype=np.float64)[on_c]
                linked = prev is not None and bool(np.all(tr.continuous[prev["i1"] + 1:i + 1]))
                if t not in inner_on:
                    inner_on[t] = tr.inner_rays_on(t, eye) if hasattr(tr, "inner_rays_on") else None
                inner = inner_on[t]
                out.append(_ev(tr, "fixation", i, j, eye=eye, target=t, entity=e,
                               target_label=tr.target_label(t), target_class=entity_class(tr, e),
                               jitter_deg=float(np.nanmean(ang)) if np.isfinite(ang).any() else 0.0,
                               ray_share=float(np.mean(cnt[i:j + 1]) / R),
                               inner_share=(None if inner is None else float(np.mean(inner[i:j + 1]) / 17.0)),
                               error_deg=float(np.nanmedian(err)) if np.isfinite(err).any() else None,
                               dwell_ms=float(1000.0 * FIX_STEP_S * (j - i + 1)),
                               distance_m=float(np.median(dist)) if dist.size else None,
                               switch=bool(linked and prev["target"] != t),
                               previous_target=prev["target_label"] if linked else None,
                               since_previous_s=tr.seconds(prev["i1"], i) if linked else None))
                prev = out[-1]
            i = j + 1
    return out


def eye_moves(tr: Track) -> List[dict]:
    out = []
    for eye in (0, 1):
        a = tr.eye_angles[:, eye]
        for i in range(1, tr.n):
            if not tr.continuous[i]:
                continue
            ch = float(np.linalg.norm(a[i] - a[i - 1]))
            if np.isfinite(ch) and ch >= EYE_MIN_DEG:
                head = float(tr.head_ang_speed[i] * tr.dt[i]) if np.isfinite(tr.head_ang_speed[i]) else float("nan")
                out.append(_ev(tr, "eye_move", i - 1, i, eye=eye, change_deg=ch, head_change_deg=head,
                               world_saccade=bool(tr.gaze_saccade[i]), world_reflex=bool(tr.gaze_reflex[i])))
    return out


def head_turns(tr: Track) -> List[dict]:
    out = []
    ang = np.nan_to_num(tr.head_ang_speed * tr.dt, nan=0.0)
    w = max(1, int(round(HEAD_WINDOW_S / 0.05)))
    i = 1
    while i < tr.n:
        tot = 0.0
        j = i
        while j < tr.n and j < i + w and tr.continuous[j]:
            tot += ang[j]
            if tot >= HEAD_MIN_DEG:
                break
            j += 1
        if tot >= HEAD_MIN_DEG and j < tr.n:
            out.append(_ev(tr, "head_turn", i - 1, j, turn_deg=float(tot)))
            i = j + 1
        else:
            i += 1
    return out


def object_motions(tr: Track) -> List[dict]:
    out = []
    for k, o in enumerate(tr.ent.objects):
        if not (o["free"] or o["kind"] == "mobile"):
            continue
        if o["kind"] == "mobile":
            # the mobile's root is its ceiling mount (it never moves by itself; the adult raises
            # and lowers it); what swings hangs below on a ball joint, so its motion is the
            # fastest of its bodies, placed by forward kinematics
            speed = tr.object_body_speed(k)
            mv = np.nan_to_num(speed) >= OBJ_MOVE_MPS
        else:
            speed = tr.obj_speed[:, k]
            mv = (np.nan_to_num(speed) >= OBJ_MOVE_MPS) | (np.nan_to_num(tr.obj_ang_speed[:, k]) >= OBJ_SPIN_RADPS)
        for i0, i1 in _intervals(mv, tr.continuous):
            disp = tr.obj_pos[i1, k] - tr.obj_pos[max(0, i0 - 1), k]
            out.append(_ev(tr, "object_motion", i0, i1, object=o["name"], k=k,
                           displacement_m=float(np.linalg.norm(disp)),
                           direction=[float(v) for v in disp],
                           peak_speed_mps=float(np.nanmax(speed[i0:i1 + 1])),
                           peak_spin_radps=float(np.nanmax(tr.obj_ang_speed[i0:i1 + 1, k]))))
    return out


def rest_height(tr: Track, k: int) -> float:
    """A free object's height when the floor alone carries it (median over those steps)."""
    fl = tr.pair_series(tr.object_mask(k), tr.ent.floor)["on"]
    z = tr.obj_pos[:, k, 2]
    sel = fl & np.isfinite(z)
    if sel.any():
        return float(np.median(z[sel]))
    return float(np.nanmin(z)) if np.isfinite(z).any() else float("nan")


def lifts_drops(tr: Track) -> List[dict]:
    out = []
    for k, o in enumerate(tr.ent.objects):
        if not o["free"]:
            continue
        rest = rest_height(tr, k)
        z = tr.obj_pos[:, k, 2]
        om = tr.object_mask(k)
        floor = tr.pair_series(om, tr.ent.floor)["on"]
        handL = tr.pair_series(om, tr.ent.hand["L"])["on"]
        handR = tr.pair_series(om, tr.ent.hand["R"])["on"]
        hand = handL | handR
        others = np.zeros(tr.ent.n, dtype=bool)
        for kk, oo in enumerate(tr.ent.objects):
            if kk != k:
                others[int(oo["entity"])] = True
        on_obj = tr.pair_series(om, others)["on"]
        lifted = (z > rest + LIFT_MIN_M) & ~floor & ~on_obj & hand
        for i0, i1 in _intervals(lifted, tr.continuous):
            out.append(_ev(tr, "lift", i0, i1, object=o["name"], k=k,
                           height_m=float(np.nanmax(z[i0:i1 + 1]) - rest),
                           hands="+".join(s for s, h in (("L", handL), ("R", handR)) if h[i0:i1 + 1].any())))
        vz = tr.obj_vel[:, k, 2]
        free_fall = (np.nan_to_num(vz) <= -DROP_SPEED_MPS) & ~hand & ~floor & ~on_obj
        for i0, i1 in _intervals(free_fall, tr.continuous):
            recent = hand[max(0, i0 - 10):i0].any()
            out.append(_ev(tr, "drop", i0, i1, object=o["name"], k=k, from_hand=bool(recent),
                           hands="+".join(s for s, h in (("L", handL), ("R", handR)) if h[max(0, i0 - 10):i0].any()),
                           fall_m=float(z[max(0, i0 - 1)] - np.nanmin(z[i0:min(tr.n, i1 + 3)]))))
    return out


def support(tr: Track) -> Dict[str, np.ndarray]:
    """Per step: floor force carried by each part group, and the centre of that support."""
    groups = {"head": tr.ent.head, "trunk": tr.ent.trunk,
              "hand_L": tr.ent.hand["L"], "hand_R": tr.ent.hand["R"],
              "arm_L": tr.ent.arm["L"], "arm_R": tr.ent.arm["R"],
              "leg_L": tr.ent.leg["L"], "leg_R": tr.ent.leg["R"]}
    out = {}
    tot = np.zeros(tr.n)
    cop_w = np.zeros(tr.n)
    cop = np.zeros((tr.n, 3))
    for name, mask in groups.items():
        s = tr.pair_series(mask, tr.ent.floor)
        out[name] = s["force"]
        tot += s["force"]
        ok = np.isfinite(s["pos"]).all(-1) & (s["force"] > 0)
        cop[ok] += s["force"][ok, None] * s["pos"][ok]
        cop_w[ok] += s["force"][ok]
    out["total"] = tot
    c = np.full((tr.n, 3), np.nan)
    ok = cop_w > 0
    c[ok] = cop[ok] / cop_w[ok, None]
    out["cop"] = c
    return out


def support_shifts(tr: Track, sup: Dict[str, np.ndarray]) -> List[dict]:
    """The centre of mass moving COM_SHIFT_M over the floor within COM_SHIFT_MAX_S, from where it
    had stayed (within half that distance) for COM_HOLD_S, and keeping at least half of the
    shift for COM_HOLD_S: a weight shift from one settled place to another, not a wobble or the
    return from one. `lateral_m` is the shift along the trunk's left axis at the start
    (positive toward the body's left)."""
    out = []
    com = tr.com[:, :2]
    wmax = int(round(COM_SHIFT_MAX_S / 0.05))
    hold = int(round(COM_HOLD_S / 0.05))
    i = 0
    while i < tr.n:
        if not np.isfinite(com[i]).all():
            i += 1
            continue
        k0 = max(0, i - hold)
        before = com[k0:i + 1]
        if not (bool(np.all(tr.continuous[k0 + 1:i + 1])) and np.isfinite(before).all()
                and bool(np.all(np.linalg.norm(before - com[i], axis=1) < 0.5 * COM_SHIFT_M))):
            i += 1
            continue
        found = None
        j = i + 1
        while j < tr.n and j - i <= wmax and tr.continuous[j] and np.isfinite(com[j]).all():
            if float(np.linalg.norm(com[j] - com[i])) >= COM_SHIFT_M:
                found = j
                break
            j += 1
        if found is None:
            i += 1
            continue
        k1 = min(tr.n - 1, found + hold)
        held = (k1 - found == hold and bool(np.all(tr.continuous[found + 1:k1 + 1]))
                and bool(np.all(np.linalg.norm(com[found:k1 + 1] - com[i], axis=1) >= 0.5 * COM_SHIFT_M)))
        if not held:
            i += 1
            continue
        d = com[found] - com[i]
        lat = tr.trunk_lat[i]
        lateral = float(np.dot(d, lat[:2])) if np.isfinite(lat).all() else float("nan")
        out.append(_ev(tr, "support_shift", i, found, shift_m=float(np.linalg.norm(d)),
                       direction=[float(v) for v in d], lateral_m=lateral))
        i = found
    return out


def posture_state(tr: Track) -> np.ndarray:
    """[N] 'back', 'side', 'belly', 'between' or 'upright'."""
    roll = np.abs(tr.roll_deg)
    out = np.full(tr.n, "between", dtype=object)
    out[roll <= BACK_MAX_DEG] = "back"
    out[(roll >= SIDE_MIN_DEG) & (roll <= SIDE_MAX_DEG)] = "side"
    out[roll >= BELLY_MIN_DEG] = "belly"
    out[np.nan_to_num(tr.sit_deg, nan=90.0) <= UPRIGHT_MAX_DEG] = "upright"
    return out


def postures(tr: Track) -> List[dict]:
    out = []
    st = posture_state(tr)
    last, last_i = None, None
    for i in range(tr.n):
        if st[i] == "between":
            continue
        if last is not None and st[i] != last and tr.continuous[i]:
            down = "L" if tr.shoulder_z[i, 0] < tr.shoulder_z[i, 1] else "R"
            out.append(_ev(tr, "posture", last_i, i, frm=last, to=str(st[i]), side_down=down,
                           roll_change_deg=float(tr.roll_deg[i] - tr.roll_deg[last_i])))
        if last != st[i]:
            last, last_i = str(st[i]), i
        else:
            last_i = i
    return out


def _frames(tr: Track, channel: int):
    """The fast lane's valid frames of one channel as one series: (values, step index of each
    frame, frame number inside its step, whether the frame follows the previous one)."""
    vals, rows, nums, cont = [], [], [], []
    for i in range(tr.n):
        f = np.asarray(tr.fast[i][channel], dtype=np.float64)
        ok = np.nonzero(np.isfinite(f))[0]
        for k, fi in enumerate(ok):
            vals.append(f[fi])
            rows.append(i)
            nums.append(int(fi))
            cont.append(bool(k > 0 or (i > 0 and tr.continuous[i])))
    return (np.asarray(vals, dtype=np.float64), np.asarray(rows, dtype=np.int64),
            np.asarray(nums, dtype=np.int64), np.asarray(cont, dtype=bool))


def transients(tr: Track) -> List[dict]:
    out = []
    w = float(tr.header.get("weight_n", 30.0))
    for i in range(tr.n):
        f = tr.fast[i]
        acc, gyro = f[0], f[1]
        fin = np.isfinite(acc)
        if fin.any():
            dev = np.abs(acc[fin] - G)
            if dev.max() >= JOLT_MS2:
                out.append(_ev(tr, "transient", i, i, kind="head_jolt", peak=float(dev.max()),
                               frame=int(np.argmax(dev))))
        if np.isfinite(gyro).any() and np.nanmax(gyro) >= SPIN_RADPS:
            out.append(_ev(tr, "transient", i, i, kind="head_spin", peak=float(np.nanmax(gyro)),
                           frame=int(np.nanargmax(gyro))))
    # impact: the largest single contact force rising by a body weight within one frame
    v, rows, nums, cont = _frames(tr, 3)
    if v.size > 1:
        rise = np.full(v.size, -np.inf)
        rise[1:] = np.where(cont[1:], v[1:] - v[:-1], -np.inf)
        last = -1
        for k in np.nonzero(rise >= IMPACT_RISE_FRACTION * w)[0]:
            if rows[k] == last:
                continue                     # one impact per step
            last = rows[k]
            out.append(_ev(tr, "transient", int(rows[k]), int(rows[k]), kind="impact",
                           peak=float(rise[k] / w), frame=int(nums[k]), force_n=float(v[k])))
    # loss of support: outside force held below a quarter body weight for 30 ms right after
    # carrying at least three quarters of it
    v, rows, nums, cont = _frames(tr, 2)
    k = 1
    while k < v.size:
        if cont[k] and v[k - 1] >= SUPPORT_HELD_FRACTION * w and v[k] < SUPPORT_LOSS_FRACTION * w:
            m = k
            while m + 1 < v.size and cont[m + 1] and v[m + 1] < SUPPORT_LOSS_FRACTION * w:
                m += 1
            if m - k + 1 >= SUPPORT_LOSS_FRAMES:
                out.append(_ev(tr, "transient", int(rows[k]), int(rows[m]), kind="support_loss",
                               peak=float((v[k - 1] - v[k:m + 1].min()) / w), frame=int(nums[k]),
                               low_ms=10.0 * (m - k + 1)))
            k = m + 1
        else:
            k += 1
    return out


def approaches(tr: Track) -> List[dict]:
    out = []
    wmax = int(round(APPROACH_MAX_S / 0.05))
    for k, o in enumerate(tr.ent.objects):
        if o["kind"] in ("caregiver", "decoration") or not (tr.object_flags[:, k] & 1).any():
            continue
        for si, side in enumerate(SIDES):
            d = np.linalg.norm(tr.palm[:, si] - tr.obj_pos[:, k], axis=-1)
            i = 0
            while i < tr.n - 1:
                if not np.isfinite(d[i]):
                    i += 1
                    continue
                j = i + 1
                best = i
                while j < tr.n and j - i <= wmax and tr.continuous[j] and np.isfinite(d[j]):
                    if d[j] < d[best]:
                        best = j
                    if d[j] > d[best] + 0.01:        # moving away again: the approach ended
                        break
                    j += 1
                closed = float(d[i] - d[best])
                if closed >= APPROACH_MIN_M and best > i:
                    path = float(np.sum(np.linalg.norm(np.diff(tr.palm[i:best + 1, si], axis=0), axis=-1)))
                    out.append(_ev(tr, "approach", i, best, side=side, object=o["name"], k=k,
                                   start_m=float(d[i]), end_m=float(d[best]), closed_m=closed,
                                   path_efficiency=closed / path if path > 1e-9 else 0.0))
                    i = best + 1
                else:
                    i += 1
    return out


def caregiver_events(tr: Track) -> List[dict]:
    out = []
    for i0, i1 in _intervals(tr.caregiver_contact.astype(bool), tr.continuous):
        out.append(_ev(tr, "caregiver", i0, i1, kind="touch"))
    pres = tr.caregiver_present.astype(bool)
    for i in range(1, tr.n):
        if pres[i] != pres[i - 1] and tr.caregiver[i]:
            out.append(_ev(tr, "caregiver", i, i, kind="arrive" if pres[i] else "leave"))
    return out


def all_primitives(tr: Track) -> List[dict]:
    """Every primitive of a track, ordered by start."""
    sup = support(tr)
    ev = (contacts(tr) + curls(tr) + hand_views(tr) + fixations(tr) + eye_moves(tr) + head_turns(tr)
          + object_motions(tr) + lifts_drops(tr) + support_shifts(tr, sup) + postures(tr)
          + transients(tr) + approaches(tr) + caregiver_events(tr))
    ev.sort(key=lambda e: (e["i0"], e["type"]))
    return ev
