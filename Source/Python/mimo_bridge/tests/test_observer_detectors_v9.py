# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Detector v9 (2026-09-25): the observer's detector and validation-case defects of OBSERVER_DESIGN.md
section 9 (docs/gain_of_function_v4/agents/observer_detectors_2026-09-25.md).

Each test fails on the v8 detectors and passes on v9, and keeps a twin the change must not alter.
The tracks are made up: the real Entities and Track's own contact-pair and time methods, with only
the per-record fields these detectors and primitives read. No world is built.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from mimo_bridge.tools.observer import detectors as D
from mimo_bridge.tools.observer import primitives as P
from mimo_bridge.tools.observer.features import Entities, Track

_KINDS = ["self", "object", "caregiver", "floor", "wall", "art", "ceiling", "static"]
_ENTS = [dict(name="right_hand", kind="self", group="hand_R_palm"),
         dict(name="left_hand", kind="self", group="hand_L_palm"),
         dict(name="upper_body", kind="self", group="trunk"),
         dict(name="floor", kind="floor", group="floor")]
_JOINTS = [dict(name=n, qpos=i, range=[-0.436, 0.436]) for i, n in enumerate(
    ["right_ff_side", "right_mf_side", "right_rf_side", "right_lf_side", "right_lf_meta",
     "left_ff_side", "left_mf_side", "left_rf_side", "left_lf_side", "left_lf_meta"])]


class _Hands:
    """A made-up track of two hands: digit curls, wrist angles, the side and cupping hinges."""
    pair_series = Track.pair_series
    entity_mask = Track.entity_mask
    object_mask = Track.object_mask
    seconds = Track.seconds

    def __init__(self, n):
        self.header = dict(entities=_ENTS, entity_kind=[_KINDS.index(e["kind"]) for e in _ENTS], objects=[],
                           joints=_JOINTS, weight_n=100.0)
        self.ent = Entities(self.header)
        self.n = n
        self.step = np.arange(n)
        self.sim_time = 0.05 * np.arange(n)
        self.dt = np.full(n, 0.05)
        self.dt[0] = np.nan
        self.continuous = np.isfinite(self.dt)
        z = np.zeros(n, dtype=bool)
        self.author = np.full(n, 8, dtype=np.int8)
        self.latched, self.rest, self.caregiver_contact, self.caregiver_present = z.copy(), z.copy(), z.copy(), z.copy()
        self.live = ~z
        self.order_seq = np.arange(n)
        self.toy_held = np.zeros(n, dtype=np.int8)
        self.roll_deg = np.zeros(n)
        self.sit_deg = np.full(n, 90.0)
        self.curl = np.full((n, 2, 5), 20.0)
        self.wrist = np.zeros((n, 2, 3))
        self.forearm = np.tile([1.0, 0.0, 0.0], (n, 2, 1))
        self.palm = np.zeros((n, 2, 3))
        self.qpos = np.zeros((n, len(_JOINTS)))
        self.c_row = np.zeros(0, dtype=np.int64)
        self.c_e = np.zeros((0, 2), dtype=np.int64)
        self.c_fmax = np.zeros(0)
        self.c_fmean = np.zeros(0)
        self.c_pos = np.zeros((0, 3))
        self._c_sorted = True

    def done(self):
        self.closure = np.nanmean(self.curl[:, :, :4], axis=-1)
        return self


def _run(tr, *fns):
    ctx = D.Context(tr, P.curls(tr))
    return [d for fn in fns for d in fn(ctx)]


def _ramp(n, i0, i1, a, b):
    """n records: a until record i0, a linear move to b at record i1, then b."""
    x = np.full(n, float(a))
    x[i0:i1 + 1] = np.linspace(a, b, i1 - i0 + 1)
    x[i1:] = b
    return x


# ---------------------------------------------------------------------- defect 3: where a turn starts
def test_v9_a_turn_starts_where_the_movement_to_its_peak_starts():
    """Production, segment 39, body 0, left index from step 391,393 (rounded): the finger bent 11
    degrees and came back to its low twice before the 17-degree bend to its peak. v8 started the turn
    at the stretch's lowest record (index 0), 2.85 s of real time before the bend began."""
    x = np.array([27.2, 29.0, 33.0, 38.3, 36.0, 33.1, 36.0, 41.2, 38.0, 33.0, 29.0, 27.6, 27.8, 28.3,
                  31.0, 35.0, 39.0, 42.0, 44.6, 44.4, 30.0])
    turns = P._turns(x, np.ones(len(x), dtype=bool), 15.0)
    i0, i1, ch = turns[0]
    # the low is 27.6 at record 11; the finger stays within 0.75 degrees of it (5% of the line)
    # through record 13 and leaves it for good after: the bend starts there (v8: record 0)
    assert (i0, i1) == (13, 18)
    assert ch == pytest.approx(44.6 - 28.3)
    # a flat low: the turn starts at its LAST record (v8: its first)
    x = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0])
    assert P._turns(x, np.ones(len(x), dtype=bool), 15.0)[0][:2] == (4, 6)
    # the twin: a plain rise from its first record keeps that start
    x = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    assert P._turns(x, np.ones(len(x), dtype=bool), 15.0) == [(0, 4, 20.0)]
    # every run still rises or falls by at least its line from its start
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0.0, 3.0, 2000))
    for a, b, c in P._turns(x, np.ones(len(x), dtype=bool), 15.0):
        assert abs(c) >= 15.0 and c == pytest.approx(x[b] - x[a])


def test_v9_an_open_close_cycle_may_pause_at_its_extremes_but_not_hold_a_smaller_turn():
    """H04 read 'the next turn starts within 2 records of the previous turn's end', true for any
    two consecutive turns while turns tiled the record. With v9's starts a pause at the top lies
    between them: the cycle is kept when the turns follow one another."""
    def cycle(pause, dip=0.0):
        n = 60
        tr = _Hands(n)
        c = np.full(n, 10.0)
        c[5:11] = np.linspace(10.0, 40.0, 6)             # close 30 degrees
        c[11:11 + pause] = 40.0                            # hold closed
        k = 11 + pause
        if dip:
            c[k:k + 3] = 40.0 - np.array([dip / 2, dip, dip / 2])   # a smaller opening and back
            k += 3
        c[k:k + 6] = np.linspace(40.0, 10.0, 6)            # open
        c[k + 6:k + 6 + pause] = 10.0
        k2 = k + 6 + pause
        c[k2:k2 + 6] = np.linspace(10.0, 40.0, 6)          # close again
        c[k2 + 6:] = 40.0
        for f in range(4):
            tr.curl[:, 1, f] = c
        return _run(tr.done(), D.H04)
    assert [d["cap"] for d in cycle(pause=6)] == ["H04"]   # 0.3 s held at each end: one cycle
    assert [d["cap"] for d in cycle(pause=0)] == ["H04"]   # the twin: no pause
    # a 12-degree opening and closing between (a curl turn under H04's 15 degrees): not one cycle
    assert cycle(pause=2, dip=12.0) == []


def test_v9_finger_rows_keep_the_catalog_timescale():
    """The catalog's timescale for H05, H07, H08 and HX2 is 250 ms - 2 s, applied to HX1 in v8.
    v8 counted a limp finger's 26 s drift as one isolated finger movement once its turn started
    where the drift did (the limp reset world: 8 bodies, 28 degrees each)."""
    def h05(rise_records):
        n = 30 + rise_records
        tr = _Hands(n)
        tr.curl[:, 1, 0] = _ramp(n, 5, 5 + rise_records, 20.0, 50.0)
        return [d for d in _run(tr.done(), D.H05)]
    fast = h05(8)                                          # 30 degrees in 0.4 s
    assert [d["sub"] for d in fast] == ["R:index"] and fast[0]["measures"]["rise_s"] == pytest.approx(0.4)
    assert h05(520) == []                                  # the same bend over 26 s: a drift
    assert len(h05(38)) == 1                               # over 1.9 s it is one act

    def h07(rise_records):
        n = 30 + rise_records
        tr = _Hands(n)
        tr.wrist[:, 1, 1] = _ramp(n, 5, 5 + rise_records, 0.0, 35.0)
        return _run(tr.done(), D.H07)
    assert [d["cap"] for d in h07(6)] == ["H07"]
    assert h07(80) == []

    def hx2(rise_records):
        n = 30 + rise_records
        tr = _Hands(n)
        tr.qpos[:, 4] = np.radians(_ramp(n, 5, 5 + rise_records, 0.0, 30.0))
        return _run(tr.done(), D.HX2)
    assert [d["cap"] for d in hx2(4)] == ["HX2"]
    assert hx2(2000) == []                                 # the limp cupping hinge's 102.8 s change


def _creep_and_flick(line, lead=0.0, flick=None, tail=0.0, flick_records=6, creep_records=60, base=0.0):
    """A turn of one joint series (degrees): still, a slow creep of `lead` over creep_records (3 s), a
    fast move of `flick` (default the full line) over flick_records, a slow creep of `tail` over
    creep_records, still again."""
    flick = float(line if flick is None else flick)
    x = [base] * 10
    x += list(base + np.linspace(0.0, lead, creep_records + 1)[1:])
    top = x[-1]
    x += list(top + np.linspace(0.0, flick, flick_records + 1)[1:])
    top = x[-1]
    x += list(top + np.linspace(0.0, tail, creep_records + 1)[1:]) if tail else []
    x += [x[-1]] * 30
    return np.array(x)


def test_v9_the_act_timescale_is_read_on_the_turns_fastest_stretch_of_its_full_line():
    """The independent review (2026-09-25, "the act timescale is measured over the whole run"): v9 read the
    catalog's 2 s as the whole turn's 10-90% rise, and a turn starts where its monotone approach starts, so
    a slow lead-in dropped a fast act (segment 39, body 0: a 23-degree finger straightening, 2.55 s rise,
    inside it the full 15 degrees in 0.60 s), and a slow creep after a flick UNDER the line carried the
    turn past the line inside the rise. The act is a movement of the row's full line within 2 s."""
    def acts(row, x):
        n = len(x)
        tr = _Hands(n)
        if row == "H05":
            tr.curl[:, 1, 0] = 20.0 + x
        elif row == "H07":
            tr.wrist[:, 1, 1] = x
        elif row == "HX1":
            tr.qpos[:, 0] = np.radians(x)                    # the right index's side angle: the fan
        else:
            tr.qpos[:, 4] = np.radians(x)                    # the right cupping hinge
        fn = dict(H05=D.H05, H07=D.H07, HX1=D.HX1, HX2=D.HX2)[row]
        return _run(tr.done(), fn)
    for row, line in (("H05", P.DIGIT_MIN_DEG), ("H07", P.WRIST_MIN_DEG), ("HX1", P.DIGIT_MIN_DEG), ("HX2", 15.0)):
        # a slow lead-in of half the line over 3 s, then the full line in 0.3 s: one act (v9 before: none)
        found = acts(row, _creep_and_flick(line, lead=0.5 * line))
        assert len(found) == 1, (row, found)
        assert found[0]["measures"]["line_s"] == pytest.approx(0.3) and found[0]["measures"]["rise_s"] > 2.0
        # a flick of 92.5% of the line in 0.3 s, then a creep of 10% of it over 3 s: the turn passes the line
        # only by creeping, 2.55 s after the flick began - no act (v9 before: one, its 10-90% rise 0.3 s)
        assert acts(row, _creep_and_flick(line, flick=0.925 * line, tail=0.1 * line)) == [], row
        # the twins: the line (and 2% more) in 0.3 s with no creep is an act; a creep alone over 26 s is none
        assert len(acts(row, _creep_and_flick(line, flick=1.02 * line))) == 1, row
        assert acts(row, _creep_and_flick(line, lead=2.0 * line, flick=0.0, creep_records=520)) == [], row
    # the measure: the shortest stretch that moves the line, wherever it lies in the turn
    tr = _Hands(40)
    x = np.r_[np.zeros(5), np.linspace(0.0, 30.0, 9), np.full(26, 30.0)]    # 30 degrees in 8 records
    assert P.line_s(tr, x, 5, 13, 30.0, 15.0) == pytest.approx(0.2)
    assert P.line_s(tr, x, 5, 13, 30.0, 31.0) == float("inf")
    assert P.line_s(tr, -x, 5, 13, -30.0, 15.0) == pytest.approx(0.2)          # a fall reads its own way


# ---------------------------------------------------------------------- defect 4: fingers alone open an episode
def test_v9_a_finger_moving_from_rest_opens_its_hands_episode():
    """v8: 'a hand moving only its fingers opens no episode' - each turn began where the previous
    one ended, so none could mark an act's start. v9: a hand turn from its own joint's 0.5 s of rest
    that rises within 2 s opens a hand episode held only to its start; turns in progress ride, and a
    limp drift from rest (rising over more than 2 s) opens nothing."""
    from mimo_bridge.tools.observer import segmenter as S

    def hand_eps(curl_index, curl_middle=None):
        n = len(curl_index)
        tr = _Hands(n)
        tr.curl[:, 1, 0] = curl_index
        if curl_middle is not None:
            tr.curl[:, 1, 1] = curl_middle
        tr.done()
        tr.ent.kind = np.asarray(tr.header["entity_kind"])
        tr.pid, tr.episode, tr.caregiver = 0, np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
        tr.c_row = np.zeros(0, dtype=np.int64)
        eps = S.segment(tr, P.curls(tr), "run", 0)            # returned latest start first
        return sorted((e["i0"], e["i1"], e["opening"]) for e in eps if e["effector"] == "hand_R")
    # the index still for 1 s, bends 30 degrees in 0.3 s, still again, then bends back and forth
    x = np.r_[np.full(20, 20.0), np.linspace(20.0, 50.0, 7), np.full(20, 50.0),
              np.tile(np.r_[np.linspace(50.0, 30.0, 4)[:-1], np.linspace(30.0, 50.0, 4)[:-1]], 6), np.full(10, 50.0)]
    eps = hand_eps(x)
    assert eps[0] == (20, 20, "digit:R:index:bend")         # the bend from rest
    assert eps[1][2] == "digit:R:index:straighten"         # after 1 s still: another onset
    assert len(eps) == 2, "the back-and-forth that follows at once continues it: no new episode"
    # the same bend drifting over 8 s opens nothing (the limp body's drifts)
    assert hand_eps(np.r_[np.full(20, 20.0), np.linspace(20.0, 50.0, 161), np.full(20, 50.0)]) == []
    # v8's tiling turns never opened one: the rule is new, keep the key stable
    assert S.hand_turn_key(dict(type="digit", side="R", digit="index", i0=20)) == ("digit", "R", "index", 20)


# ---------------------------------------------------------------------- defect 1: a release into the air
_GENTS = [dict(name="right_hand", kind="self", group="hand_R_palm"),
          dict(name="right_thdistal", kind="self", group="finger_R_thumb"),
          dict(name="right_ffdistal", kind="self", group="finger_R_index"),
          dict(name="left_hand", kind="self", group="hand_L_palm"),
          dict(name="upper_body", kind="self", group="trunk"),
          dict(name="box_light", kind="object", group="box"),
          dict(name="floor", kind="floor", group="floor")]
PALM, THUMB, INDEX, BOX, FLOOR = 0, 1, 2, 5, 6


class _Grip:
    """A made-up track of the right hand and one free 6 cm box: palm, closure, box pose, contacts."""
    pair_series = Track.pair_series
    entity_mask = Track.entity_mask
    object_mask = Track.object_mask
    contacts_at = Track.contacts_at
    seconds = Track.seconds

    def __init__(self, n):
        self.header = dict(entities=_GENTS, entity_kind=[_KINDS.index(e["kind"]) for e in _GENTS],
                           objects=[dict(entity=BOX, name="box_light", kind="box", root=1, mocap=-1, free=True)],
                           joints=[], weight_n=100.0)
        self.ent = Entities(self.header)
        self.n = n
        self.step = np.arange(n)
        self.sim_time = 0.05 * np.arange(n)
        self.dt = np.full(n, 0.05)
        self.dt[0] = np.nan
        self.continuous = np.isfinite(self.dt)
        z = np.zeros(n, dtype=bool)
        self.author = np.full(n, 8, dtype=np.int8)
        self.latched, self.rest, self.caregiver_contact, self.caregiver_present = z.copy(), z.copy(), z.copy(), z.copy()
        self.live = ~z
        self.order_seq = np.arange(n)
        self.toy_held = np.zeros(n, dtype=np.int8)
        self.roll_deg = np.zeros(n)
        self.sit_deg = np.full(n, 90.0)
        self.closure = np.full((n, 2), 30.0)
        self.palm = np.zeros((n, 2, 3))
        self.palm[:, 0] = (0.0, -0.3, 0.1)
        self.palm[:, 1] = (0.17, 0.26, 0.065)
        self.obj_pos = np.zeros((n, 1, 3))
        self.touch = []

    def contact(self, a, b, records, frames=(0, 4)):
        """Touching in these records, from the burst's frame frames[0] to frames[1] (the recorder's c_frame)."""
        self.touch += [(int(i), a, b, int(frames[0]), int(frames[1])) for i in records]

    def done(self):
        vel = lambda x: np.concatenate([np.full((1,) + x.shape[1:], np.nan),
                                        np.diff(x, axis=0) / self.dt[1:].reshape((-1,) + (1,) * (x.ndim - 1))])
        self.palm_vel = vel(self.palm)
        self.palm_speed = np.linalg.norm(self.palm_vel, axis=-1)
        self.obj_vel = vel(self.obj_pos)
        self.obj_speed = np.linalg.norm(self.obj_vel, axis=-1)
        self.obj_ang_speed = np.zeros((self.n, 1))
        self.obj_quat = np.tile([1.0, 0.0, 0.0, 0.0], (self.n, 1, 1))
        t = sorted(self.touch)
        self.frames = np.full(self.n, 5)
        self.c_row = np.asarray([r for r, _, _, _, _ in t], dtype=np.int64)
        self.c_e = np.asarray([(a, b) for _, a, b, _, _ in t], dtype=np.int64).reshape(-1, 2)
        self.c_frame = np.asarray([(f0, f1) for _, _, _, f0, f1 in t], dtype=np.int64).reshape(-1, 2)
        self.c_fmax = np.full(len(t), 2.0)
        self.c_fmean = np.full(len(t), 2.0)
        self.c_pos = np.asarray([self.obj_pos[r, 0] for r, _, _, _, _ in t]).reshape(-1, 3)
        self._c_sorted = True
        return self


def _let_go(after, last_frame=4):
    """A box lifted 3 cm off the floor by a thumb-and-finger grip (records 10-12), held, then the
    hand opens (records 26-30) and lets go at record 30, whose burst it touches up to its frame
    last_frame; after(tr, box) moves the box from record 31 (and may move it within record 30)."""
    tr = _Grip(50)
    tr.palm[10:13, 1, 2] += [0.01, 0.02, 0.03]
    tr.palm[13:, 1, 2] += 0.03
    tr.obj_pos[:, 0] = tr.palm[:, 1] + np.array([0.0, 0.0, -0.036])
    tr.closure[26:31, 1] = np.linspace(30.0, 5.0, 5)
    tr.closure[31:, 1] = 5.0
    for part in (PALM, THUMB, INDEX):
        tr.contact(part, BOX, range(5, 30))
        tr.contact(part, BOX, [30], frames=(0, last_frame))
    floor_from = after(tr, tr.obj_pos[30, 0].copy())
    tr.contact(BOX, FLOOR, list(range(0, 11)) + list(range(floor_from, 50)))
    return tr.done()


def _free_fall(tr, p):
    """The box falls from where the hand let it go (record 30) to the floor 3 cm below."""
    g = 9.81
    floor_z = p[2] - 0.03
    for i in range(31, tr.n):
        t = 0.05 * (i - 30)
        tr.obj_pos[i, 0] = (p[0], p[1], max(floor_z, p[2] - 0.5 * g * t * t))
    return 32                                               # on the floor from the second record


def test_v9_a_box_let_go_in_the_air_is_a_controlled_release():
    """v8 read the falling box's own speed (0.25 then 0.35 m/s) as an escape."""
    tr = _let_go(_free_fall)
    assert tr.obj_speed[31, 0] == pytest.approx(0.245, abs=0.002) and tr.obj_speed[32, 0] > 0.10
    dets = _run_grip(tr, D.H03)
    assert [d["cap"] for d in dets] == ["H03"]
    assert dets[0]["measures"]["escape_mps"] == pytest.approx(0.0, abs=1e-9)
    assert dets[0]["measures"]["object_speed_mps"] > 0.3

    # the twins: pulled out sideways at 1.2 m/s, or upward at 0.3 m/s, as the hand opens - escapes
    def sideways(tr, p):
        for i in range(31, tr.n):
            tr.obj_pos[i, 0] = p + np.array([1.2 * 0.05 * (i - 30), 0.0, 0.0])
        return tr.n

    def upward(tr, p):
        for i in range(31, tr.n):
            tr.obj_pos[i, 0] = p + np.array([0.0, 0.0, 0.3 * 0.05 * (i - 30)])
        return tr.n

    def flung_down(tr, p):                                  # faster than a free fall: 3 m/s down
        for i in range(31, tr.n):
            tr.obj_pos[i, 0] = p + np.array([0.0, 0.0, -3.0 * 0.05 * (i - 30)])
        return tr.n
    assert _run_grip(_let_go(sideways), D.H03) == []
    assert _run_grip(_let_go(upward), D.H03) == []
    assert _run_grip(_let_go(flung_down), D.H03) == []


def _thrown(extra_down, extra_side=0.0, t0=0.0):
    """after() for _let_go: the box leaves the hand at t0 s before record 30's time (0: the record's
    last frame) with extra_down m/s downward and extra_side m/s sideways, and falls freely from there
    (a 30 cm drop: still falling over both records after)."""
    g = 9.81

    def after(tr, p):
        start = p.copy()                                      # where it was at the moment it left
        for i in range(30, tr.n):
            t = 0.05 * (i - 30) + t0                          # seconds since it left
            tr.obj_pos[i, 0] = start + np.array([extra_side * t, 0.0, -extra_down * t - 0.5 * g * t * t])
        return tr.n
    return after


def test_v9_a_box_thrown_down_escapes_as_a_box_pulled_sideways_does():
    """The independent review (2026-09-25, "H03's escape tolerates a downward throw 8 times faster than a
    sideways one"): the allowance for gravity was the speed a free fall reaches at the END of the record,
    started a whole record before the last contact record ended (0.98 m/s over the first record after),
    compared with the record's AVERAGE velocity: a box thrown down up to 0.8 m/s faster than a free fall
    read as let go, while sideways 0.10 m/s escapes. Now the average of a fall from rest that starts at the
    last frame a hand part touched the box: the same line both ways."""
    for u, released in ((0.05, True), (0.15, False)):
        down = _run_grip(_let_go(_thrown(u)), D.H03)
        side = _run_grip(_let_go(_thrown(0.0, u)), D.H03)
        assert bool(down) is released and bool(side) is released, (u, down, side)
    # the review's twin: thrown down at 0.3 m/s on top of free fall - an escape (v9 before this: a release)
    assert _run_grip(_let_go(_thrown(0.3)), D.H03) == []
    # a free fall that began at the record's FIRST frame (the hand's last touch there) averages 0.64 m/s
    # over the next record: gravity explains it, from the frame the recorder says the touch ended
    tr = _let_go(_thrown(0.0, t0=0.04), last_frame=0)
    assert tr.obj_speed[31, 0] == pytest.approx(9.81 * 0.065, abs=0.002)
    dets = _run_grip(tr, D.H03)
    assert [d["cap"] for d in dets] == ["H03"] and dets[0]["measures"]["escape_mps"] < 1e-9
    # the same fall read as touching up to the record's last frame is 0.39 m/s faster than a fall from there
    assert _run_grip(_let_go(_thrown(0.0, t0=0.04), last_frame=4), D.H03) == []


def test_v9_the_rebuilt_hold_and_release_cases_state_their_links():
    """Defects 1 and 5: the palm-box cases rebuilt on the settled grasp, and the escape twin."""
    from mimo_bridge.tools.observer import validate as V
    lines = {c[0]: (c[2], c[3], c[4]) for c in V.CASES}
    # held, then let go in the air: a release (v8 missed it) and a drop, not a placement
    assert lines["h02_h03_hold_then_release"] == (["H02:R", "H03:R"], ["R08"], {})
    # pulled out of the closed grip: no release (and no hold: the box slides past the pads)
    assert lines["h03_object_escapes"] == ([], ["H03"], {})
    # the same hold, the hand opens, the box is pulled away as it lets go: an escape
    assert lines["h03_hand_opens_as_box_is_pulled_out"] == (["H02:R"], ["H03"], {})
    # the grasp's closing is set by physics, not a scripted 30 degrees: no truth value
    assert lines["h01_touch_then_close"] == (["H01:R", "R02:box"], [], {})
    assert lines["h01_close_400ms_late"] == ([], ["H01"], {})
    assert lines["hx1_spread_and_gather"][2] == {"HX1.fan_deg": 20.0}      # v8 renamed amount_deg
    assert "h01_close_400ms_late" in V.BOUNDARY


def _run_grip(tr, *fns):
    ctx = D.Context(tr, [])
    return [d for fn in fns for d in fn(ctx)]


# ---------------------------------------------------------------------- defect 2: a limp hand's passive fan
class _Model:
    """The model fields detectors._motor_of_qpos reads: one motor per joint of _JOINTS."""
    def __init__(self):
        self.nu = len(_JOINTS)
        self.actuator_trntype = np.zeros(self.nu, dtype=np.int64)          # every motor drives a joint
        self.actuator_trnid = np.array([[j, -1] for j in range(self.nu)])
        self.jnt_qposadr = np.array([j["qpos"] for j in _JOINTS])
        self.actuator_gear = np.ones((self.nu, 6))                           # torque per unit command: 1
        self.actuator_gainprm = np.ones((self.nu, 10))


class _Kin:
    def __init__(self):
        self.m = _Model()
        self.flex_q, self.wrist_q, self.wrist_kind = {}, {}, {}


def _settle_fan(command):
    """The reset-world limp hand's fan (body 0, steps 82-124), side joints only: the little finger's
    up 8 degrees and back, then the ring, middle and index fingers toward their limits."""
    n = 40
    tr = _Hands(n)
    deg = dict(ff=_ramp(n, 16, 24, 0.0, 22.0), mf=_ramp(n, 12, 20, 0.0, 25.0), rf=_ramp(n, 8, 14, 0.0, 25.0),
               lf=_ramp(n, 4, 9, 0.0, 8.0) + _ramp(n, 9, 24, 0.0, -10.0))
    for k, name in enumerate(("ff", "mf", "rf", "lf")):
        tr.qpos[:, k] = np.radians(deg[name])
    tr.kin = _Kin()
    tr.ctrl = np.zeros((n, len(_JOINTS)), dtype=np.float16)
    tr.ctrl[4:25, 0:4] = command                           # the right hand's four side motors
    return _run(tr.done(), D.HX1)


def test_v9_a_limp_hands_passive_fan_is_no_spread():
    """v8 counted it: each of the 8 limp bodies of the reset world made one fan of 30-32 degrees 4-6
    s after the reset, every motor command zero (and the saved worlds' limp bodies 2 more when their
    commands went to zero)."""
    assert _settle_fan(0.0) == []
    driven = _settle_fan(0.3)                               # the twin: the same fan, its motors driving it
    assert [d["key"] for d in driven] == ["R:spread"]
    assert driven[0]["measures"]["command_peak"] == pytest.approx(0.3, abs=1e-3)
    assert driven[0]["measures"]["fan_deg"] > 30.0
    # a command under MOTOR_ON is silence; a track without a model or commands is not judged (NaN)
    assert _settle_fan(0.0005) == []
    assert D._passive(float("nan")) is False and D._passive(0.0) is True and D._passive(D.MOTOR_ON) is False


def test_v9_the_scripted_world_states_its_motor_commands():
    """validate.ScriptWorld.command: toward each driven joint's movement, 0 elsewhere; all 0 'off'."""
    from types import SimpleNamespace
    from mimo_bridge.tools.observer import validate as V
    w = SimpleNamespace(d=SimpleNamespace(ctrl=np.full(3, 0.7)), motor_dofs=[(0, 1), (1, 2), (2, 0)])
    qv = np.array([-0.5, 0.2, 1e-5])                        # dof 2 moves slower than SCRIPT_MOVING_RADPS
    V.ScriptWorld.command(w, qv, "drive")
    assert w.d.ctrl.tolist() == [V.SCRIPT_COMMAND, 0.0, -V.SCRIPT_COMMAND]
    V.ScriptWorld.command(w, qv, "against")
    assert w.d.ctrl.tolist() == [-V.SCRIPT_COMMAND, 0.0, V.SCRIPT_COMMAND]
    V.ScriptWorld.command(w, qv, "off")
    assert w.d.ctrl.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        V.ScriptWorld.command(w, qv, "limp")
    names = {c[0]: (c[2], c[3]) for c in V.CASES}
    assert names["hx1_settle_fan_motors_off"] == ([], ["HX1", "H05"])
    assert names["hx1_settle_fan_driven"][0] == ["HX1:R"]
    assert names["h05_index_alone_motors_off"] == ([], ["H05"])


# ---------------------------------------------------------------------- defect 7: a support push the motors drive
class _Recorder:
    """What ScriptWorld.record_driven asks of observer_record.WorldRecorder; keeps what the world was
    at each recorded frame."""
    frames: list = []

    def __init__(self, path, **kw):
        self.error = None

    def mark(self, what):
        pass

    def on_frame(self, env, raw, info):
        _Recorder.frames.append(dict(movers=(env.saccades, env.reflex, env._cr), home=np.array(env._target_home),
                                     capacity=float(env.proprio.fatigue.cap[0]), step=info["step"], rep=info["rep"]))

    def close(self):
        pass


def test_v9_a_driven_case_keeps_the_worlds_movers_out_and_starts_at_full_strength(monkeypatch, tmp_path):
    """record_driven steps the world itself, so the world's movers would act in it as they never do
    in a posed case: the saccade generator and reflex, the critters (the squeaker is put back at its
    home by the feet every frame), and the toy's leash (BridgeEnv.step pulls the toy back within 12 cm
    of its spawn point beside the face, wherever the case's scene put it). And BridgeEnv scales every
    motor's gain by its body set's fatigue capacity, so a driven case's strength would depend on the
    driven case before it. All are held out of the case and restored after it."""
    from types import SimpleNamespace
    import mimo_bridge.observer_record as OR
    from mimo_bridge.tools.observer import validate as V
    monkeypatch.setattr(OR, "WorldRecorder", _Recorder)
    _Recorder.frames = []
    fatigue = SimpleNamespace(cap=np.full(3, 0.4))
    m = SimpleNamespace(actuator_gainprm=np.full((2, 10), 0.4), actuator_forcerange=np.full((2, 2), 0.4))
    d = SimpleNamespace(time=0.0, qpos=np.zeros(4), qvel=np.zeros(4), qacc_warmstart=np.zeros(4), ctrl=np.zeros(2),
                        mocap_pos=np.zeros((1, 3)), xfrc_applied=np.ones((3, 6)))
    home, vel = np.array([0.1, 0.2, 0.3]), np.ones(3)
    env = SimpleNamespace(saccades="saccades", reflex="reflex", _cr={"squeaker": 1}, _target_id=0, _target_home=home,
                          _target_vel=vel, proprio=SimpleNamespace(fatigue=fatigue), frame=0,
                          _native_actuator_gain=np.ones((2, 10)))

    def step(*a, ctrl=None, render=None):                  # BridgeEnv under body_function 'modulate'
        m.actuator_gainprm[:, 0] = env._native_actuator_gain[:, 0] * fatigue.cap[0]
        d.ctrl[:] = ctrl
    env.step = step
    env.restart_clock = lambda t: setattr(d, "time", t)     # BridgeEnv.restart_clock's physics side
    w = V.ScriptWorld.__new__(V.ScriptWorld)
    w.env, w.m, w.d, w.reps, w.scratch = env, m, d, 5, str(tmp_path)
    w.mj = SimpleNamespace(mj_forward=lambda m_, d_: None)
    w.place_far = lambda: None
    far = np.array([2.5, 3.0, 3.0])
    case = V.Driven(lambda world: np.ones(4), lambda t, world: np.full(2, 0.5), 2,
                    lambda t, world: world.d.mocap_pos.__setitem__(0, far))
    V.ScriptWorld.record_driven(w, "push", case)
    assert len(_Recorder.frames) == 10                      # 2 steps of 5 frames
    for f in _Recorder.frames:
        assert f["movers"] == (None, None, None)
        assert np.allclose(f["home"], far)                  # the leash re-anchored where the scene put the toy
    assert _Recorder.frames[0]["capacity"] == 1.0           # full strength, whatever ran before
    # restored: the movers, the leash, the capacity and the motors' gains; the commands back to zero
    assert (env.saccades, env.reflex, env._cr) == ("saccades", "reflex", {"squeaker": 1})
    assert env._target_home is home and env._target_vel is vel
    assert fatigue.cap.tolist() == [0.4, 0.4, 0.4]
    assert np.all(m.actuator_gainprm == 0.4) and np.all(m.actuator_forcerange == 0.4)
    assert d.ctrl.tolist() == [0.0, 0.0]
    assert np.all(d.xfrc_applied == 0.0)                    # forces a scene applied are cleared after the case


def test_v9_the_support_push_cases_drive_the_leg_and_state_their_links():
    """P05 had no scripted case (OBSERVER_DESIGN.md section 9): the push and its in-the-air twin, the left
    leg lifted in both (its weight on the pelvis)."""
    from types import SimpleNamespace
    from mimo_bridge.tools.observer import validate as V
    lines = {c[0]: (c[2], c[3]) for c in V.CASES}
    assert lines["p05_leg_pushes_the_floor"] == (["P05:R:leg"], [])
    assert lines["p05_leg_extends_in_the_air"] == ([], ["P05"])
    names = ["right_hip1", "right_knee", "right_foot1", "left_hip1", "left_knee"]
    m = SimpleNamespace(nu=5, actuator_trntype=np.zeros(5, dtype=np.int64), actuator_trnid=np.array([[j, -1] for j in range(5)]),
                        joint=lambda j: SimpleNamespace(name="robot:" + names[j]))
    w = SimpleNamespace(m=m, d=SimpleNamespace(qpos=np.zeros(5), qvel=np.zeros(5)), env=SimpleNamespace(dt=0.01), reps=5,
                        jq={n: i for i, n in enumerate(names)}, jd={n: i for i, n in enumerate(names)},
                        mocap_id=lambda name: 0)
    push, air = V.case_support_push(w, "push"), V.case_support_push(w, "in_the_air")
    assert isinstance(push, V.Driven) and push.steps == 70
    # the plant (t < 2 s): the controllers pull toward the hip -40, knee -110, ankle +30, left hip -90, knee -60
    assert push.command(10.0, w).tolist() == [-1.0, -1.0, 1.0, -1.0, -1.0]
    assert air.command(10.0, w)[0] == -1.0                  # the twin's hip toward -90
    # the push (t 2.25 s): the hip extends and the ankle points the foot down at full command; the knee
    # follows its target, 0.25 of 0.35 s from -110 toward -10 degrees
    w.d.qpos[1] = math.radians(-110.0 + (0.25 / 0.35) * 100.0)
    c = push.command(45.0, w)
    assert (c[0], c[2]) == (1.0, -1.0) and c[1] == pytest.approx(0.0, abs=1e-9)
    assert (c[3], c[4]) == (-1.0, -1.0)                     # the left leg still pulled to its lift
    w.d.qpos[0] = math.radians(-90.0)
    c = air.command(45.0, w)
    assert (c[0], c[2]) == (pytest.approx(0.0, abs=1e-12), -1.0)   # the twin's hip held at -90
    with pytest.raises(ValueError):
        V.case_support_push(w, "kick")


def test_v9_a_push_is_a_force_that_holds():
    """P05 asks the rise to hold over PUSH_HOLD_RECORDS records at each end: a one-record spike at the
    window's end, or a one-record dropout at its start, is no push."""
    from mimo_bridge.tools.observer import detectors as D
    assert D.PUSH_HOLD_RECORDS == 3
    n = 40

    class _Tr:
        pass
    tr = _Tr()
    tr.n = n
    tr.header = dict(weight_n=100.0)
    tr.continuous = np.ones(n, dtype=bool)
    knee = np.full(n, -100.0)
    knee[10:20] = np.linspace(-100.0, -60.0, 10)
    knee[20:] = -60.0
    tr.limb = {"right_knee": knee}
    tr.com = np.zeros((n, 3))
    tr.com[20:, 0] = 0.02

    def p05(force):
        ctx = _Tr()
        ctx.tr = tr
        z = np.zeros(n)
        ctx.sup = {"leg_R": np.asarray(force, dtype=float), "leg_L": z, "hand_L": z, "hand_R": z, "arm_L": z, "arm_R": z}
        ctx.det = lambda cap, i0, i1, **kw: dict(cap=cap, i0=i0, i1=i1, **kw)
        return [d for d in D.P05(ctx) if d["sub"] == "R:leg"]
    held = np.full(n, 5.0)
    held[18:] = 50.0                                        # a rise of 45 N, 0.45 body weights, that holds
    assert len(p05(held)) == 1
    spike = np.full(n, 5.0)
    spike[20] = 50.0                                        # the same rise in one record
    assert p05(spike) == []
    dip = np.full(n, 40.0)
    dip[10] = 0.0                                           # a steady 40 N with one dropout at a window's start
    dip[18:] = 45.0
    assert p05(dip) == []


# ---------------------------------------------------------------------- item 3b: when a turn is known
def _cont(n):
    return np.ones(n, dtype=bool)


def test_v9_a_turn_is_known_once_the_series_turns_back_stops_or_breaks():
    """primitives.turns_confirmed: a run is confirmed by a reversal of its line, by
    TURN_HOLD_RECORDS records without progress, or by a break in the record; until then it is open."""
    rise = list(np.linspace(0.0, 20.0, 6))                       # records 0-5, the peak at 5
    # a reversal: the first record 15 degrees under the peak confirms it
    x = np.array(rise + [18.0, 10.0, 4.0, 0.0])
    assert P.turns_confirmed(x, _cont(len(x)), 15.0)[0] == (0, 5, 20.0, 8)
    # a hold: 40 records (2 s) without going 0.75 degrees past the level reached; small wobbles
    # under that do not count as progress
    hold = P.TURN_HOLD_RECORDS
    assert hold == 40
    x = np.array(rise + [20.5, 19.8, 20.3] + [20.0] * 60)
    t = P.turns_confirmed(x, _cont(len(x)), 15.0)
    assert t[0] == (0, 6, 20.5, 5 + hold)                        # the extreme is the run's highest record
    # a break in the record ends the run at its last continuous record
    x = np.array(rise + [20.0] * 10)
    c = _cont(len(x))
    c[9] = False
    assert P.turns_confirmed(x, c, 15.0)[0][3] == 8
    # the series ends first: open
    x = np.array(rise + [20.0] * 10)
    assert P.turns_confirmed(x, _cont(len(x)), 15.0)[0][3] == -1
    # _turns keeps its three fields
    assert P._turns(x, _cont(len(x)), 15.0) == [(0, 5, 20.0)]


def test_v9_a_pause_as_long_as_an_acts_rise_ends_the_run():
    """A bend of 20 degrees, a pause, 20 degrees more: with a 2 s pause (TURN_HOLD_RECORDS) the bends
    are two runs, each a 0.25 s act; with a 1 s pause one run of 40 degrees, as before (its rise, 1.5
    s, is still inside the act timescale)."""
    def bends(pause):
        return np.array(list(np.linspace(0.0, 20.0, 6)) + [20.0] * pause + list(np.linspace(20.0, 40.0, 6)[1:]) + [40.0] * 5)
    x = bends(45)
    t = P._turns(x, _cont(len(x)), 15.0)
    assert [(a, b) for a, b, _ in t] == [(0, 5), (50, 55)] and [c for _, _, c in t] == [20.0, 20.0]
    x = bends(20)
    assert [(a, b, c) for a, b, c in P._turns(x, _cont(len(x)), 15.0)] == [(0, 30, 40.0)]


def test_v9_a_confirmed_turn_never_changes_with_later_records():
    """The property the digester needs: a turn confirmed within the first m records is the same turn,
    with the same start, extreme, change and confirming record, in the whole series; only an open
    turn can still move. Random walks, every prefix length."""
    rng = np.random.default_rng(3)
    for trial in range(20):
        x = np.cumsum(rng.normal(0.0, 1.5, 600)) + 3.0 * np.sin(np.arange(600) / 9.0)
        full = P.turns_confirmed(x, _cont(len(x)), 15.0)
        for m in range(20, 600, 11):
            for turn in P.turns_confirmed(x[:m], _cont(m), 15.0):
                if turn[3] >= 0:
                    assert turn in full, (trial, m, turn)


def test_v9_the_digester_finalizes_no_act_on_an_open_turn():
    """daemon.final_cut: a window's cut moves back before every act whose turn is still open where
    the window ends (the window after reads it again from its look-back), and out of any episode it
    then falls inside; a closed record finalizes everything."""
    from mimo_bridge.tools.observer import daemon as DM
    eps = [dict(i0=40, i1=70), dict(i0=10, i1=45)]
    dets = [dict(i0=80, i1=95, open_end=False), dict(i0=60, i1=99, open_end=True)]
    # the open act at 60 moves the cut to 59, inside the episode 40-70, then inside 10-45: 9
    assert DM.final_cut(90, eps, dets, closed=False) == 9
    assert DM.final_cut(90, eps, dets, closed=True) == 90
    # an open act starting after the cut is not finalized anyway
    assert DM.final_cut(50, [], [dict(i0=60, i1=99, open_end=True)], closed=False) == 50
    # without open acts the episode rule alone, as before
    assert DM.final_cut(90, eps, [dict(i0=80, i1=95)], closed=False) == 90
    assert DM.final_cut(50, eps, [], closed=False) == 9


def test_v9_turn_based_acts_say_whether_their_turn_is_known():
    """H05 (and the other turn-based rows) carry the turn's open_end for the digester."""
    def bend(then_back):
        n = 30 if then_back else 20
        tr = _Hands(n)
        x = np.full(n, 20.0)
        x[5:10] = np.linspace(20.0, 45.0, 5)
        x[10:] = 45.0
        if then_back:
            x[20:25] = np.linspace(45.0, 20.0, 5)
            x[25:] = 20.0
        tr.curl[:, 1, 0] = x                                   # the right index (SIDES L, R; DIGITS index first)
        return [d for d in _run(tr.done(), D.H05) if d["sub"] == "R:index"]
    open_act, = bend(False)
    assert open_act["open_end"] is True
    closed = bend(True)
    assert closed[0]["open_end"] is False and closed[0]["step0"] == open_act["step0"]


# ---------------------------------------------------------------------- defect 8: every reported row has its case
def test_v9_every_row_the_page_reports_beyond_babble_has_a_scripted_case_and_a_twin():
    """The production report of 2026-09-25 7:27 PM showed these rows or sub-rows past a stage; each
    has a case expecting it present and a twin expecting its detector absent."""
    from mimo_bridge.tools.observer import validate as V
    present = {}
    absent = {}
    for name, _, pres, abs_, _ in V.CASES:
        for p in pres:
            present.setdefault(p.split("@")[0], []).append(name)
        for a in abs_:
            absent.setdefault(a.split("@")[0].split(":")[0], []).append(name)
    rows = ["B03:R", "B03:L", "B06:L", "G02:arm_L_upper", "G03:arm_L_upper", "G05:caregiver", "H05:L:little",
            "H05:L:thumb", "H05:R:little", "H05:R:thumb", "H06:R:index", "HX3:R:little", "R02:box", "H04:R"]
    for row in rows:
        assert present.get(row), row
        assert absent.get(row.split(":")[0]), row
    lines = {c[0]: (c[2], c[3]) for c in V.CASES}
    assert lines["b03_hand_creeps_onto_cheek"] == ([], ["B03"])
    assert lines["hx3_toy_moved_into_little_fingertip"] == (["HX3:R:little@world_moved"], ["HX3@own", "R01", "R02"])
    assert lines["h06_index_bends_to_still_thumb"] == (["B02"], ["H06"])
    assert lines["b02_thumb_meets_index"] == (["B02", "H06:R:index"], [])
    assert lines["h01_touch_then_close"] == (["H01:R", "R02:box"], [])
    for name in ("b03_hand_creeps_onto_cheek", "g02_glance_at_left_upper_arm", "h06_index_bends_to_still_thumb",
                 "h05_little_with_ring", "p05_leg_extends_in_the_air"):
        assert name in V.BOUNDARY and name in lines


# ---------------------------------------------------------------------- item 9: the grasp on a smaller cube
def _fake_box_world():
    from types import SimpleNamespace
    names = ["world", "box_light", "box_medium"]
    m = SimpleNamespace(geom_size=np.array([[1.0, 1.0, 0.1], [0.03, 0.03, 0.03], [0.03, 0.03, 0.03]]),
                        geom_bodyid=np.array([0, 1, 2]), ngeom=3,
                        body_mass=np.array([0.0, 0.2, 0.4]), body_inertia=np.array([[0.0] * 3, [1.2e-4] * 3, [2.4e-4] * 3]),
                        body=lambda n: SimpleNamespace(id=names.index(n)))
    return SimpleNamespace(m=m)


def test_v9_the_grasp_cases_hold_a_smaller_cube_and_give_the_box_back():
    """Item 9 (coordinator, 8:20 PM: fix the test object): the grasp and palm-box cases make box_light a
    2.5 cm cube of the same density while they are built and recorded, and restore it after; every other
    case keeps the 6 cm box. _grasp refuses a grip under 150 degrees between the thumb's and the
    fingers' contact normals or one that lets the freed cube move over 5 mm in 3 s."""
    from mimo_bridge.tools.observer import validate as V
    assert V.GRASP_SIDE_M == 0.025 and V.GRASP_MIN_NORMALS_DEG == 150.0 and V.GRASP_MAX_HOLD_MM == 5.0
    w = _fake_box_world()
    with V._SmallBox(w):
        assert w.m.geom_size[1].tolist() == [0.0125] * 3
        assert w.m.body_mass[1] == pytest.approx(0.2 * (0.0125 / 0.03) ** 3)          # the same density
        assert w.m.body_inertia[1][0] == pytest.approx(w.m.body_mass[1] * 0.025 ** 2 / 6.0)
        assert w.m.geom_size[2].tolist() == [0.03] * 3                                  # the other boxes stay
    assert w.m.geom_size[1].tolist() == [0.03] * 3 and w.m.body_mass[1] == 0.2
    assert w.m.body_inertia[1].tolist() == [1.2e-4] * 3
    # the recording's scene hook holds the cube small at every frame and gives it back after
    calls = []
    scene = V._small_box_scene(w, lambda t, world: calls.append(t))
    scene(0.0, w)
    scene(0.2, w)
    assert calls == [0.0, 0.2] and w.m.geom_size[1][0] == 0.0125
    scene.after()
    assert w.m.geom_size[1][0] == 0.03 and w.m.body_mass[1] == 0.2
    # every grasp-family case, the palm-down ones too (fix 3: v8's palm-down arm poses pressed the upper arm
    # into the floor, 457 N), is built and recorded on the cube
    for name in ("case_h01_on_grasp", "case_hold_release_on_grasp", "_case_grasp_small", "case_top_press",
                 "case_push_resting_hand"):
        fn = getattr(V, name)
        assert fn.__name__ == name and "_SmallBox" in fn.__code__.co_names       # wrapped by _on_small_box


def test_v9_a_posed_recording_gives_back_what_its_scene_changed(monkeypatch, tmp_path):
    """ScriptWorld.record calls the case's scene `after` once the case is recorded (the small cube)."""
    from types import SimpleNamespace
    import mimo_bridge.observer_record as OR
    from mimo_bridge.tools.observer import validate as V
    monkeypatch.setattr(OR, "WorldRecorder", _Recorder)
    _Recorder.frames = []
    m = SimpleNamespace(nv=2)
    d = SimpleNamespace(time=0.0, qpos=np.zeros(3), qvel=np.zeros(2), ctrl=np.zeros(1))
    restarts = []
    env = SimpleNamespace(saccades=None, reflex=None, _cr=None, _target_home=np.zeros(3), _target_vel=np.zeros(3),
                          proprio=SimpleNamespace(fatigue=SimpleNamespace(cap=np.ones(1))), frame=0, dt=0.01,
                          restart_clock=lambda t: (restarts.append(t), setattr(d, "time", t)))
    d.time = 7.5                                            # where the case before this one left the clock
    w = V.ScriptWorld.__new__(V.ScriptWorld)
    w.env, w.m, w.d, w.reps, w.scratch = env, m, d, 5, str(tmp_path)
    w.mj = SimpleNamespace(mj_forward=lambda m_, d_: None, mj_differentiatePos=lambda m_, qv, dt, a, b: None)
    w.place_far = lambda: None
    w.command = lambda qv, motors: None
    done = []

    def scene(t, world):
        pass
    scene.after = lambda: done.append(True)
    V.ScriptWorld.record(w, "case", 2, lambda t: np.zeros(3), None, scene)
    assert done == [True] and len(_Recorder.frames) == 10
    # a posed case starts the clock over through the world too (BridgeEnv.restart_clock), never data.time alone
    assert restarts == [0.0] and abs(d.time - 10 * 0.01) < 1e-12


def test_v9_driven_cases_start_the_virtual_skin_over_with_the_clock(monkeypatch, tmp_path):
    """The observer qualification's validate gate (2026-09-26) stopped at the first physics step of its first
    driven case: ValueError "virtual skin time reversed". A world built like H84 (contact profile mechanical_v4)
    keeps a second clock in its virtual skin, which reset()'s last collect advanced to the settled time;
    ScriptWorld.drive set data.time back to 0 and left the skin's clock where it was.
    Here a BridgeEnv holding its real touch state (one skin patch of 4 taxels, VirtualTaxelSkin) steps that
    skin at every physics substep as BridgeEnv does, and carries a history (exposure memory and impaired
    function) from before the case. Two driven cases in a row must each run, and each must start its skin,
    its body function and its touch poolers as reset() starts them (BridgeEnv.restart_clock): nothing from
    the time before the case, whose clock the case has left behind."""
    from types import SimpleNamespace
    import mimo_bridge.observer_record as OR
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.touch_adapter import PatchSpec, TouchPooler
    from mimo_bridge.tools.observer import validate as V
    from lem.body_function import BodyFunction
    from lem.skin_mechanics import MechanicalExposure
    monkeypatch.setattr(OR, "WorldRecorder", _Recorder)
    timestep, substeps = 0.002, 5
    spec = PatchSpec(body_id=5, body_name="right_hand", n_points=4, scale_m=0.005)
    d = SimpleNamespace(time=0.0, qpos=np.zeros(4), qvel=np.zeros(4), qacc_warmstart=np.zeros(4), ctrl=np.zeros(2),
                        mocap_pos=np.zeros((1, 3)), xfrc_applied=np.zeros((3, 6)))
    env = BridgeEnv.__new__(BridgeEnv)                      # its touch state only, as __init__ and reset() build it
    env.model = SimpleNamespace(opt=SimpleNamespace(timestep=timestep))
    env.data, env.dt, env.frame = d, timestep * substeps, 0
    env.contact_profile, env.body_function_mode, env.rubbing_calibration = "mechanical_v4", "modulate", "v4"
    env.touch_sensitivity = np.ones((1, 2), dtype=np.float32)
    env.touch = SimpleNamespace(sensor_positions={5: np.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0],
                                                               [0.0, 0.005, 0.0], [0.005, 0.005, 0.0]])})
    env.pooler = TouchPooler([spec], env.dt, vib_window=4)
    env.pooler.mechanical_exposure = MechanicalExposure(1, pain_sensitivity=env.touch_sensitivity[:, 0])
    env._mechanical_pooler = TouchPooler([spec], timestep, vib_window=4)
    env._initialize_virtual_skin()
    env.body_function = BodyFunction(env.patch_names())
    # the history before the case: reset()'s collect at the settled crib time (100 physics steps), a memory and
    # an impaired taxel left by earlier touch
    skin = env._virtual_skin
    skin.advance([], 100 * timestep, timestep, aggregate=False)
    skin.exposure.values[:] = 0.5
    skin.function.quality[:] = 0.7
    env.body_function.quality[:] = 0.7
    env.body_function.last_time = 100 * timestep

    def step(*a, ctrl=None, render=None):                  # BridgeEnv.step's physics substeps advance the skin
        for _ in range(substeps):
            d.time += timestep
            env._virtual_skin.advance([], float(d.time), timestep, aggregate=False)
        env.body_function.last_time = float(d.time)
        env.frame += 1
    env.step = step
    w = V.ScriptWorld.__new__(V.ScriptWorld)
    w.env, w.m, w.d, w.reps, w.scratch = env, SimpleNamespace(), d, 5, str(tmp_path)
    w.mj = SimpleNamespace(mj_forward=lambda m_, d_: None)
    w.place_far = lambda: None
    case = V.Driven(lambda world: np.zeros(4), lambda t, world: np.zeros(2), 2)
    starts, skins = [], []

    def watch(s, rep, world):
        if s == 0 and rep == 0:                             # the case's first frame
            sk = world.env._virtual_skin
            skins.append(sk)
            starts.append(dict(clock=sk.last_time, exposure=float(np.abs(sk.exposure.values).max()),
                               quality=float(sk.function.quality.min()),
                               body_quality=float(world.env.body_function.quality.min())))
    for _ in range(2):
        V.ScriptWorld.drive(w, case, None, watch)           # raised "virtual skin time reversed" before the fix
    frame = timestep * substeps
    assert len(starts) == 2 and skins[0] is not skins[1]
    for start in starts:
        assert abs(start["clock"] - frame) < 1e-12          # the skin's clock is the case's clock
        assert start["exposure"] == 0.0                     # no memory from before the case
        assert start["quality"] == 1.0 and start["body_quality"] == 1.0     # intact, as reset() starts it
    assert abs(env._virtual_skin.last_time - 2 * 5 * frame) < 1e-12 and abs(d.time - 2 * 5 * frame) < 1e-12


# ---------------------------------------------------------------------- fix 2: the motors drove the movement
def test_v9_a_hand_moved_against_its_own_command_is_no_act():
    """The motor-work rule (coordinator, 8:40 PM): a hand act counts only when its joints' own motors did
    positive work on the movement - the recorded command times the joint's turn, summed over the act."""
    from mimo_bridge.tools.observer import validate as V
    driven = _settle_fan(0.3)                               # the fan with its side motors pushing its way
    assert [d["key"] for d in driven] == ["R:spread"] and driven[0]["measures"]["motor_work"] > 0
    # the same movement with the index's side motor commanded against it
    n = 40
    tr = _Hands(n)
    deg = _ramp(n, 16, 24, 0.0, 22.0)
    tr.qpos[:, 0] = np.radians(deg)
    tr.kin = _Kin()
    tr.ctrl = np.zeros((n, len(_JOINTS)), dtype=np.float16)
    tr.ctrl[4:25, 0] = -0.3                                  # the index side motor pulls the other way
    assert _run(tr.done(), D.HX1) == []
    assert D._against(-1e-6) and not D._against(1e-6) and not D._against(float("nan"))
    lines = {c[0]: (c[2], c[3]) for c in V.CASES}
    assert lines["h05_index_moved_against_its_command"] == ([], ["H05"])
    assert lines["hx2_cup_palm_moved_against_its_command"] == ([], ["HX2"])


# ---------------------------------------------------------------------- review: a roll the body drives
def _roll_track(command=None):
    """A made-up track rolling from the back onto the left side over records 10-30 (the upper body's roll 0 ->
    -80 degrees), while one motor-driven joint (qpos 0, the model's motor 0) turns 0.5 rad; the recorded
    command on that motor is `command` during the roll (None: no model and no commands, not judged)."""
    n = 60
    tr = _Hands(n)
    tr.roll_deg = _ramp(n, 10, 30, 0.0, -80.0)
    tr.shoulder_z = np.zeros((n, 2))
    tr.shoulder_z[:, 0] = -0.05                                # the left shoulder down
    tr.qpos[:, 0] = _ramp(n, 10, 30, 0.0, 0.5)
    if command is not None:
        tr.kin = _Kin()
        tr.ctrl = np.zeros((n, len(_JOINTS)), dtype=np.float16)
        tr.ctrl[10:31, 0] = command
    tr.done()
    return [d for d in D.P02(D.Context(tr, P.postures(tr)))]


def test_v9_a_roll_counts_only_when_the_bodys_own_motors_drive_it():
    """The independent review (2026-09-25): P02's scripted positive was a limp body turned by a torque from
    outside, and P02 has no motor rule, so the lock would freeze that a roll the body did not drive is P02.
    The rolls (P02, P03, P04) now ask what the hand rows ask: a command on the body's motors and positive
    motor work over the roll."""
    assert _roll_track(0.0) == []                               # limp: turned from outside (v9 before: P02)
    assert _roll_track(-0.5) == []                              # its motors pulled against the roll
    driven = _roll_track(0.5)
    assert [d["sub"] for d in driven] == ["L"] and driven[0]["measures"]["motor_work"] > 0
    assert [d["cap"] for d in _roll_track(None)] == ["P02"]     # no model, no commands: not judged


def test_v9_the_roll_cases_are_driven_by_the_body_and_the_torque_roll_is_their_twin():
    """p02_roll_to_side and p02_half_roll are driven by the body's own motors (validate.case_roll_driven: a
    roll synergy searched in the world's own physics); the fix-3 torque roll, every command zero, is the new
    twin p02_body_rolled_by_outside_torque (P02 absent by its motor rule)."""
    from types import SimpleNamespace
    from mimo_bridge.tools.observer import validate as V
    lines = {c[0]: (c[2], c[3]) for c in V.CASES}
    assert lines["p02_roll_to_side"] == (["P02"], [])
    assert lines["p02_half_roll"] == ([], ["P02"])
    assert lines["p02_body_rolled_by_outside_torque"] == ([], ["P02"])
    assert "p02_body_rolled_by_outside_torque" in V.BOUNDARY and "p02_half_roll" in V.BOUNDARY
    # the controller: held joints pulled to their start angles, the synergy's motors at full command from
    # ROLL_GO (ramped), back to holding after `stop`
    plan = dict(nu=3, hold=[(0, 0, 0, 0.1), (1, 1, 1, 0.0)], synergy={1: -1.0}, scale=1.0, stop=None)
    qpos, qvel = np.zeros(2), np.zeros(2)
    assert V.roll_command(2.0, qpos, qvel, plan).tolist() == pytest.approx([0.4, 0.0, 0.0])
    assert V.roll_command(9.0, qpos, qvel, plan).tolist() == pytest.approx([0.4, -1.0, 0.0])
    assert V.roll_command(6.0, qpos, qvel, dict(plan, scale=0.5)).tolist() == pytest.approx([0.4, -0.25, 0.0])
    assert V.roll_command(9.0, qpos, qvel, dict(plan, stop=8.0)).tolist() == pytest.approx([0.4, 0.0, 0.0])
    # the search's lines: onto the side = 3 record ends in a row at 70-120 degrees after starting on the back
    assert V.roll_reaches_side([0, -20, -50, -72, -75, -80]) and not V.roll_reaches_side([0, -50, -72, -75, 40])
    assert not V.roll_reaches_side([45, 72, 75, 80])            # it did not start on the back
    assert V.roll_stays_half([0, 20, 44, 30]) and not V.roll_stays_half([0, 20, 56]) and not V.roll_stays_half([0, 30])
    # the scene: no force from outside unless a torque is given (the twin's)
    names = ["world", "mobile_frame", "upper_body", "hip"]
    m = SimpleNamespace(ngeom=2, geom_bodyid=np.array([1, 2]), geom_contype=np.array([1, 1]),
                        geom_conaffinity=np.array([1, 1]), nu=4,
                        body=lambda b: SimpleNamespace(name=names[b] if isinstance(b, int) else b,
                                                       id=names.index(b) if isinstance(b, str) else b))
    w = SimpleNamespace(m=m, d=SimpleNamespace(xfrc_applied=np.ones((4, 6)), mocap_pos=np.zeros((1, 3))),
                        mocap_id=lambda name: 0)
    scene = V._roll_scene(w)
    scene(10.0, w)
    assert np.all(w.d.xfrc_applied == 0.0) and m.geom_contype.tolist() == [0, 1]   # the mobile's collisions off
    scene.after()
    assert m.geom_contype.tolist() == [1, 1]
    twin = V.case_roll_turned(w)
    assert twin.command(10.0, w).tolist() == [0.0] * 4          # every motor command zero


# ---------------------------------------------------------------------- review: the scenario fingerprint
def test_v9_the_scenario_fingerprint_sees_the_commands_the_eye_movers_and_the_objects(monkeypatch, tmp_path):
    """The independent review (2026-09-25): the lock's fingerprint hashed the joint positions alone, so
    h05_index_alone, h05_index_alone_motors_off and h05_index_moved_against_its_command (opposite verdicts,
    decided by the recorded commands) shared one fingerprint; so did the world's eye-mover cases and their
    own-movement twin, and the grasp cases' smaller cube is a model change. Each now changes it."""
    import mimo_bridge.observer_record as OR
    from mimo_bridge.tools.observer import validate as V
    n = 12
    qpos = np.cumsum(np.full((n, 4), 0.01), axis=0)
    base = dict(qpos=qpos, ctrl=np.full((n, 3), 0.3, dtype=np.float32), gaze_saccade=np.zeros(n, dtype=bool),
                gaze_reflex=np.zeros(n, dtype=bool), header=dict(model_fingerprint="cube-6cm"))
    variants = {                                                   # each differs from the positive in one input
        "positive": {},
        "same": {},
        "motors_off": dict(ctrl=np.zeros((n, 3), dtype=np.float32)),
        "against": dict(ctrl=np.full((n, 3), -0.3, dtype=np.float32)),
        "world_saccade": dict(gaze_saccade=np.arange(n) == 6),
        "world_reflex": dict(gaze_reflex=np.arange(n) == 6),
        "small_cube": dict(header=dict(model_fingerprint="cube-2.5cm")),
    }
    chunks = {}
    for name, change in variants.items():
        folder = tmp_path / name / "observer" / "world" / "b0"
        folder.mkdir(parents=True)
        path = folder / "chunk_000000000_1_00000.npz"
        path.write_bytes(b"")
        chunks[os.path.normcase(os.path.abspath(str(path)))] = dict(base, **change)
    monkeypatch.setattr(OR, "read_chunk", lambda p, header=None: chunks[os.path.normcase(os.path.abspath(p))])
    fp = {name: V._qpos_hash(str(tmp_path / name)) for name in variants}
    assert fp["same"] == fp["positive"]                          # the same trajectory, commands, flags and model
    for name in ("motors_off", "against", "world_saccade", "world_reflex", "small_cube"):
        assert fp[name] != fp["positive"], name                  # v9 before: one fingerprint for all


# ---------------------------------------------------------------------- review: a test per row for each rule
# The independent review (2026-09-25) planted 31 faults, one at a time, in a copy of the branch's code: 13 passed
# every world-free test - the motor gate removed from H05, from H07 and H08, from HX2 or from H04 (only HX1's gate
# had a unit test); the work rule removed from H05, from HX2, from H07 and H08 or from H04; both gates removed from
# H05; open_end dropped from HX1 (only H05's was tested); H08's timescale dropped; and the range start of item 3b
# reverted, twice. Each rule now has a test on each row it governs.
def _row_track(row, x, command="drive", tail=5):
    """A made-up track on which one hand row's joint series follows x (degrees) and its motor gets `command`:
    'drive' (0.3 the way the joint moves in each record), 'weak' (0.0005 that way: under MOTOR_ON), 'off' (0),
    'against' (0.3 the other way) or None (no model, no commands: not judged). tail: still records added after x."""
    x = np.r_[np.asarray(x, dtype=np.float64), np.full(tail, float(x[-1]))]
    n = len(x)
    tr = _Hands(n)
    kin = _Kin()
    adr = {"H05": 5, "H07": 6, "H08": 7, "HX2": 4, "HX1": 0, "H04": 5}[row]
    if row == "H05":
        tr.curl[:, 1, 0] = x
        kin.flex_q = {("R", "index"): [adr]}
    elif row == "H04":
        for f in range(4):
            tr.curl[:, 1, f] = x
        kin.flex_q = {("R", dg): [adr] for dg in ("index", "middle", "ring", "little")}
    elif row == "H07":
        tr.wrist[:, 1, 1] = x
        kin.wrist_q, kin.wrist_kind = {"R": {"hand2": adr}}, {"R": {"hand2": "flexion"}}
    elif row == "H08":
        tr.wrist[:, 1, 0] = x
        kin.wrist_q, kin.wrist_kind = {"R": {"hand1": adr}}, {"R": {"hand1": "rotation"}}
    tr.qpos[:, adr] = np.radians(x)
    if command is not None:
        tr.kin = kin
        tr.ctrl = np.zeros((n, len(_JOINTS)), dtype=np.float16)
        step = np.r_[0.0, np.sign(np.diff(x))]
        tr.ctrl[:, adr] = dict(drive=0.3, weak=0.0005, off=0.0, against=-0.3)[command] * step
    fn = dict(H04=D.H04, H05=D.H05, H07=D.H07, H08=D.H08, HX1=D.HX1, HX2=D.HX2)[row]
    return _run(tr.done(), fn)


_BEND = np.r_[np.zeros(5), np.linspace(0.0, 30.0, 9)]          # 30 degrees in 0.4 s from rest
_CYCLE = np.r_[np.full(5, 10.0), np.linspace(10.0, 40.0, 6), np.linspace(40.0, 10.0, 6)[1:],
               np.linspace(10.0, 40.0, 6)[1:], np.linspace(40.0, 10.0, 6)[1:]]


def test_v9_every_hand_row_asks_its_own_motors_to_drive_the_movement():
    """MOTOR_ON and the work rule on each row: the same movement is an act when its joint's motor drives it,
    and none when the motor is silent or pulls against it."""
    for row in ("H04", "H05", "H07", "H08", "HX1", "HX2"):
        x = _CYCLE if row == "H04" else _BEND
        assert len(_row_track(row, x, "drive")) == 1, row
        assert _row_track(row, x, "off") == [], row                 # silent motors
        assert _row_track(row, x, "weak") == [], row                # the gate: a command under MOTOR_ON
        assert _row_track(row, x, "against") == [], row             # the work rule
        assert len(_row_track(row, x, None)) == 1, row              # no commands recorded: not judged


def test_v9_every_turn_based_row_says_whether_its_turn_is_known():
    """open_end on each row: an act whose turn runs to the end of the record is open (the digester waits for
    it); the same act once the joint turns back is known."""
    back = np.r_[_BEND, np.linspace(30.0, 0.0, 9)[1:]]
    for row in ("H05", "H07", "H08", "HX1", "HX2"):
        open_act, = _row_track(row, _BEND, "drive")
        known = _row_track(row, back, "drive")
        assert open_act["open_end"] is True and known[0]["open_end"] is False, row
    cycle_open, = _row_track("H04", _CYCLE, "drive")                # its last opening runs to the end
    cycle_known, = _row_track("H04", np.r_[_CYCLE, np.linspace(10.0, 40.0, 6)[1:], np.linspace(40.0, 25.0, 4)[1:]],
                              "drive", tail=45)
    assert cycle_open["open_end"] is True
    assert cycle_known["open_end"] is False


def test_v9_the_timescale_holds_on_h08_as_on_the_other_rows():
    """H08's timescale: a wrist rotation of 35 degrees over 26 s is a drift, over 0.4 s an act."""
    assert len(_row_track("H08", np.r_[np.zeros(5), np.linspace(0.0, 35.0, 9)], None)) == 1
    assert _row_track("H08", np.r_[np.zeros(5), np.linspace(0.0, 35.0, 521)], None) == []


def test_v9_a_movement_from_a_lower_point_after_a_hold_is_its_own_turn():
    """Item 3b's range start, the record's example (segment 39, body 0, left index from step 391,931): a
    15.9-degree bend to 39.9 degrees, back to 25.5, a wobble at 36-39 for 2 s, then a bend to 47.5. The first
    run is confirmed by the hold while the finger wobbles under 39.9; the next run is searched from the lowest
    record since, so the bend from 25.5 to 47.5 is found. Searched from 39.9 (the range start reverted) it is
    7.6 degrees and found nowhere."""
    wobble = 37.5 + 1.5 * np.sin(np.arange(44) / 3.0)
    x = np.r_[np.linspace(24.0, 39.9, 6), np.linspace(39.9, 25.5, 5)[1:], np.linspace(25.5, 36.0, 5)[1:], wobble,
              np.linspace(wobble[-1], 47.5, 6)[1:], np.full(5, 47.5)]
    turns = P.turns_confirmed(x, _cont(len(x)), 15.0)
    assert [(round(x[a], 1), round(x[b], 1)) for a, b, _, _ in turns] == [(24.0, 39.9), (25.5, 47.5)]


def _posture_track(roll, command):
    n = len(roll)
    tr = _Hands(n)
    tr.roll_deg = np.asarray(roll, dtype=np.float64)
    tr.shoulder_z = np.zeros((n, 2))
    tr.shoulder_z[:, 0] = -0.05
    tr.qpos[:, 0] = np.radians(np.abs(tr.roll_deg))            # a motor-driven joint turning with the roll
    if command is not None:
        tr.kin = _Kin()
        tr.ctrl = np.zeros((n, len(_JOINTS)), dtype=np.float16)
        tr.ctrl[:, 0] = command * np.r_[0.0, np.sign(np.diff(np.abs(tr.roll_deg)))]
    tr.done()
    return D.Context(tr, P.postures(tr))


def test_v9_every_roll_row_asks_the_bodys_motors_to_drive_it():
    """The roll rows' motor rule (_body_drove) on P03 (side to back) and P04 (back to belly), as on P02."""
    back = _ramp(50, 10, 30, -80.0, 0.0)                          # from the left side onto the back
    belly = _ramp(70, 10, 50, 0.0, -170.0)                        # from the back over the left side onto the belly
    for fn, roll in ((D.P03, back), (D.P04, belly)):
        assert len(fn(_posture_track(roll, 0.5))) == 1, fn.__name__
        assert fn(_posture_track(roll, 0.0)) == [], fn.__name__
        assert fn(_posture_track(roll, -0.5)) == [], fn.__name__
        assert len(fn(_posture_track(roll, None))) == 1, fn.__name__


# ---------------------------------------------------------------------- review: the grip and the scripted truth
def test_v9_the_scripted_truth_is_judged():
    """The independent review (2026-09-25): the truth values were reported, never judged, so a locked truth could
    read nothing (HX1.amount_deg since v8) or the wrong movement without failing its case. A case now fails when
    its first detection of the capability reads a truth value off by more than TRUTH_TOLERANCE of it, or reads
    none."""
    from mimo_bridge.tools.observer import validate as V

    def judge(*values):
        dets = [dict(cap="H05", sub="R:index", measures=dict(change_deg=v)) for v in values]
        return V._judge("h05", 16, dets, ["H05"], [], {"H05.change_deg": 40.0}, "h")
    assert judge(40.0)["ok"] and judge(-40.0)["ok"] and judge(41.9)["ok"]      # within 5%; a straightening reads -40
    assert not judge(43.0)["ok"] and judge(43.0)["truth_met"] == {"H05.change_deg": False}
    assert not V._judge("h05", 16, [dict(cap="H05", sub="R:index", measures={})], ["H05"], [], {"H05.change_deg": 40.0},
                        "h")["ok"]                                             # nothing to read
    assert V.TRUTH_TOLERANCE == 0.05


def test_v9_a_posed_cube_rests_at_the_height_its_weight_gives():
    """The independent review (2026-09-25): the grasp cases posed the 3.6 g cube 1 mm into the floor (10.8 N
    recorded between the cube and the floor) and squeezed each touching fingertip 1 mm into it after the physics
    settled the grip (25.8 N on the index fingertip, 6.3 N on the thumb). The cube now stands where physics puts it
    (validate._cube_rest_height, THE GRIP IS PHYSICAL) and the grip keeps no squeeze. On a two-body model (the floor
    and a 2.5 cm, 3.6 g cube): at its rest height the floor carries the cube's weight; posed 1 mm deeper, more."""
    from types import SimpleNamespace
    import mujoco
    from mimo_bridge.tools.observer import validate as V
    assert "squeeze_m" not in V.GRASP_LAYOUT_V9
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.005"/><default><geom solref="-10000 -100"/></default>'   # 10 N per mm, as H84's
        '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
        '<body name="box_light" pos="0 0 0.02"><freejoint/><geom type="box" size="0.0125 0.0125 0.0125" mass="0.0036"/>'
        '</body></worldbody></mujoco>')
    d = mujoco.MjData(m)

    def contacts_now(q):
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
    w = SimpleNamespace(m=m, d=d, mj=mujoco, free_adr=lambda name: 0, contacts_now=contacts_now)
    q = np.array(d.qpos)
    z = V._cube_rest_height(w, q)
    assert 0.0125 - 1e-4 < z < 0.0125                                           # resting on the floor, not in it
    q_rest = np.array(q)
    q_rest[0:3] = (2.6, 2.0, z)
    weight = 0.0036 * 9.81
    at_rest = V._cube_forces(w, q_rest)["world"]
    q_deep = np.array(q_rest)
    q_deep[2] = 0.0125 - 0.001                                                  # v9 before: 1 mm into the floor
    deep = V._cube_forces(w, q_deep)["world"]
    assert abs(at_rest - weight) < 0.25 * weight and deep > 1.5 * at_rest      # (H84's world: 10.8 N at 1 mm)


def test_v9_a_posed_grip_records_the_commands_its_digits_hold_it_with(monkeypatch, tmp_path):
    """The recorder track's review (2026-09-26): posed, the lifted grasp cases recorded the index side pushing the
    cube 2.2-2.5 times harder than the thumb, because a posed frame's forces are forward dynamics of a state no
    physics settled, with no command holding the digits. The grip is now the state physics reaches with the cube free
    in the air (validate._grip_in_air), and while the hand grips the digits' motors carry the commands that make the
    settling pulls (pose.hold, THE GRIP IS HELD BY THE HAND'S MOTORS): ScriptWorld.record applies them on top of the
    frame's commands before MuJoCo computes the frame."""
    from types import SimpleNamespace
    import mimo_bridge.observer_record as OR
    from mimo_bridge.tools.observer import validate as V
    monkeypatch.setattr(OR, "WorldRecorder", _Recorder)
    _Recorder.frames = []
    m = SimpleNamespace(nv=2)
    d = SimpleNamespace(time=0.0, qpos=np.zeros(3), qvel=np.zeros(2), ctrl=np.zeros(3))
    env = SimpleNamespace(saccades=None, reflex=None, _cr=None, _target_home=np.zeros(3), _target_vel=np.zeros(3),
                          proprio=SimpleNamespace(fatigue=SimpleNamespace(cap=np.ones(1))), frame=0, dt=0.01,
                          restart_clock=lambda t: setattr(d, "time", t))     # BridgeEnv.restart_clock's physics side
    seen = []
    w = V.ScriptWorld.__new__(V.ScriptWorld)
    w.env, w.m, w.d, w.reps, w.scratch = env, m, d, 5, str(tmp_path)
    w.mj = SimpleNamespace(mj_forward=lambda m_, d_: seen.append(np.array(d_.ctrl)),
                           mj_differentiatePos=lambda m_, qv, dt, a, b: None)
    w.place_far = lambda: None
    w.command = lambda qv, motors: w.d.ctrl.__setitem__(slice(None), 0.3)

    def pose(t):
        return np.zeros(3)
    pose.hold = lambda t: {1: -0.2} if t > 1.0 else {}
    V.ScriptWorld.record(w, "grip", 2, pose)
    frames = seen[1:]                                         # the first forward is the reset pose's
    assert len(frames) == 10
    assert all(f.tolist() == [0.3, 0.3, 0.3] for f in frames[:5])                # step 0: no grip held yet
    assert all(f.tolist() == pytest.approx([0.3, -0.2, 0.3]) for f in frames[5:])  # step 1: the hold on top
    # the commands: a joint torque through its own motor (gear x gain), and the share of its range it needs
    names = ["robot:right_th_distal", "robot:right_ff_distal"]
    mm = SimpleNamespace(nu=2, actuator_trntype=np.zeros(2, dtype=np.int64), actuator_trnid=np.array([[0, -1], [1, -1]]),
                         joint=lambda j: SimpleNamespace(name=names[j]), actuator_gainprm=np.ones((2, 10)),
                         actuator_gear=np.array([[2.0] + [0.0] * 5, [0.5] + [0.0] * 5]),
                         actuator_ctrlrange=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
                         actuator_forcerange=np.array([[-1.0, 1.0], [-1.0, 0.5]]))
    hold, share = V._hold_ctrl(SimpleNamespace(m=mm), {"right_th_distal": 0.5, "right_ff_distal": 0.1})
    assert hold == {0: pytest.approx(0.25), 1: pytest.approx(0.2)} and share == pytest.approx(0.4)   # 0.2 of 0.5
    _, share = V._hold_ctrl(SimpleNamespace(m=mm), {"right_ff_distal": 0.5})
    assert share == pytest.approx(2.0)                        # over the motor's range: cannot be held exactly
    # the blend: the floor grip's commands on the floor, the air grip's lifted, none outside the grip
    g = dict(floor_hold={0: 0.1}, air=dict(hold={0: 0.3, 1: 0.2}))
    assert V._grip_hold(g, 16.0, 0.0, 17.0, 26.0) == {}
    assert V._grip_hold(g, 18.0, 0.0, 17.0, 26.0) == {0: pytest.approx(0.1), 1: pytest.approx(0.0)}
    assert V._grip_hold(g, 24.0, 1.0, 17.0, 26.0) == {0: pytest.approx(0.3), 1: pytest.approx(0.2)}
