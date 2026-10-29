# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The observer validation's cases, expectation grammar, verdicts and babble table, without a world.

Section 27 of addendum v4: positive, negative and boundary cases; a composite behaviour broken one
link at a time; false-positive rates from babble-only runs (tools/observer/validate.py). These
tests read the case table, the kept recordings' headers and first records, and the babble
summaries; none builds a MuJoCo world. The joint-only trajectories (H05, HX1, HX2) are checked by
evaluating their pose functions on a world stand-in made of the kept header's joints.

The opt-in test at the end records and digests cases in the saved world model (validate.ModelWorld,
about 1 GB). It runs only with OBSERVER_VALIDATION_DRYRUN=1; start it after
`python -m mimo_bridge.tools.commit_guard --need-gb 1.5`.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pytest

from mimo_bridge.tools.observer import catalog as CAT
from mimo_bridge.tools.observer import detectors as D
from mimo_bridge.tools.observer import validate as V
from mimo_bridge.tools.observer.detectors import ALL_IDS, EXTRA_IDS

ROOT = str(CAT.ROOT)
SCRATCH = os.path.join(ROOT, "Saved", "mimo_bridge", "observer", "validation_scratch")
KEPT = os.path.join(SCRATCH, V.DEFAULT_BASE_RECORDING)
BASELINES = os.path.join(ROOT, "Saved", "mimo_bridge", "observer", "baselines")
KNOWN = set(ALL_IDS) | set(EXTRA_IDS)


def _det(cap, sub="", **flags):
    d = dict(cap=cap, sub=sub, key=sub or cap, step0=0, step1=1, measures={})
    d.update(flags)
    return d


# ---------------------------------------------------------------------- the case table
def test_every_case_is_well_formed():
    names = [c[0] for c in V.CASES]
    assert len(names) == len(set(names)), "case names repeat"
    for name, build, present, absent, truth in V.CASES:
        assert callable(build), name
        assert isinstance(present, list) and isinstance(absent, list), name
        for want in present + absent:
            spec, _, flag = want.partition("@")
            assert V._cap_of(want) in KNOWN, (name, want)
            assert not flag or flag in V.QUALIFIERS, (name, want)
            assert spec.strip() == spec and spec, (name, want)
        for key in truth:
            cap, _, measure = key.partition(".")
            assert cap in KNOWN and measure, (name, key)
        # an expectation both present and absent (same spelling) could never be met
        assert not set(present) & set(absent), name


def test_boundary_and_known_defect_registers_name_real_cases():
    names = {c[0] for c in V.CASES}
    for name, (caps, why) in V.BOUNDARY.items():
        assert name in names, name
        assert caps and all(c in KNOWN for c in caps), name
        assert why
    for name, why in V.KNOWN_DEFECTS.items():
        assert name in names and why, name


def test_the_requested_detectors_have_positive_and_negative_cases():
    cov = V.coverage()
    grasp = ("H02", "R04", "R05", "R06", "H03")
    page = ("G01", "G03", "H05", "HX2", "HX1", "G05", "R03", "B06")
    for cap in grasp + page:
        assert cov[cap]["present"], cap
        assert cov[cap]["absent"], cap
    # a boundary case for every page detector where one was feasible (G05 has none: its contract
    # names no quantity to vary)
    for cap in ("G01", "G03", "H05", "HX2", "HX1", "R03", "B06"):
        assert cov[cap]["boundary"], cap
    # before 2026-09-24 evening the cases covered 13 detectors
    assert len(cov) >= 27


def test_the_grasp_cases_break_one_link_each():
    """Each grasp case keeps the links before the broken one and loses the ones after it."""
    case = {c[0]: (set(c[2]), set(c[3])) for c in V.CASES}
    caps = lambda s: {V._cap_of(x) for x in s}
    p, a = case["grasp_contact_no_opposition"]
    assert {"R04", "H02", "R05", "R06", "H03"} <= caps(a) and not p
    # the grip whose object the floor carries is no grasp and no hold (the physics grip); its lifted twin is
    p, a = case["grasp_opposition_no_lift"]
    assert not p and {"R04", "H02", "R05", "R06", "H03"} <= caps(a)
    p, a = case["grasp_lift_no_retention"]
    assert {"R04", "R05"} <= caps(p) and {"H02", "R06", "H03", "R08"} <= caps(a)
    p, a = case["grasp_lift_and_hold"]
    assert {"R04", "R05", "H02"} <= caps(p) and {"R06", "H03"} <= caps(a)
    p, a = case["grasp_carry_and_place"]
    assert {"R04", "R05", "H02", "R06", "H03", "R08"} <= caps(p)
    # R04 on the crib hand with real dynamics: the thumb against the fingers holding the cube in the air by the
    # hand's own motors (the posed grips' forces cannot balance: the recorder track, 2026-09-26)
    p, a = case["grasp_pinch_held_driven"]
    assert "R04:R" in p and {"H03", "R06", "R08"} <= caps(a)
    # the two lifted cases are driven by the body's own motors, never posed (the coordinator's decision of
    # 2026-09-26: a posed lifted grip carried the floor grip into the air and could not balance)
    builders = {c[0]: c[1] for c in V.CASES}
    for name, kind in (("grasp_lift_and_hold", "hold"), ("grasp_lift_no_retention", "no_retention")):
        code = builders[name].__code__
        assert "case_grasp_driven" in code.co_names and kind in code.co_consts, name
    # accidental object motion: the same touch, made by the world or by the hand
    p, a = case["grasp_toy_moved_into_still_hand"]
    assert "HX3@world_moved" in p and "HX3@own" in a
    p, a = case["grasp_hand_touches_still_toy"]
    assert any(x.startswith("HX3") and x.endswith("@own") for x in p) and "HX3@world_moved" in a


# ---------------------------------------------------------------------- expectations and verdicts
def test_expectation_qualifiers_read_the_provenance_flags():
    own = _det("R04", "R:box")
    moved = _det("R04", "R:box", world_moved=True)
    script = _det("G01", "L", scripted=True)
    touched = _det("H02", "R", caregiver_touch=True)
    assert V._match(own, "R04") and V._match(own, "R04:R") and V._match(own, "R04:R:box@own")
    assert not V._match(own, "R04:L") and not V._match(own, "R05")
    assert V._match(moved, "R04@world_moved") and not V._match(moved, "R04@own")
    assert not V._match(own, "R04@world_moved")
    assert V._match(script, "G01@scripted") and not V._match(script, "G01@own")
    assert V._match(touched, "H02@caregiver_touch") and not V._match(touched, "H02@own")
    # without a qualifier the flags are not looked at, as before the qualifiers existed
    assert V._match(moved, "R04") and V._match(script, "G01:L")


def test_an_unknown_qualifier_is_refused():
    with pytest.raises(ValueError):
        V._match(_det("R04"), "R04@nobody")


def test_verdicts_and_confusion_key_on_the_bare_detector():
    dets = [_det("HX3", "R:middle", world_moved=True, measures={"force_n": 3.0}),
            _det("R01", "R", measures={"closed_cm": 20.0})]
    c = V._judge("toy", 27, dets, ["HX3@world_moved"], ["HX3@own", "R01"], {"R01.closed_cm": 21.0}, "h")
    assert c["present_found"] == {"HX3@world_moved": True}
    assert c["absent_clear"] == {"HX3@own": True, "R01": False}
    assert not c["ok"]
    assert c["flagged"] == ["HX3:R:middle@world_moved"]
    assert c["measure_errors"]["R01.closed_cm"] == dict(scripted=21.0, measured=20.0, error=1.0, met=True)
    conf = V._confusion([c])
    assert conf == {"HX3": dict(tp=1, fn=0, fp=0, tn=1), "R01": dict(tp=0, fn=0, fp=1, tn=0)}


def test_known_defects_are_split_from_unexpected_failures(monkeypatch):
    monkeypatch.setitem(V.KNOWN_DEFECTS, "hx1_sideways_sweep", "a defect the detectors do not fix yet")
    out = V.split_failures(["hx1_sideways_sweep", "g01_saccade_head_still"])
    assert list(out["known_defects"]) == ["hx1_sideways_sweep"]
    assert out["unexpected"] == ["g01_saccade_head_still"]
    assert V.split_failures([]) == dict(known_defects={}, unexpected=[])


def test_a_builder_result_is_read_with_or_without_eyes_and_scene():
    pose = lambda t: None
    eyes = lambda t, q: q
    scene = lambda t, w: None
    assert V._built((pose, 16)) == (pose, 16, None, None)
    assert V._built((pose, 14, eyes)) == (pose, 14, eyes, None)
    assert V._built((pose, 22, None, scene)) == (pose, 22, None, scene)
    assert V._built((pose, 28, 0.001)) == (pose, 28, None, None)        # case_to_head's measured error


# ---------------------------------------------------------------------- trajectories without a world
def test_timeline_interpolates_between_keyframes_and_holds_outside():
    at = V._timeline(None, [(4.0, dict(a=0.0, b=1.0)), (8.0, dict(a=2.0)), (12.0, dict(b=3.0))])
    assert at(0.0) == dict(a=0.0, b=1.0)
    assert at(6.0)["a"] == pytest.approx(1.0) and at(6.0)["b"] == pytest.approx(1.5)
    assert at(10.0) == pytest.approx(dict(a=2.0, b=2.5))
    assert at(20.0) == dict(a=2.0, b=3.0)
    assert V._ramp(5.0, 4.0, 8.0, 0.0, 40.0) == pytest.approx(10.0)


class _JointWorld:
    """A stand-in for ScriptWorld made of the kept recording's joints and first record: enough for
    the cases that only turn hinges (base, jq, jrange, set_curl)."""

    def __init__(self, folder):
        from mimo_bridge.observer_record import read_chunk
        z = read_chunk(sorted(glob.glob(os.path.join(folder, "observer", "world", "b0", "chunk_*.npz")))[0])
        self.jq, self.jrange = {}, {}
        for j in z["header"]["joints"]:
            nm = j["name"].split(":")[-1]
            self.jq[nm] = int(j["qpos"])
            self.jrange[nm] = (float(j["range"][0]), float(j["range"][1]))
        self.q_base = np.array(z["qpos"][0])

    base = V.ScriptWorld.base
    set_curl = V.ScriptWorld.set_curl


needs_kept = pytest.mark.skipif(not os.path.isdir(KEPT), reason="the kept validation recordings are not on this machine")


def _curl(w, q, side, digit):
    pre = "left_" if side == "L" else "right_"
    key = {"index": "ff", "middle": "mf", "ring": "rf", "little": "lf", "thumb": "th"}[digit]
    roles = ("middle", "distal") if digit == "thumb" else ("knuckle", "middle", "distal")
    return float(np.mean([math.degrees(q[w.jq["%s%s_%s" % (pre, key, r)]]) for r in roles]))


def _side_sum(w, q):
    return sum(abs(math.degrees(q[w.jq["right_%s_side" % k]])) for k in ("ff", "mf", "rf", "lf"))


@needs_kept
def test_hx1_trajectories_change_the_summed_side_angles_as_documented():
    w = _JointWorld(KEPT)
    base = _side_sum(w, w.base())
    peak = {}
    for kind in ("spread", "sweep", "spread_10"):
        pose, steps = V.case_hx1(w, kind)
        assert steps == 20
        peak[kind] = _side_sum(w, pose(10.0)) - base
        assert _side_sum(w, pose(19.0)) - base == pytest.approx(0.0, abs=1e-9)     # gathered back
    assert peak["spread"] == pytest.approx(26.2, abs=0.05)
    assert peak["sweep"] == pytest.approx(32.0, abs=0.05)
    assert peak["spread_10"] == pytest.approx(10.2, abs=0.05)
    # the sweep moves every finger the same way: the gaps between neighbouring fingers do not change
    pose, _ = V.case_hx1(w, "sweep")
    side = lambda q: [math.degrees(q[w.jq["right_%s_side" % k]]) for k in ("ff", "mf", "rf", "lf")]
    assert np.diff(side(pose(10.0))) == pytest.approx(np.diff(side(w.base())), abs=1e-9)
    # the slow drift reaches the same spread only after 8 s (160 steps)
    pose, steps = V.case_hx1(w, "slow_drift")
    assert steps == 170
    assert _side_sum(w, pose(24.0)) - base < 4.0
    assert _side_sum(w, pose(164.0)) - base == pytest.approx(26.2, abs=0.05)


@needs_kept
def test_hx2_trajectories_move_only_the_cupping_hinge_or_only_the_fingers():
    w = _JointWorld(KEPT)
    j = w.jq["right_lf_meta"]
    q0 = w.base()
    for deg in (30.0, 20.0, 10.0):
        pose, steps = V.case_hx2(w, deg)
        assert math.degrees(pose(10.0)[j] - q0[j]) == pytest.approx(deg, abs=1e-6)
        assert math.degrees(pose(19.0)[j] - q0[j]) == pytest.approx(0.0, abs=1e-9)
        changed = np.nonzero(pose(10.0) != q0)[0]
        assert list(changed) == [j]
    pose, _ = V.case_hx2(w, whole_hand=True)
    q = pose(10.0)
    assert q[j] == q0[j]
    for dg in ("index", "middle", "ring", "little"):
        assert _curl(w, q, "R", dg) == pytest.approx(40.0, abs=1e-6)


@needs_kept
def test_h05_trajectories_move_one_digit_or_two():
    w = _JointWorld(KEPT)
    digits = ("index", "middle", "ring", "little", "thumb")

    def change(kind, side):
        pose, _ = V.case_h05_more(w, kind)
        a, b = pose(0.0), pose(12.0)
        return {dg: _curl(w, b, side, dg) - _curl(w, a, side, dg) for dg in digits}

    ch = change("left_index", "L")
    assert ch["index"] == pytest.approx(40.0, abs=1e-6)
    assert all(abs(ch[d]) < 1e-9 for d in digits if d != "index")
    assert all(abs(v) < 1e-9 for v in change("left_index", "R").values())
    ch = change("thumb", "R")
    assert ch["thumb"] == pytest.approx(40.0, abs=1e-6) and all(abs(ch[d]) < 1e-9 for d in digits[:4])
    ch = change("straighten", "R")
    assert ch["index"] == pytest.approx(-40.0, abs=1e-6) and all(abs(ch[d]) < 1e-9 for d in digits[1:])
    ch = change("two_fingers", "R")
    assert ch["index"] == pytest.approx(40.0, abs=1e-6) and ch["middle"] == pytest.approx(40.0, abs=1e-6)
    # H05's own reading of the pair: the other four digits move 10 degrees on average, over its 5
    others = [ch[d] for d in digits if d != "index"]
    assert np.mean(np.abs(others)) == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------- detector fixes of observer_detectors_v8
# Each test below fails on the v7 detectors and passes on v8 (defects 1-5 of
# docs/gain_of_function_v4/agents/observer_validation.md); each also keeps a twin the fix must not
# change. The tracks are made up: the real Entities and Track's own contact-pair and time methods,
# with only the per-record fields these detectors read.
from mimo_bridge.tools.observer import detectors as D
from mimo_bridge.tools.observer import primitives as P
from mimo_bridge.tools.observer.features import Entities, Track

_KINDS = ["self", "object", "caregiver", "floor", "wall", "art", "ceiling", "static"]
_ENTS = [dict(name="right_hand", kind="self", group="hand_R_palm"),
         dict(name="right_thdistal", kind="self", group="finger_R_thumb"),
         dict(name="right_ffdistal", kind="self", group="finger_R_index"),
         dict(name="left_hand", kind="self", group="hand_L_palm"),
         dict(name="upper_body", kind="self", group="trunk"),
         dict(name="head", kind="self", group="head"),
         dict(name="box_light", kind="object", group="box"),
         dict(name="floor", kind="floor", group="floor")]
PALM, THUMB, INDEX, BOX, FLOOR = 0, 1, 2, 6, 7
_SIDE_JOINTS = [dict(name=n, qpos=i, range=[-0.436, 0.436]) for i, n in enumerate(
    ["right_ff_side", "right_mf_side", "right_rf_side", "right_lf_side",
     "left_ff_side", "left_mf_side", "left_rf_side", "left_lf_side"])]


class _Synth:
    pair_series = Track.pair_series
    entity_mask = Track.entity_mask
    object_mask = Track.object_mask
    seconds = Track.seconds

    def __init__(self, n):
        self.header = dict(entities=_ENTS, entity_kind=[_KINDS.index(e["kind"]) for e in _ENTS],
                           objects=[dict(entity=BOX, name="box_light", kind="box", root=1, mocap=-1, free=True)],
                           joints=_SIDE_JOINTS, weight_n=100.0)
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
        self.closure = np.full((n, 2), 20.0)
        self.palm = np.zeros((n, 2, 3))
        self.palm[:, 0] = (0.0, -0.3, 0.1)                 # the left hand far away
        self.obj_pos = np.zeros((n, 1, 3))
        self.qpos = np.zeros((n, len(_SIDE_JOINTS)))
        self.touch = []                                    # (record, entity a, entity b)

    def contact(self, a, b, records):
        self.touch += [(int(i), a, b) for i in records]

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
        self.c_row = np.asarray([r for r, _, _ in t], dtype=np.int64)
        self.c_e = np.asarray([(a, b) for _, a, b in t], dtype=np.int64).reshape(-1, 2)
        self.c_fmax = np.full(len(t), 2.0)
        self.c_fmean = np.full(len(t), 2.0)
        self.c_pos = np.asarray([self.obj_pos[r, 0] for r, _, _ in t]).reshape(-1, 3)
        self._c_sorted = True
        return self


def _run(tr, *fns, prims=()):
    ctx = D.Context(tr, list(prims))
    return [d for fn in fns for d in fn(ctx)]


def _box_with_palm(tr, offset=(0.0, 0.0, -0.036)):
    tr.obj_pos[:, 0] = tr.palm[:, 1] + np.asarray(offset)


def test_v8_a_resting_hand_sliding_a_box_along_the_floor_is_no_hold_or_carry():
    """Defect 1: v7 called it a hold (H02) and a move while holding (R06)."""
    tr = _Synth(60)
    tr.palm[:, 1] = (0.17, 0.26, 0.065)
    tr.palm[10:41, 1, 0] += np.linspace(0.0, 0.06, 31)       # 6 cm along the floor in 1.5 s
    tr.palm[41:, 1, 0] += 0.06
    _box_with_palm(tr)
    tr.contact(PALM, BOX, range(5, 56))
    tr.contact(BOX, FLOOR, range(0, 60))                     # on the floor throughout
    dets = _run(tr.done(), D.H02, D.R06)
    assert [d["cap"] for d in dets] == []
    # the twin: the same hand carries the box 3 cm up off the floor first - a hold and a carry
    tr = _Synth(60)
    tr.palm[:, 1] = (0.17, 0.26, 0.065)
    tr.palm[10:16, 1, 2] += np.linspace(0.0, 0.03, 6)
    tr.palm[16:, 1, 2] += 0.03
    tr.palm[16:47, 1, 0] += np.linspace(0.0, 0.06, 31)
    tr.palm[47:, 1, 0] += 0.06
    _box_with_palm(tr)
    tr.contact(PALM, BOX, range(5, 56))
    tr.contact(BOX, FLOOR, range(0, 11))
    dets = _run(tr.done(), D.H02, D.R06)
    assert sorted(d["cap"] for d in dets) == ["H02", "R06"]
    h02 = [d for d in dets if d["cap"] == "H02"][0]
    assert h02["measures"]["carried_cm"] == pytest.approx(math.hypot(3.0, 6.0), abs=0.01)


def test_v8_spread_is_the_fingers_fanning_within_two_seconds():
    """Defect 2: v7 summed the four |side angles| with no time bound."""
    def hx1(targets, rise_steps):
        tr = _Synth(40 + rise_steps)
        up = np.clip((np.arange(tr.n) - 5) / float(rise_steps), 0.0, 1.0)
        for j, deg in targets.items():
            tr.qpos[:, j] = math.radians(deg) * up
        return _run(tr.done(), D.HX1)

    sweep = hx1({0: 8, 1: 8, 2: 8, 3: 8}, 4)                 # all four fingers toward the thumb
    assert sweep == []
    spread = hx1({0: 8, 2: -8, 3: -12}, 4)                  # index toward the thumb, ring and little away
    assert len(spread) == 1 and spread[0]["key"] == "R:spread"
    assert spread[0]["measures"]["fan_deg"] == pytest.approx(20.0, abs=1e-6)
    assert hx1({0: 8, 2: -8, 3: -12}, 160) == []            # the same spread drifting over 8 s
    assert len(hx1({0: 8, 2: -8, 3: -12}, 38)) == 1         # ... and over 1.9 s it is one act


def test_v8_a_box_that_slipped_and_fell_is_no_placed_release():
    """Defect 3: v7 read the box's speed only in the record after contact ended, when it had landed."""
    tr = _Synth(45)
    tr.palm[:, 1] = (0.17, 0.26, 0.065)
    tr.palm[10:13, 1, 2] += [0.01, 0.02, 0.03]
    tr.palm[13:, 1, 2] += 0.03
    _box_with_palm(tr)
    tr.obj_pos[21:, 0, 2] = tr.obj_pos[0, 0, 2]              # back on the floor within one 50 ms record
    tr.contact(PALM, BOX, range(5, 22))                      # the last touch is that record
    tr.contact(BOX, FLOOR, list(range(0, 11)) + list(range(21, 45)))
    tr.done()
    assert tr.obj_speed[21, 0] == pytest.approx(0.6, abs=1e-6)
    assert _run(tr, D.R08, prims=P.lifts_drops(tr)) == []
    # the twin: lowered to the floor and held still 150 ms before the hand lets go - placed
    tr = _Synth(45)
    tr.palm[:, 1] = (0.17, 0.26, 0.065)
    tr.palm[10:13, 1, 2] += [0.01, 0.02, 0.03]
    tr.palm[13:20, 1, 2] += 0.03
    tr.palm[20:26, 1, 2] += np.linspace(0.03, 0.0, 6)
    _box_with_palm(tr)
    tr.contact(PALM, BOX, range(5, 29))
    tr.contact(BOX, FLOOR, list(range(0, 11)) + list(range(25, 45)))
    tr.done()
    assert [d["cap"] for d in _run(tr, D.R08, prims=P.lifts_drops(tr))] == ["R08"]


def _release(withdraw_mps=0.0, box_mps=0.0):
    """A box lifted 3 cm, set down, then the hand opens (records 26-30) and lets go at 30."""
    tr = _Synth(50)
    tr.palm[:, 1] = (0.17, 0.26, 0.065)
    tr.palm[10:13, 1, 2] += [0.01, 0.02, 0.03]
    tr.palm[13:20, 1, 2] += 0.03
    tr.palm[20:26, 1, 2] += np.linspace(0.03, 0.0, 6)
    _box_with_palm(tr)
    tr.closure[:, 1] = 30.0
    tr.closure[26:31, 1] = np.linspace(30.0, 5.0, 5)
    tr.closure[31:, 1] = 5.0
    tr.palm[31:, 1, 2] += withdraw_mps * 0.05 * np.arange(1, 20)
    tr.obj_pos[31:, 0, 0] += box_mps * 0.05 * np.arange(1, 20)
    tr.contact(PALM, BOX, range(5, 31))
    tr.contact(BOX, FLOOR, list(range(0, 11)) + list(range(25, 50)))
    return tr.done()


def test_v8_a_hand_withdrawing_after_letting_go_is_a_controlled_release():
    """Defect 4: v7 compared the box's speed with the palm's."""
    assert [d["cap"] for d in _run(_release(withdraw_mps=0.2), D.H03)] == ["H03"]
    assert [d["cap"] for d in _run(_release(), D.H03)] == ["H03"]
    assert _run(_release(box_mps=1.2), D.H03) == []          # the box itself leaves at 1.2 m/s: escape


def test_v8_a_push_starting_from_rest_contact_is_a_push():
    """Defect 5: v7 read the palm's velocity only in the record before the box moved."""
    tr = _Synth(40)
    tr.palm[:, 1] = (0.17, 0.26, 0.065)
    tr.palm[11:, 1, 0] += 0.0015 * np.arange(1, 30)          # hand and box start together, 3 cm/s
    _box_with_palm(tr)
    tr.contact(PALM, BOX, range(5, 36))
    tr.contact(BOX, FLOOR, range(0, 40))
    tr.done()
    dets = _run(tr, D.O01, prims=P.object_motions(tr))
    assert [d["cap"] for d in dets] == ["O01"] and dets[0]["measures"]["direction_error_deg"] < 1.0
    # the twin: a hand arriving at 10 cm/s and pushing on - read from the record before, as in v7
    tr = _Synth(40)
    tr.palm[:, 1] = (0.17, 0.26, 0.065)
    tr.palm[5:11, 1, 0] = 0.17 - 0.005 * np.arange(6, 0, -1)
    tr.palm[11:, 1, 0] = 0.17 + 0.005 * np.arange(1, 30)
    tr.obj_pos[:, 0] = (0.17, 0.26, 0.029)
    tr.obj_pos[11:, 0, 0] += 0.005 * np.arange(1, 30)
    tr.contact(PALM, BOX, range(10, 36))
    tr.contact(BOX, FLOOR, range(0, 40))
    tr.done()
    assert [d["cap"] for d in _run(tr, D.O01, prims=P.object_motions(tr))] == ["O01"]


# ---------------------------------------------------------------------- the babble table
def _summary(kind, steps, caps, vision=None):
    return dict(schema="observer_baseline_v1", kind=kind, steps=steps, detector_version="observer_detectors_vX",
                generated=1790000000.0, vision=vision, caps=caps, recordings=[{}] * 8, episodes=3)


def test_babble_rates_per_detector_and_process(tmp_path):
    folder = tmp_path / "sig0123456789abcd"
    folder.mkdir()
    caps_limp = {"H02": dict(count=8, occasions=8, steps=20000), "G01": dict(count=0, occasions=0, steps=20000),
                 "R04": dict(count=0, occasions=0, steps=20000), "O12": dict(count=0, occasions=0, steps=20000)}
    caps_orders = {"H02": dict(count=27, occasions=24, steps=20000),
                   "G01": dict(count=2041, occasions=711, steps=20000,
                               subs={"L": dict(count=1036, occasions=356, steps=20000)}),
                   "R04": dict(count=0, occasions=0, steps=20000), "O12": dict(count=0, occasions=0, steps=20000)}
    caps_off = dict(caps_orders, G01=dict(count=605, occasions=302, steps=20000))
    for kind, caps, vision in (("limp", caps_limp, False), ("random_orders", caps_orders, True),
                               ("random_orders_vision_off", caps_off, False)):
        (folder / (kind + ".json")).write_text(json.dumps(_summary(kind, 20000, caps, vision)), encoding="utf-8")
    (folder / "not_a_summary.json").write_text("[]", encoding="utf-8")
    out_path = tmp_path / "babble_rates.json"
    out = V.babble_rates(str(folder), str(out_path))
    assert json.loads(out_path.read_text(encoding="utf-8"))["rows"]["H02"] == out["rows"]["H02"]
    assert out["world_signature"] == "sig0123456789abcd"
    assert list(out["processes"]) == ["limp", "random_orders", "random_orders_vision_off"]
    assert out["processes"]["random_orders"]["bodies"] == 8
    h02 = out["rows"]["H02"]["processes"]
    assert h02["limp"] == dict(count=8, occasions=8, body_steps=20000, per_1000_steps=0.4, occasions_per_1000_steps=0.4)
    assert h02["random_orders"]["per_1000_steps"] == 1.35
    assert out["rows"]["G01"]["subs"]["L"]["random_orders"]["per_1000_steps"] == 51.8
    assert out["false_positives_in_a_limp_body"] == {"H02": 0.4}
    assert "R04" in out["never_fired"] and "O12" not in out["never_fired"]
    assert any("30.2 to 102.0" in h for h in out["headlines"])
    # every catalog row is present, observable or not
    assert set(CAT.load()["by_id"]) <= set(out["rows"])


@pytest.mark.skipif(not os.path.isdir(BASELINES), reason="no kept babble baselines on this machine")
def test_the_written_babble_table_matches_the_kept_summaries():
    path = os.path.join(ROOT, "docs", "gain_of_function_v4", "observer_validation", "babble_rates.json")
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    folder = os.path.join(ROOT, table["source_folder"])
    if not os.path.isdir(folder):
        pytest.skip("the table's baseline folder is not on this machine")
    for kind in table["processes"]:
        with open(os.path.join(folder, kind + ".json"), encoding="utf-8") as f:
            s = json.load(f)
        if s.get("detector_version") != table["processes"][kind]["detector_version"]:
            pytest.skip("the summaries were re-digested after the table was written: rerun babble-rates")
        for cid, c in s["caps"].items():
            row = table["rows"][cid]["processes"][kind]
            assert row["count"] == c["count"] and row["occasions"] == c["occasions"], (kind, cid)
            assert row["per_1000_steps"] == pytest.approx(1000.0 * c["count"] / s["steps"], abs=1e-3), (kind, cid)


# ---------------------------------------------------------------------- opt-in: the cases in the saved world model
dry_run_only = pytest.mark.skipif(
    os.environ.get("OBSERVER_VALIDATION_DRYRUN") != "1" or not os.path.isdir(KEPT),
    reason="set OBSERVER_VALIDATION_DRYRUN=1 (about 1 GB; after tools/commit_guard) and keep "
           "Saved/mimo_bridge/observer/validation_scratch")


@dry_run_only
def test_dry_run_meets_every_expectation_but_the_known_defects(tmp_path):
    rep = V.dry_run(None, KEPT, str(tmp_path))
    assert len(rep["cases"]) == len(V.CASES)
    assert rep["failures"]["unexpected"] == [], rep["failures"]
    # the grasp is the physics-settled one: thumb and fingers push against each other past R04's line
    # (detectors v9: contact normals, OPPOSITION_MIN_DEG). The scripted grasp of 2026-09-24 FAILS this
    # under v9 - its thumb and ring finger push on adjacent faces of the box (docs/gain_of_function_v4/
    # agents/observer_recorder_2026-09-25.md) - until the validation cases' grasp is rebuilt
    by = {c["name"]: c for c in rep["cases"]}
    r04 = [d["measures"]["opposition_deg"] for d in by["grasp_lift_and_hold"]["detections"] if d["cap"] == "R04"]
    assert r04 and max(r04) >= D.OPPOSITION_MIN_DEG
    assert not any(d["cap"] in ("R04", "H02") for d in by["grasp_opposition_no_lift"]["detections"])
    # the crib hand's own pinch, real dynamics: the thumb's and the fingers' pushes all but opposite
    pinch = [d["measures"] for d in by["grasp_pinch_held_driven"]["detections"] if d["cap"] == "R04"]
    assert pinch and pinch[0]["thumb_fingers_deg"] >= 170.0 and pinch[0]["grip_share"] >= 0.8
    assert "G01:L@scripted" in by["g01_world_saccade"]["flagged"]


@dry_run_only
def test_the_scene_hook_leaves_no_world_mover_or_turned_toy_behind(tmp_path):
    w = V.ModelWorld(KEPT, str(tmp_path))
    quat0 = np.array(w.d.mocap_quat)
    for name in ("g01_world_saccade", "grasp_toy_moved_into_still_hand"):
        build = next(c[1] for c in V.CASES if c[0] == name)
        pose, steps, eyes, scene = V._built(build(w))
        w.record(name, steps, pose, eyes, scene)
        assert w.env.saccades is None and w.env.reflex is None
    w.place_far()
    assert np.array_equal(np.array(w.d.mocap_quat), quat0)


def test_the_cases_left_out_of_the_lock_stay_in_the_run_and_out_of_the_proposal():
    """NOT_LOCKED (the coordinator's decision of 2026-09-26): P02's three cases are run and recorded but a proposal
    leaves them out of its expectations, boundary registry and golden detections, lists them with their reason,
    and its `why` tells the operator the choice can be overruled at the approval."""
    names = {c[0] for c in V.CASES}
    assert set(V.NOT_LOCKED) == {"p02_roll_to_side", "p02_half_roll", "p02_body_rolled_by_outside_torque"}
    assert set(V.NOT_LOCKED) <= names
    assert all(r.startswith("not locked: the body cannot roll itself yet (trunk authority)") for r in V.NOT_LOCKED.values())
    hashes = {n: "h" + n for n in names}
    golden = {n: dict(qpos_hash="h" + n, detections=[]) for n in names}
    code, gold, why = V.proposal_parts(hashes, golden)
    assert not set(V.NOT_LOCKED) & set(code["cases"]) and set(code["cases"]) == names - set(V.NOT_LOCKED)
    assert not set(V.NOT_LOCKED) & set(code["boundary"]) and not set(V.NOT_LOCKED) & set(gold["cases"])
    assert code["not_locked"] == V.NOT_LOCKED
    assert code["cases"]["grasp_lift_and_hold"]["qpos_hash"] == "hgrasp_lift_and_hold"
    assert "NOT LOCKED" in why and "p02_roll_to_side" in why and "overrule" in why
