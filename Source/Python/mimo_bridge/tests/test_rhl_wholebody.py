# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The whole-body decision reader (docs/gain_of_function_v4/remembered_hand_learning_2026-10-11/wholebody.py).

Plain cases first (an object at the palm is in reach, one 50 cm away is not; a hidden or untouchable object never is),
then the counting rules the reader borrows from the tracker, the per-opportunity rates, the recontact grouping, the
commands and control readings, the decision rule, the refusals of broken inputs, and every defect found so far, each
with a test that fails on the code that had it:
  built Oct 11 (first version): the exposure check passing a window with no whole bin; the detector re-run keeping an
  act the digester drops; R01 reaches toward the never-collidable dragonfly read as contact opportunities; recontact
  examples listing first contacts;
  the independent review of Oct 11 (D:/UEMA_Research/external/D/UEMA_scratch/remembered_hand_learning_2026-10-11/reviews/WHOLEBODY_REVIEW.md):
  blocker 1 finalize-copy's copy order and short record files; blocker 2 decide accepting any input; blocker 3 the
  endpoint jobs as registerable rows; fix 4 abandoned futures in the saved exposure; fix 5 coverage across trainer
  processes; fix 6 exposure bins missing from the saved file and quarantined windows; fix 7 live saved acts that do
  not join; fix 8 a detector version unlike the code's; fix 9 the decision rule's statuses and catalogue tie-break;
  fix 10 each act judged at its defining record; fix 11 the reused full-roll support line; fix 12 the known answers
  and fake-folder tests; fix 13 the corrected texts; and the coordinator's 6-of-8 selection rule.
The real-data test reproduces the observer's saved acts and exposure and the independent recount's known answers; it
builds the run's MuJoCo model, so it runs only with RHL_WHOLEBODY_REAL=1 (under the memory guard).
"""
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
HERE = REPO / "docs" / "gain_of_function_v4" / "remembered_hand_learning_2026-10-11"
MODULE = HERE / "wholebody.py"
JOBS = HERE / "wholebody_endpoint_jobs.json"
REAL_RUN = Path("C:/Users/bhart/Documents/UEMA_devA_private/Saved/mimo_bridge/experiments/"
                "H84_fresh_drivers_20260918_seg41j_1526")


def _load():
    spec = importlib.util.spec_from_file_location("rhl_wholebody", str(MODULE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WB = _load()


def _versions():
    from mimo_bridge.tools.observer import daemon
    return daemon.ANALYSIS_VERSION, daemon.DETECTOR_VERSION


# ---------------------------------------------------------------------- synthetic helpers
def _feats(n=100, step0=1000, pid=1, kind=None, cont=None, authors=None, order=None, posture=None, floor=None,
           trans=None, live=None, lo=None, hi=None, outside=None, both_on=None):
    step = np.arange(step0, step0 + n, dtype=np.int64)
    ones = np.ones(n, dtype=bool)
    k = dict(hand=np.ones((n, 2), dtype=bool), reach=np.ones((n, 2), dtype=bool),
             reach_seen=np.ones((n, 2), dtype=bool), hands_apart=ones.copy(),
             reach_both=ones.copy(), on_back=ones.copy(), on_side=np.zeros(n, dtype=bool), body=ones.copy())
    k.update(kind or {})
    lv = ones.copy() if live is None else np.asarray(live, dtype=bool)
    f = dict(n=n, step=step, pid=pid, live=lv, cont=(np.r_[False, np.ones(n - 1, dtype=bool)] if cont is None
                                                     else np.asarray(cont, dtype=bool)),
             authors=np.asarray(authors or ["planner"] * n, dtype=object),
             order_seq=np.asarray(order if order is not None else np.arange(n) // 5, dtype=np.int64),
             posture=np.asarray(posture or ["back"] * n, dtype=object),
             floor_bw=np.ones(n) if floor is None else np.asarray(floor, dtype=np.float64),
             transients=trans or [], kind=k, countable=ones.copy(), palm_dist=np.full(n, 0.3),
             roll=np.zeros(n), nearest=np.full((n, 2), 0.2), hand_onsets=[], both_on=both_on or {},
             outside_force=(np.full((n, 8), 30.0) if outside is None else np.asarray(outside, dtype=np.float64)),
             frames=np.full(n, 5, dtype=np.int64))
    f["window"] = lv & (step >= (step0 if lo is None else lo)) & (step <= (step0 + n - 1 if hi is None else hi))
    return f


def _act(cap, i0, i1=None, step0=1000, sub="L", key=None, parts=None, f=0, **kw):
    d = dict(cap=cap, step0=step0 + i0, step1=step0 + (i0 + 5 if i1 is None else i1), sub=sub, key=key or sub,
             parts=parts if parts is not None else ["hand_" + sub[:1]], _f=f, _i0=i0, _i1=(i0 + 5 if i1 is None else i1))
    d.update(kw)
    d.setdefault("_id", i0)
    return d


# ---------------------------------------------------------------------- plain cases: opportunity
def test_reach_mask_counts_only_visible_collidable_eligible_objects_within_45_cm():
    palm = np.zeros((1, 2, 3))
    palm[0, 1] = [5.0, 0, 0]                                    # the right hand far from everything
    objs = np.array([[[0.40, 0, 0], [0.50, 0, 0], [0.10, 0, 0], [0.10, 0, 0], [0.10, 0, 0], [np.nan] * 3]])
    flags = np.array([[3, 3, 2, 1, 3, 3]])                      # bit 1 visible, bit 2 collidable
    eligible = np.array([True, True, True, True, False, True])  # the fifth is the caregiver
    m = WB.reach_mask(palm, objs, flags, eligible, 0.45)
    assert m.tolist() == [[True, False]]
    for k, want in enumerate([True, False, False, False, False, False]):
        e = np.zeros(6, dtype=bool)
        e[k] = eligible[k]
        assert bool(WB.reach_mask(palm, objs, flags, e, 0.45)[0, 0]) is want, k


def test_a_reach_needs_a_seen_object_a_contact_a_touchable_one():
    """Found on the baseline (Oct 11): R01 counts reaches toward the dragonfly and the bee, never collidable in this
    world. A reach (R01, R03) is read against SEEN objects within 45 cm; a contact (R02, recontact) against seen AND
    touchable ones."""
    palm = np.zeros((1, 2, 3))
    objs = np.array([[[0.20, 0, 0]]])
    eligible = np.array([True])
    seen_only = np.array([[1]])
    assert WB.reach_mask(palm, objs, seen_only, eligible, 0.45, need=1)[0].tolist() == [True, True]
    assert WB.reach_mask(palm, objs, seen_only, eligible, 0.45)[0].tolist() == [False, False]
    assert WB.ROW["R01"]["kind"] == WB.ROW["R03"]["kind"] == "reach_seen"
    assert WB.ROW["R02"]["kind"] == WB.ROW["R02_recontact"]["kind"] == "reach"
    f = _feats(n=10, kind=dict(reach_seen=np.ones((10, 2), dtype=bool), reach=np.zeros((10, 2), dtype=bool)))
    assert WB.body_row("R01", [f], [_act("R01", 2)])["successes"] == 1
    out = WB.body_row("R02", [f], [_act("R02", 2, sub="ball", key="dragonfly", parts=["hand_L"])])
    assert out["opportunity_steps"] == 0 and out["acts_outside_opportunity"] == 1 and out["occasions_per_1000"] is None


def test_reach_both_needs_one_object_near_both_palms():
    palm = np.array([[[0.0, 0, 0], [0.6, 0, 0]]])
    objs = np.array([[[0.1, 0, 0], [0.5, 0, 0]]])
    flags = np.array([[3, 3]])
    assert not WB.reach_both_mask(palm, objs, flags, np.array([True, True]), 0.45)[0]
    objs = np.array([[[0.3, 0, 0], [0.5, 0, 0]]])
    assert WB.reach_both_mask(palm, objs, flags, np.array([True, True]), 0.45)[0]
    assert not WB.reach_both_mask(palm, objs, flags, np.array([False, True]), 0.45)[0]


def test_runs_break_where_the_record_breaks():
    mask = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    cont = np.array([0, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
    assert WB.runs_of(mask, cont) == [(0, 2), (4, 5), (6, 7)]


def test_act_side_reads_parts_then_sub_row():
    assert WB.act_side(dict(parts=["hand_R"], sub="box")) == "R"
    assert WB.act_side(dict(parts=["finger_L_index"], sub="L:index")) == "L"
    assert WB.act_side(dict(parts=[], sub="L:index")) == "L"
    assert WB.act_side(dict(parts=["trunk"], sub="L-down")) is None


# ---------------------------------------------------------------------- counting rules borrowed from the tracker
def test_occasions_follow_the_tracker_one_second_gap_per_part():
    occ, n, rep = WB.occasions_of([0, 10], ["a", "a"])
    assert n == 1 and rep == 0
    occ, n, rep = WB.occasions_of([0, 20, 40], ["a", "a", "a"])
    assert n == 3 and rep == 2
    occ, n, rep = WB.occasions_of([0, 0], ["a", "b"])
    assert n == 2 and rep == 0
    assert WB.occasions_of([], [])[1:] == (0, 0)


def test_rate_is_none_without_opportunity_never_zero():
    assert WB.rate(0, 0) is None and WB.rate(3, 0) is None
    assert WB.rate(0, 500) == 0.0 and WB.rate(5, 500) == 10.0


def _det(**kw):
    d = dict(cap="R01", sub="L", key="L:box", step0=10, step1=15, body=0, pid=1, provenance="planner-selected",
             live=True, scripted=False, caregiver_touch=False, world_moved=False)
    d.update(kw)
    return d


def test_counted_follows_the_tracker_rule():
    dets = [_det(), _det(scripted=True), _det(caregiver_touch=True), _det(world_moved=True),
            _det(provenance="probe-forced"), _det(provenance="external-rest"),
            _det(evidence=dict(credit=dict(provenance="baseline"))), _det(live=False), _det(provenance="mixed")]
    assert WB.counted(dets).tolist() == [True, False, False, False, False, False, False, False, True]
    assert WB.counted(dets, live=[False] * len(dets)).sum() == 0


def test_window_candidates_include_earlier_approaches_with_a_later_contact():
    dets = [_det(step0=90, step1=99), _det(step0=99, step1=105), _det(step0=100, step1=110),
            _det(step0=200, step1=205), _det(step0=201, step1=210), _det(step0=150, step1=160, run="other")]
    assert [d["step0"] for d in WB.window_detections(dets, "mine", 100, 200)] == [99, 100, 200]


def test_contact_cohort_uses_the_defining_record_at_both_window_boundaries():
    f = _feats(n=40, lo=1010, hi=1020)
    before = _act("R02", 5, i1=15, _id=12, sub="box", parts=["hand_L"])
    after = _act("R02", 18, i1=30, _id=25, sub="box", parts=["hand_L"])
    out = WB.body_row("R02", [f], [before, after])
    assert out["successes"] == 1
    assert out["acts_counted"] == 1 and out["acts_outside_window"] == 1
    assert out["opportunity_steps"] == 22


# ---------------------------------------------------------------------- one body, one row
def test_body_row_counts_success_per_opportunity_and_reports_acts_outside_it():
    reach = np.zeros((100, 2), dtype=bool)
    reach[:50, 0] = True
    reach[60:80, 1] = True
    f = _feats(kind=dict(reach_seen=reach))
    acts = [_act("R01", 5), _act("R01", 30), _act("R01", 70, sub="R"), _act("R01", 90, sub="R")]
    out = WB.body_row("R01", [f], acts)
    assert out["opportunity_steps"] == 70 and out["opportunity_unit"] == "hand model steps"
    assert out["acts_in_opportunity"] == 3 and out["acts_outside_opportunity"] == 1
    assert out["successes"] == 3 and out["occasions"] == 3 and out["repeated_occasions"] == 1
    assert out["occasions_per_1000"] == pytest.approx(1000.0 * 3 / 70)
    assert out["opportunity_stretches"] == 2 and out["stretches_with_success"] == 2
    none = WB.body_row("R01", [_feats(kind=dict(reach_seen=np.zeros((100, 2), dtype=bool)))], [])
    assert none["opportunity_steps"] == 0 and none["occasions_per_1000"] is None


def test_a_window_counts_only_its_own_model_steps():
    out = WB.body_row("P05", [_feats(n=100, lo=1020, hi=1059)], [])
    assert out["opportunity_steps"] == 40 and out["opportunity_stretches"] == 1


def test_recontact_counts_occasions_after_the_first_on_the_same_object():
    f = _feats()
    acts = [_act("R02", 0, sub="box", key="box_light", parts=["hand_L"]),
            _act("R02", 30, sub="box", key="box_light", parts=["hand_L"]),
            _act("R02", 60, sub="box", key="box_light", parts=["hand_L"]),
            _act("R02", 10, sub="box", key="box_heavy", parts=["hand_R"])]
    # R02's occasions are per sub-row (the object kind, "box"): starts 0, 10, 30, 60 make three (10 is 0.5 s after 0)
    assert WB.body_row("R02", [f], acts)["occasions"] == 3
    got = []
    again = WB.body_row("R02_recontact", [f], acts, collect=got)
    assert again["successes"] == 2 and again["occasions"] == 2 and again["repeated_occasions"] == 1
    assert again["repeats_count_rule"] is False
    # the row's examples are its successes: the recontacts, never a first contact (defect found Oct 11)
    assert [a["step0"] for a in got] == [1030, 1060]


def test_hand_rows_count_both_hands_as_opportunity():
    out = WB.body_row("H05", [_feats(n=10)], [_act("H05", 2, sub="L:index", parts=["finger_L_index"], alone_inside=True)])
    assert out["opportunity_steps"] == 20 and out["successes"] == 1 and out["alone_inside"]["yes"] == 1


# ---------------------------------------------------------------------- fix 10: the defining record
def test_each_act_is_judged_at_its_defining_record():
    """Review fix 10: R02 begins at its approach, which R02 does not limit to 45 cm; O07 begins with one hand; R03
    begins at the look, up to 1 s before the reach. Each is judged where its own condition can hold."""
    reach = np.zeros((100, 2), dtype=bool)
    reach[20:, 0] = True                                         # the object comes within 45 cm at record 20
    f = _feats(kind=dict(reach=reach, reach_seen=reach), both_on={"box_light": np.r_[np.zeros(50), np.ones(50)] > 0})
    r02 = _act("R02", 5, i1=30, sub="box", key="box_light", parts=["hand_L"], measures=dict(time_to_contact_s=1.0))
    assert WB.defining_index(r02, f) == 25                       # 5 + 1.0 s / 50 ms
    r02["_id"] = WB.defining_index(r02, f)
    out = WB.body_row("R02", [f], [r02])
    assert out["successes"] == 1 and out["acts_judged_after_first_record"] == 1
    assert out["acts_moved_by_defining_record"] == 1             # at its first record it was outside the opportunity
    r03 = _act("R03", 10, i1=40, sub="L", measures=dict(gaze_lead_s=0.6))
    assert WB.defining_index(r03, f) == 22
    r03_late = _act("R03", 10, i1=40, sub="L", measures=dict(gaze_lead_s=-0.2))
    assert WB.defining_index(r03_late, f) == 10                  # the look came during the reach: its first record
    o07 = _act("O07", 40, i1=70, sub="box_light", parts=["hand_L", "hand_R"], objects=["box_light"])
    assert WB.defining_index(o07, f) == 50
    other = _act("H05", 7, sub="L:index", parts=["finger_L_index"])
    assert WB.defining_index(other, f) == 7


# ---------------------------------------------------------------------- commands and control
def test_act_commands_lists_commands_and_writer_shares():
    authors = ["planner"] * 5 + ["reflex_noise"] * 5 + ["planner"] * 5 + ["external_rest"] * 5
    order = [7] * 5 + [8] * 5 + [9] * 5 + [-1] * 5
    f = _feats(n=20, authors=authors, order=order)
    c = WB.act_commands(f, 3, 12)
    assert c["commands"] == [7, 8, 9] and c["command_writers"] == ["planner", "reflex_noise", "planner"]
    assert c["steps_by_writer"] == {"planner": 5, "reflex_noise": 5} and c["n_steps"] == 10
    s = WB._summarize_commands([dict(commands=c), dict(commands=WB.act_commands(f, 15, 19))])
    assert s["acts_written_by_one_writer"] == {"external_rest": 1} and s["acts_with_a_planner_command"] == 1
    assert s["share_of_model_steps_by_writer"]["planner"] == pytest.approx(5 / 15)


def test_act_control_reads_support_loss_jolts_floor_and_posture():
    f = _feats(n=40, trans=[dict(kind="support_loss", step0=1010, step1=1010)])
    c = WB.act_control(f, 5, 15, roll=False)
    assert c["support_loss"] and not c["support_kept"] and c["control_kept"]
    assert WB.act_control(f, 20, 30, roll=False)["support_kept"]
    f = _feats(n=40, floor=np.r_[np.ones(25), np.zeros(15)], trans=[dict(kind="head_jolt", step0=1021, step1=1021)])
    c = WB.act_control(f, 20, 30, roll=False)
    assert not c["support_kept"] and c["head_jolt"] and not c["control_kept"]
    c = WB.act_control(_feats(n=40, posture=["back"] * 10 + ["side"] * 30), 5, 15, roll=False)
    assert not c["control_kept"] and c["posture_end"] == "side"
    c = WB.act_control(_feats(n=40, posture=["back"] * 10 + ["side"] * 20 + ["back"] * 10), 5, 12, roll=True)
    assert c["control_kept"] and c["reached_posture_held_s"] == pytest.approx(17 * 0.05)
    assert not c["reached_posture_held_to_record_end"]


def test_support_after_a_roll_reuses_the_full_roll_definition():
    """Review fix 11: ROLLING_LEARNING_EXPERIMENT_2026-09-23.json outcome.full_roll asks for positive floor support at
    every 10 ms observation for 0.5 s: 50 fast-lane frames after the roll, read with the 50 ms floor contacts."""
    f = _feats(n=40)
    s = WB.support_after_roll(f, 10)
    assert s["supported"] is True and s["frames_read"] == 50 and s["frames_needed"] == 50
    out = np.full((40, 8), 30.0)
    out[15, 3] = 0.0                                             # one 10 ms frame with no outside force
    assert WB.support_after_roll(_feats(n=40, outside=out), 10)["supported"] is False
    floor = np.ones(40)
    floor[12] = 0.0                                              # a 50 ms record with no floor contact
    assert WB.support_after_roll(_feats(n=40, floor=floor), 10)["supported"] is False
    assert WB.support_after_roll(f, 35)["supported"] is None     # the record ends before 0.5 s: not measured
    c = WB.act_control(f, 5, 10, roll=True)
    assert c["support_after_roll"]["supported"] is True
    assert "REUSED" in WB.definitions()["supported_roll"]
    assert not any("Supported roll as one act" in m for m in WB.MISSING)


def test_support_after_roll_never_substitutes_later_frames_for_missing_ones():
    outside = np.full((40, 8), 30.0)
    outside[15, 3] = np.nan
    out = WB.support_after_roll(_feats(n=40, outside=outside), 10)
    assert out["supported"] is None and out["missing_record"] == 1015
    # Unused NaN padding is not missing observation; only the declared frames count.
    outside[:, 5:] = np.nan
    outside[15, 3] = 30.0
    assert WB.support_after_roll(_feats(n=40, outside=outside), 10)["supported"] is True


# ---------------------------------------------------------------------- the decision rule
def _reading(rows, window=(0, 999), verdict="pass"):
    """rows: {row: {body: (occasions, opportunity_steps)}} -> a reading as read_run writes it."""
    out = {}
    for r, per in rows.items():
        out[r] = dict(bodies={b: dict(occasions=o, opportunity_steps=s, occasions_per_1000=WB.rate(o, s))
                              for b, (o, s) in per.items()},
                      examples=[dict(body=0, step0=1)])
    return dict(rows=out, window=list(window), run="x", validation=dict(verdict=verdict),
                detector_version="v", analysis_detector_version="v", code_detector_version="v")


def _eight(occ, opp=1000):
    return {b: (occ[b] if isinstance(occ, (list, tuple)) else occ, opp) for b in range(8)}


def test_decision_labels_against_both_baseline_readings():
    assert WB.label(5.0, 3.0, 4.0) == "higher than both baseline readings"
    assert WB.label(4.0, 3.0, 4.0) == "within the recorded range"
    assert WB.label(2.0, 3.0, 4.0) == "lower than both baseline readings"
    assert WB.label(None, 3.0, 4.0) is None and WB.label(5.0, None, 4.0) is None


def test_the_six_of_eight_rule_and_its_chance_arithmetic():
    """The coordinator's change of Oct 11: a row is selectable only with 6 or more comparable bodies, 6 or more of them
    higher than both baseline readings. Under the earlier rule one body of eight sufficed (selected by chance 96 in
    100)."""
    chance = sum(math.comb(8, k) * (1 / 3) ** k * (2 / 3) ** (8 - k) for k in (6, 7, 8))
    assert chance == pytest.approx(129 / 6561) and 0.019 < chance < 0.020
    assert any("129/6561" in r for r in WB.DECISION_RULE)
    main = _reading({"R01": _eight(4)})
    sib = _reading({"R01": _eight(5)})
    five = _reading({"R01": _eight([9, 9, 9, 9, 9, 1, 1, 1])})
    res = WB.decide(main, sib, five, rows=["R01"])
    assert res["selected"] is None and res["rows"]["R01"]["status"] == "successes without a clear change"
    six = _reading({"R01": _eight([9, 9, 9, 9, 9, 9, 1, 1])})
    res = WB.decide(main, sib, six, rows=["R01"])
    assert res["selected"] == "R01" and res["rows"]["R01"]["higher"] == 6


def test_decide_breaks_ties_by_the_catalogues_row_sequence():
    """Review fix 9a: the last tie-break follows the catalogue's row sequence (B01 before R01), not ROW_IDS."""
    main = _reading({"R01": _eight(4), "B01": _eight(4)})
    end = _reading({"R01": _eight(9), "B01": _eight(9)})
    assert WB.decide(main, main, end, rows=["R01", "B01"])["selected"] == "B01"


def test_every_row_carries_a_status_and_each_failure_is_named():
    """Review fix 9b-d: a row with a body higher than both but no repeated success, and a row whose endpoint has
    opportunity where no body is comparable, each carry their own named status; 'candidate' no longer names two
    things."""
    main = _reading({"P02": _eight(0, 0), "B01": _eight(0), "R01": _eight(5), "P03": {b: (0, 0) for b in range(8)},
                     "O07": {b: (0, 1000 if b < 3 else 0) for b in range(8)}, "H04": _eight(1)})
    end = _reading({"P02": _eight(0, 0), "B01": _eight(0), "R01": _eight(5),
                    "P03": {b: (1, 1000) for b in range(8)},
                    "O07": {b: (4, 1000 if b < 3 else 0) for b in range(8)}, "H04": _eight(2)})
    res = WB.decide(main, main, end, rows=["P02", "B01", "R01", "P03", "O07", "H04"])
    assert res["selected"] is None
    assert res["failures"] == {"P02": "no opportunity", "B01": "no success", "R01": "successes without a clear change",
                               "P03": "no comparable body", "O07": "too few bodies with opportunity",
                               "H04": "no repeated success"}
    assert set(res["failures"].values()) <= set(WB.STATUS_LABELS)
    assert all("candidate" not in v for v in WB.ROW.values())


def _doc(reading, beside=None, sha="s", verdict="pass"):
    return dict(schema=WB.SCHEMA, verdict=verdict, code=dict(sha256=sha), reading=reading, beside=beside)


def _full(window, rate_by_row=None, verdict="pass"):
    rows = {r: _eight(4) for r in WB.ROW_IDS}
    rows.update(rate_by_row or {})
    return _reading(rows, window=window, verdict=verdict)


def test_decide_refuses_any_input_that_does_not_match(tmp_path):
    """Review blocker 2: decide read neither verdicts, windows, versions nor code; an invalid endpoint or one read over
    the wrong window still produced a selection, and a file with no rows read 'no opportunity' on every row."""
    base = _doc(_full(WB.BASELINE_WINDOW), _full(WB.BASELINE_WINDOW))
    good_end = _doc(_full(WB.ENDPOINT_WINDOW, {"R01": _eight(9)}))
    assert WB.decide_files(base, good_end, code_sha="s")["selected"] == "R01"
    cases = {
        "file's verdict": _doc(_full(WB.ENDPOINT_WINDOW), verdict="invalid"),
        "reading's verdict": _doc(_full(WB.ENDPOINT_WINDOW, verdict="invalid")),
        "window": _doc(_full(WB.BASELINE_WINDOW)),
        "code": _doc(_full(WB.ENDPOINT_WINDOW), sha="other"),
        "has 0 bodies": _doc(dict(_full(WB.ENDPOINT_WINDOW), rows={})),
    }
    for why, end in cases.items():
        with pytest.raises(WB.Refused, match=why):
            WB.decide_files(base, end, code_sha="s")
    odd = _full(WB.ENDPOINT_WINDOW)
    odd["detector_version"] = "w"
    with pytest.raises(WB.Refused, match="detector versions differ"):
        WB.decide_files(base, _doc(odd), code_sha="s")
    with pytest.raises(WB.Refused, match="sibling is missing"):
        WB.decide_files(_doc(_full(WB.BASELINE_WINDOW)), good_end, code_sha="s")
    # a saved file (JSON: body keys are text) decides the same
    p, q = tmp_path / "base.json", tmp_path / "end.json"
    p.write_text(json.dumps(WB._clean(base)), encoding="utf-8")
    q.write_text(json.dumps(WB._clean(good_end)), encoding="utf-8")
    res = WB.decide_files(json.loads(p.read_text()), json.loads(q.read_text()), code_sha="s")
    assert res["selected"] == "R01" and res["rows"]["R01"]["higher"] == 8


def test_decide_on_the_command_line_refuses_with_exit_4(tmp_path):
    sha = WB.code_sha256()
    base = _doc(_full(WB.BASELINE_WINDOW), _full(WB.BASELINE_WINDOW), sha=sha)
    p, q, o = tmp_path / "b.json", tmp_path / "e.json", tmp_path / "d.json"
    p.write_text(json.dumps(WB._clean(base)), encoding="utf-8")
    q.write_text(json.dumps(WB._clean(_doc(_full(WB.ENDPOINT_WINDOW), sha=sha, verdict="invalid"))), encoding="utf-8")
    assert WB.main(["decide", "--baseline", str(p), "--endpoint", str(q), "--out", str(o)]) == WB.EXIT_REFUSED
    q.write_text(json.dumps(WB._clean(_doc(_full(WB.ENDPOINT_WINDOW), sha=sha))), encoding="utf-8")
    assert WB.main(["decide", "--baseline", str(p), "--endpoint", str(q), "--out", str(o)]) == 0
    assert json.loads(o.read_text())["verdict"] == "pass"


def test_the_code_identity_ignores_line_endings(tmp_path):
    """Found while repairing (Oct 11): decide compares the code's sha256, and the main checkout writes this file with
    CRLF line endings (core.autocrlf) where the worktree has LF: the same commit would read as different code."""
    text = MODULE.read_bytes().replace(b"\r\n", b"\n")
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(text)
    crlf.write_bytes(text.replace(b"\n", b"\r\n"))
    assert WB.code_sha256(str(lf)) == WB.code_sha256(str(crlf)) == WB.code_sha256()


# ---------------------------------------------------------------------- fake run folders
def _fake_run(tmp_path, name="run_x", versions=None, final=None, bodies=(0,), processes=((1, 900),),
              quarantined=0, analysis_version=None):
    """A run folder with world chunk names (stream reads names only), a progress record and saved acts. processes:
    (pid, first model step) in start order; final: {pid: last finalized step}."""
    av, dv = _versions()
    versions = versions or (dv,)
    run = tmp_path / name
    for b in bodies:
        w = run / "observer" / "world" / ("b%d" % b)
        w.mkdir(parents=True)
        for k, (pid, first) in enumerate(processes):
            (w / ("header_%d_%d.json" % (pid, 1000 + k))).write_text("{}", encoding="utf-8")
            for c in range(3):
                (w / ("chunk_%09d_%d_%05d.npz" % (first + 100 * c, pid, c))).write_bytes(b"")
    an = run / "observer" / "analysis"
    an.mkdir(parents=True)
    final = final or {pid: 2000 for pid, _ in processes}
    state = dict(version=av, detector_version=analysis_version or dv, quarantined_windows=quarantined,
                 bodies={str(b): dict(final={str(p): v for p, v in final.items()}) for b in bodies})
    (an / "progress.json").write_text(json.dumps(dict(state=state, sizes={})), encoding="utf-8")
    (an / "state.json").write_text(json.dumps(state), encoding="utf-8")
    dets = [dict(cap="R01", sub="L", key="L:box", step0=1010 + 30 * i, step1=1015 + 30 * i, body=0, pid=1, run=name,
                 parts=["hand_L"], provenance="planner-selected", live=True, detector_version=v,
                 measures=dict(closed_cm=6.0))
            for i, v in enumerate(versions)]
    with gzip.open(an / "detections.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in dets:
            f.write(json.dumps(d) + "\n")
    return run


def _write_exposure(run, body, bins_by_pid):
    (run / "observer" / "analysis" / "exposure.json").write_text(
        json.dumps({str(body): {str(p): b for p, b in bins_by_pid.items()}}), encoding="utf-8")


def _fake_track(n=100, step0=1000, pid=1, live=None):
    return SimpleNamespace(n=n, step=np.arange(step0, step0 + n), pid=pid, author=np.full(n, 1, dtype=np.int8),
                           live=np.ones(n, dtype=bool) if live is None else np.asarray(live, dtype=bool))


def _bin(steps, lo, hi, cls="planner-selected"):
    return dict(steps=steps, by={cls: steps}, lo=lo, hi=hi)


def test_read_run_refuses_broken_inputs(tmp_path):
    with pytest.raises(WB.Refused, match="after its last"):
        WB.read_run(str(tmp_path), 200, 100)
    with pytest.raises(WB.Refused, match="no run folder"):
        WB.read_run(str(tmp_path / "missing"), 100, 200)
    (tmp_path / "bare").mkdir()
    with pytest.raises(WB.Refused, match="no observer world records"):
        WB.read_run(str(tmp_path / "bare"), 100, 200)
    run = _fake_run(tmp_path, final={1: 1500})
    with pytest.raises(WB.Refused, match="not finalized"):
        WB.read_run(str(run), 1000, 1999)
    _, dv = _versions()
    run = _fake_run(tmp_path, name="mixed", versions=("observer_detectors_v9", dv))
    with pytest.raises(WB.Refused, match="mixes detector versions"):
        WB.read_run(str(run), 1000, 1099)
    run = _fake_run(tmp_path, name="empty")
    with pytest.raises(WB.Refused, match="no live recorded model step"):
        WB.read_run(str(run), 1000, 1099, tracks_loader=lambda *a: [])
    with pytest.raises(WB.Refused, match="no world records"):
        WB.read_run(str(run), 1000, 1099, bodies=[3])


def _read(run, lo=1000, hi=1099, tracks=None, reach=None):
    tracks = tracks or [_fake_track()]

    def feats(tr):
        r = np.ones((tr.n, 2), dtype=bool) if reach is None else reach[:tr.n]
        return _feats(n=tr.n, step0=int(tr.step[0]), pid=tr.pid, live=tr.live, kind=dict(reach_seen=r), lo=lo, hi=hi)
    return WB.read_run(str(run), lo, hi, tracks_loader=lambda *a: tracks, features=feats)


def test_read_run_end_to_end_on_synthetic_tracks(tmp_path):
    _, dv = _versions()
    run = _fake_run(tmp_path, versions=(dv,) * 3)
    _write_exposure(run, 0, {1: {"10": _bin(100, 1000, 1099)}})
    reach = np.zeros((100, 2), dtype=bool)
    reach[:60, 0] = True
    reading = _read(run, reach=reach)
    r01 = reading["rows"]["R01"]["bodies"][0]
    assert r01["opportunity_steps"] == 60 and r01["successes"] == 2 and r01["acts_outside_opportunity"] == 1
    assert reading["validation"]["verdict"] == "pass", reading["validation"]["checks"]
    assert reading["validation"]["opportunity_consistency"][0]["r01_starting_outside_reach"] == 1
    tmpl = WB.decision_template(reading, reading)
    json.dumps(WB._clean(dict(reading=reading, decision_template=tmpl)))
    assert tmpl["baseline_readings"]["R01"][0]["spread"] == 0.0
    _write_exposure(run, 0, {1: {"10": _bin(100, 1000, 1099, cls="spontaneous")}})
    assert _read(run)["validation"]["verdict"] == "invalid"


def test_a_window_with_no_whole_exposure_bin_is_not_checked_never_a_pass(tmp_path):
    run = _fake_run(tmp_path)
    _write_exposure(run, 0, {1: {"10": _bin(100, 1000, 1099)}})
    res = WB.exposure_check(str(run), {0: {1: {10: {"planner-selected": 50}}}}, {0: {}}, 1050, 1099)
    assert res["checked"] is False and res["all_same"] is False
    _write_exposure(run, 0, {1: {"10": _bin(75, 1000, 1074)}})
    assert WB.exposure_check(str(run), {0: {1: {10: {"planner-selected": 75}}}}, {0: {}}, 1000, 1074)["all_same"]
    assert not WB.exposure_check(str(run), {0: {1: {10: {"planner-selected": 74}}}}, {0: {}}, 1000, 1074)["all_same"]


def test_the_saved_exposure_of_an_abandoned_future_is_cut_as_the_daemon_cuts_it(tmp_path):
    """Review fix 4: a crash resume inside the window. Process 1 recorded 1000-1199; process 2 resumed from 1150. The
    saved exposure counts process 1's 1150-1199 too (an abandoned future); the reader counts live records only. Each
    process's cut is applied through daemon._before_cut: process 1's bin 11 straddles the cut and is compared by its
    share before it (50 of 100), and nothing reads invalid."""
    run = _fake_run(tmp_path, processes=((1, 900), (2, 1150)), final={1: 1199, 2: 1299})
    _write_exposure(run, 0, {1: {"10": _bin(100, 1000, 1099), "11": _bin(100, 1100, 1199)},
                             2: {"11": _bin(50, 1150, 1199), "12": _bin(100, 1200, 1299)}})
    mine = {0: {1: {10: {"planner-selected": 100}, 11: {"planner-selected": 50}},
                2: {11: {"planner-selected": 50}, 12: {"planner-selected": 100}}}}
    res = WB.exposure_check(str(run), mine, {0: {1: 1150}}, 1000, 1299)
    assert res["all_same"] and res["bodies"][0]["bins_straddling_a_cut"] == 1
    # without the cut the abandoned 50 model steps read as a difference
    assert not WB.exposure_check(str(run), mine, {0: {}}, 1000, 1299)["all_same"]


def test_coverage_is_judged_for_every_process_live_in_the_window():
    """Review fix 5: the largest final over processes let an earlier process finalized past the window hide the
    resumed process live at the window's end that was not finalized."""
    ranges = {0: [(1, 900, 1149), (2, 1150, None)]}
    prog = dict(state=dict(bodies={"0": dict(final={"1": 1400, "2": 1180})}))
    with pytest.raises(WB.Refused, match="process 2 finalized through 1180, needs 1299"):
        WB.check_coverage(prog, ranges, 1000, 1299)
    prog["state"]["bodies"]["0"]["final"]["2"] = 1299
    assert WB.check_coverage(prog, ranges, 1000, 1299) == {0: {1: 1400, 2: 1299}}
    prog["state"]["bodies"]["0"]["final"]["1"] = 1100                 # process 1's live part ends at 1149
    with pytest.raises(WB.Refused, match="process 1 finalized through 1100, needs 1149"):
        WB.check_coverage(prog, ranges, 1000, 1299)


def test_a_whole_bin_missing_from_the_saved_exposure_reads_invalid(tmp_path):
    """Review fix 6: a quarantined window moves the finalized line past its steps and writes no exposure and no acts;
    comparing only the bins present in the saved file passed it."""
    run = _fake_run(tmp_path)
    _write_exposure(run, 0, {1: {"10": _bin(100, 1000, 1099)}})
    mine = {0: {1: {10: {"planner-selected": 100}, 11: {"planner-selected": 100}}}}
    res = WB.exposure_check(str(run), mine, {0: {}}, 1000, 1199)
    assert not res["all_same"] and res["bodies"][0]["whole_bins_missing_from_saved"] == [dict(pid=1, bin=11)]


def test_a_quarantined_window_or_another_detector_version_reads_invalid(tmp_path):
    """Review fixes 6 and 8."""
    run = _fake_run(tmp_path, quarantined=1)
    _write_exposure(run, 0, {1: {"10": _bin(100, 1000, 1099)}})
    v = _read(run)["validation"]
    assert v["verdict"] == "invalid" and not v["checks"]["no_quarantined_window"]
    run = _fake_run(tmp_path, name="old", analysis_version="observer_detectors_v9")
    _write_exposure(run, 0, {1: {"10": _bin(100, 1000, 1099)}})
    v = _read(run)["validation"]
    assert v["verdict"] == "invalid" and not v["checks"]["saved_detector_version_is_the_codes"]


def test_a_live_saved_act_that_does_not_join_reads_invalid_and_abandoned_acts_count_apart(tmp_path):
    """Review fix 7: abandoned acts (correctly dropped) and live acts that failed to join were one count, and the
    verdict ignored it."""
    _, dv = _versions()
    run = _fake_run(tmp_path, versions=(dv,) * 3, processes=((1, 900), (2, 1050)), final={1: 1049, 2: 2000})
    _write_exposure(run, 0, {1: {"10": dict(steps=50, by={"planner-selected": 50}, lo=1000, hi=1049)},
                             2: {"10": dict(steps=50, by={"planner-selected": 50}, lo=1050, hi=1099)}})
    t1 = _fake_track(n=50, step0=1000, pid=1)
    t2 = _fake_track(n=50, step0=1050, pid=2)
    # the saved acts are pid 1's at 1010 and 1040 (live) and 1070 (past its cut 1050: an abandoned future)
    reading = _read(run, tracks=[t1, t2])
    b = reading["bodies"][0]
    assert b["saved_acts_abandoned"] == 1 and b["saved_live_acts_not_joined"] == 0
    assert reading["validation"]["verdict"] == "pass", reading["validation"]["checks"]
    # without process 1's records its two live acts cannot join: invalid, named
    reading = _read(run, tracks=[t2])
    b = reading["bodies"][0]
    assert b["saved_live_acts_not_joined"] == 2 and reading["validation"]["verdict"] == "invalid"


# ---------------------------------------------------------------------- blocker 1: finalize-copy
def _copyable_run(tmp_path, committed=None):
    av, dv = _versions()
    run = tmp_path / "src" / "run_y"
    (run / "observer" / "world" / "b0").mkdir(parents=True)
    an = run / "observer" / "analysis"
    an.mkdir(parents=True)
    with gzip.open(an / "detections.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(dict(cap="R01", step0=1)) + "\n")
    with gzip.open(an / "episodes.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write("")
    sizes = dict(committed or {"detections.jsonl.gz": (an / "detections.jsonl.gz").stat().st_size,
                               "episodes.jsonl.gz": (an / "episodes.jsonl.gz").stat().st_size})
    state = dict(version=av, detector_version=dv, bodies={})
    (an / "progress.json").write_text(json.dumps(dict(state=state, exposure={}, fast={}, sizes=sizes)),
                                      encoding="utf-8")
    (an / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (run / "args.json").write_text("{}", encoding="utf-8")
    return run


def test_finalize_copy_copies_the_commit_files_first(tmp_path):
    """Review blocker 1: copytree copied detections.jsonl.gz before progress.json (alphabetical), so a copy taken
    between the service's append and its commit held a progress record covering acts its detections file lacked."""
    import shutil
    run = _copyable_run(tmp_path)
    before = sorted(p.name for p in (run / "observer" / "analysis").iterdir())
    order = []

    def copy(a, b):
        order.append(Path(a).name)
        return shutil.copy2(a, b)
    mark = WB.finalize_copy(str(run), str(tmp_path / "dest"), copy=copy)
    assert order[:2] == ["progress.json", "state.json"]
    assert order.index("detections.jsonl.gz") > order.index("progress.json")
    assert mark["verdict"] == "pass" and (tmp_path / "dest" / "run_y" / WB.COPY_MARK).exists()
    assert sorted(p.name for p in (run / "observer" / "analysis").iterdir()) == before   # the source is untouched


def test_finalize_copy_refuses_record_files_shorter_than_their_committed_size(tmp_path):
    run = _copyable_run(tmp_path, committed={"detections.jsonl.gz": 10 ** 6, "episodes.jsonl.gz": 0})
    with pytest.raises(WB.Refused, match="shorter than the progress record commits"):
        WB.finalize_copy(str(run), str(tmp_path / "dest"))
    with pytest.raises(WB.Refused, match="exists already"):
        WB.finalize_copy(str(run), str(tmp_path / "dest"))


# ---------------------------------------------------------------------- blocker 3: the endpoint jobs
def test_the_endpoint_jobs_are_three_complete_registry_rows_in_order():
    from mimo_bridge.tools import test_registry as TRG
    jobs = json.loads(JOBS.read_text(encoding="utf-8"))
    assert jobs["order"] == [r["id"] for r in jobs["rows"]]
    for row in jobs["rows"]:
        assert TRG.row_problems(row, TRG.DEFAULT_REQUIRED) == [], row["id"]
        assert row["result_source"]["fail_exit_codes"] == [] and row["result_source"]["verdict_key"] == "verdict"
    fin, read, dec = (r["command"] for r in jobs["rows"])
    assert "finalize-copy --run {run}" in fin and "--through 489349" in fin and "commit_guard --need-gb 4" in fin
    assert "read --run D:/UEMA_Research/external/D/UEMA_scratch/remembered_hand_learning_2026-10-11/wholebody/finalized_copies/{run_name}" in read
    assert "--lo 488350 --hi 489349" in read and "wholebody_endpoint_489350.json" in read
    assert "decide --baseline" in dec and "wholebody_baseline_seg41j_1526.json" in dec
    assert "--endpoint D:/UEMA_Research/external/D/UEMA_scratch/remembered_hand_learning_2026-10-11/wholebody/wholebody_endpoint_489350.json" in dec
    for r, cmd in zip(jobs["rows"], (fin, read, dec)):
        assert "test_runner run %s --" % r["id"] in cmd
    assert list(WB.ENDPOINT_WINDOW) == [488350, 489349] and list(WB.BASELINE_WINDOW) == [483675, 484674]


# ---------------------------------------------------------------------- fix 13: the texts
def test_the_texts_say_what_the_detectors_and_records_say():
    from mimo_bridge.tools.observer import detectors as D
    words = WB.GROUP_WORDS["hand_cycles"]
    assert "alternating" in words and "%d degrees" % D.CYCLE_MIN_DEG in words and "%d s" % D.CYCLE_MAX_S in words
    doc = WB.__doc__
    assert "4 in" in doc and "10 in" in doc and "bee" in doc and "2 acts of seg41j_1526" not in doc


# ---------------------------------------------------------------------- the real record (opt-in: builds the model)
@pytest.mark.skipif(os.environ.get("RHL_WHOLEBODY_REAL") != "1" or not REAL_RUN.exists(),
                    reason="builds the run's MuJoCo model: set RHL_WHOLEBODY_REAL=1 and run under the memory guard")
def test_the_repaired_runs_saved_record_and_the_independent_recount_are_reproduced():
    reading = WB.read_run(str(REAL_RUN), 484475, 484674, bodies=[4], reproduce=True)
    v = reading["validation"]
    assert v["verdict"] == "pass", v["checks"]
    d = v["detections"][4]
    assert d["saved"] == d["rerun"] == d["same"] > 0
    # the independent recount of Oct 11 (review note 15), model steps 483,675 to 484,674
    reading = WB.read_run(str(REAL_RUN), 483675, 484674, bodies=[4, 7])
    r01_4 = reading["rows"]["R01"]["bodies"][4]
    assert r01_4["opportunity_steps"] == 1941 and r01_4["acts_counted"] == 66 and r01_4["occasions"] == 44
    assert reading["rows"]["R01"]["bodies"][7]["opportunity_steps"] == 1251
    assert reading["rows"]["R02"]["bodies"][7]["opportunity_steps"] == 1006
    assert reading["rows"]["R02_recontact"]["bodies"][4]["occasions"] == 3
    assert reading["bodies"][4]["model_steps_by_writer"] == {"planner": 500, "reflex_noise": 340, "reflex": 160}
