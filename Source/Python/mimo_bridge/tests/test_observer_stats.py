# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The observer's statistics corrections of 2026-09-24 (outside review; OBSERVER_DESIGN.md section 10,
priorities 2, 5 and 6; docs/gain_of_function_v4/agents/observer_stats.md).

Pinned on synthetic inputs where the right answer is known, each failing on the code before the
change (observer_analysis_v3, tracker_rules_v2):
  * an act is credited to the order whose window holds its PEAK motion, not the order at its first
    record, and its intention, efference and planner-versus-babble readings follow that order;
  * the learner's lane is one record behind the world's: the contact verdict for a touch at world
    record s reads lane step s + 1, the efference window of a record written at lane step t ends at
    world record t - 1;
  * the touch verdict uses the touched thing's own bit, only patches untouched at the record before,
    and only where the world shows a touch starting; it is scored at matched no-touch moments, one
    a second, each moment counted once;
  * the occasion is the tested unit: predicted before action (the first judged act of each occasion;
    touches against the measured no-touch rate), selected intentionally and retained; the single-act
    count of "more often than babble" carries the bursts' dispersion.
"""
import numpy as np
import pytest

from mimo_bridge.tools.observer import daemon as DM
from mimo_bridge.tools.observer import evidence as EV
from mimo_bridge.tools.observer import store as ST
from mimo_bridge.tools.observer import tracker as TR


# ------------------------------------------------------------------ synthetic world and lane
class _Track:
    """The fields of features.Track the evidence join reads, for one body."""

    def __init__(self, n, nbody=4, first_step=1000):
        self.n = n
        self.step = np.arange(first_step, first_step + n)
        self.sim_time = 0.05 * np.arange(n)
        self.order_seq = np.zeros(n, dtype=np.int64)
        self.author = np.full(n, 1, dtype=np.int64)             # planner
        self.xpos = np.zeros((n, nbody, 3))
        self.eye_fwd = np.tile(np.array([1.0, 0.0, 0.0]), (n, 2, 1))
        self.continuous = np.ones(n, dtype=bool)
        self.continuous[0] = False
        self.live = np.ones(n, dtype=bool)
        self.c_row = np.zeros(0, dtype=np.int64)
        self.c_e = np.zeros((0, 2), dtype=np.int64)

    def touch(self, rows, e1, e2):
        """Entity e1 touches e2 at these track records."""
        rows = list(rows)
        self.c_row = np.r_[self.c_row, rows].astype(np.int64)
        self.c_e = np.r_[self.c_e, np.tile([e1, e2], (len(rows), 1))].astype(np.int64).reshape(-1, 2)
        o = np.argsort(self.c_row, kind="stable")
        self.c_row, self.c_e = self.c_row[o], self.c_e[o]


class _Lane:
    def __init__(self, handles=("index_R",)):
        self.senses = ["vision", "tactile", "audio", "proprioception", "vestibular"]
        self.handles = list(handles)
        self.rows, self.contact, self.eff, self.licenses = {}, {}, {}, []


# entities: 0 the right index fingertip, 1 the right palm, 2 the left palm (all skin patches),
# 3 a box (an outside thing); patch i covers body i
HEADER = dict(patch_bodies=[0, 1, 2],
              entities=[dict(code=0, kind="self", group="finger_R_index", bodies=[0]),
                        dict(code=1, kind="self", group="hand_R_palm", bodies=[1]),
                        dict(code=2, kind="self", group="hand_L_palm", bodies=[2]),
                        dict(code=3, kind="object", group="box", bodies=[3])],
              handle_names=["index_R"], handle_bodies=[0])


def _latch(step, seq, outcome, body=0, bodies=1):
    out = ["tied"] * bodies
    out[body] = outcome
    return dict(type="latch", step=step, order_seq=seq, reached_body=True,
                plan=dict(pred_wellness_gaps=dict(outcome=out, gap_to_best_alt=[0.2] * bodies, tie_band=0.0734)))


# ------------------------------------------------------------------ 1 credit at the peak motion
def _two_orders():
    """Order 7 (the planner, tied) in force at records 0-9, order 8 (the planner, decisive) from
    record 10; the fingertip moves 0.2 cm a record under order 7 and 3 cm at record 12."""
    tr = _Track(30)
    tr.order_seq[:10], tr.order_seq[10:] = 7, 8
    for i in range(1, 30):
        tr.xpos[i, 0] = tr.xpos[i - 1, 0] + np.array([0.002, 0.0, 0.0])
    tr.xpos[12:, 0] += np.array([0.03, 0.0, 0.0])
    chain = [_latch(1000, 7, "tied"), _latch(1010, 8, "best")]
    d = dict(cap="H05", sub="R:index", step0=1008, step1=1014, i0=8, i1=14, parts=["finger_R_index"],
             author="planner", order_seq=7)
    return tr, chain, d


def test_an_act_is_credited_to_the_order_in_force_at_its_peak_motion():
    """An act of another part (here the right forearm, on the same moving body) is credited to the order at its peak;
    the same movement as an act of the hands alone is credited to no single order (observer_attribution_v1)."""
    tr, chain, d = _two_orders()
    arm_header = dict(HEADER, handle_names=["index_R", "lowerarm_R"], handle_bodies=[0, 0])
    arm = dict(d, cap="B06", sub="R", parts=["arm_R"])
    j = EV.Joiner(_Lane(), chain, ["index_R"], arm_header)
    cr = j.credit(arm, tr)
    assert cr["peak_step"] == 1012 and cr["order_seq"] == 8 and cr["first_order_seq"] == 7
    assert cr["moved"] is True and cr["motion_unit"] == "cm" and cr["peak_motion"] == pytest.approx(3.2, rel=1e-6)
    j.attach([arm], 0, tr)
    ev = arm["evidence"]
    # the credited order is the planner's decisive choice; the order at the first record tied
    assert ev["credit"]["order_seq"] == 8
    assert ev["intention"]["verdict"] is True and ev["intention"]["order_seq"] == 8
    assert j.intention(arm, 0)["verdict"] is False, "credited to the first record's order it would read 'tied'"
    # the finger act of the same movement: orders 7 and 8 were both in force during it, and order 7 tied
    j.attach([d], 0, tr)
    ev = d["evidence"]
    assert ev["credit"]["rule"] == "command_intervals" and ev["credit"]["commands"] == [7, 8]
    assert "order_seq" not in ev["credit"] and ev["intention"]["verdict"] is False
    assert [i["command"] for i in ev["attribution"]["intervals"]] == [7, 8]


def test_an_act_of_the_eyes_is_timed_by_the_turn_of_the_gaze():
    tr = _Track(20)
    tr.order_seq[:6], tr.order_seq[6:] = 3, 4
    ang = np.radians([0.0] * 5 + [1.0, 2.0, 12.0, 13.0] + [13.0] * 11)      # a 10 degree jump at record 7
    tr.eye_fwd[:, 0, 0], tr.eye_fwd[:, 0, 1] = np.cos(ang), np.sin(ang)
    j = EV.Joiner(_Lane(), [], ["index_R"], HEADER)
    cr = j.credit(dict(cap="G01", parts=["eye_L"], i0=4, i1=10), tr)
    assert cr["peak_step"] == 1007 and cr["order_seq"] == 4 and cr["motion_unit"] == "deg"
    assert cr["peak_motion"] == pytest.approx(10.0, abs=1e-6)


def test_the_planner_rate_counts_occasions_by_the_author_of_the_credited_order():
    """Acts that start under babble but whose peak motion comes under a decisive planner order are
    the planner's: counted by the first record's author they were babble's."""
    def act(step, prov, cprov, intent):
        return dict(cap="X", sub="", key="k", body=0, step0=step, step1=step + 2, episode="e%d" % step,
                    provenance=prov, live=True, scripted=False, caregiver_touch=False, caregiver_present=False,
                    posture="back", objects=[], measures={"m": 1.0},
                    evidence=dict(prediction=None, intention=dict(verdict=intent),
                                  credit=dict(provenance=cprov, moved=prov != cprov, measured=True)))
    dets = [act(i * 100, "spontaneous", "planner-selected", True) for i in range(10)]
    dets += [act(1000 + i * 100, "spontaneous", "spontaneous", False) for i in range(2)]
    ex = dict(steps=2000, by_author={"planner-selected": 1000, "spontaneous": 1000}, windows={}, bodies=8)
    st = TR.evaluate(dets, ex, dict(count=0, occasions=0, steps=100000), None)["stages"]["selected_intentionally"]
    assert st["planner_occasions"] == 10 and st["babble_occasions"] == 2 and st["credited_later"] == 10
    assert st["reached"] is True and st["decisive"] == 10


def test_an_act_credited_to_a_probe_or_rest_order_is_not_counted():
    d = dict(cap="X", sub="", key="k", body=0, step0=5, step1=7, provenance="spontaneous", live=True,
             scripted=False, caregiver_touch=False, measures={},
             evidence=dict(credit=dict(provenance="external-rest", moved=True, measured=True)))
    out = TR.evaluate([d, dict(d, step0=500, step1=502, evidence=None)], dict(steps=1000, by_author={}, windows={}),
                      None, None)
    assert out["counted"] == 1


# ------------------------------------------------------------------ 2 the lane is one record behind
def test_the_production_join_reads_the_lane_one_record_behind_the_world():
    j = DM.make_joiner(_Lane(), [], HEADER)
    assert j.lane_lag == EV.TRAINER_LANE_LAG == 1
    # an efference record written at lane step 25 graded the order that drove world records
    # 20..23 (its next order began at record 24, the record the learner read at step 25)
    lane = _Lane()
    z = np.zeros((1, 1, 3))
    ach = np.array([[[1.0, 0, 0]]])
    lane.eff[20] = dict(began=15, exp_cm=z, ach_cm=0.4 * ach, sh_cm=z, exp_deg=z, ach_deg=z, sh_deg=z)
    lane.eff[25] = dict(began=20, exp_cm=0.9 * ach, ach_cm=ach, sh_cm=z, exp_deg=z, ach_deg=z, sh_deg=z)
    lane.eff[31] = dict(began=24, exp_cm=z, ach_cm=ach, sh_cm=0.95 * ach, exp_deg=z, ach_deg=z, sh_deg=z)
    j = DM.make_joiner(lane, [], HEADER)
    d = dict(cap="H05", step0=24, parts=["finger_R_index"])
    m = j.movement_prediction(d, 0, None, dict(peak_step=24))
    assert m["order_began"] == 24 and m["verdict"] is False, "record 24 belongs to the order that began there"
    m23 = j.movement_prediction(d, 0, None, dict(peak_step=23))
    assert m23["order_began"] == 20 and m23["verdict"] is True


def test_the_touch_verdict_reads_the_prediction_for_the_onset_record():
    """The lane's contact record at step t predicts world record t - 1: a touch starting at record
    1010 is judged on lane step 1011, not on 1010 (the prediction for the record before)."""
    tr = _Track(30)
    tr.touch(range(10, 14), 0, 3)                     # the fingertip touches the box at records 10-13
    lane = _Lane()
    base = np.full((3, 3), 0.05)
    for s in range(1000, 1031):
        q = np.full((1, 3, 3), 0.02)
        lane.contact[s] = dict(q=q, base=base, valid=np.ones(1, dtype=bool))
    lane.contact[1011]["q"][0, 0, 2] = 0.40           # the prediction FOR record 1010: 8 times the base rate
    j = DM.make_joiner(lane, [], HEADER)
    d = dict(cap="HX3", sub="R:index", step0=1010, step1=1013, i0=10, i1=13, parts=["finger_R_index"],
             touched_kind="box")
    p = j.contact_prediction(d, 0, tr)
    assert p["verdict"] is True and p["lane_step"] == 1011 and p["bit"] == "world"
    assert p["ratio"] == pytest.approx(8.0)


# ------------------------------------------------------------------ 3 the touch verdict and its null
def _touch_world():
    """The right palm rests on the box through records 0-39; the fingertip starts touching the box
    at record 20. The contact head gives the resting palm 0.9 (it stays touched) and the fingertip
    its base rate."""
    tr = _Track(60)
    tr.touch(range(0, 40), 1, 3)
    tr.touch(range(20, 24), 0, 3)
    lane = _Lane()
    base = np.full((3, 3), 0.05)
    for s in range(1000, 1062):
        q = np.full((1, 3, 3), 0.05)
        q[0, 1, :] = np.where(s - 1 - 1000 < 40, 0.9, 0.05)   # the palm, while it rests (lane t: record t-1)
        lane.contact[s] = dict(q=q, base=base, valid=np.ones(1, dtype=bool))
    return tr, lane


def test_a_patch_already_touched_before_the_onset_cannot_carry_the_touch_verdict():
    tr, lane = _touch_world()
    j = DM.make_joiner(lane, [], HEADER)
    d = dict(cap="H01", sub="R", step0=1020, step1=1025, i0=20, i1=25, parts=["hand_R"], touched_kind="box")
    p = j.contact_prediction(d, 0, tr)
    # the resting palm's 0.9 against a base rate of starting 0.05 (ratio 18) passed the old verdict;
    # only the fingertip could start a touch, and it was predicted at its base rate
    assert p["verdict"] is False and p["ratio"] == pytest.approx(1.0)
    assert (p["patches"], p["judged_patches"], p["starting_patches"]) == (2, 1, 1)


def test_a_touch_of_an_own_part_is_judged_on_the_own_part_bit():
    tr = _Track(30)
    tr.touch(range(10, 13), 0, 2)                     # the right fingertip touches the left palm
    lane = _Lane()
    base = np.full((3, 3), 0.05)
    for s in range(1000, 1031):
        q = np.full((1, 3, 3), 0.02)
        lane.contact[s] = dict(q=q, base=base, valid=np.ones(1, dtype=bool))
    lane.contact[1011]["q"][0, 0, 1] = 0.30
    j = DM.make_joiner(lane, [], HEADER)
    d = dict(cap="HX3", sub="R:index", step0=1010, step1=1012, i0=10, i1=12, parts=["finger_R_index"],
             touched_kind="hand_L_palm")               # primitives.entity_class: the group, never 'self'
    p = j.contact_prediction(d, 0, tr)
    assert p["bit"] == "self" and p["verdict"] is True
    # with nothing starting on these patches at the act's record, there is no touch to judge
    none = j.contact_prediction(dict(d, i0=15, i1=16, step0=1015, step1=1016), 0, tr)
    assert none["verdict"] is None and "no touch" in none["note"]


def test_a_repeat_contact_is_judged_at_its_touch_not_at_the_start_of_its_approach():
    tr = _Track(30)
    tr.touch(range(16, 20), 0, 3)
    lane = _Lane()
    base = np.full((3, 3), 0.05)
    for s in range(1000, 1031):
        lane.contact[s] = dict(q=np.full((1, 3, 3), 0.01), base=base, valid=np.ones(1, dtype=bool))
    lane.contact[1017]["q"][0, 0, 2] = 0.5
    j = DM.make_joiner(lane, [], HEADER)
    d = dict(cap="R02", sub="box", step0=1010, step1=1019, i0=10, i1=19, parts=["hand_R"],
             measures=dict(time_to_contact_s=0.30))
    p = j.contact_prediction(d, 0, tr)
    assert p["onset_step"] == 1016 and p["verdict"] is True


def test_no_touch_moments_lie_one_to_three_seconds_from_the_onset_and_each_is_counted_once():
    """Analysis v5 (independent review of 2026-09-25, finding 1): an act's no-touch moments are the
    records 1, 2 and 3 s before and after its onset at the onset's place in its order, each given to the
    earliest act of the same row and skin patches that has it; whichever window finalizes an act, it
    gets the same moments (until v5: the moments of the steps a window finalized, given to the nearest
    act of that window)."""
    tr = _Track(130)
    tr.order_seq = tr.step // 5                        # 5-step orders; the track's first one is unplaced
    tr.touch(range(0, 40), 1, 3)                       # the palm rests on the box
    tr.touch(range(20, 24), 0, 3)                      # fingertip touches at records 20 and 60
    tr.touch(range(60, 64), 0, 3)
    lane = _Lane()
    base = np.full((3, 3), 0.05)
    for s in range(1000, 1131):
        lane.contact[s] = dict(q=np.full((1, 3, 3), 0.05), base=base, valid=np.ones(1, dtype=bool))
    for rec in (1040, 1043, 1100):                     # the head flags the fingertip here (1043 is off the grid)
        lane.contact[rec + 1]["q"][0, 0, 2] = 0.5
    j = DM.make_joiner(lane, [], HEADER)
    a = dict(cap="HX3", sub="R:index", step0=1020, step1=1023, i0=20, i1=23, parts=["finger_R_index"], touched_kind="box")
    b = dict(cap="HX3", sub="R:index", step0=1060, step1=1063, i0=60, i1=63, parts=["finger_R_index"], touched_kind="box")
    both = [dict(a), dict(b)]
    j.attach(both, 0, tr)
    pa, pb = both[0]["evidence"]["prediction"], both[1]["evidence"]["prediction"]
    # a (onset 1020): 1000 is in the track's first, unplaced order; 1040 (flagged: a pass), 1060 (b's
    # touch starts: no moment) and 1080 (not flagged) are a's. b (onset 1060): 1040 and 1080 are a's
    # already (the earlier act); 1020 (a's touch starts), 1100 (flagged) and 1120 are b's
    assert (pa["null_moments"], pa["null_passed"]) == (2, 1)
    assert (pb["null_moments"], pb["null_passed"]) == (2, 1)
    assert pa["null_reach_steps"] == EV.MATCH_REACH_STEPS == 60 and "_key" not in pa and "_key" not in pb
    # the window that finalizes each act does not matter: the act alone, with every act the window found
    only_a, only_b = dict(a), dict(b)
    j.attach([only_a], 0, tr, all_dets=[only_a, dict(b)])
    j.attach([only_b], 0, tr, all_dets=[dict(a), only_b])
    assert (only_a["evidence"]["prediction"]["null_moments"], only_a["evidence"]["prediction"]["null_passed"]) == (2, 1)
    assert (only_b["evidence"]["prediction"]["null_moments"], only_b["evidence"]["prediction"]["null_passed"]) == (2, 1)
    # the digester holds back an act until its last moment is recorded: 3 s after the onset
    assert j.matched_horizon(dict(b), tr) == 1120 and j.matched_horizon(dict(cap="H05", parts=["finger_R_index"]), tr) is None
    assert EV.NULL_EVERY_STEPS == TR.OCCASION_GAP_STEPS


# ------------------------------------------------------------------ 4 the occasion is the unit
def _det(step, pred=None, kind="efference", null=(0, 0), body=0, sub=""):
    p = None if pred is None else dict(kind=kind, verdict=pred, null_moments=null[0], null_passed=null[1])
    return dict(cap="X", sub=sub, key="k", body=body, step0=step, step1=step + 2, episode="e%d:%d" % (body, step),
                provenance="spontaneous", live=True, scripted=False, caregiver_touch=False, caregiver_present=False,
                posture="back", objects=[], measures={"m": 1.0}, evidence=dict(prediction=p, intention=None))


RARE = dict(count=0, occasions=0, steps=100000)
EX = dict(steps=10000, by_author={"spontaneous": 10000}, windows={}, bodies=8)


def test_an_occasion_is_judged_by_its_first_judged_act():
    """Three bursts of five acts: each burst's first act unforeseen, the four that follow within the
    second foreseen. 12 of 15 acts is 'predicted' when acts are trials; 0 of 3 occasions is not."""
    dets = [_det(o * 1000 + k * 3, pred=(k > 0)) for o in range(3) for k in range(5)]
    st = TR.evaluate(dets, EX, RARE, None)["stages"]["predicted_before_action"]
    assert (st["predicted"], st["judged"]) == (0, 3) and (st["acts_predicted"], st["acts_judged"]) == (12, 15)
    assert st["reached"] is False
    many = [_det(o * 1000 + k * 3, pred=(k == 0)) for o in range(8) for k in range(3)]
    st2 = TR.evaluate(many, EX, RARE, None)["stages"]["predicted_before_action"]
    assert (st2["predicted"], st2["judged"]) == (8, 8) and st2["reached"] is True
    assert st2["p_value"] == pytest.approx(0.5 ** 8)


def test_touch_occasions_are_tested_against_the_measured_no_touch_rate():
    # 7 of 10 touch occasions foreseen: against one half (p 0.17) that is no evidence; the verdict
    # passes at 10 of 100 no-touch moments, so 7 of 10 is far above its chance rate
    dets = [_det(i * 100, pred=(i < 7), kind="contact_head", null=(10, 1)) for i in range(10)]
    st = TR.evaluate(dets, EX, RARE, None)["stages"]["predicted_before_action"]
    assert st["null_moments"] == 100 and st["null_passed"] == 10 and st["null_rate"] == pytest.approx(0.1)
    assert st["p_value"] == pytest.approx(TR.two_share_above(7, 10, 10, 100)) and st["reached"] is True
    # the same seven when the verdict passes at 60 of 100 no-touch moments: not above chance
    loud = [_det(i * 100, pred=(i < 7), kind="contact_head", null=(10, 6)) for i in range(10)]
    assert TR.evaluate(loud, EX, RARE, None)["stages"]["predicted_before_action"]["reached"] is False
    # without no-touch moments the touch test is unmeasured, never judged against one half
    blind = [_det(i * 100, pred=True, kind="contact_head") for i in range(10)]
    assert TR.evaluate(blind, EX, RARE, None)["stages"]["predicted_before_action"]["reached"] is None


def test_retention_counts_occasions_so_shorter_bursts_are_not_a_loss():
    """The same number of occasions in every block, but the later blocks' bursts are one act
    instead of four: counted in acts the later blocks fell below half and read as regression."""
    W = TR.RET_WINDOW
    windows = {str(w): 8 * W for w in range(4)}
    dets = []
    for w in range(4):
        for i in range(20):
            s = w * W + i * 100
            dets += [_det(s + 3 * k, body=i % 4) for k in range(4 if w < 1 else 1)]
    base = dict(count=1, occasions=1, steps=10 ** 7)
    r = TR.evaluate(dets, dict(steps=32 * W, by_author={"spontaneous": 32 * W}, windows=windows, bodies=8), base, None)
    assert r["stages"]["repeated"]["reached"] is True
    assert r["stages"]["retained"]["reached"] is True and not r["regressed"]
    assert r["stages"]["retained"]["unit"] == "occasions"


def test_the_single_act_count_carries_the_bursts_dispersion():
    """600 occasions against babble's 500 in the same steps, but the extra acts come in bursts of
    six: as independent acts 1,100 against babble's 1,000 passed (p 0.015); with the bursts'
    spread it is no evidence of more acts."""
    dets = [_det(i * 25) for i in range(500)]
    dets += [_det(20000 + i * 40 + 3 * k) for i in range(100) for k in range(6)]
    base = dict(count=1000, occasions=500, steps=10000)
    r = TR.evaluate(dets, dict(steps=10000, by_author={"spontaneous": 10000}, windows={}, bodies=8), base, None)
    assert r["stages"]["repeated"]["occasions"] == 600 and r["stages"]["repeated"]["acts"] == 1100
    assert r["dispersion"] == pytest.approx(4100 / 1100) and r["babble_dispersion"] == pytest.approx(2.0)
    assert max(TR.two_sample_above(600, 10000, 500, 10000), TR.two_sample_above(1100, 10000, 1000, 10000)) < 0.05
    assert r["stages"]["repeated"]["reached"] is False
    # with one act to every occasion the dispersed test is the plain one
    assert TR.two_sample_above_dispersed(12, 10000, 1.0, 10, 100000, 1.0) == pytest.approx(
        TR.two_sample_above(12, 10000, 10, 100000))


def test_occasion_ids_group_acts_exactly_as_the_occasions_are_counted():
    rng = np.random.default_rng(3)
    steps = np.sort(rng.integers(0, 3000, 400))
    bodies = rng.integers(0, 3, 400)
    parts = rng.integers(0, 2, 400)
    ids = TR.occasion_ids(steps, bodies, parts)
    starts = TR.occasion_starts(steps, bodies, parts)
    assert np.unique(ids).size == starts.size == TR.occasions(steps, bodies, parts)
    first = TR.first_in_occasion(steps, ids)
    assert np.array_equal(np.sort(steps[first]), starts)
    assert np.array_equal(ids[first], np.arange(starts.size)), "numbered in the order the occasions start"


# ------------------------------------------------------------------ the store and the versions
def test_the_store_keeps_the_credit_and_the_no_touch_moments():
    d = _det(10, pred=True, kind="contact_head", null=(12, 3))
    d["evidence"]["credit"] = dict(provenance="planner-selected", moved=True, measured=True)
    t = ST.DetTable.from_dicts([d, _det(50)])
    assert t.s.get(int(t["cprov"][0])) == "planner-selected" and t.s.get(int(t["cprov"][1])) == "spontaneous"
    assert t["credit_moved"].tolist() == [1, -1]
    assert (int(t["null_n"][0]), int(t["null_k"][0])) == (12, 3)
    # v4 (2026-09-25): each unit's token counts for the novelty miner's order control; v5 (2026-10-08): the columns
    # attr, alone, sel and ach of observer_attribution_v1 (test_observer_attribution.py)
    assert ST.STORE_VERSION == "observer_store_v5"


def test_the_changed_definitions_carry_new_versions():
    # v5: the observer statistics track of 2026-09-25 (test_observer_statistics_fixes.py)
    assert DM.ANALYSIS_VERSION == "observer_analysis_v5"
    # v3: the occasion as the tested unit; v4: the size half of Seen counts occasions too; v5: the
    # observer statistics track of 2026-09-25 (test_observer_statistics_fixes.py); v6: observer_attribution_v1
    assert TR.RULES_VERSION == "tracker_rules_v6"
