# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""observer_attribution_v1 (2026-10-08; the operator's instruction of 7:37 AM PDT, Oct 8, 2026 and its addition of the
same morning): an act of the hands alone is credited to no single order; and the independent review of 4342f9194
(D:/UEMA_Research/external/D/UEMA_scratch/plan_review/OBSERVER_ATTRIBUTION_REVIEW.md, 9:35 AM PDT, Oct 8, 2026).

Pinned on synthetic inputs whose right answer is known, each failing on the code before the change (analysis v5 credited
a hand act to the order one record before its largest world-frame motion; 4342f9194 judged an act's movement by every
order passing against a chance line of one half, said "not measurable" with a false cause for acts recorded before the
change, judged "planner selected" with a band of zero, and read "achieved" on another window than "predicted"):
  * every order in force during the act is listed with the movement made DURING its records, its share of the act,
    whether its own motors pushed the way the joints went, and whether its hand joint goal asked for that direction;
    the orders of the records before the act are listed as its history; each order's WHOLE HOLD is read too;
  * the act's writer class is 'mixed' when its orders came from more than one writer; its credit names no order;
  * its intention verdict needs EVERY order to be a decisive planner choice that did not ask the other way;
  * each order is one unit of the Predicted stage, the act's orders one cluster, tested against one half at the
    design effect's upper bound; under no prediction skill its false-pass rate stays at the line (simulation);
  * four separate stage readings: planner selected (no when an order was not the planner's, NOT JUDGED otherwise: no
    noise band of the goal distance is measured), predicted and achieved on one window per order (its whole hold),
    improved (not readable from one act); none read as intention;
  * acts recorded before the change are NOT JUDGED, with that reason, their count and their withdrawn earlier reading;
    the page's headline and milestone name a withdrawn pass; the stage history names the stages a rules change
    withdrew;
  * the store marks how each act's writer is attributed, and the tracker counts an old single-order credit of a hand
    act and a mixed act for neither side;
  * H05 reports the other digits' movement INSIDE the act beside its end-to-end rule;
  * the hand joint goal's order is the controller's (hand_target_control.geometry_from_model).
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import special

from mimo_bridge.tools.observer import evidence as EV
from mimo_bridge.tools.observer import store as ST
from mimo_bridge.tools.observer import tracker as TR
from mimo_bridge.tools.observer import view as V

PLANNER, REFLEX_NOISE = 1, 3


class _Kin:
    """The kinematics fields the join reads: the right index finger's three flexion joints at qpos 0-2, the right
    middle finger's at 3-5; a motor on each index joint (gear x gain 1); the hand goal lists qpos 0-5 in order."""

    def __init__(self):
        self.flex_q = {("R", "index"): [0, 1, 2], ("R", "middle"): [3, 4, 5]}
        self.m = SimpleNamespace(nu=3, actuator_trntype=np.zeros(3, dtype=int),
                                 actuator_trnid=np.array([[0, 0], [1, 0], [2, 0]]), jnt_qposadr=np.array([0, 1, 2]),
                                 actuator_gear=np.ones((3, 6)), actuator_gainprm=np.ones((3, 10)))

    @staticmethod
    def hand_goal_qpos():
        return [0, 1, 2, 3, 4, 5]


class _HandTrack:
    """Order 7 (the planner) at records 0-9, order 8 (the motor controller with exploration noise) at 10-14, order 9
    (the planner) from 15. The index joints flex 2 degrees a record from record 9 to record 17 and stand still
    after it. Orders 7 and 8 ask for flexion (goal 60 degrees) and push it (command +0.5); order 9 asks for
    extension (goal 0 degrees) and pushes it (command -0.3) while the joints keep flexing: the shape of the shared
    episode of the first-finger investigation (seg41c_0841, body 5, 483,793-483,802).
    bracket: records 0-1 run under order 6 and records 20 on under order 10 (both the motor controller's), so the holds
    of orders 7, 8 and 9 are read whole (records 2-9, 10-14, 15-19)."""

    def __init__(self, n=30, bracket=False):
        self.n = n
        self.step = np.arange(1000, 1000 + n)
        self.sim_time = 0.05 * np.arange(n)
        self.order_seq = np.r_[np.full(10, 7), np.full(5, 8), np.full(n - 15, 9)]
        self.author = np.r_[np.full(10, PLANNER), np.full(5, REFLEX_NOISE), np.full(n - 15, PLANNER)]
        self.latched = np.zeros(n, dtype=bool)
        self.latched[[0, 10, 15]] = True
        if bracket:
            self.order_seq[:2], self.author[:2] = 6, REFLEX_NOISE
            self.order_seq[20:], self.author[20:] = 10, REFLEX_NOISE
            self.latched[[2, 20]] = True
        deg = np.zeros(n)
        deg[9:18] = 2.0 * np.arange(1, 10)
        deg[18:] = deg[17]
        self.qpos = np.zeros((n, 6))
        self.qpos[:, 0:3] = np.radians(deg)[:, None]
        self.ctrl = np.zeros((n, 3))
        self.ctrl[:15] = 0.5
        self.ctrl[15:] = -0.3
        goal = np.where(np.arange(n) < 15, np.radians(60.0), 0.0)
        self.order_hand_goal = np.tile(goal[:, None], (1, 6))
        self.xpos = np.zeros((n, 2, 3))
        self.xpos[:, 0, 0] = 0.001 * deg
        self.continuous = np.ones(n, dtype=bool)
        self.continuous[0] = False
        self.live = np.ones(n, dtype=bool)
        self.kin = _Kin()
        self.header = dict(joints=[])
        self.c_row = np.zeros(0, dtype=np.int64)
        self.c_e = np.zeros((0, 2), dtype=np.int64)


HEADER = dict(patch_bodies=[], entities=[], handle_names=["index_R"], handle_bodies=[0])


def _latch(step, seq, chosen, alts, outcome="tied", body=0, bodies=1):
    """A planner latch record as daemon.latch_summary keeps it: the goal distances of the chosen order and of its
    alternatives, the hand stage's costs and the wellness evaluator's outcome."""
    def per(v):
        out = [None] * bodies
        out[body] = v
        return out
    return dict(type="latch", step=step, order_seq=seq, source="planner", reached_body=True,
                plan=dict(pred_wellness_gaps=dict(outcome=per(outcome), gap_to_best_alt=per(0.0), tie_band=0.0734),
                          goal_distance=dict(chosen=per(chosen), **{k: per(v) for k, v in alts.items()}),
                          alternatives=list(alts), pred_change=dict(proprioception=per(0.05)),
                          hand_optimizer=dict(first=per(2.0), selected=per(1.5), valid=per(True))))


def _act():
    return dict(cap="H05", sub="R:index", key="R:index:bend", step0=1008, step1=1017, i0=8, i1=17,
                parts=["finger_R_index"], author="planner", provenance="planner-selected", order_seq=7)


def _chain():
    return [_latch(1000, 7, 0.40, dict(hold=0.50, reflex=0.45, perturb_1=0.48, perturb_2=0.47), outcome="best"),
            dict(type="latch", step=1010, order_seq=8, source="reflex_noise", reached_body=None, plan={}),
            _latch(1015, 9, 0.60, dict(hold=0.70, reflex=0.65, perturb_1=0.55, perturb_2=0.66), outcome="best")]


def _joiner(lane=None):
    return EV.Joiner(lane, _chain(), ["index_R"], HEADER)


def _eff(began, exp, ach, sh):
    return dict(began=began, exp_cm=np.array([[[exp, 0.0, 0.0]]]), ach_cm=np.array([[[ach, 0.0, 0.0]]]),
                sh_cm=np.array([[[sh, 0.0, 0.0]]]))


def _lane():
    """The learner's efference records of the bracketed track (lane numbered as the world, lag 0): each order's record
    is written where the next order begins. Order 7: its own expectation exact (passes); order 8: the other body's
    exact and its own 2 cm off (fails); order 9: its own exact (passes)."""
    eff = {1002: _eff(1000, 0.0, 0.0, 0.0), 1010: _eff(1002, 1.0, 1.0, 0.0), 1015: _eff(1010, 0.0, 2.0, 2.0),
           1020: _eff(1015, 1.0, 1.0, 3.0)}
    return SimpleNamespace(eff=eff, rows={}, contact={}, senses=[])


# ------------------------------------------------------------------ the orders of a finger act
def test_every_order_in_force_is_listed_with_the_movement_made_during_it():
    tr = _HandTrack()
    att = _joiner().command_intervals(_act(), tr)
    assert att["version"] == EV.ATTRIBUTION_VERSION and att["measure"] == "R index flexion"
    assert att["act_change"] == pytest.approx(18.0)
    got = [(e["command"], e["writer"], e["records"], round(e["moved_during"], 6)) for e in att["intervals"]]
    assert got == [(7, "planner", [1009, 1009], 2.0), (8, "reflex_noise", [1010, 1014], 10.0),
                   (9, "planner", [1015, 1017], 6.0)]
    assert [round(e["share_of_act_change"], 4) for e in att["intervals"]] == [0.1111, 0.5556, 0.3333]
    # the order's own motors and its goal, beside the movement, never as its cause
    assert [e["motors_same_way_share"] for e in att["intervals"]] == [1.0, 1.0, 0.0]
    assert [e["asked"] for e in att["intervals"]] == ["same", "same", "opposite"]
    assert [e["latched_step"] for e in att["intervals"]] == [1000, 1010, 1015]
    assert "not said to be caused" in att["note"]
    # the records before the act: order 7 from its latch, where the finger had not moved yet
    assert [(h["command"], h["records"]) for h in att["history"]] == [(7, [1000, 1008])]
    assert att["writer_class"] == "mixed" and att["writer"] == "mixed" and att["n_commands"] == 3


def test_each_orders_whole_hold_is_read_beside_its_records_inside_the_act():
    """Review fix 2: the window the order's efference record grades, its whole hold, read whole only when the records
    on both sides belong to other orders; the change over it from the record before its first record to its last."""
    att = _joiner().command_intervals(_act(), _HandTrack(bracket=True))
    holds = [(e["command"], e["hold_records"], e["hold_whole"], e["hold_change"], e["hold_asked"]) for e in att["intervals"]]
    assert holds == [(7, [1002, 1009], True, pytest.approx(2.0), "same"), (8, [1010, 1014], True, pytest.approx(10.0), "same"),
                     (9, [1015, 1019], True, pytest.approx(6.0), "opposite")]
    # without the bracket, order 7's hold starts at the first record read and order 9's runs to the last: not whole
    att = _joiner().command_intervals(_act(), _HandTrack())
    assert [e["hold_whole"] for e in att["intervals"]] == [False, True, False]
    assert att["intervals"][0]["hold_change"] is None and att["intervals"][0]["hold_asked"] is None


def test_a_finger_act_is_credited_to_no_single_order_and_its_verdicts_need_every_order():
    tr = _HandTrack()
    d = _act()
    j = _joiner()
    j.attach([d], 0, tr)
    ev = d["evidence"]
    assert ev["credit"]["rule"] == "command_intervals" and "order_seq" not in ev["credit"]
    assert ev["credit"]["commands"] == [7, 8, 9] and ev["credit"]["provenance"] == "mixed"
    assert d["provenance"] == "mixed" and d["author"] == "mixed"
    assert d["first_provenance"] == "planner-selected" and d["first_author"] == "planner"
    # orders 7 and 9 were 'best' by the evaluator, but 8 was not the planner's and 9 asked for extension
    it = ev["intention"]
    assert it["verdict"] is False and [r["verdict"] for r in it["intervals"]] == [True, False, False]
    assert "asked for the other direction" in it["note"] and "not by the planner" in it["note"]
    assert "command 9's hand joint goal" in it["note"] and "order" not in it["note"]


def test_an_act_of_one_record_is_listed_under_the_command_that_made_that_record():
    """A touch at its onset record (HX3, B02: step0 == step1) holds no change between its records; its one interval is
    the command in force at that record, whose burst made it, and its history ends at the record before."""
    tr = _HandTrack()
    d = dict(_act(), cap="HX3", sub="R:index", step0=1012, step1=1012, i0=12, i1=12)
    att = _joiner().command_intervals(d, tr)
    assert [(e["command"], e["writer"], e["records"]) for e in att["intervals"]] == [(8, "reflex_noise", [1012, 1012])]
    assert att["writer_class"] == "spontaneous" and att["history"][-1]["records"][-1] == 1011
    assert ST.attribution_code(dict(d, evidence=dict(attribution=att))) == ST.ATTR_ONE_CLASS


def test_an_act_under_one_planner_order_keeps_one_writer_class():
    tr = _HandTrack()
    d = dict(_act(), step0=1001, step1=1009, i0=1, i1=9)
    att = _joiner().command_intervals(d, tr)
    assert [e["command"] for e in att["intervals"]] == [7] and att["writer_class"] == "planner-selected"
    assert _joiner().intention_intervals(d, 0, att)["verdict"] is True, "order 7: best, reached, asked this way"


# ------------------------------------------------------------------ the four stages
def test_the_four_stages_are_read_apart_and_a_selection_is_not_intention():
    tr = _HandTrack(bracket=True)
    d = _act()
    j = _joiner(_lane())
    j.attach([d], 0, tr)
    st = d["evidence"]["stages"]
    sel = st["planner_selected"]
    # order 8 was not the planner's: NO, on that fact alone; the goal distances are reported, never judged
    assert sel["verdict"] is False and "not the planner's" in sel["why"]
    assert sel["planner_commands"] == 2 and sel["commands_in_act"] == 3
    assert all("nearest" not in o for o in sel["commands"])
    assert sel["commands"][0]["margin"] == pytest.approx(0.05) and sel["commands"][0]["rank"] == 1
    assert sel["commands"][2]["rank"] == 2 and sel["commands"][2]["margin"] == pytest.approx(-0.05)
    assert sel["commands"][0]["hand_cost_first"] == 2.0 and sel["commands"][0]["hand_cost_selected"] == 1.5
    assert sel["commands"][1]["note"] == ("written by the motor controller with exploration noise (reflex_noise), "
                                          "not by the planner")
    assert "never read as intention" in sel["rule"]
    # achieved and predicted read one window per order: its whole hold
    ach = st["achieved"]
    assert ach["verdict"] is False and [o["asked"] for o in ach["commands"]] == ["same", "same", "opposite"]
    assert [o["window"] for o in ach["commands"]] == [[1002, 1009], [1010, 1014], [1015, 1019]]
    pred = d["evidence"]["prediction"]
    assert [r["window"] for r in pred["intervals"]] == [o["window"] for o in ach["commands"]]
    assert [r["order_began"] for r in pred["intervals"]] == [1002, 1010, 1015]
    assert [r["verdict"] for r in pred["intervals"]] == [True, False, True]
    assert pred["commands_judged"] == 3 and pred["commands_predicted"] == 2 and pred["verdict"] is False
    assert st["predicted"]["commands_judged"] == 3 and st["predicted"]["commands_predicted"] == 2
    assert st["improved"]["verdict"] is None and "cannot show" in st["improved"]["note"]
    # the store keeps the units for the tracker
    t = ST.DetTable.from_dicts([d])
    assert t["pred_n"].tolist() == [3] and t["pred_k"].tolist() == [2] and t["sel"].tolist() == [0]


def test_planner_selected_is_not_judged_when_every_command_was_the_planners():
    """Review fix 3: no noise band of the goal distance has been measured, so a chosen command's rank is reported and
    never judged (no band is invented)."""
    tr = _HandTrack(bracket=True)
    d = dict(_act(), step0=1003, step1=1009, i0=3, i1=9)              # inside order 7's hold
    j = _joiner()
    j.attach([d], 0, tr)
    sel = d["evidence"]["stages"]["planner_selected"]
    assert sel["verdict"] is None and "not judged" in sel["why"] and "noise band" in sel["why"]
    assert sel["commands"][0]["rank"] == 1 and sel["commands"][0]["margin"] == pytest.approx(0.05)
    assert ST.DetTable.from_dicts([d])["sel"].tolist() == [-1]


def test_an_act_of_other_parts_keeps_the_order_at_its_peak():
    tr = _HandTrack()
    j = EV.Joiner(None, [], ["index_R"], dict(HEADER, handle_names=["index_R", "lowerarm_R"], handle_bodies=[0, 0]))
    cr = j.credit(dict(cap="B06", sub="R", step0=1008, step1=1017, i0=8, i1=17, parts=["arm_R"]), tr)
    assert cr["order_seq"] in (7, 8, 9) and cr["lag_records"] == 0 and "rule" not in cr


# ------------------------------------------------------------------ the store and the tracker
def _det(step, *, attr=None, prov="planner-selected", intent=False, pred=None, hand=True, alone=None, sel=None,
         units=None):
    """units: (commands judged, commands predicted) of an act of the hands alone recorded since the change."""
    p = None if pred is None else dict(kind="efference", verdict=pred)
    if p is not None and units is not None:
        p.update(commands_judged=units[0], commands_predicted=units[1])
    ev = dict(prediction=p, intention=dict(verdict=intent), credit=dict(provenance=prov, moved=None, measured=True))
    if attr is not None:
        ev["attribution"] = dict(intervals=[dict(command=1)], writer_class=attr)
    if sel is not None:
        ev["stages"] = dict(planner_selected=dict(verdict=sel), achieved=dict(verdict=None))
    d = dict(cap="H05", sub="R:index", key="k", body=0, step0=step, step1=step + 2, episode="e%d" % step,
             provenance=prov, live=True, scripted=False, caregiver_touch=False, caregiver_present=False,
             posture="back", objects=[], measures={"m": 1.0}, parts=["finger_R_index"] if hand else ["arm_R"],
             evidence=ev)
    if alone is not None:
        d["alone_inside"] = alone
    return d


def test_the_store_marks_how_each_acts_writer_is_attributed():
    t = ST.DetTable.from_dicts([_det(0), _det(100, attr="planner-selected"), _det(200, attr="mixed"),
                                _det(300, hand=False), _det(400, alone=True, sel=False), _det(500, alone=False)])
    assert t["attr"].tolist() == [ST.ATTR_HAND_UNMEASURED, ST.ATTR_ONE_CLASS, ST.ATTR_MIXED, ST.ATTR_PEAK,
                                  ST.ATTR_HAND_UNMEASURED, ST.ATTR_HAND_UNMEASURED]
    assert t["alone"].tolist() == [-1, -1, -1, -1, 1, 0] and t["sel"].tolist() == [-1, -1, -1, -1, 0, -1]


def test_old_single_order_credits_and_mixed_acts_count_for_neither_side():
    old = [_det(i * 100, intent=True, pred=True) for i in range(5)]                      # credited to one order
    mixed = [_det(1000 + i * 100, attr="mixed", prov="mixed", intent=False, pred=True, units=(1, 1)) for i in range(5)]
    one = [_det(2000 + i * 100, attr="planner-selected", intent=True, pred=True, units=(1, 1)) for i in range(4)]
    ex = dict(steps=3000, by_author={"planner-selected": 1500, "spontaneous": 1500}, windows={}, bodies=8)
    out = TR.evaluate(old + mixed + one, ex, dict(count=0, occasions=0, steps=100000), None)
    sel = out["stages"]["selected_intentionally"]
    assert sel["decisive"] == 4 and sel["planner_occasions"] == 4 and sel["babble_occasions"] == 0
    assert sel["writer_not_measured_occasions"] == 5 and sel["mixed_writer_occasions"] == 5
    pba = out["stages"]["predicted_before_action"]
    assert pba["acts_judged"] == 9 and pba["acts_read_on_one_credited_order_not_judged"] == 5
    assert pba["earlier_rule"]["judged"] == 5 and pba["earlier_rule"]["predicted"] == 5
    readings = out["act_stage_readings"]
    assert readings["acts"] == 9 and readings["planner_selected"] == {"yes": 0, "no": 0, "not_judged": 9}
    # the page's Predicted cell says how many acts it no longer judges
    text = V.stage_text("predicted_before_action", dict(pba, reached=False), {}, {})["text"]
    assert "5 acts of the hands alone recorded before each command was listed are not judged" in text


def test_a_row_of_acts_recorded_before_the_change_reads_not_judged_with_the_true_reason():
    """Review blocker 1: the H05 row read "Not measurable: ... the learner's lane had no record at them" after the
    restart, and its pass vanished from the headline. Now: NOT JUDGED, the reason, the count and the withdrawn earlier
    reading, and the headline and the milestone name the pass that was withdrawn."""
    old = [_det(i * 100, pred=True) for i in range(8)]
    ex = dict(steps=3000, by_author={}, windows={}, bodies=8)
    out = TR.evaluate(old, ex, dict(count=0, occasions=0, steps=100000), None)
    st = out["stages"]["predicted_before_action"]
    assert st["reached"] is None and st["not_judged"] is True and "p_value" not in st
    assert "recorded before Oct 8, 2026" in st["note"] and "lane" not in st["note"]
    assert st["acts_read_on_one_credited_order_not_judged"] == 8
    assert st["earlier_rule"]["predicted"] == 8 and st["earlier_rule"]["judged"] == 8
    assert st["earlier_rule"]["passes_at_line"] is True and st["earlier_rule"]["stage_reached"] is True
    cell = V.stage_text("predicted_before_action", st, {}, {})
    assert cell["state"] == "notjudged" and cell["text"].startswith("Not judged: 8 acts of the hands alone were")
    assert "under that withdrawn rule 8 of 8 occasions were predicted, enough to pass Predicted" in cell["text"]
    # a row whose earlier test passes while Repeats does not stand would still wait: it is not named as withdrawn
    waiting = dict(st, earlier_rule=dict(st["earlier_rule"], stage_reached=False))
    assert "while the stage waits for Repeats" in V.stage_text("predicted_before_action", waiting, {}, {})["text"]
    assert V.withdrawn_rows({"H05": dict(name="x", stages=dict(predicted_before_action=waiting))}) == []
    caps = {"H05": dict(name="Individual finger isolation", stages=dict(observed_once=dict(reached=True),
                                                                      repeated=dict(reached=True),
                                                                      predicted_before_action=st))}
    m = V.milestone(caps)
    assert m["withdrawn"] == ["Individual finger isolation"]
    assert "Predicted is not judged now for Individual finger isolation" in V.milestone_line(m)
    assert V.MARKERS["notjudged"] == "Not judged"


def test_the_stage_history_names_the_stages_a_rules_change_withdrew(tmp_path):
    from mimo_bridge.tools.observer import daemon as DM
    from mimo_bridge.tools.observer import history as HI
    before = {"H05": dict(stages=dict(predicted_before_action=dict(reached=True, first_step=383684)))}
    DM.stage_history(str(tmp_path), before, 400000, control="old rules")
    after = {"H05": dict(stages=dict(predicted_before_action=dict(reached=None, not_judged=True,
                                                                  note="not judged: 8 acts ...")))}
    DM.stage_history(str(tmp_path), after, 400100, control="new rules")
    lines = [json.loads(x) for x in (tmp_path / "stage_history.jsonl").read_text(encoding="utf-8").splitlines()]
    reset = [x for x in lines if x.get("change") == "reset"][-1]
    assert reset["withdrawn"] == [dict(cap="H05", stage="predicted_before_action", first_step=383684, now="not judged",
                                       why="not judged: 8 acts ...")]
    assert HI.read_resets(str(tmp_path))[-1]["withdrawn"][0]["now"] == "not judged"
    assert "1 stage that stood before does not stand under the new rules: H05 Predicted (not judged)" == \
        V._withdrawn_text(reset["withdrawn"])[2:]


# ------------------------------------------------------------------ the Predicted rule: commands as units in clusters
def test_one_unit_per_occasion_is_the_binomial_test():
    k = np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
    assert TR.cluster_binom_above(k, np.ones(10)) == pytest.approx(TR.binom_above(8, 10, 0.5), rel=1e-9)
    assert TR.cluster_binom_above([0, 0], [1, 1]) == 1.0


def test_the_predicted_stage_counts_each_command_with_the_acts_as_clusters():
    acts = [_det(i * 100, attr="mixed", prov="mixed", pred=False, units=(4, 3)) for i in range(30)]
    ex = dict(steps=3000, by_author={}, windows={}, bodies=8)
    st = TR.evaluate(acts, ex, None, None)["stages"]["predicted_before_action"]
    assert st["commands_judged"] == 120 and st["commands_predicted"] == 90 and st["design_effect"] == pytest.approx(4.0)
    # 90 of 120 commands in 30 clusters of 4: the binomial tail at 22.5 of 30
    assert st["p_value"] == pytest.approx(float(special.betainc(22.5, 30 - 22.5 + 1.0, 0.5)), rel=1e-9)
    assert st["predicted"] == 30, "each act: 3 of its 4 commands passed, more than half"
    # under 4342f9194's rule every one of these acts failed (one of its 4 commands did not pass)
    text = V.stage_text("predicted_before_action", dict(st, reached=False), {}, {})["text"]
    assert "90 of 120 commands predicted, over 30 occasions" in text


def _null_rates(rng, reps, n, shared):
    """The false-pass rate of the Predicted rule for acts of the hands alone with NO prediction skill: each command
    passes with chance one half, independently (shared False) or all commands of an act together (shared True), the
    commands per act drawn from control run 1's acts of the hands alone (the review's table, 4,642 acts)."""
    sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])
    w = np.array([88, 1314, 1288, 741, 402, 277, 174, 153, 93, 40, 21, 20, 15, 10, 4, 2], dtype=float)
    m = rng.choice(sizes, size=(reps, n), p=w / w.sum())
    k = m * (rng.random((reps, n)) < 0.5) if shared else rng.binomial(m, 0.5)
    K, M = k.sum(axis=1).astype(float), m.sum(axis=1).astype(float)
    D = (m.astype(float) ** 2).sum(axis=1) / M
    ke, ne = K / D, M / D
    p = np.where(ke > 0, special.betainc(np.maximum(ke, 1e-12), ne - ke + 1.0, 0.5), 1.0)
    ok = (p <= TR.ALPHA) & ((2 * k > m).sum(axis=1) >= TR.MIN_COUNT)
    return float(ok.mean()), np.array([TR.cluster_binom_above(k[i], m[i]) for i in range(3)]), p[:3]


def test_with_no_prediction_skill_the_rule_passes_at_most_at_its_line():
    """Review blocker 2: the false-pass rate under no skill stays at the declared line (5 in 100), whether an act's
    commands are independent or all alike (the case the design effect is set for). The full simulation, with 6
    scenarios and 7 row sizes: D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/observer/predicted_null_simulation.json."""
    rng = np.random.default_rng(20261008)
    for n, shared in ((30, True), (300, True), (30, False)):
        rate, mine, vectorized = _null_rates(rng, 4000, n, shared)
        assert rate <= TR.ALPHA + 2 * np.sqrt(TR.ALPHA * (1 - TR.ALPHA) / 4000), (n, shared, rate)
        assert mine == pytest.approx(vectorized, rel=1e-9), "the tracker's function is the one simulated"


def test_the_rows_count_the_other_digits_inside_the_act_and_the_page_says_what_alone_means():
    dets = [_det(i * 100, attr="mixed", prov="mixed", alone=(i < 3)) for i in range(7)] + [_det(900)]
    ex = dict(steps=1000, by_author={}, windows={}, bodies=8)
    row = TR.evaluate(dets, ex, None, None)
    assert row["alone_inside"]["yes"] == 3 and row["alone_inside"]["no"] == 4 and row["alone_inside"]["not_measured"] == 1
    note = V.alone_note(row)
    assert "END within 5 degrees" in note and "3 of 7 acts measured" in note and "1 more" in note
    rows = V.finger_rows({}, {})
    assert rows[0]["label"] == "Left thumb moves; the other fingers end where they began"


def test_no_output_says_the_model_learned_no_intentional_behavior():
    """The Chosen stage reads the wellness evaluator's tie band (register entry
    OBSERVER_SELECTED_STAGE_READS_UNOPTIMIZED_EVALUATOR): the page says its 'not yet' does not show that."""
    text = V.stage_text("selected_intentionally", dict(reached=False, decisive=0, mixed_writer_occasions=2), {},
                        dict(planner_tied=True, choices_judged=10, choices_best=0))["text"]
    assert "does not show that the model learned no intentional behavior" in text
    assert "2 occasions whose acts of the hands alone ran under commands of more than one writer" in text
    chosen = [d for k, _, d in V.STAGE_TITLES if k == "selected_intentionally"][0]
    assert "credited to no single command" in chosen and "learned no intentional behavior" in chosen
    line = V.milestone_line(dict(done=[], best=0, closest=[]))
    assert "does not show that the model learned no intentional behavior" in line


# ------------------------------------------------------------------ the detector and the hand goal's order
def test_h05_reports_how_far_the_other_digits_moved_inside_the_act():
    from mimo_bridge.tools.observer import detectors as D
    curl = np.zeros((6, 2, 5))
    curl[:, 1, 1] = [0, 4, 9, 6, 2, 1]          # the right middle finger moves 9 degrees and comes back to 1
    tr = SimpleNamespace(curl=curl)
    exc = D.others_excursion(tr, 1, D.DIGITS.index("index"), 0, 5)
    assert exc["middle"] == 9.0 and exc["ring"] == 0.0 and set(exc) == {"middle", "ring", "little", "thumb"}


def test_the_hand_goal_order_is_the_controllers():
    mujoco = pytest.importorskip("mujoco")
    from mimo_bridge.hand_target_control import geometry_from_model
    from mimo_bridge.tools.observer.kinematics import hand_goal_qpos
    parts = []
    for side in ("left", "right"):
        fingers = "".join(
            '<body name="{s}_{d}1" pos="0.01 0 0"><joint name="{s}_{d}_a" type="hinge" limited="true" range="-1 1"/>'
            '<geom size="0.005"/><body name="{s}_{d}2" pos="0.01 0 0"><joint name="{s}_{d}_b" type="hinge" '
            'limited="true" range="-1 1"/><geom size="0.005"/></body></body>'.format(s=side, d=d)
            for d in ("th", "ff", "mf", "rf", "lf"))
        parts.append('<body name="{s}_hand" pos="0 {y} 0"><joint name="{s}_w" type="hinge" limited="true" '
                     'range="-1 1"/><geom size="0.01"/>{f}</body>'.format(s=side, y=0.1 if side == "left" else -0.1,
                                                                          f=fingers))
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody>%s</worldbody></mujoco>" % "".join(parts))
    names, bodies = [], []
    for side, s in (("L", "left"), ("R", "right")):
        names.append("wrist_" + side)
        bodies.append(int(m.body(s + "_hand").id))
        for digit, key in (("thumb", "th"), ("index", "ff"), ("middle", "mf"), ("ring", "rf"), ("pinky", "lf")):
            names.append("%s_%s" % (digit, side))
            bodies.append(int(m.body("%s_%s2" % (s, key)).id))
    dec = SimpleNamespace(handle_names=names, handles=[SimpleNamespace(body_id=b) for b in bodies])
    want = [int(j["qpos_address"]) for j in geometry_from_model(m, dec)["joints"]]
    assert hand_goal_qpos(m, names, bodies) == want and len(want) == 20
