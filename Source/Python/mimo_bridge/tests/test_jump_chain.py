# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The jump chain: the planner-against-reflex probe's arithmetic (docs/gain_of_function_v4/probe_planner_vs_reflex.py)
and the chain reader (mimo_bridge/tools/regression_pack.py `chain`).

On data whose answer is known: the mirrored order is exactly as far from the controller's order as the planner's,
turned the other way; goal closure; the per-start differences and their body intervals, with and without body 2;
a planner whose direction helps is credited and one whose deviation is uninformed is not; the evaluator's
agreement; the watch file's jump rule in both directions and in ln; the planner-candidates ranking read without
body 2 through the probe's own summarise; the four links in one table.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mimo_bridge.tools import regression_pack as RP

REPO = Path(__file__).resolve().parents[4]


def _probe():
    spec = importlib.util.spec_from_file_location("probe_planner_vs_reflex",
                                                  REPO / "docs" / "gain_of_function_v4" / "probe_planner_vs_reflex.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _probe()


def _axis_angle(axis, deg):
    axis = torch.tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    h = math.radians(deg) / 2
    return torch.cat([axis * math.sin(h), torch.tensor([math.cos(h)], dtype=torch.float64)]).float()


# =========================================================================== the matched control
def test_the_mirror_is_as_far_from_the_controller_as_the_chosen_order_the_other_way():
    H = 4
    reflex = torch.zeros(1, H, 9)
    chosen = torch.zeros(1, H, 9)
    for h in range(H):
        reflex[0, h, 3:7] = _axis_angle([0, 0, 1], 10.0 * h)
        chosen[0, h, 3:7] = V._qmul(_axis_angle([1, 0, 0], 20.0), reflex[0, h, 3:7])     # a 20 degree turn
    reflex[..., 0:3], chosen[..., 0:3] = 0.10, 0.13
    reflex[..., 7], chosen[..., 7] = 1.0, 1.2
    reflex[..., 8], chosen[..., 8] = 0.5, 0.9
    m = V.mirror_raw(chosen, reflex)
    assert torch.allclose(m[..., 0:3], torch.full_like(m[..., 0:3], 0.07), atol=1e-6)
    assert torch.allclose(V.turn_deg(m[..., 3:7], reflex[..., 3:7]), V.turn_deg(chosen[..., 3:7], reflex[..., 3:7]), atol=1e-3)
    # the other way: the mirror is twice the turn from the chosen order
    assert torch.allclose(V.turn_deg(m[..., 3:7], chosen[..., 3:7]), torch.full((1, H), 40.0), atol=1e-2)
    assert torch.allclose(m[..., 7], torch.full((1, H), 0.8), atol=1e-6)
    assert torch.allclose(m[..., 8], torch.full((1, H), 0.1), atol=1e-6)
    # strength stays inside [0, 1] and the radius at or above 0
    chosen[..., 8] = 0.05
    chosen[..., 7] = 3.0
    m2 = V.mirror_raw(chosen, reflex)
    assert float(m2[..., 8].max()) <= 1.0 and float(m2[..., 7].min()) >= 0.0


def test_the_mirror_of_an_unchanged_order_is_the_order():
    reflex = torch.zeros(1, 3, 9)
    reflex[..., 3:7] = _axis_angle([0, 1, 0], 33.0)
    reflex[..., 8] = 0.4
    assert torch.allclose(V.mirror_raw(reflex.clone(), reflex), reflex, atol=1e-6)


def test_goal_closure():
    assert V.closure(0.5, 0.25) == pytest.approx(0.5)
    assert V.closure(0.5, 0.75) == pytest.approx(-0.5)
    assert V.closure(0.0, 0.1) is None and V.closure(None, 0.1) is None


# ============================================================================ the per-start readings
def _row(body, chosen, reflex, mirror, well=(3.0, 3.0, 3.0), pred=(3.0, 3.0), moments=(3.0, 3.0), mirror_error=False,
         d0=1.0, matched=True):
    """One start: each order's goal closure (the distance at the end is d0 x (1 - closure))."""
    def order(c, w, p, m):
        return dict(goal_closure=c, goal_distance=[d0, d0 * (1.0 - c)], wellness_tenure=w,
                    judge_pred={"1": p, "2": p, "3": p}, wellness_moments=[m] * 5)
    o = {"chosen": order(chosen, well[0], pred[0], moments[0]), "reflex": order(reflex, well[1], pred[1], moments[1]),
         "mirror": dict(error="IK failed") if mirror_error else order(mirror, well[2], 3.0, 3.0)}
    return dict(body=body, start=0, outcomes=o, goal_distance_at_decision=d0,
                mirror_match=dict(matched=bool(matched and not mirror_error)),
                deviation=dict(turn_deg_mean=5.0, position_cm_mean=1.0, identical=False))


def test_an_informed_planner_beats_its_mirror_and_an_uninformed_one_does_not():
    informed = [_row(b, 0.20, 0.10, 0.00) for b in range(8) for _ in range(3)]
    s = V.summarise(informed, draws=500)
    pv = s["planner_value_goal_closure"]
    assert pv["mean"] == pytest.approx(0.20) and pv["interval"] == pytest.approx([0.20, 0.20])
    assert s["advantage_over_reflex_goal_closure"]["mean"] == pytest.approx(0.10)
    assert s["mirror_over_reflex_goal_closure"]["mean"] == pytest.approx(-0.10)
    assert s["physical_improvement_over_reflex"] is True
    # an uninformed deviation: whichever way the order leaves the controller's, the body closes 0.1 more
    uninformed = [_row(b, 0.20, 0.10, 0.20) for b in range(8)]
    s2 = V.summarise(uninformed, draws=500)
    assert s2["advantage_over_reflex_goal_closure"]["mean"] == pytest.approx(0.10)
    assert s2["planner_value_goal_closure"]["mean"] == pytest.approx(0.0)
    assert s2["physical_improvement_over_reflex"] is False


def test_body_2_is_left_out_of_the_readings_without_it():
    rows = [_row(b, 0.10, 0.10, 0.10) for b in range(8)]
    rows[2] = _row(2, 0.90, 0.10, 0.10)
    all8, wo = V.summarise(rows, draws=200), V.summarise(rows, exclude=(2,), draws=200)
    assert all8["planner_value_goal_closure"]["mean"] == pytest.approx(0.80 / 8)
    assert wo["planner_value_goal_closure"]["mean"] == 0.0 and wo["bodies"] == [0, 1, 3, 4, 5, 6, 7]
    assert "2" not in wo["planner_value_goal_closure"]["per_body"]


def test_a_mirror_that_could_not_run_leaves_only_its_differences_out():
    rows = [_row(b, 0.2, 0.1, 0.0, mirror_error=(b == 0)) for b in range(8)]
    s = V.summarise(rows, draws=200)
    assert s["mirror_not_executed"] == 1
    assert s["advantage_over_reflex_goal_closure"]["bodies"] == 8 and s["planner_value_goal_closure"]["bodies"] == 7


def test_the_headline_compares_the_planner_with_the_reflex_order_not_with_its_mirror():
    """Fails on the probe's first rule, which passed on planner-over-mirror alone: at 414,200 the planner's order
    did worse than the reflex order in 11 of 16 starts, and a mirror that does worse still lets such a planner pass."""
    worse = [_row(b, 0.05, 0.15, -0.20) for b in range(8) for _ in range(3)]
    s = V.summarise(worse, draws=500)
    assert s["planner_over_mirror"]["mean"] == pytest.approx(0.25), "it beats its mirror"
    assert s["planner_over_reflex"]["mean"] == pytest.approx(-0.10), "it loses to the reflex order"
    assert s["starts_nearer_than_reflex_per_100"] == 0.0
    assert s["physical_improvement_over_reflex"] is False
    better = [_row(b, 0.25, 0.15, 0.05) for b in range(8) for _ in range(3)]
    assert V.summarise(better, draws=500)["physical_improvement_over_reflex"] is True


def test_one_start_with_a_small_goal_distance_does_not_outweigh_the_rest():
    """Fails on the probe's first statistic, a mean of per-start closure ratios: at 414,200 one start began 0.074
    from its goal and its closures ran -4.0 to -9.2, against -1.5 to +0.6 for every other start."""
    rows = [_row(b, 0.30, 0.20, 0.10, d0=1.0) for b in range(8) for _ in range(3)]
    # one start 0.02 from its goal where the planner's order ended 0.1 further than the reflex order's
    rows[0] = _row(0, -4.0, 1.0, 0.0, d0=0.02)
    s = V.summarise(rows, draws=500)
    assert s["advantage_over_reflex_goal_closure"]["mean"] < 0, "the per-start ratios: one start drags it below 0"
    assert s["planner_over_reflex"]["mean"] > 0, "one scale for every start: the other 23 starts carry it"
    assert s["mean_start_distance"] == pytest.approx((23 * 1.0 + 0.02) / 24)


def test_an_unmatched_mirror_is_left_out_of_the_mirror_control():
    rows = [_row(b, 0.20, 0.10, 0.00, matched=(b != 0)) for b in range(8)]
    s = V.summarise(rows, draws=200)
    assert s["planner_over_mirror"]["bodies"] == 7 and s["planner_over_reflex"]["bodies"] == 8
    assert s["mirror_matched"] == dict(matched=7, starts=8)


def test_the_fingers_mirror_in_joint_space_around_the_controllers_projection():
    c, r = torch.tensor([[0.2, 0.5, -0.1]]), torch.tensor([[0.1, 0.4, 0.0]])
    m = V.mirror_joints(c, r, lambda x: x.clamp(-0.05, 0.25))
    assert torch.allclose(m, torch.tensor([[0.0, 0.25, 0.1]])), "2 r - c = (0, 0.3, 0.1), the second joint clamped"
    # a clamp that shrinks the mirror's deviation is caught by the executed-deviation match
    chosen = dict(turn_deg_mean=10.0, position_cm_mean=2.0, finger_joint_deg_mean=8.0, eye_deg=1.0)
    ok = dict(turn_deg_mean=10.5, position_cm_mean=1.95, finger_joint_deg_mean=7.5, eye_deg=1.05)
    cut = dict(ok, finger_joint_deg_mean=4.0)
    assert V.deviation_matched(chosen, ok)["matched"] is True
    got = V.deviation_matched(chosen, cut)
    assert got["matched"] is False and got["differs_in"] == ["finger_joint_deg_mean"]
    # below the floors two orders are the same to the body
    tiny = dict(turn_deg_mean=0.01, position_cm_mean=0.01, finger_joint_deg_mean=None, eye_deg=None)
    assert V.deviation_matched(tiny, dict(tiny, turn_deg_mean=0.08))["matched"] is True


def test_the_executed_deviation_reads_turns_positions_joints_and_eyes():
    H = 4
    r = torch.zeros(1, H, 9)
    r[..., 6] = 1.0
    c = r.clone()
    c[0, 0, 3:7] = _axis_angle([0, 0, 1], 20.0)
    c[0, 1, 0:3] = torch.tensor([3.0, 4.0, 0.0])
    d = V.executed_deviation(c, r, gaze_idx=3, joints=torch.tensor([[0.1]]), reflex_joints=torch.tensor([[0.0]]),
                             gaze=[0.0, 0.0], reflex_gaze=[math.radians(3.0), math.radians(4.0)])
    assert d["turn_deg_mean"] == pytest.approx(20.0 / 3, abs=1e-3) and d["position_cm_mean"] == pytest.approx(5.0 / 3)
    assert d["finger_joint_deg_mean"] == pytest.approx(math.degrees(0.1)) and d["eye_deg"] == pytest.approx(5.0)
    assert d["identical"] is False


def test_the_evaluators_agreement_with_what_happened():
    agree = [_row(b, 0.1, 0.1, 0.1, pred=(3.0 + 0.1 * s, 3.0), moments=(3.0 + 0.2 * s, 3.0))
             for b in range(8) for s in (1, -1)]
    a = V.summarise(agree, draws=200)["evaluator_agreement"]["50ms"]
    assert a["sign_agreement"] == 1.0 and a["slope_predicted_on_actual"] == pytest.approx(0.5)
    blind = [_row(b, 0.1, 0.1, 0.1, pred=(3.0, 3.0), moments=(3.0 + 0.2 * s, 3.0)) for b in range(8) for s in (1, -1)]
    a2 = V.summarise(blind, draws=200)["evaluator_agreement"]["150ms"]
    assert a2["slope_predicted_on_actual"] == 0.0 and a2["sign_agreement"] is None


# =============================================================================== the chain reader
def _watch(ac, spread, tied=1.0):
    def block(v, line):
        return dict(baseline_last_20_median=v, median_of_20_change_abs_max=line)
    return dict(stats=dict(action_conditioned_proprioception=block(ac, 0.0012),
                           sensitivity_proprioception=block(0.009, 1.68), chosen_tied_share=block(tied, 0.0)),
                per_body=dict(efference_action_conditioned=dict(all_bodies=block(0.386, 0.121),
                                                                without_body_2=block(0.397, 0.128)),
                              candidate_spread=dict(all_bodies=block(spread, 0.4475), without_body_2=block(spread, 0.4815))))


def test_the_watch_readings_jump_only_beyond_their_no_change_line():
    before = _watch(-0.00004, 0.0038)
    r = RP.chain_watch(before, None)
    assert r["action_conditioned_proprioception"]["all"]["jump_needs"] == pytest.approx(-0.00004 + 0.0012)
    assert r["candidate_spread"]["without_body_2"]["jump_needs"] == pytest.approx(0.0038 * math.exp(0.4815))
    # the tie share is no longer read (jump_chain_v3): the tie band is 12 to 22 times the predicted spread between the
    # candidates, so every choice ties whatever the forward model does; a watch file that still carries it changes nothing
    assert "chosen_tied_share" not in r and "chosen_tied_share" not in RP.CHAIN_WATCH
    assert "no body split" in r["sensitivity_proprioception"]["without_body_2"]["note"]
    small = RP.chain_watch(before, _watch(0.0010, 0.0060))
    assert small["action_conditioned_proprioception"]["all"]["jump"] is False
    assert small["candidate_spread"]["without_body_2"]["jump"] is False
    big = RP.chain_watch(before, _watch(0.0020, 0.0100, tied=0.9))
    assert big["action_conditioned_proprioception"]["all"]["jump"] is True
    assert big["candidate_spread"]["without_body_2"]["jump"] is True
    assert "chosen_tied_share" not in big


def _reflex_result(folder: Path, mean, interval, mirror=(0.02, [0.01, 0.03])):
    folder.mkdir(parents=True)
    block = dict(planner_over_reflex=dict(mean=mean, interval=interval),
                 planner_over_mirror=dict(mean=mirror[0], interval=mirror[1]),
                 physical_improvement_over_reflex=interval[0] > 0 and mirror[1][0] > 0)
    (folder / "result.json").write_text(json.dumps(dict(checkpoint="c", probe=V.PROBE_VERSION, evaluated_starts=24,
                                                        summary=dict(all_bodies=block, without_body_2=block))),
                                        encoding="utf-8")
    return folder


def test_the_physical_link_jumps_only_above_zero_above_the_before_and_above_its_mirror(tmp_path):
    b = _reflex_result(tmp_path / "b", 0.001, [-0.01, 0.012])
    small = _reflex_result(tmp_path / "a1", 0.010, [0.002, 0.020])
    big = _reflex_result(tmp_path / "a2", 0.030, [0.015, 0.045])
    big_no_mirror = _reflex_result(tmp_path / "a3", 0.030, [0.015, 0.045], mirror=(0.0, [-0.01, 0.01]))
    r1 = RP.chain(dict(reflex=str(b)), dict(reflex=str(small)))
    assert r1["physical_improvement"]["without_body_2"]["jump"] is False, "its mean is inside the before's interval"
    r2 = RP.chain(dict(reflex=str(b)), dict(reflex=str(big)))
    assert r2["physical_improvement"]["without_body_2"]["jump"] is True
    r3 = RP.chain(dict(reflex=str(b)), dict(reflex=str(big_no_mirror)))
    assert r3["physical_improvement"]["without_body_2"]["beats_reflex"] is True
    assert r3["physical_improvement"]["without_body_2"]["jump"] is False, "any deviation would have helped as much"


def test_link_3s_bar_is_its_interval_condition():
    """Fails on the first record's bar (before + 9.7 = 55.95): at the pre-repair rounds' interval half-widths
    (7.9-15.7 points) the interval clears 50 only at a reading of 58-66."""
    assert RP.ranking_bar(46.25, 9.7, RP.CHAIN_RANKING_HALF_WIDTHS) == pytest.approx([57.9, 65.7])
    assert RP.ranking_bar(60.0, 9.7, RP.CHAIN_RANKING_HALF_WIDTHS) == pytest.approx([69.7, 69.7])
    assert "58-66" in RP.chain({})["planner_choice_quality"]["rule"]


def test_every_link_states_its_reachability_arithmetic():
    rep = RP.chain({})
    for link in ("1", "2", "3", "4"):
        text = rep["reachability"][link]
        assert "Judged window" in text and any(ch.isdigit() for ch in text), link
    assert "Reachability arithmetic" in RP.chain_table(rep)


def test_a_before_watch_file_past_the_before_checkpoint_is_refused(tmp_path):
    w = tmp_path / "w.json"
    w.write_text(json.dumps(dict(_watch(-0.00004, 0.0038), last_report=415100, first_report=380700)), encoding="utf-8")
    with pytest.raises(ValueError, match="past the before checkpoint"):
        RP.chain(dict(watch=str(w)), before_step=415025)
    ok = RP.chain(dict(watch=str(w)), before_step=415100)
    assert ok["watch_files"]["before"]["last_report"] == 415100 and len(ok["watch_files"]["before"]["sha256"]) == 64


def test_the_ranking_is_read_without_body_2_through_the_probes_own_summary(tmp_path):
    from mimo_bridge.tests.test_planner_candidates import ranked_rows
    rows = ranked_rows("exact")
    folder = tmp_path / "round"
    folder.mkdir()
    (folder / "result.json").write_text(json.dumps(dict(checkpoint="c", probe="planner_candidates_v2",
                                                        evaluated_starts=len(rows))), encoding="utf-8")
    (folder / "starts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = RP.chain_candidates(folder, draws=200)
    assert got["all"]["ranking_goal_distance_end"]["per_100"] == 100.0
    assert got["without_body_2"]["ranking_goal_distance_end"]["per_100"] == 100.0
    assert got["all"]["ranking_goal_distance_end"]["bodies"] == 8
    assert got["without_body_2"]["ranking_goal_distance_end"]["bodies"] == 7
    rep = RP.chain(dict(candidates=str(folder)), dict(candidates=str(folder)), draws=200)
    assert rep["planner_choice_quality"]["without_body_2"]["jump"] is False, "no rise: the same round twice"


def test_the_chain_table_names_every_link_and_its_rule(tmp_path):
    w = tmp_path / "w.json"
    w.write_text(json.dumps(_watch(-0.00004, 0.0038)), encoding="utf-8")
    text = RP.chain_table(RP.chain(dict(watch=str(w))))
    for link in ("| 1 |", "| 2 |", "| 3 |", "| 4 |"):
        assert link in text
    assert "above" in text and "below" in text and "body 3" in text
    low = text.lower()
    for banned in ("honest", "barely", "slightly", "significantly", "substantially", "roughly"):
        assert banned not in low


# ======================================================== link 2: the planner recheck's captured share (2026-10-02)
def _round(folder: Path, rows, probe="planner_candidates_v3"):
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps(dict(checkpoint="c", probe=probe, evaluated_starts=len(rows))),
                                        encoding="utf-8")
    (folder / "starts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(folder)


def _forecast_rows(share, floors="kept"):
    """The planner track's synthetic rows (8 bodies x 2 starts), whose predicted token differences carry `share` of
    the real ones in every sense: floors 'kept' (rows of version 3), 'dropped' (rows of versions 1 and 2, which
    recorded no float32 floor) or 'above' (each pair's floor set above its real difference: within the floor)."""
    from mimo_bridge.tests.test_planner_candidates import synthetic_rows
    rows = copy.deepcopy(synthetic_rows(forecast=share))
    for r in rows:
        for route in r["sense_stats"].values():
            for depth in route.values():
                for st in depth.values():
                    if floors == "dropped":
                        st.pop("floor", None)
                    elif floors == "above":
                        st["floor"] = [10.0 * x for x in st["real_diff"]]
    return rows


def test_link_2_reads_the_planner_rechecks_captured_share_with_its_scale_and_judges_a_rise(tmp_path):
    """jump_chain_v2 read link 2 as the share of choices inside the tie band, a band 12 to 22 times the predicted
    spread between the candidates: 1 whatever the forward model does. v3 reads the probe's captured share (0 = the
    predicted difference between two candidates carries none of the real one, 1 = all of it), from the round link 3
    reads, and judges a rise beyond the rechecks' largest change with the after's interval above the before."""
    blind = _round(tmp_path / "blind", _forecast_rows(0.0))
    half = _round(tmp_path / "half", _forecast_rows(0.5))
    tiny = _round(tmp_path / "tiny", _forecast_rows(0.005))
    rep = RP.chain(dict(candidates=blind), dict(candidates=half), draws=200)
    link = rep["candidate_separation"]
    assert rep["version"] == "jump_chain_v3" and "tie band" in link["replaces"] and "1 = all of it" in link["reading"]
    cell = link["without_body_2"]
    assert cell["before"] == pytest.approx(0.0, abs=1e-6) and cell["after"] == pytest.approx(0.5, abs=1e-3)
    assert cell["jump_needs"] == pytest.approx(RP.CHAIN_CAPTURED_NOISE, abs=1e-6)
    assert cell["after_measurable"] is True and cell["jump"] is True
    after = rep["planner_choice_quality"]["after"]["without_body_2"]["captured_share"]
    assert after["bodies"] == 7 and set(after["senses"]) == set(RP.CHAIN_CAPTURED_SENSES)
    assert after["senses"]["proprioception"]["value"] == pytest.approx(0.5, abs=1e-3)
    # a rise inside the rechecks' largest change is no jump
    assert RP.chain(dict(candidates=blind), dict(candidates=tiny), draws=200)["candidate_separation"]["without_body_2"][
        "jump"] is False
    # the same round twice: no rise
    assert RP.chain(dict(candidates=half), dict(candidates=half), draws=200)["candidate_separation"]["all"]["jump"] is False
    # the table carries the reading, its scale ends and what it replaced; no tie-share row
    text = RP.chain_table(rep)
    assert "the planner recheck's captured share (probe planner_candidates_v3" in text
    assert "(0 = none, 100 = all of it, the goal line; below 0 = the wrong way; without body 2)" in text
    assert "12 to 22 times the predicted spread" in text and "50.00 in 100" in text
    assert "share of planner choices inside the tie band (1 = every choice tied" not in text


def test_link_2_reads_no_jump_from_a_round_without_the_float32_floor_or_within_it(tmp_path):
    blind = _round(tmp_path / "blind", _forecast_rows(0.0, floors="dropped"), probe="planner_candidates_v2")
    old = _round(tmp_path / "old", _forecast_rows(0.5, floors="dropped"), probe="planner_candidates_v2")
    floor = _round(tmp_path / "floor", _forecast_rows(0.5, floors="above"))
    rep = RP.chain(dict(candidates=blind), dict(candidates=old), draws=200)
    cell = rep["candidate_separation"]["without_body_2"]
    assert cell["after"] == pytest.approx(0.5, abs=1e-3) and cell["after_measurable"] is None
    assert cell["jump"] is False and "measured before planner_candidates_v3" in cell["note"]
    assert "rows measured before planner_candidates_v3" in RP.chain_table(rep), "the before side says so in the table"
    within = RP.chain(dict(candidates=blind), dict(candidates=floor), draws=200)
    cell = within["candidate_separation"]["without_body_2"]
    assert cell["after"] is None and cell["after_measurable"] is False and cell["jump"] is False
    assert cell["note"].startswith("not measurable")
    assert "not measurable (real difference within the float32 floor)" in RP.chain_table(within)
