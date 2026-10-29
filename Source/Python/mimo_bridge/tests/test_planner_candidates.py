# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The planner-candidate probe (docs/gain_of_function_v4/probe_planner_candidates.py).

Why the planner's choices tie is separated into three causes: the forward model's forecasts of the
candidates do not differ, the approximate evaluator (Judge B) cannot rank even the true outcomes, or
the planner does not follow predicted wellness. These tests pin the arithmetic on small synthetic
fixtures whose answer is known by construction (no model, no world), the result validator and the
scheduler manifest, and - on a small real model and body, marked slow - that the probe's candidates
and forecasts reproduce the production planning record at the same decision."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
PROBE_PATH = REPO / "docs" / "gain_of_function_v4" / "probe_planner_candidates.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_planner_candidates", PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_probe()
DRAWS = 400


# ============================================================================ synthetic rows
def synthetic_rows(forecast=1.0, evaluator=1.0, noise=1e-4, reverse_first_moment=False, seed=0,
                   K=5, T=5, d=6):
    """8 bodies x 2 starts. Each candidate has a true wellness effect; Judge B on the ACTUAL outcome
    shows `evaluator` of the grounded difference plus `noise`; the forecast carries `forecast` of the
    actual token difference and of Judge B's actual-outcome difference. reverse_first_moment makes a
    candidate's first moment the opposite of the rest of its tenure (a 50 ms look misleads)."""
    rng = np.random.default_rng(seed)
    rows = []
    for body in range(8):
        for start in range(2):
            # the planner's own order (index 0) always carries the middle effect of the five: a
            # planner that never reads predicted wellness is exactly as good as chance here
            e = np.sort(rng.normal(0.0, 0.05, size=K))
            effect = np.concatenate([[e[K // 2]], rng.permutation(np.delete(e, K // 2))])
            base = rng.normal(0.6, 0.02, size=5)
            w = np.zeros((K, T, 5))
            ja = np.zeros((K, T, 5))
            for c in range(K):
                for t in range(T):
                    e = -effect[c] if (reverse_first_moment and t == 0) else effect[c]
                    w[c, t] = base + e / 5.0 + rng.normal(0, 1e-5, 5)
                    ja[c, t] = base + evaluator * (w[c, t] - base) + rng.normal(0, noise, 5)
            jitter = 1e-7 if forecast > 0 else 0.0
            jp = {route: {str(dd): (base + forecast * (ja[:, dd - 1] - base)
                                    + rng.normal(0, 1.0, (K, 5)) * jitter).tolist()
                          for dd in P.DEPTHS} for route in ("planner", "learning")}
            stats = {}
            for route in ("planner", "learning"):
                for dd in P.DEPTHS:
                    for m in P.ALL_SENSES:
                        actual = rng.normal(0, 1, (K, d))
                        mean = actual.mean(0)
                        pred = mean + forecast * (actual - mean) + rng.normal(0, 1.0, (K, d)) * 1e-6 * (forecast > 0)
                        stats.setdefault(route, {}).setdefault(str(dd), {})[m] = P.sense_pair_stats(pred, actual, mean)
            rows.append(dict(
                body=body, start=start, step=100 + 10 * start, candidates=list(P.CANDIDATES),
                planner_wellness_weight=0.0, judge_pred=jp,
                judge_pred_fixed_motor={k: v for k, v in jp["planner"].items()},
                judge_pred_motor_only=jp["planner"]["1"],
                goal_distance_pred={str(dd): rng.uniform(0.4, 0.6, K).tolist() for dd in P.DEPTHS},
                effort=[0.001] * K, sense_stats=stats,
                route_disagreement={m: [1e-4] * K for m in P.ALL_SENSES},
                motor_token_change=[0.0] + [0.1] * (K - 1),
                actual=dict(wellness_terms=w.tolist(), judge_terms=ja.tolist(),
                            next_pe_sum=rng.uniform(0.05, 0.1, (K, T)).tolist(),
                            goal_distance=rng.uniform(0.4, 0.6, (K, T)).tolist()),
                control=dict(turn_closed_share=rng.uniform(0, 0.5, K).tolist(), closer_share=[0.6] * K,
                             chance_share=[0.5] * K, position_closed_share=[0.2] * K, eye_error_deg=[1.0] * K),
                production=dict(outcome=["tied"], gap_to_best_alt=0.001),
                structural=(dict(production_depth_forced_to_one=True, production_asked_3_returned_length=1,
                                 world_mode_first_order_identical=True, world_mode_deeper_plan_emptied=False)
                            if start == 0 else None)))
    return rows


def verdict_of(rows):
    summary = P.summarise(rows, draws=DRAWS)
    return summary, P.decide(summary)


# ============================================================================ arithmetic
def test_the_five_terms_are_the_approximate_evaluators():
    from lem.model.wellness import APPROXIMATE_TERMS
    assert tuple(P.FIVE_TERMS) == tuple(APPROXIMATE_TERMS)


def test_pairs_and_their_differences():
    assert P.pair_indices(3) == [(0, 1), (0, 2), (1, 2)]
    assert len(P.pair_indices(5)) == 10
    assert P.pair_differences([3.0, 1.0, 2.0]) == [2.0, 1.0, -1.0]


def test_a_forecast_that_carries_a_share_of_the_difference_is_measured_as_that_share():
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 1, (5, 32))
    mean = actual.mean(0)
    for share in (0.0, 0.25, 1.0):
        pred = mean + share * (actual - mean)
        st = P.sense_pair_stats(pred, actual, mean)
        assert len(st["dot"]) == 10 and len(st["err"]) == 5
        got = P.ratio_of_sums(st["dot"], st["real_diff"])
        assert got == pytest.approx(share, abs=1e-12)
        assert P.ratio_of_sums(st["pred_diff"], st["real_diff"]) == pytest.approx(share ** 2, abs=1e-12)
    exact = P.sense_pair_stats(actual, actual, mean)
    assert exact["err"] == [0.0] * 5 and all(p > 0 for p in exact["persist"])


def test_correlation_and_slope_do_not_depend_on_how_pairs_are_oriented():
    x = [0.3, -0.1, 0.2, 0.05]
    y = [0.15, -0.05, 0.1, 0.025]
    assert P.slope_through_origin(y, x) == pytest.approx(0.5)
    assert P.uncentered_corr(x, y) == pytest.approx(1.0)
    flipped = [-v for v in x[:2]] + x[2:], [-v for v in y[:2]] + y[2:]
    assert P.slope_through_origin(flipped[1], flipped[0]) == pytest.approx(0.5)
    assert P.uncentered_corr(*flipped) == pytest.approx(1.0)
    assert P.slope_through_origin([1.0], [0.0]) is None and P.uncentered_corr([0.0], [1.0]) is None
    assert P.sign_agreement([1, -1, 0], [2, 1, 3]) == 0.5


def test_the_bootstrap_resamples_whole_bodies():
    groups = {b: [b] * 3 for b in range(8)}
    got = P.cluster_bootstrap(groups, lambda items: float(len(set(items))), draws=300)
    assert got["value"] == 8.0 and got["bodies"] == 8
    lo, hi = got["interval"]
    assert hi <= 8.0 and lo < 8.0          # a resample of bodies repeats some and drops others
    means = P.body_mean_interval({b: [1.0 + 0.1 * b] for b in range(8)}, draws=300)
    assert means["bodies"] == 8 and means["bodies_above_zero"] == 8
    assert means["interval"][0] > 0 and means["mean"] == pytest.approx(1.35)


def test_selection_and_ranks_never_read_a_tie_as_the_chosen_order_being_best():
    assert P.select([1.0, 3.0, 3.0]) == 1                    # an exact tie goes to the lower index
    assert P.select([1.0, 3.0, 0.5], higher_is_better=False) == 2
    assert P.strict_rank([2.0, 2.0, 2.0, 2.0, 2.0], 0) == 1
    assert P.mid_rank([2.0, 2.0, 2.0, 2.0, 2.0], 0) == 3.0   # the chance rank of 5
    assert P.mid_rank([5.0, 1.0, 2.0], 0) == 1.0
    assert P.mid_rank([0.0, 1.0, 2.0], 0) == 3.0
    assert P.mid_rank([0.0, 1.0, 2.0], 0, higher_is_better=False) == 1.0


def test_the_tie_rule_is_the_production_one():
    from mimo_bridge.causal_chain import outcome, TIE_BAND
    for gap in (-0.2, -TIE_BAND, -0.01, 0.0, 0.01, TIE_BAND, 0.2):
        assert P.tie_outcome(gap, TIE_BAND) == outcome(gap, TIE_BAND)
    assert P.PRODUCTION_TIE_BAND == TIE_BAND


def test_hold_is_formed_as_the_production_record_forms_it():
    H, gi = 6, 4
    plan0 = torch.rand(1, H, 9)
    achieved = torch.rand(1, H, 10)
    exec_prev = torch.rand(1, H, 9)
    hold = P.hold_candidate(plan0, achieved, exec_prev, gi)
    assert torch.equal(hold[..., 0:3], achieved[..., 0:3])
    q = achieved[..., 3:7] / achieved[..., 3:7].norm(dim=-1, keepdim=True)
    body = [i for i in range(H) if i != gi]
    assert torch.allclose(hold[:, body, 3:7], q[:, body])
    assert torch.equal(hold[:, gi, 3:7], exec_prev[:, gi, 3:7])
    assert torch.equal(hold[..., 7:9], plan0[..., 7:9])
    # without the executed ledger the eyes keep the plan's row, like the production record
    alone = P.hold_candidate(plan0, achieved, None, gi)
    assert torch.equal(alone[:, gi], plan0[:, gi])
    assert P.perturb_seed(0, 344400) == 344400 and P.perturb_seed(3, 5) == 3 * 1_000_003 + 5


# ============================================================================ the verdict
def test_a_forecast_that_ignores_the_order_is_named_flat_and_the_evaluator_is_not():
    summary, v = verdict_of(synthetic_rows(forecast=0.0, evaluator=1.0))
    assert v["forward_model_flat"] is True and v["evaluator_flat"] is False
    assert v["first_flat_stage_along_the_route"] == "forward model"
    assert v["forward_model_transfer"] == pytest.approx(0.0, abs=1e-9)
    assert v["evaluator_transfer"] == pytest.approx(1.0, abs=0.01)
    # every predicted difference is zero, so no pair is decided, while true outcomes would be
    wd = summary["wellness_differences"]
    assert wd["decided_pairs_production"]["value"] == 0.0
    # exact predicted ties read as the chance rank, never as the chosen order being best
    assert summary["planner"]["chosen_rank_by_predicted_wellness"]["value"] == pytest.approx(3.0)
    assert v["planner_ignores_predicted_wellness"] is True
    # the band's error splits into the forward model's part and the evaluator's part
    dec = summary["tie_band"]["decomposition"]
    assert dec["forward_model_share"] > 0.99 and dec["evaluator_share"] < 0.01


def test_an_evaluator_blind_to_true_outcomes_is_named_flat_and_the_forecast_is_not():
    summary, v = verdict_of(synthetic_rows(forecast=1.0, evaluator=0.0, noise=0.02, seed=3))
    assert v["evaluator_flat"] is True and v["forward_model_flat"] is False
    assert v["forward_model_transfer"] == pytest.approx(1.0, abs=1e-3)
    assert v["forward_model_captured_share"] == pytest.approx(1.0, abs=1e-3)
    lo, hi = summary["evaluator"]["transfer"]["interval"]
    assert lo <= 0.0 <= hi
    assert summary["tie_band"]["decomposition"]["evaluator_share"] > 0.99


def test_a_working_chain_leaves_only_the_planner_which_never_reads_predicted_wellness():
    summary, v = verdict_of(synthetic_rows(forecast=1.0, evaluator=1.0))
    assert not v["forward_model_flat"] and not v["evaluator_flat"]
    assert v["flat_stages"] == ["planner"]
    assert summary["chain"]["transfer"]["value"] == pytest.approx(1.0, abs=0.01)
    assert summary["tie_band"]["remeasured_planner_route"] < 0.01


def test_looking_further_ahead_is_credited_only_when_it_picks_a_physically_better_order():
    summary, v = verdict_of(synthetic_rows(reverse_first_moment=True))
    block = summary["depth"]["wellness"]
    assert block["differs_share"] == 1.0
    assert v["depth3_better_physical_decision"] is True
    assert block["wellness_tenure_depth3_minus_depth1"]["bodies_above_zero"] == 8
    # when the first moment already ranks the candidates as the tenure does, depth changes nothing
    summary, v = verdict_of(synthetic_rows(reverse_first_moment=False))
    assert summary["depth"]["wellness"]["differs_share"] == 0.0
    assert summary["depth"]["wellness"]["wellness_tenure_depth3_minus_depth1"]["interval"] == [0.0, 0.0]
    assert v["depth3_better_physical_decision"] is False


# ============================================================================ the model-free helpers
def test_controllability_reads_arrival_as_one_and_stillness_as_zero():
    from mimo_bridge.body_layout import names_for
    from mimo_bridge.action_decoder import POSITION_HANDLES
    from lem.model.motor_synergy import quat_mul, rotvec_to_quat
    names = names_for("trunk_v1")

    class Model:
        handle_names = names
        GAZE_IDX = names.index("gaze")
    H = len(names)
    ach0 = torch.zeros(1, H, 10, dtype=torch.float64)
    ach0[..., 6] = 1.0
    ach0[..., 0:3] = torch.linspace(-0.1, 0.1, H * 3, dtype=torch.float64).reshape(1, H, 3)
    turn = rotvec_to_quat(torch.tensor([0.0, 0.0, math.radians(20.0)], dtype=torch.float64))
    order = torch.zeros(1, H, 9, dtype=torch.float64)
    order[..., 3:7] = quat_mul(turn.expand(1, H, 4), ach0[..., 3:7])
    order[..., 0:3] = ach0[..., 0:3] * 200.0 + 2.0          # every position target 2 cm along each axis
    order[..., 8] = 0.5
    arrived = ach0.clone()
    arrived[..., 3:7] = order[..., 3:7]
    arrived[..., 0:3] = order[..., 0:3] / 200.0
    got = P.control_of(Model, order, ach0, arrived, [0.1, 0.0], [0.1, 0.0])
    assert got["turn_closed_share"] == pytest.approx(1.0, abs=1e-6)
    assert got["position_closed_share"] == pytest.approx(1.0, abs=1e-9)
    assert got["closer_share"] == 1.0 and got["handles"] == H - 1 and got["eye_error_deg"] == 0.0
    still = P.control_of(Model, order, ach0, ach0, [0.1, 0.0], [0.0, 0.0])
    assert still["turn_closed_share"] == pytest.approx(0.0, abs=1e-9)
    assert still["position_closed_share"] == pytest.approx(0.0, abs=1e-12)
    assert still["eye_error_deg"] == pytest.approx(math.degrees(0.1))
    assert any(n in POSITION_HANDLES for n in names)


class _Proc:
    def __init__(self, command, created, pid, parents=()):
        self._command, self._created, self.pid, self._parents = command, created, pid, list(parents)

    def cmdline(self):
        return self._command

    def create_time(self):
        return self._created

    def parents(self):
        return self._parents


def test_a_probe_launched_by_a_scheduler_older_than_its_validator_refuses_before_loading(tmp_path):
    module = tmp_path / "Source/Python/mimo_bridge/tools/ability_rechecks.py"
    module.parent.mkdir(parents=True)
    module.write_text("# validator", encoding="utf-8")
    changed = module.stat().st_mtime
    scheduler = ["python", "-u", "-m", "mimo_bridge.tools.ability_rechecks", "--root", "r"]
    launcher = _Proc(["python", "-u", "probe_planner_candidates.py"], changed + 5, 11)
    old = _Proc([], 0, 10, parents=[launcher, _Proc(scheduler, changed - 60, 12)])
    assert P.stale_scheduler(tmp_path, old) == (True, 12)
    new = _Proc([], 0, 10, parents=[launcher, _Proc(scheduler, changed + 60, 13)])
    assert P.stale_scheduler(tmp_path, new) == (False, 13)
    by_hand = _Proc([], 0, 10, parents=[_Proc(["bash"], 0, 14)])
    assert P.stale_scheduler(tmp_path, by_hand) == (False, None)


def test_a_reusable_job_entry_declares_into_a_fresh_folder_before_loading_anything(tmp_path, monkeypatch):
    import hashlib
    from mimo_bridge.tools import commit_guard
    from mimo_bridge.tools import gof_probe_harness as H

    class Stop(Exception):
        pass

    def refuse(*_a, **_k):
        raise Stop()
    monkeypatch.setattr(commit_guard, "commit_reading", lambda: (1.0, 100.0, 50.0))
    monkeypatch.setattr(H, "load_frozen", refuse)
    ckpt = tmp_path / "run_x" / "checkpoints" / "ckpt_step_7"
    ckpt.mkdir(parents=True)
    with pytest.raises(Stop):
        P.main(["--checkpoint", str(ckpt), "--out-root", str(tmp_path / "out")])
    made = list((tmp_path / "out" / "run_x").glob("ckpt_step_7_*/declaration.json"))
    assert len(made) == 1
    d = json.loads(made[0].read_text(encoding="utf-8"))
    assert d["learning"] is False and d["probe"] == P.PROBE_VERSION
    assert d["script_sha256"] == hashlib.sha256(PROBE_PATH.read_bytes()).hexdigest()
    with pytest.raises(Stop):                                    # the scheduler's form
        P.main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "given")])
    assert (tmp_path / "given" / "declaration.json").is_file()
    with pytest.raises(SystemExit):
        P.main(["--checkpoint", str(ckpt)])                        # neither form
    monkeypatch.setattr(commit_guard, "commit_reading", lambda: (99.0, 100.0, 1.0))
    with pytest.raises(RuntimeError, match="Memory guard refused"):
        P.main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "refused")])
    assert not (tmp_path / "refused").exists(), "a refused run must not leave a folder behind"


def test_results_are_json_clean():
    value = {"a": np.float32(0.5), "b": [float("nan"), np.int64(3)], "c": np.arange(3)}
    assert P.clean(value) == {"a": 0.5, "b": [None, 3], "c": [0, 1, 2]}
    json.dumps(P.clean(value), allow_nan=False)


# ============================================================================ scheduler contract
def _complete_payload(tmp_path):
    rows = synthetic_rows(reverse_first_moment=True)
    summary = P.summarise(rows, draws=DRAWS)
    verdict = P.decide(summary)
    payload = dict(probe=P.PROBE_VERSION, checkpoint=str(tmp_path / "checkpoint"), script_sha256="expected",
                   learning=False, elapsed_s=12.5, bodies=8, starts_per_body=2, evaluated_starts=16,
                   candidates=list(P.CANDIDATES), depths=list(P.DEPTHS), tenure_steps=5,
                   weights_unchanged=True, exact_replay=True, all_candidate_traces_replayed_exactly=True,
                   observation_replay_checks=8, verdict=verdict, passed=verdict["depth3_better_physical_decision"],
                   route_checks_max=dict(record_pred_wellness_max_abs=1e-6, record_goal_distance_max_abs=2e-6,
                                         alternatives_execution_max_abs_cm=0.0, executed_pose_depth1_max_abs=0.0),
                   depth_pose=P.DEPTH_POSE, summary=summary)
    return json.loads(json.dumps(P.clean(payload), allow_nan=False))


def test_the_scheduler_accepts_a_complete_result_and_refuses_an_incomplete_one(tmp_path):
    from mimo_bridge.tools import ability_rechecks as worker
    payload = _complete_payload(tmp_path)
    record = dict(checkpoint=payload["checkpoint"], script_sha256="expected",
                  command=["python", "-u", str(PROBE_PATH)])
    worker.validate_result(payload, record)
    bad_values = [("script_sha256", "another"), ("weights_unchanged", False), ("evaluated_starts", 15),
                  ("all_candidate_traces_replayed_exactly", False), ("observation_replay_checks", 7),
                  ("passed", not payload["passed"]), ("learning", True), ("tenure_steps", 2)]
    for key, value in bad_values:
        bad = copy.deepcopy(payload)
        bad[key] = value
        with pytest.raises(ValueError):
            worker.validate_result(bad, record)
    bad = copy.deepcopy(payload)
    bad["route_checks_max"]["record_pred_wellness_max_abs"] = 0.01
    with pytest.raises(ValueError, match="production planning record"):
        worker.validate_result(bad, record)
    bad = copy.deepcopy(payload)
    bad["summary"]["evaluator"]["transfer"]["interval"] = None
    with pytest.raises(ValueError, match="uncertainty"):
        worker.validate_result(bad, record)
    bad = copy.deepcopy(payload)
    del bad["verdict"]["evaluator_flat"]
    with pytest.raises(ValueError, match="verdict"):
        worker.validate_result(bad, record)


def test_the_manifest_declares_the_probe_on_its_own_script():
    manifest = json.loads((REPO / "docs/gain_of_function_v3/ability_rechecks.json").read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in manifest["tasks"]}
    task = tasks["planner_candidates"]
    assert task["script"] == "docs/gain_of_function_v4/probe_planner_candidates.py"
    assert (REPO / task["script"]).is_file()
    assert task["every_steps"] > 0 and task["first_step"] > 0 and task["memory_gib"] >= P.NEED_GIB
    from mimo_bridge.tools.ability_rechecks import confined
    assert confined(REPO, task["script"]) == PROBE_PATH.resolve()


# ============================================================================ the ranking (planner_candidates_v2)
def ranked_rows(forecast="exact", chosen_better=True, seed=7):
    """synthetic_rows with known goal distances and wellness: every candidate's actual distance to the
    goal and grounded wellness held for the whole tenure; the forecast (goal distance 50 ms ahead and
    the planner route's predicted wellness) the actual values ('exact'), their reverse at twice their
    spread ('reversed') or the same value for every candidate ('flat': the forecast ignores the order);
    the chosen order ends nearer the goal and better-felt than the controller's proposal ('reflex') in
    every start when chosen_better."""
    rng = np.random.default_rng(seed)
    rows = copy.deepcopy(synthetic_rows())
    for r in rows:
        K, T = len(r["candidates"]), len(r["actual"]["goal_distance"][0])
        goal, well = rng.uniform(0.2, 0.6, K), rng.uniform(2.5, 3.5, K)
        gap = 0.05 if chosen_better else -0.05
        goal[0], well[0] = goal[2] - gap, well[2] + gap
        r["actual"]["goal_distance"] = [[float(goal[c])] * T for c in range(K)]
        r["actual"]["wellness_terms"] = [[[float(well[c]) / 5.0] * 5 for _ in range(T)] for c in range(K)]
        forecast_goal = {"exact": goal, "reversed": 1.0 - 2.0 * goal, "flat": np.full(K, 0.4)}[forecast]
        forecast_well = {"exact": well, "reversed": 12.0 - 2.0 * well, "flat": np.full(K, 3.0)}[forecast]
        r["goal_distance_pred"]["1"] = [float(x) for x in forecast_goal]
        r["judge_pred"]["planner"]["1"] = [[float(x) / 5.0] * 5 for x in forecast_well]
    return rows


GOAL_OUTCOMES = ("goal_distance_50ms", "goal_distance_end")
WELLNESS_OUTCOMES = ("wellness_50ms", "wellness_tenure")


def test_every_outcome_carries_the_pairwise_ranking_with_its_interval_spread_and_case():
    ranking = P.summarise(ranked_rows("exact"), draws=DRAWS)["ranking"]
    assert set(P.RANKING_OUTCOMES) <= set(ranking) and ranking["reference"] == "reflex"
    assert ranking["wellness_searched"] is False and "planner_comparison" in ranking["notes"]
    for name in P.RANKING_OUTCOMES:
        block = ranking[name]
        assert block["per_100"] == 100.0 and block["interval"] == [100.0, 100.0], name
        # 16 starts x 10 pairs; a pair whose two draws land within 1 in 100 of the spread is a tie
        assert block["pairs_judged"] + block["actual_ties_left_out"] == 160 and block["pairs_judged"] >= 150
        assert block["bodies"] == 8 and block["spread_ratio"] == pytest.approx(1.0)
        # the same statistic over the four candidates the planner did not choose: 6 pairs a start
        rest = block["without_chosen"]
        assert rest["per_100"] == 100.0 and rest["pairs_judged"] + rest["actual_ties_left_out"] == 96
        assert rest["planners"] == {} and rest["case"] == "better_than_chance"
    for name in GOAL_OUTCOMES:
        assert ranking[name]["planners"]["chosen"]["bodies_better"] == 8 and ranking[name]["case"] == "ranking_used"
    # the planner's objective gives wellness no weight: a wellness ranking cannot fault its search
    for name in WELLNESS_OUTCOMES:
        assert ranking[name]["planners"] == {} and ranking[name]["case"] == "better_than_chance", name


def test_a_ranking_the_chosen_order_does_not_turn_into_a_better_outcome_is_case_3():
    ranking = P.summarise(ranked_rows("exact", chosen_better=False), draws=DRAWS)["ranking"]
    for name in GOAL_OUTCOMES:
        assert ranking[name]["per_100"] == 100.0 and ranking[name]["case"] == "case_3_planner_not_better", name
    for name in WELLNESS_OUTCOMES:
        assert ranking[name]["case"] == "better_than_chance", name
    # once the objective weighs predicted wellness, the wellness rankings judge the search too
    searched = ranked_rows("exact", chosen_better=False)
    for row in searched:
        row["planner_wellness_weight"] = 0.5
    ranking = P.summarise(searched, draws=DRAWS)["ranking"]
    assert ranking["wellness_searched"] is True
    for name in WELLNESS_OUTCOMES:
        assert ranking[name]["case"] == "case_3_planner_not_better", name


def test_a_backwards_forecast_is_case_2_and_a_flat_one_case_1():
    backwards = P.summarise(ranked_rows("reversed"), draws=DRAWS)["ranking"]
    flat = P.summarise(ranked_rows("flat"), draws=DRAWS)["ranking"]
    for name in P.RANKING_OUTCOMES:
        assert backwards[name]["per_100"] == 0.0 and backwards[name]["below_chance"], name
        assert backwards[name]["case"] == "case_2_wrong_order", name
        # every pair a forecast tie: half right, 50 in 100 with no spread at all
        assert flat[name]["per_100"] == 50.0 and flat[name]["forecast_ties_counted_half"] == flat[name]["pairs_judged"]
        assert flat[name]["spread_ratio"] == 0.0 and flat[name]["case"] == "case_1_under_separated", name


def test_the_verdict_does_not_read_the_ranking():
    summary = P.summarise(synthetic_rows(reverse_first_moment=True), draws=DRAWS)
    without = {k: v for k, v in summary.items() if k != "ranking"}
    assert P.decide(summary) == P.decide(without)


def test_the_ranking_outcomes_are_the_ones_the_scheduler_requires_and_are_declared_first(tmp_path, monkeypatch):
    from mimo_bridge.tools import ability_rechecks as worker, commit_guard
    from mimo_bridge.tools import gof_probe_harness as H
    assert tuple(P.RANKING_OUTCOMES) == worker.PLANNER_RANKING_OUTCOMES
    assert P.PROBE_VERSION == "planner_candidates_v3"

    class Stop(Exception):
        pass
    monkeypatch.setattr(commit_guard, "commit_reading", lambda: (1.0, 100.0, 50.0))
    monkeypatch.setattr(H, "load_frozen", lambda *a, **k: (_ for _ in ()).throw(Stop()))
    ckpt = tmp_path / "run_x" / "checkpoints" / "ckpt_step_7"
    ckpt.mkdir(parents=True)
    with pytest.raises(Stop):
        P.main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "given")])
    declared = json.loads((tmp_path / "given" / "declaration.json").read_text(encoding="utf-8"))["ranking_rules"]
    assert declared["outcomes"] == P.RANKING_OUTCOMES and declared["reference"] == "reflex" and "case" in declared
    assert declared["notes"] == P.RANKING_NOTES


def test_the_scheduler_requires_the_ranking_of_every_outcome(tmp_path):
    from mimo_bridge.tools import ability_rechecks as worker
    payload = _complete_payload(tmp_path)
    record = dict(checkpoint=payload["checkpoint"], script_sha256="expected", command=["python", "-u", str(PROBE_PATH)])
    worker.validate_result(payload, record)
    for damage in ("missing", "interval", "value", "case"):
        bad = copy.deepcopy(payload)
        if damage == "missing":
            del bad["summary"]["ranking"]["wellness_tenure"]
        elif damage == "interval":
            bad["summary"]["ranking"]["goal_distance_end"]["interval"] = None
        elif damage == "value":
            bad["summary"]["ranking"]["goal_distance_50ms"]["per_100"] = None
        else:
            bad["summary"]["ranking"]["wellness_50ms"]["case"] = "undetermined"
        with pytest.raises(ValueError, match="ranking"):
            worker.validate_result(bad, record)
    older = dict(copy.deepcopy(payload), probe="planner_candidates_v1")
    with pytest.raises(ValueError, match="version"):
        worker.validate_result(older, record)


# ============================================================================ version 3 (2026-10-02)
def _rounding_level_tokens(rng, K=5, d=192, steps=(0, 3, 7, 12, 19), scale=1.0):
    """K float32 tokens equal to one constant token except for a few float32 grid steps per number (candidate k
    moved `steps[k]` grid steps in a random direction on every number): the vestibular token of the saved rechecks,
    whose real differences between candidates were 2 to 19 grid steps per number."""
    base = (rng.normal(0.0, 1.0, d) * scale).astype(np.float32)
    out = []
    for k in range(K):
        t = base.copy()
        direction = np.where(rng.random(d) < 0.5, -np.inf, np.inf).astype(np.float32)
        for _ in range(int(steps[k])):
            t = np.nextafter(t, direction).astype(np.float32)
        out.append(t)
    return np.stack(out)


def test_the_floor_is_d_float32_grid_steps_per_number_at_the_tokens_own_size():
    one = np.ones(192, np.float32)
    # at 1.0 the float32 grid step is 2^-23 (numpy.spacing), so the floor is (192 x 2^-23)^2
    assert P.float32_floor(one, one) == pytest.approx((192 * 2.0 ** -23) ** 2, rel=1e-12)
    # the floor follows the token's own magnitude: 1,024 times larger numbers, a 1,024 times larger grid step
    assert P.float32_floor(one * 1024, one * 1024) == pytest.approx(P.float32_floor(one, one) * 1024 ** 2, rel=1e-12)
    st = P.sense_pair_stats(np.stack([one, one * 2]), np.stack([one, one * 2]), one)
    assert len(st["floor"]) == 1 and st["floor"][0] > 0


def test_a_sense_whose_real_difference_is_within_its_float32_floor_reads_not_measurable():
    """PLANTED DEFECT: a token that differs between candidates only by float32 rounding (the vestibular token)
    gave a 'captured share' that is the ratio of two rounding residues (-55 of 100 at 1 model step in the saved
    rechecks). It must read not measurable, at any token magnitude, while a sense whose real difference is far
    above its floor keeps its exact share."""
    rng = np.random.default_rng(11)
    rows = copy.deepcopy(synthetic_rows())
    for r in rows:
        for route in ("planner", "learning"):
            for dd in P.DEPTHS:
                for scale in (1.0, 1024.0):
                    actual = _rounding_level_tokens(rng, scale=scale)
                    # a forecast whose differences are large and unrelated: the old ratio of sums is pure noise
                    pred = actual.astype(np.float64) + rng.normal(0.0, 1e-3, actual.shape)
                    st = P.sense_pair_stats(pred, actual, actual[0])
                    assert sum(st["real_diff"]) <= sum(st["floor"])
                    if scale == 1.0:
                        r["sense_stats"][route][str(dd)]["vestibular"] = st
                    else:
                        r["sense_stats"][route][str(dd)]["audio"] = st
    summary = P.summarise(rows, draws=DRAWS)
    for route in ("planner", "learning"):
        for dd in P.DEPTHS:
            for m in ("vestibular", "audio"):
                block = summary["forward_model"]["senses"][route][str(dd)][m]
                assert block["measurable"] is False and block["captured_share"]["measurable"] is False, (m, dd)
                assert block["captured_share"]["value"] is None and block["captured_share"]["interval"] is None
                assert block["captured_share"]["reason"].startswith("not measurable")
                assert block["predicted_over_realised_difference"] is None
                assert block["real_difference_over_noise_floor"] < 1
            # the senses with real differences far above their floor keep the share planted in synthetic_rows (1)
            for m in ("vision", "tactile", "proprioception"):
                block = summary["forward_model"]["senses"][route][str(dd)][m]
                assert block["measurable"] is True and block["real_difference_over_noise_floor"] > 1e6
                assert block["captured_share"]["value"] == pytest.approx(1.0, abs=1e-3)
    # the pooled Judge B senses include audio here, whose rounding residue adds nothing measurable to the pool
    pooled = summary["forward_model"]["senses"]["planner"]["1"]["judge_senses"]
    assert pooled["measurable"] is True and pooled["captured_share"]["value"] == pytest.approx(1.0, abs=1e-3)
    verdict = P.decide(summary)
    assert verdict["forward_model_captured_share_measurable"] is True


def test_rows_written_before_version_3_carry_no_floor_and_are_summarised_as_then():
    rows = copy.deepcopy(synthetic_rows())
    for r in rows:
        for route in r["sense_stats"].values():
            for depth in route.values():
                for st in depth.values():
                    st.pop("floor", None)
    block = P.summarise(rows, draws=DRAWS)["forward_model"]["senses"]["planner"]["1"]["vision"]
    assert block["measurable"] is None and block["real_difference_over_noise_floor"] is None
    assert block["captured_share"]["value"] == pytest.approx(1.0, abs=1e-3)


def test_depths_two_and_three_read_each_candidates_executed_pose(monkeypatch):
    """PLANTED DEFECT: up to version 2 every depth's transition read the decision's achieved pose, so depths 2 and
    3 predicted from a pose 1 and 2 model steps old. With executed_poses, depth 1 reads the decision's pose and
    depth d > 1 reads each candidate's own pose after d - 1 executed model steps; without them (the old route) the
    deeper depths read the decision's pose, which this test tells apart."""
    from types import SimpleNamespace
    import lem.model.predictive_state as ps
    calls = []

    def fake_transition(fm, reg, heads, state, action, duration_s=0.05, achieved_state=None,
                        advance_context=True, include_latent=False):
        used = achieved_state if achieved_state is not None else state.achieved
        calls.append(used.clone())
        return ps.PredictiveState(tokens={m: v + 1.0 for m, v in state.tokens.items()}, context=state.context,
                                  achieved=used, domain="predicted", memory_bias=state.memory_bias)
    monkeypatch.setattr(ps, "transition", fake_transition)
    K, H, F, T = 5, 3, 10, 5
    decision_pose = torch.full((1, H, F), -1.0)
    st0 = ps.PredictiveState(tokens={"proprioception": torch.zeros(1, 4), "motor": torch.zeros(1, 4)},
                             achieved=decision_pose, domain="real")
    model = SimpleNamespace(forward_model=None, encoder_registry=None, prediction_heads=None,
                            wellness=SimpleNamespace(score_predicted_wellness=lambda s, z: torch.zeros(z.shape[0], 5)))
    obs = [[dict(achieved=torch.full((1, H, F), 10.0 * c + k + 1)) for k in range(T)] for c in range(K)]
    poses = P.executed_poses(obs)
    assert sorted(poses) == [2, 3] and tuple(poses[2].shape) == (K, H, F)
    goal = {"proprioception": torch.zeros(1, 4)}
    P.predictions(model, st0, torch.zeros(K, H, 9), K, goal, ["proprioception"], poses=poses)
    assert len(calls) == 3
    assert torch.equal(calls[0], decision_pose.repeat(K, 1, 1))                      # depth 1: the decision's pose
    assert torch.equal(calls[1], torch.cat([obs[c][0]["achieved"] for c in range(K)]))   # after 1 model step
    assert torch.equal(calls[2], torch.cat([obs[c][1]["achieved"] for c in range(K)]))   # after 2 model steps
    calls.clear()
    P.predictions(model, st0, torch.zeros(K, H, 9), K, goal, ["proprioception"])        # the route of v1 and v2
    assert all(torch.equal(c, decision_pose.repeat(K, 1, 1)) for c in calls)
    assert not torch.equal(calls[1], poses[2])


def test_every_row_status_and_summary_carries_the_script_hash(tmp_path, monkeypatch):
    import hashlib
    digest = hashlib.sha256(PROBE_PATH.read_bytes()).hexdigest()
    assert P.script_sha256() == digest
    rows = copy.deepcopy(synthetic_rows())
    summary = P.summarise(rows, draws=50)
    assert summary["summarised_by_script_sha256"] == digest and summary["probe"] == P.PROBE_VERSION
    assert summary["rows_measured_by_script_sha256"] is None            # rows written before v3 carry none
    rows[0]["script_sha256"], rows[1]["script_sha256"] = "a" * 64, "b" * 64
    assert P.summarise(rows, draws=50)["rows_measured_by_script_sha256"] == ["a" * 64, "b" * 64]
    from mimo_bridge.tools import commit_guard
    from mimo_bridge.tools import gof_probe_harness as H

    class Stop(Exception):
        pass
    monkeypatch.setattr(commit_guard, "commit_reading", lambda: (1.0, 100.0, 50.0))
    monkeypatch.setattr(H, "load_frozen", lambda *a, **k: (_ for _ in ()).throw(Stop()))
    ckpt = tmp_path / "run_x" / "checkpoints" / "ckpt_step_7"
    ckpt.mkdir(parents=True)
    with pytest.raises(Stop):
        P.main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "given")])
    declared = json.loads((tmp_path / "given" / "declaration.json").read_text(encoding="utf-8"))
    assert declared["script_sha256"] == digest and declared["depth_pose"] == "executed"
    assert declared["noise_floor_rule"] == P.NOISE_FLOOR_RULE and declared["depth_pose_rule"] == P.DEPTH_POSE_RULE


def test_the_scheduler_requires_the_executed_pose_route_of_version_3(tmp_path):
    from mimo_bridge.tools import ability_rechecks as worker
    payload = _complete_payload(tmp_path)
    record = dict(checkpoint=payload["checkpoint"], script_sha256="expected", command=["python", "-u", str(PROBE_PATH)])
    worker.validate_result(payload, record)
    for damage in ("version", "pose", "check", "moved"):
        bad = copy.deepcopy(payload)
        if damage == "version":
            bad["probe"] = "planner_candidates_v2"
        elif damage == "pose":
            bad["depth_pose"] = "decision"
        elif damage == "check":
            del bad["route_checks_max"]["executed_pose_depth1_max_abs"]
        else:
            bad["route_checks_max"]["executed_pose_depth1_max_abs"] = 1e-3
        with pytest.raises(ValueError):
            worker.validate_result(bad, record)


# ============================================================================ the production route
LIVE_ARGS = sorted((REPO / "Saved" / "mimo_bridge" / "experiments").glob(
    "H84_fresh_drivers_20260918_seg39*/args.json"), key=lambda p: p.stat().st_mtime)


@pytest.mark.slow
@pytest.mark.skipif(not LIVE_ARGS, reason="no H84 segment-39 args.json on this machine")
def test_one_saved_decision_is_measured_end_to_end_on_a_small_build_of_the_production_flags(tmp_path):
    """The production run's own flags (wrist-relative hand targets, hand goals from memory, the
    normalised ranking, efference and contact heads, persistent memory), built narrow and untrained
    through the trainer's builder, in a fresh one-body world: a planning event is found, snapshotted
    and measured by measure_start - the production-record route check, the depth checks, five
    executed and physically replayed candidates, the observation replay - and the row summarises and
    serialises. Nothing is loaded from a checkpoint and no weight moves."""
    from mimo_bridge.tools import causal_test as CT
    from mimo_bridge.tools import gof_probe_harness as H
    from mimo_bridge import train_mimo as TM
    torch.set_num_threads(2)
    saved = json.loads(LIVE_ARGS[-1].read_text(encoding="utf-8"))
    small = dict(d_model=64, core_blocks=1, vision_swin_layers=1, vision_width=64, encoder_layers=1,
                 encoder_width=64, context_length=16, caregiver_record_dir="")
    ns = CT._probe_configuration(saved, "cpu", 0, True, small)
    assert ns.hand_target_mode == "wrist_relative_v1" and ns.intention_every == 10
    model = TM.build_learner(ns, None)
    frozen = H.Frozen(model=model, args=ns, info={}, limits=dict(phase_name="mature", strength_limit=1.0,
                      rom_limit=1.0, age_months=24.0), limits_source="test", load={"reinitialized": []},
                      build_s=0.0, load_s=0.0)
    session = H.Session(frozen, seed=3, babble=float(ns.babble_noise), run_dir=str(tmp_path / "scene"))
    try:
        session.warm(22)
        snap, _ = P.next_decision(session, 0)
        before = P.weights_digest(model)
        for parameter in model.params:
            parameter.requires_grad_(False)
        row = P.measure_start(session, snap, 0, 0, int(session.hold), tmp_path, checks=True)
        assert P.weights_digest(model) == before, "measuring a decision changed a weight"
    finally:
        session.close()
    assert row["exact_replay"]["chosen"] == {"physical": True, "observations": True}
    assert all(row["exact_replay"][n]["physical"] for n in P.CANDIDATES)
    assert len(row["actual"]["wellness_terms"]) == 5 and len(row["actual"]["wellness_terms"][0]) == 5
    assert row["production"]["outcome"] and row["production"]["outcome"][0] in ("best", "tied", "behind")
    st = row["structural"]
    assert st["production_depth_forced_to_one"] is True and st["production_asked_3_returned_length"] <= 1
    if st["world_mode_plan_length"]["1"] and st["world_mode_plan_length"]["3"]:
        assert st["world_mode_first_order_identical"] is True
    assert set(row["sense_stats"]["planner"]["1"]) >= {"vision", "tactile", "proprioception"}
    summary = P.summarise([row], draws=50)
    verdict = P.decide(summary)
    json.dumps(P.clean(dict(row=row, summary=summary, verdict=verdict)), allow_nan=False)
    assert (tmp_path / "start_0_0.npz").is_file()


@pytest.mark.slow
def test_the_probe_reproduces_the_production_planning_record_on_a_small_real_model(tmp_path):
    """At a real planning event of a small model and body: the probe's five candidates, their
    execution form, their predicted wellness and goal distance on the planner's route equal the
    production record's; the production chain's first order does not depend on the depth asked."""
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite, command_for_encoder
    from lem.model.predictive_state import PredictiveState
    torch.set_num_threads(1)
    model = LemLite(device="cpu", seed=0, intention=True, intention_every=5, imagination=False,
                    no_vision=True, next_frame_loss=True, smooth_actions=0.35, causal_chain=True,
                    overrides={"training": {"steps_per_phase": 3000}})
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "pc"), tag="_pc", vision=False, seed=5)
    found = None
    try:
        frames = src.reset()
        dev = dict(model.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        prev, held = None, None
        for i in range(31):
            if i in (20, 25, 30):
                # pin an earlier goal so the present moment cannot win and leave no plan (the pin
                # test_causal_chain uses for the same reason)
                remembered, value = model.goal_ring[0]
                model.goal_ring[0] = (remembered, torch.full_like(value, 1e6))
            res = model.step(frames, dev, train=False, step_index=i, executed_action=prev)
            rec = model.last_plan_record
            pi = model._last_plan_inputs
            if (rec is not None and rec.get("step") == i and "error" not in rec
                    and pi is not None and pi.get("plan0") is not None):
                found = (rec, pi, prev)
                break
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        assert found is not None, "no planning record in the small run"
        rec, pi, prev = found
        decision = pi["decision"]
        exec_prev = None if prev is None else torch.as_tensor(np.asarray(prev, np.float32))
        raws = P.candidate_raws(model, pi["plan0"], pi["motor"], decision["achieved"], exec_prev, rec["step"])
        names = list(P.CANDIDATES)
        execs = {n: P.executed_candidate(model, raws[n], decision) for n in names}
        ref = model._causal_execution_candidates([raws[n] for n in names[1:]], decision)
        assert float((ref - torch.cat([execs[n][0] for n in names[1:]])).abs().max()) <= 1e-5
        enc = command_for_encoder(torch.cat([execs[n][0] for n in names]).float())
        goal = {m: v.float() for m, v in pi["goal"].items()}
        senses = P.goal_senses(decision, goal, senses=[m for m in pi["cur"] if m != "motor"])
        st = PredictiveState(tokens={m: v.float().reshape(1, -1) for m, v in pi["cur"].items()},
                             context=decision["context"], achieved=decision["achieved"], domain="real",
                             memory_bias=decision.get("memory_bias"))
        pred = P.predictions(model, st, enc, len(names), goal, senses)
        sums = pred[1]["judge"].sum(-1).tolist()
        dists = pred[1]["goal_distance"].tolist()
        for i, n in enumerate(names):
            assert abs(sums[i] - rec["pred_wellness"][n][0]) <= 1e-4, (n, sums[i], rec["pred_wellness"][n])
            assert abs(dists[i] - rec["goal_distance"][n][0]) <= 1e-4, (n, dists[i], rec["goal_distance"][n])
        before = [p.detach().clone() for p in model.params]
        out = P.structural_checks(model, pi)
        assert out["production_depth_forced_to_one"] is False     # the small body runs world targets
        assert out["production_asked_3_returned_length"] >= 1
        if out["world_mode_plan_length"]["1"] and out["world_mode_plan_length"]["3"]:
            assert out["world_mode_first_order_identical"] is True
        assert all(torch.equal(a, b) for a, b in zip(before, model.params)), "a check changed a weight"
    finally:
        src.close()
