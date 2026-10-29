# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The latent-versus-planner comparison's calculations (mimo_bridge/tools/latent_body_compare.py); no world is built."""
import types

import numpy as np
import pytest
import torch

from lem.latent_body.learning import ShadowLearner
from mimo_bridge.tests.latent_body_fixtures import config, example, tree
from mimo_bridge.tools import latent_body_compare as LC


def _learner(**extra):
    return ShadowLearner(tree(), dict(config(), **extra))


def test_radius_is_the_measured_quantile_of_real_root_codes():
    learner = _learner(code_bound="tanh")
    examples = [example(s) for s in range(1, 61)]
    with torch.no_grad():
        codes = learner.model.encode(learner.batch(examples))
    pooled = torch.cat([codes[r].abs().flatten() for r in learner.tree.roots])
    assert LC.measure_radius(learner, examples) == pytest.approx(float(pooled.quantile(.95)))
    with pytest.raises(ValueError, match="50"):
        LC.measure_radius(learner, examples[:10])


def test_latent_order_scores_the_hold_endpoint_and_changes_only_owned_rows(monkeypatch):
    # Defects caught: the score ignoring the measured hold factor, a template row outside the tree overwritten.
    import lem.planner.latent_candidates as LCand
    learner, ex = _learner(), example()
    t = learner.tree
    target = ex.start.bones[:, 4:7].numpy() + 1.0
    template = torch.zeros(26, 9)
    template[:, 3:7] = torch.tensor([0., 0., 0., 1.])
    template[:, 8] = .5
    seen = {}
    real = LCand.search

    def spy(model, state, history, score, **kw):
        seen["score"] = score
        return real(model, state, history, score, **kw)
    monkeypatch.setattr(LCand, "search", spy)
    order, predicted, receipt = LC.latent_order(learner, ex.history, ex.start, target, template, candidates=8,
                                                radius=.5, generator=torch.Generator().manual_seed(0))
    owned = sorted(i for ids in t.owned.values() for i in ids)
    others = [i for i in range(26) if i not in owned]
    assert torch.equal(order[others], template[others])
    assert not torch.equal(order[owned, :3], template[owned, :3])
    # a prediction of no movement scores the squared distance from the start to the goal (3 cm^2 per handle)
    still = {r: types.SimpleNamespace(change=torch.zeros(8, len(t.subtree[r]) * 6)) for r in t.roots}
    assert torch.allclose(seen["score"](still), torch.full((8,), 3.))
    # one step of 1 cm along the goal direction is carried HOLD_FACTOR steps: (4.2/sqrt3 - 1)^2 per axis
    step = torch.zeros(len(t.subtree[t.roots[0]]), 6)
    step[:, :3] = 1 / np.sqrt(3)
    moving = {r: types.SimpleNamespace(change=step.flatten().repeat(8, 1)) for r in t.roots}
    expected = 3 * (LC.HOLD_FACTOR / np.sqrt(3) - 1) ** 2
    assert torch.allclose(seen["score"](moving), torch.full((8,), float(expected)), atol=1e-4)
    assert receipt["candidates"] == 8 and receipt["hold_factor"] == LC.HOLD_FACTOR


def test_latent_order_refuses_observations_not_one_step_apart():
    learner, ex = _learner(), example()
    ex.start.sim_time += .01
    with pytest.raises(ValueError, match="one trained model step"):
        LC.latent_order(learner, ex.history, ex.start, np.zeros((26, 3)), torch.zeros(26, 9), candidates=4,
                        radius=.5, generator=torch.Generator())


def test_goal_error_is_the_rms_distance_over_the_named_handles():
    p = np.zeros((26, 3)); t = np.zeros((26, 3))
    t[2] = [3., 4., 0.]
    assert LC.goal_error_cm(p, t, [2]) == pytest.approx(5.)
    assert LC.goal_error_cm(p, t, [2, 3]) == pytest.approx(np.sqrt(12.5))


def _rows(latent, direct, synergy, controller, bodies):
    # every row its own start (body, warmup): the held-out line takes one sign per start
    return [dict(goal="g", body=b, warmup=k, outcomes=dict(latent=dict(goal_error_cm=l), direct=dict(goal_error_cm=d),
                                                           synergy=dict(goal_error_cm=s), controller=dict(goal_error_cm=c)))
            for k, (l, d, s, c, b) in enumerate(zip(latent, direct, synergy, controller, bodies))]


def test_summary_counts_wins_for_the_latent_and_applies_the_declared_lines():
    # Defect caught: reference and candidate swapped (a worse latent arm would then count as wins).
    n = 18
    bodies = [b for b in range(8) for _ in range(2)] + [7, 7]
    better = _rows([1.] * n, [2.] * n, [2.] * n, [1.5] * n, bodies)
    s = LC.summarize(better, [7])
    assert s["by_goal"]["g"]["latent_vs_direct"]["wins"] == 18 and s["by_goal"]["g"]["better_than_planner"]
    assert s["held_out"]["latent_vs_synergy"]["wins"] == 4 and not s["held_out"]["passes"]   # 4 starts < 5
    worse = _rows([3.] * n, [2.] * n, [2.] * n, [1.5] * n, bodies)
    s = LC.summarize(worse, [7])
    assert s["by_goal"]["g"]["latent_vs_direct"]["losses"] == 18 and not s["by_goal"]["g"]["better_than_planner"]
    # 13 of 18 is the declared line (P = 0.048); 12 of 18 is not (P = 0.119)
    for wins, passes in ((13, True), (12, False)):
        latent = [1.] * wins + [3.] * (n - wins)
        assert LC.summarize(_rows(latent, [2.] * n, [2.] * n, [2.] * n, bodies), [7])["by_goal"]["g"][
            "better_than_planner"] is passes


def test_index_curl_flexes_only_that_index_finger():
    geometry = dict(joints=[dict(name=n, limits=[-.1, 1.5]) for n in (
        "robot:left_ff_side", "robot:left_ff_knuckle", "robot:left_ff_middle", "robot:left_ff_distal",
        "robot:left_mf_knuckle", "robot:right_ff_knuckle", "robot:right_ff_middle", "robot:right_ff_distal")])
    model = types.SimpleNamespace(_hand_geometry=geometry,
                                  hand_kinematics=types.SimpleNamespace(project_joints=lambda j: j))
    session = types.SimpleNamespace(model=model)
    joints = torch.zeros(1, 8)
    _, research, _ = LC.index_curl_request(session, torch.zeros(1, 26, 9), joints, "L")
    assert research[0].tolist() == pytest.approx([0., 1.425, 1.425, 1.425, 0., 0., 0., 0.])
    _, research, _ = LC.index_curl_request(session, torch.zeros(1, 26, 9), joints, "R")
    assert research[0].tolist() == pytest.approx([0., 0., 0., 0., 0., 1.425, 1.425, 1.425])


def test_index_readings_run_offline_and_read_a_planted_wrist_code():
    # Defect caught: the read-out fitted and scored on the same rows (an overfit R^2 would hide nothing).
    from mimo_bridge.tools.latent_body_index_readings import _r2, index_readings
    x = torch.randn(200, 3, generator=torch.Generator().manual_seed(0))
    assert _r2(x, x @ torch.tensor([[1.], [2.], [-1.]])) == pytest.approx(1.0, abs=1e-6)
    noise = torch.randn(200, 1, generator=torch.Generator().manual_seed(1))
    assert _r2(torch.randn(200, 60, generator=torch.Generator().manual_seed(2)), noise) < 0.2
    learner = _learner()
    result = index_readings(learner, [example(s) for s in range(1, 9)], every=2, codes=4)
    assert set(result["parts"]) == {"index_L", "index_R"}
    assert all(np.isfinite(v) for part in result["parts"].values() for v in part.values())
