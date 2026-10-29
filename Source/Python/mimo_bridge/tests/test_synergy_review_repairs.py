# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Behavioural regression checks for defects confirmed during the V3 takeover."""
import math

import pytest
import torch

from lem.model.motor_synergy import following_counts, rotvec_to_quat
from mimo_bridge.tests.test_motor_synergy import _lite, _pose, _order_cm


@pytest.mark.parametrize("asked_deg,moved_deg", [(20, 10), (170, 170), (90, 180), (30, 90)])
def test_chance_count_matches_uniform_random_axes(asked_deg, moved_deg):
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(17)
    axes = torch.randn(30000, 3, generator=rng, dtype=torch.float64)
    axes /= axes.norm(dim=-1, keepdim=True)
    start = torch.zeros(len(axes), 1, 10, dtype=torch.float64)
    start[..., 6] = 1
    target = start[..., :9].clone()
    target[:, 0, 3:7] = rotvec_to_quat(axes * math.radians(asked_deg))
    target[..., 8] = 1
    end = start.clone()
    end[:, 0, 3:7] = rotvec_to_quat(torch.tensor([math.radians(moved_deg), 0., 0.],
                                               dtype=torch.float64)).expand(len(axes), 4)
    count = following_counts(target, start, end, [0])
    assert abs(float(count["closer"].double().mean() - count["chance"].mean())) < 0.01
    target[..., 3:7] *= -1
    again = following_counts(target, start, end, [0])
    for key in count:
        assert torch.equal(count[key], again[key]), key


def test_stillness_and_invalid_orders_do_not_earn_following():
    pose = _pose(1)
    order = _order_cm(pose)
    order[..., 3:7] = rotvec_to_quat(torch.tensor([0.2, 0., 0.]))
    order[..., 8] = 1
    count = following_counts(order, pose, pose, [0])
    assert count["closer"].numel() == 1 and not bool(count["closer"].any())
    assert float(count["chance"].sum()) == 0
    order[:, 0, 3] = float("nan")
    assert following_counts(order, pose, pose, [0])["closer"].numel() == 0


def test_nonfinite_extra_gradient_reaches_the_optimizer_skip_guard():
    model = _lite(motor_synergy=True)
    for p in model.params:
        p.grad = torch.zeros_like(p)
    next(model.synergy.parameters()).grad.fill_(float("nan"))
    assert not math.isfinite(model._clip_training_gradients())


@pytest.mark.parametrize("column", [0, 75])
def test_nonfinite_babble_returns_fallback_without_changing_the_order(column):
    model = _lite(motor_synergy=True, synergy_babble=True)
    pose = _pose(2)
    order = _order_cm(pose)
    before = order.clone()
    with torch.no_grad():
        model.synergy.S3.weight[column, 0] = float("nan")
        out = model._synergy_babble(order, pose, 0.05, pose[..., :3] * 200, 40.)
    assert out is None and torch.equal(order, before)
    assert model.synergy_window_summary()["babble_nonfinite"] == 1


def test_synergy_can_learn_with_both_behaviour_switches_off():
    model = _lite(motor_synergy=True, synergy_planner=False, synergy_babble=False)
    assert model.synergy is not None
    assert model.receipt["motor_synergy"]["planner"] is False
    assert model.receipt["motor_synergy"]["babble"] is False
    with pytest.raises(ValueError, match="synergy-planner"):
        _lite(motor_synergy=True, synergy_planner=False, synergy_residual_l2=0.1)


def test_frozen_synergy_observation_does_not_change_its_pose_normalization():
    model = _lite(motor_synergy=True)
    centre = model.synergy.centre_mean.clone()
    pose = _pose(2)
    model._synergy_lesson(_order_cm(pose), 1, pose, 100, 2, train=False)
    assert torch.equal(model.synergy.centre_mean, centre)
    assert model.synergy.centre_weight == 0


def test_planner_and_babble_liveness_are_reported_separately():
    model = _lite(motor_synergy=True, synergy_babble=True)
    pose = _pose(2)
    raw = _order_cm(pose)
    raw[..., :3] /= 200
    space = model._synergy_space(pose, raw)
    space.init()
    space.assemble()
    with torch.no_grad():
        model._synergy_babble(_order_cm(pose), pose, 0.05, pose[..., :3] * 200, 40.)
    regimes = model.synergy_window_summary()['liveness_by_regime']
    assert set(regimes) == {'planner', 'babble'}
    assert set(regimes['planner']) == {'E1', 'E2', 'S1', 'S2'}
    assert set(regimes['babble']) == {'S1', 'S2'}
    assert model.synergy_window_summary()['liveness_by_regime'] == {}
