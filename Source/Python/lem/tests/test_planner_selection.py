# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Planning must return a command it scored, retaining better earlier candidates.

The exact scalar forward models isolate search correctness from model quality,
physical reachability and effort tradeoffs. Settings match the bridge's 20 Adam
updates at learning rate 0.1 unless the case deliberately needs one update.
"""
from types import SimpleNamespace
import math

import pytest
import torch

from lem.model.efference import attach_gaze_copy
from lem.model.intention_generator import IntentionGenerator


def make_generator(*, steps=20, strength_reg=0.0, extra_heads=None):
    modules = [torch.nn.Linear(1, 1, bias=False) for _ in range(3)]
    if extra_heads is not None:
        attach_gaze_copy(modules[2], extra_heads)
    config = {
        "body": {"n_handles": 1},
        "model": {"d_model": 1},
        "_active_modalities": ["touch", "motor"],
        "training": {"intention": {
            "n_steps": steps, "lr": 0.1, "strength_reg": strength_reg,
            "effort_per_body": True,
        }},
    }
    return IntentionGenerator(config, *modules), modules


def motor_at(values):
    motor = torch.zeros(len(values), 1, 9)
    motor[:, 0, 0] = torch.tensor(values)
    motor[..., 6] = 1.0
    motor[..., 7:9] = 0.5
    return motor


def run(generator, motor):
    state = {"touch": torch.zeros(len(motor), 1)}
    return generator.generate_action(state, state, motor)


def scalar_transition(monkeypatch, fn=None):
    seen = []

    def transition(_fm, _enc, _heads, _state, action, **kwargs):
        value = action[:, 0, :1] if fn is None else fn(action)
        seen.append(value.detach().clone())
        return SimpleNamespace(tokens={"touch": value})

    monkeypatch.setattr("lem.model.predictive_state.transition", transition)
    return seen


def test_twenty_adam_updates_cannot_discard_a_better_initial_command(monkeypatch):
    generator, _ = make_generator()
    seen = scalar_transition(monkeypatch)
    initial = motor_at([0.001])
    chosen, metadata = run(generator, initial)
    actual = chosen[:, 0, :1].square().mean().item()
    assert actual <= initial[:, 0, :1].square().mean().item() + 1e-12
    assert actual == pytest.approx(min(v.square().mean().item() for v in seen), abs=1e-12)
    assert metadata["last_goal_distance"] == pytest.approx(actual, abs=1e-12)
    assert metadata["last_loss"] == pytest.approx(actual, abs=1e-12)
    assert not chosen.requires_grad and chosen.grad_fn is None


def test_each_body_keeps_its_own_best_candidate_and_final_update_is_evaluated(monkeypatch):
    generator, _ = make_generator(steps=1)
    scalar_transition(monkeypatch)
    initial = motor_at([0.001, 1.0])
    chosen, metadata = run(generator, initial)
    assert chosen[0, 0, 0].item() == pytest.approx(0.001, abs=1e-9)
    assert chosen[1, 0, 0].item() == pytest.approx(0.9, abs=1e-6)
    actual = chosen[:, 0, :1].square().mean().item()
    assert metadata["last_goal_distance"] == pytest.approx(actual, abs=1e-7)


def test_selected_score_includes_the_effort_tradeoff(monkeypatch):
    generator, _ = make_generator(strength_reg=1.0)
    scalar_transition(monkeypatch, lambda a: a[:, 0, 8:9] - 0.5)
    chosen, metadata = run(generator, motor_at([0.0]))
    goal = (chosen[:, 0, 8:9] - 0.5).square().mean().item()
    effort = chosen[:, :, 8:9].square().sum(-1).sum(-1).mean().item()
    assert goal > 0.0, "A legitimate effort tradeoff may increase sensory goal error."
    assert goal + effort < 0.25
    assert metadata["last_goal_distance"] == pytest.approx(goal, abs=1e-7)
    assert metadata["last_effort"] == pytest.approx(effort, abs=1e-7)
    assert metadata["last_loss"] == pytest.approx(goal + effort, abs=1e-7)


def test_later_nonfinite_candidate_cannot_replace_finite_initial_command(monkeypatch):
    generator, _ = make_generator(steps=2)
    calls = [0]

    def value(action):
        calls[0] += 1
        x = action[:, 0, :1]
        return x if calls[0] == 1 else x * float("nan")

    scalar_transition(monkeypatch, value)
    initial = motor_at([0.001])
    chosen, _ = run(generator, initial)
    assert torch.isfinite(chosen).all()
    assert chosen[0, 0, 0].item() == pytest.approx(initial[0, 0, 0].item(), abs=1e-12)


def test_no_finite_candidate_raises_instead_of_issuing_an_unscored_command(monkeypatch):
    generator, _ = make_generator(steps=2)
    scalar_transition(monkeypatch, lambda a: a[:, 0, :1] * float("nan"))
    with pytest.raises((RuntimeError, ValueError), match="(?i)finite"):
        run(generator, motor_at([0.001]))


def test_planning_preserves_unregistered_gaze_copy_gradients_and_freeze_flags(monkeypatch):
    gaze_heads = torch.nn.Linear(1, 1, bias=False)
    generator, modules = make_generator(steps=2, extra_heads=gaze_heads)
    modules[0].weight.requires_grad_(False)
    gaze_heads.weight.grad = torch.full_like(gaze_heads.weight, 3.0)
    scalar_transition(monkeypatch, lambda a: gaze_heads(a[:, 0, :1]))
    initial_weights = [m.weight.detach().clone() for m in modules + [gaze_heads]]
    run(generator, motor_at([0.4]))
    assert torch.equal(gaze_heads.weight.grad, torch.full_like(gaze_heads.weight, 3.0))
    assert modules[0].weight.requires_grad is False
    assert gaze_heads.weight.requires_grad is True
    for module, before in zip(modules + [gaze_heads], initial_weights):
        assert torch.equal(module.weight, before)


def test_exception_restores_the_original_freeze_flags(monkeypatch):
    gaze_heads = torch.nn.Linear(1, 1, bias=False)
    generator, modules = make_generator(steps=2, extra_heads=gaze_heads)
    modules[0].weight.requires_grad_(False)
    gaze_heads.weight.requires_grad_(False)
    calls = [0]

    def value(action):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("injected transition failure")
        return action[:, 0, :1]

    scalar_transition(monkeypatch, value)
    with pytest.raises(RuntimeError, match="injected transition failure"):
        run(generator, motor_at([0.4]))
    assert modules[0].weight.requires_grad is False
    assert modules[1].weight.requires_grad is True
    assert gaze_heads.weight.requires_grad is False


@pytest.mark.parametrize("smooth_source", ["proposal", "executed"])
def test_captured_planner_route_scores_the_requested_gaze_without_body_smoothing(smooth_source):
    """Body-side reflexes act later; this checks the known outgoing eye request."""
    from mimo_bridge.lem_model import LemLite

    command = motor_at([0.0]).expand(1, 26, 9).clone()
    command[:, 0, 0] = 0.1
    command[:, 23, 5] = math.sin(math.radians(30) / 2)
    command[:, 23, 6] = math.cos(math.radians(30) / 2)
    previous = motor_at([0.0]).expand(1, 26, 9).clone()
    achieved = torch.zeros(1, 26, 10)
    achieved[..., 6] = 1.0
    dev = {"rom_limit": 1.0, "strength_limit": 1.0}
    actor = SimpleNamespace(
        GAZE_IDX=23, smooth_source=smooth_source, smooth_actions=0.35,
        _prev_mfa=previous, _smooth_exec_prev=previous,
        last_context=None, context_buffer=None, last_achieved=achieved,
        _last_source_mask={}, _last_dev=dev, device=torch.device("cpu"),
        _plan_parameter_versions=lambda: (), _plan_settings=lambda: (),
    )
    decision = LemLite._capture_plan_state(actor)
    imagined = LemLite.planned_execution(
        command, decision["dev"], decision["achieved"],
        previous=decision["smooth_prev"], smoothing=decision["smoothing"],
        keep_gaze_idx=decision.get("smooth_keep_gaze"),
    )
    planned_gaze = LemLite._gaze_from_handle(actor, imagined)
    requested_gaze = LemLite._gaze_from_handle(actor, command)
    assert planned_gaze[0, 0] == pytest.approx(requested_gaze[0, 0], abs=1e-7)
    # Other body rows retain the run's existing .35 new / .65 previous filter.
    assert imagined[0, 0, 0].item() == pytest.approx(7.0, abs=1e-6)


def test_action_space_summary_and_restored_parameters_describe_selected_bodies(monkeypatch):
    generator, _ = make_generator(steps=1)
    scalar_transition(monkeypatch)

    class ActionSpace:
        def init(self):
            self.z = torch.tensor([[0.001], [1.0]], requires_grad=True)
            return [self.z]

        def frozen_modules(self):
            return []

        def assemble(self):
            remainder = motor_at([0.0, 0.0])[:, 0, 1:]
            action = torch.cat([self.z, remainder], -1).unsqueeze(1)
            return action, action[..., 8:9], None

        def summary(self):
            return {"selected_z": self.z.detach().flatten().tolist()}

    space = ActionSpace()
    state = {"touch": torch.zeros(2, 1)}
    chosen, metadata = generator.generate_action(state, state, motor_at([0.001, 1.0]), action_space=space)
    assert torch.equal(chosen[:, 0, :1], space.z.detach())
    assert metadata["synergy"]["selected_z"] == pytest.approx([0.001, 0.9], abs=1e-6)


@pytest.mark.parametrize("per_body", [False, True])
def test_legacy_and_per_body_effort_formulas_survive_selection(monkeypatch, per_body):
    generator, _ = make_generator(steps=1, strength_reg=0.1)
    generator._effort_per_body = per_body
    scalar_transition(monkeypatch)
    chosen, metadata = run(generator, motor_at([0.001, 1.0]))
    squared = chosen[..., 8:9].square().sum()
    expected_effort = 0.1 * squared.item() / (2 if per_body else 1)
    expected_goal = chosen[:, 0, :1].square().mean().item()
    assert metadata["last_effort"] == pytest.approx(expected_effort, abs=1e-7)
    assert metadata["last_loss"] == pytest.approx(expected_goal + expected_effort, abs=1e-7)
