# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_intention_generator.py — Tests for intention generator (CF-9, UEMA Spec v3.4 §10).

Tests:
  1.  generate_action returns correct shape [B, n_handles, 9]
  2.  Rotation normalized (quaternion norm ≈ 1.0)
  3.  Radius and strength in [0, 1]
  4.  Loss decreases over optimization steps
  5.  Forward model params unchanged after generate_action
  6.  plan_multistep early exit on MSE < 0.01
  7.  plan_multistep safety exit on wellness < 0.2
  8.  plan_multistep returns at most max_steps actions
  9.  CrossSimulationAttention produces consensus with correct shape
  10. generate_action runs the same with the modules in train mode and in eval mode (no mode gate)
  11. Params restored on exception (try/finally safety)
"""

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lem.config.resolver import resolve
from lem.model.forward_model import ForwardModel
from lem.model.prediction_heads import PredictionHeads
from lem.model.encoder_registry import EncoderRegistry
from lem.model.intention_generator import (
    IntentionGenerator,
    CrossSimulationAttention,
    plan_multistep,
    _dict_mse,
)

YAML_PATH = str(ROOT / "lem" / "master_config.yaml")


@pytest.fixture
def resolved():
    with open(YAML_PATH) as f:
        master = yaml.safe_load(f)
    return resolve(copy.deepcopy(master))


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def models(resolved, device):
    """Build forward model, prediction heads, and encoder registry."""
    fm = ForwardModel(resolved).to(device)
    ph = PredictionHeads(resolved).to(device)
    er = EncoderRegistry(resolved).to(device)
    return fm, ph, er


def _make_encoder_outputs(resolved, device):
    """Create fake encoder outputs {mod: [B, d_model]} for all active mods."""
    d_model = resolved["model"]["d_model"]
    return {
        mod: torch.randn(1, d_model, device=device)
        for mod in resolved["_active_modalities"]
    }


def _make_motor(resolved, device):
    """Create fake motor output [B, n_handles, 9]."""
    n_handles = resolved["body"]["n_handles"]
    return torch.randn(1, n_handles, 9, device=device)


# =========================================================================
# Test 1: generate_action output shape
# =========================================================================

class TestGenerateActionShape:
    def test_output_shape(self, resolved, device, models):
        fm, ph, er = models
        ig = IntentionGenerator(resolved, fm, er, ph)
        n_handles = resolved["body"]["n_handles"]

        current = _make_encoder_outputs(resolved, device)
        desired = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        action, meta = ig.generate_action(current, desired, motor)
        assert action.shape == (1, n_handles, 9)
        assert "first_loss" in meta
        assert "last_loss" in meta


# =========================================================================
# Test 2: Rotation normalized
# =========================================================================

class TestRotationNormalized:
    def test_quaternion_norm(self, resolved, device, models):
        fm, ph, er = models
        ig = IntentionGenerator(resolved, fm, er, ph)

        current = _make_encoder_outputs(resolved, device)
        desired = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        action, _ = ig.generate_action(current, desired, motor)
        rot = action[:, :, 3:7]
        norms = rot.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Quaternion norms not unit: min={norms.min():.4f}, max={norms.max():.4f}"


# =========================================================================
# Test 3: Radius and strength bounded [0, 1]
# =========================================================================

class TestRadiusStrengthBounded:
    def test_bounds(self, resolved, device, models):
        fm, ph, er = models
        ig = IntentionGenerator(resolved, fm, er, ph)

        current = _make_encoder_outputs(resolved, device)
        desired = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        action, _ = ig.generate_action(current, desired, motor)
        radius = action[:, :, 7:8]
        strength = action[:, :, 8:9]
        assert radius.min() >= 0.0, f"Radius below 0: {radius.min():.4f}"
        assert radius.max() <= 1.0, f"Radius above 1: {radius.max():.4f}"
        assert strength.min() >= 0.0, f"Strength below 0: {strength.min():.4f}"
        assert strength.max() <= 1.0, f"Strength above 1: {strength.max():.4f}"


# =========================================================================
# Test 4: Loss decreases over optimization steps
# =========================================================================

class TestLossDecreases:
    def test_convergence(self, resolved, device, models):
        fm, ph, er = models
        # Use fewer steps for test speed
        cfg_override = copy.deepcopy(resolved)
        cfg_override.setdefault("training", {}).setdefault("intention", {})["n_steps"] = 20
        ig = IntentionGenerator(cfg_override, fm, er, ph)

        current = _make_encoder_outputs(resolved, device)
        desired = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        _, meta = ig.generate_action(current, desired, motor)
        assert meta["last_loss"] < meta["first_loss"], \
            f"Loss did not decrease: {meta['first_loss']:.6f} -> {meta['last_loss']:.6f}"


# =========================================================================
# Test 5: Forward model params unchanged after generate_action
# =========================================================================

class TestParamsUnchanged:
    def test_forward_model_frozen(self, resolved, device, models):
        fm, ph, er = models
        cfg_override = copy.deepcopy(resolved)
        cfg_override.setdefault("training", {}).setdefault("intention", {})["n_steps"] = 5
        ig = IntentionGenerator(cfg_override, fm, er, ph)

        # Snapshot all model params
        fm_snap = {n: p.clone() for n, p in fm.named_parameters()}
        er_snap = {n: p.clone() for n, p in er.named_parameters()}
        ph_snap = {n: p.clone() for n, p in ph.named_parameters()}

        current = _make_encoder_outputs(resolved, device)
        desired = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        ig.generate_action(current, desired, motor)

        for n, p in fm.named_parameters():
            assert torch.equal(p, fm_snap[n]), \
                f"Forward model param {n} changed"
        for n, p in er.named_parameters():
            assert torch.equal(p, er_snap[n]), \
                f"Encoder registry param {n} changed"
        for n, p in ph.named_parameters():
            assert torch.equal(p, ph_snap[n]), \
                f"Prediction heads param {n} changed"


# =========================================================================
# Test 6: plan_multistep early exit on convergence
# =========================================================================

class TestPlanConvergence:
    def test_early_exit_same_state(self, resolved, device, models):
        """When current state equals goal, should immediately converge."""
        fm, ph, er = models
        cfg_override = copy.deepcopy(resolved)
        cfg_override.setdefault("training", {}).setdefault("intention", {})["n_steps"] = 5
        ig = IntentionGenerator(cfg_override, fm, er, ph)

        from lem.model.wellness import WellnessEvaluators
        wellness = WellnessEvaluators(resolved).to(device)

        state = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        plan, meta = plan_multistep(
            ig, fm, er, ph, wellness.score_predicted_wellness,
            state, state, motor, max_steps=5)

        assert meta["converged"] is True
        assert meta["plan_length"] == 0


# =========================================================================
# Test 7: plan_multistep wellness safety exit
# =========================================================================

class TestPlanWellnessExit:
    def test_safety_exit(self, resolved, device, models):
        """Mock wellness returning zeros -> wellness_exit=True."""
        fm, ph, er = models
        cfg_override = copy.deepcopy(resolved)
        cfg_override.setdefault("training", {}).setdefault("intention", {})["n_steps"] = 3
        ig = IntentionGenerator(cfg_override, fm, er, ph)

        def mock_wellness(predicted_tokens, z_motor):
            return torch.zeros(1, 5, device=device)

        current = _make_encoder_outputs(resolved, device)
        goal = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        plan, meta = plan_multistep(
            ig, fm, er, ph, mock_wellness,
            current, goal, motor, max_steps=5)

        assert meta["wellness_exit"] is True
        assert meta["plan_length"] <= 1


# =========================================================================
# Test 8: plan_multistep max_steps limit
# =========================================================================

class TestPlanMaxSteps:
    def test_max_steps_enforced(self, resolved, device, models):
        fm, ph, er = models
        cfg_override = copy.deepcopy(resolved)
        cfg_override.setdefault("training", {}).setdefault("intention", {})["n_steps"] = 3
        ig = IntentionGenerator(cfg_override, fm, er, ph)

        def lenient_wellness(predicted_tokens, z_motor):
            return torch.ones(1, 5, device=device)

        current = _make_encoder_outputs(resolved, device)
        goal = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        plan, meta = plan_multistep(
            ig, fm, er, ph, lenient_wellness,
            current, goal, motor, max_steps=3)

        assert meta["plan_length"] <= 3


# =========================================================================
# Test 9: CrossSimulationAttention consensus shape
# =========================================================================

class TestCrossSimAttention:
    def test_consensus_shape(self, resolved, device, models):
        fm, ph, er = models
        cfg_override = copy.deepcopy(resolved)
        intention_cfg = cfg_override.setdefault("training", {}).setdefault("intention", {})
        intention_cfg["n_steps"] = 3
        intention_cfg["cross_sim_rollouts"] = 2  # fewer for speed
        ig = IntentionGenerator(cfg_override, fm, er, ph)
        csa = CrossSimulationAttention(cfg_override).to(device)
        n_handles = resolved["body"]["n_handles"]

        def lenient_wellness(predicted_tokens, z_motor):
            return torch.ones(1, 5, device=device)

        current = _make_encoder_outputs(resolved, device)
        goal = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        consensus_plan, attn_weights, meta = csa(
            ig, fm, er, ph, lenient_wellness,
            current, goal, motor, max_plan_steps=2)

        assert isinstance(consensus_plan, list)
        assert meta["n_rollouts"] == 2
        if consensus_plan:
            assert consensus_plan[0].shape == (1, n_handles, 9)
        if attn_weights is not None:
            assert attn_weights.dim() == 3  # [B, query_len, key_len]


# =========================================================================
# Test 10: there is no train/eval gate on planning
# =========================================================================

class TestNoTrainingModeGate:
    """This was TestIntentionGating. Its body set fm.train(), computed
    `should_run_train = not fm.training` into a local and asserted on that
    local: the same lines pass with a bare nn.Linear in place of the forward
    model, so they pinned no code in this repo. The property they named is not
    the behaviour either. mimo_bridge/lem_model.py calls .train()/.eval() zero
    times and guards the planner only with
    `len(self.goal_ring) >= 20 and step % self.intention_every == 0`,
    so every training run plans with the forward model in train mode. The one
    place that reads the flag, lem/live_test.py (`not forward_model.training`),
    never reaches it: that file calls .eval() zero times, so its intention block
    logs {"status": "waiting"} forever.
    """

    def test_generate_action_runs_in_train_and_in_eval_mode(self, resolved, device, models):
        fm, ph, er = models
        cfg_override = copy.deepcopy(resolved)
        cfg_override.setdefault("training", {}).setdefault("intention", {})["n_steps"] = 20
        ig = IntentionGenerator(cfg_override, fm, er, ph)

        current = _make_encoder_outputs(resolved, device)
        desired = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        try:
            for mode in ("train", "eval"):
                for m in (fm, ph, er):
                    m.train(mode == "train")
                    assert m.training is (mode == "train")
                action, meta = ig.generate_action(current, desired, motor)
                assert meta["n_steps"] == 20, f"{mode}: the Adam loop did not run"
                assert meta["first_loss"] is not None and meta["last_loss"] is not None, \
                    f"{mode}: generate_action returned without running a round"
                assert meta["last_loss"] < meta["first_loss"], (
                    f"{mode}: loss did not descend "
                    f"({meta['first_loss']:.6f} -> {meta['last_loss']:.6f})")
                assert action.shape == motor.shape
        finally:
            for m in (fm, ph, er):
                m.train()


# =========================================================================
# Test 11: Params restored on exception (try/finally safety)
# =========================================================================

class TestParamsRestoredOnException:
    def test_requires_grad_restored_after_failure(self, resolved, device, models):
        """If predict_from_action throws, model params must still have requires_grad=True."""
        fm, ph, er = models
        cfg_override = copy.deepcopy(resolved)
        cfg_override.setdefault("training", {}).setdefault("intention", {})["n_steps"] = 5
        ig = IntentionGenerator(cfg_override, fm, er, ph)

        current = _make_encoder_outputs(resolved, device)
        desired = _make_encoder_outputs(resolved, device)
        motor = _make_motor(resolved, device)

        # Raise from the CANONICAL transition, which is what the optimizer now descends.
        # This used to patch predict_from_action - a route generate_action no longer
        # takes - so the test silently stopped exercising anything, which is exactly how
        # the legacy path stayed in use unnoticed.
        import lem.model.predictive_state as _ps
        call_count = [0]
        orig_transition = _ps.transition

        def raising_predict(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("Simulated CUDA OOM")
            return orig_transition(*args, **kwargs)

        # Patched on the CANONICAL route. This used to patch predict_from_action, which
        # the optimizer no longer calls - so the test silently stopped exercising
        # anything, which is exactly how the legacy route stayed in use unnoticed.
        import lem.model.predictive_state as _ps
        with patch.object(_ps, 'transition', side_effect=raising_predict):
            with pytest.raises(RuntimeError, match="Simulated CUDA OOM"):
                ig.generate_action(current, desired, motor)

        # Verify all model params still have requires_grad=True
        for n, p in fm.named_parameters():
            assert p.requires_grad, \
                f"Forward model param {n} has requires_grad=False after exception"
        for n, p in er.named_parameters():
            assert p.requires_grad, \
                f"Encoder registry param {n} has requires_grad=False after exception"
        for n, p in ph.named_parameters():
            assert p.requires_grad, \
                f"Prediction heads param {n} has requires_grad=False after exception"


# =========================================================================
# Config parsing
# =========================================================================

class TestConfigParsing:
    def test_intention_config_exists(self):
        with open(YAML_PATH) as f:
            master = yaml.safe_load(f)
        intention = master.get("training", {}).get("intention", {})
        assert intention.get("enabled") is not None
        assert intention.get("n_steps") == 50
        assert intention.get("lr") == 0.1
        assert intention.get("strength_reg") == 0.01
        assert intention.get("max_plan_steps") == 5
        assert intention.get("cross_sim_rollouts") == 5
        assert intention.get("cross_sim_noise") == 0.1
