# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_latent_regularizer.py — Tests for latent regularizer (CF-11, UEMA Spec v3.4 §6.3).

Tests:
  - Predictions inside ±2σ → penalty is exactly zero
  - Predictions outside ±2σ → penalty is positive and quadratic in excess
  - Gradient flows to prediction tensors, not to running stats
  - Running stats converge toward actual encoder output distribution
  - 5 modalities scored (vision, audio, tactile, proprio, intermind), not motor
  - Config parsing and CLI override
  - LossManager reads weight and applies it in breakdown
"""

import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lem.config.resolver import resolve
from lem.training.latent_regularizer import LatentRegularizer
from lem.training.loss_manager import LossManager


YAML_PATH = str(ROOT / "lem" / "master_config.yaml")


@pytest.fixture
def resolved():
    with open(YAML_PATH) as f:
        master = yaml.safe_load(f)
    return resolve(copy.deepcopy(master))


@pytest.fixture
def device():
    return torch.device("cpu")


def _make_regularizer(d_model=768, sigma=2.0, momentum=0.99, device=None):
    """Create a LatentRegularizer with optional device placement."""
    reg = LatentRegularizer(d_model=d_model, sigma_threshold=sigma, momentum=momentum)
    if device is not None:
        reg = reg.to(device)
    return reg


def _make_predictions(mods, d_model, device, value=None):
    """Create prediction dict {mod: [B, d_model]}. If value is given, fill with it."""
    if value is not None:
        return {mod: torch.full((1, d_model), value, device=device) for mod in mods}
    return {mod: torch.randn(1, d_model, device=device) for mod in mods}


# =========================================================================
# Test 1: Inside envelope — penalty is exactly zero
# =========================================================================

class TestInsideEnvelope:
    def test_predictions_at_mean_zero_penalty(self, device):
        """Predictions exactly at running mean → zero penalty."""
        d = 64
        reg = _make_regularizer(d_model=d, device=device)

        # Initialize stats with known distribution: mean=0, var=1
        outputs = {"a": torch.randn(100, d, device=device)}
        reg.update(outputs)

        # Predict at the running mean — should be well within ±2σ
        preds = {"a": reg.running_mean.unsqueeze(0)}
        penalty = reg.penalty(preds)
        assert penalty.item() == pytest.approx(0.0, abs=1e-7)

    def test_predictions_within_1sigma_zero_penalty(self, device):
        """Predictions within ±1σ → zero penalty (well inside ±2σ envelope)."""
        d = 64
        reg = _make_regularizer(d_model=d, device=device)

        # Initialize with known distribution
        data = torch.randn(1000, d, device=device) * 2.0 + 1.0  # mean~1, std~2
        reg.update({"a": data})

        running_std = (reg.running_var + 1e-8).sqrt()
        preds = {"a": (reg.running_mean + 0.5 * running_std).unsqueeze(0)}
        penalty = reg.penalty(preds)
        assert penalty.item() == pytest.approx(0.0, abs=1e-6)

    def test_predictions_at_2sigma_boundary_zero_penalty(self, device):
        """Predictions exactly at ±2σ boundary → zero penalty (relu clips to 0)."""
        d = 32
        reg = _make_regularizer(d_model=d, device=device)

        # Set known stats directly
        reg.running_mean = torch.zeros(d, device=device)
        reg.running_var = torch.ones(d, device=device)
        reg._initialized = True

        preds = {"a": torch.full((1, d), 2.0, device=device)}  # exactly at 2σ
        penalty = reg.penalty(preds)
        assert penalty.item() == pytest.approx(0.0, abs=1e-7)


# =========================================================================
# Test 2: Outside envelope — penalty is positive and quadratic
# =========================================================================

class TestOutsideEnvelope:
    def test_predictions_at_4sigma_positive_penalty(self, device):
        """Predictions at 4σ → excess=2, penalty=mean(4.0)."""
        d = 32
        reg = _make_regularizer(d_model=d, device=device)

        # Set known stats: mean=0, var=1 → std=1
        reg.running_mean = torch.zeros(d, device=device)
        reg.running_var = torch.ones(d, device=device)
        reg._initialized = True

        # Predict at 4σ → z_score=4, excess=relu(4-2)=2, excess²=4, mean=4
        preds = {"a": torch.full((1, d), 4.0, device=device)}
        penalty = reg.penalty(preds)
        assert penalty.item() == pytest.approx(4.0, abs=1e-5)

    def test_penalty_quadratic_in_excess(self, device):
        """Penalty scales quadratically: 3σ→excess=1→1, 4σ→excess=2→4, 5σ→excess=3→9."""
        d = 16
        reg = _make_regularizer(d_model=d, device=device)

        reg.running_mean = torch.zeros(d, device=device)
        reg.running_var = torch.ones(d, device=device)
        reg._initialized = True

        penalties = []
        for sigma_val in [3.0, 4.0, 5.0]:
            preds = {"a": torch.full((1, d), sigma_val, device=device)}
            penalties.append(reg.penalty(preds).item())

        # excess = [1, 2, 3], penalty = [1, 4, 9]
        assert penalties[0] == pytest.approx(1.0, abs=1e-5)
        assert penalties[1] == pytest.approx(4.0, abs=1e-5)
        assert penalties[2] == pytest.approx(9.0, abs=1e-5)

    def test_negative_predictions_also_penalized(self, device):
        """Predictions at -4σ also trigger penalty (uses |z_score|)."""
        d = 16
        reg = _make_regularizer(d_model=d, device=device)

        reg.running_mean = torch.zeros(d, device=device)
        reg.running_var = torch.ones(d, device=device)
        reg._initialized = True

        preds_pos = {"a": torch.full((1, d), 4.0, device=device)}
        preds_neg = {"a": torch.full((1, d), -4.0, device=device)}

        p_pos = reg.penalty(preds_pos).item()
        p_neg = reg.penalty(preds_neg).item()
        assert p_pos == pytest.approx(p_neg, abs=1e-6), \
            "Positive and negative deviations should produce equal penalty"


# =========================================================================
# Test 3: Gradient flow
# =========================================================================

class TestGradientFlow:
    def test_penalty_has_gradient_to_predictions(self, device):
        """penalty() returns tensor with gradient flow to prediction inputs."""
        d = 32
        reg = _make_regularizer(d_model=d, device=device)

        reg.running_mean = torch.zeros(d, device=device)
        reg.running_var = torch.ones(d, device=device)
        reg._initialized = True

        pred = torch.full((1, d), 4.0, device=device, requires_grad=True)
        penalty = reg.penalty({"a": pred})
        penalty.backward()

        assert pred.grad is not None, "Prediction tensor should have gradient"
        assert pred.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_running_stats_have_no_gradient(self, device):
        """Running stats should never require grad (updated under no_grad)."""
        d = 32
        reg = _make_regularizer(d_model=d, device=device)

        # Update with some data
        reg.update({"a": torch.randn(10, d, device=device)})

        assert not reg.running_mean.requires_grad, "running_mean should not require grad"
        assert not reg.running_var.requires_grad, "running_var should not require grad"

    def test_update_does_not_create_grad_graph(self, device):
        """update() should not build computation graph (no_grad context)."""
        d = 32
        reg = _make_regularizer(d_model=d, device=device)

        data = torch.randn(10, d, device=device, requires_grad=True)
        reg.update({"a": data})

        # Running stats should be leaf tensors with no grad_fn
        assert reg.running_mean.grad_fn is None
        assert reg.running_var.grad_fn is None


# =========================================================================
# Test 4: Running stats convergence
# =========================================================================

class TestRunningStatsConvergence:
    def test_stats_converge_to_distribution(self, device):
        """After many updates, dimension-averaged running mean/var approximate true stats."""
        d = 64
        reg = _make_regularizer(d_model=d, momentum=0.9, device=device)  # faster convergence

        true_mean = 3.0
        true_var = 4.0  # std=2

        for _ in range(200):
            batch = torch.randn(50, d, device=device) * 2.0 + 3.0
            reg.update({"a": batch})

        # Check mean across dimensions (per-dimension has sampling noise)
        assert reg.running_mean.mean().item() == pytest.approx(true_mean, abs=0.2), \
            f"running_mean should be ~{true_mean}, got {reg.running_mean.mean().item():.3f}"
        assert reg.running_var.mean().item() == pytest.approx(true_var, abs=0.3), \
            f"running_var should be ~{true_var}, got {reg.running_var.mean().item():.3f}"

    def test_first_update_initializes_directly(self, device):
        """First update should copy batch stats directly (no EMA blend)."""
        d = 16
        reg = _make_regularizer(d_model=d, device=device)

        data = torch.randn(100, d, device=device) * 5.0 + 10.0
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0, unbiased=False)

        reg.update({"a": data})

        assert torch.allclose(reg.running_mean, batch_mean, atol=1e-6)
        assert torch.allclose(reg.running_var, batch_var, atol=1e-6)


# =========================================================================
# Test 5: Modality filtering — 5 predicted modalities, not motor
# =========================================================================

class TestModalityFiltering:
    def test_five_modalities_scored(self, resolved, device):
        """Penalty should score predictions for all _prediction_head_modalities."""
        d_model = resolved["model"]["d_model"]
        pred_mods = resolved["_prediction_head_modalities"]

        reg = _make_regularizer(d_model=d_model, device=device)
        reg.running_mean = torch.zeros(d_model, device=device)
        reg.running_var = torch.ones(d_model, device=device)
        reg._initialized = True

        # All predictions at 4σ — penalty should reflect all modalities
        preds = {mod: torch.full((1, d_model), 4.0, device=device) for mod in pred_mods}
        penalty = reg.penalty(preds)

        # Each modality contributes excess²=4 per dim, mean across all = 4.0
        assert penalty.item() == pytest.approx(4.0, abs=1e-4)

    def test_motor_excluded_from_predictions(self, resolved):
        """Motor should not be in _prediction_head_modalities."""
        assert "motor" not in resolved["_prediction_head_modalities"]

    def test_empty_predictions_zero_penalty(self, device):
        """Empty prediction dict → zero penalty."""
        reg = _make_regularizer(d_model=32, device=device)
        reg._initialized = True
        penalty = reg.penalty({})
        assert penalty.item() == 0.0


# =========================================================================
# Test 6: Config parsing
# =========================================================================

class TestConfigParsing:
    def test_latent_reg_config_loads(self):
        """latent_reg config section loads from master_config.yaml."""
        with open(YAML_PATH) as f:
            master = yaml.safe_load(f)
        cfg = master.get("training", {}).get("latent_reg", {})
        assert cfg.get("enabled") is True
        assert cfg.get("loss_weight") == 0.01
        assert cfg.get("sigma_threshold") == 2.0
        assert cfg.get("momentum") == 0.99

    def test_loss_manager_reads_latent_reg_weight(self, resolved):
        """LossManager reads latent_reg weight from config."""
        lm = LossManager(resolved)
        assert lm._latent_reg_weight == 0.01


# =========================================================================
# Test 7: LossManager integration
# =========================================================================

class TestLossManagerIntegration:
    def test_latent_reg_loss_weighted_in_breakdown(self, resolved, device):
        """latent_reg loss in breakdown equals raw penalty * 0.01 weight."""
        lm = LossManager(resolved)
        d_model = resolved["model"]["d_model"]

        predictions = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        targets = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        source_mask = {mod: "real" for mod in resolved["_active_modalities"]}

        raw_penalty = torch.tensor(5.0, device=device, requires_grad=True)

        loss, bd = lm.compute(
            predictions, targets, source_mask,
            return_breakdown=True,
            latent_reg_loss=raw_penalty,
        )

        assert "latent_reg" in bd, "latent_reg not in breakdown"
        expected = 5.0 * 0.01
        assert bd["latent_reg"].item() == pytest.approx(expected, abs=1e-6), \
            f"Expected {expected}, got {bd['latent_reg'].item()}"

    def test_latent_reg_loss_none_no_breakdown(self, resolved, device):
        """When latent_reg_loss=None, no latent_reg key in breakdown."""
        lm = LossManager(resolved)
        d_model = resolved["model"]["d_model"]

        predictions = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        targets = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        source_mask = {mod: "real" for mod in resolved["_active_modalities"]}

        loss, bd = lm.compute(
            predictions, targets, source_mask,
            return_breakdown=True,
            latent_reg_loss=None,
        )

        assert "latent_reg" not in bd, "latent_reg should not appear when loss is None"

    def test_latent_reg_contributes_to_total_loss(self, resolved, device):
        """Total loss should increase when latent_reg_loss is provided."""
        lm = LossManager(resolved)
        d_model = resolved["model"]["d_model"]

        predictions = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        targets = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        source_mask = {mod: "real" for mod in resolved["_active_modalities"]}

        loss_without = lm.compute(predictions, targets, source_mask)
        loss_with = lm.compute(
            predictions, targets, source_mask,
            latent_reg_loss=torch.tensor(100.0, device=device),
        )

        assert loss_with.item() > loss_without.item(), \
            "Total loss should increase with latent_reg penalty"
