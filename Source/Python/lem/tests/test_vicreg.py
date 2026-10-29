# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_vicreg.py — Tests for VICReg encoder regularization.

Tests:
  1. Collapsed embeddings → high variance loss
  2. Spread embeddings → near-zero variance loss
  3. Correlated dimensions → high covariance loss
  4. Independent dimensions → near-zero covariance loss
  5. Config toggle off → returns 0
  6. Batch size 1 → gracefully returns 0
  7. Per-modality → each gets own terms, averaged
  8. Integration → LossManager.compute() with VICReg enabled
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

YAML_PATH = str(ROOT / "lem" / "master_config.yaml")


def _make_config(vicreg_enabled=True, **vicreg_overrides):
    """Build a resolved config with VICReg settings."""
    with open(YAML_PATH) as f:
        raw = yaml.safe_load(f)
    raw["training"]["loss"]["vicreg"] = {
        "enabled": vicreg_enabled,
        "variance_weight": 1.0,
        "covariance_weight": 0.04,
        "variance_target": 1.0,
        **vicreg_overrides,
    }
    return resolve(copy.deepcopy(raw))


def _make_loss_manager(vicreg_enabled=True, **vicreg_overrides):
    """Build a LossManager with VICReg settings."""
    from lem.training.loss_manager import LossManager
    cfg = _make_config(vicreg_enabled, **vicreg_overrides)
    return LossManager(cfg)


# =========================================================================
# Test 1: Collapsed embeddings → high variance loss
# =========================================================================

class TestCollapsedEmbeddings:
    def test_collapsed_high_variance_loss(self):
        lm = _make_loss_manager(vicreg_enabled=True)
        # All rows identical → std = 0 across batch → max variance penalty
        row = torch.randn(1, 512)
        collapsed = row.expand(8, -1).clone()  # [8, 512], all identical
        encoder_outputs = {"vision": collapsed.unsqueeze(1)}  # [8, 1, 512]

        loss = lm.compute_vicreg(encoder_outputs)
        # Variance loss should be high (target=1.0, std=0 → relu(1-0)=1.0)
        assert loss.item() > 0.5, f"Expected high loss for collapsed embeddings, got {loss.item()}"


# =========================================================================
# Test 2: Spread embeddings → near-zero variance loss
# =========================================================================

class TestSpreadEmbeddings:
    def test_spread_low_variance_loss(self):
        lm = _make_loss_manager(
            vicreg_enabled=True,
            variance_weight=1.0,
            covariance_weight=0.0,  # isolate variance term
        )
        # Large batch so per-dim std converges to ~1.0
        torch.manual_seed(42)
        spread = torch.randn(256, 512)
        encoder_outputs = {"vision": spread.unsqueeze(1)}

        loss = lm.compute_vicreg(encoder_outputs)
        # Variance component should be near zero (std ≈ 1.0 meets target=1.0)
        assert loss.item() < 0.15, f"Expected low variance loss for spread embeddings, got {loss.item()}"


# =========================================================================
# Test 3: Correlated dimensions → high covariance loss
# =========================================================================

class TestCorrelatedDimensions:
    def test_correlated_high_covariance_loss(self):
        # Isolate covariance by zeroing variance weight
        lm = _make_loss_manager(
            vicreg_enabled=True, variance_weight=0.0, covariance_weight=1.0,
        )
        torch.manual_seed(42)
        z = torch.randn(64, 512)
        # Make col 1 identical to col 0 → perfect correlation
        z[:, 1] = z[:, 0]
        encoder_outputs = {"vision": z.unsqueeze(1)}

        loss_corr = lm.compute_vicreg(encoder_outputs)

        # Compare with independent version
        lm2 = _make_loss_manager(
            vicreg_enabled=True, variance_weight=0.0, covariance_weight=1.0,
        )
        torch.manual_seed(42)
        z_indep = torch.randn(64, 512)
        encoder_outputs_indep = {"vision": z_indep.unsqueeze(1)}
        loss_indep = lm2.compute_vicreg(encoder_outputs_indep)

        assert loss_corr.item() > loss_indep.item(), (
            f"Correlated loss ({loss_corr.item()}) should exceed "
            f"independent loss ({loss_indep.item()})"
        )


# =========================================================================
# Test 4: Independent dimensions → near-zero covariance loss
# =========================================================================

class TestIndependentDimensions:
    def test_independent_lower_than_correlated(self):
        """Independent dims should have lower covariance loss than correlated dims."""
        lm = _make_loss_manager(
            vicreg_enabled=True,
            variance_weight=0.0,  # isolate covariance
            covariance_weight=1.0,
        )
        torch.manual_seed(42)
        z_indep = torch.randn(64, 512)
        loss_indep = lm.compute_vicreg({"vision": z_indep.unsqueeze(1)})

        # Correlated version: make many dims copies of each other
        z_corr = z_indep.clone()
        for i in range(0, 256, 2):
            z_corr[:, i + 1] = z_corr[:, i]  # 128 correlated pairs
        loss_corr = lm.compute_vicreg({"vision": z_corr.unsqueeze(1)})

        assert loss_indep.item() < loss_corr.item(), (
            f"Independent ({loss_indep.item()}) should be lower than "
            f"correlated ({loss_corr.item()})"
        )


# =========================================================================
# Test 5: Config toggle off → returns 0
# =========================================================================

class TestConfigDisabled:
    def test_disabled_returns_zero(self):
        lm = _make_loss_manager(vicreg_enabled=False)
        z = torch.ones(8, 512)  # Would be high loss if enabled
        encoder_outputs = {"vision": z.unsqueeze(1)}

        loss = lm.compute_vicreg(encoder_outputs)
        assert loss.item() == 0.0, f"Disabled VICReg should return 0, got {loss.item()}"

    def test_disabled_not_in_total_loss(self):
        lm = _make_loss_manager(vicreg_enabled=False)
        # Minimal predictions/targets for compute()
        d_model = 512
        predictions = {"vision": torch.randn(1, d_model)}
        targets = {"vision": torch.randn(1, d_model)}
        source_mask = {"vision": "real"}

        loss_without = lm.compute(predictions, targets, source_mask)

        # Now with encoder_outputs (collapsed — would add loss if enabled)
        encoder_outputs = {"vision": torch.ones(8, 1, d_model)}
        loss_with = lm.compute(predictions, targets, source_mask,
                               encoder_outputs=encoder_outputs)

        assert abs(loss_with.item() - loss_without.item()) < 1e-6


# =========================================================================
# Test 6: Batch size 1 → gracefully returns 0
# =========================================================================

class TestBatchSizeOne:
    def test_batch_one_returns_zero(self):
        lm = _make_loss_manager(vicreg_enabled=True)
        z = torch.randn(1, 1, 512)  # B=1
        encoder_outputs = {"vision": z}

        loss = lm.compute_vicreg(encoder_outputs)
        assert loss.item() == 0.0, f"B=1 should return 0, got {loss.item()}"
        assert not torch.isnan(loss), "Should not produce NaN"


# =========================================================================
# Test 7: Per-modality → each gets own terms, averaged
# =========================================================================

class TestPerModality:
    def test_multi_modality_averaged(self):
        lm = _make_loss_manager(vicreg_enabled=True)
        torch.manual_seed(42)

        # Vision: collapsed (high var loss)
        collapsed = torch.ones(8, 1, 512)
        # Audio: spread (low var loss)
        spread = torch.randn(8, 1, 512)

        # Single modality losses
        loss_collapsed_only = lm.compute_vicreg({"vision": collapsed})
        loss_spread_only = lm.compute_vicreg({"audio": spread})

        # Combined: should be average of both
        loss_combined = lm.compute_vicreg({"vision": collapsed, "audio": spread})

        # Combined should be between the two individual losses
        low = min(loss_collapsed_only.item(), loss_spread_only.item())
        high = max(loss_collapsed_only.item(), loss_spread_only.item())
        assert low <= loss_combined.item() <= high + 1e-6, (
            f"Combined ({loss_combined.item()}) should be between "
            f"{low} and {high}"
        )

    def test_multi_modality_not_just_first(self):
        """Ensure all modalities contribute, not just the first."""
        lm = _make_loss_manager(vicreg_enabled=True)

        # Both collapsed → same high loss whether 1 or 2 modalities
        collapsed = torch.ones(8, 1, 512)
        loss_one = lm.compute_vicreg({"vision": collapsed})
        loss_two = lm.compute_vicreg({"vision": collapsed, "audio": collapsed.clone()})

        # Should be approximately equal (same input for both)
        assert abs(loss_one.item() - loss_two.item()) < 1e-5


# =========================================================================
# Test 8: Integration — through LossManager.compute()
# =========================================================================

class TestIntegration:
    def test_vicreg_added_to_total_loss(self):
        lm = _make_loss_manager(vicreg_enabled=True)
        d_model = 512

        predictions = {"vision": torch.randn(8, d_model)}
        targets = {"vision": torch.randn(8, d_model)}
        source_mask = {"vision": "real"}

        # Without encoder_outputs
        loss_base = lm.compute(predictions, targets, source_mask)

        # With collapsed encoder_outputs → adds VICReg penalty
        collapsed = torch.ones(8, 1, d_model)
        loss_with_vicreg = lm.compute(
            predictions, targets, source_mask,
            encoder_outputs={"vision": collapsed},
        )

        assert loss_with_vicreg.item() > loss_base.item(), (
            f"VICReg should increase total loss: "
            f"base={loss_base.item()}, with_vicreg={loss_with_vicreg.item()}"
        )

    def test_vicreg_gradients_flow(self):
        """Ensure VICReg loss contributes to encoder gradients."""
        lm = _make_loss_manager(vicreg_enabled=True)
        d_model = 512

        # Encoder output with requires_grad
        z = torch.randn(8, 1, d_model, requires_grad=True)
        predictions = {"vision": torch.randn(8, d_model)}
        targets = {"vision": torch.randn(8, d_model)}
        source_mask = {"vision": "real"}

        loss = lm.compute(
            predictions, targets, source_mask,
            encoder_outputs={"vision": z},
        )
        loss.backward()

        assert z.grad is not None, "Gradients should flow through VICReg to encoder outputs"
        assert z.grad.abs().sum() > 0, "Gradients should be nonzero"


# ---------------------------------------------------------------------------
# Test 10: Temporal VICReg — batch 1 support (2026-08-19)
# ---------------------------------------------------------------------------
class TestTemporalVICReg:
    def test_batch_one_temporal_flat_tokens_penalized(self):
        """With temporal_window > 0 a batch of 1 compares against recent frames:
        the first call has no history (0), the same token every frame is a collapse
        and gets a positive variance loss; a token that changes each frame gets less."""
        lm = _make_loss_manager(vicreg_enabled=True, temporal_window=4, covariance_weight=0.0)
        z = torch.randn(1, 1, 512)
        first = lm.compute_vicreg({"vision": z})
        assert first.item() == 0.0
        flat = None
        for _ in range(4):
            flat = lm.compute_vicreg({"vision": z.clone().requires_grad_(True)})
        assert flat.item() > 0.0
        flat.backward()   # gradient reaches the current token
        lm2 = _make_loss_manager(vicreg_enabled=True, temporal_window=4, covariance_weight=0.0)
        varied = None
        for _ in range(5):
            varied = lm2.compute_vicreg({"vision": torch.randn(1, 1, 512) * 3.0})
        assert varied.item() < flat.item()

    def test_batch_one_window_zero_still_returns_zero(self):
        lm = _make_loss_manager(vicreg_enabled=True, temporal_window=0)
        for _ in range(3):
            loss = lm.compute_vicreg({"vision": torch.randn(1, 1, 512)})
        assert loss.item() == 0.0

    def test_batch_two_unaffected_by_window(self):
        lm = _make_loss_manager(vicreg_enabled=True, temporal_window=4)
        z = torch.randn(2, 1, 512)
        a = lm.compute_vicreg({"vision": z})
        lm2 = _make_loss_manager(vicreg_enabled=True, temporal_window=0)
        b = lm2.compute_vicreg({"vision": z})
        assert torch.allclose(a, b)
