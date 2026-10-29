# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tests for per-body-set PE decomposition (compute_domain_pe).

Validates that instrumentation is correct:
  - Produces values for all 9 non-Spine body sets
  - Spine returns None (no handles)
  - Consistent with global PE (same order of magnitude)
  - Runs under torch.no_grad (no gradient side effects)
  - Handles missing data gracefully
"""

import torch
import pytest

from lem.training_step import compute_domain_pe
from lem.model.body_sets import BodySet, BODY_SET_HANDLES, NUM_HANDLES


@pytest.fixture
def synthetic_batch():
    """Create a pair of synthetic batches (current + target) with known PE."""
    B = 4
    torch.manual_seed(42)
    batch = {
        "proprioception": torch.randn(B, 288),
        "motor": torch.randn(B, 24, 10),
    }
    # Target = current + small noise (so PE is small but non-zero)
    target_batch = {
        "proprioception": batch["proprioception"] + torch.randn(B, 288) * 0.1,
        "motor": batch["motor"] + torch.randn(B, 24, 10) * 0.1,
    }
    return batch, target_batch


def test_all_body_sets_present(synthetic_batch):
    """All 10 body sets appear in output."""
    batch, target_batch = synthetic_batch
    result = compute_domain_pe(batch, target_batch)
    assert result is not None
    assert set(result.keys()) == {bs.name for bs in BodySet}


def test_spine_is_none(synthetic_batch):
    """Spine has no handles, returns None."""
    batch, target_batch = synthetic_batch
    result = compute_domain_pe(batch, target_batch)
    assert result["Spine"] is None


def test_non_spine_are_float(synthetic_batch):
    """All non-Spine body sets return finite float values."""
    batch, target_batch = synthetic_batch
    result = compute_domain_pe(batch, target_batch)
    for bs in BodySet:
        if bs == BodySet.Spine:
            continue
        val = result[bs.name]
        assert isinstance(val, float), f"{bs.name} is {type(val)}"
        assert val >= 0.0, f"{bs.name}={val} is negative"
        assert val == val, f"{bs.name} is NaN"  # NaN check


def test_identical_frames_zero_pe():
    """PE should be zero when current and target frames are identical."""
    B = 2
    data = torch.randn(B, 288)
    motor = torch.randn(B, 24, 10)
    batch = {"proprioception": data, "motor": motor}
    target_batch = {"proprioception": data.clone(), "motor": motor.clone()}
    result = compute_domain_pe(batch, target_batch)
    for bs in BodySet:
        if bs == BodySet.Spine:
            continue
        assert result[bs.name] == 0.0, f"{bs.name}={result[bs.name]}"


def test_consistent_with_global_pe(synthetic_batch):
    """Mean of domain PEs should be similar order of magnitude to global MSE."""
    batch, target_batch = synthetic_batch
    result = compute_domain_pe(batch, target_batch)

    # Global proprio MSE
    global_pe = torch.nn.functional.mse_loss(
        batch["proprioception"], target_batch["proprioception"]).item()

    # Weighted mean of domain PEs (weighted by handle count)
    total_pe = 0.0
    total_handles = 0
    for bs in BodySet:
        if bs == BodySet.Spine:
            continue
        n = len(BODY_SET_HANDLES[bs])
        total_pe += result[bs.name] * n
        total_handles += n

    domain_avg = total_pe / total_handles if total_handles > 0 else 0.0

    # Should be within 10x of each other (not exact due to motor averaging)
    ratio = domain_avg / max(global_pe, 1e-10)
    assert 0.1 < ratio < 10.0, (
        f"domain_avg={domain_avg:.6f} vs global_pe={global_pe:.6f} "
        f"ratio={ratio:.2f}")


def test_no_gradient_side_effects(synthetic_batch):
    """compute_domain_pe must not create or modify gradients."""
    batch, target_batch = synthetic_batch

    # Create tensors that require grad
    batch["proprioception"] = batch["proprioception"].requires_grad_(True)
    batch["motor"] = batch["motor"].requires_grad_(True)

    result = compute_domain_pe(batch, target_batch)

    # No gradients should have been computed
    assert batch["proprioception"].grad is None
    assert batch["motor"].grad is None
    assert result is not None


def test_missing_proprio_returns_none():
    """Returns None if proprioception data is missing."""
    batch = {"motor": torch.randn(2, 24, 10)}
    target_batch = {"motor": torch.randn(2, 24, 10)}
    result = compute_domain_pe(batch, target_batch)
    assert result is None


def test_missing_motor_still_works():
    """Works with proprio only (no motor data)."""
    B = 2
    batch = {"proprioception": torch.randn(B, 288)}
    target_batch = {"proprioception": torch.randn(B, 288)}
    result = compute_domain_pe(batch, target_batch)
    assert result is not None
    for bs in BodySet:
        if bs == BodySet.Spine:
            assert result[bs.name] is None
        else:
            assert isinstance(result[bs.name], float)


def test_handle_count_coverage():
    """Verify all 24 handles are covered by the body set mapping."""
    covered = set()
    for bs in BodySet:
        covered.update(BODY_SET_HANDLES[bs])
    # Spine has no handles, so covered should be 0..22 minus Spine's empty set
    assert len(covered) == NUM_HANDLES, f"Only {len(covered)} of {NUM_HANDLES} handles covered"
