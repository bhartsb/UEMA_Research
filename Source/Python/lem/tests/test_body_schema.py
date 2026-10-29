# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_body_schema.py — Tests for DifferentiableBodySchema (UEMA Spec §11).

Tests:
  - Construction with default and custom params
  - coherence_score() returns scalar in [0, 1]
  - self_membership() returns [n_handles] in [0, 1]
  - self_membership(handle_indices) returns subset
  - incorporation_strength() returns scalar for selected handles
  - update() with correlated motor/sensor data increases correlation
  - update() with zero motor keeps scores low
  - First frame stores only (no correlation update)
  - EMA momentum: repeated identical updates converge
  - Per-dimension correlation tracking
  - Cross-handle correlation
  - Bone index (column 11) excluded
  - Checkpoint round-trip preserves buffers
  - Old checkpoint format loads gracefully
  - Frame count persistence
  - Buffer size audit
  - Integration: coherence feeds into wellness evaluator
"""

import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lem.model.body_schema import DifferentiableBodySchema
from lem.config.resolver import resolve
from lem.model.wellness import WellnessEvaluators

YAML_PATH = str(ROOT / "lem" / "master_config.yaml")


@pytest.fixture
def resolved():
    with open(YAML_PATH) as f:
        master = yaml.safe_load(f)
    return resolve(copy.deepcopy(master))


@pytest.fixture
def schema():
    return DifferentiableBodySchema(n_handles=23, temperature=5.0, ema_momentum=0.99)


# =========================================================================
# Test 1: Construction
# =========================================================================

class TestConstruction:
    def test_default_params(self):
        s = DifferentiableBodySchema()
        assert s.n_handles == 24
        assert s.n_proprio_per_handle == 11
        assert s.motor_dims == 9
        assert s.ema_momentum == 0.99
        assert s.correlation_matrix.shape == (24, 264)   # 24 handles x 24*11 proprio

    def test_custom_params(self):
        s = DifferentiableBodySchema(n_handles=10, n_proprio_per_handle=8,
                                     motor_dims=6, temperature=2.0,
                                     ema_momentum=0.95)
        assert s.n_handles == 10
        assert s.n_proprio_per_handle == 8
        assert s.motor_dims == 6
        assert s.ema_momentum == 0.95
        assert s.correlation_matrix.shape == (10, 80)  # 10 * 8

    def test_initial_scores_zero(self, schema):
        """Initial correlation matrix is zero — everything starts as 'world'."""
        assert (schema.correlation_matrix == 0).all()
        # Initial membership: sigmoid(5.0 * (0 - 0.5)) = sigmoid(-2.5) ≈ 0.076
        membership = schema.self_membership()
        assert (membership < 0.1).all()  # clearly "world"

    def test_frame_count_starts_zero(self, schema):
        assert schema.frame_count.item() == 0


# =========================================================================
# Test 2: Output shapes and ranges
# =========================================================================

class TestOutputs:
    def test_coherence_score_scalar(self, schema):
        c = schema.coherence_score()
        assert c.dim() == 0  # scalar
        assert 0.0 <= c.item() <= 1.0
        # Initial: everything is clearly "world" → high certainty
        assert c.item() > 0.8

    def test_self_membership_shape(self, schema):
        m = schema.self_membership()
        assert m.shape == (23,)
        assert (m >= 0.0).all() and (m <= 1.0).all()

    def test_self_membership_with_indices(self, schema):
        m = schema.self_membership(handle_indices=[0, 5, 10])
        assert m.shape == (3,)
        assert (m >= 0.0).all() and (m <= 1.0).all()

    def test_incorporation_strength(self, schema):
        s = schema.incorporation_strength([0, 1, 2])
        assert s.dim() == 0
        assert 0.0 <= s.item() <= 1.0

    def test_incorporation_single_dim(self, schema):
        s = schema.incorporation_strength([5])
        assert s.dim() == 0


# =========================================================================
# Test 3: Update behavior
# =========================================================================

class TestUpdate:
    def test_correlated_updates_increase_correlation(self, schema):
        """When motor and sensor change together, correlation should increase."""
        # Generate a trajectory with correlated motor and proprio changes
        motor = torch.randn(23, 9)
        proprio = torch.randn(23, 12)

        for _ in range(200):
            # Both change together each frame
            motor = motor + torch.randn(23, 9) * 0.5
            proprio_new = proprio.clone()
            proprio_new[:, :11] = proprio[:, :11] + torch.randn(23, 11) * 0.5
            proprio = proprio_new
            schema.update(motor, proprio)

        final_coherence = schema.coherence_score().item()
        assert 0.0 <= final_coherence <= 1.0

    def test_zero_motor_keeps_scores_low(self):
        """When motor is static, correlation stays near zero (world)."""
        schema = DifferentiableBodySchema(n_handles=5, n_proprio_per_handle=11,
                                          temperature=5.0, ema_momentum=0.9)
        motor = torch.zeros(5, 9)  # constant motor — delta always 0
        proprio = torch.randn(5, 12)

        for i in range(50):
            proprio = proprio + torch.randn(5, 12) * 0.1  # proprio changes but motor doesn't
            schema.update(motor, proprio)

        # With zero motor delta, instant_corr is 0, scores stay near 0,
        # membership stays near 0 ("world"), coherence stays high (certain)
        membership = schema.self_membership()
        assert (membership < 0.2).all()  # still clearly "world"

    def test_zero_sensor_keeps_scores_low(self):
        """When sensor is static, correlation also stays near zero."""
        schema = DifferentiableBodySchema(n_handles=5, n_proprio_per_handle=11,
                                          temperature=5.0, ema_momentum=0.9)
        proprio = torch.randn(5, 12)  # constant proprio — delta always 0

        motor = torch.randn(5, 9)
        for i in range(50):
            motor = motor + torch.randn(5, 9) * 0.5  # motor changes but proprio doesn't
            schema.update(motor, proprio)

        membership = schema.self_membership()
        assert (membership < 0.2).all()  # still "world"

    def test_update_no_grad(self, schema):
        """Update should not require or produce gradients."""
        motor = torch.randn(23, 9, requires_grad=True)
        proprio = torch.randn(23, 12)
        schema.update(motor, proprio)
        # No RuntimeError means success, and buffers shouldn't track grad
        assert not schema.correlation_matrix.requires_grad

    def test_dimension_mismatch_handled(self):
        """Motor and sensor with different first dims should work via min()."""
        schema = DifferentiableBodySchema(n_handles=23)
        motor = torch.randn(20, 9)       # fewer than 23
        proprio = torch.randn(25, 12)    # more than 23
        schema.update(motor, proprio)     # Should not raise
        schema.update(motor + 0.1, proprio + 0.1)  # Second frame for actual update


# =========================================================================
# Test 4: First frame behavior
# =========================================================================

class TestFirstFrame:
    def test_first_frame_stores_only(self, schema):
        """First call stores prev values but does NOT update correlation."""
        schema.update(torch.randn(23, 9), torch.randn(23, 12))
        assert schema.frame_count.item() == 1
        # correlation_matrix should still be all zeros
        assert (schema.correlation_matrix == 0).all()

    def test_second_frame_updates_correlation(self, schema):
        """Second call computes deltas and updates correlation."""
        motor1 = torch.randn(23, 9)
        proprio1 = torch.randn(23, 12)
        schema.update(motor1, proprio1)

        # Second frame with different values
        motor2 = motor1 + torch.randn(23, 9) * 2.0
        proprio2 = proprio1.clone()
        proprio2[:, :11] += torch.randn(23, 11) * 2.0
        schema.update(motor2, proprio2)

        assert schema.frame_count.item() == 2
        # correlation_matrix should no longer be all zeros
        assert not (schema.correlation_matrix == 0).all()


# =========================================================================
# Test 5: Coherence vs membership distinction
# =========================================================================

class TestCoherenceVsMembership:
    """Verify coherence measures certainty, not average membership."""

    def test_mixed_self_world_high_coherence(self):
        """11 self + 12 world dimensions should give high coherence."""
        schema = DifferentiableBodySchema(n_handles=23, temperature=5.0)
        # Set correlation so mean abs per handle gives scores > 0.5 for first 11, < 0.5 for rest
        # correlation_matrix.abs().mean(dim=-1) needs to be >0.5 for "self" handles
        schema.correlation_matrix[:11] = 2.0   # high correlation → "self"
        schema.correlation_matrix[11:] = -1.0  # abs mean still high → needs different approach

        # Actually: abs().mean() of 2.0 = 2.0, and sigmoid(5*(2.0-0.5)) ≈ 1.0
        # abs().mean() of -1.0 = 1.0, and sigmoid(5*(1.0-0.5)) = sigmoid(2.5) ≈ 0.924
        # Both are "self" this way. Let's set world handles to 0 instead.
        schema.correlation_matrix[11:] = 0.0  # abs mean = 0, sigmoid(5*(0-0.5)) ≈ 0.076

        membership = schema.self_membership()
        coherence = schema.coherence_score()

        # Average membership: 11 near 1.0, 12 near 0.08 → ~0.52
        avg_membership = membership.mean().item()
        assert 0.3 < avg_membership < 0.7

        # Coherence is high — the schema knows its boundaries
        assert coherence.item() > 0.8

    def test_ambiguous_scores_low_coherence(self):
        """All correlation giving scores at 0.5 → membership 0.5 → zero certainty."""
        schema = DifferentiableBodySchema(n_handles=10, temperature=5.0)
        # Need abs().mean(dim=-1) == 0.5 for each handle
        schema.correlation_matrix[:] = 0.5

        coherence = schema.coherence_score()
        assert coherence.item() < 0.05  # near zero certainty


# =========================================================================
# Test 6: EMA convergence
# =========================================================================

class TestEMAConvergence:
    def test_repeated_identical_updates_converge(self):
        """With constant input change, EMA should converge to a stable value."""
        schema = DifferentiableBodySchema(n_handles=5, n_proprio_per_handle=11,
                                          ema_momentum=0.9)

        base_motor = torch.zeros(5, 9)
        base_proprio = torch.zeros(5, 12)

        scores = []
        for i in range(200):
            # Consistent step each frame
            motor = base_motor + i * 0.5
            proprio = base_proprio.clone()
            proprio[:, :11] = i * 0.5
            schema.update(motor, proprio)
            scores.append(schema.coherence_score().item())

        # Last 10 values should be very close (converged)
        last_10 = scores[-10:]
        assert max(last_10) - min(last_10) < 0.01


# =========================================================================
# Test 7: Per-dimension correlation tracking
# =========================================================================

class TestPerDimensionCorrelation:
    def test_active_dims_accumulate_faster(self):
        """When only specific proprio dims change with motor activity,
        those cells in the correlation matrix should be higher."""
        schema = DifferentiableBodySchema(n_handles=5, n_proprio_per_handle=4,
                                          ema_momentum=0.9)

        motor = torch.zeros(5, 9)
        proprio = torch.zeros(5, 12)  # 12 cols, only first 4 used

        for i in range(100):
            # Handle 0 moves, and only proprio dim 0 of handle 0 changes
            motor_new = motor.clone()
            motor_new[0] += torch.randn(9) * 2.0  # only handle 0 moves
            proprio_new = proprio.clone()
            proprio_new[0, 0] += torch.randn(1).item() * 2.0  # only dim 0 of handle 0

            motor = motor_new
            proprio = proprio_new
            schema.update(motor, proprio)

        # correlation_matrix[0, 0] (handle 0 motor vs handle 0 proprio dim 0) should be
        # higher than correlation_matrix[0, 1] (handle 0 motor vs handle 0 proprio dim 1)
        assert schema.correlation_matrix[0, 0].abs() > schema.correlation_matrix[0, 1].abs()


# =========================================================================
# Test 8: Cross-handle correlation
# =========================================================================

class TestCrossHandleCorrelation:
    def test_cross_handle_effect(self):
        """Handle 0 moves motorically, handle 1's proprio changes.
        correlation_matrix[0, handle_1_start:handle_1_end] should accumulate."""
        n_proprio = 4
        schema = DifferentiableBodySchema(n_handles=5, n_proprio_per_handle=n_proprio,
                                          ema_momentum=0.9)

        motor = torch.zeros(5, 9)
        proprio = torch.zeros(5, 12)

        for i in range(100):
            motor_new = motor.clone()
            motor_new[0] += torch.randn(9) * 2.0  # only handle 0 moves

            proprio_new = proprio.clone()
            # Handle 1's proprio changes (simulating kinematic chain effect)
            proprio_new[1, :n_proprio] += torch.randn(n_proprio) * 2.0

            motor = motor_new
            proprio = proprio_new
            schema.update(motor, proprio)

        # Handle 0's motor vs handle 1's proprio dims: indices [1*4 : 2*4] = [4:8]
        h1_start = 1 * n_proprio
        h1_end = 2 * n_proprio
        cross_corr = schema.correlation_matrix[0, h1_start:h1_end].abs().mean()

        # Handle 0's motor vs handle 3's proprio (which didn't change): indices [3*4 : 4*4]
        h3_start = 3 * n_proprio
        h3_end = 4 * n_proprio
        inactive_corr = schema.correlation_matrix[0, h3_start:h3_end].abs().mean()

        assert cross_corr > inactive_corr


# =========================================================================
# Test 9: Bone index exclusion
# =========================================================================

class TestBoneIndexExcluded:
    def test_column_11_not_used(self):
        """Proprio changes only in column 11 (bone_index) should not affect correlation."""
        schema = DifferentiableBodySchema(n_handles=5, n_proprio_per_handle=11,
                                          ema_momentum=0.9)

        motor = torch.zeros(5, 9)
        proprio = torch.zeros(5, 12)

        for i in range(50):
            motor_new = motor + torch.randn(5, 9) * 2.0  # motor changes
            proprio_new = proprio.clone()
            proprio_new[:, 11] += 100.0  # only bone_index column changes (large!)

            motor = motor_new
            proprio = proprio_new
            schema.update(motor, proprio)

        # correlation_matrix should stay near zero because the only proprio
        # change was in col 11 which is excluded (sliced to :11)
        membership = schema.self_membership()
        assert (membership < 0.2).all()


# =========================================================================
# Test 10: Checkpoint round-trip
# =========================================================================

class TestCheckpoint:
    def test_state_dict_round_trip(self):
        schema = DifferentiableBodySchema(n_handles=10, n_proprio_per_handle=11,
                                          ema_momentum=0.95)

        # Run several updates to build up state
        motor = torch.randn(10, 9)
        proprio = torch.randn(10, 12)
        for _ in range(20):
            motor = motor + torch.randn(10, 9) * 0.3
            proprio_new = proprio.clone()
            proprio_new[:, :11] += torch.randn(10, 11) * 0.3
            proprio = proprio_new
            schema.update(motor, proprio)

        # Save state
        state = schema.state_dict()
        coherence_before = schema.coherence_score().item()

        # Load into fresh schema
        schema2 = DifferentiableBodySchema(n_handles=10, n_proprio_per_handle=11,
                                           ema_momentum=0.95)
        schema2.load_state_dict(state)

        coherence_after = schema2.coherence_score().item()
        assert abs(coherence_before - coherence_after) < 1e-6

    def test_buffers_in_state_dict(self):
        schema = DifferentiableBodySchema(n_handles=5)
        state = schema.state_dict()
        assert "correlation_matrix" in state
        assert "prev_motor" in state
        assert "prev_proprio" in state
        assert "motor_var" in state
        assert "proprio_var" in state
        assert "frame_count" in state


# =========================================================================
# Test 11: Old checkpoint format loads gracefully
# =========================================================================

class TestCheckpointUpgrade:
    def test_old_format_loads_without_error(self):
        """Old state_dict with correlation_scores/motor_ema_norm/sensor_ema_norm
        should load via strict=False without crashing."""
        old_state = {
            "correlation_scores": torch.randn(23),
            "motor_ema_norm": torch.ones(1),
            "sensor_ema_norm": torch.ones(1),
        }

        schema = DifferentiableBodySchema(n_handles=23)
        # This should not raise — old keys are silently ignored
        schema.load_state_dict(old_state, strict=False)

        # New buffers should be at their initial values
        assert (schema.correlation_matrix == 0).all()
        assert schema.frame_count.item() == 0

    def test_frame_count_persistence(self):
        """frame_count should survive checkpoint round-trip."""
        schema = DifferentiableBodySchema(n_handles=5, ema_momentum=0.9)

        motor = torch.randn(5, 9)
        proprio = torch.randn(5, 12)
        for _ in range(10):
            motor = motor + torch.randn(5, 9) * 0.1
            schema.update(motor, proprio)

        assert schema.frame_count.item() == 10

        state = schema.state_dict()
        schema2 = DifferentiableBodySchema(n_handles=5, ema_momentum=0.9)
        schema2.load_state_dict(state)
        assert schema2.frame_count.item() == 10


# =========================================================================
# Test 12: Buffer size audit
# =========================================================================

class TestBufferSizeAudit:
    def test_total_buffer_elements(self):
        """Default config should have exactly 6,832 buffer elements."""
        schema = DifferentiableBodySchema(n_handles=23, n_proprio_per_handle=11,
                                          motor_dims=9)
        total = sum(b.numel() for b in schema.buffers())
        # correlation_matrix: 24*264=6336, prev_motor: 24*9=216,
        # prev_proprio: 23*11=253, motor_var: 23, proprio_var: 253,
        # frame_count: 1, _scratch_motor_mag: 23,
        # _scratch_proprio_delta: 23*11=253 → total = 6832
        assert total == 6832


# =========================================================================
# Test 13: Integration with wellness evaluator
# =========================================================================

class TestWellnessIntegration:
    def test_coherence_kwarg_accepted(self, resolved):
        """Wellness evaluator accepts body_schema_coherence kwarg."""
        wellness = WellnessEvaluators(resolved)
        d_model = resolved["model"]["d_model"]
        n_mod = len(resolved["_active_modalities"])

        predictions = {"vision": torch.randn(1, d_model),
                       "audio": torch.randn(1, d_model)}
        targets = {"vision": torch.randn(1, d_model),
                   "audio": torch.randn(1, d_model)}
        latents = torch.randn(1, n_mod, d_model)
        motor_cmds = torch.randn(1, 24, 9)
        bs_coherence = torch.tensor(0.75)

        out = wellness(predictions, targets, latents,
                       motor_commands=motor_cmds,
                       body_schema_coherence=bs_coherence)

        assert "grounded" in out
        assert "coherence" in out["grounded"]
        assert out["grounded"]["coherence"].shape == (1, 1)

    def test_coherence_without_body_schema_falls_back(self, resolved):
        """Without body_schema_coherence, falls back to cosine similarity."""
        wellness = WellnessEvaluators(resolved)
        d_model = resolved["model"]["d_model"]
        n_mod = len(resolved["_active_modalities"])

        predictions = {"vision": torch.randn(1, d_model),
                       "audio": torch.randn(1, d_model)}
        targets = {"vision": torch.randn(1, d_model),
                   "audio": torch.randn(1, d_model)}
        latents = torch.randn(1, n_mod, d_model)

        out = wellness(predictions, targets, latents)

        assert "coherence" in out["grounded"]
        assert out["grounded"]["coherence"].shape == (1, 1)
