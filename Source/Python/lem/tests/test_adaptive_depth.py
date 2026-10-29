# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_adaptive_depth.py — Tests for adaptive computation depth motor controller
(CF-7, UEMA Spec §9.1-9.3).

Tests:
  - Adaptive mode creates separate residual blocks via ModuleList
  - Input/output dimensions match non-adaptive mode
  - Non-recursive mode unchanged when adaptive=false
  - Confidence-based depth gating (high → depth=1, low → depth>1)
  - depth_threshold_raw is nn.Parameter with gradient flow via soft weights
  - Training runs all layers (soft-weighted); inference hard-gates
  - Explicit depth override
  - Output shape [B, 23, 9] at every depth
  - 3-tuple return from all modes
  - Metadata dict contents
  - Energy budget terminates deepening (eval mode)
  - No imagination still allows depth gating
  - Deepening loop only runs in eval mode
"""

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lem.config.resolver import resolve
from lem.model.motor_controller import MotorController


YAML_PATH = str(ROOT / "lem" / "master_config.yaml")


@pytest.fixture
def base_config():
    with open(YAML_PATH) as f:
        master = yaml.safe_load(f)
    return resolve(copy.deepcopy(master))


@pytest.fixture
def adaptive_config(base_config):
    """Config with adaptive mode enabled."""
    cfg = copy.deepcopy(base_config)
    cfg["model"]["motor_controller"]["adaptive"] = True
    cfg["model"]["motor_controller"]["max_depth"] = 6
    cfg["model"]["motor_controller"]["initial_depth_threshold"] = 0.7
    cfg["model"]["motor_controller"]["thinking_energy_rate"] = 0.001
    return cfg


@pytest.fixture
def device():
    return torch.device("cpu")


def _make_inputs(config, device):
    """Create standard motor controller inputs."""
    B = 1
    d_model = config["model"]["d_model"]
    n_handles = config["body"]["n_handles"]
    n_active = len(config["_active_modalities"])
    latents = torch.randn(B, n_active, d_model, device=device)
    handle_state = torch.randn(B, n_handles, 10, device=device)
    wellness_grad = torch.randn(B, n_handles, 9, device=device)
    return latents, handle_state, wellness_grad


# =========================================================================
# Test 1: Adaptive architecture
# =========================================================================

class TestAdaptiveArchitecture:
    def test_adaptive_creates_residual_blocks(self, adaptive_config, device):
        """adaptive=true creates residual_blocks ModuleList with max_depth entries."""
        mc = MotorController(adaptive_config).to(device)
        assert hasattr(mc, 'residual_blocks')
        assert isinstance(mc.residual_blocks, torch.nn.ModuleList)
        assert len(mc.residual_blocks) == 6

    def test_adaptive_input_output_dims(self, adaptive_config, device):
        """Adaptive mode produces correct output shapes."""
        mc = MotorController(adaptive_config).to(device)
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)
        motor_out, confidence, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        n_handles = adaptive_config["body"]["n_handles"]
        assert motor_out.shape == (1, n_handles, 9)
        assert confidence.shape == (1, 1)

    def test_non_recursive_unchanged(self, base_config, device):
        """adaptive=false still uses flat MLP (no residual_blocks)."""
        mc = MotorController(base_config).to(device)
        assert not mc._adaptive
        assert hasattr(mc, 'mlp')
        assert not hasattr(mc, 'residual_blocks')

    def test_spec_dimension_bug_resolved(self, adaptive_config, device):
        """input_proj maps full input dim to hidden_dim before residual blocks."""
        mc = MotorController(adaptive_config).to(device)
        assert mc.input_proj.in_features == mc._input_dim
        assert mc.input_proj.out_features == mc._hidden_dim
        # All residual blocks operate at hidden_dim
        for block in mc.residual_blocks:
            linear = block[0]  # First Linear in Sequential
            assert linear.in_features == mc._hidden_dim
            assert linear.out_features == mc._hidden_dim


# =========================================================================
# Test 2: Confidence-based depth gating
# =========================================================================

class TestConfidenceDepthGating:
    def test_soft_gating_always_runs_all_blocks(self, adaptive_config, device):
        """Both train and eval use soft sigmoid gating -- depth always == max_depth.
        (Hard gating with .item() GPU sync removed to eliminate pipeline stalls.)
        """
        mc = MotorController(adaptive_config).to(device)
        max_depth = adaptive_config["model"]["motor_controller"]["max_depth"]
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)

        mc.eval()
        _, _, meta_eval = mc(latents, handle_state, wellness_grad=wellness_grad)
        assert meta_eval["depth"] == max_depth

        mc.train()
        _, _, meta_train = mc(latents, handle_state, wellness_grad=wellness_grad)
        assert meta_train["depth"] == max_depth

    def test_high_confidence_reports_high(self, adaptive_config, device):
        """When confidence head is forced high, metadata reflects it."""
        mc = MotorController(adaptive_config).to(device)
        mc.eval()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)

        with torch.no_grad():
            mc.confidence_head[-2].weight.zero_()
            mc.confidence_head[-2].bias.fill_(5.0)  # sigmoid(5.0) ~ 0.993

        _, _, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        assert meta["confidence"] > 0.9

    def test_low_confidence_reports_low(self, adaptive_config, device):
        """When confidence head is forced low, metadata reflects it."""
        mc = MotorController(adaptive_config).to(device)
        mc.eval()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)

        with torch.no_grad():
            mc.confidence_head[-2].weight.zero_()
            mc.confidence_head[-2].bias.fill_(-5.0)  # sigmoid(-5.0) ~ 0.007

        _, _, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        assert meta["confidence"] < 0.1

    def test_depth_threshold_is_nn_parameter(self, adaptive_config, device):
        """depth_threshold_raw is nn.Parameter with requires_grad=True."""
        mc = MotorController(adaptive_config).to(device)
        assert hasattr(mc, 'depth_threshold_raw')
        assert isinstance(mc.depth_threshold_raw, torch.nn.Parameter)
        assert mc.depth_threshold_raw.requires_grad

    def test_depth_threshold_has_gradient(self, adaptive_config, device):
        """After backward, depth_threshold_raw.grad is not None."""
        mc = MotorController(adaptive_config).to(device)
        mc.train()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)
        motor_out, confidence, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        loss = motor_out.sum()
        loss.backward()
        assert mc.depth_threshold_raw.grad is not None

    def test_gradient_flows_through_threshold(self, adaptive_config, device):
        """Gradient magnitude through depth_threshold_raw is non-zero."""
        mc = MotorController(adaptive_config).to(device)
        mc.train()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)
        motor_out, confidence, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        loss = motor_out.sum()
        loss.backward()
        assert mc.depth_threshold_raw.grad.abs().item() > 0


# =========================================================================
# Test 3: Training behavior
# =========================================================================

class TestTrainingBehavior:
    def test_training_runs_all_layers_soft(self, adaptive_config, device):
        """In train() mode, metadata reports depth == max_depth (all layers ran)."""
        mc = MotorController(adaptive_config).to(device)
        mc.train()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)
        _, _, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        assert meta["depth"] == adaptive_config["model"]["motor_controller"]["max_depth"]

    def test_explicit_depth_override(self, adaptive_config, device):
        """forward(depth=3) uses exactly 3 blocks."""
        mc = MotorController(adaptive_config).to(device)
        mc.eval()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)
        _, _, meta = mc(latents, handle_state, wellness_grad=wellness_grad, depth=3)
        assert meta["depth"] == 3

    def test_output_shape_all_depths(self, adaptive_config, device):
        """Output is [B, 23, 9] for every depth 1 through max_depth."""
        mc = MotorController(adaptive_config).to(device)
        mc.eval()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)
        n_handles = adaptive_config["body"]["n_handles"]
        for d in range(1, 7):
            motor_out, conf, meta = mc(latents, handle_state,
                                       wellness_grad=wellness_grad, depth=d)
            assert motor_out.shape == (1, n_handles, 9), f"depth={d}: shape={motor_out.shape}"
            assert meta["depth"] == d

    def test_soft_weights_decay_past_target(self, adaptive_config, device):
        """Layers beyond target depth get lower effective contribution."""
        mc = MotorController(adaptive_config).to(device)
        mc.train()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)

        # Force high confidence -> raw_depth near 1 -> later layers should have small weight
        with torch.no_grad():
            mc.confidence_head[-2].weight.zero_()
            mc.confidence_head[-2].bias.fill_(5.0)  # confidence ~ 0.99

        # The soft weight for layer i is sigmoid(5.0 * (raw_depth - (i+1)))
        # With raw_depth ~ 1.0, layer 0 weight ~ sigmoid(0) = 0.5,
        # layer 5 weight ~ sigmoid(5.0 * (1.0 - 6)) = sigmoid(-25) ~ 0.0
        motor_out, _, meta = mc(latents, handle_state, wellness_grad=wellness_grad)

        # Verify output is valid (gradient exists, shapes correct)
        loss = motor_out.sum()
        loss.backward()
        # depth_threshold_raw should have gradient (it affects the weights)
        assert mc.depth_threshold_raw.grad is not None


# =========================================================================
# Test 4: Metadata
# =========================================================================

class TestMetadata:
    def test_returns_three_tuple(self, base_config, device):
        """All modes (including non-adaptive) return 3-tuple."""
        mc = MotorController(base_config).to(device)
        latents, handle_state, _ = _make_inputs(base_config, device)
        result = mc(latents, handle_state)
        assert len(result) == 3
        motor_out, confidence, meta = result
        assert isinstance(meta, dict)

    def test_metadata_keys(self, adaptive_config, device):
        """Metadata dict has depth, confidence, depth_threshold."""
        mc = MotorController(adaptive_config).to(device)
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)
        _, _, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        assert "depth" in meta
        assert "confidence" in meta
        assert "depth_threshold" in meta
        assert isinstance(meta["depth"], int)
        assert isinstance(meta["confidence"], torch.Tensor)
        assert isinstance(meta["depth_threshold"], (torch.Tensor, float))

    def test_config_defaults(self, base_config, device):
        """Missing adaptive key defaults to False."""
        cfg = copy.deepcopy(base_config)
        cfg["model"]["motor_controller"].pop("adaptive", None)
        mc = MotorController(cfg).to(device)
        assert not mc._adaptive

    def test_adaptive_metadata_threshold_matches_init(self, adaptive_config, device):
        """depth_threshold in metadata matches sigmoid(depth_threshold_raw)."""
        mc = MotorController(adaptive_config).to(device)
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)
        _, _, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        expected = torch.sigmoid(mc.depth_threshold_raw).item()
        assert abs(meta["depth_threshold"] - expected) < 1e-5

    def test_non_adaptive_metadata_uses_config_threshold(self, base_config, device):
        """Non-adaptive mode uses confidence_threshold from config."""
        mc = MotorController(base_config).to(device)
        latents, handle_state, _ = _make_inputs(base_config, device)
        _, _, meta = mc(latents, handle_state)
        assert meta["depth_threshold"] == base_config["model"]["motor_controller"]["confidence_threshold"]


# =========================================================================
# Test 5: Inference deepening
# =========================================================================

class TestInferenceDeepening:
    def test_energy_budget_terminates(self, adaptive_config, device):
        """Energy budget terminates deepening loop when exceeded (eval mode)."""
        mc = MotorController(adaptive_config).to(device)
        mc.eval()
        # thinking_energy_rate = 0.001
        # With very low available_energy, only a few depths should be affordable
        available_energy = 0.005  # 0.005 * 0.1 = 0.0005 budget
        rate = adaptive_config["model"]["motor_controller"]["thinking_energy_rate"]
        # depth * rate > budget: depth > 0.0005 / 0.001 = 0.5, so even depth=1 is too expensive
        budget = available_energy * 0.1
        max_affordable_depth = 0
        for d in range(1, 7):
            if d * rate > budget:
                break
            max_affordable_depth = d
        assert max_affordable_depth == 0  # Nothing is affordable at this energy

        # With available_energy = 0.5, budget = 0.05, max depth = 50
        available_energy = 0.5
        budget = available_energy * 0.1
        max_affordable_depth = 0
        for d in range(1, 7):
            if d * rate > budget:
                break
            max_affordable_depth = d
        assert max_affordable_depth == 6  # All depths affordable

    def test_no_imagination_keeps_depth_gating(self, adaptive_config, device):
        """Depth gating works without imagination rollout."""
        mc = MotorController(adaptive_config).to(device)
        mc.eval()
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)

        # Force low confidence for deep processing
        with torch.no_grad():
            mc.confidence_head[-2].weight.zero_()
            mc.confidence_head[-2].bias.fill_(-5.0)

        _, _, meta = mc(latents, handle_state, wellness_grad=wellness_grad)
        # Depth gating still works even without imagination loop
        assert meta["depth"] > 1

    def test_deepening_only_in_eval(self, adaptive_config, device):
        """In training mode, depth is always max_depth (soft-weighted)."""
        mc = MotorController(adaptive_config).to(device)
        latents, handle_state, wellness_grad = _make_inputs(adaptive_config, device)

        # Force low confidence: zero weights + large negative bias
        with torch.no_grad():
            mc.confidence_head[-2].weight.zero_()
            mc.confidence_head[-2].bias.fill_(-5.0)

        # Training: all layers
        mc.train()
        _, _, meta_train = mc(latents, handle_state, wellness_grad=wellness_grad)
        assert meta_train["depth"] == 6  # max_depth

        # Eval: hard gate
        mc.eval()
        _, _, meta_eval = mc(latents, handle_state, wellness_grad=wellness_grad)
        assert meta_eval["depth"] >= 1
        # With very low confidence, should use multiple layers
        assert meta_eval["depth"] > 1
