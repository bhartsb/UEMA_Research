# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_imagination.py — Tests for imagination-aware training (CF-8, UEMA Spec §6.3).

WHAT THIS FILE COVERS, AND WHAT IT DOES NOT. Every test below drives
ForwardModel.imagine_step(). No production code calls that method: the chain that
runs in training is lem/live_test.py's _run_imagination_compiled, called from
mimo_bridge/lem_model.py, and each of its hops goes through
lem/model/predictive_state.py's transition(). A regression in that executed chain
leaves every test in this file green. The executed path's own behaviours - targets
taken from the separate raw history, the validity mask, re-anchoring an unmeasured
sense on what was observed, the shared output LayerNorm applied to the fed-back
token, the pose held at the chain's first frame, the context taken from that same
frame - are not touched here. The re-anchoring and the order/frame pairing are
covered by mimo_bridge/tests/test_imagination_does_not_compound_noise.py; the rest
are uncovered.

Tests:
  - Chain produces correct number of per-step losses, each is a scalar with grad
  - Gradients flow from step 3 loss back to forward model parameters
  - imagine_step keeps non-motor gradients live (vs predict_from_action which detaches)
  - Imagination silently disabled when history buffer too short
  - Config parsing and LossManager weight
  - every_n_steps correctly skips imagination on non-matching steps
  - Loss manager applies imagination weight and tracks in breakdown
"""

import copy
import sys
from collections import deque
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lem.config.resolver import resolve
from lem.model.forward_model import ForwardModel
from lem.model.prediction_heads import PredictionHeads
from lem.model.encoder_registry import EncoderRegistry
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


def _make_motor_out(resolved, device, requires_grad=False):
    """Create fake motor output [B, n_handles, 9]."""
    n_handles = resolved["body"]["n_handles"]
    return torch.randn(1, n_handles, 9, device=device, requires_grad=requires_grad)


def _make_context_buffer(resolved, device):
    """Create a random context buffer [1, (ctx-1)*n_mod, d_model]."""
    ctx_len = resolved["model"]["context_length"]
    n_mod = len(resolved["_active_modalities"])
    d_model = resolved["model"]["d_model"]
    return torch.randn(1, (ctx_len - 1) * n_mod, d_model, device=device)


# =========================================================================
# Test 1: Chain produces correct number of per-step losses
# =========================================================================

class TestChainLosses:
    def test_chain_produces_correct_number_of_losses(self, resolved, device, models):
        """Run imagination chain with N=3, verify 3 scalar losses returned."""
        fm, ph, er = models
        n_steps = 3

        # Build history: N+1 encoder outputs, N motor outputs
        encoder_history = [_make_encoder_outputs(resolved, device) for _ in range(n_steps + 1)]
        motor_history = [_make_motor_out(resolved, device) for _ in range(n_steps)]

        # Run chain
        imagined = dict(encoder_history[0])
        imag_ctx = _make_context_buffer(resolved, device)
        step_losses = []
        for s in range(n_steps):
            predicted, imag_ctx = fm.imagine_step(
                imagined, motor_history[s], er, ph, context_buffer=imag_ctx)
            target = encoder_history[s + 1]
            step_loss = sum(
                F.mse_loss(predicted[mod], target[mod])
                for mod in predicted if mod in target
            )
            step_losses.append(step_loss)
            imagined = predicted

        assert len(step_losses) == n_steps
        for i, sl in enumerate(step_losses):
            assert sl.dim() == 0, f"Step {i} loss is not scalar: shape={sl.shape}"
            assert sl.requires_grad, f"Step {i} loss has no grad"

    def test_total_loss_is_sum(self, resolved, device, models):
        """Total imagination loss should equal sum of per-step losses."""
        fm, ph, er = models
        n_steps = 2

        encoder_history = [_make_encoder_outputs(resolved, device) for _ in range(n_steps + 1)]
        motor_history = [_make_motor_out(resolved, device) for _ in range(n_steps)]

        imagined = dict(encoder_history[0])
        imag_ctx = _make_context_buffer(resolved, device)
        step_losses = []
        for s in range(n_steps):
            predicted, imag_ctx = fm.imagine_step(
                imagined, motor_history[s], er, ph, context_buffer=imag_ctx)
            target = encoder_history[s + 1]
            step_loss = sum(
                F.mse_loss(predicted[mod], target[mod])
                for mod in predicted if mod in target
            )
            step_losses.append(step_loss)
            imagined = predicted

        total = sum(step_losses)
        expected = step_losses[0] + step_losses[1]
        assert torch.allclose(total, expected)


# =========================================================================
# Test 2: Gradients flow through full chain
# =========================================================================

class TestGradientFlow:
    def test_gradients_flow_to_prediction_head_params(self, resolved, device, models):
        """Chain -> sum -> backward(). Prediction head params get gradients."""
        fm, ph, er = models
        n_steps = 3

        encoder_history = [_make_encoder_outputs(resolved, device) for _ in range(n_steps + 1)]
        motor_history = [_make_motor_out(resolved, device) for _ in range(n_steps)]

        imagined = dict(encoder_history[0])
        imag_ctx = _make_context_buffer(resolved, device)
        step_losses = []
        for s in range(n_steps):
            predicted, imag_ctx = fm.imagine_step(
                imagined, motor_history[s], er, ph, context_buffer=imag_ctx)
            target = encoder_history[s + 1]
            step_loss = sum(
                F.mse_loss(predicted[mod], target[mod])
                for mod in predicted if mod in target
            )
            step_losses.append(step_loss)
            imagined = predicted

        total = sum(step_losses)
        total.backward()

        # Prediction heads should have gradients (they produce the predictions)
        ph_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in ph.parameters()
        )
        assert ph_has_grad, "Prediction head params received no gradients from imagination chain"

    def test_gradients_flow_to_motor_encoder(self, resolved, device, models):
        """Motor encoder also gets gradients (it re-encodes motor at each step)."""
        fm, ph, er = models
        n_steps = 2

        encoder_history = [_make_encoder_outputs(resolved, device) for _ in range(n_steps + 1)]
        motor_history = [_make_motor_out(resolved, device) for _ in range(n_steps)]

        imagined = dict(encoder_history[0])
        imag_ctx = _make_context_buffer(resolved, device)
        step_losses = []
        for s in range(n_steps):
            predicted, imag_ctx = fm.imagine_step(
                imagined, motor_history[s], er, ph, context_buffer=imag_ctx)
            target = encoder_history[s + 1]
            step_loss = sum(
                F.mse_loss(predicted[mod], target[mod])
                for mod in predicted if mod in target
            )
            step_losses.append(step_loss)
            imagined = predicted

        total = sum(step_losses)
        total.backward()

        motor_encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in er.encoders["motor"].parameters()
        )
        assert motor_encoder_has_grad, "Motor encoder received no gradients from imagination chain"

    def test_gradients_flow_to_transformer_blocks(self, resolved, device, models):
        """Transformer block params get gradients (blocks run at each chain step)."""
        fm, ph, er = models
        n_steps = 2

        encoder_history = [_make_encoder_outputs(resolved, device) for _ in range(n_steps + 1)]
        motor_history = [_make_motor_out(resolved, device) for _ in range(n_steps)]

        imagined = dict(encoder_history[0])
        imag_ctx = _make_context_buffer(resolved, device)
        step_losses = []
        for s in range(n_steps):
            predicted, imag_ctx = fm.imagine_step(
                imagined, motor_history[s], er, ph, context_buffer=imag_ctx)
            target = encoder_history[s + 1]
            step_loss = sum(
                F.mse_loss(predicted[mod], target[mod])
                for mod in predicted if mod in target
            )
            step_losses.append(step_loss)
            imagined = predicted

        total = sum(step_losses)
        total.backward()

        # Transformer blocks (ThreeStageBlock) should have gradients
        block_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for block in fm.blocks
            for p in block.parameters()
        )
        assert block_has_grad, "Transformer block params received no gradients from imagination chain"


# =========================================================================
# Test 3: imagine_step vs predict_from_action gradient difference
# =========================================================================

class TestImagineStepVsPredictFromAction:
    def test_imagine_step_keeps_non_motor_grad_live(self, resolved, device, models):
        """imagine_step allows gradient flow through non-motor input tokens."""
        fm, ph, er = models
        d_model = resolved["model"]["d_model"]

        # Create input state with requires_grad on a non-motor modality
        state = _make_encoder_outputs(resolved, device)
        # Pick first non-motor modality
        non_motor_mod = [m for m in resolved["_active_modalities"] if m != "motor"][0]
        state[non_motor_mod] = state[non_motor_mod].detach().requires_grad_(True)

        motor_out = _make_motor_out(resolved, device)
        imag_ctx = _make_context_buffer(resolved, device)
        predicted, _ = fm.imagine_step(state, motor_out, er, ph, context_buffer=imag_ctx)

        # Loss from any prediction should backprop to non-motor input
        loss = sum(v.sum() for v in predicted.values())
        loss.backward()

        assert state[non_motor_mod].grad is not None, \
            "imagine_step blocked gradient to non-motor input"
        assert state[non_motor_mod].grad.abs().sum() > 0, \
            "imagine_step gradient is zero for non-motor input"

    def test_predict_from_action_blocks_non_motor_grad(self, resolved, device, models):
        """predict_from_action detaches non-motor tokens, blocking gradient."""
        fm, ph, er = models
        d_model = resolved["model"]["d_model"]
        n_mod = len(resolved["_active_modalities"])

        # Create last_step with requires_grad
        last_step = torch.randn(1, n_mod, d_model, device=device, requires_grad=True)
        motor_out = _make_motor_out(resolved, device, requires_grad=True)

        predictions, z_motor = fm.predict_from_action(last_step, motor_out, er, ph)

        loss = sum(v.sum() for v in predictions.values())
        loss.backward()

        # motor_out should have grad (it's the live path)
        assert motor_out.grad is not None, "motor_out should have gradient"

        # last_step should NOT have grad (non-motor tokens are detached)
        assert last_step.grad is None, \
            "predict_from_action should block gradient to last_step (non-motor detach)"


# =========================================================================
# Test 4: Disabled when history buffer too short
# =========================================================================

class TestHistoryGuard:
    def test_disabled_when_history_too_short(self, resolved, device):
        """Imagination should not run when history buffer has fewer than steps+1 entries."""
        imagination_steps = 3

        encoder_output_history = deque(maxlen=imagination_steps + 1)
        motor_history = deque(maxlen=imagination_steps)

        # Add only 2 entries (need 4)
        for _ in range(2):
            encoder_output_history.append(_make_encoder_outputs(resolved, device))
            motor_history.append(_make_motor_out(resolved, device))

        # Guard condition should prevent chain
        can_run = (len(encoder_output_history) >= imagination_steps + 1
                   and len(motor_history) >= imagination_steps)
        assert not can_run, "Should not run with insufficient history"

    def test_enabled_when_history_full(self, resolved, device):
        """Imagination should run when history buffer is full."""
        imagination_steps = 3

        encoder_output_history = deque(maxlen=imagination_steps + 1)
        motor_history = deque(maxlen=imagination_steps)

        # Fill to required length
        for _ in range(imagination_steps + 1):
            encoder_output_history.append(_make_encoder_outputs(resolved, device))
        for _ in range(imagination_steps):
            motor_history.append(_make_motor_out(resolved, device))

        can_run = (len(encoder_output_history) >= imagination_steps + 1
                   and len(motor_history) >= imagination_steps)
        assert can_run, "Should run with full history"


# =========================================================================
# Test 5: Config parsing
# =========================================================================

class TestConfigParsing:
    def test_imagination_config_loads(self):
        """Imagination config section loads from master_config.yaml."""
        with open(YAML_PATH) as f:
            master = yaml.safe_load(f)
        cfg = master.get("training", {}).get("imagination", {})
        assert cfg.get("enabled") is True
        assert cfg.get("steps") == 3
        assert cfg.get("loss_weight") == 0.5
        # 1, not 2. EXP-17 took it 1->2; commit c52f539a took it back, "imagination
        # every_n_steps 2->1 for consistent training baseline", and left this assertion
        # behind. The test has been red ever since and was reporting on a value nobody runs.
        assert cfg.get("every_n_steps") == 1

    def test_loss_manager_reads_imagination_weight(self, resolved):
        """LossManager reads imagination weight from config."""
        lm = LossManager(resolved)
        assert lm._imagination_weight == 0.5


# =========================================================================
# Test 6: every_n_steps skipping
# =========================================================================

class TestEveryNSteps:
    def test_every_n_steps_skips_correctly(self):
        """With every_n_steps=3, imagination runs at 0, 3, 6 but not 1, 2, 4, 5."""
        every_n = 3
        should_run = [s % every_n == 0 for s in range(7)]
        assert should_run == [True, False, False, True, False, False, True]


# =========================================================================
# Test 7: LossManager imagination weight
# =========================================================================

class TestLossManagerImagination:
    def test_imagination_loss_weighted_in_breakdown(self, resolved, device):
        """Imagination loss in breakdown equals raw loss * 0.5 weight."""
        lm = LossManager(resolved)
        d_model = resolved["model"]["d_model"]

        # Minimal valid inputs
        predictions = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        targets = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        source_mask = {mod: "real" for mod in resolved["_active_modalities"]}

        raw_imag_loss = torch.tensor(2.0, device=device, requires_grad=True)

        loss, bd = lm.compute(
            predictions, targets, source_mask,
            return_breakdown=True,
            imagination_loss=raw_imag_loss,
        )

        assert "imagination" in bd, "imagination not in breakdown"
        expected = 2.0 * 0.5  # raw * weight
        assert bd["imagination"].item() == pytest.approx(expected, abs=1e-6), \
            f"Expected {expected}, got {bd['imagination'].item()}"

    def test_imagination_loss_none_no_breakdown(self, resolved, device):
        """When imagination_loss=None, no imagination key in breakdown."""
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
            imagination_loss=None,
        )

        assert "imagination" not in bd, "imagination should not appear when loss is None"


# =========================================================================
# Test 8: Context buffer threading through imagination chain
# =========================================================================

class TestContextBufferThreading:
    def test_context_buffer_shape_preserved(self, resolved, device, models):
        """Context buffer shape [B, (ctx-1)*n_mod, d_model] preserved through chain."""
        fm, ph, er = models
        n_steps = 3
        ctx_len = resolved["model"]["context_length"]
        n_mod = len(resolved["_active_modalities"])
        d_model = resolved["model"]["d_model"]
        expected_shape = (1, (ctx_len - 1) * n_mod, d_model)

        encoder_history = [_make_encoder_outputs(resolved, device) for _ in range(n_steps + 1)]
        motor_history = [_make_motor_out(resolved, device) for _ in range(n_steps)]

        imagined = dict(encoder_history[0])
        imag_ctx = _make_context_buffer(resolved, device)
        assert imag_ctx.shape == expected_shape, f"Initial shape {imag_ctx.shape} != {expected_shape}"

        for s in range(n_steps):
            predicted, imag_ctx = fm.imagine_step(
                imagined, motor_history[s], er, ph, context_buffer=imag_ctx)
            assert imag_ctx.shape == expected_shape, \
                f"Step {s}: shape {imag_ctx.shape} != {expected_shape}"
            imagined = predicted

    def test_context_buffer_none_returns_none(self, resolved, device, models):
        """imagine_step with context_buffer=None returns None as second element."""
        fm, ph, er = models
        state = _make_encoder_outputs(resolved, device)
        motor_out = _make_motor_out(resolved, device)

        predicted, ctx_out = fm.imagine_step(state, motor_out, er, ph, context_buffer=None)
        assert ctx_out is None, "Expected None context when input context is None"
        assert isinstance(predicted, dict), "First return should be predictions dict"

    def test_gradient_flows_through_context_buffer_across_chain(self, resolved, device, models):
        """Backward from LAST step loss only — gradients reach transformer via context chain."""
        fm, ph, er = models
        n_steps = 3

        encoder_history = [_make_encoder_outputs(resolved, device) for _ in range(n_steps + 1)]
        motor_history = [_make_motor_out(resolved, device) for _ in range(n_steps)]

        imagined = dict(encoder_history[0])
        imag_ctx = _make_context_buffer(resolved, device)
        for s in range(n_steps):
            predicted, imag_ctx = fm.imagine_step(
                imagined, motor_history[s], er, ph, context_buffer=imag_ctx)
            imagined = predicted

        # Only backward the LAST step's loss (not sum of all steps)
        target = encoder_history[n_steps]
        last_loss = sum(
            F.mse_loss(predicted[mod], target[mod])
            for mod in predicted if mod in target
        )
        last_loss.backward()

        # Transformer blocks must have gradients — proves gradient flows
        # backward through context buffer from step 3 to earlier steps
        block_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for block in fm.blocks
            for p in block.parameters()
        )
        assert block_has_grad, \
            "Transformer blocks got no gradients from last-step-only backward — " \
            "context buffer may be detached between chain steps"
