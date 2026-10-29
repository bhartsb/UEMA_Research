# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_loss_manager.py — Tests for motor loss developmental scaling.

Tests:
  - Neonate developmental limits produce motor_mse > 0 (FORWARD mode)
  - Neonate developmental limits produce motor > 0 (BASELINE mode)
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
from lem.model.prediction_heads import VoiceHead
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


class TestMotorLossNeonateScaling:
    """Motor loss must remain positive during neonate phase (rom_limit=0.05)."""

    def _make_inputs(self, resolved, device):
        d_model = resolved["model"]["d_model"]
        n_handles = resolved["body"]["n_handles"]
        predictions = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        targets = {
            mod: torch.randn(1, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        source_mask = {mod: "real" for mod in resolved["_active_modalities"]}
        motor_out = torch.randn(1, n_handles, 9, device=device)
        motor_target = torch.randn(1, n_handles, 9, device=device)
        return predictions, targets, source_mask, motor_out, motor_target

    def test_neonate_motor_mse_positive_forward(self, resolved, device):
        """FORWARD mode: motor_mse > 0 with neonate developmental limits."""
        lm = LossManager(resolved, wellness_mode="forward")
        predictions, targets, source_mask, motor_out, motor_target = \
            self._make_inputs(resolved, device)

        dev_limits = {"rom_limit": 0.05, "strength_limit": 0.02}

        loss, bd = lm.compute(
            predictions, targets, source_mask,
            motor_out=motor_out, motor_target=motor_target,
            developmental_limits=dev_limits,
            return_breakdown=True,
        )

        assert "motor_mse" in bd, "motor_mse not in breakdown"
        assert bd["motor_mse"].item() > 0, \
            f"motor_mse should be positive during neonate phase, got {bd['motor_mse'].item()}"

    def test_neonate_motor_positive_baseline(self, resolved, device):
        """BASELINE mode: motor > 0 with neonate developmental limits."""
        lm = LossManager(resolved, wellness_mode="baseline")
        predictions, targets, source_mask, motor_out, motor_target = \
            self._make_inputs(resolved, device)

        dev_limits = {"rom_limit": 0.05, "strength_limit": 0.02}

        loss, bd = lm.compute(
            predictions, targets, source_mask,
            motor_out=motor_out, motor_target=motor_target,
            developmental_limits=dev_limits,
            return_breakdown=True,
        )

        assert "motor" in bd, "motor not in breakdown"
        assert bd["motor"].item() > 0, \
            f"motor should be positive during neonate phase, got {bd['motor'].item()}"
class TestVoiceHeadIsBuiltButNeverTaught:
    """The voice head facts the explainer page states: an 80-wide output held inside
    (0, 1) by a sigmoid, read out of the sense rows by ONE learned query - and no
    lesson that can move any of it. The page draws this box; nothing tested it."""

    def _head(self, resolved):
        c = resolved["model"]["voice_head"]
        return VoiceHead(d_model=resolved["model"]["d_model"],
                         hidden_dim=c["hidden_dim"], n_heads=c["n_heads"])

    def _loss_inputs(self, resolved, device, B=2):
        d_model = resolved["model"]["d_model"]
        # requires_grad so the total loss has a graph: the point of the gradient test
        # below is that the voice head is the part of it nothing reaches
        preds = {m: torch.randn(B, d_model, device=device, requires_grad=True)
                 for m in resolved["_prediction_head_modalities"]}
        tgts = {m: torch.randn(B, d_model, device=device)
                for m in resolved["_prediction_head_modalities"]}
        return preds, tgts, {m: "real" for m in resolved["_active_modalities"]}

    def test_output_is_80_wide_and_inside_zero_to_one(self, resolved, device):
        torch.manual_seed(0)
        head = self._head(resolved).to(device)
        n_mod = len(resolved["_active_modalities"])
        # deliberately far off the trained token scale: the sigmoid still bounds it
        last_step = torch.randn(3, n_mod, resolved["model"]["d_model"],
                                device=device) * 50.0
        out = head(last_step)
        assert out.shape == (3, resolved["model"]["voice_head"]["output_dim"])
        assert bool((out > 0).all()) and bool((out < 1).all()),             "the page says every control is held between 0 and 1"

    def test_one_learned_query_reads_every_sense_row(self, resolved, device):
        """One query vector, not one per sense - and it reads ALL the rows: moving
        any single sense row moves the controls."""
        torch.manual_seed(0)
        head = self._head(resolved).to(device)
        d_model = resolved["model"]["d_model"]
        n_mod = len(resolved["_active_modalities"])
        assert tuple(head.query.shape) == (1, 1, d_model),             "one learned query of d_model numbers, shared by every row"
        base = torch.randn(2, n_mod, d_model, device=device)
        out0 = head(base)
        for row in range(n_mod):
            moved = base.clone()
            moved[:, row] = moved[:, row] + 3.0
            delta = float((head(moved) - out0).abs().max())
            assert delta > 1e-4, f"sense row {row} never reaches the controls"

    def test_no_loss_term_is_built_for_it(self, resolved, device):
        """lem_model.py calls the loss with voice_target=None every step, so the
        voice branch never runs; and even given a target the configured weight is
        0.0, so the term contributes exactly nothing to the sum."""
        torch.manual_seed(0)
        head = self._head(resolved).to(device)
        n_mod = len(resolved["_active_modalities"])
        last_step = torch.randn(2, n_mod, resolved["model"]["d_model"], device=device)
        lm = LossManager(resolved, wellness_mode="baseline")
        preds, tgts, source_mask = self._loss_inputs(resolved, device)
        _, bd = lm.compute(preds, tgts, source_mask, voice_out=head(last_step),
                           voice_target=None, return_breakdown=True)
        assert "voice" in bd or resolved["model"]["voice_head"]["loss_weight"] == 0.0
        assert "voice" not in bd, "no target is ever supplied, so no voice term exists"
        _, bd2 = lm.compute(preds, tgts, source_mask, voice_out=head(last_step),
                            voice_target=torch.rand(2, 80, device=device),
                            return_breakdown=True)
        assert float(bd2["voice"]) == 0.0,             "the configured loss_weight is 0.0, so even a target buys no lesson"

    def test_no_gradient_reaches_its_weights(self, resolved, device):
        """The direct measurement behind the page's 'no lesson reaches it': run the
        head inside a real loss and back-propagate. Every one of its tensors comes
        back with no gradient at all - they sit in the optimizer's list and can
        never be stepped."""
        torch.manual_seed(0)
        head = self._head(resolved).to(device)
        n_mod = len(resolved["_active_modalities"])
        last_step = torch.randn(2, n_mod, resolved["model"]["d_model"], device=device)
        lm = LossManager(resolved, wellness_mode="baseline")
        preds, tgts, source_mask = self._loss_inputs(resolved, device)
        loss, _ = lm.compute(preds, tgts, source_mask, voice_out=head(last_step),
                             voice_target=None, return_breakdown=True)
        loss.backward()
        ungraded = [n for n, p in head.named_parameters() if p.grad is None]
        assert len(ungraded) == len(list(head.parameters())) > 0,             f"a lesson reached the voice head: {sorted(set(n for n, p in head.named_parameters()) - set(ungraded))}"
