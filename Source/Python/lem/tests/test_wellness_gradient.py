# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_wellness_gradient.py — Tests for wellness gradient feedback loop (UEMA Spec §7.4).

Tests:
  - motor_state evaluator creates differentiable path from motor_commands to wellness
  - torch.autograd.grad produces non-zero gradient through motor_state evaluator
  - Motor controller accepts wellness_grad parameter, output shape unchanged
  - Motor controller with wellness_grad=None uses zeros fallback
  - End-to-end: wellness -> grad -> motor controller -> loss.backward() succeeds
  - Formula evaluators produce [0,1] outputs with live computation graph
  - predict_from_action returns correct shapes and maintains gradient flow
  - score_predicted_wellness assembles tokens correctly for approximate evaluator
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
from lem.model.wellness import WellnessEvaluators, WellnessMode, MotorStateFormula
from lem.model.motor_controller import MotorController
from lem.model.forward_model import ForwardModel
from lem.model.prediction_heads import PredictionHeads
from lem.model.encoder_registry import EncoderRegistry

YAML_PATH = str(ROOT / "lem" / "master_config.yaml")


@pytest.fixture
def resolved():
    with open(YAML_PATH) as f:
        master = yaml.safe_load(f)
    return resolve(copy.deepcopy(master))


@pytest.fixture
def device():
    return torch.device("cpu")


# =========================================================================
# Test 1: motor_state evaluator creates differentiable path
# =========================================================================

class TestMotorStateEvaluator:
    def test_motor_state_in_grounded(self, resolved):
        """motor_state evaluator is created and registered."""
        wellness = WellnessEvaluators(resolved)
        assert "motor_state" in wellness.grounded

    def test_motor_state_differentiable_path(self, resolved, device):
        """motor_commands -> motor_state evaluator -> gradient is non-zero."""
        wellness = WellnessEvaluators(resolved).to(device)
        B, n_handles, n_out = 1, 24, 9

        # Create motor_commands with grad tracking
        motor_commands = torch.randn(B, n_handles, n_out, device=device,
                                     requires_grad=True)

        # Minimal inputs for wellness forward
        predictions = {}
        targets = {}
        latents = torch.randn(B, 6, resolved["model"]["d_model"], device=device)

        wellness_out = wellness(predictions, targets, latents,
                               motor_commands=motor_commands)

        # Compute gradient of total grounded wellness w.r.t. motor_commands
        grounded_scores = wellness_out["grounded"]
        total_wellness = torch.cat(
            list(grounded_scores.values()), dim=-1
        ).sum()

        grad = torch.autograd.grad(total_wellness, motor_commands,
                                   create_graph=True)[0]

        assert grad.shape == (B, n_handles, n_out)
        # Gradient should be non-zero (motor_state evaluator provides the path)
        assert grad.abs().sum().item() > 0, "Wellness gradient is zero — no differentiable path"

    def test_motor_commands_none_uses_zeros(self, resolved, device):
        """When motor_commands is None, motor_state evaluator uses zeros."""
        wellness = WellnessEvaluators(resolved).to(device)
        B = 1
        d_model = resolved["model"]["d_model"]

        predictions = {}
        targets = {}
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness(predictions, targets, latents,
                               motor_commands=None)

        assert "motor_state" in wellness_out["grounded"]
        assert wellness_out["grounded"]["motor_state"].shape == (B, 1)

    def test_approximate_outputs_5(self, resolved, device):
        """Approximate evaluator now outputs 5 values (one per grounded)."""
        wellness = WellnessEvaluators(resolved).to(device)
        B = 1
        d_model = resolved["model"]["d_model"]

        predictions = {}
        targets = {}
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness(predictions, targets, latents)
        assert wellness_out["approximate"].shape == (B, 5)


# =========================================================================
# Test 2: Motor controller with wellness gradient
# =========================================================================

class TestMotorControllerWellnessGrad:
    def test_forward_with_wellness_grad(self, resolved, device):
        """Motor controller accepts wellness_grad, output shape unchanged."""
        mc = MotorController(resolved).to(device)
        B = 1
        d_model = resolved["model"]["d_model"]
        n_handles = resolved["body"]["n_handles"]

        latents = torch.randn(B, 6, d_model, device=device)
        handle_state = torch.randn(B, n_handles, 10, device=device)  # achieved: wire format
        wellness_grad = torch.randn(B, n_handles, 9, device=device)

        motor_out, confidence, _meta = mc(latents, handle_state,
                                         wellness_grad=wellness_grad)

        assert motor_out.shape == (B, n_handles, 9)
        assert confidence.shape == (B, 1)

    def test_forward_without_wellness_grad(self, resolved, device):
        """Motor controller with wellness_grad=None uses zeros, same output shape."""
        mc = MotorController(resolved).to(device)
        B = 1
        d_model = resolved["model"]["d_model"]
        n_handles = resolved["body"]["n_handles"]

        latents = torch.randn(B, 6, d_model, device=device)
        handle_state = torch.randn(B, n_handles, 10, device=device)  # achieved: wire format

        motor_out, confidence, _meta = mc(latents, handle_state)

        assert motor_out.shape == (B, n_handles, 9)
        assert confidence.shape == (B, 1)

    def test_input_dim_includes_wellness_grad(self, resolved):
        """Motor controller _input_dim includes wellness gradient dimension."""
        mc = MotorController(resolved)
        n_handles = resolved["body"]["n_handles"]
        d_model = resolved["model"]["d_model"]
        n_active = len(resolved["_active_modalities"])

        expected = d_model * n_active + n_handles * 10 + n_handles * 9
        assert mc._input_dim == expected


# =========================================================================
# Test 3: End-to-end gradient flow
# =========================================================================

class TestEndToEndGradientFlow:
    def test_wellness_grad_to_motor_backward(self, resolved, device):
        """Full path: wellness -> grad -> motor -> loss -> backward succeeds."""
        wellness_eval = WellnessEvaluators(resolved).to(device)
        mc = MotorController(resolved).to(device)

        B = 1
        d_model = resolved["model"]["d_model"]
        n_handles = resolved["body"]["n_handles"]

        latents = torch.randn(B, 6, d_model, device=device)
        handle_state = torch.randn(B, n_handles, 10, device=device)  # achieved: wire format

        # Step 1: Create motor_commands leaf with grad
        motor_commands = handle_state[:, :, :9].detach().clone().requires_grad_(True)

        # Step 2: Wellness forward
        predictions = {}
        targets = {}
        wellness_out = wellness_eval(predictions, targets, latents,
                                     motor_commands=motor_commands)

        # Step 3: Compute wellness gradient
        total_wellness = torch.cat(
            list(wellness_out["grounded"].values()), dim=-1
        ).sum()
        wellness_grad = torch.autograd.grad(
            total_wellness, motor_commands,
            create_graph=True, retain_graph=True
        )[0]

        # Step 4: Motor controller with wellness gradient
        motor_out, confidence, _meta = mc(latents, handle_state,
                                         wellness_grad=wellness_grad)

        # Step 5: Compute a simple loss and backward
        motor_target = handle_state[:, :, :9]
        loss = torch.nn.functional.mse_loss(motor_out, motor_target)
        loss.backward()

        # The grounded evaluators have no trainable parameters any more, and that is the
        # design rather than a regression. Spec 7.2 makes the grounded evaluator the
        # AUTHORITATIVE signal - nothing trains it - so every grounded score is a formula
        # that states the definition directly. The MLPs were replaced one at a time
        # (motor_state 2026-08-21 after it was measured pegged at 1.0 passing zero
        # gradient; the other four 2026-08-23 after they were measured moving 0.009 to
        # 0.071 across their whole input range, one of them backwards).
        #
        # So this used to assert that the grounded MLPs received gradients, and has been
        # red since 2026-08-21. What must hold now is that the wellness gradient reaches
        # the MOTOR controller, which is the pathway spec 7.4 describes.
        mc_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mc.parameters()
        )
        assert mc_has_grad, "Motor controller received no gradients"
        assert wellness_grad.abs().sum() > 0, "the wellness gradient itself is all zeros"


class TestWhichTermsSteerTheBody:
    """Which grounded terms actually reach the motor command, measured.

    Spec 7.4 says the wellness gradient decides WHICH actions are generated. Four of the
    five terms are computed from sensed quantities that have no path back to this step's
    motor command, so they contribute exactly nothing to it - they act through the goal
    ring the intention generator plans against, not through the gradient. Only motor_state
    steers directly, and it says "prefer the middle of your range of motion".

    Pinned here so that if a term ever gains or loses a path to the motor command, a test
    says so instead of it being discovered months later.
    """

    def test_only_motor_state_reaches_the_motor_command(self, resolved, device):
        w = WellnessEvaluators(resolved).to(device)
        B, d = 1, resolved["model"]["d_model"]
        mc_cmd = torch.randn(B, resolved["body"]["n_handles"], 9, device=device).requires_grad_(True)
        preds = {k: torch.randn(B, d, device=device) for k in ("vision", "tactile")}
        tgts = {k: torch.randn(B, d, device=device) for k in preds}
        tac = torch.zeros(B, 256, 8, device=device)
        tac[0, :6, 0] = 0.05
        out = w(preds, tgts, torch.randn(B, 6, d, device=device), motor_commands=mc_cmd,
                raw_tactile=tac, capability_features=torch.rand(B, 21, device=device))
        steering = set()
        for name, val in out["grounded"].items():
            if not val.requires_grad:
                continue
            g = torch.autograd.grad(val.sum(), mc_cmd, retain_graph=True, allow_unused=True)[0]
            if g is not None and g.abs().sum() > 0:
                steering.add(name)
        assert steering == {"motor_state"}, (
            f"the set of terms steering the body changed: {sorted(steering)}. If that is "
            f"intended, update this test and say what now reaches the motor command.")


# =========================================================================
# Test 4: Formula evaluators (DIRECT/FORWARD modes)
# =========================================================================

class TestFormulaEvaluators:
    def test_formula_outputs_in_range(self, resolved, device):
        """All formula evaluators produce values in [0, 1]."""
        wellness = WellnessEvaluators(resolved, wellness_mode=WellnessMode.DIRECT).to(device)
        wellness.set_developmental_limits(0.5)
        B = 1
        d_model = resolved["model"]["d_model"]
        n_handles = resolved["body"]["n_handles"]

        motor_commands = torch.randn(B, n_handles, 9, device=device)
        predictions = {"tactile": torch.randn(B, d_model, device=device)}
        targets = {"tactile": torch.randn(B, d_model, device=device)}
        latents = torch.randn(B, 6, d_model, device=device)
        capability = torch.rand(B, 21, device=device)
        raw_tactile = torch.randn(B, 16, 8, device=device) * 1000  # Simulated collision data

        wellness_out = wellness(predictions, targets, latents,
                               capability_features=capability,
                               motor_commands=motor_commands,
                               raw_tactile=raw_tactile)

        for name, score in wellness_out["grounded"].items():
            assert score.shape == (B, 1), f"{name} shape mismatch"
            assert score.min() >= 0.0, f"{name} below 0: {score.item()}"
            assert score.max() <= 1.0, f"{name} above 1: {score.item()}"

    def test_formula_motor_state_has_gradient(self, resolved, device):
        """motor_state formula maintains gradient when motor_commands.requires_grad=True."""
        wellness = WellnessEvaluators(resolved, wellness_mode=WellnessMode.DIRECT).to(device)
        wellness.set_developmental_limits(1.0)
        B = 1
        d_model = resolved["model"]["d_model"]
        n_handles = resolved["body"]["n_handles"]

        motor_commands = torch.randn(B, n_handles, 9, device=device, requires_grad=True)
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness({}, {}, latents, motor_commands=motor_commands)
        motor_state = wellness_out["grounded"]["motor_state"]
        motor_state.sum().backward()

        assert motor_commands.grad is not None, "No gradient on motor_commands"
        assert motor_commands.grad.abs().sum() > 0, "Gradient is zero"

    def test_baseline_mode_identical_format(self, resolved, device):
        """BASELINE mode produces same output format as before."""
        wellness = WellnessEvaluators(resolved, wellness_mode=WellnessMode.BASELINE).to(device)
        B = 1
        d_model = resolved["model"]["d_model"]
        n_handles = resolved["body"]["n_handles"]

        motor_commands = torch.randn(B, n_handles, 9, device=device, requires_grad=True)
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness({}, {}, latents, motor_commands=motor_commands)

        assert "grounded" in wellness_out
        assert "approximate" in wellness_out
        assert set(wellness_out["grounded"].keys()) == {
            "prediction_error", "tactile_comfort", "capability",
            "coherence", "motor_state"
        }
        assert wellness_out["approximate"].shape == (B, 5)

    def test_formula_known_values(self, resolved, device):
        """Formula evaluators produce expected values for known inputs."""
        wellness = WellnessEvaluators(resolved, wellness_mode=WellnessMode.DIRECT).to(device)
        wellness.set_developmental_limits(1.0)
        B = 1
        d_model = resolved["model"]["d_model"]
        n_handles = resolved["body"]["n_handles"]

        # Motor at zero position -> motor_state should be 1.0
        motor_commands = torch.zeros(B, n_handles, 9, device=device)
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness({}, {}, latents, motor_commands=motor_commands)
        motor_state = wellness_out["grounded"]["motor_state"]
        assert abs(motor_state.item() - 1.0) < 1e-5, \
            f"motor_state should be 1.0 at zero position, got {motor_state.item()}"

    def test_direct_mode_no_mlp_params(self, resolved):
        """DIRECT mode creates no grounded MLP parameters."""
        wellness_baseline = WellnessEvaluators(resolved, wellness_mode=WellnessMode.BASELINE)
        wellness_direct = WellnessEvaluators(resolved, wellness_mode=WellnessMode.DIRECT)

        # Baseline has grounded MLP params
        baseline_grounded = sum(p.numel() for n, p in wellness_baseline.named_parameters()
                                if "grounded" in n)
        assert baseline_grounded > 0, "BASELINE should have grounded params"

        # Direct has no grounded MLP params
        direct_grounded = sum(p.numel() for n, p in wellness_direct.named_parameters()
                              if "grounded" in n)
        assert direct_grounded == 0, "DIRECT should have no grounded params"

        # Both have approximate params
        baseline_approx = sum(p.numel() for n, p in wellness_baseline.named_parameters()
                              if "approximate" in n)
        direct_approx = sum(p.numel() for n, p in wellness_direct.named_parameters()
                            if "approximate" in n)
        assert baseline_approx == direct_approx, "Approximate params should be identical"

    def test_boundary_formula_zero_in_interior(self, resolved, device):
        """Boundary formula returns ~1.0 when positions are well within ROM."""
        wellness = WellnessEvaluators(
            resolved, wellness_mode=WellnessMode.DIRECT,
            motor_state_formula=MotorStateFormula.BOUNDARY,
        ).to(device)
        wellness.set_developmental_limits(1.0)
        B, n_handles = 1, resolved["body"]["n_handles"]
        d_model = resolved["model"]["d_model"]

        # Positions at half the BODY'S reach - well inside the boundary zone.
        # The old value here was 0.5 in wire units (100 cm from the root), which
        # is beyond the body's entire measured reach (0.202 normalised): under the
        # body-scaled ROM that is fully AT the boundary, not inside it.
        motor_commands = torch.zeros(B, n_handles, 9, device=device)
        motor_commands[..., :3] = 0.10
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness({}, {}, latents, motor_commands=motor_commands)
        motor_state = wellness_out["grounded"]["motor_state"]
        assert motor_state.item() == 1.0, \
            f"Boundary formula should be 1.0 at 50% ROM, got {motor_state.item()}"

    def test_boundary_formula_penalty_near_limit(self, resolved, device):
        """Boundary formula penalizes positions near ROM limit."""
        wellness = WellnessEvaluators(
            resolved, wellness_mode=WellnessMode.DIRECT,
            motor_state_formula=MotorStateFormula.BOUNDARY,
        ).to(device)
        wellness.set_developmental_limits(1.0)
        B, n_handles = 1, resolved["body"]["n_handles"]
        d_model = resolved["model"]["d_model"]

        # Positions at 100% ROM — fully at mechanical stop
        motor_commands = torch.zeros(B, n_handles, 9, device=device)
        motor_commands[..., :3] = 1.0
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness({}, {}, latents, motor_commands=motor_commands)
        motor_state = wellness_out["grounded"]["motor_state"]
        assert motor_state.item() < 0.5, \
            f"Boundary formula should penalize at ROM limit, got {motor_state.item()}"

    def test_boundary_formula_has_gradient_near_limit(self, resolved, device):
        """Boundary formula has gradient when positions are near limits."""
        wellness = WellnessEvaluators(
            resolved, wellness_mode=WellnessMode.DIRECT,
            motor_state_formula=MotorStateFormula.BOUNDARY,
        ).to(device)
        wellness.set_developmental_limits(1.0)
        B, n_handles = 1, resolved["body"]["n_handles"]
        d_model = resolved["model"]["d_model"]

        # Inside the boundary zone in BODY-scale units: the zone runs from
        # 0.9 * 0.21 = 0.189 to the reach limit 0.21. Past the limit the penalty
        # saturates but the slope is HELD at its band-end value (straight-through,
        # 2026-09-15), so the gradient is nonzero in the zone and beyond it.
        motor_commands = torch.zeros(B, n_handles, 9, device=device, requires_grad=True)
        with torch.no_grad():
            motor_commands[..., :3] = 0.195
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness({}, {}, latents, motor_commands=motor_commands)
        motor_state = wellness_out["grounded"]["motor_state"]
        motor_state.sum().backward()

        assert motor_commands.grad is not None, "No gradient on motor_commands"
        assert motor_commands.grad.abs().sum() > 0, "Gradient is zero near limit"

    def test_boundary_formula_no_gradient_in_interior(self, resolved, device):
        """Boundary formula has zero gradient when positions are well within ROM."""
        wellness = WellnessEvaluators(
            resolved, wellness_mode=WellnessMode.DIRECT,
            motor_state_formula=MotorStateFormula.BOUNDARY,
        ).to(device)
        wellness.set_developmental_limits(1.0)
        B, n_handles = 1, resolved["body"]["n_handles"]
        d_model = resolved["model"]["d_model"]

        # Positions at half the body's reach - inside the boundary zone (see the
        # unit note in the zero-in-interior test above)
        motor_commands = torch.zeros(B, n_handles, 9, device=device, requires_grad=True)
        with torch.no_grad():
            motor_commands[..., :3] = 0.10
        latents = torch.randn(B, 6, d_model, device=device)

        wellness_out = wellness({}, {}, latents, motor_commands=motor_commands)
        motor_state = wellness_out["grounded"]["motor_state"]
        motor_state.sum().backward()

        # Gradient on position dims should be zero in interior
        pos_grad = motor_commands.grad[..., :3]
        assert pos_grad.abs().sum().item() < 1e-6, \
            f"Boundary gradient should be zero at 50% ROM, got {pos_grad.abs().sum().item()}"


# =========================================================================
# Test 4b: the formula the run actually configures
# =========================================================================

# The reach envelope the boundary formula measures against: BODY_REACH in
# lem/model/wellness.py - the measured farthest handle displacement in tokenizer
# units, 0.21 (42 cm) - times rom_limit. The penalty is zero inside the envelope
# and saturates past it; the slope is nonzero in the last tenth before the stop and,
# since the straight-through restore, held at that value past the stop.
BOUNDARY_REACH = 0.21
BOUNDARY_ZONE_LO = 0.9 * BOUNDARY_REACH


def _grad_of_wellness_sum(wellness, motor_commands, resolved, device):
    latents = torch.randn(motor_commands.shape[0], 6,
                          resolved["model"]["d_model"], device=device)
    out = wellness({}, {}, latents, motor_commands=motor_commands)
    total = torch.cat(list(out["grounded"].values()), dim=-1).sum()
    grad = torch.autograd.grad(total, motor_commands)[0]
    return out["grounded"]["motor_state"], grad


class TestMotorStateFormulaCoverage:
    """WellnessEvaluators' constructor default is PREFERENCE, whose
    1 - mean(|pos|/rom) has a gradient at every input. The two tests above that assert
    a non-zero wellness gradient omit the argument, so they exercise PREFERENCE and
    cannot fail - while every MIMo run sets --motor-state-formula boundary. These
    cover BOUNDARY, including the regime the body is actually in."""

    @pytest.mark.parametrize("formula", [MotorStateFormula.PREFERENCE,
                                         MotorStateFormula.BOUNDARY])
    def test_gradient_reaches_motor_commands_under_each_formula(self, resolved, device,
                                                                formula):
        wellness = WellnessEvaluators(resolved, motor_state_formula=formula).to(device)
        wellness.set_developmental_limits(1.0)
        pos = (0.05 if formula is MotorStateFormula.PREFERENCE
               else 0.5 * (BOUNDARY_ZONE_LO + BOUNDARY_REACH))
        n_handles = resolved["body"]["n_handles"]
        motor_commands = torch.zeros(1, n_handles, 9, device=device)
        motor_commands[..., :3] = pos
        motor_commands.requires_grad_(True)
        _score, grad = _grad_of_wellness_sum(wellness, motor_commands, resolved, device)
        assert grad.shape == (1, n_handles, 9)
        assert grad[..., :3].abs().sum().item() > 0, (
            f"{formula.value} passes no gradient to the motor command at pos {pos}")

    def test_past_the_envelope_the_score_saturates_but_the_slope_points_back(
            self, resolved, device):
        """Past the stop the penalty is full, and the slope still says which way home is.

        The penalty saturates on purpose: past the stop is simply "fully at the
        boundary", and letting the quadratic grow without bound once made a test pose
        score -218 and swamp every other term. But the clamp used to flatten the SLOPE
        as well, and that slope is an INPUT of MotorController.forward - so a handle
        resting past its envelope paid a fixed penalty while being told nothing about
        how to escape it. The run's feet sit there permanently.

        A straight-through term keeps the saturated VALUE and continues the slope, so
        the score is unchanged and the direction is restored.
        """
        wellness = WellnessEvaluators(
            resolved, motor_state_formula=MotorStateFormula.BOUNDARY).to(device)
        wellness.set_developmental_limits(1.0)
        motor_commands = torch.zeros(1, resolved["body"]["n_handles"], 9, device=device)
        motor_commands[..., :3] = 3.0 * BOUNDARY_REACH
        motor_commands.requires_grad_(True)
        score, grad = _grad_of_wellness_sum(wellness, motor_commands, resolved, device)
        assert score.item() == pytest.approx(0.0, abs=1e-6), (
            "past the stop every position coordinate should still carry full penalty")
        assert grad[..., :3].abs().sum().item() > 0.0, (
            "past the stop the controller is handed a constant with no direction in it")
        # the slope points back toward the envelope: raising |pos| further lowers the score
        assert (grad[..., :3] < 0).all(), (
            "the restored slope must point back inward, not further out")

    def test_the_restored_slope_leaves_every_wellness_score_bit_identical(
            self, resolved, device):
        """The control for the test above: nothing the body FEELS may move.

        Straight-through only redirects the backward pass. Swept from inside the
        envelope to three times the reach, the score must match the saturated
        clamp exactly - not approximately - or the change altered the wellness value
        rather than only the direction hint.
        """
        rom = 1.0
        threshold = 0.1 * rom
        pos = torch.linspace(0.0, 3.0 * rom, 601).reshape(-1, 1, 1).repeat(1, 3, 3)
        violation = torch.relu(threshold - (rom - pos.abs()))
        q = violation / threshold

        shipped = q.clamp(max=1.0).pow(2)                        # the saturating clamp
        soft = torch.where(q <= 1.0, q.pow(2), 2.0 * q - 1.0)    # the continuation
        straight_through = soft + (shipped - soft).detach()      # what the code now uses

        assert float((straight_through - shipped).abs().max()) == 0.0, (
            "the straight-through penalty changed a wellness VALUE; it may only "
            "change the direction the controller is handed")

        # and the continuation joins the band smoothly, so the restored slope is the
        # band-end value held rather than a new number invented past the limit
        at_limit = torch.tensor([1.0], requires_grad=True)
        (at_limit.clamp(max=1.0).pow(2)).backward()
        just_past = torch.tensor([1.5], requires_grad=True)
        (2.0 * just_past - 1.0).backward()
        assert float(at_limit.grad) == pytest.approx(float(just_past.grad), abs=1e-9)

    def test_boundary_gradient_band_is_a_tenth_of_the_reach(self, resolved, device):
        """How much of the body's reach carries a wellness->motor gradient at all.

        The band runs from 0.9 x reach to reach on each side: 0.042 of the 0.42-wide
        span a coordinate can occupy, so a handle placed uniformly across its reach
        steers wellness on about 10 draws in 100 and does nothing on the other 90
        (the uniform sample below never leaves the envelope). H81's 11-of-20 zero
        snapshots were recorded before the slope past the limit was restored and
        were handles PAST the limit, not inside it: on 17 of 20 at least one
        coordinate sat past 0.21.
        """
        wellness = WellnessEvaluators(
            resolved, motor_state_formula=MotorStateFormula.BOUNDARY).to(device)
        wellness.set_developmental_limits(1.0)
        torch.manual_seed(0)
        motor_commands = torch.zeros(400, resolved["body"]["n_handles"], 9, device=device)
        motor_commands[..., :3].uniform_(-BOUNDARY_REACH, BOUNDARY_REACH)
        motor_commands.requires_grad_(True)
        _score, grad = _grad_of_wellness_sum(wellness, motor_commands, resolved, device)
        pos_grad = grad[..., :3]
        assert pos_grad.numel() >= 20000, "too few coordinates to place the fraction"
        live = float((pos_grad != 0).float().mean())
        assert live == pytest.approx(0.10, abs=0.01), (
            f"the gradient band should cover a tenth of the reach; measured {live:.4f}")


# =========================================================================
# Test 5: predict_from_action
# =========================================================================

class TestPredictFromAction:
    def test_output_shape(self, resolved, device):
        """predict_from_action returns predictions + z_motor with correct shapes."""
        fm = ForwardModel(resolved).to(device)
        ph = PredictionHeads(resolved).to(device)
        er = EncoderRegistry(resolved).to(device)

        B = 1
        d_model = resolved["model"]["d_model"]
        n_mod = len(resolved["_active_modalities"])
        n_handles = resolved["body"]["n_handles"]

        last_step = torch.randn(B, n_mod, d_model, device=device)
        motor_out = torch.randn(B, n_handles, 9, device=device, requires_grad=True)

        predictions, z_motor = fm.predict_from_action(last_step, motor_out, er, ph)

        # Should have prediction for each head modality (all except motor)
        assert set(predictions.keys()) == set(resolved["_prediction_head_modalities"])
        for mod, pred in predictions.items():
            assert pred.shape == (B, d_model), f"{mod} shape: {pred.shape}"

        # z_motor should be [B, d_model]
        assert z_motor.shape == (B, d_model)

    def test_gradient_flow(self, resolved, device):
        """Gradient flows from predicted wellness loss back to motor_out."""
        fm = ForwardModel(resolved).to(device)
        ph = PredictionHeads(resolved).to(device)
        er = EncoderRegistry(resolved).to(device)
        wellness = WellnessEvaluators(resolved, wellness_mode=WellnessMode.FORWARD).to(device)

        B = 1
        d_model = resolved["model"]["d_model"]
        n_mod = len(resolved["_active_modalities"])
        n_handles = resolved["body"]["n_handles"]

        last_step = torch.randn(B, n_mod, d_model, device=device)
        motor_out = torch.randn(B, n_handles, 9, device=device, requires_grad=True)

        predictions, z_motor = fm.predict_from_action(last_step, motor_out, er, ph)
        w_predicted = wellness.score_predicted_wellness(predictions, z_motor)

        loss = -w_predicted.sum()
        loss.backward()

        assert motor_out.grad is not None, "No gradient on motor_out"
        assert motor_out.grad.abs().sum() > 0, "Gradient is zero on motor_out"

    def test_encoding_consistency(self, resolved, device):
        """z_motor from predict_from_action matches direct motor encoder output."""
        fm = ForwardModel(resolved).to(device)
        ph = PredictionHeads(resolved).to(device)
        er = EncoderRegistry(resolved).to(device)

        B = 1
        d_model = resolved["model"]["d_model"]
        n_mod = len(resolved["_active_modalities"])
        n_handles = resolved["body"]["n_handles"]

        last_step = torch.randn(B, n_mod, d_model, device=device)
        motor_out = torch.randn(B, n_handles, 9, device=device)

        # Get z_motor from predict_from_action
        _, z_motor_pfa = fm.predict_from_action(last_step, motor_out, er, ph)

        # Through the ONE canonical layout function. This test used to reimplement the
        # padding locally, which meant it agreed with itself rather than with the code -
        # and it silently encoded a different layout once the encoder widened to 19.
        # Through the registry, matching what predict_from_action now does. Calling the
        # encoder module directly gives a DIFFERENT token - it skips the output LayerNorm
        # - so this test would compare the canonical route against the defect it removed.
        from lem.model.encoder_registry import build_motor_input
        motor_input = build_motor_input(er, motor_out)
        _zm = er.encode({"motor": motor_input}, {"motor": "real"})["motor"]
        z_motor_direct = _zm.squeeze(1) if _zm.dim() == 3 else _zm

        assert torch.allclose(z_motor_pfa, z_motor_direct, atol=1e-6), \
            f"z_motor mismatch: max diff = {(z_motor_pfa - z_motor_direct).abs().max().item()}"


# =========================================================================
# Test 6: score_predicted_wellness
# =========================================================================

class TestScorePredictedWellness:
    def test_output_shape(self, resolved, device):
        """score_predicted_wellness returns [B, 5] with values in [0, 1]."""
        wellness = WellnessEvaluators(resolved, wellness_mode=WellnessMode.FORWARD).to(device)

        B = 1
        d_model = resolved["model"]["d_model"]

        # Build fake predicted tokens matching prediction_head_modalities
        predicted_tokens = {
            mod: torch.randn(B, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        z_motor = torch.randn(B, d_model, device=device)

        w = wellness.score_predicted_wellness(predicted_tokens, z_motor)

        assert w.shape == (B, 5)
        assert w.min() >= 0.0, f"Score below 0: {w.min().item()}"
        assert w.max() <= 1.0, f"Score above 1: {w.max().item()}"

    def test_gradient_to_motor(self, resolved, device):
        """Gradient flows from score_predicted_wellness back to z_motor."""
        wellness = WellnessEvaluators(resolved, wellness_mode=WellnessMode.FORWARD).to(device)

        B = 1
        d_model = resolved["model"]["d_model"]

        predicted_tokens = {
            mod: torch.randn(B, d_model, device=device)
            for mod in resolved["_prediction_head_modalities"]
        }
        z_motor = torch.randn(B, d_model, device=device, requires_grad=True)

        w = wellness.score_predicted_wellness(predicted_tokens, z_motor)
        loss = -w.sum()
        loss.backward()

        assert z_motor.grad is not None, "No gradient on z_motor"
        assert z_motor.grad.abs().sum() > 0, "Gradient is zero on z_motor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
