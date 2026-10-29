# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
WellnessEvaluators — 5 grounded + 1 approximate evaluator.

Grounded evaluators compute wellness from direct sensory signals.
Two modes:
  - BASELINE: MLP evaluators (original, for backward compat)
  - DIRECT/FORWARD: Differentiable formula evaluators (no learned params)

Formula evaluators:
  - prediction_error: 1 - clamp(PE, 0, 1)
  - coherence: cosine_similarity(preds, targets).mean().clamp(0, 1)
  - capability: mean(capability_features[:, :10])
  - motor_state (preference): 1 - mean(abs(motor_pos) / rom_limit)
  - motor_state (boundary): 1 - mean(relu(threshold - margin)²), zero in ROM interior
  - tactile_comfort: neutral with no contact, a pleasant band around 1 kPa (spec 8.2),
    falling to zero by 100 kPa (spec 8.1); texture from vibPower raises the pleasant part

Approximate evaluator predicts wellness from latent representations:
  - input: d_model * n_sensory_modalities -> 256 -> 128 -> 5
  Trained to match grounded outputs. Used in FORWARD mode to score
  predicted sensory tokens from predict_from_action.
"""

from __future__ import annotations

import enum
from typing import Dict, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Tactile comfort constants, in tokenizer pressure units (1.0 = 100 kPa). Module level so
# tools/calibrate.py checks THESE rather than its own copy - a calibration program that
# duplicates the constants it verifies drifts out of sync and reports on a number nobody
# runs. Every one of them is measured; see _formula_tactile_comfort for the readings.
from .contact_wellness import (TACTILE_PLEASANT_PEAK, TACTILE_PEAK_WIDTH,
                               TACTILE_PAIN_START, TACTILE_TEXTURE_REF)


class WellnessMode(enum.Enum):
    BASELINE = "baseline"
    DIRECT = "direct"
    FORWARD = "forward"


class MotorStateFormula(enum.Enum):
    PREFERENCE = "preference"  # 1 - mean(|pos|/rom_limit). Prefers ROM center.
    BOUNDARY = "boundary"      # relu(threshold - margin)². Zero in ROM interior,
                               # penalty only near mechanical stops.


class GroundedEvaluator(nn.Module):
    """Simple 2-layer MLP: input_dim -> hidden_dim -> 1."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ApproximateEvaluator(nn.Module):
    """MLP from latents to wellness approximation."""

    def __init__(self, input_dim: int, hidden_dims: list[int], n_outputs: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_outputs))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


APPROXIMATE_TERMS = ("prediction_error", "tactile_comfort", "capability", "coherence",
                     "motor_state")
"""The grounded terms the approximate evaluator predicts, in the order it predicts them.

The approximate evaluator has one output per name here and is trained to copy exactly these
(lem/training/loss_manager._compute_wellness_loss). Grounded terms added since - self_touch is
the first - are NOT among them, so anything comparing the two must line up on this list or it
is subtracting tensors of different widths. That mismatch went unnoticed because the gauge that
made it caught every exception and reported nothing: wellness_align_err, the number that says
whether an imagined moment's score can be trusted, read None on all 49 records of the live run
after self_touch joined the grounded set, having read 0.0134 and 0.0174 on the two segments
before it.

A grounded term outside this list is invisible to the planner's scoring of imagined moments,
because the planner scores them with the approximate evaluator. Adding a term to the objective
therefore means adding it here and widening the approximate evaluator, or accepting that the
planner cannot see it."""

APPROXIMATE_EXCLUDED = ("intermind", "vestibular")
"""Input modalities the approximate evaluator never reads.

intermind is the ManyMinds placeholder (Gaussian noise). vestibular is the sense that takes
intermind's slot under --vestibular-sense on (docs/gain_of_function_v3/DESIGN.md D4); excluding it
keeps the evaluator's input at the same five tokens (vision, touch, hearing, proprioception,
motor), so a checkpoint's evaluator loads with every weight unchanged. The grounded
prediction_error term averages the error of every sense graded real EXCEPT those in
WellnessEvaluators.pe_excluded: the vestibular build puts "vestibular" there by default
(--vestibular-wellness-pe exclude), so the raw wellness sum stays the one the intermind model
computed (DESIGN D8), and takes it out with "include". lem/config/resolver.py carries the same
tuple (it must not import torch); tests/test_vestibular_sense.py holds the two equal."""


class WellnessEvaluators(nn.Module):
    """
    Complete wellness evaluation system.

    5 grounded evaluators + 1 approximate evaluator.
    Built from resolved config.

    Args:
        config: Fully-resolved config dict.
        wellness_mode: Experiment mode.
            BASELINE: MLP grounded evaluators (original behavior).
            DIRECT/FORWARD: Differentiable formula evaluators.
    """

    EVALUATOR_NAMES = [
        "prediction_error", "tactile_comfort", "capability",
        "coherence", "motor_state",
    ]

    def __init__(self, config: dict,
                 wellness_mode: WellnessMode = WellnessMode.BASELINE,
                 motor_state_formula: MotorStateFormula = MotorStateFormula.PREFERENCE):
        super().__init__()
        self._mode = wellness_mode
        self._motor_state_formula = motor_state_formula

        wellness_cfg = config["model"]["wellness"]
        grounded_cfg = wellness_cfg["grounded"]
        approx_cfg = wellness_cfg["approximate"]

        # Grounded evaluators: MLPs for BASELINE, empty for formula modes
        self.grounded = nn.ModuleDict()
        if wellness_mode == WellnessMode.BASELINE:
            for name in self.EVALUATOR_NAMES:
                cfg = grounded_cfg[name]
                self.grounded[name] = GroundedEvaluator(
                    input_dim=cfg["input_dim"],
                    hidden_dim=cfg["hidden_dim"],
                )

        # Approximate evaluator (always built — used for alignment training
        # in all modes, and for FORWARD mode motor gradient scoring)
        self.approximate = ApproximateEvaluator(
            input_dim=approx_cfg["_input_dim"],
            hidden_dims=approx_cfg["hidden_dims"],
            n_outputs=approx_cfg["n_outputs"],
        )

        self._active_modalities = list(config["_active_modalities"])
        self._sensory_indices = [i for i, m in enumerate(self._active_modalities)
                                 if m not in APPROXIMATE_EXCLUDED]
        self._n_sensory = len(self._sensory_indices)
        self._d_model = config["model"]["d_model"]

        # ROM limit for motor_state formula (updated per step via set_developmental_limits)
        self._rom_limit = 1.0
        # senses the two prediction-derived grounded terms never read, even when graded real:
        # prediction_error, and coherence's fallback likeness of prediction to arrival (used only
        # when no body map runs). Empty by default (every sense counts, the behaviour before this
        # existed); the vestibular build sets ("vestibular",) under --vestibular-wellness-pe
        # exclude (mimo_bridge/vestibular_sense.configure)
        self.pe_excluded = ()
        # derived, never a literal: 207 sat here after the body grew to 24 handles
        self._motor_dim = wellness_cfg["grounded"]["motor_state"]["input_dim"]

    def set_developmental_limits(self, rom_limit: float) -> None:
        """Update ROM limit for motor_state formula evaluator."""
        self._rom_limit = rom_limit

    # ─── Formula evaluators ────────────────────────────────────────────
    # These compute wellness directly from raw sensor data.
    # NO .detach(), NO torch.no_grad() — computation graph must stay alive.

    def _formula_prediction_error(self, predictions: Dict[str, torch.Tensor],
                                  targets: Dict[str, torch.Tensor],
                                  B: int, device: torch.device,
                                  valid: Optional[Dict[str, str]] = None) -> torch.Tensor:
        """1 - clamp(PE, 0, 1). Higher = better predictions.

        Note: averages across modalities with equal weight. The training PE loss
        uses per-modality loss_weight and source_scales, so diagnostic logs will
        show two different PE numbers that don't match — this is expected.
        """
        if predictions and targets:
            # Gather matched modalities and stack once -- no list+append loop.
            # Synthetic channels are a fresh Gaussian draw: their error is
            # irreducible, so including them holds this score down permanently and
            # reports the model as predicting badly however well it predicts the
            # real world. Spec 7.1 makes this the region where predictions
            # SUCCEED; noise is not part of that region.
            matched = [mod for mod in predictions if mod in targets
                       and mod not in getattr(self, "pe_excluded", ())
                       and (valid is None
                            or str(getattr(valid.get(mod, 'real'), 'value',
                                           valid.get(mod, 'real'))) == 'real')]
            if matched:
                pred_stack = torch.stack([predictions[m] for m in matched])  # [n, B, d]
                tgt_stack = torch.stack([targets[m] for m in matched])      # [n, B, d]
                pe = (pred_stack - tgt_stack).pow(2).mean(dim=-1).mean(dim=0, keepdim=True).T  # [B, 1]
                return 1.0 - pe.clamp(0.0, 1.0)
        return torch.ones(B, 1, device=device)

    def _formula_coherence(self, predictions: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor],
                           body_schema_coherence: Optional[torch.Tensor],
                           B: int, device: torch.device) -> torch.Tensor:
        """The self-model's own reading of the body, or, with no body map running, the
        likeness between what was predicted and what arrived.

        WHAT THE CALLER MAY SEND, and what it means for the objective. Two quantities have
        been used here and they point opposite ways:
          EXTENT - the share of everything the body map watches that it calls its own, 0 to 1: its 26 body handles AND any outside thing that has been taken in (UEMA 11.2), so 28 things on the crib runs. This
            is what the project's goal asks wellness to reward: "maintain and extend its
            perceived self". A grasped toy that starts answering the commands raises it.
          CERTAINTY - mean(|membership - 0.5| * 2), how SURE the map is, which is 1 when
            every handle is clearly mine AND when every handle is clearly not mine. Measured
            on run H84 over 1,402 records it equalled 1 - 2 x the BODY's own share (correlation -1.0000, largest gap 1.0e-05 over the live run's records). Against extent, which now counts the outside things too, the correlation is 0.9991 and the largest gap 0.0322, because the map never
            claimed half the body, so wellness rose as the perceived self shrank. It carried
            53 of every 100 parts of wellness's movement at the minutes band, so that was not
            a detail.
        The caller picks, names its choice in the snapshot (lem_model's
        last_coherence_source), and the launch flag --wellness-coherence records it.

        One number per body, or one shared number, are both accepted; the fallback when no
        body map is running is the cosine likeness of prediction to arrival, a different
        quantity again, and the source name says so."""
        if body_schema_coherence is not None:
            # ONE NUMBER PER BODY where the caller sends one. It used to send the mean over
            # all the bodies, so body 0 and body 7 were handed the same coherence and each
            # body's goal-ring value carried a batch-wide ingredient. A scalar is still
            # accepted and broadcast, for callers that have only one. The clamp makes the
            # 0-to-1 bound that the (1 + curiosity drive) multiply in lem_model.step depends
            # on true at this site rather than in the producer.
            c = body_schema_coherence.to(device).reshape(-1)
            if c.numel() == B:
                return c.reshape(B, 1).clamp(0.0, 1.0)
            if c.numel() != 1:
                raise ValueError("coherence must be one number per body or one shared "
                                 "number, got %d for %d bodies" % (c.numel(), B))
            return c.reshape(1, 1).expand(B, 1).clamp(0.0, 1.0)
        if predictions and targets:
            matched = [mod for mod in predictions if mod in targets
                       and mod not in getattr(self, "pe_excluded", ())]
            if matched:
                pred_stack = torch.stack([predictions[m] for m in matched])  # [n, B, d]
                tgt_stack = torch.stack([targets[m] for m in matched])      # [n, B, d]
                # Batch cosine similarity: [n, B] -> mean -> [B, 1]
                sims = F.cosine_similarity(pred_stack, tgt_stack, dim=-1)   # [n, B]
                return sims.mean(dim=0).clamp(0.0, 1.0).unsqueeze(-1)
        return torch.zeros(B, 1, device=device)

    def _formula_capability(self, capability_features: Optional[torch.Tensor],
                            B: int, device: torch.device) -> torch.Tensor:
        """mean(capability_features[:, :10]). 10 body-set capacity values, already [0,1]."""
        if capability_features is not None:
            cf = capability_features
            if cf.dim() == 1:
                cf = cf.unsqueeze(0)
            # Same bound, same reason as coherence above: the producer clips these ten
            # capacities (proprio_adapter.py:89 np.clip(self.cap, 0.0, 1.0)) and H83
            # reads 0.820592 to 0.848251 over 137 rows, so the clamp changes no measured
            # value; it stops the wellness sum's non-negativity depending on a bound
            # enforced in another file.
            return cf[:, :10].mean(dim=-1, keepdim=True).to(device).clamp(0.0, 1.0)
        return torch.zeros(B, 1, device=device)

    def _formula_motor_state(self, motor_commands: Optional[torch.Tensor],
                             B: int, device: torch.device, rom_limit=None) -> torch.Tensor:
        """Dispatch to preference or boundary motor_state formula. rom_limit None =
        the module's scalar (live path); a tensor = one limit per sample (re-study)."""
        if motor_commands is None:
            return torch.ones(B, 1, device=device)
        if self._motor_state_formula == MotorStateFormula.BOUNDARY:
            return self._formula_motor_state_boundary(motor_commands, rom_limit)
        return self._formula_motor_state_preference(motor_commands, rom_limit)

    def _rom_scale(self, motor_commands: torch.Tensor, rom_limit):
        """The rom limit this score is judged under. None = the module's own scalar,
        set by set_developmental_limits - the live path, unchanged. A tensor = one
        limit PER SAMPLE, which is what re-study needs: each remembered window carries
        the limit of the phase it lived under, and the ring stores it (audit P166)."""
        if rom_limit is None:
            return max(self._rom_limit, 1e-6)
        r = torch.as_tensor(rom_limit, dtype=motor_commands.dtype, device=motor_commands.device)
        return r.reshape(-1, 1, 1).clamp(min=1e-6)

    def motor_command_gradient(self, motor_commands: torch.Tensor, rom_limit=None) -> torch.Tensor:
        """The wellness slope the motor controller reads as an input (spec 7.4):
        d(sum of grounded wellness)/d(motor command) at the pose given. Only the
        motor-state term has a path to the command, so this is that term's gradient
        and nothing else - measured: with the live create_graph=True form, 0 of the
        optimizer's 286 tensors are reachable through it. A test asserts this agrees
        with the live inline construction to exactly 0.0 on the same pose and limit."""
        score = self._formula_motor_state(motor_commands, motor_commands.shape[0],
                                          motor_commands.device, rom_limit)
        g = torch.autograd.grad(score.sum(), motor_commands, allow_unused=True)[0]
        return torch.zeros_like(motor_commands) if g is None else g

    def _formula_motor_state_preference(self, motor_commands: torch.Tensor, rom_limit=None) -> torch.Tensor:
        """1 - mean(|pos| / rom_limit). Higher = closer to ROM center.

        Gradient everywhere: always pushes positions toward zero.
        """
        pos = motor_commands[..., :3]  # [B, 23, 3]
        rom = self._rom_scale(motor_commands, rom_limit)
        normalized = (pos.abs() / rom).mean(dim=(-1, -2))  # [B]
        return (1.0 - normalized).clamp(0.0, 1.0).unsqueeze(-1)  # [B, 1]

    def _formula_motor_state_boundary(self, motor_commands: torch.Tensor, rom_limit=None) -> torch.Tensor:
        """1 - mean(relu(threshold - margin)²). Zero penalty in ROM interior,
        quadratic penalty only near mechanical stops.

        margin = rom_limit - |pos|  (distance from joint limit)
        threshold = 0.1 * rom_limit (boundary zone = last 10% of ROM)

        Gradient only near limits: does NOT push toward ROM center.
        """
        pos = motor_commands[..., :3]  # [B, 23, 3]
        # BODY_REACH (B3 fix, 2026-08-26). pos is normalised by the tokenizer's
        # 200 cm position range, so comparing |pos| against rom_limit (0..1) asked
        # for a 180 cm displacement before the boundary zone even began. Measured
        # on the crib-scene body: the farthest handle sits 0.404 m from the root -
        # 0.202 normalised - so the threshold was unreachable by 4.5x and this term
        # evaluated to exactly 1.0, gradient zero, in 3,713 of 3,713 logged windows
        # across seven runs. It is the ONE grounded wellness term with a path to
        # the motor command, so the spec 7.4 wellness->motor gradient was
        # identically zero for the project's entire history. rom_limit now scales
        # the body's measured reach rather than the tokenizer's wire range.
        # THE FRAME AND THE UNIT of that 0.404 m disagree with the way the number is
        # used, re-measured 2026-09-15. The caller (lem_model.py:1477) subtracts the
        # mean of all 26 handle positions, so what is judged is displacement from the
        # handle CENTROID, and the line below compares it PER CARTESIAN COORDINATE;
        # 0.404 m is a Euclidean distance from the ROOT body. Measured in the caller's
        # frame on the crib scene's trunk_v1 body: largest per-coordinate displacement
        # 0.1779 (35.6 cm) at the rest pose, 0.2709 (54.2 cm) over 200 randomised joint
        # poses, and 0.2917 (58.3 cm) in H81's own logged poses - 1.39x this constant.
        # Changing the constant moves the boundary zone and therefore motor_state on
        # every step, so it is left as measured-in-the-wrong-frame here rather than
        # raised silently.
        BODY_REACH = 0.21          # measured max handle displacement, normalised
        rom = self._rom_scale(motor_commands, rom_limit) * BODY_REACH
        threshold = 0.1 * rom
        margin = rom - pos.abs()                          # [B, 23, 3]
        violation = torch.relu(threshold - margin)        # [B, 23, 3], 0 in interior
        # Saturate at full penalty. The old normalisation comment ("max violation =
        # threshold") was only true while rom covered the whole wire range; with rom
        # scaled to the body's reach a position beyond it made the quadratic grow
        # without bound - a test pose at 0.5 produced wellness -218, which would
        # have swamped every other term. Past the limit is simply "fully at the
        # boundary": penalty 1, not penalty 219.
        # The clamp used to flatten the SLOPE past the limit as well, and that slope is
        # an INPUT of MotorController.forward (lem_model.py:1538) - not merely a backward
        # signal. So a handle sitting past its reach envelope handed the controller a
        # constant with no direction in it: measured at rom 1.0, d(score)/d|pos| is
        # -6.66667 at the limit and exactly 0.00000 at every position beyond it. A body
        # whose feet rest past the envelope paid a fixed penalty it could not be told how
        # to escape, which is a gain-of-function block and not a safeguard.
        #
        # Straight-through fixes that without changing what the body FEELS. pen_hard is
        # the value that has always shipped; pen_soft continues the same curve past the
        # limit (2q-1 has slope 2 at q=1, matching d(q^2)/dq there, so the join is C1);
        # detaching their difference makes the forward value bit-identical - measured
        # 0.000e+00 across 601 positions from 0 to 3x rom - while the backward slope past
        # the limit becomes -6.66667, the band-end value held, pointing back inward.
        q = violation / threshold                          # 0 in interior, 1 at the limit
        pen_hard = q.clamp(max=1.0).pow(2)                 # what ships: saturates at 1
        pen_soft = torch.where(q <= 1.0, q.pow(2), 2.0 * q - 1.0)
        penalty = pen_soft + (pen_hard - pen_soft).detach()  # [B, 23, 3], forward in [0, 1]
        mean_penalty = penalty.mean(dim=(-1, -2))          # [B]
        return (1.0 - mean_penalty).unsqueeze(-1)          # [B, 1]

    def _formula_tactile_comfort(self, raw_tactile: Optional[torch.Tensor],
                                 B: int, device: torch.device) -> torch.Tensor:
        """Evaluate each contact before aggregation; retain worst local discomfort.

        A dictionary may carry features, regional sensitivity and a validity mask.
        Tensor-only recordings retain their existing eight-field input contract.
        """
        from .contact_wellness import contact_response
        if raw_tactile is None:
            return torch.full((B, 1), 0.5, device=device)
        if isinstance(raw_tactile, dict):
            return contact_response(raw_tactile["features"],
                                    raw_tactile.get("sensitivity"),
                                    raw_tactile.get("valid"),
                                    profile=raw_tactile.get('profile', 'legacy'),
                                    history=raw_tactile.get('history'))["comfort"]
        return contact_response(raw_tactile)["comfort"]

    # ─── Forward pass ──────────────────────────────────────────────────

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                latents: torch.Tensor,
                capability_features: Optional[torch.Tensor] = None,
                motor_commands: Optional[torch.Tensor] = None,
                body_schema_coherence: Optional[torch.Tensor] = None,
                raw_tactile: Optional[torch.Tensor] = None,
                source_mask: Optional[Dict[str, str]] = None,
                canonical_tokens: Optional[Dict[str, torch.Tensor]] = None) -> dict:
        """
        Args:
            predictions: {mod_name: [B, d_model]} from prediction heads.
            targets: {mod_name: [B, d_model]} ground truth embeddings.
            latents: [B, seq_length, d_model] from forward model.
            capability_features: [B, 21] motor capability features (optional).
            motor_commands: [B, n_handles, 9] current motor commands (optional).
                When provided with requires_grad=True, enables wellness gradient
                computation for motor controller input (UEMA Spec §7.4).
            body_schema_coherence: scalar [0,1] from DifferentiableBodySchema
                (optional). When provided, replaces cross-modal cosine similarity
                as coherence input (UEMA Spec §11).
            raw_tactile: raw tactile features (pre-encoding). The bridge passes
                the 8-column FEATURE slice of its point-cloud layout: pressure(0),
                shear(1), slip(2), area(3), vibration(4..7) - pressure at index 0.
                (The historical SHM layout was [B, N, 7] position(3), force_mag,
                pressure, penetration, area with pressure at 4; the evaluator reads
                whatever column its formula names, so the DOC must match the caller.)
                Used by formula tactile_comfort evaluator.
                Not needed in BASELINE mode (MLP evaluator uses prediction embeddings).

        Returns:
            dict with:
              "grounded": {evaluator_name: [B, 1]} scores
              "approximate": [B, 5] predicted wellness
        """
        B = latents.shape[0]
        device = latents.device

        if self._mode == WellnessMode.BASELINE:
            # === ORIGINAL MLP PATH (unchanged) ===
            grounded_inputs = {}

            # prediction_error: mean L2 across prediction heads (vectorized)
            if predictions and targets:
                matched = [mod for mod in predictions if mod in targets]
                if matched:
                    pred_stack = torch.stack([predictions[m] for m in matched])  # [n, B, d]
                    tgt_stack = torch.stack([targets[m] for m in matched])       # [n, B, d]
                    grounded_inputs["prediction_error"] = (
                        (pred_stack - tgt_stack).pow(2).mean(dim=-1).mean(dim=0, keepdim=True).T
                    )
                else:
                    grounded_inputs["prediction_error"] = torch.zeros(B, 1, device=device)
            else:
                grounded_inputs["prediction_error"] = torch.zeros(B, 1, device=device)

            # tactile_comfort: tactile features (or zeros if unavailable)
            if "tactile" in predictions:
                grounded_inputs["tactile_comfort"] = predictions["tactile"][:, :8]
            else:
                grounded_inputs["tactile_comfort"] = torch.zeros(B, 8, device=device)

            # capability: motor-related features (from tokenized capability data or zeros)
            if capability_features is not None:
                cf = capability_features
                if cf.dim() == 1:
                    cf = cf.unsqueeze(0)  # [21] -> [1, 21]
                grounded_inputs["capability"] = cf.to(device)
            else:
                grounded_inputs["capability"] = torch.zeros(B, 21, device=device)  # Updated dim 17→21

            # coherence: body schema motor-sensor correlation (UEMA Spec §11)
            # Falls back to cross-modal cosine similarity if body schema unavailable
            if body_schema_coherence is not None:
                # one per body, or one shared number: the formula below replaces this input
                # anyway, but the untrained network still runs on it and must be fed a shape
                _c = body_schema_coherence.to(device).reshape(-1)
                grounded_inputs["coherence"] = (_c.reshape(B, 1) if _c.numel() == B
                                                else _c.reshape(1, 1).expand(B, 1))
            elif len(predictions) >= 2:
                pred_list = list(predictions.values())
                coherence = F.cosine_similarity(pred_list[0], pred_list[1], dim=-1)
                grounded_inputs["coherence"] = coherence.unsqueeze(-1)
            else:
                grounded_inputs["coherence"] = torch.zeros(B, 1, device=device)

            # motor_state: flattened motor commands (requires_grad for wellness gradient)
            if motor_commands is not None:
                grounded_inputs["motor_state"] = motor_commands.reshape(B, -1).to(device)  # [B, n_handles*9]
            else:
                grounded_inputs["motor_state"] = torch.zeros(B, self._motor_dim, device=device)

            # Run grounded evaluators
            grounded_scores = {}
            for name, evaluator in self.grounded.items():
                grounded_scores[name] = evaluator(grounded_inputs[name])
            # motor_state fix (2026-08-21): the untrained MLP on a raw 207-dim input
            # saturates its sigmoid - measured pegged at 1.0 for entire runs, passing
            # ZERO gradient to the motor controller (the Spec 7.4 pathway was dead).
            # The grounded formula is meaningful from step 0 and differentiable.
            grounded_scores["motor_state"] = self._formula_motor_state(motor_commands, B, device)
            # prediction_error fix (2026-08-23): same class of defect as motor_state above.
            # Spec 7.1 DEFINES wellness as the region where "predictions succeed (low PE)",
            # and 7.2 makes the grounded evaluator authoritative - nothing trains it, so its
            # shape has to implement the definition rather than discover it. An untrained
            # MLP does not: probed on the H15 brain at step 451791 it ran BACKWARDS, scoring
            # PE 0.0 at 0.559 and PE 1.0 at 0.621, over a total swing of 0.062. The formula
            # is 1 - clamp(PE, 0, 1): correct by construction from frame zero, and
            # differentiable, which spec 14 requires ("gradient signal from frame zero").
            grounded_scores["prediction_error"] = self._formula_prediction_error(
                predictions, targets, B, device, valid=source_mask)
            # ...and the last three, for the same reason (2026-08-23). Probed on the H15
            # brain at step 451791, their outputs barely moved across their whole input
            # range: coherence 0.015, tactile_comfort 0.009, capability 0.071. Not reversed
            # like prediction_error was, just inert - four of the five things spec 7.1 says
            # wellness IS were contributing almost nothing, leaving motor_state and the
            # movement drive as the only signals with any pull. Every grounded score is now
            # the formula that states the definition directly. The small networks stay in
            # the module so old snapshots still load; their outputs are no longer used.
            grounded_scores["tactile_comfort"] = self._formula_tactile_comfort(raw_tactile, B, device)
            grounded_scores["capability"] = self._formula_capability(capability_features, B, device)
            grounded_scores["coherence"] = self._formula_coherence(
                predictions, targets, body_schema_coherence, B, device)
        else:
            # === FORMULA EVALUATORS (no detach — graph must stay alive) ===
            grounded_scores = {
                "prediction_error": self._formula_prediction_error(
                    predictions, targets, B, device, valid=source_mask),
                "tactile_comfort": self._formula_tactile_comfort(raw_tactile, B, device),
                "capability": self._formula_capability(capability_features, B, device),
                "coherence": self._formula_coherence(predictions, targets, body_schema_coherence, B, device),
                "motor_state": self._formula_motor_state(motor_commands, B, device),
            }

        # Approximate evaluator (Judge B).
        #
        # THE DOMAIN IT IS SCORED IN MUST BE THE DOMAIN IT WAS TRAINED IN.
        #
        # This trained on `latents` - contextual hidden states from the full transformer -
        # while score_predicted_wellness() feeds it prediction-head outputs plus a
        # registry motor token, which are ENCODER-space. Two different representation
        # levels, one network. Fixing predict_from_action removed the 67,715x scale gap
        # and left the semantic gap: the scorer still learned one kind of vector and was
        # asked about another.
        #
        # When canonical_tokens is supplied, both domains use PredictiveState-style
        # encoder-space tokens and the mismatch is gone. Without it the old latent path
        # runs, so nothing silently changes for callers that have not been updated - and
        # `approximate_domain` in the result says which one was used, because a run must
        # be able to tell.
        n_mod = len(self._active_modalities)
        if canonical_tokens:
            ordered = [canonical_tokens[m] for i, m in enumerate(self._active_modalities)
                       if i in self._sensory_indices and m in canonical_tokens]
            if len(ordered) == self._n_sensory:
                approx_input = torch.stack(
                    [t.squeeze(1) if t.dim() == 3 else t for t in ordered], dim=1
                ).reshape(B, -1)
                domain = "canonical"
            else:
                last_timestep = latents[:, -n_mod:, :]
                approx_input = last_timestep[:, self._sensory_indices, :].reshape(B, -1)
                domain = "latent (incomplete canonical tokens)"
        else:
            last_timestep = latents[:, -n_mod:, :]  # [B, n_mod, d_model]
            approx_input = last_timestep[:, self._sensory_indices, :].reshape(B, -1)
            domain = "latent"

        approximate = self.approximate(approx_input)

        return {
            "grounded": grounded_scores,
            "approximate": approximate,
            "approximate_domain": domain,
        }

    # ─── FORWARD mode scoring ──────────────────────────────────────────

    def score_predicted_wellness(self, predicted_tokens: Dict[str, torch.Tensor],
                                 z_motor: torch.Tensor) -> torch.Tensor:
        """
        Score predicted sensory tokens with the approximate evaluator.

        Used in FORWARD mode: predict_from_action produces predicted next-frame
        embeddings + encoded motor token. This assembles them in the order the
        approximate evaluator expects and scores them.

        Args:
            predicted_tokens: {mod_name: [B, d_model]} from prediction heads
                (excludes motor; an intermind or vestibular entry is ignored).
            z_motor: [B, d_model] encoded motor action token.

        Returns:
            [B, 5] predicted wellness scores from approximate evaluator.
        """
        B = z_motor.shape[0]
        device = z_motor.device

        # Assemble in _active_modalities order, excluding intermind and vestibular
        # (matches _sensory_indices used during normal approximate evaluation)
        sensory_tokens = []
        for mod in self._active_modalities:
            if mod in APPROXIMATE_EXCLUDED:
                continue
            elif mod == "motor":
                sensory_tokens.append(z_motor)
            elif mod in predicted_tokens:
                sensory_tokens.append(predicted_tokens[mod])
            else:
                sensory_tokens.append(torch.zeros(B, self._d_model, device=device))

        stacked = torch.stack(sensory_tokens, dim=1)  # [B, n_sensory, d_model]
        approx_input = stacked.reshape(B, -1)  # [B, n_sensory * d_model]
        return self.approximate(approx_input)
