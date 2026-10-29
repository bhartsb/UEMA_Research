# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
LossManager — source-aware, per-modality loss weighting.

Features:
  - Per-frame modality_valid_mask and modality_source_mask
  - Source scales: real=1.0, stale=0.3, synthetic=0.0
  - Loss ramp: linear ramp from from_weight to to_weight over N steps
  - Split-rate masking: stale vision gets loss_weight=0
  - Motor loss: separate computation (output vs next-frame handle state)
  - Wellness loss: weighted by model.wellness.loss_weight
  - VICReg: variance + covariance regularization on encoder outputs (off by default)

Total loss = sum(per_mod_raw_loss * loss_weight * source_scale)
           + motor * motor_weight * motor_scale
           + wellness * w_weight
           + gaze * gaze_weight * vision_scale
           + vicreg (variance + covariance, when enabled)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from lem.source_mask import SOURCE_REAL, SOURCE_STALE, SOURCE_SYNTHETIC, to_int

logger = logging.getLogger(__name__)


class LossRamp:
    """Linear loss weight ramp for synthetic-to-real transitions."""

    def __init__(self, from_weight: float, to_weight: float,
                 over_steps: int, enabled: bool = False):
        self.from_weight = from_weight
        self.to_weight = to_weight
        self.over_steps = over_steps
        self.enabled = enabled
        self._start_step: Optional[int] = None

    def get_weight(self, current_step: int) -> Optional[float]:
        """Return ramped weight, or None if ramp is not active."""
        if not self.enabled:
            return None
        if self._start_step is None:
            self._start_step = current_step
        progress = (current_step - self._start_step) / max(self.over_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.from_weight + (self.to_weight - self.from_weight) * progress


class LossManager:
    """
    Source-aware loss computation.

    Reads per-modality loss_weight, source_scales, and loss_ramp configs
    from resolved config.
    """

    def __init__(self, config: dict, wellness_mode: str = "baseline"):
        model_cfg = config["model"]
        training_cfg = config["training"]

        self._active_modalities = config["_active_modalities"]
        self._prediction_head_modalities = config["_prediction_head_modalities"]
        self._d_model = model_cfg["d_model"]

        # Wellness experiment mode
        self._wellness_mode = wellness_mode
        self._mse_weight = 1.0
        self._wellness_grad_weight = 1.0

        # Per-modality loss weights
        self._loss_weights: Dict[str, float] = {}
        self._loss_ramps: Dict[str, LossRamp] = {}

        for mod_name in self._active_modalities:
            mod_cfg = config["modalities"][mod_name]
            self._loss_weights[mod_name] = mod_cfg["loss_weight"]

            ramp_cfg = mod_cfg.get("loss_ramp", {})
            self._loss_ramps[mod_name] = LossRamp(
                from_weight=ramp_cfg.get("from_weight", 0.1),
                to_weight=ramp_cfg.get("to_weight", 1.0),
                over_steps=ramp_cfg.get("over_steps", 1000),
                enabled=ramp_cfg.get("enabled", False),
            )

        # Source scales -- indexed by int constants (fast) with string fallbacks
        source_scales = training_cfg["loss"]["source_scales"]
        self._source_scales = {
            SOURCE_REAL: source_scales["real"],
            SOURCE_SYNTHETIC: source_scales["synthetic"],
            SOURCE_STALE: source_scales["stale"],
            # String fallbacks for backward compatibility
            "real": source_scales["real"],
            "synthetic": source_scales["synthetic"],
            "stale": source_scales["stale"],
        }

        # Wellness loss weight
        self._wellness_weight = model_cfg["wellness"]["loss_weight"]

        # Motor config
        self._n_handles = config["body"]["n_handles"]
        self._output_per_handle = model_cfg["motor_controller"]["output_per_handle"]

        # Gaze config
        gaze_cfg = model_cfg.get("gaze_head", {})
        self._gaze_enabled = gaze_cfg.get("enabled", False)
        self._gaze_weight = gaze_cfg.get("loss_weight", 0.1)

        # Voice config
        voice_cfg = model_cfg.get("voice_head", {})
        self._voice_enabled = voice_cfg.get("enabled", False)
        self._voice_weight = voice_cfg.get("loss_weight", 0.0)

        # VICReg config
        vicreg_cfg = training_cfg["loss"].get("vicreg", {})
        self._vicreg_enabled = vicreg_cfg.get("enabled", False)
        self._vicreg_variance_weight = vicreg_cfg.get("variance_weight", 1.0)
        self._vicreg_covariance_weight = vicreg_cfg.get("covariance_weight", 0.04)
        self._vicreg_variance_target = vicreg_cfg.get("variance_target", 1.0)
        # Temporal VICReg (batch 1 support): when the batch has a single frame, the variance
        # and covariance terms are taken over the current token plus the last
        # `temporal_window` tokens of that modality (detached history). 0 = off (old
        # behaviour: return 0 at B < 2). Live loops (UE, one MIMo body) are batch 1.
        self._vicreg_temporal_window = int(vicreg_cfg.get("temporal_window", 0))
        self._vicreg_history: Dict[str, deque] = {}
        self._vicreg_warned_batch = False

        # Imagination config
        imagination_cfg = training_cfg.get("imagination", {})
        self._imagination_weight = imagination_cfg.get("loss_weight", 0.5)

        # Latent regularizer config (CF-11, UEMA Spec §6.3)
        latent_reg_cfg = training_cfg.get("latent_reg", {})
        self._latent_reg_weight = latent_reg_cfg.get("loss_weight", 0.01)

        # Intention cross-sim config (CF-9, UEMA Spec §10.4)
        intention_cfg = training_cfg.get("intention", {})
        self._cross_sim_loss_weight = intention_cfg.get("cross_sim_loss_weight", 0.0)

    def set_mode_weights(self, mse_weight: float = 1.0,
                         wellness_grad_weight: float = 1.0) -> None:
        """Set configurable weights for FORWARD mode loss computation."""
        self._mse_weight = mse_weight
        self._wellness_grad_weight = wellness_grad_weight

    # A MEASUREMENT'S PROBE (mimo_bridge LemLite.lesson_probe installs its `terms` dict here): every weighted term
    # this method sums, recorded with its graph. None in every run: the recording lines are no-ops.
    term_probe = None

    def _probe(self, name: str, term) -> None:
        if self.term_probe is not None:
            self.term_probe[name] = term

    def compute(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                source_mask: Dict[str, str],
                motor_out: Optional[torch.Tensor] = None,
                motor_target: Optional[torch.Tensor] = None,
                wellness: Optional[dict] = None,
                developmental_limits: Optional[dict] = None,
                current_step: int = 0,
                gaze_out: Optional[torch.Tensor] = None,
                gaze_target: Optional[torch.Tensor] = None,
                voice_out: Optional[torch.Tensor] = None,
                voice_target: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Dict[str, torch.Tensor]] = None,
                return_breakdown: bool = False,
                total_wellness: Optional[torch.Tensor] = None,
                forward_wellness_loss: Optional[torch.Tensor] = None,
                imagination_loss: Optional[torch.Tensor] = None,
                latent_reg_loss: Optional[torch.Tensor] = None,
                cross_sim_loss: Optional[torch.Tensor] = None,
                motor_pos_excursion: Optional[torch.Tensor] = None,
                prediction_source_mask: Optional[Dict[str, str]] = None):
        """
        Compute total training loss.

        Args:
            predictions: {mod_name: [B, d_model]} from prediction heads.
            targets: {mod_name: [B, d_model]} ground truth embeddings.
            source_mask: {mod_name: "real"|"synthetic"|"stale"} per modality.
            prediction_source_mask: Source validity of prediction targets, when
                they come from a different frame. Defaults to source_mask.
            motor_out: [B, n_handles, 9] motor controller output.
            motor_target: [B, n_handles, 9] next-frame handle state.
            wellness: Output from WellnessEvaluators.forward().
            developmental_limits: {"rom_limit": float, "strength_limit": float}.
            current_step: Current training step (for loss ramp).
            return_breakdown: If True, return (total_loss, breakdown_dict).

        Returns:
            Scalar loss tensor, or (loss, breakdown) if return_breakdown=True.
            Breakdown contains GPU tensors (no .item() calls).
        """
        # Infer device from prediction tensors for zero-loss fallbacks
        _device = next(iter(predictions.values())).device if predictions else torch.device("cpu")

        # Collect loss terms in a list, sum once — avoids per-step torch.tensor() allocation
        loss_terms = []
        breakdown = {"prediction": {}} if return_breakdown else None

        # Prediction losses (per modality, excluding motor)
        prediction_sources = (source_mask if prediction_source_mask is None
                              else prediction_source_mask)
        for mod_name in self._prediction_head_modalities:
            if mod_name not in predictions or mod_name not in targets:
                continue

            # Raw MSE loss
            raw_loss = F.mse_loss(predictions[mod_name], targets[mod_name])

            # Loss weight (with ramp if active)
            weight = self._loss_weights.get(mod_name, 1.0)
            ramp = self._loss_ramps.get(mod_name)
            if ramp is not None:
                ramp_weight = ramp.get_weight(current_step)
                if ramp_weight is not None:
                    weight = ramp_weight

            # Source scale
            source = to_int(prediction_sources.get(mod_name, SOURCE_REAL))
            scale = self._source_scales.get(source, 1.0)

            weighted = raw_loss * weight * scale
            loss_terms.append(weighted)
            self._probe("prediction." + mod_name, weighted)
            if breakdown is not None:
                breakdown["prediction"][mod_name] = weighted.detach()

        # Motor loss (mode-dependent)
        if self._wellness_mode == "direct":
            # DIRECT: motor loss = -total_wellness (maximize wellness, no MSE)
            if total_wellness is not None:
                motor_loss_val = -total_wellness
                loss_terms.append(motor_loss_val)
                self._probe("motor", motor_loss_val)
                if breakdown is not None:
                    breakdown["motor"] = motor_loss_val.detach()
        elif self._wellness_mode == "forward":
            # FORWARD: weighted MSE + weighted forward-model wellness
            motor_loss_val = None
            if motor_out is not None and motor_target is not None and self._mse_weight > 0:
                mse_loss = self._compute_motor_loss(
                    motor_out, motor_target, developmental_limits,
                    pos_excursion=motor_pos_excursion)
                motor_weight = self._loss_weights.get("motor", 1.0)
                motor_source = to_int(source_mask.get("motor", SOURCE_REAL))
                motor_scale = self._source_scales.get(motor_source, 1.0)
                weighted_mse = mse_loss * motor_weight * motor_scale * self._mse_weight
                loss_terms.append(weighted_mse)
                self._probe("motor_mse", weighted_mse)
                if breakdown is not None:
                    breakdown["motor_mse"] = weighted_mse.detach()

            if forward_wellness_loss is not None and self._wellness_grad_weight > 0:
                weighted_fw = forward_wellness_loss * self._wellness_grad_weight
                loss_terms.append(weighted_fw)
                self._probe("motor_wellness", weighted_fw)
                if breakdown is not None:
                    breakdown["motor_wellness"] = weighted_fw.detach()

            if breakdown is not None:
                breakdown["motor"] = (
                    breakdown.get("motor_mse", torch.tensor(0.0, device=_device))
                    + breakdown.get("motor_wellness", torch.tensor(0.0, device=_device))
                )
        else:
            # BASELINE: original motor MSE code (unchanged)
            motor_loss_val = None
            if motor_out is not None and motor_target is not None:
                motor_loss_val = self._compute_motor_loss(
                    motor_out, motor_target, developmental_limits,
                    pos_excursion=motor_pos_excursion,
                )
                motor_weight = self._loss_weights.get("motor", 1.0)
                motor_source = to_int(source_mask.get("motor", SOURCE_REAL))
                motor_scale = self._source_scales.get(motor_source, 1.0)
                weighted = motor_loss_val * motor_weight * motor_scale
                loss_terms.append(weighted)
                self._probe("motor", weighted)
                if breakdown is not None:
                    breakdown["motor"] = weighted.detach()

        # Wellness loss
        wellness_loss_val = None
        if wellness is not None:
            wellness_loss_val = self._compute_wellness_loss(wellness, device=_device)
            weighted = wellness_loss_val * self._wellness_weight
            loss_terms.append(weighted)
            self._probe("wellness", weighted)
            if breakdown is not None:
                breakdown["wellness"] = weighted.detach()

        # Gaze loss
        if self._gaze_enabled and gaze_out is not None and gaze_target is not None:
            gaze_loss = F.mse_loss(gaze_out, gaze_target)
            # Only count gaze loss when vision is active (not stale)
            gaze_source = to_int(source_mask.get("vision", SOURCE_REAL))
            gaze_scale = self._source_scales.get(gaze_source, 1.0)
            weighted = gaze_loss * self._gaze_weight * gaze_scale
            loss_terms.append(weighted)
            self._probe("gaze", weighted)
            if breakdown is not None:
                breakdown["gaze"] = weighted.detach()

        # Voice loss
        if self._voice_enabled and voice_out is not None and voice_target is not None:
            # The voice head outputs 80 numbers (the reserved action-slot width) and the gaze head outputs
            # 2; five synthetic targets in lem/benchmark_dispatch.py were copied from the
            # gaze target one line above and kept width 2, so F.mse_loss raised before
            # the 0.0 loss_weight could make the term inert. Say which two widths
            # disagree rather than leaving it to a broadcasting message - or, at a width
            # that does broadcast, to a silently wrong number.
            if voice_out.shape != voice_target.shape:
                raise ValueError(
                    f"voice target {tuple(voice_target.shape)} does not match voice head "
                    f"output {tuple(voice_out.shape)}")
            voice_loss = F.mse_loss(voice_out, voice_target)
            weighted = voice_loss * self._voice_weight
            loss_terms.append(weighted)
            self._probe("voice", weighted)
            if breakdown is not None:
                breakdown["voice"] = weighted.detach()

        # VICReg encoder regularization
        if encoder_outputs is not None:
            vicreg_loss = self.compute_vicreg(encoder_outputs, device=_device)
            loss_terms.append(vicreg_loss)
            self._probe("vicreg", vicreg_loss)
            if breakdown is not None:
                breakdown["vicreg"] = vicreg_loss.detach()

        # Imagination loss (multi-step prediction chain, UEMA Spec §6.3)
        if imagination_loss is not None and self._imagination_weight > 0:
            weighted = imagination_loss * self._imagination_weight
            loss_terms.append(weighted)
            self._probe("imagination", weighted)
            if breakdown is not None:
                breakdown["imagination"] = weighted.detach()

        # Latent regularizer penalty (CF-11, UEMA Spec §6.3)
        if latent_reg_loss is not None and self._latent_reg_weight > 0:
            weighted = latent_reg_loss * self._latent_reg_weight
            loss_terms.append(weighted)
            self._probe("latent_reg", weighted)
            if breakdown is not None:
                breakdown["latent_reg"] = weighted.detach()

        # Cross-simulation attention loss (CF-9, UEMA Spec §10.4)
        if cross_sim_loss is not None and self._cross_sim_loss_weight > 0:
            weighted = cross_sim_loss * self._cross_sim_loss_weight
            loss_terms.append(weighted)
            self._probe("cross_sim", weighted)
            if breakdown is not None:
                breakdown["cross_sim"] = weighted.detach()

        if not loss_terms:
            total = torch.zeros(1, device=_device, requires_grad=True)
            if return_breakdown:
                return total, breakdown
            return total

        total = sum(loss_terms)
        if return_breakdown:
            return total, breakdown
        return total

    def _compute_motor_loss(self, motor_out: torch.Tensor,
                            motor_target: torch.Tensor,
                            developmental_limits: Optional[dict],
                            pos_excursion: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Motor loss: output vs next-frame handle state, clamped to developmental ROM/strength range.

        Output layout [N, 9]: pos(3) + rot(4) + radius(1) + strength(1).
        Target layout [N, 9]: same (extracted from handle state, skipping is_active).
        """
        # Mask inactive handles (curriculum phases): zero both output and target
        # so MSE for masked handles contributes zero to loss and zero gradient.
        if developmental_limits is not None:
            handle_mask = developmental_limits.get("handle_mask")
            if handle_mask is not None:
                _mask = torch.tensor(handle_mask, dtype=motor_out.dtype,
                                     device=motor_out.device)
                _mask = _mask.unsqueeze(0).unsqueeze(-1)  # [1, n_handles, 1]
                motor_out = motor_out * _mask
                motor_target = motor_target * _mask
                if pos_excursion is not None:
                    pos_excursion = pos_excursion * _mask

        # Split output into position, rotation, radius, strength
        pos_out = motor_out[:, :, :3]
        rot_out = motor_out[:, :, 3:7]
        rad_out = motor_out[:, :, 7:8]
        str_out = motor_out[:, :, 8:9]

        pos_tgt = motor_target[:, :, :3]
        rot_tgt = motor_target[:, :, 3:7]
        rad_tgt = motor_target[:, :, 7:8]
        str_tgt = motor_target[:, :, 8:9]

        # Apply developmental limits. rom_limit may be a SCALAR (the live step) or a
        # PER-SAMPLE tensor [N] (replay: each remembered moment trained under its own
        # phase's reach - review 2026-09-06: the batch MEAN clamped neonate and infant
        # windows alike to a limit that applied to neither moment; no effect while
        # every window shares one phase, wrong the day a growth run mixes them).
        rom_limit = 1.0
        strength_limit = 1.0
        if developmental_limits is not None:
            rom_limit = developmental_limits.get("rom_limit", 1.0)
            strength_limit = developmental_limits.get("strength_limit", 1.0)

        # Position loss (target clamped to developmental ROM)
        if torch.is_tensor(rom_limit) and rom_limit.dim() > 0:
            _rl = rom_limit.to(pos_tgt.device, pos_tgt.dtype).view(-1, 1, 1)
            pos_tgt_clamped = torch.maximum(torch.minimum(pos_tgt, _rl), -_rl)
        else:
            pos_tgt_clamped = pos_tgt.clamp(-float(rom_limit), float(rom_limit))
        # Grade the command the body actually receives (every execution path clamps
        # the raw output to +/-1 before scaling to centimetres: lem_model.py:2360,
        # :2778, live_test.py:80), and charge the excursion separately so a runaway
        # output still has a gradient pulling it back inside the executable range.
        pos_out_exec = pos_out.clamp(-1.0, 1.0)
        pos_loss = F.mse_loss(pos_out_exec, pos_tgt_clamped)
        if pos_excursion is not None:
            # Audit P118, 2026-09-19. The controller's position channel is bounded at its
            # source (motor_controller.py, straight-through clamp), so motor_out already IS
            # the command the body receives and the squared excursion charge below would
            # read exactly 0.0 for it. The pull-back acts on the excursion the controller
            # REPORTS - how far its inner number ran past +/-1 - and is bounded (Huber,
            # beta 1.0). Measured on the H81 explain probe's recorded proposals, the
            # squared form read 69,960 at step_160 beside a goal term of 0.9956, and at
            # grad_clip 1.0 that single term was the whole clipped gradient budget on a
            # saturated step; the Huber form reads mean 40.6, max 219.7 on the same
            # proposals. It keeps the inward pull and cannot swamp the other lessons.
            pos_loss = pos_loss + F.smooth_l1_loss(
                pos_excursion, torch.zeros_like(pos_excursion), beta=1.0)
        else:
            # callers that hand in a raw, unbounded tensor with no controller metadata
            # keep the 2026-09-15 charge on the part outside the executable range
            pos_loss = pos_loss + F.mse_loss(pos_out, pos_out_exec.detach())

        # Rotation loss (quaternion distance, hemisphere-aligned).
        # q and -q represent the same rotation. Flip target to the same
        # hemisphere as output to avoid penalizing equivalent orientations.
        dot = (rot_out * rot_tgt).sum(dim=-1, keepdim=True)
        rot_tgt_aligned = torch.where(dot < 0, -rot_tgt, rot_tgt)
        rot_loss = F.mse_loss(rot_out, rot_tgt_aligned)

        # Radius: on the MIMo bridge the de-echo copies the controller's own radius into
        # the target (lem_model.py, live step and replay step alike), and nothing clamps
        # radius, so this term is exactly 0.0 in every phase and the radius channel is
        # trained by nothing. Grading it against the echo of the model's own command
        # would teach self-copying, which is why the de-echo exists; a target has to come
        # from the body, the way the position channel's does.
        rad_loss = F.mse_loss(rad_out, rad_tgt)

        # Strength: with the same de-echo in force this is a ONE-SIDED BARRIER, not a
        # regression term - it costs something only where the command leaves
        # [0, the phase's strength limit]. The controller's strength comes out of a
        # sigmoid, so at the mature limit of 1.0 every output is already inside and this
        # term reads exactly 0.0; at the neonate limit of 0.02 every sigmoid output is
        # outside and it becomes the largest motor term after position. H81, H82 and H83
        # log strength_limit 1.0 in every window, so it is 0.0 throughout those runs.
        # So the motor lesson grades position and rotation against the pose the body
        # reached, plus this barrier - not four graded parts.
        # (target clamped to developmental strength; tensor = per-sample)
        if torch.is_tensor(strength_limit) and strength_limit.dim() > 0:
            _sl = strength_limit.to(str_tgt.device, str_tgt.dtype).view(-1, 1, 1)
            str_tgt_clamped = torch.minimum(str_tgt.clamp(min=0), _sl)
        else:
            str_tgt_clamped = str_tgt.clamp(0, float(strength_limit))
        str_loss = F.mse_loss(str_out, str_tgt_clamped)

        return pos_loss + rot_loss + rad_loss + str_loss

    def compute_vicreg(self, encoder_outputs: Dict[str, torch.Tensor],
                       device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        VICReg encoder regularization — variance + covariance terms.

        Prevents encoder collapse during early training with sparse data.
        Applied per-modality on encoder output embeddings, then averaged.

        Args:
            encoder_outputs: {mod_name: [B, 1, d_model]} raw encoder tokens.
            device: Target device for zero-loss fallbacks.

        Returns:
            Scalar VICReg loss (0.0 if disabled or B < 2).
        """
        if not self._vicreg_enabled:
            return torch.tensor(0.0, device=device)

        var_losses = []
        cov_losses = []

        for mod_name, token in encoder_outputs.items():
            # Squeeze token dim: [B, 1, d_model] -> [B, d_model]
            z = token.squeeze(1) if token.dim() == 3 else token

            B, d = z.shape

            if B < 2:
                if self._vicreg_temporal_window <= 0:
                    if not self._vicreg_warned_batch:
                        logger.warning(
                            "VICReg requires batch_size >= 2 to be effective "
                            "(got B=%d). Returning 0. Set training.loss.vicreg.temporal_window "
                            "> 0 to compare against recent frames instead.", B
                        )
                        self._vicreg_warned_batch = True
                    return torch.tensor(0.0, device=z.device)
                # temporal mode: current token (with grad) + recent detached tokens
                hist = self._vicreg_history.setdefault(
                    mod_name, deque(maxlen=self._vicreg_temporal_window))
                past = [h.to(z.device) for h in hist]
                hist.append(z[0].detach().clone())
                if not past:
                    continue   # first frame of this modality: nothing to compare with yet
                z = torch.cat([z, torch.stack(past)], dim=0)   # [1 + H, d]
                B = z.shape[0]

            # Variance loss: penalize dims with std below target
            std = z.std(dim=0)  # [d_model]
            var_loss = F.relu(self._vicreg_variance_target - std).mean()
            var_losses.append(var_loss)

            # Covariance loss: penalize off-diagonal correlations
            z_centered = z - z.mean(dim=0)
            cov = (z_centered.T @ z_centered) / (B - 1)  # [d_model, d_model]
            cov = cov.clone()
            cov.fill_diagonal_(0)
            cov_loss = (cov ** 2).sum() / d
            cov_losses.append(cov_loss)

        if not var_losses:
            return torch.tensor(0.0, device=device)

        mean_var = torch.stack(var_losses).mean()
        mean_cov = torch.stack(cov_losses).mean()

        return (self._vicreg_variance_weight * mean_var
                + self._vicreg_covariance_weight * mean_cov)

    def _compute_wellness_loss(self, wellness: dict,
                               device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Wellness loss: approximate evaluator should match grounded evaluators.
        """
        grounded = wellness.get("grounded", {})
        approximate = wellness.get("approximate")

        if approximate is None or not grounded:
            return torch.tensor(0.0, device=device)

        # Stack grounded scores: [B, 5], in the order the approximate evaluator predicts
        # them. ONE list, shared with the gauge that measures how well the two agree, so the
        # lesson and the gauge can never line up differently.
        from lem.model.wellness import APPROXIMATE_TERMS
        grounded_vals = []
        for name in APPROXIMATE_TERMS:
            if name in grounded:
                grounded_vals.append(grounded[name])

        if not grounded_vals:
            return torch.tensor(0.0, device=device)

        grounded_tensor = torch.cat(grounded_vals, dim=-1)  # [B, 5]

        # MSE between approximate and grounded (detach grounded as target)
        return F.mse_loss(approximate, grounded_tensor.detach())
