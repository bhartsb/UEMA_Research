# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
MotorController — separate from the forward model.

Generates motor actions, not sensory predictions.

Input: d_model * n_active_modalities + n_handles * features_per_handle + n_handles * output_per_handle (wellness grad)
Architecture: Variable-depth MLP (default 6 layers, 512 hidden).
Confidence head: d_model * n_active -> 128 -> 1 for early halt.
Recursive option: 2 layers recursed 1-6 times (1.7M vs 5.0M params).
Adaptive option (CF-7, §9.1-9.3): separate residual blocks with confidence-gated depth.
  Training: soft-weighted residuals (all blocks run, sigmoid decay past target depth).
  Inference: hard gate at integer depth computed from confidence + learned threshold.

Output per handle: pos(3) + rot(4) + radius(1) + strength(1) = 9 values * n_handles.
(Spec §9.3 line 258 says 8 — that's a spec typo; actual count is 9.)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MotorController(nn.Module):
    """
    Motor action generator.

    Input: the transformer's newest token per sense laid end to end, then the handle
    state, then the wellness gradient - one concatenation, no averaging.
    Output: [B, n_handles, output_per_handle]
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config["model"]
        mc_cfg = model_cfg["motor_controller"]
        d_model = model_cfg["d_model"]
        n_active = len(config["_active_modalities"])
        n_handles = config["body"]["n_handles"]

        self._input_dim = mc_cfg["_input_dim"]
        self._n_handles = n_handles
        self._output_per_handle = mc_cfg["output_per_handle"]
        self._n_layers = mc_cfg["n_layers"]
        self._hidden_dim = mc_cfg["hidden_dim"]
        self._confidence_threshold = mc_cfg["confidence_threshold"]
        self._recursive = mc_cfg["recursive"]
        self._adaptive = mc_cfg.get("adaptive", False)
        self._max_depth = mc_cfg.get("max_depth", mc_cfg["n_layers"])
        self._thinking_energy_rate = mc_cfg.get("thinking_energy_rate", 0.001)
        self._d_model = d_model
        self._n_active = n_active
        # Follows the config, never a fixed number: 26 * 9 = 234 on the trunk body,
        # 24 * 9 = 216 on the legacy one. (The old comment here said 207, a 23-handle
        # body that no run has used since the trunk layout arrived.)
        self._wellness_grad_dim = n_handles * self._output_per_handle

        # Confidence head: d_model * n_active -> confidence_hidden -> 1
        latent_dim = d_model * n_active
        self.confidence_head = nn.Sequential(
            nn.Linear(latent_dim, mc_cfg["confidence_hidden"]),
            nn.GELU(),
            nn.Linear(mc_cfg["confidence_hidden"], 1),
            nn.Sigmoid(),
        )

        # Motor MLP
        total_output = n_handles * self._output_per_handle

        if self._adaptive:
            # CF-7 (UEMA Spec §9.1-9.3): Adaptive depth with separate residual blocks.
            # Learned depth threshold as nn.Parameter — optimized via gradient flow
            # through soft sigmoid weights during training.
            initial_threshold = mc_cfg.get("initial_depth_threshold", 0.7)
            logit = math.log(initial_threshold / (1.0 - initial_threshold))
            self.depth_threshold_raw = nn.Parameter(torch.tensor(logit))

            # Spec §9.3 line 236 has a dimension bug: each layer takes the full
            # concatenated input (d_model*6 + n_handles*8) but the residual at
            # line 243 adds the 512-dim layer output to the full-dim x — shape
            # mismatch. Resolution: input_proj maps to hidden_dim first, then
            # all residual blocks operate at hidden_dim uniformly.
            self.input_proj = nn.Linear(self._input_dim, self._hidden_dim)
            self.residual_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self._hidden_dim, self._hidden_dim),
                    nn.GELU(),
                    nn.Linear(self._hidden_dim, self._hidden_dim),
                )
                for _ in range(self._max_depth)
            ])
            self.output_proj = nn.Linear(self._hidden_dim, total_output)
        elif self._recursive:
            # 2 layers recursed 1-6 times
            self.input_proj = nn.Linear(self._input_dim, self._hidden_dim)
            self.recursive_block = nn.Sequential(
                nn.Linear(self._hidden_dim, self._hidden_dim),
                nn.GELU(),
                nn.Linear(self._hidden_dim, self._hidden_dim),
                nn.GELU(),
            )
            self.output_proj = nn.Linear(self._hidden_dim, total_output)
        else:
            # Sequential N-layer MLP
            layers = []
            in_dim = self._input_dim
            for i in range(self._n_layers):
                out_dim = self._hidden_dim if i < self._n_layers - 1 else total_output
                layers.append(nn.Linear(in_dim, out_dim))
                if i < self._n_layers - 1:
                    layers.append(nn.GELU())
                in_dim = out_dim
            self.mlp = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor,
                handle_state: torch.Tensor,
                wellness_grad: Optional[torch.Tensor] = None,
                depth: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            latents: [B, seq_length, d_model] from forward model.
            handle_state: [B, n_handles, features_per_handle] current handle state.
            wellness_grad: [B, n_handles, output_per_handle] wellness gradient
                (UEMA Spec §7.4). When None, zeros are used.
            depth: Optional explicit depth override (adaptive mode only).
                When provided, runs exactly this many residual blocks.

        Returns:
            (motor_output, confidence, metadata):
                motor_output: [B, n_handles, output_per_handle] motor targets.
                confidence: [B, 1] confidence score (sigmoid).
                metadata: dict with depth, confidence, depth_threshold.
        """
        B = latents.shape[0]

        # The last n_active tokens - one per sense, newest timestep - laid END TO END.
        # A concatenation, not an average: d_model * n_active numbers (768 at d_model
        # 128 with 6 senses), where a mean would be d_model numbers and would lose
        # which sense each number came from.
        last_tokens = latents[:, -self._n_active:, :]  # [B, n_active, d_model]
        pooled = last_tokens.reshape(B, -1)  # [B, d_model * n_active]

        # Confidence score
        confidence = self.confidence_head(pooled)  # [B, 1]

        # Wellness gradient: flatten or zero-fill
        if wellness_grad is not None:
            flat_wgrad = wellness_grad.reshape(B, -1)  # [B, n_handles * output_per_handle]
        else:
            flat_wgrad = torch.zeros(B, self._wellness_grad_dim, device=latents.device)

        # Motor input: concat pooled latents + flat handle state + wellness gradient
        flat_handles = handle_state.reshape(B, -1)  # [B, n_handles * features_per_handle]
        motor_input = torch.cat([pooled, flat_handles, flat_wgrad], dim=-1)  # [B, _input_dim]

        if self._adaptive:
            depth_threshold = torch.sigmoid(self.depth_threshold_raw)
            # Per-sample depth target: 1 when conf >> threshold, max_depth when conf << threshold.
            # confidence is [B, 1] -- keep per-sample so each item in the batch
            # gets its own adaptive depth weighting instead of collapsing to batch mean.
            raw_depth = 1.0 + (self._max_depth - 1) * torch.sigmoid(
                5.0 * (depth_threshold - confidence))  # [B, 1]

            if depth is not None:
                # Explicit depth override — hard gate, no soft weighting
                x = self.input_proj(motor_input)
                clamped_depth = max(1, min(depth, self._max_depth))
                for i in range(clamped_depth):
                    x = self.residual_blocks[i](x) + x
                computed_depth = clamped_depth
            else:
                # Soft sigmoid gating -- used for both training and inference.
                # Training needs differentiability; inference benefits from avoiding
                # .item() GPU->CPU syncs that stall the pipeline (~0.5ms each).
                # All blocks run, but contribution decays past target depth via
                # sigmoid weights. Gradient flows through in training mode.
                x = self.input_proj(motor_input)
                for i, block in enumerate(self.residual_blocks):
                    weight = torch.sigmoid(5.0 * (raw_depth - (i + 1)))
                    x = weight * block(x) + x  # Weighted residual
                computed_depth = self._max_depth  # All ran (soft-weighted)
            output = self.output_proj(x)
        elif self._recursive:
            # Recursive: 2-layer block iterated max times.
            # Always runs all iterations -- early halt via .item() caused GPU->CPU
            # sync stalls. The savings from skipping a cheap MLP iteration never
            # outweighed the pipeline stall cost.
            x = self.input_proj(motor_input)
            for _ in range(self._n_layers):
                x = self.recursive_block(x) + x  # Residual
            output = self.output_proj(x)
            computed_depth = self._n_layers
        else:
            output = self.mlp(motor_input)
            computed_depth = self._n_layers

        # Reshape to [B, n_handles, output_per_handle]
        output = output.reshape(B, self._n_handles, self._output_per_handle)

        # Apply activations: pos(0:3) bounded straight-through, rot(3:7) normalized,
        # radius+strength(7:9) sigmoid. Fused sigmoid on adjacent channels, 3-way cat.
        #
        # STRAIGHT-THROUGH BOUND on the position channel (audit P118, 2026-09-19). Every
        # execution path already clamps this channel to [-1, 1] before scaling to
        # centimetres (the body path, execution_transform and
        # differentiable_execution_transform in mimo_bridge/lem_model.py, and live_test.py),
        # so the forward value here is what the body has always received: measured
        # 0.000e+00 change in the executed centimetres on all 1,560 recorded position
        # coordinates of the H81 explain probe, planner path and body path alike. The
        # backward slope stays 1, so the head still learns.
        # Why the VALUE has to be bounded and not only the slope: the planner seeds its
        # position parameter from this tensor (intention_generator.py) and the planner's
        # own straight-through bound has a tanh whose float32 slope is exactly 0.0 at
        # |pos| >= 9.010840. 580 of those 1,560 recorded coordinates (37.2%) sat past that
        # point, so positional descent was dead on them for all 20 Adam rounds. With the
        # value bounded the seed sits at 1.0 at most, where the slope is 84.0, and the dead
        # set measured 0 of 1,560.
        raw_pos = output[:, :, :3]
        pos = raw_pos + (raw_pos.clamp(-1.0, 1.0) - raw_pos).detach()
        rot = output[:, :, 3:7]
        rot_normed = rot / (rot.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        rad_str = torch.sigmoid(output[:, :, 7:9])  # fused: radius + strength
        output = torch.cat([pos, rot_normed, rad_str], dim=-1)

        metadata = {
            # how far the head's inner number ran past the executable range, per
            # coordinate (0 inside +/-1). The bound above hides a runaway from every
            # consumer of `output`; the motor loss charges this instead
            # (loss_manager._compute_motor_loss, pos_excursion).
            "pos_excursion": (raw_pos.abs() - 1.0).clamp(min=0.0),
            "depth": computed_depth,
            "confidence": confidence.detach().mean(),
            "depth_threshold": (depth_threshold.detach()
                                if self._adaptive
                                else self._confidence_threshold),
        }

        return output, confidence, metadata

    def get_confidence(self, latents: torch.Tensor) -> torch.Tensor:
        """Get confidence score from latents (for early-halt decisions)."""
        B = latents.shape[0]
        last_tokens = latents[:, -self._n_active:, :]
        pooled = last_tokens.reshape(B, -1)
        return self.confidence_head(pooled)
