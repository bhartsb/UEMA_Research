# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
LatentRegularizer — constrain forward model predictions to encoder output distribution.

UEMA Spec v3.4 §6.3: Distribution shift mitigation via running statistics.

Tracks running mean/variance of real encoder outputs across training.
When forward model predictions deviate beyond ±σ_threshold (default 2σ),
applies a quadratic penalty on the excess z-score.  Inside the envelope,
no penalty at all.

Formula:
    z_score   = (prediction - running_mean) / running_std
    excess    = relu(|z_score| - sigma_threshold)
    penalty   = mean(excess²)

Running statistics are per-dimension, shared across modalities. All encoder
outputs live in the same d_model-dimensional space, so one set of statistics
captures the overall distribution.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


class LatentRegularizer:
    """
    Track real encoder output distribution and penalize predictions
    that drift beyond ±sigma_threshold standard deviations.

    Args:
        d_model: Embedding dimension (default 768).
        sigma_threshold: Number of std deviations before penalty activates (default 2.0).
        momentum: EMA momentum for running stats (default 0.99).
    """

    def __init__(self, d_model: int = 768, sigma_threshold: float = 2.0,
                 momentum: float = 0.99):
        self.d_model = d_model
        self.sigma_threshold = sigma_threshold
        self.momentum = momentum
        self.running_mean = torch.zeros(d_model)
        self.running_var = torch.ones(d_model)
        self._initialized = False

    def to(self, device: torch.device) -> "LatentRegularizer":
        """Move running stats to target device."""
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        return self

    @torch.no_grad()
    def update(self, encoder_outputs: Dict[str, torch.Tensor],
               valid: Optional[Dict[str, str]] = None) -> None:
        """
        Update running mean/var from real encoder outputs using EMA.

        Called every training step with real (not predicted) encoder outputs.

        Args:
            encoder_outputs: {modality: [B, d_model]} detached real encoder outputs.
        """
        # A synthetic channel is a fresh Gaussian draw each moment. Folding it into the
        # running mean/var makes the reference distribution partly noise, and the penalty
        # then pulls real predictions toward it. Excluded here and in penalty().
        if valid is not None:
            encoder_outputs = {m: v for m, v in encoder_outputs.items()
                               if str(getattr(valid.get(m, "real"), "value",
                                              valid.get(m, "real"))) == "real"}
        if not encoder_outputs:
            return

        # Stack all modality outputs: [N, d_model] where N = sum of batch sizes
        all_outputs = []
        for tensor in encoder_outputs.values():
            # Handle both [B, d_model] and [B, 1, d_model] shapes
            t = tensor.squeeze(1) if tensor.dim() == 3 else tensor
            all_outputs.append(t)

        stacked = torch.cat(all_outputs, dim=0)  # [N, d_model]
        batch_mean = stacked.mean(dim=0)  # [d_model]
        batch_var = stacked.var(dim=0, unbiased=False)  # [d_model]

        if not self._initialized:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
            self._initialized = True
        else:
            m = self.momentum
            self.running_mean.mul_(m).add_(batch_mean, alpha=1 - m)
            self.running_var.mul_(m).add_(batch_var, alpha=1 - m)

    def penalty(self, predictions: Dict[str, torch.Tensor],
                valid: Optional[Dict[str, str]] = None) -> torch.Tensor:
        """
        Compute quadratic penalty for predictions outside ±sigma_threshold.

        Gradient flows to prediction tensors but NOT to running stats
        (running stats are updated under no_grad in update()).

        Args:
            predictions: {modality: [B, d_model]} forward model predictions.
                Should contain only predicted modalities (not motor).

        Returns:
            Scalar penalty tensor with gradient flow to predictions.
            valid: {modality: "real"|"synthetic"|"stale"}; non-real channels are dropped.
        """
        if valid is not None:
            predictions = {m: v for m, v in predictions.items()
                           if str(getattr(valid.get(m, "real"), "value",
                                          valid.get(m, "real"))) == "real"}
        if not predictions:
            device = self.running_mean.device
            return torch.tensor(0.0, device=device)

        running_std = (self.running_var + 1e-8).sqrt()  # [d_model], no grad

        excess_list = []
        for tensor in predictions.values():
            t = tensor.squeeze(1) if tensor.dim() == 3 else tensor
            z_score = (t - self.running_mean) / running_std  # [B, d_model]
            excess = F.relu(z_score.abs() - self.sigma_threshold)  # [B, d_model]
            excess_list.append(excess)

        all_excess = torch.cat(excess_list, dim=0)  # [N, d_model]
        return (all_excess ** 2).mean()
