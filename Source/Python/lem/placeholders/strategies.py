# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Placeholder strategies — generate synthetic data for blocked modalities.

Each strategy produces tensors matching the exact shape the tokenizer/encoder
expects for that modality. Shape comes from resolved config, not hardcoded.

Strategies:
  - gaussian: IID N(mu, sigma) samples, clamped to [0, 1] - see GaussianStrategy for what
    that clamp does to a draw the config asks for at mean 0
  - sparse_contact: K of N sensors active, rest zero (tactile-specific)
  - replay_buffer: Pre-recorded real data replayed with jitter
"""

from __future__ import annotations

from typing import Optional

import torch


class GaussianStrategy:
    """IID Gaussian samples, clipped to a configured range (default the sensor range
    [0, 1]). Pre-allocated buffer.

    THE DEFAULT CLIP RECTIFIES A ZERO-MEAN DRAW, AND THE CONFIG ASKS FOR ONE. With mean
    0.0 and std 0.01 - what master_config.yaml resolves for intermind, the only
    generator built on the live lineage (--audio real makes hearing a measured sense;
    the std-0.1 block is proprioception's, whose data_source is real) - the clip at 0.0
    removes the whole negative half: the closed-form share of exact zeros is 50%
    (measured 50.06% over 1,310,000 numbers), mean std/sqrt(2*pi) = 0.003989. So the
    channel is a one-sided draw with a fixed positive offset, not the N(0, std) the
    config names. Which entries are zero is redrawn every step, so this is a constant
    MEAN, not a constant vector. Measured consequence (audit P012): with the weights of
    the H83-trained intermind encoder held still, this input produces a token that
    moves 0.0006 against a token size of 1.0, while an untrained encoder on the same
    input moves 0.5950 - a mean-0 clipped placeholder ends up delivering a fixed
    direction, which is what PlaceholderSystem exists to prevent. The clip range is a
    config field (placeholder.clip) whose default keeps today's behaviour; clip: null
    gives the N(mean, std) the config names. Vision's placeholder (mean 0.5, std 0.15)
    sits inside [0, 1] and is not rectified.

    generate() returns a PRE-ALLOCATED buffer that the next call overwrites in place: two
    calls return the same object. A caller that keeps the tensor across steps must clone
    it. Nothing does today - lem_model.py stashes the frame tensors with .detach(), which
    shares storage, but PlaceholderSystem.fill_frame stamps every filled modality
    "synthetic" and the re-encoded token for a synthetic modality is discarded.
    """

    def __init__(self, shape: list[int], mean: float, std: float,
                 clip=(0.0, 1.0), device: str = "cpu"):
        self.shape = shape
        self.mean = mean
        self.std = std
        self.clip = None if clip is None else (float(clip[0]), float(clip[1]))
        self.device = device
        self._buf = torch.zeros(1, *shape, device=device)

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        if batch_size != self._buf.shape[0]:
            self._buf = torch.zeros(batch_size, *self.shape, device=self.device)
        self._buf.normal_().mul_(self.std).add_(self.mean)
        if self.clip is not None:
            self._buf.clamp_(*self.clip)
        return self._buf


class SparseContactStrategy:
    """
    K of N sensors active, forces from distribution, rest zero.

    Implements AR(1) temporal correlation for sticky contact masks
    when temporal_correlation > 0.
    """

    def __init__(self, shape: list[int], n_active: int,
                 force_mean: float, force_std: float,
                 temporal_correlation: float = 0.0,
                 device: str = "cpu"):
        self.shape = shape  # [n_sensors, features_per_sensor]
        self.n_sensors = shape[0]
        self.features = shape[1] if len(shape) > 1 else 1
        self.n_active = min(n_active, self.n_sensors)
        self.force_mean = force_mean
        self.force_std = force_std
        self.temporal_correlation = temporal_correlation
        self.device = device

        # AR(1) state: which sensors are currently active
        self._active_mask: Optional[torch.Tensor] = None
        # Pre-allocated buffers
        self._result_buf = torch.zeros(1, *shape, device=device)
        self._forces_buf = torch.zeros(self.n_sensors, self.features,
                                       device=device)
        self._mask_buf = torch.zeros(self.n_sensors, dtype=torch.bool,
                                     device=device)

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        if batch_size != self._result_buf.shape[0]:
            self._result_buf = torch.zeros(batch_size, *self.shape,
                                           device=self.device)
        self._result_buf.zero_()

        for b in range(batch_size):
            if self._active_mask is not None and self.temporal_correlation > 0:
                # AR(1): keep previous mask with probability = temporal_correlation
                keep = torch.rand(self.n_sensors, device=self.device) < self.temporal_correlation
                self._mask_buf.copy_(self._active_mask)
                # For non-kept sensors, resample
                n_resample = (~keep).sum().item()
                if n_resample > 0:
                    new_active = torch.randperm(
                        self.n_sensors, device=self.device
                    )[:self.n_active]
                    new_mask = torch.zeros(self.n_sensors, dtype=torch.bool, device=self.device)
                    new_mask[new_active] = True
                    self._mask_buf[~keep] = new_mask[~keep]
            else:
                # Fresh random mask
                active_idx = torch.randperm(
                    self.n_sensors, device=self.device
                )[:self.n_active]
                self._mask_buf.zero_()
                self._mask_buf[active_idx] = True

            self._active_mask = self._mask_buf

            # Generate forces for active sensors (in-place)
            self._forces_buf.normal_().mul_(self.force_std).add_(
                self.force_mean).clamp_(0.0, 1.0)
            self._forces_buf[~self._mask_buf] = 0.0
            self._result_buf[b] = self._forces_buf

        return self._result_buf


class ReplayBufferStrategy:
    """
    Replay pre-recorded real data with optional jitter.

    Cycles through stored frames. If no data loaded, falls back to gaussian.
    """

    def __init__(self, shape: list[int], device: str = "cpu",
                 jitter_std: float = 0.01):
        self.shape = shape
        self.device = device
        self.jitter_std = jitter_std
        self._buffer: list[torch.Tensor] = []
        self._index = 0
        # Pre-allocated buffers
        self._out_buf = torch.zeros(1, *shape, device=device)
        self._jitter_buf = torch.zeros(*shape, device=device)

    def load_data(self, data: list[torch.Tensor]) -> None:
        """Load pre-recorded tensors into replay buffer."""
        self._buffer = [d.to(self.device) for d in data]
        self._index = 0

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        if batch_size != self._out_buf.shape[0]:
            self._out_buf = torch.zeros(batch_size, *self.shape,
                                        device=self.device)

        if not self._buffer:
            # Fallback to gaussian (in-place)
            self._out_buf.normal_().mul_(0.1).clamp_(0.0, 1.0)
            return self._out_buf

        for b in range(batch_size):
            src = self._buffer[self._index % len(self._buffer)]
            self._out_buf[b].copy_(src)
            if self.jitter_std > 0:
                self._jitter_buf.normal_().mul_(self.jitter_std)
                self._out_buf[b].add_(self._jitter_buf).clamp_(0.0, 1.0)
            self._index += 1

        return self._out_buf


STRATEGY_REGISTRY = {
    "gaussian": GaussianStrategy,
    "sparse_contact": SparseContactStrategy,
    "replay_buffer": ReplayBufferStrategy,
}
