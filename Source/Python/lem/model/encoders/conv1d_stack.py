# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Conv1D Stack Encoder — Audio modality.

3x Conv1D (channels from list) -> spectral token + spatial branch (ILD/ITD linear)
-> combine (concat+linear | add | gate).

Input:  [256, 2] spectral + [2] spatial
Output: [1, d_model]

All params from resolved config. Missing key = KeyError.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DStackEncoder(nn.Module):
    """
    Audio encoder with spectral (Conv1D) and spatial (ILD/ITD) branches.

    Input:  spectral [B, freq_bins, channels], spatial [B, spatial_dim]
    Output: [B, 1, d_output]
    """

    def __init__(self, conv_channels: list[int], kernel_size: int,
                 in_channels: int, spatial_branch_input_dim: int,
                 d_internal: int, d_output: int,
                 combine: str, dropout: float, bias: bool,
                 norm_type: str, pool_n_heads: int,
                 freq_band_path: str = "off", n_freq_bands: int = 16):
        super().__init__()
        self.combine_mode = combine
        # ABSOLUTE-FREQUENCY side path (2026-09-09 feasibility review,
        # measured counterexample: same 5-bin pattern at bins 40/120/200
        # produced EXACTLY identical tokens - translation-equivariant convs
        # followed by global average pooling erase absolute frequency, and no
        # amount of training can separate identical representations). Fixed
        # banded means of the RAW spectrum (16 contiguous bands x stereo)
        # through a ZERO-INITIALIZED projection added to the spectral branch:
        # zero init = bit-identical behavior at load (old checkpoints load
        # with strict=False and simply lack the key), training then earns the
        # use of frequency position. Flag-gated because the new parameters
        # change the optimizer's param groups: enable only at a segment
        # launch, not by hot resume with a stale optimizer state.
        self.freq_band_path = freq_band_path == "on"
        if self.freq_band_path:
            self._n_bands = int(n_freq_bands)
            self.band_proj = nn.Linear(self._n_bands * in_channels,
                                       d_internal, bias=False)
            nn.init.zeros_(self.band_proj.weight)

        # Spectral branch: stack of Conv1D layers
        layers = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size, padding=kernel_size // 2,
                          bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            ch_in = ch_out
        self.spectral_conv = nn.Sequential(*layers)

        # Adaptive pool + project spectral to d_internal
        self.spectral_pool = nn.AdaptiveAvgPool1d(1)
        self.spectral_proj = nn.Linear(conv_channels[-1], d_internal, bias=bias)

        # Spatial branch: linear from ILD/ITD features
        self.spatial_proj = nn.Linear(spatial_branch_input_dim, d_internal, bias=bias)

        # Combine
        if combine == "concat":
            self.combine_proj = nn.Linear(d_internal * 2, d_internal, bias=bias)
        elif combine == "gate":
            self.gate = nn.Linear(d_internal * 2, d_internal, bias=bias)
        # "add" needs no extra parameters

        # Final projection
        self.norm = _make_norm(norm_type, d_internal)
        self.proj = (nn.Linear(d_internal, d_output, bias=bias)
                     if d_internal != d_output else nn.Identity())

    def forward(self, spectral: torch.Tensor,
                spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: [B, freq_bins, channels]
            spatial: [B, spatial_dim]
        Returns:
            [B, 1, d_output]
        """
        # Spectral: [B, freq, ch] -> [B, ch, freq] for Conv1d
        s = spectral.transpose(1, 2)
        s = self.spectral_conv(s)            # [B, C_last, freq]
        s = self.spectral_pool(s).squeeze(-1)  # [B, C_last]
        s = self.spectral_proj(s)            # [B, d_internal]
        if self.freq_band_path:
            B, F_, C = spectral.shape
            nb = self._n_bands
            # contiguous equal bands of the raw spectrum: mean energy per
            # band per channel carries WHERE on the frequency axis the sound
            # lives - the quantity the conv+average path provably drops
            bands = spectral[:, : (F_ // nb) * nb, :]
            bands = bands.reshape(B, nb, F_ // nb, C).mean(dim=2)
            s = s + self.band_proj(bands.reshape(B, nb * C))

        # Spatial
        p = self.spatial_proj(spatial)       # [B, d_internal]

        # Combine
        if self.combine_mode == "concat":
            combined = self.combine_proj(torch.cat([s, p], dim=-1))
        elif self.combine_mode == "add":
            combined = s + p
        elif self.combine_mode == "gate":
            g = torch.sigmoid(self.gate(torch.cat([s, p], dim=-1)))
            combined = g * s + (1 - g) * p
        else:
            raise ValueError(f"Unknown combine mode: {self.combine_mode}")

        combined = self.norm(combined)
        combined = self.proj(combined)
        return combined.unsqueeze(1)  # [B, 1, d_output]


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def build_conv1d_stack(config: dict) -> nn.Module:
    """Factory: build Conv1DStackEncoder from resolved encoder config."""
    return Conv1DStackEncoder(
        conv_channels=config["conv_channels"],
        kernel_size=config["kernel_size"],
        in_channels=2,  # Stereo spectral input channels
        spatial_branch_input_dim=config["spatial_branch_input_dim"],
        d_internal=config["_d_internal"],
        d_output=config["_d_output"],
        combine=config["combine"],
        dropout=config["dropout"],
        bias=config["bias"],
        norm_type=config["norm_type"],
        pool_n_heads=config["pool_n_heads"],
        # .get, not [] - resolved configs stored inside pre-2026-09-09
        # checkpoints lack these keys and must keep loading unchanged
        freq_band_path=config.get("freq_band_path", "off"),
        n_freq_bands=config.get("n_freq_bands", 16),
    )
