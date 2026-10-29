# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Swin Patch Encoder — Vision modality.

Patch embed -> angular position encoding -> N Swin blocks (window attn + FFN) -> learned pool query.

Input:  [n_patches, patch_dim]
Output: [1, d_model]

All params from resolved config. Missing key = KeyError.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularPositionEncoding(nn.Module):
    """Encode a patch's angular position into a hidden representation.

    Two inputs is the minimum: azimuth and elevation, where in the world the patch looks.
    A binocular, foveated feed wants four - azimuth, elevation, which eye, and how much of
    the world one patch covers. Without the eye number the encoder cannot tell the two eyes
    apart at all (everything before the pool is per-patch and the pool is one query over the
    set), so disparity, the whole point of having two eyes, is not representable.
    """

    def __init__(self, hidden_dim: int, d_output: int, n_inputs: int = 2):
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_inputs, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_output),
        )

    def forward(self, angular_pos: torch.Tensor) -> torch.Tensor:
        return self.mlp(angular_pos)


class SwinBlock(nn.Module):
    """Single Swin Transformer block: window attention + FFN."""

    def __init__(self, dim: int, n_heads: int, window_size: int,
                 ffn_expansion: int, dropout: float, bias: bool,
                 norm_type: str):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.dim = dim

        self.norm1 = _make_norm(norm_type, dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, bias=bias, batch_first=True,
        )
        self.norm2 = _make_norm(norm_type, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_expansion, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        ws = self.window_size

        # Partition into windows
        if N % ws != 0:
            # Pad to window multiple
            pad_n = ws - (N % ws)
            x = F.pad(x, (0, 0, 0, pad_n))
            N_padded = N + pad_n
        else:
            N_padded = N
            pad_n = 0

        # SHIFTED WINDOWS (B5 fix, 2026-08-26). The windows never moved: no roll
        # or shift existed anywhere in this file, so with fixed 7-patch windows
        # NOTHING could cross a window boundary no matter how many layers ran -
        # information from patch 6 could never reach patch 7. Alternate blocks now
        # roll the sequence by half a window before partitioning (and roll back
        # after), which is the standard Swin design this encoder is named for.
        shift = (ws // 2) if getattr(self, "shifted", False) else 0
        if shift:
            x = torch.roll(x, -shift, dims=1)
        n_windows = N_padded // ws
        # [B, n_windows, ws, D]
        x_win = x.reshape(B, n_windows, ws, D)
        # [B*n_windows, ws, D]
        x_win = x_win.reshape(B * n_windows, ws, D)

        # Window self-attention
        residual = x_win
        x_win = self.norm1(x_win)
        x_win, _ = self.attn(x_win, x_win, x_win)
        x_win = x_win + residual

        # FFN
        residual = x_win
        x_win = self.norm2(x_win)
        x_win = self.ffn(x_win) + residual

        # Unpartition
        x_win = x_win.reshape(B, N_padded, D)
        if shift:
            x_win = torch.roll(x_win, shift, dims=1)
        if pad_n > 0:
            x_win = x_win[:, :N, :]

        return x_win


class LearnedPoolQuery(nn.Module):
    """Pool a variable-length sequence into a single token via learned query + cross-attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float, bias: bool):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, bias=bias, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> [B, 1, D]
        #
        # QK-NORM (B5 fix, 2026-08-26). The attention logit is q.k/sqrt(d), and
        # nothing bounded either side: measured on H31's final checkpoint the
        # largest pool logit reached 1,602,426, at which point softmax is an argmax
        # - 4,636 of 4,640 attention weights were EXACTLY 0.0 in fp32 and each head
        # read exactly one window. Combined with unshifted windows, the whole
        # vision token was a function of 21-26 of 1,160 patches (1.8-2.3%) and the
        # other ~1,130 received literally zero gradient. Normalising q and k to
        # unit length caps the logit at sqrt(d_head), which keeps the softmax soft
        # forever - measured to restore ~450 effective patches per head with the
        # already-trained weights.
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        q = self.norm(q)
        q = torch.nn.functional.normalize(q, dim=-1) * (q.shape[-1] ** 0.25)
        k = torch.nn.functional.normalize(x, dim=-1) * (x.shape[-1] ** 0.25)
        out, _ = self.attn(q, k, x)
        return out  # [B, 1, D]


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def build_swin_patch(config: dict) -> nn.Module:
    """Factory: build SwinPatchEncoder from resolved encoder config."""
    return SwinPatchEncoder(
        n_patches=config["_n_patches"],
        patch_dim=config["_patch_dim"],
        d_internal=config["_d_internal"],
        d_output=config["_d_output"],
        patch_size=config["patch_size"],
        n_swin_layers=config["n_swin_layers"],
        n_heads=config["n_heads"],
        window_size=config["window_size"],
        angular_pos_hidden=config["angular_pos_hidden"],
        ffn_expansion=config["ffn_expansion"],
        dropout=config["dropout"],
        bias=config["bias"],
        norm_type=config["norm_type"],
        pool_n_heads=config["pool_n_heads"],
        patch_subsample=config.get("patch_subsample"),
        angular_pos_inputs=config.get("angular_pos_inputs", 2),
    )


class SwinPatchEncoder(nn.Module):
    """
    Vision encoder: patch embed + angular position + Swin blocks + learned pool.

    Input:  [B, n_patches, patch_dim] (flattened RGB patches)
    Output: [B, 1, d_model]
    """

    def __init__(self, n_patches: int, patch_dim: int,
                 d_internal: int, d_output: int,
                 patch_size: int, n_swin_layers: int,
                 n_heads: int, window_size: int,
                 angular_pos_hidden: int, ffn_expansion: int,
                 dropout: float, bias: bool, norm_type: str,
                 pool_n_heads: int,
                 patch_subsample: int | None = None,
                 angular_pos_inputs: int = 2):
        super().__init__()
        self.patch_subsample = patch_subsample

        # Patch embedding: patch_dim -> d_internal
        self.patch_embed = nn.Linear(patch_dim, d_internal, bias=bias)

        # Angular position encoding
        self.angular_pos_enc = AngularPositionEncoding(
            angular_pos_hidden, d_internal, n_inputs=angular_pos_inputs,
        )

        # Swin blocks
        self.blocks = nn.ModuleList([
            SwinBlock(d_internal, n_heads, window_size,
                      ffn_expansion, dropout, bias, norm_type)
            for _ in range(n_swin_layers)
        ])
        # odd blocks shift their windows by half a window (see SwinBlock.forward);
        # a plain attribute rather than a constructor arg so old checkpoints load
        # unchanged - the shift has no parameters
        for bi, blk in enumerate(self.blocks):
            blk.shifted = bool(bi % 2)

        # Pool to single token
        self.pool = LearnedPoolQuery(d_internal, pool_n_heads, dropout, bias)

        # Project to d_output if different from d_internal
        self.proj = (nn.Linear(d_internal, d_output, bias=bias)
                     if d_internal != d_output else nn.Identity())

    def backbone_forward(self, patches: torch.Tensor,
                         angular_pos: torch.Tensor | None = None) -> torch.Tensor:
        """Run patch_embed + angular_pos + subsample + Swin blocks only.

        Used by experience replay to cache expensive backbone features (frozen)
        separately from the cheap trainable head (pool + proj).

        Args:
            patches: [B, n_patches, patch_dim]
            angular_pos: [B, n_patches, 2] (azimuth, elevation) — optional
        Returns:
            [B, N, d_internal] — post-Swin features, before pool+proj.
        """
        x = self.patch_embed(patches)

        if angular_pos is not None:
            x = x + self.angular_pos_enc(angular_pos)

        if self.patch_subsample and x.shape[1] > self.patch_subsample:
            idx = torch.randperm(x.shape[1], device=x.device)[:self.patch_subsample]
            idx = idx.sort().values
            x = x[:, idx]

        for block in self.blocks:
            x = block(x)

        return x

    def head_forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """Run pool + proj on pre-computed backbone features.

        Args:
            backbone_features: [B, N, d_internal] from backbone_forward().
        Returns:
            [B, 1, d_output]
        """
        x = self.pool(backbone_features)
        x = self.proj(x)
        return x

    def forward(self, patches: torch.Tensor,
                angular_pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            patches: [B, n_patches, patch_dim]
            angular_pos: [B, n_patches, 2] (azimuth, elevation) — optional
        Returns:
            [B, 1, d_output]
        """
        x = self.backbone_forward(patches, angular_pos)
        return self.head_forward(x)
