# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Cross-Patch Attention Encoder — used for BOTH tactile and motor.

Per-patch MLP (input -> hidden -> d_internal) -> N cross-attn layers -> learned pool query.

Input:  [N_sensors, per_patch_input_dim] (tactile) or [n_handles, 8] (motor)
Output: [1, d_model]

cross_handle_attention is an ALIAS — same factory, different default config.

All params from resolved config. Missing key = KeyError.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lem.model.attention_cap import CappedMultiheadAttention


class CrossPatchAttentionEncoder(nn.Module):
    """
    Cross-patch attention encoder.

    Input:  [B, N, input_dim]
    Output: [B, 1, d_output]
    """

    def __init__(self, per_patch_input_dim: int, per_patch_hidden: int,
                 d_internal: int, d_output: int,
                 n_cross_attn_layers: int, n_heads: int,
                 pool_n_heads: int, dropout: float, bias: bool,
                 norm_type: str, input_scale: float = 1.0,
                 patch_pos_embedding: bool = False, n_patches: int = 0):
        super().__init__()

        self.input_scale = input_scale
        # Optional learned "which patch am I" embedding (one vector per patch slot). Without
        # it the encoder only sees the field values of a patch, never its identity, so it
        # cannot build a body map from touch (found on the MIMo test bed, 2026-08-19).
        # Patch order is fixed by the tokenizer (UE sensor order / MIMo body-id order).
        self.patch_pos = None
        if patch_pos_embedding and n_patches > 0:
            self.patch_pos = nn.Parameter(torch.randn(1, n_patches, d_internal) * 0.02)

        # Per-patch MLP: input_dim -> hidden -> d_internal
        self.patch_mlp = nn.Sequential(
            nn.Linear(per_patch_input_dim, per_patch_hidden, bias=bias),
            nn.GELU(),
            nn.Linear(per_patch_hidden, d_internal, bias=bias),
        )

        # Cross-attention layers. The attention layers and the pool below are built as
        # CappedMultiheadAttention (ATTENTION_CAP, lem/model/attention_cap.py): the same parameters, and
        # nn.MultiheadAttention's own forward while they carry no bound. Only the motor encoder's are ever
        # given one (--attention-cap on); the tactile encoder's never are.
        self.layers = nn.ModuleList()
        for _ in range(n_cross_attn_layers):
            self.layers.append(nn.ModuleDict({
                "norm": _make_norm(norm_type, d_internal),
                "attn": CappedMultiheadAttention(
                    d_internal, n_heads, dropout=dropout,
                    bias=bias, batch_first=True,
                ),
                "ffn_norm": _make_norm(norm_type, d_internal),
                "ffn": nn.Sequential(
                    nn.Linear(d_internal, d_internal * 4, bias=bias),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_internal * 4, d_internal, bias=bias),
                    nn.Dropout(dropout),
                ),
            }))

        # Learned pool query
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_internal) * 0.02)
        self.pool_attn = CappedMultiheadAttention(
            d_internal, pool_n_heads, dropout=dropout,
            bias=bias, batch_first=True,
        )
        self.pool_norm = _make_norm(norm_type, d_internal)

        # Project to d_output
        self.proj = (nn.Linear(d_internal, d_output, bias=bias)
                     if d_internal != d_output else nn.Identity())

    def backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, per_patch_input_dim]
        Returns:
            [B, 1, d_output]
        """
        if self.input_scale != 1.0:
            x = x * self.input_scale
        x = self.patch_mlp(x)
        if self.patch_pos is not None:
            n = min(x.shape[1], self.patch_pos.shape[1])
            x = torch.cat([x[:, :n, :] + self.patch_pos[:, :n, :], x[:, n:, :]], dim=1)

        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x, _ = layer["attn"](x, x, x)
            x = x + residual

            residual = x
            x = layer["ffn_norm"](x)
            x = layer["ffn"](x) + residual

        return x

    def head_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pool to single token
        B = x.shape[0]
        q = self.pool_query.expand(B, -1, -1)
        q = self.pool_norm(q)
        out, _ = self.pool_attn(q, x, x)

        out = self.proj(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_forward(self.backbone_forward(x))


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def build_cross_patch_attention(config: dict) -> nn.Module:
    """Factory: build CrossPatchAttentionEncoder from resolved encoder config."""
    return CrossPatchAttentionEncoder(
        per_patch_input_dim=config["per_patch_input_dim"],
        per_patch_hidden=config["per_patch_hidden"],
        d_internal=config["_d_internal"],
        d_output=config["_d_output"],
        n_cross_attn_layers=config["n_cross_attn_layers"],
        n_heads=config["n_heads"],
        pool_n_heads=config["pool_n_heads"],
        dropout=config["dropout"],
        bias=config["bias"],
        norm_type=config["norm_type"],
        input_scale=config.get("input_scale", 1.0),
        patch_pos_embedding=bool(config.get("patch_pos_embedding", False)),
        n_patches=int(config.get("_n_patches_for_pos", 0)),
    )


# Alias: cross_handle_attention is the same factory
build_cross_handle_attention = build_cross_patch_attention
