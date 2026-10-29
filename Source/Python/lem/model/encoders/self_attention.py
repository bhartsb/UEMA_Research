# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Self-Attention Pool Encoder — InterMind modality.

Embed (d_model+3 -> d_internal) -> N self-attn layers -> learned pool query.

Input:  [n_agents, d_model+3]
Output: [1, d_model]

All params from resolved config. Missing key = KeyError.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionPoolEncoder(nn.Module):
    """
    Self-attention encoder for multi-agent (InterMind) data.

    Input:  [B, n_agents, embed_input_dim]
    Output: [B, 1, d_output]
    """

    def __init__(self, embed_input_dim: int, d_internal: int,
                 d_output: int, n_self_attn_layers: int,
                 n_heads: int, pool_n_heads: int,
                 dropout: float, bias: bool, norm_type: str):
        super().__init__()

        # Embed: embed_input_dim -> d_internal
        self.embed = nn.Linear(embed_input_dim, d_internal, bias=bias)

        # Self-attention layers
        self.layers = nn.ModuleList()
        for _ in range(n_self_attn_layers):
            self.layers.append(nn.ModuleDict({
                "norm": _make_norm(norm_type, d_internal),
                "attn": nn.MultiheadAttention(
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
        self.pool_attn = nn.MultiheadAttention(
            d_internal, pool_n_heads, dropout=dropout,
            bias=bias, batch_first=True,
        )
        self.pool_norm = _make_norm(norm_type, d_internal)

        # Project to d_output
        self.proj = (nn.Linear(d_internal, d_output, bias=bias)
                     if d_internal != d_output else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_agents, embed_input_dim]
        Returns:
            [B, 1, d_output]
        """
        x = self.embed(x)

        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x, _ = layer["attn"](x, x, x)
            x = x + residual

            residual = x
            x = layer["ffn_norm"](x)
            x = layer["ffn"](x) + residual

        # Pool to single token
        B = x.shape[0]
        q = self.pool_query.expand(B, -1, -1)
        q = self.pool_norm(q)
        out, _ = self.pool_attn(q, x, x)

        out = self.proj(out)
        return out


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def build_self_attention_pool(config: dict) -> nn.Module:
    """Factory: build SelfAttentionPoolEncoder from resolved encoder config."""
    return SelfAttentionPoolEncoder(
        embed_input_dim=config["_embed_input_dim"],
        d_internal=config["_d_internal"],
        d_output=config["_d_output"],
        n_self_attn_layers=config["n_self_attn_layers"],
        n_heads=config["n_heads"],
        pool_n_heads=config["pool_n_heads"],
        dropout=config["dropout"],
        bias=config["bias"],
        norm_type=config["norm_type"],
    )
