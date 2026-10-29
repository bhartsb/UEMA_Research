# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
MLP Project Encoder — Proprioception and Voice modalities.

N-layer MLP: input -> hidden[0] -> ... -> hidden[-1] -> d_output.

Input:  [_input_dim] flat vector
Output: [1, d_model]

All params from resolved config. Missing key = KeyError.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class MLPProjectEncoder(nn.Module):
    """
    Simple MLP encoder for flat vector inputs.

    Input:  [B, input_dim]
    Output: [B, 1, d_output]
    """

    def __init__(self, input_dim: int, hidden_dims: list[int],
                 d_output: int, activation: str,
                 dropout: float, bias: bool, norm_type: str):
        super().__init__()

        act_cls = _ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim, bias=bias),
                _make_norm(norm_type, h_dim),
                act_cls(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final projection to d_output
        layers.append(nn.Linear(in_dim, d_output, bias=bias))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            [B, 1, d_output]
        """
        out = self.mlp(x)  # [B, d_output]
        return out.unsqueeze(1)  # [B, 1, d_output]


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def build_mlp_project(config: dict) -> nn.Module:
    """Factory: build MLPProjectEncoder from resolved encoder config."""
    return MLPProjectEncoder(
        input_dim=config["_input_dim"],
        hidden_dims=config["hidden_dims"],
        d_output=config["_d_output"],
        activation=config["activation"],
        dropout=config["dropout"],
        bias=config["bias"],
        norm_type=config["norm_type"],
    )
