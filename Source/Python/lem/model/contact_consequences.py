# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Action-conditioned bodily consequences, trained with the real model's encoders.

Queries are skin sites with measured positions, not object labels or simulator
object coordinates. Commands are the actual execution-form orders normalized at
the boundary. Future achieved positions must never be supplied as action inputs.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class ContactConsequences(nn.Module):
    """Predict what a skin site will feel after a command.

    n_features is the width of ONE skin site's field vector and must match the
    producer. It is a constructor argument rather than a constant because that
    width has already changed once: the tokenizer's touch fields went from 8 to
    10 when the skin-stretch channels landed, and this module kept its 8, so
    every live call died inside nn.Linear with a bare shape error
    (site expected 3+8=11 columns, the body delivered 3+10=13). The default
    stays 8 so a saved 8-wide checkpoint still loads; the trainer passes the
    tokenizer's own field count.
    """

    def __init__(self, d_model, n_heads=4, n_handles=32, n_features=8):
        super().__init__()
        self.n_features = int(n_features)
        self.site = nn.Sequential(nn.Linear(3 + self.n_features, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.action = nn.Sequential(nn.Linear(10, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # Stable actuator identity: swapping left/right orders must not be an
        # invisible permutation of attention memory. UE adapters map their body
        # handles explicitly before this boundary.
        self.handle_embedding = nn.Embedding(n_handles, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.delta = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Linear(d_model, self.n_features))
        self.capacity = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 11))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        nn.init.zeros_(self.capacity[-1].weight)
        nn.init.zeros_(self.capacity[-1].bias)

    def forward(self, context, spatial, spatial_valid, positions, features, site_valid,
                actions, durations, capacity):
        # actions [B,H,handles,9]; time specifies the actual duration of each command.
        b, h, n, _ = actions.shape
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f'skin site features are {features.shape[-1]} wide but this predictor '
                f'was built for {self.n_features}; construct it with '
                f'n_features={features.shape[-1]} (the tokenizer touch field count) ' 
                f'or feed the width it was built for')
        elapsed = durations.cumsum(1)
        timing = elapsed[:, :, None, None].expand(b, h, n, 1)
        if n > self.handle_embedding.num_embeddings:
            raise ValueError('actuator layout exceeds the declared handle contract')
        orders = self.action(torch.cat([actions, timing], -1))
        orders = (orders + self.handle_embedding.weight[:n][None, None]).reshape(b, h * n, -1)
        memory = torch.cat([context, spatial, orders], 1)
        mask = torch.cat([torch.ones(b, context.shape[1], dtype=torch.bool, device=context.device),
                          spatial_valid, torch.ones(b, h * n, dtype=torch.bool, device=context.device)], 1)
        query = self.site(torch.cat([positions, torch.log1p(features.clamp_min(0))], -1))
        read, _ = self.attention(query, memory, memory, key_padding_mask=~mask, need_weights=False)
        hidden = self.norm(query + read)
        delta = self.delta(hidden)
        predicted_log = torch.log1p(features.clamp_min(0)) + delta
        predicted = torch.expm1(predicted_log.clamp(min=0, max=8)) * site_valid[..., None]
        pooled = (hidden * site_valid[..., None]).sum(1) / site_valid.sum(1, keepdim=True).clamp_min(1)
        condition = self.capacity(pooled)
        cap_delta = condition[:, :10]
        return dict(features=predicted, log_features=predicted_log,
                    capacity=(capacity + cap_delta).clamp(0, 1), capacity_delta=cap_delta,
                    curiosity=torch.sigmoid(condition[:, 10:11]))

    @staticmethod
    def loss(prediction, target, valid, capacity, initial_capacity, curiosity=None):
        error = (prediction['log_features'] - torch.log1p(target.clamp_min(0))).square()
        skin = (error * valid[..., None]).sum() / (valid.sum().clamp_min(1) * target.shape[-1])
        recovery = F.mse_loss(prediction['capacity_delta'], capacity - initial_capacity)
        interest = (F.mse_loss(prediction['curiosity'], curiosity) if curiosity is not None
                    else skin.new_zeros(()))
        return skin + recovery + interest, {'skin_log_mse': skin.detach(), 'capacity_mse': recovery.detach(),
                                            'curiosity_mse': interest.detach()}
