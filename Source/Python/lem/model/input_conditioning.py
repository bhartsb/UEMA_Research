# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Preserve a learned linear map while expressing its input in measured units."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'visual_linear_input_conditioning_v1'


class InputScaledLinear(nn.Linear):
    """Hooks still see the original input; only this linear path uses its scale.

    That distinction preserves liveness repair's separately normalized rows.
    The scale is part of the model state so its weights cannot be reloaded in
    different units. Old checkpoints without a scale use the identity.
    """
    def forward(self, x):
        return F.linear(x / self.input_scale, self.weight, self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'input_scale'
        state_dict = state_dict.copy()
        if key not in state_dict:
            state_dict[key] = torch.ones_like(self.input_scale)
        value = state_dict[key]
        if value.numel() != 1 or not bool(torch.isfinite(value).all()) or float(value) < 1:
            error_msgs.append(f'{key}: input scale must be a finite scalar >= 1')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


def install(linear: nn.Linear) -> InputScaledLinear:
    """Keep the original parameter objects and consume no random numbers."""
    if isinstance(linear, InputScaledLinear):
        return linear
    if type(linear) is not nn.Linear:
        raise TypeError('Input conditioning requires an ordinary Linear')
    linear.__class__ = InputScaledLinear
    linear.register_buffer('input_scale', torch.ones((), device=linear.weight.device,
                                                    dtype=linear.weight.dtype))
    return linear


@torch.no_grad()
def migrate(linear: InputScaledLinear, scale: float, optimizer=None) -> dict:
    """Carry the function and Adam history into new units exactly once.

    W' = W*s, x' = x/s; gradient moments therefore divide by s and s squared.
    Biases, step counters, parameter groups, learning rates and weight decay stay
    unchanged. Decoupled weight decay has the same effect on the represented map.
    """
    if not isinstance(linear, InputScaledLinear):
        raise TypeError('Install input scaling before migration')
    scale = float(scale)
    if not math.isfinite(scale) or scale < 1:
        raise ValueError('Input scale must be finite and >= 1')
    # Receipt and arithmetic must use the value the persistent buffer can hold.
    # Otherwise e.g. JSON 261.53 saves as float32 261.529998... and cannot reload.
    scale = float(torch.tensor(scale, dtype=linear.input_scale.dtype))
    if not math.isfinite(scale):
        raise ValueError('Input scale is not representable by the model dtype')
    previous = float(linear.input_scale)
    if previous == scale:
        return dict(version=VERSION, scale=scale, migrated=False, optimizer_moments=[])
    if previous != 1:
        raise ValueError('An already conditioned head cannot silently change units')
    weight = linear.weight
    state = {} if optimizer is None else optimizer.state.get(weight, {})
    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if name in state and (state[name].shape != weight.shape or not torch.isfinite(state[name]).all()):
            raise ValueError(f'Invalid Adam state for {name}')
    if not torch.isfinite(weight).all() or not torch.isfinite(weight * scale).all():
        raise ValueError('Conditioning would make the learned weight non-finite')
    weight.mul_(scale)
    changed=[]
    for name, power in (('exp_avg',1),('exp_avg_sq',2),('max_exp_avg_sq',2)):
        if name in state:
            state[name].div_(scale ** power)
            changed.append(name)
    linear.input_scale.fill_(scale)
    return dict(version=VERSION, scale=scale, migrated=True, optimizer_moments=changed)
