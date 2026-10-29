# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Restore trainable activation range without changing learned tensors.

The frozen correction preserves the initial function for every input. Its buffers
are checkpointed; they are not optimizer parameters. The measured output gain
matches newly responsive units to the existing responsive units' sensitivity.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'visual_activation_compensation_v1'


class VisualActivationCompensation(nn.Module):
    def __init__(self, input_width, hidden_width, output_width):
        super().__init__()
        self.register_buffer('shift', torch.zeros(hidden_width))
        self.register_buffer('gain', torch.ones(hidden_width))
        self.register_buffer('original_input_weight', torch.zeros(hidden_width, input_width))
        self.register_buffer('original_input_bias', torch.zeros(hidden_width))
        self.register_buffer('original_output_weight', torch.zeros(output_width, hidden_width))
        self.register_buffer('original_output_bias', torch.zeros(output_width))
        self.register_buffer('enabled', torch.tensor(False))
        self.enabled_py = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state_dict = state_dict.copy()
        keys = [prefix+name for name in self._buffers]
        if not any(key in state_dict for key in keys):
            # A legacy checkpoint has the identity route and no calibration.
            state_dict.update({prefix+name: (torch.ones_like(value) if name=='gain' else torch.zeros_like(value))
                               for name, value in self._buffers.items()})
            state_dict[prefix+'enabled'] = torch.tensor(False)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self.enabled_py = bool(self.enabled)
        if (not all(bool(torch.isfinite(value).all()) for value in self._buffers.values())
                or bool((self.shift < 0).any()) or bool((self.gain <= 0).any()) or bool((self.gain > 1).any())):
            error_msgs.append(prefix+'invalid visual activation calibration buffers')

    @torch.no_grad()
    def configure(self, first, final, shift, gain):
        shift = torch.as_tensor(shift, device=self.shift.device, dtype=self.shift.dtype)
        gain = torch.as_tensor(gain, device=self.gain.device, dtype=self.gain.dtype)
        if (shift.shape != self.shift.shape or gain.shape != self.gain.shape
                or not torch.isfinite(shift).all() or not torch.isfinite(gain).all()
                or (shift < 0).any() or (gain <= 0).any() or (gain > 1).any()):
            raise ValueError('Invalid measured activation shift/gain')
        if self.enabled_py:
            if not torch.equal(shift,self.shift) or not torch.equal(gain,self.gain):
                raise ValueError('Cannot replace a learned activation calibration')
            return False
        self.original_input_weight.copy_(first.weight)
        self.original_input_bias.copy_(first.bias)
        self.original_output_weight.copy_(final.weight)
        self.original_output_bias.copy_(final.bias)
        self.shift.copy_(shift)
        self.gain.copy_(gain)
        self.enabled.fill_(True)
        self.enabled_py = True
        return True

    def forward_with_preactivation(self, inputs, first, final):
        if not self.enabled_py:
            pre = first(inputs)
            return final(F.gelu(pre)), pre
        # The retained original path keeps its exact production precision.
        original = F.linear(inputs, self.original_input_weight, self.original_input_bias)
        previous = F.linear(F.gelu(original), self.original_output_weight, self.original_output_bias)
        # Both delta branches use matching full precision. Their input-gradient
        # cancellation otherwise rounds in bf16 before reaching the shared
        # input, perturbing even the initially unchanged function's derivative.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            x = inputs.float() if inputs.dtype in (torch.float16,torch.bfloat16) else inputs
            pre = F.linear(x, first.weight.to(x.dtype), first.bias.to(x.dtype))+self.shift.to(x.dtype)
            output = F.linear(F.gelu(pre)*self.gain.to(x.dtype),
                              final.weight.to(x.dtype), final.bias.to(x.dtype))
            fixed = F.linear(x,self.original_input_weight.to(x.dtype),self.original_input_bias.to(x.dtype))
            shifted = F.linear(F.gelu(fixed+self.shift.to(x.dtype))*self.gain.to(x.dtype),
                               self.original_output_weight.to(x.dtype), self.original_output_bias.to(x.dtype))
            delta = output-shifted
        return previous+delta.to(previous.dtype), pre

    def forward(self, inputs, first, final):
        return self.forward_with_preactivation(inputs,first,final)[0]
