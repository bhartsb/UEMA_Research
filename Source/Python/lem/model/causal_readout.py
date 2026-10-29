# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Isolated matched-encoder readout for experienced held-command consequences.

Not registered in the production learner. The optimizer owns only these new
weights. Raw windows must be re-encoded when the source learner changes.
"""
import torch
from torch import nn


class CausalReadout(nn.Module):
    def __init__(self, encoded_width, target_width, action_width, hidden=128):
        super().__init__()
        self.history = nn.GRU(encoded_width+target_width+1, hidden, batch_first=True)
        self.command = nn.Sequential(nn.Linear(action_width+1, hidden), nn.SiLU())
        self.dynamics = nn.GRUCell(hidden, hidden)
        self.output = nn.Linear(hidden, target_width)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, encoded, sensed, history_dt, actions, durations):
        if encoded.shape[:2] != sensed.shape[:2] or history_dt.shape != encoded.shape[:2]:
            raise ValueError('own encoded and fixed sensory history chronology differs')
        if actions.shape[:2] != durations.shape or (durations <= 0).any():
            raise ValueError('invalid executed command chronology')
        _, h = self.history(torch.cat((encoded, sensed, history_dt[..., None]), -1))
        h = h[0]; output = []
        for t in range(actions.shape[1]):
            h = self.dynamics(self.command(torch.cat((actions[:, t], durations[:, t, None]), -1)), h)
            output.append(self.output(h))
        return torch.stack(output, 1)


def body_shuffle(actions, gaze_handle, permutation):
    """Same decision/context, wrong body order, unchanged prospective gaze."""
    if actions.ndim != 4 or actions.shape[-1] != 9:
        raise ValueError('expected candidate x time x handle x channel')
    out = actions[permutation].clone()
    out[:, :, gaze_handle] = actions[:, :, gaze_handle]
    return out
