# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The situation test's rebuilt command path carries the persistent memory's bias, as the live step does
(2026-09-27): before, T1 and T2 ran the forward model without it, so the controller saw an input missing the
memory constant's image. No model or world is built: a stand-in records what the path is given."""
import torch

from mimo_bridge.tools.situation_test import _cmd


class _StandIn:
    def __init__(self):
        self.seen = []

    def forward_model(self, tokens, ctx, **kwargs):
        self.seen.append(kwargs)
        return torch.zeros(1, 6, 4), None, None

    def motor_controller(self, latents, handle_state, wellness_grad=None):
        return torch.ones(1, 3), None, None


def test_the_rebuilt_command_passes_the_memory_bias_when_the_live_step_had_one():
    model = _StandIn()
    tokens, ctx, hs = {"vision": torch.zeros(1, 1, 4)}, torch.zeros(1, 2, 4), torch.zeros(1, 26, 12)
    bias = torch.full((1, 6, 4), 7.0)
    _cmd(model, tokens, ctx, hs, memory_bias=bias)
    assert "memory_bias" in model.seen[-1] and torch.equal(model.seen[-1]["memory_bias"], bias)
    _cmd(model, tokens, ctx, hs)                       # no memory (or memory off): nothing extra passed
    assert "memory_bias" not in model.seen[-1]
