# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Duplicated-body invariance of the planner's effort penalty (external review
2026-09-08 finding 4, spec 10.6 upstream gate).

The legacy penalty sums squared strength over batch AND handles while the
prediction loss averages - so replicating one body N times multiplies the
relative effort penalty by N with identical per-body experience. The per-body
form must be invariant to duplication; the legacy form must show exactly the
N-times scaling (that measured defect is what the flag exists to fix).
"""
import torch

from lem.model.intention_generator import IntentionGenerator, effort_penalty


def test_per_body_penalty_invariant_to_duplicated_bodies():
    torch.manual_seed(0)
    s1 = torch.rand(1, 24, 1)
    s8 = s1.repeat(8, 1, 1)
    p1 = effort_penalty(s1, 0.01, per_body=True)
    p8 = effort_penalty(s8, 0.01, per_body=True)
    assert torch.allclose(p1, p8), (
        f"per-body penalty must not change when a body is duplicated: {p1} vs {p8}")


def test_legacy_penalty_scales_with_body_count():
    torch.manual_seed(0)
    s1 = torch.rand(1, 24, 1)
    s8 = s1.repeat(8, 1, 1)
    p1 = effort_penalty(s1, 0.01, per_body=False)
    p8 = effort_penalty(s8, 0.01, per_body=False)
    assert torch.allclose(8.0 * p1, p8), (
        f"legacy penalty documents the defect: 8 bodies must read 8x ({p1} vs {p8})")


def test_penalty_keeps_gradient():
    s = torch.rand(2, 24, 1, requires_grad=True)
    effort_penalty(s, 0.01, per_body=True).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_generator_reads_flag_from_config():
    cfg = {"training": {"intention": {"effort_per_body": True}},
           "body": {"n_handles": 24}, "model": {"d_model": 128},
           "_active_modalities": ["vision", "proprioception"]}
    gen = IntentionGenerator(cfg, torch.nn.Identity(), torch.nn.Identity(),
                             torch.nn.Identity())
    assert gen._effort_per_body is True
    cfg["training"]["intention"]["effort_per_body"] = False
    gen2 = IntentionGenerator(cfg, torch.nn.Identity(), torch.nn.Identity(),
                              torch.nn.Identity())
    assert gen2._effort_per_body is False


# ---------------------------------------------------------------- audit P112 (2026-09-18)
def test_bridge_carries_the_flag_to_the_generator():
    """The run depends on lem_model.py wiring --planner-effort-per-body into the
    generator; the existing tests build IntentionGenerator from a raw config and so
    pinned nothing about that line."""
    from mimo_bridge.lem_model import LemLite
    on = LemLite(device="cpu", seed=0, imagination=False, intention=True, intention_execute=False,
                 planner_effort_per_body=True)
    off = LemLite(device="cpu", seed=0, imagination=False, intention=True, intention_execute=False,
                  planner_effort_per_body=False)
    assert on.intention_gen._effort_per_body is True
    assert off.intention_gen._effort_per_body is False


def test_effort_half_ratio_to_goal_half_is_body_count():
    """26 handles at strength 0.4742 - H83's measured median commanded strength
    (cmd_exec.strength_mean over 137 windows, replay_training_stats.jsonl): the legacy
    form at 8 bodies is exactly 8.000x its one-body value and 8.000x the per-body form."""
    s1 = torch.full((1, 26, 1), 0.4742)
    s8 = s1.repeat(8, 1, 1)
    legacy8 = effort_penalty(s8, 0.01, per_body=False)
    legacy1 = effort_penalty(s1, 0.01, per_body=False)
    per8 = effort_penalty(s8, 0.01, per_body=True)
    assert abs(float(legacy8) / float(legacy1) - 8.0) < 1e-5     # float32 sums: measured 2.0e-6 off
    assert abs(float(legacy8) / float(per8) - 8.0) < 1e-5
    assert abs(float(legacy8) - 0.4677) < 2e-3          # the measured H83 effort half
