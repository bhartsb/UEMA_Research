# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The lessons' pulls on the vision head and the fold's lever (docs/gain_of_function_v4/vision_head_pulls.py).

Synthetic heads, optimizer states and rows only: no checkpoint. They pin the step a lesson would get from AdamW's
saved state alone, the shared change's projection on the offset and its undoing, the lever (the largest move one step
of at most lr per parameter can give) and that a large constant in the values' input multiplies it, and the three
declared rules."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[4]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def P():
    return _load("vision_head_pulls", "docs/gain_of_function_v4/vision_head_pulls.py")


@pytest.fixture(scope="module")
def T():
    return _load("test_vision_memory_ablation_helpers", "Source/Python/mimo_bridge/tests/test_vision_memory_ablation.py")


def test_the_step_a_lesson_gets_from_the_saved_state(P):
    p = nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=0.01, eps=1e-8)
    opt.state[p] = {"step": torch.tensor(1e6), "exp_avg": torch.zeros(3), "exp_avg_sq": torch.tensor([4.0, 1.0, 0.0])}
    g = torch.tensor([2.0, -1.0, 1e-12])
    step = P.adam_step(opt, p, g, factor=0.5)
    # -lr f g / (sqrt(v) + eps): 0.01 x 0.5 x 2 / 2, -0.01 x 0.5 x -1 / 1, and a zero second moment leaves eps
    assert step[0] == pytest.approx(-0.005, rel=1e-6)
    assert step[1] == pytest.approx(0.005, rel=1e-6)
    assert step[2] == pytest.approx(-0.01 * 0.5 * 1e-12 / 1e-8, rel=1e-6)


def test_the_shared_change_is_projected_on_the_offset_and_undone(P):
    lin = nn.Linear(2, 3)
    x = torch.randn(10, 2)
    before = [q.detach().clone() for q in lin.parameters()]
    steps = [torch.zeros_like(lin.weight), torch.tensor([1.0, 0.0, -2.0])]
    d = P.shared_change(lambda: lin(x), list(lin.parameters()), steps)
    assert torch.allclose(d, torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64), atol=1e-6)
    assert all(torch.equal(q, b) for q, b in zip(lin.parameters(), before))
    assert P.along(d, torch.tensor([2.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert P.along(d, torch.tensor([0.0, 0.0, 1.0])) == pytest.approx(-2.0)
    assert P.along(d, torch.zeros(3)) is None


def test_the_lever_and_what_a_large_constant_input_does_to_it(P):
    torch.manual_seed(3)
    d = 6
    w = nn.Linear(d, d, bias=True)
    tokens = torch.randn(20, d)
    const = 150.0 * torch.sign(torch.randn(1, d))
    u = torch.randn(d)
    u = u / u.norm()
    groups = {"all": list(w.parameters())}
    lr = 3e-4
    with_const = P.lever(lambda: w(tokens + const).mean(0) @ u, list(w.parameters()), lr, groups)
    without = P.lever(lambda: w(tokens).mean(0) @ u, list(w.parameters()), lr, groups)
    # the lever is lr x sum |gradient|: the weight's part scales with the input's mean size
    g_w = (u[:, None] * (tokens + const).mean(0)[None, :]).abs().sum() + u.abs().sum()
    assert with_const["total"] == pytest.approx(lr * float(g_w), rel=1e-5)
    assert with_const["total"] > 20 * without["total"]
    assert with_const["all"] == pytest.approx(with_const["total"])
    assert all(q.grad is None for q in w.parameters())


def lever_rows(P, steps, ratio=0.1, att=0.9, pulls=(0.2, 0.01, -0.1), per_body=(0.3, -0.3, 0.2, -0.25)):
    rows = []
    for s in steps:
        lessons = {"live": {"draw": {"along_offset": pulls[0]}}, "replay": {"live": {"along_offset": pulls[1]}},
                   "imagination": {"draw": {"along_offset": pulls[2]}}}
        lever = {"A": {"along_offset": {"total": 1.0, "attention": att}},
                 "fold": {"along_offset": {"total": ratio}}}
        rows.append(dict(step=s, lessons=lessons, lever=lever,
                         live_per_body_along_draw_offset={str(i): v for i, v in enumerate(per_body)}))
    return rows


def test_the_summary_applies_pull_noise_and_lever(P):
    steps = (380550, 386750, 389850, 397125, 399225, 410400)
    out = P.summarize(lever_rows(P, steps) + lever_rows(P, (344400,)))
    assert out["references"] == [344400] and 344400 not in out["read"]
    assert out["PULL"]["named"] == ["live"]                   # 0.2 of 0.21 positive at every checkpoint
    assert out["NOISE"]["met"]                                # per-body spread 0.29 against a mean of -0.01
    assert out["LEVER"]["met"] and out["LEVER"]["met_at"] == 6
    # the fold cuts the lever only to 0.5: not met; the attention's share 0.3: not met
    assert not P.summarize(lever_rows(P, steps, ratio=0.5))["LEVER"]["met"]
    assert not P.summarize(lever_rows(P, steps, att=0.3))["LEVER"]["met"]
    # two lessons pushing alike: none carries half at two thirds... both do (0.1 each of 0.2): named both
    both = P.summarize(lever_rows(P, steps, pulls=(0.1, 0.1, -0.1)))
    assert both["PULL"]["named"] == ["live", "replay"]
    none = P.summarize(lever_rows(P, steps, pulls=(-0.1, -0.2, -0.1)))
    assert none["PULL"]["named"] == ["no lesson pushes it"]
    steady = P.summarize(lever_rows(P, steps, per_body=(0.3, 0.31, 0.29, 0.3)))
    assert not steady["NOISE"]["met"]


def test_the_pull_groups_are_the_head_the_rows_and_the_copy(P, T):
    M = P.drivers_module().ablation_module()
    model = T.fake_model()
    g = P.pull_groups(M, model)
    assert set(g) == {"attention", "mlp", "copy"}
    ids = {id(q) for v in g.values() for q in v}
    assert not any(id(q) in ids for q in model.persistent_memory.parameters())    # the readout is left out
    assert all(id(q) in ids for q in model.efference.vision_out.parameters())
