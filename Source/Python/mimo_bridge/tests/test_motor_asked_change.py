# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tests of the asked-change path beside the motor encoder (--motor-asked-change; lem/model/motor_asked_change.py;
docs/gain_of_function_v4/agents/motor_encoder_asked_change_2026-10-03.md).

Two groups. THE PATH ALONE (no learner): its features in their declared units, the handles that ask nothing, one
handle's command reaching that handle's block only, the zero start, the output's unit, the first update's bound, the
window's gauges, its own random stream, the refusals. THE LEARNER (the small H84-shaped learner of
motor_asked_change_fixture.py, CPU, synthetic frames; one shared run, `learner_run`): off builds nothing; switching
on at a checkpoint trained without the path keeps the tokens, the predictions, the loss and the other weights'
update; a checkpoint loads both ways and is refused where it must be; every call site of the motor encoder presents
the same command form; the planner's descent leaves the path frozen; which lessons reach the path; the liveness
monitor, repair and hourly watcher cover its hidden layer; snapshots carry it.

Planted defects: docs/gain_of_function_v4/motor_encoder_asked_change/planted_defects.py runs these tests on copies of
the code with one defect each."""
import copy
import gc
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lem.model import motor_asked_change as MAC
import motor_asked_change_fixture as FX

torch.set_num_threads(1)          # beside production training: one thread

H, GAZE, D = 26, 23, 32
BODY = [i for i in range(H) if i != GAZE]


# ------------------------------------------------------------------------------------------------ helpers
def recorded_size_rows(n=6, seed=0):
    """Motor rows [n, 26, 19] at the sizes measured on H84's recorded rows: positions 33 cm root mean square, an
    asked position change of 12 cm per axis, random orientations, strength 0.2 to 0.8."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, H, MAC.ROW_F)
    x[..., 0:3] = torch.randn(n, H, 3, generator=g) * (33.0 / 200.0)
    q = torch.randn(n, H, 4, generator=g)
    x[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    x[..., 7], x[..., 8], x[..., 9] = 0.5, 0.5, 1.0
    x[..., 10:13] = x[..., 0:3] + torch.randn(n, H, 3, generator=g) * (12.0 / 200.0)
    q = torch.randn(n, H, 4, generator=g)
    x[..., 13:17] = q / q.norm(dim=-1, keepdim=True)
    x[..., 17] = torch.rand(n, H, generator=g)
    x[..., 18] = 0.2 + 0.6 * torch.rand(n, H, generator=g)
    return x


def small_encoder(seed=0):
    from lem.model.encoders.cross_patch import CrossPatchAttentionEncoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        enc = CrossPatchAttentionEncoder(per_patch_input_dim=19, per_patch_hidden=32, d_internal=D, d_output=D,
                                         n_cross_attn_layers=1, n_heads=4, pool_n_heads=4, dropout=0.0, bias=False,
                                         norm_type="layernorm", patch_pos_embedding=True, n_patches=H)
    return enc.eval()


def new_path(seed=0, random_output=False):
    p = MAC.build(H, GAZE, D, seed)
    if random_output:
        g = torch.Generator().manual_seed(seed + 100)
        with torch.no_grad():
            p.up.weight.copy_(torch.randn(p.up.weight.shape, generator=g))
            p.up.bias.copy_(torch.randn(p.up.bias.shape, generator=g))
    return p


class StubEncoder(nn.Module):
    """An encoder whose output is a given tensor per call (its first layer reads 19 numbers, as attach() checks)."""

    def __init__(self, out):
        super().__init__()
        self.patch_mlp = nn.Sequential(nn.Linear(MAC.ROW_F, 4))
        self.out = out

    def forward(self, x):
        return self.out


def quat_about(axis, degrees):
    half = math.radians(degrees) / 2.0
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    return torch.cat([a * math.sin(half), torch.tensor([math.cos(half)])])


# ------------------------------------------------------------------------------------------------ the path alone
def test_features_are_the_asked_change_in_the_declared_units():
    from lem.model import efference as E
    x = recorded_size_rows()
    f = MAC.features(x, GAZE)
    assert f.shape == (6, H, MAC.F_ASK) and f.dtype == torch.float32
    # the position change: command position minus achieved position, cm, in units of 12 cm
    want = ((x[..., 10:13] - x[..., 0:3]) * 200.0).clamp(-MAC.POSITION_CLAMP_CM, MAC.POSITION_CLAMP_CM) / 12.0
    assert torch.allclose(f[:, BODY, 0:3], want[:, BODY], atol=1e-5)
    # the same asked change the efference body head, the contact head and the size-matched swap read, in this
    # module's units (lem.model.efference.asked_change divides by 1.8 cm and 11 degrees)
    ask = E.asked_change(x[..., 10:19], x[..., 0:10], GAZE)
    assert torch.allclose(f[:, BODY, 3:6], ask[:, BODY, 3:6] * E.ASK_ROT_DEG / MAC.TURN_UNIT_DEG, atol=1e-4)
    inside = ((x[..., 10:13] - x[..., 0:3]).abs() * 200.0 < MAC.POSITION_CLAMP_CM).all(-1)
    inside[:, GAZE] = False
    assert bool(inside.any())
    assert torch.allclose(f[..., 0:3][inside], (ask[..., 0:3] * E.ASK_POS_CM / MAC.POSITION_UNIT_CM)[inside], atol=1e-4)
    assert torch.equal(f[:, BODY, 6], x[:, BODY, 18]) and torch.equal(f[:, BODY, 7], x[:, BODY, 17])
    assert bool((f[..., 8] == 1).all()) and bool((f[:, BODY, 9] == 0).all()) and bool((f[:, GAZE, 9] == 1).all())
    # the eyes' row, in its own convention: the command's azimuth and elevation, in units of 0.3 rad
    for convention in ("legacy", "physical"):
        fe = MAC.features(x, GAZE, convention)
        az_el = E.gaze_order_features(x[..., 10:19], GAZE, convention)[:, 9:11]
        assert torch.allclose(fe[:, GAZE, 3:5], az_el / MAC.EYE_UNIT_RAD, atol=1e-6)
        assert bool((fe[:, GAZE, 0:3] == 0).all()) and bool((fe[:, GAZE, 5] == 0).all())
    # lived units: 12 cm along x reads 1.0; 20 degrees about z reads (0, 0, 1.0)
    one = torch.zeros(1, H, MAC.ROW_F)
    one[..., 6] = 1.0
    one[..., 16] = 1.0
    one[..., 18] = 0.5
    one[0, 3, 10] = 12.0 / 200.0
    one[0, 5, 13:17] = quat_about([0.0, 0.0, 1.0], 20.0)
    f1 = MAC.features(one, GAZE)
    assert f1[0, 3, 0:3].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert f1[0, 5, 3:6].tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)
    # the recorded sizes give features of root mean square near 1 (the units' control)
    assert 0.8 < float(f[:, BODY, 0:3].pow(2).mean().sqrt()) < 1.2


def test_the_position_change_is_clamped_at_the_reach_clamp():
    x = recorded_size_rows(2)
    x[0, 4, 10:13] = x[0, 4, 0:3] + torch.tensor([1.0, -1.0, 0.1])     # 200 cm, -200 cm, 20 cm away
    f = MAC.features(x, GAZE)
    lim = MAC.POSITION_CLAMP_CM / MAC.POSITION_UNIT_CM
    assert f[0, 4, 0:3].tolist() == pytest.approx([lim, -lim, 20.0 / 12.0], abs=1e-4)


def test_a_handle_that_asks_nothing_reads_zero_and_keeps_the_gradient_finite():
    p = new_path(random_output=True)
    x = recorded_size_rows(5, seed=3)
    live = x.clone()
    live[..., 10:19] = 0.0                       # the live pass: the command is not chosen yet
    assert bool((MAC.features(live, GAZE) == 0).all())
    assert bool((p(live) == 0).all())            # exactly 0, whatever the output layer holds
    mixed = torch.cat([live[:2], x[2:]])
    out = p(mixed)
    assert bool((out[:2] == 0).all()) and bool((out[2:] != 0).any())
    # strength 0 on one body handle: the limb servo skips it, so it asks nothing; the other handles are unchanged
    limp = x.clone()
    limp[1, 7, 18] = 0.0
    f, f0 = MAC.features(limp, GAZE), MAC.features(x, GAZE)
    assert bool((f[1, 7] == 0).all())
    keep = torch.ones(5, H, dtype=torch.bool)
    keep[1, 7] = False
    assert torch.equal(f[keep], f0[keep])
    # a body handle without a pose asks nothing; the eyes' row needs no pose and ignores its strength
    nopose = x.clone()
    nopose[..., 0:10] = 0.0
    nopose[:, GAZE, 18] = 0.0
    f = MAC.features(nopose, GAZE)
    assert bool((f[:, BODY] == 0).all())
    assert bool((f[:, GAZE, 8] == 1).all()) and torch.equal(f[:, GAZE, 3:5], f0[:, GAZE, 3:5])
    # the hold loss's do-nothing row: position 0, identity rotation, strength 0
    nothing = x.clone()
    nothing[..., 10:19] = 0.0
    nothing[..., 16] = 1.0
    f = MAC.features(nothing, GAZE)
    assert bool((f[:, BODY] == 0).all())
    assert f[0, GAZE].tolist() == pytest.approx([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], abs=1e-7)
    # gradients stay finite through every kind of row (atan2 has none at 0, 0)
    for rows in (live, limp, nopose, nothing, mixed):
        r = rows.clone().requires_grad_(True)
        p(r).pow(2).sum().backward()
        assert bool(torch.isfinite(r.grad).all())


def test_one_handles_command_reaches_that_handles_block_only():
    p = new_path()
    x = recorded_size_rows(3, seed=5)
    z0 = p.handle_features(x)
    assert z0.shape == (3, H, MAC.F_ASK + MAC.RANK)
    changes = {"position": lambda r, i: r[:, i, 10:13].add_(0.05),
               "orientation": lambda r, i: r[:, i, 13:17].copy_(F.normalize(r[:, i, 13:17] + 0.3, dim=-1)),
               "strength": lambda r, i: r[:, i, 18].mul_(0.5),
               "radius": lambda r, i: r[:, i, 17].add_(0.1)}
    for i in range(H):
        for name, change in changes.items():
            if i == GAZE and name == "position":
                continue                               # the eyes' row has no position change
            r = x.clone()
            change(r, i)
            dz = (p.handle_features(r) - z0).abs().amax(dim=(0, 2))       # [H]
            others = torch.cat([dz[:i], dz[i + 1:]])
            assert float(dz[i]) > 1e-4, (i, name)
            assert float(others.max()) == 0.0, (i, name)


def test_every_handle_has_its_own_columns_of_the_output_layer():
    p = new_path(random_output=True)
    width = MAC.F_ASK + MAC.RANK
    assert p.up.in_features == H * width and p.up.out_features == D
    x = recorded_size_rows(4, seed=7)
    r = x.clone().requires_grad_(True)
    p(r).pow(2).sum().backward()
    per_handle = r.grad[..., 10:19].abs().sum(dim=(0, 2))                 # the output depends on every handle
    assert bool((per_handle > 0).all())
    for i in (0, 11, GAZE, 25):
        only = x.clone()
        mask = torch.ones(H, dtype=torch.bool)
        mask[i] = False
        only[:, mask, 10:19] = 0.0                                         # only handle i holds a command
        p.zero_grad(set_to_none=True)
        p(only).pow(2).sum().backward()
        g = p.up.weight.grad.reshape(D, H, width).abs().sum(dim=(0, 2))   # [H]
        assert float(g[i]) > 0 and float(g[mask].max()) == 0.0, i


def test_the_path_starts_at_zero_and_the_token_is_the_same_bits():
    x = recorded_size_rows(8, seed=1)
    enc = small_encoder()
    with torch.no_grad():
        before = enc(x).clone()
    p = new_path()
    assert float(p.up.weight.abs().max()) == 0.0 and float(p.up.bias.abs().max()) == 0.0
    p.attach(enc)
    with torch.no_grad():
        after = enc(x)
        assert torch.equal(after, before)
        assert torch.equal(F.layer_norm(after, (D,)), F.layer_norm(before, (D,)))
    # an encoder output in bf16 (the run's autocast): the sum is formed in float32, so a zero path leaves every
    # value as it is, and the LayerNorm (which computes in float32) gives the same token
    v16 = (before * 3.0).to(torch.bfloat16)
    stub = StubEncoder(v16)
    p2 = new_path()
    p2.attach(stub)
    with torch.no_grad():
        out = stub(x)
    assert out.dtype == torch.float32 and torch.equal(out, v16.float())
    assert torch.equal(F.layer_norm(out, (D,)), F.layer_norm(v16.float(), (D,)))
    # detach() takes the hook away
    with torch.no_grad():
        p.up.weight.fill_(0.1)
        assert not torch.equal(enc(x), before)
        p.detach()
        assert torch.equal(enc(x), before)


def test_one_unit_of_output_is_one_token_unit_whatever_the_pooled_vectors_size():
    x = recorded_size_rows(8, seed=2)
    g = torch.Generator().manual_seed(11)
    v0 = torch.randn(8, 1, D, generator=g, dtype=torch.float64)
    tokens = {}
    for k in (1.0, 9846.0):                       # 9,846: the LayerNorm's divisor measured on H84's motor pool
        v = (v0 * k).clone().requires_grad_(True)
        stub = StubEncoder(v)
        p = new_path(random_output=True).double()
        p.attach(stub)
        out = stub(x)
        tokens[k] = F.layer_norm(out, (D,)).detach()
        u = p(x).detach()
        # LayerNorm(v + s u) = LayerNorm(LayerNorm(v) + u) up to the LayerNorm's eps
        want = F.layer_norm(F.layer_norm(v.detach(), (D,)) + u, (D,))
        assert torch.allclose(tokens[k], want, atol=1e-3 if k == 1.0 else 1e-6)
        # s carries no gradient: the added part's gradient into the encoder's output is 0
        (out - v).pow(2).sum().backward()
        assert float(v.grad.abs().max()) == 0.0
    assert torch.allclose(tokens[1.0], tokens[9846.0], atol=1e-3)
    # the size the output moves the token by: with u of root mean square r the token changes by about r
    v = v0.clone()
    stub = StubEncoder(v)
    p = new_path(random_output=True).double()
    with torch.no_grad():
        r = float(p(x).pow(2).mean().sqrt())
        p.up.weight.mul_(0.1 / r)
        p.up.bias.mul_(0.1 / r)
        base = F.layer_norm(stub(x), (D,))
        p.attach(stub)
        moved = (F.layer_norm(stub(x), (D,)) - base).pow(2).mean().sqrt()
    assert 0.05 < float(moved) < 0.15


def test_a_small_output_survives_a_pooled_vector_of_h84s_size_in_bf16():
    """The pooled vector's numbers of size 4,096 to 16,383 lie on a bf16 grid of 32 to 64. A path output of 0.001
    token units (about 10 at the pooled vector's size) added in bf16 would be rounded away; the sum is formed in
    float32, so it reaches the token."""
    x = recorded_size_rows(8, seed=9)
    g = torch.Generator().manual_seed(12)
    v16 = (torch.randn(8, 1, D, generator=g) * 9846.0).to(torch.bfloat16)
    stub = StubEncoder(v16)
    p = new_path(random_output=True)
    with torch.no_grad():
        r = float(p(x).pow(2).mean().sqrt())
        p.up.weight.mul_(0.001 / r)
        p.up.bias.mul_(0.001 / r)
        u = p(x)
        s = v16.float().std(dim=-1, keepdim=True, unbiased=False)
        exact = v16.float() + s * u                               # the pooled vector plus 0.001 token units
        p.attach(stub)
        out = stub(x)
    assert out.dtype == torch.float32
    assert float((out - exact).abs().max()) <= 0.01               # float32's step at 16,384 is 0.002
    moved = F.layer_norm(out, (D,)) - F.layer_norm(v16.float(), (D,))
    assert 0.0005 < float(moved.pow(2).mean().sqrt()) < 0.002
    # the same sum formed in bf16 would leave many of the pooled vector's numbers where they were; in float32 none
    assert float(((v16 + (s * u).to(torch.bfloat16)) == v16).float().mean()) > 0.3
    assert float(((v16.float() + s * u) == v16.float()).float().mean()) < 0.01


def test_the_first_update_moves_the_output_by_at_most_the_gain_bound():
    lr = 3e-4                                    # H84's learning rate
    p = new_path()
    assert p.gain == pytest.approx(1.0 / 26.0)
    x = recorded_size_rows(8, seed=4)
    z = p.handle_features(x).detach()
    opt = torch.optim.Adam(p.parameters(), lr=lr)
    g = torch.Generator().manual_seed(3)
    target = torch.randn(8, 1, D, generator=g)
    (p(x) * target).sum().backward()
    opt.step()
    with torch.no_grad():
        moved = p(x).abs().amax(dim=(1, 2))                                   # [8] per row
    bound = lr * p.gain * (z.abs().sum(dim=(1, 2)) + 1.0)                     # every weight moves by lr at most
    assert bool((moved <= bound * 1.01).all())
    assert float(moved.max()) > 0.0
    # these rows hold random orientations (turns up to 180 degrees), so the sum of |z| is about 530 against the 238
    # measured on H84's recorded rows; the bound is under 1 hundredth of a token unit per update here and 0.0027 there
    assert float(bound.max()) < 0.01
    assert lr * p.gain * (238.0 + 1.0) == pytest.approx(0.00276, abs=1e-5)
    # without the gain the same update could move the output 26 times as far
    assert float(bound.max() / p.gain) > 0.1


def test_the_windows_gauges_read_the_armed_call_and_add_no_forward_pass():
    p = new_path(random_output=True)
    x = recorded_size_rows(8, seed=6)
    x[0, :, 10:19] = 0.0                         # one row without a command
    calls = []
    hook = p.net[0].register_forward_hook(lambda m, i, o: calls.append(1))
    assert p.window_summary()["rows"] == 0
    p(x)                                         # not armed: not kept
    p.arm_gauge()
    p(x)
    p(x)                                         # the arm covers one call
    assert len(p.gauge_calls) == 1 and len(calls) == 3
    s = p.window_summary()
    hook.remove()
    assert len(calls) == 3                       # the summary itself ran no layer
    assert s["rows"] == 7 and s["version"] == MAC.ASKED_CHANGE_VERSION
    assert s["output_rms"] > 0 and s["output_weight_rms"] > 0
    assert 0.8 < s["features_rms"]["position"] < 1.2
    assert s["handle_reach"]["spread"] > 0.9 and s["handle_reach"]["handles_under_1_in_1000"] == 0
    assert 0.0 <= s["liveness"]["net.0"] <= 1.0 and "net.0" in s["liveness_varying"]
    assert "encoder_output_sd" not in s          # no encoder call passed through a hook
    again = p.window_summary()                   # a fresh window
    assert again["rows"] == 0 and "output_rms" not in again
    # one handle holding all the output: the reach gauge says so
    with torch.no_grad():
        w = p.up.weight.reshape(D, H, -1)
        w[:, 1:, :] = 0.0
    p.arm_gauge()
    p(x)
    s = p.window_summary()
    assert s["handle_reach"]["largest_share"] == pytest.approx(1.0) and s["handle_reach"]["spread"] < 0.01
    assert s["handle_reach"]["handles_under_1_in_1000"] == H - 1


def test_inside_frozen_the_weights_get_no_gradient_and_the_row_keeps_its_own():
    p = new_path(random_output=True)
    x = recorded_size_rows(4, seed=8)
    with torch.no_grad():
        plain = p(x)
    r = x.clone().requires_grad_(True)
    with p.frozen():
        assert p.frozen_depth == 1
        with p.frozen():
            assert p.frozen_depth == 2
        out = p(r)
    assert p.frozen_depth == 0
    assert torch.equal(out.detach(), plain)                       # the same numbers
    out.pow(2).sum().backward()
    assert all(q.grad is None for q in p.parameters())            # no lesson reaches the weights
    assert float(r.grad[..., 10:19].abs().sum()) > 0              # the command keeps its gradient
    p(x).pow(2).sum().backward()                                  # outside: the weights learn
    assert all(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in p.parameters())
    with pytest.raises(RuntimeError, match="boom"):
        with p.frozen():
            raise RuntimeError("boom")
    assert p.frozen_depth == 0
    # the encoder names its path while attached, without a new entry in its state dict
    from lem.model.encoder_registry import imagined_motor_token
    enc = small_encoder()
    keys = set(enc.state_dict())

    class Registry:
        encoders = {"motor": enc}

        @staticmethod
        def encode(frame, mask):
            return {"motor": F.layer_norm(enc(frame["motor"]), (D,))}
    without = imagined_motor_token(Registry, x)
    p.zero_grad(set_to_none=True)
    p.attach(enc)
    assert enc.__dict__["asked_change_path"] is p and set(enc.state_dict()) == keys
    token = imagined_motor_token(Registry, x)
    assert not torch.equal(token, without)                        # the path's output is in the imagined token
    token.pow(2).sum().backward()
    assert all(q.grad is None for q in p.parameters()) and p.frozen_depth == 0
    p.detach()
    assert "asked_change_path" not in enc.__dict__


def test_the_path_is_built_in_its_own_random_stream():
    torch.manual_seed(77)
    state = torch.get_rng_state()
    a = MAC.build(H, GAZE, D, seed=3)
    assert torch.equal(torch.get_rng_state(), state)
    b, c = MAC.build(H, GAZE, D, seed=3), MAC.build(H, GAZE, D, seed=4)
    assert torch.equal(a.net[0].weight, b.net[0].weight)
    assert not torch.equal(a.net[0].weight, c.net[0].weight)
    assert a.stamp()["version"] == MAC.ASKED_CHANGE_VERSION and a.stamp()["output_gain"] == pytest.approx(1 / 26)


def test_refusals_of_the_path():
    p = new_path()
    with pytest.raises(ValueError, match="19 numbers"):
        legacy = nn.Module()
        legacy.patch_mlp = nn.Sequential(nn.Linear(10, 4))
        p.attach(legacy)
    enc = small_encoder()
    p.attach(enc)
    with pytest.raises(RuntimeError, match="attached already"):
        p.attach(enc)
    with pytest.raises(ValueError, match="19"):
        MAC.features(torch.zeros(2, H, 10), GAZE)
    with pytest.raises(ValueError, match="26 handles"):
        p.handle_features(torch.zeros(2, 24, MAC.ROW_F))


def test_the_switch_check_reads_zero_at_the_start_and_sees_an_output_layer_that_is_not_zero():
    enc = small_encoder()

    class Registry:
        encoders = {"motor": enc}

        @staticmethod
        def encode(frame, mask):
            return {"motor": F.layer_norm(enc(frame["motor"]), (D,))}

    class Model:
        encoder_registry = Registry()
        use_amp = False
    model = Model()
    with pytest.raises(ValueError, match="without --motor-asked-change on"):
        MAC.switch_check(model)
    model.motor_asked_change = new_path()
    model.motor_asked_change.attach(enc)
    check = MAC.switch_check(model)
    assert check["exact"] and check["float32"]["change_max_abs"] == 0.0 and check["rows"] == 64
    assert model.motor_asked_change.unit_n == 0          # the probe rows join no gauge
    with torch.no_grad():
        model.motor_asked_change.up.weight[0, 0] = 0.5
    check = MAC.switch_check(model)
    assert not check["exact"] and check["float32"]["change_max_abs"] > 0.0
    assert not model.motor_asked_change.bypass


# ------------------------------------------------------------------------------------------------ the learner
def _built(tmp, name, flag, ckpt=None):
    """A learner built as the trainer builds it, with its checkpoint loaded as the trainer loads it (then the
    context and the replay ring start, train_mimo.main's sequence). flag None = the launch line without the flag."""
    from mimo_bridge import train_mimo as T
    extra = [] if flag is None else ["--motor-asked-change", flag]
    args = T.parse_args(FX.argv(out=str(tmp / name), extra=extra))
    args.n_envs = int(args.n_envs)
    model = T.build_learner(args, str(tmp / name / "checkpoints"))
    loaded = None if ckpt is None else model.load_checkpoint(str(ckpt))
    model.reset_context(args.n_envs)
    model.enable_replay(args.n_envs, size_per_env=64)
    model.replay_batch = 4
    return model, args, loaded


def _seed(model, seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.noise_rng.manual_seed(seed)


def _facts(model, args):
    from mimo_bridge import train_mimo as T
    lv = model.liveness
    return dict(has_path=getattr(model, "motor_asked_change", None) is not None,
                has_attribute=hasattr(model, "motor_asked_change"),
                receipt="motor_asked_change" in model.receipt,
                receipt_params=model.receipt["component_params"].get("motor_asked_change"),
                side="motor_asked_change" in model._side_modules(),
                liveness_layers=list(lv.layer_names), liveness_rows=sorted(lv.rows.keys()),
                n_params=int(model.n_params),
                acting=T.acting_configuration(args).get("motor_asked_change", "absent"),
                frozen_in_planning=any(m is getattr(model, "motor_asked_change", None)
                                       for m in model.intention_gen._wellness_freeze),
                optimizer_has_path=(None if getattr(model, "motor_asked_change", None) is None else
                                    all(any(p is q for g in model.optimizer.param_groups for q in g["params"])
                                        for p in model.motor_asked_change.parameters())))


class LessonProbe:
    """Per lesson term: the summed absolute gradient that reaches the path's output layer (LemLite.lesson_probe)."""

    def __init__(self, params):
        self.terms, self.params = {}, params
        self.reach, self.seen = {}, {}

    def before_backward(self, model, loss):
        for k, v in self.terms.items():
            self.seen[k] = self.seen.get(k, 0) + 1
            if not torch.is_tensor(v) or not v.requires_grad:
                continue
            g = torch.autograd.grad(v, self.params, retain_graph=True, allow_unused=True)
            self.reach[k] = self.reach.get(k, 0.0) + sum(float(x.abs().sum()) for x in g if x is not None)

    def after_clip(self, model, grad_norm):
        pass


@pytest.fixture(scope="module")
def learner_run(tmp_path_factory):
    """One shared run (about 100 model steps in all): a learner trained without the path for 12 model steps and
    saved; that checkpoint resumed without the path and with it (the switch), 6 model steps each on the same frames
    and random streams; the switched learner trained on to model step 72 with every call of the motor encoder
    recorded, saved, and read with a lesson probe for 12 more; its checkpoint loaded into a build with the path and
    refused by a build without it."""
    import lem.model.encoder_registry as ER
    from mimo_bridge import copied_execution as CE
    from mimo_bridge import continuation_state as CS
    tmp = tmp_path_factory.mktemp("asked_change")
    R = {}
    # ---- trained without the path, saved
    off0, args_off, _ = _built(tmp, "off0", None)
    R["off"] = _facts(off0, args_off)
    world = FX.SyntheticWorld(args_off.n_envs, seed=0)
    rows = FX.drive(off0, world, 12, replay_every=2)
    off0.step_count = 12
    ckpt0 = off0.save_checkpoint(12)
    carry = rows[-1]["carry"]
    R["ckpt0"] = ckpt0
    del off0
    gc.collect()

    def continued(model, n, start, state, **kw):
        w, c = copy.deepcopy((world, state))
        _seed(model)
        return FX.drive(model, w, n, start=start, state=c, digest_every=1, replay_every=2,
                        digest_skip=("motor_asked_change",), **kw), w

    # ---- the checkpoint resumed without the path
    off1, _a, loaded = _built(tmp, "off1", None, ckpt0)
    R["rows_off"], _w = continued(off1, 6, 12, carry)
    del off1
    gc.collect()
    # ---- the checkpoint resumed with the path: the switch
    on1, args_on, loaded = _built(tmp, "on1", "on", ckpt0)
    R["on"] = _facts(on1, args_on)
    R["switch"] = copy.deepcopy(on1.motor_asked_change_switch)
    R["switch_stamp"] = copy.deepcopy(on1.receipt["motor_asked_change"])
    R["loaded_step"] = loaded["step"]
    calls = FX.EncoderCalls(on1)
    executed, weight_max = {}, {}
    normalised = set()
    original = ER.normalize_executed_command

    def spy(cmd, position_range_cm=ER.COMMAND_POSITION_RANGE_CM):
        out = original(cmd, position_range_cm)
        o = out.detach().float().cpu()
        if o.dim() >= 3 and o.shape[-1] == 9:
            o = o.reshape(-1, o.shape[-2], 9)               # replay normalises [batch, model steps, handles, 9]
            for b in range(o.shape[0]):
                normalised.add(o[b].numpy().tobytes())
        return out

    def before(step, prev_exec):
        calls.tag = step
        executed[step] = None if prev_exec is None else np.array(prev_exec, np.float32, copy=True)
        weight_max[step] = float(on1.motor_asked_change.up.weight.abs().max())
    ER.normalize_executed_command = spy
    try:
        R["rows_on"], w_on = continued(on1, 6, 12, carry, before_step=before)
        more = FX.drive(on1, w_on, 54, start=18, state=R["rows_on"][-1]["carry"], replay_every=2, before_step=before)
    finally:
        ER.normalize_executed_command = original
    calls.remove()
    R.update(calls=calls.calls, executed=executed, weight_max=weight_max, normalised=normalised,
             gaze_idx=on1.GAZE_IDX, convention=on1.eye_row_convention)
    # ---- snapshots (taken while the window's gauges hold the lesson's calls, as a probe's snapshot would find them)
    R["gauge_calls_at_capture"] = len(on1.motor_asked_change.gauge_calls)
    snap = CE.capture(on1)
    R["capture_has_module"] = "motor_asked_change" in snap["modules"]
    R["companion_names"] = sorted(CS.capture_learner(on1)["data"].keys())
    saved_w = on1.motor_asked_change.up.weight.detach().clone()
    with torch.no_grad():
        on1.motor_asked_change.up.weight.add_(1.0)
    CE.restore(on1, snap)
    R["restore_puts_no_weight_back"] = bool(torch.equal(on1.motor_asked_change.up.weight, saved_w + 1.0))
    with torch.no_grad():
        on1.motor_asked_change.up.weight.copy_(saved_w)
    R["summary"] = on1.motor_asked_change.window_summary()
    R["liveness_summary"] = on1.liveness.window_summary(on1)
    # ---- saved with the path
    on1.step_count = 72
    ckpt1 = on1.save_checkpoint(72)
    R["ckpt1"] = ckpt1
    R["path_state"] = {k: v.detach().clone() for k, v in on1.motor_asked_change.state_dict().items()}
    R["path_adam"] = {n: {k: v.detach().clone() for k, v in on1.optimizer.state[p].items() if torch.is_tensor(v)}
                      for n, p in on1.motor_asked_change.named_parameters() if p in on1.optimizer.state}
    # ---- which lessons reach the path
    probe = LessonProbe([on1.motor_asked_change.up.weight])
    on1.lesson_probe = probe
    on1.loss_manager.term_probe = probe.terms
    FX.drive(on1, w_on, 12, start=72, state=more[-1]["carry"], replay_every=0,
             on_step=lambda step, model, res: probe.terms.clear())
    R["reach"], R["seen"] = dict(probe.reach), dict(probe.seen)
    del on1
    gc.collect()
    # ---- loaded into a build with the path
    on2, _a, loaded = _built(tmp, "on2", "on", ckpt1)
    R["on2_state"] = {k: v.detach().clone() for k, v in on2.motor_asked_change.state_dict().items()}
    R["on2_adam"] = {n: {k: v.detach().clone() for k, v in on2.optimizer.state[p].items() if torch.is_tensor(v)}
                     for n, p in on2.motor_asked_change.named_parameters() if p in on2.optimizer.state}
    R["on2_switch"] = on2.motor_asked_change_switch
    R["on2_stamp"] = copy.deepcopy(on2.receipt["motor_asked_change"])
    R["on2_step"] = loaded["step"]
    del on2
    gc.collect()
    R["tmp"] = tmp
    return R


def test_off_builds_nothing_and_on_builds_the_path(learner_run):
    off, on = learner_run["off"], learner_run["on"]
    assert off["has_attribute"] is False and off["receipt"] is False and off["side"] is False
    assert MAC.LIVENESS_LAYER not in off["liveness_layers"]
    assert off["acting"] == "absent"                 # the acting fingerprint of a launch without the path is unchanged
    assert on["has_path"] and on["receipt"] and on["side"] and on["acting"] == "on"
    assert on["optimizer_has_path"] and on["frozen_in_planning"]
    assert on["liveness_layers"] == off["liveness_layers"] + [MAC.LIVENESS_LAYER]
    path_params = (MAC.F_ASK * MAC.HIDDEN + MAC.HIDDEN) + (MAC.HIDDEN * MAC.RANK + MAC.RANK) \
        + (H * (MAC.F_ASK + MAC.RANK) * D + D)
    assert on["receipt_params"] == path_params
    new_rows = [k for k in on["liveness_rows"] if k not in off["liveness_rows"]]
    assert new_rows == [MAC.LIVENESS_LAYER.replace(".", "__")]
    assert on["n_params"] > off["n_params"] + path_params - 1


def test_switching_on_at_a_checkpoint_keeps_tokens_predictions_loss_and_the_other_weights_update(learner_run):
    R = learner_run
    sw = R["switch"]
    assert sw is not None and sw["exact"] and sw["float32"]["bit_identical"] and sw["float32"]["change_max_abs"] == 0.0
    assert sw["at_step"] == 12 and sw["version"] == MAC.ASKED_CHANGE_VERSION and R["loaded_step"] == 12
    assert R["switch_stamp"]["switch"]["at_step"] == 12
    off, on = R["rows_off"], R["rows_on"]
    same = ("loss", "grad", "command", "gaze", "breakdown", "wellness", "replay", "torch_rng", "numpy_rng",
            "noise_rng", "tokens", "predictions", "state")
    for i in (0, 1):                                 # model steps 12 and 13: the output layer is still zero
        assert R["weight_max"][12 + i] == 0.0
        for k in same:
            assert on[i][k] == off[i][k], (12 + i, k)
    # model step 13 is not trivially equal: its delayed lesson encoded a row that holds a command, through the path
    commanded = [c for c in R["calls"] if c["tag"] == 13 and c["site"] == "step" and c["grad"]
                 and float(c["rows"][..., 13:17].norm(dim=-1).max()) > 0.5]
    assert commanded
    # and the comparison can see the path: once its output layer has moved, training differs
    assert max(R["weight_max"][s] for s in range(14, 18)) > 0.0
    assert any(on[i]["loss"] != off[i]["loss"] for i in range(2, 6))


def test_a_checkpoint_loads_both_ways_and_is_refused_where_it_must_be(learner_run):
    R = learner_run
    tmp = R["tmp"]
    assert R["on2_step"] == 72 and R["on2_switch"] is None        # no second switch
    assert set(R["on2_state"]) == set(R["path_state"])
    for k, v in R["path_state"].items():
        assert torch.equal(R["on2_state"][k], v), k
    assert float(R["path_state"]["up.weight"].abs().max()) > 0.0   # what was saved had learned something
    assert set(R["on2_adam"]) == set(R["path_adam"]) and R["path_adam"]
    for n, st in R["path_adam"].items():
        for k, v in st.items():
            assert torch.equal(R["on2_adam"][n][k], v), (n, k)
    assert R["on2_stamp"]["switch"] == R["switch_stamp"]["switch"]  # the switch stays named in every receipt
    # a build without the path refuses the checkpoint trained with it
    off2, _a, _l = _built(tmp, "off2", None)
    with pytest.raises(ValueError, match="--motor-asked-change on"):
        off2.load_checkpoint(R["ckpt1"])
    del off2
    gc.collect()
    # another version of the path is refused
    other = tmp / "other_version" / os.path.basename(R["ckpt1"])
    shutil.copytree(R["ckpt1"], other)
    receipt = json.loads((other / "model_receipt.json").read_text(encoding="utf-8"))
    receipt["motor_asked_change"]["version"] = "motor_asked_change_v0"
    (other / "model_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    on3, _a, _l = _built(tmp, "on3", "on")
    with pytest.raises(ValueError, match="motor_asked_change_v0"):
        on3.load_checkpoint(str(other))
    # a checkpoint without the stamp whose output layer is not zero is not a switch that keeps the tokens
    with torch.no_grad():
        on3.motor_asked_change.up.weight.fill_(0.01)
    import lem.training.checkpoint as CK
    real = CK.CheckpointManager.load

    def keep_planted(self, *a, **kw):
        out = real(self, *a, **kw)
        with torch.no_grad():
            on3.motor_asked_change.up.weight.fill_(0.01)
        return out
    CK.CheckpointManager.load = keep_planted
    try:
        with pytest.raises(RuntimeError, match="output layer is not zero"):
            on3.load_checkpoint(R["ckpt0"])
    finally:
        CK.CheckpointManager.load = real
    del on3
    gc.collect()


def test_every_call_site_presents_the_same_command_form(learner_run):
    R = learner_run
    gi, conv = R["gaze_idx"], R["convention"]
    calls = R["calls"]
    sites = {c["site"] for c in calls}
    expected = {"step", "replay_update", "_tenure_forward", "_contact_control_forward", "_run_imagination_compiled",
                "generate_action", "_causal_score_candidates", "_plan_chain", "_swap_test_arms"}
    assert expected <= sites, sorted(expected - sites)
    assert "other" not in sites and "transition" not in sites
    executed_sites = ("step", "replay_update", "_tenure_forward", "_contact_control_forward",
                      "_run_imagination_compiled", "_swap_test_arms")
    audits = 0
    arm_rolled, arm_sized = set(), set()
    for c in calls:
        r = c["rows"].reshape(-1, c["rows"].shape[-2], MAC.ROW_F)
        qc, qa = r[..., 13:17].norm(dim=-1), r[..., 3:7].norm(dim=-1)
        # a handle's command quaternion is absent (0) or a unit quaternion, never anything between
        assert bool(((qc < 1e-6) | ((qc - 1).abs() < 1e-3)).all()), c["site"]
        # every site hands the encoder a pose for every handle
        assert bool(((qa - 1).abs() < 1e-3).all()), c["site"]
        has = qc > 0.5
        if not bool(has.any()):
            assert c["site"] == "step"                           # only the live pass encodes a row without a command
            assert bool((r[..., 10:19] == 0).all())
            continue
        assert bool(has.all()), c["site"]                        # a commanded row holds a command for every handle
        # the positions are in the encoder's units (cm / 200): the asked change is a few cm, not thousands
        body = [i for i in range(r.shape[1]) if i != gi]
        asked_cm = (r[:, body, 10:13] - r[:, body, 0:3]) * 200.0
        asking = r[:, body, 18] > 0                                   # the hold loss's do-nothing row asks nothing
        if bool(asking.any()):
            assert float(asked_cm[asking].abs().max()) < 150.0, c["site"]
        else:
            assert c["site"] == "_tenure_forward"
        assert float(r[..., 10:13].abs().max()) < 2.0, c["site"]
        assert float(r[..., 17:19].min()) >= 0.0 and float(r[..., 17:19].max()) <= 1.0, c["site"]
        if c["site"] in executed_sites:
            # the command columns are what normalize_executed_command returned (the one place a command's units
            # change on its way to the encoder), bit for bit
            for b in range(r.shape[0]):
                assert r[b, :, 10:19].numpy().tobytes() in R["normalised"], c["site"]
        if c["site"] == "step":
            ex = R["executed"].get(c["tag"])
            assert ex is not None
            want = torch.as_tensor(ex).clone()
            want[..., 0:3] /= 200.0
            own = torch.allclose(r[..., 10:19], want, atol=1e-6)
            rolled = torch.allclose(r[..., 10:19], want.roll(1, dims=0), atol=1e-6)
            assert own or rolled, c["tag"]
            audits += int(rolled and not own)
        if c["site"] == "_swap_test_arms":
            # the swap test's added arms (no gradient, outside the liveness watching windows): on this run's 60
            # model steps each call holds another body's absolute command (the hold's first model step) or the
            # size-matched command, whose asked change of every body handle is the other body's own asked change;
            # the shifted-own arm needs 200 model steps of command stream and does not appear
            assert c["grad"] is False and c["frozen"] is False
            ex = R["executed"].get(c["tag"])
            assert ex is not None
            want = torch.as_tensor(ex).clone()
            want[..., 0:3] /= 200.0
            if torch.allclose(r[..., 10:19], want.roll(1, dims=0), atol=1e-6):
                arm_rolled.add(c["tag"])
                continue
            own_rows = torch.cat([r[..., :MAC.ACHIEVED_F], want.to(r.dtype)], dim=-1)
            asked_arm = MAC.features(r, gi, conv)[:, body, 0:6]
            asked_other = MAC.features(own_rows, gi, conv).roll(1, dims=0)[:, body, 0:6]
            assert torch.allclose(asked_arm, asked_other, atol=2e-3), (c["tag"], float((asked_arm - asked_other)
                                                                                   .abs().max()))
            assert not torch.allclose(r[..., 10:19], want, atol=1e-6)
            arm_sized.add(c["tag"])
    assert audits >= 2                                           # the swap test's rows at model steps 20, 40, 60
    # the added arms: the size-matched command on both phases (model steps 20, 21, 40, 41, 60, 61), the absolute
    # roll from this site only on the hold's first model step (21, 41, 61)
    assert {20, 21, 40, 41, 60, 61} <= arm_sized and arm_rolled == {21, 41, 61}, (sorted(arm_sized), sorted(arm_rolled))
    # the path's features at every site are in the declared units: the same function of the same row
    lesson = [c for c in calls if c["site"] == "step" and c["grad"]
              and float(c["rows"][..., 13:17].norm(dim=-1).max()) > 0.5]
    f = MAC.features(lesson[-1]["rows"], gi, conv)
    ex = torch.as_tensor(R["executed"][lesson[-1]["tag"]])
    body = [i for i in range(f.shape[1]) if i != gi]
    want = ((ex[:, body, 0:3] - lesson[-1]["rows"][:, body, 0:3] * 200.0).clamp(-40, 40) / MAC.POSITION_UNIT_CM)
    assert torch.allclose(f[:, body, 0:3], want, atol=1e-4)


def test_the_planners_descent_leaves_the_path_frozen_and_training_calls_do_not(learner_run):
    calls = learner_run["calls"]
    planning = [c for c in calls if c["site"] == "generate_action"]
    assert planning and all(c["path_trains"] is False for c in planning)
    assert all(c["grad"] for c in planning)                      # the descent's gradient passes through the path
    training = [c for c in calls if c["site"] in ("step", "replay_update", "_run_imagination_compiled")]
    assert all(c["path_trains"] for c in training)
    # every imagined row passes through the path with its weights detached; every lived row with them attached
    imagined = ("_run_imagination_compiled", "generate_action", "_causal_score_candidates", "_plan_chain")
    lived = ("step", "replay_update", "_tenure_forward", "_contact_control_forward", "_swap_test_arms")
    for c in calls:
        assert c["frozen"] is (c["site"] in imagined), c["site"]
        assert c["site"] in imagined + lived, c["site"]


def test_which_lessons_reach_the_path(learner_run):
    reach, seen = learner_run["reach"], learner_run["seen"]
    reaching = ("next_frame.proprioception", "next_frame.tactile", "next_frame.vision", "next_frame.audio",
                "next_frame.vestibular", "tenure")
    for k in reaching:
        assert seen.get(k, 0) > 0 and reach.get(k, 0.0) > 0.0, k
    # the imagination chain's rows are imagined (a pose 1 and 2 model steps old at depths 2 and 3): its loss reaches
    # the path's weights nowhere; neither do the lessons on the live pass's token (whose row holds no command) nor
    # the planner's own
    for k in ("imagination", "vicreg", "motor", "wellness", "prediction.proprioception", "prediction.vision",
              "tenure_tok"):
        assert seen.get(k, 0) > 0 and reach.get(k, 0.0) == 0.0, k


def test_the_stats_blocks_report_the_path(learner_run):
    s = learner_run["summary"]
    assert s["version"] == MAC.ASKED_CHANGE_VERSION and s["rows"] > 0 and s["encoder_calls"] > 0
    assert s["encoder_output_sd"] > 0 and s["output_weight_rms"] > 0 and s["output_rms"] > 0
    assert s["handle_reach"]["handles_under_1_in_1000"] == 0
    assert set(s["features_rms"]) == {"position", "turn", "strength", "radius", "eye"}
    lv = learner_run["liveness_summary"]
    assert MAC.LIVENESS_LAYER in lv["layers"] and MAC.LIVENESS_LAYER in lv["watching"]
    assert s["liveness"]["net.0"] == 1.0 and lv["layers"][MAC.LIVENESS_LAYER] == 1.0


def test_snapshots_carry_the_path_as_a_module_and_never_as_episode_state(learner_run):
    R = learner_run
    assert R["capture_has_module"] and R["gauge_calls_at_capture"] > 0
    # the episode companion carries no attribute of the path: its weights live in the checkpoint
    assert not [n for n in R["companion_names"] if "asked_change" in n]
    # an acting snapshot's restore leaves learned weights alone, as for every other module
    assert R["restore_puts_no_weight_back"]


def _hourly_watch():
    path = Path(__file__).resolve().parents[4] / "docs" / "h84_launch_2026-09-18" / "hourly_watch.py"
    spec = importlib.util.spec_from_file_location("hourly_watch_for_asked_change", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fresh_build_marks(model):
    """Of a learner just built: the random stream's state, and one hash over every weight the build with the path
    shares with a build without it (the liveness repair's rows of the path's layer and the path itself left out)."""
    from lem.model.efference import _named_components
    shared = []
    for comp, mod in sorted(_named_components(model).items()):
        if comp == "motor_asked_change":
            continue
        for n, p in sorted(mod.named_parameters()):
            if "motor_asked_change" not in n:
                shared.append(p.detach().float().reshape(-1))
    return FX.sha(torch.get_rng_state()), FX.sha(torch.cat(shared))


def test_the_liveness_repair_revives_the_paths_hidden_layer_and_the_watcher_reads_it(tmp_path):
    off, _a, _l = _built(tmp_path, "lv_off", None)
    marks_off = _fresh_build_marks(off)
    del off
    gc.collect()
    model, args, _l = _built(tmp_path, "lv", "on")
    # a fresh build with the path starts from the same weights everywhere else (the liveness repair's other rows
    # and its pool paths included) and leaves the run's random stream where a build without it leaves it
    assert _fresh_build_marks(model) == marks_off
    assert MAC.LIVENESS_LAYER in model.liveness.layer_names
    lin = model.motor_asked_change.net[0]
    with torch.no_grad():
        lin.bias[::2] = -30.0                     # every second unit far under the liveness line of -3
    world = FX.SyntheticWorld(args.n_envs, seed=2)
    # first no watching window can fill, so the monitor and the path's own gauge read the dead units
    model.liveness.revive_min_samples = 10 ** 9
    rows = FX.drive(model, world, 10, digest_every=0)
    monitor, own = model.liveness.window_summary(model), model.motor_asked_change.window_summary()
    assert monitor["layers"][MAC.LIVENESS_LAYER] == 0.5 and own["liveness"]["net.0"] == 0.5
    assert MAC.LIVENESS_LAYER in monitor["below_line"] or monitor["layers"][MAC.LIVENESS_LAYER] >= monitor["layer_line"]
    assert MAC.LIVENESS_LAYER not in FX.revivals(model)
    # then the repair's window fills (60 rows) and it revives the 32 dead units
    model.liveness.revive_min_samples = 60
    FX.drive(model, world, 20, start=10, state=rows[-1]["carry"], digest_every=0)
    monitor, own = model.liveness.window_summary(model), model.motor_asked_change.window_summary()
    assert MAC.LIVENESS_LAYER in FX.revivals(model)
    assert monitor["revived_units"][MAC.LIVENESS_LAYER] == MAC.HIDDEN // 2
    assert monitor["layers"][MAC.LIVENESS_LAYER] == 1.0 and own["liveness"]["net.0"] == 1.0
    # the hourly watcher's stop rule reads the layer from the liveness block of two complete reports
    hw = _hourly_watch()
    low = [dict(step=s, liveness=dict(mode="on", samples=8, window_steps=100,
                                      layers={MAC.LIVENESS_LAYER: 0.3, "forward_model.blocks.0.ffn.0": 1.0}))
           for s in (100, 200)]
    message = hw.liveness_breach(low)
    assert message and MAC.LIVENESS_LAYER in message
    fine = [dict(r, liveness=dict(r["liveness"], layers={MAC.LIVENESS_LAYER: 1.0})) for r in low]
    assert hw.liveness_breach(fine) is None


def test_the_trainers_flag():
    from mimo_bridge import train_mimo as T
    from mimo_bridge.tools import build_explainer as BE
    base = ["--out", "x", "--device", "cpu"]
    assert T.parse_args(base).motor_asked_change == "off"
    assert T.parse_args(base + ["--motor-asked-change", "on"]).motor_asked_change == "on"
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--motor-asked-change", "maybe"])
    assert "motor_asked_change" in BE.PUBLISHED_ARGS
    # an args.json written before the flag builds the model without the path
    old = T.parse_args(base)
    del old.motor_asked_change
    assert "motor_asked_change" not in T.acting_configuration(old)
