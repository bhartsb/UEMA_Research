# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--vestibular-input-path (lem/model/vestibular_input_path.py): the linear path beside the vestibular encoder.

The module's own arithmetic on a stand-in encoder of H84's shape (28 inputs, three hidden layers of 192, the last
layer's bias of spread 18), and the learner-level contract on the small H84-shaped learner of
motor_asked_change_fixture.py (CPU, synthetic frames; one shared run, `learner_run`): off builds nothing; switching it
on at a checkpoint keeps the first model step's tokens, predictions, loss and command; the path learns from the live
row and gets no gradient from the delayed lesson's re-encoding; a checkpoint with the path loads into a build with it
and is refused by a build without it.

Record: docs/gain_of_function_v4/agents/vestibular_token_2026-10-04.md."""
import copy
import gc
import json
import os
import shutil

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lem.model import vestibular_input_path as VIP
import motor_asked_change_fixture as FX

N_IN, D = 28, 192
EPS = 1e-5


def _encoder(seed=0, bias_spread=18.0):
    """A stand-in for H84's vestibular encoder: the trainer's own class at H84's shape, its last layer's bias a
    constant pattern of spread `bias_spread` (H84: 18.09 at checkpoint 479,350) and a small weight."""
    from lem.model.encoders.mlp_project import MLPProjectEncoder
    torch.manual_seed(seed)
    enc = MLPProjectEncoder(N_IN, [D, D, D], D, "gelu", 0.0, True, "layernorm")
    with torch.no_grad():
        enc.mlp[-1].bias.copy_(torch.randn(D) * bias_spread)
        enc.mlp[-1].weight.mul_(0.5)
    return enc


def _rows(n=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, N_IN, generator=g).clamp(-4, 4)


def _token(enc, x):
    v = enc(x)
    return F.layer_norm(v.float(), (D,), None, None, EPS).squeeze(1)


# ------------------------------------------------------------------------------------------------ the module
def test_the_path_starts_at_zero_and_the_token_is_the_same_bits():
    enc = _encoder()
    x = _rows()
    before = _token(enc, x)
    path = VIP.build(N_IN, D)
    assert float(path.up.weight.abs().max()) == 0.0 and path.up.bias is None
    path.attach(enc)
    after = _token(enc, x)
    assert torch.equal(before, after)
    # under bf16 autocast too (the card's precision for the encoder's Linears; the hook's sum is float32)
    path.bypass = True
    with torch.autocast("cpu", dtype=torch.bfloat16):
        off = _token(enc, x)
    path.bypass = False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        on = _token(enc, x)
    assert torch.equal(off, on)


def test_one_unit_of_output_is_one_token_unit_whatever_the_last_layers_size():
    for spread in (1.0, 18.0, 74425.0):
        enc = _encoder(bias_spread=spread)
        x = _rows()
        with torch.no_grad():
            v = enc(x).squeeze(1)
        path = VIP.build(N_IN, D)
        with torch.no_grad():
            path.up.weight.copy_(torch.randn(D, N_IN) * 0.3)
        path.attach(enc)
        with torch.no_grad():
            got = _token(enc, x)
            u = path(x)
            want = F.layer_norm(F.layer_norm(v, (D,), None, None, 0.0) + u, (D,), None, None, 0.0)
        # LayerNorm(v + s u) = LayerNorm(LayerNorm(v) + u) up to the LayerNorm's eps
        assert float((got - want).abs().max()) < 2e-3 * max(1.0, 18.0 / spread), spread


def test_the_path_is_linear_a_difference_between_bodies_and_a_change_over_time_pass_alike():
    path = VIP.build(N_IN, D)
    with torch.no_grad():
        # small enough that every row below stays under the output's cap: under the cap the path is its linear map
        path.up.weight.copy_(torch.randn(D, N_IN) * 0.05)
    a, b = _rows(1, seed=2), _rows(1, seed=3)
    d = torch.zeros(1, N_IN)
    d[0, 1] = 0.7                                     # one field (the head accelerometer's y) changes by 0.7 spreads
    with torch.no_grad():
        for row in (a, b, a + d, b + d, d):
            assert float(path(row).pow(2).mean(-1).sqrt().max()) < path.cap
        # the same change of one input moves u by the same amount on any row: under the cap there is no saturation
        assert torch.allclose(path(a + d) - path(a), path(b + d) - path(b), atol=1e-6)
        assert torch.allclose(path(a + d) - path(a), path(d), atol=1e-6)
        # and by the field's own column x the gain
        assert torch.allclose(path(d)[0], path.up.weight[:, 1] * 0.7 * path.gain, atol=1e-6)
    assert path.gain == pytest.approx(1.0 / (N_IN ** 0.5))


def test_inside_frozen_the_weight_gets_no_gradient_and_the_row_keeps_its_own():
    enc = _encoder()
    path = VIP.build(N_IN, D)
    with torch.no_grad():
        path.up.weight.copy_(torch.randn(D, N_IN) * 0.1)
    path.attach(enc)
    x = _rows().requires_grad_(True)
    open_tok = _token(enc, x)
    with path.frozen():
        frozen_tok = _token(enc, x)
        with path.frozen():                           # nests
            assert path.frozen_depth == 2
    assert path.frozen_depth == 0
    assert torch.equal(open_tok.detach(), frozen_tok.detach())           # the same numbers
    g_open = torch.autograd.grad(open_tok.pow(2).sum(), [path.up.weight, x], retain_graph=True)
    assert float(g_open[0].abs().sum()) > 0.0 and float(g_open[1].abs().sum()) > 0.0
    g_frozen = torch.autograd.grad(frozen_tok.pow(2).sum(), [path.up.weight, x], allow_unused=True)
    assert g_frozen[0] is None                        # no lesson reaches the weight
    assert float(g_frozen[1].abs().sum()) > 0.0       # the row keeps its gradient
    # the encoder's own weights are reached in both
    w = enc.mlp[0].weight
    with path.frozen():
        t = _token(enc, x)
    assert float(torch.autograd.grad(t.pow(2).sum(), w)[0].abs().sum()) > 0.0
    # an exception inside leaves the count where it was
    with pytest.raises(RuntimeError):
        with path.frozen():
            raise RuntimeError("x")
    assert path.frozen_depth == 0


def test_a_small_output_survives_the_last_layers_bf16_grid():
    """The last layer's output holds numbers up to about 59, where one bf16 step is 0.25: 0.014 token units at a
    spread of 18. A path output of 0.002 token units added in bf16 would be rounded away; the hook adds it in
    float32."""
    enc = _encoder(bias_spread=18.0)
    x = _rows(8)
    path = VIP.build(N_IN, D)
    path.attach(enc)
    with torch.no_grad():
        path.up.weight.copy_(torch.randn(D, N_IN) * 0.002)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        on = _token(enc, x)
        path.bypass = True
        off = _token(enc, x)
        path.bypass = False
        u = path(x).float()
    assert on.dtype == torch.float32
    moved = (on - off)
    want = u - u.mean(dim=-1, keepdim=True)           # the LayerNorm removes u's mean over the 192 numbers
    assert float(u.abs().max()) < 0.014               # under one bf16 step of the last layer's output
    assert float((moved - want).abs().max()) < 0.2 * float(want.abs().max())


def test_the_first_update_moves_the_output_by_at_most_the_gain_bound():
    enc = _encoder()
    path = VIP.build(N_IN, D)
    path.attach(enc)
    x = _rows(8)
    lr = 3e-4
    opt = torch.optim.AdamW([path.up.weight], lr=lr, weight_decay=0.01)
    tok = _token(enc, x)
    (1.0 - tok.std(dim=0)).clamp_min(0).mean().backward()     # the variance term of VICReg on 8 rows
    assert float(path.up.weight.grad.abs().sum()) > 0.0       # zero at the start does not mean no gradient
    opt.step()
    with torch.no_grad():
        u = path(x)
    bound = lr * x.abs().sum(dim=-1, keepdim=True) * path.gain
    assert float(u.abs().max()) > 0.0
    assert bool((u.abs() <= bound * 1.001 + 1e-9).all())
    assert float(bound.max()) < 0.003                         # token units per update


def test_the_output_is_capped_per_row_and_untouched_under_the_cap():
    """Rows whose output's root mean square is above OUTPUT_CAP are scaled back to exactly the cap, their direction
    kept; rows under it pass bit for bit; a zero weight stays an exact zero; the weight still gets a gradient."""
    assert VIP.OUTPUT_CAP == 0.3
    path = VIP.build(N_IN, D)
    assert path.cap == VIP.OUTPUT_CAP and path.stamp()["output_cap"] == VIP.OUTPUT_CAP
    x = _rows(32)
    with torch.no_grad():
        assert float(path(x).abs().max()) == 0.0                           # zero weight: exactly 0, cap or not
        path.up.weight.copy_(torch.randn(D, N_IN) * 0.01)
        small = path(x)
        plain = F.linear(x, path.up.weight) * path.gain
        assert float(small.pow(2).mean(-1).sqrt().max()) < path.cap
        assert torch.equal(small, plain)                                   # under the cap: the linear map, bit for bit
        path.up.weight.copy_(torch.randn(D, N_IN) * 3.0)
        big = path(x)
        plain = F.linear(x, path.up.weight) * path.gain
        size_plain = plain.pow(2).mean(-1).sqrt()
        size = big.pow(2).mean(-1).sqrt()
        over = size_plain > path.cap
        assert bool(over.any())
        assert torch.allclose(size[over], torch.full_like(size[over], path.cap), rtol=1e-5)
        assert bool((size <= path.cap * (1 + 1e-5)).all())                 # never above the cap on any row
        cos = F.cosine_similarity(big, plain, dim=-1)
        assert float(cos.min()) > 1.0 - 1e-5                               # the direction is kept
        if bool((~over).any()):
            assert torch.equal(big[~over], plain[~over])
    s = path.window_summary()
    assert 0.0 < s["capped_share"] <= 1.0 and s["output_cap"] == VIP.OUTPUT_CAP
    path.window_summary()
    # the capped rows still give the weight a gradient (the factor carries none)
    xg = _rows(8)
    out = path(xg)
    g = torch.autograd.grad(out.pow(2).sum(), path.up.weight)[0]
    assert float(g.abs().sum()) > 0.0
    # another cap is the same rule at another size
    other = VIP.VestibularInputPath(N_IN, D, cap=1.0)
    with torch.no_grad():
        other.up.weight.copy_(torch.randn(D, N_IN) * 3.0)
        assert float(other(x).pow(2).mean(-1).sqrt().max()) <= 1.0 * (1 + 1e-5)
        assert float(other(x).pow(2).mean(-1).sqrt().max()) > 0.3


def test_the_weight_is_held_at_the_cap_on_the_live_rows_and_untouched_under_it():
    """A live call (gradients on, the path open) whose largest row's plain output is above the cap multiplies the
    weight by cap / that size: one factor for the whole weight, so its direction is kept and the largest row sits at
    the cap. Under the cap the weight keeps its bits. A frozen call, a call without gradient and a call with the
    hold switched off (a model step that does not train) never move the weight."""
    enc = _encoder()
    path = VIP.build(N_IN, D)
    path.attach(enc)
    x = _rows(8, seed=5)

    def largest(w):
        return float((F.linear(x, w) * path.gain).pow(2).mean(-1).sqrt().max())
    with torch.no_grad():
        path.up.weight.copy_(torch.randn(D, N_IN) * 0.01)
    small = path.up.weight.detach().clone()
    assert largest(small) < path.cap
    enc(x)
    assert torch.equal(path.up.weight.detach(), small)                # under the cap: the same bits
    with torch.no_grad():
        path.up.weight.copy_(torch.randn(D, N_IN) * 2.0)
    big = path.up.weight.detach().clone()
    assert largest(big) > 5 * path.cap
    with path.frozen():
        enc(x)
    with torch.no_grad():
        enc(x)
    path.hold_enabled = False
    enc(x)
    path.hold_enabled = True
    assert torch.equal(path.up.weight.detach(), big)                  # none of the three moved it
    path.window_summary()
    out = enc(x)                                                      # the live call
    held = path.up.weight.detach()
    ratio = (held / big).flatten()
    assert float(ratio.max() - ratio.min()) < 1e-6 and 0.0 < float(ratio.mean()) < 0.2   # one factor: direction kept
    assert largest(held) == pytest.approx(path.cap, rel=1e-5)
    assert path.up.weight.requires_grad and out.requires_grad
    g = torch.autograd.grad(out.pow(2).sum(), path.up.weight)[0]
    assert float(g.abs().sum()) > 0.0                                 # the held weight still learns
    s = path.window_summary()
    assert s["weight_held_share"] == 1.0 and s["wiring"] == "on"


def test_the_two_wirings_are_the_same_path_with_another_stamp():
    a, b = VIP.build(N_IN, D, wiring="on"), VIP.build(N_IN, D, wiring="on_delayed_encoder_only")
    assert VIP.WIRINGS == ("on", "on_delayed_encoder_only") and VIP.MODES == ("off",) + VIP.WIRINGS
    assert a.delayed_detached and not b.delayed_detached and VIP.build(N_IN, D).wiring == "on"
    assert a.stamp()["mode"] == "on" and b.stamp()["mode"] == "on_delayed_encoder_only"
    assert "neither" in a.stamp()["delayed_lesson"] and "not the path" in b.stamp()["delayed_lesson"]
    x = _rows(4)
    with torch.no_grad():
        w = torch.randn(D, N_IN) * 0.01
        a.up.weight.copy_(w)
        b.up.weight.copy_(w)
        assert torch.equal(a(x), b(x))
    with pytest.raises(ValueError, match="wiring"):
        VIP.build(N_IN, D, wiring="sideways")


def test_the_change_slope_reads_the_two_parts_of_the_sum_on_the_live_rows_only():
    """window_summary's change_slope: over the live calls of the window, each body's change from its own mean, the
    summed product of the encoder's part's change and the path's change over the path's summed squared change. A
    frozen call, a call without gradient and a bypassed call add nothing; a fresh window starts empty."""
    enc = _encoder()
    path = VIP.build(N_IN, D)
    with torch.no_grad():
        path.up.weight.copy_(torch.randn(D, N_IN) * 0.02)
    path.attach(enc)
    path.window_summary()
    rows = [_rows(6, seed=40 + t) for t in range(5)]
    own, us = [], []
    for x in rows:
        enc(x)                                                       # a live call: gradients on, the path open
        with torch.no_grad():
            path.bypass = True
            v = enc(x).squeeze(1).double()
            path.bypass = False
            own.append(v / v.std(dim=-1, keepdim=True, unbiased=False))
            us.append((F.linear(x, path.up.weight) * path.gain).double())
        with path.frozen():
            enc(_rows(6, seed=90))                                   # the delayed lesson's call: not counted
        with torch.no_grad():
            enc(_rows(6, seed=91))                                   # a measuring call: not counted
    own, us = torch.stack(own), torch.stack(us)                      # [5, 6, D]
    oc, uc = own - own.mean(0, keepdim=True), us - us.mean(0, keepdim=True)
    want = float((oc * uc).sum() / (uc * uc).sum())
    s = path.window_summary()
    assert s["live_calls"] == 5
    assert s["change_slope"] == pytest.approx(want, rel=1e-4, abs=1e-6)
    assert s["path_change"] == pytest.approx(float(uc.pow(2).mean().sqrt()), rel=1e-4)
    assert s["own_change"] == pytest.approx(float(oc.pow(2).mean().sqrt()), rel=1e-4)
    fresh = path.window_summary()
    assert "change_slope" not in fresh and "live_calls" not in fresh
    enc(rows[0])
    assert "change_slope" not in path.window_summary()               # one live call is no change yet
    # an encoder's part that takes the path's change out reads -1
    path.parts = None
    a = torch.randn(4, 3, D).double()
    for t in range(4):
        path._two_parts(-a[t] + 5.0, a[t])
    assert path.window_summary()["change_slope"] == pytest.approx(-1.0, abs=1e-9)


def test_bypass_window_summary_stamp_and_refusals():
    enc = _encoder()
    path = VIP.build(N_IN, D)
    with torch.no_grad():
        path.up.weight.copy_(torch.randn(D, N_IN) * 0.1)
    path.attach(enc)
    with pytest.raises(RuntimeError, match="attached already"):
        path.attach(enc)
    x = _rows()
    with torch.no_grad():
        on = _token(enc, x)
        path.bypass = True
        off = _token(enc, x)
        path.bypass = False
    assert not torch.equal(on, off)
    s = path.window_summary()
    assert s["version"] == VIP.INPUT_PATH_VERSION and s["encoder_calls"] == 1          # the bypassed call adds nothing
    assert s["encoder_output_sd"] == pytest.approx(18.0, rel=0.2)
    assert s["output_rms"] > 0.0 and s["output_weight_rms"] == pytest.approx(0.1, rel=0.1)
    assert set(s["field_reach"]) == set(VIP.FIELD_GROUPS)
    assert "encoder_calls" not in path.window_summary()                                 # a fresh window
    st = path.stamp()
    assert st["version"] == VIP.INPUT_PATH_VERSION and st["inputs"] == N_IN and st["bias"] is False
    path.detach()
    with torch.no_grad():
        assert torch.equal(_token(enc, x), off)
    from lem.model.encoders.mlp_project import MLPProjectEncoder
    other = MLPProjectEncoder(12, [D], D, "gelu", 0.0, True, "layernorm")
    with pytest.raises(ValueError, match="reads 12"):
        VIP.build(N_IN, D).attach(other)
    with pytest.raises(ValueError, match="built for 28"):
        path(torch.zeros(2, 12))
    # the groups cover the 28 fields once each, in the encoder's own field order
    from mimo_bridge import vestibular_sense as VS
    covered = sorted(i for a, b in VIP.FIELD_GROUPS.values() for i in range(a, b))
    assert covered == list(range(len(VS.ENCODER_FIELDS))) == list(range(N_IN))
    assert VS.ENCODER_FIELDS[0:3] == ("head_acc_x", "head_acc_y", "head_acc_z")
    assert VS.ENCODER_FIELDS[15:18] == ("trunk_up_x", "trunk_up_y", "trunk_up_z")


def test_the_path_is_built_in_its_own_random_stream():
    torch.manual_seed(5)
    a = torch.rand(3)
    torch.manual_seed(5)
    VIP.build(N_IN, D, seed=0)
    assert torch.equal(torch.rand(3), a)


# ------------------------------------------------------------------------------------------------ the learner
FLAG = "--vestibular-input-path"


def _built(tmp, name, flag, ckpt=None):
    from mimo_bridge import train_mimo as T
    extra = [] if flag is None else [FLAG, flag]
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
    path = getattr(model, "vestibular_input_path", None)
    return dict(has_path=path is not None, has_attribute=hasattr(model, "vestibular_input_path"),
                receipt="vestibular_input_path" in model.receipt,
                receipt_params=model.receipt["component_params"].get("vestibular_input_path"),
                side="vestibular_input_path" in model._side_modules(),
                n_params=int(model.n_params),
                acting=T.acting_configuration(args).get("vestibular_input_path", "absent"),
                frozen_in_planning=any(m is path for m in model.intention_gen._wellness_freeze),
                last_in_optimizer=(None if path is None else
                                   model.optimizer.param_groups[0]["params"][-1] is path.up.weight),
                inputs=(None if path is None else path.n_inputs))


class LessonProbe:
    """Per lesson term: the summed absolute gradient that reaches the path's weight (LemLite.lesson_probe)."""

    def __init__(self, params, beside=()):
        self.terms, self.params = {}, params
        self.beside = list(beside)        # the old module beside the path: the vestibular encoder's own weights
        self.reach, self.seen, self.reach_beside = {}, {}, {}

    def before_backward(self, model, loss):
        for k, v in self.terms.items():
            self.seen[k] = self.seen.get(k, 0) + 1
            if not torch.is_tensor(v) or not v.requires_grad:
                continue
            g = torch.autograd.grad(v, self.params, retain_graph=True, allow_unused=True)
            self.reach[k] = self.reach.get(k, 0.0) + sum(float(x.abs().sum()) for x in g if x is not None)
            if self.beside:
                g = torch.autograd.grad(v, self.beside, retain_graph=True, allow_unused=True)
                self.reach_beside[k] = self.reach_beside.get(k, 0.0) + sum(float(x.abs().sum())
                                                                           for x in g if x is not None)

    def after_clip(self, model, grad_norm):
        pass


class VestibularCalls:
    """Every call of the vestibular encoder while attached: whether gradients were enabled and whether the path was
    frozen, with the model step the test tagged."""

    def __init__(self, model):
        reg = getattr(model.encoder_registry, "_orig_mod", model.encoder_registry)
        self.calls, self.model, self.tag = [], model, None
        self._h = reg.encoders["vestibular"].register_forward_pre_hook(self._hook)

    def _hook(self, mod, args):
        import sys
        names = []
        f = sys._getframe(1)
        while f is not None and len(names) < 60:
            names.append(f.f_code.co_name)
            f = f.f_back
        path = self.model.vestibular_input_path
        self.calls.append(dict(tag=self.tag, grad=bool(torch.is_grad_enabled()), frozen=bool(path.frozen_depth),
                               switch_check="switch_check" in names, rows=int(args[0].shape[0])))

    def remove(self):
        self._h.remove()


@pytest.fixture(scope="module")
def learner_run(tmp_path_factory):
    """One shared run: a learner trained without the path for 12 model steps and saved; that checkpoint resumed
    without the path and with it (the switch), 4 model steps each on the same frames and random streams; the switched
    learner trained on to model step 40 with every call of the vestibular encoder recorded, saved, and read with a
    lesson probe for 10 more; its checkpoint loaded into a build with the path and refused by a build without it."""
    tmp = tmp_path_factory.mktemp("vestibular_input_path")
    R = {}
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
                        digest_skip=("vestibular_input_path",), **kw), w

    off1, _a, _l = _built(tmp, "off1", None, ckpt0)
    R["rows_off"], _w = continued(off1, 4, 12, carry)
    del off1
    gc.collect()
    on1, args_on, loaded = _built(tmp, "on1", "on", ckpt0)
    R["on"] = _facts(on1, args_on)
    R["switch"] = copy.deepcopy(on1.vestibular_input_path_switch)
    R["switch_stamp"] = copy.deepcopy(on1.receipt["vestibular_input_path"])
    R["loaded_step"] = loaded["step"]
    calls = VestibularCalls(on1)
    weight_max = {}

    def before(step, prev_exec):
        calls.tag = step
        weight_max[step] = float(on1.vestibular_input_path.up.weight.abs().max())
    R["rows_on"], w_on = continued(on1, 4, 12, carry, before_step=before)
    more = FX.drive(on1, w_on, 24, start=16, state=R["rows_on"][-1]["carry"], replay_every=2, before_step=before)
    calls.remove()
    R.update(calls=calls.calls, weight_max=weight_max)
    R["summary"] = on1.vestibular_input_path.window_summary()
    from mimo_bridge import copied_execution as CE
    from mimo_bridge import continuation_state as CS
    R["capture_has_module"] = "vestibular_input_path" in CE.capture(on1)["modules"]
    R["companion_names"] = sorted(CS.capture_learner(on1)["data"].keys())
    on1.step_count = 40
    ckpt1 = on1.save_checkpoint(40)
    R["ckpt1"] = ckpt1
    R["path_state"] = {k: v.detach().clone() for k, v in on1.vestibular_input_path.state_dict().items()}
    p = on1.vestibular_input_path.up.weight
    R["path_adam"] = {k: v.detach().clone() for k, v in on1.optimizer.state[p].items() if torch.is_tensor(v)}
    _reg = getattr(on1.encoder_registry, "_orig_mod", on1.encoder_registry)
    probe = LessonProbe([p], beside=list(_reg.encoders["vestibular"].parameters()))
    on1.lesson_probe = probe
    on1.loss_manager.term_probe = probe.terms
    FX.drive(on1, w_on, 10, start=40, state=more[-1]["carry"], replay_every=0,
             on_step=lambda step, model, res: probe.terms.clear())
    R["reach"], R["seen"] = dict(probe.reach), dict(probe.seen)
    R["reach_beside"] = dict(probe.reach_beside)
    del on1
    gc.collect()
    on2, _a, loaded = _built(tmp, "on2", "on", ckpt1)
    p2 = on2.vestibular_input_path.up.weight
    R["on2_state"] = {k: v.detach().clone() for k, v in on2.vestibular_input_path.state_dict().items()}
    R["on2_adam"] = {k: v.detach().clone() for k, v in on2.optimizer.state[p2].items() if torch.is_tensor(v)}
    R["on2_switch"] = on2.vestibular_input_path_switch
    R["on2_stamp"] = copy.deepcopy(on2.receipt["vestibular_input_path"])
    R["on2_step"] = loaded["step"]
    del on2
    gc.collect()
    # ---- the other wiring, switched on at the same checkpoint: the first model step, then 10 model steps read with
    # the lesson probe; its checkpoint then resumed under the first wiring
    onb, _a, _lb = _built(tmp, "onb", "on_delayed_encoder_only", ckpt0)
    R["switch_b"] = copy.deepcopy(onb.vestibular_input_path_switch)
    R["stamp_b"] = copy.deepcopy(onb.receipt["vestibular_input_path"])
    R["rows_b"], w_b = continued(onb, 4, 12, carry)
    pb = onb.vestibular_input_path.up.weight
    _regb = getattr(onb.encoder_registry, "_orig_mod", onb.encoder_registry)
    probe_b = LessonProbe([pb], beside=list(_regb.encoders["vestibular"].parameters()))
    onb.lesson_probe = probe_b
    onb.loss_manager.term_probe = probe_b.terms
    FX.drive(onb, w_b, 10, start=16, state=R["rows_b"][-1]["carry"], replay_every=0,
             on_step=lambda step, model, res: probe_b.terms.clear())
    R["reach_b"], R["seen_b"], R["reach_beside_b"] = dict(probe_b.reach), dict(probe_b.seen), dict(probe_b.reach_beside)
    onb.lesson_probe = None
    onb.step_count = 26
    ckpt_b = onb.save_checkpoint(26)
    del onb
    gc.collect()
    onc, _a, loaded_c = _built(tmp, "onc", "on", ckpt_b)
    R["stamp_c"] = copy.deepcopy(onc.receipt["vestibular_input_path"])
    R["switch_c"] = onc.vestibular_input_path_switch
    R["step_c"] = loaded_c["step"]
    del onc
    gc.collect()
    R["tmp"] = tmp
    return R


def test_off_builds_nothing_and_on_builds_the_path(learner_run):
    off, on = learner_run["off"], learner_run["on"]
    assert off["has_attribute"] is False and off["receipt"] is False and off["side"] is False
    assert off["acting"] == "absent"                 # the acting fingerprint of a launch without the path is unchanged
    assert on["has_path"] and on["receipt"] and on["side"] and on["acting"] == "on"
    assert on["frozen_in_planning"] and on["last_in_optimizer"]
    assert on["inputs"] == N_IN
    d = on["receipt_params"] // N_IN
    assert on["receipt_params"] == N_IN * d and on["n_params"] == off["n_params"] + N_IN * d


def test_switching_on_at_a_checkpoint_keeps_the_first_model_steps_tokens_predictions_loss_and_command(learner_run):
    R = learner_run
    sw = R["switch"]
    assert sw is not None and sw["exact"] and sw["float32"]["bit_identical"] and sw["float32"]["change_max_abs"] == 0.0
    assert sw["output_weight_max_abs"] == 0.0
    assert sw["at_step"] == 12 and sw["version"] == VIP.INPUT_PATH_VERSION and R["loaded_step"] == 12
    assert R["switch_stamp"]["switch"]["at_step"] == 12
    off, on = R["rows_off"], R["rows_on"]
    same = ("loss", "grad", "command", "gaze", "breakdown", "wellness", "replay", "torch_rng", "numpy_rng",
            "noise_rng", "tokens", "predictions", "state")
    assert R["weight_max"][12] == 0.0
    for k in same:                                   # model step 12: the weight is zero through the whole step,
        assert on[0][k] == off[0][k], k              # and the other weights' update is the update without the path
    # the comparison can see the path: the live row's lessons move the weight at the first update
    assert R["weight_max"][13] > 0.0
    assert on[1]["tokens"] != off[1]["tokens"]


def test_the_path_learns_from_the_live_row_and_not_from_the_delayed_lessons_re_encoding(learner_run):
    R = learner_run
    step_calls = [c for c in R["calls"] if c["tag"] is not None and c["grad"] and not c["switch_check"]]
    live = [c for c in step_calls if not c["frozen"]]
    delayed = [c for c in step_calls if c["frozen"]]
    assert live and delayed
    steps = sorted({c["tag"] for c in step_calls})
    for s in steps[1:]:                              # every model step after the first: one live call, one frozen
        mine = [c for c in step_calls if c["tag"] == s]
        assert sum(1 for c in mine if not c["frozen"]) == 1, s
        assert sum(1 for c in mine if c["frozen"]) == 1, s
    reach, seen = R["reach"], R["seen"]
    assert reach.get("vicreg", 0.0) > 0.0            # the live row's lesson reaches the weight
    nf = {k: v for k, v in reach.items() if k.startswith("next_frame.")}
    assert any(k.startswith("next_frame.") for k in seen)
    assert all(v == 0.0 for v in nf.values()), nf    # the delayed lesson does not


def _lessons_table(reach, beside, seen, out_name=None):
    """{lesson term: (summed absolute gradient on the path's weight, on the vestibular encoder's own weights)}."""
    table = {k: (reach.get(k, 0.0), beside.get(k, 0.0)) for k in seen}
    out = os.environ.get("VT_LESSON_REACH_OUT")
    if out:
        with open(out if out_name is None else out.replace(".json", "_%s.json" % out_name), "w",
                  encoding="utf-8") as fh:
            json.dump({k: {"seen": seen[k], "path_weight": v[0], "encoder": v[1]} for k, v in sorted(table.items())},
                      fh, indent=1)
    return table


DELAYED_ROW = ("next_frame.", "efference_", "contact_head", "latent_reg")       # the lessons that read the delayed row


def test_under_on_every_lesson_trains_the_path_and_the_encoder_alike(learner_run):
    """--vestibular-input-path on, per lesson term of the run (the imagination chain and the hold-end lessons among
    them): the summed absolute gradient on the path's weight and on the vestibular encoder's own weights, the old
    module the path sits beside. The lessons grade the SUM of the two parts, so a lesson that trains one part and
    not the other can make the two grow against each other (the prediction heads' first all-slot paths, Oct 4,
    2026). Under `on` no lesson does: the live row's lessons train both parts; the lessons that read the delayed
    row's re-encoding train neither (the previous row's vestibular token enters them without gradient); the
    imagination chain and the hold-end lessons train neither (no vestibular row is encoded on their rows)."""
    R = learner_run
    seen = R["seen"]
    assert "imagination" in seen and seen["imagination"] > 0, sorted(seen)       # the chain ran in this run
    assert any(k.startswith("tenure") for k in seen), sorted(seen)               # so did a hold-end lesson
    assert "next_frame.vestibular" in seen
    table = _lessons_table(R["reach"], R["reach_beside"], seen)
    assert sorted(k for k, (a, b) in table.items() if a > 0.0 and b == 0.0) == []      # none trains the path alone
    assert sorted(k for k, (a, b) in table.items() if a == 0.0 and b > 0.0) == []      # none the encoder alone
    assert table["vicreg"][0] > 0.0 and table["vicreg"][1] > 0.0
    for k in table:
        if k.startswith(DELAYED_ROW + ("imagination", "tenure")):
            assert table[k] == (0.0, 0.0), (k, table[k])


def test_the_exception_wiring_switches_on_alike_and_its_delayed_lesson_trains_the_encoder_alone(learner_run):
    """--vestibular-input-path on_delayed_encoder_only: the switch is exact and the first model step equals the run
    without the path, as with `on`. Then the lessons table: the terms that read the delayed row's re-encoding reach
    the vestibular encoder's weights and not the path's weight (the declared exception: the path's weight is
    detached there); no lesson trains the path alone; the imagination chain and the hold-end lessons train neither."""
    R = learner_run
    sw = R["switch_b"]
    assert sw is not None and sw["exact"] and sw["float32"]["bit_identical"] and sw["output_weight_max_abs"] == 0.0
    assert R["stamp_b"]["mode"] == "on_delayed_encoder_only" and R["stamp_b"]["switch"]["at_step"] == 12
    off, b = R["rows_off"], R["rows_b"]
    for k in ("loss", "grad", "command", "gaze", "breakdown", "wellness", "replay", "torch_rng", "numpy_rng",
              "noise_rng", "tokens", "predictions", "state"):
        assert b[0][k] == off[0][k], k               # model step 12, the switch's own: equal to the run without it
    assert b[1]["tokens"] != off[1]["tokens"]        # and the comparison can see the wiring afterwards
    seen = R["seen_b"]
    assert "next_frame.vestibular" in seen and "imagination" in seen
    table = _lessons_table(R["reach_b"], R["reach_beside_b"], seen, "on_delayed_encoder_only")
    path_only = sorted(k for k, (a, c) in table.items() if a > 0.0 and c == 0.0)
    encoder_only = sorted(k for k, (a, c) in table.items() if a == 0.0 and c > 0.0)
    assert path_only == [], path_only
    assert encoder_only and all(k.startswith(DELAYED_ROW) for k in encoder_only), encoder_only
    assert "next_frame.vestibular" in encoder_only
    assert table["vicreg"][0] > 0.0 and table["vicreg"][1] > 0.0
    for k in ("imagination", "tenure", "tenure_tok"):
        if k in table:
            assert table[k] == (0.0, 0.0), (k, table[k])
    # a checkpoint trained under one wiring continues under the other: no second switch, the change stamped
    assert R["step_c"] == 26 and R["switch_c"] is None
    assert R["stamp_c"]["mode"] == "on" and R["stamp_c"]["switch"]["at_step"] == 12
    assert R["stamp_c"]["wiring_changes"] == [{"at_step": 26, "from": "on_delayed_encoder_only", "to": "on"}]


def test_a_checkpoint_loads_both_ways_and_is_refused_where_it_must_be(learner_run):
    R = learner_run
    tmp = R["tmp"]
    assert R["on2_step"] == 40 and R["on2_switch"] is None        # no second switch
    for k, v in R["path_state"].items():
        assert torch.equal(R["on2_state"][k], v), k
    assert float(R["path_state"]["up.weight"].abs().max()) > 0.0
    assert R["path_adam"] and set(R["on2_adam"]) == set(R["path_adam"])
    for k, v in R["path_adam"].items():
        assert torch.equal(R["on2_adam"][k], v), k
    assert R["on2_stamp"]["switch"] == R["switch_stamp"]["switch"]
    off2, _a, _l = _built(tmp, "off2", None)
    with pytest.raises(ValueError, match="--vestibular-input-path on"):
        off2.load_checkpoint(R["ckpt1"])
    del off2
    gc.collect()
    other = tmp / "other_version" / os.path.basename(R["ckpt1"])
    shutil.copytree(R["ckpt1"], other)
    receipt = json.loads((other / "model_receipt.json").read_text(encoding="utf-8"))
    receipt["vestibular_input_path"]["version"] = "vestibular_input_path_v0"
    (other / "model_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    on3, _a, _l = _built(tmp, "on3", "on")
    with pytest.raises(ValueError, match="vestibular_input_path_v0"):
        on3.load_checkpoint(str(other))
    # a checkpoint whose path was trained under another output cap is refused too; its own cap loads (above)
    capped = tmp / "other_cap" / os.path.basename(R["ckpt1"])
    shutil.copytree(R["ckpt1"], capped)
    receipt = json.loads((capped / "model_receipt.json").read_text(encoding="utf-8"))
    assert receipt["vestibular_input_path"]["output_cap"] == VIP.OUTPUT_CAP
    receipt["vestibular_input_path"]["output_cap"] = 1.0
    (capped / "model_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="capped at 1.0 token units"):
        on3.load_checkpoint(str(capped))
    # a checkpoint without the stamp whose weight is not zero is not a switch that keeps the tokens
    import lem.training.checkpoint as CK
    real = CK.CheckpointManager.load

    def keep_planted(self, *a, **kw):
        out = real(self, *a, **kw)
        with torch.no_grad():
            on3.vestibular_input_path.up.weight.fill_(0.01)
        return out
    CK.CheckpointManager.load = keep_planted
    try:
        with pytest.raises(RuntimeError, match="weight is not zero"):
            on3.load_checkpoint(R["ckpt0"])
    finally:
        CK.CheckpointManager.load = real
    del on3
    gc.collect()


def test_the_stats_block_and_snapshots_carry_the_path(learner_run):
    R = learner_run
    s = R["summary"]
    assert s["version"] == VIP.INPUT_PATH_VERSION and s["output_weight_rms"] > 0.0 and s["output_rms"] > 0.0
    assert s["encoder_calls"] > 0 and set(s["field_reach"]) == set(VIP.FIELD_GROUPS)
    assert R["capture_has_module"]
    assert not any("vestibular_input_path" in n for n in R["companion_names"])        # never episode state


def test_the_trainers_flag():
    from mimo_bridge import train_mimo as T
    args = T.parse_args(FX.argv())
    assert args.vestibular_input_path == "off"
    assert "vestibular_input_path" not in T.acting_configuration(args)
    args = T.parse_args(FX.argv(extra=[FLAG, "on"]))
    assert T.acting_configuration(args)["vestibular_input_path"] == "on"
    with pytest.raises(SystemExit):
        T.parse_args(FX.argv(extra=[FLAG, "maybe"]))
    # it needs the vestibular sense in the build
    a = FX.argv(extra=[FLAG, "on"])
    i = a.index("--vestibular-sense")
    a[i + 1] = "off"
    args = T.parse_args(a)
    args.n_envs = int(args.n_envs)
    with pytest.raises(ValueError, match="--vestibular-sense on"):
        T.build_learner(args, None)
