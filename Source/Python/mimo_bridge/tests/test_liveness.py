# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""lem/model/liveness.py on small stand-in models (no checkpoint, CPU): the recomputed attention
weights equal the module's own, the pool path starts at zero and learns, a revival keeps the layer's
output and makes the dead units respond, the optimiser moments are handled as documented, and the
report counts only sampled steps. The real checkpoint is tested in test_liveness_checkpoint.py."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lem.model import liveness as LV


def test_attention_weights_equal_the_modules_own():
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(16, 4, batch_first=True)
    q, k = torch.randn(3, 5, 16), torch.randn(3, 7, 16)
    _, w = mha(q, k, k, need_weights=True, average_attn_weights=False)
    for b in range(3):
        got = LV.attention_weights(mha, (q, k, k), {}, body=b)
        assert torch.allclose(got, w[b], atol=1e-6)
    x = torch.randn(2, 6, 16)
    causal = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    _, w = mha(x, x, x, attn_mask=causal, need_weights=True, average_attn_weights=False)
    got = LV.attention_weights(mha, (x, x, x), {"attn_mask": causal}, body=1)
    assert torch.allclose(got, w[1], atol=1e-6)
    kpm = torch.zeros(2, 6, dtype=torch.bool)
    kpm[:, -2:] = True
    _, w = mha(x, x, x, key_padding_mask=kpm, need_weights=True, average_attn_weights=False)
    got = LV.attention_weights(mha, (x, x, x), {"key_padding_mask": kpm}, body=0)
    assert torch.allclose(got, w[0], atol=1e-6)


def test_entropy_share_ends():
    assert float(LV.entropy_share(torch.full((1, 8), 1 / 8))) == pytest.approx(1.0, abs=1e-6)
    one = torch.zeros(1, 8)
    one[0, 3] = 1.0
    assert float(LV.entropy_share(one)) == pytest.approx(0.0, abs=1e-6)


def test_all_element_path_starts_at_zero_and_learns():
    torch.manual_seed(0)
    p = LV.AllElementPath(5, 16, 3)
    x = torch.randn(2, 5, 16)
    assert torch.equal(p(x), torch.zeros(2, 1, 16))
    loss = (p(x) - torch.ones(2, 1, 16)).pow(2).sum()
    loss.backward()
    assert float(p.up.weight.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        p(torch.randn(2, 4, 16))


class Toy:
    """A stand-in with the shape the repair expects: attribute -> Sequential(Linear, GELU, Linear)."""

    def __init__(self, dead: int = 12, offset: float = 30.0):
        torch.manual_seed(1)
        self.head = nn.Sequential(nn.Linear(10, 16), nn.GELU(), nn.Linear(16, 3))
        self.attn = nn.MultiheadAttention(10, 2, batch_first=True)
        # a large input part shared by every sample, and `dead` rows that lean against it
        self.shared = torch.randn(10)
        self.shared = offset * self.shared / self.shared.norm()
        with torch.no_grad():
            w = self.head[0].weight
            u = self.shared / self.shared.norm()
            for j in range(dead):
                w[j] = -0.5 * u + 0.05 * torch.randn(10)
        self.receipt = {}
        self.optimizer = torch.optim.AdamW(list(self.head.parameters()), lr=3e-4)

    def inputs(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        return self.shared + torch.randn(n, 10, generator=g)


def _live_share(model, x):
    with torch.no_grad():
        pre = model.head[0](x)
    return float((pre > LV.LIVE_LINE).any(0).float().mean())


def test_revival_keeps_the_output_revives_the_units_and_handles_the_moments():
    m = Toy(dead=12)
    x = m.inputs(64, 0)
    assert _live_share(m, x) <= 0.25 + 1e-6, "the stand-in has 12 of 16 units dead"
    # a few training steps so the optimiser holds moments for every tensor
    for i in range(3):
        loss = m.head(m.inputs(32, 100 + i)).pow(2).mean()
        m.optimizer.zero_grad()
        loss.backward()
        m.optimizer.step()
    rep = LV.LivenessRepair(m, "on", seed=0, sample_every=1, revive_min_samples=64,
                            layers=("head.0",), pools=())
    m.optimizer.add_param_group({"params": list(rep.parameters()), "weight_decay": 0.0})
    rep.attach_repair(m)
    rows = rep.rows["head__0"]
    with torch.no_grad():
        before_moments = {id(p): {k: v.clone() for k, v in m.optimizer.state[p].items()
                                  if torch.is_tensor(v)} for p in m.head.parameters()}
    # the watching window: trained calls (gradient on) fill it and move the centre
    m.head(x).sum().backward()
    assert rows.seen_n == 64 and rows.centre_weight > 0
    y0 = m.head(x).detach()
    rec = rep.revive(m, "head.0", step=7)
    y1 = m.head(x).detach()
    assert rec["dead"] >= 12
    # the change recorded at the switch is the change measured here, and under its bound
    assert rec["output_change_max"] <= rec["output_change_bound"]
    assert rec["revived_live_share_on_probe"] > 0.5
    # the output is unchanged up to the dead units' per-sample variation (their GELU outputs were
    # at most 0.004 in size; the mean moved into the bias)
    assert float((y1 - y0).abs().max()) < 1e-3, float((y1 - y0).abs().max())
    # the revived units now respond
    with torch.no_grad():
        pre = m.head[0](x)
        pre = rows.apply(x, pre)
    assert float((pre > LV.LIVE_LINE).any(0).float().mean()) > 0.5
    idx = torch.tensor(rep.revived["head.0"])
    st_next = m.optimizer.state[m.head[2].weight]
    assert torch.equal(st_next["exp_avg"][:, idx], torch.zeros_like(st_next["exp_avg"][:, idx]))
    live = torch.ones(16, dtype=torch.bool)
    live[idx] = False
    want = before_moments[id(m.head[2].weight)]["exp_avg_sq"][:, live].mean(1, keepdim=True)
    assert torch.allclose(st_next["exp_avg_sq"][:, idx], want.expand(-1, len(idx)))
    # moments of the untouched positions are unchanged
    assert torch.equal(st_next["exp_avg"][:, live], before_moments[id(m.head[2].weight)]["exp_avg"][:, live])
    st_old = m.optimizer.state[m.head[0].weight]
    assert float(st_old["exp_avg"][idx].abs().sum()) == 0.0
    assert torch.equal(st_old["exp_avg"][live], before_moments[id(m.head[0].weight)]["exp_avg"][live])
    assert m.receipt["liveness_repair"]["revived"] == {"head.0": len(idx)}
    # the revived units stay live over 100 training steps on the same kind of input
    target = torch.randn(3)
    for i in range(100):
        xi = m.inputs(32, 200 + i)
        loss = (m.head(xi) - target).pow(2).mean()
        m.optimizer.zero_grad()
        loss.backward()
        m.optimizer.step()
    with torch.no_grad():
        pre = rows.apply(x, m.head[0](x))
    share = float((pre[:, idx] > LV.LIVE_LINE).any(0).float().mean())
    assert share > 0.5, share
    rep.detach()


def test_the_report_counts_only_sampled_steps():
    m = Toy(dead=0)
    rep = LV.LivenessRepair(m, "report", sample_every=3, revive_min_samples=1, layers=(), pools=())
    x = m.inputs(4, 0)
    for step in range(6):
        rep.begin_step(m, step)
        m.head(x)
        m.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        rep.end_step(m, step, 4, train=False)
    out = rep.window_summary(m)
    assert out["sampled_steps"] == 2 and out["samples"] == 8 and out["window_steps"] == 6
    assert set(out["layers"]) == {"head.0"}
    assert set(out["pools"]) == {"attn"}
    assert out["line"] == LV.LIVE_LINE and out["revived"] == []
    # the next window starts empty
    out2 = rep.window_summary(m)
    assert out2["layers"] == {} and out2["sampled_steps"] == 0
    rep.detach()


def test_a_unit_held_at_one_active_value_reads_live_but_not_varying():
    """Fails on the gauge before 2026-09-26, whose only rule was the -3 line: on H84 at 415,025 it read the vision
    and touch prediction heads 100 of 100 units live while 2 and 0.5 of 100 varied from moment to moment (the memory
    deploy readings). A unit whose input sits above -3 on every sample reads live whatever it passes forward."""
    m = Toy(dead=0, offset=0.0)
    m.held = nn.Sequential(nn.Linear(10, 8), nn.GELU(), nn.Linear(8, 2))
    with torch.no_grad():
        m.held[0].weight.zero_()
        m.held[0].bias.fill_(5.0)                     # every unit held at GELU(5) = 5.0 on every moment
        m.held[0].weight[:2] = 1e-4                   # two units move by less than one bf16 step at 5.0 (0.03)
    rep = LV.LivenessRepair(m, "report", sample_every=1, revive_min_samples=1, layers=(), pools=())
    for step in range(3):
        rep.begin_step(m, step)
        x = m.inputs(8, step)
        m.head(x)
        m.held(x)
        rep.end_step(m, step, 8, train=False)
    out = rep.window_summary(m)
    assert out["layers"]["held.0"] == 1.0, "the -3 line reads every held unit live"
    assert out["layers_varying"]["held.0"] == 0.0, "none of them varies beyond its bf16 step"
    assert out["layers_varying"]["head.0"] == 1.0, "the moving layer's units all vary"
    assert out["not_varying"] == ["held.0"] and out["below_line"] == []
    # the floor: one bf16 step at the unit's own size, and never under the widest output below the -3 line
    fl = LV.varying_floor(torch.tensor([39.0, 0.0]), torch.tensor([40.0, 0.001]), "gelu")
    assert float(fl[0]) == 0.25 and float(fl[1]) == pytest.approx(abs(float(torch.nn.functional.gelu(torch.tensor(-3.0)))))
    assert float(LV.varying_floor(torch.tensor([0.0]), torch.tensor([0.0]), "relu")[0]) == 2.0 ** -8
    rep.detach()


def test_a_revival_that_finds_no_dead_unit_changes_nothing():
    m = Toy(dead=0, offset=0.0)
    x = m.inputs(16, 0)
    rep = LV.LivenessRepair(m, "on", sample_every=1, revive_min_samples=16, layers=("head.0",), pools=())
    rep.attach_repair(m)
    m.head(x)
    w0 = [p.detach().clone() for p in m.head.parameters()]
    rec = rep.revive(m, "head.0", step=1)
    assert rec["dead"] == 0 and rep.rows["head__0"].done_py and not rep.rows["head__0"].any_py
    for a, b in zip(w0, m.head.parameters()):
        assert torch.equal(a, b)
    rep.detach()


def test_later_silent_units_are_repaired_without_resetting_previously_repaired_units():
    m = Toy(dead=0, offset=0.0)
    rep = LV.LivenessRepair(m, 'on', sample_every=1, revive_min_samples=16,
                            layers=('head.0',), pools=())
    rep.attach_repair(m)
    rows = rep.rows['head__0']
    x = m.inputs(16, 0)
    m.head(x)
    rep.end_step(m, 0, 16, train=True)
    assert rows.done_py and not rows.any_py
    for step, index in ((1, 0), (2, 1)):
        saved = rows.lin.weight[0].detach().clone()
        with torch.no_grad():
            m.head[0].weight[index].zero_()
            m.head[0].bias[index] = -20
        m.head(x)
        rep.end_step(m, step, 16, train=True)
        assert bool(rows.mask[index]), 'a startup response must not exempt this unit forever'
        if index == 1:
            assert torch.equal(rows.lin.weight[0], saved), 'later repairs reset a learned row'
    assert rep.revived['head.0'] == [0, 1]
    rep.detach()


def test_the_stand_in_names_match_the_sweeps_rule():
    """The monitor names a layer '<attribute>.<submodule path>.<index in its Sequential>', the rule
    of measurements/liveness_sweep.py, so its keys equal the sweep's and the dashboard's."""
    root = SimpleNamespace()
    root.wellness = nn.Module()
    root.wellness.approximate = nn.Module()
    root.wellness.approximate.net = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4), nn.GELU(),
                                                  nn.Linear(4, 1))
    rep = LV.LivenessRepair(root, "report", sample_every=1, revive_min_samples=1, layers=(), pools=())
    rep.attach_monitor(root)
    assert rep.linear_names == ["wellness.approximate.net.0", "wellness.approximate.net.2"]
    rep.detach()


def test_the_revival_survives_a_save_and_load_and_never_runs_while_not_training():
    m = Toy(dead=12)
    x = m.inputs(32, 0)
    rep = LV.LivenessRepair(m, "on", seed=0, sample_every=1, revive_min_samples=32,
                            layers=("head.0",), pools=())
    rep.attach_repair(m)
    m.head(x).sum().backward()                 # a trained call: 32 rows watched
    rep.begin_step(m, 0)
    rep.end_step(m, 0, 32, train=False)        # not training: no revival, whatever the window
    assert not rep.rows["head__0"].done_py
    rep.begin_step(m, 1)
    rep.end_step(m, 1, 32, train=True)
    rows = rep.rows["head__0"]
    assert rows.done_py and rows.any_py and len(rep.revived["head.0"]) >= 12
    sd = {k: v.clone() for k, v in rep.state_dict().items()}
    rt = rep.runtime_state()
    y = m.head(x).detach()
    rep.detach()
    # a fresh repair module, loaded the way a resume loads it
    m2 = m
    rep2 = LV.LivenessRepair(m2, "on", seed=0, sample_every=1, revive_min_samples=32,
                             layers=("head.0",), pools=())
    rep2.load_state_dict(sd)
    assert rep2.load_runtime_state(rt) == ["head.0"]
    rep2.after_load(m2)
    rep2.attach_repair(m2)
    assert rep2.revived == rep.revived and rep2.rows["head__0"].any_py
    assert torch.allclose(m2.head(x).detach(), y, atol=1e-6)
    assert not any(k.endswith("centre_mean") for k in sd), "the centre rides runtime_state.pt"
    rep2.detach()


def test_the_probe_prints_counts_per_100_and_no_bare_decimals():
    from mimo_bridge.tools import liveness_probe as P
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals
    result = {"checkpoint_name": "ckpt_step_1", "moments": 24, "bodies": 4, "steps_per_body": 6,
              "layer_line": LV.LAYER_LINE,
              "layers": {"tenure_heads.pose.0": 0.0, "forward_model.blocks.0.ffn.0": 0.9973},
              "pools": {"encoder_registry.encoders.motor.pool_attn": 0.0013}}
    text = "\n".join(P.lines(result))
    assert "0 of every 100 units respond  UNDER THE LINE of 50" in text
    assert "100 of every 100 units respond" in text
    assert "spread 0% of an even spread" in text
    assert bare_decimals(text) == [], bare_decimals(text)


def test_attention_on_evenly_spaced_rows_is_those_rows_of_the_full_computation():
    torch.manual_seed(2)
    mha = nn.MultiheadAttention(16, 4, batch_first=True)
    x = torch.randn(3, 40, 16)
    causal = torch.triu(torch.ones(40, 40, dtype=torch.bool), diagonal=1)
    full = LV.attention_weights(mha, (x, x, x), {"attn_mask": causal}, body=2)
    part = LV.attention_weights(mha, (x, x, x), {"attn_mask": causal}, body=2, max_rows=9)
    rows = torch.linspace(0, 39, 9).round().long()
    assert part.shape == (4, 9, 40)
    assert torch.allclose(part, full[:, rows], atol=1e-6)
    got = LV.attention_weights(mha, (x, x, x), {"is_causal": True}, body=0, max_rows=9)
    assert torch.allclose(got, LV.attention_weights(mha, (x, x, x), {"attn_mask": causal}, body=0)[:, rows],
                          atol=1e-6)


def test_the_sampling_period_shares_no_factor_with_the_hold_or_the_planner():
    assert LV.sample_period(5, 10) == 7           # H84: hold 5, planner every 10
    assert LV.sample_period(7, 1) == 11
    assert LV.sample_period(1, 1) == 7
