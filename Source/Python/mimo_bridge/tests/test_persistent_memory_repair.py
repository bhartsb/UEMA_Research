# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--persistent-memory-repair centred, version 3 (lem/model/persistent_memory.py module note; the reviews:
docs/gain_of_function_v4/agents/persistent_memory_review_2026-09-25.md and persistent_memory_review2_2026-09-25.md;
the record: docs/gain_of_function_v4/agents/persistent_memory_2026-09-25.md).

Version 3 (the second review) adds: a request every body shares and that persists over updates moves only the
ordinary bias (the centre is taken on each request, before the readout's weight), one that changes from update to
update still teaches the memory (finding 3); the ring check reads production's arithmetic (finding 5); the
encoder's Adam moments restart at the switch (finding 7); the goal reading, the next-moment error with and without
the memory's varying part (finding 2); the per-body prediction audit block.

What these tests hold:
  - the default build is the memory every earlier build computed (same state dict, same outputs);
  - the switch folds production's own bias: under a bf16 autocast the heads' bias after the switch equals the
    bf16 bias before it bit for bit, where the first version's fp32 fold moved numbers by a rounding step,
    and a small varying part afterwards reaches the heads unrounded (review finding 1); it keeps the needs
    forecast; it refuses a memory whose state still varies and bodies whose production biases differ; it is
    planned before anything is written (review question 2);
  - after the switch the candidate's scales are inside the graph (the output does not depend on the stored
    rows' size and the gradient has no outward part), the stored rows start at their target sizes, a request
    every body shares and that persists reaches neither the readout's weight nor the state, the constant is an
    ordinary bias, and the gauges are added up on the device and read once per report (review findings 4 and 9);
  - a switched checkpoint refuses a build without the repair and names the way back, from the module and
    from the receipt (review findings 2 and 7);
  - LemLite's switch holds the replay ring's stored states to the line and refuses, changing nothing, when one
    is far from the reference; it drops the readout weight's Adam moments, converts the ring and moves the
    liveness centre (review finding 5);
  - through the real trainer (slow): the switch goes into the stats stream, the checkpoint receipt and every
    later receipt; a resumed segment's launch names it; the flag-off build refuses the switched checkpoint and
    names the checkpoint from before the switch; the stats block carries the gauges' window means.
"""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lem.model.persistent_memory import (PersistentMemory, MemoryHistory, MEMORY_REPAIR_VERSION, GAUGES,
                                         SWITCHED_KEY, _rms_units, switch_limit)

INPUTS, N_MOD, D, ACTION, HIDDEN = 16, 2, 4, 3, 12


def saturated(seed=0, repair="centred", d=D, weight_sd=2.0):
    """A memory whose candidate is held at +-1 by its bias (30 against input and recurrent parts of at
    most 7 here: tanh is exactly 1 in float32), so its state is one constant pattern for every body, as
    H84's has been since update 379,324; a readout of real size."""
    torch.manual_seed(seed)
    pm = PersistentMemory(INPUTS, N_MOD, d, ACTION, hidden_size=HIDDEN, repair=repair)
    p = torch.sign(torch.randn(HIDDEN))
    p[p == 0] = 1.0
    with torch.no_grad():
        pm.recurrent.bias_ih_l0[2 * HIDDEN:] = 30.0 * p
        pm.readout.weight.normal_(0.0, weight_sd)
        pm.readout.bias.normal_(0.0, 0.5)
    return pm, p


def snapshot(B=4, T=30, seed=1, carry=None, shift=20.0):
    g = torch.Generator().manual_seed(seed)
    ev = torch.randn(INPUTS, generator=g) + 0.3 * torch.randn(B, T, INPUTS, generator=g)
    t = (torch.arange(T).float() * 0.5 + shift).expand(B, -1).contiguous()
    return dict(events=ev, times=t, now=t[:, -1].clone(), carry=carry,
                carry_time=None if carry is None else t[:, 0] - 0.5)


def saturated_snapshot(pm, p, B=4, seed=1):
    """The snapshot with the saturated carry every body shares (H84's carry: the same +-1 pattern)."""
    carry = p.reshape(1, 1, -1).expand(1, B, HIDDEN).contiguous()
    return snapshot(B=B, seed=seed, carry=carry)


def switched_open(seed=7):
    """A switched memory whose input path a lesson has opened (the candidate bias no longer dominates)."""
    pm, p = saturated(seed=seed)
    s = saturated_snapshot(pm, p)
    pm.switch(s)
    with torch.no_grad():
        pm.recurrent.bias_ih_l0[2 * HIDDEN:] = 0.0
        pm.recurrent.weight_ih_l0[2 * HIDDEN:].normal_(0.0, 0.5)
    return pm, s


# ------------------------------------------------------------------------------------ the default
def test_the_default_build_is_the_old_memory():
    torch.manual_seed(3)
    old = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN)
    assert old.repair == "off"
    keys = set(old.state_dict())
    assert not any(k in keys for k in (SWITCHED_KEY, "state_ref", "event_centre", "state_centre", "needs_gain",
                                       "codes_moment", "hidden_moment", "bias_at_switch"))
    assert len(list(old.parameters())) == 15
    s = snapshot()
    a, sa = old.encode(**s)
    b, sb = old.encode_legacy(**s)
    assert torch.equal(a, b) and torch.equal(sa, sb)
    a2, _ = old.encode(**s, observe=True)          # the lived call's flag changes nothing without the repair
    assert torch.equal(a, a2)
    assert old.gauge_report() == {}
    ex = torch.randn(4, ACTION)
    assert torch.equal(old.predict_needs(a, ex), old.outcome(torch.cat([a, ex], -1)))
    with torch.autocast("cpu", dtype=torch.bfloat16):   # the default bias is production's readout, autocast and all
        assert old.bias(a).dtype == torch.bfloat16


def test_a_repaired_build_computes_the_old_memory_until_its_switch():
    pm, p = saturated()
    twin = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN)
    twin.load_state_dict({k: v for k, v in pm.state_dict().items() if k in twin.state_dict()})
    s = saturated_snapshot(pm, p)
    assert not pm.repaired
    a, _ = pm.encode(**s, observe=True)
    b, _ = twin.encode(**s)
    assert torch.equal(a, b)
    assert float(pm.event_centre_weight) == 0.0        # nothing observed before the switch
    assert pm.gauge_report() == {}


# ---------------------------------------------------------------------------------------- the switch
def test_the_switch_keeps_the_bias_and_the_needs_forecast_of_a_saturated_memory():
    pm, p = saturated()
    s = saturated_snapshot(pm, p)
    old_state, _ = pm.encode(**s)
    assert float((old_state.abs() > 0.99).float().mean()) == 1.0
    old_bias = pm.bias(old_state)
    ex = torch.randn(4, ACTION)
    old_needs = pm.predict_needs(old_state, ex)
    rec = pm.switch(s)
    assert pm.repaired and bool(getattr(pm, SWITCHED_KEY))
    assert rec["version"] == MEMORY_REPAIR_VERSION
    assert rec["bias_change_max_abs"] <= rec["bias_change_limit"]
    assert rec["production_bias_dtype"] == "float32" and rec["production_change_max_abs"] <= rec["bias_change_limit"]
    assert float(pm.readout.weight.abs().max()) == 0.0
    new_state, _ = pm.encode(**s)
    new_bias = pm.bias(new_state)
    limit = switch_limit(float(old_bias.abs().max()))
    assert float((new_bias - old_bias).abs().max()) <= limit
    torch.testing.assert_close(pm.predict_needs(new_state, ex), old_needs, rtol=0, atol=1e-6)
    # the replay ring's saved states become centred states of the reference: about 0, the same bias
    ring_state = old_state - pm.state_ref
    assert float(ring_state.abs().max()) < 1e-6
    assert float((pm.bias(ring_state) - old_bias).abs().max()) <= limit


def test_the_fold_takes_productions_own_bf16_bias_bit_for_bit():
    """Review finding 1. Production computes the memory bias inside its bf16 autocast. The first version folded
    b + W s_ref in fp32 and production's readout then rounded bf16(b + W s) where it had rounded
    bf16(W) bf16(s) + bf16(b): some numbers moved by a rounding step (1 in 18 on H84, up to 4.0). The second
    version folds production's own number and computes the bias in fp32 afterwards: the heads' input after the
    switch is the one before it, bit for bit, and a small varying part is added unrounded."""
    pm, p = saturated(seed=17, d=64, weight_sd=40.0)          # a constant of H84's size (hundreds)
    s = saturated_snapshot(pm, p)
    old_state, _ = pm.encode(**s)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        before = pm.bias(old_state)                            # production's bias
    assert before.dtype == torch.bfloat16 and float(before.float().abs().max()) > 128
    # the first version's fold, read the way production would have read it afterwards
    fold32 = (pm.readout.bias + pm.readout.weight @ old_state.mean(0)).detach()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v1_after = torch.nn.functional.linear(old_state - old_state.mean(0), torch.zeros_like(pm.readout.weight),
                                              fold32)
    assert int((v1_after.float().reshape(before.shape) != before.float()).sum()) > 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rec = pm.switch(s)
        after = pm.bias(pm.encode(**s)[0])
    assert after.dtype == torch.float32
    assert torch.equal(after, before.float())                  # bit for bit
    assert rec["production_bias_dtype"] == "bfloat16" and rec["production_numbers_changed"] == 0
    assert rec["production_minus_fp32_max_abs"] > 0.0          # production's own rounding was kept, not undone
    # afterwards a varying part of about 0.01 per number reaches the heads as computed in fp32; production's
    # bf16 readout would have rounded most of it onto the constant's grid
    with torch.no_grad():                               # a lesson has opened the input path; a small readout
        pm.recurrent.bias_ih_l0[2 * HIDDEN:] = 0.0
        pm.recurrent.weight_ih_l0[2 * HIDDEN:].normal_(0.0, 0.5)
        pm.readout.weight.normal_(0.0, 1e-2)
    view, _ = pm.encode(**s)
    exact = torch.nn.functional.linear(view, pm.readout.weight, pm.readout.bias).reshape(before.shape)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        seen = pm.bias(view)
        old_way = pm.readout(view).float().reshape(before.shape)
    assert torch.equal(seen, exact)
    varying = exact - pm.readout.bias.reshape(1, N_MOD, 64)
    moved_old_way = float(((old_way - before.float()) != 0).float().mean())
    assert 1e-3 < float(varying.square().mean().sqrt()) < 0.1 and moved_old_way < 0.5


def test_the_switch_refuses_bodies_whose_production_biases_differ():
    """One shared bias cannot give two bodies different numbers: if production's (bf16) biases of the bodies
    differ at a number by more than the line (a rounding boundary between them), the switch refuses."""
    pm, p = saturated(seed=23)
    s = saturated_snapshot(pm, p)
    old_state, _ = pm.encode_legacy(**s)
    prod = pm.readout(old_state).detach().clone()
    prod[1, 0] += 0.25                                          # body 1's number one rounding step away
    before = copy.deepcopy(pm.state_dict())
    snap = {k: s.get(k) for k in ("events", "times", "now", "carry", "carry_time")}
    with pytest.raises(ValueError, match="differ at 1 of"):
        pm._plan_fp32(snap, old_state, prod, "bfloat16")
    for k, v in pm.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_the_switch_is_planned_before_anything_is_written():
    """Review question 2: an error between the first write and the flag left a half-switched memory. The plan
    reads and checks everything and writes nothing; the apply only copies."""
    pm, p = saturated(seed=29)
    s = saturated_snapshot(pm, p)
    before = copy.deepcopy(pm.state_dict())
    plan = pm.prepare_switch(s)
    assert not pm.repaired
    for k, v in pm.state_dict().items():
        assert torch.equal(v, before[k]), k
    pm.apply_switch(plan)
    assert pm.repaired
    # the stored candidate rows now sit at their target sizes: the scales read 1
    cu, cv, su, sv = pm.candidate_scales()
    assert abs(float(cu) - 1.0) < 1e-4 and abs(float(cv) - 1.0) < 1e-4
    assert plan["record"]["candidate_input_rows_rescaled_by"] < 0.2      # the bias of 30 held it far above
    torch.testing.assert_close(pm.scales_at_switch, torch.ones(2), rtol=0, atol=1e-4)


def test_the_switch_must_come_before_the_step_reads_the_memory():
    """The fold zeroes the readout weight in place. A graph that already holds that weight (the delayed
    lesson's bias is computed before the lived call) fails at backward, so LemLite.step switches before
    the delayed lesson; switching first leaves a graph that backpropagates."""
    import inspect
    from mimo_bridge.lem_model import LemLite
    pm, p = saturated(seed=19)
    s = saturated_snapshot(pm, p)
    x = torch.randn(4, 2 * HIDDEN, requires_grad=True)
    y = pm.bias(x).square().sum()                  # a graph holding readout.weight
    pm.switch(s)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.backward()
    pm2, p2 = saturated(seed=19)
    pm2.switch(saturated_snapshot(pm2, p2))
    x2 = torch.randn(4, 2 * HIDDEN, requires_grad=True)
    pm2.bias(pm2.encode(**saturated_snapshot(pm2, p2))[0] + x2).square().sum().backward()
    assert x2.grad is not None
    src = inspect.getsource(LemLite.step)
    assert src.index("self._persistent_memory_switch(self.memory_history.snapshot())") < \
        src.index("self.persistent_memory.encode(**self._nf_memory)")


def test_the_lived_call_after_the_graph_leaves_it_backpropagating():
    """The lived call moves the candidate's second moments after the step's graphs read them (the scales are in
    the graph, whose backward reads the moments): the moments are replaced, not changed in place."""
    pm, s = switched_open(seed=41)
    view, _ = pm.encode(**s)
    loss = pm.predict_needs(view, torch.randn(4, ACTION)).sum() + pm.bias(view).sum()
    pm.encode(**snapshot(seed=42, carry=s["carry"]), observe=True)     # the lived call, in place of the step's
    loss.backward()


def test_the_switch_refuses_a_memory_whose_state_still_varies():
    torch.manual_seed(5)
    pm = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN, repair="centred")
    with torch.no_grad():
        pm.readout.weight.normal_(0.0, 2.0)
    s = snapshot()
    state, _ = pm.encode(**s)
    assert float((state - state.mean(0)).square().mean().sqrt()) > 0.01
    before = copy.deepcopy(pm.state_dict())
    with pytest.raises(ValueError, match="not one constant"):
        pm.switch(s)
    assert not pm.repaired
    for k, v in pm.state_dict().items():                # nothing was changed by the refused switch
        assert torch.equal(v, before[k]), k


def test_the_switch_happens_once():
    pm, p = saturated()
    s = saturated_snapshot(pm, p)
    pm.switch(s)
    with pytest.raises(ValueError, match="already"):
        pm.switch(s)


# --------------------------------------------------------------------------------- after the switch
def test_the_repaired_state_varies_with_its_input_and_is_centred():
    pm, s = switched_open(seed=7)
    a, _ = pm.encode(**s)
    b, _ = pm.encode(**dict(s, events=s["events"] + 0.5 * torch.randn_like(s["events"])))
    assert float((a - b).abs().max()) > 1e-3
    # the lived calls update the running means; the centred state's mean stays near 0
    for k in range(30):
        pm.encode(**snapshot(seed=10 + k, carry=s["carry"]), observe=True)
    views = torch.cat([pm.encode(**snapshot(seed=100 + k, carry=s["carry"]))[0] for k in range(10)])
    raw = views + pm.state_centre
    assert float(views.mean(0).abs().max()) < float(raw.abs().max())
    rep = pm.gauge_report()
    assert rep["repair"] == "centred" and rep["gauge_calls"] == 30
    assert 0.0 <= rep["state_share_above_0p99"] <= 1.0 and rep["state_spread_over_window"] > 0.0
    assert set(GAUGES) <= set(rep)


def test_the_repaired_gru_is_the_fused_gru_with_scaled_candidate_rows():
    """The repaired GRU is nn.GRU's own fused computation with the candidate's rows of its weights and
    biases multiplied by the two scales (the gates untouched)."""
    pm, s = switched_open(seed=21)
    x = torch.tanh(torch.randn(4, 30, HIDDEN))
    h0 = torch.tanh(torch.randn(1, 4, HIDDEN))
    with torch.no_grad():
        cu0, cv0, _su, _sv = pm.candidate_scales()
        pm._rescale_candidate_rows(3.0, 0.25)                  # the stored rows' sizes change, the scales follow
        cu, cv, _su, _sv = pm.candidate_scales()
        states, last = pm._gru_scaled(x, h0)
    assert abs(float(cu) * 3.0 / float(cu0) - 1.0) < 1e-4 and abs(float(cv) * 0.25 / float(cv0) - 1.0) < 1e-4
    twin = torch.nn.GRU(HIDDEN, HIDDEN, batch_first=True)
    with torch.no_grad():
        H = HIDDEN
        for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            getattr(twin, name).copy_(getattr(pm.recurrent, name))
        twin.weight_ih_l0[2 * H:].mul_(cu); twin.bias_ih_l0[2 * H:].mul_(cu)
        twin.weight_hh_l0[2 * H:].mul_(cv); twin.bias_hh_l0[2 * H:].mul_(cv)
        ref_states, ref_last = twin(x, h0)
    torch.testing.assert_close(states, ref_states, rtol=0, atol=1e-6)
    torch.testing.assert_close(last, ref_last[-1], rtol=0, atol=1e-6)
    u, v, z, hprev = pm.candidate_parts(x, states, h0)
    assert u.shape == v.shape == z.shape == hprev.shape == (4, 30, HIDDEN)


def test_the_candidate_scales_are_in_the_graph():
    """Review finding 4: the first version's scales were plain numbers, so a lesson's outward push grew the
    stored rows and shrank the scales (and with them the learning rate). Now the output does not depend on
    the stored rows' size, and the gradient on each part has no component along the part (Euler); the
    control (plain-number scales) has one."""
    pm, s = switched_open(seed=31)
    H = HIDDEN
    a, _ = pm.encode(**s)
    with torch.no_grad():
        pm._rescale_candidate_rows(7.0, 0.2)
    b, _ = pm.encode(**s)
    torch.testing.assert_close(a, b, rtol=0, atol=2e-5)
    g = pm.recurrent
    weights = torch.randn(4, 2 * H, generator=torch.Generator().manual_seed(3))

    def outward(in_graph):
        pm._scales_in_graph = in_graph
        view, _ = pm.encode(**s)
        grads = torch.autograd.grad((view * weights).sum(), [g.weight_ih_l0, g.bias_ih_l0, g.weight_hh_l0,
                                                              g.bias_hh_l0])
        out = []
        for gw, gb, w, bb in ((grads[0], grads[1], g.weight_ih_l0, g.bias_ih_l0),
                              (grads[2], grads[3], g.weight_hh_l0, g.bias_hh_l0)):
            G = torch.cat([gw[2 * H:].reshape(-1), gb[2 * H:]])
            W = torch.cat([w[2 * H:].reshape(-1), bb[2 * H:]]).detach()
            out.append(abs(float(G @ W)) / (float(G.norm()) * float(W.norm())))
        return out

    assert max(outward(True)) < 1e-4
    assert max(outward(False)) > 1e-2
    pm._scales_in_graph = True


def test_a_shared_request_that_persists_moves_only_the_ordinary_bias_and_one_that_changes_teaches_the_memory():
    """The second review's finding 3. Version 2 removed from the state's gradient the mean over the bodies at
    every moment, so a lesson built on a signal every body meets at the same moment could not teach the memory and
    pulled the bodies together. Version 3 removes from each request on the memory bias the RUNNING MEAN over its
    call site's updates of the request's mean over the bodies, before the readout's weight: a request every body
    shares and that persists (the Sept 23 driver, a lesson asking for another constant) reaches neither the
    readout's weight nor any of the memory's own weights, at the first update and every later one, and lands whole
    on the ordinary bias; one that changes sign from update to update reaches them (version 2's centring, the
    control, keeps it from the memory's weights); a request that differs between bodies reaches them; the call
    sites are tracked apart; the needs forecast's slot is centred the same way; the running means are kept in the
    checkpoint."""
    from lem.model.persistent_memory import REQUEST_SITES, _RunningCentre

    def fresh():
        pm, s = switched_open(seed=37)
        with torch.no_grad():
            pm.readout.weight.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(8))   # a readout that learned
        return pm, s

    d = torch.randn(N_MOD * D, generator=torch.Generator().manual_seed(5))

    def largest(pm, s, loss_of_bias, mode="request", site="delayed", readout=True):
        params = [pm.input_projection[0].weight, pm.recurrent.weight_ih_l0, pm.recurrent.weight_hh_l0,
                  pm.query.weight] + ([pm.readout.weight] if readout else [])
        pm._centring = mode
        view, _ = pm.encode(**s)
        bias = pm.bias(view, site=site).reshape(view.shape[0], -1)
        grads = torch.autograd.grad(loss_of_bias(bias), params, allow_unused=True)
        pm._centring = "request"
        return max(0.0 if gr is None else float(gr.abs().max()) for gr in grads)

    pm, s = fresh()
    control = largest(pm, s, lambda b: (b @ d).sum(), mode="off")
    persistent = [largest(pm, s, lambda b: (b @ d).sum()) for _ in range(10)]
    assert control > 0 and max(persistent) <= 1e-5 * control
    i = REQUEST_SITES.index("delayed")
    assert pm._rc_w[i] > 0 and pm._rc_w[REQUEST_SITES.index("tenure")] == 0.0      # the sites are apart
    # a fresh memory: the same request with its sign alternating from update to update
    pm, s = fresh()
    control = largest(pm, s, lambda b: (b @ d).sum(), mode="off")
    alternating = [largest(pm, s, lambda b, k=k: ((-1.0) ** k) * (b @ d).sum()) for k in range(12)]
    assert min(alternating[2:]) >= 0.5 * control
    control_enc = largest(pm, s, lambda b: (b @ d).sum(), mode="off", readout=False)
    bodies = [largest(pm, s, lambda b, k=k: ((-1.0) ** k) * (b @ d).sum(), mode="bodies", readout=False)
              for k in range(4)]
    assert control_enc > 0 and max(bodies) <= 1e-5 * control_enc
    one_body = largest(pm, s, lambda b: b[0] @ d, site="tenure")
    assert one_body > 0
    view, _ = pm.encode(**s)
    gb, = torch.autograd.grad(pm.bias(view, site="lived").sum(), [pm.readout.bias])
    assert float(gb.abs().min()) == 4.0                          # the shared request lands on the ordinary bias
    # the needs forecast's slot: its request loses the running shared part the same way
    x = torch.randn(4, 2 * HIDDEN, requires_grad=True)
    dd = torch.randn(2 * HIDDEN)
    for _ in range(3):
        g, = torch.autograd.grad((_RunningCentre.apply(x * 1.0, pm, "needs") @ dd).sum(), [x])
        assert float(g.abs().max()) <= 1e-5 * float(dd.abs().max())
    with torch.no_grad():
        pm.needs_gain.fill_(0.5)
    w_before = pm._nrc_w
    pm.predict_needs(pm.encode(**s)[0], torch.randn(4, ACTION)).sum().backward()
    assert pm._nrc_w > w_before
    # the running means are the memory's state: kept in the checkpoint and restored with it
    sd = pm.state_dict()
    assert "request_centre" in sd and float(sd["request_centre_weight"].max()) > 0.0
    again = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN, repair="centred")
    again.load_state_dict(sd)
    assert again._rc_w == pytest.approx([float(x) for x in sd["request_centre_weight"].tolist()])
    assert again._nrc_w == pytest.approx(float(sd["needs_request_centre_weight"]))


@pytest.mark.parametrize("bodies", [3, 8])
def test_a_request_that_never_changes_leaves_exactly_zero_behind(bodies):
    """Adam divides a gradient by its own running size, so a residual of the same sign at every update moves a
    weight at full-size steps however small it is. A request every body shares, the same at every update, must
    therefore leave EXACTLY zero on the readout's weight and on the state, at every one of 1,000 updates, whatever
    the number of bodies (a mean of 3 identical float numbers can round away from them)."""
    torch.manual_seed(51)
    pm = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN, repair="centred")
    pm.repair_switched_py = True
    view = torch.randn(bodies, 2 * HIDDEN, requires_grad=True)
    with torch.no_grad():
        pm.readout.weight.normal_(0.0, 0.3)
    direction = torch.randn(N_MOD * D) * 0.1 + 0.1
    for _ in range(1000):
        gw, gv = torch.autograd.grad(-(pm.bias(view, site="lived").reshape(bodies, -1) @ direction).sum(),
                                     [pm.readout.weight, view])
        assert torch.equal(gw, torch.zeros_like(gw)) and torch.equal(gv, torch.zeros_like(gv))


def test_the_centre_follows_a_steadily_changing_request_without_lag():
    """While the ordinary bias learns what a shared request asks for, the request shrinks at a steady rate. A plain
    running mean trails such a request by about the horizon times its rate, and the trailing residual has one sign
    at every update, which Adam turns into steps of a large share of the learning rate. The lag-corrected centre
    (twice the running mean less the running mean of the running mean) follows it: after 2,000 calls (ten
    horizons) of a request falling by 1 in 3,000 of its start per call, the residual on the readout's weight is
    below 1 in 100 of what a plain running mean would leave (its lag of 199 calls times the rate, times the state's
    size: an upper bound); the start-up's transient is what remains (measured 1 in 540 at 2,000 calls)."""
    torch.manual_seed(53)
    pm = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN, repair="centred")
    pm.repair_switched_py = True
    view = torch.randn(4, 2 * HIDDEN)
    direction = torch.randn(N_MOD * D)
    for t in range(2000):
        size = 3.0 - 0.001 * t
        gw, = torch.autograd.grad(-(pm.bias(view, site="delayed").reshape(4, -1) @ (size * direction)).sum(),
                                  [pm.readout.weight])
    # a plain running mean of horizon 200 trails the request by about 199 calls x 0.001 of the direction
    plain = 199 * 0.001 * float(direction.abs().max()) * float(view.abs().sum(0).max())
    assert float(gw.abs().max()) <= 1e-2 * plain


def test_the_readouts_forward_is_nn_linears():
    """The repaired readout's forward is F.linear with the readout's own weight and bias (what nn.Linear computes):
    the centring is in the backward only."""
    pm, s = switched_open(seed=47)
    with torch.no_grad():
        pm.readout.weight.normal_(0.0, 0.3)
    view, _ = pm.encode(**s)
    assert torch.equal(pm.bias(view, site="lived"), pm.readout(view.float()).reshape(-1, N_MOD, D))


def test_the_candidate_cannot_saturate_whatever_the_weights_become():
    """The candidate's two parts have fixed sizes whatever the stored rows are: grow the candidate's rows a
    thousandfold and nothing the memory computes changes; its pre-activation's root mean square stays at most
    1.5 (the two lines, 1 and 0.5, added: Minkowski), so at most (1.5 / 8.4)^2 = 3.2 in 100 of its values can
    pass the frozen regime's 8.4 (Chebyshev); the old memory's candidate with the same weights sits past 8.4
    almost everywhere."""
    x = torch.randn(64, 96) * 1000.0 + 50.0
    y = _rms_units(x)
    torch.testing.assert_close(y.square().mean(-1), torch.ones(64), rtol=1e-4, atol=1e-4)
    assert float((y.abs() > 2.65).float().mean(-1).max()) <= 1.0 / 2.65 ** 2
    pm, p = saturated(seed=11)
    pm.switch(saturated_snapshot(pm, p))
    H = HIDDEN
    with torch.no_grad():
        pm.recurrent.bias_ih_l0[2 * H:] = 0.0
        pm.recurrent.weight_ih_l0[2 * H:].normal_(0.0, 0.5)
    s = snapshot(seed=12, T=200)             # a zero start (no saturated carry to hold), a long window
    for k in range(300):                     # the second moments follow the data (horizon SCALE_HORIZON_STEPS)
        pm.encode(**snapshot(seed=200 + k % 20), observe=True)
    before, _ = pm.encode(**s)
    with torch.no_grad():                    # the candidate's rows grow a thousandfold
        for w in (pm.recurrent.weight_ih_l0, pm.recurrent.weight_hh_l0, pm.recurrent.bias_hh_l0):
            w[2 * H:].mul_(1000.0)
    after, _ = pm.encode(**s)
    torch.testing.assert_close(after, before, rtol=0, atol=1e-4)
    # the old memory with the same weights: its candidate's pre-activation sits past 8.4 almost everywhere
    delta, age = pm._check_and_time(s["events"], s["times"], s["now"], s["carry"], s["carry_time"])
    with torch.no_grad():
        g = pm.recurrent
        x_old = pm.input_projection(torch.cat([s["events"], torch.log1p(delta.clamp_min(0)).unsqueeze(-1),
                                               torch.log1p(age).unsqueeze(-1)], -1))
        _old_state, old_states = pm.encode_legacy(**s)
        hprev = torch.cat([torch.zeros(4, 1, H), old_states[:, :-1]], 1)
        gi, gh = x_old @ g.weight_ih_l0.T + g.bias_ih_l0, hprev @ g.weight_hh_l0.T + g.bias_hh_l0
        r = torch.sigmoid(gi[..., :H] + gh[..., :H])
        a_old = gi[..., 2 * H:] + r * gh[..., 2 * H:]
    assert float((a_old.abs() > 8.4).float().mean()) >= 0.9
    pm.gauge_report()
    pm.encode(**s, observe=True)
    r = pm.gauge_report()
    assert r["candidate_rms"] <= 1.6          # the lines added, 1.5, and the window change (T = 30 to 200)
    assert r["candidate_share_above_tanh_1em7"] <= (r["candidate_rms"] / 8.4) ** 2 + 1e-9
    repaired_last = (pm.encode(**s)[0] + pm.state_centre)[:, :H]
    assert float((repaired_last.abs() > 0.99).float().mean()) <= 0.3


def test_the_constant_becomes_an_ordinary_bias():
    """One AdamW step on a loss that asks every body's bias to move one way: through a fixed +-1 state
    the readout's weights move the constant by the state's size (24 numbers here, 192 on H84) times
    the learning rate; after the switch the constant moves by about the learning rate."""
    lr = 3e-4

    def constant_move(pm, s):
        opt = torch.optim.AdamW(pm.parameters(), lr=lr, weight_decay=0.0)
        direction = torch.randn(N_MOD * D, generator=torch.Generator().manual_seed(9)).sign()
        before = pm.bias(pm.encode(**s)[0]).mean(0).detach().reshape(-1)
        loss = -(pm.bias(pm.encode(**s)[0]).mean(0).reshape(-1) * direction).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        after = pm.bias(pm.encode(**s)[0]).mean(0).detach().reshape(-1)
        return float((after - before).abs().mean())

    pm, p = saturated(seed=13)
    s = saturated_snapshot(pm, p)
    legacy = constant_move(copy.deepcopy(pm), s)
    pm.switch(s)
    repaired = constant_move(pm, s)
    assert legacy >= 10 * lr
    assert repaired <= 1.5 * lr


def test_the_gauges_are_added_on_the_device_and_read_once_per_report(monkeypatch):
    """Review finding 9: the first version read its gauges to the processor at every lived call (a wait for the
    graphics card per step). Now nothing is read during the lived call; the report reads the window's means."""
    pm, s = switched_open(seed=43)
    delta, age = pm._check_and_time(s["events"], s["times"], s["now"], s["carry"], s["carry_time"])
    raw, states, x = pm._encode_centred(s["events"], delta, age, s["carry"])
    view = raw - pm.state_centre
    reads = []
    for name in ("tolist", "item", "__float__", "__bool__", "__int__"):
        real = getattr(torch.Tensor, name)
        monkeypatch.setattr(torch.Tensor, name, (lambda real, name: lambda self, *a, **k:
                                                 reads.append(name) or real(self, *a, **k))(real, name))
    for _ in range(5):
        pm._observe_lived(s["events"], raw, view, x, states, s["carry"])
    assert reads == []
    monkeypatch.undo()
    rep = pm.gauge_report()
    assert rep["gauge_calls"] == 5 and set(GAUGES) <= set(rep)
    assert pm.gauge_report() == {}                              # a new window


# ------------------------------------------------------------------------------------ the checkpoint
def test_a_switched_checkpoint_refuses_a_build_without_the_repair_and_names_the_way_back():
    pm, p = saturated()
    s = saturated_snapshot(pm, p)
    pm.switch(s)
    for k in range(3):
        pm.encode(**snapshot(seed=20 + k, carry=s["carry"]), observe=True)
    sd = copy.deepcopy(pm.state_dict())
    plain = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN)
    with pytest.raises(ValueError, match="pre_switch_checkpoint"):
        plain.load_state_dict(sd, strict=False)
    again = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN, repair="centred")
    again.load_state_dict(sd)
    assert again.repaired
    assert again._event_w == pytest.approx(pm._event_w) and again._moment_w == pytest.approx(pm._moment_w)
    a, _ = pm.encode(**s)
    b, _ = again.encode(**s)
    assert torch.equal(a, b)
    # a memory switched by the first version (no second moments) is refused by this one
    v1 = {k: v for k, v in sd.items() if k not in ("codes_moment", "hidden_moment", "moment_weight")}
    with pytest.raises(ValueError, match="first version"):
        PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN, repair="centred").load_state_dict(v1, strict=False)
    # a checkpoint from before the switch loads into a repaired build as not yet switched
    old, _ = saturated()
    fresh = PersistentMemory(INPUTS, N_MOD, D, ACTION, hidden_size=HIDDEN, repair="centred")
    fresh.load_state_dict({k: v for k, v in old.state_dict().items() if k in dict(old.named_parameters())},
                          strict=False)
    assert not fresh.repaired


def test_the_history_carry_still_advances_after_the_switch():
    pm, p = saturated()
    h = MemoryHistory(capacity=4, interval_s=0.5)
    for i in range(20):
        h.observe(torch.randn(3, INPUTS), torch.full((3,), i * 0.05), pm)
    h.carry = p.reshape(1, 1, -1).expand(1, 3, -1).contiguous().clone()   # the saturated carry
    h.carry_time = h.times[0].clone() - 0.5
    pm.switch(h.snapshot())
    for i in range(20, 80):
        h.observe(torch.randn(3, INPUTS), torch.full((3,), i * 0.05), pm)
    assert h.carry is not None and h.carry.shape == (1, 3, HIDDEN)
    assert float(h.carry.abs().max()) <= 1.0


# ----------------------------------------------------------------------------------- LemLite's switch
def _lemlite(tmp_path, repair):
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", seed=4, imagination=False, next_frame_loss=True, persistent_memory="on",
                   persistent_memory_repair=repair, checkpoint_dir=str(tmp_path))


def _saturate(model):
    pm = model.persistent_memory
    H = pm.hidden_size
    p = torch.sign(torch.randn(H, generator=torch.Generator().manual_seed(2)))
    p[p == 0] = 1.0
    with torch.no_grad():
        pm.recurrent.bias_ih_l0[2 * H:] = 30.0 * p
        pm.readout.weight.normal_(0.0, 0.5)
        pm.readout.bias.normal_(0.0, 0.5)
    return p


def _observe(model, B, t0, n=1):
    """memory_integration.observe, the lived call (it switches a repaired memory the first time)."""
    from mimo_bridge.memory_integration import observe
    out = None
    for k in range(n):
        tokens = {m: torch.randn(B, 1, model.d_model) for m in model.active}
        mask = {m: "real" for m in model.active}
        frames = [SimpleNamespace(sim_time=float(t0 + 0.05 * k)) for _ in range(B)]
        out = observe(model, tokens, mask, {}, frames, None)
    return out


def _fill_history(model, B, n, p):
    """A lived history without the lived call (so no switch yet), then the saturated carry every body
    shares."""
    pm = model.persistent_memory
    for k in range(n):
        model.memory_history.observe(torch.randn(B, pm.input_size), torch.full((B,), 0.05 * k), pm)
    model.memory_history.carry = p.reshape(1, 1, -1).expand(1, B, -1).contiguous().clone()
    model.memory_history.carry_time = model.memory_history.times[0].clone() - 0.5


def test_lemlite_default_and_flag_contract(tmp_path):
    from mimo_bridge.lem_model import LemLite
    off = _lemlite(tmp_path / "off", "off")
    assert "persistent_memory_repair" not in off.receipt
    assert not any(k.startswith("needs_gain") for k, _ in off.persistent_memory.named_parameters())
    off.last_memory = {"events": 3, "bias_rms": 1.5}
    assert off.memory_report() == {"events": 3, "bias_rms": 1.5}      # the flag-off stats block, unchanged
    with pytest.raises(ValueError, match="needs --persistent-memory on"):
        LemLite(device="cpu", seed=4, imagination=False, persistent_memory_repair="centred")
    on = _lemlite(tmp_path / "on", "centred")
    assert on.receipt["persistent_memory_repair"]["version"] == MEMORY_REPAIR_VERSION
    ids = {id(q) for g in on.optimizer.param_groups for q in g["params"]}
    assert id(on.persistent_memory.needs_gain) in ids


def _ringed_model(tmp_path, B=3):
    from lem.model.liveness import RevivedRows
    model = _lemlite(tmp_path, "centred")
    pm = model.persistent_memory
    p = _saturate(model)
    _fill_history(model, B, 40, p)
    snap = model.memory_history.snapshot()
    old_state, _ = pm.encode(**snap)
    model.enable_replay(n_envs=B, size_per_env=16)
    ring = model.replay
    ring.memory_states[:, :10] = old_state[:, None, :].detach()
    ring.ptr, ring.count = 10, 10
    pm.readout.weight.grad = torch.ones_like(pm.readout.weight)
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    rows = RevivedRows(pm.outcome[0].in_features, pm.outcome[0].out_features)
    with torch.no_grad():
        rows.centre_mean[:] = torch.randn(rows.in_f)
        rows.centre_mean[:2 * pm.hidden_size] = old_state.mean(0)
    rows.centre_weight = 0.5
    model.liveness = SimpleNamespace(mode="on", rows={"persistent_memory__outcome__0": rows})
    return model, snap, old_state, rows


def test_lemlite_switch_keeps_the_bias_and_converts_the_ring(tmp_path):
    model, snap, old_state, rows = _ringed_model(tmp_path)
    pm = model.persistent_memory
    ring = model.replay
    assert pm.readout.weight in model.optimizer.state
    old_bias = pm.bias(old_state).detach()
    before_centre = rows.centre_mean.clone()
    rec = model._persistent_memory_switch(snap)
    assert pm.readout.weight not in model.optimizer.state
    assert rec["readout_weight_adam"].startswith("dropped")
    assert rec["ring_states_checked"] == 30 and rec["ring_states_above_line"] == 0
    assert rec["ring_bias_change_max_abs"] <= rec["bias_change_limit"]
    assert rec["ring_states_converted"] == 30 and rec["pre_switch_checkpoint"] is None
    ring_bias = pm.bias(ring.memory_states[:, 9].float())
    assert float((ring_bias - old_bias).abs().max()) <= rec["bias_change_limit"]
    torch.testing.assert_close(rows.centre_mean[:2 * pm.hidden_size],
                               before_centre[:2 * pm.hidden_size] - pm.state_ref, rtol=0, atol=1e-6)
    torch.testing.assert_close(rows.centre_mean[2 * pm.hidden_size:], before_centre[2 * pm.hidden_size:],
                               rtol=0, atol=0)
    assert model.receipt["persistent_memory_repair"]["switch"]["bias_change_max_abs"] == rec["bias_change_max_abs"]
    assert model.persistent_memory_switch_unwritten is rec
    model.liveness = None


def test_lemlite_switch_drops_the_encoders_adam_moments(tmp_path):
    """The second review's finding 7: the encoder's saved moments (about 1e-11, Adam's bias correction off at the
    step count) would give its first steps after the switch 3 to 5 times the learning rate. The switch drops them
    with the readout weight's, so Adam restarts each tensor with its own bias correction."""
    model, snap, old_state, rows = _ringed_model(tmp_path)
    pm = model.persistent_memory
    g = pm.recurrent
    encoder = (pm.input_projection[0].weight, pm.input_projection[0].bias, g.weight_ih_l0, g.weight_hh_l0,
               g.bias_ih_l0, g.bias_hh_l0, pm.query.weight, pm.query.bias, pm.key.weight)
    for q in encoder:
        q.grad = torch.full_like(q, 1e-7)
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    assert all(q in model.optimizer.state for q in encoder)
    rec = model._persistent_memory_switch(snap)
    assert not any(q in model.optimizer.state for q in encoder)
    assert rec["encoder_adam"].startswith("9 of 9 tensors dropped")
    model.liveness = None


def test_the_ring_check_reads_replays_arithmetic(tmp_path):
    """The second review's finding 5: version 2 judged the ring in fp32 and refused switches whose replay would not
    change on the card, where bf16 rounds every stored state number above 0.998 in size to the same +-1. Inside a
    bf16 autocast, stored states a ten-thousandth from the saturated reference now pass (their bias in replay's
    arithmetic is the fold's), while the fp32 reading, kept beside it, is above the line."""
    model, snap, old_state, rows = _ringed_model(tmp_path)
    pm = model.persistent_memory
    with torch.no_grad():
        pm.readout.weight.mul_(80.0)
        # every stored state number a ten-thousandth inside +-1 (bf16's step just below 1 is 0.0039)
        model.replay.memory_states[:, :10] = (old_state.sign() * (1.0 - 1e-4))[:, None, :].detach()
    snap_states = pm.encode_legacy(**{k: snap[k] for k in ("events", "times", "now", "carry", "carry_time")})[0]
    assert float((snap_states.abs() > 0.9999).float().mean()) == 1.0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rec = model._persistent_memory_switch(snap)
    assert rec["ring_states_above_line"] == 0 and rec["ring_bias_change_max_abs"] <= rec["bias_change_limit"]
    assert rec["ring_bias_change_fp32_max_abs"] > rec["bias_change_limit"]
    assert rec["ring_check_batch"] == 48
    model.liveness = None


def test_the_goal_reading_takes_out_the_varying_part_and_reports_it_with_and_without_body_2(tmp_path):
    """The second review's finding 2: the gain the repair exists for. The goal reading recomputes the delayed
    lesson's predictions with the memory's varying part (the readout weight times the centred state) taken out of
    the latent, and (the third review's finding 4) with another body's and a second other body's varying part in the
    body's place; at the switch the readout weight is 0, so every arm is the lesson's own prediction; with a readout
    weight the arms differ. The report gives each sense's error under each arm and the three paired readings
    (without / with, other / with, other2 / other), over all bodies and without body 2. The extra passes through the
    heads leave the liveness watch as they found it."""
    from mimo_bridge.lem_model import LemLite
    model = LemLite(device="cpu", seed=4, imagination=False, next_frame_loss=True, persistent_memory="on",
                    persistent_memory_repair="centred", checkpoint_dir=str(tmp_path))
    pm = model.persistent_memory
    p = _saturate(model)
    B = 3
    _fill_history(model, B, 10, p)
    model._persistent_memory_switch(model.memory_history.snapshot())
    snap = model.memory_history.snapshot()
    state, _ = pm.encode(**snap)
    d = model.d_model
    nf_in = {m: torch.randn(B, 1, d) for m in model.active}
    ctx = torch.zeros(B, (model.ctx_len - 1) * model.n_mod, d)
    with torch.no_grad():
        _l, last, _c = model.forward_model(nf_in, ctx, memory_bias=pm.bias(state))
    fake_rows = SimpleNamespace(seen_max="kept", act_sum="kept", seen_n=7, probe_x="kept")
    model.liveness = SimpleNamespace(active=True, rows={"x": fake_rows})
    arms = model._memory_goal_variants(nf_in, last, state, None, B)
    with torch.no_grad():
        base = model.prediction_heads(last, current={m: v.squeeze(1) for m, v in nf_in.items()})
    assert set(arms) == {"without", "other", "other2"}
    for arm in arms.values():                                   # the readout weight is 0 at the switch
        for m in arm:
            assert torch.equal(arm[m], base[m].detach())
    assert model.liveness.active is True and fake_rows.seen_n == 7 and fake_rows.seen_max == "kept"
    with torch.no_grad():
        pm.readout.weight.normal_(0.0, 0.3)
    varying_state = state + 0.5 * torch.randn_like(state)
    arms = model._memory_goal_variants(nf_in, last, varying_state, None, B)
    assert any(float((arms["without"][m] - base[m]).abs().max()) > 1e-4 for m in base)
    # another body's varying part in the body's place: body b reads body b + 1's (mod B), b + 2's for the control
    with torch.no_grad():
        v = torch.nn.functional.linear(varying_state.float(), pm.readout.weight.float()).reshape(last.shape)
        cur = {m: x.squeeze(1) for m, x in nf_in.items()}
        for name, k in (("other", 1), ("other2", 2)):
            want = model.prediction_heads(last - v + torch.roll(v, -k, 0), current=cur)
            for m in want:
                assert torch.allclose(arms[name][m], want[m], atol=1e-6)
    tokens_raw = {m: torch.randn(B, 1, d) for m in model.active}
    senses = [m for m in model.pred_mods if m in base]
    model.memory_goal_window = None
    model._accumulate_memory_goal(base, arms, tokens_raw, senses)
    model._accumulate_memory_goal(base, arms, tokens_raw, senses)
    rep = model.memory_goal_report()
    m0 = senses[0]
    tgt = tokens_raw[m0].squeeze(1)
    e = {k: (p[m0] - tgt).square().mean(-1) for k, p in dict(arms, **{"with": base}).items()}
    assert rep[m0]["steps"] == 2
    assert rep[m0]["all_bodies"]["error_with"] == pytest.approx(float(e["with"].mean()), rel=1e-5)
    assert rep[m0]["without_body_2"]["error_without"] == pytest.approx(float(e["without"][:2].mean()), rel=1e-5)
    import math
    assert rep[m0]["all_bodies"]["ln_without_over_with"] == pytest.approx(
        math.log(float(e["without"].mean() / e["with"].mean())), rel=1e-5)
    assert rep[m0]["all_bodies"]["ln_other_over_with"] == pytest.approx(
        math.log(float(e["other"].mean() / e["with"].mean())), rel=1e-5)
    assert rep[m0]["without_body_2"]["ln_other2_over_other"] == pytest.approx(
        math.log(float(e["other2"][:2].mean() / e["other"][:2].mean())), rel=1e-5)
    assert rep[m0]["per_body_ln_other_over_with"][1] == pytest.approx(
        math.log(float(e["other"][1] / e["with"][1])), rel=1e-5)
    assert model.memory_goal_report() == {}                      # a new window
    model.liveness = None


def test_the_per_body_audit_block_pools_with_and_without_body_2():
    """The per-body block of the prediction audit (prediction_audit.paired_rows, accumulate_rows, summarize_rows):
    every body's errors and the pooled ratio, improvement over another body's order and sensitivity over all
    bodies and without body 2, by the same arithmetic as the audit's own summary; nothing read from the device
    until the report."""
    from mimo_bridge.prediction_audit import (paired_errors, paired_rows, accumulate, accumulate_rows, summarize,
                                              summarize_rows)
    g = torch.Generator().manual_seed(1)
    sums, rows = {}, {}
    for _ in range(3):
        p, old, tgt, sh = (torch.randn(8, 16, generator=g) for _ in range(4))
        accumulate(sums, {"vision": paired_errors(p, old, tgt, sh)})
        accumulate_rows(rows, {"vision": paired_rows(p, old, tgt, sh)})
    whole = summarize(sums)["modalities"]["vision"]
    per = summarize_rows(rows)
    v = per["senses"]["vision"]
    assert per["excluded_bodies"] == [2]
    assert v["all_bodies"]["ratio"] == pytest.approx(whole["prediction_mse"] / whole["persistence_mse"], rel=1e-5)
    assert v["all_bodies"]["improvement_over_shuffled"] == pytest.approx(whole["improvement_over_shuffled"], rel=1e-5)
    assert v["all_bodies"]["sensitivity"] == pytest.approx(whole["sensitivity"], rel=1e-5)
    keep = [b for b in range(8) if b != 2]
    pb = v["per_body"]
    wo = sum(pb["prediction_mse"][b] for b in keep) / sum(pb["persistence_mse"][b] for b in keep)
    assert v["without_body_2"]["ratio"] == pytest.approx(wo, rel=1e-5)
    assert v["without_body_2"]["transitions"] == 21 and pb["transitions"] == [3] * 8


def test_lemlite_switch_refuses_a_ring_far_from_the_reference_and_changes_nothing(tmp_path):
    """Review finding 5: a ring that still holds moments from before every body's state was the same (0.149 per
    state number at 380,550) would change replay's bias at the switch. The switch refuses, and nothing - the
    memory, the optimizer, the ring, the liveness centre - is changed."""
    model, snap, old_state, rows = _ringed_model(tmp_path)
    pm = model.persistent_memory
    ring = model.replay
    with torch.no_grad():
        ring.memory_states[1, 3] += 0.15
    before = {k: v.clone() for k, v in pm.state_dict().items()}
    ring_before = ring.memory_states.clone()
    centre_before = rows.centre_mean.clone()
    with pytest.raises(ValueError, match="1 of 30 written replay moments"):
        model._persistent_memory_switch(snap)
    assert not pm.repaired and pm.readout.weight in model.optimizer.state
    for k, v in pm.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert torch.equal(ring.memory_states, ring_before) and torch.equal(rows.centre_mean, centre_before)
    assert model.persistent_memory_switch is None and model.persistent_memory_switch_unwritten is None
    model.liveness = None


def test_lemlite_switches_at_the_first_lived_moment_and_reports(tmp_path):
    model = _lemlite(tmp_path, "centred")
    pm = model.persistent_memory
    p = _saturate(model)
    B = 3
    _fill_history(model, B, 10, p)
    assert not pm.repaired
    snap = model.memory_history.snapshot()
    old_bias = pm.bias(pm.encode(**snap)[0]).detach()
    calls = []
    real = model._persistent_memory_switch
    model._persistent_memory_switch = lambda s: calls.append(1) or real(s)
    state, bias, _snap, _needs = _observe(model, B, 0.5, n=1)
    assert pm.repaired and calls == [1]
    # the first lived moment after the switch: every body's bias is the old constant
    assert float((bias - old_bias.mean(0, keepdim=True)).abs().max()) <= model.persistent_memory_switch["bias_change_limit"]
    _observe(model, B, 0.6, n=2)
    assert calls == [1]
    model.last_memory = {"events": 11}
    rep = model.memory_report()
    assert rep["events"] == 11 and rep["repair_switched"] and rep["gauge_calls"] == 3
    for k in GAUGES:
        assert k in rep


def test_load_checkpoint_refuses_a_switched_receipt_and_names_the_checkpoint_from_before(tmp_path):
    """Review finding 2: after a deploy the lineage goes back only by the checkpoint saved before the switch.
    The receipt check refuses a build without the flag (and a build of another repair version) by name."""
    from lem.model.persistent_memory import MEMORY_REPAIR_VERSION as V
    off = _lemlite(tmp_path / "off", "off")
    ck = tmp_path / "ckpt_step_9"
    ck.mkdir()
    receipt = dict(body_layout=off.body_layout, persistent_memory="on",
                   persistent_memory_repair=dict(version=V, mode="centred",
                                                 switch=dict(switched_at_step=7,
                                                             pre_switch_checkpoint="D:/runs/h/ckpt_step_6")))
    (ck / "model_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="ckpt_step_6"):
        off.load_checkpoint(str(ck))
    on = _lemlite(tmp_path / "on", "centred")
    receipt["persistent_memory_repair"]["version"] = "persistent_memory_centred_v1"
    (ck / "model_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="repair version"):
        on.load_checkpoint(str(ck))


@pytest.mark.slow
def test_training_steps_switch_and_backpropagate_in_a_real_world(tmp_path):
    """A real crib world and full training steps: a learner that trained two steps without the repair
    (its history holds lived moments) is switched at the start of its third step, before the delayed
    lesson reads the memory, and that step's backward and optimizer step run; the first step of a fresh
    repaired learner switches at its lived call. Every bias stays finite and the stats block carries the
    repaired memory's gauges."""
    import numpy as np
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite
    torch.set_num_threads(1)
    world = SingleEnvSource(scene="crib", vision=False, run_dir=str(tmp_path / "world"), age=24,
                            caregiver_seconds=0, body_function="modulate", contact_profile="mechanical_v2")
    try:
        learner = LemLite(device="cpu", imagination=False, no_vision=True, next_frame_loss=True,
                          persistent_memory="on", smooth_actions=.35, checkpoint_dir=None)
        frames = world.reset()
        dev = learner.dev_limits(0)
        executed = None
        for k in range(2):
            learner.step(frames, dev, train=True, step_index=k, executed_action=executed)
            executed = np.zeros((1, len(learner.handle_names), 9), np.float32)
            frames = world.step(executed)
        pm = learner.persistent_memory
        assert learner.memory_history.previous is not None and learner._nf_memory is not None
        pm.enable_repair("centred")                      # as a relaunch with the flag would build it
        calls = []
        real = learner._persistent_memory_switch
        learner._persistent_memory_switch = lambda s: calls.append(1) or real(s)
        learner.step(frames, dev, train=True, step_index=2, executed_action=executed)
        assert calls == [1] and pm.repaired
        assert torch.isfinite(learner._memory_bias_current).all()
        frames = world.step(executed)
        learner.step(frames, dev, train=True, step_index=3, executed_action=executed)
        assert calls == [1]
        rep = learner.memory_report()
        assert rep["gauge_calls"] == 2 and "update_gate_share_above_0p99" in rep
        fresh = LemLite(device="cpu", imagination=False, no_vision=True, next_frame_loss=True,
                        persistent_memory="on", persistent_memory_repair="centred", smooth_actions=.35,
                        checkpoint_dir=None)
        fresh.step(world.reset(), fresh.dev_limits(0), train=True, step_index=0)
        assert fresh.persistent_memory.repaired
    finally:
        world.close()


def _trainer_flags(tmp_path, name, steps, repair, resume=None):
    flags = ['--out', str(tmp_path / name), '--device', 'cpu', '--n-envs', '1', '--no-vision',
             '--curriculum', 'off', '--age-binding', 'none', '--delay-binding', 'off', '--episode-steps', '0',
             '--action-repeat', '1', '--motor-hold', '1', '--minutes', '0', '--max-steps', str(steps),
             '--no-babble', '--body-function', 'modulate', '--persistent-memory', 'on',
             '--imagination', 'off', '--next-frame-loss', 'on', '--contact-profile', 'mechanical_v4',
             '--replay', 'off', '--report-every', '1', '--video-every-min', '0', '--decoder-kp', '6.0',
             '--checkpoint-every-min', '0', '--checkpoint-storage', str(tmp_path / 'storage' / name)]
    if repair:
        flags += ['--persistent-memory-repair', 'centred', '--persistent-memory-constant-return', '50']
    if resume:
        flags += ['--resume', str(resume)]
    return flags


def _records(tmp_path, name):
    return [json.loads(line) for line in
            (tmp_path / name / 'replay_training_stats.jsonl').read_text().splitlines()]


def _last_checkpoint(tmp_path, name):
    summary = json.loads((tmp_path / name / 'summary.json').read_text())
    assert summary["ok"], summary["exception"]
    return Path(summary["checkpoints"][-1]["path"])


@pytest.mark.slow
def test_the_trainer_records_the_switch_and_every_later_segment_names_it(tmp_path, monkeypatch):
    """Review findings 2 and 7, through the real trainer on the CPU (one body): a segment without the flag,
    a segment with it (it switches at its first step: a persistent_memory_switch record in the stream, with the
    checkpoint it resumed as the way back, and the switch in its checkpoint's receipt), a later segment with it
    (its launch names the loaded switch, no second switch, the receipt still carries it), and a segment without
    the flag (refused before training, naming the checkpoint from before the switch)."""
    from mimo_bridge.tools import preflight
    from mimo_bridge.train_mimo import main
    monkeypatch.setattr(preflight, 'run_probes', lambda path, env_kw=None: dict(ok=True, checks=[]))
    torch.set_num_threads(1)
    assert main(_trainer_flags(tmp_path, 'a', 3, repair=False)) == 0
    ck_a = _last_checkpoint(tmp_path, 'a')
    assert main(_trainer_flags(tmp_path, 'b', 6, repair=True, resume=ck_a)) == 0
    recs = _records(tmp_path, 'b')
    switches = [r for r in recs if r['type'] == 'persistent_memory_switch']
    assert [r['part'] for r in switches] == ['switch']
    sw = switches[0]['switch']
    assert sw['version'] == MEMORY_REPAIR_VERSION and os.path.samefile(sw['pre_switch_checkpoint'], ck_a)
    assert sw['bias_change_max_abs'] <= sw['bias_change_limit']
    blocks = [r['persistent_memory'] for r in recs if r['type'] == 'mimo']
    assert blocks and all(b.get('repair_switched') for b in blocks)
    assert all('update_gate_share_above_0p99' in b and b['gauge_calls'] >= 1 for b in blocks)
    # the goal reading (its window holds the delayed lesson's graded senses) and the per-body audit block
    goals = [b['goal'] for b in blocks if b.get('goal')]
    assert goals and all('ln_without_over_with' in g['all_bodies'] for gl in goals for g in gl.values())
    audits = [r.get('prediction_audit_per_body') for r in recs if r['type'] == 'mimo']
    assert audits and all(a is not None and 'senses' in a for a in audits)
    # the constant's return (version 4): its start and target per named slot in the switch record, its clock and
    # per-slot sizes in every report's memory block
    ret = sw['constant_return']
    assert ret['updates'] > 0 and isinstance(ret['start_per_slot'], dict) and 'tactile' in ret['start_per_slot']
    assert set(ret['start_per_slot']) == set(ret['target_per_slot'])
    clocks = [b['updates_since_switch'] for b in blocks]
    assert clocks == sorted(clocks) and clocks[-1] >= 1
    assert all(isinstance(b['constant_effective_rms_per_slot'], dict) for b in blocks)
    ck_b = _last_checkpoint(tmp_path, 'b')
    saved = json.loads((ck_b / 'model_receipt.json').read_text())['persistent_memory_repair']['switch']
    assert saved['switched_at_step'] == sw['switched_at_step']
    assert main(_trainer_flags(tmp_path, 'c', 8, repair=True, resume=ck_b)) == 0
    recs = _records(tmp_path, 'c')
    assert [r['part'] for r in recs if r['type'] == 'persistent_memory_switch'] == ['loaded']
    ck_c = _last_checkpoint(tmp_path, 'c')
    saved = json.loads((ck_c / 'model_receipt.json').read_text())['persistent_memory_repair']['switch']
    assert os.path.samefile(saved['pre_switch_checkpoint'], ck_a)
    assert main(_trainer_flags(tmp_path, 'd', 10, repair=False, resume=ck_c)) == 1
    summary = json.loads((tmp_path / 'd' / 'summary.json').read_text())
    assert 'switched persistent memory' in summary['exception'] and ck_a.name in summary['exception']


# ------------------------------------------------------------------ the constant's return (2026-09-26, version 4)
def test_the_constant_returns_to_its_target_size_and_cannot_regrow():
    """The coordinator's scale-drift finding: the switch folds the constant whole (every head's input unchanged) and
    zeroes the readout weight; the constant itself only returns with the schedule. At the switch the bias is exactly
    the fold; after k of T updates each slot above its target reads start x (target / start)^(k / T); from T on it
    reads its target; a slot under its target is left as it is; growing the stored constant tenfold changes nothing
    the heads read, and the loss's gradient has no component along the stored constant's size (no gradient can grow
    it back); the gauges read each slot; a checkpoint keeps the clock."""
    pm, p = saturated(seed=11, d=8)
    s = saturated_snapshot(pm, p)
    with torch.no_grad():
        pm.readout.bias.mul_(40.0)                          # a large constant, as H84's at 415,075
    old_state, _ = pm.encode(**s)
    old_bias = pm.bias(old_state)
    start_all = old_bias.mean(0).square().mean(-1).sqrt()
    targets = [float(start_all[0]) / 20.0, float(start_all[1]) * 2.0]   # slot 0 shrinks, slot 1 is under its target
    pm.set_constant_return(targets, 10)
    rec = pm.switch(s)
    assert rec["constant_return"]["updates"] == 10 and rec["constant_return"]["target_per_slot"][0] == pytest.approx(targets[0])
    state, _ = pm.encode(**s)
    b0 = pm.bias(state)
    assert torch.equal(b0, pm.bias(state)) and float((b0 - old_bias).abs().max()) <= switch_limit(float(old_bias.abs().max()))
    start = pm.constant_start.clone()

    def slot_rms():
        return pm.bias(state).detach().mean(0).square().mean(-1).sqrt()
    for _ in range(5):
        pm.advance_constant_return()
    r = slot_rms()
    assert float(r[0]) == pytest.approx(float(start[0]) * (targets[0] / float(start[0])) ** 0.5, rel=1e-4)
    assert float(r[1]) == pytest.approx(float(start[1]), rel=1e-6)
    for _ in range(15):
        pm.advance_constant_return()
    r = slot_rms()
    assert float(r[0]) == pytest.approx(targets[0], rel=1e-4) and float(r[1]) == pytest.approx(float(start[1]), rel=1e-6)
    with torch.no_grad():
        pm.readout.bias.mul_(10.0)                          # the stored constant grows tenfold
    r = slot_rms()
    assert float(r[0]) == pytest.approx(targets[0], rel=1e-4) and float(r[1]) == pytest.approx(targets[1], rel=1e-4)
    pm.readout.bias.grad = None
    u = torch.randn_like(b0)
    (pm.bias(state) * u).sum().backward()
    g = pm.readout.bias.grad.reshape(N_MOD, 8)
    b = pm.readout.bias.detach().reshape(N_MOD, 8)
    along = (g * b).sum(-1) / (g.norm(dim=-1) * b.norm(dim=-1))
    assert float(along.abs().max()) < 1e-4                  # capped: the gradient cannot change the size the heads read
    pm.encode(**s, observe=True)
    rep = pm.gauge_report()
    assert rep["updates_since_switch"] == 20 and rep["constant_return_updates"] == 10
    assert rep["constant_effective_rms_per_slot"][0] == pytest.approx(targets[0], rel=1e-4)
    assert rep["constant_cap_per_slot"][0] == pytest.approx(targets[0], rel=1e-4)
    assert len(rep["varying_bias_rms_per_slot"]) == N_MOD
    twin = PersistentMemory(INPUTS, N_MOD, 8, ACTION, hidden_size=HIDDEN, repair="centred")
    twin.load_state_dict(pm.state_dict())
    assert twin._return_on and float(twin.updates_since_switch) == 20.0
    torch.testing.assert_close(twin.bias(state), pm.bias(state), rtol=0, atol=0)


def test_without_a_return_the_constant_stays_as_folded():
    """A length of 0 (or no targets set): the switch writes the return off and the constant stays as folded after
    any number of updates (version 3's behaviour)."""
    pm, p = saturated(seed=12)
    s = saturated_snapshot(pm, p)
    pm.switch(s)
    state, _ = pm.encode(**s)
    b0 = pm.bias(state)
    for _ in range(50):
        pm.advance_constant_return()
    assert not pm._return_on and torch.equal(pm.bias(state), b0)
    with pytest.raises(ValueError):
        pm.set_constant_return([1.0], 10)


def test_lemlite_names_the_constant_return_targets_by_sense_slot(tmp_path):
    """LemLite sets each sense slot's pre-deploy size (PRE_DEPLOY_BIAS_RMS) and the schedule's length (the trainer's
    --persistent-memory-constant-return) before any switch, by the forward model's slot names; the trainer refuses
    the repair without that length."""
    from mimo_bridge.lem_model import LemLite
    from lem.model.persistent_memory import PRE_DEPLOY_BIAS_RMS
    model = LemLite(device="cpu", seed=4, imagination=False, next_frame_loss=True, persistent_memory="on",
                    persistent_memory_repair="centred", persistent_memory_constant_return=5000,
                    checkpoint_dir=str(tmp_path))
    pm = model.persistent_memory
    assert pm._return_updates == 5000
    assert pm._return_targets.tolist() == pytest.approx([PRE_DEPLOY_BIAS_RMS.get(m, 0.0) for m in model.active])
    from mimo_bridge.train_mimo import main
    flags = [f for f in _trainer_flags(tmp_path, 'refused', 2, repair=True)]
    i = flags.index('--persistent-memory-constant-return')
    with pytest.raises(ValueError, match="needs --persistent-memory-constant-return"):
        main(flags[:i] + flags[i + 2:])
    assert not (tmp_path / 'refused').exists()                  # refused before anything was written
