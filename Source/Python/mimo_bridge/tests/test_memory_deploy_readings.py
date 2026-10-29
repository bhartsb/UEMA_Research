# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The memory deploy's readings (mimo_bridge/tools/memory_deploy_readings.py) and their place in the regression pack.

Each reading is checked where its answer is known, and each check fails on the fault it names: the batched
attention spread equals production's per-element recomputation (a wrong mask, scale or head split breaks it);
an attention whose logits are multiplied a million-fold reads about 0 and one whose keys carry nothing reads the
even spread exactly; a body whose moments are all the same has no live unit while the others have; units planted
dead read dead under both rules; units held at one active value read dead here and live by production's -3 line;
the floor arithmetic on known outputs; the capture sees a replacement an earlier hook made (the liveness repair's
revived units); a memory that does not vary is worth exactly 0; the error without the memory's varying part equals
the replay lesson's own error on the mean state; a memory whose variation carries the next moment removes the whole
error; body 2 is left out of the pooled readings; the lines, verdicts and table; the pack's compare carries them.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.tools import memory_deploy_readings as MDR
from mimo_bridge.tools import regression_pack as RP


@pytest.fixture(scope="module")
def lite():
    """A small real LemLite (128 wide, context 4, 2 blocks) with vision, the gaze copy, the efference heads, the
    tenure pose head, the persistent memory (its readout given weights, so its bias varies with the state) and a
    4-body replay ring of 40 random moments; the gaze copy's centre set from one real latent."""
    torch.set_num_threads(2)
    from mimo_bridge.lem_model import LemLite, command_for_encoder
    torch.manual_seed(0)
    m = LemLite(device="cpu", seed=0, imagination=False, body_layout="trunk_v1", persistent_memory="on",
                self_state="learn", efference_heads=True, next_frame_loss=True, action_tenure_loss=True)
    B, H = 4, len(m.handle_names)
    m.enable_replay(n_envs=B, size_per_env=40)
    g = torch.Generator().manual_seed(1)
    mem = m.replay.memory_states.shape[-1]
    for t in range(40):
        toks = {k: torch.randn(B, 1, m.d_model, generator=g) for k in m.active}
        ach = torch.zeros(B, H, 10)
        ach[..., 0:3] = torch.randn(B, H, 3, generator=g) * 0.05
        q = torch.randn(B, H, 4, generator=g)
        ach[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
        order = torch.zeros(B, H, 9)
        order[..., 0:3] = torch.randn(B, H, 3, generator=g) * 10.0
        q = torch.randn(B, H, 4, generator=g)
        order[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
        order[..., 7] = 1.0
        order[..., 8] = 0.5
        m.replay.add(toks, ach, {"rom_limit": 1.0, "strength_limit": 1.0}, executed=order,
                     memory_state=torch.tanh(torch.randn(B, mem, generator=g)))
    with torch.no_grad():
        ro = m.persistent_memory.readout
        ro.weight.copy_(torch.randn(ro.weight.shape, generator=g) * 0.05)
    ring = dict(vars(m.replay))
    w = RP.gather_windows(ring, [2], m.ctx_len)
    with torch.no_grad():
        f = RP._replay_forward(m, w, w["executed"][:, m.ctx_len - 1])
        m.efference.ensure_centres(f["last"], command_for_encoder(w["executed"][:, m.ctx_len - 1]))
    return m, ring


@contextmanager
def patched(*pairs):
    """Set tensors to new values for the duration, then put them back exactly."""
    saved = [t.detach().clone() for t, _ in pairs]
    with torch.no_grad():
        for t, v in pairs:
            t.copy_(v)
    try:
        yield
    finally:
        with torch.no_grad():
            for (t, _), s in zip(pairs, saved):
                t.copy_(s)


def _ring_copy(ring, *keys):
    out = dict(ring)
    for k in keys:
        out[k] = ring[k].clone()
    return out


def _live(part, key, b):
    return part["units"][key]["per_body"][b]


# ================================================================================================ attention
def test_the_batched_spread_equals_productions_recomputation_element_by_element(lite):
    from lem.model import liveness as LV
    m, ring = lite
    L = m.ctx_len
    seen = {}
    parts = {"cm": m.forward_model.blocks[0].cm_attn, "ct": m.forward_model.blocks[1].ct_attn,
             "cmb": m.forward_model.blocks[0].cmb_attn, "head": m.prediction_heads.heads["vision"].attn}

    def keep(name):
        def hook(mod, args, kwargs, out):
            seen.setdefault(name, (args, dict(kwargs)))
        return hook
    hs = [mod.register_forward_hook(keep(n), with_kwargs=True) for n, mod in parts.items()]
    try:
        w = RP.gather_windows(ring, [3, 9], L)
        with torch.no_grad():
            RP._replay_forward(m, w, w["executed"][:, L - 1])
    finally:
        for h in hs:
            h.remove()
    assert set(seen) == set(parts)
    for name, mod in parts.items():
        args, kwargs = seen[name]
        mine = MDR.attention_entropy(mod, args, kwargs, chunk=500)       # a small chunk: several chunks per call
        n = mine.shape[0]
        for e in (0, 1, n // 2, n - 1):
            prod = LV.entropy_share(LV.attention_weights(mod, args, kwargs, body=e))
            assert float(mine[e]) == pytest.approx(float(prod), abs=1e-5), (name, e)
    assert "attn_mask" in seen["ct"][1], "stage 2 carries its causal mask"


def test_a_pinned_attention_reads_near_0_and_one_whose_keys_carry_nothing_reads_the_even_spread(lite):
    m, ring = lite
    E = m.d_model
    cm = m.forward_model.blocks[0].cm_attn
    cmb = m.forward_model.blocks[1].cmb_attn
    ct = m.forward_model.blocks[0].ct_attn
    wq = cm.in_proj_weight.detach().clone()
    wq[:2 * E] *= 1000.0
    bq = cm.in_proj_bias.detach().clone()
    bq[:2 * E] *= 1000.0
    zk_cmb = cmb.in_proj_weight.detach().clone()
    zk_cmb[E:2 * E] = 0.0
    zb_cmb = cmb.in_proj_bias.detach().clone()
    zb_cmb[E:2 * E] = 0.0
    zk_ct = ct.in_proj_weight.detach().clone()
    zk_ct[E:2 * E] = 0.0
    zb_ct = ct.in_proj_bias.detach().clone()
    zb_ct[E:2 * E] = 0.0
    with patched((cm.in_proj_weight, wq), (cm.in_proj_bias, bq), (cmb.in_proj_weight, zk_cmb),
                 (cmb.in_proj_bias, zb_cmb), (ct.in_proj_weight, zk_ct), (ct.in_proj_bias, zb_ct)):
        part = MDR.checkpoint_part(m, ring, n_slots=6)
    att = part["attention"]
    assert max(att["forward_model.blocks.0.cm_attn"]["values"]) < 0.02
    assert min(att["forward_model.blocks.1.cmb_attn"]["values"]) == pytest.approx(1.0, abs=1e-5)
    L = m.ctx_len
    even_causal = sum(math.log(t + 1) for t in range(L)) / (L * math.log(L))
    for v in att["forward_model.blocks.0.ct_attn"]["values"]:
        assert v == pytest.approx(even_causal, abs=1e-5), "every row spread evenly over the keys it may read"
    untouched = MDR.checkpoint_part(m, ring, n_slots=6)["attention"]
    assert min(untouched["forward_model.blocks.0.cm_attn"]["values"]) > 0.05
    assert max(untouched["forward_model.blocks.1.cmb_attn"]["values"]) < 1.0 - 1e-4
    stages = [k for k in att if ".blocks." in k]
    assert len(stages) == 3 * len(m.forward_model.blocks)
    assert {"prediction_heads.heads.vision.attn", "efference.body_pool", "efference.gaze_pool"} <= set(att)


def test_the_reading_names_which_input_a_head_reads(lite):
    m, ring = lite
    E = m.d_model
    va = m.prediction_heads.heads["vision"].attn
    ta = m.prediction_heads.heads["tactile"].attn
    pinned = va.in_proj_weight.detach().clone()
    pinned[:2 * E] *= 1000.0
    flat = ta.in_proj_weight.detach().clone()
    flat[E:2 * E] = 0.0
    flat_b = ta.in_proj_bias.detach().clone()
    flat_b[E:2 * E] = 0.0
    with patched((va.in_proj_weight, pinned), (ta.in_proj_weight, flat), (ta.in_proj_bias, flat_b)):
        part = MDR.checkpoint_part(m, ring, n_slots=6)
    p = MDR.pooled(part, [2])
    v, t = p["attention|prediction_heads.heads.vision.attn"], p["attention|prediction_heads.heads.tactile.attn"]
    assert v["value"] < 0.02 and v["beside"]["reads_most"] in ring["mods"]
    row = MDR.compare_parts(part, part, {})["rows"]["attention|prediction_heads.heads.vision.attn"]
    assert row["reads_most"][0][0] == v["beside"]["reads_most"]
    assert "the input it reads most: %s" % v["beside"]["reads_most"] in MDR.table(MDR.compare_parts(part, part, {}))
    shares = np.asarray(part["attention"]["prediction_heads.heads.vision.attn"]["key_share"])
    H = va.num_heads
    assert shares.shape[1] == len(ring["mods"]) and np.allclose(shares.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(shares * H, np.round(shares * H), atol=1e-3), "each attention head reads one input"
    assert v["beside"]["its_weight"] == pytest.approx(shares[np.asarray(part["attention"]["prediction_heads.heads.vision.attn"]["bodies"]) != 2].mean(axis=0).max())
    assert t["value"] == pytest.approx(1.0, abs=1e-5) and t["beside"]["its_weight"] == pytest.approx(1 / len(ring["mods"]), abs=1e-5)
    assert part["attention"]["forward_model.blocks.0.cmb_attn"]["key_share"] is None, "rows of 24 keys keep no shares"
    assert part["attention"]["forward_model.blocks.0.ct_attn"]["keys"] is None, "stage 2 reads moments, not senses"
    assert "beside" not in p["attention|forward_model.blocks.0.ct_attn"]


# ================================================================================================= units
def test_a_body_whose_moments_are_all_the_same_has_no_live_unit_and_body_2_leaves_the_pooled_reading(lite):
    m, ring = lite
    same = _ring_copy(ring, "tokens", "achieved", "executed", "memory_states", "rom")
    for k in ("tokens", "achieved", "executed", "memory_states", "rom"):
        same[k][2] = same[k][2, 0:1].expand_as(same[k][2]).clone()
    part = MDR.checkpoint_part(m, same, n_slots=6)
    groups = {u["group"] for u in part["units"].values()}
    assert {"forward_model.blocks.0", "forward_model.blocks.1", "prediction_heads.heads.vision", "efference.body_mlp",
            "tenure_heads.pose", "motor_controller", "persistent_memory.outcome"} <= groups
    for key, u in part["units"].items():
        assert u["per_body"][2]["live"] == 0, key
    p_all, p_wo = MDR.pooled(part), MDR.pooled(part, [2])
    for grp in groups:
        assert p_all["units|" + grp]["value"] < p_wo["units|" + grp]["value"], grp
        live0 = sum(u["per_body"][0]["live"] for u in part["units"].values() if u["group"] == grp)
        total = sum(u["units"] for u in part["units"].values() if u["group"] == grp)
        assert live0 / total > 0.5, grp
        kept = sum(u["per_body"][b]["live"] for u in part["units"].values() if u["group"] == grp for b in (0, 1, 3))
        assert p_wo["units|" + grp]["value"] == pytest.approx(kept / (3 * total))
    table = MDR.per_body(part)
    assert [r["body"] for r in table] == [0, 1, 2, 3]
    assert table[2]["units|forward_model.blocks.0"] == 0.0
    # each attention reading belongs to its own window: body 2's windows are one window repeated
    for key, rec in part["attention"].items():
        v = np.asarray(rec["values"])
        b = np.asarray(rec["bodies"])
        assert np.ptp(v[b == 2]) < 1e-6, key
        if ".blocks." in key:
            assert np.ptp(v[b == 0]) > 1e-4, key


def test_a_blocks_units_are_read_at_the_windows_newest_moment(lite):
    m, ring = lite
    L, n_mod = m.ctx_len, len(ring["mods"])
    full = {}
    lin = m.forward_model.blocks[1].ffn[0]
    def keep(mod, inp, out):                   # returns None: the layer's output is left as it is
        full.setdefault("out", out.detach().clone())
    h = lin.register_forward_hook(keep)
    cap = MDR.Capture(m, n_mod)
    try:
        w = RP.gather_windows(ring, [3, 9], L)
        with torch.no_grad():
            inp = RP._replay_inputs(m, w, w["executed"][:, L - 1])
            cap.begin()
            m.forward_model(inp["cur"], inp["ctx"])
            cap.end(w["tokens"].shape[0], w["same_episode"], w["body"])
    finally:
        h.remove()
        cap.detach()
    got, bodies = cap.units["forward_model.blocks.1.ffn.0"][0]
    assert got.shape[1] == n_mod
    assert torch.equal(got, full["out"][w["same_episode"]][:, -n_mod:, :])
    assert torch.equal(bodies, w["body"][w["same_episode"]])


def test_units_planted_dead_read_dead_under_both_rules(lite):
    m, ring = lite
    lin = m.prediction_heads.heads["vision"].mlp[0]
    U = lin.out_features
    b = lin.bias.detach().clone()
    b[: U // 2] = -1e3
    with patched((lin.bias, b)):
        part = MDR.checkpoint_part(m, ring, n_slots=6)
    for body in range(4):
        r = _live(part, "prediction_heads.heads.vision.mlp.0", body)
        assert r["live"] <= U - U // 2 and r["live_line"] <= U - U // 2
    assert _live(part, "prediction_heads.heads.audio.mlp.0", 0)["live"] > U // 2, "the other heads are untouched"


def test_units_held_at_one_active_value_read_dead_here_and_live_by_productions_minus_3_line(lite):
    m, ring = lite
    lin = m.prediction_heads.heads["tactile"].mlp[0]
    U = lin.out_features
    b = lin.bias.detach().clone()
    b[: U // 2] += 1e4
    with patched((lin.bias, b)):
        part = MDR.checkpoint_part(m, ring, n_slots=6)
    for body in range(4):
        r = _live(part, "prediction_heads.heads.tactile.mlp.0", body)
        assert r["live_line"] == U, "every unit rose above -3"
        assert r["live"] <= U - U // 2, "a variation of about 1 at 10,000 is under bf16's rounding there"


def test_the_capture_sees_the_output_an_earlier_hook_replaced(lite):
    m, ring = lite
    lin = m.prediction_heads.heads["audio"].mlp[0]
    h = lin.register_forward_hook(lambda mod, inp, out: out + 1e4)     # as the liveness repair's revived rows do
    try:
        part = MDR.checkpoint_part(m, ring, n_slots=6)
    finally:
        h.remove()
    r = _live(part, "prediction_heads.heads.audio.mlp.0", 0)
    assert r["live"] == 0 and r["live_line"] == lin.out_features


def test_the_floor_is_the_larger_of_the_units_own_rounding_and_the_spread_under_the_minus_3_line():
    gelu = torch.nn.GELU()
    assert MDR.dead_ceiling(gelu) == pytest.approx(0.0040497 / 2, rel=1e-3)
    assert MDR.dead_ceiling(torch.nn.ReLU()) == 0.0
    g = torch.Generator().manual_seed(0)
    N = 8
    noise = torch.randn(N, 1, 1, generator=g)
    pre = torch.cat([torch.full((N, 1, 1), 5.0),              # never changes: dead
                     noise,                                   # varies by about 1: live
                     1002.0 + 0.25 * noise,                   # varies by 0.25 midway between two bf16 steps 4 apart: dead
                     -5.0 + 0.5 * noise], dim=-1)             # stays under -3: its output moves by thousandths: dead
    bodies = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out = MDR.unit_liveness(pre, gelu, bodies, 2)
    assert out["units"] == 4 and out["positions"] == 1
    assert [b["live"] for b in out["per_body"]] == [1, 1]
    assert [b["live_line"] for b in out["per_body"]] == [3, 3], "the unit under -3 is dead by production's line too"
    post = gelu(pre)
    r = (post - post.to(torch.bfloat16).float()).pow(2).mean(dim=(0, 1)).sqrt()
    assert out["rounding_unit_max"] == pytest.approx(float(r.max()), rel=1e-6)
    assert out["units_floored_by_rounding"] == int((r > MDR.dead_ceiling(gelu)).sum())
    one = MDR.unit_liveness(pre[:5], gelu, torch.tensor([0, 0, 0, 0, 1]), 2)
    assert one["per_body"][1]["live"] is None, "one window has no spread"
    # one unit held near 100,000 (bf16 steps by 512 there) moving by about 30, among 399 that move by about 1: its
    # own rounding makes it dead, and it does not raise the floor of the others
    many = torch.cat([torch.randn(N, 1, 399, generator=g), 100000.0 + 30.0 * noise], dim=-1)
    out = MDR.unit_liveness(many, gelu, bodies, 2)
    assert [b["live"] for b in out["per_body"]] == [399, 399]
    with pytest.raises(TypeError):
        MDR.unit_liveness(pre, torch.nn.Identity(), bodies, 2)


# ================================================================================================= the goal
def test_a_memory_that_does_not_vary_is_worth_exactly_0(lite):
    m, ring = lite
    ro = m.persistent_memory.readout
    with patched((ro.weight, torch.zeros_like(ro.weight))):
        part = MDR.checkpoint_part(m, ring, n_slots=6)
    assert part["goal"]["memory"]["varying_part_rms"] == 0.0
    for sense, row in MDR.goal_values(part).items():
        assert row["rows"] > 0 and row["ln"] == 0.0 and row["ln_between"] == 0.0, sense


def _independent(m, ring, part, state=None):
    """Each sense's summed error through production's replay construction (regression_pack._replay_forward) and the
    gaze copy, with the ring's own memory states or one given state for every window."""
    from lem.model.efference import apply_gaze_copy
    from mimo_bridge.lem_model import command_for_encoder
    L = m.ctx_len
    w = RP.gather_windows(ring, part["slots"], L)
    if state is not None:
        w = dict(w, memory_state=state.expand_as(w["memory_state"]).clone())
    order = w["executed"][:, L - 1]
    with torch.no_grad():
        f = RP._replay_forward(m, w, order)
        preds = apply_gaze_copy(m.prediction_heads, f["preds"], f["last"], command_for_encoder(order.float()),
                                w["achieved"][:, L - 1], mask=f["lab"])
    out = {}
    for s in part["goal"]["senses"]:
        k = ring["mods"].index(s)
        ok = w["same_episode"] & f["lab"] & w["valid"][:, L, k] & w["valid"][:, L - 1, k]
        out[s] = float(((preds[s].float() - f["nxt"][s].float()).square().mean(-1))[ok].sum())
    return out, w, preds


def test_the_error_with_and_without_the_varying_part_is_the_replay_lessons_own(lite):
    m, ring = lite
    part = MDR.checkpoint_part(m, ring, n_slots=6)
    own, w, _ = _independent(m, ring, part)
    mean_state = w["memory_state"][w["same_episode"]].mean(dim=0)
    without, _, _ = _independent(m, ring, part, mean_state)
    sums = {k: np.asarray(v) for k, v in part["goal"]["sums"].items()}
    for si, s in enumerate(part["goal"]["senses"]):
        assert float(sums["own"][:, :, si].sum()) == pytest.approx(own[s], rel=1e-5), s
        assert float(sums["mean"][:, :, si].sum()) == pytest.approx(without[s], rel=1e-5), s
        assert own[s] != without[s], "this memory varies, so taking its variation out changes the prediction"
    assert part["goal"]["memory"]["varying_part_rms"] > 0.01


def test_a_memory_whose_variation_carries_the_next_moment_removes_the_whole_error(lite):
    m, ring = lite
    L = m.ctx_len
    probe = MDR.checkpoint_part(m, ring, n_slots=3)
    backs = probe["slots"]
    ends = [(int(ring["ptr"]) - 1 - b) % int(ring["size"]) for b in backs]
    starts = [e - L for e in ends]
    assert all(abs(a - b) > L for i, a in enumerate(ends) for b in ends[i + 1:]), "the windows do not overlap"
    assert all(s >= 0 for s in starts)
    _, _, preds = _independent(m, ring, probe)
    planted = _ring_copy(ring, "tokens")
    B = planted["tokens"].shape[0]
    for r in range(len(backs) * B):
        slot, body = divmod(r, B)
        for s in probe["goal"]["senses"]:
            planted["tokens"][body, ends[slot], ring["mods"].index(s)] = preds[s][r].to(planted["tokens"].dtype)
    part = MDR.checkpoint_part(m, planted, n_slots=3)
    for s, row in MDR.goal_values(part).items():
        assert row["rows"] > 0
        assert row["share"] > 0.999 and row["ln"] > 3.0, (s, row)


# ====================================================================================== pooling and compare
def _part(att, live, own, mean, B=4):
    """A hand-made part: one attention stage, one block's units (10 each), one sense's goal sums over 2 slots."""
    wb = [b for _ in range(2) for b in range(B)]
    return dict(version=MDR.READINGS_VERSION, n_slots=2, slots=[1, 2], bodies=B, windows=2 * B, window_body=wb,
                window_slot=[s for s in range(2) for _ in range(B)],
                attention={"forward_model.blocks.0.cm_attn": dict(values=[att[b] for b in wb], bodies=wb)},
                units={"forward_model.blocks.0.ffn.0": dict(group="forward_model.blocks.0", units=10, positions=6,
                                                            floor_layer=0.001, floor_unit_median=0.001,
                                                            per_body=[dict(windows=2, live=live[b], live_line=10,
                                                                           spread_median=0.1) for b in range(B)])},
                goal=dict(senses=["vision"], has_memory=True, memory=None,
                          sums=dict(own=[[[own[b]] for b in range(B)]] * 2, mean=[[[mean[b]] for b in range(B)]] * 2,
                                    between=[[[mean[b]] for b in range(B)]] * 2, unchanged=[[[1.0]] * B] * 2),
                          rows=[[[1]] * B] * 2), problems=[])


def test_pooling_leaves_body_2_out_and_the_lines_and_verdicts_follow_the_packs_rule():
    p = _part(att=[0.2, 0.2, 0.9, 0.2], live=[8, 8, 0, 8], own=[1.0, 1.0, 1.0, 1.0], mean=[1.0, 1.0, 5.0, 1.0])
    all8, wo = MDR.pooled(p), MDR.pooled(p, [2])
    assert all8["attention|forward_model.blocks.0.cm_attn"]["value"] == pytest.approx((0.2 * 3 + 0.9) / 4)
    assert wo["attention|forward_model.blocks.0.cm_attn"]["value"] == pytest.approx(0.2)
    assert all8["units|forward_model.blocks.0"]["value"] == pytest.approx(24 / 40)
    assert wo["units|forward_model.blocks.0"]["value"] == pytest.approx(24 / 30)
    assert wo["goal|vision"]["value"] == 0.0 and all8["goal|vision"]["value"] == pytest.approx(math.log(8 / 4))
    noise = [_part([0.2, 0.2, 0.2, a], [8, 8, 8, 8], [1.0] * 4, [1.0] * 4) for a in (0.20, 0.23, 0.18)]
    for n, a in zip(noise, (0.20, 0.23, 0.18)):
        n["attention"]["forward_model.blocks.0.cm_attn"]["values"] = [0.2, 0.2, 0.2, a] * 2
    lines = MDR.lines_from(noise)
    line = lines["attention|forward_model.blocks.0.cm_attn"]["without_body_2"]
    assert line == pytest.approx((0.2 * 2 + 0.23) / 3 - (0.2 * 2 + 0.18) / 3)
    before = _part([0.2] * 4, [5] * 4, [1.0] * 4, [1.0] * 4)
    after = _part([0.5, 0.5, 0.2, 0.5], [9, 9, 0, 9], [1.0] * 4, [2.0, 2.0, 1.0, 2.0])
    lines = {"attention|forward_model.blocks.0.cm_attn": {"without_body_2": 0.05, "all": 0.05},
             "units|forward_model.blocks.0": {"without_body_2": 0.5, "all": 0.5},
             "goal|vision": {"without_body_2": 0.01, "all": 0.01}}
    rep = MDR.compare_parts(before, after, lines)
    rows = rep["rows"]
    assert rows["attention|forward_model.blocks.0.cm_attn"]["without_body_2"]["verdict"] == "IMPROVED"
    assert rows["units|forward_model.blocks.0"]["without_body_2"]["verdict"] == "within noise", "0.4 against 0.5"
    assert rows["goal|vision"]["without_body_2"]["verdict"] == "IMPROVED"
    assert rep["improved"] == 2 and rep["regressed"] == 0
    assert rep["moved"]["goal"]["improved"] == ["goal|vision"]
    assert [k.split("|")[0] for k in rep["rows"]] == ["attention", "units", "goal"], "families in their order"
    assert MDR._order("attention|forward_model.blocks.1.ct_attn") < MDR._order("attention|forward_model.blocks.1.cmb_attn")
    assert MDR._order("attention|forward_model.blocks.2.cmb_attn") < MDR._order("attention|forward_model.blocks.10.cm_attn")
    worse = MDR.compare_parts(after, before, lines)
    assert worse["regressed"] == 2
    none = MDR.compare_parts(before, after, {})
    assert all(r["without_body_2"]["verdict"].startswith("no verdict") for r in none["rows"].values())
    text = MDR.table(rep)
    assert "forward model, block 1, " + MDR.attention_stages()["cm_attn"] in text
    assert "body 3" in text and "numbered from 0" in text and "2 improved" in text
    low = text.lower()
    for banned in ("honest", "truthful", "barely", "slightly", "significantly", "substantially", "roughly"):
        assert banned not in low, banned


def test_the_goal_interval_resamples_moments_and_brackets_a_steady_reading():
    p = _part([0.2] * 4, [8] * 4, [1.0] * 4, [2.0] * 4)
    iv = MDR.goal_interval(p, [2], draws=200)
    assert iv["vision"] == pytest.approx([math.log(2.0), math.log(2.0)])
    p["goal"]["sums"]["mean"] = [[[2.0]] * 4, [[4.0]] * 4]
    iv = MDR.goal_interval(p, [2], draws=400)
    assert iv["vision"][0] < math.log(3.0) < iv["vision"][1]


def test_the_stage_names_are_the_explainers_when_its_schema_has_them(monkeypatch):
    from mimo_bridge.explainer import schema
    mine = {"cm_attn": "stage 1 - a", "ct_attn": "stage 2 - b", "cmb_attn": "stage 3 - c"}
    monkeypatch.setattr(schema, "ATTENTION_STAGES", mine, raising=False)
    assert MDR.attention_stages() == mine
    assert MDR.part_label("forward_model.blocks.2.ct_attn") == "forward model, block 3, stage 2 - b"
    monkeypatch.delattr(schema, "ATTENTION_STAGES", raising=False)
    assert MDR.attention_stages() == {"cm_attn": "stage 1 - senses read each other",
                                      "ct_attn": "stage 2 - each sense reads its past",
                                      "cmb_attn": "stage 3 - everything reads everything"}
    assert MDR.part_label("forward_model.blocks.0") == "forward model, block 1: feed-forward network"
    assert MDR.part_label("prediction_heads.heads.vision") == "vision prediction head"


# ============================================================================================ inside the pack
def test_the_packs_compare_carries_the_readings_and_its_table_shows_them(monkeypatch, tmp_path):
    from mimo_bridge.tests.test_regression_pack import _ck, _obs     # the pack's own fakes
    before = _part([0.2] * 4, [5] * 4, [1.0] * 4, [1.0] * 4)
    after = _part([0.5] * 4, [9] * 4, [1.0] * 4, [2.0] * 4)
    cks = {"A": dict(_ck(), memory_deploy=before), "B": dict(_ck(), memory_deploy=after)}
    asked = []

    def fake(ck, **k):
        asked.append(k.get("internals"))
        return cks[ck]
    monkeypatch.setattr(RP, "checkpoint_readings", fake)
    monkeypatch.setattr(RP, "stats_records", lambda saved, lo, hi: [])
    o = _obs([("H05", "L:index", s, b) for b in range(8) for s in range(0, 6000, 30)], exposure_steps=6000)
    o.update(store_generation=1, baseline_path="x", detector_version="v", rules_version="r")
    monkeypatch.setattr(RP, "observer_state", lambda saved: o)
    b = dict(kind="checkpoint", lo=0, hi=3000, checkpoint="A", checkpoint_step=3000)
    a = dict(kind="checkpoint", lo=3000, hi=6000, checkpoint="B", checkpoint_step=6000)
    lines = {"goal|vision": {"without_body_2": 0.01, "all": 0.01}}
    rep = RP.compare(b, a, saved=tmp_path, lines={}, deploy_lines=lines)
    assert asked == [True, True]
    assert rep["memory_deploy"]["rows"]["goal|vision"]["without_body_2"]["verdict"] == "IMPROVED"
    assert set(rep["headlines"]) == set(RP.HEADLINES), "the six headlines are unchanged"
    assert "Memory deploy readings beyond vision" in RP.table(rep)
    asked.clear()
    rep = RP.compare(b, a, saved=tmp_path, lines={}, internals=False)
    assert asked == [False, False] and "memory_deploy" not in rep


def test_a_cached_checkpoint_without_the_readings_is_read_again(monkeypatch, tmp_path):
    ck = tmp_path / "run" / "checkpoints" / "ckpt_step_5000"
    ck.mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    name = RP.cache_name(ck, RP.MODEL_SLOTS, None)
    model = {"x": 1, "n_slots": RP.MODEL_SLOTS}
    (cache / name).write_text(json.dumps(dict(version=RP.PACK_VERSION, newest=None, model=model)), encoding="utf-8")

    class Loaded(Exception):
        pass

    def load(_ck):
        raise Loaded()
    monkeypatch.setattr(RP, "load_ring", load)
    assert RP.checkpoint_readings(ck, cache_dir=cache)["model"] == model, "the pack alone reads its cache"
    with pytest.raises(Loaded):
        RP.checkpoint_readings(ck, cache_dir=cache, internals=True)
    part = dict(version=MDR.READINGS_VERSION, n_slots=MDR.SLOTS, newest=None)
    (cache / name).write_text(json.dumps(dict(version=RP.PACK_VERSION, newest=None, model=model, memory_deploy=part)),
                              encoding="utf-8")
    assert RP.checkpoint_readings(ck, cache_dir=cache, internals=True)["memory_deploy"] == part
    with pytest.raises(Loaded):
        RP.checkpoint_readings(ck, cache_dir=cache, internals=True, internal_slots=MDR.SLOTS + 1)
    with pytest.raises(Loaded):
        RP.checkpoint_readings(ck, cache_dir=cache, internals=True, n_slots=RP.MODEL_SLOTS + 1)


# ============================================================================== the scale survey's construction
def _runtime_state(m, folder, same_tokens=False):
    """A runtime_state.pt in the layout the scale survey reads: the self history's recorded raw tokens (the context of
    each event after the first, then the newest) and the persistent memory's event history, 4 bodies."""
    from lem.model.persistent_memory import MemoryHistory
    g = torch.Generator().manual_seed(7)
    B, n, d = 4, len(m.active), m.d_model
    frames = []
    for _ in range(m.ctx_len + 2):
        f = torch.randn(1 if same_tokens else B, n * d, generator=g).expand(B, n * d).clone()
        frames.append(f)
    hist = dict(events=[dict(inputs=dict(context=f)) for f in frames[:-1]], context=frames[-1])
    pm = m.persistent_memory
    mh = MemoryHistory(128, 0.5)
    for t in range(6):
        mh.observe(torch.randn(B, pm.input_size, generator=g), torch.full((B,), 0.5 * t), pm)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(self_history=hist, persistent_memory_history=mh.state_dict()), str(folder / "runtime_state.pt"))
    return folder


def test_the_memory_bias_per_slot_is_read_from_the_recorded_history(lite, tmp_path):
    m, _ring = lite
    ck = _runtime_state(m, tmp_path / "ck")
    ro = m.persistent_memory.readout
    d = m.d_model
    planted = torch.cat([torch.full((d,), float(k + 1)) for k in range(len(m.active))])
    with patched((ro.weight, torch.zeros_like(ro.weight)), (ro.bias, planted)):
        rec = MDR.recorded_part(m, ck)
    assert rec.get("error") is None and rec["bodies"] == 4
    for k, slot in enumerate(m.active):
        assert all(row[k] == pytest.approx(k + 1.0) for row in rec["memory_bias_rms"]), slot
    vals = MDR.scale_values(rec, [2])
    assert vals["scale|memory_bias." + m.active[0]]["value"] == pytest.approx(1.0)
    assert vals["scale|memory_bias." + m.active[0]]["better"] == -1


def test_a_head_whose_keys_carry_nothing_spreads_evenly_and_a_constant_head_reads_0_between_bodies(lite, tmp_path):
    m, _ring = lite
    ck = _runtime_state(m, tmp_path / "ck")
    E = m.d_model
    ta = m.prediction_heads.heads["tactile"].attn
    zk = ta.in_proj_weight.detach().clone()
    zk[E:2 * E] = 0.0
    zb = ta.in_proj_bias.detach().clone()
    zb[E:2 * E] = 0.0
    last = m.prediction_heads.heads["vision"].mlp[-1]
    with patched((ta.in_proj_weight, zk), (ta.in_proj_bias, zb), (last.weight, torch.zeros_like(last.weight))):
        rec = MDR.recorded_part(m, ck)
    v = MDR.scale_values(rec, [2])
    assert v["scale|top_slot.tactile"]["value"] == pytest.approx(1 / len(m.active), abs=1e-5)
    assert v["scale|top_slot.tactile"]["beside"]["without_memory_bias"] == pytest.approx(1 / len(m.active), abs=1e-5)
    assert v["scale|between_bodies.vision"]["value"] == pytest.approx(0.0, abs=1e-6), "a constant head"
    assert v["scale|between_bodies.audio"]["value"] > 1e-3, "an untouched head differs between bodies"
    assert v["scale|between_bodies.vision"]["better"] == +1 and v["scale|top_slot.tactile"]["better"] == -1
    one = MDR.scale_values(rec, [0, 1, 2])
    assert one["scale|between_bodies.audio"]["value"] is None, "one body has no spread between bodies"


def test_the_scale_family_rides_the_part_and_its_verdicts_follow_each_readings_direction(lite, tmp_path):
    m, ring = lite
    ck = _runtime_state(m, tmp_path / "ck")
    part = MDR.checkpoint_part(m, ring, n_slots=3, ckpt=ck)
    assert part["recorded"].get("error") is None
    p = MDR.pooled(part, [2])
    assert any(k.startswith("scale|memory_bias.") for k in p) and "scale|between_bodies.vision" in p
    broken = MDR.checkpoint_part(m, ring, n_slots=3, ckpt=tmp_path / "nowhere")
    assert "error" in broken["recorded"] and "attention|forward_model.blocks.0.cm_attn" in MDR.pooled(broken)
    assert MDR.verdict(-5.0, 1.0, better=-1) == "IMPROVED" and MDR.verdict(-5.0, 1.0) == "REGRESSED"
    before = dict(part, recorded=dict(part["recorded"], memory_bias_rms=[[150.0] * len(m.active)] * 4))
    after = dict(part, recorded=dict(part["recorded"], memory_bias_rms=[[5.0] * len(m.active)] * 4))
    key = "scale|memory_bias." + m.active[0]
    rep = MDR.compare_parts(before, after, {key: {"without_body_2": 10.0, "all": 10.0}})
    assert rep["rows"][key]["without_body_2"]["verdict"] == "IMPROVED", "the bias fell from 150 to 5"
    assert key in rep["moved"]["scale"]["improved"]
