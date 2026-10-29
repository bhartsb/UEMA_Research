# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The memory-bias ablation at the vision head (docs/gain_of_function_v4/vision_memory_ablation.py).

Synthetic heads, memories, orders and rows only: no checkpoint, no world. They pin what the pause job's
arms and rules mean before it runs: the script's attention is the module's; the fold keeps every weight
and the prediction when the removed bias is a constant and the attention is saturated; the literal arms
compose their inputs as declared (and "centred" is "removed" when the bias has no moment-specific part);
the eyes-held order changes only the eye command; the eye angles are matched to the live moments by pose
and order; the offset-free guard is blind to a constant offset; the re-fit restores the checkpoint's
weights and optimizer state between arms and starts the folded arms at A's prediction; and the summary
applies rules S, E, Z, G and R."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lem.model.input_conditioning import install
from lem.model.liveness import RevivedRows
from lem.model.prediction_heads import PredictionHead
from mimo_bridge.replay import ReplayMemory

REPO = Path(__file__).resolve().parents[4]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def M():
    return _load("vision_memory_ablation", "docs/gain_of_function_v4/vision_memory_ablation.py")


@pytest.fixture(scope="module")
def C():
    return _load("vision_carrier_measurements", "docs/gain_of_function_v4/vision_carrier_measurements.py")


D, N_TOK, B = 8, 3, 6
KEY = "prediction_heads__heads__vision__mlp__0"


def head_with_rows(dtype=torch.float64, seed=1):
    """A vision PredictionHead with an input scale and five revived first-layer units wired by a forward
    hook (as LivenessRepair.attach_repair wires them)."""
    torch.manual_seed(seed)
    head = PredictionHead(d_model=D, n_heads=2, hidden_expansion=1, n_mlp_layers=2).to(dtype)
    install(head.mlp[0])
    with torch.no_grad():
        head.mlp[0].input_scale.fill_(40.0)
    rows = nn.ModuleDict({KEY: RevivedRows(D, D).to(dtype)})
    row = rows[KEY]
    with torch.no_grad():
        row.mask[:5] = True
        row.centre_mean.copy_(torch.randn(D, dtype=dtype))
    row.any_py = True
    head.mlp[0].register_forward_hook(lambda mod, inp, out, r=row: r.apply(inp[0], out))
    return head, rows


# ------------------------------------------------------------------------------------ the head
def test_the_scripts_attention_is_the_modules(M):
    head, _rows = head_with_rows()
    last = 3.0 * torch.randn(B, N_TOK, D, dtype=torch.float64)
    out, w = M.head_attention(head, last, last)
    assert torch.allclose(head.mlp(out), head(last), atol=1e-10, rtol=0)
    assert torch.allclose(w.sum(-1), torch.ones(B, 2, dtype=torch.float64))
    # the module's own weights
    q = head.norm(head.query.expand(B, -1, -1))
    _o, wm = head.attn(q, last, last, need_weights=True, average_attn_weights=False)
    assert torch.allclose(w, wm[:, :, 0, :], atol=1e-12)


def saturated_case(M, bias_scale=60.0, seed=2):
    """Landing tokens of size about 1 and a memory bias that is the same constant for every moment and
    large enough that every attention head reads one token (as at the post-repair checkpoints)."""
    head, _rows = head_with_rows(seed=seed)
    torch.manual_seed(seed + 10)
    landing = torch.randn(B, N_TOK, D, dtype=torch.float64)
    c = bias_scale * torch.randn(N_TOK, D, dtype=torch.float64)
    bias = c.unsqueeze(0).expand(B, -1, -1).clone()
    return head, landing, bias, c


def test_the_fold_keeps_the_prediction_when_the_bias_is_a_constant(M):
    head, landing, bias, c = saturated_case(M, bias_scale=2000.0)
    last = landing + bias
    _o, w_a = M.head_attention(head, last, last)
    assert float((w_a - w_a.mean(0, keepdim=True)).abs().max()) < 1e-9   # the same weights for every moment
    fold = M.fold_constants(head, c, w_a.mean(0))
    base = {}
    for arm in ("A", "B", "C", "D", "Fv", "Fkv"):
        keys, values, ko, extra = M.arm_inputs(arm, last, bias, centre=c, scale=0.05, fold=fold)
        out, w = M.head_attention(head, keys, values, ko, extra)
        base[arm] = head.mlp(out)
        if arm == "Fkv":
            assert torch.allclose(w, w_a, atol=1e-12)                    # the key offsets keep every weight
    for arm in ("Fv", "Fkv"):
        assert float((base[arm] - base["A"]).abs().max()) < 1e-8
    for arm in ("B", "C", "D"):
        assert float((base[arm] - base["A"]).abs().max()) > 1e-3        # the literal arms change the head
    # with no moment-specific part, centred is removed
    assert torch.allclose(base["C"], base["B"], atol=1e-12)


def test_the_key_fold_is_exact_and_the_value_fold_needs_constant_weights(M):
    head, landing, bias, c = saturated_case(M, bias_scale=0.5, seed=3)
    last = landing + bias
    _o, w_a = M.head_attention(head, last, last)
    assert float((w_a - w_a.mean(0, keepdim=True)).abs().max()) > 1e-3    # the weights vary between moments
    fold = M.fold_constants(head, c, w_a.mean(0))
    keys, values, ko, extra = M.arm_inputs("Fkv", last, bias, fold=fold)
    out_k, w_k = M.head_attention(head, keys, values, ko, extra)
    assert torch.allclose(w_k, w_a, atol=1e-12)
    # the value fold with the mean weights is then not exact per moment: rule E has to see this
    out_a, _ = M.head_attention(head, last, last)
    assert float((out_k - out_a).abs().max()) > 1e-6
    assert float((out_k - out_a).mean(0).abs().max()) < 1e-10          # exact on average over the moments


def test_the_arms_compose_their_inputs_as_declared(M):
    last = torch.randn(2, N_TOK, D)
    bias = torch.randn(2, N_TOK, D)
    centre = bias.mean(0)
    landing = last - bias
    fold = (torch.randn(2, N_TOK), torch.randn(D))
    k, v, ko, ex = M.arm_inputs("A", last, bias)
    assert k is last and v is last and ko is None and ex is None
    k, v, _, _ = M.arm_inputs("B", last, bias)
    assert torch.allclose(k, landing) and torch.allclose(v, landing)
    k, v, _, _ = M.arm_inputs("C", last, bias, centre=centre)
    assert torch.allclose(k, landing + bias - centre)
    k, v, _, _ = M.arm_inputs("D", last, bias, scale=0.1)
    assert torch.allclose(v, landing + 0.1 * bias)
    k, v, ko, ex = M.arm_inputs("Fv", last, bias, fold=fold)
    assert k is last and torch.allclose(v, landing) and ko is None and ex is fold[1]
    k, v, ko, ex = M.arm_inputs("Fkv", last, bias, fold=fold)
    assert torch.allclose(k, landing) and ko is fold[0] and ex is fold[1]
    sc = torch.randn(N_TOK, D)
    k, v, _, _ = M.arm_inputs("Fv_const", last, bias, fold=fold, switch_constant=sc)
    assert k is last and torch.allclose(v, last - sc)
    with pytest.raises(ValueError):
        M.arm_inputs("E", last, bias)


# ---------------------------------------------------------------------------------- the memory
def test_bias_parts_and_state_saturation(M):
    const = torch.randn(1, N_TOK, D).expand(10, -1, -1)
    out = M.bias_parts(const, ["vision", "tactile", "motor"])
    assert out["moment_specific_over_constant"] == 0.0 and out["moments"] == 10
    varied = const + 0.1 * torch.randn(10, N_TOK, D)
    out = M.bias_parts(varied, ["vision", "tactile", "motor"])
    assert 0.0 < out["moment_specific_over_constant"] < 1.0 and set(out["slots"]) == {"vision", "tactile", "motor"}
    s = torch.sign(torch.randn(1, 12)).expand(5, -1) * 0.999
    sat = M.state_saturation(s)
    assert sat["share_above_0p99"] == 1.0 and sat["spread_between_moments_rms"] == 0.0


def test_memory_source_reads_where_saturation_begins(M):
    from lem.model.persistent_memory import PersistentMemory
    torch.manual_seed(4)
    pm = PersistentMemory(input_size=5, n_modalities=N_TOK, d_model=D, action_size=4, hidden_size=6)
    snap = dict(events=torch.randn(2, 7, 5), times=torch.arange(7.).repeat(2, 1), now=torch.full((2,), 7.0),
                carry=None, carry_time=None)
    out = M.memory_source(pm, snap)
    assert out["events"] == 7 and 0.0 <= out["input_projection_share_above_0p99"] <= 1.0
    with torch.no_grad():
        pm.input_projection[0].weight.mul_(1000.0)
    assert M.memory_source(pm, snap)["input_projection_share_above_0p99"] > 0.9
    assert M.memory_source(None, snap)["status"].startswith("no persistent memory")


# ------------------------------------------------------------------------------------- the eyes
def test_the_held_order_changes_only_the_eye_command(M):
    from lem.model.efference import gaze_order_features
    from mimo_bridge.lem_model import LemLite
    gi = LemLite.GAZE_IDX
    torch.manual_seed(5)
    order = torch.randn(3, gi + 3, 9)
    eyes = torch.tensor([[5.0, -3.0], [-12.0, 8.0], [0.5, 0.0]])       # degrees: azimuth, elevation
    held = M.held_order(order, eyes)
    keep = torch.ones(gi + 3, 9, dtype=torch.bool)
    keep[gi, 3:7] = False
    assert torch.equal(held[:, keep], order[:, keep])
    f = gaze_order_features(held, gi)
    assert torch.allclose(torch.rad2deg(f[:, 9:11]), eyes, atol=1e-4)


def test_eye_angles_are_matched_to_the_live_moments(M):
    Bb, Hh = 2, 3
    ach = [torch.full((Bb, Hh, 10), float(k)) for k in range(4)]
    mot = [torch.full((Bb, Hh, 9), 10.0 + k) for k in range(3)]
    hist = dict(enc_history_raw=[{} for _ in range(4)], ach_history=ach, motor_history=mot)
    eff = [dict(ach=ach[k].clone(), order=mot[k].clone(), eye=torch.full((Bb, 2), 100.0 + k),
                hm=torch.full((Bb, 6), float(k)), step=500 + k) for k in (2, 0, 1)]      # any order
    data = {"_eff_hist": eff, "_eff_eye_prev": torch.full((Bb, 2), 103.0)}
    eyes, hms, notes = M.eye_track(data, hist)
    assert [float(e[0, 0]) for e in eyes] == [100.0, 101.0, 102.0, 103.0]
    assert float(hms[1][0, 0]) == 1.0 and all(n["matched"] for n in notes)
    # an entry whose pose differs from the history's is not used
    eff[1]["ach"] = eff[1]["ach"] + 1.0
    eyes, hms, notes = M.eye_track(data, hist)
    assert eyes[0] is None and 0 not in hms and not notes[0]["matched"]


# ------------------------------------------------------------------------------------- the guard
def test_the_offset_free_benefit_ignores_a_constant_offset(M):
    torch.manual_seed(6)
    n, d = 12, 5
    bodies = [k % 4 for k in range(n)]
    target = torch.randn(n, d, dtype=torch.float64)
    act = target + 0.1 * torch.randn(n, d, dtype=torch.float64)
    hold = target + 0.3 * torch.randn(n, d, dtype=torch.float64)
    g0 = M.eye_guard(act, hold, target, act - target, bodies, list(range(n)), bodies)
    off = torch.full((1, d), 2.0, dtype=torch.float64)
    g1 = M.eye_guard(act + off, hold + off, target, act + off - target, bodies, list(range(n)), bodies)
    b_raw0, b_raw1 = M.benefit(g0["err_act"], g0["err_hold"]), M.benefit(g1["err_act"], g1["err_hold"])
    assert abs(b_raw1 - b_raw0) > 0.1                          # the raw benefit is swamped by the offset
    assert M.benefit(g1["err_act_offset_free"], g1["err_hold_offset_free"]) == pytest.approx(
        M.benefit(g0["err_act_offset_free"], g0["err_hold_offset_free"]), abs=1e-9)
    # leave one body out: each row's offset is the other bodies' mean residual
    r = torch.arange(8.).reshape(8, 1)
    o = M.loo_offset(r, [0, 0, 1, 1, 2, 2, 3, 3])
    assert float(o[0]) == pytest.approx(float(r[2:].mean()))
    # the response's direction and size
    m = M.response_match(g1["response"], g0["response"])
    assert m["cosine"] == pytest.approx(1.0) and m["size_ratio"] == pytest.approx(1.0)
    # pooled over checkpoints: the sums, not a median of per-checkpoint cosines
    half = M.response_match(0.5 * g0["response"], g0["response"])
    flip = M.response_match(-g0["response"], g0["response"])
    pooled = M.pooled_response_match([m, half, flip])
    assert pooled["cosine"] == pytest.approx(0.5 / math.sqrt(2.25 * 3.0), rel=1e-9)
    assert pooled["size_ratio"] == pytest.approx(math.sqrt(2.25 / 3.0), rel=1e-9) and pooled["checkpoints"] == 3


def test_the_bootstraps_pool_by_cluster(M):
    rows = [("a", 1.0, 2.0, 1.0, 2.0), ("b", 1.0, 2.0, 1.0, 2.0)]
    out = M.cluster_bootstrap_diff(rows, draws=50)
    assert out["point"] == 0.0 and out["p05"] == 0.0 and out["p95"] == 0.0
    worse = [("a", 1.5, 2.0, 1.0, 2.0), ("b", 1.6, 2.0, 1.0, 2.0), ("c", 1.4, 2.0, 1.0, 2.0)]
    out = M.cluster_bootstrap_diff(worse, draws=200)
    assert out["point"] == pytest.approx(-0.25) and out["p95"] < 0
    g3 = M.pooled_paired_bootstrap([("a", 1.0, 1.5), ("b", 2.0, 2.5), ("c", 1.0, 1.4)], draws=200)
    assert g3["point"] == pytest.approx(0.4667, abs=1e-3) and g3["p05"] > 0


# -------------------------------------------------------------------------------------- the draw
def test_the_draw_bodies_follow_the_carriers_draw(M, C):
    mods, d_tok, handles, size, envs = ["vision", "motor"], 3, 2, 20, 3
    r = ReplayMemory(mods, d_tok, n_handles=handles, size_per_env=size, n_envs=envs, memory_size=4)
    for k in range(25):
        tok = {"vision": torch.stack([torch.full((1, d_tok), 100.0 * e + k) for e in range(envs)]),
               "motor": torch.zeros(envs, 1, d_tok)}
        r.add(tok, torch.zeros(envs, handles, 10), {"rom_limit": 1.0, "strength_limit": 1.0},
              executed=torch.zeros(envs, handles, 9), memory_state=torch.zeros(envs, 4), action_contract=1)
    w, ages = C.draw_windows(r, 8, 4, 948, 1)
    envs_drawn = M.draw_envs(r, 8, 4, 948, 1)
    assert len(envs_drawn) == len(ages) == 8
    # each window's newest vision token is 100 x its body + the moment's index: the body is recovered
    assert [int(float(w["tokens"][k, -1, 0, 0]) // 100) for k in range(8)] == envs_drawn


# ----------------------------------------------------------------------------------- the re-fit
class FakeEff(nn.Module):
    def __init__(self, d, n_mod):
        super().__init__()
        self.lat_norm = nn.LayerNorm(d)
        self.vision_in = nn.Linear(d + 2, 8)
        self.vision_out = nn.Linear(8, d)
        self.vision_idx = 0
        self.centre_weight = 1.0
        self.register_buffer("centre_mean", torch.zeros(n_mod, d))

    def vision_delta(self, latent, order, ach):
        x = self.lat_norm(latent.float() - self.centre_mean)[:, self.vision_idx]
        o = order.reshape(order.shape[0], -1)[:, :2].float()
        return self.vision_out(torch.nn.functional.gelu(self.vision_in(torch.cat([x, o], -1))))

    def gaze(self, latent, order, ach, hm):
        return 0.01 * latent[:, :1, :6].reshape(latent.shape[0], 3, 2)


class FakeMemory(nn.Module):
    def __init__(self, s, n_mod, d):
        super().__init__()
        self.readout = nn.Linear(s, n_mod * d)
        self.n_mod, self.d = n_mod, d

    def bias(self, state):
        return self.readout(state).reshape(-1, self.n_mod, self.d)


class FakeReg:
    @staticmethod
    def penalty(preds):
        return preds["vision"].square().mean()


def fake_model(seed=7):
    torch.manual_seed(seed)
    head, rows = head_with_rows(dtype=torch.float32, seed=seed)
    eff = FakeEff(D, N_TOK)
    pm = FakeMemory(4, N_TOK, D)
    with torch.no_grad():
        pm.readout.weight.mul_(1000.0)        # a large bias, as production's: one token per attention head
    params = list(head.parameters()) + list(rows.parameters()) + list(eff.parameters()) + list(pm.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.01},
                             {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
                            lr=3e-4, betas=(0.9, 0.999))

    def clip():
        return float(torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], 1.0))
    model = SimpleNamespace(prediction_heads=SimpleNamespace(heads={"vision": head}, residual=True),
                            liveness=SimpleNamespace(rows=rows), efference=eff, persistent_memory=pm,
                            latent_regularizer=FakeReg(), pred_mods=["vision", "tactile"], optimizer=opt,
                            grad_clip=1.0, _clip_training_gradients=clip, params=params)
    # one production-like update so the optimizer holds saved moments
    loss = sum(p.square().sum() for p in params) * 1e-6
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return model


def fake_set(model, n=16, seed=8):
    torch.manual_seed(seed)
    state = torch.sign(torch.randn(1, 4)).expand(n, -1).clone()        # saturated and the same for every moment
    with torch.no_grad():
        bias = model.persistent_memory.bias(state)
    landing = torch.randn(n, N_TOK, D)
    current = torch.randn(n, D)
    body = [k % 8 for k in range(n)]
    has_hold = torch.tensor([k < 8 for k in range(n)])
    return dict(landing=landing, bias=bias, current=current, target=current + 0.2 * torch.randn(n, D),
                order=torch.randn(n, 2, 9), order_hold=torch.randn(n, 2, 9), ach=torch.zeros(n, 2, 10),
                mask=torch.ones(n, dtype=torch.bool), landing_hold=landing + 0.05 * torch.randn(n, N_TOK, D),
                has_hold=has_hold, has_eye=has_hold.clone(), eye0=torch.zeros(n, 2), eye1=torch.ones(n, 2),
                hm=torch.zeros(n, 6), state=state, cluster=["1:%d" % b for b in body], body=body,
                kind=["live"] * n, transition=[0] * n)


def test_the_refit_restores_the_checkpoint_and_starts_folded_arms_at_A(M):
    model = fake_model()
    s = fake_set(model)
    params = [p for g in M.refit_parameters(model, "A").values() for p in g]
    before = [p.detach().clone() for p in params]
    opt_before = {k: v.clone() for k, v in model.optimizer.state[params[0]].items() if torch.is_tensor(v)}
    out = M.refit_all(model, {"live": s, "draw": None}, updates=40)
    runs = out["runs"]
    # one fixed factor per clipping group and fold, from arm A's first gradient (every group here is the
    # core: the fake model has no liveness module, and its copy layers are in the efference group)
    assert set(out["clip_factors"]) == {0, 1}
    for k in (0, 1):
        f = out["clip_factors"][k]
        assert 0.0 < f["core"]["factor"] <= 1.0 and f["core"]["norm"] > 0.0
        assert f["efference"]["norm"] > 0.0 and f["liveness"]["norm"] == 0.0
    assert set(runs) == {"%s|%s|%d" % (c, a, k) for c, arms in (("saved", M.ARMS_REFIT), ("fresh", M.ARMS_FRESH))
                         for a in arms for k in range(2)}
    for key, run in runs.items():
        assert "error" not in run, run.get("error")
        assert [t["update"] for t in run["trace"]] == [0, 20, 40]
    for k in range(2):
        a0 = runs["saved|A|%d" % k]["trace"][0]["ratio"]
        for arm in ("Fv", "Fkv", "Fv_const"):
            assert runs["saved|%s|%d" % (arm, k)]["trace"][0]["ratio"] == pytest.approx(a0, rel=1e-4)
        assert runs["saved|B|%d" % k]["trace"][0]["ratio"] != pytest.approx(a0, rel=1e-3)
        first = runs["saved|A|%d" % k]["first_update"]
        assert {"attention", "mlp", "copy", "readout", "all"} <= set(first)
        assert "key_offset" in runs["saved|Fkv|%d" % k]["first_update"]
        assert runs["saved|A|%d" % k]["start"]["eye"]["clusters"] == runs["saved|Fv|%d" % k]["end"]["eye"]["clusters"]
    # the checkpoint's weights and optimizer state are back where they were
    for p, v in zip(params, before):
        assert torch.equal(p, v)
    for k, v in opt_before.items():
        assert torch.equal(model.optimizer.state[params[0]][k], v)
    assert all(not p.requires_grad or p.grad is None for p in params)


# ------------------------------------------------------------------------------------ the summary
def fake_row(M, step, ra=2.0, rx=1.0, s_ratio=1e-8, z_ratio=500.0, g_equal=True):
    clusters = ["%d:%d" % (step, b) for b in range(4)]
    eye = dict(clusters=clusters, err_act_offset_free=[1.0] * 4, err_hold_offset_free=[1.1] * 4)
    worse = dict(clusters=clusters, err_act_offset_free=[1.2] * 4, err_hold_offset_free=[1.1] * 4)
    fc = dict(clusters=clusters, error=[1.0] * 4)

    def run(r, arm):
        trace = [dict(update=u, ratio=(3.0 if u < 220 else r)) for u in range(0, 401, 20)]
        end = dict(eye=eye if (g_equal or arm == "A") else worse, gaze_forecast=fc)
        return dict(trace=trace, start=dict(eye=eye, gaze_forecast=fc), end=end)
    runs = {}
    for k in range(2):
        runs["saved|A|%d" % k] = run(ra, "A")
        for arm in ("Fv", "Fkv", "Fv_const", "B"):
            runs["saved|%s|%d" % (arm, k)] = run(rx, arm)
        runs["fresh|A|%d" % k] = run(ra, "A")
        runs["fresh|Fv|%d" % k] = run(rx, "Fv")
    frozen_rows = dict(clusters=clusters, err_act_offset_free=[1.0] * 4, err_hold_offset_free=[1.1] * 4)
    fg = {arm: dict(rows=frozen_rows, response_vs_A=dict(cosine=1.0, size_ratio=1.0, dot=4.0, xx=4.0, aa=4.0))
          for arm in M.ARMS_FROZEN}
    both = {arm: dict(offset_error=(0.9 if arm in ("B", "C", "D") else 0.1), error=1.0) for arm in M.ARMS_FROZEN}
    preds = {arm: dict(err=[z_ratio if arm in ("B", "C", "D") else 2.0] * 4, pers=[1.0] * 4, cluster=clusters)
             for arm in M.ARMS_FROZEN}
    ex = dict(footprint_max_abs=1e-6, largest_value=5.0, Fv=1e-6, Fkv=2e-6)
    return dict(step=step, bias_from_states=dict(moment_specific_over_constant=s_ratio), refit=dict(runs=runs),
                frozen=dict(exactness=dict(live=ex, draw=ex), both=both), frozen_guard=fg, predictions=preds)


def test_the_summary_applies_the_declared_rules(M):
    steps = (380550, 385975, 389850, 392950, 397125, 399225)
    rows = [fake_row(M, s) for s in steps] + [fake_row(M, 344400, s_ratio=0.12)]
    out = M.summarize(rows)
    assert out["references"] == [344400] and 344400 not in out["read"]
    assert out["S"]["met"] and out["E"]["Fv"]["met"] and out["E"]["Fkv"]["met"]
    assert not any(out["Z"][a]["met"] for a in ("B", "C", "D"))
    assert out["G_frozen"]["Fv"]["G1"]["holds"] and out["G_frozen"]["Fv"]["G2"]["holds"]
    assert out["R"]["Fv"]["met"] and out["R"]["Fv"]["outcome"] == "Fv settles, A does not"
    assert not out["R"]["Fv_const"]["met"]                      # a reading beside the rule, never a licence
    # A settles too: "both settle"
    both = M.summarize([fake_row(M, s, ra=1.1) for s in steps])
    assert not both["R"]["Fv"]["met"] and both["R"]["Fv"]["outcome"] == "both settle"
    # the candidate does not settle
    no = M.summarize([fake_row(M, s, rx=1.8) for s in steps])
    assert not no["R"]["Fv"]["met"] and no["R"]["Fv"]["outcome"] == "Fv does not settle"
    # the guard fails after the re-fit: not licensed although the offset settles
    g = M.summarize([fake_row(M, s, g_equal=False) for s in steps])
    assert not g["R"]["Fv"]["met"] and not g["R"]["Fv"]["guard"]["holds"]
    # a moment-specific bias fails S; a frozen arm that removes the offset meets Z
    s = M.summarize([fake_row(M, st, s_ratio=0.01, z_ratio=1.1) for st in steps])
    assert not s["S"]["met"]
    # Z needs the offset's share too (0.9 in these rows): the ratio alone does not meet it
    assert not s["Z"]["B"]["met"]


def test_the_records_mode_reads_the_readout_alone(M, tmp_path):
    from safetensors.torch import save_file
    ck = tmp_path / "ckpt_step_123"
    ck.mkdir()
    torch.manual_seed(9)
    W, b = torch.randn(N_TOK * D, 4), torch.randn(N_TOK * D)
    save_file({"persistent_memory.readout.weight": W, "persistent_memory.readout.bias": b,
               "other.weight": torch.zeros(2)}, str(ck / "model.safetensors"))
    state = torch.sign(torch.randn(1, 4)).expand(6, -1).clone()
    f = tmp_path / "windows.pt"
    torch.save(dict(replay_window=dict(memory_state=state), replay_mods=["vision", "tactile", "motor"],
                    checkpoint=str(ck), source_step=123), f)
    out = tmp_path / "records.json"
    assert M.main(["records", "--output", str(out), str(f)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))["files"][str(f)]
    assert rec["step"] == 123 and rec["saturation"]["share_above_0p99"] == 1.0
    assert rec["bias"]["moment_specific_over_constant"] == pytest.approx(0.0, abs=1e-12)
    assert rec["bias"]["constant_rms"] == pytest.approx(float((state[0] @ W.T + b).square().mean().sqrt()), rel=1e-6)
