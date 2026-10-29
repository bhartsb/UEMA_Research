# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The next vision-regression measurements (docs/gain_of_function_v4/vision_carrier_measurements.py).

Synthetic rings, histories and heads only: no checkpoint, no world. They pin what the pause job's
readings mean before it runs: a window cut from the ring ends at the named moment and carries the
step-from moment's memory state; the ring's moments are matched to the live histories (and a
shifted history is found at its offset); the live construction is the delayed next-frame lesson's;
the pooled ratio and its cluster bootstrap; the unbiased offset; the attention logit split adds up
to the module's own weights; and the lever decomposition is exact, agrees with autograd and with a
finite difference of the top singular value, and has no lever part without a memory bias."""
import importlib.util
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
    return _load("vision_carrier_measurements", "docs/gain_of_function_v4/vision_carrier_measurements.py")


@pytest.fixture(scope="module")
def A():
    return _load("audit_visual_regression_2026_09_24", "docs/gain_of_function_v3/audit_visual_regression_2026_09_24.py")


# ------------------------------------------------------------------------------------------ ring
MODS = ["vision", "motor"]
D_TOK, HANDLES, SIZE, ENVS, MEM = 3, 2, 20, 2, 4


def filled_ring(adds=25, reset_at=None):
    """A real ReplayMemory whose moment k carries the value k everywhere (vision token k, motor token
    -k, achieved pose k, memory state k) and whose order passed WITH moment k is k (so it back-fills
    moment k - 1). reset_at: marks an episode reset before that add."""
    r = ReplayMemory(MODS, D_TOK, n_handles=HANDLES, size_per_env=SIZE, n_envs=ENVS, memory_size=MEM)
    for k in range(adds):
        if reset_at is not None and k == reset_at:
            r.mark_reset()
        tok = {"vision": torch.full((ENVS, 1, D_TOK), float(k)), "motor": torch.full((ENVS, 1, D_TOK), -float(k))}
        r.add(tok, torch.full((ENVS, HANDLES, 10), float(k)), {"rom_limit": 1.0, "strength_limit": 1.0},
              executed=torch.full((ENVS, HANDLES, 9), float(k)), memory_state=torch.full((ENVS, MEM), float(k)))
    return r


def test_window_at_ends_at_the_named_moment(M):
    r = filled_ring(25)                         # the ring (20 slots) has wrapped; the newest moment is 24
    w = M.window_at(r, back=1, ctx_len=4)       # 5 moments ending one before the newest: 19..23
    assert w["tokens"].shape == (ENVS, 5, 2, D_TOK)
    assert torch.equal(w["tokens"][:, :, 0, 0], torch.tensor([[19., 20., 21., 22., 23.]] * ENVS))
    # the step-from moment (22) carries its memory state; its outgoing order (passed with 23) is back-filled
    assert torch.equal(w["memory_state"], torch.full((ENVS, MEM), 22.))
    assert torch.equal(w["executed"][:, -2, 0, 0], torch.full((ENVS,), 23.))
    assert bool(w["has_executed"][:, -2].all()) and bool(w["same_episode"].all())
    newest = M.window_at(r, back=0, ctx_len=4)
    assert torch.equal(newest["tokens"][:, -1, 0, 0], torch.full((ENVS,), 24.))
    assert not bool(newest["has_executed"][:, -1].any())      # the newest moment's order is not known yet
    # the same keys the ring's own sampler returns, plus the reset flag
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        sampled = r.sample_windows(2, 4, "cpu")
    assert set(w) == set(sampled) | {"same_episode"}
    # a window crossing an episode reset is flagged
    crossed = M.window_at(filled_ring(25, reset_at=22), back=0, ctx_len=4)
    assert not bool(crossed["same_episode"].any())
    with pytest.raises(ValueError):
        M.window_at(filled_ring(4), back=0, ctx_len=4)


def histories_from(r, shift=0):
    """Live histories of the last four moments as lem_model.step keeps them, read back from the ring
    `shift` moments earlier than the newest (shift 0 = aligned)."""
    last = 24 - shift
    ks = [last - 3 + j for j in range(4)]
    raw = [{"vision": torch.full((ENVS, D_TOK), float(k)), "motor": torch.zeros(ENVS, D_TOK)} for k in ks]
    return dict(enc_history_raw=raw, enc_history=raw,
                ctx_history=[torch.zeros(ENVS, 3 * 2, D_TOK) for _ in ks],
                memory_bias_history=[torch.full((ENVS, 2, D_TOK), float(k)) for k in ks],
                ach_history=[torch.full((ENVS, HANDLES, 10), float(k)) for k in ks],
                # motor_history[j] is moment j's OUTGOING order, the one passed with moment j + 1
                motor_history=[torch.full((ENVS, HANDLES, 9), float(k + 1)) for k in ks[:3]],
                valid_history=[{"vision": "real", "motor": "real"} for _ in ks])


def test_alignment_matches_the_histories_and_finds_an_offset(M):
    r = filled_ring(25)
    ident = lambda order: order                                 # noqa: E731 (encoder units = stored units here)
    off, match, readings = M.find_offset(r, histories_from(r), MODS, ident)
    assert off == 0 and bool(match.all())
    assert readings[0]["worst"] == 0.0 and readings[0]["bodies_matching"] == ENVS
    off, match, readings = M.find_offset(r, histories_from(r, shift=1), MODS, ident)
    assert off == 1 and readings[0]["worst"] > M.ALIGN_LINE
    # one body's history differs (a reset, say): half the bodies still match and only they are used
    h = histories_from(r)
    h["ach_history"][2] = h["ach_history"][2].clone()
    h["ach_history"][2][1] += 5.0
    off, match, _ = M.find_offset(r, h, MODS, ident)
    assert off == 0 and match.tolist() == [True, False]


def test_histories_refuse_a_companion_without_them(M):
    with pytest.raises(ValueError):
        M.histories({"enc_history_raw": [1, 2], "ctx_history": [1, 2], "ach_history": [1, 2], "motor_history": []})
    h = M.histories({"enc_history_raw": (1, 2), "ctx_history": (1, 2), "ach_history": (1, 2), "motor_history": (1,)})
    assert h["enc_history_raw"] == [1, 2] and h["valid_history"] is None


def test_swap_slots_takes_only_the_named_modalities(M):
    B, T, n, d = 2, 3, 4, 5
    live = torch.zeros(B, T * n, d)
    other = torch.ones(B, T * n, d)
    out = M.swap_slots(live, other, n, [1, 3]).reshape(B, T, n, d)
    assert bool((out[:, :, [1, 3]] == 1).all()) and bool((out[:, :, [0, 2]] == 0).all())
    assert bool((live == 0).all())                               # the live context is not modified


def test_live_inputs_rebuild_the_delayed_lesson(M):
    r = filled_ring(25)
    h = histories_from(r)
    seen = {}

    class Registry:
        def encode(self, frames, mask):
            seen["x"] = frames["motor"].clone()
            return {"motor": frames["motor"].sum(1)[:, :D_TOK]}
    model = SimpleNamespace(d_model=D_TOK, active=["vision", "motor"], encoder_registry=Registry())
    inp = M.live_inputs(model, h, 1, motor_features=10)
    # the motor token is rebuilt from [achieved pose of moment 1, its outgoing order]
    assert torch.equal(seen["x"], torch.cat([h["ach_history"][1], h["motor_history"][1]], dim=-1))
    assert torch.equal(inp["target"]["vision"], h["enc_history_raw"][2]["vision"])
    assert torch.equal(inp["cur"]["vision"].squeeze(1), h["enc_history_raw"][1]["vision"])
    assert inp["ctx"] is not None and torch.equal(inp["bias"], h["memory_bias_history"][1])
    assert torch.equal(inp["order"], h["motor_history"][1]) and bool(inp["valid"].all())
    h["valid_history"][2] = {"vision": "stale"}
    assert not bool(M.live_inputs(model, h, 1, motor_features=10)["valid"].any())


def test_take_keeps_the_named_rows_everywhere(M):
    inp = {"mask": torch.ones(4, dtype=torch.bool), "cur": {"vision": torch.arange(4.).reshape(4, 1, 1)},
           "ctx": torch.arange(8.).reshape(4, 2), "note": "kept"}
    out = M.take(inp, torch.tensor([1, 3]))
    assert out["cur"]["vision"].flatten().tolist() == [1.0, 3.0]
    assert out["ctx"].shape == (2, 2) and out["note"] == "kept"


# ------------------------------------------------------------------------------------- summaries
def test_pooled_ratio_and_its_cluster_bootstrap(M):
    num = [3.0, 3.0, 1.5, 1.5]
    den = [2.0, 2.0, 1.0, 1.0]
    out = M.pooled_ratio(num, den, ["a", "a", "b", "b"], draws=500)
    assert out["point"] == pytest.approx(1.5) and out["p05"] == pytest.approx(1.5) and out["clusters"] == 2
    same = M.pooled_ratio([1.0, 2.0], [1.0, 2.0], ["a", "b"], draws=200)
    assert same["point"] == same["p05"] == same["p95"] == 1.0


def test_the_unbiased_offset_finds_a_constant_error(M):
    torch.manual_seed(5)
    # 200 clusters of 3 rows: the estimate's spread is about 0.006 around 0.09 (and 0.0006 around 0)
    n_clusters, rows_each, d = 200, 3, 16
    c = torch.full((d,), 0.3)
    noise = torch.randn(n_clusters * rows_each, d)
    ids = ["%d" % (k // rows_each) for k in range(n_clusters * rows_each)]
    assert M.unbiased_offset_sq(c + noise, ids) == pytest.approx(0.09, abs=0.02)
    assert abs(M.unbiased_offset_sq(noise, ids)) < 0.005
    # anatomy: a constant error on every moment is (almost) all offset
    n = 30
    cur = torch.randn(n, d)
    rows = [dict(total=cur + 0.5, base=cur + 2.0, current=cur, target=cur)]
    an = M.anatomy(rows, ["%d" % k for k in range(n)])
    assert an["offset_share_of_error"] == pytest.approx(1.0, abs=1e-6)
    assert an["head_change_rms"] == pytest.approx(2.0) and an["net_change_rms"] == pytest.approx(0.5)


# ------------------------------------------------------------------------ attention and the lever
D, N_TOK, BODIES = 8, 3, 6


def tiny_model(seed=1, bias_scale=5.0, dominant=3.0):
    """A vision PredictionHead with an existing input scale, five revived first-layer units kept in a
    real nn.ModuleDict (wired by a forward hook, as LivenessRepair.attach_repair wires them), one
    dominant direction in its value-to-output map, and latents = landing tokens + a memory bias."""
    torch.manual_seed(seed)
    head = PredictionHead(d_model=D, n_heads=2, hidden_expansion=1, n_mlp_layers=2).double()
    first = head.mlp[0]
    install(first)
    with torch.no_grad():
        first.input_scale.fill_(40.0)
        a, b = torch.randn(D, dtype=torch.float64), torch.randn(D, dtype=torch.float64)
        head.attn.in_proj_weight[2 * D:] += dominant * torch.outer(a / a.norm(), b / b.norm())
    rows = nn.ModuleDict({"prediction_heads__heads__vision__mlp__0": RevivedRows(D, D).double()})
    row = rows["prediction_heads__heads__vision__mlp__0"]
    with torch.no_grad():
        row.mask[:5] = True
        row.centre_mean.copy_(torch.randn(D, dtype=torch.float64))
    row.any_py = True
    first.register_forward_hook(lambda mod, inp, out, r=row: r.apply(inp[0], out))
    model = SimpleNamespace(prediction_heads=SimpleNamespace(heads={"vision": head}, residual=True),
                            liveness=SimpleNamespace(rows=rows), efference=None,
                            active=["vision", "tactile", "motor"])
    landing = torch.randn(BODIES, N_TOK, D, dtype=torch.float64)
    bias = bias_scale * torch.randn(1, N_TOK, D, dtype=torch.float64).expand(BODIES, -1, -1).clone()
    inp = {"bias": bias, "cur": {"vision": torch.randn(BODIES, 1, D, dtype=torch.float64)},
           "target": {"vision": torch.randn(BODIES, D, dtype=torch.float64)},
           "mask": torch.ones(BODIES, dtype=torch.bool), "order": None, "ach": None}
    return model, head, landing + bias, inp


def test_attention_split_adds_up_to_the_modules_weights(M):
    model, head, last, inp = tiny_model()
    out = M.attention_split(model, inp, last)
    assert out["weights_check_max_abs"] < 1e-5
    assert sum(out["token_weights"].values()) == pytest.approx(1.0, abs=1e-5)
    assert out["winner"] == max(out["token_weights"], key=out["token_weights"].get)
    assert 0.0 <= out["spread"] <= 1.0
    # with no memory bias, no part of any margin comes from it
    model0, _, last0, inp0 = tiny_model(bias_scale=0.0)
    assert M.attention_split(model0, inp0, last0)["bias_share_of_margin_median"] == pytest.approx(0.0, abs=1e-9)


def top_gain(head):
    attn = head.attn
    return float(torch.linalg.svdvals(attn.out_proj.weight @ attn.in_proj_weight[2 * D:])[0])


def test_the_lever_decomposition_is_exact(M, A):
    model, head, last, inp = tiny_model()
    attn = head.attn
    torch.manual_seed(7)
    P = {"in": torch.rand_like(attn.in_proj_weight) + 0.5, "out": torch.rand_like(attn.out_proj.weight) + 0.5}
    w_in, w_out = attn.in_proj_weight.detach().clone(), attn.out_proj.weight.detach().clone()
    for pre in (None, P):
        out = M.lever_decomposition(A, model, [(inp, last)], pre, shuffles=20)
        assert torch.equal(attn.in_proj_weight, w_in) and torch.equal(attn.out_proj.weight, w_out)
        assert out["attention_reconstruction_max_abs"] < 1e-10
        allr = out["routes"]["all"]
        assert allr["exactness"] < 1e-9 and allr["autograd_agreement"] < 1e-8
        assert allr["total"] == pytest.approx(allr["lever"] + allr["mean_rest"] + allr["specific"], rel=1e-9, abs=1e-15)
        # the retained unit's route and the revived units' route add up to the whole
        assert allr["total"] == pytest.approx(out["routes"]["retained_only"]["total"]
                                              + out["routes"]["revived_only"]["total"], rel=1e-8, abs=1e-15)
        assert allr["specific_shuffled_central"][0] <= allr["specific_shuffled_central"][1]
    # finite difference: step the weights by -eps * P * grad and read the top singular value
    loss_grad = {}
    attn.in_proj_weight.requires_grad_(True)
    attn.out_proj.weight.requires_grad_(True)
    q = head.norm(head.query.expand(BODIES, -1, -1))
    o, _ = attn(q, last, last)
    loss = (head.mlp(o.squeeze(1)) + inp["cur"]["vision"].squeeze(1) - inp["target"]["vision"]).square().mean()
    loss_grad["in"], loss_grad["out"] = torch.autograd.grad(loss, [attn.in_proj_weight, attn.out_proj.weight])
    attn.in_proj_weight.requires_grad_(False)
    attn.out_proj.weight.requires_grad_(False)
    eps = 1e-6
    fd = []
    for sgn in (1.0, -1.0):
        with torch.no_grad():
            attn.in_proj_weight.copy_(w_in - sgn * eps * P["in"] * loss_grad["in"])
            attn.out_proj.weight.copy_(w_out - sgn * eps * P["out"] * loss_grad["out"])
        fd.append(top_gain(head))
    with torch.no_grad():
        attn.in_proj_weight.copy_(w_in)
        attn.out_proj.weight.copy_(w_out)
    central = (fd[0] - fd[1]) / (2 * eps)
    out = M.lever_decomposition(A, model, [(inp, last)], P, routes=("all",), shuffles=0)
    assert out["routes"]["all"]["total"] == pytest.approx(central, rel=1e-5, abs=1e-9)


def test_without_a_memory_bias_there_is_no_lever(M, A):
    model, head, last, inp = tiny_model(bias_scale=0.0)
    out = M.lever_decomposition(A, model, [(inp, last)], None, routes=("all",), shuffles=0)
    assert out["routes"]["all"]["lever"] == 0.0
    # identical moments leave nothing moment-specific
    same = last[:1].expand(BODIES, -1, -1).clone()
    inp_same = dict(inp, cur={"vision": inp["cur"]["vision"][:1].expand(BODIES, -1, -1).clone()},
                    target={"vision": inp["target"]["vision"][:1].expand(BODIES, -1).clone()})
    out = M.lever_decomposition(A, model, [(inp_same, same)], None, routes=("all",), shuffles=0)
    assert out["routes"]["all"]["specific"] == pytest.approx(0.0, abs=1e-12)


def test_the_other_sensory_swap_leaves_vision_and_motor_live(M):
    n_mod, B, T, d = 4, 2, 3, 5
    model = SimpleNamespace(active=["vision", "tactile", "audio", "motor"])
    live = dict(ctx=torch.zeros(B, T * n_mod, d), valid=torch.ones(B, dtype=torch.bool), bias=None, cur={},
                target={"vision": torch.zeros(B, d)})
    rep = dict(ctx=torch.ones(B, T * n_mod, d), valid=torch.ones(B, dtype=torch.bool), bias=None, cur={},
               target={"vision": torch.ones(B, d)})
    arms = M.arms_for(model, live, rep, n_mod)
    other = arms["ctx_other_sensory_from_replay"]["ctx"].reshape(B, T, n_mod, d)
    assert bool((other[:, :, [1, 2]] == 1).all()) and bool((other[:, :, [0, 3]] == 0).all())
    vis = arms["ctx_vision_from_replay"]["ctx"].reshape(B, T, n_mod, d)
    assert bool((vis[:, :, 0] == 1).all()) and bool((vis[:, :, 1:] == 0).all())
    # every arm is graded against the live target
    assert torch.equal(arms["replay"]["target"]["vision"], live["target"]["vision"])
    with pytest.raises(ValueError):
        M.arms_for(model, live, dict(rep, ctx=torch.ones(B, 2 * n_mod, d)), n_mod)


def test_the_production_draw_records_ages_and_keeps_the_contract(M):
    r = filled_ring(25)                         # every moment's contract is 0
    assert M.draw_windows(r, 8, 4, seed=1, contract=1) == (None, [])
    r = ReplayMemory(MODS, D_TOK, n_handles=HANDLES, size_per_env=SIZE, n_envs=ENVS, memory_size=MEM)
    for k in range(25):
        tok = {"vision": torch.full((ENVS, 1, D_TOK), float(k)), "motor": torch.zeros(ENVS, 1, D_TOK)}
        r.add(tok, torch.full((ENVS, HANDLES, 10), float(k)), {"rom_limit": 1.0, "strength_limit": 1.0},
              executed=torch.full((ENVS, HANDLES, 9), float(k)), memory_state=torch.full((ENVS, MEM), float(k)),
              action_contract=1)
    w, ages = M.draw_windows(r, 8, 4, seed=1, contract=1)
    # as production's sampler: no window ends on the newest moment, none reaches past the oldest
    assert len(ages) == 8 and all(1 <= a <= 20 - 5 for a in ages)
    # each window ends `age` adds before the newest moment (24)
    assert w["tokens"][:, -1, 0, 0].tolist() == [24.0 - a for a in ages]
    w0, ages0 = M.draw_windows(r, 8, 4, seed=1, contract=None)
    assert len(ages0) == 8 and min(ages0) >= 1


def test_the_preconditioner_floors_near_silent_elements(M):
    p = nn.Parameter(torch.ones(10, 10))
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.01)
    p.grad = torch.ones(10, 10)
    opt.step()
    with torch.no_grad():
        opt.state[p]["exp_avg_sq"][0, :5] = 0.0      # five silent elements
    model = SimpleNamespace(optimizer=opt)
    out = M.adam_preconditioner(model, p, floor_percentile=10.0)
    assert out["floored_elements"] == 5 and out["lr"] == 1e-3 and out["weight_decay"] == 0.01
    assert float(out["P"].max()) < 1e3                 # 1/eps would be 1e8
    assert M.adam_preconditioner(SimpleNamespace(optimizer=opt), nn.Parameter(torch.ones(2))) is None


def test_shuffles_stay_inside_each_moment_set(M):
    gen = torch.Generator().manual_seed(3)
    for _ in range(20):
        perm = M.within_set_permutation([3, 5, 2], gen)
        assert sorted(perm[:3].tolist()) == [0, 1, 2] and sorted(perm[3:8].tolist()) == list(range(3, 8))
        assert sorted(perm[8:].tolist()) == [8, 9]


def test_the_head_input_arm_rounds_only_what_the_heads_read(M):
    B, n, d = 2, 3, 4
    latent = torch.full((B, n, d), 130.37)          # bf16 keeps steps of 1.0 between 128 and 256
    seen = {}

    class Heads:
        residual = True

        def __call__(self, last, current=None):
            seen["last"] = last.clone()
            return {"vision": last[:, 0] + current["vision"]}
    model = SimpleNamespace(forward_model=lambda cur, ctx, memory_bias=None: (None, latent, None),
                            prediction_heads=Heads())
    inp = dict(cur={"vision": torch.zeros(B, 1, d)}, ctx=None, bias=None, order=None, ach=None,
               mask=torch.ones(B, dtype=torch.bool), target={"vision": torch.zeros(B, d)})
    base, total, last = M.forward_arm(None, model, inp, "head_input_bf16")
    assert bool((seen["last"] == 130.0).all()) and bool((last == latent).all())
    assert torch.equal(total["vision"], base["vision"])          # no gaze copy attached here


def test_the_production_check_reads_productions_own_numbers(M):
    B, d = 4, 3
    target = torch.zeros(B, d)
    r = dict(err=torch.full((B,), 0.02), pers=torch.full((B,), 0.01), total=torch.full((B, d), 0.1),
             target=target)
    data = {"_nf_pred": {"vision": torch.full((B, d), 0.11)},
            "last_prediction_audit": {"vision": {"prediction_mse": 0.016, "persistence_mse": 0.01, "transitions": 4}}}
    out = M.production_check({"bf16": r}, data)
    assert out["bf16"]["error_over_production"] == pytest.approx(1.25)
    assert out["bf16"]["unchanged_over_production"] == pytest.approx(1.0)
    assert out["bf16"]["prediction_gap_over_error"] == pytest.approx(0.1, rel=1e-5)
    assert out["production"]["prediction_mse"] == 0.016


def test_the_context_ingredients_separate_the_gate_blend_from_the_embedding(M):
    B, n_mod, d, T = 2, 3, 4, 5
    emb_now = torch.randn(1, n_mod, d)
    fm = SimpleNamespace(_refresh_embedding_cache=lambda: None, _mod_emb_expanded=emb_now)
    model = SimpleNamespace(forward_model=fm, active=["vision", "tactile", "motor"], d_model=d)
    raw = [{m: torch.randn(B, d) for m in ("vision", "tactile", "motor")} for _ in range(4)]
    gated = [{m: raw[j][m] + (0.1 if m == "vision" else 0.0) for m in raw[j]} for j in range(4)]
    ctx = torch.zeros(B, T, n_mod, d)
    for jj in (1, 2, 3):                          # the context's newest moments hold gated + an older embedding
        src = 3 - jj
        for k, m in enumerate(model.active):
            ctx[:, -jj, k] = gated[src][m] + emb_now[0, k] + 0.2
    hist = dict(ctx_history=[None, None, None, ctx.reshape(B, T * n_mod, d)], enc_history=gated, enc_history_raw=raw)
    out = M.context_ingredients(model, hist)
    assert set(out) == {"vision", "tactile"}
    assert out["tactile"]["gate_blend"] == 0.0 and out["vision"]["gate_blend"] > 0.0
    for m in out:
        assert out[m]["embedding_drift"] > 0.0


def synthetic_row(step, live=2.0, replay=1.0, vision_swap=1.1, vision_half=1.2, live_bf16=3.0, head_bf16=2.05,
                  heads_bf16=2.1, fid=1.05, fid32=1.25, gap16=0.1, gap32=0.3):
    ids = ["%d:%d" % (step, b) for b in range(8)]

    def arm(v, p=1.0):
        return dict(err=[v] * 8, pers=[p] * 8, cluster=ids)
    preds = {"live": arm(live), "replay": arm(replay), "ctx_vision_from_replay": arm(vision_swap),
             "ctx_other_sensory_from_replay": arm(live), "ctx_motor_from_replay": arm(live),
             "ctx_all_from_replay": arm(replay), "bias_from_replay": arm(live), "current_from_replay": arm(live),
             "ctx_vision_newest_half_from_replay": arm(vision_half),
             "ctx_other_sensory_newest_half_from_replay": arm(live), "live_bias_fp32": arm(live),
             "live_bf16": arm(live_bf16), "replay_bf16": arm(1.5), "live_head_input_bf16": arm(head_bf16),
             "live_heads_bf16": arm(heads_bf16)}
    return {"step": step, "predictions": preds,
            "production_draw": dict(err=[1.0] * 64, pers=[1.0] * 64),
            "anatomy": {"live": dict(ratio=2.0, error=2.0, unchanged=1.0, offset_error=0.8, offset_share_of_excess=0.8),
                        "replay": dict(ratio=1.0, error=1.0, unchanged=1.0, offset_error=0.1, offset_share_of_excess=None)},
            "attention": {"live": [dict(bias_share_of_margin_median=0.9, bias_share_of_margin_all=[0.9, 0.85, 0.95])] * 3},
            "production_check": {"bf16": dict(error_over_production=fid, prediction_gap_over_error=gap16),
                                 "fp32": dict(error_over_production=fid32, prediction_gap_over_error=gap32)}}


def test_the_summary_applies_the_declared_rules(M):
    rows = [synthetic_row(s) for s in (380550, 385975, 389850, 392950, 397125, 399225)]
    rows.append(synthetic_row(344400, live=9.0))               # a reference, never pooled
    out = M.summarize(rows)
    assert out["references"] == [344400] and 344400 not in out["read"]
    assert out["pipeline_reproduces_the_frozen_audit"] is None and "void" not in out
    c1 = out["C1"]
    assert c1["met"] and c1["live_over_replay"]["point"] == pytest.approx(2.0)
    # the whole-slot swap closes ln(2/1.1)/ln 2 = 0.86, its newest half ln(2/1.2)/ln 2 = 0.74 of the gap:
    # more than half of it, so the blend (present at every age) is named
    assert c1["carriers"] == ["ctx_vision_from_replay"] and c1["causes"]["ctx_vision_from_replay"] == "the gate blend"
    assert not out["C2"]["met"] and out["C2"]["newest_over_production_draw"] == pytest.approx(1.0)
    assert out["C3"]["met"] and out["C4"]["met"] and set(out["C4"]["per_checkpoint"]) == {"389850", "392950", "397125", "399225"}
    assert out["C4"]["per_checkpoint"]["389850"] == pytest.approx(0.9)     # the median over every margin
    assert out["F"]["met"] and out["C5"]["met"] and out["C5"]["bf16_over_fp32"]["point"] == pytest.approx(1.5)
    # the head-input arm reproduces ln(2.05/2)/ln(3/2) = 0.06, the heads arm ln(2.1/2)/ln 1.5 = 0.12: neither
    assert out["C5"]["head_input_share_of_the_bf16_excess"] == pytest.approx(0.0609, abs=1e-3)
    assert out["C5"]["carrier"].startswith("the transformer")
    # a whole-slot swap whose newest half closes little is not attributed to the blend
    drift = M.summarize([synthetic_row(s, vision_half=1.9) for s in (380550, 385975, 389850)])
    assert drift["C1"]["causes"]["ctx_vision_from_replay"] == "the context's slots, cause undetermined"
    # production's arithmetic: both arms missing production's audit is the construction, not the emulation
    both = M.summarize([synthetic_row(s, fid=2.0, fid32=2.0) for s in (380550, 385975, 389850)])
    assert not both["F"]["met"] and both["C5"]["judged"] is False and not both["C5"]["met"]
    assert both["F"]["failure_reading"].startswith("both arms miss")
    only = M.summarize([synthetic_row(s, fid=2.0, fid32=1.0) for s in (380550, 385975, 389850)])
    assert only["F"]["failure_reading"].startswith("only the bf16 arm misses")
    # within 1.14 but not closer to production than fp32: not reproduced
    tie = M.summarize([synthetic_row(s, fid=1.1, fid32=1.05) for s in (380550, 385975, 389850)])
    assert not tie["F"]["met"]
    # the heads arm or the input arm reproduce the bf16 error: named where it lives
    heads = M.summarize([synthetic_row(s, heads_bf16=2.9) for s in (380550, 385975, 389850)])
    assert heads["C5"]["carrier"].startswith("inside the heads")
    inp = M.summarize([synthetic_row(s, head_bf16=3.0) for s in (380550, 385975, 389850)])
    assert inp["C5"]["carrier"].startswith("the latent input")


def test_a_pipeline_that_does_not_reproduce_the_audit_judges_nothing(M):
    rows = [synthetic_row(s) for s in (380550, 385975, 389850)]
    rows[0]["audit_draw"] = dict(ratio=3.2, recorded_ratio=2.926, reproduced=False)
    out = M.summarize(rows)
    assert out["pipeline_reproduces_the_frozen_audit"] is False and "void" in out
    assert not out["C1"]["met"] and not out["C3"]["met"] and not out["C4"]["met"] and not out["C5"]["met"]


def test_the_newest_half_swap_and_the_bias_rounding(M):
    B, T, n, d = 2, 5, 3, 4
    live, other = torch.zeros(B, T * n, d), torch.ones(B, T * n, d)
    out = M.swap_slots(live, other, n, [0], M.newest_half(T)).reshape(B, T, n, d)
    assert M.newest_half(T) == slice(2, 5)
    assert bool((out[:, 2:, 0] == 1).all()) and bool((out[:, :2, 0] == 0).all()) and bool((out[:, :, 1:] == 0).all())
    b = torch.tensor([130.37, 5.123456])
    assert M.round_bias(b, torch.bfloat16).tolist() == [130.0, 5.125]
    assert M.round_bias(b, torch.float32) is b and M.round_bias(None, torch.bfloat16) is None


def test_the_heads_arm_runs_only_the_heads_in_bf16(M):
    B, n, d = 2, 3, 4
    latent = torch.full((B, n, d), 130.37)
    seen = {}

    class Heads:
        residual = True

        def __call__(self, last, current=None):
            seen["last"] = last.clone()
            try:
                seen["autocast"] = torch.is_autocast_enabled("cpu")
            except TypeError:                                   # an older torch: no device argument
                seen["autocast"] = torch.is_autocast_cpu_enabled()
            return {"vision": last[:, 0] + current["vision"]}
    model = SimpleNamespace(forward_model=lambda cur, ctx, memory_bias=None: (None, latent, None),
                            prediction_heads=Heads())
    inp = dict(cur={"vision": torch.zeros(B, 1, d)}, ctx=None, bias=None, order=None, ach=None,
               mask=torch.ones(B, dtype=torch.bool), target={"vision": torch.zeros(B, d)})
    base, total, last = M.forward_arm(None, model, inp, "heads_bf16")
    assert bool((seen["last"] == latent).all()) and seen["autocast"] is True
    assert base["vision"].dtype == torch.float32 and bool((last == latent).all())


def test_real_heads_run_under_cpu_bf16_autocast(M):
    """The vision prediction head with its input scale and a revived-rows hook, and the efference heads'
    gaze copy, under the CPU bf16 autocast the precision arms use: no dtype error, finite outputs."""
    from lem.model.efference import EfferenceHeads
    model, head, last, inp = tiny_model()
    head.float()
    rows = model.liveness.rows["prediction_heads__heads__vision__mlp__0"]
    rows.float()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = head(last.float())
    assert out.shape == (BODIES, D) and bool(torch.isfinite(out.float()).all())
    heads = EfferenceHeads(d_model=D, n_handles=4, gaze_idx=3, head_idx=2, n_mod=N_TOK, vision_idx=0,
                           hidden=16, n_heads=2, gaze_hidden=8, vision_hidden=8)
    heads.ensure_centres(last.float())
    order = torch.zeros(BODIES, 4, 9)
    order[..., 6] = 1.0
    ach = torch.zeros(BODIES, 4, 10)
    ach[..., 6] = 1.0
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        delta = heads.vision_delta(last.float(), order, ach)
    assert delta.shape == (BODIES, D) and bool(torch.isfinite(delta.float()).all())
    # with the activation compensation on, as production's gaze copy runs since the morning repair
    assert heads.vision_activation.configure(heads.vision_in, heads.vision_out,
                                             torch.full((8,), 0.1), torch.full((8,), 0.9))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        delta = heads.vision_delta(last.float(), order, ach)
    assert delta.shape == (BODIES, D) and bool(torch.isfinite(delta.float()).all())


def test_the_lever_rule_counts_checkpoints(M):
    def row(step, total, lever, retained):
        routes = {"all": dict(total=total, lever=lever, lever_share=lever / total if total > 0 else None,
                              specific_outside_shuffles=True),
                  "retained_only": dict(lever=retained), "revived_only": dict(lever=lever - retained)}
        return {"step": step, "lever": {"adam": {"routes": routes}}}
    rows = [row(380550 + k, 1.0, 0.9, 0.8) for k in range(6)] + [row(390000, 1.0, 0.1, 0.0),
                                                                  row(344400, 1.0, 0.9, 0.9)]
    out = M.lever_rule(rows)
    assert out["read"] == 7 and out["met_at"] == 6 and out["met"]          # 344,400 is before the repair
    assert out["per_checkpoint"]["380550"]["route"] == ["retained_only"]
    assert not M.lever_rule(rows[:5])["met"]                                # fewer than 7 read
