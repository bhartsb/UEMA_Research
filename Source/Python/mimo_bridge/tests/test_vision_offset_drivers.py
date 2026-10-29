# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What keeps the vision offset alive (docs/gain_of_function_v4/vision_offset_drivers.py).

Synthetic heads, memories, optimizer states, records and rows only: no checkpoint, no world. They pin what the
pause jobs' readings and rules mean before they run: a fold made per acting call is exact where one made over
several calls is not; the bias's spread splits within and between transitions; a centre's lag reads as the
shared part of its centred input; parameters are named and grouped as production clips them; the saved second
moment's shares and eps's share of the step are read as defined; production's clip factor comes from the
training records of the window before the checkpoint; the counterfactual re-fit applies its factor and eps and
restores the checkpoint; parts move between two models and back exactly; each part's movement is projected on
the offset; and the summaries apply rules E2, K and RC."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

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
def V():
    return _load("vision_offset_drivers", "docs/gain_of_function_v4/vision_offset_drivers.py")


@pytest.fixture(scope="module")
def M(V):
    return V.ablation_module()


@pytest.fixture(scope="module")
def T():
    """The ablation's test helpers (a vision head with revived rows, a fake model with a saved AdamW state)."""
    return _load("test_vision_memory_ablation_helpers", "Source/Python/mimo_bridge/tests/test_vision_memory_ablation.py")


# ------------------------------------------------------------------------------------------- E2
def transitions_case(T, n_trans=3, bodies=4, bias_scale=2000.0, drift=5.0, seed=2):
    """One bias per acting call (the same for its bodies), moving between calls; large enough that every
    attention head of the vision head reads one token."""
    head, rows = T.head_with_rows(seed=seed)
    torch.manual_seed(seed)
    n = n_trans * bodies
    c = bias_scale * torch.randn(T.N_TOK, T.D, dtype=torch.float64)
    per_call = [c + drift * torch.randn(T.N_TOK, T.D, dtype=torch.float64) for _ in range(n_trans)]
    bias = torch.stack([per_call[k // bodies] for k in range(n)])
    s = dict(landing=torch.randn(n, T.N_TOK, T.D, dtype=torch.float64), bias=bias,
             current=torch.randn(n, T.D, dtype=torch.float64), order=torch.zeros(n, 2, 9, dtype=torch.float64),
             ach=torch.zeros(n, 2, 10, dtype=torch.float64), mask=torch.ones(n, dtype=torch.bool),
             transition=[k // bodies for k in range(n)])
    model = SimpleNamespace(prediction_heads=SimpleNamespace(heads={"vision": head}, residual=True),
                            liveness=SimpleNamespace(rows=rows), efference=None)
    return model, s


def test_a_fold_per_call_is_exact_where_one_over_several_calls_is_not(V, M, T):
    model, s = transitions_case(T)
    out = V.fold_exactness(M, model, s)
    assert out["attention_weights_spread_between_moments"] < 1e-9     # every head reads one token
    assert out["pooled"] > out["line"]                                # one constant for three calls misses
    assert out["per_transition"] <= 1e-8 < out["line"]                # one per call is exact
    assert out["per_moment"] <= 1e-8                                  # each moment's own bias: only the weights' spread
    assert set(out["per_transition_each"]) == {"0", "1", "2"}
    b = out["bias"]
    assert b["within_over_constant"] == 0.0 and b["between_over_constant"] > 1e-4 and b["transitions"] == 3
    # a set with one call (the replay draw: transition -1 everywhere) folds as the pooled fold
    one = dict(s, transition=[-1] * len(s["transition"]))
    same = V.fold_exactness(M, model, one)
    assert same["per_transition"] == pytest.approx(same["pooled"], rel=1e-12)
    # a head whose weights still vary between moments (this seed leaves one head near a tie): the fold per
    # moment keeps only that spread, and it shows
    model, s = transitions_case(T, seed=11)
    out = V.fold_exactness(M, model, s)
    assert out["attention_weights_spread_between_moments"] > 1e-6 and out["per_moment"] > out["line"]


def test_the_bias_spread_splits_within_and_between_transitions(V):
    const = torch.full((2, 3), 100.0, dtype=torch.float64)
    b = torch.stack([const, const, const + 1.0, const + 1.0])
    out = V.bias_spread(b, [0, 0, 1, 1])
    assert out["within_transition_rms"] == 0.0
    assert out["between_transition_rms"] == pytest.approx(0.5)       # the two means sit 0.5 from their mean
    assert out["constant_rms"] == pytest.approx(100.5)
    noisy = b + torch.tensor([0.1, -0.1, 0.0, 0.0], dtype=torch.float64).reshape(4, 1, 1)
    out = V.bias_spread(noisy, [0, 0, 1, 1])
    assert out["within_transition_rms"] == pytest.approx(math.sqrt(0.01 / 2), rel=1e-6)


# --------------------------------------------------------------------------------------- centres
def test_a_centres_lag_reads_as_the_shared_part(V):
    torch.manual_seed(12)
    d, n = 16, 200
    c = torch.randn(d, dtype=torch.float64)
    lag = 3.0 * torch.randn(d, dtype=torch.float64)
    x = c + lag + 0.5 * torch.randn(n, d, dtype=torch.float64)
    out = V.centred_input_parts(x, c, nn.LayerNorm(d))
    assert out["shared_rms"] == pytest.approx(float((x.mean(0) - c).square().mean().sqrt()), rel=1e-9)
    assert out["shared_share"] > 0.9                                  # the lag dominates what the layer reads
    assert out["normalised_shared_share"] > 0.9
    fresh = V.centred_input_parts(x, x.mean(0), nn.LayerNorm(d))      # a centre at the inputs' mean
    assert fresh["shared_rms"] < 1e-6 and fresh["shared_share"] < 1e-9


def test_the_centre_readings_use_the_copy_and_the_rows(V, M, T):
    model = T.fake_model()
    s = T.fake_set(model)
    out = V.centre_readings(M, model, s)
    assert set(out) == {"gaze_copy", "revived_rows"}
    x = (s["landing"] + s["bias"])[:, model.efference.vision_idx]
    assert out["gaze_copy"]["input_rms"] == pytest.approx(float(x.double().square().mean().sqrt()), rel=1e-6)


# ------------------------------------------------------------------------------------------- K
class NamedModel:
    """Module attributes named as LemLite's, for part_of, clip_membership and the optimizer readings."""

    def __init__(self):
        torch.manual_seed(13)
        self.encoder_registry = nn.Module()
        self.encoder_registry.encoders = nn.ModuleDict({"vision": nn.Linear(4, 4), "motor": nn.Linear(4, 4)})
        self.forward_model = nn.Linear(4, 4)
        heads = nn.ModuleDict({"vision": nn.Module(), "tactile": nn.Linear(4, 4)})
        heads["vision"].attn = nn.Linear(4, 4)
        heads["vision"].mlp = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        self.prediction_heads = nn.Module()
        self.prediction_heads.heads = heads
        self.persistent_memory = nn.Module()
        self.persistent_memory.readout = nn.Linear(4, 4)
        self.persistent_memory.encoder = nn.Linear(4, 4)
        self.efference = nn.Module()
        self.efference.vision_out = nn.Linear(4, 4)
        self.efference.body_mlp = nn.Linear(4, 4)
        self.liveness = nn.Module()
        self.liveness.mode = "on"
        self.liveness.rows = nn.ModuleDict({"prediction_heads__heads__vision__mlp__0": nn.Linear(4, 4)})
        self.motor_controller = nn.Linear(4, 4)
        self.self_state_mode = "perceive"
        self.params = [p for m in (self.encoder_registry, self.forward_model, self.prediction_heads,
                                   self.persistent_memory, self.efference, self.liveness, self.motor_controller)
                       for p in m.parameters()]


def test_parameters_are_named_and_grouped_as_production_clips_them(V):
    m = NamedModel()
    names = V.param_names(m)
    parts = {V.part_of(names[id(p)]) for p in m.params}
    assert {"encoder: vision", "encoder: motor", "transformer", "vision head: attention", "vision head: MLP",
            "prediction head: tactile", "memory readout", "memory encoder", "gaze copy",
            "vision head: revived rows", "motor_controller"} <= parts
    member = V.clip_membership(m)
    assert member[id(m.efference.vision_out.weight)] == "efference"
    assert member[id(m.liveness.rows["prediction_heads__heads__vision__mlp__0"].weight)] == "liveness"
    assert member[id(m.persistent_memory.readout.weight)] == "core"
    assert member[id(m.prediction_heads.heads["vision"].attn.weight)] == "core"
    m.liveness.mode = "report"                                     # production clips the rows alone only in mode on
    assert V.clip_membership(m)[id(m.liveness.rows["prediction_heads__heads__vision__mlp__0"].weight)] == "core"


def adamw_with_state(model_params, v_of, m_of=None, eps=1e-8, step=1e6):
    opt = torch.optim.AdamW(model_params, lr=3e-4, eps=eps)
    for p in model_params:
        opt.state[p] = {"step": torch.tensor(step), "exp_avg": (m_of(p) if m_of else torch.zeros_like(p)),
                        "exp_avg_sq": v_of(p)}
    return opt


def test_the_optimizer_readings_give_shares_roots_and_eps(V, T):
    model = T.fake_model()
    head = model.prediction_heads.heads["vision"]
    out_layer = [mod for mod in head.mlp if isinstance(mod, nn.Linear)][-1]
    tiny = {id(out_layer.weight), id(out_layer.bias)}
    # every tensor holds v = 1 per number, except the head's output layer, whose v sits far below eps^2
    model.optimizer = adamw_with_state(model.params, lambda p: (torch.full_like(p, 1e-24) if id(p) in tiny
                                                                 else torch.ones_like(p)), step=1e6)
    out = V.optimizer_readings(model)
    labels = V.output_layer_labels(model)["vision head"]
    assert out["eps_share_output_layers"]["vision head"]["median"] > 0.99      # eps takes the whole step
    assert all(out["eps_share"][l]["eps_at_least_half"] == 1.0 for l in labels)
    other = [k for k in out["eps_share"] if k.startswith("gaze copy.vision_out")]
    assert other and all(out["eps_share"][k]["median"] < 1e-7 for k in other)
    # the fake model has no clipping group of its own for the rows (its liveness is not a module), so the
    # rows count in the core; the gaze copy is the efference group
    n_eff = sum(p.numel() for p in model.efference.parameters())
    assert out["groups"]["efference"]["root_sum"] == pytest.approx(math.sqrt(n_eff), rel=1e-9)
    assert sum(x["share"] for x in out["core_parts"].values()) == pytest.approx(1.0, abs=1e-9)


def test_production_clip_reads_the_records_of_the_window(V, tmp_path):
    f1, f2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    f1.write_text("\n".join(json.dumps(r) for r in [
        {"type": "training", "training_step": 900, "avg_grad_norm": 400.0},
        {"type": "training", "training_step": 1000, "avg_grad_norm": 100.0},
        {"type": "mimo", "step": 1000},
        {"type": "training", "training_step": 1100, "avg_grad_norm": 300.0}]) + "\n", encoding="utf-8")
    f2.write_text(json.dumps({"type": "training", "training_step": 1100, "avg_grad_norm": 200.0}) + "\nnot json\n",
                  encoding="utf-8")
    out = V.production_clip(1100, files=[f1, f2], window=250)
    assert out["records"] == 3 and out["steps"] == [900, 1000, 1100]
    assert out["sequence"] == [400.0, 100.0, 200.0]                    # the later file's repeat of 1100 kept
    assert out["median"] == 200.0 and out["factor"] == pytest.approx(1 / 200.0)
    assert V.production_clip(5000, files=[f1, f2])["records"] == 0


def test_the_gradient_readings_give_r_and_the_saved_moments_cosine(V, T):
    model = T.fake_model()
    model.optimizer = adamw_with_state(model.params, lambda p: torch.full_like(p, 4.0), m_of=lambda p: torch.ones_like(p))
    grads = {id(p): torch.full_like(p, 2.0) for p in model.params}
    out = V.gradient_readings(model, grads, factor=0.01)
    head_label = V.output_layer_labels(model)["vision head"][0]
    # r = RMS(factor x gradient) / RMS(root of v) = 0.01 x 2 / 2 for the core; the copy (efference) at factor 1
    assert out["tensors"][head_label]["r"] == pytest.approx(0.01, rel=1e-9)
    assert out["tensors"]["gaze copy.vision_out.weight"]["r"] == pytest.approx(1.0, rel=1e-9)
    assert out["tensors"][head_label]["cosine_saved_first_moment"] == pytest.approx(1.0)
    assert out["r_output_layers"]["vision head"] == pytest.approx(0.01, rel=1e-9)


# ------------------------------------------------------------------------------------ K4 re-fits
def test_the_counterfactual_refit_applies_its_factor_and_eps_and_restores(V, M, T):
    model = T.fake_model()
    s = T.fake_set(model)
    params = V.refit_core_groups(M, model)
    member = V.clip_membership(model)
    assert params and all(member[id(p)] == "core" for p in params)
    assert all(member[id(p)] != "efference" for p in params)          # the copy is held fixed
    before = [p.detach().clone() for p in model.params]
    eps0 = model.optimizer.param_groups[0]["eps"]
    ref = V.counterfactual_refit(M, model, s, params, weight=2.0, factor=None, updates=40)
    prod = V.counterfactual_refit(M, model, s, params, weight=2.0, factor=0.003, eps=1e-30, updates=40)
    seq = V.counterfactual_refit(M, model, s, params, weight=2.0, factors_seq=[0.5, 0.25], updates=40)
    for out in (ref, prod, seq):
        assert set(out) == {0, 1}
        assert [t["update"] for t in out[0]["trace"]] == [0, 20, 40]
    assert 0.0 < ref[0]["factor_first"] <= 1.0
    assert prod[0]["factor_first"] == 0.003 and seq[0]["factor_first"] == 0.5
    # the start is the checkpoint's, whatever the condition
    assert ref[0]["trace"][0]["ratio"] == pytest.approx(prod[0]["trace"][0]["ratio"], rel=1e-12)
    for p, v in zip(model.params, before):
        assert torch.equal(p, v)
    assert model.optimizer.param_groups[0]["eps"] == eps0
    assert all(p.grad is None for p in model.params)
    assert V.fold_ratio(M, {0: {"trace": [dict(update=220, ratio=1.0), dict(update=240, ratio=3.0)]},
                            1: {"trace": [dict(update=400, ratio=2.0)]}}) == pytest.approx(2.0)
    assert V.fold_ratio(M, {0: {"trace": [dict(update=400, ratio=1.0)]}}) is None


def test_the_saved_step_is_read_and_undone(V, M, T):
    model = T.fake_model()
    s = T.fake_set(model)
    before = [p.detach().clone() for p in model.params]
    out = V.saved_step_readings(M, model, s)
    assert out["head_and_copy"]["tensors"] > 0 and out["readout"]["tensors"] == 2
    assert out["offset_rms"] > 0 and out["head_and_copy"]["shared_change_rms"] > 0
    for p, v in zip(model.params, before):
        assert torch.equal(p, v)


# ------------------------------------------------------------------------------------------ pairs
def pair_model(T, seed):
    model = T.fake_model(seed=seed)
    torch.manual_seed(seed + 100)
    model.encoder_registry = nn.Linear(3, 3)
    model.forward_model = nn.Linear(3, 3)
    with torch.no_grad():
        model.efference.centre_mean.copy_(torch.randn_like(model.efference.centre_mean))
    return model


def test_parts_move_between_models_and_back_exactly(V, M, T):
    q, p = pair_model(T, 21), pair_model(T, 22)
    s = T.fake_set(q)
    parts_q, parts_p = V.capture_parts(q), V.capture_parts(p)
    pred_q = V.pathway_prediction(M, q, s)
    assert V.apply_parts(q, parts_p, V.PARTS) == list(V.PARTS)
    pred_p_on_q = V.pathway_prediction(M, q, s)
    assert torch.allclose(pred_p_on_q, V.pathway_prediction(M, p, s), atol=1e-6)   # every moving part moved
    assert not torch.allclose(pred_p_on_q, pred_q)
    V.apply_parts(q, parts_q, V.PARTS)
    assert torch.equal(V.pathway_prediction(M, q, s), pred_q)
    # one part at a time changes only through that part
    V.apply_parts(q, parts_p, ("centre_copy",))
    assert torch.equal(q.efference.centre_mean, p.efference.centre_mean)
    assert torch.equal(q.persistent_memory.readout.weight, parts_q["readout"]["weight"])
    V.apply_parts(q, parts_q, ("centre_copy",))
    assert torch.equal(V.pathway_prediction(M, q, s), pred_q)


def test_each_parts_movement_is_projected_on_the_offset(V):
    target = torch.zeros(4, 3)
    o = torch.tensor([1.0, 0.0, 0.0])
    pred_q = target + o                                              # the later model's offset is o
    preds = {"centres": pred_q - torch.tensor([0.3, 0.2, 0.0]),     # its movement grew the offset by 0.3
             "head": pred_q + torch.tensor([0.2, 0.0, 0.0]),        # its movement reduced it by 0.2
             "copy": pred_q.clone(), "readout": pred_q - torch.tensor([0.1, 0.0, 0.0]),
             "latent": pred_q.clone(), "all": pred_q - torch.tensor([0.2, 0.2, 0.0])}
    out = V.pair_projections(pred_q, target, preds)
    a = out["arms"]
    assert a["centres"]["along_offset"] == pytest.approx(0.3) and a["head"]["along_offset"] == pytest.approx(-0.2)
    assert a["readout"]["along_offset"] == pytest.approx(0.1)
    assert out["additivity_residual"] == pytest.approx(0.0, abs=1e-12) and out["adds_up"]
    preds["all"] = pred_q - torch.tensor([2.0, 0.0, 0.0])
    assert not V.pair_projections(pred_q, target, preds)["adds_up"]


def rc_row(step, s, adds=True):
    arms = {k: dict(along_offset=v) for k, v in s.items()}
    arms["all"] = dict(along_offset=sum(s[k] for k in ("centres", "head", "copy", "readout", "latent")))
    return dict(step=step, earlier_step=step - 25, rc=dict(arms=arms, adds_up=adds, offset_rms=0.1))


def test_the_pairs_summary_names_the_part_that_pushes(V):
    push = dict(centres=0.03, head=-0.02, copy=-0.005, readout=0.01, latent=0.005)
    rows = [rc_row(390000 + 100 * k, push) for k in range(6)]
    out = V.summarize_pairs(rows)
    assert out["named"] == ["the running centres keep the offset alive"] and out["head_and_copy_correct"]
    shared = dict(centres=0.01, head=-0.02, copy=0.0, readout=0.012, latent=0.011)
    out = V.summarize_pairs([rc_row(390000 + 100 * k, shared) for k in range(6)])
    assert out["named"] == ["no single moving part carries it"]
    mixed = rows[:3] + [rc_row(391000 + 100 * k, dict(push, readout=0.05)) for k in range(3)]
    assert V.summarize_pairs(mixed)["named"] == ["no single moving part carries it"]
    # a pair whose parts do not add up is not counted; a reference pair is not pooled
    out = V.summarize_pairs(rows[:2] + [rc_row(392000, push, adds=False), rc_row(344400, push)])
    assert out["pairs_counted"] == 2
    outward = dict(push, head=0.02, copy=0.001)
    assert V.summarize_pairs([rc_row(390000 + 100 * k, outward) for k in range(3)])["head_and_copy_push_outward"]


def k_row(step, ref, prod, eps_removed, eps_share=0.0, r=1.0, e2_i=True, e2_ii=True):
    def run(v):
        return {k: dict(trace=[dict(update=u, ratio=(5.0 if u < 220 else v)) for u in range(0, 401, 20)])
                for k in range(2)}
    k4 = dict(reference=run(ref), production=run(prod), production_eps_removed=run(eps_removed),
              production_varying=run(prod), all_groups=run(ref))
    live = dict(per_transition=(1e-6 if e2_i else 1e-3), per_moment=1e-7, pooled=1e-3, line=2e-5,
                bias=dict(within_over_constant=(0.0 if e2_ii else 1e-3), between_over_constant=3e-4))
    return dict(step=step, K4=k4, E2=dict(live=live),
                K1_K3=dict(eps_share_output_layers={"vision head": dict(median=eps_share)}),
                K5=dict(r_output_layers={"vision head": r}))


def test_the_checkpoints_summary_applies_E2_and_K(V, M):
    steps = (380550, 386750, 389850, 392950, 397125, 399225)
    rows = [k_row(s, 1.0, 1.05, 1.0) for s in steps] + [k_row(344400, 1.0, 3.0, 3.0)]
    out = V.summarize_checkpoints(M, rows)
    assert out["references"] == [344400] and out["E2"]["judged"] == 5          # from 386,750 on
    assert out["E2"]["met"] and out["K"]["outcome"] == "not starved" and not out["K"]["met"]
    starved = V.summarize_checkpoints(M, [k_row(s, 1.0, 2.0, 1.0, eps_share=0.9) for s in steps])
    assert starved["K"]["outcome"] == "clipping starvation (through Adam's eps)" and starved["K"]["met"]
    diluted = V.summarize_checkpoints(M, [k_row(s, 1.0, 2.0, 2.0, r=0.05) for s in steps])
    assert diluted["K"]["outcome"] == "diluted, not clipped" and diluted["K"]["r_at_most_line_at"] == 6
    # eps removal settles but eps takes little of the step: not the eps route
    assert V.summarize_checkpoints(M, [k_row(s, 1.0, 2.0, 1.0, eps_share=0.1) for s in steps])["K"]["outcome"] == "diluted, not clipped"
    no_ref = V.summarize_checkpoints(M, [k_row(s, 2.0, 2.0, 2.0) for s in steps])
    assert no_ref["K"]["outcome"].startswith("not judged")
    e2 = V.summarize_checkpoints(M, [k_row(s, 1.0, 1.0, 1.0, e2_i=False) for s in steps])
    assert not e2["E2"]["met"] and e2["E2"]["ii_met"] and e2["E2"]["reading"].startswith("the attention weights")
