# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The visual regression audit (docs/gain_of_function_v3/audit_visual_regression_2026_09_24.py).

Its 9:28 PM PDT run of 2026-09-24 stopped at the conditioning arm's same-start check (a change of
1465.8 against a limit of 0.00002) and recorded nothing: the arm replaced the first layer's forward
with a plain linear of x / scale, dropping the division by the input_scale this morning's repair
had installed, and the frozen comparison was printed only after the training arms. These tests pin
both repairs on tiny synthetic heads, with no checkpoint and no world."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lem.model.input_conditioning import InputScaledLinear, install, migrate

REPO = Path(__file__).resolve().parents[4]
AUDIT = REPO / "docs" / "gain_of_function_v3" / "audit_visual_regression_2026_09_24.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_visual_regression_2026_09_24", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_head(existing_scale, seed=0, raw_size=300.0):
    """A two-layer head whose first layer reads inputs of RMS about raw_size with weights small
    enough that its outputs are of order 1 (production's vision head before the morning repair:
    input RMS 261.7, first-layer weight norm 0.22), with one AdamW step of history. existing_scale
    None leaves the first layer a plain Linear; otherwise it is conditioned the way this morning's
    repair conditioned production (install + migrate)."""
    torch.manual_seed(seed)
    first = nn.Linear(6, 8)
    with torch.no_grad():
        first.weight.div_(raw_size)
    head = nn.Sequential(first, nn.GELU(), nn.Linear(8, 6))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    x = torch.randn(16, 6) * raw_size
    head(x).square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if existing_scale is not None:
        install(first)
        migrate(first, existing_scale, opt)
    return head, first, opt, x


def test_conditioning_on_top_of_an_input_scale_keeps_the_starting_prediction(audit):
    head, first, opt, x = tiny_head(existing_scale=50.0)
    assert isinstance(first, InputScaledLinear) and float(first.input_scale) == 50.0
    before = head(x).detach().clone()
    weight = first.weight.detach().clone()
    moments = {k: opt.state[first.weight][k].clone() for k in ("exp_avg", "exp_avg_sq")}
    receipt, undo = audit.condition_first_linear(first, opt, x)
    change = float((head(x) - before).abs().max())
    assert change < audit.START_LIMIT
    # only the factor still missing is new: RMS of what the weight multiplies (x / 50), about 6
    effective = float((x / 50.0).square().mean().sqrt())
    assert receipt["existing_input_scale"] == 50.0
    assert receipt["input_rms_scale"] == pytest.approx(effective, rel=1e-6)
    assert receipt["combined_input_scale"] == pytest.approx(50.0 * effective, rel=1e-6)
    assert float(first.input_scale) == pytest.approx(receipt["combined_input_scale"], rel=1e-7)
    k = receipt["input_rms_scale"]
    assert torch.allclose(opt.state[first.weight]["exp_avg"], moments["exp_avg"] / k)
    assert torch.allclose(opt.state[first.weight]["exp_avg_sq"], moments["exp_avg_sq"] / k ** 2)
    # the module's own forward still divides (no replacement forward on a scaled layer)
    assert "forward" not in first.__dict__
    undo()
    assert torch.equal(first.weight, weight)
    assert float(first.input_scale) == 50.0
    for key, value in moments.items():
        assert torch.equal(opt.state[first.weight][key], value)
    assert torch.equal(head(x), before)


def test_the_failed_runs_replacement_forward_breaks_the_start_check(audit):
    """The 9:28 PM arm's arithmetic, reproduced: the weight times the raw RMS, a forward of
    x / raw RMS, and the layer's own input_scale dropped. The same-start check must refuse it."""
    head, first, opt, x = tiny_head(existing_scale=50.0)
    before = head(x).detach().clone()
    scale = max(1.0, float(x.square().mean().sqrt()))
    with torch.no_grad():
        first.weight.mul_(scale)
    first.forward = lambda v, linear=first, s=scale: torch.nn.functional.linear(v / s, linear.weight, linear.bias)
    assert float((head(x) - before).abs().max()) > 1000 * audit.START_LIMIT


def test_a_plain_linear_is_conditioned_as_this_morning(audit):
    head, first, opt, x = tiny_head(existing_scale=None)
    before = head(x).detach().clone()
    receipt, undo = audit.condition_first_linear(first, opt, x)
    assert receipt["existing_input_scale"] == 1.0
    assert receipt["input_rms_scale"] == pytest.approx(float(x.square().mean().sqrt()), rel=1e-6)
    assert float((head(x) - before).abs().max()) < audit.START_LIMIT
    assert "forward" in first.__dict__
    undo()
    assert "forward" not in first.__dict__
    assert torch.allclose(head(x), before)


def test_an_input_already_in_measured_units_is_left_unchanged(audit):
    head, first, opt, x = tiny_head(existing_scale=5000.0)
    before = head(x).detach().clone()
    receipt, undo = audit.condition_first_linear(first, opt, x)
    assert receipt["input_rms_scale"] == 1.0
    assert torch.equal(head(x), before)
    undo()


def test_the_frozen_comparison_is_written_before_any_training_arm(audit, tmp_path, monkeypatch):
    out = tmp_path / "audit"
    windows = 4
    fake = {"replay_window": {"tokens": torch.zeros(windows, 1)}, "source_step": 123}
    monkeypatch.setattr(audit, "commit_reading", lambda: None)
    monkeypatch.setattr(audit, "fixed_inputs", lambda checkpoint, cache, n: (fake, "test inputs"))

    def frozen_phase(args, data, result, save=lambda: None):
        result["models"]["before"] = {"total_mse": 1.0, "persistence_mse": 0.5}
        save()
        result["models"]["after"] = {"total_mse": 3.0, "persistence_mse": 0.5}
        result["models"]["old_prediction_heads"] = {"total_mse": 9.0}
        save()
    monkeypatch.setattr(audit, "frozen_phase", frozen_phase)
    seen = []

    def training_phase(name, checkpoint, data):
        on_disk = json.loads((out / "result.json").read_text(encoding="utf-8"))
        seen.append((name, on_disk["status"], sorted(on_disk["models"])))
        assert on_disk["models"]["after"]["total_mse"] == 3.0
        if name == "after":
            raise AssertionError("conditioning changed the starting prediction")
        return {"total_regularizer_last20_held_out_mean": 0.1}
    monkeypatch.setattr(audit, "training_phase", training_phase)
    args = SimpleNamespace(before=Path("b"), after=Path("a"), output=out, train=True, windows=windows)
    assert audit.run(args) == 1
    assert [s[0] for s in seen] == ["before", "after"]
    assert seen[0][1].startswith("frozen comparison and part swaps written")
    final = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert final["models"]["before"]["isolated_training"] == {"total_regularizer_last20_held_out_mean": 0.1}
    assert "conditioning changed" in final["models"]["after"]["isolated_training"]["error"]
    assert final["models"]["old_prediction_heads"]["total_mse"] == 9.0
    assert final["status"] == "frozen comparison complete; training arms failed for after"
    assert final["source_step"] == 123
    # the result names what its labelled readings are (2026-09-25 label fixes)
    assert set(final["definitions"]) >= {"copy_rms_all_windows", "latent_penalty_all_senses_unweighted_base",
                                         "latent_penalty_all_senses_unweighted_base_plus_copy",
                                         "isolated_training.weighted_loss"}


def test_metrics_names_the_copy_size_over_every_window(audit):
    """copy_rms_all_windows counts a window without a known executed order as zero, so with the copy
    on half the windows it is the copy's size on those windows times the square root of one half."""
    torch.manual_seed(3)
    n, d = 8, 5
    base = {"vision": torch.randn(n, d)}
    copy = torch.randn(n, d)
    mask = torch.tensor([1, 0] * (n // 2), dtype=torch.float32)
    total = {"vision": base["vision"] + copy * mask[:, None]}
    inp = {"target": {"vision": torch.randn(n, d)}, "cur": {"vision": torch.randn(n, 1, d)}}
    out = audit.metrics(base, total, inp)
    assert "copy_rms" not in out
    on_copy_windows = float(copy[mask.bool()].square().mean().sqrt())
    assert out["copy_rms_all_windows"] == pytest.approx(on_copy_windows * 0.5 ** 0.5, rel=1e-6)


def test_penalty_readings_are_all_sense_and_unweighted(audit):
    """The frozen comparison's penalty counts every predicted sense, with no lesson weight: a sense
    far outside the reference distribution raises it although vision is unchanged, and base and
    base-plus-copy differ only through vision."""
    from lem.training.latent_regularizer import LatentRegularizer
    d = 4
    reg = LatentRegularizer(d_model=d)          # reference: mean 0, spread 1, line at 2
    model = SimpleNamespace(latent_regularizer=reg)
    vision = torch.zeros(2, d)
    touch = torch.full((2, d), 5.0)             # 3 spreads beyond the line in every dimension
    base = {"vision": vision, "tactile": touch}
    total = {"vision": vision + 4.0, "tactile": touch}    # 2 beyond the line in vision
    out = audit.penalty_readings(model, base, total)
    # mean over both senses' rows of the squared excess: (0 + 9) / 2 and (4 + 9) / 2, unweighted
    assert out["latent_penalty_all_senses_unweighted_base"] == pytest.approx(4.5)
    assert out["latent_penalty_all_senses_unweighted_base_plus_copy"] == pytest.approx(6.5)
    assert set(out) == {"latent_penalty_all_senses_unweighted_base",
                        "latent_penalty_all_senses_unweighted_base_plus_copy"}


# ---------------------------------------------------------------------------- the path readings
from lem.model.prediction_heads import PredictionHead          # noqa: E402
from lem.model.liveness import RevivedRows                      # noqa: E402

D, MODS, BODIES = 8, 3, 6


def tiny_vision_model(dtype=torch.float32, seed=1, scale=40.0, dominant=3.0):
    """A PredictionHead whose first layer is an InputScaledLinear with an existing scale, whose
    first five units are revived rows kept in a real nn.ModuleDict (LivenessRepair.rows is one:
    it has no .get), wired by a forward hook as LivenessRepair.attach_repair wires them; the
    attention's value-to-output map is given one dominant direction."""
    torch.manual_seed(seed)
    head = PredictionHead(d_model=D, n_heads=2, hidden_expansion=1, n_mlp_layers=2).to(dtype)
    first = head.mlp[0]
    install(first)
    with torch.no_grad():
        first.input_scale.fill_(scale)
        a, b = torch.randn(D, dtype=dtype), torch.randn(D, dtype=dtype)
        head.attn.in_proj_weight[2 * D:] += dominant * torch.outer(a / a.norm(), b / b.norm())
    rows = nn.ModuleDict({"prediction_heads__heads__vision__mlp__0": RevivedRows(D, D).to(dtype)})
    row = rows["prediction_heads__heads__vision__mlp__0"]
    with torch.no_grad():
        row.mask[:5] = True
        row.centre_mean.copy_(torch.randn(D, dtype=dtype))
    row.any_py = True
    first.register_forward_hook(lambda mod, inp, out, r=row: r.apply(inp[0], out))
    from lem.training.latent_regularizer import LatentRegularizer
    model = SimpleNamespace(prediction_heads=SimpleNamespace(heads={"vision": head}, residual=True),
                            liveness=SimpleNamespace(rows=rows), efference=None,
                            latent_regularizer=LatentRegularizer(d_model=D), pred_mods=["vision"])
    last = torch.randn(BODIES, MODS, D, dtype=dtype)
    bias = 5.0 * torch.randn(BODIES, MODS, D, dtype=dtype)
    inp = {"bias": bias, "cur": {"vision": torch.randn(BODIES, 1, D, dtype=dtype)},
           "target": {"vision": torch.randn(BODIES, D, dtype=dtype)},
           "mask": torch.ones(BODIES, dtype=torch.bool), "order": None, "ach": None}
    return model, head, rows, last + bias, inp


def test_path_readings_run_on_a_real_moduledict_of_revived_rows(audit):
    model, head, rows, last, inp = tiny_vision_model()
    assert audit.vision_row(model) is rows["prediction_heads__heads__vision__mlp__0"]
    with torch.no_grad():
        out = audit.path_readings(model, inp, last)
    assert out["first_layer_input_scale"] == 40.0
    assert out["effective_input_rms"] == pytest.approx(out["first_layer_input_rms"] / 40.0, rel=1e-6)
    assert out["retained_units"] == D - 5
    # the value path is linear: the two parts and the constant add up to the attention output
    assert out["attention_split_check_max_abs"] < 1e-4
    assert 0.0 < out["attention_output_from_memory_bias_share"]
    for key in ("first_layer_input_top_direction_share", "revived_units_input_top_direction_share"):
        assert 0.0 < out[key] <= 1.0 + 1e-6
    assert out["attention_map_top_gain"] > 0.0
    # a build without the repair has no row, and every unit counts as retained
    bare = SimpleNamespace(prediction_heads=model.prediction_heads, liveness=None)
    assert audit.vision_row(bare) is None


@pytest.fixture(scope="module")
def decomposition():
    path = REPO / "docs" / "gain_of_function_v4" / "vision_lineage_decomposition.py"
    spec = importlib.util.spec_from_file_location("vision_lineage_decomposition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_growth_attribution_matches_finite_differences(decomposition):
    """The push on the attention gain: its sign and size against the change of the top singular
    value measured after the step, for plain descent on the prediction loss and for one AdamW
    update from the saved moments; and the retained and revived routes add up to the whole."""
    model, head, rows, last, inp = tiny_vision_model(dtype=torch.float64)
    attn = head.attn
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4)
    for _ in range(3):
        opt.zero_grad()
        (head(last) - inp["target"]["vision"]).square().mean().backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    model.optimizer = opt
    # the weights as they are before the call (the call must leave them, and their flags, unchanged)
    w_in, w_out = attn.in_proj_weight.detach().clone(), attn.out_proj.weight.detach().clone()
    assert attn.in_proj_weight.requires_grad and attn.out_proj.weight.requires_grad
    out = decomposition.growth_attribution(model, inp, last, optimizer_loaded=True)
    assert torch.equal(attn.in_proj_weight, w_in) and torch.equal(attn.out_proj.weight, w_out)
    assert attn.in_proj_weight.requires_grad and attn.out_proj.weight.requires_grad

    def top_gain(w_in, w_out):
        return float(torch.linalg.svdvals(w_out @ w_in[2 * D:])[0])
    s0 = top_gain(w_in, w_out)
    assert out["top_gain"] == pytest.approx(s0, rel=1e-5)
    # plain descent on the prediction loss (base head + current against the target; no gaze copy)
    loss = ((head(last) + inp["cur"]["vision"].squeeze(1)) - inp["target"]["vision"]).square().mean()
    g_in, g_out = torch.autograd.grad(loss, [attn.in_proj_weight, attn.out_proj.weight])
    eps = 1e-6
    fd = (top_gain(w_in - eps * g_in, w_out - eps * g_out) - s0) / eps
    rates = out["sigma_rate"]
    assert rates["prediction"] == pytest.approx(fd, rel=1e-3, abs=1e-6)
    assert rates["prediction"] == pytest.approx(
        rates["prediction_retained_units_only"] + rates["prediction_revived_units_only"], rel=1e-4, abs=1e-8)
    # one AdamW update from the saved moments, as growth_attribution builds it
    grp = opt.param_groups[0]
    b1, b2 = grp["betas"]

    def update(p):
        st = opt.state[p]
        t = float(st["step"])
        m, v = st["exp_avg"] / (1 - b1 ** t), st["exp_avg_sq"] / (1 - b2 ** t)
        return -grp["lr"] * m / (v.sqrt() + grp["eps"]) - grp["lr"] * grp["weight_decay"] * p.detach()
    u_in, u_out = update(attn.in_proj_weight), update(attn.out_proj.weight)
    # central difference: the first-order change without the second-order term, whatever the seed
    central = (top_gain(w_in + u_in, w_out + u_out) - top_gain(w_in - u_in, w_out - u_out)) / 2
    assert out["adam_step_change"] == pytest.approx(central, rel=1e-3, abs=1e-9)
