# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--liveness-rule (lem/model/liveness.py WHICH UNITS THE REVIVAL CHOOSES), on small stand-in models, CPU.

The defect (docs/gain_of_function_v4/agents/liveness_revive_varying_2026-09-26.md): LivenessRepair.revive chose units by
the -3 line alone, so a unit held at one ACTIVE value was never revived. On H84 at 415,025 the vision prediction head
read 100 of 100 units live by that line and 43 of 100 varying; the touch head 100 and 4. Every test here that names the
varying rule fails on the code before the switch (the rule did not exist); the default rule's tests hold the old
choice and the old records exactly.

The rule none and the kept-unit records (Lane E review 2026-09-26, docs/gain_of_function_v4/agents/
lane_e_reviews_2026-09-26.md section 1): production refused --liveness-repair report on H84's checkpoint, and report
would have dropped every revived unit; a varying-rule window that kept its held units and revived none sent nothing to
the stats. The tests below the line fail on d3b22ead2."""
import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lem.model import liveness as LV


class Held:
    """attribute -> Sequential(Linear(10, 8), GELU, Linear(8, 2)): units 0-4 held at GELU(5) = 5.0 (units 0 and 1 move
    by far less than one bf16 step at 5.0, which is 2 ** -5 = 0.031), unit 5 under the -3 line, units 6 and 7 moving."""

    HELD = (0, 1, 2, 3, 4)
    BELOW = (5,)
    MOVING = (6, 7)

    def __init__(self):
        torch.manual_seed(3)
        self.held = nn.Sequential(nn.Linear(10, 8), nn.GELU(), nn.Linear(8, 2))
        with torch.no_grad():
            w, b = self.held[0].weight, self.held[0].bias
            w[:6] = 0.0
            b[:5] = 5.0
            w[:2] = 1e-4
            b[5] = -20.0
            b[6:] = 0.0
        self.receipt = {}
        self.optimizer = torch.optim.AdamW(list(self.held.parameters()), lr=3e-4)

    def inputs(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(n, 10, generator=g)


def _repair(m, **kw):
    rep = LV.LivenessRepair(m, "on", seed=0, sample_every=1, revive_min_samples=32, layers=("held.0",), pools=(),
                            **kw)
    m.optimizer.add_param_group({"params": list(rep.parameters()), "weight_decay": 0.0})
    rep.attach_repair(m)
    return rep


def _train(m, n, seed0):
    for i in range(n):
        loss = (m.held(m.inputs(32, seed0 + i)) - 1.0).pow(2).mean()
        m.optimizer.zero_grad()
        loss.backward()
        m.optimizer.step()


def test_the_varying_rule_revives_a_unit_held_at_one_active_value_and_keeps_the_output():
    m = Held()
    _train(m, 3, 100)                                   # the optimiser holds moments for every tensor
    rep = _repair(m, revive_rule="varying")
    rows = rep.rows["held__0"]
    x = m.inputs(32, 0)
    m.held(x).sum().backward()                          # the watching window: one trained call of 32 rows
    assert rows.seen_n == 32 and rows.seen_lo is not None
    y0 = m.held(x).detach()
    rec = rep.revive(m, "held.0", step=9)
    y1 = m.held(x).detach()
    assert rep.revived["held.0"] == list(Held.HELD + Held.BELOW), rep.revived
    assert rec["rule"] == "varying" and rec["below_line"] == 1 and rec["held_above_line"] == 5
    # the output is kept: the held units' mean moved into the next layer's bias, their range bounds the change
    assert rec["output_change_max"] <= rec["output_change_bound"] + 1e-6
    assert float((y1 - y0).abs().max()) <= rec["output_change_bound"] + 1e-6
    # held units move by under one bf16 step at their size (0.031 at 5.0), so the bound is a small share of the output
    assert rec["output_change_bound"] < 0.01 * float(y0.abs().mean()), (rec["output_change_bound"], y0.abs().mean())
    # the revived units read the centred, normalised input with fresh weights: they now vary on the watched inputs
    assert rec["revived_varying_share_on_probe"] == 1.0
    assert rep.stamp()["revive_rule"] == "varying"
    rep.detach()


def test_the_default_rule_chooses_by_the_line_alone_and_records_what_it_recorded_before():
    m = Held()
    rep = _repair(m)
    rows = rep.rows["held__0"]
    assert rep.revive_rule == "live_line" and not rows.track_range
    x = m.inputs(32, 0)
    m.held(x).sum().backward()
    assert rows.seen_lo is None and rows.seen_hi is None, "the default window keeps no output range"
    rec = rep.revive(m, "held.0", step=9)
    assert rep.revived["held.0"] == list(Held.BELOW), "the default rule revives only the unit under the line"
    assert not {"rule", "below_line", "held_above_line", "revived_varying_share_on_probe"} & set(rec)
    assert "revive_rule" not in rep.stamp() and "revive_rule" not in rep.window_summary(m)
    rep.detach()


def test_the_varying_rule_does_not_revive_again_a_unit_that_varies_after_its_revival():
    m = Held()
    rep = _repair(m, revive_rule="varying")
    x = m.inputs(32, 0)
    m.held(x).sum().backward()
    rep.revive(m, "held.0", step=1)
    _train(m, 20, 300)                                  # training calls fill the next window (20 x 32 rows)
    rec = rep.revive(m, "held.0", step=21)
    assert rec["dead"] == 0 and rec["repaired_again"] == 0, rec
    rep.detach()


def test_the_varying_rule_keeps_a_unit_that_moves_by_more_than_one_bf16_step_at_its_size():
    m = Held()
    with torch.no_grad():
        m.held[0].bias[2] = 40.0                        # one bf16 step at 40 is 0.25
        m.held[0].weight[2] = 0.0
        m.held[0].weight[2, 0] = 0.1                    # an input spread of about 5 moves it by about 0.5
        m.held[0].bias[3] = 40.0
        m.held[0].weight[3] = 0.0
        m.held[0].weight[3, 0] = 0.01                   # about 0.05: under the step, held
    rep = _repair(m, revive_rule="varying")
    m.held(m.inputs(32, 0)).sum().backward()
    rep.revive(m, "held.0", step=1)
    assert 2 not in rep.revived["held.0"] and 3 in rep.revived["held.0"], rep.revived
    rep.detach()


def test_a_held_unit_whose_replacement_would_be_held_too_keeps_its_weights():
    """415,025's vision head: its input's shared part was 12,884 per number against a spread of 8.8, and most fresh
    replacements read by a revived unit would not vary either; reviving them would discard learned weights at every
    window for nothing. Here the layer's input is the same row every time: nothing can vary, so only the unit under
    the -3 line is revived (as the default rule revives it)."""
    m = Held()
    rep = _repair(m, revive_rule="varying")
    rows = rep.rows["held__0"]
    x = m.inputs(1, 0).expand(32, 10).contiguous()
    m.held(x).sum().backward()
    w_before = m.held[0].weight.detach().clone()
    rec = rep.revive(m, "held.0", step=3)
    assert rep.revived["held.0"] == list(Held.BELOW), rep.revived
    assert rec["held_kept_replacement_still"] == 7 and rec["held_above_line"] == 0 and rec["below_line"] == 1
    keep = [i for i in range(8) if i not in Held.BELOW]
    assert torch.equal(m.held[0].weight[keep], w_before[keep]), "a held unit that was not revived lost its weights"
    assert not bool(rows.mask[keep].any())
    rep.detach()


def test_the_rule_is_checked_and_reaches_the_trainer():
    m = Held()
    with pytest.raises(ValueError):
        LV.LivenessRepair(m, "on", layers=("held.0",), pools=(), revive_rule="sometimes")
    with pytest.raises(ValueError):
        LV.LivenessRepair(m, "report", layers=(), pools=(), revive_rule="varying")
    with pytest.raises(ValueError):
        LV.LivenessRepair(m, "report", layers=(), pools=(), revive_rule="none")
    assert LV.DEFAULT_REVIVE_RULE == "live_line" and LV.REVIVE_RULES == ("live_line", "varying", "none")
    from mimo_bridge import train_mimo as TM
    assert TM.parse_args(["--out", "x"]).liveness_rule == "live_line"
    assert TM.parse_args(["--out", "x", "--liveness-rule", "varying"]).liveness_rule == "varying"
    assert TM.parse_args(["--out", "x", "--liveness-rule", "none"]).liveness_rule == "none"
    import inspect
    from mimo_bridge.lem_model import LemLite
    assert inspect.signature(LemLite.__init__).parameters["liveness_rule"].default == "live_line"


def test_the_default_rule_stats_block_keeps_the_keys_it_had_and_records_no_quiet_window():
    """A guard, not a new behaviour (it passes on d3b22ead2 too): under the default rule the stats block carries exactly
    the keys it carried before the rule none and the kept-unit records, and a window that revives nothing stores no
    record."""
    m = Held()
    with torch.no_grad():
        m.held[0].bias[5] = 5.0                         # no unit under the -3 line: the default rule finds none
    rep = _repair(m)
    m.held(m.inputs(32, 0)).sum().backward()
    rec = rep.revive(m, "held.0", step=4)
    assert rec["dead"] == 0 and rep.revival_record == {}
    out = rep.window_summary(m)
    assert set(out) == {"mode", "layers", "pools", "line", "layer_line", "samples", "sampled_steps", "window_steps",
                        "sample_every", "hook_ms_per_step", "below_line", "layers_varying", "not_varying",
                        "varying_rule", "token_read", "revived", "revived_units", "revived_at_step", "watching",
                        "revival"}, sorted(out)
    assert out["revival"] == {} and out["watching"]["held.0"]["ready"] is True
    rep.detach()


# ---------------------------------------------------------------- the rule none and the kept-unit records (fail on d3b22ead2)
NEVER = 2 ** 63 - 1          # the memory trial's clean arm: mode on with a watching window no run reaches


def _saved_revival():
    """A Held layer whose unit under the -3 line (5) the default rule revived at step 9, and what a checkpoint keeps of
    the repair: its state dict (in model.safetensors), its centres (runtime_state.pt) and its receipt stamp."""
    m = Held()
    _train(m, 3, 100)
    rep = _repair(m)
    x = m.inputs(32, 0)
    m.held(x).sum().backward()
    rep.revive(m, "held.0", step=9)
    assert rep.revived["held.0"] == [5]
    y = m.held(x).detach()                              # the layer pair's output with the revived unit routed
    sd = {k: v.detach().clone() for k, v in rep.state_dict().items()}
    rt = {k: {"centre_mean": v["centre_mean"].clone(), "centre_weight": v["centre_weight"]}
          for k, v in rep.runtime_state().items()}
    stamp = json.loads(json.dumps(rep.stamp()))
    rep.detach()
    return m, (sd, rt, stamp), (x, y)


def _resume(m, saved, rule, window):
    """The repair rebuilt and loaded the way LemLite.load_checkpoint loads it."""
    sd, rt, stamp = saved
    rep = LV.LivenessRepair(m, "on", seed=0, sample_every=1, revive_min_samples=window, layers=("held.0",), pools=(),
                            revive_rule=rule)
    rep.load_state_dict(sd)
    assert rep.load_runtime_state(rt) == ["held.0"]
    rep.after_load(m, saved=stamp)
    m.optimizer.add_param_group({"params": list(rep.parameters()), "weight_decay": 0.0})
    rep.attach_repair(m)
    return rep


def _train_resumed(m, rep, steps=8):
    """Training steps as LemLite.step runs them (begin_step, the trained call, end_step), a stats report after each."""
    reports = []
    for i in range(steps):
        rep.begin_step(m, 10 + i)
        loss = (m.held(m.inputs(32, 400 + i)) - 1.0).pow(2).mean()
        m.optimizer.zero_grad()
        loss.backward()
        m.optimizer.step()
        rep.end_step(m, 10 + i, 32, train=True)
        reports.append(rep.window_summary(m))
    return reports


def test_the_rule_none_keeps_every_saved_part_revives_nothing_and_keeps_reporting():
    """The restart after the memory trial: the saved revived unit keeps its routing, centre and learning, the gauges
    report, and no unit is revived although the default rule revives one in the same steps. It trains bit for bit
    as the memory trial's clean arm (mode on, a watching window no run reaches)."""
    m0, saved, (x, y) = _saved_revival()
    sd, rt, stamp = saved
    arms = {}
    for rule, window in (("none", 32), ("live_line", 32), ("live_line", NEVER)):
        m = copy.deepcopy(m0)
        rep = _resume(m, saved, rule, window)
        with torch.no_grad():
            # every saved part loaded: the revived unit is routed as before the save, bit for bit
            assert torch.equal(m.held(x), y), rule
            m.held[0].weight[6] = 0.0                   # after the resume a unit falls under the -3 line
            m.held[0].bias[6] = -20.0
        arms[(rule, window)] = (m, rep, _train_resumed(m, rep))
    m, rep, reports = arms[("none", 32)]
    rows = rep.rows["held__0"]
    assert torch.equal(rows.mask, sd["rows.held__0.mask"]) and torch.equal(rows.done, sd["rows.held__0.done"])
    assert torch.equal(rows.adam_birth_weight, sd["rows.held__0.adam_birth_weight"])
    assert torch.equal(rows.adam_birth_bias, sd["rows.held__0.adam_birth_bias"])
    assert rep.revived == {"held.0": [5]}
    st = rep.stamp()
    assert st["revived"] == {"held.0": 1} and st["revived_at_step"] == {"held.0": 9} and st["revive_rule"] == "none"
    assert m.receipt["liveness_repair"]["revived_at_step"] == {"held.0": 9}
    # no unit revived and nothing watched; each repair layer's first check never comes (ready False, needed None), as in
    # the memory trial's clean arm, so the H84 hourly watcher's liveness stop treats the repair layers as that arm's
    assert rows.seen_n == 0 and rows.seen_max is None and rows.probe_x is None
    for out in reports:
        assert out["revival"] == {} and out["revive_rule"] == "none", out
        assert out["watching"] == {"held.0": {"samples": 0, "needed": None, "ready": False}}, out["watching"]
        assert "held.0" in out["layers_varying"] and isinstance(out["not_varying"], list) and "held.0" in out["layers"]
        assert out["revived_units"] == {"held.0": 1}
    # the revived unit keeps learning and its centre keeps following the input
    assert rows.centre_weight > rt["held__0"]["centre_weight"]
    assert not torch.equal(rows.lin.weight[5], sd["rows.held__0.lin.weight"][5])
    with pytest.raises(RuntimeError):
        rep.revive(m, "held.0", step=99)
    # the control: the default rule, same resume, same steps, revives the unit that fell under the line
    _m, rep_ll, reports_ll = arms[("live_line", 32)]
    assert any("held.0" in out["revival"] for out in reports_ll) and 6 in rep_ll.revived["held.0"]
    # the memory trial's clean arm trains bit for bit as the rule none
    m_w, rep_w, reports_w = arms[("live_line", NEVER)]
    assert all(out["revival"] == {} for out in reports_w) and rep_w.revived == {"held.0": [5]}
    for (ka, a), (kb, b) in zip(list(m.held.state_dict().items()) + list(rep.state_dict().items()),
                                list(m_w.held.state_dict().items()) + list(rep_w.state_dict().items())):
        assert ka == kb and torch.equal(a, b), ka
    sa, sb = m.optimizer.state_dict()["state"], m_w.optimizer.state_dict()["state"]
    assert sa.keys() == sb.keys()
    for k in sa:
        for kk, v in sa[k].items():
            assert torch.equal(v, sb[k][kk]) if torch.is_tensor(v) else v == sb[k][kk], (k, kk)
    for r in (rep, rep_ll, rep_w):
        r.detach()


def test_the_varying_rule_reports_a_window_that_keeps_every_held_unit():
    """The review's planted case: every unit held above the -3 line, none under it, the same input row throughout. The
    check keeps all 8 held units (no replacement could vary) and revives none; before the fix nothing reached the
    stats."""
    m = Held()
    with torch.no_grad():
        m.held[0].weight.zero_()
        m.held[0].bias.fill_(5.0)
    rep = _repair(m, revive_rule="varying")
    m.held(m.inputs(1, 0).expand(32, 10).contiguous()).sum().backward()
    rec = rep.revive(m, "held.0", step=3)
    assert rec["dead"] == 0 and rec["held_kept_replacement_still"] == 8 and rec["below_line"] == 0
    out = rep.window_summary(m)
    assert out["revival"]["held.0"]["held_kept_replacement_still"] == 8, out["revival"]
    assert out["revival_windows"] == {"held.0": {"windows": 1, "revived": 0, "below_line": 0, "held_above_line": 0,
                                                 "held_kept_replacement_still": 8, "probe_under_2_rows": 0}}
    assert rep.revived_at_step == {}, "a window that revives nothing is no revival"
    again = rep.window_summary(m)
    assert again["revival"] == {} and again["revival_windows"] == {}, "each report carries its own windows"
    rep.detach()


def test_a_window_whose_newest_call_has_one_row_reaches_the_stats_and_is_counted():
    """The replacement check judges on the newest call's rows only (RevivedRows.watch keeps them): a window ending on a
    single-row call keeps every held unit. The review's planted case: 6 held units kept, 0 revived, nothing recorded."""
    m = Held()
    with torch.no_grad():
        m.held[0].bias[5] = 5.0                         # unit 5 held too: units 0-5 held, 6 and 7 moving
    rep = _repair(m, revive_rule="varying")
    m.held(m.inputs(31, 0)).sum().backward()
    m.held(m.inputs(1, 1)).sum().backward()             # the window's newest call: one row
    rec = rep.revive(m, "held.0", step=5)
    assert rec["dead"] == 0 and rec["held_kept_replacement_still"] == 6 and rec["probe_rows"] == 1, rec
    out = rep.window_summary(m)
    assert out["revival"]["held.0"]["held_kept_replacement_still"] == 6
    assert out["revival_windows"]["held.0"]["probe_under_2_rows"] == 1
    assert out["revival_windows"]["held.0"]["windows"] == 1
    rep.detach()


def test_a_kept_window_never_replaces_a_revival_record_of_the_same_report():
    m = Held()
    rep = _repair(m, revive_rule="varying")
    m.held(m.inputs(32, 0)).sum().backward()
    first = rep.revive(m, "held.0", step=1)             # revives units 0-5 (5 held, 1 under the line)
    assert first["dead"] == 6
    with torch.no_grad():                               # a later window: unit 7 held, same row, no replacement varies
        m.held[0].weight[7] = 0.0
        m.held[0].bias[7] = 5.0
    m.held(m.inputs(1, 7).expand(32, 10).contiguous()).sum().backward()
    second = rep.revive(m, "held.0", step=2)
    assert second["dead"] == 0 and second["held_kept_replacement_still"] >= 1, second
    out = rep.window_summary(m)
    assert out["revival"]["held.0"]["step"] == 1, "the report's revival record was replaced by a kept window"
    t = out["revival_windows"]["held.0"]
    assert t["windows"] == 2 and t["revived"] == 6 and t["held_kept_replacement_still"] == second[
        "held_kept_replacement_still"]
    rep.detach()


def test_a_checkpoint_saved_in_mode_on_resumes_under_rule_none_with_every_repair_part(tmp_path):
    """LemLite, CPU, no vision: a checkpoint whose repair revived units loads under --liveness-rule none without the
    refusal that --liveness-repair report meets, and the next save carries every liveness tensor and the receipt's
    revival record unchanged."""
    from safetensors import safe_open
    from mimo_bridge.lem_model import LemLite
    kw = dict(device="cpu", seed=0, imagination=False, no_vision=True)
    a = LemLite(checkpoint_dir=str(tmp_path / "a"), liveness_repair="on", **kw)
    name = next(n for n in a.liveness.layer_names if n != "contact_head.ctx1")
    lin, _act, _nxt = LV.layer_parts(a, name)
    with torch.no_grad():
        lin.weight[:4] = 0.0
        lin.bias[:4] = -20.0
        lin(torch.randn(64, lin.in_features))           # a watching window through the repair's hook
    a.liveness.revive(a, name, step=5)
    assert a.liveness.revived[name][:4] == [0, 1, 2, 3]
    ckpt = a.save_checkpoint(5)
    saved = json.loads((Path(ckpt) / "model_receipt.json").read_text(encoding="utf-8"))["liveness_repair"]
    assert saved["mode"] == "on" and saved["revived_at_step"] == {name: 5}
    # production's refusal, the restart's blocker: report builds none of the saved parts
    rep = LemLite(liveness_repair="report", **kw)
    with pytest.raises(ValueError, match="liveness-rule none"):
        rep.load_checkpoint(ckpt, load_optimizer=False)
    del rep
    b = LemLite(checkpoint_dir=str(tmp_path / "b"), liveness_repair="on", liveness_rule="none", **kw)
    b.load_checkpoint(ckpt)
    assert b.liveness.revived == a.liveness.revived
    got = b.liveness.state_dict()
    for k, v in a.liveness.state_dict().items():
        assert torch.equal(v, got[k]), k
    assert b.receipt["liveness_repair"]["revived_at_step"] == {name: 5}
    assert b.receipt["liveness_repair"]["revive_rule"] == "none"
    ckpt2 = b.save_checkpoint(6)
    again = json.loads((Path(ckpt2) / "model_receipt.json").read_text(encoding="utf-8"))["liveness_repair"]
    assert again["revived"] == saved["revived"] and again["revived_at_step"] == saved["revived_at_step"]
    with safe_open(str(Path(ckpt) / "model.safetensors"), "pt") as f0, \
            safe_open(str(Path(ckpt2) / "model.safetensors"), "pt") as f1:
        keys = [k for k in f0.keys() if k.startswith("liveness_repair.")]
        assert keys and set(keys) == {k for k in f1.keys() if k.startswith("liveness_repair.")}
        for k in keys:
            assert torch.equal(f0.get_tensor(k), f1.get_tensor(k)), k


def test_the_dashboard_says_revival_is_off_under_the_rule_none():
    from mimo_bridge.tools.development_dashboard import liveness_card
    name = "prediction_heads.heads.vision.mlp.0"
    block = {"mode": "on", "layers": {name: 0.2}, "pools": {}, "samples": 8, "revived": [name],
             "watching": {name: {"samples": 0, "needed": None, "ready": False}}}
    off = liveness_card({"liveness": dict(block, revive_rule="none")})["issues"][0]["repair"]
    assert off.startswith("Revival is off (--liveness-rule none)"), off
    default = liveness_card({"liveness": block})["issues"][0]["repair"]
    assert default.startswith("Collecting fresh observations"), default
