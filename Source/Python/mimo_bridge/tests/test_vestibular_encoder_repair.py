# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--vestibular-encoder-repair (mimo_bridge/vestibular_sense.py "the encoder repair";
docs/gain_of_function_v4/agents/vestibular_encoder_2026-09-25.md; its independent review
docs/gain_of_function_v4/agents/vestibular_encoder_review_2026-09-25.md).

What is held here, on the lite model with production's condition put into its vestibular encoder (a
last-layer bias of spread 74,425 and mean -512, the deploy's copied constant as every checkpoint since
344,525 holds it, and Adam moments whose gradient scale sits below eps):
  - the flag defaults off; off stamps nothing, divides nothing and writes no new key into the receipt or the
    runtime state, whatever the checkpoint holds;
  - "rescale" divides the last layer's weight and bias by the largest power of two keeping the spread at or
    above the eps floor (4,096 on production's spread: 18.17 after it), exactly, and moves exactly that
    layer's Adam moments (first x s, second x s squared); every other weight and moment is untouched;
  - every token, and the core's output and every prediction head read from it, stays the same in float64,
    float32 and bf16 (review finding 3: a divisor of spread / 10 moved the bf16 token by 0.01; the check now
    refuses such a divisor);
  - the switch restarts the sense's next-moment normalisation average and holds it out of learning progress
    for 2,000 steps (review findings 1 and 5), and says how far the learning-progress channel moved;
  - one sense's anti-collapse lesson (VICReg at its production weight, 1/6) cannot move the dead token in
    100 Adam updates, and moves the repaired one;
  - the switch happens once per lineage: the receipt says so, and a spread at or under 1,000 is never divided
    (review finding 7: a switched copy whose record was dropped and whose spread drifted switched again);
    a switched checkpoint loads unchanged into a build without the flag and keeps its record; the check puts
    every value back and refuses when the token would change;
  - the load's line says when no optimizer was read (review, minor);
  - the trainer flag defaults off and reaches the learner; the stats window adds its gauges only with it, and counts
    each re-study once;
  - the lesson ceiling and the share gauges read the sense's lesson weight (third review, section D): at a lesson
    weight of 2 the lesson's own value equals the gauge's weights times the plain errors on every step of a real
    body; with the flag off the gauge gets the normalisation's own weights, as before;
  - the balance probe (review finding 4) reads a token that carries the balance state on H84's own layout (one
    body face down), and reads nothing from a token that only tells the bodies apart, one that varies without
    balance, or the dead one; the window feeds it from the frames it reads;
  - (slow) on a real MuJoCo body, a production-sized constant never lets the token vary in 100 training
    steps, and the repair lets it vary (the deploy's own test ran at a constant of size about 14).
"""
import copy
import json
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge import vestibular_sense as VS
from mimo_bridge.lem_model import LemLite

HERE = Path(__file__).resolve()
SRC = HERE.parents[2]                       # this tree's Source/Python

# production's constant, read from the saved weights of every checkpoint since the deploy
# (docs/gain_of_function_v3/measurements/vestibular_encoder_lineage_2026-09-25.json)
PROD_SPREAD = 74425.156176
PROD_MEAN = -511.889836
# the vestibular encoder's last layer at 411,125: weight 0.0097 per number; Adam's sqrt(v) 4.3e-10
PROD_WEIGHT_RMS = 0.0097
PROD_GRAD_SCALE = 4.3e-10
# the divisor production's spread gets: the largest power of two keeping the spread at or above
# sqrt(1e-5 / 2^-23) = 9.16 (74,425.2 / 9.16 = 8,126, so 2^12)
PROD_DIVISOR = 4096.0


def test_the_code_under_test_is_this_trees():
    # the pause runner puts the MAIN tree on PYTHONPATH; this tree's conftest must put its own first
    import mimo_bridge.lem_model as LM
    for mod in (VS, LM):
        assert Path(mod.__file__).resolve().is_relative_to(SRC), mod.__file__


def _lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   top_down_gate=True, body_layout="trunk_v1", vestibular_sense=True, **kw)


def _last(m):
    return m.encoder_registry.encoders["vestibular"].mlp[-1]


def _dead(m, seed=3, spread=PROD_SPREAD):
    """Production's condition in the lite model's vestibular encoder (see the module note)."""
    last = _last(m)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        v = torch.randn(last.bias.shape, generator=g, dtype=torch.float64)
        v = (v - v.mean()) / v.std(unbiased=False) * spread + PROD_MEAN
        last.bias.copy_(v.float())
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * PROD_WEIGHT_RMS)
    for p in m.params:                        # every tensor carries a saved Adam state, as in production
        m.optimizer.state[p] = {"step": torch.tensor(66725.0),
                                "exp_avg": torch.randn(p.shape, generator=g) * PROD_GRAD_SCALE * 0.1,
                                "exp_avg_sq": (torch.rand(p.shape, generator=g) * PROD_GRAD_SCALE) ** 2}
    return last


def _dead_checkpoint(tmp_path, tag="ck", **kw):
    m = _lite(checkpoint_dir=str(tmp_path / tag), **kw)
    _dead(m)
    return m, m.save_checkpoint(step=411125)


def _inputs(n=8, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, VS.ENCODER_DIM, generator=g).clamp(-VS.NORMALISED_BOUND, VS.NORMALISED_BOUND)


def _token(m, z):
    return m.encoder_registry.encode({"vestibular": z}, {"vestibular": "real"})["vestibular"].squeeze(1)


def _receipt(path):
    with open(os.path.join(path, "model_receipt.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------ the flag
def test_the_flag_defaults_off_and_stamps_nothing():
    off = _lite()
    assert off.vestibular_encoder_repair == "off" and off.vestibular_encoder_repair_record is None
    assert "vestibular_encoder_repair" not in off.receipt
    on = _lite(vestibular_encoder_repair="rescale")
    st = on.receipt["vestibular_encoder_repair"]
    assert st["mode"] == "rescale" and st["switch_line"] == VS.REPAIR_SWITCH_LINE == 1000.0
    assert st["version"] == VS.REPAIR_VERSION and st["check_precisions"] == ["float64", "float32", "bfloat16"]
    assert st["lp_holdout_steps"] == VS.REPAIR_LP_HOLDOUT_STEPS == 2000
    # the repair adds no parameter and no module: the two builds hold the same tensors
    assert [tuple(p.shape) for p in off.params] == [tuple(p.shape) for p in on.params]
    with pytest.raises(ValueError, match="must be one of"):
        _lite(vestibular_encoder_repair="maybe")
    with pytest.raises(ValueError, match="--vestibular-sense on"):
        LemLite(device="cpu", seed=0, imagination=False, no_vision=True, body_layout="trunk_v1",
                vestibular_encoder_repair="rescale")


def test_the_divisor_is_the_largest_power_of_two_above_the_eps_floor():
    floor = VS.repair_spread_floor(1e-5)
    assert floor == pytest.approx(math.sqrt(1e-5 / 2.0 ** -23)) and 9.15 < floor < 9.17
    assert VS.repair_divisor(PROD_SPREAD, 1e-5) == PROD_DIVISOR
    after = PROD_SPREAD / PROD_DIVISOR
    assert floor <= after < 2 * floor and after == pytest.approx(18.17, abs=0.01)
    # the eps term at that spread: under half a float32 step; at 8,192 it would be just past it
    assert 1e-5 / (2 * after ** 2) < 2.0 ** -24 < 1e-5 / (2 * (PROD_SPREAD / 8192.0) ** 2)
    for spread in (9.2 * 2.0, 1000.5, 5e5, 3.3e7):
        s = VS.repair_divisor(spread, 1e-5)
        assert math.log2(s) == int(math.log2(s)) and floor <= spread / s < 2 * floor
    assert VS.repair_divisor(15.0, 1e-5) == 1.0


def test_a_default_load_changes_nothing_and_writes_nothing_new(tmp_path):
    m, path = _dead_checkpoint(tmp_path, next_frame_norm="persistence", learning_progress=True)
    a = _lite(checkpoint_dir=str(tmp_path / "a"), next_frame_norm="persistence", learning_progress=True)
    a.load_checkpoint(path, load_optimizer=True)
    for x, y in ((_last(a).weight, _last(m).weight), (_last(a).bias, _last(m).bias)):
        assert torch.equal(x, y)
    for p_a, p_m in zip(a.params, m.params):
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(a.optimizer.state[p_a][k], m.optimizer.state[p_m][k])
    assert a.vestibular_encoder_repair_record is None
    assert "vestibular_encoder_repair" not in a.receipt
    assert a.wellness_channels.lp.holdout == {}
    z = _inputs()
    assert torch.equal(_token(a, z), _token(m, z))
    # what it saves carries no new key, in the receipt or in the runtime state
    p2 = a.save_checkpoint(step=411126)
    assert "vestibular_encoder_repair" not in _receipt(p2)
    assert set(_receipt(p2)) == set(_receipt(path))
    rt1 = torch.load(os.path.join(path, "runtime_state.pt"), weights_only=False)
    rt2 = torch.load(os.path.join(p2, "runtime_state.pt"), weights_only=False)
    assert set(rt1) == set(rt2)


# ------------------------------------------------------------------------------ the switch
def test_the_switch_divides_the_layer_exactly_and_moves_only_its_moments(tmp_path):
    m, path = _dead_checkpoint(tmp_path)
    a = _lite()
    a.load_checkpoint(path, load_optimizer=True)
    r = _lite(vestibular_encoder_repair="rescale")
    r.load_checkpoint(path, load_optimizer=True)
    rec = r.vestibular_encoder_repair_record
    assert rec is not None and rec["switched"] and rec["step"] == 411125
    s = PROD_DIVISOR
    assert rec["divided_by"] == s
    assert rec["bias_spread_before"] == pytest.approx(PROD_SPREAD, rel=1e-6)
    assert rec["bias_spread_after"] == pytest.approx(PROD_SPREAD / s, rel=1e-6)
    lm, lr = _last(m), _last(r)
    # a power of two changes no rounding: the float32 division is exact
    assert torch.equal(lr.weight, lm.weight / s) and torch.equal(lr.bias, lm.bias / s)
    assert torch.equal(lr.weight * s, lm.weight) and torch.equal(lr.bias * s, lm.bias)
    for pm_, pr_ in ((lm.weight, lr.weight), (lm.bias, lr.bias)):
        sm, sr = m.optimizer.state[pm_], r.optimizer.state[pr_]
        assert torch.equal(sr["exp_avg"], sm["exp_avg"] * s)
        assert torch.equal(sr["exp_avg_sq"], sm["exp_avg_sq"] * (s * s))
        assert float(sr["step"]) == float(sm["step"])
    n_last = len(r.encoder_registry.encoders["vestibular"].mlp) - 1
    assert sorted(rec["moments_moved"]) == sorted(["mlp.%d.weight" % n_last, "mlp.%d.bias" % n_last])
    # every other parameter and every other Adam state is the default load's
    moved = {id(lr.weight), id(lr.bias)}
    n_other = 0
    for p_a, p_r in zip(a.params, r.params):
        if id(p_r) in moved:
            continue
        assert torch.equal(p_a, p_r)
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(a.optimizer.state[p_a][k], r.optimizer.state[p_r][k])
        n_other += 1
    assert n_other == len(r.params) - 2
    assert r.receipt["vestibular_encoder_repair"]["switched"]["divided_by"] == s


def _bf16_token(m, z):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        t = m.encoder_registry.encode({"vestibular": z}, {"vestibular": "real"})["vestibular"]
    return t.squeeze(1).float()


def test_the_switch_keeps_every_token_and_every_prediction_in_every_precision(tmp_path):
    m, path = _dead_checkpoint(tmp_path)
    a = _lite()
    a.load_checkpoint(path, load_optimizer=False)
    r = _lite(vestibular_encoder_repair="rescale")
    r.load_checkpoint(path, load_optimizer=False)
    checks = r.vestibular_encoder_repair_record["checks"]
    assert set(checks) == {"float64", "float32", "bfloat16"} and all(c["within"] for c in checks.values())
    z = _inputs(64)
    # float64 from the float32 parameters: the switch's own change, the eps term at a spread of 18.17
    # (1.5e-8 of a value, the largest value here about 3)
    t64a, t64r = VS._probe_tokens(a, z), VS._probe_tokens(r, z)
    assert float((t64a - t64r).abs().max()) <= 2e-7
    # float32 and bf16, as the run computes them: within a few float32 steps of the largest value
    for fa, fr in ((_token(a, z), _token(r, z)), (_bf16_token(a, z), _bf16_token(r, z))):
        largest = float(fa.abs().max())
        assert float((fa - fr).abs().max()) <= 8 * 2 ** -23 * max(1.0, largest)
    # the dead token: the weights' part is a millionth of it (production's 3.6e-7 per number)
    ta = _token(a, z)
    assert float((ta - ta.mean(0)).square().mean().sqrt()) < 1e-5
    # the core and every prediction head, with the other senses' tokens shared and each model's own vestibular token
    B = 8
    g = torch.Generator().manual_seed(7)
    toks = {mm: torch.randn(B, 1, a.d_model, generator=g) for mm in a.active}
    toks = {mm: torch.nn.functional.layer_norm(t, (a.d_model,)) for mm, t in toks.items()}
    ctx = torch.randn(B, (a.ctx_len - 1) * a.n_mod, a.d_model, generator=g)
    outs = []
    for model in (a, r):
        tk = dict(toks)
        tk["vestibular"] = _token(model, z[:B]).unsqueeze(1)
        with torch.no_grad():
            lat, last, _ = model.forward_model(tk, ctx)
            pred = model.prediction_heads(last, current={mm: v.squeeze(1) for mm, v in tk.items()}, who="test")
        outs.append((lat, pred))
    (la, pa), (lr_, pr) = outs
    assert float((la - lr_).abs().max()) <= 1e-5
    for mm in pa:
        assert float((pa[mm] - pr[mm]).abs().max()) <= 1e-5, mm


def test_a_divisor_that_is_not_a_power_of_two_is_refused_by_the_bf16_check(tmp_path, monkeypatch):
    # review finding 3: spread / 10 kept the token in float64 and float32 and moved it by 0.01 in bf16
    m, path = _dead_checkpoint(tmp_path)
    a = _lite()
    a.load_checkpoint(path, load_optimizer=True)
    w0, b0 = _last(a).weight.clone(), _last(a).bias.clone()
    monkeypatch.setattr(VS, "repair_divisor", lambda spread, eps: spread / 10.0)
    with pytest.raises(ValueError, match="bfloat16"):
        VS.repair_encoder(a)
    assert torch.equal(_last(a).weight, w0) and torch.equal(_last(a).bias, b0)


def _vicreg_updates(model, z, n):
    """n Adam updates of the vestibular encoder alone, on its anti-collapse lesson at production's weight (the
    VICReg term averages six real senses, so one sense's share is 1/6), with the loaded Adam states. Returns the
    token's spread between the bodies (root mean square over its numbers) after each update."""
    enc = model.encoder_registry.encoders["vestibular"]
    decay = [p for p in enc.parameters() if p.dim() >= 2]
    no_decay = [p for p in enc.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    for p in enc.parameters():
        st = model.optimizer.state.get(p)
        if st:
            opt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    spreads = []
    for _ in range(n):
        opt.zero_grad()
        t = _token(model, z)
        loss = model.loss_manager.compute_vicreg({"vestibular": t}) / 6.0
        loss.backward()
        opt.step()
        with torch.no_grad():
            t = _token(model, z)
        spreads.append(float(t.std(0, unbiased=False).square().mean().sqrt()))
    return spreads


def test_learning_moves_the_token_only_after_the_switch(tmp_path):
    m, path = _dead_checkpoint(tmp_path)
    a = _lite()
    a.load_checkpoint(path, load_optimizer=True)
    r = _lite(vestibular_encoder_repair="rescale")
    r.load_checkpoint(path, load_optimizer=True)
    rr = _lite(vestibular_encoder_repair="rescale_restart")
    rr.load_checkpoint(path, load_optimizer=True)
    z = _inputs(8)
    sa = _vicreg_updates(a, z, 100)
    sr = _vicreg_updates(r, z, 100)
    srr = _vicreg_updates(rr, z, 100)
    # as is: the lesson's gradient on the last layer is about 3e-9 (below Adam's eps), so its updates reach a
    # quarter of the learning rate at most and the weights' part reaches the token divided by 74,425: about
    # 1e-5 after 100 updates (measured 2026-09-25: 8.9e-6). The line is 10 times that, and 1,500 times below
    # the smallest spread any live sense shows between bodies (0.15, the memory's saved events)
    assert max(sa) < 1e-4, max(sa)
    # repaired (spread 18.17): the same lesson spreads it into the live senses' band (0.15-0.53)
    for s in (sr, srr):
        assert s[-1] > 0.05 and s[-1] > 1000 * sa[-1], (s[-1], sa[-1])


def test_the_restart_mode_divides_the_same_and_drops_only_the_encoders_adam_state(tmp_path):
    m, path = _dead_checkpoint(tmp_path)
    r = _lite(vestibular_encoder_repair="rescale")
    r.load_checkpoint(path, load_optimizer=True)
    rr = _lite(vestibular_encoder_repair="rescale_restart")
    rr.load_checkpoint(path, load_optimizer=True)
    rec = rr.vestibular_encoder_repair_record
    assert rec["switched"] and rec["moments_moved"] == [] and rec["moment_factors"] is None
    enc_r, enc_rr = r.encoder_registry.encoders["vestibular"], rr.encoder_registry.encoders["vestibular"]
    assert sorted(rec["moments_restarted"]) == sorted(n for n, _ in enc_rr.named_parameters())
    for (n, p), (_, q) in zip(enc_r.named_parameters(), enc_rr.named_parameters()):
        assert torch.equal(p, q), n                          # the same division
        assert not rr.optimizer.state.get(q), n             # Adam restarts every encoder tensor
    mine = {id(q) for q in enc_rr.parameters()}
    for p_r, p_rr in zip(r.params, rr.params):
        if id(p_rr) in mine:
            continue
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(r.optimizer.state[p_r][k], rr.optimizer.state[p_rr][k])
    assert "Adam restarts them" in VS.repair_message(rec, 411125)
    z = _inputs(8)
    assert torch.equal(_token(r, z), _token(rr, z))


def test_the_load_line_says_when_no_optimizer_was_read(tmp_path, caplog):
    # review, minor: with no optimizer loaded the line said moments were moved (first x s ...)
    m, path = _dead_checkpoint(tmp_path)
    caplog.set_level(logging.WARNING)
    for mode in ("rescale", "rescale_restart"):
        caplog.clear()
        r = _lite(vestibular_encoder_repair=mode)
        r.load_checkpoint(path, load_optimizer=False)
        rec = r.vestibular_encoder_repair_record
        assert rec["moments_moved"] == [] and rec["moments_restarted"] == [] and rec["moment_factors"] is None
        msg = [x.getMessage() for x in caplog.records if "vestibular encoder repair switched" in x.getMessage()]
        assert msg and "this load read no optimizer" in msg[-1] and "first x" not in msg[-1]


def _lp_state(B, **per_key):
    """Learning progress's saved state, one body batch: {key: (mf, pf, ms, ps)} averages, weights filled."""
    out = {}
    for key, (mf, pf, ms, ps) in per_key.items():
        out[key] = {"mf": torch.full((B,), mf), "pf": torch.full((B,), pf), "wf": torch.ones(B),
                    "ms": torch.full((B,), ms), "ps": torch.full((B,), ps), "ws": torch.ones(B)}
    return {"state": out, "af": 1 / 200, "as": 1 / 2000, "holdout": {}}


def test_the_switch_restarts_the_next_moment_average_and_holds_out_learning_progress(tmp_path, caplog):
    # review findings 1 and 5: both averages were taken on the dead token
    kw = dict(next_frame_norm="persistence", learning_progress=True)
    m = _lite(checkpoint_dir=str(tmp_path / "ck"), **kw)
    _dead(m)
    B = 8
    # the saved state of ckpt_step_411150 in shape: the vestibular averages dead, the others alive
    m.nf_norm.load_state_dict({"stats": {"vestibular": [1.1e-13, 1.0, 66723], "audio": [0.0465, 1.0, 66724],
                                          "tactile": [0.017, 1.0, 66724], "vision": [0.0061, 1.0, 66724],
                                          "proprioception": [0.0025, 1.0, 66724]}})
    m.wellness_channels.lp.load_state_dict(_lp_state(
        B, vestibular=(1.15e-5, 1.2e-20, 5.0e-6, 1.1e-13), audio=(0.05, 0.05, 0.06, 0.05),
        tactile=(0.015, 0.017, 0.016, 0.017)))
    before = [float(x) for x in m.wellness_channels.lp.channel(B)]
    path = m.save_checkpoint(step=411125)
    caplog.set_level(logging.WARNING)
    r = _lite(checkpoint_dir=str(tmp_path / "r"), vestibular_encoder_repair="rescale_restart", **kw)
    r.load_checkpoint(path, load_optimizer=True)
    assert "vestibular" not in r.nf_norm.stats and "audio" in r.nf_norm.stats
    assert r.wellness_channels.lp.holdout == {"vestibular": 2000} and "vestibular" not in r.wellness_channels.lp.state
    done = r.vestibular_encoder_repair_record["runtime_restart"]
    assert done["next_frame_norm"]["dropped_readings"] == 66723
    assert done["next_frame_norm"]["dropped_average"] == pytest.approx(1.1e-13)
    lp = done["learning_progress"]
    # the channel is the mean over the keys with a value: before, 3 keys with the vestibular term at -1; after,
    # the other two
    after = [float(x) for x in r.wellness_channels.lp.channel(B)]
    assert lp["channel_before"] == pytest.approx(before) and lp["channel_after"] == pytest.approx(after)
    assert lp["channel_step_per_body"] == pytest.approx([a - b for a, b in zip(after, before)], abs=1e-6)
    assert before[0] < after[0]
    assert r.receipt["vestibular_encoder_repair"]["switched"]["runtime_restart"]["learning_progress"]["held_out_steps"] == 2000
    assert any("learning progress holds it out for 2000" in x.getMessage() for x in caplog.records)
    # saved and loaded again with the flag: no second switch, no second restart; the fresh average survives
    r.nf_norm.update({"vestibular": 0.003})
    p2 = r.save_checkpoint(step=411200)
    r2 = _lite(vestibular_encoder_repair="rescale_restart", **kw)
    r2.load_checkpoint(p2, load_optimizer=True)
    assert r2.vestibular_encoder_repair_record is None
    assert r2.nf_norm.scale("vestibular") == pytest.approx(0.003)
    assert r2.wellness_channels.lp.holdout == {"vestibular": 2000}


def test_the_repair_in_the_same_load_as_the_deploys_switch_keeps_the_old_token(tmp_path):
    # an intermind checkpoint whose placeholder encoder outputs one large constant (its output projection
    # grown, as H84's grew to a spread of 74,425), loaded as the deploy loaded it and with the repair
    torch.manual_seed(21)
    off = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                  top_down_gate=True, body_layout="trunk_v1", checkpoint_dir=str(tmp_path / "ck"))
    enc = off.encoder_registry.encoders["intermind"]
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        enc.embed.weight.zero_()
        b = enc.layers[0]["ffn_norm"].bias
        b.copy_(torch.randn(b.shape, generator=g))
        enc.pool_attn.out_proj.weight.mul_(20000.0)
    path = off.save_checkpoint(step=5)
    deploy = _lite()
    deploy.load_checkpoint(path, load_optimizer=False)
    fixed = _lite(vestibular_encoder_repair="rescale")
    fixed.load_checkpoint(path, load_optimizer=False)
    assert deploy.vestibular_switch is not None and fixed.vestibular_switch is not None
    s_deploy = float(_last(deploy).bias.double().std(unbiased=False))
    s_fixed = float(_last(fixed).bias.double().std(unbiased=False))
    floor = VS.repair_spread_floor(1e-5)
    assert s_deploy > 1000.0 and floor <= s_fixed < 2 * floor
    assert fixed.vestibular_encoder_repair_record["divided_by"] == VS.repair_divisor(s_deploy, 1e-5)
    ph = off.placeholders._generators["intermind"].generate(4).clone()
    old = off.encoder_registry.encode({"intermind": ph}, {"intermind": "synthetic"})["intermind"].squeeze(1)
    z = _inputs(4)
    assert torch.allclose(_token(deploy, z), old, atol=1e-5)
    assert torch.allclose(_token(fixed, z), old, atol=1e-5)


# ------------------------------------------------------------------------------ once, and the other cases
def test_the_switch_happens_once_and_a_default_build_keeps_its_record(tmp_path, caplog):
    m, path = _dead_checkpoint(tmp_path)
    r = _lite(vestibular_encoder_repair="rescale", checkpoint_dir=str(tmp_path / "r"))
    r.load_checkpoint(path, load_optimizer=True)
    p2 = r.save_checkpoint(step=411200)
    assert _receipt(p2)["vestibular_encoder_repair"]["switched"]["divided_by"] == PROD_DIVISOR
    r2 = _lite(vestibular_encoder_repair="rescale")
    r2.load_checkpoint(p2, load_optimizer=True)
    assert r2.vestibular_encoder_repair_record is None
    assert torch.equal(_last(r2).weight, _last(r).weight) and torch.equal(_last(r2).bias, _last(r).bias)
    assert r2.receipt["vestibular_encoder_repair"]["switched"]["step"] == 411125
    for p, q in ((_last(r2).weight, _last(r).weight), (_last(r2).bias, _last(r).bias)):
        assert torch.equal(r2.optimizer.state[p]["exp_avg_sq"], r.optimizer.state[q]["exp_avg_sq"])
    caplog.set_level(logging.WARNING)
    a2 = _lite(checkpoint_dir=str(tmp_path / "a2"))
    a2.load_checkpoint(p2, load_optimizer=True)
    assert torch.equal(_last(a2).bias, _last(r).bias)
    assert a2.receipt["vestibular_encoder_repair"]["switched"]["step"] == 411125
    assert a2.receipt["vestibular_encoder_repair"]["carried_by_build_mode"] == "off"
    assert any("does not undo the switch" in rr.getMessage() for rr in caplog.records)
    # the record rides on into the default build's next checkpoint
    p3 = a2.save_checkpoint(step=411300)
    assert _receipt(p3)["vestibular_encoder_repair"]["switched"]["step"] == 411125


def test_a_switched_checkpoint_whose_record_was_dropped_is_not_switched_again(tmp_path):
    # review finding 7: production's code resumed a switched checkpoint and wrote one without the record; its
    # spread had drifted (x 1.0016), and a flagged load divided again and restarted Adam
    m, path = _dead_checkpoint(tmp_path)
    r = _lite(vestibular_encoder_repair="rescale_restart", checkpoint_dir=str(tmp_path / "r"))
    r.load_checkpoint(path, load_optimizer=True)
    with torch.no_grad():
        _last(r).bias.mul_(1.0016)
    p2 = r.save_checkpoint(step=411200)
    rp = os.path.join(p2, "model_receipt.json")
    rec = _receipt(p2)
    rec.pop("vestibular_encoder_repair")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    again = _lite(vestibular_encoder_repair="rescale_restart")
    again.load_checkpoint(p2, load_optimizer=True)
    assert again.vestibular_encoder_repair_record is None
    assert again.receipt["vestibular_encoder_repair"]["not_needed"]["switched"] is False
    assert torch.equal(_last(again).bias, _last(r).bias)


def test_a_bias_at_or_under_the_switch_line_is_left_alone(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    m = _lite(checkpoint_dir=str(tmp_path / "ck"))
    _dead(m, spread=500.0)                      # held down 27 times more than at the floor, under the line
    path = m.save_checkpoint(step=5)
    r = _lite(vestibular_encoder_repair="rescale")
    r.load_checkpoint(path, load_optimizer=True)
    assert r.vestibular_encoder_repair_record is None
    assert torch.equal(_last(r).bias, _last(m).bias) and torch.equal(_last(r).weight, _last(m).weight)
    nn_ = r.receipt["vestibular_encoder_repair"]["not_needed"]
    assert nn_["switched"] is False and "not above the switch line 1000" in nn_["why"]
    warned = [x for x in caplog.records if x.levelno == logging.WARNING and "not above the switch line" in x.getMessage()]
    assert warned and "divided by 500" in warned[-1].getMessage()
    # a fresh encoder (the deploy's own start, before its constant was copied in) is left alone quietly
    fresh = _lite(vestibular_encoder_repair="rescale", checkpoint_dir=str(tmp_path / "f"))
    rec = VS.repair_encoder(fresh)
    assert rec["switched"] is False and rec["bias_spread_before"] < 1.0


def test_the_check_puts_every_value_back_and_refuses(tmp_path, monkeypatch):
    m, path = _dead_checkpoint(tmp_path)
    a = _lite()
    a.load_checkpoint(path, load_optimizer=True)
    last = _last(a)
    w0, b0 = last.weight.clone(), last.bias.clone()
    m0 = {k: v.clone() for k, v in a.optimizer.state[last.weight].items() if torch.is_tensor(v)}
    monkeypatch.setattr(VS, "REPAIR_CHECK_LINE", 0.0)
    with pytest.raises(ValueError, match="refused"):
        VS.repair_encoder(a)
    assert torch.equal(last.weight, w0) and torch.equal(last.bias, b0)
    for k, v in m0.items():
        assert torch.equal(a.optimizer.state[last.weight][k], v)
    monkeypatch.setattr(VS, "REPAIR_CHECK_LINE", 2e-5)
    # no scale-free LayerNorm after the encoders: dividing would change every token, so nothing is divided
    a.encoder_registry.output_norm = None
    with pytest.raises(ValueError, match="no LayerNorm"):
        VS.repair_encoder(a)
    assert torch.equal(last.bias, b0)


# ------------------------------------------------------------------------------ the trainer and the stats
def test_the_trainer_flag_defaults_off_and_reaches_the_learner(monkeypatch):
    import inspect
    from mimo_bridge import train_mimo as TM
    d = TM.parse_args(["--out", "x"])
    assert d.vestibular_encoder_repair == "off"
    a = TM.parse_args(["--out", "x", "--vestibular-sense", "on", "--vestibular-encoder-repair", "rescale"])
    assert a.vestibular_encoder_repair == "rescale"
    b = TM.parse_args(["--out", "x", "--vestibular-sense", "on", "--vestibular-encoder-repair",
                       "rescale_restart"])
    assert b.vestibular_encoder_repair == "rescale_restart"
    c = TM.parse_args(["--out", "x", "--vestibular-sense", "on", "--vestibular-encoder-repair",
                       "rescale_restart_fp32"])
    assert c.vestibular_encoder_repair == "rescale_restart_fp32"
    with pytest.raises(SystemExit):
        TM.parse_args(["--out", "x", "--vestibular-encoder-repair", "sometimes"])
    # it changes what the model learns, so a frozen actor's qualification binds it; it is not a physical
    # continuation flag, so a resume may switch it on
    assert TM.acting_configuration(a)["vestibular_encoder_repair"] == "rescale"
    src = inspect.getsource(TM.main)
    assert "vestibular_encoder_repair" not in src[src.index("continuation_flags = ("):src.index("continuation_config = ")]
    seen = {}

    class Stub:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr(TM, "LemLite", Stub)
    TM.build_learner(a, None)
    assert seen["vestibular_encoder_repair"] == "rescale"
    seen.clear()
    TM.build_learner(d, None)
    assert seen["vestibular_encoder_repair"] == "off"
    # an args.json written before the flag existed builds today's model
    old = SimpleNamespace(**{k: v for k, v in vars(d).items() if k != "vestibular_encoder_repair"})
    seen.clear()
    TM.build_learner(old, None)
    assert seen["vestibular_encoder_repair"] == "off"


# ------------------------------------------------------------------------------ the float32 mode (review 2, finding 4)
def test_the_float32_mode_computes_the_encoder_in_float32_and_saves_nothing():
    on = _lite(vestibular_encoder_repair="rescale_restart_fp32")
    off = _lite(vestibular_encoder_repair="rescale_restart")
    enc_on, enc_off = on.encoder_registry.encoders["vestibular"], off.encoder_registry.encoders["vestibular"]
    assert getattr(enc_on, "vestibular_repair_fp32", False) and "forward" in enc_on.__dict__
    assert "forward" not in enc_off.__dict__ and not getattr(enc_off, "vestibular_repair_fp32", False)
    assert on.receipt["vestibular_encoder_repair"]["encoder_precision"] == "float32"
    assert off.receipt["vestibular_encoder_repair"]["encoder_precision"] == "the run's"
    assert list(on.encoder_registry.state_dict()) == list(off.encoder_registry.state_dict())
    for m in (on, off):
        _dead(m)
        VS.repair_encoder(m, load_optimizer=True, restart_moments=True)
    z = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        t_on, t_off = _token(on, z).float(), _token(off, z).float()
    # in the run's bf16 the divided dead token is one value in every body (the anti-collapse lesson then has no
    # direction); with this encoder in float32 it differs between bodies as in float32
    assert bool((t_off == t_off[0:1]).all())
    assert not bool((t_on == t_on[0:1]).all())
    assert torch.allclose(t_on.double(), _token(on, z).double(), atol=1e-6)
    # a deep copy (the episode companion, the measurement's arms) rebinds the float32 forward to the copy
    enc2 = copy.deepcopy(enc_on)
    assert enc2.forward.__self__ is enc2


def test_the_float32_mode_switches_and_restarts_like_rescale_restart(tmp_path):
    m, path = _dead_checkpoint(tmp_path)
    fp = _lite(vestibular_encoder_repair="rescale_restart_fp32")
    fp.load_checkpoint(path, load_optimizer=True)
    rec = fp.vestibular_encoder_repair_record
    assert rec["switched"] and rec["divided_by"] == PROD_DIVISOR and rec["moments_restarted"]
    assert getattr(fp.encoder_registry.encoders["vestibular"], "vestibular_repair_fp32", False)


# ------------------------------------------------------------------------------ the lesson ceiling (review 2, finding 2)
def test_the_revival_ceiling_holds_the_sense_to_the_other_senses_mean():
    w = {"vision": 2.4, "audio": 0.31, "vestibular": 30.7}
    e = {"vision": 0.006, "audio": 0.05, "vestibular": 0.0055}
    got, f = VS.revival_ceiling(w, e, ["vestibular"])
    mean_other = (2.4 * 0.006 + 0.31 * 0.05) / 2
    assert f == pytest.approx(mean_other / (30.7 * 0.0055))
    assert got["vestibular"] * e["vestibular"] == pytest.approx(mean_other)
    assert got["vision"] == 2.4 and got["audio"] == 0.31 and w["vestibular"] == 30.7
    assert VS.revival_ceiling(w, e, [])[1] is None                    # the weight is not at its cap
    assert VS.revival_ceiling(w, dict(e, vestibular=1e-6), ["vestibular"])[1] == 1.0   # already under
    # the replay form stays on the device and keeps the gradient's direction at the reduced size
    t = [torch.tensor(0.006, requires_grad=True), torch.tensor(0.05, requires_grad=True),
         torch.tensor(0.0055, requires_grad=True)]
    weighted, factor = VS.revival_ceiling_terms(["vision", "audio", "vestibular"], t, w, ["vestibular"])
    assert float(weighted[2]) == pytest.approx(mean_other)
    sum(weighted).backward()
    assert float(t[2].grad) == pytest.approx(30.7 * float(factor)) and float(t[0].grad) == pytest.approx(2.4)
    assert VS.revival_ceiling_terms(["vision", "vestibular"], t[:2], w, [])[1] is None


def test_the_ceiling_and_its_gauge_read_the_lesson_weight_actually_used():
    # third review, section D, the reviewer's plant (vrev3/plant_weight.py) through the production classes: H84's
    # four other senses at the saved averages of 411,150, this sense's average restarted (its weight at the cap), its
    # head's error 11.7 times the capped divisor. Before: at a lesson weight of 2 the optimizer saw 33 in 100 while
    # the gauge said 20; at 3, 43 against 20.
    from lem.model.next_frame_norm import NextFrameNorm
    others = {"audio": 0.0465, "tactile": 0.017, "vision": 0.0061, "proprioception": 0.0025}
    for lw in (0.5, 1.0, 2.0, 3.0):
        nf = NextFrameNorm()
        nf.load_state_dict({"stats": {k: [v, 1.0, 66724] for k, v in others.items()}})
        graded = list(others) + ["vestibular"]
        w, info = nf.weights(graded, dict(others, vestibular=1e-7))
        err = dict(others, vestibular=11.7 * info["floor"])
        w2, factor = VS.revival_ceiling(w, err, info["capped"], lesson_weight=lw)
        # the lesson the optimizer sees: the weights after the ceiling, this sense's term times its lesson weight
        terms = {m: w2[m] * err[m] * (lw if m == "vestibular" else 1.0) for m in graded}
        seen = terms["vestibular"] / sum(terms.values())
        nf.add_live(err, VS.lesson_weights_used(w2, lw), info)
        reported = nf.summary()["share_normalized"]["vestibular"]
        assert reported == pytest.approx(seen, rel=1e-4), (lw, seen, reported)
        assert seen <= 1.0 / len(graded) + 1e-9, (lw, seen)
    # the replay form carries the lesson weight into the term it judges
    t = [torch.tensor(0.006), torch.tensor(0.05), torch.tensor(0.0055)]
    w = {"vision": 2.4, "audio": 0.31, "vestibular": 30.7}
    weighted, factor = VS.revival_ceiling_terms(["vision", "audio", "vestibular"], t, w, ["vestibular"],
                                                lesson_weight=2.0)
    assert float(weighted[2]) == pytest.approx((2.4 * 0.006 + 0.31 * 0.05) / 2)
    assert float(factor) == pytest.approx(((2.4 * 0.006 + 0.31 * 0.05) / 2) / (2.0 * 30.7 * 0.0055))
    unc, none = VS.revival_ceiling_terms(["vision", "audio", "vestibular"], t, w, [], lesson_weight=2.0)
    assert none is None and float(unc[2]) == pytest.approx(2.0 * 30.7 * 0.0055)
    assert VS.lesson_weights_used(w, 1.0) is w


def test_at_a_lesson_weight_of_2_the_gauge_reads_the_lesson_the_optimizer_sees(tmp_path):
    torch.manual_seed(43)
    base = _live_lite(checkpoint_dir=str(tmp_path / "ck"), next_frame_norm="persistence")
    _dead(base)
    path = base.save_checkpoint(step=5)
    m = _live_lite(next_frame_norm="persistence", vestibular_encoder_repair="rescale_restart",
                   vestibular_lesson_weight=2.0)
    m.load_checkpoint(path, load_optimizer=False)
    m.enable_replay(1, size_per_env=64)
    seen = []
    live_add, replay_add = m.nf_norm.add_live, m.nf_norm.add_replay
    m.nf_norm.add_live = lambda raw, weights, info: (seen.append(("live", dict(raw), dict(weights))),
                                                     live_add(raw, weights, info))[1]
    m.nf_norm.add_replay = lambda raw, weights: (seen.append(("replay", dict(raw), dict(weights))),
                                                 replay_add(raw, weights))[1]
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "live"), tag="_live", vision=False, seed=5)
    checked = 0
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(24):
            n_before = len(seen)
            r = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            lives = [s for s in seen[n_before:] if s[0] == "live"]
            nf_value = (r.breakdown or {}).get("next_frame")
            if lives and nf_value is not None and "vestibular" in lives[-1][1]:
                _kind, raw, weights = lives[-1]
                # the lesson's own value is the sum of the gauge's weights times the plain errors: the gauge reads
                # the weight the optimizer sees, the lesson weight 2 in it
                assert float(nf_value) == pytest.approx(sum(weights[k] * raw[k] for k in raw), rel=2e-4)
                checked += 1
            if i % 5 == 0 or held is None:
                held = r.motor_for_action
            for _ in range(5):
                frames = src.step(held, r.gaze_out, None, render=False)
            prev = held
    finally:
        src.close()
    assert checked >= 5, checked
    live = m.next_frame_norm_window_summary()
    others = [s for s in live["share_normalized"] if s != "vestibular"]
    assert live["share_normalized"]["vestibular"] <= 1.0 / (len(others) + 1) + 1e-6, live
    for _ in range(6):
        m.replay_update(batch=4)
    reps = [s for s in seen if s[0] == "replay" and "vestibular" in s[1]]
    assert reps
    # the newest re-study's gauge weight carries the lesson weight: the running weight times 2 times the ceiling's
    # factor (1 when it did not apply)
    _kind, raw, weights = reps[-1]
    w_nf, _info = m.nf_norm.weights(list(raw), None)
    f = m.vestibular_revival_replay if m.vestibular_revival_replay is not None else 1.0
    assert weights["vestibular"] == pytest.approx(w_nf["vestibular"] * 2.0 * f, rel=1e-4)


def test_with_the_flag_off_the_lesson_and_its_gauge_are_the_previous_codes(tmp_path):
    # the default build's live lesson hands the gauge the very weights the normalisation gave (the lesson weight
    # left out of the gauge as before, whatever it is), and never runs the ceiling
    torch.manual_seed(44)
    m = _live_lite(next_frame_norm="persistence", vestibular_lesson_weight=2.0)
    got, given = [], []
    weights, add_live = m.nf_norm.weights, m.nf_norm.add_live

    def w(*a, **kw):
        out = weights(*a, **kw)
        given.append(out[0])
        return out
    m.nf_norm.weights = w
    m.nf_norm.add_live = lambda raw, ww, info: (got.append(ww), add_live(raw, ww, info))[1]
    _drive_body(m, tmp_path, 8, "off")
    assert got and all(any(g is x for x in given) for g in got)
    assert m.vestibular_revival_live is None


def test_the_window_counts_each_replay_once():
    # third review, minor: replay runs every 2 live steps and its factor stayed set, so the window counted it twice
    on = _lite(vestibular_encoder_repair="rescale")
    on._last_source_mask = {"vestibular": "real"}
    on.last_prediction_audit = {}
    win = VS.VestibularWindow()
    g = torch.Generator().manual_seed(1)
    for step in range(6):
        on.last_tokens = {"vestibular": torch.randn(3, on.d_model, generator=g)}
        on.vestibular_revival_live = 0.5
        if step % 2 == 0:
            on.vestibular_revival_replay = 0.25                  # a re-study ran before this step's window read
        win.add(_frames(3), on)
    rep = win.summary()["repair"]
    assert rep["revival_ceiling"]["live"]["steps_applied"] == 6
    assert rep["revival_ceiling"]["replay"]["steps_applied"] == 3 and rep["lesson_weight"] == 1.0


def _drive_body(m, tmp_path, n, tag):
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / tag), tag="_" + tag, vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(n):
            r = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i % 5 == 0 or held is None:
                held = r.motor_for_action
            for _ in range(5):
                frames = src.step(held, r.gaze_out, None, render=False)
            prev = held
    finally:
        src.close()


def test_the_lesson_ceiling_holds_a_reviving_sense_to_its_equal_share_live_and_in_replay(tmp_path):
    torch.manual_seed(41)
    base = _live_lite(checkpoint_dir=str(tmp_path / "ck"), next_frame_norm="persistence")
    _dead(base)
    path = base.save_checkpoint(step=5)
    m = _live_lite(next_frame_norm="persistence", vestibular_encoder_repair="rescale_restart")
    m.load_checkpoint(path, load_optimizer=False)
    m.enable_replay(1, size_per_env=64)
    _drive_body(m, tmp_path, 24, "live")
    live = m.next_frame_norm_window_summary()
    assert m.vestibular_revival_live is not None                       # the weight is at its cap: it applies
    others = [s for s in live["share_normalized"] if s != "vestibular"]
    assert live["share_normalized"]["vestibular"] <= 1.0 / (len(others) + 1) + 1e-6, live
    replayed = 0
    for _ in range(6):
        if m.replay_update(batch=4) is not None:
            replayed += 1
    assert replayed and m.vestibular_revival_replay is not None and m.vestibular_revival_replay <= 1.0
    win = m.next_frame_norm_window_summary()
    rs = win.get("replay_share_normalized") or {}
    if "vestibular" in rs:
        assert rs["vestibular"] <= 1.0 / (len([s for s in rs if s != "vestibular"]) + 1) + 1e-6, win
    # the default build's lesson is untouched
    d = _live_lite(next_frame_norm="persistence")
    d.load_checkpoint(path, load_optimizer=False)
    assert d.vestibular_revival_live is None and d.vestibular_revival_replay is None


def _frames(n):
    v = np.zeros(VS.VESTIBULAR_DIM, dtype=np.float32)
    v[VS.FIELD_INDEX["valid"]] = 1.0
    return [SimpleNamespace(vestibular=v.copy()) for _ in range(n)]


def test_the_window_adds_its_gauges_only_with_the_flag(tmp_path):
    off = _lite()
    on = _lite(vestibular_encoder_repair="rescale")
    for model in (off, on):
        model._last_source_mask = {"vestibular": "real"}
        model.last_prediction_audit = {}
    w_off, w_on = VS.VestibularWindow(), VS.VestibularWindow()
    g = torch.Generator().manual_seed(1)
    toks = [torch.randn(3, on.d_model, generator=g) for _ in range(3)]
    for t in toks:
        off.last_tokens = {"vestibular": t}
        on.last_tokens = {"vestibular": t}
        w_off.add(_frames(3), off)
        w_on.add(_frames(3), on)
    s_off, s_on = w_off.summary(), w_on.summary()
    assert "repair" not in s_off
    rep = s_on["repair"]
    want_between = np.mean([float(t.std(0, unbiased=False).square().mean().sqrt()) for t in toks])
    want_change = np.mean([float((b - a).square().mean(-1).sqrt().mean()) for a, b in zip(toks, toks[1:])])
    assert rep["token_between_bodies"] == pytest.approx(want_between, rel=1e-5)
    assert rep["token_change"] == pytest.approx(want_change, rel=1e-5)
    assert rep["bias_spread"] == pytest.approx(float(_last(on).bias.double().std(unbiased=False)), rel=1e-5)
    assert rep["switch"] is None and set(rep["scale"]) >= {"token_between_bodies", "token_change", "bias_spread"}
    # the default window's record has exactly the keys it had before the flag existed
    assert set(s_off) == set(s_on) - {"repair"}


# ------------------------------------------------------------------------------ a real body
LIVE_OV = {"model": {"residual_prediction": True}, "training": {"steps_per_phase": 3000}}


def _live_lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, intention=False, imagination=False, no_vision=True,
                   next_frame_loss=True, top_down_gate=True, overrides=LIVE_OV, vestibular_sense=True, **kw)


def _reach(tmp_path, tag):
    from mimo_bridge.datasource import SingleEnvSource
    return SingleEnvSource(scene="reach", run_dir=str(tmp_path / tag), tag="_" + tag, vision=False, seed=5)


def _run(m, src, n, record):
    frames = src.reset()
    dev = dict(m.dev_limits(10 ** 9))
    dev["exploration_noise"] = 0.05
    held = prev = None
    for i in range(n):
        r = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
        record(m, frames)
        if i % 5 == 0 or held is None:
            held = r.motor_for_action
        for _ in range(5):
            frames = src.step(held, r.gaze_out, None, render=False)
        prev = held


@pytest.mark.slow
def test_a_production_sized_constant_stays_dead_and_the_repair_revives_it(tmp_path):
    # the deploy's own slow test (test_vestibular_sense.py, "starts to vary after 100 steps") ran with a
    # constant of size about 14; production's has a spread of 74,425. Same body, same seeds, 100 training steps.
    torch.manual_seed(41)
    base = _live_lite(checkpoint_dir=str(tmp_path / "ck"))
    last = _last(base)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        v = torch.randn(last.bias.shape, generator=g, dtype=torch.float64)
        last.bias.copy_(((v - v.mean()) / v.std(unbiased=False) * PROD_SPREAD + PROD_MEAN).float())
        last.weight.zero_()                 # the deploy's start: weight 0, bias = the constant
    path = base.save_checkpoint(step=5)
    late = {}
    for arm, flag in (("as_is", "off"), ("repaired", "rescale_restart")):
        torch.manual_seed(41)
        m = _live_lite(vestibular_encoder_repair=flag)
        m.load_checkpoint(path, load_optimizer=False)
        toks = []
        src = _reach(tmp_path, arm)
        try:
            _run(m, src, 100, lambda model, frames: toks.append(model.last_tokens["vestibular"].detach().clone()))
        finally:
            src.close()
        t = torch.stack(toks[-20:]).squeeze(1)
        late[arm] = float((t - t.mean(0)).norm(dim=-1).mean())
    # the deploy test's own line for "began to vary" is 1e-3. As is, a fresh optimizer's start-up correction
    # moves the weights at about a quarter of the learning rate on gradients near eps (about 3e-9 here), so
    # after 100 steps the weights' part is about 0.05 before the LayerNorm and 1e-5 after it (a token norm):
    # the dead line is 10 times below "began to vary"
    assert late["as_is"] < 1e-4, late
    assert late["repaired"] > 1e-3, late


# ------------------------------------------------------------------------------ the balance probe (review finding 4)
# H84's layout at report 410,500 (the vestibular block's per_body readings): body 2 lies face down, the other seven
# on their backs
H84_TILT = [14.1, 17.1, 114.8, 31.5, 27.8, 15.1, 26.8, 15.3]
H84_ROLL = [-3.8, -1.6, 172.7, -0.9, -1.4, -3.0, -0.7, -2.6]


def _vec_with(tilt_deg, roll_deg):
    """A raw vestibular vector whose head gravity makes `tilt_deg` from the resting direction (turned about the
    head's z) and whose trunk-up direction makes `roll_deg` of roll."""
    v = np.zeros(VS.VESTIBULAR_DIM, dtype=np.float32)
    a = np.radians(tilt_deg)
    rest = np.asarray(VS.HEAD_GRAVITY_REST)
    axis = np.cross(rest, [0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    rot = (rest * np.cos(a) + np.cross(axis, rest) * np.sin(a) + axis * np.dot(axis, rest) * (1 - np.cos(a)))
    v[0:3] = rot
    along, front, side = np.asarray(VS.CHEST_AXES)
    r = np.radians(roll_deg)
    v[15:18] = front * np.cos(r) + side * np.sin(r)
    v[VS.FIELD_INDEX["valid"]] = 1.0
    return v


def _probe_run(token_of, wobble=5.0, reports=3, steps=100, seed=0):
    """H84's bodies, each wobbling `wobble` degrees about its own posture; token_of(labels, g) -> [8, 192]."""
    g = np.random.default_rng(seed)
    probe = VS.BalanceProbe()
    for _ in range(reports):
        for _ in range(steps):
            t = np.asarray(H84_TILT) + g.normal(0, wobble, 8)
            r = np.asarray(H84_ROLL) + g.normal(0, wobble, 8)
            labels, ok = VS.balance_labels(np.stack([_vec_with(a, b) for a, b in zip(t, r)]))
            assert ok.all()
            probe.add(token_of(labels, g), labels, ok)
        probe.flush()
    return probe.read()


def _ln(x):
    return torch.nn.functional.layer_norm(x.float(), (192,))


def test_the_balance_labels_are_where_gravity_pulls_in_the_head_and_the_trunk_roll_direction():
    tilt, roll = [0.0, 30.0, 90.0, 120.0], [0.0, -45.0, 90.0, 179.0]
    raw = np.stack([_vec_with(a, b) for a, b in zip(tilt, roll)])
    labels, ok = VS.balance_labels(raw)
    assert ok.all()
    # the stats block's angles are the angles of these directions
    assert np.allclose(VS.head_tilt_deg(labels[:, 0:3]), tilt, atol=0.01)
    assert np.allclose(np.degrees(np.arctan2(labels[:, 3], labels[:, 4])), roll, atol=0.01)
    assert np.allclose(np.linalg.norm(labels[:, 0:3], axis=1), 1.0) and np.allclose(np.linalg.norm(labels[:, 3:5], axis=1), 1.0)
    # a trunk standing straight up has no roll direction; an accelerometer reading nothing has no gravity
    raw[1, 15:18] = VS.CHEST_AXES[0]
    raw[2, 0:3] = 0.0
    assert VS.balance_labels(raw)[1].tolist() == [True, False, False, True]


def test_the_balance_probe_reads_a_token_that_carries_balance_and_nothing_else():
    torch.manual_seed(0)
    M = torch.randn(5, 192, dtype=torch.float64)
    C = 3.0 * torch.randn(1, 192, dtype=torch.float64)       # the encoder's bias part (spread 18 on H84)
    ID = 3.0 * torch.randn(8, 192, dtype=torch.float64)      # one fixed code per body

    def carrying(labels, g):
        return _ln(C + torch.as_tensor(labels) @ M + 0.05 * torch.as_tensor(g.standard_normal((len(labels), 192))))
    # on H84's layout the first version read a token that carries the state exactly at 0.44 and -2.4 for the
    # trunk's roll (wobble 3 and 10 degrees; scratchpad vfix/probe_h84_layout.py): no fit on the bodies lying on
    # their backs places the one lying face down. Within each body the read follows the state.
    for wobble in (3.0, 10.0):
        r = _probe_run(carrying, wobble=wobble)
        assert r["samples"] == 2400 and r["reports"] == 3
        assert min(r["r2"].values()) > 0.9, r
        assert max(r["r2_shuffled"].values()) < 0.05, r
        assert r["within_body_turn_deg"]["head_tilt"] == pytest.approx(wobble, rel=0.15), r
        assert r["within_body_turn_deg"]["trunk_roll"] == pytest.approx(wobble, rel=0.15), r
    # a token that only tells the bodies apart: across bodies it fits every body's steady posture (8 bodies, 192
    # numbers); within each body it predicts nothing
    r = _probe_run(lambda labels, g: _ln(C + ID + 0.05 * torch.as_tensor(g.standard_normal((8, 192)))))
    assert max(r["r2"].values()) < 0.05, r
    # a varying token unrelated to balance (what the anti-collapse lesson alone can make): nothing
    r = _probe_run(lambda labels, g: _ln(torch.as_tensor(g.standard_normal((8, 192)))))
    assert max(r["r2"].values()) < 0.05, r
    # the dead token predicts no change: 0
    dead = _ln(torch.randn(1, 192))
    r = _probe_run(lambda labels, g: dead.expand(8, 192))
    assert r["r2"] == {"head_tilt": 0.0, "trunk_roll": 0.0} and r["r2_shuffled"] == r["r2"], r
    # one body: no split between bodies, no reading
    one = VS.BalanceProbe()
    one.add(torch.randn(1, 192), np.zeros((1, 5)), np.ones(1, dtype=bool))
    one.flush()
    assert one.read() is None


def test_the_window_feeds_the_probe_from_the_frames_it_reads():
    on = _lite(vestibular_encoder_repair="rescale")
    on._last_source_mask = {"vestibular": "real"}
    on.last_prediction_audit = {}
    torch.manual_seed(2)
    M = torch.randn(5, 192)
    C = 3.0 * torch.randn(1, 192)
    w = VS.VestibularWindow()
    g = np.random.default_rng(1)
    for _ in range(2):
        for _ in range(100):
            tilt = np.asarray(H84_TILT) + g.normal(0, 5.0, 8)
            roll = np.asarray(H84_ROLL) + g.normal(0, 5.0, 8)
            frames = [SimpleNamespace(vestibular=_vec_with(a, b)) for a, b in zip(tilt, roll)]
            state, ok = VS.balance_labels(np.stack([f.vestibular for f in frames]))
            assert ok.all()
            on.last_tokens = {"vestibular": _ln(C + torch.as_tensor(state, dtype=torch.float32) @ M)}
            w.add(frames, on)
        s = w.summary()
    bp = s["repair"]["balance_probe"]
    assert bp["reports"] == 2 and bp["samples"] == 1600
    assert min(bp["r2"].values()) > 0.9 and max(bp["r2_shuffled"].values()) < 0.1, bp
    assert "balance_probe" in s["repair"]["scale"]
