# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--motor-input-fold (lem/model/motor_input_fold.py; the record: docs/gain_of_function_v4/agents/
memory_motor_layer_fall_2026-09-26.md, sections 8 and 9).

Each test builds a small LemLite on the CPU (no world, no graphics card) with the repaired memory, the adaptive
controller and the liveness repair's parts (revival off), switches the memory from a saturated state and reads the motor
controller on latents that carry the memory's bias. What they hold:
  - off builds today's model: no attribute, no side module, no hook; on adds no parameter and draws nothing, so the
    parameters (names, order, values) and the random stream are the flag-off build's;
  - the fold needs the repaired memory; it waits for the memory's switch, happens once, and only in a training step;
  - exact at the fold: the controller's orders and confidence, its first residual layer's input and pre-activation, and
    its confidence head's pre-activation (own units and revived units) move by at most 1e-5 of their largest value;
  - after the fold, a change of the memory's constant (its bias, or a return's factor) no longer moves the controller's
    input; the same change moves an unfolded controller's input by the constant's image;
  - the lever is cut: the first layer's token columns read the tokens' own mean, and a request every row shares moves
    those weights by the tokens' mean, not the constant;
  - planned before written: above the exactness line the fold refuses and changes nothing;
  - the folded state survives a save and a load bit for bit and never folds twice; a build without the fold refuses a
    folded checkpoint; a receipt that says folded without the side module is refused;
  - the episode companion captures and restores the fold's state;
  - the trainer's flag reaches the built learner, and LemLite.step folds right after the memory's switch, before the
    controller or the memory is read.
"""
import copy
import inspect

import pytest
import torch

B = 3


def _model(path, fold="on", repair="centred", ret=0, **kw):
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", seed=4, imagination=False, next_frame_loss=True, persistent_memory="on",
                   persistent_memory_repair=repair, persistent_memory_constant_return=ret, think_harder=True,
                   liveness_repair="on", liveness_rule="none", motor_input_fold=fold, checkpoint_dir=str(path), **kw)


def _switched(model, const_sd=30.0, seed=2):
    """The memory saturated (its candidate held at +-1 by a bias of 30, every body's state the same pattern, as H84's
    since update 379,324), a lived history, then the memory's own switch: the readout's ordinary bias then holds a
    constant of about const_sd per number (H84: 153 against the tokens' own 5)."""
    pm = model.persistent_memory
    H = pm.hidden_size
    g = torch.Generator().manual_seed(seed)
    p = torch.sign(torch.randn(H, generator=g))
    p[p == 0] = 1.0
    with torch.no_grad():
        pm.recurrent.bias_ih_l0[2 * H:] = 30.0 * p
        pm.readout.weight.normal_(0.0, 0.5, generator=g)
        pm.readout.bias.normal_(0.0, const_sd, generator=g)
    for k in range(12):
        model.memory_history.observe(torch.randn(B, pm.input_size, generator=g), torch.full((B,), 0.05 * k), pm)
    model.memory_history.carry = p.reshape(1, 1, -1).expand(1, B, -1).contiguous().clone()
    model.memory_history.carry_time = model.memory_history.times[0].clone() - 0.5
    model._persistent_memory_switch(model.memory_history.snapshot())
    assert pm.repaired
    return pm


def _revive_some(model, seed=3):
    """Revived units in the confidence head's first layer and the first residual layer, with learned centres, as the
    liveness repair leaves them (H84: 84 of 128 and 127 of 128); the confidence head's centre sits near the constant,
    where a centre that follows the tokens sits."""
    g = torch.Generator().manual_seed(seed)
    mc = model.motor_controller
    pm = model.persistent_memory
    lv = model.liveness
    conf_rows = lv.rows["motor_controller__confidence_head__0"]
    blk_rows = lv.rows["motor_controller__residual_blocks__0__0"]
    with torch.no_grad():
        for rows, lin, k in ((conf_rows, mc.confidence_head[0], 40), (blk_rows, mc.residual_blocks[0][0], 60)):
            rows.mask[:k] = True
            rows.any_py = True
            rows.done.fill_(True)
            rows.done_py = True
            rows.lin.weight.normal_(0.0, 0.2, generator=g)
            rows.lin.bias.normal_(0.0, 0.2, generator=g)
            lin.weight[:k] = 0.0
            lin.bias[:k] = 0.0
        conf_rows.centre_mean.copy_(pm.readout.bias.detach() + 0.3 * torch.randn(conf_rows.in_f, generator=g))
        conf_rows.centre_weight = 0.4
        blk_rows.centre_mean.copy_(torch.randn(blk_rows.in_f, generator=g))
        blk_rows.centre_weight = 0.4
    return conf_rows, blk_rows


def _inputs(model, seed=5, base_sd=1.0):
    """[B, 3 + n_mod, d] latents whose newest n_mod tokens are unit-scale token values plus the memory's own bias (its
    constant and its varying part), the handle state and the wellness slope."""
    g = torch.Generator().manual_seed(seed)
    pm = model.persistent_memory
    with torch.no_grad():
        state, _ = pm.encode(**model.memory_history.snapshot())
        mb = pm.bias(state)
    base = base_sd * torch.randn(B, model.n_mod, model.d_model, generator=g)
    lat = torch.cat([torch.randn(B, 3, model.d_model, generator=g), base + mb], 1)
    H = len(model.handle_names)
    return lat, base, torch.randn(B, H, 10, generator=g), 0.1 * torch.randn(B, H, 9, generator=g)


def _read(model, lat, hs, wg):
    """The controller's orders and confidence, its first residual layer's input and pre-activation (after the repair's
    routing) and its confidence head's pre-activation, on these inputs, without learning (no centre moves)."""
    mc = model.motor_controller
    got = {}
    h1 = mc.residual_blocks[0][0].register_forward_hook(
        lambda m, i, o: got.update(x=i[0].detach().clone(), pre=o.detach().clone()))
    h2 = mc.confidence_head[0].register_forward_hook(lambda m, i, o: got.update(conf_pre=o.detach().clone()))
    try:
        with torch.no_grad():
            out, conf, _meta = mc(lat, hs, wellness_grad=wg)
    finally:
        h1.remove()
        h2.remove()
    got.update(orders=out.detach().clone(), confidence=conf.detach().clone())
    return got


def _rel(a, b):
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-12)


def _fold(model, lat, hs, wg):
    model._lag_lat, model._lag_hs, model._lag_wg = lat, hs, wg
    return model.motor_input_fold.at_step_start(model, True)


# ------------------------------------------------------------------------------------------------ the default
def test_off_builds_todays_model_and_on_adds_no_parameter_and_draws_nothing(tmp_path):
    from mimo_bridge.lem_model import LemLite
    torch.manual_seed(0)
    plain = LemLite(device="cpu", seed=4, imagination=False, next_frame_loss=True, persistent_memory="on",
                    persistent_memory_repair="centred", think_harder=True, liveness_repair="on", liveness_rule="none",
                    checkpoint_dir=str(tmp_path / "plain"))
    rng_plain = torch.get_rng_state()
    off = _model(tmp_path / "off", fold="off")
    rng_off = torch.get_rng_state()
    on = _model(tmp_path / "on", fold="on")
    rng_on = torch.get_rng_state()
    assert not hasattr(off, "motor_input_fold") and "motor_input_fold" not in off._side_modules()
    assert "motor_input_fold" not in off.receipt
    first = off.motor_controller.input_proj
    assert not first._forward_pre_hooks and not off.motor_controller.confidence_head._forward_pre_hooks
    # the same random stream after each build: the fold draws nothing
    assert torch.equal(rng_plain, rng_off) and torch.equal(rng_off, rng_on)
    for a, b in ((plain, off), (off, on)):
        pa = [(n, p) for n, p in a.motor_controller.named_parameters()]
        pb = [(n, p) for n, p in b.motor_controller.named_parameters()]
        assert [n for n, _ in pa] == [n for n, _ in pb]
        assert all(torch.equal(x, y) for (_, x), (_, y) in zip(pa, pb))
        assert len(a.params) == len(b.params) and all(x.shape == y.shape for x, y in zip(a.params, b.params))
        assert [len(g["params"]) for g in a.optimizer.param_groups] == [len(g["params"]) for g in b.optimizer.param_groups]
    assert list(on.motor_input_fold.parameters()) == []
    assert on._side_modules()["motor_input_fold"] is on.motor_input_fold
    assert on.receipt["motor_input_fold"]["folded"] is False


def test_the_fold_needs_the_repaired_memory(tmp_path):
    with pytest.raises(ValueError, match="--persistent-memory-repair centred"):
        _model(tmp_path / "a", repair="off")
    from mimo_bridge.lem_model import LemLite
    with pytest.raises(ValueError, match="--persistent-memory-repair centred"):
        LemLite(device="cpu", seed=4, imagination=False, think_harder=True, motor_input_fold="on",
                checkpoint_dir=str(tmp_path / "b"))
    with pytest.raises(ValueError, match="must be one of"):
        _model(tmp_path / "c", fold="yes")


def test_the_fold_waits_for_the_switch_happens_once_and_only_in_training(tmp_path):
    model = _model(tmp_path)
    fold = model.motor_input_fold
    b0 = model.motor_controller.input_proj.bias.detach().clone()
    assert fold.at_step_start(model, True) is None and not fold.folded_py        # the memory is not switched yet
    _switched(model)
    assert fold.at_step_start(model, False) is None and not fold.folded_py       # an evaluation step never folds
    assert torch.equal(model.motor_controller.input_proj.bias, b0)
    lat, _base, hs, wg = _inputs(model)
    rec = _fold(model, lat, hs, wg)
    assert rec is not None and fold.folded_py and bool(fold.folded)
    assert int(fold.fold_step) == model.step_count == rec["folded_at_step"]
    assert torch.equal(fold.constant_at_fold, model.persistent_memory.readout.bias.detach())
    b1 = model.motor_controller.input_proj.bias.detach().clone()
    assert not torch.equal(b1, b0)
    assert fold.at_step_start(model, True) is None                                # once
    assert torch.equal(model.motor_controller.input_proj.bias, b1)
    with pytest.raises(ValueError, match="folded already"):
        fold.fold(model)


# ------------------------------------------------------------------------------------------------ exactness
def test_the_fold_is_exact(tmp_path):
    model = _model(tmp_path)
    _switched(model)
    _revive_some(model)
    lat, _base, hs, wg = _inputs(model)
    before = _read(model, lat, hs, wg)
    rec = _fold(model, lat, hs, wg)
    after = _read(model, lat, hs, wg)
    for k in ("x", "pre", "conf_pre", "orders", "confidence"):
        assert _rel(after[k], before[k]) <= 1e-5, (k, _rel(after[k], before[k]))
    assert rec["first_layer_change_share"] <= 1e-5 and rec["confidence_change_share"] <= 1e-5
    assert rec["probe"].startswith("the controller's last lived inputs")
    assert rec["confidence_revived_units"] == 40 and rec["confidence_revived_centre_moved"]
    # the constant is large against the tokens (as on H84) and its image went into the bias
    assert min(rec["constant_rms_per_slot"]) > 10.0 and rec["constant_image_length_first_layer"] > 10.0


def test_the_fold_is_exact_for_a_revived_centre_that_has_not_learned_yet(tmp_path):
    """A revived unit's centre with no weight yet is read by the call that first sets it: it moves with the input."""
    model = _model(tmp_path)
    _switched(model)
    conf_rows, _blk = _revive_some(model)
    with torch.no_grad():
        conf_rows.centre_mean.zero_()
    conf_rows.centre_weight = 0.0
    lat, _base, hs, wg = _inputs(model)
    before = _read(model, lat, hs, wg)
    _fold(model, lat, hs, wg)
    after = _read(model, lat, hs, wg)
    assert _rel(after["conf_pre"], before["conf_pre"]) <= 1e-5


def test_the_fold_is_planned_before_anything_is_written(tmp_path, monkeypatch):
    from lem.model import motor_input_fold as MIF
    model = _model(tmp_path)
    _switched(model)
    conf_rows, _blk = _revive_some(model)
    lat, _base, hs, wg = _inputs(model)
    mc = model.motor_controller
    before = {k: v.clone() for k, v in mc.state_dict().items()}
    centre = conf_rows.centre_mean.clone()
    monkeypatch.setattr(MIF, "EXACT_LINE", -1.0)          # every change is above it
    with pytest.raises(ValueError, match="nothing was changed"):
        _fold(model, lat, hs, wg)
    assert not model.motor_input_fold.folded_py and not bool(model.motor_input_fold.folded)
    for k, v in mc.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert torch.equal(conf_rows.centre_mean, centre)


# ------------------------------------------------------------------------------------------------ the lever
def _pair(tmp_path, ret=0):
    """Two builds with the same weights and the same switched memory; the first folded, the second not."""
    folded = _model(tmp_path / "on", ret=ret)
    plain = _model(tmp_path / "off", fold="off", ret=ret)
    for m in (folded, plain):
        _switched(m)
        _revive_some(m)
    lat, _base, hs, wg = _inputs(folded)
    _fold(folded, lat, hs, wg)
    return folded, plain


def test_after_the_fold_a_change_of_the_constant_no_longer_moves_the_controllers_input(tmp_path):
    """The record's section 8: a later change of the constant (its bias's own Adam steps, or a return) moved an
    unfolded controller's input by the constant's image; folded, the constant enters the tokens and leaves them."""
    folded, plain = _pair(tmp_path)
    reads = {}
    for name, m in (("folded", folded), ("plain", plain)):
        lat1, _b, hs, wg = _inputs(m, seed=11)
        r1 = _read(m, lat1, hs, wg)
        with torch.no_grad():
            m.persistent_memory.readout.bias.mul_(0.5)        # the constant halves, as a return would move it
        lat2, _b2, _hs, _wg = _inputs(m, seed=11)
        r2 = _read(m, lat2, hs, wg)
        reads[name] = (r1, r2)
    r1, r2 = reads["folded"]
    for k in ("x", "pre", "conf_pre", "orders"):
        assert _rel(r2[k], r1[k]) <= 1e-5, (k, _rel(r2[k], r1[k]))
    p1, p2 = reads["plain"]
    assert _rel(p2["x"], p1["x"]) > 0.05                     # the unfolded input moved with the constant


def test_after_the_fold_a_running_return_no_longer_moves_the_controllers_input(tmp_path):
    folded, plain = _pair(tmp_path, ret=100)
    for m in (folded, plain):
        assert m.persistent_memory._return_on
    reads = {}
    for name, m in (("folded", folded), ("plain", plain)):
        lat1, _b, hs, wg = _inputs(m, seed=13)
        r1 = _read(m, lat1, hs, wg)
        with torch.no_grad():
            m.persistent_memory.updates_since_switch.fill_(50.0)   # half way: the constant shrinks by the factor
        lat2, _b2, _hs, _wg = _inputs(m, seed=13)
        assert float((lat2 - lat1).abs().max()) > 1.0
        reads[name] = (r1, _read(m, lat2, hs, wg))
    r1, r2 = reads["folded"]
    for k in ("x", "orders"):
        assert _rel(r2[k], r1[k]) <= 1e-5, (k, _rel(r2[k], r1[k]))
    p1, p2 = reads["plain"]
    assert _rel(p2["x"], p1["x"]) > 0.05


def test_the_lever_is_cut(tmp_path):
    """The first layer's token columns read the tokens' own mean after the fold, not the constant: a request every
    row shares moves a unit's weights by (the rows' summed input) and so its mean input by lr x sum |mean input|."""
    folded, plain = _pair(tmp_path)
    n = folded.motor_input_fold.n_tokens
    sums = {}
    for name, m in (("folded", folded), ("plain", plain)):
        lat, base, hs, wg = _inputs(m, seed=17)
        seen = {}
        lin = m.motor_controller.input_proj
        h = lin.register_forward_hook(lambda mod, i, o: seen.update(x=i[0], o=o))
        try:
            m.motor_controller.zero_grad(set_to_none=True)
            m.motor_controller(lat, hs, wellness_grad=wg)
        finally:
            h.remove()
        seen["o"].sum().backward()          # a request every row shares, on the first layer's output
        g = lin.weight.grad[:, :n]
        sums[name] = dict(mean_input=float(seen["x"][:, :n].detach().mean(0).abs().sum()),
                          grad=float(g.abs().sum(1).mean()),
                          tokens_mean=float(base.reshape(B, -1).mean(0).abs().sum()))
    f, p = sums["folded"], sums["plain"]
    # folded: the columns read the tokens' own values (to rounding), not the constant
    assert f["mean_input"] == pytest.approx(f["tokens_mean"], rel=1e-3)
    assert p["mean_input"] > 10.0 * f["mean_input"]
    assert p["grad"] > 10.0 * f["grad"]


# ------------------------------------------------------------------------------------------------ save and load
def _learn_once(model, lat, hs, wg):
    """One real optimizer step through the controller (its centres move: a training call)."""
    model.optimizer.zero_grad(set_to_none=True)
    out, _c, _m = model.motor_controller(lat, hs, wellness_grad=wg)
    out.square().mean().backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)


def _folded_checkpoint(tmp_path):
    model = _model(tmp_path / "a")
    _switched(model)
    _revive_some(model)
    lat, _base, hs, wg = _inputs(model)
    _fold(model, lat, hs, wg)
    _learn_once(model, lat, hs, wg)
    model.step_count = 12345
    path = model.save_checkpoint(12345)
    return model, path, (lat, hs, wg)


def test_the_folded_state_survives_a_save_and_load_bit_for_bit_and_never_folds_twice(tmp_path):
    model, path, (lat, hs, wg) = _folded_checkpoint(tmp_path)
    fresh = _model(tmp_path / "b")
    fresh.load_checkpoint(path)
    fold = fresh.motor_input_fold
    assert fold.folded_py and bool(fold.folded) and int(fold.fold_step) == int(model.motor_input_fold.fold_step)
    for name in ("motor_controller", "persistent_memory", "motor_input_fold"):
        a, b = getattr(model, name).state_dict(), getattr(fresh, name).state_dict()
        assert set(a) == set(b), name
        for k in a:
            assert torch.equal(a[k], b[k]), (name, k)
    for key in ("motor_controller__confidence_head__0", "motor_controller__residual_blocks__0__0"):
        assert torch.equal(model.liveness.rows[key].centre_mean, fresh.liveness.rows[key].centre_mean), key
    r0, r1 = _read(model, lat, hs, wg), _read(fresh, lat, hs, wg)
    for k in r0:
        assert torch.equal(r0[k], r1[k]), k
    bias = fresh.motor_controller.input_proj.bias.detach().clone()
    assert fold.at_step_start(fresh, True) is None
    assert torch.equal(fresh.motor_controller.input_proj.bias, bias)
    got = fresh.receipt["motor_input_fold"]
    assert got["folded"] and got["folded_at_step"] == int(fold.fold_step)
    assert got["fold"]["first_layer_change_share"] == model.motor_input_fold.record["first_layer_change_share"]


def test_a_build_without_the_fold_refuses_a_folded_checkpoint(tmp_path):
    _model_, path, _x = _folded_checkpoint(tmp_path)
    off = _model(tmp_path / "off", fold="off")
    with pytest.raises(ValueError, match="--motor-input-fold on"):
        off.load_checkpoint(path)


def test_a_receipt_that_says_folded_without_the_side_module_is_refused(tmp_path):
    import gc
    import os
    import shutil
    from safetensors.torch import load_file, save_file
    _model_, path, _x = _folded_checkpoint(tmp_path)
    bad = tmp_path / "bad" / os.path.basename(path)
    shutil.copytree(path, bad)
    loaded = load_file(str(bad / "model.safetensors"))
    w = {k: v.clone() for k, v in loaded.items() if not k.startswith("motor_input_fold.")}
    del loaded
    gc.collect()                        # Windows keeps a mapped file locked while its tensors live
    save_file(w, str(bad / "model.safetensors.tmp"))
    os.replace(str(bad / "model.safetensors.tmp"), str(bad / "model.safetensors"))
    fresh = _model(tmp_path / "b")
    with pytest.raises(ValueError, match="hold no folded"):
        fresh.load_checkpoint(str(bad))


def test_an_unfolded_checkpoint_loads_unfolded_into_a_folded_model(tmp_path):
    """A process that loads another checkpoint into a model that folded (a probe reading several saves) takes the
    checkpoint's own state: unfolded."""
    plain = _model(tmp_path / "p")
    _switched(plain)
    plain.step_count = 7
    path = plain.save_checkpoint(7)
    model = _model(tmp_path / "a")
    _switched(model)
    lat, _b, hs, wg = _inputs(model)
    _fold(model, lat, hs, wg)
    model.load_checkpoint(path)
    assert not model.motor_input_fold.folded_py and not bool(model.motor_input_fold.folded)
    assert model.receipt["motor_input_fold"]["folded"] is False


# ------------------------------------------------------------------------------------------------ the trainer
def test_the_episode_companion_carries_the_folds_state(tmp_path):
    from mimo_bridge import copied_execution as CE
    model = _model(tmp_path)
    _switched(model)
    lat, _b, hs, wg = _inputs(model)
    _fold(model, lat, hs, wg)
    plain = CE.plain_attributes(model.motor_input_fold)
    assert plain["folded_py"] is True and plain["record"]["folded_at_step"] == model.step_count
    assert "_model" not in plain and "_detach" not in plain
    saved = {k: v.clone() for k, v in model.motor_input_fold._buffers.items()}
    kept = copy.deepcopy(plain)
    model.motor_input_fold.folded_py = False
    for k, v in kept.items():
        CE._restore_plain(model.motor_input_fold, k, v, model.device)
    assert model.motor_input_fold.folded_py is True
    for k, v in saved.items():
        assert torch.equal(model.motor_input_fold._buffers[k], v)


def test_the_fold_record_rides_the_memory_switch_record(tmp_path):
    model = _model(tmp_path)
    _switched(model)
    sw = model.persistent_memory_switch_unwritten
    assert isinstance(sw, dict)
    lat, _b, hs, wg = _inputs(model)
    rec = _fold(model, lat, hs, wg)
    assert sw["motor_input_fold"] is rec
    assert model.receipt["persistent_memory_repair"]["switch"]["motor_input_fold"] is rec


def test_the_step_folds_right_after_the_memory_switch_before_anything_reads_the_memory_or_the_controller():
    from mimo_bridge.lem_model import LemLite
    src = inspect.getsource(LemLite.step)
    switch = src.index("self._persistent_memory_switch(self.memory_history.snapshot())")
    fold = src.index("self.motor_input_fold.at_step_start(self, train)")
    assert switch < fold < src.index("self.persistent_memory.encode(**self._nf_memory)")
    assert fold < src.index("self.motor_controller(")
    load = inspect.getsource(LemLite.load_checkpoint)
    assert load.index("_mif_before_load(self, path)") < load.index("res = mgr.load(")


def test_the_trainers_flag_reaches_the_built_learner(tmp_path):
    from mimo_bridge import train_mimo as TM
    base = ["--out", str(tmp_path), "--no-vision", "--device", "cpu", "--n-envs", "1", "--persistent-memory", "on",
            "--persistent-memory-repair", "centred", "--persistent-memory-constant-return", "0", "--think-harder", "on"]
    args = TM.parse_args(base)
    assert args.motor_input_fold == "off"
    m = TM.build_learner(args, None)
    assert getattr(m, "motor_input_fold", None) is None
    m = TM.build_learner(TM.parse_args(base + ["--motor-input-fold", "on"]), None)
    assert m.motor_input_fold is not None and m.receipt["motor_input_fold"]["mode"] == "on"
