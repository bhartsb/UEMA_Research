# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The explainer's capture contract, pinned.

The gap hunt found the three ways this could silently lie (plan v2, section 10):
a re-forward against the post-shift context double-counts the current frame; raw
tokens are not what a gated forward consumed; post-update weights drift. These tests
hold the recipe that avoids the first two and labels the third.
"""
import json

import numpy as np
import pytest
import torch

from mimo_bridge.lem_model import LemLite
from mimo_bridge.explainer import capture as CAP
from mimo_bridge.explainer import schema as SCH
from mimo_bridge.explainer.writer import ExplainWriter


def _model(**kw):
    kw.setdefault("next_frame_loss", True)   # populates _nf_pred / the delayed lesson
    return LemLite(device="cpu", seed=0, imagination=False, **kw)


def _drive(m, src, n, train=True):
    """The trainer's cadence: 5-step holds, executed action carried one step late."""
    frames = src.reset()
    dev = dict(m.dev_limits(10 ** 9))
    dev["exploration_noise"] = 0.05
    held = prev = None
    for i in range(n):
        res = m.step(frames, dev, train=train and (i < n - 1), step_index=i,
                     executed_action=prev)
        if i % 5 == 0 or held is None:
            held = res.motor_for_action
        for _ in range(5):
            frames = src.step(held, res.gaze_out, None, render=False)
        prev = np.array(held)
    return frames, res, dev


# ===================================== the three attention stages, pinned by behaviour
# Nothing checked the model's BEHAVIOUR here - only the numbers the explainer prints,
# which would still pass if a mask were applied to the reported weights and not to the
# value path. These run at this run's real context length, not the lite default of 4.
CTX_REAL, NMOD_REAL, D_REAL = 128, 6, 128


def _core_block(cross_modal, causal_temporal, bridge):
    from mimo_bridge.lem_model import build_config
    from lem.model.forward_model import ForwardModel
    cfg = build_config({"model": {"context_length": CTX_REAL}})
    torch.manual_seed(0)
    blk = ForwardModel(cfg).double().eval().blocks[0]
    blk.enable_cross_modal = cross_modal
    blk.enable_causal_temporal = causal_temporal
    blk.enable_cross_modal_bridge = bridge
    return blk


def _stage_delta(moment, sense, cross_modal, causal_temporal, bridge):
    """Largest output change per (moment, sense) when ONE token is perturbed, with
    only the named stages live. The perturbation is a RANDOM vector, never a constant:
    the LayerNorm in front of every stage erases a constant shift, and the test then
    passes without testing anything (it did, on the first attempt)."""
    blk = _core_block(cross_modal, causal_temporal, bridge)
    torch.manual_seed(1)
    x = torch.randn(1, CTX_REAL * NMOD_REAL, D_REAL, dtype=torch.float64)
    bump = torch.zeros_like(x)
    bump[0, moment * NMOD_REAL + sense] = torch.randn(D_REAL, dtype=torch.float64)
    with torch.no_grad():
        a, b = blk(x), blk(x + bump)
    return (a - b).abs().reshape(CTX_REAL, NMOD_REAL, D_REAL).amax(-1)


def test_stage1_never_crosses_a_moment():
    d = _stage_delta(64, 0, True, False, False)
    assert float(d[64].min()) > 0, "stage 1 did not reach the other senses at its model step"
    assert float(d[:64].max()) == 0.0 and float(d[65:].max()) == 0.0


def test_stage2_output_cannot_see_a_later_input():
    d = _stage_delta(64, 0, False, True, False)
    assert float(d[:64, 0].max()) == 0.0, "an earlier model step's output moved"
    assert float(d[65:, 0].max()) > 0, "a later model step's output did not move"
    assert float(d[:, 1:].max()) == 0.0, "stage 2 leaked across senses"


def test_stage3_is_unmasked_by_design():
    """The bridge takes no mask: the newest moment reaches the oldest and back. The
    page says so in words; this pins it so a future mask cannot make the page lie."""
    forward = _stage_delta(CTX_REAL - 1, 0, False, False, True)
    assert float(forward[0].max()) > 0, "stage 3 looks masked - the page says it is not"
    backward = _stage_delta(0, 0, False, False, True)
    assert float(backward[CTX_REAL - 1].max()) > 0


# ===================================================== forward_model capture modes
def test_capture_all_shapes_and_row_sums():
    m = _model()
    fm = m.forward_model
    B, n_mod, ctx, d = 2, m.n_mod, m.ctx_len, m.d_model
    torch.manual_seed(0)
    toks = {mod: torch.randn(B, 1, d) for mod in m.active}
    ctx_buf = torch.randn(B, (ctx - 1) * n_mod, d)
    with torch.no_grad():
        _l, _s, _c, layers = fm(toks, ctx_buf, capture_attention="all")
    assert len(layers) == len(fm.blocks)
    S = ctx * n_mod
    for lw in layers:
        assert lw["cross_modal"].shape == (B * ctx, n_mod, n_mod)
        assert lw["temporal"].shape == (B * n_mod, ctx, ctx)
        assert lw["bridge"].shape == (B, S, S)
        for key in ("cross_modal", "temporal", "bridge"):
            s = lw[key].sum(dim=-1)
            assert torch.allclose(s, torch.ones_like(s), atol=1e-4), key
        # the causal mask: no attention to the future
        upper = torch.triu(lw["temporal"], diagonal=1)
        assert float(upper.abs().max()) < 1e-6


def test_legacy_capture_contract_unchanged():
    """lem/training_step.py consumes capture_attention=True as a list of stage-2
    tensors [B*n_mod, ctx, ctx]; the 'all' mode must not have disturbed it."""
    m = _model()
    fm = m.forward_model
    B, n_mod, ctx, d = 2, m.n_mod, m.ctx_len, m.d_model
    torch.manual_seed(0)
    toks = {mod: torch.randn(B, 1, d) for mod in m.active}
    ctx_buf = torch.randn(B, (ctx - 1) * n_mod, d)
    with torch.no_grad():
        _l, _s, _c, layers = fm(toks, ctx_buf, capture_attention=True)
    assert isinstance(layers, list) and len(layers) == len(fm.blocks)
    for w in layers:
        assert torch.is_tensor(w)
        assert w.shape == (B * n_mod, ctx, ctx)


def test_capture_matches_normal_output():
    """need_weights=True switches attention off the fused kernel; the OUTPUT must
    not move beyond float noise, or capture would perturb what it reports on."""
    m = _model()
    fm = m.forward_model
    B, n_mod, ctx, d = 2, m.n_mod, m.ctx_len, m.d_model
    torch.manual_seed(0)
    toks = {mod: torch.randn(B, 1, d) for mod in m.active}
    ctx_buf = torch.randn(B, (ctx - 1) * n_mod, d)
    with torch.no_grad():
        lat0, last0, cu0 = fm(toks, ctx_buf)
        lat1, last1, cu1, _ = fm(toks, ctx_buf, capture_attention="all")
    torch.testing.assert_close(lat0, lat1, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(last0, last1, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(cu0, cu1, atol=1e-4, rtol=1e-4)


# ===================================================== the capture recipe, end to end
@pytest.mark.parametrize("memory", ["off", "on"])
def test_reforward_matches_live_step(tmp_path, memory):
    """An eval step (weights frozen) then the schema-v1 capture: the re-forward's
    last-step output must reproduce the live pass's cached _lag_last. The shifted-
    context defect (current frame counted twice) fails this immediately, and so does
    dropping the persistent memory's bias, which the live pass adds to the newest six
    tokens after the core's last LayerNorm - measured 0.0635 apart on this fixture."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _model(persistent_memory=memory)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / ("rf" + memory)), tag="_rf",
                          vision=False, seed=3)
    try:
        _drive(m, src, 8, train=True)          # last step ran train=False (see _drive)
        _layers, cap_last = CAP.capture_attention_maps(m, body=0)
        live_last = m._lag_last[0]
        torch.testing.assert_close(cap_last, live_last, atol=1e-4, rtol=1e-4)
    finally:
        src.close()


def test_snapshot_schema_and_quantization(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    m = _model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "sn"), tag="_sn",
                          vision=False, seed=3)
    try:
        frames, res, dev = _drive(m, src, 8, train=True)
        snap = CAP.build_snapshot(
            m, frames, res, run="testrun", step=7, phase="neonate", dev=dev,
            weights="frozen", body=0, eye_layout=None,
            prev_executed=np.array(res.motor_for_action),
            held_order=np.array(res.motor_for_action), order_seq=2, order_age=1,
            flags={"eval_only": True}, check_invariants=True)
        ok, errs = SCH.validate(snap)
        assert ok, errs
        s = json.dumps(snap)                       # JSON round-trip
        back = json.loads(s)
        assert back["header"]["step"] == 7
        # quantized attention decodes within one quantization step of the original
        raw_layers, _ = CAP.capture_attention_maps(m, body=0)
        q = back["attention"]["layers"][0]["stage2"]
        dec = SCH.dequantize(q)
        orig = raw_layers[0]["stage2"]
        step_sz = (q["max"] - q["min"]) / 255.0 or 1e-9
        assert np.abs(dec - orig).max() <= step_sz * 0.5 + 1e-7
        # the delayed lesson populated the of_now block (next_frame_loss on)
        assert isinstance(back["predictions"]["of_now"], dict)
        assert back["predictions"]["error_of_now"]
        # per-body wellness came from the stash, not the batch means
        assert back["drives"]["wellness_body"]
        # the handle list is the MODEL's own (trunk bodies order 26 handles; the
        # 24-name base constant once made the viewer drop the last two rows)
        assert back["motor"]["handle_names"] == list(m.handle_names)
        if isinstance(back["motor"]["held_order"], list):
            assert len(back["motor"]["held_order"]) == len(back["motor"]["handle_names"])
        # the voice block: master_config builds the head, so it ran this step. Its
        # width, its 0-to-1 bound and its stated world effect are what the page
        # prints, and nothing asserted them before - every voice claim on the page
        # rested on one-off probes.
        v = back["voice"]
        assert v["n_controls"] == 80
        assert 0.0 < v["min"] <= v["max"] < 1.0
        assert "no vocal tract" in v["world_effect"]
    finally:
        src.close()


def test_eye_reassembly_roundtrip():
    """Synthetic images -> BinocularPatcher -> tile_eye_images == the originals,
    pixel for pixel, both sources, both eyes."""
    from mimo_bridge.vision_adapter import BinocularPatcher
    rng = np.random.default_rng(0)
    layout = [("wide", 64), ("fovea", 32)]
    imgs = {name: (rng.integers(0, 256, size=(s, s, 3), dtype=np.uint8),
                   rng.integers(0, 256, size=(s, s, 3), dtype=np.uint8))
            for name, s in layout}
    patcher = BinocularPatcher([(n, s, 60.0) for n, s in layout])
    patches, _angles = patcher(imgs)
    tiled = CAP.tile_eye_images(patches, layout)
    for name, s in layout:
        np.testing.assert_array_equal(tiled[name]["left"], imgs[name][0])
        np.testing.assert_array_equal(tiled[name]["right"], imgs[name][1])
    # a wrong layout is a labeled refusal, never a wrong picture
    with pytest.raises(ValueError):
        CAP.tile_eye_images(patches, [("wide", 64)])


def test_writer_atomic_index_retention(tmp_path):
    w = ExplainWriter(str(tmp_path), keep_mb=0.02, keep_newest=3)
    for step in range(10):
        snap = {"header": {"step": step}, "pad": "x" * 4000}
        assert w.submit(snap) or True         # drops allowed; flush serializes below
        w.flush()
    w.flush()
    idx = json.loads(open(tmp_path / "explain" / "index.json").read())
    steps = [e["step"] for e in idx["snapshots"]]
    # newest 3 always survive; older history thinned, not truncated to nothing
    assert steps[-3:] == sorted(steps)[-3:]
    assert len(steps) < 10
    total_mb = sum(e["bytes"] for e in idx["snapshots"]) / 1e6
    assert total_mb <= 0.04                    # within a write of the cap
    for e in idx["snapshots"]:                 # every listed file is whole JSON
        json.loads(open(tmp_path / "explain" / e["file"]).read())


def test_stage3_reports_both_directions_and_a_share_that_adds_to_one():
    """P041: stage 3 is UNMASKED, so it attends forward as freely as backward.

    The capture only ever computed the backward curve and the page plotted it
    under "attention vs how far back" with no statement of what it omitted, so
    close to half the attention budget was invisible. Neither curve is a share -
    each entry averages a different number of cells - so a real three-way share
    over one total is reported beside them.
    """
    import numpy as np
    from mimo_bridge.explainer.capture import _bridge_aggregates

    ctx, n_mod = 8, 6
    size = ctx * n_mod
    rng = np.random.default_rng(0)
    bridge = rng.random((size, size))
    bridge /= bridge.sum(1, keepdims=True)      # a real attention matrix

    agg = _bridge_aggregates(bridge, ctx, n_mod)
    assert len(agg["lag"]) == ctx and len(agg["lag_forward"]) == ctx
    # zero distance is the same moment, so the two curves must agree there
    assert abs(agg["lag"][0] - agg["lag_forward"][0]) < 1e-12

    share = agg["direction_share"]
    assert abs(sum(share.values()) - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in share.values())
    # on uniform attention roughly half the budget points forward - the half the
    # old chart dropped
    assert share["forward"] > 0.3


def test_a_causally_masked_matrix_reports_no_forward_attention():
    """The control: mask the future and the forward share must go to zero."""
    import numpy as np
    from mimo_bridge.explainer.capture import _bridge_aggregates

    ctx, n_mod = 6, 3
    size = ctx * n_mod
    rng = np.random.default_rng(1)
    bridge = rng.random((size, size))
    frame_of = np.arange(size) // n_mod
    bridge[frame_of[:, None] < frame_of[None, :]] = 0.0      # no attending later
    bridge /= bridge.sum(1, keepdims=True)

    share = _bridge_aggregates(bridge, ctx, n_mod)["direction_share"]
    assert share["forward"] == 0.0
    assert abs(sum(share.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------- audit P119 (2026-09-18)
def test_goal_pick_records_felt_value_and_curiosity_factor(tmp_path):
    """The stored value is felt x (1 + drive); the capture must publish the winner's
    felt value, its factor, whether it wins unboosted, and the no-curiosity ceiling.
    THE WINNER IS THE PLANNER'S OWN PICK (review b3, minor, 2026-09-22): the model records it in
    last_intention (goal_age_steps / goal_ring_index, the field contract of 2026-09-23) and the
    capture follows it through the ring; without the record it names no goal. Until the model
    lane's hook lands the record is written here the way the contract states it."""
    from mimo_bridge.datasource import SingleEnvSource
    band = {"training": {"curiosity": {"enabled": True, "pe_min": 1e-4, "pe_max": 5.0}}}   # CuriositySystem needs 0 < pe_min
    m = _model(intention=True, intention_execute=False, curiosity_goal=True, overrides=band)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "gp"), tag="_gp", vision=False, seed=5)
    try:
        frames, res, dev = _drive(m, src, 24, train=True)
        assert len(m._goal_ring_drive) == len(m.goal_ring) == len(m._goal_ring_w0) == len(m._goal_ring_cq)
        stored = [float(w[0]) for _, w in m.goal_ring]
        best = max(range(len(stored)), key=stored.__getitem__)
        if not (isinstance(m.last_intention, dict) and "goal_age_steps" in m.last_intention):
            m.last_intention = dict(m.last_intention or {}, step=23,
                                    goal_age_steps=[len(stored) - 1 - best])
        snap = CAP.build_snapshot(
            m, frames, res, run="testrun", step=23, phase="neonate", dev=dev,
            weights="frozen", body=0, eye_layout=None,
            prev_executed=np.array(res.motor_for_action),
            held_order=np.array(res.motor_for_action), order_seq=2, order_age=1,
            flags={"eval_only": True}, check_invariants=True)
        it = snap["intention"]
        assert it["goal_source"] == "recorded", it
        assert abs(it["goal_stored_value"] - it["goal_felt_value"] * it["goal_curiosity_factor"]) < 1e-3
        assert 1.0 <= it["goal_curiosity_factor"] <= 2.0
        assert isinstance(it["goal_wins_without_curiosity"], bool)
        assert it["goal_felt_value"] <= it["stored_value_ceiling_no_curiosity"] + 1e-6
        assert it["best_felt_value"] >= it["goal_felt_value"] - 1e-6
    finally:
        src.close()
    # curiosity off: factor exactly 1 and the felt value IS the stored value
    m2 = _model(intention=True, intention_execute=False, curiosity_goal=False)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "gp2"), tag="_gp2", vision=False, seed=5)
    try:
        frames, res, dev = _drive(m2, src, 24, train=True)
        snap = CAP.build_snapshot(
            m2, frames, res, run="testrun", step=23, phase="neonate", dev=dev,
            weights="frozen", body=0, eye_layout=None,
            prev_executed=np.array(res.motor_for_action),
            held_order=np.array(res.motor_for_action), order_seq=2, order_age=1,
            flags={"eval_only": True}, check_invariants=True)
        it = snap["intention"]
        # this model records no pick and none is written: the capture names no goal and says why
        if not (isinstance(m2.last_intention, dict) and "goal_age_steps" in m2.last_intention):
            assert it["goal_source"] == "not recorded" and "goal_felt_value" not in it
            assert "does not record which remembered model step" in it["goal_index_absent"] or \
                "no plan has been made" in it["goal_index_absent"]
        else:
            assert it["goal_curiosity_factor"] == 1.0
        assert it["best_felt_value"] <= it["stored_value_ceiling_no_curiosity"] + 1e-6
    finally:
        src.close()
    # the deques stay aligned across a reset
    m.reset_transition_state()
    assert len(m._goal_ring_drive) == len(m.goal_ring) == 0



# ---------------------------------------------------------------- audit P076 (2026-09-18)
def test_prediction_error_source_names_the_pairing(tmp_path):
    """The prediction-error wellness bar is two different quantities. With the delayed
    prediction present it is prediction accuracy; on the first step after a reset it is
    1 - mean(head_output^2), which scores whether the heads predicted NO CHANGE. The
    snapshot must say which one produced the bar, or the page draws them as one."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "pes"), tag="_pes",
                          vision=False, seed=3)

    def _snap(step, frames, res, dev):
        return CAP.build_snapshot(m, frames, res, run="t", step=step, phase="neonate",
                                  dev=dev, weights="frozen", body=0, eye_layout=None,
                                  prev_executed=None, held_order=None, order_seq=1,
                                  order_age=0, flags={})
    try:
        frames, res, dev = _drive(m, src, 8, train=True)
        assert "delayed causal" in _snap(7, frames, res, dev)["drives"]["prediction_error_source"]
        m.reset_transition_state()          # an episode boundary or a resume
        frames, res, dev = _drive(m, src, 1, train=True)
        s2 = _snap(8, frames, res, dev)["drives"]["prediction_error_source"]
        assert "same-frame" in s2 and "no change" in s2
    finally:
        src.close()



# ---------------------------------------------------------------- audit P181 (2026-09-18)
def test_the_memory_panel_carries_the_rings_size_and_what_the_gru_read(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    m = _model(persistent_memory="on")
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "mem"), tag="_mem", vision=False, seed=3)
    try:
        frames, res, dev = _drive(m, src, 8, train=True)
        snap = CAP.build_snapshot(
            m, frames, res, run="testrun", step=7, phase="neonate", dev=dev,
            weights="frozen", body=0, eye_layout=None,
            prev_executed=np.array(res.motor_for_action),
            held_order=np.array(res.motor_for_action), order_seq=2, order_age=1,
            flags={"eval_only": True}, check_invariants=True)
        ok, errs = SCH.validate(snap)
        assert ok, errs
    finally:
        src.close()
    lm = m.last_memory
    h = m.memory_history
    assert lm["capacity"] == h.capacity and lm["interval_seconds"] == h.interval_s
    assert lm["window_seconds"] == h.capacity * h.interval_s
    assert lm["encoded_positions"] == len(h.events) + (1 if h.pending is not None else 0) + 1
    assert lm["events"] <= lm["capacity"]


# ================================================================= P080
def test_a_trunk_bodys_snapshot_names_all_26_rows_and_a_mismatch_is_refused(tmp_path):
    """P080: the H81 snapshots carry a 24-name handle list beside 26-row orders. capture.py
    now reads the model's own list, but the test pinning that builds the default 'legacy'
    body, where the model's list IS the 24-name constant, so it cannot tell the two apart.
    This one builds the 26-handle trunk body, and checks that a name list of the wrong
    length stops the snapshot with a message naming both counts."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _model(body_layout="trunk_v1")
    src = SingleEnvSource(scene="crib", run_dir=str(tmp_path / "tk"), tag="_tk", vision=False,
                          seed=3, body_layout="trunk_v1", age=24, caregiver_seconds=0)
    try:
        frames, res, dev = _drive(m, src, 8, train=True)
        kw = dict(run="testrun", step=7, phase="neonate", dev=dev, weights="frozen", body=0,
                  eye_layout=None, prev_executed=np.array(res.motor_for_action),
                  held_order=np.array(res.motor_for_action), order_seq=2, order_age=1,
                  flags={"eval_only": True})
        motor = CAP.build_snapshot(m, frames, res, **kw)["motor"]
        assert len(motor["handle_names"]) == 26 and motor["handle_names"][-2:] == ["lower_trunk", "chest"]
        for key in ("motor_out", "motor_for_action", "prev_executed", "held_order"):
            assert len(motor[key]) == len(motor["handle_names"]), key
        # the regression this guards: the 24-name base list stamped beside 26-row orders
        m.handle_names = list(m.handle_names)[:24]
        with pytest.raises(ValueError, match="24 handle names for 26 command rows"):
            CAP.build_snapshot(m, frames, res, **kw)
    finally:
        src.close()


# ---------------------------------------------------------------- audit P418 (2026-09-19)
def test_the_backward_timing_is_taken_outside_the_training_guard(tmp_path):
    """Audit P418. The inspector labels an --eval-only snapshot's 'backward' time 'after
    the forward pass (no backpropagation ran - the weights are frozen)'. That label is
    true only while timings['backward'] is computed OUTSIDE step()'s `if train:` block.
    The viewer comment that says so cited lem_model.py:2164, which on 2026-09-19 held a
    line of the tenure lesson; this holds the structure and the behaviour instead."""
    import inspect
    import os
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.tools import build_explainer as BX
    lines = inspect.getsource(LemLite.step).splitlines()
    guard = [i for i, l in enumerate(lines) if l == "        if train:"]
    assert len(guard) == 1, "step() no longer has exactly one training guard at its own indentation"
    back = [i for i, l in enumerate(lines) if l.startswith('        timings["backward"] = ')]
    assert len(back) == 1, "timings['backward'] is no longer assigned at step()'s own indentation"
    inside = [i for i, l in enumerate(lines) if "loss.backward()" in l or "self.optimizer.step()" in l]
    assert inside and all(guard[0] < i < back[0] for i in inside), \
        "the backward pass or the optimizer step left the training guard"
    between = [l for l in lines[guard[0] + 1:back[0]] if l.strip() and not l.startswith(" " * 12)]
    assert between == [], f"a statement sits between the guard and the timing at the guard's own depth: {between}"

    torch.set_num_threads(1)
    m = _model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "bt"), tag="_bt", vision=False, seed=3)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        before = [p.detach().clone() for p in m.params]
        res = m.step(frames, dev, train=False, step_index=0, executed_action=None)
        assert "backward" in res.timings_ms, "an eval step reports no 'backward' time for the page to label"
        assert res.grad_norm == 0.0
        assert all(torch.equal(a, b) for a, b in zip(before, m.params)), "an eval step moved a weight"
    finally:
        src.close()
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    assert "after the forward pass (no backpropagation ran - the weights are frozen)" in js


# ---------------------------------------------------------------- H84 items, item 8 (2026-09-19)
def test_the_memory_block_records_the_token_the_recalled_bias_lands_on(tmp_path):
    """The memory diagram drew "a third to a half the size of that token" on every run's
    page - H81's measurement (0.19 to 0.28 against 0.59) - while H84's snapshots read 3.0 to
    4.0 for the bias and recorded nothing about the token under it. The snapshot now carries
    both, for the captured body: the bias's size and the size of the core's newest-moment
    output BEFORE the bias is added. They are held here to the live pass's own tensors."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _model(persistent_memory="on")
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "lt"), tag="_lt", vision=False, seed=3)
    try:
        frames, res, dev = _drive(m, src, 8, train=True)
        # make the bias non-zero: the readout is created at zero, so a fresh model's is 0.0
        with torch.no_grad():
            m._memory_bias_current = m._memory_bias_current + 0.37
        snap = CAP.build_snapshot(
            m, frames, res, run="testrun", step=7, phase="neonate", dev=dev,
            weights="frozen", body=0, eye_layout=None,
            prev_executed=np.array(res.motor_for_action),
            held_order=np.array(res.motor_for_action), order_seq=2, order_age=1,
            flags={"eval_only": True})
        ok, errs = SCH.validate(snap)
        assert ok, errs
    finally:
        src.close()
    mem = snap["memory"]
    assert isinstance(mem, dict) and "landing_token_rms_absent" not in mem, mem
    bias = m._memory_bias_current[0].detach().float()
    _layers, cap_last = CAP.capture_attention_maps(m, body=0)
    landing = cap_last.detach().float() - bias
    assert abs(mem["bias_rms_body"] - float(bias.pow(2).mean().sqrt())) < 1e-4
    assert abs(mem["landing_token_rms"] - float(landing.pow(2).mean().sqrt())) < 1e-4
    # 0.37 was ADDED to a bias the 7 training steps had already moved off zero (measured
    # 0.3734 here), so the size is held to the tensor itself above and only bounded here
    assert 0.3 < mem["bias_rms_body"] < 0.45, "the planted bias of 0.37 is not what was recorded"
    assert mem["landing_token_rms"] > 0.1, "the core's output token cannot be all zeros"
    # memory off: the block is a labelled absence and nothing is added to it
    m2 = _model(persistent_memory="off")
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "lt2"), tag="_lt2", vision=False, seed=3)
    try:
        frames, res, dev = _drive(m2, src, 6, train=True)
        snap2 = CAP.build_snapshot(m2, frames, res, run="t", step=5, phase="neonate", dev=dev,
                                   weights="frozen", body=0, eye_layout=None, prev_executed=None,
                                   held_order=None, order_seq=1, order_age=0, flags={})
    finally:
        src.close()
    assert isinstance(snap2["memory"], str) and snap2["memory"].startswith("absent")


# --------------------------------------------- the six page/code disagreements, 2026-09-21
def test_the_snapshot_carries_the_body_maps_object_incorporation(tmp_path):
    """UEMA 11.2, the capture half. incorporation_stats() has been written into every stats
    record by train_mimo.py, and the snapshot carried none of it - so the page had no panel,
    and the toy's score, the control that says whether that score is noise, and whether the
    world lets the body move the toy at all were readable only by opening the log by hand."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _model(body_layout="trunk_v1", body_schema_updates=True)
    src = SingleEnvSource(scene="crib", run_dir=str(tmp_path / "inc"), tag="_inc", vision=False,
                          seed=3, body_layout="trunk_v1", age=24, caregiver_seconds=0)
    try:
        frames, res, dev = _drive(m, src, 8, train=True)
        snap = CAP.build_snapshot(
            m, frames, res, run="testrun", step=7, phase="neonate", dev=dev,
            weights="frozen", body=0, eye_layout=None,
            prev_executed=np.array(res.motor_for_action),
            held_order=np.array(res.motor_for_action), order_seq=2, order_age=1,
            flags={"eval_only": True})
        ok, errs = SCH.validate(snap)
        assert ok, errs
        live = m.incorporation_stats()
    finally:
        src.close()
    inc = snap["incorporation"]
    assert isinstance(inc, dict), inc
    assert set(inc) == set(live), "the snapshot drops or invents a field of incorporation_stats()"
    for key in ("handles_mine", "extent", "certainty", "carried_by_wellness",
                "objects", "objects_per_body", "objects_evidence", "objects_physics_driven"):
        assert key in inc, f"the panel draws {key} and the snapshot does not carry it"
    from mimo_bridge.lem_model import SELF_OBJECTS
    assert tuple(inc["objects"]) == SELF_OBJECTS
    assert set(inc["objects_evidence"]) == set(inc["objects"])
    assert set(inc["objects_physics_driven"]) == set(inc["objects"])
    for name, e in inc["objects_evidence"].items():
        assert set(e) == {"match", "control", "earned"}, e
        # earned is the match minus its own control, floored at 0 - the subtraction the
        # page's "the difference, which is what counts" column names
        assert abs(e["earned"] - max(0.0, e["match"] - e["control"])) < 1e-4, (name, e)
    assert all(isinstance(v, bool) for v in inc["objects_physics_driven"].values())
    # the two ends of the belonging scale the panel prints, from the map's own temperature
    sc = m._schemas[0]
    floor = float(torch.sigmoid(torch.tensor(sc.temperature * (0.0 - 0.5))))
    top = float(torch.sigmoid(torch.tensor(sc.temperature * (1.0 - 0.5))))
    assert round(floor, 4) == 0.0759 and round(top, 4) == 0.9241, (floor, top)
    for v in inc["objects"].values():
        assert floor - 1e-4 <= v <= top + 1e-4, "a belonging number left the map's own scale"

    # the map switched off is a LABELLED absence, never a missing block
    m2 = _model()                       # body_schema_updates defaults to False
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "inc2"), tag="_inc2",
                          vision=False, seed=3)
    try:
        frames, res, dev = _drive(m2, src, 6, train=True)
        snap2 = CAP.build_snapshot(m2, frames, res, run="t", step=5, phase="neonate", dev=dev,
                                   weights="frozen", body=0, eye_layout=None, prev_executed=None,
                                   held_order=None, order_seq=1, order_age=0, flags={})
    finally:
        src.close()
    assert isinstance(snap2["incorporation"], str)
    assert snap2["incorporation"].startswith("absent")


def test_the_snapshot_names_which_reading_of_the_body_map_the_coherence_score_carried(tmp_path):
    """--wellness-coherence picks between the map's EXTENT and its CERTAINTY, and the two
    move opposite ways; the flag's own help has said "recorded in the snapshot as
    last_coherence_source" since it was added, and the snapshot recorded nothing."""
    from mimo_bridge.datasource import SingleEnvSource
    for choice in ("extent", "certainty"):
        m = _model(body_layout="trunk_v1", body_schema_updates=True, wellness_coherence=choice)
        src = SingleEnvSource(scene="crib", run_dir=str(tmp_path / ("coh" + choice)),
                              tag="_coh" + choice, vision=False, seed=3,
                              body_layout="trunk_v1", age=24, caregiver_seconds=0)
        try:
            frames, res, dev = _drive(m, src, 6, train=True)
            snap = CAP.build_snapshot(m, frames, res, run="t", step=5, phase="neonate", dev=dev,
                                      weights="frozen", body=0, eye_layout=None,
                                      prev_executed=None, held_order=None, order_seq=1,
                                      order_age=0, flags={})
        finally:
            src.close()
        assert snap["drives"]["coherence_source"] == "body_schema_" + choice
        assert snap["incorporation"]["carried_by_wellness"] == choice
        # and the score really is the chosen reading, not the other one
        sc = m._schemas[0]
        want = float(sc.self_extent() if choice == "extent" else sc.coherence_score())
        assert abs(snap["drives"]["wellness_body"]["coherence"] - want) < 5e-3, choice
    # with no body map the term is a third quantity again, and the snapshot says so
    m = _model()
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "cohoff"), tag="_cohoff",
                          vision=False, seed=3)
    try:
        frames, res, dev = _drive(m, src, 6, train=True)
        snap = CAP.build_snapshot(m, frames, res, run="t", step=5, phase="neonate", dev=dev,
                                  weights="frozen", body=0, eye_layout=None, prev_executed=None,
                                  held_order=None, order_seq=1, order_age=0, flags={})
    finally:
        src.close()
    assert snap["drives"]["coherence_source"] == "prediction_cosine"


def test_the_wellness_bars_carry_every_grounded_term_and_the_ceiling_carries_the_sixth(tmp_path):
    """The drives panel draws one bar per entry of last_wellness_grounded, and the plan
    panel's "the most a moment can store" row is capture.py's arithmetic. self_touch joined
    the grounded set on 2026-09-21 and both had been written for five terms."""
    from lem.model.wellness import APPROXIMATE_TERMS, WellnessEvaluators
    from mimo_bridge.datasource import SingleEnvSource
    weight = 0.25
    m = _model(body_layout="trunk_v1", intention=True, intention_execute=False)
    src = SingleEnvSource(scene="crib", run_dir=str(tmp_path / "st"), tag="_st", vision=False,
                          seed=3, body_layout="trunk_v1", age=24, caregiver_seconds=0)
    try:
        frames, res, dev = _drive(m, src, 24, train=True)
        kw = dict(run="t", step=23, phase="neonate", dev=dev, weights="frozen", body=0,
                  eye_layout=None, prev_executed=None, held_order=None, order_seq=1,
                  order_age=0, flags={})
        snap = CAP.build_snapshot(m, frames, res, **kw)
        # the ceiling row is capture.py's own arithmetic over the model's launch weight, so
        # it is checked on the weight itself rather than by running the term's gate
        m.self_touch_wellness_weight = weight
        paid = CAP.build_snapshot(m, frames, res, **kw)
    finally:
        src.close()
    bars = snap["drives"]["wellness_body"]
    assert set(bars) == set(WellnessEvaluators.EVALUATOR_NAMES) | {"self_touch"}, bars
    assert len(bars) == 6, "the page says SIX fixed formulas score a lived model step"
    # the term the planner cannot see: the trained copy has one output per APPROXIMATE_TERMS
    assert "self_touch" not in APPROXIMATE_TERMS
    assert len(APPROXIMATE_TERMS) == 5
    base = snap["intention"]["stored_value_ceiling_no_curiosity"]
    ceiling = paid["intention"]["stored_value_ceiling_no_curiosity"]
    assert abs(ceiling - base - weight) < 1e-6, (
        "the ceiling still counts five evaluators: a fully self-touching model step can add the "
        "--self-touch-wellness weight on top of them")
    assert ceiling >= 5.0 + weight - 1e-6
    assert paid["intention"]["best_felt_value"] <= ceiling + 1e-6


def test_the_three_small_situation_table_gauges_reach_the_page_with_their_digits():
    """Audit P240, the page half: stats() cuts the three centre-movement gauges to 4 significant
    digits (they read 1e-13 to 1e-6), and the snapshot's 5-decimal rounding then printed each as
    0.0. The snapshot keeps those three as stats() gave them and rounds every other number as before."""
    from mimo_bridge.explainer import capture as CAP

    def scalars(d):
        return CAP._r({k: v for k, v in d.items() if isinstance(v, (bool, int, float, str)) or v is None})

    st = {"center_drift": 7.742e-13, "center_drift_max": 6.462e-10, "center_step": 1.122e-06,
          "occ_entropy": 0.98951234, "best_slot_cos": 0.23671234, "slots_filled": 6400, "l2_line": None}
    out = CAP.hself_block(st, scalars)
    assert out["center_drift"] == 7.742e-13 and out["center_drift_max"] == 6.462e-10
    assert out["center_step"] == 1.122e-06
    assert out["occ_entropy"] == 0.98951 and out["best_slot_cos"] == 0.23671      # rounded as before
    assert out["slots_filled"] == 6400 and out["l2_line"] is None
    assert CAP.hself_block({"slots_filled": 0}, scalars) == {"slots_filled": 0}


# ---------------------------------------------------------------- schema v3 (2026-09-22)
# The gain-of-function build. Every v3 block is read from what the model already holds after its
# step: no second forward pass beyond the one attention re-forward, and no *_window_summary(),
# which resets the stats window it summarises and would blank the trainer's next record.
_GOF_FLAGS = dict(body_layout="trunk_v1", intention=True, curiosity_goal=True,
                  causal_chain=True, efference_heads=True, vestibular_sense=True,
                  controllability=True, learning_progress=True, wellness_rank="normalized",
                  curiosity_mode="additive", consequence_self_contact="exclude",
                  contact_head=True, contact_control_wellness=0.5)
_GOF_BAND = {"training": {"curiosity": {"enabled": True, "pe_min": 1e-4, "pe_max": 5.0}}}


def _gof_source(tmp_path, tag):
    from mimo_bridge.datasource import SingleEnvSource
    return SingleEnvSource(scene="crib", run_dir=str(tmp_path / tag), tag="_" + tag, vision=False,
                           seed=3, body_layout="trunk_v1", age=24, caregiver_seconds=0)


def _window_state(m):
    """Every running window the stats records summarise, copied, so a capture that reset one
    is caught."""
    import copy
    wc = getattr(m, "wellness_channels", None)
    out = {"eff": copy.deepcopy(getattr(m, "_eff_window", None))}
    if wc is not None and wc.control is not None:
        out["control"] = copy.deepcopy(wc.control.win)
    if wc is not None and wc.lp is not None:
        out["lp"] = copy.deepcopy(getattr(wc.lp, "win", None))
    tr = getattr(m, "contact_tracker", None)
    if tr is not None:
        out["contact"] = copy.deepcopy(tr.win)
        out["contact_bce"] = copy.deepcopy(tr.bce)
    return out


def _forbid_summaries(m):
    """Every summary that resets a window raises if the capture calls it."""
    def refuse(name):
        def f(*a, **k):
            raise AssertionError(f"the capture called {name}, which resets its window")
        return f
    for name in dir(type(m)):
        if name.endswith("_window_summary"):
            setattr(m, name, refuse(name))
    wc = getattr(m, "wellness_channels", None)
    for obj, label in ((getattr(wc, "control", None), "control.summary"),
                       (getattr(wc, "lp", None), "lp.summary"),
                       (getattr(m, "contact_tracker", None), "contact_tracker.summary")):
        if obj is not None:
            obj.summary = refuse(label)


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def test_a_v3_snapshot_carries_every_build_block_from_cached_state_only(tmp_path):
    m = _model(overrides=_GOF_BAND, **_GOF_FLAGS)
    src = _gof_source(tmp_path, "gof")
    try:
        frames, res, dev = _drive(m, src, 24, train=True)
        before = _window_state(m)
        _forbid_summaries(m)
        calls = []
        fm = m.forward_model
        orig = fm.forward

        def counted(*a, **k):
            calls.append(k.get("capture_attention"))
            return orig(*a, **k)
        fm.forward = counted
        try:
            sent = np.array(res.motor_for_action)
            snap = CAP.build_snapshot(
                m, frames, res, run="testrun", step=23, phase="neonate", dev=dev,
                weights="frozen", body=0, eye_layout=None, prev_executed=sent,
                held_order=sent, order_seq=4, order_age=1, flags={"eval_only": True},
                order_sent=sent, rest_window=False, check_invariants=True)
        finally:
            fm.forward = orig
        after = _window_state(m)
    finally:
        src.close()
    # one forward pass, and it is the attention re-forward
    assert calls == ["all"], f"the capture ran forward passes {calls}; only the attention re-forward is allowed"
    # no running window moved
    for k in before:
        assert _same(before[k], after[k]), f"the capture changed the model's {k} window"
    ok, errs = SCH.validate(snap)
    assert ok, errs
    assert snap["header"]["schema_version"] == SCH.SCHEMA_VERSION == 3
    for blk in SCH.V3_BLOCKS:
        assert blk in snap, blk
    # the vestibular vector, field by field in lived units, read by the model in slot 5
    from mimo_bridge.vestibular_adapter import VESTIBULAR_FIELDS
    ves = snap["vestibular"]
    assert isinstance(ves, dict), ves
    assert [f["name"] for f in ves["fields"]] == list(VESTIBULAR_FIELDS)
    assert ves["read_by_model"] is True
    gyro = next(f for f in ves["fields"] if f["name"].startswith("head_gyro"))
    assert gyro["unit"] == "degrees/s"
    # the efference heads' stamp and window, read as they stand
    eff = snap["efference"]
    assert isinstance(eff, dict) and eff["handle_names"] == list(m.handle_names)
    assert eff["weight"] == m.receipt["efference_heads"]["weight"]
    # the wellness channels: every ranked channel with its centre, scale and signed contribution
    wc = snap["wellness_channels"]
    assert isinstance(wc, dict) and wc["mode"] == "normalized" and wc["curiosity_mode"] == "additive"
    assert wc["consequence_world_only"] is True
    from lem.model import wellness_channels as WCM
    assert wc["gate_line"] == WCM.INTEGRITY_GATE_LINE
    rows = wc["channels"]
    assert isinstance(rows, list) and rows, rows
    names = [r["channel"] for r in rows]
    assert WCM.PAIN_ROW not in names and "contact_control" in names and "controllability" in names
    for r in rows:
        assert set(r) >= {"value", "centre", "scale", "floor", "weight", "weight_in_force",
                          "contribution", "planner_sees"}, r
    assert "controllability" in wc and "learning_progress" in wc
    # foreseen touches: the channel's weight is the launch's
    cc = snap["contact_control"]
    assert isinstance(cc, dict) or cc.startswith("absent"), cc
    if isinstance(cc, dict):
        assert cc["channel_weight"] == 0.5
    # the causal chain: the newest planning event, or its labelled absence
    ch = snap["causal_chain"]
    assert isinstance(ch, dict) and "error" not in ch, ch
    assert ch["body"] == 0 and ch["alternatives"] and ch["step"] is not None
    assert "goal" in ch and "pred_wellness" in ch
    # one value per plan for the captured body, not the whole batch's list
    assert isinstance(ch["pred_wellness"]["chosen"], float), ch["pred_wellness"]
    assert ch["terms"] and len(ch["pred_wellness_terms_chosen"]) == len(ch["terms"])
    # the synergy is off here and says so
    assert snap["synergy"] == "absent: --motor-synergy off"
    # the order really sent and the rest window are recorded apart from the held order
    assert snap["motor"]["order_sent"] == snap["motor"]["held_order"]
    assert snap["motor"]["rest_window"] is False
    # the ceiling is the sum of its named parts
    it = snap["intention"]
    parts = it["stored_value_ceiling_parts"]
    assert {"five_formulas", "self_touch", "movement", "agency", "consequence", "reader",
            "curiosity_added"} <= set(parts)
    assert abs(sum(parts.values()) - it["stored_value_ceiling_no_curiosity"]) < 1e-3


def test_a_model_without_the_build_writes_labelled_absences_and_older_snapshots_still_validate(tmp_path):
    m = _model(body_layout="trunk_v1")
    src = _gof_source(tmp_path, "plain")
    try:
        frames, res, dev = _drive(m, src, 6, train=True)
        snap = CAP.build_snapshot(m, frames, res, run="t", step=5, phase="neonate", dev=dev,
                                  weights="frozen", body=0, eye_layout=None, prev_executed=None,
                                  held_order=None, order_seq=1, order_age=0, flags={})
    finally:
        src.close()
    ok, errs = SCH.validate(snap)
    assert ok, errs
    for blk in ("causal_chain", "efference", "wellness_channels", "contact_control", "synergy"):
        assert isinstance(snap[blk], str) and snap[blk].startswith("absent"), (blk, snap[blk])
    # the vestibular vector is captured whether or not the model reads it
    ves = snap["vestibular"]
    assert isinstance(ves, dict) and ves["read_by_model"] is False
    assert snap["motor"]["order_sent"].startswith("absent") and snap["motor"]["rest_window"] is None
    # a v2 snapshot (no v3 blocks) still validates; a v3 one missing a block does not
    old = {k: v for k, v in snap.items() if k not in SCH.V3_BLOCKS}
    old["header"] = dict(snap["header"], schema_version=2)
    assert SCH.validate(old)[0]
    broken = dict(snap)
    broken.pop("synergy")
    ok, errs = SCH.validate(broken)
    assert not ok and any("synergy" in e for e in errs)


def test_the_synergy_block_reads_the_built_network_and_its_babble_width():
    """--motor-synergy on: the block names the synergy's width and the model's receipt stamp."""
    m = _model(body_layout="trunk_v1", motor_synergy=True)
    blk = CAP.synergy_block(m)
    assert isinstance(blk, dict), blk
    assert blk["dim"] == int(m.synergy.k) > 0
    assert blk["stamp"] == CAP._plain(m.receipt.get("motor_synergy"))


# ---------------------------------------------------------------- review b3-wellness, minor (2026-09-22)
def test_the_goal_is_the_planners_recorded_pick_followed_through_the_ring_never_a_guess():
    """The explainer named the ring moment with the largest RAW stored value as the goal. Under
    --wellness-rank normalized the planner picks the largest NORMALISED value (the review measured
    the oldest moment in 42 of 44 choices), and at any ranking the ring moves on between the plan
    and the snapshot. _goal_index_now reads the model's own record (goal_age_steps per body, or
    goal_ring_index) and follows it by the steps since the plan; without a record it names none."""
    from mimo_bridge.explainer.capture import _goal_index_now
    ring = 200
    # recorded per body, plan on the snapshot's own step: the moment 30 back from the newest
    g = _goal_index_now({"step": 500, "goal_age_steps": [30, 7]}, 1, 500, ring)
    assert g["index"] == ring - 1 - 7 and g["fields"]["goal_source"] == "recorded"
    assert g["fields"]["goal_age_now"] == 7
    # 9 moments later the same goal is 9 moments older (one moment is appended every step)
    g = _goal_index_now({"step": 500, "goal_age_steps": [30, 7]}, 0, 509, ring)
    assert g["index"] == ring - 1 - 39 and g["fields"]["goal_age_now"] == 39
    # one number for every body is read the same way
    assert _goal_index_now({"step": 5, "goal_age_steps": 3}, 2, 5, 20)["index"] == 16
    # a goal that has left the ring since the plan is not named
    g = _goal_index_now({"step": 0, "goal_age_steps": [195]}, 0, 10, ring)
    assert g["index"] is None and "left the ring" in g["fields"]["goal_index_absent"]
    # only an index: usable on the plan's own step, never followed without an age
    assert _goal_index_now({"step": 7, "goal_ring_index": [12]}, 0, 7, ring)["index"] == 12
    g = _goal_index_now({"step": 7, "goal_ring_index": [12]}, 0, 9, ring)
    assert g["index"] is None and "no age" in g["fields"]["goal_index_absent"]
    # no record at all: the goal is not named, and the reason says so - never the raw best
    g = _goal_index_now({"step": 7, "plan_length": 1}, 0, 7, ring)
    assert g["index"] is None and g["fields"]["goal_source"] == "not recorded"
    assert "does not record which remembered model step its planner chose" in g["fields"]["goal_index_absent"]
    assert _goal_index_now(None, 0, 7, ring)["fields"]["goal_source"] == "not recorded"
    # the capture no longer takes a max over the stored values to find the goal
    import inspect
    import os
    from mimo_bridge.explainer import capture as C
    src = inspect.getsource(C.build_snapshot)
    assert "i = max(range(len(stored)), key=stored.__getitem__)" not in src
    assert "_goal_index_now(model.last_intention, b, int(step), len(ring))" in src
    # the page shows a snapshot captured before the fix for what it is
    js = open(os.path.join(os.path.dirname(C.__file__), "viewer.js"), encoding="utf-8").read()
    rows = js[js.index("function goalRows("):js.index("function intentionMini(")]
    assert "not recorded by this snapshot's capture" in rows
    assert "not necessarily the planned goal" in rows


# ---------------------------------------------------------------- field contract, 2026-09-23 (lane b3)
def test_the_wellness_channel_block_reads_the_model_lanes_new_fields_when_present(monkeypatch):
    """The model lane (agents/lane-b3-wellness.md) makes the channel list public
    (rank_channel_names), adds an unranked harm row beside pain (UNRANKED_ROWS), and stores each
    ranked channel in the ring as its distance from the body's running average, the value itself
    in wellness_channels.last_raw. The capture reads those when present and says what the ring
    stores; without them it reads the older names, as before."""
    import os
    from collections import deque
    from types import SimpleNamespace
    from lem.model import wellness_channels as WC

    norm = SimpleNamespace(centre_scale=lambda n: (0.0, 1.0))
    ring = deque([torch.tensor([[0.5, -1.0], [0.25, 0.75], [0.1, 0.2]])])      # [C=3, B=2]
    raw = torch.tensor([[2.5, 1.0], [0.25, 0.75], [0.1, 0.2]])
    wc = SimpleNamespace(normalizer=norm, control=None, lp=None, last_raw=raw)
    new = SimpleNamespace(wellness_channels=wc, wellness_rank_mode="normalized",
                          rank_channel_names=["xa", WC.PAIN_ROW, "harm"], _goal_ring_ch=ring,
                          rank_weights={"xa": 1.0}, _rank_detrend={}, last_tokens=None)
    monkeypatch.setattr(WC, "UNRANKED_ROWS", (WC.PAIN_ROW, "harm"), raising=False)
    out = CAP.wellness_channels_block(new, 1)
    assert [r["channel"] for r in out["channels"]] == ["xa"], "pain and harm are never ranked rows"
    assert out["pain"] == 0.75 and out["harm"] == 0.2
    row = out["channels"][0]
    assert row["value"] == -1.0 and row["raw_value"] == 1.0, "the stored distance and the value itself"
    assert row["contribution"] == -1.0, "the pull is measured on what the ranking ranks: the stored distance"
    assert "running average" in out["stored_as"]

    # the older model: a private channel list, pain the only unranked row, the value stored as is
    monkeypatch.delattr(WC, "UNRANKED_ROWS", raising=False)
    old = SimpleNamespace(wellness_channels=SimpleNamespace(normalizer=norm, control=None, lp=None),
                          wellness_rank_mode="normalized", _rank_channel_names=["xa", WC.PAIN_ROW],
                          _goal_ring_ch=deque([torch.tensor([[0.5, -1.0], [0.25, 0.75]])]),
                          rank_weights={"xa": 1.0}, last_tokens=None)
    out = CAP.wellness_channels_block(old, 0)
    assert [r["channel"] for r in out["channels"]] == ["xa"] and out["pain"] == 0.25
    assert "harm" not in out and out["channels"][0]["raw_value"] is None
    assert out["stored_as"] == "the channel's value"
    # the page says which line gates a goal, harm when the snapshot carries it
    js = open(os.path.join(os.path.dirname(CAP.__file__), "viewer.js"), encoding="utf-8").read()
    panel = js[js.index("function wellnessChannelsPanel("):js.index("function contactPanel(")]
    # (verify-fix-explainer, 2026-09-23: "never a goal" holds only while some remembered moment is
    # under the line - rank_values gives a body whose every moment is over it the least harmful one)
    assert ("the harm line (over it = never a goal while any remembered model step is under it)" in panel
            and "W.harm !== undefined" in panel)
