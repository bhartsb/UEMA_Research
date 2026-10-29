# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The gain-of-function flags on the real H84 checkpoint (DESIGN.md D3): with
--efference-heads (efference_v2) and --causal-chain on, loading ckpt_step_344400 reinitialises the
efference heads and NOTHING else, every saved weight and every saved Adam moment survives, and the
first steps' orders, predictions and wellness are identical to a build with the flags off (the flags
add loss terms and records only; the gaze copy into vision starts at exactly zero). Then TRAINING
with the flags on is compared (review of b1, MINOR 9: only train=False was): the first training
step's orders, wellness and every loss term but the heads' own are identical, and the update moves
the heads, the gaze copy and the core (the heads' next-moment gradient reaches it) and leaves every
weight finite. CPU, one body, the segment's own world. The checkpoint is found wherever the
archiver moved it (tests/saved_checkpoints.py); skipped only where it is nowhere."""
import numpy as np
import pytest
import torch

from mimo_bridge.tests.saved_checkpoints import h84_344400

H84 = h84_344400()
CKPT = H84.path

pytestmark = [pytest.mark.slow, H84.mark()]


def _build(out, **flags):
    from mimo_bridge.tools.self_recovery_experiment import configuration
    from mimo_bridge import train_mimo as TM
    H84.require()
    args = configuration(CKPT, out)
    args.caregiver_record_dir = ""
    for k, v in flags.items():
        setattr(args, k, v)
    m = TM.build_learner(args, None)
    res = m.load_checkpoint(str(CKPT), load_optimizer=True)
    return args, m, res


def test_flags_on_reinitialise_only_the_efference_heads_and_change_no_first_step(tmp_path):
    torch.set_num_threads(4)
    from mimo_bridge import train_mimo as TM
    from mimo_bridge.body_layout import extend_limits
    from mimo_bridge.causal_chain import CausalChainRecorder
    from mimo_bridge.datasource import SingleEnvSource
    from lem.training.curriculum import CurriculumManager

    args, off, r_off = _build(tmp_path)
    _, on, r_on = _build(tmp_path, efference_heads="on", causal_chain="on")
    assert r_off["reinitialized_components"] == []
    assert r_on["reinitialized_components"] == ["efference_heads"]
    # every saved weight loaded identically; the new ones are the only additions, at the end
    assert len(on.params) == len(off.params) + len(list(on.efference.parameters()))
    for pa, pb in zip(off.params, on.params):
        assert torch.equal(pa, pb)
    # every saved Adam moment carried over by name; none invented for the new heads
    saved = torch.load(CKPT / "optimizer.pt", weights_only=False, map_location="cpu")["optimizer"]
    n_saved = len(saved["state"])
    del saved
    eff = {id(p) for p in on.efference.parameters()}
    have = [p for g in on.optimizer.param_groups for p in g["params"] if on.optimizer.state.get(p)]
    assert len(have) == n_saved and not any(id(p) in eff for p in have)
    for pa, pb in zip(off.params, on.params):
        sa, sb = off.optimizer.state.get(pa), on.optimizer.state.get(pb)
        assert (not sa) == (not sb)
        if sa:
            assert torch.equal(sa["exp_avg"], sb["exp_avg"])
            assert torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])

    cm = CurriculumManager(steps_per_phase=args.steps_per_phase, mastery_advancement=False,
                           start_phase="mature", report_every=1000)
    dev = {**extend_limits(cm.get_dev_limits(), len(off.handle_names)), "exploration_noise": 0.05}
    kw = TM.env_kwargs_for(args, "mature", str(tmp_path))
    kw.update(caregiver_record_dir="", seed=344400)
    src = SingleEnvSource(**kw)
    rec = CausalChainRecorder(str(tmp_path), hold=args.motor_hold, strict=True)
    try:
        frames = src.reset()
        off.reset_context(1)
        on.reset_context(1)
        prev = prev_id = held = None
        seq = 0
        for i in range(3):
            step = 344400 + i
            s0 = torch.get_rng_state().clone()
            ro = off.step(frames, dev, train=False, step_index=step, executed_action=prev,
                          executed_order_id=prev_id)
            after_off = torch.get_rng_state().clone()
            torch.set_rng_state(s0)
            rn = on.step(frames, dev, train=False, step_index=step, executed_action=prev,
                         executed_order_id=prev_id)
            rec.after_step(on, step)
            latched = held is None or step % args.motor_hold == 0
            if latched:
                held, seq = ro.motor_for_action, seq + 1
            rec.on_latch(on, step, latched=latched, order_seq=seq, send_motor=held,
                         gaze_out=ro.gaze_out, source=on.last_action_source)
            assert torch.equal(after_off, torch.get_rng_state()), f"step {i}: random stream moved"
            assert np.array_equal(ro.motor_out, rn.motor_out), f"step {i}: proposal changed"
            assert np.array_equal(ro.motor_for_action, rn.motor_for_action), f"step {i}: order changed"
            assert np.array_equal(ro.gaze_out, rn.gaze_out)
            assert ro.pe_per_modality == rn.pe_per_modality and ro.wellness == rn.wellness
            assert ro.next_pe_per_modality == rn.next_pe_per_modality
            extra = sum(v for k, v in rn.breakdown.items() if k.startswith("efference_"))
            assert abs((rn.loss - extra) - ro.loss) <= 1e-5 * max(1.0, abs(ro.loss))
            if i >= 1:
                assert {"efference_body", "efference_gaze"} <= set(rn.breakdown), \
                    "from the second step the heads are graded"
            reps = max(1, args.action_repeat)          # the trainer's burst
            for rep in range(reps):
                frames = src.step(held, ro.gaze_out, dev.get("handle_mask"), render=rep == reps - 1)
            prev = TM.LemLite.gaze_to_handle(held, np.asarray(ro.gaze_out))
            prev_id = seq
    finally:
        src.close()
    w = on.efference_window_summary()
    assert w["events_next"] == 2 and w["body_cm_do_nothing_next"] > 0
    assert w["version"] == "efference_v2"
    assert all(v > 0.5 for v in w["liveness"].values()), w["liveness"]
    assert w["gaze_copy"]["events"] == 2 and w["gaze_copy"]["share_removed"] == 0.0
    assert rec.window_summary()["latch_predictions"] == 1

    # ---- TRAINING with the flags on, compared (review of b1, MINOR 9)
    src = SingleEnvSource(**kw)
    try:
        frames = src.reset()
        off.reset_context(1)
        on.reset_context(1)
        eff0 = {n: p.detach().clone() for n, p in on.efference.named_parameters()}
        core0 = [p.detach().clone() for p in on.forward_model.parameters()]
        prev = prev_id = held = None
        seq = 0
        for i in range(3):
            step = 344500 + i
            s0 = torch.get_rng_state().clone()
            ro = off.step(frames, dev, train=True, step_index=step, executed_action=prev,
                          executed_order_id=prev_id)
            torch.set_rng_state(s0)
            rn = on.step(frames, dev, train=True, step_index=step, executed_action=prev,
                         executed_order_id=prev_id)
            if i < 2:
                # before the first update that carried the heads' gradient, nothing the body is
                # sent or feels differs (the heads are graded from the second step on)
                assert np.array_equal(ro.motor_for_action, rn.motor_for_action), f"step {i}"
                assert ro.wellness == rn.wellness
            if i == 1:
                shared = {k: v for k, v in ro.breakdown.items() if k in rn.breakdown}
                assert all(abs(shared[k] - rn.breakdown[k]) <= 1e-5 * max(1.0, abs(shared[k]))
                           for k in shared), "a loss term other than the heads' own changed"
            latched = held is None or step % args.motor_hold == 0
            if latched:
                held, seq = ro.motor_for_action, seq + 1
            reps = max(1, args.action_repeat)
            for rep in range(reps):
                frames = src.step(held, ro.gaze_out, dev.get("handle_mask"), render=rep == reps - 1)
            prev = TM.LemLite.gaze_to_handle(held, np.asarray(ro.gaze_out))
            prev_id = seq
    finally:
        src.close()
    moved = [n for n, p in on.efference.named_parameters() if not torch.equal(p, eff0[n])]
    assert any(n.startswith("body_mlp") for n in moved) and any(n.startswith("gaze") for n in moved)
    assert any(n.startswith("vision_out") for n in moved), "the gaze copy into vision learns"
    assert any(not torch.equal(a, b) for a, b in zip(core0, on.forward_model.parameters())),         "the heads' next-moment gradient reaches the core"
    assert all(bool(torch.isfinite(p).all()) for p in on.params)
    assert all(on.optimizer.state.get(p) for p in on.efference.parameters()
               if p.grad is not None), "the heads' Adam moments exist after their first update"
    assert on.efference.centre_weight > 1.0 / 2000, "a training moment moved the running mean"
