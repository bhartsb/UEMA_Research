# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The liveness repair on the real H84 checkpoint (DESIGN.md D3, section 10 rule 4).

With --liveness-repair on, loading ckpt_step_344400 reinitialises the repair module and nothing
else, and every saved weight and Adam moment is carried; with "report" nothing is reinitialised.
In the segment 31 world (CPU, one body, training, the checkpoint's own optimizer): the first moment's
order is the one the model gives without the repair; each of the nine dead layers is revived with
its output changed by less than the largest change its dead units could make (measured at the
switch), the revived units respond on the watched inputs, and after 100 more training steps each
revived layer still has more than half its units responding. The learned motor and touch bypasses
must carry influence from every element on actual pool inputs. Comparing the entropy of two
differently trained encoders does not isolate the bypass or establish useful representation.

The checkpoint is found in this tree, the main tree (a worktree holds no checkpoints) or wherever
the archiver moved it (tests/saved_checkpoints.py); UEMA_CKPT_344400 still names any other place.
Skipped only where it is nowhere."""
import os

import numpy as np
import pytest
import torch

from mimo_bridge.tests.saved_checkpoints import h84_344400

H84 = h84_344400()
CKPT = H84.path

pytestmark = [pytest.mark.slow, H84.mark()]


def _build(out, load_optimizer=True, **flags):
    from mimo_bridge.tools.self_recovery_experiment import configuration
    from mimo_bridge import train_mimo as TM
    H84.require()
    args = configuration(CKPT, out)
    args.device = os.environ.get('UEMA_LIVENESS_TEST_DEVICE', args.device)
    args.caregiver_record_dir = ""
    for k, v in flags.items():
        setattr(args, k, v)
    m = TM.build_learner(args, None)
    res = m.load_checkpoint(str(CKPT), load_optimizer=load_optimizer)
    return args, m, res


def _world(args, tmp_path, seed=344400):
    from mimo_bridge import train_mimo as TM
    from mimo_bridge.datasource import SingleEnvSource
    kw = TM.env_kwargs_for(args, "mature", str(tmp_path))
    kw.update(caregiver_record_dir="", seed=seed)
    return SingleEnvSource(**kw)


def _dev(args, m, noise=0.05):
    from mimo_bridge.body_layout import extend_limits
    from lem.training.curriculum import CurriculumManager
    cm = CurriculumManager(steps_per_phase=args.steps_per_phase, mastery_advancement=False,
                           start_phase="mature", report_every=1000)
    return {**extend_limits(cm.get_dev_limits(), len(m.handle_names)), "exploration_noise": noise}


def _live_steps(models, args, src, n, hold=5, on_step=None):
    """Step every model on the SAME frames: the first model's orders drive the body, the trainer's
    latch and ledger (tests/test_gof_b5_checkpoint.py). Returns per step each model's order."""
    from mimo_bridge.lem_model import LemLite
    frames = src.reset()
    for m in models:
        m.reset_context(1)
    prev = prev_id = held = None
    seq = 0
    orders = []
    for i in range(n):
        step = 344400 + i
        res = []
        for m in models:
            dev = _dev(args, m)
            torch.manual_seed(1000 + i)
            res.append(m.step(frames, dev, train=True, step_index=step, executed_action=prev,
                              executed_order_id=prev_id))
        orders.append([r.motor_for_action.copy() for r in res])
        if on_step is not None:
            on_step(i)
        if step % hold == 0 or held is None:
            held, seq = res[0].motor_for_action, seq + 1
        gz = []
        for rep in range(args.action_repeat):
            frames = src.step(held, res[0].gaze_out, None, render=rep == args.action_repeat - 1)
            gz.append(np.asarray(frames[0].meta.get("gaze_cmd") or (0.0, 0.0), dtype=np.float32))
        prev = LemLite.gaze_to_handle(held, np.mean(np.stack(gz), axis=0)[None])
        prev_id = seq
    return orders


def test_the_flag_reinitialises_only_the_repair_and_keeps_every_adam_moment(tmp_path):
    torch.set_num_threads(4)
    from lem.model import liveness as LV
    _, off, r_off = _build(tmp_path)
    _, rep, r_rep = _build(tmp_path, liveness_repair="report")
    assert r_off["reinitialized_components"] == []
    assert r_rep["reinitialized_components"] == []
    assert len(rep.params) == len(off.params)
    del rep
    _, on, r_on = _build(tmp_path, liveness_repair="on")
    assert r_on["reinitialized_components"] == ["liveness_repair"]
    assert on.liveness.layer_names == [n for n in LV.REPAIR_LAYERS if LV.layer_parts(on,n) is not None]
    assert set(on.liveness.paths) == {"motor", "tactile"}
    extra = list(on.liveness.parameters())
    assert len(on.params) == len(off.params) + len(extra)
    carried = 0
    for pa, pb in zip(off.params, on.params):
        assert torch.equal(pa, pb)
        sa, sb = off.optimizer.state.get(pa), on.optimizer.state.get(pb)
        assert (not sa) == (not sb)
        for k in (sa or {}):
            va, vb = sa[k], sb[k]
            assert (torch.equal(va, vb) if torch.is_tensor(va) else va == vb), k
        carried += bool(sa)
    assert carried > 0, "every saved Adam moment carried by name"
    assert not any(on.optimizer.state.get(p) for p in extra)
    # the probes' execution snapshot can copy a model built with the repair (plain attributes only)
    from mimo_bridge import copied_execution
    snap = copied_execution.capture(on)
    copied_execution.restore(on, snap)
    # a checkpoint saved with the repair on refuses a load without it (what the revived units and
    # the pool paths learned lives in the repair module): judged from the receipt, before any weight
    import json
    fake = tmp_path / "ckpt_step_1"
    fake.mkdir()
    (fake / "model_receipt.json").write_text(json.dumps(
        {"body_layout": off.body_layout, "liveness_repair": on.liveness.stamp()}), encoding="utf-8")
    with pytest.raises(ValueError, match="liveness-repair on"):
        off.load_checkpoint(str(fake), load_optimizer=False)


def test_revival_keeps_outputs_and_the_repaired_parts_stay_live(tmp_path, monkeypatch):
    torch.set_num_threads(4)
    from lem.model import liveness as LV
    # a short watching window for the test (the run's is LV.REVIVE_MIN_SAMPLES), and every step
    # sampled so each window's report rests on every moment
    monkeypatch.setattr(LV, "REVIVE_MIN_SAMPLES", 8)
    monkeypatch.setattr(LV, "SAMPLE_EVERY", 1)
    args, on, _ = _build(tmp_path, liveness_repair="on")
    _, ctl, _ = _build(tmp_path, liveness_repair="report")
    assert on.liveness.revive_min_samples == 8 and ctl.liveness.sample_every == 1
    records, windows = {}, []
    first_revival = {}
    all_repairs = []
    pool_inputs = {}

    def on_step(i):
        rec = dict(on.liveness.revival_record)
        if rec:
            records.update(rec)
            on.liveness.revival_record = {}
            for name, record in rec.items():
                first_revival.setdefault(name, i)
                all_repairs.append((name, record))
        if (i + 1) % 20 == 0:
            for enc, inputs in on.liveness.read_inputs.items():
                if inputs:
                    pool_inputs[enc] = inputs[-1].detach().clone()
            windows.append((i, on.liveness.window_summary(on), ctl.liveness.window_summary(ctl)))

    src = _world(args, tmp_path)
    try:
        # the tenure pose head is read at order boundaries only (one in five moments, two rows
        # each: the order and "do nothing"), so its window fills last, near moment 20
        orders = _live_steps([on, ctl], args, src, 140, on_step=on_step)
    finally:
        src.close()
    assert np.array_equal(orders[0][0], orders[0][1]), "the repair changed the first order"
    assert set(LV.SWEEP_2026_09_22).issubset(records), sorted(records)
    for name, r in all_repairs:
        assert r["dead"] > 0, (name, r)
        # function preservation at the switch: under the largest change the dead units could make
        assert r["output_change_max"] <= r["output_change_bound"] + 1e-6, (name, r)
        assert r["revived_live_share_on_probe"] > 0.5, (name, r)
    original_last = max(first_revival[n] for n in LV.SWEEP_2026_09_22)
    assert original_last <= 39, first_revival
    # the windows after 100 more training steps: every revived layer above the section 10 line
    i_end, w_on, w_ctl = windows[-1]
    assert i_end - original_last >= 100
    for name in on.liveness.layer_names:
        assert w_on["layers"][name] > LV.LAYER_LINE, (name, w_on["layers"][name])
    # Rule 3 requires a learned route from every element. The old entropy comparison mixed
    # this effect with changes to the encoders' other weights; uniform attention can still be
    # useless. Retain the comparative readings as evidence, without making that false assertion.
    import json
    (tmp_path / 'pool_comparison.json').write_text(json.dumps(dict(
        repaired=w_on['token_read'], control=w_ctl['token_read'],
        claim='Descriptive only: the two encoders have different trained weights.')))
    for enc in ("motor", "tactile"):
        path = on.liveness.paths[enc]
        x = pool_inputs[enc].to(path.up.weight.device).requires_grad_(True)
        y = path(x)
        assert torch.isfinite(y).all() and y.square().sum() > 0, enc
        influence, = torch.autograd.grad(y.square().sum(), x)
        assert torch.isfinite(influence).all(), enc
        assert bool((influence.square().sum((0, 2)) > 0).all()), enc
