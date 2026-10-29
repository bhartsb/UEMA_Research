# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The motor synergy on the real H84 checkpoint (DESIGN.md D3, D11, section 10).

With --motor-synergy on, loading ckpt_step_344400 reinitialises the synergy and NOTHING else;
every saved weight and Adam moment survives. Before the first planning moment (the goal ring
needs 20 moments) the synergy changes no order, and its lesson never reaches the core: the orders
and every existing weight after the same training steps equal a build with the flag off. The
synergy is live on the checkpoint world's own poses and changes at initialisation and stays live
after 100 lesson batches on them (section 10 rule 4). CPU. The checkpoint is found wherever the
archiver moved it (tests/saved_checkpoints.py); skipped only where it is nowhere."""
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


def _live_steps(models, args, src, n, hold=5):
    """Step every model on the SAME frames: the first model's orders drive the body, the trainer's
    latch and ledger. Returns per step each model's order."""
    from mimo_bridge.lem_model import LemLite
    frames = src.reset()
    for m in models:
        m.reset_context(1)
    prev = prev_id = held = None
    seq = 0
    orders = []
    poses = []
    for i in range(n):
        step = 344400 + i
        res = []
        for m in models:
            dev = _dev(args, m)
            torch.manual_seed(1000 + i)
            res.append(m.step(frames, dev, train=True, step_index=step, executed_action=prev,
                              executed_order_id=prev_id))
        orders.append([r.motor_for_action.copy() for r in res])
        poses.append(models[0].last_achieved.detach().float().clone())
        if step % hold == 0 or held is None:
            held, seq = res[0].motor_for_action, seq + 1
        gz = []
        for rep in range(args.action_repeat):
            frames = src.step(held, res[0].gaze_out, None, render=rep == args.action_repeat - 1)
            gz.append(np.asarray(frames[0].meta.get("gaze_cmd") or (0.0, 0.0), dtype=np.float32))
        prev = LemLite.gaze_to_handle(held, np.mean(np.stack(gz), axis=0)[None])
        prev_id = seq
    return orders, poses


def test_the_flag_reinitialises_only_the_synergy_and_changes_nothing_before_the_first_plan(tmp_path):
    torch.set_num_threads(4)
    args, off, r_off = _build(tmp_path)
    _, on, r_on = _build(tmp_path, motor_synergy="on")
    assert r_off["reinitialized_components"] == []
    assert r_on["reinitialized_components"] == ["motor_synergy"]
    assert len(on.params) == len(off.params) + len(list(on.synergy.parameters()))
    for pa, pb in zip(off.params, on.params):
        assert torch.equal(pa, pb)
    have_off = sum(1 for g in off.optimizer.param_groups for p in g["params"] if off.optimizer.state.get(p))
    have_on = sum(1 for g in on.optimizer.param_groups for p in g["params"] if on.optimizer.state.get(p))
    assert have_on == have_off and have_off > 0, "every saved Adam moment carried by name"
    assert not any(on.optimizer.state.get(p) for p in on.synergy.parameters())
    # the same frames through both: the orders before the first plan, and every existing weight
    # after six training steps (one order's tenure graded by the lesson), are equal
    src = _world(args, tmp_path)
    try:
        orders, _ = _live_steps([off, on], args, src, 7)
    finally:
        src.close()
    for i, (a, b) in enumerate(orders):
        assert np.array_equal(a, b), f"step {i}: the synergy changed an order before planning"
    for pa, pb in zip(off.params, on.params):
        assert torch.equal(pa, pb), "the synergy's lesson reached a weight it must not"
    w = on.synergy_window_summary()
    assert w["orders"] >= 1 and w["bank"] >= 1


def test_the_synergy_is_live_on_the_worlds_own_poses_and_stays_live(tmp_path):
    torch.set_num_threads(4)
    from lem.model import motor_synergy as MS
    args, on, _ = _build(tmp_path, load_optimizer=False, motor_synergy="on")
    src = _world(args, tmp_path, seed=344401)
    try:
        _, poses = _live_steps([on], args, src, 31)
    finally:
        src.close()
    syn = on.synergy
    P = torch.cat(poses, 0)                                     # [31, H, 10]
    a0, a1 = P[:-5], P[5:]
    rot, pos = syn.achieved_change(a0, a1)
    x = syn.to_units(rot, pos)
    assert float(x.pow(2).mean().sqrt()) > 0.05, "the body moved"

    def live():
        pre = {}
        with torch.no_grad():
            z = syn.encode(x, a0, pre=pre)
            syn.delta(z + torch.randn_like(z), a0, pre=pre)
        return {k: float((v > MS.LIVE_LINE).any(0).float().mean()) for k, v in pre.items()}
    l0 = live()
    assert all(v > 0.5 for v in l0.values()), l0
    opt = torch.optim.AdamW(syn.parameters(), lr=3e-4, weight_decay=0.01)
    w = torch.ones(len(x))
    for _ in range(100):
        z = syn.encode(x, a0)
        xh = syn.from_bounded(*syn.delta(z, a0))
        loss = (w * (xh - x).pow(2).mean(-1)).sum() / w.sum() + MS.Z_L2 * z.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    l1 = live()
    assert all(v > 0.5 for v in l1.values()), l1
