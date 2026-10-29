# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The contact head on the real H84 checkpoint (DESIGN.md D3, D10, section 10).

With --contact-head on, loading ckpt_step_344400 reinitialises the contact head and NOTHING else,
every saved weight and Adam moment survives, and the model's outputs equal a build with the flag
off (the head adds a loss term and a record only): at one body on four moments, and at TWO bodies
on nine moments that include a moment of the shuffled-order control (review of b4, minor 11: the
first file ran one body only, where the control never runs). Both runs are EVALUATION steps
(train=False): with training on, the head's gradient reaches the core by design from the first
update, so outputs after it differ on purpose; the loss differs by exactly the head's term. The
head is live on the checkpoint's own LIVED moments - the saved 8-body episode's bodies 0-3, six
moments each after their restore, the lead's section-10 sweep of 2026-09-22, with the centre the
running mean the lesson keeps - and stays live after 100 training steps on those inputs (section 10
rule 4; the first file used 13 settling moments of a freshly reset world centred on their own
mean). The harness reloads the checkpoint before each body; the checkpoint holds no contact head,
so each reload reports the head reinitialised, and the test declares it as the first load's fresh
part: the harness then accepts that one part only, and only while the reload leaves the head
unchanged, so one head at its initialisation reads all 24 moments. The deploy's flag set -
vestibular sense, efference heads and the contact head together - loads the checkpoint with its
optimizer. CPU. The checkpoint is found in this tree, the main tree
(a worktree holds no checkpoints) or wherever the archiver moved it (tests/saved_checkpoints.py);
skipped only where it is nowhere."""
from types import SimpleNamespace

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


def _world(args, tmp_path, seed=344400, tag="_t"):
    from mimo_bridge import train_mimo as TM
    from mimo_bridge.datasource import SingleEnvSource
    kw = TM.env_kwargs_for(args, "mature", str(tmp_path))
    kw.update(caregiver_record_dir="", seed=seed, tag=tag)
    return SingleEnvSource(**kw)


def _dev(args, m):
    from mimo_bridge.body_layout import extend_limits
    from lem.training.curriculum import CurriculumManager
    cm = CurriculumManager(steps_per_phase=args.steps_per_phase, mastery_advancement=False,
                           start_phase="mature", report_every=1000)
    return {**extend_limits(cm.get_dev_limits(), len(m.handle_names)), "exploration_noise": 0.05}


def _burst(src, held, gaze, dev, reps):
    from mimo_bridge.env import merge_burst_contact_report
    tasks = []
    frames = None
    for rep in range(reps):
        frames = src.step(held, gaze, dev.get("handle_mask"), render=rep == reps - 1)
        tasks.append(dict(frames[0].meta.get("task") or {}))
    merged = dict(frames[0].meta.get("task") or {})
    merged.update(merge_burst_contact_report(tasks))
    frames[0].meta["task"] = merged
    return frames


def _same_outputs(off, on, srcs, args, n_steps):
    """Step both models on the same frames; every output must match. Returns the steps run."""
    from mimo_bridge import train_mimo as TM
    B = len(srcs)
    dev = _dev(args, off)
    frames = [s.reset()[0] for s in srcs]
    off.reset_context(B)
    on.reset_context(B)
    prev = prev_id = held = None
    seq = 0
    for i in range(n_steps):
        step = 344400 + i
        s0 = torch.get_rng_state().clone()
        ro = off.step(frames, dev, train=False, step_index=step, executed_action=prev,
                      executed_order_id=prev_id)
        after_off = torch.get_rng_state().clone()
        torch.set_rng_state(s0)
        rn = on.step(frames, dev, train=False, step_index=step, executed_action=prev,
                     executed_order_id=prev_id)
        assert torch.equal(after_off, torch.get_rng_state()), f"step {i}: random stream moved"
        assert np.array_equal(ro.motor_out, rn.motor_out), f"step {i}: proposal changed"
        assert np.array_equal(ro.motor_for_action, rn.motor_for_action), f"step {i}: order changed"
        assert np.array_equal(ro.gaze_out, rn.gaze_out)
        assert ro.pe_per_modality == rn.pe_per_modality and ro.wellness == rn.wellness
        assert ro.next_pe_per_modality == rn.next_pe_per_modality
        extra = rn.breakdown.get("contact_head", 0.0)
        assert abs((rn.loss - extra) - ro.loss) <= 1e-5 * max(1.0, abs(ro.loss))
        if i >= 1:
            assert "contact_head" in rn.breakdown, \
                "graded from the second moment: the reset frame is its past"
        if held is None or step % args.motor_hold == 0:
            held, seq = ro.motor_for_action, seq + 1
        frames = [_burst(s, held[b:b + 1], np.asarray(ro.gaze_out)[b:b + 1], dev,
                         max(1, args.action_repeat))[0] for b, s in enumerate(srcs)]
        prev = TM.LemLite.gaze_to_handle(held, np.asarray(ro.gaze_out))
        prev_id = seq
    return n_steps


def test_the_flag_reinitialises_only_the_head_and_changes_no_output_at_one_body(tmp_path):
    torch.set_num_threads(4)
    args, off, r_off = _build(tmp_path)
    _, on, r_on = _build(tmp_path, contact_head="on")
    assert r_off["reinitialized_components"] == []
    assert r_on["reinitialized_components"] == ["contact_head"]
    assert len(on.params) == len(off.params) + len(list(on.contact_head.parameters()))
    for pa, pb in zip(off.params, on.params):
        assert torch.equal(pa, pb)
    saved = torch.load(CKPT / "optimizer.pt", weights_only=False, map_location="cpu")["optimizer"]
    n_saved = len(saved["state"])
    del saved
    new = {id(p) for p in on.contact_head.parameters()}
    have = [p for g in on.optimizer.param_groups for p in g["params"] if on.optimizer.state.get(p)]
    assert len(have) == n_saved and not any(id(p) in new for p in have)
    for pa, pb in zip(off.params, on.params):
        sa, sb = off.optimizer.state.get(pa), on.optimizer.state.get(pb)
        assert (not sa) == (not sb)
        if sa:
            assert torch.equal(sa["exp_avg"], sb["exp_avg"])
            assert torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])
    assert not on.contact_tracker.gate("self")[0] and not on.contact_tracker.gate("world")[0]
    src = _world(args, tmp_path)
    try:
        _same_outputs(off, on, [src], args, 4)
    finally:
        src.close()
    w = on.contact_control_window_summary()
    assert w["body_moments"] == 3 and w["liveness"] is not None and w["control_steps"] == 0


def test_two_bodies_with_a_control_moment_change_no_output(tmp_path):
    """The shuffled-order control runs on moment 344407 (control period 7): a forward of the core
    and the head with another body's size-matched order, no gradient, no draw from any random
    stream the run uses. Nothing the model sends or scores may move."""
    torch.set_num_threads(4)
    args, off, _ = _build(tmp_path, load_optimizer=False)
    _, on, _ = _build(tmp_path, load_optimizer=False, contact_head="on")
    assert on.contact_control_period == 7
    calls = []
    real = on._contact_control_forward

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append(out is not None)
        return out
    on._contact_control_forward = spy
    srcs = [_world(args, tmp_path, seed=344400 + 1000 * b, tag=f"_t{b}") for b in range(2)]
    try:
        _same_outputs(off, on, srcs, args, 9)
    finally:
        for s in srcs:
            s.close()
    assert calls == [True], "the control ran once, on moment 344407"
    w = on.contact_control_window_summary()
    assert w["control_steps"] == 1 and w["body_moments"] == 16


def test_the_head_is_live_on_lived_moments_and_stays_live_after_100_steps(tmp_path, monkeypatch):
    """DESIGN section 10 rule 4 on the checkpoint's own lived moments (bodies 0-3 of the saved
    episode, six graded moments each), centred by the running mean the lesson keeps."""
    torch.set_num_threads(4)
    from mimo_bridge.tools import gof_probe_harness as H
    from mimo_bridge.tools import causal_test as CT
    from lem.model import contact_head as CH
    monkeypatch.setattr(H, "PROBES_ROOT", tmp_path / "probes")
    args, m, res = _build(tmp_path, load_optimizer=False, contact_head="on")
    assert res["reinitialized_components"] == ["contact_head"]
    info = H.resolve_checkpoint(CKPT)
    limits, lsrc = CT._limits_for_run(info["checkpoint"], m)
    # the first load's fresh part, declared: the harness's per-body reload accepts exactly it, and
    # refuses if the reload re-creates it (gof_probe_harness, FRESH PARTS AT A PER-BODY RELOAD)
    fr = H.Frozen(model=m, args=SimpleNamespace(**vars(args)), info=info, limits=dict(limits),
                  limits_source=lsrc, load=dict(reinitialized=list(res["reinitialized_components"])),
                  build_s=0.0, load_s=0.0)
    head = m.contact_head
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    got = []
    fwd = m.contact_head.forward

    def keep(latent, order_feat, prev_state, return_pre=False):
        got.append((latent.detach().clone(), order_feat.detach().clone(), prev_state.detach().clone(),
                    m.contact_head.centre_mean.detach().clone()))
        return fwd(latent, order_feat, prev_state, return_pre=return_pre)
    m.contact_head.forward = keep
    # the target of a graded moment is the skin report of the frames that moment observes (the
    # tracker is never patched: the harness snapshots it and refuses a function inside it)
    targets = []
    starts = 0
    for s, _snap in H.start_states(fr, 4):
        starts += 1
        assert s.episode_body is not None, "a lived moment of the episode, not a fresh world"
        for _ in range(6):
            t = s.frames[0].meta.get("task") or {}
            state = np.asarray(t.get("contact_patch_state_burst", t.get("contact_patch_state")))
            n = len(got)
            s.observe()
            if len(got) > n:
                targets.append(CH.state_bits(torch.as_tensor(state)).numpy()[None])
            s.policy_act()
        m._contact_prev_state = None           # the next body's first moment has no past here
    m.contact_head.forward = fwd
    assert starts == 4 and len(got) == len(targets) >= 20
    # one head at its initialisation read every body's moments: the four reloads left it as it was
    assert m.contact_head is head
    assert all(torch.equal(initial[k], v) for k, v in head.state_dict().items())
    live = m.contact_control_window_summary()["liveness"]
    assert all(v > 0.5 for v in live.values()), live
    lat = torch.cat([g[0] for g in got])
    of = torch.cat([g[1] for g in got])
    ps = torch.cat([g[2] for g in got])
    tb = torch.as_tensor(np.concatenate(targets))
    centre = m.contact_head.centre_mean.clone()

    def live_share():
        with torch.no_grad():
            _, pre = m.contact_head(lat, of, ps, return_pre=True)
        return {k: float((v > CH.LIVE_LINE).reshape(-1, v.shape[-1]).any(0).float().mean())
                for k, v in pre.items()}
    opt = torch.optim.Adam(m.contact_head.parameters(), lr=3e-4)
    valid = torch.ones(lat.shape[0], dtype=torch.bool)
    for _ in range(100):
        terms = CH.ContactHead.loss(m.contact_head(lat, of, ps), tb, valid)
        loss = torch.stack(list(terms.values())).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.equal(m.contact_head.centre_mean, centre), "training moves no running statistic"
    s1 = live_share()
    assert all(v > 0.5 for v in s1.values()), s1


def test_the_deploy_flag_set_loads_the_checkpoint_with_its_optimizer(tmp_path):
    """--vestibular-sense on with the appended heads: before b4's fix the slot switch's optimizer
    migration counted the appended modules as saved positions and refused the load."""
    torch.set_num_threads(4)
    _, m, res = _build(tmp_path, vestibular_sense="on", efference_heads="on", contact_head="on")
    assert set(res["reinitialized_components"]) == {"encoder:vestibular", "efference_heads", "contact_head"}
    sw = m.vestibular_switch
    assert sw["adam_states_saved"] == 406 and sw["adam_states_carried"] == 391
    assert sw["adam_states_dropped_with_old_encoder"] == 15
    assert m.contact_head.ctx_ord.in_features == 142
