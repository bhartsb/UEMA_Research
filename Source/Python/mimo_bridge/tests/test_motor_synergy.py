# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The motor synergy, babble in its intention space and the smoother's source (DESIGN.md D11).

What is pinned here: z = 0 asks for no change and the planner still has a gradient there; every
change stays inside its handle's measured reach; the eyes are never in the synergy; a turn asked
through the synergy is the turn the order carries; the following gauge reads the environment's
definition; the hindsight lesson grades only an order that ran its whole tenure and teaches only
the synergy; babble drawn in z turns each body handle as far on average as the independent wobble;
the planner's descent runs on z; --smooth-source executed blends with the order the body was given
and passes the eyes' row, and the planner imagines the gaze the servo is sent; and the
registration rules every new module follows (appended last in a forked random stream, side
module, own clip group, receipt stamp, refusals, exact scale migration, runtime state, episode
companion)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lem.model import motor_synergy as MS
from lem.model.motor_synergy import (MotorSynergy, SynergyActionSpace, following_cos,
                                     participation_ratio, quat_mul, rotvec_to_quat, turn_between_rad)
from mimo_bridge.lem_model import LemLite
from mimo_bridge.body_layout import names_for
from mimo_bridge.action_decoder import POSITION_HANDLES

NAMES = list(names_for("trunk_v1"))
GI = NAMES.index("gaze")


def _pose(B, seed=0):
    """[B, 26, 10] a plausible achieved pose: positions spread over ~40 cm (in units of 200 cm),
    random unit quaternions."""
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(B, len(NAMES), 10)
    p[..., 0:3] = 0.3 + 0.1 * torch.randn(B, len(NAMES), 3, generator=g)
    q = torch.randn(B, len(NAMES), 4, generator=g)
    p[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    p[..., 8] = 0.5
    return p


def _syn(k=None, seed=0):
    torch.manual_seed(seed)
    return MotorSynergy(NAMES, GI, sorted(POSITION_HANDLES), int(k or MS.SYNERGY_DIM))


def _order_cm(pose):
    """The raw pose written as an execution-form order (cm), strength 0.5."""
    o = torch.zeros(pose.shape[0], pose.shape[1], 9)
    o[..., 0:3] = pose[..., 0:3] * 200.0
    o[..., 3:7] = pose[..., 3:7]
    o[..., 8] = 0.5
    return o


# ------------------------------------------------------------------------------ the module
def test_every_constant_is_measured_and_an_unmeasured_one_refuses_to_build(monkeypatch):
    assert MS.unmeasured() == []
    assert 8 <= MS.SYNERGY_DIM <= 32
    for n in NAMES:
        if n != "gaze":
            assert 0.0 < MS.ROT_REACH_DEG[n] <= 180.0, n
    for n in POSITION_HANDLES:
        assert MS.POS_REACH_CM[n] > 0.0, n
    for key in ("date", "source"):
        assert key in MS.SYNERGY_DIM_MEASUREMENT and key in MS.CHANGE_SCALE_MEASUREMENT
        assert key in MS.REACH_MEASUREMENT and key in MS.Z_L2_MEASUREMENT
    monkeypatch.setattr(MS, "Z_L2", None)
    with pytest.raises(ValueError, match="Z_L2"):
        _syn()


def test_z_zero_asks_for_no_change_and_the_planner_has_a_gradient_there():
    s = _syn()
    pose = _pose(3)
    z = torch.zeros(3, s.k, requires_grad=True)
    rot, pos = s.delta(z, pose)
    assert float(rot.abs().max()) == 0.0 and float(pos.abs().max()) == 0.0
    (rot.sum() + pos.sum()).backward()
    assert float(z.grad.abs().sum()) > 0.0, "a zero last layer would leave the planner no gradient"
    # and the fresh synergy asks for SMALL changes at unit z (the last layer starts small)
    with torch.no_grad():
        r1, _ = s.delta(torch.randn(64, s.k), _pose(64, 1))
    assert float(r1.norm(dim=-1).mean()) < 0.05, "a fresh synergy should not command big postures"


def test_every_change_stays_inside_its_handles_reach():
    s = _syn()
    pose = _pose(4)
    with torch.no_grad():
        rot, pos = s.bound(1e4 * torch.randn(4, s.n_out))
    reach_r = torch.tensor([MS.ROT_REACH_DEG[NAMES[i]] for i in s.rot_idx]) * math.pi / 180
    reach_p = torch.tensor([MS.POS_REACH_CM[NAMES[i]] for i in s.pos_idx])
    assert bool((rot.norm(dim=-1) <= reach_r.view(1, -1) + 1e-5).all())
    assert bool((pos.abs() <= reach_p.view(1, -1, 1) + 1e-5).all())
    # well inside the reach the bound passes the change nearly unchanged
    u = 1e-3 * torch.ones(1, s.n_out)
    rot, pos = s.bound(u)
    assert torch.allclose(rot.reshape(1, -1), u[:, :3 * s.n_rot] * s.rot_scale_rad, rtol=1e-3)


def test_the_eyes_are_never_in_the_synergy():
    s = _syn()
    assert GI not in s.rot_idx and GI not in s.pos_idx
    assert s.n_rot == 25 and s.n_pos == 14 and s.n_out == 117
    pose = _pose(2)
    tmpl = torch.randn(2, len(NAMES), 9)
    rot, pos = s.delta(torch.randn(2, s.k), pose)
    o = s.order_from_delta(rot, pos, pose, tmpl)
    assert torch.equal(o[:, GI], tmpl[:, GI])
    base = _order_cm(pose)
    p = s.perturb_order(base, rot, pos)
    assert torch.equal(p[:, GI], base[:, GI])


def test_holding_still_is_the_current_pose_and_an_asked_turn_is_the_turn_the_order_carries():
    s = _syn()
    pose = _pose(2)
    tmpl = torch.zeros(2, len(NAMES), 9)
    zr = torch.zeros(2, s.n_rot, 3)
    zp = torch.zeros(2, s.n_pos, 3)
    o = s.order_from_delta(zr, zp, pose, tmpl)
    assert torch.allclose(o[:, s.rot_idx, 0:3], pose[:, s.rot_idx, 0:3])
    q0 = pose[:, s.rot_idx, 3:7]
    assert torch.allclose(turn_between_rad(q0, o[:, s.rot_idx, 3:7]), torch.zeros(2, 25, 3), atol=1e-5)
    rot = 0.3 * torch.randn(2, s.n_rot, 3)
    pos = torch.randn(2, s.n_pos, 3)
    o = s.order_from_delta(rot, pos, pose, tmpl)
    assert torch.allclose(turn_between_rad(q0, o[:, s.rot_idx, 3:7]), rot, atol=1e-4)
    assert torch.allclose((o[:, s.pos_idx, 0:3] - pose[:, s.pos_idx, 0:3]) * 200.0, pos, atol=1e-4)
    # and the achieved change of a body that did exactly that reads the same numbers back
    after = pose.clone()
    after[:, :, 0:7] = o[:, :, 0:7]
    r2, p2 = s.achieved_change(pose, after)
    assert torch.allclose(r2, rot, atol=1e-4) and torch.allclose(p2, pos, atol=1e-4)


def test_following_reads_one_for_the_asked_direction_and_minus_one_against():
    s = _syn()
    pose = _pose(1)
    rot = 0.2 * torch.randn(1, s.n_rot, 3)
    order = s.order_from_delta(rot, torch.zeros(1, s.n_pos, 3), pose, torch.zeros(1, len(NAMES), 9),
                               strength=torch.full((1, len(NAMES), 1), 0.5))
    order_cm = order.clone()
    order_cm[..., 0:3] *= 200.0
    half = pose.clone()
    half[:, s.rot_idx, 3:7] = quat_mul(rotvec_to_quat(0.5 * rot), pose[:, s.rot_idx, 3:7])
    back = pose.clone()
    back[:, s.rot_idx, 3:7] = quat_mul(rotvec_to_quat(-0.5 * rot), pose[:, s.rot_idx, 3:7])
    assert abs(float(following_cos(order_cm, pose, half, s.rot_idx)) - 1.0) < 1e-4
    assert abs(float(following_cos(order_cm, pose, back, s.rot_idx)) + 1.0) < 1e-4
    order_cm[..., 8] = 0.0
    assert float(following_cos(order_cm, pose, half, s.rot_idx)) == 0.0, "no strength: not scored"


def test_the_participation_ratio_counts_the_directions_used():
    g = torch.Generator().manual_seed(0)
    line = torch.randn(500, 1, generator=g) * torch.tensor([[1.0, 2.0, 0.5, 0.0]])
    assert abs(participation_ratio(line) - 1.0) < 1e-3
    iso = torch.randn(20000, 8, generator=g)
    assert 7.5 < participation_ratio(iso) <= 8.0


def test_the_pose_centre_learns_a_shared_offset_from_lived_moments():
    s = _syn()
    pose = _pose(4)
    s.update_centre(pose)
    assert s.centre_weight > 0.0
    assert torch.allclose(s.centre_mean, s.pose_features(pose).mean(0), atol=1e-5), \
        "the first moment sets the mean to itself (corrected for starting at zero)"


def test_the_planners_space_starts_from_encoded_proposal_and_charges_a_residual_only_when_asked():
    s = _syn()
    pose = _pose(2)
    tmpl = torch.zeros(2, len(NAMES), 9)
    tmpl[..., 6] = 1.0
    tmpl[..., 8] = 0.4
    sp = SynergyActionSpace(s, pose, tmpl)
    params = sp.init()
    assert len(params) == 3 and sp.res is None
    order, strength, pen = sp.assemble()
    assert pen is None and torch.allclose(strength, torch.full_like(strength, 0.4), atol=1e-4)
    rot, pos = s.achieved_change(pose, tmpl)
    assert torch.equal(sp.z.detach(), s.encode(s.to_units(rot, pos), pose).detach())
    assert sp.z.is_leaf and sp.z.requires_grad
    assert not torch.equal(sp.z, torch.zeros_like(sp.z)), "a nonzero proposal must not seed hold"
    sp2 = SynergyActionSpace(s, pose, tmpl, residual_l2=0.5)
    assert len(sp2.init()) == 4
    with torch.no_grad():
        sp2.res.fill_(1.0)
    _, _, pen = sp2.assemble()
    assert abs(float(pen) - 0.5 * s.n_out) < 1e-4


# ------------------------------------------------------------------------------ the model
def _lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True,
                   next_frame_loss=True, body_layout="trunk_v1", **kw)


def test_flag_off_builds_nothing_and_flag_on_appends_last_registers_and_stamps():
    off = _lite()
    on = _lite(motor_synergy=True)
    assert off.synergy is None and "motor_synergy" not in off._side_modules()
    assert "motor_synergy" not in off.receipt and "smooth_source" not in off.receipt
    # Eye requests bypass the body smoother in both modes, independently of synergy.
    planner = _lite(intention=True)
    assert planner._capture_plan_state()["smooth_keep_gaze"] == planner.GAZE_IDX
    n = sum(p.numel() for p in on.synergy.parameters())
    assert on.n_params == off.n_params + n
    assert [p.shape for p in on.params[:len(off.params)]] == [p.shape for p in off.params]
    ids = {id(p) for p in on.synergy.parameters()}
    assert all(id(p) in ids for p in on.params[len(off.params):]), "appended LAST"
    assert on._side_modules()["motor_synergy"] is on.synergy
    st = on.receipt["motor_synergy"]
    assert st["k"] == MS.SYNERGY_DIM and st["change_scale_deg"] == MS.CHANGE_SCALE_DEG
    assert st["planner"] is True and st["babble"] is False and st["residual_l2"] is None
    assert on.synergy.k == MS.SYNERGY_DIM
    assert _lite(motor_synergy=True, synergy_dim=9).synergy.k == 9
    # built in a forked random stream: the run's own stream is where a build without it leaves it
    torch.manual_seed(5)
    _lite()
    a = torch.rand(3)
    torch.manual_seed(5)
    _lite(motor_synergy=True)
    assert torch.equal(a, torch.rand(3))


def test_what_is_refused():
    with pytest.raises(ValueError, match="--motor-synergy on"):
        _lite(synergy_babble=True)
    with pytest.raises(ValueError, match="--motor-synergy on"):
        _lite(synergy_residual_l2=0.1)
    with pytest.raises(ValueError, match="above 0"):
        _lite(motor_synergy=True, synergy_residual_l2=0.0)
    with pytest.raises(ValueError, match="smooth_source"):
        _lite(smooth_source="average")


def _run_lesson(m, poses, orders, ids, start=100, train=True):
    """Feed _synergy_lesson one moment at a time: the pose each moment produced and the order in
    force while it did (with its serial)."""
    outs = []
    for t, (p, o, i) in enumerate(zip(poses, orders, ids)):
        outs.append(m._synergy_lesson(o, i, p, start + t, p.shape[0], train))
    return outs


def test_the_lesson_grades_only_an_order_that_ran_its_whole_tenure_and_teaches_only_the_synergy():
    torch.manual_seed(1)
    m = _lite(motor_synergy=True)
    B = 2
    p0 = _pose(B, 3)
    s = m.synergy
    # an order asking a turn, followed exactly by a body that turns half-way over five moments
    rot = 0.2 * torch.randn(B, s.n_rot, 3)
    ask = s.order_from_delta(rot, torch.zeros(B, s.n_pos, 3), p0, torch.zeros(B, len(NAMES), 9),
                             strength=torch.full((B, len(NAMES), 1), 0.5))
    ask[..., 0:3] *= 200.0
    poses = [p0.clone()]
    for j in range(1, 7):
        pj = p0.clone()
        pj[:, s.rot_idx, 3:7] = quat_mul(rotvec_to_quat(rot * min(j, 5) / 10.0), p0[:, s.rot_idx, 3:7])
        poses.append(pj)
    hold = _order_cm(p0)
    orders = [hold, ask, ask, ask, ask, ask, hold]
    ids = [1, 2, 2, 2, 2, 2, 3]
    outs = _run_lesson(m, poses, orders, ids)
    graded = [i for i, o in enumerate(outs) if o]
    assert graded == [5], "graded once, at the moment its fifth moment in force ended"
    loss = outs[5]["synergy"]
    for p in m.params:
        p.grad = None
    loss.backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0
               for p in m.forward_model.parameters()), "the lesson never reaches the core"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.synergy.parameters())
    w = m.synergy_window_summary()
    assert w["orders"] == B and abs(w["following_mean"] - 1.0) < 1e-3
    assert w["movement_mean"] > 0 and w["bank"] == B and set(w["liveness"]) == set(MotorSynergy.LIVE_LAYERS)
    assert w["explained_share"] is not None and w["effective_dim"] is not None
    # an order cut short (a new serial after three moments) is never graded
    m2 = _lite(motor_synergy=True)
    outs = _run_lesson(m2, poses[:5], [hold, ask, ask, ask, hold], [1, 2, 2, 2, 3])
    assert not any(outs)


def test_a_reset_forgets_the_order_in_progress_and_keeps_the_samples():
    m = _lite(motor_synergy=True)
    m._syn_pending = {"ach": _pose(2), "order": _order_cm(_pose(2)), "step": 3}
    m._syn_prev_ach = _pose(2)
    m._smooth_exec_prev = _order_cm(_pose(2))
    m._syn_bank = {"ach": torch.zeros(4, 26, 10), "x": torch.zeros(4, 117), "w": torch.zeros(4),
                   "n": 2, "ptr": 2}
    m._synergy_sigma = 3.0
    m.reset_transition_state()
    assert m._syn_pending is None and m._syn_prev_ach is None and m._smooth_exec_prev is None
    assert m._syn_bank["n"] == 2 and m._synergy_sigma == 3.0


def test_babble_in_z_turns_each_handle_as_far_as_the_independent_wobble():
    torch.manual_seed(2)
    m = _lite(motor_synergy=True, synergy_babble=True)
    B = 8
    pose = _pose(B, 4)
    mfa = _order_cm(pose)
    cur_cm = pose[..., 0:3] * 200.0
    for ns in (0.05, 0.02):
        out = m._synergy_babble(mfa.clone(), pose, ns, cur_cm, 40.0)
        rot = turn_between_rad(mfa[:, m.synergy.rot_idx, 3:7], out[:, m.synergy.rot_idx, 3:7])
        target = 10.0 * ns * MS.CHI3_MEAN
        assert abs(float(rot.norm(dim=-1).mean()) - target) < 0.02 * target
        assert torch.equal(out[:, GI, 3:7], mfa[:, GI, 3:7]), "the eyes get no babble from it"
        assert bool(((out[..., 0:3] - cur_cm).abs() <= 40.0 + 1e-4).all()), "the reach is re-clamped"
    w = m.synergy_window_summary()
    assert w["babble_steps"] == 2 and w["babble_sigma"] > 0
    # the independent wobble's own mean turn is the number babble in z is sized to
    g = torch.Generator().manual_seed(0)
    rv = torch.randn(200000, 3, generator=g) * 0.5
    assert abs(float(rv.norm(dim=-1).mean()) - 10.0 * 0.05 * MS.CHI3_MEAN) < 0.005


def test_the_planners_descent_runs_on_z_and_leaves_every_weight_frozen_as_it_was():
    torch.manual_seed(3)
    m = _lite(motor_synergy=True, intention=True)
    B = 2
    cur = {mm: torch.randn(B, m.d_model) for mm in m.active}
    goal = {mm: v + 0.5 * torch.randn_like(v) for mm, v in cur.items()}
    ach = _pose(B, 5)
    ctx = torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model)
    motor = torch.zeros(B, len(NAMES), 9)
    motor[..., 6] = 1.0
    motor[..., 8] = 0.5
    dev = m.dev_limits(0)
    flags = [p.requires_grad for p in m.params]
    space = m._synergy_space(ach, motor)

    def execution(cmd, dev_limits, handle_state=None):
        return LemLite.planned_execution(cmd, dev_limits, handle_state)
    act, meta = m.intention_gen.generate_action(cur, goal, motor, achieved_state=ach, context_buffer=ctx,
                                                execution_transform=execution, dev_limits=dev,
                                                action_space=space)
    assert [p.requires_grad for p in m.params] == flags, "every freeze flag restored exactly"
    assert meta["synergy"]["z_norm"] > 0.0, "the descent moved z"
    assert torch.allclose(act[:, GI, 3:7], F_norm(space.gaze.detach()), atol=1e-6), \
        "the eyes stay the planner's own"
    # the order is the synergy's change of the CURRENT pose
    rot, pos = m.synergy.delta(space.z.detach(), ach)
    o = m.synergy.order_from_delta(rot, pos, ach, motor)
    assert torch.allclose(act[:, m.synergy.rot_idx, 3:7], o[:, m.synergy.rot_idx, 3:7], atol=1e-6)


def F_norm(q):
    return q / q.norm(dim=-1, keepdim=True)


def test_smooth_source_executed_blends_with_the_executed_order_and_passes_the_eyes_row():
    new = _order_cm(_pose(2, 6))
    prev = _order_cm(_pose(2, 7))
    out = LemLite.smooth_blend(new, prev, 0.35, keep_row=GI)
    assert torch.equal(out[:, GI], new[:, GI])
    ref = LemLite.smooth_blend(new, prev, 0.35)
    body = [i for i in range(len(NAMES)) if i != GI]
    assert torch.equal(out[:, body], ref[:, body])
    m = _lite(smooth_actions=0.35, smooth_source="executed", intention=True)
    m._smooth_exec_prev = prev.clone()
    m._prev_mfa = new.clone()
    d = m._capture_plan_state()
    assert torch.equal(d["smooth_prev"], prev) and d["smooth_keep_gaze"] == GI
    assert m.receipt["smooth_source"] == "executed"


def test_the_gaze_the_planner_imagines_is_the_gaze_the_servo_is_sent():
    m = _lite()
    g = torch.Generator().manual_seed(8)
    cmd = torch.zeros(4, len(NAMES), 9)
    q = torch.randn(4, len(NAMES), 4, generator=g)
    cmd[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    sent = LemLite.gaze_row_as_sent(cmd, GI)
    a = m._gaze_from_handle(cmd)
    b = m._gaze_from_handle(sent)
    assert np.allclose(a, b, atol=1e-5), "rewriting the row does not change the servo's command"
    ledger = LemLite.gaze_to_handle(cmd.numpy(), a)
    assert np.allclose(sent[:, GI, 3:7].numpy(), ledger[:, GI, 3:7], atol=1e-5), \
        "the planner's row is the row the executed ledger writes"
    body = [i for i in range(len(NAMES)) if i != GI]
    assert torch.equal(sent[:, body], cmd[:, body])
    # planned_execution under executed: the eyes' row is the sent one and not blended
    dev = m.dev_limits(0)
    ach = _pose(4)
    prev = _order_cm(_pose(4, 9))
    out = LemLite.planned_execution(cmd, dev, ach, previous=prev, smoothing=0.35, keep_gaze_idx=GI)
    assert np.allclose(m._gaze_from_handle(out), a, atol=1e-5)


def test_every_new_attribute_serialises_into_the_episode_companion():
    from mimo_bridge import continuation_state as CS
    torch.manual_seed(4)
    m = _lite(motor_synergy=True, synergy_babble=True, smooth_actions=0.35, smooth_source="executed")
    B = 2
    p0 = _pose(B)
    hold = _order_cm(p0)
    _run_lesson(m, [p0] * 7, [hold] * 7, [1, 2, 2, 2, 2, 2, 3])
    m._synergy_babble(hold.clone(), p0, 0.05, p0[..., 0:3] * 200.0, 40.0)
    m._smooth_exec_prev = hold.clone()
    state = CS.capture_learner(m, include_replay=False)
    for k in ("_syn_pending", "_syn_prev_ach", "_syn_bank", "_synergy_sigma", "_smooth_exec_prev"):
        assert k in state["data"], k
    assert "_syn_window" not in state["data"], "a resumed report starts a new window"
    other = _lite(motor_synergy=True, synergy_babble=True, smooth_actions=0.35, smooth_source="executed")
    CS.restore_learner(other, state)
    assert other._synergy_sigma == m._synergy_sigma
    assert torch.equal(other._syn_bank["x"], m._syn_bank["x"])


def test_the_synergy_is_clipped_as_its_own_group():
    m = _lite(motor_synergy=True)
    for p in m.params:
        p.grad = torch.zeros_like(p)
    core = next(m.forward_model.parameters())
    core.grad.fill_(1e-4)
    for p in m.synergy.parameters():
        p.grad.fill_(10.0)
    before = core.grad.clone()
    norm = m._clip_training_gradients()
    assert torch.equal(core.grad, before), "the core's step was shrunk by the synergy's gradient"
    assert abs(norm - float(before.norm())) < 1e-6


def test_the_synergy_rides_the_checkpoint_and_a_build_without_it_refuses(tmp_path):
    torch.manual_seed(6)
    m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                body_layout="trunk_v1", motor_synergy=True, checkpoint_dir=str(tmp_path))
    m.synergy.update_centre(_pose(3))
    m._synergy_sigma = 2.5
    path = m.save_checkpoint(7)
    other = LemLite(device="cpu", seed=1, imagination=False, no_vision=True, next_frame_loss=True,
                    body_layout="trunk_v1", motor_synergy=True)
    res = other.load_checkpoint(path)
    assert "motor_synergy" not in (res.get("reinitialized_components") or [])
    assert torch.equal(other.synergy.centre_mean, m.synergy.centre_mean)
    assert other.synergy.centre_weight == m.synergy.centre_weight and other._synergy_sigma == 2.5
    for a, b in zip(m.synergy.parameters(), other.synergy.parameters()):
        assert torch.equal(a, b)
    bare = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   body_layout="trunk_v1")
    with pytest.raises(ValueError, match="--motor-synergy on"):
        bare.load_checkpoint(path)
    wrong = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                    body_layout="trunk_v1", motor_synergy=True, synergy_dim=MS.SYNERGY_DIM + 1)
    with pytest.raises(ValueError, match="--synergy-dim"):
        wrong.load_checkpoint(path)


def test_a_change_of_the_change_scales_migrates_exactly(tmp_path, monkeypatch):
    torch.manual_seed(7)
    m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                body_layout="trunk_v1", motor_synergy=True, checkpoint_dir=str(tmp_path))
    with torch.no_grad():
        for p in m.synergy.parameters():
            p.add_(0.05 * torch.randn_like(p))
    pose = _pose(3)
    z = torch.randn(3, m.synergy.k)
    x = torch.randn(3, m.synergy.n_out)
    with torch.no_grad():
        r0, p0 = m.synergy.delta(z, pose)
        rot_l = x[:, :75].reshape(3, 25, 3) * m.synergy.rot_scale_rad
        pos_l = x[:, 75:].reshape(3, 14, 3) * m.synergy.pos_scale_cm
        z0 = m.synergy.encode(x, pose)
    path = m.save_checkpoint(3)
    monkeypatch.setattr(MS, "CHANGE_SCALE_DEG", MS.CHANGE_SCALE_DEG * 2.0)
    monkeypatch.setattr(MS, "CHANGE_SCALE_CM", MS.CHANGE_SCALE_CM * 0.5)
    other = LemLite(device="cpu", seed=1, imagination=False, no_vision=True, next_frame_loss=True,
                    body_layout="trunk_v1", motor_synergy=True)
    other.load_checkpoint(path)
    with torch.no_grad():
        r1, p1 = other.synergy.delta(z, pose)
        z1 = other.synergy.encode(other.synergy.to_units(rot_l, pos_l), pose)
    assert torch.allclose(r0, r1, atol=1e-5) and torch.allclose(p0, p1, atol=1e-4)
    assert torch.allclose(z0, z1, atol=1e-4), "the encoder reads the same change as the same z"
