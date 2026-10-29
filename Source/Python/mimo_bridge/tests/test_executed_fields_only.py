# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The order the ledger records holds only what the decoder reads (motor interface audit 2026-09-25,
finding F4; train_mimo --executed-fields-only on).

The defect (docs/gain_of_function_v4/agents/motor_interface_audit.md, "Dead command fields"): the
decoder reads a position only for the 14 rows in action_decoder.POSITION_HANDLES and reads no radius,
yet the ledger stored 36 position numbers (11 orientation-only body rows and the eyes' row) and 26
radii that reach no actuator, fed them to the motor encoder and let the planner optimise them.

Pinned here:
  * LemLite.executed_fields_only puts the achieved position on every other row and 0 in every
    radius, and changes nothing else; a gradient reaches only the fields it keeps;
  * the decoder's control is EXACTLY the same before and after it (the compiled crib model, random
    orders at randomly moved poses) - and the eyes' row's strength, which the audit did not list,
    is never read either;
  * planned_execution under the flag forms the order the live path forms, and the learner's recorded
    order (LemLite.step, real crib frames) is that order in every field but the eyes' blended
    quaternion and strength (the eyes are sent the unblended order's gaze, by design);
  * the gauges: the world's position check has a twin on the position rows, and the order-size gauge
    falls by exactly the replaced rows' share;
  * the flag defaults to off and reaches the learner.
  * THE ROWS KEPT ARE THE DECODER'S LIVE POSITION ROWS (the coordinator's requirement of 2026-09-25,
    after the limb branch's --position-rows, commit e1df17484): under the decoder's default 'all' they are
    POSITION_HANDLES; under 'no_head_pelvis' (audit F3) the head's and the pelvis's position asks are
    replaced too. Every test above runs under both settings, and one probes the decoder itself: a
    position the order keeps changes the control, a position it replaces never does.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from mimo_bridge.action_decoder import POSITION_HANDLES, POSITION_ROW_CHOICES, position_handles
from mimo_bridge.body_layout import names_for
from mimo_bridge.lem_model import COMMAND_POSITION_RANGE_CM, LemLite

NAMES = list(names_for("trunk_v1"))
GI = NAMES.index("gaze")
LIVE = [i for i, n in enumerate(NAMES) if n in POSITION_HANDLES]
DEAD = [i for i, n in enumerate(NAMES) if n not in POSITION_HANDLES]
PKG = Path(__file__).resolve().parents[1]


def _rows(position_rows):
    """(kept, replaced) row indices under a decoder setting."""
    live = position_handles(position_rows)
    return ([i for i, n in enumerate(NAMES) if n in live],
            [i for i, n in enumerate(NAMES) if n not in live])


def _order_and_pose(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    cmd = torch.randn(B, len(NAMES), 9, generator=g)
    cmd[..., 0:3] *= 30.0                                   # centimetres
    cmd[..., 3:7] = cmd[..., 3:7] / cmd[..., 3:7].norm(dim=-1, keepdim=True)
    cmd[..., 7:9] = torch.rand(B, len(NAMES), 2, generator=g)
    hs = torch.randn(B, len(NAMES), 10, generator=g) * 0.1  # achieved: positions / 200 cm
    return cmd, hs


def test_the_premise_14_position_rows_11_orientation_rows_and_the_eyes():
    assert len(LIVE) == 14 and len(DEAD) == 12 and GI in DEAD
    assert sorted(NAMES[i] for i in LIVE) == sorted(POSITION_HANDLES)
    # the decoder's live rows: its default is exactly POSITION_HANDLES; audit F3's setting drops the
    # head and the pelvis (action_decoder.position_handles, the limb branch's decoder change)
    assert POSITION_ROW_CHOICES == ("all", "no_head_pelvis")
    assert position_handles("all") == set(POSITION_HANDLES)
    assert position_handles("no_head_pelvis") == set(POSITION_HANDLES) - {"head", "lower_trunk"}
    kept, replaced = _rows("no_head_pelvis")
    assert len(kept) == 12 and len(replaced) == 14 and GI in replaced


@pytest.mark.parametrize("position_rows", POSITION_ROW_CHOICES)
def test_dead_rows_carry_the_achieved_position_and_every_radius_is_zero(position_rows):
    kept, replaced = _rows(position_rows)
    cmd, hs = _order_and_pose()
    cmd.requires_grad_(True)
    out = LemLite.executed_fields_only(cmd, hs, position_rows)
    assert out.shape == cmd.shape
    assert torch.equal(out[:, kept, 0:3], cmd[:, kept, 0:3])
    assert torch.allclose(out[:, replaced, 0:3], hs[:, replaced, 0:3] * COMMAND_POSITION_RANGE_CM)
    assert torch.equal(out[..., 7], torch.zeros_like(out[..., 7]))
    assert torch.equal(out[..., 3:7], cmd[..., 3:7]) and torch.equal(out[..., 8], cmd[..., 8])
    # out of place: a gradient reaches the kept fields and none reaches the replaced ones
    out.sum().backward()
    gr = cmd.grad
    assert torch.equal(gr[:, kept, 0:3], torch.ones_like(gr[:, kept, 0:3]))
    assert float(gr[:, replaced, 0:3].abs().sum()) == 0.0 and float(gr[..., 7].abs().sum()) == 0.0
    assert torch.equal(gr[..., 3:7], torch.ones_like(gr[..., 3:7]))
    # no pose: nothing to put in place of the dead positions, the order is returned as it is
    assert LemLite.executed_fields_only(cmd, None, position_rows) is cmd
    # the legacy layout (24 rows) is read by name too
    legacy = list(names_for("legacy"))
    c24, h24 = cmd.detach()[:, :24], hs[:, :24]
    o24 = LemLite.executed_fields_only(c24, h24, position_rows)
    live = position_handles(position_rows)
    for i, n in enumerate(legacy):
        want = c24[:, i, 0:3] if n in live else h24[:, i, 0:3] * COMMAND_POSITION_RANGE_CM
        assert torch.allclose(o24[:, i, 0:3], want), n
    # the default argument is the decoder's default
    if position_rows == "all":
        assert torch.equal(LemLite.executed_fields_only(cmd.detach(), hs), out.detach())
    with pytest.raises(ValueError):
        LemLite.executed_fields_only(cmd.detach(), hs, "no_head")


# ------------------------------------------------------------------ the decoder, on the crib model
def _crib_env(tmp_path, tag):
    from mimo_bridge.env import BridgeEnv
    return BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=False, tag=tag, age=24,
                     contact_profile="mechanical_v4", caregiver_seconds=0, kp=6.0,
                     body_layout="trunk_v1")


def _decoder(env, position_rows):
    """The crib world's decoder under a position-rows setting, every other setting as the world's
    own (the limb branch's BridgeEnv passes --position-rows to exactly this constructor)."""
    from mimo_bridge.action_decoder import ActionDecoder
    d0 = env.decoder
    return ActionDecoder(env.model, kp=d0.kp, kd=d0.kd, gaze_kp=d0.gaze_kp, gaze_kd=d0.gaze_kd,
                         lam=d0.lam, ctrl_bounds=(env.env.action_space.low, env.env.action_space.high),
                         body_layout="trunk_v1", position_rows=position_rows)


def _moved_poses(env, n, rng):
    """n randomly moved poses of the crib body: (MjData, achieved pose [1, H, 10] as the ledger reads
    it) for each."""
    import mujoco
    m = env.model
    hinge = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
    base = np.array(env.data.qpos)
    bones = env.proprio.body_ids
    for _ in range(n):
        d = mujoco.MjData(m)
        d.qpos[:] = base
        for j in hinge:
            a = int(m.jnt_qposadr[j])
            lo, hi = (float(x) for x in m.jnt_range[j]) if m.jnt_limited[j] else (-0.5, 0.5)
            d.qpos[a] = np.clip(base[a] + rng.normal(0.0, 0.2), lo, hi)
        d.qvel[:] = rng.normal(0.0, 0.1, m.nv)
        mujoco.mj_forward(m, d)
        ach = np.zeros((1, len(NAMES), 10), np.float32)
        ach[0, :, 0:3] = np.asarray(d.xpos[bones]) * 100.0 / COMMAND_POSITION_RANGE_CM
        yield d, ach


def _random_order(d, bones, rng):
    order = np.zeros((len(NAMES), 9))
    order[:, 0:3] = np.asarray(d.xpos[bones]) * 100.0 + rng.normal(0.0, 12.0, (len(NAMES), 3))
    q = rng.normal(size=(len(NAMES), 4))
    order[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    order[:, 7:9] = rng.uniform(0.0, 1.0, (len(NAMES), 2))
    return order


@pytest.mark.parametrize("position_rows", POSITION_ROW_CHOICES)
def test_the_decoders_control_is_exactly_unchanged(tmp_path, position_rows):
    """48 random orders at 16 randomly moved poses of the crib body, under each of the decoder's
    position-rows settings: the decoder's control with the replaced fields equals the control
    without them, bit for bit. The eyes' row's strength is checked too: the decoder never reads it
    (the audit listed positions and radii only)."""
    env = _crib_env(tmp_path, "_dead_fields_" + position_rows)
    try:
        env.reset()
        dec = _decoder(env, position_rows)
        _, replaced = _rows(position_rows)
        rng = np.random.default_rng(7)
        bones = env.proprio.body_ids
        worst, orders = 0.0, 0
        for d, ach in _moved_poses(env, 16, rng):
            for _ in range(3):
                order = _random_order(d, bones, rng)
                gaze = rng.uniform(-0.5, 0.5, 2)
                before = dec.decode(d, order, gaze).copy()
                kept = LemLite.executed_fields_only(torch.as_tensor(order[None]), torch.as_tensor(ach),
                                                    position_rows).numpy()[0]
                after = dec.decode(d, kept, gaze).copy()
                assert np.array_equal(before, after), "a replaced field reached an actuator"
                eyes = kept.copy()
                eyes[GI, 8] = 1.0 - eyes[GI, 8]
                assert np.array_equal(dec.decode(d, eyes, gaze), after), "the eyes' strength is read"
                worst = max(worst, float(np.abs(kept[replaced, 0:3]
                                                - np.asarray(d.xpos[bones])[replaced] * 100.0).max()))
                orders += 1
        assert orders == 48
        assert worst < 1e-3, f"a replaced row sits {worst} cm from its part"
    finally:
        env.close()


@pytest.mark.parametrize("position_rows", POSITION_ROW_CHOICES)
def test_every_position_the_order_keeps_is_one_the_decoder_solves(tmp_path, position_rows):
    """F4 and the decoder agree on the live rows (the coordinator's requirement of 2026-09-25, for F3's
    --position-rows): probed on the decoder itself, not on the list. At 6 moved poses, each body row's
    ordered position is moved 8 cm along each axis in turn: a row F4 keeps changes the decoder's control
    at least once, a row F4 replaces never does. Under 'no_head_pelvis' this is what keeps the head's
    and the pelvis's unperformed position asks out of the record."""
    env = _crib_env(tmp_path, "_live_rows_" + position_rows)
    try:
        env.reset()
        dec = _decoder(env, position_rows)
        kept, replaced = _rows(position_rows)
        rng = np.random.default_rng(11)
        bones = env.proprio.body_ids
        read = {i: False for i in range(len(NAMES)) if i != GI}
        for d, _ in _moved_poses(env, 6, rng):
            order = _random_order(d, bones, rng)
            order[:, 8] = 0.8                      # every row powered
            gaze = np.zeros(2)
            base = dec.decode(d, order, gaze).copy()
            for i in read:
                for axis in range(3):
                    moved = order.copy()
                    moved[i, axis] += 8.0
                    if not np.array_equal(dec.decode(d, moved, gaze), base):
                        read[i] = True
        assert sorted(i for i, r in read.items() if r) == sorted(kept), \
            "the rows whose position the decoder reads differ from the rows F4 keeps"
        assert not any(read[i] for i in replaced if i != GI)
    finally:
        env.close()


# ------------------------------------------------------------------ the planner and the live step
def test_planned_execution_under_the_flag_forms_the_live_paths_order():
    """The live path's pieces (LemLite.step: execution_transform, the body-relative reach clamp, the
    smoother, then executed_fields_only) against planned_execution(executed_fields=True)."""
    raw = torch.randn(2, len(NAMES), 9, generator=torch.Generator().manual_seed(3)) * 0.3
    raw[..., 6] += 1.0
    raw[..., 7:] = torch.rand(2, len(NAMES), 2)
    pose = torch.randn(2, len(NAMES), 10) * 0.05
    prev = raw.detach().clone() * 50.0
    dev = {'rom_limit': 1., 'strength_limit': .8}
    live = LemLite.execution_transform(None, raw, dev)
    cur_cm = pose[..., 0:3] * 200.0
    live[..., 0:3] = cur_cm + (raw[..., 0:3].clamp(-1.0, 1.0) * 200.0 - cur_cm).clamp(-40.0, 40.0)
    smoothed = LemLite.smooth_blend(live, prev, .35)
    live = LemLite.executed_fields_only(smoothed, pose)
    planned = LemLite.planned_execution(raw, dev, pose, previous=prev, smoothing=.35,
                                        keep_gaze_idx=GI, executed_fields=True)
    body = [i for i in range(len(NAMES)) if i != GI]
    assert torch.allclose(planned[:, body], live[:, body], atol=1e-4)
    # the eyes: the dead fields match; the quaternion is the unblended command as sent
    assert torch.allclose(planned[:, GI, 0:3], live[:, GI, 0:3], atol=1e-4)
    assert torch.equal(planned[:, GI, 7], live[:, GI, 7])
    unblended = LemLite.gaze_row_as_sent(LemLite.differentiable_execution_transform(raw, dev, pose), GI)
    assert torch.allclose(planned[:, GI, 3:7], unblended[:, GI, 3:7], atol=1e-6)
    # a planning decision's limits carry the setting (LemLite._capture_plan_state), so a caller that
    # passes only decision['dev'] forms the same order
    via_dev = LemLite.planned_execution(raw, dict(dev, executed_fields_only=True), pose, previous=prev,
                                        smoothing=.35, keep_gaze_idx=GI)
    assert torch.equal(via_dev, planned)
    # and with the flag off planned_execution is what it was
    old = LemLite.smooth_blend(LemLite.gaze_row_as_sent(
        LemLite.differentiable_execution_transform(raw, dev, pose), GI), prev, .35, keep_row=GI)
    assert torch.equal(LemLite.planned_execution(raw, dev, pose, previous=prev, smoothing=.35,
                                                 keep_gaze_idx=GI), old)
    # under audit F3's decoder setting the planner keeps the same live rows as the live step: passed
    # directly, or carried by the decision's limits (_capture_plan_state writes position_rows there)
    live_nhp = LemLite.executed_fields_only(smoothed, pose, "no_head_pelvis")
    planned_nhp = LemLite.planned_execution(raw, dev, pose, previous=prev, smoothing=.35, keep_gaze_idx=GI,
                                            executed_fields=True, position_rows="no_head_pelvis")
    kept, replaced = _rows("no_head_pelvis")
    assert torch.allclose(planned_nhp[:, kept], live_nhp[:, kept], atol=1e-4)
    assert torch.allclose(planned_nhp[:, replaced, 0:3], pose[:, replaced, 0:3] * 200.0, atol=1e-4)
    via_dev_nhp = LemLite.planned_execution(raw, dict(dev, executed_fields_only=True,
                                                      position_rows="no_head_pelvis"),
                                            pose, previous=prev, smoothing=.35, keep_gaze_idx=GI)
    assert torch.equal(via_dev_nhp, planned_nhp)
    head, pelvis = NAMES.index("head"), NAMES.index("lower_trunk")
    assert not torch.allclose(planned[:, [head, pelvis], 0:3], planned_nhp[:, [head, pelvis], 0:3])


def _learner(fields):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   body_layout="trunk_v1", smooth_actions=0.35, executed_fields_only=fields,
                   overrides={"training": {"steps_per_phase": 3000}})


def test_the_learners_recorded_order_is_the_planned_order_in_every_body_field(tmp_path):
    """Real crib frames, the learner's own step: every model step's order for the body
    (res.motor_for_action) equals planned_execution of the controller's output with the step's own
    pose and smoother state, in every field of every row but the eyes' blended quaternion and
    strength; its replaced rows are the achieved positions; the smoother remembers that form."""
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(scene="crib", vision=False, age=24, run_dir=str(tmp_path / "w"),
                          tag="_fields", seed=5, body_layout="trunk_v1")
    try:
        frames = src.reset()
        # the learner's position_rows is set after construction, as the limb branch's build_learner
        # sets it from --position-rows; F4 then keeps the decoder's live rows under that setting
        for fields, position_rows in (("on", "all"), ("on", "no_head_pelvis"), ("off", "all")):
            m = _learner(fields)
            m.position_rows = position_rows
            _, replaced = _rows(position_rows)
            dev = dict(m.dev_limits(10 ** 9))
            dev["exploration_noise"] = 0.0          # babble would make the order a random draw
            prev_exec, fr, checked = None, frames, 0
            for i in range(6):
                r = m.step(fr, dev, train=True, step_index=i, executed_action=prev_exec)
                ach = m.last_achieved.detach().float()
                mfa = torch.as_tensor(r.motor_for_action)
                dead_gap = (mfa[:, replaced, 0:3] - ach[:, replaced, 0:3] * COMMAND_POSITION_RANGE_CM).abs().max()
                if fields == "on":
                    planned = LemLite.planned_execution(
                        torch.as_tensor(r.motor_out), dev, ach, previous=m._last_smooth_prev,
                        smoothing=m.smooth_actions, keep_gaze_idx=GI,
                        eye_row_convention=m.eye_row_convention, executed_fields=True,
                        position_rows=position_rows)
                    # the planning decision carries the same rows
                    assert m._capture_plan_state()["dev"]["position_rows"] == position_rows
                    body = [k for k in range(len(NAMES)) if k != GI]
                    assert torch.allclose(planned[:, body], mfa[:, body], atol=1e-4), i
                    assert torch.allclose(planned[:, GI, [0, 1, 2, 7]], mfa[:, GI, [0, 1, 2, 7]], atol=1e-4)
                    assert float(dead_gap) < 1e-4, i
                    assert float(mfa[..., 7].abs().max()) == 0.0
                    assert torch.equal(m._prev_mfa.cpu(), mfa), "the smoother remembers the recorded form"
                    checked += 1
                else:
                    # the control: without the flag the dead rows hold free numbers, not the body
                    assert float(dead_gap) > 1.0, i
                    checked += 1
                prev_exec = LemLite.gaze_to_handle(r.motor_for_action, np.asarray(r.gaze_out),
                                                   convention=m.eye_row_convention)
                fr = src.step(r.motor_for_action, np.asarray(r.gaze_out), None, render=False)
            assert checked == 6
            assert m.receipt.get("executed_fields_only", False) == (fields == "on")
    finally:
        src.close()


@pytest.mark.parametrize("position_rows", POSITION_ROW_CHOICES)
def test_in_the_wrist_relative_hand_mode_the_worlds_canonical_goal_is_the_order_sent(tmp_path, position_rows):
    """Review of 2026-09-25, finding 8: H84 runs hand_target_mode wrist_relative_v1, where the trainer
    raises when the world's canonical order differs from the order it sent in any column of any row
    (train_mimo.same_hand_goal, called on every step). F4 runs after the hand assembly and changes no
    wrist or finger position. Six real crib steps with the world's own hand geometry, as the trainer
    wires them (configure_hand_geometry, the joint goal sent with the order), the trainer's comparison
    on every step, under both of the decoder's position-rows settings."""
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.train_mimo import same_hand_goal
    _, replaced = _rows(position_rows)
    src = SingleEnvSource(scene="crib", vision=False, age=24, run_dir=str(tmp_path / "w"),
                          tag="_wrist_" + position_rows, seed=5, body_layout="trunk_v1",
                          hand_target_mode="wrist_relative_v1")
    try:
        frames = src.reset()
        torch.set_num_threads(1)
        torch.manual_seed(0)
        m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                    body_layout="trunk_v1", smooth_actions=0.35, executed_fields_only="on",
                    hand_target_mode="wrist_relative_v1", overrides={"training": {"steps_per_phase": 3000}})
        m.position_rows = position_rows
        # the trainer's wiring (train_mimo.main, before its loop)
        hand_states = src.hand_state()
        geometry = hand_states[0]["geometry"]
        initial = np.stack([s["current"] if s["held"] is None else s["held"] for s in hand_states])
        for frame, hs in zip(frames, hand_states):
            frame.meta["hand_joint_angles"] = hs["current"]
            frame.meta["hand_joint_velocities"] = hs["velocity"]
        m.configure_hand_geometry(geometry, initial)
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.0
        prev_exec, checked = None, 0
        for i in range(6):
            r = m.step(frames, dev, train=True, step_index=i, executed_action=prev_exec)
            send = np.asarray(r.motor_for_action)
            assert r.hand_joint_goal is not None, "the wrist-relative learner sends a joint goal"
            ach = m.last_achieved.detach().float().cpu().numpy()
            assert float(np.abs(send[..., 7]).max()) == 0.0
            assert np.abs(send[:, replaced, 0:3] - ach[:, replaced, 0:3] * COMMAND_POSITION_RANGE_CM).max() < 1e-3
            frames = src.step(send, np.asarray(r.gaze_out), None, render=False,
                              hand_joint_goal=np.asarray(r.hand_joint_goal))
            canonical = np.stack([f.meta["canonical_motor_targets"] for f in frames])
            assert same_hand_goal(canonical, send), \
                "the trainer would raise: the world's canonical goal differs from the order sent (step %d)" % i
            prev_exec = LemLite.gaze_to_handle(send, np.asarray(r.gaze_out), convention=m.eye_row_convention)
            checked += 1
        assert checked == 6
    finally:
        src.close()


# ------------------------------------------------------------------ the gauges
def test_the_worlds_position_check_has_a_twin_on_the_position_rows(tmp_path):
    """Every row powered and turned, each position row ordered TOWARD the way it then moves and each
    orientation-only row AWAY from it: pos_cos mixes the two (14 at +1 and 11 at -1 over the 25
    scored rows), pos_cos_position_rows reads only the rows whose position the decoder reads."""
    from mimo_bridge.tools.command_following import _turn, rest_order
    env = _crib_env(tmp_path, "_pos_rows")
    try:
        env.reset()
        rng = np.random.default_rng(0)
        order = rest_order(env, len(NAMES))
        for i in range(len(NAMES)):
            order[i, 3:7] = _turn(order[i, 3:7], rng.normal(size=3), np.radians(20.0))
        order[:, 8] = 0.5
        ids = env.proprio.body_ids
        dp = np.array([0.01, 0.0, 0.0])                       # 1 cm along x: "the step's motion"
        xp_before = np.array(env.data.xpos[ids]) - dp
        xq_before = np.array(env.data.xquat[ids])
        order[:, 0:3] = (xp_before + dp * 3.0) * 100.0        # toward the motion
        order[DEAD, 0:3] = (xp_before[DEAD] - dp * 3.0) * 100.0   # away from it
        out = env._command_response(order, xp_before, xq_before, np.zeros((len(ids), 3)),
                                    record_window=False)
        assert out["n"] == 25
        assert out["pos_cos"] == pytest.approx((14 - 11) / 25.0, abs=1e-6)
        assert out["pos_cos_position_rows"] == pytest.approx(1.0, abs=1e-6)
        # the twin follows the world's decoder, not a list: with the head and the pelvis ordered AWAY
        # from the motion, the default decoder (which solves their position) scores them, audit F3's
        # decoder (which does not) leaves them out
        head, pelvis = NAMES.index("head"), NAMES.index("lower_trunk")
        order[[head, pelvis], 0:3] = (xp_before[[head, pelvis]] - dp * 3.0) * 100.0
        out = env._command_response(order, xp_before, xq_before, np.zeros((len(ids), 3)),
                                    record_window=False)
        assert out["pos_cos_position_rows"] == pytest.approx((12 - 2) / 14.0, abs=1e-6)
        env.decoder = _decoder(env, "no_head_pelvis")
        out = env._command_response(order, xp_before, xq_before, np.zeros((len(ids), 3)),
                                    record_window=False)
        assert out["pos_cos_position_rows"] == pytest.approx(1.0, abs=1e-6)
        assert out["pos_cos"] == pytest.approx((12 - 13) / 25.0, abs=1e-6), "pos_cos is unchanged"
    finally:
        env.close()


@pytest.mark.parametrize("position_rows", POSITION_ROW_CHOICES)
def test_the_order_size_gauge_falls_by_the_replaced_rows_share(position_rows):
    """train_mimo.cmd_exec_sample averages the ordered-minus-achieved distance over the 25 body rows:
    under the flag the rows whose position the decoder does not solve read 0, so step_cm = (sum over
    the kept rows) / 25 - the 14 position rows under the decoder's default, the 12 hand rows under
    audit F3's no_head_pelvis, where the ratio to the hands' own mean is exactly 12/25 = 0.48 - while
    step_cm_hands (rows 0-11, position rows under both settings) is unchanged."""
    from mimo_bridge.train_mimo import cmd_exec_sample
    kept, _ = _rows(position_rows)
    cmd, hs = _order_and_pose(B=4, seed=9)
    before = cmd_exec_sample(cmd.numpy(), hs.numpy(), NAMES)
    after = cmd_exec_sample(LemLite.executed_fields_only(cmd, hs, position_rows).numpy(), hs.numpy(), NAMES)
    d = np.linalg.norm(cmd.numpy()[..., 0:3] - hs.numpy()[..., 0:3] * COMMAND_POSITION_RANGE_CM, axis=-1)
    assert after["step_cm"] == pytest.approx(d[:, kept].sum(-1).mean() / 25.0, rel=1e-6)
    assert after["step_cm_hands"] == pytest.approx(before["step_cm_hands"], rel=1e-9)
    assert after["strength"] == before["strength"]
    if position_rows == "no_head_pelvis":
        assert after["step_cm"] / after["step_cm_hands"] == pytest.approx(12.0 / 25.0, rel=1e-5)


def test_the_flag_defaults_off_and_reaches_the_learner():
    from mimo_bridge.train_mimo import acting_configuration, parse_args
    a = parse_args(["--out", "unused"])
    assert a.executed_fields_only == "off"
    assert parse_args(["--out", "unused", "--executed-fields-only", "on"]).executed_fields_only == "on"
    assert acting_configuration(a)["executed_fields_only"] == "off"
    src = (PKG / "train_mimo.py").read_text(encoding="utf-8")
    assert "executed_fields_only=getattr(args, 'executed_fields_only', 'off')" in src
    assert '_pos_key = "pos_cos_position_rows" if model.executed_fields_only_on else "pos_cos"' in src
    cont = src[src.index("continuation_flags = ("):src.index("continuation_config = {")]
    assert "executed_fields_only" not in cont
    step = (PKG / "lem_model.py").read_text(encoding="utf-8")
    # the step forms it after the smoother and the hand assembly; the smoother's source line stays
    i_hand = step.index("mfa = self.hand_kinematics.assemble_world(mfa, hand_joints, position_scale=1.)")
    i_f4 = step.index("mfa = self.executed_fields_only(mfa, handle_state,")
    # ...with the learner's position_rows, the decoder's own setting (F3 and F4 agree on the live rows)
    assert 'getattr(self, "position_rows", "all")' in step[i_f4:i_f4 + 200]
    assert i_hand < i_f4
    assert "self.smooth_blend(mfa, self._last_smooth_prev" in step
    with pytest.raises(ValueError):
        LemLite(device="cpu", seed=0, imagination=False, no_vision=True, executed_fields_only="yes")
