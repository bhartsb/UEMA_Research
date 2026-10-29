# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The head and lower-trunk handles' POSITION targets: added by the tree's X7 decision, removed by the
operator's decision of 2026-09-25 9:09 AM PDT, which ships behind a launch flag.

History. X7 (2026-09-18) put the head's and the lower trunk's position rows into the decoder's solve,
so "lift the head" could be asked for as a place and not only as a rotation. The motor interface audit
(docs/gain_of_function_v4/agents/motor_interface_audit.md, F3) then measured them on real orders:
they performed 1.7% (head) and 10.5% (pelvis) of their ask and turned their part 3.9 and 11.5 degrees
off its orientation-only solution. On 2026-09-25 at 9:09 AM PDT the operator decided to REMOVE them
for now and keep their orientation targets, reversing X7 (docs/gain_of_function_v3/WORKLOG.md,
"Operator decisions").

The removal is action_decoder.position_handles('no_head_pelvis'), set by train_mimo --position-rows
no_head_pelvis. The default 'all' keeps X7's rows exactly until the coordinator deploys the flag, so
these tests pin both: under the default a position step alone still drives the neck and the hips (the
mechanical wiring X7 added); under the operator's setting it drives nothing, while the orientation rows
and the wrists' position rows work as before.
"""
import os

import numpy as np
import pytest

mimo_infant = pytest.importorskip("mimo_infant")

from mimo_bridge.action_decoder import (ActionDecoder, HEAD_PELVIS_POSITION_HANDLES, POSITION_HANDLES,
                                        POSITION_ROW_CHOICES, position_handles)
from mimo_bridge.env import BridgeEnv

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                       "Saved", "mimo_bridge", "scenes_test")


@pytest.fixture(scope="module")
def env():
    e = BridgeEnv(scene="crib", run_dir=RUN_DIR, vision=False, tag="_headpos", body_layout="trunk_v1")
    e.reset()
    # let the body settle: right after reset the joints still move (measured 0.70 rad/s),
    # and the decoder's damping term alone then gives a nonzero control
    null = np.zeros((len(e.decoder.handles), 9)); null[:, 6] = 1.0
    for _ in range(60):
        e.step(null, None, None, render=False)
    yield e
    e.close()


@pytest.fixture(scope="module")
def decided(env):
    """The decoder under the operator's decision, on the same body and pose."""
    return ActionDecoder(env.model, kp=env.decoder.kp, kd=env.decoder.kd, body_layout="trunk_v1",
                         ctrl_bounds=(env.decoder.ctrl_low, env.decoder.ctrl_high), position_rows="no_head_pelvis")


def _rest_order(env):
    n = len(env.decoder.handles)
    order = np.zeros((n, 9))
    for i, h in enumerate(env.decoder.handles):
        order[i, 0:3] = np.asarray(env.data.xpos[h.body_id]) * 100.0
        q = np.asarray(env.data.xquat[h.body_id])
        order[i, 3:7] = [q[1], q[2], q[3], q[0]]
    order[:, 7] = 0.5
    return order


def _reachable_offset_cm(env, handle, size_cm=4.0):
    """A position step in a direction the handle's OWN joints can produce (a point in the column space
    of the position Jacobian), pushing every joint away from its nearer range edge: a spine hinge
    cannot move the chest straight up when the body lies supine, and the decoder clips a push into a
    joint limit to zero, so a direction needing a blocked joint is not a test of the wiring."""
    dec = env.decoder
    h = dec.handles[dec.handle_names.index(handle)]
    p_body = np.asarray(env.data.xpos[h.body_id], dtype=np.float64)
    J = np.zeros((3, len(h.joint_ids)))
    for k, j in enumerate(h.joint_ids):
        J[:, k] = np.cross(env.data.xaxis[j], p_body - np.asarray(env.data.xanchor[j], dtype=np.float64))
    m = env.model
    v = np.ones(J.shape[1])
    for k, j in enumerate(h.joint_ids):
        if m.jnt_limited[j]:
            q = float(env.data.qpos[int(m.jnt_qposadr[j])])
            mid = 0.5 * (float(m.jnt_range[j, 0]) + float(m.jnt_range[j, 1]))
            v[k] = 1.0 if mid >= q else -1.0
    d = J @ v
    return J, size_cm * d / np.linalg.norm(d)


def test_the_default_keeps_x7s_rows_and_the_operators_setting_removes_them():
    assert POSITION_ROW_CHOICES == ("all", "no_head_pelvis")
    assert {"head", "lower_trunk"} <= POSITION_HANDLES                 # X7's rows, the default
    assert position_handles("all") == POSITION_HANDLES
    assert position_handles("no_head_pelvis") == POSITION_HANDLES - {"head", "lower_trunk"}
    assert HEAD_PELVIS_POSITION_HANDLES == {"head", "lower_trunk"}
    for rows in POSITION_ROW_CHOICES:
        assert {"wrist_L", "wrist_R"} <= position_handles(rows)       # unchanged either way
        assert not {"upperarm_L", "thigh_L", "foot_L", "chest"} & position_handles(rows)
    with pytest.raises(ValueError):
        position_handles("none")


def test_the_chest_has_no_lever_so_it_stays_orientation_only(env):
    """Measured: chest_lean and chest_rot are anchored at the chest body's own origin, so
    axis x (p_body - anchor) is zero for both and no position error can be closed."""
    dec = env.decoder
    h = dec.handles[dec.handle_names.index("chest")]
    p_body = np.asarray(env.data.xpos[h.body_id], dtype=np.float64)
    for j in h.joint_ids:
        lever = np.cross(env.data.xaxis[j], p_body - np.asarray(env.data.xanchor[j], dtype=np.float64))
        assert float(np.linalg.norm(lever)) < 1e-6


@pytest.mark.parametrize("handle", ["head", "lower_trunk"])
def test_under_the_default_a_position_step_alone_drives_the_handles_joints(env, handle):
    """X7's wiring, kept by the default until the deploy."""
    dec = env.decoder
    assert dec.position_rows == "all"
    idx = dec.handle_names.index(handle)
    h = dec.handles[idx]
    assert h.joint_ids, f"{handle} drives no joints on this body"
    assert dec.use_position[idx]
    J, offset_cm = _reachable_offset_cm(env, handle)
    # orientation error zero, position target = current: only the damping term remains
    still = _rest_order(env)
    still[idx, 8] = 1.0
    ctrl0 = dec.decode(env.data, still, None)
    damping = dec.kd * float(np.abs(env.data.qvel[[h.dof_adrs[k] for k in range(len(h.joint_ids))]]).max())
    assert float(np.abs(ctrl0[h.act_ids]).max()) <= damping + 1e-9, "a zero error produced more than the damping term"
    # orientation error zero, position target displaced: the joints must push
    moved = still.copy()
    moved[idx, 0:3] = moved[idx, 0:3] + offset_cm
    ctrl1 = dec.decode(env.data, moved, None)
    assert float(np.abs(ctrl1[h.act_ids] - ctrl0[h.act_ids]).max()) > 1e-3, "a position-only error produced no control"
    # the solve's own linear prediction of the body's displacement points along the ask
    # recover dq from the control (ctrl = strength * (kp*dq - kd*qd) with qd ~ 0 at rest)
    dq = np.array([ctrl1[a] - ctrl0[a] for a in h.act_ids]) / dec.kp   # the damping part cancels
    predicted = J @ dq
    e = offset_cm / 100.0
    cos = float(np.dot(predicted, e) / (np.linalg.norm(predicted) * np.linalg.norm(e) + 1e-12))
    assert cos > 0.7, f"predicted displacement points away from the ask (cos {cos:.2f})"


@pytest.mark.parametrize("handle", ["head", "lower_trunk"])
def test_under_the_operators_setting_a_position_step_drives_nothing(env, decided, handle):
    """The decision of 2026-09-25: the same position step as above changes no control at all, so the
    head and pelvis can no longer be pulled off their orientation targets by a position ask."""
    idx = decided.handle_names.index(handle)
    assert not decided.use_position[idx]
    _, offset_cm = _reachable_offset_cm(env, handle)
    still = _rest_order(env)
    still[idx, 8] = 1.0
    moved = still.copy()
    moved[idx, 0:3] = moved[idx, 0:3] + offset_cm
    far = still.copy()
    far[idx, 0:3] = far[idx, 0:3] + np.array([12.0, -9.0, 7.0])      # beyond the reach clip too
    ctrl0 = decided.decode(env.data, still, None)
    np.testing.assert_array_equal(decided.decode(env.data, moved, None), ctrl0)
    np.testing.assert_array_equal(decided.decode(env.data, far, None), ctrl0)


@pytest.mark.parametrize("handle", ["head", "lower_trunk"])
def test_under_the_operators_setting_the_orientation_rows_still_act(env, decided, handle):
    """The orientation targets stay: a turn asked of the head (pelvis) still drives its joints, wherever
    its position target lies, and exactly as today's decoder drives an orientation-only row (today's
    decoder solves orientation alone for a row whose position is not a finite number)."""
    idx = decided.handle_names.index(handle)
    h = decided.handles[idx]
    still = _rest_order(env)
    still[idx, 8] = 1.0
    turned = still.copy()
    c, s = np.cos(0.1), np.sin(0.1)                                      # 0.2 rad about the world z axis
    qx, qy, qz, qw = turned[idx, 3:7]
    turned[idx, 3:7] = [c * qx - s * qy, c * qy + s * qx, c * qz + s * qw, c * qw - s * qz]
    c0, c1 = decided.decode(env.data, still, None), decided.decode(env.data, turned, None)
    assert float(np.abs(c1[h.act_ids] - c0[h.act_ids]).max()) > 1e-3, "an orientation step produced no control"
    _, offset_cm = _reachable_offset_cm(env, handle)
    turned_and_moved = turned.copy()
    turned_and_moved[idx, 0:3] = turned_and_moved[idx, 0:3] + offset_cm
    np.testing.assert_array_equal(decided.decode(env.data, turned_and_moved, None), c1)
    orientation_only = turned.copy()
    orientation_only[idx, 0:3] = np.nan
    np.testing.assert_array_equal(env.decoder.decode(env.data, orientation_only, None), c1)


def test_hand_position_following_is_unchanged_in_kind(env, decided):
    """The wrist keeps its position rows under both settings; a 5 cm step still moves the arm group."""
    for dec in (env.decoder, decided):
        idx = dec.handle_names.index("wrist_R")
        still = _rest_order(env)
        still[idx, 8] = 1.0
        moved = still.copy()
        moved[idx, 0:3] = moved[idx, 0:3] + np.array([0.0, 0.0, 5.0])
        c0 = dec.decode(env.data, still, None)
        c1 = dec.decode(env.data, moved, None)
        acts = dec.handles[idx].act_ids
        assert float(np.abs(c1[acts] - c0[acts]).max()) > 1e-6


def test_the_world_takes_the_setting_and_reports_it_only_when_it_is_not_the_default(env):
    """BridgeEnv hands position_rows to its decoder; a world on the default reports nothing new."""
    assert env.position_rows == "all" and env.decoder.position_rows == "all"
    null = np.zeros((len(env.decoder.handles), 9)); null[:, 6] = 1.0
    assert "position_rows" not in env.step(null, None, None, render=False).info
    other = BridgeEnv(scene="crib", run_dir=RUN_DIR, vision=False, tag="_headpos_decided", body_layout="trunk_v1",
                      position_rows="no_head_pelvis")
    try:
        other.reset()
        names = other.decoder.handle_names
        assert not other.decoder.use_position[names.index("head")]
        assert not other.decoder.use_position[names.index("lower_trunk")]
        assert other.decoder.use_position[names.index("wrist_L")]
        assert other.step(null, None, None, render=False).info["position_rows"] == "no_head_pelvis"
    finally:
        other.close()
    with pytest.raises(ValueError):
        BridgeEnv(scene="crib", run_dir=RUN_DIR, vision=False, tag="_headpos_bad", body_layout="trunk_v1",
                  position_rows="none")
