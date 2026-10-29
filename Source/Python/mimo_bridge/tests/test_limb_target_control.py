# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The motor interface audit's calf, chest and arm fixes (limb_target_control.py; F2a and F2b of
docs/gain_of_function_v4/agents/motor_interface_audit.md, revision d7a348f03).

F2a: a calf (chest) order is stored as the closest pose of its own joints on its parent's ordered pose
(the thigh's row; the upper body the pelvis's row implies), and the target the decoder reads is
re-placed on the parent where it is before every 10 ms frame. F2b: the arm's four targets are replaced
by the world poses of the joint solution the decoder's own arm solve settles on, inside the joint
ranges with the chest where it is. All are off by default and change nothing then.
"""
import glob
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge import limb_target_control as L

# the audit's saved world model and one body's observer records (segment 40, H84), read in place
SAVED_WORLD = Path(r"C:\Users\bhart\Documents\UEMA_devA_private\Saved\gof_observer_20260924"
                   r"\motor_interface_audit\observer_copy\H84_fresh_drivers_20260918_seg40\world")


def _random_xyzw(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def _angle_deg(a_wxyz, b_wxyz):
    a = np.asarray(a_wxyz) / np.linalg.norm(a_wxyz)
    b = np.asarray(b_wxyz) / np.linalg.norm(b_wxyz)
    r = L.qmul(L.qconj(a), b)
    return math.degrees(2.0 * math.atan2(np.linalg.norm(r[1:]), abs(float(r[0]))))


def _turned(q_xyzw, axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return L.xyzw_aligned(L.qmul(L.q_axis_angle(axis, angle), L.unit_wxyz_from_xyzw(q_xyzw)), q_xyzw)


def _off_axis_deg(parent_xyzw, child_xyzw, body_quat_wxyz, axis):
    """The exact angle between the child's pose relative to the parent and the closest turn about
    the hinge axis (any angle): 2 atan2(|v_perp|, sqrt(w^2 + (v.axis)^2)), well conditioned at 0."""
    qb = np.asarray(body_quat_wxyz, dtype=np.float64) / np.linalg.norm(body_quat_wxyz)
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    j = L.qmul(L.qconj(L.qmul(L.unit_wxyz_from_xyzw(parent_xyzw), qb)), L.unit_wxyz_from_xyzw(child_xyzw))
    s = float(j[1:] @ a)
    perp = np.linalg.norm(j[1:] - s * a)
    return math.degrees(2.0 * math.atan2(perp, math.hypot(float(j[0]), s)))


# ------------------------------------------------------------------ one hinge, closed form
def test_the_closest_knee_angle_beats_a_dense_search_inside_the_range():
    rng = np.random.default_rng(20260925)
    body = L.unit_wxyz_from_xyzw([0.05, -0.1, 0.08, 0.99])
    axis = np.array([0.1, -1.0, 0.2]) / np.linalg.norm([0.1, -1.0, 0.2])
    ref, lo, hi = 0.04, -2.53, 0.07
    grid = np.linspace(lo, hi, 20001)[::40]
    for case in range(300):
        parent = _random_xyzw(rng)
        if case % 3 == 0:            # an order that is a knee pose, perturbed a little
            t = rng.uniform(lo, hi)
            child = L.xyzw_aligned(L.qmul(L.qmul(L.unit_wxyz_from_xyzw(parent), body),
                                          L.q_axis_angle(axis, t - ref)), [0, 0, 0, 1])
            child = child + rng.normal(scale=0.02, size=4)
        else:
            child = _random_xyzw(rng)
        q, residual = L.one_axis_goal(parent, child, body, axis, ref, lo, hi)
        assert lo - 1e-12 <= q <= hi + 1e-12
        placed = L.one_axis_child_wxyz(L.unit_wxyz_from_xyzw(parent), body, axis, q, ref)
        # the residual it reports is the angle it leaves
        assert abs(_angle_deg(placed, L.unit_wxyz_from_xyzw(child)) - math.degrees(residual)) < 1e-9
        dense = min(_angle_deg(L.one_axis_child_wxyz(L.unit_wxyz_from_xyzw(parent), body, axis, t, ref),
                               L.unit_wxyz_from_xyzw(child)) for t in grid)
        assert math.degrees(residual) <= dense + 1e-9


def test_a_zero_quaternion_means_the_identity_as_the_decoder_reads_it():
    body, axis = np.array([1.0, 0, 0, 0]), np.array([0, -1.0, 0])
    q_zero, r_zero = L.one_axis_goal([0, 0, 0, 0], [0, 0, 0, 1e-14], body, axis, 0.0, -2.5, 0.07)
    q_id, r_id = L.one_axis_goal([0, 0, 0, 1], [0, 0, 0, 1], body, axis, 0.0, -2.5, 0.07)
    assert (q_zero, r_zero) == (q_id, r_id) == (0.0, 0.0)


# ------------------------------------------------------------------ a hand-built pair of legs
mujoco = pytest.importorskip('mujoco')

_LEG = '''<body name="{side}_thigh" pos=".005 {y} -.007" quat=".96 .1 -.2 .15">
  <joint name="{side}_hip1" axis="0 1 0" range="-2.3 .35"/>
  <joint name="{side}_hip2" axis="1 0 0" pos=".002 0 .001" range="-.9 .3"/>
  <joint name="{side}_hip3" axis="0 0 1" range="-.55 .7"/>
  <geom type="capsule" size=".03 .06" pos="0 0 -.06"/>
  <body name="{side}_calf" pos="0 0 -.135" quat=".99 .05 .08 -.1">
    <joint name="{side}_knee" axis=".1 -1 .2" pos="0 .004 .002" range="-2.53 .07" ref=".04"/>
    <geom type="capsule" size=".025 .05" pos="0 0 -.05"/>
  </body></body>'''


@pytest.fixture
def legs():
    """Two legs on a free hip, with a tilted knee axis, a turned calf frame and a knee reference
    angle: a frame or order mistake in the knee's placement shows up in every check below."""
    xml = ('<mujoco><compiler angle="radian"/><worldbody><body name="hip" pos="0 0 .5">'
           '<freejoint name="root"/><geom type="sphere" size=".04"/>'
           + _LEG.format(side='left', y=.051) + _LEG.format(side='right', y=-.051)
           + '</body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    names = ['thigh_L', 'calf_L', 'thigh_R', 'calf_R']
    bodies = ['left_thigh', 'left_calf', 'right_thigh', 'right_calf']
    joints = [[model.joint(s + '_hip' + k).id for k in '123'] if b.endswith('thigh') else
              [model.joint(s + '_knee').id] for b, s in zip(bodies, ['left', 'left', 'right', 'right'])]
    decoder = SimpleNamespace(handle_names=names, arm_groups=[], use_position=[False] * 4,
                              handles=[SimpleNamespace(body_id=model.body(b).id, joint_ids=j, reach_m=0.01)
                                       for b, j in zip(bodies, joints)])
    return model, data, decoder


def _pose(model, data, rng):
    data.qpos[:] = model.qpos0
    data.qpos[3:7] = rng.normal(size=4)
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    for j in range(1, model.njnt):
        lo, hi = model.jnt_range[j]
        data.qpos[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
    mujoco.mj_kinematics(model, data)


def _hold_order(data, decoder):
    order = np.zeros((len(decoder.handle_names), 9))
    order[:, 7], order[:, 8] = 0.5, 0.8
    for i, h in enumerate(decoder.handles):
        order[i, :3] = data.xpos[h.body_id] * 100.0
        order[i, 3:7] = np.asarray(data.xquat[h.body_id])[[1, 2, 3, 0]]
    return order


def _leg_order(data, decoder, rng, calf_turn=0.6):
    """Every row at its part's pose, the calves turned off the knee's axis at random, and the thighs'
    ORDER turned away from where the thighs are."""
    order = _hold_order(data, decoder)
    for i in (1, 3):
        order[i, 3:7] = _turned(order[i, 3:7], rng.normal(size=3), calf_turn)
    for i in (0, 2):
        order[i, 3:7] = _turned(order[i, 3:7], [0.3, 0.9, -0.2], 0.4)
    return order


def test_the_knee_placement_is_mujocos_own(legs):
    model, data, decoder = legs
    rng = np.random.default_rng(3)
    control = L.LimbTargetControl(model, decoder, parent_relative='calf')
    for _ in range(40):
        _pose(model, data, rng)
        for calf in control.calves:
            q = data.qpos[calf.qpos_address]
            ours = L.one_axis_child_wxyz(np.array(data.xquat[calf.thigh_body]), calf.body_quat,
                                         calf.axis, q, calf.ref)
            assert _angle_deg(ours, data.xquat[calf.calf_body]) < 1e-9


def test_the_stored_calf_row_is_the_closest_knee_pose_on_the_ordered_thigh(legs):
    """The audit's F2a test, first half: the stored calf row relative to the stored thigh row lies at
    most 1e-6 degrees from the knee's poses, for calf targets that were off the knee's axis."""
    model, data, decoder = legs
    rng = np.random.default_rng(11)
    control = L.LimbTargetControl(model, decoder, parent_relative='calf')
    before = []
    for _ in range(60):
        _pose(model, data, rng)
        order = _leg_order(data, decoder, rng)
        motor, cache, ledger = control.execute(data, order, order)
        for calf, angle in zip(control.calves, cache['knee_goals']):
            before.append(_off_axis_deg(order[calf.thigh_row, 3:7], order[calf.calf_row, 3:7],
                                        calf.body_quat, calf.axis))
            assert _off_axis_deg(ledger[calf.thigh_row, 3:7], ledger[calf.calf_row, 3:7],
                                 calf.body_quat, calf.axis) <= 1e-6
            assert calf.lo <= angle <= calf.hi
            # the stored row is the goal on the ORDERED thigh, not on the thigh where it is
            placed = L.one_axis_child_wxyz(L.unit_wxyz_from_xyzw(order[calf.thigh_row, 3:7]),
                                           calf.body_quat, calf.axis, angle, calf.ref)
            assert _angle_deg(placed, L.unit_wxyz_from_xyzw(ledger[calf.calf_row, 3:7])) < 1e-9
        # nothing else in the order changed: every other row and the calf's other columns
        calf_rows = [c.calf_row for c in control.calves]
        others = [i for i in range(len(order)) if i not in calf_rows]
        np.testing.assert_array_equal(ledger[others], order[others])
        np.testing.assert_array_equal(ledger[calf_rows][:, [0, 1, 2, 7, 8]], order[calf_rows][:, [0, 1, 2, 7, 8]])
    assert np.median(before) > 10.0, "the orders were not off the knee's axis: the check proves nothing"


def test_the_regenerated_calf_target_turns_with_the_thigh(legs, monkeypatch):
    """The audit's F2a test, second half: turning the thigh between two frames turns the
    regenerated calf target with it, constant relative to the actual thigh to 1e-6 degrees; the
    order is projected once and the ledger does not move."""
    model, data, decoder = legs
    rng = np.random.default_rng(5)
    control = L.LimbTargetControl(model, decoder, parent_relative='calf')
    _pose(model, data, rng)
    order = _leg_order(data, decoder, rng)
    first, cache, ledger1 = control.execute(data, order, order)
    rel1 = {c.name: L.qmul(L.qconj(np.array(data.xquat[c.thigh_body])), L.unit_wxyz_from_xyzw(first[c.calf_row, 3:7]))
            for c in control.calves}
    monkeypatch.setattr(L, 'one_axis_goal', lambda *a, **k: pytest.fail('a held order was projected again'))
    for j in ('left_hip1', 'left_hip2', 'right_hip1', 'right_hip3'):
        data.qpos[model.jnt_qposadr[model.joint(j).id]] += 0.3
    data.qpos[3:7] = L.qmul(L.q_axis_angle([0, 0, 1], 0.5), data.qpos[3:7])
    mujoco.mj_kinematics(model, data)
    second, cache2, ledger2 = control.execute(data, order, order, cache)
    turned = 0
    for c in control.calves:
        rel2 = L.qmul(L.qconj(np.array(data.xquat[c.thigh_body])), L.unit_wxyz_from_xyzw(second[c.calf_row, 3:7]))
        assert _angle_deg(rel1[c.name], rel2) <= 1e-6
        turned += _angle_deg(L.unit_wxyz_from_xyzw(first[c.calf_row, 3:7]),
                             L.unit_wxyz_from_xyzw(second[c.calf_row, 3:7])) > 5.0
    assert turned == 2, "the thighs did not turn: the check proves nothing"
    np.testing.assert_array_equal(ledger1, ledger2)
    assert cache2 is cache


def test_the_decoder_sees_a_pure_knee_turn_toward_the_goal(legs):
    """Placed on the thigh as the decoder sees it, the remaining turn is about the knee's world axis
    and as large as the knee's own remaining angle: the knee's one column can close all of it."""
    from mimo_bridge.action_decoder import quat_xyzw_to_mat, rotmat_log
    model, data, decoder = legs
    rng = np.random.default_rng(8)
    control = L.LimbTargetControl(model, decoder, parent_relative='calf')
    for _ in range(20):
        _pose(model, data, rng)
        order = _leg_order(data, decoder, rng)
        motor, cache, _ = control.execute(data, order, order)
        for c, goal in zip(control.calves, cache['knee_goals']):
            w = rotmat_log(quat_xyzw_to_mat(motor[c.calf_row, 3:7]) @ np.asarray(data.xmat[c.calf_body]).reshape(3, 3).T)
            remaining = goal - data.qpos[c.qpos_address]
            assert np.linalg.norm(w - remaining * np.asarray(data.xaxis[c.joint])) < 1e-9


def test_the_planners_calf_rows_are_the_rows_the_world_records(legs):
    """Review of 2026-09-25, finding 5: with the calves projected, the planner imagines the calf rows
    the body receives. calf_rows_torch (the planner's twin) gives the world's ledger rows: the knee goal
    on the ORDERED thigh, the same sign, every other number untouched; in float64 to 1e-7 degrees, in
    the planner's float32 to 1e-3 degrees."""
    import torch
    model, data, decoder = legs
    rng = np.random.default_rng(31)
    control = L.LimbTargetControl(model, decoder, parent_relative='calf')
    geometry = L.check_calf_geometry(control.calf_geometry(), decoder.handle_names)
    assert L.LimbTargetControl(model, decoder).calf_geometry() == []
    worst64, worst32, off = 0.0, 0.0, []
    for _ in range(60):
        _pose(model, data, rng)
        order = _leg_order(data, decoder, rng, calf_turn=1.2)
        _, _, ledger = control.execute(data, order, order)
        for dtype in (torch.float64, torch.float32):
            ours = L.calf_rows_torch(torch.as_tensor(order[None], dtype=dtype), geometry)[0].double().numpy()
            for c in control.calves:
                err = _angle_deg(L.unit_wxyz_from_xyzw(ours[c.calf_row, 3:7]),
                                 L.unit_wxyz_from_xyzw(ledger[c.calf_row, 3:7]))
                if dtype is torch.float64:
                    worst64 = max(worst64, err)
                    assert float(ours[c.calf_row, 3:7] @ ledger[c.calf_row, 3:7]) > 0, 'the sign the world records'
                else:
                    worst32 = max(worst32, err)
            others = [i for i in range(len(order)) if i not in [c.calf_row for c in control.calves]]
            if dtype is torch.float64:
                np.testing.assert_array_equal(ours[others], order[others])
                np.testing.assert_array_equal(ours[[1, 3]][:, [0, 1, 2, 7, 8]], order[[1, 3]][:, [0, 1, 2, 7, 8]])
        off += [_off_axis_deg(order[c.thigh_row, 3:7], order[c.calf_row, 3:7], c.body_quat, c.axis)
                for c in control.calves]
    assert worst64 < 1e-7 and worst32 < 1e-3, (worst64, worst32)
    assert np.median(off) > 10.0, 'the orders were not off the knee axis: the check proves nothing'
    # the planner's gradient reaches both rows through it: a calf row a little off a knee pose inside
    # the range (at a range end the knee angle is the end whatever the calf row, as in the world)
    c = control.calves[0]
    base = _hold_order(data, decoder)
    base[c.thigh_row, 3:7] = _random_xyzw(rng)
    knee = L.one_axis_child_wxyz(L.unit_wxyz_from_xyzw(base[c.thigh_row, 3:7]), c.body_quat, c.axis,
                                 0.5 * (c.lo + c.hi), c.ref)
    base[c.calf_row, 3:7] = _turned(L.xyzw_aligned(knee, [0, 0, 0, 1]), [0.2, 0.1, 0.9], 0.05)
    order = torch.as_tensor(base[None], dtype=torch.float64).requires_grad_(True)
    out = L.calf_rows_torch(order, geometry)
    (out[0, 1, 3:7] * torch.tensor([0.3, -0.2, 0.5, 0.1], dtype=torch.float64)).sum().backward()
    g = order.grad[0].numpy()
    assert np.isfinite(g).all() and np.abs(g[0, 3:7]).sum() > 0 and np.abs(g[1, 3:7]).sum() > 0
    assert np.abs(g[2:]).sum() == 0, 'the left calf row depends on the left leg only'
    # the geometry is checked against the learner's handle table
    with pytest.raises(ValueError):
        L.check_calf_geometry(geometry[:1], decoder.handle_names)
    with pytest.raises(ValueError):
        L.check_calf_geometry(geometry, ['calf_L', 'thigh_L', 'thigh_R', 'calf_R'])


def test_the_planner_imagines_the_projected_calf_only_when_the_launch_projects_it():
    """planned_execution applies the twin after everything else when the decision carries the geometry
    (_capture_plan_state puts it there only when the learner was configured with it); without it the
    planner's executed order is today's exactly. The trainer reads the geometry from the worlds'
    frames, and refuses frames that do not carry one, or carry different ones."""
    import torch
    from mimo_bridge.body_layout import names_for
    from mimo_bridge.lem_model import LemLite
    names = list(names_for('trunk_v1'))
    rng = np.random.default_rng(12)
    geometry = [dict(name=calf, thigh_row=names.index(thigh), calf_row=names.index(calf),
                     body_quat_wxyz=list(L._unit([0.99, 0.05, 0.08, -0.1])),
                     axis=list(L._unit([0.1, -1.0, 0.2])), ref=0.04, lo=-2.53, hi=0.07)
                for thigh, calf in L.CALF_HANDLES]
    geometry = L.check_calf_geometry(geometry, names)
    raw = torch.zeros(2, len(names), 9)
    for b in range(2):
        for i in range(len(names)):
            raw[b, i, 3:7] = torch.as_tensor(_random_xyzw(rng))
    raw[..., 8] = 0.6
    ach = torch.zeros(2, len(names), 10)
    dev = {'rom_limit': 1., 'strength_limit': 1.}
    plain = LemLite.planned_execution(raw, dev, ach, smoothing=0.35, previous=raw.flip(0))
    again = LemLite.planned_execution(raw, dict(dev), ach, smoothing=0.35, previous=raw.flip(0))
    assert torch.equal(plain, again)
    for kw, limits in (({}, dict(dev, calf_projection=geometry)), ({'calf_projection': geometry}, dev)):
        projected = LemLite.planned_execution(raw, limits, ach, smoothing=0.35, previous=raw.flip(0), **kw)
        assert torch.allclose(projected, L.calf_rows_torch(plain, geometry))
        calf_rows = [g['calf_row'] for g in geometry]
        keep = [i for i in range(len(names)) if i not in calf_rows]
        assert torch.equal(projected[:, keep], plain[:, keep])
        assert not torch.allclose(projected[:, calf_rows, 3:7], plain[:, calf_rows, 3:7])
    # the decision carries it only when configured
    from types import SimpleNamespace
    p = torch.nn.Parameter(torch.ones(1))
    stub = SimpleNamespace(last_context=torch.ones(1, 3, 4), context_buffer=torch.zeros(1, 3, 4),
                           _prev_mfa=None, last_achieved=torch.zeros(1, 26, 10), _last_source_mask={},
                           _last_dev={'rom_limit': 1.}, smooth_actions=.35, _hand_previous_joints=None,
                           _hand_current_angles=None, _hand_current_velocities=None, params=[p],
                           device=torch.device('cpu'), GAZE_IDX=names.index('gaze'))
    stub._plan_parameter_versions = lambda: LemLite._plan_parameter_versions(stub)
    stub._plan_settings = lambda: None
    d = LemLite._capture_plan_state(stub)
    assert 'calf_projection' not in d and 'calf_projection' not in d['dev']
    stub.calf_projection = geometry
    d = LemLite._capture_plan_state(stub)
    assert d['calf_projection'] == geometry and d['dev']['calf_projection'] == geometry
    assert 'calf_projection' not in stub._last_dev, 'the live limits are untouched'
    # the trainer's read of the worlds' frames
    frame = {'limb_target_geometry': {'calf': geometry}}
    assert L.planner_calf_geometry([frame, dict(frame)], 'calf') == geometry
    assert L.planner_calf_geometry([{}, {}], 'off') is None
    with pytest.raises(RuntimeError, match='no calf geometry'):
        L.planner_calf_geometry([frame, {}], 'calf_chest')
    other = {'limb_target_geometry': {'calf': [dict(g, ref=0.05) for g in geometry]}}
    with pytest.raises(RuntimeError, match='different'):
        L.planner_calf_geometry([frame, other], 'calf')
    # a public name: an episode companion never carries it back
    assert "calf_projection = None" in inspect.getsource(LemLite.__init__)


def test_off_changes_nothing_and_unknown_modes_are_refused(legs):
    model, data, decoder = legs
    control = L.LimbTargetControl(model, decoder)
    assert not control.enabled and control.contract is None and not control.calves and not control.arms
    assert control.chest is None
    with pytest.raises(ValueError):
        L.LimbTargetControl(model, decoder, parent_relative='chest')
    with pytest.raises(ValueError):
        L.LimbTargetControl(model, decoder, reach_projection='arms_head')


# ------------------------------------------------------------------ a hand-built trunk
_TRUNK = '''<mujoco><compiler angle="radian"/><worldbody><body name="hip" pos="0 0 .5">
  <freejoint name="root"/><geom type="sphere" size=".05"/>
  {legs}
  <body name="lower_body" pos=".002 0 .076" quat=".995 .03 -.05 .02">
    <joint name="hip_lean1" axis="1 0 0" pos="0 0 -.05" range="-.244 .244"/>
    <joint name="hip_rot1" axis="0 0 1" pos="0 0 -.05" range="-.314 .314"/>
    <joint name="hip_bend1" axis="0 1 0" pos="0 0 -.05" range="-.297 .532"/>
    <geom type="capsule" size=".05 .03"/>
    <body name="upper_body" pos="-.002 0 .091" quat=".998 -.02 .04 .03">
      <joint name="hip_lean2" axis="1 0 0" pos="0 0 -.05" range="-.471 .471"/>
      <joint name="hip_rot2" axis="0 0 1" pos="0 0 -.05" range="-.314 .314"/>
      <joint name="hip_bend2" axis="0 1 0" pos="0 0 -.05" range="-.297 .532"/>
      <geom type="capsule" size=".05 .03"/>
      <body name="chest" pos="0 0 .09" quat=".997 .04 .02 -.05">
        <joint name="chest_lean" axis="1 .1 0" range="-.157 .157"/>
        <joint name="chest_rot" axis="0 .1 1" range="-.314 .314" ref=".02"/>
        <geom type="capsule" size=".04 .04"/>
      </body></body></body></body></worldbody>
  <tendon>
    <fixed name="hip_bend" limited="true" range="-.01 .01"><joint joint="hip_bend1" coef="1"/><joint joint="hip_bend2" coef="-1"/></fixed>
    <fixed name="hip_rotation" limited="true" range="-.01 .01"><joint joint="hip_rot1" coef="1"/><joint joint="hip_rot2" coef="-1"/></fixed>
    <fixed name="hip_lean" limited="true" range="-.01 .01"><joint joint="hip_lean1" coef="1"/><joint joint="hip_lean2" coef="-.5"/></fixed>
  </tendon></mujoco>'''


@pytest.fixture
def trunk():
    """MIMo's trunk layout by hand, every frame turned a little and the chest's axes tilted: the pelvis
    on a free root, the upper body on three hinges the tendons tie to the pelvis's, the chest's two;
    the two legs beside the pelvis, as in MIMo."""
    xml = _TRUNK.format(legs=_LEG.format(side='left', y=.051) + _LEG.format(side='right', y=-.051))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joints = lambda *n: [model.joint(x).id for x in n]
    parts = [('thigh_L', 'left_thigh', joints('left_hip1', 'left_hip2', 'left_hip3')),
             ('calf_L', 'left_calf', joints('left_knee')),
             ('thigh_R', 'right_thigh', joints('right_hip1', 'right_hip2', 'right_hip3')),
             ('calf_R', 'right_calf', joints('right_knee')),
             ('lower_trunk', 'lower_body', joints('hip_lean1', 'hip_rot1', 'hip_bend1')),
             ('chest', 'chest', joints('chest_lean', 'chest_rot'))]
    decoder = SimpleNamespace(handle_names=[p[0] for p in parts], arm_groups=[],
                              use_position=[False] * 4 + [True, False],
                              handles=[SimpleNamespace(body_id=model.body(b).id, joint_ids=j, reach_m=.05)
                                       for _, b, j in parts])
    return model, data, decoder


def _trunk_pose(model, data, rng, coupled=True):
    """A pose with the upper body's hinges where MIMo's tendons hold them (or anywhere)."""
    data.qpos[:] = model.qpos0
    data.qpos[3:7] = _random_xyzw(rng)
    for j in range(1, model.njnt):
        lo, hi = model.jnt_range[j]
        data.qpos[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
    if coupled:
        a = lambda n: model.jnt_qposadr[model.joint(n).id]
        data.qpos[a('hip_bend2')] = data.qpos[a('hip_bend1')]
        data.qpos[a('hip_rot2')] = data.qpos[a('hip_rot1')]
        data.qpos[a('hip_lean2')] = np.clip(2 * data.qpos[a('hip_lean1')], -.471, .471)
    mujoco.mj_kinematics(model, data)


def test_the_upper_body_the_pelvis_order_implies_is_mujocos_own(trunk):
    """A pelvis order the pelvis can take is decomposed into its hinge angles, the upper body's are
    placed as the tendons tie them, and the implied upper body is where MuJoCo puts it."""
    model, data, decoder = trunk
    chest = L._Chest(model, decoder)
    assert [c[1:] for c in chest.coupling] == [(1.0, -0.5, 0.0), (1.0, -1.0, 0.0), (1.0, -1.0, 0.0)]
    rng = np.random.default_rng(21)
    adr = lambda joints: [model.jnt_qposadr[j] for j in joints]
    for _ in range(25):
        _trunk_pose(model, data, rng)
        order = _hold_order(data, decoder)
        goal = chest.goal(np.array(data.xquat[chest.root_body]), order)
        np.testing.assert_allclose(goal['pelvis'], data.qpos[adr(decoder.handles[chest.pelvis_row].joint_ids)], atol=1e-7)
        upper = [model.joint(n).id for n in ('hip_lean2', 'hip_rot2', 'hip_bend2')]
        np.testing.assert_allclose(goal['upper'], data.qpos[adr(upper)], atol=2e-7)
        assert _angle_deg(goal['upper_wxyz'], data.xquat[chest.upper_body]) < 1e-5
        # the chest's own pose, asked for, is left where it is
        np.testing.assert_allclose(goal['chest'], data.qpos[adr(decoder.handles[chest.chest_row].joint_ids)], atol=1e-6)


def test_the_stored_chest_row_is_the_closest_chest_pose_on_the_implied_upper_body(trunk):
    """The audit's F2a test for the chest, first half: the stored chest row, relative to the upper
    body the pelvis's ordered row implies, lies at most 1e-6 degrees from the chest's two-joint
    poses, and no chest pose inside the ranges lies closer to the ordered chest row."""
    model, data, decoder = trunk
    control = L.LimbTargetControl(model, decoder, parent_relative='calf_chest')
    chest = control.chest
    rng = np.random.default_rng(22)
    grid = np.stack(np.meshgrid(np.linspace(chest.chest.lo[0], chest.chest.hi[0], 121),
                                np.linspace(chest.chest.lo[1], chest.chest.hi[1], 241), indexing='ij'),
                    axis=-1).reshape(-1, 2)
    before = []
    for case in range(30):
        _trunk_pose(model, data, rng)
        order = _hold_order(data, decoder)
        order[chest.pelvis_row, 3:7] = _turned(order[chest.pelvis_row, 3:7], rng.normal(size=3),
                                               0.15 if case % 2 else 0.6)
        order[chest.chest_row, 3:7] = _turned(order[chest.chest_row, 3:7], rng.normal(size=3), 0.7)
        motor, cache, ledger = control.execute(data, order, order)
        goal = cache['chest_goal']
        stored = L.unit_wxyz_from_xyzw(ledger[chest.chest_row, 3:7])
        _, residual = chest.chest.closest(goal['upper_wxyz'], stored, chest.GRID_CHEST)
        assert math.degrees(residual) <= 1e-6
        assert np.all(goal['chest'] >= chest.chest.lo) and np.all(goal['chest'] <= chest.chest.hi)
        asked = L.unit_wxyz_from_xyzw(order[chest.chest_row, 3:7])
        closest = np.max(np.abs(chest.chest.place(goal['upper_wxyz'], grid) @ asked))
        assert math.degrees(goal['residual']) <= math.degrees(2 * math.acos(min(1.0, float(closest)))) + 1e-6
        assert abs(_angle_deg(stored, asked) - math.degrees(goal['residual'])) < 1e-6
        before.append(math.degrees(goal['residual']))
        # the pelvis's hinge angles are inside their ranges, and no pelvis pose on the grid is closer
        assert np.all(goal['pelvis'] >= chest.pelvis.lo) and np.all(goal['pelvis'] <= chest.pelvis.hi)
        pelvis_grid = np.stack(np.meshgrid(*[np.linspace(a, b, 21) for a, b in zip(chest.pelvis.lo, chest.pelvis.hi)],
                                           indexing='ij'), axis=-1).reshape(-1, 3)
        root = np.array(data.xquat[chest.root_body])
        asked_pelvis = L.unit_wxyz_from_xyzw(order[chest.pelvis_row, 3:7])
        best_pelvis = math.degrees(2 * math.acos(min(1.0, float(np.max(np.abs(chest.pelvis.place(root, pelvis_grid) @ asked_pelvis))))))
        assert math.degrees(goal['pelvis_off_reach']) <= best_pelvis + 1e-6
        # nothing but the chest's orientation changed in the ledger
        others = [i for i in range(len(order)) if i not in (chest.chest_row, 1, 3)]
        np.testing.assert_array_equal(ledger[others], order[others])
        np.testing.assert_array_equal(ledger[chest.chest_row, [0, 1, 2, 7, 8]], order[chest.chest_row, [0, 1, 2, 7, 8]])
    assert np.median(before) > 5.0, "the chest orders were performable already: the check proves nothing"


def test_the_regenerated_chest_target_turns_with_the_upper_body(trunk, monkeypatch):
    """The audit's F2a test for the chest, second half: turning the upper body between two frames
    turns the regenerated chest target with it, constant relative to the actual upper body to 1e-6
    degrees, and with the chest's hinges at the goal it is exactly the chest's pose."""
    model, data, decoder = trunk
    rng = np.random.default_rng(23)
    control = L.LimbTargetControl(model, decoder, parent_relative='calf_chest')
    row, upper = control.chest.chest_row, control.chest.upper_body
    _trunk_pose(model, data, rng)
    order = _hold_order(data, decoder)
    order[row, 3:7] = _turned(order[row, 3:7], [1.0, 0.4, 0.2], 0.5)
    first, cache, ledger1 = control.execute(data, order, order)
    rel1 = L.qmul(L.qconj(np.array(data.xquat[upper])), L.unit_wxyz_from_xyzw(first[row, 3:7]))
    monkeypatch.setattr(L._Chest, 'goal', lambda *a, **k: pytest.fail('a held order was projected again'))
    for n in ('hip_lean1', 'hip_bend1', 'hip_bend2', 'hip_lean2'):
        data.qpos[model.jnt_qposadr[model.joint(n).id]] += 0.15
    data.qpos[3:7] = L.qmul(L.q_axis_angle([0, 1, 0], 0.4), data.qpos[3:7])
    mujoco.mj_kinematics(model, data)
    second, cache2, ledger2 = control.execute(data, order, order, cache)
    rel2 = L.qmul(L.qconj(np.array(data.xquat[upper])), L.unit_wxyz_from_xyzw(second[row, 3:7]))
    assert _angle_deg(rel1, rel2) <= 1e-6
    assert _angle_deg(L.unit_wxyz_from_xyzw(first[row, 3:7]), L.unit_wxyz_from_xyzw(second[row, 3:7])) > 5.0
    np.testing.assert_array_equal(ledger1, ledger2)
    assert cache2 is cache
    data.qpos[[model.jnt_qposadr[j] for j in decoder.handles[row].joint_ids]] = cache['chest_goal']['chest']
    mujoco.mj_kinematics(model, data)
    assert _angle_deg(L.unit_wxyz_from_xyzw(second[row, 3:7]), data.xquat[model.body('chest').id]) < 1e-9


# ------------------------------------------------------------------ the trainer's side
def _meta(contract, ledger):
    m = {}
    if contract is not None:
        m.update(limb_target_contract=contract, ledger_motor_targets=ledger)
    return m


def test_the_trainer_refuses_worlds_that_disagree_with_the_launch():
    names = ['thigh_L', 'calf_L', 'thigh_R', 'calf_R']
    sent = np.random.default_rng(1).normal(size=(2, 4, 9))
    ledger = sent.copy()
    ledger[:, 1, 3:7] += 0.1
    # every fix off, worlds off: today's ledger (None)
    assert L.ledger_from_frames([{}, {}], sent, names) is None
    # a world projecting orders the launch did not ask for
    with pytest.raises(RuntimeError, match='did not ask for'):
        L.ledger_from_frames([_meta(L.CALF_CONTRACT, ledger[0]), {}], sent, names)
    # the launch asks, the worlds do not (a resume that kept the saved settings)
    with pytest.raises(RuntimeError, match='resume-env-settings launch'):
        L.ledger_from_frames([{}, {}], sent, names, parent_relative='calf')
    ok = L.ledger_from_frames([_meta(L.CALF_CONTRACT, ledger[0]), _meta(L.CALF_CONTRACT, ledger[1])],
                              sent, names, parent_relative='calf')
    np.testing.assert_array_equal(ok, ledger)
    # a world that changed a row the calf fix does not own
    bad = ledger.copy()
    bad[1, 0, 3] += 1e-9
    with pytest.raises(RuntimeError, match='thigh_L'):
        L.ledger_from_frames([_meta(L.CALF_CONTRACT, bad[0]), _meta(L.CALF_CONTRACT, bad[1])],
                             sent, names, parent_relative='calf')
    bad = ledger.copy()
    bad[0, 1, 8] = 0.0                                   # the calf's strength is never the fix's
    with pytest.raises(RuntimeError, match='calf_L'):
        L.ledger_from_frames([_meta(L.CALF_CONTRACT, bad[0]), _meta(L.CALF_CONTRACT, bad[1])],
                             sent, names, parent_relative='calf')


def test_the_replay_contract_is_todays_when_the_limb_fixes_are_off():
    hand = {'hand_target_contract': 'wrist_relative_joint_v1'}
    world = {'hand_target_contract': 'world_v0'}
    # the formula lem_model used before the fix: int(all(hand contract == wrist_relative_joint_v1))
    for metas in ([hand, hand], [hand, world], [world, world], [{}, {}]):
        old = int(all(m.get('hand_target_contract') == 'wrist_relative_joint_v1' for m in metas))
        assert L.moment_contract(metas) == old
    assert L.replay_contract('world', 'off') is None
    assert L.replay_contract('wrist_relative_v1', 'off') == 1
    both = {'limb_target_contract': L.contract_of('calf_chest', 'arms')}
    assert L.moment_contract([dict(hand, **both)] * 2) == 7
    assert L.moment_contract([dict(world, limb_target_contract=L.CALF_CONTRACT)] * 2) == 2
    assert L.moment_contract([dict(hand, **both), hand]) == 1
    assert L.moment_contract([dict(hand, limb_target_contract=L.ARM_CONTRACT)]) == 1
    assert L.replay_contract('wrist_relative_v1', 'calf') == 3
    assert L.replay_contract('wrist_relative_v1', 'calf_chest') == 7
    assert L.replay_contract('world', 'calf') == 2
    assert L.action_contract_parts(dict(both)) == [L.CALF_CONTRACT, L.CHEST_CONTRACT]
    assert L.action_contract_parts({'limb_target_contract': L.contract_of('off', 'arms')}) == []
    assert L.action_contract_parts({}) == [] and L.action_contract_parts(None) == []


def test_the_head_and_pelvis_position_rows_setting_has_its_own_contract():
    """--position-rows no_head_pelvis (the operator's decision of 2026-09-25 9:09 AM PDT): the head's and
    the pelvis's position numbers are then read by nothing, so the orders recorded under it carry their
    own transition part and replay bit, and the trainer refuses worlds that solve other rows than the
    launch asked for, in either direction."""
    from mimo_bridge.transition import Pending, known_action_contract
    rows = {'position_rows': 'no_head_pelvis'}
    hand = {'hand_target_contract': 'wrist_relative_joint_v1'}
    # a world on the default reports nothing: today's contracts exactly
    assert L.moment_contract([hand, hand]) == 1
    assert L.replay_contract('wrist_relative_v1', 'off', 'all') == 1
    assert L.replay_contract('world', 'off', 'all') is None
    # the setting's part (after the fingers' and the limbs') and bit
    assert L.action_contract_parts(rows) == [L.POSITION_ROWS_CONTRACT]
    both = {'limb_target_contract': L.contract_of('calf_chest', 'arms')}
    assert L.action_contract_parts(dict(rows, **both)) == [L.CALF_CONTRACT, L.CHEST_CONTRACT,
                                                           L.POSITION_ROWS_CONTRACT]
    assert L.moment_contract([dict(hand, **rows)] * 2) == 9
    assert L.moment_contract([dict(rows)] * 2) == 8
    assert L.moment_contract([dict(hand, **rows), hand]) == 1           # worlds split between settings
    assert L.moment_contract([dict(hand, **rows, **both)] * 2) == 15
    assert L.replay_contract('wrist_relative_v1', 'off', 'no_head_pelvis') == 9
    assert L.replay_contract('world', 'off', 'no_head_pelvis') == 8
    assert L.replay_contract('wrist_relative_v1', 'calf_chest', 'no_head_pelvis') == 15
    for good in (L.POSITION_ROWS_CONTRACT, 'wrist_relative_joint_v1+' + L.POSITION_ROWS_CONTRACT,
                 '+'.join(['wrist_relative_joint_v1', L.CALF_CONTRACT, L.CHEST_CONTRACT, L.POSITION_ROWS_CONTRACT])):
        assert known_action_contract(good), good
    for bad in (L.POSITION_ROWS_CONTRACT + '+wrist_relative_joint_v1', L.POSITION_ROWS_CONTRACT + '+' + L.CALF_CONTRACT,
                'world_v0+' + L.POSITION_ROWS_CONTRACT, L.POSITION_ROWS_CONTRACT + '+' + L.POSITION_ROWS_CONTRACT):
        assert not known_action_contract(bad), bad
    # the rows the physics received do not change meaning, so no per-frame rows are required
    p = Pending(env_id=0, episode_id=0, frame_t=1, state_t={}, achieved_t=None, context_t=None)
    record = p.complete(executed_action=np.zeros((2, 9)), state_t1={}, frame_t1=2, duration_s=.05,
                        action_contract=L.POSITION_ROWS_CONTRACT)
    assert record.action_contract == L.POSITION_ROWS_CONTRACT
    # the trainer's check
    L.check_position_rows([{}, {}])
    L.check_position_rows([rows, rows], 'no_head_pelvis')
    with pytest.raises(RuntimeError, match='resume-env-settings launch'):
        L.check_position_rows([{}, {}], 'no_head_pelvis')                  # a resume that kept the saved worlds
    with pytest.raises(RuntimeError):
        L.check_position_rows([rows, rows], 'all')                          # worlds the launch did not ask for
    with pytest.raises(RuntimeError):
        L.check_position_rows([rows, {}], 'no_head_pelvis')


def test_replay_never_mixes_position_rows_settings_in_one_window():
    import torch
    from mimo_bridge.replay import ReplayMemory
    memory = ReplayMemory(['touch'], 2, n_handles=2, size_per_env=40, n_envs=1)
    for i in range(30):
        memory.add({'touch': torch.zeros(1, 1, 2)}, torch.zeros(1, 2, 10), {},
                   executed=torch.full((1, 2, 9), float(i)), action_contract=1 if i < 18 else 9)
    sample = memory.sample_windows(64, 3, 'cpu',
                                   action_contract=L.replay_contract('wrist_relative_v1', 'off', 'no_head_pelvis'))
    assert sample is not None and float(sample['executed'].min()) >= 18.0
    # today's launch studies only the moments before the setting (a window's last moment is its target;
    # its own outgoing order is not part of the window's contract)
    before = memory.sample_windows(64, 3, 'cpu', action_contract=L.replay_contract('wrist_relative_v1', 'off'))
    assert before is not None and float(before['executed'][:, :-1].max()) <= 17.0


@pytest.mark.parametrize('hand, parent_relative, rows, contract', [
    ('world', 'off', 'all', None),                              # every fix off: replay takes any window
    ('wrist_relative_v1', 'off', 'all', 1),                     # the live run's launch: the finger fix alone
    ('wrist_relative_v1', 'calf', 'all', 3),
    ('wrist_relative_v1', 'calf_chest', 'no_head_pelvis', 15),  # fingers 1 | calf 2 | chest 4 | rows 8
    ('world', 'off', 'no_head_pelvis', 8),
])
def test_replay_studies_the_contract_build_learner_sets(monkeypatch, tmp_path, hand, parent_relative, rows,
                                                         contract):
    """build_learner sets parent_relative_targets and position_rows on the learner after building it, and
    LemLite.replay_update reads them by name (getattr, with today's values as defaults) to choose which
    stored stretches replay re-studies. A misspelt name on either side reads the default with no error:
    under a calf deploy, replay would ask for stretches recorded before the deploy (contract 1) and, once
    every stored stretch postdates it, find none (prod_tests_review_2026-09-25.md, finding 3). Builds no
    model: the real replay_update runs on a stand-in learner and a stand-in store that returns no window."""
    from mimo_bridge import train_mimo as TM
    from mimo_bridge.lem_model import LemLite

    class Learner:
        def __init__(self, **kw):
            self.hand_target_mode = kw['hand_target_mode']   # LemLite keeps it as given (lem_model.py)

    monkeypatch.setattr(TM, 'LemLite', Learner)
    a = TM.parse_args(['--out', str(tmp_path), '--hand-target-mode', hand,
                       '--parent-relative-targets', parent_relative, '--position-rows', rows])
    built = TM.build_learner(a, None)
    asked = []
    built.replay = SimpleNamespace(sample_windows=lambda batch, ctx, device, **kw: asked.append(kw))
    built.ctx_len, built.device = 4, 'cpu'
    assert LemLite.replay_update(built, batch=1) is None       # no window: returns before any model use
    assert asked == [{} if contract is None else {'action_contract': contract}]
    assert contract == L.replay_contract(hand, parent_relative, rows)


def test_replay_never_mixes_calf_contracts_in_one_window():
    import torch
    from mimo_bridge.replay import ReplayMemory
    memory = ReplayMemory(['touch'], 2, n_handles=2, size_per_env=40, n_envs=1)
    for i in range(30):
        memory.add({'touch': torch.zeros(1, 1, 2)}, torch.zeros(1, 2, 10), {},
                   executed=torch.full((1, 2, 9), float(i)), action_contract=1 if i < 18 else 3)
    sample = memory.sample_windows(64, 3, 'cpu', action_contract=L.replay_contract('wrist_relative_v1', 'calf'))
    assert sample is not None and float(sample['executed'].min()) >= 18.0


def test_the_transition_carries_the_parent_relative_contracts_with_their_substeps():
    from mimo_bridge.transition import Pending, known_action_contract
    for good in ('world_v0', 'wrist_relative_joint_v1', L.CALF_CONTRACT, L.CALF_CONTRACT + '+' + L.CHEST_CONTRACT,
                 'wrist_relative_joint_v1+' + L.CALF_CONTRACT + '+' + L.CHEST_CONTRACT):
        assert known_action_contract(good), good
    for bad in (L.ARM_CONTRACT, L.CHEST_CONTRACT, 'world_v0+' + L.CALF_CONTRACT,
                L.CALF_CONTRACT + '+wrist_relative_joint_v1', L.CALF_CONTRACT + '+' + L.CALF_CONTRACT):
        assert not known_action_contract(bad), bad
    p = Pending(env_id=0, episode_id=0, frame_t=1, state_t={}, achieved_t=None, context_t=None)
    args = dict(executed_action=np.zeros((2, 9)), state_t1={}, frame_t1=2, duration_s=.05)
    record = p.complete(**args, action_contract=L.CALF_CONTRACT, motor_substeps=np.zeros((5, 2, 9)))
    assert record.action_contract == L.CALF_CONTRACT
    with pytest.raises(ValueError, match='substeps'):
        p.complete(**args, action_contract=L.CALF_CONTRACT + '+' + L.CHEST_CONTRACT)
    with pytest.raises(ValueError, match='joint goal'):
        p.complete(**args, action_contract='wrist_relative_joint_v1+' + L.CALF_CONTRACT,
                   motor_substeps=np.zeros((5, 2, 9)))
    with pytest.raises(ValueError, match='Unknown'):
        p.complete(**args, action_contract=L.ARM_CONTRACT, motor_substeps=np.zeros((5, 2, 9)))


def test_the_flags_default_off_everywhere_a_resume_reads_them():
    from mimo_bridge import continuation_state as CS
    from mimo_bridge import train_mimo as TM
    from mimo_bridge.env import BridgeEnv
    a = TM.parse_args(['--out', 'x'])
    assert a.parent_relative_targets == 'off' and a.reach_projection == 'off'
    kw = TM.env_kwargs_for(a, None, 'x')
    assert 'parent_relative_targets' not in kw and 'reach_projection' not in kw, 'left out at the defaults'
    kw = TM.env_kwargs_for(TM.parse_args(['--out', 'x', '--parent-relative-targets', 'calf']), None, 'x')
    assert kw['parent_relative_targets'] == 'calf' and 'reach_projection' not in kw
    defaults = inspect.signature(BridgeEnv.__init__).parameters
    assert defaults['parent_relative_targets'].default == 'off'
    assert defaults['reach_projection'].default == 'off'
    b = TM.parse_args(['--out', 'x', '--parent-relative-targets', 'calf_chest', '--reach-projection', 'arms'])
    assert (b.parent_relative_targets, b.reach_projection) == ('calf_chest', 'arms')
    # not continuation flags: a resume of the old command is never refused for them
    src = inspect.getsource(TM.main)
    flags = src.split('continuation_flags = (')[1].split(')')[0]
    assert 'parent_relative' not in flags and 'reach_projection' not in flags
    # a deploy with --resume-env-settings launch writes them into the saved worlds; bad values refused
    assert {'parent_relative_targets', 'reach_projection'} <= set(CS.ENV_KWARGS_MIGRATABLE)
    state = dict(version=CS.VERSION, worlds=[dict(env_kwargs=dict(scene='crib'), world={}, payload={})],
                 config={}, trainer=dict(step=1))
    up, changes = CS.migrate_env_kwargs(state, dict(parent_relative_targets='calf_chest', reach_projection='arms'),
                                        env_settings='launch')
    assert up['worlds'][0]['env_kwargs'] == dict(scene='crib', parent_relative_targets='calf_chest',
                                                  reach_projection='arms')
    up, changes = CS.migrate_env_kwargs(state, dict(parent_relative_targets='calf'))
    assert changes == [] and up is state, 'the default keeps the saved settings'
    for bad in (dict(parent_relative_targets='chest'), dict(reach_projection='arms_head')):
        with pytest.raises(ValueError):
            CS.migrate_env_kwargs(state, bad, env_settings='launch')


def test_the_position_rows_flag_defaults_to_todays_rows_everywhere_a_resume_reads_it():
    """--position-rows (the operator's F3 decision) defaults to 'all', today's rows, in the parser, the
    launch's world settings, the world, the decoder and the learner's replay choice; a deploy with
    --resume-env-settings launch writes it into the saved worlds, and a plain resume keeps them."""
    from mimo_bridge import continuation_state as CS
    from mimo_bridge import train_mimo as TM
    from mimo_bridge.action_decoder import ActionDecoder
    from mimo_bridge.env import BridgeEnv
    a = TM.parse_args(['--out', 'x'])
    assert a.position_rows == 'all'
    assert 'position_rows' not in TM.env_kwargs_for(a, None, 'x'), 'left out at the default'
    assert TM.env_kwargs_for(TM.parse_args(['--out', 'x', '--position-rows', 'no_head_pelvis']), None,
                             'x')['position_rows'] == 'no_head_pelvis'
    assert inspect.signature(BridgeEnv.__init__).parameters['position_rows'].default == 'all'
    assert inspect.signature(ActionDecoder.__init__).parameters['position_rows'].default == 'all'
    assert TM.parse_args(['--out', 'x', '--position-rows', 'no_head_pelvis']).position_rows == 'no_head_pelvis'
    with pytest.raises(SystemExit):
        TM.parse_args(['--out', 'x', '--position-rows', 'none'])
    src = inspect.getsource(TM.main)
    flags = src.split('continuation_flags = (')[1].split(')')[0]
    assert 'position_rows' not in flags, 'a resume of the old command is never refused for it'
    assert "learner.position_rows = getattr(args, 'position_rows', 'all')" in inspect.getsource(TM.build_learner)
    assert 'position_rows' in CS.ENV_KWARGS_MIGRATABLE
    state = dict(version=CS.VERSION, worlds=[dict(env_kwargs=dict(scene='crib'), world={}, payload={})],
                 config={}, trainer=dict(step=1))
    up, changes = CS.migrate_env_kwargs(state, dict(position_rows='no_head_pelvis'), env_settings='launch')
    assert up['worlds'][0]['env_kwargs'] == dict(scene='crib', position_rows='no_head_pelvis')
    assert changes == [dict(world=0, name='position_rows', old=None, new='no_head_pelvis')]
    up, changes = CS.migrate_env_kwargs(state, dict(position_rows='no_head_pelvis'))
    assert changes == [] and up is state, 'the default keeps the saved settings'
    with pytest.raises(ValueError):
        CS.migrate_env_kwargs(state, dict(position_rows='none'), env_settings='launch')


# a saved world as H84's companions hold them before these fixes: no parent_relative_targets,
# reach_projection or position_rows (the settings named in the order H84's saved worlds carry them)
H84_LIKE_WORLD = dict(scene='crib', run_dir='x/scenes', vision=True, render_every=1, touch_scale_mult=1.0,
                      seed=7, tag='_b0', kp=6.0, joint_damping_scale=3.0, hinge_armature=0.02,
                      finger_armature=0.0004, hand_target_mode='wrist_relative_v1', toy_hand_off='off',
                      body_layout='trunk_v1')


def test_a_resume_at_the_defaults_leaves_the_saved_worlds_settings_byte_identical():
    """Rollback safety (review of 2026-09-25, finding 2): a resume of a command without the three flags,
    under --resume-env-settings launch as H84 runs, writes none of them into the saved worlds, so an
    episode saved after the merge with the flags off still resumes on code that predates them. A world
    that carries a setting keeps receiving the launch's value, the default included (the rollback)."""
    import json
    from mimo_bridge import continuation_state as CS
    from mimo_bridge import train_mimo as TM
    argv = ['--out', 'x', '--resume-env-settings', 'launch', '--hand-target-mode', 'wrist_relative_v1',
            '--toy-hand-off', 'off', '--body-layout', 'trunk_v1']
    args = TM.parse_args(argv)
    launch = TM.env_kwargs_for(args, None, 'x')
    assert not set(TM.LIMB_ENV_DEFAULTS) & set(launch)
    worlds = [dict(env_kwargs=dict(H84_LIKE_WORLD, tag='_b%d' % i), world={}, payload={}) for i in range(8)]
    state = dict(version=CS.VERSION, worlds=worlds, config={}, trainer=dict(step=1))
    before = [json.dumps(w['env_kwargs']).encode() for w in state['worlds']]
    for settings in ('launch', 'saved'):
        up, changes = CS.migrate_env_kwargs(state, args, launch_kwargs=launch, env_settings=settings)
        assert changes == [] and up is state, settings
        assert [json.dumps(w['env_kwargs']).encode() for w in up['worlds']] == before, settings
    # the same resume that turns a fix on writes it, and only it
    on = TM.parse_args(argv + ['--position-rows', 'no_head_pelvis'])
    up, changes = CS.migrate_env_kwargs(state, on, launch_kwargs=TM.env_kwargs_for(on, None, 'x'),
                                        env_settings='launch')
    assert {c['name'] for c in changes} == {'position_rows'} and len(changes) == 8
    assert all(w['env_kwargs'] == dict(v['env_kwargs'], position_rows='no_head_pelvis')
               for w, v in zip(up['worlds'], state['worlds']))
    # the rollback: a world that carries a setting gets the launch's default written back
    deployed = dict(state, worlds=[dict(w, env_kwargs=dict(w['env_kwargs'], position_rows='no_head_pelvis',
                                                           parent_relative_targets='calf'))
                                   for w in state['worlds']])
    up, changes = CS.migrate_env_kwargs(deployed, args, launch_kwargs=launch, env_settings='launch')
    assert {(c['name'], c['new']) for c in changes} == {('position_rows', 'all'), ('parent_relative_targets', 'off')}
    assert all(w['env_kwargs']['position_rows'] == 'all' and w['env_kwargs']['parent_relative_targets'] == 'off'
               for w in up['worlds'])
    # a world the old code already wrote at the default keeps it, unchanged
    written = dict(state, worlds=[dict(w, env_kwargs=dict(w['env_kwargs'], **TM.LIMB_ENV_DEFAULTS))
                                  for w in state['worlds']])
    up, changes = CS.migrate_env_kwargs(written, args, launch_kwargs=launch, env_settings='launch')
    assert changes == [] and up is written


def test_the_window_summary_is_absent_until_an_order_is_projected():
    assert L.window_summary({}) is None
    w = {}
    L.window_add(w, [None, dict(calf=[dict(off_reach_before_deg=12.0), dict(off_reach_before_deg=4.0)],
                                chest=dict(off_reach_before_deg=9.0),
                                arm=[dict(active=3, converged=True, wrist_changed_deg=2.0, wrist_moved_cm=5.0,
                                          lowerarm_changed_deg=10.0, upperarm_changed_deg=20.0, ms=4.0),
                                     dict(active=0)])])
    s = L.window_summary(w)
    assert s['body_orders'] == 1 and s['arm_orders'] == 1 and s['arm_flow_not_settled'] == 0
    assert s['calf_off_reach_before_deg']['n'] == 2 and s['hand_moved_cm']['p50'] == 5.0
    assert s['chest_off_reach_before_deg']['p50'] == 9.0
    assert s['by_author'] is None, 'no author given: no split'
    # per author: how far the planner's own orders were moved, apart from babble's
    L.window_add(w, [dict(calf=[dict(off_reach_before_deg=30.0)])], author='planner')
    L.window_add(w, [dict(calf=[dict(off_reach_before_deg=2.0)])], author='reflex_noise')
    s = L.window_summary(w)
    assert s['body_orders'] == 3 and s['calf_off_reach_before_deg']['n'] == 4
    assert s['by_author']['planner']['calf_off_reach_before_deg']['p50'] == 30.0
    assert s['by_author']['reflex_noise']['body_orders'] == 1


# ------------------------------------------------------------------ the arms and the chest, on the real body
def _saved_body():
    models = sorted(glob.glob(str(SAVED_WORLD / 'models' / '*.mjb.gz')))
    chunks = sorted(glob.glob(str(SAVED_WORLD / 'b0' / 'chunk_*.npz')))
    if not models or not chunks:
        pytest.skip('the audit\'s saved world model and observer records are not on this machine')
    from mimo_bridge.observer_record import load_world_model, read_chunk
    from mimo_bridge.action_decoder import ActionDecoder
    model = load_world_model(models[0])
    decoder = ActionDecoder(model, kp=6.0, kd=0.3, body_layout='trunk_v1')
    qpos = np.concatenate([read_chunk(c)['qpos'] for c in chunks[:4]])
    return model, decoder, qpos


@pytest.fixture(scope='module')
def real_body():
    return _saved_body()


def _arm_reach_cm(model, prefix):
    """The farthest the hand's origin gets from the shoulder over the elbow's and wrist's ranges (the
    audit's arm_reach_cm, its straight-arm line of 24.37 cm): grid, then the best cell refined."""
    import itertools
    from scipy.optimize import minimize
    d = mujoco.MjData(model)
    js = [model.joint('robot:' + prefix + n).id for n in ('elbow', 'hand1', 'hand2', 'hand3')]
    adr = [int(model.jnt_qposadr[j]) for j in js]
    bu, bh = model.body(prefix + 'upper_arm').id, model.body(prefix + 'hand').id
    grids = [np.linspace(*model.jnt_range[j], n) for j, n in zip(js, (41, 9, 9, 9))]

    def dist(v):
        d.qpos[:] = model.qpos0
        d.qpos[adr] = v
        mujoco.mj_kinematics(model, d)
        return 100.0 * float(np.linalg.norm(d.xpos[bh] - d.xpos[bu]))
    best = max(itertools.product(*grids), key=dist)
    r = minimize(lambda v: -dist(v), np.asarray(best), method='L-BFGS-B', bounds=[tuple(model.jnt_range[j]) for j in js])
    return max(dist(best), -float(r.fun))


def _random_arm_order(data, decoder, rng, turn=1.6):
    order = _hold_order(data, decoder)
    order[:, 8] = 0.6
    for g in decoder.arm_groups:
        for mi in g['members']:
            order[mi, 3:7] = _turned(order[mi, 3:7], rng.normal(size=3), rng.uniform(0.0, turn))
            order[mi, 8] = rng.uniform(0.2, 1.0)
        order[g['members'][0], :3] += rng.uniform(-40.0, 40.0, size=3)   # execution_transform's 40 cm per axis
    return order


def _decoder_resting_point(model, data, order, qpos, iterations=400):
    """The decoder's own stacked arm solve iterated with forward kinematics: each step applies the
    joint change it solved (no physics). The servo gain is made tiny so no rail or rescale binds,
    and the change is read back from the control it hands MuJoCo: ctrl = owner strength x kp x dq."""
    from mimo_bridge.action_decoder import ActionDecoder
    dec = ActionDecoder(model, kp=1e-3, kd=0.0, body_layout='trunk_v1')
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_kinematics(model, data)
    for _ in range(iterations):
        ctrl = dec.decode(data, order, None)
        largest = 0.0
        for g in dec.arm_groups:
            wrist = dec.handles[g['members'][0]]
            for k, (a, j) in enumerate(zip(wrist.act_ids, wrist.joint_ids)):
                owner = g['owners'][k]
                dq = ctrl[a] / (1e-3 * float(np.clip(order[owner, 8], 0.0, 1.0)))
                data.qpos[model.jnt_qposadr[j]] += dq
                largest = max(largest, abs(dq))
        mujoco.mj_kinematics(model, data)
        if largest < 1e-10:
            break


def test_every_projected_hand_target_is_within_reach_and_the_decoder_rests_on_all_four(real_body):
    """The audit's F2b test: over 100 random orders at recorded poses, every projected hand target
    lies within the arm's reach from the shoulder at the latch, and the decoder's re-solve loop,
    iterated with forward kinematics, rests within 0.1 degrees and 0.1 cm of all four targets.

    Where it does not, its loop has stopped with a joint on its stop, at a local minimum of its own
    solve short of a pose the arm can reach (3 of 200 when measured; holding the stopped joint and
    solving the others again ends in the same place: motor_fixes_limbs_2026-09-25.md). That is the
    decoder's local search, outside F2b. The random turns here reach 92 degrees per row, beyond the
    recorded orders; on 960 real arm-orders the loop met all four targets in 941."""
    model, decoder, qpos = real_body
    data = mujoco.MjData(model)
    control = L.LimbTargetControl(model, decoder, reach_projection='arms')
    reach = {side: _arm_reach_cm(model, prefix) for side, prefix in (('L', 'left_'), ('R', 'right_'))}
    assert 23.0 < reach['L'] < 26.0 and abs(reach['L'] - reach['R']) < 0.05
    rng = np.random.default_rng(250925)
    beyond_before, rested, stalled = 0, 0, []
    for k in rng.choice(len(qpos), 100, replace=False):
        data.qpos[:] = qpos[k]
        mujoco.mj_kinematics(model, data)
        order = _random_arm_order(data, decoder, rng)
        motor, cache, ledger = control.execute(data, order, order)
        for g in decoder.arm_groups:
            wrist, upper = g['members'][0], g['members'][2]
            side = decoder.handle_names[wrist][-1]
            shoulder = data.xpos[decoder.handles[upper].body_id] * 100.0
            beyond_before += np.linalg.norm(order[wrist, :3] - shoulder) > reach[side]
            assert np.linalg.norm(motor[wrist, :3] - shoulder) <= reach[side] + 1e-6
            np.testing.assert_array_equal(ledger[g['members']], motor[g['members']])
        scratch = mujoco.MjData(model)
        _decoder_resting_point(model, scratch, motor, qpos[k])
        for arm in control.arms:
            worst_deg = max(_angle_deg(L.unit_wxyz_from_xyzw(motor[mi, 3:7]), scratch.xquat[arm.body[mi]])
                            for mi in arm.members)
            wrist = arm.members[0]
            worst_cm = float(np.linalg.norm(motor[wrist, :3] - scratch.xpos[arm.body[wrist]] * 100.0))
            if worst_deg <= 0.1 and worst_cm <= 0.1:
                rested += 1
                continue
            q_end = scratch.qpos[arm.addresses]
            on_stop = bool(np.any((np.abs(q_end - arm.lo) < 1e-9) | (np.abs(q_end - arm.hi) < 1e-9)))
            stalled.append((int(k), round(worst_deg, 2), round(worst_cm, 2), on_stop))
    assert beyond_before >= 40, 'the random orders did not reach past the arm: the check proves nothing'
    assert all(s[-1] for s in stalled), [s for s in stalled if not s[-1]]
    assert rested + len(stalled) == 200 and rested >= 190, (rested, stalled)


def test_moderate_orders_leave_nothing_unmet_at_the_decoders_resting_point(real_body):
    """Orders with turns up to 34 degrees per arm row and the hand up to 20 cm per axis (the ring's
    real orders ask a median 30 degrees per arm row and 23 cm of the hand): the decoder's own loop
    meets all four projected targets."""
    model, decoder, qpos = real_body
    data = mujoco.MjData(model)
    control = L.LimbTargetControl(model, decoder, reach_projection='arms')
    rng = np.random.default_rng(1925)
    for k in rng.choice(len(qpos), 40, replace=False):
        data.qpos[:] = qpos[k]
        mujoco.mj_kinematics(model, data)
        order = _random_arm_order(data, decoder, rng, turn=0.6)
        for g in decoder.arm_groups:
            hand = decoder.handles[g['members'][0]].body_id
            order[g['members'][0], :3] = data.xpos[hand] * 100.0 + rng.uniform(-20.0, 20.0, size=3)
        motor, cache, ledger = control.execute(data, order, order)
        scratch = mujoco.MjData(model)
        _decoder_resting_point(model, scratch, motor, qpos[k])
        for arm in control.arms:
            for mi in arm.members:
                assert _angle_deg(L.unit_wxyz_from_xyzw(motor[mi, 3:7]), scratch.xquat[arm.body[mi]]) <= 0.1
            wrist = arm.members[0]
            assert np.linalg.norm(motor[wrist, :3] - scratch.xpos[arm.body[wrist]] * 100.0) <= 0.1


def test_the_arm_objective_has_the_decoders_gradient(real_body):
    model, decoder, qpos = real_body
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[123]
    mujoco.mj_kinematics(model, data)
    control = L.LimbTargetControl(model, decoder, reach_projection='arms')
    order = _random_arm_order(data, decoder, np.random.default_rng(4))
    for arm in control.arms:
        f = arm.objective(model, mujoco.MjData(model), np.array(data.qpos), order, arm.active(order, None))
        q = np.clip(data.qpos[arm.addresses] + 0.05, arm.lo + 1e-3, arm.hi - 1e-3)
        _, analytic = f(q)
        numeric = np.zeros_like(q)
        for i in range(len(q)):
            e = np.zeros_like(q)
            e[i] = 1e-6
            numeric[i] = (f(q + e)[0] - f(q - e)[0]) / 2e-6
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_a_reachable_order_is_left_where_it_was(real_body):
    model, decoder, qpos = real_body
    data = mujoco.MjData(model)
    control = L.LimbTargetControl(model, decoder, reach_projection='arms')
    inside = [k for k in range(len(qpos))
              if all(np.all((qpos[k][a.addresses] > a.lo) & (qpos[k][a.addresses] < a.hi)) for a in control.arms)]
    assert len(inside) >= 3
    # (a recorded joint a hair past its stop is moved onto the stop, as the decoder's clip moves it)
    for k in inside[::max(1, len(inside) // 3)][:3]:
        data.qpos[:] = qpos[k]
        mujoco.mj_kinematics(model, data)
        order = _random_arm_order(data, decoder, np.random.default_rng(k))
        for g in decoder.arm_groups:           # the arm's own pose: reachable by definition
            for mi in g['members']:
                b = decoder.handles[mi].body_id
                order[mi, :3] = data.xpos[b] * 100.0
                order[mi, 3:7] = np.asarray(data.xquat[b])[[1, 2, 3, 0]]
        motor, cache, _ = control.execute(data, order, order)
        for g in decoder.arm_groups:
            for mi in g['members']:
                assert _angle_deg(L.unit_wxyz_from_xyzw(motor[mi, 3:7]), L.unit_wxyz_from_xyzw(order[mi, 3:7])) < 1e-4
                assert np.linalg.norm(motor[mi, :3] - order[mi, :3]) < 1e-4 or not decoder.use_position[mi]


def test_finger_rows_follow_the_projected_wrist_in_the_ledger(real_body):
    """With finger goals on, the ledger's finger rows stay the finger goal placed on the recorded
    wrist: on the projected wrist, not on the one that was asked for."""
    import torch
    from mimo_bridge.hand_target_control import HandKinematics, geometry_from_model
    model, decoder, qpos = real_body
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[40]
    mujoco.mj_kinematics(model, data)
    kin = HandKinematics(geometry_from_model(model, decoder))
    goal = kin.project_joints(torch.full((kin.n_joints,), 0.2, dtype=torch.float64)).numpy()
    order = kin.canonical_from_joints(_random_arm_order(data, decoder, np.random.default_rng(9)), goal)
    control = L.LimbTargetControl(model, decoder, reach_projection='arms')
    motor, cache, ledger = control.execute(data, order, order, hand=(kin, goal))
    fingers = [i for h in kin.geometry['hands'] for i in h['finger_indices']]
    np.testing.assert_allclose(ledger[fingers, :7], kin.canonical_from_joints(ledger, goal)[fingers, :7], atol=1e-12)
    assert np.abs(ledger[fingers, :3] - order[fingers, :3]).max() > 1.0, 'the wrist did not move: proves nothing'
    np.testing.assert_array_equal(ledger[fingers, 7:], order[fingers, 7:])
    # the rows the decoder reads for the fingers are hand_target_control's business, not this one's
    np.testing.assert_array_equal(motor[fingers], order[fingers])


def test_the_chest_on_the_real_body(real_body):
    """MIMo's own trunk (couplings: lean 2 = 2 x lean 1, twist and bend equal): at recorded poses
    with the pelvis's hinges inside their ranges and the upper body's placed as the tendons tie them,
    the upper body the pelvis's order implies is where MuJoCo puts it, and the stored chest row is a
    chest pose on it. (Recorded poses hold the upper body's hinges only within the tendons' soft
    limit of the coupled angles; that measured gap is in motor_fixes_limbs_2026-09-25.md.)"""
    model, decoder, qpos = real_body
    data = mujoco.MjData(model)
    chest = L._Chest(model, decoder)
    assert [c[1:] for c in chest.coupling] == [(1.0, -0.5, 0.0), (1.0, -1.0, 0.0), (1.0, -1.0, 0.0)]
    pelvis_adr = [model.jnt_qposadr[j] for j in decoder.handles[chest.pelvis_row].joint_ids]
    upper_adr = [model.jnt_qposadr[j] for j in range(model.body_jntadr[chest.upper_body],
                                                       model.body_jntadr[chest.upper_body] + 3)]
    rng = np.random.default_rng(31)
    for k in rng.choice(len(qpos), 20, replace=False):
        data.qpos[:] = qpos[k]
        data.qpos[pelvis_adr] = np.clip(data.qpos[pelvis_adr], chest.pelvis.lo + 1e-6, chest.pelvis.hi - 1e-6)
        data.qpos[upper_adr] = chest.upper_from_pelvis(data.qpos[pelvis_adr])
        mujoco.mj_kinematics(model, data)
        order = _hold_order(data, decoder)
        order[chest.chest_row, 3:7] = _turned(order[chest.chest_row, 3:7], rng.normal(size=3), 0.5)
        goal = chest.goal(np.array(data.xquat[chest.root_body]), order)
        assert _angle_deg(goal['upper_wxyz'], data.xquat[chest.upper_body]) < 1e-5
        stored = L.unit_wxyz_from_xyzw(chest.canonical_xyzw(goal, order))
        _, residual = chest.chest.closest(goal['upper_wxyz'], stored, chest.GRID_CHEST)
        assert math.degrees(residual) <= 1e-6


# ------------------------------------------------------------------ the real world, end to end
@pytest.fixture(scope='module')
def crib(tmp_path_factory):
    from mimo_bridge.env import BridgeEnv
    kw = dict(scene='crib', vision=False, age=24, caregiver_seconds=0, kp=6.0, body_layout='trunk_v1',
              cmd_horizon=25, hand_target_mode='wrist_relative_v1', gaze_reflex=False,
              spontaneous_saccades=False)
    on = BridgeEnv(run_dir=str(tmp_path_factory.mktemp('limbs_on')), tag='_limbs_on',
                   parent_relative_targets='calf_chest', reach_projection='arms', **kw)
    off = BridgeEnv(run_dir=str(tmp_path_factory.mktemp('limbs_off')), tag='_limbs_off', **kw)
    on.reset()
    off.reset()
    yield on, off
    on.close()
    off.close()


def _env_hold(env):
    order = np.zeros((len(env.decoder.handle_names), 9))
    order[:, 7], order[:, 8] = 0.5, 0.7
    for i, h in enumerate(env.decoder.handles):
        order[i, :3] = env.data.xpos[h.body_id] * 100.0
        order[i, 3:7] = np.asarray(env.data.xquat[h.body_id])[[1, 2, 3, 0]]
    return order


def test_a_world_with_the_fixes_off_reports_nothing_new(crib):
    _, off = crib
    assert off.limb_target_control is None
    out = off.step(_env_hold(off), (0.0, 0.0))
    assert not {'ledger_motor_targets', 'limb_target_contract', 'limb_target_report',
                'limb_target_geometry'} & set(out.info)
    assert '_limb_target_cache' not in off.save_full_state()['attrs']


def test_a_world_with_the_fixes_on_regenerates_projects_and_keeps_its_order_across_a_restore(crib):
    on, _ = crib
    names = list(on.decoder.handle_names)
    order = _env_hold(on)
    for i in (names.index('calf_L'), names.index('calf_R')):
        order[i, 3:7] = _turned(order[i, 3:7], [1.0, 0, 0], 0.5)
    order[names.index('chest'), 3:7] = _turned(order[names.index('chest'), 3:7], [0.2, 1.0, 0.3], 0.4)
    wrist = names.index('wrist_R')
    order[wrist, :3] += [30.0, -25.0, 20.0]
    first = on.step(order, (0.0, 0.0))
    info = first.info
    assert info['limb_target_contract'] == L.contract_of('calf_chest', 'arms')
    # the knees the learner's planner needs (only when the calves are projected), in the planner's rows
    assert info['limb_target_geometry'] == {'calf': on.limb_target_control.calf_geometry()}
    L.check_calf_geometry(info['limb_target_geometry']['calf'], names)
    ledger = info['ledger_motor_targets']
    control = on.limb_target_control
    for c in control.calves:
        assert _off_axis_deg(ledger[c.thigh_row, 3:7], ledger[c.calf_row, 3:7], c.body_quat, c.axis) <= 1e-6
    assert np.linalg.norm(ledger[wrist, :3] - order[wrist, :3]) > 5.0
    saved = on.save_full_state()
    assert saved['attrs']['_limb_target_cache'] is not None
    after_save = [on.step(order, (0.0, 0.0)) for _ in range(4)]
    for later in after_save:
        np.testing.assert_array_equal(later.info['ledger_motor_targets'], ledger)
        # the arm rows stay world targets for the whole order
        np.testing.assert_array_equal(later.info['executed_motor_targets'][wrist],
                                      info['executed_motor_targets'][wrist])
    # the calf and chest rows follow their parents: relative to the parent the decoder saw, the goal
    thighs = [np.array(on.data.xquat[c.thigh_body])[[1, 2, 3, 0]] for c in control.calves]
    upper = np.array(on.data.xquat[control.chest.upper_body])
    step = on.step(order, (0.0, 0.0))
    for c, goal, thigh in zip(control.calves, on._limb_target_cache['knee_goals'], thighs):
        sent = step.info['executed_motor_targets'][c.calf_row, 3:7]
        assert _off_axis_deg(thigh, sent, c.body_quat, c.axis) <= 1e-6
        assert abs(L.one_axis_goal(thigh, sent, c.body_quat, c.axis, c.ref, c.lo, c.hi)[0] - goal) < 1e-9
    chest_goal = on._limb_target_cache['chest_goal']['chest']
    sent = L.unit_wxyz_from_xyzw(step.info['executed_motor_targets'][control.chest.chest_row, 3:7])
    assert _angle_deg(sent, control.chest.chest.place(upper, chest_goal)) < 1e-9
    # a restored world replays the same order exactly, projection included
    on.restore_full_state(saved)
    again = on.step(order, (0.0, 0.0))
    np.testing.assert_array_equal(again.info['ledger_motor_targets'], ledger)
    np.testing.assert_array_equal(again.info['executed_motor_targets'], after_save[0].info['executed_motor_targets'])
