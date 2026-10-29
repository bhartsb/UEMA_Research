# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The body-goal comparison's witnesses and ranking (docs/gain_of_function_v3/
compare_hand_body_goals_2026-09-24.py, 2026-09-25).

The scheduler now runs its hand_and_body suite: closing and opening the hand, a left-wrist reach, a
supported trunk turn and a left-elbow flex. These tests pin the elbow request against a small MuJoCo
arm (the request must be what the body's own elbow joint does when it flexes), the witness check that
says whether a request moved the requested part beyond the controller's endpoint, and the outside
review's ranking over the arms of each trial. No model checkpoint or saved world is loaded.
"""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / 'docs/gain_of_function_v3/compare_hand_body_goals_2026-09-24.py'


def _load():
    sys.path.insert(0, str(SCRIPT.parent))          # its sibling helpers (compare_synergy_physical)
    spec = importlib.util.spec_from_file_location('compare_hand_body_goals', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load()

ARM = """
<mujoco>
  <worldbody>
    <body name="left_upper_arm" pos="0 0.1 0.5" euler="0 30 0">
      <joint name="robot:left_shoulder" type="hinge" axis="1 0 0"/>
      <geom type="capsule" size="0.02 0.05" pos="0 0 0.05"/>
      <body name="left_lower_arm" pos="0 0 0.12" euler="15 0 0">
        <joint name="robot:left_elbow" type="hinge" axis="0 -1 0" range="-146 5"/>
        <geom type="capsule" size="0.02 0.04" pos="0 0 0.04"/>
        <body name="left_hand" pos="0.007 0.009 0.11">
          <geom type="sphere" size="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
BODIES = ('left_lower_arm', 'left_hand', 'left_upper_arm')


def arm():
    import mujoco
    model = mujoco.MjModel.from_xml_string(ARM)
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('robot:left_shoulder').id]] = 0.3
    data.qpos[model.jnt_qposadr[model.joint('robot:left_elbow').id]] = -0.5
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def pose_command(model, data):
    """A controller request equal to where the three arm handles are now (positions in the raw
    command's units, cm / 200; orientations x y z w), strength 0.5."""
    command = torch.zeros(1, len(BODIES), 9)
    for i, name in enumerate(BODIES):
        b = model.body(name).id
        command[0, i, :3] = torch.tensor(data.xpos[b] * 100.0 / 200.0)
        w, x, y, z = data.xquat[b]
        command[0, i, 3:7] = torch.tensor([x, y, z, w])
        command[0, i, 8] = 0.5
    return command


def angle_deg(q_xyzw, r_xyzw):
    return math.degrees((Rotation.from_quat(q_xyzw) * Rotation.from_quat(r_xyzw).inv()).magnitude())


def test_the_suites_are_the_ones_the_scheduler_validates_and_every_goal_has_a_check():
    from mimo_bridge.tools import ability_rechecks as worker
    assert C.GOAL_SUITES == worker.BODY_GOAL_SUITES
    assert set(C.GOAL_SUITES['hand_and_body']) == set(C.WITNESS_PARTS)
    assert C.TURN_DEG == 11.0 and C.REACH_CM == 1.8, 'the planners\' own search bounds (EFFERENCE_SCALES)'


@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_the_elbow_request_is_what_the_body_does_when_its_elbow_flexes_or_extends(sign):
    """The forearm and hand requests turned about the elbow hinge through its anchor, toward flexion
    (sign 1) or extension (sign -1, the mirrored control): turning the real joint by the same angle must
    put the forearm and hand exactly there; the wrist's requested move, when a coordinate would pass
    1.8 cm, is shortened along its own direction (never bent off it)."""
    mujoco, model, data = arm()
    session = SimpleNamespace(env=SimpleNamespace(model=model, data=data),
                              handle_names=['lowerarm_L', 'wrist_L', 'upperarm_L'])
    before = pose_command(model, data)
    request = C.elbow_flex_request(session, before, sign)
    elbow = model.jnt_qposadr[model.joint('robot:left_elbow').id]
    data.qpos[elbow] -= sign * math.radians(C.TURN_DEG)   # MIMo flexes toward its lower limit
    mujoco.mj_forward(model, data)
    turned_arm = pose_command(model, data)
    for row in (0, 1):
        assert angle_deg(request[0, row, 3:7].numpy(), turned_arm[0, row, 3:7].numpy()) < 1e-4, BODIES[row]
        assert angle_deg(request[0, row, 3:7].numpy(), before[0, row, 3:7].numpy()) == pytest.approx(C.TURN_DEG, abs=1e-4)
    old, new = before[0, 1, :3].numpy() * 200.0, turned_arm[0, 1, :3].numpy() * 200.0
    move = new - old
    expected = old + move * min(1.0, C.REACH_CM / np.abs(move).max())
    assert np.allclose(request[0, 1, :3].numpy() * 200.0, expected, atol=1e-4)
    assert np.abs(move).max() > 0.5, 'the fixture must move the hand for the position check to mean anything'
    assert torch.equal(request[0, 2], before[0, 2]), 'the upper arm request is the controller\'s'
    assert torch.equal(request[..., 7:], before[..., 7:]) and torch.equal(request[0, 0, :3], before[0, 0, :3])


def test_the_wrist_reach_asks_above_the_controllers_own_request_and_its_mirror_below():
    """The reach is built from the controller's request (review, 2026-09-25: built from the actual
    wrist it could lie outside the planners' search region, and its direction need not point up)."""
    names = ['wrist_L', 'chest', 'gaze']
    session = SimpleNamespace(handle_names=names, model=SimpleNamespace(
        _hand_geometry=dict(joints=[]), hand_kinematics=SimpleNamespace(project_joints=lambda j: j)))
    raw = torch.zeros(1, 3, 9)
    raw[0, :, :3] = torch.tensor([[0.10, 0.02, 0.30], [0.0, 0.0, 0.25], [0.0, 0.0, 0.0]])
    raw[0, :, 6] = 1.0
    up, _, recipe = C.witness_request(session, raw, torch.zeros(1, 0), 'wrist_reach_witness')
    down, _, _ = C.witness_request(session, raw, torch.zeros(1, 0), 'wrist_reach_witness', mirror=True)
    step = C.REACH_CM / 200.0
    assert torch.allclose(up[0, 0, :3], raw[0, 0, :3] + torch.tensor([0.0, 0.0, step]))
    assert torch.allclose(down[0, 0, :3], raw[0, 0, :3] - torch.tensor([0.0, 0.0, step]))
    assert torch.equal(up[0, 1:], raw[0, 1:]) and torch.equal(down[0, 1:], raw[0, 1:])
    assert "controller's own request" in recipe


def handles(names, rows):
    """[H, 7] rows: position cm and x y z w, identity where not given."""
    out = np.zeros((len(names), 7))
    out[:, 6] = 1.0
    for name, row in rows.items():
        out[names.index(name)] = row
    return out


def turned(deg, axis=(0.0, 0.0, 1.0)):
    return np.concatenate([np.zeros(3), Rotation.from_rotvec(np.radians(deg) * np.asarray(axis)).as_quat()])


def wrist_check(witness_end, mirror_end, mirror_asked=-0.63):
    names = ['wrist_L', 'chest']
    at = lambda z: np.array([0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0])
    none = dict(controller=np.zeros(2), witness=np.zeros(2), mirror=np.zeros(2))
    return C.witness_check(
        'wrist_reach_witness', names, [], handles(names, {}),
        raw=dict(controller=handles(names, {'wrist_L': at(0.0)}), witness=handles(names, {'wrist_L': at(1.8)})),
        executed=dict(controller=handles(names, {'wrist_L': at(0.0)}), witness=handles(names, {'wrist_L': at(0.63)}),
                      mirror=handles(names, {'wrist_L': at(mirror_asked)})),
        ends=dict(controller=handles(names, {'wrist_L': at(0.1)}), witness=handles(names, {'wrist_L': at(witness_end)}),
                  mirror=handles(names, {'wrist_L': at(mirror_end)})),
        start_joints=np.zeros(2), raw_joints=none, executed_joints=none, end_joints=none)


def test_the_witness_check_measures_the_request_beyond_the_controller_in_centimetres_with_its_mirror():
    check = wrist_check(0.4, -0.2)
    assert check['unit'] == 'cm' and check['parts'] == ['wrist_L'] and check['informative'] is True
    assert check['requested_raw'] == pytest.approx(1.8) and check['requested_executed'] == pytest.approx(0.63)
    assert check['mirror_requested_executed'] == pytest.approx(-0.63) and check['mirror_request_blocked'] is False
    assert check['achieved_beyond_controller'] == pytest.approx(0.3) and check['achieved_share'] == pytest.approx(0.3 / 0.63)
    assert check['mirror_beyond_controller'] == pytest.approx(-0.3) and check['mirror_share'] == pytest.approx(-0.3 / 0.63)
    assert check['separation_from_controller'] == pytest.approx(0.3) and check['moved_from_start'] == pytest.approx(0.4)
    assert wrist_check(-0.1, -0.3)['informative'] is False, 'the request moved the wrist the wrong way'
    # both requests end at the same place: a push that happens whatever is asked, not the request's doing
    pushed = wrist_check(0.4, 0.4)
    assert pushed['achieved_share'] > 0 and pushed['mirror_share'] == pushed['achieved_share']
    assert pushed['informative'] is False
    # the mirror cannot go down (the wrist lies on the mat): it stays at the controller's endpoint and
    # does not count against a witness that moved the wrist up (review, 2026-09-25)
    blocked = wrist_check(0.4, 0.1)
    assert blocked['mirror_share'] == pytest.approx(0.0) and blocked['informative'] is True
    # a mirror clamped at the command itself does not even ask for the reverse move
    unasked = wrist_check(0.4, 0.1, mirror_asked=0.0)
    assert unasked['mirror_request_blocked'] is True and unasked['informative'] is True


def test_the_witness_check_measures_turns_in_degrees_and_finger_joints_in_degrees():
    names = ['lower_trunk', 'chest', 'wrist_L']
    none = dict(controller=np.zeros(3), witness=np.zeros(3), mirror=np.zeros(3))
    check = C.witness_check(
        'supported_trunk_turn_witness', names, [], handles(names, {}),
        raw=dict(controller=handles(names, {}), witness=handles(names, {'lower_trunk': turned(11), 'chest': turned(11)})),
        executed=dict(controller=handles(names, {}), witness=handles(names, {'lower_trunk': turned(3.85), 'chest': turned(3.85)}),
                      mirror=handles(names, {'lower_trunk': turned(-3.85), 'chest': turned(-3.85)})),
        ends=dict(controller=handles(names, {}), witness=handles(names, {'lower_trunk': turned(2.0), 'chest': turned(2.0)}),
                  mirror=handles(names, {'lower_trunk': turned(-1.0), 'chest': turned(-1.0)})),
        start_joints=np.zeros(3), raw_joints=none, executed_joints=none, end_joints=none)
    assert check['unit'] == 'degrees' and check['requested_raw'] == pytest.approx(11.0)
    assert check['requested_executed'] == pytest.approx(3.85) and check['achieved_share'] == pytest.approx(2.0 / 3.85)
    assert check['mirror_requested_executed'] == pytest.approx(-3.85)
    assert check['mirror_share'] == pytest.approx(-1.0 / 3.85) and check['informative'] is True
    fingers = C.witness_check(
        'curl_witness', names, [0, 2], handles(names, {}),
        raw=dict(controller=handles(names, {}), witness=handles(names, {})),
        executed=dict(controller=handles(names, {}), witness=handles(names, {}), mirror=handles(names, {})),
        ends=dict(controller=handles(names, {}), witness=handles(names, {}), mirror=handles(names, {})),
        start_joints=np.zeros(3),
        raw_joints=dict(controller=np.zeros(3), witness=np.array([1.0, 9.0, 1.0])),
        executed_joints=dict(controller=np.zeros(3), witness=np.array([0.35, 9.0, 0.35]),
                             mirror=np.array([-0.35, 9.0, -0.35])),
        end_joints=dict(controller=np.full(3, 0.1), witness=np.array([0.3, 9.0, 0.3]),
                        mirror=np.array([-0.05, 9.0, -0.05])))
    assert fingers['parts'] == 'finger joints' and fingers['unit'] == 'degrees'
    assert fingers['requested_executed'] == pytest.approx(math.degrees(0.35))
    assert fingers['achieved_share'] == pytest.approx(0.2 / 0.35) and fingers['informative'] is True
    assert fingers['mirror_share'] == pytest.approx(-0.15 / 0.35) and fingers['mirror_request_blocked'] is False
    assert fingers['moved_from_start'] == pytest.approx(math.degrees(0.3))


def test_the_elbow_check_reads_the_elbow_joint_so_a_shoulder_turn_cannot_count():
    """The forearm's world orientation also turns with the shoulder; the elbow goal is judged on the
    elbow joint's own angle (review, 2026-09-25). Flexing lowers the angle."""
    names = ['lowerarm_L', 'wrist_L']
    axis = np.array([0.0, -1.0, 0.0])                     # flexion turns about +y: minus the hinge axis
    flexion = lambda deg: np.concatenate([np.zeros(3), Rotation.from_rotvec(np.radians(deg) * -axis).as_quat()])
    none = dict(controller=np.zeros(2), witness=np.zeros(2), mirror=np.zeros(2))

    def check(witness_deg, mirror_deg, shoulder=0.0):
        return C.witness_check(
            'elbow_flex_witness', names, [], handles(names, {}),
            raw=dict(controller=handles(names, {}), witness=handles(names, {'lowerarm_L': flexion(11.0)})),
            executed=dict(controller=handles(names, {}), witness=handles(names, {'lowerarm_L': flexion(3.85)}),
                          mirror=handles(names, {'lowerarm_L': flexion(-3.85)})),
            ends=dict(controller=handles(names, {}), witness=handles(names, {'lowerarm_L': flexion(shoulder)}),
                      mirror=handles(names, {})),
            start_joints=np.zeros(2), raw_joints=none, executed_joints=none, end_joints=none,
            elbow=dict(axis=axis, start=-0.35, controller=-0.35, witness=-0.35 - math.radians(witness_deg),
                       mirror=-0.35 - math.radians(mirror_deg)))
    flexed = check(2.0, -1.5)
    assert flexed['kind'] == 'elbow' and flexed['parts'] == 'left elbow joint' and flexed['unit'] == 'degrees'
    assert flexed['requested_raw'] == pytest.approx(11.0) and flexed['requested_executed'] == pytest.approx(3.85)
    assert flexed['achieved_beyond_controller'] == pytest.approx(2.0) and flexed['mirror_beyond_controller'] == pytest.approx(-1.5)
    assert flexed['mirror_requested_executed'] == pytest.approx(-3.85)
    assert flexed['informative'] is True and flexed['elbow_start_deg'] == pytest.approx(math.degrees(-0.35))
    # the forearm turned 3 degrees about the flexion axis by the shoulder, the elbow did not move
    shoulder = check(0.0, 0.0, shoulder=3.0)
    assert shoulder['achieved_beyond_controller'] == pytest.approx(0.0) and shoulder['informative'] is False


def test_a_goal_summary_counts_the_informative_bodies_with_the_medians_of_both_requests():
    checks = [dict(unit='degrees', parts='left elbow joint', informative=i < 5, requested_raw=11.0, requested_executed=3.9,
                   mirror_requested_executed=-3.9 if i else 0.0, mirror_request_blocked=not i,
                   achieved_beyond_controller=0.5 * i, mirror_beyond_controller=-0.25 * i, achieved_share=0.1 * i,
                   mirror_share=-0.05 * i, separation_from_controller=1.0 + i, moved_from_start=2.0) for i in range(8)]
    out = C.witness_summary(checks, 8)
    assert out['bodies'] == 8 and out['informative_bodies'] == 5 and out['unit'] == 'degrees'
    assert out['mirror_request_blocked_bodies'] == 1 and out['mirror_requested_executed'] == pytest.approx(-3.9)
    assert out['achieved_share'] == pytest.approx(0.35) and out['mirror_share'] == pytest.approx(-0.175)
    assert out['achieved_beyond_controller'] == pytest.approx(1.75) and out['mirror_beyond_controller'] == pytest.approx(-0.875)


def trial(body, forecast, actual, goal='elbow_flex_witness', check=None):
    return dict(body=body, goal=goal, witness_check=check, outcomes={
        m: dict(predicted_goal_error=float(p), physical_goal_error=float(a))
        for m, p, a in zip(('controller', 'direct', 'synergy'), forecast, actual)})


def test_a_whole_suite_assembles_a_result_the_scheduler_accepts(tmp_path):
    """ranking_and_witnesses over 8 bodies x the five goals of hand_and_body, placed in a result the
    way main() places them, read back from JSON and judged by the scheduler's validator."""
    import json
    from mimo_bridge.tools import ability_rechecks as worker
    rng = np.random.default_rng(5)
    goals = C.GOAL_SUITES['hand_and_body']
    rows = []
    for goal in goals:
        for body in range(8):
            actual = rng.uniform(0.2, 1.0, 3)
            check = dict(kind='rotation', parts=['lowerarm_L'], unit='degrees', requested_raw=11.0,
                         requested_executed=3.9, achieved_beyond_controller=float(rng.normal()), achieved_share=0.3,
                         separation_from_controller=1.5, moved_from_start=2.0, informative=bool(body % 2))
            rows.append(trial(body, actual + rng.normal(0, 0.3, 3), actual, goal, check))
    ranking, witnesses = C.ranking_and_witnesses(rows, goals, ('controller', 'direct', 'synergy'), 'controller', 8)
    assert set(ranking['by_goal']) == set(goals) and ranking['reference'] == 'controller'
    assert ranking['optimized_arms'] == ['direct', 'synergy'], 'the arms found by searching this same forecast'
    assert all(w['informative_bodies'] == 4 and w['bodies'] == 8 for w in witnesses.values())
    metric = dict(means=dict(controller=1.0, direct=1.1, synergy=1.2), paired_bootstrap95=[-0.1, 0.2])
    payload = dict(checkpoint=str(tmp_path / 'ckpt'), learning=False, elapsed_s=5.0, bodies=8, goals=goals,
                   goal_suite='hand_and_body', weights_unchanged=True, exact_replay=True, script_sha256='s',
                   summary={m: metric for m in ('physical_goal_error', 'hand_joint_error_deg', 'wrist_goal_error_cm')},
                   by_goal={g: dict(witness_check=witnesses[g]) for g in goals}, ranking=ranking)
    payload = json.loads(json.dumps(payload))
    record = dict(checkpoint=payload['checkpoint'], script_sha256='s',
                  command=['python', '-u', str(SCRIPT), '--checkpoint', 'c', '--out', 'o', '--goal-suite', 'hand_and_body'])
    worker.validate_result(payload, record)


def test_the_ranking_runs_over_the_arms_of_each_trial_with_the_controller_as_the_reference():
    rng = np.random.default_rng(3)
    arms = ('controller', 'direct', 'synergy')
    good, overrated = [], []
    for body in range(8):
        actual = np.array([1.0, 0.6, 0.7]) + rng.normal(0, 0.01, 3)
        good.append(trial(body, actual, actual))
        worse = np.array([0.5, 0.6, 0.7]) + rng.normal(0, 0.01, 3)
        # the search lands where the forecast overrates the command: predicted best, actually worst
        overrated.append(trial(body, 8.0 * (1.2 - worse), worse))
    used = C.ranking_of(good, arms, 'controller', 'elbow')
    assert used['per_100'] == 100.0 and used['case'] == 'ranking_used' and set(used['planners']) == {'direct', 'synergy'}
    wrong = C.ranking_of(overrated, arms, 'controller', 'elbow')
    assert wrong['per_100'] == 0.0 and wrong['below_chance'] and wrong['case'] == 'case_2_wrong_order'
    assert wrong['spread_ratio'] == pytest.approx(8.0, rel=1e-6)
