# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The fingers get their own armature, small enough that their motors can turn them.

Until 2026-09-22 every hinge except the eyes got the limbs' 0.02, which made each finger
about 200 times harder to turn than its own link: full torque on the index knuckle for one
model step (0.05 s) moved it 0.2 degrees. See the finger_armature notes in env.py.
"""
import numpy as np
import pytest

pytest.importorskip("mimo_infant")

from mimo_bridge.env import BridgeEnv, _is_finger_joint

MODEL_STEP_S = 0.05          # action_repeat 5 x the 10 ms env step


@pytest.fixture(scope="module")
def env():
    e = BridgeEnv(scene="reach", vision=False, tag="_pytest_finger_armature",
                  joint_damping_scale=3.0, hinge_armature=0.02, finger_armature=0.0004)
    yield e
    e.close()


def _armature(env, joint):
    return float(env.model.dof_armature[int(env.model.jnt_dofadr[env.model.joint(joint).id])])


def test_finger_joint_names():
    for name in ("robot:right_ff_knuckle", "robot:left_th_swivel", "robot:right_lf_meta",
                 "robot:left_mf_distal", "robot:right_rf_side"):
        assert _is_finger_joint(name)
    for name in ("robot:right_hand1", "robot:left_foot2", "robot:right_big_toe", "robot:left_toes",
                 "robot:right_elbow", "robot:left_eye_horizontal", "robot:head_tilt"):
        assert not _is_finger_joint(name)


def test_each_joint_family_gets_its_own_armature(env):
    fingers = [env.model.joint(j).name for j in range(env.model.njnt)
               if _is_finger_joint(env.model.joint(j).name)]
    assert len(fingers) == 44, fingers           # 5 digits x 2 hands, 22 hinges per hand
    for name in fingers:
        assert 0.0004 <= _armature(env, name) < 0.001, (name, _armature(env, name))
    assert _armature(env, "robot:right_elbow") >= 0.02
    assert _armature(env, "robot:left_hand2") >= 0.02
    assert _armature(env, "robot:left_eye_horizontal") < 0.001


def test_full_torque_moves_a_finger_within_one_model_step(env):
    """Measured on the crib body: 0.2 degrees at the old 0.0201 total, 33.7 with none added."""
    env.reset()
    for _ in range(20):
        env.step(ctrl=np.zeros(env.model.nu), render=False)
    snap = env.save_full_state()
    j = env.model.joint("robot:right_ff_knuckle").id
    adr = int(env.model.jnt_qposadr[j])
    act = env.model.actuator("act:right_ff_knuckle").id
    ends = []
    for sign in (+1.0, -1.0):
        env.restore_full_state(snap)
        q0 = float(env.data.qpos[adr])
        ctrl = np.zeros(env.model.nu)
        ctrl[act] = sign
        for _ in range(int(round(MODEL_STEP_S / env.dt))):
            env.step(ctrl=ctrl, render=False)
        ends.append(float(env.data.qpos[adr]) - q0)
    separation_deg = np.rad2deg(ends[0] - ends[1])
    assert separation_deg > 5.0, f"index knuckle moved {separation_deg:.2f} degrees in {MODEL_STEP_S} s"
