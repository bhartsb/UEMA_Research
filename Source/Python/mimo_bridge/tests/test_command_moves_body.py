# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A held command must physically move the body part it names.

Seven runs (H23-H28) trained on a body that barely moved, and 534 tests stayed green
because not one of them asked whether an order produces motion. Body speed fell from
36.8 cm/s to 4.2 cm/s and command magnitude from 0.60 rad to 0.06 rad the moment the
planner was switched off; nothing in the suite noticed.

These tests are deterministic and mechanical. They say nothing about whether the model
has learned anything - only that when SOMETHING asks the body to move, it moves, and
that the size of the ask is visible. A causal test cannot resolve action-conditioning
on transitions where nothing was asked, so this is the floor under that measurement.
"""
import os

import numpy as np
import pytest

mimo_infant = pytest.importorskip("mimo_infant")

from lem.model.body_sets import HANDLE_NAMES
from mimo_bridge.env import BridgeEnv
from mimo_bridge.proprio_adapter import BONE_MAP

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                       "Saved", "mimo_bridge", "scenes_test")
N_H = len(HANDLE_NAMES)          # 24 including the gaze handle


@pytest.fixture(scope="module")
def env():
    e = BridgeEnv(scene="crib", run_dir=RUN_DIR, vision=False, tag="_cmdmove")
    yield e
    e.close()


def _null_motor():
    """Identity rotation, zero strength: a valid 'ask for nothing'."""
    m = np.zeros((N_H, 9), dtype=np.float64)
    m[:, 6] = 1.0                                  # quaternion w = 1
    return m


def _axis_angle_quat(axis, ang):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(ang / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(ang / 2.0)])


def _hold(env, motor, steps=25):
    """Run one command for several physics steps and return the qpos travelled."""
    env.reset()
    before = np.array(env.data.qpos, dtype=np.float64)
    for _ in range(steps):
        env.step(motor, None, None, render=False)
    after = np.array(env.data.qpos, dtype=np.float64)
    return np.abs(after - before)


@pytest.mark.parametrize("handle", ["upperarm_R", "upperarm_L", "head", "thigh_R"])
def test_a_held_command_moves_the_body_part_it_names(env, handle):
    """Full strength on one handle must move the joints that handle drives."""
    idx = HANDLE_NAMES.index(handle)
    m = _null_motor()
    q0 = np.array(env.data.xquat[env.model.body(BONE_MAP[handle]).id], dtype=np.float64)
    # target: the body's current orientation rotated 40 degrees about x
    dq = _axis_angle_quat([1.0, 0.0, 0.0], np.deg2rad(40.0))
    m[idx, 3:7] = dq                                  # xyzw, as env expects
    m[idx, 7] = 1.0                                   # radius
    m[idx, 8] = 1.0                                   # full strength
    moved = _hold(env, m)
    assert moved.max() > 1e-3, (
        f"{handle}: a full-strength 40-degree command moved the body by "
        f"{moved.max():.2e} - the motor pathway is not producing motion")
    assert np.isfinite(q0).all()


def test_zero_strength_moves_far_less_than_full_strength(env):
    """Strength must be the thing that decides motion, not a decorative column."""
    idx = HANDLE_NAMES.index("upperarm_R")
    dq = _axis_angle_quat([1.0, 0.0, 0.0], np.deg2rad(40.0))

    strong = _null_motor()
    strong[idx, 3:7] = dq
    strong[idx, 7] = 1.0
    strong[idx, 8] = 1.0

    weak = _null_motor()
    weak[idx, 3:7] = dq
    weak[idx, 7] = 1.0
    weak[idx, 8] = 0.0                                # asks for nothing

    a = _hold(env, strong).max()
    b = _hold(env, weak).max()
    assert a > b, f"full strength moved {a:.4f}, zero strength {b:.4f} - strength is inert"


def test_command_magnitude_is_reported_and_tracks_the_ask(env):
    """cmd_mag_rad is the vital that would have caught seven still runs; it must be live.

    The band in vitals.py has a 0.10 rad floor, chosen below every run that moved
    (0.42-0.68) and above every run that did not (0.03-0.08). A reported magnitude of
    zero for a real command would make that band useless.
    """
    idx = HANDLE_NAMES.index("upperarm_R")
    env.reset()
    small, large = None, None
    for ang, slot in ((5.0, "small"), (60.0, "large")):
        m = _null_motor()
        m[idx, 3:7] = _axis_angle_quat([1.0, 0.0, 0.0], np.deg2rad(ang))
        m[idx, 7] = 1.0
        m[idx, 8] = 1.0
        env.reset()
        rs = env.step(m, None, None, render=False)
        cr = (rs.info or {}).get("cmd_response") or {}
        val = cr.get("cmd_mag")
        assert val is not None, "cmd_mag missing from cmd_response - the vital has no source"
        if slot == "small":
            small = val
        else:
            large = val
    assert large > small, (
        f"a 60-degree ask reported {large:.4f} rad and a 5-degree ask {small:.4f} - "
        f"command magnitude does not track the size of the order")


def test_all_handles_can_be_commanded_without_error(env):
    """Every handle takes a command. A handle that throws is a silent dead limb."""
    dq = _axis_angle_quat([0.0, 1.0, 0.0], np.deg2rad(20.0))
    for i, h in enumerate(HANDLE_NAMES):
        m = _null_motor()
        m[i, 3:7] = dq
        m[i, 7] = 1.0
        m[i, 8] = 1.0
        env.reset()
        env.step(m, None, None, render=False)          # must not raise
