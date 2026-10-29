# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""How the effort that drains the fatigue model is computed (review env-senses MINOR 6, 2026-09-23).

proprio_adapter.FatigueModel is drained by effort_per_set, which has always been |actuator force
x joint speed|. MIMo's actuator force is the motor's output before its gear (on the 24-month body
the gears run 0.0026 N m to 12.62 N m per unit of force), so that is not power and it weighs the
body sets wrongly against each other. The
corrected power (force x gear x joint speed, watts) is built behind BridgeEnv(effort_power=
"with_gear"); the default "no_gear" keeps every run's formula, because switching changes how
fast each body set tires and so the force its motors can give (the operator's decision).
"""
import inspect

import numpy as np
import pytest

from mimo_bridge import env as E
from mimo_bridge.proprio_adapter import EFFORT_POWER_MODES, ProprioAdapter


def test_the_default_is_the_formula_every_run_used():
    assert inspect.signature(E.BridgeEnv.__init__).parameters["effort_power"].default == "no_gear"
    assert inspect.signature(ProprioAdapter.__init__).parameters["effort_power"].default == "no_gear"
    assert EFFORT_POWER_MODES == ("no_gear", "with_gear")


@pytest.fixture(scope="module")
def crib(tmp_path_factory):
    env = E.BridgeEnv(scene="crib", run_dir=str(tmp_path_factory.mktemp("effort")), vision=False,
                      tag="_efforttest", age=24, contact_profile="mechanical_v4",
                      body_function="modulate", caregiver_seconds=0, kp=6.0,
                      body_layout="trunk_v1", cmd_horizon=25)
    env.reset()
    yield env
    env.close()


def test_an_unknown_mode_is_refused(crib):
    with pytest.raises(ValueError):
        ProprioAdapter(crib.model, crib.data, body_layout="trunk_v1", effort_power="watts")


def _per_set(env, per_actuator):
    out = np.zeros(len(env.proprio.fatigue.cap))
    sets = env.proprio.act_set
    for a in range(env.model.nu):
        if sets[a] >= 0:
            out[sets[a]] += per_actuator[a]
    return out


def test_each_mode_computes_its_own_formula(crib):
    env = crib
    rng = np.random.default_rng(9)
    ctrl = rng.uniform(-0.8, 0.8, env.model.nu)
    for _ in range(10):
        env.step(ctrl=ctrl)
    m, d = env.model, env.data
    force = np.asarray(d.actuator_force, dtype=np.float64)
    qd = np.asarray(d.qvel, dtype=np.float64)[env.proprio.act_dof]
    gear = np.asarray(m.actuator_gear[:, 0], dtype=np.float64)
    assert env.proprio.effort_power == "no_gear"
    no_gear = env.proprio.effort_per_set()
    assert np.array_equal(no_gear, _per_set(env, np.abs(force * qd))), "the default formula moved"
    try:
        env.proprio.effort_power = "with_gear"
        with_gear = env.proprio.effort_per_set()
    finally:
        env.proprio.effort_power = "no_gear"
    assert np.allclose(with_gear, _per_set(env, np.abs(force * gear * qd)), rtol=1e-12, atol=0)
    # the gear changes the body sets' relative drain, not only the overall size
    ratio = with_gear[no_gear > 1e-9] / no_gear[no_gear > 1e-9]
    assert ratio.max() / ratio.min() > 10.0, ratio


def test_the_setting_survives_a_full_state_restore(crib):
    """restore_full_state replaces the fatigue model's whole __dict__ from the snapshot, so the
    setting must not live there."""
    env = crib
    env.proprio.effort_power = "with_gear"
    try:
        snap = env.save_full_state()
        env.step(ctrl=np.zeros(env.model.nu))
        env.restore_full_state(snap)
        assert env.proprio.effort_power == "with_gear"
        old = dict(snap)
        old["fatigue"] = {"cap": np.ones(10), "accum": np.zeros(10), "recover": 0.05,
                          "drain": 0.02, "p_ref": 1.0}
        env.restore_full_state(old)
        assert env.proprio.effort_power == "with_gear"
    finally:
        env.proprio.effort_power = "no_gear"
