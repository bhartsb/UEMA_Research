# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Settle tests for the inter-finger coupling tendons (MIMo_metav2.xml).

The tendons are combined-limit bands (juncturae tendinum), not springs: inside the
band adjacent knuckles move freely at zero constraint force; at the band edge one
finger cannot keep flexing while its neighbour stays extended. Style follows the
newborn-posture commit: drive the real body, settle, measure.
"""
import os

import numpy as np
import pytest

mimo_infant = pytest.importorskip("mimo_infant")

from mimo_bridge.env import BridgeEnv

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                       "Saved", "mimo_bridge", "scenes_test")

# band widths as written in the XML (radians)
BAND = {"ff_mf": 0.52, "mf_rf": 0.44, "rf_lf": 0.35}
SOFT = 0.15   # limited tendons are soft constraints; slack allowed beyond the band


@pytest.fixture(scope="module")
def env():
    e = BridgeEnv(scene="reach", run_dir=RUN_DIR, vision=False, tag="_pytest_tendon")
    yield e
    e.close()


def _knuckle_qpos(env, side, finger):
    j = env.model.joint(f"robot:{side}_{finger}_knuckle").id
    return float(env.data.qpos[env.model.jnt_qposadr[j]])


def _drive(env, side, finger, seconds=1.5, ctrl_val=1.0):
    """Full-flex torque on one knuckle, everything else 0, settle, return qpos of all four."""
    env.reset()
    ctrl = np.zeros(env.model.nu)
    a = env.model.actuator(f"act:{side}_{finger}_knuckle").id
    ctrl[a] = ctrl_val
    for _ in range(int(seconds / env.dt)):
        env.step(ctrl=ctrl, render=False)
    return {f: _knuckle_qpos(env, side, f) for f in ("ff", "mf", "rf", "lf")}


def test_tendons_present_in_compiled_model(env):
    names = [env.model.tendon(i).name for i in range(env.model.ntendon)]
    for side in ("right", "left"):
        for pair in ("ff_mf", "mf_rf", "rf_lf"):
            assert f"finger_couple_{side}_{pair}" in names


def test_ring_full_flex_drags_little_finger(env):
    q = _drive(env, "right", "rf")
    # the band limit is the invariant: ring cannot run away from little by more than
    # the rf_lf band (plus constraint softness)
    assert q["rf"] > 0.6, f"ring knuckle barely flexed ({q['rf']:.2f} rad) - actuator dead?"
    assert q["rf"] - q["lf"] <= BAND["rf_lf"] + SOFT, \
        f"ring ran {q['rf'] - q['lf']:.2f} rad ahead of little; band is {BAND['rf_lf']}"
    assert q["lf"] > 0.15, f"little finger was not dragged (lf={q['lf']:.2f} rad)"


def test_index_is_most_independent(env):
    q = _drive(env, "right", "ff")
    assert q["ff"] > 0.6, f"index knuckle barely flexed ({q['ff']:.2f} rad)"
    assert q["ff"] - q["mf"] <= BAND["ff_mf"] + SOFT
    # index enjoys the widest band: it must get FURTHER ahead of its neighbour than
    # ring can get ahead of little (the enslaving ordering, by construction)
    q2 = _drive(env, "right", "rf")
    assert (q["ff"] - q["mf"]) > (q2["rf"] - q2["lf"]) - 1e-6


def test_inside_band_costs_nothing(env):
    """A gentle flex that stays inside the band must leave the neighbours at rest:
    the tendon is a limit, not a spring - no pull before the edge."""
    q = _drive(env, "right", "ff", ctrl_val=0.12, seconds=1.0)
    if q["ff"] - q["mf"] < BAND["ff_mf"] - 0.05:      # premise: we stayed inside the band
        assert abs(q["mf"]) < 0.05, f"middle moved ({q['mf']:.3f} rad) with the pair inside the band"
        assert abs(q["rf"]) < 0.05 and abs(q["lf"]) < 0.05


def test_left_hand_mirrors(env):
    q = _drive(env, "left", "rf")
    assert q["rf"] - q["lf"] <= BAND["rf_lf"] + SOFT
    assert q["lf"] > 0.15
