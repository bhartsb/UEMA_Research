# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The lift-head-and-look milestone must be measurable and honest: at rest the
squeaker sits far outside the gaze cone; tilting the head toward the feet must
close the angle, and the seen-flag plus the one-shot milestone event must fire
exactly when the view is earned. Signs are DISCOVERED, never assumed.
"""
import numpy as np
import pytest

from mimo_bridge.env import BridgeEnv


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = BridgeEnv(scene="crib", run_dir=str(tmp_path_factory.mktemp("headlook")),
                  vision=True, eye_size=256, fovea_size=288, fovea_fov=24.0,
                  render_every=2, seed=44, tag="_hl", mobile=True, critters=True)
    e.reset()
    yield e
    e.close()


def test_squeaker_starts_unseen_and_head_tilt_earns_the_view(env):
    r = env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=True)
    t0 = r.info["task"]
    assert t0.get("squeaker_view_deg", 0) > 45.0, \
        "a supine baby must NOT see the pedal at rest"
    assert t0.get("squeaker_seen") is False
    j = env.model.joint("robot:head_tilt").id
    qa = int(env.model.jnt_qposadr[j])
    lo, hi = env.model.jnt_range[j]
    base = t0["squeaker_view_deg"]
    best = (base, 0.0)
    for tilt in (lo, hi):                      # discover which direction is feet-ward
        env.data.qpos[qa] = tilt
        env._mj.mj_forward(env.model, env.data)
        r = env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=True)
        v = r.info["task"]["squeaker_view_deg"]
        if v < best[0]:
            best = (v, float(tilt))
        env.data.qpos[qa] = 0.0
        env._mj.mj_forward(env.model, env.data)
    assert best[0] < base - 15.0, \
        f"full head tilt must close the view angle by 15+ deg ({base:.0f} -> {best[0]:.0f})"
    # the flag and the one-shot milestone follow the geometry exactly
    env.data.qpos[qa] = best[1]
    env._mj.mj_forward(env.model, env.data)
    r = env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=True)
    t1 = r.info["task"]
    assert t1["squeaker_seen"] == (t1["squeaker_view_deg"] < 30.0)
    if t1["squeaker_seen"]:
        assert t1["headlook_events"] >= 1, "the first look must count as a milestone"
    env.data.qpos[qa] = 0.0
    env._mj.mj_forward(env.model, env.data)
