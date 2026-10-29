# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The attention critters and the ring, live in a real crib env.

One env build covers: the ring camera exists and renders; the dragonfly hovers then
darts and its angle is logged; the bee roams with flickering wings; the squeaker sits
past the feet and hops when struck (driven here by teleporting it onto a foot - the
contingency contract is contact -> hop, however the contact arrives).
"""
import numpy as np
import pytest

from mimo_bridge.env import BridgeEnv


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = BridgeEnv(scene="crib", run_dir=str(tmp_path_factory.mktemp("critters")),
                  vision=True, eye_size=256, fovea_size=288, fovea_fov=24.0,
                  render_every=2, seed=7, tag="_crt", mobile=True,
                  ring_fov=150.0, ring_size=128, critters=True)
    e.reset()
    yield e
    e.close()


def test_ring_camera_renders(env):
    # the ring shares the wide renderer (no extra GL context): eye_size px over 150 deg
    img = env.render_ring()
    assert img is not None and img.shape == (256, 256, 3)
    assert env.reflex.ring_rows == 16


def test_critter_bodies_exist_and_log(env):
    for _ in range(30):
        raw = env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=True)
    task = raw.info.get("task", {})
    assert task.get("critter_visitor") == "bee", "the bee visits first"
    assert "bee_deg" in task and "squeaker_hits" in task
    assert "dragonfly_deg" not in task, "the absent visitor logs no angle"


def test_bee_lands_on_flowers_and_goes_silent(env):
    cr = env._cr
    assert cr["flowers"], "the bouquet must offer flowers"
    saw_landed = saw_flight = False
    for _ in range(1200):   # 12 s: fly to a flower (~3 s), land 2-4 s
        env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=False)
        if cr["bee_state"] == "landed":
            saw_landed = True
            p = np.asarray(env.data.mocap_pos[cr["bee_id"]])
            fl = cr["flowers"][cr["bee_flower_i"]]
            assert np.linalg.norm(p - fl) < 0.03, "landed means ON the flower"
        else:
            saw_flight = True
        if saw_landed and saw_flight:
            break
    assert saw_landed, "the bee must land on a flower within 12 s"


def test_dragonfly_patrols_fast_then_hovers(env):
    cr = env._cr
    # force the dragonfly on stage for this test
    env._critter_show("bee", False)
    env._critter_show("dragonfly", True)
    cr["visitor"] = "dragonfly"
    cr["visitor_until"] = env.frame + 10**9
    cr["df_prev"] = None
    darts0 = cr["df_darts"]
    saw_hover = saw_patrol = False
    for _ in range(900):    # 9 s: laps take ~2 s, lookout pauses 1.5-3.5 s
        env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=False)
        st = cr["df_state"]
        saw_hover = saw_hover or st == "hover"
        saw_patrol = saw_patrol or st == "patrol"
        if cr["df_darts"] > darts0 and saw_hover and saw_patrol:
            break
    assert cr["df_darts"] > darts0 and saw_hover and saw_patrol


def test_bee_moves_and_wings_flicker(env):
    cr = env._cr
    env._critter_show("dragonfly", False)
    env._critter_show("bee", True)
    cr["visitor"] = "bee"
    cr["visitor_until"] = env.frame + 10**9
    cr["bee_state"], cr["bee_prev"] = "roam", None
    cr["bee_anchor"] = env._critter_waypoint(cr["rng"])
    cr["bee_state_until"] = env.frame + 10**9      # keep it flying for this test
    env.data.mocap_pos[cr["bee_id"]] = cr["bee_anchor"]
    p0 = np.array(env.data.mocap_pos[env._cr["bee_id"]])
    alphas = set()
    for _ in range(120):
        env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=False)
        alphas.add(round(float(env.model.geom_rgba[env._cr["bee_wings"][0]][3]), 2))
    p1 = np.array(env.data.mocap_pos[env._cr["bee_id"]])
    assert np.linalg.norm(p1 - p0) > 0.025, \
        "bee should loop (roam circles a fixed anchor at radius 0.035 now)"
    assert len(alphas) >= 2, "wing alpha should flicker between two values"


def test_squeaker_hops_on_contact(env):
    cr = env._cr
    hits0 = cr["sq_hits"]
    # drive the panel onto the left foot: contact is contact, however it arrives
    foot = np.array(env.data.xpos[env.model.body("left_foot").id])
    cr["sq_home"] = foot.copy()
    cr["sq_home"][2] = max(0.02, foot[2])
    cr["sq_cool_until"] = -1
    hopped = False
    for _ in range(120):
        env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=False)
        if cr["sq_hits"] > hits0:
            hopped = True
            break
    assert hopped, "squeaker must register a body contact"
    task = env.step(None, None, None, ctrl=np.zeros(env.model.nu), render=False).info["task"]
    assert task["squeaker_hits"] >= hits0 + 1


def test_second_reset_keeps_the_visitor_rotation_alive(env):
    """The bee-lock defect (2026-09-06): _critters_reset computed visitor_until from
    self.frame BEFORE reset zeroed it, so from episode 2 onward the deadline sat past
    the end of every episode and the scheduler never fired - bee forever, dragonfly
    gone (10 of 306 live records). Every frame-keyed critter deadline must be
    reachable within an episode after ANY reset, not just the first."""
    still = np.zeros((24, 9)); still[:, 6] = 1.0
    for _ in range(120):                      # a few real steps first
        env.step(still, None, None, render=False)
    env.frame = 10_000                        # simulate a full episode's frame count -
                                              # the defect only bites when the old count
                                              # is large (120 frames would pass broken code)
    env.reset()                               # episode 2 - where the defect lived
    assert env.frame == 0
    cr = env._cr
    horizon = int(45.0 / env.dt)              # a visitor window is 25-40 s
    assert cr["visitor_until"] <= horizon, \
        f"bee deadline {cr['visitor_until']} frames is beyond any window - bee-locked"
    assert cr.get("df_until", 0) <= horizon
    # and the rotation actually reaches the dragonfly: jump to the deadline and gap
    for _ in range(3):
        env.frame = max(env.frame, int(cr["visitor_until"]))
        env.step(still, None, None, render=False)
        if cr["visitor"] == "dragonfly":
            break
    assert cr["visitor"] == "dragonfly", f"rotation stuck on {cr['visitor']}"
