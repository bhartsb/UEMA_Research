# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The hand press split into world and self (DESIGN.md D9 and section 8, 2026-09-22).

hand_pressing fired on the body pressing ITSELF - 399 of 414 pressing steps in a
random-command probe were pure self-contact - and the consequence term paid for it. The world
now reports the same press twice more: the part pressed by something outside the body, and
the part pressed by the body's own parts. The old keys must not move.
"""
import numpy as np
import pytest

from mimo_bridge.env import HAND_PRESS_LINE_KPA, split_hand_press


def test_a_press_entirely_by_the_body_itself_is_all_self():
    out = split_hand_press([150_000.0, 500.0], [1.0, 0.0])
    assert out["hand_press_self_kpa"] == pytest.approx(150.0)
    assert out["hand_pressing_self"] is True
    assert out["hand_press_world_kpa"] == pytest.approx(0.5)
    assert out["hand_pressing_world"] is False


def test_a_press_by_the_floor_is_all_world():
    out = split_hand_press([8_000.0], [0.0])
    assert out["hand_press_world_kpa"] == pytest.approx(8.0) and out["hand_pressing_world"]
    assert out["hand_press_self_kpa"] == 0.0 and not out["hand_pressing_self"]


def test_a_patch_pressed_half_by_each_gives_half_its_pressure_to_each():
    out = split_hand_press([6_000.0], [0.5])
    assert out["hand_press_world_kpa"] == pytest.approx(3.0)
    assert out["hand_press_self_kpa"] == pytest.approx(3.0)


def test_the_split_uses_the_old_line_and_never_exceeds_the_whole_press():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform(0, 20_000, 12)
        s = rng.uniform(0, 1, 12)
        out = split_hand_press(p, s)
        whole = p.max() / 1000.0
        assert out["hand_press_world_kpa"] <= whole + 1e-9
        assert out["hand_press_self_kpa"] <= whole + 1e-9
        # the strongest patch gives at least half its pressure to one side
        assert max(out["hand_press_world_kpa"], out["hand_press_self_kpa"]) >= whole / 2 - 1e-9
        assert out["hand_pressing_world"] == (out["hand_press_world_kpa"] > HAND_PRESS_LINE_KPA)
        assert out["hand_pressing_self"] == (out["hand_press_self_kpa"] > HAND_PRESS_LINE_KPA)
    assert HAND_PRESS_LINE_KPA == 2.0, "the old hand_pressing line; changing it re-means the old key"
    assert split_hand_press([], []) == {}


@pytest.fixture(scope="module")
def crib(tmp_path_factory):
    from mimo_bridge.env import BridgeEnv
    env = BridgeEnv(scene="crib", run_dir=str(tmp_path_factory.mktemp("press")), vision=False,
                    tag="_presstest", age=24, contact_profile="mechanical_v4",
                    body_function="modulate", caregiver_seconds=0, kp=6.0,
                    body_layout="trunk_v1", cmd_horizon=25)
    env.reset()
    yield env
    env.close()


def _hand_rows(env):
    return [i for i, nm in enumerate(env.patch_names())
            if any(k in nm for k in ("hand", "distal", "middle", "knuckle", "metacarpal", "thbase", "thhub"))]


def test_the_live_world_reports_all_keys_and_the_old_ones_are_unchanged(crib):
    env = crib
    hp = _hand_rows(env)
    for _ in range(40):
        raw = env.step(ctrl=np.zeros(env.model.nu))
        t = raw.info["task"]
        for k in ("hand_press_kpa", "hand_pressing", "hand_press_world_kpa", "hand_pressing_world",
                  "hand_press_self_kpa", "hand_pressing_self"):
            assert k in t, k
        assert t["hand_pressing"] == (t["hand_press_kpa"] > 2.0), "the old key's rule moved"
        assert t["hand_press_world_kpa"] <= t["hand_press_kpa"] + 1e-9
        assert t["hand_press_self_kpa"] <= t["hand_press_kpa"] + 1e-9
        # recomputed from the pooler's own share, the split is exact
        share = np.asarray(env.pooler.last_self)[hp, 0]
        audit = raw.info.get("physical_touch_audit")
        pressure = (audit if audit is not None else raw.patches)[hp, 0]
        want = split_hand_press(pressure, share)
        assert t["hand_press_self_kpa"] == pytest.approx(want["hand_press_self_kpa"], abs=1e-6)
        assert t["hand_press_world_kpa"] == pytest.approx(want["hand_press_world_kpa"], abs=1e-6)


def test_lying_face_down_the_hands_press_the_world(crib):
    """Placed prone, the hands meet the floor: the world part of the press must appear."""
    import mujoco
    env = crib
    env.reset()
    m, d = env.model, env.data
    qa = int(m.jnt_qposadr[m.joint("mimo_orientation").id])
    ax = np.asarray(d.xpos[m.body("head").id]) - np.asarray(d.xpos[m.body("hip").id])
    ax[2] = 0.0
    ax /= np.linalg.norm(ax)
    qr = np.array([0.0, *ax])                                  # half a turn about the long axis
    w1, x1, y1, z1 = qr
    w2, x2, y2, z2 = d.qpos[qa + 3:qa + 7]
    d.qpos[qa + 3:qa + 7] = [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                             w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]
    d.qpos[qa + 2] += 0.08
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    world = []
    for _ in range(250):
        world.append(env.step(ctrl=np.zeros(m.nu)).info["task"]["hand_press_world_kpa"])
    assert max(world) > 0.0, "a prone body's hands never registered pressing the floor"
