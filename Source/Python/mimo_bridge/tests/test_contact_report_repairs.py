# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What a contact reports: its direction, how many pairs there are, and how far a hand is.

Three separate readings were wrong in ways no test could see, each found by reading the code
against what it claims. The unit here is metres and the frame matters in every one of them.
"""
import numpy as np
import pytest

from mimo_bridge.touch_adapter import ContactInfo, PatchSpec, TouchPooler


def _contact(body_id=1, other="right_hand", force=1.0, world=(0.0, 0.0, 1.0),
             body=None, pair=(1, 2), is_self=True, depth=0.0, speed=0.0):
    return ContactInfo(body_id=body_id, pair_key=pair, normal_force=force,
                       tangential_body=np.zeros(3), pos_body=np.zeros(3),
                       pos_world=np.zeros(3),
                       normal_world=np.asarray(world, dtype=np.float64),
                       penetration_m=depth, rel_normal_speed=speed,
                       other_body_name=other, other_is_self=is_self,
                       normal_body=None if body is None else np.asarray(body, dtype=np.float64))


DT = 0.01


def _pooler(n=2):
    specs = [PatchSpec(body_id=i + 1, body_name="p%d" % i, n_points=1, scale_m=0.01)
             for i in range(n)]
    return TouchPooler(specs, DT)


def _forces(n=2):
    """One sensing point per patch, pressed with one newton straight in."""
    return {i + 1: np.array([[0.0, 0.0, 1.0]]) for i in range(n)}


# ------------------------------------------------------------------ the push direction
def test_the_push_direction_is_given_in_the_body_frame_that_the_rest_of_the_row_uses():
    """The first three numbers of a contact row are the contact's place measured from the
    body. The push direction written beside them was measured from the world, so one and the
    same self-touch gave a direction that changed as the trunk rolled, and every floor contact
    read straight up whatever the body was doing."""
    p = _pooler()
    # the same push, seen from a body that has rolled a quarter turn: the world says up, the
    # body says sideways
    p.update(_forces(), [_contact(body_id=1, world=(0.0, 0.0, 1.0), body=(0.0, 1.0, 0.0))])
    assert np.allclose(p.last_contact[0, :3], [0.0, 1.0, 0.0]), (
        "the push direction was written in the world's frame: %s" % p.last_contact[0, :3])


def test_a_contact_with_no_body_frame_direction_falls_back_to_the_world_one():
    """A caller that has not computed the body-frame direction still gets a direction rather
    than three zeros, and nothing reads a zero vector as a push pointing nowhere."""
    p = _pooler()
    p.update(_forces(), [_contact(body_id=1, world=(0.0, 0.0, 1.0), body=None)])
    assert np.allclose(p.last_contact[0, :3], [0.0, 0.0, 1.0])


# ------------------------------------------------------------------ how many pairs
def test_one_self_touch_is_one_pair_however_many_surfaces_feel_it():
    """Both surfaces of a self-touch sense it, so the environment builds one record per side
    and each side used to add the other's name to its own patch. Summing the patches counted
    every pair twice: the live run reported 2.60 pairs against 2.45 patches."""
    p = _pooler()
    # one contact between two of the body's own parts, sensed from both sides
    both = [_contact(body_id=1, other="p1", pair=(7, 9)),
            _contact(body_id=2, other="p0", pair=(7, 9))]
    p.update(_forces(), both)
    assert p.last_self_pairs == 1, (
        "one contact felt from two sides counted as %d pairs" % p.last_self_pairs)


def test_two_separate_self_touches_are_two_pairs():
    p = _pooler()
    p.update(_forces(), [_contact(body_id=1, other="p1", pair=(7, 9)),
                         _contact(body_id=2, other="p0", pair=(7, 9)),
                         _contact(body_id=1, other="p1", pair=(3, 4)),
                         _contact(body_id=2, other="p0", pair=(3, 4))])
    assert p.last_self_pairs == 2


def test_a_contact_with_the_world_is_not_a_self_pair():
    p = _pooler()
    p.update(_forces(), [_contact(body_id=1, other="floor", pair=(7, 9), is_self=False)])
    assert p.last_self_pairs == 0
    assert float(p.last_self[0, 0]) == 0.0, "a floor contact counted as the body's own"


# ------------------------------------------------------------------ how far the hand is
@pytest.mark.slow
def test_how_far_a_hand_is_from_the_mobile_is_measured_surface_to_surface():
    """The gauge subtracted each shape's BOUNDING SPHERE radius from the distance between
    centres. A bounding sphere holds the whole shape, so that number can only under-report the
    real gap, and for the mobile it under-reports enormously: the string is a capsule 4 mm
    thick and 0.60 m long whose bounding sphere has a radius of 0.304 m. On the live segment
    24 of 43 windows reported a closest approach of exactly 0.000 m and not one of those
    windows recorded a hand touching the mobile.

    This builds the crib scene, takes both readings on the same frames, and requires the gauge
    to agree with MuJoCo's own surface-to-surface distance rather than with the bound."""
    import mujoco
    import os
    import tempfile
    from mimo_bridge.env import BridgeEnv
    out = os.path.join(tempfile.gettempdir(), "mimo_reach_gauge_test")
    os.makedirs(out, exist_ok=True)
    env = BridgeEnv(scene="crib", run_dir=out, age=24.0, vision=False, mobile=True,
                    critters=False, caregiver_mode="off", body_layout="trunk_v1",
                    contact_profile="mechanical_v4", seed=0, tag="_gaugetest")
    env.reset()
    m, d = env.model, env.data
    hg, mg = list(env._hand_geoms), sorted(env._mobile_geoms)
    # the defect's own cause, pinned: one shape of the mobile has a bounding sphere far
    # larger than the shape inside it
    assert max(float(m.geom_rbound[g]) for g in mg) > 0.25
    rng = np.random.default_rng(0)
    for _ in range(25):
        env.step(ctrl=np.clip(0.35 * rng.standard_normal(m.nu), -1.0, 1.0))
    # take the gauge straight off a step's own task block
    raw = env.step(ctrl=np.zeros(m.nu))
    gap = float(raw.info["task"]["mobile_hand_gap_m"])
    truth = min(float(mujoco.mj_geomDistance(m, d, int(a), int(b), 2.0, None))
                for a in hg for b in mg)
    bound = min(float(np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b])
                      - m.geom_rbound[a] - m.geom_rbound[b]) for a in hg for b in mg)
    assert gap == pytest.approx(max(0.0, truth), abs=1e-6), (
        "the gauge reads %.4f against a true surface distance of %.4f" % (gap, truth))
    assert truth > bound, (
        "this frame cannot tell the two apart, so it does not test the repair")
