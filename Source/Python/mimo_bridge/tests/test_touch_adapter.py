# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Hand-worked tests for every derived touch number.

Worked example used below (two skin bodies, dt = 0.01 s):
  body 7 "palm": 4 points, spacing 0.01 m -> one point area = 0.0001 m2 = 100 mm2
  body 9 "tip":  3 points, spacing 0.002 m -> one point area = 0.000004 m2 = 4 mm2
  body 3 "eye": 1 point, spacing 1.0 -> NOT skin, must be dropped

Step 1 on the palm: one contact, normal force 3 N, tangential (4, 0, 0) N,
two of the four points feel force.
  area      = 2 points x 100 mm2 = 200 mm2 = 0.0002 m2
  pressure  = 3 N / 0.0002 m2 = 15000 Pa
  shear     = 4 N / 0.0002 m2 = 20000 Pa
  slip      = 0 (first time we see this contact)
Step 2: the same contact has moved 2 mm along the skin:
  slip      = 0.002 m / 0.01 s = 0.2 m/s = 200 mm/s
"""
import math

import numpy as np
import pytest

from mimo_bridge.touch_adapter import PUSH_ARRIVAL_SCALE_CMS, PUSH_DEPTH_SCALE_CM
from mimo_bridge.touch_adapter import (
    Goertzel, TouchPooler, PatchSpec, ContactInfo, build_collisions, bone_hash,
    N_PATCH_FIELDS,
)

DT = 0.01


def specs():
    return [
        PatchSpec(body_id=9, body_name="tip", n_points=3, scale_m=0.002),
        PatchSpec(body_id=7, body_name="palm", n_points=4, scale_m=0.01),
        PatchSpec(body_id=3, body_name="eye", n_points=1, scale_m=1.0),
    ]


def contact(body_id, normal, tang, pos_body, pair=(1, 2), pos_world=(0, 0, 0), n_world=(0, 0, 1),
            pen=0.0, speed=0.0):
    return ContactInfo(body_id=body_id, pair_key=pair, normal_force=normal,
                       tangential_body=np.array(tang, dtype=float), pos_body=np.array(pos_body, dtype=float),
                       pos_world=np.array(pos_world, dtype=float), normal_world=np.array(n_world, dtype=float),
                       penetration_m=pen, rel_normal_speed=speed)


def test_patch_order_and_eye_dropped():
    p = TouchPooler(specs(), DT)
    assert p.n_patches == 2
    assert [s.body_id for s in p.specs] == [7, 9]          # body id order, eye gone
    assert p.point_area_mm2.tolist() == [100.0, 4.0]


def test_pressure_shear_area_by_hand():
    p = TouchPooler(specs(), DT)
    forces = {7: np.array([[0, 0, 2.0], [0, 0, 1.0], [0, 0, 0], [0, 0, 0]]), 9: np.zeros((3, 3))}
    f = p.update(forces, [contact(7, 3.0, (4, 0, 0), (0.0, 0.0, 0.0))])
    assert f.shape == (2, N_PATCH_FIELDS)
    palm = f[0]
    assert palm[3] == pytest.approx(200.0)        # mm2
    assert palm[0] == pytest.approx(15000.0)      # Pa
    assert palm[1] == pytest.approx(20000.0)      # Pa
    assert palm[2] == 0.0                         # no slip on first sight
    assert palm[8] == 1.0 and palm[9] == 1.0      # stretch labeled rest
    assert np.all(f[1] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32))


def test_slip_by_hand():
    p = TouchPooler(specs(), DT)
    forces = {7: np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])}
    p.update(forces, [contact(7, 1.0, (0, 0, 0), (0.0, 0.0, 0.0))])
    f = p.update(forces, [contact(7, 1.0, (0, 0, 0), (0.002, 0.0, 0.0))])
    assert f[0, 2] == pytest.approx(200.0)        # mm/s
    # a different contact pair does not inherit slip
    f = p.update(forces, [contact(7, 1.0, (0, 0, 0), (0.1, 0.0, 0.0), pair=(5, 6))])
    assert f[0, 2] == 0.0


def test_no_area_means_no_pressure():
    """Force reported by a contact but no skin point active -> area 0 -> pressure 0, no NaN."""
    p = TouchPooler(specs(), DT)
    f = p.update({7: np.zeros((4, 3))}, [contact(7, 3.0, (4, 0, 0), (0, 0, 0))])
    assert f[0, 0] == 0.0 and f[0, 1] == 0.0 and f[0, 3] == 0.0
    assert not np.isnan(f).any()


def test_goertzel_by_hand():
    """A 10 Hz sine of amplitude 2 N sampled at 100 Hz for 1 s -> power 4.0 at 10 Hz, ~0 elsewhere.
    A constant 5 N -> all four bands ~0 (DC is not one of the bands)."""
    g = Goertzel((5, 10, 15, 20), dt=0.01, window=100)
    t = np.arange(100) * 0.01
    pw = g.power(2.0 * np.sin(2 * math.pi * 10.0 * t))
    assert pw[1] == pytest.approx(4.0, abs=1e-9)
    assert abs(pw[0]) < 1e-9 and abs(pw[2]) < 1e-9 and abs(pw[3]) < 1e-9
    pw = g.power(np.full(100, 5.0))
    assert np.all(np.abs(pw) < 1e-9)


def test_vibration_bands_through_pooler():
    """Push a 20 Hz wobble of amplitude 1 N of normal force for one full window -> band 20 Hz = 1.0."""
    p = TouchPooler(specs(), DT, vib_window=100)
    forces = {7: np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])}
    for i in range(100):
        n = 3.0 + 1.0 * math.sin(2 * math.pi * 20.0 * i * DT)
        f = p.update(forces, [contact(7, n, (0, 0, 0), (0, 0, 0))])
    assert f[0, 7] == pytest.approx(1.0, abs=1e-6)   # 20 Hz band
    assert abs(f[0, 4]) < 1e-6 and abs(f[0, 5]) < 1e-6 and abs(f[0, 6]) < 1e-6


def test_collisions_units_and_cap():
    p = TouchPooler(specs(), DT)
    cs = [contact(7, float(i + 1), (0, 0, 0), (0, 0, 0), pair=(i, 100), pos_world=(0.1, 0.2, 0.3),
                  n_world=(0, 0, 1), pen=0.004, speed=0.5) for i in range(20)]
    p.update({7: np.zeros((4, 3))}, cs)
    cols = build_collisions(cs, p, {7: bone_hash("left_hand")}, DT)
    assert len(cols) == 16
    assert cols[0]["bone_hash"] == bone_hash("left_hand")
    assert cols[0]["penetration_cm"] == pytest.approx(0.4)
    assert cols[0]["impact_speed_cms"] == pytest.approx(50.0)
    assert np.allclose(cols[0]["point_cm"], [10.0, 20.0, 30.0])
    # strongest first: normal 20 N -> impulse 20 N * 0.01 s = 0.2 N s along the normal
    assert np.allclose(cols[0]["impulse_ns"], [0, 0, 0.2])
    assert cols[0]["frame_count"] == 1 and cols[0]["duration_s"] == pytest.approx(0.01)


def test_too_many_skin_bodies_rejected():
    many = [PatchSpec(i, f"b{i}", 1, 0.01) for i in range(257)]
    with pytest.raises(ValueError):
        TouchPooler(many, DT)


# ---------------------------------------------------------- the body touching itself
# Nothing in the repository ever set other_is_self before 2026-09-21, so the lines that
# add up a patch's own-part force and collect which own parts are on it ran in no test -
# and they produce the two token columns the self-touch audit and the self-touch wellness
# term both read.
def self_contact(body_id, normal, other_name, is_self=True, pair=(1, 2), pen=0.0, speed=0.0,
                 n_world=(0, 0, 1), n_body=None):
    return ContactInfo(body_id=body_id, pair_key=pair, normal_force=normal,
                       tangential_body=np.zeros(3), pos_body=np.zeros(3),
                       pos_world=np.zeros(3), normal_world=np.array(n_world, dtype=float),
                       normal_body=None if n_body is None else np.array(n_body, dtype=float),
                       penetration_m=pen, rel_normal_speed=speed,
                       other_body_name=other_name, other_is_self=is_self)


def test_own_part_force_and_own_part_count_are_accumulated_per_patch():
    """The palm is pressed by one of the body's own parts at 3 N and by the floor at 1 N.

    share: 3 of the 4 N on that patch came from the body itself, so 0.75, where 0 = nothing
      on this patch is the body and 1 = everything on it is.
    parts: one different own part is on it, reported over four and capped, so 0.25, where
      0 = no own part is on this patch and 1 = four or more are.
    """
    p = TouchPooler(specs(), DT)
    forces = {7: np.array([[0, 0, 2.0], [0, 0, 1.0], [0, 0, 0], [0, 0, 0]])}
    p.update(forces, [self_contact(7, 3.0, "left_th", pair=(1, 2)),
                      self_contact(7, 1.0, "floor", is_self=False, pair=(3, 4))])
    assert p.last_self[0].tolist() == pytest.approx([0.75, 0.25])
    assert p.last_self[1].tolist() == [0.0, 0.0], "a patch nothing touched carries nothing"
    assert p.last_self_force == pytest.approx(3.0)
    assert p.last_total_force == pytest.approx(4.0)
    assert p.last_self_pairs == 1

    # a second own part arrives on the same patch
    p.update(forces, [self_contact(7, 3.0, "left_th", pair=(1, 2)),
                      self_contact(7, 2.0, "right_th", pair=(5, 6)),
                      self_contact(7, 1.0, "floor", is_self=False, pair=(3, 4))])
    assert p.last_self[0, 0] == pytest.approx(5.0 / 6.0)
    assert p.last_self[0, 1] == pytest.approx(0.5)
    assert p.last_self_pairs == 2

    # the count is over four and capped there, so a hand wrapped in five parts reads 1.0
    p.update(forces, [self_contact(7, 1.0, "own_%d" % i, pair=(10 + i, 20 + i))
                      for i in range(6)])
    assert p.last_self[0, 0] == pytest.approx(1.0), "every newton came from the body itself"
    assert p.last_self[0, 1] == pytest.approx(1.0)


def test_one_self_touch_is_one_pair_even_though_both_sides_feel_it():
    """Both surfaces of a self-touch sense it, so env.py builds one record per side. The two
    records carry the same pair of shapes, so the pair count must read 1 and not 2."""
    p = TouchPooler(specs(), DT)
    forces = {7: np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
              9: np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0]])}
    p.update(forces, [self_contact(7, 2.0, "tip", pair=(4, 8)),
                      self_contact(9, 2.0, "palm", pair=(4, 8))])
    assert p.last_self_pairs == 1
    assert p.last_self[0, 1] == pytest.approx(0.25) and p.last_self[1, 1] == pytest.approx(0.25)


def test_a_patch_with_no_force_has_no_self_share_and_raises_nothing():
    """A contact reported at zero force, and one reported negative, must not divide by zero
    or push the share past its own scale."""
    p = TouchPooler(specs(), DT)
    f = p.update({7: np.zeros((4, 3))}, [self_contact(7, 0.0, "left_th"),
                                         self_contact(7, -5.0, "right_th", pair=(7, 8))])
    assert not np.isnan(p.last_self).any() and not np.isnan(f).any()
    assert p.last_self[0, 0] == 0.0, "no force means no share of it"
    assert p.last_self[0, 1] == pytest.approx(0.5), "both own parts are still on the patch"
    assert p.last_self_force == 0.0 and p.last_total_force == 0.0


def test_the_push_columns_convert_depth_from_metres_and_arrival_from_metres_per_second():
    """The three things the felt columns never say: which way the push came from, how deep
    it is and how fast it arrived. The one existing test fed numbers that were already
    scaled, so the two conversions ran unasserted.

    depth:   metres -> centimetres -> over PUSH_DEPTH_SCALE_CM
    arrival: metres per second -> cm/s -> over PUSH_ARRIVAL_SCALE_CMS
    The expected numbers are computed from those two constants, never typed, so a change of
    divisor cannot leave this test asserting the old one.

    WHY THE DIVISORS ARE WHAT THEY ARE. They were 10 cm and 500 cm/s, the UE collision
    reader's. On H84 segment 29's snapshot at step 262,057 the deepest contact filled 0.00366
    of the depth scale and the fastest arrival 0.00018, while pressure in the same rows reached
    2.93 of its own scale. Measured on 2026-09-21 over 3,000 model steps of motor babble on the
    crib body, the 99th percentile contact pressed 0.108 cm deep and the 99th percentile arrival
    came in at 2.21 cm/s, so the divisors are now 0.1 cm and 2 cm/s, and a checkpoint trained on
    the old ones is migrated exactly on load (test_touch_push_scale.py)."""
    p = TouchPooler(specs(), DT)
    forces = {7: np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])}
    p.update(forces, [self_contact(7, 1.0, "floor", is_self=False,
                                   n_world=(0, 0, 1), n_body=(0, -1, 0),
                                   pen=0.004, speed=0.5)])
    row = p.last_contact[0]
    assert row[:3].tolist() == pytest.approx([0.0, -1.0, 0.0]), (
        "the direction is taken in the sensing body's own frame, beside pos_body")
    assert row[3] == pytest.approx(0.4 / PUSH_DEPTH_SCALE_CM), (
        "4 mm deep is 0.4 cm over the depth scale")
    assert row[4] == pytest.approx(50.0 / PUSH_ARRIVAL_SCALE_CMS), (
        "0.5 m/s is 50 cm/s over the arrival scale")
    # the world direction is the fallback when the caller has no body-frame one
    p.update(forces, [self_contact(7, 1.0, "floor", is_self=False, n_world=(1, 0, 0),
                                   pen=0.0, speed=0.0)])
    assert p.last_contact[0][:3].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert p.last_contact[0][3] == 0.0 and p.last_contact[0][4] == 0.0
    # a caller with no closing speed at all reads zero rather than raising
    p.update(forces, [ContactInfo(body_id=7, pair_key=(1, 2), normal_force=1.0,
                                  tangential_body=np.zeros(3), pos_body=np.zeros(3),
                                  pos_world=np.zeros(3), normal_world=np.array([0., 0., 1.]),
                                  penetration_m=0.01, rel_normal_speed=None)])
    assert p.last_contact[0][3] == pytest.approx(1.0 / PUSH_DEPTH_SCALE_CM)
    assert p.last_contact[0][4] == 0.0
    # the row keeps the STRONGEST contact on the patch, which is the one it is sorted by
    p.update(forces, [self_contact(7, 1.0, "a", is_self=False, pair=(1, 2),
                                   n_world=(0, 0, 1), pen=0.001, speed=0.1),
                      self_contact(7, 9.0, "b", is_self=False, pair=(3, 4),
                                   n_world=(0, 1, 0), pen=0.005, speed=0.2)])
    assert p.last_contact[0][:3].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert p.last_contact[0][3] == pytest.approx(0.5 / PUSH_DEPTH_SCALE_CM)      # 5 mm deep
    assert p.last_contact[0][4] == pytest.approx(20.0 / PUSH_ARRIVAL_SCALE_CMS)  # 0.2 m/s
    # a patch no contact reached carries no push at all
    assert p.last_contact[1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_how_much_of_the_depth_and_arrival_scales_the_live_run_actually_uses(record_property):
    """The same measurement, re-run against whatever the newest training snapshot holds, so
    how much of each scale the live run uses stays visible. Nothing is gated on it: the numbers
    are recorded, and the contract asserted is that a depth and an arrival speed are finite and
    never negative, read on the scale the snapshot's own legend names.

    NOT CAPPED AT 1. Since 2026-09-21 each divisor is the crib body's 99th percentile, so the
    99th percentile contact reads about 1 and the rare deepest and fastest read more - measured,
    3.1 and 7.0 - the way pressure in the same rows already reaches 2.93 of its own scale. A
    snapshot written before the change carries the old legend (10 cm, 500 cm/s) and reads far
    below 1."""
    import glob
    import json
    import os
    from mimo_bridge.lem_model import TOUCH_TOKEN_F, TOUCH_VALID_COL
    root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Saved",
                        "mimo_bridge", "experiments")
    rows = source = None
    for path in sorted(glob.glob(os.path.join(root, "*", "explain", "step_*.json")),
                       key=os.path.getmtime, reverse=True):
        try:
            with open(path, encoding="utf-8") as handle:
                cloud = (json.load(handle).get("senses") or {}).get("touch_cloud")
        except (OSError, ValueError):
            continue
        if not cloud:
            continue
        a = np.asarray(cloud, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != TOUCH_TOKEN_F:
            continue
        valid = a[a[:, TOUCH_VALID_COL] > 0.5]
        if len(valid):
            rows = valid
            source = os.path.basename(os.path.dirname(os.path.dirname(path)))
            break
    if rows is None:
        pytest.skip("no explainer snapshot under Saved/ carries a touch cloud of this width")
    depth, arrival = rows[:, 16], rows[:, 17]
    record_property("live_touch_rows", int(len(rows)))
    record_property("depth_share_of_its_scale_max", float(depth.max()))
    record_property("arrival_share_of_its_scale_max", float(arrival.max()))
    record_property("pressure_share_of_its_scale_max", float(rows[:, 3].max()))
    print("%s: %d valid contact rows; the deepest contact read %.5f of its depth scale and "
          "the fastest arrival %.5f of its speed scale, while pressure reached %.3f of its own"
          % (source, len(rows), depth.max(), arrival.max(), rows[:, 3].max()))
    assert np.all(np.isfinite(depth)) and np.all(np.isfinite(arrival))
    assert depth.min() >= 0.0, "a contact of negative depth: %.4f" % depth.min()
    assert arrival.min() >= 0.0, "a contact arriving at a negative speed: %.4f" % arrival.min()
