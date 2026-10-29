# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The world recorder's second format (observer_world_v2, 2026-09-25) on a small purpose-built world.

No crib world is built here (a few megabytes, no torch): the crib-world proofs that a recording world
replies with the same bytes, ends in the same saved state and leaves learning bit-identical stay in
test_observer_record.py, which a training pause runs. Pinned here:

  * contact normals: a thumb and a finger squeezing a box from opposite sides are recorded pushing
    against each other (each row's normal points from its first entity, the hand part, into the box);
  * the inner rings (2 and 6 degrees) are cast like the centre ray: a 5 cm toy 37 cm away takes the
    2-degree ring and none of the 6- or 12-degree rays; the v1 columns keep their shape;
  * the fast lane is 32-bit: a force above float16's 65,504 is stored as itself, not infinity;
  * the order handed at a burst's first frame is stored once per change and read back per record,
    with nothing kept where raw actuator commands drove the body or a scheduled rest replaced it; the
    caller's arrays are never written;
  * v1 chunks still read: a track built from them says no normals and no inner rings were recorded,
    and R04 measures nothing there instead of zero, where the v2 chunk of a box hanging between the
    digits, held up by their squeeze alone, is a grip (detectors._grip);
  * the rebuild used for measuring (tools/observer/reconstruct.py) gives a recorded chunk's own normals
    and inner rays back from its stored state.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mimo_bridge import observer_record as OR
from mimo_bridge.observer_record import WorldRecorder, orders_in_force, read_chunk

from observer_tiny_world import world as _world


def _run(tmp_path, steps=40, squeeze=0.05, orders=None, rest_steps=(), ctrl_steps=(), lifted=False):
    """Step the tiny world: the two digits close on the box (their targets squeeze metres inward from
    where they start); record. lifted: the box hangs between the digits, off the floor."""
    mujoco, m, d, env = _world(lifted)
    rec = WorldRecorder(str(tmp_path / "observer" / "world"), body=0, run="tiny", chunk_steps=100,
                        budget_bytes=2 ** 30, reserve_bytes=0)
    rec.mark("reset", env)
    d.ctrl[:] = [squeeze, -squeeze]
    for s in range(steps):
        for rep in range(5):
            for _ in range(5):
                mujoco.mj_step(m, d)
            env.frame += 1
            obs = dict(step=s, rep=rep, reps=5, author="planner", latched=rep == 0, order_seq=s // 5,
                       bout=True, rest=s in rest_steps)
            order = None if orders is None else orders(s, s in ctrl_steps)
            rec.on_frame(env, None, obs, order)
    return rec, m, env


def _chunk(rec):
    rec.close()
    chunks = sorted(Path(rec.dir).glob("chunk_*.npz"))
    assert chunks
    return read_chunk(str(chunks[0]))


def test_a_squeeze_is_recorded_as_two_normals_pushing_against_each_other(tmp_path):
    rec, m, _ = _run(tmp_path)
    z = _chunk(rec)
    h = z["header"]
    assert h["version"] == "observer_world_v2" and set(OR.FIELDS) <= set(h["fields"]) == z["fields"]
    ents = {e["name"]: e["code"] for e in h["entities"]}
    box, thumb, index, floor = ents["box_light"], ents["right_thdistal"], ents["right_ffdistal"], ents["floor"]
    last = len(z["step"]) - 1
    rows = {tuple(int(x) for x in z["c_e"][j]): z["c_normal"][j] for j in np.nonzero(z["c_row"] == last)[0]}
    assert (thumb, box) in rows and (index, box) in rows, rows.keys()
    n_th, n_ix = rows[(thumb, box)], rows[(index, box)]
    assert np.linalg.norm(n_th) == pytest.approx(1.0, abs=2e-3)
    # the thumb pushes the box toward +y (toward the index), the index toward -y
    assert n_th[1] > 0.99 and n_ix[1] < -0.99
    assert math.degrees(math.acos(float(np.clip(np.dot(n_th, n_ix), -1, 1)))) > 170.0
    # the box rests on the floor: the floor's code comes after the box's (static geoms are coded last),
    # so the pair's normal points from the box down into the floor
    assert rows[(box, floor)][2] < -0.99


def test_the_inner_rings_see_a_small_toy_the_outer_ring_misses(tmp_path):
    rec, m, _ = _run(tmp_path, steps=2)
    z = _chunk(rec)
    h = z["header"]
    assert h["inner_rings_deg"] == [2.0, 6.0] and h["inner_ring_rays"] == 8 and h["ring_half_angle_deg"] == 12.0
    toy = next(e["code"] for e in h["entities"] if e["name"] == "target")
    assert z["ray_ent"].shape == (2, 2, 9) and z["ray_in_ent"].shape == (2, 2, 16)
    for eye in (0, 1):
        assert z["ray_ent"][0, eye, 0] == toy, "the centre ray lands on the toy"
        assert (z["ray_ent"][0, eye, 1:] != toy).all(), "the 12-degree ring misses a 5 cm toy at 37 cm"
        # the 2-degree ring, 1.3 cm off the axis at 37 cm, lies on the toy's 5 cm face...
        assert (z["ray_in_ent"][0, eye, :8] == toy).all()
        # ...and the 6-degree ring, 3.9 cm off (2.75 cm on each axis at the diagonals), beside it
        assert (z["ray_in_ent"][0, eye, 8:] != toy).all()
        assert np.all(np.abs(z["ray_in_dist"][0, eye, :8] - 0.37) < 0.02)


def test_the_fast_lane_stores_a_force_past_the_16_bit_limit(tmp_path):
    rec, m, _ = _run(tmp_path, steps=3)
    rec._rows[-1]["fast"][3, 0] = 1.0e6              # a fingertip pushed into the head's sphere
    z = _chunk(rec)
    assert z["fast"].dtype == np.float32
    # the three force lanes (this world has no head sensors, so the first two lanes are empty)
    assert z["fast"][-1, 3, 0] == 1.0e6 and np.isfinite(z["fast"][:, 2:, :5]).all()


def test_the_order_handed_at_a_bursts_first_frame_is_recorded_once_per_change(tmp_path):
    H = 2
    given = {}

    def orders(s, ctrl_driven):
        motor = np.zeros((H, 9), dtype=np.float32)
        motor[:, 0] = 10.0 * (s // 5)                # a new order every 5 steps
        motor[:, 6] = 1.0
        motor[:, 8] = 0.2
        gaze = np.array([0.01 * s, -0.02], dtype=np.float32)
        mask = np.array([True, s % 2 == 0])
        goal = np.arange(4, dtype=np.float32) + s // 5
        given[s] = (motor.copy(), gaze.copy(), mask.copy(), goal.copy())
        return (motor, gaze, mask, np.zeros(2) if ctrl_driven else None, goal)
    rec, _, _ = _run(tmp_path, steps=20, orders=orders, rest_steps={7}, ctrl_steps={12})
    z = _chunk(rec)
    of = orders_in_force(z)
    assert of is not None and len(of["index"]) == 20
    for s in range(20):
        if s in (7, 12):
            assert of["index"][s] == -1, "no order kept where a rest hold or raw commands drove the body"
        else:
            k = of["index"][s]
            np.testing.assert_array_equal(of["motor"][k], given[s][0])
            np.testing.assert_array_equal(of["mask"][k], given[s][2].astype(np.int8))
            np.testing.assert_array_equal(of["hand_goal"][k], given[s][3])
        np.testing.assert_array_equal(z["order_gaze"][s], given[s][1])
    # one row per change: the order changes every 5 steps and the mask every step here, and the
    # records after the rest and the raw-command step start a row again
    assert len(z["order_row"]) == 18
    # the caller's arrays were never written
    for s, (motor, gaze, mask, goal) in given.items():
        assert motor[0, 0] == 10.0 * (s // 5) and goal[0] == s // 5


def test_an_order_with_unset_targets_held_over_its_steps_is_one_row():
    """A NaN target (an order leaving a position unset) equals itself for the change rows: the same
    order held for 5 records is one row, the next order another."""
    motor = np.zeros((2, 9), dtype=np.float32)
    motor[:, 0:3] = np.nan
    motor[:, 6] = 1.0
    later = motor.copy()
    later[:, 8] = 0.2
    orders = [WorldRecorder.take_order((m, [0.0, 0.0], None, None, None), {})
              for m in [motor] * 5 + [later] * 5]
    cols = OR.order_columns(orders, 2)
    assert cols["order_row"].tolist() == [0, 5]
    assert np.isnan(cols["order_motor"][:, :, 0:3]).all() and cols["order_motor"][1, 0, 8] == np.float32(0.2)


def test_take_order_copies_and_keeps_only_what_drove_the_body():
    motor = np.ones((3, 9))
    got = WorldRecorder.take_order((motor, [0.1, 0.2], None, None, None), dict(rest=False))
    got["motor"][0, 0] = 7.0
    assert motor[0, 0] == 1.0 and got["mask"] is None and got["hand_goal"] is None
    assert WorldRecorder.take_order((motor, None, None, np.zeros(3), None), {})["motor"] is None
    assert WorldRecorder.take_order((motor, [0.3, 0.4], None, None, None), dict(rest=True)) == dict(
        motor=None, gaze=pytest.approx(np.array([0.3, 0.4], dtype=np.float32)))
    assert WorldRecorder.take_order(None, {}) is None


def _v1(z):
    """A v2 chunk as the v1 recorder wrote it: no v2 columns, no fields in the header."""
    v2_columns = ("c_normal", "ray_in_ent", "ray_in_dist", "ray_in_cell", "order_on", "order_gaze",
                  "order_row", "order_motor", "order_mask", "order_hand_goal")
    out = {k: v for k, v in z.items() if k not in v2_columns}
    out["header"] = {k: v for k, v in z["header"].items() if k != "fields"}
    out["fields"] = OR.recorded_fields(out["header"])
    return out


def test_a_v1_chunk_reads_and_measures_no_opposition_rather_than_zero(tmp_path):
    """The box hangs between the thumb and the index, held up by their squeeze alone: on the v2 chunk
    the hand grips it (R04: the pushes straight against each other and balanced, nothing else under
    it, no slip); the same chunk read as v1 measures nothing."""
    from mimo_bridge.tools.observer import detectors as D
    from mimo_bridge.tools.observer import primitives as P
    from mimo_bridge.tools.observer.features import Track
    rec, m, _ = _run(tmp_path, steps=30, lifted=True)
    z = _chunk(rec)
    kin = _TinyKin()
    v2 = Track([z], kin)
    v1 = Track([_v1(z)], kin)
    assert v2.has_normals and v2.has_inner_rings and np.isfinite(v2.c_normal).all()
    assert not v1.has_normals and not v1.has_inner_rings and np.isnan(v1.c_normal).all()
    assert v1.ray_in_ent is None
    ents = {e["name"]: e["code"] for e in z["header"]["entities"]}
    box, floor = ents["box_light"], ents["floor"]
    assert not ((z["c_e"] == box).any(axis=1) & (z["c_e"] == floor).any(axis=1)).any(), "the box never lands"
    r04 = D.R04(D.Context(v2, P.contacts(v2)))
    assert [d["cap"] for d in r04] == ["R04"] and r04[0]["measures"]["opposition_deg"] > 170.0
    assert r04[0]["measures"]["others_weight_share"] == 0.0 and r04[0]["grip_pair"] == "thumb-fingers"
    assert D.R04(D.Context(v1, P.contacts(v1))) == []


class _TinyKin:
    """What Track._measure reads from Kinematics, for a world without a hand or a head's face."""
    head = None

    def curl_deg(self, q):
        return np.zeros(q.shape[:-1] + (2, 5))

    def wrist_deg(self, q):
        return np.zeros(q.shape[:-1] + (2, 3))

    def thumb_swivel_deg(self, q):
        return np.zeros(q.shape[:-1] + (2,))

    def eye_deg(self, q):
        return np.zeros(q.shape[:-1] + (2, 2))

    def place(self, *a, **k):
        pass

    def measure(self):
        z3 = np.zeros(3)
        return dict(tips=np.zeros((2, 5, 3)), palm=np.zeros((2, 3)), palm_normal=np.zeros((2, 3)),
                    forearm=np.zeros((2, 3)), head_c=z3, head_face=z3, head_up=z3, mouth=z3,
                    head_R=np.eye(3), eye_pos=np.zeros((2, 3)), eye_fwd=np.tile([1.0, 0, 0], (2, 1)),
                    roll_deg=0.0, sit_deg=90.0, trunk_pos=z3, belly=z3, trunk_lat=z3, trunk_R=np.eye(3),
                    shoulder_z=np.zeros(2), com=z3, root_pos=z3, xpos=np.zeros((8, 3)),
                    xquat=np.tile([1.0, 0, 0, 0], (8, 1)))


def test_the_rebuild_for_measuring_gives_a_recordings_own_normals_and_inner_rays(tmp_path):
    from mimo_bridge.tools.observer.reconstruct import Reconstructor
    rec, m, _ = _run(tmp_path, steps=30)
    z = _chunk(rec)
    got = Reconstructor(z["header"], str(Path(rec.dir).parent)).fill(_v1(z))
    assert got["reconstructed"] and {"contact_normals", "inner_rings"} <= got["fields"]
    same = got["agreement"]["rays_same"]
    assert same == got["agreement"]["rays"], "the rebuilt centre ray and 12-degree ring are the stored ones"
    np.testing.assert_array_equal(got["ray_in_ent"], z["ray_in_ent"])
    last = len(z["step"]) - 1
    j = np.nonzero(z["c_row"] == last)[0]
    ok = np.isfinite(got["c_normal"][j]).all(axis=1)
    assert ok.any()
    # a still squeeze: the rebuilt normal of the last record is the recorded one (float16 storage)
    assert np.abs(got["c_normal"][j][ok] - z["c_normal"][j][ok]).max() < 2e-3
