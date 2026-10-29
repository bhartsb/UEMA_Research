# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The observer's baseline controls of 2026-09-25 (tools/observer/baseline.py; OBSERVER_DESIGN.md
section 9): random actuator commands from production's saved worlds with their own vision, the yoked
controls (the learner's own recorded orders replayed in shuffled time order, and in their recorded order
from a random first order), re-recording from a newer checkpoint beside the old, and summaries that
name what their recording cannot measure.

No crib world is built (fakes, and the small world of observer_tiny_world.py); the real recordings run
in a training pause (docs/gain_of_function_v4/pause_jobs_observer_recorder_2026-09-25.json).
"""
import json
import math
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge.observer_record import WorldRecorder
from mimo_bridge.tools.observer import baseline as B
from mimo_bridge.tools.observer import daemon as DM

from observer_tiny_world import world as tiny_world
from test_baseline_saved_worlds import MANIFEST, SIG, _FakeEnv, _FakeRecorder, _fake_checkpoint, _summary


def _rot(axis, deg):
    """A unit quaternion (w, x, y, z) turning deg about axis."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2
    return np.r_[math.cos(h), math.sin(h) * a]


def _mat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------ random actuator commands, saved worlds
def test_random_ctrl_starts_from_the_saved_world_and_drives_raw_commands(tmp_path, monkeypatch):
    ck, _, prod, run = _fake_checkpoint(tmp_path)
    plan = B.saved_world_plan(B.extract_saved_worlds(str(ck), str(tmp_path / "w")), str(run), "like", 2)
    monkeypatch.setattr("mimo_bridge.env.BridgeEnv", _FakeEnv)
    monkeypatch.setattr("mimo_bridge.observer_record.WorldRecorder", _FakeRecorder)
    out = B.record_one(str(run), "random_ctrl", 0, 10, str(tmp_path / "stage"), 1000, "like", plan[0])
    env, rec = _FakeEnv.made[-1], _FakeRecorder.made[-1]
    assert env.resets == 0 and rec.marks == ["restore"] and out["restored"]["ok"]
    assert rec.kw["extra"]["start"] == "saved_worlds" and rec.kw["extra"]["vision"] is True
    # raw commands in [-1, 1], one draw held for the motor hold (5 steps of 5 frames), no motor order
    assert all(s["motor"] is None and s["ctrl"] is not None for s in env.stepped)
    assert np.abs(np.stack([s["ctrl"] for s in env.stepped])).max() <= 1.0
    assert np.array_equal(env.stepped[0]["ctrl"], env.stepped[24]["ctrl"])
    assert not np.array_equal(env.stepped[0]["ctrl"], env.stepped[25]["ctrl"])
    # the eyes rendered as training renders them, and the recorder given each frame's commands
    assert [r for _, r in env.steps[:5]] == [False] * 4 + [True]
    assert all(o[0] is None and o[3] is not None for o in rec.orders)
    assert not prod.exists()


def test_random_orders_hand_the_recorder_their_orders(tmp_path, monkeypatch):
    ck, _, _, run = _fake_checkpoint(tmp_path)
    plan = B.saved_world_plan(B.extract_saved_worlds(str(ck), str(tmp_path / "w")), str(run), "like", 2)
    monkeypatch.setattr("mimo_bridge.env.BridgeEnv", _FakeEnv)
    monkeypatch.setattr("mimo_bridge.observer_record.WorldRecorder", _FakeRecorder)
    B.record_one(str(run), "random_orders", 1, 5, str(tmp_path / "stage"), 1001, "like", plan[1])
    env, rec = _FakeEnv.made[-1], _FakeRecorder.made[-1]
    for s, o in zip(env.stepped, rec.orders):
        assert o[3] is None and np.array_equal(o[0], s["motor"])


# ------------------------------------------------------------------ the yoke: re-anchoring
def test_an_order_re_anchored_asks_for_the_same_movement_from_another_pose():
    rng = np.random.default_rng(0)
    H = 4
    pos0 = rng.normal(0, 0.3, (H, 3))
    q0 = np.stack([_rot(rng.normal(size=3), rng.uniform(0, 180)) for _ in range(H)])
    motor = np.zeros((H, 9))
    motor[:, 0:3] = 100.0 * pos0 + rng.normal(0, 3.0, (H, 3))          # a few centimetres away
    qt = np.stack([_rot(rng.normal(size=3), rng.uniform(0, 90)) for _ in range(H)])
    motor[:, 3:7] = qt[:, [1, 2, 3, 0]]                                  # x, y, z, w as the motor carries it
    motor[:, 7], motor[:, 8] = 1.0, rng.uniform(0, 0.3, H)
    motor[3] = [5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.1]            # the eyes' row, kept as recorded
    rel = B.relative_order(motor, pos0, q0, keep_rows=[3])
    # at the pose it was issued from, the order comes back as it was
    back = B.anchor_order(rel, pos0, q0)
    assert np.allclose(back[:, 0:3], motor[:, 0:3]) and np.allclose(back[:, 7:], motor[:, 7:])
    for h in range(3):
        assert abs(abs(np.dot(back[h, 3:7], motor[h, 3:7])) - 1.0) < 1e-12
    # from another pose: the same displacement (cm) and the same turn (world frame)
    pos1 = pos0 + rng.normal(0, 0.2, (H, 3))
    q1 = np.stack([_rot(rng.normal(size=3), rng.uniform(0, 180)) for _ in range(H)])
    new = B.anchor_order(rel, pos1, q1)
    for h in range(3):
        assert np.allclose(new[h, 0:3] - 100.0 * pos1[h], motor[h, 0:3] - 100.0 * pos0[h])
        turn_then = _mat(qt[h]) @ _mat(q0[h]).T
        turn_now = _mat(np.r_[new[h, 6], new[h, 3:6]]) @ _mat(q1[h]).T
        assert np.allclose(turn_now, turn_then, atol=1e-9)
    assert np.array_equal(new[3], motor[3]), "the eyes' row is replayed as recorded"
    # a missing target stays missing
    motor2 = motor.copy()
    motor2[0, 0:3] = np.nan
    assert np.isnan(B.anchor_order(B.relative_order(motor2, pos0, q0), pos1, q1)[0, 0:3]).all()


# ------------------------------------------------------------------ the yoke: units from a recording
def _recorded_orders(tmp_path, steps=40, hold=5, rest=range(20, 25), probe=range(30, 35), chunk_steps=100):
    """The small world recorded with orders: a latch every `hold` steps, a scheduled rest and a probe
    in between. Returns (world folder, {step: (motor, gaze)} as handed)."""
    mujoco, m, d, env = tiny_world()
    world = tmp_path / "run" / "observer" / "world"
    rec = WorldRecorder(str(world), body=0, run="run", chunk_steps=chunk_steps, budget_bytes=2 ** 30,
                        reserve_bytes=0)
    rec.mark("reset", env)
    handed = {}
    motor = None
    scratch = mujoco.MjData(m)            # mj_step leaves xpos one timestep behind qpos: place the state
    for s in range(steps):
        scratch.qpos[:] = d.qpos
        mujoco.mj_kinematics(m, scratch)
        here = np.array(scratch.xpos[env.proprio.body_ids])
        if s % hold == 0:
            motor = np.zeros((2, 9), dtype=np.float32)
            motor[:, 0:3] = 100.0 * here.astype(np.float32) + [1.0 + s, -2.0, 0.5]
            motor[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
            motor[:, 7], motor[:, 8] = 1.0, 0.2
        gaze = np.array([0.01 * s, -0.01 * s], dtype=np.float32)
        author = "probe" if s in probe else "planner"
        handed[s] = (motor.copy(), gaze.copy(), here)
        d.ctrl[:] = [0.02 * math.sin(0.3 * s), -0.02 * math.sin(0.3 * s)]
        for rep in range(5):
            for _ in range(5):
                mujoco.mj_step(m, d)
            env.frame += 1
            rec.on_frame(env, None, dict(step=s, rep=rep, reps=5, author=author, latched=(s % hold == 0),
                                         order_seq=s // hold, bout=True, rest=s in rest),
                         (motor, gaze, None, None, None))
    rec.close()
    return str(world), handed


def test_the_yoke_takes_the_learners_own_orders_and_leaves_rest_and_probes_out(tmp_path):
    world, handed = _recorded_orders(tmp_path)
    y = B.yoke_units([world], 0)
    src = y["source"]
    # latches at 0, 5, ..., 35; step 0 has no record before it; 20 is a rest and 30 a probe
    assert [u["step"] for u in y["units"]] == [5, 10, 15, 25, 35]
    assert src["units"] == 5 and src["authors"] == {"planner": 5} and src["unit_steps"] == {"5": 5}
    assert y["handle_names"] == ["thumb", "index"]
    for u in y["units"]:
        s = u["step"]
        motor, _, _ = handed[s]
        # the displacement asked for, from the handles' pose at the record before the order began
        pos_before = handed[s][2]                 # captured as step s began: the record before it
        assert np.allclose(u["rel"]["dpos_cm"], motor[:, 0:3] - 100.0 * pos_before, atol=2e-3)
        np.testing.assert_array_equal(u["gaze"], np.stack([handed[t][1] for t in range(s, s + 5)]))
        assert u["mask"] is None and u["hand_goal"] is None and u["author"] == "planner"
    # orders at or after the checkpoint's step are left out
    assert [u["step"] for u in B.yoke_units([world], 0, before_step=16)["units"]] == [5, 10, 15]


def test_an_order_runs_on_across_a_chunk_boundary(tmp_path):
    """Chunks of 7 records cut the 5-step orders (an independent review found units of 2, 4, 5, 1, 3...
    records): an order continues into the next chunk when the header, the world episode, the steps and
    the order sequence number carry on."""
    world, handed = _recorded_orders(tmp_path, chunk_steps=7)
    assert len(list(Path(world, "b0").glob("chunk_*.npz"))) == 6
    y = B.yoke_units([world], 0)
    assert [u["step"] for u in y["units"]] == [5, 10, 15, 25, 35]
    assert y["source"]["unit_steps"] == {"5": 5}
    for u in y["units"]:
        np.testing.assert_array_equal(u["gaze"], np.stack([handed[t][1] for t in range(u["step"], u["step"] + 5)]))


def test_a_zero_quaternion_is_the_decoders_no_rotation_not_nan():
    """action_decoder.quat_xyzw_to_mat reads a zero quaternion as the identity (the world-aligned
    orientation); the yoke keeps that instead of dividing by zero."""
    H = 2
    motor = np.zeros((H, 9))
    motor[:, 0:3] = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    pos = np.zeros((H, 3))
    q_here = np.tile(_rot([0.0, 0.0, 1.0], 30.0), (H, 1))
    rel = B.relative_order(motor, pos, q_here)
    assert np.isfinite(rel["qrel"]).all()
    new = B.anchor_order(rel, pos + 0.1, q_here)
    assert np.isfinite(new).all()
    # the target is the world-aligned orientation: x, y, z, w = 0, 0, 0, +-1
    assert np.allclose(np.abs(new[:, 6]), 1.0) and np.allclose(new[:, 3:6], 0.0, atol=1e-12)


def test_the_same_variant_recorded_by_the_older_format_is_kept_aside():
    old = dict(start="saved_worlds", vision=True, checkpoint_step=397125)
    new = dict(start="saved_worlds", vision=True, checkpoint_step=397125,
               recorded_fields=["contact_normals", "fast_float32", "inner_rings", "orders"])
    assert B.aside_name("random_orders", old, new) == "random_orders_ckpt_397125_v1"
    assert B.aside_name("random_orders", dict(old, recorded_fields=new["recorded_fields"]), new) is None
    assert B.aside_name("limp", dict(vision=False), dict(vision=False, recorded_fields=["orders"])) == "limp_v1"
    assert B.aside_name("random_orders", old, dict(new, checkpoint_step=411150)) == "random_orders_ckpt_397125"


def test_the_pause_job_records_only_after_its_tests_passed_in_the_same_pause(tmp_path, monkeypatch):
    import importlib.util
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "observer_recorder_pause_job", root / "docs" / "gain_of_function_v4" / "observer_recorder_pause_job.py")
    J = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(J)
    pauses = tmp_path / "pauses"
    assert J.tests_passed_in_this_pause(str(pauses)) is None

    def receipt(name, jobs, finished=False, mtime=0.0):
        folder = pauses / name
        folder.mkdir(parents=True)
        r = dict(run="r", started=1.0, jobs=jobs, **(dict(finished=2.0) if finished else {}))
        (folder / "receipt.json").write_text(json.dumps(r), encoding="utf-8")
        os.utime(folder / "receipt.json", (mtime, mtime))
    passed = dict(name=J.TESTS_JOB, status="done", exit_code=0)
    receipt("20260925_010000", [passed], finished=True, mtime=1000.0)       # an earlier pause's pass
    assert J.tests_passed_in_this_pause(str(pauses)) is None
    receipt("20260925_020000", [dict(passed, exit_code=1)], mtime=2000.0)   # this pause: failed
    assert J.tests_passed_in_this_pause(str(pauses)) is None
    receipt("20260925_030000", [passed], mtime=3000.0)                      # this pause: passed
    assert J.tests_passed_in_this_pause(str(pauses)).endswith("receipt.json")
    monkeypatch.setattr(J, "PAUSES", str(tmp_path / "nothing"))
    with pytest.raises(SystemExit):
        J.require_tests(False)
    J.require_tests(True)                                                   # a deliberate manual run


def test_a_recording_without_orders_cannot_yoke(tmp_path):
    world, _ = _recorded_orders(tmp_path, steps=12)
    for p in Path(world, "b0").glob("header_*.json"):
        h = json.loads(p.read_text(encoding="utf-8"))
        h.pop("fields")                                   # as observer_world_v1 wrote it
        p.write_text(json.dumps(h), encoding="utf-8")
    from mimo_bridge import observer_record as OR
    OR._HEADERS.clear()
    with pytest.raises(ValueError, match="orders"):
        B.yoke_units([world], 0)


# ------------------------------------------------------------------ the yoke: replay
def test_the_yoked_replay_plays_every_order_whole_in_a_shuffled_order(monkeypatch):
    env = _FakeEnv()
    env.data.xpos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
    rec = _FakeRecorder("x", 0)
    H = 3
    units = []
    for k in range(4):
        motor = np.zeros((H, 9))
        motor[:, 0:3] = 100.0 * env.data.xpos[1:4] + [k, 0.0, 0.0]      # k cm along x from each handle
        motor[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
        motor[:, 8] = 0.1 * (k + 1)
        rel = B.relative_order(motor, env.data.xpos[1:4], env.data.xquat[1:4], keep_rows=[2])
        units.append(dict(rel=rel, mask=None, hand_goal=None, gaze=np.full((2 + k % 2, 2), float(k))))
    out = B.replay_orders(env, rec, units, steps=25, reps=5, rng=np.random.default_rng(3), render=True)
    # each order held for its own number of steps with its own eye command, strength telling which
    seen = [round(float(s["motor"][0, 8]) * 10) - 1 for s in env.stepped[::5]]
    runs = [seen[0]]
    for a, b in zip(seen, seen[1:]):
        if b != a:
            runs.append(b)
    assert sorted(runs[:4]) == [0, 1, 2, 3], "one pass plays every unit once, in a permutation"
    assert runs[:4] != [0, 1, 2, 3] or runs[4:8] != [0, 1, 2, 3]
    for s in env.stepped[::5]:
        k = round(float(s["motor"][0, 8]) * 10) - 1
        assert np.allclose(s["motor"][0, 0:3] - 100.0 * env.data.xpos[1], [k, 0.0, 0.0])
        assert np.array_equal(s["gaze"], [float(k), float(k)])
    assert out["units_replayed"] == 4 and out["passes"] >= 2 and len(env.stepped) == 25 * 5
    assert {f["author"] for f in rec.frames} == {"yoked"}
    starts = [f for f in rec.frames if f["latched"]]
    assert len(starts) == 5 * out["orders"], "every frame of each order's first step is latched"
    assert all(o[0] is not None and o[3] is None for o in rec.orders)


def test_the_sequence_yoke_plays_the_orders_in_their_recorded_order_from_a_random_first(monkeypatch):
    """yoked_sequence (the independent review's finding 6: production's consecutive orders move each hand
    back the way it came, which a shuffle breaks): the units in their recorded order, starting at a
    random one and running on through the last to the first."""
    env = _FakeEnv()
    env.data.xpos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
    rec = _FakeRecorder("x", 0)
    H, n = 3, 6
    units = []
    for k in range(n):
        motor = np.zeros((H, 9))
        motor[:, 0:3] = 100.0 * env.data.xpos[1:4]
        motor[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
        motor[:, 8] = 0.1 * (k + 1)
        rel = B.relative_order(motor, env.data.xpos[1:4], env.data.xquat[1:4], keep_rows=[2])
        units.append(dict(rel=rel, mask=None, hand_goal=None, gaze=np.zeros((2, 2))))
    out = B.replay_orders(env, rec, units, steps=30, reps=5, rng=np.random.default_rng(5), render=False,
                          keep_sequence=True)
    played = [round(float(s["motor"][0, 8]) * 10) - 1 for s in env.stepped[::10]]   # 2 steps x 5 frames each
    first = out["first_unit"]
    assert played == [(first + i) % n for i in range(15)], "recorded order, from the first unit, round again"
    assert out["units_replayed"] == n and out["passes"] == 3 and out["orders"] == 15
    firsts = {B.replay_orders(_FakeEnv(), _FakeRecorder("x", 0), units, steps=2, reps=1,
                              rng=np.random.default_rng(s), render=False, keep_sequence=True)["first_unit"]
              for s in range(40)}
    assert len(firsts) > 1, "the first unit is drawn, not always the history's first"


def test_record_one_yoked_needs_saved_worlds_and_matching_handles(tmp_path, monkeypatch):
    ck, _, _, run = _fake_checkpoint(tmp_path)
    plan = B.saved_world_plan(B.extract_saved_worlds(str(ck), str(tmp_path / "w")), str(run), "like", 2)
    monkeypatch.setattr("mimo_bridge.env.BridgeEnv", _FakeEnv)
    monkeypatch.setattr("mimo_bridge.observer_record.WorldRecorder", _FakeRecorder)
    with pytest.raises(ValueError, match="saved worlds"):
        B.record_one(str(run), "yoked", 0, 5, str(tmp_path / "stage"), 1000, "like", plan[0])
    H = 3
    motor = np.zeros((H, 9))
    motor[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    rel = B.relative_order(motor, np.zeros((H, 3)), np.tile([1.0, 0, 0, 0], (H, 1)))
    yoke = dict(units=[dict(rel=rel, mask=None, hand_goal=None, gaze=np.zeros((5, 2)))],
                handle_names=["a", "b", "other"], source=dict(units=1, worlds=["x"]))
    path = tmp_path / "yoke.pkl"
    path.write_bytes(pickle.dumps(yoke))
    with pytest.raises(ValueError, match="handles"):
        B.record_one(str(run), "yoked", 0, 5, str(tmp_path / "stage"), 1000, "like", dict(plan[0], yoke=str(path)))
    yoke["handle_names"] = ["a", "b", "gaze"]
    path.write_bytes(pickle.dumps(yoke))
    out = B.record_one(str(run), "yoked", 0, 7, str(tmp_path / "stage"), 1000, "like", dict(plan[0], yoke=str(path)))
    rec = _FakeRecorder.made[-1]
    assert out["yoke"]["units_replayed"] == 1 and out["yoke"]["passes"] == 2 and out["restored"]["ok"]
    assert rec.kw["extra"]["baseline"] == "yoked" and rec.kw["extra"]["yoke"]["method"].startswith("re-anchored")
    assert "random permutation" in rec.kw["extra"]["yoke"]["method"]
    out = B.record_one(str(run), "yoked_sequence", 0, 7, str(tmp_path / "stage2"), 1000, "like",
                       dict(plan[0], yoke=str(path)))
    rec = _FakeRecorder.made[-1]
    assert out["yoke"]["first_unit"] == 0 and rec.kw["extra"]["baseline"] == "yoked_sequence"
    assert "recorded order" in rec.kw["extra"]["yoke"]["method"]


def test_run_yoked_builds_each_bodys_orders_from_the_checkpoints_run(tmp_path, monkeypatch):
    ck, _, _, run = _fake_checkpoint(tmp_path)
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "Saved"))
    monkeypatch.setattr(B, "_extract_in_child", lambda ctx, checkpoint, folder: B.extract_saved_worlds(checkpoint, folder))
    asked = {}

    def yoke_in_child(ctx, worlds, bodies, folder, before_step):
        asked.update(worlds=worlds, bodies=bodies, before_step=before_step)
        os.makedirs(folder, exist_ok=True)
        return [dict(body=b, file=os.path.join(folder, "b%d.pkl" % b),
                     source=dict(units=10 + b, worlds=worlds)) for b in range(bodies)]
    monkeypatch.setattr(B, "_yoke_in_child", yoke_in_child)
    seen = []

    def record_all(ctx, jobs, workers):
        for j in jobs:
            seen.append(j)
            (Path(j[4]) / "observer" / "world" / ("b%d" % j[2])).mkdir(parents=True, exist_ok=True)
        return [dict(body=j[2], records=j[3], vision=True, restored=dict(ok=True),
                     yoke=dict(units_replayed=5, passes=1)) for j in jobs]
    monkeypatch.setattr(B, "_record_all", record_all)
    monkeypatch.setattr(B, "summarize", lambda out_dir, kind, vision=None: _summary({}, vision=vision))
    with pytest.raises(ValueError, match="from-checkpoint"):
        B.run(str(run), "yoked", 2, 10, 2, root=str(tmp_path / "rec"))
    path = B.run(None, "yoked", 2, 10, 2, root=str(tmp_path / "rec"), from_checkpoint=str(ck))
    assert asked == dict(worlds=[os.path.join(str(run), "observer", "world")], bodies=2, before_step=1200)
    assert [j[1] for j in seen] == ["yoked", "yoked"] and [os.path.basename(j[7]["yoke"]) for j in seen] == ["b0.pkl", "b1.pkl"]
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    assert Path(path).name == "yoked.json" and s["yoke"]["units_per_body"] == [10, 11]
    assert s["yoke"]["before_step"] == 1200 and s["yoke"]["replayed"] == [5, 5]


# ------------------------------------------------------------------ what a summary can measure
def test_a_summary_names_what_its_recording_cannot_measure(tmp_path):
    rec = tmp_path / "rec"
    world = rec / "observer" / "world"
    header = dict(scene="crib", nq=10, nu=4, objects=[dict(name="target")], patch_names=["a"], handle_names=["h"],
                  extra=dict(vision=True, baseline="random_orders"))
    for b, fields in ((0, ["contact_normals", "inner_rings", "orders", "fast_float32"]), (1, None)):
        (world / ("b%d" % b)).mkdir(parents=True)
        h = dict(header, **({"fields": fields} if fields else {}))
        (world / ("b%d" % b) / "header_1_1000.json").write_text(json.dumps(h), encoding="utf-8")
    assert B.recording_fields(str(rec)) == set(), "a group counts only when every header has it"
    un = B.unmeasured_caps(set())
    # O05 (shaking) walks the contact intervals but never reads `held`: measured on v1 as before
    assert set(un) == {"R04", "H02", "H03", "H09", "R06", "R07"} and "contact normals" in un["R04"]
    assert B.unmeasured_caps({"contact_normals"}) == {}
    analysis = rec / "observer" / "analysis"
    analysis.mkdir(parents=True)
    state = dict(version=DM.ANALYSIS_VERSION, detector_version=DM.DETECTOR_VERSION, bodies={}, cuts={},
                 sealed=1.0, analysis_id="a")
    exposure = {"0": {"1": {"0": dict(steps=100, by={"random": 100}, abs=0, abs_by={})}}}
    (analysis / "progress.json").write_text(json.dumps(dict(state=state, exposure=exposure, fast={}, sizes={})),
                                            encoding="utf-8")
    s = B.summarize(str(rec), "random_orders")
    assert "R04" not in s["caps"] and "H02" not in s["caps"] and "H05" in s["caps"]
    assert set(s["unmeasured_caps"]) == set(un) and s["recorded_fields"] == []


def test_compare_lists_a_capability_one_summary_could_not_measure(tmp_path, monkeypatch):
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "Saved"))
    final = tmp_path / "Saved" / "observer" / "baselines" / SIG
    final.mkdir(parents=True)
    new = _summary(B.source_fields(dict(MANIFEST, step=411150), 8), generated=2.0)
    new["caps"] = {"R04": dict(count=3, occasions=3), "H05": dict(count=100, occasions=50)}
    new["recorded_fields"] = ["contact_normals"]
    old = dict(_summary(B.source_fields(MANIFEST, 8)), kind="random_orders_ckpt_396050")
    old["caps"] = {"H05": dict(count=120, occasions=60)}
    old["unmeasured_caps"] = {"R04": "the recording carries no contact normals"}
    (final / "random_orders.json").write_text(json.dumps(new), encoding="utf-8")
    (final / "random_orders_ckpt_396050.json").write_text(json.dumps(old), encoding="utf-8")
    c = B.compare(SIG, "random_orders", "random_orders_ckpt_396050")
    rows = {r["cap"]: r for r in c["rows"]}
    assert rows["R04"]["measured"] == [True, False] and rows["R04"]["p_occasions"] is None
    assert "contact normals" in rows["R04"]["note"] and rows["R04"]["acts"] == [3, None]
    assert rows["H05"]["p_occasions"] is not None and c["checkpoint_steps"] == [411150, 396050]
    assert c["recorded_fields"] == [["contact_normals"], None]
