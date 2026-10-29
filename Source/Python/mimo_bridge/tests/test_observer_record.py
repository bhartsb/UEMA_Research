# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The developmental observer's recorders change nothing the learner sees or learns.

Addendum v4 Part 6 (docs/gain_of_function_v4/OBSERVER_DESIGN.md): the observer measures
development from the simulator's own state and never feeds the learner. Pinned here:

  * a world recording for the observer replies to the learner with the same bytes, and ends in
    the same full world state, as the same world not recording (same seed, same commands), in
    the one-world source and in the worker pool the trainer uses;
  * a learner trained in closed loop while everything records (world recorder, model lane,
    per-body errors) ends with bit-identical weights, optimizer moments, losses and orders;
  * a recorder failure switches off only the recorder; the world keeps stepping;
  * the saccade flag (motor interface audit F6): a spontaneous saccade that starts on a burst's first
    frame is flagged, and a paused recorder never flags a saccade it did not watch;
  * an episode save does not carry the observer's model attributes;
  * no runtime module imports the analysis package (mimo_bridge.tools.observer), and the
    trainer reads nothing the observer writes;
  * what the recorder writes: header, chunk arrays, provenance, contact rows, gaze rays, and since
    observer_world_v2 (2026-09-25) contact normals, the inner foveal rings, a 32-bit fast lane and the
    order each world was handed - a world driven by orders replies with the same bytes and ends in the
    same state while they are recorded, and the recorded orders are the ones handed (the small-world
    checks of the same fields are in test_observer_record_v2.py).
"""
import glob
import json
import os
import re
import time as _time
from pathlib import Path

import numpy as np
import pytest
import torch

from mimo_bridge import datasource as DS
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.env import merge_burst_contact_report
from mimo_bridge.lem_model import LemLite
from mimo_bridge.observer_model_lane import ModelLane, SENSES, read_model_chunk
from mimo_bridge.observer_record import (AUTHOR_CODES, KIND_NAMES, OBSERVER_WORLD_VERSION,
                                         read_chunk)

PKG = Path(__file__).resolve().parents[1]           # Source/Python/mimo_bridge
WALL = slice(24, 32)        # tokenizer.config HDR_WALL_CLOCK: the frame's wall-clock double
REPS = 5                    # frames per model step (--action-repeat 5)


class _FixedClock:
    """datasource's time module with a fixed wall clock, so two runs compare byte for byte."""
    def __getattr__(self, name):
        return getattr(_time, name)

    @staticmethod
    def time():
        return 1000.0


def _same(a, b, path="state", _seen=None):
    """Exact equality of nested snapshots (plain objects compared field by field); returns the
    first differing path or None."""
    _seen = set() if _seen is None else _seen
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return path + " (keys)"
        for k in a:
            d = _same(a[k], b[k], f"{path}.{k}", _seen)
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return path + " (length)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _same(x, y, f"{path}[{i}]", _seen)
            if d:
                return d
        return None
    if isinstance(a, np.random.Generator):
        return _same(a.bit_generator.state, b.bit_generator.state, path, _seen)
    if isinstance(a, np.random.RandomState):
        return _same(a.get_state(legacy=False), b.get_state(legacy=False), path, _seen)
    if (hasattr(a, "__dict__") and type(a) is type(b) and not isinstance(a, type)
            and not callable(a)):
        if (id(a), id(b)) in _seen:
            return None
        _seen.add((id(a), id(b)))
        return _same(vars(a), vars(b), path, _seen)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or x.dtype != y.dtype:
            return path + " (shape/dtype)"
        if x.dtype.kind in "fc":
            return None if np.array_equal(x, y, equal_nan=True) else path
        return None if np.array_equal(x, y) else path
    if torch.is_tensor(a):
        return None if (torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype
                        and torch.equal(a, b)) else path
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return None
    try:
        return None if a == b else path
    except Exception:
        return path


def _nsteps(path):
    with np.load(path) as z:
        return len(z["step"])


def _payload_key(p):
    """A payload with its wall clock (the only field that reads the host clock) zeroed."""
    q = dict(p)
    fb = bytearray(q["frame_bytes"])
    fb[WALL] = bytes(8)
    q["frame_bytes"] = bytes(fb)
    return q


def _ctrl_plan(nu, steps, seed=0):
    """Random actuator commands in [-1, 1], held for a whole model step, with still stretches,
    so the body moves, touches the floor and itself, and stops."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, nu) if (s // 4) % 2 == 0 else np.zeros(nu) for s in range(steps)]


def _observe(step, rep, latched, order_seq, author="random_actions"):
    return dict(step=step, rep=rep, reps=REPS, author=author, latched=latched,
                order_seq=order_seq, rest=False, bout=True)


def _crib(tmp_path, name, **kw):
    return SingleEnvSource(**dict(dict(scene="crib", vision=False, age=24, run_dir=str(tmp_path / name),
                                       tag="_" + name, seed=3), **kw))


# ------------------------------------------------------------------ the one-world source
def _drive_one_world(tmp_path, name, record, steps=14, **opts):
    """Drive one crib world with held random commands; returns the learner-facing payloads, the
    final full world state, the recorder, and the world's own qpos / infant body positions at
    every model step's last frame (what the record must reproduce)."""
    src = _crib(tmp_path, name)
    payloads, truth = [], []
    try:
        if record:
            dirs = src.observer_open(str(tmp_path / name / "observer"), run=name,
                                     **dict(dict(chunk_steps=5, images_every_s=0.0, extra=dict(test=True)),
                                            **opts))
            assert dirs == [str(tmp_path / name / "observer" / "b0")]
        src.reset()
        H = src.n_handles
        motor = np.zeros((1, H, 9), np.float32)
        motor[..., 6] = 1.0
        own = sorted(src.env._own_body_ids)
        import mujoco
        scratch = mujoco.MjData(src.env.model)      # kinematics of the stored state, off the live world
        for step, ctrl in enumerate(_ctrl_plan(src.env.model.nu, steps)):
            for rep in range(REPS):
                src.step(motor, None, None, ctrl=ctrl, render=False,
                         observe=_observe(step, rep, rep == 0, step) if record else None)
                payloads.append(_payload_key(src._last_payload))
            d = src.env.data
            # mj_step leaves xpos at its last evaluated substep, one physics timestep behind qpos,
            # so the truth for the stored qpos is its own forward kinematics in the live model
            scratch.qpos[:] = d.qpos
            scratch.mocap_pos[:] = d.mocap_pos
            scratch.mocap_quat[:] = d.mocap_quat
            mujoco.mj_kinematics(src.env.model, scratch)
            truth.append(dict(qpos=np.array(d.qpos), xpos=np.array(scratch.xpos[own]),
                              mocap=np.array(d.mocap_pos), ctrl=np.array(d.ctrl)))
        state = src.env.save_full_state()
        rec = src.observer
        if record:
            assert rec is not None and rec.enabled, getattr(rec, "error", None)
    finally:
        src.close()
    return payloads, state, rec, truth, own


def test_a_recording_world_replies_with_the_same_bytes_and_ends_in_the_same_state(tmp_path, monkeypatch):
    monkeypatch.setattr(DS, "time", _FixedClock())
    off_p, off_s, _, _, _ = _drive_one_world(tmp_path, "off", record=False)
    on_p, on_s, rec, _, _ = _drive_one_world(tmp_path, "on", record=True)
    assert len(off_p) == len(on_p) == 14 * REPS
    for i, (a, b) in enumerate(zip(off_p, on_p)):
        d = _same(a, b, f"payload[{i}]")
        assert d is None, f"the recorder changed what the learner receives: {d}"
    d = _same(off_s, on_s)
    assert d is None, f"the recorder changed the world's saved state: {d}"
    assert rec.records == 14, "one record per model step"


def test_what_the_recorder_writes(tmp_path, monkeypatch):
    import mujoco
    monkeypatch.setattr(DS, "time", _FixedClock())
    _, _, rec, truth, own = _drive_one_world(tmp_path, "w", record=True, steps=14)
    folder = Path(rec.dir)
    chunks = sorted(folder.glob("chunk_*.npz"))
    assert len(chunks) == 3, "5-step chunks: 5 + 5 + the 4 flushed at close"
    headers = sorted(folder.glob("header_*.json"))
    assert len(headers) == 1
    hdr = json.loads(headers[0].read_text(encoding="utf-8"))
    assert hdr["version"] == OBSERVER_WORLD_VERSION and hdr["body"] == 0
    assert hdr["kind_names"] == KIND_NAMES and hdr["author_codes"] == AUTHOR_CODES
    ents = hdr["entities"]
    kinds = {e["kind"] for e in ents}
    assert {"self", "object", "floor", "wall", "art"} <= kinds
    groups = {e.get("group") for e in ents if e["kind"] == "self"}
    for g in ("trunk", "head", "hand_L_palm", "hand_R_palm", "foot_L", "foot_R", "arm_L_upper",
              "leg_R_lower", "finger_L_thumb", "finger_R_index", "finger_R_little"):
        assert g in groups, g
    assert "other" not in groups, [e["name"] for e in ents if e.get("group") == "other"]
    art = [e for e in ents if e["kind"] == "art"]
    assert sorted(e["panel"] for e in art) == ["e", "n", "s", "w"]
    assert all(len(e["motifs"]) in (6, 8) for e in art)
    assert hdr["weight_n"] > 0 and hdr["env_dt_s"] > 0 and hdr["nu"] > 0
    assert len(hdr["joints"]) > 50, "the infant's hinges, each with its range"
    z = [read_chunk(str(c)) for c in chunks]
    step = np.concatenate([c["step"] for c in z])
    assert step.tolist() == list(range(14))
    # the stored body state is the world's own, to half a storage quantum
    qpos = np.concatenate([c["qpos"] for c in z])
    true_q = np.stack([t["qpos"] for t in truth])
    assert qpos.shape == true_q.shape == (14, hdr["nq"])
    assert np.abs(qpos - true_q).max() <= 0.51e-6
    mocap = np.concatenate([c["mocap"] for c in z])
    assert np.abs(mocap[:, :, :3] - np.stack([t["mocap"] for t in truth])).max() <= 0.51e-6
    ctrl = np.concatenate([c["ctrl"] for c in z])
    assert np.abs(ctrl - np.stack([t["ctrl"] for t in truth])).max() <= 2e-3, "ctrl kept as float16"
    # the saved model plus the stored qpos and mocap place every infant body where the
    # simulator had it: the analysis's forward kinematics is the simulator's
    mjb = folder.parent / hdr["model_file"]
    assert mjb.is_file() and mjb.name == hdr["model_fingerprint"][:32] + ".mjb.gz"
    from mimo_bridge.observer_record import load_world_model
    m = load_world_model(str(mjb))
    d = mujoco.MjData(m)
    worst = 0.0
    for i in range(14):
        d.qpos[:] = qpos[i]
        if m.nmocap:
            d.mocap_pos[:] = mocap[i, :, :3]
            d.mocap_quat[:] = mocap[i, :, 3:]
        mujoco.mj_kinematics(m, d)
        worst = max(worst, float(np.abs(d.xpos[own] - truth[i]["xpos"]).max()))
    assert worst < 1e-5, f"forward kinematics from the record misplaces a body by {worst} m"
    for c in z:
        n = len(c["step"])
        assert c["header_name"] == headers[0].name and c["header"]["body"] == 0
        assert c["object_pose"].shape == (n, len(hdr["objects"]), 7)
        assert c["object_flags"].shape == (n, len(hdr["objects"]))
        assert c["ray_ent"].shape == (n, 2, 1 + hdr["ring_rays"])
        assert c["fast"].shape == (n, 5, 8)
        assert (c["frames"] == REPS).all()
        assert (c["author"] == AUTHOR_CODES["random_actions"]).all()
        assert c["latched"].all(), "a burst whose first frame latched an order is a latched step"
        assert c["bout"].all() and not c["rest"].any()
        assert (c["order_seq"] == c["step"]).all(), "order 0 is order 0, not 'unknown'"
        assert np.isfinite(c["fast"][:, :, :REPS]).all() and np.isnan(c["fast"][:, :, REPS:]).all()
        ok = c["ray_ent"][c["ray_ent"] >= 0]
        assert ok.size and ok.max() < len(ents)
        if len(c["c_row"]):
            assert c["c_row"].max() < n and (c["c_e"] < len(ents)).all()
            assert (c["c_fmax"] >= c["c_fmean"] - 1e-6).all() and (c["c_fmean"] >= 0).all()
            assert (c["c_frame"][:, 0] <= c["c_frame"][:, 1]).all() and (c["c_frame"] < REPS).all()
    pairs = np.concatenate([c["c_e"] for c in z])
    kinds_seen = {(ents[int(a)]["kind"], ents[int(b)]["kind"]) for a, b in pairs}
    assert any("floor" in k and "self" in k for k in kinds_seen), "the lying body rests on the floor"
    # the body's weight is carried: total support force over a still stretch is near body weight
    still = np.concatenate([c["fast"][:, 2, :REPS] for c in z])[-8:]
    assert 0.5 * hdr["weight_n"] < float(np.median(still)) < 2.0 * hdr["weight_n"]
    status = json.loads(next(folder.glob("status_*.json")).read_text(encoding="utf-8"))
    assert status["records"] == 14 and status["dropped"] == 0 and status["written_bytes"] > 0
    assert status["ms_per_record"] is not None and status["ms_per_record"] < 50
    # observer_world_v2 (2026-09-25): every field group written, each contact row's normal a unit
    # vector, the inner rings beside the v1 bundle, the fast lane 32-bit
    assert set(hdr["fields"]) == {"contact_normals", "inner_rings", "orders", "fast_float32"}
    for c in z:
        n = len(c["step"])
        assert c["fields"] == set(hdr["fields"]) and c["fast"].dtype == np.float32
        assert c["ray_in_ent"].shape == (n, 2, 16) and c["ray_in_dist"].shape == (n, 2, 16)
        if len(c["c_row"]):
            lengths = np.linalg.norm(c["c_normal"], axis=1)
            assert np.all(np.abs(lengths - 1.0) < 2e-3), "a force-weighted normal of unit length per row"
        # raw commands drove this world: no motor order is kept, and no eye command was handed
        assert not c["order_on"].any() and np.isnan(c["order_gaze"]).all()


def _drive_with_orders(tmp_path, name, record, steps=12):
    """One crib world driven by ORDERS and an eye command, the way production drives it (no raw
    actuator commands): a new order every 5 steps, each handle asked for where it is moved up to 2 cm
    at a random orientation and strength up to 0.3. Returns the learner-facing payloads, the final
    full world state, the recorder and the (motor, gaze) handed at each step."""
    import math
    src = _crib(tmp_path, name)
    payloads, handed = [], []
    try:
        if record:
            src.observer_open(str(tmp_path / name / "observer"), run=name, chunk_steps=5, images_every_s=0.0)
        src.reset()
        H = src.n_handles
        rng = np.random.default_rng(1)
        motor = None
        for step in range(steps):
            if step % 5 == 0:
                here = 100.0 * np.asarray(src.env.data.xpos[np.asarray(src.env.proprio.body_ids)], dtype=np.float64)
                motor = np.zeros((1, H, 9), np.float32)
                motor[0, :, 0:3] = here + rng.uniform(-2.0, 2.0, (H, 3))
                q = rng.normal(size=(H, 4))
                motor[0, :, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
                motor[0, :, 7] = 1.0
                motor[0, :, 8] = rng.uniform(0.0, 0.3, H)
            gaze = np.array([[0.05 * math.sin(step), 0.02]], np.float32)
            handed.append((motor.copy(), gaze.copy()))
            for rep in range(REPS):
                src.step(motor, gaze, None, render=False,
                         observe=_observe(step, rep, rep == 0, step // 5, "planner") if record else None)
                payloads.append(_payload_key(src._last_payload))
        state = src.env.save_full_state()
        rec = src.observer
    finally:
        src.close()
    return payloads, state, rec, handed


def test_a_world_driven_by_orders_replies_the_same_while_its_orders_are_recorded(tmp_path, monkeypatch):
    """observer_world_v2 records the order each world is handed (the yoked baseline replays them):
    the recording world still replies with the same bytes and ends in the same saved state, and the
    recorded orders and eye commands are exactly the ones handed."""
    from mimo_bridge.observer_record import orders_in_force
    monkeypatch.setattr(DS, "time", _FixedClock())
    off_p, off_s, _, _ = _drive_with_orders(tmp_path, "ord_off", record=False)
    on_p, on_s, rec, handed = _drive_with_orders(tmp_path, "ord_on", record=True)
    assert len(off_p) == len(on_p) == 12 * REPS
    for i, (a, b) in enumerate(zip(off_p, on_p)):
        d = _same(a, b, f"payload[{i}]")
        assert d is None, f"recording the orders changed what the learner receives: {d}"
    d = _same(off_s, on_s)
    assert d is None, f"recording the orders changed the world's saved state: {d}"
    z = [read_chunk(str(c)) for c in sorted(Path(rec.dir).glob("chunk_*.npz"))]
    assert np.concatenate([c["step"] for c in z]).tolist() == list(range(12))
    for c in z:
        of = orders_in_force(c)
        for i, s in enumerate(c["step"]):
            motor, gaze = handed[int(s)]
            assert of["index"][i] >= 0, "an order drove every step"
            np.testing.assert_array_equal(of["motor"][of["index"][i]], motor[0])
            np.testing.assert_array_equal(c["order_gaze"][i], gaze[0])
        assert (c["order_mask"] == -1).all(), "no handle mask was handed"
    # one row per change: the orders changed at steps 0, 5 and 10, the first records of the three
    # 5-step chunks
    assert [len(c["order_row"]) for c in z] == [1, 1, 1]


def test_the_disk_budget_drops_steps_and_lists_them(tmp_path, monkeypatch):
    """Over its budget, or with the disk below the reserve, the recorder writes no chunk and
    names the dropped steps; the world never notices."""
    monkeypatch.setattr(DS, "time", _FixedClock())
    _, _, rec, _, _ = _drive_one_world(tmp_path, "budget", record=True, steps=12, budget_bytes=1)
    folder = Path(rec.dir)
    assert not list(folder.glob("chunk_*.npz"))
    gaps = [json.loads(line) for line in (folder / "gaps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(g["first_step"], g["last_step"], g["rows"]) for g in gaps] == [(0, 4, 5), (5, 9, 5), (10, 11, 2)]
    assert "budget" in gaps[0]["reason"] and rec.dropped == 12
    _, _, rec2, _, _ = _drive_one_world(tmp_path, "reserve", record=True, steps=5,
                                        reserve_bytes=2 ** 62)
    gaps = (Path(rec2.dir) / "gaps.jsonl").read_text(encoding="utf-8")
    assert "never takes the last" in gaps and not list(Path(rec2.dir).glob("chunk_*.npz"))


def test_a_recorder_failure_turns_off_only_the_recorder(tmp_path, monkeypatch):
    src = _crib(tmp_path, "fail")
    try:
        src.observer_open(str(tmp_path / "fail" / "observer"), run="fail", chunk_steps=5)
        src.reset()

        def boom(*a, **k):
            raise RuntimeError("a deliberate recorder fault")
        monkeypatch.setattr(src.observer, "_frame_accumulate", boom)
        motor = np.zeros((1, src.n_handles, 9), np.float32)
        motor[..., 6] = 1.0
        for rep in range(REPS):
            frames = src.step(motor, None, None, render=False, observe=_observe(0, rep, rep == 0, 0))
        assert len(frames) == 1
        assert not src.observer.enabled
        err = json.loads((Path(src.observer.dir) / "recorder_error.json").read_text(encoding="utf-8"))
        assert "a deliberate recorder fault" in err["reason"]
        src.step(motor, None, None, render=False, observe=_observe(1, 0, True, 1))   # still steps
    finally:
        src.close()


# ------------------------------------------------------------------ the saccade flag (audit F6)
def _drive_saccades(tmp_path, name, steps, force=(), paused=()):
    """One recording crib world. force: (step, rep) frames on which the world's own saccade
    generator is made to start a saccade (its next switch moved to the frame about to step, so the
    start goes through gaze_reflex.SpontaneousSaccades.apply as any other); paused: steps during
    which the recorder is switched off (the worker's gate, datasource._observe_frame, then never
    calls it). Returns the recorded gaze_saccade per recorded step, the truth per step (the
    generator's count rose during that step's burst) and whether the start came on the first frame."""
    src = _crib(tmp_path, name)
    truth, first = {}, {}
    try:
        src.observer_open(str(tmp_path / name / "observer"), run=name, chunk_steps=5, images_every_s=0.0)
        src.reset()
        sac = src.env.saccades
        assert sac is not None, "the crib world's saccade generator is on by default"
        motor = np.zeros((1, src.n_handles, 9), np.float32)
        motor[..., 6] = 1.0
        for step in range(steps):
            src.observer.enabled = step not in paused
            n0 = sac.n_saccades
            for rep in range(REPS):
                if (step, rep) in force:
                    sac._next_switch = src.env.frame
                src.step(motor, None, None, render=False, observe=_observe(step, rep, rep == 0, step))
                if rep == 0:
                    first[step] = sac.n_saccades != n0
            truth[step] = sac.n_saccades != n0
        rec = src.observer
    finally:
        src.close()
    recorded = {}
    for c in sorted(Path(rec.dir).glob("chunk_*.npz")):
        z = read_chunk(str(c))
        for s, flag in zip(z["step"], z["gaze_saccade"]):
            recorded[int(s)] = bool(flag)
    return recorded, truth, first


def test_a_saccade_that_starts_on_a_bursts_first_frame_is_flagged(tmp_path):
    """The burst's first frame has already stepped when the recorder first reads the generator's
    count, so reading the baseline there missed every start on that frame - 1 in 5 starts, the hold
    between starts being drawn uniformly (motor interface audit, F6). The baseline is now the count
    at the end of the previous burst (or at the reset, where the generator always starts one)."""
    recorded, truth, first = _drive_saccades(tmp_path, "sac_first", 20, force={(5, 0), (13, 0)})
    assert first[5] and first[13], "the forced starts came on their bursts' first frames"
    assert first[0], "a reset world starts a saccade on its very first frame"
    assert sorted(recorded) == list(range(20))
    assert recorded == truth, {s: (recorded[s], truth[s]) for s in truth if recorded[s] != truth[s]}
    assert recorded[0] and recorded[5] and recorded[13]


def test_a_paused_recorder_never_flags_a_saccade_it_did_not_watch(tmp_path):
    """Switched off over steps 3-5 while a saccade starts in step 4, switched on again at step 6: the
    first burst after the pause must not be flagged for the start it never saw (a baseline kept from
    before the pause would flag it). The records of the watched steps still match the truth."""
    recorded, truth, first = _drive_saccades(tmp_path, "sac_paused", 9, force={(4, 2)}, paused={3, 4, 5})
    assert truth[4] and not first[4], "the start fell inside the paused steps"
    assert sorted(recorded) == [0, 1, 2, 6, 7, 8]
    assert not truth[6] and recorded[6] is False
    assert all(recorded[s] == truth[s] for s in recorded)


# ------------------------------------------------------------------ the worker pool
def test_the_worker_pool_replies_with_the_same_bytes_while_recording(tmp_path, monkeypatch):
    """The trainer's path: spawned workers, the recorder started by its own command and fed by
    the step tuple's ninth element, recording AFTER the reply is sent."""
    from mimo_bridge.datasource import PoolSource
    seen = []
    real = DS._payload_to_frame

    def keep(p, i):
        seen.append((i, _payload_key(p)))
        return real(p, i)
    monkeypatch.setattr(DS, "_payload_to_frame", keep)

    def run(name, record):
        seen.clear()
        pool = PoolSource(2, dict(scene="crib", vision=False, age=24, run_dir=str(tmp_path / name),
                                  tag="_" + name, seed=11))
        try:
            if record:
                dirs = pool.observer_open(str(tmp_path / name / "observer"), run=name, chunk_steps=3)
                assert [Path(d).name for d in dirs] == ["b0", "b1"]
            pool.reset()
            ext = seen[0][1]["extended_handles"]
            motor = np.zeros((2, DS.N_HANDLES if ext is None else len(ext), 9), np.float32)
            motor[..., 6] = 1.0                       # identity orders; the ctrl drives the body
            for step, ctrl in enumerate(_ctrl_plan(pool.info[0]["nu"], 6, seed=2)):
                for rep in range(REPS):
                    pool.step(motor, None, None, ctrl=np.stack([ctrl, -ctrl]), render=False,
                              observe=_observe(step, rep, rep == 0, step) if record else None)
            if record:
                pool.observer_close()
            state = pool.save_full_state()
        finally:
            pool.close()
        return list(seen), state

    off, off_state = run("pool_off", False)
    on, on_state = run("pool_on", True)
    assert len(off) == len(on) == 2 + 2 * 6 * REPS
    for k, (a, b) in enumerate(zip(off, on)):
        d = _same(a, b, f"reply[{k}]")
        assert d is None, f"the recorder changed a worker's reply: {d}"
    for i in range(2):
        d = _same(off_state[i]["world"], on_state[i]["world"], f"world[{i}]")
        assert d is None, f"the recorder changed a worker's world: {d}"
        chunks = sorted((tmp_path / "pool_on" / "observer" / f"b{i}").glob("chunk_*.npz"))
        assert sum(_nsteps(c) for c in chunks) == 6


# ------------------------------------------------------------------ the learner
def _learner():
    torch.manual_seed(0)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   contact_head=True, efference_heads=True, action_tenure_loss=True,
                   overrides={"training": {"steps_per_phase": 3000}})


def _closed_loop(tmp_path, name, record, steps=12):
    """Two bodies, the trainer's order: step the learner, latch an order every 5 steps, run a
    5-frame burst per body, merge the burst's contact report into the frame, repeat."""
    torch.set_num_threads(1)
    m = _learner()
    lane = None
    srcs = [_crib(tmp_path, f"{name}{i}", seed=5 + i) for i in range(2)]
    if record:
        for i, s in enumerate(srcs):
            s.observer_open(str(tmp_path / name / "observer" / "world" / f"src{i}"), run=name, chunk_steps=4)
        lane = ModelLane(str(tmp_path / name / "observer" / "model"), run=name, n_envs=2,
                         handle_names=list(m.handle_names), chunk_steps=4)
        m.observer_per_body_errors = True
    out = dict(loss=[], motor=[], gaze=[])
    try:
        frames = [s.reset()[0] for s in srcs]
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        prev = held = None
        order_seq = 0
        for i in range(steps):
            r = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            out["loss"].append(float(r.loss))
            out["gaze"].append(np.asarray(r.gaze_out).copy())
            latched = held is None or i % 5 == 0
            if latched:
                held = r.motor_for_action.copy()
                order_seq += 1
            out["motor"].append(held.copy())
            if lane is not None:
                lane.add(i, m, author=getattr(m, "last_action_source", None) or "unknown",
                         latched=latched, order_seq=order_seq, rest=False, bout=True)
            new = []
            for bi, s in enumerate(srcs):
                tasks, f = [], None
                for rep in range(REPS):
                    f = s.step(held[bi:bi + 1], np.asarray(r.gaze_out)[bi:bi + 1], None, render=False,
                               observe=(_observe(i, rep, latched and rep == 0, order_seq, "planner")
                                        if record else None))[0]
                    tasks.append(dict(f.meta.get("task") or {}))
                f.meta["task"] = {**(f.meta.get("task") or {}), **merge_burst_contact_report(tasks)}
                new.append(f)
            frames = new
            prev = LemLite.gaze_to_handle(held, np.asarray(r.gaze_out))
    finally:
        for s in srcs:
            s.close()
        if lane is not None:
            lane.close()
    out["params"] = [p.detach().clone() for p in m.params]
    out["opt"] = m.optimizer.state_dict()
    return m, out, lane


def _compare_learning(a, b):
    if a["loss"] != b["loss"]:
        return "loss"
    for k in ("motor", "gaze"):
        for x, y in zip(a[k], b[k]):
            if not np.array_equal(x, y):
                return k
    if len(a["params"]) != len(b["params"]):
        return "parameter count"
    for j, (x, y) in enumerate(zip(a["params"], b["params"])):
        if not torch.equal(x, y):
            return f"parameter {j}"
    return _same(a["opt"], b["opt"], "optimizer")


def test_learning_is_bit_identical_while_the_observer_records(tmp_path, monkeypatch):
    monkeypatch.setattr(DS, "time", _FixedClock())
    _, off, _ = _closed_loop(tmp_path, "off", record=False)
    m, on, lane = _closed_loop(tmp_path, "on", record=True)
    d = _compare_learning(off, on)
    if d is not None:
        _, again, _ = _closed_loop(tmp_path, "off_again", record=False)
        control = _compare_learning(off, again)
        pytest.fail(f"with the observer recording, learning differs at {d}; the same run without it "
                    f"twice {'also differs at ' + control + ' (nondeterminism)' if control else 'agrees'}")
    # the observer's own lane was not vacuous
    assert lane.error is None and not lane.enabled
    chunks = sorted((tmp_path / "on" / "observer" / "model").glob("chunk_*.npz"))
    assert chunks
    z = [read_model_chunk(str(c)) for c in chunks]
    assert np.concatenate([c["step"] for c in z]).tolist() == list(range(12))
    per = np.concatenate([c["next_pe_body"] for c in z])
    assert per.shape == (12, 2, len(SENSES))
    assert np.isfinite(per[1:, :, SENSES.index("proprioception")]).all(), \
        "the per-body next-moment error for every step after the first"
    mean = np.concatenate([c["next_pe"] for c in z])
    j = SENSES.index("proprioception")
    rows = np.isfinite(per[:, 0, j])
    assert np.allclose(per[rows, :, j].mean(1), mean[rows, j], rtol=1e-4, atol=1e-7), \
        "the per-body errors average to the batch error the lesson reads"
    assert any("contact_q" in c for c in z), "the contact head's probabilities were copied"
    assert any("eff_exp_cm" in c for c in z), "an order's expected movement was copied"
    well = np.concatenate([c["wellness"] for c in z])
    assert well.shape[1:] == (len(z[0]["wellness_names"]), 2) and np.isfinite(well[1:]).any(),         "each grounded wellness channel per body"
    # the world lane recorded both bodies for every step, with the learner's orders (observer_world_v2)
    for i in range(2):
        wz = sorted((tmp_path / "on" / "observer" / "world" / f"src{i}" / "b0").glob("chunk_*.npz"))
        assert sum(_nsteps(c) for c in wz) == 12
        chunks = [read_chunk(str(c)) for c in wz]
        assert all(c["order_on"].all() for c in chunks), "the learner's orders drove every step and were kept"
        assert all(np.isfinite(c["order_gaze"]).all() for c in chunks), "with the learner's eye command"


def test_an_episode_save_does_not_carry_the_observer_attributes():
    from mimo_bridge import continuation_state as CS
    m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True)
    m.observer_per_body_errors = True
    m.observer_moment = {"step": 1, "senses": ["vision"], "next_pe_per_body": np.zeros((1, 2))}
    m.observer_contact = {"step": 1}
    data = CS.capture_learner(m, include_replay=False)["data"]
    assert not [k for k in data if k.startswith("observer")]


# ------------------------------------------------------------------ the import boundary
def test_no_runtime_module_imports_the_analysis_and_the_trainer_reads_no_observer_file():
    runtime = [p for p in PKG.rglob("*.py")
               if "tests" not in p.parts and "tools" not in p.parts and "explainer" not in p.parts]
    assert len(runtime) > 20
    pattern = re.compile(r"^\s*(?:from|import)\s+[\w.]*tools(?:\.observer\b|\s+import\s+[^\n]*\bobserver\b)",
                         flags=re.M)
    for p in runtime:
        text = p.read_text(encoding="utf-8", errors="replace")
        imports = pattern.findall(text)
        assert not imports, (p.name, imports)
    assert pattern.findall("from .tools.observer import x\nfrom .tools import observer\n"
                           "import mimo_bridge.tools.observer\nhelp='mimo_bridge.tools.observer'\n"
                           "from .tools import commit_guard\n") == [
        "from .tools.observer", "from .tools import observer", "import mimo_bridge.tools.observer"]
    trainer = (PKG / "train_mimo.py").read_text(encoding="utf-8")
    # the trainer's only observer calls: start the recorders, add a lane row, close the lane
    calls = set(re.findall(r"observer_\w+\.\w+\(|src\.observer_\w+\(", trainer))
    assert calls <= {"observer_lane.add(", "observer_lane.close(", "src.observer_open("}, calls
    # the readers exist for the analysis; nothing on the training path calls them, and the two
    # writers never open a file for reading
    for name in ("train_mimo.py", "datasource.py", "lem_model.py", "env.py"):
        text = (PKG / name).read_text(encoding="utf-8")
        assert not re.search(r"\b(read_chunk|read_model_chunk|read_header)\b", text), name
    import inspect
    from mimo_bridge.observer_record import WorldRecorder
    own_outputs = ('open(os.path.join(self.dir, "gaps.jsonl"), "a"',
                   'open(os.path.join(folder, stem + ".json"), "w"', 'open(tmp, "wb")')
    for cls in (WorldRecorder, ModelLane):
        src = inspect.getsource(cls)
        for ok in own_outputs:
            src = src.replace(ok, "")
        assert "np.load(" not in src and "json.load(" not in src and "open(" not in src, \
            f"{cls.__name__} opens a file other than its own outputs"
    lem = (PKG / "lem_model.py").read_text(encoding="utf-8")
    reads = [line for line in lem.splitlines()
             if re.search(r"self\.observer_(moment|contact)\b", line) and "= {" not in line
             and not line.strip().startswith("#") and "self.observer_moment: Optional" not in line
             and "self.observer_contact: Optional" not in line]
    assert not reads, f"the learner reads its observer copies: {reads}"
