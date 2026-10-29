# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Babble from production's saved worlds (tools/observer/baseline.py; OBSERVER_DESIGN.md section 10,
priority 4): the chance control every stage is judged against, started from a production
checkpoint's saved eight-world episode instead of a fresh world.

Pinned here without building a production world:
  * the saved episode is read as the trainer reads it on resume, including a companion whose path
    passes Windows' 260 characters, and split into one file per body that keeps every saved setting
    and state, with the caregiver's event tape detached: babble writes nothing into production's
    caregiver recording or scene folders;
  * a babble body is built from its saved settings with only the two output folders changed,
    restored (never reset), checked, and recorded with its source in the recorder's header; a world
    that is not in its saved state is not recorded; the reset-world path is unchanged;
  * the restore check passes an exact world and names each field that differs (a small MuJoCo
    model, not the crib);
  * placement: saved-world babble becomes the comparison the tracker reads (daemon.load_baseline)
    and the reset-world one moves aside; a later reset-world recording goes beside it; the vision
    set-aside works as before;
  * the summary names its source and keeps every existing field, and a re-summary keeps them;
  * a summary of a sealed digest still names its world;
  * each capability's rate from saved worlds against the reset-world one, with the test of the two
    counts, from the two summaries alone (compare);
  * the command line takes a checkpoint without a run folder.
The restore of a real production world is checked by the recording itself (restore_receipt on all
eight bodies before their first order), in the training pause that runs it.
"""
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge.tools.observer import baseline as B
from mimo_bridge.tools.observer import daemon as DM

SIG = "74302686aba93fca"
PHASES = ("absent", "engaged")


def _write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def _fake_checkpoint(tmp_path, n=2, motor_hold=5, vision=True):
    """A checkpoint folder whose episode companion (written by continuation_state.write_episode, as
    the trainer writes it) sits past 260 characters, holding n worlds whose caregivers carry event
    tapes pointing into a 'production' folder that must never be written."""
    from mimo_bridge import continuation_state as CS
    from mimo_bridge.event_tape import EventTape
    run = tmp_path / "run"
    ck = run / "checkpoints" / "ckpt_step_1200"
    ck.mkdir(parents=True)
    _write(run / "args.json", dict(action_repeat=5, motor_hold=motor_hold, no_vision=not vision))
    prod = tmp_path / "prod"
    worlds = []
    for i in range(n):
        tape = EventTape(prod / "caregiver_records", prod / "scenes" / ("caregiver_storage_%d.json" % (100 + i)))
        tape.count = 7 + i
        world = dict(qpos=np.arange(6.0) + i, qvel=np.linspace(0.0, 1.0, 5), act=None, ctrl=np.zeros(3),
                     mocap_pos=np.zeros((1, 3)), mocap_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                     time=19803.0, frame=1980250, integration_state=np.arange(4.0),
                     caregiver=dict(events=tape, scene_phase=PHASES[i % 2], style=i % 3, time=19802.5),
                     mutable_geometry=dict(geom_rgba=np.ones((2, 4))))
        kw = dict(scene="crib", vision=vision, seed=i, tag="_w%d" % i, run_dir=str(prod / "scenes"),
                  caregiver_record_dir=str(prod / "caregiver_records"), caregiver_mode="responsive")
        worlds.append(dict(env_kwargs=kw, world=world, echo=np.zeros((3, 10)),
                           payload=dict(sim_time=19803.0), elapsed=1.0))
    state = dict(version=CS.VERSION, worlds=worlds, learner=dict(weights=np.zeros(16)),
                 trainer=dict(step=1200),
                 config=dict(action_repeat=5, motor_hold=motor_hold, n_envs=n, no_vision=not vision))
    storage = tmp_path / ("s" * 60) / ("t" * 60) / ("u" * 60)
    receipt = CS.write_episode(ck, state, storage_root=storage)
    return ck, receipt, prod, run


# ------------------------------------------------------------------ the saved episode, split
def test_a_saved_episode_past_260_characters_is_split_into_worlds_that_write_nowhere(tmp_path):
    from mimo_bridge.storage_locations import sha256_file
    ck, receipt, prod, run = _fake_checkpoint(tmp_path)
    assert len(receipt["storage_path"]) > 260          # H84's companions sit at 253-268 characters
    out = tmp_path / "stage" / "saved_worlds"
    m = B.extract_saved_worlds(str(ck), str(out))
    assert m["step"] == 1200 and m["checkpoint"] == str(ck) and m["run_dir"] == str(run)
    assert m["companion_sha256"] == receipt["sha256"] and len(m["companion"]) > 260
    assert (m["action_repeat"], m["motor_hold"], m["n_envs"]) == (5, 5, 2)
    assert json.loads((out / "saved_worlds.json").read_text(encoding="utf-8"))["step"] == 1200
    for i, row in enumerate(m["worlds"]):
        assert (row["body"], row["sim_time"], row["frame"]) == (i, 19803.0, 1980250)
        assert row["caregiver_phase"] == PHASES[i] and row["caregiver_events"] == 7 + i
        with open(row["file"], "rb") as f:
            spec = pickle.load(f)
        assert spec["env_kwargs"]["run_dir"] == str(prod / "scenes")   # kept as saved (babble_world_kwargs redirects)
        np.testing.assert_array_equal(spec["world"]["qpos"], np.arange(6.0) + i)
        events = spec["world"]["caregiver"]["events"]
        assert isinstance(events, B.DiscardedEvents) and len(events) == 7 + i
        events.append({"time": 1.0})
        assert len(events) == 8 + i
    assert not prod.exists(), "babble wrote into production's caregiver or scene folder"
    assert sha256_file(receipt["storage_path"]) == receipt["sha256"], "the saved companion changed"


def test_detaching_the_recording_changes_nothing_else_and_not_the_saved_snapshot():
    from mimo_bridge.event_tape import EventTape
    tape = EventTape("C:/nowhere/records", "C:/nowhere/status.json")
    tape.count = 5
    snap = dict(qpos=np.zeros(3), caregiver=dict(events=tape, scene_phase="engaged"))
    out = B.detach_recording(snap)
    assert snap["caregiver"]["events"] is tape
    assert out["qpos"] is snap["qpos"] and out["caregiver"]["scene_phase"] == "engaged"
    assert isinstance(out["caregiver"]["events"], B.DiscardedEvents) and len(out["caregiver"]["events"]) == 5
    assert B.detach_recording(out) is out
    plain = dict(qpos=np.zeros(3))
    assert B.detach_recording(plain) is plain


def test_the_babble_copy_keeps_every_saved_setting_but_the_two_output_folders(tmp_path):
    saved = dict(scene="crib", vision=True, seed=3, tag="_w3", body_function="modulate",
                 run_dir="C:/prod/scenes", caregiver_record_dir="C:/prod/caregiver_records",
                 caregiver_speech_dir="C:/speech", toy_hand_off="follow")
    kw = B.babble_world_kwargs(saved, str(tmp_path / "b3"))
    assert kw["run_dir"] == os.path.join(str(tmp_path / "b3"), "scene") and kw["caregiver_record_dir"] == ""
    rest = ("run_dir", "caregiver_record_dir")
    assert {k: v for k, v in kw.items() if k not in rest} == {k: v for k, v in saved.items() if k not in rest}
    assert saved["run_dir"] == "C:/prod/scenes"


def test_a_saved_world_keeps_its_own_vision_and_timing(tmp_path):
    ck, _, _, run = _fake_checkpoint(tmp_path)
    m = B.extract_saved_worlds(str(ck), str(tmp_path / "w"))
    with pytest.raises(ValueError, match="vision"):
        B.saved_world_plan(m, None, "off", 2)
    plan = B.saved_world_plan(m, None, "on", 2)
    assert [(p["body"], p["reps"], p["hold"], p["step"]) for p in plan] == [(0, 5, 5, 1200), (1, 5, 5, 1200)]
    with pytest.raises(ValueError, match="bodies"):
        B.saved_world_plan(m, None, "like", 3)
    _write(run / "args.json", dict(action_repeat=5, motor_hold=10))
    with pytest.raises(ValueError, match="motor_hold"):
        B.saved_world_plan(m, str(run), "like", 2)


# ------------------------------------------------------------------ one babble body
class _FakeEnv:
    """Stands in for BridgeEnv: records how it was built, restored, reset and stepped."""
    made = []

    def __init__(self, **kw):
        self.kw = kw
        self.model = SimpleNamespace(nu=3, geom_rgba=np.zeros((2, 4)))
        self.data = SimpleNamespace(qpos=np.zeros(6), qvel=np.zeros(5), time=0.0, act=np.zeros(0),
                                    mocap_pos=np.zeros((1, 3)), mocap_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                                    xpos=np.zeros((4, 3)))
        self.data.xquat = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
        self.proprio = SimpleNamespace(body_ids=[1, 2, 3], handle_names=["a", "b", "gaze"])
        self.frame = 0
        self.caregiver = SimpleNamespace(state=dict(scene_phase="starting", style=0, time=0.0))
        self.restored_with, self.resets, self.steps, self.closed = None, 0, [], False
        self.stepped = []
        _FakeEnv.made.append(self)

    def restore_full_state(self, snap):
        self.restored_with = snap
        d = self.data
        d.qpos, d.qvel, d.time = np.array(snap["qpos"]), np.array(snap["qvel"]), snap["time"]
        d.mocap_pos, d.mocap_quat = np.array(snap["mocap_pos"]), np.array(snap["mocap_quat"])
        self.model.geom_rgba = np.array(snap["mutable_geometry"]["geom_rgba"])
        self.frame = snap["frame"]
        self.caregiver.state = dict(snap["caregiver"])

    def reset(self):
        self.resets += 1

    def step(self, motor, gaze, mask, ctrl=None, render=None, hand_joint_goal=None):
        self.steps.append((None if motor is None else np.array(motor), render))
        self.stepped.append(dict(motor=None if motor is None else np.array(motor),
                                 gaze=None if gaze is None else np.array(gaze), mask=mask,
                                 ctrl=None if ctrl is None else np.array(ctrl), hand_joint_goal=hand_joint_goal))
        return SimpleNamespace()

    def close(self):
        self.closed = True


class _FakeRecorder:
    made = []

    def __init__(self, out_dir, body, run="", **kw):
        self.kw = dict(kw, out_dir=out_dir, body=body, run=run)
        self.marks, self.frames, self.records, self.error = [], [], 0, None
        self.orders = []
        _FakeRecorder.made.append(self)

    def mark(self, event):
        self.marks.append(event)

    def on_frame(self, env, raw, obs, order=None):
        self.frames.append(obs)
        self.orders.append(order)
        self.records += obs["rep"] == obs["reps"] - 1

    def close(self):
        pass


def test_a_babble_body_starts_from_its_saved_world_and_is_refused_when_it_did_not_restore(tmp_path, monkeypatch):
    ck, _, prod, run = _fake_checkpoint(tmp_path)
    plan = B.saved_world_plan(B.extract_saved_worlds(str(ck), str(tmp_path / "w")), str(run), "like", 2)
    monkeypatch.setattr("mimo_bridge.env.BridgeEnv", _FakeEnv)
    monkeypatch.setattr("mimo_bridge.observer_record.WorldRecorder", _FakeRecorder)
    stage = tmp_path / "stage"
    out = B.record_one(str(run), "random_orders", 1, 10, str(stage), 1001, "like", plan[1])
    env, rec = _FakeEnv.made[-1], _FakeRecorder.made[-1]
    assert env.kw["run_dir"] == os.path.join(str(stage), "scratch", "b1", "scene")
    assert env.kw["caregiver_record_dir"] == "" and (env.kw["seed"], env.kw["tag"]) == (1, "_w1")
    assert env.resets == 0 and rec.marks == ["restore"], "a saved world is restored, never reset"
    assert isinstance(env.restored_with["caregiver"]["events"], B.DiscardedEvents)
    np.testing.assert_array_equal(env.restored_with["qpos"], np.arange(6.0) + 1)
    assert out["restored"]["ok"] and out["restored"]["differs"] == [] and out["records"] == 10
    extra = rec.kw["extra"]
    assert (extra["start"], extra["checkpoint"], extra["checkpoint_step"], extra["saved_body"]) == \
        ("saved_worlds", str(ck), 1200, 1)
    assert extra["source_run"] == "run" and extra["vision"] is True and extra["order_strength"] == B.ORDER_STRENGTH
    # five frames per model step, a new order every five model steps, the eyes rendered at each burst's end
    assert len(env.steps) == 50 and [r for _, r in env.steps[:5]] == [False, False, False, False, True]
    assert [rec.frames[i]["order_seq"] for i in range(0, 50, 5)] == [0] * 5 + [1] * 5
    assert not prod.exists()

    class _Drifting(_FakeEnv):
        def restore_full_state(self, snap):
            super().restore_full_state(snap)
            self.data.qpos = self.data.qpos + 1e-6

    monkeypatch.setattr("mimo_bridge.env.BridgeEnv", _Drifting)
    with pytest.raises(RuntimeError, match="qpos"):
        B.record_one(str(run), "random_orders", 0, 10, str(tmp_path / "stage2"), 1000, "like", plan[0])
    assert _FakeEnv.made[-1].closed and _FakeEnv.made[-1].steps == []


def test_the_reset_world_path_still_resets_and_says_so(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    _write(run / "args.json", dict(action_repeat=5, motor_hold=5))
    monkeypatch.setattr("mimo_bridge.env.BridgeEnv", _FakeEnv)
    monkeypatch.setattr("mimo_bridge.observer_record.WorldRecorder", _FakeRecorder)
    out = B.record_one(str(run), "limp", 0, 2, str(tmp_path / "stage"), 1000)
    env, rec = _FakeEnv.made[-1], _FakeRecorder.made[-1]
    assert env.resets == 1 and rec.marks == ["reset"] and env.restored_with is None
    assert env.kw["seed"] == 1000 and env.kw["caregiver_record_dir"] == ""
    assert rec.kw["extra"]["start"] == "reset_world" and "restored" not in out and out["records"] == 2


# ------------------------------------------------------------------ the restore check
XML = """<mujoco><worldbody>
  <body name="box" pos="0 0 0.3"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
  <body name="arm" pos="0.3 0 0.1"><joint name="hinge" type="hinge" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/></body>
  <body name="ball" mocap="true" pos="0 0.3 0.2"><geom type="sphere" size="0.03" contype="0" conaffinity="0"/></body>
</worldbody><actuator><motor joint="hinge" gear="1"/></actuator></mujoco>"""


def test_the_restore_check_passes_an_exact_world_and_names_what_differs():
    import mujoco
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    d.ctrl[:] = 0.3
    for _ in range(20):
        mujoco.mj_step(m, d)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.zeros(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, integration, spec)
    # the physics part of BridgeEnv.save_full_state
    snap = dict(qpos=np.array(d.qpos), qvel=np.array(d.qvel), act=None, ctrl=np.array(d.ctrl),
                mocap_pos=np.array(d.mocap_pos), mocap_quat=np.array(d.mocap_quat), time=float(d.time),
                frame=20, integration_state=integration,
                mutable_geometry=dict(geom_rgba=np.array(m.geom_rgba), body_pos=np.array(m.body_pos)))
    # another world put in that state the way BridgeEnv.restore_full_state does it
    d2 = mujoco.MjData(m)
    mujoco.mj_setState(m, d2, integration, spec)
    mujoco.mj_forward(m, d2)
    mujoco.mj_setState(m, d2, integration, spec)
    env = SimpleNamespace(model=m, data=d2, _mj=mujoco, frame=20, caregiver=None)
    r = B.restore_receipt(env, snap)
    assert r["ok"] and r["differs"] == [] and (r["qpos_max_abs"], r["qvel_max_abs"]) == (0.0, 0.0)
    assert r["integration_state_equal"] is True and r["geometry_compared"] == ["geom_rgba", "body_pos"]
    assert r["sim_time"] == r["saved_sim_time"] > 0
    # one velocity off by a billionth of a metre per second
    d2.qvel[0] += 1e-9
    r = B.restore_receipt(env, snap)
    assert not r["ok"] and {"qvel", "integration_state"} <= set(r["differs"]) and 0 < r["qvel_max_abs"] < 2e-9
    d2.qvel[:] = snap["qvel"]
    # an object hidden (the caregiver withdraws a toy by editing its transparency)
    m.geom_rgba[0, 3] = 0.0
    assert B.restore_receipt(env, snap)["differs"] == ["geom_rgba"]
    m.geom_rgba[0, 3] = snap["mutable_geometry"]["geom_rgba"][0, 3]
    env.frame = 21
    assert B.restore_receipt(env, snap)["differs"] == ["frame"]
    env.frame = 20
    # the caregiver in another scene phase than it was saved in
    env.caregiver = SimpleNamespace(state=dict(scene_phase="absent", style=1, time=5.0))
    r = B.restore_receipt(env, dict(snap, caregiver=dict(scene_phase="engaged", style=1, time=5.0)))
    assert r["differs"] == ["caregiver_phase"] and r["saved_caregiver_phase"] == "engaged"
    # a snapshot saved before the adult's current shapes: migrated on restore, compared without them
    env.caregiver = None
    r = B.restore_receipt(env, dict(snap, mutable_geometry=dict(geom_size=np.ones((1, 3)))))
    assert r["ok"] and r["geometry_migrated"] == ["geom_size"]


# ------------------------------------------------------------------ where a recording goes
MANIFEST = dict(checkpoint="C:/runs/H84/checkpoints/ckpt_step_396050", step=396050, companion="C:/c.pt.gz",
                companion_sha256="ab" * 32,
                worlds=[dict(body=i, sim_time=19803.0, caregiver_phase=PHASES[i % 2]) for i in range(8)])


def _summary(fields, vision=True, generated=1.0):
    s = dict(schema="observer_baseline_v1", kind="random_orders", world_signature=SIG, steps=20000,
             occasions_per="body and part", detector_version=DM.DETECTOR_VERSION, generated=generated,
             vision=vision, order_strength=0.3, order_reach_cm=3.0, caps={"G01": dict(count=generated)},
             novelty_runs={}, novelty_steps=20000, episodes=104)
    s.update(fields)
    return s


def _stage(tmp_path, name):
    stage = tmp_path / "stages" / name
    (stage / "observer" / "world" / "b0").mkdir(parents=True)
    (stage / "marker.txt").write_text(name, encoding="utf-8")
    return str(stage)


def _marker(folder):
    return (folder / "marker.txt").read_text(encoding="utf-8")


def test_saved_world_babble_becomes_the_comparison_and_the_reset_one_moves_aside(tmp_path, monkeypatch):
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "Saved"))
    final = tmp_path / "Saved" / "observer" / "baselines" / SIG
    (final / "random_orders").mkdir(parents=True)
    (final / "random_orders" / "marker.txt").write_text("reset", encoding="utf-8")
    _write(final / "random_orders.json", _summary({}))            # recorded 09-24 morning: no start field
    (final / "random_orders_vision_off").mkdir()
    _write(final / "random_orders_vision_off.json", dict(_summary({}, vision=False), kind="random_orders_vision_off"))

    saved = _summary(B.source_fields(MANIFEST, 8, "C:/runs/H84"), generated=2.0)
    present, real_move = [], B.shutil.move

    def move(src, dst):          # the digester reads the comparison every cycle: never missing
        present.append((final / "random_orders.json").exists())
        return real_move(src, dst)

    monkeypatch.setattr(B.shutil, "move", move)
    path = B.place(str(final), "random_orders", saved, _stage(tmp_path, "saved1"))
    monkeypatch.setattr(B.shutil, "move", real_move)
    assert present == [True, True], "old recording moved aside, new one moved in, the file there throughout"
    assert path == str(final / "random_orders.json")
    now = json.loads(Path(path).read_text(encoding="utf-8"))
    assert now["start"] == "saved_worlds" and now["checkpoint_step"] == 396050 and now["source_run"] == "H84"
    assert now["source"].startswith("from saved worlds") and "ckpt_step_396050" in now["source"]
    assert _marker(final / "random_orders") == "saved1"
    aside = json.loads((final / "random_orders_reset_world.json").read_text(encoding="utf-8"))
    assert aside["kind"] == "random_orders_reset_world" and aside["start"] == "reset_world"
    assert aside["source"].startswith("from a world reset") and aside["caps"]["G01"]["count"] == 1.0
    assert aside["recording_dir"] == str(final / "random_orders_reset_world")
    assert _marker(final / "random_orders_reset_world") == "reset"
    assert (final / "random_orders_vision_off.json").exists()
    base = DM.load_baseline(SIG)                                   # what the tracker is judged against
    assert base["start"] == "saved_worlds" and base["generated"] == 2.0

    # a reset-world recording made later goes beside the saved-world comparison, never over it
    later = _summary(B.source_fields(None, run_dir="C:/runs/H84"), generated=3.0)
    path = B.place(str(final), "random_orders", later, _stage(tmp_path, "reset2"))
    assert path == str(final / "random_orders_reset_world.json")
    assert DM.load_baseline(SIG)["generated"] == 2.0 and _marker(final / "random_orders") == "saved1"
    assert _marker(final / "random_orders_reset_world") == "reset2"

    # a newer saved-world recording replaces the comparison; the reset-world one stays aside, and the
    # previous checkpoint's recording is kept beside it (2026-09-25: compared, never deleted)
    newer = _summary(B.source_fields(dict(MANIFEST, step=399000), 8), generated=4.0)
    B.place(str(final), "random_orders", newer, _stage(tmp_path, "saved2"))
    assert DM.load_baseline(SIG)["checkpoint_step"] == 399000 and _marker(final / "random_orders") == "saved2"
    assert _marker(final / "random_orders_reset_world") == "reset2"
    assert _marker(final / "random_orders_ckpt_396050") == "saved1"
    kept = json.loads((final / "random_orders_ckpt_396050.json").read_text(encoding="utf-8"))
    assert kept["kind"] == "random_orders_ckpt_396050" and kept["checkpoint_step"] == 396050
    assert kept["recording_dir"] == str(final / "random_orders_ckpt_396050")
    assert sorted(p.name for p in final.glob("*.json")) == [
        "random_orders.json", "random_orders_ckpt_396050.json", "random_orders_reset_world.json",
        "random_orders_vision_off.json"]
    # the same checkpoint recorded again replaces its own recording, as before
    again = _summary(B.source_fields(dict(MANIFEST, step=399000), 8), generated=5.0)
    B.place(str(final), "random_orders", again, _stage(tmp_path, "saved3"))
    assert DM.load_baseline(SIG)["generated"] == 5.0 and _marker(final / "random_orders") == "saved3"
    assert len(list(final.glob("*.json"))) == 4


def test_reset_world_recordings_keep_the_vision_set_aside_as_before(tmp_path):
    final = tmp_path / "b"
    (final / "random_orders").mkdir(parents=True)
    first = _summary({})
    first.pop("vision")                                            # the first baselines: no field, vision off
    _write(final / "random_orders.json", first)
    B.place(str(final), "random_orders", _summary(B.source_fields(None), generated=2.0), _stage(tmp_path, "on"))
    assert json.loads((final / "random_orders_vision_off.json").read_text(encoding="utf-8"))["vision"] is False
    assert json.loads((final / "random_orders.json").read_text(encoding="utf-8"))["vision"] is True
    B.place(str(final), "random_orders", _summary(B.source_fields(None), generated=3.0), _stage(tmp_path, "on2"))
    assert json.loads((final / "random_orders.json").read_text(encoding="utf-8"))["generated"] == 3.0
    assert sorted(p.name for p in final.glob("*.json")) == ["random_orders.json", "random_orders_vision_off.json"]
    assert B.aside_name("random_orders", dict(vision=True, start="saved_worlds"), dict(vision=False)) == \
        "random_orders_saved_worlds_vision_on"


# ------------------------------------------------------------------ the summary
def test_a_run_from_a_checkpoint_names_its_source_and_keeps_every_summary_field(tmp_path, monkeypatch):
    ck, receipt, _, run = _fake_checkpoint(tmp_path)
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "Saved"))
    monkeypatch.setattr(B, "_extract_in_child", lambda ctx, checkpoint, folder: B.extract_saved_worlds(checkpoint, folder))
    jobs_seen = []

    def record_all(ctx, jobs, workers):
        out = []
        for run_dir, kind, body, steps, stage, seed, vision, saved in jobs:
            jobs_seen.append((run_dir, body, seed))
            assert os.path.exists(saved["file"]) and saved["body"] == body and (saved["reps"], saved["hold"]) == (5, 5)
            (Path(stage) / "observer" / "world" / ("b%d" % body)).mkdir(parents=True, exist_ok=True)
            out.append(dict(body=body, records=steps, seconds=1.0, error=None, vision=True,
                            restored=dict(ok=True, differs=[])))
        return out

    monkeypatch.setattr(B, "_record_all", record_all)
    fields = dict(schema="observer_baseline_v1", kind="random_orders", world_signature=SIG, steps=20,
                  occasions_per="body and part", detector_version=DM.DETECTOR_VERSION, generated=5.0,
                  vision=True, order_strength=0.3, order_reach_cm=3.0, caps={}, novelty_runs={},
                  novelty_steps=20, episodes=0)
    monkeypatch.setattr(B, "summarize", lambda out_dir, kind, vision=None: dict(fields, vision=vision))
    path = B.run(None, "random_orders", 2, 10, 2, root=str(tmp_path / "recording"), from_checkpoint=str(ck))
    assert path == str(tmp_path / "Saved" / "observer" / "baselines" / SIG / "random_orders.json")
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    assert set(fields) <= set(s) and s["steps"] == 20 and s["vision"] is True
    assert (s["start"], s["checkpoint"], s["checkpoint_step"]) == ("saved_worlds", str(ck), 1200)
    assert s["source"].startswith("from saved worlds") and str(ck) in s["source"]
    assert s["companion_sha256"] == receipt["sha256"] and s["saved_sim_time_s"] == [19803.0, 19803.0]
    assert s["saved_caregiver_phase"] == list(PHASES) and s["source_run"] == "run"
    assert [r["body"] for r in s["recordings"]] == [0, 1] and s["recordings"][0]["restored"]["ok"]
    assert jobs_seen == [(str(run), 0, 1000), (str(run), 1, 1001)]    # the checkpoint's own run folder
    kept = Path(s["recording_dir"]) / "scratch" / "saved_worlds"
    assert (kept / "saved_worlds.json").exists() and not list(kept.glob("b*.pkl"))
    assert DM.load_baseline(SIG)["start"] == "saved_worlds"


def test_a_re_summary_keeps_where_the_bodies_started_and_the_order_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "Saved"))
    final = tmp_path / "Saved" / "observer" / "baselines" / SIG
    world = final / "random_orders_reset_world" / "observer" / "world" / "b0"
    world.mkdir(parents=True)
    _write(world / "header_1_1000.json", dict(extra=dict(baseline="random_orders", vision=True)))
    _write(final / "random_orders_reset_world.json",
           dict(_summary(B.source_fields(None, run_dir="C:/runs/H84")), kind="random_orders_reset_world",
                order_strength=None, order_reach_cm=None))

    def summarize(out_dir, kind, vision=None):       # what summarize() gives an aside's name
        return dict(_summary({}, vision=vision, generated=9.0), kind=kind, order_strength=None, order_reach_cm=None)

    monkeypatch.setattr(B, "summarize", summarize)
    s = json.loads(Path(B.resummarize(SIG, "random_orders_reset_world")).read_text(encoding="utf-8"))
    assert s["start"] == "reset_world" and s["source"].startswith("from a world reset") and s["source_run"] == "H84"
    assert (s["order_strength"], s["order_reach_cm"]) == (B.ORDER_STRENGTH, B.ORDER_REACH_CM)
    assert s["generated"] == 9.0 and s["resummarized_from"] == DM.DETECTOR_VERSION


def test_a_sealed_recording_still_names_its_world(tmp_path):
    out = tmp_path / "rec"
    world = out / "observer" / "world" / "b0"
    world.mkdir(parents=True)
    header = dict(scene="crib", nq=10, nu=4, objects=[dict(name="target")], patch_names=["a", "b"],
                  handle_names=["h"], extra=dict(vision=True, baseline="random_orders"))
    _write(world / "header_1_1000.json", header)
    analysis = out / "observer" / "analysis"
    analysis.mkdir(parents=True)
    state = dict(version=DM.ANALYSIS_VERSION, detector_version=DM.DETECTOR_VERSION, bodies={}, cuts={},
                 sealed=1.0, analysis_id="a")
    exposure = {"0": {"1": {"0": dict(steps=100, by={"random": 100}, abs=0, abs_by={})}}}
    _write(analysis / "progress.json", dict(state=state, exposure=exposure, fast={}, sizes={}))
    s = B.summarize(str(out), "random_orders")
    assert s["world_signature"] == DM.world_signature(header) and s["steps"] == 100 and s["vision"] is True


def test_the_comparison_of_saved_and_reset_babble_tests_each_capability(tmp_path, monkeypatch):
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "Saved"))
    final = tmp_path / "Saved" / "observer" / "baselines" / SIG
    final.mkdir(parents=True)
    saved = _summary(B.source_fields(MANIFEST, 8), generated=2.0)
    saved["caps"] = {"G01": dict(count=400, occasions=200), "H05": dict(count=100, occasions=50),
                     "P02": dict(count=0, occasions=0)}
    reset = dict(_summary(B.source_fields(None)), kind="random_orders_reset_world")
    reset["caps"] = {"G01": dict(count=100, occasions=60), "H05": dict(count=101, occasions=49),
                     "P02": dict(count=0, occasions=0)}
    _write(final / "random_orders.json", saved)
    _write(final / "random_orders_reset_world.json", reset)
    c = B.compare(SIG)
    assert c["baselines"] == ["random_orders", "random_orders_reset_world"]
    assert c["starts"] == ["saved_worlds", "reset_world"] and c["comparable"] and c["checkpoint_step"] == 396050
    rows = {r["cap"]: r for r in c["rows"]}
    assert rows["G01"]["occasions_per_1000_steps"] == [10.0, 3.0] and rows["G01"]["p_occasions"] < 1e-6
    assert rows["H05"]["p_occasions"] > 0.5 and rows["P02"]["p_occasions"] is None
    assert [r["cap"] for r in c["rows"]] == ["G01", "H05", "P02"]
    _write(final / "random_orders_reset_world.json", dict(reset, detector_version="observer_detectors_v4"))
    assert B.compare(SIG)["comparable"] is False


# ------------------------------------------------------------------ the command line
def test_the_command_takes_a_checkpoint_without_a_run_folder(monkeypatch, capsys):
    from mimo_bridge.tools.observer import __main__ as M
    got = {}

    def run(run_dir, kind, bodies, steps, workers, root=None, vision="like", from_checkpoint=None):
        got.update(run_dir=run_dir, kind=kind, bodies=bodies, steps=steps, workers=workers, vision=vision,
                   from_checkpoint=from_checkpoint)
        return "placed"

    monkeypatch.setattr(B, "run", run)
    assert M.main(["baseline", "--from-checkpoint", "C:/ck/ckpt_step_1", "--steps", "2500", "--workers", "8"]) == 0
    assert got == dict(run_dir=None, kind="random_orders", bodies=8, steps=2500, workers=8, vision="like",
                       from_checkpoint="C:/ck/ckpt_step_1")
    with pytest.raises(SystemExit):
        M.main(["baseline"])
    with pytest.raises(SystemExit):                  # a kept aside's name is not a chance process to record
        M.main(["baseline", "--like", "C:/run", "--kind", "random_orders_reset_world"])
    monkeypatch.setattr(B, "compare", lambda signature, kind, other=None: dict(signature=signature, kind=kind,
                                                                               other=other))
    capsys.readouterr()
    assert M.main(["baseline", "--compare", SIG]) == 0
    assert json.loads(capsys.readouterr().out) == dict(signature=SIG, kind="random_orders", other=None)
    assert M.main(["baseline", "--compare", SIG, "--other", "random_orders_ckpt_397125"]) == 0
    assert json.loads(capsys.readouterr().out)["other"] == "random_orders_ckpt_397125"
    with pytest.raises(SystemExit):                  # the yoke replays production's orders in its saved worlds
        M.main(["baseline", "--like", "C:/run", "--kind", "yoked"])
    got.clear()

    def run_yoked(run_dir, kind, bodies, steps, workers, root=None, vision="like", from_checkpoint=None,
                  yoke_from=None):
        got.update(kind=kind, from_checkpoint=from_checkpoint, yoke_from=yoke_from)
        return "placed"
    monkeypatch.setattr(B, "run", run_yoked)
    assert M.main(["baseline", "--kind", "yoked", "--from-checkpoint", "C:/ck/ckpt_step_1",
                   "--yoke-from", "C:/runs/a", "C:/runs/b"]) == 0
    assert got == dict(kind="yoked", from_checkpoint="C:/ck/ckpt_step_1", yoke_from=["C:/runs/a", "C:/runs/b"])


def test_babble_rows_carry_their_burst_spread_and_an_occasion_level_size_line():
    """tracker_rules_v4: babble's summary gives the tracker its measured burst spread (acts_sq) and
    the 95th percentile of each occasion's FIRST measured act, the line the size half of Seen tests
    the learner's occasions against."""
    import numpy as np
    from mimo_bridge.tools.observer import baseline as B
    from mimo_bridge.tools.observer import tracker as TR

    def det(step, body, sub, m):
        return dict(step0=step, body=body, sub=sub, measures={} if m is None else {"m": m})
    ds = ([det(0, 0, "L", None), det(3, 0, "L", 9.0), det(5, 0, "L", 50.0)]   # one occasion: 3 acts
          + [det(0, 0, "R", 2.0)]                                           # the other hand: its own
          + [det(100 * i, 1, "L", float(i)) for i in range(1, 21)])         # 20 one-act occasions
    row = B.chance_row(ds, ["m", "higher"], 1000.0)
    assert row["count"] == 24 and row["occasions"] == 22
    assert row["acts_sq"] == 3 ** 2 + 1 + 20
    assert TR.babble_dispersion(row) == (9 + 1 + 20) / 24.0
    # the burst's first MEASURED act is 9.0, not its larger 50.0; the act without the measure is skipped
    firsts = [9.0, 2.0] + [float(i) for i in range(1, 21)]
    assert row["measured_occasions"] == 22
    assert row["measures_occasion"]["m"] == [float(np.percentile(firsts, p)) for p in (5, 50, 95)]
    assert row["measures"]["m"][2] == float(np.percentile([9.0, 50.0, 2.0] + [float(i) for i in range(1, 21)], 95))
    assert "measures_occasion" not in B.chance_row(ds, None, 1000.0)
