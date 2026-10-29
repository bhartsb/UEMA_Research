# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/remembered_hand_learning_2026-10-11/reader.py without a world: the hand-shape score on known
answers, the joins (the remembered moment at lane step t - age, world record t - age - 1; the decision at world record
t - 1; the hold at world records t to t + 4; wellness at lane steps t, t + 1, t + 5), every missing-case reason, the
lane-to-world join check and the ring check on synthetic records, the coverage gaps, the trainer's joint-error windows,
and the loading refusals (another run, another body, a body index the run does not have, duplicate steps) on small
fake chunk files. No MuJoCo model is built and no checkpoint is loaded.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mimo_bridge.hand_target_control import CONTRACT, HandKinematics

REPO = Path(__file__).resolve().parents[4]
TOOL_DIR = REPO / "docs/gain_of_function_v4/remembered_hand_learning_2026-10-11"


@pytest.fixture
def R():
    """The reader loaded under its own module name (a bare "reader" could be another module in sys.modules)."""
    spec = importlib.util.spec_from_file_location("rhl_reader_tool", TOOL_DIR / "reader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------------ hand geometry
GEOMETRY = dict(contract=CONTRACT, handle_names=["wrist_L", "thumb_L", "index_L"], bodies=[], joints=[],
                hands=[dict(side="L", wrist_index=0, finger_indices=[1, 2], finger_bodies=[])], constraints=[])
KIN = HandKinematics(GEOMETRY)
FINGERS = [1, 2]


def quat_xyzw(axis, deg):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return np.r_[axis * math.sin(h), math.cos(h)]


def qmul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def rot(q, v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).apply(v)


def pose(wrist_q, finger_offsets_cm, finger_q=None):
    """[3, 10]: a wrist at (10, 20, 30) cm with orientation wrist_q, fingers at wrist + wrist_q * offset (cm)."""
    p = np.zeros((3, 10))
    w = np.array([10.0, 20.0, 30.0])
    p[0, 0:3], p[0, 3:7] = w / 200.0, wrist_q
    for k, off in enumerate(finger_offsets_cm):
        p[1 + k, 0:3] = (w + rot(wrist_q, off)) / 200.0
        p[1 + k, 3:7] = wrist_q if finger_q is None else finger_q[k]
    return p


WQ = quat_xyzw([0.3, -0.5, 0.8], 70.0)
OFFS = [np.array([2.0, 1.0, 0.5]), np.array([3.0, -1.0, 0.0])]


def test_identical_poses_score_zero(R):
    p = pose(WQ, OFFS)
    hm = R.hand_measures(KIN, FINGERS, p[None], p[None])
    assert hm["hand_shape_error"][0] == pytest.approx(0.0, abs=1e-12)
    assert hm["finger_position_error_cm"][0] == pytest.approx(0.0, abs=1e-9)
    assert hm["finger_orientation_error_deg"][0] == pytest.approx(0.0, abs=1e-5)


def test_displacement_of_one_unit_on_one_axis_scores_one_third(R):
    """Both fingers 1.8 cm along the wrist's own x axis: (1.8 / 1.8)^2 on one of three axes = 1/3, 1.8 cm."""
    goal = pose(WQ, [o + np.array([1.8, 0, 0]) for o in OFFS])
    hm = R.hand_measures(KIN, FINGERS, pose(WQ, OFFS)[None], goal[None])
    assert hm["hand_shape_error"][0] == pytest.approx(1 / 3, abs=1e-6)
    assert hm["finger_position_error_cm"][0] == pytest.approx(1.8, abs=1e-9)
    assert hm["finger_orientation_error_deg"][0] == pytest.approx(0.0, abs=1e-5)


def test_the_wrist_moving_alone_changes_nothing(R):
    """The score is in each wrist's frame: the whole hand turned and moved together reads 0."""
    other = qmul_xyzw(quat_xyzw([0, 0, 1], 40.0), WQ)
    a, g = pose(WQ, OFFS), pose(other, OFFS)
    g[:, 0:3] += np.array([5.0, -3.0, 2.0]) / 200.0
    hm = R.hand_measures(KIN, FINGERS, a[None], g[None])
    assert hm["hand_shape_error"][0] == pytest.approx(0.0, abs=1e-9)


def test_a_turn_of_one_unit_scores_one_third(R):
    """Both fingers turned 11 degrees about the wrist's x axis: (11 / 11)^2 on one of three axes = 1/3, 11 degrees."""
    local = quat_xyzw([1, 0, 0], 11.0)
    turned = [qmul_xyzw(WQ, local)] * 2          # wrist * local turn: the turn about the wrist's own x axis
    hm = R.hand_measures(KIN, FINGERS, pose(WQ, OFFS)[None], pose(WQ, OFFS, turned)[None])
    assert hm["hand_shape_error"][0] == pytest.approx(1 / 3, abs=1e-5)
    assert hm["finger_orientation_error_deg"][0] == pytest.approx(11.0, abs=1e-4)
    assert hm["finger_position_error_cm"][0] == pytest.approx(0.0, abs=1e-9)


def test_closed_share_known_answers(R):
    assert R.closed_share(0.5, 0.5) == 0.0                    # no movement
    assert R.closed_share(0.5, 0.0) == 1.0                    # reached the goal
    assert R.closed_share(0.5, 0.75) == pytest.approx(-0.5)   # moved away
    assert R.closed_share(0.0, 0.1) is None                   # the goal was the decision's own pose
    assert R.closed_share(None, 0.1) is None


def test_direction_label(R):
    assert R.direction_of(12.0, 10.0) == ("closing", "at_or_above_curl_line")
    assert R.direction_of(-3.0, 10.0) == ("opening", "below_curl_line")
    assert R.direction_of(-10.0, 10.0) == ("opening", "at_or_above_curl_line")
    assert R.direction_of(0.0, 10.0) == ("none", "below_curl_line")
    assert R.direction_of(float("nan"), 10.0) == (None, None)


def test_gaps(R):
    assert R.gaps_in(np.arange(10, 20), 10, 19) == []
    assert R.gaps_in(np.r_[10:13, 16:20], 10, 19) == [(13, 15)]
    assert R.gaps_in(np.arange(10, 15), 10, 19) == [(15, 19)]


def test_refuse_path(R):
    for p in ("C:/Users/bhart/Documents/UEMA_devA_private/Saved/x.json",
              "C:/x/docs/gain_of_function_v4/test_registry.json", "C:/x/run/control.json",
              "C:/x/docs/gain_of_function_v4/preflight/abc.json"):
        with pytest.raises(SystemExit):
            R.refuse_path(p)
    R.refuse_path("D:/UEMA_Research/external/D/UEMA_scratch/remembered_hand_learning_2026-10-11/reader/x.json")
    R.refuse_path("D:/UEMA_Research/external/D/UEMA_scratch/remembered_hand_learning_2026-10-11/reader/learning_reading_control.json")


def test_body_index_refused(R):
    with pytest.raises(R.Refusal):
        R.check_bodies([8], 8)
    with pytest.raises(R.Refusal):
        R.check_bodies([], 8)
    R.check_bodies([0, 7], 8)


# ------------------------------------------------------------------------------------------------ synthetic run
T, AGE, SEQ = 100, 20, 7


def make_world(first=60, last=130, episode_change_at=None, seq_change_at=None, drop=()):
    steps = np.array([s for s in range(first, last + 1) if s not in drop])
    n = len(steps)
    ep = np.zeros(n, dtype=np.int64)
    if episode_change_at is not None:
        ep[steps >= episode_change_at] = 1
    seq = np.where(steps >= T, SEQ, SEQ - 1).astype(np.int64)
    seq[steps >= T + 5] = SEQ + 1
    if seq_change_at is not None:
        seq[steps == seq_change_at] = SEQ + 9
    latched = (steps % 5) == 0
    qpos = np.zeros((n, 4))
    qpos[:, 1] = steps * 1e-3                    # every record's hand hinges differ (FakePhys.joint_addresses)
    goal = np.zeros((n, 2))
    lane = dict(step=steps, pid=np.ones(n, dtype=np.int64), episode=ep, order_seq=seq, latched=latched, qpos=qpos,
                goal=goal, contacts=[dict(rows=[], kinds=np.zeros(1, dtype=np.int64), groups=[""])] * n,
                motor=np.zeros((n, 3, 9)), gaze_index=2)
    lane["index"] = {int(s): i for i, s in enumerate(steps)}
    return lane


class FakePhys:
    kin, fingers, joint_addresses = KIN, FINGERS, [1, 2]


def poses_by_record():
    """The remembered record (T - AGE - 1) has the goal pose; the decision record (T - 1) is 1.8 cm away on one axis;
    record T is half way; record T + 4 arrived; every other record is the start pose."""
    start, goal = pose(WQ, OFFS), pose(WQ, [o + np.array([1.8, 0, 0]) for o in OFFS])
    half = pose(WQ, [o + np.array([0.9, 0, 0]) for o in OFFS])
    special = {T - AGE - 1: goal, T: half, T + 4: goal}
    return lambda b, s: special.get(int(s), start)


def plan_row(age=AGE, bodies=1, seq=SEQ, bodily=None):
    return dict(type="latch", step=T, order_seq=seq, source="planner", plan=dict(
        bodies=bodies, goal=dict(ring_index=[199 - age] * bodies, age_steps=[age] * bodies, ring_size=200,
                                 value_ranked=[1.0] * bodies, bodily=bodily or [1.5] * bodies,
                                 consequence=[0.0] * bodies),
        hand_optimizer=dict(valid=[True] * bodies, stopped_by="completed", first=[1.0] * bodies,
                            selected=[0.5] * bodies),
        wellness_rank=dict(contrib_chosen={"agency": [0.4] * bodies, "movement": [0.9] * bodies},
                           lead_over_runner_up={"agency": [0.1] * bodies}, channels_now={"agency": [0.2] * bodies}),
        chosen_rank=[1] * bodies, plan0=dict(pos_step_cm=[1.0] * bodies)))


def model_lane(steps=range(60, 131), value=0.25):
    w = {int(s): np.full((6, 1), value) + 0.001 * int(s) for s in steps}
    return dict(wellness=w, steps=np.asarray(sorted(w)))


def measure(R, world, plans=None, lane=None, lag=1):
    plans = {T: plan_row()} if plans is None else plans
    closure = {T - AGE - 1: 30.0, T - 1: 10.0, T + 4: 25.0}
    return R.measure_decisions(plans, [T], {0: world}, lane or model_lane(), FakePhys(), poses_by_record(),
                               lambda b, s: np.array([closure.get(int(s), 10.0)] * 2), lag, 5, [0], 10.0)


def test_a_decision_joins_the_right_records(R):
    rows = measure(R, make_world())
    r = rows[0]
    assert "missing" not in r, r.get("missing")
    ph = r["physical"]
    assert ph["world_records"] == dict(remembered=T - AGE - 1, decision=T - 1, plus1=T, hold_end=T + 4)
    assert r["goal"]["remembered_model_step"] == T - AGE and r["goal"]["age_consistent"]
    assert ph["start_distance"] == pytest.approx(1 / 3, abs=1e-6)
    assert ph["closed_share_plus1"] == pytest.approx(0.75, abs=1e-5)       # half the distance: a quarter of the score
    assert ph["closed_share_hold_end"] == pytest.approx(1.0, abs=1e-6)
    assert ph["direction"] == "closing" and ph["asked_closure_change_deg"] == pytest.approx(20.0)
    assert ph["achieved_closure_change_deg"] == pytest.approx(15.0)
    # wellness at lane steps T, T + 1 and T + 5 (world records T - 1, T and T + 4)
    assert r["wellness"]["decision"]["capability"] == pytest.approx(0.25 + 0.001 * T)
    assert r["wellness"]["plus1"]["capability"] == pytest.approx(0.25 + 0.001 * (T + 1))
    assert r["wellness"]["hold_end"]["capability"] == pytest.approx(0.25 + 0.001 * (T + 5))
    assert r["goal"]["top_channels"][0] == dict(channel="movement", contribution=0.9)


def test_each_missing_reason(R):
    def reason(**kw):
        rows = measure(R, make_world(**{k: v for k, v in kw.items() if k in ("first", "last", "episode_change_at",
                                                                             "seq_change_at", "drop")}),
                       plans=kw.get("plans"))
        return rows[0].get("missing", "")
    assert reason(first=T - AGE).startswith("remembered moment outside the recorded world span")
    assert reason(episode_change_at=T + 2).startswith("episode reset inside the hold")
    assert reason(episode_change_at=T - 10).startswith("remembered moment in another episode")
    assert reason(seq_change_at=T + 2).startswith("the hold did not carry the plan's command")
    assert reason(drop=(T + 3,)).startswith("no world record for a needed model step")
    assert reason(drop=(T - AGE - 1,)).startswith("no world record for a needed model step")
    assert reason(last=T + 3).startswith("no world record for a needed model step")
    assert reason(plans={}).startswith("no plan record at this planning step")


def test_wellness_only_where_recorded(R):
    rows = measure(R, make_world(), lane=model_lane(steps=range(60, T + 5)))
    assert rows[0]["wellness"]["hold_end"] is None and rows[0]["wellness"]["missing"] == ["hold_end"]
    assert "physical" in rows[0]                     # the physical outcome does not depend on the model lane


def test_goal_at_the_decision_itself_has_no_closed_share(R):
    plans = {T: plan_row(age=0)}
    rows = R.measure_decisions(plans, [T], {0: make_world()}, model_lane(), FakePhys(),
                               lambda b, s: pose(WQ, OFFS), lambda b, s: np.array([10.0, 10.0]), 1, 5, [0], 10.0)
    assert rows[0]["physical"]["start_distance"] == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["physical"]["closed_share_hold_end"] is None
    assert "closed_share_undefined" in rows[0]


def test_the_same_recorded_hand_shape_has_no_closed_share_whatever_the_noise(R):
    """Found on seg41u_1718 (model step 483,680, body 3, a goal of age 0): the score's arithmetic left a start distance of
    7e-16 and the closed share read -1e15. When every hand hinge holds the same recorded value in the remembered and the
    decision records the start is 0 by definition and the closed share undefined."""
    world = make_world()
    i_mem, i0 = world["index"][T - AGE - 1], world["index"][T - 1]
    world["qpos"][i_mem] = world["qpos"][i0]
    start = pose(WQ, OFFS)
    noisy = start.copy()
    noisy[1, 0] += 1e-15

    def pose_of(b, s):
        return noisy if int(s) == T - AGE - 1 else start
    rows = R.measure_decisions({T: plan_row()}, [T], {0: world}, model_lane(), FakePhys(), pose_of,
                               lambda b, s: np.array([10.0, 10.0]), 1, 5, [0], 10.0)
    ph = rows[0]["physical"]
    assert ph["closed_share_hold_end"] is None and ph["closed_share_plus1"] is None
    assert "closed_share_undefined" in rows[0] and ph["start_largest_hinge_change_deg"] == 0.0
    s = R.summary_block(rows)
    assert s["closed_share_undefined"] == 1 and s["closed_share_hold_end_mean"] is None
    assert s["pooled_closed_share_hold_end"] is None


def test_summary_counts_and_shares(R):
    rows = measure(R, make_world()) + [dict(step=T + 10, body=0, missing="no plan record at this planning step")]
    s = R.summary_block(rows)
    assert s["decisions"] == 2 and s["measured"] == 1 and s["measured_share"] == 0.5
    assert s["missing_reasons"] == {"no plan record at this planning step": 1}
    assert s["closed_share_hold_end_mean"] == pytest.approx(1.0, abs=1e-6)
    assert s["pooled_closed_share_hold_end"] == pytest.approx(1.0, abs=1e-6)
    assert s["wellness_changes"]["capability"]["hold_end"]["mean_change"] == pytest.approx(0.005)


def test_join_check_finds_the_configured_lag_only_when_it_is_true(R):
    """Handles move 1 cm per model step along x; the plan's pos_step_cm was measured against world record t - 1."""
    world = make_world()
    world["motor"][:] = 0.0

    def pose_at(b, s):
        p = np.zeros((3, 10))
        p[:, 0] = float(s) / 200.0
        return p
    plans = []
    for t in (90, 100, 110):
        r = plan_row()
        r["step"] = t
        r["plan"]["plan0"]["pos_step_cm"] = [float(np.linalg.norm(np.array([0.0 - (t - 1), 0, 0])))]
        plans.append(r)
    good = R.join_check(plans, {0: world}, pose_at, 1, [0])
    assert good["ok"] and good["readings"]["record_t_minus_1"]["median_cm"] == pytest.approx(0.0)
    assert not R.join_check(plans, {0: world}, pose_at, 0, [0])["ok"]
    assert not R.join_check(plans, {0: world}, pose_at, 2, [0])["ok"]


def test_ring_check(R):
    lane = model_lane()
    bodily = [float(np.nansum(lane["wellness"][T - AGE][:, 0]))]
    assert R.ring_check([plan_row(bodily=bodily)], lane, [0])["ok"]
    assert not R.ring_check([plan_row(bodily=[bodily[0] + 0.01])], lane, [0])["ok"]
    assert not R.ring_check([plan_row(age=AGE + 1, bodily=bodily)], lane, [0])["ok"]


def test_joint_error_windows_match_the_trainer_definition(R, tmp_path):
    """World record s's error is |hinge - goal| in degrees; the stats record at step S with 2 body observations per
    body covers records S - 2 and S - 1."""
    world = make_world()
    for i, s in enumerate(world["step"]):
        world["qpos"][i, 1:3] = math.radians(1.0) * s
    stats = tmp_path / "stats.jsonl"
    rec = lambda S: dict(type="mimo", step=S, hand_control=dict(  # noqa: E731
        joint_error_deg=float(np.mean([S - 2, S - 1])), body_observations=2.0))
    stats.write_text("\n".join(json.dumps(rec(S)) for S in (80, 90)) + "\n", encoding="utf-8")
    src = dict(stats=str(stats))
    out = R.joint_error_windows(src, {0: world}, [1, 2])
    assert [w["recomputed"] for w in out] == pytest.approx([w["saved"] for w in out], abs=1e-9)
    shifted = R.joint_error_windows(src, {0: world}, [1, 2], shift=1)
    assert all(abs(w["recomputed"] - w["saved"]) == pytest.approx(1.0) for w in shifted)


# ------------------------------------------------------------------------------------------------ loading refusals
def write_world_chunk(folder, run, body, steps, pid=11, seq=0, header_body=None, start_ms=1000):
    """A minimal observer_world_v2 chunk and its header: 4 qpos numbers, one command row, no contacts."""
    folder.mkdir(parents=True, exist_ok=True)
    hname = "header_%d_%d.json" % (pid, start_ms)
    header = dict(version="observer_world_v2", run=run, body=body if header_body is None else header_body,
                  nmocap=0, objects=[], qpos_quantum=[1e-6] * 4, fields=["orders"], entity_kind=[0],
                  entities=[dict(code=0, kind="self", group="trunk")], model_file="models/x.mjb.gz",
                  handle_names=["a", "b", "gaze"], handle_bodies=[1, 2, 3])
    (folder / hname).write_text(json.dumps(header), encoding="utf-8")
    n = len(steps)
    q = np.zeros((n, 4), dtype=np.int64)
    np.savez_compressed(folder / ("chunk_%09d_%d_%05d.npz" % (steps[0], pid, seq)), step=np.asarray(steps),
                        episode=np.zeros(n, dtype=np.int32), order_seq=np.ones(n, dtype=np.int64),
                        latched=np.zeros(n, dtype=bool), author=np.ones(n, dtype=np.int8),
                        qpos_q0=q[0], qpos_dq=np.diff(q, axis=0).astype(np.int32), c_row=np.zeros(0, dtype=np.int32),
                        c_e=np.zeros((0, 2), dtype=np.int16), order_on=np.ones(n, dtype=bool),
                        order_row=np.zeros(1, dtype=np.int32), order_motor=np.zeros((1, 3, 9), dtype=np.float32),
                        order_mask=np.ones((1, 3), dtype=np.int8), order_hand_goal=np.zeros((1, 2), dtype=np.float32),
                        order_gaze=np.zeros((n, 2), dtype=np.float32), header=np.asarray(hname))


def test_world_lane_loads_and_refuses_another_run_or_body(R, tmp_path):
    good = tmp_path / "w" / "b2"
    write_world_chunk(good, "RUN_A", 2, list(range(100, 110)))
    write_world_chunk(good, "RUN_A", 2, list(range(110, 120)), seq=1)
    lane = R.load_world_lane(str(good), 2, "RUN_A")
    assert list(lane["step"]) == list(range(100, 120))
    dropped = R.load_world_lane(str(good), 2, "RUN_A", drop={"chunk_%09d_%d_%05d.npz" % (110, 11, 1)})
    assert R.gaps_in(dropped["step"], 100, 119) == [(110, 119)]
    with pytest.raises(R.Refusal, match="run RUN_A, not RUN_B"):
        R.load_world_lane(str(good), 2, "RUN_B")
    with pytest.raises(R.Refusal, match="body 2's records, not body 3's"):
        R.load_world_lane(str(good), 3, "RUN_A")
    with pytest.raises(R.Refusal, match="no world lane folder"):
        R.load_world_lane(str(tmp_path / "w" / "b8"), 8, "RUN_A")


def test_world_lane_refuses_a_step_recorded_twice_in_one_process(R, tmp_path):
    f = tmp_path / "w" / "b0"
    write_world_chunk(f, "RUN_A", 0, list(range(100, 110)))
    write_world_chunk(f, "RUN_A", 0, list(range(105, 115)), seq=1)
    with pytest.raises(R.Refusal, match="twice"):
        R.load_world_lane(str(f), 0, "RUN_A")


def test_a_relaunch_drops_the_abandoned_future(R, tmp_path):
    """A second process that resumed at step 105 makes the first process's records from 105 on an abandoned future
    (tools/observer/stream.abandoned_from); episodes are told apart by (process, episode)."""
    f = tmp_path / "w" / "b0"
    write_world_chunk(f, "RUN_A", 0, list(range(100, 112)), pid=11)
    write_world_chunk(f, "RUN_A", 0, list(range(105, 115)), pid=22, start_ms=2000)
    lane = R.load_world_lane(str(f), 0, "RUN_A")
    assert list(lane["step"]) == list(range(100, 115))
    assert list(lane["pid"]) == [11] * 5 + [22] * 10


def test_model_lane_refuses_another_run(R, tmp_path):
    f = tmp_path / "model"
    f.mkdir()
    (f / "header_5_1000.json").write_text(json.dumps(dict(run="RUN_B", n_envs=8)), encoding="utf-8")
    np.savez_compressed(f / "chunk_000000100_5_00000.npz", step=np.arange(100, 103),
                        wellness_names=np.asarray(list(R.WELLNESS_NAMES)), wellness=np.zeros((3, 6, 8), np.float32))
    with pytest.raises(R.Refusal, match="names run RUN_B, not RUN_A"):
        R.load_model_lane(str(f), "RUN_A", 8)
    lane = R.load_model_lane(str(f), "RUN_B", 8)
    assert sorted(lane["wellness"]) == [100, 101, 102] and lane["wellness"][100].shape == (6, 8)
    with pytest.raises(R.Refusal, match="records 8 bodies, not the run's 4"):
        R.load_model_lane(str(f), "RUN_B", 4)


def test_the_world_model_is_found_beside_the_body_folders(R, tmp_path):
    """The header's model_file is relative to <run>/observer/world, the body folders' parent (a bug found on the first
    real run: the reader joined it to the body folder and found no model)."""
    body = tmp_path / "run" / "observer" / "world" / "b0"
    body.mkdir(parents=True)
    (tmp_path / "run" / "observer" / "world" / "models").mkdir()
    (tmp_path / "run" / "observer" / "world" / "models" / "abc.mjb.gz").write_bytes(b"")
    header = dict(model_file="models/abc.mjb.gz")
    assert Path(R.world_root_of(str(body)), header["model_file"]).is_file()
    assert Path(R.world_root_of(str(body) + "/"), header["model_file"]).is_file()
    assert not Path(str(body), header["model_file"]).is_file()


def test_the_chain_files_renamed_aside_are_read_oldest_first(R, tmp_path):
    """causal_chain.py renames the file aside past 50 MB (causal_chain_<YYYYmmdd_HHMMSS>[_k].jsonl); a later row for the
    same step (a relaunch) replaces the earlier; a "plan" row is a plan that did not reach the body."""
    def row(step, mark, kind="latch"):
        return json.dumps(dict(type=kind, step=step, mark=mark, plan=dict(goal={}))) + "\n"
    (tmp_path / "causal_chain_20261010_170000.jsonl").write_text(row(10, "a") + row(20, "a"), encoding="utf-8")
    (tmp_path / "causal_chain_20261010_170000_1.jsonl").write_text(row(20, "b"), encoding="utf-8")
    (tmp_path / "causal_chain.jsonl").write_text(row(30, "c") + row(40, "c", kind="plan"), encoding="utf-8")
    (tmp_path / "causal_chain_notes.jsonl").write_text(row(10, "z"), encoding="utf-8")      # not a renamed chain file
    names = [Path(p).name for p in R.chain_files(str(tmp_path / "causal_chain.jsonl"))]
    assert names == ["causal_chain_20261010_170000.jsonl", "causal_chain_20261010_170000_1.jsonl",
                     "causal_chain.jsonl"]
    plans, unreached = R.load_plans(str(tmp_path / "causal_chain.jsonl"))
    assert {s: r["mark"] for s, r in plans.items()} == {10: "a", 20: "b", 30: "c"}
    assert list(unreached) == [40]


def test_hand_contact_kinds(R):
    """Entity groups as the observer names them: a finger touching the same hand's palm is same_hand, the other hand
    or the trunk own_body, a toy object, the floor surface; a contact between two non-hand parts is not the hand's."""
    from mimo_bridge.observer_record import KIND_SELF, KIND_OBJECT, KIND_FLOOR
    kinds = np.array([KIND_SELF, KIND_SELF, KIND_SELF, KIND_SELF, KIND_OBJECT, KIND_FLOOR])
    groups = ["finger_L_index", "hand_L_palm", "finger_R_thumb", "trunk", "box", "floor"]

    def kinds_of(*rows):
        return R.contact_kinds(dict(rows=list(rows), kinds=kinds, groups=groups))
    assert kinds_of((0, 1)) == {"same_hand"}
    assert kinds_of((0, 2)) == {"own_body"}
    assert kinds_of((3, 1)) == {"own_body"}
    assert kinds_of((4, 2)) == {"object"}
    assert kinds_of((0, 5)) == {"surface"}
    assert kinds_of((3, 5)) == set()


def test_a_missing_plan_row_is_explained_or_reported(R):
    """A latch row is written when its hold resolves, at model step t + hold: a run whose last model step is earlier
    never writes it (found on seg41j_1526: the decision at 484,670 resolves at 484,675; the run's last step is
    484,674). That is a counted reason, not a fault; a row missing for no known reason is reported."""
    reasons, unexplained = R.plan_gap_reasons([484650, 484660, 484670], {484660: {}}, {484650: {}}, 5, 484674)
    assert reasons[484650].startswith("the plan did not reach the body")
    assert reasons[484670].startswith("the run ended before the causal chain wrote this decision's record")
    assert unexplained == []
    reasons, unexplained = R.plan_gap_reasons([484650, 484660], {484660: {}}, {}, 5, 484674)
    assert reasons[484650] == "no plan record at this planning step" and unexplained == [484650]


# ------------------------------------------------------------------------------------------------ review of Oct 11
# READER_REVIEW.md (D:/UEMA_Research/external/D/UEMA_scratch/remembered_hand_learning_2026-10-11/reviews): B1 start-distance strata, F1 the
# verdict's checks on the rows the measuring code wrote, F2 a two-body fixture and a Physics.pose test, F3 the code
# record and `compare`, note 6 the last-decision rule.
AGES = (20, 30)


def pose_lin(b, s):
    """Every handle moves 1 cm per model step along world x; the hand's shape never changes."""
    p = pose(WQ, OFFS)
    p[:, 0] += float(s) / 200.0
    return p


def pos_step_of(m):
    """plan0.pos_step_cm of a plan at model step m whose command targets the origin: the mean distance of the two body
    handles (the eyes' row, handle 2, is left out) from it, where the model saw them: world record m - 1."""
    return float(np.linalg.norm(pose_lin(0, m - 1)[[0, 1], 0:3] * 200.0, axis=1).mean())


def two_body_lane():
    w = {int(s): np.stack([np.full(6, 0.25 + 0.001 * s), np.full(6, 0.5 + 0.002 * s)], axis=1) for s in range(40, 131)}
    return dict(wellness=w, steps=np.asarray(sorted(w)))


def two_body_plans(lane, ages=AGES):
    plans = {}
    for s in range(70, 131, 10):           # remembered steps T - 20 = 80 and T - 30 = 70 are planning steps too
        r = plan_row(bodies=2, bodily=[float(np.nansum(lane["wellness"][s - a][:, b])) for b, a in enumerate(ages)])
        r["step"] = s
        r["plan"]["goal"]["age_steps"] = list(ages)
        r["plan"]["goal"]["ring_index"] = [199 - a for a in ages]
        r["plan"]["hand_optimizer"]["ring_index"] = [199 - a for a in ages]
        r["plan"]["plan0"]["pos_step_cm"] = [pos_step_of(s)] * 2
        plans[s] = r
    return plans


def two_body_rows(R):
    lane, world = two_body_lane(), {0: make_world(first=40), 1: make_world(first=40)}
    plans = two_body_plans(lane)
    rows = R.measure_decisions({T: plans[T]}, [T], world, lane, FakePhys(), pose_lin,
                               lambda b, s: np.array([10.0, 10.0]), 1, 5, [0, 1], 10.0)
    return rows, plans, world, lane


def test_two_bodies_each_use_their_own_goal_age(R):
    """F2: with every body given body 0's goal age (a fault the review planted), body 1's remembered moment would be
    T - 20 instead of T - 30; each row must carry its own body's age, step and world record."""
    rows, plans, world, lane = two_body_rows(R)
    for r, age in zip(rows, AGES):
        assert r["goal"]["age_steps"] == age and r["goal"]["remembered_model_step"] == T - age
        assert r["physical"]["world_records"]["remembered"] == T - age - 1
    assert R.row_ring_check(rows, plans, lane)["ok"]


def test_the_row_checks_pass_on_correct_rows_and_fail_on_each_planted_fault(R):
    """F1: the verdict's checks read the rows the measuring code wrote."""
    import copy
    rows, plans, world, lane = two_body_rows(R)
    checks = dict(ring=lambda rs, ps: R.row_ring_check(rs, ps, lane), ring_index=R.ring_index_check,
                  identity=lambda rs, ps: R.row_identity_check(rs, 1, 5),
                  join=lambda rs, ps: R.row_join_check(rs, ps, world, pose_lin))
    for name, check in checks.items():
        assert check(rows, plans)["ok"], name
    # the remembered record taken without the lag
    bad = copy.deepcopy(rows)
    for r in bad:
        r["physical"]["world_records"]["remembered"] += 1
    assert not checks["identity"](bad, plans)["ok"]
    assert not checks["join"](bad, plans)["ok"]
    assert not R.row_join_check(bad, plans, world, pose_lin)["groups"]["remembered"]["ok"]
    # every body given body 0's goal age
    bad = copy.deepcopy(rows)
    bad[1]["goal"]["remembered_model_step"] = T - AGES[0]
    assert not checks["ring"](bad, plans)["ok"]
    # the hand stage aimed at another ring entry than the goal
    other = copy.deepcopy(plans)
    other[T]["plan"]["hand_optimizer"]["ring_index"][1] += 1
    assert not checks["ring_index"](rows, other)["ok"]
    # a decision record off by one
    bad = copy.deepcopy(rows)
    for r in bad:
        r["physical"]["world_records"]["decision"] += 1
    assert not checks["join"](bad, plans)["ok"]


def test_physics_pose_reads_x_y_z_w(R):
    """F2: MuJoCo's xquat is w x y z; Physics.pose writes x y z w (the layout of every pose the score reads)."""
    import types
    phys = object.__new__(R.Physics)
    phys.mj = types.SimpleNamespace(mj_kinematics=lambda m, d: None)
    phys.m = None
    phys.handle_bodies = np.array([2, 0])
    phys.d = types.SimpleNamespace(qpos=np.zeros(3), xpos=np.array([[0.0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3]]),
                                   xquat=np.array([[1.0, 0, 0, 0], [0, 0, 0, 1], [0.5, 0.1, 0.2, 0.3]]))
    out = phys.pose(np.zeros(3))
    assert out.shape == (2, 10)
    assert out[0, 0:3] == pytest.approx(np.array([10.0, 20.0, 30.0]) / 200.0)      # metres x 100 / 200
    assert list(out[0, 3:7]) == [0.1, 0.2, 0.3, 0.5]                               # x y z w
    assert list(out[1, 3:7]) == [0.0, 0.0, 0.0, 1.0]


ONE_HINGE = dict(contract=CONTRACT, handle_names=["wrist_L", "index_L", "gaze"],
                 bodies=[dict(name="finger", parent=-1, side="L", position=[0.02, 0.005, 0.0],
                              quaternion=[1.0, 0.0, 0.0, 0.0], joints=[0])],
                 joints=[dict(name="hinge", source_id=0, qpos_address=0, axis=[0.0, 0.0, 1.0], anchor=[0.0, 0.0, 0.0],
                              limits=[-1.5, 1.5], reference=0.0)],
                 hands=[dict(side="L", wrist_index=0, finger_indices=[1], finger_bodies=[0])], constraints=[])


def test_kinematic_agreement_catches_quaternions_read_in_the_wrong_order(R):
    """F1: the scored wrist-frame finger pose against the hand geometry's own hinge kinematics."""
    import types
    kin = HandKinematics(ONE_HINGE)
    phys = types.SimpleNamespace(kin=kin, geometry=ONE_HINGE)
    angles = np.array([[0.3], [-0.7], [1.1]])
    template = torch.zeros(3, 3, 10, dtype=torch.float64)
    template[:, 0, 0:3] = torch.tensor([0.05, -0.02, 0.1], dtype=torch.float64)
    template[:, 0, 3:7] = torch.as_tensor(quat_xyzw([0.2, 0.9, -0.3], 50.0))
    poses = kin.assemble_world(template, torch.as_tensor(angles)).numpy()
    pos, rot = R.kinematic_agreement(phys, poses, angles)
    assert pos <= R.KIN_POS_TOL_CM and rot <= R.KIN_ROT_TOL_DEG
    wrong = poses.copy()
    wrong[..., 3:7] = poses[..., [6, 3, 4, 5]]
    pos, rot = R.kinematic_agreement(phys, wrong, angles)
    assert pos > R.KIN_POS_TOL_CM and rot > R.KIN_ROT_TOL_DEG


def strata_rows(starts, ends):
    rows = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        rows.append(dict(step=10 * i, body=i % 2, physical=dict(
            start_distance=s, hold_end=dict(hand_shape_error=e), plus1=dict(hand_shape_error=(s + e) / 2),
            closed_share_hold_end=1 - e / s, reduction_hold_end=s - e)))
    return rows


def test_strata_are_the_quartile_edges_and_start_sum_shares_of_their_source(R):
    """B1: edges = numpy's linear 25/50/75th percentiles of the source's start distances; weight = each stratum's share
    of the summed start distance; a start equal to an edge belongs to the stratum above it."""
    starts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    edges, weights, counts = R.strata_from_starts(starts)
    assert edges == pytest.approx([2.75, 4.5, 6.25]) and counts == [2, 2, 2, 2]
    assert weights == pytest.approx([3 / 36, 7 / 36, 11 / 36, 15 / 36])
    e2, _, c2 = R.strata_from_starts([1.0, 2.0, 3.0, 4.0, 5.0])
    assert e2 == pytest.approx([2.0, 3.0, 4.0]) and c2 == [1, 1, 1, 2]


def fake_reading(rows, run="RUN_A", window=None, pinned=None, strata=None, verdict="pass", planning_steps=99):
    window = window or dict(first_model_step=1, last_model_step=1000, planning_steps=planning_steps)
    o = dict(pooled_closed_share_hold_end=0.1, start_distance_mean=1.0, end_error_mean=0.9, decisions=len(rows),
             measured=len(rows), equal_body_mean_pooled_closed_share_hold_end=dict(mean=0.1))
    return dict(verdict=verdict, run=run, window=window, bodies=[0, 1], decisions=rows, lane_lag=1,
                hold_model_steps=5, plan_every_model_steps=10,
                code=dict(pinned=pinned or dict(reader_sha256="r", score_modules={"a.py": "1"},
                                                source_python_tree="t", source_python_dirty=False, reader_dirty=False)),
                strata=None if strata is None else dict(sha256=strata["sha256"]),
                summary=dict(overall=o, per_body={}))


def test_strata_are_frozen_and_refuse_other_edges(R):
    rows = strata_rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [0.5, 2.5, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0])
    strata = R.make_strata(fake_reading(rows))
    R.check_strata(strata)
    edited = json.loads(json.dumps(strata))
    edited["edges"][1] = 4.0
    with pytest.raises(R.Refusal, match="content hash"):
        R.check_strata(edited)
    edited["sha256"] = R._content_hash({k: v for k, v in edited.items() if k != "sha256"})
    with pytest.raises(R.Refusal, match="not the quartile edges of its own start distances"):
        R.check_strata(edited)
    with pytest.raises(R.Refusal, match="valid reading only"):
        R.make_strata(fake_reading(rows, verdict="invalid"))


def test_per_stratum_results_and_the_pooled_share_at_the_source_mix(R):
    """B1: at its own mix the source's weighted share equals its raw pooled share; another reading with another mix of
    the same per-stratum results reads the source's share once weighted, whatever its raw share."""
    starts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ends = [s * f for s, f in zip(starts, [0.5, 0.5, 1.0, 1.0, 0.8, 0.8, 0.9, 0.9])]
    strata = R.make_strata(fake_reading(strata_rows(starts, ends)))
    rows = strata_rows(starts, ends)
    blocks, mix = R.stratum_blocks(rows, strata)
    assert [b["decisions"] for b in blocks] == [2, 2, 2, 2]
    assert [b["pooled_closed_share_hold_end"] for b in blocks] == pytest.approx([0.5, 0.0, 0.2, 0.1])
    raw = 1 - sum(ends) / sum(starts)
    assert mix == pytest.approx(raw)
    # the same per-stratum shares, but only the large starts doubled: the raw share moves, the source-mix share does not
    s2, e2 = starts + [7.0, 8.0], ends + [7.0 * 0.9, 8.0 * 0.9]
    rows2 = strata_rows(s2, e2)
    blocks2, mix2 = R.stratum_blocks(rows2, strata)
    assert 1 - sum(e2) / sum(s2) != pytest.approx(raw)
    assert mix2 == pytest.approx(raw)


def test_summary_strata_and_equal_body_means(R):
    rows, plans, world, lane = two_body_rows(R)
    for r, s, e in zip(rows, (1.0, 3.0), (0.5, 3.0)):
        r["physical"].update(start_distance=s, closed_share_hold_end=1 - e / s, reduction_hold_end=s - e)
        r["physical"]["hold_end"]["hand_shape_error"] = e
        r["physical"]["plus1"]["hand_shape_error"] = e
    strata = R.make_strata(fake_reading(strata_rows([0.5, 1.0, 2.0, 3.0], [0.5, 0.5, 1.0, 1.0])))
    summ = R.summarize(rows, [0, 1], strata)
    o = summ["overall"]
    assert o["equal_body_mean_pooled_closed_share_hold_end"]["mean"] == pytest.approx((0.5 + 0.0) / 2)
    assert o["pooled_closed_share_hold_end"] == pytest.approx(1 - 3.5 / 4.0)
    assert len(o["strata"]) == 4 and sum(b["decisions"] for b in o["strata"]) == 2
    assert "strata" in summ["per_body"]["0"] and "strata" in o["by_direction"]["closing"]


def test_compare_refuses_other_code_other_strata_and_unequal_windows(R):
    """F3 and note 6: readings compared only with one pinned code record, the frozen strata from the source reading,
    and the same number of planning steps."""
    rows = strata_rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [0.5, 2.5, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0])
    strata = R.make_strata(fake_reading(rows))
    base = fake_reading(rows, strata=strata)
    end = fake_reading(rows, run="RUN_B", window=dict(first_model_step=5, last_model_step=9, planning_steps=99),
                       strata=strata)
    table, refusals = R.compare_readings({"base": base, "end": end}, strata)
    assert refusals == [] and set(table) == {"base", "end"}
    other_code = json.loads(json.dumps(end))
    other_code["code"]["pinned"]["score_modules"]["a.py"] = "2"
    assert any("score module a.py" in x for x in R.compare_readings({"base": base, "end": other_code}, strata)[1])
    other_reader = json.loads(json.dumps(end))
    other_reader["code"]["pinned"]["reader_sha256"] = "q"
    assert any("reader_sha256" in x for x in R.compare_readings({"base": base, "end": other_reader}, strata)[1])
    dirty = json.loads(json.dumps(end))
    dirty["code"]["pinned"]["source_python_dirty"] = True
    assert any("uncommitted" in x for x in R.compare_readings({"base": base, "end": dirty}, strata)[1])
    no_strata = json.loads(json.dumps(end))
    no_strata["strata"] = None
    assert any("not the frozen" in x for x in R.compare_readings({"base": base, "end": no_strata}, strata)[1])
    longer = json.loads(json.dumps(end))
    longer["window"]["planning_steps"] = 100
    assert any("planning steps" in x for x in R.compare_readings({"base": base, "end": longer}, strata)[1])
    moved = json.loads(json.dumps(base))
    moved["decisions"][0]["physical"]["start_distance"] = 1.5
    assert any("not the measured starts" in x for x in R.compare_readings({"base": moved, "end": end}, strata)[1])
    assert any("not among the readings" in x for x in R.compare_readings({"end": end}, strata)[1])


def test_the_endpoint_step_reads_every_window_with_one_code_and_compares(R, tmp_path):
    """F3: the completion job reads both baselines and the endpoint now, with the frozen strata, writes each reading,
    and compares them; a strata file with another hash is refused before any reading."""
    rows = strata_rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [0.5, 2.5, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0])
    strata = R.make_strata(fake_reading(rows, run="j"))
    path = tmp_path / "strata.json"
    path.write_text(json.dumps(strata), encoding="utf-8")
    seen = []

    def read(src, first, last, bodies, strata=None, last_decision=None):
        seen.append((src["run_name"], first, last, last_decision))
        return fake_reading(rows, run=src["run_name"], strata=strata), 0
    out, code = R.endpoint_reading(str(path), strata["sha256"], [("x/j", 1, 1000, 990), ("x/u", 1, 1000, 990)],
                                   ("x/e", 2001, 3000, 2990), str(tmp_path / "out"), read=read)
    assert code == 0 and out["verdict"] == "pass" and out["refusals"] == []
    assert seen == [("j", 1, 1000, 990), ("u", 1, 1000, 990), ("e", 2001, 3000, 2990)]
    assert len(out["reading_files"]) == 3 and all(Path(p).is_file() for p in out["reading_files"])
    assert [r["value"] for r in out["readings"]] == [0.1, 0.1, 0.1]
    with pytest.raises(R.Refusal, match="not the frozen"):
        R.endpoint_reading(str(path), "0" * 64, [("x/j", 1, 1000, 990)], ("x/e", 2001, 3000, 2990),
                           str(tmp_path / "out"), read=read)


def test_the_code_record_hashes_the_reader_and_every_score_module(R):
    import lem.model.physical_goal  # noqa: F401  (loaded by every reading; loaded here when this test runs alone)
    rec = R.code_record()
    pinned = rec["pinned"]
    assert set(pinned["score_modules"]) == set(R.SCORE_MODULES)
    assert all(len(v) == 64 for v in pinned["score_modules"].values())
    assert pinned["reader_sha256"] == R.sha256_file(R.__file__) and pinned["score_reference_sha256"]
    assert "lem/model/physical_goal.py" in rec["loaded_modules"]
    assert R.code_record()["pinned_sha256"] == rec["pinned_sha256"]


def test_the_last_decision_rule_drops_and_counts(R):
    """Note 6: the endpoint window's decision at 489,340 has no counterpart in the baseline window."""
    kept, dropped = R.protocol_steps(list(range(488350, 489341, 10)), 489330, 8)
    assert kept[-1] == 489330 and len(kept) == 99
    assert dropped["planning_steps"] == [489340] and dropped["decision_body_pairs"] == 8
    assert R.protocol_steps([10, 20], None, 8) == ([10, 20], None)


# ------------------------------------------------------------------------------------------------ a whole reading
# THE ROW CHECKS MUST REACH THE VERDICT (review re-check of Oct 11: a fourth planted fault left them out of the verdict
# and every unit test still passed, since none ran a whole reading). A synthetic one-body run folder (args, causal chain,
# observer world and model lanes) is read by read_window with a fake Physics on the one-hinge hand: the reading passes,
# and each row check made to fail makes it invalid.
S_AGE, S_STEPS = 23, (40, 139)
R_NAMES = ("capability", "coherence", "motor_state", "prediction_error", "self_touch", "tactile_comfort")


class SyntheticPhysics:
    """Physics' interface without MuJoCo: the wrist at x = qpos[0] x 100 cm, the one finger hinge at qpos[1]."""
    model_file, geometry_sha256 = "models/synthetic.mjb.gz", "synthetic"

    def __init__(self):
        self.geometry, self.kin = ONE_HINGE, HandKinematics(ONE_HINGE)
        self.fingers, self.joint_addresses, self.handle_names = [1], [1], list(ONE_HINGE["handle_names"])

    def pose(self, qpos):
        template = torch.zeros(1, 3, 10, dtype=torch.float64)
        template[0, 0, 0] = float(qpos[0]) * 100.0 / 200.0
        template[0, 0, 3:7] = torch.as_tensor(WQ)
        return self.kin.assemble_world(template, torch.tensor([[float(qpos[1])]], dtype=torch.float64))[0].numpy()

    def closure(self, qpos):
        return np.array([10.0, 10.0])


def synthetic_run(tmp_path):
    """(run folder, physics): plans at 80, 90, 100 and 110, each aiming at the moment 23 model steps back."""
    from mimo_bridge.observer_model_lane import read_model_chunk
    run = tmp_path / "RUN_S"
    steps = np.arange(S_STEPS[0], S_STEPS[1] + 1)
    n = len(steps)
    (run / "observer" / "world" / "b0").mkdir(parents=True)
    (run / "observer" / "model").mkdir(parents=True)
    (run / "args.json").write_text(json.dumps(dict(out=str(run), n_envs=1, motor_hold=5, intention_every=10,
                                                   goal_locality="off", hand_goals="memory",
                                                   hand_response_calibration="", self_touch_wellness=0.0)),
                                   encoding="utf-8")
    header = dict(version="observer_world_v2", run="RUN_S", body=0, nmocap=0, objects=[], qpos_quantum=[1e-6, 1e-6],
                  fields=["orders"], entity_kind=[0], entities=[dict(code=0, kind="self", group="trunk")],
                  model_file=SyntheticPhysics.model_file, handle_names=ONE_HINGE["handle_names"], handle_bodies=[1, 2, 3])
    w = run / "observer" / "world" / "b0"
    (w / "header_7_1000.json").write_text(json.dumps(header), encoding="utf-8")
    q = np.stack([steps * 10000, steps * 1000], axis=1).astype(np.int64)    # x = step cm; hinge = step mrad (1e-6 quanta)
    latch = np.nonzero(steps % 5 == 0)[0].astype(np.int32)
    np.savez_compressed(w / "chunk_000000040_7_00000.npz", step=steps, episode=np.zeros(n, np.int32),
                        order_seq=(steps // 5).astype(np.int64), latched=steps % 5 == 0, author=np.ones(n, np.int8),
                        qpos_q0=q[0], qpos_dq=np.diff(q, axis=0).astype(np.int32), c_row=np.zeros(0, np.int32),
                        c_e=np.zeros((0, 2), np.int16), order_on=np.ones(n, bool), order_row=latch,
                        order_motor=np.zeros((len(latch), 3, 9), np.float32),
                        order_mask=np.ones((len(latch), 3), np.int8),
                        order_hand_goal=np.zeros((len(latch), 1), np.float32), order_gaze=np.zeros((n, 2), np.float32),
                        header=np.asarray("header_7_1000.json"))
    m = run / "observer" / "model"
    (m / "header_7_1000.json").write_text(json.dumps(dict(run="RUN_S", n_envs=1)), encoding="utf-8")
    well = (0.1 + 0.001 * steps[:, None, None] + 0.01 * np.arange(6)[None, :, None]).astype(np.float32)
    np.savez_compressed(m / "chunk_000000040_7_00000.npz", step=steps, wellness_names=np.asarray(list(R_NAMES)),
                        wellness=well)
    lane = read_model_chunk(str(m / "chunk_000000040_7_00000.npz"))
    phys = SyntheticPhysics()
    lines = []
    for t in (80, 90, 100, 110):
        i = int(np.nonzero(lane["step"] == t - S_AGE)[0][0])
        pos = phys.pose(np.array([(t - 1) * 0.01, (t - 1) * 1e-3]))[[0, 1], 0:3] * 200.0
        lines.append(json.dumps(dict(type="latch", step=t, order_seq=t // 5, source="planner", plan=dict(
            bodies=1, goal=dict(ring_index=[199 - S_AGE], age_steps=[S_AGE], ring_size=200, value_ranked=[1.0],
                                bodily=[float(np.nansum(np.asarray(lane["wellness"][i][:, 0], dtype=np.float64)))],
                                consequence=[0.0]),
            hand_optimizer=dict(valid=[True], ring_index=[199 - S_AGE], first=[1.0], selected=[0.5],
                                stopped_by="completed"),
            wellness_rank=dict(contrib_chosen={"agency": [0.1]}, lead_over_runner_up={"agency": [0.1]},
                               channels_now={"agency": [0.2]}),
            chosen_rank=[1], plan0=dict(pos_step_cm=[float(np.linalg.norm(pos, axis=1).mean())])))))
    (run / "causal_chain.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run, phys


def read_synthetic(R, folder):
    run, phys = synthetic_run(folder)
    src = R.run_sources(str(run), [0])
    cache = {(SyntheticPhysics.model_file, R.world_root_of(src["world"][0])): phys}
    return R.read_window(src, 75, 119, [0], physics_cache=cache)


def test_a_whole_synthetic_reading_passes_and_every_row_check_reaches_the_verdict(R, tmp_path, monkeypatch):
    result, code = read_synthetic(R, tmp_path / "clean")
    assert code == 0 and result["verdict"] == "pass", result["problems"]
    assert result["summary"]["overall"]["measured"] == 4
    assert all(v["ok"] for v in result["row_checks"].values())
    assert result["summary"]["overall"]["remembered_lane_steps_since_latch_counts"] == {"2": 4}
    for name, func in (("ring", "row_ring_check"), ("ring_index", "ring_index_check"),
                       ("identity", "row_identity_check"), ("join", "row_join_check"),
                       ("kinematics", "row_kinematics_check")):
        with monkeypatch.context() as mp:
            mp.setattr(R, func, lambda *a, **k: dict(ok=False, rule="planted failure"))
            res, code = read_synthetic(R, tmp_path / name)
        assert code == R.INVALID_EXIT and res["verdict"] == "invalid", name
        assert any(p.startswith("row check %s failed" % name) for p in res["problems"]), res["problems"]
