# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--finger-joint-servo: one meaning for the finger command (action_decoder.py, THE FINGER COMMAND'S MEANING).

The first-finger investigation of Oct 8, 2026 (finding C, D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/body_execution/REVIEW.md):
the planner search and the hand stage choose one goal angle per finger hinge, while the decoder servoed each finger
handle's pose and could push a hinge against its own goal. With the setting 'on' each finger hinge is servoed toward
its own goal; 'off' (the default) must leave every number as it was.

Two groups:
  IDENTITY WITH THE SETTING OFF - the decoder, the replay contract helpers and the transition contract check against
  the production commit's own code (read with git show), the world's call of the decoder and its reports, the
  trainer's world settings and fingerprint, and a resume's saved world settings.
  THE CORRECTION'S OWN CHECK - with the setting on: every finger hinge's command has the sign of its own goal minus its
  angle and the chain keeps one shared scale; the recorded case of episode v2 (model step 483,790, body 5) flips the
  middle hinge's command from against its goal to toward it; the world passes the latched goal and reports the
  contract; the coupling tendons and the goal's feasibility check stay; the hand stage refuses a fixed response
  fit under the other meaning, and the probe harness refuses a world that runs the other rule.
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mimo_infant")

from mimo_bridge import action_decoder as AD
from mimo_bridge import limb_target_control as LT
from mimo_bridge import transition as TR

# the production commit this branch starts from (fix/explainer-audit-2026-09-15 at the operator's instruction of
# 7:37 AM PDT, Oct 8, 2026); the identity tests read its files with git show
BASE = "15569a136"
REPO = Path(__file__).resolve().parents[4]


def _production_module(relpath: str, name: str):
    """The production commit's version of a mimo_bridge module, loaded beside the branch's (relative imports resolve
    inside mimo_bridge). Skips when git or the commit is not available."""
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", "%s:%s" % (BASE, relpath)], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip("production commit %s not readable: %s" % (BASE, exc))
    spec = importlib.util.spec_from_loader("mimo_bridge._production_" + name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "mimo_bridge"
    sys.modules[spec.name] = mod
    exec(compile(src, "%s@%s" % (relpath, BASE), "exec"), mod.__dict__)
    return mod


def _crib(tmp_path, tag, **kw):
    from mimo_bridge.env import BridgeEnv
    return BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=False, tag=tag, age=24,
                     contact_profile="mechanical_v4", caregiver_seconds=0, kp=6.0, body_layout="trunk_v1",
                     hand_target_mode="wrist_relative_v1", **kw)


@pytest.fixture(scope="module")
def crib(tmp_path_factory):
    env = _crib(tmp_path_factory.mktemp("finger_servo"), "_finger_servo")
    env.reset()
    yield env
    env.close()


def _decoder(env, module=AD):
    d0 = env.decoder
    return module.ActionDecoder(env.model, kp=d0.kp, kd=d0.kd, gaze_kp=d0.gaze_kp, gaze_kd=d0.gaze_kd, lam=d0.lam,
                                ctrl_bounds=(env.env.action_space.low, env.env.action_space.high),
                                body_layout="trunk_v1")


def _moved(env, n, rng, speed=0.1):
    """n randomly moved states of the crib body (every hinge within its range)."""
    m = env.model
    hinge = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
    base = np.array(env.data.qpos)
    for _ in range(n):
        d = mujoco.MjData(m)
        d.qpos[:] = base
        for j in hinge:
            a = int(m.jnt_qposadr[j])
            lo, hi = (float(x) for x in m.jnt_range[j]) if m.jnt_limited[j] else (-0.5, 0.5)
            d.qpos[a] = np.clip(base[a] + rng.normal(0.0, 0.2), lo, hi)
        d.qvel[:] = rng.normal(0.0, speed, m.nv) if speed else 0.0
        mujoco.mj_forward(m, d)
        yield d


def _order(env, d, rng, strength=None):
    bones = env.proprio.body_ids
    n = len(bones)
    order = np.zeros((n, 9))
    order[:, 0:3] = np.asarray(d.xpos[bones]) * 100.0 + rng.normal(0.0, 3.0, (n, 3))
    q = rng.normal(size=(n, 4))
    order[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    order[:, 7:9] = rng.uniform(0.0, 1.0, (n, 2))
    if strength is not None:
        order[:, 8] = strength
    return order


def _feasible_goal(env, d, rng):
    import torch
    hc = env.hand_target_control
    q = np.asarray(d.qpos[hc._joint_addresses], dtype=float)
    raw = q + rng.normal(0.0, 0.5, q.shape)
    return hc.kinematics.project_joints(torch.as_tensor(raw)).numpy()


# ====================================================================== IDENTITY WITH THE SETTING OFF
def test_off_the_decoder_is_the_production_decoder_bit_for_bit(crib):
    """48 random commands at 16 randomly moved states of the crib body (hinge speeds drawn too): the branch's decoder
    without finger goals gives the production commit's control, every actuator, bit for bit."""
    prod = _production_module("Source/Python/mimo_bridge/action_decoder.py", "action_decoder")
    new, old = _decoder(crib), _decoder(crib, prod)
    rng = np.random.default_rng(3)
    n = 0
    for d in _moved(crib, 16, rng):
        for _ in range(3):
            order = _order(crib, d, rng)
            gaze = rng.uniform(-0.5, 0.5, 2)
            assert np.array_equal(new.decode(d, order, gaze), old.decode(d, order, gaze))
            assert np.array_equal(new.decode(d, order, gaze, None, finger_goals=None), old.decode(d, order, gaze))
            n += 1
    assert n == 48


def test_off_the_contract_helpers_are_the_productions():
    """replay_contract, moment_contract and action_contract_parts under every combination the production commit
    knows, and transition.known_action_contract over every ordered subset of the production parts, equal the
    production commit's answers; the new bit and part appear only with the setting on."""
    plt = _production_module("Source/Python/mimo_bridge/limb_target_control.py", "limb_target_control")
    ptr = _production_module("Source/Python/mimo_bridge/transition.py", "transition")
    for hand in ("world", "wrist_relative_v1"):
        for parent in LT.PARENT_RELATIVE_CHOICES:
            for rows in AD.POSITION_ROW_CHOICES:
                for eye in ("legacy", "physical"):
                    want = plt.replay_contract(hand, parent, rows, eye)
                    assert LT.replay_contract(hand, parent, rows, eye) == want
                    assert LT.replay_contract(hand, parent, rows, eye, "off") == want
                    on = LT.replay_contract(hand, parent, rows, eye, "on")
                    assert on == (want or 0) | LT.FINGER_SERVO_CONTRACT_BIT
    metas = []
    for hand in ("world_v0", "wrist_relative_joint_v1"):
        for limb in (None, LT.CALF_CONTRACT, LT.CALF_CONTRACT + "+" + LT.CHEST_CONTRACT, LT.ARM_CONTRACT):
            for rows in (None, "no_head_pelvis"):
                m = dict(hand_target_contract=hand)
                if limb:
                    m["limb_target_contract"] = limb
                if rows:
                    m["position_rows"] = rows
                metas.append(m)
    for m in metas:
        assert LT.action_contract_parts(m) == plt.action_contract_parts(m)
        assert LT.moment_contract([m, m]) == plt.moment_contract([m, m])
        assert LT.moment_contract([m]) == plt.moment_contract([m])
    assert LT.moment_contract([]) == plt.moment_contract([])
    import itertools
    old_parts = ptr.ACTION_CONTRACT_PARTS
    for r in range(1, len(old_parts) + 1):
        for combo in itertools.permutations(old_parts, r):
            c = "+".join(combo)
            assert TR.known_action_contract(c) == ptr.known_action_contract(c), c
    assert TR.known_action_contract("world_v0")
    assert TR.ACTION_CONTRACT_PARTS[:len(old_parts)] == old_parts


def test_off_the_world_calls_the_decoder_as_before_and_reports_nothing_new(tmp_path, monkeypatch):
    """A crib world built without the setting, and one built with it 'off': the decoder is called with no finger
    goal, the world reports no 'finger_servo' and the production motor contract, and the two worlds step the same
    commands to the same states, bit for bit."""
    worlds = [_crib(tmp_path / "a", "_off_a"), _crib(tmp_path / "b", "_off_b", finger_joint_servo="off")]
    try:
        calls = []
        for w in worlds:
            real = w.decoder.decode

            def spy(*a, _real=real, **k):
                calls.append(k)
                return _real(*a, **k)
            monkeypatch.setattr(w.decoder, "decode", spy)
            w.reset()
        rng = np.random.default_rng(5)
        hc = worlds[0].hand_target_control
        for step in range(6):
            d = worlds[0].data
            order = _order(worlds[0], d, rng, strength=0.5)
            goal = _feasible_goal(worlds[0], d, rng)
            outs = [w.step(order, (0.0, 0.0), None, render=False, hand_joint_goal=goal) for w in worlds]
            for o in outs:
                assert "finger_servo" not in o.info
                assert o.info["motor_contract"] == AD.MOTOR_CONTRACT
            assert np.array_equal(worlds[0].data.qpos, worlds[1].data.qpos), step
            assert np.array_equal(outs[0].info["decoded_ctrl"], outs[1].info["decoded_ctrl"]), step
        assert calls and all("finger_goals" not in k for k in calls)
        assert worlds[0].finger_joint_servo == "off" and hc is not None
    finally:
        for w in worlds:
            w.close()


def test_off_the_trainers_world_settings_fingerprint_and_resume_are_unchanged():
    """Without the flag (and with it 'off'): parse_args reads 'off', env_kwargs_for names no new setting, the acting
    configuration holds no new key, and a resume under --resume-env-settings launch writes nothing into saved worlds
    that lack the setting. With it 'on' all three carry it."""
    import json
    from mimo_bridge import continuation_state as CS
    from mimo_bridge import train_mimo as TM
    argv = ["--out", "x", "--resume-env-settings", "launch", "--hand-target-mode", "wrist_relative_v1",
            "--toy-hand-off", "off", "--body-layout", "trunk_v1"]
    for extra in ([], ["--finger-joint-servo", "off"]):
        args = TM.parse_args(argv + extra)
        assert args.finger_joint_servo == "off"
        launch = TM.env_kwargs_for(args, None, "x")
        assert "finger_joint_servo" not in launch
        assert "finger_joint_servo" not in TM.acting_configuration(args)
        # a saved world as H84's companions hold them (tests/test_limb_target_control.py H84_LIKE_WORLD)
        world = dict(scene="crib", run_dir="x/scenes", vision=True, render_every=1, touch_scale_mult=1.0, seed=7,
                     tag="_b0", kp=6.0, joint_damping_scale=3.0, hinge_armature=0.02, finger_armature=0.0004,
                     hand_target_mode="wrist_relative_v1", toy_hand_off="off", body_layout="trunk_v1")
        state = dict(version=CS.VERSION, worlds=[dict(env_kwargs=dict(world), world={}, payload={})], config={},
                     trainer=dict(step=1))
        before = json.dumps(state["worlds"][0]["env_kwargs"]).encode()
        for settings in ("launch", "saved"):
            up, changes = CS.migrate_env_kwargs(state, args, launch_kwargs=launch, env_settings=settings)
            assert changes == [] and up is state
            assert json.dumps(up["worlds"][0]["env_kwargs"]).encode() == before
    on = TM.parse_args(argv + ["--finger-joint-servo", "on"])
    assert TM.env_kwargs_for(on, None, "x")["finger_joint_servo"] == "on"
    assert TM.acting_configuration(on)["finger_joint_servo"] == "on"
    up, changes = CS.migrate_env_kwargs(state, on, launch_kwargs=TM.env_kwargs_for(on, None, "x"),
                                        env_settings="launch")
    assert [(c["name"], c["new"]) for c in changes] == [("finger_joint_servo", "on")]
    assert up["worlds"][0]["env_kwargs"]["finger_joint_servo"] == "on"
    with pytest.raises(ValueError):
        CS.migrate_env_kwargs(state, SimpleNamespace(finger_joint_servo="sometimes"), env_settings="launch")


# ====================================================================== THE CORRECTION'S OWN CHECK
def test_on_every_finger_hinge_is_commanded_toward_its_own_goal(crib):
    """16 moved states at rest and 16 with hinge speeds drawn from N(0, 0.3 rad/s), random feasible hand goals,
    every row powered: with the finger goals each finger chain's commands are strength x (kp x (goal - angle) -
    kd x hinge speed) times ONE shared factor in (0, 1] (the decoder's rescale), so at rest each finger hinge's
    command has the sign of its goal minus its angle (unless the hinge already sits on its goal), and in motion it
    opposes that sign only where kd x speed toward the goal exceeds kp x (goal - angle) (the speed term braking);
    every actuator outside the finger chains is the command without finger goals, bit for bit."""
    dec = _decoder(crib)
    hc = crib.hand_target_control
    rng = np.random.default_rng(11)
    finger_acts = {a for i in dec.finger_handles for a in dec.handles[i].act_ids}
    checked = 0
    for speed in (0.0, 0.3):
        for d in _moved(crib, 16, rng, speed=speed):
            order = _order(crib, d, rng, strength=0.6)
            goal = _feasible_goal(crib, d, rng)
            goals = hc.goals_by_joint(goal)
            on = dec.decode(d, order, (0.0, 0.0), finger_goals=goals)
            off = dec.decode(d, order, (0.0, 0.0))
            others = [a for a in range(crib.model.nu) if a not in finger_acts]
            assert np.array_equal(on[others], off[others])
            for i in dec.finger_handles:
                h = dec.handles[i]
                need = np.array([goals[j] - d.qpos[int(crib.model.jnt_qposadr[j])] for j in h.joint_ids])
                turning = np.array([d.qvel[int(dof)] for dof in h.dof_adrs])
                u = np.array([on[a] for a in h.act_ids])
                demand = 0.6 * (dec.kp * need - dec.kd * turning)
                live = np.abs(demand) > 1e-9
                ratios = u[live] / demand[live]
                assert np.all(ratios > 0) and np.all(ratios <= 1 + 1e-9), (dec.handle_names[i], ratios)
                assert np.allclose(ratios, ratios[0], rtol=1e-9, atol=1e-12), (dec.handle_names[i], ratios)
                if speed == 0.0:
                    toward = np.abs(need) > 1e-6
                    assert np.all(np.sign(u[toward]) == np.sign(need[toward])), (dec.handle_names[i], need, u)
                checked += 1
    assert checked == 2 * 16 * len(dec.finger_handles) == 320


def test_on_the_recorded_case_turns_the_middle_hinge_toward_its_goal(crib):
    """Episode v2 of the investigation, body 5, model step 483,790 (planner command 96759): the left index hinges at
    side 4.12, knuckle 5.46, middle 41.84, distal 31.80 degrees, turning at -25.7, -109.2, -12.2, -15.7 degrees/s;
    the latched goal side 4.72, knuckle 56.67, middle 19.94, distal 27.64; strength 0.398. The recorded command was
    knuckle +0.285, middle +0.525 (bend, against the middle's goal), distal -0.345. Without finger goals the decoder
    gives the same signs here; with them the middle hinge is commanded to straighten, toward its goal, and every index
    hinge's command has the sign of its own goal minus its angle. (The middle finger's knuckle is set to 30 degrees,
    current and goal, so the recorded index goal satisfies the index-middle coupling band of 0.52 rad.)"""
    m, hc = crib.model, crib.hand_target_control
    dec = _decoder(crib)
    d = mujoco.MjData(m)
    d.qpos[:] = crib.data.qpos
    d.qvel[:] = 0.0
    names = ["robot:left_ff_side", "robot:left_ff_knuckle", "robot:left_ff_middle", "robot:left_ff_distal"]
    angle = np.radians([4.12, 5.46, 41.84, 31.80])
    speed = np.radians([-25.7, -109.2, -12.2, -15.7])
    target = np.radians([4.72, 56.67, 19.94, 27.64])
    for n, a, v in zip(names, angle, speed):
        j = m.joint(n).id
        d.qpos[int(m.jnt_qposadr[j])] = a
        d.qvel[int(m.jnt_dofadr[j])] = v
    for n in ("robot:left_mf_knuckle", "robot:left_rf_knuckle", "robot:left_lf_knuckle"):
        # the other knuckles at 30 degrees: every coupling band of the left hand holds for the current angles
        # and for the goal (the bands couple index-middle, middle-ring and ring-little)
        d.qpos[int(m.jnt_qposadr[m.joint(n).id])] = math.radians(30.0)
    mujoco.mj_forward(m, d)
    import torch
    goal = np.asarray(d.qpos[hc._joint_addresses], dtype=float).copy()
    order_names = [jt["name"] for jt in hc.geometry["joints"]]
    rows = [order_names.index(n) for n in names]
    goal[rows] = target
    # the production feasibility map (the right hand's settled angles may sit a hair outside a soft band); the
    # left index goal is feasible as recorded and keeps its numbers
    goal = hc.kinematics.project_joints(torch.as_tensor(goal)).numpy()
    assert np.allclose(goal[rows], target, atol=1e-12)
    hc.kinematics.validate_joints(goal)              # within the hinge and coupling limits
    bones = crib.proprio.body_ids
    request = np.zeros((len(bones), 9))
    request[:, 0:3] = np.asarray(d.xpos[bones]) * 100.0
    request[:, 3:7] = np.asarray(d.xquat[bones])[:, [1, 2, 3, 0]]
    index = list(dec.handle_names).index("index_L")
    request[index, 8] = 0.398                         # only the left index chain is powered
    executed, _ = hc.execute(d, request, None, goal)  # the world's fingertip targets for this goal
    acts = [m.actuator(a).id for a in ("act:left_ff_side", "act:left_ff_knuckle", "act:left_ff_middle",
                                       "act:left_ff_distal")]
    off = dec.decode(d, executed, None)[acts]
    on = dec.decode(d, executed, None, finger_goals=hc.goals_by_joint(goal))[acts]
    assert off[1] > 0.1 and off[2] > 0.1 and off[3] < -0.3, off   # the recorded signs: middle against its goal
    need = target - angle
    assert np.all(np.sign(on) == np.sign(need)), (on, need)       # each hinge toward its own goal
    assert on[2] < -0.1 and on[1] > 0.5, on


def test_on_the_world_passes_the_latched_goal_and_reports_the_contract(tmp_path, monkeypatch):
    """A crib world with the setting on: every decoder call receives the latched joint goal by MuJoCo joint id, the
    world reports 'finger_servo' and the extended motor contract, the moment carries replay's bit, the transition's
    contract gains its part (and is known), and the coupling tendons are the off world's."""
    env = _crib(tmp_path / "on", "_on", finger_joint_servo="on")
    ref = _crib(tmp_path / "off", "_on_ref")
    try:
        got = []
        real = env.decoder.decode

        def spy(*a, **k):
            got.append(k.get("finger_goals"))
            return real(*a, **k)
        monkeypatch.setattr(env.decoder, "decode", spy)
        env.reset()
        rng = np.random.default_rng(13)
        hc = env.hand_target_control
        for _ in range(3):
            order = _order(env, env.data, rng, strength=0.5)
            goal = _feasible_goal(env, env.data, rng)
            out = env.step(order, (0.0, 0.0), None, render=False, hand_joint_goal=goal)
            assert got[-1] == hc.goals_by_joint(goal)
            assert out.info["finger_servo"] == AD.FINGER_SERVO_CONTRACT
            assert out.info["motor_contract"] == AD.MOTOR_CONTRACT + "+" + AD.FINGER_SERVO_CONTRACT
            assert LT.moment_contract([out.info]) & LT.FINGER_SERVO_CONTRACT_BIT
            parts = LT.action_contract_parts(out.info)
            assert parts[-1] == AD.FINGER_SERVO_CONTRACT
            assert TR.known_action_contract("+".join(["wrist_relative_joint_v1"] + parts))
            LT.check_finger_servo([out.info], "on")
            with pytest.raises(RuntimeError):
                LT.check_finger_servo([out.info], "off")
        names = sorted(env.model.tendon(i).name for i in range(env.model.ntendon))
        assert names == sorted(ref.model.tendon(i).name for i in range(ref.model.ntendon))
        assert sum(n.startswith("finger_couple_") for n in names) == 6
        bad = np.asarray(env.data.qpos[hc._joint_addresses], dtype=float).copy()
        k = [jt["name"] for jt in hc.geometry["joints"]].index("robot:left_ff_knuckle")
        bad[k] = math.radians(200.0)
        with pytest.raises(ValueError):
            env.step(_order(env, env.data, rng, strength=0.5), (0.0, 0.0), None, render=False, hand_joint_goal=bad)
    finally:
        env.close()
        ref.close()


def test_the_setting_is_refused_where_no_joint_goal_exists(tmp_path):
    from mimo_bridge.env import BridgeEnv
    with pytest.raises(ValueError):
        BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=False, tag="_world_mode", finger_joint_servo="on")
    with pytest.raises(ValueError):
        BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=False, tag="_bad", finger_joint_servo="yes",
                  hand_target_mode="wrist_relative_v1")
    assert not TR.known_action_contract("finger_joint_servo_v1")
    assert TR.known_action_contract("wrist_relative_joint_v1+finger_joint_servo_v1")
    assert TR.known_action_contract("wrist_relative_joint_v1+parent_relative_calf_v1+finger_joint_servo_v1")
    assert not TR.known_action_contract("finger_joint_servo_v1+wrist_relative_joint_v1")
    assert TR.ACTION_CONTRACT_PARTS[-1] == AD.FINGER_SERVO_CONTRACT


def test_a_wrist_relative_transition_under_the_setting_validates():
    ex = np.zeros((26, 9)); ex[:, 6] = 1.0
    t = TR.Pending(env_id=0, episode_id=0, frame_t=1, state_t={"frame": 1}, achieved_t=None, context_t=None,
                   proposed_action=ex).complete(
        executed_action=ex, state_t1={"frame": 2}, frame_t1=2, duration_s=0.05,
        action_contract="wrist_relative_joint_v1+finger_joint_servo_v1",
        motor_substeps=np.repeat(ex[None], 5, 0), hand_joint_goal=np.zeros(44))
    assert t.action_contract.endswith("finger_joint_servo_v1")


def test_on_the_hand_stage_refuses_a_response_fit_under_the_pose_servo():
    """The hand stage's fixed response (lem/model/hand_response.py) predicts each hinge's 250 ms change from
    coefficients fit on executed commands. The admitted calibration (hand_response_calibration_380200.json) declares
    no command meaning, so it was fit under the pose servo. A learner with --hand-goals memory configures it as before
    without the setting and with it 'off'; with the setting 'on' it refuses that calibration and accepts one whose
    receipt declares the joint servo; with the setting 'off' or absent it refuses a receipt declaring the joint servo;
    with hand goals off the response is not used and nothing is refused."""
    import json
    import torch
    from lem.model.hand_response import POSE_SERVO_MEANING, command_meaning, content_hash
    from mimo_bridge.lem_model import LemLite
    from test_hand_response import fixture
    admitted = json.loads((REPO / "docs/gain_of_function_v3/measurements/hand_response_calibration_380200.json")
                          .read_text(encoding="utf-8"))
    assert command_meaning(admitted) == POSE_SERVO_MEANING and admitted["training_examples"] == 28
    torch.set_num_threads(1)
    model = LemLite(device="cpu", no_vision=True, imagination=False, next_frame_loss=True, intention=True,
                    hand_target_mode="wrist_relative_v1", hand_goals="memory", smooth_actions=.35,
                    smooth_source="executed", overrides={"model": {"context_length": 2}})
    geometry, receipt, _ = fixture()
    geometry["handle_names"] = model.handle_names

    def calibration(**extra):
        r = {k: v for k, v in dict(receipt, **extra).items() if k != "sha256"}
        r["geometry_sha256"] = content_hash(geometry)
        r["sha256"] = content_hash({k: v for k, v in r.items() if k != "sha256"})
        return dict(r, calibration_file_sha256="a" * 64)

    model._hand_response_calibration = calibration()
    model.configure_hand_geometry(geometry, torch.zeros(2, 1))           # no setting on the learner: as before
    model.finger_joint_servo = "off"
    model.configure_hand_geometry(geometry, torch.zeros(2, 1))
    assert torch.equal(model.hand_response.coefficients, torch.tensor(receipt["coefficients"], dtype=torch.float64))
    model.finger_joint_servo = "on"
    with pytest.raises(ValueError, match="refit"):
        model.configure_hand_geometry(geometry, torch.zeros(2, 1))
    model._hand_response_calibration = calibration(command_meaning=AD.FINGER_SERVO_CONTRACT)
    model.configure_hand_geometry(geometry, torch.zeros(2, 1))
    assert command_meaning(model.hand_response.receipt) == AD.FINGER_SERVO_CONTRACT
    # the reverse pairing (the independent review of Oct 8, 2026, item (d)): a receipt fit under the joint servo, as an
    # on-run's checkpoint would restore it, with the setting off (or absent from a learner built without it) is
    # refused too
    for rule in ("off", None):
        if rule is None:
            del model.finger_joint_servo
        else:
            model.finger_joint_servo = rule
        with pytest.raises(ValueError, match="refit"):
            model.configure_hand_geometry(geometry, torch.zeros(2, 1))
    model.hand_goals = "off"
    model._hand_response_calibration = calibration()
    model.configure_hand_geometry(geometry, torch.zeros(2, 1))
    model._hand_response_calibration = calibration(command_meaning=AD.FINGER_SERVO_CONTRACT)
    model.configure_hand_geometry(geometry, torch.zeros(2, 1))           # hand goals off: the response is not used


def test_a_probe_session_whose_world_runs_another_finger_rule_is_refused(tmp_path):
    """The probe harness (tools/gof_probe_harness.py) builds a Session's learner from the run's args.json and its world
    from overrides, another run's settings or a saved world. With hand goals on, a world whose --finger-joint-servo
    differs from the learner's is refused, in both directions (the independent review's re-check of 852f21504, item
    (d)): the learner's hand stage would predict with a calibration checked against the other rule. The learner with a
    world of its own rule is accepted, its hand stage configured with the admitted calibration (a crib world at age 24
    with H84's world settings has that calibration's hand geometry)."""
    import torch
    import motor_asked_change_fixture as fx
    from mimo_bridge import train_mimo as T
    from mimo_bridge.tools import gof_probe_harness as G
    cal = REPO / "docs/gain_of_function_v3/measurements/hand_response_calibration_380200.json"
    world = ["--scene", "crib", "--age-binding", "fixed:24", "--contact-profile", "mechanical_v4", "--decoder-kp", "6.0",
             "--joint-damping-scale", "3.0", "--hinge-armature", "0.02", "--finger-armature", "0.0004",
             "--body-function", "modulate", "--hand-target-mode", "wrist_relative_v1", "--caregiver-seconds", "0",
             "--hand-goals", "memory", "--hand-response-calibration", str(cal)]
    torch.set_num_threads(1)
    args = T.parse_args(fx.argv(n_envs=1, out=str(tmp_path / "out"), extra=world))
    args.n_envs = 1
    model = T.build_learner(args, None)
    model.reset_context(1)
    assert model.finger_joint_servo == "off" and model.hand_goals == "memory"
    # the developmental limits name a phase of train_mimo's tables ("mature": the crib, no added delays)
    frozen = G.Frozen(model=model, args=args, info={}, limits={"phase_name": "mature"}, limits_source="test", load={},
                      build_s=0.0, load_s=0.0)
    with pytest.raises(ValueError, match="finger-joint-servo on but the learner"):
        G.Session(frozen, run_dir=str(tmp_path / "w_on"), vision=False, overrides={"finger_joint_servo": "on"})
    session = G.Session(frozen, run_dir=str(tmp_path / "w_off"), vision=False)       # the learner's own rule
    try:
        assert session.env.finger_joint_servo == "off" and model.hand_response is not None
    finally:
        session.close()
    model.finger_joint_servo = "on"          # the learner's rule on, the world's off
    with pytest.raises(ValueError, match="finger-joint-servo off but the learner"):
        G.Session(frozen, run_dir=str(tmp_path / "w_off2"), vision=False)
