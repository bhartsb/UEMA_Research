# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Integration tests against the real MIMo v2 body (needs the editable mimo_infant install).

Slow-ish (a few seconds per env build). Checks the things that only the real model can tell:
all 23 tracked bones exist, every handle drives at least one joint, a real step encodes into
a frame the model's tokenizer reads back, and the pooled touch numbers are sane.
"""
import os

import numpy as np
import pytest

mimo_infant = pytest.importorskip("mimo_infant")

from tokenizer import config as C
from lem.model.body_sets import HANDLE_NAMES
from mimo_bridge.env import BridgeEnv, scene_xml, SCENES
from mimo_bridge.proprio_adapter import BONE_MAP
from mimo_bridge import frame_encoder as FE

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Saved", "mimo_bridge", "scenes_test")


@pytest.fixture(scope="module")
def env():
    e = BridgeEnv(scene="reach", run_dir=RUN_DIR, vision=True, tag="_pytest")
    yield e
    e.close()


def test_all_tracked_bones_exist_and_chains_drive_joints(env):
    names = [env.model.body(i).name for i in range(env.model.nbody)]
    for h in HANDLE_NAMES:
        assert BONE_MAP[h] in names, h
    driven = env.decoder.handle_joint_names()
    for h in HANDLE_NAMES:
        if h == "gaze":
            # deliberate: the eye hinges are driven by the dedicated eye servo, sourced
            # from this handle's orientation model-side. An empty chain here is the
            # design, not a gap (see HANDLE_CHAINS).
            assert len(driven[h]) == 0
            continue
        assert len(driven[h]) >= 1, f"handle {h} drives no joint"
    # wrists drive the 3 wrist joints plus the arm (elbow + 3 shoulder) for the position rows;
    # index finger drives 4 joints (side, knuckle, middle, distal)
    assert {"robot:right_hand1", "robot:right_hand2", "robot:right_hand3", "robot:right_elbow"} <= set(driven["wrist_R"])
    assert len(driven["wrist_R"]) == 7
    assert len(driven["index_R"]) == 4 and len(driven["pinky_L"]) == 5
    assert len(env.decoder.eye) == 4


def test_patch_count_and_names(env):
    assert env.n_patches == 53                      # 55 sensing bodies minus the two eyes
    names = env.patch_names()
    assert "left_eye" not in names and "head" in names and "right_thdistal" in names
    assert names == sorted(names, key=lambda n: env.model.body(n).id)


def test_step_encodes_and_decodes(env):
    r = env.reset()
    for _ in range(10):
        motor = np.zeros((24, 9)); motor[:, 6] = 1.0; motor[:, 8] = 0.3
        r = env.step(motor, gaze=(0.1, 0.0))
    assert r.patches.shape == (53, 10) and r.proprio.bones.shape == (24, 11)
    assert r.eye_left.shape == (256, 256, 3) and r.eye_left.dtype == np.uint8
    assert not np.array_equal(r.eye_left, r.eye_right)      # real eyes, unlike MIMo's SimpleVision bug
    handles = np.zeros((24, 10), np.float32); handles[:, 6] = 1; handles[:, 9] = 1
    buf = FE.encode_frame(frame_number=r.frame, timestamp=r.sim_time, dt=r.dt, wall_clock=0.0,
                          gaze_az_deg=r.gaze_az_el_deg[0], gaze_el_deg=r.gaze_az_el_deg[1],
                          bones=r.proprio.bones, bone_hashes=r.proprio.hashes, handles=handles,
                          energy=r.proprio.energy, body_sets=r.proprio.body_sets,
                          patches=r.patches, collisions=r.collisions, vision_valid=True)
    tf = FE.decode_with_model_tokenizer(bytes(buf))
    assert tf.proprio.shape == (24, 12) and tf.touch_impulses.shape == (256, 10)
    # MIMo has no soft skin, so both principal-stretch channels arrive at rest, which
    # the tokenizer re-zeroes - the page says they are a constant zero here
    assert not tf.touch_impulses[:, 8:10].any()
    assert tf.touch_valid_counts[0] == 53
    # quaternions came back unit length, positions relative to wrist_L (row 0 is zero)
    assert np.allclose(np.linalg.norm(tf.proprio[:, 0:4].numpy(), axis=1), 1.0, atol=1e-4)
    assert np.allclose(tf.proprio[0, 4:7].numpy(), 0.0)
    # no NaNs anywhere
    assert not np.isnan(r.patches).any() and not np.isnan(r.proprio.bones).any()


def test_touch_numbers_are_sane_when_hanging(env):
    """In the reach scene MIMo hangs by the upper body; the ball may or may not be touched.
    Patches with area 0 must have pressure 0; pressures must be finite and not absurd."""
    r = env.reset()
    for _ in range(20):
        r = env.step(np.zeros((24, 9)))
    p = r.patches
    assert np.all(p[p[:, 3] == 0, 0] == 0)
    assert np.all(np.isfinite(p))
    assert p[:, 0].max() < 1e7          # Pa; a 10 N force on 1 mm2 would be 1e7
    assert np.all(p[:, 8] == 1.0) and np.all(p[:, 9] == 1.0)


def test_all_scenes_build():
    for kind in SCENES:
        xml = scene_xml(kind)
        assert "MIMo_modelv2.xml" in xml and "MIMo_metav2.xml" in xml


def test_wrist_position_ik_moves_hand_toward_target(env):
    """Ask the right wrist to go 10 cm forward and 10 cm up (orientation held). After 1 s the
    hand should be much closer to that point (was 14 cm away, ends ~1 cm in the tuning run)."""
    env.reset()
    hand = env.model.body("right_hand").id
    p0 = env.data.xpos[hand].copy(); q = env.data.xquat[hand]
    motor = np.zeros((24, 9)); motor[:, 6] = 1
    target = p0 + np.array([0.10, 0.0, 0.10])
    motor[6, 0:3] = target * 100; motor[6, 3:7] = [q[1], q[2], q[3], q[0]]; motor[6, 8] = 1.0
    d0 = np.linalg.norm(target - p0)
    for _ in range(100):
        env.step(motor)
    d1 = np.linalg.norm(target - env.data.xpos[hand])
    assert d1 < 0.4 * d0, (d0, d1)


# ---------------------------------------------------------------- sharper, binocular eyes
@pytest.fixture(scope="module")
def fovea_env():
    e = BridgeEnv(scene="reach", run_dir=RUN_DIR, vision=True, tag="_pytest_fovea",
                  eye_size=256, fovea_size=288, fovea_fov=24.0)
    yield e
    e.close()


def test_fovea_cameras_are_boresighted_with_the_eyes(fovea_env):
    """A narrow camera inside each eye: same point, same direction, less world per pixel."""
    m, d = fovea_env.model, fovea_env.data
    names = [m.camera(i).name for i in range(m.ncam)]
    assert "fovea_left" in names and "fovea_right" in names
    for eye, fov in (("eye_left", "fovea_left"), ("eye_right", "fovea_right")):
        a, b = m.camera(eye).id, m.camera(fov).id
        assert m.cam_fovy[b] == pytest.approx(24.0)
        assert m.cam_fovy[a] == pytest.approx(60.0)
        assert np.linalg.norm(d.cam_xpos[a] - d.cam_xpos[b]) < 1e-9
        fa = -np.asarray(d.cam_xmat[a]).reshape(3, 3)[:, 2]
        fb = -np.asarray(d.cam_xmat[b]).reshape(3, 3)[:, 2]
        assert float(np.dot(fa, fb)) > 1 - 1e-9      # same forward axis


def test_layout_and_patch_count(fovea_env):
    assert fovea_env.vision_layout == [("wide", 256, 60.0), ("fovea", 288, 24.0)]
    assert fovea_env.patcher.n_total == 1160         # (256 + 324) patches, both eyes


def test_render_flag_decides_which_frames_carry_an_image(fovea_env):
    """The trainer says which frames it will read, so no render is wasted and none stale."""
    zero = np.zeros((24, 9))
    skipped = fovea_env.step(zero, (0.0, 0.0), render=False)
    assert skipped.eye_left is None and skipped.fovea_left is None
    drawn = fovea_env.step(zero, (0.0, 0.0), render=True)
    assert drawn.eye_left is not None and drawn.eye_left.shape == (256, 256, 3)
    assert drawn.fovea_left is not None and drawn.fovea_left.shape == (288, 288, 3)
    assert drawn.fovea_right is not None and drawn.fovea_right.shape == (288, 288, 3)


def test_the_two_foveae_see_the_scene_from_different_places(fovea_env):
    """4.9 cm of baseline has to actually show up as different pixels, or stereo is a fiction."""
    raw = fovea_env.step(np.zeros((24, 9)), (0.0, 0.0), render=True)
    l, r = raw.fovea_left.astype(np.float32), raw.fovea_right.astype(np.float32)
    assert np.abs(l - r).mean() > 0.5


def test_no_fovea_still_works(env):
    """The fovea is optional: without it the feed is the wide pair alone."""
    assert env.vision_layout == [("wide", 256, 60.0)]
    assert env.patcher.n_total == 512
    raw = env.step(np.zeros((24, 9)), (0.0, 0.0), render=True)
    assert raw.eye_left is not None and raw.fovea_left is None


def test_caregiver_puts_the_ball_back_in_view():
    """A ball nobody can see teaches nothing about reaching. Measured need: it was within
    10 degrees of an eye axis on 0.88% of body-steps across every run since H12."""
    e = BridgeEnv(scene="reach", run_dir=RUN_DIR, vision=False, tag="_pytest_care",
                  caregiver_seconds=0.5)          # 50 env steps of 10 ms
    try:
        e.reset()
        far = np.array(e.data.cam_xpos[e.model.camera("eye_left").id]) + np.array([0.0, 0.0, -0.4])
        e.data.mocap_pos[e._target_id] = far
        e._target_home = far.copy()
        seen_unseen, rescued = 0.0, e._rescues
        for _ in range(120):
            raw = e.step(np.zeros((24, 9)), (0.0, 0.0), render=False)
            t = raw.info["task"]
            seen_unseen = max(seen_unseen, t.get("ball_unseen_s", 0.0))
            if e._rescues > rescued:
                break
        assert e._rescues > rescued, f"never rescued; longest unseen stretch {seen_unseen:.2f} s"
        ball = np.array(e.data.mocap_pos[e._target_id])
        cid = e.model.camera("eye_left").id
        fwd = -np.asarray(e.data.cam_xmat[cid]).reshape(3, 3)[:, 2]
        v = ball - np.asarray(e.data.cam_xpos[cid])
        off = np.degrees(np.arccos(np.clip(np.dot(fwd, v / np.linalg.norm(v)), -1, 1)))
        assert off < 30.0, f"put back but still {off:.1f} deg off axis"
        assert 0.2 < np.linalg.norm(v) < 0.5, "put back at an unreachable distance"
    finally:
        e.close()


def test_no_caregiver_by_default_in_the_env():
    e = BridgeEnv(scene="reach", run_dir=RUN_DIR, vision=False, tag="_pytest_nocare")
    try:
        e.reset()
        assert e.caregiver_seconds == 0.0
        for _ in range(30):
            e.step(np.zeros((24, 9)), (0.0, 0.0), render=False)
        assert e._rescues == 0
    finally:
        e.close()


# ------------------------------------------------- the clocks the senses are read on
def test_the_hearing_clock_is_env_dt_and_not_the_construction_value():
    """Every other sense in _collect re-reads env.dt each step - proprio.read(self.dt),
    build_collisions(..., self.dt), mechanical_exposure.update(..., self.dt). Hearing did
    not: CribAudio stored dt at construction and advanced the speech cursor and the
    transient decay by that. The repository's own caregiver tests set env.dt after
    construction, and at 0.1 s against a 0.01 s construction value the speech cursor
    would run at one tenth of the world's rate."""
    e = BridgeEnv(scene="crib", run_dir=RUN_DIR, vision=False, audio_real=True,
                  mobile=True, age=24, tag="_pytest_clock")
    try:
        e.reset()
        assert e._audio is not None and e.dt == 0.01
        e._audio.speech.start(np.full(16000, 0.1, dtype=np.float32),
                              np.asarray(e.data.cam_xpos[e.model.camera("eye_left").id]) + [0.3, 0., 0.])
        e.dt = 0.02                      # what the caregiver tests do after construction
        t0, c0 = e._audio._t, e._audio.speech.cursor
        e.step(np.zeros((24, 9)), (0.0, 0.0), render=False)
        assert e._audio._t - t0 == pytest.approx(0.02), (
            "the hearing clock advanced %.4f s while env.dt says 0.02 s"
            % (e._audio._t - t0))
        assert e._audio.speech.cursor - c0 == 320, (
            "the speech cursor advanced %d samples; 0.02 s at 16000 Hz is 320"
            % (e._audio.speech.cursor - c0))
    finally:
        e.close()


def test_a_sensory_delay_is_refused_because_it_would_reach_nothing():
    """MIMo implements sensory_delay inside _get_obs, and no sense path in BridgeEnv
    consumes what _get_obs returns: _collect reads touch.sensor_outputs, _contacts(),
    proprio.read() and get_vestibular_obs() straight off live MuJoCo data. Any setting
    therefore left the encoders reading undelayed senses.

    That was measured, not assumed: sensory_delay 4 against 0 left proprioception, touch
    and vestibular identical on 12 of 12 steps at a maximum difference of 0.000e+00, while
    the same probe with motor_delay 3 against 0 differed on 11 of 12 - so the zero was a
    real zero and not a blind probe. A knob that silently does nothing is worse than no
    knob, especially one the curriculum rebuilt every body for, so the constructor now
    refuses it. Delivering a real sensory delay means queueing the sense arrays in
    BridgeEnv.step, which is a separate change."""
    with pytest.raises(ValueError, match="sensory_delay reaches nothing"):
        BridgeEnv(scene="crib", run_dir=RUN_DIR, vision=False, audio_real=False,
                  mobile=True, age=24, seed=3, sensory_delay=4, tag="_pytest_delay4")


def test_the_motor_delay_is_not_refused_and_does_reach_the_body():
    """The control for the test above: the other half of the same tuple still works.

    If this ever passes trivially - if the delayed and undelayed bodies stop differing -
    then the motor delay has broken too and the refusal above is no longer evidence of
    anything."""
    kw = dict(scene="crib", run_dir=RUN_DIR, vision=False, audio_real=False,
              mobile=True, age=24, seed=3)
    e0 = BridgeEnv(motor_delay=0, tag="_pytest_mdelay0", **kw)
    e3 = BridgeEnv(motor_delay=3, tag="_pytest_mdelay3", **kw)
    try:
        assert e0.env.motor_delay == 0 and e3.env.motor_delay == 3
        e0.reset(); e3.reset()
        rng = np.random.default_rng(11)
        differed = 0
        for _ in range(12):
            ctrl = rng.uniform(-1, 1, e0.model.nu)
            r0 = e0.step(ctrl=ctrl.copy())
            r3 = e3.step(ctrl=ctrl.copy())
            if float(np.abs(r0.proprio.bones - r3.proprio.bones).max()) > 0.0:
                differed += 1
        assert differed >= 10, (
            "the motor delay reached the body on only %d of 12 steps; the sensory-delay "
            "refusal test is only meaningful while this probe can detect a real delay"
            % differed)
    finally:
        e0.close(); e3.close()
