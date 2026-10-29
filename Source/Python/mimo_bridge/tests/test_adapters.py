# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Hand-worked tests for the proprio adapter, action decoder and vision adapter.

These build tiny MuJoCo models by hand (no MIMo needed) so every number can be checked.
"""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from lem.model.body_sets import HANDLE_NAMES, BodySet
from mimo_bridge.proprio_adapter import BONE_MAP, FatigueModel, body_set_for_mimo_body, BODY_SET_ORDER
from mimo_bridge.action_decoder import (quat_xyzw_to_mat, rotmat_log, damped_least_squares, HANDLE_CHAINS)
from mimo_bridge.vision_adapter import patch_grid_angles, image_to_patches, EyePatcher, letterbox_to_bgra


# ------------------------------------------------------------------ bone map
def test_bone_map_covers_all_24_handles_in_order():
    assert list(BONE_MAP.keys()) == list(HANDLE_NAMES)
    assert len(set(BONE_MAP.values())) == 24           # no MIMo body used twice
    assert BONE_MAP["gaze"] == "left_eye"              # the eyes are a felt bone
    # fingers go to the distal (tip) segments, wrists to the hands
    assert BONE_MAP["thumb_L"] == "left_thdistal" and BONE_MAP["wrist_R"] == "right_hand"
    assert all(h in HANDLE_CHAINS for h in HANDLE_NAMES)


def test_body_set_mapping():
    assert body_set_for_mimo_body("left_ffdistal") == BodySet.Hand_L
    assert body_set_for_mimo_body("right_hand") == BodySet.Hand_R
    assert body_set_for_mimo_body("left_upper_arm") == BodySet.Arm_L
    assert body_set_for_mimo_body("right_lower_leg") == BodySet.Leg_R
    assert body_set_for_mimo_body("left_big_toe") == BodySet.Foot_L
    assert body_set_for_mimo_body("head") == BodySet.Head and body_set_for_mimo_body("chest") == BodySet.Spine
    assert body_set_for_mimo_body("mimo_location") is None
    assert [b.value for b in BODY_SET_ORDER] == list(range(10))


def test_fatigue_model_by_hand():
    """dcap/dt = 0.05 (1 - cap) - 0.02 E.  With E = 1 W forever, cap -> 1 - 0.02/0.05 = 0.6.
    With no effort cap stays at 1. Accumulated energy = E * t."""
    f = FatigueModel(n_sets=2)
    for _ in range(1000):
        f.step(np.array([0.0, 1.0]), dt=0.1)        # 100 s
    assert f.cap[0] == pytest.approx(1.0)
    assert f.cap[1] == pytest.approx(0.6, abs=0.01)
    assert f.accum[1] == pytest.approx(100.0)


# ------------------------------------------------------------------ rotations
def test_quat_and_log_by_hand():
    # 90 degrees about z as xyzw
    q = np.array([0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    R = quat_xyzw_to_mat(q)
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)
    assert np.allclose(rotmat_log(R), [0, 0, math.pi / 2], atol=1e-9)
    assert np.allclose(rotmat_log(np.eye(3)), 0)


def test_damped_least_squares_splits_rotation():
    """Two joints with the same axis z, wanted rotation 1 rad about z -> each gets ~0.5 rad.
    One joint about z, another about x, wanted rotation about z only -> x joint gets ~0."""
    A = np.array([[0, 0], [0, 0], [1, 1]], dtype=float)
    dq = damped_least_squares(A, np.array([0, 0, 1.0]), lam=1e-6)
    assert np.allclose(dq, [0.5, 0.5], atol=1e-4)
    A = np.array([[0, 1], [0, 0], [1, 0]], dtype=float)
    dq = damped_least_squares(A, np.array([0, 0, 0.3]), lam=1e-6)
    assert np.allclose(dq, [0.3, 0.0], atol=1e-4)


# ------------------------------------------------------------------ decoder on a tiny model
TINY_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-90 90" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="tip" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" range="-90 90" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="a1" joint="j1" ctrlrange="-1 1" ctrllimited="true" gear="1"/>
    <motor name="a2" joint="j2" ctrlrange="-1 1" ctrllimited="true" gear="1"/>
  </actuator>
</mujoco>
"""


def test_servo_sign_and_split_on_tiny_chain(monkeypatch):
    """Command the tip to be rotated +0.4 rad about z. Both joints share axis z, so each
    should get dq = 0.2 and ctrl = strength * kp * 0.2 = 0.5 * 2 * 0.2 = 0.2 (no damping
    because the joints are still)."""
    import mimo_bridge.action_decoder as AD
    m = mujoco.MjModel.from_xml_string(TINY_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # shrink the 23-handle table to one handle that uses our tiny chain
    monkeypatch.setattr(AD, "HANDLE_NAMES", ("tipH",))
    monkeypatch.setattr(AD, "BONE_MAP", {"tipH": "tip"})
    monkeypatch.setattr(AD, "HANDLE_CHAINS", {"tipH": ("tip", "base")})
    monkeypatch.setattr(AD, "EYE_JOINTS", {})
    dec = AD.ActionDecoder(m, kp=2.0, kd=0.05)
    motor = np.zeros((1, 9))
    ang = 0.4
    motor[0, 3:7] = [0, 0, math.sin(ang / 2), math.cos(ang / 2)]
    motor[0, 8] = 0.5
    ctrl = dec.decode(d, motor)
    assert ctrl.shape == (2,)
    assert ctrl[0] == pytest.approx(0.2, abs=1e-3) and ctrl[1] == pytest.approx(0.2, abs=1e-3)
    # opposite rotation -> negative ctrl; zero strength -> zero ctrl
    motor[0, 3:7] = [0, 0, math.sin(-ang / 2), math.cos(-ang / 2)]
    ctrl = dec.decode(d, motor)
    assert ctrl[0] == pytest.approx(-0.2, abs=1e-3)
    motor[0, 8] = 0.0
    assert np.all(dec.decode(d, motor) == 0)
    # big rotation saturates - but by a DIRECTION-PRESERVING rescale now, not a
    # per-element clip. The old clip flattened every over-rail channel to 1.0,
    # destroying the ratios between joints (measured: 68.7% of demanded magnitude
    # discarded, ~20% of channels delivering exactly nothing). The group is scaled
    # as a unit: its largest channel sits ON the rail, none exceed it, and the
    # ratio between channels survives.
    motor[0, 8] = 1.0
    motor[0, 3:7] = [0, 0, math.sin(1.5 / 2), math.cos(1.5 / 2)]
    ctrl = dec.decode(d, motor)
    assert np.max(np.abs(ctrl)) == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.abs(ctrl) <= 1.0 + 1e-9)


# ------------------------------------------------------------------ the eye servo
# One (azimuth, elevation) pair drives four eye actuators. Until this test the only
# ActionDecoder test in the repository monkeypatched EYE_JOINTS to {} before building
# the decoder, so the eye path - the mechanism behind the gaze box's "one command,
# both eyes" - was never run by anything. A sign error on the right eye would cross or
# splay the eyes with no test failing, and gaze_reflex.py:150-157 records an azimuth
# convention bug that already drove the eye the wrong way once.
EYE_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="head" pos="0 0 0">
      <joint name="neck" type="hinge" axis="0 0 1" range="-1.57 1.57" limited="true"/>
      <geom type="sphere" size="0.05"/>
      <body name="left_eye" pos="0.03 0.02 0">
        <joint name="robot:left_eye_horizontal" type="hinge" axis="0 0 1"
               range="-0.785 0.785" limited="true"/>
        <joint name="robot:left_eye_vertical" type="hinge" axis="0 1 0"
               range="-0.820 0.576" limited="true"/>
        <geom type="sphere" size="0.01"/>
      </body>
      <body name="right_eye" pos="0.03 -0.02 0">
        <joint name="robot:right_eye_horizontal" type="hinge" axis="0 0 -1"
               range="-0.785 0.785" limited="true"/>
        <joint name="robot:right_eye_vertical" type="hinge" axis="0 1 0"
               range="-0.820 0.576" limited="true"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="a_neck" joint="neck" ctrlrange="-1 1" ctrllimited="true" gear="1"/>
    <motor name="a_lh" joint="robot:left_eye_horizontal" ctrlrange="-1 1" ctrllimited="true" gear="1"/>
    <motor name="a_lv" joint="robot:left_eye_vertical" ctrlrange="-1 1" ctrllimited="true" gear="1"/>
    <motor name="a_rh" joint="robot:right_eye_horizontal" ctrlrange="-1 1" ctrllimited="true" gear="1"/>
    <motor name="a_rv" joint="robot:right_eye_vertical" ctrlrange="-1 1" ctrllimited="true" gear="1"/>
  </actuator>
</mujoco>
"""


def _eye_decoder(monkeypatch):
    """A decoder on a four-eye-joint body, with EYE_JOINTS left alone."""
    import mimo_bridge.action_decoder as AD
    m = mujoco.MjModel.from_xml_string(EYE_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    monkeypatch.setattr(AD, "HANDLE_NAMES", ("headH",))
    monkeypatch.setattr(AD, "BONE_MAP", {"headH": "head"})
    monkeypatch.setattr(AD, "HANDLE_CHAINS", {"headH": ("head",)})
    dec = AD.ActionDecoder(m, kp=2.0, kd=0.3, gaze_kp=1.0, gaze_kd=0.3)
    return m, d, dec


def _eye_act(m, name):
    return int(m.actuator(name).id)


def test_one_gaze_command_moves_both_eyes_with_the_right_one_mirrored(monkeypatch):
    """One (az, el) pair reaches four eye actuators; the two horizontal commands have
    opposite sign (the right eye's hinge axis is 0 0 -1 in MIMo_modelv2.xml), the two
    vertical commands share a sign, and no limb actuator is touched."""
    m, d, dec = _eye_decoder(monkeypatch)
    assert len(dec.eye) == 4, "the decoder wired fewer than four eye actuators"
    motor = np.zeros((1, 9))
    motor[0, 6] = 1.0                    # identity quaternion
    motor[0, 8] = 0.0                    # no limb strength: only the eye path writes
    ctrl = dec.decode(d, motor, gaze=(0.30, 0.20))
    lh, lv, rh, rv = (_eye_act(m, n) for n in ("a_lh", "a_lv", "a_rh", "a_rv"))
    assert np.count_nonzero(ctrl) == 4, "one (az, el) pair should drive four actuators"
    assert ctrl[_eye_act(m, "a_neck")] == 0.0
    assert ctrl[lh] > 0 and ctrl[rh] < 0, "the horizontal pair must be mirrored"
    assert ctrl[lv] > 0 and ctrl[rv] > 0, "the vertical pair must share a sign"
    # gaze_kp 1.0 from a still start: ctrl = gaze_kp * (sign * target - q), q = 0
    assert ctrl[lh] == pytest.approx(0.30, abs=1e-9)
    assert ctrl[rh] == pytest.approx(-0.30, abs=1e-9)
    assert ctrl[lv] == pytest.approx(0.20, abs=1e-9)
    # azimuth flips -> both horizontals flip, the verticals do not
    ctrl2 = dec.decode(d, motor, gaze=(-0.30, 0.20))
    assert ctrl2[lh] == pytest.approx(-ctrl[lh], abs=1e-9)
    assert ctrl2[rh] == pytest.approx(-ctrl[rh], abs=1e-9)
    assert ctrl2[lv] == pytest.approx(ctrl[lv], abs=1e-9)


def test_gaze_target_is_clipped_to_each_eye_joint_range(monkeypatch):
    """A command past an eye's mechanical range aims at the range edge, so the servo
    does not keep pressing into the stop. Azimuth range +-0.785 rad."""
    m, d, dec = _eye_decoder(monkeypatch)
    motor = np.zeros((1, 9))
    motor[0, 6] = 1.0
    far = dec.decode(d, motor, gaze=(5.0, 0.0)).copy()
    edge = dec.decode(d, motor, gaze=(0.785, 0.0)).copy()
    lh, rh = _eye_act(m, "a_lh"), _eye_act(m, "a_rh")
    assert far[lh] == pytest.approx(edge[lh], abs=1e-9)
    assert far[rh] == pytest.approx(edge[rh], abs=1e-9)
    assert far[lh] == pytest.approx(0.785, abs=1e-9)
    assert np.all(np.abs(far) <= 1.0 + 1e-9), "eye ctrl must stay inside the rails"


# ------------------------------------------------------------------ vision
def test_patch_angles_by_hand():
    """256x256 image, fovy 60: tan(30) = 0.57735. Centre patches (cols 7,8) are at
    +-(1/16)*tan30 -> atan(0.03608) = 2.066 deg. GAZE-COMMAND azimuth since
    4325142fc (positive = eye-left = image content LEFT of centre): the far-left
    column centre reads +28.42 deg. This test asserted the old retinal sign for
    three days after the flip. Top row has positive elevation (unchanged). Since audit
    P026 (2026-09-18) the elevation is the exact ray elevation asin(y/|ray|): the
    top-LEFT patch (ang[0]) reads 25.455 deg, not atan(y) = 28.425 - only the centre
    column keeps the old value."""
    ang = patch_grid_angles(256, 256, 60.0)
    assert ang.shape == (256, 2)
    row7 = ang[7 * 16:(7 * 16) + 16]
    assert row7[0, 0] == pytest.approx(math.atan((15 / 16) * math.tan(math.radians(30))), abs=1e-6)
    assert math.degrees(row7[0, 0]) == pytest.approx(28.42, abs=0.01)
    assert math.degrees(row7[7, 0]) == pytest.approx(2.066, abs=0.01)
    assert math.degrees(row7[8, 0]) == pytest.approx(-2.066, abs=0.01)
    assert ang[0, 1] > 0 and ang[-1, 1] < 0                      # top row up, bottom row down
    assert math.degrees(ang[0, 1]) == pytest.approx(25.455, abs=0.01)
    assert math.degrees(ang[7, 1]) == pytest.approx(28.41, abs=0.02)      # centre column: unchanged


def test_image_to_patches_layout():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[16:32, 0:16, 0] = 255           # bottom-left patch, red channel full
    p = image_to_patches(img)
    assert p.shape == (4, 768)
    # patch index 2 = row 1, col 0 (row-major); red channel is every 3rd value starting at 0
    assert p[2, 0::3].min() == 1.0 and p[2, 1::3].max() == 0.0
    assert p[0].max() == 0.0 and p[1].max() == 0.0 and p[3].max() == 0.0
    ep = EyePatcher(32, 32, 60.0)
    patches, angles = ep(img, img, (0.1, 0.0), (-0.1, 0.0))
    assert patches.shape == (8, 768) and angles.shape == (8, 2)
    assert angles[0, 0] - angles[4, 0] == pytest.approx(0.2, abs=1e-6)   # eye yaw offset applied


def test_letterbox_bgra():
    img = np.zeros((4, 4, 3), dtype=np.uint8); img[..., 0] = 200   # red
    out = letterbox_to_bgra(img, 10, 8)
    assert out.shape == (8, 10, 4)
    assert out[2, 3, 2] == 200 and out[2, 3, 0] == 0 and out[2, 3, 3] == 255   # R lands in the BGRA R slot
    assert out[0, 0, 2] == 0


# ------------------------------------------------------------------ crib hearing
def test_the_crib_hearing_clock_follows_the_caller_not_the_construction_dt():
    """CribAudio stored dt once at construction and advanced the speech cursor and the
    transient decay by that stored value, while the caregiver state machine reads env.dt
    live every tick. Nothing asserted the two agree, and the repository's own caregiver
    tests set env.dt after construction (test_caregiver_exchange.py sets e.dt=.1). The
    caller now owns the clock; omitting dt keeps every existing caller bit-identical."""
    from mimo_bridge.audio_adapter import CribAudio
    head = np.zeros(3); right = np.array([1., 0., 0.]); fwd = np.array([0., 1., 0.])

    a = CribAudio(0.01)
    a.speech.start(np.full(16000, 0.1, dtype=np.float32), [0.2, 0., 0.])
    a.render(head, right, fwd, dt=0.02)
    assert a.speech.cursor == 320, (
        "speech advanced %d samples for a 0.02 s step at 16000 Hz; 320 is one step of "
        "the caller's clock" % a.speech.cursor)
    assert a._t == pytest.approx(0.02)

    b = CribAudio(0.01)
    b.speech.start(np.full(16000, 0.1, dtype=np.float32), [0.2, 0., 0.])
    b.render(head, right, fwd)
    assert b.speech.cursor == 160 and b._t == pytest.approx(0.01), (
        "omitting dt must still use the construction value")

    # passing the construction value explicitly is bit-identical to omitting it
    c = CribAudio(0.01); e = CribAudio(0.01)
    c.squeak(np.array([0.3, 0., 0.])); e.squeak(np.array([0.3, 0., 0.]))
    spec_c, meta_c = c.render(head, right, fwd)
    spec_e, meta_e = e.render(head, right, fwd, dt=0.01)
    np.testing.assert_array_equal(spec_c, spec_e)
    assert meta_c == meta_e


def test_the_two_ear_loudness_producers_agree_on_what_a_decibel_is():
    """One field carries the left-right loudness difference and two producers write it:
    the caregiver's speech (speech.py) and every synthesized source (audio_adapter.py).
    Until 6c1d160cc they disagreed twice over - 10*log10 against 20*log10, and a
    head-centre distance against per-ear distances - so the same source read -1.8709 dB
    on the adapter path against -3.7417 to -6.1899 dB on the speech path, a 2.00x to
    3.31x rescale depending on which source happened to be loudest that frame.

    Two things are pinned. Where both ear distances sit at or under the 0.2 m
    attenuation floor the distance term cancels on both paths and the two must agree
    exactly. Away from the floor a residual remains, because speech attenuates as 1/r
    (speech.py) and the adapter as 1/r^1.5 (a deliberate choice, audio_adapter.py) - it
    measures 1.2241 dB at most over 0.05 to 1.5 m at any azimuth, against the 2.00x to
    3.31x the two log conventions used to cost.
    """
    from mimo_bridge.audio_adapter import CribAudio
    from mimo_bridge.speech import SpeechStream
    freqs = np.geomspace(40., 8000., 256)
    head = np.zeros(3); right = np.array([1., 0., 0.]); fwd = np.array([0., 1., 0.])

    def both(pos):
        st = SpeechStream(); st.start(np.full(4000, .1, dtype=np.float32), [0, 0, 0])
        st.position = np.asarray(pos, float)
        _, sm = st.render(.01, head, right, fwd, freqs)
        ca = CribAudio(0.01); ca.squeak(np.asarray(pos, float))
        _, am = ca.render(head, right, fwd)
        return sm["ild_db"], am["ild_db"]

    for dist in (0.05, 0.10, 0.15):
        s, a = both([dist, 0., 0.])
        assert s == pytest.approx(a, abs=1e-9), (
            "inside the 0.2 m attenuation floor the two paths must be identical; "
            "%.2f m gives speech %.4f dB against adapter %.4f dB" % (dist, s, a))
        assert s == pytest.approx(-3.7417, abs=1e-3), (
            "20*log10 of the 0.65 head-shadow ratio is -3.7417 dB; 10*log10 gives -1.8709")

    worst = max(abs(s - a) for s, a in
                (both([d * np.cos(t), d * np.sin(t), 0.])
                 for d in np.linspace(0.05, 1.5, 15) for t in np.linspace(-np.pi, np.pi, 9)))
    assert worst < 1.3, "the two paths drifted to %.4f dB apart (1.2241 measured)" % worst

    # a synthesized source must carry distance at all: with one head-centre attenuation
    # the 1/r term cancelled between the ears and every distance read the same number
    vals = [round(both([d, 0., 0.])[1], 4) for d in (0.15, 0.25, 0.338, 1.2)]
    assert len(set(vals)) == 4, (
        "a non-speech source reports %s - the same loudness difference at every "
        "distance is the head-centre attenuation defect" % (vals,))


# ------------------------------------------------------------------ the patch direction tag
def _reference_direction(size, fovy, yaw_deg, pitch_deg):
    """Independent of the production code: unit ray (forward, left, up) per patch centre,
    rotated by R_up(yaw) @ R_left(pitch) with plain 3x3 matrices; azimuth = atan2(left,
    forward), elevation = asin(up)."""
    from mimo_bridge.vision_adapter import PATCH
    ty = math.tan(math.radians(fovy) / 2.0)
    rows = cols = size // PATCH
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    R_left = np.array([[math.cos(pitch), 0.0, -math.sin(pitch)],
                       [0.0, 1.0, 0.0],
                       [math.sin(pitch), 0.0, math.cos(pitch)]])
    R_up = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                     [math.sin(yaw), math.cos(yaw), 0.0],
                     [0.0, 0.0, 1.0]])
    out = np.zeros((rows * cols, 2))
    for r in range(rows):
        for c in range(cols):
            x = (2.0 * (c * PATCH + PATCH / 2.0) / size - 1.0) * ty
            y = (1.0 - 2.0 * (r * PATCH + PATCH / 2.0) / size) * ty
            v = np.array([1.0, -x, y]); v = v / np.linalg.norm(v)
            w = R_up @ (R_left @ v)
            out[r * cols + c] = (math.atan2(w[1], w[0]), math.asin(w[2]))
    return out


def test_the_patch_direction_tag_is_the_composed_direction():
    """Audit P026: the (azimuth, elevation) tag on every vision patch is the patch's own
    ray turned by the eye's two joints, referenced to the head - within 0.01 of a patch
    span at both cameras and at nine gaze pairs spanning the eye joint's own limits
    (horizontal -45..45, vertical -47..33 degrees in the scene XML). The old arithmetic
    (base + (yaw, pitch)) must fail the same tolerance so nobody reinstates it."""
    from mimo_bridge.vision_adapter import BinocularPatcher, PATCH
    gazes = [(0, 0), (0, 33), (0, -47), (45, 0), (-45, 0), (45, 33), (-45, -47), (20, 10), (-20, 25)]
    for size, fovy in ((256, 60.0), (288, 24.0)):
        span = math.radians(fovy) * PATCH / size
        bp = BinocularPatcher([("cam", size, fovy)])
        img = np.zeros((size, size, 3), dtype=np.uint8)
        worst = (0.0, None)
        for yaw_deg, pitch_deg in gazes:
            _p, a = bp({"cam": (img, img)}, (math.radians(yaw_deg), math.radians(pitch_deg)), (0.0, 0.0))
            got = a[0::2, :2].astype(np.float64)                     # left eye rows
            ref = _reference_direction(size, fovy, yaw_deg, pitch_deg)
            daz = np.abs((got[:, 0] - ref[:, 0] + np.pi) % (2 * np.pi) - np.pi)
            gap = float(max(daz.max(), np.abs(got[:, 1] - ref[:, 1]).max()))
            if gap > worst[0]:
                worst = (gap, (yaw_deg, pitch_deg))
        assert worst[0] < 0.01 * span, (
            "worst gap %.4f deg = %.3f patches at gaze %s (camera %d px / %g deg)"
            % (math.degrees(worst[0]), worst[0] / span, worst[1], size, fovy))
    # regression guard: the old additive arithmetic is more than one wide patch out at pitch 25
    base = patch_grid_angles(256, 256, 60.0).astype(np.float64)
    added = base + np.array([0.0, math.radians(25.0)])
    ref = _reference_direction(256, 60.0, 0.0, 25.0)
    wide_span = math.radians(60.0) * PATCH / 256
    assert float(np.abs(added[:, 0] - ref[:, 0]).max()) > wide_span, "adding the pitch would pass; it must not"


def test_the_patch_elevation_is_the_ray_elevation_not_its_tangent():
    from mimo_bridge.vision_adapter import PATCH
    a = patch_grid_angles(256, 256, 60.0)
    ty = math.tan(math.radians(60.0) / 2.0)
    rows = cols = 256 // PATCH
    for r in range(rows):
        for c in range(cols):
            x = (2.0 * (c * PATCH + PATCH / 2.0) / 256 - 1.0) * ty
            y = (1.0 - 2.0 * (r * PATCH + PATCH / 2.0) / 256) * ty
            assert abs(float(a[r * cols + c, 1]) - math.asin(y / math.sqrt(1.0 + x * x + y * y))) < 1e-6
    assert math.degrees(a[0, 1]) == pytest.approx(25.455, abs=0.01)
    assert math.degrees(a[0, 1]) != pytest.approx(28.425, abs=0.01)
