# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Unit tests for the snap-to-motion gaze reflex (gaze_reflex.py).

Pure numpy - no MuJoCo, no GL. Synthetic 256x256 frames: a still scene, a blob that
jumps in the periphery, a global brightness shift, a blob in the fovea's territory.
The invariants tested are the circuit's defining properties: peripheral transients
snap, global change and central change do not, a moving eye suppresses detection,
one snap is followed by quiet (refractory), and a spot that keeps flickering loses
priority (habituation).
"""
import numpy as np
import pytest

from mimo_bridge.gaze_reflex import GazeReflex, AZ_RANGE, EL_RANGE

SIZE, FOVY, DT = 256, 60.0, 0.01
EYES = ((0.0, 0.0), (0.0, 0.0))   # both eyes straight ahead


def make_reflex(**kw):
    return GazeReflex(size=SIZE, fovy_deg=FOVY, dt=DT, **kw)


def frame(gray: int = 60) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), gray, dtype=np.uint8)


def with_blob(img: np.ndarray, cx: int, cy: int, r: int = 14, val: int = 220) -> np.ndarray:
    out = img.copy()
    y, x = np.ogrid[:SIZE, :SIZE]
    out[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = val
    return out


def test_peripheral_transient_snaps_toward_it():
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    # blob appears upper-left periphery. The angle table carries gaze-command azimuth
    # (positive = eye-left, source-negated 2026-09-06), so an image-LEFT blob takes a
    # POSITIVE (leftward) command. The pre-fix assertion pinned the mirror (proven by
    # pixels: it drove the eye AWAY from every lateral transient).
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=1)
    assert rf.n_snaps == 1 and rf.active
    g = rf.apply(np.zeros(2))
    assert g[0] > 0.1, "an image-left blob takes a positive (leftward) eye command"
    assert g[1] > 0.1, "blob in the top of the image must pull elevation positive"
    assert AZ_RANGE[0] <= g[0] <= AZ_RANGE[1] and EL_RANGE[0] <= g[1] <= EL_RANGE[1]


def test_global_shift_does_not_snap():
    rf = make_reflex()
    rf.observe(frame(60), EYES, 0.0, frame=0)
    rf.observe(frame(110), EYES, 0.0, frame=1)   # whole scene brightens (head moved, light)
    assert rf.n_snaps == 0


def test_central_motion_is_the_foveas_job():
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    rf.observe(with_blob(frame(), cx=SIZE // 2, cy=SIZE // 2, r=10), EYES, 0.0, frame=1)
    assert rf.n_snaps == 0


def test_moving_eye_suppresses_detection():
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, eye_speed=3.0, frame=1)
    assert rf.n_snaps == 0   # efference copy says the smear is self-made


def test_refractory_one_snap_then_quiet():
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=1)
    assert rf.n_snaps == 1
    # a second transient right behind the first: inside the refractory window
    rf.observe(with_blob(frame(), cx=220, cy=220), EYES, 0.0, frame=3)
    assert rf.n_snaps == 1
    # well after the refractory (0.4 s / 0.01 s = 40 frames) it fires again
    for _ in range(rf.snap_hold_steps):
        rf.apply(None)
    rf.observe(frame(), EYES, 0.0, frame=60)
    rf.observe(with_blob(frame(), cx=220, cy=220), EYES, 0.0, frame=61)
    assert rf.n_snaps == 2


def test_habituation_a_flickering_spot_loses_priority():
    rf = make_reflex()
    fr = 0
    snaps_before = 0
    # same spot flashes every 50 frames (outside refractory each time); the two frames of
    # each flash are adjacent renders, the way _collect actually feeds the reflex
    for k in range(30):
        rf.observe(frame(), EYES, 0.0, frame=fr)
        rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=fr + 1)
        fr += 50
        for _ in range(rf.snap_hold_steps):
            rf.apply(None)
        if k == 4:
            snaps_before = rf.n_snaps
    assert snaps_before >= 1, "the first flashes must snap"
    assert rf.n_snaps < 30, "a spot that flickers forever must stop capturing gaze"


def test_apply_blends_and_passes_through():
    rf = make_reflex()
    assert rf.apply(None) is None                       # nothing pending, None stays None
    g0 = np.array([0.1, -0.1])
    assert np.allclose(rf.apply(g0), g0)                # nothing pending, passthrough
    rf.observe(frame(), EYES, 0.0, frame=0)
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=1)
    tgt = rf._target.copy()
    g = rf.apply(g0)
    assert np.allclose(g, 0.3 * g0 + 0.7 * tgt)         # the blend, w = 0.7
    g_none = rf.apply(None)
    assert np.allclose(g_none, tgt)                     # model silent -> reflex speaks alone
    for _ in range(rf.snap_hold_steps):
        rf.apply(g0)
    assert not rf.active                                 # the timeout runs out


def test_stale_frame_pair_is_ignored():
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    # 200 frames later (> max_frame_gap): a diff against a stale image proves nothing
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=200)
    assert rf.n_snaps == 0


# ------------------------------------------------------------------ review fixes 2026-08-24
def test_the_snap_holds_until_the_eye_arrives():
    """The defect this replaces: the snap lasted three env steps = 30 ms, and the eye plant
    cannot cross a saccade in 30 ms. Measured on the real body, a step with a snap pending
    moved the eyes 0.0039 deg against 0.0042 with none - the reflex counted snaps and moved
    nothing. It now steers until the eye is there."""
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=1)
    tgt = rf._target.copy()
    far = ((0.0, 0.0), (0.0, 0.0))
    for _ in range(10):                       # ten steps of a still eye: still steering
        rf.apply(np.zeros(2), far)
    assert rf.active, "gave up before the eye moved at all"
    at = ((float(tgt[0]), float(tgt[1])), (float(tgt[0]), float(tgt[1])))
    rf.apply(np.zeros(2), at)                 # eye reports it has arrived
    assert not rf.active, "kept steering after arrival"


def test_the_snap_gives_up_eventually():
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=1)
    stuck = ((0.0, 0.0), (0.0, 0.0))
    for _ in range(rf.snap_hold_steps + 1):
        rf.apply(np.zeros(2), stuck)
    assert not rf.active, "an eye that never arrives must not be steered forever"


def test_habituation_follows_the_world_not_the_retina():
    """It used to be indexed by retinal patch, so after the eye moved the suppression sat
    at the old image address and muted whatever landed there instead."""
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=1)
    assert rf.n_snaps == 1
    az, el = rf._target
    # the same WORLD direction is still habituated when the eye is somewhere else
    assert rf._decayed(rf._hab_key(float(az), float(el)), 1) > 0.5
    # a direction 30 degrees away is not
    assert rf._decayed(rf._hab_key(float(az) + np.radians(30), float(el)), 1) == 0.0


def test_habituation_recovers_in_seconds_not_a_minute():
    """It used to add the raw uint8 difference - 50 units for one bright transient - and
    decay 0.995 per observe, which is 78 s back to baseline, not the couple of seconds the
    docstring promised."""
    rf = make_reflex()
    rf.observe(frame(), EYES, 0.0, frame=0)
    rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=1)
    key = rf._hab_key(float(rf._target[0]), float(rf._target[1]))
    assert rf._decayed(key, 1) == pytest.approx(1.0, abs=0.01)
    after_10s = rf._decayed(key, 1 + int(10.0 / rf.dt))
    assert after_10s < 0.05, f"still {after_10s:.3f} habituated after ten seconds"


def test_render_cadence_does_not_silently_disable_the_reflex():
    """max_frame_gap was 10 FRAMES while renders land action_repeat frames apart, so
    action_repeat above 10 rejected every pair and the reflex went quiet with no error."""
    for repeat in (5, 10, 12, 20):
        rf = make_reflex()
        rf.observe(frame(), EYES, 0.0, frame=0)
        rf.observe(with_blob(frame(), cx=40, cy=40), EYES, 0.0, frame=repeat)
        assert rf.n_snaps == 1, f"silent at action_repeat {repeat}"
