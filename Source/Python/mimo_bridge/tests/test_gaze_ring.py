# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The far-periphery ring: motion outside the wide cone must snap gaze; motion the
wide eye owns must not come from the ring; when both fields see motion the wide
detector wins. The ring is reflex-only anatomy - these tests drive GazeReflex
directly with synthetic frames, no env needed.
"""
import numpy as np

from mimo_bridge.gaze_reflex import GazeReflex
from mimo_bridge.vision_adapter import patch_grid_angles

EYES = ((0.0, 0.0), (0.0, 0.0))


def _reflex():
    return GazeReflex(size=256, fovy_deg=60.0, dt=0.01, ring_size=128, ring_fovy=150.0)


def _gray(size):
    return np.full((size, size, 3), 60, dtype=np.uint8)


def _cell_for_angle(angles, az_deg, el_deg=0.0):
    d = np.hypot(angles[..., 0] - np.radians(az_deg), angles[..., 1] - np.radians(el_deg))
    return np.unravel_index(int(np.argmin(d)), d.shape)


def _light(img, cell, patch=16, value=250):
    r, c = cell
    img[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = value
    return img


def test_ring_snaps_to_motion_outside_the_wide_cone():
    rf = _reflex()
    ring_angles = patch_grid_angles(128, 128, 150.0, 16).reshape(8, 8, 2)
    cell = _cell_for_angle(ring_angles, 55.0)   # far outside the 30-deg wide half-angle
    assert rf.ring_periphery[cell], "chosen cell must be ring territory"
    rf.observe(_gray(256), EYES, 0.0, frame=0, ring_img=_gray(128))
    rf.observe(_gray(256), EYES, 0.0, frame=5, ring_img=_light(_gray(128), cell))
    assert rf.n_snaps_ring == 1 and rf.n_snaps == 1
    assert rf.active
    # the angle table is in gaze-command azimuth (source-negated 2026-09-06), so the
    # snap target IS the cell's stored angle (the old image-convention table made this
    # same addition mirror every lateral snap - proven by pixels)
    want_az = float(np.clip(ring_angles[cell][0], -0.785, 0.785))
    assert abs(float(rf._target[0]) - want_az) < np.radians(12), \
        f"target az {np.degrees(rf._target[0]):.1f} deg, wanted ~{np.degrees(want_az):.1f}"


def test_ring_ignores_motion_the_wide_eye_owns():
    rf = _reflex()
    ring_angles = patch_grid_angles(128, 128, 150.0, 16).reshape(8, 8, 2)
    cell = _cell_for_angle(ring_angles, 10.0)   # inside the wide cone
    assert not rf.ring_periphery[cell]
    rf.observe(_gray(256), EYES, 0.0, frame=0, ring_img=_gray(128))
    rf.observe(_gray(256), EYES, 0.0, frame=5, ring_img=_light(_gray(128), cell))
    assert rf.n_snaps_ring == 0 and rf.n_snaps == 0


def test_wide_detection_outranks_the_ring():
    rf = _reflex()
    wide_angles = patch_grid_angles(256, 256, 60.0, 16).reshape(16, 16, 2)
    wcell = _cell_for_angle(wide_angles, 20.0)      # wide periphery (fovea mask is 12 deg)
    ring_angles = patch_grid_angles(128, 128, 150.0, 16).reshape(8, 8, 2)
    rcell = _cell_for_angle(ring_angles, 55.0)
    rf.observe(_gray(256), EYES, 0.0, frame=0, ring_img=_gray(128))
    rf.observe(_light(_gray(256), wcell), EYES, 0.0, frame=5,
               ring_img=_light(_gray(128), rcell))
    assert rf.n_snaps == 1 and rf.n_snaps_ring == 0, "the wide field owns the snap"
    want_az = float(wide_angles[wcell][0])   # table already in command azimuth (2026-09-06)
    assert abs(float(rf._target[0]) - want_az) < np.radians(8)


def test_no_ring_built_means_old_behavior():
    rf = GazeReflex(size=256, fovy_deg=60.0, dt=0.01)
    rf.observe(_gray(256), EYES, 0.0, frame=0)
    rf.observe(_gray(256), EYES, 0.0, frame=5)
    assert rf.n_snaps == 0 and rf.n_snaps_ring == 0
