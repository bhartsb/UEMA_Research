# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Snap-to-motion gaze reflex: endowed anatomy, body-side, permanent.

A newborn orients its eyes toward peripheral motion from birth - the superior
colliculus does it, subcortically, before any learning. This module is that circuit
for the bridge body. It lives in the env worker, never in LEM: no parameters train,
no loss sees it, and the model learns to work with (and around) it the same way it
works with PCC tone under the physics handles. Adults keep this reflex for life, so
the weight does NOT fade with competence.

Parameters come from the biology of the circuit, not from tuning for an outcome:
  * PERIPHERY ONLY - collicular receptive fields are peripheral; the central patches
    (the fovea's job) are masked out of detection.
  * SACCADE SUPPRESSION - a moving eye smears the whole image; detection is gated
    off while the eye joints are moving fast (efference copy: we read qvel, we do
    not infer it from the image).
  * LOCALIZED TRANSIENTS - the median patch difference is subtracted, so a global
    brightness/registration shift (head motion the velocity gate missed) does not
    trigger; only a spot that changed MORE than the scene did can win.
  * REFRACTORY + HABITUATION - one snap, then ~0.4 s of quiet; a spot that keeps
    flickering raises its own threshold and loses priority (a mobile does not
    capture gaze forever).

The detector runs on the wide-field left eye image at patch resolution (the same
16 px patches the encoder reads), costs one uint8 diff + a few reductions, and
produces a gaze target in the SAME command space the model's gaze head uses:
(azimuth, elevation) radians added to the current eye position, clamped to the real
eye joint ranges. The snap itself blends the model's gaze command with the target
for a few steps: gaze = (1-w) * model_gaze + w * target.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mimo_bridge.vision_adapter import patch_grid_angles, patch_grid_rays, compose_gaze

# real eye joint ranges (MIMo_modelv2.xml, same clamps as the babble in lem_model.py)
AZ_RANGE = (-0.785, 0.785)     # +/- 45 deg
EL_RANGE = (-0.820, 0.576)     # -47 to +33 deg

# THE EYES' ROW CONVENTION (motor interface audit 2026-09-25, finding F1; train_mimo
# --eye-row-convention). The learner's eye row is a quaternion whose yaw is the horizontal eye
# angle and whose pitch carries the elevation the eye servo is sent (the servo sets the vertical
# eye joint to plus that elevation: action_decoder.EYE_JOINTS). MIMo's vertical eye joint turns
# about (0, -1, 0) (MIMo_modelv2.xml, robot:left_eye_vertical), so the eye's PHYSICAL
# head-relative rotation - the achieved eye row proprioception reports (proprio_adapter.read:
# conj(head) x left eye) - has pitch = MINUS the vertical joint angle.
#   legacy    order row pitch = PLUS the elevation sent: every run up to the deploy. The ordered
#             and the achieved eye rows read the vertical angle with opposite signs, so the motor
#             lesson grades a correctly executed elevation as its mirror image (on 3,040 recorded
#             poses the two readings differ by a median 12.7 degrees, twice the vertical angle).
#   physical  order row pitch = MINUS the elevation sent: the order, the sense and the servo share
#             the eye's physical rotation (tests/test_eye_row_convention.py checks it on the
#             compiled crib model against ProprioAdapter.read).
EYE_ROW_CONVENTIONS = ("legacy", "physical")


def eye_pitch_sign(convention: str = "legacy") -> float:
    """+1.0 (legacy) or -1.0 (physical): an eye row's pitch = this sign x the elevation sent to the
    eye servo. Refuses any other name, so a mistyped convention can never pass as one of them."""
    if convention == "legacy":
        return 1.0
    if convention == "physical":
        return -1.0
    raise ValueError("eye row convention must be one of %s, not %r" % (EYE_ROW_CONVENTIONS, convention))


class GazeReflex:
    """Peripheral-motion snap for one body. observe() on every fresh render,
    apply() on every step's gaze command."""

    def __init__(self, size: int, fovy_deg: float, dt: float, patch: int = 16,
                 mask_deg: float = 16.0, weight: float = 0.7, k_median: float = 4.0,
                 abs_min: float = 5.0, snap_hold_s: float = 0.30, arrive_deg: float = 3.0,
                 refractory_s: float = 0.4, eye_speed_gate: float = 0.7,
                 max_gap_s: float = 0.25, habituation_tau_s: float = 2.5,
                 habituation_grid_deg: float = 6.0,
                 ring_size: int = 0, ring_fovy: float = 0.0):
        """size/fovy_deg: the wide eye camera. dt: env step seconds. mask_deg: full angle
        of the central region excluded from detection (the fovea's territory).
        eye_speed_gate: rad/s of eye joint speed above which detection is suppressed.
        MEASURED, not chosen: eye speed runs to 0.643 rad/s at the 95th percentile when no
        snap is steering and has a median of 0.756 while one is, so 0.7 sits between them.
        It was 1.5, picked while the eyes were frozen by the armature bug - above the 1.235
        the eye can actually reach, so saccade suppression never once fired and the reflex
        could trigger on the smear from its own movement (calibrate.py flagged it NEVER).
        abs_min = 5.0, from measurement (2026-08-24). Its job is separating still-scene
        noise from real fast events; a slowly drifting ball is genuinely sub-threshold
        (its peak patch change overlaps settling noise entirely, measured: drift p50 1.8
        vs nothing-happening p99 2.7 - no separation exists, so no floor can detect
        drift). The distributions that CAN be separated: nothing-happening max 2.91 vs
        the smallest live event peak 8.8. Floor sits at their geometric middle.
        abs_min: minimum mean |uint8 diff| a patch needs on top of beating the median -
        rejects sensor-noise-level ripples in an otherwise still scene.
        snap_hold_s: how long a snap keeps steering before it gives up. The command is
        released as soon as the eye is within arrive_deg of the target, so this is a
        timeout, not a duration - see apply()."""
        self.dt = float(dt)
        self.rows = self.cols = size // patch
        self.angles = patch_grid_angles(size, size, fovy_deg, patch).reshape(self.rows, self.cols, 2)
        # the same cells as unit rays, for composing the snap target with the eye's gimbal
        # (audit P026): the angle table is for masks and habituation cells, the ray table
        # for "which gimbal command centres this cell"
        self.rays = patch_grid_rays(size, size, fovy_deg, patch).reshape(self.rows, self.cols, 3)
        self.patch = patch
        self.weight = float(weight)
        self.k_median = float(k_median)
        self.abs_min = float(abs_min)
        self.snap_hold_steps = max(1, int(round(snap_hold_s / dt)))
        self.arrive_rad = np.radians(float(arrive_deg))
        self.refractory_steps = max(1, int(round(refractory_s / dt)))
        self.eye_speed_gate = float(eye_speed_gate)
        self.max_gap_steps = max(1, int(round(max_gap_s / dt)))
        self.hab_tau_s = float(habituation_tau_s)
        self.hab_cell = np.radians(float(habituation_grid_deg))
        self.hab_sigma = np.radians(float(habituation_grid_deg)) * 1.5
        # central mask: patches whose angle from the axis is inside mask_deg/2
        half = np.radians(mask_deg) / 2.0
        ang = self.angles
        self.periphery = (np.abs(ang[..., 0]) > half) | (np.abs(ang[..., 1]) > half)
        # Far-periphery ring (2026-09-05, operator plan): a rod-like second field.
        # The wide eye covers 60 degrees; measured, the body's own hands sit at a mean
        # 58 degrees off the camera axis and are inside the wide cone only 10% of the
        # time - the busiest self-caused mover is invisible as motion. The ring is a
        # coarse extra camera (~150 degrees at 128 px, 0.85 px/deg - too blurry to
        # recognise, enough to notice) whose ONLY reader is this reflex: the model
        # never sees its pixels, so no lesson and no input distribution changes.
        # The wide detector owns everything inside the wide cone; the ring detects
        # only outside it, at lower priority, sharing the same world-angle
        # habituation so a looked-at direction is boring in both fields.
        self.ring_rows = self.ring_cols = (ring_size // patch) if ring_size else 0
        self._prev_ring: Optional[np.ndarray] = None
        self._prev_ring_frame: int = -10**9
        self.n_snaps_ring = 0
        if self.ring_rows:
            self.ring_angles = patch_grid_angles(ring_size, ring_size, ring_fovy,
                                                 patch).reshape(self.ring_rows, self.ring_cols, 2)
            self.ring_rays = patch_grid_rays(ring_size, ring_size, ring_fovy,
                                             patch).reshape(self.ring_rows, self.ring_cols, 3)
            wide_half = np.radians(fovy_deg) / 2.0
            rang = self.ring_angles
            self.ring_periphery = (np.abs(rang[..., 0]) > wide_half) | (np.abs(rang[..., 1]) > wide_half)
        # state
        self._prev: Optional[np.ndarray] = None      # (rows, cols) float32 patch means
        self._prev_frame: int = -10**9
        # Habituation lives in WORLD angle, not retina. Keyed by a coarse (azimuth,
        # elevation) cell of the gaze command space, so a spot that has just been looked at
        # stays habituated after the eye moves - which is what "a flickering spot loses
        # priority" has to mean. Keyed by retinal patch it followed the eye instead, and
        # suppressed whatever happened to land where the last thing was.
        self._hab: dict = {}                         # (cell_az, cell_el) -> (strength, frame)
        self._target: Optional[np.ndarray] = None    # (2,) az, el gaze command
        self._steps_left = 0                         # timeout countdown, not a duration
        self._quiet_until = -10**9                   # frame number the refractory ends at
        self.n_snaps = 0
        self.n_observed = 0

    # ------------------------------------------------------------------ detection
    def _pool(self, img: np.ndarray, rows: Optional[int] = None) -> np.ndarray:
        """(H, W, 3) uint8 -> (rows, rows) float32 grayscale patch means."""
        rows = self.rows if rows is None else rows
        p = self.patch
        g = img[: rows * p, : rows * p].astype(np.float32).mean(axis=2)
        return g.reshape(rows, p, rows, p).mean(axis=(1, 3))

    def _detect(self, cur, prev, periphery, angles, yaw: float, pitch: float, frame: int):
        """One field's transient detection. Returns (az, el) world gaze target or None.
        exponential habituation, not a divisor: dividing by (1 + h) with h capped near 5
        left a bright repeated flash far above threshold - it captured gaze on 30 of 30
        flashes. exp(-h) suppresses a saturated spot below the absolute floor while a few
        seconds of quiet brings it back."""
        diff = np.abs(cur - prev)
        diff = diff - np.median(diff)                # localized transients only
        diff = np.where(periphery, diff, 0.0)        # collicular fields are peripheral
        eff = diff * np.exp(-self._hab_map(yaw, pitch, frame, angles))
        r, c = np.unravel_index(int(np.argmax(eff)), eff.shape)
        peak = float(eff[r, c])
        floor = float(np.median(np.abs(diff[periphery]))) if periphery.any() else 0.0
        if peak < self.abs_min or peak < self.k_median * max(floor, 1e-3):
            return None
        # target = the gimbal command that centres the transient's cell: the cell's own
        # ray turned by the eye's current (yaw, pitch) - COMPOSED, not added (audit P026,
        # 2026-09-18). Addition was exact only with a level eye; at 25 deg of pitch it
        # named an azimuth 10.194 deg (2.72 wide patches) off the cell, and on the ring
        # table the added elevation was 32.462 deg (1.73 of its own patch) out before the
        # eye even moved. The azimuth convention (positive = eye-left) is unchanged.
        rays = self.rays if angles is self.angles else self.ring_rays
        t = compose_gaze(rays[r, c].reshape(1, 3), float(yaw), float(pitch))[0]
        az = float(np.clip(t[0], *AZ_RANGE))
        el = float(np.clip(t[1], *EL_RANGE))
        return az, el

    def observe(self, img: Optional[np.ndarray], eye_yaw_pitch, eye_speed: float,
                frame: int, ring_img: Optional[np.ndarray] = None) -> None:
        """Feed one fresh left-eye render (and the far-periphery ring's, if built).
        eye_yaw_pitch: ((l_yaw, l_pitch), (r_yaw, r_pitch)) from the env. eye_speed:
        max |qvel| of the eye joints, rad/s. The wide field detects first; the ring
        only answers where the wide field cannot see, at lower priority."""
        if img is None:
            return
        cur = self._pool(img)
        prev, prev_frame = self._prev, self._prev_frame
        self._prev, self._prev_frame = cur, frame
        ring_cur = ring_prev = None
        ring_prev_frame = -10**9
        if ring_img is not None and self.ring_rows:
            ring_cur = self._pool(ring_img, rows=self.ring_rows)
            ring_prev, ring_prev_frame = self._prev_ring, self._prev_ring_frame
            self._prev_ring, self._prev_ring_frame = ring_cur, frame
        self.n_observed += 1
        if frame < self._quiet_until or self._steps_left > 0:
            return                                   # refractory / mid-snap
        if eye_speed > self.eye_speed_gate:
            return                                   # saccade suppression (efference copy)
        (ly, lp), (ry, rp) = eye_yaw_pitch
        yaw, pitch = 0.5 * (ly + ry), 0.5 * (lp + rp)
        hit = None
        # a gap in SECONDS, not frames: the old frame limit was 10, while renders land
        # action_repeat frames apart, so raising action_repeat past 10 rejected every
        # pair and silenced the reflex with no error (measured: silent at 12)
        if prev is not None and (frame - prev_frame) <= self.max_gap_steps:
            hit = self._detect(cur, prev, self.periphery, self.angles, yaw, pitch, frame)
        if (hit is None and ring_cur is not None and ring_prev is not None
                and (frame - ring_prev_frame) <= self.max_gap_steps):
            hit = self._detect(ring_cur, ring_prev, self.ring_periphery,
                               self.ring_angles, yaw, pitch, frame)
            if hit is not None:
                self.n_snaps_ring += 1
        if hit is None:
            return
        az, el = hit
        self._target = np.array([az, el], dtype=np.float64)
        self._steps_left = self.snap_hold_steps
        self._quiet_until = frame + self.refractory_steps
        # One unit of habituation per capture, decaying with a real time constant, so a
        # spot recovers in a few seconds. It used to add the raw uint8 difference - fifty
        # units for one bright transient - which at 0.995 per observe took about 78 s to
        # fall back under 1, against a docstring promising "a couple of seconds".
        key = self._hab_key(az, el)
        prior = self._decayed(key, frame)
        self._hab[key] = (prior + 1.0, frame)
        self.n_snaps += 1

    # ------------------------------------------------------------------ habituation
    def _hab_key(self, az: float, el: float):
        return (int(round(az / self.hab_cell)), int(round(el / self.hab_cell)))

    def _decayed(self, key, frame: int) -> float:
        v = self._hab.get(key)
        if v is None:
            return 0.0
        strength, when = v
        return float(strength * np.exp(-max(0, frame - when) * self.dt / self.hab_tau_s))

    def _hab_map(self, yaw: float, pitch: float, frame: int,
                 angles: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-patch habituation, looked up by where each patch points in the WORLD.

        Spread over a neighbourhood, not one cell. A moving thing covers several patches,
        so habituating the exact cell that won just handed the next flash to the patch
        beside it - measured, the target walked to a new cell every flash and the spot
        captured gaze on 30 of 30 flashes. A looked-at direction now damps its surroundings
        with a Gaussian falloff, which is also what "this place is boring" should mean.
        """
        angles = self.angles if angles is None else angles
        out = np.zeros(angles.shape[:2], dtype=np.float32)
        if not self._hab:
            return out
        # same convention as the snap that WRITES the ledger: the table is in gaze-
        # command azimuth (source-negated 2026-09-06), so yaw + table IS the world
        # angle - record and lookup meet at the true spot
        # (test_habituation_a_flickering_spot_loses_priority guards the pairing).
        az = yaw + angles[..., 0]
        el = pitch + angles[..., 1]
        two_sig2 = 2.0 * self.hab_sigma ** 2
        for key, (strength, when) in list(self._hab.items()):
            v = float(strength * np.exp(-max(0, frame - when) * self.dt / self.hab_tau_s))
            if v < 0.01:
                del self._hab[key]                   # forgotten; keep the dict small
                continue
            kaz, kel = key[0] * self.hab_cell, key[1] * self.hab_cell
            out += v * np.exp(-((az - kaz) ** 2 + (el - kel) ** 2) / two_sig2)
        return out

    # ------------------------------------------------------------------ command
    def apply(self, gaze, eye_yaw_pitch=None) -> Optional[np.ndarray]:
        """Blend a pending snap into the model's gaze command. Called every env step.

        The snap is held until the eye ARRIVES, not for a fixed number of steps. Three
        steps of blending was 30 ms, and the eye plant cannot cross a 30 degree saccade in
        30 ms - measured on the real body, a snap moved the eyes 0.0039 deg per step
        against 0.0042 with no snap, which is to say not at all. The reflex counted snaps
        and moved nothing. Now the target holds until the eye is within arrive_deg or the
        timeout expires, which is what an orienting reflex does.
        """
        if self._steps_left <= 0 or self._target is None:
            return gaze
        if eye_yaw_pitch is not None:
            (ly, lp), (ry, rp) = eye_yaw_pitch
            here = np.array([0.5 * (ly + ry), 0.5 * (lp + rp)])
            if float(np.linalg.norm(here - self._target)) < self.arrive_rad:
                self._steps_left = 0                 # arrived: hand the eyes back
                return gaze
        self._steps_left -= 1
        if gaze is None:
            return self._target.copy()
        g = np.asarray(gaze, dtype=np.float64).reshape(-1)[:2]
        return (1.0 - self.weight) * g + self.weight * self._target

    @property
    def active(self) -> bool:
        return self._steps_left > 0

    def reset(self) -> None:
        """Called at episode reset, where env.frame goes back to 0. Every deadline here is
        keyed to that counter, so without this the first episode works and every later one
        is dead: quiet-until stays thousands of frames in the "future" (refractory forever),
        the stale-pair guard sees a negative gap and passes a pre-reset image as the
        previous frame, and habituation recorded before the reset can never decay - the
        elapsed time max(0, frame - when) pins at zero - so those directions stay
        suppressed permanently. Measured: 10 snaps in the 600 frames before a reset, 0 in
        the 600 after. Habituation is cleared outright: the reset teleports the world, so
        what was boring there no longer refers to anything. Cumulative counters survive."""
        self._prev = None
        self._prev_frame = -10**9
        self._prev_ring = None
        self._prev_ring_frame = -10**9
        self._quiet_until = -10**9
        self._steps_left = 0
        self._target = None
        self._hab.clear()



class SpontaneousSaccades:
    """The brainstem's own eye movement: jump, hold, jump again. Endowed, permanent.

    A newborn's eyes move without any reason to - the saccade generator fires on its own,
    from birth, for life. That wandering is not noise around the real behaviour; it IS the
    data the visual system learns oculomotor control from. Our eyes moved only when the
    reflex fired (about 5 times a minute of body time), which starves the gaze-action
    pathway of the (eye moved, image shifted) pairs it learns "what looking does" from. At
    an infant's ~2 fixations a second this supplies roughly 24x the data.

    Jump-and-hold, NOT per-step noise: noise smears the image every frame and nothing can
    be predicted from it. A saccade lands and the image holds still - stable views the
    next-moment lesson can feed on, from constantly moving eyes.

    The offset ADDS to whatever the model commands, so when gaze becomes purposeful the
    wander recentres on the chosen target instead of fighting it, exactly like the motor
    babble under the limb commands. While a reflex snap is steering, resampling pauses.
    """

    def __init__(self, dt: float, seed: int = 0, hold_s: Tuple[float, float] = (0.3, 0.6),
                 median_amp_deg: float = 5.0, amp_spread: float = 0.7, max_amp_deg: float = 30.0):
        """Amplitudes are LOGNORMAL, not Gaussian. Human saccades are heavily skewed small -
        a median near 4-5 degrees with a long tail past 20 - and a Gaussian has no such
        skew: sigma 12 produced a median 2D jump of 22 degrees, five times too large
        (measured). Oversized jumps also cost fixation time, because a slow eye is still
        travelling when the next one fires: at the old settings the eye was in flight 69%
        of the time against a human's ~10%, so the jump-and-hold this class exists to
        provide had degenerated into continuous motion and a permanently smeared image."""
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.hold_s = hold_s
        self.log_median = np.log(np.radians(median_amp_deg))
        self.amp_spread = float(amp_spread)
        self.max_amp = np.radians(max_amp_deg)
        self._offset = np.zeros(2)
        self._next_switch = 0
        self.n_saccades = 0

    def apply(self, gaze, frame: int, reflex_active: bool = False) -> np.ndarray:
        if frame >= self._next_switch and not reflex_active:
            amp = min(float(np.exp(self.rng.normal(self.log_median, self.amp_spread))),
                      self.max_amp)
            th = self.rng.uniform(0.0, 2 * np.pi)
            self._offset = np.array([amp * np.cos(th), amp * np.sin(th)])
            self._next_switch = frame + max(1, int(round(
                self.rng.uniform(*self.hold_s) / self.dt)))
            self.n_saccades += 1
        base = np.zeros(2) if gaze is None else np.asarray(gaze, dtype=np.float64).reshape(-1)[:2]
        # (reset() below rewinds _next_switch when env.frame goes back to 0)
        out = base + self._offset
        out[0] = float(np.clip(out[0], *AZ_RANGE))
        out[1] = float(np.clip(out[1], *EL_RANGE))
        return out

    def reset(self) -> None:
        """Episode reset: env.frame goes back to 0, and a next-switch left in the old
        clock's future would hold ONE offset for the entire next episode. Measured: 11
        saccades in the 600 frames before a reset, 1 after."""
        self._next_switch = 0
