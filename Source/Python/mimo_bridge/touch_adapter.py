# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Touch adapter: MIMo skin points and contacts -> model touch patches and collisions.

Plain words: MIMo gives us thousands of tiny skin points, each with a force arrow, plus
MuJoCo's list of contacts. The model wants at most 256 "patches", each with 10 numbers:

  0 normal pressure  Pa      = (sum of normal force on the patch) / (patch contact area)
  1 shear            Pa      = |sum of tangential force on the patch| / (patch contact area)
  2 slip rate        mm/s    = how far the contact point moved on the skin since last step / dt
  3 contact area     mm2     = (number of skin points that feel force) x (one point's area)
  4..7 vibration     N^2     = Goertzel power of the patch's normal-force history at
                               5, 10, 15, 20 Hz (the model's band centres)
  8,9 stretch        ratio   = 1.0 (MIMo has no soft skin; labeled "rest")

One patch = one MIMo sensing body (53 skin bodies; the two eye bodies are not skin).
Patch order = MIMo body id order (same order MIMo flattens its touch output in).

Every rule here is pure numpy so it can be checked by hand (see tests/test_touch_adapter.py).
No MuJoCo import in this file: env.py collects the raw inputs and hands them over.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from tokenizer.config import TOUCH_MAX_IMPULSES, TOUCH_MAX_COLLISIONS

N_PATCH_FIELDS = 10            # matches TOUCH_IMPULSE_STRIDE / 4
VIB_BANDS_HZ = (5.0, 10.0, 15.0, 20.0)   # tokenizer/config.py: historical names _50HZ.._400HZ
DEFAULT_VIB_WINDOW = 100       # samples; at dt = 0.01 s this is 1 s, so 5/10/15/20 Hz are exact bins
ACTIVE_FORCE_EPS = 1e-6        # N; a skin point with |force| above this counts as touched
# The 53 sensing bodies grouped the way a person would name them. A hand is its palm and
# every finger segment; a foot is its sole, toes and big toe; the trunk is chest, upper
# body, lower body and hip. Used for the per-region touch report and for asking whether
# the body's map of itself agrees with where it is being touched.
SKIN_REGIONS = (
    ("head", ("head",)),
    ("trunk", ("chest", "upper_body", "lower_body", "hip")),
    ("left hand", ("left_hand", "left_ff", "left_mf", "left_rf", "left_lf", "left_th")),
    ("right hand", ("right_hand", "right_ff", "right_mf", "right_rf", "right_lf", "right_th")),
    ("left lower arm", ("left_lower_arm",)),
    ("right lower arm", ("right_lower_arm",)),
    ("left upper arm", ("left_upper_arm",)),
    ("right upper arm", ("right_upper_arm",)),
    ("left upper leg", ("left_upper_leg",)),
    ("right upper leg", ("right_upper_leg",)),
    ("left lower leg", ("left_lower_leg",)),
    ("right lower leg", ("right_lower_leg",)),
    ("left foot", ("left_foot", "left_toes", "left_big_toe")),
    ("right foot", ("right_foot", "right_toes", "right_big_toe")),
)


def region_of(body_name: str) -> str:
    """Which named region a sensing body belongs to; '' when none claims it."""
    for region, prefixes in SKIN_REGIONS:
        for prefix in prefixes:
            if body_name == prefix or body_name.startswith(prefix):
                return region
    return ""


SKIN_SCALE_MAX = 0.5           # m; bodies with a larger sensor spacing are placeholders (eyes), not skin


def bone_hash(name: str) -> int:
    """Stable 32-bit hash for a bone/body name (UE uses GetTypeHash(FName); we use crc32).

    The model's tokenizer never reads this field, so any stable hash is fine. Kept
    deterministic so tests and logs can match names to hashes.
    """
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


@dataclass(frozen=True)
class PatchSpec:
    """One patch = one MIMo sensing body."""
    body_id: int
    body_name: str
    n_points: int
    scale_m: float          # sensor point spacing in metres (MIMo touch scale)

    @property
    def point_area_m2(self) -> float:
        return self.scale_m * self.scale_m


# HOW DEEP AND HOW FAST, ON THIS BODY'S OWN SCALE. The first cut divided depth by 10 cm and
# the closing speed by 500 cm/s, the divisors of the UE collision reader. MuJoCo's soft contacts
# on the crib body are far smaller than that: measured 2026-09-21 over 3,000 model steps of
# motor babble, 19,118 contacts and 4,102 arrivals, the 99th percentile contact pressed 0.108 cm
# deep and the 99th percentile arrival came in at 2.21 cm/s, so on the old divisors they read
# 0.0108 and 0.0044 of their scales - the push's direction reached the model and its depth and
# speed did not, while pressure in the same rows reads up to 2.93 of its own scale. Each divisor
# is now the 99th percentile of what this body produces, rounded: 0.1 cm and 2 cm/s, so the 99th
# percentile reads about 1. The deepest contact measured, 0.312 cm, reads 3.1 and the fastest
# arrival, 14.0 cm/s, reads 7.0. A checkpoint trained on the old divisors is migrated exactly on
# load (lem_model._migrate_touch_push_scale), and every checkpoint records which it used.
PUSH_DEPTH_SCALE_CM = 0.1
PUSH_ARRIVAL_SCALE_CMS = 2.0
LEGACY_PUSH_SCALE = (10.0, 500.0)      # the divisors in force before 2026-09-21


@dataclass
class ContactInfo:
    """One MuJoCo contact as seen from one sensing body (env.py builds these)."""
    body_id: int                 # the sensing MIMo body
    pair_key: tuple              # (geom1, geom2) - identifies the same contact across steps
    normal_force: float          # N, >= 0, along the contact normal
    tangential_body: np.ndarray  # (3,) N, tangential (friction) force in the body frame
    pos_body: np.ndarray         # (3,) m, contact point in the sensing body frame (on its surface)
    pos_world: np.ndarray        # (3,) m
    normal_world: np.ndarray     # (3,) unit, pointing into the sensing body (MIMo convention)
    penetration_m: float         # m, >= 0
    rel_normal_speed: float      # m/s, closing speed along the normal (>= 0 when approaching)
    other_body_name: str = ""
    # THE SAME DIRECTION IN THE SENSING BODY'S OWN FRAME. The push columns of a touch token
    # sit in the same row as pos_body, which is measured from the body, and were being filled
    # from normal_world, which is measured from the world. The trunk rolls - the run tracks
    # how far - so one and the same self-touch gave a direction that changed as the body
    # turned, and every floor contact read straight up whatever the body was doing.
    normal_body: Optional[np.ndarray] = None
    # THE ONE FACT THAT MAKES SELF-TOUCH SPECIAL: both surfaces are the body's own, so the
    # command that moved one and the sensation on the other are two consequences of one
    # action. Nothing else in the world produces that pairing. Set by env.py from MuJoCo's
    # body tree (same root body = the same body).
    other_is_self: bool = False
    rel_tangent_speed: Optional[float] = None  # m/s at the actual contact point
    friction_power_w: Optional[float] = None  # dissipated power for this contact


class Goertzel:
    """Power of one frequency in a ring buffer of samples.

    power = (2 |X_k| / N)^2  -> for a full-window sine of amplitude A at that frequency
    this returns A^2 (units: N^2 when the samples are forces). Generalized Goertzel, so
    the frequency need not sit on an integer bin; tests use dt = 0.01 and N = 100 where
    5/10/15/20 Hz ARE integer bins and the answer is exact.
    """

    def __init__(self, freqs_hz: Sequence[float], dt: float, window: int):
        self.freqs = np.asarray(freqs_hz, dtype=np.float64)
        self.dt = float(dt)
        self.window = int(window)
        # bin index k (possibly fractional) for each frequency
        self.k = self.freqs * self.window * self.dt
        self.omega = 2.0 * math.pi * self.k / self.window
        self.coeff = 2.0 * np.cos(self.omega)

    def power(self, samples: np.ndarray) -> np.ndarray:
        """samples: (window,) oldest first. Returns (n_freqs,) power in amplitude^2 units."""
        x = np.asarray(samples, dtype=np.float64)
        n = x.shape[0]
        out = np.zeros(self.freqs.shape[0], dtype=np.float64)
        if n == 0:
            return out
        for i in range(self.freqs.shape[0]):
            s_prev = 0.0
            s_prev2 = 0.0
            c = self.coeff[i]
            for v in x:
                s = v + c * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            # |X_k|^2 (generalized: use the complex form so fractional k works)
            w = self.omega[i]
            real = s_prev - s_prev2 * math.cos(w)
            imag = s_prev2 * math.sin(w)
            mag2 = real * real + imag * imag
            out[i] = 4.0 * mag2 / (n * n)
        return out


class TouchPooler:
    """Turns per-point forces + contacts into [n_patches, 10] derived fields.

    Stateful: keeps a per-patch normal-force history (for vibration) and the last contact
    positions (for slip). Call update() once per env step, in step order.
    """

    def __init__(self, patch_specs: Sequence[PatchSpec], dt: float,
                 vib_window: int = DEFAULT_VIB_WINDOW,
                 band_hz: Sequence[float] = VIB_BANDS_HZ,
                 max_patches: int = TOUCH_MAX_IMPULSES):
        specs = [s for s in patch_specs if s.scale_m < SKIN_SCALE_MAX]
        specs = sorted(specs, key=lambda s: s.body_id)
        if len(specs) > max_patches:
            raise ValueError(f"{len(specs)} skin bodies but the frame holds at most {max_patches} patches")
        self.specs: List[PatchSpec] = specs
        self.n_patches = len(specs)
        self.dt = float(dt)
        self.body_to_patch: Dict[int, int] = {s.body_id: i for i, s in enumerate(specs)}
        self.point_area_mm2 = np.array([s.point_area_m2 * 1e6 for s in specs], dtype=np.float64)
        self.goertzel = Goertzel(band_hz, dt, vib_window)
        self.vib_window = int(vib_window)
        self._hist = np.zeros((self.n_patches, self.vib_window), dtype=np.float64)  # normal force N
        self.last_self = np.zeros((self.n_patches, 2), dtype=np.float32)
        self.last_contact = np.zeros((self.n_patches, 5), dtype=np.float32)
        self.last_self_force = 0.0
        self.last_total_force = 0.0
        self.last_self_pairs = 0
        self._last_contact_pos: Dict[tuple, list] = {}  # (body_id, pair_key) -> [pos_body, ...]
        self._contact_first_step: Dict[tuple, int] = {}
        self._step = 0
        self.last_fields: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- core
    def update(self, point_forces: Dict[int, np.ndarray],
               contacts: Sequence[ContactInfo], include_vibration: bool = True) -> np.ndarray:
        """point_forces: {body_id: (n_points, 3) force vectors in body frame, N}.
        contacts: ContactInfo list for this step.
        Returns fields (n_patches, 10) float32 in model units.
        """
        n = self.n_patches
        fields = np.zeros((n, N_PATCH_FIELDS), dtype=np.float64)
        fields[:, 8] = 1.0
        fields[:, 9] = 1.0

        # contact area from active skin points
        n_active = np.zeros(n, dtype=np.float64)
        for body_id, f in point_forces.items():
            p = self.body_to_patch.get(body_id)
            if p is None:
                continue
            f = np.asarray(f)
            if f.size == 0:
                continue
            mag = np.linalg.norm(f.reshape(-1, 3), axis=1)
            n_active[p] = float(np.count_nonzero(mag > ACTIVE_FORCE_EPS))
        area_mm2 = n_active * self.point_area_mm2
        area_m2 = area_mm2 * 1e-6

        # sum forces per patch from contacts; slip per contact
        normal_sum = np.zeros(n, dtype=np.float64)
        tang_sum = np.zeros((n, 3), dtype=np.float64)
        slip_max = np.zeros(n, dtype=np.float64)
        friction_power = np.zeros(n, dtype=np.float64)
        # review 2026-09-06, two jobs with two keyings:
        #  FRESHNESS (feeds the audio impact gate's frame_count) stays keyed by
        #  (body, pair) - the proven semantics; per-point keys made resting points
        #  look freshly-arrived and the 355-thumps machine gun came back (caught by
        #  its conviction test on the first try).
        #  SLIP matches each point to the NEAREST stored point of its pair (3 cm
        #  gate): flat-on-flat contact yields several simultaneous points per pair,
        #  and one shared slot read their alternation as centimetre "slip" spikes.
        seen_pairs = set()
        # _last_contact_pos maps (body, pair) -> list of last-step point positions
        new_pos: Dict[tuple, list] = {}
        # per patch: how much of this step's force came from the body's own parts, and how
        # many distinct own-parts are on it. Zero for the floor, the toy and the adult.
        self_force = np.zeros(n, dtype=np.float64)
        self_parts = [set() for _ in range(n)]
        # THE PUSH ITSELF, not just its size. The patch fields say how hard, how much it
        # rubbed and how fast it slid; they never say which WAY the push came from, how
        # deep it is, or how fast it arrived. Those three are what the collision entries
        # carried, and until 2026-09-21 the entries were assembled and read by nobody. Kept
        # for the strongest contact on each patch, which is the one the row is sorted by.
        strongest = np.zeros(n, dtype=np.float64)
        contact_cols = np.zeros((n, 5), dtype=np.float64)
        self_pair_keys = set()
        for c in contacts:
            p = self.body_to_patch.get(c.body_id)
            if p is None:
                continue
            force = max(0.0, float(c.normal_force))
            if force >= strongest[p]:
                strongest[p] = force
                # the direction in the sensing body's own frame, to match pos_body in the
                # same row; the world direction is the fallback for a caller that has none
                nb = c.normal_body if c.normal_body is not None else c.normal_world
                contact_cols[p, :3] = np.asarray(nb, dtype=np.float64).reshape(-1)[:3]
                # depth in cm and closing speed in cm/s, each over this body's own scale
                # (PUSH_DEPTH_SCALE_CM, PUSH_ARRIVAL_SCALE_CMS)
                contact_cols[p, 3] = float(c.penetration_m) * 100.0 / PUSH_DEPTH_SCALE_CM
                contact_cols[p, 4] = (float(c.rel_normal_speed or 0.0) * 100.0
                                      / PUSH_ARRIVAL_SCALE_CMS)
            if getattr(c, "other_is_self", False):
                self_force[p] += max(0.0, float(c.normal_force))
                self_parts[p].add(c.other_body_name)
                # ONE CONTACT IS ONE PAIR. Both surfaces of a self-touch sense it, so env.py
                # builds one record per side and each side added the other's name to its own
                # patch. Summing the patches therefore counted every pair twice: the live run
                # reported 2.60 pairs against 2.45 patches. The two records of one contact
                # carry the same pair of shapes, so counting those counts pairs.
                self_pair_keys.add(c.pair_key)
            normal_sum[p] += max(0.0, float(c.normal_force))
            tang_sum[p] += np.asarray(c.tangential_body, dtype=np.float64)
            if c.rel_tangent_speed is not None:
                slip_max[p] = max(slip_max[p], c.rel_tangent_speed * 1000.)
            if c.friction_power_w is not None:
                friction_power[p] += c.friction_power_w
            pair = (c.body_id, c.pair_key)
            seen_pairs.add(pair)
            pos = np.asarray(c.pos_body, dtype=np.float64)
            cand = self._last_contact_pos.get(pair)
            if cand and self.dt > 0 and c.rel_tangent_speed is None:
                ds = [float(np.linalg.norm(pos - q)) for q in cand]
                j = int(np.argmin(ds))
                if ds[j] < 0.03:                     # same physical point, moved
                    slip_max[p] = max(slip_max[p], ds[j] / self.dt * 1000.0)
                    cand.pop(j)                      # one match per stored point
            new_pos.setdefault(pair, []).append(pos.copy())
            self._contact_first_step.setdefault(pair, self._step)
        self._last_contact_pos = new_pos
        # forget pairs that ended
        for pair in list(self._contact_first_step.keys()):
            if pair not in seen_pairs:
                self._contact_first_step.pop(pair, None)

        with np.errstate(divide="ignore", invalid="ignore"):
            pressure = np.where(area_m2 > 0, normal_sum / area_m2, 0.0)
            shear = np.where(area_m2 > 0, np.linalg.norm(tang_sum, axis=1) / area_m2, 0.0)
        fields[:, 0] = pressure
        fields[:, 1] = shear
        fields[:, 2] = slip_max
        fields[:, 3] = area_mm2
        self.friction_power_density = np.divide(friction_power, area_m2,
            out=np.zeros_like(friction_power), where=area_m2 > 0)

        # vibration: push normal force into history, Goertzel per patch that has any history
        self._hist = np.roll(self._hist, -1, axis=1)
        self._hist[:, -1] = normal_sum
        active_hist = np.any(self._hist != 0.0, axis=1)
        for p in np.nonzero(active_hist)[0] if include_vibration else ():
            fields[p, 4:8] = self.goertzel.power(self._hist[p])

        self._step += 1
        # SELF-TOUCH, kept beside the fields rather than inside them: the ten fields are the
        # layout the UE avatar's C++ side also writes, and two of them mean skin stretch
        # there. Share is 0 to 1 of this patch's force; parts is how many different own
        # parts are on it, over 4 and capped, so both columns live on the same 0-to-1 scale
        # as everything else the encoder reads.
        share = np.where(normal_sum > 0.0, self_force / np.maximum(normal_sum, 1e-12), 0.0)
        parts = np.array([min(len(x), 4) / 4.0 for x in self_parts], dtype=np.float64)
        self.last_self = np.stack([share, parts], axis=1).astype(np.float32)
        self.last_contact = contact_cols.astype(np.float32)
        self.last_self_force = float(self_force.sum())
        self.last_total_force = float(normal_sum.sum())
        self.last_self_pairs = len(self_pair_keys)
        self.last_fields = fields.astype(np.float32)
        return self.last_fields

    def contact_duration_s(self, body_id: int, pair_key: tuple) -> float:
        first = self._contact_first_step.get((body_id, pair_key))
        if first is None:
            return 0.0
        return (self._step - first) * self.dt

    def contact_frame_count(self, body_id: int, pair_key: tuple) -> int:
        first = self._contact_first_step.get((body_id, pair_key))
        if first is None:
            return 0
        return self._step - first


# ---------------------------------------------------------------- collisions
COLLISION_FIELDS = 16  # floats per collision entry (64 B); see frame_encoder for the packing


def build_collisions(contacts: Sequence[ContactInfo], pooler: Optional[TouchPooler],
                     body_hash: Dict[int, int], dt: float,
                     max_entries: int = TOUCH_MAX_COLLISIONS) -> List[dict]:
    """Pick up to 16 strongest contacts and express them as the model's bone collision entries.

    Units follow the model's reader: point cm (world), normal unit, penetration cm,
    impact speed cm/s, impulse N*s (force x dt), duration s, frame count.
    """
    ordered = sorted(contacts, key=lambda c: -float(c.normal_force))
    out = []
    for c in ordered[:max_entries]:
        dur = pooler.contact_duration_s(c.body_id, c.pair_key) if pooler else 0.0
        cnt = pooler.contact_frame_count(c.body_id, c.pair_key) if pooler else 0
        out.append({
            "bone_hash": int(body_hash.get(c.body_id, 0)),
            "point_cm": (np.asarray(c.pos_world, dtype=np.float64) * 100.0).astype(np.float32),
            "normal": np.asarray(c.normal_world, dtype=np.float32),
            "penetration_cm": float(c.penetration_m * 100.0),
            "impact_speed_cms": float(c.rel_normal_speed * 100.0),
            "impulse_ns": (np.asarray(c.normal_world, dtype=np.float64) * float(c.normal_force) * dt).astype(np.float32),
            "duration_s": float(dur),
            "frame_count": int(cnt),
        })
    return out
