# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Touch tokenizer — derived tactile patch & bone collision reader.

Reads from CPU buffer touch slice (offset 14400, size 16384).
Patches: up to 256 FTMCDerivedPatch entries (8 floats per sensor).
Collisions: up to TOUCH_MAX_COLLISIONS (16) bone collision events (contact point, normal,
  penetration, impact speed).

Output: patches [256, 10], collisions [16, 8], valid counts.

Per-patch fields (from ProcessDerivedFields on C++ side):
  [0] normalPressure  (Pa)      — GPU Neo-Hookean pressure (area-weighted avg)
  [1] shearMagnitude  (Pa)      — tangential force / area (geometric) or Coulomb friction during slip
  [2] slipRate        (mm/s)    — contact point displacement / dt
  [3] contactArea     (mm²)     — max single-impulse spherical cap
  [4] vibPower_5Hz    (N²/bin)  — Goertzel PSD at 5 Hz  (field name historical: _50Hz)
  [5] vibPower_10Hz   (N²/bin)  — Goertzel PSD at 10 Hz (field name historical: _100Hz)
  [6] vibPower_15Hz   (N²/bin)  — Goertzel PSD at 15 Hz (field name historical: _200Hz)
  [7] vibPower_20Hz   (N²/bin)  — Goertzel PSD at 20 Hz (field name historical: _400Hz)
  [8] principalStretch1 (ratio) — skin patch stretch along its major axis, emitted as
                                  (λ1 - 1) / 0.5: 0 = resting length (matches the
                                  zero padding of absent patches), +0.4 = stretched
                                  20%, -0.4 = compressed 20%
  [9] principalStretch2 (ratio) — same for the minor axis (λ2)

2026-09-14: fields 8-9 were being read and thrown away ("outputs first 8") - the
UE avatar computes both stretches per patch from the mesh deformation gradient
(they register skin pulling near a bending joint even with no contact), the C++
side has always written all 10, and the model never saw them.
"""

import struct
from typing import Tuple

import numpy as np
import torch

from ..config import (
    TokenizerConfig,
    TOUCH_OFFSET, TOUCH_MAX_IMPULSES, TOUCH_MAX_COLLISIONS,
    TOUCH_IMPULSE_STRIDE, TOUCH_COLLISION_STRIDE,
    TCH_NUM_IMPULSES, TCH_NUM_COLLISIONS,
    TCH_IMPULSE_DATA, TCH_COLLISION_DATA,
)

# Derived tactile patch: 10 × float32 = 40 bytes (matches FTMCDerivedPatch in TMCTypes.h)
_PATCH_DT = np.dtype([('fields', np.float32, 10)])
assert _PATCH_DT.itemsize == TOUCH_IMPULSE_STRIDE

_COLLISION_DT = np.dtype([
    ('_bone_hash', np.uint32),       #  4 bytes
    ('contact_pt', np.float32, 3),   # 12 bytes — contact point (cm)
    ('normal', np.float32, 3),       # 12 bytes — contact normal
    ('pen_depth', np.float32),       #  4 bytes — cm
    ('impact_speed', np.float32),    #  4 bytes — cm/s
    ('_impulse', np.float32, 3),     # 12 bytes — impact impulse vector
    ('_duration', np.float32),       #  4 bytes — seconds
    ('_frame_count', np.int32),      #  4 bytes
    ('_pad', np.uint8, 8),           #  8 bytes — alignment
])  # total: 64 bytes
assert _COLLISION_DT.itemsize == TOUCH_COLLISION_STRIDE

# Per-field normalization divisors calibrated from live SHM data (feet + hands + spine).
# Target: typical values land in ~[0, 0.5]; peaks during active manipulation may reach ~1.0.
# Index: 0=normalPressure(Pa), 1=shearMag(Pa), 2=slipRate, 3=contactArea(mm²), 4-7=vibPower PSD (N²/bin) at 5/10/15/20 Hz
# Recalibrated 2026-02-21: contactArea observed up to ~17,414 mm² (large torso/limb sensors),
# divisor raised from 200 → 20,000. normalPressure now true Pa after proportional skinThickness fix.
# Recalibrated 2026-03-05: SkinThicknessFactor 1→2 halves pressures (÷200k→÷100k).
# shearMagnitude was dead (no friction model); Coulomb friction (μ=0.5) now gives 5k–50k Pa (÷500→÷50k).
# Fields 8-9 (principal stretches): the raw value is a length RATIO resting at 1.0,
# so it is re-zeroed to (λ - 1) before dividing by 0.5 - a real 20% stretch reads
# 0.4, and a patch at rest reads 0 exactly like the zero padding of absent patches.
_FIELD_NORMS = np.array([100_000, 50_000, 200, 20_000, 10, 10, 10, 10, 0.5, 0.5], dtype=np.float32)
_STRETCH_REST = 1.0    # raw λ at resting length; subtracted before normalization


def tokenize_touch(
    slot_bytes: bytes,
    config: TokenizerConfig,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Tokenize touch slice from a CPU buffer slot.

    Args:
        slot_bytes: Full parity slot bytes.
        config: Tokenizer configuration.

    Returns:
        patches:    [256, 10] float32 on device, zero-padded (all 10 written fields).
            Per patch: [normalPressure(Pa), shearMagnitude(Pa), slipRate(mm/s), contactArea(mm²),
                        vibPower_5Hz, vibPower_10Hz, vibPower_15Hz, vibPower_20Hz (N²/bin),
                        principalStretch1, principalStretch2 (as (λ-1)/0.5; 0 = rest)]
        collisions: [16, 8] float32 on device, zero-padded.
            Per collision: [contact_point(3), contact_normal(3),
                            penetration(1), impact_speed(1)]
        valid_patch_count: int
        valid_collision_count: int
    """
    tb = TOUCH_OFFSET

    n_patches = struct.unpack_from('<i', slot_bytes, tb + TCH_NUM_IMPULSES)[0]
    n_patches = max(0, min(n_patches, TOUCH_MAX_IMPULSES))

    n_collisions = struct.unpack_from('<i', slot_bytes, tb + TCH_NUM_COLLISIONS)[0]
    n_collisions = max(0, min(n_collisions, TOUCH_MAX_COLLISIONS))

    # --- Vectorized derived-patch parsing ---
    patches = np.zeros((TOUCH_MAX_IMPULSES, 10), dtype=np.float32)
    if n_patches > 0:
        patch_start = tb + TCH_IMPULSE_DATA
        raw = np.frombuffer(slot_bytes, dtype=_PATCH_DT,
                            count=n_patches, offset=patch_start)
        fields = raw['fields'].copy()
        fields[:, 8:10] -= _STRETCH_REST   # length ratios rest at 1.0; re-zero first
        patches[:n_patches] = fields / _FIELD_NORMS

    # --- Vectorized collision parsing (unchanged) ---
    collisions = np.zeros((TOUCH_MAX_COLLISIONS, 8), dtype=np.float32)
    if n_collisions > 0:
        col_start = tb + TCH_COLLISION_DATA
        raw_col = np.frombuffer(slot_bytes, dtype=_COLLISION_DT,
                                count=n_collisions, offset=col_start)

        contact_pt = raw_col['contact_pt'] / config.position_range_cm  # [n, 3]
        normal = raw_col['normal']                                      # [n, 3] already unit-length
        pen_depth = raw_col['pen_depth'].reshape(-1, 1) / 10.0
        impact_spd = raw_col['impact_speed'].reshape(-1, 1) / config.velocity_range_cms

        collisions[:n_collisions] = np.hstack([contact_pt, normal, pen_depth, impact_spd])

    patches_t = torch.from_numpy(patches)      # Stay on CPU; _to_device handles GPU transfer
    collisions_t = torch.from_numpy(collisions)

    return patches_t, collisions_t, n_patches, n_collisions
