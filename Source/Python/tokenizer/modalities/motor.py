# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Motor tokenizer — physics handle command normalization.

Reads from CPU buffer motor slice (offset 12352, size 2048).
Output: handles [n_handles, 10], energy [3].

Handle indices 0-23 (names: lem.model.body_sets.HANDLE_NAMES):
  0-5:   Left arm (wrist + 5 fingers)
  6-11:  Right arm (wrist + 5 fingers)
  12:    Head
  13-14: Left upper/lower arm
  15-16: Right upper/lower arm
  17-18: Left thigh/calf
  19-20: Right thigh/calf
  21-22: Left/right foot
  23:    Eyes (gaze)
"""

import struct
from typing import Tuple

import numpy as np
import torch

from ..config import (
    TokenizerConfig,
    MOTOR_OFFSET, MOTOR_MAX_HANDLES, MOTOR_HANDLE_STRIDE,
    MOT_HANDLES, MOT_ENERGY,
    HND_TARGET_POS, HND_TARGET_ROT, HND_RADIUS, HND_STRENGTH, HND_IS_ACTIVE,
    NRG_PH_WORK, NRG_PCC_HOLDING, NRG_TOTAL,
    NUM_BODY_SETS, BODY_SET_STATE_STRIDE, MOT_BODY_SETS,
    BST_CAPACITY, BST_ACCUMULATED_ENERGY,
)

NUM_HANDLES = 24  # All slots active since the eyes became handle 23 (2026-08-24)


def tokenize_motor(
    slot_bytes: bytes,
    config: TokenizerConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize motor slice from a CPU buffer slot.

    Args:
        slot_bytes: Full parity slot bytes.
        config: Tokenizer configuration.

    Returns:
        handles: [NUM_HANDLES, 10] float32 on device.
            Per handle: [target_pos(3), target_rot(4), radius(1), strength(1), is_active(1)]
        energy: [3] float32 on device.
            [ph_work, pcc_holding, total] normalized by energy_max.
        capability: [21] float32 on device.
            [capacity(10), accum_energy(10), total_energy(1)] for capability wellness.
    """
    mb = MOTOR_OFFSET
    handles = np.zeros((NUM_HANDLES, 10), dtype=np.float32)

    for i in range(NUM_HANDLES):
        hb = mb + MOT_HANDLES + i * MOTOR_HANDLE_STRIDE

        # Target position (cm) -> normalized
        px, py, pz = struct.unpack_from('<3f', slot_bytes, hb + HND_TARGET_POS)

        # Target rotation (quat XYZW)
        rx, ry, rz, rw = struct.unpack_from('<4f', slot_bytes, hb + HND_TARGET_ROT)

        radius = struct.unpack_from('<f', slot_bytes, hb + HND_RADIUS)[0]
        strength = struct.unpack_from('<f', slot_bytes, hb + HND_STRENGTH)[0]
        is_active = float(struct.unpack_from('<I', slot_bytes, hb + HND_IS_ACTIVE)[0])

        handles[i] = [
            px / config.position_range_cm,
            py / config.position_range_cm,
            pz / config.position_range_cm,
            rx, ry, rz, rw,
            radius,
            strength,
            is_active,
        ]

    # Energy
    eb = mb + MOT_ENERGY
    ph_work = struct.unpack_from('<f', slot_bytes, eb + NRG_PH_WORK)[0]
    pcc_hold = struct.unpack_from('<f', slot_bytes, eb + NRG_PCC_HOLDING)[0]
    total_e = struct.unpack_from('<f', slot_bytes, eb + NRG_TOTAL)[0]

    energy = np.array([
        ph_work / config.energy_max,
        pcc_hold / config.energy_max,
        total_e / config.energy_max,
    ], dtype=np.float32)

    # Body set muscle capacity (10 body sets × 2 floats each)
    bsb = mb + MOT_BODY_SETS
    capability = np.zeros(21, dtype=np.float32)

    for i in range(NUM_BODY_SETS):
        offset = bsb + i * BODY_SET_STATE_STRIDE
        capacity_val = struct.unpack_from('<f', slot_bytes, offset + BST_CAPACITY)[0]
        accum_energy = struct.unpack_from('<f', slot_bytes, offset + BST_ACCUMULATED_ENERGY)[0]
        capability[i] = capacity_val  # First 10 elements: capacity [0-1]
        capability[NUM_BODY_SETS + i] = accum_energy / config.energy_max  # Next 10: normalized accumulated

    capability[20] = total_e / config.energy_max  # Total energy (same as energy[2])

    handles_t = torch.from_numpy(handles)    # Stay on CPU; _to_device handles GPU transfer
    energy_t = torch.from_numpy(energy)
    capability_t = torch.from_numpy(capability)

    return handles_t, energy_t, capability_t
