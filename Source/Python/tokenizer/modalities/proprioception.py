# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Proprioception tokenizer — root-relative bone transforms.

Reads from CPU buffer proprioception slice (offset 8256, size 2048).
Output: [n_bones, 12] float32 — quat(4), root-rel pos(3), vel(3), fatigue(1), bone_index(1).
"""

import struct

import numpy as np
import torch

from ..config import (
    TokenizerConfig,
    PROPRIO_OFFSET, PROPRIO_MAX_BONES, PROPRIO_BONE_STRIDE,
    PRO_NUM_BONES, PRO_BONE_DATA,
    BONE_ROTATION, BONE_TRANSLATION, BONE_VELOCITY, BONE_FATIGUE,
)


def tokenize_proprioception(
    slot_bytes: bytes,
    config: TokenizerConfig,
) -> torch.Tensor:
    """
    Tokenize proprioception slice from a CPU buffer slot.

    Args:
        slot_bytes: Full parity slot bytes.
        config: Tokenizer configuration.

    Returns:
        proprio: [n_bones, 12] float32 on device.
            Per bone: [quat_x, quat_y, quat_z, quat_w,
                       pos_x, pos_y, pos_z,   (root-relative, normalized)
                       vel_x, vel_y, vel_z,   (normalized)
                       fatigue, bone_index]
    """
    pb = PROPRIO_OFFSET
    n_bones = struct.unpack_from('<i', slot_bytes, pb + PRO_NUM_BONES)[0]
    n_bones = max(0, min(n_bones, PROPRIO_MAX_BONES))

    if n_bones == 0:
        return torch.zeros(0, 12, dtype=torch.float32, device=config.device)

    bones = np.zeros((n_bones, 12), dtype=np.float32)
    root_pos = None

    for i in range(n_bones):
        bb = pb + PRO_BONE_DATA + i * PROPRIO_BONE_STRIDE

        # Quaternion XYZW
        qx, qy, qz, qw = struct.unpack_from('<4f', slot_bytes, bb + BONE_ROTATION)

        # Translation (cm, world-space)
        tx, ty, tz = struct.unpack_from('<3f', slot_bytes, bb + BONE_TRANSLATION)

        # Velocity (cm/s, world-space)
        vx, vy, vz = struct.unpack_from('<3f', slot_bytes, bb + BONE_VELOCITY)

        # Fatigue [0, 1]
        fatigue = struct.unpack_from('<f', slot_bytes, bb + BONE_FATIGUE)[0]

        pos = np.array([tx, ty, tz], dtype=np.float32)

        if i == 0:
            root_pos = pos.copy()

        # Root-relative position, normalized
        rel_pos = (pos - root_pos) / config.position_range_cm

        # Normalized velocity
        vel_norm = np.array([vx, vy, vz], dtype=np.float32) / config.velocity_range_cms

        bones[i] = [
            qx, qy, qz, qw,
            rel_pos[0], rel_pos[1], rel_pos[2],
            vel_norm[0], vel_norm[1], vel_norm[2],
            fatigue,
            float(i) / config.proprio_index_range,   # row index on the row's own 0..1 scale
        ]

    return torch.from_numpy(bones)  # Stay on CPU; _to_device handles GPU transfer
