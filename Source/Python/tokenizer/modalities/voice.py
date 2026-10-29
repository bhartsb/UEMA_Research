# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Voice tokenizer — vocal tract synthesis state.

Reads from CPU buffer voice slice (offset 13376, size 256).
Output: [56] float32 — all normalized to [0, 1].
"""

import math
import struct

import numpy as np
import torch

from ..config import (
    TokenizerConfig,
    VOICE_OFFSET,
    VOC_TARGET_FREQ, VOC_TARGET_TENS, VOC_TONGUE_INDEX,
    VOC_TONGUE_DIAMETER, VOC_VIBRATO_AMOUNT, VOC_VIBRATO_FREQ,
    VOC_INTENSITY, VOC_LOUDNESS, VOC_VELUM_TARGET,
    VOC_AUTO_WOBBLE, VOC_ALWAYS_VOICE, VOC_SYNTH_TIME,
    VOC_DIAMETERS, VOC_NUM_DIAMETERS,
)


def tokenize_voice(
    slot_bytes: bytes,
    config: TokenizerConfig,
) -> torch.Tensor:
    """
    Tokenize voice slice from a CPU buffer slot.

    Args:
        slot_bytes: Full parity slot bytes.
        config: Tokenizer configuration.

    Returns:
        voice: [56] float32 on device.
            [freq_norm, tenseness, tongue_index_norm, tongue_diam_norm,
             vibrato_amount_norm, vibrato_freq_norm, intensity, loudness,
             velum_norm, auto_wobble, always_voice, synth_time_wrapped,
             diameters[0..43]]
    """
    vb = VOICE_OFFSET

    freq = struct.unpack_from('<f', slot_bytes, vb + VOC_TARGET_FREQ)[0]
    tenseness = struct.unpack_from('<f', slot_bytes, vb + VOC_TARGET_TENS)[0]
    tongue_idx = struct.unpack_from('<f', slot_bytes, vb + VOC_TONGUE_INDEX)[0]
    tongue_diam = struct.unpack_from('<f', slot_bytes, vb + VOC_TONGUE_DIAMETER)[0]
    vibrato_amt = struct.unpack_from('<f', slot_bytes, vb + VOC_VIBRATO_AMOUNT)[0]
    vibrato_freq = struct.unpack_from('<f', slot_bytes, vb + VOC_VIBRATO_FREQ)[0]
    intensity = struct.unpack_from('<f', slot_bytes, vb + VOC_INTENSITY)[0]
    loudness = struct.unpack_from('<f', slot_bytes, vb + VOC_LOUDNESS)[0]
    velum = struct.unpack_from('<f', slot_bytes, vb + VOC_VELUM_TARGET)[0]
    auto_wobble = struct.unpack_from('<I', slot_bytes, vb + VOC_AUTO_WOBBLE)[0]
    always_voice = struct.unpack_from('<I', slot_bytes, vb + VOC_ALWAYS_VOICE)[0]
    synth_time = struct.unpack_from('<d', slot_bytes, vb + VOC_SYNTH_TIME)[0]

    # Diameters (44 floats)
    diameters = struct.unpack_from(f'<{VOC_NUM_DIAMETERS}f', slot_bytes, vb + VOC_DIAMETERS)

    # Normalize
    out = np.zeros(56, dtype=np.float32)

    # Frequency: log scale, log(f/140) / log(1000/140) -> ~[0, 1]
    freq_clamped = max(freq, 1.0)
    out[0] = max(0.0, min(1.0, math.log(freq_clamped / 140.0) / math.log(1000.0 / 140.0)))

    out[1] = tenseness  # already [0, 1]
    out[2] = (tongue_idx - 12.0) / 17.0
    out[3] = (tongue_diam - 2.0) / 1.5
    out[4] = vibrato_amt / 0.1
    out[5] = vibrato_freq / 20.0
    out[6] = intensity   # already [0, 1]
    out[7] = loudness    # already [0, 1]
    out[8] = (velum - 0.01) / 0.39
    out[9] = float(auto_wobble)
    out[10] = float(always_voice)
    out[11] = math.fmod(synth_time, 10.0) / 10.0

    # Diameters normalized by max expected diameter
    for i in range(VOC_NUM_DIAMETERS):
        out[12 + i] = diameters[i] / 3.5

    return torch.from_numpy(out)  # Stay on CPU; _to_device handles GPU transfer
