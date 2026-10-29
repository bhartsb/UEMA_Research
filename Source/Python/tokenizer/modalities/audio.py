# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Audio tokenizer — reads pre-computed FFT spectral data from SHM.

UE C++ HearingCaptureComponent computes a 256-bin stereo Hann-windowed FFT,
normalizes to [0,1] log-dB, and writes it into the audio slice. Python just
reads and reshapes. No GPU work needed.

Outputs: spectral [256, 2], metadata [5].
"""

import struct
from typing import Tuple

import numpy as np
import torch

from ..config import (
    TokenizerConfig,
    AUDIO_OFFSET,
    AUD_NUM_SAMPLES, AUD_SAMPLE_RATE, AUD_NUM_CHANNELS,
    AUD_LEFT_RMS, AUD_RIGHT_RMS, AUD_ILD_DB, AUD_ITD_MS,
    AUD_AZIMUTH, AUD_FRONT_BACK,
    AUD_SPECTRAL_NBINS, AUD_SPECTRAL,
    AUDIO_SPECTRAL_BINS,
)


def tokenize_audio(
    slot_bytes: bytes,
    config: TokenizerConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read pre-computed spectral and metadata from SHM audio slice.

    Args:
        slot_bytes: Full parity slot bytes.
        config: Tokenizer configuration.

    Returns:
        spectral: [n_bins, 2] float32 CPU (log-dB magnitude, normalized [0,1])
        metadata: [5] float32 CPU (ILD_norm, ITD_norm, az_norm, fb_conf, left_rms)
    """
    ab = AUDIO_OFFSET

    # Parse binaural metadata
    left_rms = struct.unpack_from('<f', slot_bytes, ab + AUD_LEFT_RMS)[0]
    ild_db = struct.unpack_from('<f', slot_bytes, ab + AUD_ILD_DB)[0]
    itd_ms = struct.unpack_from('<f', slot_bytes, ab + AUD_ITD_MS)[0]
    azimuth = struct.unpack_from('<f', slot_bytes, ab + AUD_AZIMUTH)[0]
    fb_conf = struct.unpack_from('<f', slot_bytes, ab + AUD_FRONT_BACK)[0]

    # Read pre-computed spectral data from SHM (written by UE C++ FFT engine)
    n_bins = struct.unpack_from('<i', slot_bytes, ab + AUD_SPECTRAL_NBINS)[0]
    n_bins = max(0, min(n_bins, AUDIO_SPECTRAL_BINS))

    if n_bins > 0:
        spec_raw = np.frombuffer(
            slot_bytes, dtype=np.float32,
            offset=ab + AUD_SPECTRAL, count=n_bins * 2,
        )
        # SHM layout: L[256] then R[256] → reshape to [2, 256] → transpose to [256, 2]
        spectral = spec_raw.reshape(2, n_bins).T.copy()
    else:
        spectral = np.zeros((config.fft_n_bins, 2), dtype=np.float32)

    spectral_t = torch.from_numpy(spectral)  # CPU tensor

    # Metadata normalization (CPU tensor)
    meta = torch.tensor([
        ild_db / 20.0,     # [-1, 1]
        itd_ms / 1.0,      # [-1, 1]
        azimuth / 90.0,    # [-1, 1]
        fb_conf,           # already [-1, 1]
        left_rms,          # already [0, 1]
    ], dtype=torch.float32)

    return spectral_t, meta
