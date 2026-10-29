# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Unit tests for the audio tokenizer.

Tests read pre-computed spectral data from SHM (FFT computed in UE C++).
Constructs synthetic audio buffers with known spectral patterns and verifies
output shapes, dtypes, value ranges, and metadata normalization.
"""

import math
import struct
import sys
import os

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tokenizer.config import (
    TokenizerConfig,
    AUDIO_OFFSET, AUDIO_SIZE, FRAME_BUFFER_SIZE_ALIGNED,
    AUD_NUM_SAMPLES, AUD_SAMPLE_RATE, AUD_NUM_CHANNELS,
    AUD_LEFT_RMS, AUD_RIGHT_RMS, AUD_ILD_DB, AUD_ITD_MS,
    AUD_AZIMUTH, AUD_FRONT_BACK,
    AUD_SPECTRAL_NBINS, AUD_SPECTRAL,
    AUDIO_SPECTRAL_BINS,
)
from tokenizer.modalities.audio import tokenize_audio


def build_audio_slot(
    spectral_left: np.ndarray = None,
    spectral_right: np.ndarray = None,
    n_bins: int = 256,
    n_raw_samples: int = 480,
    sample_rate: int = 48000,
) -> bytes:
    """Build a fake parity slot with pre-computed spectral data.

    If spectral_left/right are None, generates a flat 0.5 spectrum.
    """
    buf = bytearray(FRAME_BUFFER_SIZE_ALIGNED)
    ab = AUDIO_OFFSET

    # Audio metadata header
    struct.pack_into('<i', buf, ab + AUD_NUM_SAMPLES, n_raw_samples)
    struct.pack_into('<i', buf, ab + AUD_SAMPLE_RATE, sample_rate)
    struct.pack_into('<i', buf, ab + AUD_NUM_CHANNELS, 2)
    struct.pack_into('<f', buf, ab + AUD_LEFT_RMS, 0.35)
    struct.pack_into('<f', buf, ab + AUD_RIGHT_RMS, 0.30)
    struct.pack_into('<f', buf, ab + AUD_ILD_DB, 3.5)
    struct.pack_into('<f', buf, ab + AUD_ITD_MS, 0.2)
    struct.pack_into('<f', buf, ab + AUD_AZIMUTH, 45.0)
    struct.pack_into('<f', buf, ab + AUD_FRONT_BACK, 0.8)

    # Spectral bin count
    struct.pack_into('<i', buf, ab + AUD_SPECTRAL_NBINS, n_bins)

    # Spectral data: L[n_bins] then R[n_bins]
    if spectral_left is None:
        spectral_left = np.full(n_bins, 0.5, dtype=np.float32)
    if spectral_right is None:
        spectral_right = np.full(n_bins, 0.5, dtype=np.float32)

    spec_offset = ab + AUD_SPECTRAL
    buf[spec_offset:spec_offset + n_bins * 4] = spectral_left[:n_bins].astype(np.float32).tobytes()
    buf[spec_offset + n_bins * 4:spec_offset + n_bins * 8] = spectral_right[:n_bins].astype(np.float32).tobytes()

    return bytes(buf)


def test_audio_shapes():
    """Verify output tensor shapes, dtypes, and devices (CPU)."""
    config = TokenizerConfig()
    slot = build_audio_slot()

    spectral, meta = tokenize_audio(slot, config)

    assert spectral.shape == (256, 2), f"Spectral shape: {spectral.shape}"
    assert meta.shape == (5,), f"Meta shape: {meta.shape}"

    assert spectral.dtype == torch.float32
    assert meta.dtype == torch.float32

    # Should be CPU tensors (FFT done in UE C++, no GPU needed)
    assert spectral.device.type == "cpu"
    assert meta.device.type == "cpu"

    print("PASS: test_audio_shapes")


def test_audio_spectral_range():
    """Verify spectral output is in [0, 1]."""
    config = TokenizerConfig()

    left = np.random.rand(256).astype(np.float32)
    right = np.random.rand(256).astype(np.float32)
    slot = build_audio_slot(spectral_left=left, spectral_right=right)

    spectral, _ = tokenize_audio(slot, config)

    assert spectral.min() >= 0.0, f"Spectral min: {spectral.min()}"
    assert spectral.max() <= 1.0, f"Spectral max: {spectral.max()}"

    print("PASS: test_audio_spectral_range")


def test_audio_spectral_readback():
    """Verify spectral data round-trips through SHM layout correctly."""
    config = TokenizerConfig()

    # Create known spectral pattern
    left = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    right = np.linspace(1.0, 0.0, 256, dtype=np.float32)
    slot = build_audio_slot(spectral_left=left, spectral_right=right)

    spectral, _ = tokenize_audio(slot, config)

    # spectral should be [256, 2] with left in column 0, right in column 1
    np.testing.assert_allclose(spectral[:, 0].numpy(), left, atol=1e-6)
    np.testing.assert_allclose(spectral[:, 1].numpy(), right, atol=1e-6)

    print("PASS: test_audio_spectral_readback")


def test_audio_peak_detection():
    """Verify that a spectral peak at a known bin is preserved."""
    config = TokenizerConfig()

    # Create spectrum with a clear peak at bin 10
    left = np.full(256, 0.1, dtype=np.float32)
    left[10] = 0.95
    right = np.full(256, 0.1, dtype=np.float32)
    right[10] = 0.90

    slot = build_audio_slot(spectral_left=left, spectral_right=right)
    spectral, _ = tokenize_audio(slot, config)

    left_spectrum = spectral[:, 0]
    peak_bin = torch.argmax(left_spectrum).item()
    assert peak_bin == 10, f"Peak at bin {peak_bin}, expected 10"

    print(f"PASS: test_audio_peak_detection (peak at bin {peak_bin})")


def test_audio_metadata_normalization():
    """Verify metadata normalization values."""
    config = TokenizerConfig()
    slot = build_audio_slot()

    _, meta = tokenize_audio(slot, config)
    meta_np = meta.numpy()

    # ILD_dB=3.5 / 20.0 = 0.175
    assert abs(meta_np[0] - 0.175) < 0.001, f"ILD norm: {meta_np[0]}"
    # ITD_ms=0.2 / 1.0 = 0.2
    assert abs(meta_np[1] - 0.2) < 0.001, f"ITD norm: {meta_np[1]}"
    # Azimuth=45.0 / 90.0 = 0.5
    assert abs(meta_np[2] - 0.5) < 0.001, f"Az norm: {meta_np[2]}"
    # FrontBack=0.8
    assert abs(meta_np[3] - 0.8) < 0.001, f"FB: {meta_np[3]}"
    # LeftRMS=0.35
    assert abs(meta_np[4] - 0.35) < 0.001, f"RMS: {meta_np[4]}"

    print("PASS: test_audio_metadata_normalization")


def test_audio_empty():
    """Verify graceful handling of zero spectral bins."""
    config = TokenizerConfig()

    # Build slot with n_bins=0 (no spectral data from UE)
    slot = build_audio_slot(n_bins=0)

    spectral, meta = tokenize_audio(slot, config)

    assert spectral.shape == (256, 2)
    assert (spectral == 0).all(), "Empty spectral should be zeros"

    print("PASS: test_audio_empty")


if __name__ == '__main__':
    test_audio_shapes()
    test_audio_spectral_range()
    test_audio_spectral_readback()
    test_audio_peak_detection()
    test_audio_metadata_normalization()
    test_audio_empty()
    print("\nAll audio tests passed!")
