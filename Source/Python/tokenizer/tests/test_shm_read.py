# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
SHM Tactile Read Diagnostic — verify C++ and Python agree on the 8-field
derived tactile patch layout before running training.

Reads the touch slice directly from shared memory (no torch, no GPU).
Prints raw field ranges, normalization check, and multi-frame staleness.

Usage:
    cd Source\\Python
    python -m tokenizer.tests.test_shm_read
    python -m tokenizer.tests.test_shm_read --frames 10
"""

import argparse
import mmap
import struct
import sys
import time

import numpy as np

from tokenizer.config import (
    SHM_TOTAL_SIZE, SHM_MAGIC, SHM_OFF_MAGIC, SHM_OFF_LATEST_FRAME,
    SHM_CPU_DATA_OFFSET, FRAME_BUFFER_SIZE_ALIGNED,
    TOUCH_OFFSET, TOUCH_MAX_IMPULSES,
    TCH_NUM_IMPULSES, TCH_IMPULSE_DATA,
    HDR_MODALITY_MASK, MOD_TOUCH,
)
from tokenizer.modalities.touch import _FIELD_NORMS, _PATCH_DT

_FIELD_NAMES = [
    "normalPressure (Pa)",
    "shearMagnitude (Pa)",
    "slipRate (mm/s)",
    "contactArea (mm\u00b2)",
    "vibPower_5Hz (N²/bin)",
    "vibPower_10Hz (N²/bin)",
    "vibPower_15Hz (N²/bin)",
    "vibPower_20Hz (N²/bin)",
]

_SHM_NAME = "LEM_Pipeline"


def _open_shm():
    """Open the CPU shared memory region. Returns mmap or None."""
    try:
        return mmap.mmap(-1, SHM_TOTAL_SIZE, _SHM_NAME, access=mmap.ACCESS_READ)
    except Exception as e:
        print(f"ERROR: Cannot open SHM '{_SHM_NAME}': {e}")
        print("Is UE5 running with the pipeline active?")
        return None


def _read_slot(shm):
    """Read the latest parity slot bytes and frame number."""
    magic = struct.unpack_from('<I', shm, SHM_OFF_MAGIC)[0]
    if magic != SHM_MAGIC:
        print(f"ERROR: Bad SHM magic: 0x{magic:08X} (expected 0x{SHM_MAGIC:08X})")
        return None, -1

    frame_num = struct.unpack_from('<q', shm, SHM_OFF_LATEST_FRAME)[0]
    slot_idx = frame_num % 2
    base = SHM_CPU_DATA_OFFSET + slot_idx * FRAME_BUFFER_SIZE_ALIGNED
    slot_bytes = bytes(shm[base:base + FRAME_BUFFER_SIZE_ALIGNED])
    return slot_bytes, frame_num


def _parse_touch(slot_bytes):
    """Extract raw (unnormalized) patch data from a slot buffer.

    Returns (raw_patches [n, 8] float32, n_patches int).
    """
    tb = TOUCH_OFFSET
    n_patches = struct.unpack_from('<i', slot_bytes, tb + TCH_NUM_IMPULSES)[0]
    n_patches = max(0, min(n_patches, TOUCH_MAX_IMPULSES))

    if n_patches == 0:
        return np.zeros((0, 8), dtype=np.float32), 0

    patch_start = tb + TCH_IMPULSE_DATA
    raw = np.frombuffer(slot_bytes, dtype=_PATCH_DT, count=n_patches, offset=patch_start)
    return raw['fields'].copy(), n_patches


def _print_field_ranges(patches, n_patches):
    """Print min/max/mean for each raw field."""
    print(f"\nField ranges (across {n_patches} active sensors):")
    for i, name in enumerate(_FIELD_NAMES):
        col = patches[:, i]
        print(f"  [{i}] {name:25s}  min={col.min():12.4f}  max={col.max():12.4f}  mean={col.mean():12.4f}")


def _print_active_dead(patches, n_patches):
    """Print active vs dead sensor counts."""
    any_nonzero = np.any(patches != 0, axis=1)
    n_active = int(any_nonzero.sum())
    n_dead = n_patches - n_active
    print(f"\nActive sensors (any field > 0): {n_active} / {n_patches}")
    print(f"Dead sensors (all fields = 0):  {n_dead} / {n_patches}")
    return n_active


def _print_normalization_check(patches):
    """Show what values look like after applying tokenizer normalization."""
    normed = patches / _FIELD_NORMS
    print("\nNormalization check (applying tokenizer divisors):")
    for i, name in enumerate(_FIELD_NAMES):
        col = normed[:, i]
        divisor = _FIELD_NORMS[i]
        status = ""
        if col.max() > 2.0:
            status = "  <-- WARNING: >2.0, divisor may be too small"
        div_str = f"/ {divisor:g}"
        print(f"  [{i}] {name:25s} {div_str:>10s}:  min={col.min():.6f}  max={col.max():.6f}{status}")


def _multi_frame_sample(shm, n_frames):
    """Read multiple frames and report per-field deltas."""
    print(f"\n--- Multi-frame sample ({n_frames} frames) ---")
    prev = None
    deltas = []
    frames_seen = 0
    last_frame_num = -1

    for _ in range(n_frames * 20):  # poll up to 20× per requested frame
        slot_bytes, frame_num = _read_slot(shm)
        if slot_bytes is None:
            break
        if frame_num == last_frame_num:
            time.sleep(0.001)  # Windows Ctrl+C safe
            continue

        last_frame_num = frame_num
        patches, n_patches = _parse_touch(slot_bytes)

        if n_patches > 0 and prev is not None:
            # Compare overlapping sensor range
            overlap = min(patches.shape[0], prev.shape[0])
            if overlap > 0:
                delta = np.abs(patches[:overlap] - prev[:overlap]).mean(axis=0)
                deltas.append(delta)

        if n_patches > 0:
            prev = patches.copy()

        frames_seen += 1
        if frames_seen >= n_frames:
            break
        time.sleep(0.02)

    if not deltas:
        print("WARNING: Could not collect frame-to-frame deltas (too few frames or no sensors).")
        return

    mean_delta = np.mean(deltas, axis=0)
    print("\nFrame-to-frame delta (mean absolute change per field):")
    for i, name in enumerate(_FIELD_NAMES):
        status = "changing" if mean_delta[i] > 1e-9 else "static"
        print(f"  [{i}] {name:25s}  \u0394={mean_delta[i]:.6f}  ({status})")

    all_static = np.all(mean_delta < 1e-9)
    if all_static:
        print("\nWARNING: No change across frames. SHM may be stale (UE5 paused or not writing).")


def main():
    parser = argparse.ArgumentParser(description="SHM Tactile Read Diagnostic")
    parser.add_argument("--frames", type=int, default=5,
                        help="Number of frames to sample for staleness check (default: 5)")
    args = parser.parse_args()

    shm = _open_shm()
    if shm is None:
        sys.exit(1)

    try:
        slot_bytes, frame_num = _read_slot(shm)
        if slot_bytes is None:
            sys.exit(1)

        print("=== SHM Tactile Diagnostic ===")
        print(f"Frame number: {frame_num}")

        # Check touch modality bit
        mod_mask = struct.unpack_from('<I', slot_bytes, HDR_MODALITY_MASK)[0]
        touch_valid = bool(mod_mask & MOD_TOUCH)
        print(f"Touch modality bit: {'SET' if touch_valid else 'NOT SET'}")
        if not touch_valid:
            print("WARNING: Touch modality not marked as filled in this frame.")

        # Parse touch
        patches, n_patches = _parse_touch(slot_bytes)
        print(f"\nSensors reported: {n_patches} / {TOUCH_MAX_IMPULSES} max")

        if n_patches > TOUCH_MAX_IMPULSES:
            print(f"WARNING: num_sensors={n_patches} exceeds max {TOUCH_MAX_IMPULSES}. Check C++ TOUCH_MAX_IMPULSES.")

        if n_patches == 0:
            print("\nWARNING: No active sensors. Is the avatar making contact with anything?")
            print("(Touch slice header reports 0 patches.)")
        else:
            _print_field_ranges(patches, n_patches)
            n_active = _print_active_dead(patches, n_patches)

            if n_active == 0:
                print("\nWARNING: All sensors report zero values. Is the avatar making contact?")
            else:
                _print_normalization_check(patches)

        # Multi-frame sampling
        if args.frames > 1:
            _multi_frame_sample(shm, args.frames)

    finally:
        shm.close()


if __name__ == '__main__':
    # Allow running as: python tokenizer/tests/test_shm_read.py
    import os
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _source_python = os.path.normpath(os.path.join(_script_dir, '..', '..'))
    if _source_python not in sys.path:
        sys.path.insert(0, _source_python)

    main()
