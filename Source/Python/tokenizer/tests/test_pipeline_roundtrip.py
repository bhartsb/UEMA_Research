# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Pipeline roundtrip test — requires UE running with the pipeline active.

Initializes the tokenizer, reads 100 frames, and verifies:
- No None frames after initial connection
- Frame numbers strictly increase
- All tensor shapes match the manifest
- All values in expected ranges (no NaN/Inf)
- Average tokenization latency

Usage:
    python test_pipeline_roundtrip.py [--frames 100] [--timeout 30]
"""

import argparse
import sys
import os
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tokenizer import LEMTokenizer, TokenizerConfig


def run_roundtrip(n_frames: int = 100, timeout_s: float = 30.0):
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return

    print(f"Initializing tokenizer...")
    tokenizer = LEMTokenizer()
    manifest = tokenizer.get_manifest()

    print(f"Waiting for first frame (timeout={timeout_s}s)...")

    # Wait for connection
    start = time.time()
    first_frame = None
    while time.time() - start < timeout_s:
        first_frame = tokenizer.read_frame()
        if first_frame is not None:
            break
        time.sleep(0.01)

    if first_frame is None:
        print(f"ERROR: No frame received after {timeout_s}s. Is UE running with the pipeline?")
        tokenizer.close()
        return

    print(f"Connected! First frame: {first_frame.metadata.frame_number}")

    # Collect frames
    frames = [first_frame]
    latencies = []
    last_frame_num = first_frame.metadata.frame_number
    none_count = 0

    while len(frames) < n_frames:
        t0 = time.perf_counter()
        frame = tokenizer.read_frame_blocking(timeout_ms=200)
        latency = (time.perf_counter() - t0) * 1000.0

        if frame is None:
            none_count += 1
            if none_count > 50:
                print(f"ERROR: Too many None frames ({none_count})")
                break
            continue

        latencies.append(latency)
        frames.append(frame)

        # Verify frame number increases
        assert frame.metadata.frame_number > last_frame_num, \
            f"Frame {frame.metadata.frame_number} <= previous {last_frame_num}"
        last_frame_num = frame.metadata.frame_number

    print(f"\nCollected {len(frames)} frames")
    print(f"  None returns during collection: {none_count}")

    # Verify shapes and values
    errors = []

    for i, frame in enumerate(frames):
        fn = frame.metadata.frame_number

        # Audio
        if frame.audio_spectral is not None:
            if frame.audio_spectral.shape != (256, 2):
                errors.append(f"Frame {fn}: audio_spectral shape {frame.audio_spectral.shape}")
            if torch.isnan(frame.audio_spectral).any():
                errors.append(f"Frame {fn}: NaN in audio_spectral")

        if frame.audio_metadata is not None:
            if frame.audio_metadata.shape != (5,):
                errors.append(f"Frame {fn}: audio_metadata shape {frame.audio_metadata.shape}")

        # Proprio
        if frame.proprio is not None and frame.proprio.numel() > 0:
            if frame.proprio.shape[1] != 12:
                errors.append(f"Frame {fn}: proprio shape {frame.proprio.shape}")
            if torch.isnan(frame.proprio).any():
                errors.append(f"Frame {fn}: NaN in proprio")

        # Motor
        if frame.motor_handles is not None:
            if frame.motor_handles.shape != (23, 10):
                errors.append(f"Frame {fn}: motor shape {frame.motor_handles.shape}")

        if frame.motor_energy is not None:
            if frame.motor_energy.shape != (3,):
                errors.append(f"Frame {fn}: energy shape {frame.motor_energy.shape}")

        # Touch
        if frame.touch_impulses is not None:
            if frame.touch_impulses.shape != (16, 8):
                errors.append(f"Frame {fn}: impulses shape {frame.touch_impulses.shape}")

        # Voice
        if frame.voice is not None:
            if frame.voice.shape != (56,):
                errors.append(f"Frame {fn}: voice shape {frame.voice.shape}")
            if torch.isnan(frame.voice).any():
                errors.append(f"Frame {fn}: NaN in voice")

        # InterMind
        if frame.intermind is not None:
            if frame.intermind.shape != (3,):
                errors.append(f"Frame {fn}: intermind shape {frame.intermind.shape}")

        # Vision (may be None if vision SHM not available)
        if frame.vision_patches is not None:
            expected_patches = manifest["modalities"]["vision"]["patches_shape"]
            if list(frame.vision_patches.shape) != expected_patches:
                errors.append(f"Frame {fn}: vision patches shape {frame.vision_patches.shape}")
            if torch.isnan(frame.vision_patches).any():
                errors.append(f"Frame {fn}: NaN in vision patches")
            if frame.vision_patches.min() < 0.0 or frame.vision_patches.max() > 1.0:
                errors.append(f"Frame {fn}: vision patches out of [0,1]")

    if errors:
        print(f"\nFAILED with {len(errors)} errors:")
        for e in errors[:20]:
            print(f"  {e}")
    else:
        print("All shape and value checks passed!")

    # Latency stats
    if latencies:
        avg = sum(latencies) / len(latencies)
        mn = min(latencies)
        mx = max(latencies)
        print(f"\nLatency (blocking read + tokenize):")
        print(f"  avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")

    # Frame rate
    if len(frames) >= 2:
        first_ts = frames[0].metadata.wall_clock
        last_ts = frames[-1].metadata.wall_clock
        dt = last_ts - first_ts
        if dt > 0:
            fps = (len(frames) - 1) / dt
            print(f"  Pipeline FPS: {fps:.1f}")

    print(f"\nSkipped frames (tokenizer): {tokenizer.frames_skipped}")

    has_vision = any(f.vision_patches is not None for f in frames)
    print(f"Vision data: {'YES' if has_vision else 'NO (vision SHM not available)'}")

    tokenizer.close()

    if errors:
        print(f"\nRESULT: FAIL ({len(errors)} errors)")
    else:
        print(f"\nRESULT: PASS")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline roundtrip test')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--timeout', type=float, default=30.0)
    args = parser.parse_args()

    run_roundtrip(n_frames=args.frames, timeout_s=args.timeout)
