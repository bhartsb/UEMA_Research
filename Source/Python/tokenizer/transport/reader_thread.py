# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Reader Thread — background capture thread for decoupling SHM polling from processing.

Busy-polls CPU shared memory and captures GPU vision textures into a ring buffer.
Main thread pulls the latest captured frame without blocking on SHM or GPU fences.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .cpu_reader import CPUReader
from .gpu_reader import GPUReader

log = logging.getLogger(__name__)


@dataclass
class CapturedFrame:
    """A single captured frame with CPU data and optional GPU vision tensors."""
    frame_number: int
    slot_bytes: bytes
    vision_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    capture_time: float  # time.perf_counter() when capture completed


class ReaderThread:
    """Background thread that continuously captures frames from SHM + GPU."""

    def __init__(self, cpu_reader: CPUReader, gpu_reader: GPUReader,
                 ring_size: int = 4):
        self._cpu_reader = cpu_reader
        self._gpu_reader = gpu_reader
        self._ring_size = ring_size

        # Ring buffer (lock-protected)
        self._ring: list[Optional[CapturedFrame]] = [None] * ring_size
        self._write_idx: int = 0
        self._lock = threading.Lock()

        # Thread state
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Stats
        self.frames_captured: int = 0
        self.frames_skipped: int = 0
        self._last_captured_frame: int = -1
        self._capture_times_ms: deque = deque(maxlen=1000)

    def start(self):
        """Start the background capture thread."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="LEM-ReaderThread")
        self._thread.start()
        log.info("Reader thread started (ring_size=%d)", self._ring_size)

    def stop(self):
        """Stop the background capture thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("Reader thread stopped (captured=%d, skipped=%d)",
                 self.frames_captured, self.frames_skipped)

    def get_latest_new_frame(self, after_frame: int) -> Optional[CapturedFrame]:
        """
        Return the most recent captured frame with frame_number > after_frame.
        Returns None if no new frame is available.
        Thread-safe (called from main thread).
        """
        with self._lock:
            # Search ring for the newest frame after after_frame
            best: Optional[CapturedFrame] = None
            for slot in self._ring:
                if slot is not None and slot.frame_number > after_frame:
                    if best is None or slot.frame_number > best.frame_number:
                        best = slot
            return best

    def get_stats(self) -> dict:
        """Return capture statistics."""
        times = list(self._capture_times_ms)
        if times:
            times_sorted = sorted(times)
            avg = sum(times) / len(times)
            return {
                "frames_captured": self.frames_captured,
                "frames_skipped": self.frames_skipped,
                "capture_avg_ms": avg,
                "capture_min_ms": times_sorted[0],
                "capture_max_ms": times_sorted[-1],
                "capture_p50_ms": times_sorted[len(times_sorted) // 2],
                "capture_p95_ms": times_sorted[int(len(times_sorted) * 0.95)],
            }
        return {
            "frames_captured": self.frames_captured,
            "frames_skipped": self.frames_skipped,
        }

    def _run(self):
        """Background thread main loop."""
        gpu_available = self._gpu_reader.available and self._gpu_reader._interop is not None
        interop = self._gpu_reader._interop if gpu_available else None

        tex_keys = ["pano_left", "pano_right", "fovea_left", "fovea_right"]

        while self._running:
            # 1. Poll CPU SHM
            result = self._cpu_reader.read_latest_cpu_frame()
            if result is None:
                time.sleep(0)  # yield, no timer dependency
                continue

            frame_number, slot_bytes = result
            t0 = time.perf_counter()

            # Track skips
            if self._last_captured_frame > 0:
                gap = frame_number - self._last_captured_frame - 1
                if gap > 0:
                    self.frames_skipped += gap

            self._last_captured_frame = frame_number

            # 2. GPU vision capture (N-1 frame)
            vision_tensors = None
            if gpu_available and frame_number > 1:
                try:
                    prev = frame_number - 1
                    slot = prev % 2
                    slot_name = f"slot_{slot}"

                    # Wait for D3D12 fence (N-1 guaranteed complete)
                    interop.wait_fence(prev)

                    # Copy array → linear (4 textures)
                    for key in tex_keys:
                        interop.copy_texture_to_linear(f"{slot_name}_{key}")

                    # No interop.synchronize() or torch.cuda.synchronize() —
                    # both copy_texture_to_linear (cuMemcpy2D on stream 0) and
                    # clone() (PyTorch on stream 0) are ordered on the same
                    # CUDA stream. A cuCtxSynchronize() here would stall the
                    # GPU pipeline including UE's D3D12 render work.
                    tensors = []
                    for key in tex_keys:
                        capsule = interop.to_dlpack_texture(f"{slot_name}_{key}")
                        tensor = torch.from_dlpack(capsule).clone()
                        tensors.append(tensor)

                    vision_tensors = tuple(tensors)
                except Exception as e:
                    log.warning("Reader thread GPU capture failed (frame %d): %s",
                                frame_number, e)

            t1 = time.perf_counter()
            capture_ms = (t1 - t0) * 1000.0
            self._capture_times_ms.append(capture_ms)

            # 3. Store in ring buffer
            captured = CapturedFrame(
                frame_number=frame_number,
                slot_bytes=slot_bytes,
                vision_tensors=vision_tensors,
                capture_time=t1,
            )

            with self._lock:
                self._ring[self._write_idx % self._ring_size] = captured
                self._write_idx += 1

            self.frames_captured += 1
