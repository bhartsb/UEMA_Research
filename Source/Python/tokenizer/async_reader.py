# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
AsyncFrameReader — background thread continuously polling SHM.

Maintains a ring buffer of tokenized frames. Training loop calls
get_latest_frame() which returns immediately with the most recent
frame or None. Stale frames (reused across training steps) get
source_mask set to "stale" by the caller.

Thread safety: single producer (reader thread), single consumer (training thread).
Uses a threading.Lock on the ring buffer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional, Tuple

import torch

from .config import TokenizedFrame
from .tokenizer import LEMTokenizer

log = logging.getLogger(__name__)


class AsyncFrameReader:
    """
    Background-threaded frame reader with ring buffer.

    Usage:
        reader = AsyncFrameReader(tokenizer, ring_size=3, vision_every_n=2)
        reader.start()
        while training:
            result = reader.get_latest_frame()
            if result is None:
                continue
            frame, is_stale, capture_event = result
            ...
        reader.close()
    """

    def __init__(self, tokenizer: LEMTokenizer, ring_size: int = 3,
                 max_stale_age_ms: float = 50.0, vision_every_n: int = 1):
        self._tokenizer = tokenizer
        self._ring_size = ring_size
        self._max_stale_age_ms = max_stale_age_ms
        self._vision_every_n = vision_every_n

        self._ring: deque = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0  # overwritten before consumed
        self.throttle_sleeps = 0  # ring full, yielded GPU
        self._last_consumed_frame: int = -1

    def start(self):
        """Start the background reader thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True,
                                        name="AsyncFrameReader")
        self._thread.start()
        log.info("AsyncFrameReader started (ring_size=%d, max_stale_age=%.0fms)",
                 self._ring_size, self._max_stale_age_ms)

    def stop(self):
        """Stop the background reader thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("AsyncFrameReader stopped (captured=%d, dropped=%d, throttled=%d)",
                 self.frames_captured, self.frames_dropped, self.throttle_sleeps)

    def _reader_loop(self):
        """Background thread: continuously poll SHM and tokenize."""
        capture_device = self._tokenizer.config.device
        use_cuda = (capture_device is not None and
                    hasattr(capture_device, 'type') and
                    capture_device.type == 'cuda')
        vision_counter = 0

        while self._running:
            # Throttle: if ring is full and nothing consumed, sleep to yield GPU
            with self._lock:
                ring_full = len(self._ring) == self._ring_size
            if ring_full:
                self.throttle_sleeps += 1
                time.sleep(0.002)  # 2ms yield (was 8ms)
                continue

            # Skip vision capture on non-vision frames to avoid GPU contention
            # with UE rendering on the capture device (WDDM GPU).
            skip_vision = (self._vision_every_n > 1 and
                           vision_counter % self._vision_every_n != 0)

            try:
                frame = self._tokenizer.read_frame(skip_vision=skip_vision)
            except Exception:
                log.exception("AsyncFrameReader: error reading frame")
                time.sleep(0.01)
                continue

            if frame is not None:
                vision_counter += 1
                # Record CUDA event marking completion of all tokenizer GPU work
                # on the capture device. The training thread can wait for this
                # event instead of a full torch.cuda.synchronize(device).
                event = None
                if use_cuda:
                    event = torch.cuda.Event()
                    event.record()  # Records on stream 0 of capture_device
                timestamp = time.perf_counter()
                with self._lock:
                    if len(self._ring) == self._ring_size:
                        self.frames_dropped += 1
                    self._ring.append((frame, timestamp, event))
                self.frames_captured += 1
            else:
                # No frame available — yield CPU (1ms for Windows Ctrl+C safety)
                time.sleep(0.001)

    def get_latest_frame(self) -> Optional[Tuple[TokenizedFrame, bool, Optional[torch.cuda.Event]]]:
        """
        Non-blocking: return (TokenizedFrame, is_stale, capture_event) or None.

        A frame is stale if it was already consumed or older than max_stale_age_ms.
        Drains old frames from the ring so the reader can capture new ones.

        capture_event: CUDA event recorded after tokenizer's GPU work completed.
        Call event.synchronize() instead of torch.cuda.synchronize(device) to
        avoid blocking on unrelated GPU work (e.g. UE rendering).
        """
        with self._lock:
            if not self._ring:
                return None
            frame, timestamp, event = self._ring[-1]  # most recent
            self._ring.clear()  # drain — frees slots for reader thread

        age_ms = (time.perf_counter() - timestamp) * 1000.0
        fn = frame.metadata.frame_number
        is_stale = (fn <= self._last_consumed_frame) or (age_ms > self._max_stale_age_ms)
        self._last_consumed_frame = fn

        return (frame, is_stale, event)

    def get_stats(self) -> dict:
        """Return current ring utilization stats (thread-safe snapshot)."""
        with self._lock:
            fill_level = len(self._ring)
        return {
            "fill_level": fill_level,
            "ring_size": self._ring_size,
            "captured": self.frames_captured,
            "dropped": self.frames_dropped,
            "throttle_sleeps": self.throttle_sleeps,
        }

    def close(self):
        """Stop thread and release tokenizer."""
        self.stop()
        self._tokenizer.close()

    @property
    def frames_processed(self):
        """Proxy for tokenizer.frames_processed."""
        return self._tokenizer.frames_processed

    @property
    def frames_skipped(self):
        """Proxy for tokenizer.frames_skipped."""
        return self._tokenizer.frames_skipped
