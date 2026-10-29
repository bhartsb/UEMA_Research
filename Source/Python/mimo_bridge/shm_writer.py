# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Mode B transport: write LEM_Pipeline + LEM_Vision exactly as UE does, read LEM_Action_Buffer.

Layouts come from the model's own reader constants (tokenizer/config.py) and the model's own
action writer (lem/action/writer.py, lem/action/shm.py). Windows named shared memory via mmap
tagnames, the same call the model's CPUReader uses.

LEM_Pipeline (62720 B):
  0   uint32 magic "LEMP" (0x4C454D50)   4 uint32 version 1   8 int64 latest frame
  16  uint32 num_slots 2  20 uint32 slot_size 31232  24 uint32 cpu_data_offset 256
  32  char[200] manifest path (zeros)   256.. slot 0, slot 1 (frame % 2)
LEM_Vision (~33.5 MB): header 64 B (magic "LEMV", version, latest frame, pano w/h, fovea w/h,
  bpp 4, pixel offset 64, writing flag) then pano L, pano R, fovea L, fovea R as BGRA8.
LEM_Action_Buffer (64 + 1176 B): the model creates it; we read frame_id, flags, 23x9 motor
  floats, 2 gaze floats.
"""
from __future__ import annotations

import mmap
import struct
import time
from typing import Optional, Tuple

import numpy as np

from tokenizer import config as C
from lem.action import writer as AW
from lem.action import shm as ASHM


class PipelineSHMWriter:
    def __init__(self, name: str = "LEM_Pipeline", manifest_path: str = ""):
        self.name = name
        self.total = C.SHM_TOTAL_SIZE
        self.num_slots = C.NUM_PARITY_SLOTS
        self.slot_size = C.FRAME_BUFFER_SIZE_ALIGNED
        self.data_offset = C.SHM_CPU_DATA_OFFSET
        self.shm = mmap.mmap(-1, self.total, name, access=mmap.ACCESS_WRITE)
        self.shm[:] = b"\x00" * self.total
        struct.pack_into("<I", self.shm, C.SHM_OFF_MAGIC, C.SHM_MAGIC)
        struct.pack_into("<I", self.shm, C.SHM_OFF_VERSION, C.SHM_VERSION)
        struct.pack_into("<q", self.shm, C.SHM_OFF_LATEST_FRAME, 0)
        struct.pack_into("<III", self.shm, C.SHM_OFF_NUM_SLOTS, self.num_slots, self.slot_size, self.data_offset)
        mp = manifest_path.encode("utf-8")[:C.SHM_MANIFEST_PATH_MAX - 1]
        self.shm[C.SHM_OFF_MANIFEST_PATH:C.SHM_OFF_MANIFEST_PATH + len(mp)] = mp
        self.frames_written = 0

    def write(self, frame_number: int, slot_bytes: bytes) -> None:
        """slot_bytes: a full slot (31232 B) whose header frame_number == frame_number."""
        if len(slot_bytes) != self.slot_size:
            raise ValueError(f"slot must be {self.slot_size} bytes, got {len(slot_bytes)}")
        slot = frame_number % self.num_slots
        base = self.data_offset + slot * self.slot_size
        self.shm[base:base + self.slot_size] = slot_bytes
        struct.pack_into("<q", self.shm, C.SHM_OFF_LATEST_FRAME, int(frame_number))   # publish last (like UE)
        self.frames_written += 1

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass


class VisionSHMWriter:
    def __init__(self, name: str = "LEM_Vision"):
        self.name = name
        self.total = C.VISION_TOTAL_SIZE
        self.shm = mmap.mmap(-1, self.total, name, access=mmap.ACCESS_WRITE)
        struct.pack_into("<I", self.shm, C.VISION_OFF_MAGIC, C.VISION_MAGIC)
        struct.pack_into("<I", self.shm, C.VISION_OFF_VERSION, C.VISION_VERSION)
        struct.pack_into("<q", self.shm, C.VISION_OFF_LATEST_FRAME, 0)
        struct.pack_into("<iiii", self.shm, C.VISION_OFF_PANO_WIDTH, C.PANO_WIDTH, C.PANO_HEIGHT, C.FOVEA_WIDTH, C.FOVEA_HEIGHT)
        struct.pack_into("<i", self.shm, C.VISION_OFF_BPP, 4)
        struct.pack_into("<i", self.shm, C.VISION_OFF_PIXEL_OFFSET, C.VISION_PIXEL_OFFSET)
        struct.pack_into("<i", self.shm, C.VISION_OFF_WRITING, 0)

    def write(self, frame_number: int, left_eye_rgb: np.ndarray, right_eye_rgb: np.ndarray) -> None:
        """Each eye is letterboxed (no scaling) into its panorama AND its fovea region as BGRA8."""
        from mimo_bridge.vision_adapter import letterbox_to_bgra
        struct.pack_into("<i", self.shm, C.VISION_OFF_WRITING, 1)
        lp = letterbox_to_bgra(left_eye_rgb, C.PANO_WIDTH, C.PANO_HEIGHT).tobytes()
        rp = letterbox_to_bgra(right_eye_rgb, C.PANO_WIDTH, C.PANO_HEIGHT).tobytes()
        lf = letterbox_to_bgra(left_eye_rgb, C.FOVEA_WIDTH, C.FOVEA_HEIGHT).tobytes()
        rf = letterbox_to_bgra(right_eye_rgb, C.FOVEA_WIDTH, C.FOVEA_HEIGHT).tobytes()
        self.shm[C.VISION_OFF_LEFT_PANO:C.VISION_OFF_LEFT_PANO + len(lp)] = lp
        self.shm[C.VISION_OFF_RIGHT_PANO:C.VISION_OFF_RIGHT_PANO + len(rp)] = rp
        self.shm[C.VISION_OFF_LEFT_FOVEA:C.VISION_OFF_LEFT_FOVEA + len(lf)] = lf
        self.shm[C.VISION_OFF_RIGHT_FOVEA:C.VISION_OFF_RIGHT_FOVEA + len(rf)] = rf
        struct.pack_into("<q", self.shm, C.VISION_OFF_LATEST_FRAME, int(frame_number))
        struct.pack_into("<i", self.shm, C.VISION_OFF_WRITING, 0)

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass


class ActionSHMReader:
    """Reads what the model's ActionBufferWriter wrote (lem/action/writer.py layout)."""

    def __init__(self, name: str = "LEM_Action_Buffer", buffer_size: int = AW.ACTION_FRAME_SIZE, retries: int = 600):
        self.name = name
        self.total = ASHM.ACTION_SHM_HEADER_SIZE + buffer_size
        self.shm = None
        for _ in range(retries):
            try:
                self.shm = mmap.mmap(-1, self.total, name, access=mmap.ACCESS_READ)
                magic = struct.unpack_from("<I", self.shm, ASHM.OFF_SHM_MAGIC)[0]
                if magic == ASHM.ACTION_MAGIC:
                    break
                self.shm.close(); self.shm = None
            except Exception:
                self.shm = None
            time.sleep(0.1)
        self.last_frame_id = 0

    @property
    def is_open(self) -> bool:
        return self.shm is not None

    def read(self) -> Optional[Tuple[int, int, np.ndarray, np.ndarray]]:
        """Returns (frame_id, flags, motor [23,9], gaze [2]) when a new action is present, else None."""
        if self.shm is None:
            return None
        base = ASHM.ACTION_SHM_HEADER_SIZE
        frame_id, ts_us, flags = struct.unpack_from(AW._HEADER_FMT, self.shm, base + AW.OFF_FRAME_ID)
        if frame_id <= self.last_frame_id:
            return None
        motor = np.frombuffer(self.shm, dtype="<f4", count=23 * 9, offset=base + AW.OFF_MOTOR).reshape(23, 9).copy()
        gaze = np.frombuffer(self.shm, dtype="<f4", count=2, offset=base + AW.OFF_GAZE).copy()
        self.last_frame_id = frame_id
        return frame_id, flags, motor, gaze

    def close(self):
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception:
                pass
            self.shm = None
