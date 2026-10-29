# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
CPU Reader — shared memory reader for LEM pipeline data.

Opens two Windows named shared memory regions:
  - LEM_Pipeline (27,904 bytes): CPU modality data (audio, proprio, motor, touch, voice)
  - LEM_Vision   (~33.5 MB):    GPU-readback vision pixels (pano L/R, fovea L/R)

Follows the same mmap pattern as Scripts/pipeline_smoke_test.py.
"""

import logging
import mmap
import struct
from typing import Optional, Tuple

from ..config import (
    TokenizerConfig,
    SHM_MAGIC, SHM_HEADER_SIZE,
    SHM_OFF_MAGIC, SHM_OFF_VERSION, SHM_OFF_LATEST_FRAME,
    SHM_OFF_NUM_SLOTS, SHM_OFF_SLOT_SIZE, SHM_OFF_CPU_DATA_OFFSET,
    SHM_OFF_MANIFEST_PATH, SHM_MANIFEST_PATH_MAX,
    SHM_TOTAL_SIZE,
    VISION_MAGIC,
    VISION_OFF_MAGIC, VISION_OFF_LATEST_FRAME, VISION_OFF_WRITING,
    VISION_OFF_PANO_WIDTH, VISION_OFF_PANO_HEIGHT,
    VISION_OFF_FOVEA_WIDTH, VISION_OFF_FOVEA_HEIGHT,
    VISION_OFF_BPP, VISION_OFF_PIXEL_OFFSET,
    VISION_TOTAL_SIZE,
    VISION_PANO_BYTES, VISION_FOVEA_BYTES,
)

log = logging.getLogger(__name__)


class CPUReader:
    """Reads pipeline data from Windows named shared memory regions."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self._cpu_shm: Optional[mmap.mmap] = None
        self._vision_shm: Optional[mmap.mmap] = None

        # Header info (populated on open)
        self.num_slots: int = 2
        self.slot_size: int = 13824
        self.cpu_data_offset: int = 256
        self.manifest_path: str = ""

        # State
        self.last_cpu_frame: int = -1
        self.last_vision_frame: int = -1
        self.has_vision: bool = False
        self._warned_no_pipeline: bool = False
        self._warned_no_vision: bool = False

    def open(self) -> bool:
        """Open shared memory regions. Returns True if CPU SHM opened successfully."""
        # --- CPU shared memory ---
        try:
            self._cpu_shm = mmap.mmap(
                -1, SHM_TOTAL_SIZE,
                self.config.cpu_shared_memory_name,
                access=mmap.ACCESS_READ,
            )
        except Exception as e:
            if not self._warned_no_pipeline:
                log.warning("Cannot open CPU shared memory '%s': %s",
                            self.config.cpu_shared_memory_name, e)
                self._warned_no_pipeline = True
            return False

        # Validate magic
        magic = struct.unpack_from('<I', self._cpu_shm, SHM_OFF_MAGIC)[0]
        if magic != SHM_MAGIC:
            log.error("Bad CPU SHM magic: 0x%08X (expected 0x%08X)", magic, SHM_MAGIC)
            self._cpu_shm.close()
            self._cpu_shm = None
            return False

        # Read header
        self.num_slots, self.slot_size, self.cpu_data_offset = struct.unpack_from(
            '<III', self._cpu_shm, SHM_OFF_NUM_SLOTS
        )

        # Read manifest path
        raw = self._cpu_shm[SHM_OFF_MANIFEST_PATH:SHM_OFF_MANIFEST_PATH + SHM_MANIFEST_PATH_MAX]
        self.manifest_path = raw.split(b'\x00')[0].decode('utf-8', errors='replace')

        log.info("CPU SHM connected: slots=%d, slot_size=%d, manifest='%s'",
                 self.num_slots, self.slot_size, self.manifest_path)

        # --- Vision shared memory (optional) ---
        try:
            self._vision_shm = mmap.mmap(
                -1, VISION_TOTAL_SIZE,
                self.config.vision_shared_memory_name,
                access=mmap.ACCESS_READ,
            )
            vis_magic = struct.unpack_from('<I', self._vision_shm, VISION_OFF_MAGIC)[0]
            if vis_magic != VISION_MAGIC:
                log.debug("CPU vision SHM magic mismatch: 0x%08X (D3D12 path active)", vis_magic)
                self._vision_shm.close()
                self._vision_shm = None
            else:
                self.has_vision = True
                log.info("Vision SHM connected (%d bytes)", VISION_TOTAL_SIZE)
        except Exception as e:
            if not self._warned_no_vision:
                log.info("Vision SHM not available: %s (vision tokenization disabled)", e)
                self._warned_no_vision = True
            self._vision_shm = None
            self.has_vision = False

        return True

    @property
    def is_open(self) -> bool:
        return self._cpu_shm is not None

    def read_latest_cpu_frame(self) -> Optional[Tuple[int, bytes]]:
        """
        Non-blocking read of the latest CPU frame.

        Returns (frame_number, slot_bytes) or None if no new frame.
        slot_bytes is the full parity slot (frame header + all modality slices).
        """
        if self._cpu_shm is None:
            return None

        frame = struct.unpack_from('<q', self._cpu_shm, SHM_OFF_LATEST_FRAME)[0]
        if frame <= 0 or frame <= self.last_cpu_frame:
            return None

        slot = frame % self.num_slots
        base = self.cpu_data_offset + slot * self.slot_size
        slot_bytes = bytes(self._cpu_shm[base:base + self.slot_size])

        self.last_cpu_frame = frame
        return (frame, slot_bytes)

    def read_vision_pixels(self) -> Optional[Tuple[int, bytes, bytes, bytes, bytes]]:
        """
        Non-blocking read of vision pixel data.

        Returns (frame_number, left_pano, right_pano, left_fovea, right_fovea) or None.
        Each is raw BGRA8 bytes. Checks bWriting flag to avoid tearing.
        """
        if self._vision_shm is None:
            return None

        # Check if writer is mid-update
        writing = struct.unpack_from('<i', self._vision_shm, VISION_OFF_WRITING)[0]
        if writing:
            return None

        frame = struct.unpack_from('<q', self._vision_shm, VISION_OFF_LATEST_FRAME)[0]
        if frame <= 0 or frame <= self.last_vision_frame:
            return None

        # Read dimensions from header (in case they differ from defaults)
        pano_w = struct.unpack_from('<i', self._vision_shm, VISION_OFF_PANO_WIDTH)[0]
        pano_h = struct.unpack_from('<i', self._vision_shm, VISION_OFF_PANO_HEIGHT)[0]
        fovea_w = struct.unpack_from('<i', self._vision_shm, VISION_OFF_FOVEA_WIDTH)[0]
        fovea_h = struct.unpack_from('<i', self._vision_shm, VISION_OFF_FOVEA_HEIGHT)[0]
        bpp = struct.unpack_from('<i', self._vision_shm, VISION_OFF_BPP)[0]
        pixel_off = struct.unpack_from('<i', self._vision_shm, VISION_OFF_PIXEL_OFFSET)[0]

        pano_bytes = pano_w * pano_h * bpp
        fovea_bytes = fovea_w * fovea_h * bpp

        off_lp = pixel_off
        off_rp = off_lp + pano_bytes
        off_lf = off_rp + pano_bytes
        off_rf = off_lf + fovea_bytes

        left_pano = bytes(self._vision_shm[off_lp:off_lp + pano_bytes])
        right_pano = bytes(self._vision_shm[off_rp:off_rp + pano_bytes])
        left_fovea = bytes(self._vision_shm[off_lf:off_lf + fovea_bytes])
        right_fovea = bytes(self._vision_shm[off_rf:off_rf + fovea_bytes])

        self.last_vision_frame = frame
        return (frame, left_pano, right_pano, left_fovea, right_fovea)

    def close(self):
        """Release shared memory handles."""
        if self._cpu_shm is not None:
            self._cpu_shm.close()
            self._cpu_shm = None
        if self._vision_shm is not None:
            self._vision_shm.close()
            self._vision_shm = None
        self.has_vision = False
        log.info("CPU reader closed")
