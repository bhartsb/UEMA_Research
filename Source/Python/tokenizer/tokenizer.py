# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
LEMTokenizer — main tokenizer class.

Reads raw multi-modal pipeline output from shared memory,
tokenizes each modality into normalized GPU-resident tensors,
and emits TokenizedFrame objects for downstream consumption.
"""

import logging
import struct
import time
from typing import Optional

import numpy as np
import torch

from .config import (
    TokenizerConfig, FrameMetadata, TokenizedFrame,
    HEADER_OFFSET,
    HDR_FRAME_NUMBER, HDR_TIMESTAMP, HDR_MODALITY_MASK,
    HDR_DELTA_TIME, HDR_WALL_CLOCK,
    HDR_GAZE_AZIMUTH, HDR_GAZE_ELEVATION,
    MOD_AUDIO, MOD_PROPRIOCEPTION, MOD_MOTOR, MOD_TOUCH, MOD_VOICE,
)
from .transport.cpu_reader import CPUReader
from .transport.gpu_reader import GPUReader
from .modalities.audio import tokenize_audio
from .modalities.proprioception import tokenize_proprioception
from .modalities.motor import tokenize_motor, NUM_HANDLES as MOTOR_NUM_HANDLES
from .modalities.touch import tokenize_touch
from .modalities.voice import tokenize_voice
from .modalities.vision import VisionTokenizer
from .manifest import (
    load_pipeline_manifest, apply_manifest_overrides,
    write_tokenizer_manifest,
)

log = logging.getLogger(__name__)


class LEMTokenizer:
    """
    Multi-modal tokenizer that reads from the UE5 pipeline shared memory
    and produces normalized, GPU-resident PyTorch tensors.

    Usage:
        tokenizer = LEMTokenizer()
        while True:
            frame = tokenizer.read_frame()
            if frame is not None:
                # Use frame.vision_patches, frame.audio_spectral, etc.
                pass
        tokenizer.close()
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()

        # 1. Load pipeline manifest and apply overrides
        pipeline_manifest = load_pipeline_manifest(self.config.pipeline_manifest_path)
        if pipeline_manifest is not None:
            self.config = apply_manifest_overrides(self.config, pipeline_manifest)

        # 2. Initialize CPU reader (opens shared memory)
        self.cpu_reader = CPUReader(self.config)
        if not self.cpu_reader.open():
            log.warning("CPU shared memory not available — tokenizer will return None until pipeline starts")

        # Update manifest path from SHM header if available
        if self.cpu_reader.manifest_path and pipeline_manifest is None:
            manifest = load_pipeline_manifest(self.cpu_reader.manifest_path)
            if manifest is not None:
                self.config = apply_manifest_overrides(self.config, manifest)

        # 3. Initialize vision tokenizer (precomputes LUTs)
        self.vision_tokenizer = VisionTokenizer(self.config)

        # 4. Initialize GPU reader
        self.gpu_reader = GPUReader(self.config)

        # 5. (FFT now computed in UE C++ — no hann_window needed)

        # 6. Write tokenizer manifest
        self._manifest = write_tokenizer_manifest(self.config)

        # 7. Pre-allocated stubs (avoid per-frame CUDA alloc through the
        #    caching allocator — its per-device mutex contends with training
        #    thread allocations, causing 30ms+ stalls on inference thread)
        device = self.config.device
        self._intermind_zero = torch.zeros(3, dtype=torch.float32)  # CPU
        # Audio zeros on CPU (FFT now computed in UE C++, read from SHM)
        self._zero_audio_spectral = torch.zeros(
            self.config.fft_n_bins, 2, dtype=torch.float32)
        self._zero_audio_meta = torch.zeros(5, dtype=torch.float32)
        # Non-vision/non-audio zeros on CPU (avoid unnecessary capture→training GPU hop)
        self._zero_proprio = torch.zeros(0, 12, dtype=torch.float32)
        # One zero row per handle tokenize_motor returns (24 since the eyes became
        # handle 23); a fixed 23 here left frames without the motor bit one row short.
        self._zero_motor_handles = torch.zeros(MOTOR_NUM_HANDLES, 10, dtype=torch.float32)
        self._zero_motor_energy = torch.zeros(3, dtype=torch.float32)
        self._zero_motor_capability = torch.zeros(21, dtype=torch.float32)
        self._zero_touch_impulses = torch.zeros(16, 8, dtype=torch.float32)
        self._zero_touch_collisions = torch.zeros(16, 8, dtype=torch.float32)
        self._zero_voice = torch.zeros(56, dtype=torch.float32)

        # 8. State tracking
        self.last_frame: int = -1
        self.frames_processed: int = 0
        self.frames_skipped: int = 0

        log.info("LEMTokenizer initialized (device=%s, vision_gpu=%s, vision_shm=%s)",
                 self.config.device, self.gpu_reader.available, self.cpu_reader.has_vision)

    def _parse_frame_header(self, slot_bytes: bytes) -> FrameMetadata:
        """Extract frame metadata from the slot's header."""
        h = HEADER_OFFSET
        return FrameMetadata(
            frame_number=struct.unpack_from('<q', slot_bytes, h + HDR_FRAME_NUMBER)[0],
            timestamp=struct.unpack_from('<d', slot_bytes, h + HDR_TIMESTAMP)[0],
            modality_valid_mask=struct.unpack_from('<I', slot_bytes, h + HDR_MODALITY_MASK)[0],
            dt=struct.unpack_from('<f', slot_bytes, h + HDR_DELTA_TIME)[0],
            wall_clock=struct.unpack_from('<d', slot_bytes, h + HDR_WALL_CLOCK)[0],
            gaze_azimuth_deg=struct.unpack_from('<f', slot_bytes, h + HDR_GAZE_AZIMUTH)[0],
            gaze_elevation_deg=struct.unpack_from('<f', slot_bytes, h + HDR_GAZE_ELEVATION)[0],
        )

    def read_frame(self, skip_vision: bool = False) -> Optional[TokenizedFrame]:
        """
        Non-blocking read. Returns None if no new frame available.

        Polls SHM directly and captures GPU vision textures inline.
        Single-threaded — no GIL contention.

        Args:
            skip_vision: If True, skip D3D12 fence wait + texture copy +
                patchification. Use when the training loop will reuse a
                cached vision token anyway (vision_every_n_frames > 1).
        """
        result = self.cpu_reader.read_latest_cpu_frame()
        if result is None:
            return None

        frame_number, slot_bytes = result

        # Track skips
        if self.last_frame > 0:
            skipped = frame_number - self.last_frame - 1
            if skipped > 0:
                self.frames_skipped += skipped
                if self.frames_skipped % 100 == 0:
                    log.debug("Total frames skipped: %d", self.frames_skipped)

        self.last_frame = frame_number

        # Parse frame header
        metadata = self._parse_frame_header(slot_bytes)
        mask = metadata.modality_valid_mask
        device = self.config.device

        # Tokenize CPU modalities (use pre-allocated zeros for missing ones
        # to avoid CUDA caching allocator contention with training thread)
        if mask & MOD_AUDIO:
            audio_spectral, audio_meta = tokenize_audio(
                slot_bytes, self.config)
        else:
            audio_spectral = self._zero_audio_spectral
            audio_meta = self._zero_audio_meta

        if mask & MOD_PROPRIOCEPTION:
            proprio = tokenize_proprioception(slot_bytes, self.config)
        else:
            proprio = self._zero_proprio

        if mask & MOD_MOTOR:
            motor_handles, motor_energy, motor_capability = tokenize_motor(slot_bytes, self.config)
        else:
            motor_handles = self._zero_motor_handles
            motor_energy = self._zero_motor_energy
            motor_capability = self._zero_motor_capability

        if mask & MOD_TOUCH:
            touch_impulses, touch_collisions, imp_count, col_count = tokenize_touch(
                slot_bytes, self.config)
        else:
            touch_impulses = self._zero_touch_impulses
            touch_collisions = self._zero_touch_collisions
            imp_count, col_count = 0, 0

        if mask & MOD_VOICE:
            voice = tokenize_voice(slot_bytes, self.config)
        else:
            voice = self._zero_voice

        # InterMind stub (pre-allocated zero vector)
        intermind = self._intermind_zero

        # Vision: inline GPU capture + tokenize (N-1 frame, stream-ordered)
        vision_patches = None
        vision_angular_pos = None
        vision_eye_id = None
        vision_source_id = None

        if not skip_vision and self.gpu_reader.available and frame_number > 1:
            prev = frame_number - 1
            slot = prev % 2
            textures = self.gpu_reader.read_vision_textures(slot, prev)
            if textures is not None:
                pano_l, pano_r, fovea_l, fovea_r = textures
                vision_patches, vision_angular_pos, vision_eye_id, vision_source_id = \
                    self.vision_tokenizer.tokenize(
                        pano_l, pano_r, fovea_l, fovea_r,
                        metadata.gaze_azimuth_deg,
                        metadata.gaze_elevation_deg,
                    )
        elif not skip_vision and not self.gpu_reader.available and self.cpu_reader.has_vision:
            # CPU path (no D3D12 interop): BGRA8 pixels from the LEM_Vision shared memory
            # region written by a CPU producer (e.g. the MIMo bridge). Same tokenizer math
            # as the GPU path; read_vision_pixels returns None until a new frame is there.
            pixels = self.cpu_reader.read_vision_pixels()
            if pixels is not None:
                _vf, pano_l, pano_r, fovea_l, fovea_r = pixels
                vision_patches, vision_angular_pos, vision_eye_id, vision_source_id = \
                    self.vision_tokenizer.tokenize(
                        pano_l, pano_r, fovea_l, fovea_r,
                        metadata.gaze_azimuth_deg,
                        metadata.gaze_elevation_deg,
                    )

        self.frames_processed += 1

        return TokenizedFrame(
            metadata=metadata,
            vision_patches=vision_patches,
            vision_angular_pos=vision_angular_pos,
            vision_eye_id=vision_eye_id,
            vision_source_id=vision_source_id,
            audio_spectral=audio_spectral,
            audio_metadata=audio_meta,
            proprio=proprio,
            motor_handles=motor_handles,
            motor_energy=motor_energy,
            motor_capability=motor_capability,
            touch_impulses=touch_impulses,
            touch_collisions=touch_collisions,
            touch_valid_counts=(imp_count, col_count),
            voice=voice,
            intermind=intermind,
        )

    def read_frame_blocking(self, timeout_ms: int = 100) -> Optional[TokenizedFrame]:
        """
        Blocking read with timeout. Polls until a new frame arrives or timeout.

        Args:
            timeout_ms: Maximum wait time in milliseconds.

        Returns:
            TokenizedFrame or None if timed out.
        """
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while time.perf_counter() < deadline:
            frame = self.read_frame()
            if frame is not None:
                return frame
            time.sleep(0.001)  # 1ms yield — allows Ctrl+C on Windows
        return None

    def get_manifest(self) -> dict:
        """Returns the tokenizer manifest describing all output shapes and semantics."""
        return self._manifest

    def close(self):
        """Release shared memory handles and GPU resources."""
        self.cpu_reader.close()
        self.gpu_reader.close()
        log.info("LEMTokenizer closed. Processed %d frames, skipped %d",
                 self.frames_processed, self.frames_skipped)
