# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
ActionBufferWriter — packs motor/gaze/voice tensors into SHM.

Layout (1176 bytes, contract Section 5):
    Offset  Size   Field
    0       8      frame_id      (uint64, model step counter)
    8       8      timestamp_us  (uint64, microseconds)
    16      4      flags         (uint32, bit 0=motor, 1=gaze, 2=voice)
    20      828    motor_commands (float32[207] = 23 handles x 9 fields)
    848     8      gaze_target   (float32[2] = azimuth, elevation radians)
    856     320    voice_params  (float32[80] = Pink Trombone, zeros for now)
"""

from __future__ import annotations

import logging
import struct
import time
from typing import TYPE_CHECKING

from .shm import ActionSHM

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

# Action frame layout offsets (within the action data region)
OFF_FRAME_ID = 0        # uint64
OFF_TIMESTAMP_US = 8    # uint64
OFF_FLAGS = 16          # uint32
OFF_MOTOR = 20          # float32[207] = 828 bytes (23 handles × 9 fields)
OFF_GAZE = 848          # float32[2]   = 8 bytes
OFF_VOICE = 856         # float32[80]  = 320 bytes
ACTION_FRAME_SIZE = 1176

# Flag bits
FLAG_MOTOR = 0x1
FLAG_GAZE = 0x2
FLAG_VOICE = 0x4

# Struct format for the 20-byte header: uint64 + uint64 + uint32
_HEADER_FMT = '<QQI'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 20


class ActionBufferWriter:
    """Writes action tensors to a shared memory buffer for UE consumption."""

    def __init__(self, config: dict):
        self._enabled = config.get("action_buffer", {}).get("enabled", False)
        self._shm: ActionSHM | None = None
        self._frame_counter: int = 0

        self._write_timeout_ms: float = 5.0
        self._slow_write_count: int = 0

        self._truncated_frames: int = 0

        if self._enabled:
            ab = config["action_buffer"]
            if int(ab["buffer_size"]) != ACTION_FRAME_SIZE:
                # The offsets below are fixed, so a differently sized region means the
                # consumer reads the gaze and voice fields at the wrong addresses.
                # ModelBridgeComponent.h is the authority: static_assert
                # sizeof(FActionFrame) == 1176 with NUM_HANDLES 23.
                raise ValueError(
                    f"action_buffer.buffer_size {ab['buffer_size']} != the frame layout's "
                    f"{ACTION_FRAME_SIZE}; the UE struct (ModelBridgeComponent.h, "
                    f"static_assert(sizeof(FActionFrame) == 1176)) is the authority")
            self._write_timeout_ms = float(ab.get("write_timeout_ms", 5))
            self._shm = ActionSHM(ab["shm_name"], ab["buffer_size"])
            if not self._shm.open():
                log.warning("Action buffer SHM unavailable — writer is no-op")
                self._shm = None
                self._enabled = False
            else:
                log.info("Action buffer writer ready (SHM: %s, %d bytes)",
                         ab["shm_name"], ab["buffer_size"])

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    def write(self, motor_commands, gaze_target=None, voice_params=None) -> None:
        """
        Pack and write one action frame to SHM.

        Args:
            motor_commands: [23, 9] or [1, 23, 9] float tensor — motor output.
            gaze_target: [2] or [1, 2] float tensor — azimuth/elevation (radians), or None.
            voice_params: [80] or [1, 80] float tensor — vocal tract params, or None.
        """
        if not self._enabled or self._shm is None or self._shm.buf is None:
            return

        t0 = time.perf_counter()

        self._frame_counter += 1
        timestamp_us = time.perf_counter_ns() // 1000

        # Build flags
        flags = FLAG_MOTOR
        if gaze_target is not None:
            flags |= FLAG_GAZE
        if voice_params is not None:
            flags |= FLAG_VOICE

        buf = self._shm.buf
        base = self._shm.data_offset

        # Header (20 bytes)
        struct.pack_into(_HEADER_FMT, buf, base + OFF_FRAME_ID,
                         self._frame_counter, timestamp_us, flags)

        # Motor [n_handles, 9] -> n*9 floats. The C buffer layout is fixed at 23 handles
        # (OFF_GAZE sits at 848, immediately after 23x9 floats), so when the model sends
        # the 24-handle body (handle 23 = gaze, 2026-08-24) only the 23 limb handles are
        # packed here - the gaze handle's direction travels in the buffer's own gaze field
        # below, which existed all along. When the UE side grows the buffer to 24, this
        # slice and the comment go. The body grew past 24 first: the trunk_v1 layout has
        # 26 handles, so rows 24 (lower_trunk) and 25 (chest) are dropped here with no
        # field of their own to travel in - the warning names them rather than losing them
        # silently.
        mc = motor_commands.detach()
        if mc.device.type != "cpu":
            mc = mc.cpu()
        mc = mc.float()
        if mc.numel() and mc.reshape(-1).numel() // 9 >= 24:
            sent = mc.reshape(-1).numel() // 9
            if sent > 24:
                self._truncated_frames += 1
                if self._truncated_frames <= 5:
                    log.warning("action buffer packs 23 of %d handles; rows 23..%d do not "
                                "reach UE (row 23 is gaze and travels in the gaze field; "
                                "rows 24+ have no field)", sent, sent - 1)
            mc = mc.reshape(-1, 9)[:23]
        motor_flat = mc.reshape(-1)
        n = len(motor_flat)
        struct.pack_into(f'<{n}f', buf, base + OFF_MOTOR, *motor_flat.tolist())

        # Gaze [2] floats
        if gaze_target is not None:
            gt = gaze_target.detach()
            if gt.device.type != "cpu":
                gt = gt.cpu()
            gaze_flat = gt.float().reshape(-1)
            struct.pack_into('<2f', buf, base + OFF_GAZE, *gaze_flat.tolist())

        # Voice [80] floats
        if voice_params is not None:
            vp = voice_params.detach()
            if vp.device.type != "cpu":
                vp = vp.cpu()
            voice_flat = vp.float().reshape(-1)
            nv = len(voice_flat)
            struct.pack_into(f'<{nv}f', buf, base + OFF_VOICE, *voice_flat.tolist())

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > self._write_timeout_ms:
            self._slow_write_count += 1
            if self._slow_write_count <= 5:
                log.warning("Action write took %.1fms (limit: %.0fms)",
                            elapsed_ms, self._write_timeout_ms)

    def close(self) -> None:
        """Release SHM resources."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        self._enabled = False
