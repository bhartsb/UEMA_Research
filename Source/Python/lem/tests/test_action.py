# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Tests for ActionBufferWriter — round-trip pack/unpack via SHM.

Uses mmap.ACCESS_READ as the "consumer" to verify written bytes.
"""

from __future__ import annotations

import mmap
import struct
import sys

import pytest
import torch

from lem.action.shm import ActionSHM, ACTION_MAGIC, ACTION_VERSION, ACTION_SHM_HEADER_SIZE
from lem.action.writer import (
    ActionBufferWriter,
    OFF_FRAME_ID, OFF_TIMESTAMP_US, OFF_FLAGS,
    OFF_MOTOR, OFF_GAZE, OFF_VOICE,
    ACTION_FRAME_SIZE,
    FLAG_MOTOR, FLAG_GAZE, FLAG_VOICE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(enabled: bool = True, shm_name: str = "LEM_Action_Test") -> dict:
    """Build a minimal config dict for testing."""
    return {
        "action_buffer": {
            "enabled": enabled,
            "shm_name": shm_name,
            "buffer_size": ACTION_FRAME_SIZE,
            "write_timeout_ms": 5,
        },
        "body": {"n_handles": 23},
        "model": {"motor_controller": {"output_per_handle": 9}},
    }


def _open_reader(shm_name: str) -> mmap.mmap:
    """Open a read-only mmap to the same named SHM (consumer side)."""
    total = ACTION_SHM_HEADER_SIZE + ACTION_FRAME_SIZE
    return mmap.mmap(-1, total, shm_name, access=mmap.ACCESS_READ)


def _read_header(reader: mmap.mmap):
    """Read SHM-level header (magic, version)."""
    magic, version = struct.unpack_from('<II', reader, 0)
    return magic, version


def _read_frame(reader: mmap.mmap):
    """Read action frame fields from consumer mmap."""
    base = ACTION_SHM_HEADER_SIZE
    frame_id, timestamp_us, flags = struct.unpack_from('<QQI', reader, base + OFF_FRAME_ID)

    motor = struct.unpack_from('<207f', reader, base + OFF_MOTOR)
    gaze = struct.unpack_from('<2f', reader, base + OFF_GAZE)
    voice = struct.unpack_from('<80f', reader, base + OFF_VOICE)

    return {
        "frame_id": frame_id,
        "timestamp_us": timestamp_us,
        "flags": flags,
        "motor": motor,
        "gaze": gaze,
        "voice": voice,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActionSHM:
    """Tests for the SHM layer."""

    def test_open_close(self):
        shm = ActionSHM("LEM_Action_SHM_Test", ACTION_FRAME_SIZE)
        assert shm.open()
        assert shm.buf is not None
        assert shm.data_offset == ACTION_SHM_HEADER_SIZE

        # Verify magic/version via consumer
        reader = _open_reader("LEM_Action_SHM_Test")
        magic, version = _read_header(reader)
        assert magic == ACTION_MAGIC
        assert version == ACTION_VERSION

        reader.close()
        shm.close()
        assert shm.buf is None

    def test_zero_initialized(self):
        shm = ActionSHM("LEM_Action_SHM_Zero", ACTION_FRAME_SIZE)
        assert shm.open()

        reader = _open_reader("LEM_Action_SHM_Zero")
        # Data region should be all zeros (after the 8-byte magic+version)
        data_start = 8  # magic(4) + version(4)
        raw = reader[data_start:ACTION_SHM_HEADER_SIZE + ACTION_FRAME_SIZE]
        assert raw == b'\x00' * len(raw)

        reader.close()
        shm.close()


class TestActionBufferWriter:
    """Tests for the writer."""

    def test_pack_unpack_roundtrip(self):
        """Write known tensors, verify every field at expected offset."""
        config = _make_config(shm_name="LEM_Action_RT")
        writer = ActionBufferWriter(config)
        assert writer.enabled

        # Create known tensors
        motor = torch.arange(207, dtype=torch.float32).reshape(23, 9)
        gaze = torch.tensor([1.5, -0.5], dtype=torch.float32)

        writer.write(motor, gaze_target=gaze)

        reader = _open_reader("LEM_Action_RT")
        frame = _read_frame(reader)

        assert frame["frame_id"] == 1
        assert frame["timestamp_us"] > 0
        assert frame["flags"] == FLAG_MOTOR | FLAG_GAZE

        # Verify motor values
        for i in range(207):
            assert abs(frame["motor"][i] - float(i)) < 1e-6, f"motor[{i}] mismatch"

        # Verify gaze
        assert abs(frame["gaze"][0] - 1.5) < 1e-6
        assert abs(frame["gaze"][1] - (-0.5)) < 1e-6

        # Voice should be zeros
        for i in range(80):
            assert frame["voice"][i] == 0.0

        reader.close()
        writer.close()

    def test_flags_motor_only(self):
        """Motor only — flag should be 0x1."""
        config = _make_config(shm_name="LEM_Action_FlagM")
        writer = ActionBufferWriter(config)

        motor = torch.zeros(23, 9)
        writer.write(motor)

        reader = _open_reader("LEM_Action_FlagM")
        frame = _read_frame(reader)
        assert frame["flags"] == FLAG_MOTOR

        reader.close()
        writer.close()

    def test_flags_motor_gaze(self):
        """Motor + gaze — flag should be 0x3."""
        config = _make_config(shm_name="LEM_Action_FlagMG")
        writer = ActionBufferWriter(config)

        motor = torch.zeros(23, 9)
        gaze = torch.zeros(2)
        writer.write(motor, gaze_target=gaze)

        reader = _open_reader("LEM_Action_FlagMG")
        frame = _read_frame(reader)
        assert frame["flags"] == FLAG_MOTOR | FLAG_GAZE

        reader.close()
        writer.close()

    def test_flags_all(self):
        """Motor + gaze + voice — flag should be 0x7."""
        config = _make_config(shm_name="LEM_Action_FlagAll")
        writer = ActionBufferWriter(config)

        motor = torch.zeros(23, 9)
        gaze = torch.zeros(2)
        voice = torch.zeros(80)
        writer.write(motor, gaze_target=gaze, voice_params=voice)

        reader = _open_reader("LEM_Action_FlagAll")
        frame = _read_frame(reader)
        assert frame["flags"] == FLAG_MOTOR | FLAG_GAZE | FLAG_VOICE

        reader.close()
        writer.close()

    def test_frame_id_monotonic(self):
        """Frame ID should increment with each write."""
        config = _make_config(shm_name="LEM_Action_Mono")
        writer = ActionBufferWriter(config)

        motor = torch.zeros(23, 9)
        reader = _open_reader("LEM_Action_Mono")

        for i in range(1, 11):
            writer.write(motor)
            frame = _read_frame(reader)
            assert frame["frame_id"] == i

        reader.close()
        writer.close()

    def test_known_values_all_ones(self):
        """Write all-ones motor, verify 207 floats are all 1.0."""
        config = _make_config(shm_name="LEM_Action_Ones")
        writer = ActionBufferWriter(config)

        motor = torch.ones(23, 9)
        writer.write(motor)

        reader = _open_reader("LEM_Action_Ones")
        frame = _read_frame(reader)

        for i in range(207):
            assert abs(frame["motor"][i] - 1.0) < 1e-6, f"motor[{i}] not 1.0"

        reader.close()
        writer.close()

    def test_batch_dim_squeeze(self):
        """Writer should handle [1, 23, 9] shaped tensors (batch dim)."""
        config = _make_config(shm_name="LEM_Action_Batch")
        writer = ActionBufferWriter(config)

        motor = torch.arange(207, dtype=torch.float32).reshape(1, 23, 9)
        gaze = torch.tensor([[0.1, 0.2]], dtype=torch.float32)  # [1, 2]
        writer.write(motor, gaze_target=gaze)

        reader = _open_reader("LEM_Action_Batch")
        frame = _read_frame(reader)

        for i in range(207):
            assert abs(frame["motor"][i] - float(i)) < 1e-6
        assert abs(frame["gaze"][0] - 0.1) < 1e-6
        assert abs(frame["gaze"][1] - 0.2) < 1e-6

        reader.close()
        writer.close()

    def test_disabled_is_noop(self):
        """Disabled writer should not crash on write()."""
        config = _make_config(enabled=False, shm_name="LEM_Action_Noop")
        writer = ActionBufferWriter(config)
        assert not writer.enabled

        motor = torch.zeros(23, 9)
        writer.write(motor)  # Should not raise
        assert writer.frame_counter == 0

        writer.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_gpu_tensor_handling(self):
        """CUDA tensors should be moved to CPU and packed correctly."""
        config = _make_config(shm_name="LEM_Action_GPU")
        writer = ActionBufferWriter(config)

        motor = torch.arange(207, dtype=torch.float32, device="cuda:0").reshape(23, 9)
        gaze = torch.tensor([0.3, -0.7], dtype=torch.float32, device="cuda:0")
        writer.write(motor, gaze_target=gaze)

        reader = _open_reader("LEM_Action_GPU")
        frame = _read_frame(reader)

        for i in range(207):
            assert abs(frame["motor"][i] - float(i)) < 1e-6
        assert abs(frame["gaze"][0] - 0.3) < 1e-5
        assert abs(frame["gaze"][1] - (-0.7)) < 1e-5

        reader.close()
        writer.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
    def test_fp16_tensor_handling(self):
        """fp16 tensors should be upcasted to fp32 for packing."""
        config = _make_config(shm_name="LEM_Action_FP16")
        writer = ActionBufferWriter(config)

        motor = torch.ones(23, 9, dtype=torch.float16, device="cuda:0")
        writer.write(motor)

        reader = _open_reader("LEM_Action_FP16")
        frame = _read_frame(reader)

        for i in range(207):
            assert abs(frame["motor"][i] - 1.0) < 1e-3

        reader.close()
        writer.close()

    def test_overwrite_updates(self):
        """Second write should overwrite the first."""
        config = _make_config(shm_name="LEM_Action_OW")
        writer = ActionBufferWriter(config)

        motor_a = torch.zeros(23, 9)
        motor_b = torch.ones(23, 9) * 42.0

        writer.write(motor_a)
        writer.write(motor_b)

        reader = _open_reader("LEM_Action_OW")
        frame = _read_frame(reader)

        assert frame["frame_id"] == 2
        for i in range(207):
            assert abs(frame["motor"][i] - 42.0) < 1e-6

        reader.close()
        writer.close()
