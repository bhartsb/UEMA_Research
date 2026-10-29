# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Mode B transport tests: what we write into shared memory, the MODEL'S OWN readers read back.

  * PipelineSHMWriter -> tokenizer/transport/cpu_reader.py CPUReader.read_latest_cpu_frame
  * VisionSHMWriter   -> CPUReader.read_vision_pixels
  * lem/action/writer.py ActionBufferWriter (the model's) -> ActionSHMReader (ours)
Windows named shared memory only (skipped elsewhere).
"""
import sys

import numpy as np
import pytest
import torch

if not sys.platform.startswith("win"):
    pytest.skip("Windows named shared memory only", allow_module_level=True)

from tokenizer import config as C
from tokenizer.config import TokenizerConfig
from tokenizer.transport.cpu_reader import CPUReader
from mimo_bridge import frame_encoder as FE
from mimo_bridge.shm_writer import PipelineSHMWriter, VisionSHMWriter, ActionSHMReader
from mimo_bridge.tests.test_frame_roundtrip import encode


def test_pipeline_shm_roundtrip_through_model_reader():
    w = PipelineSHMWriter(name="LEM_Pipeline_test_bridge")
    cfg = TokenizerConfig(device="cpu", cpu_shared_memory_name="LEM_Pipeline_test_bridge",
                          vision_shared_memory_name="LEM_Vision_test_bridge_none")
    r = CPUReader(cfg)
    assert r.open()
    assert (r.num_slots, r.slot_size, r.cpu_data_offset) == (2, C.FRAME_BUFFER_SIZE_ALIGNED, 256)
    assert r.read_latest_cpu_frame() is None          # nothing published yet
    for fn in (1, 2, 3):
        b, _ = encode(seed=fn)
        # stamp the frame number into the header like the env would
        buf = bytearray(b)
        import struct
        struct.pack_into("<q", buf, C.HDR_FRAME_NUMBER, fn)
        w.write(fn, bytes(buf))
        got = r.read_latest_cpu_frame()
        assert got is not None
        n, slot = got
        assert n == fn and slot == bytes(buf)             # bit for bit through the model's reader
        assert r.read_latest_cpu_frame() is None          # same frame not delivered twice
        # and the model's tokenizer decodes it
        tf = FE.decode_with_model_tokenizer(slot)
        assert tf.metadata.frame_number == fn and tf.proprio.shape == (24, 12)
    r.close(); w.close()


def test_vision_shm_letterbox_through_model_reader():
    vw = VisionSHMWriter(name="LEM_Vision_test_bridge")
    cfg = TokenizerConfig(device="cpu", cpu_shared_memory_name="LEM_Pipeline_test_bridge2",
                          vision_shared_memory_name="LEM_Vision_test_bridge")
    pw = PipelineSHMWriter(name="LEM_Pipeline_test_bridge2")   # reader needs the cpu region to open
    r = CPUReader(cfg)
    assert r.open() and r.has_vision
    left = np.zeros((256, 256, 3), np.uint8); left[..., 0] = 200          # red
    right = np.zeros((256, 256, 3), np.uint8); right[..., 2] = 150         # blue
    vw.write(7, left, right)
    res = r.read_vision_pixels()
    assert res is not None
    fn, lp, rp, lf, rf = res
    assert fn == 7 and len(lp) == C.VISION_PANO_BYTES and len(lf) == C.VISION_FOVEA_BYTES
    lp = np.frombuffer(lp, np.uint8).reshape(C.PANO_HEIGHT, C.PANO_WIDTH, 4)
    cy, cx = C.PANO_HEIGHT // 2, C.PANO_WIDTH // 2
    assert tuple(lp[cy, cx]) == (0, 0, 200, 255)        # BGRA: red eye lands in R
    assert tuple(lp[0, 0]) == (0, 0, 0, 255)            # border
    rf = np.frombuffer(rf, np.uint8).reshape(C.FOVEA_HEIGHT, C.FOVEA_WIDTH, 4)
    assert tuple(rf[C.FOVEA_HEIGHT // 2, C.FOVEA_WIDTH // 2]) == (150, 0, 0, 255)
    r.close(); vw.close(); pw.close()


def test_action_buffer_written_by_model_read_by_bridge():
    from lem.action.writer import ActionBufferWriter
    cfg = {"action_buffer": {"enabled": True, "shm_name": "LEM_Action_Buffer_test_bridge", "buffer_size": 1176, "write_timeout_ms": 5}}
    aw = ActionBufferWriter(cfg)
    assert aw.enabled
    rd = ActionSHMReader(name="LEM_Action_Buffer_test_bridge", retries=5)
    assert rd.is_open and rd.read() is None
    motor = torch.arange(24 * 9, dtype=torch.float32).reshape(24, 9) / 10.0
    gaze = torch.tensor([0.25, -0.5])
    aw.write(motor_commands=motor, gaze_target=gaze)
    got = rd.read()
    assert got is not None
    fid, flags, m, g = got
    assert fid == 1 and flags == 3
    # the C buffer holds 23 limb handles; handle 23 (gaze) travels in the gaze field
    assert np.array_equal(m, motor.numpy()[:23]) and np.array_equal(g, gaze.numpy())
    assert rd.read() is None
    aw.close(); rd.close()
