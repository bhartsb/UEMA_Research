# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Encode a frame with mimo_bridge.frame_encoder, decode it with the MODEL'S OWN tokenizer.

Two levels:
  1. bit for bit: the raw float32 values we wrote come back through struct.unpack at the
     model's offsets unchanged (exact equality, no tolerance).
  2. through the tokenizer: tokenize_proprioception / tokenize_motor / tokenize_touch give
     back exactly the normalized values the model would see (hand-computed divisors).
"""
import struct

import numpy as np
import pytest
import torch

from tokenizer import config as C
from tokenizer.config import TokenizerConfig
from tokenizer.modalities.proprioception import tokenize_proprioception
from tokenizer.modalities.motor import tokenize_motor
from tokenizer.modalities.touch import tokenize_touch, _FIELD_NORMS
from mimo_bridge import frame_encoder as FE
from mimo_bridge.touch_adapter import bone_hash


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    bones = rng.uniform(-50, 50, size=(24, 11)).astype(np.float32)
    q = rng.normal(size=(24, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    bones[:, 0:4] = q
    bones[:, 10] = rng.uniform(0, 1, 24)
    hashes = [bone_hash(f"bone{i}") for i in range(24)]
    handles = rng.uniform(-100, 100, size=(24, 10)).astype(np.float32)
    handles[:, 7] = rng.uniform(0, 1, 24); handles[:, 8] = rng.uniform(0, 1, 24)
    handles[:, 9] = (rng.uniform(0, 1, 24) > 0.3).astype(np.float32)
    energy = [1.5, 2.5, 4.0]
    body_sets = np.column_stack([rng.uniform(0, 1, 10), rng.uniform(0, 100, 10)]).astype(np.float32)
    patches = rng.uniform(0, 1e4, size=(53, 10)).astype(np.float32)
    patches[:, 8:10] = 1.0
    cols = [dict(bone_hash=bone_hash("left_hand"), point_cm=np.array([1.0, 2.0, 3.0], np.float32),
                 normal=np.array([0.0, 0.0, 1.0], np.float32), penetration_cm=0.25, impact_speed_cms=12.5,
                 impulse_ns=np.array([0.0, 0.0, 0.02], np.float32), duration_s=0.05, frame_count=5)
            for _ in range(20)]
    return bones, hashes, handles, energy, body_sets, patches, cols


def encode(seed=0, vision=True):
    bones, hashes, handles, energy, body_sets, patches, cols = make_inputs(seed)
    buf = FE.encode_frame(frame_number=1234, timestamp=12.34, dt=0.01, wall_clock=99.5,
                          gaze_az_deg=10.0, gaze_el_deg=-5.0, bones=bones, bone_hashes=hashes,
                          handles=handles, energy=energy, body_sets=body_sets, patches=patches,
                          collisions=cols, vision_valid=vision)
    return bytes(buf), (bones, hashes, handles, energy, body_sets, patches, cols)


def test_sizes_come_from_model_constants():
    assert len(FE.new_frame_buffer()) == C.FRAME_BUFFER_SIZE_ALIGNED == 31232
    assert FE.FRAME_SIZE == 31040


def test_header_bit_for_bit():
    b, _ = encode()
    assert struct.unpack_from('<q', b, C.HDR_FRAME_NUMBER)[0] == 1234
    assert struct.unpack_from('<d', b, C.HDR_TIMESTAMP)[0] == 12.34
    assert struct.unpack_from('<I', b, C.HDR_MODALITY_MASK)[0] == (C.MOD_VISION | C.MOD_PROPRIOCEPTION | C.MOD_MOTOR | C.MOD_TOUCH)
    assert struct.unpack_from('<f', b, C.HDR_DELTA_TIME)[0] == np.float32(0.01)
    assert struct.unpack_from('<d', b, C.HDR_WALL_CLOCK)[0] == 99.5
    assert struct.unpack_from('<f', b, C.HDR_GAZE_AZIMUTH)[0] == 10.0
    assert struct.unpack_from('<f', b, C.HDR_GAZE_ELEVATION)[0] == -5.0
    # audio and voice bits off, and those slices are all zero
    assert b[C.AUDIO_OFFSET:C.AUDIO_OFFSET + C.AUDIO_SIZE] == bytes(C.AUDIO_SIZE)
    assert b[C.VOICE_OFFSET:C.VOICE_OFFSET + C.VOICE_SIZE] == bytes(C.VOICE_SIZE)


def test_proprio_bit_for_bit_and_tokenized():
    b, (bones, hashes, *_rest) = encode()
    assert struct.unpack_from('<i', b, C.PROPRIO_OFFSET + C.PRO_NUM_BONES)[0] == 24
    for i in range(24):
        bb = C.PROPRIO_OFFSET + C.PRO_BONE_DATA + i * C.PROPRIO_BONE_STRIDE
        assert struct.unpack_from('<I', b, bb + C.BONE_NAME_HASH)[0] == hashes[i]
        assert struct.unpack_from('<4f', b, bb + C.BONE_ROTATION) == tuple(bones[i, 0:4].tolist())
        assert struct.unpack_from('<3f', b, bb + C.BONE_TRANSLATION) == tuple(bones[i, 4:7].tolist())
        assert struct.unpack_from('<3f', b, bb + C.BONE_VELOCITY) == tuple(bones[i, 7:10].tolist())
        assert struct.unpack_from('<f', b, bb + C.BONE_FATIGUE)[0] == bones[i, 10]
    # through the model's tokenizer: root-relative / 200, vel / 500, fatigue, index
    cfg = TokenizerConfig(device="cpu")
    t = tokenize_proprioception(b, cfg)
    assert t.shape == (24, 12)
    root = bones[0, 4:7]
    for i in range(24):
        assert torch.allclose(t[i, 0:4], torch.from_numpy(bones[i, 0:4]))
        exp_pos = (bones[i, 4:7] - root) / 200.0
        assert torch.allclose(t[i, 4:7], torch.from_numpy(exp_pos.astype(np.float32)), atol=1e-6)
        assert torch.allclose(t[i, 7:10], torch.from_numpy((bones[i, 7:10] / 500.0).astype(np.float32)), atol=1e-6)
        assert t[i, 10] == bones[i, 10]
        assert t[i, 11] == float(i) / 25.0     # TokenizerConfig.proprio_index_range
    assert t.reshape(-1).shape[0] == 288   # what the proprio encoder wants (24 bones x 12)


def test_motor_bit_for_bit_and_tokenized():
    b, (_b, _h, handles, energy, body_sets, *_r) = encode()
    for i in range(24):
        hb = C.MOTOR_OFFSET + C.MOT_HANDLES + i * C.MOTOR_HANDLE_STRIDE
        assert struct.unpack_from('<3f', b, hb + C.HND_TARGET_POS) == tuple(handles[i, 0:3].tolist())
        assert struct.unpack_from('<4f', b, hb + C.HND_TARGET_ROT) == tuple(handles[i, 3:7].tolist())
        assert struct.unpack_from('<f', b, hb + C.HND_RADIUS)[0] == handles[i, 7]
        assert struct.unpack_from('<f', b, hb + C.HND_STRENGTH)[0] == handles[i, 8]
        assert struct.unpack_from('<I', b, hb + C.HND_IS_ACTIVE)[0] == int(handles[i, 9])
    cfg = TokenizerConfig(device="cpu")
    h, e, cap = tokenize_motor(b, cfg)
    assert h.shape == (24, 10) and e.shape == (3,) and cap.shape == (21,)
    exp = handles.copy(); exp[:, 0:3] /= 200.0
    assert torch.allclose(h, torch.from_numpy(exp), atol=1e-6)
    assert torch.allclose(e, torch.tensor(energy) / 1000.0)
    assert torch.allclose(cap[:10], torch.from_numpy(body_sets[:, 0]))
    assert torch.allclose(cap[10:20], torch.from_numpy(body_sets[:, 1] / 1000.0), atol=1e-6)
    assert cap[20] == pytest.approx(energy[2] / 1000.0)


def test_touch_bit_for_bit_and_tokenized():
    b, (*_r, patches, cols) = encode()
    tb = C.TOUCH_OFFSET
    assert struct.unpack_from('<i', b, tb + C.TCH_NUM_IMPULSES)[0] == 53
    assert struct.unpack_from('<i', b, tb + C.TCH_NUM_COLLISIONS)[0] == 16     # capped from 20
    raw = np.frombuffer(b, dtype='<f4', count=53 * 10, offset=tb + C.TCH_IMPULSE_DATA).reshape(53, 10)
    assert np.array_equal(raw, patches)                                          # bit for bit
    cb = tb + C.TCH_COLLISION_DATA
    assert struct.unpack_from('<I', b, cb + C.COL_BONE_HASH)[0] == bone_hash("left_hand")
    assert struct.unpack_from('<3f', b, cb + C.COL_CONTACT_POINT) == (1.0, 2.0, 3.0)
    assert struct.unpack_from('<f', b, cb + C.COL_PENETRATION_DEPTH)[0] == 0.25
    assert struct.unpack_from('<f', b, cb + C.COL_IMPACT_SPEED)[0] == 12.5
    assert struct.unpack_from('<i', b, cb + C.COL_FRAME_COUNT)[0] == 5
    cfg = TokenizerConfig(device="cpu")
    imp, col, ni, nc = tokenize_touch(b, cfg)
    assert (ni, nc) == (53, 16) and imp.shape == (256, 10) and col.shape == (16, 8)
    # all 10 written fields arrive (widened 2026-09-14). The two stretch ratios rest at
    # 1.0 and are re-zeroed before the divisors, so a resting patch reads 0 exactly like
    # the zero padding of an absent one.
    want = patches.copy()
    want[:, 8:10] -= 1.0
    assert torch.allclose(imp[:53], torch.from_numpy(want / _FIELD_NORMS), atol=1e-6)
    assert torch.all(imp[:53, 8:10] == 0)
    assert torch.all(imp[53:] == 0)
    # collision row: point/200, normal, pen/10, speed/500
    exp = np.array([1 / 200, 2 / 200, 3 / 200, 0, 0, 1, 0.025, 12.5 / 500], dtype=np.float32)
    assert torch.allclose(col[0], torch.from_numpy(exp), atol=1e-6)


def test_decode_with_model_tokenizer_matches_direct_calls():
    b, _ = encode()
    tf = FE.decode_with_model_tokenizer(b)
    cfg = TokenizerConfig(device="cpu")
    assert tf.metadata.frame_number == 1234 and tf.metadata.dt == np.float32(0.01)
    assert torch.equal(tf.proprio, tokenize_proprioception(b, cfg))
    assert torch.equal(tf.motor_handles, tokenize_motor(b, cfg)[0])
    assert torch.equal(tf.touch_impulses, tokenize_touch(b, cfg)[0])
    assert tf.touch_valid_counts == (53, 16)
    assert tf.voice is None and tf.audio_spectral is None


def test_two_encodes_of_same_inputs_are_identical_bytes():
    a, _ = encode(seed=3)
    b, _ = encode(seed=3)
    assert a == b


def test_rejects_too_many_bones_or_patches():
    with pytest.raises(ValueError):
        FE.write_proprio(FE.new_frame_buffer(), np.zeros((25, 11)))
    with pytest.raises(ValueError):
        FE.write_touch(FE.new_frame_buffer(), np.zeros((257, 10)), [])
