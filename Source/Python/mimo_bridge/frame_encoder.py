# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Frame encoder: pack one LEM sensory frame (31040 bytes) exactly as the UE pipeline does.

Every offset and size is IMPORTED from the model's own reader constants
(Source/Python/tokenizer/config.py). Nothing is hand-copied. The matching decoder is the
model's own tokenizer (tokenize_proprioception / tokenize_motor / tokenize_touch /
tokenize_voice / the header parser), so "encode then decode" in the tests proves the
model reads back exactly what we wrote.

Inputs are plain numpy arrays in the model's units (cm, cm/s, Pa, mm, mm2, N^2):

  header:   frame_number, timestamp (s), dt (s), wall_clock (s), gaze az/el (deg), mask bits
  proprio:  bones (n, 11) = [qx,qy,qz,qw, px,py,pz (cm), vx,vy,vz (cm/s), fatigue] + hashes
  motor:    handles (24, 10) = [px,py,pz (cm), qx,qy,qz,qw, radius, strength, is_active]; row 23 = gaze
            energy (3), body sets (10, 2) = [capacity, accumulated]
  touch:    patches (n, 10), collisions list (see touch_adapter.build_collisions)
  voice:    optional (not used by MIMo; slice left zero and its mask bit off)
  audio:    optional (not used by MIMo; slice left zero and its mask bit off)
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Sequence

import numpy as np

from tokenizer import config as C

FRAME_SIZE = C.FRAME_BUFFER_SIZE           # 31040
SLOT_SIZE = C.FRAME_BUFFER_SIZE_ALIGNED    # 31232

# sanity: the model's constants must describe a frame that fits its slot
assert C.VOICE_OFFSET + C.VOICE_SIZE == FRAME_SIZE
assert C.TOUCH_OFFSET + C.TOUCH_SIZE == C.VOICE_OFFSET
assert C.TCH_COLLISION_DATA + C.TOUCH_MAX_COLLISIONS * C.TOUCH_COLLISION_STRIDE <= C.TOUCH_SIZE
assert C.PRO_BONE_DATA + C.PROPRIO_MAX_BONES * C.PROPRIO_BONE_STRIDE <= C.PROPRIO_SIZE
assert C.MOT_BODY_SETS + C.NUM_BODY_SETS * C.BODY_SET_STATE_STRIDE <= C.MOTOR_SIZE


def new_frame_buffer() -> bytearray:
    """A zeroed slot-sized buffer (the frame is the first FRAME_SIZE bytes)."""
    return bytearray(SLOT_SIZE)


# ------------------------------------------------------------------ header
def write_header(buf: bytearray, frame_number: int, timestamp: float, dt: float,
                 wall_clock: float, gaze_az_deg: float, gaze_el_deg: float, mask: int) -> None:
    h = C.HEADER_OFFSET
    struct.pack_into('<q', buf, h + C.HDR_FRAME_NUMBER, int(frame_number))
    struct.pack_into('<d', buf, h + C.HDR_TIMESTAMP, float(timestamp))
    struct.pack_into('<I', buf, h + C.HDR_MODALITY_MASK, int(mask) & 0xFFFFFFFF)
    struct.pack_into('<f', buf, h + C.HDR_DELTA_TIME, float(dt))
    struct.pack_into('<d', buf, h + C.HDR_WALL_CLOCK, float(wall_clock))
    struct.pack_into('<f', buf, h + C.HDR_GAZE_AZIMUTH, float(gaze_az_deg))
    struct.pack_into('<f', buf, h + C.HDR_GAZE_ELEVATION, float(gaze_el_deg))


# ------------------------------------------------------------------ proprio
def write_proprio(buf: bytearray, bones: np.ndarray, name_hashes: Optional[Sequence[int]] = None,
                  scale: float = 1.0) -> None:
    """bones: (n_bones, 11) float = quat xyzw, pos cm, vel cm/s, fatigue. n_bones <= 24."""
    bones = np.asarray(bones, dtype=np.float32)
    n = int(bones.shape[0])
    if n > C.PROPRIO_MAX_BONES:
        raise ValueError(f"{n} bones > PROPRIO_MAX_BONES {C.PROPRIO_MAX_BONES}")
    if bones.shape[1] != 11:
        raise ValueError("bones must be (n, 11)")
    pb = C.PROPRIO_OFFSET
    struct.pack_into('<i', buf, pb + C.PRO_NUM_BONES, n)
    for i in range(n):
        bb = pb + C.PRO_BONE_DATA + i * C.PROPRIO_BONE_STRIDE
        h = int(name_hashes[i]) if name_hashes is not None else 0
        struct.pack_into('<I', buf, bb + C.BONE_NAME_HASH, h & 0xFFFFFFFF)
        struct.pack_into('<4f', buf, bb + C.BONE_ROTATION, *bones[i, 0:4].tolist())
        struct.pack_into('<3f', buf, bb + C.BONE_TRANSLATION, *bones[i, 4:7].tolist())
        struct.pack_into('<f', buf, bb + C.BONE_SCALE, float(scale))
        struct.pack_into('<3f', buf, bb + C.BONE_VELOCITY, *bones[i, 7:10].tolist())
        struct.pack_into('<f', buf, bb + C.BONE_FATIGUE, float(bones[i, 10]))


# ------------------------------------------------------------------ motor
def write_motor(buf: bytearray, handles: np.ndarray, energy: Sequence[float],
                body_sets: Optional[np.ndarray] = None) -> None:
    """handles: (n<=24, 10) = pos cm(3), quat xyzw(4), radius, strength, is_active(0/1).
    energy: (3,) = ph_work, pcc_holding, total.  body_sets: (10, 2) = capacity, accumulated."""
    handles = np.asarray(handles, dtype=np.float32)
    n = int(handles.shape[0])
    if n > C.MOTOR_MAX_HANDLES or handles.shape[1] != 10:
        raise ValueError("handles must be (n<=24, 10)")
    mb = C.MOTOR_OFFSET
    for i in range(n):
        hb = mb + C.MOT_HANDLES + i * C.MOTOR_HANDLE_STRIDE
        struct.pack_into('<3f', buf, hb + C.HND_TARGET_POS, *handles[i, 0:3].tolist())
        struct.pack_into('<4f', buf, hb + C.HND_TARGET_ROT, *handles[i, 3:7].tolist())
        struct.pack_into('<f', buf, hb + C.HND_RADIUS, float(handles[i, 7]))
        struct.pack_into('<f', buf, hb + C.HND_STRENGTH, float(handles[i, 8]))
        struct.pack_into('<I', buf, hb + C.HND_IS_ACTIVE, 1 if handles[i, 9] > 0.5 else 0)
    eb = mb + C.MOT_ENERGY
    struct.pack_into('<f', buf, eb + C.NRG_PH_WORK, float(energy[0]))
    struct.pack_into('<f', buf, eb + C.NRG_PCC_HOLDING, float(energy[1]))
    struct.pack_into('<f', buf, eb + C.NRG_TOTAL, float(energy[2]))
    if body_sets is not None:
        bs = np.asarray(body_sets, dtype=np.float32)
        for i in range(min(C.NUM_BODY_SETS, bs.shape[0])):
            off = mb + C.MOT_BODY_SETS + i * C.BODY_SET_STATE_STRIDE
            struct.pack_into('<f', buf, off + C.BST_CAPACITY, float(bs[i, 0]))
            struct.pack_into('<f', buf, off + C.BST_ACCUMULATED_ENERGY, float(bs[i, 1]))


# ------------------------------------------------------------------ touch
# Collision entry (64 B) layout, mirrored from the tokenizer's _COLLISION_DT in
# tokenizer/modalities/touch.py and the COL_* offsets in tokenizer/config.py:
#   0  uint32 bone hash | 4 float3 point cm | 16 float3 normal | 28 float penetration cm
#   32 float impact speed cm/s | 36 float3 impulse | 48 float duration s | 52 int32 frame count | 56 pad 8
def write_touch(buf: bytearray, patches: np.ndarray, collisions: Sequence[dict]) -> None:
    patches = np.asarray(patches, dtype=np.float32)
    n = int(patches.shape[0])
    if n > C.TOUCH_MAX_IMPULSES:
        raise ValueError(f"{n} patches > TOUCH_MAX_IMPULSES")
    if n and patches.shape[1] * 4 != C.TOUCH_IMPULSE_STRIDE:
        raise ValueError("patches must be (n, 10)")
    nc = min(len(collisions), C.TOUCH_MAX_COLLISIONS)
    tb = C.TOUCH_OFFSET
    struct.pack_into('<i', buf, tb + C.TCH_NUM_IMPULSES, n)
    struct.pack_into('<i', buf, tb + C.TCH_NUM_COLLISIONS, nc)
    if n:
        start = tb + C.TCH_IMPULSE_DATA
        raw = np.ascontiguousarray(patches, dtype='<f4').tobytes()
        buf[start:start + len(raw)] = raw
    for i in range(nc):
        c = collisions[i]
        cb = tb + C.TCH_COLLISION_DATA + i * C.TOUCH_COLLISION_STRIDE
        struct.pack_into('<I', buf, cb + C.COL_BONE_HASH, int(c["bone_hash"]) & 0xFFFFFFFF)
        struct.pack_into('<3f', buf, cb + C.COL_CONTACT_POINT, *[float(x) for x in c["point_cm"]])
        struct.pack_into('<3f', buf, cb + C.COL_CONTACT_NORMAL, *[float(x) for x in c["normal"]])
        struct.pack_into('<f', buf, cb + C.COL_PENETRATION_DEPTH, float(c["penetration_cm"]))
        struct.pack_into('<f', buf, cb + C.COL_IMPACT_SPEED, float(c["impact_speed_cms"]))
        struct.pack_into('<3f', buf, cb + C.COL_IMPACT_IMPULSE, *[float(x) for x in c["impulse_ns"]])
        struct.pack_into('<f', buf, cb + C.COL_CONTACT_DURATION, float(c["duration_s"]))
        struct.pack_into('<i', buf, cb + C.COL_FRAME_COUNT, int(c["frame_count"]))


# ------------------------------------------------------------------ audio (crib hearing)
def write_audio(buf: bytearray, spectral01, meta: dict) -> None:
    """Write the avatar's hearing contract from crib synthesis (audio_adapter):
    a [256, 2] log-dB-normalized stereo spectrum + binaural metadata, laid out
    exactly where UE's HearingCaptureComponent writes and tokenize_audio reads."""
    import numpy as np
    ab = C.AUDIO_OFFSET
    spec = np.ascontiguousarray(np.asarray(spectral01, dtype='<f4'))
    n_bins = int(min(spec.shape[0], C.AUDIO_SPECTRAL_BINS))
    struct.pack_into('<i', buf, ab + C.AUD_NUM_SAMPLES, 0)      # no PCM: spectrum-direct
    struct.pack_into('<i', buf, ab + C.AUD_SAMPLE_RATE, 0)
    struct.pack_into('<i', buf, ab + C.AUD_NUM_CHANNELS, 2)
    struct.pack_into('<f', buf, ab + C.AUD_LEFT_RMS, float(meta.get("left_rms", 0.0)))
    struct.pack_into('<f', buf, ab + C.AUD_RIGHT_RMS, float(meta.get("right_rms", 0.0)))
    struct.pack_into('<f', buf, ab + C.AUD_ILD_DB, float(meta.get("ild_db", 0.0)))
    struct.pack_into('<f', buf, ab + C.AUD_ITD_MS, float(meta.get("itd_ms", 0.0)))
    # slot conventions (tokenize_audio): azimuth in DEGREES (/90 normalize on read);
    # spectrum PLANAR - all left bins then all right bins, reshape(2, n).T on read
    struct.pack_into('<f', buf, ab + C.AUD_AZIMUTH, float(np.degrees(meta.get("azimuth", 0.0))))
    struct.pack_into('<f', buf, ab + C.AUD_FRONT_BACK, float(meta.get("front_back", 0.0)))
    struct.pack_into('<i', buf, ab + C.AUD_SPECTRAL_NBINS, n_bins)
    raw = np.ascontiguousarray(spec[:n_bins].T).tobytes()
    buf[ab + C.AUD_SPECTRAL: ab + C.AUD_SPECTRAL + len(raw)] = raw


# ------------------------------------------------------------------ voice (optional)
def write_voice(buf: bytearray, params12: Sequence[float], synth_time: float,
                diameters: Sequence[float]) -> None:
    vb = C.VOICE_OFFSET
    offs = [C.VOC_TARGET_FREQ, C.VOC_TARGET_TENS, C.VOC_TONGUE_INDEX, C.VOC_TONGUE_DIAMETER,
            C.VOC_VIBRATO_AMOUNT, C.VOC_VIBRATO_FREQ, C.VOC_INTENSITY, C.VOC_LOUDNESS,
            C.VOC_VELUM_TARGET, C.VOC_AUTO_WOBBLE, C.VOC_ALWAYS_VOICE]
    for o, v in zip(offs, params12):
        struct.pack_into('<f', buf, vb + o, float(v))
    struct.pack_into('<d', buf, vb + C.VOC_SYNTH_TIME, float(synth_time))
    d = list(diameters)[:C.VOC_NUM_DIAMETERS]
    struct.pack_into(f'<{len(d)}f', buf, vb + C.VOC_DIAMETERS, *[float(x) for x in d])


# ------------------------------------------------------------------ one-shot
def encode_frame(*, frame_number: int, timestamp: float, dt: float, wall_clock: float,
                 gaze_az_deg: float, gaze_el_deg: float,
                 bones: np.ndarray, bone_hashes: Optional[Sequence[int]],
                 handles: np.ndarray, energy: Sequence[float], body_sets: Optional[np.ndarray],
                 patches: np.ndarray, collisions: Sequence[dict],
                 vision_valid: bool = False,
                 audio_spectral=None, audio_meta: Optional[dict] = None) -> bytearray:
    """Pack a whole frame. Mask = Proprio | Motor | Touch (+ Vision if vision_valid,
    + Audio when the crib's synthesized hearing is supplied). Voice is left off."""
    buf = new_frame_buffer()
    mask = C.MOD_PROPRIOCEPTION | C.MOD_MOTOR | C.MOD_TOUCH
    if vision_valid:
        mask |= C.MOD_VISION
    if audio_spectral is not None:
        mask |= C.MOD_AUDIO
    write_header(buf, frame_number, timestamp, dt, wall_clock, gaze_az_deg, gaze_el_deg, mask)
    write_proprio(buf, bones, bone_hashes)
    write_motor(buf, handles, energy, body_sets)
    write_touch(buf, patches, collisions)
    if audio_spectral is not None:
        write_audio(buf, audio_spectral, audio_meta or {})
    return buf


def decode_with_model_tokenizer(slot_bytes: bytes, device: str = "cpu"):
    """Decode a frame with the MODEL'S OWN tokenizer functions -> TokenizedFrame.

    This is the same code path the live model uses on UE frames (tokenizer.read_frame),
    minus the shared-memory read and the GPU vision capture. Vision is attached by the
    data source separately (Mode A) or read from LEM_Vision (Mode B).
    """
    from tokenizer.config import TokenizerConfig, TokenizedFrame
    from tokenizer.tokenizer import LEMTokenizer
    from tokenizer.modalities.proprioception import tokenize_proprioception
    from tokenizer.modalities.motor import tokenize_motor
    from tokenizer.modalities.touch import tokenize_touch
    from tokenizer.modalities.voice import tokenize_voice
    from tokenizer.modalities.audio import tokenize_audio
    import torch

    cfg = TokenizerConfig(device=device)
    meta = LEMTokenizer._parse_frame_header(None, slot_bytes)  # pure function of the bytes
    mask = meta.modality_valid_mask
    out = dict(metadata=meta)
    if mask & C.MOD_PROPRIOCEPTION:
        out["proprio"] = tokenize_proprioception(slot_bytes, cfg)
    if mask & C.MOD_MOTOR:
        h, e, cap = tokenize_motor(slot_bytes, cfg)
        out.update(motor_handles=h, motor_energy=e, motor_capability=cap)
    if mask & C.MOD_TOUCH:
        imp, col, ni, nc = tokenize_touch(slot_bytes, cfg)
        out.update(touch_impulses=imp, touch_collisions=col, touch_valid_counts=(ni, nc))
    else:
        # 10 columns, the width tokenize_touch returns: the no-touch branch used to build
        # an 8-wide zero tensor, so a reader that trusted the shape saw a different sense
        # depending on whether anything was touching
        out.update(touch_impulses=torch.zeros(C.TOUCH_MAX_IMPULSES, 10),
                   touch_collisions=torch.zeros(C.TOUCH_MAX_COLLISIONS, 8))
    if mask & C.MOD_VOICE:
        out["voice"] = tokenize_voice(slot_bytes, cfg)
    if mask & C.MOD_AUDIO:
        s, m = tokenize_audio(slot_bytes, cfg)
        out.update(audio_spectral=s, audio_metadata=m)
    out["intermind"] = torch.zeros(3)
    return TokenizedFrame(**out)
