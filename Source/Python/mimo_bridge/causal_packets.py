# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Immutable pre-ANN sensory windows and prospective held-command outcomes.

This isolated research contract does not alter the live learner. Targets are
decoded experienced channels, independent of every learned encoder snapshot.
"""
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import torch

CONTRACT = 'raw_sensory_held_command_v1'
MODALITIES = ('vision', 'tactile', 'audio', 'proprioception')


def sensory_payload(payload):
    """Retain the native pre-ANN wire and sensed spatial side channel only."""
    keys = ('frame_bytes', 'extended_handles', 'eye_left', 'eye_right',
            'fovea_left', 'fovea_right', 'vision_layout', 'sim_time', 'dt',
            'frame', 'patches', 'bones', 'vestibular', 'eye_angles', 'gaze_deg',
            'touch_pos', 'touch_sensitivity')
    out = {k: deepcopy(payload.get(k)) for k in keys}
    # Required transport fields that the encoder does not consume.
    out.update(n_contacts=0, ctrl=np.empty(0, np.float32))
    out['info'] = {k: deepcopy(payload['info'][k]) for k in
                   ('mechanical_contact', 'skin_function_granularity')
                   if k in payload.get('info', {})}
    return out


def frame_from_payload(payload):
    from .datasource import _payload_to_frame
    return _payload_to_frame(payload, 0)


def fixed_targets(frame):
    """Units-fixed proprioception and EXPERIENCED pressure/location moments.

    Proprioception is the deterministic byte decoder, before any ANN. Positions
    are centimetres relative to the first sensed handle, as in the live wire.
    Rotation columns remove quaternion sign ambiguity. Spatial moments are the
    existing sensed pressure-weighted channel, not simulator object geometry.
    """
    from tokenizer.config import TokenizerConfig
    from tokenizer.modalities.touch import _FIELD_NORMS
    from .skin_contract import frame_skin_features
    p = frame.tokenized.proprio.detach().cpu().numpy()
    q = p[:, :4].astype(np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if (norm < .5).any():
        raise ValueError('invalid sensed quaternion')
    x, y, z, w = (q / norm).T
    rot = np.stack((1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w),
                    2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)), -1)
    spatial = frame.meta.get('skin_function_granularity', {})
    moments = np.asarray(spatial.get('surface_moments_root', []), np.float64)
    if moments.ndim != 2 or moments.shape[1] != 20 or not len(moments):
        raise ValueError('actual sensed contact moments are required')
    pressure = frame_skin_features(frame)[:len(moments), 0].cpu().numpy()
    pressure = pressure.astype(np.float64) * float(_FIELD_NORMS[0]) / 1000
    if len(pressure) != len(moments):
        raise ValueError('pressure and spatial site identities differ')
    active = moments[:, 18] > 0
    centroid = moments[:, :3] * 100 * active[:, None]
    covariance = moments[:, 3:9] * 10000 * active[:, None]
    ij = np.triu_indices(3)
    second = covariance + (centroid[:, :, None]*centroid[:, None, :])[:, ij[0], ij[1]]
    groups = dict(position_cm=p[:, 4:7]*TokenizerConfig().position_range_cm,
                  rotation_6d=rot, touch_kpa=pressure,
                  touch_first_kpa_cm=pressure[:, None]*centroid,
                  touch_second_kpa_cm2=pressure[:, None]*second)
    groups = {k: np.asarray(v, np.float32).reshape(-1) for k, v in groups.items()}
    if not all(np.isfinite(v).all() for v in groups.values()):
        raise ValueError('nonfinite sensed target')
    return groups


def packed_target(groups):
    return np.concatenate(list(groups.values())).astype(np.float32)


def target_layout(groups):
    out = {}; start = 0
    for key, values in groups.items():
        out[key] = [start, start+len(values)]; start += len(values)
    return out


def validate(packet):
    if packet.get('contract') != CONTRACT:
        raise ValueError('unknown causal packet')
    if packet['input_provenance'] != 'predecision_raw_senses_and_precommitted_commands':
        raise ValueError('posthoc commands cannot establish prospective prediction')
    if not packet.get('reflex_off') or not packet.get('saccades_off'):
        raise ValueError('future reactive gaze is not known at decision time')
    context = packet['context']; times = np.array([p['sim_time'] for p in context])
    if len(context) != 6 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('context must be a strictly causal, ordered sensory window')
    from mimo_bridge.frame_encoder import decode_with_model_tokenizer
    for raw in context:
        metadata=decode_with_model_tokenizer(raw['frame_bytes']).metadata
        if (metadata.timestamp!=raw['sim_time'] or metadata.frame_number!=raw['frame']
            or np.float32(metadata.dt)!=np.float32(raw['dt'])):
            raise ValueError('native sensory header disagrees with relabelled transport chronology')
    if times[-1] != packet['decision_time']:
        raise ValueError('raw inputs must end exactly at the decision')
    cmd = np.asarray(packet['commands']); dt = np.asarray(packet['durations'])
    y = np.asarray(packet['targets']); outcome = np.asarray(packet['outcome_times'])
    start_frame=frame_from_payload(context[-1])
    groups=fixed_targets(start_frame)
    if cmd.shape != (4,5,len(start_frame.tokenized.proprio),9) or dt.shape != (4,5):
        raise ValueError('command duration shape mismatch')
    if y.shape[:2] != dt.shape or outcome.shape != dt.shape:
        raise ValueError('outcome and command chronology mismatch')
    if not all(np.isfinite(a).all() for a in (cmd, dt, y, outcome)) or (dt <= 0).any():
        raise ValueError('invalid causal sequence values')
    if not np.allclose(outcome, times[-1]+dt.cumsum(1), atol=1e-7, rtol=0):
        raise ValueError('future outcome time differs from the actual executed duration')
    if not np.allclose(dt,.05,atol=1e-7,rtol=0) or not np.allclose(np.diff(times),.05,atol=1e-7,rtol=0):
        raise ValueError('actual observation intervals differ from declared five-frame cadence')
    phase=packet['phase']
    if phase not in range(5) or not np.isclose(times[-1]-packet['order_started'],phase*dt[0,0],atol=1e-7,rtol=0):
        raise ValueError('imposed order age differs from the recorded physical start')
    gaze = int(packet['gaze_handle'])
    if gaze!=23:raise ValueError('this native MIMo packet requires its actual gaze handle 23')
    if not np.array_equal(cmd[:, :, gaze], np.broadcast_to(cmd[0, 0, gaze], cmd[:, :, gaze].shape)):
        raise ValueError('body-only action controls must retain identical fixed gaze')
    if packet.get('world_id') is None or not packet.get('source_checkpoint_sha256'):
        raise ValueError('missing world or learned source identity')
    expected = packed_target(groups)
    if packet['target_layout']!=target_layout(groups) or y.shape!=(4,5,len(expected)):
        raise ValueError('fixed target layout or outcome width differs from actual sensed channels')
    if not np.array_equal(expected, np.asarray(packet['start'])):
        raise ValueError('start target was not the fixed pre-ANN sensory readout')
    if not np.array_equal(packet['replay_target'], y[0]):
        raise ValueError('identical held-command physical replay differed')
    return packet


def write_packet(folder, packet, max_bytes=512*1024**2, reserve_bytes=2*1024**3):
    validate(packet); folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    raw = pickle.dumps(packet, protocol=5); digest = hashlib.sha256(raw).hexdigest()
    payload = gzip.compress(raw, compresslevel=1, mtime=0)
    current = sum(p.stat().st_size for p in folder.glob('*.packet.gz'))
    if current+len(payload) > max_bytes or shutil.disk_usage(folder).free-len(payload) < reserve_bytes:
        raise OSError('bounded causal recording budget or free-space reserve reached; records retained')
    path = folder/(digest+'.packet.gz')
    if path.exists():
        if path.read_bytes() != payload: raise ValueError('immutable packet identity collision')
    else:
        pending = path.with_suffix('.pending'); pending.write_bytes(payload); pending.replace(path)
    return dict(path=path.name, sha256=digest, compressed_bytes=len(payload))


def load_packet(path):
    path = Path(path); raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != path.name.split('.')[0]:
        raise ValueError('recorded raw packet checksum differs')
    return validate(pickle.loads(raw))


@torch.no_grad()
def encode_own_window(model, context):
    """Each snapshot supplies its OWN encoder; targets never use that encoder."""
    frames = [frame_from_payload(p) for p in context]
    raw, mask, _, _ = model._batch_frames(frames)
    if any(mask.get(k) != 'real' for k in MODALITIES):
        raise ValueError('complete real vision, touch, audio and proprioception required')
    registry = model.encoder_registry
    modes = {id(m): m.training for m in registry.modules()}
    try:
        registry.eval()
        from lem.model.spatial import encode_spatial
        encoded, regions, valid = encode_spatial(registry, {k: raw[k] for k in MODALITIES}, mask)
        # Preserve the existing retinal grid and local touch tokens. Pooling
        # everything back to one token would repeat the contact-geometry gap.
        features = torch.cat([encoded[k].reshape(len(context), -1) for k in MODALITIES]+
                             [(regions*valid[..., None]).reshape(len(context), -1), valid.float()], -1)
        return features.detach().cpu(), np.stack([packed_target(fixed_targets(f)) for f in frames])
    finally:
        for module in registry.modules(): module.training = modes[id(module)]


def tensor_digest(value):
    """Exact value digest for old model/Adam preservation, with stable key order."""
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous(); h.update(str((a.dtype, tuple(a.shape))).encode())
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(repr(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for y in x: visit(y)
        else: h.update(repr(x).encode())
    visit(value); return h.hexdigest()
