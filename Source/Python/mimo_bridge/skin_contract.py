# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Geometry and regional tissue sensitivity shared by MIMo and richer UE skin.

No sensor index is an anatomical meaning. Names identify producer regions for
calibration and bookkeeping; the learned representation receives physical positions.
UE producers supply the explicit orthogonal world-to-root coordinate transform.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SkinLayout:
    region_ids: tuple[str, ...]
    positions_root_m: np.ndarray
    sensitivity: np.ndarray
    position_resolution: str

    def __post_init__(self):
        n = len(self.region_ids)
        if self.positions_root_m.shape != (n, 3) or self.sensitivity.shape != (n, 2):
            raise ValueError('skin geometry and sensitivity must match the named sites')
        if not np.isfinite(self.positions_root_m).all() or not np.isfinite(self.sensitivity).all():
            raise ValueError('nonfinite skin geometry or sensitivity')
        if (self.sensitivity < 0).any():
            raise ValueError('skin sensitivity cannot be negative')

    @classmethod
    def from_ue_cm(cls, region_ids, positions_world_cm, root_world_cm,
                   world_to_root, sensitivity):
        rotation = np.asarray(world_to_root, float)
        # A reflection is allowed: UE and another body's coordinates can have
        # different handedness. Scaling/shearing is not a coordinate convention.
        if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError('world_to_root must be an orthogonal coordinate transform')
        pos = (np.asarray(positions_world_cm) - np.asarray(root_world_cm)) / 100.
        return cls(tuple(region_ids), (pos @ rotation.T).astype(np.float32),
                   np.asarray(sensitivity, np.float32), 'surface_sites')

    def attach(self, bridge_frame, physical_fields, *, body_function=None, mechanical_history=None):
        """Attach a richer UE skin packet without truncating it to the legacy wire.

        physical_fields columns: Pa, Pa, mm/s, mm2, four vibration powers (N2).
        The existing SHM tokenizer remains readable; this explicit side channel
        supplies the richer producer's sites to both wellness and the new learner.
        """
        from dataclasses import replace
        from tokenizer.modalities.touch import _FIELD_NORMS
        values = np.asarray(physical_fields, np.float32)
        if values.shape != (len(self.region_ids), 8) or not np.isfinite(values).all():
            raise ValueError('physical skin fields must match all sites and be finite')
        meta=dict(bridge_frame.meta or {}, skin_geometry='surface_sites_root_axes')
        if body_function is not None:
            from lem.body_function import physical_drives
            if body_function.site_ids!=self.region_ids or mechanical_history is None:
                raise ValueError('UE function needs matching site identities and measured mechanical history')
            load,stim=physical_drives(values[:,0],mechanical_history,self.sensitivity)
            body_function.advance(load,stim,float(bridge_frame.sim_time))
            meta['body_function']=dict(body_function.record(),mode='modulate')
            meta['physical_touch_audit']=values.copy()
            values=body_function.observe(values)
        return replace(bridge_frame, touch_pos=self.positions_root_m.copy(),
                       touch_sensitivity=self.sensitivity.copy(),
                       skin_features=values / _FIELD_NORMS[:8],
                       meta=meta)


def frame_skin_features(frame):
    import torch
    rich = getattr(frame, 'skin_features', None)
    return torch.as_tensor(rich, dtype=torch.float32) if rich is not None else frame.tokenized.touch_impulses


def batch_skin_features(frames, device):
    import torch
    rows = [frame_skin_features(f).to(device) for f in frames]
    n = max(len(row) for row in rows)
    return torch.stack([torch.nn.functional.pad(row, (0, 0, 0, n - len(row))) for row in rows])


def regional_sensitivity(names, profile_path=''):
    """Neutral gains until a producer/body-specific calibration is provided.

    Density already determines MIMo skin resolution. Gains do not invent a universal
    relationship between receptor count, pain and pleasure. Profiles must name their
    calibration source and define separate pain/pleasant-touch gains per region.
    """
    out = np.ones((len(names), 2), dtype=np.float32)
    if not profile_path:
        return out
    profile = json.loads(Path(profile_path).read_text(encoding='utf-8'))
    if not profile.get('source'):
        raise ValueError('a regional skin profile must identify its calibration source')
    known = set(names)
    unknown = set(profile['regions']) - known
    if unknown:
        raise ValueError(f'skin profile names regions absent from this body: {sorted(unknown)}')
    for i, name in enumerate(names):
        if name in profile['regions']:
            entry = profile['regions'][name]
            out[i] = [entry['pain_gain'], entry['pleasant_gain']]
    if not np.isfinite(out).all() or (out < 0).any():
        raise ValueError('regional gains must be finite and nonnegative')
    return out


def measured_contact_centres(body_ids, fallback_positions, root_world, contacts):
    """MIMo's force-weighted contact location per sensing body, in metres.

    Uncontacted sites retain their body centre. Multiple contacts on one body still
    share one pressure patch: this explicitly remains coarser than UE surface skin.
    """
    result = np.array(fallback_positions, dtype=np.float32, copy=True)
    index = {int(body): i for i, body in enumerate(body_ids)}
    positions, weights = np.zeros_like(result), np.zeros(len(result))
    for c in contacts:
        i = index.get(int(c.body_id))
        if i is not None and c.normal_force > 0:
            positions[i] += (np.asarray(c.pos_world) - root_world) * c.normal_force
            weights[i] += c.normal_force
    active = weights > 0
    result[active] = positions[active] / weights[active, None]
    return result
