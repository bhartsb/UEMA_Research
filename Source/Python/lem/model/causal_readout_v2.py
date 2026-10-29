# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Separate development readout; no production registration or intrinsic reward.

Weights are shared over body handles and skin sites. Site count/index never
selects an output row. Inputs and targets retain their distinct sensed frames.
"""
import torch
from torch import nn


GROUPS = ('position_cm', 'rotation_6d', 'touch_kpa',
          'touch_first_kpa_cm', 'touch_second_kpa_cm2')


def rotation_columns(q):
    """Quaternion xyzw -> sign-invariant first two world rotation columns."""
    if q.shape[-1] != 4 or not torch.isfinite(q).all():
        raise ValueError('invalid quaternion shape/values')
    norm = q.norm(dim=-1, keepdim=True)
    if (norm < .5).any():
        raise ValueError('invalid sensed/commanded quaternion norm')
    x, y, z, w = (q / norm).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w),
                        2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)), -1)


def layout_for(handles, sites):
    result = {}; start = 0
    for name, width in zip(GROUPS, (3*handles, 6*handles, sites, 3*sites, 6*sites)):
        result[name] = [start, start+width]; start += width
    return result


def split_senses(sensed, handles, sites):
    layout = layout_for(handles, sites)
    if sensed.shape[-1] != layout[GROUPS[-1]][1]:
        raise ValueError('fixed sensory width differs from declared handle/site counts')
    parts = {g: sensed[..., lo:hi] for g, (lo, hi) in layout.items()}
    body = torch.cat((parts['position_cm'].reshape(*sensed.shape[:-1], handles, 3),
                      parts['rotation_6d'].reshape(*sensed.shape[:-1], handles, 6)), -1)
    skin = torch.cat((parts['touch_kpa'][..., None],
                      parts['touch_first_kpa_cm'].reshape(*sensed.shape[:-1], sites, 3),
                      parts['touch_second_kpa_cm2'].reshape(*sensed.shape[:-1], sites, 6)), -1)
    return body, skin


def join_senses(body, skin):
    return torch.cat((body[..., :3].flatten(-2), body[..., 3:].flatten(-2),
                      skin[..., 0], skin[..., 1:4].flatten(-2),
                      skin[..., 4:].flatten(-2)), -1)


class LocalCausalReadout(nn.Module):
    """Own sensory history + prospective orders -> experienced channel changes.

    Body position targets are first-handle-relative cm. Achieved/order positions
    are world cm, scaled by the same fixed 200 cm. Skin positions are root-relative
    m, scaled by 2 m. No equality between these coordinate frames is assumed.
    Spatial attention consumes existing retinal/local-touch encoder regions.
    """
    def __init__(self, encoded_width, region_width, handles, sites, hidden=48,
                 achieved_input=True):
        super().__init__()
        self.handles = int(handles); self.sites = int(sites)
        self.hidden = int(hidden); self.achieved_input = bool(achieved_input)
        self.global_history = nn.GRU(encoded_width+1, hidden, batch_first=True)
        self.body_history = nn.GRU(9+9+hidden+1, hidden, batch_first=True)
        self.skin_query = nn.Linear(13, hidden)
        self.region_key = nn.Linear(region_width, hidden, bias=False)
        self.region_value = nn.Linear(region_width, hidden, bias=False)
        self.skin_history = nn.GRU(10+3+hidden+hidden+1, hidden, batch_first=True)
        self.body_step = nn.GRUCell(11+9+hidden+1, hidden)
        self.body_coupling = nn.GRUCell(hidden, hidden)
        self.skin_step = nn.GRUCell(hidden+hidden+1, hidden)
        self.body_output = nn.Linear(hidden, 9)
        self.skin_output = nn.Linear(hidden, 10)
        for output in (self.body_output, self.skin_output):
            nn.init.zeros_(output.weight); nn.init.zeros_(output.bias)

    def forward(self, encoded, regions, region_valid, sensed, achieved,
                site_position, history_dt, actions, durations, blind=False):
        b, t, _ = encoded.shape; h, s = self.handles, self.sites
        if (sensed.shape[:2] != (b,t) or achieved.shape != (b,t,h,9)
            or site_position.shape != (b,t,s,3) or history_dt.shape != (b,t)
            or actions.ndim != 4 or actions.shape[0] != b or actions.shape[2:] != (h,9)
            or durations.shape != actions.shape[:2] or (durations <= 0).any()
            or regions.shape[:2] != (b,t) or region_valid.shape != regions.shape[:3]
            or not region_valid.any(-1).all()):
            raise ValueError('causal local readout input shape/time/spatial mask differs')
        if not self.achieved_input:
            achieved = torch.zeros_like(achieved)
        body, skin = split_senses(sensed, h, s)
        g, _ = self.global_history(torch.cat((encoded, history_dt[...,None]), -1))
        gb = g[:,:,None].expand(-1,-1,h,-1)
        dtb = history_dt[:,:,None,None].expand(-1,-1,h,-1)
        bh = torch.cat((body, achieved, gb, dtb), -1).permute(0,2,1,3).reshape(b*h,t,-1)
        _, bh = self.body_history(bh); bh = bh[0].reshape(b,h,-1)
        query = self.skin_query(torch.cat((skin, site_position), -1))
        keys = self.region_key(regions)
        score = torch.einsum('btsh,btrh->btsr', query, keys) / self.hidden**.5
        score = score.masked_fill(~region_valid[:,:,None], -torch.inf)
        spatial = torch.einsum('btsr,btrh->btsh', score.softmax(-1), self.region_value(regions))
        gs = g[:,:,None].expand(-1,-1,s,-1)
        dts = history_dt[:,:,None,None].expand(-1,-1,s,-1)
        sh = torch.cat((skin, site_position, spatial, gs, dts), -1)
        _, sh = self.skin_history(sh.permute(0,2,1,3).reshape(b*s,t,-1))
        sh = sh[0].reshape(b,s,-1)
        output = []; global_now = g[:,-1]
        for tick in range(actions.shape[1]):
            cmd = actions[:,tick]
            cmd_pose = torch.cat((cmd[...,:3]/200., rotation_columns(cmd[...,3:7])), -1)
            cmd_features = torch.cat((cmd_pose, cmd[...,7:9]), -1)
            # Error is relative to the LAST OBSERVED pose, never future truth.
            error = cmd_pose-achieved[:,-1] if self.achieved_input else torch.zeros_like(cmd_pose)
            if blind:
                cmd_features = torch.zeros_like(cmd_features); error = torch.zeros_like(error)
            dt = durations[:,tick,None,None]
            bh = self.body_step(torch.cat((cmd_features, error,
                global_now[:,None].expand(-1,h,-1),dt.expand(-1,h,-1)), -1).reshape(b*h,-1),
                bh.reshape(b*h,-1)).reshape(b,h,-1)
            # Every handle can be affected by another handle's prospective
            # command (including trunk and the moving coordinate reference).
            coupling = torch.einsum('bih,bjh->bij',bh,bh)/self.hidden**.5
            others = torch.einsum('bij,bjh->bih',coupling.softmax(-1),bh)
            bh = self.body_coupling(others.reshape(b*h,-1),bh.reshape(b*h,-1)).reshape(b,h,-1)
            attention = torch.einsum('bsh,bkh->bsk',sh,bh)/self.hidden**.5
            body_context = torch.einsum('bsk,bkh->bsh',attention.softmax(-1),bh)
            sh = self.skin_step(torch.cat((body_context,global_now[:,None].expand(-1,s,-1),
                dt.expand(-1,s,-1)),-1).reshape(b*s,-1),sh.reshape(b*s,-1)).reshape(b,s,-1)
            output.append(join_senses(self.body_output(bh),self.skin_output(sh)))
        return torch.stack(output,1)


def group_scales(rows, layout):
    """One outcome/contrast scale per physical group, shared by ALL site rows.

    All train sites participate, including zero-change sites. No per-site mask
    can leave an unseen site's output weights untrained. Quiet groups have no
    contrast term; their ordinary prediction term remains present.
    """
    delta = torch.stack([r['target']-r['start'] for r in rows])
    centered = delta-delta.mean(1,keepdim=True)
    scale = torch.ones(delta.shape[-1]); contrast = torch.zeros_like(scale)
    evidence = {}
    for group,(lo,hi) in layout.items():
        rms = delta[...,lo:hi].square().mean().sqrt()
        effect = centered[...,lo:hi].square().mean().sqrt()
        floor = 8*torch.finfo(delta.dtype).eps*torch.stack([
            r['target'][...,lo:hi].abs().amax() for r in rows]).amax().clamp_min(1.)
        scale[lo:hi] = rms if rms>floor else 1.
        contrast[lo:hi] = effect/scale[lo] if effect>floor else 0.
        evidence[group] = dict(outcome_rms=float(rms),action_contrast_rms=float(effect),
                              numerical_floor=float(floor),contrast_informative=bool(effect>floor))
    return scale, contrast, evidence


def grouped_objective(prediction, target, layout, contrast_scale, use_contrast):
    """Equal group mean losses; contrast term is RMS-normalized on fit data.

    prediction/target are decision x candidate x time x fixed channel. A common
    trajectory cannot reduce the centered term. Each informative group's term
    equals one for persistence on the fit set, fixing its relative weight
    from measured signal energy rather than a favorable evaluation result.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError('contrast training requires intact four-candidate decision groups')
    ordinary = []; contrast = []
    pc = prediction-prediction.mean(1,keepdim=True)
    tc = target-target.mean(1,keepdim=True)
    for lo,hi in layout.values():
        ordinary.append((prediction[...,lo:hi]-target[...,lo:hi]).square().mean())
        c = contrast_scale[lo]
        if use_contrast and c>0:
            contrast.append(((pc[...,lo:hi]-tc[...,lo:hi])/c).square().mean())
    ordinary = torch.stack(ordinary).mean()
    contrast = torch.stack(contrast).mean() if contrast else ordinary*0
    return ordinary+contrast, dict(outcome=ordinary.detach(),contrast=contrast.detach())
