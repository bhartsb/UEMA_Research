# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The bridge's delayed bodily lesson and current-need choice interface.

The neural head lives in LemLite's optimizer/checkpoint. This adapter pairs a
past sensory frame with the entire sequence of orders actually executed after it.
It never trains on an imagined target or crosses an episode reset.
"""
from __future__ import annotations
from collections import deque
import numpy as np
import torch

from lem.model.spatial import encode_spatial
from lem.model.contact_wellness import contact_response, current_need_value


def detached(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    return value


def skin_observation(frames, device):
    """All instrumented skin sites, including currently untouched ones.

    Positions are relative to the root in metres, converted to the existing
    half-metre tactile coordinate scale. UE producers can supply surface sites
    and regional sensitivity through the same frame fields. Missing geometry
    fails explicitly; zero coordinates must not masquerade as located skin.
    """
    from .skin_contract import batch_skin_features
    features = batch_skin_features(frames, device)
    b, n, _ = features.shape
    pos = torch.zeros(b, n, 3, device=device)
    valid = torch.zeros(b, n, dtype=torch.bool, device=device)
    sensitivity = torch.ones(b, n, 2, device=device)
    for i, f in enumerate(frames):
        points = getattr(f, 'touch_pos', None)
        if points is None:
            raise ValueError('contact learning requires measured skin positions')
        if np.asarray(points).ndim != 2 or np.asarray(points).shape[1] != 3:
            raise ValueError('skin positions must have three coordinates per site')
        if getattr(f, 'skin_features', None) is not None and len(points) != len(f.skin_features):
            raise ValueError('rich skin geometry must cover every physical site')
        k = min(n, len(points))
        pos[i, :k] = torch.as_tensor(np.asarray(points), device=device)[:k] / .5
        valid[i, :k] = True
        gains = getattr(f, 'touch_sensitivity', None)
        if gains is not None:
            if np.asarray(gains).shape != (len(points), 2):
                raise ValueError('regional sensitivity must match the skin layout')
            sensitivity[i, :k] = torch.as_tensor(np.asarray(gains), device=device)[:k]
    capacity = torch.stack([f.tokenized.motor_capability[:10] for f in frames]).to(device)
    if not all(torch.isfinite(x).all() for x in (features, pos, capacity, sensitivity)):
        raise ValueError('nonfinite bodily observation')
    return dict(features=features, positions=pos, valid=valid,
                sensitivity=sensitivity, capacity=capacity)


class ConsequenceLesson:
    def __init__(self, horizon_steps=5):
        if horizon_steps < 1:
            raise ValueError('consequence horizon must be positive')
        self.horizon = horizon_steps
        self.reset()

    def reset(self):
        self.pending = None
        self.stats = {}
        self.current = None
        self.goal_conditions = deque(maxlen=200)

    def _state(self, model, frame, mask, context, skin):
        tokens, spatial, valid = encode_spatial(model.encoder_registry, frame, mask,
                                                model.forward_model.modality_embedding)
        if any(m not in tokens for m in model.active):
            raise ValueError('contact lesson needs a complete observed state')
        _, hidden, _ = model.forward_model(tokens, context)
        return hidden, spatial, valid

    def update(self, model, frames, frame, mask, context, command, train, curiosity=0.):
        from .lem_model import command_for_encoder
        skin = skin_observation(frames, model.device)
        skin['curiosity'] = torch.full((len(frames), 1), float(curiosity), device=model.device)
        times = np.asarray([f.sim_time for f in frames])
        counts = np.asarray([f.frame for f in frames])
        loss = None
        p = self.pending
        if p is not None:
            dt = times - p['last_time']
            contiguous = (command is not None and command.shape[0] == len(frames)
                          and np.all(counts > p['last_frame']) and np.all(dt > 0)
                          and np.array_equal(skin['valid'].cpu(), p['skin']['valid'].cpu()))
            if not contiguous:
                self.pending = None
            else:
                p['actions'].append(command_for_encoder(command.detach().float()))
                p['durations'].append(torch.as_tensor(dt, dtype=torch.float32, device=model.device))
                p['targets'].append(detached(skin))
                p['last_time'], p['last_frame'] = times, counts
                if len(p['actions']) == self.horizon:
                    hidden, spatial, valid = self._state(model, p['frame'], p['mask'], p['context'], p['skin'])
                    path_losses, path_stats = [], []
                    for h, outcome in enumerate(p['targets'], 1):
                        prediction = model.contact_consequences(hidden, spatial, valid,
                            p['skin']['positions'], p['skin']['features'], p['skin']['valid'],
                            torch.stack(p['actions'][:h], 1), torch.stack(p['durations'][:h], 1), p['skin']['capacity'])
                        term, stats = model.contact_consequences.loss(prediction, outcome['features'], outcome['valid'],
                                                                     outcome['capacity'], p['skin']['capacity'], outcome['curiosity'])
                        path_losses.append(term)
                        path_stats.append(stats)
                    loss = torch.stack(path_losses).mean()
                    stats = {k: torch.stack([s[k] for s in path_stats]).mean() for k in path_stats[0]}
                    persistence = ((torch.log1p(skin['features'].clamp_min(0)) -
                                    torch.log1p(p['skin']['features'].clamp_min(0))) ** 2)
                    persistence = (persistence * skin['valid'][..., None]).sum() / (skin['valid'].sum() * 8).clamp_min(1)
                    self.stats = {k: float(v) for k, v in stats.items()}
                    self.stats.update(persistence_log_mse=float(persistence),
                                      duration_s=float(torch.stack(p['durations'], 1).mean(0).sum()))
                    self.pending = None
        if self.pending is None:
            self.pending = dict(frame=detached(frame), mask=dict(mask), context=detached(context),
                                skin=detached(skin), actions=[], durations=[], targets=[],
                                last_time=times, last_frame=counts)
        self.current = dict(frame=detached(frame), mask=dict(mask), context=detached(context),
                            skin=detached(skin))
        # This is measured bodily condition, used to revalue memories at decision time.
        response = contact_response(skin['features'], skin['sensitivity'], skin['valid'])
        self.stats.update(local_pain=float(response['pain'].mean()),
                          pleasant_contact=float(response['pleasant'].mean()))
        return loss if train else None

    def forecast(self, model, candidates, durations, body_only=False):
        """Score candidate EXECUTED sequences with one present-need interpretation.

        candidates [K,B,H,handles,9], in centimetres/xyzw/strength. Caller must use
        the shared execution transform and smoother BEFORE this boundary. This
        interface is used in isolated behavior comparisons before production use.
        """
        from .lem_model import command_for_encoder
        if getattr(model, 'contact_profile', 'legacy') in ('mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
            raise ValueError('This experimental contact forecast lacks pressure-history outputs; it cannot score mechanical_v2 choices')
        if self.current is None:
            raise RuntimeError('observe the real body before forecasting')
        c, forecasts, scores = self.current, [], []
        with torch.no_grad():
            hidden, spatial, valid = self._state(model, c['frame'], c['mask'], c['context'], c['skin'])
            for orders in candidates:
                path, values = [], []
                for h in range(1, orders.shape[1] + 1):
                    pred = model.contact_consequences(hidden, spatial, valid,
                        c['skin']['positions'], c['skin']['features'], c['skin']['valid'],
                        command_for_encoder(orders[:, :h]), durations[:, :h], c['skin']['capacity'])
                    response = contact_response(pred['features'], c['skin']['sensitivity'], c['skin']['valid'])
                    value = current_need_value(response['pain'], response['pleasant'], pred['capacity'],
                                               c['skin']['capacity'],
                                               None if body_only else pred['curiosity'],
                                               None if body_only else c['skin']['curiosity'])
                    values.append(value.squeeze(-1))
                    path.append(dict(features=pred['features'], capacity=pred['capacity'], pain=response['pain']))
                scores.append(torch.stack(values).mean(0))
                forecasts.append(path)
        scores = torch.stack(scores)
        if not torch.isfinite(scores).all():
            raise ValueError('nonfinite forecast cannot choose an action')
        return scores, forecasts

    def record_goal(self, curiosity):
        c = self.current['skin']
        response = contact_response(c['features'], c['sensitivity'], c['valid'])
        self.goal_conditions.append(dict(pain=response['pain'].detach(),
                                         pleasant=response['pleasant'].detach(),
                                         capacity=c['capacity'].detach(), curiosity=float(curiosity)))

    def revalue_goals(self, current_curiosity):
        c = self.current['skin']
        return torch.stack([current_need_value(g['pain'], g['pleasant'], g['capacity'], c['capacity'],
                            curiosity_satisfaction=torch.full_like(g['pain'], g['curiosity']),
                            curiosity_need=torch.full_like(g['pain'], float(current_curiosity))).squeeze(-1)
                            for g in self.goal_conditions])
