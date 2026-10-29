# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Retain spatial features alongside the existing modality summaries.

The global tokens and their checkpoint shapes stay intact. Consequence queries
can attend to local features directly, before information is pooled into one token.
"""
from __future__ import annotations
import torch


def vision_regions(features, angles, grid=4):
    """Separate retinal cells for each eye and source resolution, with no labels.

    Coordinates remain in the features' existing angular position encoding. Cell
    assignment uses retinal extent per eye/source, so changing gaze does not reorder
    the grid. Empty cells are masked; camera identity and scale never get averaged.
    """
    if angles.shape[-1] < 4:
        raise ValueError('spatial vision requires azimuth, elevation, eye and span')
    if features.shape[:2] != angles.shape[:2]:
        raise ValueError('vision subsampling lost patch-to-angle correspondence')
    batches, masks = [], []
    for x, a in zip(features, angles):
        groups = torch.unique(a[:, 2:4], dim=0, sorted=True)
        cells, good = [], []
        for group in groups:
            keep = (a[:, 2:4] == group).all(-1)
            coords = a[keep, :2]
            extent = (coords.amax(0) - coords.amin(0)).clamp_min(1e-6)
            cell = (((coords - coords.amin(0)) / extent) * grid).long().clamp(0, grid - 1)
            index = cell[:, 1] * grid + cell[:, 0]
            sums = x.new_zeros(grid * grid, x.shape[-1]).index_add(0, index, x[keep])
            counts = x.new_zeros(grid * grid).index_add(0, index, x.new_ones(len(index)))
            cells.append(sums / counts.clamp_min(1)[:, None])
            good.append(counts > 0)
        batches.append(torch.cat(cells))
        masks.append(torch.cat(good))
    max_n = max(len(x) for x in batches)
    return (torch.stack([torch.nn.functional.pad(x, (0, 0, 0, max_n - len(x))) for x in batches]),
            torch.stack([torch.nn.functional.pad(x, (0, max_n - len(x))) for x in masks]))


def encode_spatial(registry, frame, source_mask, modality_embedding=None):
    """Run shared encoders once; return global tokens and local features with masks."""
    tokens, local, masks = {}, [], []
    for name in registry.active_modalities:
        if name not in frame or source_mask.get(name) == 'disabled':
            continue
        enc, data = registry.encoders[name], frame[name]
        if name == 'vision' and source_mask.get(name) == 'real':
            cap = getattr(enc, 'patch_subsample', None)
            if cap and data['patches'].shape[1] > cap:
                raise ValueError('spatial vision needs deterministic full patch geometry; increase patch cap')
            vision_device = next(enc.parameters()).device
            features = enc.backbone_forward(data['patches'].to(vision_device), data['angular_pos'].to(vision_device))
            token = enc.head_forward(features)
            regions, valid = vision_regions(enc.proj(features), data['angular_pos'].to(vision_device))
            primary = registry._primary_device or vision_device
            token, regions, valid = token.to(primary), regions.to(primary), valid.to(primary)
            if registry.output_norm is not None:
                regions = registry.output_norm(regions)
            if modality_embedding is not None:
                regions = regions + modality_embedding.weight[registry.active_modalities.index(name)]
            local.append(regions)
            masks.append(valid)
        elif name == 'tactile' and source_mask.get(name) == 'real':
            features = enc.backbone_forward(data)
            token = enc.head_forward(features)
            sites = enc.proj(features)
            if registry.output_norm is not None:
                sites = registry.output_norm(sites)
            if modality_embedding is not None:
                sites = sites + modality_embedding.weight[registry.active_modalities.index(name)]
            local.append(sites)
            masks.append(data[..., -1] > 0)
        else:
            tokens[name] = registry._encode_one(name, data, enc)
            continue
        tokens[name] = registry.output_norm(token) if registry.output_norm is not None else token
    first = next(iter(tokens.values()))
    if not local:
        return tokens, first.new_zeros(first.shape[0], 0, first.shape[-1]), torch.zeros(
            first.shape[0], 0, device=first.device, dtype=torch.bool)
    return tokens, torch.cat(local, 1), torch.cat(masks, 1)
