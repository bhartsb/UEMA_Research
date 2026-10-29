# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Local skin responses, independent of the number of numerical sampling patches.

Inputs use the existing tokenizer units: pressure / 100 kPa, area / 20000 mm2.
Sensitivity describes tissue; subdividing its numerical representation preserves
area and sensitivity. No object identity or desired action enters this calculation.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

WELLNESS_CONTRACT = 'local_contact_v1_2026_09_11'
TACTILE_PLEASANT_PEAK = 0.05
TACTILE_PEAK_WIDTH = 0.6
# LEGACY BRANCH ONLY - profile='legacy'. Every current run launches
# --contact-profile mechanical_v2_local, which replaces this ramp wholesale with
# skin_mechanics.mechanical_load.
#
# Renamed 2026-09-15: it was TACTILE_PAIN_START, which named the wrong mechanism.
# 0.1 is 10,000 Pa. That is 2.34x capillary closing pressure (32 mmHg = 4,266 Pa),
# the line above which sustained load impedes perfusion - and 22x BELOW the
# first-discomfort pressure the live profile is anchored to (219,400 Pa, the 2-5
# year subgroup mean in Alrimy et al. 2022). So it marks the onset of harmful
# SUSTAINED LOAD, not the onset of acute pain, and it never fired in a run that
# uses the mechanical profile.
TACTILE_SUSTAINED_LOAD_START = 0.1
TACTILE_PAIN_START = TACTILE_SUSTAINED_LOAD_START   # old name, kept so saved code and
                                                    # frozen snapshots still import
TACTILE_TEXTURE_REF = 0.5


def contact_response(features: torch.Tensor, sensitivity: Optional[torch.Tensor] = None,
                     valid: Optional[torch.Tensor] = None, *, profile: str = 'legacy',
                     history: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
    """Return separate local discomfort/pleasantness and their bodily summary.

    sensitivity[..., 0:2] = pain sensitivity, pleasant-touch sensitivity. Defaults
    are neutral gains, not invented fingertip calibrations. UE can supply measured
    regional gains. Area is the represented contact area, never a receptor count.
    Legacy recordings with pressure but no area use equal patch weights, explicitly
    reported as missing_area; producers with measured area must supply it.
    """
    if features.ndim != 3 or features.shape[-1] < 8 or features.shape[1] == 0:
        raise ValueError("contact features must have shape [batch, sites, >=8]")
    if not torch.isfinite(features).all():
        raise ValueError("nonfinite skin input cannot be scored as comfort")
    pressure = features[..., 0].clamp_min(0)
    active = pressure > 1e-6
    if valid is not None:
        active = active & valid.bool()
    if sensitivity is None:
        sensitivity = torch.ones(*pressure.shape, 2, device=features.device,
                                 dtype=features.dtype)
    if sensitivity.shape != (*pressure.shape, 2):
        raise ValueError("sensitivity must match sites and contain two regional gains")
    if not torch.isfinite(sensitivity).all() or (sensitivity < 0).any():
        raise ValueError("regional sensitivity must be finite and nonnegative")
    local_pain = (((pressure * sensitivity[..., 0] - TACTILE_SUSTAINED_LOAD_START)
                   / (1.0 - TACTILE_SUSTAINED_LOAD_START)).clamp(0, 1) * active)
    if profile in ('mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
        from lem.skin_mechanics import PROFILE, WIDE_RANGE_PROFILE, mechanical_load
        if history is None:
            raise ValueError('mechanical_v2 requires the physical contact history')
        # mechanical_v4 keeps every onset, time constant and unit of v3 and
        # changes only how the summed load is squashed into 0-1. The exponential
        # squash stops separating loads above 6.5x the pressure reference, and
        # measured drops from 0.30 m up land between 3.48x and 8.02x it, so two
        # real falls score the same number. See lem/skin_mechanics.AVERSION_SHAPES.
        shape = WIDE_RANGE_PROFILE if profile == 'mechanical_v4' else PROFILE
        local_pain = mechanical_load(pressure * 100000., history, sensitivity[...,0], shape)
        if valid is not None:
            local_pain = local_pain * valid.bool()
    elif profile != 'legacy':
        raise ValueError(f'unknown contact profile {profile}')
    gentle = torch.exp(-((torch.log10(pressure.clamp_min(1e-5))
                          - math.log10(TACTILE_PLEASANT_PEAK)) ** 2)
                       / (2 * TACTILE_PEAK_WIDTH ** 2))
    texture = torch.tanh(features[..., 4:8].abs().mean(-1) / TACTILE_TEXTURE_REF)
    local_pleasant = (gentle * (0.5 + 0.5 * texture) * sensitivity[..., 1]).clamp(0, 1) * active
    area = features[..., 3].clamp_min(0)
    missing = active & (area <= 0)
    weights = torch.where(missing, torch.ones_like(area), area) * active
    pleasant = (local_pleasant * weights).sum(-1) / weights.sum(-1).clamp_min(1e-12)
    pain = local_pain.amax(-1)
    # Severe local discomfort remains visible irrespective of gentle contact elsewhere.
    comfort = (0.5 + 0.35 * pleasant) * (1 - pain)
    return {"comfort": comfort.unsqueeze(-1), "pain": pain.unsqueeze(-1),
            "pleasant": pleasant.unsqueeze(-1), "local_pain": local_pain,
            "local_pleasant": local_pleasant, "active": active,
            "missing_area": missing.sum(-1, keepdim=True)}


def current_need_value(pain, pleasant, capacity, current_capacity,
                       curiosity_satisfaction=None, curiosity_need=None):
    """Value consequences against PRESENT needs with distress kept explicit.

    Inputs are in [0,1]. Recovery matters in proportion to present fatigue. Optional
    curiosity satisfaction uses the existing bounded curiosity drive.
    Returns bodily utility with discomfort dominating bounded positive components.
    """
    need = (1 - current_capacity.detach()).clamp(0, 1)
    recovery = (need * capacity.clamp(0, 1)).mean(-1, keepdim=True)
    interest = torch.zeros_like(pain)
    if curiosity_satisfaction is not None and curiosity_need is not None:
        interest = curiosity_satisfaction.clamp(0, 1) * curiosity_need.detach().clamp(0, 1)
    satisfaction = (pleasant.clamp(0, 1) + recovery + interest) / 3
    return (1 - pain.clamp(0, 1)) * satisfaction - pain.clamp(0, 1)
