# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Versioned mechanical aversion reference; not a calibrated human pain scale.

The pressure reference is the 2–5 year subgroup mean first-discomfort pressure
from Alrimy et al., doi:10.3389/fpsyg.2022.963765 (1 cm2 rubber tip).
The accumulation, recovery and rubbing scales are explicit engineering settings.
They model a reversible aversive signal, not tissue damage or injury probability.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

MECHANICAL_CONTRACT = 'mechanical_contact_v2_2026_09_12'


@dataclass(frozen=True)
class MechanicalProfile:
    pressure_reference_pa: float = 219400.
    pressure_exposure_start_fraction: float = .25
    pressure_exposure_seconds: float = 60.
    recovery_seconds: float = 10.
    impact_trace_seconds: float = .1
    shear_reference_pa: float = 21940.
    rubbing_reference_j_m2: float = 2194.
    # How the summed load becomes a bounded aversion. See aversion_from_load.
    # 'exponential' is the 2026-09-12 contract and stays the default so every
    # existing profile and checkpoint keeps its exact numbers.
    aversion_shape: str = 'exponential'
    # Which friction work a measured friction power stores as rubbing. See RUBBING_RULES.
    # 'power_above_onset' is the 2026-09-12 contract and stays the default.
    rubbing_rule: str = 'power_above_onset'

    def __post_init__(self):
        numbers = [v for v in asdict(self).values() if not isinstance(v, str)]
        if any(not math.isfinite(x) or x <= 0 for x in numbers):
            raise ValueError('mechanical profile values must be finite and positive')
        if self.pressure_exposure_start_fraction >= 1:
            raise ValueError('exposure onset fraction must be below one')
        if self.aversion_shape not in AVERSION_SHAPES:
            raise ValueError(f'unknown aversion shape {self.aversion_shape}')
        if self.rubbing_rule not in RUBBING_RULES:
            raise ValueError(f'unknown rubbing rule {self.rubbing_rule}')


# Both map a nonnegative load to [0,1), are zero at zero load and rise without
# ever reaching one. They differ only in how fast they stop resolving.
#
#   exponential   1-exp(-load)   stops carrying information at load 5.5, which is
#                                6.5x the pressure reference. MEASURED 2026-09-16:
#                                drops from 0.30 m up land between 3.48x and 8.02x
#                                that reference, so the whole upper half of a fall
#                                ladder sits in its flat tail and two falls there
#                                cannot be told apart.
#   hyperbolic    load/(1+load)  still separates loads far past that. At load 21,
#                                one more unit of load moves it 0.0021, against
#                                0.00000000076 for the exponential.
#
# Both are chosen shapes for a bounded aversive signal. Neither is a calibrated
# human pain scale and neither claims tissue damage.
AVERSION_SHAPES = ('exponential', 'hyperbolic')

# What a measured friction power stores as rubbing work, once the shear stress exceeds the onset
# (shear_reference_pa):
#   power_above_onset  all of it (2026-09-12; onset 21.94 kPa, a tenth of the pressure reference)
#   excess_shear       only the work of the shear stress above the onset: power x (1 - onset/shear).
#                      This is what the unmeasured path below already stores, max(shear - onset, 0)
#                      x slip, so the two paths agree.
RUBBING_RULES = ('power_above_onset', 'excess_shear')


def aversion_from_load(load, shape):
    """Bounded aversion from summed dimensionless load. Torch in, torch out."""
    import torch
    if shape == 'exponential':
        return -torch.expm1(-load)
    if shape == 'hyperbolic':
        return load / (1.0 + load)
    raise ValueError(f'unknown aversion shape {shape}')


PROFILE = MechanicalProfile()
# Same onsets, same time constants, same units - only the shape of the final
# squash differs, so a load that scored 0 still scores 0 and the ordering of any
# two loads is unchanged. What changes is that the ordering stays READABLE at the
# pressures real falls produce.
WIDE_RANGE_PROFILE = MechanicalProfile(aversion_shape='hyperbolic')

# ORDINARY-CONTACT RUBBING RECALIBRATION (operator decision 2026-09-25 9:09 AM PDT: separate ordinary
# support and rubbing from harmful pressure and shear; docs/gain_of_function_v4/agents/
# ordinary_pain_calibration_2026-09-25.md). Rubbing stores only the work of the shear stress above the
# first-discomfort PRESSURE (219.4 kPa, the one measured mechanical pain threshold here; no measured
# shear pain threshold was found). Everything else is PROFILE's: the peak and exposure memories, the
# 10 s rubbing memory and its 2,194 J/m2 unit. MEASURED on H84 segments 36-40 (sampled footprints):
# rubbing footprints' shear stress median 75 kPa, 99th percentile 274 kPa, so the onset leaves 97 of
# 100 ordinary rubbing contacts storing nothing, and keeps 1-14 of every 100 parts of today's rubbing
# input per body (pain_calibration_measurements.py rules). Only the world's pain memory uses it; the
# model's load formula and the tissue-impairment memory are unchanged.
EXCESS_SHEAR_PROFILE = MechanicalProfile(shear_reference_pa=219400., rubbing_rule='excess_shear')
# The launch setting (BridgeEnv rubbing_calibration): 'v4' = today's memories only;
# 'excess_shear_v1_shadow' = today's memories feed pain, the recalibrated ones are advanced beside them
# and reported; 'excess_shear_v1' = the recalibrated memories feed pain, today's are reported.
RUBBING_CALIBRATIONS = {'v4': None, 'excess_shear_v1_shadow': EXCESS_SHEAR_PROFILE,
                        'excess_shear_v1': EXCESS_SHEAR_PROFILE}


def load_parts_np(pressure_pa, history, sensitivity, profile=PROFILE):
    """[..., 3] acute, sustained and rubbing load: skin_mechanics.mechanical_load in numpy, before the squash."""
    p = profile
    pressure = np.maximum(np.asarray(pressure_pa, dtype=np.float64), 0)
    h = np.asarray(history, dtype=np.float64)
    g = np.asarray(sensitivity, dtype=np.float64)
    peak = np.maximum(pressure, h[..., 0])
    return np.stack([np.maximum(peak * g / p.pressure_reference_pa - 1, 0),
                     h[..., 1] * g / (p.pressure_reference_pa * p.pressure_exposure_seconds),
                     h[..., 2] * g / p.rubbing_reference_j_m2], axis=-1)


def pain_np(load, shape):
    """aversion_from_load in numpy."""
    load = np.maximum(np.asarray(load, dtype=np.float64), 0)
    if shape == 'exponential':
        return -np.expm1(-load)
    if shape == 'hyperbolic':
        return load / (1.0 + load)
    raise ValueError(f'unknown aversion shape {shape}')


class MechanicalExposure:
    """Per-site physical history, exact leaky integration for piecewise constant input.

    Columns: decaying peak pressure (Pa), pressure exposure (Pa s), frictional
    work per area (J/m2). Histories belong to a body and its physical episode.
    """
    def __init__(self, sites, profile=PROFILE, pain_sensitivity=None):
        self.profile = profile
        self.values = np.zeros((int(sites), 3), dtype=np.float64)
        self.pain_sensitivity = np.ones(int(sites)) if pain_sensitivity is None else np.asarray(pain_sensitivity,dtype=float).copy()
        if self.pain_sensitivity.shape != (int(sites),) or not np.isfinite(self.pain_sensitivity).all() or (self.pain_sensitivity < 0).any():
            raise ValueError('pain sensitivity must be finite, nonnegative and site-aligned')

    def update(self, physical_fields, dt, friction_power_density=None):
        x=np.asarray(physical_fields, dtype=float)
        if x.shape[0] != len(self.values) or x.shape[1] < 4 or not np.isfinite(x).all():
            raise ValueError('invalid physical skin fields')
        if not math.isfinite(dt) or dt <= 0: raise ValueError('skin dt must be positive')
        p=self.profile
        pressure=np.maximum(x[:,0],0)
        shear=np.maximum(x[:,1],0)
        slip=np.maximum(x[:,2],0)/1000.  # mm/s -> m/s
        self.values[:,0]=np.maximum(pressure,self.values[:,0]*math.exp(-dt/p.impact_trace_seconds))
        gain=self.pain_sensitivity
        onset=p.pressure_exposure_start_fraction*p.pressure_reference_pa/np.maximum(gain,1e-12)
        rate=np.where(gain>0,np.maximum(pressure-onset,0),0)
        # Loaded sites accumulate towards a bounded exposure; unloaded sites recover
        # faster. Choosing time constants is separate from physiological validation.
        tau=np.where(rate > 0,p.pressure_exposure_seconds,p.recovery_seconds)
        decay=np.exp(-dt/tau)
        self.values[:,1]=self.values[:,1]*decay+rate*tau*(-np.expm1(-dt/tau))
        shear_onset=p.shear_reference_pa/np.maximum(gain,1e-12)
        work_rate=np.where(gain>0,np.maximum(shear-shear_onset,0)*slip,0)
        if friction_power_density is not None:
            measured=np.asarray(friction_power_density,dtype=float)
            if measured.shape != pressure.shape or not np.isfinite(measured).all() or (measured < 0).any():
                raise ValueError('friction power density must be finite, nonnegative and site-aligned')
            if getattr(p,'rubbing_rule','power_above_onset')=='excess_shear':
                # only the work of the shear above the onset: power x (1 - onset / shear)
                kept=np.where(shear>shear_onset,1-shear_onset/np.maximum(shear,1e-300),0)
                work_rate=np.where(gain>0,measured*kept,0)
            else:
                work_rate=np.where((gain>0)&(shear > shear_onset),measured,0)
        decay=math.exp(-dt/p.recovery_seconds)
        self.values[:,2]=self.values[:,2]*decay+work_rate*p.recovery_seconds*(-math.expm1(-dt/p.recovery_seconds))
        return self.values.copy()


def mechanical_load(pressure_pa, history, sensitivity, profile=PROFILE):
    """Torch implementation; sensitivity changes tissue response, never force units."""
    import torch
    p=profile
    if history.shape != (*pressure_pa.shape,3) or not torch.isfinite(history).all() or (history < 0).any():
        raise ValueError('mechanical history must be finite, nonnegative and site-aligned')
    peak=torch.maximum(pressure_pa,history[...,0])
    acute=(peak*sensitivity/p.pressure_reference_pa-1).clamp_min(0)
    sustained=history[...,1]*sensitivity/(p.pressure_reference_pa*p.pressure_exposure_seconds)
    rubbing=history[...,2]*sensitivity/p.rubbing_reference_j_m2
    # Zero means no modeled aversion. One is the asymptote for an extreme load;
    # under the default exponential shape twice the pressure reference gives
    # 0.632 acutely, not a human pain rating.
    return aversion_from_load(acute+sustained+rubbing, p.aversion_shape)
