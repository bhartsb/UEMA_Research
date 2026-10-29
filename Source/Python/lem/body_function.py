# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Shared, versioned local sensory functioning, independent of embodiment and policy.

This is a simulation reference, not a calibrated human pain or injury model. Physical
load changes an actual observation operator. Its state is an interoceptive observation,
not a reward or an instruction about which movement to perform.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

CONTRACT = 'body_function_v1_2026_09_12'
FUNCTION_FIELDS = ('resolution_fraction', 'load_fraction', 'stimulation_fraction')


@dataclass(frozen=True)
class FunctionProfile:
    # Endpoints describe representational precision, not human pain magnitudes.
    # At full function the original float32 observation is returned exactly.
    minimum_bits: int = 2
    maximum_bits: int = 23
    pressure_range_pa: float = 100000.
    onset_seconds: float = .1
    recovery_seconds: float = 10.
    positive_fraction: float = .02

    def __post_init__(self):
        if not (1 <= self.minimum_bits < self.maximum_bits <= 23):
            raise ValueError('resolution endpoints must fit float32 precision')
        if not (0 <= self.positive_fraction <= 1):
            raise ValueError('positive modulation must be bounded')
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.pressure_range_pa,self.onset_seconds,self.recovery_seconds)):
            raise ValueError('invalid body-function units or times')


class BodyFunction:
    """One body's persistent local functioning. q=1 means original sensing available.

    Negative modulation contracts distinguishable resolution. Positive stimulation
    makes a small bounded increase while recovery is possible; it cannot exceed the
    original sensor's capacity. Subdividing a site does not change its local dynamics.
    """
    def __init__(self, site_ids, profile=None):
        self.site_ids=tuple(str(x) for x in site_ids)
        if not self.site_ids or len(set(self.site_ids)) != len(self.site_ids):
            raise ValueError('body-function sites must have stable unique identities')
        self.profile=profile or FunctionProfile()
        self.quality=np.ones(len(self.site_ids),dtype=np.float64)
        self.last_time=None
        self.last_load=np.zeros_like(self.quality)
        self.last_stimulation=np.zeros_like(self.quality)

    def advance(self, load, stimulation, now):
        load=np.asarray(load,dtype=np.float64);stimulation=np.asarray(stimulation,dtype=np.float64)
        if load.shape != self.quality.shape or stimulation.shape != load.shape:
            raise ValueError('body-function inputs must retain site identity')
        if not np.isfinite(load).all() or not np.isfinite(stimulation).all() or not math.isfinite(now):
            raise ValueError('nonfinite body-function observation')
        if (load<0).any() or (load>1).any() or (stimulation<0).any() or (stimulation>1).any():
            raise ValueError('body-function drive outside normalized physical reference')
        if self.last_time is not None and now < self.last_time:
            raise ValueError('body-function clock reversed without restoring its state')
        dt=0. if self.last_time is None else now-self.last_time
        p=self.profile
        target=np.clip(1-load+p.positive_fraction*stimulation*(1-load),0,1)
        # Small positive modulation accelerates recovery when headroom exists.
        recovery=p.recovery_seconds/(1+p.positive_fraction*stimulation)
        tau=np.where(target<self.quality,p.onset_seconds,recovery)
        self.quality=target+(self.quality-target)*np.exp(-dt/tau)
        self.last_load=load.copy();self.last_stimulation=stimulation.copy();self.last_time=float(now)
        return self.features()

    def features(self):
        return np.stack((self.quality,self.last_load,self.last_stimulation),-1).astype(np.float32)

    def observe(self, physical):
        """Quantize pressure/shear only; preserve their units and physical audit input.

        Uses fixed-width range blocks, so high pressure is not silently clipped to
        an ordinary touch. Geometry, area, proprioception and other senses stay real.
        """
        values=np.asarray(physical)
        if values.ndim != 2 or len(values) != len(self.site_ids) or values.shape[1]<2:
            raise ValueError('physical skin observation must match the body sites')
        if not np.isfinite(values).all():raise ValueError('nonfinite physical observation')
        result=values.copy()
        p=self.profile
        for i,q in enumerate(self.quality):
            if q >= 1-1e-12:continue
            bits=int(np.clip(round(p.minimum_bits+(p.maximum_bits-p.minimum_bits)*q),p.minimum_bits,p.maximum_bits))
            step=p.pressure_range_pa/(2**bits-1)
            result[i,:2]=np.round(values[i,:2]/step)*step
        return result

    def record(self):
        return dict(contract=CONTRACT,site_ids=list(self.site_ids),time=self.last_time,
                    fields=list(FUNCTION_FIELDS),values=self.features().tolist(),
                    profile=asdict(self.profile),scope='local pressure and shear resolution')

    def state_dict(self):
        return dict(contract=CONTRACT,site_ids=self.site_ids,profile=asdict(self.profile),
                    quality=self.quality.copy(),last_load=self.last_load.copy(),
                    last_stimulation=self.last_stimulation.copy(),last_time=self.last_time)

    def load_state_dict(self,state):
        if state['contract']!=CONTRACT or tuple(state['site_ids'])!=self.site_ids or state['profile']!=asdict(self.profile):
            raise ValueError('body-function identity/profile changed without migration')
        for k in ('quality','last_load','last_stimulation'):
            value=np.asarray(state[k],dtype=np.float64)
            if value.shape!=self.quality.shape or not np.isfinite(value).all() or ((value<0)|(value>1)).any():
                raise ValueError('invalid saved body function')
            setattr(self,k,value.copy())
        self.last_time=state['last_time']
        if self.last_time is not None and not math.isfinite(self.last_time):
            raise ValueError('invalid saved body clock')


def physical_drives(pressure_pa, history, sensitivity, profile=None):
    """Reuse the versioned mechanical cause estimate, without treating it as wellness.

    This repeats skin_mechanics.mechanical_load in numpy because it runs over
    thousands of taxels at every physical substep. The two must agree: whenever
    one changes, change both. The squash comes from the profile for exactly that
    reason, so a profile with a different aversion shape cannot leave this copy
    silently on the old one.

    The default stays PROFILE, and mechanical_v4 does NOT pass its wide-range
    profile here. That is deliberate, not an oversight: this output drives TISSUE
    IMPAIRMENT, not felt pain, and whether impairment should keep resolving above
    6.5x the pressure reference is a separate question from whether pain should.
    Changing it would change how fast sensors degrade on every run.
    """
    from lem.skin_mechanics import PROFILE, aversion_from_load
    p=PROFILE if profile is None else profile
    pressure=np.maximum(np.asarray(pressure_pa,dtype=np.float64),0)
    history=np.asarray(history,dtype=np.float64);g=np.asarray(sensitivity,dtype=np.float64)
    if history.shape != (*pressure.shape,3) or g.shape != (*pressure.shape,2):
        raise ValueError('physical drive layout mismatch')
    peak=np.maximum(pressure,history[...,0])
    load=np.maximum(peak*g[...,0]/p.pressure_reference_pa-1,0)
    load+=history[...,1]*g[...,0]/(p.pressure_reference_pa*p.pressure_exposure_seconds)
    load+=history[...,2]*g[...,0]/p.rubbing_reference_j_m2
    # same squash the torch path applies, selected by the profile so the two
    # cannot drift apart; 'exponential' reproduces the previous -np.expm1(-load)
    load=(-np.expm1(-load) if p.aversion_shape=='exponential'
          else load/(1.0+load))
    stimulation=np.clip(pressure/p.pressure_reference_pa*np.maximum(g[...,1],0),0,1)*(1-load)
    return load,stimulation
