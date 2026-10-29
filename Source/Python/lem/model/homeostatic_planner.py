# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Compare complete predicted self-state trajectories, preserving local deficits.

This is a sequence comparator over learned forecasts. It is not a reward-trained
motor policy. Callers own candidate generation, actual command smoothing and units.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class FunctionCriterion:
    minimum_resolution: float
    maximum_uncertainty: float
    calibration: str
    minimum_capacity: float | None = None
    def __post_init__(self):
        if (not (0<self.minimum_resolution<=1) or not math.isfinite(self.maximum_uncertainty)
                or self.maximum_uncertainty<=0 or not self.calibration.strip()):
            raise ValueError('homeostatic choice requires calibrated physical bounds and uncertainty')
        if self.minimum_capacity is not None and not 0<self.minimum_capacity<=1:
            raise ValueError('invalid calibrated actuator availability')


def choose_sequence(mean,variance,valid,durations,criterion,kind=None):
    """mean/variance [body,candidate,time,site,field]; one physical site counted once.

    Returns -1 when evidence cannot support choice, never an accidental action-zero.
    Resolution and actual actuator availability have separately calibrated bounds;
    load/stimulation remain perceptual context rather than assigned pleasure.
    """
    if mean.ndim!=5 or mean.shape!=variance.shape or durations.shape!=mean.shape[:3]:
        raise ValueError('self-state trajectory layout mismatch')
    b,k,t,n,_=mean.shape
    if min(b,k,t,n)==0 or valid.shape!=(b,n) or (durations<=0).any() or not torch.isfinite(durations).all():
        raise ValueError('invalid sites or duration')
    finite=torch.isfinite(mean).all((-1,-2,-3)) & torch.isfinite(variance).all((-1,-2,-3))
    finite=finite & (variance>=0).all((-1,-2,-3))
    finite=finite & ((mean>=0)&(mean<=1)).all((-1,-2,-3))
    mask=valid[:,None,None,:]
    uncertainty=variance[...,0].sqrt().masked_fill(~mask,0).amax((-1,-2))
    eligible=finite & (uncertainty<=criterion.maximum_uncertainty) & valid.any(-1)[:,None]
    reference=torch.full((b,n),criterion.minimum_resolution,device=mean.device)
    if kind is not None and ((kind==1)&valid).any():
        if criterion.minimum_capacity is None:raise ValueError('motor functioning requires its own calibrated range')
        reference=torch.where(kind==1,criterion.minimum_capacity,reference)
    deficit=(reference[:,None,None,:]-mean[...,0]).clamp_min(0).masked_fill(~mask,0)
    # Worst local loss prevents a healthy patch count hiding one impaired site.
    cost=(deficit.amax(-1)*durations).sum(-1)
    cost=cost.masked_fill(~eligible,float('inf'))
    best=cost.argmin(-1);best=torch.where(eligible.any(-1),best,torch.full_like(best,-1))
    return dict(index=best,cost=cost,eligible=eligible,uncertainty=uncertainty)


def evaluate_candidates(module,percept,executed,dt,valid,criterion):
    """All candidates start from identical beliefs; no imagined state leaks to a sibling."""
    if executed.ndim!=4:raise ValueError('candidate commands must be [body,candidate,time,command]')
    b,k,t,a=executed.shape
    if dt.shape!=(b,k,t):raise ValueError('each imagined command needs its actual duration')
    branch={key:value[:,None].expand(b,k,*value.shape[1:]).reshape(b*k,*value.shape[1:])
            for key,value in percept.items()}
    mean,var=module.forecast(branch,executed.reshape(b*k,t,a),dt.reshape(b*k,t))
    mean=mean.reshape(b,k,t,*mean.shape[2:]);var=var.reshape_as(mean)
    result=choose_sequence(mean,var,valid,dt,criterion,percept.get('kind'))
    result.update(mean=mean,variance=var)
    return result
