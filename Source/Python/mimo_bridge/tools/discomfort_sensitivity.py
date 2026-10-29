# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Expose the consequences of uncertain pressure and engineered duration settings."""
from dataclasses import replace,asdict
import json
from pathlib import Path
import argparse
import numpy as np
import torch
from lem.skin_mechanics import PROFILE,MechanicalExposure,mechanical_load


def run(out):
    profiles={'reference':PROFILE,
        'child_group_mean_minus_one_sd':replace(PROFILE,pressure_reference_pa=144300.),
        'child_group_mean_plus_one_sd':replace(PROFILE,pressure_reference_pa=294500.),
        'duration_half':replace(PROFILE,pressure_exposure_seconds=30.,recovery_seconds=5.),
        'duration_double':replace(PROFILE,pressure_exposure_seconds=120.,recovery_seconds=20.),
        'rubbing_half_scale':replace(PROFILE,rubbing_reference_j_m2=1097.),
        'rubbing_double_scale':replace(PROFILE,rubbing_reference_j_m2=4388.)}
    cases=[('standing_support',14710.,0.,0.,60.),('moderate_sustained_pressure',100000.,0.,0.,60.),
           ('concentrated_brief_load',438800.,0.,0.,.01),('loaded_rubbing',100000.,50000.,100.,1.)]
    rows=[]
    for name,p in profiles.items():
        for label,pressure,shear,slip,duration in cases:
            e=MechanicalExposure(1,p);fields=np.zeros((1,8));fields[0,:4]=pressure,shear,slip,100.
            h=e.update(fields,duration)
            before=float(mechanical_load(torch.tensor([pressure]),torch.tensor(h),torch.ones(1),p))
            h=e.update(np.zeros_like(fields),10.)
            after=float(mechanical_load(torch.zeros(1),torch.tensor(h),torch.ones(1),p))
            rows.append(dict(profile=name,situation=label,duration_s=duration,aversion_during=before,aversion_after_10s_unloaded=after))
    assert all(r['aversion_during']==0 for r in rows if r['situation']=='standing_support')
    assert all(r['aversion_after_10s_unloaded']<r['aversion_during'] for r in rows if r['situation']!='standing_support')
    result=dict(status='passed_controls',profiles={k:asdict(v) for k,v in profiles.items()},rows=rows,
        interpretation='0 means no modeled aversion; 1 is its asymptote. SD variants are sensitivity cases, not confidence limits. Duration, recovery and rubbing scales are engineering assumptions, not measured toddler thresholds.')
    Path(out).write_text(json.dumps(result,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
