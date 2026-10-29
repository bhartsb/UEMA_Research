# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measure real action effects on sensing function before fitting or licensing choice.

All arms restore the complete physical state. Outcomes stay external to the learner.
This uses an engineering modulation profile, not human pain/injury thresholds.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from mimo_bridge.datasource import SingleEnvSource


def run(out,worlds=3,seconds=1.):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for seed in range(worlds):
        src=SingleEnvSource(scene='crib',vision=False,age=24,seed=seed,run_dir=str(out/f'world_{seed}'),
            body_layout='trunk_v1',caregiver_seconds=0,body_function='modulate',contact_profile='mechanical_v2')
        try:
            f=src.reset()[0];n=len(f.bones)
            held=np.zeros((1,n,9));held[0,:,:3]=f.bones[:,4:7];held[0,:,3:7]=f.bones[:,:4];held[...,8]=.3
            for _ in range(100):f=src.step(held)[0]
            # Target the achieved posture after settling, retaining true xyzw layout.
            held[0,:,:3]=f.bones[:,4:7];held[0,:,3:7]=f.bones[:,:4]
            rng=np.random.default_rng(seed+328)
            candidates=[held.copy()]
            for _ in range(5):
                c=held.copy();q=c[...,3:7]+rng.normal(0,.1,c[...,3:7].shape)
                c[...,3:7]=q/np.linalg.norm(q,axis=-1,keepdims=True);candidates.append(c)
            state=src.save_full_state();trajectories=[];bones=[]
            for c in candidates+[candidates[0]]:
                src.restore_full_state(state);trace=[];pose=[]
                for _ in range(round(seconds/src.env.dt)):
                    f=src.step(c)[0]
                    trace.append(np.asarray(f.meta['body_function']['values'])[:,0]);pose.append(f.bones.copy())
                trajectories.append(np.asarray(trace));bones.append(np.asarray(pose))
            np.testing.assert_array_equal(trajectories[0],trajectories[-1])
            np.testing.assert_array_equal(bones[0],bones[-1])
            # Compare area-independent local functioning, not a hand-assigned touch reward.
            outcomes=[float(q[-1].min()) for q in trajectories[:-1]]
            rows.append(dict(world=seed,minimum_resolution_by_command=outcomes,
                spread=max(outcomes)-min(outcomes),identical_command_replay_error=0.,
                pressure_profile=src.env.body_function.record()['profile']))
        finally:src.close()
    result=dict(worlds=rows,horizon_seconds=seconds,commands_per_world=6,
        scope='Physical reachability and replay only; no fitted predictor or intention verdict.')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--worlds',type=int,default=3)
    p.add_argument('--seconds',type=float,default=1.)
    a=p.parse_args();run(a.out,a.worlds,a.seconds)
