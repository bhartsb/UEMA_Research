# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Matched body branches under a fixed contact fixture; this is not a learned skill test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mimo_bridge.env import BridgeEnv
from lem.model.contact_wellness import contact_response
from tokenizer.modalities.touch import _FIELD_NORMS


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for seed in (501,502,503):
        env=BridgeEnv(scene='crib',vision=False,age=24,body_layout='trunk_v1',
            contact_profile='mechanical_v2',gaze_reflex=False,spontaneous_saccades=False,
            run_dir=str(out/f'scene_{seed}'),seed=seed)
        try:
            f=env.reset();target=env._target_id;env._target_id=None
            # A world-fixed 25-mm-radius sphere touching the hand. Moving the
            # hand can escape it; no body-attached force or movement-label reward.
            center=env.data.xpos[env.model.body('right_hand').id].copy()
            # Locate actual collision surfaces before imposing a 0.5-mm preload.
            # A center-to-center guess previously produced centimetres of overlap
            # and an invalid impulse; that failed fixture is retained in v1.
            sphere=env.model.geom('target_geom').id
            hand=env.model.body('right_hand').id
            hand_geoms=np.flatnonzero(env.model.geom_bodyid==hand)
            lo,hi=0.,.15
            for _ in range(40):
                offset=(lo+hi)/2
                env.data.mocap_pos[target]=center+np.array([0.,0.,offset])
                env._mj.mj_forward(env.model,env.data)
                gap=min(env._mj.mj_geomDistance(env.model,env.data,sphere,int(g),1.,None) for g in hand_geoms)
                if gap < -.0005: lo=offset
                else: hi=offset
            snapshot=env.save_full_state()
            controls=[('zero',np.zeros(env.model.nu))]
            for name in ('right_shoulder_horizontal','right_shoulder_abduction','right_elbow','right_wrist_flexion'):
                for sign in (-1,1):
                    c=np.zeros(env.model.nu);c[env.model.actuator('act:'+name).id]=sign*.7
                    controls.append((f'{name}_{sign:+d}',c))
            controls.append(('zero_replay',controls[0][1]))
            trials=[]
            for name,ctrl in controls:
                env.restore_full_state(snapshot);aversion=[];force=pressure=0.
                for i in range(50):
                    f=env.step(ctrl=ctrl,render=False)
                    if not f.info['physical_state_finite']: raise RuntimeError('nonfinite physical branch')
                    r=contact_response(torch.tensor(f.patches[:,:8]/_FIELD_NORMS[:8]).unsqueeze(0),
                        profile='mechanical_v2',history=torch.tensor(f.info['mechanical_contact']['history']).unsqueeze(0))
                    aversion.append(float(r['pain']))
                    pressure=max(pressure,float(f.patches[:,0].max()))
                    force=max(force,float((f.patches[:,0]*f.patches[:,3]*1e-6).sum()))
                trials.append(dict(command=name,mean_aversion=float(np.mean(aversion)),
                    final_aversion=aversion[-1],maximum_total_normal_force_n=force,
                    maximum_patch_pressure_pa=pressure))
            replay=abs(trials[0]['mean_aversion']-trials[-1]['mean_aversion'])
            if replay>1e-7: raise RuntimeError(f'branch history was not restored: {replay}')
            spread=max(t['mean_aversion'] for t in trials[:-1])-min(t['mean_aversion'] for t in trials[:-1])
            rows.append(dict(seed=seed,initial_minimum_surface_gap_m=float(gap),trials=trials,repeat_error=replay,action_effect_spread=spread))
            print(json.dumps(rows[-1]),flush=True)
        finally: env.close()
    result=dict(status='passed' if all(r['action_effect_spread']>.01 for r in rows) else 'failed_action_spread',
        criterion='Same-state branches replay within 1e-7 and aversion spread exceeds .01 on a 0-to-1 engineered scale in every world',
        cases=rows,scope='The body can change its contact outcome under a fixed diagnostic fixture. No learned action selection or biological pain calibration.')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    r=run(a.out);raise SystemExit(0 if r['status']=='passed' else 1)
