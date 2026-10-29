# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Exercise a physical caregiver episode past the old reset boundary, without a learner."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from mimo_bridge.env import BridgeEnv
from lem.skin_mechanics import MECHANICAL_CONTRACT


def run(out,seconds=310.):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    env=BridgeEnv(scene='crib',run_dir=str(out/'scene'),vision=False,age=24,
        body_layout='trunk_v1',contact_profile='mechanical_v2',caregiver_mode='responsive',
        caregiver_speech_dir=str(Path('Saved/mimo_bridge/caregiver_speech').resolve()),
        caregiver_record_dir=str(Path('D:/UEMA_Research/external/D/UEMA_recordings')/out.name),
        mobile=True,audio_real=True,gaze_reflex=False,spontaneous_saccades=False,seed=431)
    try:
        f=env.reset();start=f.sim_time;prior=f.sim_time;readings=[];peak=0.;wall=time.perf_counter()
        actuators=[env.model.actuator(n).id for n in ('act:hip_bend','act:hip_twist','act:hip_lean','act:chest_twist','act:chest_lean')]
        n=round(seconds/env.dt)
        for i in range(n):
            ctrl=np.zeros(env.model.nu)
            # Diagnostic variation in physical controls; not claimed as learned behavior.
            for j,a in enumerate(actuators): ctrl[a]=.12*np.sin((i*env.dt)/3.+j)
            f=env.step(ctrl=ctrl,render=False)
            if f.sim_time <= prior or not f.info['physical_state_finite']:
                raise RuntimeError('physical episode lost continuity')
            prior=f.sim_time
            p=f.info['mechanical_contact'];peak=max(peak,p['pressure_peak_pa'])
            if i%1000==0 or i==n-1:
                recording=f.info['caregiver'].get('recording')
                if not recording or recording['error'] or recording['dropped']:
                    raise RuntimeError(f'caregiver recording gap: {recording}')
                readings.append(dict(simulation_seconds=f.sim_time-start,
                    pressure_peak_pa=p['pressure_peak_pa'],pressure_exposure_pa_s=p['pressure_exposure_pa_s'],
                    rubbing_work_j_m2=p['rubbing_work_j_m2'],events=f.info['caregiver']['events'],recording=recording))
                print(json.dumps(readings[-1]),flush=True)
        state=env.save_full_state()
        first=[]
        for _ in range(10): first.append(env.step(ctrl=np.zeros(env.model.nu),render=False).info['mechanical_contact']['history'])
        env.restore_full_state(state)
        second=[]
        for _ in range(10): second.append(env.step(ctrl=np.zeros(env.model.nu),render=False).info['mechanical_contact']['history'])
        np.testing.assert_allclose(first,second,rtol=0,atol=1e-9)
        result=dict(status='passed_continuity',simulated_seconds=f.sim_time-start,physics_steps=n*env.env.frame_skip,
            observation_steps=n,scheduled_resets=0,max_pressure_pa=peak,readings=readings,
            elapsed_computer_seconds=time.perf_counter()-wall,mechanical_snapshot_replay_exact=True,
            contract=MECHANICAL_CONTRACT,scope='Physical continuity, varied controls, contact history, caregiver and disk recording. No model trained; no intention claim.')
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    finally: env.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seconds',type=float,default=310.)
    a=ap.parse_args();run(a.out,a.seconds)
