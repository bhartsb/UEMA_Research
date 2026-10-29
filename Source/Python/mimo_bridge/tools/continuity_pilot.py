# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Train a checkpoint copy with real body, senses, memory, replay and USB recording."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from mimo_bridge.tools.causal_test import _build_model,_limits_for_run
from mimo_bridge.train_mimo import parse_args,env_kwargs_for
from mimo_bridge.datasource import PoolSource
from mimo_bridge.body_layout import extend_limits


def run(checkpoint,out,steps=150,device='cpu',bodies=2,replay_batch=4):
    torch.set_num_threads(2);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    m,args,_,_=_build_model(checkpoint,device,0,intention=True)
    m.load_checkpoint(checkpoint,load_optimizer=True);m.enable_replay(bodies,size_per_env=max(512,steps+5))
    settings=vars(parse_args(['--out',str(out)]));settings.update(args)
    settings['caregiver_record_dir']=str(Path('D:/UEMA_Research/external/D/UEMA_recordings')/out.name)
    dev,_=_limits_for_run(checkpoint,m);dev=extend_limits(dev,len(m.handle_names))
    kw=env_kwargs_for(SimpleNamespace(**settings),dev['phase_name'],str(out/'scenes'))
    source=PoolSource(bodies,kw);times=[];replay=[];max_pressure=0.;before={k:v.detach().clone() for k,v in m.persistent_memory.named_parameters()}
    errors_before=m._nonfinite_steps
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
    try:
        frames=source.reset();held=executed=None
        cycle_start=time.perf_counter()
        for i in range(steps):
            start=time.perf_counter()
            r=m.step(frames,dev,train=True,executed_action=executed,
                     executed_order_id=None if i == 0 else (i-1)//5)
            times.append(time.perf_counter()-start)
            if not np.isfinite(r.loss): raise RuntimeError('nonfinite learning loss')
            if held is None or i%5==0: held=r.motor_for_action.copy()
            gazes=[]
            for sub in range(5):
                frames=source.step(held,r.gaze_out,dev.get('handle_mask'),render=sub==4)
                gazes.append([f.meta.get('gaze_cmd',(0.,0.)) for f in frames])
                max_pressure=max(max_pressure,max(f.meta['mechanical_contact']['pressure_peak_pa'] for f in frames))
            executed=m.gaze_to_handle(held,np.asarray(gazes).mean(0))
            if m.replay.count>=m.ctx_len+3:
                rr=m.replay_update(batch=replay_batch)
                if rr is None or not np.isfinite(rr['replay_loss']): raise RuntimeError('replay did not complete')
                replay.append(rr['replay_loss'])
        changed=[k for k,v in m.persistent_memory.named_parameters() if not torch.equal(before[k],v)]
        required=('input_projection.0.weight','recurrent.weight_ih_l0','readout.weight','outcome.0.weight')
        if not all(k in changed for k in required): raise RuntimeError(f'memory is not learning through every path: {changed}')
        if not replay or m._nonfinite_steps!=errors_before: raise RuntimeError('missing replay or nonfinite update')
        storage=[f.meta.get('caregiver',{}).get('recording') for f in frames]
        if not all(s and not s['error'] and s['dropped']==0 for s in storage): raise RuntimeError(f'recording failed: {storage}')
        result=dict(status='passed_integration',steps=steps,bodies=bodies,replay_batch=replay_batch,device=device,
            simulation_seconds=[f.sim_time for f in frames],replay_updates=len(replay),
            memory_parameters_with_updates=changed,memory=m.last_memory,max_pressure_pa=max_pressure,
            mean_model_step_ms=float(np.mean(times))*1000,nonfinite_updates=0,storage=storage,
            mean_complete_cycle_ms=(time.perf_counter()-cycle_start)*1000/steps,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            scope='Copied-model training, planning, replay and sensing. Pilot weights are not adopted; intention unmeasured.')
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(result,indent=2));return result
    finally: source.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=150);ap.add_argument('--device',default='cpu')
    ap.add_argument('--bodies',type=int,default=2);ap.add_argument('--replay-batch',type=int,default=4)
    a=ap.parse_args();run(a.checkpoint,a.out,a.steps,a.device,a.bodies,a.replay_batch)
