# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Learn actual function on a copied brain while preserving all parent weights.

This bounded side experiment never changes the live learner or crowns a choice
winner. It exercises true modulation, history, and rehearsal with real MIMo input.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .causal_test import _build_model,_limits_for_run
from .migrate_trunk import modules
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.body_layout import extend_limits
from lem.training.checkpoint import CheckpointManager


def run(checkpoint,out,steps=120):
    torch.set_num_threads(2);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    m,args,_,_=_build_model(checkpoint,'cpu',0,intention=True)
    m.load_checkpoint(checkpoint)
    if m.self_state_mode!='learn':raise ValueError('copy learning requires an isolated self-state learner')
    parents={f'{k}.{n}':v.clone() for k,mod in modules(m).items() if k!='self_state' for n,v in mod.state_dict().items()}
    original_self={n:v.clone() for n,v in m.self_state.state_dict().items()}
    extra={id(p) for p in m.self_state.parameters()}
    def preserve_parent(optimizer,_args,_kwargs):
        # Keep autograd available to the old planner, including its wellness
        # derivatives; suppress only old parameter updates at the optimizer boundary.
        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) not in extra:p.grad=None
    hook=m.optimizer.register_step_pre_hook(preserve_parent)
    dev,_=_limits_for_run(checkpoint,m);dev=extend_limits(dev,len(m.handle_names))
    src=SingleEnvSource(scene='crib',vision=True,age=24,run_dir=str(out/'world'),seed=37,
        body_layout=m.body_layout,caregiver_seconds=0,body_function='modulate',contact_profile='mechanical_v2')
    limits=[];losses=[];held=executed=None;start=time.perf_counter()
    try:
        frames=src.reset()
        for i in range(steps):
            result=m.step(frames,dev,train=True,executed_action=executed)
            if not np.isfinite(result.loss):raise ValueError('copy learning became nonfinite')
            if held is None or i%5==0:held=result.motor_for_action.copy()
            gaze=[]
            for sub in range(5):
                frames=src.step(held,result.gaze_out,dev.get('handle_mask'),render=sub==4)
                gaze.append([f.meta.get('gaze_cmd',(0.,0.)) for f in frames])
            executed=m.gaze_to_handle(held,np.asarray(gaze).mean(0))
            limits.append((m.last_self_state['resolution_min'],m.last_self_state['motor_capacity_min']))
            losses.append(dict(step=m.step_count,**m.last_self_state))
        for k,mod in modules(m).items():
            if k=='self_state':continue
            for n,v in mod.state_dict().items():
                if not torch.equal(v,parents[f'{k}.{n}']):raise AssertionError('parent weights changed')
        m.checkpoints=CheckpointManager(m.config,str(out/'checkpoints'))
        parent_meta=json.loads(Path(checkpoint,'meta.json').read_text(encoding='utf-8'))
        phase=dict(parent_meta.get('curriculum_state') or {})
        phase['phase']=dev['phase_name'];phase['phase_step']=int(phase.get('phase_step',0))+steps
        saved=m.save_checkpoint(m.step_count,curriculum_state=phase)
        settings=dict(args,body_function='modulate',self_state='learn',n_envs=1,
            caregiver_mode='off',caregiver_seconds=0,babble_bouts='off',motor_hold=5,action_repeat=5,
            _pilot_entrypoint='mimo_bridge.tools.train_self_copy',_pilot_environment=src.env_kwargs,
            _parent_weights_frozen=True)
        (out/'args.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
        (out/'observations.json').write_text(json.dumps(losses,indent=2,allow_nan=False),encoding='utf-8')
        record=dict(status='completed_copy_learning',checkpoint=saved,parent=str(checkpoint),steps=steps,
            prior_tensors_exact=len(parents),new_self_tensors_updated=sum(not torch.equal(v,original_self[n]) for n,v in m.self_state.state_dict().items()),
            minimum_sensory_function=float(np.asarray(limits)[:,0].min()),minimum_motor_availability=float(np.asarray(limits)[:,1].min()),
            elapsed_seconds=time.perf_counter()-start,last_observation=m.last_self_state,
            scope='Real modulation and isolated new learning. One world, no behavioral generalization or choice claim; never adopted live.')
        (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(record,indent=2));return record
    finally:hook.remove();src.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=120);a=p.parse_args();run(a.checkpoint,a.out,a.steps)
