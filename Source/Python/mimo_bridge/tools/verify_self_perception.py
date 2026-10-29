# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measured preservation and learning-path checks on a parent.

This is an integration verification, not a demonstration of infant development.
Both learners receive the same actual observations and random streams. Observer
learning must leave every pre-existing weight and every issued command unchanged.
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
from .prepare_continuity import native_model


def run(parent,extended,out,steps=8):
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    old,args=native_model(Path(parent));new,_=native_model(Path(extended))
    old.load_checkpoint(parent);new.load_checkpoint(extended)
    dev,_=_limits_for_run(parent,old);dev=extend_limits(dev,len(old.handle_names))
    from mimo_bridge.train_mimo import parse_args,env_kwargs_for
    config=parse_args(['--out',str(out)]);vars(config).update(args)
    kw=env_kwargs_for(config,dev['phase_name'],str(out));kw.update(run_dir=str(out/'world'),body_function='observe',caregiver_record_dir='')
    src=SingleEnvSource(**kw)
    before={k:v.detach().clone() for k,v in new.self_state.named_parameters()}
    times=[];command=None;base_step=old.step_count
    try:
        frames=src.reset()
        for i in range(steps):
            rng=torch.get_rng_state();noise=old.noise_rng.get_state()
            a=old.step(frames,dev,train=True,step_index=base_step+i,executed_action=command)
            torch.set_rng_state(rng);new.noise_rng.set_state(noise)
            started=time.perf_counter()
            b=new.step(frames,dev,train=True,step_index=base_step+i,executed_action=command)
            times.append(time.perf_counter()-started)
            np.testing.assert_array_equal(a.motor_for_action,b.motor_for_action)
            for module,component in modules(old).items():
                for name,v in component.state_dict().items():
                    if not torch.equal(v,modules(new)[module].state_dict()[name]):
                        raise AssertionError(f'existing learning changed: {module}.{name} at step {i}')
            command=a.motor_for_action
            for _ in range(5):frames=src.step(command,a.gaze_out,dev.get('handle_mask'))
        updated=[k for k,v in new.self_state.named_parameters() if not torch.equal(v,before[k])]
        needed=('site_encoder.0.weight','recurrent.weight_ih','transition.weight_ih','future.2.weight','relation_value.weight')
        if not all(k in updated for k in needed):raise AssertionError(f'missing self learning: {updated}')
        result=dict(status='passed',parent=str(parent),extended=str(extended),steps=steps,
            prior_weights_exact_after_each_update=True,issued_commands_exact=True,
            old_tensor_count=sum(len(m.state_dict()) for m in modules(old).values()),
            self_parameters_updated=updated,self_state=new.last_self_state,
            mean_extended_model_step_ms=1000*float(np.mean(times)),simulation_time=frames[0].sim_time,
            scope='Real sensory input; neutral observer preservation and new learning. Intention unproved.')
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(result,indent=2));return result
    finally:src.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('extended')
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=8)
    a=p.parse_args();run(a.parent,a.extended,a.out,a.steps)
