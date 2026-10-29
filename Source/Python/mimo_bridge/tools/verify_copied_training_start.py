# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Exercise the actual copied learner load, Adam migration and two native updates.

This is an instrument startup check. It neither saves trained weights nor changes
a production learner, and it cannot provide an experimental behavior verdict.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mimo_bridge.train_mimo import build_learner,env_kwargs_for
from mimo_bridge.datasource import SingleEnvSource
from .self_recovery_experiment import configuration
from .causal_test import _limits_for_run
from .caregiver_learning_trial import verify_loaded,Orders
from .training_continuation import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--device',default='cpu')
    p.add_argument('--production-pid',type=int);p.add_argument('--production-stats',type=Path)
    p.add_argument('--gpu-reserve-mib',type=int,default=512)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    if not 2<=args.steps<=100:raise ValueError('Startup benchmark is bounded to 2..100 updates')
    a=configuration(args.checkpoint,args.out)
    a.device=args.device
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.25,device=args.device)
    production=None
    if args.production_pid:
        import psutil
        production=psutil.Process(args.production_pid);production_created=production.create_time()
    def sample():
        if production is not None and (not production.is_running() or production.create_time()!=production_created):
            raise RuntimeError('Production process changed; copied benchmark stopped')
        value=dict(timestamp=time.time())
        if args.device.startswith('cuda'):
            # On Windows WDDM, CUDA mem_get_info can report a process-visible
            # budget that is much larger than physical free device memory.
            index=torch.cuda.device(args.device).idx
            whole=subprocess.run(['nvidia-smi',f'--id={index}',
                '--query-gpu=memory.free,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            free_mib,total_mib,used_mib,utilization=[float(x.strip()) for x in whole.stdout.strip().split(',')]
            free=int(free_mib*1024**2);total=int(total_mib*1024**2)
            value.update(gpu_free_bytes=free,gpu_total_bytes=total,
                gpu_used_bytes=int(used_mib*1024**2),gpu_utilization_percent=utilization,
                device_memory_source='nvidia-smi physical device',
                copy_allocated_bytes=torch.cuda.memory_allocated(args.device),
                copy_reserved_bytes=torch.cuda.memory_reserved(args.device),
                copy_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device))
            if free<args.gpu_reserve_mib*1024**2:
                atomic_json(args.out/'resource_stop.json',dict(value,reason='GPU physical free reserve reached',
                    reserve_mib=args.gpu_reserve_mib,capability_pass=False))
                raise RuntimeError('GPU free reserve reached; copied benchmark stopped')
        if args.production_stats and args.production_stats.exists():
            with args.production_stats.open('rb') as f:
                f.seek(max(0,args.production_stats.stat().st_size-100000));lines=f.read().splitlines()
            for line in reversed(lines):
                try:value['production_last_stats']=json.loads(line);break
                except (ValueError,UnicodeError):pass
        return value
    resources=[sample()];started=time.time()
    model=build_learner(a,str(args.out/'unused_checkpoints'))
    loaded=model.load_checkpoint(str(args.checkpoint),load_optimizer=True)
    verify_loaded(model,loaded,args.checkpoint,True)
    histories=len(model.optimizer.state)
    dev,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
    source=SingleEnvSource(**env_kwargs_for(a,dev['phase_name'],str(args.out/'world')))
    try:
        model.reset_transition_state();frames=source.reset();orders=Orders(a,64231);losses=[];durations=[]
        resources.append(sample())
        for tick in range(args.steps):
            begin=time.time()
            limits=orders.limits(dev,tick,source.env.dt*a.action_repeat,True)
            result=model.step(frames,limits,train=True,step_index=model.step_count,
                executed_action=orders.executed,executed_order_id=orders.order or None)
            if not np.isfinite(result.loss):raise ValueError('Nonfinite native copied update')
            if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
            predicted=time.time()
            losses.append(float(result.loss));frames=orders.advance(model,result,source,tick,True)
            durations.append(dict(update_seconds=predicted-begin,physics_render_seconds=time.time()-predicted))
            if tick%10==0:
                resources.append(sample());atomic_json(args.out/'progress.json',dict(updates=tick+1,resources=resources,durations=durations))
        resources.append(sample())
        atomic_json(args.out/'report.json',dict(checkpoint=str(args.checkpoint),native_constructor=True,
            optimizer_histories_verified_before_updates=histories,retained_histories_after_updates=len(model.optimizer.state),
            executed_updates=args.steps,losses=losses,limits=dev,limits_source=origin,capability_pass=False,
            device=args.device,wall_seconds=time.time()-started,resources=resources,step_durations=durations,
            scope='Copied training startup verified; no trained weights saved or production process changed.'))
    finally:source.close()


if __name__=='__main__':main()
