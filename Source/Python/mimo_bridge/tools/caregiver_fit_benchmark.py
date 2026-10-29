# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Bounded native CUDA fitting only; no comparison result or live promotion."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import psutil

from .caregiver_fit_workers import run_fits,study_ownership,resources,process_matches
from .caregiver_learning_trial import write,settings
from .causal_test import _limits_for_run
from .development_dashboard import last_training_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--speech',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--production-pid',type=int,required=True)
    parser.add_argument('--production-stats',type=Path,required=True);parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args()
    if not 50<=args.steps<=100:raise ValueError('Native fitting benchmark is bounded to 50..100 updates per copy')
    args.out.mkdir(parents=True,exist_ok=False)
    production=dict(pid=args.production_pid,process_created=psutil.Process(args.production_pid).create_time())
    baseline=last_training_record(args.production_stats)
    config=settings(args.checkpoint,args.out,args.speech,'cuda:0',24031)
    limits,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
    if not limits.get('phase_name'):raise ValueError('No measured parent body limits')
    specification=SimpleNamespace(out=args.out,checkpoint=args.checkpoint,speech=args.speech,worlds=2,seed=24031,
        train_steps=args.steps,device='cuda:0',fit_workers=2,fit_gpu_reserve_mib=1024,fit_worker_budget_mib=1024,
        fit_ram_reserve_gib=4.,fit_startup_updates=50)
    started=time.time();samples=[]
    write(args.out/'declaration.json',dict(instrument_benchmark=True,capability_pass=False,
        parent=str(args.checkpoint.resolve()),steps_per_copy=args.steps,worlds=2,seeds=[24031,24032],
        device='cuda:0',amp=vars(config).get('amp'),body_limits=limits,limits_source=origin,
        startup='Second admission only after the first copy reports at least 50 updates and physical headroom permits.',
        production=production,baseline_production=baseline,maximum_seconds=1200,
        scope='Two responsive fitting copies only. No full behavioral battery, scientific comparison, or production change.'))
    def sample(snapshot,workers):
        row=last_training_record(args.production_stats)
        records=[]
        for worker in workers:
            folder=Path(worker.get('segment_out',worker['out']));progress=folder/'status.json';device=folder/'worker_resources.json'
            records.append(dict(identity=worker,progress=json.loads(progress.read_text()) if progress.is_file() else {},
                resources=json.loads(device.read_text()) if device.is_file() else {}))
        samples.append(dict(snapshot,workers=records,production={k:row.get(k) for k in ('step','timestamp','nan_steps')}))
        write(args.out/'resource_samples.json',samples)
        if not process_matches(production):raise RuntimeError('Production process changed; stop copied benchmark')
        if row.get('nan_steps',0)>baseline.get('nan_steps',0):raise RuntimeError('Production reported a new nonfinite step')
        if time.time()-started>1200:raise RuntimeError('Bounded benchmark time reached; copied partials remain preserved')
    completed=False;error=None
    try:
        with study_ownership(args.out):
            checkpoints,paths=run_fits(specification,limits,[1,0],
                task_keys={'responsive_0','responsive_1'},on_poll=sample)
        completed=True
    except Exception as failure:
        error=str(failure);raise
    finally:
        final=last_training_record(args.production_stats)
        report=dict(complete=completed,capability_pass=False,error=error,wall_seconds=time.time()-started,
            resource_samples=len(samples),maximum_active_workers=max((len(s['workers']) for s in samples),default=0),
            maximum_workers_past_startup=max((sum(w['progress'].get('step',0)>0 for w in s['workers']) for s in samples),default=0),
            minimum_free_gpu_bytes=min((s['gpu_free_bytes'] for s in samples),default=None),
            peak_device_used_bytes=max((s['gpu_used_bytes'] for s in samples),default=None),
            minimum_free_ram_bytes=min((s['ram_available_bytes'] for s in samples),default=None),
            production_start={k:baseline.get(k) for k in ('step','timestamp','nan_steps')},
            production_end={k:final.get(k) for k in ('step','timestamp','nan_steps')},
            current_resources=resources('cuda:0'),
            scope='Resource and native startup/fit/save benchmark. Original caregiver comparison remains unchanged and unfinished.')
        if completed:
            report['fits']={f'{arm}_{index}':json.loads((folder/'fit.json').read_text()) for (arm,index),folder in paths.items()}
        write(args.out/'benchmark_report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
