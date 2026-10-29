# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Persist and run declared copied-world experiments, one worker at a time.

This runner never controls the live trainer. A stopped measurement leaves its
attempt intact; a replacement uses a new directory and the same declaration.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from .training_continuation import atomic_json

REPO=Path(__file__).resolve().parents[4]
MODULES={
    'mimo_bridge.tools.developmental_probe',
    'mimo_bridge.tools.stillness_learning_trial',
    'mimo_bridge.tools.self_grounded_dynamics_trial',
    'mimo_bridge.tools.developmental_representation_probe',
}


def read(path):
    def finite(number):
        value=float(number)
        if not math.isfinite(value):raise ValueError('nonfinite queue record')
        return value
    return json.loads(Path(path).read_text(encoding='utf-8'),
        parse_float=finite,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite queue record')))


def schedule_recurring(specification,state,now):
    """Declare a fresh checkpoint test once, coalescing backlog per test policy."""
    policies=specification.get('recurring',[])
    generated=state.setdefault('generated_jobs',[])
    if not policies:return
    continuation=REPO/'Saved/mimo_bridge/training_continuation.json'
    if not continuation.is_file():return
    from .training_continuation import latest_complete
    current=read(continuation);checkpoint=latest_complete(current['active_run'])
    if checkpoint is None:return
    step=int(checkpoint.name.removeprefix('ckpt_step_'))
    for policy in policies:
        digest=hashlib.sha256(json.dumps(policy,sort_keys=True,allow_nan=False).encode()).hexdigest()
        ledger=state.setdefault('policies',{}).setdefault(policy['id'],dict(digest=digest,last_step=None,last_at=None))
        if ledger['digest']!=digest:raise ValueError('Recurring test changed; declare a new policy ID')
        if policy['minimum_new_steps']<1 or policy['minimum_hours']<1:raise ValueError('Unbounded recurring test cadence')
        if any(j.get('policy_id')==policy['id'] and state['jobs'].get(j['id'],{}).get('state') not in ('complete','repair_required') for j in generated):continue
        if ledger['last_step'] is not None and (step-ledger['last_step']<policy['minimum_new_steps'] or now-ledger['last_at']<policy['minimum_hours']*3600):continue
        job=dict(policy['job']);job.update(id=policy['id']+'-step-'+str(step),policy_id=policy['id'],
            checkpoint=str(checkpoint),declared_at=now,
            args=['--checkpoint',str(checkpoint),*job['args']])
        validate_job(job);generated.append(job);ledger.update(last_step=step,last_at=now)


def alive(record):
    try:
        p=psutil.Process(int(record['pid']))
        return p.is_running() and abs(p.create_time()-record['process_created'])<1
    except (psutil.Error,KeyError,ValueError,TypeError):return False


def validate_job(job):
    if job['module'] not in MODULES:raise ValueError('unknown copied-world test module')
    if not job['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in job['id']):
        raise ValueError('invalid experiment identity')
    if not isinstance(job['args'],list) or not all(isinstance(x,str) for x in job['args']):
        raise ValueError('experiment arguments must be separate strings')
    if '--out' in job['args']:raise ValueError('queue owns immutable attempt directories')
    if not all(job.get(k) for k in ('question','decision','scope')):
        raise ValueError('experiment needs a question, decision and scope')
    if not 1<=job.get('max_attempts',2)<=3:raise ValueError('unbounded experiment retries')


def checkpoint_digest(path):
    path=Path(path)
    for name in ('model.safetensors','model_receipt.json','runtime_state.pt'):
        if not (path/name).is_file():raise ValueError('incomplete experiment checkpoint')
    digest=hashlib.sha256()
    for name in ('model.safetensors','model_receipt.json'):
        with (path/name).open('rb') as source:
            for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def poll_attempt(attempt):
    # A brief locked/being-replaced status file cannot make a live worker look
    # absent and launch a second copy. Process identity controls concurrency.
    if alive(attempt):return 'running'
    folder=Path(attempt['out']);status=folder/'status.json';report=folder/'report.json'
    stage=read(status).get('stage') if status.is_file() else None
    if stage=='complete' and report.is_file():
        read(report)  # Syntax and finite constants must be valid before completion.
        attempt['report_sha256']=hashlib.sha256(report.read_bytes()).hexdigest()
        attempt['completed_at']=report.stat().st_mtime
        return 'complete'
    return 'interrupted' if stage not in ('failed',) else 'failed'


def resource_status(storage):
    """Reserve room for production saves and the Windows paging/log volume."""
    import shutil
    record=dict(recording_free_bytes=shutil.disk_usage(storage).free,
        workspace_free_bytes=shutil.disk_usage(REPO).free,
        available_memory_bytes=psutil.virtual_memory().available)
    reasons=[]
    if record['recording_free_bytes']<4*1024**3:reasons.append('Recording drive has less than 4 GiB free')
    if record['workspace_free_bytes']<512*1024**2:reasons.append('Log and paging drive has less than 512 MiB free')
    if record['available_memory_bytes']<3*1024**3:reasons.append('Less than 3 GiB of RAM remains for production and copied worlds')
    record['pause_reason']='; '.join(reasons) or None
    return record


def launch_pause(state_path,now):
    """A dated coordination hold delays new copies, never a running test or trainer."""
    path=Path(state_path).with_name(Path(state_path).stem+'_control.json')
    if not path.exists():return None
    control=read(path)
    if control.get('cmd')!='pause_launches':return None
    until=control.get('until')
    if isinstance(until,bool) or not isinstance(until,(int,float)) or not math.isfinite(until):
        raise ValueError('Measurement launch pause requires a finite expiry')
    if until<=now:return None
    return dict(until=until,reason=str(control.get('reason','Reserved for another copied experiment')),path=str(path))


def tick(manifest,state_path,storage,launch=True):
    specification=read(manifest)
    if specification.get('schema')!=1:raise ValueError('unsupported experiment queue')
    state=read(state_path) if Path(state_path).is_file() else dict(schema=1,jobs={})
    now=time.time();schedule_recurring(specification,state,now)
    jobs=[*specification['jobs'],*state.get('generated_jobs',[])]
    if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('duplicate experiment identity')
    for job in jobs:validate_job(job)
    running=False
    for job in jobs:
        digest=hashlib.sha256(json.dumps(job,sort_keys=True,allow_nan=False).encode()).hexdigest()
        row=state['jobs'].setdefault(job['id'],dict(declaration_sha256=digest,attempts=[],state='queued'))
        if row['declaration_sha256']!=digest:
            raise ValueError('declared experiment changed; use a new experiment ID')
        if job['id'] in specification.get('cancelled_jobs',{}):
            row['state']='cancelled';row['cancel_reason']=specification['cancelled_jobs'][job['id']]
            if row['attempts']:running|=alive(row['attempts'][-1])
            continue
        if row['attempts']:
            try:row['state']=poll_attempt(row['attempts'][-1])
            except (ValueError,OSError) as error:
                row['state']='record_error';row['error']=str(error)
            running|=alive(row['attempts'][-1])
            if row['state']=='running':row.pop('error',None)
        if row['state'] in ('failed','interrupted') and len(row['attempts'])>=job.get('max_attempts',2):
            row['state']='repair_required'
    pause=launch_pause(state_path,now)
    if pause:
        state['launch_pause']=pause
        state['pause_reason']=pause['reason']+'. Existing tests and production continue; the launch hold expires automatically.'
    else:
        state.pop('launch_pause',None)
        state.pop('pause_reason',None)
    if launch and not running and not pause:
        import shutil
        storage=Path(storage).resolve();storage.mkdir(parents=True,exist_ok=True)
        state['resources']=resource_status(storage)
        if state['resources']['pause_reason']:
            state['pause_reason']=state['resources']['pause_reason']+'. Waiting to start another copy; production remains independent.'
        else:
            state.pop('pause_reason',None)
            for job in jobs:
                row=state['jobs'][job['id']]
                if row['state'] not in ('queued','interrupted','failed'):continue
                if any(state['jobs'].get(key,{}).get('state')!='complete' for key in job.get('after',[])):
                    row['waiting_for']=job.get('after',[]);continue
                checkpoint=job.get('checkpoint')
                if checkpoint:row['checkpoint_sha256']=checkpoint_digest(checkpoint)
                attempt_dir=storage/job['id']/('attempt_'+str(len(row['attempts'])+1))
                if attempt_dir.exists():raise ValueError('attempt directory already exists without a queue receipt')
                attempt_dir.parent.mkdir(parents=True,exist_ok=True)
                log_path=attempt_dir.parent/(attempt_dir.name+'.log')
                cmd=[sys.executable,'-u','-m',job['module'],*job['args'],'--out',str(attempt_dir)]
                environment=dict(os.environ,PYTHONPATH=str(REPO/'Source/Python'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                with log_path.open('xb') as log:
                    process=subprocess.Popen(cmd,cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                attempt=dict(pid=process.pid,process_created=psutil.Process(process.pid).create_time(),
                    started_at=now,out=str(attempt_dir),log=str(log_path),command=cmd)
                source=REPO/'Source/Python'/Path(*job['module'].split('.')).with_suffix('.py')
                if source.is_file():attempt['worker_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
                row['attempts'].append(attempt);row['state']='running';running=True;break
    state.update(timestamp=now,controller_pid=os.getpid(),running=running,
        manifest=str(Path(manifest).resolve()),storage=str(Path(storage).resolve()))
    atomic_json(Path(state_path),state)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--storage',type=Path,required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args();a.state.parent.mkdir(parents=True,exist_ok=True)
    # Atomic directory ownership prevents two launchers selecting the same job.
    lock=a.state.with_suffix('.lock');owner=lock/'owner.json'
    if lock.exists():
        record=read(owner) if owner.exists() else {}
        if alive(record):raise RuntimeError('experiment queue already has a controller')
        if not owner.exists():raise RuntimeError('queue lock lacks an owner; inspect before recovery')
        owner.unlink();lock.rmdir()
    lock.mkdir();atomic_json(owner,dict(pid=os.getpid(),process_created=psutil.Process().create_time()))
    try:
        while True:
            try:tick(a.manifest,a.state,a.storage)
            except OSError as error:
                # A temporarily unavailable recording drive/status file must not
                # erase the durable queue or kill its controller.
                print(f'Queue storage unavailable; preserving jobs: {error}',flush=True)
            if not a.watch:break
            time.sleep(15)
    finally:
        owner.unlink(missing_ok=True);lock.rmdir()


if __name__=='__main__':main()
