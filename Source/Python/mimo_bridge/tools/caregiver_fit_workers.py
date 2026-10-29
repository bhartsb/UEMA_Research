# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Resource-bounded, process-isolated fitting for the unchanged caregiver trial.

The coordinator alone publishes completed_fits.json. A worker owns one unique
attempt and never fits a yoked condition before its recorded donor is complete.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


class CopyResourcesUnavailable(RuntimeError):
    pass


def read_live_json(path, previous=None):
    """Read optional worker telemetry without stopping a learning process.

    Windows can briefly deny a read while the writer atomically replaces a
    status file. A stale display must never authorize another worker. Required
    checkpoint and completion receipts deliberately do not use this fallback.
    """
    error=None
    for attempt in range(3):
        try:
            value=json.loads(Path(path).read_text(encoding='utf-8'))
            if not isinstance(value,dict):raise ValueError('Expected a telemetry object')
            return value,True,None
        except (OSError,ValueError) as caught:
            error=f'{type(caught).__name__}: {caught}'
            if attempt<2:time.sleep(.025)
    return dict(previous or {}),False,error


def save_allowance(out):
    """Conservative copied-save budget from completed/partial checkpoint receipts.

    With no copied measurement, reserve 256 MiB per save. Two such allowances
    protect a concurrent pending save and this worker's last exact partial.
    """
    largest=0
    for name in ('fit.json','paused_fit.json'):
        for receipt in Path(out).glob(f'**/{name}'):
            data=json.loads(receipt.read_text())
            if not data.get('checkpoint'):continue
            checkpoint=Path(data['checkpoint'])
            if checkpoint.is_dir():
                largest=max(largest,sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()))
    return max(256*1024**2,2*largest+64*1024**2)


def pause_backoff(previous,completed,now):
    """Only actual additional updates reset a repeated-pause streak."""
    advanced=completed>previous.get('completed_updates',0)
    streak=1 if advanced else previous.get('streak',0)+1
    return dict(completed_updates=completed,streak=streak,
        next_admission_at=now+min(300,15*2**min(streak-1,5)),last_pause_at=now)


def resources(device):
    import psutil
    result=dict(timestamp=time.time(),ram_available_bytes=psutil.virtual_memory().available)
    if str(device).startswith('cuda'):
        index=str(device).split(':')[-1] if ':' in str(device) else '0'
        answer=subprocess.run(['nvidia-smi',f'--id={index}',
            '--query-gpu=memory.free,memory.total,memory.used',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        free,total,used=[float(value.strip()) for value in answer.stdout.strip().split(',')]
        result.update(gpu_free_bytes=int(free*1024**2),gpu_total_bytes=int(total*1024**2),
            gpu_used_bytes=int(used*1024**2),gpu_source='nvidia-smi physical device')
    return result


def admission(snapshot,active,*,reserve_mib=1024,worker_mib=1024,ram_reserve_gib=4.,worker_ram_gib=1.5):
    """Use measured physical free memory; do not double-book startup headroom."""
    if any(not row.get('startup_measured') for row in active):
        return False,'Waiting for the first copied worker to finish its measured startup'
    if snapshot['ram_available_bytes']<(ram_reserve_gib+worker_ram_gib)*1024**3:
        return False,'Preserving the production RAM reserve'
    if snapshot.get('gpu_free_bytes',float('inf'))<(reserve_mib+worker_mib)*1024**2:
        return False,'Preserving the production GPU reserve; using at most the current workers'
    return True,None


def ready_tasks(worlds,completed,active,donors):
    """Prefer closing already-started independent donor pairs, deterministically."""
    ready=[]
    for first in range(0,worlds,2):
        for arm in ('yoked','responsive'):
            for index in (first,first+1):
                key=f'{arm}_{index}'
                if key in completed or key in active:continue
                if arm=='yoked' and f'responsive_{donors[index]}' not in completed:continue
                ready.append((arm,index))
    return ready


def process_matches(record):
    import psutil
    try:
        process=psutil.Process(int(record['pid']))
        return process.is_running() and abs(process.create_time()-float(record['process_created']))<.01
    except (psutil.Error,KeyError,ValueError,TypeError):return False


def assert_no_active_workers(out):
    """Fail closed for serial resumes too, including a child absent from its journal."""
    import psutil
    owner=dict(pid=os.getpid(),process_created=psutil.Process().create_time())
    out=Path(out);journal=out/'fit_coordinator.json'
    previous=json.loads(journal.read_text()) if journal.exists() else {}
    if process_matches(previous.get('owner',{})) and previous['owner']!=owner:
        raise RuntimeError('Another caregiver fitting coordinator is active')
    records=list(previous.get('active',[]))
    for pattern in ('responsive_*/worker_process.json','yoked_*/worker_process.json',
                    'responsive_*/attempt_*/worker_process.json','yoked_*/attempt_*/worker_process.json',
                    'responsive_*/segments/*/worker_process.json','yoked_*/segments/*/worker_process.json',
                    'responsive_*/attempt_*/segments/*/worker_process.json','yoked_*/attempt_*/segments/*/worker_process.json'):
        records.extend(json.loads(path.read_text()) for path in out.glob(pattern))
    if any(process_matches(row) for row in records):
        raise RuntimeError('Prior copied workers are still saving; resume after their recorded processes exit')


@contextmanager
def study_ownership(out):
    """OS-held ownership dies with the coordinator; the append-only attempts survive."""
    lock=Path(out)/'study_owner.lock'
    with lock.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'\0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another caregiver study process owns this output directory') from error
        try:
            assert_no_active_workers(out)
            yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def guard_for(payload):
    from .caregiver_learning_trial import write
    folder=Path(payload.get('segment_out',payload['out']));last={}
    def before_save(completed):
        allowance=payload.get('save_allowance_bytes',256*1024**2)
        # Another process can consume disk after our early admission. Retain
        # the exact in-memory boundary and wait rather than spend the protected
        # reserve or throw away the unsaved updates. No learning occurs here.
        while shutil.disk_usage(folder).free<4*1024**3+allowance:
            write(folder/'status.json',dict(stage='waiting_to_save',timestamp=time.time(),pid=os.getpid(),
                completed_updates=completed,reason='Exact copied state remains in memory until its checkpoint fits above the recording reserve',
                required_free_bytes=4*1024**3+allowance,free_bytes=shutil.disk_usage(folder).free))
            time.sleep(5)
    def guard(model,step):
        if step%int(payload.get('check_every_updates',10)):return last
        if (folder/'stop_requested.json').exists() or not process_matches(payload['coordinator']):
            raise CopyResourcesUnavailable('Coordinator stopped; preserve this copied attempt')
        snapshot=resources(payload['device']);last.clear();last.update(snapshot)
        import psutil
        last['copy_process_rss_bytes']=psutil.Process().memory_info().rss
        if str(payload['device']).startswith('cuda'):
            import torch
            last.update(copy_cuda_allocated_bytes=torch.cuda.memory_allocated(payload['device']),
                copy_cuda_reserved_bytes=torch.cuda.memory_reserved(payload['device']),
                copy_cuda_peak_bytes=torch.cuda.max_memory_allocated(payload['device']))
        write(folder/'worker_resources.json',dict(last,completed_updates=step))
        if snapshot.get('gpu_free_bytes',float('inf'))<payload['gpu_reserve_mib']*1024**2:
            raise CopyResourcesUnavailable('Physical GPU reserve reached')
        if snapshot['ram_available_bytes']<payload['ram_reserve_gib']*1024**3:
            raise CopyResourcesUnavailable('Production RAM reserve reached')
        allowance=payload.get('save_allowance_bytes',256*1024**2)
        if shutil.disk_usage(folder).free<4*1024**3+2*allowance:
            raise CopyResourcesUnavailable('Preserving recording reserve plus the next copied checkpoint save')
        return last
    guard.before_save=before_save
    return guard


def worker(path):
    import psutil
    from .caregiver_learning_trial import fit,settings,write
    payload=json.loads(Path(path).read_text());folder=Path(payload['out'])
    segment=Path(payload.get('segment_out',payload['out']))
    # Publish before any model/GPU allocation. If the coordinator died between
    # Popen and its journal write, every later serial/parallel resume sees us.
    write(segment/'worker_process.json',dict(pid=os.getpid(),process_created=psutil.Process().create_time(),
        coordinator=payload['coordinator'],started=time.time()))
    try:
        if not process_matches(payload['coordinator']):
            raise CopyResourcesUnavailable('Coordinator stopped before copied startup; no learning started')
        import torch
        torch.set_num_threads(1)
        if payload['device'].startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(.25,device=payload['device'])
        args=settings(Path(payload['parent']),folder,Path(payload['speech']),payload['device'],payload['seed'])
        checkpoint=fit(args,Path(payload['parent']),folder,payload['seed'],payload['steps'],payload['limits'],
            Path(payload['tape']) if payload['tape'] else None,progress_out=folder,
            step_guard=guard_for(payload),resume_checkpoint=payload.get('resume_checkpoint'),segment_out=segment)
        write(segment/'worker_result.json',dict(state='complete',checkpoint=str(checkpoint),timestamp=time.time()))
        return 0
    except CopyResourcesUnavailable as error:
        path=folder/'paused_fit.json';partial=json.loads(path.read_text()) if path.exists() else {}
        write(segment/'worker_result.json',dict(state='resource_paused',error=str(error),timestamp=time.time(),
            completed_updates=partial.get('completed_updates',0),checkpoint=partial.get('checkpoint'),
            resumable_contract=partial.get('resumable_contract')))
        return 75
    except Exception as error:
        import traceback
        write(segment/'worker_result.json',dict(state='failed',error=str(error),traceback=traceback.format_exc(),timestamp=time.time()))
        raise


def run_fits(args,limits,donors,*,task_keys=None,on_poll=None):
    import psutil
    from .caregiver_learning_trial import completed_fit,next_attempt,settings,write
    from . import caregiver_fit_continuation as continuation
    owner=dict(pid=os.getpid(),process_created=psutil.Process().create_time())
    journal=args.out/'fit_coordinator.json'
    assert_no_active_workers(args.out)
    previous=json.loads(journal.read_text()) if journal.exists() else {}
    completed={};checkpoints={};fit_paths={};active={};capacity=args.fit_workers;pauses=previous.get('resource_backoff',{})
    worker_ram_gib=max(1.5,float(previous.get('worker_ram_budget_gib',1.5)))
    allowance=save_allowance(args.out)
    expected={f'{arm}_{index}' for arm in ('responsive','yoked') for index in range(args.worlds)}
    if task_keys is not None:
        if not task_keys or not set(task_keys).issubset(expected):raise ValueError('Invalid bounded fitting task subset')
        expected=set(task_keys)
    for arm in ('responsive','yoked'):
        for index in range(args.worlds):
            if f'{arm}_{index}' not in expected:continue
            found=completed_fit(args.out/f'{arm}_{index}',args.train_steps)
            if found:
                folder,receipt=found;completed[f'{arm}_{index}']=folder
                checkpoints[arm,index]=Path(receipt['checkpoint']);fit_paths[arm,index]=folder
    for index in range(args.worlds):
        if f'yoked_{index}' not in completed:continue
        donor=completed.get(f'responsive_{donors[index]}')
        if donor is None:raise ValueError('Completed yoked fit has no preserved responsive donor')
        expected_dose=json.loads((donor/'fit.json').read_text())['source_utterances']
        actual=json.loads((completed[f'yoked_{index}']/'fit.json').read_text())['source_utterances']
        if actual!=expected_dose:raise ValueError('Preserved yoked speech dose differs from donor')
    def publish():
        write(args.out/'completed_fits.json',{key:str(folder/'fit.json') for key,folder in completed.items()})
        write(journal,dict(owner=owner,requested_workers=args.fit_workers,effective_workers=capacity,
            active=[row['receipt'] for row in active.values()],complete=list(completed),timestamp=time.time(),
            resource_backoff=pauses,worker_ram_budget_gib=worker_ram_gib,save_allowance_bytes=allowance,
            unchanged='Original declaration, checkpoint, CUDA/AMP, seeds, controls and update budgets'))
    publish()
    try:
        while len(completed)<len(expected):
            for key,row in list(active.items()):
                code=row['process'].poll()
                if code is None:continue
                row['log'].close();del active[key]
                result_path=row['segment']/'worker_result.json'
                result=json.loads(result_path.read_text()) if result_path.exists() else {}
                if code==75 and result.get('state')=='resource_paused':
                    capacity=1;pauses[key]=pause_backoff(pauses.get(key,{}),
                        int(result.get('completed_updates',row['resumed_after'])),time.time())
                    publish()
                    continue
                if code!=0 or result.get('state')!='complete':
                    raise RuntimeError(f'Copied worker {key} failed: {result.get("error",code)}')
                found=completed_fit(args.out/key,args.train_steps)
                if not found:raise RuntimeError('Worker claimed completion without a valid full-budget fit')
                folder,receipt=found;arm,index=row['arm'],row['index']
                if arm=='yoked':
                    donor=json.loads((completed[f'responsive_{donors[index]}']/'fit.json').read_text())
                    if receipt['source_utterances']!=donor['source_utterances']:
                        raise ValueError('yoked source speech dose/timing differs from its donor')
                completed[key]=folder;checkpoints[arm,index]=Path(receipt['checkpoint']);fit_paths[arm,index]=folder;publish()
                allowance=save_allowance(args.out)
            if len(completed)==len(expected):break
            snapshot=resources(args.device);observed=[];worker_progress=[]
            if on_poll:on_poll(snapshot,[row['receipt'] for row in active.values()])
            for key,row in active.items():
                progress_path=row['segment']/'status.json'
                progress,progress_fresh,progress_error=read_live_json(progress_path,row.get('last_progress'))
                if progress_fresh:row['last_progress']=progress
                resource_path=row['segment']/'worker_resources.json'
                resource,resource_fresh,resource_error=read_live_json(resource_path,row.get('last_resource'))
                if resource_fresh:row['last_resource']=resource
                if resource.get('copy_process_rss_bytes'):
                    measured=resource['copy_process_rss_bytes']/1024**3
                    worker_ram_gib=max(worker_ram_gib,measured*1.15+.25)
                startup=getattr(args,'fit_startup_updates',100)
                remaining=args.train_steps-row['resumed_after']
                fresh=progress_fresh and resource_fresh
                observed.append(dict(startup_measured=fresh and progress.get('segment_updates',progress.get('step',0))>=min(startup,remaining)))
                worker_progress.append(dict(progress,key=key,path=str(row['folder']),telemetry_stale=not fresh,
                    telemetry_read_errors={name:error for name,error in
                        (('status',progress_error),('resources',resource_error)) if error}))
            allowed,reason=admission(snapshot,observed,reserve_mib=args.fit_gpu_reserve_mib,
                worker_mib=args.fit_worker_budget_mib,ram_reserve_gib=args.fit_ram_reserve_gib,worker_ram_gib=worker_ram_gib)
            if shutil.disk_usage(args.out).free<4*1024**3+2*allowance:
                allowed=False;reason='Preserving recording reserve plus temporary and last-partial save space'
            candidates=[task for task in ready_tasks(args.worlds,completed,active,donors) if f'{task[0]}_{task[1]}' in expected]
            waiting=[task for task in candidates if pauses.get(f'{task[0]}_{task[1]}',{}).get('next_admission_at',0)>time.time()]
            candidates=[task for task in candidates if task not in waiting]
            if waiting and not candidates and allowed:reason='Resource pause: waiting for the recorded retry time; completed updates remain saved'
            if len(active)<capacity and allowed and candidates:
                arm,index=candidates[0];key=f'{arm}_{index}'
                tape=None
                if arm=='yoked':tape=json.loads((completed[f'responsive_{donors[index]}']/'fit.json').read_text())['adult_tape']
                def expected_binding(folder):
                    a=settings(args.checkpoint,folder,args.speech,args.device,args.seed+index)
                    return continuation.binding(a,args.checkpoint,folder,args.seed+index,args.train_steps,limits,tape)
                resumed=continuation.latest(args.out/key,expected_binding)
                if resumed:
                    folder,resume_checkpoint=resumed
                    resumed_after=json.loads((resume_checkpoint/'fit_continuation.json').read_text())['trial_step']
                else:
                    folder=next_attempt(args.out/key);folder.mkdir(parents=True,exist_ok=False)
                    resume_checkpoint=None;resumed_after=0
                    write(folder/'attempt.json',dict(original_parent=str(args.checkpoint),world_seed=args.seed+index,arm=arm,
                        resumed_from_partial=False,preserved_prior=str(args.out/key) if folder!=args.out/key else None,
                        reason='Independent full-budget attempt from original parent; incompatible older partials retained'))
                segment=continuation.segment(folder)
                payload=dict(parent=str(args.checkpoint.resolve()),out=str(folder.resolve()),speech=str(args.speech.resolve()),
                    device=args.device,seed=args.seed+index,steps=args.train_steps,limits=limits,tape=tape,coordinator=owner,
                    gpu_reserve_mib=args.fit_gpu_reserve_mib,ram_reserve_gib=args.fit_ram_reserve_gib,
                    save_allowance_bytes=allowance,
                    resume_checkpoint=None if resume_checkpoint is None else str(resume_checkpoint.resolve()),segment_out=str(segment.resolve()))
                spec=segment/'worker_spec.json';write(spec,payload);log=(segment/'worker.log').open('ab')
                command=[sys.executable,'-u','-m',__name__,'--worker',str(spec)]
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                receipt=dict(pid=process.pid,process_created=psutil.Process(process.pid).create_time(),key=key,out=str(folder),segment_out=str(segment),command=command)
                active[key]=dict(process=process,log=log,folder=folder,segment=segment,resumed_after=resumed_after,receipt=receipt,arm=arm,index=index);publish()
            write(args.out/'status.json',dict(stage='fit_parallel' if active else 'waiting_for_resources',
                pid=os.getpid(),timestamp=time.time(),requested_workers=args.fit_workers,effective_workers=len(active),maximum_workers=capacity,
                completed_copies=len(completed),total_copies=len(expected),workers=worker_progress,
                resources=snapshot,waiting_reason=reason,resource_backoff=pauses,
                worker_ram_budget_gib=worker_ram_gib,save_allowance_bytes=allowance))
            time.sleep(2)
        return checkpoints,fit_paths
    finally:
        for row in active.values():
            if row['process'].poll() is None:
                write(row['segment']/'stop_requested.json',dict(reason='Coordinator stopped; save copied state',timestamp=time.time()))
            row['log'].close()


if __name__=='__main__':
    # The worker and fit() must share one exception class. Running this file as
    # -m otherwise creates a second class under __main__ and skips partial-save
    # handling when fit() imports the canonical package module.
    from mimo_bridge.tools.caregiver_fit_workers import worker as canonical_worker
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path,required=True)
    raise SystemExit(canonical_worker(parser.parse_args().worker))
