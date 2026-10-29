# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Registered, shared checkpoint retention across production restarts.

Only enrolled checkpoint directories and their recorded companions are eligible.
Study outputs are never enrolled implicitly. A global budget supplements the
free-space reserve; named recovery and experiment dependencies always win.
"""
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
import uuid

CONTRACT='checkpoint_retention_v1'
POLICY_NAME='.checkpoint_retention.json'
log=logging.getLogger(__name__)


def read(path):
    def invalid(value):raise ValueError('Nonfinite retention metadata: '+value)
    with Path(path).open(encoding='utf-8-sig') as stream:return json.load(stream,parse_constant=invalid)


def within(path,roots):
    path=Path(path).resolve()
    return any(path!=root and path.is_relative_to(root) for root in roots)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()


def find_policy(directory):
    directory=Path(directory).resolve()
    for parent in (directory,*directory.parents):
        candidate=parent/POLICY_NAME
        if candidate.is_file():return candidate
    return None


def policy(path):
    value=read(path)
    if value.get('schema')!=CONTRACT:raise ValueError('Unknown checkpoint retention policy')
    for key in ('checkpoint_run_roots','companion_roots','reference_roots'):
        if not isinstance(value.get(key),list) or not value[key]:raise ValueError('Missing registered '+key)
        value[key]=[Path(p).resolve() for p in value[key]]
    value['checkpoint_roots']=[Path(p).resolve() for p in value.get('checkpoint_roots',[])]
    for root in [*value['checkpoint_run_roots'],*value['companion_roots'],*value['reference_roots']]:
        if root==Path(root.anchor):raise ValueError('A volume root cannot be enrolled for checkpoint retirement')
    for key in ('maximum_checkpoint_bytes','minimum_free_bytes','minimum_recovery_snapshots'):
        number=value.get(key)
        if not isinstance(number,int) or isinstance(number,bool) or number<1:raise ValueError('Invalid retention '+key)
    if value['minimum_recovery_snapshots']<2:raise ValueError('At least two recovery checkpoints must remain')
    value['path']=Path(path).resolve()
    value['storage']=Path(value.get('storage',value['checkpoint_run_roots'][0])).resolve()
    if any(root.anchor!=value['storage'].anchor for root in
           [*value['checkpoint_run_roots'],*value['checkpoint_roots'],*value['companion_roots']]):
        raise ValueError('One retention budget must own one storage volume')
    value['audit_path']=Path(value['audit_path']).resolve()
    return value


def checkpoint_directories(p):
    roots=set(p['checkpoint_roots'])
    for group in p['checkpoint_run_roots']:
        for candidate in group.glob('*/checkpoints'):
            if not within(candidate,[group]):raise ValueError('Checkpoint root escapes registered run directory')
            roots.add(candidate.resolve())
    return sorted(roots,key=str)


def owned_save_directory(p,directory):
    directory=Path(directory).resolve()
    return any(directory==r or within(directory,[r]) for r in [*p['companion_roots'],*checkpoint_directories(p)])


def checkpoint(path):
    """Cheap structural completeness; the native writer publishes atomically."""
    path=Path(path).resolve()
    if not re.fullmatch(r'ckpt_step_\d+',path.name):return None
    try:
        meta=read(path/'meta.json');receipt=read(path/'model_receipt.json')
        if meta['step']!=int(path.name.removeprefix('ckpt_step_')):return None
        required=['model.safetensors','optimizer.pt','runtime_state.pt','meta.json','model_receipt.json']
        if not all((path/name).stat().st_size>0 for name in required):return None
        episode=read(path/'episode_state.json') if (path/'episode_state.json').exists() else None
        if receipt.get('episode_continuation') and episode is None:return None
        if episode:
            target=Path(episode.get('storage_path',str(path/'episode_state.pt.gz'))).resolve()
            if target.stat().st_size!=episode['bytes'] or episode['step']!=meta['step']:return None
        else:target=None
        files=[f for f in path.rglob('*') if f.is_file()]
        if any(not within(f,[path]) for f in files):raise ValueError('Checkpoint contains an external file link')
        return dict(path=path,step=meta['step'],timestamp=float(meta['timestamp']),
            bytes=sum(f.stat().st_size for f in files),companion=target,episode=episode)
    except (OSError,KeyError,TypeError,json.JSONDecodeError):return None


def command_paths(command):
    result=set()
    for i,arg in enumerate(command[:-1]):
        if arg in ('--checkpoint','--resume'):result.add(Path(command[i+1]).resolve())
    return result


def source_paths(value):
    result=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('checkpoint','parent','resume') and isinstance(item,str) and re.search(r'ckpt_step_\d+$',item):
                result.add(Path(item).resolve())
            elif key=='command' and isinstance(item,list):result.update(command_paths(item))
            elif isinstance(item,(dict,list)):result.update(source_paths(item))
    elif isinstance(value,list):
        for item in value:result.update(source_paths(item))
    return result


def protected(p,*,processes=True):
    pins={Path(path).resolve() for path in p.get('protected_checkpoints',[])}
    for entry in p.get('protection_files',[]):
        path=Path(entry['path'])
        if not path.exists():raise ValueError('Protection source is missing: '+str(path))
        data=read(path);kind=entry['kind']
        if kind=='queue':
            jobs=data['jobs'];active={k for k,v in jobs.items() if v.get('state') not in ('complete','cancelled')}
            for key in active:pins.update(source_paths(jobs[key]))
            for job in data.get('generated_jobs',[]):
                if job['id'] in active:pins.update(source_paths(job))
            if entry.get('specification'):
                for job in read(entry['specification']).get('jobs',[]):
                    if job['id'] not in jobs or job['id'] in active:pins.update(source_paths(job))
        elif kind=='study':
            out=Path(data['out']);status=read(out/'status.json') if (out/'status.json').exists() else {}
            if status.get('stage')!='complete':pins.update(source_paths(data))
        elif kind=='supervisor':pins.update(command_paths(data['command']))
        elif kind=='pins':pins.update(source_paths(data))
        else:raise ValueError('Unknown checkpoint protection source')
    if processes:
        import psutil
        for proc in psutil.process_iter(['name','cmdline']):
            try:
                if 'python' in (proc.info['name'] or '').lower():
                    if proc.info['cmdline'] is None:raise ValueError('Cannot inspect an active Python checkpoint dependency')
                    pins.update(command_paths(proc.info['cmdline']))
            except psutil.NoSuchProcess:continue
            except psutil.AccessDenied:raise ValueError('Cannot inspect an active Python checkpoint dependency')
    return pins


def references(p):
    """Include retained, staged, replaced and copied-checkpoint manifests."""
    owners={}
    roots=set(p['reference_roots'])|set(checkpoint_directories(p))
    manifests=set()
    for root in roots:
        manifests.update(path.resolve() for path in root.rglob('episode_state.json'))
    for manifest in manifests:
        value=read(manifest)
        target=Path(value.get('storage_path',str(manifest.parent/'episode_state.pt.gz'))).resolve()
        owners.setdefault(target,set()).add(manifest.parent.resolve())
    return owners


def total_bytes(rows):
    companions={r['companion'] for r in rows if r['companion'] is not None and not within(r['companion'],[r['path']])}
    return sum(r['bytes'] for r in rows)+sum(path.stat().st_size for path in companions)


def append_audit(p,record):
    path=p['audit_path'];path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


@contextmanager
def lock(p):
    path=p['path'].with_suffix('.lock');stream=path.open('a+b')
    try:
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:stream.close()


def retire(p,row,owners,*,replaced=False):
    target=row['path'];roots=checkpoint_directories(p)
    allowed=re.fullmatch(r'\.replaced_ckpt_step_\d+_[0-9a-f]+',target.name) if replaced else re.fullmatch(r'ckpt_step_\d+',target.name)
    if target.parent not in roots or not allowed:raise ValueError('Retirement target is outside registered checkpoint directory')
    files=[f for f in target.rglob('*') if f.is_file()]
    if any(not within(f,[target]) for f in files):raise ValueError('Retirement target contains an external file link')
    # A different checkpoint can reference an in-directory companion as well.
    if any(within(blob,[target]) and refs-{target} for blob,refs in owners.items()):
        raise ValueError('Another checkpoint still references a file inside this checkpoint')
    blob=row['companion'];external=blob is not None and not within(blob,[target])
    delete_blob=bool(external and not (owners.get(blob,set())-{target}) and within(blob,p['companion_roots']))
    if external and delete_blob:
        if blob.suffixes[-2:]!=['.pt','.gz'] or digest(blob)!=row['episode']['sha256']:
            raise ValueError('Retired companion fails its recorded type or integrity check')
    record=dict(schema=CONTRACT,event='retirement_prepared',id=uuid.uuid4().hex,timestamp=time.time(),
        checkpoint=str(target),step=row['step'],checkpoint_timestamp=row.get('timestamp'),
        files={str(f.relative_to(target)):
            dict(bytes=f.stat().st_size,sha256=digest(f)) for f in files},
        companion=None if blob is None else str(blob),delete_companion=delete_blob,
        companion_sha256=None if row['episode'] is None else row['episode']['sha256'])
    append_audit(p,record) # Durable evidence precedes any deletion.
    shutil.rmtree(target)
    if delete_blob:blob.unlink()
    append_audit(p,dict(schema=CONTRACT,event='retirement_complete',id=record['id'],timestamp=time.time()))
    return record


def maintain(path,*,apply=False,reserve_additional=0,processes=True):
    p=policy(path)
    if not apply:return _maintain(p,False,reserve_additional,processes)
    with lock(p):return _maintain(p,True,reserve_additional,processes)


def thinning_cursor(p):
    if not p['audit_path'].exists():return None
    prepared={};last=None
    with p['audit_path'].open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            if row['event']=='retirement_prepared':prepared[row['id']]=row
            elif row['event']=='retirement_complete':
                old=prepared.get(row['id'],{})
                if old.get('checkpoint_timestamp') is not None:last=old['checkpoint_timestamp']
    return last


def thinning_order(rows,cursor=None):
    """Skip one older eligible snapshot between removals, across later calls too.

    When that pass reaches the recent protected boundary, repeat over the
    remaining older history. The journal retains the last chronological cursor
    so hourly maintenance does not silently become contiguous-oldest deletion.
    """
    pending=list(rows);ordered=[]
    while pending:
        later=[] if cursor is None else [row for row in pending if row['timestamp']>cursor]
        selected=later[1] if len(later)>1 else pending[0]
        ordered.append(selected);pending.remove(selected);cursor=selected['timestamp']
    return ordered


def finish_interrupted_companions(p,owners):
    """Recover only a journaled deletion interrupted after checkpoint removal."""
    if not p['audit_path'].exists():return []
    prepared={};complete=set()
    with p['audit_path'].open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            if row['event']=='retirement_prepared':prepared[row['id']]=row
            elif row['event']=='retirement_complete':complete.add(row['id'])
    recovered=[]
    for identity,row in prepared.items():
        if identity in complete or Path(row['checkpoint']).exists() or not row['delete_companion']:continue
        blob=Path(row['companion']).resolve()
        if owners.get(blob) or not within(blob,p['companion_roots']):continue
        if blob.exists():
            if blob.suffixes[-2:]!=['.pt','.gz'] or digest(blob)!=row['companion_sha256']:continue
            blob.unlink()
        append_audit(p,dict(schema=CONTRACT,event='retirement_complete',id=identity,timestamp=time.time(),
                           recovered_interrupted_companion=True))
        recovered.append(str(blob))
    return recovered


def finish_replaced_checkpoints(p,pins):
    result=[]
    for root in checkpoint_directories(p):
        for old in root.glob('.replaced_ckpt_step_*'):
            match=re.fullmatch(r'\.replaced_ckpt_step_(\d+)_[0-9a-f]+',old.name)
            if match is None or old.resolve() in pins:continue
            current=root/('ckpt_step_'+match[1])
            if checkpoint(current) is None:continue # The backup can still be the only recovery.
            receipt=read(old/'episode_state.json') if (old/'episode_state.json').exists() else None
            blob=None if receipt is None else Path(receipt['storage_path']).resolve()
            row=dict(path=old.resolve(),step=int(match[1]),companion=blob,episode=receipt)
            result.append(retire(p,row,references(p),replaced=True))
    return result


def _maintain(p,apply,reserve_additional,processes):
    rows=[]
    for root in checkpoint_directories(p):
        for path in root.glob('ckpt_step_*'):
            item=checkpoint(path)
            if item is not None:rows.append(item)
    rows.sort(key=lambda row:(row['timestamp'],row['step']))
    pins=protected(p,processes=processes)
    pins.update(r['path'] for r in rows[-p['minimum_recovery_snapshots']:])
    replacements=finish_replaced_checkpoints(p,pins) if apply else []
    owners=references(p)
    recovered=finish_interrupted_companions(p,owners) if apply else []
    before=total_bytes(rows);free=shutil.disk_usage(p['storage']).free
    planned=[];retained=list(rows)
    eligible=[row for row in rows if row['path'] not in pins and not any(
        within(blob,[row['path']]) and refs-{row['path']} for blob,refs in owners.items())]
    for row in thinning_order(eligible,thinning_cursor(p)):
        if total_bytes(retained)<=p['maximum_checkpoint_bytes'] and free>=p['minimum_free_bytes']+reserve_additional:break
        if row['path'] in pins:continue
        if any(within(blob,[row['path']]) and refs-{row['path']} for blob,refs in owners.items()):continue
        if apply:
            # Active readers/jobs can appear after the first plan was produced.
            if row['path'] in protected(p,processes=processes):continue
            owners=references(p)
            record=retire(p,row,owners)
            free=shutil.disk_usage(p['storage']).free
        else:
            external=row['companion'] is not None and not within(row['companion'],[row['path']])
            removable=external and not owners.get(row['companion'],set())-{row['path']} and within(row['companion'],p['companion_roots'])
            reclaimed=row['bytes']+(row['companion'].stat().st_size if removable else 0)
            free+=reclaimed
            record=dict(checkpoint=str(row['path']),step=row['step'],bytes=reclaimed,delete_companion=bool(removable))
        planned.append(record);retained.remove(row)
        for refs in owners.values():refs.discard(row['path'])
    after=total_bytes(retained)
    blobs={path.resolve() for root in p['companion_roots'] for path in root.rglob('*.pt.gz')}
    target_met=after<=p['maximum_checkpoint_bytes'] and free>=p['minimum_free_bytes']+reserve_additional
    pinned_bytes=total_bytes([row for row in retained if row['path'] in pins])
    report=dict(schema=CONTRACT,applied=apply,registered_checkpoint_count=len(rows),before_bytes=before,
        after_bytes=after,free_bytes=free,reserve_additional=reserve_additional,retired=planned,
        recovered_interrupted_companions=recovered,
        retired_replaced_checkpoints=replacements,
        protected_checkpoint_bytes=pinned_bytes,pinned_budget_exceeded=pinned_bytes>p['maximum_checkpoint_bytes'],
        largest_complete_checkpoint_bytes=max((row['bytes']+(row['companion'].stat().st_size if row['companion']
            and not within(row['companion'],[row['path']]) else 0) for row in retained),default=0),
        registered_companion_bytes=sum(blob.stat().st_size for blob in blobs),
        unreferenced_registered_companion_bytes=sum(blob.stat().st_size for blob in blobs if not owners.get(blob)),
        unreferenced_scope='No reference in the registered scan roots; may be unpublished or historical. Not deletion authority.',
        protected_checkpoints=sorted(map(str,pins)),
        target_met=target_met,
        reason='Protected recovery and experiment evidence is never removed to force a storage target')
    if apply and not target_met:
        log.warning('Storage target unmet: retained checkpoints %d bytes, protected %d bytes, free %d bytes; no protected evidence removed',
                    after,pinned_bytes,free)
    return report


def ensure_registered_space(directory,additional=0):
    try:
        path=find_policy(directory)
        if path is None:return None
        p=policy(path)
        if not owned_save_directory(p,directory):return None
        roots=checkpoint_directories(p)
        rows=[row for root in roots for candidate in root.glob('ckpt_step_*')
              if (row:=checkpoint(candidate)) is not None]
        deferred=any(any(root.glob('.replaced_ckpt_step_*')) for root in roots)
        # Native save checks model, runtime and physical payload separately.
        # Avoid scanning every study/process/reference three times when there
        # is room and no deletion is needed. The watcher still gives full audits.
        if (not deferred and total_bytes(rows)<=p['maximum_checkpoint_bytes']
                and shutil.disk_usage(p['storage']).free>=p['minimum_free_bytes']+additional):
            return dict(target_met=True,cleanup_needed=False)
        return maintain(path,apply=True,reserve_additional=additional)
    except (OSError,ValueError) as error:
        # Another writer may own maintenance. Native payload-space validation
        # still runs; storage housekeeping must not turn a saved model into loss.
        log.warning('Checkpoint retention could not run: %s',error)
        return None


def retire_replaced(directory,current):
    """Same-step replacement retires its old companion after new publication."""
    path=find_policy(directory)
    if path is None:return False
    p=policy(path);directory=Path(directory).resolve()
    if directory.parent not in checkpoint_directories(p):return False
    receipt=read(directory/'episode_state.json') if (directory/'episode_state.json').exists() else None
    blob=None if receipt is None else Path(receipt['storage_path']).resolve()
    row=dict(path=directory,step=int(current.name.removeprefix('ckpt_step_')),companion=blob,episode=receipt)
    with lock(p):retire(p,row,references(p),replaced=True)
    return True
