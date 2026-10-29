# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Read-only model identity for retained and explicitly retired test subjects."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=128)
def file_hash(path,size,modified_ns,changed_ns):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def sha256(path):
    path=Path(path);stat=path.stat()
    return file_hash(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


@lru_cache(maxsize=16)
def retired_records(path,size,modified_ns,changed_ns):
    prepared={};complete=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row.get('schema')!='checkpoint_retention_v1':continue
        if row.get('event')=='retirement_prepared':prepared[row['id']]=row
        if row.get('event')=='retirement_complete' and row['id'] in prepared:
            complete.append(prepared[row['id']])
    return complete


def checkpoint_identity(checkpoint,expected=None):
    """A retirement receipt preserves historical identity, never rerun availability."""
    checkpoint=Path(checkpoint).resolve();weights=checkpoint/'model.safetensors'
    if weights.is_file():
        digest=sha256(weights)
        if expected and expected!=digest:raise ValueError('Preserved checkpoint weights changed after review')
        return dict(checkpoint=str(checkpoint),model_sha256=digest,availability='retained')
    if not expected:raise ValueError('Retired snapshot requires a previously captured model identity')
    from lem.training.retention import find_policy
    located=find_policy(checkpoint)
    if located:
        policy=json.loads(located.read_text(encoding='utf-8-sig'))
        if policy.get('schema')!='checkpoint_retention_v1':raise ValueError('Unknown retirement provenance')
        enrolled=[Path(path).resolve() for path in policy.get('checkpoint_run_roots',[])+policy.get('checkpoint_roots',[])]
        if not any(checkpoint.is_relative_to(root) and checkpoint!=root for root in enrolled):
            raise ValueError('Snapshot is outside the registered retirement policy')
        audit=Path(policy['audit_path']).resolve()
        if not audit.is_relative_to(Path(policy['storage']).resolve()):raise ValueError('Retirement audit leaves registered storage')
        stat=audit.stat()
        for record in reversed(retired_records(str(audit),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)):
            if Path(record['checkpoint']).resolve()!=checkpoint:continue
            if record['files']['model.safetensors']['sha256']!=expected:
                raise ValueError('Retired model identity differs from the reviewed source')
            return dict(checkpoint=str(checkpoint),model_sha256=expected,availability='retired',
                retirement=dict(audit=str(audit),id=record['id'],timestamp=record['timestamp'],
                    record_sha256=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()),
                limitation='The hash-bound historical result remains; these retired weights cannot currently be rerun.')
    raise ValueError('Source weights are unavailable and no matching completed retirement receipt exists')
