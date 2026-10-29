# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from pathlib import Path
import shutil
import torch
import pytest
from lem.training.checkpoint import CheckpointManager
from lem.training.retention import CONTRACT,POLICY_NAME,maintain,policy,protected,digest
from mimo_bridge.continuation_state import write_episode,load as load_episode


def setup_policy(tmp_path):
    for name in ('runs','physical','studies'):(tmp_path/name).mkdir()
    path=tmp_path/POLICY_NAME
    data=dict(schema=CONTRACT,checkpoint_run_roots=[str(tmp_path/'runs')],
        companion_roots=[str(tmp_path/'physical')],reference_roots=[str(tmp_path)],
        maximum_checkpoint_bytes=10**10,minimum_free_bytes=1,minimum_recovery_snapshots=2,
        protected_checkpoints=[],protection_files=[],audit_path=str(tmp_path/'retired.jsonl'))
    path.write_text(json.dumps(data));return path,data


def store(path,data):path.write_text(json.dumps(data))


def native_checkpoint(root,step,physical):
    config={'checkpoint':{'format':'safetensors','keep_last_n':48,'progressive_resize':False}}
    manager=CheckpointManager(config,str(root))
    modules=[torch.nn.Linear(3,3) for _ in range(5)]
    optimizer=torch.optim.Adam([p for m in modules for p in m.parameters()])
    sum(p.square().sum() for m in modules for p in m.parameters()).backward();optimizer.step()
    actual={'step':step,'weights':{k:v.detach().clone() for k,v in modules[0].state_dict().items()}}
    def companion(staging,final):
        torch.save({'earned_history':torch.arange(12)},staging/'runtime_state.pt')
        write_episode(staging,dict(version='episode_continuation_v1',trainer={'step':step},
            worlds=[{'payload':{'sim_time':float(step)}}],test_history=torch.arange(32)),
            storage_root=physical,published_path=final)
    saved=Path(manager.save(step,*modules,optimizer,{'episode_continuation':'episode_continuation_v1'},
                            companion_writer=companion))
    actual['path']=saved;actual['companion']=Path(json.loads((saved/'episode_state.json').read_text())['storage_path'])
    return manager,actual


def four_checkpoints(tmp_path):
    path,data=setup_policy(tmp_path);rows=[]
    for step in range(4):
        # Restart directories do not reset the global storage allowance.
        manager,row=native_checkpoint(tmp_path/'runs'/f'restart_{step}'/'checkpoints',step,tmp_path/'physical')
        rows.append(row)
    return path,data,rows,manager


def test_native_checkpoint_retirement_removes_external_state_and_keeps_exact_recovery(tmp_path):
    path,data,rows,manager=four_checkpoints(tmp_path)
    original_hash=digest(rows[0]['companion']);data['maximum_checkpoint_bytes']=1;store(path,data)
    report=maintain(path,apply=True,processes=False)
    assert len(report['retired'])==2 and not report['target_met'] # Recovery wins over impossible budget.
    for row in rows[:2]:assert not row['path'].exists() and not row['companion'].exists()
    for row in rows[2:]:
        loaded=manager._load_state(row['path'],map_location='cpu')
        assert loaded['step']==row['step']
        assert loaded['optimizer']['state']
        for key,value in row['weights'].items():assert torch.equal(loaded['encoder_registry'][key],value)
        assert torch.equal(load_episode(row['path'])['test_history'],torch.arange(32))
    records=[json.loads(line) for line in (tmp_path/'retired.jsonl').read_text().splitlines()]
    assert records[0]['event']=='retirement_prepared' and records[1]['event']=='retirement_complete'
    assert records[0]['companion_sha256']==original_hash and records[0]['files']['optimizer.pt']['sha256']


def test_shared_companion_from_retained_study_is_never_deleted(tmp_path):
    path,data,rows,_=four_checkpoints(tmp_path)
    reference=tmp_path/'studies'/'completed_fit'/'ckpt_step_0';reference.mkdir(parents=True)
    shutil.copyfile(rows[0]['path']/'episode_state.json',reference/'episode_state.json')
    data['maximum_checkpoint_bytes']=1;store(path,data)
    report=maintain(path,apply=True,processes=False)
    assert not rows[0]['path'].exists() and rows[0]['companion'].exists()
    assert reference.exists() and not report['retired'][0]['delete_companion']


def test_unfinished_study_and_queued_sources_are_pinned_across_global_restarts(tmp_path):
    path,data,rows,_=four_checkpoints(tmp_path)
    study=tmp_path/'studies'/'active';study.mkdir();(study/'status.json').write_text('{"stage":"failed"}')
    launch=tmp_path/'study_launch.json';store(launch,{'out':str(study),'parent':str(rows[0]['path'])})
    queue=tmp_path/'queue.json';store(queue,{'jobs':{'later':{'state':'queued'}},'generated_jobs':[
        {'id':'later','checkpoint':str(rows[1]['path'])}]})
    data['protection_files']=[{'kind':'study','path':str(launch)},{'kind':'queue','path':str(queue)}]
    data['maximum_checkpoint_bytes']=1;store(path,data)
    report=maintain(path,apply=True,processes=False)
    assert not report['retired'] and all(row['path'].exists() for row in rows)
    assert set(map(str,[row['path'] for row in rows]))<=set(report['protected_checkpoints'])


def test_dry_run_never_deletes_and_external_unregistered_files_are_left_alone(tmp_path):
    path,data,rows,_=four_checkpoints(tmp_path)
    outside=tmp_path/'valuable_external.pt.gz';shutil.copyfile(rows[0]['companion'],outside)
    receipt=json.loads((rows[0]['path']/'episode_state.json').read_text());receipt['storage_path']=str(outside)
    store(rows[0]['path']/'episode_state.json',receipt)
    data['maximum_checkpoint_bytes']=1;store(path,data)
    dry=maintain(path,processes=False)
    assert dry['retired'] and all(row['path'].exists() for row in rows)
    assert not (tmp_path/'retired.jsonl').exists()
    maintain(path,apply=True,processes=False)
    assert outside.exists()


def test_new_native_saves_apply_global_policy_instead_of_per_restart_48(tmp_path):
    path,data=setup_policy(tmp_path);data['maximum_checkpoint_bytes']=1;store(path,data)
    rows=[]
    for step in range(4):
        _,row=native_checkpoint(tmp_path/'runs'/f'segment_{step}'/'checkpoints',step,tmp_path/'physical');rows.append(row)
    assert [row['path'].exists() for row in rows]==[False,False,True,True]
    assert [row['companion'].exists() for row in rows]==[False,False,True,True]


def test_same_step_atomic_replacement_retires_only_old_companion(tmp_path):
    path,data=setup_policy(tmp_path)
    root=tmp_path/'runs'/'segment'/'checkpoints'
    _,first=native_checkpoint(root,5,tmp_path/'physical')
    _,replacement=native_checkpoint(root,5,tmp_path/'physical')
    assert first['companion']!=replacement['companion']
    assert not first['companion'].exists() and replacement['companion'].exists()
    assert load_episode(replacement['path'])['trainer']['step']==5


def test_corrupt_protection_manifest_prevents_retirement(tmp_path):
    path,data,rows,_=four_checkpoints(tmp_path)
    broken=tmp_path/'queue.json';broken.write_text('{not complete')
    data['protection_files']=[{'kind':'queue','path':str(broken)}];data['maximum_checkpoint_bytes']=1;store(path,data)
    with pytest.raises(ValueError):maintain(path,apply=True,processes=False)
    assert all(row['path'].exists() for row in rows)


def test_active_process_checkpoint_flags_are_protected(tmp_path,monkeypatch):
    path,data,rows,_=four_checkpoints(tmp_path)
    import psutil
    class Process:
        info={'name':'python.exe','cmdline':['python.exe','-m','some_trial','--checkpoint',str(rows[0]['path'])]}
    monkeypatch.setattr(psutil,'process_iter',lambda fields:[Process()])
    assert rows[0]['path'] in protected(policy(path))


def test_interrupted_retirement_only_finishes_recorded_unshared_companion(tmp_path):
    path,data,rows,_=four_checkpoints(tmp_path)
    from lem.training.retention import append_audit
    orphan=rows[0]['companion']
    append_audit(policy(path),dict(event='retirement_prepared',id='interrupted-test',
        checkpoint=str(rows[0]['path']),delete_companion=True,companion=str(orphan),companion_sha256=digest(orphan)))
    shutil.rmtree(rows[0]['path'])
    report=maintain(path,apply=True,processes=False)
    assert report['recovered_interrupted_companions']==[str(orphan)] and not orphan.exists()


def test_local_companion_shared_by_other_checkpoint_pins_entire_directory(tmp_path):
    path,data,rows,_=four_checkpoints(tmp_path)
    local=rows[0]['path']/'episode_state.pt.gz';shutil.copyfile(rows[0]['companion'],local)
    receipt=json.loads((rows[0]['path']/'episode_state.json').read_text());receipt['storage_path']=str(local)
    store(rows[0]['path']/'episode_state.json',receipt)
    outside=tmp_path/'studies'/'independent';outside.mkdir();store(outside/'episode_state.json',receipt)
    data['maximum_checkpoint_bytes']=1;store(path,data)
    maintain(path,apply=True,processes=False)
    assert rows[0]['path'].exists() and local.exists()


def test_incomplete_new_checkpoint_never_displaces_complete_recovery(tmp_path):
    path,data,rows,_=four_checkpoints(tmp_path)
    (rows[3]['path']/'episode_state.json').unlink()
    data['maximum_checkpoint_bytes']=1;store(path,data)
    report=maintain(path,apply=True,processes=False)
    assert rows[1]['path'].exists() and rows[2]['path'].exists()
    assert rows[3]['path'].exists() # Unclassified incomplete evidence is not silently erased.
    assert str(rows[3]['path']) not in report['protected_checkpoints']


def test_replacement_housekeeping_failure_keeps_new_native_save_and_defers_backup(tmp_path,monkeypatch):
    path,data=setup_policy(tmp_path);root=tmp_path/'runs'/'segment'/'checkpoints'
    _,first=native_checkpoint(root,5,tmp_path/'physical')
    import lem.training.retention as retention
    original=retention.retire_replaced
    monkeypatch.setattr(retention,'retire_replaced',lambda *a:(_ for _ in ()).throw(OSError('another writer owns retention')))
    # The save still returns its valid new checkpoint; post-publication global
    # maintenance repairs the deferred backup without losing either new file.
    _,replacement=native_checkpoint(root,5,tmp_path/'physical')
    assert replacement['path'].exists() and replacement['companion'].exists()
    assert not first['companion'].exists()
    assert not list(root.glob('.replaced_ckpt_step_*'))


def test_malformed_policy_disables_cleanup_without_breaking_native_save(tmp_path):
    path,data=setup_policy(tmp_path);path.write_text('{partial policy')
    _,row=native_checkpoint(tmp_path/'runs'/'live'/'checkpoints',9,tmp_path/'physical')
    assert load_episode(row['path'])['trainer']['step']==9
    assert not (tmp_path/'retired.jsonl').exists()


def test_temporal_thinning_skips_old_snapshots_across_restart_directories_and_later_calls(tmp_path):
    path,data=setup_policy(tmp_path);rows=[]
    for step in range(7):
        _,row=native_checkpoint(tmp_path/'runs'/f'restart_{step}'/'checkpoints',step,tmp_path/'physical');rows.append(row)
    from lem.training.retention import checkpoint,total_bytes,thinning_order
    descriptors=[checkpoint(row['path']) for row in rows]
    # A budget requiring two removals should keep the intervening older moment.
    data['maximum_checkpoint_bytes']=total_bytes(descriptors)-total_bytes([descriptors[0],descriptors[2]])
    store(path,data);report=maintain(path,apply=True,processes=False)
    assert [record['step'] for record in report['retired']]==[0,2]
    assert rows[1]['path'].exists() and rows[3]['path'].exists()
    # Later maintenance resumes the same alternating pass, rather than deleting
    # oldest survivor1 and erasing an uninterrupted prefix of the history.
    remaining=[descriptor for descriptor in descriptors if descriptor['path'].exists()]
    data['maximum_checkpoint_bytes']=total_bytes(remaining)-total_bytes([descriptors[4]])
    store(path,data);report=maintain(path,apply=True,processes=False)
    assert [record['step'] for record in report['retired']]==[4]
    assert all(rows[step]['path'].exists() for step in (1,3,5,6))
