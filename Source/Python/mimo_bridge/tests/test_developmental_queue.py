# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from pathlib import Path
import pytest
from mimo_bridge.tools import developmental_queue as queue


def test_missing_report_or_dead_worker_cannot_mark_test_complete(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'alive',lambda _:False)
    attempt=dict(out=str(tmp_path))
    (tmp_path/'status.json').write_text('{"stage":"complete"}')
    assert queue.poll_attempt(attempt)=='interrupted'
    (tmp_path/'report.json').write_text('{"value":NaN}')
    with pytest.raises(ValueError):queue.poll_attempt(attempt)
    (tmp_path/'report.json').write_text('{"value":0}')
    assert queue.poll_attempt(attempt)=='complete'
    assert len(attempt['report_sha256'])==64


def test_queue_keeps_negatives_and_rejects_posthoc_change(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'alive',lambda _:False)
    job=dict(id='body-1',module='mimo_bridge.tools.developmental_probe',args=[],
        question='Does it turn?',decision='Measure a new posture',scope='copy',max_attempts=1)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema=1,jobs=[job])))
    state=tmp_path/'state.json';queue.tick(manifest,state,tmp_path,launch=False)
    data=queue.read(state);attempt=tmp_path/'failed';attempt.mkdir()
    (attempt/'status.json').write_text('{"stage":"failed"}')
    data['jobs']['body-1']['attempts']=[dict(out=str(attempt))];state.write_text(json.dumps(data))
    result=queue.tick(manifest,state,tmp_path,launch=False)
    assert result['jobs']['body-1']['state']=='repair_required'
    assert (attempt/'status.json').is_file()
    job['question']='Was that a successful roll?';manifest.write_text(json.dumps(dict(schema=1,jobs=[job])))
    with pytest.raises(ValueError,match='changed'):queue.tick(manifest,state,tmp_path,launch=False)


def test_queue_refuses_unknown_command_and_shell_like_id():
    base=dict(id='one',module='mimo_bridge.tools.developmental_probe',args=[],question='a',decision='b',scope='c')
    queue.validate_job(base)
    for changes in (dict(module='arbitrary.module'),dict(id='../../escape'),dict(args=['--out','overwritten'])):
        with pytest.raises(ValueError):queue.validate_job(dict(base,**changes))


def test_live_process_blocks_duplicate_even_with_unreadable_status(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'alive',lambda _:True)
    monkeypatch.setattr(queue,'read',lambda _: (_ for _ in ()).throw(PermissionError('writer holds file')))
    assert queue.poll_attempt(dict(out=str(tmp_path)))=='running'


def test_dwell_requires_continuous_real_time():
    from mimo_bridge.tools.developmental_probe import longest_dwell
    assert longest_dwell([0,.1,.2,.3,.4],[True,True,False,True,True])==pytest.approx(.1)
    with pytest.raises(ValueError):longest_dwell([0,0],[True,True])


def test_expiring_launch_hold_preserves_active_test_and_releases_next_slot(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'alive',lambda _:True)
    monkeypatch.setattr(queue,'poll_attempt',lambda _:'running')
    clock=[1000.];monkeypatch.setattr(queue.time,'time',lambda:clock[0])
    job=dict(id='body-1',module='mimo_bridge.tools.developmental_probe',args=[],question='Turn?',decision='Measure',scope='copy')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema=1,jobs=[job])))
    state=tmp_path/'state.json';first=queue.tick(manifest,state,tmp_path,launch=False)
    first['jobs']['body-1']['attempts']=[dict(pid=99,out='existing')]
    state.write_text(json.dumps(first))
    control=tmp_path/'state_control.json'
    control.write_text(json.dumps(dict(cmd='pause_launches',until=1100.,reason='Finish continuation proof')))
    held=queue.tick(manifest,state,tmp_path)
    assert held['running'] and held['jobs']['body-1']['state']=='running'
    assert held['jobs']['body-1']['attempts']==first['jobs']['body-1']['attempts']
    assert held['launch_pause']['until']==1100.
    clock[0]=1101.
    released=queue.tick(manifest,state,tmp_path,launch=False)
    assert 'launch_pause' not in released and 'pause_reason' not in released


def test_launch_hold_blocks_start_until_expiry(tmp_path,monkeypatch):
    clock=[1000.];monkeypatch.setattr(queue.time,'time',lambda:clock[0])
    job=dict(id='body-1',module='mimo_bridge.tools.developmental_probe',args=[],question='Turn?',decision='Measure',scope='copy')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema=1,jobs=[job])))
    state=tmp_path/'state.json'
    (tmp_path/'state_control.json').write_text(json.dumps(dict(cmd='pause_launches',until=1100.)))
    calls=[]
    monkeypatch.setattr(queue,'resource_status',lambda _:calls.append(True) or dict(pause_reason='fixture'))
    held=queue.tick(manifest,state,tmp_path)
    assert not calls and held['jobs']['body-1']['attempts']==[]
    clock[0]=1101.;queue.tick(manifest,state,tmp_path)
    assert calls==[True]


def test_overflow_number_is_not_completed_evidence(tmp_path):
    path=tmp_path/'report.json';path.write_text('{"value":1e999}')
    with pytest.raises(ValueError):queue.read(path)


def test_recording_space_alone_does_not_hide_low_memory_or_system_disk(tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=(8*1024**3 if path==tmp_path else 100*1024**2)))
    monkeypatch.setattr(queue.psutil,'virtual_memory',lambda:SimpleNamespace(available=2*1024**3))
    status=queue.resource_status(tmp_path)
    assert 'paging' in status['pause_reason'] and 'RAM' in status['pause_reason']


def test_retired_instrument_preserves_its_declaration_and_attempts(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'alive',lambda _:False)
    job=dict(id='old-probe',module='mimo_bridge.tools.developmental_probe',args=[],
        question='Does it turn?',decision='Check control',scope='Copy')
    manifest=tmp_path/'manifest.json';state=tmp_path/'state.json'
    manifest.write_text(json.dumps(dict(schema=1,jobs=[job])))
    first=queue.tick(manifest,state,tmp_path,launch=False)
    manifest.write_text(json.dumps(dict(schema=1,jobs=[job],cancelled_jobs={'old-probe':'State leaked between branches; rerun fixed instrument.'})))
    result=queue.tick(manifest,state,tmp_path,launch=False)
    assert result['jobs']['old-probe']['state']=='cancelled'
    assert result['jobs']['old-probe']['declaration_sha256']==first['jobs']['old-probe']['declaration_sha256']


def test_recurring_jobs_are_coalesced_and_cannot_redeclare_after_results(tmp_path,monkeypatch):
    from mimo_bridge.tools import training_continuation
    monkeypatch.setattr(queue,'REPO',tmp_path)
    saved=tmp_path/'Saved/mimo_bridge';saved.mkdir(parents=True)
    (saved/'training_continuation.json').write_text('{"active_run":"run"}')
    checkpoint=[tmp_path/'checkpoints/ckpt_step_100']
    monkeypatch.setattr(training_continuation,'latest_complete',lambda _:checkpoint[0])
    job=dict(title='Observe',question='Does it turn?',decision='Locate loss of control',scope='Frozen copy',
        module='mimo_bridge.tools.developmental_probe',args=['--profile','body_floor'])
    spec=dict(recurring=[dict(id='body-v2',minimum_new_steps=50,minimum_hours=2,job=job)])
    state=dict(jobs={})
    queue.schedule_recurring(spec,state,100)
    assert len(state['generated_jobs'])==1
    checkpoint[0]=tmp_path/'checkpoints/ckpt_step_200'
    queue.schedule_recurring(spec,state,10000)
    assert len(state['generated_jobs'])==1  # pending older copy, no backlog explosion
    state['jobs']['body-v2-step-100']=dict(state='complete')
    queue.schedule_recurring(spec,state,10000)
    assert len(state['generated_jobs'])==2
    assert state['generated_jobs'][0]['checkpoint'].endswith('ckpt_step_100')
    spec['recurring'][0]['job']['question']='Changed after the result'
    with pytest.raises(ValueError,match='changed'):queue.schedule_recurring(spec,state,20000)
