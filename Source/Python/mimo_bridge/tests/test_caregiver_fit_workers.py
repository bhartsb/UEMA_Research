# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Coordinator fixtures exercise process scheduling, not model competence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tools import caregiver_fit_workers as workers
from mimo_bridge.tools import caregiver_learning_trial as trial
from mimo_bridge.tools import caregiver_fit_continuation as continuation


def test_yoked_tasks_require_actual_completed_donors_and_unique_ownership():
    assert workers.ready_tasks(4,{},set(),[1,0,3,2])==[
        ('responsive',0),('responsive',1),('responsive',2),('responsive',3)]
    ready=workers.ready_tasks(4,{'responsive_0':'done'},{'responsive_1'},[1,0,3,2])
    assert ready[0]==('yoked',1)
    assert ('yoked',0) not in ready and ('responsive',1) not in ready


def test_admission_waits_for_measured_startup_and_physical_reserves():
    enough=dict(ram_available_bytes=10*1024**3,gpu_free_bytes=3*1024**3)
    assert workers.admission(enough,[])[0]
    assert not workers.admission(enough,[dict(startup_measured=False)])[0]
    assert workers.admission(enough,[dict(startup_measured=True)])[0]
    assert not workers.admission(dict(enough,gpu_free_bytes=1900*1024**2),[])[0]
    assert not workers.admission(dict(enough,ram_available_bytes=5*1024**3),[])[0]


def fake_fit(folder,seed,dose=None):
    folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/'checkpoint';checkpoint.mkdir(exist_ok=True)
    values=dose if dose is not None else [[0,str(seed)]]
    tape=folder/'adult.json';tape.write_text(json.dumps(dict(dose=values)))
    data=dict(checkpoint=str(checkpoint),online_updates=100,offline_replay_updates=0,
        source_utterances=values,adult_tape=str(tape))
    (folder/'fit.json').write_text(json.dumps(data));return folder,data


def fake_completed(base,steps):
    found=[]
    for folder in [base,*sorted(base.glob('attempt_*'))]:
        if (folder/'fit.json').exists():found.append((folder,json.loads((folder/'fit.json').read_text())))
    assert len(found)<=1
    return found[0] if found else None


def setup_scheduler(monkeypatch,tmp_path,pause_once=False,pause_count=None):
    import psutil
    launched=[];instances=[];clock={'pid':10000,'peak':0,'time':100.};paused={}
    class ProcessIdentity:
        def __init__(self,pid=None):self.pid=pid
        def create_time(self):return 100.
        def is_running(self):return True
    class Child:
        def __init__(self,command,**kw):
            self.payload=json.loads(Path(command[-1]).read_text());self.folder=Path(self.payload['out'])
            self.segment=Path(self.payload.get('segment_out',self.folder))
            self.polls=0;self.code=None;clock['pid']+=1;self.pid=clock['pid'];instances.append(self)
            if self.payload['tape']:assert Path(self.payload['tape']).is_file()
            launched.append(self.payload)
            clock['peak']=max(clock['peak'],sum(x.code is None for x in instances))
        def poll(self):
            if self.code is not None:return self.code
            self.polls+=1
            (self.segment/'status.json').write_text(json.dumps(dict(stage='fit',step=100,segment_updates=100,steps=100,arm=self.folder.name)))
            (self.segment/'worker_resources.json').write_text(json.dumps(dict(copy_process_rss_bytes=1024**3)))
            if self.polls<3:return None
            key=self.payload['seed'],bool(self.payload['tape'])
            count=pause_count if pause_count is not None else int(pause_once)
            if count and paused.get(key,0)<count and not self.payload['tape']:
                paused[key]=paused.get(key,0)+1;self.code=75
                (self.folder/'partial_preserved.txt').write_text('This retained partial is not a completed fit.')
                (self.segment/'worker_result.json').write_text(json.dumps(dict(state='resource_paused')))
            else:
                dose=json.loads(Path(self.payload['tape']).read_text())['dose'] if self.payload['tape'] else None
                _,data=fake_fit(self.folder,self.payload['seed'],dose);self.code=0
                (self.segment/'worker_result.json').write_text(json.dumps(dict(state='complete',checkpoint=data['checkpoint'])))
            return self.code
    monkeypatch.setattr(psutil,'Process',ProcessIdentity)
    monkeypatch.setattr(workers.subprocess,'Popen',Child)
    monkeypatch.setattr(workers.time,'sleep',lambda seconds:clock.update(time=clock['time']+seconds))
    monkeypatch.setattr(workers.time,'time',lambda:clock['time'])
    monkeypatch.setattr(workers,'resources',lambda _:dict(timestamp=100,ram_available_bytes=20*1024**3,gpu_free_bytes=8*1024**3))
    monkeypatch.setattr(workers.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*1024**3))
    monkeypatch.setattr(trial,'completed_fit',fake_completed)
    args=SimpleNamespace(out=tmp_path,checkpoint=tmp_path/'parent',speech=tmp_path/'speech',device='cuda:0',
        worlds=2,seed=11,train_steps=100,fit_workers=2,fit_gpu_reserve_mib=1024,
        fit_worker_budget_mib=1024,fit_ram_reserve_gib=4.)
    return args,launched,clock


def test_two_workers_preserve_budget_device_and_donor_dose(monkeypatch,tmp_path):
    args,launched,clock=setup_scheduler(monkeypatch,tmp_path)
    checkpoints,paths=workers.run_fits(args,dict(rom_limit=1),[1,0])
    assert len(checkpoints)==len(paths)==4 and clock['peak']==2
    assert len({x['out'] for x in launched})==4
    assert all(x['parent']==str(args.checkpoint.resolve()) and x['device']=='cuda:0' and x['steps']==100 for x in launched)
    index=json.loads((tmp_path/'completed_fits.json').read_text())
    assert len(index)==4
    for child in launched:
        if child['tape']:
            donor=json.loads(Path(child['tape']).read_text())['dose']
            assert json.loads((Path(child['out'])/'fit.json').read_text())['source_utterances']==donor


def test_bounded_native_benchmark_subset_cannot_run_unrequested_yoked_fits(monkeypatch,tmp_path):
    args,launched,clock=setup_scheduler(monkeypatch,tmp_path)
    polls=[]
    checkpoints,_=workers.run_fits(args,{},[1,0],task_keys={'responsive_0','responsive_1'},
        on_poll=lambda snapshot,active:polls.append(active))
    assert set(checkpoints)=={('responsive',0),('responsive',1)}
    assert len(launched)==2 and all(row['tape'] is None for row in launched)
    assert polls and clock['peak']==2


def test_resume_reuses_completed_fit_and_preserves_partial_on_reserve(monkeypatch,tmp_path):
    args,launched,_=setup_scheduler(monkeypatch,tmp_path,pause_once=True)
    folder,_=fake_fit(tmp_path/'responsive_0',11)
    original=(folder/'fit.json').read_bytes()
    checkpoints,_=workers.run_fits(args,{},[1,0])
    assert len(checkpoints)==4 and (folder/'fit.json').read_bytes()==original
    assert not any(x['seed']==11 and x['tape'] is None for x in launched)
    assert (tmp_path/'responsive_1'/'partial_preserved.txt').is_file()
    assert (tmp_path/'responsive_1'/'attempt_002'/'fit.json').is_file()
    assert json.loads((tmp_path/'fit_coordinator.json').read_text())['effective_workers']==1


def test_guard_stops_on_lost_coordinator_before_more_learning(monkeypatch,tmp_path):
    monkeypatch.setattr(workers,'process_matches',lambda _:False)
    guard=workers.guard_for(dict(out=str(tmp_path),coordinator={},device='cpu'))
    with pytest.raises(workers.CopyResourcesUnavailable,match='Coordinator stopped'):guard(None,0)


def test_resource_pause_saves_partial_without_claiming_completed_fit(monkeypatch,tmp_path):
    import torch
    saved=[]
    class Model:
        encoder_registry=SimpleNamespace(state_dict=lambda:dict(x=torch.tensor(1.)))
        def save_checkpoint(self,step,companion_writer):
            folder=tmp_path/'checkpoints'/str(step);folder.mkdir(parents=True)
            companion_writer(folder,folder);return str(folder)
    source=SimpleNamespace(env=SimpleNamespace(caregiver=SimpleNamespace(bank=SimpleNamespace(groups={'x':['one','two']}))),
        reset=lambda:[],close=lambda:saved.append('closed'))
    monkeypatch.setattr(trial,'learner',lambda *args:(Model(),123))
    monkeypatch.setattr(trial,'source',lambda *args,**kw:source)
    monkeypatch.setattr(continuation,'binding',lambda *a:dict())
    monkeypatch.setattr(continuation,'save',lambda checkpoint,final,src,model,state,*args,**kw:saved.append(state))
    def pause(*args):raise workers.CopyResourcesUnavailable('Physical GPU reserve reached')
    with pytest.raises(workers.CopyResourcesUnavailable):
        trial.fit(SimpleNamespace(),tmp_path/'parent',tmp_path,11,100,dict(phase_name='mature'),step_guard=pause)
    receipt=json.loads((tmp_path/'paused_fit.json').read_text())
    assert receipt['completed_updates']==0 and not receipt['reusable_as_completed_fit']
    assert Path(receipt['checkpoint']).is_dir() and not (tmp_path/'fit.json').exists()
    assert saved[0]['step']==123 and saved[-1]=='closed'


def test_module_entry_uses_canonical_worker_exception_class(monkeypatch,tmp_path):
    import runpy
    import sys
    seen=[]
    def canonical(path):
        seen.append(path);return 75
    monkeypatch.setattr(workers,'worker',canonical)
    monkeypatch.setattr(sys,'argv',['caregiver_fit_workers','--worker',str(tmp_path/'spec.json')])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(workers.__file__,run_name='__main__')
    assert result.value.code==75 and seen==[tmp_path/'spec.json']


def test_resume_detects_orphan_child_not_yet_in_coordinator_journal(monkeypatch,tmp_path):
    orphan=tmp_path/'responsive_1'/'attempt_002';orphan.mkdir(parents=True)
    (orphan/'worker_process.json').write_text(json.dumps(dict(pid=321,process_created=42.)))
    monkeypatch.setattr(workers,'process_matches',lambda row:row.get('pid')==321)
    with pytest.raises(RuntimeError,match='still saving'):workers.assert_no_active_workers(tmp_path)
    monkeypatch.setattr(workers,'process_matches',lambda row:False)
    workers.assert_no_active_workers(tmp_path)
    assert (orphan/'worker_process.json').is_file()


def test_exclusive_study_ownership_covers_serial_and_parallel_launches(tmp_path):
    with workers.study_ownership(tmp_path):
        with pytest.raises(RuntimeError,match='owns this output'):
            with workers.study_ownership(tmp_path):pytest.fail('Concurrent study admitted')
    with workers.study_ownership(tmp_path):pass


def test_worker_checks_dead_coordinator_before_allocating_learner(monkeypatch,tmp_path):
    spec=tmp_path/'worker_spec.json';spec.write_text(json.dumps(dict(out=str(tmp_path),coordinator={},device='cpu')))
    monkeypatch.setattr(workers,'process_matches',lambda _:False)
    monkeypatch.setattr(trial,'fit',lambda *a,**kw:pytest.fail('Allocated copied learner after coordinator died'))
    assert workers.worker(spec)==75
    assert json.loads((tmp_path/'worker_process.json').read_text())['pid']>0
    assert json.loads((tmp_path/'worker_result.json').read_text())['state']=='resource_paused'


def test_pause_after_updates_preserves_exact_latest_world_orders_and_step(monkeypatch,tmp_path):
    import torch
    saved=[];weight=torch.tensor(1.)
    class Model:
        encoder_registry=SimpleNamespace(state_dict=lambda:dict(x=weight))
        updates=0
        def step(self,frames,*a,**kw):self.updates+=1;weight.add_(1);return SimpleNamespace(loss=.1)
        def save_checkpoint(self,step,companion_writer):
            folder=tmp_path/'checkpoints'/str(step);folder.mkdir(parents=True)
            companion_writer(folder,folder);return str(folder)
    model=Model();src=SimpleNamespace(env=SimpleNamespace(dt=.01,caregiver=SimpleNamespace(
        bank=SimpleNamespace(groups={'x':['one','two']}),state={})),position=0,
        reset=lambda:[],close=lambda:saved.append('closed'))
    class Orders:
        def __init__(self,*a):
            self.held=0;self.executed=0;self.order=0;self.bout=True;self.until=5
            self.rng=SimpleNamespace(bit_generator=SimpleNamespace(state={'seed':11}))
        def limits(self,*a):return {}
        def advance(self,model,result,source,index,*a):
            source.position=index+1;self.executed=source.position;self.held=index+10;self.order=index+1
            return [SimpleNamespace(sim_time=source.position*.05,meta={'order':self.order})]
    monkeypatch.setattr(trial,'learner',lambda *a:(model,123))
    monkeypatch.setattr(trial,'source',lambda *a,**kw:src)
    monkeypatch.setattr(trial,'Orders',Orders)
    monkeypatch.setattr(continuation,'binding',lambda *a:dict())
    monkeypatch.setattr(continuation,'save',lambda checkpoint,final,source,learner,state,*a,**kw:
        saved.append(dict(state,position=source.position,updates=learner.updates,weight=float(weight))))
    def guard(model,index):
        if index==2:raise workers.CopyResourcesUnavailable('reserve')
    with pytest.raises(workers.CopyResourcesUnavailable):
        trial.fit(SimpleNamespace(action_repeat=5,device='cpu'),tmp_path/'parent',tmp_path,11,100,
            dict(phase_name='mature'),step_guard=guard)
    receipt=json.loads((tmp_path/'paused_fit.json').read_text())
    assert receipt['completed_updates']==2 and Path(receipt['checkpoint']).name=='125'
    meta=saved[0].pop('fit_resume');assert meta['contract']==continuation.CONTRACT
    assert meta['initial_encoder_sha256'] and meta['utterances']==[]
    assert saved[0]==dict(step=125,trial_step=2,held=11,executed=2,order=2,bout=True,until=5,frame_meta=[{'order':2}],
        bout_rng={'seed':11},position=2,updates=2,weight=3.)
    assert saved[-1]=='closed' and not (tmp_path/'fit.json').exists()


def test_resume_with_a_completed_yoked_copy_keeps_full_task_count(monkeypatch,tmp_path):
    args,launched,_=setup_scheduler(monkeypatch,tmp_path)
    _,donor=fake_fit(tmp_path/'responsive_0',11)
    fake_fit(tmp_path/'yoked_1',12,donor['source_utterances'])
    checkpoints,_=workers.run_fits(args,{},[1,0])
    assert len(checkpoints)==4 and len(launched)==2
    assert {(x['seed'],bool(x['tape'])) for x in launched}=={(12,False),(11,True)}


def test_segmented_orphan_is_detected_and_old_stop_requests_are_preserved(monkeypatch,tmp_path):
    folder=tmp_path/'responsive_0';folder.mkdir()
    (folder/'stop_requested.json').write_text('old retained stop')
    old=continuation.segment(folder);(old/'worker_process.json').write_text(json.dumps(dict(pid=777)))
    newer=continuation.segment(folder)
    assert old!=newer and not (newer/'stop_requested.json').exists()
    assert (folder/'stop_requested.json').read_text()=='old retained stop'
    monkeypatch.setattr(workers,'process_matches',lambda row:row.get('pid')==777)
    with pytest.raises(RuntimeError,match='still saving'):workers.assert_no_active_workers(tmp_path)


def test_coordinator_resumes_eligible_partial_in_same_attempt_with_a_new_process_segment(monkeypatch,tmp_path):
    args,launched,_=setup_scheduler(monkeypatch,tmp_path)
    folder=tmp_path/'responsive_0';checkpoint=folder/'checkpoints'/'partial';checkpoint.mkdir(parents=True)
    (checkpoint/'fit_continuation.json').write_text(json.dumps(dict(trial_step=60)))
    old=continuation.segment(folder);(old/'worker_result.json').write_text('retained pause receipt')
    (old/'stop_requested.json').write_text('retained old control')
    monkeypatch.setattr(continuation,'latest',lambda base,expected:(folder,checkpoint) if base==folder else None)
    checkpoints,_=workers.run_fits(args,{},[1,0])
    resumed=[row for row in launched if row['resume_checkpoint']]
    assert len(checkpoints)==4 and len(resumed)==1
    assert resumed[0]['out']==str(folder.resolve()) and resumed[0]['resume_checkpoint']==str(checkpoint.resolve())
    assert resumed[0]['steps']==100 and Path(resumed[0]['segment_out'])!=old
    assert (old/'worker_result.json').read_text()=='retained pause receipt'
    assert (old/'stop_requested.json').read_text()=='retained old control'
    assert not (folder/'attempt_002').exists()


def test_four_resource_pauses_wait_and_retry_instead_of_killing_study(monkeypatch,tmp_path):
    args,launched,clock=setup_scheduler(monkeypatch,tmp_path,pause_count=4)
    checkpoints,_=workers.run_fits(args,{},[1,0])
    assert len(checkpoints)==4
    journal=json.loads((tmp_path/'fit_coordinator.json').read_text())
    assert journal['resource_backoff']['responsive_0']['streak']==4
    assert clock['time']>=100+15+30+60+120
    assert len(launched)==12  # four retained incomplete fixture attempts per responsive world


def test_pause_backoff_only_resets_when_completed_updates_increase():
    prior=workers.pause_backoff({},40,10)
    assert prior['streak']==1
    for _ in range(4):prior=workers.pause_backoff(prior,40,prior['next_admission_at'])
    assert prior['streak']==5
    advanced=workers.pause_backoff(prior,41,1000)
    assert advanced['streak']==1 and advanced['next_admission_at']==1015


def test_recording_guard_pauses_before_consuming_checkpoint_reserve(monkeypatch,tmp_path):
    monkeypatch.setattr(workers,'process_matches',lambda _:True)
    monkeypatch.setattr(workers,'resources',lambda _:dict(ram_available_bytes=10*1024**3))
    monkeypatch.setattr(workers.shutil,'disk_usage',lambda _:SimpleNamespace(free=4.25*1024**3))
    guard=workers.guard_for(dict(out=str(tmp_path),coordinator={},device='cpu',ram_reserve_gib=4,gpu_reserve_mib=1024))
    with pytest.raises(workers.CopyResourcesUnavailable,match='next copied checkpoint'):guard(None,0)


def test_unexpected_disk_race_waits_without_learning_or_spending_protected_reserve(monkeypatch,tmp_path):
    free={'bytes':4.1*1024**3};waits=[]
    monkeypatch.setattr(workers.shutil,'disk_usage',lambda _:SimpleNamespace(free=free['bytes']))
    def wait(seconds):waits.append(seconds);free['bytes']=4.6*1024**3
    monkeypatch.setattr(workers.time,'sleep',wait)
    guard=workers.guard_for(dict(out=str(tmp_path),device='cpu'))
    guard.before_save(2999)
    status=json.loads((tmp_path/'status.json').read_text())
    assert waits==[5] and status['stage']=='waiting_to_save' and status['completed_updates']==2999
    assert status['free_bytes']<status['required_free_bytes']


def test_live_read_retries_a_windows_sharing_race(monkeypatch,tmp_path):
    path=tmp_path/'status.json';path.write_text('{"step":1610}')
    original=Path.read_text;calls=[]
    def read(candidate,*args,**kwargs):
        calls.append(candidate)
        if len(calls)==1:raise PermissionError(13,'File is being replaced',str(candidate))
        return original(candidate,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(workers.time,'sleep',lambda _:None)
    value,fresh,error=workers.read_live_json(path,{'step':1601})
    assert value=={'step':1610} and fresh and error is None and len(calls)==2


@pytest.mark.parametrize('failure',[PermissionError(13,'Sharing violation'),FileNotFoundError(),ValueError('Bad JSON')])
def test_optional_telemetry_retains_cached_progress_but_cannot_admit_another_worker(monkeypatch,tmp_path,failure):
    def read(*args,**kwargs):raise failure
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(workers.time,'sleep',lambda _:None)
    value,fresh,error=workers.read_live_json(tmp_path/'status.json',{'step':1601})
    assert value=={'step':1601} and not fresh and type(failure).__name__ in error
    assert not workers.admission({'ram_available_bytes':20*1024**3},[{'startup_measured':fresh}])[0]


@pytest.mark.parametrize('telemetry',['status.json','worker_resources.json'])
def test_optional_status_io_failure_does_not_stop_learning_or_lose_completed_fits(monkeypatch,tmp_path,telemetry):
    args,launched,clock=setup_scheduler(monkeypatch,tmp_path)
    original=Path.read_text;failures=[]
    def read(candidate,*pos,**kw):
        if candidate.name==telemetry and 'segments' in candidate.parts:
            failures.append(str(candidate));raise PermissionError(13,'Concurrent replacement',str(candidate))
        return original(candidate,*pos,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    checkpoints,_=workers.run_fits(args,{},[1,0])
    assert failures and len(checkpoints)==4 and len(launched)==4
    assert clock['peak']==1  # stale telemetry never authorizes another copy
    assert not list(tmp_path.glob('**/stop_requested.json'))
    assert len(json.loads((tmp_path/'completed_fits.json').read_text()))==4


@pytest.mark.parametrize('contents',['{"step":','[]'])
def test_malformed_optional_json_is_visible_as_stale(monkeypatch,tmp_path,contents):
    path=tmp_path/'status.json';path.write_text(contents)
    monkeypatch.setattr(workers.time,'sleep',lambda _:None)
    value,fresh,error=workers.read_live_json(path,{'step':1601})
    assert value=={'step':1601} and not fresh and error


def test_required_completion_receipt_read_stays_strict(monkeypatch,tmp_path):
    args,_,_=setup_scheduler(monkeypatch,tmp_path)
    original=Path.read_text
    def read(candidate,*pos,**kw):
        if candidate.name=='worker_result.json':raise PermissionError(13,'Required receipt unreadable')
        return original(candidate,*pos,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(PermissionError,match='Required receipt unreadable'):
        workers.run_fits(args,{},[1,0])
    assert json.loads((tmp_path/'completed_fits.json').read_text())=={}
