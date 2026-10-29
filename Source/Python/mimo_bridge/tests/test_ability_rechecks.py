# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from pathlib import Path
import json
import time

import psutil
import pytest
from mimo_bridge.tools import ability_rechecks as worker
from mimo_bridge.tools.ability_rechecks import due_checkpoint, confined


def test_missed_milestone_runs_first_complete_checkpoint_and_failed_does_not_advance():
    task=dict(first_step=100,every_steps=30)
    candidates=[(99,Path('old')),(105,Path('first')),(140,Path('later'))]
    assert due_checkpoint(task,{},candidates)==(105,Path('first'))
    assert due_checkpoint(task,dict(attempt_step=105,status='failed'),candidates)==(105,Path('first'))
    assert due_checkpoint(task,dict(completed_step=105),candidates)==(140,Path('later'))
    assert due_checkpoint(task,dict(completed_step=140),candidates) is None


def test_manifest_paths_cannot_leave_workspace(tmp_path):
    assert confined(tmp_path,'docs/probe.py')==tmp_path/'docs/probe.py'
    with pytest.raises(ValueError):confined(tmp_path,'../other/probe.py')


@pytest.fixture
def tiny_worker(tmp_path,monkeypatch):
    """Real bounded children, fake resource readings; no learner or GPU is loaded."""
    from mimo_bridge.tools import commit_guard
    script=tmp_path/'qualify_hand_execution_2026-09-24.py'
    script.write_text('''import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--out');a=p.parse_args()
out=Path(a.out);out.mkdir()
(out/'started').write_text('ready')
for _ in range(1000):
    if (out/'release').exists():break
    time.sleep(.01)
else:raise RuntimeError('test child timed out')
(out/'result.json').write_text(json.dumps({'checkpoint':a.checkpoint,'passed':False,
 'learning':False,'elapsed_s':1,'goals':['curl','extension','index_selective'],
 'modes':['fixed_world','moving_wrist'],'comparisons':24,'exact_replay':True,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'summary':{g:{m:1. for m in ['fixed_world','moving_wrist']} for g in ['curl','extension','index_selective']}}))
''',encoding='utf-8')
    run=tmp_path/'Saved/mimo_bridge/experiments/declared_test'
    checkpoint=run/'checkpoints/ckpt_step_105';checkpoint.mkdir(parents=True)
    continuation=tmp_path/'Saved/mimo_bridge/training_continuation.json'
    continuation.write_text(json.dumps({'active_run':str(run)}))
    manifest=tmp_path/'manifest.json'
    task=dict(id='tiny',title='Bounded hand test',script=script.name,first_step=100,every_steps=30,memory_gib=1)
    manifest.write_text(json.dumps(dict(lineage='declared',tasks=[task])))
    folder=tmp_path/'records';folder.mkdir()
    monkeypatch.setattr(worker,'complete_candidates',lambda _: [checkpoint])
    monkeypatch.setattr(commit_guard,'commit_reading',lambda:(1,100,99))
    monkeypatch.setattr(worker,'trainer_running',lambda:True)
    # every task here has its test registry row; the registry check itself is tested in test_test_registry.py
    monkeypatch.setattr(worker,'registry_refusals',lambda root,ids:{})
    info=dict(root=tmp_path,manifest=manifest,folder=folder,script=script,checkpoint=checkpoint)
    yield info
    state=worker.read(folder/'status.json')
    record=state.get('tasks',{}).get('tiny',{})
    children,_=worker.matching_children(record)
    for child in children:
        try:
            process=psutil.Process(child['pid'])
            descendants=process.children(recursive=True)
            for descendant in descendants:descendant.kill()
            process.kill()
            psutil.wait_procs(descendants+[process],timeout=2)
        except psutil.NoSuchProcess:pass


def tick(info):
    return worker.cycle(info['root'],info['manifest'],info['folder'])


def until(info,predicate):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        state=tick(info)
        if predicate(state):return state
        time.sleep(.02)
    raise AssertionError('The bounded fake child did not reach its declared state')


def test_running_child_keeps_heartbeat_and_survives_worker_restart_without_duplicate(tiny_worker):
    first=tick(tiny_worker)
    record=first['tasks']['tiny'];child=record['child'];output=Path(record['output'])
    assert child['create_time'] and child['command']==record['command']
    again=tick(tiny_worker)
    assert again['timestamp']>=first['timestamp']
    assert again['tasks']['tiny']['attempts']==1
    assert again['tasks']['tiny']['child']['pid']==child['pid']
    # Simulate a worker dying after Popen but before it saves the child identity.
    again['tasks']['tiny'].pop('child')
    again['tasks']['tiny']['status']='starting'
    worker.atomic_json(tiny_worker['folder']/'status.json',again)
    recovered=tick(tiny_worker)
    assert recovered['tasks']['tiny']['attempts']==1
    assert recovered['tasks']['tiny']['status']=='running'
    assert recovered['tasks']['tiny']['child']['command'][1:]==child['command'][1:]
    until(tiny_worker,lambda _:output.is_dir())
    (output/'release').write_text('finish')
    finished=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='complete')
    assert finished['tasks']['tiny']['completed_step']==105
    assert finished['tasks']['tiny']['outcome'] is False  # a measured failed ability is a completed test
    assert output.with_suffix('.receipt.json').is_file()
    assert tick(tiny_worker)['tasks']['tiny']['attempts']==1


def test_existing_complete_result_is_adopted_after_parent_and_child_exit(tiny_worker):
    first=tick(tiny_worker);record=first['tasks']['tiny'];output=Path(record['output'])
    until(tiny_worker,lambda _:output.is_dir())
    (output/'release').write_text('finish')
    # No worker monitoring is needed while the child finishes.
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        children,_=worker.matching_children(record)
        if not children:break
        time.sleep(.02)
    recovered=tick(tiny_worker)
    assert recovered['tasks']['tiny']['status']=='complete'
    assert recovered['tasks']['tiny']['attempts']==1


def test_two_failed_children_leave_durable_receipts_then_wait_for_code_change(tiny_worker):
    tiny_worker['script'].write_text('raise RuntimeError("declared test failure")\n')
    tick(tiny_worker)
    first=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')
    assert first['tasks']['tiny']['attempts']==1
    tick(tiny_worker)
    second=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')
    assert second['tasks']['tiny']['attempts']==2
    blocked=tick(tiny_worker)
    assert blocked['tasks']['tiny']['status']=='repair needed'
    assert 'completed_step' not in blocked['tasks']['tiny']
    assert len(blocked['tasks']['tiny']['history'])==2
    assert len(list(tiny_worker['folder'].glob('*.receipt.json')))==2
    tiny_worker['script'].write_text('raise RuntimeError("changed test failure")\n')
    changed=tick(tiny_worker)
    assert changed['tasks']['tiny']['attempts']==1
    assert changed['tasks']['tiny']['status'] in ('starting','running')


def test_pause_blocks_new_launch_but_still_reconciles_an_existing_child(tiny_worker):
    control=tiny_worker['root']/'Saved/mimo_bridge/supervisor_control.json'
    control.write_text(json.dumps({'cmd':'pause'}))
    assert tick(tiny_worker)['state']=='production paused for repair'
    assert not list(tiny_worker['folder'].glob('*.process.json'))
    control.write_text(json.dumps({'cmd':'run'}))
    started=tick(tiny_worker);output=Path(started['tasks']['tiny']['output'])
    control.write_text(json.dumps({'cmd':'pause'}))
    assert tick(tiny_worker)['tasks']['tiny']['attempts']==1
    until(tiny_worker,lambda _:output.is_dir())
    (output/'release').write_text('finish')
    finished=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='complete')
    assert finished['tasks']['tiny']['completed_step']==105


def test_an_unlaunchable_entry_is_marked_for_repair_and_never_pauses_training_or_blocks_others(tiny_worker,monkeypatch):
    """A manifest entry whose extra input is missing (path_arguments, 2026-09-24) must not request a
    checkpoint or a memory window it can never use, and must not stop the other comparisons."""
    from mimo_bridge.tools import ability_comparison_window
    requested=[]
    monkeypatch.setattr(ability_comparison_window,'request',lambda *args:requested.append(args) or False)
    manifest=json.loads(tiny_worker['manifest'].read_text())
    broken=dict(manifest['tasks'][0],id='needs_input',path_arguments={'--calibration':'missing/calibration.json'})
    manifest['tasks'].insert(0,broken)
    tiny_worker['manifest'].write_text(json.dumps(manifest))
    state=tick(tiny_worker)
    assert state['tasks']['needs_input']['status']=='repair needed'
    assert 'missing file' in state['tasks']['needs_input']['reason']
    assert 'attempts' not in state['tasks']['needs_input'] and not requested
    assert not (tiny_worker['checkpoint'].parents[1]/'control.json').exists()
    assert state['tasks']['tiny']['status'] in ('starting','running'), 'the other comparison must still launch'
    command=state['tasks']['tiny']['command']
    assert command[3:]==['--checkpoint',str(tiny_worker['checkpoint']),'--out',state['tasks']['tiny']['output']]


def test_reused_pid_is_not_accepted_as_the_original_child(monkeypatch):
    class Reused:
        pid=91
        def create_time(self):return 201
        def cmdline(self):return ['python','old.py']
        def status(self):return 'running'
    monkeypatch.setattr(worker.psutil,'Process',lambda _:Reused())
    monkeypatch.setattr(worker.psutil,'process_iter',lambda _ :[])
    record=dict(command=['python','old.py'],started_at=100,
                child=dict(pid=91,create_time=101,command=['python','old.py']))
    assert worker.matching_children(record)==([],None)


def test_interrupted_child_is_recorded_before_a_fresh_second_attempt(tiny_worker):
    started=tick(tiny_worker);record=started['tasks']['tiny'];output=Path(record['output'])
    until(tiny_worker,lambda _:output.is_dir())
    process=psutil.Process(record['child']['pid'])
    children=process.children(recursive=True)
    for child in children:child.kill()
    process.kill()
    psutil.wait_procs(children+[process],timeout=2)
    failed=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')
    assert failed['tasks']['tiny']['attempts']==1
    assert output.with_suffix('.receipt.json').is_file()
    restarted=tick(tiny_worker)
    assert restarted['tasks']['tiny']['attempts']==2
    assert restarted['tasks']['tiny']['output']!=str(output)


def test_unreadable_saved_process_identity_blocks_duplicate_launch(monkeypatch):
    monkeypatch.setattr(worker.psutil,'Process',lambda pid:(_ for _ in ()).throw(psutil.AccessDenied(pid)))
    state=dict(tasks={'tiny':dict(status='running',command=['python','probe.py'],started_at=100,
                                 child=dict(pid=91,create_time=101,command=['python','probe.py']))})
    worker.reconcile_task(state,'tiny')
    assert state['state']=='waiting for process identity'
    assert state['tasks']['tiny']['status']=='process identity unavailable'


def test_result_for_another_checkpoint_is_rejected(tmp_path):
    output=tmp_path/'attempt';output.mkdir()
    (output/'result.json').write_text(json.dumps({'checkpoint':str(tmp_path/'wrong'),'passed':True}))
    result=worker.finished_result(dict(output=str(output),checkpoint=str(tmp_path/'right')))
    assert result['status']=='failed'
    assert 'checkpoint differs' in result['reason']


@pytest.mark.parametrize('stop_kind',['repair_required','consumed_stop','archiver_only'])
def test_all_supervisor_stop_reasons_prevent_new_probe(tiny_worker,stop_kind):
    run=tiny_worker['checkpoint'].parent.parent
    if stop_kind=='repair_required':
        (run/'stop_reason.json').write_text(json.dumps({'resume_policy':'repair_required','reason':'Control repair'}))
    elif stop_kind=='consumed_stop':
        (run/'control_done_test.json').write_text(json.dumps({'cmd':'snapshot_and_stop','auto_resume':False}))
    else:
        (tiny_worker['root']/'Saved/mimo_bridge/supervisor_control.json').write_text(json.dumps({'archiver':'off'}))
    status=tick(tiny_worker)
    assert status['state']=='production paused for repair'
    assert status['pause_reason']
    assert not list(tiny_worker['folder'].glob('*.process.json'))


@pytest.mark.parametrize('payload',[
    {'message':'some result'},
    {'checkpoint':'MATCH','learning':False,'elapsed_s':1,'passed':True},
])
def test_partial_or_identity_free_results_do_not_complete(tmp_path,payload):
    output=tmp_path/'attempt';output.mkdir()
    checkpoint=tmp_path/'checkpoint'
    if payload.get('checkpoint')=='MATCH':payload['checkpoint']=str(checkpoint)
    (output/'result.json').write_text(json.dumps(payload))
    record=dict(output=str(output),checkpoint=str(checkpoint),script_sha256='test',
                command=['python','-u','qualify_hand_execution_2026-09-24.py'])
    assert worker.finished_result(record)['status']=='failed'


def test_real_completed_measurements_satisfy_script_contracts():
    # Read only: schema validation of the actual saved comparisons, no model load.
    root=Path(__file__).resolve().parents[4]
    found=0
    scripts={
        'self_touch_choices':'probe_saved_self_touch_choices.py',
        'hand_execution':'qualify_hand_execution_2026-09-24.py',
    }
    for directory,script in scripts.items():
        paths=list((root/'Saved/gof_takeover_20260924').glob(directory+'*/result.json'))
        for path in paths:
            payload=json.loads(path.read_text())
            if script=='probe_saved_self_touch_choices.py' and not payload.get('script_sha256'):
                continue  # historical v1 predates scheduler-bound output identity
            record=dict(checkpoint=payload['checkpoint'],script_sha256=payload.get('script_sha256'),
                        command=['python','-u',script])
            worker.validate_result(payload,record)
            found+=1
    if not found:pytest.skip('No retained private comparison receipt in this checkout')


def test_a_memory_window_covers_the_checkpoint_the_task_waits_for(tiny_worker,monkeypatch):
    """The live step read from the statistics lags the requested snapshot; a window asked for
    below the task's checkpoint lists no task and releases before running it (2026-09-24)."""
    from mimo_bridge.tools import ability_comparison_window,commit_guard
    seen={}
    def fake_request(root,folder,state,run,supervisor,minimum_step):
        seen['minimum']=minimum_step
        return False
    monkeypatch.setattr(ability_comparison_window,'request',fake_request)
    monkeypatch.setattr(commit_guard,'commit_reading',lambda:(99,100,1))
    run=tiny_worker['checkpoint'].parents[1]
    (run/'replay_training_stats.jsonl').write_text(json.dumps(dict(type='mimo',step=100))+'\n',encoding='utf-8')
    result=tick(tiny_worker)
    assert result['tasks']['tiny']['status']=='waiting for memory'
    assert seen['minimum']>=105, 'the window must reach the checkpoint (105) the task waits for'


def test_no_test_starts_between_a_released_window_and_the_trainers_relaunch(tiny_worker,monkeypatch):
    """With no window and no trainer, the machine only looks free: the supervisor is about to
    relaunch training (the gaze test ran beside it for 14 minutes on 2026-09-24)."""
    monkeypatch.setattr(worker,'trainer_running',lambda:False)
    result=tick(tiny_worker)
    assert result['tasks']['tiny']['status']=='waiting for the supervisor to relaunch training'
    assert not list(tiny_worker['folder'].glob('*.process.json')), 'a test started in the relaunch gap'


def test_saved_window_memory_shortfall_becomes_bounded_repair(tiny_worker,monkeypatch):
    from mimo_bridge.tools import ability_comparison_window,commit_guard
    state=dict(tasks={'tiny':{}},comparison_window=dict(state='ready',ready_at=time.time()-301,
               checkpoint=str(tiny_worker['checkpoint'])))
    worker.atomic_json(tiny_worker['folder']/'status.json',state)
    monkeypatch.setattr(ability_comparison_window,'advance',lambda *args:True)
    monkeypatch.setattr(commit_guard,'commit_reading',lambda:(99,100,1))
    result=tick(tiny_worker)
    assert result['tasks']['tiny']['status']=='memory capacity needs repair'
    assert result['tasks']['tiny']['memory_window_step']==105
    assert not list(tiny_worker['folder'].glob('*.process.json'))


@pytest.mark.parametrize('digest',[None,'wrong'])
def test_self_touch_result_requires_matching_script_identity(tmp_path,digest):
    record=dict(checkpoint=str(tmp_path/'checkpoint'),script_sha256='expected',
                command=['python','-u','probe_saved_self_touch_choices.py'])
    payload=dict(checkpoint=record['checkpoint'],learning=False,elapsed_s=1,script_sha256=digest)
    with pytest.raises(ValueError,match='script identity'):worker.validate_result(payload,record)


def test_eye_result_requires_all_starts_horizons_and_verified_branches(tmp_path):
    import copy
    row=dict(bodies=8,candidate_mean=1.,comparison_mean=2.,paired95_difference=[-1.4,-.6])
    record=dict(checkpoint=str(tmp_path/'checkpoint'),script_sha256='expected',
                command=['python','-u','probe_saved_gaze.py'])
    payload=dict(checkpoint=record['checkpoint'],script_sha256='expected',learning=False,elapsed_s=1.,
        bodies=8,evaluated_bodies=8,commands=['hold','left','right','up','down'],starts_per_body=2,
        command_branches=80,completed_command_branches=80,replays_per_branch=2,exact_replay=True,
        all_candidate_traces_replayed_exactly=True,weights_unchanged=True,gaze_reflex=False,
        spontaneous_saccades=False,helper_sha256={'helper':'hash'},source_sha256={'source':'hash'},
        runtime_sha256='hash',episode={'worlds':8},summary={start:{horizon:{metric:dict(row)
            for metric in ('eye_prediction_error_degrees','visual_prediction_mse')}
            for horizon in ('50ms','100ms','150ms')} for start in ('saved_starts','after_head_command')})
    worker.validate_result(payload,record)
    for key,value in [('completed_command_branches',79),('weights_unchanged',False),
                      ('script_sha256','another script'),('gaze_reflex',True)]:
        bad=copy.deepcopy(payload);bad[key]=value
        with pytest.raises(ValueError):worker.validate_result(bad,record)
    bad=copy.deepcopy(payload);del bad['summary']['after_head_command']['150ms']
    with pytest.raises(ValueError,match='means'):worker.validate_result(bad,record)


def test_two_failures_of_one_test_version_hold_it_across_checkpoints_until_a_declared_file_changes(tiny_worker,monkeypatch):
    """2026-09-24 9:12 PM: the eye test failed twice at 397,050; after the resume its next checkpoint
    was a new candidate, the per-checkpoint count restarted at 0 and training stopped again for the
    same failing code. Failures now count per test version: the script and its declared files."""
    helper=tiny_worker['root']/'helper.py';helper.write_text('# probe helper v1\n',encoding='utf-8')
    manifest=json.loads(tiny_worker['manifest'].read_text())
    manifest['tasks'][0]['depends']=['helper.py']
    tiny_worker['manifest'].write_text(json.dumps(manifest))
    tiny_worker['script'].write_text('raise RuntimeError("declared test failure")\n')
    tick(tiny_worker)
    first=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')
    assert first['tasks']['tiny']['failures_in_a_row']==1
    # the next resume writes into a new run folder: only a later checkpoint is a candidate
    later=tiny_worker['checkpoint'].parent/'ckpt_step_140';later.mkdir()
    monkeypatch.setattr(worker,'complete_candidates',lambda _: [later])
    tick(tiny_worker)
    second=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')
    record=second['tasks']['tiny']
    assert record['attempt_step']==140 and record['attempts']==1 and record['failures_in_a_row']==2
    assert json.loads(Path(record['output']).with_suffix('.receipt.json').read_text())['failures_in_a_row']==2
    even_later=tiny_worker['checkpoint'].parent/'ckpt_step_170';even_later.mkdir()
    monkeypatch.setattr(worker,'complete_candidates',lambda _: [even_later])
    held=tick(tiny_worker)
    assert held['tasks']['tiny']['status']=='repair needed' and 'step 140' in held['tasks']['tiny']['reason']
    assert held['tasks']['tiny']['attempt_step']==140, 'no third attempt of the same version'
    # a repair in a declared file (the harness, not the script) is a new version
    helper.write_text('# probe helper v2: the fix\n',encoding='utf-8')
    retried=tick(tiny_worker)
    assert retried['tasks']['tiny']['status'] in ('starting','running')
    assert retried['tasks']['tiny']['attempt_step']==170


def test_a_missing_declared_dependency_is_marked_for_repair_and_never_launches(tiny_worker):
    manifest=json.loads(tiny_worker['manifest'].read_text())
    manifest['tasks'][0]['depends']=['no_such_helper.py']
    tiny_worker['manifest'].write_text(json.dumps(manifest))
    state=tick(tiny_worker)
    assert state['tasks']['tiny']['status']=='repair needed'
    assert 'dependency is missing' in state['tasks']['tiny']['reason'] and 'attempts' not in state['tasks']['tiny']


def test_the_production_manifest_declares_the_probe_harness_for_every_task():
    root=Path(__file__).resolve().parents[4]
    manifest=json.loads((root/'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    for task in manifest['tasks']:
        depends=task.get('depends') or []
        assert 'Source/Python/mimo_bridge/tools/gof_probe_harness.py' in depends, task['id']
        assert worker.dependency_paths(root,task), task['id']


# ------------------------------------------------ plain manifest arguments (2026-09-25)
@pytest.mark.parametrize('arguments,reason',[
    ('--goal-suite hand_and_body','list of plain words'),
    (['--goal-suite',3],'list of plain words'),
    (['hand_and_body'],'plain value after an option'),
    (['--goal-suite','../outside'],'plain value'),
    (['--goal-suite','docs/x.json'],'plain value'),
    (['--goal-suite','C:\\x'],'plain value'),
    (['--goal-suite','a..b'],'plain value'),
    (['--goal-suite=hand'],'plain long option'),
    (['-g','hand'],'plain long option'),
    (['--checkpoint','x'],'plain long option'),
    (['--out','x'],'plain long option'),
    (['--calibration','x'],'plain long option'),
    (['--goal-suite','a','--goal-suite','b'],'plain long option'),
])
def test_plain_arguments_refuse_paths_and_the_options_the_scheduler_or_a_path_argument_owns(arguments,reason):
    task=dict(arguments=arguments,path_arguments={'--calibration':'docs/calibration.json'})
    with pytest.raises(ValueError,match=reason):worker.plain_arguments(task)


def test_plain_arguments_follow_the_checkpoint_and_output_and_precede_the_path_arguments(tmp_path):
    (tmp_path/'probe.py').write_text('',encoding='utf-8');(tmp_path/'cal.json').write_text('{}',encoding='utf-8')
    task=dict(script='probe.py',arguments=['--goal-suite','hand_and_body','--bodies','8','--compare-execution-contracts'],
              path_arguments={'--calibration':'cal.json'})
    assert worker.plain_arguments(task)==task['arguments']
    command=worker.task_command(tmp_path,task,'ckpt','out')
    assert command[3:]==['--checkpoint','ckpt','--out','out','--goal-suite','hand_and_body','--bodies','8',
                         '--compare-execution-contracts','--calibration',str((tmp_path/'cal.json').resolve())]
    assert worker.task_command(tmp_path,dict(script='probe.py'),'ckpt','out')[3:]==['--checkpoint','ckpt','--out','out']
    assert worker.plain_arguments(dict(script='probe.py'))==[]


def test_a_changed_plain_argument_is_a_new_test_version_and_a_task_without_them_keeps_its_version(tmp_path):
    import hashlib
    helper=tmp_path/'helper.py';helper.write_text('# v1\n',encoding='utf-8')
    task=dict(depends=['helper.py'])
    before=hashlib.sha256(b'abc');before.update(b'helper.py');before.update(hashlib.sha256(helper.read_bytes()).digest())
    assert worker.test_version(tmp_path,task,'abc')==before.hexdigest(), 'a task without arguments changed version'
    suite=worker.test_version(tmp_path,dict(task,arguments=['--goal-suite','hand_and_body']),'abc')
    hand=worker.test_version(tmp_path,dict(task,arguments=['--goal-suite','hand']),'abc')
    assert len({before.hexdigest(),suite,hand})==3


def test_the_child_receives_the_plain_arguments(tiny_worker):
    tiny_worker['script'].write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--out');p.add_argument('--goal-suite')
a=p.parse_args();out=Path(a.out);out.mkdir();(out/'argv.json').write_text(json.dumps(vars(a)))
''',encoding='utf-8')
    manifest=json.loads(tiny_worker['manifest'].read_text())
    manifest['tasks'][0]['arguments']=['--goal-suite','hand_and_body']
    tiny_worker['manifest'].write_text(json.dumps(manifest))
    record=tick(tiny_worker)['tasks']['tiny']
    assert record['command'][3:]==['--checkpoint',str(tiny_worker['checkpoint']),'--out',record['output'],
                                   '--goal-suite','hand_and_body']
    received=Path(record['output'])/'argv.json'
    until(tiny_worker,lambda state:received.is_file())
    assert json.loads(received.read_text())['goal_suite']=='hand_and_body'


def test_an_entry_whose_arguments_are_not_plain_is_marked_for_repair_and_never_launches(tiny_worker):
    manifest=json.loads(tiny_worker['manifest'].read_text())
    manifest['tasks'][0]['arguments']=['--goal-suite','../outside']
    tiny_worker['manifest'].write_text(json.dumps(manifest))
    state=tick(tiny_worker)
    assert state['tasks']['tiny']['status']=='repair needed' and 'plain value' in state['tasks']['tiny']['reason']
    assert 'attempts' not in state['tasks']['tiny']
    assert not list(tiny_worker['folder'].glob('*.process.json'))


def test_the_production_manifest_runs_the_body_comparison_on_every_body_goal_and_declares_its_helpers():
    root=Path(__file__).resolve().parents[4]
    manifest=json.loads((root/'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    tasks={t['id']:t for t in manifest['tasks']}
    body=tasks['body_and_synergy']
    assert worker.plain_arguments(body)==['--goal-suite','hand_and_body']
    assert {'Source/Python/mimo_bridge/tools/command_ranking.py','docs/gain_of_function_v3/compare_synergy_physical.py',
            'docs/gain_of_function_v3/audit_hand_target_conflicts.py'}<=set(body['depends'])
    assert 'Source/Python/mimo_bridge/tools/command_ranking.py' in tasks['planner_candidates']['depends']
    for task in manifest['tasks']:
        worker.plain_arguments(task)


# ------------------------------------------------ the body comparison's result, with the ranking (2026-09-25)
def body_payload(tmp_path,suite='hand_and_body'):
    import numpy as np
    from mimo_bridge.tools.command_ranking import ranking
    goals=worker.BODY_GOAL_SUITES[suite]
    rng=np.random.default_rng(0)

    def block():
        rows=[dict(body=b,names=['controller','direct','synergy'],predicted=rng.normal(size=3).tolist(),
                   actual=rng.normal(size=3).tolist()) for b in range(8) for _ in range(2)]
        return ranking(rows,'x',planners=('direct','synergy'),reference='controller',draws=200)
    metric=dict(means=dict(controller=1.,direct=1.1,synergy=1.2),paired_bootstrap95=[-.1,.2])
    check=dict(bodies=8,expected_bodies=8,informative_bodies=5,unit='degrees',parts=['chest'],requested_raw=11.,
               requested_executed=3.9,achieved_beyond_controller=1.,achieved_share=.26,
               separation_from_controller=1.2,moved_from_start=2.)
    payload=dict(checkpoint=str(tmp_path/'checkpoint'),learning=False,elapsed_s=10.,bodies=8,goals=goals,goal_suite=suite,
                 weights_unchanged=True,exact_replay=True,script_sha256='expected',
                 summary={m:dict(metric) for m in ('physical_goal_error','hand_joint_error_deg','wrist_goal_error_cm')},
                 by_goal={g:dict(witness_check=dict(check)) for g in goals},
                 ranking=dict(pooled=block(),by_goal={g:block() for g in goals}))
    command=(['python','-u','compare_hand_body_goals_2026-09-24.py','--checkpoint','c','--out','o']
             +(['--goal-suite',suite] if suite!='hand' else []))
    return json.loads(json.dumps(payload)),dict(checkpoint=payload['checkpoint'],script_sha256='expected',command=command)


@pytest.mark.parametrize('suite',['hand','hand_and_body'])
def test_the_body_comparison_result_is_judged_against_the_suite_the_scheduler_asked_for(tmp_path,suite):
    payload,record=body_payload(tmp_path,suite)
    worker.validate_result(payload,record)
    other='hand' if suite=='hand_and_body' else 'hand_and_body'
    wrong=dict(payload,goals=worker.BODY_GOAL_SUITES[other])
    with pytest.raises(ValueError,match='scope'):worker.validate_result(wrong,record)
    with pytest.raises(ValueError,match='scope'):worker.validate_result(dict(payload,goal_suite=other),record)
    unknown=dict(record,command=record['command'][:7]+['--goal-suite','everything'])
    with pytest.raises(ValueError,match='scope'):worker.validate_result(payload,unknown)


@pytest.mark.parametrize('damage',['no_ranking','pooled_interval','pooled_case','goal_block_missing','goal_value_nan',
                                   'witness_missing','witness_count','witness_nan','weights'])
def test_an_incomplete_body_comparison_result_is_refused(tmp_path,damage):
    import copy
    payload,record=body_payload(tmp_path)
    bad=copy.deepcopy(payload)
    goal='elbow_flex_witness'
    if damage=='no_ranking':del bad['ranking']
    elif damage=='pooled_interval':bad['ranking']['pooled']['interval']=None
    elif damage=='pooled_case':bad['ranking']['pooled']['case']='not a case'
    elif damage=='goal_block_missing':del bad['ranking']['by_goal'][goal]
    elif damage=='goal_value_nan':bad['ranking']['by_goal'][goal]['per_100']=float('nan')
    elif damage=='witness_missing':del bad['by_goal'][goal]['witness_check']
    elif damage=='witness_count':bad['by_goal'][goal]['witness_check']['informative_bodies']=9
    elif damage=='witness_nan':bad['by_goal'][goal]['witness_check']['separation_from_controller']=float('nan')
    else:bad['weights_unchanged']=False
    with pytest.raises(ValueError):worker.validate_result(bad,record)


def test_a_goal_where_every_arm_ended_alike_is_a_measured_outcome_not_a_failure(tmp_path):
    from mimo_bridge.tools.command_ranking import ranking
    payload,record=body_payload(tmp_path)
    alike=ranking([dict(body=b,names=['controller','direct','synergy'],predicted=[1.,2.,3.],actual=[4.,4.,4.])
                   for b in range(8)],'x',planners=('direct','synergy'),reference='controller',draws=50)
    payload['ranking']['by_goal']['elbow_flex_witness']=json.loads(json.dumps(alike))
    payload['by_goal']['elbow_flex_witness']['witness_check']['achieved_share']=None
    worker.validate_result(payload,record)
    # every arm alike in every trial of every goal: the pooled block is undetermined too, a measurement
    payload['ranking']['pooled']=json.loads(json.dumps(alike))
    worker.validate_result(payload,record)
    # but a block that claims pairs in many bodies must carry its interval
    payload['ranking']['pooled']=dict(payload['ranking']['by_goal']['curl_witness'],interval=None,case='undetermined')
    with pytest.raises(ValueError,match='ranking'):worker.validate_result(payload,record)


def test_a_malformed_result_fails_the_attempt_instead_of_stalling_the_scheduler(tmp_path):
    """A list where the validator expects an object raised AttributeError, which finished_result did not
    catch: the error left cycle() every 30 s and the task stayed 'running' (review, 2026-09-25)."""
    payload,record=body_payload(tmp_path)
    out=tmp_path/'attempt';out.mkdir()
    for damage in ('summary','by_goal','ranking'):
        bad=dict(payload,**{damage:['not','an','object']})
        (out/'result.json').write_text(json.dumps(bad),encoding='utf-8')
        got=worker.finished_result(dict(record,output=str(out)))
        assert got['status']=='failed' and 'Unreadable or mismatched result' in got['reason'], damage


STALE='RuntimeError: the ability scheduler (process 7) started before the last change to tools/ability_rechecks.py, so it may not hold this probe\'s result validator'


def test_a_scheduler_knows_whether_it_started_after_its_own_last_change(tmp_path):
    source=tmp_path/'ability_rechecks.py';source.write_text('# scheduler\n',encoding='utf-8')
    changed=source.stat().st_mtime
    process=lambda started:type('P',(),{'create_time':lambda self:started})()
    assert worker.scheduler_is_current(process(changed+5),source) is True
    assert worker.scheduler_is_current(process(changed-60),source) is False
    assert worker.scheduler_is_current(process(changed),tmp_path/'missing.py') is False


def test_an_out_of_date_scheduler_counts_refusals_so_its_window_releases_and_a_current_one_runs_the_test(tiny_worker,monkeypatch):
    """probe_planner_candidates refuses a scheduler that started before the last change to its validator.
    The out-of-date scheduler counts the refusals as failed attempts: two hold the test with 'repair
    needed', which the comparison window treats as done, so training resumes (review of 2026-09-25: a
    status of its own kept the window, and training, paused). A current scheduler takes them back and
    runs the test; the hold used to outlive the restart that cures it."""
    tiny_worker['script'].write_text('import sys\nprint(%r)\nsys.exit(1)\n'%STALE,encoding='utf-8')
    monkeypatch.setattr(worker,'scheduler_is_current',lambda:False)
    tick(tiny_worker)
    until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')
    tick(tiny_worker)
    second=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')['tasks']['tiny']
    assert second['failures_in_a_row']==2 and 'restart the ability scheduler' in second['reason']
    held=tick(tiny_worker)['tasks']['tiny']
    assert held['status']=='repair needed', 'a status the window releases on'
    assert len(list(tiny_worker['folder'].glob('*.process.json')))==2, 'no third launch under the same scheduler'
    monkeypatch.setattr(worker,'scheduler_is_current',lambda:True)
    restarted=tick(tiny_worker)['tasks']['tiny']
    assert restarted['status'] in ('starting','running') and restarted['attempts']==1
    assert all(h.get('stale_refusal_forgiven') for h in restarted['history'])


@pytest.mark.parametrize('stale',[True,False])
def test_refusals_an_older_scheduler_counted_as_failures_are_taken_back(tiny_worker,monkeypatch,stale):
    """The state a scheduler on commit fd713b068 leaves after two refusals at one checkpoint: two failed
    attempts, the version held. A restarted (current) scheduler takes the refusals back and runs the
    test; two real failures stay held."""
    monkeypatch.setattr(worker,'scheduler_is_current',lambda:True)
    import hashlib
    folder=tiny_worker['folder'];history=[]
    for n in (1,2):
        out=folder/f'tiny_105_attempt{n}_{n}'
        out.with_suffix('.log').write_text((STALE if stale else 'RuntimeError: declared test failure')+'\n',encoding='utf-8')
        history.append(dict(status='failed',attempt_step=105,output=str(out)))
    task=json.loads(tiny_worker['manifest'].read_text())['tasks'][0]
    digest=hashlib.sha256(tiny_worker['script'].read_bytes()).hexdigest()
    version=worker.test_version(tiny_worker['root'],task,digest)
    worker.atomic_json(folder/'status.json',dict(tasks={'tiny':dict(
        status='failed',attempt_step=105,attempts=2,script_sha256=digest,test_version=version,
        failed_version=version,failures_in_a_row=2,history=history,output=str(folder/'tiny_105_attempt2_2'))}))
    record=tick(tiny_worker)['tasks']['tiny']
    if stale:
        assert record['status'] in ('starting','running') and record['attempts']==1
        assert all(h.get('stale_refusal_forgiven') for h in record['history'][:2])
    else:
        assert record['status']=='repair needed' and record['failures_in_a_row']==2


def test_a_fix_in_a_declared_file_gives_the_same_checkpoint_its_attempts_back(tiny_worker):
    """Attempts at one checkpoint counted per script alone, so a repair in a declared file released the
    held version but the checkpoint's two used attempts still blocked it (review, 2026-09-25)."""
    helper=tiny_worker['root']/'helper.py';helper.write_text('# probe helper v1\n',encoding='utf-8')
    manifest=json.loads(tiny_worker['manifest'].read_text())
    manifest['tasks'][0]['depends']=['helper.py']
    tiny_worker['manifest'].write_text(json.dumps(manifest))
    tiny_worker['script'].write_text('raise RuntimeError("declared test failure")\n',encoding='utf-8')
    tick(tiny_worker)
    until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')
    tick(tiny_worker)
    second=until(tiny_worker,lambda s:s['tasks']['tiny']['status']=='failed')
    assert second['tasks']['tiny']['attempts']==2 and second['tasks']['tiny']['attempt_step']==105
    assert tick(tiny_worker)['tasks']['tiny']['status']=='repair needed'
    helper.write_text('# probe helper v2: the fix\n',encoding='utf-8')
    retried=tick(tiny_worker)['tasks']['tiny']
    assert retried['status'] in ('starting','running') and retried['attempt_step']==105 and retried['attempts']==1
