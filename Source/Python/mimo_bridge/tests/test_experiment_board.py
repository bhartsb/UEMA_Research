# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import hashlib
import json
import os

import pytest

from mimo_bridge.tools import experiment_board as board


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def job(root, state='fit'):
    folder=root/'run'
    write(folder/'status.json', dict(stage=state,timestamp=990,pid=123))
    write(folder/'declaration.json', {})
    return dict(id='example',run_dir='run'), folder


def test_finished_without_result_is_never_a_pass(tmp_path):
    spec,_=job(tmp_path,'complete')
    assert board.job_state(tmp_path,spec,1000)['state']=='result_missing'


def test_causal_progress_preserves_declared_work_and_never_calls_fit_an_ability():
    declaration=dict(train_worlds=list(range(6)),test_worlds=list(range(6,10)),states_per_world=8)
    row=board.causal_readout_progress(declaration,dict(stage='recording_raw_causal_windows',decisions_saved=33))
    assert (row['done'],row['total'])==(33,80) and 'estimate' not in row
    fitted=board.causal_readout_progress(declaration,dict(stage='fitting_isolated_auxiliary',condition='action_blind_stage_2'))
    assert (fitted['done'],fitted['total'])==(4,4) and 'state' not in fitted
    for bad in (-1,81,True,float('nan')):
        with pytest.raises(ValueError):
            board.causal_readout_progress(declaration,dict(stage='recording_raw_causal_windows',decisions_saved=bad))
    with pytest.raises(ValueError):
        board.causal_readout_progress(declaration,dict(stage='encoding_own_raw_windows',complete=8,total=79))


def test_missing_process_is_not_running_and_stale_status_is_visible(tmp_path,monkeypatch):
    spec,_=job(tmp_path)
    monkeypatch.setattr(board,'process_alive',lambda *args:False)
    assert board.job_state(tmp_path,spec,1000)['state']=='stopped'
    monkeypatch.setattr(board,'process_alive',lambda *args:True)
    assert board.job_state(tmp_path,spec,1500)['state']=='status_stale'


def test_completed_report_requires_matching_review_and_finite_values(tmp_path):
    spec,folder=job(tmp_path,'complete')
    report=folder/'report.json';write(report,{'gain':.001})
    assert board.job_state(tmp_path,spec,1000)['state']=='completed_unreviewed'
    write(tmp_path/'review.json',dict(report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
          state='insufficient',summary='Too small',decision='Repair grounding'))
    spec['review']='review.json'
    assert board.job_state(tmp_path,spec,1000)['state']=='insufficient'
    write(report,{'gain':.01})
    changed=board.job_state(tmp_path,spec,1000)
    assert changed['state']=='completed_unreviewed' and 'result' not in changed
    assert 'review_issue' in changed
    report.write_text('{"gain":NaN}',encoding='utf-8')
    assert board.job_state(tmp_path,spec,1000)['state']=='record_error'
    report.write_text('{"gain":1e999}',encoding='utf-8')
    assert board.job_state(tmp_path,spec,1000)['state']=='record_error'


# 2026-09-22: the board no longer forecasts finish times. the style guide bans time estimates
# (operator 2026-09-08), so the four tests that asserted `estimate` fields now assert their
# absence and keep every count they checked.
def test_completed_copy_is_not_double_counted_and_no_finish_time_is_forecast(tmp_path):
    epoch=1_700_000_000  # portable to FAT/exFAT recording drives (which reject 1970 dates)
    for i in range(3):
        path=tmp_path/f'responsive_{i}'/'fit.json'
        write(path,{'online_updates':100})
        os.utime(path,(epoch+100+i*200,epoch+100+i*200))
    p=board.caregiver_progress(tmp_path,{'worlds':3,'train_steps':100},
         dict(arm='responsive_2',stage='fit',step=100,timestamp=510),1)
    assert p['done']==300 and p['total']==600
    assert 'estimate' not in p
    assert p['phases'][2]['state']=='waiting'
    current=board.caregiver_progress(tmp_path,{'worlds':3,'train_steps':100},
         dict(arm='yoked_0',stage='fit',step=20,timestamp=530),1)
    assert current['done']==320


def test_only_registered_public_artifacts_can_be_served_outside_repo(tmp_path):
    root=tmp_path/'repo';outside=tmp_path/'drive'
    write(root/'docs/development_dashboard/work.json',dict(schema=1,jobs=[dict(id='registered',run_dir=str(outside))]))
    assert board.artifact(root,'registered','report.json')==outside/'report.json'
    for name in ('../secret.json','weights.pt','arbitrary.json'):
        with pytest.raises(ValueError):board.artifact(root,'registered',name)
    with pytest.raises(ValueError):board.artifact(root,'unknown','report.json')


def test_resumed_caregiver_fit_counts_active_and_completed_attempt_once(tmp_path):
    declaration=dict(worlds=2,train_steps=100,seeds=[11,12])
    target=tmp_path/'responsive_1'/'attempt_002'
    write(target/'attempt.json',dict(arm='responsive',world_seed=12))
    status=dict(stage='fit',arm='attempt_002',seed=12,step=35,timestamp=1000)
    assert board.caregiver_progress(tmp_path,declaration,status,1)['done']==35
    write(target/'fit.json',dict(online_updates=100))
    write(target/'probe_follow_visible'/'result.json',{})
    write(tmp_path/'completed_fits.json',{'responsive_1':str(target/'fit.json')})
    result=board.caregiver_progress(tmp_path,declaration,status,1)
    assert result['done']==100 and '1 of 24 behavior tests' in result['detail']
    write(tmp_path/'completed_fits.json',{'responsive_1':str(tmp_path.parent/'foreign'/'fit.json')})
    with pytest.raises(ValueError,match='registered arm'):
        board.caregiver_progress(tmp_path,declaration,status,1)


def test_failed_or_running_caregiver_has_counts_and_no_calendar_forecast(tmp_path):
    for i in range(3):
        write(tmp_path/f'responsive_{i}'/'fit.json',dict(online_updates=100,copy_wall_seconds=200))
    result=board.caregiver_progress(tmp_path,dict(worlds=3,train_steps=100),
        dict(stage='failed',timestamp=1000),1)
    assert result['done']==300 and 'estimate' not in result
    running=board.caregiver_progress(tmp_path,dict(worlds=3,train_steps=100),
        dict(stage='fit',arm='yoked_0',step=0,timestamp=1000),1)
    assert running['done']==300 and 'estimate' not in running


def test_worker_restarts_change_no_count_and_invalid_durations_still_fail(tmp_path):
    epoch=1_700_000_000
    for i,(end,pid) in enumerate([(100,11),(300,11),(500,11),(1500,22)]):
        fit=tmp_path/f'responsive_{i}'/'fit.json'
        write(fit,dict(online_updates=100));os.utime(fit,(epoch+end,epoch+end))
        write(fit.parent/'status.json',dict(pid=pid))
    result=board.caregiver_progress(tmp_path,dict(worlds=4,train_steps=100),
        dict(stage='fit',arm='yoked_0',step=0,timestamp=2000),1)
    assert result['done']==400 and 'estimate' not in result
    (tmp_path/'responsive_0'/'fit.json').write_text('{"online_updates":100,"copy_wall_seconds":-5}')
    with pytest.raises(ValueError,match='duration'):
        board.caregiver_progress(tmp_path,dict(worlds=4,train_steps=100),
            dict(stage='fit',arm='yoked_0',step=0,timestamp=2000),1)


def test_parallel_caregiver_counts_each_copy_once_and_uses_current_concurrency(tmp_path):
    declaration=dict(worlds=2,train_steps=100,seeds=[11,12])
    for i in (0,1):
        write(tmp_path/f'responsive_{i}'/'fit.json',dict(online_updates=100,copy_wall_seconds=200))
    status=dict(stage='fit_parallel',timestamp=1000,workers=[
        dict(arm='yoked_0',step=20),dict(arm='yoked_1',step=40)])
    result=board.caregiver_progress(tmp_path,declaration,status,1)
    assert result['done']==260 and result['total']==400
    assert 'estimate' not in result
    duplicate=dict(status,workers=[status['workers'][0],status['workers'][0]])
    with pytest.raises(ValueError,match='Multiple active workers'):
        board.caregiver_progress(tmp_path,declaration,duplicate,1)


def test_developmental_observations_name_each_control(tmp_path):
    spec,folder=job(tmp_path,'complete');spec['kind']='developmental_probe'
    write(folder/'declaration.json',dict(profile='body_floor'))
    rows=[dict(condition='present',arm=arm,minimum_front_up=1.,maximum_horizontal_displacement_m=.01)
          for arm in ('candidate','baseline')]
    write(folder/'report.json',dict(results=[dict(seed=11,conditions=rows)]))
    result=board.job_state(tmp_path,spec,1000)
    assert 'Model control' in result['observations'][0]['label']
    assert 'Pose-hold control' in result['observations'][2]['label']


def test_sensory_forecast_cards_show_action_effect_errors_not_foveal_percentages(tmp_path):
    spec,folder=job(tmp_path,'complete');spec['kind']='developmental_probe'
    rows={name:dict(worlds=2,world_values={'action_effect_error_removed':[-.02,-.04]})
          for name in ('proprioception','tactile','vision')}
    write(folder/'report.json',dict(modalities=rows,results=[{},{}]))
    result=board.job_state(tmp_path,spec,1000)
    assert len(result['observations'])==3
    assert '3.00% more error' in result['observations'][0]['display']
    assert 'fovea' not in result['observation_scale']


def test_bad_job_does_not_hide_other_jobs(tmp_path):
    spec,folder=job(tmp_path)
    (folder/'status.json').write_text('broken',encoding='utf-8')
    write(tmp_path/'docs/development_dashboard/work.json',dict(schema=1,jobs=[spec,dict(id='next')]))
    result=board.build(tmp_path,1000)
    assert [j['state'] for j in result['jobs']]==['record_error','not_started']
    (tmp_path/'docs/development_dashboard/work.json').write_text('broken',encoding='utf-8')
    assert board.build(tmp_path,1000)['error']


def test_unlaunched_plan_and_missing_status_do_not_look_queued(tmp_path,monkeypatch):
    assert board.job_state(tmp_path,dict(id='next'),1000)['state']=='not_started'
    write(tmp_path/'launch.json',dict(pid=123,started_at=900))
    monkeypatch.setattr(board,'process_alive',lambda *args:False)
    # 2026-09-22: a registered folder that does not exist is "result files missing (path)",
    # not "stopped"; a folder that exists without a status record still reads "stopped".
    gone=board.job_state(tmp_path,dict(id='missing',run_dir='missing',launch='launch.json'),1000)
    assert gone['state']=='files_missing' and gone['failure']==f"result files missing ({tmp_path/'missing'})"
    (tmp_path/'present').mkdir()
    assert board.job_state(tmp_path,dict(id='present',run_dir='present',launch='launch.json'),1000)['state']=='stopped'
    monkeypatch.setattr(board,'process_alive',lambda *args:True)
    assert board.job_state(tmp_path,dict(id='missing',run_dir='missing',launch='launch.json'),1000)['state']=='starting'


def test_null_planned_state_is_named_as_unrecorded(tmp_path):
    assert board.job_state(tmp_path,dict(id='x',planned_state=None),1000)['state']=='no_state_recorded'


def test_no_job_state_carries_a_time_forecast(tmp_path,monkeypatch):
    """Every progress branch that used to forecast (developmental worlds, measured probes,
    caregiver copies) keeps its counts and writes no estimate."""
    monkeypatch.setattr(board,'process_alive',lambda *args:True)
    folder=tmp_path/'probe'
    write(folder/'declaration.json',dict(seeds=[1,2,3],steps=10))
    write(folder/'status.json',dict(stage='world',timestamp=995,world=2,frame=3,completed_durations_seconds=[10.0,12.0]))
    spec=dict(id='p',run_dir='probe',kind='developmental_probe')
    state=board.job_state(tmp_path,spec,1000)
    assert state['progress']['done']==1 and 'estimate' not in state['progress']
    write(folder/'status.json',dict(stage='measure',timestamp=995,step=4,steps=10,completed_durations_seconds=[5.0]))
    state=board.job_state(tmp_path,dict(id='q',run_dir='probe'),1000)
    assert state['progress']['done']==4 and 'estimate' not in state['progress']
    import inspect
    assert "'earliest'" not in inspect.getsource(board) and "'latest'" not in inspect.getsource(board)


def test_job_lineage_names_the_tested_checkpoint_relation(tmp_path):
    root=tmp_path
    run=root/'Saved/mimo_bridge/experiments/current'
    write(run/'args.json',{})
    bounds={str(run.resolve()):500}
    inside=dict(id='a',checkpoint=str(run/'checkpoints/ckpt_step_400'))
    later=dict(id='b',checkpoint=str(run/'checkpoints/ckpt_step_600'))
    assert board.job_lineage(root,inside,bounds)=='ancestor_checkpoint'
    assert board.job_lineage(root,later,bounds)=='not_linked_to_current_model'
    assert board.job_lineage(root,dict(id='c'),bounds)=='unknown'
    folder=root/'job';write(folder/'declaration.json',dict(checkpoint=str(run/'checkpoints/ckpt_step_100')))
    assert board.job_lineage(root,dict(id='d',run_dir='job'),bounds)=='ancestor_checkpoint'


def test_completed_behavior_is_visible_without_certifying_learning(tmp_path):
    spec,folder=job(tmp_path,'complete');spec['kind']='early_probe'
    rows={k:dict(aligned_fraction=.25,samples=100) for k in ('follow_visible','follow_hidden','return_visible','return_hidden','cue_correct','cue_wrong')}
    write(folder/'report.json',dict(results=rows))
    result=board.job_state(tmp_path,spec,1000)
    assert result['state']=='completed_unreviewed'
    assert len(result['observations'])==6 and result['observations'][0]['value']==.25
    rows['follow_visible']['samples']=0;write(folder/'report.json',dict(results=rows))
    assert board.job_state(tmp_path,spec,1000)['state']=='record_error'
