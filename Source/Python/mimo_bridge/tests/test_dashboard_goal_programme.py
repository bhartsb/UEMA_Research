# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The capability board must preserve scope and refuse unsupported evidence."""
import json
import os
from pathlib import Path

import pytest

from mimo_bridge.tools import development_dashboard as dashboard


def fixture(root):
    catalog = json.loads((dashboard.REPO / 'docs/gain_of_function_v3/dashboard_goal_catalog.json').read_text(encoding='utf-8'))
    path = root / 'docs/gain_of_function_v3/dashboard_goal_catalog.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(catalog), encoding='utf-8')
    for group in ('highlights','capabilities','investigations','events'):
        for item in catalog[group]:
            source = root / item['source']
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('Saved research record', encoding='utf-8')
    return path, catalog


def test_missing_goal_evidence_is_withheld_without_losing_live_state(tmp_path):
    _, data = fixture(tmp_path)
    source = tmp_path / data['highlights'][0]['source']
    source.unlink()
    result = dashboard.goal_catalog(tmp_path)
    assert result['highlights'][0]['source_available'] is False
    assert result['highlights'][0]['source_url'] is None
    assert result['highlights'][1]['source_available'] is True


def test_catalogue_sources_cannot_escape_project(tmp_path):
    path, data = fixture(tmp_path)
    data['highlights'][0]['source'] = '../outside.md'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='outside'):
        dashboard.goal_catalog(tmp_path)


def test_dated_catalogue_covers_every_upgrade_and_all_proof_stages(tmp_path):
    _, catalogue = fixture(tmp_path)
    result = dashboard.goal_catalog(tmp_path)
    assert result['reviewed_at'] == catalogue['reviewed_at']
    assert {n for goal in result['capabilities'] for n in goal['upgrades']} == set(range(1,11))
    assert [s['id'] for s in result['stages']] == ['predict','choose','repeat','transfer']
    assert not any('passed' in goal['stage_states'] for goal in result['capabilities'])
    assert all(job['next'] and job['completion'] for job in result['investigations'])
    assert all(event['impact'] and event['retained'] and event['next'] for event in result['events'])


def test_invalid_proof_stage_is_not_published(tmp_path):
    path, data = fixture(tmp_path)
    data['capabilities'][0]['stage_states'][0] = 'nearly_great'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid capability'):
        dashboard.goal_catalog(tmp_path)


def test_live_experiment_estimate_uses_measured_rate_and_remaining_updates(tmp_path, monkeypatch):
    path, data = fixture(tmp_path)
    activity = data['activity'] = dict(kind='experiment', run='private/test/experiments/candidate',
                                     result='private/test/result.json', start_step=372375, updates=300)
    path.write_text(json.dumps(data), encoding='utf-8')
    run = tmp_path / activity['run']
    run.mkdir(parents=True)
    record = dict(type='mimo',step=activity['start_step']+100,timestamp=995,model_steps_per_s=2)
    (run/'replay_training_stats.jsonl').write_text(json.dumps(record)+'\n',encoding='utf-8')
    monkeypatch.setattr(dashboard.time,'time',lambda:1000)
    observed = dashboard.goal_catalog(tmp_path)['activity']
    assert observed['done'] == 100
    assert observed['seconds_remaining'] == 100
    assert observed['state'] == 'running'
    monkeypatch.setattr(dashboard.time,'time',lambda:2000)
    stale = dashboard.goal_catalog(tmp_path)['activity']
    assert stale['state'] == 'stale'
    assert stale['seconds_remaining'] is None


def test_completed_comparison_does_not_claim_repair_qualification(tmp_path):
    path, data = fixture(tmp_path)
    data['activity'] = dict(kind='experiment', run='private/test/experiments/candidate',
                           result='private/test/result.json', start_step=372375, updates=300)
    path.write_text(json.dumps(data), encoding='utf-8')
    result = tmp_path / data['activity']['result']
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({'passed':True}),encoding='utf-8')
    activity = dashboard.goal_catalog(tmp_path)['activity']
    assert activity['state'] == 'complete'
    assert activity['result_note'] == 'Experiment finished; qualification is a separate test.'
    assert activity['seconds_remaining'] is None


def test_physical_search_estimate_tracks_trials_and_withholds_stale_rate(tmp_path, monkeypatch):
    path, data = fixture(tmp_path)
    data['activity'] = dict(kind='physical', status_path='private/rolling/status.json',
                           timing_origin='private/rolling/progress_origin.json')
    path.write_text(json.dumps(data), encoding='utf-8')
    status = tmp_path/data['activity']['status_path']
    status.parent.mkdir(parents=True)
    origin = tmp_path/data['activity']['timing_origin']
    origin.write_text(json.dumps({'started_at':600}), encoding='utf-8')
    status.write_text(json.dumps(dict(stage='measuring',timestamp=1000,completed_candidates=20,
                                     maximum_candidates=96,attained=False)), encoding='utf-8')
    monkeypatch.setattr(dashboard.time, 'time', lambda:1010)
    activity = dashboard.goal_catalog(tmp_path)['activity']
    assert activity['state'] == 'running'
    assert activity['seconds_remaining'] == 1520
    assert activity['unit'] == 'movement sequences checked'
    assert activity['passed'] is None
    monkeypatch.setattr(dashboard.time, 'time', lambda:1700)
    stale = dashboard.goal_catalog(tmp_path)['activity']
    assert stale['state'] == 'stale'
    assert stale['seconds_remaining'] is None
    status.write_text(json.dumps(dict(stage='complete',timestamp=1690,completed_candidates=96,
                                     maximum_candidates=96,attained=False)), encoding='utf-8')
    completed = dashboard.goal_catalog(tmp_path)['activity']
    assert completed['state'] == 'complete'
    assert completed['passed'] is False
    assert 'not reached' in completed['result_note']
    status.write_text(json.dumps(dict(stage='complete',timestamp=1690,completed_candidates=41,
                                     physical_roll_found=True)), encoding='utf-8')
    attained = dashboard.goal_catalog(tmp_path)['activity']
    assert attained['passed'] is True
    assert 'Supported roll found' in attained['result_note']


def test_recovery_checkpoint_uses_remaining_production_updates(tmp_path):
    path, data = fixture(tmp_path)
    data['recovery'] = dict(step=375250,meaning='Recheck after replacing old-command replay observations.')
    path.write_text(json.dumps(data),encoding='utf-8')
    run = tmp_path/'Saved/mimo_bridge/experiments/production'
    run.mkdir(parents=True)
    record = dict(type='mimo',step=372375,timestamp=1000,model_steps_per_s=.5)
    (run/'replay_training_stats.jsonl').write_text(json.dumps(record)+'\n',encoding='utf-8')
    control = tmp_path/'Saved/mimo_bridge/training_continuation.json'
    control.write_text(json.dumps({'active_run':str(run)}),encoding='utf-8')
    recovery = dashboard.goal_catalog(tmp_path)['activity']['recovery']
    assert recovery['step'] == 375250
    assert recovery['remaining_updates'] == 2875
    assert recovery['seconds_estimate'] == 5750
    assert recovery['rate_recorded_at'] == 1000


def test_object_fixture_counts_saved_comparisons_without_claiming_a_skill(tmp_path, monkeypatch):
    path, data = fixture(tmp_path)
    data['activity'] = dict(kind='fixture',run='private/grasp',
                           trial_files=['left_open','left_close','right_open','right_close'])
    path.write_text(json.dumps(data),encoding='utf-8')
    run=tmp_path/'private/grasp';run.mkdir(parents=True)
    (run/'declaration.json').write_text('{}',encoding='utf-8')
    os.utime(run/'declaration.json',(50,50))
    (run/'left_open.json').write_text('{}',encoding='utf-8')
    (run/'left_close.json').write_text('{}',encoding='utf-8')
    (run/'status.json').write_text(json.dumps(dict(stage='measuring',timestamp=100)),encoding='utf-8')
    monkeypatch.setattr(dashboard.time,'time',lambda:110)
    running=dashboard.goal_catalog(tmp_path)['activity']
    assert running['done'] == 2
    assert running['total'] == 4
    assert running['seconds_remaining'] == 50
    monkeypatch.setattr(dashboard.time,'time',lambda:800)
    assert dashboard.goal_catalog(tmp_path)['activity']['seconds_remaining'] is None
    (run/'report.json').write_text(json.dumps(dict(learned_skill_tested=False)),encoding='utf-8')
    complete=dashboard.goal_catalog(tmp_path)['activity']
    assert complete['state'] == 'complete'
    assert complete['result_note'] == 'Physical fixture finished; its results are under review.'


def test_repair_work_and_next_full_run_use_measured_durations(tmp_path,monkeypatch):
    path,data=fixture(tmp_path)
    data['activity']=dict(kind='repair',start_step=380200,work_status_path='work/status.json',
        progress_unit='model-selected hand goals compared',next_timing_result='work/qualification.json',next_updates=300)
    path.write_text(json.dumps(data),encoding='utf-8')
    work=tmp_path/'work';work.mkdir()
    (work/'status.json').write_text(json.dumps(dict(state='running',completed=2,total=8,started_at=880,elapsed_s=120)))
    (work/'qualification.json').write_text(json.dumps(dict(steps=300,elapsed_s=740)))
    monkeypatch.setattr(dashboard.time,'time',lambda:1000)
    observed=dashboard.goal_catalog(tmp_path)['activity']
    assert observed['seconds_remaining']==360
    assert observed['work_status_label']=='2 of 8 model-selected hand goals compared'
    assert observed['next_seconds_estimate']==740 and observed['next_estimate_includes_setup']
    monkeypatch.setattr(dashboard.time,'time',lambda:2000)
    stale=dashboard.goal_catalog(tmp_path)['activity']
    assert stale['seconds_remaining'] is None
    assert 'overdue' in stale['work_status_label']


def test_superseded_fixture_does_not_claim_to_be_running(tmp_path, monkeypatch):
    path, data = fixture(tmp_path)
    data['activity'] = dict(kind='fixture',run='private/grasp',trial_files=['left_open','left_close'])
    path.write_text(json.dumps(data),encoding='utf-8')
    run=tmp_path/'private/grasp';run.mkdir(parents=True)
    (run/'declaration.json').write_text('{}',encoding='utf-8')
    (run/'left_open.json').write_text('{}',encoding='utf-8')
    (run/'status.json').write_text(json.dumps(dict(stage='stopped_superseded_fixture',timestamp=100,reason='Scoring correction')),encoding='utf-8')
    monkeypatch.setattr(dashboard.time,'time',lambda:110)
    activity=dashboard.goal_catalog(tmp_path)['activity']
    assert activity['state']=='stopped' and activity['seconds_remaining'] is None
    assert activity['result_note']=='Scoring correction'


def test_production_milestone_uses_live_run_and_never_claims_ability_pass(tmp_path, monkeypatch):
    path, data = fixture(tmp_path)
    data['activity'] = dict(kind='training',run='private/old',start_step=372375,target_step=375250)
    path.write_text(json.dumps(data),encoding='utf-8')
    run=tmp_path/'Saved/mimo_bridge/experiments/live';run.mkdir(parents=True)
    supervisor=run.parents[1]/'training_continuation.json'
    supervisor.write_text(json.dumps(dict(active_run=str(run))))
    pause=run.parents[1]/'supervisor_control.json'
    pause.write_text(json.dumps(dict(cmd='run')))
    stats=run/'replay_training_stats.jsonl'
    stats.write_text(json.dumps(dict(type='mimo',step=373250,timestamp=100,model_steps_per_s=2))+'\n')
    monkeypatch.setattr(dashboard.time,'time',lambda:110)
    activity=dashboard.goal_catalog(tmp_path)['activity']
    assert activity['state']=='running' and activity['seconds_remaining']==1000
    pause.write_text(json.dumps(dict(cmd='pause')))
    activity=dashboard.goal_catalog(tmp_path)['activity']
    assert activity['state']=='paused' and activity['seconds_remaining'] is None
    stats.write_text(json.dumps(dict(type='mimo',step=375250,timestamp=100,model_steps_per_s=2))+'\n')
    activity=dashboard.goal_catalog(tmp_path)['activity']
    assert activity['state']=='milestone_due' and not activity.get('passed')


def test_a_repair_run_with_fresh_stats_reads_running_under_its_pause(tmp_path, monkeypatch):
    # the relaunch review of Oct 5, 2026, finding F10: the repair run trains under {"cmd": "pause", "repair_run": run}
    path, data = fixture(tmp_path)
    data['activity'] = dict(kind='training',run='private/old',start_step=372375,target_step=375250)
    path.write_text(json.dumps(data),encoding='utf-8')
    run=tmp_path/'Saved/mimo_bridge/experiments/live';run.mkdir(parents=True)
    (run.parents[1]/'training_continuation.json').write_text(json.dumps(dict(active_run=str(run))))
    pause=run.parents[1]/'supervisor_control.json'
    pause.write_text(json.dumps(dict(cmd='pause',repair_run=str(run))))
    (run/'replay_training_stats.jsonl').write_text(json.dumps(dict(type='mimo',step=373250,timestamp=100,model_steps_per_s=2))+'\n')
    monkeypatch.setattr(dashboard.time,'time',lambda:110)
    assert dashboard.goal_catalog(tmp_path)['activity']['state']=='running'
    monkeypatch.setattr(dashboard.time,'time',lambda:100+601)          # the repair trainer ended: its stats age
    assert dashboard.goal_catalog(tmp_path)['activity']['state']=='paused'
    pause.write_text(json.dumps(dict(cmd='pause',repair_run=str(tmp_path/'another'))))
    monkeypatch.setattr(dashboard.time,'time',lambda:110)
    assert dashboard.goal_catalog(tmp_path)['activity']['state']=='paused'
