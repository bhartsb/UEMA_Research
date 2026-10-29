# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import hashlib
import json
import struct
from pathlib import Path

import pytest

from mimo_bridge.tools import development_dashboard as dashboard


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(root):
    saved=root/'Saved/mimo_bridge'
    parent=saved/'experiments/parent'
    run=saved/'experiments/current'
    write(parent/'args.json',{})
    write(run/'args.json',{'resume':str(parent/'checkpoints/ckpt_step_100')})
    write(saved/'training_continuation.json',dict(active_run=str(run),timestamp=995,status='training'))
    stats=run/'replay_training_stats.jsonl'
    stats.write_text(json.dumps(dict(type='mimo',step=150,timestamp=990))+'\n{"type":',encoding='utf-8')
    receipt=root/'docs/result.json'
    write(receipt,{'gate':'PASS'})
    record=dict(id='example',capability='predict',date='2026-09-11',kind='offline_component',outcome='demonstrated',subject=str(parent/'checkpoints/ckpt_step_90'),title='Example',summary='Separate fit',missing='Transfer',method='Frozen fit',receipts=[dict(path='docs/result.json',sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())],checks=[dict(receipt=0,path='gate',equals='PASS')])
    registry=root/'docs/development_dashboard/evidence.json'
    write(registry,dict(schema=1,abilities=[dict(id='predict')],records=[record]))
    return run,parent,receipt,registry


def test_live_data_updates_and_partial_log_write_does_not_erase_record(tmp_path):
    run,_,_,_=fixture(tmp_path)
    first=dashboard.build_state(tmp_path,now=1000)
    assert first['training']['step']==150
    assert first['training']['record_age_seconds']==10
    (run/'replay_training_stats.jsonl').write_text(json.dumps(dict(type='mimo',step=200,timestamp=1000))+'\n',encoding='utf-8')
    assert dashboard.build_state(tmp_path,now=1001)['training']['step']==200


def test_missing_supervisor_is_unknown_not_running(tmp_path):
    fixture(tmp_path)
    (tmp_path/'Saved/mimo_bridge/training_continuation.json').unlink()
    state=dashboard.build_state(tmp_path,now=1000)
    assert state['training']['supervisor_status']=='unknown'
    assert state['training']['step'] is None


def test_changed_or_missing_receipt_withholds_a_previously_passing_claim(tmp_path):
    _,_,receipt,_=fixture(tmp_path)
    assert dashboard.build_state(tmp_path,now=1000)['records'][0]['valid']
    write(receipt,{'gate':'FAIL'})
    assert not dashboard.build_state(tmp_path,now=1000)['records'][0]['valid']
    receipt.unlink()
    assert not dashboard.build_state(tmp_path,now=1000)['records'][0]['valid']


def test_separate_predictor_pass_never_becomes_a_live_model_ability(tmp_path):
    fixture(tmp_path)
    record=dashboard.build_state(tmp_path,now=1000)['records'][0]
    assert record['valid']
    assert record['lineage']=='ancestor_checkpoint'
    assert record['current_live_ability_proven'] is False


def test_context_and_memory_migrations_keep_the_real_parent_and_fork_boundary(tmp_path):
    run,parent,_,_=fixture(tmp_path)
    initial=run/'checkpoints/ckpt_step_100'
    context=tmp_path/'Saved/mimo_bridge/context_copy/checkpoints/ckpt_step_100'
    old=parent/'checkpoints/ckpt_step_100'
    write(run/'args.json',dict(resume=str(initial)))
    write(initial/'continuity_migration.json',dict(checkpoint=str(initial),parent=str(context)))
    write(context/'context_growth.json',dict(checkpoint=str(context),parent=str(old)))
    state=dashboard.build_state(tmp_path,now=1000)
    assert state['records'][0]['lineage']=='ancestor_checkpoint'
    bounds=dashboard.ancestry_bounds(tmp_path,state['training'])
    assert bounds[str(parent)]==100
    assert dashboard.lineage_relation(tmp_path,str(parent/'checkpoints/ckpt_step_101'),bounds)=='not_linked_to_current_model'


def test_incomplete_choice_receipt_cannot_remain_valid_after_extraction_fails(tmp_path):
    _,_,_,registry=fixture(tmp_path)
    data=dashboard.read_json(registry)
    data['records'][0]['id']='wellness-choice'
    write(registry,data)
    record=dashboard.build_state(tmp_path,now=1000)['records'][0]
    assert not record['valid']
    assert 'review_issue' in record


def test_rollback_excludes_later_parent_evidence_and_other_branches(tmp_path):
    _,parent,_,registry=fixture(tmp_path)
    data=dashboard.read_json(registry)
    data['records'][0].update(kind='live_behavior',subject=str(parent/'checkpoints/ckpt_step_120'))
    write(registry,data)
    state=dashboard.build_state(tmp_path,now=1000)
    assert state['records'][0]['lineage']=='not_linked_to_current_model'
    assert state['abilities'][0]['evidence'] is None


def test_nonfinite_source_cannot_support_a_claim(tmp_path):
    _,_,receipt,registry=fixture(tmp_path)
    receipt.write_text('{"gate":"PASS","score":NaN}',encoding='utf-8')
    data=dashboard.read_json(registry)
    data['records'][0]['receipts'][0]['sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
    write(registry,data)
    assert not dashboard.build_state(tmp_path,now=1000)['records'][0]['valid']


def mp4(path,complete=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b''.join(struct.pack('!I4s',8,kind) for kind in ([b'ftyp',b'mdat',b'moov'] if complete else [b'ftyp',b'mdat'])))


def test_video_requires_both_finished_views_and_ignores_event_selected_clips(tmp_path):
    run,_,_,_=fixture(tmp_path)
    outside=run/'videos/step_140_120000.mp4'
    eyes=run/'videos/step_140_120000_eye.mp4'
    mp4(outside);mp4(eyes,False)
    assert dashboard.build_state(tmp_path,now=1000)['videos']==[]
    mp4(eyes)
    mp4(run/'videos/moment_headlook_step_149.mp4')
    mp4(run/'videos/moment_headlook_step_149_eye.mp4')
    videos=dashboard.build_state(tmp_path,now=1000)['videos']
    assert len(videos)==1 and videos[0]['step']==140
    assert videos[0]['kind']=='observation' and videos[0]['events']==[]


def test_entry_preserves_old_dashboard_and_restart_does_not_replace_archive(tmp_path):
    saved=tmp_path/'Saved/mimo_bridge';saved.mkdir(parents=True)
    (saved/'dashboard.html').write_text('original',encoding='utf-8')
    dashboard.install_entry(tmp_path,8766)
    dashboard.install_entry(tmp_path,8766)
    assert (saved/'dashboard_legacy.html').read_text()=='original'
    assert '127.0.0.1:8766' in (saved/'dashboard.html').read_text()


def test_old_videos_cannot_claim_recorded_control_ownership(tmp_path):
    run,_,_,_=fixture(tmp_path)
    mp4(run/'videos/step_140.mp4');mp4(run/'videos/step_140_eye.mp4')
    assert dashboard.build_state(tmp_path,now=1000)['videos'][0]['control_annotated'] is False
    write(run/'videos/step_140_control.json',dict(version='body_control_video_v1',
          observer_overlay=True,frames=[dict(frame=0,body_control='scheduled_hold')]))
    assert dashboard.build_state(tmp_path,now=1000)['videos'][0]['control_annotated'] is True
    (run/'videos/step_140_control.json').write_text('{incomplete')
    assert dashboard.build_state(tmp_path,now=1000)['videos'][0]['control_annotated'] is False


def test_served_paths_cannot_leave_project(tmp_path):
    with pytest.raises(ValueError):dashboard.confined(tmp_path,'../secret.json')


def test_video_seek_range_and_out_of_bounds():
    assert dashboard.range_bounds('bytes=4-8',20)==(4,8,True)
    assert dashboard.range_bounds('bytes=-4',20)==(16,19,True)
    with pytest.raises(ValueError):dashboard.range_bounds('bytes=20-',20)
    with pytest.raises(ValueError):dashboard.range_bounds('bytes=4-2',20)


def test_browser_aborting_a_video_request_does_not_trigger_a_second_response(tmp_path):
    from types import SimpleNamespace
    handler=object.__new__(dashboard.Handler)
    handler.headers={'Host':'127.0.0.1:8766'}
    handler.path='/files/Saved/mimo_bridge/video.mp4'
    handler.server=SimpleNamespace(root=tmp_path)
    def disconnected(*args):raise ConnectionAbortedError('browser navigated away')
    def second_response(*args):raise AssertionError('cannot send an error to a closed browser')
    handler.send_file=disconnected
    handler.send_error=second_response
    handler.do_GET()


def test_system_repairs_never_replace_behavior_evidence(tmp_path):
    _,_,_,registry=fixture(tmp_path)
    data=dashboard.read_json(registry)
    repair=dict(data['records'][0],id='repair',kind='implementation')
    data['records'].append(repair)
    write(registry,data)
    state=dashboard.build_state(tmp_path,now=1000)
    assert state['records'][-1]['valid']
    assert state['abilities'][0]['evidence']['id']=='example'
    assert not state['records'][-1]['current_live_ability_proven']


def test_legacy_refresh_updates_both_pages_and_surfaces_history_failure(tmp_path,monkeypatch):
    from mimo_bridge.tools import dashboard as old, lifetime
    saved=tmp_path/'Saved/mimo_bridge'
    saved.mkdir(parents=True)
    monkeypatch.setattr(old,'build',lambda:(saved/'dashboard_legacy.html').write_text('diagnostics'))
    monkeypatch.setattr(lifetime,'build',lambda **kw:(saved/'lifetime.html').write_text('history'))
    assert dashboard.refresh_legacy(tmp_path)['status']=='updated'
    first=dashboard.legacy_state(tmp_path,now=1000)
    assert first['dashboard'] and first['history']
    assert len(list((saved/'dashboard_history').glob('*.html')))==1
    def fail(**kw):raise ValueError('history read failed')
    monkeypatch.setattr(lifetime,'build',fail)
    assert dashboard.refresh_legacy(tmp_path)['status']=='failed'
    assert dashboard.legacy_state(tmp_path,now=1001)['updater']['error']=='history read failed'


def test_history_uses_fork_boundary_and_cannot_claim_achievements(tmp_path,monkeypatch):
    from mimo_bridge.tools import lifetime
    run,parent,_,_=fixture(tmp_path)
    saved=tmp_path/'Saved/mimo_bridge'
    monkeypatch.setattr(lifetime,'SAVED',str(saved))
    monkeypatch.setattr(lifetime,'EXP',str(saved/'experiments'))
    monkeypatch.setattr(lifetime,'OUT_HTML',str(saved/'lifetime.html'))
    monkeypatch.setattr(lifetime,'OUT_JSON',str(saved/'lifetime.json'))
    rows=[dict(type='mimo',step=s,timestamp=900+s,task_hand_fov=.02) for s in range(90,111)]
    # The discarded future must not inflate the inherited observation.
    for row in rows:
        if row['step']>100:row['task_hand_fov']=1.
    (parent/'replay_training_stats.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (saved/'lifetime.html').write_text('WON: obsolete claim')
    lifetime.build(force=True)
    history=dashboard.read_json(saved/'lifetime.json')
    assert history[0]['linked'] and history[0]['step']==100
    assert history[0]['eyes']==.02
    html=(saved/'lifetime.html').read_text(encoding='utf-8')
    assert 'Intentional rolling and purposeful recovery remain unproved' in html
    assert 'WON' not in html
    assert (saved/'lifetime_before_review_2026-09-12.html').read_text()=='WON: obsolete claim'


def native_observation_fixture(root,reviewed=True):
    from mimo_bridge.tools.review_completed_developmental import review
    run,parent,_,registry=fixture(root)
    checkpoint=parent/'checkpoints/ckpt_step_90';checkpoint.mkdir(parents=True)
    (checkpoint/'model.safetensors').write_bytes(b'identified frozen checkpoint')
    digest=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()
    folder=root/'Saved/native';folder.mkdir()
    declaration=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,profile='toy',worlds=1,
        seeds=[11],instrument_smoke=False,steps=160,probe_version='physical_geometry_v3')
    rows=[dict(condition=condition,arm=arm,scored_seconds=8.,target_center_in_fovea_fraction=.01,
        longest_foveal_dwell_s=.05,minimum_front_up=.99,maximum_horizontal_displacement_m=.01)
        for condition in ('present','relocated','removed','interrupted') for arm in ('candidate','baseline')]
    report=dict(results=[dict(seed=11,same_action_replay_exact=True,
        complete_model_replay=dict(complete_model_replay_exact=True),conditions=rows)])
    write(folder/'declaration.json',declaration);write(folder/'report.json',report)
    write(folder/'status.json',dict(stage='complete',timestamp=999))
    job=dict(id='development-v3-toy-step-90',title='Toy control',question='Does the target control its look?',
        decision='Repair the failed target control',scope='Frozen observation',kind='developmental_probe',
        run_dir=str(folder),abilities=['gaze_control'])
    if reviewed:
        review_path=root/'docs/native_review.json';write(review_path,review(folder,job['id']))
        job['review']='docs/native_review.json'
    write(root/'docs/development_dashboard/work.json',dict(schema=1,jobs=[job]))
    definition=dict(id='gaze_control',tier=1,name='Control looking',minimum_environment={'description':'Visible object'},
        natural_driver='Available sensory distinctions',human_comparison={'text':'Infant eye control'},
        tests=[{'question':'Does target motion control looking?'}],implementation_gap='Retention and reachability')
    write(root/'docs/gain_function_2026-09-13/requirements.json',dict(tier_names={'1':'First control'},abilities=[definition]))
    data=dashboard.read_json(registry);data['abilities'].append(dict(id='correct'));write(registry,data)
    return checkpoint,folder


def test_reviewed_negative_is_linked_to_model_history_and_ability_without_passing(tmp_path):
    checkpoint,folder=native_observation_fixture(tmp_path)
    state=dashboard.build_state(tmp_path,now=1000);record=state['records'][-1]
    assert record['valid'] and record['lineage']=='ancestor_checkpoint'
    assert record['validation']=='reviewed_observation' and not record['current_live_ability_proven']
    assert state['abilities'][-1]['evidence']['id']==record['id']
    ability=state['development']['stages'][0]['abilities'][0]
    assert ability['observations'][0]['summary']==record['summary'] and not ability['acquired']
    assert '/experiments/' in record['sources'][1]['url']
    (checkpoint/'model.safetensors').write_bytes(b'changed frozen checkpoint')
    changed=dashboard.build_state(tmp_path,now=1000)
    assert not changed['abilities'][-1]['evidence']['valid']
    assert not changed['development']['stages'][0]['abilities'][0]['observations']


def test_future_native_observation_attaches_automatically_with_no_manual_review(tmp_path):
    checkpoint,folder=native_observation_fixture(tmp_path,reviewed=False)
    state=dashboard.build_state(tmp_path,now=1000);record=state['records'][-1]
    assert record['validation']=='automated_validated_observation'
    assert record['valid'] and not record['current_live_ability_proven']
    assert state['work']['jobs'][0]['state']=='measured'
    assert state['abilities'][-1]['evidence']['id']==record['id']
    report=dashboard.read_json(folder/'report.json');report['results'][0]['conditions'].pop()
    write(folder/'report.json',report)
    assert not dashboard.build_state(tmp_path,now=1000)['records'][-1]['valid']


@pytest.mark.parametrize('damage',['identity_missing','identity_changed','replay_failed','world_missing','smoke'])
def test_automatic_observation_fails_closed_on_missing_identity_or_controls(tmp_path,damage):
    checkpoint,folder=native_observation_fixture(tmp_path,reviewed=False)
    declaration=dashboard.read_json(folder/'declaration.json');report=dashboard.read_json(folder/'report.json')
    if damage=='identity_missing':declaration.pop('checkpoint_sha256')
    elif damage=='identity_changed':declaration['checkpoint_sha256']='0'*64
    elif damage=='replay_failed':report['results'][0]['complete_model_replay']['complete_model_replay_exact']=False
    elif damage=='world_missing':report['results']=[]
    else:declaration['instrument_smoke']=True
    write(folder/'declaration.json',declaration);write(folder/'report.json',report)
    state=dashboard.build_state(tmp_path,now=1000)
    assert not state['records'][-1]['valid'] and not state['records'][-1]['current_live_ability_proven']


@pytest.mark.parametrize('complete,matching,valid',[(True,True,True),(False,True,False),(True,False,False)])
def test_retired_snapshot_keeps_historical_observation_only_with_matching_completed_audit(tmp_path,complete,matching,valid):
    checkpoint,folder=native_observation_fixture(tmp_path)
    weights=checkpoint/'model.safetensors';digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    audit=tmp_path/'retirement.jsonl'
    prepared=dict(schema='checkpoint_retention_v1',event='retirement_prepared',id='receipt-1',timestamp=1000,
        checkpoint=str(checkpoint),files={'model.safetensors':{'sha256':digest if matching else '0'*64}})
    lines=[prepared]
    if complete:lines.append(dict(schema='checkpoint_retention_v1',event='retirement_complete',id='receipt-1'))
    audit.write_text('\n'.join(json.dumps(row) for row in lines)+'\n')
    write(tmp_path/'.checkpoint_retention.json',dict(schema='checkpoint_retention_v1',storage=str(tmp_path),
        checkpoint_run_roots=[str(checkpoint.parents[2])],checkpoint_roots=[],audit_path=str(audit)))
    weights.unlink()
    record=dashboard.build_state(tmp_path,now=1001)['records'][-1]
    assert record['valid'] is valid and not record['current_live_ability_proven']
    if valid:
        assert record['model_identity']['availability']=='retired'
        assert 'cannot currently be rerun' in record['missing']


# --------------------------------------------------------------------------------------------
# Gain-of-function build (2026-09-22): the verdict line, the ten-item panel, the lineage
# gauges, the settings list, body-labelled skin regions, history pruning, no time estimates.
# --------------------------------------------------------------------------------------------
STEP_RATE_REASON=('model steps per second 0.397 under one third of own hour-1 baseline 1.249 two hourly '
                  'checks running (sustained step-time blow-up)')


def test_verdict_names_a_purposeful_pause_and_the_stop_rule_with_its_numbers(tmp_path):
    run,_,_,_=fixture(tmp_path)
    saved=tmp_path/'Saved/mimo_bridge'
    write(saved/'training_continuation.json',dict(active_run=str(run),timestamp=995,
          status='paused for repair or explicit instruction',pause_reason='explicit supervisor pause'))
    write(saved/'supervisor_control.json',dict(cmd='pause',reason='training paused during the gain-of-function build'))
    write(run/'stop_reason.json',dict(reason=STEP_RATE_REASON,resume_policy='repair_required',timestamp=985))
    write(run/'stage_done.json',dict(steps=150,finished_at=992,finished_at_human='2026-09-22 17:17:12'))
    v=dashboard.build_state(tmp_path,now=1000)['verdict']
    assert v['state']=='paused'
    assert v['sentence']=='Training is paused on purpose: training paused during the gain-of-function build.'
    assert '0.397 model steps per second' in v['facts'][0] and 'the line is 0.416' in v['facts'][0]
    assert 'lineage step 150' in v['facts'][1]
    (saved/'supervisor_control.json').unlink()
    v=dashboard.build_state(tmp_path,now=1000)['verdict']
    assert v['state']=='stopped' and v['sentence'].startswith('Training is stopped: the trainer ran 0.397 model steps per second')
    for banned in ('needs attention','Earlier chooser','Self-perception learning is on'):
        assert banned not in json.dumps(v)


def test_the_verdict_names_the_program_that_wrote_the_stop(tmp_path):
    run,_,_,_=fixture(tmp_path)
    saved=tmp_path/'Saved/mimo_bridge'
    write(saved/'training_continuation.json',dict(active_run=str(run),timestamp=995,
          status='paused for repair or explicit instruction',pause_reason='explicit supervisor pause'))
    write(saved/'supervisor_control.json',dict(cmd='pause',reason='motor fixes kill line crossed'))
    write(run/'stop_reason.json',dict(reason='motor fixes kill line crossed',resume_policy='repair_required',
          timestamp=985,writer='docs/gain_of_function_v4/motor_fixes_watch.py'))
    v=dashboard.build_state(tmp_path,now=1000)['verdict']
    assert v['facts'][0].startswith('The motor fixes kill-line watcher'), v['facts']
    assert 'hourly watcher' not in v['facts'][0]


def test_verdict_running_needs_fresh_supervisor_and_record(tmp_path):
    fixture(tmp_path)
    v=dashboard.build_state(tmp_path,now=1000)['verdict']
    assert v['state']=='running' and v['sentence'].startswith('Training is running: current is at lineage step 150')
    v=dashboard.build_state(tmp_path,now=5000)['verdict']
    assert v['state']=='unconfirmed' and 'not confirmed' in v['sentence']


def test_the_ten_items_panel_reads_the_newest_record_absent_and_present(tmp_path):
    run,_,_,_=fixture(tmp_path)
    state=dashboard.build_state(tmp_path,now=1000)
    panel=state['gain_of_function']
    assert panel['total']==10 and panel['measured']==0
    assert all(not row['measured'] and row['value'] is None for row in panel['items'])
    assert state['headline']['title'].startswith('No gain-of-function item is shown working yet')
    assert '--causal-chain' in state['what_next'] and '--vestibular-sense' in state['what_next']
    row=dict(type='mimo',step=160,timestamp=999,causal_chain=dict(chosen_best_share=0.5,alternatives_n=4),
             controllability=dict(agreement_mean=0.3,control_mean=0.0,gate_open=True),
             vestibular=dict(improvement_over_persistence=0.2))
    (run/'replay_training_stats.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    resume=dashboard.read_json(run/'args.json')['resume']
    write(run/'args.json',{'resume':resume,'causal_chain':'on','controllability':'on','vestibular_sense':'on'})
    state=dashboard.build_state(tmp_path,now=1000)
    by={r['item']:r for r in state['gain_of_function']['items']}
    assert state['gain_of_function']['measured']==3
    # rewritten 2026-09-23 (fix-dashboards): item 1 has no goal line (the planner does not aim at
    # predicted wellness, reviews/dashboards.md M1) and the headline leads with abilities (M6)
    from mimo_bridge.tools.lineage_series import item_text
    assert by[1]['big']=='50 in 100 plans' and 'No validated success target: the planner does not aim' in item_text(by[1])
    assert by[2]['value']==0.3 and 'No success threshold has been validated for this measurement.' in item_text(by[2])
    assert by[10]['value']==0.2 and not by[7]['measured']
    # rewritten 2026-09-23 (verify-fix-dashboards): an item beats its control only by the
    # producer's own evidence test (controllability.gate_open), never by a bare ">"
    assert state['headline']['title'].startswith("1 item beats its control by the producer's own evidence test")
    assert state['headline']['detail'].startswith('3 of 10 items have a direct measurement')


def test_lineage_endpoint_payload_and_settings_list(tmp_path):
    fixture(tmp_path)
    payload=dashboard.lineage_payload(tmp_path)
    assert payload['schema']==1 and payload['x']['step']==[150]
    assert [s['run'] for s in payload['segments']]==['parent','current']
    assert {c['id'] for c in payload['cards']}>={'item1','item10','machine'}
    json.dumps(payload,allow_nan=False)
    assert (tmp_path/'Saved/mimo_bridge/lineage_series').is_dir(), 'offsets are cached under Saved/mimo_bridge/lineage_series'
    state=dashboard.build_state(tmp_path,now=1000)
    settings={s['flag']:s for s in state['training']['settings']}
    for flag in ('--causal-chain','--efference-heads','--vestibular-sense','--wellness-rank','--controllability',
                 '--learning-progress','--curiosity-mode','--consequence-self-contact','--contact-head',
                 '--contact-control-wellness','--motor-synergy','--synergy-dim','--synergy-babble',
                 '--synergy-residual-l2','--smooth-source','--next-frame-norm','--efference-weight','--wellness-weights'):
        assert settings[flag]['value'].startswith('not present at launch'), flag
    assert '_record' not in state['training'] and '_args' not in state['training']


def test_skin_regions_name_the_body_and_the_time_scope():
    single=dict(n_envs=8,skin_regions={'head':dict(patches=1,self_patches=0),'left hand':dict(patches=0)})
    view=dashboard.skin_regions_view(single)
    assert view['scope']=='one body at the last model step of the window' and view['bodies'][0]['body']==8
    assert list(view['bodies'][0]['regions'])==['head'], 'untouched regions are not listed'
    per=dict(skin_regions_per_body=[{'trunk':dict(patches=2.5,self_patches=0.5)},{}])
    view=dashboard.skin_regions_view(per)
    assert view['scope']=='window means per body' and [b['body'] for b in view['bodies']]==[1,2]
    posture=dashboard.posture_view(dict(body_orientation=[dict(valid=True,roll_degrees=-174.0,maximum_roll_degrees=179.0),dict(valid=False)]))
    assert posture==[dict(body=1,roll=174.0,episode_max=179.0)]


def test_legacy_refresh_keeps_the_newest_48_history_copies(tmp_path,monkeypatch):
    from mimo_bridge.tools import dashboard as old, lifetime
    saved=tmp_path/'Saved/mimo_bridge';history=saved/'dashboard_history';history.mkdir(parents=True)
    names=[f'dashboard_2026-09-{1+i//24:02d}_{i%24:02d}00.html' for i in range(60)]
    for name in names:(history/name).write_text('x')
    monkeypatch.setattr(old,'build',lambda:(saved/'dashboard_legacy.html').write_text('diagnostics'))
    monkeypatch.setattr(lifetime,'build',lambda **kw:None)
    assert dashboard.refresh_legacy(tmp_path)['status']=='updated'
    kept=sorted(p.name for p in history.glob('dashboard_*.html'))
    assert len(kept)==48 and names[0] not in kept and names[-1] in kept


def test_calculated_work_estimates_do_not_promise_learning_dates():
    # The owner explicitly requested calculated work estimates on 2026-09-23.
    root=Path(dashboard.__file__).parent
    page=(root/'development_dashboard.html').read_text(encoding='utf-8')+(root/'development_dashboard.js').read_text(encoding='utf-8')
    assert 'How this estimate is calculated' in page
    assert 'Learning phase expected' not in page
    for banned in ('Training status needs attention','Earlier chooser','live training continues',
                   'Body 1, the baby in the recordings','honest'):
        assert banned not in page, banned
