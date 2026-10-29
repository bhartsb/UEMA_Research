# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Owner-facing evidence must preserve denominators, time bases and test identity."""
import json
from datetime import datetime

import pytest

from mimo_bridge.tools.training_brief import assessment, work_timeline
from mimo_bridge.tools.lineage_series import probe_words
from mimo_bridge.tools import lineage_series


def test_prediction_is_command_aware_evidence_not_a_success_rate():
    record = {'controllability': dict(source='efference', agreement_mean=.43, control_mean=.034,
                                    measured_share=.3125, orders=160, gate_open=True)}
    result = assessment(record, dict(motor_synergy='on', synergy_planner='off', synergy_babble='off'), {})
    assert '43.0%' in result['comparison'] and '3.4%' in result['comparison']
    assert 'same body state' in result['comparison']
    assert '50 of 160 commands' in result['sample'] and 'not a success rate' in result['sample']
    assert 'faster learning' in result['significance']
    record['controllability']['source'] = 'command'
    assert assessment(record, {}, {})['comparison'] is None
    record['controllability']['source'] = 'efference'
    record['controllability']['gate_open'] = False
    assert 'encouraging' not in assessment(record, {}, {})['title']


def test_the_now_summary_and_item_5_read_synergy_use_from_the_same_counts():
    """Item 5's card prints the Now summary's synergy detail (development_dashboard.js), so both decide
    whether the synergy's outputs wrote executed orders from the record's own counts
    (lineage_series.synergy_outputs_used): plans and babble_steps beside the executed orders by author.
    Before, item 5 read the counts and the summary read the launch flags, so one card could say 'not
    directing it' beside 'This run also uses its outputs for planning'."""
    order = dict(part_orders=200, closer=110, chance_count=60)
    flags = [dict(motor_synergy='on', synergy_planner=p, synergy_babble=e) for p in ('on', 'off') for e in ('on', 'off')]
    for plans in (0, 12):
        for babble_steps in (0, 40):
            for authors in (('planner',), ('reflex_noise',), ('planner', 'reflex_noise')):
                record = dict(synergy=dict(plans=plans, babble_steps=babble_steps,
                                           following_by_author={a: dict(order) for a in authors}))
                counted = (plans > 0 and 'planner' in authors) or (babble_steps > 0 and 'reflex_noise' in authors)
                for args in flags:
                    item = next(r for r in lineage_series.gain_of_function(record, args, {})['items'] if r['item'] == 5)
                    brief = assessment(record, args, {})
                    assert ('comparison pending' in item['big']) == counted, (record, args, item['big'])
                    assert (brief['synergy'] == 'Learning coordination and issuing commands') == counted, (record, args)
    # the planner allowed but nothing planned through the synergy: not directing it, by the count
    record = dict(synergy=dict(plans=0, babble_steps=0, following_by_author={'planner': dict(order)}))
    brief = assessment(record, dict(motor_synergy='on', synergy_planner='on'), {})
    assert brief['synergy'] == 'Learning from movement; not yet directing it'
    assert 'not used to plan or explore' in brief['synergy_detail']
    # a record that counts no executed orders by author: the launch settings still describe the run
    assert assessment({}, dict(motor_synergy='on'), {})['synergy'] == 'Learning coordination and issuing commands'


@pytest.mark.parametrize('unit,lived,indirect', [
    ('body position sense, standing in for the balance sense', 'proprioception, standing in for the balance sense this build lacks', True),
    ('the balance sense', 'prediction of its balance sense (vestibular)', False),
    ('share', '', True),
])
def test_posture_result_uses_recorded_sense_not_shared_metric_id(unit, lived, indirect):
    result = probe_words(dict(metric='posture_neck_share_50ms', value=0, unit=unit, lived=lived), 'posture')
    assert result['stand_in'] is indirect
    if not indirect:
        # the sentence names the sense as the model's registry spells it (lineage_series.probe_words)
        assert 'the vestibular prediction foresaw' in result['sentence']
        assert 'balance' not in result['sentence']


def make_run(tmp_path):
    run = tmp_path/'run'; run.mkdir()
    start = datetime(2026, 9, 23, 10).timestamp()
    rows = [dict(type='launch', step=1000, timestamp=start)]
    rows += [dict(type='mimo', step=1000+i*100, timestamp=start+i*200) for i in range(1, 12)]
    (run/'replay_training_stats.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    saves = [dict(reason='hourly', time='2026-09-23 10:10:00', elapsed_min=10),
             dict(reason='hourly', time='2026-09-23 11:10:20', elapsed_min=70+20/60)]
    (run/'checkpoints.json').write_text(json.dumps(saves))
    args = dict(n_envs=8, max_steps=0, minutes=0, report_every=100, checkpoint_every_min=60)
    return rows[-1], args


def test_timeline_uses_differences_in_time_and_steps_and_does_not_schedule_tests(tmp_path):
    record,args = make_run(tmp_path)
    result = work_timeline(tmp_path, {'run':'run'}, record, args, record['timestamp']+10, True)
    assert result['rate']['updates_per_second'] == .5
    assert '1,100 new' in result['experience'] and '1,000 inherited' in result['experience']
    report = next(e for e in result['events'] if e['title']=='Next training report')
    assert report['due'] == record['timestamp']+200
    save = next(e for e in result['events'] if e['title']=='Next saved model')
    assert save['due'] == pytest.approx(datetime(2026,9,23,12,10,40).timestamp())
    assert [e['due'] for e in result['events']] == sorted(e['due'] for e in result['events'])
    assert result['test_status'].startswith('Not scheduled')
    assert 'not measured' in result['test_duration']
    assert 'no configured finishing time' in result['finish']


def test_paused_overdue_and_missing_reports_cannot_get_fresh_estimates(tmp_path):
    record,args = make_run(tmp_path)
    paused = work_timeline(tmp_path, {'run':'run'}, record, args, record['timestamp']+10, False)
    assert all(e['due'] is None for e in paused['events'])
    assert 'last recorded' in paused['practice_estimate']
    overdue = work_timeline(tmp_path, {'run':'run'}, record, args, record['timestamp']+300, True)
    report = next(e for e in overdue['events'] if e['title']=='Next training report')
    assert report['status'].startswith('Overdue') and report['due'] < record['timestamp']+300
    stale = work_timeline(tmp_path, {'run':'run'}, record, args, record['timestamp']+1000, True)
    assert all(e['due'] is None for e in stale['events'])
    args['max_steps'] = 2200
    limited = work_timeline(tmp_path, {'run':'run'}, record, args, record['timestamp'], True)
    assert 'global update 2,200' in limited['finish'] and '3 minutes remain' in limited['finish']


def test_tied_choices_have_no_random_success_baseline_and_negative_gap_is_a_direction():
    record = dict(causal_chain=dict(chosen_best_share=0, chosen_tied_share=1,
                                   alternatives_n=4, gap_to_best_alt_mean=-.1))
    row = lineage_series.gain_of_function(record, {'causal_chain':'on'}, {})['items'][0]
    text = lineage_series.item_text(row)
    assert 'only if exactly one option were distinguishably best' in text
    assert 'below the best alternative by 2%' in text
    assert lineage_series.SERIES['cc.chosen_best']['floor_series'] is None


def test_failed_save_does_not_reuse_old_completed_save_as_the_next_attempt_clock(tmp_path):
    record,args = make_run(tmp_path)
    record['checkpoint_failures'] = 1
    result = work_timeline(tmp_path, {'run':'run'}, record, args, record['timestamp'], True)
    assert not any(e['title']=='Next saved model' for e in result['events'])


def test_history_explains_negative_prediction_scores_and_uses_the_available_series():
    scale = lineage_series.scale_text(lineage_series.SERIES['eff.body_shuffled'])
    assert 'Below 0% means more prediction error' in scale and '-100% means twice' in scale
    panel = dict(title='Movement following', keys=['syn.following', 'syn.following_all'])
    defs = {'syn.following': {}, 'syn.following_all': {'latest':dict(value=.3, step=100, run='r')}}
    result = lineage_series.panel_texts(panel, defs, [])
    assert 'every-command line includes the current controller' in result['scale']
    assert 'each synergy-written order' not in result['scale']
    assert 'every joint moves together' not in lineage_series.scale_text(lineage_series.SERIES['syn.dim'])
