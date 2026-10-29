# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The dashboard reads the newest record instead of pinned paths, and dates what it cannot refresh.

Written by the dashboard staleness check of 2026-09-24 (docs/gain_of_function_v4/agents/
dashboard_staleness.md). The operator found items that looked stale: the Now card named a repair
of segment 36 while production ran segment 39, the model-health card linked a step-344,400 sweep as
its only source, the newest segment's start printed in 24-hour time, the Over time definitions link
opened the 2026-09-17 tree whose dates are out of date, the header's records carried no dates, and
the scheduled comparisons' results never reached the page.
"""
import json
import os
import re
import time
from pathlib import Path

from mimo_bridge.tools import development_dashboard as dd
from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, doubled_sentences, long_numbers, unanchored_numbers

BANNED = re.compile(r"(?i)\b(honest\w*|truthful|barely|slightly|roughly|significantly|substantially|in band|"
                    r"healthy|armed|nominal|a bit)\b")
TWENTY_FOUR_HOUR = re.compile(r"(?<![\d:])(?:[01]\d|2[0-3]):[0-5]\d(?!\s*(?:AM|PM))")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def plain(text, where):
    assert not bare_decimals(text), (where, bare_decimals(text)[:3])
    assert not unanchored_numbers(text), (where, unanchored_numbers(text)[:3])
    assert not long_numbers(text), where
    assert not doubled_sentences(text), where
    assert not BANNED.search(text), (where, BANNED.search(text).group(0))
    assert not TWENTY_FOUR_HOUR.search(text), (where, TWENTY_FOUR_HOUR.search(text).group(0))


# --------------------------------------------------------------------------- clock times
def test_clock_times_are_am_pm_with_a_zone_never_24_hour():
    stamp = time.mktime((2026, 9, 24, 18, 5, 0, 0, 0, -1))
    text = dd.clock(stamp)
    assert text.startswith('Sep 24, 2026, 6:05 PM'), text
    assert not TWENTY_FOUR_HOUR.search(text) and '18:05' not in text
    assert dd.clock(stamp, date=False).startswith('6:05 PM')
    assert dd.clock(None) == 'an unrecorded time' and dd.clock(True) == 'an unrecorded time'
    zone = dd._zone_name(time.localtime(stamp))
    assert ' ' not in zone, 'the zone is abbreviated (PDT), never spelled out'
    assert dd.local_stamp('2026-09-23 00:40') == time.mktime((2026, 9, 23, 0, 40, 0, 0, 0, -1))
    assert dd.clock(dd.local_stamp('2026-09-23 00:40')).startswith('Sep 23, 2026, 12:40 AM')
    assert dd.local_stamp('not a time') is None


def test_the_newest_segment_start_prints_in_am_pm(tmp_path):
    saved = tmp_path/'Saved/mimo_bridge'
    parent, run = saved/'experiments/H9_parent', saved/'experiments/H9_parent_seg2'
    write(parent/'args.json', {})
    write(run/'args.json', {'resume': str(parent/'checkpoints/ckpt_step_100')})
    started = time.mktime((2026, 9, 24, 18, 5, 0, 0, 0, -1))
    (run/'replay_training_stats.jsonl').write_text(
        json.dumps(dict(type='launch', timestamp=started, step=100, code_commit='abc1234')) + '\n'
        + json.dumps(dict(type='mimo', step=150, timestamp=started+60)) + '\n', encoding='utf-8')
    changed = dd.what_changed(tmp_path, {str(run): 150, str(parent): 100})
    assert 'started Sep 24, 2026, 6:05 PM' in changed and '18:05' not in changed, changed


# --------------------------------------------------------------------------- the Now card's present work
def catalog_fixture(root):
    catalog = json.loads((dd.REPO/'docs/gain_of_function_v3/dashboard_goal_catalog.json').read_text(encoding='utf-8'))
    catalog['activity'] = dict(kind='repair', title='Old repair', start_step=380200, status_label='Production remains saved and paused')
    write(root/'docs/gain_of_function_v3/dashboard_goal_catalog.json', catalog)
    return catalog


def production(root, step, stamp, supervisor_stamp, status='training', records=None):
    saved = root/'Saved/mimo_bridge'
    run = saved/'experiments/H84_fresh_drivers_20260918_seg39_resume'
    run.mkdir(parents=True, exist_ok=True)
    rows = records if records is not None else [dict(type='mimo', step=step, timestamp=stamp, model_steps_per_s=0.5)]
    (run/'replay_training_stats.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    write(saved/'training_continuation.json', dict(active_run=str(run), status=status, timestamp=supervisor_stamp))
    return run


def scheduler(root, tasks, window=None, stamp=1000):
    write(root/dd.ABILITY_MANIFEST, dict(schema=1, lineage='H84', tasks=[
        dict(id=k, title=v['title'], first_step=380500, every_steps=3000, memory_gib=12) for k, v in tasks.items()]))
    state = dict(timestamp=stamp, state='running' if window else 'waiting', tasks=tasks)
    if window:
        state['comparison_window'] = window
        state['active_task'] = next((k for k, v in tasks.items() if v.get('status') == 'running'), None)
    write(root/dd.ABILITY_STATUS, state)
    return state


def test_a_finished_repair_card_gives_way_to_live_production_and_the_notes_are_dated(tmp_path):
    catalog_fixture(tmp_path)
    production(tmp_path, 396100, 990, 995)
    scheduler(tmp_path, {'gaze': dict(title='Eye comparison', due_step=396950, status='complete'),
                         'body': dict(title='Body comparison', due_step=399050, status='complete')})
    data = dd.goal_catalog(tmp_path, now=1000)
    activity = data['activity']
    assert activity['kind'] == 'production' and activity['state'] == 'running'
    assert activity['step'] == 396100 and activity['run_name'].endswith('seg39_resume')
    assert (activity['done'], activity['total']) == (2150, 3000)
    assert activity['seconds_remaining'] == 850/0.5
    assert activity['next'] == 'Eye comparison at step 396,950' and activity['then'] == 'Body comparison at step 399,050'
    assert data['superseded_activity']['saved_step'] == 380200
    assert 'saved at step 380,200' in data['staleness'] and 'lineage step 396,100' in data['staleness']
    # no launch record past the saved step in this fixture: the note never claims a resume it cannot see
    assert 'resumed' not in data['staleness'] and 'moved past that step' in data['staleness']
    assert data['as_of'].startswith('Notes written ') and 'before production resumed' not in data['as_of']
    assert activity['next_calculation'].startswith('850 model steps ÷ 0.50 model steps per second')
    assert activity['then_due_at'] is None and 'No clock time' in activity['then_calculation']
    for text in (data['staleness'], data['as_of'], activity['status_label'], activity['purpose'], activity['estimate_basis'],
                 activity['next_detail'], activity['next'], activity['then'], activity['next_calculation'],
                 activity['then_calculation']):
        plain(text, 'production card')


def resumed_lineage(root, written_after=False):
    """Segment 37 resumed past the repair stop at step 380,200 (with its continuation receipt and the
    qualification it names), segment 39 resumed from it later; the catalogue written before or after."""
    saved = root/'Saved/mimo_bridge/experiments'
    seg37, seg39 = saved/'H84_x_seg37_hand_vision', saved/'H84_x_seg39'
    write(seg37/'args.json', {'resume': str(saved/'H84_x_seg36/checkpoints/ckpt_step_380200')})
    write(seg39/'args.json', {'resume': str(seg37/'checkpoints/ckpt_step_390650')})
    write(saved/'H84_x_seg36/args.json', {})
    (seg37/'replay_training_stats.jsonl').write_text(
        json.dumps(dict(type='launch', timestamp=400, step=380200, code_commit='a1')) + '\n', encoding='utf-8')
    (seg39/'replay_training_stats.jsonl').write_text(
        json.dumps(dict(type='launch', timestamp=900, step=390650, code_commit='b2')) + '\n'
        + json.dumps(dict(type='mimo', step=396100, timestamp=990, model_steps_per_s=0.5)) + '\n', encoding='utf-8')
    proof = root/'private/qualification/result.json'
    write(proof, dict(passed=True, completed_at=390, steps=300, checks=dict(
        initial_learning_state_preserved=True, eight_worlds_advanced_exactly=True, no_world_resets=True)))
    write(seg37/'continuation_receipt.json', dict(reason='Resume the preserved learner and eight worlds.', proof=str(proof)))
    write(root/'Saved/mimo_bridge/training_continuation.json', dict(active_run=str(seg39), status='training', timestamp=995))
    catalog = catalog_fixture(root)
    path = root/'docs/gain_of_function_v3/dashboard_goal_catalog.json'
    os.utime(path, (950, 950) if written_after else (300, 300))
    return catalog


def test_the_resume_past_the_repair_stop_joins_the_repair_history_and_dates_the_notes(tmp_path):
    resumed_lineage(tmp_path)
    data = dd.goal_catalog(tmp_path, now=1000)
    assert data['superseded_activity']['resumed_at'] == 400
    assert data['as_of'] == f"Notes written {dd.clock(300)}, before production resumed at {dd.clock(400)}"
    assert f"Training resumed at {dd.clock(400)}; its newest record is lineage step 396,100" in data['staleness']
    event = data['events'][-1]
    assert event['phase'] == 'Resumed' and event['step'] == 380200 and event['source_available']
    assert event['impact'] == (f"Training resumed at {dd.clock(400)} as segment 37 from saved step 380,200: Resume the "
                               f"preserved learner and eight worlds. Its declared qualification passed at {dd.clock(390)} "
                               "over 300 private updates.")
    assert event['retained'] == ('Qualification checks passed: learning state preserved; all eight worlds advanced '
                                 'exactly; no world reset.')
    assert event['next'].startswith(f"Training has restarted 1 more time since, most recently at {dd.clock(900)} from "
                                    "step 390,650 (segment 39).")
    for text in (event['impact'], event['retained'], event['next'], data['staleness'], data['as_of']):
        plain(text, 'resume event')


def test_a_catalogue_saved_after_the_resume_is_not_dated_before_it(tmp_path):
    resumed_lineage(tmp_path, written_after=True)
    data = dd.goal_catalog(tmp_path, now=1000)
    assert data['as_of'] == f"Notes written {dd.clock(950)}"
    assert data['staleness'].startswith(f"Written {dd.clock(950)}, but its Now card still described production saved at "
                                        f"step 380,200, which training passed at {dd.clock(400)}")


def test_pauses_window_failures_and_a_missing_scheduler_say_what_they_are(tmp_path):
    catalog_fixture(tmp_path)
    run = production(tmp_path, 396100, 100, 995, status='paused for repair or explicit instruction')
    data = json.loads((tmp_path/'Saved/mimo_bridge/training_continuation.json').read_text())
    write(tmp_path/'Saved/mimo_bridge/training_continuation.json', dict(data, pause_reason='explicit supervisor pause'))
    activity = dd.production_activity(tmp_path, checks={}, now=1000)
    assert activity['state'] == 'paused' and activity['status_label'] == 'Training paused: explicit supervisor pause.'
    failed = dict(timestamp=990, tasks={}, comparison_window=dict(state='repair required', checkpoint=str(run/'checkpoints/ckpt_step_396050'),
                                                                 reason='A stop superseded the scheduled comparison.'))
    activity = dd.production_activity(tmp_path, checks=failed, now=1000)
    assert activity['state'] == 'attention'
    assert activity['status_label'] == ('The scheduled-comparison pause ended in "repair required": A stop superseded the '
                                        'scheduled comparison.')
    production(tmp_path, 396100, 990, 995)
    missing = dd.production_activity(tmp_path, checks={}, now=1000)
    assert missing['state'] == 'running' and 'status file is missing' in missing['status_label']
    assert missing['scheduler_overdue'] is True


def test_a_task_the_manifest_no_longer_lists_is_not_scheduled(tmp_path):
    catalog_fixture(tmp_path)
    production(tmp_path, 396100, 990, 995)
    scheduler(tmp_path, {'gaze': dict(title='Eye comparison', due_step=396950, status='complete')})
    state = json.loads((tmp_path/dd.ABILITY_STATUS).read_text())
    state['tasks']['retired'] = dict(title='Retired comparison', due_step=396500)
    write(tmp_path/dd.ABILITY_STATUS, state)
    activity = dd.production_activity(tmp_path, now=1000)
    assert activity['next'] == 'Eye comparison at step 396,950'


def test_upgrade_cards_carry_the_newest_scheduled_round_and_drop_the_untested_claim():
    panel = dict(items=[dict(item=7, test_scope='Last saved-model test: update 362,325. It predates this restart; the '
                                                'current version has not been tested here.'), dict(item=5), dict(item=2)])
    results = dict(rows=[dict(id='gaze_prediction', latest=dict(described=True, step=393950, when='Sep 24, 2026, 6:17 PM PDT',
                                                                   headline='Eye movement 150 ms ahead was predicted well.')),
                         dict(id='body_and_synergy', latest=dict(described=True, step=396050, when='Sep 24, 2026, 7:20 PM PDT',
                                                                    headline='Neither planner beat the starting controller.'))])
    items = {i['item']: i for i in dd.link_scheduled(panel, results)['items']}
    assert items[7]['test_scope'] == ('Newest scheduled comparison: checkpoint 393,950, finished Sep 24, 2026, 6:17 PM PDT. '
                                      'Eye movement 150 ms ahead was predicted well. The probe reading below is from update 362,325.')
    assert 'has not been tested here' not in items[7]['test_scope']
    assert items[5]['test_scope'].startswith('Newest scheduled comparison: checkpoint 396,050')
    assert 'test_scope' not in items[2]


def test_24_hour_times_written_elsewhere_print_in_am_pm():
    text = dd.am_pm_text('Waits for the queue runner (its state file was last written 2026-09-15 05:29).')
    assert text == f"Waits for the queue runner (its state file was last written {dd.clock(dd.local_stamp('2026-09-15 05:29'))})."
    assert '05:29' not in text and '5:29 AM' in text
    tree = dd.am_pm_tree(dict(when='Running since 2026-09-13 17:56:58 at D:/x', id='job-2026-09-13 17:56', run_dir='D:/2026-09-13 17:56'))
    assert '5:56 PM' in tree['when'] and tree['id'] == 'job-2026-09-13 17:56' and tree['run_dir'] == 'D:/2026-09-13 17:56'


def test_a_build_page_whose_writer_stopped_says_so(tmp_path):
    page, data = tmp_path/'docs/build_status.html', tmp_path/'Saved/mimo_bridge/build_status.json'
    page.parent.mkdir(parents=True)
    page.write_text('<html></html>', encoding='utf-8')
    write(data, dict(every_s=15))
    os.utime(data, (1000, 1000))
    fresh = {r['label']: r for r in dd.project_records(tmp_path, now=1030)}['Build progress']
    assert fresh['stale'] is False and fresh['updated'].startswith('its data file last written')
    stopped = {r['label']: r for r in dd.project_records(tmp_path, now=1100)}['Build progress']
    assert stopped['stale'] is True and stopped['updated'] == 'its writer stopped; data last written ' + dd.clock(1000)


def test_a_repair_card_stays_while_production_is_still_at_its_saved_step(tmp_path):
    catalog_fixture(tmp_path)
    production(tmp_path, 380200, 990, 995)
    data = dd.goal_catalog(tmp_path, now=1000)
    assert data['activity']['kind'] == 'repair' and 'staleness' not in data
    assert data['written_text'] == dd.clock((tmp_path/'docs/gain_of_function_v3/dashboard_goal_catalog.json').stat().st_mtime)


def test_the_comparison_window_card_counts_done_and_names_the_running_comparison(tmp_path):
    catalog_fixture(tmp_path)
    run = production(tmp_path, 396000, 700, 995)
    tasks = {'body': dict(title='Body comparison', due_step=399050, completed_step=396050, status='complete'),
             'touch': dict(title='Touch comparison', due_step=395950, completed_step=392950, status='running'),
             'gaze': dict(title='Eye comparison', due_step=396950, status='complete')}
    window = dict(state='ready', checkpoint=str(run/'checkpoints/ckpt_step_396050'), targets={'body': 396025, 'touch': 395950})
    scheduler(tmp_path, tasks, window, stamp=990)
    activity = dd.goal_catalog(tmp_path, now=1000)['activity']
    assert activity['state'] == 'comparisons' and activity['title'] == 'Scheduled comparisons on saved step 396,050'
    assert (activity['done'], activity['total']) == (1, 2)
    assert activity['status_label'] == ('Training is saved and paused while the comparisons use its memory: 1 of 2 done; '
                                        'running now: Touch comparison.')
    assert activity['next'] == 'Eye comparison at step 396,950' and activity['next_remaining_updates'] == 900
    assert activity['next_due_at'] is None, 'no time is calculated while training is paused'
    assert activity['then'] == 'Body comparison at step 399,050'


def test_a_relaunched_run_without_a_training_record_says_so(tmp_path):
    catalog_fixture(tmp_path)
    production(tmp_path, None, None, 995, records=[dict(type='launch', timestamp=980, step=396050, code_commit='abc')])
    scheduler(tmp_path, {'gaze': dict(title='Eye comparison', due_step=396950, status='complete')})
    data = dd.goal_catalog(tmp_path, now=1000)
    assert data['activity']['state'] == 'starting' and 'relaunched from saved step 396,050' in data['activity']['status_label']
    assert 'staleness' in data


def test_a_scheduler_that_stopped_reporting_is_named_on_the_next_card(tmp_path):
    catalog_fixture(tmp_path)
    production(tmp_path, 396100, 990, 995)
    scheduler(tmp_path, {'gaze': dict(title='Eye comparison', due_step=396950, status='complete')}, stamp=100)
    activity = dd.production_activity(tmp_path, now=1000)
    assert activity['scheduler_overdue'] is True
    assert 'The scheduler last reported' in activity['next_detail'] and 'no comparison starts' in activity['next_detail']
    scheduler(tmp_path, {'gaze': dict(title='Eye comparison', due_step=396950, status='complete')}, stamp=995)
    assert 'scheduler_overdue' not in dd.production_activity(tmp_path, now=1000)


def test_a_reached_due_step_reads_overdue_never_zero_updates_away(tmp_path):
    catalog_fixture(tmp_path)
    production(tmp_path, 396100, 990, 995)
    scheduler(tmp_path, {'gaze': dict(title='Eye comparison', due_step=396050, status='complete')}, stamp=100)
    activity = dd.production_activity(tmp_path, now=1000)
    assert activity['next_remaining_updates'] is None, 'a reached step has no updates left to count'
    assert activity['next_overdue'] == ('Overdue: due at step 396,050; this check has not run; '
                                        'production training is running.')
    assert activity['scheduler_status'].startswith('Scheduler status unavailable since')
    control = tmp_path/'Saved/mimo_bridge/supervisor_control.json'
    control.write_text('{"cmd": "pause"}')
    assert dd.production_activity(tmp_path, now=1000)['next_overdue'].endswith('production training is paused.')
    control.write_text('{}')
    assert dd.production_activity(tmp_path, now=5000)['next_overdue'].endswith('the training status is unavailable.')
    scheduler(tmp_path, {'gaze': dict(title='Eye comparison', due_step=396950, status='complete')}, stamp=995)
    ahead = dd.production_activity(tmp_path, now=1000)
    assert ahead['next_overdue'] is None and ahead['next_remaining_updates'] == 850


def test_without_a_supervisor_record_the_catalogue_activity_is_kept(tmp_path):
    catalog_fixture(tmp_path)
    assert dd.production_activity(tmp_path, now=1000) is None
    assert dd.goal_catalog(tmp_path, now=1000)['activity']['kind'] == 'repair'


# --------------------------------------------------------------------------- scheduled comparison results
BODY = dict(bodies=8, hold_seconds=0.25, summary=dict(
    physical_goal_error=dict(against_controller=dict(direct=dict(paired_bootstrap95=[0.05, 0.15], bodies_better=1),
                                                     synergy=dict(paired_bootstrap95=[0.01, 0.09], bodies_better=2))),
    hand_joint_error_deg=dict(means=dict(controller=5.2, direct=5.5, synergy=5.4))))
TOUCH = dict(passed_private_comparison=False, evaluated_held_out_trials=16, selected_touches=2,
             selected_patch_physically_reachable_trials=3, comparisons=dict(
    controller=dict(trials=16, selected_success_fraction=0.125, comparison_success_fraction=0.125),
    hold=dict(trials=16, selected_success_fraction=0.125, comparison_success_fraction=0.0625),
    mean_exploration=dict(trials=16, selected_success_fraction=0.125, comparison_success_fraction=0.109375)))
UNTOUCHED = dict(TOUCH, selected_touches=0, selected_patch_physically_reachable_trials=0)
GAZE = dict(summary=dict(saved_starts={'150ms': dict(
    eye_prediction_error_degrees=dict(candidate_mean=2.7, comparison_mean=6.18, error_reduction_fraction=0.564,
                                      candidate_error_lower_beyond_sample_uncertainty=True),
    visual_prediction_mse=dict(candidate_mean=0.012, comparison_mean=0.0125, error_reduction_fraction=0.0374,
                               candidate_error_lower_beyond_sample_uncertainty=False))}))
HAND = dict(comparisons=24, summary=dict(curl=dict(fixed_world=30.8, moving_wrist=10.8),
                                         extension=dict(fixed_world=32.1, moving_wrist=13.5),
                                         index_selective=dict(fixed_world=29.7, moving_wrist=7.6)))


def receipts(root, task, payloads, failed=0):
    history = []
    for i, payload in enumerate(payloads):
        folder = root/dd.ABILITY_STATUS.rsplit('/', 1)[0]/f'{task}_{380550+3000*i}'
        write(folder/'result.json', payload)
        history.append(dict(status='complete', attempt_step=380550+3000*i, started_at=1000+i*100,
                            completed_at=1060+i*100, result=str(folder/'result.json')))
    history += [dict(status='failed', attempt_step=380550, reason='The child exited') for _ in range(failed)]
    return history


def test_each_scheduled_comparison_shows_its_newest_result_in_plain_sentences(tmp_path):
    tasks = {'body_and_synergy': dict(title='Body goals', due_step=392950, status='complete',
                                      history=receipts(tmp_path, 'body_and_synergy', [BODY, BODY])),
             'chosen_self_touch': dict(title='Self-touch', due_step=392950, status='complete',
                                       history=receipts(tmp_path, 'chosen_self_touch', [UNTOUCHED, TOUCH])),
             'gaze_prediction': dict(title='Eyes', due_step=392950, status='complete',
                                     history=receipts(tmp_path, 'gaze_prediction', [GAZE, GAZE], failed=3)),
             'hand_following': dict(title='Hands', due_step=392950, status='running',
                                    history=receipts(tmp_path, 'hand_following', [HAND])),
             'new_task': dict(title='A comparison added later', due_step=399050, status='waiting',
                              history=receipts(tmp_path, 'new_task', [dict(anything=1)]))}
    scheduler(tmp_path, tasks)
    out = dd.ability_results(tmp_path)
    rows = {r['id']: r for r in out['rows']}
    body = rows['body_and_synergy']['latest']
    assert body['step'] == 383550 and body['seconds'] == 60 and body['url'].endswith('/body_and_synergy_383550/result.json')
    assert body['headline'] == ('Neither planner beat the starting controller: the synergy planner came closer to the goal '
                                'in 2 of 8 bodies, the direct planner in 1 of 8.')
    assert 'exceeds the sample uncertainty against the synergy planner and the direct planner' in body['detail']
    assert rows['body_and_synergy']['trend'] == ('In 2 completed rounds since step 380,550, neither planner beat the '
                                                 'controller beyond the sample uncertainty.')
    touch = rows['chosen_self_touch']['latest']['headline']
    assert touch == ('The chosen command touched its chosen skin patch in 2 of 16 held-out starts; some command touched '
                     'that patch in 3. It did not beat every alternative beyond the sample uncertainty, the declared pass.')
    assert rows['chosen_self_touch']['trend'] == ('Passed its declared comparison in 0 of 2 completed rounds since step '
                                                  '380,550. In 1 of them no command touched the skin being compared.')
    eyes = rows['gaze_prediction']
    assert eyes['latest']['headline'] == ('Eye movement 150 ms ahead was predicted with 56% less error than assuming the '
                                          'eyes stay still (2.7 degrees against 6.2 degrees).')
    assert 'Only the eye-angle difference exceeds the sample uncertainty.' in eyes['latest']['detail']
    assert eyes['failed'] == 3 and '3 further attempts ended without a result' in eyes['trend']
    hands = rows['hand_following']['latest']
    assert hands['headline'] == ('Hand targets that follow the wrist ended 7.6 to 14 degrees from the finger goals after '
                                 '1 second, against 30 to 32 degrees with the old fixed-world targets.')
    assert rows['new_task']['latest']['headline'] == 'Completed; the result file holds its numbers.'
    assert out['completed'] == 8 and out['comparisons'] == 5 and out['first_step'] == 380550
    assert out['summary'].startswith('8 scheduled physical comparisons have finished on saved checkpoints')
    assert out['lead'] == ("In the newest rounds, eye forecasts beat 'the eyes stay still'; wrist-following hand targets "
                           "followed better than the old targets (a controls test); the learned planners did not beat the "
                           "starting controller; the chosen self-touch did not beat its alternatives.")
    assert rows['body_and_synergy']['next'] == 'Next round at step 392,950.'
    assert 'Each command ran twice and replayed exactly.' in body['detail'] and '0.25 seconds later' in body['detail']
    assert 'the wrist-following run of each replayed exactly' in hands['detail']
    assert 'from one saved model step in each of the 8 bodies' in eyes['latest']['detail']
    for row in out['rows']:
        latest = row['latest']
        for text in (latest['headline'], latest['detail'], latest['limit'], row.get('trend') or '', latest['when']):
            plain(text, row['id'])
    plain(out['summary'], 'summary')
    plain(out['lead'], 'lead')
    empty = dd._touch_words(UNTOUCHED)['headline']
    assert empty == ('No command touched the chosen skin patches in any of the 16 held-out starts, so this round cannot '
                     'tell the choice apart from the alternatives.')


def test_a_result_outside_the_project_or_unreadable_is_skipped(tmp_path):
    history = [dict(status='complete', attempt_step=380550, result=str(tmp_path.parent/'outside.json')),
               dict(status='complete', attempt_step=383550, result=str(tmp_path/'missing.json'))]
    scheduler(tmp_path, {'body_and_synergy': dict(title='Body goals', due_step=386550, history=history)})
    row = dd.ability_results(tmp_path)['rows'][0]
    assert row['completed'] == 0 and 'latest' not in row


def test_the_real_scheduler_results_print_plain_sentences():
    """Every sentence the live page would print from this repository's scheduler receipts."""
    if not (dd.REPO/dd.ABILITY_STATUS).is_file():
        return
    out = dd.ability_results(dd.REPO)
    for row in out['rows']:
        latest = row.get('latest') or {}
        for text in (latest.get('headline', ''), latest.get('detail', ''), latest.get('limit', ''), row.get('trend') or ''):
            plain(text, row['id'])


def test_the_test_schedule_names_the_scheduled_comparison_and_the_timeline_carries_it():
    results = dict(rows=[dict(id='body_and_synergy', title='Reachable body goals and coordinated control', every_steps=3000, due_step=399050,
                              latest=dict(step=396050, when='Sep 24, 2026, 7:20 PM PDT', seconds=391))])
    activity = dict(kind='production', step=396100, rate=0.5, next_step=396950, next_due_at=2700.0,
                    next='Predict eye movement and the resulting view at step 396,950')
    brief = dd.comparison_schedule(dict(test_status='Not scheduled: no launch or completion date is recorded.', events=[]),
                                   results, activity, dict(timestamp=990), 1000, True)
    assert brief['test_status'].startswith("Scheduled: 'Reachable body goals and coordinated control' compares the "
                                           "synergy planner with the starting controller on a saved checkpoint every 3,000")
    assert 'latest at step 396,050 (Sep 24, 2026, 7:20 PM PDT); next at step 399,050' in brief['test_status']
    assert brief['test_scheduled'] is True and brief['test_duration'].startswith('Its latest round took 7 minutes')
    event = brief['events'][0]
    assert event['title'] == 'Next scheduled physical comparison' and event['status'] == 'Estimated' and event['due'] == 2700.0
    assert event['calculation'].startswith('850 model steps ÷ 0.50 model steps per second')
    assert brief['test_status'].endswith('No scheduled comparison covers synergy exploration.')
    explored = dict(rows=results['rows']+[dict(id='synergy_exploration', title='Synergy exploration against per-joint noise')])
    assert 'exploration' not in dd.comparison_schedule(dict(events=[]), explored, activity, dict(timestamp=990), 1000, True)['test_status']
    paused = dd.comparison_schedule(dict(events=[]), results, activity, dict(timestamp=990), 1000, False)
    assert paused['events'][0]['due'] is None and paused['events'][0]['status'].startswith('Estimate unavailable')
    for text in (brief['test_status'], brief['test_duration'], event['detail'], event['calculation']):
        plain(text, 'schedule')


# --------------------------------------------------------------------------- model health sources
def test_model_health_names_its_current_record_and_links_each_earlier_file_it_prints():
    stamp = time.mktime((2026, 9, 24, 19, 11, 0, 0, 0, -1))
    card = dd.liveness_card({'step': 396000, 'timestamp': stamp, 'liveness': {'layers': {'forward_model.blocks.0.ffn.0': 1.0}}})
    assert card['source_text'].startswith('Readings from the newest training record: lineage step 396,000, written Sep 24, 2026, 7:11 PM')
    assert card['source'] is None, 'the step-344,400 sweep is not the source of current readings'
    labels = [h['label'] for h in card['history_sources']]
    assert labels and all('ckpt_step_344400' in label for label in labels)
    assert not any('Liveness sweep' in label for label in labels), 'the sweep is linked only when its lines are printed'
    text = ' '.join(ln['text'] for ln in card['lines'])
    assert 'measured Sep 23, 2026, 12:40 AM' in text and '00:40' not in text
    known = dd.liveness_card({})
    assert known['source'].endswith('liveness_sweep_ckpt_step_344400.json')
    assert any('Liveness sweep of ckpt_step_344400 (2026-09-22)' == h['label'] for h in known['history_sources'])


# --------------------------------------------------------------------------- header, definitions
def test_project_records_carry_their_write_times_and_the_build_page_its_data_time(tmp_path):
    review = tmp_path/'docs/gain_of_function_v3/MOTOR_CONTROL_REVIEW_2026-09-23.md'
    review.parent.mkdir(parents=True)
    review.write_text('review', encoding='utf-8')
    page, data = tmp_path/'docs/build_status.html', tmp_path/'Saved/mimo_bridge/build_status.json'
    page.write_text('<html></html>', encoding='utf-8')
    write(data, {})
    old = time.mktime((2026, 9, 23, 15, 16, 0, 0, 0, -1))
    os.utime(data, (old, old))
    records = {r['label']: r for r in dd.project_records(tmp_path)}
    assert set(records) == {'Motor control review', 'Build progress'}
    assert records['Build progress']['updated'] == 'its writer stopped; data last written ' + dd.clock(old)
    assert records['Motor control review']['updated'].startswith('file last written ')
    assert all('Current investigation' not in r['label'] for r in records.values())


def test_the_explainer_link_says_when_and_from_which_run_it_was_built(tmp_path):
    page = tmp_path/'Saved/mimo_bridge/explainer.html'
    page.parent.mkdir(parents=True)
    page.write_text('<!DOCTYPE html>\n<html lang="en">\n<head>\n<title>LEM explainer — H84_fresh_drivers_20260918_seg38_observer</title>',
                    encoding='utf-8')
    built = time.mktime((2026, 9, 24, 6, 43, 0, 0, 0, -1))
    os.utime(page, (built, built))
    state = dd.explainer_state(tmp_path, dict(run_name='H84_fresh_drivers_20260918_seg39_resume_20260924_193821'))
    assert state['label'].startswith('built Sep 24, 2026, 6:43 AM') and state['label'].endswith('from segment 38')
    assert state['current'] is False and 'Training is now in segment 39' in state['detail']
    same = dd.explainer_state(tmp_path, dict(run_name='H84_fresh_drivers_20260918_seg38_observer'))
    assert same['current'] is True and 'Training is now' not in same['detail']
    assert dd.explainer_state(tmp_path/'nothing', {}) is None


def test_the_2026_09_13_abilities_link_their_september_13_definitions():
    """developmental_status names the page that defines its 25 abilities (the 2026-09-13 registry):
    the September 13 revision, never the 2026-09-17 tree whose dates run September 16 to 19."""
    from mimo_bridge.tools.developmental_status import build
    development = build(dd.REPO, {'jobs': []})
    assert development['definitions'] == '/files/docs/MIMo_Gain_of_Function_Tree_2026-09-06.html'
    assert development['definitions_label'] == 'Definitions and tests (revised 2026-09-13, kept as history)'
    page = dd.REPO/'docs/MIMo_Gain_of_Function_Tree_2026-09-06.html'
    assert 'revised September 13' in page.read_text(encoding='utf-8')
    assert len(development['stages']) == 4 and sum(len(s['abilities']) for s in development['stages']) == 25


def test_the_page_carries_no_undated_current_labels_or_24_hour_clocks():
    root = Path(dd.__file__).parent
    page = (root/'development_dashboard.html').read_text(encoding='utf-8')
    script = (root/'development_dashboard.js').read_text(encoding='utf-8')
    assert 'Current investigation' not in page
    assert 'MIMo_Gain_of_Function_Tree_2026-09-17.html' not in page
    assert 'id="project-records"' in page and 'id="explainer-age"' in page and 'data-catalog-note' in page
    for name in ('dated', 'clockAt'):
        line = next(ln for ln in script.splitlines() if ln.startswith(f'const {name}='))
        assert "'en-US'" in line and "hour:'numeric'" in line and "timeZoneName:'short'" in line, line
    assert 'hourCycle' not in script and 'hour12:false' not in script


# --------------------------------------------------------------------------- the newer scheduled comparisons
def planner_payload(tmp_path, **rows):
    """A planner-candidate result in the probe's own format: its summarise() and decide() over the
    planner track's synthetic rows (tests/test_planner_candidates.py), as its main() assembles it."""
    from test_planner_candidates import P, DRAWS, synthetic_rows
    summary = P.summarise(synthetic_rows(**rows), draws=DRAWS)
    verdict = P.decide(summary)
    payload = dict(probe=P.PROBE_VERSION, checkpoint=str(tmp_path / 'checkpoint'), learning=False, elapsed_s=12.5,
                   bodies=8, starts_per_body=2, evaluated_starts=16, candidates=list(P.CANDIDATES),
                   depths=list(P.DEPTHS), summary=summary, verdict=verdict,
                   passed=verdict['depth3_better_physical_decision'])
    return json.loads(json.dumps(P.clean(payload), allow_nan=False))


def test_the_planner_candidate_result_names_the_first_flat_stage_in_plain_sentences(tmp_path):
    working = dd._planner_words(planner_payload(tmp_path, reverse_first_moment=True))
    assert working['headline'] == ('Forecast and evaluator pass differences on, but the planner does not choose by '
                                   'predicted wellness: its chosen command ranks 3 of 5 by predicted wellness (3 = chance, '
                                   '1 = always the best).')
    assert working['first'] == 'planner' and working['passed'] is True
    assert working['detail'].endswith('a better physical decision, the declared pass.')
    blind = dd._planner_words(planner_payload(tmp_path, forecast=0.0, evaluator=1.0))
    assert blind['headline'].startswith("The planner recheck's first flat stage is the forward model: the forecast carried "
                                        "0% of the difference the candidates' actual outcomes made (0% = it ignores the "
                                        "command")
    assert blind['lead'] == "the planner recheck's first flat stage is the forward model's forecast"
    assert 'Of that miss, the forecast accounts for' in blind['detail']
    judge = dd._planner_words(planner_payload(tmp_path, forecast=1.0, evaluator=0.0, noise=0.02, seed=3))
    assert judge['headline'].startswith("The planner recheck's first flat stage is the evaluator: the evaluator, scoring "
                                        "the actual outcomes, saw")
    assert judge['detail'].endswith('not a better physical decision by the declared rule (interval above zero).')
    for words in (working, blind, judge):
        for text in (words['headline'], words['detail'], words['limit']):
            plain(text, 'planner candidates')


def test_the_hand_memory_result_reads_in_plain_sentences():
    payload = json.loads((dd.REPO/'docs/gain_of_function_v3/measurements/hand_memory_route_2026-09-24.json')
                         .read_text(encoding='utf-8'))
    words = dd._hand_memory_words(payload)
    assert words['headline'] == ('With the learned hand stage the hand ended 14% closer to the remembered hand shape '
                                 'than the same decision without it; 4 of 4 held-out bodies improved, and the round '
                                 'passed its declared admission.')
    assert words['tile'] == '14% less hand-shape error' and words['passed'] is True
    assert 'Fingertip position error 1.7 cm against 1.8 cm' in words['detail']
    assert 'fingertip angle error 31 degrees against 34 degrees' in words['detail']
    for text in (words['headline'], words['detail'], words['limit']):
        plain(text, 'hand memory route')


def test_new_comparisons_show_their_first_round_then_their_rounds(tmp_path):
    payload = planner_payload(tmp_path, reverse_first_moment=True)
    tasks = {'planner_candidates': dict(title='Why planner choices tie', due_step=399050, status='waiting'),
             'hand_memory_route': dict(title='Remembered hand goals', due_step=399050, status='waiting')}
    scheduler(tmp_path, tasks)
    rows = {r['id']: r for r in dd.ability_results(tmp_path)['rows']}
    assert rows['planner_candidates']['next'] == 'First round at step 399,050.' and 'latest' not in rows['planner_candidates']
    tasks['planner_candidates']['history'] = receipts(tmp_path, 'planner_candidates', [payload, payload])
    scheduler(tmp_path, tasks)
    out = dd.ability_results(tmp_path)
    row = {r['id']: r for r in out['rows']}['planner_candidates']
    assert row['next'] == 'Next round at step 399,050.'
    assert row['trend'] == ('First flat stage in 2 completed rounds since step 380,550: planner: 2 rounds. Looking '
                            'further ahead gave a better physical decision in 2 of 2. The result files do not record '
                            'which version of the probe script measured them.')
    assert out['lead'] == "In the newest rounds, the planner recheck's first flat stage is the planner's own selection."
    plain(row['trend'], 'planner trend')


def test_planner_rounds_are_labelled_by_the_probe_script_that_measured_them(tmp_path):
    """The own-command analysis review (2026-10-02): three versions of the planner-candidate probe produced its nine
    rounds, so a series over the rounds mixes measurements whose meaning changed (v3 feeds depths 2 and 3 the
    executed pose and reads a sense below its float32 floor as not measurable). Each round is labelled by the first
    8 characters of its script's SHA-256, the series groups the rounds by it, and the round's text says what a
    production tie is and which reading is the planner-side one."""
    rounds = []
    for k, (sha, step) in enumerate((("1467b636" + "0" * 56, 399100), ("b5cbb3be" + "1" * 56, 408450),
                                     ("b5cbb3be" + "1" * 56, 411125), ("7819ad7a" + "2" * 56, 450525))):
        payload = planner_payload(tmp_path, reverse_first_moment=True)
        payload.update(script_sha256=sha, checkpoint=f"/runs/x/checkpoints/ckpt_step_{step}")
        rounds.append(dd._planner_words(payload))
    assert [w['script'] for w in rounds] == ['1467b636', 'b5cbb3be', 'b5cbb3be', '7819ad7a']
    assert [w['checkpoint_step'] for w in rounds] == [399100, 408450, 411125, 450525]
    assert rounds[-1]['detail'].endswith(f"Measured by the probe script 7819ad7a (the first 8 characters of its "
                                         f"SHA-256, {rounds[-1]['probe']}).")
    text = dd._trend('planner_candidates', rounds, ' since step 399,100')
    assert text.endswith("The rounds come from 3 versions of the probe script, labelled by the first 8 characters of "
                         "its SHA-256 ('unrecorded' = a result file without it): 1467b636, 1 round (step 399,100); "
                         "b5cbb3be, 2 rounds (steps 408,450 to 411,125); 7819ad7a, 1 round (step 450,525). A reading can "
                         "change meaning between versions.")
    same = dd._trend('planner_candidates', rounds[1:3], '')
    assert same.endswith("All 2 rounds come from one version of the probe script, b5cbb3be (the first 8 characters of "
                         "its SHA-256).")
    # what a production tie is, against the round's own band and spread (0.0734 against 0.0052, the spread the
    # saved rechecks measured on the order of; the synthetic rows' own spread is wider than the band), and the
    # planner-side reading of the forward model
    payload = planner_payload(tmp_path, reverse_first_moment=True)
    payload['summary']['evaluator']['candidate_spread_median']['predicted'] = 0.0052
    detail = dd._planner_words(payload)['detail']
    assert ("A production tie is a predicted difference inside the production band, the evaluator's own error on a "
            "difference between two candidates; that band is 14 times the spread the evaluator predicted between the "
            "five candidates here (median over the starts), so every production choice reads tied whatever the forward "
            "model predicts: the tie count is no evidence about the forward model.") in detail
    assert ("The forecast captured 100% of the real difference between two candidates' next tokens (pooled over vision, "
            "tactile, audio and proprioception, 50 ms ahead; 0% = none of it, 100% = all of it; 95% interval") in detail
    assert "This captured share is the planner-side reading of the forward model." in detail
    payload['summary']['evaluator']['candidate_spread_median']['predicted'] = 0.09
    assert "exceeded the production band here, so a production choice could leave the tie band" in \
        dd._planner_words(payload)['detail']
    # a round whose pooled real difference was within the float32 floor says the share is not measurable
    payload['summary']['forward_model']['senses']['planner']['1']['judge_senses']['captured_share'] = dict(
        value=None, interval=None, measurable=False)
    detail = dd._planner_words(payload)['detail']
    assert "so the forecast's captured share is not measurable this round." in detail
    assert "planner-side reading" not in detail
    for w in rounds:
        plain(w['detail'], 'planner round with its script')
    plain(detail, 'planner round, not measurable')
    plain(text, 'planner trend by script')


def test_a_card_can_bind_what_a_tie_is_and_the_captured_share_on_their_own(tmp_path):
    """The goal catalogue's planner cards print production's tie count (the observer's report); since the own-command
    analysis review of 2026-10-02 a tie is no evidence about the forward model, so those cards also bind the round's tie
    sentence and its captured share (the planner-side reading) as the shows 'tie' and 'captured': the same sentences
    the round's detail carries."""
    payload = planner_payload(tmp_path, reverse_first_moment=True)
    payload['summary']['evaluator']['candidate_spread_median']['predicted'] = 0.0052
    words = dd._planner_words(payload)
    assert {'tie', 'captured'} <= set(dd._SCHEDULED_SHOWS)
    assert words['tie'].startswith("A production tie is a predicted difference inside the production band")
    assert words['tie'].endswith("the tie count is no evidence about the forward model.") and words['tie'] in words['detail']
    assert words['captured'].startswith("The forecast captured 100% of the real difference between two candidates' next")
    assert words['captured'].endswith("This captured share is the planner-side reading of the forward model.")
    assert words['captured'] in words['detail']
    payload['summary']['forward_model']['senses']['planner']['1']['judge_senses']['captured_share'] = dict(
        value=None, interval=None, measurable=False)
    words = dd._planner_words(payload)
    assert words['captured'].endswith("so the forecast's captured share is not measurable this round.")
    # bound into a card, they follow the newest round like the headline (the synthetic round's own spread is wider
    # than the band, so its tie sentence is the other branch)
    newest = planner_payload(tmp_path)
    tasks = {'planner_candidates': dict(title='Why planner choices tie', due_step=399050, status='complete',
                                        history=receipts(tmp_path, 'planner_candidates', [newest]))}
    scheduler(tmp_path, tasks)
    rows = {r['id']: r for r in dd.ability_results(tmp_path)['rows']}
    card = dict(checked_at='2099-01-01 00:00', meaning='Choices:',
                measured=[dict(scheduled='planner_candidates', show='tie', into='meaning'),
                          dict(scheduled='planner_candidates', show='captured', into='meaning')])
    dd.bind_card(tmp_path, card, rows, ({}, None))
    said = dd._planner_words(newest)
    assert card['meaning'] == f"Choices: {said['tie']} {said['captured']}"
    assert said['tie'].startswith('The spread the evaluator predicted between the five candidates exceeded')
    assert not card['freshness']['missing']
    for text in (words['captured'], card['meaning']):
        plain(text, 'tie and captured shows')


def test_a_highlight_tied_to_a_scheduled_comparison_shows_its_newest_round():
    data = dict(highlights=[dict(title='Eye-angle prediction', value='52% less error', scope='Saved model at step 362325',
                                 scheduled='gaze_prediction', source_url='/files/old.md', source_available=True),
                            dict(title='Hand route', value='14% less hand-shape error', scope='Step 380,200',
                                 scheduled='hand_memory_route', source_url='/files/route.json', source_available=True),
                            dict(title='Planner repair', value='118 to 2 worse choices', scope='Sep 23')])
    results = dict(rows=[dict(id='gaze_prediction', tile='56% less error', completed=5, first_step=380550,
                              latest=dict(described=True, step=393950, when='Sep 24, 2026, 6:17 PM PDT', url='/files/r.json')),
                         dict(id='hand_memory_route', completed=0)])
    tiles = dd.link_highlights(data, results)['highlights']
    assert tiles[0]['value'] == '56% less error' and tiles[0]['live'] is True and tiles[0]['source_url'] == '/files/r.json'
    assert tiles[0]['scope'] == ('Newest scheduled round: saved step 393,950, finished Sep 24, 2026, 6:17 PM PDT; '
                                 '5 rounds since step 380,550')
    assert tiles[1]['value'] == '14% less hand-shape error' and 'live' not in tiles[1], 'no scheduled result yet'
    assert tiles[2]['value'] == '118 to 2 worse choices'


def test_a_finished_private_test_on_the_now_card_gives_way_to_production(tmp_path, monkeypatch):
    monkeypatch.setattr(dd.time, 'time', lambda: 1000)
    catalog = catalog_fixture(tmp_path)
    catalog['activity'] = dict(kind='experiment', title='A private comparison', run='private/exp/experiments/candidate',
                               result='private/exp/result.json', start_step=380200, updates=300)
    write(tmp_path/'docs/gain_of_function_v3/dashboard_goal_catalog.json', catalog)
    write(tmp_path/'private/exp/result.json', dict(passed=True))
    (tmp_path/'private/exp/experiments/candidate').mkdir(parents=True)
    production(tmp_path, 396100, 990, 995)
    data = dd.goal_catalog(tmp_path, now=1000)
    assert data['activity']['kind'] == 'production'
    assert data['superseded_activity'] == dict(title='A private comparison', state='complete', written_at=data['written_at'])
    assert data['staleness'].startswith(f"Written {data['written_text']}; its Now card named 'A private comparison', "
                                        "which is finished. Production training is the present work: its newest record is "
                                        "lineage step 396,100")
    plain(data['staleness'], 'superseded test')
    # a private test still reporting keeps the card
    (tmp_path/'private/exp/result.json').unlink()
    (tmp_path/'private/exp/experiments/candidate/replay_training_stats.jsonl').write_text(
        json.dumps(dict(type='mimo', step=380300, timestamp=990, model_steps_per_s=1.0)) + '\n', encoding='utf-8')
    assert dd.goal_catalog(tmp_path, now=1000)['activity']['kind'] == 'experiment'


def test_a_held_comparison_shows_its_reason_never_an_out_of_reach_step(tmp_path):
    """The coordinator held two comparisons at 9:14 PM PDT on 2026-09-24 by moving their first step to
    999,999,999 with a 'held' reason. The scheduler kept the eye test's last status ("waiting for
    requested checkpoint"), and the page would have printed "First round at step 999,999,999"."""
    catalog_fixture(tmp_path)
    production(tmp_path, 397300, 990, 995)
    note = ('2026-09-24 9:14 PM PDT: held (first_step 999999999, was 380500) until its replay fix passes its '
            'one-body confirmation. Restore first_step 380500 when the fix is confirmed.')
    write(tmp_path/dd.ABILITY_MANIFEST, dict(schema=1, lineage='H84', tasks=[
        dict(id='gaze', title='Eye comparison', held=note, first_step=999999999, every_steps=3000, memory_gib=12),
        dict(id='body', title='Body comparison', first_step=380500, every_steps=3000, memory_gib=12),
        dict(id='route', title='Remembered hand goals', held='2026-09-24 9:14 PM PDT: held (first_step 999999999, '
             'planned 399050) until its validator is committed.', first_step=999999999, every_steps=3000,
             memory_gib=12)]))
    write(tmp_path/dd.ABILITY_STATUS, dict(timestamp=1000, state='waiting', tasks={
        'gaze': dict(title='Eye comparison', due_step=999999999, status='waiting for requested checkpoint'),
        'body': dict(title='Body comparison', due_step=399050, status='complete'),
        'route': dict(title='Remembered hand goals', due_step=999999999)}))
    rows = {r['id']: r for r in dd.ability_results(tmp_path)['rows']}
    assert rows['gaze']['next'] == ('No round is scheduled. 2026-09-24 9:14 PM PDT: held until its replay fix passes its '
                                    'one-body confirmation. Restore its schedule from step 380,500 when the fix is '
                                    'confirmed.')
    assert rows['gaze']['due_step'] is None and rows['route']['due_step'] is None
    assert rows['body']['next'] == 'First round at step 399,050.'
    view = dd.with_holds(tmp_path, json.loads((tmp_path/dd.ABILITY_STATUS).read_text(encoding='utf-8')))
    assert view['tasks']['gaze']['status'] == 'held' and view['tasks']['gaze']['due_step'] is None
    assert '999' not in json.dumps(view['tasks']['gaze']) and '999' not in json.dumps(rows)
    activity = dd.goal_catalog(tmp_path, now=1000)['activity']
    assert activity['next'] == 'Body comparison at step 399,050' and activity['next_step'] == 399050
    assert '999' not in json.dumps(activity), 'a held comparison is never the next or the later card'
    for text in (rows['gaze']['next'], rows['route']['next'], activity['next'], activity.get('then') or ''):
        plain(text, 'held comparison')
    # a held task the scheduler is still running keeps its own status, and says no later round follows
    running = dict(timestamp=1000, tasks={'gaze': dict(title='Eye comparison', status='running', due_step=999999999),
                                          'body': dict(title='Body comparison', due_step=399050, status='complete')})
    view = dd.with_holds(tmp_path, running)['tasks']['gaze']
    assert view['status'] == 'running' and view['due_step'] is None and view['held'].startswith('2026-09-24 9:14 PM PDT')
    row = {r['id']: r for r in dd.ability_results(tmp_path, running)['rows']}['gaze']
    assert row['next'].startswith('A round is running now; no later round is scheduled. 2026-09-24 9:14 PM PDT: held')
    plain(row['next'], 'running held comparison')


def test_an_overdue_round_says_its_due_step_and_why_it_has_not_run(tmp_path):
    """The dashboard decision of Oct 21, 2026: the page printed "Next round at step 482,550" while training stood at
    489,350, paused. An overdue round now names its due step, the newest training step and whether training is paused;
    a round still ahead keeps "Next round" or "First round"."""
    catalog_fixture(tmp_path)
    production(tmp_path, 400000, 990, 995)
    write(tmp_path/dd.ABILITY_MANIFEST, dict(schema=1, lineage='H84', tasks=[
        dict(id='body', title='Body comparison', first_step=380500, every_steps=3000, memory_gib=12),
        dict(id='later', title='Later comparison', first_step=380500, every_steps=3000, memory_gib=12)]))
    write(tmp_path/dd.ABILITY_STATUS, dict(timestamp=1000, state='waiting', tasks={
        'body': dict(title='Body comparison', due_step=399050, status='complete'),
        'later': dict(title='Later comparison', due_step=401000, status='complete')}))
    rows = {r['id']: r for r in dd.ability_results(tmp_path)['rows']}
    assert rows['body']['next'].startswith('Due at step 399,050; production training stands at step 400,000 and is ')
    assert rows['later']['next'] == 'First round at step 401,000.'
    # paused: the supervisor's pause command decides the ending
    write(tmp_path/'Saved/mimo_bridge/supervisor_control.json', dict(cmd='pause'))
    rows = {r['id']: r for r in dd.ability_results(tmp_path)['rows']}
    assert rows['body']['next'].endswith('and is paused; this round has not run.')
    # running: no pause command, no pause reason, status 'training'
    (tmp_path/'Saved/mimo_bridge/supervisor_control.json').unlink()
    path = tmp_path/'Saved/mimo_bridge/training_continuation.json'
    supervisor = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    supervisor.update(status='training')
    supervisor.pop('pause_reason', None)
    write(path, supervisor)
    rows = {r['id']: r for r in dd.ability_results(tmp_path)['rows']}
    assert rows['body']['next'].endswith('and is running; this round has not run yet.')


# --------------------------------------------------------------------------- the outside review's ranking (2026-09-25)
def ranking_block(forecast, planners_better=True):
    """A tools/command_ranking.py block over 8 bodies x 2 starts of three arms (the controller, then two
    planners 0.5 and 0.8 below or above it), read back from JSON. forecast(actual, body)."""
    import numpy as np
    from mimo_bridge.tools.command_ranking import ranking
    rng = np.random.default_rng(0)
    rows = []
    for start in range(16):
        actual = np.full(3, rng.normal(0.0, 1.0))
        actual[1:] += np.array([-0.5, -0.8]) * (1.0 if planners_better else -1.0)
        rows.append(dict(body=start // 2, names=['controller', 'direct', 'synergy'], actual=actual.tolist(),
                         predicted=list(forecast(actual, start // 2))))
    block = ranking(rows, 'x', planners=('direct', 'synergy'), reference='controller', draws=300)
    return json.loads(json.dumps(block))


LINES = "50 is a coin, 80 to 90 a ranking a planner can optimize"


def test_the_ranking_sentence_is_in_the_words_of_the_statistic():
    used = dd._ranking_sentence(ranking_block(lambda a, b: a), 'the goal')
    assert used == ("On the goal, the forecast picked the actual winner in 48 of 48 pairs of commands (95% interval 100 "
                    f"to 100 in 100; {LINES}), with the commands' forecasts 100% as far apart as their actual outcomes: "
                    "it ranks the commands better than a coin, and beyond the sample uncertainty the planner's command "
                    "beat the command the planner's search starts from.")
    backwards = dd._ranking_sentence(ranking_block(lambda a, b: -7.9 * a), 'the goal')
    assert 'picked the actual winner in 0 of 48 pairs of commands (95% interval 0 to 0 in 100' in backwards
    assert backwards.endswith("forecasts 7.9 times as far apart as their actual outcomes: it picks the loser more often "
                              "than a coin would.")
    # four bodies ordered right and four backwards, forecasts a thousandth as far apart as the outcomes
    under = dd._ranking_sentence(ranking_block(lambda a, b: (0.001 if b < 4 else -0.001) * a), 'the goal')
    assert 'picked the actual winner in 50 of 100 pairs of commands (95% interval' in under
    assert under.endswith("forecasts 0.1% as far apart as their actual outcomes: its ranking is not measurably better than "
                          "a coin (the 95% interval reaches 50) and it separates the commands less than reality does.")
    # a result written at commit fd713b068 carries the label's first name and reads the same
    first_name = dd._ranking_sentence(dict(json.loads(json.dumps(ranking_block(
        lambda a, b: (0.001 if b < 4 else -0.001) * a))), case='case_1_action_insensitive'), 'the goal')
    assert first_name == under
    worse = dd._ranking_sentence(ranking_block(lambda a, b: a, planners_better=False), 'the goal',
                                 reference="the controller's proposal (where its search starts)", extra='a last clause')
    assert worse.endswith("but beyond the sample uncertainty the planner's command did not beat the controller's "
                          "proposal (where its search starts); a last clause.")
    # near the ends the pairs are counted, never rounded to a perfect or an empty score
    near = dd._ranking_sentence(dict(per_100=99.6, pairs_right=239.0, pairs_judged=240, interval=[98.8, 99.7],
                                     spread_ratio=0.4, case='better_than_chance', below_chance=False), 'x')
    assert 'in 239 of 240 pairs of commands (95% interval 98.8 to 99.7 in 100;' in near and '100 of 100' not in near
    low = dd._ranking_sentence(dict(per_100=0.4, pairs_right=1.0, pairs_judged=240, interval=[0.0, 0.6],
                                    spread_ratio=2.0, case='case_2_wrong_order', below_chance=True), 'x')
    assert 'in 1 of 240 pairs of commands (95% interval 0 to under 1 in 100;' in low
    assert dd._ranking_sentence(None, 'x') is None and dd._ranking_sentence(dict(per_100=None), 'x') is None
    for text in (used, backwards, under, worse, near, low):
        plain(text, 'ranking sentence')


def test_the_planner_candidate_result_states_how_often_the_forecast_picks_the_actual_winner(tmp_path):
    payload = planner_payload(tmp_path, reverse_first_moment=True)
    words = dd._planner_words(payload)
    block = payload['summary']['ranking']['goal_distance_end']
    assert words['ranking'].startswith("On distance to the planner's goal (the remembered model step it steers toward) when "
                                       "the command ends, the forecast picked the actual winner in "
                                       f"{round(block['per_100'])} of 100 pairs of commands")
    assert words['ranking'] in words['detail'] and LINES in words['ranking']
    rest = block['without_chosen']
    assert words['ranking'].endswith("the planner's own command was found by searching this same forecast, and over the "
                                     f"other four it picked the actual winner in {round(rest['per_100'])} of 100 pairs "
                                     "of commands.")
    assert words['detail'].endswith('a better physical decision, the declared pass.')
    older = json.loads(json.dumps(payload))
    del older['summary']['ranking']                       # a planner_candidates_v1 round
    earlier = dd._planner_words(older)
    assert earlier['ranking'] is None and 'picked the actual winner' not in earlier['detail']
    for text in (words['detail'], earlier['detail']):
        plain(text, 'planner ranking')


def test_the_body_result_names_every_goal_its_witness_reach_and_the_ranking():
    goals = ['curl_witness', 'extension_witness', 'wrist_reach_witness', 'supported_trunk_turn_witness',
             'elbow_flex_witness']
    reached = dict(curl_witness=(8, 21.0, 30.0, 'degrees'), extension_witness=(8, 18.0, 29.0, 'degrees'),
                   wrist_reach_witness=(6, 0.21, 0.63, 'cm'), supported_trunk_turn_witness=(2, -0.4, 3.9, 'degrees'),
                   elbow_flex_witness=(5, 1.2, 3.9, 'degrees'))
    payload = dict(BODY, goals=goals, goal_suite='hand_and_body',
                   ranking=dict(pooled=ranking_block(lambda a, b: -7.9 * a, planners_better=False),
                                arms=['controller', 'direct', 'synergy'], optimized_arms=['direct', 'synergy']),
                   by_goal={g: dict(witness_check=dict(bodies=8, informative_bodies=n, achieved_beyond_controller=got,
                                                       requested_executed=asked, unit=unit, mirror_share=-0.1))
                            for g, (n, got, asked, unit) in reached.items()})
    words = dd._body_words(payload)
    assert words['suite'] == 'hand_and_body'
    assert ("Goal: where a reference command whose target is to close the hand, open the hand, raise the left wrist, "
            "turn the trunk or bend the left elbow actually took the body from the same saved model step.") in words['detail']
    assert words['ranking'].startswith("On each command's distance from the goal, the forecast picked the actual winner "
                                       "in 0 of 48 pairs of commands")
    assert words['ranking'].endswith("it picks the loser more often than a coin would; 2 of the 3 commands in each trial "
                                     "were found by searching this same forecast, so its errors favor them.")
    assert words['ranking'] in words['detail']
    assert ("The reference command moved the part it asked for further than both the controller's own command and the "
            "same request reversed did: close the hand in 8 of 8 bodies (median over the 8 bodies: 21 degrees "
            "of the 30 degrees requested once blended with the previous command); open the hand in 8 of 8 bodies (median "
            "over the 8 bodies: 18 degrees of the 29 degrees requested once blended with the previous command); raise the "
            "left wrist in 6 of 8 bodies (median over the 8 bodies: 0.21 cm of the 0.63 cm requested once blended with "
            "the previous command); turn the trunk in 2 of 8 bodies (median over the 8 bodies: -0.4 degrees of the 3.9 "
            "degrees requested once blended with the previous command); bend the left elbow in 5 of 8 bodies (median over "
            "the 8 bodies: 1.2 degrees of the 3.9 degrees requested once blended with the previous command).") in words['detail']
    # a round of commit fd713b068 ran no reversed request: the page does not claim one
    earlier = dd._body_words(dict(payload, by_goal={g: dict(witness_check=dict(bodies=8, informative_bodies=n))
                                                    for g, (n, *_rest) in reached.items()}))
    assert "further than the controller's own command did: close the hand in 8 of 8 bodies;" in earlier['detail']
    assert 'reversed' not in earlier['detail']
    old = dd._body_words(BODY)                             # a round written before the ranking existed
    assert old['ranking'] is None and 'close the hand or open the hand' in old['detail'] and old['suite'] == 'hand'
    assert 'picked the actual winner' not in old['detail'] and 'further than the controller' not in old['detail']
    for text in (words['detail'], words['headline'], old['detail'], earlier['detail']):
        plain(text, 'body ranking')
    # the rounds' trend says when the suites differ
    trend = dd._trend('body_and_synergy', [old, words], ' since step 380,550')
    assert trend.endswith("1 of those rounds also had the wrist, trunk and elbow goals; the other the hand goals alone.")
    trend = dd._trend('body_and_synergy', [old, old, words], ' since step 380,550')
    assert trend.endswith("1 of those rounds also had the wrist, trunk and elbow goals; the other 2 the hand goals alone.")
    plain(trend, 'body trend')


# --------------------------------------------------------------------------- the capability programme's cards (2026-09-30)
# Operator, 2026-09-30: "The 'Build accurate, repeatable control' tab is completely out of date. The dashboard is
# supposed to be always kept current and 100% correct!" Every card's numbers come from its newest result file; its
# hand-written text only explains; every card says when its text was checked and is marked out of date when a
# result newer than that check exists.
def catalogue():
    return json.loads((dd.REPO/dd.CATALOG).read_text(encoding='utf-8'))


def cards(catalog):
    for group in dd.CARD_GROUPS:
        for card in catalog.get(group, []):
            yield group, card


def test_every_card_says_when_its_text_was_checked_and_its_text_carries_no_typed_measurement():
    catalog = catalogue()
    fixed = [d['text'] for d in catalog.get('fixed_definitions', [])]
    assert fixed and all(d.get('why') for d in catalog['fixed_definitions']), 'each fixed definition says why it is fixed'
    for group, card in cards(catalog):
        name = f"{group}: {card.get('title') or card.get('id') or card.get('item')}"
        assert dd.local_stamp(card.get('checked_at')) is not None, f"{name} has no checked_at time"
        assert not dd.typed_numbers(card, fixed), (name, dd.typed_numbers(card, fixed))
        for binding in card.get('measured') or []:
            assert binding.get('scheduled') or binding.get('file') or binding.get('training'), name
            if binding.get('scheduled'):
                assert binding.get('show') in dd._SCHEDULED_SHOWS, name
    # the checker itself: a typed result fails, a test definition or a place in training and time does not
    assert dd.typed_numbers(dict(evidence='Eye angle: 52% less error than holding still.'))
    assert dd.typed_numbers(dict(status='6 of 6 scheduled rounds'))
    assert dd.typed_numbers(dict(measured=[dict(say='The best turn reached 116.8 degrees ({deg}).')]))
    assert not dd.typed_numbers(dict(evidence='Where the eyes point 150 ms after a command, since step 380,550 '
                                              '(segments 36 to 39), checked Sep 24, 2026 at 9:23 PM PDT; record '
                                              'HANDOFF_2026-09-24.md, commit 9c1daed3a, flag --liveness-rule; {n} rounds.'),
                                 ['150 ms'])
    assert not dd.typed_numbers(dict(id='hand_1', upgrades=[2, 5], step=380200, checked_at='2026-09-30 20:00'))
    # the independent review's holes (2026-09-30): each is a typed measurement
    for text in ('decreased 40%', 'Mark 30 rounds', 'won 3/8 rounds', '12 = good', 'between step 5 and 9 bodies',
                 'from 9:28 to 9:45 PM'):
        assert dd.typed_numbers(dict(evidence=text)), text
    assert dd.typed_numbers(dict(fixes=[dict(name='Command selection: 118 to 2', impact='')]))
    assert dd.typed_numbers(dict(measured=[dict(name='the 64-model-step audit', say='')]))


def test_a_malformed_card_or_a_newer_source_document_is_marked_and_the_tab_survives(tmp_path):
    completed = time.mktime((2026, 9, 30, 19, 15, 0, 0, 0, -1))
    catalog = scheduled_card_fixture(tmp_path, '2026-09-30 20:00', completed)
    catalog['highlights'].append(dict(title='Broken', value='', checked_at='2026-09-30 20:00', meaning='m',
                                      measured=[dict(file='docs/spec.html', say='{absent}', into='meaning'),
                                                dict(scheduled='gaze_prediction', show='tile', into='stage_evidence.x')]))
    write(tmp_path/dd.CATALOG, catalog)
    tiles = dd.goal_catalog(tmp_path, now=completed + 60)['highlights']
    assert tiles[0]['value'] == '56% less error' and tiles[0]['freshness']['out_of_date'] is False
    assert tiles[1]['freshness']['out_of_date'] is True and 'could not be read' in tiles[1]['freshness']['stale_text']
    # a card whose explanation comes from a document is out of date once that document is newer than its check
    doc = tmp_path/'docs/record.md'
    doc.write_text('record', encoding='utf-8')
    later = dd.local_stamp('2026-09-30 21:00')
    os.utime(doc, (later, later))
    catalog['highlights'] = [dict(catalog['highlights'][0], source='docs/record.md')]
    write(tmp_path/dd.CATALOG, catalog)
    tile = dd.goal_catalog(tmp_path, now=completed + 60)['highlights'][0]
    assert tile['freshness']['out_of_date'] is True and tile['freshness']['since'] == later
    assert 'record.md (written ' in tile['freshness']['text']


def scheduled_card_fixture(root, checked_at, completed_at=2_000_000_000.0):
    folder = root/'Saved/mimo_bridge/ability_rechecks/gaze_prediction_437125'
    write(folder/'result.json', GAZE)
    tasks = {'gaze_prediction': dict(title='Predict eye movement and the resulting view', due_step=449125, status='complete',
                                     history=[dict(status='complete', attempt_step=437125, started_at=completed_at - 60,
                                                   completed_at=completed_at, result=str(folder/'result.json'))])}
    scheduler(root, tasks)
    catalog = dict(schema=1, title='t', source='docs/spec.html', stages=catalogue()['stages'], capabilities=[], events=[],
                   investigations=[], attention=[], hand_actions=[], upgrade_progress=[],
                   highlights=[dict(title='Eye-angle prediction', value='', tone='evidence', checked_at=checked_at,
                                    meaning='Where the eyes point 150 ms after an eye command.',
                                    measured=[dict(scheduled='gaze_prediction', show='tile', into='value'),
                                              dict(scheduled='gaze_prediction', show='trend', into='meaning')])])
    write(root/'docs/spec.html', {})
    write(root/dd.CATALOG, catalog)
    return catalog


def test_a_card_older_than_its_newest_result_is_marked_out_of_date_with_since_when(tmp_path):
    completed = time.mktime((2026, 9, 30, 19, 15, 0, 0, 0, -1))
    scheduled_card_fixture(tmp_path, '2026-09-30 18:00', completed)
    tile = dd.goal_catalog(tmp_path, now=completed + 60)['highlights'][0]
    assert tile['value'] == '56% less error', 'the number comes from the newest result, never from the hand text'
    assert tile['meaning'].startswith('Where the eyes point 150 ms after an eye command. Eye-angle prediction beat')
    fresh = tile['freshness']
    assert fresh['out_of_date'] is True and fresh['since'] == completed and fresh['since_text'] == dd.clock(completed)
    assert fresh['stale_text'].startswith(f"Out of date since {dd.clock(completed)}: a newer result arrived after this "
                                          f"text was checked ({dd.clock(dd.local_stamp('2026-09-30 18:00'))})")
    assert "'Predict eye movement and the resulting view' on saved step 437,125" in fresh['text']
    assert tile['source_url'].endswith('/gaze_prediction_437125/result.json')
    # the same card checked after that result is current
    scheduled_card_fixture(tmp_path, '2026-09-30 20:00', completed)
    fresh = dd.goal_catalog(tmp_path, now=completed + 60)['highlights'][0]['freshness']
    assert fresh['out_of_date'] is False and fresh['text'].startswith(f"Text checked {dd.clock(dd.local_stamp('2026-09-30 20:00'))}")
    # a card without a checked time is out of date
    catalog = scheduled_card_fixture(tmp_path, None, completed)
    del catalog['highlights'][0]['checked_at']
    write(tmp_path/dd.CATALOG, catalog)
    assert dd.goal_catalog(tmp_path, now=completed + 60)['highlights'][0]['freshness']['out_of_date'] is True
    for text in (fresh['text'], tile['meaning'], tile['freshness']['stale_text']):
        plain(text, 'card date')


def test_a_missing_source_shows_that_it_is_missing_and_a_file_binding_fills_its_numbers(tmp_path):
    report = tmp_path/'Saved/run/report.json'
    write(report, dict(completed_candidates=96, best=dict(maximum_roll_from_start_deg=116.75368560130835)))
    stamp = time.mktime((2026, 9, 23, 22, 9, 0, 0, 0, -1))
    os.utime(report, (stamp, stamp))
    card = dict(checked_at='2026-09-30 20:00', evidence='No supported-roll test of the learner has run.',
                measured=[dict(file='Saved/run/report.json', name='the rolling search', into='evidence',
                               say='The best of {n} sequences turned the body {deg}.',
                               values=dict(n=dict(path='completed_candidates', count=True),
                                           deg=dict(path='best.maximum_roll_from_start_deg', unit='degrees'))),
                          dict(file='Saved/run/absent.json', say='{x}', values=dict(x=dict(path='x')), into='evidence'),
                          dict(scheduled='planner_candidates', show='headline', into='evidence')])
    dd.bind_card(tmp_path, card, {}, ({}, None))
    assert card['evidence'].startswith('No supported-roll test of the learner has run. The best of 96 sequences turned '
                                       'the body 117 degrees. (Missing: Saved/run/absent.json is missing or unreadable; '
                                       'no number is shown.)')
    assert "(Missing: the scheduled comparison 'planner_candidates' has no completed result" in card['evidence']
    fresh = card['freshness']
    assert fresh['out_of_date'] is False and len(fresh['missing']) == 2 and 'Missing:' in fresh['text']
    assert f"the rolling search ({dd.clock(stamp)})" in fresh['text']
    # a newer result file makes the card out of date
    later = dd.local_stamp('2026-09-30 21:00')
    os.utime(report, (later, later))
    dd._RESULTS.clear()
    card2 = dict(checked_at='2026-09-30 20:00', measured=[card['measured'][0]])
    dd.bind_card(tmp_path, card2, {}, ({}, None))
    assert card2['freshness']['out_of_date'] is True and card2['freshness']['since'] == later


def test_a_live_reading_dates_nothing_but_a_broken_assumption_marks_the_card_out_of_date(tmp_path):
    record = dict(type='mimo', step=444600, timestamp=2_000_000_000.0,
                  prediction_audit_per_body=dict(senses=dict(vision=dict(all_bodies=dict(ratio=0.80, transitions=800)))))
    binding = dict(training=True, into='summary',
                   say="Vision's next-picture error was {ratio} of the unchanged picture's over {n} model steps.",
                   values=dict(ratio=dict(path='prediction_audit_per_body.senses.vision.all_bodies.ratio', percent=True),
                               n=dict(path='prediction_audit_per_body.senses.vision.all_bodies.transitions', count=True)),
                   assumes=[dict(path='prediction_audit_per_body.senses.vision.all_bodies.ratio', below=1,
                                 means="vision beats 'the picture stays the same' in live training")])
    card = dict(checked_at='2026-09-30 20:00', summary='In live training:', measured=[binding])
    dd.bind_card(tmp_path, card, {}, (record, None))
    assert card['summary'] == "In live training: Vision's next-picture error was 80% of the unchanged picture's over 800 model steps."
    assert card['freshness']['out_of_date'] is False, 'a reading newer than the text is live, not out of date'
    record['prediction_audit_per_body']['senses']['vision']['all_bodies']['ratio'] = 1.3
    card = dict(checked_at='2026-09-30 20:00', summary='In live training:', measured=[binding])
    dd.bind_card(tmp_path, card, {}, (record, None))
    assert card['freshness']['out_of_date'] is True and card['freshness']['since'] == 2_000_000_000.0
    assert "the text assumes that vision beats 'the picture stays the same' in live training" in card['freshness']['stale_text']


def test_relaunches_are_read_from_the_launch_receipt(tmp_path):
    write(tmp_path/'Saved/receipt.json', dict(events=[
        dict(at='Sep 29, 2026, 06:00:33 AM PDT', event='relaunch: old supervisor ended', changes={'--liveness-rule': 'live_line'},
             checkpoint='C:/x/checkpoints/ckpt_step_416100', reason='The watcher stopped at 416,100.'),
        dict(at='Sep 29, 2026, 06:00:33 AM PDT', event='relaunch: trainer launched', run='C:/x/H84_seg41_liverule'),
        dict(at='Sep 29, 2026, 06:05:33 AM PDT', event='relaunch: first stats record', reinitialized=[]),
        dict(at='Sep 29, 2026, 11:19:37 AM PDT', event='relaunch: checkpoint refused', why='Missing complete checkpoint component'),
        dict(at='Sep 29, 2026, 11:19:37 AM PDT', event='relaunch: old supervisor ended', changes={},
             checkpoint='C:/x/checkpoints/ckpt_step_421461', reason='MemoryError at 10:49 AM while saving.'),
        dict(at='Sep 29, 2026, 11:22:38 AM PDT', event='relaunch: first stats record', reinitialized=[])]))
    event = dd.restart_events(tmp_path, 'Saved/receipt.json', 'Segment 41 relaunches after stops')
    assert event['title'] == 'Segment 41 relaunches after stops: 2 so far' and event['step'] == 421461
    at = dd.clock(dd._receipt_time('Sep 29, 2026, 11:19:37 AM PDT'))
    assert event['impact'] == (f"After each stop training was relaunched from the newest complete saved checkpoint. The "
                               f"newest relaunch, {at}, resumed from step 421,461. The reason recorded for each relaunch "
                               "is listed in its details.")
    assert event['retained'].startswith('2 of 2 relaunches reported in their first training record that no learned part '
                                        'was reinitialized. The relaunch tool refused an incomplete checkpoint 1 time')
    assert event['fixes'][0]['name'].endswith('from saved step 416,100, with --liveness-rule changed')
    assert event['fixes'][1]['impact'] == 'Recorded reason: MemoryError at 10:49 AM while saving.'
    # a malformed receipt row (changes written as a list) costs nothing
    write(tmp_path/'Saved/odd.json', dict(events=[dict(event='relaunch: old supervisor ended', changes=['--x'], at=None)]))
    assert dd.restart_events(tmp_path, 'Saved/odd.json')['fixes'][0]['name'].endswith('no setting changed')
    for text in (event['impact'], event['retained'], event['next'], event['fixes'][0]['name']):
        plain(text, 'relaunch event')
    assert dd.restart_events(tmp_path, 'Saved/absent.json') is None


def data_root():
    """The tree whose results the live page reads: this checkout, or UEMA_DASHBOARD_DATA_ROOT when the
    code is checked from a worktree without the Saved folder."""
    return Path(os.environ.get('UEMA_DASHBOARD_DATA_ROOT') or dd.REPO)


def test_no_card_of_the_live_catalogue_is_older_than_its_newest_result():
    """FAILS when a scheduled round, a result file or a broken live assumption is newer than a card's
    checked text: re-check the card's text against the named result and update its checked_at."""
    import pytest
    root = data_root()
    if not (root/dd.ABILITY_STATUS).is_file():
        pytest.skip(f'no scheduler status under {root}; set UEMA_DASHBOARD_DATA_ROOT to the live tree')
    data = dd.goal_catalog(root, catalog_path=dd.REPO/dd.CATALOG)
    stale, missing = [], []
    for group, card in cards(data):
        fresh = card.get('freshness') or {}
        name = f"{group}: {card.get('title') or card.get('id') or card.get('item')}"
        if fresh.get('out_of_date'):
            stale.append(f"{name}: {fresh.get('stale_text')}")
        missing += [f"{name}: {m}" for m in fresh.get('missing') or []]
    assert not stale, 'cards older than their newest result:\n' + '\n'.join(stale)
    assert not missing, 'cards whose result is missing:\n' + '\n'.join(missing)
    for group, card in cards(data):
        for key in ('value', 'meaning', 'evidence', 'next', 'finding', 'status', 'summary', 'detail', 'impact',
                    'retained', 'physical_detail'):
            if isinstance(card.get(key), str):
                plain(card[key], f"{group} {card.get('title') or card.get('item')} {key}")
        for text in card.get('stage_evidence') or []:
            plain(text, f"{group} {card.get('item')} stage evidence")
        plain((card.get('freshness') or {}).get('text') or '', f"{group} freshness")
