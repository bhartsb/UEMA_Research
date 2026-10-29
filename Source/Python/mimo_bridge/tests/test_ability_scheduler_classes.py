# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Test classes, triggers and the remembered-hand-goals hold in the ability scheduler (2026-09-25).

The operator's decisions of 9:09 AM PDT (docs/gain_of_function_v3/WORKLOG.md): split the scheduled
tests into cheap ones that run often and heavy ones that run less often or on a trigger, and hold
training for review when a round of the remembered hand-goal comparison fails, because H84 runs hand
goals from memory. Design and measurements: docs/gain_of_function_v4/agents/scheduler_classes_2026-09-25.md.
Real bounded children run where the scheduler's process handling is under test; no learner is loaded.
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from mimo_bridge.tools import ability_comparison_window as window
from mimo_bridge.tools import ability_rechecks as worker

REPO = Path(__file__).resolve().parents[4]
MEASURED_HAND_MEMORY = REPO/'docs/gain_of_function_v3/measurements/hand_memory_route_2026-09-24.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def context(root, **values):
    base = dict(root=root, head=None, flags=None, gauges={}, live_step=None, ride_along_steps=3000)
    base.update(values)
    return base


TRIGGERED = dict(id='body', title='Body', script='probe.py', **{'class': 'triggered'}, first_step=100, every_steps=12000,
                 memory_gib=12, triggers=dict(lines=['body', 'hand']))


# ------------------------------------------------------------------------------------------ the schedule
def test_a_frequent_task_keeps_its_cadence_and_a_triggered_one_waits_for_its_longest_interval(tmp_path):
    frequent = dict(TRIGGERED, **{'class': 'frequent'}, every_steps=3000, triggers=None)
    assert worker.schedule(frequent, dict(completed_step=1000), context(tmp_path), {}) == (4000, 4000, [])
    assert worker.schedule(TRIGGERED, dict(completed_step=1000), context(tmp_path), {}) == (13000, 13000, [])
    # never completed: its first step, like any task
    assert worker.schedule(TRIGGERED, {}, context(tmp_path), {}) == (100, 100, [])


def test_a_fired_trigger_joins_the_next_window_and_pauses_training_only_after_the_ride_along_wait(tmp_path):
    record = dict(completed_step=1000)
    tasks = dict(hand=dict(title='Hand', line_changed=dict(step=1500, before=True, after=False)), body=record)
    run, opened, reasons = worker.schedule(TRIGGERED, record, context(tmp_path, live_step=1600), tasks)
    assert run == 1001, 'any checkpoint after its last round, inside a window'
    assert opened == 1600+3000, 'it pauses training itself only once the ride-along wait passed'
    assert reasons == ['Hand went from meeting its declared line to missing it at step 1,500']
    assert record['triggered']['since_step'] == 1600
    # the wait counts from when the trigger was first seen, not from each later cycle
    assert worker.schedule(TRIGGERED, record, context(tmp_path, live_step=2900), tasks)[1] == 4600
    # never later than its longest interval
    record['triggered']['since_step'] = 12500
    assert worker.schedule(TRIGGERED, record, context(tmp_path, live_step=12600), tasks)[1] == 13000
    # a held task (first step out of reach) stays held whatever fires
    held = dict(TRIGGERED, first_step=999999999)
    assert worker.schedule(held, dict(completed_step=1000), context(tmp_path, live_step=1600), tasks)[:2] == (999999999, 999999999)


def test_a_line_crossing_fires_for_the_task_itself_at_its_own_round_and_for_others_only_after_their_round(tmp_path):
    ctx = context(tmp_path)
    own = dict(completed_step=2000, line_changed=dict(step=2000, before=False, after=True), title='Body')
    assert worker.fired_triggers(TRIGGERED, own, ctx, dict(body=own)), 'a confirmation round after its own crossing'
    own['completed_step'] = 5000         # the confirmation round ran
    assert not worker.fired_triggers(TRIGGERED, own, ctx, dict(body=own))
    other = dict(title='Hand', line_changed=dict(step=5000, before=True, after=False))
    assert not worker.fired_triggers(TRIGGERED, own, ctx, dict(body=own, hand=other)), 'measured at the same checkpoint'
    other['line_changed']['step'] = 5100
    assert worker.fired_triggers(TRIGGERED, own, ctx, dict(body=own, hand=other))


def git(root, *args, date=None):
    env = dict(os.environ, GIT_AUTHOR_DATE=date or '', GIT_COMMITTER_DATE=date or '') if date else None
    return subprocess.run(['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'core.autocrlf=false',
                           *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def test_a_committed_change_to_a_declared_file_fires_and_a_change_elsewhere_does_not(tmp_path):
    git(tmp_path, 'init', '-q')
    (tmp_path/'model.py').write_text('a = 1\n', encoding='utf-8')
    (tmp_path/'other.py').write_text('b = 1\n', encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-q', '-m', 'first', date='2026-01-01T00:00:00')
    first = worker.code_head(tmp_path)
    (tmp_path/'other.py').write_text('b = 2\n', encoding='utf-8')
    git(tmp_path, 'commit', '-q', '-am', 'elsewhere', date='2026-01-02T00:00:00')
    elsewhere = worker.code_head(tmp_path)
    assert worker.changed_code(tmp_path, first, elsewhere, ['model.py']) == []
    (tmp_path/'model.py').write_text('a = 2\n', encoding='utf-8')
    git(tmp_path, 'commit', '-q', '-am', 'declared', date='2026-01-03T00:00:00')
    head = worker.code_head(tmp_path)
    assert worker.changed_code(tmp_path, first, head, ['model.py']) == ['model.py']
    task = dict(TRIGGERED, triggers=dict(code=['model.py']))
    record = dict(completed_step=1000, basis=dict(code_commit=first))
    reasons = worker.fired_triggers(task, record, context(tmp_path, head=head), {})
    assert reasons == ['committed code changed since its last round: model.py']
    # the commit that was HEAD when a round started, for rounds that ran before triggers existed
    noon = time.mktime(time.strptime('2026-01-02T12:00:00', '%Y-%m-%dT%H:%M:%S'))
    assert worker.commit_before(tmp_path, noon) == elsewhere
    # git cannot tell (an unknown commit, or no repository): nothing fires
    assert worker.changed_code(tmp_path, '0'*40, head, ['model.py']) is None
    assert worker.code_head(tmp_path/'not_a_repository') is None


def test_only_a_change_of_a_learning_flag_fires_the_flags_trigger(tmp_path):
    base = dict(hand_goals='memory', decoder_kp=1.0, resume='a', out='a', checkpoint_storage='a', caregiver_record_dir='a')
    write(tmp_path/'a/args.json', base)
    write(tmp_path/'b/args.json', dict(base, resume='b', out='b', checkpoint_storage='b', caregiver_record_dir='b'))
    write(tmp_path/'c/args.json', dict(base, decoder_kp=2.0))
    a, b, c = (worker.production_flags(tmp_path/x) for x in 'abc')
    assert a == b and a != c and worker.production_flags(tmp_path/'missing') is None
    task = dict(TRIGGERED, triggers=dict(flags=True))
    record = dict(completed_step=1000, basis=dict(flags=a))
    assert not worker.fired_triggers(task, record, context(tmp_path, flags=b), {})
    assert worker.fired_triggers(task, record, context(tmp_path, flags=c), {})


def stats(run, records):
    run.mkdir(parents=True, exist_ok=True)
    with (run/'replay_training_stats.jsonl').open('w', encoding='utf-8') as out:
        out.write(json.dumps(dict(type='launch', step=0)) + '\n')
        for i, (share, error) in enumerate(records):
            out.write(json.dumps(dict(type='mimo', step=100*(i+1), efference=dict(gaze_deg_err=[1., 2., 6.*(1-share)],
                      gaze_deg_do_nothing=[2., 4., 6.]), hand_control=dict(joint_error_deg=error))) + '\n')


def test_a_live_gauge_averages_the_newest_ten_records_and_fires_past_its_tolerance(tmp_path):
    run = tmp_path/'run'
    stats(run, [(0.5, 16.0)]*9)
    assert worker.gauge_values(run) == {}, 'fewer than ten records (a run younger than about 1,000 updates)'
    stats(run, [(0.2, 30.0)]*5 + [(0.5, 16.0)]*10)
    gauges = worker.gauge_values(run)
    assert gauges['eye_forecast_share_150ms']['value'] == pytest.approx(0.5)
    assert gauges['hand_joint_error_deg'] == dict(value=16.0, records=10, step=1500)
    task = dict(TRIGGERED, triggers=dict(gauges=[dict(name='eye_forecast_share_150ms', tolerance=0.08)]))
    moved = dict(completed_step=1000, basis=dict(gauges=dict(eye_forecast_share_150ms=0.40)))
    within = dict(completed_step=1000, basis=dict(gauges=dict(eye_forecast_share_150ms=0.45)))
    assert worker.fired_triggers(task, moved, context(tmp_path, gauges=gauges), {})
    assert not worker.fired_triggers(task, within, context(tmp_path, gauges=gauges), {})


@pytest.mark.parametrize('change,reason', [
    (dict(**{'class': 'weekly'}), 'class'),
    (dict(**{'class': 'frequent'}), 'Only a triggered'),
    (dict(triggers=dict(code=['probe.py'], sometimes=True)), 'Unknown'),
    (dict(triggers=dict(code=['no_such_file.py'])), 'missing path'),
    (dict(triggers=dict(code=['dir\\probe.py'])), 'written with /'),
    (dict(triggers=dict(gauges=[dict(name='mood', tolerance=1)])), 'known name'),
    (dict(triggers=dict(gauges=[dict(name='hand_joint_error_deg', tolerance=0)])), 'positive tolerance'),
    (dict(triggers=dict(lines=['nobody'])), 'declared ability tests'),
    (dict(first_step_under_hold=5), 'hold_training_on_failed_comparison'),
    (dict(every_steps=0), 'every_steps'),
])
def test_an_entry_that_cannot_be_scheduled_is_refused_with_its_reason(tmp_path, change, reason):
    (tmp_path/'probe.py').write_text('', encoding='utf-8')
    task = dict(TRIGGERED, **change)
    with pytest.raises(ValueError, match=reason):
        worker.validate_schedule(tmp_path, task, ['body', 'hand'], 3000)


def test_the_first_step_of_a_task_that_holds_training_is_out_of_reach_of_a_scheduler_without_the_hold():
    task = dict(first_step=999999999, first_step_under_hold=399050, hold_training_on_failed_comparison=True)
    assert worker.first_step(task) == 399050
    # a scheduler before 2026-09-25 reads first_step alone: it never runs the task
    assert task['first_step'] >= 10**8
    assert worker.first_step(dict(task, hold_training_on_failed_comparison=False)) == 999999999


def test_the_polling_is_fast_while_training_stands_still_for_a_window():
    for state in ('preparing', 'saving', 'ready'):
        assert worker.poll_seconds(dict(comparison_window=dict(state=state))) == worker.WINDOW_POLL_S == 5
    for state in ('released', 'held for review', 'repair required'):
        assert worker.poll_seconds(dict(comparison_window=dict(state=state))) == worker.POLL_S == 30
    assert worker.poll_seconds(None) == 30


# ------------------------------------------------------------------------------------------ the declared lines
LINE_TASKS = {'hand_following': 'qualify_hand_execution_2026-09-24.py', 'body_and_synergy': 'compare_hand_body_goals_2026-09-24.py',
              'gaze_prediction': 'probe_saved_gaze.py', 'chosen_self_touch': 'probe_saved_self_touch_choices.py',
              'planner_candidates': 'probe_planner_candidates.py'}


def test_the_declared_line_is_the_dashboards_reading_of_every_saved_result():
    """Read only: every completed round in this checkout's scheduler folder, judged by the scheduler's
    declared line and by the page's own reading (development_dashboard._won); they must agree."""
    from mimo_bridge.tools import development_dashboard as D
    folder = REPO/'Saved/mimo_bridge/ability_rechecks'
    compared = 0
    for receipt in sorted(folder.glob('*.receipt.json')):
        r = json.loads(receipt.read_text(encoding='utf-8'))
        task = next((t for t in LINE_TASKS if receipt.name.startswith(t+'_')), None)
        if r.get('status') != 'complete' or task is None or not Path(r['result']).is_file():
            continue
        payload = json.loads(Path(r['result']).read_text(encoding='utf-8'))
        try:
            words = D._RESULT_WORDS[task](payload)
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
        assert worker.declared_line(LINE_TASKS[task], payload) == D._won(task, words), receipt.name
        compared += 1
    if not compared:
        pytest.skip('No completed scheduled round in this checkout')


# ------------------------------------------------------------------------------------------ the scheduler loop
@pytest.fixture
def loop(tmp_path, monkeypatch):
    """A declared lineage with one running trainer (fake), a checkpoint that exists once saved, and a
    bounded child script; memory is short beside the trainer and ample once it stops."""
    from mimo_bridge.tools import commit_guard
    saved = tmp_path/'Saved/mimo_bridge'
    run = saved/'experiments/declared_seg'
    run.mkdir(parents=True)
    folder = saved/'ability_rechecks'
    folder.mkdir()
    write(saved/'training_continuation.json', dict(active_run=str(run), timestamp=time.time()))
    write(saved/'supervisor_control.json', dict(cmd='run'))
    (run/'replay_training_stats.jsonl').write_text(json.dumps(dict(type='mimo', step=105)) + '\n', encoding='utf-8')
    checkpoint = run/'checkpoints/ckpt_step_108'

    class Trainer:
        pid = 71

        def create_time(self):
            return 123.
    running = [Trainer()]
    monkeypatch.setattr(window, 'processes', lambda *args: list(running))
    monkeypatch.setattr(worker, 'trainer_running', lambda: bool(running))
    monkeypatch.setattr(worker, 'complete_candidates', lambda _: [checkpoint] if checkpoint.is_dir() else [])
    monkeypatch.setattr(window, 'complete_candidates', lambda _: [checkpoint] if checkpoint.is_dir() else [])
    monkeypatch.setattr(commit_guard, 'commit_reading', lambda: (99, 100, 1) if running else (1, 100, 99))
    # every task here has its test registry row; the registry check itself is tested in test_test_registry.py
    monkeypatch.setattr(worker, 'registry_refusals', lambda root, ids: {})

    def finish_save():
        running.clear()
        checkpoint.mkdir(parents=True)
        physical = checkpoint/'episode.pt'
        physical.write_bytes(b'preserved eight worlds')
        control = window.read(run/'control.json')
        if control is not None:                  # the trainer consumed the window's save-and-stop order
            window.atomic_json(run/'control_done_window.json', control)
            (run/'control.json').unlink()
        window.atomic_json(run/'stage_done.json', dict(ok=True, steps=108, finished_at=time.time(), final_checkpoint=str(checkpoint)))
        window.atomic_json(checkpoint/'episode_state.json', dict(step=108, times=[1.]*8, storage_path=str(physical),
                           bytes=physical.stat().st_size, sha256=window.sha256_file(physical)))
    info = dict(root=tmp_path, saved=saved, run=run, folder=folder, checkpoint=checkpoint, running=running,
                finish_save=finish_save, manifest=tmp_path/'manifest.json')
    yield info
    state = worker.read(folder/'status.json')
    for record in (state.get('tasks') or {}).values():
        children, _ = worker.matching_children(record)
        for child in children:
            try:
                process = psutil.Process(child['pid'])
                family = process.children(recursive=True) + [process]
                for member in family:
                    member.kill()
                psutil.wait_procs(family, timeout=2)
            except psutil.NoSuchProcess:
                pass


def tick(info):
    return worker.cycle(info['root'], info['manifest'], info['folder'])


def until(info, predicate, seconds=30):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        state = tick(info)
        if predicate(state):
            return state
        time.sleep(.05)
    raise AssertionError('The bounded child did not reach its declared state')


HAND_MEMORY_SCRIPT = '''import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--out');p.add_argument('--calibration');a=p.parse_args()
out=Path(a.out);out.mkdir()
here=Path(__file__)
payload=json.loads((here.parent/'payload.json').read_text())
if payload.get('crash'):sys.exit('declared test crash')
payload.update(checkpoint=a.checkpoint,script_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
               calibration_file_sha256=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest())
(out/'result.json').write_text(json.dumps(payload))
'''


def hand_memory(info, passed=True, crash=False):
    """The remembered hand-goal comparison as the production manifest declares it, on a script that
    writes the comparison's own recorded result (passed at 3:37 AM PDT on 2026-09-24), or a failed one."""
    docs = info['root']/'docs'
    docs.mkdir()
    (docs/'compare_hand_memory_route_2026-09-24.py').write_text(HAND_MEMORY_SCRIPT, encoding='utf-8')
    (docs/'calibration.json').write_text('{"sha256": "x"}', encoding='utf-8')
    payload = json.loads(MEASURED_HAND_MEMORY.read_text(encoding='utf-8'))
    if not passed:
        payload['summary']['hand_shape_error']['bodies_improved'] = 3
        payload['passed'] = False
    payload['crash'] = crash
    write(docs/'payload.json', payload)
    production = {t['id']: t for t in json.loads((REPO/'docs/gain_of_function_v3/ability_rechecks.json').read_text(
        encoding='utf-8'))['tasks']}['hand_memory_route']
    task = dict(production, script='docs/compare_hand_memory_route_2026-09-24.py', path_arguments={'--calibration': 'docs/calibration.json'},
                depends=[], first_step_under_hold=100, every_steps=30)
    write(info['manifest'], dict(lineage='declared', ride_along_steps=3000, tasks=[task]))
    return task


def open_window_and_run(info):
    first = tick(info)
    control = window.read(info['saved']/'supervisor_control.json')
    assert first['comparison_window']['state'] == 'saving' and control['cmd'] == 'pause'
    # ONE SAVE PER PAUSE: the window's own save-and-stop is the only trainer control; no snapshot first
    assert window.read(info['run']/'control.json')['cmd'] == 'snapshot_and_stop'
    assert 'snapshot_requested_at' not in first
    info['finish_save']()
    return until(info, lambda s: s['tasks']['hand_memory_route'].get('status') in ('complete', 'held for review', 'failed', 'repair needed')
                 and s['tasks']['hand_memory_route'].get('completed_at'))


def test_a_failed_hand_memory_round_leaves_the_window_held_and_the_supervisor_paused(loop):
    hand_memory(loop, passed=False)
    done = open_window_and_run(loop)
    record = done['tasks']['hand_memory_route']
    assert record['status'] == 'held for review' and record['outcome'] is False and record['completed_step'] == 108
    assert 'did not pass its declared comparison' in record['reason'] and 'H84 runs hand goals from memory' in record['reason']
    receipt = json.loads(Path(record['output']).with_suffix('.receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['outcome'] is False, 'a measured failure is a completed round'
    owned = done['comparison_window']['owned_control']
    held = tick(loop)
    assert held['comparison_window']['state'] == 'held for review'
    assert held['state'] == 'training held for review' and 'did not pass' in held['pause_reason']
    # how to end it: edit only "cmd" and keep the other keys, as UTF-8 (review of 2026-09-25)
    assert 'edit only the "cmd" field' in held['pause_reason'] and 'UTF-8' in held['pause_reason']
    assert window.read(loop['saved']/'supervisor_control.json') == owned, 'the supervisor stays paused under the window'
    token = held['comparison_window']['token']
    assert window.read(loop['folder']/f'window_{token}.json')['state'] == 'held for review'
    # more cycles change nothing and start nothing
    launched = len(list(loop['folder'].glob('*.process.json')))
    for _ in range(3):
        again = tick(loop)
    assert again['comparison_window']['state'] == 'held for review' and again['tasks']['hand_memory_route']['status'] == 'held for review'
    assert window.read(loop['saved']/'supervisor_control.json') == owned
    assert len(list(loop['folder'].glob('*.process.json'))) == launched
    # the operator's review ends when the control changes; the supervisor then resumes training
    write(loop['saved']/'supervisor_control.json', dict(cmd='run'))
    released = tick(loop)
    assert released['comparison_window']['state'] == 'review released'
    assert released['tasks']['hand_memory_route']['review_hold']['released_at']
    after = tick(loop)
    assert after['tasks']['hand_memory_route']['status'] == 'complete' and after.get('state') != 'training held for review'
    assert window.read(loop['saved']/'supervisor_control.json') == dict(cmd='run'), 'the scheduler never rewrote the control'


def test_a_passed_hand_memory_round_releases_the_window_as_before(loop):
    hand_memory(loop, passed=True)
    done = open_window_and_run(loop)
    assert done['tasks']['hand_memory_route']['status'] == 'complete' and done['tasks']['hand_memory_route']['outcome'] is True
    released = tick(loop)
    assert released['comparison_window']['state'] == 'released'
    assert window.read(loop['saved']/'supervisor_control.json') == dict(cmd='run')
    assert 'review_hold' not in released['tasks']['hand_memory_route']


def test_a_hand_memory_attempt_without_a_valid_result_follows_the_retry_rules_and_never_holds(loop):
    """The coordinator's decision of 2026-09-25 (the review's gap a): two attempts without a result are no
    evidence that the hand stage fails, so training is not held; a warning stays in the scheduler's status
    and on the dashboard's scheduled-results panel until a round completes."""
    hand_memory(loop, crash=True)
    open_window_and_run(loop)
    second = until(loop, lambda s: s['tasks']['hand_memory_route'].get('attempts') == 2
                   and s['tasks']['hand_memory_route'].get('status') == 'failed')
    assert 'review_hold' not in second['tasks']['hand_memory_route']
    released = until(loop, lambda s: (s.get('comparison_window') or {}).get('state') == 'released')
    assert released['tasks']['hand_memory_route']['status'] == 'repair needed'
    assert window.read(loop['saved']/'supervisor_control.json') == dict(cmd='run')
    warned = tick(loop)
    warning = warned['tasks']['hand_memory_route']['warning']
    assert warned['warnings'] == [warning] and 'is unmeasured' in warning and 'ended without a result' in warning
    assert 'unchecked until a round completes' in warning


def test_the_unmeasured_warning_lasts_until_a_round_completes(tmp_path, monkeypatch):
    record = dict(status='running', attempt_step=200, attempts=2, test_version='v', failed_version='v', failures_in_a_row=1,
                  command=['python', '-u', 'x.py'], output=str(tmp_path/'a1'), title='Remembered hand goals',
                  hold_on_failed_comparison='The mechanism steers production.')
    state = dict(tasks=dict(hand=record))
    monkeypatch.setattr(worker, 'matching_children', lambda r: ([], None))
    monkeypatch.setattr(worker, 'finished_result', lambda r: dict(status='failed', reason='no result'))
    worker.reconcile_task(state, 'hand')
    assert record['failures_in_a_row'] == 2 and record['unmeasured']['step'] == 200 and 'review_hold' not in record
    assert 'is unmeasured' in worker.task_warning(record, in_window=False)
    record.update(status='running', attempt_step=300, output=str(tmp_path/'a2'))
    monkeypatch.setattr(worker, 'finished_result', lambda r: dict(status='complete', result='r', result_sha256='s',
                                                                   outcome=True, line_met=True))
    worker.reconcile_task(state, 'hand')
    assert 'unmeasured' not in record and worker.task_warning(record, in_window=False) is None


def test_a_task_that_holds_training_never_runs_beside_the_trainer(loop, monkeypatch):
    from mimo_bridge.tools import commit_guard
    hand_memory(loop, passed=False)
    loop['finish_save']()                         # a complete checkpoint exists beside a running trainer
    loop['running'].append(object())
    monkeypatch.setattr(commit_guard, 'commit_reading', lambda: (1, 100, 99))   # memory would allow it
    requested = []
    monkeypatch.setattr(window, 'request', lambda *args: requested.append(args[-1]) or False)
    state = tick(loop)
    assert not list(loop['folder'].glob('*.process.json')), 'launched beside training'
    assert state['tasks']['hand_memory_route']['status'] == 'waiting for a saved window' and requested == [108]


# ------------------------------------------------------------------------------------------ fixed defects
TINY = '''import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--out');a=p.parse_args()
out=Path(a.out);out.mkdir()
for _ in range(2000):
    if (out/'release').exists():break
    time.sleep(.01)
(out/'result.json').write_text(json.dumps({'checkpoint':a.checkpoint,'passed':False,
 'learning':False,'elapsed_s':1,'goals':['curl','extension','index_selective'],
 'modes':['fixed_world','moving_wrist'],'comparisons':24,'exact_replay':True,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'summary':{g:{m:(2. if m=='fixed_world' else 1.) for m in ['fixed_world','moving_wrist']} for g in ['curl','extension','index_selective']}}))
'''


def tiny(info, **task):
    script = info['root']/'qualify_hand_execution_2026-09-24.py'
    script.write_text(TINY, encoding='utf-8')
    entry = dict(id='tiny', title='Bounded hand test', script=script.name, first_step=100, every_steps=30, memory_gib=1)
    entry.update(task)
    write(info['manifest'], dict(lineage='declared', ride_along_steps=3000, tasks=[entry]))
    return script


def test_a_held_test_version_never_asks_the_trainer_for_a_checkpoint(loop):
    """08:40:31 AM on 2026-09-25: the planner test, held after two failures of its version, asked the
    relaunched trainer for a snapshot (about 20 s of training and a 1.37 GB folder)."""
    script = tiny(loop)
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    task = json.loads(loop['manifest'].read_text())['tasks'][0]
    version = worker.test_version(loop['root'], task, digest)
    write(loop['folder']/'status.json', dict(tasks=dict(tiny=dict(status='failed', failed_version=version, failures_in_a_row=2,
                                                                   attempt_step=90, attempts=2, test_version=version))))
    state = tick(loop)
    assert state['tasks']['tiny']['status'] == 'repair needed'
    assert not (loop['run']/'control.json').exists() and 'snapshot_requested_at' not in state
    assert 'comparison_window' not in state


def test_a_hold_taken_back_is_refreshed_before_a_window_reads_it(loop):
    """9:10 to 9:14 AM on 2026-09-25: after a current scheduler took back two stale refusals, the planner
    test still read 'repair needed', and a window's pending list skips tasks marked so."""
    tiny(loop)
    write(loop['folder']/'status.json', dict(tasks=dict(tiny=dict(status='repair needed', failed_version=None, failures_in_a_row=0,
                                                                   attempts=0, reason='Two attempts of this version failed'))))
    state = tick(loop)
    assert state['tasks']['tiny']['status'] != 'repair needed'


def test_a_task_too_big_to_run_beside_training_asks_for_its_window_without_a_snapshot(loop, monkeypatch):
    tiny(loop)
    seen = []
    monkeypatch.setattr(window, 'request', lambda root, folder, state, run, supervisor, minimum: seen.append(minimum) or False)
    state = tick(loop)
    assert seen == [105] and state['tasks']['tiny']['status'] == 'waiting for memory'
    assert not (loop['run']/'control.json').exists(), 'a snapshot before the window is a second save'


def test_a_triggered_task_before_its_own_pause_step_joins_a_window_but_never_opens_one(loop, monkeypatch):
    tiny(loop, **{'class': 'triggered'}, every_steps=12000, triggers=dict(lines=['tiny']))
    write(loop['folder']/'status.json', dict(tasks=dict(tiny=dict(status='complete', completed_step=100, title='Bounded hand test',
                                                                   line_met=True, basis={},
                                                                   line_changed=dict(step=100, before=False, after=True)))))
    requested = []
    monkeypatch.setattr(window, 'request', lambda *args: requested.append(args[-1]) or False)
    state = tick(loop)
    record = state['tasks']['tiny']
    assert record['ride_step'] == 101 and record['due_step'] == 105+3000 and not requested
    assert not (loop['run']/'control.json').exists()
    loop['finish_save']()                         # a saved window at step 108 opened by another task
    ready = dict(state='ready', token='t', run=str(loop['run']), checkpoint=str(loop['checkpoint']), ready_at=time.time(),
                 targets=dict(tiny=101), owned_control=dict(cmd='pause'), previous_control=dict(cmd='run'))
    monkeypatch.setattr(window, 'advance', lambda *args: True)
    status = worker.read(loop['folder']/'status.json')
    status['comparison_window'] = ready
    write(loop['folder']/'status.json', status)
    joined = tick(loop)
    assert joined['tasks']['tiny']['status'] in ('starting', 'running') and joined['tasks']['tiny']['attempt_step'] == 108
    # the window's targets include a task riding along
    assert window.first_eligible(dict(due_step=3105, ride_step=101), 108) == 101


def test_a_missing_script_is_held_with_its_reason_instead_of_stopping_every_cycle(loop):
    write(loop['manifest'], dict(lineage='declared', tasks=[dict(id='gone', title='Gone', script='no_such_script.py',
                                                                  first_step=100, every_steps=30, memory_gib=1)]))
    state = tick(loop)
    assert state['tasks']['gone']['status'] == 'repair needed' and 'cannot be launched' in state['tasks']['gone']['reason']


def test_a_running_round_records_its_peak_memory_and_its_declared_line(loop, monkeypatch):
    from mimo_bridge.tools import commit_guard
    tiny(loop)
    loop['checkpoint'].mkdir(parents=True)
    monkeypatch.setattr(commit_guard, 'commit_reading', lambda: (1, 100, 99))    # it fits beside the trainer
    first = tick(loop)
    record = first['tasks']['tiny']
    output = Path(record['output'])
    # sampled on every look while the child runs (the venv launcher and the interpreter it starts)
    running = until(loop, lambda s: output.is_dir() and s['tasks']['tiny'].get('peak_private_commit_gib'))
    assert running['tasks']['tiny']['peak_private_commit_gib'] > 0
    (output/'release').write_text('finish')
    done = until(loop, lambda s: s['tasks']['tiny']['status'] == 'complete')
    receipt = json.loads(output.with_suffix('.receipt.json').read_text())
    assert receipt['peak_private_commit_gib'] > 0 and receipt['line_met'] is True


def test_a_round_that_crosses_its_declared_line_records_the_crossing(tmp_path, monkeypatch):
    state = dict(tasks=dict(t=dict(status='running', attempt_step=200, line_met=False, command=['python', '-u', 'x.py'],
                                    output=str(tmp_path/'attempt'), title='T', launch_basis=dict(code_commit='abc'))))
    monkeypatch.setattr(worker, 'matching_children', lambda record: ([], None))
    monkeypatch.setattr(worker, 'finished_result',
                        lambda record: dict(status='complete', result='r', result_sha256='s', outcome=True, line_met=True))
    worker.reconcile_task(state, 't')
    record = state['tasks']['t']
    assert record['line_changed'] == dict(step=200, at=record['completed_at'], before=False, after=True)
    assert record['line_met'] is True and record['basis'] == dict(code_commit='abc') and 'launch_basis' not in record
    assert json.loads((tmp_path/'attempt.receipt.json').read_text())['line_met'] is True
    # an unreadable line keeps the one before, so a later crossing is still seen
    record.update(status='running', attempt_step=300)
    monkeypatch.setattr(worker, 'finished_result',
                        lambda record: dict(status='complete', result='r', result_sha256='s', outcome=None, line_met=None))
    worker.reconcile_task(state, 't')
    assert record['line_met'] is True and record['line_changed']['step'] == 200


# ------------------------------------------------------------------------------------------ the independent review
# docs/gain_of_function_v4/agents/scheduler_classes_review_2026-09-25.md: its eight demonstrations, turned
# into regression tests. Each named a defect in commit 266a290ca; each now asserts the repair.
def test_a_trigger_first_seen_before_the_run_has_a_record_joins_pauses_and_waits_from_the_first_known_step(tmp_path):
    """A relaunched segment has no record for its first 100 updates; a trigger first seen then (the flags
    trigger always is) started its wait at the last completed step, so its own pause was due at once."""
    record = dict(completed_step=400000)
    tasks = dict(hand=dict(title='Hand', line_changed=dict(step=401000, before=True, after=False)), body=record)
    run, opened, reasons = worker.schedule(TRIGGERED, record, context(tmp_path, live_step=None), tasks)
    assert reasons and run == 400001, 'it joins any pause at once'
    assert opened == 412000 and record['triggered']['since_step'] is None, 'but opens none of its own yet'
    run, opened, _ = worker.schedule(TRIGGERED, record, context(tmp_path, live_step=406000), tasks)
    assert record['triggered']['since_step'] == 406000 and opened == 409000, 'the wait starts at the first known step'
    fresh = dict(completed_step=400000)
    tasks['body'] = fresh
    assert worker.schedule(TRIGGERED, fresh, context(tmp_path, live_step=406000), tasks)[1] == 409000


def test_a_round_launched_on_a_young_segment_keeps_the_previous_gauge_values_in_its_basis(tmp_path):
    record = dict(completed_step=1000, basis=dict(gauges=dict(hand_joint_error_deg=15.0, eye_forecast_share_150ms=0.45)))
    young = context(tmp_path, gauges={})          # fewer than ten records in the segment: nothing read
    assert worker.launch_gauges(record, young) == dict(hand_joint_error_deg=15.0, eye_forecast_share_150ms=0.45)
    read = context(tmp_path, gauges=dict(hand_joint_error_deg=dict(value=16.0, records=10, step=2000)))
    assert worker.launch_gauges(record, read) == dict(hand_joint_error_deg=16.0, eye_forecast_share_150ms=0.45)
    task = dict(TRIGGERED, triggers=dict(gauges=[dict(name='hand_joint_error_deg', tolerance=1.65)]))
    after = dict(completed_step=2000, basis=dict(gauges=worker.launch_gauges(record, young)))
    moved = dict(hand_joint_error_deg=dict(value=40.0, records=10, step=3000))
    assert worker.fired_triggers(task, after, context(tmp_path, gauges=moved), {}), 'the gauge still watches'


def test_the_filled_in_commit_of_an_older_round_is_the_one_production_ran_not_a_merged_branch_commit(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'checkout', '-q', '-b', 'production')
    (tmp_path/'model.py').write_text('a = 1\n', encoding='utf-8')
    (tmp_path/'notes.md').write_text('x\n', encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-q', '-m', 'c1', date='2026-01-01T01:00:00')
    c1 = worker.code_head(tmp_path)
    git(tmp_path, 'checkout', '-q', '-b', 'side')
    (tmp_path/'model.py').write_text('a = 2\n', encoding='utf-8')
    git(tmp_path, 'commit', '-q', '-am', 'side change', date='2026-01-01T03:00:00')
    git(tmp_path, 'checkout', '-q', 'production')
    (tmp_path/'notes.md').write_text('y\n', encoding='utf-8')
    git(tmp_path, 'commit', '-q', '-am', 'c2', date='2026-01-01T05:00:00')
    git(tmp_path, 'merge', '-q', '--no-ff', 'side', '-m', 'merge', date='2026-01-01T06:00:00')
    head = worker.code_head(tmp_path)
    started = time.mktime(time.strptime('2026-01-01T04:00:00', '%Y-%m-%dT%H:%M:%S'))    # production held c1
    basis = worker.commit_before(tmp_path, started)
    assert basis == c1
    assert worker.changed_code(tmp_path, basis, head, ['model.py']) == ['model.py'], 'the merged change fires'


@pytest.mark.parametrize('encoding', ['utf-16', 'utf-8-sig'])
def test_a_control_written_as_utf16_or_with_a_byte_order_mark_ends_a_hold_without_raising(tmp_path, encoding):
    saved = tmp_path/'Saved/mimo_bridge'
    run = saved/'experiments/r'
    run.mkdir(parents=True)
    write(saved/'training_continuation.json', dict(active_run=str(run)))
    owned = dict(cmd='pause', ability_window_token='t')
    state = dict(tasks=dict(hand=dict(review_hold=dict(step=5, reason='failed'))),
                 comparison_window=dict(state='held for review', token='t', run=str(run), owned_control=owned,
                                        review=dict(hand={})))
    (saved/'supervisor_control.json').write_bytes('{"cmd": "run", "archiver": "off"}'.encode(encoding))
    folder = saved/'ability_rechecks'
    folder.mkdir()
    assert window.advance(tmp_path, folder, state, run) is False
    assert state['comparison_window']['state'] == 'review released' and state['tasks']['hand']['review_hold']['released_at']
    assert window.read_control(saved/'supervisor_control.json') == dict(cmd='run', archiver='off')
    # the supervisor reads UTF-8 only: it keeps training paused and names why, which the scheduler reports
    from mimo_bridge.tools.training_continuation import pause_reason
    assert pause_reason(saved, run) == 'unreadable supervisor control'
    (saved/'supervisor_control.json').write_bytes(b'\x00 not json')
    assert window.read_control(saved/'supervisor_control.json') is window.UNREADABLE


def release_children(info):
    for out in info['folder'].glob('*_attempt*'):
        if out.is_dir():
            (out/'release').write_text('finish')


def run_until(info, predicate, seconds=40):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        state = tick(info)
        release_children(info)
        if predicate(state):
            return state
        time.sleep(.05)
    raise AssertionError('The window did not reach its declared state')


def hand_stats(run, errors, first_step):
    with (run/'replay_training_stats.jsonl').open('w', encoding='utf-8') as out:
        out.write(json.dumps(dict(type='launch', step=0)) + '\n')
        for i, error in enumerate(errors):
            out.write(json.dumps(dict(type='mimo', step=first_step+i, hand_control=dict(joint_error_deg=error))) + '\n')


def test_a_listed_test_whose_trigger_stops_firing_before_it_launches_still_runs_and_the_pause_ends(loop):
    """The review's gauge case: records written between the request and the trainer's stop moved the gauge
    back inside its tolerance, the listed test could no longer run on the checkpoint and the pause stayed
    open with training stopped. A fired trigger now holds until the test's next round."""
    script = loop['root']/'qualify_hand_execution_2026-09-24.py'
    script.write_text(TINY, encoding='utf-8')
    tasks = [dict(id='riding', title='Riding', script=script.name, **{'class': 'triggered'}, first_step=50, every_steps=12000,
                  memory_gib=1, triggers=dict(gauges=[dict(name='hand_joint_error_deg', tolerance=1.3)])),
             dict(id='often', title='Often', script=script.name, first_step=50, every_steps=30, memory_gib=1)]
    write(loop['manifest'], dict(lineage='declared', ride_along_steps=3000, tasks=tasks))
    write(loop['folder']/'status.json', dict(tasks=dict(riding=dict(status='complete', completed_step=50, title='Riding',
          basis=dict(gauges=dict(hand_joint_error_deg=10.0)), line_met=True))))
    hand_stats(loop['run'], [12.0]*10, 96)                          # the newest ten average 12.0: moved 2.0 > 1.3
    first = tick(loop)
    assert first['comparison_window']['state'] == 'saving' and set(first['comparison_window']['targets']) == {'riding', 'often'}
    with (loop['run']/'replay_training_stats.jsonl').open('a', encoding='utf-8') as out:
        for step in (106, 107, 108):                                  # the average now reads 11.1, inside 1.3
            out.write(json.dumps(dict(type='mimo', step=step, hand_control=dict(joint_error_deg=9.0))) + '\n')
    loop['finish_save']()
    done = run_until(loop, lambda s: (s.get('comparison_window') or {}).get('state') == 'released')
    assert done['tasks']['riding']['completed_step'] == 108 and done['tasks']['often']['completed_step'] == 108
    assert window.read(loop['saved']/'supervisor_control.json') == dict(cmd='run'), 'training resumes'


def test_an_old_review_hold_on_a_listed_test_turns_the_pause_into_a_reported_hold(loop):
    """The review's second path: a review hold left open (its pause lost its ownership during the failed
    round) kept a later pause 'ready' forever. The held test cannot run, so the pause ends: held for review,
    for the earlier round, with its reason in the status and on the page."""
    script = loop['root']/'qualify_hand_execution_2026-09-24.py'
    script.write_text(TINY, encoding='utf-8')
    hand = dict(id='hand_memory_route', title='Remembered hand goals', script=script.name, first_step=999999999,
                first_step_under_hold=50, hold_training_on_failed_comparison=True, every_steps=30, memory_gib=1)
    write(loop['manifest'], dict(lineage='declared', ride_along_steps=3000, tasks=[hand]))
    loop['finish_save']()
    owned = dict(cmd='pause', ability_window_token='t2')
    write(loop['saved']/'supervisor_control.json', owned)
    write(loop['folder']/'status.json', dict(
        tasks=dict(hand_memory_route=dict(status='held for review', completed_step=60, attempt_step=60, title='Remembered hand goals',
                                          review_hold=dict(step=60, reason='Remembered hand goals: the round at step 60 did not pass.'))),
        comparison_window=dict(state='ready', token='t2', run=str(loop['run']), checkpoint=str(loop['checkpoint']),
                               ready_at=time.time(), started_at=time.time()-60, minimum_step=105,
                               targets=dict(hand_memory_route=90), owned_control=owned, previous_control=dict(cmd='run'))))
    state = tick(loop)
    assert state['comparison_window']['state'] == 'held for review' and state['state'] == 'training held for review'
    assert 'step 60 did not pass' in state['pause_reason']
    assert window.read(loop['saved']/'supervisor_control.json') == owned


def test_a_test_due_on_the_pause_checkpoint_runs_in_it_instead_of_opening_a_second_pause(loop):
    """The review's longest-interval case: a pause opened by a triggered test at its longest interval
    released as soon as its listed test finished, although the frequent test was due on its checkpoint."""
    script = loop['root']/'qualify_hand_execution_2026-09-24.py'
    script.write_text(TINY, encoding='utf-8')
    tasks = [dict(id='heavy', title='Heavy', script=script.name, **{'class': 'triggered'}, first_step=10, every_steps=60, memory_gib=1),
             dict(id='often', title='Often', script=script.name, first_step=10, every_steps=30, memory_gib=1)]
    write(loop['manifest'], dict(lineage='declared', ride_along_steps=3000, tasks=tasks))
    write(loop['folder']/'status.json', dict(tasks=dict(heavy=dict(status='complete', completed_step=40, title='Heavy', basis={}),
                                                        often=dict(status='complete', completed_step=75, title='Often'))))
    (loop['run']/'replay_training_stats.jsonl').write_text(json.dumps(dict(type='mimo', step=100)) + '\n', encoding='utf-8')
    first = tick(loop)
    assert first['comparison_window']['targets'] == dict(heavy=100)
    loop['finish_save']()                                            # the pause's checkpoint is step 108
    done = run_until(loop, lambda s: (s.get('comparison_window') or {}).get('state') == 'released')
    assert done['tasks']['heavy']['completed_step'] == 108 and done['tasks']['often']['completed_step'] == 108
    assert done['comparison_window']['not_run'] == {}


def test_a_request_is_refused_without_raising_while_the_control_is_unreadable(loop):
    (loop['saved']/'supervisor_control.json').write_bytes('{"cmd": "run"}'.encode('utf-16'))
    state = dict(tasks={})
    supervisor = dict(active_run=str(loop['run']), timestamp=time.time())
    assert window.request(loop['root'], loop['folder'], state, loop['run'], supervisor, 105) is False
    assert 'comparison_window' not in state


def test_a_test_running_past_twice_the_longest_completed_round_is_reported_not_killed(tmp_path, monkeypatch):
    history = [dict(status='complete', started_at=1000., completed_at=1600.)]              # 10 minutes
    record = dict(status='running', started_at=time.time()-1300, history=history, title='Gaze', command=['p'],
                  output=str(tmp_path/'o'))
    state = dict(tasks=dict(gaze=record), comparison_window=dict(state='ready'))
    monkeypatch.setattr(worker, 'matching_children', lambda r: ([dict(pid=os.getpid(), create_time=0, command=['p'])], None))
    worker.reconcile_task(state, 'gaze')
    assert record['status'] == 'running' and record['stuck']['limit_minutes'] == 20.0
    text = worker.task_warning(record, in_window=True)
    assert 'may be stuck' in text and "a person's decision" in text and 'training stays stopped' in text
    record['started_at'] = time.time()-60
    worker.reconcile_task(state, 'gaze')
    assert 'stuck' not in record


# ------------------------------------------------------------------------------------------ the production manifest
def test_the_production_manifest_declares_its_classes_triggers_and_the_hand_goal_hold():
    manifest = json.loads((REPO/'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    tasks = {t['id']: t for t in manifest['tasks']}
    ids = list(tasks)
    assert manifest['ride_along_steps'] == 3000
    for task in tasks.values():
        worker.validate_schedule(REPO, task, ids, manifest['ride_along_steps'])
    assert {k for k, t in tasks.items() if worker.task_class(t) == 'frequent'} == {'hand_memory_route'}
    # 3,000 -> 12,000 updates by the operator's yes of Oct 1, 2026 (commit f11afbffc; the manifest's cadence_note)
    assert tasks['hand_memory_route']['every_steps'] == 12000
    assert all(t['every_steps'] == 12000 for k, t in tasks.items() if k != 'hand_memory_route')
    gauges = {g['name']: g['tolerance'] for t in tasks.values() for g in (t.get('triggers') or {}).get('gauges', [])}
    assert gauges == dict(eye_forecast_share_150ms=0.07, hand_joint_error_deg=1.65), 'the measured tolerances'
    hand = tasks['hand_memory_route']
    assert hand['hold_training_on_failed_comparison'] is True and 'held' not in hand
    assert hand['first_step'] >= 10**8 and worker.first_step(hand) == 399050
    assert '9:09 AM' in hand['decided'] and 'steers production' in hand['decided']
    assert {'docs/gain_of_function_v3/compare_synergy_physical.py', 'docs/gain_of_function_v3/prove_takeover_restart.py'} <= set(hand['depends'])
    assert tasks['chosen_self_touch']['first_step'] >= 10**8 and tasks['chosen_self_touch']['held'].strip()


def test_the_script_imports_what_the_manifest_declares_it_runs_on():
    """compare_hand_memory_route imports two helpers from its own folder; a change in either is a new
    version of the test (test_version), so the manifest declares both."""
    text = (REPO/'docs/gain_of_function_v3/compare_hand_memory_route_2026-09-24.py').read_text(encoding='utf-8')
    assert 'from compare_synergy_physical import' in text and 'from prove_takeover_restart import' in text


# ------------------------------------------------------------------------------------------ the dashboard
def test_the_dashboard_says_training_is_held_for_review(tmp_path):
    from mimo_bridge.tools import development_dashboard as dd
    saved = tmp_path/'Saved/mimo_bridge'
    run = saved/'experiments/H84_x_seg40'
    run.mkdir(parents=True)
    (run/'replay_training_stats.jsonl').write_text(json.dumps(dict(type='mimo', step=411400, timestamp=100)) + '\n', encoding='utf-8')
    write(saved/'training_continuation.json', dict(active_run=str(run), status='paused for repair or explicit instruction',
                                                   timestamp=995, pause_reason='explicit supervisor pause'))
    reason = 'Remembered hand goals: the round at step 411,350 did not pass its declared comparison.'
    checks = dict(timestamp=990, tasks={}, comparison_window=dict(state='held for review', reason=reason))
    activity = dd.production_activity(tmp_path, checks=checks, now=1000)
    assert activity['state'] == 'attention'
    assert activity['status_label'] == 'Training is held for review: ' + reason
    script = (REPO/'Source/Python/mimo_bridge/tools/development_dashboard.js').read_text(encoding='utf-8')
    assert "training held for review" in script and "t.status==='held for review'" in script
    # the scheduler's warnings: first in the button's summary and on the panel above the results
    assert "tasks.find(t=>t.warning)" in script and "'ability-results-warning'" in script
    assert '.ability-results-warning{' in (REPO/'Source/Python/mimo_bridge/tools/development_dashboard.css').read_text(encoding='utf-8')


def test_the_scheduled_results_rows_carry_the_schedulers_warning(tmp_path):
    from mimo_bridge.tools import development_dashboard as dd
    write(tmp_path/dd.ABILITY_MANIFEST, dict(schema=2, lineage='H84', tasks=[
        dict(id='hand_memory_route', title='Remembered hand goals', first_step=999999999, first_step_under_hold=399050,
             every_steps=3000, memory_gib=12, **{'class': 'frequent'})]))
    warning = 'Remembered hand goals is unmeasured: its last two attempts ended without a result.'
    write(tmp_path/dd.ABILITY_STATUS, dict(timestamp=990, warnings=[warning], tasks=dict(hand_memory_route=dict(
        status='repair needed', due_step=402050, warning=warning))))
    rows = {r['id']: r for r in dd.ability_results(tmp_path)['rows']}
    assert rows['hand_memory_route']['warning'] == warning


def test_the_dashboard_says_a_triggered_comparison_runs_on_its_triggers_and_at_least_every_interval(tmp_path):
    from mimo_bridge.tools import development_dashboard as dd
    write(tmp_path/dd.ABILITY_MANIFEST, dict(schema=2, lineage='H84', tasks=[
        dict(id='body_and_synergy', title='Reachable body goals and coordinated control', first_step=380500,
             every_steps=12000, memory_gib=12, **{'class': 'triggered'}),
        dict(id='hand_memory_route', title='Remembered hand goals', first_step=999999999, first_step_under_hold=399050,
             every_steps=3000, memory_gib=12, **{'class': 'frequent'})]))
    write(tmp_path/dd.ABILITY_STATUS, dict(timestamp=990, tasks=dict(body_and_synergy=dict(due_step=420350))))
    rows = {r['id']: r for r in dd.ability_results(tmp_path)['rows']}
    assert rows['body_and_synergy']['cadence'] == 'triggered' and rows['hand_memory_route']['cadence'] == 'frequent'
    brief = dd.comparison_schedule({}, dict(rows=list(rows.values())), None, {}, 1000, False)
    assert ('on a saved checkpoint when one of its declared triggers fires, and at least every 12,000 training updates; '
            'next at step 420,350.') in brief['test_status']
