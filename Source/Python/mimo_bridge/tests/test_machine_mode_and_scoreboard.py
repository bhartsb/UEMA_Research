# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The machine mode (tools/machine_mode.py) and the gain-of-function scoreboard (tools/gof_scoreboard.py).

The operator's recovery directive of Oct 5, 2026, sections 12 and 13: exactly one machine mode at the top of the
dashboard, the week's hours in each mode with the reason of every training pause, and five stages whose states are
computed from test registry rows. Every folder here is a temporary one; the processes are small dictionaries in the
form test_runner.scan_processes() returns.
"""
import json
from pathlib import Path

import pytest

from mimo_bridge.tools import gof_scoreboard as board
from mimo_bridge.tools import machine_mode as mm
from mimo_bridge.tools import test_registry as tr
from test_run_in_training_pause import complete_row, write_registry

NOW = tr.parse_clock('Oct 5, 2026 3:00 PM PDT')
HOUR = 3600.0
LINEAGE = 'H84_fresh_drivers_20260918'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def trainer(pid, run):
    return dict(pid=pid, ppid=1, created=NOW - 600, name='python.exe',
                cmdline=['python.exe', '-u', '-m', 'mimo_bridge.train_mimo', '--out',
                         'C:/r/Saved/mimo_bridge/experiments/' + run])


@pytest.fixture
def machine(tmp_path):
    saved = tmp_path/'Saved/mimo_bridge'
    run = saved/'experiments'/(LINEAGE + '_seg41o_1431')
    run.mkdir(parents=True)
    write(saved/'training_continuation.json', dict(active_run=str(run)))
    write(saved/'supervisor_control.json', dict(cmd='run', reason='relaunched'))
    write_registry(tmp_path, [complete_row('probe'), complete_row('heavy')])
    return dict(root=tmp_path, saved=saved, run=run, path=tmp_path/'docs/gain_of_function_v4/test_registry.json')


def audit_with(**more):
    return dict(dict(available=True, running_rows=[], test_processes=[], development_runs=[]), **more)


# ------------------------------------------------------------------------------------------ the mode now
def test_exactly_one_mode_is_computed_from_the_trainer_the_control_file_and_the_running_rows(machine):
    root = machine['root']
    training = mm.mode(root, processes=[trainer(10, LINEAGE + '_seg41o_1431')], audit_view=audit_with(), now=NOW)
    assert training['mode'] == 'TRAINING' and training['errors'] == []
    assert training['sentence'].startswith('TRAINING: the trainer of run %s_seg41o_1431 is alive (process 10' % LINEAGE)
    write(machine['saved']/'supervisor_control.json', dict(cmd='pause', reason='hold for the review bench'))
    live = audit_with(running_rows=[dict(id='probe', alive=True, how='its job process 5 is alive',
                                         resource_class='TEST_WINDOW', production_run=None)])
    window = mm.mode(root, processes=[], audit_view=live, now=NOW)
    assert window['mode'] == 'TEST WINDOW' and window['live_tests'] == ['probe']
    assert window['sentence'] == ('TEST WINDOW: no production trainer process is alive and 1 test that needs '
                                  'training stopped is live: probe.')
    paused = mm.mode(root, processes=[], audit_view=audit_with(), now=NOW)
    assert paused['mode'] == 'PAUSED FOR OPERATOR' and paused['reason'] == 'hold for the review bench'
    assert paused['sentence'] == ('PAUSED FOR OPERATOR: no production trainer process is alive and no test that needs '
                                  'training stopped is running. Reason on record (supervisor_control.json): hold for '
                                  'the review bench.')
    for found in (training, window, paused):
        assert found['mode'] in mm.MODES and found['modes'] == list(mm.MODES)


def test_the_tests_beside_a_hold_are_counted_and_named_under_paused_for_operator(machine):
    """Found by the pages work on Oct 5, 2026 at 4:24 AM PDT: the sentence read "3 tests are live: A, B." while it named
    two. Each test is one named row or one process without a row, named by its process id; two processes of one row
    count once. Since the dashboard review's finding F9 (Oct 5, 2026) a test that can run beside training, and a test
    process no row names, make no test window: they are listed under PAUSED FOR OPERATOR."""
    write(machine['saved']/'supervisor_control.json', dict(cmd='pause', reason='held'))
    live = audit_with(running_rows=[dict(id='probe', alive=True, how='its job process 5 is alive',
                                         resource_class='BESIDE_TRAINING', production_run=None)],
                      test_processes=[dict(pid=5, row='probe'), dict(pid=6, row='probe'), dict(pid=30848, row=None)])
    paused = mm.mode(machine['root'], processes=[], audit_view=live, now=NOW)
    assert paused['mode'] == 'PAUSED FOR OPERATOR'
    assert paused['sentence'].endswith('Reason on record (supervisor_control.json): held. 2 tests run beside the hold: '
                                       'probe, 1 test process that no row names (process 30848).')
    two = audit_with(test_processes=[dict(pid=8, row=None), dict(pid=9, row=None)])
    assert mm.mode(machine['root'], processes=[], audit_view=two, now=NOW)['sentence'].endswith(
        '2 tests run beside the hold: 2 test processes that no row names (processes 8, 9).')


def test_a_running_row_whose_process_is_gone_and_a_stale_segment_row_do_not_make_a_test_window(machine):
    write(machine['saved']/'supervisor_control.json', dict(cmd='pause', reason='held'))
    stale = audit_with(running_rows=[dict(id='probe', alive=False, how=None, resource_class='TEST_WINDOW',
                                          production_run=None),
                                     dict(id='segment', alive=True, how='the trainer of run x is alive',
                                          resource_class='PRODUCTION_SEGMENT', production_run='x')])
    assert mm.mode(machine['root'], processes=[], audit_view=stale, now=NOW)['mode'] == 'PAUSED FOR OPERATOR'


def test_the_mode_states_what_does_not_fit(machine):
    root = machine['root']
    beside = audit_with(running_rows=[dict(id='heavy', alive=True, how='its job process 7 is alive',
                                           resource_class='TEST_WINDOW', production_run=None)])
    found = mm.mode(root, processes=[trainer(10, LINEAGE + '_seg41o_1431')], audit_view=beside, now=NOW)
    assert found['mode'] == 'TRAINING' and found['errors'] == [
        'Test heavy needs training stopped (resource class TEST_WINDOW) and is running beside the trainer.']
    other = mm.mode(root, processes=[trainer(10, LINEAGE + '_seg41x_0000')], audit_view=audit_with(), now=NOW)
    assert "is not the run the supervisor names" in other['errors'][0]
    # stopped with no pause and no stop reason on record
    unexplained = mm.mode(root, processes=[], audit_view=audit_with(), now=NOW)
    assert unexplained['mode'] == 'PAUSED FOR OPERATOR' and unexplained['reason'] is None
    assert unexplained['errors'] == ['Training is stopped: no production trainer process is alive, '
                                     'supervisor_control.json does not say pause, and the active run holds no stop '
                                     'reason.']
    write(machine['run']/'stop_reason.json', dict(reason='the hourly watch: C: under 8 GB', timestamp=NOW - 60))
    stopped = mm.mode(root, processes=[], audit_view=audit_with(), now=NOW)
    assert stopped['reason'] == 'the hourly watch: C: under 8 GB' and stopped['errors'] == []
    assert "Reason on record (the active run's stop_reason.json)" in stopped['sentence']


def test_the_mode_reads_the_machine_itself_when_nothing_is_given(machine):
    write(machine['saved']/'supervisor_control.json', dict(cmd='pause', reason='held for the operator'))
    found = mm.mode(machine['root'], path=machine['path'], now=NOW)
    assert found['mode'] in mm.MODES, 'computed from the real process scan and the real audit'


# ------------------------------------------------------------------------------------------ experiment trainers
# Found on Oct 13, 2026: the trainers of four registered experiment runs on D:/UEMA_Research/external/D/UEMA_scratch (their --out outside
# Saved/mimo_bridge/experiments) trained for 6.67 hours of the week and the mode read PAUSED FOR OPERATOR while one was
# alive. A row names its run folder by its last run's log file's folder or by the folder after --out in its recorded
# command (the registry keeps a command's first 60 arguments, so --out is often cut off).
def exp_trainer(pid, folder, created=NOW - 600, ppid=1):
    return dict(pid=pid, ppid=ppid, created=created, name='python.exe',
                cmdline=['python.exe', '-u', '-m', 'mimo_bridge.train_mimo', '--out', str(folder).replace('\\', '/')])


def slashed(path):
    return str(path).replace('\\', '/')


def test_an_experiment_trainer_a_row_names_reads_training_with_the_experiment_sentence(machine):
    root = machine['root']
    write(machine['saved']/'supervisor_control.json', dict(cmd='pause', reason='production held for the body-1 arm'))
    folder = root/'scratch/body1_extension/run/B1E_x'
    elsewhere = root/'scratch/other/run/B1F_y'
    write_registry(root, [
        complete_row('probe'), complete_row('heavy'),
        complete_row('exp_arm', status='running', last_run=dict(
            started_at=NOW - 600, log=str(folder/'trainer.log'), resource_class='PRODUCTION_SEGMENT', pid=20,
            pid_created=NOW - 600, command=['python.exe', '-u', '-m', 'mimo_bridge.train_mimo', '--a', '1'])),
        complete_row('exp_pid', status='running', last_run=dict(
            started_at=NOW - 300, log=str(root/'test_runs/logs/exp_pid/run.log'), resource_class='PRODUCTION_SEGMENT',
            pid=21, pid_created=NOW - 300))])
    live = audit_with(running_rows=[dict(id='exp_arm', alive=True, how='its job process 20 is alive',
                                         resource_class='PRODUCTION_SEGMENT', production_run=None)])
    # the Windows venv launcher and its child carry one command line: one trainer
    found = mm.mode(root, processes=[exp_trainer(20, folder), exp_trainer(22, folder, ppid=20)], audit_view=live,
                    now=NOW, path=machine['path'])
    assert found['mode'] == 'TRAINING' and found['training_of'] == 'experiment' and found['errors'] == []
    assert found['sentence'] == (
        'TRAINING (experiment run, not production): the trainer of run B1E_x (run folder %s), row exp_arm, is alive '
        '(process 20, since %s).' % (slashed(folder), tr.clock(NOW - 600)))
    assert [(t['pid'], t['row'], t['run']) for t in found['experiment_trainers']] == [(20, 'exp_arm', 'B1E_x')]
    assert found['since'] == NOW - 600 and found['trainers'] == []
    # a test that needs training stopped beside an experiment trainer is listed as an error, as beside production's
    beside = audit_with(running_rows=live['running_rows'] + [dict(id='heavy', alive=True, how='its job process 7 is '
                                                                  'alive', resource_class='TEST_WINDOW',
                                                                  production_run=None)])
    assert mm.mode(root, processes=[exp_trainer(20, folder)], audit_view=beside, now=NOW, path=machine['path'])[
        'errors'] == ['Test heavy needs training stopped (resource class TEST_WINDOW) and is running beside the trainer.']
    # a row's recorded job process (process id and creation time within 1 s) names a trainer whose folder no row names
    by_pid = mm.mode(root, processes=[exp_trainer(21, elsewhere, created=NOW - 300)], audit_view=audit_with(), now=NOW,
                     path=machine['path'])
    assert by_pid['mode'] == 'TRAINING' and by_pid['experiment_trainers'][0]['row'] == 'exp_pid'
    assert by_pid['experiment_trainers'][0]['how'] == 'it is the recorded job process of row exp_pid'
    assert by_pid['sentence'].startswith('TRAINING (experiment run, not production): the trainer of run B1F_y (run '
                                         'folder %s), row exp_pid, is alive (process 21' % slashed(elsewhere))


def test_a_trainer_process_no_row_names_keeps_the_mode_and_is_listed_as_an_error(machine):
    root = machine['root']
    write(machine['saved']/'supervisor_control.json', dict(cmd='pause', reason='held for the operator'))
    folder = root/'scratch/unregistered/run/U1'
    write_registry(root, [complete_row('probe'), complete_row('exp_pid', status='running', last_run=dict(
        started_at=NOW - 300, log=str(root/'test_runs/logs/exp_pid/run.log'), pid=21, pid_created=NOW - 300))])
    found = mm.mode(root, processes=[exp_trainer(30, folder)], audit_view=audit_with(), now=NOW, path=machine['path'])
    assert found['mode'] == 'PAUSED FOR OPERATOR' and found['training_of'] is None and found['experiment_trainers'] == []
    assert found['errors'] == [
        'Process 30 (started %s) is a trainer (mimo_bridge.train_mimo) whose run folder is %s, and no test registry row '
        'names it (no row\'s last run names that folder by the folder after --out in its recorded command or by its log '
        'file\'s folder, and no row records it as its job process): the machine mode does not count its training.'
        % (tr.clock(NOW - 600), slashed(folder))]
    assert [t['pid'] for t in found['unnamed_trainers']] == [30]
    # a reused process id: the row's recorded process id with another creation time is not the row's process
    reused = mm.mode(root, processes=[exp_trainer(21, folder, created=NOW - 200)], audit_view=audit_with(), now=NOW,
                     path=machine['path'])
    assert reused['mode'] == 'PAUSED FOR OPERATOR' and [t['pid'] for t in reused['unnamed_trainers']] == [21]
    # no --out argument, beside a live production trainer: the mode stays production's TRAINING, the error is listed
    bare = dict(exp_trainer(31, folder), cmdline=['python.exe', '-m', 'mimo_bridge.train_mimo', '--steps', '5'])
    both = mm.mode(root, processes=[trainer(10, LINEAGE + '_seg41o_1431'), bare], audit_view=audit_with(), now=NOW,
                   path=machine['path'])
    assert both['mode'] == 'TRAINING' and both['training_of'] == 'production'
    assert both['sentence'].startswith('TRAINING: the trainer of run %s_seg41o_1431 is alive' % LINEAGE)
    assert len(both['errors']) == 1 and 'whose command line carries no --out argument' in both['errors'][0]


# ------------------------------------------------------------------------------------------ the week
def stats(folder, launch_at, records_until=None, command=None):
    lines = [json.dumps(dict(type='launch', timestamp=launch_at, step=1000, code_commit='abc123def'))]
    at = launch_at + 300
    while records_until is not None and at <= records_until:
        lines.append(json.dumps(dict(type='mimo', step=1100, timestamp=at, model_steps_per_s=0.4)))
        at += 300
    if command is not None:
        lines.append(json.dumps(dict(dict(type='overseer_cmd'), **command)))
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'replay_training_stats.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_the_weeks_hours_and_each_pauses_reason_come_from_the_run_folders(machine, monkeypatch):
    from mimo_bridge.tools import test_runner
    monkeypatch.setattr(test_runner, 'scan_processes', lambda: [])
    root, exp = machine['root'], machine['saved']/'experiments'
    start = NOW - 100 * HOUR
    a, b, c = (exp/(LINEAGE + s) for s in ('_seg41a_0100', '_seg41b_0200', '_seg41o_1431'))
    stats(a, start, start + 10 * HOUR)                                   # 10 h of training, ended by stage_done
    write(a/'stage_done.json', dict(finished_at=start + 10 * HOUR, steps=2000))
    write(a/'stop_reason.json', dict(reason='the watch: step rate under 0.42', timestamp=start + 10 * HOUR))
    stats(b, start + 14 * HOUR, start + 34 * HOUR,                       # a 4 h pause, then 20 h of training
          command=dict(timestamp=start + 34 * HOUR, step=3000, cmd='snapshot_and_stop',
                       reason='coordinated evaluation pause'))
    write(b/'stage_done.json', dict(finished_at=start + 34 * HOUR, steps=3000))
    stats(c, start + 40 * HOUR, start + 60 * HOUR)                       # a 6 h pause, 20 h, then no stage_done
    write(machine['saved']/'supervisor_control.json', dict(cmd='pause', reason='hold for the operator'))
    # test time inside the pauses: a pause receipt of 2 h in the first, a test row's last run of 3 h in the second,
    # and a run of 5 h in the open pause that overlaps a receipt of 4 h by 2 h (counted once: 7 h in all)
    write(root/'Saved/gof_observer_20260924/pauses/p1/receipt.json',
          dict(started=start + 11 * HOUR, finished=start + 13 * HOUR, jobs=[dict(name='probe')],
               owned_control=dict(cmd='pause', reason='Coordinated evaluation pause: probe')))
    write(root/'Saved/gof_observer_20260924/pauses/p2/receipt.json',
          dict(started=start + 70 * HOUR, finished=start + 74 * HOUR, jobs=[dict(name='heavy')]))
    write_registry(root, [
        complete_row('probe', last_run=dict(started_at=start + 35 * HOUR, ended_at=start + 38 * HOUR)),
        complete_row('heavy', last_run=dict(started_at=start + 72 * HOUR, ended_at=start + 77 * HOUR)),
        complete_row('segment', status='running', last_run=dict(
            started_at=start, ended_at=None, log='Saved/mimo_bridge/experiments/%s_seg41a_0100/trainer.log' % LINEAGE))])
    week = mm.week(root, now=NOW, path=machine['path'])
    # 168 h in all: 50 of training; pauses of 68 (before the first run), 4, 6 and 40 h, 12 of them test time
    assert week['hours'] == {'TRAINING': 50.0, 'TEST WINDOW': 12.0, 'PAUSED FOR OPERATOR': 106.0}
    assert week['runs'] == 3 and week['lineage'] == LINEAGE and week['window_hours'] == 168.0
    pauses = [p for p in week['pauses'] if p['after_run']]
    assert [(p['hours'], p['test_window_hours'], p['paused_for_operator_hours']) for p in pauses] == [
        (4.0, 2.0, 2.0), (6.0, 3.0, 3.0), (40.0, 7.0, 33.0)]
    assert [(p['reason'], p['reason_source']) for p in pauses] == [
        ('the watch: step rate under 0.42', '%s_seg41a_0100/stop_reason.json' % LINEAGE),
        ('coordinated evaluation pause', 'the overseer_cmd record of %s_seg41b_0200' % LINEAGE),
        ('hold for the operator', 'supervisor_control.json')]
    assert pauses[-1]['open'] is True and pauses[-1]['to_text'] == 'now' and pauses[1]['stopped_at_step'] == 3000
    # the time before the first run of the window is a pause with no run before it and no reason on record
    first = week['pauses'][0]
    assert first['after_run'] is None and first['hours'] == 68.0 and first['reason'] == 'not recorded in the run folder'
    assert len(week['not_carried']) == 4 and 'supervisor_control.json holds only its present content' in week['not_carried'][0]


def test_a_pause_without_any_record_says_so(machine, monkeypatch):
    from mimo_bridge.tools import test_runner
    monkeypatch.setattr(test_runner, 'scan_processes', lambda: [])
    exp = machine['saved']/'experiments'
    start = NOW - 50 * HOUR
    stats(exp/(LINEAGE + '_seg41a_0100'), start, start + 10 * HOUR, command=dict(timestamp=start + 10 * HOUR, step=9,
                                                                                 cmd='snapshot_and_stop'))
    stats(exp/(LINEAGE + '_seg41o_1431'), start + 20 * HOUR, start + 50 * HOUR)
    week = mm.week(machine['root'], now=NOW, path=machine['path'])
    middle = next(p for p in week['pauses'] if p['after_run'] == LINEAGE + '_seg41a_0100')
    assert middle['reason'] == 'not recorded in the run folder' and middle['reason_source'] is None
    assert middle['hours'] == 10.0 and middle['test_window_hours'] == 0.0


def test_a_live_trainers_run_counts_as_training_up_to_now(machine, monkeypatch):
    from mimo_bridge.tools import test_runner
    name = LINEAGE + '_seg41o_1431'
    monkeypatch.setattr(test_runner, 'scan_processes', lambda: [trainer(10, name)])
    stats(machine['run'], NOW - 5 * HOUR, NOW - 1 * HOUR)                # its newest record is 1 h old
    week = mm.week(machine['root'], now=NOW, path=machine['path'])
    assert week['hours']['TRAINING'] == 5.0 and not any(p['open'] for p in week['pauses'])


def test_the_week_counts_an_experiment_runs_life_as_training_inside_the_production_pause(machine, monkeypatch):
    from mimo_bridge.tools import test_runner
    monkeypatch.setattr(test_runner, 'scan_processes', lambda: [])
    root, exp = machine['root'], machine['saved']/'experiments'
    start = NOW - 50 * HOUR
    a = exp/(LINEAGE + '_seg41a_0100')
    stats(a, start, start + 10 * HOUR)                                   # production: 10 h, then a 40 h open pause
    write(a/'stage_done.json', dict(finished_at=start + 10 * HOUR, steps=2000))
    write(machine['saved']/'supervisor_control.json', dict(cmd='pause', reason='hold for the operator'))
    scratch = root/'scratch'
    x, y, z, w = (scratch/'run'/n for n in ('B1LO_x', 'B1C_y', 'B1E_z', 'unregistered_w'))
    stats(x, start + 8 * HOUR, start + 12 * HOUR)                        # 2 h inside production's life, 2 h after it
    write(x/'stage_done.json', dict(finished_at=start + 12 * HOUR, steps=10))
    stats(y, start + 20 * HOUR, start + 22 * HOUR)                       # named by --out in its recorded command
    write(y/'stage_done.json', dict(finished_at=start + 22 * HOUR, steps=10))
    stats(z, start + 30 * HOUR, start + 31 * HOUR)                       # no stage_done: ends at its newest record
    stats(w, start + 35 * HOUR, start + 36 * HOUR)                       # no row names it: no training on record
    write_registry(root, [
        complete_row('arm_x', status='done', last_run=dict(started_at=start + 8 * HOUR, ended_at=start + 12 * HOUR,
                                                           log=str(x/'trainer.log'), resource_class='PRODUCTION_SEGMENT')),
        complete_row('arm_y', status='done', last_run=dict(
            started_at=start + 20 * HOUR, ended_at=start + 22 * HOUR, log=str(scratch/'test_runs/logs/arm_y/1.log'),
            resource_class='PRODUCTION_SEGMENT',
            command=['python.exe', '-u', '-m', 'mimo_bridge.train_mimo', '--out', str(y)])),
        complete_row('arm_z', status='failed', last_run=dict(started_at=start + 30 * HOUR, log=str(z/'trainer.log'),
                                                             resource_class='PRODUCTION_SEGMENT')),
        # a test that needs training stopped, 21 h to 24 h: its first hour is inside arm_y's life and counts as TRAINING
        complete_row('probe', status='done', last_run=dict(started_at=start + 21 * HOUR, ended_at=start + 24 * HOUR))])
    week = mm.week(root, now=NOW, path=machine['path'])
    # 168 h: production 10; experiment outside it 2 + 2 + 1 = 5; test window 2 (22 h to 24 h); the rest 151
    assert week['hours'] == {'TRAINING': 15.0, 'TEST WINDOW': 2.0, 'PAUSED FOR OPERATOR': 151.0}
    assert week['training_split'] == {'production': 10.0, 'experiment': 5.0} and week['runs'] == 1
    assert [(e['row'], e['run'], e['hours'], e['ended_by']) for e in week['experiment_runs']] == [
        ('arm_x', 'B1LO_x', 4.0, 'stage_done.json'), ('arm_y', 'B1C_y', 2.0, 'stage_done.json'),
        ('arm_z', 'B1E_z', 1.0, 'its newest stats record')]
    assert week['experiment_runs'][0]['folder'] == slashed(x)
    # the pause stays production's pause: its experiment-training hours leave its PAUSED FOR OPERATOR share
    pause = next(p for p in week['pauses'] if p['after_run'] == LINEAGE + '_seg41a_0100')
    assert (pause['hours'], pause['experiment_training_hours'], pause['test_window_hours'],
            pause['paused_for_operator_hours']) == (40.0, 5.0, 2.0, 33.0)
    assert pause['experiment_runs'] == ['B1LO_x', 'B1C_y', 'B1E_z'] and pause['open'] is True
    assert pause['reason'] == 'hold for the operator'
    assert 'experiment' in week['definition'] and 'an hour inside both counts once, as production' in week['definition']
    # the trainer of arm_z still alive: its life runs to now (20 h), as a live production trainer's does
    monkeypatch.setattr(test_runner, 'scan_processes', lambda: [exp_trainer(40, z)])
    alive = mm.week(root, now=NOW, path=machine['path'])
    assert alive['training_split'] == {'production': 10.0, 'experiment': 24.0}
    assert alive['experiment_runs'][-1]['to_text'] == 'now'


# ------------------------------------------------------------------------------------------ the scoreboard
READING = dict(name='goal distance, planned minus controller', value=-0.12, unit='token units',
               meaning='below 0 = the planned command ended nearer its goal', baseline='0: no nearer',
               statistically_clear='yes', statistical_basis='95% interval over 8 bodies entirely below 0',
               gof_material='yes')


def evidence_row(tmp, name, stage, readings, exit_code=0, claim='planned commands end nearer their goal'):
    """A row run through the registry's runner functions whose result file, written during the run, holds `readings`
    (none when None): the runner copies them into the result it seals (the dashboard review of Oct 5, 2026, finding
    F4: readings count only from the run's own result file)."""
    path = tmp/'docs/gain_of_function_v4/test_registry.json'
    data = tr.load(path) if path.exists() else dict(rows=[])
    result_file = tmp/('%s_result.json' % name)
    rows = [r for r in data['rows'] if r['id'] != name] + [complete_row(
        name, evidence_for=dict(stage=stage, claim=claim), result_source=dict(file=str(result_file), keys=['readings']))]
    write_registry(tmp, rows)
    assert tr.job_started(name, path=path) is None
    verdict = 'pass' if exit_code == 0 else 'fail'          # the runner copies a file's readings with its verdict
    result_file.write_text(json.dumps(dict(verdict=verdict, readings=readings) if readings is not None else
                                      dict(verdict=verdict)), encoding='utf-8')
    assert tr.finish_job(name, exit_code=exit_code, path=path)[0] is None
    return path


def test_no_row_names_a_stage_so_no_gain_of_function_is_demonstrated(tmp_path):
    write_registry(tmp_path, [complete_row('probe', status='done')])
    found = board.scoreboard(tmp_path, defects=[])
    assert found['headline'] == 'No gain of function demonstrated.' and found['gain_of_function_demonstrated'] is False
    assert [s['key'] for s in found['stages']] == ['SENSING', 'PREDICTION', 'ACTION_CAUSALITY', 'PLANNER',
                                                   'INTENTIONAL_FUNCTION']
    assert all(s['state'] == 'NOT DEMONSTRATED' and s['reasons'] == ['No test registry row names this stage as '
                                                                     'evidence.'] for s in found['stages'])
    assert found['sentence'] == 'No gain of function demonstrated. 0 of 5 stages are demonstrated.'
    assert found['stages'][3]['required_evidence'] == ('Planned command produces a better real physical outcome than '
                                                       'the motor controller baseline.')


def test_a_stage_is_demonstrated_only_by_a_runners_pass_with_both_judgements_and_no_open_blocker(tmp_path):
    path = evidence_row(tmp_path, 'planner_gate', 'PLANNER', [READING])
    planner = lambda found: next(s for s in found['stages'] if s['key'] == 'PLANNER')   # noqa: E731
    shown = board.scoreboard(tmp_path, defects=[])
    assert planner(shown)['state'] == 'DEMONSTRATED' and planner(shown)['evidence_rows'][0]['holds'] is True
    assert shown['gain_of_function_demonstrated'] is False, 'only the fifth stage is gain of function'
    assert shown['sentence'] == 'No gain of function demonstrated. 1 of 5 stages are demonstrated: Planner works.'
    # an open GOF-BLOCKER defect that names the stage
    blocker = dict(id='PLANNER_SEARCH_UNSTABLE', title='t', stage='PLANNER', severity='GOF-BLOCKER', state='FOUND')
    blocked = planner(board.scoreboard(tmp_path, defects=[blocker]))
    assert blocked['state'] == 'NOT DEMONSTRATED' and blocked['reasons'] == [
        '1 open GOF-BLOCKER defect names this stage: PLANNER_SEARCH_UNSTABLE (FOUND).']
    assert planner(board.scoreboard(tmp_path, defects=[dict(blocker, state='VERIFIED')]))['state'] == 'DEMONSTRATED'
    assert planner(board.scoreboard(tmp_path, defects=[dict(blocker, severity='TRAINING-RISK')]))['state'] == 'DEMONSTRATED'
    # each way the evidence does not hold
    for readings, why in (([dict(READING, gof_material=tr.GOF_ANSWERS[2])],
                           'is not judged large enough to matter for gain of function (GOF threshold not yet defined)'),
                          ([dict(READING, statistically_clear='not judged', statistical_basis='')],
                           'is not statistically clear (not judged)'),
                          (None, 'its result carries no structured reading with the two judgements')):
        evidence_row(tmp_path, 'planner_gate', 'PLANNER', readings)
        stage = planner(board.scoreboard(tmp_path, defects=[]))
        assert stage['state'] == 'NOT DEMONSTRATED' and why in stage['reasons'][0], why
    # readings typed into the row after a sealed PASS are not the run's: the result differs from the sealed one (F4)
    evidence_row(tmp_path, 'planner_gate', 'PLANNER', None)
    row = tr.rows_with_id(tr.load(path), 'planner_gate')[0]
    assert tr.update_row('planner_gate', dict(result=dict(row['result'], readings=[READING])), path=path)
    typed = planner(board.scoreboard(tmp_path, defects=[]))
    assert typed['state'] == 'NOT DEMONSTRATED' and "is not the one the runner sealed at the run's end" in \
        typed['reasons'][0] and 'typed into the row' in typed['reasons'][0]
    evidence_row(tmp_path, 'planner_gate', 'PLANNER', [READING], exit_code=1)
    assert 'its outcome is FAIL' in planner(board.scoreboard(tmp_path, defects=[]))['reasons'][0]
    # a PASS typed into the file is not a runner's
    data = tr.load(path)
    data['rows'][0].update(status='done', outcome='PASS', result=dict(verdict='PASS', readings=[READING]))
    data['rows'][0]['last_run']['state'] = 'PASS'
    path.write_text(tr.dumps(data) + '\n', encoding='utf-8')
    assert "its PASS does not carry the runner's verified seal" in planner(
        board.scoreboard(tmp_path, defects=[]))['reasons'][0]


def test_gain_of_function_is_demonstrated_by_the_fifth_stage_alone(tmp_path):
    evidence_row(tmp_path, 'reach_and_repeat', 'INTENTIONAL_FUNCTION', [READING],
                 claim='the left hand reaches a chosen toy on 8 of 8 bodies and again a day later')
    found = board.scoreboard(tmp_path, defects=[])
    assert found['gain_of_function_demonstrated'] is True
    assert found['demonstrated_abilities'] == ['the left hand reaches a chosen toy on 8 of 8 bodies and again a day later']
    assert found['headline'].startswith('Gain of function demonstrated: the left hand reaches')


def test_this_checkouts_scoreboard_says_no_gain_of_function_is_demonstrated():
    repo = Path(__file__).resolve().parents[4]
    found = board.scoreboard(repo, repo)
    assert found['headline'] == 'No gain of function demonstrated.' and found['errors'] == []
    assert [s['state'] for s in found['stages']] == ['NOT DEMONSTRATED'] * 5
    blocked = {s['key']: [b['id'] for b in s['blocking_defects']] for s in found['stages']}
    assert 'VESTIBULAR_TOKEN_NO_HEAD_TILT' in blocked['SENSING'] and 'PLANNER_SEARCH_UNSTABLE' in blocked['PLANNER']
