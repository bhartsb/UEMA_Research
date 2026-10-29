# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The runner entry point (tools/test_runner.py) and the registry's fields of Oct 5, 2026 (tools/test_registry.py).

The operator's recovery directive, sections 6 to 8: a test starts only from a complete row; the runner writes the
start (running, start time, commit, checkpoint, owner, resource class, process id and creation time) and the end (PASS,
FAIL, INVALID or STOPPED, end time, result file, next action); a hand write may not take the runner's fields; the audit
finds a running row without a live process and a live test process without a running row. Every registry here is a
temporary file and every process is one this file starts: a short Python command, never a world, a model or a trainer.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mimo_bridge.tools import test_registry as tr
from mimo_bridge.tools import test_runner as runner
from test_run_in_training_pause import complete_row, write_registry

PY = sys.executable
NEW = 'Oct 5, 2026 2:00 AM PDT'                       # a written_at after tr.ADDED_FROM: the row owes the added fields
DIAGRAM = dict(input='the saved checkpoint 481,800', operation='run the command once',
               measurement='exit code: 0 or another number', decision='pass keeps the switch off; fail goes back')


def new_row(name, **changes):
    """A complete row written after the contract's time: every field of Oct 5, 2026 filled."""
    row = complete_row(name, written_at=NEW, component='TEST_REGISTRY',
                       gof_link='A test whose row is wrong or missing costs a training pause.',
                       target_metric='exit code of the command: 0 = ran to its end, any other = did not',
                       baseline='the command not run: no exit code', statistical_line='one run; the code is exact',
                       material_gof_line=tr.NOT_YET_DEFINED, diagram=dict(DIAGRAM),
                       cost=dict(training_stopped=False, memory_gb=1, where='the processor', time_limit_s=120),
                       on_pass='the switch stays off', on_fail='the cause goes to the builder')
    row.update(changes)
    return row


@pytest.fixture
def reg(tmp_path, monkeypatch):
    """A temporary registry with one complete new row per name, the runner's files in the test's own folder, the
    defect register's rule switched off (its own tests hold it) and the manifest's ids fixed."""
    monkeypatch.setenv('UEMA_TEST_RUNS_DIR', str(tmp_path/'runs'))
    monkeypatch.setattr(tr, 'component_ids', lambda root=None: {'TEST_REGISTRY', 'DASHBOARD', 'EXPLAINER'})
    monkeypatch.setattr(tr, 'code_state', lambda cwd=None, keep_s=30.0: ('abc123def', False, 'a/test-branch'))
    path = write_registry(tmp_path, [new_row(n) for n in ('probe', 'look', 'third')])
    return dict(path=path, tmp=tmp_path)


def row_of(reg, name):
    return next(r for r in tr.load(reg['path'])['rows'] if r['id'] == name)


def run(reg, name, code, **kw):
    kw.setdefault('check_rule', False)
    return runner.run_command(name, [PY, '-c', code], path=reg['path'], root=reg['tmp'], **kw)


# ------------------------------------------------------------------------------------------ the added fields
@pytest.mark.parametrize('field', tr.ADDED_FIELDS)
def test_a_row_written_after_the_contracts_time_is_refused_without_each_added_field(reg, field):
    rows = [{k: v for k, v in new_row('probe').items() if k != field}]
    path = write_registry(reg['tmp'], rows)
    why = tr.refusal('probe', path=path)
    assert why is not None and '(%s)' % field in why
    assert runner.refusal('probe', path=path, check_rule=False) == why


def test_a_row_written_before_the_contracts_time_is_judged_as_before_and_lists_its_gaps(reg):
    path = write_registry(reg['tmp'], [complete_row('old')])            # written Oct 2, no added field
    assert tr.refusal('old', path=path) is None
    view = tr.dashboard_view(reg['tmp'], audit=False)
    old = view['groups'][0]['rows'][0]
    assert old['gaps'] == ['component', 'gain-of-function link', 'direct target metric', 'baseline', 'statistical line',
                           'material gain-of-function line', 'diagram', 'time limit in seconds']
    assert old['diagram_missing'] and old['component_missing']
    assert [e['kind'] for e in view['errors']] == ['no component or diagram']


@pytest.mark.parametrize('change,expected', [
    (dict(diagram=dict(DIAGRAM, decision='')), 'the diagram has no decision phrase (diagram.decision)'),
    (dict(diagram=dict(DIAGRAM, input='x' * 121)), "the diagram's input phrase is 121 characters"),
    (dict(diagram='input -> operation'), 'the diagram is not {input, operation, measurement, decision}'),
    (dict(component='HEAD_REPAIR'), "component 'HEAD_REPAIR' is not an id of the component manifest"),
    (dict(cost=dict(training_stopped=False, memory_gb=1, where='cpu')), 'cost gives no time limit in seconds'),
    (dict(material_gof_line='  '), 'material gain-of-function line is empty'),
])
def test_the_added_fields_are_judged_in_their_declared_form(reg, change, expected):
    path = write_registry(reg['tmp'], [new_row('probe', **change)])
    assert expected in tr.refusal('probe', path=path)


def test_a_done_row_may_omit_its_diagram(reg):
    row = {k: v for k, v in new_row('probe', status='done').items() if k != 'diagram'}
    assert tr.row_problems(row, tr.DEFAULT_REQUIRED) == []
    assert 'diagram is missing' in dict((k, t) for k, t in tr.row_problems(dict(row, status='queued'),
                                                                           tr.DEFAULT_REQUIRED))['diagram']


# ------------------------------------------------------------------------------------------ the run
def test_a_run_writes_its_start_and_its_end_by_itself(reg):
    seen = reg['tmp']/'seen.json'
    code, state, log = runner.run_command(
        'probe', [PY, '-c', 'import json,os,sys; json.dump(dict(row=os.environ["UEMA_TEST_ROW"], '
                            'run=os.environ["UEMA_TEST_RUN"]), open(sys.argv[1], "w")); print("ran")', str(seen)],
        path=reg['path'], root=reg['tmp'], check_rule=False, checkpoint='D:/x/ckpt_step_481800', owner='the builder')
    assert (code, state) == (0, 'PASS') and 'ran' in Path(log).read_text()
    row = row_of(reg, 'probe')
    last = row['last_run']
    assert row['status'] == 'done' and row['outcome'] == 'PASS' and row['next_action'] == 'the switch stays off'
    assert last['state'] == 'PASS' and last['exit_code'] == 0 and last['commit'] == 'abc123def'
    assert last['commit_dirty'] is False and last['checkpoint'] == 'D:/x/ckpt_step_481800'
    assert last['owner'] == 'the builder' and last['resource_class'] == 'BESIDE_TRAINING'
    assert isinstance(last['pid'], int) and isinstance(last['pid_created'], float)
    assert last['runner_pid'] == os.getpid() and last['runner'] == 'test_runner.py run'
    assert last['started_at'] <= last['ended_at'] and last['log'] == str(log)
    assert json.loads(seen.read_text()) == dict(row='probe', run=last['run_id']), 'the job carried the two marks'
    assert tr.sealed(row), 'the last run carries the seal its fields give'
    assert row['result']['verdict'] == 'PASS' and row['result']['run_id'] == last['run_id']


def test_the_four_outcomes_write_the_result_and_the_next_action(reg):
    assert run(reg, 'probe', 'raise SystemExit(1)')[:2] == (1, 'FAIL')
    fail = row_of(reg, 'probe')
    assert (fail['status'], fail['outcome'], fail['next_action']) == ('failed', 'FAIL', 'the cause goes to the builder')
    assert run(reg, 'look', 'raise SystemExit(7)')[:2] == (2, 'INVALID')
    invalid = row_of(reg, 'look')
    assert invalid['status'] == 'failed' and invalid['outcome'] == 'INVALID'
    assert invalid['next_action'].startswith('FOLLOW-UP OWED: this run ended INVALID (exit code 7')
    assert invalid['last_run']['follow_up_owed'] is True and invalid['last_run']['next_action_owner'] == 'a/test-branch'
    assert run(reg, 'third', 'import time; time.sleep(60)', timeout_s=1.5, poll_s=0.2)[:2] == (3, 'STOPPED')
    stopped = row_of(reg, 'third')
    assert stopped['status'] == 'failed' and stopped['outcome'] == 'STOPPED'
    assert 'its time limit of 1.5 s passed' in stopped['last_run']['judged_by']
    assert stopped['next_action'].startswith('FOLLOW-UP OWED: this run ended STOPPED')
    for row in (fail, invalid, stopped):
        assert tr.sealed(row) and row['result']['verdict'] == row['outcome']
    view = tr.dashboard_view(reg['tmp'], audit=False)
    by_id = {v['id']: v for g in view['groups'] for v in g['rows']}
    assert [by_id[n]['section'] for n in ('probe', 'look', 'third')] == ['FAILED', 'INVALID', 'STOPPED']
    assert by_id['look']['follow_up_owed'] and not by_id['probe']['follow_up_owed']
    assert sorted(e['row'] for e in view['errors'] if e['kind'] == 'follow-up owed') == ['look', 'third']
    assert tr.write_follow_up('look', 'run it again with the checkpoint path corrected', 'the builder', 'the builder',
                              path=reg['path']) is None
    view = tr.dashboard_view(reg['tmp'], audit=False)
    look = next(v for g in view['groups'] for v in g['rows'] if v['id'] == 'look')
    assert not look['follow_up_owed'] and look['next_action'].startswith('run it again with the checkpoint path')


def test_the_result_files_verdict_key_is_read_before_the_exit_code(reg):
    result = reg['tmp']/'out/verdict.json'
    result.parent.mkdir()
    reading = dict(name='tactile error', value=92, unit='of 100', meaning='100 = no better than no change',
                   baseline='100', statistically_clear='not judged', statistical_basis='', gof_material=tr.GOF_ANSWERS[2])
    write = ('import json,sys; json.dump(dict(verdict=sys.argv[2], readings=json.loads(sys.argv[3])), '
             'open(sys.argv[1], "w"))')
    path = write_registry(reg['tmp'], [new_row('probe', result_source=dict(file=str(result), keys=['verdict']))])
    reg = dict(reg, path=path)
    assert runner.run_command('probe', [PY, '-c', write, str(result), 'fail', json.dumps([reading, dict(name='x')])],
                              path=path, root=reg['tmp'], check_rule=False)[:2] == (1, 'FAIL'), 'exit 0, verdict fail'
    row = row_of(reg, 'probe')
    assert row['status'] == 'failed' and row['last_run']['exit_code'] == 0
    assert row['last_run']['judged_by'] == "key 'verdict' of %s" % result and row['last_run']['result_file'] == str(result)
    assert row['result']['readings'] == [reading] and len(row['result']['readings_refused']) == 1
    # a PASS in the file beside an exit code other than 0: the two disagree
    assert runner.run_command('probe', [PY, '-c', write + '; raise SystemExit(3)', str(result), 'PASS', '[]'],
                              path=path, root=reg['tmp'], check_rule=False)[:2] == (2, 'INVALID')
    assert 'says PASS but the job ended with exit code 3' in row_of(reg, 'probe')['last_run']['judged_by']
    # a result file left from an earlier run is not this run's verdict
    old = time.time() - 3600
    os.utime(result, (old, old))
    assert runner.run_command('probe', [PY, '-c', 'pass'], path=path, root=reg['tmp'], check_rule=False)[1] == 'PASS'
    assert 'was last written before this run started' in row_of(reg, 'probe')['last_run']['judged_by']
    # a row that requires the verdict key gets no PASS from an exit code
    path = write_registry(reg['tmp'], [new_row('probe', result_source=dict(file=str(result), keys=['verdict'],
                                                                         verdict_required=True))])
    assert runner.run_command('probe', [PY, '-c', 'pass'], path=path, root=reg['tmp'], check_rule=False)[1] == 'INVALID'


def test_read_outcome_by_exit_code(reg):
    row = new_row('probe')
    assert tr.read_outcome(row, 0)[0] == 'PASS' and tr.read_outcome(row, 1)[0] == 'FAIL'
    assert [tr.read_outcome(row, c)[0] for c in (2, 3, 5, 124, -9, 3221225477, None)] == ['INVALID'] * 7
    strict = new_row('probe', result_source=dict(file='x.json', keys=['verdict'], fail_exit_codes=[]))
    assert tr.read_outcome(strict, 1)[0] == 'INVALID'
    assert tr.read_outcome(row, 0, stopped='stopped: by hand')[0] == 'STOPPED'
    assert tr.read_outcome(row, 0, not_started='not started: refused')[0] == 'INVALID'


def test_a_command_that_cannot_start_ends_invalid(reg):
    code, state, _ = runner.run_command('probe', [str(reg['tmp']/'no_such_program.exe')], path=reg['path'],
                                        root=reg['tmp'], check_rule=False)
    assert (code, state) == (2, 'INVALID')
    assert row_of(reg, 'probe')['last_run']['judged_by'].startswith('not started: the command could not be started')


# ------------------------------------------------------------------------------------------ refusals before a start
def test_the_runner_refuses_before_execution(reg, capsys):
    marker = reg['tmp']/'ran.txt'
    cmd = [PY, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("x")', str(marker)]
    for name, expected in (('nobody', "no row in the test registry has id 'nobody'"),):
        assert runner.run_command(name, cmd, path=reg['path'], root=reg['tmp'], check_rule=False)[0] == 4
        assert expected in capsys.readouterr().out
    assert tr.block_row('probe', 'the checkpoint is being copied', 'the copy to D: ends', 'the builder',
                        path=reg['path']) is None
    assert runner.run_command('probe', cmd, path=reg['path'], root=reg['tmp'], check_rule=False)[0] == 4
    assert 'its row is blocked: the checkpoint is being copied' in capsys.readouterr().out
    assert tr.cancel_row('look', 'superseded by probe', 'the builder', path=reg['path']) is None
    assert runner.run_command('look', cmd, path=reg['path'], root=reg['tmp'], check_rule=False)[0] == 4
    assert not marker.exists(), 'no refused job ran'
    assert [row_of(reg, n)['status'] for n in ('probe', 'look')] == ['queued', 'cancelled']
    assert tr.unblock_row('probe', 'the builder', path=reg['path']) is None
    assert runner.run_command('probe', cmd, path=reg['path'], root=reg['tmp'], check_rule=False)[0] == 0
    assert marker.exists()


def test_a_row_that_needs_training_stopped_is_refused_while_a_production_trainer_runs(reg):
    path = write_registry(reg['tmp'], [new_row('probe', cost=dict(training_stopped=True, memory_gb=12,
                                                                   where='training pause', time_limit_s=60))])
    trainer = dict(pid=4242, ppid=1, created=1.0, name='python.exe',
                   cmdline=['python.exe', '-u', '-m', 'mimo_bridge.train_mimo', '--out',
                            'C:/r/Saved/mimo_bridge/experiments/H84_x_seg41o_1431'])
    why = runner.refusal('probe', path=path, check_rule=False, processes=[trainer])
    assert 'needs training stopped' in why and 'pid 4242, run H84_x_seg41o_1431' in why
    unit_test_trainer = dict(trainer, cmdline=['python.exe', '-m', 'mimo_bridge.train_mimo', '--out', 'D:/tmp/t1'])
    assert runner.refusal('probe', path=path, check_rule=False, processes=[unit_test_trainer]) is None


def test_one_run_of_a_row_at_a_time(reg):
    assert tr.job_started('probe', path=reg['path'], pid=os.getpid()) is None
    assert 'one run of a row at a time' in runner.refusal('probe', path=reg['path'], check_rule=False)


# ------------------------------------------------------------------------------------------ hand writes
def test_a_hand_write_may_not_take_the_runners_fields(reg):
    for fields in (dict(status='running'), dict(status='done'), dict(status='failed'), dict(outcome='PASS'),
                   dict(last_run=dict(state='PASS')), dict(next_action='deploy it')):
        with pytest.raises(tr.RegistryError, match='written only by a runner function'):
            tr.update_row('probe', fields, path=reg['path'])
    assert row_of(reg, 'probe')['status'] == 'queued' and 'outcome' not in row_of(reg, 'probe')
    assert tr.update_row('probe', dict(status='written', decision='waits for the checkpoint'), path=reg['path'])
    assert row_of(reg, 'probe')['decision'] == 'waits for the checkpoint'
    # while a run is live its status is the runner's
    assert tr.job_started('probe', path=reg['path'], pid=os.getpid()) is None
    assert 'its status is the runner\'s until the run ends' in tr.cancel_row('probe', 'changed my mind', 'x',
                                                                           path=reg['path'])
    assert row_of(reg, 'probe')['status'] == 'running'


def test_a_hand_edit_of_the_file_is_found_by_the_seal(reg):
    assert run(reg, 'probe', 'raise SystemExit(1)')[1] == 'FAIL'
    data = tr.load(reg['path'])
    row = next(r for r in data['rows'] if r['id'] == 'probe')
    row.update(status='done', outcome='PASS')                         # typed into the file, past every writer
    reg['path'].write_text(tr.dumps(data) + '\n', encoding='utf-8')
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=[], code_root=reg['tmp'])
    assert [m['kind'] for m in found['mismatches']] == ["status differs from the runner's record"]
    row['last_run']['state'] = 'PASS'                                  # and the last run edited to agree
    reg['path'].write_text(tr.dumps(data) + '\n', encoding='utf-8')
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=[], code_root=reg['tmp'])
    assert [m['kind'] for m in found['mismatches']] == ["status without the runner's record"]
    # a status typed into a row that never ran
    data = tr.load(reg['path'])
    next(r for r in data['rows'] if r['id'] == 'look')['status'] = 'done'
    reg['path'].write_text(tr.dumps(data) + '\n', encoding='utf-8')
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=[], code_root=reg['tmp'])
    assert any(m['row'] == 'look' and 'it has no last run at all' in m['text'] for m in found['mismatches'])


def test_the_baseline_accepts_rows_as_they_stood_and_any_other_unsealed_status_is_an_error(reg, monkeypatch):
    """The baseline's own list of rows as they stood is accepted (listed as unverified); any other runner status
    without the seal is an error whatever date its last run carries (the dashboard review of Oct 5, 2026, finding
    F5)."""
    old = dict(started='Oct 4, 2026 4:02 AM PDT', ended='Oct 4, 2026 4:10 AM PDT', exit_code=0, log='x.log',
               outcome='exit code 0', runner='run_in_training_pause.py', receipt='r.json')
    path = write_registry(reg['tmp'], [complete_row('legacy', status='done', last_run=old),
                                       complete_row('typed', status='failed')])
    base = runner.make_baseline(tr.load(path))
    assert sorted(base['rows']) == ['legacy', 'typed']
    target = runner.baseline_path(reg['tmp'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(base), encoding='utf-8')
    found = runner.audit(path=path, root=reg['tmp'], processes=[], code_root=reg['tmp'])
    assert found['ok'] and [u['row'] for u in found['unverified']] == ['legacy', 'typed']
    assert 'as it stood in the baseline' in found['unverified'][0]['text']
    # a row finished later by a runner of the older kind: an error, with or without seal_required_from, whatever the
    # date its last run carries
    for when in ('Oct 5, 2026 3:00 AM PDT', 'Oct 1, 2026 3:00 AM PDT'):
        path = write_registry(reg['tmp'], [complete_row('legacy', status='done', last_run=old),
                                           complete_row('typed', status='failed'),
                                           complete_row('later', status='done', last_run=dict(old, ended=when))])
        for required in (None, 'Oct 5, 2026 2:00 AM PDT'):
            target.write_text(json.dumps(dict(base, seal_required_from=required)), encoding='utf-8')
            found = runner.audit(path=path, root=reg['tmp'], processes=[], code_root=reg['tmp'])
            assert [(m['kind'], m['row']) for m in found['mismatches']] == [
                ("status without the runner's record", 'later')], (when, required)
            assert [u['row'] for u in found['unverified']] == ['legacy', 'typed']


# ------------------------------------------------------------------------------------------ the audit, both directions
def spawn(tmp, env=None, script=None, seconds=120):
    """A real, short-lived Python process: `script` (a path it runs) or a sleep, with `env` added to its environment.
    The runner's marks this test process may carry (when these tests run as a registered test) are not passed on:
    a planted process carries only the marks its test gives it."""
    code = 'import time; time.sleep(%d)' % seconds
    if script is not None:
        Path(script).parent.mkdir(parents=True, exist_ok=True)
        Path(script).write_text(code, encoding='utf-8')
    base = {k: v for k, v in os.environ.items() if k not in (runner.ENV_ROW, runner.ENV_RUN, runner.ENV_DEV)}
    return subprocess.Popen([PY, str(script)] if script is not None else [PY, '-c', code],
                            env=dict(base, **(env or {})), stdin=subprocess.DEVNULL)


def tree_of(proc):
    """The scan's entries for a started process and the processes under it (the Windows venv launcher's child)."""
    import psutil
    deadline = time.time() + 20
    while time.time() < deadline:                 # the launcher's child appears a short time after the launcher
        try:
            ids = {proc.pid} | {c.pid for c in psutil.Process(proc.pid).children(recursive=True)}
        except psutil.Error:
            ids = {proc.pid}
        found = [p for p in runner.scan_processes() if p['pid'] in ids]
        if found and all(p['cmdline'] for p in found):
            return found
        time.sleep(0.1)
    return found


@pytest.fixture
def planted(reg):
    started = []

    def start(**kw):
        proc = spawn(reg['tmp'], **kw)
        started.append(proc)
        return proc
    yield start
    for proc in started:
        runner.end_tree(proc.pid)


def kinds(found):
    return sorted(m['kind'] for m in found['mismatches'])


def test_a_planted_orphan_process_carrying_a_rows_mark_is_detected(reg, planted):
    proc = planted(env={runner.ENV_ROW: 'probe', runner.ENV_RUN: 'r1'})
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=tree_of(proc), code_root=reg['tmp'])
    assert kinds(found) == ['test process without a running row']
    assert 'runs test probe (run r1), and its row says queued' in found['mismatches'][0]['text']
    ghost = planted(env={runner.ENV_ROW: 'never_written', runner.ENV_RUN: 'r2'})
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=tree_of(ghost), code_root=reg['tmp'])
    assert kinds(found) == ['test process without a row']


def test_a_planted_test_process_started_without_the_runner_is_detected(reg, planted):
    proc = planted(script=reg['tmp']/'docs/gain_of_function_v4/probe_planted.py')
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=tree_of(proc), code_root=reg['tmp'])
    assert kinds(found) == ['test process without a row']
    assert 'a script of a test folder (probe_planted.py)' in found['mismatches'][0]['text']
    # a watch program in the same folder, and a plain command, are not test processes
    for quiet in (planted(script=reg['tmp']/'docs/gain_of_function_v4/coordinator/memory_watch_loop.py'), planted()):
        assert runner.audit(path=reg['path'], root=reg['tmp'], processes=tree_of(quiet), code_root=reg['tmp'])['ok']


def test_a_planted_orphan_running_row_is_detected(reg, planted):
    proc = planted()
    assert tr.job_started('probe', path=reg['path'], pid=proc.pid, runner_pid=proc.pid) is None
    live = runner.audit(path=reg['path'], root=reg['tmp'], processes=tree_of(proc), code_root=reg['tmp'])
    assert live['ok'] and live['running_rows'][0]['alive'] and 'its job process' in live['running_rows'][0]['how']
    runner.end_tree(proc.pid)
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=[], code_root=reg['tmp'])
    assert kinds(found) == ['running row without a live process']
    assert 'is gone: the run ended without its end being written' in found['mismatches'][0]['text']
    # the same process id reused by another process is not the row's process
    reused = [dict(pid=proc.pid, ppid=1, created=time.time() + 500, name='python.exe', cmdline=['python.exe'])]
    assert not runner.audit(path=reg['path'], root=reg['tmp'], processes=reused, marks=lambda pid: {},
                            code_root=reg['tmp'])['ok']
    # closing it writes STOPPED through the runner's own function
    assert runner.close_run('probe', 'the builder', 'its process was ended by the test', path=reg['path'],
                            root=reg['tmp'], processes=[]) == 'STOPPED'
    row = row_of(reg, 'probe')
    assert row['status'] == 'failed' and row['outcome'] == 'STOPPED' and tr.sealed(row)
    assert runner.audit(path=reg['path'], root=reg['tmp'], processes=[], code_root=reg['tmp'])['ok']


def test_a_run_in_progress_is_matched_by_its_mark_and_passes_the_audit(reg):
    import threading
    release = reg['tmp']/'release'
    result = {}

    def go():
        result['r'] = runner.run_command(
            'probe', [PY, '-c', 'import pathlib,sys,time\nwhile not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.1)',
                      str(release)], path=reg['path'], root=reg['tmp'], check_rule=False, poll_s=0.2)
    thread = threading.Thread(target=go)
    thread.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not (row_of(reg, 'probe').get('last_run') or {}).get('pid'):
            time.sleep(0.1)
        last = row_of(reg, 'probe')['last_run']
        import psutil
        ids = {last['pid']} | {c.pid for c in psutil.Process(last['pid']).children(recursive=True)}
        scan = [p for p in runner.scan_processes() if p['pid'] in ids]
        found = runner.audit(path=reg['path'], root=reg['tmp'], processes=scan, code_root=reg['tmp'])
        assert found['ok'], found['mismatches']
        assert [t['row'] for t in found['test_processes']] == ['probe'] and found['running_rows'][0]['alive']
    finally:
        release.write_text('go')
        thread.join(60)
    assert result['r'][1] == 'PASS'


def test_a_production_segment_row_is_running_only_while_the_trainer_of_its_run_is_alive(reg):
    stale = complete_row('segment', status='running', last_run=dict(
        started='Oct 4, 2026 4:02 AM PDT', ended=None, exit_code=None, outcome='running',
        log='Saved/mimo_bridge/experiments/H84_x_seg41m_0402/trainer.log', runner='relaunch_h84_seg41_2026-09-29.py'))
    path = write_registry(reg['tmp'], [stale])
    # the row as it stood before the seal existed: in the baseline's own list (finding F5 makes any other unsealed
    # runner status an error)
    target = runner.baseline_path(reg['tmp'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runner.make_baseline(tr.load(path))), encoding='utf-8')
    other = dict(pid=10, ppid=1, created=5.0, name='python.exe',
                 cmdline=['python.exe', '-m', 'mimo_bridge.train_mimo', '--out',
                          'C:/r/Saved/mimo_bridge/experiments/H84_x_seg41o_1431'])
    # another run's trainer is alive: the segment's own run is not training, and nothing holds training
    found = runner.audit(path=path, root=reg['tmp'], processes=[other], marks=lambda pid: {}, code_root=reg['tmp'])
    assert [(m['kind'], m['row']) for m in found['mismatches']] == [('running row without a live process', 'segment')]
    assert 'no trainer process of that run is alive; the trainer alive is of run H84_x_seg41o_1431' in \
        found['mismatches'][0]['text'] and 'H84_x_seg41m_0402' in found['mismatches'][0]['text']
    assert found['running_rows'][0]['state'] == 'LOST'
    # no production trainer is alive at all: HELD, with the hold's reason from the supervisor's control file
    saved = reg['tmp']/'Saved/mimo_bridge'
    (saved/'experiments/H84_x_seg41o_1431').mkdir(parents=True)
    (saved/'supervisor_control.json').write_text(json.dumps(dict(cmd='pause', reason='hold for the review bench')))
    (saved/'training_continuation.json').write_text(json.dumps(dict(
        active_run=str(saved/'experiments/H84_x_seg41o_1431'))))
    (saved/'experiments/H84_x_seg41o_1431/stage_done.json').write_text(json.dumps(dict(
        steps=481800, finished_at=tr.parse_clock('Oct 4, 2026 4:23 PM PDT'))))
    # the row's run (seg41m_0402) is not the run training is held in (seg41o_1431): it ended at a relaunch before the
    # hold, so it reads LOST (the dashboard review of Oct 5, 2026, finding L3)
    found = runner.audit(path=path, root=reg['tmp'], processes=[], marks=lambda pid: {}, code_root=reg['tmp'])
    assert [(m['kind'], m['row']) for m in found['mismatches']] == [('running row without a live process', 'segment')]
    assert found['running_rows'][0]['state'] == 'LOST'
    assert ('training is held in run H84_x_seg41o_1431: run H84_x_seg41m_0402 ended at a relaunch before the hold'
            in found['mismatches'][0]['text'])
    # the segment of the held run: HELD, with the hold's reason from the supervisor's control file
    path = write_registry(reg['tmp'], [dict(stale, last_run=dict(
        stale['last_run'], log='Saved/mimo_bridge/experiments/H84_x_seg41o_1431/trainer.log'))])
    target.write_text(json.dumps(runner.make_baseline(tr.load(path))), encoding='utf-8')
    found = runner.audit(path=path, root=reg['tmp'], processes=[], marks=lambda pid: {}, code_root=reg['tmp'])
    assert [(m['kind'], m['row']) for m in found['mismatches']] == [('production segment held', 'segment')]
    held = found['running_rows'][0]
    assert held['state'] == 'HELD' and held['alive'] is False and held['hold']['reason'] == 'hold for the review bench'
    assert held['held_text'] == ('no production trainer process is alive. Training stopped at model step 481,800 of run '
                                 'H84_x_seg41o_1431 (Oct 4, 2026 4:23 PM PDT). Reason on record '
                                 '(supervisor_control.json): hold for the review bench.')
    assert 'HELD: ' + held['held_text'] in found['mismatches'][0]['text']
    own = dict(other, cmdline=['python.exe', '-m', 'mimo_bridge.train_mimo', '--out',
                               'C:/r/Saved/mimo_bridge/experiments/H84_x_seg41o_1431'])
    found = runner.audit(path=path, root=reg['tmp'], processes=[own], marks=lambda pid: {}, code_root=reg['tmp'])
    assert found['mismatches'] == [] and found['running_rows'][0]['state'] == 'RUNNING'
    assert found['running_rows'][0]['how'] == 'the trainer of run H84_x_seg41o_1431 is alive (pid 10)'


def test_a_segment_start_and_its_stated_end_go_through_the_runners_functions(reg):
    me = os.getpid()
    run_ = runner.segment_start('probe', 'C:/r/Saved/mimo_bridge/experiments/H84_x_seg41p_0900', me, path=reg['path'],
                                root=reg['tmp'], checkpoint='ckpt_step_481800', owner='the coordinator',
                                check_rule=False)
    row = row_of(reg, 'probe')
    assert row['status'] == 'running' and row['last_run']['resource_class'] == 'PRODUCTION_SEGMENT'
    assert row['last_run']['production_run'].endswith('H84_x_seg41p_0900') and row['last_run']['pid'] == me
    assert run_.run_id == row['last_run']['run_id']
    with pytest.raises(runner.Refused, match='needs its basis'):
        runner.segment_finish('probe', 'STOPPED', '', 'the coordinator', path=reg['path'], root=reg['tmp'])
    assert runner.segment_finish('probe', 'STOPPED', 'training held at checkpoint 481,800 by the pause of Oct 4',
                                 'the coordinator', path=reg['path'], root=reg['tmp']) == 'STOPPED'
    row = row_of(reg, 'probe')
    assert row['outcome'] == 'STOPPED' and tr.sealed(row)
    assert row['last_run']['judged_by'].startswith('stated by the coordinator: training held at checkpoint 481,800')


def test_processes_under_the_older_runners_and_the_scheduler_are_matched_to_their_rows(reg):
    def proc(pid, ppid, *cmd):
        return dict(pid=pid, ppid=ppid, created=float(pid), name='python.exe', cmdline=list(cmd))
    old_runner = proc(100, 1, 'python.exe', 'C:/wt/x/docs/gain_of_function_v4/prediction_heads_dead/heads_registry.py',
                      'run', 'probe', 'D:/s/probe.log', '--', 'python.exe', 'cpu_checks.py')
    job = proc(101, 100, 'python.exe', 'cpu_checks.py')
    scheduler = proc(200, 1, 'python.exe', '-u', '-m', 'mimo_bridge.tools.ability_rechecks', '--root', 'C:/r')
    check = proc(201, 200, 'python.exe', '-u', 'C:/r/docs/gain_of_function_v3/probe_saved_gaze.py', '--checkpoint',
                 'C:/c', '--out', 'C:/r/Saved/mimo_bridge/ability_rechecks/look_481800_attempt1_1791149651258960700')
    capped = proc(300, 1, 'python.exe', 'C:/wt/y/docs/gain_of_function_v4/dashboard_recovery/dashboard_registry.py',
                  'capped', '--', 'python.exe', '-m', 'pytest', 'tests')
    dev_child = proc(301, 300, 'python.exe', '-m', 'pytest', 'tests')
    scan = [old_runner, job, scheduler, check, capped, dev_child]
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=scan, marks=lambda pid: {}, code_root=reg['tmp'])
    assert sorted(m['row'] for m in found['mismatches']) == ['look', 'probe'], 'both rows say queued'
    assert [d['pid'] for d in found['development_runs']] == [300]
    assert tr.job_started('probe', path=reg['path'], runner='heads_registry.py run', runner_pid=999999) is None
    assert tr.job_started('look', path=reg['path'], runner='ability_rechecks.py', runner_pid=999998) is None
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=scan, marks=lambda pid: {}, code_root=reg['tmp'])
    assert found['ok'], found['mismatches']
    assert {r['id']: r['alive'] for r in found['running_rows']} == dict(probe=True, look=True)


# ------------------------------------------------------------------------------------------ the page's sections
def test_the_sections_are_computed_and_only_queued_is_open(reg):
    rows = [new_row('q'), new_row('w', status='written'), new_row('b', blocked=dict(reason='waits', waits_on='x')),
            new_row('c', status='cancelled', cancelled=dict(reason='superseded')),
            complete_row('old_done', status='done'), complete_row('old_failed', status='failed'),
            new_row('incomplete', question='')]
    path = write_registry(reg['tmp'], rows)
    assert tr.job_started('q', path=path, pid=os.getpid()) is None
    assert tr.finish_job('q', exit_code=0, path=path)[1] == 'PASS'
    assert tr.queue_row('q', 'the builder', path=path) is None            # queued again after a PASS
    view = tr.dashboard_view(reg['tmp'], audit=False)
    assert [s['key'] for s in view['sections']] == list(tr.SECTIONS) == view['section_order']
    assert [s['key'] for s in view['sections'] if s['open_by_default']] == ['QUEUED']
    by_id = {v['id']: v for g in view['groups'] for v in g['rows']}
    assert {k: v['section'] for k, v in by_id.items()} == dict(
        q='QUEUED', w='QUEUED', b='BLOCKED', c='STOPPED', old_done='DONE', old_failed='FAILED', incomplete='BLOCKED')
    assert by_id['q']['card']['state'] == 'QUEUED (last run: PASS)' and by_id['q']['outcome'] == 'PASS'
    assert by_id['b']['section_reason'] == 'Blocked: waits. It waits on: x.'
    assert by_id['incomplete']['section_reason'] == 'Its row is incomplete: question is empty.'
    assert by_id['c']['section_reason'] == 'Cancelled by hand: superseded.'
    counts = {s['key']: s['count'] for s in view['sections']}
    assert counts == dict(QUEUED=2, RUNNING=0, BLOCKED=2, FAILED=1, INVALID=0, STOPPED=1, DONE=1)
    assert sorted(set(by_id['q']['card'])) == ['age', 'component', 'conclusion', 'direct_target', 'next_action',
                                               'question', 'state']
    names = {n['name']: n for n in view['names_used']}
    assert set(names) >= {'QUEUED', 'BLOCKED', 'FAILED', 'STOPPED', 'DONE', 'PASS', 'TEST_REGISTRY'}
    assert 'RUNNING' not in names and 'INVALID' not in names, 'only the names this page shows'
    full = tr.dashboard_view(reg['tmp'], audit=False, rows_in='sections')
    assert 'groups' not in full and sorted(v['id'] for v in full['sections'][0]['rows']) == ['q', 'w']


def test_a_readings_two_judgements_are_printed_and_a_malformed_reading_is_named():
    reading = dict(name='tactile error', value=92, unit='of 100', meaning='100 = no better than no change',
                   baseline='100 (no change)', statistically_clear='yes', statistical_basis='10 reports, interval 90 to 94',
                   gof_material=tr.GOF_ANSWERS[2])
    assert tr.reading_problems(reading) == []
    assert tr.reading_lines(reading) == [
        'tactile error: 92 of 100 (100 = no better than no change)', 'Baseline: 100 (no change)',
        'Statistically clear: yes (10 reports, interval 90 to 94)',
        'Large enough to matter for gain of function: GOF threshold not yet defined']
    bad = dict(reading, statistically_clear='clearly', gof_material='big', statistical_basis='')
    assert tr.reading_problems(bad) == ['statistically clear is not one of yes, no, not judged',
                                        'large enough to matter for gain of function is not one of yes, no, GOF '
                                        'threshold not yet defined']
    assert 'statistically clear says yes without its basis' in tr.reading_problems(dict(reading, statistical_basis=''))


# ------------------------------------------------------------------------------------------ the development run
def test_a_development_run_is_marked_and_logged_and_writes_no_row(reg):
    seen = reg['tmp']/'dev.json'
    before = reg['path'].read_bytes()
    code = runner.dev_run([PY, '-c', 'import json,os,sys; json.dump(dict(dev=os.environ.get("UEMA_DEV_RUN"), '
                                     'row=os.environ.get("UEMA_TEST_ROW")), open(sys.argv[1], "w"))', str(seen)],
                          'the builder', 'unit tests while building')
    assert code == 0 and json.loads(seen.read_text()) == dict(dev='the builder|unit tests while building', row=None)
    assert reg['path'].read_bytes() == before
    lines = [json.loads(x) for x in (reg['tmp']/'runs/development_runs.jsonl').read_text().splitlines()]
    assert [x['event'] for x in lines] == ['started', 'ended'] and lines[1]['exit_code'] == 0
    assert runner.dev_run([PY, '-c', 'pass'], '', 'no owner') == 4


def test_the_command_line_runs_a_row_and_audits(reg, capsys, monkeypatch):
    from mimo_bridge.tools import defect_register as dr
    monkeypatch.setattr(runner, 'scan_processes', lambda: [])
    monkeypatch.setattr(dr, 'load', lambda root=None, path=None: dict(defects=[]))   # the 24-hour rule holds nothing
    assert runner.main(['run', 'probe', '--registry', str(reg['path']), '--root', str(reg['tmp']), '--owner', 'me',
                        '--', PY, '-c', 'print(1)']) == 0
    row = row_of(reg, 'probe')
    assert (row['status'], row['outcome'], row['last_run']['owner']) == ('done', 'PASS', 'me') and tr.sealed(row)
    assert 'job probe: PASS (exit code 0)' in capsys.readouterr().out
    assert runner.main(['audit', '--registry', str(reg['path']), '--root', str(reg['tmp'])]) == 0
    assert capsys.readouterr().out.startswith('0 running rows, each with a live process; 0 live test processes, each '
                                              'with its running row. A test process here is')
    assert runner.main(['run', 'nobody', '--registry', str(reg['path']), '--root', str(reg['tmp']), '--', PY, '-c',
                        'pass']) == runner.EXIT_REFUSED
    assert "no row in the test registry has id 'nobody'" in capsys.readouterr().out


def test_the_runners_own_processes_are_never_test_processes_and_a_wrapper_is_judged_by_its_own_arguments(reg):
    def proc(pid, ppid, *cmd):
        return dict(pid=pid, ppid=ppid, created=float(pid), name='python.exe', cmdline=list(cmd))
    scan = [proc(400, 1, 'python.exe', '-m', 'mimo_bridge.tools.test_runner', 'run', 'probe', '--', 'python.exe', '-m',
                 'pytest', 'tests'),
            proc(401, 400, 'python.exe', '-m', 'pytest', 'tests'),
            proc(500, 1, 'python.exe', '-m', 'mimo_bridge.tools.test_runner', 'dev', '--owner', 'me', '--purpose', 'x',
                 '--', 'python.exe', '-m', 'pytest', 't'),
            proc(501, 500, 'python.exe', '-m', 'pytest', 't'),
            proc(600, 1, 'python.exe', 'C:/x/wrapper.py', '--', 'python.exe', '-m', 'pytest', 't'),
            proc(601, 600, 'python.exe', '-m', 'pytest', 't')]
    marks = {401: {runner.ENV_ROW: 'probe', runner.ENV_RUN: 'r1'}, 501: {runner.ENV_DEV: 'me|x'}}
    assert tr.job_started('probe', path=reg['path'], run_id='r1', runner_pid=400) is None
    found = runner.audit(path=reg['path'], root=reg['tmp'], processes=scan, marks=lambda pid: marks.get(pid, {}),
                         code_root=reg['tmp'])
    assert [(m['kind'], m['pid']) for m in found['mismatches']] == [('test process without a row', 601)]
    assert [d['pid'] for d in found['development_runs']] == [501]
    assert sorted(t['pid'] for t in found['test_processes']) == [401, 601]
    assert runner.is_test_runner(scan[0]) and runner.is_test_runner(scan[2]) and not runner.is_test_runner(scan[4])
    assert runner.signature_of(scan[4]) is None and runner.signature_of(scan[5]) == 'pytest'
