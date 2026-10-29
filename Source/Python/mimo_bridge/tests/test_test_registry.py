# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The test registry and its enforcement points (the style guide "NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE", operator
2026-10-02: "This entire testing process seems to be spiraling out of control").

mimo_bridge/tools/test_registry.py judges a row, writes a job's status and last_run back, and builds the Tests page.
The enforcement points use it: the pause runner (docs/gain_of_function_v4/run_in_training_pause.py, --check included),
the held-pause runner (docs/gain_of_function_v4/coordinator/run_jobs_in_held_pause.py), the ability scheduler
(tools/ability_rechecks.py) and the dashboard's GET /api/tests. Every registry here is a temporary file. Planted
defects - a missing row, a wrong id, an empty field, a missing field, an empty list, a cost without memory, a result
source without its file, an unknown status, a cancelled row, two rows with one id, an unreadable file - must each be
caught. No world, model or trainer is involved: the runners' trainers and memory guard are the simulated ones of
test_run_in_training_pause.py, and the scheduler's child is the bounded script of test_ability_rechecks.py.
"""
import importlib.util
import json
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from mimo_bridge.tools import ability_rechecks as worker
from mimo_bridge.tools import test_registry as tr
from test_ability_rechecks import tick, tiny_worker, until  # noqa: F401 (fixture)
from test_run_in_training_pause import outside, runner, write  # noqa: F401 (fixtures)
from test_run_in_training_pause import JOB_NAMES, complete_row, write_registry

REPO = Path(__file__).resolve().parents[4]
SCHEDULER_CHECK = worker.registry_refusals          # the real check; the scheduler fixtures replace it
CLOCK = re.compile(r'^[A-Z][a-z]{2} \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M P[DS]T$')


def _without(field):
    return lambda rows: [{k: v for k, v in r.items() if k != field} if r['id'] == 'probe' else r for r in rows]


def _changed(**changes):
    return lambda rows: [dict(r, **changes) if r['id'] == 'probe' else r for r in rows]


# (defect, how it changes the registry's rows, what the refusal must say)
PLANTED = [
    ('missing row', lambda rows: [r for r in rows if r['id'] != 'probe'], "no row in the test registry has id 'probe'"),
    ('wrong id', lambda rows: [dict(r, id='probe_a') if r['id'] == 'probe' else r for r in rows],
     "a row has the near id 'probe_a'"),
    ('empty field', _changed(question='   '), 'question is empty (question)'),
    ('missing field', _without('steps'), 'steps is missing (steps)'),
    ('empty list', _changed(inputs=['', ' ']), 'inputs is empty (inputs)'),
    ('cost without memory', _changed(cost=dict(training_stopped=True, where='training pause')),
     'cost gives no memory in GB (cost.memory_gb)'),
    ('result source without its file', _changed(result_source=dict(keys=['passed'])),
     'the result source names no file (result_source.file)'),
    ('unknown status', _changed(status='pending'), "status 'pending' is not one of"),
    ('cancelled row', _changed(status='cancelled'), 'says cancelled'),
    ('two rows with one id', lambda rows: rows + [complete_row('probe')], "2 rows in the test registry have id 'probe'"),
]
RUNNER_PLANTED = [p for p in PLANTED if p[0] in ('missing row', 'wrong id', 'empty field')]


def registry_with(tmp, defect=None):
    rows = [complete_row(name) for name in JOB_NAMES]
    return write_registry(tmp, defect(rows) if defect else rows)


# ------------------------------------------------------------------------------------------ the shared module
@pytest.mark.parametrize('name,defect,expected', PLANTED, ids=[p[0] for p in PLANTED])
def test_every_planted_defect_refuses_its_job_and_only_its_job(tmp_path, name, defect, expected):
    path = registry_with(tmp_path, defect)
    assert tr.refusal('look', path=path) is None, 'a complete row beside the defect still runs'
    why = tr.refusal('probe', path=path)
    assert why is not None and expected in why
    assert tr.refusals(['probe', 'look'], path=path) == {'probe': why}


def test_a_complete_row_runs_and_an_unreadable_or_missing_registry_refuses_every_job(tmp_path):
    path = registry_with(tmp_path)
    assert tr.refusals(['probe', 'look'], path=path) == {}
    path.write_text('{"rows": [', encoding='utf-8')                     # a half-written file
    assert all('cannot be read' in why for why in tr.refusals(['probe', 'look'], path=path).values())
    path.unlink()
    refused = tr.refusals(['probe'], path=path)
    assert 'does not exist' in refused['probe']


def test_the_registry_declares_its_own_required_fields(tmp_path):
    rows = [{k: v for k, v in complete_row('probe').items() if k != 'measures'}]
    assert 'measures is missing' in tr.refusal('probe', path=write_registry(tmp_path, rows))
    lighter = [f for f in tr.DEFAULT_REQUIRED if f != 'measures']
    assert tr.refusal('probe', path=write_registry(tmp_path, rows, required=lighter)) is None
    without_id = [f for f in tr.DEFAULT_REQUIRED if f != 'id']
    assert 'id' in tr.required_fields(dict(required=without_id)), 'jobs are matched by id, always required'


def test_a_write_back_changes_only_its_fields_and_keeps_another_writers_edit(tmp_path, monkeypatch):
    path = registry_with(tmp_path)
    before = tr.load(path)
    real_load = tr.load
    edits = []

    def load_while_another_writer_edits(p=None, **kw):
        data = real_load(p, **kw)
        if not edits:                       # another writer replaces the file right after this writer's first read
            edits.append(True)
            theirs = real_load(p)
            theirs['rows'][1]['decision'] = 'written by another writer'
            Path(p).write_text(tr.dumps(theirs) + '\n', encoding='utf-8')
        return data

    monkeypatch.setattr(tr, 'load', load_while_another_writer_edits)
    assert tr.update_row('probe', dict(status='written', decision='decided by the first writer'), path=path)
    after = real_load(path)
    assert after['rows'][1]['decision'] == 'written by another writer', 'the other writer\'s edit is kept'
    assert after['rows'][0]['status'] == 'written' and after['rows'][0]['decision'] == 'decided by the first writer'
    for key, value in before['rows'][0].items():
        if key not in ('status', 'decision'):
            assert after['rows'][0][key] == value, key
    assert after['rows'][2:] == before['rows'][2:] and after['required'] == before['required']
    assert not list(tmp_path.rglob('*.lock')) and not list(tmp_path.rglob('*.tmp')), 'no lock or temporary file left'
    assert tr.update_row('nobody', dict(status='queued'), path=path) is False


def test_job_started_and_finished_write_status_and_last_run_and_never_raise(tmp_path):
    path = registry_with(tmp_path)
    t0 = time.time()
    assert tr.job_started('probe', started=t0, log='C:/x/probe.log', runner='r.py', receipt='C:/x/receipt.json',
                          path=path) is None
    row = next(r for r in tr.load(path)['rows'] if r['id'] == 'probe')
    assert row['status'] == 'running' and CLOCK.match(row['last_run']['started']) and row['last_run']['ended'] is None
    assert tr.job_finished('probe', exit_code=0, started=t0, ended=t0 + 61, log='C:/x/probe.log', runner='r.py',
                           path=path) is None
    row = next(r for r in tr.load(path)['rows'] if r['id'] == 'probe')
    assert row['status'] == 'done' and row['last_run']['exit_code'] == 0 and CLOCK.match(row['last_run']['ended'])
    assert tr.job_finished('look', exit_code=3, path=path) is None
    assert next(r for r in tr.load(path)['rows'] if r['id'] == 'look')['status'] == 'failed'
    assert 'no single row' in tr.job_started('nobody', path=path)
    path.write_text('not json', encoding='utf-8')
    assert 'cannot be read' in tr.job_finished('probe', exit_code=0, path=path)


def test_clock_times_are_am_pm_pacific():
    assert tr.clock(1790986080) == 'Oct 2, 2026 5:08 PM PDT'
    assert tr.clock(1767312000) == 'Jan 1, 2026 4:00 PM PST'
    assert tr.parse_clock('Oct 2, 2026 5:08 PM PDT') == 1790986080
    assert tr.parse_clock('Oct 2, 2026, 5:08 PM PDT') == 1790986080
    assert tr.parse_clock('not a time') is None


def test_the_writer_keeps_the_registry_layout():
    """The production file's layout (Oct 2, 2026 5:27 PM): one-space indent; cost, result_source, code and short lists
    on one line; sentences one per line."""
    text = ('{\n "required": ["id", "status"],\n "rows": [\n  {\n   "id": "a",\n   "steps": [\n'
            '    "Copy the hierarchy file and load it. Load the checkpoint. Encode the 2,000 rows.",\n'
            '    "Per body: build a fresh world from the checkpoint\'s episode."\n   ],\n'
            '   "cost": {"training_stopped": true, "memory_gb": 12, "where": "training pause"},\n'
            '   "result_source": {"file": "D:/x/verdict.json", "keys": ["kinds", "lines"]},\n'
            '   "status": "done",\n   "result": null,\n'
            '   "code": ["docs/gain_of_function_v4/coordinator/latent_full_body_2026-10-01.py", "b.py"]\n  }\n ]\n}\n')
    assert tr.dumps(json.loads(text)) + '\n' == text


# ------------------------------------------------------------------------------------------ the pause runner
def jobs_file(fixture, *jobs, check=False):
    path = fixture['tmp']/'jobs_registry.json'
    write(path, [dict(dict(cwd='.', need_gb=2, timeout_s=60), **j) for j in jobs])
    sys.argv[:] = ['run_in_training_pause.py', '--jobs', str(path)] + (['--check'] if check else [])
    return path


def rows_of(fixture):
    return {r['id']: r for r in tr.load(fixture['registry'])['rows']}


@pytest.mark.parametrize('name,defect,expected', RUNNER_PLANTED, ids=[p[0] for p in RUNNER_PLANTED])
def test_the_check_refuses_a_jobs_file_with_a_planted_defect(outside, capsys, name, defect, expected):
    registry_with(outside['tmp'], defect)
    jobs_file(outside, dict(name='look', cmd=['PY', '-c', 'print(1)']), dict(name='probe', cmd=['PY', '-c', 'print(2)']),
              check=True)
    control = (outside['saved']/'supervisor_control.json').read_bytes()
    assert outside['module'].main() == 1
    printed = json.loads(capsys.readouterr().out)
    assert list(printed['registry_refusals']) == ['probe'] and expected in printed['registry_refusals']['probe']
    assert not (outside['tmp']/'pauses').exists(), '--check writes nothing'
    assert (outside['saved']/'supervisor_control.json').read_bytes() == control


@pytest.mark.parametrize('name,defect,expected', RUNNER_PLANTED, ids=[p[0] for p in RUNNER_PLANTED])
def test_a_planted_defect_refuses_the_jobs_file_with_a_receipt_before_training_stops(outside, name, defect, expected):
    registry_with(outside['tmp'], defect)
    marker = outside['tmp']/'ran.txt'
    path = jobs_file(outside, dict(name='look', cmd=['PY', '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("x")',
                                                     str(marker)]),
                     dict(name='probe', cmd=['PY', '-c', 'print(2)']))
    control = (outside['saved']/'supervisor_control.json').read_bytes()
    assert outside['module'].main() == 1
    assert (outside['saved']/'supervisor_control.json').read_bytes() == control, 'no pause was written'
    assert not (outside['run']/'control.json').exists(), 'no save was asked of the trainer'
    assert not marker.exists(), 'no job ran'
    receipt = json.loads(next((outside['tmp']/'pauses').glob('*_refused/receipt.json')).read_text(encoding='utf-8'))
    assert receipt['refused'] == 'test registry' and receipt['jobs_files'] == [str(path)]
    look, probe = receipt['jobs']
    assert probe['status'] == 'refused by the test registry' and expected in probe['reason']
    assert look['status'].startswith('not run: another job')
    if name != 'missing row' and name != 'wrong id':
        assert rows_of(outside)['probe']['status'] == 'queued', 'a refused row is not written to'


def test_jobs_with_complete_rows_run_and_get_their_status_and_last_run_written(outside):
    before = rows_of(outside)
    jobs_file(outside, dict(name='probe', cmd=['PY', '-c', 'print("ran")']),
              dict(name='look', cmd=['PY', '-c', 'raise SystemExit(3)']))
    assert outside['module'].main() == 0
    rows = rows_of(outside)
    probe, look = rows['probe'], rows['look']
    assert probe['status'] == 'done' and probe['last_run']['exit_code'] == 0
    assert CLOCK.match(probe['last_run']['started']) and CLOCK.match(probe['last_run']['ended'])
    assert probe['last_run']['log'].endswith('probe.log') and probe['last_run']['runner'] == 'run_in_training_pause.py'
    assert probe['last_run']['receipt'].endswith('receipt.json')
    assert look['status'] == 'failed' and look['last_run']['exit_code'] == 3
    # the fields of Oct 5, 2026: the outcome, the next action and the result are the runner's to write too
    assert (probe['outcome'], probe['next_action'], probe['result']['verdict']) == ('PASS', 'nothing changes', 'PASS')
    assert look['outcome'] == 'INVALID' and look['next_action'].startswith('FOLLOW-UP OWED: this run ended INVALID')
    assert tr.sealed(probe) and tr.sealed(look) and probe['last_run']['run_id'] != look['last_run']['run_id']
    assert isinstance(probe['last_run']['pid'], int) and probe['last_run']['runner_pid'] is not None
    assert probe['last_run']['resource_class'] == 'TEST_WINDOW'
    assert probe['last_run']['checkpoint'].endswith('ckpt_step_411350')
    for name in rows:
        for key, value in before[name].items():
            if key not in ('status', 'last_run', 'outcome', 'next_action', 'follow_up', 'result'):
                assert rows[name][key] == value, (name, key)
    receipt = json.loads(next((outside['tmp']/'pauses').glob('*/receipt.json')).read_text(encoding='utf-8'))
    assert [j['registry_writes'] for j in receipt['jobs']] == [['running', 'done'], ['running', 'failed']]
    assert receipt['resumed'] is True


def test_a_row_cancelled_during_the_pause_refuses_its_job_alone(outside):
    cancel = ('import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text(encoding="utf-8")); '
              '[r.update(status="cancelled") for r in d["rows"] if r["id"]=="look"]; '
              'p.write_text(json.dumps(d), encoding="utf-8")')
    jobs_file(outside, dict(name='probe', cmd=['PY', '-c', cancel, str(outside['registry'])]),
              dict(name='look', cmd=['PY', '-c', 'print("must not run")']))
    assert outside['module'].main() == 0
    receipt = json.loads(next((outside['tmp']/'pauses').glob('*/receipt.json')).read_text(encoding='utf-8'))
    probe, look = receipt['jobs']
    assert probe['status'] == 'done' and probe['exit_code'] == 0
    assert look['status'] == 'refused by the test registry' and 'says cancelled' in look['reason']
    assert not (Path(receipt['jobs'][0]['log']).parent/'look.log').exists()
    assert rows_of(outside)['look']['status'] == 'cancelled'


def test_a_job_the_memory_guard_refuses_gets_a_failed_row_that_says_it_never_started(outside, monkeypatch):
    module = outside['module']
    monkeypatch.setattr(module, 'guard', lambda need: (False, 'REFUSED (test)'))
    jobs_file(outside, dict(name='probe', guard_wait_s=0, cmd=['PY', '-c', 'print(1)']))
    assert module.main() == 0
    row = rows_of(outside)['probe']
    assert row['status'] == 'failed' and row['last_run']['exit_code'] is None
    assert row['last_run']['outcome'].startswith('not started: refused by the memory guard')


# ------------------------------------------------------------------------------------------ the held-pause runner
@pytest.fixture
def held(runner, monkeypatch):
    """The held-pause runner on the runner fixture's temporary tree: training held by a pause, no trainer, no watchdog."""
    spec = importlib.util.spec_from_file_location(
        'held_runner_registry_under_test', REPO/'docs/gain_of_function_v4/coordinator/run_jobs_in_held_pause.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('ROOT', 'SAVED', 'OUT', 'HEAVY_LOCK', 'PY', 'guard'):
        monkeypatch.setattr(module.rtp, name, getattr(runner['module'], name))
    monkeypatch.setattr(module, 'WATCHDOG', [])
    monkeypatch.setattr(module, 'trainer_running', lambda: False)
    write(runner['saved']/'supervisor_control.json', dict(cmd='pause', reason='a test hold'))
    return dict(runner, held=module)


def held_jobs(fixture, *jobs):
    path = fixture['tmp']/'jobs_held.json'
    write(path, [dict(dict(cwd='.', need_gb=2, timeout_s=60), **j) for j in jobs])
    return path


@pytest.mark.parametrize('name,defect,expected', RUNNER_PLANTED, ids=[p[0] for p in RUNNER_PLANTED])
def test_the_held_pause_runner_refuses_before_taking_its_turn(held, monkeypatch, name, defect, expected):
    module = held['held']
    registry_with(held['tmp'], defect)
    path = held_jobs(held, dict(name='look', cmd=['PY', '-c', 'print(1)']), dict(name='probe', cmd=['PY', '-c', 'print(2)']))
    monkeypatch.setattr(module.rtp, 'take_runner_turn', lambda *a, **k: pytest.fail('the runner turn was taken'))
    with pytest.raises(SystemExit, match='1 of 2 jobs have no complete row in the test registry'):
        module.main([str(path)])
    receipt = json.loads(next((held['tmp']/'pauses').glob('*_held_refused/receipt.json')).read_text(encoding='utf-8'))
    assert receipt['jobs_files'] == [str(path)] and expected in receipt['jobs'][1]['reason']
    assert json.loads((held['saved']/'supervisor_control.json').read_text())['cmd'] == 'pause', 'the hold is untouched'


def test_the_held_pause_runner_reads_the_registry_again_after_its_turn_wait(held, monkeypatch):
    module = held['held']
    path = held_jobs(held, dict(name='probe', cmd=['PY', '-c', 'print(1)']))

    def wait_while_the_row_is_cancelled(*args, **kwargs):
        registry_with(held['tmp'], _changed(status='cancelled'))
        return 60

    monkeypatch.setattr(module.rtp, 'take_runner_turn', wait_while_the_row_is_cancelled)
    with pytest.raises(SystemExit, match='no complete row'):
        module.main([str(path)])
    receipt = json.loads(next((held['tmp']/'pauses').glob('*_held_refused/receipt.json')).read_text(encoding='utf-8'))
    assert receipt['turn_waited_s'] == 60 and 'says cancelled' in receipt['jobs'][0]['reason']


def test_the_held_pause_runner_writes_its_jobs_rows(held):
    module = held['held']
    path = held_jobs(held, dict(name='probe', cmd=['PY', '-c', 'print(1)']),
                     dict(name='look', cmd=['PY', '-c', 'raise SystemExit(3)']))
    assert module.main([str(path)]) == 0
    rows = rows_of(held)
    assert rows['probe']['status'] == 'done' and rows['look']['status'] == 'failed'
    assert rows['probe']['last_run']['runner'] == 'run_jobs_in_held_pause.py' and rows['look']['last_run']['exit_code'] == 3


# ------------------------------------------------------------------------------------------ the ability scheduler
def test_the_scheduler_holds_a_task_without_a_complete_row_and_runs_it_once_the_row_is_complete(tiny_worker, monkeypatch):
    from mimo_bridge.tools import ability_comparison_window
    requested = []
    monkeypatch.setattr(ability_comparison_window, 'request', lambda *args: requested.append(args) or False)
    monkeypatch.setattr(worker, 'registry_refusals', SCHEDULER_CHECK)
    root = tiny_worker['root']
    for rows, expected in (([], "no row in the test registry has id 'tiny'"),                       # missing row
                           ([complete_row('tiny_a')], "a row has the near id 'tiny_a'"),            # wrong id
                           ([complete_row('tiny', why='')], 'why it matters is empty (why)')):     # empty field
        write_registry(root, rows)
        record = tick(tiny_worker)['tasks']['tiny']
        assert record['status'] == 'repair needed' and record['reason'].startswith('The test registry refuses this task')
        assert expected in record['reason'] and 'attempts' not in record
    (root/'docs/gain_of_function_v4/test_registry.json').unlink()                                # no registry file
    assert 'does not exist' in tick(tiny_worker)['tasks']['tiny']['reason']
    assert not requested and not list(tiny_worker['folder'].glob('*.process.json')), 'no window, no launch'
    assert not (tiny_worker['checkpoint'].parents[1]/'control.json').exists(), 'no checkpoint was requested'
    write_registry(root, [complete_row('tiny')])
    started = tick(tiny_worker)['tasks']['tiny']
    assert started['status'] in ('starting', 'running') and started['attempts'] == 1 and 'reason' not in started


def test_a_round_already_running_finishes_when_its_row_goes_away_and_the_next_one_is_held(tiny_worker, monkeypatch):
    monkeypatch.setattr(worker, 'registry_refusals', SCHEDULER_CHECK)
    write_registry(tiny_worker['root'], [complete_row('tiny')])
    output = Path(tick(tiny_worker)['tasks']['tiny']['output'])
    write_registry(tiny_worker['root'], [])
    assert tick(tiny_worker)['tasks']['tiny']['status'] == 'running', 'a running round is never stopped for its row'
    until(tiny_worker, lambda _: output.is_dir())
    (output/'release').write_text('finish')
    until(tiny_worker, lambda s: s['tasks']['tiny']['status'] == 'complete')
    held_record = tick(tiny_worker)['tasks']['tiny']
    assert held_record['status'] == 'repair needed' and held_record['reason'].startswith(
        'The test registry refuses this task') and "no row in the test registry has id 'tiny'" in held_record['reason']


# ------------------------------------------------------------------------------------------ the Tests page
def page_rows(tmp):
    t = 'Oct 2, 2026 %s PM PDT'
    rows = [complete_row('old_done', status='done', written_at=t % '1:00',
                         result=dict(read_at=t % '2:00', values=['V = 0.79 (0 = exact, 1 = no better)'], verdict='pass'),
                         decision='adopted'),
            complete_row('new_done', status='done', written_at=t % '3:00',
                         result_source=dict(file='out/verdict.json', keys=['passed'])),
            complete_row('running_one', status='running', written_at=t % '2:30',
                         last_run=dict(started=t % '2:31', ended=None, exit_code=None, log='C:/x.log', outcome='running')),
            complete_row('broken', status='queued', written_at=t % '4:00', question='', cost=dict(training_stopped=True))]
    (tmp/'out').mkdir(exist_ok=True)
    (tmp/'out/verdict.json').write_text('{}', encoding='utf-8')
    return rows


def test_the_tests_page_groups_rows_newest_first_and_names_what_is_missing(tmp_path):
    write_registry(tmp_path, page_rows(tmp_path))
    view = tr.dashboard_view(tmp_path)
    assert view['available'] and [g['status'] for g in view['groups']] == ['running', 'queued', 'done']
    assert [v['id'] for v in view['groups'][2]['rows']] == ['new_done', 'old_done'], 'newest written_at first'
    broken = view['groups'][1]['rows'][0]
    assert broken['incomplete_text'].startswith('Incomplete - the runners will refuse it: ')
    assert 'question is empty' in broken['incomplete_text'] and 'cost gives no memory in GB' in broken['incomplete_text']
    assert 'cost.memory_gb' not in broken['incomplete_text'], 'plain labels on the page'
    new = view['groups'][2]['rows'][0]
    assert CLOCK.match(new['result_file']['modified']) and new['result_file_text'].startswith('Result file written ')
    assert 'result_file_text' not in view['groups'][2]['rows'][1], 'a result file that does not exist has no time'
    assert view['groups'][0]['rows'][0]['last_run_text'].startswith('Last run: started Oct 2, 2026 2:31 PM PDT')
    labels = [s['label'] for s in view['groups'][2]['rows'][1]['sections']]
    assert labels == ['Inputs', 'Steps', 'Measures', 'Pass line and its basis', 'Cost', 'On pass / on fail', 'Result',
                      'Decision', 'Written at', 'Code files']
    assert view['sentence'].startswith('4 rows: 1 running, 1 queued, 2 done. 1 row is refused')


def _cost_section(view, row_id):
    row = next(v for g in view['groups'] for v in g['rows'] if v['id'] == row_id)
    return next(s['lines'] for s in row['sections'] if s['label'] == 'Cost')


def test_the_tests_page_states_each_rows_time_limit_beside_its_cost(tmp_path):
    """cost.time_limit_s, in seconds (a pause job's "timeout_s" in its jobs file): a number above 0 is printed in the
    row's Cost section as "Stops by itself after N seconds", with hours and minutes from 60 seconds up. Every other row
    says "No time limit stated": a row without the key, with None, with 0, with text in that place, and a row with no
    cost at all. The key is printed once, never again as a "time limit s: N" line."""
    cost = dict(training_stopped=True, memory_gb=12, where='training pause')
    rows = [complete_row('limited', cost=dict(cost, time_limit_s=7500, basis='twice the longest earlier run')),
            complete_row('short', cost=dict(cost, time_limit_s=45)),
            complete_row('one', cost=dict(cost, time_limit_s=1)),
            complete_row('two_minutes', cost=dict(cost, time_limit_s=120.0)),
            complete_row('long', cost=dict(cost, time_limit_s=27000)),
            complete_row('part_seconds', cost=dict(cost, time_limit_s=90.5)),
            complete_row('no_key'),
            complete_row('none', cost=dict(cost, time_limit_s=None, basis='production training has no end set here')),
            complete_row('zero', cost=dict(cost, time_limit_s=0)),
            complete_row('as_text', cost=dict(cost, time_limit_s='until model step 465,800')),
            complete_row('no_cost', cost=None)]
    write_registry(tmp_path, rows)
    view = tr.dashboard_view(tmp_path)
    assert _cost_section(view, 'limited') == ['Training stopped: yes', 'Memory: 12 GB', 'Where: training pause',
                                              'Stops by itself after 7,500 seconds (2 hours 5 minutes)',
                                              'basis: twice the longest earlier run']
    assert _cost_section(view, 'short')[3:] == ['Stops by itself after 45 seconds']
    assert _cost_section(view, 'one')[3:] == ['Stops by itself after 1 second']
    assert _cost_section(view, 'two_minutes')[3:] == ['Stops by itself after 120 seconds (2 minutes)']
    assert _cost_section(view, 'long')[3:] == ['Stops by itself after 27,000 seconds (7 hours 30 minutes)']
    assert _cost_section(view, 'part_seconds')[3:] == ['Stops by itself after 90.5 seconds']
    assert _cost_section(view, 'no_key') == ['Training stopped: yes', 'Memory: 2 GB', 'Where: training pause',
                                             'No time limit stated']
    assert _cost_section(view, 'none')[3:] == ['No time limit stated', 'basis: production training has no end set here']
    assert _cost_section(view, 'zero')[3:] == ['No time limit stated in seconds (the row says: 0)']
    assert _cost_section(view, 'as_text')[3:] == ['No time limit stated in seconds (the row says: until model step 465,800)']
    assert _cost_section(view, 'no_cost') == ['No time limit stated'], 'a row without a cost still says so'
    for group in view['groups']:
        for row in group['rows']:
            lines = _cost_section(view, row['id'])
            stated = [x for x in lines if x.startswith(('Stops by itself after ', 'No time limit stated'))]
            assert len(stated) == 1, (row['id'], lines)
            assert not any(x.startswith('time limit s:') for x in lines), (row['id'], lines)
    assert tr._duration_text(3661) == '1 hour 1 minute 1 second' and tr._duration_text(10800) == '3 hours'


def test_a_stated_time_limit_of_a_pause_job_equals_a_timeout_in_a_jobs_file():
    """The pause runner ends a job at the "timeout_s" of the JOBS FILE (run_in_training_pause.run_jobs, 3600 when the
    job gives none), never at the row's number. A row whose time_limit_s equals no timeout_s of a job of its name
    would print a limit on the Tests page that no program applies. Read from this checkout's own registry and its jobs
    files (docs/gain_of_function_v4/pause_jobs_*.json and coordinator/pause_jobs_*.json). Not judged here: a row with
    no stated limit, and a row whose job is in none of those files (a worktree's own jobs file, a scheduler task, a
    production segment)."""
    folder = REPO/'docs/gain_of_function_v4'
    files = sorted(folder.glob('pause_jobs_*.json')) + sorted((folder/'coordinator').glob('pause_jobs_*.json'))
    registry = tr.registry_path(REPO)
    if not files or not registry.is_file():
        pytest.skip('this checkout holds no jobs files or no registry')
    timeouts = {}
    for path in files:
        jobs = json.loads(path.read_text(encoding='utf-8-sig'))
        for job in jobs if isinstance(jobs, list) else jobs.get('jobs', []):
            if isinstance(job, dict) and isinstance(job.get('name'), str):
                timeouts.setdefault(job['name'], {})[path.name] = job.get('timeout_s', 3600)
    judged, wrong = 0, []
    for row in tr.load(registry)['rows']:
        cost = row.get('cost') if isinstance(row, dict) else None
        limit = cost.get('time_limit_s') if isinstance(cost, dict) else None
        if limit is None or row.get('id') not in timeouts:
            continue
        judged += 1
        if limit not in timeouts[row['id']].values():
            wrong.append('%s states %s s; its jobs say %s' % (row['id'], limit, timeouts[row['id']]))
    assert not wrong, wrong
    assert judged, 'no row with a stated limit has its job in these files: the check judged nothing'


def test_the_tests_page_lists_scheduled_checks_without_a_row_and_an_unreadable_registry(tmp_path):
    write_registry(tmp_path, [complete_row('hand_following')])
    write(tmp_path/'docs/gain_of_function_v3/ability_rechecks.json',
          dict(tasks=[dict(id='hand_following', title='Hand following'), dict(id='gaze_prediction', title='Eye forecast')]))
    view = tr.dashboard_view(tmp_path)
    assert view['scheduler_gaps'] == [dict(id='gaze_prediction', title='Eye forecast', text='No row yet.')]
    assert "'repair needed'" in view['scheduler_gaps_note']
    tr.registry_path(tmp_path).write_text('{"rows": ', encoding='utf-8')
    view = tr.dashboard_view(tmp_path)
    assert view['available'] is False and 'refuse every test' in view['sentence']


def test_the_dashboard_serves_the_tests_page_live_from_the_file(tmp_path):
    from http.server import ThreadingHTTPServer
    from mimo_bridge.tools.development_dashboard import Handler
    write_registry(tmp_path, page_rows(tmp_path))
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.root = tmp_path
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:%d' % server.server_address[1]
    try:
        first = json.loads(urllib.request.urlopen(base + '/api/tests', timeout=20).read())
        assert sum(len(g['rows']) for g in first['groups']) == 4
        write_registry(tmp_path, page_rows(tmp_path) + [complete_row('added_now', status='queued')])
        second = json.loads(urllib.request.urlopen(base + '/api/tests', timeout=20).read())
        assert any(v['id'] == 'added_now' for g in second['groups'] for v in g['rows']), 'read again on every request'
        page = urllib.request.urlopen(base + '/', timeout=20).read().decode('utf-8')
        assert 'id="tab-tests"' in page and 'id="tests"' in page
        script = urllib.request.urlopen(base + '/assets/development_dashboard.js', timeout=20).read().decode('utf-8')
        # since Oct 5, 2026 the page reads the rows inside their seven sections (the recovery directive, section 7A)
        assert "fetch('/api/tests?view=sections'" in script and "if(name==='tests')" in script
        by_section = json.loads(urllib.request.urlopen(base + '/api/tests?view=sections', timeout=20).read())
        assert sum(s['count'] for s in by_section['sections']) == 5 and 'groups' not in by_section
    finally:
        server.shutdown()
        server.server_close()
