# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The registry file stays readable and writable by the OLDER test_registry module that production's running helpers
hold in memory.

Production's helpers were started before this work and keep their copy of tools/test_registry.py until they are
restarted: the ability scheduler (started Oct 2, 2026) holds the file as of commit d7ae56088, the dashboard on port
8766 (started Oct 3, 2026) the file as of commit 2e47e8ad2, which is also the file in production's checkout. They
read and write the SAME registry file as the runner of this work. So this work adds no status value (the six of
STATUSES stay) and carries its vocabulary in added fields. These tests load each older module as it was (from
production's checkout by its path, and from git by its commit), let the NEW runner write every kind of row, and then
ask the OLD module to load the file, judge every row, render the Tests page, and write into it.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mimo_bridge.tools import test_registry as new
from mimo_bridge.tools import test_runner as runner
from test_run_in_training_pause import complete_row, write_registry
from test_test_runner import new_row

REPO = Path(__file__).resolve().parents[4]
MODULE = 'Source/Python/mimo_bridge/tools/test_registry.py'
PRODUCTION = Path('C:/Users/bhart/Documents/UEMA_devA_private')
OLD = [('the ability scheduler, commit d7ae56088', 'git', 'd7ae56088'),
       ('the dashboard on port 8766, commit 2e47e8ad2', 'git', '2e47e8ad2'),
       ("production's checkout", 'file', PRODUCTION/MODULE)]
PY = sys.executable
OLD_STATUSES = ('written', 'queued', 'running', 'done', 'failed', 'cancelled')


def old_module(kind, where, tmp_path):
    """The older module, loaded under its own name from a copy of its text; the test is skipped when its source
    (the commit, or production's checkout) is not on this machine."""
    if kind == 'git':
        done = subprocess.run(['git', '-C', str(REPO), 'show', '%s:%s' % (where, MODULE)], capture_output=True)
        if done.returncode != 0:
            pytest.skip('commit %s is not in this repository' % where)
        source = done.stdout
    else:
        if not Path(where).is_file():
            pytest.skip('%s is not on this machine' % where)
        source = Path(where).read_bytes()
    target = tmp_path/'old_module'/'test_registry_old.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
    spec = importlib.util.spec_from_file_location('test_registry_old_%d' % abs(hash((kind, str(where)))), target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not hasattr(module, 'finish_job'), 'this is the module from before the runner record existed'
    return module


@pytest.fixture(params=OLD, ids=[name for name, _, _ in OLD])
def old(request, tmp_path):
    _name, kind, where = request.param
    return old_module(kind, where, tmp_path)


@pytest.fixture
def written(tmp_path, monkeypatch):
    """A registry in which the NEW code wrote every kind of row: a queued row with all the added fields; rows the
    runner ended PASS, FAIL, INVALID and STOPPED; a blocked row; a cancelled row; a row with a follow-up; a row the
    runner left running; and a row written before Oct 5 that carries a component and a diagram added later."""
    monkeypatch.setenv('UEMA_TEST_RUNS_DIR', str(tmp_path/'runs'))
    monkeypatch.setattr(new, 'component_ids', lambda root=None: {'TEST_REGISTRY', 'PLANNER_SEARCH'})
    monkeypatch.setattr(new, 'code_state', lambda cwd=None, keep_s=30.0: ('abc123def', False, 'a/test-branch'))
    names = ('queued', 'passes', 'fails', 'invalid', 'stopped', 'blocked', 'cancelled', 'followed', 'live', 'reads')
    verdict_file = tmp_path/'out/verdict.json'
    verdict_file.parent.mkdir()
    rows = [new_row(n) for n in names[:-1]] + [new_row('reads', result_source=dict(file=str(verdict_file),
                                                                                  keys=['verdict']))]
    path = write_registry(tmp_path, rows + [complete_row('legacy')])

    def run(name, code, **more):
        return runner.run_command(name, [PY, '-c', code], path=path, root=tmp_path, check_rule=False, **more)[1]
    assert run('passes', 'pass') == 'PASS' and run('fails', 'raise SystemExit(1)') == 'FAIL'
    assert run('invalid', 'raise SystemExit(7)') == 'INVALID' and run('followed', 'raise SystemExit(9)') == 'INVALID'
    assert run('stopped', 'import time; time.sleep(60)', timeout_s=1.0, poll_s=0.2) == 'STOPPED'
    reading = dict(name='tactile error', value=92, unit='of 100', meaning='100 = no better than no change',
                   baseline='100', statistically_clear='not judged', statistical_basis='',
                   gof_material=new.GOF_ANSWERS[2])
    assert runner.run_command('reads', [PY, '-c', 'import json,sys; json.dump(dict(verdict="pass", '
                                                  'readings=json.loads(sys.argv[2])), open(sys.argv[1], "w"))',
                                        str(verdict_file), json.dumps([reading])], path=path, root=tmp_path,
                              check_rule=False)[1] == 'PASS'
    assert new.block_row('blocked', 'the checkpoint is being copied', 'the copy ends', 'a builder', path=path) is None
    assert new.cancel_row('cancelled', 'superseded', 'a builder', path=path) is None
    assert new.write_follow_up('followed', 'run it again on the copied checkpoint', 'a builder', 'a builder',
                               path=path) is None
    assert new.job_started('live', path=path, pid=os.getpid()) is None
    assert new.update_row('legacy', dict(component='PLANNER_SEARCH', diagram=dict(
        input='32 saved decisions', operation='run the search twice', measurement='degrees between the two commands',
        decision='under 1 degree: the search is stable')), path=path)
    return dict(path=path, root=tmp_path, names=names + ('legacy',))


def test_rows_written_by_the_new_runner_load_and_are_judged_complete_by_the_older_module(old, written):
    data = old.load(written['path'])
    required = old.required_fields(data)
    assert required == new.required_fields(new.load(written['path'])), 'the required list was not changed'
    by_id = {row['id']: row for row in data['rows']}
    assert sorted(by_id) == sorted(written['names'])
    for name, row in by_id.items():
        assert old.row_problems(row, required) == [], name
        assert row['status'] in OLD_STATUSES and row['status'] in old.STATUSES, (name, row['status'])
    assert {name: row['status'] for name, row in by_id.items()} == dict(
        queued='queued', passes='done', fails='failed', invalid='failed', stopped='failed', blocked='queued',
        cancelled='cancelled', followed='failed', live='running', reads='done', legacy='queued')
    # the older runners' own gate: a complete queued row may run, a cancelled one is refused, as before
    assert old.refusal('queued', path=written['path']) is None and old.refusal('legacy', path=written['path']) is None
    assert 'cancelled' in old.refusal('cancelled', path=written['path'])
    assert old.refusals(['queued', 'passes', 'nobody'], path=written['path'])['nobody'] is not None


def test_the_older_module_renders_the_tests_page_from_the_same_file(old, written):
    view = old.dashboard_view(written['root'])
    assert view['available'] is True
    rows = [row for group in view['groups'] for row in group['rows']]
    assert sorted(row['id'] for row in rows) == sorted(written['names'])
    assert [group['status'] for group in view['groups']] == ['running', 'queued', 'failed', 'done', 'cancelled']
    assert not [row['id'] for row in rows if row.get('incomplete_text')], 'no row reads as incomplete'
    assert all(row['status_label'] != 'Status missing or unknown' for row in rows)
    for row in rows:
        labels = [section['label'] for section in row['sections']]
        assert {'Inputs', 'Steps', 'Measures', 'Cost', 'Result', 'Decision'} <= set(labels), row['id']
        json.dumps(row)                               # every value the older page gets is plain JSON
    passes = next(row for row in rows if row['id'] == 'passes')
    assert passes['last_run_text'].startswith('Last run: started ') and 'exit code 0' in passes['last_run_text']
    result = next(section for section in passes['sections'] if section['label'] == 'Result')
    assert any(line.startswith('Verdict: PASS') for line in result['lines']), 'the new result reads on the old page'
    reads = next(section for row in rows if row['id'] == 'reads' for section in row['sections']
                 if section['label'] == 'Result')
    assert any(line.startswith('readings: ') and 'tactile error' in line for line in reads['lines']), (
        'a structured reading is printed by the old page as plain text')


def test_both_modules_write_the_file_in_one_layout(old, written):
    data = new.load(written['path'])
    assert old.dumps(data) == new.dumps(data), 'alternating writers would otherwise rewrite every line'
    before = written['path'].read_text(encoding='utf-8')
    assert old.dumps(old.load(written['path'])) + '\n' == before, 'an old write of an unchanged row changes no byte'


def test_the_older_writers_keep_working_and_keep_the_added_fields(old, written):
    path = written['path']
    kept = {name: {k: v for k, v in row.items() if k in ('component', 'diagram', 'gof_link', 'outcome', 'blocked')}
            for name, row in ((r['id'], r) for r in new.load(path)['rows'])}
    assert old.update_row('queued', dict(decision='waits for the test window'), path=path) is True
    assert old.job_started('legacy', runner='run_in_training_pause.py', log='a.log', path=path) is None
    assert old.job_finished('legacy', exit_code=0, runner='run_in_training_pause.py', log='a.log', path=path) is None
    after = {row['id']: row for row in new.load(path)['rows']}
    assert after['queued']['decision'] == 'waits for the test window' and after['legacy']['status'] == 'done'
    for name, fields in kept.items():
        assert {k: after[name].get(k) for k in fields} == fields, 'the old writer changed an added field of ' + name
    # the new audit: an older runner's unsealed record written after the baseline is an error, whatever its date
    # (the dashboard review of Oct 5, 2026, finding F5); a baseline written at the merge accepts it as it stands
    found = runner.audit(path=path, root=written['root'], processes=[], code_root=written['root'])
    assert ("status without the runner's record", 'legacy') in [(m['kind'], m['row']) for m in found['mismatches']]
    target = runner.baseline_path(written['root'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runner.make_baseline(new.load(path))), encoding='utf-8')
    found = runner.audit(path=path, root=written['root'], processes=[], code_root=written['root'])
    assert 'legacy' in [u['row'] for u in found['unverified']]
    assert 'legacy' not in [m['row'] for m in found['mismatches']]


def test_an_older_runners_later_run_hides_what_an_earlier_new_run_left_in_the_row(old, written):
    """The older job_started and job_finished replace last_run and leave outcome, next_action, result and follow_up
    as an earlier run of the new runner wrote them. The new views read the runner's record, so none of those is shown
    for the later run."""
    path = written['path']
    assert old.job_started('followed', runner='ability_rechecks.py', log='b.log', path=path) is None
    assert old.job_finished('followed', exit_code=0, runner='ability_rechecks.py', log='b.log', path=path) is None
    row = next(r for r in new.load(path)['rows'] if r['id'] == 'followed')
    assert row['status'] == 'done' and row['outcome'] == 'INVALID', 'the copied field still says INVALID'
    assert new.recorded_outcome(row) is None and new.verified_outcome(row) is None
    assert new.section_of(row, new.DEFAULT_REQUIRED) == 'DONE'
    view = new.added_view(row, new.DEFAULT_REQUIRED)
    assert view['outcome'] is None and view['next_action'] is None and view['follow_up_owed'] is False
    assert view['result_from_earlier_run'] is True and view['readings'] == [] and view['unjudged_result'] is False
    assert view['card']['conclusion'] == 'No verdict is written in the row.'
    assert view['card']['state'] == 'DONE' and view['run']['sealed'] is False


def test_a_new_runner_ends_a_row_the_older_runner_started(old, written):
    path = written['path']
    assert old.job_started('queued', runner='run_in_training_pause.py', log='c.log', path=path) is None
    why, state = new.finish_job('queued', exit_code=0, path=path, runner='run_in_training_pause.py')
    row = next(r for r in new.load(path)['rows'] if r['id'] == 'queued')
    assert (why, state) == (None, 'PASS') and row['status'] == 'done' and new.sealed(row)
    assert old.row_problems(row, old.required_fields(old.load(path))) == []


def test_production_registry_as_it_stands_gains_the_added_fields_and_stays_complete_for_the_older_module(old, tmp_path):
    """The migration of this work (registry_added_fields_2026-10-05.py) applied to a COPY of production's registry:
    the older module judges every row exactly as it judged it before, and the file's other fields are unchanged."""
    source = PRODUCTION/'docs/gain_of_function_v4/test_registry.json'
    if not source.is_file():
        pytest.skip("production's registry is not on this machine")
    spec = importlib.util.spec_from_file_location(
        'registry_added_fields', REPO/'docs/gain_of_function_v4/dashboard_recovery/registry_added_fields_2026-10-05.py')
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    target = tmp_path/'docs/gain_of_function_v4/test_registry.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    before = old.load(target)
    required = old.required_fields(before)
    judged_before = {row['id']: old.row_problems(row, required) for row in before['rows']}
    migration.apply(target)
    after = old.load(target)
    assert {row['id']: old.row_problems(row, required) for row in after['rows']} == judged_before
    assert [row['id'] for row in after['rows']] == [row['id'] for row in before['rows']]
    for was, now in zip(before['rows'], after['rows']):
        assert {k: v for k, v in now.items() if k not in ('component', 'diagram')} == \
            {k: v for k, v in was.items() if k not in ('component', 'diagram')}, was['id']
        assert all(row['status'] in old.STATUSES for row in (was, now))
    assert migration.missing(target) == [], 'every row that is not done has a component and a diagram'
    assert old.dashboard_view(tmp_path)['available'] is True
