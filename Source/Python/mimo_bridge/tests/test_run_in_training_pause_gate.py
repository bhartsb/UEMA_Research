# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The pause runner's GATE jobs (docs/gain_of_function_v4/run_in_training_pause.py; the coordinator's
qualification gate for the pause in which the three observer branches merge). The coordinator's decision of
2026-09-26: the qualification runs on a candidate merge commit in its own worktree, never on production's
tree, so a failed gate never keeps training stopped - training resumes on production's code as it stands,
and production moves to the candidate only when every gate passed. The receipt lists each gate's pass or
fail, in the scheduler's review hold as well (before, the hold's path skipped the gates and a failed proof
left no trace). The trainer and the supervisor are simulated as in test_run_in_training_pause.py.
"""
import json
import sys

from test_run_in_training_pause import outside, receipt_of, runner, write  # noqa: F401 (fixtures)


def _jobs(fixture, *jobs):
    path = fixture['tmp']/'jobs_gate.json'
    write(path, [dict(dict(cwd='.', need_gb=2, timeout_s=60), **j) for j in jobs])
    sys.argv[:] = ['run_in_training_pause.py', '--jobs', str(path)]


def test_a_failed_gate_resumes_training_and_the_receipt_names_it(outside):
    module = outside['module']
    _jobs(outside, dict(name='proof', gate=True, cmd=['PY', '-c', 'raise SystemExit(1)']),
          dict(name='clean', gate=True, cmd=['PY', '-c', 'print("passed")']),
          dict(name='after', cmd=['PY', '-c', 'print("still runs")']))
    assert module.main() == 3, 'a failed gate is reported by the exit code'
    control = json.loads((outside['saved']/'supervisor_control.json').read_text(encoding='utf-8'))
    assert control['cmd'] == 'run' and 'pause_runner' not in control, 'the pause is lifted: training resumes'
    receipt = receipt_of(outside)
    assert receipt['resumed'] is True and 'not_lifted' not in receipt
    assert receipt['gates_failed'] == ['proof'] and receipt['gates_passed'] is False
    assert [(g['name'], g['passed']) for g in receipt['gates']] == [('proof', False), ('clean', True)]
    assert [j['name'] for j in receipt['jobs']] == ['proof', 'clean', 'after'], 'every job still ran'


def test_passed_gates_resume_whatever_an_ordinary_job_did(outside):
    module = outside['module']
    _jobs(outside, dict(name='proof', gate=True, cmd=['PY', '-c', 'print("passed")']),
          dict(name='yoke', cmd=['PY', '-c', 'raise SystemExit(3)']))
    assert module.main() == 0
    control = json.loads((outside['saved']/'supervisor_control.json').read_text(encoding='utf-8'))
    assert control['cmd'] == 'run'
    receipt = receipt_of(outside)
    assert receipt['resumed'] is True and receipt['gates_failed'] == [] and receipt['gates_passed'] is True
    assert [(g['name'], g['passed']) for g in receipt['gates']] == [('proof', True)]


def test_a_refused_gate_fails(outside, monkeypatch):
    module = outside['module']
    monkeypatch.setattr(module, 'guard', lambda need: (need < 5, 'REFUSED (test)' if need >= 5 else 'ALLOWED (test)'))
    _jobs(outside, dict(name='big', gate=True, need_gb=9, guard_wait_s=0, cmd=['PY', '-c', 'print("never")']),
          dict(name='small', gate=True, cmd=['PY', '-c', 'print("ran")']))
    assert module.main() == 3
    receipt = receipt_of(outside)
    assert receipt['gates_failed'] == ['big'] and receipt['gates'][0]['status'] == 'refused by the memory guard'


def test_the_review_hold_judges_its_gates_and_keeps_the_hold(runner):
    module = runner['module']
    before = (runner['saved']/'supervisor_control.json').read_bytes()
    _jobs(runner, dict(name='proof', gate=True, cmd=['PY', '-c', 'raise SystemExit(1)']),
          dict(name='after', cmd=['PY', '-c', 'print("still runs")']))
    assert module.main() == 3
    assert (runner['saved']/'supervisor_control.json').read_bytes() == before, 'the hold is never lifted by the runner'
    receipt = receipt_of(runner)
    assert receipt['review_hold'] == 'held' and receipt['resumed'] is False
    assert receipt['gates_failed'] == ['proof'] and [g['passed'] for g in receipt['gates']] == [False]
    assert [j['name'] for j in receipt['jobs']] == ['proof', 'after']


def test_the_review_hold_reports_passed_gates(runner):
    module = runner['module']
    _jobs(runner, dict(name='proof', gate=True, cmd=['PY', '-c', 'print("passed")']))
    assert module.main() == 0
    receipt = receipt_of(runner)
    assert receipt['gates_passed'] is True and receipt['gates_failed'] == []


def test_gate_jobs_run_in_the_candidate_worktree(outside):
    module = outside['module']
    candidate = outside['tmp']/'candidate'
    candidate.mkdir()
    marker = outside['tmp']/'where.txt'
    _jobs(outside, dict(name='proof', gate=True, cwd='{candidate}',
                        cmd=['PY', '-c', 'import os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(os.getcwd() + "|" + sys.argv[2])',
                             str(marker), '{candidate}/docs']))
    sys.argv += ['--candidate', str(candidate)]
    assert module.main() == 0
    where, arg = marker.read_text().split('|')
    root = str(candidate.resolve()).replace(chr(92), '/')
    assert where.replace(chr(92), '/') == root and arg == root + '/docs'


def test_a_candidate_job_without_a_candidate_refuses(outside):
    module = outside['module']
    _jobs(outside, dict(name='proof', gate=True, cwd='{candidate}', cmd=['PY', '-c', 'print(1)']))
    assert module.main() == 1
    assert not (outside['tmp']/'pauses').exists() or not list((outside['tmp']/'pauses').glob('*/receipt.json'))
