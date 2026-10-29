# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The training supervisor keeps exactly one developmental-observer digester alive
(training_continuation.ensure_observer), as it does the archiver and the ability scheduler."""
from types import SimpleNamespace

from mimo_bridge.tools import training_continuation as tc


def test_a_running_digester_prevents_another_launch(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=7, info=dict(cmdline=['python', '-u', '-m', 'mimo_bridge.tools.observer', 'watch'],
                                               create_time=990.0))
    monkeypatch.setattr(tc.psutil, 'process_iter', lambda fields: [process])
    monkeypatch.setattr(tc, 'launch', lambda *a: (_ for _ in ()).throw(AssertionError('duplicate')))
    assert tc.ensure_observer(tmp_path, tmp_path, now=1000.0) == {'status': 'running', 'pids': [7]}


def test_a_digester_that_stopped_reporting_is_stopped_and_recorded(tmp_path, monkeypatch):
    import json
    killed = []
    process = SimpleNamespace(pid=8, info=dict(cmdline=['python', '-m', 'mimo_bridge.tools.observer', 'watch'],
                                               create_time=100.0), kill=lambda: killed.append(8))
    monkeypatch.setattr(tc.psutil, 'process_iter', lambda fields: [process])
    (tmp_path / 'observer').mkdir()
    (tmp_path / 'observer' / 'daemon_status.json').write_text(json.dumps({'time': 200.0}), encoding='utf-8')
    fresh = tc.ensure_observer(tmp_path, tmp_path, now=200.0 + tc.OBSERVER_STALL_S - 1)
    assert fresh['status'] == 'running' and not killed
    got = tc.ensure_observer(tmp_path, tmp_path, now=200.0 + tc.OBSERVER_STALL_S + 1)
    assert got['status'].startswith('stalled') and killed == [8]
    assert json.loads((tmp_path / 'observer' / 'stalled_digester.json').read_text(encoding='utf-8'))['stopped_pids'] == [8]


def test_start_then_back_off_and_a_held_lock_blocks_a_second(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(tc.psutil, 'process_iter', lambda fields: [])
    monkeypatch.setattr(tc, 'launch', lambda command, *a: sent.append(command) or SimpleNamespace(pid=9))
    got = tc.ensure_observer(tmp_path, tmp_path, now=1000.0)
    assert got['status'] == 'started' and got['started_at'] == 1000.0
    assert sent[0][-3:] == ['-m', 'mimo_bridge.tools.observer', 'watch']
    assert 'delayed' in tc.ensure_observer(tmp_path, tmp_path, last_start=1000.0, now=1100.0)['status']
    lock = tc.OwnerLock(tmp_path / 'observer' / 'owner.lock')
    try:
        assert 'holds its lock' in tc.ensure_observer(tmp_path, tmp_path)['status']
    finally:
        lock.close()
    assert len(sent) == 1


def test_the_supervisor_records_the_digester_every_sweep(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(tc, 'ensure_observer', lambda *a, **k: seen.append(1) or {'status': 'running', 'pids': [1]})
    monkeypatch.setattr(tc, 'ensure_archiver', lambda *a, **k: {'status': 'running', 'pids': [1]})
    monkeypatch.setattr(tc, 'ensure_ability_rechecks', lambda *a, **k: {'status': 'test'})
    run = tmp_path / 'run'
    run.mkdir()
    root = tmp_path / 'supervision'
    root.mkdir()
    cmd = ['python', '-m', 'mimo_bridge.train_mimo', '--out', str(run)]
    process = SimpleNamespace(pid=100, info={'cmdline': cmd})
    monkeypatch.setattr(tc, 'processes', lambda kind, requested=None: [process] if kind == 'trainer' else [])
    monkeypatch.setattr(tc, 'launch', lambda *a: None)

    class EndLoop(Exception):
        pass
    monkeypatch.setattr(tc.time, 'sleep', lambda _: (_ for _ in ()).throw(EndLoop()))
    try:
        tc.supervise(run, root, tmp_path / 'hourly_watch.py')
    except EndLoop:
        pass
    import json
    state = json.loads((root / 'training_continuation.json').read_text(encoding='utf-8'))
    assert seen and state['observer'] == {'status': 'running', 'pids': [1]}
