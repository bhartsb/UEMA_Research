# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/run_in_training_pause.py during the ability scheduler's review hold.

The coordinator's decision of 2026-09-25 (the scheduler review's gap b): while a failed remembered-hand-
goals round holds training for review, training is already stopped, so the coordinator's heavy jobs run on
the held pause's checkpoint and supervisor_control.json is never touched - a change to it would end the
hold (ability_comparison_window.held_by_review). Outside a hold the runner's refusals are unchanged. No
trainer, model or world is involved: the trainer list and the memory guard are replaced.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO/'docs/gain_of_function_v4/run_in_training_pause.py'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


# NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE (the style guide, 2026-10-02): the runner refuses a job without a complete row
# in <ROOT>/docs/gain_of_function_v4/test_registry.json. The fixture's ROOT is the test's own folder, so its registry is
# a temporary one holding a complete row for every job name these tests use (test_test_registry.py tests the refusals).
JOB_NAMES = ('probe', 'look', 'proof', 'clean', 'after', 'yoke', 'big', 'small', 'light', 'slow', 'arm')


def complete_row(name, **changes):
    """A registry row with every required field filled, as the registry's "fields" declare them."""
    row = dict(id=name, title='Test job ' + name, question='Does job %s exit with code 0?' % name,
               why='The runner tests read its receipt.', inputs=['the jobs file of the test'],
               steps=['run the command in the pause'], measures=['exit code: 0 = passed, any other = failed'],
               pass_line='exit code 0', pass_line_basis='the command exits 0 when it ran',
               cost=dict(training_stopped=True, memory_gb=2, where='training pause'),
               result_source=dict(file='receipt.json', keys=['jobs']), on_pass='nothing changes',
               on_fail='nothing changes', status='queued', result=None, decision=None,
               written_at='Oct 2, 2026 5:30 PM PDT', code=['docs/gain_of_function_v4/run_in_training_pause.py'])
    row.update(changes)
    return row


def write_registry(root, rows, required=None):
    """A test registry under `root` in the production file's form; returns its path."""
    from mimo_bridge.tools import test_registry
    path = root/'docs/gain_of_function_v4/test_registry.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(about='a test registry', required=list(required or test_registry.DEFAULT_REQUIRED), fields={},
                rows=list(rows))
    path.write_text(test_registry.dumps(data) + '\n', encoding='utf-8')
    return path


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('run_in_training_pause_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = tmp_path/'Saved/mimo_bridge'
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'SAVED', saved)
    monkeypatch.setattr(module, 'PAUSE', saved/'supervisor_control.json')
    monkeypatch.setattr(module, 'OUT', tmp_path/'pauses')
    monkeypatch.setattr(module, 'HEAVY_LOCK', tmp_path/'heavy_job.lock')   # never the real lock in a test
    monkeypatch.setattr(module, 'PY', Path(sys.executable))
    monkeypatch.setattr(module, 'guard', lambda need: (True, 'ALLOWED (test)'))
    running = []
    monkeypatch.setattr(module, 'trainers', lambda: list(running))
    run = saved/'experiments/H84_x_seg41'
    checkpoint = run/'checkpoints/ckpt_step_411350'
    checkpoint.mkdir(parents=True)
    owned = dict(cmd='pause', ability_window_token='held', reason='Saved comparison window: scheduled ability tests need training memory.')
    write(saved/'supervisor_control.json', owned)
    write(saved/'training_continuation.json', dict(active_run=str(run)))
    window = dict(state='held for review', token='held', run=str(run), checkpoint=str(checkpoint), held_at=1000.,
                  owned_control=owned, reason='Remembered hand goals: the round at step 411,350 did not pass.')
    write(saved/'ability_rechecks/status.json', dict(tasks={}, comparison_window=window, active_task=None))
    marker = tmp_path/'marker.txt'
    jobs = tmp_path/'jobs.json'
    write(jobs, [dict(name='probe', cwd='.', need_gb=2, timeout_s=60,
                      cmd=['PY', '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])', str(marker),
                           '{checkpoint}'])])
    monkeypatch.setattr(sys, 'argv', ['run_in_training_pause.py', '--jobs', str(jobs)])
    registry = write_registry(tmp_path, [complete_row(name) for name in JOB_NAMES])
    assert module.registry_path() == registry, 'the runner reads the registry under its own ROOT'
    return dict(module=module, saved=saved, run=run, checkpoint=checkpoint, owned=owned, marker=marker, running=running,
                tmp=tmp_path, registry=registry)


def test_jobs_run_on_the_held_pause_and_the_hold_stays(runner):
    module = runner['module']
    before = (runner['saved']/'supervisor_control.json').read_bytes()
    assert module.refusals() == []
    assert module.main() == 0
    assert runner['marker'].read_text() == str(runner['checkpoint']), 'the job ran on the held pause checkpoint'
    assert (runner['saved']/'supervisor_control.json').read_bytes() == before, 'the hold is never lifted'
    assert not (runner['run']/'control.json').exists(), 'no save was asked of a stopped trainer'
    receipt = json.loads(next((runner['tmp']/'pauses').glob('*/receipt.json')).read_text())
    assert receipt['review_hold'] == 'held' and receipt['resumed'] is False
    assert receipt['jobs'][0]['status'] == 'done' and receipt['jobs'][0]['exit_code'] == 0


@pytest.mark.parametrize('change,refusal', [
    ('trainer', 'a trainer runs during the review hold'),
    ('control', 'the review hold no longer owns the supervisor control'),
    ('task', 'the ability scheduler runs'),
])
def test_a_hold_that_no_longer_stops_training_by_itself_is_refused(runner, change, refusal):
    module = runner['module']
    if change == 'trainer':
        runner['running'].append(object())
    elif change == 'control':
        write(runner['saved']/'supervisor_control.json', dict(cmd='pause', reason='the operator'))
    else:
        status = json.loads((runner['saved']/'ability_rechecks/status.json').read_text())
        write(runner['saved']/'ability_rechecks/status.json', dict(status, active_task='hand_memory_route'))
    assert any(refusal in r for r in module.refusals())
    assert module.main() == 1 and not runner['marker'].exists()


def test_outside_a_hold_a_paused_supervisor_still_refuses(runner):
    module = runner['module']
    status = json.loads((runner['saved']/'ability_rechecks/status.json').read_text())
    write(runner['saved']/'ability_rechecks/status.json', dict(status, comparison_window=dict(status['comparison_window'],
                                                                                             state='review released')))
    runner['running'].append(object())
    assert 'the supervisor is already paused' in module.refusals()


# THE PAUSE IS LIFTED ONLY WHEN IT IS STILL THE RUNNER'S OWN (2026-09-25). Before, the last step wrote
# {"cmd": "run"} over whatever the control held, so a pause written during the jobs by the operator, a
# watcher's stop or the scheduler was lifted with it. The trainer and the supervisor are simulated: the
# save request is consumed at the first check, and a trainer reappears once the control says run.
@pytest.fixture
def outside(runner, monkeypatch):
    module, saved, run = runner['module'], runner['saved'], runner['run']
    status = json.loads((saved/'ability_rechecks/status.json').read_text())
    write(saved/'ability_rechecks/status.json', dict(status, comparison_window=dict(status['comparison_window'],
                                                                                   state='released')))
    write(saved/'supervisor_control.json', {'cmd': 'run', 'archiver': 'off'})
    stopped = []

    def trainers():
        if not stopped:
            return [object()]
        control = json.loads((saved/'supervisor_control.json').read_text(encoding='utf-8'))
        return [object()] if control.get('cmd') == 'run' else []

    def complete_after(run_, step0):
        (run/'control.json').unlink(missing_ok=True)
        stopped.append(True)
        return runner['checkpoint']

    monkeypatch.setattr(module, 'trainers', trainers)
    monkeypatch.setattr(module, 'complete_after', complete_after)
    monkeypatch.setattr(module, 'latest_step', lambda run_: None)
    return runner


def job_file(runner, code, *args):
    jobs = runner['tmp']/'jobs_outside.json'
    write(jobs, [dict(name='probe', cwd='.', need_gb=2, timeout_s=60, cmd=['PY', '-c', code, *map(str, args)])])
    sys.argv[:] = ['run_in_training_pause.py', '--jobs', str(jobs)]


def receipt_of(runner):
    return json.loads(next((runner['tmp']/'pauses').glob('*/receipt.json')).read_text())


def test_the_runner_lifts_its_own_pause_and_keeps_the_other_keys(outside):
    module = outside['module']
    job_file(outside, 'print("job")')
    assert module.refusals() == []
    assert module.main() == 0
    control = json.loads((outside['saved']/'supervisor_control.json').read_text())
    assert control['cmd'] == 'run' and control['archiver'] == 'off' and 'pause_runner' not in control
    receipt = receipt_of(outside)
    assert receipt['resumed'] is True and 'not_lifted' not in receipt


def test_a_pause_written_by_someone_else_during_the_jobs_stays(outside):
    module = outside['module']
    theirs = {'cmd': 'pause', 'reason': 'the operator'}
    job_file(outside, 'import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps(%r))' % theirs,
             outside['saved']/'supervisor_control.json')
    assert module.main() == 2
    assert json.loads((outside['saved']/'supervisor_control.json').read_text()) == theirs
    receipt = receipt_of(outside)
    assert receipt['resumed'] is False and receipt['control_now'] == theirs and 'left as it is' in receipt['not_lifted']


def test_a_control_that_arrives_before_the_save_request_is_never_replaced(outside, monkeypatch):
    module, run = outside['module'], outside['run']
    theirs = {'cmd': 'snapshot', 'reason': 'someone else'}
    monkeypatch.setattr(module, 'latest_step', lambda run_: write(run/'control.json', theirs))
    job_file(outside, 'raise SystemExit(3)')
    with pytest.raises(SystemExit, match='another trainer control arrived'):
        module.main()
    assert json.loads((run/'control.json').read_text()) == theirs, 'the other command is kept'
    control = json.loads((outside['saved']/'supervisor_control.json').read_text())
    assert control['cmd'] == 'run' and control['archiver'] == 'off', 'our own pause is lifted again'
    assert receipt_of(outside)['jobs'] == []


@pytest.mark.parametrize('content,refusal', [
    ('{"cmd": "run", "archiver": "off"}'.encode('utf-16'), 'unreadable supervisor control'),
    (b'{"cmd": "ru', 'cannot be read'),
])
def test_a_control_the_supervisor_cannot_read_refuses(outside, content, refusal):
    module = outside['module']
    (outside['saved']/'supervisor_control.json').write_bytes(content)
    assert any(refusal in r for r in module.refusals())


def test_a_refused_job_waits_for_memory_instead_of_being_skipped(outside, monkeypatch):
    module = outside['module']
    answers = iter([(False, 'REFUSED (test)'), (False, 'REFUSED (test)'), (True, 'ALLOWED (test)')])
    monkeypatch.setattr(module, 'guard', lambda need: next(answers))
    monkeypatch.setattr(module.time, 'sleep', lambda s: None)
    job_file(outside, 'print("job")')
    assert module.main() == 0
    row = receipt_of(outside)['jobs'][0]
    assert row['status'] == 'done' and row['guard_waited_s'] == 60 and row['guard'] == 'ALLOWED (test)'


def test_a_job_the_guard_never_allows_is_recorded_as_refused_after_its_wait(outside, monkeypatch):
    module = outside['module']
    monkeypatch.setattr(module, 'guard', lambda need: (False, 'REFUSED (test)'))
    monkeypatch.setattr(module.time, 'sleep', lambda s: None)
    jobs = outside['tmp']/'jobs_wait.json'
    write(jobs, [dict(name='probe', cwd='.', need_gb=2, timeout_s=60, guard_wait_s=90, cmd=['PY', '-c', 'print(1)'])])
    sys.argv[:] = ['run_in_training_pause.py', '--jobs', str(jobs)]
    assert module.main() == 0
    row = receipt_of(outside)['jobs'][0]
    assert row['status'] == 'refused by the memory guard' and row['guard_waited_s'] == 90


def test_an_unconsumed_trainer_control_refuses(outside):
    write(outside['run']/'control.json', {'cmd': 'video'})
    assert 'the active run holds an unconsumed control.json' in outside['module'].refusals()


def _dead_pid():
    import psutil
    pid = 4_000_001
    while psutil.pid_exists(pid):
        pid += 7
    return pid


def test_the_heavy_lock_admits_one_heavy_job_and_clears_a_dead_holder(runner, monkeypatch):
    """2026-09-26: at 6:15 PM a 27 GB trial arm was admitted by the guard, other runners' jobs started beside it and
    it stopped itself at the reserve. Jobs of HEAVY_GB or more now hold one lock for their whole run."""
    import os
    module = runner['module']
    lock = runner['tmp']/'heavy_job.lock'
    assert module.acquire_heavy_lock('a', lock=lock, pid=os.getppid())
    assert not module.acquire_heavy_lock('b', lock=lock, pid=os.getpid())      # a live holder keeps it
    module.release_heavy_lock(lock=lock, pid=os.getpid())                        # not ours: kept
    assert lock.exists()
    module.release_heavy_lock(lock=lock, pid=os.getppid())
    assert not lock.exists()
    lock.write_text(json.dumps(dict(pid=_dead_pid(), job='gone', started=0)), encoding='utf-8')
    assert module.acquire_heavy_lock('c', lock=lock, pid=os.getpid())           # a dead holder is cleared
    assert json.loads(lock.read_text(encoding='utf-8'))['job'] == 'c'


def test_a_heavy_job_waits_for_the_lock_and_releases_it(runner, monkeypatch):
    import os
    module = runner['module']
    lock = runner['tmp']/'heavy_job.lock'
    monkeypatch.setattr(module, 'HEAVY_LOCK', lock)
    folder = runner['tmp']/'pause'
    folder.mkdir()
    job = dict(name='big', cwd='.', need_gb=12, guard_wait_s=0, timeout_s=60, cmd=['PY', '-c', 'print(1)'])
    lock.write_text(json.dumps(dict(pid=os.getppid(), job='other', started=0)), encoding='utf-8')
    receipt = dict(jobs=[])
    module.run_jobs([job], runner['checkpoint'], runner['run'], folder, receipt)
    assert receipt['jobs'][0]['status'] == 'refused: another heavy job held the machine'
    assert json.loads(lock.read_text(encoding='utf-8'))['job'] == 'other'       # the holder's lock untouched
    lock.unlink()
    receipt = dict(jobs=[])
    module.run_jobs([job, dict(job, name='light', need_gb=2)], runner['checkpoint'], runner['run'], folder, receipt)
    assert [r['status'] for r in receipt['jobs']] == ['done', 'done'] and not lock.exists()


def test_a_lighter_job_waits_while_a_heavy_job_holds_the_machine(runner):
    import subprocess
    import time as _time
    module = runner['module']
    folder = runner['tmp']/'pause_light'
    folder.mkdir()
    light = dict(name='light', cwd='.', need_gb=2, guard_wait_s=0, timeout_s=60, cmd=['PY', '-c', 'print(1)'])
    # the holder is a live process outside this test's parents (a lock held by a parent is the caller's own)
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        _time.sleep(0.3)
        module.HEAVY_LOCK.write_text(json.dumps(dict(pid=other.pid, job='arm', started=0)), encoding='utf-8')
        receipt = dict(jobs=[])
        module.run_jobs([light], runner['checkpoint'], runner['run'], folder, receipt)
    finally:
        _end_family(other)
    assert receipt['jobs'][0]['status'] == 'refused: a heavy job held the machine'
    module.HEAVY_LOCK.write_text(json.dumps(dict(pid=_dead_pid(), job='arm', started=0)), encoding='utf-8')
    receipt = dict(jobs=[])
    module.run_jobs([light], runner['checkpoint'], runner['run'], folder, receipt)
    assert receipt['jobs'][0]['status'] == 'done'                               # a dead holder does not block


def test_a_timed_out_job_ends_its_whole_process_tree(runner):
    """The starter test's review (2026-09-26): on a timeout subprocess.run ended only its own child (the pause
    watchdog) and left the trainer under it running after the heavy-job lock was released."""
    import time as _time
    import psutil
    module = runner['module']
    folder = runner['tmp']/'pause_timeout'
    folder.mkdir()
    pidfile = runner['tmp']/'grandchild.pid'
    child = ("import subprocess, sys, time\n"
             "subprocess.Popen([sys.executable, '-c', \"import os, pathlib, sys, time; "
             "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(120)\", sys.argv[1]])\n"
             "time.sleep(120)\n")
    job = dict(name='slow', cwd='.', need_gb=12, timeout_s=8, cmd=['PY', '-c', child, str(pidfile)])
    receipt = dict(jobs=[])
    module.run_jobs([job], runner['checkpoint'], runner['run'], folder, receipt)
    assert receipt['jobs'][0]['status'] == 'timed out' and receipt['jobs'][0]['ended'] >= 2
    deadline = _time.time() + 10
    while not pidfile.exists() and _time.time() < deadline:
        _time.sleep(0.2)
    grandchild = int(pidfile.read_text())
    assert not psutil.pid_exists(grandchild) or psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE
    assert not module.HEAVY_LOCK.exists()


def _end_family(*procs):
    """End each stand-in and its children (the venv's python.exe starts the real interpreter as a child)."""
    import psutil
    family = []
    for proc in procs:
        try:
            parent = psutil.Process(proc.pid)
            family += parent.children(recursive=True) + [parent]
        except psutil.NoSuchProcess:
            continue
    for p in family:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(family, timeout=10)


def test_the_lock_stays_held_while_its_job_runs_after_the_runner_ends(runner):
    """The starter test's re-review (2026-09-26, section 5): a runner that ended alone left its job's trainer running
    while the lock read as free. The lock now names the job's process too and is held while either one lives."""
    import subprocess
    import time as _time
    module = runner['module']
    lock = runner['tmp']/'heavy_job.lock'
    sleep = [sys.executable, '-c', 'import time; time.sleep(60)']
    runner_proc, job_proc = subprocess.Popen(sleep), subprocess.Popen(sleep)
    try:
        _time.sleep(0.3)
        assert module.acquire_heavy_lock('arm', lock=lock, pid=runner_proc.pid, need_gb=27)
        assert not module.name_heavy_job_process(job_proc.pid, lock=lock, pid=4242)   # only the holder names its job
        assert module.name_heavy_job_process(job_proc.pid, lock=lock, pid=runner_proc.pid)
        held = json.loads(lock.read_text(encoding='utf-8'))
        assert held['job_pid'] == job_proc.pid and held['need_gb'] == 27 and held['job_pid_created']
        _end_family(runner_proc)                                                       # the runner ends alone
        assert module.heavy_lock_held_by_other(lock=lock)                              # its job still runs
        assert not module.acquire_heavy_lock('next', lock=lock)
    finally:
        _end_family(runner_proc, job_proc)
    assert not module.heavy_lock_held_by_other(lock=lock)                              # both ended: free
    assert module.acquire_heavy_lock('next', lock=lock)


def test_a_reused_process_number_does_not_hold_the_lock(runner):
    import os
    module = runner['module']
    lock = runner['tmp']/'heavy_job.lock'
    # a live process whose creation time differs from the one recorded: the number was reused after the holder ended
    lock.write_text(json.dumps(dict(pid=os.getppid(), pid_created=1.0, job='old', started=0)), encoding='utf-8')
    assert not module.heavy_lock_held_by_other(lock=lock)
    assert module.acquire_heavy_lock('new', lock=lock)


def test_a_heavy_job_runs_with_its_own_process_and_need_in_the_lock(runner):
    module = runner['module']
    folder = runner['tmp']/'pause_named'
    folder.mkdir()
    child = ("import os, sys, time\n"
             "time.sleep(1.5)\n"
             "print(open(sys.argv[1], encoding='utf-8').read())\n"
             "print('PID', os.getpid(), os.getppid())\n")
    job = dict(name='arm', cwd='.', need_gb=12, timeout_s=60, cmd=['PY', '-c', child, str(module.HEAVY_LOCK)])
    receipt = dict(jobs=[])
    module.run_jobs([job], runner['checkpoint'], runner['run'], folder, receipt)
    assert receipt['jobs'][0]['status'] == 'done'
    text = (folder/'arm.log').read_text(encoding='utf-8')
    held = json.loads(text[:text.rindex('PID')])                    # the lock as the job read it (indented JSON)
    pids = [int(x) for x in text[text.rindex('PID'):].split()[1:]]
    assert held['job'] == 'arm' and held['need_gb'] == 12
    assert held['job_pid'] in pids          # the job itself, or the venv launcher that started the interpreter
    assert not module.HEAVY_LOCK.exists()


def test_a_trial_trainer_is_named_by_its_own_suffix_and_the_queue_waiter_is_not_one():
    """9:29 PM PDT 2026-09-26: machine_busy's first form matched any "*_trainer.py", including the queues' own
    wait_no_trainer.py, so every waiter saw itself and the others as trainers and all waited with the machine free."""
    spec = importlib.util.spec_from_file_location(
        'held_pause_runner_under_test', REPO/'docs/gain_of_function_v4/coordinator/run_jobs_in_held_pause.py')
    held = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(held)
    assert held.is_trial_trainer('C:/Users/x/docs/gain_of_function_v4/persistent_memory_trial_trainer.py')
    assert not held.is_trial_trainer('docs/gain_of_function_v4/coordinator/wait_no_trainer.py')
    assert not held.is_trial_trainer('mimo_bridge/tests/test_persistent_memory_trial_trainer.py')
    assert not held.is_trial_trainer(None)


def test_a_heavy_job_records_its_peak_in_the_lock_while_it_runs(runner, monkeypatch):
    module = runner['module']
    monkeypatch.setattr(module, 'PEAK_POLL_S', 0.2)
    folder = runner['tmp']/'pause_peak'
    folder.mkdir()
    child = ("import sys, time\n"
             "block = bytearray(300 * 1024 * 1024)\n"      # 300 MiB held, then released
             "time.sleep(2.0)\n"
             "del block\n"
             "time.sleep(0.6)\n"
             "print(open(sys.argv[1], encoding='utf-8').read())\n")
    job = dict(name='arm', cwd='.', need_gb=12, timeout_s=60, cmd=['PY', '-c', child, str(module.HEAVY_LOCK)])
    receipt = dict(jobs=[])
    module.run_jobs([job], runner['checkpoint'], runner['run'], folder, receipt)
    assert receipt['jobs'][0]['status'] == 'done'
    held = json.loads((folder/'arm.log').read_text(encoding='utf-8'))
    assert held['peak_gb'] >= 0.25                          # the 300 MiB block, seen while it was held
    assert not module.HEAVY_LOCK.exists()


def test_a_reservation_gives_the_next_heavy_turns_to_the_named_jobs_until_it_lapses(runner):
    import os
    import time as _time
    module = runner['module']
    lock = module.HEAVY_LOCK
    reservation = lock.with_name('heavy_job.next')
    write(reservation, dict(prefix='memory_starter', until=_time.time() + 600))
    assert not module.acquire_heavy_lock('eye_fix5_jump', lock=lock, pid=os.getppid())       # another job waits
    assert module.acquire_heavy_lock('memory_starter_smoke', lock=lock, pid=os.getppid())    # the reserved job goes
    module.release_heavy_lock(lock=lock, pid=os.getppid())
    write(reservation, dict(prefix='memory_starter', until=_time.time() - 1))                  # lapsed: the safety net
    assert module.acquire_heavy_lock('eye_fix5_jump', lock=lock, pid=os.getppid())
    module.release_heavy_lock(lock=lock, pid=os.getppid())
    reservation.unlink()
    assert module.acquire_heavy_lock('eye_fix5_jump', lock=lock, pid=os.getppid())          # none: as before


def test_a_queue_waits_out_a_reservation_for_other_jobs_but_not_its_own(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'held_pause_runner_busy_under_test', REPO/'docs/gain_of_function_v4/coordinator/run_jobs_in_held_pause.py')
    held = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(held)
    monkeypatch.setattr(held, 'trainer_running', lambda: False)
    monkeypatch.setattr(held, 'trial_trainer_running', lambda: False)
    monkeypatch.setattr(held.rtp, 'heavy_lock_held_by_other', lambda: False)
    monkeypatch.setattr(held.rtp, 'heavy_lock_is_own', lambda: False)
    monkeypatch.setattr(held.rtp, 'runner_turn_held_by_other', lambda: False)   # never the machine's real turn
    monkeypatch.setattr(held.rtp, 'runner_turn_is_own', lambda: False)
    monkeypatch.setattr(held.rtp, 'next_reservation', lambda: {'prefix': 'memory_starter', 'until': 1e12})
    assert 'reserved for memory_starter' in held.machine_busy()                 # a queue with other jobs waits
    assert 'reserved' in held.machine_busy('eye_fix5')
    assert held.machine_busy('memory_starter') is None                           # the reserved jobs' own runner goes
    monkeypatch.setattr(held.rtp, 'heavy_lock_is_own', lambda: True)
    assert held.machine_busy() is None                  # a job inside its own heavy turn: the reserved jobs wait on it
    monkeypatch.setattr(held.rtp, 'heavy_lock_is_own', lambda: False)
    monkeypatch.setattr(held.rtp, 'next_reservation', lambda: None)
    assert held.machine_busy() is None


def test_a_lock_held_by_the_callers_own_runner_is_not_another_jobs(runner):
    """2026-09-27: the skin check's test job, run under the runner that held the lock for it, read the lock as another
    job's and waited 10,800 s on itself. A lock held by the caller or one of its parents is its own."""
    import os
    import subprocess
    import time as _time
    module = runner['module']
    lock = runner['tmp']/'heavy_job.lock'
    lock.write_text(json.dumps(dict(pid=os.getppid(), job='skin_off_fixed_rays_tests', started=0)), encoding='utf-8')
    assert not module.heavy_lock_held_by_other(lock=lock)                  # the caller's parent holds it: its own
    assert module.heavy_lock_is_own(lock=lock)
    lock.write_text(json.dumps(dict(pid=4_000_001, job_pid=os.getpid(), job='x', started=0)), encoding='utf-8')
    assert not module.heavy_lock_held_by_other(lock=lock)                  # the job named in the lock is the caller
    assert module.heavy_lock_is_own(lock=lock)
    lock.write_text(json.dumps(dict(pid=os.getppid(), pid_created=1.0, job='old', started=0)), encoding='utf-8')
    assert not module.heavy_lock_is_own(lock=lock)                         # a parent's reused number is not its own
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        _time.sleep(0.3)
        lock.write_text(json.dumps(dict(pid=other.pid, job='arm', started=0)), encoding='utf-8')
        assert module.heavy_lock_held_by_other(lock=lock)                  # another live runner: busy
    finally:
        _end_family(other)


# ONE RUNNER'S JOBS FILE AT A TIME (2026-09-27): the motor confidence gate lost 7 of its 13 jobs on Sep 26 when another
# runner's heavy job started between its jobs and its lighter jobs gave up after their 900 s wait.
def test_a_runner_holds_the_turn_for_its_whole_jobs_file_and_releases_it(runner):
    module = runner['module']
    turn = runner['tmp']/'runner_turn.lock'
    assert module.runner_turn_path() == turn                        # beside the (test's) heavy-job lock
    seen = runner['tmp']/'seen.txt'
    jobs = runner['tmp']/'jobs_turn.json'
    look = 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(str(pathlib.Path(sys.argv[2]).exists()))'
    write(jobs, [dict(name='look', cwd='.', need_gb=2, timeout_s=60, cmd=['PY', '-c', look, str(seen), str(turn)])])
    sys.argv[:] = ['run_in_training_pause.py', '--jobs', str(jobs)]
    assert module.main() == 0
    assert seen.read_text() == 'True'                                # held while the job ran
    assert not turn.exists()                                         # released at the end


def test_the_runner_turn_is_one_runners_and_its_jobs_see_it_as_their_own(runner):
    import os
    import subprocess
    import time as _time
    module = runner['module']
    turn = module.runner_turn_path()
    assert module.acquire_runner_turn('a.json', pid=os.getppid())
    assert not module.runner_turn_held_by_other() and module.runner_turn_is_own()   # a job under that runner
    assert not module.acquire_runner_turn('b.json')                  # a second runner waits
    module.release_runner_turn()                                     # not this process's: kept
    assert turn.exists()
    module.release_runner_turn(pid=os.getppid())
    assert not turn.exists()
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        _time.sleep(0.3)
        turn.write_text(json.dumps(dict(pid=other.pid, job='other.json', started=0)), encoding='utf-8')
        assert module.runner_turn_held_by_other() and not module.runner_turn_is_own()
        assert not module.acquire_runner_turn('b.json')
    finally:
        _end_family(other)
    turn.write_text(json.dumps(dict(pid=_dead_pid(), job='other.json', started=0)), encoding='utf-8')
    assert not module.runner_turn_held_by_other()                    # its runner ended: free
    assert module.acquire_runner_turn('b.json')
    assert json.loads(turn.read_text(encoding='utf-8'))['job'] == 'b.json'


def test_a_runner_waits_for_the_turn_and_for_a_reservation_of_other_jobs(runner, monkeypatch):
    import time as _time
    module = runner['module']
    turn = module.runner_turn_path()
    reservation = module.HEAVY_LOCK.with_name('heavy_job.next')
    write(turn, dict(pid=4_000_001, job='gone.json', started=0))     # a holder that has ended does not block
    assert module.take_runner_turn('a.json', 'eye_fix5_jump', poll_s=0) == 0
    module.release_runner_turn()
    write(reservation, dict(prefix='memory_starter', until=_time.time() + 600))
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            reservation.unlink()                                     # the reserved runner finished and cleared it

    monkeypatch.setattr(module.time, 'sleep', sleep)
    assert module.take_runner_turn('b.json', 'eye_fix5_jump', poll_s=30) == 90   # waited out the reservation
    module.release_runner_turn()
    write(reservation, dict(prefix='memory_starter', until=_time.time() + 600))
    assert module.take_runner_turn('c.json', 'memory_starter_smoke', poll_s=30) == 0   # the reserved runner goes


def test_the_queue_waiter_waits_for_another_runners_turn_but_not_its_own(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'held_pause_runner_turn_under_test', REPO/'docs/gain_of_function_v4/coordinator/run_jobs_in_held_pause.py')
    held = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(held)
    monkeypatch.setattr(held, 'trainer_running', lambda: False)
    monkeypatch.setattr(held, 'trial_trainer_running', lambda: False)
    monkeypatch.setattr(held.rtp, 'heavy_lock_held_by_other', lambda: False)
    monkeypatch.setattr(held.rtp, 'heavy_lock_is_own', lambda: False)
    monkeypatch.setattr(held.rtp, 'next_reservation', lambda: None)
    monkeypatch.setattr(held.rtp, 'runner_turn_held_by_other', lambda: True)
    monkeypatch.setattr(held.rtp, '_lock_holder', lambda path: {'job': 'pause_jobs_x.json'})
    assert 'another runner has its turn (pause_jobs_x.json)' in held.machine_busy()
    monkeypatch.setattr(held.rtp, 'runner_turn_held_by_other', lambda: False)
    monkeypatch.setattr(held.rtp, 'runner_turn_is_own', lambda: True)
    monkeypatch.setattr(held.rtp, 'next_reservation', lambda: {'prefix': 'memory_starter', 'until': 1e12})
    assert held.machine_busy() is None                               # a job inside its own runner's turn
