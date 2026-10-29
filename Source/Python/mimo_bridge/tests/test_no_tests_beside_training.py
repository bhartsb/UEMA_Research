# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The test settings stop a session while production training runs (conftest._live_trainer_pids), and while any
trainer process runs, recorded by the supervisor or not (conftest._train_mimo_processes, 2026-09-26), a trial trainer
script included, and while a runner's heavy job holds the machine (conftest._heavy_job_holder, 2026-09-26 evening)."""
import json
import subprocess
import sys
import time

import psutil
import pytest

import conftest
from conftest import _heavy_job_holder, _live_trainer_pids, _train_mimo_processes

# stand-in names, so the supervisor, the watchers, the runners and other sessions never take these processes for trainers
WRAPPER = "_uema_stand_in_wrapper.py"


def test_a_recorded_trainer_that_is_alive_and_runs_train_mimo_is_found(tmp_path):
    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "uema_stand_in_trainer"])
    try:
        time.sleep(0.5)
        rec = tmp_path / "training_continuation.json"
        rec.write_text(json.dumps({"trainer_pids": [fake.pid]}), encoding="utf-8")
        # a stand-in word, so the supervisor and watchers never take this process for a trainer
        assert _live_trainer_pids(str(rec), marker="uema_stand_in_trainer") == [fake.pid]
    finally:
        fake.kill()
        fake.wait(10)
    # the same record once that process has ended: nothing runs, so tests may start
    assert _live_trainer_pids(str(rec), marker="uema_stand_in_trainer") == []


def test_a_recorded_process_that_is_not_a_trainer_or_an_unreadable_record_blocks_nothing(tmp_path):
    rec = tmp_path / "training_continuation.json"
    rec.write_text(json.dumps({"trainer_pids": [__import__("os").getpid()]}), encoding="utf-8")
    assert _live_trainer_pids(str(rec)) == []
    assert _live_trainer_pids(str(tmp_path / "missing.json")) == []


def _end_family(procs):
    """End each stand-in and its children (the venv's python.exe starts the real interpreter as a child)."""
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


def test_any_python_process_running_the_trainer_is_found_whether_recorded_or_not():
    # a stand-in module name, so the supervisor, the watchers and other sessions never take these for a trainer
    module = "uema_stand_in.trainer_module"
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    as_module = subprocess.Popen(sleep + ["-m", module, "--resume", "x"])           # like python -m <trainer>
    as_script = subprocess.Popen(sleep + ["D:\\x\\uema_stand_in\\trainer_module.py"])  # like python <trainer path>
    look_alike = subprocess.Popen(sleep + ["tests/test_uema_stand_in_trainer_module.py", "-k", module + "_case"])
    try:
        time.sleep(0.5)
        found = set(_train_mimo_processes(module, wrapper=WRAPPER))
        assert as_module.pid in found and as_script.pid in found
        assert look_alike.pid not in found              # a test file or a -k word only contains the name
    finally:
        _end_family([as_module, as_script, look_alike])
    # once they have ended nothing runs the stand-in, so tests may start
    assert _train_mimo_processes(module, wrapper=WRAPPER) == []


def test_the_session_stops_while_any_trainer_process_runs_unless_told_it_is_light(monkeypatch):
    monkeypatch.delenv("UEMA_TESTS_BESIDE_TRAINING", raising=False)
    monkeypatch.setattr(conftest, "_live_trainer_pids", lambda: [])        # the supervisor records no trainer
    monkeypatch.setattr(conftest, "_heavy_job_holder", lambda: None)       # and no runner holds the heavy-job lock
    monkeypatch.setattr(conftest, "_train_mimo_processes", lambda: [4242])  # but a trial arm runs
    with pytest.raises(pytest.exit.Exception) as stop:
        conftest.pytest_sessionstart(None)
    assert stop.value.returncode == 4
    assert stop.value.msg.startswith("training is not paused; not starting") and "4242" in stop.value.msg
    monkeypatch.setenv("UEMA_TESTS_BESIDE_TRAINING", "1")
    conftest.pytest_sessionstart(None)                  # the override for a test measured as light still starts
    monkeypatch.delenv("UEMA_TESTS_BESIDE_TRAINING")
    monkeypatch.setattr(conftest, "_train_mimo_processes", lambda: [])
    conftest.pytest_sessionstart(None)                  # no trainer: the session starts


def test_a_trial_trainer_script_is_found_but_a_test_file_of_the_same_name_is_not(tmp_path):
    # the memory trial's arm trains inside persistent_memory_trial_trainer.py; no argument names train_mimo
    wrapper = tmp_path / ("persistent_memory" + WRAPPER)
    look_alike = tmp_path / ("test_persistent_memory" + WRAPPER)
    for script in (wrapper, look_alike):
        script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    trial = subprocess.Popen([sys.executable, "-u", str(wrapper), "--liveness-revival", "off"])
    test = subprocess.Popen([sys.executable, str(look_alike)])
    code = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(wrapper)])  # names it, runs -c
    try:
        time.sleep(0.5)
        found = set(_train_mimo_processes("uema_stand_in.none", wrapper=WRAPPER))
        assert trial.pid in found
        assert test.pid not in found and code.pid not in found
    finally:
        _end_family([trial, test, code])
    assert _train_mimo_processes("uema_stand_in.none", wrapper=WRAPPER) == []


def test_the_queues_waiter_is_not_a_trial_trainer(tmp_path):
    # 9:29 PM PDT 2026-09-26: the first form matched any "*_trainer.py", and the queues' own wait_no_trainer.py with it
    waiter = tmp_path / "wait_no_trainer.py"
    waiter.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(waiter)])
    try:
        time.sleep(0.5)
        assert proc.pid not in set(_train_mimo_processes("uema_stand_in.none"))     # the default wrapper name
    finally:
        _end_family([proc])


def test_the_heavy_job_lock_blocks_a_session_unless_its_holder_runs_this_session(tmp_path):
    lock = tmp_path / "heavy_job.lock"
    assert _heavy_job_holder(str(lock)) is None                            # no lock
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(0.5)
        lock.write_text(json.dumps({"pid": other.pid, "job": "arm", "started": 0}), encoding="utf-8")
        assert _heavy_job_holder(str(lock))["job"] == "arm"               # another live process holds it
    finally:
        _end_family([other])
    assert _heavy_job_holder(str(lock)) is None                            # its holder has ended
    import os
    for mine in (os.getpid(), os.getppid()):                               # this session, or the runner that started it
        lock.write_text(json.dumps({"pid": mine, "job": "tests", "started": 0}), encoding="utf-8")
        assert _heavy_job_holder(str(lock)) is None
    lock.write_text("not json", encoding="utf-8")
    assert _heavy_job_holder(str(lock)) is None


def test_the_session_stops_while_a_heavy_job_holds_the_machine_unless_told_it_is_light(monkeypatch):
    monkeypatch.delenv("UEMA_TESTS_BESIDE_TRAINING", raising=False)
    monkeypatch.setattr(conftest, "_live_trainer_pids", lambda: [])
    monkeypatch.setattr(conftest, "_train_mimo_processes", lambda: [])    # no command line names a trainer
    monkeypatch.setattr(conftest, "_heavy_job_holder", lambda: {"pid": 4343, "job": "driver_trains_in_process"})
    with pytest.raises(pytest.exit.Exception) as stop:
        conftest.pytest_sessionstart(None)
    assert stop.value.returncode == 4 and "4343" in stop.value.msg and "driver_trains_in_process" in stop.value.msg
    monkeypatch.setenv("UEMA_TESTS_BESIDE_TRAINING", "1")
    conftest.pytest_sessionstart(None)


def test_a_heavy_job_whose_runner_ended_still_blocks_a_session(tmp_path):
    # the lock names the job's own process too (2026-09-26): a job left running by a runner that ended alone still
    # holds the machine; a lock whose recorded creation time no longer matches is a reused number
    import os
    lock = tmp_path / "heavy_job.lock"
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(0.5)
        lock.write_text(json.dumps({"pid": 4_000_001, "job_pid": job.pid, "job": "arm"}), encoding="utf-8")
        assert _heavy_job_holder(str(lock))["job"] == "arm"
        lock.write_text(json.dumps({"pid": 4_000_001, "job_pid": job.pid, "job_pid_created": 1.0, "job": "arm"}),
                        encoding="utf-8")
        assert _heavy_job_holder(str(lock)) is None
        lock.write_text(json.dumps({"pid": 4_000_001, "job_pid": os.getppid(), "job": "tests"}), encoding="utf-8")
        assert _heavy_job_holder(str(lock)) is None                        # the job that runs this session
    finally:
        _end_family([job])
