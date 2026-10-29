# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""tools/commit_guard never counts on the page file growing into the last DISK_RESERVE_GIB of its drive.

2026-09-25: the guard judged against Windows' current commit limit, which rises every time the page file
grows; with the relaunched trainer and a dozen agents' work running, the page file grew from 40,000 to
58,833 MB and C: fell to 3.6 GB free while every guard reading said ALLOWED. These numbers are the
machine's own: 29.71 GiB of RAM, c:\\pagefile.sys 40000 60000 (MB), 73.8 GiB Windows limit, 67.2 GiB
committed with training running.
"""
from mimo_bridge.tools import commit_guard as G

RAM = 29.71
PAGES = [("C:/", 40000 / 1024.0, 60000 / 1024.0)]


def test_with_room_on_the_drive_windows_current_limit_stands():
    limit = G.effective_limit(73.8, RAM, PAGES, free_of=lambda d: 28.8)
    assert abs(limit - 73.8) < 1e-9


def test_a_filling_drive_lowers_the_limit_to_what_it_can_still_hold():
    # 14 GiB free leaves 2 GiB above the 12 GiB reserve: boot size (29.71 + 39.06) plus 2
    limit = G.effective_limit(73.8, RAM, PAGES, free_of=lambda d: 14.0)
    assert abs(limit - (RAM + 40000 / 1024.0 + 2.0)) < 1e-9
    assert not G.verdict(67.2, limit, 2.0, 6.0), "a light test beside training is refused when C: is filling"


def test_growth_never_counts_past_the_page_files_largest_size():
    limit = G.effective_limit(200.0, RAM, PAGES, free_of=lambda d: 500.0)
    assert abs(limit - (RAM + 60000 / 1024.0)) < 1e-9


def test_a_system_managed_page_file_keeps_windows_limit():
    assert G.effective_limit(73.8, RAM, None, free_of=lambda d: 1.0) == 73.8


def test_a_drive_under_the_reserve_refuses_all_work(monkeypatch):
    monkeypatch.setattr(G, "heavy_job_extra_gb", lambda: (0.0, None))   # no heavy job running, whatever the machine holds
    monkeypatch.setattr(G, "_performance", lambda: (10.0, 73.8, 20.0, RAM))
    monkeypatch.setattr(G, "page_files", lambda: PAGES)
    monkeypatch.setattr(G, "disk_free_gib", lambda d: 5.0)
    assert G.disk_short(PAGES, free_of=lambda d: 5.0) == [("C:/", 5.0)]
    assert G.main(["--need-gb", "1"]) == 3, "memory would fit, but C: is under its reserve"
    monkeypatch.setattr(G, "disk_free_gib", lambda d: 40.0)
    assert G.main(["--need-gb", "1"]) == 0


def test_commit_reading_keeps_its_three_numbers_for_every_caller(monkeypatch):
    monkeypatch.setattr(G, "_performance", lambda: (67.2, 73.8, 7.0, RAM))
    monkeypatch.setattr(G, "page_files", lambda: PAGES)
    monkeypatch.setattr(G, "disk_free_gib", lambda d: 14.0)
    committed, limit, free_ram = G.commit_reading()
    assert committed == 67.2 and free_ram == 7.0 and limit < 73.8


# A running heavy job's unused declared need stays free (2026-09-26, the heads repair's re-review, finding m6): agents'
# light readings, each allowed, took the machine from 52.8 to 59.5 GB committed beside a trial arm whose stop line was
# 62.8 GB. These tests use their own lock file, never the runners' real one.

def _sleeper():
    import subprocess
    import sys
    import time
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    time.sleep(0.3)
    return proc


def _end(proc):
    import psutil
    try:
        family = psutil.Process(proc.pid).children(recursive=True) + [psutil.Process(proc.pid)]
    except psutil.NoSuchProcess:
        return
    for p in family:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(family, timeout=10)


def test_another_runners_heavy_job_keeps_its_unused_need_free(tmp_path):
    import json
    lock = tmp_path / "heavy_job.lock"
    assert G.heavy_job_extra_gb(str(lock)) == (0.0, None)                     # no lock
    job = _sleeper()
    try:
        lock.write_text(json.dumps({"pid": job.pid, "job_pid": job.pid, "job": "arm", "need_gb": 5.0}),
                        encoding="utf-8")
        extra, holder = G.heavy_job_extra_gb(str(lock))
        assert holder["job"] == "arm" and 4.5 < extra <= 5.0                  # a sleeping python commits little of 5
        lock.write_text(json.dumps({"pid": job.pid, "job_pid": job.pid, "job": "old runner"}), encoding="utf-8")
        assert G.heavy_job_extra_gb(str(lock))[0] == 0.0                      # a lock that declares no need
    finally:
        _end(job)
    lock.write_text(json.dumps({"pid": job.pid, "job_pid": job.pid, "job": "arm", "need_gb": 5.0}), encoding="utf-8")
    assert G.heavy_job_extra_gb(str(lock)) == (0.0, None)                     # its processes have ended


def test_a_heavy_job_still_waiting_for_its_admission_holds_back_nothing(tmp_path):
    """2026-09-27: the 37 GB drift trial held the lock while it waited for the guard (it could never fit beside the
    machine's 35 GB), and its whole need, reserved, refused every agent's light job from 7:31 to 8:51 AM. Until the
    runner names the job's process in the lock, the job has not started and has no room to keep."""
    import json
    lock = tmp_path / "heavy_job.lock"
    runner = _sleeper()
    job = _sleeper()
    try:
        lock.write_text(json.dumps({"pid": runner.pid, "job": "token_drift_trial", "need_gb": 37.0}), encoding="utf-8")
        extra, holder = G.heavy_job_extra_gb(str(lock))
        assert extra == 0.0 and holder["job"] == "token_drift_trial"          # waiting: nothing reserved
        lock.write_text(json.dumps({"pid": runner.pid, "job_pid": job.pid, "job": "token_drift_trial",
                                    "need_gb": 37.0}), encoding="utf-8")
        assert 36.5 < G.heavy_job_extra_gb(str(lock))[0] <= 37.0              # started: its room is kept free
    finally:
        _end(job)
        _end(runner)


def test_the_heavy_jobs_own_guard_calls_count_nothing_extra(tmp_path):
    import json
    import os
    lock = tmp_path / "heavy_job.lock"
    for mine in (os.getpid(), os.getppid()):                                  # the job itself, or its runner above it
        lock.write_text(json.dumps({"pid": 4242, "job_pid": mine, "job": "arm", "need_gb": 27.0}), encoding="utf-8")
        assert G.heavy_job_extra_gb(str(lock))[0] == 0.0
    # a live process whose recorded creation time differs is a reused number, not the holder
    lock.write_text(json.dumps({"pid": os.getppid(), "pid_created": 1.0, "job": "arm", "need_gb": 27.0}),
                    encoding="utf-8")
    assert G.heavy_job_extra_gb(str(lock)) == (0.0, None)


def test_the_guard_refuses_work_that_would_eat_a_running_heavy_jobs_room(monkeypatch):
    monkeypatch.setattr(G, "_performance", lambda: (55.0, 68.8, 5.0, RAM))
    monkeypatch.setattr(G, "page_files", lambda: None)
    monkeypatch.setattr(G, "heavy_job_extra_gb", lambda: (6.0, {"job": "arm", "need_gb": 27.0}))
    assert G.main(["--need-gb", "2"]) == 3            # 55 + 2 + 6 = 63 > 68.8 - 6
    monkeypatch.setattr(G, "heavy_job_extra_gb", lambda: (0.0, None))
    assert G.main(["--need-gb", "2"]) == 0            # 55 + 2 = 57: allowed with no heavy job running


def test_a_heavy_job_past_its_peak_holds_back_only_the_need_above_that_peak(tmp_path):
    # the memory control run's end readings: its trainer peaked, ended, and the guard still held back 23.7 GB for it
    import json
    lock = tmp_path / "heavy_job.lock"
    job = _sleeper()
    try:
        lock.write_text(json.dumps({"pid": job.pid, "job_pid": job.pid, "job": "arm", "need_gb": 5.0, "peak_gb": 4.6}),
                        encoding="utf-8")
        extra, _holder = G.heavy_job_extra_gb(str(lock))
        assert abs(extra - 0.4) < 0.01                       # 5.0 declared, 4.6 already reached at its peak
        lock.write_text(json.dumps({"pid": job.pid, "job_pid": job.pid, "job": "arm", "need_gb": 5.0, "peak_gb": 6.0}),
                        encoding="utf-8")
        assert G.heavy_job_extra_gb(str(lock))[0] == 0.0     # a peak past the declared need holds back nothing
    finally:
        _end(job)
