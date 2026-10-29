# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Make Source/Python importable (tokenizer, lem, mimo_bridge) for pytest."""
import os
import sys

SRC_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_PY not in sys.path:
    sys.path.insert(0, SRC_PY)

# PYTEST'S TEMPORARY FOLDERS LIVE ON D: WHEN IT IS THERE (2026-09-25). Tests that start trainer micro-runs
# write checkpoints into tmp_path; with a dozen agents testing at once C: fell to 6.9 GB free at 7:14 PM PDT,
# under the H84 hourly watcher's 8 GB training stop, and C: also holds Windows' page file. pytest reads
# PYTEST_DEBUG_TEMPROOT when it first makes a temporary folder, after this file is imported. Set it yourself
# to choose another place.
if "PYTEST_DEBUG_TEMPROOT" not in os.environ and os.path.isdir("D:/"):
    try:
        os.makedirs("D:/UEMA_Research/external/D/UEMA_scratch/pytest", exist_ok=True)
        os.environ["PYTEST_DEBUG_TEMPROOT"] = "D:/UEMA_Research/external/D/UEMA_scratch/pytest"
    except OSError:
        pass
# git refuses a repository on D: as "dubious ownership" (the drive reports another owner), which broke the tests
# that build a temporary repository (test_ability_scheduler_classes, found 2026-09-25): mark repositories safe for
# git commands started by this test process only, never in the user's git settings
if (os.environ.get("PYTEST_DEBUG_TEMPROOT", "").upper().startswith("D:")
        and "GIT_CONFIG_COUNT" not in os.environ):
    os.environ.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*")


# NO TEST SESSION BESIDE PRODUCTION TRAINING (2026-09-25). Agent test runs building worlds ran beside the live trainer
# three times that day (3:48-3:51 PM; 8:07:53-8:08:34 PM; 8:11:33-8:11:49 PM): each checked for a pause only at
# launch, or ran a file believed light that holds a crib-world test. The memory guard already refuses guarded work
# while training runs, so every test session belongs in a training pause. A session started while a trainer the
# supervisor records is alive stops before collecting, with the pause watchdog's words, so retry loops waiting for a
# pause keep waiting. A test measured as light may still run with UEMA_TESTS_BESIDE_TRAINING=1.
# ANY TRAINER PROCESS STOPS A SESSION TOO (2026-09-26). A trainer the supervisor does not record, such as a memory trial
# arm, shares the graphics card and the memory, and since probes stopped hiding the card at import a session that
# collects the card tests uses the card too. Such a trainer is found by its command line, as the held-pause runner and
# the supervisor find one: a python process with mimo_bridge.train_mimo as an argument (the supervisor's element test,
# training_continuation.py:52-54), or with the script mimo_bridge/train_mimo.py, which most launch lines name.
# TRIAL TRAINERS AND HEAVY JOBS STOP A SESSION TOO (2026-09-26, 8:21 PM PDT, the merge review's first finding). The memory
# repair's trial arm trains inside a wrapper script (persistent_memory_trial_trainer.py) and several proofs train inside a
# driver's own process, so no command line names train_mimo, and sessions started beside the arm. A wrapper is found by
# its script's name (*_trial_trainer.py, never a test file; "*_trainer.py" also matched the queues' wait_no_trainer.py,
# 9:29 PM PDT); a driver shows only through the runners' heavy-job lock, which
# every runner job declaring 10 GB or more holds while it runs. A session that such a job runs itself (the lock's
# holder is one of its parents) still starts.
_CONTINUATION = "C:/Users/bhart/Documents/UEMA_devA_private/Saved/mimo_bridge/training_continuation.json"
_HEAVY_LOCK = "C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_observer_20260924/heavy_job.lock"


def _live_trainer_pids(path=_CONTINUATION, marker="train_mimo"):
    """The supervisor's recorded trainer processes that are alive and still run train_mimo (read-only)."""
    try:
        import json
        import psutil
        with open(path, encoding="utf-8") as f:
            pids = json.load(f).get("trainer_pids") or []
    except Exception:
        return []
    live = []
    for pid in pids:
        try:
            if marker in " ".join(psutil.Process(int(pid)).cmdline()):
                live.append(int(pid))
        except Exception:
            continue
    return live


def _script_of(cmdline):
    """The script a python command line runs: its first argument after the interpreter's own options, or None when it
    runs -c or -m."""
    rest = list(cmdline[1:])
    while rest:
        a = rest.pop(0)
        if a in ("-c", "-m"):
            return None
        if a in ("-X", "-W"):
            rest = rest[1:]
            continue
        if not a.startswith("-"):
            return a.replace("\\", "/")
    return None


def _train_mimo_processes(module="mimo_bridge.train_mimo", wrapper="_trial_trainer.py"):
    """Every python process that runs `module`, as `-m module` or as the script module.py, or whose script is a trial
    trainer (a name ending in `wrapper`, not a test file), recorded by the supervisor or not (read-only). An argument
    must equal the name or end in the script's path, so a test file or a -k word that only contains the name matches
    nothing."""
    try:
        import psutil
    except Exception:
        return []
    script = module.replace(".", "/") + ".py"
    found = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if p.info["pid"] == os.getpid() or "python" not in (p.info["name"] or "").lower():
                continue
            cmdline = p.cmdline()
        except Exception:
            continue
        args = [a.replace("\\", "/") for a in cmdline[1:]]
        runs = _script_of(cmdline) or ""
        if any(a == module or a == script or a.endswith("/" + script) for a in args) or (
                runs.endswith(wrapper) and not runs.rsplit("/", 1)[-1].startswith("test_")):
            found.append(p.info["pid"])
    return found


def _heavy_job_holder(path=_HEAVY_LOCK):
    """The runners' heavy-job lock ({pid, job, started, ...}) while its runner or the job the runner started is alive
    (the same process, not a reused number) and neither is this session or one of its parents, else None (read-only)."""
    try:
        import json
        import psutil
        with open(path, encoding="utf-8") as f:
            holder = json.load(f)
        live = set()
        for pid, created in ((holder.get("pid"), holder.get("pid_created")),
                             (holder.get("job_pid"), holder.get("job_pid_created"))):
            if pid is None:
                continue
            try:
                now = psutil.Process(int(pid)).create_time()
            except psutil.Error:
                continue
            if created is None or abs(now - float(created)) < 1.0:
                live.add(int(pid))
        if not live:
            return None
        me = psutil.Process(os.getpid())
        return None if live & ({me.pid} | {p.pid for p in me.parents()}) else holder
    except Exception:
        return None


def pytest_sessionstart(session):
    if os.environ.get("UEMA_TESTS_BESIDE_TRAINING") == "1":
        return
    pids = _live_trainer_pids()
    if pids:
        import pytest
        pytest.exit("training is not paused; not starting: production training is running (trainer %s). Run tests "
                    "inside a training pause through mimo_bridge.tools.pause_watchdog; a test measured as light may "
                    "set UEMA_TESTS_BESIDE_TRAINING=1." % pids, returncode=4)
    others = _train_mimo_processes()
    if others:
        import pytest
        pytest.exit("training is not paused; not starting: a trainer process is running (python process %s runs "
                    "mimo_bridge.train_mimo or a trial trainer script), which the supervisor may not record, such as "
                    "a memory trial arm. Run tests through mimo_bridge.tools.pause_watchdog once no trainer runs; a "
                    "test measured as light may set UEMA_TESTS_BESIDE_TRAINING=1." % others, returncode=4)
    holder = _heavy_job_holder()
    if holder:
        import pytest
        pytest.exit("training is not paused; not starting: a heavy job holds the machine (runner process %s, job %s), "
                    "which may train inside its own process where no command line names a trainer. Run tests through "
                    "mimo_bridge.tools.pause_watchdog once it ends; a test measured as light may set "
                    "UEMA_TESTS_BESIDE_TRAINING=1." % (holder.get("pid"), holder.get("job")), returncode=4)


def pytest_configure(config):
    # the end-to-end tests build a real MuJoCo body; marked so they can be deselected
    config.addinivalue_line("markers", "slow: builds a MuJoCo environment")
