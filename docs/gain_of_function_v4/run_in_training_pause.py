"""Run declared heavy jobs in ONE training pause, then resume training.

Heavy jobs - anything that builds MuJoCo worlds or loads a model: scripted validation worlds, babble
recordings, saved-state probes, browser screenshots - cannot run beside the eight-world H84 trainer: the
machine then sits inside the memory guard's 6 GB reserve, and a squeeze grows Windows' page file onto
C: (2026-09-24 1:29 PM: C: reached 0.8 GB free and stopped training). This script gathers them into one
pause, the way the operator's instructions allow downtime: for changes and evaluations.

    python docs/gain_of_function_v4/run_in_training_pause.py --jobs <jobs.json> [--check]

jobs.json is a list of {"name", "cmd": [...], "cwd" (relative to the repository, default Source/Python),
"need_gb" (the memory guard's need), "timeout_s", "gate" (optional)}. "{checkpoint}" and "{run}" in a
command are replaced by the pause's own complete checkpoint and the paused run, and "{candidate}" in a
command or cwd by --candidate, the candidate merge commit's own worktree (a job run there puts its own
tree's Source/Python first on its path: this runner's PYTHONPATH names production's). --check reads and
prints the plan only. A GATE job (the coordinator's qualification gate for the pause in which the three observer
branches merge) passes when it finishes with exit code 0; a failure, a refusal or a timeout fails it. The
receipt lists every gate's pass or fail (`gates`, `gates_failed`, `gates_passed`), in the review hold too.
A gate never keeps training stopped (the coordinator's decision of 2026-09-26): the qualification runs on a
candidate merge commit in its own worktree, never on production's tree, so training resumes in either case
on production's code as it stands. Production moves to the candidate commit only when every gate passed; a
failed gate goes to the authors. The exit code says which: 3 when a gate failed.

What it does, in order:
  0. refuses the whole jobs file when the test registry refuses any of its jobs (test_registry.refusal: no complete
     row in docs/gain_of_function_v4/test_registry.json whose id is the job's name, a block written into the row,
     or the defect register's 24-hour rule; the style guide: NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE), --check
     included, and writes a receipt naming each job's reason into <time>_refused/ before anything stops training.
     During the pause each job's row is read again right before the job starts (a row cancelled, emptied or
     blocked meanwhile refuses that job), and its status ("running", then "done" or
     "failed") and last_run {started, ended, exit_code, log, outcome, runner, receipt} are written back
     (mimo_bridge/tools/test_registry.py);
  1. refuses while the ability scheduler holds a saved window or runs a task (a control change would
     cancel the window's remaining comparisons: ability_comparison_window.advance), while no trainer or
     more than one runs, while the active run holds an unconsumed control.json, or while the supervisor
     would not resume training by itself (training_continuation.pause_reason: a pause, a control file it
     cannot read, a repair stop, a deliberate stop) - EXCEPT during the scheduler's review
     hold (a failed remembered-hand-goals round, ability_comparison_window.hold_for_review): training is
     already stopped then, so the jobs run on the held pause's checkpoint and supervisor_control.json is
     never touched, which keeps the hold (the coordinator's decision of 2026-09-25);
  2. writes the supervisor pause, then snapshot_and_stop with auto_resume to the running trainer, and waits
     for the trainer to exit and for a complete checkpoint at or after the request;
  3. runs each job behind the memory guard, logging to Saved/gof_observer_20260924/pauses/<time>/;
     a job that fails or is refused is recorded and does not stop the others;
  4. lifts the pause ({"cmd": "run"}, every other key kept) ONLY when supervisor_control.json still holds
     the exact pause step 2 wrote, and waits until the supervisor has relaunched a trainer. A pause that
     someone else wrote during the jobs (the operator, a watcher's stop, the scheduler) is left in place,
     and the receipt names it: the scheduler's windows release by the same rule
     (ability_comparison_window.release), and before 2026-09-25 this step overwrote any pause;
  5. writes receipt.json with every job's command, exit code, duration and log path, and each gate's pass
     or fail.
A failure of the pause machinery itself (no checkpoint, the trainer not exiting) leaves the pause in
place and says so: the supervisor then needs a person, as the watcher's stops do.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "Saved/mimo_bridge"
PY = ROOT / ".venv_mimo/Scripts/python.exe"
CWD = ROOT / "Source/Python"
PAUSE = SAVED / "supervisor_control.json"
OUT = ROOT / "Saved/gof_observer_20260924/pauses"
sys.path.insert(0, str(CWD))
from mimo_bridge.tools import test_registry  # noqa: E402
from mimo_bridge.tools.ability_comparison_window import UNREADABLE, read_control  # noqa: E402
from mimo_bridge.tools.training_continuation import pause_reason  # noqa: E402
RUNNER = "run_in_training_pause.py"


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def trainers():
    sys.path.insert(0, str(CWD))
    from mimo_bridge.tools.training_continuation import processes
    return processes("trainer")


def guard(need_gb):
    r = subprocess.run([str(PY), "-m", "mimo_bridge.tools.commit_guard", "--need-gb", str(need_gb)],
                       cwd=str(CWD), capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


# ONE HEAVY JOB AT A TIME, ACROSS EVERY RUNNER (2026-09-26). The guard judges the machine at a job's start only, and a
# trainer grows for minutes after it: at 6:15 PM the memory repair's 27 GB trial arm was admitted at 33.4 GB committed,
# other runners' jobs started beside it, the machine reached 63.1 GB, and the arm stopped itself at the reserve before its
# first save. Every job that declares HEAVY_GB or more now holds this lock for its whole run; the lock names its runner's
# process, and a lock whose process has ended is removed.
HEAVY_GB = 10.0
HEAVY_LOCK = ROOT / "Saved" / "gof_observer_20260924" / "heavy_job.lock"


def _lock_holder(lock):
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _created(pid):
    """A process's creation time, or None when it has ended or cannot be read."""
    import psutil
    try:
        return psutil.Process(int(pid)).create_time()
    except (psutil.Error, ValueError, TypeError):
        return None


def _alive(pid, created=None):
    """True while process `pid` runs and, when `created` is recorded, is the same process (not a reused number)."""
    now = _created(pid) if pid is not None else None
    return now is not None and (created is None or abs(now - float(created)) < 1.0)


# THE LOCK COVERS THE JOB'S OWN PROCESS TOO (2026-09-26, the starter test's re-review, section 5). The lock named only
# the runner; a runner that ended alone (a crash, or a kill of that one process) left its job's watchdog, harness and
# trainer running while the lock read as free, and the next heavy job would start beside them. The runner now writes the
# job's process into the lock when it starts it, and the lock is held while either process lives.
def holder_alive(holder):
    """True while the lock's runner or the job it started still runs."""
    if not holder:
        return False
    return (_alive(holder.get("pid"), holder.get("pid_created")) or
            (holder.get("job_pid") is not None and _alive(holder.get("job_pid"), holder.get("job_pid_created"))))


# THE COORDINATOR CAN RESERVE THE NEXT HEAVY TURNS (2026-09-26, 11:27 PM PDT). The lock had no order: whichever waiting
# runner asked first after a release took it, so the memory starter test's six runs, which the operator put ahead of the
# rest, would have waited behind every other queue. While heavy_job.next holds {"prefix", "until"} and its time has not
# passed, only a job whose name starts with the prefix may take the lock (docs/gain_of_function_v4/coordinator/
# reserve_next_heavy.py sets and clears it; the time is the safety net if the reserved jobs never come).
def next_reservation(lock=None):
    """The reservation {prefix, until, ...} while it is in force, else None."""
    path = Path(lock or HEAVY_LOCK).with_name("heavy_job.next")
    data = _lock_holder(path)
    if not data or time.time() > float(data.get("until") or 0):
        return None
    return data


def _take_lock(lock, record):
    """Create `lock` holding `record` when it is free, first clearing one whose processes have all ended (or an unreadable
    one older than 60 s); True when taken."""
    holder = _lock_holder(lock)
    if lock.exists() and holder is not None and not holder_alive(holder):
        lock.unlink(missing_ok=True)
    elif lock.exists() and holder is None and time.time() - lock.stat().st_mtime > 60:
        lock.unlink(missing_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return True


def acquire_heavy_lock(name, lock=None, pid=None, need_gb=None):
    """Take the heavy-job lock for job `name` if it is free (or held by processes that have all ended) and no reservation
    names another job; True when taken. `need_gb` is the job's declared need, which the memory guard keeps free for the
    part not yet used."""
    reserved = next_reservation(lock)
    if reserved is not None and not str(name).startswith(str(reserved.get("prefix", ""))):
        return False
    pid = int(pid or os.getpid())
    return _take_lock(Path(lock or HEAVY_LOCK),
                      dict(pid=pid, pid_created=_created(pid), job=name, started=time.time(), need_gb=need_gb))


# THE LOCK RECORDS THE JOB'S PEAK SO FAR (2026-09-26, 11:23 PM PDT, the efference lane's report). The memory guard keeps
# free a heavy job's declared need minus what its processes commit now; after the memory control run's trainer ended,
# its light end readings made the guard hold back 23.7 GB the job would never use again, and agents' probes waited on
# it. The runner now writes the job tree's largest private commit so far into the lock every PEAK_POLL_S seconds, and
# the guard keeps free only the declared need above that peak.
PEAK_POLL_S = 10.0


def _tree_private_gib(pid):
    import psutil
    try:
        root = psutil.Process(int(pid))
        procs = [root] + root.children(recursive=True)
    except (psutil.Error, ValueError, TypeError):
        return 0.0
    total = 0
    for p in procs:
        try:
            mi = p.memory_info()
            total += getattr(mi, "private", 0) or getattr(mi, "rss", 0)
        except psutil.Error:
            pass
    return total / 1024.0 ** 3


def record_heavy_job_peak(gib, lock=None, pid=None):
    """Raise the lock's recorded peak (peak_gb) to `gib` if this runner holds the lock and `gib` is higher."""
    lock = Path(lock or HEAVY_LOCK)
    holder = _lock_holder(lock)
    if holder is None or int(holder.get("pid", -1)) != int(pid or os.getpid()):
        return False
    if gib <= float(holder.get("peak_gb") or 0.0):
        return False
    holder["peak_gb"] = round(float(gib), 3)
    write(lock, holder)
    return True


def wait_recording_peak(proc, timeout_s, poll_s=None):
    """proc.wait(timeout_s), sampling the job tree's private commit and recording its peak in the lock meanwhile."""
    poll_s = PEAK_POLL_S if poll_s is None else poll_s
    deadline = time.time() + float(timeout_s)
    while True:
        record_heavy_job_peak(_tree_private_gib(proc.pid))
        remaining = deadline - time.time()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout_s)
        try:
            return proc.wait(timeout=min(poll_s, remaining))
        except subprocess.TimeoutExpired:
            continue


def name_heavy_job_process(job_pid, lock=None, pid=None):
    """Write the started job's process into the lock this runner holds (nothing when another process holds it)."""
    lock = Path(lock or HEAVY_LOCK)
    holder = _lock_holder(lock)
    if holder is None or int(holder.get("pid", -1)) != int(pid or os.getpid()):
        return False
    holder.update(job_pid=int(job_pid), job_pid_created=_created(job_pid))
    write(lock, holder)
    return True


def end_tree(pid, timeout=30):
    """End process `pid` and every process under it; returns how many were ended."""
    import psutil
    try:
        root = psutil.Process(int(pid))
    except psutil.NoSuchProcess:
        return 0
    tree = root.children(recursive=True) + [root]
    for p in tree:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(tree, timeout=timeout)
    return len(tree)


def _own_family(pid=None):
    """This process and its parents (or only `pid` when given): a lock held by one of them is the caller's own."""
    if pid is not None:
        return {int(pid)}
    import psutil
    try:
        me = psutil.Process(os.getpid())
        return {me.pid} | {p.pid for p in me.parents()}
    except psutil.Error:
        return {os.getpid()}


def _held_by_family(holder, pid=None):
    """True when the lock's runner, or the job it started, is the caller or one of its parents: the same process, not a
    reused number (its creation time matches when the lock recorded one)."""
    mine = _own_family(pid)
    for key in ("pid", "job_pid"):
        try:
            number = int(holder.get(key))
        except (TypeError, ValueError):
            continue
        if number in mine and _alive(number, holder.get(key + "_created")):
            return True
    return False


def heavy_lock_is_own(lock=None, pid=None):
    """True while the heavy-job lock is held by the caller or one of its parents: the caller runs inside its own turn."""
    holder = _lock_holder(Path(lock or HEAVY_LOCK))
    return holder is not None and _held_by_family(holder, pid)


def heavy_lock_held_by_other(lock=None, pid=None):
    """True while another runner holds the heavy-job lock and it, or the job it started, still runs. A lock held by the
    caller or one of its parents is its own (2026-09-27, 2:34 AM: the skin check's test job, run by the runner that
    held the lock for it, waited 10,800 s on itself because only the runner's own pid counted as its own)."""
    holder = _lock_holder(Path(lock or HEAVY_LOCK))
    if holder is None or _held_by_family(holder, pid):
        return False
    return holder_alive(holder)


def release_heavy_lock(lock=None, pid=None):
    """Remove the heavy-job lock if this process holds it."""
    lock = Path(lock or HEAVY_LOCK)
    holder = _lock_holder(lock)
    if holder is not None and int(holder.get("pid", -1)) == int(pid or os.getpid()):
        lock.unlink(missing_ok=True)


# ONE RUNNER'S JOBS FILE AT A TIME (2026-09-27). The heavy-job lock is taken per job and released between jobs, so
# another runner's heavy job could start in the gap, and the first runner's next lighter jobs then waited at most
# guard_wait_s (900 s by default) for it and were refused: the motor confidence gate lost 7 of its 13 jobs that way
# (Sep 26, 9:31 to 11:23 PM) and the balance repair its measurement and rehearsal (by 12:29 AM), and nothing re-ran them.
# A runner now holds runner_turn.lock, beside the heavy-job lock, for its whole jobs file; other runners and the queues'
# waiter (run_jobs_in_held_pause.machine_busy) wait for it with no time limit, since a turn ends with its own jobs, each
# bounded by its timeout. Within a turn the heavy-job lock still marks the heavy job for the memory guard and the tests.
def runner_turn_path(lock=None):
    """The runner turn's lock file: `lock`, or runner_turn.lock beside the heavy-job lock."""
    return Path(lock) if lock else Path(HEAVY_LOCK).with_name("runner_turn.lock")


def acquire_runner_turn(label, lock=None, pid=None):
    """Take the runner turn for `label` (the jobs files) when no live runner holds it; True when taken."""
    pid = int(pid or os.getpid())
    return _take_lock(runner_turn_path(lock), dict(pid=pid, pid_created=_created(pid), job=label, started=time.time()))


def runner_turn_is_own(lock=None, pid=None):
    """True while the runner turn is held by the caller or one of its parents: the caller is one of that turn's jobs."""
    holder = _lock_holder(runner_turn_path(lock))
    return holder is not None and _held_by_family(holder, pid)


def runner_turn_held_by_other(lock=None, pid=None):
    """True while another live runner, not the caller or one of its parents, holds the runner turn."""
    holder = _lock_holder(runner_turn_path(lock))
    if holder is None or _held_by_family(holder, pid):
        return False
    return holder_alive(holder)


def release_runner_turn(lock=None, pid=None):
    """Remove the runner turn if this process holds it."""
    lock = runner_turn_path(lock)
    holder = _lock_holder(lock)
    if holder is not None and int(holder.get("pid", -1)) == int(pid or os.getpid()):
        lock.unlink(missing_ok=True)


def take_runner_turn(label, first_job, poll_s=30):
    """Wait, with no time limit, until this runner holds the runner turn. While a reservation (heavy_job.next) is in force
    only a runner whose first job it names may take the turn. Returns the seconds waited."""
    waited = 0
    while True:
        reserved = next_reservation()
        ours = reserved is None or str(first_job).startswith(str(reserved.get("prefix", "")))
        if ours and acquire_runner_turn(label):
            return waited
        if waited % 600 == 0:
            print("waiting for the runner turn: %s" % ("the next turns are reserved for %s" % reserved.get("prefix")
                                                       if not ours else "held by %s" % (
                                                           _lock_holder(runner_turn_path()) or {}).get("job")),
                  flush=True)
        time.sleep(poll_s)
        waited += poll_s


def held_review(state=None):
    """The scheduler's saved pause while it is held for review, or None. Training is already stopped under
    it: jobs may run on its checkpoint, and the hold stays because nothing here writes
    supervisor_control.json (a change to that file ends the hold: ability_comparison_window.held_by_review)."""
    state = state if state is not None else (read(SAVED / "ability_rechecks/status.json", {}) or {})
    window = state.get("comparison_window") or {}
    return window if window.get("state") == "held for review" else None


def refusals():
    out = []
    state = read(SAVED / "ability_rechecks/status.json", {}) or {}
    held = held_review(state)
    if held is not None:
        # THE COORDINATOR'S DECISION OF 2026-09-25 (the scheduler review's gap b): jobs use the stop the
        # review hold already made, without lifting it
        if read(PAUSE) != held.get("owned_control"):
            out.append("the review hold no longer owns the supervisor control")
        if trainers():
            out.append("a trainer runs during the review hold")
        if state.get("active_task"):
            out.append("the ability scheduler runs %s" % state["active_task"])
        if not held.get("checkpoint") or not held.get("run"):
            out.append("the held pause names no checkpoint")
        return out
    window = (state.get("comparison_window") or {}).get("state")
    if window in ("preparing", "saving", "ready"):
        out.append("the ability scheduler holds a saved window (%s)" % window)
    if state.get("active_task"):
        out.append("the ability scheduler runs %s" % state["active_task"])
    active = (read(SAVED / "training_continuation.json", {}) or {}).get("active_run")
    if not active:
        out.append("training_continuation.json names no active run")
    else:
        run = Path(active)
        # every reason the supervisor would not relaunch after the stop, read by the supervisor's own rule:
        # an unreadable control (UTF-16 from PowerShell included) read as no pause before 2026-09-25, and
        # the pause written over it then dropped its other keys, such as "archiver": "off"
        why = pause_reason(SAVED, run)
        if why == "explicit supervisor pause":
            out.append("the supervisor is already paused")
        elif why:
            out.append("the supervisor would not resume training after the pause: %s" % why)
        if (run / "control.json").exists():
            out.append("the active run holds an unconsumed control.json")
    if read_control(PAUSE) is UNREADABLE:
        out.append("supervisor_control.json cannot be read; rewrite it as UTF-8 JSON first")
    if len(trainers()) != 1:
        out.append("expected exactly one trainer, found %d" % len(trainers()))
    return out


# NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE (the style guide, operator 2026-10-02: "This entire testing process seems to
# be spiraling out of control"). Every job needs one complete row in docs/gain_of_function_v4/test_registry.json whose
# id is the job's name; mimo_bridge/tools/test_registry.py judges the row. One job without one refuses its WHOLE jobs
# file before the pause is written: a jobs file is one plan (a gate and the jobs after it, steps that read earlier
# steps' results), and running part of it would stop training for a plan that cannot finish - or run a trial whose
# gate never ran. The refusal receipt names each job's reason.
def registry_path():
    """The test registry of this runner's repository (ROOT; a test's temporary tree when a test replaces ROOT)."""
    return test_registry.registry_path(ROOT)


def registry_refusals(jobs):
    """{job name: why it may not run} for every job without a complete registry row; {} when every job has one."""
    return test_registry.refusals([j["name"] for j in jobs], path=registry_path())


def write_registry_refusal(jobs, refused, suffix="_refused", **extra):
    """Write the receipt of a jobs file refused for its registry rows (OUT/<time><suffix>/receipt.json) before anything
    stops training: each refused job with its reason, every other job as not run. Returns the receipt's path."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for n in range(1, 100):
        folder = OUT / (stamp + suffix + ("" if n == 1 else "_%d" % n))
        try:
            folder.mkdir(parents=True)
            break
        except FileExistsError:
            continue
    now = time.time()
    rows = [dict(name=j["name"], status="refused by the test registry", reason=refused[j["name"]],
                 recorded=now) if j["name"] in refused else
            dict(name=j["name"], status="not run: another job of this jobs file has no complete row in the test registry",
                 recorded=now)
            for j in jobs]
    receipt = dict(extra, started=now, finished=now, refused="test registry", registry=str(registry_path()), jobs=rows,
                   note=("Refused before anything stopped training: every job needs one complete row in the test "
                         "registry whose id is the job's name (the style guide: NO TEST RUNS WITHOUT ITS ROW ON THE TESTS "
                         "PAGE). No pause was written, no checkpoint was requested and no job ran. Write or complete "
                         "the rows named here, then launch the jobs file again."))
    write(folder / "receipt.json", receipt)
    return folder / "receipt.json"


def _registry_note(row, status, why):
    row.setdefault("registry_writes", []).append(status if why is None else "%s not written: %s" % (status, why))


def _registry_started(registry, row, folder, log, runner, pid=None, checkpoint=None):
    """Write the started job into its row (test_registry.job_started): status running, the start time, the commit of
    this runner's repository, the pause's checkpoint, the job's process id and creation time, and the run id whose
    mark the job's processes carry (row["run_id"], set before the job started)."""
    if registry is not None:
        _registry_note(row, "running", test_registry.job_started(
            row["name"], started=row.get("started"), log=log, runner=runner, receipt=folder / "receipt.json",
            path=registry, pid=pid, checkpoint=None if checkpoint is None else str(checkpoint),
            run_id=row.get("run_id"), command=row.get("cmd"), cwd=ROOT))


def _registry_finished(registry, row, folder, runner, outcome=None):
    """Write the ended job into its row (test_registry.finish_job, which reads the outcome: the result file's verdict
    key, else the exit code): PASS gives status "done"; FAIL, INVALID and STOPPED give "failed". A job this runner
    ended at its timeout is STOPPED; a job that never started (a refusal, a failed gate before it) is INVALID. The
    receipt row keeps the outcome and the status that was written."""
    if registry is None:
        return
    code = row.get("exit_code") if row.get("status") == "done" else None
    if outcome is None:
        outcome = (None if row.get("status") == "done" else
                   "timed out; %s processes ended" % row.get("ended") if row.get("status") == "timed out" else
                   "not started: %s" % row.get("status"))
    why, state = test_registry.finish_job(
        row["name"], exit_code=code, outcome=outcome, started=row.get("started"), log=row.get("log"), runner=runner,
        receipt=folder / "receipt.json", path=registry, run_id=row.get("run_id"), root=ROOT)
    if state:
        row["outcome"] = state
    _registry_note(row, test_registry.STATUS_OF_OUTCOME.get(state, "done" if code == 0 else "failed"), why)


def complete_after(run, step0):
    from mimo_bridge.tools.training_continuation import complete_candidates
    for path in complete_candidates(run):
        if int(path.name.removeprefix("ckpt_step_")) >= step0:
            return path
    return None


def latest_step(run):
    path = Path(run) / "replay_training_stats.jsonl"
    last = None
    try:
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - 2 ** 20))
            for line in f.read().decode("utf-8", "replace").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") == "mimo":
                    last = r.get("step")
    except OSError:
        pass
    return last


def lift(ours, receipt):
    """Write {"cmd": "run"} (every other key kept) only while supervisor_control.json still holds exactly
    the pause this run wrote; otherwise leave the file alone, record who holds it, and return False."""
    current = read_control(PAUSE)
    if current != ours:
        receipt.update(not_lifted=("supervisor_control.json changed during the pause, so it was left as it is and "
                                   "training stays paused until its writer resumes it"),
                       control_now="unreadable" if current is UNREADABLE else current)
        return False
    write(PAUSE, dict({k: v for k, v in ours.items() if k != "pause_runner"}, cmd="run",
                      reason="Coordinated evaluation pause finished; training resumes", timestamp=time.time()))
    return True


def run_jobs(jobs, checkpoint, run, folder, receipt, *, stop_after_gate_failure=False, registry=None, runner=RUNNER):
    """Run guarded jobs; optionally block later jobs after an explicit gate fails.

    Ungated failures may be intentional baseline measurements. They do not block
    the later comparison. The active legacy runners retain their existing policy.

    With `registry` (the test registry's path; both runners pass it) each job's row is read again right before the job
    starts, and a missing, incomplete or cancelled row refuses that job alone (the pause has begun by then); the row's
    status and last_run are written back: "running" once the job's process starts, then "done" (exit code 0) or
    "failed" (another exit code, a timeout, a refusal, a failed gate before it). Each write's outcome is listed in the
    job's receipt row (registry_writes). Without `registry` (a library call; the gate rules' tests) nothing is read or
    written there.
    """
    for job in jobs:
        failed_gate = next((previous["name"] for definition, previous in zip(jobs, receipt["jobs"])
                            if definition.get("gate") and
                            not (previous.get("status") == "done" and previous.get("exit_code") == 0)), None)
        if stop_after_gate_failure and failed_gate:
            row = dict(name=job["name"], status="blocked by prerequisite", blocked_by=failed_gate, recorded=time.time())
            _registry_finished(registry, row, folder, runner,
                               outcome="not started: the gate job %s before it did not pass" % failed_gate)
            receipt["jobs"].append(row)
            write(folder / "receipt.json", receipt)
            continue
        if registry is not None:
            why = test_registry.refusal(job["name"], path=registry)
            if why:
                row = dict(name=job["name"], status="refused by the test registry", reason=why,
                           recorded=time.time())
                receipt["jobs"].append(row)
                write(folder / "receipt.json", receipt)
                print(json.dumps({k: row[k] for k in ("name", "status", "reason")}), flush=True)
                continue
        cmd = [str(x).replace("{checkpoint}", str(checkpoint)).replace("{run}", str(run)) for x in job["cmd"]]
        cmd = [str(PY) if x == "PY" else x for x in cmd]
        row = dict(name=job["name"], cmd=cmd, started=time.time())
        waited, limit = 0.0, float(job.get("guard_wait_s", 900))
        heavy = float(job.get("need_gb", 4)) >= HEAVY_GB
        if heavy:
            need = float(job.get("need_gb", 4))
            got = acquire_heavy_lock(job["name"], need_gb=need)
            while not got and waited < limit:
                time.sleep(30)
                waited += 30
                got = acquire_heavy_lock(job["name"], need_gb=need)
            if not got:
                row.update(status="refused: another heavy job held the machine", heavy_lock=_lock_holder(HEAVY_LOCK),
                           guard_waited_s=waited, seconds=round(time.time() - row["started"], 1))
                _registry_finished(registry, row, folder, runner, outcome="not started: another heavy job held the "
                                   "machine for the whole %d s wait" % waited)
                receipt["jobs"].append(row)
                write(folder / "receipt.json", receipt)
                print(json.dumps({k: row[k] for k in ("name", "status", "seconds")}), flush=True)
                continue
        else:
            # a lighter job admitted while a heavy one is still growing can push the machine past the reserve, so it
            # waits for the heavy job to finish too
            while heavy_lock_held_by_other() and waited < limit:
                time.sleep(30)
                waited += 30
            if heavy_lock_held_by_other():
                row.update(status="refused: a heavy job held the machine", heavy_lock=_lock_holder(HEAVY_LOCK),
                           guard_waited_s=waited, seconds=round(time.time() - row["started"], 1))
                _registry_finished(registry, row, folder, runner, outcome="not started: a heavy job held the machine "
                                   "for the whole %d s wait" % waited)
                receipt["jobs"].append(row)
                write(folder / "receipt.json", receipt)
                print(json.dumps({k: row[k] for k in ("name", "status", "seconds")}), flush=True)
                continue
        ok, reading = guard(job.get("need_gb", 4))
        # A REFUSED JOB WAITS FOR MEMORY, IT IS NOT SKIPPED (2026-09-25): in the evening pause 5 of 9 jobs were
        # refused at their start while agents' tests held memory, and the pause ended with them never run.
        # The guard is asked again every 30 s for up to guard_wait_s (default 900 s; 0 keeps the old skip).
        while not ok and waited < limit:
            time.sleep(30)
            waited += 30
            ok, reading = guard(job.get("need_gb", 4))
        row["guard"] = reading
        if waited:
            row["guard_waited_s"] = waited
        if not ok:
            row.update(status="refused by the memory guard")
            _registry_finished(registry, row, folder, runner, outcome="not started: refused by the memory guard after "
                               "a %d s wait (its reading is in the receipt)" % waited)
            if heavy:
                release_heavy_lock()
        else:
            log = folder / (job["name"] + ".log")
            env = dict(os.environ, PYTHONPATH=str(CWD), PYTHONUNBUFFERED="1")
            if registry is not None:
                # THE JOB'S PROCESSES CARRY THE ROW'S MARK (Oct 5, 2026; directive section 8): test_runner.audit()
                # matches every live test process to a running row by these two values of its environment
                row["run_id"] = test_registry.new_run_id(job["name"], row["started"])
                env.update(UEMA_TEST_ROW=job["name"], UEMA_TEST_RUN=row["run_id"])
            try:
                with log.open("wb") as stream:
                    proc = subprocess.Popen(cmd, cwd=str(ROOT / job.get("cwd", "Source/Python")), env=env,
                                            stdout=stream, stderr=subprocess.STDOUT)
                    if heavy:
                        name_heavy_job_process(proc.pid)
                    _registry_started(registry, row, folder, log, runner, pid=proc.pid, checkpoint=checkpoint)
                    try:
                        if heavy:
                            code = wait_recording_peak(proc, job.get("timeout_s", 3600))
                        else:
                            code = proc.wait(timeout=job.get("timeout_s", 3600))
                        row.update(status="done", exit_code=code, log=str(log))
                    except subprocess.TimeoutExpired:
                        # END THE WHOLE TREE (2026-09-26, the starter test's review): subprocess.run's timeout ended
                        # only its own child, the pause watchdog, and left the trainer under it running after the
                        # heavy-job lock was released, so the next heavy job started a second trainer beside it
                        row.update(status="timed out", log=str(log), ended=end_tree(proc.pid))
            finally:
                if heavy:
                    release_heavy_lock()
            if row.get("status") == "timed out":
                _registry_finished(registry, row, folder, runner, outcome="timed out after %s s; %s processes ended"
                                   % (job.get("timeout_s", 3600), row.get("ended")))
            else:
                _registry_finished(registry, row, folder, runner)
        row["seconds"] = round(time.time() - row["started"], 1)
        receipt["jobs"].append(row)
        write(folder / "receipt.json", receipt)
        print(json.dumps({k: row[k] for k in ("name", "status", "seconds") if k in row} | {"exit_code": row.get("exit_code")}), flush=True)


def gate_results(jobs, receipt):
    """Each GATE job's pass or fail, written into the receipt: `gates` (name, passed, status, exit code),
    `gates_failed` and `gates_passed`. A gate passes only when its job ran and exited 0. The gates decide
    whether production moves to the candidate code the jobs judged, never whether training resumes (the
    coordinator's decision of 2026-09-26). Returns the names of the failed gates."""
    gates = [dict(name=row["name"], passed=bool(row.get("status") == "done" and row.get("exit_code") == 0),
                  status=row.get("status"), exit_code=row.get("exit_code"))
             for row, job in zip(receipt["jobs"], jobs) if job.get("gate")]
    if not gates:
        return []
    failed = [g["name"] for g in gates if not g["passed"]]
    receipt.update(gates=gates, gates_failed=failed, gates_passed=not failed)
    return failed


def with_candidate(jobs, candidate):
    """The jobs with "{candidate}" in each cmd and cwd replaced by the candidate worktree, and the refusals:
    a job that names {candidate} without --candidate, or a candidate folder that does not exist. The gate
    jobs judge the candidate's code there while this runner - production's own - pauses and resumes
    production's training."""
    uses = [j["name"] for j in jobs if "{candidate}" in json.dumps([j.get("cmd"), j.get("cwd")])]
    if not uses:
        return jobs, []
    if not candidate:
        return jobs, ["jobs %s name {candidate}: pass --candidate <worktree>" % ", ".join(uses)]
    if not Path(candidate).is_dir():
        return jobs, ["the candidate worktree %s does not exist" % candidate]
    root = str(Path(candidate).resolve()).replace("\\", "/")
    out = []
    for j in jobs:
        j = dict(j, cmd=[str(x).replace("{candidate}", root) for x in j["cmd"]])
        if "cwd" in j:
            j["cwd"] = str(j["cwd"]).replace("{candidate}", root)
        out.append(j)
    return out, []


def gate_line(failed):
    return (" | GATES FAILED (%s): production must not move to the candidate code; the failure goes to its "
            "authors" % ", ".join(failed)) if failed else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--candidate", help="the candidate merge commit's own worktree the gate jobs judge: replaces "
                                        "{candidate} in each job's cmd and cwd")
    a = ap.parse_args()
    jobs, missing = with_candidate(json.loads(Path(a.jobs).read_text(encoding="utf-8")), a.candidate)
    sup = read(SAVED / "training_continuation.json", {}) or {}
    run = Path(sup.get("active_run", ""))
    problems = refusals() + missing
    unregistered = registry_refusals(jobs)
    held = held_review()
    print(json.dumps(dict(run=str(run), jobs=[j["name"] for j in jobs], refusals=problems,
                          registry_refusals=unregistered, review_hold=bool(held), candidate=a.candidate), indent=2),
          flush=True)
    if unregistered and not a.check:
        receipt = write_registry_refusal(jobs, unregistered, jobs_files=[str(a.jobs)], other_refusals=problems)
        print("receipt: %s | REFUSED before anything stopped training: %d of %d jobs have no complete row in the test "
              "registry" % (receipt, len(unregistered), len(jobs)), flush=True)
    if a.check or problems or unregistered:
        return 1 if problems or unregistered else 0
    waited = take_runner_turn(Path(a.jobs).name, jobs[0]["name"] if jobs else "")
    try:
        if waited:
            # the machine and the registry may have changed while this runner waited for its turn: read them again
            sup = read(SAVED / "training_continuation.json", {}) or {}
            run = Path(sup.get("active_run", ""))
            problems, held, unregistered = refusals(), held_review(), registry_refusals(jobs)
            if unregistered:
                write_registry_refusal(jobs, unregistered, jobs_files=[str(a.jobs)], other_refusals=problems,
                                       turn_waited_s=waited)
            if problems or unregistered:
                print(json.dumps(dict(refusals=problems, registry_refusals=unregistered, turn_waited_s=waited),
                                 indent=2), flush=True)
                return 1
        return _run_in_turn(jobs, run, held)
    finally:
        release_runner_turn()


def _run_in_turn(jobs, run, held):
    folder = OUT / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)
    if held is not None:
        # inside the review hold: no pause written, no save requested, no resume; the hold stays
        checkpoint, run = Path(held["checkpoint"]), Path(held["run"])
        receipt = dict(run=str(run), started=time.time(), jobs=[], checkpoint=str(checkpoint), stopped=held.get("held_at"),
                       review_hold=held.get("token"),
                       note=("Ran inside the scheduler's review hold: training was already stopped, "
                             "supervisor_control.json was not changed, and the hold stays."))
        write(folder / "receipt.json", receipt)
        run_jobs(jobs, checkpoint, run, folder, receipt, registry=registry_path())
        failed = gate_results(jobs, receipt)     # the gates are judged in the hold as everywhere else
        receipt.update(resumed=False, finished=time.time())
        write(folder / "receipt.json", receipt)
        print("receipt: " + str(folder / "receipt.json") + " | the review hold stays; training resumes when the operator "
              "releases it" + gate_line(failed), flush=True)
        return 3 if failed else 0
    receipt = dict(run=str(run), started=time.time(), jobs=[])
    previous = read_control(PAUSE) or {}
    if previous is UNREADABLE:
        raise SystemExit("supervisor_control.json became unreadable after the checks; nothing was written")
    ours = dict(previous, cmd="pause", reason="Coordinated evaluation pause: " + ", ".join(j["name"] for j in jobs),
                timestamp=time.time(), pause_runner=folder.name)
    write(PAUSE, ours)
    receipt.update(owned_control=ours)
    step0 = latest_step(run) or 0
    try:
        # exclusive: a command that arrived after the refusals were read is never replaced
        with (run / "control.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(cmd="snapshot_and_stop", auto_resume=True, reason="coordinated evaluation pause"), stream)
    except FileExistsError:
        lifted = lift(ours, receipt)
        receipt.update(error="another trainer control arrived before the stop request; no job ran"
                             + ("; the pause was lifted" if lifted else "; the pause was changed by someone else and stays"))
        write(folder / "receipt.json", receipt)
        raise SystemExit(receipt["error"])
    deadline = time.monotonic() + 1200
    checkpoint = None
    while time.monotonic() < deadline:
        checkpoint = checkpoint or complete_after(run, step0)
        if checkpoint is not None and not trainers() and not (run / "control.json").exists():
            break
        time.sleep(5)
    else:
        receipt.update(error="the trainer did not stop with a complete checkpoint within 20 minutes; the pause stays")
        write(folder / "receipt.json", receipt)
        raise SystemExit(receipt["error"])
    receipt.update(checkpoint=str(checkpoint), stopped=time.time())
    write(folder / "receipt.json", receipt)
    run_jobs(jobs, checkpoint, run, folder, receipt, registry=registry_path())
    # a failed gate does not keep training stopped: the jobs judged a candidate commit in its own worktree,
    # and training resumes on production's code as it stands (the coordinator's decision of 2026-09-26)
    failed = gate_results(jobs, receipt)
    write(folder / "receipt.json", receipt)
    if not lift(ours, receipt):
        receipt.update(resumed=False, finished=time.time())
        write(folder / "receipt.json", receipt)
        print("receipt: " + str(folder / "receipt.json") + " | NOT RESUMED: " + receipt["not_lifted"] + gate_line(failed),
              flush=True)
        return 2
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and not trainers():
        time.sleep(5)
    receipt.update(resumed=bool(trainers()), finished=time.time())
    write(folder / "receipt.json", receipt)
    print("receipt: " + str(folder / "receipt.json") + (" | training relaunched" if receipt["resumed"] else
                                                        " | NO TRAINER after 10 minutes: inspect the supervisor")
          + gate_line(failed), flush=True)
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
