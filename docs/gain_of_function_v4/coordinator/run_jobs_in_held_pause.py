"""Run pause jobs while training is already held by someone else's pause (2026-09-26: the H84 hourly watcher's memory
stop held training from 8:11 PM; the pause runner refuses a held pause, so the evaluation jobs waiting for a test pause
would otherwise wait on the downtime that is already happening).

It refuses unless supervisor_control.json says pause and no trainer runs. It never writes supervisor_control.json or
any run's files. Every job runs inside mimo_bridge.tools.pause_watchdog, so whoever lifts the pause ends the running
job within 5 s and the later jobs refuse to start: lifting it at any moment is safe. The jobs run on the held run's
newest checkpoint, one after another, behind the memory guard (run_in_training_pause.run_jobs).

usage: python run_jobs_in_held_pause.py [--candidate <worktree>] <jobs.json> [<jobs.json> ...]
"{candidate}" and gate jobs work as in run_in_training_pause.py: gates are judged into the receipt and decide only
whether production moves to the candidate, never the hold.

NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE (the style guide, operator 2026-10-02): when any job has no complete row in
docs/gain_of_function_v4/test_registry.json (a row whose id is the job's name), the jobs files are refused before the
runner turn is taken and again once it is held, with a receipt in <time>_held_refused/ naming each job's reason; then
SystemExit, as for this runner's other refusals. Each job's row is read again before it starts and gets its status and
last_run written back (run_in_training_pause.run_jobs, mimo_bridge/tools/test_registry.py)."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_in_training_pause as rtp  # noqa: E402

WATCHDOG = ["PY", str(Path(__file__).resolve().parents[3] / "Source/Python/mimo_bridge/tools/pause_watchdog.py"), "--"]
RUNNER = "run_jobs_in_held_pause.py"


def refuse_unregistered(paths, jobs, **extra):
    """SystemExit, after writing a refusal receipt (rtp.write_registry_refusal, folder <time>_held_refused), when any
    job has no complete row in the test registry; nothing when every job has one."""
    unregistered = rtp.registry_refusals(jobs)
    if unregistered:
        receipt = rtp.write_registry_refusal(jobs, unregistered, suffix="_held_refused",
                                             jobs_files=[str(p) for p in paths], **extra)
        raise SystemExit("refused: %d of %d jobs have no complete row in the test registry (%s); receipt %s"
                         % (len(unregistered), len(jobs), ", ".join(sorted(unregistered)), receipt))


def trainer_running():
    """A python process that IS a trainer: `-m mimo_bridge.train_mimo` or a script path ending in train_mimo.py as an
    argument. Matching the text anywhere in a command line (before 2026-09-26) also matched read-only checks and shells
    that merely mention the name, and refused jobs with no trainer running (the probe GPU fix's report)."""
    import psutil
    module = "mimo_bridge." + "train_mimo"
    for p in psutil.process_iter(["name", "cmdline"]):
        if (p.info["name"] or "").lower() not in ("python.exe", "python", "python3"):
            continue
        args = p.info["cmdline"] or []
        if module in args or any(a.replace("\\", "/").endswith("mimo_bridge/train_" + "mimo.py") for a in args):
            return True
    return False


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


# A TRIAL TRAINER IS NAMED *_trial_trainer.py (2026-09-26, 9:29 PM PDT). The first form of this test matched any script
# ending in "_trainer.py", which includes this folder's own wait_no_trainer.py: from 9:21 PM every waiting queue saw
# itself and the other waiters as trainers, and all three waited with the machine free.
TRIAL_TRAINER_SUFFIX = "_trial_trainer.py"


def is_trial_trainer(script, wrapper=TRIAL_TRAINER_SUFFIX):
    """True for a trial trainer's script path (a name ending in `wrapper`); never a test file (test_*)."""
    name = (script or "").replace("\\", "/").rsplit("/", 1)[-1]
    return name.endswith(wrapper) and not name.startswith("test_")


def trial_trainer_running(wrapper=TRIAL_TRAINER_SUFFIX):
    """A python process other than this one whose script is a trial trainer: a wrapper that runs train_mimo inside its
    own process, such as the memory trial's persistent_memory_trial_trainer.py, which trainer_running() cannot see (the
    merge review's first finding, 8:21 PM PDT 2026-09-26)."""
    import os
    import psutil
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] == os.getpid() or (p.info["name"] or "").lower() not in ("python.exe", "python", "python3"):
            continue
        if is_trial_trainer(_script_of(p.info["cmdline"] or []), wrapper):
            return True
    return False


def machine_busy(prefix=None):
    """Why a queue should not start its next jobs file yet, or None: a trainer, a trial trainer, a runner's heavy job
    holding the heavy-job lock (a driver that trains inside its own process shows only through that lock), another
    runner's turn (rtp.take_runner_turn: one runner's jobs file at a time), or a reservation of the next heavy turns
    for other jobs (reserve_next_heavy.py; `prefix` names the waiting file's own
    jobs, which the reservation may be for). The queues wait on this before each jobs file, with no time limit, so a
    reservation longer than a job's guard wait never makes the job refuse; main() still refuses only a trainer. A job
    already inside its own heavy turn (the lock is its runner's) is not held back by a reservation (2026-09-27): the
    reserved jobs cannot start until that turn ends, so waiting on them would wait on itself."""
    if trainer_running():
        return "a trainer runs"
    if trial_trainer_running():
        return "a trial trainer runs"
    if rtp.heavy_lock_held_by_other():
        return "a heavy job holds the machine (%s)" % (rtp._lock_holder(rtp.HEAVY_LOCK) or {}).get("job")
    if rtp.runner_turn_held_by_other():
        return "another runner has its turn (%s)" % (rtp._lock_holder(rtp.runner_turn_path()) or {}).get("job")
    reserved = rtp.next_reservation()
    if (reserved is not None and not (prefix and str(prefix).startswith(str(reserved.get("prefix", ""))))
            and not (rtp.heavy_lock_is_own() or rtp.runner_turn_is_own())):
        return "the next heavy turns are reserved for %s" % reserved.get("prefix")
    return None


def main(paths, candidate=None, *, stop_after_gate_failure=False):
    jobs = []
    for p in paths:
        for j in json.loads(Path(p).read_text(encoding="utf-8")):
            j = dict(j)
            if "pause_watchdog" not in " ".join(map(str, j["cmd"])):
                j["cmd"] = WATCHDOG + list(j["cmd"])
            jobs.append(j)
    jobs, missing = rtp.with_candidate(jobs, candidate)
    if missing:
        raise SystemExit("refused: " + "; ".join(missing))
    refuse_unregistered(paths, jobs)       # before the turn: nothing waits for a turn it could not use
    # the whole jobs file runs in one runner turn (rtp.take_runner_turn), so no other runner's heavy job starts
    # between its jobs; the checks below are read once the turn is this runner's
    waited = rtp.take_runner_turn(", ".join(Path(p).name for p in paths), jobs[0]["name"] if jobs else "")
    try:
        return _run_in_turn(paths, jobs, waited, stop_after_gate_failure=stop_after_gate_failure)
    finally:
        rtp.release_runner_turn()


def _run_in_turn(paths, jobs, waited, *, stop_after_gate_failure=False):
    refuse_unregistered(paths, jobs, turn_waited_s=waited)     # the registry may have changed during the wait
    control = json.loads((rtp.SAVED / "supervisor_control.json").read_text(encoding="utf-8")) \
        if (rtp.SAVED / "supervisor_control.json").exists() else {}
    if control.get("cmd") != "pause":
        raise SystemExit("refused: training is not held by a pause")
    if trainer_running():
        raise SystemExit("refused: a trainer is running")
    cont = json.loads((rtp.SAVED / "training_continuation.json").read_text(encoding="utf-8"))
    run = Path(cont["active_run"])
    ckpts = sorted((run / "checkpoints").glob("ckpt_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    checkpoint = ckpts[-1]
    folder = rtp.ROOT / "Saved" / "gof_observer_20260924" / "pauses" / (time.strftime("%Y%m%d_%H%M%S") + "_held")
    folder.mkdir(parents=True)
    receipt = dict(held_by=control, run=str(run), checkpoint=str(checkpoint), started=time.time(),
                   jobs_files=[str(p) for p in paths], jobs=[], turn_waited_s=waited,
                   stop_after_gate_failure=stop_after_gate_failure)
    rtp.write(folder / "receipt.json", receipt)
    rtp.run_jobs(jobs, checkpoint, run, folder, receipt, stop_after_gate_failure=stop_after_gate_failure,
                 registry=rtp.registry_path(), runner=RUNNER)
    failed = rtp.gate_results(jobs, receipt)
    receipt["finished"] = time.time()
    rtp.write(folder / "receipt.json", receipt)
    print("receipt:", folder / "receipt.json", "| supervisor_control.json untouched" + rtp.gate_line(failed), flush=True)
    return 3 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    cand = None
    if args[:1] == ["--candidate"]:
        cand, args = args[1], args[2:]
    sys.exit(main(args, cand))
