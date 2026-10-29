# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Which hourly watchers a relaunch ends (the independent review of Oct 9, 2026, finding F5). The relaunch ended only the
watchers whose arguments name the --parent folder, so the comparison's launches, which name --parent seg41o_1431, would
have left the active run's watcher running: today the completed cap arm seg41c_1629's (processes 22080 and 5136),
which the operator decided on Oct 8 to stop once no trainer depends on it. main() now ends the watchers of every folder
stopped_watcher_runs names once every refusal has passed (the memory guard, the last, included: the re-check of
2557e6902), the old supervisor is ended and no trainer runs. Only the selection functions are called here, on command
lines written as text, and main()'s text is read for where it ends processes: no process is listed, started or ended,
and main() is never run."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "docs/gain_of_function_v4/coordinator/relaunch_h84_seg41_2026-09-29.py"
MAIN = r"C:\Users\bhart\Documents\UEMA_devA_private"
EXP = Path(MAIN) / "Saved" / "mimo_bridge" / "experiments"      # names only; nothing under it is read
PY = MAIN + r"\.venv_mimo\Scripts\python.exe"
WATCH = MAIN + r"\docs\h84_launch_2026-09-18\hourly_watch.py"
PARENT = EXP / "H84_fresh_drivers_20260918_seg41o_1431"
CAP_ARM = EXP / "H84_fresh_drivers_20260918_seg41c_1629"


def relaunch():
    spec = importlib.util.spec_from_file_location("relaunch_watchers_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_launch_with_parent_ends_the_active_runs_watcher_too():
    module = relaunch()
    runs = module.stopped_watcher_runs(PARENT, dict(active_run=str(CAP_ARM)))
    assert runs == [PARENT, CAP_ARM]
    selected = lambda cmdline: any(module.stale_watcher(cmdline, folder) for folder in runs)   # noqa: E731 (main's test)
    # the command lines of watcher processes 22080 (the venv launcher) and 5136 (its child) and of supervisor 29044, as
    # listed at 2:52 AM PDT, Oct 9 (the review's launch_prereq_review/processes_now.txt)
    watcher = [PY, "-u", WATCH, str(CAP_ARM), "8760"]
    supervisor = [PY, "-u", "-m", "mimo_bridge.tools.training_continuation", "--run", str(CAP_ARM), "--root",
                  str(EXP.parent), "--watcher", WATCH, "--replace-watcher"]
    assert selected(watcher) and not selected(supervisor)
    assert not module.stale_watcher(watcher, PARENT)                 # the rule before Oct 9 left it running
    # a watcher of another run (control run 2, seg41c_1433) is not one of them
    assert not selected([PY, "-u", WATCH, str(EXP / "H84_fresh_drivers_20260918_seg41c_1433"), "8760"])


def test_without_another_active_run_only_the_parents_watchers_are_ended_as_before():
    module = relaunch()
    assert module.stopped_watcher_runs(PARENT, dict(active_run=str(PARENT))) == [PARENT]
    assert module.stopped_watcher_runs(PARENT, dict(active_run=str(PARENT).replace("\\", "/") + "/")) == [PARENT]
    assert module.stopped_watcher_runs(PARENT, {}) == [PARENT]
    assert module.stopped_watcher_runs(PARENT, dict(active_run="")) == [PARENT]


def test_main_ends_the_watchers_only_after_its_last_refusal():
    """main() is never run here; its text is read. The re-check of 2557e6902 (Oct 9, 2026) found the watchers ended
    before the memory guard, which can still refuse; and a run folder that exists already was refused only by
    run.mkdir(), after the old supervisor had been ended. In main()'s text: the folder's refusal comes before the old
    supervisor is ended, and the guard's refusal before the watchers are ended and before the folder is made."""
    import inspect
    text = inspect.getsource(relaunch().main)
    at = text.index
    assert at("if run.exists():") < at("p.terminate()")                               # the supervisor's ending
    assert at("ok, reading = L.guard()") < at("watched = stopped_watcher_runs(")
    assert at("raise SystemExit(\"refused: the memory guard") < at("for p in stale:")
    assert at("for p in stale:") < at("\n    run.mkdir()")                 # the statement, not the comment naming it
    # every other refusal of main() comes before the first process is ended
    first_end = at("p.terminate()")
    for refusal in ("refused: a trainer is running", "is not a pause", "refused by the training preflight",
                    "raise SystemExit(\"refused: \" + stale)", "cannot be this segment's row"):
        assert at(refusal) < first_end, refusal
