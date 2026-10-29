# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The machine's mode, exactly one of TRAINING, TEST WINDOW and PAUSED FOR OPERATOR, and the week's hours in each.

The operator's recovery directive of Oct 5, 2026, section 12: "The top of the dashboard must always show exactly one
machine mode", and "Dashboard reports cumulative hours this week in each mode and the reason for every training
pause."

TWO KINDS OF TRAINER. A production trainer is a mimo_bridge.train_mimo process whose command line names a run folder
under Saved/mimo_bridge/experiments. An experiment trainer is a mimo_bridge.train_mimo process whose command line names
no such folder and which a test registry row names as its run: its run folder (the folder after --out on its command
line) is a folder the row's last run names (the folder after --out in the row's recorded command, or the folder of the
row's last run's log file; a row whose last run names a folder under Saved/mimo_bridge/experiments is production's),
or it is the row's recorded job process (the same process id, created within 1 s of the recorded creation time).
Found on Oct 13, 2026: four experiment trainer runs on D:/UEMA_Research/external/D/UEMA_scratch trained the model for 6.67 hours of the week
ending 4:30 AM PDT (the union of their lives as week() reads them), and the dashboard counted those hours as PAUSED
FOR OPERATOR.

THE MODE NOW (mode()), computed from the inputs read at the call:
  INPUT      the machine's processes (test_runner.scan_processes); Saved/mimo_bridge/supervisor_control.json (its cmd
             and reason) and the active run's stop_reason.json; the test registry's running rows with a live process
             and the live test processes (test_runner.audit); and, when a trainer process that names no production run
             folder is alive, the test registry's rows (to find the row that names its run).
  OPERATION  1. A production trainer process is alive: TRAINING.
             2. Else an experiment trainer process is alive: TRAINING, with the sentence "TRAINING (experiment run, not
                production): the trainer of run R (run folder F), row X, is alive (process N, since T)."
             3. Else a test that needs training stopped is live (a running row with a live process whose resource class
                is TEST_WINDOW), or the pause in supervisor_control.json is a pause runner's own (it names
                "pause_runner"): TEST WINDOW.
             4. Else: PAUSED FOR OPERATOR, with the reason written in supervisor_control.json when its cmd is pause,
                else the active run's stop reason, else the sentence that no reason is on record; the tests that run
                beside the hold (resource class BESIDE_TRAINING, and test processes no row names) are listed under it.
                The dashboard review of Oct 5, 2026, finding F9: a test that can run beside training made the band read
                TEST WINDOW while training was held for the operator.
  OUTPUT     {mode, sentence, reason, since, training_of ("production", "experiment" or None), experiment_trainers
             [{pid, row, run, folder, since_text, how}], unnamed_trainers [{pid, folder, since_text}], errors}. An error
             is listed when a test that needs training stopped is live beside a trainer (directive section 12), when
             training is stopped with no pause and no stop reason on record, and, in every mode, for each trainer
             process that is neither production's nor named by a row: the mode does not count it as training.

THE WEEK (week()): the last 7 x 24 hours ending now.
  INPUT      every run folder of the active run's lineage under Saved/mimo_bridge/experiments (production); every run
             folder a test registry row names as its experiment run (the folder after --out in the row's recorded
             command, or the folder of its last run's log file, outside Saved/mimo_bridge/experiments; only a folder
             holding replay_training_stats.jsonl with a "launch" record is a run folder). A run's life runs from its
             first "launch" stats record (the trainer's start) to its stage_done.json finished_at (the trainer's end;
             a run without one ends at its newest stats record; a run whose trainer process is alive ends now). Also
             read: the pause receipts under Saved/gof_observer_20260924/pauses; the test registry's rows (each row's
             last run); supervisor_control.json.
  OPERATION  production = the summed length of the union of the production lives inside the window; TRAINING = the
             summed length of the union of the production and experiment lives inside the window (an hour inside
             both counts once, as production); experiment = TRAINING - production. Each stretch with no production
             life (between two production lives, before the first and after the last) is a training pause. Inside a
             pause: experiment training hours = the union of the experiment lives inside it; TEST WINDOW hours = the
             union of the pause receipts' job spans and of the last runs of test rows whose resource class is
             TEST_WINDOW, outside the experiment lives (overlaps counted once); the rest of the pause is PAUSED FOR
             OPERATOR.
  OUTPUT     {hours {TRAINING, TEST WINDOW, PAUSED FOR OPERATOR}, training_split {production, experiment},
             experiment_runs [{row, run, folder, from_text, to_text, hours, ended_by}], pauses [{from, to, hours,
             test_window_hours, experiment_training_hours, experiment_runs, paused_for_operator_hours, reason,
             reason_source}], not_carried [what the records do not hold]}. The three hours sum to the window's 168
             hours, within the rounding of each to 0.01 hour.
A pause's reason is read, in this order, from: the stopped run's stop_reason.json; an "overseer_cmd" record of its
stats file that carries a reason; a pause receipt that starts inside the pause (its supervisor pause's reason); for
the pause still open, supervisor_control.json. Else the reason is "not recorded in the run folder".
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

from . import test_registry as tr

MODES = ("TRAINING", "TEST WINDOW", "PAUSED FOR OPERATOR")
MODE_MEANINGS = {
    "TRAINING": "a production trainer process is alive, or the trainer process of an experiment run that a test registry "
                "row names as its run (its run folder or its recorded job process) is alive",
    "TEST WINDOW": "no production trainer process and no trainer of a registered experiment run is alive, and a "
                   "registered test that needs training stopped is running or a pause runner's own pause holds "
                   "training for its tests",
    "PAUSED FOR OPERATOR": "no production trainer process and no trainer of a registered experiment run is alive, and no "
                           "test that needs training stopped is running: training waits for a person's word; tests "
                           "that can run beside training may run, and are listed",
}
WEEK_S = 7 * 24 * 3600.0
PAUSES_RELATIVE = Path("Saved/gof_observer_20260924/pauses")
LOOK_BACK_S = 3 * 24 * 3600.0     # a run folder last changed this long before the window's start is not read
NOT_CARRIED = (
    "supervisor_control.json holds only its present content: the reason of an earlier pause is on record only when "
    "the stopped run wrote a stop_reason.json, when its stats file's overseer_cmd record carries a reason, or when a "
    "pause receipt starts inside the pause",
    "a test registry row holds only its last run: an earlier run of the same row inside the week is not counted as "
    "test-window time, and an earlier experiment trainer run of the same row is not counted as training",
    "a test, a development run or an experiment trainer run that wrote no registry row and no pause receipt is in no "
    "record, and so is an experiment trainer run whose row names its folder neither by the folder after --out in the "
    "recorded command (the registry keeps a command's first 60 arguments) nor by its log file's folder: such time "
    "counts as PAUSED FOR OPERATOR",
    "a run folder holds no record of its trainer's process ending without stage_done.json: such a run is ended at "
    "its newest stats record",
)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def _hours(seconds) -> float:
    return round(max(0.0, seconds) / 3600.0, 2)


def _norm(text) -> str:
    return str(text).replace("\\", "/")


def _out_argument(cmdline):
    """The folder after --out (or in --out=FOLDER) on a command line (a list of arguments), else None."""
    args = [str(a) for a in cmdline or []] if isinstance(cmdline, (list, tuple)) else []
    for i, a in enumerate(args):
        if a == "--out" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--out="):
            return a[len("--out="):]
    return None


def _resolved(text, base) -> Path:
    """A folder named in a record: absolute as written, else under `base` (the data checkout)."""
    path = Path(_norm(text).rstrip("/"))
    return path if path.is_absolute() else Path(base) / path


def folder_key(path) -> str:
    """A run folder's path as two records are compared: slashes forward, no slash at the end, lower case (Windows
    paths do not tell case apart)."""
    return _norm(path).rstrip("/").lower()


def _row_started(last) -> float:
    at = last.get("started_at")
    if isinstance(at, (int, float)):
        return float(at)
    return tr.parse_clock(last.get("started")) or 0.0


# ------------------------------------------------------------------------------------------ the experiment trainers
def experiment_folders(rows, data_root=None) -> list:
    """The run folders test registry rows name as their experiment runs: [{row, folder (a Path), started_at}], the
    newest last run first. A row names a folder by its last run's recorded command (the folder after --out) and by the
    folder of its last run's log file. A row whose last run names a folder under Saved/mimo_bridge/experiments
    (test_runner.named_run) is production's and is left out. A folder two rows name belongs to the row whose last run
    started later. Whether a folder is a run folder (it holds replay_training_stats.jsonl) is not checked here."""
    from . import test_runner
    base = Path(data_root or tr.REPO)
    found = {}
    for row in rows or []:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        last = row.get("last_run") if isinstance(row.get("last_run"), dict) else None
        if not last or test_runner.named_run(row):
            continue
        started = _row_started(last)
        texts = [_out_argument(last.get("command"))]
        log = _norm(last.get("log") or "").strip().rstrip("/")
        if "/" in log:
            texts.append(log.rsplit("/", 1)[0])
        for text in texts:
            if not text:
                continue
            folder = _resolved(text, base)
            key = folder_key(folder)
            if key not in found or started > found[key]["started_at"]:
                found[key] = dict(row=row["id"], folder=folder, started_at=started)
    return sorted(found.values(), key=lambda e: -e["started_at"])


def other_trainers(processes) -> list:
    """The mimo_bridge.train_mimo processes whose command line names no run folder under Saved/mimo_bridge/experiments
    (not production's). A process whose parent is one too is the same trainer (the Windows venv launcher and its
    child), as test_runner.production_trainers counts it."""
    from . import test_runner
    found = [p for p in processes or [] if test_runner.is_trainer(p) and not test_runner.trainer_runs(p)]
    ids = {p["pid"] for p in found}
    return [p for p in found if p.get("ppid") not in ids]


def match_trainers(others, rows, data_root=None) -> tuple:
    """(the experiment trainers [{pid, created, row, run, folder, since_text, how}], the trainer processes no row names
    [{pid, created, folder, since_text}]) among `others` (other_trainers). A trainer is a row's when its run folder (the
    folder after --out) is a folder the row names (experiment_folders), else when it is the row's recorded job process
    (last_run.pid, created within 1 s of last_run.pid_created) and the row is not production's."""
    from . import test_runner
    base = Path(data_root or tr.REPO)
    owners = {folder_key(e["folder"]): e for e in experiment_folders(rows, base)}
    named, unnamed = [], []
    for p in others or []:
        out = _out_argument(p.get("cmdline"))
        folder = _resolved(out, base) if out else None
        entry = owners.get(folder_key(folder)) if folder is not None else None
        row_id, how = (entry["row"], "its run folder is the folder row %s names" % entry["row"]) if entry else (None, None)
        if row_id is None:
            for row in rows or []:
                last = row.get("last_run") if isinstance(row, dict) and isinstance(row.get("last_run"), dict) else {}
                created = last.get("pid_created")
                if last.get("pid") == p["pid"] and isinstance(created, (int, float)) \
                        and abs(float(created) - float(p["created"])) < 1.0 and not test_runner.named_run(row):
                    row_id, how = row["id"], "it is the recorded job process of row %s" % row["id"]
                    break
        shown = None if folder is None else _norm(folder)
        if row_id is None:
            unnamed.append(dict(pid=p["pid"], created=p["created"], folder=shown, since_text=tr.clock(p["created"])))
        else:
            named.append(dict(pid=p["pid"], created=p["created"], row=row_id,
                              run=None if folder is None else folder.name, folder=shown,
                              since_text=tr.clock(p["created"]), how=how))
    return named, unnamed


def unnamed_text(t) -> str:
    """The error line of a trainer process that is neither production's nor named by a row."""
    return ("Process %s (started %s) is a trainer (mimo_bridge.train_mimo) %s, and no test registry row names it (no "
            "row's last run names that folder by the folder after --out in its recorded command or by its log file's "
            "folder, and no row records it as its job process): the machine mode does not count its training."
            % (t["pid"], t["since_text"], "whose run folder is %s" % t["folder"] if t.get("folder") else
               "whose command line carries no --out argument"))


# ------------------------------------------------------------------------------------------ the mode now
def window_of(live_rows, control):
    """(the live rows that make a test window, whether the pause on record is a pause runner's own): a test window is a
    live test that needs training stopped (resource class TEST_WINDOW), or supervisor_control.json's pause written by a
    pause runner (it names "pause_runner"). A test that can run beside training makes none (the dashboard review of
    Oct 5, 2026, finding F9)."""
    window_rows = [r for r in live_rows or [] if r.get("resource_class") == "TEST_WINDOW"]
    runner_pause = isinstance(control, dict) and control.get("cmd") == "pause" and bool(control.get("pause_runner"))
    return window_rows, runner_pause


def window_class(row) -> bool:
    """True when a row's last run counts as test-window time in the week: its resource class (the runner's record,
    else the row's own cost) is TEST_WINDOW (finding F9)."""
    last = row.get("last_run") if isinstance(row, dict) and isinstance(row.get("last_run"), dict) else {}
    return (last.get("resource_class") or tr.default_resource_class(row)) == "TEST_WINDOW"


def _experiment_sentence(experiment) -> str:
    phrases = ["the trainer of run %s (run folder %s), row %s, is alive (process %s, since %s)"
               % (t["run"] or "with no --out argument", t["folder"] or "not on its command line", t["row"], t["pid"],
                  t["since_text"]) for t in experiment]
    return "TRAINING (experiment %s, not production): %s." % ("run" if len(experiment) == 1 else "runs",
                                                             "; ".join(phrases))


def mode(data_root=None, processes=None, audit_view=None, now=None, path=None) -> dict:
    """THE MODE NOW (the module text). `processes` and `audit_view` replace the machine's own scan and audit; `path`
    is the test registry (the data checkout's when None)."""
    from . import test_runner
    data_root = Path(data_root or tr.REPO)
    now = time.time() if now is None else float(now)
    saved = data_root / "Saved/mimo_bridge"
    processes = test_runner.scan_processes() if processes is None else processes
    control = _read(saved / "supervisor_control.json", {}) or {}
    continuation = _read(saved / "training_continuation.json", {}) or {}
    if audit_view is None:
        try:
            audit_view = test_runner.audit(path=path, root=data_root, now=now, processes=processes)
        except Exception as exc:
            audit_view = dict(available=False, running_rows=[], test_processes=[], development_runs=[],
                              error="%s: %s" % (type(exc).__name__, exc))
    trainers = test_runner.production_trainers(processes)
    run = Path(str(continuation.get("active_run") or "")) if continuation.get("active_run") else None
    stop = _read(run / "stop_reason.json") if run else None
    done = _read(run / "stage_done.json") if run else None
    live_rows = [r for r in audit_view.get("running_rows") or [] if r.get("alive") and not r.get("production_run")]
    live_tests = audit_view.get("test_processes") or []
    window_rows, runner_pause = window_of(live_rows, control)
    errors = []
    if not audit_view.get("available", True):
        errors.append("The audit of test processes could not run (%s): tests that are live are not counted in the "
                      "mode." % audit_view.get("error"))
    # the trainer processes that are not production's: each is a registered experiment run's or is named by no row
    experiment, unnamed = [], []
    others = other_trainers(processes)
    if others:
        try:
            rows = tr.load(path or tr.registry_path(data_root))["rows"]
        except Exception as exc:
            rows = []
            errors.append("The test registry could not be read (%s: %s): a trainer process outside production is not "
                          "matched to a row." % (type(exc).__name__, exc))
        experiment, unnamed = match_trainers(others, rows, data_root)
    own_rows = {t["row"] for t in experiment}
    since = None
    training_of = None
    if trainers or experiment:
        name = "TRAINING"
        reason = None
        if trainers:
            training_of = "production"
            runs = sorted({r for t in trainers for r in test_runner.trainer_runs(t)})
            since = min(t["created"] for t in trainers)
            sentence = "TRAINING: the trainer of run %s is alive (process %s, since %s)." % (
                ", ".join(runs), ", ".join(str(t["pid"]) for t in trainers), tr.clock(since))
            if run is not None and run.name not in runs:
                errors.append("The live trainer's run (%s) is not the run the supervisor names (%s)."
                              % (", ".join(runs), run.name))
        else:
            training_of = "experiment"
            since = min(t["created"] for t in experiment)
            sentence = _experiment_sentence(experiment)
        beside = [r for r in audit_view.get("running_rows") or [] if r.get("alive")
                  and r.get("resource_class") == "TEST_WINDOW" and r.get("id") not in own_rows]
        for r in beside:
            errors.append("Test %s needs training stopped (resource class TEST_WINDOW) and is running beside the "
                          "trainer." % r["id"])
    elif window_rows or runner_pause:
        name = "TEST WINDOW"
        ids = sorted({r["id"] for r in window_rows})
        sentence = "TEST WINDOW: no production trainer process is alive and %s." % (
            "%d %s that %s training stopped %s live: %s" % (
                len(ids), "test" if len(ids) == 1 else "tests", "needs" if len(ids) == 1 else "need",
                "is" if len(ids) == 1 else "are", ", ".join(ids)) if ids else
            "the pause in supervisor_control.json is a pause runner's own (%s), held for its tests"
            % control.get("pause_runner"))
        reason = str(control.get("reason") or "").strip() or None if control.get("cmd") == "pause" else None
    else:
        name = "PAUSED FOR OPERATOR"
        if control.get("cmd") == "pause":
            reason = str(control.get("reason") or "").strip() or "supervisor_control.json says pause and gives no reason"
            source = "supervisor_control.json"
        elif isinstance(stop, dict) and stop.get("reason"):
            reason, source = str(stop["reason"]), "the active run's stop_reason.json"
        else:
            reason, source = None, None
            errors.append("Training is stopped: no production trainer process is alive, supervisor_control.json does "
                          "not say pause, and the active run holds no stop reason.")
        sentence = ("PAUSED FOR OPERATOR: no production trainer process is alive and no test that needs training "
                    "stopped is running. " +
                    ("Reason on record (%s): %s" % (source, reason.rstrip(".") + ".") if reason else
                     "No reason is on record."))
        # the tests that run beside the hold: rows with a live process whose class lets them run beside training, and
        # live test processes no row names (the audit lists each of the latter as an error)
        beside_ids = sorted({r["id"] for r in live_rows} | {t["row"] for t in live_tests if t.get("row")})
        unnamed_tests = sorted(str(t.get("pid")) for t in live_tests if not t.get("row"))
        if beside_ids or unnamed_tests:
            listed = beside_ids + (["%d test %s that no row names (%s %s)" % (
                len(unnamed_tests), "process" if len(unnamed_tests) == 1 else "processes",
                "process" if len(unnamed_tests) == 1 else "processes", ", ".join(unnamed_tests))]
                if unnamed_tests else [])
            count = len(beside_ids) + len(unnamed_tests)
            sentence += " %d %s beside the hold: %s." % (count, "test runs" if count == 1 else "tests run",
                                                          ", ".join(listed))
        if isinstance(done, dict) and isinstance(done.get("finished_at"), (int, float)):
            since = float(done["finished_at"])
    for t in unnamed:
        errors.append(unnamed_text(t))
    held_at = done.get("steps") if isinstance(done, dict) else None
    return dict(mode=name, modes=list(MODES), meaning=MODE_MEANINGS[name], sentence=sentence, reason=reason,
                since=since, since_text=None if since is None else tr.clock(since), training_of=training_of,
                active_run=None if run is None else run.name, held_at_step=held_at,
                trainers=[dict(pid=t["pid"], runs=test_runner.trainer_runs(t)) for t in trainers],
                experiment_trainers=[{k: v for k, v in t.items() if k != "created"} for t in experiment],
                unnamed_trainers=[{k: v for k, v in t.items() if k != "created"} for t in unnamed],
                live_tests=sorted({r["id"] for r in live_rows} | {t["row"] for t in live_tests if t.get("row")}),
                development_runs=len(audit_view.get("development_runs") or []), errors=errors,
                read_at_text=tr.clock(now))


# ------------------------------------------------------------------------------------------ the week
def _launch_record(stats):
    try:
        with open(stats, "rb") as stream:
            for line in stream.read(262_144).split(b"\n"):
                if b'"type": "launch"' in line[:48]:
                    row = json.loads(line)
                    return row if isinstance(row, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def _tail_records(stats, size=1_048_576):
    """(the newest record's timestamp, the newest overseer_cmd record) from the last `size` bytes of a stats file."""
    newest, command = None, None
    try:
        with open(stats, "rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            stream.seek(max(0, end - size))
            lines = stream.read().split(b"\n")
    except OSError:
        return None, None
    for line in lines:
        head = line[:64]
        if b'"type": "overseer_cmd"' in head:
            try:
                command = json.loads(line)
            except ValueError:
                pass
        match = re.search(rb'"timestamp": ([0-9.]+)', line[:200])
        if match:
            try:
                newest = max(newest or 0.0, float(match.group(1)))
            except ValueError:
                pass
    return newest, command


def _changed_before(folder, since) -> bool:
    """True when the folder and its stats file were last changed before `since` (or cannot be read)."""
    try:
        return max(folder.stat().st_mtime, (folder / "replay_training_stats.jsonl").stat().st_mtime) < since
    except OSError:
        return True


def life_of(folder):
    """The trainer's life of one run folder: {run, start, end, ended_by, stop_reason, command, resumed_from_step,
    code_commit}; None when its stats file holds no "launch" record. start is the run's first "launch" record; end is
    stage_done.json's finished_at, else the newest stats record."""
    folder = Path(folder)
    stats = folder / "replay_training_stats.jsonl"
    launch = _launch_record(stats)
    start = launch.get("timestamp")
    if not isinstance(start, (int, float)):
        return None
    done = _read(folder / "stage_done.json")
    newest, command = _tail_records(stats)
    end = done.get("finished_at") if isinstance(done, dict) and isinstance(done.get("finished_at"),
                                                                         (int, float)) else newest
    stop = _read(folder / "stop_reason.json")
    return dict(run=folder.name, start=float(start), end=None if end is None else float(end),
                ended_by="stage_done.json" if isinstance(done, dict) else "its newest stats record",
                stop_reason=stop.get("reason") if isinstance(stop, dict) else None, command=command,
                resumed_from_step=launch.get("step"), code_commit=launch.get("code_commit"))


def run_lives(data_root, lineage=None, since=None) -> list:
    """The trainer's lives of the lineage's run folders under Saved/mimo_bridge/experiments: [{run, start, end,
    ended_by, stop_reason, command, ...}] (life_of) sorted by start. Folders last changed before `since` are skipped."""
    experiments = Path(data_root) / "Saved/mimo_bridge/experiments"
    out = []
    try:
        folders = [p for p in experiments.iterdir() if p.is_dir() and (lineage is None or p.name.startswith(lineage))]
    except OSError:
        return out
    for folder in folders:
        if since is not None and _changed_before(folder, since):
            continue
        life = life_of(folder)
        if life is not None:
            out.append(life)
    out.sort(key=lambda r: r["start"])
    return out


def experiment_lives(rows, data_root=None, since=None) -> list:
    """The trainer's lives of the run folders test registry rows name as their experiment runs (experiment_folders):
    [life_of fields + {row, folder}] sorted by start. A folder that holds no replay_training_stats.jsonl with a
    "launch" record is no run folder and is skipped, and so is a folder last changed before `since`."""
    out = []
    for entry in experiment_folders(rows, data_root):
        folder = entry["folder"]
        if since is not None and _changed_before(folder, since):
            continue
        life = life_of(folder)              # None when the folder holds no stats file with a "launch" record
        if life is not None:
            out.append(dict(life, row=entry["row"], folder=_norm(folder)))
    out.sort(key=lambda r: r["start"])
    return out


def _receipts(data_root, since) -> list:
    """The pause receipts written since `since`: [{folder, start, end, reason, jobs}] (job spans from started to
    finished)."""
    out = []
    folder = Path(data_root) / PAUSES_RELATIVE
    try:
        entries = [p for p in folder.iterdir() if p.is_dir() and p.stat().st_mtime >= since]
    except OSError:
        return out
    for entry in entries:
        receipt = _read(entry / "receipt.json")
        if not isinstance(receipt, dict) or not isinstance(receipt.get("started"), (int, float)):
            continue
        control = receipt.get("owned_control") or receipt.get("held_by") or {}
        out.append(dict(folder=entry.name, start=float(receipt["started"]),
                        end=float(receipt.get("finished") or receipt["started"]),
                        reason=str(control.get("reason") or "").strip() or None,
                        jobs=[j.get("name") for j in receipt.get("jobs") or [] if isinstance(j, dict)]))
    return out


def _union_inside(spans, low, high) -> float:
    """The summed length of the spans' parts inside [low, high], overlaps counted once."""
    clipped = sorted((max(a, low), min(b, high)) for a, b in spans if b > low and a < high)
    total, at = 0.0, low
    for a, b in clipped:
        a = max(a, at)
        if b > a:
            total += b - a
            at = b
    return total


def week(data_root=None, now=None, span_s=WEEK_S, path=None) -> dict:
    """THE WEEK (the module text)."""
    from . import test_runner
    data_root = Path(data_root or tr.REPO)
    now = time.time() if now is None else float(now)
    low = now - span_s
    saved = data_root / "Saved/mimo_bridge"
    continuation = _read(saved / "training_continuation.json", {}) or {}
    active = Path(str(continuation.get("active_run") or "")).name
    match = re.match(r"(.+?)_seg\d", active)
    lineage = match.group(1) if match else (active or None)
    try:
        rows = tr.load(path or tr.registry_path(data_root))["rows"]
    except tr.RegistryError:
        rows = []
    lives = run_lives(data_root, lineage, since=low - LOOK_BACK_S)
    scan = test_runner.scan_processes()
    alive_runs = {r for t in test_runner.production_trainers(scan) for r in test_runner.trainer_runs(t)}
    alive_folders = {folder_key(_resolved(out, data_root)) for p in other_trainers(scan)
                     for out in [_out_argument(p.get("cmdline"))] if out}

    def inside(life, alive):
        end = now if alive else life["end"]
        if end is None or end <= low or life["start"] >= now:
            return None
        return dict(life, end=min(end, now))
    spans = [s for s in (inside(life, life["run"] in alive_runs) for life in lives) if s is not None]
    experiments = [s for s in (inside(life, folder_key(life["folder"]) in alive_folders)
                               for life in experiment_lives(rows, data_root, since=low - LOOK_BACK_S)) if s is not None]
    experiment_spans = [(s["start"], s["end"]) for s in experiments]
    control = _read(saved / "supervisor_control.json", {}) or {}
    receipts = _receipts(data_root, low - 24 * 3600.0)
    test_spans = [(r["start"], r["end"]) for r in receipts]
    for row in rows:
        last = row.get("last_run") if isinstance(row, dict) and isinstance(row.get("last_run"), dict) else {}
        if last.get("resource_class") == "PRODUCTION_SEGMENT" or test_runner.named_run(row):
            continue                          # a production segment's run is training, not a test window
        if not window_class(row):
            continue                          # a test that can run beside training does not make a test window (F9)
        a = last.get("started_at") or tr.parse_clock(last.get("started"))
        b = last.get("ended_at") or tr.parse_clock(last.get("ended")) or (now if row.get("status") == "running"
                                                                          else None)
        if a is not None and b is not None and b > a:
            test_spans.append((float(a), float(b)))
    production_spans = [(s["start"], s["end"]) for s in spans]
    production = _union_inside(production_spans, low, now)
    training = _union_inside(production_spans + experiment_spans, low, now)
    pauses, at, previous = [], low, None
    edges = [(s["start"], s["end"], s) for s in sorted(spans, key=lambda s: s["start"])]
    for start, end, life in edges + [(now, now, None)]:
        if start > at + 1.0:
            # inside the pause: experiment training first, then test-window time outside it, then the rest
            trained = _union_inside(experiment_spans, at, start)
            busy = _union_inside(test_spans + experiment_spans, at, start)
            reason, source = None, None
            if previous is not None and previous.get("stop_reason"):
                reason, source = str(previous["stop_reason"]), "%s/stop_reason.json" % previous["run"]
            elif previous is not None and isinstance(previous.get("command"), dict) and previous["command"].get(
                    "reason"):
                reason, source = str(previous["command"]["reason"]), "the overseer_cmd record of %s" % previous["run"]
            else:
                found = [r for r in receipts if at - 60 <= r["start"] <= start and r.get("reason")]
                if found:
                    reason, source = found[0]["reason"], "pause receipt %s" % found[0]["folder"]
                elif life is None and control.get("cmd") == "pause" and control.get("reason"):
                    reason, source = str(control["reason"]), "supervisor_control.json"
            pauses.append(dict(
                start=at, end=start, from_text=tr.clock(at), to_text="now" if life is None else tr.clock(start),
                hours=_hours(start - at), test_window_hours=_hours(busy - trained),
                experiment_training_hours=_hours(trained),
                experiment_runs=[e["run"] for e in experiments if e["end"] > at and e["start"] < start],
                paused_for_operator_hours=_hours(start - at - busy),
                after_run=None if previous is None else previous["run"],
                stopped_at_step=None if previous is None or not isinstance(previous.get("command"), dict)
                else previous["command"].get("step"),
                reason=reason or "not recorded in the run folder", reason_source=source, open=life is None))
        at = max(at, end)
        previous = life or previous
    test_window = sum(p["test_window_hours"] for p in pauses)
    paused = sum(p["paused_for_operator_hours"] for p in pauses)
    return dict(window_from=tr.clock(low), window_to=tr.clock(now), window_hours=_hours(span_s),
                hours={"TRAINING": _hours(training), "TEST WINDOW": round(test_window, 2),
                       "PAUSED FOR OPERATOR": round(paused, 2)},
                training_split={"production": _hours(production), "experiment": _hours(training - production)},
                experiment_runs=[dict(row=e["row"], run=e["run"], folder=e["folder"], from_text=tr.clock(
                    max(e["start"], low)), to_text="now" if e["end"] >= now else tr.clock(e["end"]),
                    hours=_hours(_union_inside([(e["start"], e["end"])], low, now)), ended_by=e["ended_by"])
                    for e in experiments],
                pauses=pauses, runs=len(spans), lineage=lineage, not_carried=list(NOT_CARRIED),
                definition=("The week is the last 168 hours ending now. TRAINING: the trainer's lives, from each "
                            "run's launch record to its stage_done.json (else its newest stats record), of the "
                            "production lineage's run folders under Saved/mimo_bridge/experiments (production) and "
                            "of the run folders test registry rows name as their experiment runs, outside "
                            "Saved/mimo_bridge/experiments (experiment: the folder after --out in a row's recorded "
                            "command, or its last run's log file's folder); an hour inside both counts once, as "
                            "production. A training pause is a stretch with no production trainer life. TEST WINDOW: "
                            "inside a pause and outside experiment lives, the spans of pause receipts and of the last "
                            "runs of test rows that need training stopped (resource class TEST_WINDOW). PAUSED FOR "
                            "OPERATOR: the rest of each pause."))
