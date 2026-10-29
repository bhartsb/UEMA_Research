# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Run the gain-of-function probe suite on every new complete checkpoint of a lineage (D15).

    python -m mimo_bridge.tools.gof_probe_watch --lineage H84_fresh_drivers_20260918 [--interval 300]
    python -m mimo_bridge.tools.gof_probe_watch --lineage H84_fresh_drivers_20260918 --once

A separate process that never touches the trainer. Every `interval` seconds it lists the
checkpoints of every run folder whose name starts with the lineage prefix
(Saved/mimo_bridge/experiments/<prefix>*/checkpoints/ckpt_step_*), keeps those that are
COMPLETE by the training supervisor's own test (tools/training_continuation.complete_candidates:
weights, optimizer, runtime state, receipt, and the episode companion's receipt when the model
requires one) and whose newest file is at least a minute old, and skips any whose four probes
need no run (gof_probe_harness.needs_run: a result is in the index, or failures written by THIS
code SAME_CODE_ATTEMPTS times in a row; a failure written by an older commit or harness version
is run again, and a first failure of this code once more on a later pass). For each remaining
one, oldest first, one at a time: the commit guard is read (the suite's measured need,
gof_probe_harness.NEED_GB, must leave the guard's reserve under the commit limit; otherwise it
waits for the next pass), then the suite runs in a CHILD process on the CPU
(CUDA_VISIBLE_DEVICES=-1), so every gigabyte it held is returned when it ends.

WHERE IT STARTS. The first start of a lineage records a START STEP in watch_status.json: the
newest complete checkpoint's step at that moment (0 with --include-existing, or --start-step N).
Every later start of the same lineage reads it back, so checkpoints written while the watcher
was down, or still queued when it stopped, are measured on the next start instead of being taken
for "already there" (review of the probes, 2026-09-23). Checkpoints at or below the start step
are left alone.

COST BESIDE A RUNNING TRAINER: UNMEASURED. The suite took 877 s at a 3.57 GB peak with training
paused (2026-09-22). The hourly watcher's step-rate stop rule stopped segment 31, so the trainer's
model steps per second with one suite running beside it must be measured before this watcher is
started beside a trainer (a deploy step in docs/gain_of_function_v3/agents/fix-probes.md).

It writes Saved/mimo_bridge/gof_probes/watch_status.json (what it is doing, for dashboards, and
the start step per lineage) and appends to Saved/mimo_bridge/gof_probes/watch.log.
"""
from __future__ import annotations

import os
if __name__ == "__main__":
    # RUN AS A PROGRAM, A PROBE WORKS ON THE CPU: the card is hidden here, before torch starts, as it always was.
    # Importing the module never changes the process's devices (2026-09-26): the hide used to run at import, so a
    # test process that collected any probe test lost the card and skipped the card continuation test. A program
    # that imports a probe and wants the CPU hides the card at its own entry point (record:
    # docs/gain_of_function_v4/agents/probe_gpu_hiding_2026-09-26.md).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

PROBES = ("gaze", "body", "binding", "posture")
MIN_AGE_S = 60.0          # a checkpoint's newest file must be this old: the trainer writes a save as
                          # a temporary folder renamed into place, and the companion is written in
                          # the same transaction; a minute is ample margin over that rename.


def _root():
    from mimo_bridge.tools import gof_probe_harness as H
    return H.PROBES_ROOT


def complete_checkpoints(experiments: Path, prefix: str):
    """(step, path) of every complete checkpoint in run folders starting with prefix."""
    from mimo_bridge.tools.training_continuation import complete_candidates
    out = []
    for run in sorted(experiments.glob(prefix + "*")):
        if not run.is_dir():
            continue
        for path in complete_candidates(run):
            try:
                out.append((int(path.name.removeprefix("ckpt_step_")), path))
            except ValueError:
                continue
    return sorted(out, key=lambda x: (x[0], str(x[1])))


def newest_mtime(path: Path) -> float:
    return max((f.stat().st_mtime for f in path.iterdir() if f.is_file()), default=0.0)


def fully_indexed(checkpoint: Path, root: Path) -> bool:
    """True when none of the four probes needs a run on this checkpoint (H.needs_run)."""
    from mimo_bridge.tools import gof_probe_harness as H
    return not any(H.needs_run(str(checkpoint), p, root) for p in PROBES)


def pending(experiments: Path, prefix: str, root: Path, start_step: int, now: float,
            min_age: float = MIN_AGE_S, gave_up: Optional[set] = None):
    """(step, path) of complete checkpoints above start_step, old enough, needing a probe run,
    and not given up on in this process."""
    out = []
    for step, path in complete_checkpoints(experiments, prefix):
        if step <= int(start_step):
            continue
        if gave_up and os.path.normcase(str(path.resolve())) in gave_up:
            continue
        if now - newest_mtime(path) < min_age:
            continue
        if fully_indexed(path, root):
            continue
        out.append((step, path))
    return out


def _log(root: Path, msg: str):
    root.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with open(root / "watch.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def read_status(root: Path) -> dict:
    try:
        return json.loads((root / "watch_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _status(root: Path, **kw):
    """Write watch_status.json, keeping every lineage's start step already recorded there."""
    from mimo_bridge.tools import gof_probe_harness as H
    old = read_status(root)
    starts = dict(old.get("start_steps") or {})
    starts.update(kw.pop("start_steps", {}) or {})
    H._atomic_json(root / "watch_status.json", dict(kw, start_steps=starts, pid=os.getpid(),
                                                    time=time.strftime("%Y-%m-%d %H:%M:%S")))


def start_step_for(root: Path, lineage: str, experiments: Path, *, include_existing: bool = False,
                   explicit: Optional[int] = None) -> tuple:
    """(step, how): the step above which checkpoints are probed. An explicit --start-step wins;
    then the step recorded for this lineage by an earlier start; else 0 with --include-existing;
    else the newest complete checkpoint's step now (a first start does not queue the lineage's
    whole history)."""
    if explicit is not None:
        return int(explicit), "given with --start-step"
    rec = (read_status(root).get("start_steps") or {}).get(lineage)
    if isinstance(rec, dict) and rec.get("step") is not None:
        return int(rec["step"]), "recorded in watch_status.json at %s" % rec.get("since")
    if include_existing:
        return 0, "--include-existing on a first start"
    found = complete_checkpoints(experiments, lineage)
    return (found[-1][0] if found else 0), "the newest complete checkpoint at the first start"


def run_one(path: Path, root: Path, python: str = sys.executable) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    cmd = [python, "-m", "mimo_bridge.tools.gof_probe_suite", "--ckpt", str(path), "--skip-indexed"]
    if root != _root():
        cmd += ["--root", str(root)]
    here = Path(__file__).resolve().parents[2]            # Source/Python
    env["PYTHONPATH"] = str(here) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=str(here), env=env)
    return int(proc.returncode)


def main(argv=None) -> int:
    from mimo_bridge.tools.gof_probe_suite import guard
    from mimo_bridge.tools import gof_probe_harness as H
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lineage", required=True, help="run-folder name prefix, e.g. H84_fresh_drivers_20260918")
    ap.add_argument("--experiments", default=str(H.SAVED / "experiments"))
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between passes")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--include-existing", action="store_true",
                    help="on a lineage's first start, also probe checkpoints that were already complete")
    ap.add_argument("--start-step", type=int, default=None,
                    help="probe checkpoints above this step (recorded for the lineage from now on)")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else _root()
    experiments = Path(a.experiments)
    start, how = start_step_for(root, a.lineage, experiments, include_existing=a.include_existing,
                                explicit=a.start_step)
    record = (read_status(root).get("start_steps") or {}).get(a.lineage)
    if a.start_step is not None or not isinstance(record, dict):
        record = dict(step=start, since=time.strftime("%Y-%m-%d %H:%M:%S"), how=how)
    starts = {a.lineage: record}
    _log(root, f"watching {a.lineage}* in {experiments}; probing checkpoints above step {start} ({how}); "
               f"harness {H.HARNESS_VERSION}, code {H.code_commit().get('commit')}")
    _status(root, state="starting", lineage=a.lineage, start_steps=starts)
    gave_up, attempts = set(), {}
    while True:
        # the commit the suite children will stamp (H.refresh_code_commit: judged with the commit
        # this process started on, a newer child's failure would count as other code for ever)
        H.refresh_code_commit()
        todo = pending(experiments, a.lineage, root, start, time.time(), gave_up=gave_up)
        _status(root, state="idle" if not todo else "pending", pending=[str(p) for _, p in todo],
                lineage=a.lineage, start_steps=starts)
        for step, path in todo:
            ok, reading = guard()
            if not ok:
                _log(root, f"waiting: commit guard refused ({reading}) before {path.name}")
                _status(root, state="waiting for memory", reading=reading, next=str(path), lineage=a.lineage,
                        start_steps=starts)
                break
            _log(root, f"probing {path} ({reading})")
            _status(root, state="probing", checkpoint=str(path), step=step, lineage=a.lineage, start_steps=starts)
            t0 = time.time()
            code = run_one(path, root)
            _log(root, f"finished {path.name} in {time.time() - t0:.0f} s, exit {code}")
            if code != 0 and not fully_indexed(path, root):
                # a probe that fails writes its own failure record: it is run again on a later
                # pass until it has failed H.SAME_CODE_ATTEMPTS times on this code, then waits for
                # other code (H.needs_run). A suite process that dies without writing records is
                # run again on a later pass too; after H.SAME_CODE_ATTEMPTS nonzero exits this
                # process gives the checkpoint up, so one bad checkpoint cannot loop the watcher;
                # a restart retries it
                key = os.path.normcase(str(path.resolve()))
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] >= H.SAME_CODE_ATTEMPTS:
                    gave_up.add(key)
        if a.once:
            return 0
        time.sleep(max(5.0, float(a.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
