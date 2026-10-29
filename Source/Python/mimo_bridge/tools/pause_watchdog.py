# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Run a command only while training is paused; end it the moment the pause lifts.

    python -m mimo_bridge.tools.pause_watchdog [--poll 5] -- <command ...>

Agents run their guarded tests and measurements inside coordinated training pauses. Twice on 2026-09-25 a run
started inside a pause kept going after the pause lifted, beside the relaunched production trainer (3:48-3:50 PM
PDT, 8.4 GB; 8:07:54-8:08:34 PM), because the pause was checked only when the run started. This checks before
starting and every --poll seconds while the command runs. When Saved/mimo_bridge/supervisor_control.json no longer
says "pause" (an unreadable file counts as lifted: the supervisor itself then keeps training stopped, but nobody can
tell), it ends the command's whole process tree and exits 5. The supervisor relaunches training on its own poll
(15 s) and a trainer takes longer than that to build its worlds, so a 5 s poll ends the run first.

A REPAIR RUN TRAINS UNDER A PAUSE (the relaunch review of Oct 5, 2026, finding F2): the training preflight's repair run
leaves supervisor_control.json {"cmd": "pause", "repair_run": <run folder>} while its trainer runs, so that nothing
relaunches it. Such a pause counts as training, not paused, while a trainer of that run folder is alive: no command
starts beside the repair trainer, and a command already running is ended at the next poll once it starts. After the
repair trainer ends, the same pause counts as paused again (the measurements of its gates run then).

Exit codes: the command's own, or 4 when training is not paused at the start, or 5 when the pause lifted mid-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[4]
CONTROL = ROOT / "Saved" / "mimo_bridge" / "supervisor_control.json"


def trainer_of(run) -> bool:
    """True while a trainer process (mimo_bridge.train_mimo) names the run folder `run` among its arguments."""
    want = os.path.normcase(os.path.abspath(str(run)))
    for p in psutil.process_iter(["cmdline"]):
        try:
            args = p.info["cmdline"] or []
        except psutil.Error:
            continue
        if "mimo_bridge.train_mimo" in args and any(
                os.path.normcase(os.path.abspath(str(x))) == want for x in args[1:]):
            return True
    return False


def paused(control: Path = None, alive=None) -> bool:
    """True when supervisor_control.json says pause and no repair run named by it trains (`alive`: the test of a live
    trainer of a run folder, trainer_of unless given)."""
    control = CONTROL if control is None else control
    try:
        value = json.loads(Path(control).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if not (isinstance(value, dict) and value.get("cmd") == "pause"):
        return False
    if value.get("repair_run"):
        return not (alive or trainer_of)(value["repair_run"])
    return True


def end_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
        family = parent.children(recursive=True) + [parent]
    except psutil.Error:
        return
    for p in family:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(family, timeout=20)


def run(command, poll: float = 5.0, control: Path = None, alive=None) -> int:
    if not paused(control, alive):
        print("pause_watchdog: training is not paused; not starting", flush=True)
        return 4
    proc = subprocess.Popen(command)
    while True:
        try:
            return proc.wait(timeout=poll)
        except subprocess.TimeoutExpired:
            pass
        if not paused(control, alive):
            end_tree(proc)
            print("pause_watchdog: the pause lifted; the run was ended", flush=True)
            return 5


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    cut = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args(argv[:cut])
    command = argv[cut + 1:]
    if not command:
        print("pause_watchdog: no command given", file=sys.stderr)
        return 2
    return run(command, a.poll)


if __name__ == "__main__":
    sys.exit(main())
