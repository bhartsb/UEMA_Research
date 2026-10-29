# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""tools/pause_watchdog ends a run the moment the training pause lifts (two runs outlived their pause on 2026-09-25)."""
import json
import sys
import threading
import time

from mimo_bridge.tools import pause_watchdog as W


def _control(path, cmd):
    path.write_text(json.dumps({"cmd": cmd}), encoding="utf-8")


def test_it_will_not_start_outside_a_pause(tmp_path):
    c = tmp_path / "control.json"
    _control(c, "run")
    assert W.run([sys.executable, "-c", "print(1)"], poll=0.1, control=c) == 4
    assert W.run([sys.executable, "-c", "print(1)"], poll=0.1, control=tmp_path / "missing.json") == 4


def test_a_run_inside_the_pause_returns_its_own_exit_code(tmp_path):
    c = tmp_path / "control.json"
    _control(c, "pause")
    assert W.run([sys.executable, "-c", "raise SystemExit(7)"], poll=0.1, control=c) == 7


def test_the_run_and_its_children_end_when_the_pause_lifts(tmp_path):
    c = tmp_path / "control.json"
    marker = tmp_path / "child_pid.txt"
    _control(c, "pause")
    code = ("import subprocess,sys,time,pathlib;"
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
            "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
    threading.Timer(1.5, _control, args=(c, "run")).start()
    t0 = time.monotonic()
    assert W.run([sys.executable, "-c", code, str(marker)], poll=0.2, control=c) == 5
    assert time.monotonic() - t0 < 30
    import psutil
    child = int(marker.read_text())
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_an_unreadable_control_counts_as_lifted(tmp_path):
    c = tmp_path / "control.json"
    c.write_bytes('{"cmd": "pause"}'.encode("utf-16"))
    assert W.paused(c) is False
