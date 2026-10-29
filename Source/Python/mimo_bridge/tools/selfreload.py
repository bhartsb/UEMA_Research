# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Long-running tools re-exec themselves when their source changes on disk.

Three times in one day a daemon kept running code that had since been fixed: the
dashboard overwrote a rewritten page with the old layout twice, and the overseer spent
half a day 'checking' runs with no knowledge of the vitals bands that had been added -
the exact violation it was built to catch sat on disk unread. nohup keeps a process
alive across everything, including its own obsolescence. So each loop calls
exec_if_changed(): newest .py mtime under mimo_bridge moved -> replace this process with
a fresh one running the current code, same module, same arguments.
"""
from __future__ import annotations

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../mimo_bridge


def stamp() -> float:
    t = 0.0
    for p in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        try:
            t = max(t, os.path.getmtime(p))
        except OSError:
            pass
    return t


def exec_if_changed(t0: float, module: str, argv, log=None) -> None:
    if stamp() > t0 + 0.5:
        if log:
            log(f"{module}: source changed on disk; re-executing with current code")
        # On Windows os.execv does NOT replace the process - it spawns a child and the
        # parent lingers, so every reload doubled the daemon (two overseers, two
        # dashboards observed 2026-08-24). Spawn detached, then exit hard.
        import subprocess
        # CREATE_NO_WINDOW, not DETACHED_PROCESS: on Windows a detached console app gets
        # its OWN VISIBLE CONSOLE, so every reload popped a window on the user's desktop.
        subprocess.Popen([sys.executable, "-m", module] + list(argv),
                         creationflags=0x08000000,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, close_fds=True)
        os._exit(0)
