# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A process's own record of every file it writes inside one folder (Oct 5, 2026).

The second dashboard (development_dashboard --root PRODUCTION --write-log FILE) serves production's data with another
checkout's code and must write nothing into production's checkout. Comparing file times there cannot show that:
production's own dashboard writes the same folders. This module gives the evidence from inside the process:

    INPUT      `root` (the folder to watch) and `log_path` (a file outside it).
    OPERATION  sys.addaudithook installs a function that Python calls for every audited event of this process. For
               the events that create or change a file - "open" with a write, append, create or truncate mode or flag,
               "os.rename" (also raised by os.replace), "os.remove", "os.rmdir", "os.mkdir", "os.truncate", "os.link",
               "os.symlink", "shutil.rmtree", "shutil.copyfile", "shutil.move" - it resolves each path argument and,
               when the path lies inside `root`, appends one JSON line {at, event, paths} to `log_path`.
    OUTPUT     `log_path`, created empty when the hook is installed (so a log that exists shows the hook ran), then
               one line per write inside `root`. An empty file after the pages were served is the evidence that the
               process wrote nothing there. The hook cannot be removed: it lasts as long as the process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
PATH_EVENTS = ("os.rename", "os.remove", "os.rmdir", "os.mkdir", "os.truncate", "os.link", "os.symlink",
               "shutil.rmtree", "shutil.copyfile", "shutil.move")


def _text(path):
    """The text of a path argument, or None for an argument that is not a path (a file descriptor, None)."""
    try:
        value = os.fspath(path)
    except TypeError:
        return None
    return os.fsdecode(value) if isinstance(value, bytes) else value


def inside(path, root_text) -> bool:
    text = _text(path)
    if text is None:
        return False
    full = os.path.abspath(text).replace("\\", "/").lower().rstrip("/")
    return full == root_text or full.startswith(root_text + "/")


def is_write(mode, flags) -> bool:
    """True when an "open" event's mode string or flags create or change the file."""
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & WRITE_FLAGS)


def watch_writes(root, log_path) -> None:
    """Install the audit hook (the module text) for this process."""
    root_text = str(Path(root).resolve()).replace("\\", "/").lower().rstrip("/")
    log_path = str(Path(log_path).resolve())
    if inside(log_path, root_text):
        raise ValueError("the write log %s lies inside the watched folder %s" % (log_path, root))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    # the log is created, empty, when the hook is installed: a log that exists shows the hook ran in this process
    open(log_path, "w", encoding="utf-8").close()
    busy = threading.local()

    def hook(event, args):
        if getattr(busy, "on", False):
            return
        if event == "open":
            args = tuple(args) + (None, None, None)
            paths = [args[0]] if is_write(args[1], args[2]) else []
        elif event in PATH_EVENTS:
            paths = list(args[:2])
        else:
            return
        hits = [_text(p) for p in paths if inside(p, root_text)]
        if not hits:
            return
        busy.on = True
        try:
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(at=time.time(), event=event, paths=hits)) + "\n")
        except OSError:
            pass
        finally:
            busy.on = False
    sys.addaudithook(hook)


def read_log(log_path) -> list:
    """The log's lines as dicts; [] for a missing or empty log."""
    try:
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            out.append(dict(event="unreadable line", paths=[line[:200]]))
    return out
