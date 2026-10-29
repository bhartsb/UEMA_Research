# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Paths the records name on the development machine, read from wherever this checkout is.

The records this release carries (the test registry, run statistics, results, observer and movement-viewer indexes,
the explainer's sources) name files by absolute path inside D:\\UEMA_Research, the folder the release was built in;
the saved training state and its command name files inside C:\\Users\\bhart\\Documents\\UEMA_devA_private, the
development repository the state was trained in (the caregiver speech library, the calibrations); the saved worlds
name the development machine's data folders (D:\\UEMA_scratch and others, which the release mirrors under external/D);
some tool defaults name any of them. Every such path is re-rooted at this checkout (RECORDED_ROOTS):
D:\\UEMA_Research\\docs\\x.json -> <checkout>\\docs\\x.json (on Linux and macOS, <checkout>/docs/x.json),
D:\\UEMA_scratch\\run\\scenes -> <checkout>\\external\\D\\UEMA_scratch\\run\\scenes.

localize(path) does that for one path (str in, str out; bytes in, bytes out; a pathlib path keeps its type; any
other path unchanged). install() makes Python's own file functions do it for every path they are given:
builtins.open/io.open, os.open, os.stat/lstat, os.listdir/scandir, os.mkdir, os.remove/unlink/rmdir,
os.rename/replace, os.utime, os.chmod, os.access and os.path.exists/isfile/isdir/islink/lexists, so pathlib, glob,
shutil, wave and json readers follow. A path that leaves Python (a C library opening a file itself, a subprocess's
working folder) is not re-rooted: pass it through localize(). A recorded root that IS this checkout maps to itself.
install() is idempotent; mimo_bridge/__init__.py calls it, so every tool of this package has it.
"""
from __future__ import annotations

import builtins
import io
import os
import sys
from pathlib import Path, PurePath

REPO = Path(__file__).resolve().parents[3]              # Source/Python/mimo_bridge/repo_paths.py -> the checkout
# {recorded root: where it is in this checkout, relative to the checkout}
RECORDED_ROOTS = {
    "D:/UEMA_Research": "",                             # the folder the release was built in (its records)
    "C:/Users/bhart/Documents/UEMA_devA_private": "",   # the development repository (the saved state, its command)
    # the development machine's data folders, which the release mirrors under external/D (the saved worlds name
    # D:/UEMA_scratch/... as the folder their scene files are written to)
    "D:/UEMA_scratch": "external/D/UEMA_scratch",
    "D:/UEMA_preserved": "external/D/UEMA_preserved",
    "D:/UEMA_recordings": "external/D/UEMA_recordings",
    "D:/UEMA_movement_viewer": "external/D/UEMA_movement_viewer",
    "D:/UEMA_observer_history": "external/D/UEMA_observer_history",
}
_installed = False


def _norm(text: str) -> str:
    t = text.replace("\\", "/")
    for long_prefix in ("//?/", "//./"):
        if t.startswith(long_prefix):
            t = t[len(long_prefix):]
    return t


def _same(a, b) -> bool:
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


_TARGET = {r.lower(): REPO.joinpath(*[p for p in t.split("/") if p]) for r, t in RECORDED_ROOTS.items()}
_ACTIVE = tuple(sorted((r for r, t in _TARGET.items() if not _same(r, t)), key=len, reverse=True))
# never re-rooted: the running Python environment (on the development machine it sits inside the development
# repository, .venv_mimo), any virtual environment folder at the top of a recorded root, and the process's own
# temporary and output folders (a test runner of the development machine sets them inside D:/UEMA_scratch)
_PYTHON = tuple({_norm(os.path.abspath(p)).lower().rstrip("/") for p in
                 (sys.prefix, sys.base_prefix, sys.exec_prefix, *(os.environ.get(k) for k in
                  ("TEMP", "TMP", "TMPDIR", "UEMA_EVIDENCE_DIR", "UEMA_TMP_DIR")))
                 if p})


def _root_of(path):
    """The recorded root `path` lies in (lower case), or None."""
    try:
        t = _norm(os.fspath(path) if not isinstance(path, bytes) else path.decode("utf-8", "replace")).lower()
    except TypeError:
        return None
    if any(t == p or t.startswith(p + "/") for p in _PYTHON):
        return None
    for root in _ACTIVE:
        if t == root or t.startswith(root + "/"):
            first = t[len(root):].lstrip("/").split("/", 1)[0]
            if first.startswith(".venv") or first in ("venv", "env"):
                return None
            return root
    return None


def is_recorded(path) -> bool:
    """True when `path` names a place inside one of RECORDED_ROOTS other than this checkout."""
    return _root_of(path) is not None


def localize(path):
    """`path` re-rooted at this checkout when it names a place inside one of RECORDED_ROOTS (any slash, any case, with
    or without the \\\\?\\ prefix) that is not this checkout; else `path` itself."""
    if isinstance(path, int):
        return path
    root = _root_of(path)
    if root is None:
        return path
    raw = path.decode("utf-8", "replace") if isinstance(path, bytes) else os.fspath(path)
    rest = [p for p in _norm(raw)[len(root):].split("/") if p]
    local = _TARGET[root].joinpath(*rest)
    if isinstance(path, bytes):
        return os.fsencode(str(local))
    if isinstance(path, PurePath):
        return type(path)(local) if isinstance(path, Path) else local
    return str(local)


def _wrap_first(fn):
    def wrapper(path, *args, **kwargs):
        return fn(localize(path), *args, **kwargs)
    wrapper.__wrapped__ = fn
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def _wrap_two(fn):
    def wrapper(src, dst, *args, **kwargs):
        return fn(localize(src), localize(dst), *args, **kwargs)
    wrapper.__wrapped__ = fn
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapper


def _wrap_listing(fn):
    def wrapper(path=".", *args, **kwargs):
        return fn(localize(path), *args, **kwargs)
    wrapper.__wrapped__ = fn
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapper


def install() -> bool:
    """Re-root recorded paths in Python's file functions for this process; True when it changed anything."""
    global _installed
    if _installed or not _ACTIVE:
        return False
    opener = _wrap_first(io.open)
    builtins.open = opener
    io.open = opener
    for name in ("open", "stat", "lstat", "mkdir", "remove", "unlink", "rmdir", "utime", "chmod", "access"):
        if hasattr(os, name):
            setattr(os, name, _wrap_first(getattr(os, name)))
    for name in ("listdir", "scandir"):
        setattr(os, name, _wrap_listing(getattr(os, name)))
    for name in ("rename", "replace"):
        setattr(os, name, _wrap_two(getattr(os, name)))
    for name in ("exists", "isfile", "isdir", "islink", "lexists"):
        if hasattr(os.path, name):
            setattr(os.path, name, _wrap_first(getattr(os.path, name)))
    _installed = True
    return True
