# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Refuse to start heavy work beside the live trainer when Windows' commit charge is near its limit.

    python -m mimo_bridge.tools.commit_guard [--need-gb 3] [--json]

Exit 0 and print the reading when the committed memory plus the work's own need stays at least
--reserve-gb (default 6) under the commit limit; exit 3 otherwise. Windows fails the NEXT process
that asks for memory, not the one that used it: on 2026-09-21 a second trainer beside H84 took the
commit charge to 77.3 of 78.1 GB and the training supervisor died of it. Commit charge, not RAM in
use, is what fails (GetPerformanceInfo CommitTotal against CommitLimit).

THE LIMIT NEVER COUNTS ON THE PAGE FILE EATING C: (2026-09-25). Windows grows the page file (here
c:\\pagefile.sys, 40,000 MB at boot, up to 60,000 MB) as the commit charge nears the limit, and every
growth raises the limit. The guard judged against the grown limit, so it kept allowing work while the
page file took C:: at 3:54 PM PDT, with the relaunched trainer and a dozen agents' tests and
re-digests running, the page file stood at 58,833 MB and C: had 3.6 GB free (both read by the observer
statistics agent at that time), under the observer
recorders' reserve and near the hourly watcher's 8 GB training stop. The limit judged now is Windows'
current limit, but never more than physical memory plus the page file's size at boot plus the room
the page file's drive has above DISK_RESERVE_GIB; and a drive already under that reserve refuses all
work. commit_reading() returns that limit, so every caller (the ability scheduler, the probe suite,
the liveness probe, machine_stats) judges by it unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

# the observer's recorders never take the last 12 GiB of a disk (observer_record.DEFAULT_RESERVE_BYTES),
# and the H84 hourly watcher stops training under 8 GB free (hourly_watch.DISK_STOP_GB): the page file
# must not be counted on to grow into either
DISK_RESERVE_GIB = 12.0
GIB = 1024.0 ** 3

# A RUNNING HEAVY JOB'S UNUSED NEED STAYS FREE (2026-09-26, the prediction heads repair's re-review, finding m6). The
# guard judged each piece of work against the machine at that work's start only, so agents' light readings, each one
# allowed, took the machine from 52.8 to 59.5 GB committed beside the memory trial's arm, whose own stop line was
# 62.8 GB. A heavy job's runner holds this lock with the job's declared need (run_in_training_pause.py); while it runs,
# work that is not part of it also keeps free the part of that need the job's processes have not committed yet.
HEAVY_LOCK = "C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_observer_20260924/heavy_job.lock"


def _same_process(pid, created):
    """True while `pid` runs and, when its creation time was recorded, is still that process."""
    import psutil
    try:
        now = psutil.Process(int(pid)).create_time()
    except (psutil.Error, ValueError, TypeError):
        return False
    return created is None or abs(now - float(created)) < 1.0


def _tree_private_gb(pid):
    """The private commit of process `pid` and of every process under it, in GiB (0 once it has ended)."""
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
    return total / GIB


def heavy_job_extra_gb(lock_path=HEAVY_LOCK, me=None):
    """(GiB to keep free, the lock's record) while a runner's heavy job runs: its declared need minus what its
    processes have committed so far. (0.0, the record) when this process or one of its parents is that runner or its
    job (a heavy job's own guard calls), when the lock declares no need (runners before this change), or when the job
    has not started (no job process in the lock: the runner still waits for this guard); (0.0, None) with no lock, or
    when the runner and its job have both ended."""
    try:
        import psutil
        with open(lock_path, encoding="utf-8") as f:
            holder = json.load(f)
        me = psutil.Process(int(me or os.getpid()))
        mine = {me.pid} | {p.pid for p in me.parents()}
    except Exception:
        return 0.0, None
    live = [int(p) for p, c in ((holder.get("pid"), holder.get("pid_created")),
                                (holder.get("job_pid"), holder.get("job_pid_created")))
            if p is not None and _same_process(p, c)]
    if not live:
        return 0.0, None
    if mine & set(live) or holder.get("need_gb") is None:
        return 0.0, holder
    # NOTHING RUNS YET (2026-09-27): a runner holds the lock while its heavy job waits for this guard to admit it, and
    # the job's process is named in the lock only once it starts. Reserving the whole need of a job that had not started
    # refused every other job on the machine for as long as it waited: the 37 GB drift trial could never fit beside the
    # machine's 35 GB and held every agent's light work back from 7:31 to 8:51 AM. The room kept free is a running
    # job's room to grow; a job that has not started has none to keep.
    if holder.get("job_pid") is None:
        return 0.0, holder
    root = holder.get("job_pid") if holder.get("job_pid") in live else live[0]
    # the declared need above the job's peak so far (the runner records peak_gb): a job past its trainer's peak, in its
    # light end readings, no longer holds back what it will not use again (2026-09-26, 11:23 PM PDT)
    used = max(_tree_private_gb(root), float(holder.get("peak_gb") or 0.0))
    return max(0.0, float(holder["need_gb"]) - used), holder


def _performance():
    """(committed, current limit, free RAM, physical total) in GiB from Windows, or None off Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t),
                    ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
                    ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t), ("HandleCount", wintypes.DWORD),
                    ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]

    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        return None
    page = float(pi.PageSize)
    return (pi.CommitTotal * page / GIB, pi.CommitLimit * page / GIB, pi.PhysicalAvailable * page / GIB,
            pi.PhysicalTotal * page / GIB)


def page_files():
    """[(drive, size at boot GiB, largest size GiB)] from the registry's PagingFiles, or None when a
    page file is system-managed (no sizes) or the registry cannot be read."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management")
        entries = winreg.QueryValueEx(key, "PagingFiles")[0]
    except (ImportError, OSError):
        return None
    out = []
    for entry in entries or []:
        parts = str(entry).split()
        if len(parts) < 3 or not parts[0][:1].isalpha() or parts[0][1:2] != ":":
            return None
        try:
            initial, largest = float(parts[1]), float(parts[2])
        except ValueError:
            return None
        if initial <= 0 or largest < initial:
            return None
        out.append((parts[0][:2].upper() + "/", initial / 1024.0, largest / 1024.0))
    return out or None


def disk_free_gib(drive):
    try:
        return shutil.disk_usage(drive).free / GIB
    except OSError:
        return 0.0


def effective_limit(current_limit, physical_total, pages, free_of=None):
    """The commit limit the guard judges against: Windows' current limit, never more than physical memory
    plus the page files' sizes at boot plus the growth their drives can hold above DISK_RESERVE_GIB (and
    no more than their largest sizes). Without page-file sizes (system-managed) Windows' limit stands."""
    if not pages:
        return current_limit
    free_of = free_of or disk_free_gib
    base = physical_total + sum(initial for _d, initial, _l in pages)
    room = sum(min(max(0.0, free_of(drive) - DISK_RESERVE_GIB), largest - initial)
               for drive, initial, largest in pages)
    return min(current_limit, base + room)


def commit_reading():
    """(committed_gb, limit_gb, free_ram_gb), in GiB, or None off Windows. limit_gb is effective_limit:
    what the page file may grow to without taking its drive under DISK_RESERVE_GIB."""
    r = _performance()
    if r is None:
        return None
    committed, current, free_ram, physical = r
    return committed, effective_limit(current, physical, page_files()), free_ram


def disk_short(pages=None, free_of=None):
    """[(drive, free GiB)] for every page-file drive already under DISK_RESERVE_GIB."""
    pages = page_files() if pages is None else pages
    free_of = free_of or disk_free_gib
    return [(d, free_of(d)) for d, _i, _l in (pages or []) if free_of(d) < DISK_RESERVE_GIB]


def verdict(committed_gb, limit_gb, need_gb, reserve_gb):
    """True when committed + need leaves at least reserve under the limit."""
    return committed_gb + need_gb <= limit_gb - reserve_gb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--need-gb", type=float, default=3.0,
                    help="memory the work about to start will commit (a CPU model+env probe: about 2-3)")
    ap.add_argument("--reserve-gb", type=float, default=6.0,
                    help="headroom that must remain under the commit limit afterwards")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    raw = _performance()
    if raw is None:
        print("commit charge unreadable on this platform; allowed")
        return 0
    committed, current, free_ram, physical = raw
    pages = page_files()
    limit = effective_limit(current, physical, pages)
    short = disk_short(pages)
    extra, holder = heavy_job_extra_gb()
    ok = verdict(committed, limit, a.need_gb + extra, a.reserve_gb) and not short
    rec = {"committed_gb": round(committed, 2), "limit_gb": round(limit, 2), "windows_limit_gb": round(current, 2),
           "free_ram_gb": round(free_ram, 2), "need_gb": a.need_gb, "reserve_gb": a.reserve_gb,
           "disk_reserve_gb": DISK_RESERVE_GIB, "disk_short": [[d, round(f, 1)] for d, f in short],
           "heavy_job": (holder or {}).get("job"), "heavy_job_unused_gb": round(extra, 2), "allowed": ok}
    if a.json:
        print(json.dumps(rec))
    else:
        why = ("; %s has %.1f GB free, under the %.0f GB the page file's drive keeps"
               % (short[0][0].rstrip("/"), short[0][1], DISK_RESERVE_GIB)) if short else ""
        heavy = ("; the running heavy job %s may still commit %.1f GB of its declared %.1f GB, kept free too"
                 % (holder.get("job"), extra, float(holder["need_gb"]))) if extra > 0 else ""
        print("%s: %.1f of %.1f GB committed (Windows' limit %.1f GB; the guard never counts on the page file "
              "growing into the last %.0f GB of its drive), %.1f GB RAM free; this work needs %.1f GB and %.1f GB "
              "must stay free under the limit%s%s" % ("ALLOWED" if ok else "REFUSED", committed, limit, current,
                                                       DISK_RESERVE_GIB, free_ram, a.need_gb, a.reserve_gb, heavy,
                                                       why))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
