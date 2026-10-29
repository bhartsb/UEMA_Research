# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Run the four gain-of-function probes on one checkpoint (DESIGN.md D15, section 9).

    python -m mimo_bridge.tools.gof_probe_suite --ckpt <checkpoint folder> [--skip-indexed]

gaze (item 7), body (item 8), binding (item 9), posture (item 10), one after the other in this
process, on the CPU, sharing the checkpoint's episode companion (loaded once, about 40 s for
H84's 600 MB file). Each probe writes its own JSON and index line; a probe that fails writes a
record saying so (status "failed: ...") instead of silently leaving a gap, and the others still
run. A suite.json beside the probes lists the four headlines.

Before starting, the Windows commit charge is checked (tools/commit_guard): the suite refuses to
start when its measured need (gof_probe_harness.NEED_GB) would leave less than the guard's
reserve under the commit limit.
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
import gc
import importlib
import json
import time
import traceback
from pathlib import Path

from mimo_bridge.tools import gof_probe_harness as H

PROBES = ("gaze", "body", "binding", "posture")
ITEMS = {"gaze": 7, "body": 8, "binding": 9, "posture": 10}


def guard(need_gb: float = H.NEED_GB, reserve_gb: float = 6.0):
    """(allowed, reading) from tools/commit_guard; allowed off Windows."""
    from mimo_bridge.tools.commit_guard import commit_reading, heavy_job_extra_gb, verdict
    r = commit_reading()
    if r is None:
        return True, "commit charge unreadable on this platform"
    committed, limit, free = r
    extra, _holder = heavy_job_extra_gb()     # a running heavy job's unused declared need stays free (2026-09-26)
    ok = verdict(committed, limit, need_gb + extra, reserve_gb)
    return ok, "%.1f of %.1f GB committed, %.1f GB RAM free; need %.1f GB, reserve %.1f GB, heavy job %.1f GB" % (
        committed, limit, free, need_gb, reserve_gb, extra)


def _peak_gb():
    """This process's peak memory so far (GB): Windows' peak working set, else the peak resident
    size; None when unreadable. This is not the process's private commit charge."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None) or getattr(mi, "rss", None)
        return round(float(peak) / 1024 ** 3, 2) if peak else None
    except Exception:
        return None


def _peak_commit_gb():
    """Windows peak private commit in decimal GB; the guard itself uses 1024-based GiB."""
    try:
        import psutil
        peak = getattr(psutil.Process().memory_info(), 'peak_pagefile', None)
        return round(float(peak) / 1e9, 2) if peak is not None else None
    except Exception:
        return None


def run_suite(ckpt, *, probes=PROBES, root=None, skip_indexed: bool = False, options=None) -> dict:
    info = H.resolve_checkpoint(ckpt)
    t0 = time.time()
    options = options or {}
    out = dict(checkpoint=info["checkpoint"], run=info["run"], lineage_step=info["lineage_step"],
               started=time.strftime("%Y-%m-%d %H:%M:%S"), probes={})
    for name in probes:
        # skip only a probe whose newest record is a result, or whose newest records are failures of
        # THIS code H.SAME_CODE_ATTEMPTS times in a row (a failure written by an older commit or
        # harness version, or a first failure of this code, is run again: H.needs_run)
        if skip_indexed and not H.needs_run(info["checkpoint"], name, root):
            out["probes"][name] = dict(status="skipped: already in the index")
            continue
        t1 = time.time()
        try:
            mod = importlib.import_module(f"mimo_bridge.tools.gof_probe_{name}")
            res = mod.run(info["checkpoint"], root=root, **options.get(name, {}))
            out["probes"][name] = dict(status=res.get("status"), headline=res.get("headline"),
                                       path=res.get("path"), elapsed_s=res.get("elapsed_s"))
        except Exception as error:
            msg = f"failed: {type(error).__name__}: {error}"[:500]
            record = dict(config=dict(harness=H.HARNESS_VERSION), arms={}, controls={}, findings={},
                          headline=H.headline(f"{name}_unmeasured", None, "", None, None,
                                              "unmeasured: the probe stopped with an error (see status)"),
                          status=msg, traceback=traceback.format_exc()[-4000:],
                          elapsed_s=round(time.time() - t1, 1))
            try:
                path = H.write_result(info, name, ITEMS[name], record, root=root)
            except Exception:
                path = None
            out["probes"][name] = dict(status=msg, path=None if path is None else str(path),
                                       elapsed_s=record["elapsed_s"])
        gc.collect()
        out["probes"][name]["peak_memory_gb"] = _peak_gb()
        out["probes"][name]["peak_commit_gb"] = _peak_commit_gb()
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["peak_memory_gb"] = _peak_gb()
    out["peak_commit_gb"] = _peak_commit_gb()
    base = Path(root) if root else H.PROBES_ROOT
    path = base / info["run"] / info["name"] / "suite.json"
    # a run of some probes keeps the other probes' entries from earlier runs of this checkpoint
    try:
        earlier = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        earlier = {}
    merged = dict(earlier.get("probes") or {})
    for k, v in out["probes"].items():
        if str(v.get("status", "")).startswith("skipped"):
            continue
        # the result writer already put this run's entry here (harness.refresh_suite, with its time,
        # harness version and commit); keep those and add the suite's timing and memory
        same = v.get("path") and (merged.get(k) or {}).get("path") == v.get("path")
        merged[k] = dict(merged[k], **v) if same else v
    record = dict(out, probes=merged, runs=(earlier.get("runs") or []) + [dict(
        started=out["started"], probes=list(out["probes"]), elapsed_s=out["elapsed_s"],
        peak_memory_gb=out["peak_memory_gb"])])
    H._atomic_json(path, record)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probes", default=",".join(PROBES), help="comma-separated subset of " + ",".join(PROBES))
    ap.add_argument("--skip-indexed", action="store_true",
                    help="skip probes this checkpoint already has in the index (a failure written by an older "
                         "commit or harness version is run again, and a first failure of this code once more)")
    ap.add_argument("--root", default=None, help="output root (default Saved/mimo_bridge/gof_probes)")
    ap.add_argument("--no-guard", action="store_true", help="skip the commit-charge check (tests only)")
    a = ap.parse_args(argv)
    if not a.no_guard:
        ok, reading = guard()
        print(("ALLOWED: " if ok else "REFUSED: ") + reading, flush=True)
        if not ok:
            return 3
    res = run_suite(a.ckpt, probes=[p for p in a.probes.split(",") if p], root=a.root, skip_indexed=a.skip_indexed)
    print(json.dumps({k: (v.get("headline") or {}).get("lived", v.get("status")) for k, v in res["probes"].items()},
                     indent=1), flush=True)
    return 0 if all(not str(v.get("status", "")).startswith("failed") for v in res["probes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
