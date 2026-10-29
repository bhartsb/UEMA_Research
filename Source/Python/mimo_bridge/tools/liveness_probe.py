# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Which learned parts of a checkpoint respond: the liveness sweep of DESIGN.md section 10 as a tool.

    python -m mimo_bridge.tools.liveness_probe <checkpoint folder> [--bodies 4] [--steps 6]
                                               [--out DIR] [--threads 4]

The checkpoint's model is built on the CPU the way its run built it and loaded without learning
(tools/gof_probe_harness.load_frozen); bodies 0.. of its own episode each live `--steps` moments of
the model's own behaviour (learning off), as the lead's sweep of 2026-09-22 did
(measurements/liveness_sweep.py: 4 bodies x 6 moments). On every moment lem/model/liveness.py's
monitor records, for each Linear followed by GELU/ReLU inside an nn.Sequential, which hidden units
received an input above -3 (a unit below that line passes almost nothing forward or back), and for
each attention module how evenly it spreads (0 = it reads one input, 1 = every input evenly),
recomputed from the module's own inputs; for the motor and touch encoders' pools also how evenly the
pooled token reads its elements (`token_read`). A checkpoint trained with --liveness-repair on is
probed with its revived units and pool paths in place.

Writes <out>/liveness_probe_<run>_<checkpoint>.json (default out: docs/gain_of_function_v3/
measurements, beside the lead's sweep, never over it) with the sweep's keys (dead_layers,
live_layers, attention) and the stats block's (layers, pools, token_read), and prints each layer as
units responding out of every 100. Nothing is trained and no checkpoint or run folder is written.
Refuses to start (exit 3) when the Windows commit charge could not take the probe's measured need
(tools/gof_probe_harness.NEED_GB).
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
import datetime as _dt
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MEASUREMENTS = REPO / "docs" / "gain_of_function_v3" / "measurements"


def _guard(need_gb: float) -> bool:
    from mimo_bridge.tools.commit_guard import commit_reading, heavy_job_extra_gb, verdict
    r = commit_reading()
    if r is None:
        return True
    committed, limit, _free = r
    extra, _holder = heavy_job_extra_gb()     # a running heavy job's unused declared need stays free (2026-09-26)
    ok = verdict(committed, limit, need_gb + extra, 6.0)
    print("commit charge %.1f of %.1f GB; this probe needs %.1f GB (and %.1f GB a running heavy job may still use): %s"
          % (committed, limit, need_gb, extra, "allowed" if ok else "REFUSED"), flush=True)
    return ok


def peak_memory_gb() -> dict:
    """This process's peak commit (private bytes) and peak working set, in GB (None off Windows):
    the measured need to pass to tools/commit_guard next time."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"peak_commit_gb": round(getattr(mi, "peak_pagefile", 0) / 1024 ** 3, 2) or None,
                "peak_working_set_gb": round(getattr(mi, "peak_wset", 0) / 1024 ** 3, 2) or None}
    except Exception:
        return {"peak_commit_gb": None, "peak_working_set_gb": None}


def per100(share: float) -> int:
    return int(round(100.0 * float(share)))


def probe(checkpoint: str, *, bodies: int = 4, steps: int = 6, threads: int = 4) -> dict:
    import torch
    from mimo_bridge.tools import gof_probe_harness as H
    from lem.model import liveness as LV
    t0 = time.time()
    fr = H.load_frozen(checkpoint, threads=threads)
    model = fr.model
    own = getattr(model, "liveness", None)
    if own is not None:
        # the model's own monitor (a --liveness-repair report|on build) samples every moment here
        mon = own
        mon.sample_every = 1
        drive = False
    else:
        mon = LV.LivenessRepair(model, "report", sample_every=1, revive_min_samples=1,
                                layers=(), pools=())
        drive = True
    mon.window_summary(model)          # attaches the hooks and starts an empty window
    n = 0
    k = 0
    for s, _snap in H.start_states(fr, int(bodies)):
        for _ in range(int(steps)):
            if drive:
                mon.begin_step(model, k)
            s.observe()
            if drive:
                mon.end_step(model, k, 1, train=False)
            s.policy_act()
            n += 1
            k += 1
    block = mon.window_summary(model)
    info = fr.info
    out = {"probe": "liveness", "tool": "mimo_bridge.tools.liveness_probe",
           "version": LV.LIVENESS_VERSION, "checkpoint": info["checkpoint"], "run": info["run"],
           "checkpoint_name": info["name"], "lineage_step": info.get("lineage_step"),
           "time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "code": H.code_commit(), "moments": n, "bodies": int(bodies), "steps_per_body": int(steps),
           "learning": "off (train=False, no optimizer)",
           "live_line": LV.LIVE_LINE, "layer_line": LV.LAYER_LINE,
           "model_liveness_mode": None if own is None else own.mode,
           # the lead's sweep's keys (measurements/liveness_sweep.py), so the two read alike
           "dead_layers": {k2: v for k2, v in block["layers"].items() if v < LV.LAYER_LINE},
           "live_layers": {k2: v for k2, v in block["layers"].items() if v >= LV.LAYER_LINE},
           "attention": {k2: {"entropy_share_of_uniform": v} for k2, v in block["pools"].items()
                         if not k2.endswith(".token_read")},
           # the stats block's keys (field contract 2026-09-23)
           "layers": block["layers"], "pools": block["pools"], "token_read": block["token_read"],
           # beside the -3 line: the share of each layer's units whose output VARIES over the moments (a unit held at
           # one active value passes the -3 line on every moment; lem/model/liveness.py, A UNIT THAT VARIES)
           "layers_varying": block.get("layers_varying"), "not_varying": block.get("not_varying"),
           "samples": block["samples"], "revived": block["revived"],
           "revived_units": block["revived_units"], "seconds": round(time.time() - t0, 1),
           **peak_memory_gb()}
    if drive:
        mon.detach()
    return out


def lines(result: dict):
    """The result as sentences a person reads (DESIGN section 11: counts per 100, no bare decimals)."""
    yield ("%s, %d lived moments (%d bodies x %d moments, learning off). A unit responds when its input "
           "rises above -3 on some moment." % (result["checkpoint_name"], result["moments"],
                                                result["bodies"], result["steps_per_body"]))
    varying = result.get("layers_varying") or {}
    for name, v in sorted(result["layers"].items(), key=lambda kv: kv[1]):
        flag = "  UNDER THE LINE of 50" if v < result["layer_line"] else ""
        vary = ("" if name not in varying else
                "; %d of every 100 vary from moment to moment%s" % (
                    per100(varying[name]), "  UNDER THE LINE of 50" if varying[name] < result["layer_line"] else ""))
        yield "  %s: %d of every 100 units respond%s%s" % (name, per100(v), flag, vary)
    for name, v in sorted(result["pools"].items(), key=lambda kv: kv[1]):
        yield ("  %s: spread %d%% of an even spread (0%% = reads one input, 100%% = every input evenly)"
               % (name, per100(v)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--bodies", type=int, default=4, help="bodies of the checkpoint's episode (sweep: 4)")
    ap.add_argument("--steps", type=int, default=6, help="moments each body lives (sweep: 6)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(MEASUREMENTS),
                    help="folder for liveness_probe_<run>_<checkpoint>.json")
    ap.add_argument("--need-gb", type=float, default=None,
                    help="commit charge this probe may add (default tools/gof_probe_harness.NEED_GB); "
                         "each result records its own peak_commit_gb to re-check it")
    a = ap.parse_args(argv)
    from mimo_bridge.tools.gof_probe_harness import NEED_GB
    if not _guard(NEED_GB if a.need_gb is None else float(a.need_gb)):
        return 3
    result = probe(a.checkpoint, bodies=a.bodies, steps=a.steps, threads=a.threads)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("liveness_probe_%s_%s.json" % (result["run"], result["checkpoint_name"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    for ln in lines(result):
        print(ln)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
