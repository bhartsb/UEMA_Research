# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Past Development reports rebuilt from the observer's own records, under today's rules (2026-09-30).

The digester keeps every run's committed detections (each run's analysis/detections.jsonl.gz, cached as
columns in the lineage store) and every run's exposure (analysis/progress.json: body-steps per 100-step
bin, author and caregiver presence). A report "as of" a past moment T is today's tracker run over:
  * the runs that started before T, all of whose steps came before T when T is the start of the next
    run (choose T at a run's start: as_of_run_starts); a run still recording at T cannot be cut at T,
    because neither the detections nor the exposure carry wall time;
  * their lineage cuts from those runs alone (a resume after T had not yet abandoned anything);
  * babble's recording of today (refused when it was recorded after T: that control did not exist yet).
What it cannot rebuild, and leaves out with a note in the report's "rebuilt" field: the novelty miner
and the O12 row built on it (they need the episode store, 42 MB of JSON), the planner's horizon and
choices (the learner lane is pruned as the digester goes), the fast lanes' anchors and recorder health.

It loads no model and builds no world: numpy columns, a few small JSON files and the tracker's tests.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import catalog as CAT
from . import history as HI
from . import store as ST
from . import tracker as TR
from .daemon import (ANALYSIS_VERSION, DETECTOR_VERSION, exposure_for_tracker, lineage_cuts, quality_map,
                     run_start_ms, world_signature, _clean)

MISSING = ["the O12 row and the unexpected discoveries (they need the episode store)",
           "how far ahead the learner predicts and the planner's choices (the learner's lane is not kept)",
           "the fast moments and recorder health"]


def _json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def lineage_run_dirs(saved: str, lineage: str) -> List[str]:
    runs = [p for p in glob.glob(os.path.join(saved, "experiments", lineage + "*"))
            if os.path.isdir(os.path.join(p, "observer", "world"))]
    return sorted(runs, key=run_start_ms)


def recording_ended(run_dir: str) -> float:
    """When a run's recorders last wrote (seconds): the newest world header, status file or chunk of any
    body. Status files are rewritten while recording and never removed, so this holds after the raw
    chunks are cleaned away."""
    newest = 0.0
    for pattern in ("header_*.json", "status_*.json", "chunk_*.npz"):
        for p in glob.glob(os.path.join(run_dir, "observer", "world", "b*", pattern)):
            try:
                newest = max(newest, os.path.getmtime(p))
            except OSError:
                pass
    return newest


def best_moment(saved: str, lineage: str, target: float) -> Optional[float]:
    """The latest moment at or before `target` at which no run of the lineage was recording, so a
    rebuild there is exact: `target` itself when every run started before it had stopped writing, else
    the start of the run that was recording then. None when there is no such moment."""
    runs = lineage_run_dirs(saved, lineage)
    for t in sorted({target} | {run_start_ms(r) / 1000.0 for r in runs if run_start_ms(r) / 1000.0 <= target},
                    reverse=True):
        before = [r for r in runs if run_start_ms(r) / 1000.0 < t]
        if before and all(recording_ended(r) < t for r in before):
            return t
    return None


def as_of_run_starts(saved: str, lineage: str) -> List[dict]:
    """Every moment a rebuild can be exact at: each run's start (its earliest world header), with the
    run's first recorded step: [{time, run, first_step}], oldest first."""
    out = []
    for r in lineage_run_dirs(saved, lineage):
        st = (_json(os.path.join(r, "observer", "analysis", "progress.json"), {}) or {}).get("state") or {}
        out.append(dict(time=run_start_ms(r) / 1000.0, run=os.path.basename(r), first_step=st.get("first_step")))
    return out


def _load_store(folder: str, quality: Dict[str, str]) -> tuple:
    """The lineage store's detection columns and per-run offsets, read only (no episodes)."""
    for _ in range(3):                      # the digester may replace a generation while we read
        meta = _json(os.path.join(folder, "store_meta.json"))
        if not meta:
            break
        if meta.get("version") != ST.STORE_VERSION:
            raise RuntimeError("the store is %s, this code reads %s" % (meta.get("version"), ST.STORE_VERSION))
        strings = ST.Strings(meta["strings"])
        t = ST.DetTable(strings, quality)
        try:
            t.load(os.path.join(folder, "store_dets_%d.npz" % int(meta["generation"])))
        except OSError:
            time.sleep(1.0)
            continue
        return t, meta.get("offsets") or {}
    raise RuntimeError("the lineage store could not be read from %s" % folder)


def rebuild(saved: str, lineage: str, until: float, now_ok: bool = False) -> dict:
    """The lineage report as of `until` (seconds), every stage by today's tracker (module doc). With
    now_ok, the run still recording is included up to its committed records (the check that a rebuild
    of now matches the live report)."""
    cat = CAT.load()
    quality = quality_map(cat)
    runs = [r for r in lineage_run_dirs(saved, lineage) if run_start_ms(r) / 1000.0 < until]
    if not runs:
        raise ValueError("no run of %s started before %s" % (lineage, time.ctime(until)))
    progress = {}
    for r in runs:
        p = _json(os.path.join(r, "observer", "analysis", "progress.json"))
        if p is None:
            raise RuntimeError("no committed digest in %s" % r)
        st = p.get("state") or {}
        if (st.get("version"), st.get("detector_version")) != (ANALYSIS_VERSION, DETECTOR_VERSION):
            raise RuntimeError("%s was digested by %s/%s, not today's rules" % (r, st.get("version"),
                                                                               st.get("detector_version")))
        progress[os.path.basename(r)] = p
    still = [os.path.basename(r) for r in runs if recording_ended(r) >= until]
    if still and not now_ok:
        raise ValueError("%s still recorded after %s: rebuild at a moment no run was recording (best_moment)"
                         % (", ".join(still), time.ctime(until)))
    # the store's columns, then each included run's records it has not read yet (up to the committed size)
    t, offsets = _load_store(os.path.join(saved, "observer", lineage, "store"), quality)
    for name, p in progress.items():
        st, sizes = p["state"], p.get("sizes") or {}
        off = offsets.get(name) or {}
        if off.get("analysis_id") not in (None, st.get("analysis_id")):
            raise RuntimeError("the store read another digest of %s" % name)
        path = ST.record_file(os.path.join(saved, "experiments", name, "observer", "analysis"), "detections")
        start, end = int(off.get(os.path.basename(path), 0)), int(sizes.get(os.path.basename(path), 0))
        if end > start:
            rows, _ = ST.read_records(path, start, end)
            for d in rows:
                d.setdefault("run", name)
                t.add(d)
    t.freeze()
    keep_runs = np.asarray([t.s.index.get(n, -9) for n in progress], dtype=np.int64)
    rows = np.nonzero(np.isin(t["run"], keep_runs))[0]
    sub = ST.DetTable(t.s, quality)
    for k, _ in ST.FIELDS:
        sub.cols[k] = t.cols[k][rows]
    states = {n: p["state"] for n, p in progress.items()}
    cuts_by_run = {n: st.get("cuts", {}) for n, st in states.items()}
    line = lineage_cuts(states)
    live = np.ones(len(sub), dtype=bool)
    for name, cuts in cuts_by_run.items():
        rid = sub.s.index.get(name)
        for body, per in (cuts or {}).items():
            for pid, cut in per.items():
                live[(sub["run"] == rid) & (sub["body"] == int(body)) & (sub["pid"] == int(pid))
                     & (sub["step0"] >= int(cut))] = False
    for name, cut in line.items():
        live[(sub["run"] == sub.s.index.get(name)) & (sub["step0"] >= int(cut))] = False
    exposure = exposure_for_tracker([(progress[n]["exposure"], cuts_by_run[n], line.get(n)) for n in progress])
    header = None
    for r in runs:
        hs = sorted(glob.glob(os.path.join(r, "observer", "world", "b*", "header_*.json")), key=os.path.getmtime)
        if hs:
            header = _json(hs[-1], header)
    signature = world_signature(header) if header else None
    base_full = _json(os.path.join(saved, "observer", "baselines", str(signature), "random_orders.json"), {}) or {}
    if base_full.get("detector_version") not in (None, DETECTOR_VERSION):
        base_full = {}
    if base_full and float(base_full.get("generated") or 0) > until:
        raise ValueError("babble was recorded after %s: that comparison did not exist yet" % time.ctime(until))
    baselines = base_full.get("caps", {})
    tracked = TR.track_all(sub, exposure, baselines, cat, live=live)
    if "O12" in tracked:
        tracked["O12"]["stages"] = {s: dict(reached=None, first_step=None, note="not rebuilt: needs the episode store")
                                    for s in TR.STAGES}
        tracked["O12"].pop("subs", None)
        tracked["O12"].pop("no_overseer", None)
    step_now = int(sub["step0"].max()) if len(sub) else None          # as the daemon's report counts it
    report = dict(
        version=ANALYSIS_VERSION, detector_version=DETECTOR_VERSION, rules_version=TR.RULES_VERSION,
        generated=float(until), last_step=step_now, runs=list(progress), world_signature=signature,
        current_run=list(progress)[-1],
        baseline=dict(available=bool(baselines), signature=signature, generated=base_full.get("generated"),
                      kind="random_orders" if baselines else None, caps=len(baselines), steps=base_full.get("steps"),
                      order_reach_cm=base_full.get("order_reach_cm"), order_strength=base_full.get("order_strength"),
                      vision=bool(base_full.get("vision", False)) if baselines else None,
                      start=base_full.get("start"), checkpoint_step=base_full.get("checkpoint_step")),
        counts=dict(detections=int(len(sub)), live_detections=int(live.sum()),
                    bodies=sorted(int(b) for b in np.unique(sub["body"])) if len(sub) else []),
        recording=dict(first_step=min((int(s["first_step"]) for s in states.values() if s.get("first_step") is not None),
                                      default=None), lineage_cuts=line),
        exposure={b: dict(steps=v["steps"], absent_steps=v["absent_steps"], by_author=v["by_author"])
                  for b, v in exposure.items()},
        capabilities=tracked, timeline=TR.timeline(tracked), horizon={}, novelty=[], replays={},
        catalog_rows=len(cat["capabilities"]),
        rebuilt=dict(at=time.time(), as_of=float(until), missing=MISSING,
                     how="today's tracker over the observer's committed records of the runs started before this "
                         "moment (observer/rebuild.py)"))
    return _clean(report)


def compare_reports(a: dict, b: dict, skip=("O12",)) -> dict:
    """Every stage verdict, p-value and count of two reports side by side: the rows, sub-rows and
    caregiver-away columns that differ (the check that a rebuild of now matches the live report)."""
    diffs, n = [], 0

    def cells(rep):
        for cid, row in (rep.get("capabilities") or {}).items():
            if cid in skip:
                continue
            yield (cid, "", "row"), row.get("stages") or {}, row.get("counted")
            yield (cid, "", "away"), row.get("no_overseer") or {}, None
            for sk, sr in (row.get("subs") or {}).items():
                yield (cid, sk, "row"), sr.get("stages") or {}, sr.get("counted")
    left = {k: (s, c) for k, s, c in cells(a)}
    for k, s, c in cells(b):
        n += 1
        if k not in left:
            diffs.append(dict(cell=k, why="missing on one side"))
            continue
        s0, c0 = left[k]
        if c0 != c:
            diffs.append(dict(cell=k, why="counted %s against %s" % (c0, c)))
        for stage in TR.STAGES:
            x, y = s0.get(stage) or {}, s.get(stage) or {}
            for f in ("reached", "occasions", "predicted", "judged", "decisive"):
                if x.get(f) != y.get(f):
                    diffs.append(dict(cell=k, stage=stage, field=f, a=x.get(f), b=y.get(f)))
            px, py = x.get("p_value"), y.get("p_value")
            if (px is None) != (py is None) or (px is not None and not np.isclose(px, py, rtol=1e-6, atol=1e-300)):
                diffs.append(dict(cell=k, stage=stage, field="p_value", a=px, b=py))
    return dict(cells=n, differences=diffs)


def write_rebuilt(saved: str, lineage: str, until: float, root: Optional[str] = None) -> str:
    rep = rebuild(saved, lineage, until)
    folder = os.path.join(root or HI.history_root(), lineage)
    data = json.dumps(rep).encode("utf-8")
    return HI.write_snapshot(data, folder, when=until, rebuilt=True, rep=rep)
