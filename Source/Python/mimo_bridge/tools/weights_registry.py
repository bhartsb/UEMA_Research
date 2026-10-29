# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Snapshot lineage: know what every saved brain IS and where it came from.

    python -m mimo_bridge.tools.weights_registry            # backfill + rebuild index

Two artifacts:
  * lineage.json INSIDE each checkpoint folder (run, step, phase, flags, parent
    weights, code commit) - written live by train_mimo for new snapshots; this tool
    backfills the ones saved before lineage existed (code commit inferred as the
    newest commit before the run started, marked "inferred").
  * Saved/mimo_bridge/weights_registry.md - one table of every snapshot on disk,
    newest first, with its parent chain. The morning-report reader's map.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SAVED = os.path.join(REPO, "Saved", "mimo_bridge")

FLAG_KEYS = ("n_envs", "eval_only", "limp", "no_babble", "random_actions",
             "replay", "scene", "start_phase", "minutes")


def git_commits():
    """[(unix_time, short_hash)] newest first, for inferring which code a run used."""
    try:
        outp = subprocess.check_output(["git", "log", "--format=%ct %h", "-200"],
                                       cwd=REPO, text=True, stderr=subprocess.DEVNULL)
        return [(int(a), b) for a, b in (l.split() for l in outp.splitlines())]
    except Exception:
        return []


def commit_for(t, commits):
    for ct, h in commits:
        if ct <= t:
            return h + " (inferred)"
    return "unknown"


def backfill():
    commits = git_commits()
    made = 0
    for base in ("experiments", "runs"):
        for run_dir in sorted(glob.glob(os.path.join(SAVED, base, "*"))):
            args_p = os.path.join(run_dir, "args.json")
            if not os.path.isdir(run_dir) or not os.path.exists(args_p):
                continue
            args = json.load(open(args_p))
            run_t = os.path.getmtime(args_p)
            meta_list = []
            try:
                meta_list = json.load(open(os.path.join(run_dir, "checkpoints.json")))
            except Exception:
                pass
            by_step = {m["step"]: m for m in meta_list}
            for ck in sorted(glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_step_*"))):
                lp = os.path.join(ck, "lineage.json")
                if os.path.exists(lp):
                    prev = json.load(open(lp))
                    if not prev.get("backfilled"):
                        continue        # written live by the run itself: authoritative
                try:
                    meta = json.load(open(os.path.join(ck, "meta.json")))
                except Exception:
                    meta = {}
                step = meta.get("step") or int(ck.rsplit("_", 1)[-1])
                row = by_step.get(step, {})
                lineage = {
                    "run": os.path.basename(run_dir), "out": run_dir, "step": step,
                    "reason": row.get("reason", "unknown"),
                    "saved_at": row.get("time") or time.strftime("%Y-%m-%d %H:%M:%S",
                                                                 time.localtime(meta.get("timestamp", run_t))),
                    "phase": (meta.get("curriculum_state") or {}).get("phase", "fixed"),
                    # a process keeps the code it started with, so infer from RUN start,
                    # not snapshot time (a long run saves hourly across later commits)
                    "code_commit": commit_for(run_t, commits),
                    "resumed_from": args.get("resume"),
                    "flags": {k: args.get(k) for k in FLAG_KEYS},
                    "backfilled": True,
                }
                json.dump(lineage, open(lp, "w"), indent=2)
                made += 1
    return made


def rebuild_index():
    rows = []
    for base in ("experiments", "runs"):
        for lp in glob.glob(os.path.join(SAVED, base, "*", "checkpoints", "ckpt_step_*", "lineage.json")):
            r = json.load(open(lp))
            r["path"] = os.path.dirname(lp)
            rows.append(r)
    rows.sort(key=lambda r: r.get("saved_at", ""), reverse=True)
    L = ["# Weights registry - what every snapshot is",
         "",
         "Every saved brain on disk, newest first. 'grew from' chains a snapshot back to the",
         "weights it resumed; 'fresh' means it started from random. Full detail sits in each",
         "snapshot's own lineage.json.",
         "",
         "| saved | run | step | phase | why saved | grew from | mode | code |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = r.get("flags") or {}
        mode = "eval-frozen" if f.get("eval_only") else "training"
        if f.get("limp"):
            mode += "+limp"
        if f.get("random_actions"):
            mode += "+random"
        parent = r.get("resumed_from") or "fresh"
        if parent != "fresh":
            parent = os.path.basename(parent.rstrip("\\/"))
        L.append(f"| {r.get('saved_at','?')} | {r['run']} | {r['step']} | {r.get('phase','?')} "
                 f"| {r.get('reason','?')} | {parent} | {mode} | {r.get('code_commit','?')} |")
    out = os.path.join(SAVED, "weights_registry.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return out, len(rows)


def main():
    made = backfill()
    out, n = rebuild_index()
    print(f"backfilled {made} lineage files; index of {n} snapshots -> {out}")


if __name__ == "__main__":
    main()
