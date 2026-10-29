# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""C1-only trainer OFF proof: reference, repeated reference, candidate, then resume.

Runs real trainers in separate processes. C2 must only compile this file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def same(a, b):
    """Exhaustive value equality, including optimizer keys and list lengths."""
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def statistics(path, excluded):
    """Only exact, predeclared paths may be excluded; no wildcard 'time' filter."""
    def visit(value, path=()):
        if isinstance(value, dict):
            return {k: visit(v, path + (str(k),)) for k, v in value.items()
                    if ".".join(path + (str(k),)) not in excluded}
        if isinstance(value, list):
            return [visit(v, path + (str(i),)) for i, v in enumerate(value)]
        return value
    return [visit(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def newest(run):
    paths = list((run / "checkpoints").glob("ckpt_step_*"))
    if not paths:
        raise ValueError(f"Trainer produced no checkpoint: {run}")
    return max(paths, key=lambda p: int(p.name.rsplit("_", 1)[-1]))


def comparison(a, b, excluded):
    from safetensors.torch import load_file
    ca, cb = newest(a), newest(b)
    weights = same(load_file(str(ca / "model.safetensors")), load_file(str(cb / "model.safetensors")))
    # Local trainer-created optimizer files contain repository types and are read
    # only after these subprocesses complete; never use downloaded checkpoint files.
    oa = torch.load(ca / "optimizer.pt", map_location="cpu", weights_only=False)
    ob = torch.load(cb / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer = same(oa["optimizer"], ob["optimizer"])
    sa = statistics(a / "replay_training_stats.jsonl", excluded)
    sb = statistics(b / "replay_training_stats.jsonl", excluded)
    checks = dict(checkpoint_step=ca.name == cb.name, weights=weights, optimizer=optimizer,
                  statistics=bool(sa) and same(sa, sb))
    return dict(checks=checks, passed=all(checks.values()), stats_records=[len(sa), len(sb)])


RUNNER = ("import sys, torch; sys.path.insert(0, sys.argv[1]); torch.set_num_threads(1); "
          "from mimo_bridge import train_mimo; raise SystemExit(train_mimo.main(sys.argv[2:]))")


def run(args):
    out = Path(args.output).resolve()
    if out.exists() or "saved" in {part.lower() for part in out.parts}:
        raise ValueError("Use a new identity output directory outside Saved")
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if not spec["measurement"] or not all(type(spec[k]) is int and spec[k] > 0 for k in ("steps", "resume_steps")):
        raise ValueError("Declare measured test size and provenance")
    argv = spec["arguments"]
    managed = ("--out", "--resume", "--max-steps", "--device", "--latent-")
    if not isinstance(argv, list) or any(not isinstance(v, str) or v.startswith(managed) for v in argv):
        raise ValueError("Output, resume, step, device and latent flags belong to the proof runner")
    if not isinstance(spec["excluded_stat_paths"], list) or any(
            not isinstance(k, str) or "*" in k for k in spec["excluded_stat_paths"]):
        raise ValueError("Declare exact measured timing/path exclusions")
    roots = {"reference": args.production, "repeat": args.production, "candidate": args.candidate}
    revisions = {}
    for name, source in roots.items():
        source = Path(source).resolve()
        if not (source / "mimo_bridge" / "train_mimo.py").is_file():
            raise ValueError("Supply each Source/Python directory")
        result = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=source,
                                capture_output=True, text=True, check=True)
        if result.stdout.strip():
            raise ValueError(f"Freeze and commit the {name} source before proving identity")
        revisions[name] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
            capture_output=True, text=True, check=True).stdout.strip()
    out.mkdir(parents=True)
    receipt = dict(completed=False, authority="none", revisions=revisions, spec=spec)
    try:
        for name, source in roots.items():
            for phase, steps in (("fresh", spec["steps"]), ("resumed", spec["steps"] + spec["resume_steps"])):
                target = out / (name + "_" + phase)
                command = [sys.executable, "-B", "-c", RUNNER, str(Path(source).resolve()),
                           *argv, "--device", "cpu", "--out", str(target), "--max-steps", str(steps)]
                if name == "candidate":
                    command += ["--latent-body-hierarchy", "off", "--latent-planner", "off"]
                if phase == "resumed":
                    command += ["--resume", str(newest(out / (name + "_fresh")))]
                with (out / (name + "_" + phase + ".log")).open("wb") as stream:
                    process = subprocess.run(command, cwd=source,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(Path(source).resolve())),
                        stdout=stream, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"{name}/{phase} failed with exit {process.returncode}; inspect its log")
        receipt["comparisons"] = {phase: {
            other: comparison(out / ("reference_" + phase), out / (other + "_" + phase), spec["excluded_stat_paths"])
            for other in ("repeat", "candidate")} for phase in ("fresh", "resumed")}
        receipt["completed"] = True
        receipt["passed"] = all(row["passed"] for phase in receipt["comparisons"].values() for row in phase.values())
    except Exception as error:
        receipt["error"] = repr(error)
        receipt["next_test"] = "C1 attributes failure before rerunning; all runs remain preserved"
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return 0 if receipt["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("production", "candidate", "spec", "output"):
        parser.add_argument("--" + flag, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
