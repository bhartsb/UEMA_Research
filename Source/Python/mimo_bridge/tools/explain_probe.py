# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Checkpoint probe: step a snapshot's model in the world it trained in, and write
explainer snapshots every K steps.

    python -m mimo_bridge.tools.explain_probe --ckpt <run>/checkpoints/ckpt_step_N \
        [--n-steps 200] [--every 10] [--n-envs 2] [--out DIR]

Why this exists instead of a bare `train_mimo --resume --eval-only` (plan v2 gap
register, finding 1): train_mimo builds the model from ITS OWN flags, so a bare
resume constructs a ctx-4 model against a ctx-64 checkpoint, drops the
--next-frame-loss caches (emptying the snapshot's predictions block), misreads
residual-mode weights, and runs the neonate phase against a mature checkpoint. This
tool derives the full train_mimo argument list from the run's own args.json (every
model- and env-shaping flag - the causal_test._build_model precedent), takes
--start-phase from the checkpoint's lineage.json, forces --eval-only, and writes to
a FRESH --out - never the probed run's directory, whose args.json, control.json and
stats a second process would corrupt.

The probe's snapshots carry weights: "frozen" - eval-only skips the optimizer step,
so the attention re-forward is exact. Note: train_mimo still writes one end
checkpoint (~49 MB at lite size) into the probe dir; the whole dir is disposable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# argparse store_true flags in train_mimo.parse_args - everything else takes a value
STORE_TRUE = {"no_vision", "probe_touch", "eval_only", "no_babble", "limp",
              "random_actions", "fresh_optimizer", "fill_ring_first",
              "no_intention_execute"}

# run-specific keys the probe must own, not inherit. checkpoint_storage and
# caregiver_record_dir point into the probed run's LIVE storage: inheriting them
# would overwrite the run's D: args.json (train_mimo writes vars(args) there on
# every launch) and land the probe's end checkpoint in the store the supervisor
# resumes from. self_choice would build and qualify a second frozen actor - an
# operational decoration the snapshot never captures. episode_continuation is
# forced to "fresh" below: the probe's flags legitimately differ from the saved
# physical episode's config (eval-only, n-envs), so an exact episode resume is
# refused by design; fresh eval-only worlds never touch the companion.
SKIP = {"out", "resume", "minutes", "max_steps", "n_envs", "tag",
        "video_every_min", "video_seconds", "checkpoint_every_min", "eval_only",
        "explain_every_min", "explain_steps", "explain_body", "explain_keep_mb",
        "start_phase", "report_every", "reinit", "fresh_optimizer",
        "checkpoint_storage", "caregiver_record_dir",
        "self_choice", "self_choice_receipt", "episode_continuation"}


def derive_argv(run_args: Dict[str, Any], lineage: Optional[Dict[str, Any]],
                ckpt: str, out: str, *, n_steps: int = 200, every: int = 11,
                n_envs: int = 2, minutes: float = 60.0, body: int = 0,
                keep_mb: float = 500.0) -> List[str]:
    """The run's own flags + the probe's overrides -> a train_mimo argument list.

    Pure function; the fixture test holds that an H62-shaped args.json comes back
    with ctx 64, residual/next-frame/top-down-gate intact, the lineage phase, a
    fresh out dir and eval-only - and that the result round-trips through
    train_mimo.parse_args.
    """
    argv: List[str] = []
    for k, v in run_args.items():
        if k in SKIP or v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if k in STORE_TRUE or isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]
    phase = (lineage or {}).get("phase")
    if not phase or phase == "fixed":
        phase = run_args.get("start_phase") or "neonate"
    argv += ["--resume", ckpt, "--out", out, "--eval-only",
             "--episode-continuation", "fresh",
             "--start-phase", str(phase),
             "--minutes", str(minutes), "--max-steps", str(n_steps),
             "--n-envs", str(n_envs), "--video-every-min", "0",
             "--report-every", "50",
             "--explain-steps", str(every), "--explain-body", str(body),
             "--explain-keep-mb", str(keep_mb),
             "--tag", "_explainprobe"]
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint folder (ckpt_step_N)")
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--every", type=int, default=11,
                    help="snapshot every K model steps. 11, not 10: a cadence that "
                         "divides evenly into --motor-hold (5) lands every snapshot on "
                         "an order-latch step, so the held phase - the four steps in "
                         "five during which the body is executing an order the model "
                         "already stopped proposing - is never captured and the "
                         "viewer's order-age field reads 0 forever")
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=60.0, help="wall-clock ceiling")
    ap.add_argument("--body", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="FRESH output dir (default: Saved/mimo_bridge/experiments/"
                         "<run>_explainprobe_<stamp>). Never a live or historical "
                         "run's own directory.")
    a = ap.parse_args(argv)

    ckpt = os.path.abspath(a.ckpt)
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.exists(args_path):
        print(f"no args.json beside {ckpt} ({args_path}) - cannot replicate the "
              f"run's flags; refusing to build a guessed model", file=sys.stderr)
        return 2
    run_args = json.load(open(args_path, encoding="utf-8"))
    lineage = None
    lin_path = os.path.join(ckpt, "lineage.json")
    if os.path.exists(lin_path):
        lineage = json.load(open(lin_path, encoding="utf-8"))

    out = a.out or os.path.join(
        os.path.dirname(run_dir),
        f"{os.path.basename(run_dir)}_explainprobe_{time.strftime('%Y%m%d-%H%M%S')}")
    out = os.path.abspath(out)
    if os.path.abspath(run_dir) == out or os.path.exists(
            os.path.join(out, "replay_training_stats.jsonl")):
        print(f"--out {out} is (or already holds) a run - the probe needs a FRESH "
              f"directory so it cannot corrupt an existing run's files", file=sys.stderr)
        return 2

    argv2 = derive_argv(run_args, lineage, ckpt, out, n_steps=a.n_steps,
                        every=a.every, n_envs=a.n_envs, minutes=a.minutes,
                        body=a.body)
    print("explain_probe -> train_mimo", " ".join(argv2))
    from mimo_bridge import train_mimo as TM
    rc = TM.main(argv2)
    print(f"probe done (rc {rc}); snapshots in {os.path.join(out, 'explain')}")
    print("build the page:  python -m mimo_bridge.tools.build_explainer --run", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
