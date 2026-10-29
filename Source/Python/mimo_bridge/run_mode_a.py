# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Phase 1 driver: one MIMo env drives the lite LEM model in Mode A.

    python -m mimo_bridge.run_mode_a --minutes 10 --scene reach --out Saved/mimo_bridge/phase1

Writes a JSON Lines stats file (one record per report interval) and a summary JSON.
Exits non-zero on any exception (that is the Phase 1 pass/fail).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mimo_bridge.run_mode_a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = time only)")
    ap.add_argument("--scene", default="reach")
    ap.add_argument("--out", default=os.path.join("Saved", "mimo_bridge", "phase1"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-every", type=int, default=2, help="render eyes every N env steps")
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--wellness-mode", default="baseline")
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stats_path = os.path.join(args.out, "mode_a_stats.jsonl")
    summary_path = os.path.join(args.out, "mode_a_summary.json")
    t_start = time.time()
    ok = False
    err = None
    step = 0
    n_nan = 0
    try:
        src = SingleEnvSource(scene=args.scene, run_dir=os.path.join(args.out, "scenes"), vision=not args.no_vision,
                              render_every=args.render_every, seed=args.seed, age=args.age)
        model = LemLite(device=args.device, no_vision=args.no_vision, wellness_mode=args.wellness_mode, seed=args.seed)
        frames = src.reset()
        model.reset_context(1)
        deadline = t_start + args.minutes * 60.0
        acc = {"loss": 0.0, "n": 0, "pe": {}, "wellness": {}, "t_env": 0.0, "t_model": 0.0}
        first_loss = None
        with open(stats_path, "w") as sf:
            while time.time() < deadline and (args.max_steps <= 0 or step < args.max_steps):
                dev = model.dev_limits(step)
                t0 = time.perf_counter()
                res = model.step(frames, dev, train=True, step_index=step)
                t1 = time.perf_counter()
                if not np.isfinite(res.loss):
                    n_nan += 1
                frames = src.step(res.motor_for_action[0], None if res.gaze_out is None else res.gaze_out[0],
                                  dev.get("handle_mask"))
                t2 = time.perf_counter()
                step += 1
                if first_loss is None:
                    first_loss = res.loss
                acc["loss"] += res.loss; acc["n"] += 1
                acc["t_env"] += (t2 - t1); acc["t_model"] += (t1 - t0)
                for k, v in res.pe_per_modality.items():
                    acc["pe"][k] = acc["pe"].get(k, 0.0) + v
                for k, v in res.wellness.items():
                    acc["wellness"][k] = acc["wellness"].get(k, 0.0) + v
                if step % args.report_every == 0:
                    n = acc["n"]
                    rec = {"type": "mode_a", "step": step, "timestamp": time.time(), "elapsed_s": time.time() - t_start,
                           "avg_loss": acc["loss"] / n, "pe": {k: v / n for k, v in acc["pe"].items()},
                           "wellness": {k: v / n for k, v in acc["wellness"].items()},
                           "model_ms": 1000 * acc["t_model"] / n, "env_ms": 1000 * acc["t_env"] / n,
                           "steps_per_s": n / (acc["t_model"] + acc["t_env"]),
                           "sim_time": frames[0].sim_time, "n_contacts": frames[0].n_contacts,
                           "touch_active_patches": int((frames[0].patches[:, 3] > 0).sum()),
                           "grad_norm": res.grad_norm, "motor_confidence": res.motor_confidence,
                           "sources": res.source_mask, "breakdown": res.breakdown, "nan_steps": n_nan}
                    sf.write(json.dumps(rec) + "\n"); sf.flush()
                    log.info("step %d | loss %.4f | pe %s | %.1f steps/s (model %.1f ms, env %.1f ms) | contacts %d | active patches %d",
                             step, rec["avg_loss"], {k: round(v, 4) for k, v in rec["pe"].items()}, rec["steps_per_s"],
                             rec["model_ms"], rec["env_ms"], rec["n_contacts"], rec["touch_active_patches"])
                    acc = {"loss": 0.0, "n": 0, "pe": {}, "wellness": {}, "t_env": 0.0, "t_model": 0.0}
        ok = True
    except Exception:
        err = traceback.format_exc()
        log.error("Mode A run FAILED:\n%s", err)
    finally:
        try:
            src.close()
        except Exception:
            pass
    elapsed = time.time() - t_start
    summary = {"ok": ok and n_nan == 0, "exception": err, "steps": step, "elapsed_s": elapsed, "nan_steps": n_nan,
               "steps_per_s": step / elapsed if elapsed > 0 else 0, "scene": args.scene, "minutes_requested": args.minutes,
               "vision": not args.no_vision, "render_every": args.render_every}
    json.dump(summary, open(summary_path, "w"), indent=2)
    log.info("summary: %s", summary)
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
