# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""How much bigger can this model get on this GPU before a step costs too much?

    python -m mimo_bridge.tools.capacity_benchmark [--budget 0.5] [--write <path.json>]

WHY. The bridge runs 2 three-stage attention blocks at d_model 128 because of one
literal in LITE_OVERRIDES (lem_model.py); the full design says 8 blocks. Vision
runs 2 swin blocks where the design says 4. Nothing records what any of that
costs, so the question "can we go bigger" has only ever been answered by opinion.

WHAT IT MEASURES, on the real shapes the run uses, in the run's own dtype:
  the CORE (ForwardModel) at several (blocks, d_model) pairs
  the VISION encoder at several (swin blocks, d_internal) pairs
each timed as forward-only and forward+backward, median of repeated trials after
warmup, with torch.cuda.synchronize around every timed region.

HOW THE PER-STEP COST IS COMPUTED. A model step is not one core pass. From the
training loop with H83's flags (replay_every 4, replay_batch 48, imagination_steps
3, intention_every 7): the live pass is one forward+backward at the live batch;
replay adds one forward+backward at batch 48 every 4th step; the imagination chain
adds forward-only passes; the planner adds forward+backward rounds every 7th step.
Those multipliers are declared in PASSES below and printed with the result, so the
arithmetic is auditable rather than hidden.

The measured baseline is H83's own logged model_steps_per_s 0.6494, i.e. 1,540 ms
per model step. A configuration passes the budget when the ADDED core+vision time
is under budget x 1540 ms.

NOT MEASURED HERE: whether more capacity helps. This answers cost only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# H83 measured: model_steps_per_s 0.6494 -> 1,540 ms per model step.
STEP_MS = 1000.0 / 0.6494392952843702

# Amortised core passes per model step, from the training loop with H83's flags.
PASSES = {
    "live_fwd_bwd_live_batch": 1.0,      # every step
    "replay_fwd_bwd_batch48": 0.25,      # --replay-every 4
    "imagination_fwd_live_batch": 0.75,  # 3 chained predictions, on replay steps
    "planner_fwd_bwd_live_batch": 0.0,   # filled from the generator's own n_steps
}


def timed(fn, n=30, warmup=10):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return statistics.median(out), min(out), max(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=0.5,
                    help="fraction of the 1,540 ms step the extra capacity may add")
    ap.add_argument("--batch", type=int, default=8, help="bodies per step (H83 n_envs)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--write", default="")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device; this benchmark measures the training GPU")
        return 2
    dev = torch.device("cuda:0")
    print(f"device {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/2**20:,.0f} MiB")
    print(f"baseline model step {STEP_MS:,.0f} ms (H83 model_steps_per_s 0.6494)")
    print(f"budget {args.budget:.0%} -> at most {args.budget*STEP_MS:,.0f} ms may be added")
    print()

    from mimo_bridge.lem_model import build_config
    from lem.model.forward_model import ForwardModel

    # planner rounds, read from the code rather than assumed
    base_cfg = build_config({"model": {"context_length": 128}})
    n_steps = int(base_cfg.get("intention", {}).get("n_steps", 20))
    PASSES["planner_fwd_bwd_live_batch"] = n_steps / 7.0    # --intention-every 7
    print(f"per-step core passes (amortised): {PASSES}")
    print(f"  planner rounds read from config intention.n_steps = {n_steps}, "
          f"every 7th step -> {n_steps/7.0:.2f} fwd+bwd per step")
    print()

    n_mod = len([m for m, v in base_cfg["modalities"].items() if v.get("enabled", True)])
    ctx = 128
    results = {"step_ms": STEP_MS, "budget": args.budget, "passes": dict(PASSES),
               "n_mod": n_mod, "ctx": ctx, "batch": args.batch, "core": [], "vision": []}

    # ------------------------------------------------------------------ core
    print("CORE (ForwardModel): blocks x d_model, ctx 128, "
          f"{n_mod} senses, batch {args.batch}")
    print(f"  {'blocks':>6} {'d_model':>7} {'params':>10} {'fwd ms':>8} {'fwd+bwd':>8} "
          f"{'replay48':>9} {'added/step':>11} {'% of step':>10}  verdict")
    base_cost = None
    import os as _os
    _sweep = _os.environ.get("SWEEP")
    _pairs = ([tuple(int(x) for x in p.split(",")) for p in _sweep.split(";")] if _sweep
              else [(2, 128), (3, 128), (4, 128), (6, 128), (8, 128),
                    (2, 192), (4, 192), (2, 256), (4, 256)])
    for blocks, d in _pairs:
        try:
            cfg = build_config({"model": {"d_model": d, "context_length": ctx,
                                          "forward_model": {"n_layers": blocks}}})
            fm = ForwardModel(cfg).to(dev)
            params = sum(p.numel() for p in fm.parameters())
            blk = fm.blocks[0]

            def run(batch, backward):
                x = torch.randn(batch, ctx * n_mod, d, device=dev,
                                requires_grad=backward)
                def go():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = x
                        for b in fm.blocks:
                            h = b(h)
                    if backward:
                        h.float().square().mean().backward()
                return go

            f_med, _, _ = timed(run(args.batch, False), args.trials)
            fb_med, _, _ = timed(run(args.batch, True), args.trials)
            r_med, _, _ = timed(run(48, True), args.trials)
            vram = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()

            cost = (PASSES["live_fwd_bwd_live_batch"] * fb_med
                    + PASSES["replay_fwd_bwd_batch48"] * r_med
                    + PASSES["imagination_fwd_live_batch"] * f_med
                    + PASSES["planner_fwd_bwd_live_batch"] * fb_med)
            if base_cost is None:
                base_cost = cost
            added = cost - base_cost
            pct = 100.0 * added / STEP_MS
            ok = "OK" if pct <= args.budget * 100 else "OVER"
            print(f"  {blocks:6d} {d:7d} {params:10,d} {f_med:8.2f} {fb_med:8.2f} "
                  f"{r_med:9.2f} {added:11.1f} {pct:9.1f}%  {ok}")
            results["core"].append(dict(blocks=blocks, d_model=d, params=params,
                                        fwd_ms=f_med, fwd_bwd_ms=fb_med,
                                        replay48_ms=r_med, total_ms=cost,
                                        added_ms=added, pct_of_step=pct,
                                        vram_peak_mib=vram, within_budget=pct <= args.budget*100))
            del fm
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {blocks:6d} {d:7d}  FAILED: {type(e).__name__}: {str(e)[:70]}")
            results["core"].append(dict(blocks=blocks, d_model=d, error=str(e)[:200]))
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- vision
    print()
    print("VISION encoder: swin blocks x d_internal, the run's own patch count")
    print(f"  {'swin':>5} {'d_int':>6} {'params':>10} {'fwd ms':>8} {'fwd+bwd':>8} "
          f"{'added/step':>11} {'% of step':>10}  verdict")
    from lem.model.encoder_registry import EncoderRegistry
    vbase = None
    _vs = _os.environ.get("VSWEEP")
    _vp = ([tuple(int(x) for x in p.split(",")) for p in _vs.split(";")] if _vs
           else [(2, 128), (3, 128), (4, 128), (6, 128), (2, 192), (4, 192), (2, 256)])
    for nsw, d in _vp:
        try:
            cfg = build_config({"model": {"d_model": d, "context_length": ctx},
                                "modalities": {"vision": {"encoder": {
                                    "n_swin_layers": nsw, "d_internal": d,
                                    "angular_pos_hidden": d}}}})
            reg = EncoderRegistry(cfg).to(dev)
            enc = reg.encoders["vision"]
            params = sum(p.numel() for p in enc.parameters())
            npatch = 1160
            def run(backward):
                px = torch.randn(args.batch, npatch, 16 * 16 * 3, device=dev,
                                 requires_grad=backward)
                ang = torch.randn(args.batch, npatch, 4, device=dev)
                def go():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = enc(px, ang) if _takes_two(enc) else enc(px)
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if backward:
                        t.float().square().mean().backward()
                return go
            f_med, _, _ = timed(run(False), max(10, args.trials // 2))
            fb_med, _, _ = timed(run(True), max(10, args.trials // 2))
            vram = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
            cost = fb_med + 0.25 * fb_med          # live + replay share
            if vbase is None:
                vbase = cost
            added = cost - vbase
            pct = 100.0 * added / STEP_MS
            print(f"  {nsw:5d} {d:6d} {params:10,d} {f_med:8.2f} {fb_med:8.2f} "
                  f"{added:11.1f} {pct:9.1f}%  {'OK' if pct <= args.budget*100 else 'OVER'}")
            results["vision"].append(dict(swin_layers=nsw, d_internal=d, params=params,
                                          fwd_ms=f_med, fwd_bwd_ms=fb_med, added_ms=added,
                                          pct_of_step=pct, vram_peak_mib=vram))
            del reg
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {nsw:5d} {d:6d}  FAILED: {type(e).__name__}: {str(e)[:70]}")
            results["vision"].append(dict(swin_layers=nsw, d_internal=d, error=str(e)[:200]))
            torch.cuda.empty_cache()

    if args.write:
        path = os.path.abspath(args.write)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            try:
                existing = json.load(open(path, encoding="utf-8"))
            except Exception:
                existing = {}
        existing["capacity_benchmark"] = results
        json.dump(existing, open(path, "w", encoding="utf-8"), indent=1)
        print(f"\nwritten to {path}")
    return 0


def _takes_two(module):
    import inspect
    try:
        return len(inspect.signature(module.forward).parameters) >= 2
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(main())
