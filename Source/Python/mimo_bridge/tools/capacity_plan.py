# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The proposed model, component by component, and what it does to a step.

    python -m mimo_bridge.tools.capacity_plan [--write <path.json>]

Builds every learned component twice - at today's configuration and at the
proposed one - with the same constructors lem_model.py uses, counts parameters,
times the pieces that change, and turns the timings into a per-step cost with the
pass multipliers the training loop actually runs (live 1 forward+backward; replay
0.25 at batch 48 through the core only, since replay re-studies STORED tokens and
does not re-run the encoders; imagination 0.75 forward through the core; the
planner 20/7 forward+backward through the core and the motor encoder).

The baseline is H83's own logged 0.6494 model steps per second, 1,540 ms.

Proposed: d_model 256; core 8 blocks; vision 12 swin blocks x 256; tactile, audio,
motor and proprioception at 4 layers x 256; intermind unchanged (loss weight 0.0,
synthetic source - nothing learns from it).
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

STEP_MS = 1000.0 / 0.6494392952843702
CTX, NMOD, B = 128, 6, 8
PLANNER_ROUNDS_PER_STEP = 20 / 7.0

PROPOSED = {
    "model": {"d_model": 256, "context_length": CTX, "forward_model": {"n_layers": 8}},
    "modalities": {
        "vision": {"encoder": {"n_swin_layers": 12, "d_internal": 256, "angular_pos_hidden": 256}},
        "tactile": {"encoder": {"n_cross_attn_layers": 4, "d_internal": 256, "per_patch_hidden": 256}},
        "motor": {"encoder": {"n_cross_attn_layers": 4, "d_internal": 256, "per_patch_hidden": 256}},
        "audio": {"encoder": {"d_internal": 256, "conv_channels": [256] * 6}},
        "proprioception": {"encoder": {"d_internal": 256, "hidden_dims": [256] * 5}},
        # intermind: left as LITE_OVERRIDES has it - 1 layer, 128 wide
    },
}
TODAY = {"model": {"context_length": CTX}}


def timed(fn, n=20, warmup=6):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        xs.append(s.elapsed_time(e))
    return statistics.median(xs)


def count(m):
    return sum(p.numel() for p in m.parameters())


def build(overrides, dev):
    import torch
    from mimo_bridge.lem_model import build_config
    from lem.model.encoder_registry import EncoderRegistry
    from lem.model.forward_model import ForwardModel
    from lem.model.prediction_heads import PredictionHeads
    from lem.model.motor_controller import MotorController
    from lem.model.wellness import WellnessEvaluators
    cfg = build_config(overrides)
    parts = {
        "encoders": EncoderRegistry(cfg).to(dev),
        "core": ForwardModel(cfg).to(dev),
        "prediction_heads": PredictionHeads(cfg).to(dev),
        "motor_controller": MotorController(cfg).to(dev),
        "wellness": WellnessEvaluators(cfg).to(dev),
    }
    return cfg, parts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", default="")
    args = ap.parse_args()
    import torch
    dev = torch.device("cuda:0")

    out = {"step_ms": STEP_MS, "today": {}, "proposed": {}, "timing": {}}
    rows = []
    for label, ov in (("today", TODAY), ("proposed", PROPOSED)):
        cfg, parts = build(ov, dev)
        d = cfg["model"]["d_model"]
        nb = cfg["model"]["forward_model"]["n_layers"]
        r = {"d_model": d, "core_blocks": nb}
        for name, enc in parts["encoders"].encoders.items():
            r[f"enc:{name}"] = count(enc)
        for k in ("core", "prediction_heads", "motor_controller", "wellness"):
            r[k] = count(parts[k])
        r["total"] = sum(v for k, v in r.items() if k not in ("d_model", "core_blocks"))
        out[label] = r

        # timings of the pieces that change, at the live batch
        t = {}
        vis = parts["encoders"].encoders["vision"]
        px = torch.randn(B, 1160, 768, device=dev, requires_grad=True)
        ang = torch.randn(B, 1160, 4, device=dev)
        def gv():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = vis(px, ang)
            (y[0] if isinstance(y, (tuple, list)) else y).float().square().mean().backward()
        t["vision_fb_ms"] = timed(gv)
        core = parts["core"]
        def gc(batch):
            def go():
                h = torch.randn(batch, CTX * NMOD, d, device=dev, requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for blk in core.blocks:
                        h = blk(h)
                h.float().square().mean().backward()
            return go
        def gf():
            h = torch.randn(B, CTX * NMOD, d, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                for blk in core.blocks:
                    h = blk(h)
        t["core_fb_ms"] = timed(gc(B))
        t["core_fwd_ms"] = timed(gf)
        t["core_replay48_fb_ms"] = timed(gc(48), n=10, warmup=3)
        # the four grown encoders, live batch
        for name, shape in (("tactile", (B, 32, None)), ("motor", (B, 26, None)),
                            ("proprioception", (B, None, None)), ("audio", None)):
            enc = parts["encoders"].encoders[name]
            if name == "audio":
                conv = next(m for m in enc.modules() if isinstance(m, torch.nn.Conv1d))
                x = (torch.randn(B, 256, conv.in_channels, device=dev), torch.randn(B, 2, device=dev))
            elif name == "proprioception":
                x = (torch.randn(B, enc.mlp[0].in_features, device=dev),)
            else:
                first = next(m for m in enc.modules() if isinstance(m, torch.nn.Linear))
                x = (torch.randn(shape[0], shape[1], first.in_features, device=dev),)
            def ge(enc=enc, x=x):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = enc(*x)
                y.float().square().mean().backward()
            t[f"{name}_fb_ms"] = timed(ge)
        out["timing"][label] = t
        del parts
        torch.cuda.empty_cache()

    T, P = out["timing"]["today"], out["timing"]["proposed"]

    def step_cost(t):
        enc = t["vision_fb_ms"] + t["tactile_fb_ms"] + t["motor_fb_ms"] + t["proprioception_fb_ms"] + t["audio_fb_ms"]
        return (t["core_fb_ms"] * 1.0
                + t["core_replay48_fb_ms"] * 0.25
                + t["core_fwd_ms"] * 0.75
                + t["core_fb_ms"] * PLANNER_ROUNDS_PER_STEP
                + enc * 1.0
                + t["motor_fb_ms"] * PLANNER_ROUNDS_PER_STEP)   # the planner re-encodes the order
    c0, c1 = step_cost(T), step_cost(P)
    added = c1 - c0
    new_step = STEP_MS + added
    out["result"] = dict(model_ms_today=c0, model_ms_proposed=c1, added_ms=added,
                         step_ms_today=STEP_MS, step_ms_proposed=new_step,
                         slower_pct=100 * added / STEP_MS,
                         steps_per_s_today=1000 / STEP_MS, steps_per_s_proposed=1000 / new_step)

    a, b = out["today"], out["proposed"]
    print(f"{'component':24s} {'today':>12s} {'proposed':>12s} {'x':>6s}")
    keys = ["enc:vision", "enc:tactile", "enc:audio", "enc:proprioception", "enc:motor", "enc:intermind",
            "core", "prediction_heads", "motor_controller", "wellness", "total"]
    labels = {"enc:vision": f"vision  swin {2}x128 -> 12x256", "core": f"core  {a['core_blocks']}x{a['d_model']} -> {b['core_blocks']}x{b['d_model']}"}
    for k in keys:
        lab = labels.get(k, k.replace("enc:", "encoder "))
        print(f"{lab:24s} {a[k]:12,d} {b[k]:12,d} {b[k]/max(a[k],1):6.1f}")
    print()
    print("timings, live batch 8 (ms)           today   proposed")
    for k in ("vision_fb_ms", "core_fb_ms", "core_fwd_ms", "core_replay48_fb_ms",
              "tactile_fb_ms", "motor_fb_ms", "proprioception_fb_ms", "audio_fb_ms"):
        print(f"  {k:32s} {T[k]:7.2f} {P[k]:10.2f}")
    r = out["result"]
    print()
    print(f"model time per step   today {r['model_ms_today']:7.1f} ms   proposed {r['model_ms_proposed']:7.1f} ms   added {r['added_ms']:+.1f} ms")
    print(f"whole step            today {r['step_ms_today']:7.0f} ms   proposed {r['step_ms_proposed']:7.0f} ms")
    print(f"                      today {r['steps_per_s_today']:.3f} steps/s   proposed {r['steps_per_s_proposed']:.3f} steps/s")
    print(f"SLOWER BY {r['slower_pct']:.1f}%")

    if args.write:
        path = os.path.abspath(args.write)
        existing = {}
        if os.path.exists(path):
            try:
                existing = json.load(open(path, encoding="utf-8"))
            except Exception:
                existing = {}
        existing["capacity_plan"] = out
        json.dump(existing, open(path, "w", encoding="utf-8"), indent=1)
        print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
