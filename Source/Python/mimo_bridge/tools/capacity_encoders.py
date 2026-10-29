# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Can the other five sense encoders get bigger, and what does it cost together?

    python -m mimo_bridge.tools.capacity_encoders [--write <path.json>]

WHY. tools/capacity_benchmark.py and tools/capacity_combined_fit.py sized the core
and the vision encoder: core 8 blocks x 256 plus vision 12 swin blocks x 256 peaks
at 5,907 MiB on the 8,188 MiB card. The five other encoders - tactile, audio,
proprioception, motor, intermind - sit at one attention layer (or a 128-wide MLP /
conv stack) each, and nothing measured what growing them costs.

WHAT IT MEASURES, each encoder alone on the run's real input shapes, bf16, median
of repeated trials after warmup, torch.cuda.synchronize around every timed region:
  forward+backward at batch 8 (the live pass) at several depths and widths
Then EVERYTHING TOGETHER on the card at once - core 8x256, vision 12x256, the five
encoders at the candidate size, a second core standing in for the planner (the
real planner freezes the live core in place and holds no copy, so this over-counts
by 42.3 MB at 8x256 - conservative, not wrong), and AdamW
moments for all of it - timing the replay pass at batch 48, which is where memory
runs out first (measured: 431 ms fits, 3,342 ms is the card spilling).

Every encoder's input width is read off the BUILT module, never assumed, so a
config change that alters an input contract shows up here as a shape error rather
than a silently wrong measurement.

NOT MEASURED HERE: whether a bigger encoder helps. This answers cost only.
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

STEP_MS = 1000.0 / 0.6494392952843702          # H83 measured model step
BATCH = 8
CTX, NMOD = 128, 6

# (label, layers, width). "layers" means attention layers for the attention
# encoders, hidden layers for the proprioception MLP, conv layers for audio.
SIZES = [("today", 1, 128), ("2 layers", 2, 128), ("4 layers", 4, 128),
         ("width 256", 1, 256), ("4 x 256", 4, 256)]


def timed(fn, n=20, warmup=6):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        xs.append(s.elapsed_time(e))
    return statistics.median(xs)


def encoder_overrides(layers, width):
    """Config overrides that size the five non-vision encoders together."""
    return {
        "tactile": {"encoder": {"n_cross_attn_layers": layers, "d_internal": width,
                                "per_patch_hidden": width}},
        "motor": {"encoder": {"n_cross_attn_layers": layers, "d_internal": width,
                              "per_patch_hidden": width}},
        "intermind": {"encoder": {"n_self_attn_layers": layers, "d_internal": width}},
        "audio": {"encoder": {"d_internal": width,
                              "conv_channels": [width] * (2 + layers)}},
        "proprioception": {"encoder": {"d_internal": width,
                                       "hidden_dims": [width] * (1 + layers)}},
    }


def inputs_for(name, enc, cfg, batch, dev):
    """Real-shaped input for one encoder, widths read off the built module."""
    import torch
    t = type(enc).__name__
    if t == "CrossPatchAttentionEncoder":
        first = next(m for m in enc.modules() if isinstance(m, torch.nn.Linear))
        n = int(cfg["modalities"][name]["encoder"].get("_n_patches_for_pos") or 32)
        if name == "motor":
            n = 26                     # bridge body: 26 handles (trunk_v1)
        return (torch.randn(batch, n, first.in_features, device=dev),)
    if t == "MLPProjectEncoder":
        return (torch.randn(batch, enc.mlp[0].in_features, device=dev),)
    if t == "SelfAttentionPoolEncoder":
        n = int(cfg["modalities"][name].get("pipeline", {}).get("n_agents", 5))
        return (torch.randn(batch, n, enc.embed.in_features, device=dev),)
    if t == "Conv1DStackEncoder":
        conv = next(m for m in enc.modules() if isinstance(m, torch.nn.Conv1d))
        spatial = int(cfg["modalities"][name]["encoder"].get("spatial_branch_input_dim", 2))
        return (torch.randn(batch, 256, conv.in_channels, device=dev),
                torch.randn(batch, spatial, device=dev))
    raise TypeError(f"no input rule for {t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", default="")
    args = ap.parse_args()

    import torch
    from mimo_bridge.lem_model import build_config
    from lem.model.encoder_registry import EncoderRegistry
    from lem.model.forward_model import ForwardModel

    dev = torch.device("cuda:0")
    names = ["tactile", "audio", "proprioception", "motor", "intermind"]
    out = {"step_ms": STEP_MS, "batch": BATCH, "alone": {}, "together": []}

    print(f"device {torch.cuda.get_device_name(0)}; model step {STEP_MS:,.0f} ms (H83)")
    print()
    print("EACH ENCODER ALONE - forward+backward at batch 8 (the live pass)")
    base_ms = {}
    for label, layers, width in SIZES:
        cfg = build_config({"model": {"context_length": CTX},
                            "modalities": encoder_overrides(layers, width)})
        reg = EncoderRegistry(cfg).to(dev)
        row = {}
        for name in names:
            enc = reg.encoders[name]
            params = sum(p.numel() for p in enc.parameters())
            try:
                x = inputs_for(name, enc, cfg, BATCH, dev)

                def go(enc=enc, x=x):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = enc(*x)
                    y.float().square().mean().backward()
                ms = timed(go)
                row[name] = dict(params=params, fwd_bwd_ms=ms)
            except Exception as e:
                row[name] = dict(params=params, error=f"{type(e).__name__}: {str(e)[:90]}")
        out["alone"][label] = row
        if label == "today":
            base_ms = {n: row[n].get("fwd_bwd_ms", 0.0) for n in names}
        print(f"  {label}")
        for name in names:
            r = row[name]
            if "error" in r:
                print(f"    {name:15s} {r['params']:>10,d} params   FAILED {r['error']}")
            else:
                added = r["fwd_bwd_ms"] - base_ms.get(name, r["fwd_bwd_ms"])
                print(f"    {name:15s} {r['params']:>10,d} params   {r['fwd_bwd_ms']:6.2f} ms"
                      f"   added {added:+6.2f} ms = {100*added/STEP_MS:+5.2f}% of a step")
        del reg
        torch.cuda.empty_cache()

    print()
    print("EVERYTHING ON THE CARD AT ONCE - core 8x256, vision 12x256, the five encoders at")
    print("the size shown, the planner's frozen core copy, AdamW moments; replay pass at batch 48")
    print(f"  {'encoders':>10} {'enc params':>11} {'total params':>13} {'replay48':>9} {'peak MiB':>9}  verdict")
    for label, layers, width in SIZES:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        mods = encoder_overrides(layers, width)
        mods["vision"] = {"encoder": {"n_swin_layers": 12, "d_internal": 256,
                                      "angular_pos_hidden": 256}}
        cfg = build_config({"model": {"d_model": 256, "context_length": CTX,
                                      "forward_model": {"n_layers": 8}},
                            "modalities": mods})
        try:
            core = ForwardModel(cfg).to(dev)
            planner = ForwardModel(cfg).to(dev).eval()
            for p in planner.parameters():
                p.requires_grad_(False)
            reg = EncoderRegistry(cfg).to(dev)
            params = list(core.parameters()) + list(reg.parameters())
            opt = torch.optim.AdamW(params, lr=3e-4)
            (sum(p.float().sum() * 0 for p in params)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            enc_params = sum(sum(p.numel() for p in reg.encoders[n].parameters()) for n in names)
            total = sum(p.numel() for p in params)

            def replay():
                h = torch.randn(48, CTX * NMOD, 256, device=dev, requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for b in core.blocks:
                        h = b(h)
                h.float().square().mean().backward()
            r_ms = timed(replay, n=10, warmup=3)
            peak = torch.cuda.max_memory_allocated() / 2**20
            # 431 ms is the fitting replay time measured with the smaller encoders;
            # a jump of several times means the card is spilling, not computing
            verdict = ("fits" if peak < 8188 * 0.92 and r_ms < 431 * 1.5
                       else "SPILLING" if r_ms >= 431 * 1.5 else "tight")
            print(f"  {label:>10} {enc_params:>11,d} {total:>13,d} {r_ms:8.0f} ms {peak:>8,.0f}  {verdict}")
            out["together"].append(dict(encoders=label, layers=layers, width=width,
                                        encoder_params=enc_params, total_params=total,
                                        replay48_ms=r_ms, peak_mib=peak, verdict=verdict))
            del core, planner, reg, opt
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:>10}  OUT OF MEMORY")
            out["together"].append(dict(encoders=label, verdict="out of memory"))
        torch.cuda.empty_cache()

    if args.write:
        path = os.path.abspath(args.write)
        existing = {}
        if os.path.exists(path):
            try:
                existing = json.load(open(path, encoding="utf-8"))
            except Exception:
                existing = {}
        existing["capacity_encoders"] = out
        json.dump(existing, open(path, "w", encoding="utf-8"), indent=1)
        print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
