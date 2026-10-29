# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Scaling calculator: what model sizes fit our ALWAYS-LEARNING embodied loop on what
hardware, and what experience throughput they come with.

    python -m mimo_bridge.tools.scaling_calc

Not an offline-LLM calculator: our loop trains online at a control rate (one
forward+backward per action step), so the binding constraints are (1) optimizer-state
memory, (2) step latency at small batch, (3) env experience throughput (CPU physics),
(4) data adequacy for the parameter count.

Hardware constants (published rack numbers; NVIDIA quotes FP8 with sparsity for
GB200 - dense is ~half - and GB300's gain is memory + FP4, its dense FP8 is roughly
GB200-class. Sources:
  https://www.nvidia.com/en-us/data-center/gb200-nvl72/
  https://www.together.ai/gpu/nvidia-gb300-nvl72
  https://introl.com/blog/why-nvidia-gb300-nvl72-blackwell-ultra-matters )
"""
from __future__ import annotations

from dataclasses import dataclass

TOKENS_PER_STEP = 700        # effective tokens through the model per step (512 vision
                             # patches dominate; 6 modality tokens + context are noise)
CONTROL_DT_S = 0.05          # one model step per 50 ms of body time (action-repeat 5 x 10 ms)
BYTES_PER_PARAM_TRAIN = 18   # bytes of GPU memory per PARAMETER for online training:
                             # bf16 weights+grads (4) + fp32 master + Adam m,v (12) + overhead
MEM_USABLE_FRACTION = 0.70   # activations, replay, env buffers eat the rest
CHINCHILLA_RATIO = 20        # training tokens per parameter for data adequacy (heuristic)
ENV_STEPS_PER_S_BODY = 100.0 # headless MuJoCo MIMo with full touch, per body, server core
CORES_PER_BODY = 2.0

@dataclass
class Hardware:
    name: str
    peak_flops: float       # dense tensor FLOP/s at the precision we would train in
    mfu: float              # REAL-WORLD utilization for our loop, not a hope:
                            #  - laptop: MEASURED from our own step times (~5%)
                            #  - single GPU: estimated 15% (small-batch, latency-bound)
                            #  - racks: 30% (CoreWeave reports 42-51% MFU for offline
                            #    large-batch H100/GB200 training; online streaming costs some)
    gpu_mem_bytes: float
    bodies: int             # parallel MuJoCo bodies (measured where we have measurements)
    note: str = ""


HW = [
    # peak ~30 TFLOP/s BF16 tensor dense for 4070 Laptop; our MEASURED model-step time
    # (3.3M params, 16 bodies, ~130 ms model-only) back-solves to ~1.7 TFLOP/s effective
    # -> mfu ~0.055. Bodies = 16, measured (i9-14900HX, 24 cores, sharing with rendering).
    Hardware("RTX 4070 Laptop (this machine, measured)", 30e12, 0.055, 8.2e9, 16,
             "8 GB VRAM, 30 GB RAM, i9-14900HX 24C/32T; numbers from our own runs"),
    # Our ACTUAL cloud box: GCP g4-standard-96, flex-start (currently TERMINATED).
    # 2x RTX PRO 6000 Blackwell Server: BF16 dense ~252 TFLOP/s each, FP8 ~504; 96 GB
    # GDDR7 each (192 GB total); HOSTKEY/Akamai reviews measure 75-80% of peak in
    # practice -> with small-batch online overhead use mfu 0.15. 96 vCPU -> ~40 bodies.
    Hardware("GCP g4-standard-96 (ours, 2x RTX PRO 6000, flex)", 2 * 504e12, 0.15, 2 * 96e9, 40,
             "our reprovisioned instance 'rtx-6000-2gpu-flex'; FP8 dense 1.0 PFLOP/s total"),
    Hardware("GB200 NVL72 (rack)", 360e15, 0.30, 13.5e12, 1296,
             "72 GPUs + 36 Grace (2,592 cores); 13.5 TB HBM3e; 720 PF FP8 sparse = 360 dense"),
    Hardware("GB300 NVL72 (rack)", 400e15, 0.30, 21e12, 1296,
             "Blackwell Ultra: 21 TB HBM3e (40 TB w/ Grace); MLPerf v5.1: ~27% faster than GB200"),
]

# ---- sensory profiles ------------------------------------------------------------
# BRIDGE (MIMo, what every number so far used): 512 low-res eye patches + 256 touch
# patches through tiny encoders -> ~700 effective trunk-equivalent tokens per 50 ms step.
# FULL AVATAR (UE, master_config): panorama 2520x1560 + fovea 512x512 at 16 px patches
# = 16,253 vision patches at 24 Hz against a 60 Hz control rate -> amortized ~6,501
# patch-tokens per step, but patches traverse only the vision ENCODER (~15% of total
# parameters), so effective trunk-equivalent tokens = 0.15*6501 + ~300 trunk = ~1,275
# per 16.7 ms step. Touch: 256 TMC-derived patches (+ stretch/shear channels planned).
# Avatar sim is GPU-bound (Chaos + tet skin): measured 13-34 fps today -> ~0.5x realtime
# per body per strong GPU (CTO framerate report 2026-06-28).
PROFILES = [
    # (name, effective tokens/step, control dt s, bodies_fn(hw), realtime_per_body)
    ("bridge (MIMo, simplified senses)", 700, 0.05, lambda hw: hw.bodies, 1.0),
    ("FULL AVATAR (UE, spec senses)", 1275, 1.0 / 60.0,
     lambda hw: {"RTX 4070 Laptop (this machine, measured)": 0,
                 "GCP g4-standard-96 (ours, 2x RTX PRO 6000, flex)": 2,
                 "GB200 NVL72 (rack)": 72,
                 "GB300 NVL72 (rack)": 72}.get(hw.name, 1), 0.5),
]

MODEL_SIZES = [
    (3.3e6,   "3.3M  (now, d=128)"),
    (10e6,    "10M   (d~192)"),
    (30e6,    "30M   (d~320)"),
    (50e6,    "50M   (d~384)"),
    (100e6,   "100M  (d~512)"),
    (300e6,   "300M  (spec, d=768)"),
    (3e9,     "3B    (GR00T/pi0 class)"),
    (30e9,    "30B"),
    (70e9,    "70B"),
    (200e9,   "200B"),
]


def fmt_params(p):
    return f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M"


def fmt_time(days):
    if days < 1/24:
        return f"{days*24*60:.0f} min"
    if days < 1:
        return f"{days*24:.1f} h"
    if days < 365:
        return f"{days:.1f} d"
    return f"{days/365:.1f} y"


def analyze(hw: Hardware):
    max_params_mem = hw.gpu_mem_bytes * MEM_USABLE_FRACTION / BYTES_PER_PARAM_TRAIN
    eff_flops = hw.mfu * hw.peak_flops
    print(f"\n=== {hw.name}")
    print(f"    {hw.note}")
    print(f"    effective training speed: {eff_flops/1e12:,.0f} TFLOP/s "
          f"({hw.mfu:.0%} of {hw.peak_flops/1e12:,.0f} TFLOP/s peak)")
    print(f"    MEMORY ceiling: {fmt_params(max_params_mem)} parameters trainable online "
          f"({BYTES_PER_PARAM_TRAIN} bytes/parameter optimizer state)")
    for prof_name, tokens, dt, bodies_fn, rt_per_body in PROFILES:
        bodies = bodies_fn(hw)
        if bodies == 0:
            print(f"\n    -- {prof_name}: cannot run on this hardware "
                  f"(avatar sim + training exceed one 8 GB laptop GPU)")
            continue
        env_x = bodies * rt_per_body
        p_lat = eff_flops * dt / (6 * tokens * bodies)
        print(f"\n    -- {prof_name}")
        print(f"       bodies: {bodies:,} at {rt_per_body:.1f}x realtime each -> "
              f"{env_x:.0f} days of body-time per wall-clock day")
        print(f"       LATENCY ceiling: {fmt_params(p_lat)} parameters if every "
              f"{dt*1000:.0f} ms control step trains synchronously at batch {bodies}")
        print(f"       {'model size':26s} {'fits mem':>8s} {'realtime':>9s} "
              f"{'step time':>10s} {'Chinchilla-adequate in':>23s}")
        for p, label in MODEL_SIZES:
            fits = "yes" if p <= max_params_mem else "NO"
            rt = "yes" if p <= p_lat else "NO"
            step_ms = 6 * p * tokens * bodies / eff_flops * 1e3
            steps_s = bodies * rt_per_body / dt
            days = (CHINCHILLA_RATIO * p) / (steps_s * 6) / 86400
            print(f"       {label:26s} {fits:>8s} {rt:>9s} {step_ms:8.0f}ms "
                  f"{fmt_time(days):>23s}")


if __name__ == "__main__":
    print("Scaling the always-learning embodied loop (online training, 50 ms control step)")
    print(f"assumptions: {TOKENS_PER_STEP} tokens/step; per-hardware REAL-WORLD "
          f"utilization (laptop measured from our runs); {BYTES_PER_PARAM_TRAIN} bytes/parameter "
          f"optimizer state; {ENV_STEPS_PER_S_BODY:.0f} env-steps/s/body headless; "
          f"Chinchilla x{CHINCHILLA_RATIO} data heuristic")
    for hw in HW:
        analyze(hw)
