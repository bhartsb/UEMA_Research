# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Throughput calculator, anchored to MEASURED run data.

    python -m mimo_bridge.tools.throughput_calc

Our MIMo loop is SEQUENTIAL: each step runs the model forward+backward, THEN the physics
for every body. The step costs the SUM of the two halves, and no number of CPU cores
removes the model half. (The UE avatar case is different - simulation and training sit on
different GPUs and overlap - and is modelled separately below.)

Anchors, from run H10 (3.3M model, 16 bodies, 6.0 h, 407 timing records), using RUN MEANS
rather than any single sample:

    model 327.9 ms + physics 189.1 ms per step   ->   1.50x realtime aggregate measured
    (whole-run check: 40591 steps in 6.00 h at 16 bodies x 50 ms of body time per step)

Two calibrations follow and drive every row:

1. PHYSICS: 37.8 ms per env-step per body with 16 bodies in parallel worker processes on
   24 cores. The dominant term is EYE RENDERING, not touch: per env step per body, MuJoCo
   physics is ~1.7 ms, the 5,332-point touch adds ~3.2 ms, and rendering two eyes adds the
   rest. That matters for scaling: rendering moves to a second GPU, touch does not.

2. MODEL: at 3.3M params, 700 tokens and batch 16 the step is LAUNCH-OVERHEAD bound - a few
   percent of the card's peak, because the matmuls are far too small to fill it. Modelled as
   overhead + FLOPs/(peak x util), with util rising with width. The util curve is a
   ONE-POINT FIT at 3.3M with an assumed exponent; it is the least-validated constant here
   and it flatters larger models, so treat >=100M rows as optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- measured anchors (H10, this laptop) -----------------------------------------
MEAS_MODEL_MS = 327.9
MEAS_ENV_MS = 189.1
MEAS_PARAMS = 3.3e6
MEAS_BODIES = 16
MEAS_TOKENS = 700
ACTION_REPEAT = 5
ENV_DT = 0.01                     # 10 ms of body time per env step
LAPTOP_PEAK = 30e12               # BF16 dense tensor FLOP/s, RTX 4070 Laptop
ENV_MS_PER_BODY_STEP = MEAS_ENV_MS / ACTION_REPEAT      # 48.9 ms, 16 bodies in parallel
CORES_MEAS = 24                   # physical cores the measurement ran on


def util(params: float) -> float:
    """Fraction of peak actually achieved. Calibrated at 3.3M from measurement; rises
    with width because the matmuls get big enough to use the machine."""
    d = (params / 3.3e6) ** 0.5 * 128          # rough width for a given size
    return min(0.35, 0.0225 * (d / 128) ** 1.1)   # 2.25% at d=128 reproduces the anchor


# solve the launch overhead that reproduces the measurement exactly
_flops_meas = 6 * MEAS_PARAMS * MEAS_TOKENS * MEAS_BODIES
_compute_ms_meas = _flops_meas / (LAPTOP_PEAK * util(MEAS_PARAMS)) * 1e3
OVERHEAD_MS = max(0.0, MEAS_MODEL_MS - _compute_ms_meas)


@dataclass
class HW:
    name: str
    peak: float          # BF16/FP8 dense tensor FLOP/s available for training
    gpus: int
    vram_gb: float
    cores: int           # physical CPU cores available for physics
    core_speed: float = 1.0     # relative to our laptop core
    overhead_ms: float = OVERHEAD_MS


FLEET = [
    # cores = PHYSICAL cores usable for physics; core_speed relative to our laptop's
    # MEASURED average (which mixed 8 strong P-cores with 16 weak E-cores AND shared the
    # GPU with eye rendering). render_relief = physics speedup from moving eye rendering
    # to a second GPU so it stops competing with training.
    HW("RTX 4070 laptop (measured)", 30e12, 1, 8, 24, 1.0, OVERHEAD_MS),
    HW("g4-standard-96, 1 GPU (train+render)", 504e12, 1, 96, 48, 1.35, 25.0),
    HW("g4-standard-96, GPU0 trains / GPU1 renders", 504e12, 1, 96, 48, 1.7, 25.0),
    HW("g4-standard-96, both GPUs train (data-parallel)", 1008e12, 2, 192, 48, 1.35, 40.0),
    HW("GB200 NVL72 (learner) + Grace bodies", 360e15, 72, 13500, 1296, 1.5, 15.0),
]

SIZES = [(3.3e6, "3.3M"), (10e6, "10M"), (30e6, "30M"), (50e6, "50M"),
         (100e6, "100M"), (300e6, "300M")]


def step_ms(hw: HW, params: float, bodies: int) -> tuple:
    flops = 6 * params * MEAS_TOKENS * bodies
    model = hw.overhead_ms + flops / (hw.peak * util(params)) * 1e3
    # physics: bodies spread over cores; 2 cores per body, oversubscription slows linearly
    per_body = ENV_MS_PER_BODY_STEP / hw.core_speed
    slots = max(1, int(hw.cores / 1.5))     # measured: 16 bodies on 24 cores
    env = per_body * ACTION_REPEAT * max(1.0, bodies / slots)
    return model, env


def best_bodies(hw: HW, params: float) -> int:
    """Bodies that maximize experience: more bodies = more body-time per step, until
    physics or the model's batch cost dominates."""
    best, best_x = 1, 0.0
    for b in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1296):
        if b > max(4, hw.cores):          # never more bodies than cores
            break
        m, e = step_ms(hw, params, b)
        x = b * ACTION_REPEAT * ENV_DT / ((m + e) / 1e3)
        if x > best_x:
            best, best_x = b, x
    return best


# ---- the FULL UE AVATAR at FULL simulation -----------------------------------------
# Anchors, all from the project's own measured documents (no half measures modelled):
#
#  * "27-28 FPS, full body, all senses" - already observed with the old-style full body,
#    all senses on (Head_VocalTract_TetCook_Code_Review_2026-07-20). The whole tet budget
#    is designed AROUND a 36 ms frame allowance, so 36 ms is the full-simulation frame.
#  * Fingertip subdivision study (CTO_Report_Pinch_Framerate_2026-06-28): 43 / 34 / 18 fps
#    at subdivision 0 / 1 / 2 under a hard pinch (~1,275 contacts), contact cost now
#    LINEAR in contact count. Full-body contact - the infant's normal condition - is
#    5,000-10,000 contacts, which is the risk case against the 36 ms budget.
#  * Vision pipeline (AdaptiveVisionComponent_Documentation, RTX 3080, 512 px cameras):
#    camera capture 80-100 ms full / 30-50 ms tiered, panorama stitch 20-30 ms. On an
#    RTX PRO 6000 (~3x a 3080 for this work) tiered capture is ~10-25 ms - it fits inside
#    the frame only in tiered mode, and it is the biggest single risk to the budget.
#  * UE advances 1/60 s of avatar time per tick in fixed-timestep training mode.
#
# Topology (the intended one): GPU1 runs the simulation, GPU0 trains the model. Different
# GPUs and our action-repeat already tolerates a one-step lag, so they OVERLAP: a step
# costs max(sim, model), not sim + model. The simulation is GPU-bound, so putting N
# avatars on one sim GPU splits its frame rate - experience scales with SIM GPUs.
AVATAR_TOKENS = 1275
AVATAR_TICK_S = 1.0 / 60.0
FULL_SIM_MS_DEV = 36.0            # 27-28 FPS with the OLD-STYLE full body, all senses, on the
                                  # cloud VM; it is the DESIGN TARGET for the new one-piece body
HEAVY_CONTACT_MS_DEV = 110.0      # ESTIMATE: 5-10k contacts scaled linearly off the measured
                                  # 53 ms / 1,275-contact pinch (21 ms fixed + 0.025 ms/contact)
VISION_RISK_MS_DEV = 150.0        # ESTIMATE: heavy contact AND un-amortized panoramic capture
RTX_PRO_SPEEDUP = 1.0             # the 27-28 FPS observation was already ON the cloud VM:
                                  # applying a further speedup would double-count it


def avatar_table(params=280e6):
    print()
    print("=== FULL UE AVATAR (full body, all senses) with the %.0fM model" % (params / 1e6))
    print("    sim on one GPU, model on the other, overlapped -> step = max(sim, model)")
    print("    frame anchors (dev box): %.0f ms designed/observed full sim, %.0f ms heavy "
          "full-body contact, %.0f ms if vision capture cannot be amortized"
          % (FULL_SIM_MS_DEV, HEAVY_CONTACT_MS_DEV, VISION_RISK_MS_DEV))
    print("    RTX PRO 6000 assumed %.1fx the dev box; UE advances %.1f ms of avatar time per tick"
          % (RTX_PRO_SPEEDUP, AVATAR_TICK_S * 1000))
    print()
    cases = [("full sim, as designed", FULL_SIM_MS_DEV),
             ("+ heavy full-body contact", HEAVY_CONTACT_MS_DEV),
             ("+ un-amortized vision", VISION_RISK_MS_DEV)]
    setups = [("1 VM (1 sim GPU + 1 train GPU)", 1, 504e12, 25.0),
              ("4 VMs", 4, 4 * 504e12, 30.0),
              ("16 VMs", 16, 16 * 504e12, 35.0),
              ("64 VMs", 64, 64 * 504e12, 40.0)]
    hdr = ("    %-32s %-26s %6s %8s %9s %8s %7s %14s %11s"
           % ("setup", "case", "sims", "sim ms", "model ms", "bound", "xRT",
              "1 body-day in", "Chinchilla"))
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for name, sims, peak, ovh in setups:
        for cname, dev_ms in cases:
            sim_ms = dev_ms / RTX_PRO_SPEEDUP
            flops = 6 * params * AVATAR_TOKENS * sims
            model_ms = ovh + flops / (peak * util(params)) * 1e3
            step_ms = max(sim_ms, model_ms)
            bound = "sim" if sim_ms >= model_ms else "model"
            xrt = sims * AVATAR_TICK_S / (step_ms / 1e3)
            day_h = 24.0 / xrt
            chin_d = (20 * params) / (sims * (1000.0 / step_ms) * 6) / 86400
            print("    %-32s %-26s %6d %8.1f %9.0f %8s %7.2f %11.1f h %9.0f d"
                  % (name, cname, sims, sim_ms, model_ms, bound, xrt, day_h, chin_d))
        print()
    print("    xRT = avatar-time lived per wall-time across all avatars. One avatar at 1.00x")
    print("    means the baby lives in real time. 'Chinchilla' = wall time to accumulate 20")
    print("    tokens per parameter of lived experience.")


def main():
    print("Throughput anchored to H10 RUN MEANS: model %.1f ms + physics %.1f ms per step, "
          "16 bodies, 3.3M -> 1.50x realtime measured" % (MEAS_MODEL_MS, MEAS_ENV_MS))
    print("MIMo loop is SEQUENTIAL (model then physics): step = sum of both halves")
    print(f"derived: {ENV_MS_PER_BODY_STEP:.1f} ms per env-step per body (~"
          f"{1000/ENV_MS_PER_BODY_STEP:.0f} steps/s/body = {ENV_DT*1000/ENV_MS_PER_BODY_STEP:.2f}x "
          f"realtime per body); model launch overhead {OVERHEAD_MS:.0f} ms/step\n")
    hdr = (f"{'hardware':38s} {'model':>6s} {'bodies':>7s} {'model ms':>9s} {'phys ms':>8s} "
           f"{'steps/s':>8s} {'xRT':>6s} {'body-days/day':>14s} {'8h run':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for hw in FLEET:
        for params, label in SIZES:
            b = best_bodies(hw, params)
            m, e = step_ms(hw, params, b)
            sps = 1000.0 / (m + e)
            xrt = b * ACTION_REPEAT * ENV_DT * sps
            body_days = xrt          # 1x realtime aggregate = 1 body-day per wall-day
            eight_h = xrt * 8 / 24   # body-days produced by an 8-hour run
            print(f"{hw.name:38s} {label:>6s} {b:7d} {m:9.0f} {e:8.0f} {sps:8.2f} "
                  f"{xrt:6.1f} {body_days:14.1f} {eight_h:9.1f}d")
        print()


if __name__ == "__main__":
    main()
    avatar_table(280e6)
