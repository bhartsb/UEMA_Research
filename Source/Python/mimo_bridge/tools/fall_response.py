# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Drop the body and measure what the skin and the wellness model actually report.

    python -m mimo_bridge.tools.fall_response [--scene crib] [--heights 0.1,0.3,0.66,0.9]
        [--profile mechanical_v2_local] [--write <path.json>]

WHY. Whether a fall costs the model anything was being argued from a
constant-deceleration estimate - impact speed, an assumed stopping distance, an
assumed contact patch. None of that is what the simulator reports. MuJoCo's
contact forces concentrate on a few skin points, so the pressure a patch actually
sees is not the body weight divided by a plausible area. This drops the body and
records what the skin channel and the comfort model return, step by step.

WHAT IT MEASURES, per drop height:
  peak per-patch pressure in Pa, straight off touch_adapter's field 0
  the pain and comfort the live contact profile returns for that peak
  how many steps the contact lasts, so a brief spike is not read as a sustained load

The resting baseline is measured first in the same posture, because "a fall
hurts" only means something against what lying still already reads.

NOT AN INJURY MODEL. This reports what the simulated skin and the current
wellness formula do. It does not claim the simulated tissue matches an infant's,
and no number here licenses a claim about what the model feels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PRESSURE_COL = 0        # touch_adapter.update: fields[:, 0] = pressure, Pa
# A trial whose starting pose overlaps geometry by more than this is thrown away:
# the solver ejects the overlap with a force no fall produces. The settled pose
# rests at -0.00007 m, so this admits normal resting contact and nothing more.
PENETRATION_LIMIT_M = 1e-3
AREA_COL = 3            # fields[:, 3] = area_mm2


def read_patches(step):
    a = np.asarray(getattr(step, "patches", None), dtype=np.float64)
    if a.ndim < 2 or a.shape[0] == 0:
        return None
    return a


def skin_memories(env):
    """The skin's memories (today's, the recalibrated one, the per-patch copy), to restore before a trial."""
    skin = getattr(env, "_virtual_skin", None)
    if skin is None:
        return None
    pain = getattr(skin, "pain_exposure", None)
    return (skin.exposure.values.copy(), None if pain is None else pain.values.copy(),
            env.pooler.mechanical_exposure.values.copy())


def set_skin_memories(env, memories):
    if memories is None:
        return
    skin = env._virtual_skin
    skin.exposure.values[:] = memories[0]
    if memories[1] is not None and getattr(skin, "pain_exposure", None) is not None:
        skin.pain_exposure.values[:] = memories[1]
    env.pooler.mechanical_exposure.values[:] = memories[2]


def full_pain(step, profile):
    """{rule: body pain} of this frame from the whole pain model (peak, exposure and rubbing memories):
    both rules under a rubbing calibration (the env's report), else the live rule as 'v4'; {} without memories."""
    mc = (getattr(step, "info", None) or {}).get("mechanical_contact") or {}
    report = mc.get("rubbing_calibration")
    if report:
        return dict(report["pain"])
    history = mc.get("history")
    if history is None or profile == "legacy":
        return {}
    from lem.skin_mechanics import PROFILE, WIDE_RANGE_PROFILE, load_parts_np, pain_np
    shape = WIDE_RANGE_PROFILE if profile == "mechanical_v4" else PROFILE
    parts = load_parts_np(np.asarray(step.patches)[:, PRESSURE_COL], np.asarray(history), 1.0, shape)
    return {"v4": float(pain_np(parts.sum(-1), shape.aversion_shape).max())}


def wellness_for(peak_pa, profile, sites=53, hold_s=0.0):
    """Pain and comfort the live formula returns for a peak of this size."""
    import torch
    from lem.model.contact_wellness import contact_response
    f = torch.zeros(1, sites, 8)
    f[..., 0] = peak_pa / 1e5          # the tokenizer's unit: Pa / 100,000
    f[..., 3] = 0.01
    f[..., 4:8] = 0.5
    kw = {}
    if profile != "legacy":
        h = torch.zeros(1, sites, 3)
        h[..., 0] = peak_pa            # decaying peak-pressure trace, Pa
        h[..., 1] = peak_pa * hold_s   # pressure exposure, Pa.s
        kw["history"] = h
    r = contact_response(f, profile=profile, **kw)
    return float(r["pain"].mean()), float(r["comfort"].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--heights", default="0.05,0.10,0.20,0.30,0.45,0.66,0.90")
    ap.add_argument("--profile", default="mechanical_v2_local")
    ap.add_argument("--age", type=float, default=None,
                    help="body age in months; the skin grid grows with it")
    ap.add_argument("--settle", type=int, default=200)
    ap.add_argument("--fall-steps", type=int, default=120)
    ap.add_argument("--trials", type=int, default=7, help="repeats per height; the spread is the noise")
    ap.add_argument("--footprint-guard", default="off",
                    help="BridgeEnv contact_footprint_guard (contact_footprint.FOOTPRINT_GUARDS; mechanical_v3/v4 "
                         "only). The trials are seeded, so a ladder with it on and one with it off drop the "
                         "body through the same physics and differ only in how the skin scores contact")
    ap.add_argument("--rubbing-calibration", default="v4",
                    help="BridgeEnv rubbing_calibration (lem/skin_mechanics.RUBBING_CALIBRATIONS). Any value but "
                         "'v4' also scores every drop with the whole pain model under both rubbing rules")
    ap.add_argument("--write", default="")
    args = ap.parse_args()

    from mimo_bridge.env import BridgeEnv
    from lem.model.body_sets import HANDLE_NAMES

    kw = dict(scene=args.scene, vision=False, seed=3, contact_profile=args.profile)
    if args.footprint_guard != "off":
        kw["contact_footprint_guard"] = args.footprint_guard
    if args.rubbing_calibration != "v4":
        kw["rubbing_calibration"] = args.rubbing_calibration
    if args.age is not None:
        kw["age"] = args.age
    env = BridgeEnv(**kw)
    env.reset()
    nh = len(getattr(env, "handle_names", None) or HANDLE_NAMES)
    limp = np.zeros((nh, 9), np.float32)
    limp[..., 6] = 1.0                      # unit quaternion, zero strength: no muscle

    # --- resting baseline in the same posture ---
    for _ in range(args.settle):
        step = env.step(limp, None, None, render=False)
    rest = []
    for _ in range(60):
        step = env.step(limp, None, None, render=False)
        a = read_patches(step)
        if a is not None:
            rest.append(float(a[:, PRESSURE_COL].max()))
    rest = np.array(rest)
    rest_peak = float(np.median(rest)) if rest.size else 0.0
    rest_pain, rest_comfort = wellness_for(rest_peak, args.profile)

    snap = env.snapshot() if hasattr(env, "snapshot") else None
    qpos0 = np.array(env.data.qpos)
    qvel0 = np.array(env.data.qvel)
    # every trial starts from the rested skin memories, so a drop's whole-model pain is its own and not
    # the previous drop's fading rubbing (the peak-only numbers never read a memory)
    memories0 = skin_memories(env)

    print(f"scene {args.scene}   contact profile {args.profile}")
    print(f"  resting peak patch pressure (median of 60 steps): {rest_peak:,.0f} Pa")
    print(f"  resting pain {rest_pain:.4f}   comfort {rest_comfort:.4f}")
    print()
    print("  drop     peak patch    peak across trials         steps   pain    pain range      vs rest")
    print("  height   median        min - max                  in ct   median  min - max       pressure")

    out = {"contract": "fall_response_v1", "scene": args.scene, "profile": args.profile,
           "resting": {"peak_patch_Pa": round(rest_peak, 1), "pain": round(rest_pain, 4),
                       "comfort": round(rest_comfort, 4)},
           "drops": []}
    if args.footprint_guard != "off":
        out["footprint_guard"] = args.footprint_guard
    if args.rubbing_calibration != "v4":
        out["rubbing_calibration"] = args.rubbing_calibration

    # Repeats, because the peak depends on the posture the body happens to land in,
    # not on height alone: a single trial at 0.20 m read 1225x resting while one at
    # 0.66 m read 25x. Each trial starts from the settled pose with a small random
    # tilt, so the spread across trials IS the measurement's own noise.
    rng = np.random.default_rng(7)
    rejected = []
    import mujoco
    for h in [float(x) for x in args.heights.split(",") if x.strip()]:
        peaks, contacts, full, firsts = [], [], [], []
        for t in range(args.trials):
            env.data.qpos[:] = qpos0
            env.data.qvel[:] = qvel0
            env.data.qpos[2] += h             # raise the root by the drop height
            if t:                             # trial 0 is the unperturbed drop
                env.data.qpos[3:7] += rng.normal(0, 0.02, 4)
                q = env.data.qpos[3:7]
                env.data.qpos[3:7] = q / max(np.linalg.norm(q), 1e-9)
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)

            # REJECT a trial that starts INSIDE something. Raising the root by h
            # can place the body within another geom - measured, a 0.20 m raise in
            # the crib buries the head 34.5 mm inside the hanging toy - and the
            # solver then ejects it with a force no fall produces (12,703 N on a
            # 10.894 kg body, 119 g). That is the measurement's artifact, not the
            # simulator's physics, and it must not be reported as a fall.
            deepest = min((c.dist for c in env.data.contact[:env.data.ncon]), default=0.0)
            if deepest < -PENETRATION_LIMIT_M:
                rejected.append(dict(height_m=h, trial=t, penetration_m=round(deepest, 5)))
                continue

            set_skin_memories(env, memories0)
            peak, in_contact, full_max, full_first = 0.0, 0, {}, None
            for _ in range(args.fall_steps):
                step = env.step(limp, None, None, render=False)
                now = full_pain(step, args.profile)
                if full_first is None:
                    full_first = dict(now)          # the drop's first frame: the body before it lands
                for rule, v in now.items():
                    full_max[rule] = max(full_max.get(rule, 0.0), float(v))
                a = read_patches(step)
                if a is None:
                    continue
                p = float(a[:, PRESSURE_COL].max())
                if p > rest_peak * 1.5:
                    in_contact += 1
                peak = max(peak, p)
            peaks.append(peak)
            contacts.append(in_contact)
            full.append(full_max)
            firsts.append(full_first or {})

        if not peaks:
            print(f"  {h:5.2f} m   every trial started inside geometry - not measurable in this scene")
            out["drops"].append(dict(height_m=h, trials=args.trials, measured=0,
                                     note="all trials rejected for starting overlap"))
            continue
        peaks = np.array(peaks)
        med = float(np.median(peaks))
        pain, comfort = wellness_for(med, args.profile)
        lo_pain, _ = wellness_for(float(peaks.min()), args.profile)
        hi_pain, _ = wellness_for(float(peaks.max()), args.profile)
        ratio = med / rest_peak if rest_peak > 0 else float("nan")
        print(f"  {h:5.2f} m  {med:11,.0f} Pa  {float(peaks.min()):11,.0f} - {float(peaks.max()):<11,.0f}"
              f"  {int(np.median(contacts)):5d}   {pain:6.4f}  {lo_pain:6.4f}-{hi_pain:<6.4f} {ratio:8.1f}x")
        out["drops"].append(dict(height_m=h, trials=args.trials, measured=len(peaks),
                                 peak_patch_Pa_median=round(med, 1),
                                 peak_patch_Pa_min=round(float(peaks.min()), 1),
                                 peak_patch_Pa_max=round(float(peaks.max()), 1),
                                 # every trial, in trial order, so two ladders can be paired trial by trial
                                 peak_patch_Pa_trials=[round(float(p), 1) for p in peaks],
                                 # the largest body pain of each drop from the WHOLE pain model (peak,
                                 # exposure and rubbing memories), per rubbing rule
                                 full_pain_trials={rule: [round(f.get(rule, float("nan")), 4) for f in full]
                                                   for rule in sorted({k for f in full for k in f})},
                                 # and its pain at the drop's first frame, before it lands: the rested background
                                 full_pain_first_trials={rule: [round(f.get(rule, float("nan")), 4) for f in firsts]
                                                         for rule in sorted({k for f in firsts for k in f})},
                                 steps_above_resting_median=int(np.median(contacts)),
                                 pain_at_median=round(pain, 4),
                                 pain_min=round(lo_pain, 4), pain_max=round(hi_pain, 4),
                                 comfort_at_median=round(comfort, 4),
                                 peak_over_resting=round(ratio, 2)))

    counts = getattr(env, "_footprint_guard_counts", None)
    if args.footprint_guard != "off" and counts is not None:
        # what the guard changed over the whole ladder (rest, settle and every drop)
        out["footprint_guard_counts"] = dict(counts)
    env.close()

    out["rejected_for_starting_overlap"] = rejected
    if rejected:
        print()
        print(f"  {len(rejected)} trials thrown away for starting inside geometry:")
        for r in rejected[:12]:
            print(f"    {r['height_m']:.2f} m trial {r['trial']}: {r['penetration_m']*1000:.1f} mm of overlap")
    hurt = [d for d in out["drops"] if d.get("pain_at_median", 0) > 0.01]
    print()
    if hurt:
        first = min(hurt, key=lambda d: d["height_m"])
        print(f"  the lowest drop that costs anything is {first['height_m']:.2f} m"
              f" ({first['peak_patch_Pa_median']:,.0f} Pa median, pain {first['pain_at_median']:.4f})")
        out["lowest_costly_drop_m"] = first["height_m"]
    else:
        print("  NO drop in this range produced pain above 0.01 - falling is free here")
        out["lowest_costly_drop_m"] = None

    if args.write:
        path = os.path.abspath(args.write)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            try:
                existing = json.load(open(path, encoding="utf-8"))
            except Exception:
                existing = {}
        # keyed by profile: the same drop reads differently under each, so a
        # flat key would let one profile's ladder overwrite another's
        section = existing.get("fall_response")
        if not isinstance(section, dict) or "profile" in section:
            section = {} if not isinstance(section, dict) else {section["profile"]: section}
        # a guarded or recalibrated ladder gets its own key, so it never overwrites the profile's own anchor
        key = "+".join([args.profile] + [s for s in (args.footprint_guard, args.rubbing_calibration)
                                         if s not in ("off", "v4")])
        section[key] = out
        existing["fall_response"] = section
        tmp = path + ".tmp"
        json.dump(existing, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, path)
        print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
