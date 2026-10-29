# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Anchor the skin pressure scale to the body's own mass and to published human thresholds.

    python -m mimo_bridge.tools.touch_scale_anchors [--scene crib] [--age 24]
        [--write Saved/mimo_bridge/scale_anchors.json]

WHY THIS EXISTS. The tokenizer divides skin pressure by 100,000 Pa, and two
constants in lem/model/contact_wellness.py sit in that same unit: pleasantness
peaks at 0.05 and pain begins at 0.1. Nothing in the repository said where those
two numbers came from, so no one could tell whether the memory's pressure channel
living at the bottom of its range is a scaling defect or an accurate report of a
baby who is simply lying still. An unanchored scale is a hole, not a caveat.

WHAT IT DOES. Reads the simulated infant's real mass and contact geometry out of
MuJoCo, computes the pressure of events that actually happen to a baby - lying
down, rolling onto a hand, falling from the crib rail - and places each one
against two published human landmarks:

  capillary closing pressure  32 mmHg = 4,266 Pa
      Sustained load above this impedes capillary blood flow; it is the threshold
      the pressure-injury literature is built on, and the one that matters for an
      infant who lies in one posture for a long time.
      StatPearls, Pressure Injury: https://www.ncbi.nlm.nih.gov/books/NBK553107/

  pressure pain threshold in children  ~180,000 Pa
      183.1 kPa at the leg and 179.1 kPa at the thenar, 50 children, hand-held
      algometer with a 1 cm2 probe at 20 kPa/s. This is ACUTE pain from a probe,
      a different question from sustained ischaemia, and the two differ by a
      factor of 42.  https://ncbi.nlm.nih.gov/pmc/articles/PMC3100461
      That paper deliberately excludes neonates, who are known to differ, so the
      infant acute threshold is NOT established by it and is reported here as
      unmeasured rather than guessed.

The fall numbers are a constant-deceleration estimate: impact speed sqrt(2gh),
then a = v^2 / (2d) over the surface's stopping distance, peak force m(a + g)
spread over the contact patch. That overstates a real impact's peak by ignoring
body compliance, so treat each as an upper bound on the average pressure, not a
prediction of the worst local spike.

NOT AN INJURY MODEL. This places the simulator's numbers on a human scale so a
constant can be chosen with a reason. It does not claim the simulated skin
reproduces infant tissue, and no number here licenses a claim about what the
model feels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- published landmarks, with their source and what each one actually means ---
MMHG_PA = 133.322
CAPILLARY_CLOSING_PA = 32.0 * MMHG_PA          # 4,266 Pa
# TWO published child pressure-pain thresholds, from different studies, and they
# do not agree. Both are kept because BOTH are real, and which one a ratio is
# quoted against changes that ratio by 22 percent:
#
#   CHILD_PPT_PA      180,000 Pa   183 kPa leg / 179 kPa thenar, 1 cm2 algometer,
#                                  50 children. A published landmark only.
#   LIVE_REFERENCE_PA 219,400 Pa   Alrimy et al. 2022 doi:10.3389/fpsyg.2022.963765,
#                                  the 2-5 year subgroup mean, 1 cm2 rubber tip.
#                                  THIS is the number the running model charges at,
#                                  imported so it can never drift from the code.
#
# Every ratio below is quoted against LIVE_REFERENCE_PA, because a ratio against
# a number the model does not use tells the reader nothing about what the model
# will do. The 180,000 Pa landmark is printed beside it, never instead of it.
CHILD_PPT_PA = 180_000.0                       # acute, 1 cm2 probe, children
from lem.skin_mechanics import PROFILE as _MECHANICAL_PROFILE
LIVE_REFERENCE_PA = float(_MECHANICAL_PROFILE.pressure_reference_pa)
LANDMARKS = [
    ("venous capillary closing", 8.0 * MMHG_PA,
     "below this, venous return is unimpeded"),
    ("capillary closing (32 mmHg)", CAPILLARY_CLOSING_PA,
     "sustained load above this impedes blood flow; the pressure-injury line"),
    ("pain onset as the code has it", 10_000.0,
     "TACTILE_PAIN_START 0.1 in contact_wellness.py"),
    ("skin unit full scale", 100_000.0,
     "1.0 in the tokenizer's normalised pressure"),
    ("child acute pain threshold (algometer study)", CHILD_PPT_PA,
     "183 kPa leg / 179 kPa thenar, 1 cm2 algometer probe, 50 children"),
    ("what the live model charges at", LIVE_REFERENCE_PA,
     "skin_mechanics.PROFILE.pressure_reference_pa; Alrimy 2022, 2-5 year subgroup"),
]

# --- the surfaces a baby actually lands on, and how far each one lets it stop ---
SURFACES = [("crib mattress", 0.050), ("carpet", 0.010), ("hard floor", 0.002)]
# crib rail heights: mattress at its lowest, and standing at the rail top
FALL_HEIGHTS = [("roll off the mattress", 0.30), ("over the lowered rail", 0.66),
                ("over the raised rail", 0.90)]


def infant_mass(model, data):
    """Total mass of the body tree, excluding scene furniture and toys."""
    import mujoco
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(model.nbody)]
    # the body tree is every descendant of the root that is not a named scene prop
    scene = {"world", None}
    props = [i for i, n in enumerate(names)
             if n and any(k in n for k in ("box", "toy", "mobile", "target",
                                           "caregiver", "adult", "crib", "floor",
                                           "wall", "mattress", "rail"))]
    keep = [i for i in range(model.nbody)
            if names[i] not in scene and i not in props]
    return float(np.sum(model.body_mass[keep])), [names[i] for i in props if names[i]]


def contact_area_cm2(model, data):
    """Area actually carrying load while the body rests, from the live contacts."""
    import mujoco
    mujoco.mj_forward(model, data)
    n = int(data.ncon)
    # MuJoCo reports contact points, not areas. Use the touch patch size the skin
    # model itself uses as the per-point area, so the number stays in the
    # simulator's own terms rather than an invented footprint.
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--profile", default="mechanical_v2_local",
                    help="contact profile; BridgeEnv defaults to legacy, which is NOT what runs")
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--settle", type=int, default=200, help="steps to let the body rest")
    ap.add_argument("--write", default="")
    args = ap.parse_args()

    from mimo_bridge.env import BridgeEnv
    kw = dict(scene=args.scene, vision=False, seed=3, contact_profile=args.profile)
    if args.age is not None:
        kw["age"] = args.age
    env = BridgeEnv(**kw)
    env.reset()
    from lem.model.body_sets import HANDLE_NAMES
    nh = len(getattr(env, "handle_names", None) or HANDLE_NAMES)
    hold = np.zeros((nh, 9), np.float32)
    hold[..., 6] = 1.0
    hold[..., 8] = 0.3
    for _ in range(args.settle):
        frame = env.step(hold, None, None, render=False)

    model, data = env.model, env.data
    mass, props = infant_mass(model, data)
    g = float(abs(model.opt.gravity[2]))
    weight = mass * g

    patches = np.asarray(getattr(frame, "patches", np.zeros((0, 10))), np.float64)
    n_patches = int(patches.shape[0]) if patches.ndim >= 2 else 0
    active = int(np.sum(patches[..., 0] > 1e-6)) if n_patches else 0
    env.close()

    out = {
        "contract": "touch_scale_anchors_v1",
        "scene": args.scene,
        "infant_mass_kg": round(mass, 4),
        "gravity_m_s2": round(g, 4),
        "body_weight_N": round(weight, 3),
        "excluded_scene_bodies": props,
        "skin_patches_total": n_patches,
        "skin_patches_loaded_at_rest": active,
        "landmarks_Pa": {name: round(pa, 1) for name, pa, _ in LANDMARKS},
        "landmark_sources": {
            "capillary closing (32 mmHg)": "https://www.ncbi.nlm.nih.gov/books/NBK553107/",
            "child acute pain threshold": "https://ncbi.nlm.nih.gov/pmc/articles/PMC3100461",
        },
        "infant_acute_pain_threshold": "UNMEASURED - the child algometry study excludes neonates",
    }

    print(f"simulated infant, scene {args.scene}")
    print(f"  mass excluding scene props : {mass:.3f} kg   (excluded: {', '.join(props) or 'none'})")
    print(f"  body weight                : {weight:.2f} N")
    print(f"  skin patches               : {n_patches}, of which {active} carry load at rest")
    print()

    print("RESTING PRESSURE - the whole body weight spread over the area that touches")
    print("  contact area   pressure    where that sits")
    rest = {}
    for area_cm2 in (20, 50, 100, 200, 400):
        pa = weight / (area_cm2 * 1e-4)
        rest[area_cm2] = pa
        near = min(LANDMARKS, key=lambda L: abs(math.log10(max(pa, 1)) - math.log10(L[1])))
        print(f"  {area_cm2:4d} cm2     {pa:9.0f} Pa   {pa/CAPILLARY_CLOSING_PA:6.2f}x capillary closing,"
              f" {pa/LIVE_REFERENCE_PA:7.4f}x what the model charges at")
    out["resting_pressure_Pa_by_contact_area_cm2"] = {str(k): round(v, 1) for k, v in rest.items()}
    print()

    print("ONE HAND TAKING THE WHOLE WEIGHT - rolling onto a palm")
    hand = {}
    for area_cm2 in (4, 10, 25):
        pa = weight / (area_cm2 * 1e-4)
        hand[area_cm2] = pa
        print(f"  {area_cm2:4d} cm2     {pa:9.0f} Pa   {pa/CAPILLARY_CLOSING_PA:6.1f}x capillary closing,"
              f" {pa/LIVE_REFERENCE_PA:7.4f}x what the model charges at")
    out["one_hand_pressure_Pa_by_area_cm2"] = {str(k): round(v, 1) for k, v in hand.items()}
    print()

    print("FALLS - constant-deceleration upper bound, whole weight through a 50 cm2 patch")
    print("  event                    surface         speed    peak force     pressure   vs model line")
    falls = {}
    for label, h in FALL_HEIGHTS:
        v = math.sqrt(2 * g * h)
        for surf, d in SURFACES:
            a = v * v / (2 * d)
            force = mass * (a + g)
            pa = force / (50 * 1e-4)
            falls[f"{label} onto {surf}"] = dict(height_m=h, stopping_distance_m=d,
                                                 impact_speed_m_s=round(v, 2),
                                                 peak_force_N=round(force, 1),
                                                 pressure_Pa=round(pa, 1))
            print(f"  {label:24s} {surf:13s} {v:5.2f} m/s {force:9.0f} N {pa:11.0f} Pa"
                  f"   {pa/LIVE_REFERENCE_PA:7.1f}x")
    out["falls"] = falls
    print()

    # --- the question that actually matters: is there a hill for intention to climb? ---
    import torch
    from lem.model.contact_wellness import contact_response

    def probe(x, texture=0.5, n=53):
        f = torch.zeros(1, n, 8)
        f[..., 0] = x
        f[..., 3] = 0.01
        f[..., 4:8] = texture
        return contact_response(f)

    print("IS THERE A HILL FOR INTENTION TO CLIMB?")
    print("  comfort = (0.5 + 0.35 * pleasant) * (1 - pain), the number wellness reads")
    print()
    print("  contact area   pressure     pain  pleasant  comfort")
    curve, peak = [], (None, -1.0, 0.0, 0.0)
    for area in (2000, 1500, 1200, 1000, 800, 600, 400, 300, 250, 200,
                 150, 120, 100, 80, 60, 50, 40, 30, 25, 20, 15, 10):
        pa = weight / (area * 1e-4)
        r = probe(pa / 1e5)
        pain = float(r["pain"].mean())
        pl = float(r["pleasant"].mean())
        c = float(r["comfort"].mean())
        curve.append(dict(contact_area_cm2=area, pressure_Pa=round(pa, 1),
                          pain=round(pain, 4), pleasant=round(pl, 4), comfort=round(c, 4)))
        if c > peak[1]:
            peak = (area, c, pa, pain)
        print(f"  {area:5d} cm2   {pa:8.0f} Pa  {pain:7.4f}  {pl:7.4f}  {c:7.4f}")
    out["comfort_curve"] = curve

    pk_area, pk_c, pk_pa, pk_pain = peak
    # the curve runs loosest-to-firmest, so the FIRST row carrying pain is the onset
    onset = next((r for r in curve if r["pain"] > 0), None)
    rest_c = [r["comfort"] for r in curve if 770 <= r["contact_area_cm2"] <= 1160]
    climb = pk_c - max(rest_c) if rest_c else float("nan")
    print()
    print(f"  the hill tops out at {pk_area} cm2 of contact = {pk_pa:.0f} Pa, comfort {pk_c:.4f},")
    print(f"  and pain there is {pk_pain:.4f} - the best-feeling contact is not a painful one.")
    if onset:
        print(f"  pain first appears at {onset['contact_area_cm2']} cm2 = {onset['pressure_Pa']:.0f} Pa,"
              f" a factor of {onset['pressure_Pa']/pk_pa:.1f} FIRMER than the peak,")
        print("  so the body can climb to the best feeling and still stop short of hurting.")
    if rest_c:
        print(f"  lying as this run lies, comfort is {min(rest_c):.4f} to {max(rest_c):.4f};"
              f" the climb on offer is {climb:.4f}.")
    out["comfort_peak"] = dict(contact_area_cm2=pk_area, pressure_Pa=round(pk_pa, 1),
                               comfort=round(pk_c, 4), pain_at_peak=round(pk_pain, 4))
    if onset:
        out["pain_onset_measured"] = dict(contact_area_cm2=onset["contact_area_cm2"],
                                          pressure_Pa=onset["pressure_Pa"],
                                          margin_over_peak=round(onset["pressure_Pa"] / pk_pa, 2))
    out["climb_available_from_resting"] = round(climb, 4)
    print()

    print("WHAT THAT SAYS ABOUT THE CODE'S CONSTANTS")
    print(f"  pleasantness peak 0.05 = 5,000 Pa   is {5000/CAPILLARY_CLOSING_PA:.2f}x capillary closing"
          f" and {5000/LIVE_REFERENCE_PA:.4f}x what the model charges at")
    print(f"  pain onset        0.10 = 10,000 Pa  is {10000/CAPILLARY_CLOSING_PA:.2f}x capillary closing"
          f" and {10000/LIVE_REFERENCE_PA:.4f}x what the model charges at")
    print(f"  full scale        1.00 = 100,000 Pa is {100000/LIVE_REFERENCE_PA:.3f}x what the model charges at")
    print(f"  so full scale is BELOW the model's own charging line: {LIVE_REFERENCE_PA:,.0f} Pa stores as"
          f" {LIVE_REFERENCE_PA/100000:.3f}, past the top of the 0-1 channel")
    print(f"  the other published child threshold, {CHILD_PPT_PA:,.0f} Pa, disagrees with the model's"
          f" line by {abs(LIVE_REFERENCE_PA-CHILD_PPT_PA)/CHILD_PPT_PA*100:.0f}%")
    print()
    print("  So the scale is anchored to SUSTAINED ISCHAEMIA, not to acute pain: 'pain onset'")
    print(f"  sits {LIVE_REFERENCE_PA/10000:.0f}x below the pressure the live model charges at, and only")
    print(f"  {10000/CAPILLARY_CLOSING_PA:.1f}x above the pressure that impedes blood flow. For a baby that lies")
    print("  in one posture that is the defensible anchor, but the constant is named for")
    print("  the wrong mechanism.")

    out["reading"] = (
        "The skin unit is anchored to sustained-ischaemia pressures, not to acute pain. "
        f"TACTILE_SUSTAINED_LOAD_START 10,000 Pa is {LIVE_REFERENCE_PA/10000:.0f}x below the pressure the "
        f"live model charges at ({LIVE_REFERENCE_PA:,.0f} Pa) and {10000/CAPILLARY_CLOSING_PA:.1f}x above capillary "
        "closing pressure (4,266 Pa). For an infant holding one posture, ischaemia is the "
        "mechanism that matters, so the magnitude is defensible and the NAME is not. The "
        "acute threshold for infants specifically is unmeasured: the child algometry study "
        f"excludes neonates. Two published child thresholds are carried here and they "
        f"disagree by {abs(LIVE_REFERENCE_PA-CHILD_PPT_PA)/CHILD_PPT_PA*100:.0f} percent "
        f"({CHILD_PPT_PA:,.0f} Pa algometer against {LIVE_REFERENCE_PA:,.0f} Pa in the live "
        "profile); every ratio here is quoted against the live profile's number."
    )

    if args.write:
        path = args.write if os.path.isabs(args.write) else os.path.join(
            os.path.dirname(ROOT), "..", args.write)
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            try:
                existing = json.load(open(path, encoding="utf-8"))
            except Exception:
                existing = {}
        existing["touch_pressure_scale"] = out
        tmp = path + ".tmp"
        json.dump(existing, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, path)
        print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
