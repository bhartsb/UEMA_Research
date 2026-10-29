# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Compare candidate impact-severity signals: which one grades with fall height?

    python -m mimo_bridge.tools.impact_grading [--profile mechanical_v2_local]
        [--age 24] [--heights ...] [--trials 7] [--write <path.json>]

WHY. The signal wellness currently reads - peak per-patch pressure - does not
grade with how far the body fell. Measured on the live path at age 24 months it
INVERTS: pain 0.9991 at a 0.30 m drop against 0.9164 at 0.90 m. And measured
across 30 drop peaks, that pressure is anti-correlated with impact force at
-0.234, because it is divided by the sensor-support area rather than a contact
area (mimo_bridge/tools/contact_concentration.py).

Before any aversion term is rewritten, the replacement signal has to be MEASURED
to grade, and its reference constant measured too: a constant in an objective is
never chosen before the measurement that justifies it.

WHAT IT MEASURES, every physics tick - not every observation step, because a
0.005 s solver tick can peak and be gone before the 0.04 s observation reports it:

  total_normal_force_N   every contact normal on every sensing body, summed
  site_normal_force_N    the single hardest-loaded body, in newtons
  site_pressure_Pa       the signal wellness reads today, for comparison

For each it prints the median across trials at a height, the spread across trials
at ONE height - that spread is the measurement's own noise - and whether the
medians rise with height without exception.

A signal is usable as impact severity only if the rise across heights is larger
than the spread within a height. Both numbers are printed so that is checkable
rather than asserted.
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

# Same rejection as fall_response: a trial that starts inside geometry is ejected
# by the solver with a force no fall produces, and that is the measurement's own
# artifact rather than the simulator's physics.
PENETRATION_LIMIT_M = 1e-3


def wellness_pain(peak_pa, profile, sites=53):
    """What contact_wellness scores for a peak of this size on this profile."""
    import torch
    from lem.model.contact_wellness import contact_response
    f = torch.zeros(1, sites, 8)
    f[..., 0] = peak_pa / 1e5
    f[..., 3] = 0.01
    f[..., 4:8] = 0.5
    kw = {}
    if profile != "legacy":
        h = torch.zeros(1, sites, 3)
        h[..., 0] = peak_pa
        kw["history"] = h
    return float(contact_response(f, profile=profile, **kw)["pain"].mean())


def ordering_accuracy(rows, key):
    """How often ONE fall from higher up reads worse than ONE fall from lower.

    Medians can order correctly while single falls do not, and learning happens
    on single falls. For every pair of heights this counts every (lower trial,
    higher trial) pair and asks whether the higher one read worse. 1.00 means
    always; 0.50 means a coin toss; below 0.50 means the signal is backwards.
    """
    wins = ties = total = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            low = rows[i]["trials"][key]
            high = rows[j]["trials"][key]
            for a in low:
                for b in high:
                    total += 1
                    if b > a:
                        wins += 1
                    elif b == a:
                        ties += 1
    if not total:
        return float("nan"), 0
    return (wins + 0.5 * ties) / total, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--profile", default="mechanical_v2_local")
    ap.add_argument("--age", type=float, default=24.0)
    ap.add_argument("--heights", default="0.05,0.10,0.30,0.45,0.66,0.90")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--settle", type=int, default=200)
    ap.add_argument("--fall-steps", type=int, default=120)
    ap.add_argument("--write", default="")
    args = ap.parse_args()

    from mimo_bridge.env import BridgeEnv
    from lem.model.body_sets import HANDLE_NAMES
    import mujoco

    kw = dict(scene=args.scene, vision=False, seed=3, contact_profile=args.profile)
    if args.age is not None:
        kw["age"] = args.age
    env = BridgeEnv(**kw)
    env.reset()
    nh = len(getattr(env, "handle_names", None) or HANDLE_NAMES)
    limp = np.zeros((nh, 9), np.float32)
    limp[..., 6] = 1.0

    sensing = {s.body_id for s in env.pooler.specs}
    body_of_geom = env.model.geom_bodyid
    peak = {}

    def observe(_previous_velocity=None):
        """Runs once per physics tick, before observation thinning."""
        d = env.data
        total = 0.0
        per_body = {}
        force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(body_of_geom[c.geom1]), int(body_of_geom[c.geom2])
            hit = b1 if b1 in sensing else (b2 if b2 in sensing else None)
            if hit is None:
                continue
            mujoco.mj_contactForce(env.model, d, i, force)
            n = abs(float(force[0]))       # column 0 is the contact-frame normal
            total += n
            per_body[hit] = per_body.get(hit, 0.0) + n
        peak["total"] = max(peak.get("total", 0.0), total)
        peak["site"] = max(peak.get("site", 0.0), max(per_body.values(), default=0.0))

    inner = getattr(env, "_mechanical_substep", None)

    def chained(previous_velocity):
        if inner is not None:
            inner(previous_velocity)
        observe(previous_velocity)

    env.env._bridge_contact_observer = chained

    for _ in range(args.settle):
        env.step(limp, None, None, render=False)

    peak.clear()
    for _ in range(60):
        env.step(limp, None, None, render=False)
    weight_n = float(env.model.body_mass.sum()) * 9.81
    rest = dict(total_N=peak.get("total", 0.0), site_N=peak.get("site", 0.0))
    print(f"scene {args.scene}   profile {args.profile}   age {args.age} months")
    print(f"  RESTING, peak over 60 steps: whole body {rest['total_N']:,.1f} N, "
          f"hardest body {rest['site_N']:,.1f} N")
    print(f"  one reference: this body's own weight is {weight_n:,.1f} N")
    print()

    qpos0 = np.array(env.data.qpos)
    qvel0 = np.array(env.data.qvel)
    rng = np.random.default_rng(7)
    heights = [float(x) for x in args.heights.split(",") if x.strip()]
    rows, rejected = [], []

    for h in heights:
        tot, site, pres, hurt = [], [], [], []
        for t in range(args.trials):
            env.data.qpos[:] = qpos0
            env.data.qvel[:] = qvel0
            env.data.qpos[2] += h
            if t:                          # trial 0 is the unperturbed drop
                env.data.qpos[3:7] += rng.normal(0, 0.02, 4)
                q = env.data.qpos[3:7]
                env.data.qpos[3:7] = q / max(np.linalg.norm(q), 1e-9)
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            deepest = min((c.dist for c in env.data.contact[:env.data.ncon]), default=0.0)
            if deepest < -PENETRATION_LIMIT_M:
                rejected.append(dict(height_m=h, trial=t, penetration_m=round(deepest, 5)))
                continue
            peak.clear()
            p_pa = 0.0
            for _ in range(args.fall_steps):
                step = env.step(limp, None, None, render=False)
                a = np.asarray(getattr(step, "patches", None), dtype=np.float64)
                if a.ndim == 2 and a.shape[0]:
                    p_pa = max(p_pa, float(a[:, 0].max()))
            tot.append(peak.get("total", 0.0))
            site.append(peak.get("site", 0.0))
            pres.append(p_pa)
            hurt.append(wellness_pain(p_pa, args.profile))
        if not tot:
            print(f"  {h:5.2f} m   every trial started inside geometry")
            continue

        def stat(v):
            return dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v)))

        rows.append(dict(height_m=h, measured=len(tot), total_N=stat(tot),
                         site_N=stat(site), site_Pa=stat(pres), pain=stat(hurt),
                         trials=dict(total_N=tot, site_N=site, site_Pa=pres, pain=hurt)))

    env.close()

    def report(key, unit, label):
        print(label)
        print(f"  height  n         median   {'min - max':<26}spread   vs previous")
        ok, prev = True, None
        for r in rows:
            v = r[key]
            sp = v["max"] / max(v["min"], 1e-9)
            note = ""
            if prev is not None:
                note = "RISES" if v["median"] > prev else "*** FALLS ***"
                if v["median"] <= prev:
                    ok = False
            prev = v["median"]
            print(f"  {r['height_m']:5.2f} m {r['measured']:2d} {v['median']:14,.0f}{unit}  "
                  f"{v['min']:11,.0f} - {v['max']:<12,.0f}{sp:6.1f}x   {note}")
        if not rows:
            return dict(monotone=False, note="nothing measured")
        rise = rows[-1][key]["median"] / max(rows[0][key]["median"], 1e-9)
        worst = max(r[key]["max"] / max(r[key]["min"], 1e-9) for r in rows)
        print(f"  lowest height to highest : {rise:,.1f}x")
        print(f"  worst spread at ONE height: {worst:,.1f}x")
        print(f"  rises at every height without exception: {'YES' if ok else 'NO'}")
        print(f"  usable as impact severity (rise beats within-height spread): "
              f"{'YES' if ok and rise > worst else 'NO'}")
        print()
        return dict(monotone=bool(ok), rise_low_to_high=round(float(rise), 2),
                    worst_spread_within_height=round(float(worst), 2),
                    usable=bool(ok and rise > worst))

    v_tot = report("total_N", " N", "CANDIDATE 1 - total normal force on the whole body")
    v_site = report("site_N", " N", "CANDIDATE 2 - normal force on the hardest-loaded body")
    v_pres = report("site_Pa", " Pa", "WHAT WELLNESS READS TODAY - peak per-patch pressure")

    print("CAN ONE FALL BE RANKED AGAINST ANOTHER?")
    print("  Every pair of trials from two different heights. 1.00 = the higher")
    print("  fall always reads worse, 0.50 = a coin toss, below 0.50 = backwards.")
    order = {}
    for label, key in (("total normal force", "total_N"),
                       ("hardest-body normal force", "site_N"),
                       ("peak patch pressure", "site_Pa"),
                       ("the pain wellness scores", "pain")):
        acc, n = ordering_accuracy(rows, key)
        order[key] = round(float(acc), 4)
        verdict = ("usable" if acc >= 0.90 else
                   "weak" if acc >= 0.75 else
                   "a coin toss" if acc >= 0.45 else "BACKWARDS")
        print(f"  {label:28s} {acc:.3f} over {n:,} pairs   {verdict}")
    print()

    usable = [n for n, v in (("total normal force", v_tot),
                             ("hardest-body normal force", v_site),
                             ("peak patch pressure", v_pres)) if v.get("usable")]
    print("VERDICT")
    if usable:
        print("  signals that grade with fall height here: " + ", ".join(usable))
    else:
        print("  NO candidate graded cleanly across this whole ladder")

    out = dict(contract="impact_grading_v1", scene=args.scene, profile=args.profile,
               age_months=args.age, trials=args.trials, body_weight_N=round(weight_n, 2),
               resting={k: round(v, 3) for k, v in rest.items()}, heights=rows,
               rejected_for_starting_overlap=rejected,
               verdict=dict(total_normal_force=v_tot, site_normal_force=v_site,
                            site_pressure_today=v_pres, usable=usable),
               ordering_accuracy=order)
    if rejected:
        print(f"  {len(rejected)} trials thrown away for starting inside geometry")

    if args.write:
        path = os.path.abspath(args.write)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            try:
                existing = json.load(open(path, encoding="utf-8"))
            except Exception:
                existing = {}
        existing.setdefault("impact_grading", {})[args.profile] = out
        tmp = path + ".tmp"
        json.dump(existing, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, path)
        print(f"  written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
