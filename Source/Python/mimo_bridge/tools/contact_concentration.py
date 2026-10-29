# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Decompose the skin pressure a patch reports, and find where a spike comes from.

    python -m mimo_bridge.tools.contact_concentration [--scene crib]
        [--heights 0.05,0.10,0.20,0.30] [--trials 5] [--write <path.json>]

WHY. touch_adapter computes

    pressure_Pa = normal_sum / (n_active_points * one_point_area)

and the two halves come from DIFFERENT sources. `normal_sum` is MuJoCo's contact
solver, summed over the contacts it reports for that body. `n_active_points` is
MIMo's own skin sensor grid, counting the points whose force vector exceeds
1e-6 N. Nothing ties them together: a contact solver impulse can land with a
large force while only one or two sensor points register it, and the divisor is
then one point's area rather than the area the body is really resting on.

That is why resting pressure reads far above what body-weight-over-contact-area
predicts, and it is the first suspect for a drop from 0.20 m reporting a higher
peak than a drop from 0.30 m.

WHAT THIS PRINTS, for the hardest-pressed patch at each moment:
  the force MuJoCo reported, the number of sensor points that felt anything,
  the area those points imply, and the pressure that division produces
so a spike can be attributed to the numerator or to the denominator rather than
guessed at.
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

PRESSURE, SHEAR, SLIP, AREA = 0, 1, 2, 3


def pooler_of(env):
    return getattr(env, "pooler", None)


def describe_skin(env):
    """How many patches, how many sensor points, and how big one point is."""
    pooler = pooler_of(env)
    specs = getattr(pooler, "specs", None) or getattr(pooler, "patch_specs", None)
    rows = []
    if specs:
        for s in specs:
            rows.append(dict(body=getattr(s, "body_name", "?"),
                             points=int(getattr(s, "n_points", 0)),
                             scale_m=float(getattr(s, "scale_m", 0.0)),
                             point_area_mm2=float(getattr(s, "scale_m", 0.0)) ** 2 * 1e6))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--profile", default="mechanical_v2_local",
                    help="contact profile; BridgeEnv defaults to legacy, which is NOT what runs")
    ap.add_argument("--age", type=float, default=None,
                    help="body age in months; the skin grid GROWS, so this changes the point count")
    ap.add_argument("--heights", default="0.05,0.10,0.20,0.30")
    ap.add_argument("--trials", type=int, default=5)
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

    skin = describe_skin(env)
    out = {"contract": "contact_concentration_v1", "scene": args.scene,
           "age_months": args.age, "skin": skin}

    if skin:
        pts = np.array([r["points"] for r in skin])
        ar = np.array([r["point_area_mm2"] for r in skin])
        print("THE SKIN GRID")
        print(f"  patches (one per sensing body)      : {len(skin)}")
        print(f"  sensor points, total                : {int(pts.sum()):,}")
        print(f"  points per patch                    : min {pts.min()}, median {int(np.median(pts))}, max {pts.max()}")
        print(f"  one point's area                    : min {ar.min():.1f}, median {np.median(ar):.1f}, max {ar.max():.1f} mm2")
        print(f"  whole skin area if every point felt : {(pts*ar).sum()/100:,.0f} cm2")
        out["skin_summary"] = dict(patches=len(skin), points_total=int(pts.sum()),
                                   point_area_mm2_median=float(np.median(ar)),
                                   full_skin_area_cm2=float((pts * ar).sum() / 100))
        print()

    def snapshot_max_patch(step):
        a = np.asarray(getattr(step, "patches", None), dtype=np.float64)
        if a.ndim < 2 or a.shape[0] == 0:
            return None
        i = int(np.argmax(a[:, PRESSURE]))
        area_mm2 = a[i, AREA]
        pa = a[i, PRESSURE]
        # invert the division to recover the force the solver actually reported
        force_N = pa * area_mm2 * 1e-6
        pt = skin[i]["point_area_mm2"] if i < len(skin) and skin else float("nan")
        n_pts = area_mm2 / pt if pt and pt == pt and pt > 0 else float("nan")
        return dict(patch=i, pressure_Pa=pa, area_mm2=area_mm2,
                    force_N=force_N, active_points=n_pts,
                    body=skin[i]["body"] if i < len(skin) and skin else "?")

    for _ in range(args.settle):
        step = env.step(limp, None, None, render=False)
    rest = [snapshot_max_patch(env.step(limp, None, None, render=False)) for _ in range(60)]
    rest = [r for r in rest if r]
    if rest:
        med = int(np.argsort([r["pressure_Pa"] for r in rest])[len(rest) // 2])
        r = rest[med]
        print("RESTING, the hardest-pressed patch (median step)")
        print(f"  body {r['body']}   pressure {r['pressure_Pa']:,.0f} Pa")
        print(f"  MuJoCo reported {r['force_N']:.3f} N on it")
        print(f"  MIMo sensor points that felt anything: {r['active_points']:.0f}")
        print(f"  so the divisor is {r['area_mm2']:.1f} mm2 - the area of those points, "
              f"not of the body's footprint")
        out["resting_max_patch"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in r.items()}
        print()

    qpos0 = np.array(env.data.qpos)
    qvel0 = np.array(env.data.qvel)
    rng = np.random.default_rng(7)
    print("DROPS - the step where each trial peaks, decomposed")
    print("  height  trial   force N   points   area mm2     pressure Pa   body")
    out["drops"] = []
    for h in [float(x) for x in args.heights.split(",") if x.strip()]:
        for t in range(args.trials):
            env.data.qpos[:] = qpos0
            env.data.qvel[:] = qvel0
            env.data.qpos[2] += h
            if t:
                env.data.qpos[3:7] += rng.normal(0, 0.02, 4)
                q = env.data.qpos[3:7]
                env.data.qpos[3:7] = q / max(np.linalg.norm(q), 1e-9)
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            best = None
            for _ in range(args.fall_steps):
                r = snapshot_max_patch(env.step(limp, None, None, render=False))
                if r and (best is None or r["pressure_Pa"] > best["pressure_Pa"]):
                    best = r
            if best:
                print(f"  {h:5.2f} m   {t:2d}   {best['force_N']:9.2f} {best['active_points']:8.0f}"
                      f" {best['area_mm2']:10.1f} {best['pressure_Pa']:15,.0f}   {best['body']}")
                out["drops"].append(dict(height_m=h, trial=t,
                                         **{k: (round(v, 4) if isinstance(v, float) else v)
                                            for k, v in best.items()}))
    env.close()

    if out["drops"]:
        import collections
        print()
        print("  WHERE THE SPIKES COME FROM")
        by1 = [d for d in out["drops"] if d["active_points"] <= 1.5]
        print(f"  peaks landing on ONE sensor point: {len(by1)} of {len(out['drops'])}")
        if by1:
            print(f"    those read {min(d['pressure_Pa'] for d in by1):,.0f} to "
                  f"{max(d['pressure_Pa'] for d in by1):,.0f} Pa")
        rest_of = [d for d in out["drops"] if d["active_points"] > 1.5]
        if rest_of:
            print(f"  peaks spread over more than one point: {len(rest_of)}, reading "
                  f"{min(d['pressure_Pa'] for d in rest_of):,.0f} to "
                  f"{max(d['pressure_Pa'] for d in rest_of):,.0f} Pa")
        f = [d["force_N"] for d in out["drops"]]
        a = [d["area_mm2"] for d in out["drops"]]
        pr = [d["pressure_Pa"] for d in out["drops"]]
        fs, as_, ps = (max(f)/max(min(f),1e-9), max(a)/max(min(a),1e-9),
                       max(pr)/max(min(pr),1e-9))
        print(f"  force across all peaks  : {min(f):,.1f} to {max(f):,.1f} N        a factor of {fs:,.0f}")
        print(f"  DIVISOR across all peaks: {min(a):,.0f} to {max(a):,.0f} mm2      a factor of {as_:,.0f}")
        print(f"  pressure that results   : {min(pr):,.0f} to {max(pr):,.0f} Pa  a factor of {ps:,.0f}")

        # Attribution, NOT by comparing those three spans. pressure = force/area
        # exactly, so log P = log F - log A and the spread of log P splits into
        # two shares that sum to 1 with no leftover:
        #     share_force = cov(log P,  log F) / var(log P)
        #     share_area  = cov(log P, -log A) / var(log P)
        # A ratio of max/min spans does NOT decompose anything: force and area
        # move together here (a fingertip landing has both a small area and its
        # own force), so the wider raw span belongs partly to the other term.
        # The span ratio said the divisor moves 9.6x as much as the force; the
        # decomposition says neither dominates. The span ratio was wrong.
        lf, la, lp = (np.log(np.array(f)), np.log(np.array(a)), np.log(np.array(pr)))
        vp = float(np.var(lp))
        if vp > 1e-12 and len(f) > 2:
            sf = float(np.cov(lp, lf, bias=True)[0, 1]) / vp
            sa = float(np.cov(lp, -la, bias=True)[0, 1]) / vp
            rfp = float(np.corrcoef(lf, lp)[0, 1])
            print("  WHERE THE SPREAD COMES FROM (shares sum to 1, no remainder)")
            print(f"    how hard the landing was  : {sf*100:6.1f}%")
            print(f"    which part happened to hit: {sa*100:6.1f}%")
            # The conclusion FOLLOWS the numbers. A share can leave 0-100% because
            # force and area move together (a fingertip landing has a small area
            # AND a small force), and then one term explains more than all of the
            # spread while the other works against it.
            if rfp < -0.05:
                print(f"    force vs pressure correlation: {rfp:+.3f} - NEGATIVE")
                print("  The harder the landing, the LOWER this number tends to read.")
                print("  It is not a weak severity signal, it points the wrong way.")
                verdict = "area dominates; pressure is anti-correlated with impact force"
            elif sa > sf:
                print(f"    force vs pressure correlation: {rfp:+.3f}")
                print("  Which body part touched decides the reading more than how")
                print("  hard the landing was.")
                verdict = "area dominates"
            else:
                print(f"    force vs pressure correlation: {rfp:+.3f}")
                print("  How hard the landing was decides the reading more than")
                print("  which body part touched.")
                verdict = "force dominates"
            out["attribution"] = dict(force_span=round(fs, 1), divisor_span=round(as_, 1),
                                      pressure_span=round(ps, 1),
                                      share_force=round(sf, 4), share_area=round(sa, 4),
                                      corr_log_force_log_pressure=round(rfp, 4),
                                      verdict=verdict,
                                      method="log-variance decomposition; "
                                             "the earlier max/min span ratio was not a "
                                             "decomposition and is retracted")
        else:
            out["attribution"] = dict(force_span=round(fs, 1), divisor_span=round(as_, 1),
                                      pressure_span=round(ps, 1),
                                      note="too few peaks to decompose")
        parts = {}
        for d in out["drops"]:
            parts.setdefault(d["body"], []).append(d["pressure_Pa"])
        print("  peaks by body part:")
        for b, v in sorted(parts.items(), key=lambda kv: -len(kv[1])):
            print(f"    {b:22s} {len(v):2d} peaks, {min(v):,.0f} to {max(v):,.0f} Pa")
        out["attribution"] = dict(force_span=round(fs, 1), divisor_span=round(as_, 1),
                                  pressure_span=round(ps, 1),
                                  divisor_over_force=round(as_/max(fs, 1e-9), 2))

    if args.write:
        path = os.path.abspath(args.write)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            try:
                existing = json.load(open(path, encoding="utf-8"))
            except Exception:
                existing = {}
        existing["contact_concentration"] = out
        tmp = path + ".tmp"
        json.dump(existing, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, path)
        print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
