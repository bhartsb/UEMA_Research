# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measure the body, then check every hard-coded constant against what it measured.

    python -m mimo_bridge.tools.calibrate [--steps 400] [--scene crib]

Why this exists. Every threshold in the bridge is a number somebody chose: the pressure
that counts as a pleasant touch, the speed that counts as moving, the prediction error the
curiosity gate calls interesting, the image difference the gaze reflex calls motion. A
number chosen from intuition and never checked against the body is a guess wearing a
uniform, and the ones that turned out wrong were not wrong by a little:

  * tactile pressure was read as MPa when the tokenizer emits Pa/1e5 - a factor of ten -
    and averaged over 256 patches when six are ever in contact, so the score sat pinned at
    0.999 with a whole-run swing of 0.009
  * the anti-buzz armature added 0.02 to joints whose own armature is 0.0002, and froze
    the eyes for seven runs
  * the gaze reflex's frame-gap limit was shorter than the render interval it had to
    tolerate, silencing it with no error

None of those needed cleverness to find. They needed somebody to print the range.

What it reports per constant:
    LIVE      the threshold sits inside the observed range - it can actually fire both ways
    NEVER     the observed values never reach it - the branch behind it is dead
    ALWAYS    every observed value passes it - the test is a no-op
    FLAT      the quantity it gates barely varies at all - the signal is diluted or saturated
    ?         nothing observed, so nothing can be said

FLAT and NEVER are not automatically bugs. A pain threshold SHOULD be untouched by a baby
lying quietly. The point is that the report says so out loud rather than letting a dead
branch look like a working one.
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


def q(v):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return None
    return dict(n=int(a.size), min=float(a.min()), p5=float(np.percentile(a, 5)),
                med=float(np.median(a)), p95=float(np.percentile(a, 95)),
                max=float(a.max()), std=float(a.std()))


def verdict(stat, thr, kind="above"):
    """Where does a threshold sit relative to what the body actually produces?"""
    if stat is None:
        return "?", "nothing observed"
    lo, hi = stat["min"], stat["max"]
    if hi - lo < 1e-12:
        return "FLAT", f"the quantity never varies (always {lo:.6g})"
    if kind == "above":
        if thr > hi:
            return "NEVER", f"threshold {thr:.6g} is above the highest observed {hi:.6g}"
        if thr < lo:
            return "ALWAYS", f"threshold {thr:.6g} is below the lowest observed {lo:.6g}"
    else:
        if thr < lo:
            return "NEVER", f"threshold {thr:.6g} is below the lowest observed {lo:.6g}"
        if thr > hi:
            return "ALWAYS", f"threshold {thr:.6g} is above the highest observed {hi:.6g}"
    pct = float((np.asarray([lo, hi]) < thr).mean())
    return "LIVE", f"threshold {thr:.6g} sits inside [{lo:.6g}, {hi:.6g}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--out", default=None, help="write the measurements as json")
    a = ap.parse_args(argv)

    import torch
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.gaze_reflex import GazeReflex, AZ_RANGE, EL_RANGE
    from mimo_bridge.movement_drive import MovementDrive
    from mimo_bridge.vision_adapter import acuity
    from lem.model.wellness import (WellnessEvaluators, TACTILE_PLEASANT_PEAK,
                                    TACTILE_PAIN_START, TACTILE_TEXTURE_REF)
    import mimo_bridge.lem_model as LM

    run = os.path.join(os.path.dirname(ROOT), "..", "Saved", "mimo_bridge", "calibrate")
    env = BridgeEnv(scene=a.scene, run_dir=run, vision=True, eye_size=256,
                    fovea_size=288, fovea_fov=24.0, seed=17, gaze_reflex=True,
                    caregiver_seconds=15.0)
    env.reset()
    m, d = env.model, env.data

    obs = {k: [] for k in ("pressure_active", "pressure_allpatch", "vib_active", "n_touching",
                           "hand_view_deg", "gaze_ball_deg", "eye_pitch_deg", "eye_speed",
                           "eye_yaw", "eye_pitch_joint", "patch_diff_peak", "contacts",
                           "body_speed", "jitter")}
    zero = np.zeros((24, 9))
    eye_dofs = [int(m.jnt_dofadr[m.joint(n).id]) for n in
                ("robot:left_eye_horizontal", "robot:left_eye_vertical")]
    prev_img = None
    for i in range(a.steps):
        raw = env.step(zero, (0.0, 0.0), render=(i % 5 == 4))
        t = raw.info.get("task", {})
        p = raw.patches
        touching = p[:, 0] > 0
        obs["n_touching"].append(float(touching.sum()))
        obs["pressure_allpatch"].append(float(p[:, 0].mean() / 1e5))
        if touching.any():
            obs["pressure_active"].append(float(p[touching, 0].mean() / 1e5))
            obs["vib_active"].append(float(p[touching, 4:8].mean() / 10.0))
        for key, tk in (("hand_view_deg", "hand_view_deg"), ("gaze_ball_deg", "gaze_ball_deg"),
                        ("eye_pitch_deg", "eye_pitch_deg")):
            if t.get(tk) is not None:
                obs[key].append(float(t[tk]))
        obs["contacts"].append(float(raw.n_contacts))
        obs["eye_speed"].append(max(abs(float(d.qvel[x])) for x in eye_dofs))
        obs["eye_yaw"].append(float(d.qpos[int(m.jnt_qposadr[m.joint("robot:left_eye_horizontal").id])]))
        obs["eye_pitch_joint"].append(float(d.qpos[int(m.jnt_qposadr[m.joint("robot:left_eye_vertical").id])]))
        if raw.eye_left is not None:
            cur = raw.eye_left[:256, :256].astype(np.float32).mean(axis=2)
            cur = cur.reshape(16, 16, 16, 16).mean(axis=(1, 3))
            if prev_img is not None:
                dd = np.abs(cur - prev_img)
                obs["patch_diff_peak"].append(float((dd - np.median(dd)).max()))
            prev_img = cur

    stats = {k: q(v) for k, v in obs.items()}
    rf = env.reflex
    env.close()

    print(f"\n=== measured over {a.steps} steps of the {a.scene} scene "
          f"({a.steps * 0.01:.1f} s of body time) ===")
    print(f"{'quantity':24s} {'min':>11s} {'median':>11s} {'max':>11s}")
    for k, s in stats.items():
        if s:
            print(f"  {k:22s} {s['min']:11.5g} {s['med']:11.5g} {s['max']:11.5g}")
        else:
            print(f"  {k:22s} {'-':>11s} {'-':>11s} {'-':>11s}   never observed")

    # ------------------------------------------------------------------ the constants
    checks = []

    def check(name, where, thr, stat, kind="above", note=""):
        v, why = verdict(stat, thr, kind)
        checks.append((v, name, where, why, note))

    # imported, never copied: a calibrator holding its own copy of a constant will one day
    # pass a number nobody runs (this one did, one edit after the value changed)
    check(f"tactile pleasant peak ({TACTILE_PLEASANT_PEAK*100:g} kPa)",
          "wellness.TACTILE_PLEASANT_PEAK",
          TACTILE_PLEASANT_PEAK, stats["pressure_active"], "above",
          "spec 8.2's 500 Pa is a fingertip on a texture; this body's contact is heavier")
    check(f"legacy sustained-load ramp start ({TACTILE_PAIN_START*100:g} kPa; "
          f"2.3x capillary closing, not an acute pain line, and unused under "
          f"--contact-profile mechanical_v2_local)",
          "wellness.TACTILE_SUSTAINED_LOAD_START",
          TACTILE_PAIN_START, stats["pressure_active"], "above", "spec 8.1")
    check("tactile texture reference", "wellness.TACTILE_TEXTURE_REF",
          TACTILE_TEXTURE_REF, stats["vib_active"], "above", "tanh reference for vibPower")
    check("reflex abs_min", "gaze_reflex.GazeReflex",
          rf.abs_min if rf else 5.0, stats["patch_diff_peak"], "above",
          "still-scene guard: ALWAYS is correct here in a moving crib (still-scene noise "
          "maxes at 2.9, smallest live event 8.8; the floor sits between at 5.0)")
    check("reflex eye_speed_gate", "gaze_reflex.GazeReflex",
          rf.eye_speed_gate if rf else 1.5, stats["eye_speed"], "above",
          "saccade suppression; above the eye's real speed = never suppresses")
    check("caregiver interval 15 s", "env.BridgeEnv",
          15.0, q([a.steps * 0.01]), "above", "vs the length of this sample")

    print(f"\n=== constants against those ranges ===")
    for v, name, where, why, note in checks:
        print(f"  [{v:6s}] {name}")
        print(f"           {why}")
        if note:
            print(f"           ({note})")
        print(f"           {where}")

    # ------------------------------------------------------------------ joint ranges
    print(f"\n=== declared ranges vs the model's real joint limits ===")
    for jn, declared in (("robot:left_eye_horizontal", AZ_RANGE),
                         ("robot:left_eye_vertical", EL_RANGE)):
        lo, hi = m.jnt_range[m.joint(jn).id]
        ok = abs(lo - declared[0]) < 1e-3 and abs(hi - declared[1]) < 1e-3
        print(f"  [{'OK    ' if ok else 'WRONG '}] {jn}: model [{lo:.3f}, {hi:.3f}] "
              f"vs code [{declared[0]:.3f}, {declared[1]:.3f}]")
    for jn in ("robot:left_eye_horizontal", "robot:right_elbow"):
        j = m.joint(jn).id
        dof = int(m.jnt_dofadr[j])
        print(f"  {jn}: armature {m.dof_armature[dof]:.5f}, damping {m.dof_damping[dof]:.5f}")

    # ------------------------------------------------------------------ acuity
    print(f"\n=== what the eyes resolve ===")
    for label, size, fov in (("wide", 256, 60.0), ("fovea", 288, 24.0)):
        ac = acuity(size, fov)
        print(f"  {label:6s} {size}px over {fov:g} deg = {ac['px_per_deg']:5.2f} px/deg, "
              f"{ac['cycles_per_deg']:4.2f} cycles/deg, about 20/{ac['snellen']:.0f}")

    # ------------------------------------------------------------------ wellness spread
    print(f"\n=== does each wellness term actually move? ===")
    cfg = LM.build_config()
    w = WellnessEvaluators(cfg)
    rows = {}
    for trial in range(24):
        B = 1
        mc = torch.randn(B, cfg["body"]["n_handles"], 9) * (0.2 + 0.05 * trial)
        preds = {k: torch.randn(B, cfg["model"]["d_model"]) for k in ("vision", "tactile")}
        tgts = {k: preds[k] + torch.randn_like(preds[k]) * (0.02 * trial) for k in preds}
        tac = torch.zeros(B, 256, 8)
        pr = stats["pressure_active"]["med"] if stats["pressure_active"] else 0.04
        tac[0, :6, 0] = pr * (0.2 + 0.4 * trial)
        tac[0, :6, 4:8] = 0.3
        out = w(preds, tgts, torch.randn(B, 6, cfg["model"]["d_model"]),
                motor_commands=mc, raw_tactile=tac, capability_features=torch.rand(B, 21))
        for k, v in out["grounded"].items():
            rows.setdefault(k, []).append(float(v))
    for k, v in rows.items():
        s = q(v)
        flag = "FLAT  " if s["max"] - s["min"] < 0.05 else "moves "
        print(f"  [{flag}] {k:18s} {s['min']:.4f} .. {s['max']:.4f}   swing {s['max']-s['min']:.4f}")

    if a.out:
        json.dump({"stats": stats,
                   "checks": [{"verdict": v, "name": n, "where": wh, "why": y}
                              for v, n, wh, y, _ in checks]},
                  open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
