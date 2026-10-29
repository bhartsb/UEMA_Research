# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Then vs now, same world, side by side - so improvement (or its absence) is VISIBLE.

The operator's problem (2026-09-06): "the avatar still seems hardly differentiable from
when things started - just random flailing. I need to ascertain the real improvement."
This tool answers it with eyes, not gauges: the SAME crib world (same seed) is lived
twice - once by an untrained brain with today's architecture (what day one looked like)
and once by the trained checkpoint - recorded with the same camera, then stitched side
by side. Behavior counters from both runs are written next to the clip and to the
dashboard: mobile strikes by the hand, seconds with the fovea on the hand, squeaker
presses, head-lift looks, commanded movement size.

python -m mimo_bridge.tools.then_vs_now --ckpt <ckpt_dir> [--seconds 30] [--seed 11]
Run at a segment boundary (trainer down - this uses the GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run  # noqa: E402

OUT_BASE = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge",
                        "then_vs_now")


def _fresh_like(ckpt, device, intention):
    """An architecture-identical, properly initialized, UNTRAINED brain - what the
    model was on day one - built by the same recipe as the trained one, with the
    checkpoint load skipped."""
    import mimo_bridge.lem_model as LM
    orig = LM.LemLite.load_checkpoint
    LM.LemLite.load_checkpoint = lambda self, *a, **k: None
    try:
        model, a, _s, _f = _build_model(ckpt, device, seed=0, intention=intention)
    finally:
        LM.LemLite.load_checkpoint = orig
    return model, a


def rollout(model, a, limits, seconds, seed, video_path, tag):
    from mimo_bridge.datasource import SingleEnvSource
    env_kw = dict(scene="crib", run_dir=tempfile.mkdtemp(prefix=f"tvn{tag}_"),
                  vision=True, fovea_size=int(a.get("fovea_size") or 288),
                  fovea_fov=float(a.get("fovea_fov") or 24.0))
    for sk, ek in (("decoder_kp", "kp"), ("joint_damping_scale", "joint_damping_scale"),
                   ("hinge_armature", "hinge_armature"), ("finger_armature", "finger_armature"), ("caregiver_seconds", "caregiver_seconds"),
                   ("gaze_kp", "gaze_kp"), ("gaze_kd", "gaze_kd")):
        if a.get(sk) is not None:
            env_kw[ek] = a[sk]
    env_kw["cmd_horizon"] = int(a.get("action_repeat") or 5) * int(a.get("motor_hold") or 5)
    if a.get("audio") is not None:
        env_kw["audio_real"] = a.get("audio") == "real"
    for k, on in (("critters", "critters"), ("mobile", "mobile")):
        if a.get(on) is not None:
            env_kw[k] = a.get(on) == "on"
    for k in ("ring_fov", "ring_size"):
        if a.get(k) is not None:
            env_kw[k] = a[k]
    if limits.get("age_months") is not None:
        env_kw["age"] = float(limits["age_months"])
    reps = int(a.get("action_repeat") or 5)
    n_frames = int(seconds / 0.01)
    src = SingleEnvSource(tag=f"_tvn{tag}", seed=seed, **env_kw)
    counts = {"strikes_hand": 0, "strikes_any": 0, "squeaker": 0, "fovea_on_hand": 0,
              "headlook": 0, "orders": 0, "cmd_mag_sum": 0.0}
    try:
        model.reset_transition_state()
        frames = src.reset()
        src.env.start_video(video_path, n_frames, size=480, every=10)
        dev = dict(limits)
        for i in range(n_frames // reps):
            with torch.no_grad():
                res = model.step(frames, dev, train=False, step_index=i)
            for _ in range(reps):
                frames = src.step(res.motor_for_action, res.gaze_out, None,
                                  render=True)
                t = frames[0].meta.get("task") or {}
                counts["strikes_hand"] += int(bool(t.get("mobile_touched_hand")))
                counts["strikes_any"] += int(bool(t.get("mobile_touched")))
                counts["fovea_on_hand"] += int(bool(t.get("hand_in_fovea")))
            cr = (frames[0].meta.get("cmd_response") or {})
            if cr.get("cmd_mag") is not None:
                counts["cmd_mag_sum"] += float(cr["cmd_mag"])
                counts["orders"] += 1
            tk = frames[0].meta.get("task") or {}
            counts["squeaker"] = int(tk.get("squeaker_hits") or 0)       # cumulative
            counts["headlook"] = int(tk.get("headlook_events") or 0)     # cumulative
        # wait for the video writer to flush (it saves when full)
        for _ in range(40):
            if src.env._video is None:
                break
            src.env.step(None, None, None, ctrl=np.zeros(src.env.model.nu), render=False)
    finally:
        src.close()
    counts["fovea_on_hand_s"] = counts.pop("fovea_on_hand") * 0.01
    counts["mean_order_rad"] = (counts.pop("cmd_mag_sum") / counts.pop("orders")) \
        if counts.get("orders") else 0.0
    return counts


def stitch(left_mp4, right_mp4, out_mp4, left_label, right_label):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    L = imageio.mimread(left_mp4, memtest=False)
    R = imageio.mimread(right_mp4, memtest=False)
    n = min(len(L), len(R))
    frames = []
    for i in range(n):
        a, b = L[i], R[i]
        h = min(a.shape[0], b.shape[0])
        combo = np.concatenate([a[:h], b[:h]], axis=1)
        img = Image.fromarray(combo)
        dr = ImageDraw.Draw(img)
        dr.text((10, 8), left_label, fill=(255, 255, 80))
        dr.text((a.shape[1] + 10, 8), right_label, fill=(80, 255, 120))
        frames.append(np.asarray(img))
    imageio.mimwrite(out_mp4, frames, fps=10, quality=7)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(OUT_BASE, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M")

    now_model, a, _s, _f = _build_model(args.ckpt, args.device, 0,
                                        intention=(a_int := True))
    limits, _ = _limits_for_run(args.ckpt, now_model)
    limits["_action_repeat"] = int(a.get("action_repeat") or 5)
    day1_model, _a2 = _fresh_like(args.ckpt, args.device, intention=a_int)

    p_then = os.path.join(OUT_BASE, f"day1_{stamp}.mp4")
    p_now = os.path.join(OUT_BASE, f"now_{stamp}.mp4")
    c_then = rollout(day1_model, a, limits, args.seconds, args.seed, p_then, "d1")
    c_now = rollout(now_model, a, limits, args.seconds, args.seed, p_now, "now")

    combo = os.path.join(OUT_BASE, f"then_vs_now_{stamp}.mp4")
    try:
        stitch(p_then, p_now, combo, "DAY-ONE BRAIN", "TONIGHT'S BRAIN")
    except Exception as e:
        combo = None
        print(f"stitch failed ({e}) - the two single clips stand", flush=True)

    rep = {"ckpt": os.path.abspath(args.ckpt), "seconds": args.seconds,
           "seed": args.seed, "day1": c_then, "now": c_now,
           "video": combo or p_now, "stamp": stamp}
    json.dump(rep, open(os.path.join(OUT_BASE, f"counts_{stamp}.json"), "w"), indent=1)
    json.dump(rep, open(os.path.join(OUT_BASE, "latest.json"), "w"), indent=1)

    print(f"\nSame world, {args.seconds:.0f} s each:")
    print(f"  day-one brain : {c_then['strikes_hand']} hand strikes on the mobile, "
          f"fovea on its hand {c_then['fovea_on_hand_s']:.1f} s, "
          f"{c_then['squeaker']} squeaker presses, {c_then['headlook']} head-lift looks, "
          f"orders {c_then['mean_order_rad']:.2f} rad")
    print(f"  tonight brain : {c_now['strikes_hand']} hand strikes on the mobile, "
          f"fovea on its hand {c_now['fovea_on_hand_s']:.1f} s, "
          f"{c_now['squeaker']} squeaker presses, {c_now['headlook']} head-lift looks, "
          f"orders {c_now['mean_order_rad']:.2f} rad")
    print(f"  video: {combo or p_now}   ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
