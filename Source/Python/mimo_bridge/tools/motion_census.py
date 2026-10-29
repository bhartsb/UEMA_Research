# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Motion census: is the avatar moving too fast, too slow, or un-infant-like - and is
that CHOSEN or accidental? (operator 2026-09-06: "the avatar is moving all the time
quickly which seems too much")

Runs the current brain in the crib on CPU (safe beside a live GPU trainer), tracks the
hands and feet in world space, and reports against Saved/mimo_bridge/infant_reference
.json - matched by CAPABILITY STAGE, not calendar age (the AI timeline is its own):
  hand/foot speed medians and peaks (cm/s)
  duty cycle: fraction of time any limb moves above the real movement-unit floor
  bout structure: median movement-bout and pause lengths (infants move in bouts)
  kick frequency in active bouts (Hz)
Writes Saved/mimo_bridge/motion_census.json; the gap clock's sanity section reads it.
python -m mimo_bridge.tools.motion_census --ckpt <ckpt_dir> [--seconds 30]
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
SAVED = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge")

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    t0 = time.time()
    ref = json.load(open(os.path.join(SAVED, "infant_reference.json")))
    floor = float(ref["movement_unit_floor_cm_s"])
    stage = ref["stages"]["reaching_and_batting (first deliberate contact - closest to the sim's current capability stage)"]

    model, a, _s, _f = _build_model(args.ckpt, args.device, 0, intention=False)
    if a.get("audio") == "real":
        model.config["modalities"].setdefault("audio", {})["data_source"] = "real"
    limits, _ = _limits_for_run(args.ckpt, model)
    from mimo_bridge.datasource import SingleEnvSource
    env_kw = dict(scene="crib", run_dir=tempfile.mkdtemp(prefix="census_"), vision=True,
                  fovea_size=int(a.get("fovea_size") or 288),
                  fovea_fov=float(a.get("fovea_fov") or 24.0))
    if a.get("audio") is not None:
        env_kw["audio_real"] = a.get("audio") == "real"
    for k, on in (("critters", "critters"), ("mobile", "mobile")):
        if a.get(on) is not None:
            env_kw[k] = a.get(on) == "on"
    reps = int(a.get("action_repeat") or 5)
    src = SingleEnvSource(tag="_census", seed=180, **env_kw)
    bodies = ("right_hand", "left_hand", "right_foot", "left_foot")
    speeds = {b: [] for b in bodies}
    goal_counts = {"strikes": 0, "fovea": 0}
    try:
        m, d = src.env.model, src.env.data
        ids = {b: m.body(b).id for b in bodies}
        model.reset_transition_state()
        frames = src.reset()
        dev = dict(limits)
        prev = {b: np.array(d.xpos[ids[b]]) for b in bodies}
        n_orders = int(args.seconds / 0.01 / reps)
        for i in range(n_orders):
            with torch.no_grad():
                res = model.step(frames, dev, train=False, step_index=i)
            for _ in range(reps):
                frames = src.step(res.motor_for_action, res.gaze_out, None, render=True)
                for b in bodies:
                    p = np.array(d.xpos[ids[b]])
                    speeds[b].append(float(np.linalg.norm(p - prev[b]) / 0.01 * 100.0))
                    prev[b] = p
                t = frames[0].meta.get("task") or {}
                goal_counts["strikes"] += int(bool(t.get("mobile_touched_hand")))
                goal_counts["fovea"] += int(bool(t.get("hand_in_fovea")))
    finally:
        src.close()

    any_speed = np.max(np.stack([np.array(speeds[b]) for b in bodies]), axis=0)
    moving = any_speed > floor
    duty = float(moving.mean())
    # bout structure: runs of moving / still
    bouts, pauses, run, state = [], [], 0, moving[0]
    for mv in moving:
        if mv == state:
            run += 1
        else:
            (bouts if state else pauses).append(run * 0.01)
            run, state = 1, mv
    (bouts if state else pauses).append(run * 0.01)
    # kick frequency: foot-speed peaks above half its own p95, per active second
    foot = np.maximum(np.array(speeds["right_foot"]), np.array(speeds["left_foot"]))
    thr = 0.5 * np.percentile(foot, 95)
    peaks = int(np.sum((foot[1:-1] > thr) & (foot[1:-1] >= foot[:-2])
                       & (foot[1:-1] >= foot[2:])))
    active_s = max(float(moving.sum()) * 0.01, 1e-6)
    kick_hz = peaks / active_s / 2.0          # up+down per cycle

    hands = np.concatenate([speeds["right_hand"], speeds["left_hand"]])
    rep_lo, rep_hi = stage["hand_speed_cm_s_typical"]
    pk_lo, pk_hi = stage["reach_peak_cm_s"]
    d_lo, d_hi = stage["duty_cycle_moving_active_alert"]
    med, p95 = float(np.median(hands[hands > floor])), float(np.percentile(hands, 95))

    def band(v, lo, hi):
        return "below" if v < lo else "above" if v > hi else "inside"

    out = {"computed_at": time.strftime("%Y-%m-%d %H:%M"), "ckpt": os.path.abspath(args.ckpt),
           "seconds": args.seconds,
           "hand_speed_median_cm_s": med, "hand_speed_p95_cm_s": p95,
           "duty_cycle": duty, "kick_hz_active": kick_hz,
           "bout_median_s": float(np.median(bouts)) if bouts else None,
           "pause_median_s": float(np.median(pauses)) if pauses else None,
           "verdicts": {
               "hand_speed": f"median {med:.0f} cm/s is {band(med, rep_lo, rep_hi)} the "
                             f"infant reaching-stage band {rep_lo}-{rep_hi}; peaks "
                             f"{p95:.0f} cm/s are {band(p95, pk_lo, pk_hi)} the reach-peak "
                             f"band {pk_lo}-{pk_hi}",
               "duty": f"moving {100*duty:.0f}% of the time vs the active-alert infant "
                       f"band {int(100*d_lo)}-{int(100*d_hi)}% - "
                       + ("continuous motion, no real pauses: un-infant-like restlessness"
                          if duty > d_hi else "inside the band" if duty >= d_lo
                          else "stiller than an active infant"),
               "bouts": (f"median movement bout "
                         f"{float(np.median(bouts)) if bouts else 0:.1f}s, median pause "
                         f"{float(np.median(pauses)) if pauses else 0:.2f}s vs infant "
                         f"bouts of 1-5s with real pauses")},
           "goal_service": {
               "strikes_per_min": goal_counts["strikes"] / max(args.seconds / 60, 1e-6),
               "fovea_on_hand_pct": 100.0 * goal_counts["fovea"] / max(len(any_speed), 1)},
           "criterion": "the DECIDING criterion is goal service (operator 2026-09-06: "
                        "'we want movement that allows our goals') - strikes, "
                        "eyes-on-hand, learnable self-caused change. The infant bands "
                        "are sanity context: deviation is allowed when the goal "
                        "numbers say the deviation earns its keep",
           "reference": "capability-stage matched (reaching_and_batting), not calendar age"}
    json.dump(out, open(os.path.join(SAVED, "motion_census.json"), "w"), indent=1)
    for k, v in out["verdicts"].items():
        print(f"{k}: {v}")
    print(f"written motion_census.json ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
