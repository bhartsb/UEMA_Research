# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Who is batting the mobile - the planner or the babble? (operator-ordered 2026-09-04)

Behaviorally, striking is established: nine consecutive active hours surviving a
world reset. This measures the STRICT claim - that the planner's chosen orders,
not the exploration noise under them, produce the strikes - by running the same
checkpoint through the live acting pathway in three arms on identical worlds:

  normal      planner on,  babble at the run's setting   (the live configuration)
  no_babble   planner on,  babble 0                      (pure chosen orders)
  no_planner  planner off, babble at the run's setting   (the H23-era floor)

Each arm: a fresh environment with the SAME seed, N model steps, orders latched
and executed exactly as in training (motor_hold x action_repeat). Strikes are
read from the environment's own task flags (mobile_touched_hand / mobile_touched),
sampled two ways per model step: the last env frame (parity with the live stats)
and any-frame-within-the-step (catches within-order strikes).

Reading: no_babble holding near normal while no_planner collapses = the strikes
are the planner's. no_babble collapsing toward no_planner = the babble's.
Training is never touched: this runs on CPU against a checkpoint, like the
situation test.

Usage: python -m mimo_bridge.tools.strike_attribution <ckpt> [n_steps] [out_json]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch


def _arm(ckpt: str, a, limits, intention: bool, babble: float, n_steps: int,
         probe_dir: str, seed_env: int = 71):
    from mimo_bridge.tools.causal_test import _build_model
    from mimo_bridge.datasource import SingleEnvSource

    model, _a, _sp, _f = _build_model(ckpt, "cpu", seed=0, intention=intention)
    model.load_checkpoint(ckpt, load_optimizer=False)
    hold = int(a.get("motor_hold") or 5)
    reps = int(a.get("action_repeat") or 5)
    dev = dict(limits)
    dev["exploration_noise"] = float(babble)

    kw = dict(scene=str(a.get("scene") or "crib"), run_dir=probe_dir, vision=True,
              fovea_size=int(a.get("fovea_size") or 288),
              fovea_fov=float(a.get("fovea_fov") or 24.0), seed=seed_env,
              tag="_strike", mobile=(a.get("mobile") == "on"),
              caregiver_seconds=float(a.get("caregiver_seconds") or 0.0))
    for sk, ek in (("decoder_kp", "kp"), ("joint_damping_scale", "joint_damping_scale"),
                   ("hinge_armature", "hinge_armature"), ("finger_armature", "finger_armature"),
                   ("gaze_kp", "gaze_kp"), ("gaze_kd", "gaze_kd")):
        if a.get(sk) is not None:
            kw[ek] = a[sk]
    if limits.get("age_months") is not None:
        kw["age"] = float(limits["age_months"])
    src = SingleEnvSource(**kw)

    last_hand = any_hand = any_touch = 0
    speeds = []
    try:
        frames = src.reset()
        model.reset_context(1)
        model.reset_transition_state()
        held = prev = None
        order_seq, prev_oid = 0, None
        for i in range(n_steps):
            res = model.step(frames, dev, train=False, step_index=i,
                             executed_action=prev, executed_order_id=prev_oid)
            if i % hold == 0 or held is None:
                held = res.motor_for_action
                order_seq += 1
            step_hand = step_touch = False
            for _ in range(reps):
                frames = src.step(held, res.gaze_out, None, render=True)
                task = (frames[0].meta or {}).get("task", {})
                step_hand = step_hand or bool(task.get("mobile_touched_hand"))
                step_touch = step_touch or bool(task.get("mobile_touched"))
                sp = task.get("mobile_speed_cms")
                if sp is not None:
                    speeds.append(float(sp))
            last_hand += int(bool(task.get("mobile_touched_hand")))
            any_hand += int(step_hand)
            any_touch += int(step_touch)
    finally:
        src.close()
    return {"intention": intention, "babble": babble, "model_steps": n_steps,
            "orders": order_seq,
            "hand_strike_rate_lastframe": last_hand / n_steps,
            "hand_strike_rate_anyframe": any_hand / n_steps,
            "any_touch_rate_anyframe": any_touch / n_steps,
            "mobile_speed_mean_cms": float(np.mean(speeds)) if speeds else None,
            "mobile_speed_max_cms": float(np.max(speeds)) if speeds else None}


def reading(normal_rate: float, no_babble_rate: float, floor_rate: float) -> str:
    """The attribution rule, pure so it can be tested: planner-driven when the
    chosen orders alone keep at least half the full rate AND the full rate
    clears twice the babble-only floor; babble-driven when removing babble
    collapses the rate below a quarter; otherwise the seed was not enough."""
    if no_babble_rate >= 0.5 * normal_rate and normal_rate > 2 * max(floor_rate, 1e-9):
        return "planner-driven"
    if no_babble_rate < 0.25 * normal_rate:
        return "babble-driven"
    return "mixed-or-underpowered"


def run(ckpt: str, n_steps: int = 3000, out_json: str | None = None,
        seed_env: int = 71):
    torch.set_num_threads(2)
    from mimo_bridge.tools.causal_test import _build_model, _limits_for_run

    model, a, _sp, _f = _build_model(ckpt, "cpu", seed=0, intention=False)
    model.load_checkpoint(ckpt, load_optimizer=False)
    limits, _src = _limits_for_run(ckpt, model)
    del model
    babble = float(a.get("babble_noise") or 0.05)
    probe_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))),
                             "strike_probe")

    arms = {}
    for name, intent, bab in (("normal", True, babble),
                              ("no_babble", True, 0.0),
                              ("no_planner", False, babble)):
        t0 = time.time()
        arms[name] = _arm(ckpt, a, limits, intent, bab, n_steps, probe_dir,
                          seed_env=seed_env)
        arms[name]["wall_min"] = round((time.time() - t0) / 60, 1)
        print(f"arm {name}: {json.dumps(arms[name])}", flush=True)

    nm, nb, np_ = arms["normal"], arms["no_babble"], arms["no_planner"]
    key = "hand_strike_rate_anyframe"
    verdict = {
        "planner_share_reading": reading(nm[key], nb[key], np_[key]),
        "note": "rates are per model step, same env seed all arms; one seed - "
                "treat a close call as underpowered, not decided"}
    out = {"ckpt": ckpt, "when": time.strftime("%Y-%m-%d %H:%M"),
           "n_steps_per_arm": n_steps, "env_seed": seed_env,
           "arms": arms, "verdict": verdict}
    dest = out_json or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))),
                                    "strike_attribution.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(json.dumps(out["verdict"], indent=1))
    print("->", dest)
    return out


if __name__ == "__main__":
    run(sys.argv[1],
        n_steps=int(sys.argv[2]) if len(sys.argv) > 2 else 3000,
        out_json=sys.argv[3] if len(sys.argv) > 3 else None,
        seed_env=int(sys.argv[4]) if len(sys.argv) > 4 else 71)
