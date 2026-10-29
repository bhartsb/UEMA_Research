# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Hand-exposure A/B probe: does --goal-gaze-hold change how often the hand is
in view / in the fovea? (2026-09-09: the H67_ah segment stopped on its declared
hand-in-fovea-under-3% rule; every post-reboot window read 1.7-2.9%. Before any
training segment adopts the gaze-hold mechanism, this measures OFFLINE whether
holding the planned gaze target raises exposure at all - measure before demand.)

Both arms: the same checkpoint, the same env seed and world distribution, the
model's normal acting loop with reflex/saccades LIVE (this measures the
integrated system the trainer runs, not an isolated eye assay). The only
difference is the goal_gaze_hold flag. Reports per-arm fractions of frames with
the hand inside the wide view cone and inside the fovea cone, plus how often
the hold was actually engaged, and the head-movement magnitude during held
intervals (the mechanism is HEAD-RELATIVE: if heads move much between plans,
the held eye angles do not track a world target - a measured limitation, not a
hidden one).

python -m mimo_bridge.tools.hand_exposure_probe --ckpt <ckpt_dir> [--steps 600]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SAVED = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge")

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run  # noqa: E402
from mimo_bridge.tools.sensor_action_contract_check import resolved_env_kwargs  # noqa: E402
from mimo_bridge.lem_model import LemLite  # noqa: E402
from mimo_bridge.tools.probe_clock import ProbeClock


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    d = abs(float(np.dot(q0, q1)) /
            max(float(np.linalg.norm(q0) * np.linalg.norm(q1)), 1e-9))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def run_arm(ckpt: str, seed: int, steps: int, hold_flag: str) -> dict:
    live, a, _s, _f = _build_model(ckpt, "cpu", seed, intention=True,
                                   arg_overrides={"goal_gaze_hold": hold_flag})
    limits, _ = _limits_for_run(ckpt, live)
    reps = int(a.get("action_repeat") or 5)
    hold = int(a.get("motor_hold") or 5)
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    ckpt_phase = None
    try:
        cs = json.load(open(os.path.join(ckpt, "meta.json"))).get("curriculum_state") or {}
        ckpt_phase = cs.get("phase")
        if ckpt_phase == "fixed":
            ckpt_phase = None
    except OSError:
        pass
    env_kw, env_phase = resolved_env_kwargs(run_dir, phase_override=ckpt_phase)
    # a DIFFERENT world per CLI seed (09-09 follow-up review: the seed changed
    # model randomness only, so 'three seeds' would have replayed one world)
    env_kw = dict(env_kw, seed=int(env_kw.get("seed") or 0) + 7000 * seed)
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(**env_kw)
    clock = ProbeClock(float(src.env.dt), reps, hold)
    base_noise = float(a.get("babble_noise") or 0.0)
    # the trainer's REST SCHEDULE, reproduced (09-09 follow-up review: the
    # pilot omitted it): bouts of 2-5 s, rests of 1-3 s; during rest the
    # executed command is an active hold of the CURRENT achieved pose
    # (positions x200 cm, achieved quaternion) and exploration noise is off -
    # train_mimo.py's 'babble-bouts real' branch, same arithmetic. The bout
    # RNG is seeded per arm identically, so both arms see the same schedule;
    # it is a fresh draw of the trainer's DISTRIBUTION, not the run's replay.
    bouts_real = str(a.get("babble_bouts") or "off") == "real"
    bout_rng = np.random.default_rng(4200 + seed)
    tsec = clock.model_step_seconds
    bout_on, bout_until = True, 0
    n_fov = n_view = n_frames = holds_set = 0
    head_rot = []           # maximum rotation from start over each full hold
    head_disp = []
    q_hold = h_hold = None
    hold_rotation_max = 0.0
    held_start_time = None
    measured_hold_seconds = []
    schedule = []
    try:
        live.reset_transition_state()
        frames = src.reset()
        held, prev_exec = None, None
        t0 = time.time()
        for i in range(steps):
            if bouts_real and i >= bout_until:
                bout_on = not bout_on
                dur_s = (bout_rng.uniform(2.0, 5.0) if bout_on
                         else bout_rng.uniform(1.0, 3.0))
                bout_until = i + max(1, int(dur_s / tsec))
                schedule.append({'start_step': i, 'active': bout_on,
                                 'model_steps': bout_until - i,
                                 'actual_seconds': (bout_until - i) * tsec})
            dev = dict(limits)
            dev["exploration_noise"] = base_noise if (not bouts_real or bout_on) else 0.0
            with torch.no_grad():
                res = live.step(frames, dev, train=False, step_index=i,
                                executed_action=prev_exec)
            if i % hold == 0 or held is None:
                held = res.motor_for_action
            send = held
            if bouts_real and not bout_on and live.last_achieved is not None:
                send = np.array(held, copy=True)
                _ach = live.last_achieved.detach().float().cpu().numpy()
                n_h = min(send.shape[1], _ach.shape[1])
                send[:, :n_h, 0:3] = _ach[:, :n_h, 0:3] * 200.0
                _q = _ach[:, :n_h, 3:7]
                _qn = np.linalg.norm(_q, axis=-1, keepdims=True)
                _ok = (_qn > 1e-6).squeeze(-1)
                send[:, :n_h, 3:7] = np.where(_ok[..., None],
                                              _q / np.maximum(_qn, 1e-6),
                                              send[:, :n_h, 3:7])
            if getattr(live, "_gaze_hold", None) is not None:
                holds_set += 1
            # HEAD bone is row 12 of the handle/bone order (proprio_adapter
            # BONE_MAP: 12 hand bones first, then "head"); bones cols 0:4 are
            # the achieved world quaternion, 4:7 the position in cm
            if i % hold == 0:
                q_hold = np.array(frames[0].bones[12, 0:4], dtype=np.float64, copy=True)
                h_hold = np.array(frames[0].bones[12, 4:7], dtype=np.float64, copy=True)
                held_start_time = float(frames[0].sim_time)
                hold_rotation_max = 0.0
            step_start = float(frames[0].sim_time)
            gz = []
            for r in range(reps):
                frames = src.step(send, res.gaze_out, None, render=(r == reps - 1))
                hold_rotation_max = max(hold_rotation_max, _quat_angle_deg(
                    q_hold, np.asarray(frames[0].bones[12, 0:4], dtype=np.float64)))
                for f in frames:
                    t = (f.meta.get("task") or {})
                    n_frames += 1
                    n_fov += int(bool(t.get("hand_in_fovea")))
                    n_view += int(bool(t.get("hand_in_view")))
                gz.append(np.stack([np.asarray(f.meta.get("gaze_cmd") or (0.0, 0.0),
                                               dtype=np.float32) for f in frames]))
            clock.verify_elapsed(step_start, float(frames[0].sim_time), tsec)
            if (i + 1) % hold == 0:
                measured_hold_seconds.append(clock.verify_elapsed(
                    held_start_time, float(frames[0].sim_time), clock.held_order_seconds))
                head_rot.append(hold_rotation_max)
                head_disp.append(float(np.linalg.norm(
                    np.asarray(frames[0].bones[12, 4:7], dtype=np.float64) - h_hold)))
            prev_exec = LemLite.gaze_to_handle(send, np.mean(np.stack(gz), axis=0))
        per = (time.time() - t0) / max(1, steps)
    finally:
        src.close()
    hr = np.asarray(head_rot)
    return {"arm": hold_flag, "model_steps": steps, "frames": n_frames,
            "hand_in_fovea_frac": n_fov / max(1, n_frames),
            "hand_in_view_frac": n_view / max(1, n_frames),
            "hold_engaged_frac": holds_set / max(1, steps),
            "rest_schedule_reproduced": bouts_real,
            'physics_step_seconds': clock.physics_step_seconds,
            'model_step_seconds': clock.model_step_seconds,
            'held_order_seconds': clock.held_order_seconds,
            'measured_hold_seconds': measured_hold_seconds,
            'bout_schedule': schedule,
            # Head rotation bounds its contribution to gaze drift; rotation
            # about the viewing axis need not change the gaze direction.
            # Translation, target motion and eye commands also affect gaze.
            "head_rotation_deg_per_interval": {
                "median": float(np.median(hr)) if hr.size else None,
                "p90": float(np.percentile(hr, 90)) if hr.size else None,
                "frac_over_fovea_half_angle_12deg":
                    float((hr > 12.0).mean()) if hr.size else None},
            "head_move_cm_per_interval_median": float(np.median(head_disp)) if head_disp else None,
            "env_seed": env_kw["seed"],
            "env_phase": env_phase, "sec_per_step": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = {"protocol_version": 2, "ckpt_path": os.path.abspath(args.ckpt), "seed": args.seed,
           "source_sha256": __import__("hashlib").sha256(
               open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "arms": [run_arm(args.ckpt, args.seed, args.steps, f)
                    for f in ("off", "on")],
           "note": "hand_in_fovea = hand-body origin within the half-fovea cone "
                   "of either eye axis (geometric proxy: no occlusion or pixel "
                   "check). Same checkpoint, same env seed, same reproduced "
                   "rest schedule, reflex/saccades live in both arms; the flag "
                   "is the only difference. Head rotation is measured across the "
                   "full hold and does not by itself measure fixation error. One CLI seed = "
                   "one world; vary --seed for different worlds. A pilot "
                   "reading, not a verdict."}
    os.makedirs(os.path.join(SAVED, "hand_exposure"), exist_ok=True)
    p = os.path.join(SAVED, "hand_exposure",
                     f"{os.path.basename(os.path.normpath(args.ckpt))}"
                     f"_s{args.seed}_gazehold_v2.json")
    tmp = p + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    print(json.dumps(out, indent=1))
    print("written:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
