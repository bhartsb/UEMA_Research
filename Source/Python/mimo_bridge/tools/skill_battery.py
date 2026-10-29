# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Skill battery v2: behavioral comparison of frozen checkpoints on the
REACH-THE-SENSED-TARGET task (rebuilt 2026-09-09 after the battery audit).

The audited v1 defects, each repaired here:
  object mismatch      v1 counted contact with any `mobile_*` geometry but called
                       it "sighted" via gaze at the RED TARGET - different
                       objects. v2's primary task uses the red target for BOTH:
                       gaze_on_ball paired with ball_touched_hand. Contact with
                       the hanging mobile structure is kept as a separate,
                       clearly-labeled exploratory column.
  sampling             v1 read the contact flag once per action burst (final
                       frame) - contacts inside a burst vanished. v2 reads every
                       env frame and scores contact EVENTS with a declared
                       release separation (RELEASE_GAP frames contact-free ends
                       an event), so chatter is one event, not many.
  world independence   v1 reused one env across episodes; behavior-dependent
                       random draws let episodes diverge between models. v2
                       rebuilds the env per episode with a recorded per-episode
                       seed and records the initial target placement.
  model state          v1 reused one model object across both blocks; body-schema
                       and horizon statistics carried over. v2 rebuilds the model
                       from the checkpoint per (checkpoint, block); within-block
                       test-time adaptation is part of the evaluated system and
                       is declared in the artifact.
  replay honesty       the replay check is labeled as what it is: first-50-command
                       wrist reproducibility within 0.5 cm plus contact-flag
                       sequence agreement; the env is restored to its pre-replay
                       terminal state afterward.
  wording              the 3x-SEM rule is a CHOSEN decision rule on sampling
                       spread, not an instrument resolution; no comparison here
                       proves equivalence, retention, or transfer. Policy is
                       each checkpoint's own historical settings, so differences
                       confound weights with policy (a fixed-policy mode is the
                       declared next step, not silently pretended).

Still not measured (stated): occlusion-checked seeing, selected-goal identity
inside the model, obstacle/goal-switch correction, outcome devaluation. Those
need the acceptance-task design in the 09-09 battery audit.

python -m mimo_bridge.tools.skill_battery --ckpts <old> [<mid> ...] <new>
    [--episodes 6] [--steps 500] [--train-seed 100] [--transfer-seed 500]
"""
from __future__ import annotations

import argparse
import hashlib
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
from mimo_bridge.tools.sensor_action_contract_check import (  # noqa: E402
    resolved_env_kwargs, free_ram_gb)
from mimo_bridge.lem_model import LemLite  # noqa: E402

RIGHT_WRIST = 6
SIGHT_WINDOW = 25    # model steps: a touch this soon after gaze-on-target is "sighted"
RELEASE_GAP = 10     # env frames without contact that END a contact event
POLICY_KEYS = ("babble_noise", "babble_bouts", "intention", "intention_every",
               "imagination_steps", "curiosity_goal", "body_schema_updates",
               "goal_locality", "hierarchical_self", "plan_horizon")


def _ckpt_phase(ckpt: str):
    try:
        cs = json.load(open(os.path.join(ckpt, "meta.json"))).get("curriculum_state") or {}
        ph = cs.get("phase")
        return None if ph == "fixed" else ph
    except OSError:
        return None


class ContactEvents:
    """Rising-edge event scorer with a release separation, fed EVERY env frame."""

    def __init__(self, release_gap: int = RELEASE_GAP):
        self.events = []          # [onset_frame, duration_frames]
        self._open = None
        self._gap = 0
        self.release_gap = release_gap

    def feed(self, frame: int, touched: bool):
        if touched:
            if self._open is None:
                self._open = [frame, 1]
            else:
                self._open[1] += 1
            self._gap = 0
        elif self._open is not None:
            self._gap += 1
            if self._gap >= self.release_gap:
                self.events.append(tuple(self._open))
                self._open = None

    def close(self):
        if self._open is not None:
            self.events.append(tuple(self._open))
            self._open = None
        return self.events


def run_episode(live, src, limits, a, steps, replay_check=False):
    """One frozen-model episode. Contact flags are read from EVERY env frame."""
    reps = int(a.get("action_repeat") or 5)
    hold = int(a.get("motor_hold") or 5)
    dev = dict(limits)
    dev["exploration_noise"] = float(a.get("babble_noise") or 0.0)
    live.reset_transition_state()
    frames = src.reset()
    ball_ev, mobile_ev = ContactEvents(), ContactEvents()
    counts = dict(gaze_steps=0, sighted_ball_touches=0,
                  first_ball_contact_env_frame=None)
    target0 = None
    env0 = getattr(src, "env", None)
    if env0 is not None and getattr(env0, "_target_id", None) is not None:
        target0 = [round(float(v), 4) for v in env0.data.mocap_pos[env0._target_id]]
    last_gaze_step = -10**9
    prev_ball = False
    held, prev_exec = None, None
    snap, recorded, wrist_track, ball_track = None, [], [], []
    env_frame = 0
    if replay_check:
        snap = src.env.save_full_state()
    for i in range(steps):
        with torch.no_grad():
            res = live.step(frames, dev, train=False, step_index=i,
                            executed_action=prev_exec)
        if i % hold == 0 or held is None:
            held = res.motor_for_action
        gz = []
        for r in range(reps):
            frames = src.step(held, res.gaze_out, None, render=(r == reps - 1))
            gz.append(np.stack([np.asarray(f.meta.get("gaze_cmd") or (0.0, 0.0),
                                           dtype=np.float32) for f in frames]))
            # EVERY frame is scored (v1 sampled only the burst's final frame)
            task = (frames[0].meta or {}).get("task") or {}
            ball = bool(task.get("ball_touched_hand"))
            mobile = bool(task.get("mobile_touched_hand"))
            ball_ev.feed(env_frame, ball)
            mobile_ev.feed(env_frame, mobile)
            if ball and counts["first_ball_contact_env_frame"] is None:
                counts["first_ball_contact_env_frame"] = env_frame
            if ball and not prev_ball and i - last_gaze_step <= SIGHT_WINDOW:
                counts["sighted_ball_touches"] += 1
            prev_ball = ball
            env_frame += 1
        task = (frames[0].meta or {}).get("task") or {}
        if task.get("gaze_on_ball"):
            counts["gaze_steps"] += 1
            last_gaze_step = i
        prev_exec = LemLite.gaze_to_handle(held, np.mean(np.stack(gz), axis=0))
        if replay_check and i < 50:
            recorded.append((np.array(held, copy=True),
                             None if res.gaze_out is None else np.array(res.gaze_out)))
            wrist_track.append(frames[0].bones[RIGHT_WRIST, 4:7].copy())
            ball_track.append(bool(task.get("ball_touched_hand")))
    counts["ball_contact_events"] = ball_ev.close()
    counts["mobile_contact_events"] = mobile_ev.close()
    counts["ball_touch_onsets"] = len(counts["ball_contact_events"])
    counts["mobile_touch_onsets"] = len(counts["mobile_contact_events"])
    counts["initial_target_pos"] = target0
    if replay_check and snap is not None:
        end_state = src.env.save_full_state()
        src.env.restore_full_state(snap)
        dev_cm, ball_match = 0.0, True
        for k, (act, g) in enumerate(recorded):
            for r in range(reps):
                f2 = src.step(act, g, None, render=(r == reps - 1))
            dev_cm = max(dev_cm, float(np.linalg.norm(
                f2[0].bones[RIGHT_WRIST, 4:7] - wrist_track[k])))
            t2 = (f2[0].meta or {}).get("task") or {}
            if bool(t2.get("ball_touched_hand")) != ball_track[k]:
                ball_match = False
        src.env.restore_full_state(end_state)   # audit: never leave the env mid-prefix
        counts["replay_wrist_max_dev_cm"] = round(dev_cm, 3)
        counts["replay_wrist_ok"] = bool(dev_cm < 0.5)
        counts["replay_ball_flags_match"] = ball_match
    return counts


def _compare(name, better: np.ndarray, worse: np.ndarray) -> dict:
    """Per-episode difference under a CHOSEN 3x-SEM decision rule (audit: this is
    a sampling-spread rule, not an instrument resolution; systematic bias will
    not appear in it, and no result here proves equivalence or retention)."""
    diff = better.astype(float) - worse.astype(float)
    n = len(diff)
    if n < 3:
        return {"comparison": name, "verdict": "not_evaluated",
                "reason": f"{n} episodes; needs >= 3"}
    sem = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    mean = float(diff.mean())
    line = 3.0 * sem
    if sem == 0.0:
        v = "no_difference_measured" if mean == 0.0 else "difference_detected"
    else:
        v = "difference_detected" if abs(mean) > line else \
            "no_difference_above_decision_rule"
    return {"comparison": name, "verdict": v,
            "mean_ball_onsets_diff_per_episode": round(mean, 3),
            "sem": round(sem, 4), "decision_rule_3sem": round(line, 3),
            "a_total": int(better.sum()), "b_total": int(worse.sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="checkpoint dirs, OLDEST FIRST")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--train-seed", type=int, default=100)
    ap.add_argument("--transfer-seed", type=int, default=500)
    args = ap.parse_args(argv)
    if len(args.ckpts) < 2:
        ap.error("needs at least two checkpoints (oldest first)")
    if free_ram_gb() < 6.0:
        print(f"REFUSED: {free_ram_gb():.1f} GB free RAM < 6.0 floor")
        return 2

    ref_ckpt = args.ckpts[-1]
    ref_run = os.path.dirname(os.path.dirname(os.path.abspath(ref_ckpt)))
    env_kw, env_phase = resolved_env_kwargs(ref_run, phase_override=_ckpt_phase(ref_ckpt))
    from mimo_bridge.datasource import SingleEnvSource

    out = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "battery_version": 2,
           "task": "reach the RED TARGET (gaze and touch refer to the same object); "
                   "contact events scored every env frame with release gap "
                   f"{RELEASE_GAP} frames; mobile-structure contact kept as a "
                   "separate exploratory column",
           "episodes": args.episodes, "steps_per_episode": args.steps,
           "train_seed": args.train_seed, "transfer_seed": args.transfer_seed,
           "curriculum_phase": env_phase,
           "resolved_env_kwargs": {k: v for k, v in env_kw.items() if k != "run_dir"},
           "policy_note": "each checkpoint runs its OWN historical policy settings "
                          "(recorded per checkpoint below) - differences confound "
                          "weights with policy; a fixed-policy mode is the declared "
                          "next instrument step",
           "adaptation_note": "the model is rebuilt from the checkpoint per "
                              "(checkpoint, block); body-schema/horizon statistics "
                              "may adapt WITHIN a block - declared test-time "
                              "adaptation, reset at each block boundary",
           "checkpoints": [], "rows": [],
           "source_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb")
                                           .read()).hexdigest()}

    onsets = {}
    for ci, ckpt in enumerate(args.ckpts):
        step_n = None
        if os.path.exists(os.path.join(ckpt, "meta.json")):
            step_n = json.load(open(os.path.join(ckpt, "meta.json"))).get("step")
        for block, seed in (("trained", args.train_seed), ("transfer", args.transfer_seed)):
            # fresh model per block: block boundaries reset test-time adaptation
            live, a, _s, _f = _build_model(ckpt, "cpu", 0, intention=True)
            limits, _ = _limits_for_run(ckpt, live)
            if ci == 0 and block == "trained" or not any(
                    c["path"] == os.path.abspath(ckpt) for c in out["checkpoints"]):
                out["checkpoints"].append({"path": os.path.abspath(ckpt), "step": step_n,
                                           "policy": {k: a.get(k) for k in POLICY_KEYS}})
            ep_counts = []
            for e in range(args.episodes):
                ep_seed = seed + 1000 * e     # independent recorded world per episode
                kw = dict(env_kw, seed=ep_seed)
                src = SingleEnvSource(**kw)
                try:
                    c = run_episode(live, src, limits, a, args.steps,
                                    replay_check=(e == 0))
                finally:
                    src.close()
                c.update(ckpt=os.path.basename(os.path.normpath(ckpt)),
                         block=block, episode=e, episode_seed=ep_seed)
                out["rows"].append(c)
                ep_counts.append(c["ball_touch_onsets"])
                print(f"{c['ckpt']} {block} ep{e} (seed {ep_seed}): ball_onsets "
                      f"{c['ball_touch_onsets']} sighted {c['sighted_ball_touches']} "
                      f"first_contact {c['first_ball_contact_env_frame']} "
                      f"gaze {c['gaze_steps']} mobile_onsets {c['mobile_touch_onsets']}"
                      + (f" replay_dev {c.get('replay_wrist_max_dev_cm')}cm"
                         if e == 0 else ""), flush=True)
            onsets[(ci, block)] = np.array(ep_counts)

    last = len(args.ckpts) - 1
    comps = [_compare("newest vs oldest, first seed block",
                      onsets[(last, "trained")], onsets[(0, "trained")])]
    for mid in range(1, last):
        comps.append(_compare(f"newest vs checkpoint {mid}, first seed block",
                              onsets[(last, "trained")], onsets[(mid, "trained")]))
    comps.append(_compare("newest, second vs first seed block",
                          onsets[(last, "transfer")], onsets[(last, "trained")]))
    comps.append(_compare("newest vs oldest, second seed block",
                          onsets[(last, "transfer")], onsets[(0, "transfer")]))
    out["comparisons"] = comps
    rr = [r for r in out["rows"] if "replay_wrist_ok" in r]
    out["replay_note"] = {
        "wrist_ok_all": bool(rr) and all(r["replay_wrist_ok"] for r in rr),
        "ball_flags_match_all": bool(rr) and all(r.get("replay_ball_flags_match")
                                                 for r in rr),
        "meaning": "first-50-command wrist reproducibility within 0.5 cm plus "
                   "target-contact flag agreement on those prefixes - NOT full "
                   "episode, policy, or all-sense reproducibility"}
    acq = comps[0]
    out["verdict"] = (
        "exploratory contact-rate comparison: " + acq["verdict"]
        + (f" (newest-minus-oldest {acq.get('mean_ball_onsets_diff_per_episode')} "
           f"target-touch events/episode against the chosen decision rule "
           f"{acq.get('decision_rule_3sem')})" if "sem" in acq else "")
        + "; no comparison here proves acquisition, equivalence, retention, or "
          "transfer - see battery audit for the acceptance-task design")

    os.makedirs(os.path.join(SAVED, "skill_battery"), exist_ok=True)
    p = os.path.join(SAVED, "skill_battery",
                     time.strftime("battery_v2_%Y%m%d_%H%M%S") + ".json")
    tmp = p + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, allow_nan=False)
    os.replace(tmp, p)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
