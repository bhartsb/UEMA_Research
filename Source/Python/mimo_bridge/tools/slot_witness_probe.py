# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Slot-witness probe (q5, declared 2026-09-06): do the dark self-model's live slots
track NAMEABLE situations, or is the concentration degenerate clustering?

Method, on a frozen checkpoint (no training, tracker observes exactly as live):
one rollout under the run's own developmental limits and babble noise, with
SCHEDULED rest windows (4 s move / 2 s pose-hold rest, the trainer's own rest
override code) so "stillness" exists as a labeled situation. Each model step
records the slot the tracker assigns the moment to (argmax of cosine similarity
between the fused raw tokens and the slot centers - the same fusion observe()
computes) beside env-labeled events:
  rest_window        scheduled (ground truth by construction)
  bee_wide/_fovea    bee inside the 60-degree wide cone / the 24-degree fovea
                     (structural half-angles, not chosen numbers)
  dragonfly_hover    the pre-dart hover tell
  mobile_touch(_hand) contact with the mobile (any part / hand)
  hand_fovea         own hand inside the fovea
  squeak             squeaker hit count increased this step
  reflex_snap        the endowed gaze reflex fired

Verdict statistic per event: the best slot's LIFT = P(winner=slot | event) /
P(winner=slot), with a circular-shift control (the same statistic on the event
track shifted by random offsets - preserves both series' autocorrelation; the
95th percentile of shifted lifts is the chance line). An event whose real lift
clears its own chance line has a slot witness; report coverage too (share of
event steps that slot wins). Events with under 30 steps are UNMEASURED, stated.

Writes Saved/mimo_bridge/slot_witness/<ckpt>_s<seed>.json and prints a summary.
python -m mimo_bridge.tools.slot_witness_probe --ckpt <ckpt_dir> [--steps 4500]
    [--seed 0] [--warm 100] [--shifts 500]
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
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SAVED = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge")

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run  # noqa: E402
from mimo_bridge.lem_model import LemLite  # noqa: E402


def fuse_winner(model):
    """The moment's slot, computed the way HierarchicalSelfDark.observe fuses:
    per-modality raw tokens (zeros for a missing sense), L2-normalized concat,
    cosine against the filled slot centers, argmax. Returns (winner, top_sim)."""
    hs = model._hself
    toks = model.last_tokens
    ref = next(iter(toks.values()))
    B = ref.shape[0]
    parts = []
    for m in hs.modalities:
        v = toks.get(m)
        parts.append(v.detach().float().reshape(B, -1)
                     if v is not None and v.shape[0] == B
                     else torch.zeros(B, hs.d_model))
    x = F.normalize(torch.cat(parts, dim=-1), dim=-1)
    k = int(hs._filled_cpu)
    sim = x @ hs.centers[:k].T.cpu()
    w = int(sim.argmax(dim=-1)[0])
    return w, float(sim[0, w])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=int, default=100)
    ap.add_argument("--shifts", type=int, default=500)
    args = ap.parse_args(argv)

    model, a, _s, _f = _build_model(args.ckpt, "cpu", args.seed, intention=True)
    if a.get("audio") == "real":
        model.config["modalities"].setdefault("audio", {})["data_source"] = "real"
    assert model._hself is not None, (
        "no dark self-model on this checkpoint/build - q5 cannot be probed here")
    _ape = getattr(model, "_agency_pay_ema", None)
    if _ape is not None and torch.is_tensor(_ape) and _ape.numel() != 1:
        model._agency_pay_ema = _ape.mean().reshape(1)
    if a.get("intention_every") is not None:
        model.intention_every = int(a["intention_every"])
    limits, _ = _limits_for_run(args.ckpt, model)
    from mimo_bridge.datasource import SingleEnvSource
    reps = int(a.get("action_repeat") or 5)
    hold = int(a.get("motor_hold") or 5)
    env_kw = dict(scene="crib", run_dir=tempfile.mkdtemp(prefix="slotwit_"), vision=True,
                  fovea_size=int(a.get("fovea_size") or 288),
                  fovea_fov=float(a.get("fovea_fov") or 24.0))
    for _sk, _ek in (("decoder_kp", "kp"), ("joint_damping_scale", "joint_damping_scale"),
                     ("hinge_armature", "hinge_armature"), ("finger_armature", "finger_armature"), ("touch_scale_mult", "touch_scale_mult"),
                     ("caregiver_seconds", "caregiver_seconds"), ("gaze_kp", "gaze_kp"),
                     ("gaze_kd", "gaze_kd"), ("ring_fov", "ring_fov"), ("ring_size", "ring_size")):
        if a.get(_sk) is not None:
            env_kw[_ek] = a[_sk]
    env_kw["cmd_horizon"] = reps * hold
    if a.get("gaze_reflex") is not None:
        env_kw["gaze_reflex"] = a.get("gaze_reflex") == "on"
    if a.get("spontaneous_saccades") is not None:
        env_kw["spontaneous_saccades"] = a.get("spontaneous_saccades") != "off"
    if limits.get("age_months") is not None:
        env_kw["age"] = float(limits["age_months"])
    if a.get("audio") is not None:
        env_kw["audio_real"] = a.get("audio") == "real"
    for k, on in (("critters", "critters"), ("mobile", "mobile")):
        if a.get(on) is not None:
            env_kw[k] = a.get(on) == "on"
    src = SingleEnvSource(tag="_slotwit", seed=170 + args.seed, **env_kw)

    wide_half = 30.0   # the wide eye camera covers a fixed 60-degree cone
    fovea_half = float(a.get("fovea_fov") or 24.0) / 2.0
    BOUT_STEPS, REST_STEPS = 80, 40      # 4 s move / 2 s rest at 50 ms per model step
    winners, sims, resting_track = [], [], []
    ev = {k: [] for k in ("bee_wide", "bee_fovea", "dragonfly_hover", "mobile_touch",
                          "mobile_touch_hand", "hand_fovea", "squeak", "reflex_snap")}
    prev_counts = {"squeaker_hits": None, "reflex_snaps": None}
    t0 = time.time()
    try:
        model.reset_transition_state()
        frames = src.reset()
        dev = dict(limits)
        dev["exploration_noise"] = float(a.get("babble_noise") or 0.0)
        held, prev_exec = None, None
        for i in range(args.steps):
            resting = (i % (BOUT_STEPS + REST_STEPS)) >= BOUT_STEPS
            with torch.no_grad():
                res = model.step(frames, dev, train=False, step_index=i,
                                 executed_action=prev_exec)
            if i == 20:
                per = (time.time() - t0) / 21.0
                print(f"[timing] {per:.2f}s per step - projected "
                      f"{per * args.steps / 60.0:.0f} min for --steps {args.steps}",
                      flush=True)
            w, s = fuse_winner(model)
            winners.append(w)
            sims.append(s)
            resting_track.append(resting)
            if i % hold == 0 or held is None:
                held = res.motor_for_action
            send = held
            if resting and model.last_achieved is not None:
                # the trainer's own rest override (train_mimo babble-bouts real):
                # target = current achieved pose, model's own strength, quats normalized
                send = np.array(held, copy=True)
                ach = model.last_achieved.detach().float().cpu().numpy()
                n_h = min(send.shape[1], ach.shape[1])
                send[:, :n_h, 0:3] = ach[:, :n_h, 0:3]
                q = ach[:, :n_h, 3:7]
                qn = np.linalg.norm(q, axis=-1, keepdims=True)
                ok = (qn > 1e-6).squeeze(-1)
                send[:, :n_h, 3:7] = np.where(ok[..., None], q / np.maximum(qn, 1e-6),
                                              send[:, :n_h, 3:7])
            burst_tasks, gz_burst = [], []
            for rep in range(reps):
                frames = src.step(send, res.gaze_out, None, render=(rep == reps - 1))
                burst_tasks.append(dict(frames[0].meta.get("task") or {}))
                gz_burst.append(np.stack([np.asarray(
                    f.meta.get("gaze_cmd") or (0.0, 0.0), dtype=np.float32)
                    for f in frames]))
            prev_exec = LemLite.gaze_to_handle(send, np.mean(np.stack(gz_burst), axis=0))
            # events over the burst: OR for booleans, min for angles, diff for counters
            bee = [t.get("bee_deg") for t in burst_tasks if t.get("bee_deg") is not None]
            bee = min(bee) if bee else None
            ev["bee_wide"].append(bee is not None and bee < wide_half)
            ev["bee_fovea"].append(bee is not None and bee < fovea_half)
            ev["dragonfly_hover"].append(any(t.get("dragonfly_hovering") for t in burst_tasks))
            ev["mobile_touch"].append(any(t.get("mobile_touched") for t in burst_tasks))
            ev["mobile_touch_hand"].append(any(t.get("mobile_touched_hand") for t in burst_tasks))
            ev["hand_fovea"].append(any(t.get("hand_in_fovea") for t in burst_tasks))
            for key, evk in (("squeaker_hits", "squeak"), ("reflex_snaps", "reflex_snap")):
                cur = burst_tasks[-1].get(key)
                prev = prev_counts[key]
                ev[evk].append(cur is not None and prev is not None and cur > prev)
                prev_counts[key] = cur if cur is not None else prev
    finally:
        src.close()

    n = len(winners)
    W = np.asarray(winners[args.warm:])
    R = np.asarray(resting_track[args.warm:], dtype=bool)
    events = {k: np.asarray(v[args.warm:], dtype=bool) for k, v in ev.items()}
    events["rest_window"] = R
    m = len(W)
    base = np.bincount(W, minlength=64).astype(float) / m
    active = [int(i) for i in np.nonzero(base >= 0.01)[0]]
    rng = np.random.default_rng(4242 + args.seed)

    def best_lift(track: np.ndarray):
        idx = np.nonzero(track)[0]
        if len(idx) == 0:
            return None
        cnt = np.bincount(W[idx], minlength=64).astype(float) / len(idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            lifts = np.where(base > 0, cnt / np.maximum(base, 1e-12), 0.0)
        s = int(lifts.argmax())
        return {"slot": s, "lift": float(lifts[s]), "coverage": float(cnt[s]),
                "base": float(base[s])}

    out_events = {}
    for name, track in events.items():
        cnt = int(track.sum())
        if cnt < 30:
            out_events[name] = {"steps": cnt, "verdict": "UNMEASURED - under 30 steps "
                                "in this probe window"}
            continue
        real = best_lift(track)
        shifted = []
        for _ in range(args.shifts):
            off = int(rng.integers(args.warm, m - args.warm))
            r = best_lift(np.roll(track, off))
            if r is not None:
                shifted.append(r["lift"])
        chance95 = float(np.percentile(shifted, 95)) if shifted else None
        p = (float(np.mean([s >= real["lift"] for s in shifted]))
             if shifted else None)
        out_events[name] = {"steps": cnt, **real, "chance95_lift": chance95,
                            "p_shift": p,
                            "witness": bool(chance95 is not None
                                            and real["lift"] > chance95)}
    top_share = float(base.max())
    ent = float(-(base[base > 0] * np.log(base[base > 0])).sum() / np.log(64))
    result = {
        "ckpt": os.path.basename(os.path.normpath(args.ckpt)), "seed": args.seed,
        "steps": n, "warm_excluded": args.warm, "shifts": args.shifts,
        "sim_minutes": round(n * 0.05 / 60.0, 2),
        "slots_active_1pct": len(active), "active_slots": active,
        "top_slot_share": round(top_share, 4), "occupancy_entropy_norm": round(ent, 4),
        "events": out_events,
        "note": "lift = P(winner=slot|event)/P(winner=slot); chance line = 95th "
                "percentile of the same statistic under circular time-shifts",
    }
    os.makedirs(os.path.join(SAVED, "slot_witness"), exist_ok=True)
    outp = os.path.join(SAVED, "slot_witness",
                        f"{result['ckpt']}_s{args.seed}.json")
    tmp = outp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1, allow_nan=False)
    os.replace(tmp, outp)
    print(json.dumps(result, indent=1))
    print(f"written: {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
