# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""World-model battery: three questions in the operator's own terms, measured.

  A. "When I move my hand I know what I'll see" - how much better is the next VIEW
     predicted with the model's own executed order than with another moment's order?
     (correct-vs-shuffled visual prediction error, the causal test's vision column,
     reported as a percentage.)
  B. "When I touch something I anticipate the feel" - on the order BEFORE a real
     touch event, does predicted touch-change rise above quiet-moment baseline?
     (predicted tactile delta on pre-event orders / on quiet orders.)
  C. "I can look away and look back and things are consistent" - gaze is driven
     away for K seconds and returned; the model's prediction for the returned view
     is scored against the TRUE returned view vs the view it was just seeing.
     advantage(K) > 0 = it re-paints what is back there rather than assuming the
     world is wherever it last looked. Swept across K inside and beyond the 16 s
     context window.

Runs on a checkpoint at segment boundaries (trainer down). Writes JSON + three
plain sentences. python -m mimo_bridge.tools.world_model_battery --ckpt <dir>

Known limit of C (stated, not hidden): the prediction is scored mid-swing, so a
positive advantage is memory PLUS swing extrapolation (the eye is sweeping A-ward
and continuation alone points that way). The decay of the advantage across K -
especially past the 64-order context - is the informative curve. A changed-while-
away variant (move an object at A during the away phase; does the prediction match
the OLD content?) would isolate content memory; not built yet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimo_bridge.tools.causal_test import (_build_model, _limits_for_run,  # noqa: E402
                                           _collect, _predict)


def _cos(a, b):
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def probe_world_kwargs(a, limits, run_dir, *, scene, seed, tag, overrides=None):
    """The world the checkpoint trained in, built by the trainer's own
    train_mimo.env_kwargs_for, with only the battery's own settings changed.

    Until 2026-09-23 both parts built this world by hand from a dozen args.json keys and
    never passed body_layout or contact_profile. Every H84 segment trains the trunk_v1
    body (26 handle rows, the proprioception encoder takes 26 x 12 = 312 numbers), but
    the hand-built world was the env's default 24-row legacy body (288 numbers) with the
    legacy contact profile, so the first model step on an H84 checkpoint raised and the
    battery could not measure the H84 lineage (fix-rows-r1-world_model_battery.py).

    `a` is the trainer's argument namespace as a dict (causal_test._build_model returns
    it); any key it lacks takes the trainer's default. The battery's own settings: the
    scene, a scratch run_dir, the seed and tag, and never recording or replaying the
    caregiver (the run's caregiver_record_dir is its live recording store)."""
    from mimo_bridge import train_mimo
    ns = train_mimo.parse_args(["--out", run_dir])
    vars(ns).update(a or {})
    # the phase the limits were logged under; 'fixed' (no curriculum) or an unknown
    # name maps to None, as sensor_action_contract_check.latest_phase does
    phase = limits.get("phase_name")
    phase = phase if phase in train_mimo.PHASE_AGE else None
    kw = train_mimo.env_kwargs_for(ns, phase, run_dir)
    kw.update(scene=scene, run_dir=run_dir, seed=int(seed), tag=tag,
              caregiver_record_dir="", caregiver_replay="")
    if limits.get("age_months") is not None and "age" not in kw:
        kw["age"] = float(limits["age_months"])
    kw.update(overrides or {})
    return kw


def part_a_and_b(model, args, a, limits):
    """One collection stream serves both A (vision, correct vs shuffled order) and
    B (touch anticipation on pre-event orders)."""
    from mimo_bridge.datasource import SingleEnvSource
    import tempfile
    env_kw = probe_world_kwargs(a, limits, tempfile.mkdtemp(prefix="wmb_"),
                                scene=args.scene, seed=140, tag="_wmb")
    src = SingleEnvSource(**env_kw)
    try:
        trs = _collect(model, src, args.n, int(a.get("motor_hold") or 5),
                       limits=dict(limits), babble=float(a.get("babble_noise") or 0.05))
    finally:
        src.close()
    if len(trs) < 40:
        raise SystemExit(f"only {len(trs)} transitions")

    g = torch.Generator().manual_seed(9)
    vis_correct, vis_shuffled = [], []
    tac_pred_delta, tac_true_delta = [], []
    for tr in trs:
        pred_c = _predict(model, tr, tr["act"], args.device)
        j = int(torch.randint(len(trs), (1,), generator=g))
        pred_s = _predict(model, tr, trs[j]["act"], args.device)
        if "vision" in pred_c:
            t1 = tr["t1"]["vision"].float()
            vis_correct.append(float(F.mse_loss(pred_c["vision"].float().cpu().reshape(t1.shape), t1)))
            vis_shuffled.append(float(F.mse_loss(pred_s["vision"].float().cpu().reshape(t1.shape), t1)))
        if "tactile" in pred_c:
            t0 = tr["t"]["tactile"].float()
            t1 = tr["t1"]["tactile"].float()
            tac_pred_delta.append(float((pred_c["tactile"].float().cpu().reshape(t1.shape) - t0)
                                        .pow(2).mean().sqrt()))
            tac_true_delta.append(float((t1 - t0).pow(2).mean().sqrt()))

    vc, vs = float(np.mean(vis_correct)), float(np.mean(vis_shuffled))
    A = {"vision_err_own_order": vc, "vision_err_other_order": vs,
         "own_order_advantage_pct": 100.0 * (vs - vc) / vc if vc else None,
         "n": len(vis_correct)}

    td = np.array(tac_true_delta)
    pd = np.array(tac_pred_delta)
    event = td >= np.percentile(td, 85)          # a real touch change arrives at t+1
    quiet = td <= np.percentile(td, 50)
    B = {"predicted_touch_change_before_event": float(pd[event].mean()),
         "predicted_touch_change_quiet": float(pd[quiet].mean()),
         "anticipation_ratio": float(pd[event].mean() / max(pd[quiet].mean(), 1e-9)),
         "actual_ratio_ceiling": float(td[event].mean() / max(td[quiet].mean(), 1e-9)),
         "n_event": int(event.sum()), "n_quiet": int(quiet.sum())}
    return A, B


def part_c(model, args, a, limits):
    """Look-away-look-back. Still body; gaze scripted: HOLD at A, AWAY for K orders,
    RETURN. At the return order the model's next-frame vision prediction is scored
    against the true returned view vs the away view it was just seeing."""
    from mimo_bridge.datasource import SingleEnvSource
    import tempfile
    # the eyes go only where the plan sends them, and nothing in the scene moves on its
    # own while they are away: no reflex, no saccades, no mobile, no critters, and no
    # caregiver (the hand-built world before 2026-09-23 had none by the env's default)
    env_kw = probe_world_kwargs(a, limits, tempfile.mkdtemp(prefix="wmc_"),
                                scene=args.scene, seed=150, tag="_wmc",
                                overrides=dict(gaze_reflex=False, spontaneous_saccades=False,
                                               mobile=False, critters=False,
                                               caregiver_mode="off", caregiver_seconds=0.0))
    A_G, AWAY_G = (0.55, -0.25), (-0.55, 0.15)      # rad: wall-side vs other side
    # the eye TRAVELS (~110 deg/s servo): the ~86 deg swing back takes ~16 orders of
    # 50 ms. The first draft scored the order right after the return command, when the
    # true view was still the away view 5.5 deg into the swing - it measured eye travel,
    # not memory. Score the prediction made LATE in the swing (arrival next order)
    # against a reference of what A actually held (mean of the last hold-phase views).
    RET = 20
    out = {}
    for K in args.away_orders:
        src = SingleEnvSource(**dict(env_kw, tag=f"_wmc{K}"))
        try:
            model.reset_transition_state()
            frames = src.reset()
            dev = dict(limits) if limits else dict(model.dev_limits(10 ** 9))
            # one row per body row of THIS world (24 legacy, 26 trunk_v1): identity
            # rotation, zero strength. A fixed 24 rows is refused by a trunk_v1 world.
            still = np.zeros((src.n_handles, 9)); still[:, 6] = 1.0
            reps = int(a.get("action_repeat") or 5)
            plan = [A_G] * 20 + [AWAY_G] * K + [A_G] * RET
            hold_views, preds_ret = [], []
            away_view = None
            for i, gz in enumerate(plan):
                with torch.no_grad():
                    model.step(frames, dev, train=False, step_index=i)
                tok = model.last_tokens["vision"].detach().float().cpu()
                if i < 20:
                    hold_views.append(tok)
                if i == 20 + K - 1:
                    away_view = tok                       # last view before turning back
                if i >= 20 + K:
                    preds_ret.append(model.last_predictions["vision"].detach().float().cpu())
                for _ in range(reps):
                    frames = src.step(still, np.array(gz), None, render=True)
            with torch.no_grad():
                model.step(frames, dev, train=False, step_index=len(plan))
            arrived = model.last_tokens["vision"].detach().float().cpu()
            ref_A = torch.stack(hold_views[-5:]).mean(0)
            pred_late = preds_ret[-4]                     # arrival is ~next order here
            adv = _cos(pred_late, ref_A) - _cos(pred_late, away_view)
            out[str(K)] = {"seconds_away": K * reps * 0.01,
                           "cos_to_what_A_held": _cos(pred_late, ref_A),
                           "cos_to_away_view": _cos(pred_late, away_view),
                           "cos_arrived_vs_ref": _cos(arrived, ref_A),   # scene stability check
                           "look_back_advantage": adv}
        finally:
            src.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--away-orders", type=int, nargs="+", default=[8, 32, 96])
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    t0 = time.time()
    model, a, _src, _f = _build_model(args.ckpt, args.device, 0, intention=False)
    if a.get("audio") == "real":
        model.config["modalities"].setdefault("audio", {})["data_source"] = "real"
    limits, _ = _limits_for_run(args.ckpt, model)
    limits["_action_repeat"] = int(a.get("action_repeat") or 5)

    A, B = part_a_and_b(model, args, a, limits)
    C = part_c(model, args, a, limits)

    rep = {"ckpt": os.path.abspath(args.ckpt), "A_see_own_action": A,
           "B_touch_anticipation": B, "C_look_back": C, "wall_s": time.time() - t0}
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))),
                           "world_model")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"battery_{os.path.basename(args.ckpt)}.json")
    json.dump(rep, open(out, "w"), indent=1)

    print(f"\nDoes moving its hand tell it what it will see? Knowing its own order "
          f"improves its guess of the next view by {A['own_order_advantage_pct']:.1f}% "
          f"over a stranger's order (0% = its hand teaches its eyes nothing; "
          f"n={A['n']}).")
    r, cl = B["anticipation_ratio"], B["actual_ratio_ceiling"]
    pct = 100.0 * (r - 1.0) / max(cl - 1.0, 1e-9)
    print(f"Does it feel a touch coming? It is {pct:.0f}% of the way from a brain that "
          f"never sees touch coming to one that always does (score {r:.2f}, where "
          f"never-knows = 1.00 and always-knows = {cl:.1f}; "
          f"n={B['n_event']} touch events vs {B['n_quiet']} quiet moments).")
    for k, v in C.items():
        print(f"Looking away for {v['seconds_away']:.1f}s and back: it scores "
              f"{v['look_back_advantage']:+.3f}, where 0 = it assumes the world is "
              f"whatever it last saw and positive = it expects what was really back "
              f"there. Part of a positive score is the eye already sweeping that way, "
              f"so trust the trend across away-times.")
    print(f"\nwritten {out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
