# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Fourteen-step planning probe - what stands between today's planner and a
fourteen-instant journey, measured on a frozen checkpoint. No training, no license
writes: every deep-chain number here is a FORCED PROBE (license unearned) and must be
labeled that way wherever it is quoted.

One model step = 50 ms of body time; the imagination chain's depth unit is the model
step. "One instant" = the world's change across one model step, re-measured HERE on
this checkpoint (anchors drift with the encoder; the stored scale_anchors.json was
measured once on one checkpoint).

Measures, from ONE rollout under the checkpoint's own developmental limits:
  displacement   token distance between moments k steps apart, k=1..depth, per sense
                 and anchor-mean - the curve that converts "instants" claims (token
                 MSE saturates toward the unrelated-moments level; linear division
                 by one_instant overstates far distances)
  fidelity       forced imagination chains to --depth on real executed actions:
                 per-depth error in anchor units vs three references (chain-start
                 persistence = the license's own do-nothing baseline, which gets
                 EASIER with depth; previous-frame copy = the depth-matched harder
                 reference; win rate under the license's exact sum-over-graded rule)
  planning       forced _plan_chain calls at --plan-depths from live moments with the
                 model's own chosen goal: realized plan length vs asked depth (guard
                 trims), imagined start/final distance, wall seconds per call
  journey        the model's OWN planning behavior during the rollout: goal
                 persistence (how many consecutive plan events keep the same goal,
                 within this checkpoint's one-instant), closest real approach to each
                 chosen goal afterward (the measured arrival floor - the hard-coded
                 0.01 convergence line was never measured), executed travel toward
                 the goal between events, per-sense share of goal distance

Writes Saved/mimo_bridge/depth_ladder/<ckpt>_s<seed>.json and prints a summary.
python -m mimo_bridge.tools.depth_ladder_probe --ckpt <ckpt_dir> [--seed 0]
    [--steps 650] [--depth 14] [--start-stride 10] [--plan-stride 120]
    [--plan-depths 3,6,14]
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
from mimo_bridge.lem_model import LemLite  # noqa: E402  (gaze_to_handle static)
from lem.live_test import _run_imagination_compiled, _src_to_str  # noqa: E402

ANCHOR_MODS = ("vision", "tactile", "audio", "proprioception")


def _amean(a, b, mods=ANCHOR_MODS):
    """Anchor-units distance: mean over the anchor senses of token MSE."""
    ms = [m for m in mods if m in a and m in b]
    if not ms:
        return None
    return float(np.mean([F.mse_loss(a[m].float(), b[m].float()).item() for m in ms]))


def _per_sense(a, b, mods=ANCHOR_MODS):
    return {m: float(F.mse_loss(a[m].float(), b[m].float()).item())
            for m in mods if m in a and m in b}


def _med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.median(xs)) if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=650)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--start-stride", type=int, default=10)
    ap.add_argument("--plan-stride", type=int, default=120)
    ap.add_argument("--plan-depths", default="3,6,14")
    ap.add_argument("--warm", type=int, default=60)
    ap.add_argument("--approach-window", type=int, default=200,
                    help="steps after a plan event over which closest approach is taken")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)          # leave cores for the live trainer's env workers
    t0 = time.time()
    plan_depths = [int(x) for x in args.plan_depths.split(",") if x.strip()]
    timing = {}

    model, a, _s, _f = _build_model(args.ckpt, "cpu", args.seed, intention=True)
    if a.get("audio") == "real":
        model.config["modalities"].setdefault("audio", {})["data_source"] = "real"
    assert model.imagination_enabled, (
        "this checkpoint trained with imagination off: the history deques the probe "
        "reads never fill - probe another checkpoint or build with imagination on")
    # review 2026-09-06: runtime_state carries agency_pay_ema at the LIVE batch size
    # (n_envs=8); at B=1 the un-guarded broadcast in the goal-selection path crashes
    # the first plan. Keep the earned average, conformed to this probe's batch.
    _ape = getattr(model, "_agency_pay_ema", None)
    if _ape is not None and torch.is_tensor(_ape) and _ape.numel() != 1:
        model._agency_pay_ema = _ape.mean().reshape(1)
    # plan cadence as trained (a latent _build_model gap; default happens to match
    # H67 but is not guaranteed)
    if a.get("intention_every") is not None:
        model.intention_every = int(a["intention_every"])
    limits, _ = _limits_for_run(args.ckpt, model)
    from mimo_bridge.datasource import SingleEnvSource
    reps = int(a.get("action_repeat") or 5)
    hold = int(a.get("motor_hold") or 5)
    # env built the way the run was trained (causal_test review E9 idiom): servo kp,
    # damping, caregiver, command horizon, reflex, body age, peripheral ring - the
    # defaults are a DIFFERENT world (kp 8 vs 2, base-age body, no ring)
    env_kw = dict(scene="crib", run_dir=tempfile.mkdtemp(prefix="ladder_"), vision=True,
                  fovea_size=int(a.get("fovea_size") or 288),
                  fovea_fov=float(a.get("fovea_fov") or 24.0))
    for _src_key, _env_key in (("decoder_kp", "kp"),
                               ("joint_damping_scale", "joint_damping_scale"),
                               ("hinge_armature", "hinge_armature"), ("finger_armature", "finger_armature"),
                               ("touch_scale_mult", "touch_scale_mult"),
                               ("caregiver_seconds", "caregiver_seconds"),
                               ("gaze_kp", "gaze_kp"), ("gaze_kd", "gaze_kd"),
                               ("ring_fov", "ring_fov"), ("ring_size", "ring_size")):
        if a.get(_src_key) is not None:
            env_kw[_env_key] = a[_src_key]
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
    src = SingleEnvSource(tag="_ladder", seed=170 + args.seed, **env_kw)

    D = args.depth
    raw, obs, valid, motor, ctxs, achs = [], [], [], [], [], []
    plan_probe, journey = [], []
    _ctrl_rng = np.random.default_rng(7700 + args.seed)   # matched-random-goal picks
    t_collect = time.time()
    try:
        model.reset_transition_state()
        frames = src.reset()
        dev = dict(limits)
        # the movement regime the checkpoint trained under: _limits_for_run zeroes
        # exploration noise, the live run babbles at args babble_noise (0.05 restores
        # 0.512 rad orders; without it the off-plan steps drive a stiller body than
        # the one the checkpoint knows)
        dev["exploration_noise"] = float(a.get("babble_noise") or 0.0)
        held, prev_exec = None, None
        for i in range(args.steps):
            with torch.no_grad():
                res = model.step(frames, dev, train=False, step_index=i,
                                 executed_action=prev_exec)
            if i == 20:
                _per = (time.time() - t_collect) / 21.0
                print(f"[timing] {_per:.2f}s per collect step - projected collect "
                      f"{_per * args.steps / 60.0:.0f} min for --steps {args.steps}")
            ok = len(model.enc_history_raw) > 0 and len(model.enc_history) > 0
            raw.append({m: t.detach().float().clone()
                        for m, t in model.enc_history_raw[-1].items()} if ok else None)
            obs.append({m: t.detach().float().clone()
                        for m, t in model.enc_history[-1].items()} if ok else None)
            valid.append({m: _src_to_str(v) for m, v in model.valid_history[-1].items()}
                         if len(model.valid_history) else {})
            motor.append(model.motor_history[-1].detach().float().clone()
                         if len(model.motor_history) else None)
            ctxs.append(model.ctx_history[-1].detach().float().clone()
                        if len(model.ctx_history) else None)
            achs.append(model.ach_history[-1].detach().float().clone()
                        if len(model.ach_history) else None)

            # the model's OWN plan fired this step: capture the goal it chose (the ring
            # as it stands is the ring the plan read - the append precedes the plan
            # inside the same step)
            li = model.last_intention
            if li is not None and li.get("step") == i and len(model.goal_ring) >= 20:
                with torch.no_grad():
                    if (getattr(model, "goal_locality", False)
                            and model._gl_held_tok is not None):
                        # the live policy: the HELD committed goal, not a fresh argmax
                        goal = {m: v.detach().float().clone()
                                for m, v in model._gl_held_tok.items()}
                    else:
                        wl = torch.stack([w for _, w in model.goal_ring])   # [T, B]
                        bt = int(wl.argmax(dim=0)[0])
                        goal = {m: torch.stack([t[m] for t, _ in model.goal_ring]
                                               )[bt].detach().float().clone()
                                for m in model.active}
                # ring census (goal-locality pre-build measurement, 2026-09-06 eve):
                # per event, in the POST-GATE space goal selection lives in - the
                # self-measured reach radius (distance to the entry k back), what
                # fraction of the ring a locality rule would admit, the wellness
                # spread inside that ball vs the whole ring vs step-to-step jitter
                # (the noise floor), and what the restricted rule WOULD have picked
                census = None
                if len(model.enc_history) > 0:
                    with torch.no_grad():
                        cur_g = {m: t.detach().float()
                                 for m, t in model.enc_history[-1].items()}
                        vh_now = valid[-1] if valid else {}
                        cmods = [m for m in ANCHOR_MODS
                                 if vh_now.get(m) == "real" and m in cur_g]
                        entries = list(model.goal_ring)     # oldest -> newest
                        T = len(entries)
                        dists = [float(np.mean([F.mse_loss(
                            cur_g[m], entries[j][0][m].float()).item()
                            for m in cmods if m in entries[j][0]]))
                            for j in range(T)]
                        ws = [float(entries[j][1][0]) for j in range(T)]
                        ages = [T - 1 - j for j in range(T)]
                        r_at = {k: next((dists[j] for j in range(T)
                                         if ages[j] == k), None)
                                for k in (6, 14, 25)}
                        R = r_at.get(14)
                        census = {"R6": r_at.get(6), "R14": R, "R25": r_at.get(25)}
                        if R is not None:
                            inside = [j for j in range(T) if dists[j] <= R]
                            census["frac_inside"] = len(inside) / max(T, 1)
                            census["ball_empty"] = not inside
                            jit = [abs(ws[j + 1] - ws[j]) for j in range(T - 1)]
                            census["w_jitter_adjacent"] = _med(jit)
                            census["w_spread_all"] = float(np.percentile(ws, 75)
                                                           - np.percentile(ws, 25))
                            if inside:
                                wi = [ws[j] for j in inside]
                                census["w_spread_inside"] = (
                                    float(np.percentile(wi, 75)
                                          - np.percentile(wi, 25)))
                                jb = max(inside, key=lambda j: ws[j])
                                census["restricted_age"] = ages[jb]
                                census["restricted_dist"] = dists[jb]
                                census["w_cost_of_locality"] = max(ws) - ws[jb]
                                census["restricted_goal"] = {
                                    m: entries[jb][0][m].detach().float().clone()
                                    for m in model.active if m in entries[jb][0]}
                            census["unrestricted_age"] = ages[int(np.argmax(ws))]
                # ATTRIBUTION CONTROL (operator 2026-09-07): a random ring moment at
                # MATCHED distance (within +-20% of the chosen goal's distance).
                # If chosen goals get approached no better than matched random ones,
                # "arrivals" are world drift, not plan steering.
                ctrl_goal = None
                if len(model.enc_history) > 0:
                    with torch.no_grad():
                        curJ = {m: t.detach().float()
                                for m, t in model.enc_history[-1].items()}
                        vhJ = valid[-1] if valid else {}
                        mJ = [m for m in ANCHOR_MODS
                              if vhJ.get(m) == "real" and m in curJ]
                        gd_real = _amean(curJ, goal, mJ)
                        if gd_real:
                            cand = []
                            for jj, (tt, _w) in enumerate(model.goal_ring):
                                dj = _amean(curJ, tt, [m for m in mJ if m in tt])
                                if not (dj and abs(dj - gd_real) <= 0.2 * gd_real):
                                    continue
                                # not the chosen goal itself (or its near-twin):
                                # a control that IS the goal proves nothing
                                dg = _amean(tt, goal, [m for m in mJ if m in tt])
                                if dg is not None and dg < 0.1 * gd_real:
                                    continue
                                cand.append(jj)
                            if cand:
                                pickj = cand[int(_ctrl_rng.integers(len(cand)))]
                                ctrl_goal = {m: model.goal_ring[pickj][0][m]
                                             .detach().float().clone()
                                             for m in model.active
                                             if m in model.goal_ring[pickj][0]}
                journey.append({"step": i, "goal": goal, "ctrl_goal": ctrl_goal,
                                "license": (model.last_plan_license or {}).get(
                                    "licensed"),
                                "census": census,
                                "meta": {k: li.get(k) for k in
                                         ("plan_length", "horizon", "converged",
                                          "final_mse", "start_mse", "gap_closed_pct")}})

            # forced planning ladder at chosen live moments (label: forced probe -
            # depths beyond the license; nothing is written back to the model)
            if (i % args.plan_stride == 0 and i >= args.warm
                    and len(model.goal_ring) >= 20 and len(model.enc_history) > 0):
                with torch.no_grad():
                    if (getattr(model, "goal_locality", False)
                            and model._gl_held_tok is not None):
                        goal = {m: v.detach().float().clone()
                                for m, v in model._gl_held_tok.items()}
                    else:
                        wl = torch.stack([w for _, w in model.goal_ring])
                        bt = int(wl.argmax(dim=0)[0])
                        goal = {m: torch.stack([t[m] for t, _ in model.goal_ring]
                                               )[bt].detach().float()
                                for m in model.active}
                    # the live planner starts from the GATED tokens (the ring stores
                    # gated too); last_tokens is the RAW path - mixing the two spaces
                    # shrinks distances by up to (1-g)^2 wherever td_gate is open
                    cur = {m: model.enc_history[-1][m].detach().float()
                           for m in model.active if m in model.enc_history[-1]}
                    mo = torch.as_tensor(res.motor_out, dtype=torch.float32)
                for N in plan_depths:
                    tp = time.perf_counter()
                    with torch.enable_grad():
                        plan, meta = model._plan_chain(cur, goal, mo, max_steps=N)
                    _sm, _fm = meta.get("start_mse"), meta.get("final_mse")
                    plan_probe.append(
                        {"step": i, "asked_depth": N, "plan_len": len(plan),
                         "wall_s": round(time.perf_counter() - tp, 3),
                         "start_mse": _sm if (_sm is not None and _sm == _sm) else None,
                         "final_mse": _fm if (_fm is not None and _fm == _fm) else None,
                         "converged": bool(meta.get("converged")),
                         "valid": bool(meta.get("valid", True))})

            if i % hold == 0 or held is None:
                held = res.motor_for_action
            gz_burst = []
            for rep in range(reps):
                # render only the last substep, as the trainer does (rendering every
                # substep also feeds the gaze reflex 5x per burst vs once live)
                frames = src.step(held, res.gaze_out, None,
                                  render=(rep == reps - 1))
                gz_burst.append(np.stack([np.asarray(
                    f.meta.get("gaze_cmd") or (0.0, 0.0), dtype=np.float32)
                    for f in frames]))
            # the trainer's ledger convention: handle 23 carries the burst's MEAN
            # measured gaze, not the command the controller happened to emit
            prev_exec = LemLite.gaze_to_handle(held,
                                               np.mean(np.stack(gz_burst), axis=0))
    finally:
        src.close()
    timing["collect_s"] = round(time.time() - t_collect, 1)

    n = len(raw)

    def _real(i, j):
        """The anchor senses that are REAL at both frames - a synthetic placeholder
        (fresh Gaussian each step) is a large constant MSE that would pollute every
        distance it entered."""
        return [m for m in ANCHOR_MODS
                if valid[i].get(m) == "real" and valid[j].get(m) == "real"]

    # collection starts after --warm, same regime as the fidelity block (the
    # reset-settle transient is not "ordinary experience" for the anchors)
    usable = [i for i in range(args.warm, n)
              if raw[i] is not None and obs[i] is not None]
    if not usable:
        raise SystemExit(
            "no usable frames collected - the history deques never filled "
            f"(imagination_enabled={model.imagination_enabled}); refusing to write "
            "a JSON full of nulls")

    # ---- displacement curve (this checkpoint's own anchors) ----
    t_disp = time.time()
    disp, disp_sense = {}, {}
    for k in list(range(1, D + 1)):
        # stride pairs by k so windows do not overlap (overlap inflates n, not the
        # median, but the JSON's n should mean independent-ish samples)
        cand = usable[::max(1, k)]
        pairs = [(_amean(raw[t], raw[t + k], _real(t, t + k)),
                  _per_sense(raw[t], raw[t + k], _real(t, t + k)))
                 for t in cand if t + k < n and raw[t + k] is not None]
        disp[k] = _med([p[0] for p in pairs])
        disp_sense[k] = {m: _med([p[1].get(m) for p in pairs])
                         for m in ANCHOR_MODS if pairs and m in pairs[0][1]}
    # usable starts at --warm, so the far pair gap must leave room above it
    # (600 with warm 60 in 650 steps left ZERO pairs on the first run)
    far_gap = max(D + 1, min(600, n - args.warm - 30))
    far_pairs = [_amean(raw[t], raw[t + far_gap], _real(t, t + far_gap))
                 for t in usable[::14]
                 if t + far_gap < n and raw[t + far_gap] is not None]
    one_instant = disp.get(1)
    obs_one_instant = _med([_amean(obs[t], obs[t + 1], _real(t, t + 1))
                            for t in usable
                            if t + 1 < n and obs[t + 1] is not None])
    timing["displacement_s"] = round(time.time() - t_disp, 1)

    # ---- forced fidelity ladder ----
    t_fid = time.time()
    fid = {d: {"err": [], "pers": [], "prev": [], "win": [], "err_sense": []}
           for d in range(1, D + 1)}
    n_starts = 0
    for t in range(args.warm, n - D - 1, args.start_stride):
        window = range(t, t + D + 1)
        if any(raw[j] is None or obs[j] is None for j in window):
            continue
        # motor[i] holds the action that PRODUCED frame i (appended one step late in
        # the model); the chain wants the action OUT of frame t+s, which is
        # motor[t+s+1] - review 2026-09-06 caught the probe re-introducing
        # causal_test's documented defect 1 (wrong action index)
        if any(motor[j] is None for j in range(t + 1, t + D + 1)):
            continue
        if ctxs[t] is None or achs[t] is None:
            continue
        enc_slice = [obs[j] for j in window]
        raw_slice = [raw[j] for j in window]
        val_slice = [valid[j] for j in window]
        mot_slice = [motor[j] for j in range(t + 1, t + D + 1)]
        pure, imag = [], []
        with torch.no_grad():
            _run_imagination_compiled(
                enc_slice, mot_slice, D, model.forward_model,
                model.prediction_heads, model.encoder_registry, ctxs[t],
                model.active, model.n_mod, latent_regularizer=None,
                valid_history=val_slice, achieved_state=achs[t],
                target_history=raw_slice, pure_out=pure, imagined_out=imag)
        if len(imag) < D:
            continue
        n_starts += 1
        base0 = raw[t]
        for d in range(1, D + 1):
            tgt = raw[t + d]
            vh = valid[t + d]
            graded = [m for m in model.pred_mods
                      if m != "motor" and m in tgt and m in base0
                      and vh.get(m) == "real"]
            rec = fid[d]
            rec["err"].append(_amean(imag[d - 1], tgt, graded))
            rec["err_sense"].append(_per_sense(imag[d - 1], tgt, graded))
            rec["pers"].append(_amean(base0, tgt, graded))
            rec["prev"].append(_amean(raw[t + d - 1], tgt, graded))
            if graded:
                # ONE rule for both sides of the win (the training license compares
                # live_test's graded set against its own - the sets coincide while
                # source masks are complete, but the probe does not inherit the
                # asymmetry): chain error and persistence both summed over `graded`
                chain_lic = float(sum(F.mse_loss(imag[d - 1][m], tgt[m])
                                      for m in graded))
                pers_lic = float(sum(F.mse_loss(base0[m], tgt[m]) for m in graded))
                rec["win"].append(1.0 if chain_lic < pers_lic else 0.0)
    fidelity = {d: {"err_anchor_median": _med(v["err"]),
                    "persistence_anchor_median": _med(v["pers"]),
                    "prev_frame_anchor_median": _med(v["prev"]),
                    "win_rate_license_rule": (float(np.mean(v["win"])) if v["win"]
                                              else None),
                    "err_per_sense_median": {m: _med([s.get(m) for s in v["err_sense"]])
                                             for m in ANCHOR_MODS},
                    "n": len(v["err"])}
                for d, v in fid.items()}
    timing["fidelity_s"] = round(time.time() - t_fid, 1)

    # ---- journey analysis: persistence, closest approach, executed travel ----
    t_j = time.time()
    pers_runs, approaches, approaches_code, travel, sense_share = [], [], [], [], []
    approaches_ctrl = []
    run_len = 1
    for e in range(len(journey)):
        ev = journey[e]
        t_ev, goal = ev["step"], ev["goal"]
        if t_ev >= n or obs[t_ev] is None:
            continue
        gmods = _real(t_ev, t_ev)         # senses real at the goal's own moment
        sense_share.append(_per_sense(obs[t_ev], goal, gmods))
        if e > 0:
            p_ev = journey[e - 1]["step"]
            gd = _amean(journey[e - 1]["goal"], goal,
                        [m for m in gmods if valid[p_ev].get(m) == "real"])
            same = (gd is not None and obs_one_instant is not None
                    and gd < obs_one_instant)
            if same:
                run_len += 1
            else:
                pers_runs.append(run_len)
                run_len = 1
        # closest real approach to this goal afterward - only events with a FULL
        # window (a truncated window can only bias the closest approach high)
        if t_ev + 1 + args.approach_window <= n:
            best, best_code, best_ctrl = None, None, None
            cgoal = ev.get("ctrl_goal")
            for t2 in range(t_ev + 1, t_ev + 1 + args.approach_window):
                if obs[t2] is None:
                    continue
                dcur = _amean(obs[t2], goal,
                              [m for m in gmods if valid[t2].get(m) == "real"])
                if dcur is not None and (best is None or dcur < best):
                    best = dcur
                # the code's own convergence rule averages over ALL shared mods,
                # motor included, no validity filter (lem_model _plan_chain)
                dcode = _amean(obs[t2], goal, sorted(set(obs[t2]) & set(goal)))
                if dcode is not None and (best_code is None or dcode < best_code):
                    best_code = dcode
                if cgoal is not None:
                    dctl = _amean(obs[t2], cgoal,
                                  [m for m in gmods if valid[t2].get(m) == "real"])
                    if dctl is not None and (best_ctrl is None or dctl < best_ctrl):
                        best_ctrl = dctl
            if best is not None:
                approaches.append(best)
            if best_code is not None:
                approaches_code.append(best_code)
            if best_ctrl is not None:
                approaches_ctrl.append(best_ctrl)
        # executed travel toward THIS goal by the next event
        if e + 1 < len(journey):
            t_nx = journey[e + 1]["step"]
            if t_nx < n and obs[t_nx] is not None:
                tmods = [m for m in gmods if valid[t_nx].get(m) == "real"]
                d0 = _amean(obs[t_ev], goal, tmods)
                d1 = _amean(obs[t_nx], goal, tmods)
                if d0 is not None and d1 is not None:
                    travel.append(d0 - d1)
    if journey:
        pers_runs.append(run_len)
    goal_dists = [m["meta"]["start_mse"] for m in journey
                  if m["meta"].get("start_mse") is not None]
    ev_gaps = [journey[e + 1]["step"] - journey[e]["step"]
               for e in range(len(journey) - 1)]
    # ---- ring census aggregates (goal-locality pre-build measurement) ----
    cen = [j["census"] for j in journey if j.get("census")]
    rc = None
    if cen:
        def _cm(key):
            return _med([c.get(key) for c in cen])
        # would restricted goals persist? same rule as the live persistence stat:
        # consecutive events whose restricted pick stays within one instant
        r_runs, rl = [], 1
        rg = [c.get("restricted_goal") for c in cen]
        for e in range(1, len(rg)):
            gd = (_amean(rg[e - 1], rg[e]) if rg[e - 1] and rg[e] else None)
            if (gd is not None and obs_one_instant is not None
                    and gd < obs_one_instant):
                rl += 1
            else:
                r_runs.append(rl)
                rl = 1
        r_runs.append(rl)
        spread_in, jit = _cm("w_spread_inside"), _cm("w_jitter_adjacent")
        rc = {"n_events": len(cen),
              "R6_median": _cm("R6"), "R14_median": _cm("R14"),
              "R25_median": _cm("R25"),
              "frac_ring_inside_R14_median": _cm("frac_inside"),
              "ball_empty_pct": 100.0 * float(np.mean(
                  [bool(c.get("ball_empty")) for c in cen])),
              "wellness_spread_inside_ball_median": spread_in,
              "wellness_spread_whole_ring_median": _cm("w_spread_all"),
              "wellness_jitter_adjacent_median": jit,
              "spread_inside_over_jitter": (spread_in / jit
                                            if spread_in and jit else None),
              "wellness_cost_of_locality_median": _cm("w_cost_of_locality"),
              "restricted_pick_age_steps_median": _cm("restricted_age"),
              "restricted_pick_age_steps_p90": (
                  float(np.percentile([c["restricted_age"] for c in cen
                                       if c.get("restricted_age") is not None], 90))
                  if any(c.get("restricted_age") is not None for c in cen)
                  else None),
              "unrestricted_pick_age_steps_median": _cm("unrestricted_age"),
              "restricted_goal_persistence_events_median": _med(
                  [float(x) for x in r_runs]),
              "note": "post-gate space (where goal selection lives); R14 = distance "
                      "to the ring entry 14 steps back = the online self-measured "
                      "reach; spread_inside_over_jitter <= ~1 means the best-felt "
                      "rule inside the ball would be picking noise"}
    journey_out = {
        "n_plan_events": len(journey),
        "goal_persistence_events_median": _med([float(x) for x in pers_runs]),
        "goal_persistence_events_max": (int(max(pers_runs)) if pers_runs else None),
        "nominal_steps_between_events": int(a.get("intention_every") or 5),
        # events are only SUCCESSFUL plans (an emptied plan leaves no record), so
        # real gaps stretch past the nominal cadence
        "actual_steps_between_events_median": _med([float(g) for g in ev_gaps]),
        "license_at_events_median": _med([float(j["license"]) for j in journey
                                          if j.get("license") is not None]),
        "closest_approach_median": _med(approaches),
        "closest_approach_p10": (float(np.percentile(approaches, 10))
                                 if approaches else None),
        "closest_approach_random_ctrl_median": _med(approaches_ctrl),
        "attribution_note": "ctrl = random ring moment at matched distance (+-20%); "
                            "chosen-goal approach must beat this or arrivals are "
                            "world drift, not steering",
        "n_ctrl_events": len(approaches_ctrl),
        "arrival_line_in_code": 0.01,
        "reached_code_line_pct": (
            100.0 * float(np.mean([x < 0.01 for x in approaches_code]))
            if approaches_code else None),
        "reached_code_line_note": "computed with the code's own rule (mean over all "
                                  "shared token senses INCLUDING motor, no validity "
                                  "filter) on observed gated tokens",
        "reached_one_instant_pct": (
            100.0 * float(np.mean([x < obs_one_instant for x in approaches]))
            if approaches and obs_one_instant else None),
        "executed_travel_per_event_median": _med(travel),
        "goal_distance_at_choice_median": _med(goal_dists),
        "goal_distance_per_sense_median": {m: _med([s.get(m) for s in sense_share])
                                           for m in ANCHOR_MODS},
        "plan_meta_medians": {k: _med([j["meta"].get(k) for j in journey])
                              for k in ("plan_length", "final_mse", "start_mse",
                                        "gap_closed_pct")},
        "ring_census": rc}
    timing["journey_s"] = round(time.time() - t_j, 1)
    timing["total_s"] = round(time.time() - t0, 1)

    out = {"computed_at": time.strftime("%Y-%m-%d %H:%M"),
           "ckpt": os.path.abspath(args.ckpt), "seed": args.seed,
           "steps_collected": n, "label": "FORCED PROBE - depths beyond the earned "
           "license are measurements, not earned capability. Nothing is persisted to "
           "disk; the IN-MEMORY plan license does keep updating during the rollout "
           "(the chain and its win EMAs run every step, as live)",
           "units": "token MSE, anchor-mean over " + ",".join(ANCHOR_MODS) +
                    " unless suffixed; one model step = 50 ms",
           "anchors_this_ckpt": {
               "one_instant_raw": one_instant,
               "one_instant_postgate": obs_one_instant,
               "far_%d_steps" % far_gap: _med(far_pairs)},
           "displacement_curve": {str(k): v for k, v in disp.items()},
           "displacement_per_sense": {str(k): v for k, v in disp_sense.items()},
           "fidelity": {str(d): v for d, v in fidelity.items()},
           "fidelity_n_starts": n_starts,
           "plan_ladder": plan_probe,
           "journey": journey_out,
           "timing_s": timing,
           "env": {k: v for k, v in env_kw.items() if k != "run_dir"},
           "dev_limits": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                          for k, v in limits.items()}}
    os.makedirs(os.path.join(SAVED, "depth_ladder"), exist_ok=True)
    name = os.path.basename(os.path.normpath(args.ckpt))
    run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))))
    path = os.path.join(SAVED, "depth_ladder", f"{run}_{name}_s{args.seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"one instant (this ckpt): raw {one_instant}, post-gate {obs_one_instant}; "
          f"far({far_gap} steps) { _med(far_pairs)}")
    for d in (1, 3, 6, 10, 14):
        if d <= D and fidelity.get(d):
            f_ = fidelity[d]
            print(f"depth {d:2d}: chain err {f_['err_anchor_median']}, persistence "
                  f"{f_['persistence_anchor_median']}, prev-frame "
                  f"{f_['prev_frame_anchor_median']}, win {f_['win_rate_license_rule']}")
    print(f"journey: {journey_out['n_plan_events']} events, persistence(median) "
          f"{journey_out['goal_persistence_events_median']} events, closest approach "
          f"{journey_out['closest_approach_median']} (code line 0.01), travel/event "
          f"{journey_out['executed_travel_per_event_median']}")
    if rc:
        print(f"ring census: R14 {rc['R14_median']}, {100*(rc['frac_ring_inside_R14_median'] or 0):.0f}% "
              f"of ring inside, spread-inside/jitter {rc['spread_inside_over_jitter']}, "
              f"wellness cost of locality {rc['wellness_cost_of_locality_median']}, "
              f"restricted pick age median {rc['restricted_pick_age_steps_median']} steps "
              f"(p90 {rc['restricted_pick_age_steps_p90']}), restricted persistence "
              f"{rc['restricted_goal_persistence_events_median']} events")
    print(f"written {path} ({timing['total_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
