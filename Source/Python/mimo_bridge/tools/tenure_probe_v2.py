# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tenure probe suite v2 (2026-09-02): re-measure every number the reviews challenged.

Why v1's numbers were thrown out (findings F9-F19): every v1 probe re-derived tenure
windows offline and segmented them one step late; the freshness control was confounded
with posture (a body's begin pose fingerprints its own order); the ceiling ridge leaked
through unblocked folds; "bed accuracy" was training accuracy; "hand motion +0.34" used
all 24 handles; sanity correlations ran at n~80.

v2 fixes each by construction:
- COLLECTION records the model's OWN live tenure state: the pending begin-state the
  model built at each boundary, and the pre-step caches its end-computation consumed.
  Windows therefore match live segmentation exactly - there is nothing to re-derive.
- The offline estimator is VALIDATED event-by-event against the live-logged rank1
  before any control runs: if the offline number does not reproduce the live number on
  the same events, the suite refuses to report controls.
- The wrong-order control is POSTURE-MATCHED: the decoy order comes from the event
  whose begin pose is nearest, so posture fingerprints cannot answer for the order.
- The ceiling ridge runs on blocked contiguous folds with an inner-validated lambda.
- Motion columns report hand handles (0..11) and all-handles separately.

Usage:
  python -m mimo_bridge.tools.tenure_probe_v2 collect <ckpt> <out.pt> [n_steps] [n_envs] [seed_env]
  python -m mimo_bridge.tools.tenure_probe_v2 analyze <out.pt> [report.json]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SENSES = ("vision", "tactile", "proprioception")


# ------------------------------------------------------------------ collection
def collect(ckpt: str, out_path: str, n_steps: int = 1800, n_envs: int = 2,
            run_dir: str | None = None, seed_env: int = 41):
    torch.set_num_threads(2)          # a training run owns most of this machine's CPU
    from mimo_bridge.tools.causal_test import _build_model, _limits_for_run
    from mimo_bridge.datasource import PoolSource, SingleEnvSource
    from mimo_bridge.lem_model import LemLite, command_for_encoder

    model, a, src_path, found = _build_model(ckpt, "cpu", seed=0, intention=True)
    model.load_checkpoint(ckpt, load_optimizer=False)
    limits, limits_src = _limits_for_run(ckpt, model)
    hold = int(a.get("motor_hold") or 5)
    reps = int(a.get("action_repeat") or 5)
    dev = dict(limits)
    dev["exploration_noise"] = float(a.get("babble_noise") or 0.05)

    rd = run_dir or os.path.join(os.path.dirname(out_path), "probe_env")
    kw = dict(scene=str(a.get("scene") or "crib"), run_dir=rd, vision=True,
              fovea_size=int(a.get("fovea_size") or 288),
              fovea_fov=float(a.get("fovea_fov") or 24.0), seed=int(seed_env), tag="_pv2",
              mobile=(a.get("mobile") == "on"))   # without this the self-motion
              # gate has nothing to open on and vision's slice reads as zero
    for sk, ek in (("decoder_kp", "kp"), ("joint_damping_scale", "joint_damping_scale"),
                   ("hinge_armature", "hinge_armature"), ("finger_armature", "finger_armature"), ("caregiver_seconds", "caregiver_seconds"),
                   ("gaze_kp", "gaze_kp"), ("gaze_kd", "gaze_kd")):
        if a.get(sk) is not None:
            kw[ek] = a[sk]
    kw["cmd_horizon"] = reps * hold
    if limits.get("age_months") is not None:
        kw["age"] = float(limits["age_months"])
    src = PoolSource(n_envs, kw) if n_envs > 1 else SingleEnvSource(**kw)

    # The match term only computes under train=True (its gradient graph is the
    # point in production). Collection must therefore step with train=True while
    # provably NOT training: the optimizer's step is disabled, and a parameter
    # fingerprint taken before is asserted identical after - the suite refuses to
    # save events from a drifted model.
    model.optimizer.step = (lambda *a, **k: None)
    with torch.no_grad():
        fp_before = float(sum(p.abs().sum() for p in model.params))

    events = []
    try:
        frames = src.reset()
        model.reset_context(n_envs)
        model.reset_transition_state()
        held = prev_exec = None
        order_seq, prev_oid = 0, None
        prev_pend = None
        prev_pend_id = None
        t0 = time.time()
        for i in range(n_steps):
            # the caches as they stand NOW are what step(i)'s end-computation will
            # consume (it reads them before refreshing them to step i's state)
            pre = {"raw": model._nf_raw, "mask": model._nf_mask,
                   "ach": model._nf_ach, "hv": model._nf_handvis}
            res = model.step(frames, dev, train=True, step_index=i,
                             executed_action=prev_exec, executed_order_id=prev_oid)
            if "tenure_tok" in res.breakdown and prev_pend is not None:
                B = n_envs
                with torch.no_grad():
                    tb = _sense_slices(model, prev_pend["raw"], prev_pend["mask"], B)
                    te = _sense_slices(model, pre["raw"], pre["mask"], B)
                if tb is not None and te is not None and pre["ach"] is not None:
                    events.append({
                        "tb": tb, "te": te,
                        "order": prev_pend["order"].clone(),
                        "ach_b": prev_pend["ach"][..., :3].clone(),
                        "ach_e": pre["ach"][..., :3].clone(),
                        "hv_b": prev_pend["hand_vis"].clone().bool(),
                        "hv_e": (pre["hv"].clone().bool() if pre["hv"] is not None
                                 else torch.zeros(B, dtype=torch.bool)),
                        "live": dict(model.last_tenure_tok),
                        "step": i})
            p = model._tenure_pending
            if p is not None and id(p) != prev_pend_id:
                prev_pend, prev_pend_id = p, id(p)
            if i % hold == 0 or held is None:
                held = res.motor_for_action
                order_seq += 1
            gz = []
            for rep in range(reps):
                frames = src.step(held, res.gaze_out, None, render=(rep == reps - 1))
                gz.append(np.stack([np.asarray(f.meta.get("gaze_cmd") or (0.0, 0.0),
                                               dtype=np.float32) for f in frames]))
            prev_exec = LemLite.gaze_to_handle(held, np.mean(np.stack(gz), axis=0))
            prev_oid = order_seq
            if i and i % 100 == 0:
                # RAM guard (2026-09-02): this collector's train=True gradient
                # graphs plus vision envs starved H62's trainer to an OOM crash at
                # 110 min. A probe must yield, never fight a training run for the
                # machine - stop gracefully and keep the events collected so far.
                import psutil
                _avail = psutil.virtual_memory().available
                if _avail < 3.5e9:
                    print(f"  RAM guard: {_avail/1e9:.1f} GB free < 3.5 GB - "
                          f"stopping with {len(events)} events", flush=True)
                    break
            if i and i % 200 == 0:
                print(f"  {i}/{n_steps} steps, {len(events)} events, "
                      f"{time.time() - t0:.0f}s", flush=True)
    finally:
        src.close()

    with torch.no_grad():
        fp_after = float(sum(p.abs().sum() for p in model.params))
    if fp_after != fp_before:
        raise SystemExit(f"the model DRIFTED during collection ({fp_before} -> "
                         f"{fp_after}) - events discarded")
    print(f"parameter fingerprint held: {fp_before:.6f}")
    meta = {"ckpt": ckpt, "args_json": src_path, "limits_source": limits_src,
            "n_steps": n_steps, "n_envs": n_envs, "hold": hold, "reps": reps,
            "d_model": model.d_model, "n_events": len(events)}
    torch.save({"events": events, "meta": meta,
                "order_proj": {k: v.detach().clone() for k, v in
                               model.tenure_heads["order_proj"].state_dict().items()},
                "proj_in": next(model.tenure_heads["order_proj"].parameters()).shape[-1]},
               out_path)
    print(f"saved {len(events)} events -> {out_path}")
    return out_path


def _sense_slices(model, raw, mask, B):
    """The live _live_tokens computation, replicated exactly (senses only, cached
    masks, registry encode). Returns {sense: [B, d]} for the senses that were real."""
    if raw is None or mask is None:
        return None
    r2 = {m: raw[m] for m in SENSES if m in raw}
    m2 = {m: mask[m] for m in r2 if m in mask}
    if not r2:
        return None
    e2 = model.encoder_registry.encode(r2, m2)
    return {m: e2[m].reshape(B, model.d_model).detach().clone()
            for m in r2 if m in e2 and mask.get(m) == "real"}


# ------------------------------------------------------------------ analysis
def _delta(ev, d_model, device="cpu"):
    """The live delta: per-sense change, hand-regard gate on vision, slice-norm."""
    B = ev["order"].shape[0]
    z = torch.zeros(B, d_model)
    hv = (ev["hv_b"] & ev["hv_e"]).float().unsqueeze(1)
    parts = []
    for m in SENSES:
        if m in ev["tb"] and m in ev["te"]:
            dpart = ev["te"][m] - ev["tb"][m]
        else:
            dpart = z
        if m == "vision":
            dpart = dpart * hv
        dpart = dpart / (dpart.std(dim=-1, keepdim=True) + 1e-6)
        parts.append(dpart)
    return torch.cat(parts, dim=-1), hv.squeeze(1)


def analyze(events_path: str, report_path: str | None = None):
    from mimo_bridge.lem_model import command_for_encoder
    blob = torch.load(events_path, weights_only=False)
    events, meta = blob["events"], blob["meta"]
    d = meta["d_model"]
    if len(events) < 40:
        raise SystemExit(f"only {len(events)} events - collect longer")

    proj = torch.nn.Sequential(
        torch.nn.Linear(blob["proj_in"], 256), torch.nn.GELU(),
        torch.nn.Linear(256, 3 * d))
    proj.load_state_dict(blob["order_proj"])
    proj.eval()

    B = events[0]["order"].shape[0]
    with torch.no_grad():
        qs = [proj(command_for_encoder(ev["order"]).reshape(B, -1)) for ev in events]
        deltas, hvs = zip(*[_delta(ev, d) for ev in events])

    # ---- 1) estimator validation against the live log, event by event
    bank = []
    agree, live_r, off_r = 0, [], []
    for k, ev in enumerate(events):
        q, delta = qs[k], deltas[k]
        sims = [F.cosine_similarity(q, delta, dim=-1) / 0.2]
        if B > 1:
            for shift in range(1, B):
                sims.append(F.cosine_similarity(q, torch.roll(delta, shift, 0), dim=-1) / 0.2)
        for dd in bank:
            sims.append(F.cosine_similarity(q, dd, dim=-1) / 0.2)
        if len(sims) >= 4:
            logits = torch.stack(sims, dim=1)
            r1 = float((torch.argmax(logits, 1) == 0).float().mean())
            lv = ev["live"].get("rank1")
            if lv is not None:
                off_r.append(r1); live_r.append(float(lv))
                agree += int(abs(r1 - float(lv)) < 1e-6)
        bank.append(delta.detach())
        if len(bank) > 7:
            bank.pop(0)
    n_val = len(live_r)
    validated = n_val > 0 and agree / n_val > 0.95
    out = {"n_events": len(events), "n_bodies": B,
           "estimator_validation": {"n": n_val, "exact_agree_frac": agree / max(1, n_val),
                                    "live_rank1_mean": float(np.mean(live_r)) if live_r else None,
                                    "offline_rank1_mean": float(np.mean(off_r)) if off_r else None,
                                    "validated": bool(validated)}}
    if not validated:
        print(json.dumps(out, indent=2))
        raise SystemExit("estimator does NOT reproduce the live log - controls refused")

    # flatten to per-body rows
    Q = torch.cat(qs)                       # [N*B, 3d]
    D = torch.cat(deltas)                   # [N*B, 3d]
    HV = torch.cat(hvs)                     # [N*B]
    pose_b = torch.cat([ev["ach_b"].reshape(B, -1) for ev in events])
    ORD = torch.cat([ev["order"].reshape(B, -1) for ev in events])
    eid = torch.tensor([k for k in range(len(events)) for _ in range(B)])
    N = Q.shape[0]

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=-1)

    # ---- 2) posture-matched wrong-order control (the F9 confound removed)
    pd = torch.cdist(pose_b, pose_b)
    pd[torch.arange(N), torch.arange(N)] = float("inf")
    same = eid.unsqueeze(0) == eid.unsqueeze(1)
    pd[same] = float("inf")                 # never match within the same instant
    nn_idx = pd.argmin(dim=1)
    true_s = cos(Q, D)
    wrong_s = cos(Q[nn_idx], D)
    out["posture_matched_wrong_order"] = {
        "true_beats_wrong_frac": float((true_s > wrong_s).float().mean()),
        "chance": 0.5,
        "true_cos_mean": float(true_s.mean()), "wrong_cos_mean": float(wrong_s.mean()),
        "mean_posture_dist_used": float(pd[torch.arange(N), nn_idx][torch.isfinite(pd[torch.arange(N), nn_idx])].mean())}

    # ---- 3) order-specificity across matched postures (freshness, deconfounded)
    other_d = cos(Q, D[nn_idx])
    out["posture_matched_other_outcome"] = {
        "own_beats_matched_other_frac": float((true_s > other_d).float().mean()),
        "chance": 0.5}

    # ---- 4) per-sense slice advantages (vs same-instant decoys where B>1)
    sl_out = {}
    for si, name in ((0, "vision"), (1, "tactile"), (2, "proprioception")):
        sl = slice(si * d, (si + 1) * d)
        a_t = cos(Q[:, sl], D[:, sl])
        if B > 1:
            rolls = []
            for k in range(len(events)):
                q, delta = qs[k], deltas[k]
                rd = torch.stack([cos(q[:, sl], torch.roll(delta, s, 0)[:, sl])
                                  for s in range(1, B)]).mean(0)
                rolls.append(rd)
            a_d = torch.cat(rolls)
            sl_out[name] = {"advantage_mean": float((a_t - a_d).mean()),
                            "n": N}
        else:
            sl_out[name] = {"advantage_mean": None, "n": N}
    sl_out["vision_gate_open_frac"] = float(HV.mean())
    out["slice_advantages"] = sl_out

    # ---- 5) physical ceiling: ridge order -> delta, blocked folds, inner lambda
    X = ORD.numpy().astype(np.float64)
    Y = D.numpy().astype(np.float64)
    ordr = np.argsort(eid.numpy(), kind="stable")
    X, Y = X[ordr], Y[ordr]
    k = 5
    fold = np.repeat(np.arange(k), int(np.ceil(len(X) / k)))[:len(X)]
    lambdas = [1e-2, 1e-1, 1.0, 10.0, 100.0]
    r2s, coss = [], []
    for f in range(k):
        tr, te_ = fold != f, fold == f
        Xt, Yt = X[tr], Y[tr]
        inner = np.arange(len(Xt)) % 5 == 0
        best, best_l = -1e9, lambdas[0]
        for lam in lambdas:
            W = _ridge(Xt[~inner], Yt[~inner], lam)
            r = _r2(Yt[inner], Xt[inner] @ W)
            if r > best:
                best, best_l = r, lam
        W = _ridge(Xt, Yt, best_l)
        P = X[te_] @ W
        r2s.append(_r2(Y[te_], P))
        num = (P * Y[te_]).sum(1)
        den = np.linalg.norm(P, axis=1) * np.linalg.norm(Y[te_], axis=1) + 1e-12
        coss.append(float(np.mean(num / den)))
    out["order_to_delta_ceiling"] = {
        "held_out_r2_mean": float(np.mean(r2s)), "held_out_r2_per_fold": r2s,
        "held_out_cos_mean": float(np.mean(coss)), "folds": k,
        "note": "blocked contiguous folds, inner-validated lambda"}

    # ---- 6) motion columns: hand handles vs all handles
    hand_mv = torch.cat([(ev["ach_e"][:, :12] - ev["ach_b"][:, :12]).norm(dim=-1).mean(1)
                         for ev in events])
    all_mv = torch.cat([(ev["ach_e"] - ev["ach_b"]).norm(dim=-1).mean(1)
                        for ev in events])
    out["motion"] = {"hand_mean": float(hand_mv.mean()), "all_mean": float(all_mv.mean()),
                     "corr_true_score_vs_hand_motion": _corr(true_s, hand_mv),
                     "corr_true_score_vs_all_motion": _corr(true_s, all_mv), "n": N}

    print(json.dumps(out, indent=2))
    if report_path:
        json.dump(out, open(report_path, "w"), indent=2)
        print("report ->", report_path)
    return out


def _ridge(X, Y, lam):
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ Y)


def _r2(Y, P):
    ss_res = float(((Y - P) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = float(a.norm() * b.norm()) + 1e-12
    return float((a * b).sum() / den)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "collect":
        collect(sys.argv[2], sys.argv[3],
                n_steps=int(sys.argv[4]) if len(sys.argv) > 4 else 1800,
                n_envs=int(sys.argv[5]) if len(sys.argv) > 5 else 2,
                seed_env=int(sys.argv[6]) if len(sys.argv) > 6 else 41)
    elif mode == "analyze":
        analyze(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        raise SystemExit(__doc__)
