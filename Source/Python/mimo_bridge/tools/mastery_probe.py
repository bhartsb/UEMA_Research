# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Mastery snapshot probe - the spec's split detector run READ-ONLY over a checkpoint.

Decided 2026-09-06 (gain-of-function tree). The Many Minds split (spec 15.1) fires when
a motor domain's own prediction error is low, flat and consistent. None of that machinery
exists yet, and our logged errors are not on the detector's scale - so this probe measures
the landscape the detector would see, without writing anything into the model:

  1. collect transitions under the run's own regime (its babble, limits, env recipe),
     through the same plumbing the causal test certified (full encode + forward route);
  2. per-token next-frame prediction error, per modality;
  3. discover motor domains: cluster the 23 handles + gaze by co-activation, with the
     link threshold taken from a circular-shift null - no hand-named domains;
  4. infer each domain's sensor tokens: activation<->token-motion correlation beyond the
     same null - no hand-picked senses;
  5. report each domain's PE mean / variance / consistency against BOTH the spec's
     face-value constants (0.05 / 0.01 / 0.95) and lines anchored to the measured floor
     (the best-predicted token's PE), because the spec's constants were never calibrated
     to this implementation's scale.

Anchored verdict = mean below k x floor AND 95% of transitions below k x floor (variance
is reported but folded into consistency, which subsumes it in practice). Nobody declares
mastery here: the output is the distance to the detector's own line.

python -m mimo_bridge.tools.mastery_probe --ckpt <ckpt_dir> [--n 350] [--seed 0]
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

from mimo_bridge.tools.causal_test import (_build_model, _limits_for_run,  # noqa: E402
                                           _collect, _predict)

SPEC_PE, SPEC_VAR, SPEC_CONS = 0.05, 0.01, 0.95
ANCHOR_KS = (2.0, 3.0, 5.0)


def _smooth(x: np.ndarray, w: int = 10) -> np.ndarray:
    if x.shape[0] < w:
        return x.copy()
    k = np.ones(w) / w
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)


def _corr(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """corr(a, each column of B); zero-variance columns correlate 0."""
    az = (a - a.mean()) / (a.std() + 1e-9)
    Bz = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return (Bz * az[:, None]).mean(0)


def _null_p95_cols(a: np.ndarray, B: np.ndarray, rng, reps: int = 40) -> np.ndarray:
    """Per-column p95 of |corr| under circular time-shifts of a - the measured
    'no relationship' line, instead of an invented constant."""
    n = len(a)
    vals = [np.abs(_corr(np.roll(a, int(rng.integers(n // 4, 3 * n // 4))), B))
            for _ in range(reps)]
    return np.percentile(np.stack(vals), 95, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--limp-n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    t0 = time.time()

    model, a, src_path, found = _build_model(args.ckpt, args.device, args.seed,
                                             intention=False)
    hold = int(a.get("motor_hold") or 5)
    limits, limits_src = _limits_for_run(args.ckpt, model)
    limits["_action_repeat"] = int(a.get("action_repeat") or 5)
    # probe regime: planner OFF + the run's own babble. Matches the trained motion
    # statistics (causal-test precedent) without an unrepresentative cold goal ring.
    babble = float(a.get("babble_noise") or 0.05)

    from mimo_bridge.datasource import SingleEnvSource
    env_kw = dict(scene=args.scene, run_dir=None,
                  vision=True, fovea_size=int(a.get("fovea_size") or 288),
                  fovea_fov=float(a.get("fovea_fov") or 24.0))
    for sk, ek in (("decoder_kp", "kp"), ("joint_damping_scale", "joint_damping_scale"),
                   ("hinge_armature", "hinge_armature"), ("finger_armature", "finger_armature"), ("touch_scale_mult", "touch_scale_mult"),
                   ("caregiver_seconds", "caregiver_seconds"),
                   ("gaze_kp", "gaze_kp"), ("gaze_kd", "gaze_kd")):
        if a.get(sk) is not None:
            env_kw[ek] = a[sk]
    env_kw["cmd_horizon"] = limits["_action_repeat"] * hold
    if a.get("gaze_reflex") is not None:
        env_kw["gaze_reflex"] = a.get("gaze_reflex") == "on"
    if a.get("spontaneous_saccades") is not None:
        env_kw["spontaneous_saccades"] = a.get("spontaneous_saccades") != "off"
    if limits.get("age_months") is not None:
        env_kw["age"] = float(limits["age_months"])
    # the trained WORLD, not a bare probe world: real hearing, critters, mobile and the
    # ring change what the senses carry, and mastery is measured on the lived stream.
    if a.get("audio") is not None:
        env_kw["audio_real"] = a.get("audio") == "real"
    if a.get("critters") is not None:
        env_kw["critters"] = a.get("critters") == "on"
    if a.get("mobile") is not None:
        env_kw["mobile"] = a.get("mobile") == "on"
    for k in ("ring_fov", "ring_size"):
        if a.get(k) is not None:
            env_kw[k] = a[k]
    if a.get("audio") == "real":
        # the trainer marks audio real at build time; without this the batch path
        # fills placeholder audio and the probe would score a sense the model never saw
        model.config["modalities"].setdefault("audio", {})["data_source"] = "real"
    import tempfile
    env_kw["run_dir"] = tempfile.mkdtemp(prefix="mastery_probe_")

    def stream(tag, sd, n, limp=False):
        s = SingleEnvSource(tag=f"_{tag}{args.seed}", seed=sd, **env_kw)
        try:
            return _collect(model, s, n, hold, limp=limp, limits=dict(limits),
                            babble=(0.0 if limp else babble))
        finally:
            s.close()

    print(f"collecting: {args.n} active + {args.limp_n} limp transitions "
          f"(babble {babble}, limits {limits.get('phase_name')} "
          f"rom {limits.get('rom_limit')} str {limits.get('strength_limit')})", flush=True)
    A = stream("mp", 100 + args.seed, args.n)
    L = stream("mplimp", 500 + args.seed, args.limp_n, limp=True)
    if len(A) < 40:
        raise SystemExit(f"only {len(A)} active transitions - not enough to cluster")
    cmd_mags = [tr.get("cmd_mag") for tr in A if tr.get("cmd_mag")]
    print(f"collected {len(A)} active / {len(L)} limp; mean order size "
          f"{np.mean(cmd_mags):.3f} rad", flush=True)

    synth = {m for m in model.active
             if str(model.config["modalities"].get(m, {}).get("data_source", "real")) != "real"}
    mods = [m for m in model.pred_mods if m in A[0]["t"] and m not in synth]

    def token_pe(trs):
        """Per-DIMENSION squared error. Each sense is a single token here, so token
        granularity gives only 4 channels - far too coarse for domain PE. The token's
        dimensions (post-LayerNorm, O(1) scale, comparable) are the sensor channels."""
        pe = {m: [] for m in mods}
        dl = {m: [] for m in mods}
        for tr in trs:
            preds = _predict(model, tr, tr["act"], args.device)
            for m in mods:
                t1 = tr["t1"][m].float()
                p = preds[m].float().cpu().reshape(t1.shape)
                pe[m].append(((p - t1) ** 2).reshape(-1).numpy())
                dl[m].append((t1 - tr["t"][m].float()).abs().reshape(-1).numpy())
        return ({m: np.stack(v) for m, v in pe.items()},
                {m: np.stack(v) for m, v in dl.items()})

    pe, delta = token_pe(A)           # each (T, n_tok_m)
    pe_limp, _ = token_pe(L) if L else ({}, {})

    # ---- the measured floors: what the model achieves on its best-predicted tokens
    tok_means = np.concatenate([pe[m].mean(0) for m in mods])
    floor_active = float(np.percentile(tok_means, 5))
    floor_limp = (float(np.percentile(
        np.concatenate([pe_limp[m].mean(0) for m in mods]), 5)) if pe_limp else None)
    floor = floor_active

    # ---- motor channels: per-handle ACHIEVED motion over time. Commanded magnitudes
    # under babble are independent noise by construction (the exploration is injected
    # per handle), so coordination lives in what the body actually did.
    ach = np.stack([tr["ach"].reshape(-1, tr["ach"].shape[-1]).numpy() for tr in A])
    mag = np.linalg.norm(np.diff(ach, axis=0), axis=-1)          # (T-1, H)
    mag = np.concatenate([mag[:1], mag], axis=0)                 # align to T
    H = mag.shape[1]
    try:
        # the layout's own names for H rows; until 2026-09-23 rows past the 24 legacy names were
        # all labelled "gaze", so in trunk_v1 the lower trunk and the chest read as eyes
        from mimo_bridge.body_layout import names_for_rows
        names = list(names_for_rows(H))
    except Exception:
        names = [f"h{i}" for i in range(H)]
    S = _smooth(mag)
    alive = S.std(0) > 1e-6
    rng = np.random.default_rng(args.seed)

    # ---- domain discovery: co-activation clustering, null-anchored link threshold.
    # Babble drives every handle with a shared arousal wave; the common mode across
    # handles is removed first (smoke run: it fused all 23 handles into one blob), so
    # the clustering sees RELATIVE coordination. A handle that never departs from the
    # common mode ends up a singleton, which is a report, not an error.
    Sc = S - S.mean(axis=1, keepdims=True)
    idx = np.where(alive)[0]
    Z = (Sc[:, idx] - Sc[:, idx].mean(0)) / (Sc[:, idx].std(0) + 1e-9)
    C = np.abs(Z.T @ Z / len(Z))
    # family-wise null (max statistic): with 276 pairs, a per-pair 5% line GUARANTEES
    # ~14 false links, and connected components chain them into one blob - the smoke
    # run's fused body was that artifact. Per shuffle, take the MAX |corr| over all
    # pairs; the p95 of those maxima is the line a single real link must clear.
    null_max = []
    for _ in range(60):
        sh = np.stack([np.roll(Z[:, j], int(rng.integers(len(Z) // 4, 3 * len(Z) // 4)))
                       for j in range(Z.shape[1])], axis=1)
        Cn = np.abs(sh.T @ Z / len(Z))
        null_max.append(float(Cn[np.triu_indices_from(Cn, 1)].max()))
    link_thr = float(np.percentile(null_max, 95))
    adj = C > link_thr
    np.fill_diagonal(adj, False)
    seen, clusters = set(), []
    for s0 in range(len(idx)):
        if s0 in seen:
            continue
        comp, stack = [], [s0]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(int(v) for v in np.where(adj[u])[0] if v not in seen)
        clusters.append(sorted(int(idx[u]) for u in comp))

    # ---- per-domain sensor tokens and PE statistics
    D_all = np.concatenate([delta[m] for m in mods], axis=1)
    pe_all = np.concatenate([pe[m] for m in mods], axis=1)
    mod_of = np.concatenate([[m] * pe[m].shape[1] for m in mods])
    report = {"ckpt": os.path.abspath(args.ckpt), "n_active": len(A), "n_limp": len(L),
              "babble": babble, "mean_order_rad": float(np.mean(cmd_mags)) if cmd_mags else None,
              "floor_active_p5": floor_active, "floor_limp_p5": floor_limp,
              "link_thr_null_p95": link_thr, "spec": {"pe": SPEC_PE, "var": SPEC_VAR,
                                                      "consistency": SPEC_CONS},
              "modality_pe_mean": {m: float(pe[m].mean()) for m in mods},
              "inactive_handles": [names[i] for i in range(H) if not alive[i]],
              "clusters": []}
    print(f"\nfloors: active p5 {floor_active:.5f}"
          + (f"  limp p5 {floor_limp:.5f}" if floor_limp is not None else "")
          + f"   link null p95 {link_thr:.3f}")
    print("modality PE means: " + "  ".join(f"{m} {pe[m].mean():.4f}" for m in mods))
    dens = float(adj.mean())
    print(f"link density {dens:.2f} above the family-wise line (dense = real global "
          f"coordination, sparse = chained noise)")
    print(f"channels: {pe_all.shape[1]} token dimensions across {len(mods)} senses "
          f"(~{0.05 * pe_all.shape[1]:.0f} would couple by chance at the per-channel "
          f"line - domains near that size are noise)")

    print("\nDOMAINS (discovered, not named):")
    for ci, comp in enumerate(sorted(clusters, key=len, reverse=True)):
        s_c = S[:, comp].mean(1)
        p95 = _null_p95_cols(s_c, D_all, rng, reps=100)
        coupled = np.where(np.abs(_corr(s_c, D_all)) > p95)[0]
        row = {"handles": [names[i] for i in comp], "n_sensor_channels": int(len(coupled)),
               "sensor_channels_by_modality": {m: int((mod_of[coupled] == m).sum())
                                               for m in mods}}
        if len(coupled):
            dpe = pe_all[:, coupled].mean(1)
            row.update(pe_mean=float(dpe.mean()), pe_var=float(dpe.var()),
                       floor_ratio=float(dpe.mean() / floor),
                       spec_pass=bool(dpe.mean() < SPEC_PE and dpe.var() < SPEC_VAR
                                      and (dpe < SPEC_PE).mean() > SPEC_CONS),
                       spec_consistency=float((dpe < SPEC_PE).mean()),
                       anchored={f"{k:g}x": {"mean_pass": bool(dpe.mean() < k * floor),
                                             "consistency": float((dpe < k * floor).mean())}
                                 for k in ANCHOR_KS})
        report["clusters"].append(row)
        hs = ",".join(row["handles"][:6]) + ("..." if len(comp) > 6 else "")
        if len(coupled):
            anch = "  ".join(f"{k:g}x:{row['anchored'][f'{k:g}x']['consistency']:.2f}"
                             for k in ANCHOR_KS)
            bym = " ".join(f"{m[:3]}:{row['sensor_channels_by_modality'][m]}"
                           for m in mods if row['sensor_channels_by_modality'][m])
            print(f"  [{ci}] {len(comp)} handles ({hs}) -> {len(coupled)} channels "
                  f"({bym}) | PE {row['pe_mean']:.4f} var {row['pe_var']:.5f} "
                  f"= {row['floor_ratio']:.1f}x floor | spec {'PASS' if row['spec_pass'] else 'no'} "
                  f"(cons {row['spec_consistency']:.2f}) | below-kxfloor {anch}")
        else:
            print(f"  [{ci}] {len(comp)} handles ({hs}) -> no coupled sensor channels "
                  f"beyond the null")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))),
                           "mastery_probe")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"mastery_{os.path.basename(args.ckpt)}_{int(t0)}.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"\nwritten {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
