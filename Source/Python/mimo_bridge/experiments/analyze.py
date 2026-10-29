# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Read the stats files the runs wrote and answer each experiment's question with numbers.

Every function returns a plain dict (written to results.json) and, where useful, saves a PNG.
Kept simple on purpose: means over windows, early-vs-late comparisons, all seeds reported.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


def read_jsonl(path: str, rec_type: Optional[str] = None) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec_type is None or r.get("type") == rec_type:
                out.append(r)
    return out


def run_mimo_records(run_dir: str) -> List[dict]:
    return read_jsonl(os.path.join(run_dir, "replay_training_stats.jsonl"), "mimo")


def series(recs: List[dict], key: str, sub: Optional[str] = None):
    xs, ys = [], []
    for r in recs:
        v = r.get(key)
        if sub is not None:
            v = (v or {}).get(sub)
        if v is None:
            continue
        xs.append(r["step"]); ys.append(float(v))
    return np.array(xs), np.array(ys)


def mean_pe_series(recs):
    xs, ys = [], []
    for r in recs:
        pe = r.get("pe") or {}
        real = {k: v for k, v in pe.items() if k not in ("audio", "intermind")}   # synthetic modalities excluded
        if real:
            xs.append(r["step"]); ys.append(float(np.mean(list(real.values()))))
    return np.array(xs), np.array(ys)


def window_mean(xs, ys, lo_frac, hi_frac):
    if len(xs) == 0:
        return float("nan")
    n = len(xs)
    a, b = int(n * lo_frac), max(int(n * hi_frac), int(n * lo_frac) + 1)
    return float(np.mean(ys[a:b]))


def plot_lines(curves: Dict[str, tuple], title: str, ylabel: str, path: str, logy: bool = False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    for name, (x, y) in curves.items():
        if len(x):
            ax.plot(x, y, label=name, linewidth=1.2)
    ax.set_title(title); ax.set_xlabel("model step"); ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


# ------------------------------------------------------------------ E1
def analyze_e1(run_dir: str, fig_dir: str) -> dict:
    recs = run_mimo_records(run_dir)
    summ = json.load(open(os.path.join(run_dir, "summary.json"))) if os.path.exists(os.path.join(run_dir, "summary.json")) else {}
    out = {"run_dir": run_dir, "ok": summ.get("ok"), "steps": summ.get("steps"), "elapsed_min": (summ.get("elapsed_s") or 0) / 60,
           "nan_steps": int(sum(r.get("nan_steps", 0) for r in recs)), "pe_first10pct": {}, "pe_last10pct": {},
           "pe_fell": {}, "token_temporal_std_last": recs[-1].get("token_temporal_std") if recs else None,
           "vicreg_first": None, "vicreg_last": None}
    mods = sorted({k for r in recs for k in (r.get("pe") or {})})
    curves = {}
    for m in mods:
        xs, ys = series(recs, "pe", m)
        out["pe_first10pct"][m] = window_mean(xs, ys, 0.0, 0.1)
        out["pe_last10pct"][m] = window_mean(xs, ys, 0.9, 1.0)
        out["pe_fell"][m] = bool(out["pe_last10pct"][m] < out["pe_first10pct"][m])
        curves[m] = (xs, ys)
    xs, vs = series(recs, "vicreg")
    if len(xs):
        out["vicreg_first"], out["vicreg_last"] = float(vs[0]), float(vs[-1])
    stds = [r.get("token_temporal_std") or {} for r in recs]
    out["token_temporal_std_min_over_run"] = {m: min((s.get(m, 9) for s in stds if m in s), default=None) for m in mods}
    if curves:
        out["figure"] = plot_lines(curves, "E1: prediction error per modality (1 env, reach)", "MSE in embedding space",
                                   os.path.join(fig_dir, "E1_pe.png"), logy=True)
    return out


# ------------------------------------------------------------------ E2
def analyze_e2(runs: Dict[str, Dict[int, str]], fig_dir: str, threshold: float = 0.02) -> dict:
    """runs = {"curriculum_on": {seed: run_dir}, "curriculum_off": {seed: run_dir}}"""
    out = {"arms": {}, "threshold": threshold, "per_seed": []}
    curves = {}
    for arm, seeds in runs.items():
        arm_res = {}
        for seed, rd in sorted(seeds.items()):
            recs = run_mimo_records(rd)
            xs, ys = mean_pe_series(recs)
            if len(xs) == 0:
                continue
            hit = xs[ys < threshold]
            res = {"seed": seed, "arm": arm, "steps": int(xs[-1]) if len(xs) else 0,
                   "pe_first_quarter": window_mean(xs, ys, 0.0, 0.25), "pe_second_half": window_mean(xs, ys, 0.5, 1.0),
                   "pe_final": float(ys[-1]), "steps_to_threshold": int(hit[0]) if len(hit) else None,
                   "auc_pe": float(np.trapezoid(ys, xs) / max(1, xs[-1] - xs[0])) if len(xs) > 1 else float("nan")}
            arm_res[seed] = res
            out["per_seed"].append(res)
            curves[f"{arm} s{seed}"] = (xs, ys)
        if arm_res:
            out["arms"][arm] = {
                "n_seeds": len(arm_res),
                "pe_second_half_mean": float(np.mean([r["pe_second_half"] for r in arm_res.values()])),
                "pe_second_half_per_seed": {s: r["pe_second_half"] for s, r in arm_res.items()},
                "auc_mean": float(np.mean([r["auc_pe"] for r in arm_res.values()])),
                "steps_to_threshold": {s: r["steps_to_threshold"] for s, r in arm_res.items()},
            }
    # paired comparison per seed
    if "curriculum_on" in out["arms"] and "curriculum_off" in out["arms"]:
        wins = 0; total = 0; pairs = []
        for s, ron in out["arms"]["curriculum_on"]["pe_second_half_per_seed"].items():
            roff = out["arms"]["curriculum_off"]["pe_second_half_per_seed"].get(s)
            if roff is None:
                continue
            total += 1; wins += int(ron < roff)
            pairs.append({"seed": s, "on": ron, "off": roff, "curriculum_lower": bool(ron < roff)})
        out["paired"] = {"seeds_where_curriculum_lower_pe": wins, "seeds_compared": total, "pairs": pairs}
    out["figure"] = plot_lines(curves, "E2: mean real-modality PE, curriculum on vs off (all seeds)", "mean PE",
                               os.path.join(fig_dir, "E2_pe.png"), logy=True)
    return out


# ------------------------------------------------------------------ E3
def analyze_e3(run_dir: str, fig_dir: str) -> dict:
    p = os.path.join(run_dir, "touch_probe.npz")
    out = {"run_dir": run_dir, "probe_file": p}
    if not os.path.exists(p):
        out["error"] = "no touch_probe.npz"
        return out
    z = np.load(p, allow_pickle=True)
    steps, idx, emb = z["steps"], z["patch_index"], z["emb"].astype(np.float32)
    regions = z["regions"]
    labels = regions[idx]
    out["rows"] = int(len(steps)); out["steps_range"] = [int(steps.min()), int(steps.max())]
    out["regions_seen"] = {str(r): int(c) for r, c in zip(*np.unique(labels, return_counts=True))}
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    # drop regions with fewer than 10 samples (cannot be cross-validated or clustered)
    u, c = np.unique(labels, return_counts=True)
    keep = np.isin(labels, u[c >= 10])
    steps, idx, emb, labels = steps[keep], idx[keep], emb[keep], labels[keep]
    n = len(steps)
    order = np.argsort(steps, kind="stable")
    steps, idx, emb, labels = steps[order], idx[order], emb[order], labels[order]
    thirds = {"early": slice(0, n // 3), "middle": slice(n // 3, 2 * n // 3), "late": slice(2 * n // 3, n)}
    out["windows"] = {}
    for name, sl in thirds.items():
        e, l = emb[sl], labels[sl]
        res = {"rows": int(len(e)), "n_regions": int(len(np.unique(l)))}
        if len(np.unique(l)) >= 2 and len(e) >= 20:
            try:
                res["silhouette_by_region"] = float(silhouette_score(e, l, sample_size=min(2000, len(e)), random_state=0))
            except Exception as ex:
                res["silhouette_error"] = str(ex)
            try:
                clf = LogisticRegression(max_iter=500)
                uu, cc = np.unique(l, return_counts=True)
                m2 = np.isin(l, uu[cc >= 5])
                e, l = e[m2], l[m2]
                k = min(5, int(min(np.unique(l, return_counts=True)[1])))
                if k >= 2 and len(np.unique(l)) >= 2:
                    sc = cross_val_score(clf, e, l, cv=k)
                    res["linear_probe_region_accuracy"] = float(sc.mean())
                    res["chance"] = float(max(np.unique(l, return_counts=True)[1]) / len(l))
            except Exception as ex:
                res["probe_error"] = str(ex)
        out["windows"][name] = res
    # region-prediction improves over time? compare early vs late probe accuracy
    e_acc = out["windows"]["early"].get("linear_probe_region_accuracy")
    l_acc = out["windows"]["late"].get("linear_probe_region_accuracy")
    out["probe_accuracy_improved"] = (None if e_acc is None or l_acc is None else bool(l_acc > e_acc))
    e_s = out["windows"]["early"].get("silhouette_by_region"); l_s = out["windows"]["late"].get("silhouette_by_region")
    out["silhouette_improved"] = (None if e_s is None or l_s is None else bool(l_s > e_s))
    # 2-D picture of late embeddings by region (PCA)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        e, l = emb[thirds["late"]], labels[thirds["late"]]
        if len(e) > 10:
            pc = PCA(n_components=2).fit_transform(e)
            fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
            for r in np.unique(l):
                m = l == r
                ax.scatter(pc[m, 0], pc[m, 1], s=6, label=str(r), alpha=0.6)
            ax.set_title("E3: late tactile per-patch embeddings (PCA), colour = body region")
            ax.legend(fontsize=7, markerscale=2); fig.tight_layout()
            fp = os.path.join(fig_dir, "E3_touch_pca.png"); fig.savefig(fp); plt.close(fig); out["figure"] = fp
    except Exception as ex:
        out["figure_error"] = str(ex)
    return out


# ------------------------------------------------------------------ E4
def analyze_e4(runs: Dict[str, str], fig_dir: str) -> dict:
    """runs = {"model_imag_on": dir, "model_imag_off": dir, "random_actions": dir}"""
    out = {"arms": {}}
    curves_motor = {}
    for arm, rd in runs.items():
        recs = run_mimo_records(rd)
        trecs = read_jsonl(os.path.join(rd, "replay_training_stats.jsonl"), "training")
        xs_c, cos = series(recs, "cmd_response_cos")
        xs_r, ratio = series(recs, "cmd_response_ratio")
        xs_w, wg = series(recs, "wellness_grad_norm")
        xs_a, acc = series(recs, "cmd_response_acc_cos")
        xs_wc, wcos = series(recs, "cmd_response_window_cos")
        xs_d, drive = series(recs, "curiosity_drive")
        pe_all = [float(np.mean(list((r.get("pe") or {}).values()))) for r in recs if r.get("pe")]
        # motor loss from the training records' loss_breakdown
        mx, my = [], []
        for r in trecs:
            bd = r.get("loss_breakdown") or {}
            if "motor" in bd:
                mx.append(r["training_step"]); my.append(float(bd["motor"]))
        mx, my = np.array(mx), np.array(my)
        out["arms"][arm] = {
            "cmd_response_cos_mean": float(np.mean(cos)) if len(cos) else None,
            "cmd_response_ratio_mean": float(np.mean(ratio)) if len(ratio) else None,
            "cmd_response_acc_cos_mean": float(np.mean(acc)) if len(acc) else None,
            "cmd_response_window_cos_mean": float(np.mean(wcos)) if len(wcos) else None,
            "curiosity_drive_mean": float(np.mean(drive)) if len(drive) else None,
            "curiosity_drive_frac_zero": float(np.mean(drive == 0)) if len(drive) else None,
            "mean_pe_all_modalities": float(np.mean(pe_all)) if pe_all else None,
            "wellness_grad_norm_mean": float(np.mean(wg)) if len(wg) else None,
            "motor_loss_first10pct": window_mean(mx, my, 0, 0.1), "motor_loss_last10pct": window_mean(mx, my, 0.9, 1.0),
            "steps": int(mx[-1]) if len(mx) else 0,
        }
        curves_motor[arm] = (mx, my)
    a, b = out["arms"].get("model_imag_on"), out["arms"].get("model_imag_off")
    if a and b and a["motor_loss_last10pct"] == a["motor_loss_last10pct"] and b["motor_loss_last10pct"] == b["motor_loss_last10pct"]:
        out["imagination_arm_lower_final_motor_loss"] = bool(a["motor_loss_last10pct"] < b["motor_loss_last10pct"])
    out["figure"] = plot_lines(curves_motor, "E4: motor loss (model vs random actions, imagination on/off)", "motor loss",
                               os.path.join(fig_dir, "E4_motor_loss.png"), logy=True)
    return out


# ------------------------------------------------------------------ E5
def analyze_e5(runs: Dict[int, str], stability_dir: Optional[str], fig_dir: str) -> dict:
    out = {"throughput": {}, "stability": None}
    for n, rd in sorted(runs.items()):
        recs = run_mimo_records(rd)
        if not recs:
            continue
        tail = recs[len(recs) // 3:]   # skip warm-up
        out["throughput"][n] = {
            "model_steps_per_s": float(np.mean([r["model_steps_per_s"] for r in tail])),
            "env_steps_per_s_total": float(np.mean([r["env_steps_per_s"] for r in tail])),
        }
    if stability_dir:
        recs = run_mimo_records(stability_dir)
        summ = json.load(open(os.path.join(stability_dir, "summary.json"))) if os.path.exists(os.path.join(stability_dir, "summary.json")) else {}
        out["stability"] = {"ok": summ.get("ok"), "steps": summ.get("steps"), "elapsed_min": (summ.get("elapsed_s") or 0) / 60,
                            "nan_steps": int(sum(r.get("nan_steps", 0) for r in recs)),
                            "checkpoints": summ.get("checkpoints"), "exception": summ.get("exception")}
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ns = sorted(out["throughput"]);
        if ns:
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
            ax.plot(ns, [out["throughput"][n]["env_steps_per_s_total"] for n in ns], "o-", label="env steps/s (all bodies)")
            ax.plot(ns, [out["throughput"][n]["model_steps_per_s"] for n in ns], "s--", label="model steps/s")
            ax.set_xlabel("N envs"); ax.set_ylabel("steps / s"); ax.set_xscale("log", base=2); ax.grid(alpha=0.3); ax.legend()
            ax.set_title("E5: throughput vs N"); fig.tight_layout()
            fp = os.path.join(fig_dir, "E5_throughput.png"); fig.savefig(fp); plt.close(fig); out["figure"] = fp
    except Exception as ex:
        out["figure_error"] = str(ex)
    return out
