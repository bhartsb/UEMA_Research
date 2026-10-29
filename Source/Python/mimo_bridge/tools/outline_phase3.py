# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Write outline section 14 (Phase 3 experiments) from experiments/results/results.json."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(SRC, "..", ".."))
RES = os.path.join(REPO, "Saved", "mimo_bridge", "experiments", "results", "results.json")
sys.path.insert(0, SRC)
from mimo_bridge.tools.outline_upsert import upsert  # noqa: E402


def f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def main():
    r = json.load(open(RES, encoding="utf-8"))
    E1, E2, E3, E3b, E4, E5 = (r.get(k, {}) for k in ("E1", "E2", "E3", "E3b", "E4", "E5"))
    rows = []
    pf = E1.get("pe_first10pct", {}); pl = E1.get("pe_last10pct", {}); sm = E1.get("token_temporal_std_min_over_run", {}) or {}
    rows.append(("E1", "Does the model train at all on a different body? (1 body, reach, 60 min, lite model)",
                 f"ran {f(E1.get('elapsed_min'),3)} min, {E1.get('steps')} steps, NaN steps {E1.get('nan_steps')}; PE first 10% -> last 10%: "
                 + ", ".join(f"{m} {f(pf.get(m))} -> {f(pl.get(m))}" for m in ("vision", "tactile", "proprioception"))
                 + f"; VICReg {f(E1.get('vicreg_last'))} (batch 1 = always 0); token temporal std min " + ", ".join(f"{m} {f(v,3)}" for m, v in sm.items() if m in ('vision','tactile','proprioception')),
                 "PE fell and no NaN: met. 'VICReg healthy' cannot be read at batch 1. The PE of 1e-4 says the tokens became almost perfectly predictable (low-variance), the thing VICReg normally pushes against; with 4 bodies (E1b) PE stays 0.2 to 0.6 and VICReg is alive."))
    arms = E2.get("arms", {}); pr = E2.get("paired", {})
    rows.append(("E2", "Does the developmental curriculum help? curriculum on (6 phases, MIMo age 1..24 months, delays) vs off (static table = neonate for the whole run, age 18, no delays); 5 seeds, 4 bodies, 12 min each",
                 f"mean PE over the second half: on {f(arms.get('curriculum_on',{}).get('pe_second_half_mean'))} vs off {f(arms.get('curriculum_off',{}).get('pe_second_half_mean'))}; seeds where curriculum is lower: {pr.get('seeds_where_curriculum_lower_pe')} of {pr.get('seeds_compared')}; per seed (on / off): "
                 + "; ".join(f"s{p['seed']} {f(p['on'])} / {f(p['off'])}" for p in pr.get("pairs", [])),
                 ("met" if (pr.get('seeds_compared') and pr.get('seeds_where_curriculum_lower_pe', 0) > pr['seeds_compared'] / 2) else "not met") + " (all seeds reported). Note the two arms also differ in how much the body moves (neonate strength 0.02 vs up to 0.9), so PE is not only about the curriculum schedule."))
    for tag, E in (("E3", E3), ("E3b", E3b)):
        w = E.get("windows", {})
        rows.append((tag, "Does the touch encoder learn a body map from self-touch? (selfbody scene, touch + proprio, vision off, 4 bodies" + (", child phase so the body can move)" if tag == "E3b" else ", neonate phase)"),
                     f"{E.get('rows')} per-patch embeddings from the tactile encoder (hook before pooling); silhouette by body region early {f(w.get('early',{}).get('silhouette_by_region'))} -> late {f(w.get('late',{}).get('silhouette_by_region'))}; linear probe 'which region is touched' accuracy early {f(w.get('early',{}).get('linear_probe_region_accuracy'))} -> late {f(w.get('late',{}).get('linear_probe_region_accuracy'))} (chance {f(w.get('late',{}).get('chance'))}); regions seen {E.get('regions_seen')}",
                     f"clusters by region improve: {f(E.get('silhouette_improved'))}; region prediction improves: {f(E.get('probe_accuracy_improved'))}"))
    a4 = E4.get("arms", {})
    rows.append(("E4", "Does the wellness/curiosity gradient reach the motor controller? forward wellness mode, child phase, 4 bodies, 15 min: model with imagination on / off vs random actions",
                 "; ".join(f"{k}: one-step cmd/response cos {f(v.get('cmd_response_cos_mean'),3)} (acc cos {f(v.get('cmd_response_acc_cos_mean'),3)}, size ratio {f(v.get('cmd_response_ratio_mean'),3)}), |wellness grad| at controller {f(v.get('wellness_grad_norm_mean'),3)}, curiosity drive mean {f(v.get('curiosity_drive_mean'),3)} (zero {f(v.get('curiosity_drive_frac_zero'),3)} of the time), motor loss {f(v.get('motor_loss_first10pct'))} -> {f(v.get('motor_loss_last10pct'))}" for k, v in a4.items()),
                 f"commands correlate with the next body change: not met at the one-step scale (cos 0.02 to 0.03; the servo itself follows a held command with cos 0.98, but the model's target changes every step and the body moves ~2 percent of the asked rotation per step). Wellness gradient non-zero at the controller in every arm: {f(all((v.get('wellness_grad_norm_mean') or 0) > 0 for v in a4.values()) if a4 else None)}. Imagination arm lower final motor loss: {f(E4.get('imagination_arm_lower_final_motor_loss'))}, but not comparable: in the imagination-off arm the curiosity drive was 0 all run (mean PE over all modalities incl. synthetic ones stayed above 0.4), which zeroes the forward-wellness motor term (the model's own curiosity gate)."))
    th = E5.get("throughput", {}); st = E5.get("stability") or {}
    rows.append(("E5", "Throughput and infrastructure: N = 1, 4, 16 bodies; longest stable run",
                 "; ".join(f"N={n}: {f(v['model_steps_per_s'],3)} model steps/s = {f(v['env_steps_per_s_total'],3)} body steps/s" for n, v in sorted(th.items(), key=lambda kv: int(kv[0]))) + f"; stability run: {f(st.get('elapsed_min'),3)} min, {st.get('steps')} steps, ok {f(st.get('ok'))}, NaN {st.get('nan_steps')}, {len(st.get('checkpoints') or [])} checkpoints",
                 "throughput measured: met (the model step, about 130 ms, is the limit; bodies add almost nothing up to 16). 8 hours stable: not met in one night; the longest run is listed."))
    # re-runs after the eight fixes (2026-08-19 morning)
    E1v2, E3c, E4v3, E5L = (r.get(k, {}) for k in ("E1_v2", "E3c", "E4_v3", "E5_longrun"))
    if E1v2:
        rows.append(("E1 (after fixes)", "1 body, 30 min, VICReg now active at batch 1 (temporal window 8)",
                     "PE first 10% -> last 10%: " + ", ".join(f"{m} {f(E1v2.get('pe_first10pct',{}).get(m))} -> {f(E1v2.get('pe_last10pct',{}).get(m))}" for m in ("vision","tactile","proprioception"))
                     + f"; VICReg {f(E1v2.get('vicreg_first'))} -> {f(E1v2.get('vicreg_last'))}; NaN {E1v2.get('nan_steps')}",
                     f"PE fell: {f(all(E1v2.get('pe_fell',{}).get(m, False) for m in ('vision','tactile','proprioception') if m in E1v2.get('pe_fell',{})))}; VICReg active with one body: {f((E1v2.get('vicreg_last') or 0) > 0)}"))
    if E3c:
        w = E3c.get("windows", {})
        rows.append(("E3 (after fixes)", "self-touch, child phase, touch encoder with patch identity embedding",
                     f"silhouette early {f(w.get('early',{}).get('silhouette_by_region'))} -> late {f(w.get('late',{}).get('silhouette_by_region'))}; probe accuracy early {f(w.get('early',{}).get('linear_probe_region_accuracy'))} -> late {f(w.get('late',{}).get('linear_probe_region_accuracy'))} (chance {f(w.get('late',{}).get('chance'))})",
                     f"clusters improve: {f(E3c.get('silhouette_improved'))}; region prediction improves: {f(E3c.get('probe_accuracy_improved'))}"))
    if E4v3:
        a = E4v3.get("arms", {})
        rows.append(("E4 (after fixes)", "curiosity ignores fake senses; hand position IK; 10-step held-command measure",
                     "; ".join(f"{k}: window cos {f(v.get('cmd_response_window_cos_mean'),3)}, one-step cos {f(v.get('cmd_response_cos_mean'),3)}, curiosity {f(v.get('curiosity_drive_mean'),3)} (zero {f(v.get('curiosity_drive_frac_zero'),3)}), motor loss {f(v.get('motor_loss_first10pct'))} -> {f(v.get('motor_loss_last10pct'))}" for k, v in a.items()),
                     f"commands correlate over a 10-step window (cos > 0.3, model arms): {f(all((v.get('cmd_response_window_cos_mean') or 0) > 0.3 for k, v in a.items() if k.startswith('model')) if a else None)}; imagination arm lower final motor loss: {f(E4v3.get('imagination_arm_lower_final_motor_loss'))}"))
    if E5L:
        st2 = E5L.get("stability") or {}
        rows.append(("E5 long run", "as many bodies as the laptop carries, 8 hours planned, hourly snapshots",
                     f"{f(st2.get('elapsed_min'),3)} min, {st2.get('steps')} steps, ok {f(st2.get('ok'))}, NaN {st2.get('nan_steps')}, {len(st2.get('checkpoints') or [])} snapshots",
                     f"8 hours stable: {f((st2.get('elapsed_min') or 0) >= 470 and bool(st2.get('ok')))}"))
    html = ["<h2>14. Phase 3 done: experiments E1 to E5 (numbers from Saved/mimo_bridge/experiments/results/results.json)</h2>",
            "<p>Plain words: each experiment asks one question. 'PE' is the model's prediction error in its own embedding space (lower = it predicts the next moment better). Everything was run on this laptop with the lite model, several runs sharing one GPU.</p>",
            "<table><tr><th>ID</th><th>Question and setup</th><th>Measured</th><th>Criterion</th></tr>"]
    for a, b, c, d in rows:
        html.append(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>")
    html.append("</table>")
    html.append("<p>Figures: <code>docs/mimo_bridge_report/</code> (E1_pe.png, E2_pe.png, E3_touch_pca.png, E4_motor_loss.png, E5_throughput.png). Full report: <code>docs/MIMo_Bridge_Overnight_Report.html</code>.</p>")
    upsert("PHASE3", "\n".join(html), outline=os.path.join(REPO, "docs", "MIMo_Testbed_Implementation_Outline_2026-08-19.html"))
    print("section 14 written")


if __name__ == "__main__":
    main()
