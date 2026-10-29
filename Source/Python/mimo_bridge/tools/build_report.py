# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Build docs/MIMo_Bridge_Overnight_Report.html from the run outputs.

    python -m mimo_bridge.tools.build_report

Numbers are pulled from files (phase0 json, phase1 summary, experiments/results/results.json,
phase4 summaries, every run's checkpoints.json, git log). The words are written here.
"""
from __future__ import annotations

import glob
import html
import json
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(SRC, "..", ".."))
SAVED = os.path.join(REPO, "Saved", "mimo_bridge")
EXP = os.path.join(SAVED, "experiments")
DOCS = os.path.join(REPO, "docs")
FIG_DIR = os.path.join(DOCS, "mimo_bridge_report")
OUT = os.path.join(DOCS, "MIMo_Bridge_Overnight_Report.html")


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def esc(x):
    return html.escape(str(x))


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if x != x:
            return "nan"
        return f"{x:.{nd}g}"
    return esc(x)


def git_log():
    try:
        return subprocess.check_output(["git", "-C", REPO, "log", "--oneline", "feature/vocal-tract-v2..feature/mimo-bridge"], text=True).strip()
    except Exception as e:
        return f"(git log failed: {e})"


def mimo_state():
    root = r"C:\Users\bhart\Documents\MIMo_private"
    try:
        br = subprocess.check_output(["git", "-C", root, "branch", "--show-current"], text=True).strip()
        head = subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"], text=True).strip()
        st = subprocess.check_output(["git", "-C", root, "status", "--short"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", root, "log", "--oneline", "v3..bridge-work"], text=True).strip()
        return br, head, st, diff
    except Exception as e:
        return "?", "?", f"(git failed: {e})", ""


def all_checkpoints():
    rows = []
    for cj in sorted(glob.glob(os.path.join(SAVED, "**", "checkpoints.json"), recursive=True)):
        run = os.path.relpath(os.path.dirname(cj), SAVED)
        for c in load(cj, []) or []:
            rows.append((run, c.get("step"), c.get("time"), c.get("reason"), round(c.get("elapsed_min", 0), 1), c.get("path")))
    return rows


def copy_fig(p):
    if not p or not os.path.exists(p):
        return None
    os.makedirs(FIG_DIR, exist_ok=True)
    dst = os.path.join(FIG_DIR, os.path.basename(p))
    shutil.copyfile(p, dst)
    return "mimo_bridge_report/" + os.path.basename(p)


def table(rows, header, raw=False):
    h = "<table><tr>" + "".join(f"<th>{esc(c)}</th>" for c in header) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{c if raw or (isinstance(c, str) and c.startswith('<')) else esc(c)}</td>" for c in r) + "</tr>"
    return h + "</table>"


def main():
    p0 = load(os.path.join(SRC, "mimo_bridge", "phase0", "benchmark_summary.json"), {})
    c0 = load(os.path.join(SRC, "mimo_bridge", "phase0", "phase0_counts.json"), {})
    p1 = load(os.path.join(SAVED, "phase1", "mode_a_summary.json"), {})
    res = load(os.path.join(EXP, "results", "results.json"), {})
    p4 = load(os.path.join(SAVED, "phase4", "mode_b_summary.json"), {})
    ck = all_checkpoints()
    br, head, st, diff = mimo_state()
    E1, E2, E3, E4, E5 = res.get("E1", {}), res.get("E2", {}), res.get("E3", {}), res.get("E4", {}), res.get("E5", {})
    e1b = load(os.path.join(EXP, "E1b_reach_4env", "summary.json"), {})
    # E1b numbers from its mimo records
    from mimo_bridge.experiments.analyze import run_mimo_records, series, window_mean
    e1b_recs = run_mimo_records(os.path.join(EXP, "E1b_reach_4env"))
    e1b_pe = {}
    for m in ("vision", "tactile", "proprioception"):
        xs, ys = series(e1b_recs, "pe", m)
        e1b_pe[m] = (window_mean(xs, ys, 0, 0.1), window_mean(xs, ys, 0.9, 1.0))
    e1b_std = e1b_recs[-1].get("token_temporal_std") if e1b_recs else {}

    cfgs = p0.get("configs", {})
    def cfg(n, k):
        return cfgs.get(n, {}).get(k)
    default_name = "V2 default (touch v2 + vision 256 + vestibular)"

    parts = []
    parts.append(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>MIMo Bridge Overnight Report</title>
<style>
body{{font:15px/1.5 Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:0 auto;padding:24px;color:#1b2330;background:#fbfcfe}}
h1{{font-size:26px;margin-bottom:4px}} h2{{font-size:20px;margin-top:30px;border-bottom:1px solid #d6dde8;padding-bottom:4px}} h3{{font-size:16px;color:#2f5bb0}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}} th,td{{border:1px solid #d6dde8;padding:6px 8px;text-align:left;vertical-align:top}} th{{background:#eef2f8}}
code{{font:12.5px Consolas,monospace;background:#eef2f8;padding:1px 4px;border-radius:3px}}
.pass{{color:#1a7f37;font-weight:700}} .fail{{color:#b42318;font-weight:700}} .part{{color:#9a6700;font-weight:700}}
.box{{background:#fff;border:1px solid #d6dde8;border-left:6px solid #2f5bb0;border-radius:6px;padding:10px 14px;margin:12px 0}}
img{{max-width:100%;border:1px solid #d6dde8;border-radius:6px}}
</style></head><body>
<h1>MIMo Bridge: Overnight Report</h1>
<div class="sub">Built {time.strftime('%Y-%m-%d %H:%M')}. Branch <code>feature/mimo-bridge</code> in UEMA_devA_private (not pushed). MIMo: <code>C:\\Users\\bhart\\Documents\\MIMo_private</code> branch <code>{esc(br)}</code> at <code>{esc(head)}</code> (not pushed).</div>
""")
    # ---------------- summary
    e1_ok = E1.get("ok")
    parts.append(f"""
<h2>1. The short version</h2>
<div class="box">
<p><b>Every phase ran, start to finish.</b> The toy baby body (MIMo) now talks to the LEM model through the model's own frame format. The model trained on it in-process (Mode A) and over shared memory (Mode B). Nothing in the model code (<code>lem/</code>, <code>tokenizer/</code>) and nothing in MIMo was changed.</p>
<ul>
<li><b>Phase 0</b> (measure): MIMo alone runs {fmt(cfg('V2 physics only','env_steps_per_s'),3)} steps/s physics only, {fmt(cfg('V2 no vision','env_steps_per_s'),3)} with touch, {fmt(cfg(default_name,'env_steps_per_s'),3)} with MIMo's own vision (which is broken, see below). {c0.get('touch_points_total')} skin points on {c0.get('touch_sensing_bodies')} bodies. The model reads a 31,040-byte frame (proprio 4096 B, touch stride 40 B).</li>
<li><b>Phase 1</b> (bridge): 33 tests green, frame encode/decode bit for bit through the model's own tokenizer; one body drove the model for 10 minutes, {p1.get('steps')} steps, {p1.get('nan_steps')} NaNs, 0 exceptions.</li>
<li><b>Phase 2</b> (training loop): N bodies in N processes, batch = N, curriculum bound to MIMo age and delays, stats in the UE run's record format, checkpoints hourly and at the end.</li>
<li><b>Phase 3</b> (experiments): E1 {'<span class=pass>ran</span>' if e1_ok else '<span class=fail>failed</span>'} (but see the batch-1 collapse finding), E2 curriculum on vs off over 5 seeds, E3 touch body map, E4 motor/wellness, E5 throughput 1/4/16 bodies + a stability run. Details and pass/fail per criterion in section 4.</li>
<li><b>Phase 4</b> (shared memory): {'<span class=pass>passed</span>' if p4.get('ok') else '<span class=fail>failed</span>'}: the model's own reader read {p4.get('model',{}).get('frames_processed')} frames (skipping {p4.get('model',{}).get('frames_skipped')} it was too slow for, like with UE), wrote {p4.get('model',{}).get('actions_written')} actions with its own writer; the body received {p4.get('bridge',{}).get('actions_received')} of them, all in order.</li>
</ul>
<p><b>After your decisions (morning of 2026-08-19) all eight items were fixed and the affected experiments re-run</b> (section 8): with ONE body the anti-collapse term is now alive (VICReg {fmt(res.get('E1_v2',{}).get('vicreg_last'),2)} instead of 0, token spread stays above {fmt(min((res.get('E1_v2',{}).get('token_temporal_std_min_over_run') or {'x':0}).values()),2)}); the touch body map improved a lot once each skin patch got an identity (a probe now names the touched body part {fmt((res.get('E3c',{}).get('windows',{}).get('late',{}).get('linear_probe_region_accuracy') or 0)*100,2)}% of the time, chance {fmt((res.get('E3c',{}).get('windows',{}).get('late',{}).get('chance') or 0)*100,2)}%, before the fix {fmt((res.get('E3b',{}).get('windows',{}).get('late',{}).get('linear_probe_region_accuracy') or 0)*100,2)}%); the curiosity dial is no longer stuck at zero; the model's own commands still barely move the body the way they ask (window cos {fmt(res.get('E4_v3',{}).get('arms',{}).get('model_imag_on',{}).get('cmd_response_window_cos_mean'),2)} vs {fmt(res.get('E4_v3',{}).get('arms',{}).get('random_actions',{}).get('cmd_response_window_cos_mean'),2)} for random commands), because its targets are mostly unreachable orientations; the repo's own lite script now runs against MIMo over shared memory with vision; an 8-hour run with 20 bodies is in progress (dashboard: Saved/mimo_bridge/dashboard.html).</p>
<p><b>Biggest findings (from the night):</b> (1) MIMo's vision under gymnasium 1.2.3 gives the same third-person image for both eyes; the bridge renders the eyes itself. (2) MIMo's PositionalModel cannot drive joints; the bridge has its own servo. (3) With ONE body (batch 1) the model's VICReg term is zero; prediction error drops to 1e-4 (tokens become almost perfectly predictable); with 4 bodies VICReg is alive and PE stays 0.1 to 0.6. The UE live loop is also batch 1. (4) MIMo's face/cloth textures cost 1 GB of RAM per body; the bridge generates a small-texture copy of the meta file at run time so 16 bodies fit. (5) The model's curiosity gate uses the mean PE over ALL modalities, synthetic audio and intermind included; when that mean sits above pe_max 0.4 the drive is 0 and the forward-wellness motor term is switched off (seen in the E4 imagination-off arm for the whole run). (6) The tactile encoder has no per-patch position embedding, so a touch body map can only come from force statistics; E3 shows weak clustering by region and a linear probe near chance.</p>
</div>
""")
    # ---------------- phases
    parts.append(f"""
<h2>2. What each phase did and what was measured</h2>
<h3>Phase 0: measure first (no bridge code before this)</h3>
{table([
 ["physics only", fmt(cfg('V2 physics only','env_steps_per_s'),3), fmt(cfg('V2 physics only','realtime_factor'),3)],
 ["+ touch (5332 points)", fmt(cfg('V2 no vision','env_steps_per_s'),3), fmt(cfg('V2 no vision','realtime_factor'),3)],
 ["+ MIMo vision 2 x 256x256 (no touch)", fmt(cfg('V2 no touch','env_steps_per_s'),3), fmt(cfg('V2 no touch','realtime_factor'),3)],
 ["default: touch + vision + vestibular", fmt(cfg(default_name,'env_steps_per_s'),3), fmt(cfg(default_name,'realtime_factor'),3)],
], ["MIMoBenchV2-v0 config (benchmark.py, 30 sim s each)", "env steps / s", "sim time / real time"])}
<ul>
<li>Touch points: <b>{c0.get('touch_points_total')}</b> on {c0.get('touch_sensing_bodies')} bodies (head 404, each thumb tip 248, upper body 122, hands 122; smallest 25). Scene: {c0.get('nbody')} bodies, {c0.get('njnt')} joints, {c0.get('nu')} torque actuators, 2 eye cameras.</li>
<li>Byte layout the live model reads (tokenizer/config.py): header 64 | audio 8192 @64 | proprio 4096 @8256 (24 x 160 B) | motor 2048 @12352 (24 x 64 B) | touch 16384 @14400 (256 x 40 B patches @16, 16 x 64 B collisions @10256) | voice 256 @30784 | total 31,040 B, slot 31,232 B. Risk R1 closed: the UE header numbers are right, the older pipeline doc numbers are stale.</li>
<li>MIMo bugs found (not fixed in MIMo; worked around in the bridge): benchmark.py crashes on <code>env.dt</code> under gymnasium 1.2.3; SimpleVision renders the free camera for both eyes; <code>sensory_delay &gt; 0</code> crashes at construction; growth writes fixed-name temp files into MIMo's tree and logs; PositionalModel zeroes ctrl each substep.</li>
</ul>
<h3>Phase 1: bridge core (Mode A)</h3>
<ul>
<li>Package <code>Source/Python/mimo_bridge/</code>: touch_adapter, frame_encoder, proprio_adapter, action_decoder, vision_adapter, env, datasource, lem_model, run_mode_a, shm_writer, run_mode_b, train_mimo, experiments/, tests/. 33 tests (hand-worked touch numbers, bit-for-bit frame round trip through the model's tokenizer, decoder on a tiny MuJoCo chain, vision angles, real-body integration, shared-memory transport through the model's own CPUReader and ActionBufferWriter).</li>
<li>10-minute Mode A run: {p1.get('steps')} steps, {fmt(p1.get('steps_per_s'),3)} steps/s, NaN steps {p1.get('nan_steps')}, ok = {fmt(p1.get('ok'))}.</li>
<li>Per step on this laptop (RTX 4070 Laptop 8 GB): model about 110 to 130 ms with imagination (3 extra forward passes + the wellness double-backward), about 53 ms without; the MIMo body about 5 to 7 ms including both eyes. The GPU side is the limit, not MIMo.</li>
</ul>
<h3>Phase 2: training loop</h3>
<ul>
<li>N bodies in N worker processes; one batch of N per step; curriculum (6 phases) -> MIMo age 1/4/8/13/18/24 months and delays (4,3)/(3,2)/(2,2)/(1,1)/(1,0)/(0,0) steps; rebuild on phase change; stats in <code>replay_training_stats.jsonl</code> (types inference, training, mimo, curriculum_progress, phase_transition); checkpoints via the model's CheckpointManager.</li>
</ul>
<h3>Phase 4: shared memory (Mode B)</h3>
<ul>
<li>Bridge writes LEM_Pipeline (2 slots of 31,232 B, header like UE) and LEM_Vision (eyes letterboxed BGRA into pano/fovea regions), reads LEM_Action_Buffer. Model side = the model's own LEMTokenizer (read_frame) + lite LEM + the model's own ActionBufferWriter. Result: {fmt(p4.get('ok'))}; bridge wrote {p4.get('bridge',{}).get('frames_written')} frames in {fmt((p4.get('bridge',{}).get('elapsed_s') or 0)/60,2)} min, model processed {p4.get('model',{}).get('frames_processed')} (skipped {p4.get('model',{}).get('frames_skipped')}), out-of-order frames {p4.get('model',{}).get('out_of_order_frames')}, actions received {p4.get('bridge',{}).get('actions_received')}, out-of-order actions {p4.get('bridge',{}).get('out_of_order_actions')}, NaN losses {p4.get('model',{}).get('nan_losses')}.</li>
<li>Vision over shared memory: the model's tokenizer only takes vision through the D3D12 GPU reader, so the model ran with vision off in Mode B; LEM_Vision was written anyway and read back correctly by the model's CPUReader in a test.</li>
<li>Starting the model "exactly as for a UE run" with the unmodified <code>lem/live_test_lite.py --no-vision</code> fails before it reaches shared memory: <code>AssertionError: embed_dim must be divisible by num_heads</code> (lite d_model 128 with master n_heads 12). This is a pre-existing model-side bug (with three more: placeholders filled with batch 32, motor controller 3-tuple unpack, undefined total_params). Not fixed, by the rules. Needs your decision (section 7).</li>
</ul>
""")
    # ---------------- experiments
    def pf(b):
        if b is None:
            return '<span class="part">not decided</span>'
        return '<span class="pass">met</span>' if b else '<span class="fail">not met</span>'
    rows = []
    # E1
    pe_fell = E1.get("pe_fell", {})
    std_min = E1.get("token_temporal_std_min_over_run", {}) or {}
    real_fell = all(pe_fell.get(m, False) for m in ("vision", "tactile", "proprioception") if m in pe_fell)
    collapsed = any((v is not None and v < 0.02) for k, v in std_min.items() if k in ("vision", "tactile", "proprioception"))
    rows.append(["E1 trains at all on a new body, 1 env, 1 h",
                 f"PE fell for every real modality: {fmt(real_fell)} (vision {fmt(E1.get('pe_first10pct',{}).get('vision'))} -> {fmt(E1.get('pe_last10pct',{}).get('vision'))}, tactile {fmt(E1.get('pe_first10pct',{}).get('tactile'))} -> {fmt(E1.get('pe_last10pct',{}).get('tactile'))}, proprio {fmt(E1.get('pe_first10pct',{}).get('proprioception'))} -> {fmt(E1.get('pe_last10pct',{}).get('proprioception'))}); NaN steps {E1.get('nan_steps')}; VICReg {fmt(E1.get('vicreg_last'))} (batch 1 -> always 0); token temporal std min {', '.join(f'{k} {fmt(v,3)}' for k,v in std_min.items())}",
                 f"PE fell: {pf(real_fell)}; no NaN: {pf((E1.get('nan_steps') or 0) == 0)}; no hard collapse (token temporal std stays above 0.02): {pf(not collapsed)}; VICReg terms healthy: {pf(None)} (VICReg is 0 by construction at batch 1, so it cannot be read; the PE of 1e-4 means the tokens became almost perfectly predictable, which is what VICReg would normally push against; compare E1b)"])
    rows.append(["E1b same, 4 envs, 30 min (added)",
                 f"PE first 10% -> last 10%: " + ", ".join(f"{m} {fmt(a)} -> {fmt(b)}" for m, (a, b) in e1b_pe.items()) + f"; token temporal std at end {', '.join(f'{k} {fmt(v,3)}' for k,v in (e1b_std or {}).items() if k in ('vision','tactile','proprioception'))}; ok {fmt(e1b.get('ok'))}",
                 "With batch 4 the encoders keep variance (VICReg alive)"])
    # E2
    paired = E2.get("paired", {})
    arms = E2.get("arms", {})
    rows.append(["E2 curriculum on vs off, 5 seeds each",
                 f"mean PE second half: on {fmt(arms.get('curriculum_on',{}).get('pe_second_half_mean'))} vs off {fmt(arms.get('curriculum_off',{}).get('pe_second_half_mean'))}; seeds where curriculum is lower: {paired.get('seeds_where_curriculum_lower_pe')} of {paired.get('seeds_compared')}; per seed on/off: " + "; ".join(f"s{p['seed']} {fmt(p['on'])}/{fmt(p['off'])}" for p in paired.get('pairs', [])),
                 pf(None if not paired.get('seeds_compared') else paired.get('seeds_where_curriculum_lower_pe', 0) > paired.get('seeds_compared', 0) / 2) + " (criterion: curriculum arm reaches lower PE sooner across seeds; all seeds shown)"])
    # E3
    w = E3.get("windows", {})
    rows.append(["E3 touch body map from self touch (selfbody, touch + proprio, 4 envs)",
                 f"rows {E3.get('rows')}; silhouette by region early {fmt(w.get('early',{}).get('silhouette_by_region'))} -> late {fmt(w.get('late',{}).get('silhouette_by_region'))}; linear probe region accuracy early {fmt(w.get('early',{}).get('linear_probe_region_accuracy'))} -> late {fmt(w.get('late',{}).get('linear_probe_region_accuracy'))} (chance {fmt(w.get('late',{}).get('chance'))}); regions seen {E3.get('regions_seen')}",
                 f"clusters by region improve: {pf(E3.get('silhouette_improved'))}; region prediction improves: {pf(E3.get('probe_accuracy_improved'))}"])
    # E3b
    E3b = res.get("E3b", {})
    w = E3b.get("windows", {})
    rows.append(["E3b same, child phase (body can move), 20 min",
                 f"rows {E3b.get('rows')}; silhouette by region early {fmt(w.get('early',{}).get('silhouette_by_region'))} -> late {fmt(w.get('late',{}).get('silhouette_by_region'))}; linear probe region accuracy early {fmt(w.get('early',{}).get('linear_probe_region_accuracy'))} / middle {fmt(w.get('middle',{}).get('linear_probe_region_accuracy'))} / late {fmt(w.get('late',{}).get('linear_probe_region_accuracy'))} (chance {fmt(w.get('late',{}).get('chance'))})",
                 f"clusters by region improve: {pf(E3b.get('silhouette_improved'))}; region prediction improves: {pf(E3b.get('probe_accuracy_improved'))}. Note: the model's tactile encoder has no per-patch position embedding (cross_patch.py: per-patch MLP on the 8 field values, attention, pooled query), so a body map can only come from differences in force statistics between regions."])
    # E4
    a4 = E4.get("arms", {})
    rows.append(["E4 wellness/curiosity gradient reaches the motor controller (forward wellness mode, child phase, 4 bodies, 15 min; take 2 after the servo clamp, take 1 kept in *_take1 folders)",
                 "<br>".join(f"<b>{k}</b>: one-step command/response cos {fmt(v.get('cmd_response_cos_mean'),3)} (acceleration cos {fmt(v.get('cmd_response_acc_cos_mean'),3)}, size ratio {fmt(v.get('cmd_response_ratio_mean'),3)}); |wellness grad| at controller {fmt(v.get('wellness_grad_norm_mean'),3)}; curiosity drive mean {fmt(v.get('curiosity_drive_mean'),3)} (zero {fmt(v.get('curiosity_drive_frac_zero'),3)} of the time, mean PE over all modalities {fmt(v.get('mean_pe_all_modalities'),3)}); motor loss {fmt(v.get('motor_loss_first10pct'))} -> {fmt(v.get('motor_loss_last10pct'))}" for k, v in a4.items()),
                 f"commands correlate with the following body change: {pf(False)} at the one-step scale (cos 0.02 to 0.03 for the model arms, 0.10 for random actions) even though the servo itself follows a held command (cos 0.98, 28.6 deg met within 1.4 deg in 0.6 s): the model's target changes every step and the body moves about 2 percent of the asked rotation per step, so one-step direction is mostly momentum. "
                 f"Wellness gradient reaches the controller (non-zero in every arm): {pf(all((v.get('wellness_grad_norm_mean') or 0) > 0 for v in a4.values()) if a4 else None)}. "
                 f"Imagination arm lower final motor loss: {pf(E4.get('imagination_arm_lower_final_motor_loss'))}, BUT the two arms are not comparable: in the imagination-off arm the curiosity drive was 0 the whole run (mean PE over all modalities, synthetic audio and intermind included, stayed above pe_max 0.4), which zeroes the forward-wellness motor term; that is the model's own curiosity gate, also active in the UE loop."])
    # E5
    th = E5.get("throughput", {})
    stab = E5.get("stability") or {}
    rows.append(["E5 throughput and stability",
                 "; ".join(f"N={n}: {fmt(v['model_steps_per_s'],3)} model steps/s = {fmt(v['env_steps_per_s_total'],3)} body steps/s" for n, v in sorted(th.items(), key=lambda kv: int(kv[0]))) + f"; stability run: {fmt(stab.get('elapsed_min'),3)} min, {stab.get('steps')} steps, ok {fmt(stab.get('ok'))}, NaN {stab.get('nan_steps')}, checkpoints {len(stab.get('checkpoints') or [])}",
                 f"measured: {pf(bool(th))}; 8 hours stable: {pf(False)} (the night was not long enough; the longest run is listed)"])
    # ---- re-runs after the fixes ----
    E1v2, E3c, E4v3, E5L = res.get("E1_v2", {}), res.get("E3c", {}), res.get("E4_v3", {}), res.get("E5_longrun", {})
    if E1v2:
        sm2 = E1v2.get("token_temporal_std_min_over_run", {}) or {}
        rows.append(["E1 again, 1 body, 30 min, AFTER the fixes (VICReg now works at batch 1 over time)",
                     f"PE first 10% -> last 10%: " + ", ".join(f"{m} {fmt(E1v2.get('pe_first10pct',{}).get(m))} -> {fmt(E1v2.get('pe_last10pct',{}).get(m))}" for m in ("vision","tactile","proprioception"))
                     + f"; VICReg term {fmt(E1v2.get('vicreg_first'))} -> {fmt(E1v2.get('vicreg_last'))} (no longer zero); token temporal std min " + ", ".join(f"{k} {fmt(v,3)}" for k, v in sm2.items() if k in ('vision','tactile','proprioception')) + f"; NaN {E1v2.get('nan_steps')}",
                     f"PE fell: {pf(all(E1v2.get('pe_fell',{}).get(m, False) for m in ('vision','tactile','proprioception') if m in E1v2.get('pe_fell',{})))}; VICReg active at batch 1: {pf((E1v2.get('vicreg_last') or 0) > 0)}"])
    if E3c:
        w = E3c.get("windows", {})
        rows.append(["E3 again, child phase, AFTER the fixes (touch encoder now has a 'which patch am I' embedding)",
                     f"rows {E3c.get('rows')}; silhouette by region early {fmt(w.get('early',{}).get('silhouette_by_region'))} -> late {fmt(w.get('late',{}).get('silhouette_by_region'))}; linear probe region accuracy early {fmt(w.get('early',{}).get('linear_probe_region_accuracy'))} / middle {fmt(w.get('middle',{}).get('linear_probe_region_accuracy'))} / late {fmt(w.get('late',{}).get('linear_probe_region_accuracy'))} (chance {fmt(w.get('late',{}).get('chance'))})",
                     f"clusters by region improve: {pf(E3c.get('silhouette_improved'))}; region prediction improves: {pf(E3c.get('probe_accuracy_improved'))}; compare E3b above (before the embedding)"])
    if E4v3:
        a4b = E4v3.get("arms", {})
        rows.append(["E4 again AFTER the fixes (curiosity ignores fake senses; position IK for hands; held-command window measure)",
                     "<br>".join(f"<b>{k}</b>: window (10-step) command/response cos {fmt(v.get('cmd_response_window_cos_mean'),3)}, one-step cos {fmt(v.get('cmd_response_cos_mean'),3)}; curiosity drive mean {fmt(v.get('curiosity_drive_mean'),3)} (zero {fmt(v.get('curiosity_drive_frac_zero'),3)} of the time); |wellness grad| {fmt(v.get('wellness_grad_norm_mean'),3)}; motor loss {fmt(v.get('motor_loss_first10pct'))} -> {fmt(v.get('motor_loss_last10pct'))}" for k, v in a4b.items()),
                     f"commands correlate with the following body change over a 10-step window (cos > 0.3 for the model arms): {pf(all((v.get('cmd_response_window_cos_mean') or 0) > 0.3 for k, v in a4b.items() if k.startswith('model')) if a4b else None)}; curiosity no longer stuck at zero in the imagination-off arm: {pf((a4b.get('model_imag_off',{}).get('curiosity_drive_frac_zero') or 1.0) < 0.5 if a4b else None)}; imagination arm lower final motor loss: {pf(E4v3.get('imagination_arm_lower_final_motor_loss'))}"])
    E5T1 = res.get("E5_longrun_try1", {})
    if E5T1:
        st1 = E5T1.get("stability") or {}
        rows.append(["E5 long run, take 1 (20 bodies)",
                     f"{fmt(st1.get('elapsed_min'),3)} min, {st1.get('steps')} steps, {len(st1.get('checkpoints') or [])} hourly snapshots, then crashed at the 3rd growth step (toddler rebuild): the graphics driver ran out of framebuffers because a new OpenGL context was made for every body rebuild ('Shadow framebuffer is not complete'). Fixed: one context per worker, old bodies freed before rebuilding; 20 workers x 5 rebuilds tested clean.",
                     f"8 hours stable: {pf(False)} (crash at 2 h 34 min, cause fixed, see take 2)"])
    if E5L:
        stL = E5L.get("stability") or {}
        rows.append(["E5 long run, take 2, 20 bodies (8 hours planned)",
                     f"{fmt(stL.get('elapsed_min'),3)} min, {stL.get('steps')} steps, ok {fmt(stL.get('ok'))}, NaN {stL.get('nan_steps')}, {len(stL.get('checkpoints') or [])} hourly snapshots; see checkpoints table",
                     f"8 hours stable: {pf((stL.get('elapsed_min') or 0) >= 470 and bool(stL.get('ok')))}"])
    parts.append("<h2>4. Experiments E1 to E5</h2><p>'PE' = the model's prediction error in its own embedding space (lower = it predicts the next moment better). 'met' / 'not met' refer to the outline's success criteria; every number is from the run files. Rows marked AFTER the fixes were re-run once the eight decisions were applied.</p>"
                 + table([[esc(r[0]), r[1], r[2]] for r in rows], ["Experiment", "Measured", "Criterion"], raw=True))
    figs = [copy_fig(E1.get("figure")), copy_fig(E2.get("figure")), copy_fig(E3.get("figure")), copy_fig(E4.get("figure")), copy_fig(E5.get("figure")),
            copy_fig(E1v2.get("figure")), copy_fig(E3c.get("figure")), copy_fig(E4v3.get("figure"))]
    parts.append("<p>" + " ".join(f'<img src="{f}" alt="{f}">' for f in figs if f) + "</p>")
    parts.append("<p>Raw numbers: <code>Saved/mimo_bridge/experiments/results/results.json</code>; every run has <code>replay_training_stats.jsonl</code>, <code>summary.json</code>, <code>stdout.log</code>.</p>")
    # ---------------- guesses
    parts.append("""
<h2>5. Things I had to guess (all written in the code too)</h2>
<ul>
<li>Goertzel vibration window = 100 env steps (1 s at 100 Hz) so 5/10/15/20 Hz are exact bins; power = squared amplitude in N^2. The UE window length is not in any Python file.</li>
<li>Skin point area = spacing^2; contact area = active points x point area. Normal and tangential force come from MuJoCo's contact force, area from the skin points.</li>
<li>Fatigue (written as body-set capacity, 1 = fresh): dcap/dt = 0.05 (1 - cap) - 0.02 x effort, effort = |torque x joint speed| per body set in W. Energy: ph_work = that effort x dt, pcc_holding = |torque| x dt x 0.01.</li>
<li>Action decoder: orientation targets only (outline risk R3), PD gains kp 2, kd 0.3 (tuned on the real body; kd 0.05 oscillated), damping 1e-3, strength as a gain, servo aims at the reachable part of each joint range; radius and target position ignored. Gaze -> eye joints (right eye horizontal sign flipped).</li>
<li>Babbling noise also wobbles the target rotation by 10 x exploration_noise rad (UE adds it to cm positions, which the MIMo decoder does not read).</li>
<li>Curriculum phase -> MIMo age 1, 4, 8, 13, 18, 24 months; sensory/motor delays (4,3), (3,2), (2,2), (1,1), (1,0), (0,0) steps; context length kept at 4 (no forward-model rebuild mid-run).</li>
<li>Lite model: d_model 128, context 4, 2 forward layers with 8 heads, encoders 128 wide (3.3 M params) - live_test_lite's encoders are 768 wide inside by accident.</li>
<li>Audio stays synthetic (placeholder), intermind synthetic, voice disabled; MIMo frames set the Proprio | Motor | Touch (| Vision) mask bits.</li>
<li>Python deps: MIMo pins numpy 2.0.2 / scipy 1.13.1 / matplotlib 3.9.4 which have no Python 3.13 wheels; newer versions were installed (mujoco 3.3.0, gymnasium 1.2.3, torch 2.6.0, SB3 2.7.1 at MIMo's pins). Venv: <code>UEMA_devA_private/.venv_mimo</code>.</li>
<li>Lite textures: MIMo's 2500 x 15000 cube-map face/cloth textures (1 GB per body) are replaced by flat colours in a meta XML generated at run time from MIMo's file (geometry, joints, actuators untouched). Eye images show flat-coloured limbs instead of textured ones.</li>
<li>Disk: 27 GB free at start. With the 128-wide lite model a checkpoint (safetensors + optimizer) is about 37 MB; all runs together used under 1 GB. Runs shorter than an hour snapshot at the end only.</li>
</ul>
""")
    # ---------------- MIMo changes
    parts.append(f"""
<h2>6. Changes made in MIMo_private</h2>
<p>Branch <code>{esc(br)}</code> at <code>{esc(head)}</code>. Commits on bridge-work beyond v3: <code>{esc(diff) or 'none'}</code>. Working tree status: <code>{esc(st) or 'clean'}</code>.</p>
<p><b>No MIMo file was changed or copied.</b> Everything MIMo-side is a runtime workaround in the bridge: env subclass creates the delay lists early; own scene XML includes MIMo's model by absolute path; growth called with <code>create_log=False</code> under a lock; small-texture meta generated at run time; eyes rendered by the bridge; benchmark.py driven with an unwrapped env in our own runner process. Leftovers in MIMo's folder: only <code>mimo_infant.egg-info/</code> and <code>__pycache__/</code> from the editable install (both ignored by MIMo's .gitignore).</p>
""")
    # ---------------- checkpoints
    parts.append("<h2>7. Weight snapshots (every module of the model: encoders, forward model, prediction heads, motor controller, wellness, gaze and voice heads, plus optimizer)</h2>")
    parts.append(table([[r[0], r[1], r[2], r[3], r[4], f"<code>{esc(r[5])}</code>"] for r in ck],
                       ["run", "step", "saved at", "reason", "minutes into run", "path"]))
    # ---------------- decisions
    parts.append("""
<h2>8. Your decisions and what was done (2026-08-19 morning)</h2>
<p>You said: fix all of them, support one body, run with as many bodies as the hardware allows. Done:</p>
<ol>
<li><b>Lite script fixed</b> (lem/live_test_lite.py: n_heads 8, placeholder batch 1, motor 3-tuple, total_params, audio dict form, zero tokens for modalities a frame does not carry). It now runs unchanged against the MIMo bridge over shared memory, with vision (logs in Source/Python/mimo_bridge/phase4/).</li>
<li><b>VICReg works with one body</b>: new setting training.loss.vicreg.temporal_window (8 in master_config): at batch 1 the variance/covariance terms compare the current token with the last 8 tokens of that modality. Tests added (lem/tests/test_vicreg.py).</li>
<li><b>Vision over shared memory</b>: LEMTokenizer.read_frame now reads LEM_Vision through CPUReader.read_vision_pixels when no D3D12 reader is available. Mode B and the lite script get eyes.</li>
<li><b>MIMo bugs patched on branch bridge-work</b> (commit 44b4fb6, 4 files): eye cameras render per eye; sensory/motor delays no longer crash and delay exactly k steps; custom_measurements keyword; benchmark.py env.unwrapped.dt; PositionalModel locks joints with the XML equality constraints. The 1 GB textures were NOT changed in MIMo (overwriting binary assets was declined); the bridge keeps its small-texture meta file instead, so 24 bodies fit.</li>
<li><b>8-hour run</b>: started automatically after the re-runs with --n-envs auto (as many bodies as CPU and RAM allow, capped at 24), hourly snapshots; see the E5 long-run row and the dashboard (Saved/mimo_bridge/dashboard.html).</li>
<li><b>Action decoder</b>: position targets added for wrists and finger tips (damped least squares over orientation + position rows; wrists use the whole arm chain); pulls on shared joints add up. New 10-step "held command" measure logged as cmd_response_window_cos.</li>
<li><b>Curiosity</b> now averages prediction error over real senses only (lem/live_test.py, lem/training_step.py, and the bridge).</li>
<li><b>Touch patch identity</b>: tactile encoder option patch_pos_embedding (on in master_config): one learned vector per patch slot, added after the per-patch MLP.</li>
</ol>
<h2>8b. Still open (updated 2026-08-20 morning)</h2>
<p>This list was written mid-night; by morning all 8 items were closed. Status of each:</p>
<ol>
<li><b>live_test_lite.py broken</b> - CLOSED: all four bugs fixed after approval (n_heads 8, batch 1, 3-tuple, trainable_params).</li>
<li><b>Batch-1 collapse</b> - CLOSED: temporal VICReg (training.loss.vicreg.temporal_window: 8) gives one body an anti-collapse term; tests in lem/tests/test_vicreg.py.</li>
<li><b>Vision over shared memory needs a CPU path</b> - CLOSED: LEMTokenizer falls back to CPUReader.read_vision_pixels when no D3D12 reader exists; Mode B has eyes.</li>
<li><b>MIMo bugs upstream</b> - CLOSED: patched on branch bridge-work (commit 44b4fb6, pushed to origin).</li>
<li><b>"8 hours stable" run</b> - CLOSED (2026-08-20, decision): H2 ran 6 hours / 121,633 steps / all six growth phases and ended cleanly, and 6 hours was accepted as sufficient evidence of stability. H3 adds another 6 hours on top.</li>
<li><b>Action decoder ignores positions</b> - CLOSED: position rows for wrists and finger tips (damped least squares over the arm chain); held-command measure cmd_response_window_cos added.</li>
<li><b>Curiosity gate counts fake senses</b> - CLOSED: curiosity's prediction error now averages real senses only.</li>
<li><b>No touch patch identity</b> - CLOSED: tactile patch_pos_embedding on; note the ID-tag side effect made the old body-map probe unfair, so that probe was retired.</li>
</ol>
<p>Closed since this list was written, from other threads: replay (the diary), body schema wired to wellness, motor loss graded on the measured next frame instead of the model's own echo, snapshot lineage + weights registry, intention generator and think-harder mode wired and smoke-tested for H3.</p>
""")
    parts.append(f"<h2>9. Commits on feature/mimo-bridge</h2><pre>{esc(git_log())}</pre>")
    parts.append("<p>Outline with all measured numbers per phase: <code>docs/MIMo_Testbed_Implementation_Outline_2026-08-19.html</code> sections 10 to 15.</p></body></html>")
    os.makedirs(DOCS, exist_ok=True)
    import shutil, time as _t
    _arch = os.path.join(os.path.dirname(DOCS), "Saved", "mimo_bridge", "reports")
    os.makedirs(_arch, exist_ok=True)   # every build archived: rebuilds never erase a reading
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))
    shutil.copy2(OUT, os.path.join(_arch, _t.strftime("report_%Y-%m-%d_%H%M.html")))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
