# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The old dashboard (tools/dashboard.py) and the whole-life page (tools/lifetime.py) after the
gain-of-function build of 2026-09-22: new rows, each with an explanation; the ledger chip on
the lineage step; stop-rule stops named; anchors refused across model widths; the page scoped to
the active run, its lineage and the newest other run; the ten-item yardsticks."""
import html as _html
import importlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
TD = importlib.import_module("test_dashboard_metric_semantics")
from mimo_bridge.tools import dashboard, lifetime  # noqa: E402

LO = {"vision": 0.1, "tactile": 0.1, "proprioception": 0.01}
BLOCKS = {
    "prediction_audit": {"modalities": {m: {"improvement_over_shuffled": 0.0005, "improvement_over_persistence": -0.05,
                                            "sensitivity": 0.01} for m in ("vision", "tactile", "proprioception")}},
    "wellness": {"prediction_error": 0.97, "tactile_comfort": 0.17, "self_touch": 0.0},
    "consequence": {"kinds_mean": 0.97, "rate": 0.865, "weight": 0.5, "goal_rate": 1.0},
    "exec_source": {"planner": 0.5, "reflex_noise": 0.4, "reflex": 0.1},
    "incorporation": {"extent": 0.25, "objects_evidence": {"toy": {"match": 0.39, "control": 0.05}}},
    "self_contact_pairs": 1.56, "self_touch_controlled": {"improvement_over_shuffled": -0.0117, "rows": 23},
    "causal_chain": {"chosen_best_share": 0.4, "alternatives_n": 4, "planner_events": 10, "wellness_change_corr": 0.1},
    "efference": {"body_cm_err_latch": 1.2, "body_cm_do_nothing_latch": 2.0, "body_cm_shuffled_latch": 1.9,
                  "gaze_deg_err": [1.0, 2.0, 3.0], "gaze_deg_do_nothing": [2.0, 4.0, 6.0]},
    "machine": {"commit_gb": 40.0, "commit_limit_gb": 68.8, "gpu_used_mib": 4000, "gpu_total_mib": 8188, "step_ms": 800},
    "vestibular": {"improvement_over_persistence": 0.1, "trunk_roll_deg": -12.0},
    "controllability": {"agreement_mean": 0.2, "control_mean": 0.01, "measured_share": 0.4},
    "learning_progress": {"value_mean": 0.002},
    "wellness_rank": {"mode": "normalized", "winner_margin_share": {"coherence": 0.6, "integrity": 0.1},
                      "integrity_gated_share": 0.01},
    "hand_press": {"world_share": 0.1, "self_share": 0.7},
    "contact_control": {"head_auc": 0.7, "success_self": 0.2, "success_control": 0.05},
    "self_contact_events": {"onsets": 3.0, "in_view_share": 0.2},
    "synergy": {"explained_share": 0.3, "effective_dim": 2.5, "dim": 8},
    "next_frame_norm": {"weight_effective": {"vision": 1.0, "audio": 0.2}},
}
NEW_LABELS = ("Action use in prediction (prediction audit)", "Better than a copy of the present (prediction audit)",
              "Wellness channels (window means over the bodies)", "Consequence term in wellness",
              "Who authored the executed command", "Body map co-movement (incorporation)",
              "Self-contact (own body touching itself)", "Causal chain of each plan",
              "Efference heads (predicted body and eye movement)", "Machine load (memory, card, step time)",
              "Vestibular prediction and trunk posture", "Controllability channel", "Learning-progress channel",
              "Normalised goal ranking", "Hand pressing split into world and self", "Contact prediction and control",
              "Self-touch onsets", "Motor synergy map", "Next-model-step loss normalisation")


def _rec(step, **extra):
    r = {"type": "mimo", "step": step, "timestamp": 0, "elapsed_s": 600, "phase": "mature", "n_envs": 8,
         "model_steps_per_s": 4.0, "env_dt_s": 0.01, "pe": LO, "next_pe": LO,
         "contract": {"exec_seen": step - 1, "nf_aligned": step - 1, "step": step}}
    r.update(extra)
    return r


def _rows(h):
    return {_html.unescape(label): _html.unescape(value)
            for label, value in re.findall(r"<tr[^>]*><th>(.*?)</th><td>(.*?)</td></tr>", h, flags=re.S)}


def test_every_new_row_renders_with_an_explanation_and_no_zero_for_missing(tmp_path):
    recs = [_rec(101 + i) for i in range(8)] + [_rec(190 + i, **BLOCKS) for i in range(8)]
    h = TD._render(tmp_path, recs=recs)
    rows = _rows(h)
    for label in NEW_LABELS:
        assert label in rows, f"row not rendered: {label}"
        key = dashboard.doc_for(label)
        assert key is not None and dashboard.ROW_DOC[key]["use"] >= 8 and dashboard.ROW_DOC[key]["clear"] >= 8, label
        assert f'data-doc="{_html.escape(key)}"' in h or f'data-doc="{key}"' in h, label
    # DESIGN.md section 11 (2026-09-23): shares between named ends and counts per 100, never 0.0005
    assert "0.05% for vision" in rows["Action use in prediction (prediction audit)"]
    assert "chance 20 of every 100 plans" in rows["Causal chain of each plan"]
    assert "Hand pressing on its OWN body counts" in rows["Consequence term in wellness"]
    (tmp_path / "bare").mkdir()
    bare = _rows(TD._render(tmp_path / "bare"))
    assert not any(label in bare for label in NEW_LABELS), "absent fields render no row, never a 0"


def test_the_ledger_chip_divides_a_lineage_counter_by_the_lineage_step(tmp_path):
    # H84: exec_seen 344,399 at lineage step 344,400 while the segment added 16,025 steps
    recs = [_rec(328_400 + 100 * i, contract={"exec_seen": 328_399 + 100 * i, "step": 328_400 + 100 * i})
            for i in range(16)]
    h = TD._render(tmp_path, recs=recs, args={"resume": "x/ckpt_step_328375"})
    chip = _html.unescape(re.search(r"action ledger ([^<]*)</span>", h).group(1))
    assert chip.startswith("100.0% of the lineage's 329,900 steps"), chip
    assert "2149" not in h


def test_a_stop_rule_stop_is_named_with_its_numbers_not_finished(tmp_path):
    run = tmp_path / "experiments" / "R1"
    h = None
    old_render = TD._render

    def with_stop(tmp):
        # _render writes the run into tmp/staging then renames it to experiments/R1
        return old_render(tmp, recs=[_rec(101 + i) for i in range(16)])
    # render once to create the folder, then add summary + stop reason and render again
    with_stop(tmp_path)
    (run / "summary.json").write_text(json.dumps({"ok": True, "steps": 116, "elapsed_s": 60}), encoding="utf-8")
    (run / "stop_reason.json").write_text(json.dumps({
        "reason": "model steps per second 0.397 under one third of own hour-1 baseline 1.249 two hourly checks",
        "resume_policy": "repair_required", "timestamp": 1_790_122_503}), encoding="utf-8")
    past = 1_790_000_000
    os.utime(run / "replay_training_stats.jsonl", (past, past))
    old_saved, old_out = dashboard.SAVED, dashboard.OUT
    dashboard.SAVED, dashboard.OUT = str(tmp_path), str(tmp_path / "dash.html")
    try:
        dashboard.build()
        h = (tmp_path / "dash.html").read_text(encoding="utf-8")
    finally:
        dashboard.SAVED, dashboard.OUT = old_saved, old_out
    rows = _rows(h)
    assert "stopped by the hourly watcher program" in rows["Status"]
    assert "0.397 model steps per second" in rows["Status"] and "waits for a repair" in rows["Status"]
    assert 'class="tag t-bad">stopped by a stop rule<' in h
    assert ">finished<" not in h


def test_anchors_from_a_narrower_model_are_refused(tmp_path):
    intention = {"plan_length": 1, "final_mse": 1.27, "gap_closed_pct": None}
    anchors = {"one_instant": 0.0255, "half_minute_apart": 0.556, "goal_arrival": 0.01, "measured": "2026-09-06",
               "ckpt": str(tmp_path / "H67" / "checkpoints" / "ckpt_step_1")}
    (tmp_path / "H67").mkdir()
    (tmp_path / "H67" / "args.json").write_text("{}", encoding="utf-8")
    label, text = dashboard.plan_distance_row(intention, anchors, {"d_model": 192})
    assert "unanchored for this model" in text and "128-wide" in text and "192-wide" in text
    assert "% of the distance between two unrelated" not in text
    label, same = dashboard.plan_distance_row(intention, anchors, {})
    assert "instants of world-change short" in same
    assert dashboard.doc_for(label) is not None


def _run(base, name, rows, args=None, mtime=None):
    d = base / "experiments" / name
    d.mkdir(parents=True)
    (d / "args.json").write_text(json.dumps(args or {}), encoding="utf-8")
    p = d / "replay_training_stats.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if mtime:
        os.utime(p, (mtime, mtime))
    return d


def test_the_page_describes_the_active_run_its_lineage_and_the_newest_other_run_only(tmp_path):
    saved = tmp_path / "Saved" / "mimo_bridge"
    root_run = _run(saved, "L", [_rec(100 + i) for i in range(10)], mtime=1_700_000_000)
    live = _run(saved, "L_seg2", [_rec(200 + i) for i in range(10)],
                {"resume": str(root_run / "checkpoints" / "ckpt_step_109")}, mtime=1_700_000_500)
    _run(saved, "old_other", [_rec(1 + i) for i in range(10)], mtime=1_600_000_000)
    newest_other = _run(saved, "new_other", [_rec(1 + i) for i in range(10)], mtime=1_700_000_900)
    (saved / "training_continuation.json").write_text(json.dumps({"active_run": str(live), "timestamp": 1}), encoding="utf-8")
    old_saved = dashboard.SAVED
    dashboard.SAVED = str(saved)
    try:
        scoped = {os.path.basename(d): role for d, role in dashboard.scoped_runs()}
    finally:
        dashboard.SAVED = old_saved
    assert scoped == {"L_seg2": "live", "L": "ancestor", "new_other": "other"}
    info = dashboard.describe_light(str(root_run))
    assert info["role"] == "ancestor" and info["steps"] == 109 and "rec" not in info


def test_compact_records_keep_every_field_read_across_records(tmp_path):
    rows = [_rec(1, movement={"self_factor": 1.0, "local_speed": 3}, intention={"plan_length": 2, "final_mse": 1},
                 tenure_tok={"rank1": 0.2, "n_decoys": 14}, nan_steps=0, roll_deg_max=30.0, follow_bout=0.1,
                 recon_dec=LO, task_hand_ball_dist_m=0.4, wellness={"movement": 0.5})]
    d = _run(tmp_path, "R", rows)
    old_saved = dashboard.SAVED
    dashboard.SAVED = str(tmp_path)
    try:
        compact = dashboard.compact_records(str(d / "replay_training_stats.jsonl"))
    finally:
        dashboard.SAVED = old_saved
    c = compact[0]
    assert c["movement"] == {"self_factor": 1.0} and c["intention"] == {"plan_length": 2}
    assert c["tenure_tok"] == {"rank1": 0.2} and c["nan_steps"] == 0 and "wellness" not in c
    assert "contract" not in c, "the full newest record is read separately from the stats tail"


def test_lifetime_sweeps_the_lineage_only_and_names_the_ten_item_yardsticks(tmp_path, monkeypatch):
    saved = tmp_path / "Saved" / "mimo_bridge"
    parent = _run(saved, "P", [_rec(100 + i, timestamp=1_790_000_000 + i,
                                     prediction_audit=BLOCKS["prediction_audit"], cmd_response_cos_held=0.1)
                               for i in range(10)])
    child = _run(saved, "P_seg2", [_rec(200 + i, timestamp=1_790_001_000 + i, causal_chain={"chosen_best_share": 0.5})
                                   for i in range(10)], {"resume": str(parent / "checkpoints" / "ckpt_step_109")})
    _run(saved, "unrelated", [_rec(1 + i, timestamp=1_790_000_000 + i) for i in range(10)])
    (saved / "training_continuation.json").write_text(json.dumps({"active_run": str(child), "timestamp": 1}), encoding="utf-8")
    monkeypatch.setattr(lifetime, "SAVED", str(saved))
    monkeypatch.setattr(lifetime, "EXP", str(saved / "experiments"))
    monkeypatch.setattr(lifetime, "OUT_HTML", str(saved / "lifetime.html"))
    monkeypatch.setattr(lifetime, "OUT_JSON", str(saved / "lifetime.json"))
    lifetime.build(force=True)
    runs = json.loads((saved / "lifetime.json").read_text(encoding="utf-8"))
    assert [r["run"] for r in runs] == ["P", "P_seg2"], "the unrelated run is not swept"
    assert runs[0]["gof_act"] == 0.0005 and runs[0]["gof_cc"] is None
    assert runs[1]["gof_cc"] == 0.5 and runs[1]["gof_act"] is None
    page = (saved / "lifetime.html").read_text(encoding="utf-8")
    assert "Gain-of-function yardsticks" in page and "1. Chose the plan it predicted would feel best" in page
    assert page.count("Unmeasured: no probe result on this lineage") == 4
    assert "Intentional rolling and purposeful recovery remain unproved" in page
