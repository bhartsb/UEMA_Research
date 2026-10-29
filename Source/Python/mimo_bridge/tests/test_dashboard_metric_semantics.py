# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Regression tests for claims shown on the MIMo dashboard.

These tests intentionally police semantics, not typography.  The dashboard is an
instrument panel; a proxy must not be upgraded into a scientific conclusion.
"""
from pathlib import Path
import inspect

import mimo_bridge.tools.dashboard as dashboard
import mimo_bridge.tools.overseer as overseer


def _src(module) -> str:
    return inspect.getsource(module)


def test_dashboard_distinguishes_physics_frames_from_model_updates():
    s = _src(dashboard)
    assert "Simulator physics frames per second" in s
    assert "model observes and learns only once per action-repeat" in s
    assert "how many sense-snapshots per second the model processes" not in s


def test_dashboard_does_not_claim_imagination_is_trusted_from_real_state_mae():
    s = _src(dashboard)
    assert "imagination can be trusted" not in s
    assert "Judge-B agreement on recent real states" in s


def test_dashboard_does_not_call_body_discrimination_situational_awareness():
    s = _src(dashboard)
    assert "Does it treat its 16 bodies differently?" not in s
    assert "situational awareness is improving" not in s
    assert "Between-body action signatures (uncalibrated)" in s


def test_overseer_goal_verdict_uses_causal_not_reconstruction_pe():
    s = _src(overseer)
    start = s.index("# ------------- the verdict")
    # widened 2026-09-08: the prediction/ability split grew the block
    verdict = s[start:start + 7000]
    assert 'last.get("next_pe")' in verdict
    assert 'last.get("pe", {})' not in verdict


def test_overseer_verdict_separates_prediction_from_ability():
    """Follow-up review 2026-09-08: a prediction-error level must never produce
    ability language, and ability claims come only from behavioral artifacts."""
    s = _src(overseer)
    start = s.index("# ------------- the verdict")
    verdict = s[start:start + 7000]
    assert "PREDICTION: causal next-state error is" in verdict
    assert "ABILITY (behavioral evidence only)" in verdict
    assert "skill_battery" in verdict
    assert "Gain-of-function verdict" not in verdict
    # the old ability-flavored wording sourced from a PE threshold stays gone
    assert "causal next-state prediction is strong" not in verdict


def test_angle_and_contact_thresholds_are_not_called_purposeful_behaviour():
    s = _src(dashboard) + _src(overseer)
    assert "HAS ROLLED OVER" not in s
    assert "it has gained a real motor function (rolling over)" not in s
    assert "eyes found it themselves" not in s
    # 2026-09-09 operator catch: a 30-deg cone crossing was called a purposeful look
    assert "head lifted on purpose" not in s
    assert "in view NOW" not in s


def test_rendered_trend_line_makes_no_awareness_claim():
    """The grep tests above cannot see this: the sentence was built at runtime.

    A source search for the rendered phrase passes whether or not the page prints
    it, so this one renders the line and reads the output instead.
    """
    trend = {"roll_deg_max": "improving", "discrimination": "improving",
             "action_signature_ratio": "improving"}
    label, word = dashboard.TREND_LABEL, dashboard.TREND_WORD
    out = ", ".join(f"{label.get(k, k)} is {word.get(v, v)}" for k, v in trend.items())
    assert "situational awareness" not in out, out
    assert "improving" not in out, out
    assert "uncalibrated proxy" in out, out


def test_contract_overseer_success_is_scoped_to_plumbing():
    """18 checks, of which 2 only read source text. One total hid that."""
    from mimo_bridge.tools import contract_overseer as co
    rep = co.run(["C1", "C3"])
    for k in ("n_runtime", "n_runtime_pass", "n_static", "n_static_pass"):
        assert k in rep, f"{k} missing from the report"
    assert rep["n_runtime"] + rep["n_static"] == rep["n_checks"]
    assert rep["n_static"] >= 1, "C1 is a static check and must be counted as one"
    src = inspect.getsource(co.main)
    assert "NOT a scientific validation" in src


# --------------------------------------------------------------------------------
# The eighteen audit items, checked on RENDERED OUTPUT. A source grep cannot see a
# sentence assembled at runtime, which is how "situational awareness is improving"
# survived a green suite while the page was printing it.
# --------------------------------------------------------------------------------
import json
import os


def _render(tmp_path, args=None, recs=None, lineage=None, preflight=None, videos=(), extra=None):
    """Render one synthetic run through the real dashboard and return its HTML."""
    run = tmp_path / "staging"
    (run / "checkpoints" / "ckpt_step_200").mkdir(parents=True)
    base_args = {"n_envs": 8, "scene": "crib", "action_repeat": 5, "motor_hold": 5,
                 "intention": "off", "imagination": "off", "replay": "on",
                 "next_frame_loss": "on", "curiosity_band": "0.02:0.4",
                 "resume": "x/ckpt_step_100", "minutes": 60.0}
    base_args.update(args or {})
    (run / "args.json").write_text(json.dumps(base_args), encoding="utf-8")
    (run / "checkpoints.json").write_text(json.dumps(
        [{"step": 200, "path": str(run / "checkpoints" / "ckpt_step_200"),
          "time": "2026-08-26 00:00:00"}]), encoding="utf-8")
    (run / "checkpoints" / "ckpt_step_200" / "lineage.json").write_text(
        json.dumps(lineage or {"code_commit": "deadbeef",
                               "resumed_from": "x/ckpt_step_100", "step": 200}),
        encoding="utf-8")
    if preflight is not None:
        (run / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")

    def rec(step, pe, npe):
        return {"type": "mimo", "step": step, "timestamp": 0, "elapsed_s": 600,
                "phase": "mature", "n_envs": 8, "model_steps_per_s": 4.0,
                "env_dt_s": 0.01, "env_steps_per_s": 32.0, "pe": pe, "next_pe": npe,
                "curiosity_drive": 0.5,
                "movement": {"local_speed": 1.0, "stillness_debt": 0.0,
                             "movement_wellness": 0.5, "self_factor": 1.0},
                "cmd_response_cos_held": -0.1, "cmd_response_ratio": 0.05,
                "motor_discrimination_ratio": 0.09, "roll_deg_max": 123.0,
                "task_eye_pitch": 35.0,
                "task_box_touch_rate": {"light": 0.0, "medium": 0.0, "heavy": 0.13},
                "task_box_touch_hand_rate": 0.0, "wellness_align_err": 0.007,
                "task_gaze_on_ball_rate": 0.3, "task_gaze_on_ball_earned_rate": 0.279,
                "task_gaze_ball_deg": 40.0,
                "contract": {"exec_seen": 100, "nf_aligned": 100, "step": step},
                **(extra or {})}

    lo = {"vision": 0.1, "tactile": 0.1, "proprioception": 0.01}
    hi = {"vision": 0.2, "tactile": 0.2, "proprioception": 0.02}
    body = recs or ([rec(101 + i, lo, lo) for i in range(8)]
                    + [rec(190 + i, hi, hi) for i in range(8)])
    (run / "replay_training_stats.jsonl").write_text(
        "\n".join(json.dumps(r) for r in body), encoding="utf-8")

    if videos:
        (run / "videos").mkdir(exist_ok=True)
        for v in videos:
            (run / "videos" / v).write_bytes(bytes(1))
    (tmp_path / "experiments").mkdir(exist_ok=True)
    os.rename(str(run), str(tmp_path / "experiments" / "R1"))
    old_saved, old_out = dashboard.SAVED, dashboard.OUT
    dashboard.SAVED = str(tmp_path)
    dashboard.OUT = str(tmp_path / "dash.html")
    try:
        dashboard.build()
        return (tmp_path / "dash.html").read_text(encoding="utf-8")
    finally:
        dashboard.SAVED, dashboard.OUT = old_saved, old_out


def test_item1_model_rate_and_physics_rate_are_separate_numbers(tmp_path):
    h = _render(tmp_path)
    assert "Model transitions per second" in h
    assert "Simulator physics frames per second" in h
    assert "4.00 per body" in h        # model rate, as logged
    assert "20.0 per body" in h        # physics rate = 4.0 model steps x action_repeat 5


def test_failed_saves_and_peak_allocation_reach_the_panel(tmp_path):
    """2026-09-18: a failed hourly save no longer ends the run, so the panel must show
    the count; the trainer's own peak allocation is the number the size decision rests on."""
    h = _render(tmp_path, extra={"checkpoint_failures": 2, "peak_vram_mb": 3991.4,
                                 "checkpoint_last_error": "OSError: The device is not ready"})
    assert "Saves that failed this segment (should always be 0)" in h
    assert "2 - last error: OSError: The device is not ready" in h
    assert "3,991 MiB allocated by the trainer" in h
    (tmp_path / "quiet").mkdir()
    quiet = _render(tmp_path / "quiet")          # the fixture renders one run per folder
    # the help table is embedded in every page; the ROW labels appear only when rendered
    assert "Saves that failed this segment (should always be 0)" not in quiet
    assert "Peak allocation on the card, last window" not in quiet


def test_item2_lineage_step_is_separated_from_this_run(tmp_path):
    h = _render(tmp_path)
    assert "Learning steps (checkpoint lineage)" in h
    assert "New steps in this run" in h
    assert "97 (resumed at step 100)" in h     # last step 197 minus the resume step


def test_item3_folder_age_and_process_runtime_are_separate(tmp_path):
    h = _render(tmp_path)
    assert "own clock" in h
    assert "run folder created" in h


def test_item4_constant_self_factor_is_reported_as_bypassed(tmp_path):
    h = _render(tmp_path)
    assert "Co-movement channel (body map)" in h
    assert "BYPASSED" in h
    assert "self-caused 1.00" not in h


def test_item5_judge_b_row_hidden_when_imagination_off(tmp_path):
    assert "Judge-B agreement" not in _render(tmp_path)


def test_item5b_judge_b_row_shown_when_imagination_on(tmp_path):
    assert "Judge-B agreement" in _render(tmp_path, args={"imagination": "on"})


def test_item7_command_following_splits_direction_from_magnitude(tmp_path):
    pf = {"ok": True, "checks": [{"name": "limb follows a command over 250 ms: cosine",
                                  "value": 0.8711, "ok": True}]}
    h = _render(tmp_path, preflight=pf)
    assert "Does the body move the way it was told (direction)" in h
    assert "How much of the asked turn happens (size only)" in h
    assert "Mechanical probe at launch" in h
    assert "A held command on a quiet body measures +0.87, so large gaps" not in h


def test_item7b_probe_row_absent_when_no_preflight_file(tmp_path):
    """The +0.87 used to be a constant in the source. With no preflight it must vanish."""
    assert "Mechanical probe at launch" not in _render(tmp_path)


def test_item9_box_contact_says_any_body_part(tmp_path):
    h = _render(tmp_path)
    assert "any body part" in h
    assert "A leg lying against a box counts" in h


def test_item11_earned_look_states_cause_is_unknown(tmp_path):
    h = _render(tmp_path)
    assert "Ball near the gaze (cause unknown)" in h
    assert "eyes found it themselves" not in h


def test_item12_curiosity_is_not_an_interest_dial(tmp_path):
    h = _render(tmp_path)
    assert "interest dial" not in h
    assert "deterministic function of recent prediction error" in h


def test_item14_body_count_is_read_from_the_run(tmp_path):
    h = _render(tmp_path, args={"n_envs": 8})
    assert "16 bodies" not in h
    assert "8 bodies run at once" in h


def test_item15_eye_pitch_is_reported_not_classified(tmp_path):
    h = _render(tmp_path)
    assert "looking level" not in h
    assert "Eye pitch" in h


def test_item16_provenance_shows_the_commit_the_trainer_loaded(tmp_path):
    h = _render(tmp_path, lineage={"code_commit": "deadbeef",
                                   "resumed_from": "x/ckpt_step_100"})
    assert "code commit the trainer loaded" in h
    assert "deadbeef" in h
    assert ("does NOT match the current source HEAD" in h
            or "MATCHES the current source HEAD" in h)


def test_gates_panel_lists_every_switch_with_a_state(tmp_path):
    h = _render(tmp_path)
    for gate in ("intention", "imagination", "next_frame_loss", "body_schema",
                 "plan depth cap", "scale gate", "plan safety exit"):
        assert gate in h, gate
    assert "bypassed" in h        # body_schema, computed from the constant self_factor


def test_chip_strip_reports_causal_direction_and_causal_test_state(tmp_path):
    h = _render(tmp_path)
    assert "causal error" in h
    assert "worse" in h                       # 0.1 -> 0.2 across the synthetic windows
    assert "five-control causal test: not run" in h
    assert "action ledger" in h


def test_windowed_comparison_not_single_record_endpoints():
    """A single-record endpoint flipped the sign of H28's tactile trend; windows did not."""
    recs = [{"next_pe": {"tactile": v}} for v in ([0.05] * 10 + [0.30] + [0.20] * 10)]
    k, w = dashboard.window_means(recs, "next_pe", ("tactile",))
    first, last = w["tactile"]
    assert last > first                       # the window sees the rise
    assert k >= 1


def test_item13_operator_note_naming_a_dead_run_is_flagged_stale():
    """The 'H23b: transition contract live' banner rode the page through H24-H28."""
    note = "H23b: transition contract live - imagination OFF"
    assert dashboard.stale_note(note, []) == ["H23b"]
    assert dashboard.stale_note(note, [{"name": "H28_tc", "status": "running"}]) == ["H23b"]
    assert dashboard.stale_note(note, [{"name": "H23b_x", "status": "running"}]) == []
    assert dashboard.stale_note("rebuilt the encoder", []) == []


# --------------------------------------------------------------------------------
# Row explanations and the design review. Every number on the page must say what it
# is, how it was computed, and what would fool it - because most of this project's
# wrong turns were a correct number carrying a wrong label.
# --------------------------------------------------------------------------------

GATE_LABELS = {
    "gate", "setting", "run", "intention", "imagination", "replay", "next_frame_loss",
    "top_down_gate", "babble", "random_actions", "gaze_reflex", "spontaneous_saccades",
    "movement_drive", "body_schema", "plan depth cap", "scale gate", "plan safety exit",
    "motor_hold", "smooth_actions", "curiosity_band", "wellness_mode", "rom_limit",
    "strength_limit", "delays", "code commit the trainer loaded", "resumed from",
    "newest checkpoint step",
}


def test_every_doc_entry_is_complete():
    """t/w/h/f are the four things a reader needs; a missing one is a silent gap."""
    for name, reg in (("DOC", dashboard.DOC), ("ROW_DOC", dashboard.ROW_DOC)):
        for key, d in reg.items():
            for field in ("t", "w", "h", "f", "use", "clear"):
                assert field in d, f"{name}[{key!r}] missing {field!r}"
            assert 8 < len(d["t"].strip()) < 70, f"{name}[{key!r}].t should be a short title"
            for field in ("w", "h", "f"):
                assert len(d[field].strip()) > 20, f"{name}[{key!r}].{field} is too thin"


def test_no_primary_row_scores_below_eight():
    """The design gate: below 8 on usefulness or clarity means rewrite or demote."""
    bad = [(k, v["use"], v["clear"]) for k, v in dashboard.ROW_DOC.items()
           if (v["use"] < 8 or v["clear"] < 8) and not v.get("secondary")]
    assert not bad, f"primary rows scoring below 8: {bad}"
    for k, v in dashboard.DOC.items():
        assert v["use"] >= 8 and v["clear"] >= 8, f"chip {k} scores {v['use']}/{v['clear']}"


def test_every_secondary_row_is_actually_a_weak_one():
    """Demotion must be earned - it is not a place to hide inconvenient numbers."""
    for k, v in dashboard.ROW_DOC.items():
        if v.get("secondary"):
            assert v["use"] < 8 or v["clear"] < 8, (
                f"{k} is demoted but scores {v['use']}/{v['clear']} - it belongs in the "
                f"main table")


def test_doc_lookup_is_case_insensitive():
    """Matching exactly left 'Hand pressure above threshold' with no explanation."""
    assert dashboard.doc_for("Hand pressure above threshold") is not None
    assert dashboard.doc_for("hand pressure above threshold") is not None
    assert dashboard.doc_for("a label that matches nothing at all") is None


def test_every_rendered_data_row_has_an_explanation(tmp_path):
    """Rendered output, not the registry - a row can exist with no matching key."""
    import re
    import html as _html
    h = _render(tmp_path)
    rows = re.findall(r"<tr([^>]*)><th>(.*?)</th>", h)
    assert rows, "no rows rendered at all"
    missing = sorted({_html.unescape(l) for a, l in rows
                      if "data-doc" not in a and _html.unescape(l) not in GATE_LABELS})
    assert not missing, f"rows with no explanation: {missing}"


def test_the_popup_and_its_payload_are_present(tmp_path):
    h = _render(tmp_path)
    for piece in ('id="ov"', 'id="bt"', "What this is", "How it is computed",
                  "What would fool it", "var D=", "Escape"):
        assert piece in h, piece
    assert 'role="button"' in h and 'tabindex="0"' in h, "not keyboard reachable"


def test_windows_never_straddle_a_restart():
    """A resumed run writes to the same folder; elapsed_s resets and marks the seam.

    Measured on H30: record 69, step 708475 at elapsed 1645s, then step 708500 at 18s.
    Before this, the 'first window' was entirely imagination-off and the 'last window'
    entirely imagination-on, so the printed trend was partly a config difference.
    """
    recs = ([{"step": 100 + i, "elapsed_s": i * 10} for i in range(40)]
            + [{"step": 200 + i, "elapsed_s": i * 10} for i in range(60)])
    seg, n, seam = dashboard.split_at_resume(recs)
    assert n == 1 and seam == 200
    assert len(seg) == 60
    assert seg[0]["elapsed_s"] == 0


def test_no_restart_means_no_trimming():
    recs = [{"step": i, "elapsed_s": i * 10} for i in range(30)]
    seg, n, seam = dashboard.split_at_resume(recs)
    assert n == 0 and seam is None and len(seg) == 30


def test_inert_command_pathway_is_distinguished_from_missing():
    """0.0 means zero-padded and never trained; None means the checkpoint predates the
    widening. An empty slice used to return nan and read as neither."""
    import inspect
    src = inspect.getsource(dashboard.command_pathway_weight)
    assert "MOTOR_IN" in src and "return None" in src
    assert "getmtime" in src, "no mtime tie-break: ckpt_step_N and ckpt_step_N_tc19 collide"


def test_the_two_videos_are_one_clip(tmp_path):
    """Outside view and eye view are the same recording and must play together.

    env.start_video writes both from a single session at fps 10, keeping every 10th
    env step; a measured pair reads 300 frames / 30.0 s each. Shown separately, a
    reader has to hold one in their head while watching the other, which is the only
    thing the pair is for.
    """
    import re
    h = _render(tmp_path, videos=("step_1.mp4", "step_1_eye.mp4"))
    vids = re.findall(r'<video class="vpair" data-pair="([^"]+)"\s+src="([^"]+)"([^>]*)>', h)
    assert len(vids) == 2, f"expected a synced pair, got {len(vids)}"
    assert vids[0][0] == vids[1][0], "the pair is not grouped under one key"
    assert "controls" in vids[0][2], "the lead video has no controls"
    assert "controls" not in vids[1][2], "the follower has its own controls and will desync"
    assert "video.vpair" in h and "timeupdate" in h, "no sync script"
    assert "they play together" in h


def test_a_lone_video_is_not_described_as_a_pair(tmp_path):
    h = _render(tmp_path, videos=("step_1.mp4",))
    assert "they play together" not in h
    assert "The baby from outside" in h


def test_every_percentage_says_whether_it_is_good_or_bad():
    """A bare "+24%" tells a reader nothing when smaller is better.

    Three row families printed a percentage. Only one said "getting worse"; the other
    two left the sign convention to be guessed, and it is the opposite of the usual one
    because every quantity here is an error.
    """
    assert "GETTING WORSE" in dashboard.fmt_change(0.10, 0.13)
    assert "IMPROVING" in dashboard.fmt_change(0.13, 0.10)
    assert "ABOUT THE SAME" in dashboard.fmt_change(0.100, 0.101)
    for a, b in ((0.10, 0.13), (0.13, 0.10), (0.100, 0.101)):
        assert "smaller is better" in dashboard.fmt_change(a, b)
    assert "no starting value" in dashboard.fmt_change(0, 0.5)


def test_the_verdict_reaches_the_rendered_page(tmp_path):
    """Source is not output - the formatter has to actually be wired into the rows."""
    h = _render(tmp_path)
    import re
    import html as _html
    rows = re.findall(r"<th>(Guessing[^<]*|Re-describing[^<]*)</th><td>(.*?)</td>", h)
    assert rows, "no prediction-error rows rendered"
    for lbl, val in rows:
        v = _html.unescape(val)
        assert any(w in v for w in ("IMPROVING", "GETTING WORSE", "ABOUT THE SAME")), (
            f"{_html.unescape(lbl)!r} shows a change with no verdict: {v[:90]}")
        assert "smaller is better" in v, f"{_html.unescape(lbl)!r} omits the direction"


def test_the_self_model_row_says_whether_the_table_is_about_todays_moments(tmp_path):
    """2026-09-19, audit P238: the row used to call every table row a 'recurring state discovered from
    experience'. It now counts the different patterns stored and the slots that are copies, and - when the run reports them -
    says what share of today's moments belong to a slot and how alike a moment and its nearest slot are, the two
    numbers that tell a table about today from a record of other moments."""
    hs = {"slots_filled": 6400, "slots_distinct": 6400, "slots_twinned": 0, "l2_sig": -1, "l3_sig": -1,
          "l2_steps_to_first_evidence": 150000, "l3_steps_to_first_evidence": 2500000,
          "moments_with_a_slot": 0.0, "best_slot_cos": 0.2367}
    h = _render(tmp_path, extra={"hself": hs})
    assert "6,400 slots in use hold 6,400 different patterns (0 slots are copies of another slot" in h
    assert "0 of 100 of today&#x27;s model steps belong to a slot" in h or \
           "0 of 100 of today's model steps belong to a slot" in h
    # DESIGN.md section 11 (2026-09-23): a cosine is printed as a share of the way between named ends
    assert "24% of the way from &#x27;nothing in common&#x27; to &#x27;the same pattern&#x27; with its nearest slot" in h         or "24% of the way from 'nothing in common' to 'the same pattern' with its nearest slot" in h
    # segments 6 and 7 of H84 wrote the two gauges under their first names; both are read
    (tmp_path / "first_names").mkdir()
    first = _render(tmp_path / "first_names", extra={"hself": {**{k: v for k, v in hs.items()
                    if k not in ("moments_with_a_slot", "best_slot_cos")},
                    "moments_with_a_row": 0.0, "best_row_cos": 0.2367}})
    assert "24% of the way from" in first and "with its nearest slot" in first
    (tmp_path / "older").mkdir()
    old = _render(tmp_path / "older", extra={"hself": {k: v for k, v in hs.items()
                                                       if k not in ("moments_with_a_slot", "best_slot_cos")}})
    assert "belong to a slot" not in old and "6,400 slots in use hold 6,400 different patterns" in old


def test_the_named_things_row_says_what_is_watched_and_whether_the_reader_is_on(tmp_path):
    """2026-09-20: the dashboard must show the self-model's named things, what they are being
    correlated with, how far the evidence has to go, and whether the reader is changing what
    the planner aims at."""
    es = {"names": ["caregiver", "toy_ball", "mobile"], "teacher": "wellness",
          "presence_mean": [0.8, 0.3, 0.9], "l2_link": [0.42, -0.05, 0.11],
          "l2_ticks": 7, "windows_needed": 11, "l2_counted_names": [], "l2_window_moments": 10000,
          "reader": {"gate_open": False, "applied_weight": 0.0, "weight": 0.25,
                     "why": "no entity has passed the evidence test yet"}}
    h = _render(tmp_path, extra={"entity_self": es})
    assert "Named things the self-model watches" in h
    assert "3 named things" in h and "nothing counted yet" in h
    assert ("caregiver present 80 of 100 model steps, link 42% of the way from &#x27;no link&#x27;" in h
            or "caregiver present 80 of 100 model steps, link 42% of the way from 'no link'" in h)
    assert "11 windows of 10,000 model steps" in h and "7 windows lived" in h
    assert "The reader that would turn a counted link into a goal the planner aims at is off" in h
    (tmp_path / "on").mkdir()
    es2 = dict(es, reader=dict(es["reader"], gate_open=True, applied_weight=0.25))
    h2 = _render(tmp_path / "on", extra={"entity_self": es2})
    assert "is ON: each unit of link times presence adds 25% of a wellness unit" in h2
    (tmp_path / "none").mkdir()
    assert "Named things the self-model watches" not in _render(tmp_path / "none")
