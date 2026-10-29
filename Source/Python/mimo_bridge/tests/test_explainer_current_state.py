# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The explainer's selected-run box for a COMPLETED run segment while the supervisor is paused (Oct 15, 2026):
the page built for the completed B1K run, which ended at model step 494,025, must "identify training as paused at
494025" and must not present the production record's run seg41r_1539, which ended at 489,350, as the current state.
A wording correction of the same day: the control file establishes the production supervisor's pause, not that no
separately owned experiment trains. The run records its end in stage_done.json (train_mimo writes it at --max-steps);
the supervisor's control file reads pause."""
import json

from mimo_bridge.explainer.current_state import current_state, current_state_html


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _state(tmp_path, done, cmd="pause"):
    run, saved = tmp_path / "B1K_skip13_492800", tmp_path / "saved"
    run.mkdir(parents=True)
    if done is not None:
        _write(run / "stage_done.json", done)
    _write(saved / "training_continuation.json", {"active_run": str(tmp_path / "seg41r_1539")})
    _write(saved / "supervisor_control.json", {"cmd": cmd})
    stats = {"training": [{"training_step": 494025, "timestamp": 1791958561.0}]}
    return current_state(str(run), {"n_envs": 8}, stats, str(saved), [])


DONE = {"run": "B1K_skip13_492800", "ok": True, "steps": 494025, "finished_at": 1791958636.2247992}


def test_a_completed_segment_says_where_it_ended_and_that_training_is_paused(tmp_path):
    state = _state(tmp_path, DONE)
    assert state["run_ended"]["steps"] == 494025 and state["supervisor_paused"] is True
    page = current_state_html(state)
    assert "Completed run segment: it ended at model step 494,025" in page
    assert "stage_done.json records its end at model step 494,025" in page
    assert "The production supervisor is paused" in page and "seg41r_1539, another run this page does not show" in page
    assert "Selected experiment or earlier run segment" not in page


def test_without_a_pause_or_a_completed_end_the_box_claims_neither(tmp_path):
    page = current_state_html(_state(tmp_path, DONE, cmd="run"))
    assert "Completed run segment" in page and "The production supervisor is paused" not in page
    page = current_state_html(_state(tmp_path / "unfinished", {"ok": False, "steps": 494025}))
    assert "Selected experiment or earlier run segment" in page
    assert "Completed run segment" not in page and "The production supervisor is paused" not in page
    page = current_state_html(_state(tmp_path / "none", None))
    assert "Completed run segment" not in page and "The production supervisor is paused" not in page


def test_the_production_records_repair_run_that_ended_says_where_it_ended(tmp_path):
    """The independent review of Oct 15, 2026 (S4): seg41r_1539 is the production record's run and the supervisor's
    pause names it as the repair run; its stage_done.json records its end at 489,350, yet its box read "Repair run:
    bounded, with no automatic restart when it ends" with no end, as if still under way."""
    import json as _json
    run, saved = tmp_path / "seg41r_1539", tmp_path / "saved"
    run.mkdir(parents=True)
    (run / "stage_done.json").write_text(_json.dumps({"ok": True, "steps": 489350, "finished_at": 1791769936.0}),
                                         encoding="utf-8")
    saved.mkdir()
    (saved / "training_continuation.json").write_text(_json.dumps({"active_run": str(run)}), encoding="utf-8")
    (saved / "supervisor_control.json").write_text(_json.dumps({"cmd": "pause", "repair_run": str(run)}),
                                                   encoding="utf-8")
    state = current_state(str(run), {"n_envs": 8}, {"training": [{"training_step": 489350}]}, str(saved), [])
    page = current_state_html(state)
    assert "Repair run: bounded; it ended at model step 489,350 and is not restarted automatically" in page
    assert "stage_done.json records its end at model step 489,350" in page
    assert "with no automatic restart when it ends" not in page
    assert "production record still names" not in page
