# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Reconstruction means the decoders; `pe` is predicted change; no cell reads None.

On 2026-08-27 H34 - the first run whose model could read its own action - was declared
"RECONSTRUCTION BROKEN on 3 senses, up to 6x next-state error" by the dashboard, the
watchdog and the overseer, while its next-state error was the best on record. The rule
read the field `pe`: the prediction heads applied to THIS moment, scored against THIS
moment. With future-only heads that is the model's guess of the next instant minus the
present - predicted change - which is small for a model that predicts persistence and
grows the moment it predicts movement. The real reconstruction, the decoders' error, was
0.0057 and healthy, logged only as a mean inside the training records.

These tests hold every reader to the decoders, keep `pe` on the page under its real name,
and make sure a missing value renders as a dash.
"""
import importlib
import inspect
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
TD = importlib.import_module("test_dashboard_metric_semantics")
from mimo_bridge.tools import dashboard, overseer
from mimo_bridge import lem_model as LM, train_mimo as TM


def _rec(step, pe, npe, rd=None):
    r = {"type": "mimo", "step": step, "timestamp": 0, "elapsed_s": 600, "phase": "mature",
         "n_envs": 8, "model_steps_per_s": 4.0, "env_dt_s": 0.01, "env_steps_per_s": 32.0,
         "pe": pe, "next_pe": npe, "curiosity_drive": 0.5,
         "movement": {"local_speed": 9.0, "stillness_debt": 0.0, "movement_wellness": 0.5, "self_factor": 1.0},
         "cmd_response_cos_held": 0.1, "cmd_response_ratio": 0.5, "cmd_mag_rad": 0.4,
         "contract": {"exec_seen": step - 100, "nf_aligned": step - 100, "step": step}}
    if rd is not None:
        r["recon_dec"] = rd
    return r


# ------------------------------------------------------------ the sources agree
def test_the_trainer_emits_the_decoders_error_per_sense():
    assert "recon_dec_per_modality" in inspect.getsource(LM)
    assert "last_recon_dec" in inspect.getsource(LM.LemLite.step)
    src = inspect.getsource(TM)
    assert '"recon_dec":' in src and 'acc["rd"]' in src


def test_pe_is_documented_as_predicted_change():
    src = inspect.getsource(LM.LemLite.step)
    assert "PREDICTED CHANGE, not reconstruction" in src


def test_dashboard_reads_reconstruction_from_the_decoders_never_from_pe():
    src = inspect.getsource(dashboard)
    assert 'window_means(seg, "recon_dec"' in src
    assert 'info["change_win"] = window_means(seg, "pe"' in src
    assert 'info["recon_win"] = window_means(seg, "pe"' not in src
    assert dashboard.doc_for("Predicted change of vision") is not None, "the row has no popup explanation"


def test_overseer_regression_rule_reads_the_prediction_not_predicted_change():
    src = inspect.getsource(overseer)
    assert 'ys = [(r.get("next_pe") or {}).get(m)' in src
    assert 'ys = [r["pe"].get(m)' not in src


# ------------------------------------------------------------ the page, rendered
def test_predicted_change_rising_is_not_a_fault(tmp_path):
    """H34's shape: pe far above next-state, decoders healthy and below it."""
    lo_next = {"vision": 0.025, "tactile": 0.03, "proprioception": 0.004}
    pe_hi = {"vision": 0.14, "tactile": 0.06, "proprioception": 0.013}
    rd_ok = {"vision": 0.006, "tactile": 0.005, "proprioception": 0.002}
    recs = [_rec(101 + i, pe_hi, lo_next, rd_ok) for i in range(16)]
    h = TD._render(tmp_path, recs=recs)
    assert "RECONSTRUCTION BROKEN" not in h
    assert "is broken" not in h.split("</style>")[-1].split("var DOCS")[0] if "var DOCS" in h else "FAULT - re-describing vision" not in h
    assert "FAULT - re-describing vision" not in h and "FAULT - re-describing tactile" not in h
    # rewritten 2026-09-23 (verify-fix-dashboards): the chip names the measured ratio and its
    # line instead of the status word "healthy" (the style guide writing rules, DESIGN.md section 11)
    assert "reconstruction intact (worst" in h and "a working model stays under 100%" in h
    assert "Predicted change of" in h
    assert "Re-describing" in h


def test_decoders_above_next_state_is_still_a_fault(tmp_path):
    lo_next = {"vision": 0.025, "tactile": 0.03, "proprioception": 0.004}
    pe = {"vision": 0.02, "tactile": 0.02, "proprioception": 0.003}
    rd_bad = {"vision": 0.5, "tactile": 0.4, "proprioception": 0.2}
    recs = [_rec(101 + i, pe, lo_next, rd_bad) for i in range(16)]
    h = TD._render(tmp_path, recs=recs)
    assert "RECONSTRUCTION BROKEN on 3 sense(s)" in h
    assert "FAULT - re-describing vision is broken" in h


def test_a_log_without_decoder_error_reports_not_logged_rather_than_a_fault(tmp_path):
    lo_next = {"vision": 0.025, "tactile": 0.03, "proprioception": 0.004}
    pe_hi = {"vision": 0.14, "tactile": 0.06, "proprioception": 0.013}
    recs = [_rec(101 + i, pe_hi, lo_next) for i in range(16)]
    h = TD._render(tmp_path, recs=recs)
    assert "RECONSTRUCTION BROKEN" not in h
    assert "reconstruction not logged on this run" in h


def test_older_logs_use_the_decoders_mean_from_the_training_records(tmp_path):
    lo_next = {"vision": 0.025, "tactile": 0.03, "proprioception": 0.004}
    pe_hi = {"vision": 0.14, "tactile": 0.06, "proprioception": 0.013}
    recs = [_rec(101 + i, pe_hi, lo_next) for i in range(16)]
    trs = [{"type": "training", "training_step": 101 + i, "timestamp": 0,
            "loss_breakdown": {"recon": 0.006, "recon_weight": 0.02}} for i in range(16)]
    h = TD._render(tmp_path, recs=recs + trs)
    assert "RECONSTRUCTION BROKEN" not in h
    assert "mean over senses" in h
    assert "This log predates per-sense recording" in h


def test_no_cell_reads_none():
    page = "<table><tr><td>H8</td><td>crashed</td><td>None</td></tr></table>"
    assert "None" not in dashboard.tidy_html(page)
    assert "&mdash;" in dashboard.tidy_html(page)
    src = inspect.getsource(dashboard)
    assert ".write(tidy_html(" in src, "the page must be tidied at the one place it is written"


# ------------------------------------------------------------ the model produces it
def test_a_training_step_reports_the_decoders_error_for_real_senses(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    m = LM.LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                   overrides={"training": {"steps_per_phase": 3000}})
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path), tag="_rd", vision=False, seed=2)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9)); dev["exploration_noise"] = 0.05
        held = prev = None; seen = None
        for i in range(6):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
            seen = res.recon_dec_per_modality
    finally:
        src.close()
    assert seen, "no decoder error reported"
    assert set(seen) <= {"vision", "tactile", "proprioception"}
    assert "tactile" in seen and "proprioception" in seen
    assert all(np.isfinite(v) and v >= 0 for v in seen.values())
    assert set(seen).isdisjoint({"audio", "intermind"}), "synthetic senses have no reconstruction"


def test_contact_turnover_is_logged_and_explained():
    """Touch is a discrete-event channel and that was invisible: about two patches start
    touching and two stop every 50 ms against 5.6 in contact, the set changes on 95% of
    steps, and those steps carry 98% of the touch prediction error. The counts that WERE
    logged - contacts, active patches, pressure - stayed flat across the whole of H34 and
    H35 while the set underneath them turned over completely, so "did the world change?"
    could not be answered from the record."""
    from mimo_bridge import train_mimo as TM
    src = inspect.getsource(TM)
    assert '"contact_turnover"' in src
    assert 'acc["born"]' in src and 'acc["died"]' in src and 'acc["turn_n"]' in src
    assert "born=0.0, died=0.0, turn_n=0" in src, "the accumulator must be initialised and reset"
    # doc_for returns the ROW_DOC KEY, not the entry
    key = dashboard.doc_for("How fast the skin's contacts turn over")
    assert key is not None, "the row renders with no explanation"
    d = dashboard.ROW_DOC[key]
    assert "discrete-event" in d["t"]
    assert "98%" in d["f"], "the reason the row matters must be on the row"


def test_the_turnover_row_renders_when_the_run_logs_it(tmp_path):
    recs = [dict(_rec(101 + i, {"vision": 0.02, "tactile": 0.03, "proprioception": 0.004},
                      {"vision": 0.02, "tactile": 0.03, "proprioception": 0.004}),
                 contact_turnover={"born": 2.1, "died": 2.0}, touch_active_patches=5.6)
            for i in range(16)]
    h = TD._render(tmp_path, recs=recs)
    assert "How fast the skin&#x27;s contacts turn over" in h or "How fast the skin's contacts turn over" in h
    # 2.0 is printed "2" since the section-11 formatters (plain_numbers.fixed, 2026-09-23)
    assert "2.1 patches start touching" in h and "2 patches stop" in h


def test_the_step_log_does_not_call_predicted_change_reconstruction():
    """The dashboard, the watchdog and the overseer were all corrected on 2026-08-27; the
    trainer's own step line still said recon_pe for the same quantity, which is where a
    reader looks first."""
    from mimo_bridge import train_mimo as TM
    src = inspect.getsource(TM)
    assert "recon_pe %s" not in src, "the step line still labels predicted change as reconstruction"
    assert "predicted change %s" in src


# ------------------------------------------------- the rule under residual prediction
def test_residual_run_with_recon_above_next_state_is_NOT_a_fault(tmp_path):
    """With --residual-prediction on, the heads predict the CHANGE and get the present
    for free, so next-state error collapses by construction and reconstruction sits
    above it in a healthy model - measured 1.16x (H36) to 2.45x (H43) while the decoders
    trained at their 0.02 floor weight. The old premise fired on every healthy residual
    run for a week."""
    lo_next = {"vision": 0.02, "tactile": 0.025, "proprioception": 0.004}
    pe = {"vision": 0.02, "tactile": 0.02, "proprioception": 0.003}
    rd_steady = {"vision": 0.03, "tactile": 0.04, "proprioception": 0.009}   # above next
    recs = [_rec(101 + i, pe, lo_next, rd_steady) for i in range(16)]
    h = TD._render(tmp_path, args={"residual_prediction": "on"}, recs=recs)
    body = h.split("var D=", 1)[0]      # the static popup dict always carries the key
    assert "RECONSTRUCTION BROKEN" not in body
    assert "FAULT - re-describing" not in body
    assert "expected with residual prediction on" in h.lower() or "EXPECTED with residual" in h
    assert "reconstruction steady" in h


def test_residual_run_blowup_against_its_own_history_IS_a_fault(tmp_path):
    """The H30 damage signature - 0.007 to 0.9 inside a window - must still fire with
    residual on, judged against the run's own history instead of next-state error."""
    lo_next = {"vision": 0.02, "tactile": 0.025, "proprioception": 0.004}
    pe = {"vision": 0.02, "tactile": 0.02, "proprioception": 0.003}
    rd_ok = {"vision": 0.007, "tactile": 0.006, "proprioception": 0.003}
    rd_blown = {"vision": 0.9, "tactile": 0.8, "proprioception": 0.6}
    recs = ([_rec(101 + i, pe, lo_next, rd_ok) for i in range(8)]
            + [_rec(109 + i, pe, lo_next, rd_blown) for i in range(8)])
    h = TD._render(tmp_path, args={"residual_prediction": "on"}, recs=recs)
    assert "RECONSTRUCTION BROKEN" in h
    assert "FAULT - re-describing vision is broken" in h
    assert "its own value" in h
    assert "own recent value" in h


def test_the_fault_popup_carries_the_rows_numbers(tmp_path):
    """The popup used to be a static paragraph; the operator asked for the numbers the
    row text shows. A dynamic doc entry keyed by the exact row label is injected into
    the page's popup dict at render time."""
    lo_next = {"vision": 0.025, "tactile": 0.03, "proprioception": 0.004}
    pe = {"vision": 0.02, "tactile": 0.02, "proprioception": 0.003}
    rd_bad = {"vision": 0.5, "tactile": 0.4, "proprioception": 0.2}
    recs = [_rec(101 + i, pe, lo_next, rd_bad) for i in range(16)]
    h = TD._render(tmp_path, recs=recs)
    import json as _json
    js = h.split("var D=", 1)[1].split(";function op", 1)[0]
    docs = _json.loads(js)
    key = "FAULT - re-describing vision is broken"
    assert key in docs, "the exact row label must key a live popup entry"
    # rewritten 2026-09-23 (verify-fix-dashboards): the measured numbers are carried as their
    # ratio (0.5 against 0.025 = 20 times); a raw encoded error is never printed (DESIGN.md
    # section 11: never a mean squared error)
    assert "20 times what predicting the NEXT state costs" in docs[key]["w"], (
        "the popup must carry the measured numbers, not only prose")
    # and the diagnostic (non-fault) row carries its numbers too (0.006 against 0.025)
    rd_ok = {"vision": 0.006, "tactile": 0.005, "proprioception": 0.002}
    recs2 = [_rec(101 + i, pe, lo_next, rd_ok) for i in range(16)]
    h2 = TD._render(tmp_path / "b", recs=recs2)
    js2 = h2.split("var D=", 1)[1].split(";function op", 1)[0]
    docs2 = _json.loads(js2)
    key2 = "Re-describing vision now (diagnostic, not causal)"
    assert key2 in docs2 and "24% of the next-state error" in docs2[key2]["w"]
