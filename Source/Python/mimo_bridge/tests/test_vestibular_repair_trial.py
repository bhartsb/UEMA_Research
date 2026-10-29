# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The vestibular encoder repair's two-arm trial, its pure parts
(docs/gain_of_function_v4/vestibular_repair_trial_2026-10-02.py): the window, the float32 floor, the readings, the
switch's exactness, lines (b1), (b2) and (c), the control, the kill reading, the card's licence and the arm commands.
Synthetic records only; no trainer, no checkpoint, no file outside tmp_path."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

DOCS = Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4"
SCRIPT = DOCS / "vestibular_repair_trial_2026-10-02.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("vestibular_repair_trial", SCRIPT)
START = 455800
CKPT = "C:\\runs\\H84\\checkpoints\\ckpt_step_455800"
F = T.FLOOR
RATIOS = {"vision": 0.80, "tactile": 1.00, "audio": 1.00, "proprioception": 0.98}


def _rec(step, collapse=0.03, between=0.09, persistence=1e-5, ios=-0.4, ioc=0.01, ratios=None, nan=0, live=1.0,
         repair=True, body2=None):
    bodies = [collapse] * 8
    if body2 is not None:
        bodies[2] = body2
    ve = {"liveness": {"encoder_registry.encoders.vestibular.mlp.%d" % i: live for i in (0, 4, 8)}}
    if repair:
        ve["repair"] = {"token_between_bodies": between, "token_change": 0.004,
                        "balance_probe": {"r2": {"head_tilt": 0.1, "trunk_roll": 0.05},
                                          "r2_shuffled": {"head_tilt": -0.01, "trunk_roll": -0.02}}}
    return {"type": "mimo", "step": step, "nan_steps": nan, "model_steps_per_s": 0.47,
            "token_temporal_std": {"vestibular": collapse},
            "token_temporal_std_bodies": {"vestibular": bodies},
            "vestibular": ve,
            "prediction_audit": {"modalities": {"vestibular": {
                "transitions": 800, "prediction_mse": persistence * (1 - ios) if ios is not None else 0.0,
                "persistence_mse": persistence, "improvement_over_persistence": ios, "shuffled_transitions": 40,
                "improvement_over_shuffled": ioc, "sensitivity": 0.03}}},
            "next_frame_norm": {"ratio": dict(ratios or RATIOS), "share_normalized": {"vestibular": 0.15},
                                "replay_share_normalized": {"vestibular": 0.05}, "capped_steps": {"vestibular": 0}}}


def _dead(step, **kw):
    kw.setdefault("ratios", RATIOS)
    return _rec(step, collapse=1.0e-7, between=None, persistence=5e-13, ios=None, ioc=-0.002, repair=False, **kw)


def _arm(make, n=20, **kw):
    return [make(START + 100 * (i + 1), **kw) for i in range(n)]


def _switch(f32=0.0, bf16=0.0, f64=4.59e-8, divided_by=4096.0, spread=18.17, mode="rescale_restart"):
    return {"type": "vestibular_encoder_repair", "step": START, "switch": {
        "switched": True, "divided_by": divided_by, "bias_spread_after": spread, "mode": mode,
        "runtime_restart": {"next_frame_norm": {"dropped_readings": 111398}},
        "checks": {"float64": {"change_max_abs": f64, "largest": 3.0294}, "float32": {"change_max_abs": f32},
                   "bfloat16": {"change_max_abs": bf16}}}}


def _card(checkpoint=CKPT, mode="rescale_restart", em_change=0.0, ring=True, finished=True, complete=True,
          device="cuda:0"):
    block = {"change_max_abs": em_change, "largest": 3.0, "line": 2e-5, "within": True}
    em = {p: {k: dict(block) for k in ("vestibular_token", "core_every_position", "core_newest_step", "head_vision",
                                       "head_tactile", "head_audio", "head_proprioception", "head_vestibular")}
          for p in ("float32", "bfloat16")}
    return {"finished": "2026-10-03 01:00:00" if finished else None,
            "rows": [{"checkpoint": checkpoint, "complete": complete, "device": device, "EM": em,
                      "EM_ring_bit_identical": ring, "switch": {"switched": True},
                      "EM32": {"vestibular_token": {"change_max_abs": 0.0064, "change_rms": 0.0016}},
                      "EM_dead_token_spread_between_bodies": {"float32": 2.4e-7, "bfloat16": 0.0}}],
            "verdicts": {"MODE": {"mode": mode, "licensed_by_the_card": True, "why": "the token revives in bf16"}}}


def _records(on, off, switch=None, mode="rescale_restart", stopped=False):
    on_rec = dict(arm="on", exit=0, checkpoint=CKPT, mode=mode, final_step=START + 2000,
                  window_reports=T.window(on, START), switch_record=switch or _switch(mode=mode),
                  switch_records_found=1, kill_reading={"stop": True} if stopped else None)
    off_rec = dict(arm="off", exit=0, checkpoint=CKPT, final_step=START + 2000, window_reports=T.window(off, START),
                   switch_record=None, switch_records_found=0)
    return off_rec, on_rec


# ------------------------------------------------------------------------------------------------ the constants
def test_the_floor_is_the_planner_probes_float32_floor_for_a_token_of_size_1():
    probe = _load("probe_planner_candidates", DOCS / "probe_planner_candidates.py")
    ones = np.ones(192, dtype=np.float32)
    assert T.FLOOR == pytest.approx(192 * 2.0 ** -23)
    assert probe.float32_floor(ones, ones) == pytest.approx(T.FLOOR_SQ, rel=1e-12)
    # production's dead token reads at most 2.4 grid steps per number (vestibular_token_2026-10-02.md): far under it
    assert 2.8e-7 < T.FLOOR / 80


def test_the_stop_lines_are_the_watchers():
    assert T.stop_multiple("audio") == pytest.approx(1.43 * 1.2)
    assert T.stop_multiple("vision") == pytest.approx(3.29 * 1.2)


def test_the_window_is_updates_1001_to_2000():
    rows = T.window(_arm(_rec), START)
    assert [r["step"] for r in rows] == [START + s for s in range(1100, 2001, 100)]
    assert len(rows) == (T.WIN_TO - T.WIN_FROM) // T.REPORT_EVERY == 10


# ------------------------------------------------------------------------------------------------ the verdict
def test_a_revived_token_against_production_passes_every_line():
    off, on = _records(_arm(_rec), _arm(_dead))
    v = T.judge(off, on, _card())
    assert v["passed"] is True, v
    assert all(v["lines"][k]["met"] is True for k in ("a", "b1", "b2", "c"))
    assert v["control"]["dead"] is True
    assert v["lines"]["b2"]["own_command"]["n"] == 10


def test_a_token_that_stays_dead_fails_the_spread_and_the_measurability_lines():
    off, on = _records(_arm(_dead), _arm(_dead))
    on["window_reports"] = T.window([_rec(START + 100 * (i + 1), collapse=1e-7, between=2e-7, persistence=5e-13,
                                          ios=None) for i in range(20)], START)
    v = T.judge(off, on, _card())
    assert v["passed"] is False
    assert v["lines"]["b1"]["met"] is False and v["lines"]["b2"]["met"] is False


def test_body_2_is_left_out_of_the_collapse_line_and_every_other_body_counts():
    off, on = _records(_arm(_rec, body2=1e-7), _arm(_dead))
    assert T.judge(off, on, _card())["lines"]["b1"]["met"] is True
    recs = _arm(_rec)
    for r in recs:
        r["token_temporal_std_bodies"]["vestibular"][3] = 1e-7
    off, on = _records(recs, _arm(_dead))
    v = T.judge(off, on, _card())
    assert v["lines"]["b1"]["met"] is False and v["lines"]["b1"]["collapse_every_kept_body"] is False


def test_a_spread_over_time_without_a_spread_between_bodies_is_not_enough():
    # stored tokens: the encoder's weight changes alone raise the collapse watch; the spread between the bodies at one
    # model step is read through one encoder and stays at the dead token's rounding
    off, on = _records(_arm(_rec, between=2e-7), _arm(_dead))
    v = T.judge(off, on, _card())
    assert v["lines"]["b1"]["met"] is False and v["lines"]["b1"]["between_bodies"] is False


def test_readings_computed_on_a_change_under_the_floor_are_not_measurable():
    off, on = _records(_arm(_rec, persistence=1e-12), _arm(_dead))
    v = T.judge(off, on, _card())
    assert v["lines"]["b2"]["met"] is False and v["lines"]["b2"]["change_above_floor"] is False
    off, on = _records(_arm(_rec, ios=None), _arm(_dead))
    assert T.judge(off, on, _card())["lines"]["b2"]["readings_computed"] is False


def test_a_control_whose_token_varies_gives_no_verdict():
    off, on = _records(_arm(_rec), _arm(_rec, repair=False))
    v = T.judge(off, on, _card())
    assert v["passed"] is None and "control" in v["why_none"]


def test_the_switch_must_read_exactly_zero_in_float32_and_bfloat16():
    off, on = _records(_arm(_rec), _arm(_dead), switch=_switch(f32=1e-9))
    assert T.judge(off, on, _card())["lines"]["a"]["met"] is False
    off, on = _records(_arm(_rec), _arm(_dead), switch=_switch(divided_by=4000.0))
    assert T.judge(off, on, _card())["lines"]["a"]["met"] is False
    off, on = _records(_arm(_rec), _arm(_dead))
    assert T.judge(off, on, _card(em_change=1e-8))["lines"]["a"]["met"] is False
    assert T.judge(off, on, _card(ring=False))["lines"]["a"]["met"] is False
    # the card read another checkpoint: no EM reading on this one, so no verdict
    v = T.judge(off, on, _card(checkpoint="C:\\runs\\other\\ckpt_step_450525"))
    assert v["lines"]["a"]["met"] is None and v["passed"] is None
    # the off arm switched: not production's behaviour
    off["switch_records_found"] = 1
    assert T.judge(off, on, _card())["lines"]["a"]["met"] is False


def test_the_float32_mode_is_not_exact_in_the_runs_precision():
    off, on = _records(_arm(_rec), _arm(_dead), mode="rescale_restart_fp32")
    v = T.judge(off, on, _card(mode="rescale_restart_fp32"))
    assert v["lines"]["a"]["met"] is False and "float32" in v["lines"]["a"]["note"]
    assert v["passed"] is False


def test_another_sense_past_the_watchs_stop_line_fails_c_and_under_it_passes():
    worse = dict(RATIOS, audio=RATIOS["audio"] * 1.8)        # ln 1.8 = 0.588, the line ln 1.716 = 0.540
    off, on = _records(_arm(_rec, ratios=worse), _arm(_dead))
    v = T.judge(off, on, _card())
    assert v["lines"]["c"]["met"] is False and v["lines"]["c"]["senses"]["audio"]["past_stop_line"] is True
    assert v["passed"] is False
    mild = dict(RATIOS, audio=RATIOS["audio"] * 1.6, vision=RATIOS["vision"] * 3.5)
    off, on = _records(_arm(_rec, ratios=mild), _arm(_dead))
    assert T.judge(off, on, _card())["lines"]["c"]["met"] is True


def test_a_stopped_arm_fails_and_a_missing_record_or_window_gives_no_verdict():
    off, on = _records(_arm(_rec), _arm(_dead), stopped=True)
    assert T.judge(off, on, _card())["passed"] is False
    assert T.judge(None, on, _card())["passed"] is None
    off, on = _records(_arm(_rec, n=15), _arm(_dead))
    v = T.judge(off, on, _card())
    assert v["passed"] is None and "window" in v["why_none"]
    off, on = _records(_arm(_rec), _arm(_dead))
    on["failed"] = "refused: the graphics card's measurement licensed no mode"
    assert T.judge(off, on, _card())["passed"] is None


# ------------------------------------------------------------------------------------------------ the kill reading
def test_the_kill_reading_waits_for_five_reports_after_report_1_and_stops_past_the_stop_line():
    off = _arm(_dead)
    same = _arm(_rec)
    assert T.kill_reading(same[:5], off, START) is None                 # reports 2-5: four after report 1
    stop, d = T.kill_reading(same[:6], off, START)                      # update 600: five
    assert stop is False and d["k"] == 5 and d["steps"][0] == START + 200
    # report 1 alone at ten times the level never counts
    spike = copy.deepcopy(same)
    spike[0]["next_frame_norm"]["ratio"]["audio"] = 10.0
    assert T.kill_reading(spike[:6], off, START)[0] is False
    worse = _arm(_rec, ratios=dict(RATIOS, tactile=2.0))
    stop, d = T.kill_reading(worse[:6], off, START)
    assert stop is True and d["crossed"] == ["tactile"]
    # the newest 10 matched reports are read, never more
    assert T.kill_reading(same, off, START)[1]["k"] == 10


def test_the_kill_reading_stops_on_a_non_finite_step_and_on_two_low_liveness_reports():
    off = _arm(_dead)
    nan = _arm(_rec)
    nan[1]["nan_steps"] = 1
    assert T.kill_reading(nan[:2], off, START) == (True, {"nonfinite": True})
    low = _arm(_rec)
    low[2] = _rec(START + 300, live=0.8)
    assert T.kill_reading(low[:6], off, START)[0] is False              # one report under 0.9
    low[3] = _rec(START + 400, live=0.8)
    stop, d = T.kill_reading(low[:6], off, START)
    assert stop is True and d["liveness_low_at"] == START + 400


# ------------------------------------------------------------------------------------------------ the card and commands
def test_the_card_licenses_a_mode_only_on_its_own_checkpoint_read_to_its_end_on_the_card():
    assert T.card_mode(_card(), CKPT)[0] == "rescale_restart"
    assert T.card_mode(_card(), "C:/runs/H84/checkpoints/ckpt_step_455800")[0] == "rescale_restart"
    assert T.card_mode(_card(checkpoint="C:\\runs\\other\\ckpt_step_450525"), CKPT)[0] is None
    assert T.card_mode(_card(finished=False), CKPT)[0] is None
    assert T.card_mode(_card(complete=False), CKPT)[0] is None
    assert T.card_mode(_card(device="cpu"), CKPT)[0] is None
    assert T.card_mode(_card(mode=None), CKPT)[0] is None
    assert T.card_mode(None, CKPT)[0] is None


PRODUCTION = ["python.exe", "-u", "-m", "mimo_bridge.train_mimo", "--n-envs", "8", "--report-every", "100",
              "--checkpoint-every-min", "60.0", "--minutes", "0", "--out", "C:\\Saved\\run", "--resume",
              "C:\\Saved\\old\\ckpt_step_451425", "--checkpoint-storage", "C:\\Saved\\run\\checkpoints",
              "--caregiver-record-dir", "D:\\rec", "--observer", "record", "--observer-reserve-gb", "10",
              "--vestibular-sense", "on", "--persistent-memory-repair", "centred",
              "--persistent-memory-constant-return", "0", "--fill-ring-first",
              "--latent-body-hierarchy", "shadow", "--latent-body-config", "C:\\cfg.json",
              "--latent-body-output", "D:\\latent", "--latent-body-resume", "auto", "--latent-body-log-mb", "8192"]


def test_the_arms_run_productions_command_with_only_the_folders_schedule_shadow_and_repair_changed(monkeypatch,
                                                                                                    tmp_path):
    monkeypatch.setattr(T.ST, "production_command", lambda: list(PRODUCTION))
    off = T.arm_command("off", CKPT, tmp_path / "off", START + 2000)
    on = T.arm_command("on", CKPT, tmp_path / "on", START + 2000, mode="rescale_restart")
    for cmd, flag in ((off, "off"), (on, "rescale_restart")):
        assert cmd.count("--vestibular-encoder-repair") == 1
        assert cmd[cmd.index("--vestibular-encoder-repair") + 1] == flag
        assert cmd[cmd.index("--resume") + 1] == CKPT and cmd.count("--resume") == 1
        assert cmd[cmd.index("--max-steps") + 1] == str(START + 2000)
        assert cmd[cmd.index("--latent-body-hierarchy") + 1] == "off" and "--latent-body-output" not in cmd
        assert cmd[cmd.index("--observer") + 1] == "off" and "--observer-reserve-gb" not in cmd
        assert cmd[cmd.index("--persistent-memory-constant-return") + 1] == "0"
        assert "--fill-ring-first" in cmd and cmd[cmd.index("--n-envs") + 1] == "8"
        assert "C:\\Saved\\run" not in cmd and "D:\\rec" not in cmd
    with pytest.raises(ValueError):
        T.arm_command("on", CKPT, tmp_path / "on", START + 2000, mode=None)


def test_both_arms_refuse_without_the_cards_licence_and_run_nothing(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    T._write(tmp_path / "card.json", _card(mode=None))
    ran = []
    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: ran.append(a))
    args = T.argparse.Namespace(root=str(tmp_path), card=str(tmp_path / "card.json"), checkpoint=CKPT,
                                branch_root=str(tmp_path))
    assert T.run_arm("off", args) == 2
    assert T._read(out / "arm_off.json")["failed"].startswith("refused")
    T._write(out / "arm_off.json", dict(arm="off", exit=0, checkpoint=CKPT, final_step=START + 2000))
    assert T.run_arm("on", args) == 2
    assert T._read(out / "arm_on.json")["failed"].startswith("refused") and not ran
    # a card for another checkpoint licenses nothing here either
    T._write(tmp_path / "card.json", _card(checkpoint="C:\\runs\\other\\ckpt_step_450525"))
    assert T.run_arm("on", args) == 2 and not ran
