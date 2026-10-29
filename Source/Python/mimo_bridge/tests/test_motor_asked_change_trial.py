# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tests of the pure parts of the asked-change trial's driver (docs/gain_of_function_v4/
motor_asked_change_trial_2026-10-03.py): the readings it takes from stats reports, its declared lines S1, S2 and C,
its kill reading, the arms' command, and its refusals. No trainer, no world, no torch."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4"


def _load():
    spec = importlib.util.spec_from_file_location("motor_asked_change_trial", DOCS / "motor_asked_change_trial_2026-10-03.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load()
START = 470000


def swap_block(own, ctl, n=5, bodies=8):
    """A per-body swap block for every sense: each body's mean own-command error `own` and control error `ctl`."""
    def per(o, c):
        return dict(size_matched_transitions=[n] * bodies, size_matched_own_prediction_mse=[o] * bodies,
                    size_matched_action_mse=[c] * bodies, shuffled_transitions=[n] * bodies,
                    matched_prediction_mse=[o] * bodies, shuffled_action_mse=[c * 2.0] * bodies)
    return dict(senses={s: dict(per_body=per(own, ctl)) for s in T.OTHER_SENSES})


def report(step, own=1.0, ctl=1.0, ratios=None, live=1.0, nan=0, jitter=0.0):
    ratios = ratios or {}
    r = dict(type="mimo", step=step, nan_steps=nan,
             next_frame_norm=dict(ratio={s: ratios.get(s, 0.5) for s in T.OTHER_SENSES + ("vestibular",)}),
             prediction_audit_hold_first=swap_block(own + jitter, ctl),
             prediction_audit_per_body=swap_block(own + jitter, ctl),
             tenure_pe=dict(pose=0.9), tenure_gap=dict(pose=0.1), tenure_events=20,
             contact_control=dict(gate_self_open=False, gate_world_open=False, gate_self_open_share=0.0,
                                  gate_world_open_share=0.0),
             controllability=dict(gate_open=False, gate_mean=0.0, gate_line=1.0),
             liveness=dict(layers={T.PATH_LAYER: live}, below_line=[]),
             motor_asked_change=dict(output_rms=0.2, output_weight_rms=0.01, liveness={"net.0": live},
                                     handle_reach=dict(spread=0.9), features_rms=dict(position=1.0)),
             model_steps_per_s=0.5)
    return r


def arm_records(own, ctl=1.0, **kw):
    """3,000 updates of reports; the own-command error alternates a little so the resampling has a spread."""
    return [report(START + 100 * i, own=own, ctl=ctl, jitter=(0.002 if i % 2 else -0.002), **kw)
            for i in range(1, 31)]


def arm_record(recs, arm, switch=None):
    return dict(arm=arm, exit=0, start_step=START, final_step=START + T.UPDATES, checkpoint="ckpt_step_%d" % START,
                windows=[T.window(recs, START, lo, hi) for lo, hi in T.WINDOWS], kill_reading=None,
                switch_records_found=0 if switch is None else 1, switch_record=switch)


def exact_switch():
    return dict(type=T.SWITCH_TYPE, step=START,
                switch=dict(exact=True, rows=64, output_weight_max_abs=0.0, version="motor_asked_change_v1",
                            at_step=START, float32=dict(change_max_abs=0.0, bit_identical=True),
                            autocast=dict(change_max_abs=0.0, bit_identical=True)))


def readout(position_r2, turn_r2=0.001, control=0.0005):
    groups = dict(asked_position_change_cm=dict(r2=position_r2, control_q975=control),
                  asked_turn_deg=dict(r2=turn_r2, control_q975=control))
    return dict(pools=dict(without_body_2=dict(pooled=dict(token_bf16_grid=dict(groups=groups)))))


# ------------------------------------------------------------------------------------------------ readings
def test_the_swap_cells_leave_body_2_out_and_hold_sums():
    r = report(START + 100, own=0.8, ctl=1.0)
    cells = T.swap_cells(r, "prediction_audit_hold_first", "proprioception")
    assert len(cells) == 7 and all(c[0] == 5 for c in cells)
    assert cells[0][1] == pytest.approx(0.8 * 5) and cells[0][2] == pytest.approx(5.0)
    absolute = T.swap_cells(r, "prediction_audit_hold_first", "proprioception", "shuffled")
    assert absolute[0][2] == pytest.approx(10.0)
    assert T.swap_cells({}, "prediction_audit_hold_first", "proprioception") == []
    # a body without transitions is left out
    r["prediction_audit_hold_first"]["senses"]["proprioception"]["per_body"]["size_matched_transitions"][0] = 0
    assert len(T.swap_cells(r, "prediction_audit_hold_first", "proprioception")) == 6


def test_the_own_command_reading_is_one_minus_the_error_ratio_with_a_seeded_standard_error():
    cells = [(5, 4.0, 5.0), (5, 3.0, 5.0), (5, 5.0, 5.0), (5, 4.0, 5.0)]
    a = T.own_command_reading(cells)
    assert a["value"] == pytest.approx(1 - 16.0 / 20.0) and a["transitions"] == 20 and a["cells"] == 4
    assert a["se"] > 0 and a == T.own_command_reading(cells)          # the same seed, the same number
    same = T.own_command_reading([(5, 4.0, 5.0)] * 6)
    assert same["value"] == pytest.approx(0.2) and same["se"] == pytest.approx(0.0, abs=1e-12)
    assert T.own_command_reading([]) is None and T.own_command_reading([(5, 1.0, 0.0)]) is None


def test_the_hold_loss_ratio_and_the_paths_live_share():
    assert T.hold_loss_ratio(report(1)) == pytest.approx(0.9)
    assert T.hold_loss_ratio({}) is None
    assert T.path_live(report(1, live=0.7)) == 0.7
    assert T.path_live(dict(motor_asked_change=dict(liveness={"net.0": 0.4}))) == 0.4
    assert T.path_live({}) is None


# ------------------------------------------------------------------------------------------------ the lines
def test_s1_needs_the_reading_and_its_difference_above_their_lines():
    off = T.window(arm_records(own=1.0), START, *T.JUDGED)
    on = T.window(arm_records(own=0.9), START, *T.JUDGED)
    s1 = T.judge_s1(on, off)
    assert s1["met"] is True and s1["on"]["value"] == pytest.approx(0.1, abs=0.003)
    assert s1["on"]["transitions"] == 10 * 7 * 5
    assert s1["goal_ladder"]["share_of_reference"] == pytest.approx(0.1 / 0.08, rel=0.05)
    # the same reading in both arms: no difference, not met
    assert T.judge_s1(on, on)["met"] is False
    # a reading inside its own noise: not met
    noisy = T.window([report(START + 2000 + 100 * i, own=1.0 + (0.3 if i % 2 else -0.3004)) for i in range(1, 11)],
                     START, *T.JUDGED)
    assert T.judge_s1(noisy, off)["met"] is False
    # no size-matched reading in the stats (code without the swap test's arm): no reading, said in words
    bare = [dict(type="mimo", step=START + 2000 + 100 * i) for i in range(1, 11)]
    none = T.judge_s1(T.window(bare, START, *T.JUDGED), off)
    assert none["met"] is None and "size-matched" in none["why_none"]


def test_s2_is_read_out_2s_rule_on_the_on_arm_and_not_on_the_off_arm():
    assert T.judge_s2(readout(0.30), readout(0.004))["met"] is True
    assert T.judge_s2(readout(0.03), readout(0.004))["met"] is False            # under one handle's share
    assert T.judge_s2(readout(0.30, control=0.5), readout(0.004))["met"] is False   # under its permuted control
    assert T.judge_s2(readout(0.001, turn_r2=0.2), readout(0.004))["met"] is True   # the turn alone is enough
    assert T.judge_s2(readout(0.30), readout(0.30))["met"] is False             # the control carries it too
    assert T.judge_s2(None, readout(0.004))["met"] is None


def test_the_verdict_passes_only_with_s1_s2_and_c_and_names_the_other_outcomes():
    off = arm_record(arm_records(own=1.0), "off")
    on = arm_record(arm_records(own=0.9), "on", switch=exact_switch())
    v = T.judge(off, on, readout(0.004), readout(0.30))
    assert v["passed"] is True and "uses the body's own command" in v["outcome"]
    assert v["lines"]["C"]["met"] is True and len(v["trend"]["windows"]) == 3
    assert v["readings"]["free"]["on"]["hold_loss_ratio"]["mean"] == pytest.approx(0.9)
    assert v["readings"]["path"]["live_share_lowest"] == 1.0
    # the token carries it, the main route does not use it
    flat = arm_record(arm_records(own=1.0), "on", switch=exact_switch())
    v = T.judge(off, flat, readout(0.004), readout(0.30))
    assert v["passed"] is False and "does not use it" in v["outcome"] and "not rising" in v["outcome"]
    # the path did not put it into the token
    v = T.judge(off, on, readout(0.004), readout(0.004))
    assert v["passed"] is False and "did not put the asked change into the token" in v["outcome"]
    # another sense past the stop line over the judged window
    worse = arm_record(arm_records(own=0.9, ratios=dict(tactile=0.5 * 1.8)), "on", switch=exact_switch())
    v = T.judge(off, worse, readout(0.004), readout(0.30))
    assert v["passed"] is False and v["lines"]["C"]["met"] is False
    assert v["lines"]["C"]["senses"]["tactile"]["past_stop_line"] is True
    # no verdict: a missing record, a missing read-out, a switch that was not exact, a killed arm
    assert T.judge(None, on, None, None)["passed"] is None
    assert T.judge(off, on, None, readout(0.30))["passed"] is None
    bad = exact_switch()
    bad["switch"]["float32"]["change_max_abs"] = 1e-6
    v = T.judge(off, arm_record(arm_records(own=0.9), "on", switch=bad), readout(0.004), readout(0.30))
    assert v["passed"] is None and "switch record" in v["why_none"]
    killed = dict(on, kill_reading=dict(stop=True, detail={}), failed="stopped by the kill criterion")
    assert T.judge(off, killed, readout(0.004), readout(0.30))["passed"] is False


def test_the_trend_says_whether_the_reading_was_still_rising():
    off = arm_record(arm_records(own=1.0), "off")
    rising = [report(START + 100 * i, own=1.0 - 0.00005 * i, jitter=(0.0005 if i % 2 else -0.0005))
              for i in range(1, 31)]
    on = arm_record(rising, "on", switch=exact_switch())
    t = T.trend(on["windows"], off["windows"])
    assert t["still_rising"] is True and t["rise_second_to_third"] > 0
    flat = arm_record(arm_records(own=0.9), "on")
    assert T.trend(flat["windows"], off["windows"])["still_rising"] is False


# ------------------------------------------------------------------------------------------------ the kill reading
def test_the_kill_reading():
    off = arm_records(own=1.0)
    # nothing to judge before 5 matched reports after report 1
    assert T.kill_reading(arm_records(own=0.9)[:5], off, START) is None
    stop, detail = T.kill_reading(arm_records(own=0.9)[:6], off, START)
    assert stop is False and detail["k"] == 5
    # a non-finite step stops at once
    stop, detail = T.kill_reading([report(START + 100, nan=1)], off, START)
    assert stop is True and detail["nonfinite"] is True
    # the path's hidden layer under 0.9 live share in 2 consecutive reports
    low = [report(START + 100, live=0.8), report(START + 200, live=1.0), report(START + 300, live=0.85)]
    assert T.kill_reading(low, off, START) is None
    low.append(report(START + 400, live=0.7))
    stop, detail = T.kill_reading(low, off, START)
    assert stop is True and detail["path_live_low_at"] == START + 400
    # another sense past the watcher's stop multiple (1.716 for touch) over the newest matched reports
    worse = arm_records(own=0.9, ratios=dict(tactile=0.5 * 1.8))
    stop, detail = T.kill_reading(worse[:8], off, START)
    assert stop is True and detail["crossed"] == ["tactile"]
    under = arm_records(own=0.9, ratios=dict(tactile=0.5 * 1.6))
    assert T.kill_reading(under[:8], off, START)[0] is False
    assert T.VT.stop_multiple("tactile") == pytest.approx(1.716) and T.VT.stop_multiple("vision") == pytest.approx(3.948)


# ------------------------------------------------------------------------------------------------ the arms
def test_the_arms_command_is_productions_with_the_folders_replaced_and_the_flag_given(monkeypatch, tmp_path):
    production = ["python", "-u", "-m", "mimo_bridge.train_mimo", "--seed", "7", "--out", "C:/prod/run",
                  "--resume", "C:/prod/ckpt_step_1", "--checkpoint-storage", "C:/prod/ck", "--max-steps", "9",
                  "--latent-body-hierarchy", "shadow", "--observer", "on", "--vestibular-encoder-repair",
                  "rescale_restart", "--persistent-memory-constant-return", "2000", "--motor-asked-change", "on",
                  "--caregiver-record-dir", "C:/prod/care", "--report-every", "100", "--minutes", "480"]
    monkeypatch.setattr(T.ST, "production_command", lambda: list(production))
    for arm in ("off", "on"):
        cmd = T.arm_command(arm, "D:/x/ckpt_step_470000", tmp_path / arm, 473000)
        assert cmd[-2:] == ["--motor-asked-change", arm] and cmd.count("--motor-asked-change") == 1
        assert "C:/prod/run" not in cmd and "C:/prod/ckpt_step_1" not in cmd and "C:/prod/care" not in cmd
        assert cmd[cmd.index("--resume") + 1] == "D:/x/ckpt_step_470000"
        assert cmd[cmd.index("--max-steps") + 1] == "473000" and cmd[cmd.index("--out") + 1] == str(tmp_path / arm)
        assert cmd[cmd.index("--latent-body-hierarchy") + 1] == "off" and cmd[cmd.index("--observer") + 1] == "off"
        # production's other settings stay: the seed, the vestibular repair, the memory's constant return
        assert cmd[cmd.index("--seed") + 1] == "7"
        assert cmd[cmd.index("--vestibular-encoder-repair") + 1] == "rescale_restart"
        assert cmd[cmd.index("--persistent-memory-constant-return") + 1] == "2000"
    with pytest.raises(ValueError):
        T.arm_command("maybe", "D:/x/ckpt_step_1", tmp_path, 2)


def test_the_code_gate_names_what_is_missing(tmp_path):
    src = tmp_path / "Source" / "Python"
    (src / "mimo_bridge").mkdir(parents=True)
    (src / "lem" / "model").mkdir(parents=True)
    (src / "mimo_bridge" / "train_mimo.py").write_text("ap.add_argument('--other')\n", encoding="utf-8")
    (src / "mimo_bridge" / "prediction_audit.py").write_text("ARMS = ()\n", encoding="utf-8")
    missing = T.code_ready(tmp_path)
    assert len(missing) == 4 and any("size-matched" in m for m in missing) and any("flag" in m for m in missing)
    (src / "lem" / "model" / "motor_asked_change.py").write_text("", encoding="utf-8")
    (src / "mimo_bridge" / "train_mimo.py").write_text(
        "ap.add_argument('--motor-asked-change')\n'prediction_audit_hold_first'\n", encoding="utf-8")
    (src / "mimo_bridge" / "prediction_audit.py").write_text(
        "ARMS = ('size_matched',)\ndef summarize_hold_first(): pass\n", encoding="utf-8")
    assert T.code_ready(tmp_path) == []
    # this branch's own tree: the path and the flag are here; the swap test's reading comes with production's tip
    here = T.code_ready(Path(__file__).resolve().parents[4])
    assert not any("asked-change path" in m or "flag" in m for m in here)


def test_the_watch_gate_reads_the_registry_row(tmp_path):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps(dict(rows=[dict(id=T.WATCH_ROW, status="written", result=None)])), encoding="utf-8")
    ended, words = T.watch_ended(reg)
    assert ended is False and "has not ended" in words
    # readings written into the row during the watch are not its end (the row of Oct 3, 3:17 AM PDT)
    reg.write_text(json.dumps(dict(rows=[dict(id=T.WATCH_ROW, status="running",
                                              result=dict(values=["interim"], verdict="no verdict"))])),
                   encoding="utf-8")
    ended, words = T.watch_ended(reg)
    assert ended is False and "during the watch" in words
    reg.write_text(json.dumps(dict(rows=[dict(id=T.WATCH_ROW, status="done", result=dict(verdict="pass"))])),
                   encoding="utf-8")
    assert T.watch_ended(reg)[0] is True
    reg.write_text(json.dumps(dict(rows=[dict(id=T.WATCH_ROW, status="failed", result=dict(verdict="fail"))])),
                   encoding="utf-8")
    assert T.watch_ended(reg)[0] is True
    reg.write_text(json.dumps(dict(rows=[])), encoding="utf-8")
    assert T.watch_ended(reg)[0] is False


def test_the_switch_record_must_read_zero_in_every_precision():
    assert T.switch_exact(exact_switch())["exact"] is True
    assert T.switch_exact(None)["exact"] is None
    moved = exact_switch()
    moved["switch"]["autocast"]["bit_identical"] = False
    assert T.switch_exact(moved)["exact"] is False
    weights = exact_switch()
    weights["switch"]["output_weight_max_abs"] = 0.01
    assert T.switch_exact(weights)["exact"] is False


def test_the_declared_numbers():
    assert T.UPDATES == 3000 and T.WINDOWS == ((0, 1000), (1000, 2000), (2000, 3000)) and T.JUDGED == (2000, 3000)
    assert T.Z == 3.14 and T.R2_LINE == 0.04 and T.KILL_MIN_REPORTS == 5 and T.KILL_WINDOW == 10
    assert T.JUDGED_SENSE == "proprioception" and T.JUDGED_PHASE == "hold_first"
    assert math.isclose(T.VT.stop_multiple("audio"), 1.43 * 1.2)
