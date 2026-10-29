# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The stop lines and pass lines of the vestibular input path's switch in production (docs/gain_of_function_v4/
vestibular_input_path_watch.py), judged by the H84 hourly watcher (docs/h84_launch_2026-09-18/hourly_watch.py).

Hand-made stats records in run folders linked the way the supervisor links them: 30 full reports before the switch in
a parent run, then a run that carries the vestibular_input_path_switch record at model step 479,350, and
continuations. What is held here:
  - a run that no switch record reaches is not judged, by the module and by the hourly watcher's own program; the
    asked-change watch and the vestibular repair watch on the same folders read nothing of this switch, and their
    own instances keep their names;
  - every planted crossing stops and a clean record does not: the switch record not as declared, a non-finite update,
    a sense past its line in reports 2 and 3, a sense's 20-report median past its stop multiple (a stop) or its alert
    multiple (an alert, never a stop), the step rate under its line in two windows running, the path's output above
    its growth line or not a finite number; the path has no hidden-layer line and no warning about one;
  - the growth line counts one update of the path's weight per model step;
  - pass line T1 (head tilt read from the token by the trainer's own probe) and T2 (vestibular's prediction error
    below the error of predicting no change) read as declared; a stream without the probe reads unmeasured;
  - the hourly watcher's program writes the stop into the run, and its sweep prints the lines with no bare score
    decimal and no banned name."""
import collections
import importlib.util
import json
import math
import runpy
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, unanchored_numbers

HERE = Path(__file__).resolve()
TREE = HERE.parents[4]
SCRIPT = TREE / "docs" / "gain_of_function_v4" / "vestibular_input_path_watch.py"
MOTOR = TREE / "docs" / "gain_of_function_v4" / "motor_asked_change_watch.py"
VESTIBULAR = TREE / "docs" / "gain_of_function_v4" / "vestibular_repair_watch.py"
WATCHER = TREE / "docs" / "h84_launch_2026-09-18" / "hourly_watch.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load(SCRIPT, "vestibular_input_path_watch")
M = _load(MOTOR, "motor_asked_change_watch_fresh")
V = _load(VESTIBULAR, "vestibular_repair_watch_fresh_for_input_path")
BASE = {"vision": 1.0, "tactile": 1.0, "audio": 1.2, "proprioception": 0.9, "vestibular": 0.85}
N = 479350
ARGS = {"report_every": 100, "replay": "on", "replay_every": 2, "device": "cuda:0", "vestibular_input_path": "on"}
SWITCH = {"rows": 64, "output_weight_max_abs": 0.0,
          "float32": {"change_max_abs": 0.0, "token_rms": 1.0, "bit_identical": True},
          "autocast": {"change_max_abs": 0.0, "token_rms": 1.0, "bit_identical": True},
          "exact": True, "version": "vestibular_input_path_v2", "checkpoint": "ckpt_step_479350", "at_step": N}
HEALTHY_DISK = collections.namedtuple("usage", "total used free")(1e12, 0.0, 500e9)


def _mimo(step, nf_steps=100, ratio=None, sps=0.42, nan=0, output=None, path=True, tilt=None, improvement=-0.003,
          probe=True, capped=None, slope=None, weight=0.01):
    """A stats record as the trainer writes it, with the fields the watch reads. tilt: the probe's head-tilt reading
    (its control reads -0.01)."""
    r = {"type": "mimo", "step": step, "timestamp": 1.0e9 + step,
         "next_frame_norm": {"steps": nf_steps, "ratio": dict(BASE if ratio is None else ratio)},
         "nan_steps": nan, "model_steps_per_s": sps,
         "machine": {"gpu_used_mib": 6760.0, "gpu_total_mib": 8188.0},
         "vestibular": {"improvement_over_persistence": improvement, "repair": {
             "token_between_bodies": 0.41, "token_change": 0.04}}}
    if probe:
        t = -0.01 if tilt is None else tilt
        r["vestibular"]["repair"]["balance_probe"] = {
            "r2": {"head_tilt": t, "trunk_roll": t / 2.0}, "r2_shuffled": {"head_tilt": -0.01, "trunk_roll": -0.1},
            "within_body_turn_deg": {"head_tilt": 20.3, "trunk_roll": 4.5}, "samples": 8000, "reports": 10}
    if path:
        out = 0.0002 * max(0, (step - N)) / 100.0 if output is None else output
        r["vestibular_input_path"] = {"version": "vestibular_input_path_v2", "encoder_calls": 200,
                                      "encoder_output_sd": 18.09, "output_weight_rms": weight, "output_rms": out,
                                      "field_reach": {"head_accelerometer": 0.01, "trunk_up": 0.008}}
        if capped is not None:
            r["vestibular_input_path"]["capped_share"] = capped
            r["vestibular_input_path"]["output_cap"] = 0.3
        if slope is not None:
            r["vestibular_input_path"].update(change_slope=slope, own_change=0.13, path_change=0.25, live_calls=100)
    return r


def _run(exp, name, parent=None, ck_step=None, records=(), args=None):
    run = exp / name
    run.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps(ARGS if args is None else args), encoding="utf-8")
    if parent is not None:
        (run / "continuation_receipt.json").write_text(json.dumps({
            "parent_run": str(exp / parent),
            "checkpoint": str(exp / parent / "checkpoints" / ("ckpt_step_%d" % ck_step)),
            "timestamp": time.time(), "reason": "test"}), encoding="utf-8")
    with open(run / "replay_training_stats.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return run


def _switch(step=N, switch=None):
    return {"type": "vestibular_input_path_switch", "step": step, "timestamp": 1.0,
            "switch": SWITCH if switch is None else switch}


def _before(n=30, **kw):
    return [_mimo(N - 50 - 100 * (n - 1 - i), path=False, **{k: (v(i) if callable(v) else v) for k, v in kw.items()})
            for i in range(n)]


def _step(k):
    """The model step of full report k after the switch at 479,350 (the first record, 479,400, is partial)."""
    return 479400 + 100 * k


def _after(n, first=1, partial=True, **kw):
    out = [_mimo(479400, nf_steps=50)] if partial else []
    for k in range(first, n + 1):
        out.append(_mimo(_step(k), **{key: (v(k) if callable(v) else v) for key, v in kw.items()}))
    return out


def _lineage(tmp_path, after=(), switch=None, before=None, args=None):
    exp = tmp_path / "experiments"
    P = _run(exp, "H84_before", parent="H84_grandparent", ck_step=476300,
             records=_before() if before is None else before, args=dict(ARGS, vestibular_input_path="off"))
    F = _run(exp, "H84_switch", parent="H84_before", ck_step=N, records=[_switch(switch=switch)] + list(after),
             args=args)
    return SimpleNamespace(exp=exp, P=P, F=F)


def _ev(run, write=True):
    return W.evaluate(run, W.BASE.StatsReader(), write=write)


def _stop_line(sense):
    return BASE[sense] * W.BASE.k6_levels(sense)[1]


def _alert_line(sense):
    return BASE[sense] * W.BASE.k6_levels(sense)[0]


def _hourly(monkeypatch, run, hours="1"):
    monkeypatch.setenv("UEMA_WATCH_SAVED", str(Path(run).parent.parent / "_saved"))
    monkeypatch.setenv("UEMA_WATCH_ARCHIVE", str(Path(run).parent.parent / "_archive"))
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: HEALTHY_DISK)
    try:
        import psutil
        monkeypatch.setattr(psutil, "virtual_memory", lambda: collections.namedtuple("vm", "available")(64e9))
    except ImportError:
        pass
    g = runpy.run_path(str(WATCHER))
    g["configure"]([str(WATCHER), str(run), hours])
    return g


def test_the_caps_share_is_printed_as_a_reading_and_never_judged(tmp_path, monkeypatch):
    """The share of rows the path's cap scaled back: carried by compact(), read from the newest full report, printed
    with both ends of its scale, and no stop or alert at any value; a record without it says so."""
    assert W.compact(_mimo(_step(3), capped=0.25))["path"]["capped_share"] == 0.25
    assert W.compact(_mimo(_step(3)))["path"]["capped_share"] is None
    L = _lineage(tmp_path / "with", after=_after(30, capped=lambda k: 0.0 if k < 30 else 1.0))
    ev = _ev(L.F, write=False)
    assert ev["why"] is None and not ev["judgement"].get("alerts")
    assert ev["judgement"]["readings"]["capped_share"] == {"step": _step(30), "share": 1.0, "cap": 0.3}
    printed = _hourly(monkeypatch, L.F)["vestibular_input_path_lines"]()
    line = [p for p in printed if "its output's cap" in p]
    assert len(line) == 1 and ("at model step 482,400 the cap (30% of one token unit) scaled back 100% of the "
                               "report's rows") in line[0], line
    assert not bare_decimals(line[0]) and not unanchored_numbers(line[0])
    L2 = _lineage(tmp_path / "without", after=_after(5))
    assert _ev(L2.F, write=False)["judgement"]["readings"]["capped_share"] is None
    printed = _hourly(monkeypatch, L2.F)["vestibular_input_path_lines"]()
    assert any("no report carries the share yet" in p for p in printed)


def test_the_two_parts_slope_is_printed_as_a_reading_and_never_judged(tmp_path, monkeypatch):
    """The encoder's own part against the path's: carried by compact(), the newest full report's value and the lowest
    since the switch, printed with both ends of its scale; no stop and no alert at any value."""
    assert W.compact(_mimo(_step(3), slope=-0.4))["path"]["change_slope"] == -0.4
    assert W.compact(_mimo(_step(3)))["path"]["change_slope"] is None
    L = _lineage(tmp_path / "with", after=_after(30, slope=lambda k: -1.0 if k == 12 else -0.25))
    ev = _ev(L.F, write=False)
    assert ev["why"] is None and not ev["judgement"].get("alerts")
    assert ev["judgement"]["readings"]["two_parts"] == {"step": _step(30), "slope": -0.25, "lowest": -1.0,
                                                        "reports": 30}
    printed = _hourly(monkeypatch, L.F)["vestibular_input_path_lines"]()
    line = [p for p in printed if "the encoder's own part against the path's part" in p]
    assert len(line) == 1 and "at model step 482,400 it reads -25%, the lowest since the switch -100%" in line[0], line
    assert not bare_decimals(line[0]) and not unanchored_numbers(line[0])
    L2 = _lineage(tmp_path / "without", after=_after(5))
    assert _ev(L2.F, write=False)["judgement"]["readings"]["two_parts"] is None
    printed = _hourly(monkeypatch, L2.F)["vestibular_input_path_lines"]()
    assert any("the encoder's own part against the path's part" in p and "no report carries it yet" in p
               for p in printed)


# ------------------------------------------------------------------------------ the standing lines
def test_the_standing_windows_start_at_n_plus_2000_and_need_a_report_past_their_end():
    full = [W.compact(_mimo(_step(k), tilt=0.4, slope=-0.05, weight=0.08)) for k in range(1, 41)]
    wins = W.standing_windows(full, N + W.T1_WINDOW[0])
    # reports 1 to 19 lie before N + 2,000; windows of reports 20 to 29 and 30 to 39; report 40 opens a third
    assert [w["model_steps"] for w in wins] == [[481351, 482350], [482351, 483350]]
    assert [w["reports"] for w in wins] == [10, 10] and [w["index"] for w in wins] == [0, 1]
    assert all(w["t1"] == "met" and w["head_tilt"] == 0.4 and w["slope"] == -0.05 and w["weight"] == 0.08
               for w in wins)
    assert W.standing_alerts(wins) == []
    assert W.standing_windows([], None) == []
    absolute = W.standing_windows(full, None)                       # on the model step count's own thousands
    assert absolute[0]["model_steps"][1] % 1000 == 0 and absolute[0]["reports"] >= 5


def test_the_target_under_its_line_in_two_windows_running_is_one_alert_and_never_a_stop(tmp_path, capsys, monkeypatch):
    """T1 as a standing line: met in the first window after N + 2,000, missed from the second on. The alert comes at
    the second missed window, once, with the change slope and the weight size beside it; no stop at any reading."""
    L = _lineage(tmp_path, after=_after(60, tilt=lambda k: 0.4 if k < 30 else 0.1, slope=-0.04, weight=0.08))
    st = W.standing(L.F, W.BASE.StatsReader())
    assert st["N"] == N and [w["t1"] for w in st["windows"]] == ["met", "missed", "missed", "missed"]
    assert [a["id"] for a in st["alerts"]] == ["T1-standing@484350"]
    capsys.readouterr()
    assert W.breach(L.F, W.BASE.StatsReader()) is None
    out = capsys.readouterr().out
    assert out.count("THE TARGET IS MISSED WHILE THE PATH IS ON") == 1
    assert "model steps 482,351 to 483,350 and 483,351 to 484,350" in out
    assert "the change slope -4.0% (its line -10%)" in out and "the path's weight size 40% of its line" in out
    for line in out.splitlines():
        assert not bare_decimals(line) and not unanchored_numbers(line), line
    assert [a["id"] for a in W.load_state(L.F)["alerts"]] == ["T1-standing@484350"]
    assert W.breach(L.F, W.BASE.StatsReader()) is None
    assert "THE TARGET IS MISSED" not in capsys.readouterr().out             # printed once
    printed = _hourly(monkeypatch, L.F)["vestibular_input_path_lines"]()
    line = [p for p in printed if "the standing lines (judged" in p]
    assert len(line) == 1 and "model steps 484,351 to 485,350" in line[0] and "standing alerts so far: 1" in line[0]
    assert not bare_decimals(line[0]) and not unanchored_numbers(line[0])
    m6 = [p for p in printed if "WHAT THIS LINE IS" in p]
    assert len(m6) == 1 and "the first 232 model steps after the switch" in m6[0] and "30% of one token unit" in m6[0]


def test_the_change_slope_and_the_weight_size_each_have_a_standing_line(tmp_path, capsys):
    L = _lineage(tmp_path / "slope", after=_after(60, tilt=0.4, slope=lambda k: -0.04 if k < 40 else -0.3, weight=0.08))
    st = W.standing(L.F, W.BASE.StatsReader())
    assert [a["id"] for a in st["alerts"]] == ["S1@485350"]
    assert "the change slope is under its line of -10%" in st["alerts"][0]["why"]
    L2 = _lineage(tmp_path / "weight", after=_after(60, tilt=0.4, slope=-0.04, weight=lambda k: 0.08 if k < 30 else 0.5))
    st2 = W.standing(L2.F, W.BASE.StatsReader())
    assert [a["id"] for a in st2["alerts"]] == ["W1@484350"]
    assert "the path's weight size 250% of its line" in st2["alerts"][0]["why"]
    # one window across a line and back is no alert; a later run of two windows is the alert
    L3 = _lineage(tmp_path / "twice", after=_after(
        90, tilt=lambda k: 0.1 if (30 <= k < 40 or 50 <= k < 70) else 0.4, slope=-0.04, weight=0.08))
    ids = [a["id"] for a in W.standing(L3.F, W.BASE.StatsReader())["alerts"]]
    assert ids == ["T1-standing@486350"]
    for L_ in (L, L2, L3):
        assert W.breach(L_.F, W.BASE.StatsReader()) is None            # never a stop
    capsys.readouterr()


def test_the_standing_lines_go_on_past_the_watchs_end_and_past_the_reach_of_its_walk(tmp_path, capsys, monkeypatch):
    """Past N + 10,000 the stop lines are closed and the standing lines are still judged; and a run with the flag on
    that no switch record reaches (the switch lies further back than the watch walks) is judged on the model step
    count's own thousands, its alerts kept in its own folder."""
    L = _lineage(tmp_path / "late", after=_after(130, tilt=lambda k: 0.4 if k < 110 else 0.1, slope=-0.04, weight=0.08))
    assert W.breach(L.F, W.BASE.StatsReader()) is None
    assert W.load_state(L.F).get("closed")                              # the watch is over
    ids = [a["id"] for a in W.load_state(L.F)["alerts"]]
    assert ids == ["T1-standing@492350"], ids                           # raised 3,000 model steps past the watch's end
    exp = tmp_path / "far" / "experiments"
    far = _run(exp, "H84_far", records=[_mimo(600000 + 100 * k, tilt=0.1, slope=-0.04, weight=0.08)
                                         for k in range(1, 32)])
    assert W.BASE.find_switch(far, W.BASE.StatsReader()) is None
    st = W.standing(far, W.BASE.StatsReader())
    assert st["N"] is None and [w["model_steps"] for w in st["windows"]][:2] == [[600001, 601000], [601001, 602000]]
    assert [a["id"] for a in st["alerts"]] == ["T1-standing@602000"]
    capsys.readouterr()
    assert W.breach(far, W.BASE.StatsReader()) is None
    assert "THE TARGET IS MISSED WHILE THE PATH IS ON" in capsys.readouterr().out
    assert [a["id"] for a in W.load_state(far)["alerts"]] == ["T1-standing@602000"]
    printed = _hourly(monkeypatch, far)["vestibular_input_path_lines"]()
    assert len(printed) == 1 and "the standing lines (judged" in printed[0]
    off = _run(exp, "H84_off", records=[_mimo(600000 + 100 * k, tilt=0.1, path=False) for k in range(1, 32)],
               args=dict(ARGS, vestibular_input_path="off"))
    assert W.standing(off, W.BASE.StatsReader()) is None and W.breach(off, W.BASE.StatsReader()) is None
    both = _run(exp, "H84_other_value", records=[_mimo(600000 + 100 * k, tilt=0.4) for k in range(1, 32)],
                args=dict(ARGS, vestibular_input_path="on_delayed_encoder_only"))
    assert W.standing(both, W.BASE.StatsReader())["alerts"] == []


# ------------------------------------------------------------------------------ the module and its constants
def test_the_rule_runs_private_instances_and_the_other_watches_keep_their_names():
    assert Path(W.__file__).resolve() == SCRIPT.resolve()
    assert W.BASE.SWITCH_TYPE == W.VEST.SWITCH_TYPE == W.SWITCH_TYPE == "vestibular_input_path_switch"
    assert W.BASE.STATE_NAME == W.VEST.STATE_NAME == "vestibular_input_path_watch.json"
    # fresh loads of the two files this module runs: their own names, and the multiples this watch judges by
    assert M.SWITCH_TYPE == "motor_asked_change_switch" and M.STATE_NAME == "motor_asked_change_watch.json"
    assert V.SWITCH_TYPE == "vestibular_encoder_repair" and V.STATE_NAME == "vestibular_repair_watch.json"
    for key in ("W1", "RISE", "alert_margin", "stop_margin", "before_reports", "level_reports", "window_reports",
                "horizon_steps", "K5_share", "K5_consecutive"):
        assert W.LINES[key] == V.LINES[key], key
    assert W.BASE.k6_levels("audio") == V.k6_levels("audio") and W.BASE.k6_levels("vision") == V.k6_levels("vision")
    assert M.LINES["growth_per_optimizer_update"] == pytest.approx(3e-4 * 239.0 / 26.0)       # the motor watch's own
    assert M.check_switch is not W.check_switch and M.compact is not W.compact


def test_the_growth_bound_and_the_path_constants_are_the_builds():
    assert W.PATH_GAIN == pytest.approx(1.0 / math.sqrt(28.0))
    assert W.GROWTH_PER_MODEL_STEP == pytest.approx(W.LEARNING_RATE * W.SUMMED_INPUT * W.PATH_GAIN)
    assert W.LINES["growth_per_optimizer_update"] == W.GROWTH_PER_MODEL_STEP
    assert W.BASE.growth_line(100, 1.0) == pytest.approx(100 * W.GROWTH_PER_MODEL_STEP)
    src = (TREE / "Source" / "Python" / "lem" / "model" / "vestibular_input_path.py").read_text(encoding="utf-8")
    assert 'INPUT_PATH_VERSION = "%s"' % W.PATH_VERSION in src
    from lem.model import vestibular_input_path as VIP
    assert VIP.output_gain(28) == pytest.approx(W.PATH_GAIN)
    assert VIP.OUTPUT_CAP == W.OUTPUT_CAP and W.M6_REACH_STEPS == 232
    assert W.GROWTH_PER_MODEL_STEP * W.M6_REACH_STEPS <= W.OUTPUT_CAP < W.GROWTH_PER_MODEL_STEP * (W.M6_REACH_STEPS + 1)
    assert W.T1_R2 == V.LINES["A4_r2"] and W.T1_MARGIN == V.LINES["A4_margin"]
    assert (W.T2_DRAWS, W.T2_BLOCK) == (V.LINES["A5_draws"], V.LINES["A5_block"])


# ------------------------------------------------------------------------------ which runs are judged
def test_a_run_no_switch_reaches_is_quiet(tmp_path, monkeypatch, capsys):
    L = _lineage(tmp_path, after=_after(3))
    assert W.breach(L.P, W.BASE.StatsReader()) is None and W.lines(L.P, W.BASE.StatsReader()) == []
    assert W.main([str(L.P)]) == 0
    assert "no vestibular_input_path_switch record reaches" in capsys.readouterr().out
    hw = _hourly(monkeypatch, L.P)
    assert hw["vestibular_input_path_breach"]() is None and hw["vestibular_input_path_lines"]() == []
    assert "VESTIBULAR INPUT PATH" not in capsys.readouterr().out
    assert not (L.P / W.STATE_NAME).exists()
    # the other two watches read nothing of this switch on the switched run
    assert M.breach(L.F, M.StatsReader()) is None and M.lines(L.F, M.StatsReader()) == []
    assert V.breach(L.F, V.StatsReader()) is None and V.lines(L.F, V.StatsReader()) == []


def test_the_hourly_watchers_three_rules_are_separate_instances(tmp_path, monkeypatch):
    L = _lineage(tmp_path, after=_after(3))
    hw = _hourly(monkeypatch, L.F)
    assert hw["vestibular_input_path_lines"]()
    mine, motor, vest = hw["_vestibular_input_path_rule"](), hw["_motor_asked_change_rule"](), hw["_vestibular_rule"]()
    assert mine.BASE is not motor and mine.VEST is not vest and motor.BASE is not vest
    assert motor.SWITCH_TYPE == "motor_asked_change_switch" and vest.SWITCH_TYPE == "vestibular_encoder_repair"
    assert hw["motor_asked_change_breach"]() is None and hw["motor_asked_change_lines"]() == []
    assert hw["vestibular_repair_breach"]() is None and hw["vestibular_repair_lines"]() == []
    assert not (L.F / motor.STATE_NAME).exists() and not (L.F / vest.STATE_NAME).exists()


# ------------------------------------------------------------------------------ the stop lines
def test_a_clean_switch_holds_every_line_through_the_watch(tmp_path, capsys):
    L = _lineage(tmp_path, after=_after(101))
    ev = _ev(L.F)
    assert ev["why"] is None and ev["watch_over"] and not ev["warnings"], ev["warnings"]
    lines = ev["judgement"]["lines"]
    assert "M3" not in lines                                          # the path has no hidden layer
    assert all(lines[k]["state"] == "held" for k in ("M1", "M2", "M4_early", "M4", "M5", "M6")), lines
    assert W.breach(L.F, W.BASE.StatsReader()) is None
    assert "INCOMPLETE" not in capsys.readouterr().out
    assert W.load_state(L.F).get("closed")                            # judged to its end: no later check judges it


@pytest.mark.parametrize("change,words", [
    ({"exact": False}, "does not say"),
    ({"float32": {"change_max_abs": 1e-7, "token_rms": 1.0, "bit_identical": False}}, "differs from the token"),
    ({"output_weight_max_abs": 0.01}, "was not zero"),
    ({"rows": 8}, "8 probe rows"),
    ({"version": "vestibular_input_path_v0"}, "vestibular_input_path_v0")])
def test_the_switch_record_is_judged_at_once(tmp_path, change, words):
    L = _lineage(tmp_path, switch=dict(SWITCH, **change))
    ev = _ev(L.F)
    assert ev["id"] == "M1:switch" and words in ev["why"] and ev["why"].startswith("vestibular input path:"), ev["why"]
    assert "asked-change" not in ev["why"] and "motor" not in ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]


def test_on_the_graphics_card_the_check_must_have_run_under_autocast(tmp_path):
    sw = {k: v for k, v in SWITCH.items() if k != "autocast"}
    assert "autocast" in _ev(_lineage(tmp_path / "card", switch=sw).F)["why"]
    assert _ev(_lineage(tmp_path / "cpu", switch=sw, args=dict(ARGS, device="cpu")).F)["why"] is None


def test_a_non_finite_update_stops(tmp_path):
    L = _lineage(tmp_path, after=_after(4, nan=lambda k: 1 if k == 3 else 0))
    ev = _ev(L.F)
    assert ev["id"] == "M2@%d" % _step(3) and "non-finite" in ev["why"]
    (L.F / "trainer.log").write_text("x\n12:00 WARNING optimizer step skipped: non-finite gradient\n", encoding="utf-8")
    L2 = _lineage(tmp_path / "log", after=_after(4))
    (L2.F / "trainer.log").write_text("x\n12:00 WARNING optimizer step skipped: non-finite gradient\n",
                                      encoding="utf-8")
    ev2 = _ev(L2.F)
    assert ev2["id"].startswith("M2:log:") and "skipped" in ev2["why"]


def test_a_sense_past_its_line_in_reports_2_and_3_stops_and_one_report_does_not(tmp_path):
    high = dict(BASE, tactile=BASE["tactile"] * W.LINES["W1"]["tactile"] * 1.1)
    one = _lineage(tmp_path / "one", after=_after(3, ratio=lambda k: high if k == 2 else BASE))
    assert _ev(one.F)["why"] is None
    two = _lineage(tmp_path / "two", after=_after(3, ratio=lambda k: high if k in (2, 3) else BASE))
    ev = _ev(two.F)
    assert ev["id"] == "M4:reports2-3" and "tactile" in ev["why"], ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]


def test_the_other_senses_are_judged_in_every_20_report_window_at_two_levels(tmp_path, capsys):
    stop = _lineage(tmp_path / "stop", after=_after(21, ratio=lambda k: dict(BASE, audio=1.1 * _stop_line("audio"))
                                                    if k >= 2 else BASE))
    ev = _ev(stop.F)
    assert ev["id"] == "M4:w0" and "audio" in ev["why"] and ev["why"].startswith("vestibular input path:"), ev["why"]
    mild = dict(BASE, audio=0.5 * (_alert_line("audio") + _stop_line("audio")))
    alert = _lineage(tmp_path / "alert", after=_after(21, ratio=lambda k: mild if k >= 2 else BASE))
    ev = _ev(alert.F)
    assert ev["why"] is None and [a["id"] for a in ev["judgement"]["alerts"]] == ["M4-alert:w0"]
    assert ev["judgement"]["alerts"][0]["why"].startswith("vestibular input path ALERT")
    assert W.breach(alert.F, W.BASE.StatsReader()) is None
    assert capsys.readouterr().out.count("vestibular input path ALERT") == 1
    assert W.breach(alert.F, W.BASE.StatsReader()) is None            # printed once
    assert "vestibular input path ALERT" not in capsys.readouterr().out
    # vestibular's own ratio: an alert only, whatever its size
    own = _lineage(tmp_path / "own", after=_after(21, ratio=lambda k: dict(BASE, vestibular=30.0)))
    ev = _ev(own.F)
    assert ev["why"] is None and any("vestibular" in a.get("senses", []) for a in ev["judgement"]["alerts"])


def test_the_step_rate_under_its_line_in_two_windows_running_stops(tmp_path):
    slow = lambda k: 0.42 if k < 2 else (0.42 * 0.85 if k < 22 else 0.42 * 0.85 * 0.85)                  # noqa: E731
    L = _lineage(tmp_path, after=_after(41, sps=slow))
    ev = _ev(L.F)
    assert ev["id"] == "M5:w1" and "model steps per second" in ev["why"], ev["why"]
    once = _lineage(tmp_path / "once", after=_after(41, sps=lambda k: 0.42 if k < 2 else 0.42 * 0.85))
    assert _ev(once.F)["why"] is None


def test_the_paths_output_above_its_growth_line_stops_and_counts_one_update_per_model_step(tmp_path, capsys):
    k = 5
    line = W.GROWTH_PER_MODEL_STEP * (_step(k) - N)
    under = _lineage(tmp_path / "under", after=_after(k, output=lambda j: 0.9 * line if j == k else None))
    assert _ev(under.F)["why"] is None
    over = _lineage(tmp_path / "over", after=_after(k, output=lambda j: 1.1 * line if j == k else None))
    ev = _ev(over.F)
    assert ev["id"] == "M6@%d" % _step(k) and "growth line" in ev["why"], ev["why"]
    assert "one update of the path's weight each" in ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]
    # with --replay on the asked-change watch counts 1.5 updates per model step; this one counts 1
    assert ev["judgement"]["optimizer_updates_per_model_step"] == 1.0
    nan = _lineage(tmp_path / "nan", after=_after(3, output=lambda j: float("nan") if j == 2 else None))
    assert "not a finite number" in _ev(nan.F)["why"]
    big = _lineage(tmp_path / "big", after=_after(3, output=lambda j: None), switch=None)
    assert _ev(big.F)["why"] is None


def test_without_the_reports_before_the_switch_the_unconditional_lines_still_stop(tmp_path, capsys):
    L = _lineage(tmp_path, before=_before(5), after=_after(4, nan=lambda k: 1 if k == 3 else 0))
    ev = _ev(L.F)
    assert ev["id"].startswith("M2@")
    assert any("only 5 of the 20 full reports" in w for w in ev["warnings"])
    assert not any("hidden layer" in w for w in ev["warnings"])       # this path has none
    assert W.breach(L.F, W.BASE.StatsReader())
    assert "VESTIBULAR INPUT PATH STOP LINES INCOMPLETE" in capsys.readouterr().out
    bare = _lineage(tmp_path / "bare", after=_after(3, path=False))
    assert any("vestibular_input_path block" in w for w in _ev(bare.F)["warnings"])


def test_a_stop_is_not_repeated_on_its_own_crossing_and_reading_by_hand_writes_nothing(tmp_path, capsys):
    L = _lineage(tmp_path, after=_after(4, nan=lambda k: 1 if k == 3 else 0))
    assert W.main([str(L.F)]) == 1 and not (L.F / W.STATE_NAME).exists()
    why = W.breach(L.F, W.BASE.StatsReader())
    assert why and W.load_state(L.F)["crossings"][0]["id"] == "M2@%d" % _step(3)
    assert W.breach(L.F, W.BASE.StatsReader()) == why                 # its stop never reached the run: judged again
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    assert W.breach(L.F, W.BASE.StatsReader()) is None                # taken


# ------------------------------------------------------------------------------ the pass lines
def test_line_t1_reads_as_declared(tmp_path):
    lo, hi = W.T1_WINDOW
    in_window = lambda k: N + lo < _step(k) <= N + hi                                                     # noqa: E731
    met = _lineage(tmp_path / "met", after=_after(40, tilt=lambda k: 0.4 if in_window(k) else 0.1))
    A = _ev(met.F)["judgement"]["acceptance"]
    assert A["T1"]["state"] == "met" and A["T1"]["model_steps"] == [N + lo + 1, N + hi]
    assert A["T1"]["reading"]["head_tilt"]["r2"] == pytest.approx(0.4)
    assert A["T1"]["reading"]["head_tilt"]["margin"] == pytest.approx(0.41)
    assert A["before"]["head_tilt"]["r2"] == pytest.approx(-0.01)
    assert A["T1_late"]["state"] == "waiting"
    low = _lineage(tmp_path / "low", after=_after(40, tilt=0.2))
    assert _ev(low.F)["judgement"]["acceptance"]["T1"]["state"] == "missed"
    # the reading at the line with a control too close to it is missed: 0.25 above the control is asked
    none = _lineage(tmp_path / "none", after=_after(40, probe=False))
    t1 = _ev(none.F)["judgement"]["acceptance"]["T1"]
    assert t1["state"] == "unmeasured" and "--vestibular-encoder-repair" in t1["why"]
    early = _lineage(tmp_path / "early", after=_after(10, tilt=0.4))
    assert _ev(early.F)["judgement"]["acceptance"]["T1"]["state"] == "waiting"
    late = _lineage(tmp_path / "late", after=_after(101, tilt=lambda k: 0.5 if k > 60 else 0.1))
    A = _ev(late.F)["judgement"]["acceptance"]
    assert A["T1"]["state"] == "missed" and A["T1_late"]["state"] == "met"
    assert sum(1 for w in A["windows"] if w["complete"]) == 10


def test_line_t2_reads_as_declared(tmp_path):
    better = _lineage(tmp_path / "better", after=_after(101, improvement=lambda k: 0.02 + 0.001 * (k % 3)))
    A = _ev(better.F)["judgement"]["acceptance"]
    assert A["T2"]["state"] == "met" and A["T2"]["p5_of_median"] > 0 and A["T2"]["reports"] == 20
    worse = _lineage(tmp_path / "worse", after=_after(101, improvement=lambda k: -0.003 + 0.001 * (k % 3)))
    assert _ev(worse.F)["judgement"]["acceptance"]["T2"]["state"] == "missed"
    assert _ev(_lineage(tmp_path / "short", after=_after(30)).F)["judgement"]["acceptance"]["T2"]["state"] == "waiting"
    vals = [0.01, 0.02, -0.01, 0.03, 0.02, 0.01, 0.00, 0.02, 0.01, 0.03]
    assert W.improvement_p5(vals) == pytest.approx(W.improvement_p5(vals))            # seeded
    assert min(vals) <= W.improvement_p5(vals) <= max(vals)


def test_the_five_minute_check_does_not_compute_the_pass_lines(tmp_path):
    L = _lineage(tmp_path, after=_after(30, tilt=0.4))
    assert "acceptance" not in W.evaluate(L.F, W.BASE.StatsReader(), write=False, readings=False)["judgement"] or \
        not W.evaluate(L.F, W.BASE.StatsReader(), write=False, readings=False)["judgement"]["acceptance"]


# ------------------------------------------------------------------------------ the hourly watcher
def test_the_hourly_watchers_own_program_writes_the_stop_into_the_run(tmp_path, monkeypatch):
    L = _lineage(tmp_path, after=_after(21, ratio=lambda k: dict(BASE, audio=1.1 * _stop_line("audio"))
                                        if k >= 2 else BASE))
    hw = _hourly(monkeypatch, L.F)
    why = hw["quick_breach"]()
    assert why and why.startswith("vestibular input path:") and "audio" in why, why
    hw["stop_run"](why)
    assert json.loads((L.F / "control.json").read_text())["cmd"] == "snapshot_and_stop"
    assert json.loads((L.F / "stop_reason.json").read_text())["reason"] == why
    assert W.load_state(L.F)["crossings"][0]["id"] == "M4:w0"
    assert not (L.F / M.STATE_NAME).exists() and not (L.F / V.STATE_NAME).exists()


def test_the_hourly_sweep_prints_the_lines_with_no_bare_decimal(tmp_path, monkeypatch):
    lo, hi = W.T1_WINDOW
    L = _lineage(tmp_path, after=_after(101, tilt=lambda k: 0.4 if _step(k) > N + lo else 0.1,
                                        improvement=lambda k: 0.02 + 0.001 * (k % 3)))
    printed = _hourly(monkeypatch, L.F)["vestibular_input_path_lines"]()
    assert printed[0].startswith("  vestibular input path: switched on at model step 479,350")
    assert any("pass line T1" in p and "met over model steps" in p for p in printed), printed
    assert any("pass line T2" in p and "met" in p for p in printed)
    assert any("changed the path's flag and nothing else" in p for p in printed)
    assert not any("hidden layer" in p for p in printed)
    for line in printed:
        assert not bare_decimals(line) and not unanchored_numbers(line), line
    for banned in ("hearing", "body position", "moment", "balance", "asked-change"):
        assert not any(banned in p for p in printed), banned
