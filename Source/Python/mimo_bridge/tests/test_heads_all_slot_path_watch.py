# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The stop lines and pass lines of the all-slot paths' switch in production (docs/gain_of_function_v4/
heads_all_slot_path_watch.py), judged by the H84 hourly watcher (docs/h84_launch_2026-09-18/hourly_watch.py).

Hand-made stats records in run folders linked the way the supervisor links them (continuation_receipt.json): 30 full
reports before the switch in a parent run (after model step 474,875, the reference's first model step), then a run
that carries the heads_all_slot_path_switch record at model step 479,350 (a checkpoint off the hundreds, as a
requested stop writes it). What is held here:
  - this watch runs on private instances: a fresh load of the asked-change watch and of the vestibular watch keep
    their own switch types, state files and lines; the multiples are the vestibular watch's;
  - a run that no switch record reaches is not judged and prints nothing, by the module and by the hourly watcher's
    own program;
  - every planted crossing stops and every clean record does not: the switch record not as declared, a non-finite
    update, one path's hidden layer under 0.9 live share in 2 reports running (by the monitor's reading or by the
    path's own), a sense past its line in reports 2 and 3, a sense's 20-report median past its stop multiple, the step
    rate under 0.9 in two windows running, a path's output above its sense's real change per number or not a finite
    number (and the alert, never a stop, on a path's fixed vector pointing against the old head's);
  - the stop's words name this switch and carry no bare score decimal; a stop is not repeated on its own crossing;
    reading by hand writes nothing;
  - the two stop lines on the audit: proprioception's own-command reading over a window of 1,000 model steps at or
    under its floor stops (and is not judged when the reports before the switch do not read above the floor); a
    head's audited error above its value before the switch by more than its spread stops in the second window
    running and not in one window alone; the spreads and the output line are the numbers of their measurement files;
  - the gradient alert (never a stop), the early lines (the lessons' gradient reaches the transformer, every head
    reads more than one slot), the late pass lines (tactile's error below no change by more than its noise,
    vestibular's own-command reading above its noise, pooled), and the watched readings with their goal references;
  - the hourly watcher's own program writes the stop into the run and its sweep prints the lines."""
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
SCRIPT = TREE / "docs" / "gain_of_function_v4" / "heads_all_slot_path_watch.py"
MOTOR = TREE / "docs" / "gain_of_function_v4" / "motor_asked_change_watch.py"
VESTIBULAR = TREE / "docs" / "gain_of_function_v4" / "vestibular_repair_watch.py"
WATCHER = TREE / "docs" / "h84_launch_2026-09-18" / "hourly_watch.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load(SCRIPT, "heads_all_slot_path_watch")
M = _load(MOTOR, "motor_asked_change_watch_fresh")
V = _load(VESTIBULAR, "vestibular_repair_watch_fresh")
HEADS = ("vision", "tactile", "audio", "proprioception", "vestibular")
BASE = {"vision": 0.7, "tactile": 1.02, "audio": 0.9, "proprioception": 0.8, "vestibular": 1.0}
N = 479350
ARGS = {"report_every": 100, "replay": "on", "replay_every": 2, "device": "cuda:0", "heads_all_slot_path": "on",
        "motor_asked_change": "on"}
SWITCH = {"rows": 64, "output_weight_max_abs": 0.0, "output_bias_max_abs": 0.0,
          "float32": {"change_max_abs": 0.0, "bit_identical": True, "same_dtype": True},
          "autocast": {"change_max_abs": 0.0, "bit_identical": True, "same_dtype": True},
          "exact": True, "version": "heads_all_slot_path_v2", "checkpoint": "ckpt_step_479350", "at_step": N}
HEALTHY_DISK = collections.namedtuple("usage", "total used free")(1e12, 0.0, 500e9)
OWN = {"vision": 0.10, "tactile": 0.0, "audio": 0.0, "proprioception": 0.30, "vestibular": 0.0}   # before the switch
MOVE = {"vision": 0.7, "tactile": 0.0007, "audio": 0.17, "proprioception": 1.1, "vestibular": 0.002}
HF_RATIO = {"vision": 0.72, "tactile": 1.05, "audio": 0.9, "proprioception": 0.65, "vestibular": 1.0}


def _hold_first(own=None, move=None, ratio=None, bodies=8):
    """A prediction_audit_hold_first block: per body 5 transitions per sense; the size-matched command's error is
    0.01 and the own command's 0.01 x (1 - own gain), so the pooled own-command reading is the gain."""
    own, move, ratio = dict(OWN, **(own or {})), dict(MOVE, **(move or {})), dict(HF_RATIO, **(ratio or {}))
    senses = {}
    for m in HEADS:
        jit = [1.0 + 0.01 * ((b * 7) % 5) for b in range(bodies)]
        senses[m] = {"per_body": {"size_matched_transitions": [5] * bodies,
                                  "size_matched_own_prediction_mse": [0.01 * (1.0 - own[m]) * j for j in jit],
                                  "size_matched_action_mse": [0.01 * j for j in jit],
                                  # the audited error: per body 5 transitions, the no-change error 0.02 and the
                                  # prediction error 0.02 x the ratio, so the pooled error ratio is the ratio
                                  "transitions": [5] * bodies,
                                  "prediction_mse": [0.02 * ratio[m] * j for j in jit],
                                  "persistence_mse": [0.02 * j for j in jit]},
                     "without_body_2": {"sensitivity_size_matched": move[m], "ratio": ratio[m],
                                        "improvement_over_size_matched": own[m]}}
    return {"excluded_bodies": [2], "senses": senses}


REAL_CHANGE = {"vision": 0.09, "tactile": 0.16, "audio": 0.2, "proprioception": 0.08, "vestibular": 0.065}


def _mimo(step, nf_steps=100, ratio=None, sps=0.45, nan=0, live=None, output=None, path=True, monitor=True,
          own=None, move=None, hf_ratio=None, head_read=None, gradient=None, fixed=None, ema=True):
    """A stats record as the trainer writes it, with the fields the watch reads. path=False: a record of a run
    without the paths (before the switch). live / output / head_read: {head: value} overrides."""
    r = {"type": "mimo", "step": step, "timestamp": 1.0e9 + step, "elapsed_s": 600.0,
         "next_frame_norm": dict({"steps": nf_steps, "ratio": dict(BASE if ratio is None else ratio)},
                                 **({"persistence_ema": {m: v * v for m, v in REAL_CHANGE.items()}} if ema else {})),
         "nan_steps": nan, "model_steps_per_s": sps,
         "tenure_pe": {"pose": 0.0002}, "tenure_gap": {"pose": 0.0},
         "machine": {"gpu_used_mib": 6760.0, "gpu_total_mib": 8188.0},
         "efference": {"gaze_copy": {"vision_err_without_copy": 0.86}},
         "liveness": {"mode": "on", "layers": {"forward_model.blocks.0.ffn.0": 1.0},
                      "pools": {"prediction_heads.heads.%s.attn" % m: 0.0 for m in HEADS}},
         "prediction_audit_hold_first": _hold_first(own, move, hf_ratio)}
    if path:
        base_out = 0.0002 * max(0, (step - N)) / 100.0
        lv, out, hr = dict(live or {}), dict(output or {}), dict(head_read or {})
        r["heads_all_slot_path"] = {
            "version": "heads_all_slot_path_v2", "rows": 64, "centre_weight": 0.5,
            "fixed": {m: dict({"old_rms": 0.03, "path_rms": 0.002, "sum_rms": 0.03, "cosine": 0.1, "rows": 32},
                              **dict(fixed or {}).get(m, {})) for m in HEADS},
            "heads": {m: {"output_weight_rms": 0.001, "output_rms": out.get(m, base_out), "live": lv.get(m, 1.0),
                          "varying": 1.0, "slot_reach": {"spread": 0.95, "largest_share": 0.2}} for m in HEADS},
            "head_read": {m: {"spread": hr.get(m, 0.8), "largest_share": 0.3, "rows": 32} for m in HEADS},
            "lesson_gradient": {m: {"rms": dict(gradient or {}).get(m, 1e-4), "rows": 32} for m in HEADS}}
        if monitor:
            for m in HEADS:
                r["liveness"]["layers"][W.path_layer(m)] = lv.get(m, 1.0)
    return r


def _run(exp, name, parent=None, ck_step=None, records=(), args=None):
    run = exp / name
    run.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps(ARGS if args is None else args), encoding="utf-8")
    if parent is not None:
        pp = exp / parent
        (run / "continuation_receipt.json").write_text(json.dumps({
            "parent_run": str(pp), "checkpoint": str(pp / "checkpoints" / ("ckpt_step_%d" % ck_step)),
            "timestamp": time.time(), "reason": "test"}), encoding="utf-8")
    with open(run / "replay_training_stats.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return run


def _switch(step=N, switch=None):
    return {"type": "heads_all_slot_path_switch", "step": step, "timestamp": 1.0,
            "switch": SWITCH if switch is None else switch}


def _before(n=30, **kw):
    """n full reports before the switch (model steps ending at 479,300), all above the reference's first step."""
    return [_mimo(479300 - 100 * (n - 1 - i), path=False, **{k: (v(i) if callable(v) else v) for k, v in kw.items()})
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
             records=_before() if before is None else before, args=dict(ARGS, heads_all_slot_path="off"))
    F = _run(exp, "H84_switch", parent="H84_before", ck_step=N, records=[_switch(switch=switch)] + list(after),
             args=args)
    return SimpleNamespace(exp=exp, P=P, F=F)


def _ev(run, write=True):
    return W.evaluate(run, W.StatsReader(), write=write)


def _stop_line(sense):
    return BASE[sense] * W.k6_levels(sense)[1]


def _clean(text):
    assert "asked-change" not in text.lower() and "motor token" not in text and "motor_asked_change" not in text, text
    assert not bare_decimals(text) and not unanchored_numbers(text), text


# ------------------------------------------------------------------------------ what the module is built from
def test_this_watch_runs_on_private_instances_and_the_other_watches_keep_their_names():
    assert Path(W.__file__).resolve() == SCRIPT.resolve()
    assert W.MW is not M and W.BASE is not V and W.MW.BASE is W.BASE
    assert W.MW.SWITCH_TYPE == W.BASE.SWITCH_TYPE == W.SWITCH_TYPE == "heads_all_slot_path_switch"
    assert W.MW.STATE_NAME == W.BASE.STATE_NAME == "heads_all_slot_path_watch.json"
    # a fresh load of each base file is as its own file says
    assert M.SWITCH_TYPE == "motor_asked_change_switch" and M.STATE_NAME == "motor_asked_change_watch.json"
    assert M.PATH_VERSION == "motor_asked_change_v1" and M.compact is not W.compact and M.acceptance is not W.acceptance
    assert M.LINES["growth_per_optimizer_update"] == pytest.approx(3e-4 * 239.0 / 26.0) and M.LINES["output_alert"] == 1.0
    assert V.SWITCH_TYPE == "vestibular_encoder_repair" and V.STATE_NAME == "vestibular_repair_watch.json"
    for key in ("W1", "W20", "RISE", "alert_margin", "stop_margin", "before_reports", "level_reports",
                "window_reports", "horizon_steps", "K4_liveness", "K4_consecutive", "K5_share", "K5_consecutive"):
        assert W.LINES[key] == V.LINES[key], key
    assert W.k6_levels("audio") == V.k6_levels("audio") and W.k6_levels("vision") == V.k6_levels("vision")
    assert W.LINES["growth_per_optimizer_update"] == pytest.approx(3e-4 * W.SUMMED_HIDDEN / math.sqrt(192.0))
    assert W.LINES["output_alert"] == W.OUTPUT_ALERT == 2.0
    src = (TREE / "Source" / "Python" / "lem" / "model" / "heads_all_slot_path.py").read_text(encoding="utf-8")
    assert 'ALL_SLOT_PATH_VERSION = "%s"' % W.PATH_VERSION in src
    assert 'return "%s.paths.%s.net.0" % (SIDE_NAME, head)' in src and W.path_layer("tactile") == \
        "heads_all_slot_path.paths.tactile.net.0"


def test_a_run_no_switch_reaches_is_quiet(tmp_path, monkeypatch, capsys):
    L = _lineage(tmp_path, after=_after(3))
    assert W.breach(L.P, W.StatsReader()) is None and W.lines(L.P, W.StatsReader()) == []
    assert W.main([str(L.P)]) == 0
    assert "no heads_all_slot_path_switch record reaches" in capsys.readouterr().out
    hw = _hourly(monkeypatch, L.P)
    assert hw["heads_all_slot_path_breach"]() is None and hw["heads_all_slot_path_lines"]() == []
    assert "ALL-SLOT" not in capsys.readouterr().out
    assert not (L.P / W.STATE_NAME).exists()
    # the asked-change watch and the vestibular watch read nothing of this switch on the switched run
    assert M.breach(L.F, M.StatsReader()) is None and M.lines(L.F, M.StatsReader()) == []
    assert V.breach(L.F, V.StatsReader()) is None and V.lines(L.F, V.StatsReader()) == []


# ------------------------------------------------------------------------------ the stop lines
def test_a_clean_switch_holds_every_line_through_the_watch(tmp_path, capsys):
    L = _lineage(tmp_path, after=_after(101))
    ev = _ev(L.F)
    assert ev["why"] is None and ev["watch_over"] and not ev["warnings"], (ev["why"], ev["warnings"])
    lines = ev["judgement"]["lines"]
    assert {k: v["state"] for k, v in lines.items()} == {k: "held" for k in ("M1", "M2", "M3", "M4_early", "M4",
                                                                             "M5", "M6", "M7", "M8", "M9")}, lines
    assert lines["M7"]["judged"] and len(lines["M7"]["windows"]) == 10 and len(lines["M8"]["windows"]) == 10
    assert W.breach(L.F, W.StatsReader()) is None and capsys.readouterr().out == ""
    assert not (L.F / "control.json").exists() and list(W.load_state(L.F)) == ["closed"]


@pytest.mark.parametrize("change, words", [
    (dict(exact=False), "does not say"),
    (dict(float32={"change_max_abs": 0.0015, "bit_identical": False}), "differs from the prediction"),
    (dict(autocast=None), "autocast"),
    (dict(output_weight_max_abs=0.01), "not zero"),
    (dict(rows=8), "probe rows"),
    (dict(version="heads_all_slot_path_v0"), "version"),
])
def test_a_switch_record_that_is_not_as_declared_stops_at_once(tmp_path, change, words):
    sw = dict(SWITCH, **change)
    if change.get("autocast", 1) is None:
        del sw["autocast"]
    L = _lineage(tmp_path, switch=sw)
    ev = _ev(L.F)
    assert ev["id"] == "M1:switch" and words in ev["why"] and ev["why"].startswith(W.LABEL + ":"), ev["why"]
    _clean(ev["why"])
    # a CPU run needs no autocast block
    if change.get("autocast", 1) is None:
        L2 = _lineage(tmp_path / "cpu", switch=sw, args=dict(ARGS, device="cpu"))
        assert _ev(L2.F)["why"] is None


def test_a_non_finite_update_stops(tmp_path):
    L = _lineage(tmp_path, after=_after(4, nan=lambda k: 1 if k == 3 else 0))
    ev = _ev(L.F)
    assert ev["id"] == "M2@%d" % _step(3) and "non-finite" in ev["why"]
    _clean(ev["why"])


def test_one_paths_hidden_layer_low_in_two_reports_stops_and_one_report_does_not(tmp_path):
    low = lambda k: {"tactile": 0.5} if k in (3, 4) else None                                    # noqa: E731
    L = _lineage(tmp_path, after=_after(6, live=low))
    ev = _ev(L.F)
    assert ev["id"] == "M3@%d" % _step(4) and "hidden layer" in ev["why"], ev["why"]
    _clean(ev["why"])
    one = _lineage(tmp_path / "one", after=_after(6, live=lambda k: {"tactile": 0.5} if k == 3 else None))
    assert _ev(one.F)["why"] is None
    # the monitor reads the layer live while the path's own gauge reads it low: the lower reading is judged
    recs = _after(6)
    for r in recs[3:5]:
        r["heads_all_slot_path"]["heads"]["vestibular"]["live"] = 0.4
    own = _lineage(tmp_path / "own", after=recs)
    assert _ev(own.F)["id"].startswith("M3@")


def test_a_sense_past_its_line_in_reports_2_and_3_stops(tmp_path):
    up = lambda k: dict(BASE, proprioception=BASE["proprioception"] * W.LINES["W1"]["proprioception"] * 1.05) \
        if k in (2, 3) else BASE                                                                 # noqa: E731
    L = _lineage(tmp_path, after=_after(4, ratio=up))
    ev = _ev(L.F)
    assert ev["id"] == "M4:reports2-3" and "proprioception" in ev["why"]
    _clean(ev["why"])
    one = _lineage(tmp_path / "one", after=_after(4, ratio=lambda k: up(2) if k == 2 else BASE))
    assert _ev(one.F)["why"] is None


def test_a_senses_window_median_past_its_stop_multiple_stops_and_past_its_alert_multiple_alerts(tmp_path, capsys):
    rise = lambda k: dict(BASE, proprioception=1.10 * _stop_line("proprioception")) if k >= 2 else BASE   # noqa: E731
    L = _lineage(tmp_path, after=_after(25, ratio=rise))
    ev = _ev(L.F)
    assert ev["id"] == "M4:w0" and "proprioception" in ev["why"]
    _clean(ev["why"])
    mid = 0.5 * (BASE["audio"] * W.k6_levels("audio")[0] + _stop_line("audio"))
    A = _lineage(tmp_path / "alert", after=_after(25, ratio=lambda k: dict(BASE, audio=mid) if k >= 2 else BASE))
    ev = _ev(A.F)
    assert ev["why"] is None and len(ev["judgement"]["alerts"]) == 1
    assert W.breach(A.F, W.StatsReader()) is None
    out = capsys.readouterr().out
    assert "ALERT" in out and "audio" in out and "asked-change" not in out
    assert W.breach(A.F, W.StatsReader()) is None and capsys.readouterr().out == ""            # printed once


def test_the_step_rate_under_its_line_in_two_windows_running_stops(tmp_path):
    L = _lineage(tmp_path, after=_after(45, sps=lambda k: 0.45 if k < 2 else (0.39 if k < 22 else 0.34)))
    ev = _ev(L.F)
    assert ev["id"] == "M5:w1" and "model steps per second" in ev["why"]
    _clean(ev["why"])
    once = _lineage(tmp_path / "once", after=_after(45, sps=lambda k: 0.45 if k < 2 else 0.39))
    assert _ev(once.F)["why"] is None


def test_a_path_output_above_its_senses_real_change_stops(tmp_path):
    # the line is the sense's own real change per number: 1.2 x it stops, 0.8 x it does not, whichever head
    L = _lineage(tmp_path, after=_after(6, output=lambda k: {"tactile": 1.2 * REAL_CHANGE["tactile"]} if k == 5 else None))
    ev = _ev(L.F)
    assert ev["id"] == "M6@%d" % _step(5) and "real change" in ev["why"] and "tactile" in ev["why"], ev["why"]
    _clean(ev["why"])
    m6 = ev["judgement"]["lines"]["M6"]
    assert m6["state"] == "crossed" and m6["newest"]["step"] == _step(6)
    under = _lineage(tmp_path / "under",
                     after=_after(6, output=lambda k: {"tactile": 0.8 * REAL_CHANGE["tactile"]} if k == 5 else None))
    ev = _ev(under.F)
    assert ev["why"] is None
    m6 = ev["judgement"]["lines"]["M6"]
    assert m6["state"] == "held" and m6["largest_share_of_line"] == pytest.approx(0.8)
    # the same output is under one sense's line and over another's: 0.1 token units is 62% of tactile's real change
    # and 154% of vestibular's
    mixed = _lineage(tmp_path / "mixed", after=_after(6, output=lambda k: {"tactile": 0.1} if k == 5 else None))
    assert _ev(mixed.F)["why"] is None
    mixed2 = _lineage(tmp_path / "mixed2", after=_after(6, output=lambda k: {"vestibular": 0.1} if k == 5 else None))
    ev = _ev(mixed2.F)
    assert ev["id"] == "M6@%d" % _step(5) and "vestibular" in ev["why"]
    # the first design's growth line no longer stops: an output above that line and under the real change
    line1 = W.growth_line(_step(1) - N, 1.5)
    assert line1 == pytest.approx(W.GROWTH_PER_UPDATE * 1.5 * (_step(1) - N)) and 1.1 * line1 < REAL_CHANGE["audio"]
    old = _lineage(tmp_path / "old", after=_after(6, output=lambda k: {"audio": 1.1 * line1} if k == 1 else None))
    assert _ev(old.F)["why"] is None
    # an output that is not a finite number stops (the base's judgement)
    nan = _lineage(tmp_path / "nan", after=_after(6, output=lambda k: {"vision": float("nan")} if k == 4 else None))
    assert _ev(nan.F)["id"] == "M6@%d" % _step(4)
    # a report without the senses' real change is judged against the value before the switch
    blind = _lineage(tmp_path / "blind", after=[_mimo(479400, nf_steps=50)] + [
        _mimo(_step(k), ema=False, output={"tactile": 1.1 * REAL_CHANGE["tactile"]}) for k in range(1, 4)])
    assert _ev(blind.F)["id"] == "M6@%d" % _step(1)
    # with no real change before the switch either it cannot be judged: a warning, never a silent pass
    blind2 = _lineage(tmp_path / "blind2", before=_before(ema=False), after=[_mimo(479400, nf_steps=50)] + [
        _mimo(_step(k), ema=False, output={"tactile": 5.0}) for k in range(1, 4)])
    ev = _ev(blind2.F)
    assert ev["why"] is None and any("real change" in w for w in ev["warnings"]), ev["warnings"]
    assert W.M6_SHARE == 1.0 and W.LINES["M6_share"] == 1.0
    # a sense whose token stops changing does not put the line at 0: its real change is taken as at least half its
    # value at the last full report before the switch (vestibular: 0.065, the floor 0.0325)
    def dead(out):
        recs = [_mimo(479400, nf_steps=50)]
        for k in range(1, 5):
            r = _mimo(_step(k), output={"vestibular": out})
            r["next_frame_norm"]["persistence_ema"]["vestibular"] = 1e-18
            recs.append(r)
        return recs
    under = _lineage(tmp_path / "dead_under", after=dead(0.9 * W.M6_FLOOR * REAL_CHANGE["vestibular"]))
    ev = _ev(under.F)
    assert ev["why"] is None and any("floor" in w for w in ev["warnings"]), ev["warnings"]
    over = _lineage(tmp_path / "dead_over", after=dead(1.2 * W.M6_FLOOR * REAL_CHANGE["vestibular"]))
    ev = _ev(over.F)
    assert ev["id"] == "M6@%d" % _step(1) and "vestibular" in ev["why"], ev["why"]
    _clean(ev["why"])
    assert W.M6_FLOOR == 0.5 and W.reference_change([W.compact(_mimo(479300, path=False))])["tactile"] == pytest.approx(
        REAL_CHANGE["tactile"])


def test_the_first_segments_own_reports_cross_the_output_line_at_479600(tmp_path):
    # the stopped run's readings (Saved/.../H84_fresh_drivers_20260918_seg41n_0856, the record's section 13): the
    # tactile path's output and the tactile sense's running mean squared change at the four reports
    seen = {479400: (0.009723, 0.02427), 479500: (0.081399, 0.024424), 479600: (0.52003, 0.024862),
            479700: (2.69332, 0.025352)}
    recs = []
    for step, (out, ema) in seen.items():
        r = _mimo(step, nf_steps=(50 if step == 479400 else 100), output={"tactile": out})
        r["next_frame_norm"]["persistence_ema"]["tactile"] = ema
        recs.append(r)
    L = _lineage(tmp_path, after=recs)
    ev = _ev(L.F)
    assert ev["id"] == "M6@479600", ev["id"]
    shares = {s_: o / math.sqrt(e) for s_, (o, e) in seen.items()}
    assert shares[479500] < 1.0 < shares[479600] and shares[479400] < 0.1 and shares[479700] > 16


def test_a_path_fixed_vector_against_the_old_heads_stops(tmp_path):
    pair = {"tactile": {"old_rms": 0.12, "path_rms": 0.6 * REAL_CHANGE["tactile"], "cosine": -0.97, "sum_rms": 0.03}}
    L = _lineage(tmp_path, after=_after(6, fixed=lambda k: pair if k >= 3 else None))
    ev = _ev(L.F)
    assert ev["id"] == "M9@%d" % _step(3) and "fixed vector" in ev["why"] and "tactile" in ev["why"], ev["why"]
    _clean(ev["why"])
    assert ev["judgement"]["lines"]["M9"]["state"] == "crossed"
    assert ev["judgement"]["lines"]["M6"]["state"] == "held"
    # a small fixed vector, or one that does not point against the old head's, does not stop
    small = {"tactile": {"path_rms": 0.4 * REAL_CHANGE["tactile"], "cosine": -0.97}}
    along = {"tactile": {"path_rms": 0.9 * REAL_CHANGE["tactile"], "cosine": -0.5}}
    for name, f in (("small", small), ("along", along)):
        Q = _lineage(tmp_path / name, after=_after(6, fixed=f))
        ev = _ev(Q.F)
        assert ev["why"] is None and ev["judgement"]["lines"]["M9"]["state"] == "held", name
    # a report without the block (a version 1 report) is not judged by this line
    recs = _after(6)
    for r in recs:
        r["heads_all_slot_path"].pop("fixed")
    for_v1 = _ev(_lineage(tmp_path / "noblock", after=recs).F)
    assert for_v1["why"] is None and for_v1["judgement"]["lines"]["M9"]["state"] == "held"
    assert W.FIXED_ALERT_SHARE == 0.5 and W.FIXED_ALERT_COSINE == -0.9


def test_a_stop_is_not_repeated_on_its_own_crossing_and_reading_by_hand_writes_nothing(tmp_path):
    L = _lineage(tmp_path, after=_after(4, nan=lambda k: 1 if k == 3 else 0))
    assert _ev(L.F, write=False)["why"] and not (L.F / W.STATE_NAME).exists()
    assert W.lines(L.F, W.StatsReader(), write=False) and not (L.F / W.STATE_NAME).exists()
    why = W.breach(L.F, W.StatsReader())
    assert why and (L.F / W.STATE_NAME).exists() and not (L.F / "motor_asked_change_watch.json").exists()
    # the watcher's stop reaches the run: the crossing is taken and not judged again
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    assert W.breach(L.F, W.StatsReader()) is None
    # a stop that never reached its run is judged again
    L2 = _lineage(tmp_path / "again", after=_after(4, nan=lambda k: 1 if k == 3 else 0))
    assert W.breach(L2.F, W.StatsReader()) and W.breach(L2.F, W.StatsReader())


# ------------------------------------------------------------------------------ the two stop lines on the audit
def test_proprioception_losing_its_own_command_reading_stops_and_is_not_judged_without_a_reading_before(tmp_path):
    # from report 30 on (model step 482,400) the own command predicts no better than the size-matched command
    lost = lambda k: {"proprioception": 0.0} if k >= 30 else None                                  # noqa: E731
    L = _lineage(tmp_path, after=_after(60, own=lost))
    ev = _ev(L.F)
    stop = ev["judgement"]["stop"]
    assert stop["line"] == "M7" and stop["id"] == "M7:w4" and stop["step"] == 483300, stop
    assert "proprioception" in ev["why"] and "standard error" in ev["why"]
    _clean(ev["why"])
    # a window in which the reading is lower and still above its floor does not stop
    lower = _lineage(tmp_path / "lower", after=_after(60, own=lambda k: {"proprioception": 0.1} if k >= 30 else None))
    assert _ev(lower.F)["why"] is None
    # the reports before the switch hold no reading above the floor (a lineage without the asked-change path): the
    # line is not judged and says why
    none = _lineage(tmp_path / "none", before=_before(own={"proprioception": 0.0}),
                    after=_after(60, own={"proprioception": 0.0}))
    ev = _ev(none.F)
    m7 = ev["judgement"]["lines"]["M7"]
    assert ev["why"] is None and m7["state"] == "not judged" and "no reading to keep" in m7["why"]


def test_a_heads_audited_error_above_its_spread_stops_in_the_second_window_running(tmp_path):
    up = W.M8_SPREAD["tactile"] + 0.01
    risen = lambda k: {"tactile": HF_RATIO["tactile"] + up} if k >= 10 else None                   # noqa: E731
    L = _lineage(tmp_path, after=_after(40, hf_ratio=risen))
    ev = _ev(L.F)
    stop = ev["judgement"]["stop"]
    # window 2 (model steps 480,351 to 481,350) is the first above the line, window 3 the second
    assert stop["line"] == "M8" and stop["id"] == "M8:w3:tactile" and stop["step"] == 482300, stop
    assert "tactile head's audited error" in ev["why"]
    _clean(ev["why"])
    w = ev["judgement"]["lines"]["M8"]["windows"]
    assert w[0]["heads"]["tactile"]["over"] is False and w[1]["heads"]["tactile"]["over"] is True
    assert w[1]["heads"]["tactile"]["share_of_spread"] == pytest.approx(up / W.M8_SPREAD["tactile"], rel=1e-6)
    # one window alone above the line, or a rise under the spread in every window, does not stop
    once = _lineage(tmp_path / "once", after=_after(
        40, hf_ratio=lambda k: {"tactile": HF_RATIO["tactile"] + up} if 10 <= k < 20 else None))
    assert _ev(once.F)["why"] is None
    under = _lineage(tmp_path / "under", after=_after(
        40, hf_ratio={"tactile": HF_RATIO["tactile"] + W.M8_SPREAD["tactile"] - 0.005}))
    assert _ev(under.F)["why"] is None
    # every head has its own spread
    v = _lineage(tmp_path / "vest", after=_after(
        40, hf_ratio={"vestibular": HF_RATIO["vestibular"] + W.M8_SPREAD["vestibular"] + 0.005}))
    assert _ev(v.F)["judgement"]["stop"]["id"] == "M8:w2:vestibular"


def test_the_spreads_and_the_output_line_are_the_numbers_of_their_measurement_files():
    heads_dir = TREE / "docs" / "gain_of_function_v4" / "restart_candidate" / "heads"
    audit = json.loads((heads_dir / "false_stops_lineage_result.json").read_text(encoding="utf-8"))["audit"]
    for m in ("vision", "tactile", "audio", "vestibular"):
        measured = audit["largest_rise"][m]["largest"]
        # the constant is the measured largest rise, rounded up to the digits written
        assert measured <= W.M8_SPREAD[m] <= measured + 0.001, (m, measured)
    measured = audit["largest_rise"]["proprioception"]["window_to_window"]["largest_rise_from_one_window_to_a_later_one"]
    assert measured <= W.M8_SPREAD["proprioception"] <= measured + 0.001
    growth = json.loads((heads_dir / "growth_line_result.json").read_text(encoding="utf-8"))
    largest = max(f["summed_abs_gelu_largest"] for f in growth["folds"])
    assert largest <= W.SUMMED_HIDDEN + 0.05 and W.SUMMED_HIDDEN <= largest + 0.5
    assert W.GROWTH_PER_UPDATE == pytest.approx(3e-4 * W.SUMMED_HIDDEN / math.sqrt(192.0))
    # the output line M6 (a path's output over its sense's real change): every reading without a runaway is under
    # the line, and the first segment's report at 479,600 is over it
    m6 = json.loads((heads_dir / "m6_line_result.json").read_text(encoding="utf-8"))
    assert m6["line"] == W.M6_SHARE and m6["floor"] == W.M6_FLOOR
    # the floor is under every living sense's lowest real change on the lineage, as a share of its value before
    # the switch, and the bench's largest outputs stay under the line on the lineage's lowest levels
    assert all(v > W.M6_FLOOR for v in m6["lineage"]["lowest_over_reference_while_the_token_lived"].values())
    assert all(v < W.M6_SHARE for v in m6["bench"]["worst_case_share_on_the_lineage"].values())
    assert 0 < m6["bench"]["largest_share"] < W.M6_SHARE and m6["bench"]["readings_over_the_line"] == 0
    assert 0 < m6["repaired_cpu_run"]["largest_share"] < W.M6_SHARE
    assert m6["repaired_cpu_run"]["readings_over_the_line"] == 0 and m6["repaired_cpu_run"]["readings"] > 0
    assert m6["first_segment"]["first_report_over_the_line"] == 479600
    assert m6["first_design_cpu_run"]["first_model_step_over_the_line"] is not None
    # the pair line M9: no model step of the repaired design's run crosses it; the first design's crosses it
    # before it crosses M6
    pair = m6["pair"]
    assert pair["share"] == W.FIXED_ALERT_SHARE and pair["cosine"] == W.FIXED_ALERT_COSINE
    assert pair["repaired_cpu_run"]["model_steps_read"] > 300 and pair["repaired_cpu_run"]["model_steps_crossing"] == 0
    assert pair["first_design_cpu_run"]["first_model_step_crossing"] is not None
    assert pair["first_design_cpu_run"]["first_model_step_crossing"] <= \
        m6["first_design_cpu_run"]["first_model_step_over_the_line"]


# ------------------------------------------------------------------------------ the alert, the early and late lines
def test_a_lessons_gradient_under_its_liveness_line_alerts_and_never_stops(tmp_path, capsys):
    L = _lineage(tmp_path, after=_after(40, gradient={"tactile": 3e-10}))
    ev = _ev(L.F)
    assert ev["why"] is None
    alerts = [a for a in ev["judgement"]["alerts"] if a["id"] == "G-alert:tactile"]
    assert len(alerts) == 1 and alerts[0]["step"] == 480400 and "never a stop" in alerts[0]["why"]
    _clean(alerts[0]["why"])
    assert not [a for a in ev["judgement"]["alerts"] if a["id"] == "G-alert:vestibular"]
    assert W.breach(L.F, W.StatsReader()) is None and "ALERT" in capsys.readouterr().out
    assert W.breach(L.F, W.StatsReader()) is None and capsys.readouterr().out == ""      # printed once
    E = ev["judgement"]["acceptance"]["early"]
    assert E["gradient"]["tactile"]["state"] == "missed" and E["gradient"]["vestibular"]["state"] == "met"
    assert E["state"] == "missed"


def test_the_early_lines_and_the_late_pass_lines(tmp_path):
    # after the switch: tactile's error falls below the no-change error, vestibular's error is lower with the own
    # command, audio's whole prediction answers to one slot
    L = _lineage(tmp_path, after=_after(101, own={"vestibular": 0.05}, hf_ratio={"tactile": 0.95},
                                        head_read={"audio": 0.1}))
    A = _ev(L.F)["judgement"]["acceptance"]
    assert A["reference_reports"] == 30 and A["reference_steps"] == [476400, 479300]
    E = A["early"]
    assert E["model_steps"] == [N + 2001, N + 3000] and E["reports"] == 10 and E["complete"]
    assert {m: v["state"] for m, v in E["gradient"].items()} == {"tactile": "met", "vestibular": "met"}
    assert E["head_read"]["audio"]["state"] == "missed" and E["head_read"]["tactile"]["state"] == "met"
    assert E["head_read"]["tactile"]["old_pool_spread"] == 0.0 and E["head_read"]["tactile"]["path_slot_reach"] == 0.95
    assert E["state"] == "missed"
    l1, l2 = A["late"]["L1"], A["late"]["L2"]
    assert l1["model_steps"] == [N + 9001, N + 10000] and l1["state"] == "met"
    assert l1["reading"]["value"] == pytest.approx(0.95) and l1["reference"] == pytest.approx(HF_RATIO["tactile"])
    assert l2["model_steps"] == [N + 2001, N + 10000] and l2["state"] == "met"
    assert l2["reading"]["value"] == pytest.approx(0.05) and l2["late_window"]["value"] == pytest.approx(0.05)
    assert l1["goal"] == W.GOAL["tactile"]["error"] and l2["goal"] == W.GOAL["vestibular"]["own"]
    # the watched readings of every 1,000 model steps
    assert len(A["windows"]) == 10 and all(w["complete"] for w in A["windows"])
    assert A["windows"][3]["heads"]["proprioception"]["own_command"]["value"] == pytest.approx(OWN["proprioception"])
    # nothing learned: both late lines are missed; a short record waits
    flat = _ev(_lineage(tmp_path / "flat", after=_after(101)).F)["judgement"]["acceptance"]
    assert flat["late"]["L1"]["state"] == "missed" and flat["late"]["L2"]["state"] == "missed"
    assert flat["early"]["state"] == "met"
    short = _ev(_lineage(tmp_path / "short", after=_after(12)).F)["judgement"]["acceptance"]
    assert short["early"]["state"] == "waiting" and short["late"]["L1"]["state"] == "waiting"
    assert short["late"]["L2"]["state"] == "waiting"


def test_the_sweeps_lines_name_this_switch_and_carry_no_bare_decimal(tmp_path):
    L = _lineage(tmp_path, after=_after(101, own={"vestibular": 0.05}, hf_ratio={"tactile": 0.95}))
    printed = W.lines(L.F, W.StatsReader(), write=False)
    assert printed and printed[0].lstrip().startswith(W.LABEL)
    text = "\n".join(printed)
    for m in HEADS:
        assert "%s head, watched" % m in text
    for words in ("--heads-all-slot-path", "early lines", "late pass line for tactile", "late pass line for vestibular",
                  "proprioception's own-command reading in every 1,000 model steps", "every head's audited error",
                  "liveness reading"):
        assert words in text, words
    for p in printed:
        _clean(p)
    # a short record prints too
    for p in W.lines(_lineage(tmp_path / "short", after=_after(12)).F, W.StatsReader(), write=False):
        _clean(p)


# ------------------------------------------------------------------------------ the hourly watcher
def _hourly(monkeypatch, run, hours="1"):
    """The hourly watcher's program loaded as the supervisor starts it for `run`, with healthy resources and its
    Saved folders in the test's own folder (never the real supervisor files)."""
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


def test_the_hourly_watchers_own_program_stops_the_run_and_keeps_its_three_rules_apart(tmp_path, monkeypatch):
    rise = lambda k: dict(BASE, audio=1.10 * _stop_line("audio")) if k >= 2 else BASE             # noqa: E731
    L = _lineage(tmp_path, after=_after(25, ratio=rise))
    hw = _hourly(monkeypatch, L.F)
    why = hw["quick_breach"]()
    assert why and why.startswith(W.LABEL + ":") and "audio" in why, why
    hw["stop_run"](why)
    assert json.loads((L.F / "control.json").read_text())["cmd"] == "snapshot_and_stop"
    reason = json.loads((L.F / "stop_reason.json").read_text())
    assert reason["resume_policy"] == "repair_required" and reason["reason"] == why
    heads, motor, vest = hw["_heads_all_slot_path_rule"](), hw["_motor_asked_change_rule"](), hw["_vestibular_rule"]()
    assert heads.MW is not motor and motor.BASE is not vest and heads.BASE is not vest and heads.BASE is not motor.BASE
    assert motor.SWITCH_TYPE == "motor_asked_change_switch" and vest.SWITCH_TYPE == "vestibular_encoder_repair"
    assert motor.LINES["growth_per_optimizer_update"] == pytest.approx(3e-4 * 239.0 / 26.0)
    assert vest.LINES == V.LINES and motor.compact is not heads.compact
    assert hw["motor_asked_change_breach"]() is None and hw["vestibular_repair_breach"]() is None
    assert hw["heads_all_slot_path_lines"]()


# ------------------------------------------------------------------------------ a line read as an alert for one run
# The operator's decision of Oct 6, 2026: control run 1 of the attention cap experiment was stopped by M3 (a path's
# hidden layer under 0.9 live share) at model step 484,600; control run 2 and the cap arm read M3 as an alert and end at
# model step 484,675, and every other line stops them as before.
CAP_RESUME = (r"C:\Users\bhart\Documents\UEMA_devA_private\Saved\mimo_bridge\experiments"
              r"\H84_fresh_drivers_20260918_seg41o_1431\checkpoints\ckpt_step_481800")


def test_a_line_read_as_an_alert_for_the_run_is_an_alert_and_every_other_line_still_stops(tmp_path, capsys):
    low = lambda k: {"tactile": 0.5} if k in (3, 4) else None                                    # noqa: E731
    L = _lineage(tmp_path, after=_after(6, live=low))
    alerts = {"M3": "a test's reason"}
    assert W.breach(L.F, W.StatsReader(), alert_instead_of_stop=alerts) is None
    printed = capsys.readouterr().out
    assert "ALERT for this run, not a stop (a test's reason)" in printed and "hidden layer" in printed, printed
    assert not (L.F / "control.json").exists()
    assert W.MW.ALERT_INSTEAD_OF_STOP == {}                      # set for the one call only
    ev = W._with_alerts(alerts, W.evaluate, L.F, W.StatsReader(), False)
    assert ev["why"] is None and ev["judgement"]["lines"]["M3"]["state"] == "crossed, an alert for this run"
    alert = [a for a in ev["judgement"]["alerts"] if a["line"] == "M3"]
    assert len(alert) == 1 and alert[0]["id"] == "M3-alert-for-this-run:%s:M3@%d" % (Path(ev["chain"][-1]).name, _step(4))
    _clean(alert[0]["why"])
    # the alert is printed once: the next check records nothing new
    assert W.breach(L.F, W.StatsReader(), alert_instead_of_stop=alerts) is None
    assert "ALERT for this run" not in capsys.readouterr().out
    # another line crossing later is not hidden behind the earlier alert: a non-finite update at report 5 stops the run
    both = _lineage(tmp_path / "both", after=_after(6, live=low, nan=lambda k: 1 if k == 5 else 0))
    why = W.breach(both.F, W.StatsReader(), alert_instead_of_stop=alerts)
    assert why and "non-finite" in why, why
    # without the alert the same records stop the run on M3, as before
    plain = _lineage(tmp_path / "plain", after=_after(6, live=low))
    why = W.breach(plain.F, W.StatsReader())
    assert why and "hidden layer" in why, why


def test_two_sibling_runs_crossing_the_alert_line_at_one_model_step_each_print_their_own_alert(tmp_path, capsys):
    """Control run 2 and the cap arm both continue the switch's lineage, and the deploy run's state file holds the alerts
    of every run under it. Two runs crossing M3 at the same model step each print their own alert once (the check of the
    M3 alert, branch review/m3-alert-end-step-2026-10-06, commit d795e90b7: with an id without the run, the second run's
    5-minute check printed nothing)."""
    exp = tmp_path / "experiments"
    _run(exp, "H84_before", parent="H84_grandparent", ck_step=476300, records=_before(),
         args=dict(ARGS, heads_all_slot_path="off"))
    _run(exp, "H84_switch", parent="H84_before", ck_step=N, records=[_switch()] + _after(2))
    recs = [_mimo(_step(k), live={"tactile": 0.5} if k in (5, 6) else None) for k in range(3, 9)]
    a = _run(exp, "H84_arm_control2", parent="H84_switch", ck_step=_step(2), records=recs)
    b = _run(exp, "H84_arm_cap", parent="H84_switch", ck_step=_step(2), records=recs)
    alerts = {"M3": "a test's reason"}
    printed = []
    for run in (a, b, a, b):
        assert W.breach(run, W.StatsReader(), alert_instead_of_stop=alerts) is None
        printed.append(capsys.readouterr().out.count("ALERT for this run"))
    assert printed == [1, 1, 0, 0], printed
    held = sorted((x["id"], Path(x["run"]).name) for x in W.load_state(exp / "H84_switch").get("alerts") or []
                  if x.get("line") == "M3")
    assert held == [("M3-alert-for-this-run:H84_arm_cap:M3@%d" % _step(6), "H84_arm_cap"),
                    ("M3-alert-for-this-run:H84_arm_control2:M3@%d" % _step(6), "H84_arm_control2")], held


@pytest.mark.parametrize("max_steps, alert", [(484675, True), (484800, False), (0, False)])
def test_the_hourly_watcher_reads_m3_as_an_alert_only_for_the_cap_experiments_later_arms(tmp_path, monkeypatch,
                                                                                          max_steps, alert):
    low = lambda k: {"tactile": 0.5} if k in (3, 4) else None                                    # noqa: E731
    L = _lineage(tmp_path, after=_after(6, live=low), args=dict(ARGS, resume=CAP_RESUME, max_steps=max_steps))
    hw = _hourly(monkeypatch, L.F)
    assert bool(hw["heads_all_slot_path_alerts"]()) is alert
    why = hw["heads_all_slot_path_breach"]()
    if alert:
        assert why is None and set(hw["heads_all_slot_path_alerts"]()) == {"M3"}
    else:
        assert why and "hidden layer" in why, why
    other = _lineage(tmp_path / "other", after=_after(6, live=low),
                     args=dict(ARGS, resume=CAP_RESUME.replace("481800", "480829"), max_steps=484675))
    assert _hourly(monkeypatch, other.F)["heads_all_slot_path_alerts"]() == {}


# The operator's decision for the remembered-hand continuation (apply_m3_alert.py): M3 is an alert only for the run resuming
# checkpoint 484,675 of seg41j_1526 with max_steps 489350; the cap experiment's later arms keep their own entry.
RHL_RESUME = (r"C:\Users\bhart\Documents\UEMA_devA_private\Saved\mimo_bridge\experiments"
              r"\H84_fresh_drivers_20260918_seg41j_1526\checkpoints\ckpt_step_484675")


@pytest.mark.parametrize("resume, max_steps, alert", [
    (RHL_RESUME, 489350, True), (RHL_RESUME, 484675, False), (RHL_RESUME, 0, False),
    (RHL_RESUME.replace("484675", "484600"), 489350, False), (CAP_RESUME, 489350, False), (CAP_RESUME, 484675, True)])
def test_the_hourly_watcher_reads_m3_as_an_alert_only_for_the_named_runs(tmp_path, monkeypatch, resume, max_steps, alert):
    low = lambda k: {"tactile": 0.5} if k in (3, 4) else None                                    # noqa: E731
    L = _lineage(tmp_path, after=_after(6, live=low), args=dict(ARGS, resume=resume, max_steps=max_steps))
    hw = _hourly(monkeypatch, L.F)
    assert bool(hw["heads_all_slot_path_alerts"]()) is alert
    why = hw["heads_all_slot_path_breach"]()
    if alert:
        assert why is None and set(hw["heads_all_slot_path_alerts"]()) == {"M3"}
    else:
        assert why and "hidden layer" in why, why
