# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The stop lines of the asked-change path's switch in production (docs/gain_of_function_v4/
motor_asked_change_watch.py), judged by the H84 hourly watcher (docs/h84_launch_2026-09-18/hourly_watch.py).

Hand-made stats records in run folders linked the way the supervisor links them (continuation_receipt.json): 30
full reports before the switch in a parent run, then a run that carries the motor_asked_change_switch record at model
step 465,825 (a checkpoint off the hundreds, as a requested stop writes it), and continuations. What is held here:
  - a run that no switch record reaches is not judged: no stop, no line, nothing printed, by the module and by the
    hourly watcher's own program; the vestibular watch on the same folders reads nothing of this switch, and its
    lines and its file are as production's;
  - every planted crossing stops and every clean record does not: the switch record not as declared (not exact, a
    changed token, a missing autocast check on the graphics card, an output layer that is not zero), a non-finite
    update, the path's hidden layer under 0.9 live share in 2 reports running (one report is not a stop), a sense
    past its line in reports 2 and 3 (one report is not), a sense's 20-report median past its stop multiple (a stop)
    or past its alert multiple (an alert, printed once, recorded, never a stop), the vestibular sense's ratio (an
    alert only, whatever its size), the step rate under 0.9 of the window before in two windows running, and the
    path's output above its growth line (and the alert at one token unit);
  - the growth line counts optimizer updates: 1.5 per model step with --replay on --replay-every 2;
  - the lines without the reports before the switch: the switch record, non-finite updates, the hidden layer and the
    output are judged, and the check says loudly what is not;
  - a stop is not repeated on its own crossing, and one that never reached its run is judged again; reading by hand
    writes nothing;
  - line S1: the reference from the reports before the switch above model step 457,825, the judged window of model
    steps N + 2,001 to N + 3,000, met when the reading clears 3.14 standard errors and exceeds the reference by 3.14
    standard errors of the difference, missed when the path changes nothing; its arithmetic equals the trial
    driver's on the same cells;
  - the hourly watcher's own program writes the stop into the relaunched run, and its sweep prints the lines with no
    bare score decimal (DESIGN.md section 11);
  - what this watch does itself (the module note, WHAT THIS MODULE DOES ITSELF): a run that resumed the checkpoint the
    switched run saved at its launch step is watched and the revert at that model step is not; an exception inside a
    read loses no record; a block that is not an object raises nothing; a sense at 0 before the switch is named and
    the others are judged; the hidden layer's count runs over the reports that carry a reading and reads the lower of
    its two readings; the watch ends at the report that closes the last window; an unreadable args.json stops nothing
    by itself; an output that is not a finite number stops."""
import collections
import importlib.util
import json
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, unanchored_numbers

HERE = Path(__file__).resolve()
TREE = HERE.parents[4]
SCRIPT = TREE / "docs" / "gain_of_function_v4" / "motor_asked_change_watch.py"
VESTIBULAR = TREE / "docs" / "gain_of_function_v4" / "vestibular_repair_watch.py"
WATCHER = TREE / "docs" / "h84_launch_2026-09-18" / "hourly_watch.py"
TRIAL = TREE / "docs" / "gain_of_function_v4" / "motor_asked_change_trial_2026-10-03.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load(SCRIPT, "motor_asked_change_watch")
V = _load(VESTIBULAR, "vestibular_repair_watch_fresh")
SENSES = ("vision", "tactile", "audio", "proprioception")
BASE = {"vision": 1.0, "tactile": 1.0, "audio": 1.2, "proprioception": 0.9, "vestibular": 0.2}
N = 465825
PER_STEP = 1.5
ARGS = {"report_every": 100, "replay": "on", "replay_every": 2, "device": "cuda:0", "motor_asked_change": "on"}
SWITCH = {"rows": 64, "output_weight_max_abs": 0.0, "output_bias_max_abs": 0.0,
          "float32": {"change_max_abs": 0.0, "token_rms": 1.0, "bit_identical": True},
          "autocast": {"change_max_abs": 0.0, "token_rms": 1.0, "bit_identical": True},
          "exact": True, "version": "motor_asked_change_v1", "checkpoint": "ckpt_step_465825", "at_step": N}
HEALTHY_DISK = collections.namedtuple("usage", "total used free")(1e12, 0.0, 500e9)


def _hold_first(own_gain=0.0, bodies=8):
    """A prediction_audit_hold_first block: per body 5 transitions; the size-matched command's error is 0.01 and
    the own command's 0.01 x (1 - own_gain), so the pooled reading is own_gain."""
    return {"excluded_bodies": [2], "senses": {"proprioception": {"per_body": {
        "size_matched_transitions": [5] * bodies,
        "size_matched_own_prediction_mse": [0.01 * (1.0 - own_gain) * (1.0 + 0.01 * ((b * 7) % 5)) for b in range(bodies)],
        "size_matched_action_mse": [0.01 * (1.0 + 0.01 * ((b * 7) % 5)) for b in range(bodies)]}}}}


def _mimo(step, nf_steps=100, ratio=None, sps=0.42, nan=0, live=1.0, output=None, own_gain=0.0, path=True,
          hold_first=True, monitor=True):
    """A stats record as the trainer writes it, with the fields the watch reads. path=False: a record of a run
    without the path (before the switch)."""
    r = {"type": "mimo", "step": step, "timestamp": 1.0e9 + step, "elapsed_s": 600.0,
         "next_frame_norm": {"steps": nf_steps, "ratio": dict(BASE if ratio is None else ratio)},
         "nan_steps": nan, "model_steps_per_s": sps,
         "tenure_pe": {"pose": 0.0002}, "tenure_gap": {"pose": 0.0},
         "contact_control": {"gate_self_open_share": 0.0, "gate_world_open_share": 0.0},
         "controllability": {"gate_commands": 20, "gate_open_commands": 20, "shadow_gate_commands": 20,
                             "shadow_gate_open_commands": 20, "gates_same_commands": 20,
                             "gate_only_open_commands": 0, "shadow_gate_only_open_commands": 0},
         "causal_chain": {"chosen_tied_share": 1.0, "candidate_spread_mean": 0.003, "tie_band": 0.07},
         "machine": {"gpu_used_mib": 6760.0, "gpu_total_mib": 8188.0},
         "liveness": {"mode": "on", "layers": {"forward_model.blocks.0.ffn.0": 1.0}}}
    if hold_first:
        r["prediction_audit_hold_first"] = _hold_first(own_gain)
    if path:
        out = 0.0005 * max(0, (step - N)) / 100.0 if output is None else output
        r["motor_asked_change"] = {"version": "motor_asked_change_v1", "rows": 64, "encoder_output_sd": 9846.0,
                                   "output_weight_rms": 0.0001, "output_rms": out,
                                   "handle_reach": {"spread": 0.9, "largest_share": 0.08, "handles_under_1_in_1000": 0},
                                   "features_rms": {"position": 0.99, "turn": 1.01, "strength": 0.5, "radius": 0.5,
                                                    "eye": 0.95},
                                   "liveness": {"net.0": live}}
        if monitor:
            r["liveness"]["layers"]["motor_asked_change.net.0"] = live
    return r


def _run(exp, name, parent=None, ck_step=None, records=(), args=None, ck_parent=None):
    """ck_parent: the run folder that holds the resumed checkpoint when it is not the receipt's parent run."""
    run = exp / name
    run.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps(ARGS if args is None else args), encoding="utf-8")
    if parent is not None:
        pp = exp / parent
        cp = exp / (ck_parent or parent)
        (run / "continuation_receipt.json").write_text(json.dumps({
            "parent_run": str(pp), "checkpoint": str(cp / "checkpoints" / ("ckpt_step_%d" % ck_step)),
            "timestamp": time.time(), "reason": "test"}), encoding="utf-8")
    with open(run / "replay_training_stats.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return run


def _switch(step=N, switch=None):
    return {"type": "motor_asked_change_switch", "step": step, "timestamp": 1.0, "switch": SWITCH if switch is None
            else switch}


def _before(n=30, **kw):
    """n full reports before the switch (model steps ending at 465,800), on the code that records the block."""
    return [_mimo(465800 - 100 * (n - 1 - i), path=False, **{k: (v(i) if callable(v) else v) for k, v in kw.items()})
            for i in range(n)]


def _step(k):
    """The model step of full report k after the switch at 465,825 (the first record, 465,900, is partial)."""
    return 465900 + 100 * k


def _after(n, first=1, partial=True, **kw):
    out = [_mimo(465900, nf_steps=75)] if partial else []
    for k in range(first, n + 1):
        out.append(_mimo(_step(k), **{key: (v(k) if callable(v) else v) for key, v in kw.items()}))
    return out


def _lineage(tmp_path, after=(), switch=None, before=None, args=None, parent_start=462700):
    exp = tmp_path / "experiments"
    P = _run(exp, "H84_before", parent="H84_grandparent", ck_step=parent_start,
             records=_before() if before is None else before, args=dict(ARGS, motor_asked_change="off"))
    F = _run(exp, "H84_switch", parent="H84_before", ck_step=N, records=[_switch(switch=switch)] + list(after),
             args=args)
    return SimpleNamespace(exp=exp, P=P, F=F)


def _ev(run, write=True):
    return W.evaluate(run, W.StatsReader(), write=write)


def _stop_line(sense):
    return BASE[sense] * W.k6_levels(sense)[1]


def _alert_line(sense):
    return BASE[sense] * W.k6_levels(sense)[0]


# ------------------------------------------------------------------------------ what the module is built from
def test_the_rule_module_is_this_trees_and_the_vestibular_watch_is_unchanged():
    assert Path(W.__file__).resolve() == SCRIPT.resolve()
    # the multiples are the vestibular watch's own (a fresh load of its file), and loading this module changed
    # nothing of that fresh instance
    for key in ("W1", "W20", "RISE", "alert_margin", "stop_margin", "before_reports", "level_reports",
                "window_reports", "horizon_steps", "K4_liveness", "K4_consecutive", "K5_share", "K5_consecutive"):
        assert W.LINES[key] == V.LINES[key], key
    assert W.LINES["M3_liveness"] == V.LINES["K4_liveness"] and W.LINES["M5_share"] == V.LINES["K5_share"]
    assert V.SWITCH_TYPE == "vestibular_encoder_repair" and V.STATE_NAME == "vestibular_repair_watch.json"
    assert "M3_liveness" not in V.LINES and V.compact is not W.compact
    assert W.BASE.SWITCH_TYPE == W.SWITCH_TYPE == "motor_asked_change_switch"
    assert W.k6_levels("audio") == V.k6_levels("audio") and W.k6_levels("vision") == V.k6_levels("vision")
    # the vestibular watch's file is the file production's commit holds, when git can say so
    try:
        diff = subprocess.run(["git", "-C", str(TREE), "diff", "--quiet", "fix/explainer-audit-2026-09-15", "--",
                               "docs/gain_of_function_v4/vestibular_repair_watch.py"], capture_output=True)
    except OSError:
        pytest.skip("git is not available")
    if diff.returncode not in (0, 1):
        pytest.skip("this tree is not a checkout with production's branch")
    assert diff.returncode == 0, "vestibular_repair_watch.py differs from production's"


def test_the_growth_bound_and_the_path_constants_are_the_builds():
    # the bound per optimizer update: learning rate x summed input size / 26; the trial driver holds the same number
    assert W.LINES["growth_per_optimizer_update"] == pytest.approx(3e-4 * 239.0 / 26.0)
    T = _load(TRIAL, "motor_asked_change_trial")
    assert T.GROWTH_BOUND == pytest.approx(W.LINES["growth_per_optimizer_update"])
    assert T.SWITCH_TYPE == W.SWITCH_TYPE and T.PATH_LAYER == W.PATH_LAYER and T.Z == W.LINES["S1_z"]
    assert tuple(T.EXCLUDED_BODIES) == tuple(W.EXCLUDED_BODIES) and T.JUDGED_SENSE == W.JUDGED_SENSE
    assert T.PHASES["hold_first"] == W.HOLD_FIRST and T.GOAL_REFERENCE == W.LINES["S1_goal_reference"]
    assert tuple(T.JUDGED) == tuple(W.LINES["S1_window"])
    src = (TREE / "Source" / "Python" / "lem" / "model" / "motor_asked_change.py").read_text(encoding="utf-8")
    assert 'ASKED_CHANGE_VERSION = "%s"' % W.PATH_VERSION in src and 'LIVENESS_LAYER = "%s"' % W.PATH_LAYER in src
    # optimizer updates per model step from the run's settings
    assert W.optimizer_updates_per_model_step({"replay": "on", "replay_every": 2}) == 1.5
    assert W.optimizer_updates_per_model_step({"replay": "off", "replay_every": 2}) == 1.0
    assert W.optimizer_updates_per_model_step({}) == 1.0
    assert W.growth_line(100, 1.5) == pytest.approx(150 * 3e-4 * 239.0 / 26.0)


def test_the_readings_arithmetic_is_the_trial_drivers():
    T = _load(TRIAL, "motor_asked_change_trial")
    rec = _mimo(_step(5), own_gain=0.03)
    mine = W.swap_cells(rec)
    theirs = T.swap_cells(rec, "prediction_audit_hold_first", "proprioception")
    assert [tuple(c) for c in mine] == [tuple(c) for c in theirs] and len(mine) == 7        # body 2 left out
    a, b = W.own_command_reading(mine), T.own_command_reading(theirs)
    assert a == b and a["value"] == pytest.approx(0.03) and a["transitions"] == 35
    assert W.hold_loss_ratio(rec) == T.hold_loss_ratio(rec) == 1.0
    assert W.path_live(rec) == T.path_live(rec) == 1.0
    assert W.path_live(_mimo(_step(5), live=0.5, monitor=False)) == 0.5                      # the path's own gauge


# ------------------------------------------------------------------------------ which runs are judged
def test_a_run_no_switch_reaches_is_quiet(tmp_path, monkeypatch, capsys):
    L = _lineage(tmp_path, after=_after(3))
    assert W.breach(L.P, W.StatsReader()) is None and W.lines(L.P, W.StatsReader()) == []
    assert W.main([str(L.P)]) == 0
    assert "no motor_asked_change_switch record reaches" in capsys.readouterr().out
    hw = _hourly(monkeypatch, L.P)
    assert hw["motor_asked_change_breach"]() is None and hw["motor_asked_change_lines"]() == []
    assert "ASKED-CHANGE" not in capsys.readouterr().out
    assert not (L.P / W.STATE_NAME).exists()
    # the vestibular watch reads nothing of this switch on the switched run either
    assert V.breach(L.F, V.StatsReader()) is None and V.lines(L.F, V.StatsReader()) == []


def test_the_hourly_watchers_two_rules_are_separate_instances(tmp_path, monkeypatch):
    # the hourly watcher loads the vestibular watch and this program in one process: the vestibular rule must keep
    # its own switch type, state file, record reader and lines whatever this program replaced on its private instance
    L = _lineage(tmp_path, after=_after(3))
    hw = _hourly(monkeypatch, L.F)
    assert hw["motor_asked_change_lines"]()                         # loads this program (and its private instance)
    vest, motor = hw["_vestibular_rule"](), hw["_motor_asked_change_rule"]()
    assert vest is not motor.BASE and vest.LINES is not motor.LINES
    assert vest.SWITCH_TYPE == "vestibular_encoder_repair" and vest.STATE_NAME == "vestibular_repair_watch.json"
    assert vest.compact is not motor.compact and "M3_liveness" not in vest.LINES
    assert vest.LINES == V.LINES                                    # equal to a fresh load of the file
    assert type(vest.READER) is vest.StatsReader and vest.StatsReader is not motor.StatsReader
    assert vest.find_switch is not motor.find_switch
    assert hw["vestibular_repair_breach"]() is None and hw["vestibular_repair_lines"]() == []
    assert not (L.F / vest.STATE_NAME).exists()


def test_which_runs_are_watched(tmp_path):
    L = _lineage(tmp_path, after=_after(4))
    assert _ev(L.F)["N"] == N and _ev(L.P) is None
    # a continuation after the switch is watched, on the whole timeline
    G = _run(L.exp, "H84_cont", parent="H84_switch", ck_step=_step(4), records=_after(8, first=5, partial=False))
    ev = _ev(G)
    assert ev["N"] == N and ev["judgement"]["after_full_reports"] == 8 and Path(ev["deploy"]) == L.F
    # the revert (a run that resumed the checkpoint before the switch, without the flag) is not
    R = _run(L.exp, "H84_revert", parent="H84_before", ck_step=465800, records=[_mimo(465900, path=False)],
             args=dict(ARGS, motor_asked_change="off"))
    assert _ev(R) is None
    # nor the revert that resumed the checkpoint AT the switch's model step, which lies in the parent's folder:
    # linked to that parent, or linked to the stopped switched run (as the relaunch script links a stopped run)
    R2 = _run(L.exp, "H84_revert_at_N", parent="H84_before", ck_step=N,
              records=[_mimo(465900, nf_steps=75, path=False), _mimo(466000, path=False)],
              args=dict(ARGS, motor_asked_change="off"))
    R3 = _run(L.exp, "H84_revert_at_N_linked_to_the_switched_run", parent="H84_switch", ck_parent="H84_before",
              ck_step=N, records=[_mimo(465900, nf_steps=75, path=False), _mimo(466000, path=False)],
              args=dict(ARGS, motor_asked_change="off"))
    assert _ev(R2) is None and _ev(R3) is None
    # a second load of the same checkpoint with the flag writes its own switch record: that run is the deploy
    F2 = _run(L.exp, "H84_switch_again", parent="H84_switch", ck_parent="H84_before", ck_step=N,
              records=[_switch()] + _after(3))
    ev = _ev(F2)
    assert Path(ev["deploy"]) == F2 and ev["judgement"]["before_reports"] == 30
    assert ev["judgement"]["after_full_reports"] == 3


def test_a_stop_before_the_first_model_step_does_not_end_the_watch(tmp_path):
    # the switched run consumed a stop at its first poll: it holds the switch record and no report and saved the
    # switched state at N in its own folder; the run that resumed that checkpoint carries the path and no switch
    # record (the lineage's case: seg40_resume_20260925_030344 saved at its launch step 402,150)
    exp = tmp_path / "experiments"
    _run(exp, "H84_before", parent="H84_grandparent", ck_step=462700, records=_before(),
         args=dict(ARGS, motor_asked_change="off"))
    F = _run(exp, "H84_switch", parent="H84_before", ck_step=N, records=[_switch()])
    rise = lambda k: dict(BASE, audio=1.10 * _stop_line("audio")) if k >= 2 else BASE                      # noqa: E731
    G = _run(exp, "H84_switch_resume", parent="H84_switch", ck_step=N, records=_after(25, ratio=rise))
    ev = _ev(G)
    assert ev is not None and ev["N"] == N and Path(ev["deploy"]) == F
    assert ev["judgement"]["before_reports"] == 30 and ev["judgement"]["after_full_reports"] == 25
    assert ev["id"] == "M4:w0" and "audio" in ev["why"]
    H = _run(exp, "H84_later", parent="H84_switch_resume", ck_step=_step(25),
             records=_after(30, first=26, partial=False, ratio=rise))
    assert _ev(H)["id"] == "M4:w0"
    assert _ev(F)["why"] is None                                    # the switched run itself: judged, nothing to stop
    # the vestibular watch's own walk does not reach that run (its rule is unchanged)
    assert W.BASE.find_switch(G, W.StatsReader()) is None


# ------------------------------------------------------------------------------ the stop lines
def test_a_clean_switch_holds_every_line_through_the_watch(tmp_path, capsys):
    L = _lineage(tmp_path, after=_after(101))
    ev = _ev(L.F)
    assert ev["why"] is None and ev["watch_over"] and not ev["warnings"], (ev["why"], ev["warnings"])
    lines = ev["judgement"]["lines"]
    assert {k: v["state"] for k, v in lines.items()} == {k: "held" for k in ("M1", "M2", "M3", "M4_early", "M4",
                                                                             "M5", "M6")}, lines
    assert lines["M4"]["windows"] == 5 and not ev["judgement"]["alerts"]
    assert W.breach(L.F, W.StatsReader()) is None and capsys.readouterr().out == ""
    # the check that finds the watch over with nothing crossed closes it: the state file holds that mark alone
    assert not (L.F / "control.json").exists() and list(W.load_state(L.F)) == ["closed"]


@pytest.mark.parametrize("change, words", [
    (dict(exact=False), "does not say"),
    (dict(float32={"change_max_abs": 0.0015, "token_rms": 1.0, "bit_identical": False}), "differs from the token"),
    (dict(autocast={"change_max_abs": 0.0, "token_rms": 1.0, "bit_identical": False}), "differs from the token"),
    (dict(output_weight_max_abs=0.02), "output layer was not zero"),
    (dict(rows=8), "probe rows, not 64"),
    (dict(version="motor_asked_change_v0"), "path version"),
])
def test_the_switch_record_is_judged_at_once(tmp_path, change, words):
    L = _lineage(tmp_path, switch=dict(SWITCH, **change))
    ev = _ev(L.F)
    assert ev["id"] == "M1:switch" and words in ev["why"], ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]


def test_on_the_graphics_card_the_check_must_have_run_under_autocast(tmp_path):
    no_autocast = {k: v for k, v in SWITCH.items() if k != "autocast"}
    assert "autocast" in _ev(_lineage(tmp_path / "card", switch=no_autocast).F)["why"]
    cpu = _lineage(tmp_path / "cpu", switch=no_autocast, args=dict(ARGS, device="cpu"))
    assert _ev(cpu.F)["why"] is None


def test_a_skipped_optimizer_update_in_the_trainers_log_stops(tmp_path, capsys):
    # the stats' nan_steps counts non-finite live losses only; a non-finite gradient norm or replay loss skips the
    # optimizer update and is written in the trainer's log alone
    skipped = "2026-10-03 13:00:00 ERROR non-finite grad norm at step 466123 - optimizer step skipped (#%d)\n"
    L = _lineage(tmp_path, after=_after(6))
    (L.F / "trainer.log").write_text("load done\nstep 466000 loss 1.0\n", encoding="utf-8")
    reader = W.StatsReader()
    assert W.breach(L.F, reader) is None
    with open(L.F / "trainer.log", "a", encoding="utf-8") as fh:
        fh.write(skipped % 1)
        fh.write("non-finite replay loss - optimizer step skip")           # a partly written last line
    why = W.breach(L.F, reader)
    assert why and "skipped" in why and "(#1)" in why, why
    assert W.load_state(L.F)["crossings"][0]["id"] == "M2:log:H84_switch#1"
    assert not bare_decimals(why) and not unanchored_numbers(why), why
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    assert W.breach(L.F, reader) is None                       # taken: the same line does not stop again
    # while the stopped run is still the newest run, a further line is a new crossing
    with open(L.F / "trainer.log", "a", encoding="utf-8") as fh:
        fh.write("ped (#2)\n")
    assert "2 line(s)" in W.breach(L.F, reader)
    # once a later run continues it, the stopped run's lines belong to the taken stop ...
    J = _run(L.exp, "H84_continued", parent="H84_switch", ck_step=_step(6), records=_after(9, first=7, partial=False))
    assert W.breach(J, W.StatsReader()) is None
    # ... and a line in the continuation's own log stops it
    (J / "trainer.log").write_text(skipped % 3, encoding="utf-8")
    assert "H84_continued holds 1 line(s)" in W.breach(J, W.StatsReader())
    printed = W.lines(J, W.StatsReader(), write=False)
    assert any("skipped optimizer update: 3 so far" in p for p in printed), printed[1]
    for p in printed:
        assert not bare_decimals(p) and not unanchored_numbers(p), p
    # a run without a trainer.log, and a log without such a line, stop nothing
    clean = _lineage(tmp_path / "clean", after=_after(6))
    assert W.breach(clean.F, W.StatsReader()) is None
    assert _ev(clean.F)["judgement"]["lines"]["M2"]["skipped_updates_in_the_logs"] == 0


def test_a_non_finite_update_stops(tmp_path):
    L = _lineage(tmp_path, after=_after(6, nan=lambda k: 1 if k == 5 else 0))
    ev = _ev(L.F)
    assert ev["id"] == "M2@%d" % _step(5) and "non-finite" in ev["why"]
    # a partial record counts too
    L2 = _lineage(tmp_path / "partial", after=[_mimo(465900, nf_steps=75, nan=2)])
    assert _ev(L2.F)["id"] == "M2@465900"


def test_the_paths_hidden_layer_under_its_live_share_in_two_reports_running_stops(tmp_path):
    one = _lineage(tmp_path / "one", after=_after(8, live=lambda k: 0.8 if k == 4 else 1.0))
    assert _ev(one.F)["why"] is None
    apart = _lineage(tmp_path / "apart", after=_after(8, live=lambda k: 0.8 if k in (3, 5) else 1.0))
    assert _ev(apart.F)["why"] is None
    two = _lineage(tmp_path / "two", after=_after(8, live=lambda k: 0.85 if k in (4, 5) else 1.0))
    ev = _ev(two.F)
    assert ev["id"] == "M3@%d" % _step(5) and "hidden layer" in ev["why"], ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]
    # read from the path's own gauge when the whole-model monitor carries no reading
    own = _lineage(tmp_path / "own", after=_after(8, live=lambda k: 0.85 if k in (4, 5) else 1.0, monitor=False))
    assert _ev(own.F)["id"] == "M3@%d" % _step(5)
    # a report without a reading between two low readings neither counts nor ends the count ...
    gaps = _after(8)
    for k, r in enumerate(gaps[1:], start=1):
        r["liveness"]["layers"].pop("motor_asked_change.net.0")
        if k in (3, 5):
            r["motor_asked_change"]["liveness"] = {"net.0": 0.2}
        elif k == 4:
            r["motor_asked_change"].pop("liveness")
    ev = _ev(_lineage(tmp_path / "gaps", after=gaps).F)
    assert ev["id"] == "M3@%d" % _step(5), ev["id"]
    # ... and one low reading among reports without a reading is not a stop
    lone = _after(8)
    for k, r in enumerate(lone[1:], start=1):
        r["liveness"]["layers"].pop("motor_asked_change.net.0")
        r["motor_asked_change"].pop("liveness")
        if k == 4:
            r["motor_asked_change"]["liveness"] = {"net.0": 0.2}
    ev = _ev(_lineage(tmp_path / "lone", after=lone).F)
    assert ev["why"] is None and ev["judgement"]["lines"]["M3"]["reports_without_a_reading"] == 7
    assert any("newest 3 full reports carry no live share" in w for w in ev["warnings"])
    # the lower of the two readings is judged: the monitor at 100% does not hide 30% on the commanded rows
    low = _after(8)
    for r in low[1:]:
        r["motor_asked_change"]["liveness"] = {"net.0": 0.3}
    ev = _ev(_lineage(tmp_path / "gauge", after=low).F)
    assert ev["id"] == "M3@%d" % _step(2) and "lower of" in ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]


def test_a_sense_past_its_line_in_reports_2_and_3_stops_and_one_report_does_not(tmp_path):
    line = BASE["audio"] * W.LINES["W1"]["audio"]

    def audio(*ks):
        return lambda k: dict(BASE, audio=1.05 * line) if k in ks else BASE
    L = _lineage(tmp_path / "both", after=_after(3, ratio=audio(2, 3)))
    ev = _ev(L.F)
    assert ev["id"] == "M4:reports2-3" and "audio" in ev["why"], ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]
    assert _ev(_lineage(tmp_path / "one", after=_after(3, ratio=audio(2))).F)["why"] is None
    assert _ev(_lineage(tmp_path / "first", after=_after(3, ratio=audio(1))).F)["why"] is None     # report 1: printed
    assert _ev(_lineage(tmp_path / "under", after=_after(3, ratio=lambda k: dict(BASE, audio=0.95 * line))).F
               )["why"] is None


def test_the_other_senses_are_judged_in_every_20_report_window_at_two_levels(tmp_path, capsys):
    # proprioception at 1.10 of its stop line from report 30: the window of reports 22-41 holds 12 such reports, so its
    # median is past the line: a stop at that window
    rise = lambda k: dict(BASE, proprioception=1.10 * _stop_line("proprioception")) if k >= 30 else BASE   # noqa: E731
    L = _lineage(tmp_path / "stop", after=_after(45, ratio=rise))
    ev = _ev(L.F)
    assert ev["id"] == "M4:w1" and "proprioception" in ev["why"] and ev["judgement"]["stop"]["step"] == _step(41)
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]
    # the same sense between its alert line and its stop line: an alert, printed once and recorded, never a stop
    mild = lambda k: dict(BASE, tactile=1.05 * _alert_line("tactile")) if k >= 2 else BASE                 # noqa: E731
    A = _lineage(tmp_path / "alert", after=_after(45, ratio=mild))
    assert W.breach(A.F, W.StatsReader()) is None
    out = capsys.readouterr().out
    assert out.count("asked-change path ALERT") == 2 and "tactile" in out, out
    held = W.load_state(A.F)["alerts"]
    assert [a["id"] for a in held] == ["M4-alert:w0", "M4-alert:w1"]
    assert not (A.F / "control.json").exists() and not (A.F / "stop_reason.json").exists()
    assert W.breach(A.F, W.StatsReader()) is None and "ALERT" not in capsys.readouterr().out
    for a in held:
        assert not bare_decimals(a["why"]) and not unanchored_numbers(a["why"]), a["why"]
    # vision has its own multiples: at 3 times its level it stops nothing and alerts nothing
    calm = lambda k: dict(BASE, vision=3.0 * BASE["vision"]) if k >= 2 else BASE                           # noqa: E731
    C = _lineage(tmp_path / "vision", after=_after(45, ratio=calm))
    ev = _ev(C.F)
    assert ev["why"] is None and not ev["judgement"]["alerts"]


def test_the_vestibular_sense_is_an_alert_only(tmp_path, capsys):
    up = lambda k: dict(BASE, vestibular=5.0 * BASE["vestibular"]) if k >= 2 else BASE                      # noqa: E731
    L = _lineage(tmp_path, after=_after(25, ratio=up))
    ev = _ev(L.F)
    assert ev["why"] is None, ev["why"]
    alerts = ev["judgement"]["alerts"]
    assert [a["id"] for a in alerts] == ["M4-alert:w0"] and alerts[0]["senses"] == ["vestibular"]
    assert "this sense has no stop line" in alerts[0]["why"]
    # the alert sits at the sense's largest rise on its own record since its revival x the vestibular watch's margin:
    # at 3 times its level nothing is printed
    assert W.LINES["alert_only_multiple"] == pytest.approx(W.LINES["alert_only_rise"] * V.LINES["stop_margin"],
                                                           abs=0.005)
    calm = lambda k: dict(BASE, vestibular=3.0 * BASE["vestibular"]) if k >= 2 else BASE                    # noqa: E731
    assert not _ev(_lineage(tmp_path / "calm", after=_after(25, ratio=calm)).F)["judgement"]["alerts"]
    # a switch whose 20 reports before reach back past the sense's revival (model step 455,800) gives it no level:
    # the sense is not judged there
    exp = tmp_path / "early" / "experiments"
    M = 457000
    _run(exp, "H84_before", parent="H84_grandparent", ck_step=454000,
         records=[_mimo(M - 100 * (29 - i), path=False) for i in range(30)], args=dict(ARGS, motor_asked_change="off"))
    F = _run(exp, "H84_switch", parent="H84_before", ck_step=M,
             records=[_switch(step=M)] + [_mimo(M + 100 * k, ratio=up(k)) for k in range(1, 26)])
    ev = _ev(F)
    assert ev["N"] == M and ev["judgement"]["before_reports"] == 30 and ev["why"] is None
    assert not ev["judgement"]["alerts"]


def test_the_level_before_the_switch_is_the_larger_of_the_20_and_the_100_report_median(tmp_path):
    # audio dips in the 20 reports before the switch: the recovery to its longer level is not a stop
    before = _before(100, ratio=lambda i: dict(BASE, audio=1.2 if i < 80 else 0.5))
    L = _lineage(tmp_path, before=before, after=_after(25), parent_start=455800)
    ev = _ev(L.F)
    assert ev["why"] is None and ev["judgement"]["before"]["level"]["audio"] == pytest.approx(1.2)
    assert ev["judgement"]["before"]["median_ratio"]["audio"] == pytest.approx(0.5)
    assert ev["judgement"]["before"]["level_reports"] == 100


def test_the_speed_line_names_the_first_windows_reference(tmp_path):
    first = W.lines(_lineage(tmp_path / "first", after=_after(25)).F, W.StatsReader(), write=False)
    later = W.lines(_lineage(tmp_path / "later", after=_after(45)).F, W.StatsReader(), write=False)
    assert any("speed:" in p and "in the 20 reports before the switch (under" in p for p in first)
    assert any("speed:" in p and "in the window before it (under" in p for p in later)


def test_the_step_rate_under_its_line_in_two_windows_running_stops(tmp_path):
    slow = lambda k: 0.42 if k <= 21 else 0.42 * 0.85 if k <= 41 else 0.42 * 0.85 * 0.85                  # noqa: E731
    L = _lineage(tmp_path / "two", after=_after(65, sps=slow))
    ev = _ev(L.F)
    assert ev["id"] == "M5:w2" and "model steps per second" in ev["why"], ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]
    once = lambda k: 0.42 if k <= 21 else 0.42 * 0.85                                                      # noqa: E731
    assert _ev(_lineage(tmp_path / "once", after=_after(65, sps=once)).F)["why"] is None
    # the first window is judged against the 20 reports before the switch
    first = _lineage(tmp_path / "first", after=_after(45, sps=lambda k: 0.42 * 0.85 if k <= 21 else 0.42 * 0.7))
    assert _ev(first.F)["id"] == "M5:w1"


def test_the_paths_output_above_its_growth_line_stops(tmp_path, capsys):
    # the line at report k (model step 465,900 + 100 k, 75 + 100 k model steps after the switch) counts 1.5 optimizer
    # updates per model step
    line = lambda k: W.growth_line(_step(k) - N, PER_STEP)                                                 # noqa: E731
    under = _lineage(tmp_path / "under", after=_after(6, output=lambda k: 0.95 * line(k)))
    assert _ev(under.F)["why"] is None
    over = _lineage(tmp_path / "over", after=_after(6, output=lambda k: 1.05 * line(k) if k == 4 else 0.3 * line(k)))
    ev = _ev(over.F)
    assert ev["id"] == "M6@%d" % _step(4) and "growth line" in ev["why"], ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]
    # counted at 1 optimizer update per model step (replay off) the same output is over the line sooner
    args = dict(ARGS, replay="off")
    assert _ev(_lineage(tmp_path / "noreplay", after=_after(6, output=lambda k: 0.95 * line(k)), args=args).F
               )["id"] == "M6@%d" % _step(1)
    # an output of one token unit under the line: an alert, printed once and recorded, never a stop
    big = _lineage(tmp_path / "unit", after=_after(12, output=lambda k: 1.2 if k >= 10 else 0.01))
    assert W.breach(big.F, W.StatsReader()) is None
    out = capsys.readouterr().out
    assert out.count("asked-change path ALERT") == 1 and "token unit" in out
    assert [a["id"] for a in W.load_state(big.F)["alerts"]] == ["M6-alert:unit"]
    assert W.breach(big.F, W.StatsReader()) is None and "ALERT" not in capsys.readouterr().out


def test_an_output_that_is_not_a_finite_number_stops(tmp_path):
    L = _lineage(tmp_path, after=_after(6, output=lambda k: float("nan") if k == 4 else 0.001))
    ev = _ev(L.F)
    assert ev["id"] == "M6@%d" % _step(4) and "not a finite number" in ev["why"], ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]


def test_an_unreadable_args_file_stops_nothing_by_itself_and_says_so(tmp_path, capsys):
    # at 80 of every 100 parts of the growth line counted at 1.5 optimizer updates per model step, the same output is
    # over a line counted at 1: with the run's args.json unreadable the rate is unknown and the line is not judged
    line = lambda k: W.growth_line(_step(k) - N, PER_STEP)                                                 # noqa: E731
    L = _lineage(tmp_path, after=_after(6, output=lambda k: 0.8 * line(k)))
    assert W.breach(L.F, W.StatsReader()) is None
    assert "could not be read" not in capsys.readouterr().out       # (the output's alert at one token unit prints)
    (L.F / "args.json").write_text("{", encoding="utf-8")
    assert W.breach(L.F, W.StatsReader()) is None
    assert "args.json could not be read" in capsys.readouterr().out
    printed = W.lines(L.F, W.StatsReader(), write=False)
    assert any("an unknown number of optimizer updates per model step" in p for p in printed)
    for p in printed:
        assert not bare_decimals(p) and not unanchored_numbers(p), p
    # the lines that need no rate are still judged
    L2 = _lineage(tmp_path / "nan", after=_after(6, nan=lambda k: 1 if k == 4 else 0))
    (L2.F / "args.json").write_text("{", encoding="utf-8")
    assert "non-finite" in W.breach(L2.F, W.StatsReader())
    L3 = _lineage(tmp_path / "out", after=_after(6, output=lambda k: float("inf") if k == 2 else 0.001))
    (L3.F / "args.json").write_text("{", encoding="utf-8")
    assert "not a finite number" in W.breach(L3.F, W.StatsReader())


def test_a_sense_at_zero_before_the_switch_is_named_and_the_others_are_judged(tmp_path, capsys):
    quiet = dict(BASE, audio=0.0)
    rise = lambda k: dict(quiet, proprioception=1.10 * _stop_line("proprioception")) if k >= 2 else quiet  # noqa: E731
    L = _lineage(tmp_path, before=_before(30, ratio=quiet), after=_after(25, ratio=rise))
    why = W.breach(L.F, W.StatsReader())
    out = capsys.readouterr().out
    assert why and "proprioception" in why and "M4:w0" == W.load_state(L.F)["crossings"][0]["id"]
    assert "audio: the reports before the switch hold no next-model-step error ratio above 0" in out
    printed = W.lines(L.F, W.StatsReader(), write=False)
    assert printed
    for p in printed:
        assert not bare_decimals(p) and not unanchored_numbers(p), p


def test_the_watch_ends_at_the_report_that_closes_the_last_window(tmp_path):
    # audio at 3 times its level from report 83; a relaunch left the record at 475,900 (report 100) partial, so no
    # full report at or past N + 10,000 exists yet: the watch is not over and its last window is not yet judged
    loud = dict(BASE, audio=3.0 * BASE["audio"])
    recs = _after(100, ratio=lambda k: loud if k >= 83 else BASE)
    recs[-1]["next_frame_norm"]["steps"] = 63
    ev = _ev(_lineage(tmp_path / "open", after=recs).F)
    assert not ev["watch_over"] and ev["why"] is None and ev["judgement"]["lines"]["M4"]["windows"] == 4
    assert ev["judgement"]["after_full_reports"] == 99
    # the next full reports arrive: the first one closes the watch, the 20 full reports ending at it are judged, and
    # the records after it are not
    more = recs + [_mimo(476000 + 100 * i, ratio=loud, nan=1 if i == 2 else 0) for i in range(3)]
    ev = _ev(_lineage(tmp_path / "closed", after=more).F)
    assert ev["watch_over"] and ev["id"] == "M4:end" and "audio" in ev["why"], (ev["id"], ev["why"])
    assert ev["judgement"]["after_full_reports"] == 100 and ev["judgement"]["newest_step"] == 476000
    one = [dict(type="mimo", step=N + 100, nf_steps=100)]
    assert W.watch_records(one, N) == (one, False)


def test_an_exception_inside_a_read_loses_no_record(tmp_path, monkeypatch, capsys):
    rise = lambda k: dict(BASE, audio=1.10 * _stop_line("audio")) if k >= 2 else BASE                      # noqa: E731
    L = _lineage(tmp_path, after=_after(25, ratio=rise))

    def failing_once(module):
        real, left = module.compact, [1]

        def once(r):
            if r.get("type") == "mimo" and r.get("step") == _step(3) and left[0]:
                left[0] -= 1
                raise MemoryError("planted, once")
            return real(r)
        return once
    monkeypatch.setattr(W, "compact", failing_once(W))
    reader = W.StatsReader()
    with pytest.raises(MemoryError):
        W.breach(L.F, reader)
    why = W.breach(L.F, reader)                       # the next 5-minute check with the same reader
    assert why and "audio" in why
    assert W.evaluate(L.F, reader, write=False)["judgement"]["after_full_reports"] == 25
    # through the hourly watcher's own program: the failed check says so and stops nothing, the next one stops
    L2 = _lineage(tmp_path / "hourly", after=_after(25, ratio=rise))
    hw = _hourly(monkeypatch, L2.F)
    rule = hw["_motor_asked_change_rule"]()
    monkeypatch.setattr(rule, "compact", failing_once(rule))
    capsys.readouterr()
    assert hw["motor_asked_change_breach"]() is None
    assert "ASKED-CHANGE PATH RULE FAILED" in capsys.readouterr().out
    why = hw["motor_asked_change_breach"]()
    assert why and "audio" in why


def test_a_transition_count_that_is_not_a_number_drops_its_cell_and_raises_nothing(tmp_path):
    recs = _after(8, nan=lambda k: 1 if k == 7 else 0)
    per = recs[3]["prediction_audit_hold_first"]["senses"]["proprioception"]["per_body"]
    per["size_matched_transitions"][4] = float("nan")
    per["size_matched_transitions"][5] = [5]
    L = _lineage(tmp_path, after=recs)
    reader = W.StatsReader()
    ev = W.evaluate(L.F, reader, write=False)
    assert ev["id"] == "M2@%d" % _step(7) and ev["judgement"]["after_full_reports"] == 8
    assert len(W.swap_cells(recs[3])) == 5 and len(W.swap_cells(recs[4])) == 7


@pytest.mark.parametrize("key, value", [
    ("liveness", "off"), ("tenure_pe", 0.0002), ("tenure_gap", "x"), ("contact_control", [1]),
    ("next_frame_norm", "off"), ("machine", "none"), ("causal_chain", 3), ("prediction_audit_hold_first", "x"),
    ("controllability", "x"), ("motor_asked_change", "on")])
def test_a_block_that_is_not_an_object_raises_nothing_and_leaves_no_hole(tmp_path, key, value):
    recs = _after(8, nan=lambda k: 1 if k == 7 else 0)
    recs[3][key] = value                              # report 3 (recs[0] is the partial record)
    L = _lineage(tmp_path, after=recs)
    reader = W.StatsReader()
    ev = W.evaluate(L.F, reader, write=False)
    assert ev["id"] == "M2@%d" % _step(7)
    assert ev["judgement"]["after_full_reports"] == (7 if key == "next_frame_norm" else 8)
    assert W.lines(L.F, reader, write=False)


def test_without_the_reports_before_the_switch_the_unconditional_lines_still_stop(tmp_path, capsys):
    line = lambda k: W.growth_line(_step(k) - N, PER_STEP)                                                 # noqa: E731
    L = _lineage(tmp_path / "out", before=_before(5), after=_after(4, output=lambda k: 2.0 * line(k)))
    why = W.breach(L.F, W.StatsReader())
    out = capsys.readouterr().out
    assert why and "growth line" in why
    assert "ASKED-CHANGE PATH STOP LINES INCOMPLETE" in out and "only 5 of the 20" in out
    for kw, cid in ((dict(nan=1), "M2@"), (dict(live=0.5), "M3@")):
        ev = _ev(_lineage(tmp_path / cid[:2], before=_before(5), after=_after(4, **kw)).F)
        assert ev["id"].startswith(cid), ev["id"]
    # a stream without the path's block says so
    bare = _lineage(tmp_path / "bare", after=_after(3, path=False))
    assert any("motor_asked_change block" in w for w in _ev(bare.F)["warnings"])
    assert any("live share" in w for w in _ev(bare.F)["warnings"])


# ------------------------------------------------------------------------------ crossings and the hourly watcher
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


def test_the_watcher_started_for_a_relaunched_run_judges_the_whole_timeline_and_stops_that_run(tmp_path, monkeypatch):
    rise = lambda k: dict(BASE, audio=1.10 * _stop_line("audio")) if k >= 2 else BASE                      # noqa: E731
    # the switched run ran to report 12, then the supervisor relaunched it: the relaunched run holds reports 13-25
    L = _lineage(tmp_path, after=_after(12, ratio=rise))
    G = _run(L.exp, "H84_cont", parent="H84_switch", ck_step=_step(12), records=_after(25, first=13, partial=False,
                                                                                         ratio=rise))
    assert _hourly(monkeypatch, L.F)["quick_breach"]() is None
    hg = _hourly(monkeypatch, G)
    why = hg["quick_breach"]()
    assert why and why.startswith("asked-change path:") and "audio" in why, why
    hg["stop_run"](why)
    assert json.loads((G / "control.json").read_text())["cmd"] == "snapshot_and_stop"
    reason = json.loads((G / "stop_reason.json").read_text())
    assert reason["resume_policy"] == "repair_required" and reason["reason"] == why
    assert not (L.F / "control.json").exists() and not (L.F / "stop_reason.json").exists()
    crossing = W.load_state(L.F)["crossings"][0]
    assert crossing["run"] == str(G) and crossing["id"] == "M4:w0"
    assert W.taken_crossings(L.F) == {"M4:w0": _step(21)}
    # a deliberate continuation is not stopped again on the same window ...
    J = _run(L.exp, "H84_after_review", parent="H84_cont", ck_step=_step(25),
             records=_after(40, first=26, partial=False))
    assert _hourly(monkeypatch, J)["quick_breach"]() is None
    # ... and the vestibular watch's state file was never written for this switch
    assert not (L.F / V.STATE_NAME).exists()


def test_a_stop_whose_file_a_resume_step_renamed_is_taken(tmp_path, monkeypatch):
    # the older resume step renames the stopped run's stop_reason.json to resolved_stop_reason_<time>.json: the
    # deliberate continuation must not be stopped again on the same crossing
    rise = lambda k: dict(BASE, audio=1.10 * _stop_line("audio")) if 2 <= k <= 21 else BASE                # noqa: E731
    L = _lineage(tmp_path, after=_after(25, ratio=rise))
    hg = _hourly(monkeypatch, L.F)
    why = hg["quick_breach"]()
    assert why and "audio" in why
    hg["stop_run"](why)
    (L.F / "stop_reason.json").replace(L.F / "resolved_stop_reason_1791000000.json")
    J = _run(L.exp, "H84_after_review", parent="H84_switch", ck_step=_step(25),
             records=_after(45, first=26, partial=False))
    assert W.breach(J, W.StatsReader()) is None
    assert W.taken_crossings(L.F, write=False) == {"M4:w0": _step(21)}
    # without either file the crossing is judged again (its stop never reached the run)
    (L.F / "resolved_stop_reason_1791000000.json").unlink()
    state = W.load_state(L.F)
    for c in state["crossings"]:
        c.pop("taken", None)
    (L.F / W.STATE_NAME).write_text(json.dumps(state), encoding="utf-8")
    assert "audio" in W.breach(J, W.StatsReader())


def test_a_taken_crossing_does_not_cover_model_steps_that_a_rollback_runs_again(tmp_path):
    # the switched run stops on a non-finite update at report 9 and saves there; the stop is taken
    L = _lineage(tmp_path, after=_after(9, nan=lambda k: 1 if k == 9 else 0))
    why = W.breach(L.F, W.StatsReader())
    assert why and W.load_state(L.F)["crossings"][0]["owner"] == "H84_switch"
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    # a deliberate continuation from the stop's own checkpoint is not stopped on that crossing ...
    J = _run(L.exp, "H84_continued", parent="H84_switch", ck_step=_step(9),
             records=_after(14, first=10, partial=False))
    assert W.breach(J, W.StatsReader()) is None
    # ... but a rollback to report 4 runs reports 5 to 9 again: a non-finite update in the new report 6 stops, and
    # so does one in the new report 9 itself
    H = _run(L.exp, "H84_rollback", parent="H84_switch", ck_step=_step(4),
             records=_after(8, first=5, partial=False, nan=lambda k: 1 if k == 6 else 0))
    assert _ev(H)["id"] == "M2@%d" % _step(6)
    H2 = _run(L.exp, "H84_rollback_2", parent="H84_switch", ck_step=_step(4),
              records=_after(9, first=5, partial=False, nan=lambda k: 1 if k == 9 else 0))
    assert _ev(H2)["id"] == "M2@%d" % _step(9)
    # a window's crossing likewise: the rollback's own reports of that window are judged afresh
    rise = lambda k: dict(BASE, audio=1.10 * _stop_line("audio")) if k >= 2 else BASE                      # noqa: E731
    L2 = _lineage(tmp_path / "window", after=_after(25, ratio=rise))
    why = W.breach(L2.F, W.StatsReader())
    assert W.load_state(L2.F)["crossings"][0]["id"] == "M4:w0"
    (L2.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    R = _run(L2.exp, "H84_rollback", parent="H84_switch", ck_step=_step(10),
             records=_after(25, first=11, partial=False, ratio=rise))
    assert _ev(R)["id"] == "M4:w0"
    clean = _run(L2.exp, "H84_rollback_clean", parent="H84_switch", ck_step=_step(1),
                 records=_after(25, first=2, partial=False))
    assert _ev(clean)["why"] is None


def test_the_records_a_stopped_run_wrote_before_it_exited_do_not_stop_its_continuation(tmp_path):
    # non-finite updates in reports 30 and 31: the stop is written on report 30, the trainer exits after report 31
    L = _lineage(tmp_path / "nan", after=_after(31, nan=lambda k: 1 if k in (30, 31) else 0))
    why = W.breach(L.F, W.StatsReader())
    assert why and W.load_state(L.F)["crossings"][0]["id"] == "M2@%d" % _step(30)
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    # while the stopped run is still the newest run, its next faulty record is reported (the stop did not end it)
    assert _ev(L.F)["id"] == "M2@%d" % _step(31)
    # the operator continues from the stop's checkpoint: the clean continuation is not stopped on report 31
    H = _run(L.exp, "H84_continued", parent="H84_switch", ck_step=_step(31), records=_after(36, first=32, partial=False))
    assert W.breach(H, W.StatsReader()) is None
    # ... and a non-finite update of its own stops it
    H2 = _run(L.exp, "H84_continued_2", parent="H84_switch", ck_step=_step(31),
              records=_after(36, first=32, partial=False, nan=lambda k: 1 if k == 34 else 0))
    assert _ev(H2)["id"] == "M2@%d" % _step(34)
    # the growth line likewise: the output above the line in reports 30 and 31 only
    line = lambda k: W.growth_line(_step(k) - N, PER_STEP)                                                 # noqa: E731
    L2 = _lineage(tmp_path / "out", after=_after(31, output=lambda k: 1.5 * line(k) if k in (30, 31) else 0.01))
    why = W.breach(L2.F, W.StatsReader())
    assert W.load_state(L2.F)["crossings"][0]["id"] == "M6@%d" % _step(30)
    (L2.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    J = _run(L2.exp, "H84_continued", parent="H84_switch", ck_step=_step(31),
             records=_after(36, first=32, partial=False, output=0.01))
    assert W.breach(J, W.StatsReader()) is None
    # the hidden layer: low in reports 29 to 32 (the stop on report 30, two more low reports before the exit)
    L3 = _lineage(tmp_path / "live", after=_after(32, live=lambda k: 0.5 if k >= 29 else 1.0))
    why = W.breach(L3.F, W.StatsReader())
    assert W.load_state(L3.F)["crossings"][0]["id"] == "M3@%d" % _step(30)
    (L3.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    K = _run(L3.exp, "H84_continued", parent="H84_switch", ck_step=_step(32), records=_after(37, first=33, partial=False))
    assert W.breach(K, W.StatsReader()) is None


def test_a_stop_written_into_a_run_that_was_already_continued_does_not_count(tmp_path):
    # a watcher left running on the ended run finds the crossing after the continuation has begun and writes its stop
    # into the ended run: no trainer is stopped, so the continuation's own watcher must still report the crossing
    rise = lambda k: dict(BASE, audio=1.10 * _stop_line("audio")) if k >= 2 else BASE                      # noqa: E731
    L = _lineage(tmp_path, after=_after(25, ratio=rise))
    H = _run(L.exp, "H84_continued", parent="H84_switch", ck_step=_step(25),
             records=_after(30, first=26, partial=False, ratio=rise))
    receipt = json.loads((H / "continuation_receipt.json").read_text(encoding="utf-8"))
    receipt["timestamp"] = time.time() - 600.0                      # the continuation began ten minutes ago
    (H / "continuation_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    why = W.breach(L.F, W.StatsReader())                            # the left-over watcher of the ended run
    assert why and W.load_state(L.F)["crossings"][0]["id"] == "M4:w0"
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    assert W.taken_crossings(L.F, write=False) == {"M4:w0": _step(21)}
    again = W.breach(H, W.StatsReader())
    assert again and "audio" in again and W.load_state(L.F)["crossings"][-1]["run"] == str(H)
    # the same stop written before the continuation began stands (the deliberate continuation is not stopped)
    receipt["timestamp"] = time.time() + 600.0
    (H / "continuation_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    state = W.load_state(L.F)
    state["crossings"] = state["crossings"][:1]
    (L.F / W.STATE_NAME).write_text(json.dumps(state), encoding="utf-8")
    assert W.breach(H, W.StatsReader()) is None


def test_the_last_window_is_judged_by_the_next_runs_watcher_and_a_closed_watch_is_left_alone(tmp_path, capsys):
    # audio over its stop line from report 82; the run stops (a relaunch) after the closing report, before its own
    # watcher's next check; the next run began more than 10,100 model steps after the switch
    loud = dict(BASE, audio=1.10 * _stop_line("audio"))
    L = _lineage(tmp_path / "late", after=_after(101, ratio=lambda k: loud if k >= 82 else BASE))
    G = _run(L.exp, "H84_next", parent="H84_switch", ck_step=_step(101), records=_after(104, first=102, partial=False))
    assert W.BASE.find_switch(G, W.StatsReader()) is None       # the vestibular watch's walk does not reach it
    why = W.breach(G, W.StatsReader())
    assert why and "audio" in why and W.load_state(L.F)["crossings"][0]["id"] == "M4:end"
    # a run that began past the slack is not watched
    far = _run(L.exp, "H84_far", parent="H84_next", ck_step=_step(125), records=_after(128, first=126, partial=False))
    assert _ev(far) is None
    # a clean watch judged to its end is closed: later checks judge nothing, and the sweep still prints its lines
    C = _lineage(tmp_path / "clean", after=_after(101))
    reader = W.StatsReader()
    assert W.breach(C.F, reader) is None and W.load_state(C.F)["closed"]["newest_step"] == _step(100)
    with open(C.F / "replay_training_stats.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_mimo(_step(102), nan=5)) + "\n")
    assert W.breach(C.F, reader) is None
    assert W.lines(C.F, W.StatsReader(), write=False)
    # before its end the watch is not closed
    early = _lineage(tmp_path / "early", after=_after(50))
    assert W.breach(early.F, W.StatsReader()) is None and "closed" not in W.load_state(early.F)


def test_a_stream_without_full_reports_is_named(tmp_path, capsys):
    # a run launched without --next-frame-norm writes no next_frame_norm block: no record is a full report
    recs = _after(6, live=0.3, output=5.0)
    for r in recs:
        r.pop("next_frame_norm")
    L = _lineage(tmp_path, after=recs)
    assert W.breach(L.F, W.StatsReader()) is None
    out = capsys.readouterr().out
    assert "7 stats records after the switch and none is a full report" in out
    # non-finite updates are still judged on such a stream
    recs[4]["nan_steps"] = 2
    L2 = _lineage(tmp_path / "nan", after=recs)
    assert "non-finite" in W.breach(L2.F, W.StatsReader())


def test_reports_whose_block_reads_rows_zero_are_named(tmp_path, capsys):
    recs = _after(6)
    for r in recs[1:]:
        r["motor_asked_change"] = {"version": "motor_asked_change_v1", "rows": 0, "output_weight_rms": 0.5}
    L = _lineage(tmp_path, after=recs)
    assert W.breach(L.F, W.StatsReader()) is None
    assert "the newest 3 full reports carry the path's block and no output size" in capsys.readouterr().out


def test_a_stop_that_never_reached_its_run_is_judged_again_and_a_taken_one_is_not(tmp_path):
    line = lambda k: W.growth_line(_step(k) - N, PER_STEP)                                                 # noqa: E731
    L = _lineage(tmp_path, after=_after(8, output=lambda k: 1.5 * line(k) if k in (3, 7) else 0.2 * line(k)))
    why = W.breach(L.F, W.StatsReader())
    assert why and W.taken_crossings(L.F) == {}
    assert W.breach(L.F, W.StatsReader()) == why
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")
    # taken: the same crossing is not reported again; the later one is
    ev = _ev(L.F)
    assert ev["id"] == "M6@%d" % _step(7), ev["id"]


def test_reading_by_hand_writes_nothing(tmp_path, capsys):
    rise = lambda k: dict(BASE, tactile=1.05 * _alert_line("tactile")) if k >= 2 else BASE                 # noqa: E731
    L = _lineage(tmp_path, after=_after(25, ratio=rise, nan=lambda k: 1 if k == 24 else 0))
    listing = sorted(p.name for p in L.F.iterdir())
    assert W.main([str(L.F)]) == 1
    out = capsys.readouterr().out
    assert "A STOP LINE IS CROSSED" in out and "switched on at model step 465,825" in out
    assert sorted(p.name for p in L.F.iterdir()) == listing


def test_the_hourly_watchers_own_program_writes_the_stop_into_the_relaunched_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: HEALTHY_DISK)
    monkeypatch.setenv("UEMA_WATCH_SAVED", str(tmp_path / "_saved"))
    monkeypatch.setenv("UEMA_WATCH_ARCHIVE", str(tmp_path / "_archive"))
    L = _lineage(tmp_path, after=_after(2))
    G = _run(L.exp, "H84_cont", parent="H84_switch", ck_step=_step(2),
             records=_after(8, first=3, partial=False, live=lambda k: 0.6 if k >= 6 else 1.0))
    monkeypatch.setattr(sys, "argv", [str(WATCHER), str(G), "1"])
    monkeypatch.setattr(time, "sleep", lambda _: None)
    runpy.run_path(str(WATCHER), run_name="__main__")
    out = capsys.readouterr().out
    assert "STOP RULE REACHED: asked-change path:" in out and "hidden layer" in out, out[-2000:]
    assert json.loads((G / "control.json").read_text())["cmd"] == "snapshot_and_stop"
    assert "hidden layer" in json.loads((G / "stop_reason.json").read_text())["reason"]


# ------------------------------------------------------------------------------ the pass lines and what a person reads
def test_line_s1_reads_as_declared(tmp_path):
    # the path changes nothing: the reading after the switch is the reference's: missed
    flat = _lineage(tmp_path / "flat", after=_after(101))
    acc = _ev(flat.F)["judgement"]["acceptance"]
    assert acc["reference"]["reports"] == 30 and acc["reference"]["transitions"] == 30 * 35
    assert acc["reference"]["steps"] == [462900, 465800]
    assert acc["S1"]["state"] == "missed" and acc["S1_late"]["state"] == "missed"
    assert acc["S1"]["model_steps"] == [N + 2001, N + 3000] and acc["S1"]["reports"] == 10
    assert acc["S1"]["reading"]["transitions"] == 350
    # the switch is off the hundreds, so the first window holds 9 full reports (the record at 465,900 is partial)
    assert len(acc["windows"]) == 10 and all(w["complete"] for w in acc["windows"])
    assert [w["reports"] for w in acc["windows"]] == [9] + [10] * 9
    # the main route uses the own command from report 15 on: met in the judged window
    used = _lineage(tmp_path / "used", after=_after(101, own_gain=lambda k: 0.02 if k >= 15 else 0.0))
    acc = _ev(used.F)["judgement"]["acceptance"]
    assert acc["S1"]["state"] == "met" and acc["S1"]["reading"]["value"] == pytest.approx(0.02)
    assert acc["S1"]["share_of_goal_reference"] == pytest.approx(0.25)
    assert acc["S1"]["smallest_rise_the_line_can_declare"] == pytest.approx(
        3.14 * acc["S1"]["se_difference"])
    # before the judged window is complete: waiting; reports before model step 457,825 never enter the reference
    early = _lineage(tmp_path / "early", after=_after(20))
    assert _ev(early.F)["judgement"]["acceptance"]["S1"]["state"] == "waiting"
    old = _lineage(tmp_path / "old", before=[_mimo(457000 + 100 * i, path=False) for i in range(30)],
                   after=_after(31, own_gain=0.02), parent_start=456900)
    acc = _ev(old.F)["judgement"]["acceptance"]
    assert acc["reference"]["steps"][0] == 457900 and acc["S1"]["state"] == "met"
    # a reference whose records carry no block: unmeasured, never met
    none = _lineage(tmp_path / "none", before=_before(30, hold_first=False), after=_after(31, own_gain=0.02))
    assert _ev(none.F)["judgement"]["acceptance"]["S1"]["state"] == "unmeasured"


def test_the_five_minute_check_does_not_compute_the_pass_lines(tmp_path):
    L = _lineage(tmp_path, after=_after(31))
    assert W.evaluate(L.F, W.StatsReader(), write=False, readings=False)["judgement"]["acceptance"] == {}
    assert W.evaluate(L.F, W.StatsReader(), write=False)["judgement"]["acceptance"]["S1"]["state"] == "missed"


def test_the_hourly_sweep_prints_the_lines_with_no_bare_decimal(tmp_path, monkeypatch):
    L = _lineage(tmp_path, after=_after(45, own_gain=lambda k: 0.004 if k >= 10 else 0.0))
    (L.F / W.READOUT_NAME).write_text(json.dumps({"checkpoint": "D:/x/ckpt_step_468900", "line": {
        "read": True, "above": True, "groups": {
            "asked_position_change_cm": {"r2": 0.81, "control_q975": -0.0003, "above_both_lines": True},
            "asked_turn_deg": {"r2": 0.79, "control_q975": 0.0002, "above_both_lines": True}}}}), encoding="utf-8")
    printed = _hourly(monkeypatch, L.F)["motor_asked_change_lines"]()
    assert printed[0].startswith("  asked-change path: switched on at model step 465,825")
    assert any("pass line S1" in p and "met" in p for p in printed), printed
    assert any("pass line S2" in p and "ckpt_step_468900" in p for p in printed)
    assert any("the first full reports after the switch" in p for p in printed)
    assert any("gate in force stood open" in p for p in printed)
    for line in printed:
        assert not bare_decimals(line) and not unanchored_numbers(line), line
    for banned in ("hearing", "body position", "moment", "balance"):
        assert not any(banned in p for p in printed), banned


# ------------------------------------------------------------------------------ the segment tool (reference, verdict)
SEGMENT = TREE / "docs" / "gain_of_function_v4" / "motor_asked_change_segment.py"


def _settings_line(run):
    printed = W.lines(run, W.StatsReader(), write=False)
    for line in printed:
        assert not bare_decimals(line) and not unanchored_numbers(line), line
    return next(line for line in printed if "launch settings around the switch" in line)


def test_the_launch_settings_around_the_switch_are_printed(tmp_path):
    # the switch's relaunch changed the flag alone, and no relaunch before it changed a setting
    alone = _settings_line(_lineage(tmp_path / "alone", after=_after(3)).F)
    assert "the switch's relaunch changed motor_asked_change and nothing else" in alone
    assert "no relaunch changed a setting" in alone
    # another setting changed at the switch's relaunch: a stop on a shared line cannot be put down to the path
    both = _settings_line(_lineage(tmp_path / "both", after=_after(3), args=dict(ARGS, device="cuda:1")).F)
    assert "changed device, motor_asked_change - MORE THAN THE PATH CHANGED AT THE SWITCH" in both
    # a setting the run before does not hold (a build that added a flag) is named too; at an off value it is named
    # apart and does not make the line say that more than the path changed
    added = _settings_line(_lineage(tmp_path / "added", after=_after(3), args=dict(ARGS, new_flag="on")).F)
    assert "changed motor_asked_change, new_flag - MORE THAN THE PATH" in added
    unset = _settings_line(_lineage(tmp_path / "unset", after=_after(3),
                                    args=dict(ARGS, new_flag="off", other_flag=None, a_count=0)).F)
    assert "changed motor_asked_change (and 3 setting(s) the run before does not hold, each at an off value: " \
           "a_count, new_flag, other_flag) and nothing else" in unset
    # a relaunch 12 full reports before the switch changed a setting: named, with its count, and marked as inside
    # the 20 reports the early line and the step rate's first window read
    exp = tmp_path / "earlier" / "experiments"
    held = dict(ARGS, motor_asked_change="off", self_percept_release="off")
    _run(exp, "H84_a", parent="H84_grandparent", ck_step=462700, records=_before()[:18], args=held)
    _run(exp, "H84_b", parent="H84_a", ck_step=464600, records=_before()[18:],
         args=dict(held, self_percept_release="on"))
    F = _run(exp, "H84_switch", parent="H84_b", ck_step=N, records=[_switch()] + _after(3),
             args=dict(ARGS, self_percept_release="on"))
    ev = _ev(F)
    assert ev["judgement"]["before_reports"] == 30
    ls = ev["judgement"]["readings"]["launch_settings"]
    assert ls["besides_the_flag"] == [] and [x["run"] for x in ls["within_the_20_before"]] == ["H84_b"]
    assert ls["unread"] == []
    assert ls["before"][0]["launch"]["changed"] == {"self_percept_release": ["off", "on"]}
    line = _settings_line(F)
    assert "changed motor_asked_change and nothing else" in line
    assert ("the relaunch at model step 464,600 changed self_percept_release, with 12 full reports between it and "
            "the switch (INSIDE THE 20 REPORTS") in line
    # the 5-minute check reads no args.json but the switched run's
    assert "launch_settings" not in W.evaluate(F, W.StatsReader(), write=False, readings=False)["judgement"]["readings"]
    # before a switch: the same reading on the run a switch would resume
    S = _load(SEGMENT, "motor_asked_change_segment")
    doc = S.before_a_switch(exp / "H84_b")
    assert doc["newest_full_report"] == 465800 and doc["full_reports"] == 30 and doc["first_step"] == 462900
    assert [x["run"] for x in doc["inside_the_20_before"]] == ["H84_b"]
    assert doc["inside_the_20_before"][0]["full_reports_after"] == 12
    assert not doc["the_20_reports_before_hold_no_settings_change"]
    assert S.main(["before", "--run", str(exp / "H84_b")]) == 2
    quiet = S.before_a_switch(exp / "H84_a")
    assert quiet["full_reports"] == 18 and quiet["the_20_reports_before_hold_no_settings_change"]
    assert S.main(["before", "--run", str(exp / "H84_a")]) == 0
    # a run just relaunched with a changed setting, before its first full report: named, with 0 reports after it
    fresh = _run(exp, "H84_c", parent="H84_b", ck_step=465800, records=[_mimo(465900, nf_steps=75, path=False)],
                 args=dict(held, self_percept_release="on", replay_every=4))
    doc = S.before_a_switch(fresh)
    assert doc["newest_full_report"] == 465800 and doc["full_reports"] == 30
    assert [(x["run"], x["full_reports_after"]) for x in doc["inside_the_20_before"]] == [("H84_c", 0), ("H84_b", 12)]
    assert doc["relaunches_that_changed_a_setting"][0]["launch"]["changed"] == {"replay_every": [2, 4]}
    # a relaunch whose settings cannot be read is not counted as clean
    (exp / "H84_c" / "args.json").write_text("{", encoding="utf-8")
    doc = S.before_a_switch(fresh)
    assert [x["run"] for x in doc["relaunches_whose_settings_could_not_be_read"]] == ["H84_c"]
    assert not doc["the_20_reports_before_hold_no_settings_change"]
    # a switched run stopped at its launch: the run that resumed it reads the switch's own relaunch as the deploy's
    stopped = tmp_path / "stopped" / "experiments"
    _run(stopped, "H84_before", parent="H84_grandparent", ck_step=462700, records=_before(),
         args=dict(ARGS, motor_asked_change="off"))
    _run(stopped, "H84_switch", parent="H84_before", ck_step=N, records=[_switch()], args=dict(ARGS, device="cuda:1"))
    G = _run(stopped, "H84_switch_resume", parent="H84_switch", ck_step=N, records=_after(3))
    line = _settings_line(G)
    assert "changed device, motor_asked_change - MORE THAN THE PATH CHANGED AT THE SWITCH" in line
    # ... and its own relaunch, inside the watch, changed a setting back: named too
    assert "after the switch: the relaunch at model step 465,825 changed device - A SETTING CHANGED INSIDE THE WATCH" \
        in line
    assert "after the switch: no relaunch changed a setting" in alone


def test_the_reference_reads_the_lineages_full_reports_above_model_step_457825(tmp_path):
    S = _load(SEGMENT, "motor_asked_change_segment")
    exp = tmp_path / "experiments"
    # two run folders of the lineage: the second re-ran model step 458,100 (its record is the newer one and is kept);
    # a partial record and a record at or below 457,825 never enter; another lineage's folder is not read
    a = [_mimo(457800, path=False), _mimo(457900, nf_steps=75, path=False)] + \
        [_mimo(458000 + 100 * i, path=False, own_gain=0.5) for i in range(2)]
    b = [_mimo(458100 + 100 * i, path=False) for i in range(24)]
    for r in b:
        r["timestamp"] += 1000.0
    _run(exp, S.PREFIX + "41x", records=a)
    _run(exp, S.PREFIX + "41y", records=b)
    _run(exp, "H99_other", records=[_mimo(458000 + 100 * i, path=False, own_gain=0.9) for i in range(30)])
    rows = S.lineage_reports(exp)
    assert [r["step"] for r in rows] == [458000 + 100 * i for i in range(25)]
    assert rows[0]["run"].endswith("41x") and rows[1]["run"].endswith("41y")
    ref = S.reference_reading(rows)
    assert ref["read"] and ref["reports"] == 25 and ref["transitions"] == 25 * 35 and ref["steps"] == [458000, 460400]
    # one report at a reading of 0.5 among 25 at 0: the pooled reading is the cells' own arithmetic
    assert ref["value"] == pytest.approx(W.own_command_reading([c for r in rows for c in r["s1"]])["value"])
    assert ref["ten_report_windows"]["count"] == 16
    assert ref["smallest_rise_the_line_can_declare"] == pytest.approx(
        3.14 * (ref["se"] ** 2 + ref["ten_report_windows"]["se_median"] ** 2) ** 0.5)
    assert S.reference_reading([])["read"] is False


def test_the_reference_command_also_reads_the_reports_a_switch_now_would_hold(tmp_path):
    S = _load(SEGMENT, "motor_asked_change_segment")
    exp = tmp_path / "experiments"
    # 130 full reports above model step 457,825; the first 30 read 0.5, the newest 100 read 0: the watch program
    # builds the reference from the up to 100 full reports before a switch, so a switch now would meet 0
    records = [_mimo(458000 + 100 * i, path=False, own_gain=0.5 if i < 30 else 0.0) for i in range(130)]
    _run(exp, S.PREFIX + "41x", records=records)
    out = tmp_path / "reference.json"
    assert S.main(["reference", "--experiments", str(exp), "--output", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    every, now = doc["reference"], doc["at_a_switch_now"]
    assert every["reports"] == 130 and every["steps"] == [458000, 470900] and every["value"] > 0.1
    assert W.LINES["level_reports"] == 100
    assert now["reports"] == 100 and now["steps"] == [461000, 470900] and now["transitions"] == 100 * 35
    rows = S.lineage_reports(exp)
    assert now["value"] == pytest.approx(W.own_command_reading([c for r in rows[-100:] for c in r["s1"]])["value"])
    assert abs(now["value"]) < 1e-9
    # the watch program's own reference for a switch right after the newest of those reports is the same reading
    F = _run(exp, S.PREFIX + "41y", parent=S.PREFIX + "41x", ck_step=470900,
             records=[_switch(step=470900, switch=dict(SWITCH, at_step=470900, checkpoint="ckpt_step_470900"))] +
                     [_mimo(471000 + 100 * i) for i in range(3)])
    ref = W.evaluate(F, W.StatsReader(), write=False)["judgement"]["acceptance"]["reference"]
    assert ref["reports"] == 100 and ref["steps"] == now["steps"]
    assert ref["transitions"] == now["transitions"] and ref["value"] == pytest.approx(now["value"])


def test_the_readout_command_is_the_branchs_readout_job():
    S = _load(SEGMENT, "motor_asked_change_segment")
    cmd = S.readout_command("on", "D:/ck/ckpt_step_468900", "D:/run/args.json", "D:/t.npz", "D:/out/readout_on",
                            threads=2, python="py")
    assert cmd == ["py", "-u", str(S.MEASURE), "--token-only", "--asked-change", "on", "--checkpoint",
                   "D:/ck/ckpt_step_468900", "--args", "D:/run/args.json", "--transitions", "D:/t.npz", "--out",
                   "D:/out/readout_on", "--threads", "2"]
    assert S.MEASURE.exists() and S.TRIAL.exists()
    # the trial driver builds the same command for an arm's final checkpoint
    src = S.TRIAL.read_text(encoding="utf-8")
    assert '"--token-only", "--asked-change", arm,' in src and '"--threads", str(int(a.threads))]' in src


def test_the_segments_verdict_names_the_outcome(tmp_path):
    S = _load(SEGMENT, "motor_asked_change_segment")

    def outcome(name, **kw):
        L = _lineage(tmp_path / name, **kw)
        return S.verdict_of(W.evaluate(L.F, W.StatsReader(), write=False))
    v = outcome("met", after=_after(101, own_gain=lambda k: 0.02 if k >= 15 else 0.0))
    assert v["outcome"].startswith("S1 met over model steps N + 2,001") and v["stop"] is None and v["watch_over"]
    v = outcome("late", after=_after(101, own_gain=lambda k: 0.02 if k >= 60 else 0.0))
    assert "met over N + 9,001" in v["outcome"] and v["S1"]["state"] == "missed" and v["S1_late"]["state"] == "met"
    v = outcome("never", after=_after(101))
    assert v["outcome"].startswith("S1 not met through 10,000 model steps")
    v = outcome("early", after=_after(12))
    assert v["outcome"].startswith("no verdict yet")
    v = outcome("stopped", after=_after(12, nan=lambda k: 1 if k == 9 else 0))
    assert v["outcome"].startswith("a stop line was crossed: M2") and v["stop"]["id"] == "M2@%d" % _step(9)
