# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The vestibular encoder repair's kill lines (docs/gain_of_function_v4/vestibular_repair_watch.py), judged by the H84
hourly watcher (docs/h84_launch_2026-09-18/hourly_watch.py): both reviews' findings on the lines, and the relaunch
case.

The lineage before the switch is H84's own: the next-moment ratios and step rates of the full reports of runs
..._seg40_resume_20260925_093334 and ..._112146 (read from their stats on 2026-09-25), in run folders linked the
way the supervisor links them (continuation_receipt.json). The deploy is put at checkpoint 411,125, so the 20
reports before it span both folders. What is held here:
  - BEFORE is the 20 full reports before the switch across run folders, and their medians are the lineage's own;
  - the lines on the other senses take the cost of the state each report is in (the context forms, over several
    saved moments), never above the sense's own base; report 1 is printed and never judged, reports 2 and 3 are
    judged together; the lines are judged in every 20-report window through the whole watch (review 2, finding 1):
    K3 against the window before it, two windows running, granting only the allowance's growth; K6 against the
    level before the switch (the larger of the 20-report and the 100-report median before it) at two levels, an
    alert (printed once and recorded, never a stop) and a stop (the fourth review's fix round), which stop the
    fourth review's fast collapse (hearing 3, 4 or 5 times its level from reports 2 to 62) and the third review's
    creep at its own rates within 2 windows; the step rate on two windows running, against the level before the
    switch a reading;
  - the lines hold their measured chance of stopping a watch with nothing wrong, and of an alert with nothing wrong
    (at most 5 in 100 on both stretches in sample, and out of sample: lines drawn from one half of the lineage and
    judged on the other), read from the committed measurement with the plants' windows, and the measuring tool
    counts what judge() stops;
  - a cost read on one saved moment, at another step, or from another mode's arm is refused, and the card's result
    is read before the CPU fallback's; the switch, a dead token and the encoder are judged whatever the reports
    before the switch and the cost hold, and every 5-minute check says loudly what cannot be judged (review 2,
    finding 3);
  - the lesson weight reading applies the Sept 23 lane's rule (third review, section D);
  - which runs are watched: every continuation after the switch and a relaunch that switched again, never the
    revert; a crashed branch's later reports are off the timeline; a parent named by its archive folder is read
    from the folder that holds its stats;
  - THE RELAUNCH CASE: the supervisor ends a watcher and starts one only by the file name hourly_watch.py (real
    processes); the watcher it starts for a relaunched run judges the whole timeline and writes the stop into that
    run only; a taken stop suppresses only its own crossing: a deliberate continuation is not stopped again on it,
    and is stopped by a new window's or a new record's crossing (review 2, finding 5); a stop that never reached its
    run is judged again; one end-to-end run of the hourly watcher's own program;
  - reading by hand writes nothing, a recorded crossing whose stop was taken included (review 2, finding 6);
  - the acceptance lines (A2 on the median of its band; A3 against the largest rise the lineage showed, so hearing at
    2.2 times its level reads missed), and the printed lines carry no bare score decimal (DESIGN.md section 11).
"""
import collections
import importlib.util
import json
import runpy
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, unanchored_numbers

HERE = Path(__file__).resolve()
TREE = HERE.parents[4]
SCRIPT = TREE / "docs" / "gain_of_function_v4" / "vestibular_repair_watch.py"
WATCHER = TREE / "docs" / "h84_launch_2026-09-18" / "hourly_watch.py"


def _load():
    spec = importlib.util.spec_from_file_location("vestibular_repair_watch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load()
SENSES = ("vision", "tactile", "audio", "proprioception")


def test_the_senses_go_by_the_registry_s_names_with_the_explainer_s_meanings():
    """The watcher's printed lines name each sense as the model registry does, and the plain meaning it prints
    beside a name is the opening of the explainer's own definition (mimo_bridge/explainer/schema.py
    SENSE_DEFINITIONS), so the two cannot drift apart. Until 2026-10-03 the lines said "touch", "hearing", "body
    position" and "balance"."""
    from mimo_bridge.explainer.schema import SENSE_DEFINITIONS
    assert W.SENSE_WORDS == {s: s for s in SENSES}
    assert set(W.SENSE_MEANS) == set(SENSES) | {"vestibular"}
    for sense, meaning in W.SENSE_MEANS.items():
        assert SENSE_DEFINITIONS[sense].startswith(meaning), (sense, meaning)
    assert W._named("audio") == "audio (what the two ears hear)"

# H84's own full and partial reports: (step, next-moment window steps, vision, touch, audio, proprioception ratios,
# model steps per second); the vestibular collapse watch read 0.0 and nan_steps 0 in every one
RUN_D = "H84_fresh_drivers_20260918_seg40_resume_20260925_093334"        # resumed 408,450
REAL_D = [(408500, 50, 3.4051, 0.94475, 0.55655, 1.2526, 0.5068), (408600, 100, 2.1479, 1.0133, 0.84807, 0.785, 0.52144),
          (408700, 100, 2.563, 0.98036, 0.91194, 0.85769, 0.51409), (408800, 100, 1.8526, 0.84427, 1.0173, 1.1008, 0.50979),
          (408900, 100, 2.396, 0.88095, 0.6818, 0.97675, 0.5051), (409000, 100, 2.0391, 0.83668, 0.85695, 0.95851, 0.50162),
          (409100, 100, 2.3775, 0.78553, 0.8785, 0.89698, 0.51715), (409200, 100, 2.9262, 0.74082, 0.84511, 0.83733, 0.51122),
          (409300, 100, 2.7042, 0.85406, 0.58898, 0.76527, 0.52143), (409400, 100, 1.1848, 0.82942, 1.0292, 1.182, 0.52655),
          (409500, 100, 1.2566, 0.93021, 0.85501, 0.75652, 0.51676), (409600, 100, 1.6782, 0.90079, 0.86514, 0.86385, 0.51503),
          (409700, 100, 2.736, 0.88494, 0.77463, 0.92732, 0.50548), (409800, 100, 1.8928, 0.85564, 0.94536, 1.1406, 0.50486),
          (409900, 100, 1.5314, 0.78436, 0.92455, 1.2767, 0.52187), (410000, 100, 1.7755, 0.82829, 0.85681, 1.2057, 0.52681),
          (410100, 100, 1.4498, 0.75153, 0.9823, 1.2434, 0.48482), (410200, 100, 1.6706, 0.88818, 0.61151, 1.1757, 0.48073),
          (410300, 100, 1.3476, 0.81938, 0.66284, 1.3148, 0.48564), (410400, 100, 1.7496, 0.94881, 1.0076, 1.3425, 0.48744)]
RUN_E = "H84_fresh_drivers_20260918_seg40_resume_20260925_112146"        # resumed 410,400
REAL_E = [(410500, 100, 1.7674, 0.809, 0.98492, 1.0861, 0.48722), (410600, 100, 1.5115, 0.95154, 0.39406, 0.99375, 0.49585),
          (410700, 100, 2.3099, 0.95357, 0.83307, 1.255, 0.50635), (410800, 100, 1.28, 0.89949, 1.2152, 1.0648, 0.50748),
          (410900, 100, 1.4949, 0.704, 1.0663, 1.1739, 0.49587), (411000, 100, 1.6052, 0.72096, 0.95315, 1.0826, 0.49757),
          (411100, 100, 3.0063, 0.85084, 0.90201, 1.1226, 0.50564), (411125, 25, 1.7665, 0.87341, 1.2316, 1.1482, 0.4663)]
N = 411125
# the medians over the 20 full reports 409,200-411,100 (scratchpad vfix/drift.py on the whole lineage, 2026-09-25)
BASE = {"vision": 1.6744, "tactile": 0.8525, "audio": 0.8836, "proprioception": 1.1316}
SPS_BASE = 0.5056
# the declared cost: the context forms' rise (later_reports, what the lines take) and the newest-moment form's
# (first_report, printed only), read on 3 saved moments at a revived spread of 0.30 between the bodies
COST = {"step": N, "checkpoint": "ckpt_step_411125", "revived_spread": 0.30, "moments": 3,
        "first_report": {"vision": 0.0006, "tactile": 0.0, "audio": 0.18, "proprioception": 0.0019},
        "later_reports": {"vision": 0.0006, "tactile": 0.0, "audio": 0.05, "proprioception": 0.0019},
        "mode": "rescale_restart", "arm": "CORE_bf16", "device": "cuda:0"}
SWITCH = {"switched": True, "divided_by": 4096.0, "bias_spread_after": 18.17, "mode": "rescale_restart",
          "checks": {"float64": {"within": True}, "float32": {"within": True}, "bfloat16": {"within": True}},
          "runtime_restart": {"next_frame_norm": {"dropped_readings": 66723},
                              "learning_progress": {"held_out_steps": 2000}}}
W1, W20 = W.LINES["W1"], W.LINES["W20"]
DEAD = W.LINES["K2_line"] / 10.0              # a collapse watch reading under the dead-token line
ALIVE = max(0.3, 3.0 * W.LINES["A2_line"])    # one over both lines


def _level():
    """K6's level before the switch on the lineage fixture: the larger of the 20-report and the 26-report median."""
    rows = [W.compact(r) for r in _real(REAL_D) + _real(REAL_E)]
    return W.level_before([r for r in rows if W.is_full(r)])[1]


def _line(sense, k=None, allowance=0.05, base=None):
    """K6's stop line (every window alike) with the declared cost's allowance."""
    return ((_level()[sense] if base is None else base) + allowance) * W.k6_levels(sense)[1]


def _alert_line(sense, allowance=0.05, base=None):
    return ((_level()[sense] if base is None else base) + allowance) * W.k6_levels(sense)[0]
HEALTHY_DISK = collections.namedtuple("usage", "total used free")(1e12, 0.0, 500e9)


def _mimo(step, nf_steps=100, ratio=None, sps=0.5, tts=None, live=1.0, nan=0, capped=0, between=None, probe=None,
          improvement=None, share=0.2, replay_share=0.2, lesson_weight=1.0):
    ratio = dict(BASE if ratio is None else ratio)
    others = {s: (1.0 - share) / 4 for s in SENSES}
    return {"type": "mimo", "step": step, "timestamp": 1.0e9 + step, "elapsed_s": 600.0,
            "next_frame_norm": {"steps": nf_steps, "ratio": dict(ratio, vestibular=1.0),
                                "capped_steps": ({"vestibular": capped} if capped else {}),
                                "share_normalized": dict(others, vestibular=share),
                                "replay_share_normalized": dict(others, vestibular=replay_share)},
            "token_temporal_std": {"vestibular": ALIVE if tts is None else tts},
            "vestibular": {"liveness": {"encoder_registry.encoders.vestibular.mlp.0": live,
                                        "encoder_registry.encoders.vestibular.mlp.4": 1.0},
                           "improvement_over_persistence": improvement,
                           "repair": {"token_between_bodies": between, "balance_probe": probe,
                                      "lesson_weight": lesson_weight,
                                      "revival_ceiling": {"live": {"steps_applied": 100, "steps_held_down": 40,
                                                                   "factor_mean_held": 0.3}}}},
            "nan_steps": nan, "model_steps_per_s": sps,
            "learning_progress": {"value_mean": 0.0, "per_sense": {"vestibular": 0.0}}}


def _real(rows):
    return [_mimo(s, n, dict(zip(SENSES, v[:4])), sps=v[4], tts=0.0) for s, n, *v in rows]


def _run(exp, name, parent=None, ck_step=None, records=(), launched=None, parent_path=None):
    run = exp / name
    run.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps({"report_every": 100}), encoding="utf-8")
    if parent is not None:
        pp = Path(parent_path) if parent_path else exp / parent
        (run / "continuation_receipt.json").write_text(json.dumps({
            "parent_run": str(pp), "checkpoint": str(pp / "checkpoints" / ("ckpt_step_%d" % ck_step)),
            "timestamp": time.time() if launched is None else launched, "reason": "test"}), encoding="utf-8")
    with open(run / "replay_training_stats.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return run


def _switch(step=N, switch=None):
    return {"type": "vestibular_encoder_repair", "step": step, "timestamp": 1.0, "switch": switch or SWITCH}


def _lineage(tmp_path, after=(), switch=None, cost=None, e_parent_path=None):
    exp = tmp_path / "experiments"
    D = _run(exp, RUN_D, parent="H84_test_parent_of_D", ck_step=408450, records=_real(REAL_D))
    E = _run(exp, RUN_E, parent=RUN_D, ck_step=410400, records=_real(REAL_E), parent_path=e_parent_path)
    F = _run(exp, "H84_deploy", parent=RUN_E, ck_step=N, records=[_switch(switch=switch)] + list(after))
    (F / W.COST_NAME).write_text(json.dumps({"verdicts": {"expected_rise": COST if cost is None else cost}}),
                                 encoding="utf-8")
    return SimpleNamespace(exp=exp, D=D, E=E, F=F)


def _step(k):
    """The step of full report k after the switch at 411,125 (the first record, 411,200, is partial)."""
    return 411200 + 100 * k


def _after(n, first=1, partial=True, **kw):
    """(A partial record at 411,200, then) full reports first..n with the given fields (callables of k)."""
    out = [_mimo(411200, nf_steps=75)] if partial else []
    for k in range(first, n + 1):
        out.append(_mimo(_step(k), **{key: (v(k) if callable(v) else v) for key, v in kw.items()}))
    return out


def _ev(run, write=True):
    return W.evaluate(run, W.StatsReader(), write=write)


def _audio(first=None, later=None):
    """Hearing's ratio per report: `first` in report 2, `later` from report 3 (report 1 at its base)."""
    return lambda k: dict(BASE, audio=(BASE["audio"] if k == 1 else first if k == 2 else later))


# ------------------------------------------------------------------------------ the timeline
def test_the_rule_module_is_this_trees():
    assert Path(W.__file__).resolve() == SCRIPT.resolve()
    assert W.ROOT == TREE


def test_before_is_the_20_full_reports_before_the_switch_across_run_folders(tmp_path):
    L = _lineage(tmp_path)
    before = W.before_reports(W.StatsReader(), L.F, N)
    assert [r["step"] for r in before] == list(range(409200, 411101, 100))
    assert {r["run"] for r in before} == {RUN_D, RUN_E}
    for s in SENSES:
        assert statistics.median(r["ratio"][s] for r in before) == pytest.approx(BASE[s], abs=5e-5)
    assert statistics.median(r["sps"] for r in before) == pytest.approx(SPS_BASE, abs=5e-5)


def test_a_parent_named_by_its_archive_folder_is_read_from_the_folder_holding_its_stats(tmp_path):
    # review 2, finding 3: a resume path can name the archive folder on D:, which holds checkpoints and no stats
    archive = tmp_path / "archive" / RUN_D
    (archive / "checkpoints").mkdir(parents=True)
    L = _lineage(tmp_path, e_parent_path=archive)
    assert W.run_folder(archive, L.exp) == L.D
    assert len(W.before_reports(W.StatsReader(), L.F, N)) == 20


def test_which_runs_are_watched(tmp_path):
    L = _lineage(tmp_path, after=_after(3))
    G = _run(L.exp, "H84_deploy_cont", parent="H84_deploy", ck_step=N + 1000, records=_after(15, first=11,
                                                                                          partial=False))
    I = _run(L.exp, "H84_redeploy", parent="H84_deploy", ck_step=N, records=[_switch()])
    H = _run(L.exp, "H84_revert", parent="H84_deploy", ck_step=N, records=_after(30, tts=0.0))
    J = _run(L.exp, "H84_revert_cont", parent="H84_revert", ck_step=N + 900, records=[])
    reader = W.StatsReader()
    assert W.find_switch(L.F, reader)[0] == L.F
    deploy, sw, chain = W.find_switch(G, reader)
    assert deploy == L.F and [r.name for r in chain] == ["H84_deploy", "H84_deploy_cont"]
    assert W.find_switch(I, reader)[0] == I and [r.name for r in W.find_switch(I, reader)[2]] == ["H84_redeploy"]
    assert W.find_switch(H, reader) is None and W.find_switch(J, reader) is None
    assert W.find_switch(L.E, reader) is None and W.breach(H, W.StatsReader()) is None
    L2 = _lineage(tmp_path / "branch", after=_after(30, ratio=lambda k: dict(BASE, audio=9.0) if k > 27 else BASE))
    G2 = _run(L2.exp, "H84_deploy_cont", parent="H84_deploy", ck_step=_step(27),
              records=_after(40, first=28, partial=False))
    after = W.after_records(reader, W.find_switch(G2, reader)[2], N)
    steps = [r["step"] for r in after if W.is_full(r)]
    assert steps == sorted(set(steps)) and steps[-1] == _step(40) and len(steps) == 40
    assert all(r["ratio"]["audio"] != 9.0 for r in after)


# ------------------------------------------------------------------------------ the cost and the lines
def test_the_cost_is_the_context_forms_read_on_several_moments_at_the_switch_step(tmp_path):
    cost = W.load_cost(_lineage(tmp_path / "c").F / W.COST_NAME, N)
    assert cost["R"]["audio"] == 0.05 and cost["R_newest"]["audio"] == 0.18 and cost["moments"] == 3
    with pytest.raises(ValueError, match="another checkpoint"):
        W.load_cost(_lineage(tmp_path / "x", cost=dict(COST, step=411150)).F / W.COST_NAME, N)
    with pytest.raises(ValueError, match="1 saved model step"):
        W.load_cost(_lineage(tmp_path / "m", cost=dict(COST, moments=1)).F / W.COST_NAME, N)
    with pytest.raises(ValueError, match="no rise for tactile, audio, proprioception"):
        W.load_cost(_lineage(tmp_path / "y", cost=dict(COST, later_reports={"vision": 0.0})).F / W.COST_NAME, N)
    L = _lineage(tmp_path / "pause")
    (L.F / W.COST_NAME).unlink()
    assert W.cost_path_for(L.F) in (W.PAUSE_RESULT_CARD, W.PAUSE_RESULT)


def test_the_cost_comes_from_the_arm_of_the_mode_the_switch_ran(tmp_path, monkeypatch):
    # third review, section E: the cost was read from the float32 anti-collapse arm, the lesson that in bf16 moves
    # nothing; now from the arm of the mode MODE chose, and refused for a switch that ran another mode
    path = _lineage(tmp_path / "c").F / W.COST_NAME
    cost = W.load_cost(path, N, mode="rescale_restart")
    assert cost["arm"] == "CORE_bf16" and cost["mode"] == "rescale_restart"
    with pytest.raises(ValueError, match="mode rescale_restart .*switch ran mode rescale_restart_fp32"):
        W.load_cost(path, N, mode="rescale_restart_fp32")
    old = _lineage(tmp_path / "old", cost={k: v for k, v in COST.items() if k not in ("mode", "arm")})
    ev = _ev(old.F)
    assert ev["cost"] is None and "another revival" in ev["cost_refused"], ev["cost_refused"]
    # the graphics card's result before the CPU's (the CPU job is the fallback)
    folder = tmp_path / "pause_folder"
    folder.mkdir()
    card, cpu = folder / "card.json", folder / "cpu.json"
    monkeypatch.setattr(W, "PAUSE_RESULT_CARD", card)
    monkeypatch.setattr(W, "PAUSE_RESULT", cpu)
    L = _lineage(tmp_path / "p")
    (L.F / W.COST_NAME).unlink()
    assert W.cost_path_for(L.F) == cpu
    card.write_text("{}", encoding="utf-8")
    assert W.cost_path_for(L.F) == card
    cpu.write_text(json.dumps({"verdicts": {"expected_rise": COST}}), encoding="utf-8")
    assert W.cost_path_for(L.F) == cpu                  # the card's job left no cost: the CPU fallback's
    card.write_text(json.dumps({"verdicts": {"expected_rise": COST}}), encoding="utf-8")
    assert W.cost_path_for(L.F) == card


def test_the_allowance_follows_the_spread_and_never_exceeds_the_base():
    cost = dict(R={"audio": 0.29}, sigma_RV=0.30)
    assert W.allowance(cost, "audio", None) == 0.29
    assert W.allowance(cost, "audio", 0.05) == 0.29
    assert W.allowance(cost, "audio", 0.60) == pytest.approx(0.29 * (0.53 / 0.30) ** 2)
    assert W.allowance(cost, "audio", 5.0) == W.allowance(cost, "audio", 0.53)
    # a revival may at most double a sense's ratio before the wander (review 2, finding 1)
    assert W.allowance(cost, "audio", 0.60, base=0.8836) == pytest.approx(0.8836)
    assert W.allowance(dict(R={"audio": 0.29}, sigma_RV=None), "audio", 0.5) == 0.29


def test_report_1_is_printed_reports_2_and_3_are_judged_together(tmp_path):
    line2 = (BASE["audio"] + 0.05) * W1["audio"]
    # report 1 far past any line (the revival's first steps), reports 2 and 3 inside: no stop
    L = _lineage(tmp_path / "r1", after=_after(4, ratio=lambda k: dict(BASE, audio=9.0) if k == 1 else BASE,
                                              between=0.1))
    ev = _ev(L.F)
    assert ev["why"] is None and ev["judgement"]["readings"]["report1"]["ratios"]["audio"] == 9.0
    printed = W.lines(L.F, W.StatsReader())
    assert any("Report 1, printed and never judged" in line and "audio at" in line for line in printed)
    # report 2 alone past its line, report 3 back inside: the lineage's own single-report noise, no stop (third
    # review's fix round: single report-2 crossings stopped 4.6 in 100 switch positions since 382,800)
    L = _lineage(tmp_path / "r2", after=_after(4, ratio=_audio(first=1.05 * line2, later=BASE["audio"]),
                                              between=0.1))
    ev = _ev(L.F)
    assert ev["why"] is None and ev["judgement"]["lines"]["K1"]["state"] == "held", ev["why"]
    # reports 2 and 3 both past it: stop at report 3
    L = _lineage(tmp_path / "r3", after=_after(4, ratio=_audio(first=1.05 * line2, later=1.10 * line2),
                                              between=0.1))
    ev = _ev(L.F)
    why = ev["why"]
    assert why and "reports 2 and 3 after the switch" in why and "105%" in why, why
    # a stop reason is read alone, so it names the sense by the registry's name with its plain meaning
    assert "the error of audio (what the two ears hear) one model step ahead reached" in why, why
    assert ev["id"] == "K1:reports2-3" and ev["judgement"]["stop"]["step"] == _step(3)
    assert not bare_decimals(why) and not unanchored_numbers(why), why


def _window_of(k):
    """The window (1 = reports 2-21) full report k belongs to; 0 for report 1."""
    return 0 if k < 2 else (k - 2) // 20 + 1


def test_the_other_senses_are_judged_in_every_20_report_window_through_the_watch(tmp_path):
    a = min(BASE["audio"], 0.05 * (0.45 / 0.30) ** 2)
    # healthy through two windows, then window 3 (reports 42-61) past the stop line against the level before the
    # switch (K6): the first version judged once, at report 21, and missed it
    ratio = lambda k: dict(BASE, audio=(BASE["audio"] if k <= 41 else 1.10 * _line("audio", 3, a)))   # noqa: E731
    L = _lineage(tmp_path / "w", after=_after(61, ratio=ratio, between=0.45))
    ev = _ev(L.F)
    assert ev["why"] and "the 20 full reports 42-61" in ev["why"] and ev["id"] == "K6:w2", ev["why"]
    assert "the stop line" in ev["why"] and "over the 26 full reports before it" in ev["why"], ev["why"]
    assert not bare_decimals(ev["why"]) and not unanchored_numbers(ev["why"]), ev["why"]
    K6 = ev["judgement"]["lines"]["K6"]
    assert K6["windows"] == 3 and K6["crossed_windows"] == ["K6:w2"]
    # K3 against the window before stops only on the same sense crossing in two windows running: one window's
    # rise past W20 from a dip (the lineage's audio at step 377,000) is not a stop by itself
    dip = lambda k: dict(BASE, audio=BASE["audio"] * (0.3 if _window_of(k) == 1 else 1.0))     # noqa: E731
    L = _lineage(tmp_path / "dip", after=_after(41, ratio=dip, between=0.45))
    ev = _ev(L.F)
    assert ev["why"] is None and ev["judgement"]["lines"]["K3"]["over_this_window"] == ["audio"], ev["why"]
    L = _lineage(tmp_path / "dip2", after=_after(61, ratio=dip, between=0.45))
    assert _ev(L.F)["why"] is None
    # two windows running, each past W20 against the one before it: K3 stops in the second (vision: its K6 stop line,
    # the level before the switch times 120% of its largest rise on record, lies above one window's step of W20)
    rise = lambda k: dict(BASE, vision=BASE["vision"] * (1.0 if _window_of(k) <= 1 else              # noqa: E731
                                                         1.1 * W20["vision"] if _window_of(k) == 2 else
                                                         (1.1 * W20["vision"]) ** 2))
    L = _lineage(tmp_path / "rise", after=_after(61, ratio=rise, between=0.45))
    ev = _ev(L.F)
    assert ev["id"] == "K3:w2" and "two windows running" in ev["why"], ev["why"]
    # the step rate: under 90 in 100 of the window before in two windows running stops; one window does not, and
    # the rate against the reports before the switch is a reading
    L = _lineage(tmp_path / "s1", after=_after(41, sps=lambda k: 0.5 if k <= 21 else 0.44, between=0.1))
    ev = _ev(L.F)
    assert ev["why"] is None and ev["judgement"]["lines"]["K5"]["under_line_this_window"], ev["why"]
    sb = ev["judgement"]["readings"]["speed_against_before"]
    assert sb[-1]["window"] == 2 and sb[-1]["share_of_before"] == pytest.approx(0.44 / SPS_BASE, rel=1e-3)
    slow = lambda k: 0.5 if k <= 21 else 0.44 if k <= 41 else 0.385                                # noqa: E731
    L = _lineage(tmp_path / "s2", after=_after(61, sps=slow, between=0.1))
    ev = _ev(L.F)
    assert ev["why"] and ev["id"] == "K5:w2" and "88%" in ev["why"] and "second window" in ev["why"], ev["why"]
    # with no reports before the switch the later windows are judged against the window before them
    exp = tmp_path / "nobefore" / "experiments"
    F = _run(exp, "H84_deploy", parent="H84_missing_parent", ck_step=N,
             records=[_switch()] + _after(61, sps=lambda k: 0.5 if k <= 21 else 0.44 if k <= 41 else 0.385))
    ev = _ev(F)
    assert ev["why"] and ev["id"] == "K5:w2", ev["why"]


def test_a_later_window_is_granted_only_the_allowances_growth(tmp_path):
    # third review, section B: a later window judged against the window before it re-granted the whole cost, which
    # the window before already held; now only the allowance's growth since then
    cost = dict(R={s: 0.05 for s in SENSES}, sigma_RV=0.30)
    before = [W.compact(_mimo(409200 + 100 * i)) for i in range(20)]
    after = [W.compact(r) for r in _after(41, between=0.30)]
    sw = W.compact(_switch())
    _why, _cid, rep = W.judge(before, after, sw, dict(cost, R_newest={}, moments=3), N)
    k3 = rep["lines"]["K3"]
    # window 2's median equals window 1's; its line is REF x W20 exactly (no growth at the same spread)
    assert k3["window"] == 1 and k3["shares"]["audio"] == pytest.approx(1.0 / W20["audio"], rel=1e-6)
    after = [W.compact(r) for r in _after(41, between=lambda k: 0.30 if k <= 21 else 0.45)]
    _why, _cid, rep = W.judge(before, after, sw, dict(cost, R_newest={}, moments=3), N)
    grow = 0.05 * (0.45 / 0.30) ** 2 - 0.05
    assert rep["lines"]["K3"]["shares"]["audio"] == pytest.approx(
        BASE["audio"] / ((BASE["audio"] + grow) * W20["audio"]), rel=1e-6)


# the fourth review's flat plant (vrev4/plant.py): a switch at 415,000, the reports before it and after it flat at these
# levels, the token alive, the record's example cost; a fault multiplies one sense's ratio from the report named
FLAT_N = 415000
FLAT_BASE = {"vision": 1.0, "tactile": 1.0, "audio": 0.8836, "proprioception": 1.0}
PLANT_COST = {"R": {"vision": 0.001, "tactile": 0.001, "audio": 0.0039, "proprioception": 0.001},
              "R_newest": {s: None for s in SENSES}, "sigma_RV": 0.30, "moments": 3, "source": "planted"}
# the third review's W20, the lines its creep was planted against (vrev3/plant_creep.py)
OLD_W20 = {"vision": 2.10, "tactile": 1.59, "audio": 1.29, "proprioception": 1.28}


def _flat_rec(step, ratio, sps=10.0):
    return {"type": "mimo", "step": step, "timestamp": step, "nf_steps": 100, "ratio": dict(ratio, vestibular=5.0),
            "capped_vest": 100, "share_vest": 0.2, "replay_share_vest": 0.2, "tts_vest": ALIVE,
            "liveness": {"mlp.0": 1.0, "mlp.4": 1.0, "mlp.8": 1.0}, "nan_steps": 0, "sps": sps,
            "improvement": -0.1, "probe": None, "between": 0.45, "ceiling": None, "lp_vest": 0.0, "lp_value": 0.0}


def _flat_plant(factor, speed=None):
    """judge() on the flat plant: `factor(k, sense)` multiplies full report k's ratio, `speed(k)` its step rate; the
    100 reports before the switch at the flat levels. Returns (the stop's id, the stop's window (1 = reports 2-21),
    the first window K6's alert level fired in, the report)."""
    before = [_flat_rec(FLAT_N - 100 * (100 - i), FLAT_BASE) for i in range(100)]
    after = [_flat_rec(FLAT_N + 100 * k, {s: FLAT_BASE[s] * factor(k, s) for s in SENSES},
                       10.0 * (speed(k) if speed else 1.0)) for k in range(1, 102)]
    switch = {"type": W.SWITCH_TYPE, "step": FLAT_N, "switch": dict(SWITCH)}
    why, cid, rep = W.judge(before, after, switch, PLANT_COST, FLAT_N, taken={})
    rep["stop_why"] = why

    def window(c):
        return None if not c else 1 if c.startswith("K1:reports") else int(c.split(":w")[1]) + 1
    alerts = [a["window"] + 1 for a in rep.get("alerts") or []] + [
        window(c) for c in (rep["lines"].get("K6") or {}).get("crossed_windows") or []]
    return cid, window(cid), (min(alerts) if alerts else None), rep


def _onset_window(report):
    return 1 if report <= 21 else (report - 2) // 20 + 1


def test_the_fourth_reviews_fast_collapse_stops_within_2_windows():
    # the fourth review (section 2, vrev4/plant.py): hearing jumping to 3, 4 or 5 times its level after report 3
    # passed every stop (K6's line had grown to 3.3-5.4 times hearing's level by windows 2-4). Now K6's stop line is
    # the same in every window: each jump stops in the window it starts in or the next
    for factor in (3.0, 4.0, 5.0):
        for first in (2, 13, 22, 42, 62):
            cid, win, alert, rep = _flat_plant(lambda k, s: factor if (s == "audio" and k >= first) else 1.0)
            assert cid and win <= _onset_window(first) + 1, (factor, first, cid, rep.get("stop"))
            assert alert is None or alert <= win
    # the review's three: x3 from report 13 in window 2, x4 from report 42 in window 3, x5 from report 62 in window 4
    for factor, first, want in ((3.0, 13, "K6:w1"), (4.0, 42, "K6:w2"), (5.0, 62, "K6:w3")):
        assert _flat_plant(lambda k, s: factor if (s == "audio" and k >= first) else 1.0)[0] == want


def test_the_third_reviews_creep_at_its_own_rates_stops_by_window_2():
    # third review, section B (vrev3/plant_creep.py through the branch's own judge): each other sense's ratio
    # multiplied every report so each window reads 0.97 of ITS line against the window before (that review's W20:
    # vision 2.10, touch 1.59, hearing 1.29, body position 1.28), the step rate 0.92 of the window before; the fourth
    # review found the fix round's test planted 0.97 of the NEW W20 instead. It stops in the second window (reports
    # 22-41), through touch (1.98 times its level against a stop line of 1.72), with K3 and K5 holding
    rates = {s: 0.97 * OLD_W20[s] for s in SENSES}
    cid, win, alert, rep = _flat_plant(lambda k, s: rates[s] ** (k / 20.0), lambda k: 0.92 ** (k / 20.0))
    assert cid == "K6:w1" and win == 2 and rep["stop"]["step"] == FLAT_N + 4100, (cid, rep.get("stop"))
    assert "the 20 full reports 22-41" in rep["stop_why"], rep["stop_why"]
    assert "tactile (the skin's sense of touch)" in rep["stop_why"], rep["stop_why"]
    assert rep["lines"]["K3"]["crossed_windows"] == [] and rep["lines"]["K5"]["crossed_windows"] == []
    # each sense alone at that review's rate (the fourth review asked which pass): touch stops in window 2; hearing
    # and body position (1.42 and 1.41 times their level in window 2, under the largest rise on record, 1.43) and
    # vision (3.07, under its own, 3.29) stop in window 3, where each passes its stop line
    for sense, want in (("tactile", 2), ("audio", 3), ("proprioception", 3), ("vision", 3)):
        cid, win, alert, _rep = _flat_plant(lambda k, s: rates[s] ** (k / 20.0) if s == sense else 1.0)
        assert win == want and alert == want, (sense, cid, win, alert)


def _false_stops_doc():
    return json.loads((TREE / "docs" / "gain_of_function_v3" / "measurements" /
                       "vestibular_repair_false_stops_2026-09-26.json").read_text(encoding="utf-8"))


def test_the_kill_lines_hold_their_measured_false_stop_chance_in_and_out_of_sample():
    # decided by the coordinator (2026-09-26): at most 5 stops in 100 watches with nothing switched; after the fourth
    # review, for the stop and for the alert level, OUT OF SAMPLE: the lines drawn from one half of the lineage (split
    # at 380,000) and judged on the other, both ways. The lines' wander and largest rise are the measured ones
    doc = _false_stops_doc()
    larger = doc["wander"]["larger_of_the_two"]
    assert larger["W1"] == W1 and larger["W20"] == W20 and larger["RISE"] == W.LINES["RISE"]
    assert doc["wander"]["quantile"] == W.LINES["wander_quantile"]
    for key in ("W1", "W20", "RISE", "alert_margin", "stop_margin", "level_reports", "K5_share", "K3_consecutive",
                "K5_consecutive", "wander_quantile"):
        assert doc["lines"][key] == W.LINES[key], key
    assert list(doc["lines"]["K1_reports"]) == list(W.LINES["K1_reports"])
    # the steady senses share one largest rise; vision its own
    assert len({W.LINES["RISE"][s] for s in W.LINES["steady_senses"]}) == 1
    for name in ("since_382800", "since_the_sept_23_deploy"):
        st = doc["stretches"][name]
        assert st["any_stop_in_100"] <= 5.0 and st["alert_level_in_100"] <= 5.0 and st["positions"] > 100, (name, st)
    oos = doc["out_of_sample"]
    assert oos["split"] == 380000
    for name in ("lines_A_judged_on_B", "lines_B_judged_on_A"):
        st = oos[name]
        assert st["positions"] > 100, (name, st)
        assert st["any_stop_in_100"] <= 5.0 and st["alert_level_in_100"] <= 5.0, (name, st)
    # the lines drawn from each half are the rule's own output on that half
    for half in ("lines_A", "lines_B"):
        assert set(oos[half]) == {"W1", "W20", "RISE"}
    # and the fourth review's recount of the earlier lines (21755aab9) out of sample, measured the same way
    assert doc["compare"]["label"] == "21755aab9"
    assert max(v["any_stop_in_100"] for v in doc["compare"]["out_of_sample"].values()) > 5.0


def test_the_plants_stop_within_2_windows_with_lines_drawn_from_either_half():
    # the coordinator's line (a): the fourth review's fast collapse and the third review's creep at its own rates stop
    # within 2 windows (the window the fault starts in or the next), with the whole lineage's lines and with each
    # half's lines (flat plant), and on the lineage's own reports (real plant: every switch position whose unplanted
    # watch neither stops nor alerts)
    plants = _false_stops_doc()["plants"]
    fast = ["fast: hearing x%d from report %d" % (f, r) for f in (3, 4, 5) for r in (2, 13, 22, 42, 62)]
    creep = "the third review's creep: all four senses at 0.97 of that review's line per window, the step rate 0.92"
    for name in fast + [creep]:
        p = plants[name]
        limit = _onset_window(p["first_report"]) + 1
        for flat in ("flat", "flat_lines_A", "flat_lines_B"):
            assert p[flat]["stop_window"] and p[flat]["stop_window"] <= limit, (name, flat, p[flat])
            assert p[flat]["alert_window"] and p[flat]["alert_window"] <= limit, (name, flat, p[flat])
        for real in ("real_since_382800", "real_lines_A_on_B", "real_lines_B_on_A"):
            r = p[real]
            assert r["clean"] > 100 and r["alert_within_2_windows"] == r["clean"], (name, real, r)
            if name in fast:
                assert r["stop_within_2_windows"] == r["clean"], (name, real, r)
            else:
                # the creep on the lineage's own reports: the stop within 2 windows at 76 to 84 of every 100 positions
                # (a real window-2 touch reading under 87 in 100 of its level leaves the creep under the stop line
                # until window 3); the alert level at every position
                assert r["stop_within_2_windows"] >= 0.7 * r["clean"], (name, real, r)


def test_the_switch_record_itself_is_judged(tmp_path):
    for i, (change, word) in enumerate(((dict(switched=False), "was divided"), (dict(bias_spread_after=9.09), "spread"),
                                        (dict(checks={"float32": {"within": True}, "bfloat16": {"within": False}}),
                                         "bfloat16"),
                                        (dict(runtime_restart={}), "restart"))):
        why = _ev(_lineage(tmp_path / str(i), after=_after(2), switch=dict(SWITCH, **change)).F)["why"]
        assert why and "not as declared" in why and word in why, why


def test_a_token_that_stays_dead_after_2000_updates_stops(tmp_path):
    assert _ev(_lineage(tmp_path / "a", after=_after(22, tts=lambda k: DEAD if k <= 19 else ALIVE)).F)["why"] is None
    assert _ev(_lineage(tmp_path / "b", after=_after(24, tts=lambda k: ALIVE if k in (21, 23) else DEAD)).F)["why"] is None
    why = _ev(_lineage(tmp_path / "c", after=_after(24, tts=DEAD)).F)["why"]
    assert why and "stayed dead" in why, why


def _one_encoder(rows, value, fresh=True):
    """The stored-token drift fix's fields on each record: the balance token's collapse watch through one encoder
    (`value`, a number or a callable of the record's step), its window closed at the report's own thousand (fresh),
    or marked stale by the trainer."""
    for r in rows:
        v = value(r["step"]) if callable(value) else value
        r["token_spread_one_encoder"] = {"vestibular": v}
        r["token_spread_window"] = {"end_step": r["step"] - r["step"] % 1000, "fresh": fresh}
    return rows


def test_the_dead_token_stop_and_a2_judge_the_one_encoder_reading_when_the_records_carry_it(tmp_path):
    """A revived encoder whose token answers nothing while its weights move: through one encoder its token is the same
    every moment (0), while the stored tokens read its drift (ALIVE, over both lines). On the stored reading the watch
    never stopped and A2 was met on drift alone; on the one-encoder reading K2 stops the run and A2 is missed."""
    ev = _ev(_lineage(tmp_path / "silent", after=_one_encoder(_after(45, tts=ALIVE), 0.0)).F)
    assert ev["why"] and "stayed dead" in ev["why"], ev["why"]
    k2 = ev["judgement"]["lines"]["K2"]
    assert k2["reading"] == "one encoder" and k2["judged_reports"] > 0
    a2 = ev["judgement"]["acceptance"]["A2"]
    assert a2["reading"] == "one encoder" and a2["state"].startswith("missed"), a2
    # the same records judged on the stored tokens, as before the fix: no stop, A2 met
    stored = _ev(_lineage(tmp_path / "stored", after=_after(45, tts=ALIVE)).F)
    assert stored["why"] is None and stored["judgement"]["lines"]["K2"]["reading"] == "stored tokens"
    assert stored["judgement"]["acceptance"]["A2"]["state"].startswith("met")
    said = "\n".join(W.lines(_lineage(tmp_path / "said", after=_after(24)).F, W.StatsReader(), write=False))
    assert "read from stored tokens, which count the encoder's drift as movement" in said
    # a revived token that answers its input reads over both lines through one encoder, whatever the stored tokens say
    alive = _ev(_lineage(tmp_path / "alive", after=_one_encoder(_after(45, tts=DEAD), ALIVE)).F)
    assert alive["why"] is None and alive["judgement"]["acceptance"]["A2"]["state"].startswith("met")
    # a stale one-encoder reading is not judged: the stored reading, and the lines say so
    stale = _ev(_lineage(tmp_path / "stale", after=_one_encoder(_after(24, tts=DEAD), ALIVE, fresh=False)).F)
    assert stale["why"] and "stayed dead" in stale["why"]
    assert stale["judgement"]["lines"]["K2"]["reading"] == "stored tokens"
    text = "\n".join(W.lines(_lineage(tmp_path / "print", after=_one_encoder(_after(24), 0.0)).F, W.StatsReader(),
                             write=False))
    assert "read through one encoder" in text and not bare_decimals(text)


def test_the_encoder_non_finite_updates_and_liveness(tmp_path):
    after = _after(3)
    after[0]["nan_steps"] = 2                                  # the partial record at 411,200
    why = _ev(_lineage(tmp_path / "nan", after=after).F)["why"]
    assert why and "non-finite" in why and "411200" in why, why
    assert _ev(_lineage(tmp_path / "one", after=_after(4, live=lambda k: 0.5 if k == 2 else 1.0)).F)["why"] is None
    why = _ev(_lineage(tmp_path / "two", after=_after(4, live=lambda k: 0.5 if k in (2, 3) else 1.0)).F)["why"]
    assert why and "mlp.0" in why and "fewer than 90 of every 100 units" in why, why


def test_without_the_reports_before_the_switch_or_the_cost_the_unconditional_lines_still_stop(tmp_path, capsys):
    # the parent chain broken: no reports before the switch (review 2, finding 3's reproduction)
    exp = tmp_path / "experiments"
    F = _run(exp, "H84_deploy", parent="H84_missing_parent", ck_step=N,
             records=[_switch()] + _after(26, tts=DEAD, nan=lambda k: 1 if k == 25 else 0))
    (F / W.COST_NAME).write_text(json.dumps({"verdicts": {"expected_rise": COST}}), encoding="utf-8")
    ev = _ev(F)
    assert ev["why"] and ("stayed dead" in ev["why"] or "non-finite" in ev["why"]), ev["why"]
    assert any("only 0 of the 20 full reports before the switch" in w for w in ev["warnings"])
    lines = ev["judgement"]["lines"]
    assert lines["K3"]["state"] == "waiting" and lines["K5"]["state"] == "waiting" and lines["K6"]["state"] == "waiting"
    # the 5-minute check says what it cannot judge, loudly, every time
    F2 = _run(exp, "H84_deploy2", parent="H84_missing_parent", ck_step=N, records=[_switch()] + _after(3))
    assert W.breach(F2, W.StatsReader()) is None
    out = capsys.readouterr().out
    assert "VESTIBULAR REPAIR KILL LINES INCOMPLETE" in out and "0 of the 20 full reports" in out
    assert "no declared cost" in out
    # without the cost: K1's report part and K3 wait, the switch and the token are judged
    L = _lineage(tmp_path / "nocost", after=_after(24, tts=DEAD), cost=dict(COST, moments=1))
    ev = _ev(L.F)
    assert ev["why"] and "stayed dead" in ev["why"] and "1 saved model step" in ev["cost_refused"]
    assert any("no declared cost" in w for w in ev["warnings"])


# ------------------------------------------------------------------------------ the relaunch case
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


def test_the_supervisor_ends_and_starts_watchers_only_by_the_hourly_watchers_file_name(tmp_path):
    from mimo_bridge.tools import training_continuation as TC
    L = _lineage(tmp_path)
    G = _run(L.exp, "H84_deploy_cont", parent="H84_deploy", ck_step=N + 1000)
    sleeper = "import time\ntime.sleep(120)\n"
    hourly = tmp_path / "copy_a" / "hourly_watch.py"
    other = tmp_path / "copy_b" / "vestibular_repair_watch.py"
    for p in (hourly, other):
        p.parent.mkdir()
        p.write_text(sleeper, encoding="utf-8")
    assert TC.is_watcher_command([sys.executable, "-u", str(hourly), str(L.F), "8760"])
    assert not TC.is_watcher_command([sys.executable, "-u", str(other), str(L.F)])
    procs = [subprocess.Popen([sys.executable, "-u", str(hourly), str(L.F), "8760"]),
             subprocess.Popen([sys.executable, "-u", str(other), str(L.F)])]
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not TC.processes("watcher", L.F):
            time.sleep(0.2)
        found = TC.processes("watcher", L.F)
        assert [p.pid for p in found] == [procs[0].pid], [p.info["cmdline"] for p in found]
        assert TC.processes("watcher", G) == []
        TC.stop_watchers(L.F)                                   # what the supervisor does at a relaunch
        procs[0].wait(timeout=20)
        assert procs[0].returncode is not None and procs[1].poll() is None
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=20)


def test_the_watcher_started_for_a_relaunched_run_judges_the_whole_timeline_and_stops_that_run(tmp_path, monkeypatch):
    line20 = _line("audio", 1)

    def audio(k):
        return dict(BASE, audio=1.10 * line20 if k >= 2 else BASE["audio"])
    # the deploy ran to report 12, then the supervisor relaunched it: the relaunched run holds reports 13-25
    L = _lineage(tmp_path, after=_after(12, ratio=audio, between=0.2))
    G = _run(L.exp, "H84_deploy_cont", parent="H84_deploy", ck_step=_step(12),
             records=_after(25, first=13, partial=False, ratio=audio, between=0.2))
    hf = _hourly(monkeypatch, L.F)
    assert hf["quick_breach"]() is None
    hg = _hourly(monkeypatch, G)
    why = hg["quick_breach"]()
    assert why and why.startswith("vestibular encoder repair:") and "audio (what the two ears hear)" in why, why
    hg["stop_run"](why)
    assert json.loads((G / "control.json").read_text())["cmd"] == "snapshot_and_stop"
    reason = json.loads((G / "stop_reason.json").read_text())
    assert reason["resume_policy"] == "repair_required" and reason["reason"] == why
    assert not (L.F / "control.json").exists() and not (L.F / "stop_reason.json").exists()
    assert not bare_decimals(why) and not unanchored_numbers(why), why
    crossing = W.load_state(L.F)["crossings"][0]
    assert crossing["run"] == str(G) and crossing["id"] == "K6:w0"
    assert W.taken_crossings(L.F) == {"K6:w0": _step(21)}
    # a deliberate continuation of the repaired lineage is not stopped again on the same window ...
    J = _run(L.exp, "H84_after_review", parent="H84_deploy_cont", ck_step=_step(25),
             records=_after(40, first=26, partial=False, ratio=lambda k: BASE, between=0.2))
    assert _hourly(monkeypatch, J)["quick_breach"]() is None
    # ... but the next window past its line is a stop (review 2, finding 5: only the taken crossing is suppressed)
    L2 = _lineage(tmp_path / "w2", after=_after(41, ratio=lambda k: dict(
        BASE, audio=1.10 * (line20 if k <= 21 else _line("audio", 2))), between=0.2))
    W.record_crossing(L2.F, dict(id="K6:w0", line="K6", reason="r", run=str(L2.F), step=_step(21)))
    (L2.F / "stop_reason.json").write_text(json.dumps({"reason": "r"}), encoding="utf-8")
    ev = _ev(L2.F)
    assert ev["id"] == "K6:w1", ev["why"]


def test_a_taken_dead_token_stop_does_not_silence_a_later_one(tmp_path):
    # dead in reports 20-22 (stopped and taken at 22), alive again, then dead again in 41-43
    L = _lineage(tmp_path, after=_after(45, tts=lambda k: DEAD if k in (20, 21, 22, 41, 42, 43) else ALIVE))
    ev = _ev(L.F)
    assert ev["id"] == "K2@%d" % _step(22)
    W.record_crossing(L.F, dict(id=ev["id"], line="K2", reason=ev["why"], run=str(L.F), step=_step(22)))
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": ev["why"]}), encoding="utf-8")
    ev = _ev(L.F)
    assert ev["id"] == "K2@%d" % _step(43), ev["why"]


def test_a_stop_that_never_reached_its_run_is_judged_again(tmp_path):
    line20 = _line("audio", 1)
    L = _lineage(tmp_path, after=_after(22, ratio=lambda k: dict(BASE, audio=1.1 * line20), between=0.2))
    why = W.breach(L.F, W.StatsReader())
    assert why
    assert W.taken_crossings(L.F) == {}
    assert W.breach(L.F, W.StatsReader()) == why


def test_the_hourly_watchers_own_program_writes_the_stop_into_the_relaunched_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: HEALTHY_DISK)
    monkeypatch.setenv("UEMA_WATCH_SAVED", str(tmp_path / "_saved"))
    monkeypatch.setenv("UEMA_WATCH_ARCHIVE", str(tmp_path / "_archive"))
    L = _lineage(tmp_path, after=_after(2))
    G = _run(L.exp, "H84_deploy_cont", parent="H84_deploy", ck_step=_step(2),
             records=_after(26, first=3, partial=False, tts=DEAD))
    monkeypatch.setattr(sys, "argv", [str(WATCHER), str(G), "1"])
    monkeypatch.setattr(time, "sleep", lambda _: None)
    runpy.run_path(str(WATCHER), run_name="__main__")
    out = capsys.readouterr().out
    assert "STOP RULE REACHED: vestibular encoder repair:" in out and "stayed dead" in out, out[-2000:]
    assert json.loads((G / "control.json").read_text())["cmd"] == "snapshot_and_stop"
    assert "stayed dead" in json.loads((G / "stop_reason.json").read_text())["reason"]


def test_a_run_no_switch_reaches_is_quiet(tmp_path, monkeypatch, capsys):
    L = _lineage(tmp_path)
    assert W.breach(L.E, W.StatsReader()) is None and W.lines(L.E, W.StatsReader()) == []
    assert _hourly(monkeypatch, L.E)["vestibular_repair_lines"]() == []
    assert "VESTIBULAR" not in capsys.readouterr().out


# ------------------------------------------------------------------------------ acceptance, and what a person reads
def test_the_acceptance_lines_read_as_declared(tmp_path):
    probe = {"r2": {"head_tilt": 0.6, "trunk_roll": 0.4}, "r2_shuffled": {"head_tilt": -0.05, "trunk_roll": -0.04},
             "within_body_turn_deg": {"head_tilt": 4.0, "trunk_roll": 2.0}}
    reports = 101        # full report 100 is at N + 10,075: the last the watch reads; 101 is past its end
    after = _after(reports, capped=lambda k: 5 if k <= 3 else 0, tts=ALIVE, between=0.3, probe=probe,
                   improvement=lambda k: 0.08 + 0.01 * (k % 3))
    L = _lineage(tmp_path / "met", after=after)
    ev = _ev(L.F)
    acc = ev["judgement"]["acceptance"]
    assert ev["why"] is None and ev["judgement"]["after_full_reports"] == 100 and ev["watch_over"]
    assert acc["A2"]["state"] == "met" and acc["A3"]["state"] == "met"
    assert acc["A4"]["state"] == "met so far" and acc["A5"]["state"] == "met" and acc["A5"]["block"] == 5, acc
    assert acc["A4"]["read"]["trunk_roll"]["margin_over_shuffled"] == pytest.approx(0.44)
    assert ev["judgement"]["readings"]["capped_reports"] == 3
    printed = W.lines(L.F, W.StatsReader())
    assert len(printed) == 11
    for line in printed:
        assert not bare_decimals(line), line
        assert not unanchored_numbers(line), line
    low = W.LINES["A2_line"] / 2.0
    # the other senses' last window between K6's alert and stop lines: A3 misses, an alert, no stop
    missed = _after(reports, tts=lambda k: low if k >= 40 else ALIVE,
                    ratio=lambda k: dict(BASE, audio=0.5 * (_alert_line("audio", 0.0) + _line("audio", allowance=0.0)))
                    if k > 80 else BASE,
                    probe={"r2": {"head_tilt": 0.6, "trunk_roll": 0.1}, "r2_shuffled": {"head_tilt": 0.0,
                                                                                       "trunk_roll": 0.0}},
                    improvement=lambda k: -0.05 + 0.01 * (k % 3), between=0.1)
    ev = _ev(_lineage(tmp_path / "missed", after=missed).F)
    assert ev["why"] is None, ev["why"]
    acc = ev["judgement"]["acceptance"]
    assert acc["A2"]["state"] == "missed" and acc["A2"]["lowest"] == low and acc["A2"]["median"] == low
    assert acc["A3"]["state"] == "missed" and acc["A3"]["missed"] == ["audio"]
    assert acc["A4"]["state"] == "missed so far" and acc["A5"]["state"] == "missed"
    # (the fixture's report 101 lies past the watch's end, so its last window is the 20 reports ending there)
    assert [a["id"] for a in ev["judgement"]["alerts"]] == ["K6-alert:end"]


def test_a3_reads_missed_for_hearing_at_2_2_times_its_level(tmp_path):
    # the fourth review (section 2): A3's line was BASE x W20, so hearing at 2.2 times its level read "met" (95 in 100
    # of its line). Now: the level before the switch times the largest rise the lineage showed (143% for hearing)
    after = _after(101, tts=ALIVE, between=0.1, ratio=lambda k: dict(BASE, audio=2.2 * BASE["audio"]) if k > 81
                   else BASE)
    acc = _ev(_lineage(tmp_path, after=after).F)["judgement"]["acceptance"]["A3"]
    assert acc["state"] == "missed" and acc["missed"] == ["audio"], acc
    assert acc["multiple_of_base"]["audio"] == pytest.approx(2.2, rel=1e-3)
    assert acc["line_multiple"]["audio"] == pytest.approx(W.LINES["RISE"]["audio"])
    # the level printed as a multiple of the level before and of BASE beside the line
    printed = W.lines(_lineage(tmp_path / "p", after=after).F, W.StatsReader())
    line = next(p for p in printed if "acceptance (never a stop)" in p)
    assert "audio at 220% of its level before the switch" in line and "against a line of 143%" in line, line
    assert not bare_decimals(line) and not unanchored_numbers(line), line


def test_a2_is_judged_on_the_median_so_a_revival_that_reads_like_the_measurement_meets_it(tmp_path):
    # the fourth review (section 3): A2 asked every report from 4,000 to 10,000 updates for at least the LOWEST of the
    # six stretches the revival was measured on, which a live revival drawn like the measurement meets 50 in 100 times.
    # Now the median of those reports: readings within 10 in 100 of the six measured stretches (the committed OT
    # measurement on 415,075, the core arm after 100 updates, bf16 tokens) meet it; half the measurement does not
    doc = json.loads((TREE / "docs" / "gain_of_function_v3" / "measurements" /
                      "vestibular_encoder_over_time_415075_2026-09-26.json").read_text(encoding="utf-8"))
    import importlib.util as iu
    spec = iu.spec_from_file_location("vem_for_a2", TREE / "docs" / "gain_of_function_v4" /
                                      "vestibular_encoder_measurements.py")
    M = iu.module_from_spec(spec)
    spec.loader.exec_module(M)
    measured = M.ot_verdict(doc["OT"], doc["collapse_watch_live"])["by_encoder"]["CORE@100"]["bfloat16"][
        "predicted_watch_body0"]
    assert len(measured) == 6 and min(measured) == pytest.approx(W.LINES["A2_line"], rel=5e-3)
    like = lambda k: measured[k % 6] * (0.9 if (k // 6) % 2 == 0 else 1.1)                   # noqa: E731
    acc = _ev(_lineage(tmp_path / "like", after=_after(101, tts=like, between=0.1)).F)["judgement"]["acceptance"]
    assert acc["A2"]["state"] == "met" and acc["A2"]["lowest"] < W.LINES["A2_line"], acc["A2"]
    half = lambda k: 0.5 * like(k)                                                            # noqa: E731
    acc = _ev(_lineage(tmp_path / "half", after=_after(101, tts=half, between=0.1)).F)["judgement"]["acceptance"]
    assert acc["A2"]["state"] == "missed", acc["A2"]


def test_a_mild_rise_is_an_alert_printed_once_and_recorded_never_a_stop(tmp_path, capsys):
    # the coordinator's two levels (the fourth review's fix round): touch 1.55 times its level from report 2 (the
    # review's plant, which passed every line with A3 "met") is over K6's alert line and under its stop line
    rise = lambda k: dict(BASE, tactile=1.55 * BASE["tactile"]) if k >= 2 else BASE                  # noqa: E731
    L = _lineage(tmp_path, after=_after(45, ratio=rise, between=0.1))
    assert W.breach(L.F, W.StatsReader()) is None
    out = capsys.readouterr().out
    assert out.count("vestibular encoder repair ALERT") == 2 and "tactile (the skin's sense of touch)" in out, out  # windows 1 and 2, once each
    held = W.load_state(L.F)["alerts"]
    assert [a["id"] for a in held] == ["K6-alert:w0", "K6-alert:w1"] and held[0]["run"] == str(L.F)
    assert not (L.F / "control.json").exists() and not (L.F / "stop_reason.json").exists()
    for a in held:
        assert not bare_decimals(a["why"]) and not unanchored_numbers(a["why"]), a["why"]
    # the next 5-minute check prints nothing new; the hourly sweep lists them
    assert W.breach(L.F, W.StatsReader()) is None and "ALERT" not in capsys.readouterr().out
    printed = W.lines(L.F, W.StatsReader())
    assert any("alerts (a window over the alert line, never a stop): 2 window(s)" in p for p in printed), printed
    # reading by hand records nothing
    L2 = _lineage(tmp_path / "hand", after=_after(45, ratio=rise, between=0.1))
    assert W.main([str(L2.F)]) == 0 and not (L2.F / W.STATE_NAME).exists()


def test_the_level_before_the_switch_is_the_larger_of_the_20_and_the_100_report_median(tmp_path):
    # a dip in the 20 reports before the switch (hearing's 0.84 to 0.007 in seg36 is the lineage's own) does not
    # lower the level K6 judges against, so the recovery from it is not a stop; BASE (K1, K3's first window) keeps
    # the 20 reports
    before = [W.compact(_mimo(400000 + 100 * i, ratio=dict(BASE, audio=0.9 if i < 80 else 0.35))) for i in range(100)]
    after = [W.compact(r) for r in _after(45, between=0.1, partial=False)]
    base, level, n = W.level_before(before)
    assert n == 100 and base["audio"] == pytest.approx(0.35) and level["audio"] == pytest.approx(0.9)
    _why, _cid, rep = W.judge(before, after, W.compact(_switch()), dict(
        R={s: 0.0 for s in SENSES}, R_newest={}, sigma_RV=None, moments=3), 410000)
    assert rep["before"]["level"]["audio"] == pytest.approx(0.9) and rep["before"]["level_reports"] == 100
    assert rep["lines"]["K6"]["state"] == "held" and not rep["alerts"], rep["lines"]["K6"]
    # judged against the 20 reports alone, hearing back at its level reads 2.5 times it: a stop (under K1's single-
    # report line, 2.75 times BASE, so K6's)
    _why, cid, _rep = W.judge(before[-20:], after, W.compact(_switch()), dict(
        R={s: 0.0 for s in SENSES}, R_newest={}, sigma_RV=None, moments=3), 410000)
    assert cid == "K6:w0", cid
    # the watcher reads up to 100 full reports back through the parent chain (the fixture holds 26)
    ev = _ev(_lineage(tmp_path, after=_after(3)).F)
    assert ev["judgement"]["before"]["level_reports"] == 26 and ev["judgement"]["before_reports"] == 26


def test_the_hourly_sweep_prints_the_repairs_lines(tmp_path, monkeypatch):
    L = _lineage(tmp_path, after=_after(3, between=0.1))
    printed = _hourly(monkeypatch, L.F)["vestibular_repair_lines"]()
    assert printed[0].startswith("  vestibular repair: the vestibular encoder was divided at step 411,125")
    # the first line gives every sense name the lines below use its plain meaning (the style guide "ONE WORD PER THING":
    # the registry's names, each defined where first used), and no line uses the words these lines printed until
    # 2026-10-03
    for sense, meaning in W.SENSE_MEANS.items():
        assert "%s = %s" % (sense, meaning) in printed[0], sense
    old = ("touch", "hearing", "body position", "balance sense", "balance token", "balance encoder", "balance head",
           "next-moment", "moment")
    assert not [(word, line[:90]) for line in printed for word in old
                if word in line.replace("the skin's sense of touch", "")]
    assert any("the switch and reports 2 and 3 after it" in line and "held" in line for line in printed)
    assert any("against their level before the switch" in line for line in printed)
    assert any("no full report judged yet" in line or "units responding" in line for line in printed)
    for line in printed:
        assert not bare_decimals(line) and not unanchored_numbers(line), line


def test_reading_by_hand_writes_nothing(tmp_path, capsys):
    line20 = _line("audio", 1)
    L = _lineage(tmp_path, after=_after(22, ratio=lambda k: dict(BASE, audio=1.1 * line20), between=0.2))
    why = W.breach(L.F, W.StatsReader())                      # a recorded crossing ...
    (L.F / "stop_reason.json").write_text(json.dumps({"reason": why}), encoding="utf-8")   # ... whose stop was taken
    state_before = (L.F / W.STATE_NAME).read_bytes()
    listing = sorted(p.name for p in L.F.iterdir())
    assert W.main([str(L.F)]) == 0
    assert "divided at step 411,125" in capsys.readouterr().out
    assert (L.F / W.STATE_NAME).read_bytes() == state_before and sorted(p.name for p in L.F.iterdir()) == listing


# ------------------------------------------------------------------------------ the lesson weight (third review, D)
def test_the_lesson_weight_reading_applies_the_sept_23_lanes_rule(tmp_path):
    from mimo_bridge import vestibular_sense as VS
    # the watcher's copy of the rule is the declared one (vestibular_sense.lesson_weight_rule)
    for s, n, w0 in ((0.2, 5, 1.0), (0.33, 5, 1.0), (0.5, 5, 1.0), (0.11, 5, 1.0), (0.4, 5, 0.5), (0.9, 4, 1.0)):
        assert W.rule_weight(s, n, w0) == pytest.approx(VS.lesson_weight_rule(s, n, w0))
    # the lane's formula: 1.0 at or under the equal share, else m (1 - s) / ((1 - m) s)
    assert VS.lesson_weight_rule(0.2, 5) == 1.0 and VS.lesson_weight_rule(0.15, 5) == 1.0
    assert VS.lesson_weight_rule(1.0 / 3.0, 5) == pytest.approx(0.2 * (2.0 / 3.0) / (0.8 / 3.0))
    # the reading: the 20 full reports ending 10,000 updates after the switch, those whose weight left its cap
    reports = 101
    after = _after(reports, capped=lambda k: 3 if k < 95 else 0, share=0.3, replay_share=0.36, tts=ALIVE,
                   between=0.3)
    ev = _ev(_lineage(tmp_path / "few", after=after).F)
    lw = ev["judgement"]["readings"]["lesson_weight_rule"]
    assert lw["state"] == "unmeasured" and "cap" in lw["why"], lw
    after = _after(reports, capped=lambda k: 3 if k < 60 else 0, share=0.3, replay_share=0.36, tts=ALIVE,
                   between=0.3)
    ev = _ev(_lineage(tmp_path / "read", after=after).F)
    lw = ev["judgement"]["readings"]["lesson_weight_rule"]
    assert lw["state"] == "read" and lw["share"] == pytest.approx(0.36) and lw["graded"] == 5
    assert lw["weight_by_rule"] == pytest.approx(VS.lesson_weight_rule(0.36, 5, 1.0))
    printed = W.lines(_lineage(tmp_path / "read2", after=after).F, W.StatsReader())
    line = next(p for p in printed if "lesson weight by the Sept 23 lane's rule" in p)
    assert "the rule gives" in line and not bare_decimals(line), line


def test_the_false_stop_tool_counts_what_the_watchers_judge_stops(tmp_path):
    spec = importlib.util.spec_from_file_location("false_stops", TREE / "docs" / "gain_of_function_v4" /
                                                  "vestibular_repair_false_stops.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    # a flat lineage never stops; one report at 1,000 times its level never stops either (K1 needs reports 2 and 3,
    # a window's median ignores one report); two reports running stop the position whose reports 2 and 3 they are
    rows = [{"type": "mimo", "step": 100 * i, "timestamp": float(i), "nf_steps": 100, "ratio": dict(BASE),
             "sps": 0.5, "tts_vest": None, "liveness": None, "nan_steps": 0, "between": None, "improvement": None,
             "probe": None, "ceiling": None, "run": "r"} for i in range(300)]
    flat = T.measure(W, rows, 0)
    assert flat["positions"] == 300 - 19 - 100 and flat["any_stop_in_100"] == 0.0
    spiky = [dict(r, ratio=dict(BASE, audio=BASE["audio"] * 1000.0)) if i in (150,) else r for i, r in enumerate(rows)]
    assert T.measure(W, spiky, 0)["any_stop_in_100"] == 0.0
    pair = [dict(r, ratio=dict(BASE, audio=BASE["audio"] * 1000.0)) if i in (150, 151) else r
            for i, r in enumerate(rows)]
    got = T.measure(W, pair, 0)
    assert got["per_line_in_100"]["K1"] > 0 and got["separate_runs_of_positions_that_stop"] >= 1
    assert got["watches_sharing_no_report"] == len(range(0, got["positions"], 121))
