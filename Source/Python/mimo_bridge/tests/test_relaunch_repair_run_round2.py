# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Round-two review probes for commit 960ccfb1c, taken into the suite with the fixes of its findings R1 to R6.

Written by the independent round-two review of Oct 5, 2026 (docs/gain_of_function_v4/coordinator/
relaunch_repair_run_round2_review_2026-10-05.md, branch review/relaunch-repair-run-round2-2026-10-05, commit 5803e11d0)
and taken into the suite by the coordinator: the probes that recorded a failure as it stood now assert the repaired
behavior, its three strict-xfail probes pass and lost their marks, and tests cover the mutations no test caught (HF, HJ,
HL, H4, RB; PH is in tests/test_training_preflight.py). The real hourly watcher runs on temporary folders with a
simulated clock (the re-check's harness); the relaunch script's main() runs with every process call replaced (the
review's harness). Nothing here looks at, ends or starts a real trainer, supervisor or watcher, and nothing is written
outside pytest's temporary folder.

Times used below are production's, read on Oct 5, 2026 from H84_fresh_drivers_20260918_seg41o_1431 (the run of
checkpoint 481,800, which training is held at): its trainer resumed 479,350, wrote its first full stats record at model
step 479,400 167.48 s after its start and its newest at 481,800 6,705.6 s after its start (0.36706 model steps per
second), so it spent 31.3 s before its first model step.
"""
import importlib.util
import json
import math
import runpy
from pathlib import Path

import pytest

from mimo_bridge.tests.test_h84_hourly_watch import row
from mimo_bridge.tests.test_relaunch_repair_run_recheck import (CONTACT, SEG41C_ARRIVALS, WATCHER, arrivals,
                                                                 contact_records,
                                                                 watcher_started_before_the_first_record)
from mimo_bridge.tests.test_relaunch_repair_run_review import SCRIPT, main_under_probe, record
from mimo_bridge.tests.test_training_preflight import STOP_CRITERIA, _repair_setup
from mimo_bridge.tools import training_preflight as TP

RATE = 2400 / (6705.6 - 167.48)              # seg41o_1431: 479,400 -> 481,800
LOAD_S = 167.48 - 50 / RATE                   # its seconds before its first model step (479,350 -> 479,400 in 167.48 s)


def seg41o_start_run(tmp_path):
    """A start run with seg41o_1431's measured times; returns its checkpoint at 481,800."""
    run = tmp_path / "experiments" / "seg41o_like"
    ck = run / "checkpoints" / "ckpt_step_481800"
    ck.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps({"resume": "C:/x/checkpoints/ckpt_step_479350"}), encoding="utf-8")
    (run / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=step, elapsed_s=elapsed, causal_pe_per_modality=dict(vision=0.01))) + "\n"
        for step, elapsed in ((479400, 167.48), (481800, 6705.6))), encoding="utf-8")
    return ck


def repair_run_records(steps, start=481800, vision=0.1):
    """(seconds after the trainer's start, record) of a repair run from `start` at seg41o_1431's pace: one full stats
    record per 100 model steps, the first at the first model step divisible by 100 (train_mimo.py: step += 1, then
    flush_stats when step % report_every == 0)."""
    out = []
    for step in range(start + 100, start + steps + 1, 100):
        out.append((LOAD_S + (step - start) / RATE,
                    row(step=step, causal_pe_per_modality=dict(vision=vision, tactile=.04, proprioception=.005))))
    return out


def stop_read_at(at):
    """The model steps after the start at which the trainer reads a stop written `at` seconds after its start, at
    seg41o_1431's pace (it reads control.json when its model step is divisible by 25; 481,800 is)."""
    return 25 * math.ceil((at - LOAD_S) * RATE / 25)


# ------------------------------------------------------------------ R1: the first reading from the repair run's own record
def test_a_bound_the_watcher_first_judges_after_the_run_ends_is_refused(tmp_path, monkeypatch):
    # from 481,800 the first full stats record comes 100 model steps in; the watcher judges at most 330 s after it,
    # 122 model steps at seg41o_1431's pace; the trainer reads the stop within 25 more: 247
    ck = seg41o_start_run(tmp_path)
    block = dict(start_checkpoint=str(ck), stop_criteria=STOP_CRITERIA)
    for bound in (189, 200, 247):
        assert TP.first_reading_problems(dict(block, max_model_steps=bound)), bound            # refused
    assert TP.first_reading_problems(dict(block, max_model_steps=248)) == []                   # accepted
    # the run bounded at 248 model steps: records at 481,900 (303.7 s) and 482,000 (576.1 s), each reading vision causal
    # error 0.1 against the 0.04 stop line; the watcher finds the first at its 330 s look and judges at 630 s
    records = repair_run_records(248)
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "w", monkeypatch, records,
                                                            criteria=STOP_CRITERIA, hours=1)
    assert stop == {"cmd": "snapshot_and_stop"} and at == 630
    assert stop_read_at(at) <= 248                     # the trainer reads the stop before its bound (225 model steps in)
    # at 200 model steps, which the check of 960ccfb1c accepted, the run had ended (its last record 576.1 s) first
    assert round(repair_run_records(200)[-1][0], 1) == 576.1 and stop_read_at(at) > 200


def test_the_launch_commands_report_rate_reaches_the_first_reading_check(tmp_path):
    # --report-every is a RUN_FLAG (it does not change the fingerprint) and the relaunch's --set may change it
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    block = TP._load(TP.record_path(fp, root))["repair_run"]
    TP.set_repair_run(fp, dict(block, max_model_steps=600), "tester", root=root, registry=reg)
    for n in (4, 5, 7):
        TP.set_gate(fp, n, TP.REPAIR_TARGET, "fails today", ["row_%d" % n], "tester", root=root)
    # from 1,000 the trainer's first full stats record would be at 2,000; a run that ends at 1,600 writes none
    cmd = TP.apply_sets(recorded, [("--report-every", "1000"), ("--max-steps", "1600")])
    why = TP.launch_refusal(cmd, root=root, src=src, registry=reg)
    assert why and "up to 1146" in why
    assert TP.launch_refusal(TP.apply_sets(recorded, [("--max-steps", "1600")]), root=root, src=src,
                             registry=reg) is None                                              # at 100: judged


# ------------------------------------------------------------------ R3: a non-finite or never-written stop field
def test_a_stop_field_reading_infinity_or_nan_stops_the_run_at_the_first_check(tmp_path, monkeypatch):
    # json.dumps writes float("inf") as Infinity and json.loads reads it back; the window mean leaves non-finite
    # readings out, so at 960ccfb1c the field read as absent and was judged after the hour of grace (4,170 s)
    for name, value in (("inf", float("inf")), ("nan", float("nan"))):
        records = [(at, row(step=447400 + 100 * i, causal_pe_per_modality=dict(vision=value, tactile=.04,
                                                                               proprioception=.005)))
                   for i, at in enumerate(arrivals(until=1500))]
        stop, at, why = watcher_started_before_the_first_record(tmp_path / name, monkeypatch, records,
                                                                criteria=STOP_CRITERIA, hours=1)
        assert stop == {"cmd": "snapshot_and_stop"} and at == 270 + 300, (name, at, why)
        assert "causal_pe_per_modality.vision reads %s in the newest record" % name in why


def test_a_new_field_stop_line_is_judged_while_the_run_trains_or_the_bound_is_refused(tmp_path, monkeypatch):
    # a stop entry with new_field (a field only the repair's configuration writes) is not checked against the start
    # run's stats; when the run never writes it (a misspelled path, or the switch not taking effect), its absence is
    # judged after the hour of grace, so the preflight counts that hour into the first reading
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    block = TP._load(TP.record_path(fp, root))["repair_run"]
    new = dict(name="cap_share", path="attention_cap.capped_shre", op=">", line=0.5, from_run_hour=0, mode="stop",
               new_field="written only with the repair's switch on")
    why = TP.repair_block_problems(dict(block, stop_criteria=[new], max_model_steps=1000), reg)
    assert any("up to 1654" in p for p in why), why
    assert TP.repair_block_problems(dict(block, stop_criteria=[new], max_model_steps=1655), reg) == []
    # a run of 1,000 model steps: its 10 records span 2,452 s (303.7 to 2,755.5 s, the last at the bound), and the
    # field's absence stops nothing in it, while the real field (capped_share 0.9) was over the 0.5 line in every record
    records = [(at, dict(rec, attention_cap=dict(capped_share=0.9))) for at, rec in repair_run_records(1000)]
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "short", monkeypatch, records, criteria=[new],
                                                            hours=2)
    assert round(records[-1][0], 1) == 2755.5 and stop is None
    # a run of 1,657 model steps (the first reading at seg41o_1431's pace) is stopped on the absence while it trains
    records = [(at, dict(rec, attention_cap=dict(capped_share=0.9))) for at, rec in repair_run_records(1657)]
    stop, at, why = watcher_started_before_the_first_record(tmp_path / "long", monkeypatch, records, criteria=[new],
                                                            hours=2)
    assert stop == {"cmd": "snapshot_and_stop"} and "capped_shre is absent" in why
    assert stop_read_at(at) <= 1657


# ------------------------------------------------------------------ R2: a field absent for more than an hour
def test_a_state_dependent_field_absent_for_an_hour_stops_a_run_whose_line_holds(tmp_path, monkeypatch):
    # the pattern of vestibular.improvement_over_persistence in production segment seg41s_1146_resume_20261001_213502
    # (read Oct 5, 2026): present in its 3rd full stats record, absent from the 4th to the 19th (None while the
    # persistence error is under 1e-12, mimo_bridge/prediction_audit.py line 179), then present again. The watcher's
    # rule stands; the preflight refuses such a stop field (test_training_preflight.py,
    # test_a_stop_field_absent_for_an_hour_of_the_start_runs_records_is_refused)
    present = "00100000000000000001101111111110111111111100"
    entry = dict(name="vestibular_improvement", path="vestibular.improvement_over_persistence", op="<", line=-1e30,
                 from_run_hour=0, mode="stop")
    times = arrivals(first=(285, 513, 739), until=285 + 230 * (len(present) - 1))
    records = []
    for i, at in enumerate(times[:len(present)]):
        rec = row(step=451600 + 100 * i)
        if present[i] == "1":
            rec["vestibular"] = dict(improvement_over_persistence=0.25)
        records.append((at, rec))
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, records, criteria=[entry], hours=3)
    assert stop == {"cmd": "snapshot_and_stop"} and "is absent" in why and at == 4500


# ------------------------------------------------------------------ R4: the repair run is used once it trains
def _note_raising(module, monkeypatch, error):
    note0 = module.L.note

    def note(**kw):
        if str(kw.get("event", "")).startswith("relaunch: supervisor attached to the repair run"):
            raise error
        return note0(**kw)
    monkeypatch.setattr(module.L, "note", note)


def test_a_failed_note_after_the_attach_no_longer_ends_the_script_before_it_marks_the_run(tmp_path, monkeypatch):
    module, p = main_under_probe(tmp_path, monkeypatch, "stats")       # a full stats record exists at once
    _note_raising(module, monkeypatch, OSError("the launch receipt could not be written"))
    module.main()
    assert len(p.started) == 1 and record(p)["launched_at"]                     # trained, supervised, used once
    again = TP.apply_sets(p.launched["cmd"], [("--max-steps", "3000")])
    assert TP.launch_plan(again, root=p.root, src=p.src)["kind"] == "normal"


def test_a_repair_run_that_trained_is_used_once_whatever_ends_the_script(tmp_path, monkeypatch):
    # an interrupt after the attach (a closed window, a tool's time limit) ends the script before it marks the record;
    # the attempt it noted before the trainer started holds training past the start, so the record is used
    module, p = main_under_probe(tmp_path, monkeypatch, "stats")
    _note_raising(module, monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        module.main()
    rr = record(p)
    assert not rr["launched_at"] and rr["attempts"][0]["run"] == str(Path(p.launched["run"]).resolve())
    again = TP.apply_sets(p.launched["cmd"], [("--max-steps", "3000")])
    assert TP.launch_plan(again, root=p.root, src=p.src)["kind"] == "normal"
    why = TP.launch_refusal(again, root=p.root, src=p.src, registry=p.reg)
    assert why and "never normal training" in why


# ------------------------------------------------------------------ R5: a record marked by hand
def test_mark_launched_needs_the_repair_runs_existing_folder_and_states_then_continue_it(tmp_path, monkeypatch):
    # the relaunch's note when its own marking fails names `training_preflight launched FP --run RUN --by NAME`
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    repair = tmp_path / "experiments" / "run_repair"
    (repair / "checkpoints" / "ckpt_step_3000").mkdir(parents=True)
    (repair / "args.json").write_text(json.dumps({"resume": str(start)}), encoding="utf-8")
    with pytest.raises(TP.PreflightError, match="an existing folder"):
        TP.mark_launched(fp, "", "the coordinator", root=root)
    # a relative path is stored as the folder it names, so the repair run's own checkpoints are states
    monkeypatch.chdir(tmp_path)
    rr = TP.mark_launched(fp, "experiments/run_repair", "the coordinator", root=root)
    assert rr["run"] == str(repair.resolve())
    TP.add_state(fp, str(repair / "checkpoints" / "ckpt_step_3000"), "gates judged again", "tester", root=root)


# ------------------------------------------------------------------ R6: entries and errors the watcher misread
def test_a_stop_entry_with_an_op_other_than_under_or_over_stops_the_run_naming_it(tmp_path):
    ns = runpy.run_path(str(WATCHER))
    run = tmp_path / "run"
    run.mkdir()
    entry = dict(name="hand_written", path="causal_pe_per_modality.vision", op="<=", line=0.04, mode="stop")
    (run / "declared_criteria.json").write_text(json.dumps([entry]), encoding="utf-8")
    ns["declared_stops"].__globals__["RUN"] = str(run)
    window = [row(step=1000, causal_pe_per_modality=dict(vision=0.01))]          # 0.01 is under its 0.04 line
    why = ns["declared_stops"](window, 2.0, 7200.0)
    assert why and "declared stop criterion 1 cannot be judged" in why and "'<='" in why
    assert "declared stop criterion 1 cannot be judged" in ns["declared_criteria"](window, 2.0, 7200.0)


def test_one_failed_read_of_the_criteria_file_is_read_again_and_stops_nothing(tmp_path, monkeypatch):
    # a file another program holds for a moment: at 960ccfb1c one PermissionError stopped a run whose lines hold at
    # 570 s; it is now read again at the next 5-minute check
    import builtins
    real_open = builtins.open
    failed = []

    def open_once_refused(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        if str(file).endswith("declared_criteria.json") and mode == "r" and not failed:
            failed.append(str(file))
            raise PermissionError(13, "the file is held by another process", str(file))
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", open_once_refused)
    records = [(at, row(step=447400 + 100 * i)) for i, at in enumerate(SEG41C_ARRIVALS)]   # vision 0.01: holds
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, records, criteria=STOP_CRITERIA,
                                                            hours=1)
    assert failed and stop is None, why


def test_a_criteria_file_unreadable_at_two_checks_running_stops_the_run(tmp_path, monkeypatch):
    # mutations H4 (an unreadable file stops nothing at the 5-minute check) and HJ (a file that is not a list read as
    # one): a garbled file and a JSON object each stop the run at the second 5-minute check, 870 s
    records = [(at, row(step=447400 + 100 * i)) for i, at in enumerate(SEG41C_ARRIVALS)]
    for name, raw in (("garbled", b'[{"name": '), ("object", json.dumps(dict(STOP_CRITERIA[0])).encode("utf-8"))):
        stop, at, why = watcher_started_before_the_first_record(tmp_path / name, monkeypatch, records, raw_criteria=raw,
                                                                hours=1)
        assert stop == {"cmd": "snapshot_and_stop"} and at == 270 + 600, (name, at, why)
        assert "unreadable at 2 5-minute checks running" in why


def test_a_malformed_alert_entry_is_named_and_the_hourly_sweep_goes_on(tmp_path, monkeypatch, capsys):
    # at 960ccfb1c the sweep read float(c["line"]) of every entry unguarded and ended the watcher at each sweep
    bad = [dict(STOP_CRITERIA[0]), dict(name="hand_alert", path="cmd_response_cos_held", op="<", line="x",
                                        mode="alert")]
    records = [(at, row(step=447400 + 100 * i)) for i, at in enumerate(arrivals(until=3600))]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, records, criteria=bad, hours=2)
    assert stop is None, why
    assert "criterion 2 CANNOT BE JUDGED (ValueError" in capsys.readouterr().out


def test_an_error_in_the_declared_stops_stops_the_run_naming_it(tmp_path, monkeypatch):
    # mutation HL: without the guard in quick_breach the error ends the watcher, and the supervisor's next sweep starts
    # another, which ends the same way
    def broken(*a, **k):
        raise RuntimeError("an unforeseen entry")
    records = [(at, row(step=447400 + 100 * i)) for i, at in enumerate(SEG41C_ARRIVALS)]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, records, criteria=STOP_CRITERIA,
                                                            hours=1, patch={"declared_stops": broken})
    assert stop == {"cmd": "snapshot_and_stop"} and at == 270 + 300
    assert "the declared stop lines could not be judged (RuntimeError: an unforeseen entry)" in why


def test_the_hourly_sweep_alone_stops_on_a_stop_field_absent_for_an_hour(tmp_path, monkeypatch):
    # mutation HF: the sweep passes the span of the run's records to declared_criteria; with the 5-minute checks
    # replaced by none, a misspelled stop field is stopped at the second hourly sweep (6,870 s: the records then span
    # 6,443 s, the first sweep's 2,993 s)
    typo = [dict(CONTACT, path="contact_control.gate_self_maen")]
    stop, at, why = watcher_started_before_the_first_record(tmp_path, monkeypatch, contact_records(1), criteria=typo,
                                                            hours=2, patch={"quick_breach": lambda: None})
    assert stop == {"cmd": "snapshot_and_stop"} and "gate_self_maen is absent" in why and at == 6870


# ------------------------------------------------------------------ RB: the stop request before its note
def test_stop_trainer_writes_the_stop_request_when_its_note_fails(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("relaunch_round2_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def note(**kw):
        raise OSError("the launch receipt could not be written")
    monkeypatch.setattr(module.L, "note", note)
    run = tmp_path / "run"
    run.mkdir()
    module.stop_trainer(run, "no supervisor attached to the repair run")
    assert json.loads((run / "control.json").read_text(encoding="utf-8"))["cmd"] == "snapshot_and_stop"
