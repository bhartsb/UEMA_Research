# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The hourly watcher's planner monitor and an admitted hand calibration replacement (Oct 9, 2026; the matched comparison
of the finger fix, the operator's decision received by 11:53 PM PDT, Oct 8).

The planner contract a run inherits names the hand calibration its lineage was qualified with (seg37's continuation
receipt: the admitted fit, payload fe953beb...). The finger arm launches with --replace-saved-hand-calibration, and its
hand stage reports the refit's payload f68754c8...; before this rule the monitor held every run to the contract's hash
and would have stopped that arm at its first 5-minute check. The monitor now follows the contract's hash through the
replacements the trainer admitted and recorded: in the run's stats stream (type hand_calibration_replacement) and in
the model receipt of the checkpoint a later run resumes (hand_calibration_replacements). The control arm, which records
no replacement, is still held to the admitted fit, and any other hash still stops. Every folder here is temporary."""
import copy
import json
import runpy
from pathlib import Path

WATCH = runpy.run_path(str(Path(__file__).resolve().parents[4] / "docs/h84_launch_2026-09-18/hourly_watch.py"))

ADMITTED = "fe953beb426f2d403b58761d27b69ba2e3ec61e39b78d777fc2bea2b0fcece9e"   # hand_response_calibration_380200
REFIT = "f68754c866483a9e30318bf0697b08c450dd5ad53dd4d996861d67a6a4aeef3c"      # ..._finger_joint_servo_v1_481800
OTHER = "c" * 64


def hand_record(step, calibration):
    """One latched plan of 2 bodies with a scored body search and a scored hand search (test_planner_monitor's form)."""
    return {"type": "latch", "step": step, "plan": {"bodies": 2, "optimizer": [{
        "depth": 1, "selection": "best_evaluated_per_body_v1", "stopped_by": "step_budget",
        "per_body": {"objective": {"start": [2.0, 2.0], "selected": [1.9, 1.8]},
                     "goal": {"start": [1.0, 1.0], "selected": [0.9, 0.9]},
                     "effort": {"start": [1.0, 1.0], "selected": [0.8, 0.9]}}}],
        "hand_optimizer": dict(version="hand_joint_response_v1", goal="wrist_relative_finger_shape",
                               calibration_sha256=calibration, steps=20, horizon_ms=250, raw_joint_bound_deg=11.,
                               first=[2., 0.], selected=[1., 0.], valid=[True, False], selected_iteration=[1, 0],
                               trace=[[2., 0.]] + [[1., 0.]] * 20, stopped_by="completed")}}


ADMISSION = "C:/repo/docs/gain_of_function_v3/measurements/hand_choice_admission_record_481800.json"


def report(old=ADMITTED, new=REFIT, at=1791530000.0, checkpoint=None, admission=ADMISSION):
    """A replacement report in the trainer's form (LemLite.hand_calibration_replacement_report). checkpoint None: the
    checkpoint the run that records it resumed (arm fills it in, as the trainer writes its --resume there)."""
    return dict(setting="--replace-saved-hand-calibration", checkpoint=checkpoint,
                model_step=481800, model_sha256="4" * 64,
                replaced=dict(payload_sha256=old, calibration_file_sha256="9" * 64, command_meaning="pose_servo"),
                replaced_by=dict(payload_sha256=new, calibration_file_sha256="a" * 64,
                                 command_meaning="finger_joint_servo_v1", geometry_sha256="6" * 64),
                admission=dict(path=admission, file_sha256="0" * 64, decisions=16),
                launch=dict(finger_joint_servo="on", hand_goals="memory", hand_target_mode="wrist_relative_v1"),
                at=at, at_text="Oct 9, 2026 01:00:00 AM", world_geometry_sha256="6" * 64)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def lineage(tmp_path, *, hand_contract=True):
    """seg37 declares the contract (the admitted fit); seg41o continues it without a declaration and holds checkpoint
    481,800, whose receipt saved the admitted fit and records no replacement."""
    exp = tmp_path / "experiments"
    contract = dict(planner_selection_contract="best_evaluated_per_body_v1", planner_bodies=2)
    if hand_contract:
        contract.update(hand_selection_contract="hand_joint_response_v1", hand_calibration_sha256=ADMITTED)
    write(exp / "seg37" / "continuation_receipt.json", contract)
    write(exp / "seg41o" / "continuation_receipt.json", dict(parent_run=str(exp / "seg37")))
    start = exp / "seg41o" / "checkpoints" / "ckpt_step_481800"
    write(start / "model_receipt.json", dict(hand_response=dict(sha256=ADMITTED, calibration_file_sha256="9" * 64)))
    return exp, start


def arm(exp, name, resumed, reported, replacements=(), args=None):
    """A run that resumed `resumed` (args.json), names its parent run in its continuation receipt (as the relaunch
    writes it: no contract), records `replacements` in its stats stream and one hand search reporting `reported`. A run
    with replacements was launched with --replace-saved-hand-calibration ADMISSION, as the trainer's own records need;
    `args` overrides args.json."""
    run = exp / name
    write(run / "continuation_receipt.json", dict(parent_run=str(resumed.parent.parent), checkpoint=str(resumed)))
    settings = dict(resume=str(resumed), max_steps=484675)
    if replacements:
        settings["replace_saved_hand_calibration"] = ADMISSION
    settings.update(args or {})
    write(run / "args.json", settings)
    filled = []
    for r in replacements:
        r = copy.deepcopy(r)
        if isinstance(r, dict) and r.get("checkpoint") is None:
            r["checkpoint"] = str(resumed)
        filled.append(r)
    lines = [json.dumps(dict(type="hand_calibration_replacement", timestamp=1.0, step=481800, replacement=r))
             for r in filled] + [json.dumps(dict(type="mimo", step=481900, nan_steps=0))]
    write(run / "replay_training_stats.jsonl", "\n".join(lines) + "\n")
    write(run / "causal_chain.jsonl", json.dumps(hand_record(481810, reported)) + "\n")
    return run


def test_the_control_arm_is_held_to_the_admitted_fit(tmp_path):
    exp, start = lineage(tmp_path)
    result = WATCH["planner_monitor"](arm(exp, "control", start, ADMITTED))
    assert result["stop_reason"] is None and result["hand"]["body_comparisons"] == 1
    assert result["hand_calibration_expected"] == ADMITTED and result["hand_calibration_replacements_followed"] == []
    assert result["selection_contract_origin"] == str((exp / "seg37").resolve())
    # a control arm whose hand stage reported the refit, with no replacement recorded, is stopped as before
    result = WATCH["planner_monitor"](arm(exp, "control_refit", start, REFIT))
    assert result["status"] == "repair_required" and "lacks its qualified contract" in result["stop_reason"]
    assert json.loads((exp / "control_refit" / "planner_monitor.json").read_text())["status"] == "repair_required"


def test_the_finger_arm_follows_its_admitted_replacement(tmp_path):
    exp, start = lineage(tmp_path)
    result = WATCH["planner_monitor"](arm(exp, "finger", start, REFIT, [report()]))
    assert result["stop_reason"] is None and result["hand"]["searches"] == 1 and result["hand"]["improvements"] == 1
    assert result["hand_calibration_expected"] == REFIT
    followed = result["hand_calibration_replacements_followed"]
    assert len(followed) == 1 and followed[0]["replaced"] == ADMITTED and followed[0]["replaced_by"] == REFIT
    assert followed[0]["admission"].endswith("hand_choice_admission_record_481800.json")
    # the trainer recorded the replacement, but the hand stage still reports the old fit (the episode companion's
    # restore that LemLite._keep_hand_calibration_replacement undoes): stopped
    result = WATCH["planner_monitor"](arm(exp, "finger_old", start, ADMITTED, [report()]))
    assert result["status"] == "repair_required" and "lacks its qualified contract" in result["stop_reason"]


def test_a_hash_that_neither_the_contract_nor_an_admitted_replacement_names_stops(tmp_path):
    exp, start = lineage(tmp_path)
    for name, replacements in (("finger_other", [report()]), ("control_other", [])):
        result = WATCH["planner_monitor"](arm(exp, name, start, OTHER, replacements))
        assert result["status"] == "repair_required" and "lacks its qualified contract" in result["stop_reason"], name


def test_a_replacement_of_another_calibration_or_a_malformed_one_stops(tmp_path):
    exp, start = lineage(tmp_path)
    result = WATCH["planner_monitor"](arm(exp, "elsewhere", start, REFIT, [report(old=OTHER)]))
    assert result["status"] == "repair_required" and result["hand_calibration_expected"] == ADMITTED
    assert result["hand_calibration_replacement_error"] == (
        "A hand calibration replacement replaces payload %s, but the hand contract expects payload %s at that point."
        % (OTHER, ADMITTED))
    assert result["stop_reason"].startswith("A hand calibration replacement replaces payload")
    broken = report()
    del broken["replaced_by"]
    result = WATCH["planner_monitor"](arm(exp, "broken", start, REFIT, [broken]))
    assert result["status"] == "repair_required" and "lacks its replaced or replacing" in result["stop_reason"]
    result = WATCH["planner_monitor"](arm(exp, "empty", start, REFIT, [None]))
    assert result["status"] == "repair_required" and "holds no replacement report" in result["stop_reason"]


def test_a_stream_record_the_runs_own_launch_does_not_name_stops(tmp_path):
    # the independent review of Oct 9, 2026, finding F1, probe W1: a well-formed record was followed whoever wrote it
    exp, start = lineage(tmp_path)
    forged = report(new=OTHER, admission="Z:/no/such/admission_record.json", checkpoint="Z:/elsewhere")
    result = WATCH["planner_monitor"](arm(exp, "no_flag", start, OTHER, [forged],
                                          args=dict(replace_saved_hand_calibration="")))
    assert result["status"] == "repair_required" and result["hand_calibration_expected"] == ADMITTED
    assert result["hand_calibration_replacements_followed"] == []
    assert result["stop_reason"].startswith("The run's stats stream holds a hand_calibration_replacement record, but "
                                            "its args.json names no --replace-saved-hand-calibration.")
    # the launch named another admission record than the one the record names
    result = WATCH["planner_monitor"](arm(exp, "other_admission", start, OTHER,
                                          [report(new=OTHER, admission="Z:/no/such/admission_record.json")]))
    assert result["status"] == "repair_required" and "names the admission record Z:/no/such" in result["stop_reason"]
    # the record replaced the calibration of another checkpoint than the run resumed
    result = WATCH["planner_monitor"](arm(exp, "other_checkpoint", start, REFIT, [report(checkpoint="Z:/elsewhere")]))
    assert result["status"] == "repair_required" and "replaced the calibration of checkpoint Z:/elsewhere" in \
        result["stop_reason"]
    # the trainer's own form, with the same paths spelled with the other separators: followed
    spelled = report(admission=ADMISSION.replace("/", "\\"), checkpoint=str(start).replace("\\", "/"))
    result = WATCH["planner_monitor"](arm(exp, "spelled", start, REFIT, [spelled]))
    assert result["stop_reason"] is None and result["hand_calibration_expected"] == REFIT


def test_a_resumed_receipt_whose_list_does_not_end_at_its_own_saved_calibration_stops(tmp_path):
    # finding F1, probe W2: a receipt listing a replacement to a third hash while its own hand_response still names the
    # admitted fit was followed
    exp, start = lineage(tmp_path)
    write(start / "model_receipt.json", dict(hand_response=dict(sha256=ADMITTED, calibration_file_sha256="9" * 64),
                                             hand_calibration_replacements=[report(new=OTHER)]))
    result = WATCH["planner_monitor"](arm(exp, "receipt_only", start, OTHER))
    assert result["status"] == "repair_required" and result["hand_calibration_expected"] == ADMITTED
    assert ("lists hand calibration replacements whose last replacing payload is %s, but its own saved hand "
            "calibration is payload %s" % (OTHER, ADMITTED)) in result["stop_reason"]
    # a list whose last entry cannot be read, or a receipt with no saved calibration beside its list, stops too
    for name, saved in (("unreadable_last", dict(hand_response=dict(sha256=REFIT), hand_calibration_replacements=[
            report(), dict(replaced=None)])),
                        ("no_saved", dict(hand_calibration_replacements=[report()]))):
        checkpoint = exp / "seg41o" / "checkpoints" / ("ckpt_step_%s" % name)
        write(checkpoint / "model_receipt.json", saved)
        result = WATCH["planner_monitor"](arm(exp, "receipt_" + name, checkpoint, REFIT))
        assert result["status"] == "repair_required" and "lists hand calibration replacements" in \
            result["stop_reason"], name


def test_a_later_resume_follows_the_replacement_its_checkpoint_keeps(tmp_path):
    # a continuation of the finger arm: no replacement at its own launch (the checkpoint saved the refit already); the
    # resumed checkpoint's model_receipt.json keeps the replacement (LemLite._restore_hand_response_calibration)
    exp, start = lineage(tmp_path)
    finger = arm(exp, "finger", start, REFIT, [report()])
    saved = finger / "checkpoints" / "ckpt_step_484675"
    write(saved / "model_receipt.json", dict(hand_response=dict(sha256=REFIT, calibration_file_sha256="a" * 64),
                                             hand_calibration_replacements=[report()]))
    result = WATCH["planner_monitor"](arm(exp, "finger_next", saved, REFIT))
    assert result["stop_reason"] is None and result["hand_calibration_expected"] == REFIT
    result = WATCH["planner_monitor"](arm(exp, "finger_next_old", saved, ADMITTED))
    assert result["status"] == "repair_required" and "lacks its qualified contract" in result["stop_reason"]
    # the same replacement read from the resumed receipt and from the stream counts once
    result = WATCH["planner_monitor"](arm(exp, "finger_twice", saved, REFIT, [report()]))
    assert result["stop_reason"] is None and len(result["hand_calibration_replacements_followed"]) == 1
    # a second admitted replacement chains from the first
    result = WATCH["planner_monitor"](arm(exp, "finger_third", saved, OTHER, [report(old=REFIT, new=OTHER, at=2.0)]))
    assert result["stop_reason"] is None and result["hand_calibration_expected"] == OTHER


def test_an_unreadable_resumed_receipt_is_a_note_and_the_contract_still_holds(tmp_path):
    exp, start = lineage(tmp_path)
    bare = exp / "seg41o" / "checkpoints" / "ckpt_step_481700"
    bare.mkdir(parents=True)
    result = WATCH["planner_monitor"](arm(exp, "control_bare", bare, ADMITTED))
    assert result["stop_reason"] is None and result["hand_calibration_expected"] == ADMITTED
    assert "cannot be read" in result["hand_calibration_replacement_notes"][0]
    result = WATCH["planner_monitor"](arm(exp, "finger_bare", bare, REFIT, [report()]))
    assert result["stop_reason"] is None and result["hand_calibration_expected"] == REFIT


def test_a_run_without_a_hand_contract_reads_no_replacement(tmp_path):
    exp, start = lineage(tmp_path, hand_contract=False)
    run = arm(exp, "no_hand_contract", start, OTHER, [report(old=OTHER)])
    result = WATCH["planner_monitor"](run)
    assert result["stop_reason"] is None and result["hand_calibration_expected"] is None
    assert result["hand_calibration_replacements_followed"] == [] and "hand_calibration_replacement_notes" not in result
    assert not result["hand_selection_contract_required"]


def test_the_dry_runs_recorded_decisions_pass_with_its_replacement_record_and_stop_without_it(tmp_path):
    # the learning-off dry run's run B (row activation_dry_run_481800, PASS at 12:10 AM PDT, Oct 9): its causal log
    # (a hand search over 8 bodies at model step 20 reporting the refit) and the replacement record its trainer wrote,
    # copied unchanged into finger_dry_run_decisions_481800.json. Its item W1 read planner_integrity by hand: a stop
    # under seg37's hash, none under the refit's; the monitor now reaches the second through the record alone
    fixture = json.loads((Path(__file__).with_name("finger_dry_run_decisions_481800.json")).read_text(encoding="utf-8"))
    record = fixture["hand_calibration_replacement"]
    assert record["replacement"]["replaced"]["payload_sha256"] == ADMITTED
    assert record["replacement"]["replaced_by"]["payload_sha256"] == REFIT
    exp = tmp_path / "experiments"
    write(exp / "seg37" / "continuation_receipt.json", dict(
        planner_selection_contract="best_evaluated_per_body_v1", planner_bodies=8,
        hand_selection_contract="hand_joint_response_v1", hand_calibration_sha256=ADMITTED))
    write(exp / "seg41o" / "continuation_receipt.json", dict(parent_run=str(exp / "seg37")))
    start = exp / "seg41o" / "checkpoints" / "ckpt_step_481800"
    write(start / "model_receipt.json", dict(hand_response=dict(sha256=ADMITTED, calibration_file_sha256="9" * 64)))
    for name, lines in (("finger", [record]), ("finger_without_record", [])):
        run = exp / name
        write(run / "continuation_receipt.json", dict(parent_run=str(exp / "seg41o"), checkpoint=str(start)))
        # run B's own launch settings, which the trainer wrote into the record: the checkpoint it resumed and the
        # admission record it was launched with (finding F1 checks the record against them)
        write(run / "args.json", dict(resume=record["replacement"]["checkpoint"], max_steps=26,
                                      replace_saved_hand_calibration=record["replacement"]["admission"]["path"]))
        write(run / "replay_training_stats.jsonl", "".join(json.dumps(r) + "\n" for r in lines))
        write(run / "causal_chain.jsonl", "".join(json.dumps(r) + "\n" for r in fixture["causal_chain"]))
    result = WATCH["planner_monitor"](exp / "finger")
    assert result["stop_reason"] is None and result["status"] == "search_checked"
    assert result["hand"]["searches"] == 1 and result["hand"]["malformed_results"] == 0
    assert result["body_comparisons"] == 8 and result["hand_calibration_expected"] == REFIT
    result = WATCH["planner_monitor"](exp / "finger_without_record")
    assert result["stop_reason"] == ("Hand search at step 20 lacks its qualified contract or complete body "
                                     "comparisons. Save and repair required.")
    assert result["status"] == "repair_required" and result["hand"]["malformed_results"] == 1


def test_the_rule_itself_on_plain_lists():
    expected = WATCH["expected_hand_calibration"]
    assert expected(None, [report()]) == (None, [], None)
    assert expected(ADMITTED, []) == (ADMITTED, [], None)
    got, followed, problem = expected(ADMITTED, [report(), copy.deepcopy(report())])
    assert (got, len(followed), problem) == (REFIT, 1, None)
    got, followed, problem = expected(ADMITTED, [report(old=REFIT, new=OTHER)])
    assert got == ADMITTED and followed == [] and "expects payload %s" % ADMITTED in problem
