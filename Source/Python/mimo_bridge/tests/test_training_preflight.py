# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The training preflight (operator, Oct 5, 2026): a new or changed training configuration launches only when its seven
gates read PASS, or N/A with a reason and evidence; a qualified configuration continues without a new record; the
refusal names every gate that blocks and why."""
import json
import time
from pathlib import Path

import pytest

from mimo_bridge.tools import training_preflight as TP


# the flags the stand-in trainer's parser declares (training_preflight.trainer_flags reads them from its source)
FLAGS = ("--intention", "--fill-ring-first", "--resume", "--out", "--report-every", "--vision-input-calibration",
         "--planner-search", "--handle-lesson", "--max-steps", "--checkpoint-storage", "--heads-all-slot-path",
         "--attention-cap", "--attention-cap-bound", "--latent-body-hierarchy", "--latent-body-resume",
         "--latent-body-output")
TRAINER = ("import argparse\n\n\ndef parse_args(argv=None):\n    ap = argparse.ArgumentParser()\n"
           + "".join('    ap.add_argument("%s")\n' % f for f in FLAGS) + "    return ap.parse_args(argv)\n")


def _src(tmp_path):
    src = tmp_path / "src"
    for rel, text in (("lem/model/core.py", "x = 1\n"), ("mimo_bridge/train_mimo.py", TRAINER),
                      ("mimo_bridge/tools/dash.py", "z = 3\n"), ("mimo_bridge/tests/test_a.py", "t = 4\n")):
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return src


def _cmd(calib=None, **extra):
    cmd = ["python", "-u", "-m", "mimo_bridge.train_mimo", "--intention", "on", "--fill-ring-first",
           "--resume", "C:/runs/a/ckpt_1", "--out", "C:/runs/a", "--report-every", "100"]
    if calib:
        cmd += ["--vision-input-calibration", str(calib)]
    for k, v in extra.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return cmd


def _all_gates(fp, root, status="PASS"):
    for n, _name, _q in TP.GATES:
        TP.set_gate(fp, n, status, "reason %d" % n, ["row_%d" % n], "tester", root=root)


def _run(tmp_path, name, steps, resume=None):
    """A run folder with checkpoints ckpt_step_<n> and an args.json naming the checkpoint it resumed from."""
    run = tmp_path / "experiments" / name
    for n in steps:
        (run / "checkpoints" / ("ckpt_step_%d" % n)).mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text('{"resume": %s}' % ('"%s"' % str(resume).replace("\\", "/") if resume else "null"),
                                   encoding="utf-8")
    return run


def _at(cmd, checkpoint):
    cmd = list(cmd)
    cmd[cmd.index("--resume") + 1] = str(checkpoint)
    return cmd


def test_a_qualified_configuration_continues_and_per_run_flags_do_not_count(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    a = _run(tmp_path, "run_a", (100, 200))
    fp = Path(TP.new_record(_at(_cmd(), a / "checkpoints/ckpt_step_100"), "qualified once", "tester",
                            root=root, src=src)).stem
    _all_gates(fp, root)
    moved = _at(_cmd(), a / "checkpoints/ckpt_step_200")         # a later checkpoint of the same run
    moved[moved.index("--out") + 1] = "C:/runs/b"
    moved[moved.index("--report-every") + 1] = "50"
    assert TP.launch_refusal(moved, root=root, src=src) is None


def test_a_checkpoint_of_another_branch_or_an_earlier_one_needs_the_state_gates_reconsidered(tmp_path):
    # the operator, Oct 5, 2026: "a preflight from one lineage/state must not automatically authorize an arbitrary
    # different checkpoint"
    src, root = _src(tmp_path), tmp_path / "pre"
    a = _run(tmp_path, "run_a", (100, 200, 300))
    child = _run(tmp_path, "run_child", (350,), resume=a / "checkpoints/ckpt_step_200")
    branch = _run(tmp_path, "run_branch", (400,), resume=a / "checkpoints/ckpt_step_100")
    other = _run(tmp_path, "run_other", (500,))
    fp = Path(TP.new_record(_at(_cmd(), a / "checkpoints/ckpt_step_200"), "x", "tester", root=root, src=src)).stem
    _all_gates(fp, root)
    assert TP.launch_refusal(_at(_cmd(), child / "checkpoints/ckpt_step_350"), root=root, src=src) is None
    for ck in (a / "checkpoints/ckpt_step_100", branch / "checkpoints/ckpt_step_400", other / "checkpoints/ckpt_step_500"):
        why = TP.launch_refusal(_at(_cmd(), ck), root=root, src=src)
        assert why.startswith("BLOCKED") and "does not continue them" in why and "5 Causal controllability" in why
    TP.add_state(fp, str(branch / "checkpoints/ckpt_step_400"), "gates 1-5 and 7 read again on this branch", "tester",
                 root=root)
    assert TP.launch_refusal(_at(_cmd(), branch / "checkpoints/ckpt_step_400"), root=root, src=src) is None


def test_the_reference_configuration_gives_no_permission_to_launch(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    TP.reference(_cmd(), "held when the rule was adopted", "tester", root=root, src=src)
    why = TP.launch_refusal(_cmd(), root=root, src=src)
    assert why.startswith("BLOCKED") and "never preflighted" in why


def test_a_changed_flag_without_a_record_is_refused_and_the_difference_is_named(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    TP.reference(_cmd(), "held at adoption", "tester", root=root, src=src)
    why = TP.launch_refusal(_cmd(planner_search="trust_v2"), root=root, src=src)
    assert why.startswith("BLOCKED") and "no preflight record" in why
    assert "flag --planner-search: absent -> trust_v2" in why


def test_a_change_of_model_code_needs_a_record_and_a_change_of_a_tool_does_not(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    TP.reference(_cmd(), "held at adoption", "tester", root=root, src=src)
    (src / "mimo_bridge/tools/dash.py").write_text("z = 30\n", encoding="utf-8")
    (src / "mimo_bridge/tests/test_a.py").write_text("t = 40\n", encoding="utf-8")
    assert TP.configuration(_cmd(), src)["fingerprint"] == TP._load(root / TP.BASELINE)["fingerprint"]
    (src / "lem/model/core.py").write_text("x = 10\n", encoding="utf-8")
    why = TP.launch_refusal(_cmd(), root=root, src=src)
    assert "code: 1 file(s) differ: lem/model/core.py" in why


def test_a_changed_file_named_by_a_flag_changes_the_configuration(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    calib = tmp_path / "calib.json"
    calib.write_text('{"a": 1}', encoding="utf-8")
    TP.reference(_cmd(calib), "held at adoption", "tester", root=root, src=src)
    calib.write_text('{"a": 2}', encoding="utf-8")
    assert "the file named by --vision-input-calibration changed" in TP.launch_refusal(_cmd(calib), root=root, src=src)


def test_every_gate_that_blocks_is_named_with_its_reason(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    cmd = _cmd(planner_search="trust_v2")
    path = TP.new_record(cmd, "the repaired search on", "tester", root=root, src=src)
    fp = Path(path).stem
    why = TP.launch_refusal(cmd, root=root, src=src)
    assert "7 of 7 gates block launch" in why and "gate 1 Information flow: OPEN - no result recorded" in why
    _all_gates(fp, root)
    TP.set_gate(fp, 5, "FAIL", "19 of 26 handles' predictions miss their own consequence",
                ["investigate_motor_handle_reach_481800"], "tester", root=root)
    why = TP.launch_refusal(cmd, root=root, src=src)
    assert "1 of 7 gates block launch" in why
    assert ("gate 5 Causal controllability: FAIL - 19 of 26 handles' predictions miss their own consequence "
            "(evidence: investigate_motor_handle_reach_481800)") in why


def test_na_needs_a_reason_and_evidence_and_pass_needs_evidence(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    cmd = _cmd(planner_search="trust_v2")
    fp = Path(TP.new_record(cmd, "x", "tester", root=root, src=src)).stem
    with pytest.raises(TP.PreflightError):
        TP.set_gate(fp, 3, "N/A", "", ["row"], "tester", root=root)
    with pytest.raises(TP.PreflightError):
        TP.set_gate(fp, 3, "N/A", "the change does not reach scaling", [], "tester", root=root)
    with pytest.raises(TP.PreflightError):
        TP.set_gate(fp, 3, "PASS", "measured", [], "tester", root=root)


def test_a_complete_record_allows_the_launch(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    cmd = _cmd(planner_search="trust_v2")
    fp = Path(TP.new_record(cmd, "x", "tester", root=root, src=src)).stem
    _all_gates(fp, root, "PASS")
    TP.set_gate(fp, 3, "N/A", "the change does not reach any constant or scale", ["commit abc"], "tester", root=root)
    assert TP.launch_refusal(cmd, root=root, src=src) is None


def test_without_a_reference_or_a_record_every_launch_is_refused(tmp_path):
    src, root = _src(tmp_path), tmp_path / "pre"
    why = TP.launch_refusal(_cmd(), root=root, src=src)
    assert why.startswith("BLOCKED") and "No reference configuration is recorded" in why


def test_set_flags_as_the_relaunch_script_applies_them():
    cmd = TP.apply_sets(_cmd(), [("planner-search", "trust_v2"), ("--intention", "off")])
    flags = TP.flags_of(cmd)
    assert flags["--planner-search"] == "trust_v2" and flags["--intention"] == "off"
    assert flags["--fill-ring-first"] is True


# THE REPAIR RUN (the operator, Oct 5, 2026): "We want the preflight to prevent wasting training, not prevent the
# controlled training needed to repair a known failure."
def _registry(tmp_path, rows=("stop_row",)):
    p = tmp_path / "registry.json"
    p.write_text('{"rows": [%s]}' % ", ".join('{"id": "%s", "status": "written"}' % r for r in rows), encoding="utf-8")
    return p


STOP_CRITERIA = [dict(name="vision_causal_error", path="causal_pe_per_modality.vision", op=">", line=0.04,
                      from_run_hour=0, mode="stop", note="S1"),
                 dict(name="following", path="cmd_response_cos_held", op="<", line=0.3, mode="alert", note="S2")]


def _repair_setup(tmp_path, **extra):
    """A repair run's record (target gates 4, 5 and 7) from run_a's ckpt_step_1000, at most 2,000 model steps; `extra`
    adds configuration flags to the command (_cmd)."""
    src, root, reg = _src(tmp_path), tmp_path / "pre", _registry(tmp_path)
    a = _run(tmp_path, "run_a", (1000,))
    start = a / "checkpoints/ckpt_step_1000"
    # the start run's full stats records: the newest carries every stop criterion's field (training_preflight
    # .stop_field_problems); the first came 167 s after its start, then 100 model steps per 273 s, as seg41o_1431's
    # (training_preflight.first_reading_problems: the watcher's first reading about 188 model steps in)
    (a / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=step, elapsed_s=elapsed, causal_pe_per_modality=dict(vision=0.01),
        cmd_response_cos_held=0.5)) + "\n" for step, elapsed in ((900, 167.0), (1000, 440.0))), encoding="utf-8")
    cmd = _at(_cmd(handle_lesson="on", **extra), start)
    fp = Path(TP.new_record(cmd, "repair the handles' direction", "tester", root=root, src=src)).stem
    for n in (1, 2, 3, 6):
        TP.set_gate(fp, n, "PASS", "measured", ["row_%d" % n], "tester", root=root)
    block = dict(purpose="19 of 26 handles' predictions miss their own consequence (gates 4, 5, 7)",
                 target_gates=[4, 5, 7], repairs=["a per-handle lesson"], affected_components=["TRANSFORMER"],
                 start_checkpoint=str(start), rollback_checkpoint=str(start), stop_lines="S1 S2 S3",
                 stop_lines_row="stop_row", stop_criteria=STOP_CRITERIA, max_model_steps=2000,
                 reachability="the first 5-minute check reads the stop lines about 330 model steps after the start",
                 no_auto_continue="the supervisor stays paused", remeasure={"4": "row_4b", "5": "row_5b", "7": "row_7b"})
    TP.set_repair_run(fp, block, "tester", root=root, registry=reg)
    for n in (4, 5, 7):
        TP.set_gate(fp, n, TP.REPAIR_TARGET, "fails today", ["row_%d" % n], "tester", root=root)
    return src, root, reg, a, start, cmd, fp


def _bounded(cmd, end):
    return TP.apply_sets(cmd, [("--max-steps", str(end))])


def test_a_repair_run_launches_with_its_target_gates_failed_only_as_declared(tmp_path):
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    plan = TP.launch_plan(cmd, root=root, src=src)
    assert plan["kind"] == "repair_run" and plan["max_steps"] == 3000 and plan["stop_criteria"] == STOP_CRITERIA
    assert TP.launch_refusal(_bounded(cmd, 3000), root=root, src=src, registry=reg) is None
    why = TP.launch_refusal(cmd, root=root, src=src, registry=reg)                 # --max-steps absent: unbounded
    assert "not bounded as declared" in why
    assert "not bounded" in TP.launch_refusal(_bounded(cmd, 3001), root=root, src=src, registry=reg)
    TP.set_gate(fp, 6, "FAIL", "a stop line on an internal statistic", ["row_6"], "tester", root=root)
    why = TP.launch_refusal(_bounded(cmd, 3000), root=root, src=src, registry=reg)
    assert "gate 6 Objective/test alignment: FAIL" in why and "does not target" in why


def test_a_repair_run_is_refused_without_its_declared_stop_lines_and_cannot_target_a_gate_it_does_not_name(tmp_path):
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    with pytest.raises(TP.PreflightError):
        TP.set_gate(fp, 6, TP.REPAIR_TARGET, "x", ["e"], "tester", root=root)
    empty = _registry(tmp_path, rows=())
    why = TP.launch_refusal(_bounded(cmd, 3000), root=root, src=src, registry=empty)
    assert "stop lines' row stop_row is not in the test registry" in why
    with pytest.raises(TP.PreflightError):                         # a re-measurement named for every target gate
        TP.set_repair_run(fp, dict(TP._load(TP.record_path(fp, root))["repair_run"], remeasure={"4": "r"}), "tester",
                          root=root, registry=reg)


def test_a_repair_runs_stop_lines_must_be_criteria_the_watcher_can_judge_and_one_must_stop(tmp_path):
    # the watcher (docs/h84_launch_2026-09-18/hourly_watch.py, declared_criteria) judges entries {name, path, op, line,
    # from_run_hour, mode}; a stop line it cannot read never stops the run
    assert TP.stop_criteria_problems(STOP_CRITERIA) == []
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    block = TP._load(TP.record_path(fp, root))["repair_run"]
    for bad, expected in (
            ([], "stop criteria is missing"),
            ("vision over 0.04", "must be a list"),
            ([dict(STOP_CRITERIA[0], mode="alert")], "no stop criterion has mode stop"),
            ([dict(STOP_CRITERIA[0], op=">=")], "op must be < or >"),
            ([dict(STOP_CRITERIA[0], line="0.04")], "line must be a number"),
            ([dict(STOP_CRITERIA[0], line=float("nan"))], "line must be a number"),
            ([dict(STOP_CRITERIA[0], path="")], "has no path"),
            ([dict(STOP_CRITERIA[0], from_run_hour=-1)], "from_run_hour must be a number"),
            ([dict(STOP_CRITERIA[0], mode="stop_now")], "mode must be alert or stop")):
        problems = TP.repair_block_problems(dict(block, stop_criteria=bad), reg)
        assert any(expected in p for p in problems), (bad, problems)
    # the record's own block is checked again at launch: a block edited by hand after it was written is refused
    rec = TP._load(TP.record_path(fp, root))
    rec["repair_run"]["stop_criteria"] = [dict(STOP_CRITERIA[1])]
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(cmd, 3000), root=root, src=src, registry=reg)
    assert "no stop criterion has mode stop" in why


def test_after_the_repair_run_normal_training_needs_the_target_gates_measured_again(tmp_path):
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    child = _run(tmp_path, "run_repair", (3000,), resume=start)
    TP.mark_launched(fp, str(child), "tester", root=root)
    later = _at(cmd, child / "checkpoints/ckpt_step_3000")
    assert TP.launch_plan(later, root=root, src=src)["kind"] == "normal"
    why = TP.launch_refusal(later, root=root, src=src, registry=reg)
    assert "gate 5 Causal controllability: FAIL - REPAIR TARGET" in why and "never normal training" in why
    # the same repair run cannot be launched a second time
    assert TP.launch_plan(_bounded(cmd, 3000), root=root, src=src)["kind"] == "normal"
    for n in (4, 5, 7):
        TP.set_gate(fp, n, "PASS", "measured again after the repair run", ["row_%db" % n], "tester", root=root)
    # the gates were judged again on the repaired model: only a state added after the launch, the checkpoint they were
    # judged on, authorizes normal training (the relaunch review of Oct 5, 2026, finding F9)
    why = TP.launch_refusal(later, root=root, src=src, registry=reg)
    assert why.startswith("BLOCKED") and "ran its repair run" in why and "not one" in why
    TP.add_state(fp, str(child / "checkpoints/ckpt_step_3000"), "gates 4, 5 and 7 judged again on it", "tester",
                 root=root)
    assert TP.launch_refusal(later, root=root, src=src, registry=reg) is None
    assert "ran its repair run" in TP.launch_refusal(cmd, root=root, src=src, registry=reg)    # the start: refused
    rec = TP._load(TP.record_path(fp, root))
    for g in rec["gates"]:
        if g["n"] == 7:
            g["checked_at_s"] = rec["repair_run"]["launched_at_s"] - 1        # judged before the run: not again
    TP._save(TP.record_path(fp, root), rec)
    assert "must be measured again after that run" in TP.launch_refusal(later, root=root, src=src, registry=reg)


# EXACT FLAGS (the relaunch review of Oct 5, 2026, finding F8): argparse reads a unique prefix of a flag, so the
# preflight refuses a command whose flags the trainer would read otherwise than the preflight does
def test_a_flag_the_trainer_would_read_as_another_is_refused(tmp_path):
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    for extra, expected in ((["--max-step", "0"], "--max-step is not a flag of the trainer's parser (the parser would "
                                                  "read it as --max-steps)"),
                            (["--max-steps=0"], "--max-steps=0: write the flag and its value as two arguments")):
        why = TP.launch_refusal(_bounded(cmd, 3000) + extra, root=root, src=src, registry=reg)
        assert why.startswith("BLOCKED") and expected in why, why
        with pytest.raises(TP.PreflightError):
            TP.new_record(cmd + extra, "x", "tester", root=tmp_path / "other", src=src)
    assert TP.launch_refusal(_bounded(cmd, 3000), root=root, src=src, registry=reg) is None


def test_the_real_trainers_flags_are_read_with_the_parsers_it_imports():
    flags = TP.trainer_flags()
    # the trainer's own parser and latent_body_shadow.add_arguments, which parse_args calls with it
    assert {"--max-steps", "--resume", "--out", "--latent-body-hierarchy", "--latent-body-log-mb"} <= flags
    assert TP.flag_problems(["python", "-m", "mimo_bridge.train_mimo", "--max-steps", "5", "--latent-body-hierarchy",
                             "shadow"]) == []
    assert TP.flag_problems(["python", "-m", "mimo_bridge.train_mimo", "--max-step", "0"]) == [
        "--max-step is not a flag of the trainer's parser (the parser would read it as --max-steps)"]


def test_a_trainer_source_that_declares_no_flag_refuses_every_launch(tmp_path):
    src = _src(tmp_path)
    (src / "mimo_bridge/train_mimo.py").write_text("y = 2\n", encoding="utf-8")
    assert "cannot be read" in TP.launch_refusal(_cmd(), root=tmp_path / "pre", src=src)


def test_a_stop_criterion_whose_field_the_start_run_does_not_carry_needs_its_reason(tmp_path):
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    block = TP._load(TP.record_path(fp, root))["repair_run"]
    typo = dict(block, stop_criteria=[dict(STOP_CRITERIA[0], path="causal_pe_per_modality.visoin")])
    assert any("causal_pe_per_modality.visoin reads no number" in p for p in TP.repair_block_problems(typo, reg))
    new = dict(block, stop_criteria=[dict(STOP_CRITERIA[0], path="handle_lesson.loss",
                                          new_field="written only with --handle-lesson on")])
    assert TP.repair_block_problems(new, reg) == []
    assert any("new_field, when given" in p for p in TP.repair_block_problems(
        dict(block, stop_criteria=[dict(STOP_CRITERIA[0], new_field=" ")]), reg))
    assert any("reachability is missing" in p for p in TP.repair_block_problems(dict(block, reachability=""), reg))


def test_a_repair_run_that_ends_before_its_stop_lines_are_first_read_is_refused(tmp_path):
    # the re-check of Oct 5, 2026, finding N2, and the round-two review's R1: from model step 1,000 the first full stats
    # record comes at 1,100 (report every 100); the watcher judges 330 s after it at most, 121 model steps at the start
    # run's fastest pace (100 per 273 s); the trainer reads the stop within 25 more: 246 model steps in
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    block = TP._load(TP.record_path(fp, root))["repair_run"]
    assert TP.first_reading_problems(dict(block, max_model_steps=247)) == []
    why = TP.first_reading_problems(dict(block, max_model_steps=246))
    assert why and "before the watcher first judges stop line vision_causal_error" in why[0] and "up to 246" in why[0]
    # the launch command's report rate: with --report-every 1000 the first record would come 1,000 model steps in
    assert "up to 1146" in TP.first_reading_problems(dict(block, max_model_steps=600), report_every=1000)[0]
    refused = TP.launch_refusal(TP.apply_sets(cmd, [("--report-every", "1000"), ("--max-steps", "1600")]), root=root,
                                src=src, registry=reg)
    assert refused and "up to 1146" in refused
    # a stop line from run-hour 1 (the record that crosses the hour comes up to 100 model steps after it), and one whose
    # field the start run's stats never showed (the hour of grace after the first record, plus that record interval)
    late = dict(block, stop_criteria=[dict(STOP_CRITERIA[0], from_run_hour=1)], max_model_steps=1000)
    assert "up to 1554" in TP.first_reading_problems(late)[0]
    new = dict(block, stop_criteria=[dict(STOP_CRITERIA[0], path="x.y", new_field="written only by the repair")],
               max_model_steps=1500)
    assert "up to 1654" in TP.first_reading_problems(new)[0]
    (a / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=1000)) + "\n", encoding="utf-8")
    assert "cannot be computed" in TP.first_reading_problems(block)[0]


def test_a_launch_is_judged_at_its_own_length_and_report_rate(tmp_path):
    # the declared bound is 2,000 model steps; a --max-steps under it ends the run sooner, and the first reading is
    # judged on the run the command starts (the round-two review's R1, from the command launch_refusal judges)
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    why = TP.launch_refusal(_bounded(cmd, 1200), root=root, src=src, registry=reg)
    assert why and "ends 200 model steps after its start" in why and "up to 246" in why
    assert TP.launch_refusal(_bounded(cmd, 1247), root=root, src=src, registry=reg) is None
    # a command without --report-every runs at the trainer's default, whatever the start run's args.json says
    (a / "args.json").write_text(json.dumps({"resume": None, "report_every": 1000}), encoding="utf-8")
    bare = _bounded(cmd, 1247)
    del bare[bare.index("--report-every"):bare.index("--report-every") + 2]
    assert TP.launch_refusal(bare, root=root, src=src, registry=reg) is None
    assert "up to 1146" in TP.first_reading_problems(dict(TP._load(TP.record_path(fp, root))["repair_run"],
                                                          max_model_steps=600))[0]


def test_the_trainers_default_report_rate_is_the_one_the_preflight_assumes():
    import ast
    tree = ast.parse((TP.SRC / TP.TRAINER_FILE).read_text(encoding="utf-8"))
    defaults = [kw.value.value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and any(isinstance(x, ast.Constant) and x.value == "--report-every" for x in node.args)
                for kw in node.keywords if kw.arg == "default"]
    assert defaults == [TP.TRAINER_REPORT_EVERY]


def test_a_stop_field_absent_for_an_hour_of_the_start_runs_records_is_refused(tmp_path):
    # the round-two review's R2: the watcher stops a run whose stop field is absent for an hour, though its line holds;
    # the pattern of vestibular.improvement_over_persistence in seg41s_1146_resume_20261001_213502 (absent from its 4th
    # to its 19th record)
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    block = TP._load(TP.record_path(fp, root))["repair_run"]
    present = "00100000000000000001101111111110111111111100"
    records = []
    for i, bit in enumerate(present):
        r = dict(type="mimo", step=900 + 100 * i, elapsed_s=167.0 + 230 * i, causal_pe_per_modality=dict(vision=0.01),
                 cmd_response_cos_held=0.5)
        if bit == "1" or i == len(present) - 1:
            r["vestibular"] = dict(improvement_over_persistence=0.25)
        records.append(r)
    (a / "replay_training_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    entry = dict(name="vestibular_improvement", path="vestibular.improvement_over_persistence", op="<", line=-1e30,
                 from_run_hour=0, mode="stop")
    why = TP.stop_field_problems(dict(block, stop_criteria=[entry]))
    assert why and "reads no number for 3910 s in a row" in why[0]
    assert TP.stop_field_problems(dict(block, stop_criteria=[STOP_CRITERIA[0]])) == []


def test_a_launch_attempt_that_trained_uses_the_repair_run_even_unmarked(tmp_path):
    # the round-two review's R4: the relaunch records its attempt before the trainer starts; once the attempt's folder
    # holds training past the start, the record is used whether or not the script lived to mark it
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    attempt = tmp_path / "experiments" / "run_attempt"
    attempt.mkdir(parents=True)
    TP.note_attempt(fp, str(attempt), "tester", root=root)
    bounded = _bounded(cmd, 3000)
    assert TP.launch_plan(bounded, root=root, src=src)["kind"] == "repair_run"      # nothing trained yet: usable
    (attempt / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=1100)) + "\n",
                                                         encoding="utf-8")
    assert TP.launch_plan(bounded, root=root, src=src)["kind"] == "normal"          # trained: used once
    assert "never normal training" in TP.launch_refusal(bounded, root=root, src=src, registry=reg)
    with pytest.raises(TP.PreflightError, match="launched already"):
        TP.note_attempt(fp, str(attempt), "tester", root=root)
    with pytest.raises(TP.PreflightError, match="needs its own record"):
        TP.set_repair_run(fp, TP._load(TP.record_path(fp, root))["repair_run"], "tester", root=root, registry=reg)
    # and the states after it continue that attempt's folder (the re-check's N3 through repair_launch)
    with pytest.raises(TP.PreflightError, match="is not one"):
        TP.add_state(fp, str(start), "judged again", "tester", root=root)


def test_marking_a_launch_needs_the_repair_runs_existing_folder(tmp_path, monkeypatch, capsys):
    # the round-two review's R5: `launched FP --by NAME` without --run wrote run "", and every later state was refused
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    unmarked = lambda: TP._load(TP.record_path(fp, root))["repair_run"]["launched_at"] is None     # noqa: E731
    for run in ("", str(tmp_path / "not_made")):
        with pytest.raises(TP.PreflightError, match="an existing folder"):
            TP.mark_launched(fp, run, "tester", root=root)
        assert unmarked()
    # the command line (the operator, Oct 5, 2026: behaviour, not only the message's wording, decides this test):
    # without --run it refuses by its own condition, before it reads or marks any record; with --run the same command
    # marks the record
    mark0, calls = TP.mark_launched, []
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: calls.append(r) or mark0(f, r, by, root=root))
    assert TP.main(["launched", fp, "--by", "tester"]) == 2
    assert calls == [] and unmarked()
    assert "launched needs --run" in capsys.readouterr().out
    repair = tmp_path / "experiments" / "run_repair"
    repair.mkdir(parents=True)
    assert TP.main(["launched", fp, "--run", str(repair), "--by", "tester"]) == 0
    assert calls == [str(repair)] and not unmarked()
    assert TP.descends_from(str(tmp_path / "EXPERIMENTS" / "run_a" / "checkpoints" / "ckpt_step_1000").replace(
        "\\", "/"), str(a))                                                    # letter case and slashes (mutation PH)


def test_after_the_repair_run_a_state_must_continue_the_repaired_model(tmp_path):
    # the re-check of Oct 5, 2026, finding N3: a state added after the launch named the start checkpoint, or a run
    # that never saw the repair run, and allowed normal training from it
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    repair = _run(tmp_path, "run_repair", (3000,), resume=start)
    later = _run(tmp_path, "run_later", (4000,), resume=repair / "checkpoints/ckpt_step_3000")
    other = _run(tmp_path, "run_other", (5000,))
    TP.mark_launched(fp, str(repair), "tester", root=root)
    for checkpoint in (start, other / "checkpoints/ckpt_step_5000"):
        with pytest.raises(TP.PreflightError, match="is not one"):
            TP.add_state(fp, str(checkpoint), "judged again", "tester", root=root)
    for checkpoint in (repair / "checkpoints/ckpt_step_3000", later / "checkpoints/ckpt_step_4000"):
        TP.add_state(fp, str(checkpoint), "judged again", "tester", root=root)
    assert TP.descends_from(str(later / "checkpoints/ckpt_step_4000"), str(repair))
    assert not TP.descends_from(str(start), str(repair))


# THE EXPERIMENT (the operator, Oct 6, 2026): the cap's repair run and its matched cap-off control; a gate a run does not
# target stays FAIL and no longer blocks this experiment's launch only with a complete exception; the learning signal and
# the outcome validity are confirmed with evidence; the arms differ only in the cap's flags
CAP_FLAGS = ["--attention-cap", "--attention-cap-bound"]


def _experiment(fp_r, fp_c, excepted):
    return dict(id="cap_vs_control", authorization="Keep each marked FAIL, with an explicit exception for this experiment",
                authorized_by="the operator", authorized_at="Oct 6, 2026 1:48 AM PDT",
                arms=dict(repair=fp_r, control=fp_c), differs_only_in=list(CAP_FLAGS),
                exceptions={str(n): dict(failure="gate %d fails" % n, why_measurable="the failure is shared by both arms",
                                         evidence=["row_%d" % n]) for n in excepted},
                learning_signal=dict(text="each capped module receives its lessons' gradient", evidence=["row_ls"]),
                outcome_validity=dict(text="each row reads L against C", evidence=["row_ov"]))


def _experiment_setup(tmp_path):
    src, root, reg = _src(tmp_path), tmp_path / "pre", _registry(tmp_path)
    a = _run(tmp_path, "run_a", (1000,))
    start = a / "checkpoints/ckpt_step_1000"
    (a / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=step, elapsed_s=elapsed, causal_pe_per_modality=dict(vision=0.01),
        cmd_response_cos_held=0.5)) + "\n" for step, elapsed in ((900, 167.0), (1000, 440.0))), encoding="utf-8")
    control = _at(_cmd(), start)
    cap = TP.apply_sets(control, [("--attention-cap", "on"), ("--attention-cap-bound", "3")])
    fp_r = Path(TP.new_record(cap, "the cap arm", "tester", root=root, src=src)).stem
    fp_c = Path(TP.new_record(control, "the matched control", "tester", root=root, src=src)).stem
    common = dict(start_checkpoint=str(start), rollback_checkpoint=str(start), stop_lines="S1 S2",
                  stop_lines_row="stop_row", stop_criteria=STOP_CRITERIA, max_model_steps=2000,
                  affected_components=["TRANSFORMER"], no_auto_continue="the supervisor stays paused",
                  reachability="the first check reads the stop lines about 330 model steps in")
    TP.set_repair_run(fp_r, dict(common, role="repair", purpose="the cap's repair of gates 4, 5 and 7",
                                 target_gates=[4, 5, 7], repairs=["ATTENTION_CAP"],
                                 remeasure={"4": "row_4b", "5": "row_5b", "7": "row_7b"},
                                 experiment=_experiment(fp_r, fp_c, [1])), "tester", root=root, registry=reg)
    TP.set_repair_run(fp_c, dict(common, role="control", purpose="the matched control of experiment cap_vs_control",
                                 target_gates=[], repairs=[], experiment=_experiment(fp_r, fp_c, [1, 4, 5, 7])),
                      "tester", root=root, registry=reg)
    for fp, failing in ((fp_r, [1]), (fp_c, [1, 4, 5, 7])):
        for n in (2, 3, 6):
            TP.set_gate(fp, n, "PASS", "measured", ["row_%d" % n], "tester", root=root)
        for n in failing:
            TP.set_gate(fp, n, "FAIL", "fails on 481,800", ["row_%d" % n], "tester", root=root)
    for n in (4, 5, 7):
        TP.set_gate(fp_r, n, TP.REPAIR_TARGET, "fails today", ["row_%d" % n], "tester", root=root)
    return src, root, reg, start, cap, control, fp_r, fp_c


def _edit_block(root, fp, **changes):
    rec = TP._load(TP.record_path(fp, root))
    rec["repair_run"].update(changes)
    TP._save(TP.record_path(fp, root), rec)


def test_each_arm_launches_with_its_failing_gates_excepted_and_only_then(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    assert TP.launch_plan(_bounded(cap, 3000), root=root, src=src)["kind"] == "repair_run"
    assert TP.launch_plan(_bounded(control, 3000), root=root, src=src)["kind"] == "repair_run"
    assert TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg) is None
    assert TP.launch_refusal(_bounded(control, 3000), root=root, src=src, registry=reg) is None
    # an exception removed: the gate's FAIL blocks again
    rec = TP._load(TP.record_path(fp_r, root))
    del rec["repair_run"]["experiment"]["exceptions"]["1"]
    TP._save(TP.record_path(fp_r, root), rec)
    why = TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)
    assert why and "gate 1 Information flow: FAIL" in why
    # an exception does not stand in for a gate never judged (OPEN)
    rec = TP._load(TP.record_path(fp_c, root))
    rec["gates"] = [dict(g, status="OPEN") if g["n"] == 4 else g for g in rec["gates"]]
    TP._save(TP.record_path(fp_c, root), rec)
    why = TP.launch_refusal(_bounded(control, 3000), root=root, src=src, registry=reg)
    assert why and "gate 4 Local stability: OPEN" in why


def test_an_experiment_without_its_learning_signal_or_outcome_validity_is_blocked(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    block = TP._load(TP.record_path(fp_r, root))["repair_run"]
    for key in ("learning_signal", "outcome_validity"):
        exp = dict(block["experiment"], **{key: dict(text="", evidence=[])})
        with pytest.raises(TP.PreflightError, match=key.replace("_", " ")):
            TP.set_repair_run(fp_r, dict(block, experiment=exp), "tester", root=root, registry=reg)
    _edit_block(root, fp_r, experiment=dict(block["experiment"], learning_signal={}))       # written by hand
    why = TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)
    assert why and "learning signal needs its text and evidence" in why


def test_the_arms_must_match_but_for_the_caps_flags(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    _edit_block(root, fp_c, max_model_steps=1500)
    assert "the arms differ in max model steps" in TP.launch_refusal(_bounded(cap, 3000), root=root, src=src,
                                                                     registry=reg)
    _edit_block(root, fp_c, max_model_steps=2000, stop_criteria=[dict(STOP_CRITERIA[0], line=0.05)])
    assert "the arms differ in stop criteria" in TP.launch_refusal(_bounded(cap, 3000), root=root, src=src,
                                                                   registry=reg)
    _edit_block(root, fp_c, stop_criteria=STOP_CRITERIA)
    rec = TP._load(TP.record_path(fp_c, root))
    rec["flags"] = dict(rec["flags"], **{"--planner-search": "trust_v1"})                      # a second difference
    TP._save(TP.record_path(fp_c, root), rec)
    why = TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)
    assert why and "flags differ beyond" in why and "--planner-search" in why
    TP.record_path(fp_c, root).unlink()
    assert "has no preflight record" in TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)


def test_a_control_run_targets_nothing_and_exists_only_within_an_experiment(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    block = TP._load(TP.record_path(fp_c, root))["repair_run"]
    with pytest.raises(TP.PreflightError, match="exists only within an experiment"):
        TP.set_repair_run(fp_c, {k: v for k, v in block.items() if k != "experiment"}, "tester", root=root,
                          registry=reg)
    with pytest.raises(TP.PreflightError, match="targets no gate"):
        TP.set_repair_run(fp_c, dict(block, target_gates=[4], remeasure={"4": "r"}), "tester", root=root, registry=reg)
    rblock = TP._load(TP.record_path(fp_r, root))["repair_run"]
    with pytest.raises(TP.PreflightError, match="exception is for a gate the run does not target"):
        TP.set_repair_run(fp_r, dict(rblock, experiment=_experiment(fp_r, fp_c, [1, 5])), "tester", root=root,
                          registry=reg)


def test_normal_training_ignores_the_experiments_exceptions(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    run = tmp_path / "experiments" / "run_control"
    run.mkdir(parents=True)
    TP.mark_launched(fp_c, str(run), "tester", root=root)                   # the control arm used once
    why = TP.launch_refusal(_bounded(control, 3000), root=root, src=src, registry=reg)
    assert why and "gate 1 Information flow: FAIL" in why                    # the exception allows no normal training


# THE REPEATED CONTROL (the operator, Oct 6, 2026: "one cap-ON arm and two cap-OFF controls"): two control runs of one
# configuration share its record; each use counts once and the record stays usable until its uses reach repeats
def _repeating_control(tmp_path, repeats=2):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    block = TP._load(TP.record_path(fp_c, root))["repair_run"]
    TP.set_repair_run(fp_c, dict(block, repeats=repeats), "tester", root=root, registry=reg)
    runs = []
    for name in ("c1", "c2", "c3"):
        folder = tmp_path / "experiments" / name
        folder.mkdir(parents=True)
        runs.append(folder)
    return src, root, reg, _bounded(control, 3000), fp_c, runs


def test_a_control_may_repeat_as_declared_and_no_more(tmp_path):
    src, root, reg, bounded, fp_c, runs = _repeating_control(tmp_path)
    assert TP.launch_plan(bounded, root=root, src=src)["kind"] == "repair_run"
    TP.mark_launched(fp_c, str(runs[0]), "tester", root=root)
    assert TP.launch_plan(bounded, root=root, src=src)["kind"] == "repair_run"           # one use of two
    assert TP.launch_refusal(bounded, root=root, src=src, registry=reg) is None
    with pytest.raises(TP.PreflightError, match="marked launched already"):
        TP.mark_launched(fp_c, str(runs[0]), "tester", root=root)
    TP.mark_launched(fp_c, str(runs[1]), "tester", root=root)
    assert TP.launch_plan(bounded, root=root, src=src)["kind"] == "normal"               # both uses spent
    with pytest.raises(TP.PreflightError, match="are marked already"):
        TP.mark_launched(fp_c, str(runs[2]), "tester", root=root)
    with pytest.raises(TP.PreflightError, match="launched already"):
        TP.note_attempt(fp_c, str(runs[2]), "tester", root=root)
    assert "gate 1 Information flow: FAIL" in TP.launch_refusal(bounded, root=root, src=src, registry=reg)


def test_only_a_control_may_repeat(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    block = TP._load(TP.record_path(fp_r, root))["repair_run"]
    with pytest.raises(TP.PreflightError, match="only a control run may repeat"):
        TP.set_repair_run(fp_r, dict(block, repeats=2), "tester", root=root, registry=reg)
    for bad in (0, "two", True):
        with pytest.raises(TP.PreflightError, match="repeats must be a whole number"):
            TP.set_repair_run(fp_r, dict(block, repeats=bad), "tester", root=root, registry=reg)


def test_an_attempt_that_trained_counts_as_one_use_of_a_repeating_control(tmp_path):
    src, root, reg, bounded, fp_c, runs = _repeating_control(tmp_path)
    for i, run in enumerate(runs[:2]):
        TP.note_attempt(fp_c, str(run), "tester", root=root)
        (run / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=1100)) + "\n",
                                                         encoding="utf-8")
        expected = "repair_run" if i == 0 else "normal"
        assert TP.launch_plan(bounded, root=root, src=src)["kind"] == expected
    # marking the first attempt's run later counts the same use, not a third
    TP.mark_launched(fp_c, str(runs[0]), "tester", root=root)
    assert len(TP.repair_uses(TP._load(TP.record_path(fp_c, root))["repair_run"])) == 2


# THE CONTINUATION (the operator's plan of Oct 11, 2026, "Prepare one complete launch decision": "The current repair route
# has already used its allowed run. Prepare a narrowly named continuation route, preserving the other holds; do not
# simply remove that restriction."): one more bounded run of a record whose repair run was used, from that run's end
CONTINUATION_ROWS = ("stop_row", "cont_stop_row", "completion_row")


def _completed_repair_run(tmp_path, root, fp, start, end=3000, name="run_repair"):
    """_repair_setup's repair run launched from `start`, trained to its end and judged again: its folder's args.json
    resumes the start and names the launched bound, its stage_done.json reads ok at that model step, its full stats
    records carry the stop criteria's fields (100 model steps per 273 s, as _repair_setup's start run), and target gates
    4, 5 and 7 are judged again after the launch, FAIL (each for one part)."""
    rep = tmp_path / "experiments" / name
    for n in (2000, end):
        (rep / "checkpoints" / ("ckpt_step_%d" % n)).mkdir(parents=True, exist_ok=True)
    (rep / "args.json").write_text(json.dumps(dict(resume=str(start), max_steps=end, report_every=100)),
                                   encoding="utf-8")
    # the trainer's end, between the launch (100 s before the gates below) and the target gates' judgement again
    (rep / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=end, finished_at=time.time() - 50,
                                                         finished_at_human="the repair run's end")), encoding="utf-8")
    (rep / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=step, elapsed_s=elapsed, causal_pe_per_modality=dict(vision=0.01),
        cmd_response_cos_held=0.5)) + "\n" for step, elapsed in ((end - 100, 167.0), (end, 440.0))), encoding="utf-8")
    TP.note_attempt(fp, str(rep), "tester", root=root)
    TP.mark_launched(fp, str(rep), "tester", root=root)
    rec = TP._load(TP.record_path(fp, root))
    rec["repair_run"]["launched_at_s"] -= 100            # launched 100 s before its target gates are judged again
    TP._save(TP.record_path(fp, root), rec)
    for n in (4, 5, 7):
        TP.set_gate(fp, n, "FAIL", "judged again after the repair run: its finger part PASS, another part FAIL",
                    ["row_%db" % n], "tester", root=root)
    return rep


def _latent_flags(tmp):
    """The continuation's run flags: the latent body's saved state (a file the setup writes) and a new output folder."""
    return {"--latent-body-resume": str(Path(tmp) / "latent" / "latest.pt"),
            "--latent-body-output": str(Path(tmp) / "latent" / "continuation")}


def _latent_files(tmp_path):
    state = Path(_latent_flags(tmp_path)["--latent-body-resume"])
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(b"latent state")


def _continuation_block(start, **changes):
    """A continuation block from `start`; its run flags name the latent files of the test's folder (start's run folder's
    grandparent: <tmp>/experiments/<run>/checkpoints/<checkpoint>)."""
    block = dict(purpose="does further experience improve the repaired model's own hand goals (1,000 model steps)",
                 start_checkpoint=str(start), rollback_checkpoint=str(start), stop_lines="S1 S2",
                 stop_lines_row="cont_stop_row", stop_criteria=STOP_CRITERIA,
                 reachability="the first 5-minute check reads the stop lines about 246 model steps after the start",
                 max_model_steps=1000, no_auto_continue="the supervisor stays paused naming the run",
                 completion_jobs_row="completion_row",
                 authorization="Run the continuation from the repaired end checkpoint as written.",
                 authorized_by="the operator", authorized_at="Oct 11, 2026 9:00 AM PDT",
                 run_flags=_latent_flags(Path(start).parents[3]),
                 exceptions={str(n): dict(failure="gate %d fails for its other part" % n,
                                          why_measurable="the continuation's readings do not depend on that part",
                                          evidence=["row_%db" % n]) for n in (4, 5, 7)})
    block.update(changes)
    return block


def _continuation_setup(tmp_path):
    """_repair_setup's record with the latent body on (--latent-body-hierarchy shadow), its repair run completed in
    run_repair at 3,000, and its continuation written from run_repair's ckpt_step_3000 (at most 1,000 model steps); the
    continuation's command carries the block's run flags."""
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path, latent_body_hierarchy="shadow")
    reg = _registry(tmp_path, rows=CONTINUATION_ROWS)
    rep = _completed_repair_run(tmp_path, root, fp, start)
    end = rep / "checkpoints/ckpt_step_3000"
    _latent_files(tmp_path)
    TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=reg)
    cont = TP.apply_sets(_at(cmd, end), sorted(_latent_flags(tmp_path).items()))
    return src, root, reg, start, rep, end, cont, fp


def _edit_continuation(root, fp, **changes):
    rec = TP._load(TP.record_path(fp, root))
    rec["continuation"].update(changes)
    TP._save(TP.record_path(fp, root), rec)


def _record(root, fp):
    return TP._load(TP.record_path(fp, root))


def test_a_continuation_of_the_completed_repair_run_launches_bounded_with_its_failing_gates_excepted(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    plan = TP.launch_plan(cont, root=root, src=src)
    assert plan["kind"] == "continuation" and plan["block"] == "continuation"
    assert plan["start_step"] == 3000 and plan["max_steps"] == 4000 and plan["stop_criteria"] == STOP_CRITERIA
    assert TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg) is None
    assert TP.launch_refusal(_bounded(cont, 3247), root=root, src=src, registry=reg) is None   # past the first reading
    rec = _record(root, fp)
    # the used repair run's block is kept as written, the gates still read FAIL, and no model state is added
    assert rec["repair_run"]["run"] == str(rep.resolve()) and rec["repair_run"]["launched_at"]
    assert [x["run"] for x in rec["repair_run"]["attempts"]] == [str(rep.resolve())]
    assert [g["status"] for g in rec["gates"]] == ["PASS", "PASS", "PASS", "FAIL", "FAIL", "PASS", "FAIL"]
    assert [s["checkpoint"] for s in rec["states"]] == [str(start)]
    assert rec["continuation"]["role"] == "continuation" and rec["continuation"]["launched_at"] is None
    # the repair run's own start is normal training, refused while gates fail
    why = TP.launch_refusal(_bounded(_at(cont, start), 3000), root=root, src=src, registry=reg)
    assert why and "3 of 7 gates block launch" in why


def test_a_continuation_starts_exactly_at_the_repair_runs_end_checkpoint(tmp_path):
    # the review of Oct 11, 2026, finding F4: continues() accepted a checkpoint of any run resumed from the end
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    other = _run(tmp_path, "run_other", (5000,))
    branch = _run(tmp_path, "run_branch", (3500,), resume=rep / "checkpoints/ckpt_step_2000")
    child = _run(tmp_path, "run_child", (3100,), resume=end)
    (child / "replay_training_stats.jsonl").write_text((rep / "replay_training_stats.jsonl").read_text("utf-8"),
                                                       encoding="utf-8")
    for wrong in (other / "checkpoints/ckpt_step_5000",          # not descending from the repair run
                  start,                                         # the repair run's own start
                  rep / "checkpoints/ckpt_step_2000",            # before the repair run's end
                  branch / "checkpoints/ckpt_step_3500",         # a run resumed from before its end
                  child / "checkpoints/ckpt_step_3100"):         # a run resumed from its end: not the end itself
        with pytest.raises(TP.PreflightError, match="is not the repair run's end checkpoint"):
            TP.set_continuation(fp, _continuation_block(wrong), "tester", root=root, registry=reg)
    # written by hand: refused at launch
    for wrong, steps in ((rep / "checkpoints/ckpt_step_2000", 3000), (other / "checkpoints/ckpt_step_5000", 6000),
                         (child / "checkpoints/ckpt_step_3100", 4100)):
        _edit_continuation(root, fp, start_checkpoint=str(wrong), rollback_checkpoint=str(wrong))
        why = TP.launch_refusal(_bounded(_at(cont, wrong), steps), root=root, src=src, registry=reg)
        assert why and why.startswith("BLOCKED: continuation") and "is not the repair run's end checkpoint" in why
    _edit_continuation(root, fp, start_checkpoint=str(end), rollback_checkpoint=str(end))
    assert TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg) is None


def test_a_later_checkpoint_while_the_continuation_is_unused_is_normal_training(tmp_path):
    # the review's finding F8 (its planted fault P1: launch_refusal matching a later checkpoint by continues() judged it
    # as the continuation, while launch_plan called it normal and the relaunch wrote no stop lines, pause or attempt)
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    child = _run(tmp_path, "run_child", (3100,), resume=end)
    later = _bounded(_at(cont, child / "checkpoints/ckpt_step_3100"), 4000)          # inside the continuation's bound
    assert TP.continuation_available(_record(root, fp)["continuation"])
    assert TP.launch_plan(later, root=root, src=src)["kind"] == "normal"
    why = TP.launch_refusal(later, root=root, src=src, registry=reg)
    assert why and why.startswith("BLOCKED: configuration") and "3 of 7 gates block launch" in why
    # judged as the continuation by hand, the launch is still refused: it does not resume the start
    why = TP._continuation_refusal(later, _record(root, fp), _record(root, fp)["continuation"],
                                   TP.configuration(later, src), root, reg)
    assert why and "not the continuation's start" in why


def test_a_gate_missing_from_the_record_is_never_read_as_pass(tmp_path):
    # the review's finding F8 (its planted fault P2: a gate missing from the record defaulting to PASS went unseen)
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    rec = _record(root, fp)
    rec["gates"] = [g for g in rec["gates"] if g["n"] != 2]
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 2 Gradient flow: OPEN - no result recorded" in why
    rec = _record(root, fp)
    rec["gates"] = [g for g in rec["gates"] if g["n"] != 5]                       # an excepted FAIL gate missing
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 5 Causal controllability: OPEN - no result recorded" in why


def test_a_continuation_of_a_repair_run_that_did_not_complete_is_refused(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    for done, expected in ((dict(ok=True, steps=2500), "did not complete"),       # stopped before its end
                           (dict(ok=False, steps=3000), "did not complete"),      # ended by an exception
                           (None, "did not complete")):                           # no stage_done.json
        if done is None:
            (rep / "stage_done.json").unlink()
        else:
            (rep / "stage_done.json").write_text(json.dumps(done), encoding="utf-8")
        with pytest.raises(TP.PreflightError, match=expected):
            TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=reg)
        why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
        assert why and expected in why
    # completed, but naming no time its trainer ended (the target gates' judgement is read against it: finding F6)
    (rep / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=3000)), encoding="utf-8")
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "the repair run's end time cannot be read" in why
    (rep / "args.json").write_text(json.dumps(dict(resume=str(start))), encoding="utf-8")
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "the repair run's end cannot be read" in why


def test_a_continuation_beyond_its_bound_or_unbounded_is_refused(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    for cmd in (_bounded(cont, 4001), cont, _bounded(cont, 3000), _bounded(cont, 0)):
        why = TP.launch_refusal(cmd, root=root, src=src, registry=reg)
        assert why and "not bounded as declared" in why and "at most at model step 4000" in why
    # a run that ends before the watcher first reads its stop lines (the repair run's check, reused)
    why = TP.launch_refusal(_bounded(cont, 3200), root=root, src=src, registry=reg)
    assert why and "the continuation ends 200 model steps after its start" in why and "up to 246" in why
    with pytest.raises(TP.PreflightError, match="duration is not bounded"):
        TP.set_continuation(fp, _continuation_block(end, max_model_steps=0), "tester", root=root, registry=reg)


def test_a_continuation_without_the_operators_authorization_purpose_or_completion_jobs_is_refused(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    for key in ("authorization", "authorized_by", "authorized_at", "purpose", "completion_jobs_row",
                "no_auto_continue", "rollback_checkpoint", "reachability"):
        with pytest.raises(TP.PreflightError, match="the continuation's %s is missing" % key.replace("_", " ")):
            TP.set_continuation(fp, _continuation_block(end, **{key: " "}), "tester", root=root, registry=reg)
    _edit_continuation(root, fp, authorization="")                                          # written by hand
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "the continuation's authorization is missing" in why
    with pytest.raises(TP.PreflightError, match="who writes it"):
        TP.set_continuation(fp, _continuation_block(end), " ", root=root, registry=reg)


def test_a_failing_gate_without_the_continuations_exception_is_refused(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    block = _record(root, fp)["continuation"]
    _edit_continuation(root, fp, exceptions={k: v for k, v in block["exceptions"].items() if k != "5"})
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 5 Causal controllability: FAIL" in why and "only with its exception" in why
    # an incomplete exception
    partial = dict(block["exceptions"], **{"5": dict(block["exceptions"]["5"], evidence=[])})
    with pytest.raises(TP.PreflightError, match="the exception for gate 5 lacks its evidence"):
        TP.set_continuation(fp, _continuation_block(end, exceptions=partial), "tester", root=root, registry=reg)
    _edit_continuation(root, fp, exceptions=partial)
    assert "gate 5 Causal controllability: FAIL" in TP.launch_refusal(_bounded(cont, 4000), root=root, src=src,
                                                                      registry=reg)
    _edit_continuation(root, fp, exceptions=block["exceptions"])
    assert TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg) is None
    # an exception does not stand in for a gate never judged (OPEN) or one still FAIL - REPAIR TARGET
    TP.set_gate(fp, 5, TP.REPAIR_TARGET, "fails today", ["row_5"], "tester", root=root)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 5 Causal controllability reads FAIL - REPAIR TARGET" in why
    rec = _record(root, fp)
    rec["gates"] = [dict(g, status="OPEN") if g["n"] == 5 else g for g in rec["gates"]]
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 5 Causal controllability: OPEN" in why
    # a target gate of the repair run judged only before that run launched
    TP.set_gate(fp, 5, "FAIL", "judged again", ["row_5b"], "tester", root=root)
    rec = _record(root, fp)
    rec["gates"] = [dict(g, checked_at_s=rec["repair_run"]["launched_at_s"] - 1) if g["n"] == 7 else g
                    for g in rec["gates"]]
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 7 End-to-end causal dry run was a target of the repair run" in why


def test_an_exception_never_turns_a_fail_into_pass_or_na(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    block = _record(root, fp)["continuation"]
    claims = dict(block["exceptions"], **{"5": dict(block["exceptions"]["5"], status="PASS")})
    with pytest.raises(TP.PreflightError, match="reads status PASS: an exception leaves its gate FAIL"):
        TP.set_continuation(fp, _continuation_block(end, exceptions=claims), "tester", root=root, registry=reg)
    _edit_continuation(root, fp, exceptions=claims)                                          # written by hand
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "reads status PASS" in why and "gate 5 Causal controllability: FAIL" in why
    assert _record(root, fp)["gates"][4]["status"] == "FAIL"                       # the gate itself is unchanged
    # an exception naming a gate that reads PASS or N/A is refused: an exception is for a FAIL gate
    _edit_continuation(root, fp, exceptions=dict(block["exceptions"], **{"1": block["exceptions"]["4"]}))
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 1 Information flow reads PASS and the continuation names an exception for it" in why
    _edit_continuation(root, fp, exceptions=block["exceptions"])
    TP.set_gate(fp, 5, "N/A", "said not to apply", ["row_5c"], "tester", root=root)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 5 Causal controllability reads N/A and the continuation names an exception" in why


def test_a_continuation_without_its_registry_rows_or_judgeable_stop_lines_is_refused(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    for rows, missing in ((("stop_row", "completion_row"), "the stop lines' row cont_stop_row is not in the test "
                                                           "registry"),
                          (("stop_row", "cont_stop_row"), "the completion jobs' row completion_row is not in the test "
                                                          "registry")):
        other = _registry(tmp_path, rows=rows)
        why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=other)
        assert why and missing in why
        with pytest.raises(TP.PreflightError, match="is not in the test registry"):
            TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=other)
    cancelled = tmp_path / "cancelled.json"
    cancelled.write_text(json.dumps(dict(rows=[dict(id="cont_stop_row", status="written"),
                                               dict(id="completion_row", status="cancelled")])), encoding="utf-8")
    assert "the completion jobs' row completion_row reads status cancelled" in TP.launch_refusal(
        _bounded(cont, 4000), root=root, src=src, registry=cancelled)
    reg = _registry(tmp_path, rows=CONTINUATION_ROWS)
    for bad, expected in (([dict(STOP_CRITERIA[0], path="causal_pe_per_modality.visoin")], "reads no number"),
                          ([dict(STOP_CRITERIA[0], mode="alert")], "no stop criterion has mode stop"),
                          ([], "stop criteria is missing")):
        with pytest.raises(TP.PreflightError, match=expected):
            TP.set_continuation(fp, _continuation_block(end, stop_criteria=bad), "tester", root=root, registry=reg)
        _edit_continuation(root, fp, stop_criteria=bad)
        why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
        assert why and expected in why


def test_a_continuation_is_used_once(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    repair_before = _record(root, fp)["repair_run"]
    run = _run(tmp_path, "run_cont", (4000,), resume=end)
    TP.note_attempt(fp, str(run), "tester", root=root, block="continuation")
    assert TP.launch_plan(_bounded(cont, 4000), root=root, src=src)["kind"] == "continuation"    # nothing trained yet
    (run / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=3100)) + "\n", encoding="utf-8")
    # the attempt trained: used, though not marked; the same start is now normal training, refused while gates fail
    assert TP.launch_plan(_bounded(cont, 4000), root=root, src=src)["kind"] == "normal"
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "3 of 7 gates block launch" in why and "gate 5 Causal controllability: FAIL" in why
    with pytest.raises(TP.PreflightError, match="this continuation was launched already"):
        TP.note_attempt(fp, str(run), "tester", root=root, block="continuation")
    with pytest.raises(TP.PreflightError, match="a continuation is used once"):
        TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=reg)
    TP.mark_launched(fp, str(run), "tester", root=root, block="continuation")
    with pytest.raises(TP.PreflightError, match="this continuation was launched already"):
        TP.mark_launched(fp, str(run), "tester", root=root, block="continuation")
    rec = _record(root, fp)
    assert rec["continuation"]["run"] == str(run.resolve()) and rec["repair_run"] == repair_before
    with pytest.raises(TP.PreflightError, match="needs its own record"):                # the repair route: unchanged
        TP.set_repair_run(fp, rec["repair_run"], "tester", root=root, registry=reg)
    with pytest.raises(TP.PreflightError, match="the block must be one of"):
        TP.mark_launched(fp, str(run), "tester", root=root, block="sequel")
    # THE EXTENSION (branch feat/preflight-continuation-extension-2026-10-12): "extension" names a block of its own file
    # now; with no extension written it is still refused
    with pytest.raises(TP.PreflightError, match="no extension of configuration"):
        TP.mark_launched(fp, str(run), "tester", root=root, block="extension")


def test_normal_training_after_the_continuation_is_judged_exactly_as_without_it(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    run = _run(tmp_path, "run_cont", (4000,), resume=end)
    TP.mark_launched(fp, str(run), "tester", root=root, block="continuation")
    later = run / "checkpoints/ckpt_step_4000"

    def both(checkpoint, steps):
        """The refusal of a launch from `checkpoint`, with the record as it is and with its continuation removed."""
        cmd = _bounded(_at(cont, checkpoint), steps)
        with_it = TP.launch_refusal(cmd, root=root, src=src, registry=reg)
        rec = _record(root, fp)
        kept = rec.pop("continuation")
        TP._save(TP.record_path(fp, root), rec)
        without = TP.launch_refusal(cmd, root=root, src=src, registry=reg)
        rec["continuation"] = kept
        TP._save(TP.record_path(fp, root), rec)
        return with_it, without

    for checkpoint, steps in ((end, 4000), (later, 0), (later, 5000), (start, 3000)):
        with_it, without = both(checkpoint, steps)
        assert with_it and with_it == without, (checkpoint, with_it, without)
        assert "FAIL" in with_it
    # the gates judged PASS on the repaired model: the state rule decides, as before
    for n in (4, 5, 7):
        TP.set_gate(fp, n, "PASS", "measured again", ["row_%dc" % n], "tester", root=root)
    with_it, without = both(later, 0)
    assert with_it == without and "ran its repair run" in with_it
    TP.add_state(fp, str(end), "gates judged on the repaired model", "tester", root=root)
    assert both(later, 0) == (None, None)


def test_a_continuation_needs_a_launched_repair_run_of_role_repair(tmp_path):
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    reg = _registry(tmp_path, rows=CONTINUATION_ROWS)
    rep = _run(tmp_path, "run_repair", (2000, 3000), resume=start)
    end = rep / "checkpoints/ckpt_step_3000"
    with pytest.raises(TP.PreflightError, match="never launched"):
        TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=reg)
    rec = _record(root, fp)
    rec["continuation"] = _continuation_block(end)                                            # written by hand
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(_at(cmd, end), 4000), root=root, src=src, registry=reg)
    assert why and why.startswith("BLOCKED: continuation") and "never launched" in why
    # an attempt that never trained does not launch the repair run either
    TP.note_attempt(fp, str(rep), "tester", root=root)
    assert "never launched" in TP.launch_refusal(_bounded(_at(cmd, end), 4000), root=root, src=src, registry=reg)
    # a record with no repair run, and a control's record
    bare = Path(TP.new_record(_cmd(planner_search="trust_v2"), "no repair run", "tester", root=root, src=src)).stem
    with pytest.raises(TP.PreflightError, match="has no repair run"):
        TP.set_continuation(bare, _continuation_block(end), "tester", root=root, registry=reg)
    assert TP.continuation_block_problems(_continuation_block(end), dict(repair_run=dict(role="control")), reg)[0] == (
        "a continuation continues a repair run (role repair); this record's run is a control run")
    with pytest.raises(TP.PreflightError, match="no continuation in the record"):
        TP.note_attempt(bare, str(rep), "tester", root=root, block="continuation")


def test_the_command_line_writes_a_continuation_and_marks_its_launch(tmp_path, monkeypatch, capsys):
    src, root, reg, a, start, cmd, fp = _repair_setup(tmp_path)
    reg = _registry(tmp_path, rows=CONTINUATION_ROWS)
    rep = _completed_repair_run(tmp_path, root, fp, start)
    end = rep / "checkpoints/ckpt_step_3000"
    _latent_files(tmp_path)
    set0, mark0 = TP.set_continuation, TP.mark_launched
    monkeypatch.setattr(TP, "set_continuation", lambda f, b, by, *x, **k: set0(f, b, by, root=root, registry=reg))
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: mark0(f, r, by, root=root, **k))
    path = tmp_path / "continuation_block.json"
    path.write_text(json.dumps(_continuation_block(end)), encoding="utf-8-sig")
    assert TP.main(["continuation", fp, "--by", "tester"]) == 2
    assert "continuation needs --block-file" in capsys.readouterr().out
    assert TP.main(["continuation", fp, "--block-file", str(path), "--by", "tester"]) == 0
    assert "continuation written" in capsys.readouterr().out
    repair_before = _record(root, fp)["repair_run"]
    run = tmp_path / "experiments" / "run_cont"
    run.mkdir(parents=True)
    assert TP.main(["launched", fp, "--run", str(run), "--continuation", "--by", "tester"]) == 0
    rec = _record(root, fp)
    assert rec["continuation"]["launched_at"] and rec["continuation"]["run"] == str(run.resolve())
    assert rec["repair_run"] == repair_before
    assert TP.main(["launched", fp, "--run", str(run), "--continuation", "--by", "tester"]) == 2
    assert "this continuation was launched already" in capsys.readouterr().out


# THE REVIEW OF OCT 11, 2026 (D:/UEMA_Research/external/D/UEMA_scratch/remembered_hand_learning_2026-10-11/reviews/
# PREFLIGHT_CONTINUATION_REVIEW.md): findings F1 to F8, each with a test that fails without its repair
def test_f1_a_continuation_declares_its_own_stop_lines_in_rows_not_yet_run(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    # the repair run's own stop lines' row (stop_row, as _repair_setup declares it), even reading written
    with pytest.raises(TP.PreflightError, match="the continuation's stop lines' row stop_row is the repair run's own"):
        TP.set_continuation(fp, _continuation_block(end, stop_lines_row="stop_row"), "tester", root=root, registry=reg)
    _edit_continuation(root, fp, stop_lines_row="stop_row")                                  # written by hand
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "stop_row is the repair run's own" in why
    _edit_continuation(root, fp, stop_lines_row="cont_stop_row")
    # rows already run (done, failed, running) are refused for either; written and queued are accepted
    for status in ("done", "failed", "running"):
        for row in ("cont_stop_row", "completion_row"):
            registry = tmp_path / ("registry_%s_%s.json" % (row, status))
            registry.write_text(json.dumps(dict(rows=[dict(id=r, status=status if r == row else "queued")
                                                      for r in CONTINUATION_ROWS])), encoding="utf-8")
            why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=registry)
            assert why and "%s reads status %s" % (row, status) in why, (row, status, why)
            with pytest.raises(TP.PreflightError, match="reads status %s" % status):
                TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=registry)
    queued = tmp_path / "registry_queued.json"
    queued.write_text(json.dumps(dict(rows=[dict(id=r, status="queued") for r in CONTINUATION_ROWS])),
                      encoding="utf-8")
    assert TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=queued) is None


def test_f2_a_rewrite_keeps_the_recorded_attempts_and_a_used_block_is_never_written_again(tmp_path):
    # the review's probe: an attempt noted (its trainer started, no stats yet), the block written again without it,
    # then the first folder trained, and the continuation stayed available: a second continuation. The re-check of
    # Oct 11, 2026: refusing every rewrite after an attempt left no retry after a trainer that died before training;
    # the attempt is carried into the new block instead
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    # attempts carried in an input block are not taken as the record's own (only the relaunch script records them)
    block = TP.set_continuation(fp, dict(_continuation_block(end), attempts=[dict(run="x")]), "tester", root=root,
                                registry=reg)
    assert "attempts" not in block and "attempts" not in _record(root, fp)["continuation"]
    run = _run(tmp_path, "run_cont", (4000,), resume=end)
    TP.note_attempt(fp, str(run), "tester", root=root, block="continuation")
    TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=reg)   # the attempt trained nothing
    assert [a["run"] for a in _record(root, fp)["continuation"]["attempts"]] == [str(run.resolve())]
    assert TP.continuation_available(_record(root, fp)["continuation"])
    (run / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=3100)) + "\n", encoding="utf-8")
    # the carried attempt trained after the rewrite: it is the one use
    assert not TP.continuation_available(_record(root, fp)["continuation"])
    assert TP.launch_plan(_bounded(cont, 4000), root=root, src=src)["kind"] == "normal"
    assert "3 of 7 gates block launch" in TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    with pytest.raises(TP.PreflightError, match="was used .an attempt trained past its start"):
        TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=reg)
    # a launch mark alone refuses a rewrite too
    src2, root2, reg2, start2, rep2, end2, cont2, fp2 = _continuation_setup(tmp_path / "marked")
    marked = _run(tmp_path / "marked", "run_cont", (4000,), resume=end2)
    TP.mark_launched(fp2, str(marked), "tester", root=root2, block="continuation")
    with pytest.raises(TP.PreflightError, match="was used .launched"):
        TP.set_continuation(fp2, _continuation_block(end2), "tester", root=root2, registry=reg2)


def test_f3_an_exception_excuses_only_the_failure_judged_before_the_block_was_written(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    written = _record(root, fp)["continuation"]["written_at_s"]
    assert TP._number(written)
    _edit_continuation(root, fp, written_at_s=written - 10)          # the block written 10 s before the new judgement
    TP.set_gate(fp, 5, "FAIL", "a NEW failure, judged after the block", ["row_5_new"], "tester", root=root)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 5 Causal controllability was judged again since its exception was written" in why
    TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=reg)      # its exception again
    assert TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg) is None
    rec = _record(root, fp)
    del rec["continuation"]["written_at_s"]                                                   # written by hand
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "cannot be tied to the failure it excuses" in why


def test_f5_the_continuations_run_flags_are_carried_exactly_and_its_latent_output_is_new(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    latent = _latent_flags(tmp_path)
    # the launch without them, or with another value
    bare = _bounded(_at(_cmd(handle_lesson="on", latent_body_hierarchy="shadow"), end), 4000)
    why = TP.launch_refusal(bare, root=root, src=src, registry=reg)
    assert why and "the launch's --latent-body-resume reads absent" in why and "--latent-body-output reads absent" in why
    moved = TP.apply_sets(_bounded(cont, 4000), [("--latent-body-resume", str(tmp_path / "latent" / "fresh.pt"))])
    why = TP.launch_refusal(moved, root=root, src=src, registry=reg)
    assert why and "the launch's --latent-body-resume reads %s" % (tmp_path / "latent" / "fresh.pt") in why
    # the block: required, run flags only (not the chosen ones, not configuration flags), both latent flags named
    for flags, expected in (({}, "the continuation's run flags is missing"),
                            (dict(latent, **{"--out": "x"}), "run flag --out"),
                            (dict(latent, **{"--max-steps": "4000"}), "run flag --max-steps"),
                            (dict(latent, **{"--intention": "on"}), "run flag --intention"),
                            ({"--latent-body-resume": latent["--latent-body-resume"]},
                             "run_flags names --latent-body-output"),
                            (dict(latent, **{"--latent-body-resume": str(tmp_path / "missing.pt")}),
                             "is not a file")):
        with pytest.raises(TP.PreflightError, match=expected):
            TP.set_continuation(fp, _continuation_block(end, run_flags=flags), "tester", root=root, registry=reg)
    # an output folder that exists already: refused when written and at launch
    Path(latent["--latent-body-output"]).mkdir(parents=True)
    with pytest.raises(TP.PreflightError, match="exists already"):
        TP.set_continuation(fp, _continuation_block(end), "tester", root=root, registry=reg)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "the latent body output folder %s exists already" % latent["--latent-body-output"] in why


def test_f6_target_gates_are_judged_again_after_the_repair_run_ended(tmp_path):
    src, root, reg, start, rep, end, cont, fp = _continuation_setup(tmp_path)
    finished = json.loads((rep / "stage_done.json").read_text(encoding="utf-8"))["finished_at"]
    rec = _record(root, fp)
    assert rec["repair_run"]["launched_at_s"] < finished - 1                    # judged after the launch, before the end
    rec["gates"] = [dict(g, checked_at_s=finished - 1) if g["n"] == 7 else g for g in rec["gates"]]
    TP._save(TP.record_path(fp, root), rec)
    why = TP.launch_refusal(_bounded(cont, 4000), root=root, src=src, registry=reg)
    assert why and "gate 7 End-to-end causal dry run was a target of the repair run, which ended" in why


def test_current_reads_the_supervisors_record_and_the_block_key_stays_a_string(tmp_path, monkeypatch):
    """--current reads the supervisor's recorded command (SUPERVISOR_RECORD); CONTINUATION is the continuation block's key.
    Both were once bound to the name CONTINUATION, so --current ended with "'str' object has no attribute 'read_text'"
    (the continuation block review of Oct 11, 2026, note N3)."""
    from types import SimpleNamespace
    record = tmp_path / "training_continuation.json"
    record.write_text(json.dumps(dict(command=["python", "-u", "-m", "mimo_bridge.train_mimo", "--n-envs", "8"])),
                      encoding="utf-8")
    monkeypatch.setattr(TP, "SUPERVISOR_RECORD", record)
    cmd = TP._command(SimpleNamespace(current=True, cmd_file=None, set=[]))
    assert cmd[-2:] == ["--n-envs", "8"]
    assert TP.CONTINUATION == "continuation" and "continuation" in TP.LAUNCH_BLOCKS
