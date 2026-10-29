# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE FRESH INSTANCE (the operator's authorization of Oct 12, 2026, 6:22 AM PDT: "implement and review the narrow
preflight change that admits these two runs while preserving its existing checks and all previous run records").

The world of these tests mirrors the body-1 experiment of Oct 12, 2026 on stand-ins: configuration J (seg41j's) was the
repair arm of an earlier experiment, its repair run and its continuation are used; configuration S (J with one flag
changed, the stand-in for --proprio-token-skip 1,3) is a new record. The new experiment pairs S (role repair, a record of
its own) with a fresh instance of J (role control) from the same start. Nothing here touches a real record: every file
is in pytest's temporary folder.

The three requirements of the coordinator's scope (Oct 12, 2026):
  (1) exactly this experiment's two fresh arm instances are admitted;
  (2) every existing refusal holds for every other case (new_record on an existing record, set_repair_run replacing a
      launched block, arms_problems);
  (3) the earlier record and its launch history stay byte-identical.
"""
import json
import time
from pathlib import Path

import pytest

from mimo_bridge.tools import training_preflight as TP

FLAGS = ("--intention", "--resume", "--out", "--report-every", "--max-steps", "--checkpoint-storage",
         "--proprio-token-skip", "--handle-lesson", "--latent-body-output")
TRAINER = ("import argparse\n\n\ndef parse_args(argv=None):\n    ap = argparse.ArgumentParser()\n"
           + "".join('    ap.add_argument("%s")\n' % f for f in FLAGS) + "    return ap.parse_args(argv)\n")
STOP_CRITERIA = [dict(name="vision_causal_error", path="causal_pe_per_modality.vision", op=">", line=0.04,
                      from_run_hour=0, mode="stop", note="S1"),
                 dict(name="following", path="cmd_response_cos_held", op="<", line=0.3, mode="alert", note="S2")]
NEW = "body1_excluded_vs_pushon"           # the new experiment
OLD = "finger_vs_control"                  # the earlier experiment of configuration J
ROWS = ("old_stop_row", "skip_stop_row", "pushon_stop_row", "done_row")


def _src(tmp_path):
    src = tmp_path / "src"
    for rel, text in (("lem/model/core.py", "x = 1\n"), ("mimo_bridge/train_mimo.py", TRAINER)):
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return src


def _registry(tmp_path):
    p = tmp_path / "registry.json"
    rows = [dict(id=r, status="done" if r == "done_row" else "written") for r in ROWS]
    p.write_text(json.dumps(dict(rows=rows)), encoding="utf-8")
    return p


def _bounded(cmd, end):
    return TP.apply_sets(cmd, [("--max-steps", str(end))])


def _experiment(exp_id, arms, excepted):
    return dict(id=exp_id, authorization="the operator's words", authorized_by="the operator",
                authorized_at="Oct 12, 2026 6:22 AM PDT", arms=arms, differs_only_in=["--proprio-token-skip"],
                exceptions={str(n): dict(failure="gate %d fails at the start" % n,
                                         why_measurable="the failure is common to both arms",
                                         evidence=["row_%d" % n]) for n in excepted},
                learning_signal=dict(text="the encoder receives its lessons' gradient", evidence=["row_ls"]),
                outcome_validity=dict(text="the reading compares encoders on fixed rows", evidence=["row_ov"]))


def _block(start, role, experiment, row, **more):
    b = dict(role=role, purpose="the %s arm" % role, target_gates=[], repairs=[], experiment=experiment,
             affected_components=["ENCODER_PROPRIOCEPTION"], start_checkpoint=str(start),
             rollback_checkpoint=str(start), stop_lines="S1 S2", stop_lines_row=row, stop_criteria=STOP_CRITERIA,
             max_model_steps=2000, no_auto_continue="the supervisor stays paused",
             reachability="the first check reads the stop lines about 330 model steps in")
    b.update(more)
    return b


def _world(tmp_path):
    """J's record with its used repair run (an earlier experiment's repair arm, launched and trained) and a used
    continuation; S's record (one flag changed) with its repair run in the new experiment. Returns a namespace dict."""
    src, root, reg = _src(tmp_path), tmp_path / "pre", _registry(tmp_path)
    a = tmp_path / "experiments" / "run_a"
    start = a / "checkpoints" / "ckpt_step_1000"
    start.mkdir(parents=True)
    (a / "args.json").write_text('{"resume": null}', encoding="utf-8")
    (a / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=step, elapsed_s=elapsed, causal_pe_per_modality=dict(vision=0.01),
        cmd_response_cos_held=0.5)) + "\n" for step, elapsed in ((900, 167.0), (1000, 440.0))), encoding="utf-8")
    base = ["python", "-u", "-m", "mimo_bridge.train_mimo", "--intention", "on", "--resume", str(start), "--out",
            "C:/runs/x", "--report-every", "100", "--proprio-token-skip", "3"]
    skip = TP.apply_sets(base, [("--proprio-token-skip", "1,3")])
    other = TP.apply_sets(base, [("--handle-lesson", "on")])          # the earlier experiment's control
    fp_j = Path(TP.new_record(base, "configuration J", "tester", root=root, src=src)).stem
    fp_o = Path(TP.new_record(other, "the earlier control", "tester", root=root, src=src)).stem
    old_exp = dict(_experiment(OLD, dict(repair=fp_j, control=fp_o), [2, 3]), differs_only_in=["--handle-lesson"])
    TP.set_repair_run(fp_j, _block(start, "repair", old_exp, "old_stop_row", target_gates=[1], repairs=["the old fix"],
                                   remeasure={"1": "row_1b"}), "tester", root=root, registry=reg)
    TP.set_gate(fp_j, 1, TP.REPAIR_TARGET, "fails today", ["row_1"], "tester", root=root)
    for n in (2, 3):
        TP.set_gate(fp_j, n, "FAIL", "fails at the start", ["row_%d" % n], "tester", root=root)
    for n in (4, 5, 6, 7):
        TP.set_gate(fp_j, n, "PASS", "measured", ["row_%d" % n], "tester", root=root)
    # the earlier repair run: noted, trained past its start, marked launched (as the relaunch script wrote seg41j's)
    old_run = tmp_path / "experiments" / "run_j"
    old_run.mkdir(parents=True)
    TP.note_attempt(fp_j, str(old_run), "relaunch", root=root)
    (old_run / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=1100)) + "\n",
                                                         encoding="utf-8")
    TP.mark_launched(fp_j, str(old_run), "relaunch", root=root)
    TP.set_gate(fp_j, 1, "FAIL", "judged again after the repair run: one part FAIL", ["row_1b"], "tester", root=root)
    # a used continuation, kept as the relaunch wrote it (its content is not under test; its bytes are)
    rec = TP._load(TP.record_path(fp_j, root))
    rec["continuation"] = dict(role="continuation", start_checkpoint=str(old_run / "checkpoints/ckpt_step_3000"),
                               launched_at="Oct 11, 2026 3:42 PM PDT", launched_at_s=1.0, run=str(old_run),
                               attempts=[dict(run=str(old_run), at="Oct 11, 2026 3:39 PM PDT", at_s=1.0, by="relaunch")])
    TP._save(TP.record_path(fp_j, root), rec)
    time.sleep(0.01)
    # S: a record of its own, the repair arm of the new experiment
    arms = dict(repair=None, control=fp_j)
    fp_s = TP.configuration(skip, src)["fingerprint"]
    arms["repair"] = fp_s
    TP.new_record(skip, "body 1 excluded", "tester", root=root, src=src)
    TP.set_repair_run(fp_s, _block(start, "repair", _experiment(NEW, arms, [2, 3, 4, 5, 6, 7]), "skip_stop_row",
                                   target_gates=[1], repairs=["body 1 left out of the lesson"],
                                   remeasure={"1": "row_1c"}), "tester", root=root, registry=reg)
    TP.set_gate(fp_s, 1, TP.REPAIR_TARGET, "body 1's part fails", ["row_1"], "tester", root=root)
    for n in (2, 3, 4, 5, 6, 7):
        TP.set_gate(fp_s, n, "FAIL", "fails at the start", ["row_%d" % n], "tester", root=root)
    instance = _block(start, "control", _experiment(NEW, arms, [1, 2, 3]), "pushon_stop_row")
    return dict(src=src, root=root, reg=reg, start=start, base=base, skip=skip, fp_j=fp_j, fp_s=fp_s, fp_o=fp_o,
                instance=instance, old_run=old_run, arms=arms)


def _bytes(w):
    return TP.record_path(w["fp_j"], w["root"]).read_bytes()


def _write_instance(w, **changes):
    block = dict(w["instance"], **changes)
    return TP.set_instance(w["fp_j"], block, "tester", root=w["root"], registry=w["reg"])


def _refusal(w, cmd):
    return TP.launch_refusal(_bounded(cmd, 3000), root=w["root"], src=w["src"], registry=w["reg"])


# ------------------------------------------------------------------ (1) and (3): the two fresh arms are admitted
def test_both_fresh_arms_are_admitted_and_the_earlier_record_stays_byte_identical(tmp_path):
    w = _world(tmp_path)
    before = _bytes(w)
    # without the instance the new experiment's repair arm is refused as before (arms_problems)
    why = _refusal(w, w["skip"])
    assert why and "the control arm's record does not name experiment %s in the role control" % NEW in why
    _write_instance(w)
    assert _bytes(w) == before
    assert _refusal(w, w["skip"]) is None
    assert _refusal(w, w["base"]) is None
    plan_c = TP.launch_plan(_bounded(w["base"], 3000), root=w["root"], src=w["src"])
    plan_s = TP.launch_plan(_bounded(w["skip"], 3000), root=w["root"], src=w["src"])
    assert plan_c["kind"] == "repair_run" and plan_c["block"] == "instance:%s" % NEW and plan_c["max_steps"] == 3000
    assert plan_s["kind"] == "repair_run" and plan_s["block"] == "repair_run"
    assert plan_c["stop_criteria"] == plan_s["stop_criteria"] == STOP_CRITERIA
    held = json.loads(TP.instance_path(w["fp_j"], NEW, w["root"]).read_text(encoding="utf-8"))
    assert held["record_sha256"] == TP._sha(before) and held["block"]["role"] == "control"
    assert _bytes(w) == before


def test_each_instance_is_used_once_its_launch_is_written_in_its_own_file(tmp_path):
    w = _world(tmp_path)
    before = _bytes(w)
    _write_instance(w)
    plan = TP.launch_plan(_bounded(w["base"], 3000), root=w["root"], src=w["src"])
    run = tmp_path / "experiments" / "run_pushon"
    run.mkdir(parents=True)
    TP.note_attempt(w["fp_j"], str(run), "launcher", root=w["root"], block=plan["block"])
    TP.mark_launched(w["fp_j"], str(run), "launcher", root=w["root"], block=plan["block"])
    assert _bytes(w) == before                                     # the earlier record's launch history unchanged
    held = json.loads(TP.instance_path(w["fp_j"], NEW, w["root"]).read_text(encoding="utf-8"))
    assert held["block"]["run"] == str(run.resolve()) and held["block"]["launched_at"]
    assert [x["run"] for x in held["block"]["attempts"]] == [str(run.resolve())]
    with pytest.raises(TP.PreflightError, match="launched already"):
        TP.mark_launched(w["fp_j"], str(run), "launcher", root=w["root"], block=plan["block"])
    with pytest.raises(TP.PreflightError, match="launched already"):
        TP.note_attempt(w["fp_j"], str(run), "launcher", root=w["root"], block=plan["block"])
    # once used, the same command is normal training, which the failing gates refuse
    assert TP.launch_plan(_bounded(w["base"], 3000), root=w["root"], src=w["src"])["kind"] == "normal"
    why = _refusal(w, w["base"])
    assert why and "gate 1 Information flow: FAIL" in why and "of 7 gates block launch" in why
    with pytest.raises(TP.PreflightError, match="was used"):
        _write_instance(w)
    # an attempt that trained past the start counts as the use, whether or not it was marked
    w2 = _world(tmp_path / "second")
    _write_instance(w2)
    run2 = tmp_path / "second" / "experiments" / "run_pushon"
    run2.mkdir(parents=True)
    TP.note_attempt(w2["fp_j"], str(run2), "launcher", root=w2["root"], block="instance:%s" % NEW)
    (run2 / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=1100)) + "\n",
                                                      encoding="utf-8")
    assert TP.launch_plan(_bounded(w2["base"], 3000), root=w2["root"], src=w2["src"])["kind"] == "normal"


def test_an_attempt_that_never_trained_is_carried_when_the_instance_is_written_again(tmp_path):
    w = _world(tmp_path)
    _write_instance(w)
    run = tmp_path / "experiments" / "run_dead"
    run.mkdir(parents=True)
    TP.note_attempt(w["fp_j"], str(run), "launcher", root=w["root"], block="instance:%s" % NEW)
    block = _write_instance(w, purpose="written again after a trainer that died before training")
    assert [x["run"] for x in block["attempts"]] == [str(run.resolve())]


# ------------------------------------------------------------------ (2): every existing refusal holds
def test_the_existing_refusals_hold(tmp_path):
    w = _world(tmp_path)
    with pytest.raises(TP.PreflightError, match="a record for configuration %s exists" % w["fp_j"]):
        TP.new_record(w["base"], "a second record", "tester", root=w["root"], src=w["src"])
    rec = TP._load(TP.record_path(w["fp_j"], w["root"]))
    with pytest.raises(TP.PreflightError, match="was launched .*a new repair run needs its own record"):
        TP.set_repair_run(w["fp_j"], w["instance"], "tester", root=w["root"], registry=w["reg"])
    assert TP._load(TP.record_path(w["fp_j"], w["root"])) == rec
    # without an instance J's command is normal training, refused by its failing gates
    assert TP.launch_plan(_bounded(w["base"], 3000), root=w["root"], src=w["src"])["kind"] == "normal"
    assert "gate 1 Information flow: FAIL" in _refusal(w, w["base"])


@pytest.mark.parametrize("change, expected", [
    (dict(stop_criteria=[dict(STOP_CRITERIA[0], line=0.05)] + STOP_CRITERIA[1:]), "the arms differ in stop criteria"),
    (dict(max_model_steps=1500), "the arms differ in max model steps"),
])
def test_arms_problems_compare_a_fresh_instance_as_a_repair_run_block(tmp_path, change, expected):
    w = _world(tmp_path)
    _write_instance(w, **change)
    assert expected in _refusal(w, w["skip"])
    assert expected in (_refusal(w, w["base"]) if change.get("max_model_steps") is None else
                        TP.launch_refusal(_bounded(w["base"], 2500), root=w["root"], src=w["src"], registry=w["reg"]))


def test_arms_whose_flags_differ_beyond_the_experiment_or_whose_starts_differ_are_refused(tmp_path):
    w = _world(tmp_path)
    _write_instance(w)
    rec = TP._load(TP.record_path(w["fp_s"], w["root"]))
    rec["flags"] = dict(rec["flags"], **{"--handle-lesson": "on"})              # a second difference
    TP._save(TP.record_path(w["fp_s"], w["root"]), rec)
    why = _refusal(w, w["base"])
    assert why and "flags differ beyond" in why and "--handle-lesson" in why
    w2 = _world(tmp_path / "b")
    later = w2["start"].parent / "ckpt_step_1100"
    later.mkdir()
    rec = TP._load(TP.record_path(w2["fp_s"], w2["root"]))
    rec["repair_run"]["start_checkpoint"] = str(later)
    TP._save(TP.record_path(w2["fp_s"], w2["root"]), rec)
    _write_instance(w2)
    why = _refusal(w2, w2["base"])
    assert why and "the arms start from different checkpoints" in why


def test_the_other_arm_must_name_this_experiment_in_the_other_role(tmp_path):
    w = _world(tmp_path)
    _write_instance(w)
    rec = TP._load(TP.record_path(w["fp_s"], w["root"]))
    rec["repair_run"]["experiment"]["id"] = "another_experiment"
    TP._save(TP.record_path(w["fp_s"], w["root"]), rec)
    why = _refusal(w, w["base"])
    assert why and "the repair arm's record does not name experiment %s in the role repair" % NEW in why


@pytest.mark.parametrize("make, expected", [
    (lambda w: _write_instance(w, role="repair", target_gates=[1], repairs=["x"], remeasure={"1": "r"}),
     "a fresh instance is a control run"),
    (lambda w: _write_instance(w, experiment=dict(w["instance"]["experiment"], id=OLD)),
     "is the record's used repair run's"),
    (lambda w: _write_instance(w, stop_lines_row="old_stop_row"), "is the used repair run's"),
    (lambda w: _write_instance(w, stop_lines_row="done_row"), "reads status done"),
    (lambda w: _write_instance(w, repeats=2), "a fresh instance is used once"),
    (lambda w: _write_instance(w, experiment=dict(w["instance"]["experiment"], id="../escape")),
     "names the fresh instance's file"),
    (lambda w: _write_instance(w, start_checkpoint=str(w["start"].parent / "ckpt_step_1100"),
                               rollback_checkpoint=str(w["start"].parent / "ckpt_step_1100")),
     "is not a model state the record names"),
    (lambda w: TP.set_instance(w["fp_s"], dict(w["instance"]), "tester", root=w["root"], registry=w["reg"]),
     "the record's own repair run was not used"),
    (lambda w: TP.set_instance("0123456789abcdef", dict(w["instance"]), "tester", root=w["root"], registry=w["reg"]),
     "no record for configuration"),
])
def test_an_instance_outside_its_conditions_is_refused_when_written(tmp_path, make, expected):
    w = _world(tmp_path)
    before = _bytes(w)
    with pytest.raises(TP.PreflightError, match=expected.replace("(", r"\(").replace(".", r"\.")):
        make(w)
    assert _bytes(w) == before
    assert not (w["root"] / TP.INSTANCES_DIR).exists()            # nothing written


def test_two_unused_instances_from_one_start_are_refused(tmp_path):
    w = _world(tmp_path)
    _write_instance(w)
    second = dict(w["instance"]["experiment"], id=NEW + "_b")
    with pytest.raises(TP.PreflightError, match="starts from the same checkpoint and is unused"):
        _write_instance(w, experiment=second)
    # written by hand beside it, the launch refuses to choose
    held = json.loads(TP.instance_path(w["fp_j"], NEW, w["root"]).read_text(encoding="utf-8"))
    held["experiment_id"] = NEW + "_b"
    held["block"]["experiment"]["id"] = NEW + "_b"
    TP._save(TP.instance_path(w["fp_j"], NEW + "_b", w["root"]), held)
    why = _refusal(w, w["base"])
    assert why and "2 unused fresh instances" in why
    assert TP.launch_plan(_bounded(w["base"], 3000), root=w["root"], src=w["src"])["kind"] == "normal"


def test_an_exception_excuses_only_the_failure_judged_before_the_instance_was_written(tmp_path):
    w = _world(tmp_path)
    _write_instance(w)
    time.sleep(0.01)
    TP.set_gate(w["fp_j"], 2, "FAIL", "judged again", ["row_2c"], "tester", root=w["root"])
    why = _refusal(w, w["base"])
    assert why and "gate 2 Gradient flow was judged again since the fresh instance was written" in why
    _write_instance(w)
    assert _refusal(w, w["base"]) is None


def test_a_failing_gate_without_the_instances_exception_is_refused(tmp_path):
    w = _world(tmp_path)
    exp = dict(w["instance"]["experiment"])
    exp["exceptions"] = {k: v for k, v in exp["exceptions"].items() if k != "3"}
    _write_instance(w, experiment=exp)
    why = _refusal(w, w["base"])
    assert why and "gate 3 Scale/conditioning: FAIL" in why and "FAIL with an exception of its experiment" in why


def test_a_launch_beyond_the_bound_or_from_another_checkpoint_is_not_the_instance(tmp_path):
    w = _world(tmp_path)
    _write_instance(w)
    assert "not bounded as declared" in TP.launch_refusal(_bounded(w["base"], 3001), root=w["root"], src=w["src"],
                                                          registry=w["reg"])
    later = w["start"].parent / "ckpt_step_1100"
    later.mkdir()
    moved = TP.apply_sets(_bounded(w["base"], 3000), [("--resume", str(later))])
    assert TP.launch_plan(moved, root=w["root"], src=w["src"])["kind"] == "normal"
    assert "gate 1 Information flow: FAIL" in TP.launch_refusal(moved, root=w["root"], src=w["src"], registry=w["reg"])


def test_the_command_line_writes_an_instance_and_marks_its_launch(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path)
    before = _bytes(w)
    set0, mark0 = TP.set_instance, TP.mark_launched
    monkeypatch.setattr(TP, "set_instance", lambda f, b, by, *x, **k: set0(f, b, by, root=w["root"], registry=w["reg"]))
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: mark0(f, r, by, root=w["root"], **k))
    path = tmp_path / "instance_block.json"
    path.write_text(json.dumps(w["instance"]), encoding="utf-8-sig")
    assert TP.main(["instance", w["fp_j"], "--by", "tester"]) == 2
    assert "instance needs --block-file" in capsys.readouterr().out
    assert TP.main(["instance", w["fp_j"], "--block-file", str(path), "--by", "tester"]) == 0
    assert "fresh instance written" in capsys.readouterr().out
    run = tmp_path / "experiments" / "run_cli"
    run.mkdir(parents=True)
    assert TP.main(["launched", w["fp_j"], "--run", str(run), "--instance", NEW, "--by", "tester"]) == 0
    assert "fresh instance %s marked launched" % NEW in capsys.readouterr().out
    assert TP.main(["launched", w["fp_j"], "--run", str(run), "--instance", NEW, "--by", "tester"]) == 2
    assert "launched already" in capsys.readouterr().out
    assert TP.main(["launched", w["fp_j"], "--run", str(run), "--instance", NEW, "--continuation", "--by", "t"]) == 2
    assert _bytes(w) == before
