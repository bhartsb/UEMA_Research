# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Review probes for commit 291962552 (the training preflight's EXPERIMENT: a repair run and its matched control).

Written by the scoped review of Oct 6, 2026 (docs/gain_of_function_v4/coordinator/
preflight_experiment_review_2026-10-06.md, branch review/preflight-experiment-2026-10-06). The relaunch script's main() runs both arms in the intended sequence with every
process call replaced (the review's harness); nothing here looks at, ends or starts a real trainer, supervisor or
watcher, and nothing is written outside pytest's temporary folder. A probe asserting the behaviour a finding asks for,
which fails on 291962552, is marked xfail(strict=True) with the finding's id (X1 to X4).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tests.test_relaunch_repair_run_review import SCRIPT
from mimo_bridge.tests.test_training_preflight import (STOP_CRITERIA, _at, _bounded, _cmd, _edit_block, _experiment,
                                                       _experiment_setup, _registry, _run, _src)
from mimo_bridge.tools import training_preflight as TP

CAP_SETS = [("--attention-cap", "on"), ("--attention-cap-bound", "3")]


def two_arm_harness(tmp_path, monkeypatch):
    """The relaunch script's main() wired to the experiment of tests/test_training_preflight.py _experiment_setup (the
    cap arm fp_r and the control arm fp_c, both from run_a's ckpt_step_1000, 2,000 model steps) and temporary Saved/
    folders. The supervisor stand-in records what the real one records: its pid, the run, and the trainer's own
    command with the supervisor's interpreter first (training_continuation.supervise)."""
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    a = start.parents[1]
    spec = importlib.util.spec_from_file_location("relaunch_experiment_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    L = module.L
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "training_continuation.json").write_text(json.dumps(dict(active_run=str(a), command=control)),
                                                      encoding="utf-8")
    (saved / "supervisor_control.json").write_text(json.dumps({"cmd": "pause", "reason": "held", "archiver": "off"}),
                                                   encoding="utf-8")
    monkeypatch.setattr(L, "SAVED", saved)
    monkeypatch.setattr(L, "EXP", tmp_path / "experiments")
    monkeypatch.setattr(L, "RECEIPT", saved / "seg41_launch_receipt.json")
    monkeypatch.setattr(L, "stamp", lambda: "Oct 6, 2026, 2:00:00 AM PDT")
    monkeypatch.setattr(L, "trainers", lambda: [])
    monkeypatch.setattr(L, "guard", lambda: (True, {"committed_gb": 1.0}))

    def newest(parent):                 # the run's newest checkpoint folder (completeness is not under test here)
        cks = sorted((Path(parent) / "checkpoints").glob("ckpt_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
        if not cks:
            raise SystemExit("refused: %s has no complete eight-world checkpoint" % parent)
        return cks[-1]
    monkeypatch.setattr(module, "newest_complete_checkpoint", newest)
    monkeypatch.setattr(module, "manifest_refusal", lambda root=None: None)
    import psutil

    class NoProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, *x, **k):
            return []

        def terminate(self):
            raise AssertionError("the probe tried to end process %s" % self.pid)

    monkeypatch.setattr(psutil, "process_iter", lambda *x, **k: [])
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: ([], []))
    monkeypatch.setattr(psutil, "Process", NoProcess)
    plan0, refusal0, mark0, note0 = TP.launch_plan, TP.launch_refusal, TP.mark_launched, TP.note_attempt
    asked = []
    monkeypatch.setattr(TP, "launch_plan", lambda cmd, *x, **k: plan0(cmd, root=root, src=src))
    monkeypatch.setattr(TP, "launch_refusal", lambda cmd, *x, **k: asked.append(
        (refusal0(cmd, root=root, src=src, registry=reg), [str(c) for c in cmd])) or asked[-1][0])
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: mark0(f, r, by, root=root))
    monkeypatch.setattr(TP, "note_attempt", lambda f, r, by, *x, **k: note0(f, r, by, root=root))
    clock = {"now": 2_000_000.0}

    class Clock:
        @staticmethod
        def time():
            return clock["now"]

        @staticmethod
        def sleep(s):
            clock["now"] += s

        @staticmethod
        def strftime(fmt, *x):
            return time.strftime(fmt, *x)
    monkeypatch.setattr(module, "time", Clock())
    launched = []

    class Trainer:
        pid, returncode = 4242, None

        def poll(self):
            return None

    def launch_trainer(cmd, run):
        launched.append(dict(cmd=[str(c) for c in cmd], run=run))
        (run / "load_report.json").write_text(json.dumps({"reinitialized_components": []}), encoding="utf-8")
        (run / "replay_training_stats.jsonl").write_text(json.dumps({"type": "mimo", "step": 1100}) + "\n",
                                                         encoding="utf-8")
        return Trainer()
    monkeypatch.setattr(L, "launch_trainer", launch_trainer)

    def popen(cmd, **kw):
        cmd = [str(c) for c in cmd]
        assert "mimo_bridge.tools.training_continuation" in cmd, cmd
        if kw.get("stdout") is not None:
            kw["stdout"].close()
        (saved / "training_continuation.json").write_text(json.dumps(dict(
            supervisor_pid=777, active_run=cmd[cmd.index("--run") + 1],
            command=[sys.executable] + launched[-1]["cmd"][1:])), encoding="utf-8")
        return SimpleNamespace(pid=777)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    return module, SimpleNamespace(src=src, root=root, reg=reg, a=a, start=start, cap=cap, control=control, fp_r=fp_r,
                                   fp_c=fp_c, saved=saved, launched=launched, asked=asked)


def run_main(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["relaunch.py", *argv])
    module.main()


def kept(cmd):
    return {k: v for k, v in TP.flags_of(cmd).items() if k not in TP.RUN_FLAGS}


def rr_of(p, fp):
    return TP._load(TP.record_path(fp, p.root))["repair_run"]


# ------------------------------------------------------------------ the relaunch carries both arms in the sequence
def test_the_control_arm_then_the_cap_arm_with_parent_and_sets_launch_as_the_experiment_declares(tmp_path, monkeypatch):
    module, p = two_arm_harness(tmp_path, monkeypatch)
    run_main(module, monkeypatch, "--tag", "c", "--reason", "the matched control")
    control = p.launched[0]
    assert TP.flags_of(control["cmd"])["--resume"] == str(p.start) and TP.flags_of(control["cmd"])["--max-steps"] == "3000"
    assert "--attention-cap" not in control["cmd"] and rr_of(p, p.fp_c)["launched_at"]
    (control["run"] / "checkpoints" / "ckpt_step_3000").mkdir(parents=True)      # the control's end save
    state = json.loads((p.saved / "training_continuation.json").read_text(encoding="utf-8"))
    assert state["active_run"] == str(control["run"])           # the supervisor's record names the control run now
    run_main(module, monkeypatch, "--tag", "r", "--reason", "the cap arm", "--parent", str(p.a),
             "--set", "attention-cap", "on", "--set", "attention-cap-bound", "3")
    capr = p.launched[1]
    assert TP.flags_of(capr["cmd"])["--resume"] == str(p.start) and TP.flags_of(capr["cmd"])["--max-steps"] == "3000"
    mine, theirs = kept(capr["cmd"]), kept(control["cmd"])
    assert {k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)} == {"--attention-cap",
                                                                                     "--attention-cap-bound"}
    assert TP.configuration(capr["cmd"], p.src)["fingerprint"] == p.fp_r and rr_of(p, p.fp_r)["launched_at"]
    assert [why for why, _cmd in p.asked] == [None, None]
    ctl = json.loads((p.saved / "supervisor_control.json").read_text(encoding="utf-8"))
    assert ctl["cmd"] == "pause" and ctl["repair_run"] == str(capr["run"]) and ctl["archiver"] == "off"


def test_the_cap_arm_without_parent_resumes_the_controls_end_and_is_refused(tmp_path, monkeypatch):
    module, p = two_arm_harness(tmp_path, monkeypatch)
    run_main(module, monkeypatch, "--tag", "c", "--reason", "the matched control")
    (p.launched[0]["run"] / "checkpoints" / "ckpt_step_3000").mkdir(parents=True)
    with pytest.raises(SystemExit, match="refused by the training preflight"):
        run_main(module, monkeypatch, "--tag", "r", "--reason", "the cap arm",
                 "--set", "attention-cap", "on", "--set", "attention-cap-bound", "3")
    assert len(p.launched) == 1 and not rr_of(p, p.fp_r)["launched_at"]
    assert "gate 1 Information flow: FAIL" in p.asked[-1][0]          # judged as normal training, not as the arm


def test_either_order_and_neither_arm_reaches_normal_training_through_its_exceptions(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    for fp, cmd in ((fp_r, cap), (fp_c, control)):
        assert TP.launch_refusal(_bounded(cmd, 3000), root=root, src=src, registry=reg) is None
        run = tmp_path / "experiments" / ("used_" + fp)
        run.mkdir(parents=True)
        TP.mark_launched(fp, str(run), "tester", root=root)
        why = TP.launch_refusal(_bounded(cmd, 3000), root=root, src=src, registry=reg)
        assert why and "gate 1 Information flow: FAIL" in why and TP.launch_plan(_bounded(cmd, 3000), root=root,
                                                                                  src=src)["kind"] == "normal"


# ------------------------------------------------------------------ X1: an arm's run length
@pytest.mark.xfail(strict=True, reason="X1: an arm's --max-steps may be under its declared bound, so the arms can run "
                                       "different numbers of updates")
def test_an_arm_launched_shorter_than_the_declared_length_is_refused(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    assert TP.launch_refusal(_bounded(cap, 2500), root=root, src=src, registry=reg)        # 1,500 of the 2,000


def test_an_arm_launched_shorter_than_the_declared_length_is_accepted(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    assert TP.launch_refusal(_bounded(control, 3000), root=root, src=src, registry=reg) is None
    assert TP.launch_refusal(_bounded(cap, 2500), root=root, src=src, registry=reg) is None


# ------------------------------------------------------------------ X2: the contents of the files flags name
def _experiment_with_a_named_file(tmp_path, change_between):
    """_experiment_setup's experiment with --vision-input-calibration naming a file; `change_between` rewrites the
    file after the control's record is written and its launch judged, before the cap's record is written."""
    src, root, reg = _src(tmp_path), tmp_path / "pre", _registry(tmp_path)
    a = _run(tmp_path, "run_a", (1000,))
    start = a / "checkpoints/ckpt_step_1000"
    (a / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=step, elapsed_s=elapsed, causal_pe_per_modality=dict(vision=0.01),
        cmd_response_cos_held=0.5)) + "\n" for step, elapsed in ((900, 167.0), (1000, 440.0))), encoding="utf-8")
    calib = tmp_path / "vision_calibration.json"
    calib.write_text('{"gain": 1.0}', encoding="utf-8")
    control = _at(_cmd(calib), start)
    cap = TP.apply_sets(control, CAP_SETS)
    fp_c = Path(TP.new_record(control, "the matched control", "tester", root=root, src=src)).stem
    common = dict(start_checkpoint=str(start), rollback_checkpoint=str(start), stop_lines="S1 S2",
                  stop_lines_row="stop_row", stop_criteria=STOP_CRITERIA, max_model_steps=2000,
                  affected_components=["TRANSFORMER"], no_auto_continue="the supervisor stays paused",
                  reachability="the first check reads the stop lines about 330 model steps in")
    fp_r_expected = TP.configuration(cap, src)["fingerprint"]
    TP.set_repair_run(fp_c, dict(common, role="control", purpose="the matched control", target_gates=[], repairs=[],
                                 experiment=_experiment(fp_r_expected, fp_c, [1, 4, 5, 7])), "tester", root=root,
                      registry=reg)
    for n in (2, 3, 6):
        TP.set_gate(fp_c, n, "PASS", "measured", ["row_%d" % n], "tester", root=root)
    for n in (1, 4, 5, 7):
        TP.set_gate(fp_c, n, "FAIL", "fails on 481,800", ["row_%d" % n], "tester", root=root)
    if change_between:
        calib.write_text('{"gain": 2.0}', encoding="utf-8")
    fp_r = Path(TP.new_record(cap, "the cap arm", "tester", root=root, src=src)).stem
    return src, root, reg, start, cap, control, fp_r, fp_c, common


def _cap_arm_after_the_file_changed(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c, common = _experiment_with_a_named_file(tmp_path, True)
    _edit_block(root, fp_c, experiment=_experiment(fp_r, fp_c, [1, 4, 5, 7]))
    TP.set_repair_run(fp_r, dict(common, role="repair", purpose="the cap", target_gates=[4, 5, 7],
                                 repairs=["ATTENTION_CAP"], remeasure={"4": "r4", "5": "r5", "7": "r7"},
                                 experiment=_experiment(fp_r, fp_c, [1])), "tester", root=root, registry=reg)
    for n in (2, 3, 6):
        TP.set_gate(fp_r, n, "PASS", "measured", ["row_%d" % n], "tester", root=root)
    TP.set_gate(fp_r, 1, "FAIL", "fails on 481,800", ["row_1"], "tester", root=root)
    for n in (4, 5, 7):
        TP.set_gate(fp_r, n, TP.REPAIR_TARGET, "fails today", ["row_%d" % n], "tester", root=root)
    return src, root, reg, cap, fp_r, fp_c


def test_arms_whose_named_files_differ_are_accepted(tmp_path):
    # the control's record was written (and its launch judged) with the calibration file reading gain 1.0; the file
    # then read 2.0 when the cap's record was written: the arms' configurations differ in that file's content
    src, root, reg, cap, fp_r, fp_c = _cap_arm_after_the_file_changed(tmp_path)
    files_c = TP._load(TP.record_path(fp_c, root))["files"]["--vision-input-calibration"]
    files_r = TP._load(TP.record_path(fp_r, root))["files"]["--vision-input-calibration"]
    assert files_c != files_r
    assert TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg) is None


@pytest.mark.xfail(strict=True, reason="X2: arms_problems compares flags and code but not the contents of the files the "
                                       "flags name")
def test_arms_whose_named_files_differ_are_refused(tmp_path):
    src, root, reg, cap, fp_r, fp_c = _cap_arm_after_the_file_changed(tmp_path)
    assert TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)


CAP_SETS_FLAGS = [f for f, _v in CAP_SETS]


# ------------------------------------------------------------------ X3: an excepted gate marked other than FAIL
@pytest.mark.xfail(strict=True, reason="X3: a gate the experiment names an exception for may read N/A or PASS, and then "
                                       "launches as if never failing")
def test_a_gate_with_an_exception_must_read_fail(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    TP.set_gate(fp_r, 1, "N/A", "the cap does not reach the vestibular token", ["row_1"], "tester", root=root)
    assert TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)


def test_a_failing_gate_entered_as_pass_needs_no_exception(tmp_path):
    # the arms' records are new (the cap's merge changed the code, so the held record c007d2bb5af08224 matches neither);
    # nothing ties their gates to that record's seven FAILs
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    rec = TP._load(TP.record_path(fp_r, root))
    rec["repair_run"]["experiment"]["exceptions"] = {}
    TP._save(TP.record_path(fp_r, root), rec)
    TP.set_gate(fp_r, 1, "PASS", "judged on the new configuration", ["row_1"], "tester", root=root)
    assert TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg) is None


# ------------------------------------------------------------------ X4: what the arms' records do not compare
def test_each_arm_checks_only_its_own_copy_of_the_experiment(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    block = rr_of(SimpleNamespace(root=root), fp_c)
    _edit_block(root, fp_c, experiment=dict(block["experiment"], learning_signal={}, outcome_validity={}))
    assert TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg) is None     # the cap arm launches
    assert "learning signal needs its text and evidence" in TP.launch_refusal(_bounded(control, 3000), root=root,
                                                                             src=src, registry=reg)


@pytest.mark.xfail(strict=True, reason="X4: differs_only_in may name any flag; the cap setting is then not the only "
                                       "difference")
def test_the_arms_may_differ_only_in_the_repairs_own_flags(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    for fp in (fp_r, fp_c):
        block = rr_of(SimpleNamespace(root=root), fp)
        _edit_block(root, fp, experiment=dict(block["experiment"], differs_only_in=CAP_SETS_FLAGS + ["--planner-search"]))
    rec = TP._load(TP.record_path(fp_c, root))
    rec["flags"] = dict(rec["flags"], **{"--planner-search": "trust_v1"})
    TP._save(TP.record_path(fp_c, root), rec)
    assert TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)


# ------------------------------------------------------------------ test gaps: checks no commit test exercises
@pytest.mark.parametrize("change, expected", [
    (lambda rec: rec.update(code_hash="0" * 64), "the arms run different code"),
    (lambda rec: rec["repair_run"].update(start_checkpoint="C:/elsewhere/checkpoints/ckpt_step_1000"),
     "the arms start from different checkpoints"),
    (lambda rec: rec["repair_run"]["experiment"].update(id="another_experiment"), "does not name experiment"),
    (lambda rec: rec["repair_run"].update(role="repair"), "does not name experiment"),
])
def test_the_other_arms_record_must_be_this_experiments_matched_arm(tmp_path, change, expected):
    # mutations N06 (code), N07 (start) and N08 (the other record's experiment and role) of the scoped review
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    rec = TP._load(TP.record_path(fp_c, root))
    change(rec)
    TP._save(TP.record_path(fp_c, root), rec)
    why = TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)
    assert why and expected in why


def test_a_record_must_be_the_arm_its_experiment_names(tmp_path):
    # mutation N09: the cap's record naming another configuration as the experiment's repair arm
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    block = rr_of(SimpleNamespace(root=root), fp_r)
    _edit_block(root, fp_r, experiment=dict(block["experiment"], arms=dict(repair="0123456789abcdef", control=fp_c)))
    why = TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg)
    assert why and "is not the repair arm the experiment cap_vs_control names" in why
