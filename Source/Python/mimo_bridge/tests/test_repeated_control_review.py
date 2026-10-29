# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Review probes for commit 71f5dac23 (the repeated control: one cap-ON arm and two cap-OFF controls).

Written by the scoped review of Oct 6, 2026 (docs/gain_of_function_v4/coordinator/repeated_control_review_2026-10-06.md,
branch review/repeated-control-2026-10-06). The relaunch script's main() runs the intended sequence (control 1, control 2,
the cap arm) with every process call replaced, through the experiment review's harness; nothing here looks at, ends or
starts a real trainer, supervisor or watcher, and nothing is written outside pytest's temporary folder.
"""
import json

import pytest

from mimo_bridge.tests.test_preflight_experiment_review import kept, rr_of, run_main, two_arm_harness
from mimo_bridge.tests.test_training_preflight import _bounded, _experiment_setup
from mimo_bridge.tools import training_preflight as TP

CAP = ("--set", "attention-cap", "on", "--set", "attention-cap-bound", "3")


def with_two_controls(tmp_path, monkeypatch):
    module, p = two_arm_harness(tmp_path, monkeypatch)
    block = rr_of(p, p.fp_c)
    TP.set_repair_run(p.fp_c, dict(block, repeats=2), "tester", root=p.root, registry=p.reg)
    return module, p


def control(module, monkeypatch, p, tag):
    run_main(module, monkeypatch, "--tag", tag, "--reason", "a cap-off control", "--parent", str(p.a))
    (p.launched[-1]["run"] / "checkpoints" / "ckpt_step_3000").mkdir(parents=True)      # its end save
    return p.launched[-1]


def uses(p, fp):
    return [str(u[2]) for u in TP.repair_uses(rr_of(p, fp))]


def test_two_controls_then_the_cap_arm_and_no_fourth_run(tmp_path, monkeypatch):
    module, p = with_two_controls(tmp_path, monkeypatch)
    c1 = control(module, monkeypatch, p, "a")
    state = json.loads((p.saved / "training_continuation.json").read_text(encoding="utf-8"))
    assert state["active_run"] == str(c1["run"]) and kept(state["command"]) == kept(c1["cmd"])
    c2 = control(module, monkeypatch, p, "b")                   # from control 1's recorded command, --parent the start
    for c in (c1, c2):
        assert TP.flags_of(c["cmd"])["--resume"] == str(p.start) and TP.flags_of(c["cmd"])["--max-steps"] == "3000"
        assert TP.configuration(c["cmd"], p.src)["fingerprint"] == p.fp_c
    assert kept(c2["cmd"]) == kept(c1["cmd"]) and c2["run"] != c1["run"]
    assert uses(p, p.fp_c) == [str(c1["run"].resolve()), str(c2["run"].resolve())]
    rr = rr_of(p, p.fp_c)
    assert rr["run"] == str(c1["run"].resolve()) and [m["run"] for m in rr["launches"]] == [str(c2["run"].resolve())]
    ctl = json.loads((p.saved / "supervisor_control.json").read_text(encoding="utf-8"))
    assert ctl["repair_run"] == str(c2["run"]) and ctl["cmd"] == "pause"
    run_main(module, monkeypatch, "--tag", "c", "--reason", "the cap arm", "--parent", str(p.a), *CAP)
    cap = p.launched[-1]
    assert TP.configuration(cap["cmd"], p.src)["fingerprint"] == p.fp_r and rr_of(p, p.fp_r)["launched_at"]
    assert {k for k in set(kept(cap["cmd"])) | set(kept(c2["cmd"]))
            if kept(cap["cmd"]).get(k) != kept(c2["cmd"]).get(k)} == {"--attention-cap", "--attention-cap-bound"}
    assert [why for why, _cmd in p.asked] == [None, None, None]
    # a third control and a second cap arm: judged as normal training and refused before anything is ended or started
    for argv in (("--tag", "d", "--reason", "a third control", "--parent", str(p.a)),
                 ("--tag", "e", "--reason", "the cap arm again", "--parent", str(p.a)) + CAP):
        with pytest.raises(SystemExit, match="refused by the training preflight"):
            run_main(module, monkeypatch, *argv)
        assert "gate 1 Information flow: FAIL" in p.asked[-1][0]
    assert len(p.launched) == 3 and len(uses(p, p.fp_c)) == 2 and len(uses(p, p.fp_r)) == 1


def test_control_2_without_parent_resumes_control_1s_end_and_is_refused(tmp_path, monkeypatch):
    module, p = with_two_controls(tmp_path, monkeypatch)
    control(module, monkeypatch, p, "a")
    with pytest.raises(SystemExit, match="refused by the training preflight"):
        run_main(module, monkeypatch, "--tag", "b", "--reason", "a cap-off control")
    assert len(p.launched) == 1 and len(uses(p, p.fp_c)) == 1


def test_a_control_that_crashes_before_training_is_no_use(tmp_path, monkeypatch):
    module, p = with_two_controls(tmp_path, monkeypatch)
    L, launch0 = module.L, module.L.launch_trainer

    class Crashed:
        pid, returncode = 4243, 1

        def poll(self):
            return 1

    def crash_at_load(cmd, run):                 # exits at its load: stage_done.json names the start step
        p.launched.append(dict(cmd=[str(c) for c in cmd], run=run))
        (run / "stage_done.json").write_text(json.dumps({"steps": 1000, "ok": False}), encoding="utf-8")
        return Crashed()
    monkeypatch.setattr(L, "launch_trainer", crash_at_load)
    with pytest.raises(SystemExit, match="exited early with 1"):
        run_main(module, monkeypatch, "--tag", "a", "--reason", "a cap-off control", "--parent", str(p.a))
    rr = rr_of(p, p.fp_c)
    assert len(rr["attempts"]) == 1 and uses(p, p.fp_c) == [] and not rr["launched_at"]
    monkeypatch.setattr(L, "launch_trainer", launch0)
    control(module, monkeypatch, p, "b")
    control(module, monkeypatch, p, "c")
    assert len(uses(p, p.fp_c)) == 2 and len(rr_of(p, p.fp_c)["attempts"]) == 3


def test_a_folder_marked_in_another_path_form_and_noted_as_an_attempt_is_one_use(tmp_path):
    src, root, reg, start, cap, control_cmd, fp_r, fp_c = _experiment_setup(tmp_path)
    block = TP._load(TP.record_path(fp_c, root))["repair_run"]
    TP.set_repair_run(fp_c, dict(block, repeats=2), "tester", root=root, registry=reg)
    run = tmp_path / "experiments" / "Run_C1"
    run.mkdir(parents=True)
    TP.note_attempt(fp_c, str(run), "tester", root=root)
    (run / "replay_training_stats.jsonl").write_text(json.dumps(dict(type="mimo", step=1100)) + "\n", encoding="utf-8")
    TP.mark_launched(fp_c, str(run).replace("\\", "/").upper(), "tester", root=root)
    assert len(TP.repair_uses(TP._load(TP.record_path(fp_c, root))["repair_run"])) == 1
    assert TP.launch_plan(_bounded(control_cmd, 3000), root=root, src=src)["kind"] == "repair_run"


def test_a_failed_attach_after_the_first_model_steps_spends_one_of_the_two_controls(tmp_path):
    # the round-three review's T4, for the repeated control: an attempt stopped before its first full stats record
    # (stage_done.json past the start) is one use, so one real control run remains
    src, root, reg, start, cap, control_cmd, fp_r, fp_c = _experiment_setup(tmp_path)
    block = TP._load(TP.record_path(fp_c, root))["repair_run"]
    TP.set_repair_run(fp_c, dict(block, repeats=2), "tester", root=root, registry=reg)
    stopped = tmp_path / "experiments" / "stopped_for_want_of_a_supervisor"
    stopped.mkdir(parents=True)
    TP.note_attempt(fp_c, str(stopped), "tester", root=root)
    (stopped / "stage_done.json").write_text(json.dumps({"steps": 1050}), encoding="utf-8")
    assert len(TP.repair_uses(TP._load(TP.record_path(fp_c, root))["repair_run"])) == 1


def test_after_the_experiment_neither_configuration_reaches_normal_training(tmp_path):
    src, root, reg, start, cap, control_cmd, fp_r, fp_c = _experiment_setup(tmp_path)
    block = TP._load(TP.record_path(fp_c, root))["repair_run"]
    TP.set_repair_run(fp_c, dict(block, repeats=2), "tester", root=root, registry=reg)
    for fp, name in ((fp_c, "c1"), (fp_c, "c2"), (fp_r, "r")):
        run = tmp_path / "experiments" / name
        run.mkdir(parents=True)
        TP.mark_launched(fp, str(run), "tester", root=root)
    for cmd in (control_cmd, cap):
        assert TP.launch_plan(_bounded(cmd, 3000), root=root, src=src)["kind"] == "normal"
        why = TP.launch_refusal(_bounded(cmd, 3000), root=root, src=src, registry=reg)
        assert why and "gate 1 Information flow: FAIL" in why
    # a second control run's checkpoint cannot become a normal-training state of the control's record: states must
    # continue the first marked run (repair_launch), whatever the repeats
    end2 = tmp_path / "experiments" / "c2" / "checkpoints" / "ckpt_step_3000"
    end2.mkdir(parents=True)
    with pytest.raises(TP.PreflightError, match="is not one"):
        TP.add_state(fp_c, str(end2), "judged again", "tester", root=root)
