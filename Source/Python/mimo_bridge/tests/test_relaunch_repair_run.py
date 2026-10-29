# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The relaunch script's part of the training preflight's REPAIR RUN (the operator, Oct 5, 2026): the launch ends at the
declared bound and passes the preflight's own bound check, the run's stop lines reach the hourly watcher as declared
criteria, and the supervisor is left a pause naming the run, under which it relaunches nothing and keeps the run's
watcher. Nothing here starts or ends a process: the script's functions are called on temporary folders."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from mimo_bridge.tests.test_training_preflight import STOP_CRITERIA, _repair_setup
from mimo_bridge.tools import training_continuation as tc
from mimo_bridge.tools import training_preflight as TP

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "docs/gain_of_function_v4/coordinator/relaunch_h84_seg41_2026-09-29.py"
WATCHER = REPO / "docs/h84_launch_2026-09-18/hourly_watch.py"


def relaunch(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("relaunch_repair_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.L, "EXP", tmp_path / "experiments")
    monkeypatch.setattr(module.L, "stamp", lambda: "Oct 5, 2026, 6:00:00 PM PDT")
    return module


def test_the_relaunch_bounds_a_repair_run_as_its_record_declares_and_the_preflight_allows_it(tmp_path, monkeypatch):
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    module = relaunch(monkeypatch, tmp_path)
    preflight = SimpleNamespace(launch_plan=lambda cmd: TP.launch_plan(cmd, root=root, src=src))
    run = module.run_folder("x")
    cmd, changes, plan = module.launch_command(recorded, start, run, [], preflight)
    flags = TP.flags_of(cmd)
    assert plan["kind"] == "repair_run" and flags["--max-steps"] == "3000"           # start 1000 + 2000 model steps
    assert flags["--resume"] == str(start) and flags["--out"] == str(run)
    assert TP.launch_refusal(cmd, root=root, src=src, registry=reg) is None
    # a --set of a shorter end is kept and judged; one past the bound is refused by the preflight
    cmd, changes, plan = module.launch_command(recorded, start, run, [("max-steps", "2500")], preflight)
    assert TP.flags_of(cmd)["--max-steps"] == "2500"
    assert TP.launch_refusal(cmd, root=root, src=src, registry=reg) is None
    cmd, changes, plan = module.launch_command(recorded, start, run, [("max-steps", "0")], preflight)
    assert "not bounded as declared" in TP.launch_refusal(cmd, root=root, src=src, registry=reg)


def test_a_relaunch_from_another_checkpoint_is_normal_training_and_is_refused_while_the_target_gates_fail(
        tmp_path, monkeypatch):
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    module = relaunch(monkeypatch, tmp_path)
    preflight = SimpleNamespace(launch_plan=lambda cmd: TP.launch_plan(cmd, root=root, src=src))
    later = a / "checkpoints/ckpt_step_1500"
    later.mkdir(parents=True)
    cmd, changes, plan = module.launch_command(recorded, later, module.run_folder("x"), [], preflight)
    assert plan["kind"] == "normal" and TP.flags_of(cmd)["--max-steps"] == "0"
    why = TP.launch_refusal(cmd, root=root, src=src, registry=reg)
    assert "FAIL - REPAIR TARGET" in why and "never normal training" in why


def test_the_repair_runs_stop_lines_reach_the_watcher_and_the_pause_names_the_run(tmp_path, monkeypatch):
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    module = relaunch(monkeypatch, tmp_path)
    plan = TP.launch_plan(recorded, root=root, src=src)
    run = module.run_folder("x")
    run.mkdir(parents=True)
    written = module.write_stop_criteria(run, plan)
    assert written == run / "declared_criteria.json"
    assert json.loads(written.read_text(encoding="utf-8")) == STOP_CRITERIA
    # the real watcher judging this file: tests/test_relaunch_repair_run_review.py (run_watcher)
    saved = tmp_path / "saved"
    saved.mkdir()
    control = module.control_after_launch(plan, "x", run, "3000")
    assert control["cmd"] == "pause" and control["repair_run"] == str(run) and "model step 3000" in control["reason"]
    (saved / "supervisor_control.json").write_text(json.dumps(control), encoding="utf-8")
    assert tc.pause_reason(saved, run) == "explicit supervisor pause"          # nothing is relaunched
    assert tc.repair_run_pause(saved, run) is True                             # the watcher keeps running
    normal = module.control_after_launch(dict(plan, kind="normal"), "x", run, "0")
    assert normal["cmd"] == "run" and "repair_run" not in normal


def test_a_set_of_the_folders_or_the_checkpoint_the_script_chooses_is_refused(tmp_path, monkeypatch):
    # the relaunch review of Oct 5, 2026, finding F7: a --set of --resume launched a checkpoint the script never
    # verified while its receipt named another; a --set of --out sent the trainer where no watcher looked
    src, root, reg, a, start, recorded, fp = _repair_setup(tmp_path)
    module = relaunch(monkeypatch, tmp_path)
    preflight = SimpleNamespace(launch_plan=lambda cmd: TP.launch_plan(cmd, root=root, src=src))
    for flag in ("resume", "--out", "checkpoint-storage"):
        try:
            module.launch_command(recorded, start, module.run_folder("x"), [(flag, str(tmp_path / "elsewhere"))],
                                  preflight)
        except SystemExit as exc:
            assert "refused: --set --" in str(exc) and "--parent" in str(exc)
        else:
            raise AssertionError("a --set of %s was not refused" % flag)


def test_the_controls_other_keys_are_kept(tmp_path, monkeypatch):
    # finding F11: {"archiver": "off"} beside the pause was dropped, and the supervisor then started an archiver
    module = relaunch(monkeypatch, tmp_path)
    found = {"cmd": "pause", "reason": "held", "previous_reason": "older", "archiver": "off"}
    repair = module.control_after_launch(dict(kind="repair_run", fingerprint="f"), "x", tmp_path / "run", "3000",
                                         previous=found)
    normal = module.control_after_launch(dict(kind="normal", fingerprint="f"), "x", tmp_path / "run", "0",
                                         previous=found)
    assert repair["archiver"] == "off" and repair["cmd"] == "pause" and "previous_reason" not in repair
    assert normal == {"archiver": "off", "cmd": "run", "reason": normal["reason"]}
