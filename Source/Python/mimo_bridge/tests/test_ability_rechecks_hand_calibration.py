# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The ability scheduler and the two arms of the finger comparison (Oct 9, 2026; the operator's decision received by
11:53 PM PDT, Oct 8).

During the comparison and after it, while supervisor_control.json holds the pause the relaunch writes for a repair run
({"cmd": "pause", "repair_run": <run>}), the scheduler's cycle returns before it chooses a checkpoint, asks for a
saved window or a snapshot, or starts a test: no arm's checkpoint is taken. When a run of the finger lineage is later
trained and tested, its checkpoints saved the refit hand calibration, and the remembered hand-goal comparison's
--calibration, the admitted fit the manifest names, would be refused at load. The scheduler now passes the calibration
file each checkpoint's receipt names (checkpoint_calibration). The calibration files are the repository's own; the
checkpoints, runs and receipts are temporary."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from mimo_bridge.tests.test_ability_rechecks import tick, tiny_worker  # noqa: F401 (the fixture is used by name)
from mimo_bridge.tools import ability_rechecks as worker

REPO = Path(__file__).resolve().parents[4]
ADMITTED = REPO / "docs/gain_of_function_v3/measurements/hand_response_calibration_380200.json"
REFIT = REPO / "docs/gain_of_function_v3/measurements/hand_response_calibration_finger_joint_servo_v1_481800.json"
MEASURED = REPO / "docs/gain_of_function_v3/measurements/hand_memory_route_2026-09-24.json"
SCRIPT = REPO / "docs/gain_of_function_v3/compare_hand_memory_route_2026-09-24.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_task():
    manifest = json.loads((REPO / "docs/gain_of_function_v3/ability_rechecks.json").read_text(encoding="utf-8"))
    return {t["id"]: t for t in manifest["tasks"]}["hand_memory_route"]


def arm_checkpoint(tmp_path, name, calibration, launched=None, step=484675, receipt=True):
    """A checkpoint of run `name` whose model_receipt.json saved `calibration` (as the trainer saves it: the payload
    with calibration_file_sha256 added) and whose run's args.json names `launched` as --hand-response-calibration."""
    run = tmp_path / "experiments" / name
    checkpoint = run / "checkpoints" / ("ckpt_step_%d" % step)
    checkpoint.mkdir(parents=True)
    if receipt:
        payload = json.loads(Path(calibration).read_text(encoding="utf-8"))
        (checkpoint / "model_receipt.json").write_text(json.dumps(dict(
            hand_response=dict(payload, calibration_file_sha256=sha(calibration)))), encoding="utf-8")
    (run / "args.json").write_text(json.dumps(dict(
        hand_response_calibration=str(launched if launched is not None else calibration))), encoding="utf-8")
    return checkpoint


def calibration_of(command):
    return Path(command[command.index("--calibration") + 1])


def test_the_control_arms_checkpoint_gets_the_admitted_fit_as_today(tmp_path):
    checkpoint = arm_checkpoint(tmp_path, "control", ADMITTED)
    command = worker.task_command(REPO, manifest_task(), checkpoint, tmp_path / "out")
    assert calibration_of(command) == ADMITTED.resolve()
    assert command[3:6] == ["--checkpoint", str(checkpoint), "--out"]


def test_the_finger_arms_checkpoint_gets_the_refit_its_receipt_names(tmp_path):
    checkpoint = arm_checkpoint(tmp_path, "finger", REFIT)
    command = worker.task_command(REPO, manifest_task(), checkpoint, tmp_path / "out")
    assert calibration_of(command) == REFIT.resolve()
    # a checkpoint stored outside its run folder: lineage.json names the run
    stored = tmp_path / "elsewhere" / "ckpt_step_484675"
    stored.mkdir(parents=True)
    (stored / "model_receipt.json").write_bytes((checkpoint / "model_receipt.json").read_bytes())
    (stored / "lineage.json").write_text(json.dumps(dict(out=str(checkpoint.parent.parent))), encoding="utf-8")
    assert calibration_of(worker.task_command(REPO, manifest_task(), stored, tmp_path / "out")) == REFIT.resolve()


def test_without_a_matching_file_the_manifests_file_is_passed_as_before(tmp_path):
    # the run names no file, a file outside the workspace, a file of other bytes, or one whose payload hash differs
    outside = tmp_path / "outside_refit.json"
    outside.write_bytes(REFIT.read_bytes())
    cases = {"names_nothing": "", "outside": outside, "other_bytes": ADMITTED}
    for name, launched in cases.items():
        checkpoint = arm_checkpoint(tmp_path, name, REFIT, launched=launched)
        assert calibration_of(worker.task_command(REPO, manifest_task(), checkpoint, tmp_path / name)) == \
            ADMITTED.resolve(), name
    payload_differs = arm_checkpoint(tmp_path, "payload_differs", REFIT)
    receipt = json.loads((payload_differs / "model_receipt.json").read_text(encoding="utf-8"))
    receipt["hand_response"]["sha256"] = "0" * 64
    (payload_differs / "model_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert calibration_of(worker.task_command(REPO, manifest_task(), payload_differs, tmp_path / "p")) == \
        ADMITTED.resolve()
    # no receipt, an unreadable receipt, a receipt without a hand calibration: the manifest's file
    bare = arm_checkpoint(tmp_path, "no_receipt", REFIT, receipt=False)
    broken = arm_checkpoint(tmp_path, "broken_receipt", REFIT)
    (broken / "model_receipt.json").write_text("{", encoding="utf-8")
    handless = arm_checkpoint(tmp_path, "handless", REFIT)
    (handless / "model_receipt.json").write_text(json.dumps(dict(hand_response=None)), encoding="utf-8")
    for checkpoint in (bare, broken, handless, tmp_path / "missing" / "ckpt_step_1"):
        assert calibration_of(worker.task_command(REPO, manifest_task(), checkpoint, tmp_path / "x")) == \
            ADMITTED.resolve()


def test_other_tasks_and_other_flags_are_not_bound_to_the_checkpoint(tmp_path):
    checkpoint = arm_checkpoint(tmp_path, "finger", REFIT)
    other = dict(manifest_task(), script="docs/gain_of_function_v3/qualify_hand_execution_2026-09-24.py")
    assert calibration_of(worker.task_command(REPO, other, checkpoint, tmp_path / "o")) == ADMITTED.resolve()
    renamed = dict(manifest_task(), path_arguments={"--response": str(ADMITTED.relative_to(REPO))})
    command = worker.task_command(REPO, renamed, checkpoint, tmp_path / "r")
    assert Path(command[command.index("--response") + 1]) == ADMITTED.resolve()
    # the validation pass the planner makes without a checkpoint is unchanged
    assert worker.path_arguments(REPO, manifest_task()) == ["--calibration", str(ADMITTED.resolve())]


def test_the_validator_binds_the_result_to_the_file_the_scheduler_passed(tmp_path):
    checkpoint = arm_checkpoint(tmp_path, "finger", REFIT)
    record = dict(checkpoint=str(checkpoint), script_sha256=sha(SCRIPT),
                  command=worker.task_command(REPO, manifest_task(), checkpoint, tmp_path / "out"))
    # the filed result of Sep 24 in the form the script writes; its script_sha256 is the script of that day (the script
    # changed since: 9c1daed3a), so the current script's hash is put in, as a round of today's script would record it
    payload = json.loads(MEASURED.read_text(encoding="utf-8"))
    payload = dict(copy.deepcopy(payload), checkpoint=str(checkpoint), calibration_file_sha256=sha(REFIT),
                   script_sha256=sha(SCRIPT))
    worker.validate_result(payload, record)
    with pytest.raises(ValueError, match="calibration identity"):
        worker.validate_result(dict(payload, calibration_file_sha256=sha(ADMITTED)), record)


def test_the_repair_run_pause_keeps_every_arm_checkpoint_from_the_scheduler(tiny_worker, monkeypatch):
    """The relaunch writes {"cmd": "pause", "repair_run": <run>} before an arm's trainer starts and leaves it after the
    trainer ends (relaunch_h84_seg41_2026-09-29.control_after_launch). A due checkpoint of that run is not taken: no
    window is asked for, no snapshot request reaches the run, no test starts."""
    from mimo_bridge.tools import ability_comparison_window
    requested = []
    monkeypatch.setattr(ability_comparison_window, "request", lambda *a: requested.append(a) or False)
    run = tiny_worker["checkpoint"].parent.parent
    control = tiny_worker["root"] / "Saved/mimo_bridge/supervisor_control.json"
    control.write_text(json.dumps({"cmd": "pause", "repair_run": str(run), "reason": "repair run of configuration "
                                   "fb4042699f77030a: it ends at model step 484675 and is not continued automatically"}))
    for trainer in (True, False):                                   # while the arm trains, and after it ended
        monkeypatch.setattr(worker, "trainer_running", lambda: trainer)
        status = tick(tiny_worker)
        assert status["state"] == "production paused for repair" and status["pause_reason"]
        assert "attempts" not in status["tasks"]["tiny"] and "command" not in status["tasks"]["tiny"]
    assert not requested and not (run / "control.json").exists()
    assert not list(tiny_worker["folder"].glob("*.process.json"))
