# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The continuation rehearsal's checks (docs/gain_of_function_v4/remembered_hand_learning_2026-10-11/
continuation_rehearsal.py) on a known-good fixture and on planted faults (the operator's rule of Oct 10, 2026: a
known-good case must pass, and deliberately broken settings must fail)."""
import importlib.util
import json
from pathlib import Path

import pytest

TOOL = (Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4" / "remembered_hand_learning_2026-10-11"
        / "continuation_rehearsal.py")
spec = importlib.util.spec_from_file_location("continuation_rehearsal", TOOL)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

STOP_LIST = [dict(name="joint", path="hand_control.joint_error_deg", op=">", line=15.0, mode="stop"),
             dict(name="nan", path="nan_steps", op=">", line=0, mode="stop"),
             dict(name="vision", path="prediction_audit_per_body.senses.vision.without_body_2.ratio", op=">",
                  line=0.94, mode="stop")]


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def good_arm(tmp, start=484675, steps=100, resume="D:/x/latest.pt", first=2874, keep=True, receipt=None,
             lessons=None, extra_events=(), full_step=None, nan=0, vision=0.7, joint=12.0):
    arm = tmp / "arm"
    (arm / "run").mkdir(parents=True, exist_ok=True)
    (arm / "arm.log").write_text(("WARNING --replace-saved-hand-calibration: x %s\n" % R.KEEP_LINE) if keep else "log\n",
                                 encoding="utf-8")
    lessons = steps - 2 if lessons is None else lessons
    events = [dict(type="started", resume=resume)] + [dict(type="latent_body_lesson", updates=first + i)
                                                     for i in range(lessons)] + list(extra_events)
    write_lines(arm / "latent_body" / "events.jsonl", events)
    full = dict(type="mimo", step=start + steps if full_step is None else full_step, nan_steps=nan,
                hand_control=dict(joint_error_deg=joint),
                prediction_audit_per_body=dict(senses=dict(vision=dict(without_body_2=dict(ratio=vision)))))
    write_lines(arm / "run" / "replay_training_stats.jsonl", [dict(type="launch", step=start), full])
    end = arm / "run" / "checkpoints" / ("ckpt_step_%d" % (start + steps))
    end.mkdir(parents=True, exist_ok=True)
    (end / "model_receipt.json").write_text(json.dumps(dict(hand_response=dict(
        sha256=R.CALIBRATION_SHA256 if receipt is None else receipt))), encoding="utf-8")
    (end / "model.safetensors").write_bytes(b"x")
    (arm / "run" / "comparison_records").mkdir(parents=True, exist_ok=True)
    (arm / "run" / "comparison_records" / "steps.jsonl").write_text("{}\n", encoding="utf-8")
    return arm, end


def test_known_good_arm_passes_every_check(tmp_path):
    arm, end = good_arm(tmp_path)
    assert R.check_calibration_kept(arm, end)["ok"]
    assert R.check_latent_continues(arm / "latent_body", "D:/x/latest.pt", 2874, 100)["ok"]
    assert R.check_stats(arm, 484675, 100, STOP_LIST)["ok"]
    assert R.check_comparison_records(arm)["ok"]


def test_calibration_replaced_or_another_saved_fails(tmp_path):
    arm, end = good_arm(tmp_path / "a", keep=False)
    assert not R.check_calibration_kept(arm, end)["ok"]
    arm, end = good_arm(tmp_path / "b", receipt="0" * 64)
    assert not R.check_calibration_kept(arm, end)["ok"]


@pytest.mark.parametrize("kw,why", [
    (dict(first=1), "the latent learner started fresh, not from its saved 2,873 updates"),
    (dict(resume="D:/other/latest.pt"), "another saved state named"),
    (dict(lessons=10), "too few lessons for the run's model steps"),
    (dict(extra_events=[dict(type="failed", error="x")]), "a failure record"),
    (dict(extra_events=[dict(type="excluded_transitions", transitions=[dict(env_id=0)])]), "an excluded transition"),
])
def test_latent_planted_faults_fail(tmp_path, kw, why):
    arm, _end = good_arm(tmp_path, **kw)
    assert not R.check_latent_continues(arm / "latent_body", "D:/x/latest.pt", 2874, 100)["ok"], why


def test_latent_gap_in_updates_fails(tmp_path):
    arm, _end = good_arm(tmp_path)
    lines = R.read_lines(arm / "latent_body" / "events.jsonl")
    lines = [r for r in lines if r.get("updates") != 2900]
    write_lines(arm / "latent_body" / "events.jsonl", lines)
    assert not R.check_latent_continues(arm / "latent_body", "D:/x/latest.pt", 2874, 100)["ok"]


@pytest.mark.parametrize("kw", [dict(full_step=484700), dict(nan=1), dict(vision=float("nan")), dict(joint=None)])
def test_stats_planted_faults_fail(tmp_path, kw):
    arm, _end = good_arm(tmp_path, **kw)
    assert not R.check_stats(arm, 484675, 100, STOP_LIST)["ok"]


def test_stats_wrong_launch_step_fails(tmp_path):
    arm, _end = good_arm(tmp_path, start=481800)
    assert not R.check_stats(arm, 484675, 100, STOP_LIST)["ok"]


def test_missing_comparison_records_fail(tmp_path):
    arm, _end = good_arm(tmp_path)
    import shutil
    shutil.rmtree(arm / "run" / "comparison_records")
    assert not R.check_comparison_records(arm)["ok"]


def test_with_flag_replaces_only_that_value():
    cmd = ["py", "-m", "mimo_bridge.train_mimo", "--a", "1", "--latent-body-resume", "old", "--b", "2"]
    out = R.with_flag(cmd, "--latent-body-resume", "new")
    assert out == ["py", "-m", "mimo_bridge.train_mimo", "--a", "1", "--latent-body-resume", "new", "--b", "2"]
    with pytest.raises(ValueError):
        R.with_flag(cmd, "--missing", "x")
