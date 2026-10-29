# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/finger_command_interface/hand_calibration_admission.py without a model: the binding of the
candidate (part B), the record the comparison writes (part C) read by the trainer's replacement check (part A), and the
record check. No checkpoint is loaded and no world is built here.

  THE BINDING - the bound receipt keeps every field of the candidate but three (source_model_sha256 set to the model's
  SHA-256, its note, its status), carries the candidate's payload and file SHA-256, and passes validate_receipt; a
  candidate with a broken hash, the pose-servo meaning or an existing binding is refused. On the real candidate of row
  hand_response_refit_on_481800 (read only, when present) the coefficients are the candidate's, bit for bit.
  THE RECORD - finish() on 16 decisions of bodies 4-7 writes a record that LemLite's replacement accepts end to end
  (the real trainer code hash of this checkout); fewer decisions, a validity problem or a failed line give 'invalid',
  'invalid' and 'fail'; verify re-checks the record, the script's bytes and the candidate.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch

from lem.model import hand_response as HR
from lem.model.hand_response import content_hash
from mimo_bridge.action_decoder import FINGER_SERVO_CONTRACT
from test_hand_response import fixture as response_fixture

REPO = Path(__file__).resolve().parents[4]
TOOL_DIR = REPO / "docs/gain_of_function_v4/finger_command_interface"
CANDIDATE = Path("D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/hand_refit/run_481800/"
                 "candidate_hand_response_finger_joint_servo_v1.json")
GEOMETRY = Path("D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/hand_refit/geometry_481800.json")


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL_DIR))
    return importlib.import_module("hand_calibration_admission")


def candidate_of(geometry, receipt):
    c = {k: v for k, v in receipt.items() if k != "sha256"}
    c.update(geometry_sha256=content_hash(geometry), command_meaning=FINGER_SERVO_CONTRACT, source_model_sha256=None,
             source_model_note="no learner was built", receipt_status="candidate, not admitted",
             coefficients=[[0.01, .7, .2]])
    c["sha256"] = content_hash(c)
    return c


def test_the_binding_keeps_the_candidate_and_names_the_model(tool):
    geometry, receipt, _ = response_fixture()
    cand = candidate_of(geometry, receipt)
    bound = tool.bind_candidate(cand, "f" * 64, "X:/run/checkpoints/ckpt_step_481800", "a" * 64, "now", "b" * 64)
    assert bound["source_model_sha256"] == "a" * 64 and bound["sha256"] == content_hash(
        {k: v for k, v in bound.items() if k != "sha256"})
    changed = {k for k in set(cand) | set(bound) if cand.get(k) != bound.get(k)}
    assert changed == {"source_model_sha256", "source_model_note", "receipt_status", "admission_binding", "sha256"}
    b = bound["admission_binding"]
    assert b["candidate_payload_sha256"] == cand["sha256"] and b["candidate_file_sha256"] == "f" * 64
    assert b["model_sha256"] == "a" * 64 and b["checkpoint_step"] == 481800 and b["rule"] == tool.BINDING_RULE
    assert b["candidate_receipt_status"] == "candidate, not admitted"
    HR.validate_receipt(bound, geometry)
    assert torch.equal(HR.CalibratedHandResponse(geometry, bound).coefficients,
                       torch.tensor(cand["coefficients"], dtype=torch.float64))
    broken = dict(cand, ridge=0.5)
    with pytest.raises(ValueError, match="content hash"):
        tool.bind_candidate(broken, "f" * 64, "c", "a" * 64, "now", "b" * 64)
    pose = {k: v for k, v in cand.items() if k not in ("sha256", "command_meaning")}
    pose["sha256"] = content_hash(pose)
    with pytest.raises(ValueError, match="command meaning"):
        tool.bind_candidate(pose, "f" * 64, "c", "a" * 64, "now", "b" * 64)
    with pytest.raises(ValueError, match="already bound"):
        tool.bind_candidate(bound, "f" * 64, "c", "a" * 64, "now", "b" * 64)


def test_the_real_candidate_binds_with_its_coefficients_unchanged(tool):
    if not CANDIDATE.is_file() or not GEOMETRY.is_file():
        pytest.skip("the refit's candidate or geometry is not on this machine")
    raw = CANDIDATE.read_bytes()
    cand = json.loads(raw)
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    bound = tool.bind_candidate(cand, hashlib.sha256(raw).hexdigest(), tool.CKPT_481800, "a" * 64, "now", "b" * 64)
    HR.validate_receipt(bound, geometry)
    assert bound["coefficients"] == cand["coefficients"] and bound["geometry_sha256"] == content_hash(geometry)
    assert HR.command_meaning(bound) == FINGER_SERVO_CONTRACT and bound["training_examples"] == 624


def test_paths_under_saved_or_the_measurements_folder_are_refused(tool, tmp_path):
    for bad in (tool.SAVED_ROOT + "/mimo_bridge/x.json",
                str(REPO / "docs/gain_of_function_v3/measurements/x.json")):
        with pytest.raises(SystemExit):
            tool.refuse_path(bad)
    tool.refuse_path(tmp_path / "fine.json")


def rows_of(differences, bodies=(4, 5, 6, 7), per_body=4):
    rows, k = [], 0
    for b in bodies:
        for s in range(per_body):
            d = differences[k]
            k += 1
            end = {"without": 2.0, "refit": 2.0 + d, "admitted_forced": 2.0 + d / 2}
            rows.append(dict(body=b, start=s, difference=d, difference_admitted_forced=d / 2,
                             difference_refit_minus_admitted_forced=d / 2, sensory_goals_identical=True,
                             at_decision=dict(hand_shape_error=3.0),
                             arms={arm: dict(end=dict(hand_shape_error=e, finger_position_error_cm=1.0,
                                                      finger_orientation_error_deg=30.0),
                                             closed_share=1.0 - e / 3.0) for arm, e in end.items()}))
    return rows


DIFFS = [-0.31, -0.22, -0.41, -0.18, -0.27, -0.35, -0.12, -0.44, -0.29, -0.16, -0.38, -0.24, -0.33, -0.20, -0.47, -0.26]


def test_the_comparisons_record_is_accepted_by_the_trainers_replacement(tool, tmp_path, monkeypatch, capsys):
    """End to end without a model run: the bound receipt and finish()'s record on 16 decisions, read by
    LemLite._restore_hand_response_calibration with --replace-saved-hand-calibration, with the real trainer code hash
    of this checkout; then the tool's verify on the same files."""
    from mimo_bridge.lem_model import LemLite
    monkeypatch.setattr(tool, "setup_imports", lambda threads: None)     # the test process keeps its environment
    torch.set_num_threads(1)
    model = LemLite(device="cpu", no_vision=True, imagination=False, next_frame_loss=True, intention=True,
                    hand_target_mode="wrist_relative_v1", hand_goals="memory", smooth_actions=.35,
                    smooth_source="executed", overrides={"model": {"context_length": 2}})
    model.finger_joint_servo = "on"
    geometry, receipt, _ = response_fixture()
    geometry["handle_names"] = model.handle_names
    ckpt = tmp_path / "ckpt_step_481800"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"weights")
    model_sha = hashlib.sha256(b"weights").hexdigest()
    saved = {k: v for k, v in receipt.items() if k != "sha256"}
    saved["geometry_sha256"] = content_hash(geometry)
    saved["sha256"] = content_hash(saved)
    (ckpt / "model_receipt.json").write_text(json.dumps({"hand_response": dict(saved, calibration_file_sha256="9" * 64)}),
                                             encoding="utf-8")
    cand = candidate_of(geometry, receipt)
    cand_file = tmp_path / "candidate.json"
    tool.write_json(cand_file, cand)
    bound = tool.bind_candidate(cand, tool.sha256_file(cand_file), ckpt, model_sha, "now", "b" * 64,
                                candidate_path=cand_file)
    receipt_file = tmp_path / "bound.json"
    tool.write_json(receipt_file, bound)
    base = dict(version=HR.ADMISSION_VERSION, checkpoint=str(ckpt), model_sha256=model_sha,
                receipt_payload_sha256=bound["sha256"], receipt_file_sha256=tool.sha256_file(receipt_file),
                finger_joint_servo="on", command_meaning=FINGER_SERVO_CONTRACT, tested_route=HR.ADMISSION_ROUTE,
                learning=False, code_hash=HR.trainer_code_hash(), script=dict(path="x", sha256="y"))
    record = tool.finish(base, rows_of(DIFFS), [], {}, [4, 5, 6, 7], bound["geometry_sha256"], bound,
                         weights_unchanged=True)
    assert record["verdict"] == "pass" and record["line"]["P1"] and record["line"]["P2"]
    assert record["sha256"] == content_hash({k: v for k, v in record.items() if k != "sha256"})
    record_file = tmp_path / "record.json"
    tool.write_json(record_file, record)
    model.configure_hand_response_calibration(receipt_file)
    model.replace_saved_hand_calibration = str(record_file)
    model._restore_hand_response_calibration(ckpt)
    assert model._hand_response_calibration["sha256"] == bound["sha256"]
    assert model.receipt["hand_calibration_replacements"][-1]["admission"]["sha256"] == record["sha256"]
    model.configure_hand_geometry(geometry, torch.zeros(2, 1))
    assert torch.equal(model.hand_response.coefficients, torch.tensor([[0.01, .7, .2]], dtype=torch.float64))
    # the same record re-read by the tool's verify: the trainer's checks pass and the candidate is kept, while the
    # script the record names ('x') is not found, so verify answers 1
    capsys.readouterr()
    code = tool.main(["verify", "--record", str(record_file), "--receipt", str(receipt_file), "--checkpoint", str(ckpt)])
    printed = json.loads(capsys.readouterr().out)
    assert code == 1 and printed["ok"] is False
    assert isinstance(printed["checks"]["admission_check"], dict)
    assert printed["checks"]["candidate_kept"] is True and printed["checks"]["script_unchanged"] is False


def test_a_short_an_invalid_and_a_failing_run_are_named(tool):
    geometry, receipt, _ = response_fixture()
    bound = tool.bind_candidate(candidate_of(geometry, receipt), "f" * 64, "c", "a" * 64, "now", "b" * 64)
    base = dict(version=HR.ADMISSION_VERSION, model_sha256="a" * 64)
    short = tool.finish(base, rows_of(DIFFS[:1], bodies=(4,), per_body=1), [], {"4": dict(decisions=1)}, [4],
                        bound["geometry_sha256"], bound, weights_unchanged=True)
    assert short["verdict"] == "invalid" and "development run" in short["verdict_reason"]
    broken = tool.finish(base, rows_of(DIFFS), ["body 4 decision 0: the repeated hold did not reproduce exactly"], {},
                         [4, 5, 6, 7], bound["geometry_sha256"], bound, weights_unchanged=True)
    assert broken["verdict"] == "invalid" and broken["exact_replay"] is False
    worse = tool.finish(base, rows_of([abs(d) for d in DIFFS]), [], {}, [4, 5, 6, 7], bound["geometry_sha256"], bound,
                        weights_unchanged=True)
    assert worse["verdict"] == "fail" and "P1 not met" in worse["verdict_reason"]
    other = tool.finish(base, rows_of(DIFFS), [], {}, [4, 5, 6, 7], "e" * 64, bound, weights_unchanged=True)
    assert other["verdict"] == "invalid" and "hand geometry" in other["verdict_reason"]


def test_the_lockstep_sees_the_setting_only_from_the_first_completed_hand_stage(monkeypatch):
    """hca_lockstep.sensitivity_rule on invented child records: one replacement record (admitted replaced by the
    synthetic calibration), the episode state restored the admitted one, the hand stage predicted with the synthetic
    one, and the first differing model step not before the first completed hand stage (a plan's recorded step is one
    more than its row's model step)."""
    monkeypatch.syspath_prepend(str(TOOL_DIR))
    L = importlib.import_module("hca_lockstep")
    rep = dict(replacements=[dict(replaced=dict(payload_sha256="A"), replaced_by=dict(payload_sha256="S"))],
               calibration_after_episode_state="A", predictor_sha256="S", admitted_sha256="A",
               synthetic_calibration_sha256="S", plans=[dict(step=31, hand_stage="awaiting_native_joint_readings"),
                                                        dict(step=41, hand_stage="completed")])
    off = dict(predictor_sha256="A")
    assert L.sensitivity_rule(off, rep, dict(first_difference=dict(step=40, fields=["command", "hand_goal"])))["pass"]
    assert not L.sensitivity_rule(off, rep, dict(first_difference=dict(step=39, fields=["loss"])))["pass"]
    assert not L.sensitivity_rule(off, rep, dict(first_difference=None))["pass"]
    kept = dict(rep, predictor_sha256="A")
    assert not L.sensitivity_rule(off, kept, dict(first_difference=dict(step=40, fields=["command"])))["pass"]
    rows_a = dict(rows=[dict(step=30, x=1), dict(step=31, x=2)], final_state="f", plans=[])
    rows_b = dict(rows=[dict(step=30, x=1), dict(step=31, x=3)], final_state="f", plans=[])
    assert L.compare(rows_a, rows_b)["first_difference"] == dict(step=31, fields=["x"])
