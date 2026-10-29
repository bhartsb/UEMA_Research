# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--replace-saved-hand-calibration (docs/gain_of_function_v4/finger_command_interface/
HAND_CALIBRATION_ADMISSION_DESIGN.md, part A): the declared replacement of a checkpoint's saved hand calibration.

Two directions:
  WITHOUT THE SETTING - LemLite._restore_hand_response_calibration computes what production's computes (production's
  function read with git show from the commit this branch starts from, run on the same learner and the same files:
  the same refusal text or the same calibration and receipt, and no new attribute on the learner) and never computes
  the trainer code hash; the trainer's parser reads '' and the builder sets nothing; the acting configuration holds no
  new key.
  WITH THE SETTING - a requested calibration that differs from the saved one replaces it only under an admission
  record that passes every binding (a JSON object; payload and file bytes, the checkpoint's model weights named by the
  record and by the calibration's source_model_sha256, the trainer code, the finger rule and both command meanings, the
  hand geometry, the declared line recomputed from its decisions with P1 and P2 each required, verdict pass and passed
  true); each binding that fails refuses the resume; the world's hand geometry is checked when the hand path is set
  up; the replacement survives the episode companion's restore of the saved calibration; train_mimo.main's own
  statements write the replacement's stats record after the hand path's set-up; a later resume carries the record;
  the trainer code hash is the training preflight's (one function, lem/trainer_code.py).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from lem.model import hand_response as HR
from lem.model.hand_response import admission_line, content_hash
from mimo_bridge.action_decoder import FINGER_SERVO_CONTRACT
from mimo_bridge.lem_model import LemLite
from test_hand_response import fixture as response_fixture

REPO = Path(__file__).resolve().parents[4]
# production's HEAD this branch starts from (fix/explainer-audit-2026-09-15, Oct 8, 2026 4:06 PM PDT)
BASE = "648215b3a"
CODE = "c" * 64
DIFFERENCES = [-0.31, -0.22, -0.41, -0.18, -0.27, -0.35, -0.12, -0.44,
               -0.29, -0.16, -0.38, -0.24, -0.33, -0.20, -0.47, -0.26]


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    return LemLite(device="cpu", no_vision=True, imagination=False, next_frame_loss=True, intention=True,
                   hand_target_mode="wrist_relative_v1", hand_goals="memory", smooth_actions=.35,
                   smooth_source="executed", overrides={"model": {"context_length": 2}})


def payload(geometry, receipt, **extra):
    r = {k: v for k, v in dict(receipt, **extra).items() if k != "sha256"}
    r["geometry_sha256"] = content_hash(geometry)
    r["sha256"] = content_hash({k: v for k, v in r.items() if k != "sha256"})
    return r


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decisions(values=DIFFERENCES):
    return [dict(body=b, start=s, difference=float(values[4 * i + s])) for i, b in enumerate(HR.ADMISSION_BODIES)
            for s in range(HR.ADMISSION_STARTS_PER_BODY)]


def make_record(requested, file_sha, model_sha, *, code=CODE, rule="on", rehash=True, **changes):
    rec = dict(version=HR.ADMISSION_VERSION, receipt_payload_sha256=requested["sha256"], receipt_file_sha256=file_sha,
               model_sha256=model_sha, code_hash=code, finger_joint_servo=rule, command_meaning=FINGER_SERVO_CONTRACT,
               geometry_sha256=requested["geometry_sha256"], tested_route=HR.ADMISSION_ROUTE,
               held_out_bodies=list(HR.ADMISSION_BODIES), weights_unchanged=True, exact_replay=True,
               nonfinger_commands_unchanged=True, sensory_goals_identical=True, hand_stage_completed=True,
               learning=False, validity_problems=[], decisions=decisions(), checkpoint="ckpt_step_10",
               written="Oct 8, 2026 5:00:00 PM PDT")
    rec["line"] = admission_line(rec["decisions"])
    rec.update(verdict="pass", passed=True)
    rec.update(changes)
    if "decisions" in changes and "line" not in changes:
        rec["line"] = admission_line(rec["decisions"])
    if rehash:
        rec["sha256"] = content_hash({k: v for k, v in rec.items() if k != "sha256"})
    return rec


@pytest.fixture
def case(model, tmp_path, monkeypatch):
    """A checkpoint that saved a pose-servo calibration, a requested joint-servo calibration bound to the checkpoint's
    weights, and a passing record; the learner's hand state reset to a fresh launch's."""
    monkeypatch.setattr(HR, "trainer_code_hash", lambda source_python=None: CODE)
    geometry, receipt, _ = response_fixture()
    geometry["handle_names"] = model.handle_names
    ckpt = tmp_path / "ckpt_step_10"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"the model weights the comparison ran")
    model_sha = hashlib.sha256((ckpt / "model.safetensors").read_bytes()).hexdigest()
    saved_payload = payload(geometry, receipt)
    saved_file = tmp_path / "admitted.json"
    saved_sha = write_json(saved_file, saved_payload)
    saved = dict(saved_payload, calibration_file_sha256=saved_sha)
    write_json(ckpt / "model_receipt.json", {"hand_response": saved, "body_layout": model.body_layout})
    requested_payload = payload(geometry, receipt, coefficients=[[0.01, .7, .2]], command_meaning=FINGER_SERVO_CONTRACT,
                                source_model_sha256=model_sha)
    requested_file = tmp_path / "bound.json"
    requested_sha = write_json(requested_file, requested_payload)
    record = make_record(requested_payload, requested_sha, model_sha)
    record_file = tmp_path / "record.json"
    write_json(record_file, record)
    for name in ("replace_saved_hand_calibration", "hand_calibration_replacement"):
        if name in vars(model):
            delattr(model, name)
    model._hand_response_calibration = None
    model.hand_response = None
    model.hand_goals = "memory"
    model.finger_joint_servo = "on"
    model.receipt.pop("hand_response", None)
    model.receipt.pop("hand_calibration_replacements", None)
    return dict(model=model, geometry=geometry, receipt=receipt, ckpt=ckpt, model_sha=model_sha, saved=saved,
                saved_file=saved_file, requested_payload=requested_payload, requested_file=requested_file,
                requested_sha=requested_sha, record=record, record_file=record_file, tmp=tmp_path)


def production_restore():
    """Production's LemLite._restore_hand_response_calibration at BASE, read with git show and compiled as a function of
    the mimo_bridge package (its relative import resolves there). Skips when git or the commit is not readable."""
    try:
        src = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(REPO), "show",
                              "%s:Source/Python/mimo_bridge/lem_model.py" % BASE],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip("production commit %s not readable: %s" % (BASE, exc))
    start = src.index("    def _restore_hand_response_calibration(self, checkpoint):")
    end = src.index("\n    def ", start + 10)
    body = "\n".join(line[4:] for line in src[start:end].splitlines())
    namespace = dict(__name__="mimo_bridge._production_restore", __package__="mimo_bridge", json=json, copy=copy,
                     Path=Path)
    exec(compile(body, "lem_model.py@%s" % BASE, "exec"), namespace)
    return namespace["_restore_hand_response_calibration"]


def outcome(fn, model, ckpt, calibration, hand_goals):
    """(error text or None, calibration after, receipt's hand_response after, attribute names) of one call."""
    model._hand_response_calibration = copy.deepcopy(calibration)
    model.hand_goals = hand_goals
    model.receipt.pop("hand_response", None)
    names = set(vars(model))
    try:
        fn(model, ckpt)
        error = None
    except ValueError as exc:
        error = str(exc)
    return (error, copy.deepcopy(model._hand_response_calibration), copy.deepcopy(model.receipt.get("hand_response")),
            set(vars(model)) - names)


# ------------------------------------------------------------------------------------------------ without the setting
def situations(case):
    """(checkpoint, requested calibration, hand goals) of seven resumes without the setting: the saved calibration with
    nothing, the same or another requested; no saved calibration with a requested one bound to these weights or to
    others, or none (hand goals on and off)."""
    ckpt = case["ckpt"]
    requested = dict(case["requested_payload"], calibration_file_sha256=case["requested_sha"])
    unbound = copy.deepcopy(requested)
    unbound["source_model_sha256"] = "d" * 64
    bare = case["tmp"] / "bare"
    bare.mkdir(exist_ok=True)
    (bare / "model.safetensors").write_bytes((ckpt / "model.safetensors").read_bytes())
    return [(ckpt, None, "memory"), (ckpt, case["saved"], "memory"), (ckpt, requested, "memory"),
            (bare, requested, "memory"), (bare, unbound, "memory"), (bare, None, "memory"), (bare, None, "off")]


def test_without_the_setting_the_restore_computes_what_productions_computes(case):
    """Seven situations (situations), production's function and the branch's on the same learner and files. Same
    refusal text or same calibration, same receipt, no new attribute."""
    prod = production_restore()
    m = case["model"]
    seen_errors = 0
    for where, calibration, goals in situations(case):
        a = outcome(prod, m, where, calibration, goals)
        b = outcome(LemLite._restore_hand_response_calibration, m, where, calibration, goals)
        assert a == b, (where.name, goals)
        seen_errors += a[0] is not None
        assert "hand_calibration_replacements" not in m.receipt
    assert seen_errors == 3          # the differing request, the unbound request, hand goals without a calibration


def test_without_the_setting_the_trainer_code_hash_is_never_computed(case, monkeypatch):
    """The trainer code hash (lem/trainer_code.py: every file of the trainer code read and hashed) is read by the
    admission check alone: in the seven resumes without the setting neither the hash nor the file reading under it
    runs; with the setting and a differing request it runs once (so the counter sees a call when there is one)."""
    from lem import trainer_code as TC
    calls = []
    real = TC.file_hashes
    monkeypatch.setattr(TC, "file_hashes", lambda *a, **k: calls.append("file_hashes") or real(*a, **k))
    monkeypatch.setattr(HR, "trainer_code_hash", lambda source_python=None: calls.append("trainer_code_hash") or CODE)
    m = case["model"]
    for where, calibration, goals in situations(case):
        outcome(LemLite._restore_hand_response_calibration, m, where, calibration, goals)
    assert calls == []
    m.hand_goals = "memory"
    m.configure_hand_response_calibration(case["requested_file"])
    m.replace_saved_hand_calibration = str(case["record_file"])
    m._restore_hand_response_calibration(case["ckpt"])
    assert calls == ["trainer_code_hash"] and m.hand_calibration_replacement is not None


def test_without_the_setting_the_trainer_builds_the_production_learner(tmp_path):
    """parse_args reads '' without the flag; build_learner sets no attribute for it; the acting configuration holds
    no key for it. With the flag, all three carry it."""
    import motor_asked_change_fixture as fx
    from mimo_bridge import train_mimo as T
    torch.set_num_threads(1)
    args = T.parse_args(fx.argv(n_envs=1, out=str(tmp_path / "out")))
    args.n_envs = int(args.n_envs)
    assert args.replace_saved_hand_calibration == ""
    assert "replace_saved_hand_calibration" not in T.acting_configuration(args)
    learner = T.build_learner(args, None)
    assert "replace_saved_hand_calibration" not in vars(learner)
    assert "hand_calibration_replacement" not in vars(learner)
    on = T.parse_args(fx.argv(n_envs=1, out=str(tmp_path / "out2"),
                              extra=["--replace-saved-hand-calibration", str(tmp_path / "record.json")]))
    on.n_envs = int(on.n_envs)
    assert T.acting_configuration(on)["replace_saved_hand_calibration"] == str(tmp_path / "record.json")
    assert T.build_learner(on, None).replace_saved_hand_calibration == str(tmp_path / "record.json")


# ------------------------------------------------------------------------------------------------ with the setting
def test_a_passing_record_replaces_the_saved_calibration_and_records_it(case):
    m = case["model"]
    m.configure_hand_response_calibration(case["requested_file"])
    m.replace_saved_hand_calibration = str(case["record_file"])
    m._restore_hand_response_calibration(case["ckpt"])
    requested = dict(case["requested_payload"], calibration_file_sha256=case["requested_sha"])
    assert m._hand_response_calibration == requested and m.receipt["hand_response"] == requested
    report = m.receipt["hand_calibration_replacements"][-1]
    assert len(m.receipt["hand_calibration_replacements"]) == 1
    assert report["replaced"]["payload_sha256"] == case["saved"]["sha256"]
    assert report["replaced"]["command_meaning"] == HR.POSE_SERVO_MEANING
    assert report["replaced_by"]["payload_sha256"] == requested["sha256"]
    assert report["replaced_by"]["command_meaning"] == FINGER_SERVO_CONTRACT
    assert report["model_sha256"] == case["model_sha"] and report["admission"]["sha256"] == case["record"]["sha256"]
    assert report["admission"]["file_sha256"] == hashlib.sha256(case["record_file"].read_bytes()).hexdigest()
    assert report["launch"]["finger_joint_servo"] == "on" and report["world_geometry_sha256"] is None
    assert "calibration" not in report
    # the episode companion restores the checkpoint's own calibration after the load (continuation_state carries every
    # _* attribute); the hand path's set-up checks the world's geometry and keeps the replacement in force
    m._hand_response_calibration = copy.deepcopy(case["saved"])
    m.configure_hand_geometry(case["geometry"], torch.zeros(2, 1))
    assert torch.equal(m.hand_response.coefficients, torch.tensor([[0.01, .7, .2]], dtype=torch.float64))
    assert HR.command_meaning(m.hand_response.receipt) == FINGER_SERVO_CONTRACT
    assert m._hand_response_calibration == requested
    report = m.receipt["hand_calibration_replacements"]
    assert len(report) == 1 and report[0]["world_geometry_sha256"] == content_hash(case["geometry"])
    assert m.hand_calibration_replacement_report()["world_geometry_sha256"] == content_hash(case["geometry"])


def test_the_episode_companion_carries_the_saved_calibration_under_a_private_name(case):
    """The reason the replacement is put back in configure_hand_geometry: capture_learner keeps
    _hand_response_calibration (every _* attribute) and restore_learner writes it back after the checkpoint load."""
    from mimo_bridge import continuation_state as CS
    m = case["model"]
    m._hand_response_calibration = copy.deepcopy(case["saved"])
    state = CS.capture_learner(m, include_replay=False)
    assert state["data"]["_hand_response_calibration"] == case["saved"]
    assert "replace_saved_hand_calibration" not in state["data"] and "hand_calibration_replacement" not in state["data"]


def test_the_world_geometry_must_be_the_replacements(case):
    m = case["model"]
    m.configure_hand_response_calibration(case["requested_file"])
    m.replace_saved_hand_calibration = str(case["record_file"])
    m._restore_hand_response_calibration(case["ckpt"])
    other = copy.deepcopy(case["geometry"])
    other["joints"][0]["limits"][1] = 2.5
    with pytest.raises(ValueError, match="world's hand geometry"):
        m.configure_hand_geometry(other, torch.zeros(2, 1))


def _break(case, what):
    """The case with one binding broken: (requested payload, requested file sha, record, launch rule)."""
    req, sha, model_sha, rule = copy.deepcopy(case["requested_payload"]), case["requested_sha"], case["model_sha"], "on"
    rec = None
    if what == "payload":
        rec = make_record(dict(req, sha256="e" * 64), sha, model_sha)
    elif what == "file bytes":
        rec = make_record(req, "e" * 64, model_sha)
    elif what == "record model":
        rec = make_record(req, sha, "e" * 64)
    elif what == "calibration model":
        req = payload(case["geometry"], case["receipt"], coefficients=[[0.01, .7, .2]],
                      command_meaning=FINGER_SERVO_CONTRACT, source_model_sha256="e" * 64)
        sha = write_json(case["requested_file"], req)
        rec = make_record(req, sha, model_sha)
    elif what == "code":
        rec = make_record(req, sha, model_sha, code="e" * 64)
    elif what == "finger rule":
        rule = "off"
    elif what == "record meaning":
        rec = make_record(req, sha, model_sha, command_meaning=HR.POSE_SERVO_MEANING)
    elif what == "geometry":
        rec = make_record(req, sha, model_sha, geometry_sha256="e" * 64)
    elif what == "route":
        rec = make_record(req, sha, model_sha, tested_route="a private surrogate")
    elif what == "bodies":
        rec = make_record(req, sha, model_sha, held_out_bodies=[0, 1, 2, 3])
    elif what in ("weights_unchanged", "exact_replay", "nonfinger_commands_unchanged", "sensory_goals_identical",
                  "hand_stage_completed"):
        rec = make_record(req, sha, model_sha, **{what: False})
    elif what == "learning":
        rec = make_record(req, sha, model_sha, learning=True)
    elif what == "validity problems":
        rec = make_record(req, sha, model_sha, validity_problems=["a repeated hold differed"])
    elif what == "a body not improved":
        values = list(DIFFERENCES)
        values[4:8] = [0.05, -0.02, 0.04, -0.01]
        rec = make_record(req, sha, model_sha, decisions=decisions(values))
    elif what == "incomplete":
        rec = make_record(req, sha, model_sha, decisions=decisions()[:15])
    elif what == "stated line":
        line = admission_line(decisions())
        rec = make_record(req, sha, model_sha, line=dict(line, mean=line["mean"] - 0.1))
    elif what == "verdict":
        rec = make_record(req, sha, model_sha, verdict="fail", passed=False)
    elif what == "content hash":
        rec = make_record(req, sha, model_sha)
        rec["checkpoint"] = "another"
    elif what == "version":
        rec = make_record(req, sha, model_sha, version="hand_calibration_admission_v0")
    elif what == "not a JSON object":
        rec = [make_record(req, sha, model_sha)]           # valid JSON: a list holding a passing record
    elif what == "P1 failing alone":
        # every body's mean of d below 0 (P2 holds) and the mean of d below 0, but the 16 values spread so far around
        # it that the exact sign-flip count exceeds floor(0.05 x 2^16) = 3,276 of 65,536 patterns
        values = [-0.90, 0.85, -0.80, 0.80, -0.70, 0.75, -0.60, 0.50,
                  0.95, -0.90, 0.60, -0.70, -0.40, 0.45, -0.50, 0.40]
        line = admission_line(decisions(values))
        assert line["complete"] and line["P2"] and line["mean"] < 0 and not line["P1"]
        assert line["signflip_count"] > line["signflip_line"] == 3276
        rec = make_record(req, sha, model_sha, decisions=decisions(values))
    elif what == "calibration meaning":
        # the requested calibration declares no command meaning (it reads the pose servo's) while the record states
        # the joint servo's, which the launch's on rule executes
        req = payload(case["geometry"], case["receipt"], coefficients=[[0.01, .7, .2]], source_model_sha256=model_sha)
        assert HR.command_meaning(req) == HR.POSE_SERVO_MEANING
        sha = write_json(case["requested_file"], req)
        rec = make_record(req, sha, model_sha)
    elif what == "passed false":
        rec = make_record(req, sha, model_sha, passed=False)          # verdict 'pass' kept
    if rec is not None:
        write_json(case["record_file"], rec)
    return rule


BROKEN = ["payload", "file bytes", "record model", "calibration model", "code", "finger rule", "record meaning",
          "geometry", "route", "bodies", "weights_unchanged", "exact_replay", "nonfinger_commands_unchanged",
          "sensory_goals_identical", "hand_stage_completed", "learning", "validity problems", "a body not improved",
          "incomplete", "stated line", "verdict", "content hash", "version", "not a JSON object", "P1 failing alone",
          "calibration meaning", "passed false"]
# the refusal text these cases must give: the binding each one breaks, not an earlier binding
REFUSAL_TEXT = {
    "not a JSON object": "the admission record is not a JSON object",
    "P1 failing alone": "(complete True, P1 False, P2 True,",
    "calibration meaning": "declare the command meanings %r and %r; this launch executes %r" % (
        FINGER_SERVO_CONTRACT, HR.POSE_SERVO_MEANING, FINGER_SERVO_CONTRACT),
    "passed false": "does not read verdict pass",
}


@pytest.mark.parametrize("what", BROKEN)
def test_each_failed_binding_refuses_the_replacement(case, what):
    m = case["model"]
    rule = _break(case, what)
    m.finger_joint_servo = rule
    m.configure_hand_response_calibration(case["requested_file"])
    m.replace_saved_hand_calibration = str(case["record_file"])
    with pytest.raises(ValueError, match="replace-saved-hand-calibration") as refused:
        m._restore_hand_response_calibration(case["ckpt"])
    if what in REFUSAL_TEXT:
        assert REFUSAL_TEXT[what] in str(refused.value)
    assert getattr(m, "hand_calibration_replacement", None) is None
    assert "hand_calibration_replacements" not in m.receipt


def test_the_replacement_needs_the_wrist_relative_hand_path(case):
    m = case["model"]
    m.configure_hand_response_calibration(case["requested_file"])
    m.replace_saved_hand_calibration = str(case["record_file"])
    m.hand_target_mode = "world"
    try:
        with pytest.raises(ValueError, match="wrist_relative_v1"):
            m._restore_hand_response_calibration(case["ckpt"])
    finally:
        m.hand_target_mode = "wrist_relative_v1"


def test_with_nothing_to_replace_todays_rule_holds(case):
    """The setting on a checkpoint that already saved the requested calibration (a later resume of the replacing run),
    or that saved none: no replacement, no record, today's acceptance or refusal."""
    m = case["model"]
    m.replace_saved_hand_calibration = str(case["record_file"])
    m._hand_response_calibration = copy.deepcopy(case["saved"])
    m._restore_hand_response_calibration(case["ckpt"])
    assert m._hand_response_calibration == case["saved"] and getattr(m, "hand_calibration_replacement", None) is None
    bare = case["tmp"] / "bare"
    bare.mkdir()
    (bare / "model.safetensors").write_bytes((case["ckpt"] / "model.safetensors").read_bytes())
    m.configure_hand_response_calibration(case["requested_file"])     # bound to these weights
    m._restore_hand_response_calibration(bare)
    assert getattr(m, "hand_calibration_replacement", None) is None
    (bare / "model.safetensors").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="different checkpoint weights"):
        m._restore_hand_response_calibration(bare)


def test_a_later_resume_carries_the_record_without_the_setting(case):
    m = case["model"]
    m.configure_hand_response_calibration(case["requested_file"])
    m.replace_saved_hand_calibration = str(case["record_file"])
    m._restore_hand_response_calibration(case["ckpt"])
    m.configure_hand_geometry(case["geometry"], torch.zeros(2, 1))
    later = case["tmp"] / "ckpt_step_20"
    later.mkdir()
    (later / "model.safetensors").write_bytes(b"weights trained further")
    write_json(later / "model_receipt.json", dict(m.receipt))
    first = copy.deepcopy(m.receipt["hand_calibration_replacements"])
    del m.replace_saved_hand_calibration, m.hand_calibration_replacement
    m.receipt.pop("hand_calibration_replacements")
    m.configure_hand_response_calibration(case["requested_file"])
    m._restore_hand_response_calibration(later)
    assert m.receipt["hand_calibration_replacements"] == first
    assert m.receipt["hand_response"]["sha256"] == case["requested_payload"]["sha256"]


# ------------------------------------------------------------------------------------------------ the shared pieces
def test_trainer_code_hash_is_the_training_preflights_code_hash():
    """One function (lem/trainer_code.py) for both readers; its rules are tested in test_trainer_code_hash.py."""
    from lem import trainer_code as TC
    from mimo_bridge.tools import training_preflight as TP
    assert HR.trainer_code_hash is TC.trainer_code_hash
    src = Path(HR.__file__).resolve().parents[2]
    code = TP.code_hashes(src)
    expected = hashlib.sha256("\n".join("%s %s" % kv for kv in sorted(code.items())).encode()).hexdigest()
    assert HR.trainer_code_hash(src) == expected == HR.trainer_code_hash()
    assert TP.configuration(["python", "-m", "mimo_bridge.train_mimo"], src)["code_hash"] == expected


def main_replacement_record():
    """train_mimo.main's own statements that write a replacement's stats record (the `if` on
    getattr(model, "hand_calibration_replacement", None) in main, after the learner's configure_hand_geometry), read
    from the trainer's source file and compiled as record(sf, model, step): the lines main runs, without running the
    trainer. Returns (record, first line number in train_mimo.py)."""
    import ast
    import textwrap
    import time
    from mimo_bridge import train_mimo as T
    source = Path(T.__file__).read_text(encoding="utf-8")
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    found = []
    for node in ast.walk(main):
        for field in ("body", "orelse"):
            block = getattr(node, field, None)
            for i, stmt in enumerate(block if isinstance(block, list) else []):
                if (isinstance(stmt, ast.If) and ast.unparse(stmt.test)
                        == "getattr(model, 'hand_calibration_replacement', None) is not None"):
                    found.append((block, i, stmt))
    assert len(found) == 1, "train_mimo.main writes the replacement's stats record in exactly one place"
    block, i, stmt = found[0]
    assert any(ast.unparse(s).startswith("model.configure_hand_geometry(") for s in block[:i]), \
        "the record follows the learner's configure_hand_geometry (the world geometry check) in the same block"
    text = textwrap.dedent("\n".join(source.splitlines()[stmt.lineno - 1:stmt.end_lineno]))
    namespace = dict(json=json, time=time)
    exec(compile("def record(sf, model, step):\n" + textwrap.indent(text, "    "), str(T.__file__), "exec"), namespace)
    return namespace["record"], stmt.lineno


def test_main_writes_the_replacement_into_the_stats_stream_after_the_hand_set_up(case):
    """The stats record train_mimo.main writes after a replacement: nothing at a launch that replaced nothing; after a
    replacement and the hand path's set-up, one line {"type": "hand_calibration_replacement", "timestamp", "step",
    "replacement": the learner's hand_calibration_replacement_report()} holding the world's geometry hash and no copy
    of the calibration. Whether main reaches these lines on a real resume is an item of the activation's end-to-end
    dry run (HAND_CALIBRATION_ADMISSION_DESIGN.md, section 10)."""
    import io
    record, _line = main_replacement_record()
    m = case["model"]
    stream = io.StringIO()
    record(stream, m, 10)
    assert stream.getvalue() == ""
    m.configure_hand_response_calibration(case["requested_file"])
    m.replace_saved_hand_calibration = str(case["record_file"])
    m._restore_hand_response_calibration(case["ckpt"])
    m._hand_response_calibration = copy.deepcopy(case["saved"])          # the episode companion's restore
    m.configure_hand_geometry(case["geometry"], torch.zeros(2, 1))
    record(stream, m, 10)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert set(written) == {"type", "timestamp", "step", "replacement"}
    assert written["type"] == "hand_calibration_replacement" and written["step"] == 10
    assert isinstance(written["timestamp"], float)
    assert written["replacement"] == json.loads(json.dumps(m.hand_calibration_replacement_report(), default=str))
    assert written["replacement"]["world_geometry_sha256"] == content_hash(case["geometry"])
    assert written["replacement"]["replaced"]["payload_sha256"] == case["saved"]["sha256"]
    assert written["replacement"]["replaced_by"]["payload_sha256"] == case["requested_payload"]["sha256"]
    assert written["replacement"]["admission"]["sha256"] == case["record"]["sha256"]
    assert "calibration" not in written["replacement"]


def test_the_exact_sign_flip_count_and_the_line():
    rng = np.random.default_rng(7)
    for n in (5, 8, 17):
        v = rng.normal(-0.2, 0.3, n)
        brute = sum(1 for s in itertools.product((1.0, -1.0), repeat=n)
                    if abs(float(np.mean(np.asarray(s) * v))) >= abs(float(v.mean())) - 1e-12)
        assert HR.signflip_count(v) == (brute, 2 ** n)
    line = admission_line(decisions())
    assert line["complete"] and line["P1"] and line["P2"] and line["passes"]
    assert line["signflip_line"] == 3276 and line["signflip_count"] == 2
    assert set(line["body_means"]) == {"4", "5", "6", "7"}
    values = list(DIFFERENCES)
    values[0:4] = [0.3, 0.2, 0.25, 0.1]
    worse = admission_line(decisions(values))
    assert not worse["P2"] and not worse["passes"]
    assert not admission_line(decisions()[:-1])["passes"]
    nan = decisions()
    nan[3]["difference"] = float("nan")
    assert not admission_line(nan)["passes"]
