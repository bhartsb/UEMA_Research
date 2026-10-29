# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/finger_command_interface/activation_dry_run.py on synthetic outputs: no trainer and no world.

  THE COMMANDS - each run differs from the activation candidate only in its declared flags (--out, --checkpoint-storage,
  --caregiver-record-dir, --latent-body-output, --max-steps, --resume; run B adds --eval-only and
  --episode-continuation fresh); a candidate carrying the attention cap or lacking an activation setting is refused.
  THE START COPY - a byte copy is accepted with every file's sha256; a copy that differs is refused.
  THE CHECK - a synthetic dry run whose outputs are what the design says reads pass; each contradicted item reads fail
  (a changed parameter, a changed buffer outside the running statistics, a changed optimizer tensor, a replacement
  record without the world geometry check, a companion holding the admitted fit, a planning decision naming another
  calibration, worlds without the finger rule, a launch that refused the replacement, the code changed between verify
  and launch, the supervisor's record changed); each missing output reads invalid.
  THE WATCHER - the hourly watcher's own planner_integrity accepts run B's recorded decisions with the refit's hash and
  names a stop under the inherited contract (seg37's hash); a decision the watcher would stop under the refit reads fail.
  THE WATCH - the trainer watch ends a process tree whose commit charge enters the guard's reserve, and records the
  supervisor's record while the trainer lives.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
TOOL_DIR = REPO / "docs/gain_of_function_v4/finger_command_interface"
SRC = REPO / "Source" / "Python"

REFIT, REFIT_FILE = "f" * 64, "e" * 64
ADMITTED = "a" * 64
MODEL, GEOMETRY, CODE = "d" * 64, "c" * 64, "b" * 64
SERVO = "finger_joint_servo_v1"
PARAMETERS = {"forward_model.w": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7.0,
              "forward_model.b": torch.tensor([0.5, -1.25, 2.0, 3.0])}
BUFFERS = {"persistent_memory.event_centre": torch.zeros(5),
           "persistent_memory.event_centre_weight": torch.tensor(0.5, dtype=torch.float64),
           "heads_all_slot_path.centre_mean": torch.ones(2, 3),
           "gaze_head.az_scale": torch.tensor(1.0)}


@pytest.fixture
def tool(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(TOOL_DIR))
    t = importlib.import_module("activation_dry_run")
    prod = tmp_path / "prod"
    prod.mkdir()
    cal, rec = prod / "calibration.json", prod / "record.json"
    cal.write_text(json.dumps(dict(sha256=REFIT)), encoding="utf-8")
    rec.write_text(json.dumps(dict(sha256="9" * 64, verdict="pass")), encoding="utf-8")
    monkeypatch.setattr(t, "CALIBRATION", cal)
    monkeypatch.setattr(t, "RECORD", rec)
    monkeypatch.setattr(t, "CHECKPOINT", prod / "ckpt_step_481800")
    monkeypatch.setattr(t, "EXPECTED", dict(t.EXPECTED, refit_payload=REFIT, refit_file=REFIT_FILE,
                                            admitted_payload=ADMITTED, model=MODEL, geometry=GEOMETRY, code_hash=CODE,
                                            record_file=hashlib.sha256(rec.read_bytes()).hexdigest()))
    contract = prod / "continuation_receipt.json"
    contract.write_text(json.dumps(dict(planner_selection_contract="best_evaluated_per_body_v1", planner_bodies=8,
                                        hand_selection_contract="hand_joint_response_v1",
                                        hand_calibration_sha256=ADMITTED)), encoding="utf-8")
    monkeypatch.setattr(t, "INHERITED_CONTRACT", contract)
    monkeypatch.setattr(t, "WATCHER", REPO / "docs/h84_launch_2026-09-18/hourly_watch.py")
    return t


def candidate(t, extra=()):
    return ([sys.executable, "-u", "-m", "mimo_bridge.train_mimo", "--n-envs", "8", "--finger-joint-servo", "on",
             "--resume-env-settings", "launch", "--hand-goals", "memory", "--hand-target-mode", "wrist_relative_v1",
             "--hand-response-calibration", str(t.CALIBRATION), "--replace-saved-hand-calibration", str(t.RECORD),
             "--resume", str(t.CHECKPOINT), "--out", "C:\\Saved\\mimo_bridge\\experiments\\run",
             "--checkpoint-storage", "C:\\Saved\\mimo_bridge\\experiments\\run\\checkpoints",
             "--caregiver-record-dir", "D:\\UEMA_Research\\external\\D\\UEMA_recordings\\H84_seg41_caregiver",
             "--latent-body-output", "D:\\UEMA_latent_body\\seg41_v2", "--max-steps", "484675", "--fill-ring-first",
             "--caregiver-present-seconds", "60", "--babble-noise", "0.05"] + list(extra))


def args_of(t, cmd):
    def typed(v):
        if v is True:
            return True
        for kind in (int, float):
            try:
                return kind(v)
            except ValueError:
                pass
        return v
    out = {f[2:].replace("-", "_"): typed(v) for f, v in t.flag_values(cmd).items()}
    out.setdefault("eval_only", False)
    return out


def inventory():
    comps = {}
    for name, kind in [(n, "parameter") for n in PARAMETERS] + [(n, "buffer") for n in BUFFERS]:
        comp, key = name.split(".", 1)
        comps.setdefault(comp, {"tensors": {}})["tensors"][key] = {"kind": kind}
    return {"components": comps}


def optimizer_state():
    g = torch.Generator().manual_seed(3)
    return {"optimizer": {"state": {0: {"step": torch.tensor(7.0), "exp_avg": torch.randn(3, 4, generator=g),
                                        "exp_avg_sq": torch.rand(3, 4, generator=g)},
                                    1: {"step": torch.tensor(7.0), "exp_avg": torch.randn(4, generator=g),
                                        "exp_avg_sq": torch.rand(4, generator=g)}},
                          "param_groups": [{"lr": 3e-4, "betas": (0.9, 0.999), "params": [0, 1]}]}}


def checkpoint(folder, storage, step, calibration, finger_rule, tensors=None, opt=None, receipt_extra=None):
    """A checkpoint folder as CheckpointManager and continuation_state write one, with a synthetic episode companion."""
    from safetensors.torch import save_file
    from mimo_bridge import continuation_state as cs
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in (tensors or dict(PARAMETERS, **BUFFERS)).items()},
              str(folder / "model.safetensors"))
    torch.save(opt or optimizer_state(), folder / "optimizer.pt")
    torch.save({"vis_adv_ema": 0.0}, folder / "runtime_state.pt")
    (folder / "meta.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    receipt = dict(checkpoint_inventory=inventory(), episode_continuation=cs.VERSION, hand_response=calibration)
    receipt.update(receipt_extra or {})
    (folder / "model_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    kw = dict(hand_target_mode="wrist_relative_v1", **({} if finger_rule is None else {"finger_joint_servo": finger_rule}))
    state = dict(version=cs.VERSION, worlds=[dict(env_kwargs=dict(kw), payload=dict(sim_time=1.0)) for _ in range(8)],
                 learner=dict(data={"_hand_response_calibration": dict(calibration)}), trainer=dict(step=step),
                 config={})
    cs.write_episode(folder, state, storage_root=str(storage))
    return folder


def refit_cal():
    return dict(sha256=REFIT, calibration_file_sha256=REFIT_FILE, command_meaning=SERVO)


def replacement_record(t, step, **change):
    r = dict(setting="--replace-saved-hand-calibration", model_step=481800, model_sha256=MODEL,
             replaced=dict(payload_sha256=ADMITTED, command_meaning="finger_pose_servo"),
             replaced_by=dict(payload_sha256=REFIT, calibration_file_sha256=REFIT_FILE, command_meaning=SERVO,
                              geometry_sha256=GEOMETRY),
             admission=dict(file_sha256=t.EXPECTED["record_file"], sha256="9" * 64, code_hash=CODE, verdict="pass"),
             launch=dict(finger_joint_servo="on", hand_goals="memory", hand_target_mode="wrist_relative_v1"),
             world_geometry_sha256=GEOMETRY)
    for key, value in change.items():
        r[key] = value
    return {"type": "hand_calibration_replacement", "timestamp": 1.0, "step": step, "replacement": r}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def facts(**change):
    f = dict(git_head="1" * 40, source_tree="6" * 40, code_hash=CODE, mimo_head="2" * 40, mimo_clean=True,
             supervisor_cmd="pause",
             supervisor_control_sha256="3" * 64, continuation_active_run="run", continuation_command_sha256="4" * 64,
             continuation_trainer_pids=[], trainers=[], experiments=["a", "b"], archive_keep_sha256="5" * 64,
             latent_production_latest=[1, 2.0], caregiver_states=["x"])
    f.update(change)
    return f


def plan_line(step, calibration=REFIT, valid=None, stopped="completed"):
    """A causal-chain latch line holding one planning decision, in the shape the recorder writes (the body search's
    per-body comparison and the hand stage's 20 updates), so the watcher's own planner_integrity can judge it."""
    first, best = [2.0] * 8, [1.5] * 8
    hand = dict(version="hand_joint_response_v1", calibration_sha256=calibration, goal="wrist_relative_finger_shape",
                horizon_ms=250, steps=20, raw_joint_bound_deg=11.0, first=first, selected=best,
                selected_iteration=[1] * 8, valid=[True] * 8 if valid is None else valid,
                trace=[first] + [best] * 20, stopped_by=stopped)
    descent = dict(depth=1, selection="best_evaluated_per_body_v1", stopped_by="step_budget",
                   per_body={name: dict(start=[1.0] * 8, selected=[0.9] * 8) for name in ("objective", "goal", "effort")})
    return {"type": "latch", "step": step, "bodies": 8, "reached_body": True,
            "plan": {"bodies": 8, "optimizer": [descent], "hand_optimizer": hand, "eye_optimizer": []}}


@pytest.fixture
def dry(tool, tmp_path):
    """A synthetic dry run whose outputs are what the design says: the check reads pass."""
    t = tool
    root = tmp_path / "dry"
    admitted = dict(sha256=ADMITTED, calibration_file_sha256="9" * 64)
    checkpoint(t.CHECKPOINT, tmp_path / "prod_store", 481800, admitted, None)
    t.copy_start(t.CHECKPOINT, t.run_paths(root, "A")["start"])
    cmds = t.derive_commands(candidate(t), root)
    rep_receipt = dict(hand_calibration_replacements=[dict(replaced=dict(payload_sha256=ADMITTED),
                                                           replaced_by=dict(payload_sha256=REFIT),
                                                           world_geometry_sha256=GEOMETRY)])
    steps = []
    for key in ("A", "B"):
        p = t.run_paths(root, key)
        n = t.RUNS[key]["steps"]
        (p["out"]).mkdir(parents=True, exist_ok=True)
        (p["out"] / "args.json").write_text(json.dumps(args_of(t, cmds[key])), encoding="utf-8")
        (p["out"] / "summary.json").write_text(json.dumps(dict(ok=True, exception=None, steps=n)), encoding="utf-8")
        tensors = dict(PARAMETERS, **BUFFERS)
        if key == "B":     # the running statistics move in every forward pass
            tensors["persistent_memory.event_centre"] = tensors["persistent_memory.event_centre"] + 0.25
            tensors["heads_all_slot_path.centre_mean"] = tensors["heads_all_slot_path.centre_mean"] * 0.5
        checkpoint(p["end"], p["caregiver"] / "checkpoint_state" / p["out"].name, n, refit_cal(), "on", tensors,
                   receipt_extra=rep_receipt)
        if key == "A":
            write_rows(p["out"] / "replay_training_stats.jsonl",
                       [{"type": "launch", "step": 481800}, replacement_record(t, 481800)])
        else:
            mimo = {"type": "mimo", "step": 26, "replay_loss": None, "motor_contract": ["arm_v3+" + SERVO] * 8,
                    "intention": {"step": 20, "hand_descent": dict(plan_line(20)["plan"]["hand_optimizer"])},
                    "intention_window": {"decisions": 1, "selected_plans": 1}}
            write_rows(p["out"] / "replay_training_stats.jsonl",
                       [{"type": "launch", "step": 0}, replacement_record(t, 0),
                        {"type": "training", "avg_grad_norm": 0.0}, mimo])
            write_rows(p["out"] / "causal_chain.jsonl", [plan_line(20)])
        verify = dict(ok=True, output=dict(ok=True, checks=dict(admission_check=dict(code_hash=CODE, model_sha256=MODEL),
                                                              script_unchanged=True, candidate_kept=True)),
                      started="Oct 08, 2026 11:00:00 PM PDT")
        steps.append(dict(run=key, verify=verify, facts_verify=facts(), facts_launch=facts(), facts_end=facts(),
                          trainer=dict(command=cmds[key], stopped=None, exit_code=0, seconds=60.0,
                                       supervisor_trainer_pids_seen=[], supervisor_active_run_changed=False,
                                       supervisor_command_changed=False, commit_peak_gb=60.0, commit_base_gb=37.0)))
    (root / "driver_record.json").write_text(json.dumps(dict(steps=steps)), encoding="utf-8")
    return root


def judged(t, root):
    r = t.judge(root, SRC)
    return r, {i["id"]: i for i in r["items"]}


# ---------------------------------------------------------------------------------------------- the commands
def test_each_run_differs_from_the_candidate_only_in_its_declared_flags(tool, tmp_path):
    t = tool
    cmds = t.derive_commands(candidate(t), tmp_path / "dry")
    a = t.differences(candidate(t), cmds["A"])
    b = t.differences(candidate(t), cmds["B"])
    assert set(a) == set(t.RUN_CHANGES)
    assert set(b) == set(t.RUN_CHANGES) | {"--eval-only", "--episode-continuation"}
    assert a["--max-steps"][1] == "481800" and b["--max-steps"][1] == "26"
    assert b["--episode-continuation"][1] == "fresh" and b["--eval-only"][1] is True
    for key in ("A", "B"):
        v = t.flag_values(cmds[key])
        assert Path(v["--resume"]) == t.run_paths(tmp_path / "dry", key)["start"]
        assert "experiments" not in v["--resume"] and "experiments" not in v["--out"]
        assert all(str(tmp_path / "dry") in str(Path(v[f])) for f in ("--out", "--checkpoint-storage",
                                                                       "--caregiver-record-dir", "--latent-body-output"))


@pytest.mark.parametrize("change", [["--attention-cap", "on"], ["--eval-only"]])
def test_a_candidate_with_the_cap_or_eval_only_is_refused(tool, tmp_path, change):
    with pytest.raises(ValueError, match="carries"):
        tool.derive_commands(candidate(tool, change), tmp_path)


def test_a_candidate_without_the_finger_rule_is_refused(tool, tmp_path):
    cmd = candidate(tool)
    cmd[cmd.index("--finger-joint-servo") + 1] = "off"
    with pytest.raises(ValueError, match="finger-joint-servo"):
        tool.derive_commands(cmd, tmp_path)


# ---------------------------------------------------------------------------------------------- the start copy
def test_the_start_copy_is_checked_byte_for_byte(tool, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("model.safetensors", "episode_state.json", "meta.json"):
        (src / name).write_bytes(name.encode() * 3)
    dst = tmp_path / "copy" / "ckpt_step_481800"
    receipt = tool.copy_start(src, dst)
    assert receipt["identical"] and set(receipt["files"]) == {"model.safetensors", "episode_state.json", "meta.json"}
    (dst / "meta.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        tool.copy_start(src, dst)


# ---------------------------------------------------------------------------------------------- the check
def test_outputs_as_designed_read_pass(tool, dry):
    r, items = judged(tool, dry)
    assert r["verdict"] == "pass", r["verdict_reason"]
    assert {"P1", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7",
            "W1"} <= set(items)
    w = items["W1"]["evidence"]
    assert w["under_refit"]["stop_reason"] is None and w["under_refit"]["hand_searches"] == 1
    assert "lacks its qualified contract" in w["under_inherited"]["stop_reason"] and w["inherited_hash"] == ADMITTED
    moved = {d["name"] for d in items["B7"]["evidence"]["model"]["differing"]}
    assert moved == {"persistent_memory.event_centre", "heads_all_slot_path.centre_mean"}
    assert items["A6"]["evidence"]["model"]["file_sha256_equal"] is True
    names = [x["name"] for x in r["readings"]]
    assert any("parameters changed (A)" in n for n in names) and any("planning decisions" in n for n in names)


def rewrite_tensors(t, folder, **changes):
    from safetensors.torch import save_file
    tensors = dict(PARAMETERS, **BUFFERS)
    tensors.update(changes)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(folder / "model.safetensors"))


def test_a_changed_parameter_reads_fail(tool, dry):
    rewrite_tensors(tool, tool.run_paths(dry, "A")["end"], **{"forward_model.b": torch.tensor([0.5, -1.25, 2.0, 3.5])})
    r, items = judged(tool, dry)
    assert r["verdict"] == "fail" and items["A6"]["state"] == "failed"
    assert "1 parameters changed" in items["A6"]["evidence"]["why"]


def test_a_running_statistic_may_move_in_b_but_not_in_a(tool, dry):
    rewrite_tensors(tool, tool.run_paths(dry, "A")["end"],
                    **{"persistent_memory.event_centre": torch.full((5,), 0.25)})
    r, items = judged(tool, dry)
    assert items["A6"]["state"] == "failed" and "buffers outside" in items["A6"]["evidence"]["why"]
    assert items["B7"]["state"] == "held"


def test_another_buffer_moving_in_b_reads_fail(tool, dry):
    rewrite_tensors(tool, tool.run_paths(dry, "B")["end"], **{"gaze_head.az_scale": torch.tensor(2.0),
                                                              "persistent_memory.event_centre": torch.full((5,), 0.25)})
    r, items = judged(tool, dry)
    assert r["verdict"] == "fail" and "gaze_head.az_scale" in items["B7"]["evidence"]["why"]


def test_a_changed_optimizer_tensor_reads_fail(tool, dry):
    opt = optimizer_state()
    opt["optimizer"]["state"][1]["exp_avg_sq"] = opt["optimizer"]["state"][1]["exp_avg_sq"] + 1e-6
    torch.save(opt, tool.run_paths(dry, "B")["end"] / "optimizer.pt")
    r, items = judged(tool, dry)
    assert items["B7"]["state"] == "failed" and "optimizer state differs" in items["B7"]["evidence"]["why"]


def test_a_replacement_record_without_the_geometry_check_reads_fail(tool, dry):
    stats = tool.run_paths(dry, "A")["out"] / "replay_training_stats.jsonl"
    write_rows(stats, [{"type": "launch", "step": 481800}, replacement_record(tool, 481800, world_geometry_sha256=None)])
    r, items = judged(tool, dry)
    assert items["A2"]["state"] == "failed" and "world_geometry_sha256" in items["A2"]["evidence"]["why"]


def test_no_replacement_record_reads_fail(tool, dry):
    write_rows(tool.run_paths(dry, "A")["out"] / "replay_training_stats.jsonl", [{"type": "launch", "step": 481800}])
    r, items = judged(tool, dry)
    assert r["verdict"] == "fail" and "0 hand_calibration_replacement records" in items["A2"]["evidence"]["why"]


def test_a_companion_holding_the_admitted_fit_reads_fail(tool, dry, tmp_path):
    p = tool.run_paths(dry, "A")
    import shutil
    shutil.rmtree(p["end"])
    checkpoint(p["end"], tmp_path / "other_store", 481800, dict(sha256=ADMITTED, calibration_file_sha256="9" * 64),
               "on", receipt_extra=dict(hand_calibration_replacements=[]))
    r, items = judged(tool, dry)
    assert items["A5"]["state"] == "failed" and "not the refit" in items["A5"]["evidence"]["why"]
    assert items["A4"]["state"] == "failed"


def test_worlds_without_the_finger_rule_in_the_companion_read_fail(tool, dry, tmp_path):
    p = tool.run_paths(dry, "A")
    import shutil
    shutil.rmtree(p["end"])
    checkpoint(p["end"], tmp_path / "other_store", 481800, refit_cal(), None,
               receipt_extra=dict(hand_calibration_replacements=[dict(
                   replaced=dict(payload_sha256=ADMITTED), replaced_by=dict(payload_sha256=REFIT),
                   world_geometry_sha256=GEOMETRY)]))
    r, items = judged(tool, dry)
    assert items["A5"]["state"] == "failed" and "finger rules" in items["A5"]["evidence"]["why"]


def without_latest_plan(t, root):
    stats = t.run_paths(root, "B")["out"] / "replay_training_stats.jsonl"
    old = [json.loads(x) for x in stats.read_text(encoding="utf-8").splitlines()]
    for r in old:
        if r.get("type") == "mimo":
            r["intention"] = None
    write_rows(stats, old)


def test_no_planning_decision_complete_on_all_8_bodies_reads_invalid(tool, dry):
    write_rows(tool.run_paths(dry, "B")["out"] / "causal_chain.jsonl", [plan_line(20, valid=[True] * 7 + [False])])
    without_latest_plan(tool, dry)
    r, items = judged(tool, dry)
    assert r["verdict"] == "invalid" and items["B4"]["state"] == "not reached" and items["W1"]["state"] == "not reached"


def test_a_decision_only_in_the_reports_holds_b4_but_leaves_the_watcher_unread(tool, dry):
    (tool.run_paths(dry, "B")["out"] / "causal_chain.jsonl").unlink()
    r, items = judged(tool, dry)
    assert items["B4"]["state"] == "held" and items["B4"]["evidence"]["complete_latest_plans_steps"] == [20]
    assert items["W1"]["state"] == "not reached" and r["verdict"] == "invalid"


def test_a_report_naming_the_admitted_fit_reads_fail(tool, dry):
    stats = tool.run_paths(dry, "B")["out"] / "replay_training_stats.jsonl"
    old = [json.loads(x) for x in stats.read_text(encoding="utf-8").splitlines()]
    for r in old:
        if r.get("type") == "mimo":
            r["intention"]["hand_descent"]["calibration_sha256"] = ADMITTED
    write_rows(stats, old)
    r, items = judged(tool, dry)
    assert items["B4"]["state"] == "failed"


def test_a_planning_decision_naming_another_calibration_reads_fail(tool, dry):
    write_rows(tool.run_paths(dry, "B")["out"] / "causal_chain.jsonl", [plan_line(20), plan_line(30, ADMITTED)])
    r, items = judged(tool, dry)
    assert r["verdict"] == "fail" and items["B4"]["state"] == "failed"


def test_a_decision_the_watcher_would_stop_under_the_refit_reads_fail(tool, dry):
    bad = plan_line(30)
    bad["plan"]["optimizer"][0]["per_body"]["objective"]["selected"] = [1.1] * 8     # worse than its start, 1.0
    write_rows(tool.run_paths(dry, "B")["out"] / "causal_chain.jsonl", [plan_line(20), bad])
    r, items = judged(tool, dry)
    assert items["W1"]["state"] == "failed" and "worse total objective" in items["W1"]["evidence"]["why"]


def test_worlds_not_reporting_the_finger_rule_read_fail(tool, dry):
    stats = tool.run_paths(dry, "B")["out"] / "replay_training_stats.jsonl"
    old = [json.loads(x) for x in stats.read_text(encoding="utf-8").splitlines()]
    for r in old:
        if r.get("type") == "mimo":
            r["motor_contract"] = ["arm_v3"] * 8
    write_rows(stats, old)
    r, items = judged(tool, dry)
    assert items["B5"]["state"] == "failed"


def test_a_gradient_step_in_b_reads_fail(tool, dry):
    stats = tool.run_paths(dry, "B")["out"] / "replay_training_stats.jsonl"
    old = [json.loads(x) for x in stats.read_text(encoding="utf-8").splitlines()]
    for r in old:
        if r.get("type") == "training":
            r["avg_grad_norm"] = 12.5
    write_rows(stats, old)
    r, items = judged(tool, dry)
    assert items["B3"]["state"] == "failed"


@pytest.mark.parametrize("text,state,verdict", [
    ("ValueError: --replace-saved-hand-calibration: the admission record was measured on trainer code x", "failed", "fail"),
    ("RuntimeError: worker 3 did not answer for 180 s (hung)", "not reached", "invalid")])
def test_a_launch_that_ended_in_an_error(tool, dry, text, state, verdict):
    p = tool.run_paths(dry, "A")
    (p["out"] / "summary.json").write_text(json.dumps(dict(ok=False, exception=text, steps=481800)), encoding="utf-8")
    r, items = judged(tool, dry)
    assert items["A1"]["state"] == state and r["verdict"] == verdict


def test_no_summary_reads_invalid(tool, dry):
    (tool.run_paths(dry, "B")["out"] / "summary.json").unlink()
    r, items = judged(tool, dry)
    assert r["verdict"] == "invalid" and items["B1"]["state"] == "not reached"


def driver(root):
    return json.loads((root / "driver_record.json").read_text(encoding="utf-8"))


def put_driver(root, d):
    (root / "driver_record.json").write_text(json.dumps(d), encoding="utf-8")


@pytest.mark.parametrize("name,value", [("source_tree", "7" * 40), ("code_hash", "0" * 64), ("mimo_head", "7" * 40)])
def test_the_code_changed_between_verify_and_launch_reads_fail(tool, dry, name, value):
    d = driver(dry)
    d["steps"][0]["facts_launch"][name] = value
    put_driver(dry, d)
    r, items = judged(tool, dry)
    assert items["A0"]["state"] == "failed" and "code changed" in items["A0"]["evidence"]["why"]


def test_a_commit_outside_source_python_between_verify_and_launch_is_allowed(tool, dry):
    d = driver(dry)
    d["steps"][0]["facts_launch"]["git_head"] = "7" * 40      # the registry's own commit, say
    d["steps"][0]["facts_end"]["git_head"] = "7" * 40
    put_driver(dry, d)
    r, items = judged(tool, dry)
    assert items["A0"]["state"] == "held" and r["verdict"] == "pass"


def test_the_supervisor_recording_the_dry_run_reads_fail(tool, dry):
    d = driver(dry)
    d["steps"][1]["trainer"]["supervisor_trainer_pids_seen"] = [4242]
    put_driver(dry, d)
    r, items = judged(tool, dry)
    assert items["B0"]["state"] == "failed" and "supervisor's record changed" in items["B0"]["evidence"]["why"]


def test_a_write_into_a_protected_place_reads_fail(tool, dry):
    d = driver(dry)
    d["steps"][0]["facts_end"]["experiments"] = ["a", "b", "new_run"]
    put_driver(dry, d)
    r, items = judged(tool, dry)
    assert items["A0"]["state"] == "failed" and "experiments changed" in items["A0"]["evidence"]["why"]


def test_verify_not_ok_reads_fail(tool, dry):
    d = driver(dry)
    d["steps"][1]["verify"] = dict(ok=False, output=dict(ok=False, checks=dict(admission_check="refused: code")))
    put_driver(dry, d)
    r, items = judged(tool, dry)
    assert items["B0"]["state"] == "failed"


def test_a_refused_start_reads_invalid(tool, tmp_path):
    root = tmp_path / "refused"
    root.mkdir()
    put_driver(root, dict(refused="supervisor_control.json cmd is 'run', not 'pause'", steps=[]))
    r, items = judged(tool, root)
    assert r["verdict"] == "invalid" and items["A0"]["state"] == "not reached"


# ---------------------------------------------------------------------------------------------- the driver
class Driver:
    """cmd_run with every outside step replaced: what it was asked to do, in sequence."""

    def __init__(self, t, monkeypatch, tmp_path, facts_seq=None, verify_ok=True, guard_ok=True, stopped=None):
        self.calls = []
        self.facts_seq = list(facts_seq or [])
        self.root = tmp_path / "drv"
        cand = tmp_path / "candidate.json"
        cand.write_text(json.dumps(candidate(t)), encoding="utf-8")
        self.args = type("A", (), dict(root=str(self.root), candidate=str(cand), runs=["A", "B"]))()
        monkeypatch.setattr(t, "facts", self.facts)
        monkeypatch.setattr(t, "copy_start", lambda *a, **k: self.calls.append("copy") or dict(identical=True))
        monkeypatch.setattr(t, "run_verify", lambda ck: self.calls.append("verify") or dict(ok=verify_ok))
        monkeypatch.setattr(t, "run_guard", lambda need: self.calls.append("guard %s" % need) or dict(
            allowed=guard_ok, reading={}))
        monkeypatch.setattr(t, "run_trainer", lambda cmd, log: self.calls.append(
            "trainer %s" % t.flag_values(cmd)["--max-steps"]) or dict(command=cmd, stopped=stopped))
        monkeypatch.setattr(t, "judge", lambda root: self.calls.append("judge") or dict(
            verdict="pass", verdict_reason="ok", items=[]))

    def facts(self):
        self.calls.append("facts")
        return self.facts_seq.pop(0) if self.facts_seq else facts()


def test_the_driver_runs_a_then_b_each_after_its_verify_and_guard(tool, monkeypatch, tmp_path):
    d = Driver(tool, monkeypatch, tmp_path)
    assert tool.cmd_run(d.args) == 0
    assert [c for c in d.calls if c != "facts"] == ["copy", "verify", "guard 32.0", "trainer 481800", "verify",
                                                     "guard 32.0", "trainer 26", "judge"]
    rec = json.loads((d.root / "driver_record.json").read_text(encoding="utf-8"))
    assert [s["run"] for s in rec["steps"]] == ["A", "B"] and rec["verdict"] == "pass"
    assert json.loads((d.root / "commands" / "run_B_command.json").read_text(encoding="utf-8"))[-1] == "--eval-only"


@pytest.mark.parametrize("first", [facts(supervisor_cmd="run"), facts(trainers=[{"pid": 1}]),
                                   facts(continuation_trainer_pids=[5]), facts(code_hash="0" * 64),
                                   facts(mimo_clean=False), facts(pause_runners=[{"pid": 9}])])
def test_the_driver_starts_nothing_when_a_precondition_fails(tool, monkeypatch, tmp_path, first):
    d = Driver(tool, monkeypatch, tmp_path, facts_seq=[first])
    assert tool.cmd_run(d.args) == 2
    assert not any(c.startswith(("copy", "trainer")) for c in d.calls)


def test_the_driver_refuses_a_launch_when_the_code_changed_after_the_verify(tool, monkeypatch, tmp_path):
    d = Driver(tool, monkeypatch, tmp_path, facts_seq=[facts(), facts(), facts(source_tree="8" * 40)])
    assert tool.cmd_run(d.args) == 2
    assert not any(c.startswith("trainer") for c in d.calls)


@pytest.mark.parametrize("verify_ok,guard_ok", [(False, True), (True, False)])
def test_the_driver_refuses_without_verify_or_guard(tool, monkeypatch, tmp_path, verify_ok, guard_ok):
    d = Driver(tool, monkeypatch, tmp_path, verify_ok=verify_ok, guard_ok=guard_ok)
    assert tool.cmd_run(d.args) == 2
    assert not any(c.startswith("trainer") for c in d.calls)


def test_the_driver_stops_after_a_run_the_commit_watch_ended(tool, monkeypatch, tmp_path):
    d = Driver(tool, monkeypatch, tmp_path, stopped="ended at ...: inside the guard's 6 GB reserve")
    assert tool.cmd_run(d.args) == 2
    assert [c for c in d.calls if c.startswith("trainer")] == ["trainer 481800"]


# ---------------------------------------------------------------------------------------------- the watch
def test_the_watch_ends_a_trainer_inside_the_guards_reserve(tool, tmp_path):
    rec = tool.run_trainer([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "log.txt",
                           poll_s=0.2, commit=lambda: (80.0, 84.4), gpu=lambda: None,
                           continuation=lambda: {"trainer_pids": [], "active_run": "r", "command": ["x"]})
    assert rec["stopped"] and "inside the guard's 6 GB reserve" in rec["stopped"]
    assert rec["seconds"] < 30 and rec["commit_peak_gb"] == 80.0


def test_the_watch_records_the_supervisor_while_the_trainer_lives(tool, tmp_path):
    seen = iter([{"trainer_pids": [], "active_run": "r", "command": ["x"]}] +
                [{"trainer_pids": [99], "active_run": "other", "command": ["y"], "status": "training"}] * 1000)
    rec = tool.run_trainer([sys.executable, "-c", "import time; time.sleep(1.0)"], tmp_path / "log.txt",
                           poll_s=0.1, commit=lambda: (40.0, 84.4), gpu=lambda: 100.0,
                           continuation=lambda: next(seen))
    assert rec["stopped"] is None and rec["exit_code"] == 0
    assert rec["supervisor_trainer_pids_seen"] == [99] and rec["supervisor_active_run_changed"]
    assert rec["supervisor_command_changed"] and rec["gpu_peak_mib"] == 100.0
