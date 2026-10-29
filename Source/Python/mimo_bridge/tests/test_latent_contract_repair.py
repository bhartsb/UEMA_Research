# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE LATENT CONTRACT REPAIR (Oct 10, 2026; defect LATENT_SHADOW_REJECTS_REPAIRED_FINGER_CONTRACT).

The shadow under the repaired finger servo's command contract, wrist_relative_joint_v1+finger_joint_servo_v1, declared in
its own configuration (docs/gain_of_function_v4/latent_body_shadow_config_h84_finger_joint_servo_v1_2026-10-10.json),
beside the unchanged Sep 30 configuration (wrist_relative_joint_v1). No trainer file changes.

The observations and commands are REAL H84 rows: two bodies, model steps 481,800 to 481,806, copied from the learning-on
control arm's shadow record by docs/gain_of_function_v4/latent_contract_repair/make_fixture.py
(latent_contract_rows_481802.npz; its provenance in latent_contract_rows_481802.json). They were recorded under
wrist_relative_joint_v1. Each test decides which contract its replayed frames report and its transitions carry, as the
trainer would write them with the servo off or on; no recording is changed. The tests use the real configuration
files, so the hierarchy has its real size (latent 8, hidden 128).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lem.latent_body.checkpoint import fingerprint
from lem.latent_body.learning import ShadowLearner, validate_config
from lem.latent_body.trajectory import make_batch, row_options
from lem.latent_body.tree import skin_owner
from lem.model.contact_head import PATCH_NAMES
from mimo_bridge import copied_execution as CE
from mimo_bridge.latent_body_runtime import ShadowRuntime, observation, observed_contract
from mimo_bridge.latent_body_shadow import _RUNTIMES, add_arguments, observe_shadow
from mimo_bridge.limb_target_control import action_contract_parts
from mimo_bridge.tools import latent_body_start_state as S
from mimo_bridge.transition import TransitionV1, known_action_contract

REPO = Path(__file__).resolve().parents[4]
DOCS = REPO / "docs" / "gain_of_function_v4"
OLD_CONFIG = DOCS / "latent_body_shadow_config_h84_2026-09-30.json"
NEW_CONFIG = DOCS / "latent_body_shadow_config_h84_finger_joint_servo_v1_2026-10-10.json"
ROWS = Path(__file__).with_name("latent_contract_rows_481802.npz")
LEVELS = "arm,hand,index"
OLD, NEW = "wrist_relative_joint_v1", "wrist_relative_joint_v1+finger_joint_servo_v1"


# ------------------------------------------------------------------------------------------------ the recorded rows
def recorded():
    data = np.load(ROWS)
    bodies = sorted({int(k.split("_")[0][1:]) for k in data.files})
    return {b: {k.split("_", 1)[1]: data[k] for k in data.files if k.startswith("b%d_" % b)} for b in bodies}


TREE = S.tree_for(LEVELS)
DATA = recorded()
FIRST = int(DATA[min(DATA)]["frame"][0])          # 481,800
N_FRAMES = len(DATA[min(DATA)]["frame"])          # 7: frames 481,800 to 481,806


def patch_bits(contact):
    """The skin's patch states that give the recorded per-handle contact (one patch of each touching handle)."""
    owner_patch = {}
    for name in PATCH_NAMES:
        owner_patch.setdefault(skin_owner(name), name)
    bits = [0] * len(PATCH_NAMES)
    for h, value in zip(TREE.handle_names, contact):
        if value:
            bits[PATCH_NAMES.index(owner_patch[h])] = 1
    return bits


def frame(body, k, servo):
    d = DATA[body]
    meta = {"task": {"contact_patch_state": patch_bits(d["contact"][k])}, "hand_target_contract": OLD}
    if servo:
        meta["finger_servo"] = "finger_joint_servo_v1"       # what a world reports under --finger-joint-servo on
    return SimpleNamespace(env_index=body, sim_time=float(d["sim_time"][k]), bones=d["bones"][k].astype(np.float32),
                           meta=meta)


def transition(body, k, contract):
    """The transition that ends at frame index k (k >= 2: the recording holds those from frame 481,801 on)."""
    d, j = DATA[body], k - 2
    return TransitionV1(env_id=body, episode_id=int(d["episode"][k]), frame_t=int(d["frame"][k - 1]),
                        frame_t1=int(d["frame"][k]), duration_s=.05, state_t={}, state_t1={}, achieved_t=None,
                        context_t=None, proposed_action=None, executed_action=d["executed"][j],
                        decoded_ctrl=d["decoded_ctrl"][j], order_id=int(d["order_id"][j]), action_contract=contract,
                        motor_substeps=d["motor_substeps"][j], hand_joint_goal=d["hand_joint_goal"][j],
                        source_mask_t={}, source_mask_t1={})


def step_inputs(k, servo, contract):
    frames = [frame(b, k, servo) for b in DATA]
    trs = [] if k < 2 else [transition(b, k, contract) for b in DATA]
    return frames, trs, int(DATA[min(DATA)]["frame"][k]), int(DATA[min(DATA)]["episode"][k])


def arguments(tmp_path, config, name="shadow", resume=""):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    args.latent_body_hierarchy, args.latent_body_config = "shadow", str(config)
    args.latent_body_output, args.latent_body_resume = str(tmp_path / name), resume
    args.out, args.eval_only, args.report_every = str(tmp_path / "production"), False, 100
    return args


def hook_run(args, servo, contract, frames=range(N_FRAMES)):
    """The trainer's own hook (observe_shadow), called once per model step as train_mimo calls it."""
    model = torch.nn.Linear(2, 2)
    model.handle_names = TREE.handle_names
    for k in frames:
        fr, trs, step, episode = step_inputs(k, servo, contract)
        observe_shadow(args, model, fr, trs, step, episode)
    return model


def resume_run(args, first=2, last=N_FRAMES):
    """The trainer's first model steps after a resume from a checkpoint saved under the pose servo, as the episode
    companion of 481,800 hands them over (docs/gain_of_function_v4/latent_contract_repair/evidence/
    companion_481800_contract.json, read by read_companion_contract.py): at the first model step the frames carry the
    restored reports (hand_target_contract wrist_relative_joint_v1, no finger_servo; train_mimo.py sets f.meta from
    the companion's frame_meta) and the transitions the trainer closes are the restored open ones, executed before the
    checkpoint under the restored prev_action_contract wrist_relative_joint_v1; from the next model step on the worlds
    report the servo and the trainer writes the repaired contract. Here the first model step is frame index `first`,
    whose recorded transition was executed under wrist_relative_joint_v1 as the restored one was."""
    model = torch.nn.Linear(2, 2)
    model.handle_names = TREE.handle_names
    for k in range(first, last):
        boundary = k == first
        fr, trs, step, episode = step_inputs(k, servo=not boundary, contract=OLD if boundary else NEW)
        observe_shadow(args, model, fr, trs, step, episode)
    return model


def events(folder):
    return [json.loads(x) for x in (Path(folder) / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def rows(folder):
    path = Path(folder) / "trajectory.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


# ------------------------------------------------------------------------------------------------ the declaration
def test_repaired_configuration_is_the_sep30_one_with_its_contract_declared():
    old = json.loads(OLD_CONFIG.read_text(encoding="utf-8"))
    new = json.loads(NEW_CONFIG.read_text(encoding="utf-8"))
    validate_config(old)
    validate_config(new)
    assert S.config_differences(old, new) == ["action_contract", "measurement"]
    assert (old["action_contract"], new["action_contract"]) == (OLD, NEW) and NEW == S.REPAIRED_CONTRACT
    assert known_action_contract(NEW)
    assert fingerprint(old) != fingerprint(new)               # a state of one never loads under the other


def test_declared_contract_is_the_one_finger_servo_frames_report():
    servo_on = {"hand_target_contract": OLD, "finger_servo": "finger_joint_servo_v1"}
    assert action_contract_parts(servo_on) == ["finger_joint_servo_v1"]
    assert observed_contract(SimpleNamespace(meta=servo_on)) == NEW
    assert observed_contract(SimpleNamespace(meta={"hand_target_contract": OLD})) == OLD


# ------------------------------------------------------------------------------------------------ the failure, and the repair
def test_the_sep30_configuration_reproduces_the_failure_and_disables_only_the_shadow(tmp_path):
    args = arguments(tmp_path, OLD_CONFIG)
    model = hook_run(args, servo=True, contract=NEW, frames=range(3))
    assert _RUNTIMES[model] is False
    record = events(tmp_path / "shadow")
    assert [e["type"] for e in record] == ["started", "failed"]
    assert "Action contract changed" in record[-1]["reason"] and record[-1]["saved_updates"] == 0
    assert rows(tmp_path / "shadow") == []


def test_the_repaired_configuration_records_learns_labels_and_saves_through_the_trainer_hook(tmp_path):
    args = arguments(tmp_path, NEW_CONFIG)
    model = hook_run(args, servo=True, contract=NEW)
    runtime = _RUNTIMES[model]
    assert runtime is not False and runtime.learner.updates == N_FRAMES - 2
    record = events(tmp_path / "shadow")
    assert record[0]["config"]["action_contract"] == NEW and not [e for e in record if e["type"] == "failed"]
    assert [e["updates"] for e in record if e["type"] == "latent_body_lesson"] == list(range(1, N_FRAMES - 1))
    written = rows(tmp_path / "shadow")
    assert len(written) == len(DATA) * (N_FRAMES - 2) and {r["action_contract"] for r in written} == {NEW}
    assert sorted({r["end"]["frame"] for r in written}) == [FIRST + k for k in range(2, N_FRAMES)]
    state = torch.load(tmp_path / "shadow" / "latest.pt", map_location="cpu", weights_only=True)
    assert state["config"]["action_contract"] == NEW and state["updates"] == N_FRAMES - 2


# ------------------------------------------------------------------------------------------------ the resume boundary
def test_the_resume_boundary_is_recorded_as_excluded_and_the_shadow_learns_on(tmp_path):
    """The check's FAIL of 4:19:44 AM PDT, Oct 10 (latent_contract_check_481800), at the runtime of this commit (marked
    NOT FOR MERGE): the first transitions after a resume from a pose-servo checkpoint carry wrist_relative_joint_v1;
    each is named in one excluded_transitions record with its contract and its model steps, its outcome kept as
    physical history, and the shadow learns from the next transitions, every row labelled with the repaired contract.
    (On the runtime before this commit the same model steps stop the shadow with "Action contract changed", the
    check's records; the earlier tests began without a transition and gave every frame the servo's report.)"""
    args = arguments(tmp_path, NEW_CONFIG)
    model = resume_run(args)
    runtime = _RUNTIMES[model]
    assert runtime is not False
    record = events(tmp_path / "shadow")
    kinds = [e["type"] for e in record]
    assert kinds[:2] == ["started", "excluded_transitions"] and "failed" not in kinds
    boundary = record[1]
    first = int(DATA[min(DATA)]["frame"][2])
    assert boundary["step"] == boundary["first_step"] == first and boundary["declared_contract"] == NEW
    assert [(x["env_id"], x["frame_t"], x["frame_t1"], x["action_contract"]) for x in boundary["transitions"]] == [
        (b, first - 1, first, OLD) for b in DATA]
    lessons = [e for e in record if e["type"] == "latent_body_lesson"]
    assert [e["updates"] for e in lessons] == [1, 2, 3]            # from the third call, as in the control arms
    written = rows(tmp_path / "shadow")
    assert len(written) == 3 * len(DATA) and {r["action_contract"] for r in written} == {NEW}
    assert min(r["history"]["frame"] for r in written) == first     # the boundary outcome kept as history
    assert S.boundary_answer([boundary], NEW, expected=len(DATA))[:2] == (True, len(DATA))


def test_a_transition_of_another_contract_after_the_first_model_step_still_stops_the_shadow(tmp_path):
    args = arguments(tmp_path, NEW_CONFIG)
    model = torch.nn.Linear(2, 2)
    model.handle_names = TREE.handle_names
    for k in range(2, 5):
        fr, trs, step, episode = step_inputs(k, servo=k != 4, contract=NEW if k != 4 else OLD)
        observe_shadow(args, model, fr, trs, step, episode)
    assert _RUNTIMES[model] is False
    record = events(tmp_path / "shadow")
    assert "Action contract changed" in record[-1]["reason"]
    assert not [e for e in record if e["type"] == "excluded_transitions"]


def test_the_old_configuration_at_the_same_resume_excludes_nothing(tmp_path):
    """The control arms' path: under the Sep 30 configuration the restored transitions carry its own contract and are
    taken as before; no excluded record is written."""
    args = arguments(tmp_path, OLD_CONFIG, name="old")
    model = torch.nn.Linear(2, 2)
    model.handle_names = TREE.handle_names
    for k in range(2, N_FRAMES):
        fr, trs, step, episode = step_inputs(k, servo=False, contract=OLD)
        observe_shadow(args, model, fr, trs, step, episode)
    record = events(tmp_path / "old")
    assert [e["updates"] for e in record if e["type"] == "latent_body_lesson"] == [1, 2, 3]
    assert not [e for e in record if e["type"] in ("excluded_transitions", "failed")]


def test_a_relaunch_resumed_under_the_servo_excludes_nothing(tmp_path):
    """A supervisor relaunch of a run already under the servo: the restored transitions carry the repaired contract,
    the new runtime's first model step takes them, and its saved history joins them."""
    args = arguments(tmp_path, NEW_CONFIG, resume="auto")
    resume_run(args, last=5)                                         # the first launch: frame indices 2 to 4
    again = torch.nn.Linear(2, 2)
    again.handle_names = TREE.handle_names
    for k in range(5, N_FRAMES):                                     # the relaunch starts at frame index 5
        fr, trs, step, episode = step_inputs(k, servo=True, contract=NEW)
        observe_shadow(args, again, fr, trs, step, episode)
    assert _RUNTIMES[again] is not False
    record = events(tmp_path / "shadow")
    assert [e["type"] for e in record].count("excluded_transitions") == 1 and [e["type"] for e in record].count(
        "started") == 2
    assert [e["updates"] for e in record if e["type"] == "latent_body_lesson"] == [1, 2, 3]


def test_the_run_check_reads_the_boundary_record():
    def record(**kw):
        base = dict(type="excluded_transitions", step=481800, first_step=481800, declared_contract=NEW,
                    transitions=[dict(env_id=b, frame_t=481799, frame_t1=481800, action_contract=OLD) for b in range(8)])
        base.update(kw)
        return base
    assert S.boundary_answer([], NEW)[0] and S.boundary_answer([], NEW, expected=8)[0] is False
    assert S.boundary_answer([record()], NEW, expected=8)[:2] == (True, 8)
    assert not S.boundary_answer([record(step=481801)], NEW)[0]                      # not the first model step
    assert not S.boundary_answer([record(), record()], NEW)[0]                       # more than one record
    bad = record()
    bad["transitions"][0]["action_contract"] = NEW
    assert not S.boundary_answer([bad], NEW)[0]                                      # a declared-contract transition
    late = record()
    late["transitions"][0]["frame_t"] = 481800
    assert not S.boundary_answer([late], NEW)[0]                                     # executed after the start


# ------------------------------------------------------------------------------------------------ never mixed
def test_a_learner_window_never_mixes_two_contracts(tmp_path):
    args = arguments(tmp_path, NEW_CONFIG)
    hook_run(args, servo=True, contract=NEW)
    examples = [S.example_from_row(r) for r in rows(tmp_path / "shadow")]
    learner = ShadowLearner(TREE, json.loads(NEW_CONFIG.read_text(encoding="utf-8")))
    learner.learn(examples[:2])
    before = learner.state_dict()
    mixed = examples[2:4]
    mixed[1] = copy.deepcopy(mixed[1])
    mixed[1].action_contract = OLD
    with pytest.raises(ValueError, match="Action contract changed"):
        learner.learn(mixed)
    assert CE.equal_tree(before, learner.state_dict()) and learner.updates == 1
    cfg = learner.config
    with pytest.raises(ValueError, match="Action contract changed"):
        make_batch(TREE, mixed, scales=cfg["scales"], duration_s=cfg["duration_s"], action_contract=NEW,
                   hand_frame=cfg["hand_action_frame"], **row_options(cfg))
    # the extractor refuses a command of the other contract in the middle of a body's stream
    extractor = ShadowLearner(TREE, cfg).extractor
    seq = [observation(frame(1, k, True), TREE, int(DATA[1]["frame"][k]), int(DATA[1]["episode"][k])) for k in range(4)]
    extractor.observe(seq[0])
    extractor.observe(seq[1])
    assert extractor.observe(seq[2], DATA[1]["executed"][0], action_contract=NEW,
                             frame_t=int(DATA[1]["frame"][1])).action_contract == NEW
    with pytest.raises(ValueError, match="Action contract changed"):
        extractor.observe(seq[3], DATA[1]["executed"][1], action_contract=OLD, frame_t=int(DATA[1]["frame"][2]))


def test_a_transition_labelled_against_its_frames_is_refused_before_the_extractor(tmp_path):
    runtime = ShadowRuntime(arguments(tmp_path, NEW_CONFIG), TREE.handle_names)
    for k in range(2):
        fr, trs, step, episode = step_inputs(k, True, NEW)
        runtime.observe(fr, trs, step, episode, train=True)
    fr, _, step, episode = step_inputs(2, True, NEW)
    with pytest.raises(ValueError, match="Transition and physical command contracts differ"):
        runtime.observe(fr, [transition(b, 2, OLD) for b in DATA], step, episode, train=True)
    assert runtime.learner.updates == 0


# ------------------------------------------------------------------------------------------------ saved states
def test_a_saved_state_carries_its_contract_and_resumes_only_under_its_configuration(tmp_path):
    args = arguments(tmp_path, NEW_CONFIG, resume="auto")
    first = ShadowRuntime(args, TREE.handle_names)
    for k in range(5):
        fr, trs, step, episode = step_inputs(k, True, NEW)
        first.observe(fr, trs, step, episode, train=True)
    assert first.learner.updates == 3
    state = torch.load(first.latest, map_location="cpu", weights_only=True)
    assert state["config"]["action_contract"] == NEW
    with pytest.raises(ValueError, match="configuration changed"):
        ShadowLearner(TREE, json.loads(OLD_CONFIG.read_text(encoding="utf-8"))).load_state_dict(state)
    again = ShadowRuntime(args, TREE.handle_names)               # a relaunch with the same command
    assert again.learner.updates == 3 and CE.equal_tree(again.learner.state_dict(), first.learner.state_dict())
    for k in range(5, N_FRAMES):
        fr, trs, step, episode = step_inputs(k, True, NEW)
        again.observe(fr, trs, step, episode, train=True)
    assert again.learner.updates == N_FRAMES - 2                # the saved history joined the next frames


def test_the_old_contract_path_is_unchanged_and_its_states_still_load(tmp_path):
    args = arguments(tmp_path, OLD_CONFIG, name="old")
    model = hook_run(args, servo=False, contract=OLD)
    assert _RUNTIMES[model].learner.updates == N_FRAMES - 2
    assert {r["action_contract"] for r in rows(tmp_path / "old")} == {OLD}
    latest = tmp_path / "old" / "latest.pt"
    old = S.state_summary(latest, TREE, json.loads(OLD_CONFIG.read_text(encoding="utf-8")))
    assert old["action_contract"] == OLD and old["updates"] == N_FRAMES - 2
    with pytest.raises(ValueError, match="configuration changed"):
        S.state_summary(latest, TREE, json.loads(NEW_CONFIG.read_text(encoding="utf-8")))
    resumed = ShadowRuntime(arguments(tmp_path, OLD_CONFIG, name="old", resume=str(latest)), TREE.handle_names)
    assert resumed.learner.updates == N_FRAMES - 2


# ------------------------------------------------------------------------------------------------ the declared start
def test_a_fresh_start_is_a_named_file_equal_to_the_runtimes_own_fresh_build(tmp_path):
    config = S.load_config(NEW_CONFIG)
    out = tmp_path / "start" / "latent_start_fresh.pt"
    record = S.write_start(out, S.fresh_start(TREE, config), S.start_record("fresh", NEW_CONFIG, config, TREE, LEVELS,
                                                                            "trunk_v1"))
    assert record["sha256"] == S.sha256_file(out) and json.loads(Path(str(out) + ".json").read_text())["kind"] == "fresh"
    summary = S.state_summary(out, TREE, config)
    assert (summary["action_contract"], summary["updates"], summary["start"]["kind"]) == (NEW, 0, "fresh")
    assert CE.equal_tree(torch.load(out, weights_only=True)["model"], ShadowLearner(TREE, config).model.state_dict())
    args = arguments(tmp_path, NEW_CONFIG, resume=str(out))
    hook_run(args, servo=True, contract=NEW)
    record_events = events(tmp_path / "shadow")
    assert Path(record_events[0]["resume"]) == out
    assert [e["updates"] for e in record_events if e["type"] == "latent_body_lesson"][0] == 1
    assert S.sha256_file(out) == record["sha256"]                # the start file is read, never written
    result = S.verify_run(tmp_path / "shadow", NEW_CONFIG, record, N_FRAMES, old_config_path=OLD_CONFIG)
    assert result["verdict"] == "pass", result["problems"]
    assert result["restore"]["refused_under_old"] is True and result["restore"]["continued_to"] == N_FRAMES - 1


def test_a_copied_start_carries_the_old_weights_only_and_names_its_source(tmp_path):
    hook_run(arguments(tmp_path, OLD_CONFIG, name="old"), servo=False, contract=OLD)
    source = tmp_path / "old" / "latest.pt"
    before = S.sha256_file(source)
    source_state = torch.load(source, map_location="cpu", weights_only=True)
    config = S.load_config(NEW_CONFIG)
    learner, summary = S.copy_start(TREE, config, source_state)
    assert (summary["action_contract"], summary["updates"]) == (OLD, N_FRAMES - 2)
    assert CE.equal_tree(learner.model.state_dict(), source_state["model"])
    assert learner.updates == 0 and learner.examples == 0 and learner.extractor.previous == {}
    assert learner.optimizer.state_dict()["state"] == {}
    out = tmp_path / "start" / "latent_start_copy.pt"
    summary.update(path=str(source), sha256=before)
    record = S.write_start(out, learner, S.start_record("copy", NEW_CONFIG, config, TREE, LEVELS, "trunk_v1",
                                                         source=summary))
    assert S.sha256_file(source) == before and record["source"]["sha256"] == before
    assert S.state_summary(out, TREE, config)["start"]["source"]["action_contract"] == OLD
    hook_run(arguments(tmp_path, NEW_CONFIG, name="from_copy", resume=str(out)), servo=True, contract=NEW)
    assert [e["updates"] for e in events(tmp_path / "from_copy") if e["type"] == "latent_body_lesson"][0] == 1


def test_a_copy_is_refused_across_meanings_and_a_start_is_never_written_over_records(tmp_path):
    hook_run(arguments(tmp_path, OLD_CONFIG, name="old"), servo=False, contract=OLD)
    state = torch.load(tmp_path / "old" / "latest.pt", map_location="cpu", weights_only=True)
    with pytest.raises(ValueError, match="already holds the contract"):
        S.copy_start(TREE, S.load_config(OLD_CONFIG), state)
    other = S.load_config(NEW_CONFIG)
    other["scales"] = copy.deepcopy(other["scales"])
    other["scales"]["change"][0] *= 2
    with pytest.raises(ValueError, match="differs in"):
        S.copy_start(TREE, other, state)
    assert S.target_problems(tmp_path / "old" / "latest.pt")              # exists, and beside records
    assert any("records" in p for p in S.target_problems(tmp_path / "old" / "start.pt"))
    assert any("Saved" in p for p in S.target_problems(tmp_path / "Saved" / "x" / "start.pt"))
    assert any("preserved" in p for p in S.target_problems(Path("D:/UEMA_latent_body/seg41_v2/start.pt")))
    assert S.target_problems(tmp_path / "new_folder" / "start.pt") == []


# ------------------------------------------------------------------------------------------------ the run check
def test_the_run_check_fails_on_each_broken_part(tmp_path):
    config = S.load_config(NEW_CONFIG)
    out = tmp_path / "start" / "s.pt"
    record = S.write_start(out, S.fresh_start(TREE, config),
                           S.start_record("fresh", NEW_CONFIG, config, TREE, LEVELS, "trunk_v1"))
    hook_run(arguments(tmp_path, NEW_CONFIG, resume=str(out)), servo=True, contract=NEW)
    good = tmp_path / "shadow"
    args_file = tmp_path / "args.json"
    args_file.write_text(json.dumps(dict(finger_joint_servo="on", latent_body_config=str(NEW_CONFIG))))
    log = tmp_path / "arm.log"
    log.write_text("step 481900 | ...\n")
    assert S.verify_run(good, NEW_CONFIG, record, N_FRAMES, log=log, args_path=args_file)["verdict"] == "pass"

    def broken(name, change):
        folder = tmp_path / name
        import shutil
        shutil.copytree(good, folder)
        change(folder)
        return S.verify_run(folder, NEW_CONFIG, record, N_FRAMES, log=log, args_path=args_file)

    def relabel(folder):
        lines = (folder / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace(NEW, OLD)
        (folder / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def drop_lesson(folder):
        lines = (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
        (folder / "events.jsonl").write_text("\n".join(lines[:2] + lines[3:]) + "\n", encoding="utf-8")

    def add_failure(folder):
        with (folder / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(type="failed", authority="none", reason="x", saved_updates=0)) + "\n")

    assert broken("relabelled", relabel)["verdict"] == "fail"
    assert broken("gap", drop_lesson)["verdict"] == "fail"
    assert broken("failed", add_failure)["verdict"] == "fail"
    assert broken("no_state", lambda f: (f / "latest.pt").unlink())["verdict"] == "fail"
    log.write_text(S.FAILED_LINE + "; its output remains unused\n")
    assert S.verify_run(good, NEW_CONFIG, record, N_FRAMES, log=log, args_path=args_file)["verdict"] == "fail"
    log.write_text("")
    args_file.write_text(json.dumps(dict(finger_joint_servo="off", latent_body_config=str(NEW_CONFIG))))
    assert S.verify_run(good, NEW_CONFIG, record, N_FRAMES, log=log, args_path=args_file)["verdict"] == "fail"
    assert S.verify_run(good, NEW_CONFIG, record, N_FRAMES + 3)["verdict"] == "fail"     # fewer updates than steps give
