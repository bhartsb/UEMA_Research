# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The whole-body row convention (trajectory.BodyRows, Oct 1, 2026): frames, the eyes, learned numbers, scales, round
trips through handle_targets, contacts, and the shadow switch. Each test names the defect it catches."""
import argparse
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from lem.latent_body.learning import ShadowLearner, validate_config
from lem.latent_body.losses import weighted_mse
from lem.latent_body.model import LocalPrediction
from lem.latent_body.trajectory import (GROUP_OF, PART_GROUPS, Example, Observation, action_rows, body_lesson_weights,
                                        body_rows, change_rows, end_positions, executed_rows, eye_quaternion,
                                        eye_row_physical, eye_yaw_pitch, make_batch, predicted_rows, quaternion,
                                        rotation_delta, row_options, row_scales)
from lem.latent_body.tree import arm_hand_tree, full_body_tree
from lem.model.contact_head import PATCH_NAMES
from lem.planner.latent_rollout import handle_targets
from mimo_bridge.body_layout import names_for

NAMES = names_for("trunk_v1")
CONTRACT = "wrist_relative_joint_v1"


def unit_scales(value=1.):
    return dict(state=[value] * 12, change=[value] * 6, action=[value] * 8)


def config(contract=CONTRACT, **spec):
    base = dict(architecture=dict(latent_dim=2, hidden=16, message_dim=2), scales=unit_scales(),
                weights=dict(future=1., contact=1., action=1., child=1., consistency=1., latent=.001),
                learning_rate=.003, gradient_clip=10., duration_s=.05, action_contract=contract, seed=7,
                measurement="synthetic unit fixture only", data_quality="plumbing", substrate_receipt="",
                hand_action_frame="actual_wrist", digit_frame="wrist")
    if "wrist_relative_joint_v1" not in contract:
        base.pop("hand_action_frame"), base.pop("digit_frame")
    rows = dict(version="contract_v1", eye_row_convention="legacy", future_rows="own",
                part_scales={g: unit_scales(1. + k) for k, g in enumerate(sorted(PART_GROUPS))})
    rows.update(spec)
    return dict(base, body_rows=rows)


def _quats(n, seed):
    return torch.tensor(R.random(n, random_state=seed).as_quat(), dtype=torch.float32)


def _obs(frame, seed, base=None):
    g = torch.Generator().manual_seed(seed)
    bones = torch.zeros(26, 11)
    bones[:, :4] = _quats(26, seed)
    bones[:, 4:7] = torch.randn(26, 3, generator=g) * 10
    bones[:, 7:10] = torch.randn(26, 3, generator=g) * 20
    bones[:, 10] = 1
    gi = NAMES.index("gaze")
    bones[gi, :4] = eye_quaternion(torch.tensor(0.2 + .01 * seed), torch.tensor(-0.1 - .01 * seed))   # head-relative
    return Observation(0, 0, frame, frame * .05, bones, torch.zeros(26))


def _example(seed=0, contract=CONTRACT):
    g = torch.Generator().manual_seed(100 + seed)
    h, s, e = _obs(0, seed), _obs(1, seed + 1), _obs(2, seed + 2)
    executed = torch.zeros(26, 9)
    executed[:, :3] = s.bones[:, 4:7] + torch.randn(26, 3, generator=g) * 2
    executed[:, 3:7] = _quats(26, 50 + seed)
    gi = NAMES.index("gaze")
    executed[gi, 3:7] = eye_quaternion(torch.tensor(0.4), torch.tensor(0.25))   # a pure yaw-pitch eye row
    executed[:, 7:9] = torch.tensor([.5, .7])
    return Example(h, s, e, executed, contract)


# ------------------------------------------------------------------ declaration
def test_the_whole_body_declaration_is_complete_and_refuses_what_it_cannot_represent():
    validate_config(config())
    for broken in (dict(eye_row_convention="sideways"), dict(version="v0"), dict(future_rows="some"),
                   dict(part_scales={g: unit_scales() for g in sorted(PART_GROUPS) if g != "gaze"})):
        with pytest.raises(ValueError):
            validate_config(config(**broken))
    with pytest.raises(ValueError, match="upper body"):          # the chest goal sits on an unreported body
        body_rows(config("wrist_relative_joint_v1+parent_relative_calf_v1+parent_relative_chest_v1"))
    with pytest.raises(ValueError, match="digit_scales"):
        validate_config(dict(config(), digit_scales=unit_scales()))
    with pytest.raises(ValueError, match="digit_frame wrist"):
        validate_config(dict(config(), digit_frame="world"))
    # a tree beyond the pilot needs the declaration; the pilot does not
    cfg = config()
    cfg.pop("body_rows")
    with pytest.raises(ValueError, match="body_rows"):
        ShadowLearner(full_body_tree(NAMES), cfg)
    ShadowLearner(arm_hand_tree(NAMES), cfg)
    ShadowLearner(full_body_tree(NAMES), config())


# ------------------------------------------------------------------ the eyes
def test_the_legacy_eye_row_is_the_physical_rotation_with_its_pitch_reversed():
    # Defect caught: learning the legacy row as the eye's rotation. Measured on 6,000 seg41_v2 rows the eye's pitch
    # moves toward minus the commanded pitch (correlation 0.551, against -0.332 for plus).
    yaw, pitch = torch.tensor([0.3, -0.5]), torch.tensor([0.2, -0.4])
    row = eye_quaternion(yaw, pitch)
    phys = eye_row_physical(row, "legacy")
    y2, p2 = eye_yaw_pitch(phys)
    assert torch.allclose(y2, yaw, atol=1e-6) and torch.allclose(p2, -pitch, atol=1e-6)
    assert torch.allclose(eye_row_physical(phys, "legacy"), quaternion(row), atol=1e-6)     # its own inverse
    assert torch.allclose(eye_row_physical(row, "physical"), quaternion(row))
    # the left eye's vertical hinge axis is 0 -1 0 (MIMo_modelv2.xml): a joint angle v is the rotation Ry(-v)
    v = 0.3
    assert np.allclose(R.from_rotvec([0, -v, 0]).as_quat(), eye_quaternion(torch.tensor(0.), torch.tensor(-v)).numpy(),
                       atol=1e-6)


def test_an_eye_command_equal_to_the_eye_asks_for_no_turn():
    # Defect caught: the eye command compared with the eye in two conventions (the turn is then the elevation sum).
    body = row_options(config())["body"]
    ex = _example()
    gi = NAMES.index("gaze")
    ex.start.bones[gi, :4] = eye_quaternion(torch.tensor(0.3), torch.tensor(-0.2))      # physical: pitch -0.2
    ex.executed[gi, 3:7] = eye_quaternion(torch.tensor(0.3), torch.tensor(0.2))         # legacy row: pitch +0.2
    a = action_rows(ex.start, ex.executed, [gi], NAMES, "wrist", body)
    assert a[0, 3:6].abs().max() < 1e-5


def test_the_eyes_carried_by_the_head_do_not_move_in_their_rows():
    # Defect caught: the eye's world position change (the head carrying it) taught as the eye's own movement.
    body = row_options(config())["body"]
    ex = _example(3)
    hi, gi = NAMES.index("head"), NAMES.index("gaze")
    turn = R.from_rotvec([0.2, -0.1, 0.3])
    end = ex.end
    end.bones = ex.start.bones.clone()
    pivot = ex.start.bones[hi, 4:7].double().numpy()
    end.bones[hi, :4] = torch.tensor((turn * R.from_quat(ex.start.bones[hi, :4].double().numpy())).as_quat(),
                                     dtype=torch.float32)
    end.bones[gi, 4:7] = torch.tensor(turn.apply(ex.start.bones[gi, 4:7].double().numpy() - pivot) + pivot,
                                      dtype=torch.float32)
    rows = change_rows(ex.start, end, [hi, gi], NAMES, "wrist", body)
    assert rows[1].abs().max() < 1e-4                                    # head-relative: nothing moved
    assert change_rows(ex.start, end, [gi], NAMES, "wrist")[0, :3].norm() > 0.1   # the world saw it carried
    # the world end position comes back from the head's change (end_positions)
    world = end_positions(ex.start, [hi, gi], rows.unsqueeze(0), NAMES, "wrist", body)[0]
    assert torch.allclose(world[1], end.bones[gi, 4:7], atol=1e-3)


# ------------------------------------------------------------------ learned numbers
def test_only_issued_and_executed_numbers_are_learned_as_commands():
    # Defects caught: an unread position (arm segments, legs, chest, eyes) learned as a command; radius or strength
    # learned though the hierarchy never issues them; the eyes' position change graded; unrecorded contact graded.
    tree = full_body_tree(NAMES)
    body = row_options(config(contact_recorded=sorted(arm_hand_tree(NAMES).owners)))["body"]
    w = body_lesson_weights(tree, body)
    act = {n: w[n][1].reshape(-1, 8) for n in w}
    for part, read in (("hand_L", True), ("index_R", True), ("trunk", True), ("head", True), ("arm_L", False),
                       ("chest", False), ("thigh_L", False), ("calf_R", False), ("foot_L", False), ("gaze", False)):
        assert bool(act[part][:, :3].all()) is read and not act[part][:, 3:6].eq(0).any(), part
        assert act[part][:, 6:].abs().sum() == 0, part
    change = w["gaze"][0].reshape(-1, 6)
    assert change[0, :3].abs().sum() == 0 and change[0, 3:].eq(1).all()
    contact = {n: w[n][2] for n in w}
    assert contact["gaze"].sum() == 0 and contact["foot_L"].sum() == 0 and contact["index_L"].sum() == 1
    # future_rows own: the trunk grades only its own row, the hand only the wrist
    trunk = w["trunk"][0].reshape(-1, 6)
    assert trunk.sum() == 6 and trunk[list(tree.subtree["trunk"]).index(NAMES.index("lower_trunk"))].eq(1).all()
    # the operator's no_head_pelvis_position_v1: the head's and pelvis's positions are then read by nothing
    body2 = row_options(config("wrist_relative_joint_v1+no_head_pelvis_position_v1"))["body"]
    w2 = body_lesson_weights(tree, body2)
    assert w2["head"][1].reshape(-1, 8)[:, :3].sum() == 0 and w2["trunk"][1].reshape(-1, 8)[:, :3].sum() == 0
    # with every contact recorded, the legs' skin is graded
    w3 = body_lesson_weights(tree, row_options(config())["body"])
    assert w3["foot_L"][2].sum() == 1


def test_weighted_lessons_ignore_unweighted_numbers_and_match_the_plain_mean_when_all_count():
    p, t = torch.randn(4, 6), torch.randn(4, 6)
    w = torch.tensor([1., 1., 1., 0., 0., 0.])
    moved = t.clone()
    moved[:, 3:] += 5
    assert torch.isclose(weighted_mse(p, t, w), weighted_mse(p, moved, w))
    assert torch.isclose(weighted_mse(p, t, w), (p - t)[:, :3].square().mean())
    assert torch.isclose(weighted_mse(p, t, torch.ones(6)), torch.nn.functional.mse_loss(p, t))


def test_every_row_has_its_part_groups_scale():
    # Defect caught: one pooled scale (the index repair found a 5.75 cm pooled action scale against finger targets
    # 0.85 cm away); or a row given another group's scale.
    body = row_options(config())["body"]
    ids = list(range(26))
    got = row_scales(NAMES, ids, unit_scales(), "action", None, body)
    order = sorted(PART_GROUPS)
    for k, i in enumerate(ids):
        assert torch.all(got[k] == 1. + order.index(GROUP_OF[NAMES[i]])), NAMES[i]
    ex = _example(5)
    tree = full_body_tree(NAMES)
    b = make_batch(tree, [ex], scales=unit_scales(), duration_s=.05, action_contract=CONTRACT,
                   hand_frame="actual_wrist", digit_frame="wrist", body=body)
    raw = action_rows(ex.start, executed_rows(ex.executed, ex.start, NAMES, "actual_wrist", body),
                      list(tree.owned["foot_L"]), NAMES, "wrist", body)
    assert torch.allclose(b.action["foot_L"][0] * (1. + order.index("leg")), raw.flatten(), atol=1e-5)


# ------------------------------------------------------------------ round trips
def _round_trip(contract, template_shift=3.):
    cfg = config(contract)
    rows = row_options(cfg)
    body = rows["body"]
    tree = full_body_tree(NAMES)
    ex = _example(7, contract)
    if "parent_relative_calf_v1" in contract:
        # the calf row as the world records it: a goal relative to the thigh's ORDERED row
        for c, p in (("calf_L", "thigh_L"), ("calf_R", "thigh_R")):
            ci, pi = NAMES.index(c), NAMES.index(p)
            rel = R.from_quat(ex.start.bones[pi, :4].double().numpy()).inv() * R.from_quat(ex.executed[ci, 3:7].double().numpy())
            ex.executed[ci, 3:7] = torch.tensor((R.from_quat(ex.executed[pi, 3:7].double().numpy()) * rel).as_quat(),
                                                dtype=torch.float32)
    executed = executed_rows(ex.executed, ex.start, NAMES, cfg.get("hand_action_frame", "recorded"), body)
    predictions = {}
    for node in tree.nodes:
        own = list(tree.owned[node.name])
        a = action_rows(ex.start, executed, own, NAMES, rows["digit_frame"], body)
        a = a / row_scales(NAMES, own, cfg["scales"], "action", None, body)
        predictions[node.name] = LocalPrediction(None, None, a.flatten().unsqueeze(0), {})
    template = ex.executed.clone()
    template[:, :3] += template_shift
    out = handle_targets(tree, predictions, ex.start, template, cfg["scales"],
                         cfg.get("hand_action_frame", "recorded"), rows["digit_frame"], None, body)
    return ex, template, out, body


@pytest.mark.parametrize("contract", [CONTRACT, CONTRACT + "+parent_relative_calf_v1",
                                      CONTRACT + "+no_head_pelvis_position_v1"])
def test_handle_targets_reissue_the_executed_command(contract):
    # Defects caught: the eye row issued in the physical convention or its turn rotated by the head; the calves left on
    # the actual thigh under the calf contract; a digit not placed back on the issued wrist; an unread position issued.
    ex, template, out, body = _round_trip(contract)
    for i, name in enumerate(NAMES):
        angle = rotation_delta(ex.executed[i, 3:7], out[i, 3:7]).norm()
        assert angle < 1e-4, (name, float(angle))
        if name in body.position_read:
            assert torch.allclose(out[i, :3], ex.executed[i, :3], atol=1e-3), name
        else:
            assert torch.equal(out[i, :3], template[i, :3]), name          # no new authority over unread numbers
        assert torch.equal(out[i, 7:9], template[i, 7:9])


def test_a_planted_eye_defect_breaks_the_round_trip(monkeypatch):
    import lem.latent_body.trajectory as T
    monkeypatch.setattr(T, "eye_row_physical", lambda q, convention: T.quaternion(q))   # no conversion anywhere
    ex, template, out, body = _round_trip(CONTRACT)
    gi = NAMES.index("gaze")
    # without the conversion the round trip still closes, but the learned turn is not the eye's turn
    a = T.action_rows(ex.start, ex.executed, [gi], NAMES, "wrist", body)[0, 3:6]
    monkeypatch.undo()
    b = action_rows(ex.start, ex.executed, [gi], NAMES, "wrist", body)[0, 3:6]
    assert (a - b).norm() > 0.1


# ------------------------------------------------------------------ predictions by owner
def test_with_own_rows_each_handle_is_predicted_by_its_owner():
    # Defect caught: scoring or testing a part on the descendant rows its own lesson no longer grades.
    tree = full_body_tree(NAMES)
    body = row_options(config())["body"]
    preds = {n.name: LocalPrediction(torch.full((1, len(tree.subtree[n.name]) * 6), float(k)), None, None, {})
             for k, n in enumerate(tree.nodes)}
    rows = predicted_rows(tree, preds, "trunk", body)[0]
    k_of = {n.name: k for k, n in enumerate(tree.nodes)}
    for r, i in enumerate(tree.subtree["trunk"]):
        assert rows[r].eq(float(k_of[tree.owners[NAMES[i]]])).all()
    subtree = row_options(dict(config(), body_rows=dict(config()["body_rows"], future_rows="subtree")))["body"]
    assert predicted_rows(tree, preds, "trunk", subtree)[0].eq(float(k_of["trunk"])).all()


# ------------------------------------------------------------------ learning end to end
def test_the_whole_body_learns_on_cpu_without_touching_global_randomness():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    learner = ShadowLearner(full_body_tree(NAMES), config())
    record = learner.learn([_example(k) for k in range(4)])
    assert torch.equal(torch.rand(3), expected)
    assert record["updates"] == 1 and all(np.isfinite(v) for v in record["loss_before_update"].values())


# ------------------------------------------------------------------ contacts and the trainer path
def test_the_full_body_shadow_observes_every_skin_patch(tmp_path):
    from mimo_bridge.latent_body_runtime import ShadowRuntime, observation
    from mimo_bridge.latent_body_shadow import add_arguments
    contact = [0] * len(PATCH_NAMES)
    for name in ("left_toes", "hip", "head", "right_lower_leg"):
        contact[PATCH_NAMES.index(name)] = 1
    value = _obs(1, 0)
    frame = SimpleNamespace(env_index=0, sim_time=value.sim_time, bones=value.bones.numpy().copy(),
                            meta={"task": {"contact_patch_state": contact}})
    full = observation(frame, full_body_tree(NAMES), 1, 0)
    on = {NAMES[i] for i in torch.nonzero(full.contact).flatten().tolist()}
    assert on == {"foot_L", "lower_trunk", "head", "calf_R"}
    assert observation(frame, arm_hand_tree(NAMES), 1, 0).contact.sum() == 0      # the pilot keeps its meaning
    # the eye convention of the trainer must be the hierarchy's
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["--latent-body-levels", "full_body"])
    args.latent_body_hierarchy, args.latent_body_config = "shadow", str(tmp_path / "config.json")
    args.latent_body_output, args.out, args.eval_only, args.report_every = str(tmp_path / "shadow"), str(tmp_path / "p"), False, 1
    (tmp_path / "config.json").write_text(json.dumps(config()), encoding="utf-8")
    args.eye_row_convention = "physical"
    with pytest.raises(ValueError, match="eye row convention"):
        ShadowRuntime(args, NAMES)
    args.eye_row_convention = "legacy"
    runtime = ShadowRuntime(args, NAMES)
    assert set(runtime.tree.owners) == set(NAMES)


@pytest.mark.parametrize("whole", [True, False])
def test_a_batch_holds_the_same_rows_as_its_examples_one_at_a_time(whole):
    # Defect caught: a row function that indexes the example dimension instead of the handle dimension once examples
    # are stacked (make_batch computes every example at once since Oct 1, 2026).
    cfg = config() if whole else {k: v for k, v in config().items() if k != "body_rows"}
    tree = full_body_tree(NAMES) if whole else arm_hand_tree(NAMES)
    learner = ShadowLearner(tree, cfg)
    examples = [_example(k) for k in range(5)]
    together = learner.batch(examples)
    for k, ex in enumerate(examples):
        alone = learner.batch([ex])
        for field in ("state", "history", "action", "change", "contact"):
            for name, value in getattr(alone, field).items():
                assert torch.allclose(getattr(together, field)[name][k], value[0], atol=1e-5), (field, name)


def _turn_about(ex, pivot_handle, moved, rotvec):
    """The end observation = the start with `moved` handles turned rigidly about pivot_handle's origin."""
    turn = R.from_rotvec(rotvec)
    end = ex.end
    end.bones = ex.start.bones.clone()
    pivot = ex.start.bones[NAMES.index(pivot_handle), 4:7].double().numpy()
    for name in moved:
        i = NAMES.index(name)
        end.bones[i, 4:7] = torch.tensor(turn.apply(ex.start.bones[i, 4:7].double().numpy() - pivot) + pivot,
                                         dtype=torch.float32)
        end.bones[i, :4] = torch.tensor((turn * R.from_quat(ex.start.bones[i, :4].double().numpy())).as_quat(),
                                        dtype=torch.float32)
    return end


def test_a_leg_turned_at_the_hip_moves_calf_and_foot_only_by_carry():
    # Defects caught: a carried row's displacement graded (the foot code then learns the calf's movement: R^2 0.49-0.55
    # in a small fit with world rows); a carried row's orientation made parent-relative (its command is a world
    # orientation); the world end position of a carried chain (thigh -> calf -> foot) not rebuilt through its parents.
    body = row_options(config())["body"]
    ex = _example(11)
    leg = ["thigh_L", "calf_L", "foot_L"]
    end = _turn_about(ex, "thigh_L", leg, [0.25, -0.1, 0.2])
    ids = [NAMES.index(n) for n in leg]
    rows = change_rows(ex.start, end, ids, NAMES, "wrist", body)
    world = change_rows(ex.start, end, ids, NAMES, "wrist")
    assert rows[1:, :3].abs().max() < 1e-4                         # calf and foot: no displacement of their own
    assert world[1:, :3].norm(dim=-1).min() > 0.1                   # the world saw them carried
    assert torch.allclose(rows[:, 3:], world[:, 3:], atol=1e-5)     # their turns stay world turns
    ends = end_positions(ex.start, ids, rows.unsqueeze(0), NAMES, "wrist", body)[0]
    assert torch.allclose(ends, end.bones[ids, 4:7], atol=2e-3)
    w = body_lesson_weights(full_body_tree(NAMES), body)
    for part in ("calf_L", "foot_L", "chest", "arm_L"):
        change = w[part][0].reshape(-1, 6)
        tree = full_body_tree(NAMES)
        k = list(tree.subtree[part]).index(NAMES.index({"arm_L": "upperarm_L"}.get(part, part)))
        assert change[k, :3].sum() == 0 and change[k, 3:].eq(1).all(), part
    assert "lowerarm_L" not in body.positionless and "thigh_L" not in body.positionless   # moved by their own part


def test_a_shoulder_turn_carries_the_arm_through_the_chest_chain():
    # Defect caught: end positions of nested carried rows (lower_trunk -> chest -> upper arm) built from the wrong
    # parent pose.
    body = row_options(config())["body"]
    ex = _example(12)
    moved = ["chest", "upperarm_R", "lowerarm_R", "wrist_R"]
    end = _turn_about(ex, "lower_trunk", moved, [0.0, 0.15, -0.1])
    ids = [NAMES.index(n) for n in ["lower_trunk"] + moved]
    rows = change_rows(ex.start, end, ids, NAMES, "wrist", body)
    ends = end_positions(ex.start, ids, rows.unsqueeze(0), NAMES, "wrist", body)[0]
    assert torch.allclose(ends, end.bones[ids, 4:7], atol=2e-3)
