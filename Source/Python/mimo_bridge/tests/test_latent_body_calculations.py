# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Calculations of the latent body hierarchy against independent references (review of Sep 30, 2026).

Each test was checked by planting the defect it names: before these tests, a body-frame rotation delta, a flipped
rotation sign, a reversed target composition, a contact target read from the start, a chain anchor on the wrong
handle, a consistency loss on the wrong handles and a flipped measured change all passed the latent suite.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from lem.latent_body.hierarchy import CHANGE_DIM
from lem.latent_body.losses import cross_level_consistency
from lem.latent_body.model import LocalPrediction
from lem.latent_body.trajectory import make_batch, rotation_delta
from lem.planner.latent_rollout import handle_targets
from mimo_bridge.tests.latent_body_fixtures import config, example, observation, tree


def _unit(rows, seed):
    return torch.nn.functional.normalize(torch.randn(rows, 4, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_rotation_delta_is_the_world_frame_turn_from_start_to_end():
    start, end = _unit(64, 0), _unit(64, 1)
    expected = (R.from_quat(end.double().numpy()) * R.from_quat(start.double().numpy()).inv()).as_rotvec()
    assert np.allclose(rotation_delta(start, end).numpy(), expected, atol=1e-4)


def test_handle_targets_reissue_the_executed_command_under_measured_scales():
    t, ex = tree(), example()
    ex.start.bones[:, :4] = _unit(26, 2)
    ex.executed[:, 3:7] = _unit(26, 3)
    ex.executed[:, :3] += torch.randn(26, 3, generator=torch.Generator().manual_seed(4)) * 3
    scales = dict(config()["scales"], action=[2., 3., 4., .5, .7, .9, 1., 1.])
    batch = make_batch(t, [ex], scales=scales, duration_s=.05, action_contract="world_v0")
    predictions = {n.name: LocalPrediction(batch.change[n.name], batch.contact[n.name], batch.action[n.name], {})
                   for n in t.nodes}
    template = ex.executed.clone()
    template[:, :3] = 0
    template[:, 3:7] = torch.tensor([0., 0., 0., 1.])
    out = handle_targets(t, predictions, ex.start, template, scales)
    owned = sorted(i for ids in t.owned.values() for i in ids)
    assert torch.allclose(out[owned, :3], ex.executed[owned, :3], atol=1e-4)
    turn = R.from_quat(out[owned, 3:7].double().numpy()) * R.from_quat(ex.executed[owned, 3:7].double().numpy()).inv()
    assert turn.magnitude().max() < 1e-4


def test_contact_target_is_the_outcome_contact():
    t, ex = tree(), example()
    assert not torch.equal(ex.start.contact, ex.end.contact)
    batch = make_batch(t, [ex], scales=config()["scales"], duration_s=.05, action_contract="world_v0")
    for n in t.nodes:
        assert torch.equal(batch.contact[n.name][0], ex.end.contact[list(t.subtree[n.name])])


def test_every_part_of_a_chain_is_placed_relative_to_its_upper_arm():
    t, ex = tree(), example()
    batch = make_batch(t, [ex], scales=config()["scales"], duration_s=.05, action_contract="world_v0")
    for side in "LR":
        anchor = ex.start.bones[t.handle_names.index("upperarm_" + side), 4:7]
        for name in ("arm_", "hand_", "index_"):
            ids = list(t.subtree[name + side])
            rows = batch.state[name + side].reshape(len(ids), 12)
            assert torch.allclose(rows[:, :3], ex.start.bones[ids, 4:7] - anchor)


def test_consistency_is_zero_exactly_when_parent_and_child_agree_on_shared_handles():
    t = tree()
    per_handle = torch.randn(1, 26, CHANGE_DIM, generator=torch.Generator().manual_seed(5))
    logits = torch.randn(1, 26, generator=torch.Generator().manual_seed(6))
    agree = {n.name: LocalPrediction(per_handle[:, list(t.subtree[n.name])].flatten(1),
                                     logits[:, list(t.subtree[n.name])], torch.zeros(1, 0), {}) for n in t.nodes}
    assert float(cross_level_consistency(t, agree)) < 1e-12
    agree["index_L"].change = agree["index_L"].change + 1
    assert float(cross_level_consistency(t, agree)) > 0.1


def test_measured_change_has_the_direction_of_the_movement():
    from mimo_bridge.tools.latent_body_intervention import actual_change
    t = tree()
    before, after = observation(0), observation(0)
    i = t.handle_names.index("wrist_L")
    after.bones[i, 4:7] += torch.tensor([1., -2., .5])
    after.bones[i, :4] = torch.tensor(R.from_rotvec([0., 0., .3]).as_quat(), dtype=torch.float32)
    change = actual_change(t, before, after, [i])
    assert torch.allclose(change, torch.tensor([[1., -2., .5, 0., 0., .3]]), atol=1e-5)


def _wrist_relative_example():
    ex = example()
    ex.action_contract = "wrist_relative_joint_v1"
    ex.start.bones[:, :4] = _unit(26, 7)
    ex.executed[:, 3:7] = _unit(26, 8)
    ex.executed[:, :3] += torch.randn(26, 3, generator=torch.Generator().manual_seed(9)) * 5
    return ex


def test_finger_rows_move_with_the_wrist_like_a_rigid_hand():
    from lem.latent_body.trajectory import actual_wrist_frame
    t, ex = tree(), _wrist_relative_example()
    moved = actual_wrist_frame(ex.executed, ex.start, t.handle_names)
    for side in "LR":
        w = t.handle_names.index("wrist_" + side)
        asked = R.from_quat(ex.executed[w, 3:7].double().numpy())
        actual = R.from_quat(ex.start.bones[w, :4].double().numpy())
        for digit in ("thumb", "index", "pinky"):
            i = t.handle_names.index(f"{digit}_{side}")
            want = actual.apply(asked.inv().apply((ex.executed[i, :3] - ex.executed[w, :3]).double().numpy()))
            assert np.allclose(moved[i, :3].numpy(), want + ex.start.bones[w, 4:7].numpy(), atol=1e-3)
            turn = (actual * asked.inv() * R.from_quat(ex.executed[i, 3:7].double().numpy())
                    * R.from_quat(moved[i, 3:7].double().numpy()).inv())
            assert turn.magnitude() < 1e-4
    others = [i for i, n in enumerate(t.handle_names) if n.split("_")[0] not in ("thumb", "index", "middle", "ring", "pinky")]
    assert torch.equal(moved[others], ex.executed[others])


def test_actual_wrist_actions_reissue_the_recorded_command():
    t, ex = tree(), _wrist_relative_example()
    batch = make_batch(t, [ex], scales=config()["scales"], duration_s=.05,
                       action_contract="wrist_relative_joint_v1", hand_frame="actual_wrist")
    predictions = {n.name: LocalPrediction(batch.change[n.name], batch.contact[n.name], batch.action[n.name], {})
                   for n in t.nodes}
    template = ex.executed.clone()
    template[:, :3] = 0
    template[:, 3:7] = torch.tensor([0., 0., 0., 1.])
    out = handle_targets(t, predictions, ex.start, template, config()["scales"], "actual_wrist")
    owned = sorted(i for ids in t.owned.values() for i in ids)
    assert torch.allclose(out[owned, :3], ex.executed[owned, :3], atol=1e-3)
    turn = R.from_quat(out[owned, 3:7].double().numpy()) * R.from_quat(ex.executed[owned, 3:7].double().numpy()).inv()
    assert turn.magnitude().max() < 1e-4


def test_switches_bound_codes_and_leave_older_configurations_unchanged():
    import pytest
    from lem.latent_body.learning import ShadowLearner
    from lem.latent_body.checkpoint import fingerprint
    plain = ShadowLearner(tree(), config())
    assert "code_bound" not in plain.config and fingerprint(plain.config) == fingerprint(config())
    bounded = ShadowLearner(tree(), dict(config(), code_bound="tanh"))
    batch = bounded.batch([example()])
    with torch.no_grad():
        for part in bounded.model.parts.values():
            part.encoder[-1].bias.fill_(50.)
        assert all(z.abs().max() <= 1 for z in bounded.model.encode(batch).values())
        for part in plain.model.parts.values():
            part.encoder[-1].bias.fill_(50.)
        assert all(z.abs().max() > 1 for z in plain.model.encode(plain.batch([example()])).values())
    with pytest.raises(ValueError, match="wrist_relative"):
        ShadowLearner(tree(), dict(config(), hand_action_frame="actual_wrist"))
    with pytest.raises(ValueError):
        ShadowLearner(tree(), dict(config(), unknown_switch=1))


def test_the_synergy_route_receives_the_command_the_handle_route_issues(monkeypatch):
    import lem.planner.latent_rollout as LR
    from types import SimpleNamespace
    t, ex = tree(), _wrist_relative_example()
    batch = make_batch(t, [ex], scales=config()["scales"], duration_s=.05,
                       action_contract="wrist_relative_joint_v1", hand_frame="actual_wrist")
    predictions = {n.name: LocalPrediction(batch.change[n.name], batch.contact[n.name], batch.action[n.name], {})
                   for n in t.nodes}
    seen = {}
    monkeypatch.setattr(LR, "through_synergy", lambda syn, tr, start, requested, template: seen.setdefault("r", requested))
    choice = SimpleNamespace(predictions=predictions)
    LR.realize(choice, t, ex.start, ex.executed, config()["scales"], synergy=object(), hand_frame="actual_wrist")
    assert torch.equal(seen["r"], handle_targets(t, predictions, ex.start, ex.executed, config()["scales"], "actual_wrist"))
