# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy

import pytest
import torch

from lem.latent_body.trajectory import TrajectoryExtractor, make_batch, rotation_delta
from mimo_bridge.tests.latent_body_fixtures import config, example, observation, order, tree


def extractor():
    return TrajectoryExtractor(tree(), duration_s=.05, action_contract="world_v0")


def test_extractor_uses_the_executed_action_of_the_completed_interval():
    ex = extractor()
    assert ex.observe(observation(0)) is None
    assert ex.observe(observation(1), order(observation(0)), action_contract="world_v0", frame_t=0) is None
    executed = order(observation(1))
    result = ex.observe(observation(2), executed, action_contract="world_v0", frame_t=1)
    executed.fill_(99)
    assert result.start.frame == 1 and result.end.frame == 2
    assert torch.equal(result.executed, order(observation(1)))
    batch = make_batch(tree(), [result], scales=config()["scales"], duration_s=.05, action_contract="world_v0")
    assert batch.change["index_L"][0, 0] == pytest.approx(.1)
    assert batch.action["index_L"][0, 0] == pytest.approx(.2)


@pytest.mark.parametrize("change", ["episode", "gap", "time"])
def test_no_examples_span_a_reset_gap_or_wrong_horizon(change):
    ex = extractor()
    ex.observe(observation(0)); ex.observe(observation(1))
    obs = observation(2)
    if change == "episode": obs.episode_id = 1
    if change == "gap": obs.frame = 10
    if change == "time": obs.sim_time += 1
    assert ex.observe(obs, order(obs), action_contract="world_v0", frame_t=1) is None
    assert ex.dropped == 1


def test_command_contract_or_wrong_start_is_refused():
    ex = extractor()
    ex.observe(observation(0)); ex.observe(observation(1))
    with pytest.raises(ValueError, match="contract"):
        ex.observe(observation(2), order(observation(1)), action_contract="another", frame_t=1)
    with pytest.raises(ValueError, match="aligned"):
        ex.observe(observation(2), order(observation(1)), action_contract="world_v0", frame_t=0)


def test_bodies_and_saved_history_are_isolated():
    ex = extractor()
    for step in range(2):
        for body in range(2): ex.observe(observation(step, body))
    saved = copy.deepcopy(ex.state_dict())
    restored = extractor(); restored.load_state_dict(saved)
    restored.previous[0][0].bones.fill_(42)
    assert ex.previous[0][0].bones[0, 3] == 1
    assert restored.previous[1][0].bones[0, 3] == 1
    assert saved["previous"][0][0]["bones"][0, 3] == 1


def test_opposite_quaternions_are_the_same_motion_and_learning_inputs_are_detached():
    q = torch.tensor([[0., 0., 0., 1.], [1., 0., 0., 0.]])
    assert torch.equal(rotation_delta(q, -q), torch.zeros(2, 3))
    ex = example()
    ex.executed.requires_grad_()
    # A recorded example owns its physical data, even if the caller had a graph.
    from lem.latent_body.trajectory import Example
    ex = Example(ex.history, ex.start, ex.end, ex.executed, ex.action_contract)
    batch = make_batch(tree(), [ex], scales=config()["scales"], duration_s=.05, action_contract="world_v0")
    # (the whole-body lesson weights are None without body_rows)
    assert not any(x.requires_grad for values in vars(batch).values() if values for x in values.values())


@pytest.mark.parametrize("field", ["bones", "contact", "executed"])
def test_nonfinite_physical_data_is_refused(field):
    ex = example()
    target = ex.executed if field == "executed" else getattr(ex.start, field)
    target.flatten()[0] = float("nan")
    with pytest.raises(ValueError):
        make_batch(tree(), [ex], scales=config()["scales"], duration_s=.05, action_contract="world_v0")
