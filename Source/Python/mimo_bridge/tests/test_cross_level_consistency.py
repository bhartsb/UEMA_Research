# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy

import torch

from lem.latent_body.learning import ShadowLearner
from lem.latent_body.losses import cross_level_consistency, dynamics_loss
from mimo_bridge.tests.latent_body_fixtures import config, example, tree


def test_parent_child_disagreement_is_measured_on_shared_physical_channels():
    learner = ShadowLearner(tree(), config())
    batch = learner.batch([example()])
    z = {root: torch.zeros(1, 2) for root in learner.tree.roots}
    predictions, _ = learner.model.decode(batch.state, batch.history, z)
    for pred in predictions.values():
        pred.change = torch.zeros_like(pred.change)
        pred.contact_logits = torch.zeros_like(pred.contact_logits)
    assert float(cross_level_consistency(learner.tree, predictions)) == 0
    predictions["index_L"].change += 1
    assert float(cross_level_consistency(learner.tree, predictions)) > 0


def test_child_prediction_error_reaches_parent_intention_emitter():
    learner = ShadowLearner(tree(), config())
    batch = learner.batch([example()])
    roots = {root: torch.zeros(1, 2) for root in learner.tree.roots}
    predictions, _ = learner.model.decode(batch.state, batch.history, roots)
    (predictions["index_L"].change - batch.change["index_L"]).square().mean().backward()
    assert learner.model.parts["arm_L"].decoder[-1].weight.grad.abs().sum() > 0
    assert learner.model.parts["hand_L"].decoder[-1].weight.grad.abs().sum() > 0


def test_hindsight_lesson_grades_future_action_and_contact_without_wellness():
    learner = ShadowLearner(tree(), config())
    batch = learner.batch([example(), example(2)])
    loss, terms, _ = dynamics_loss(learner.model, batch, learner.config["weights"])
    assert set(terms) == {"future", "contact", "action", "child", "consistency", "latent"}
    loss.backward()
    assert all(learner.model.parts[name].encoder[-1].weight.grad is not None for name in learner.model.parts)
    changed = copy.deepcopy(batch)
    changed.change["index_L"] += 1
    other, _, _ = dynamics_loss(learner.model, changed, learner.config["weights"])
    assert float(other) != float(loss)
