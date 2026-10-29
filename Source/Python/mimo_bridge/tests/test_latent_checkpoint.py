# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy

import pytest
import torch

from lem.latent_body import checkpoint
from lem.latent_body.learning import ShadowLearner
from mimo_bridge import copied_execution as CE
from mimo_bridge.tests.latent_body_fixtures import config, example, observation, tree


def test_independent_checkpoint_keeps_weights_adam_history_rng_and_next_update(tmp_path):
    original = ShadowLearner(tree(), config())
    original.learn([example()])
    original.extractor.observe(observation(0))
    original.extractor.observe(observation(1))
    path = tmp_path / "hierarchy.pt"
    checkpoint.save(path, original)
    restored = ShadowLearner(tree(), config())
    checkpoint.load(path, restored)
    assert CE.equal_tree(original.state_dict(), restored.state_dict())
    assert torch.equal(torch.rand(3, generator=original.generator), torch.rand(3, generator=restored.generator))
    original.learn([example(2)]); restored.learn([example(2)])
    assert CE.equal_tree(original.state_dict(), restored.state_dict())
    assert list(tmp_path.iterdir()) == [path]


def test_incompatible_configuration_refuses_without_changing_learned_state(tmp_path):
    learner = ShadowLearner(tree(), config())
    before = learner.state_dict()
    bad = copy.deepcopy(before); bad["configuration"] = "wrong"
    with pytest.raises(ValueError, match="configuration"):
        learner.load_state_dict(bad)
    assert CE.equal_tree(before, learner.state_dict())


def test_nonfinite_update_restores_weights_and_adam(monkeypatch):
    learner = ShadowLearner(tree(), config())
    learner.learn([example()])
    before = learner.state_dict()
    real_step = learner.optimizer.step
    def broken_step():
        real_step()
        with torch.no_grad(): next(learner.model.parameters()).fill_(float("inf"))
    monkeypatch.setattr(learner.optimizer, "step", broken_step)
    with pytest.raises(FloatingPointError, match="restored"):
        learner.learn([example(2)])
    assert CE.equal_tree(before, learner.state_dict())


def test_missing_measurements_and_false_clean_data_are_refused():
    for key in ("measurement", "learning_rate", "gradient_clip"):
        cfg = config(); cfg[key] = None
        with pytest.raises((ValueError, TypeError)):
            ShadowLearner(tree(), cfg)
    cfg = config(); cfg["data_quality"] = "post_repair"
    with pytest.raises(ValueError, match="receipt"):
        ShadowLearner(tree(), cfg)


def test_corrupt_adam_checkpoint_is_refused_without_losing_earned_state():
    learner = ShadowLearner(tree(), config())
    learner.learn([example()])
    before = learner.state_dict()
    bad = copy.deepcopy(before)
    next(iter(bad["optimizer"]["state"].values()))["exp_avg_sq"].fill_(float("inf"))
    with pytest.raises(ValueError, match="Non-finite"):
        learner.load_state_dict(bad)
    assert CE.equal_tree(before, learner.state_dict())
