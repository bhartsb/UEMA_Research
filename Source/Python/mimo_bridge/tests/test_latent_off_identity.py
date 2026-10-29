# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import argparse
import builtins
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.latent_body_shadow import add_arguments, observe_shadow, validate_arguments
from mimo_bridge.tests.latent_body_fixtures import config, example, tree


def test_off_hook_does_not_import_allocate_read_or_mutate(monkeypatch):
    class Forbidden:
        def __getattr__(self, key): raise AssertionError(key)
    args = SimpleNamespace(latent_body_hierarchy="off")
    def forbidden(*args, **kwargs): raise AssertionError("OFF path performed I/O/import")
    with monkeypatch.context() as patch:
        # __import__ last: monkeypatch.setattr itself imports inspect on first use
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(builtins, "__import__", forbidden)
        assert observe_shadow(args, Forbidden(), Forbidden(), Forbidden(), 0, 0) is None


def test_flags_default_off_and_never_enable_production_control():
    parser = argparse.ArgumentParser(); add_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.latent_body_hierarchy == "off" and defaults.latent_planner == "off"
    validate_arguments(defaults)
    for flags in (("--latent-body-hierarchy", "control"), ("--latent-planner", "control"),
                  ("--latent-planner", "clone"), ("--latent-planner", "shadow")):
        with pytest.raises(ValueError): validate_arguments(parser.parse_args(flags))


def test_shadow_optimizer_preserves_existing_parameters_optimizer_and_random_streams():
    from lem.latent_body.learning import ShadowLearner
    from mimo_bridge import copied_execution as CE
    rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    learner = ShadowLearner(tree(), config())
    assert torch.equal(rng, torch.get_rng_state())
    assert py_rng == random.getstate()
    assert CE.equal_tree(np_rng, np.random.get_state())
    # A planted leak into the global RNG or an existing optimizer changes the next
    # production update. Each branch runs with its own continuing random stream.
    torch.random.default_generator.manual_seed(19)
    baseline = torch.nn.Linear(3, 2)
    candidate = copy.deepcopy(baseline)
    optimizers = [torch.optim.Adam(m.parameters()) for m in (baseline, candidate)]
    streams = [torch.get_rng_state().clone() for _ in range(2)]
    changed = copy.deepcopy(learner.model.state_dict())
    for step in range(4):
        outputs = []
        for index, model in enumerate((baseline, candidate)):
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(streams[index])
                optimizers[index].zero_grad()
                y = model(torch.randn(4, 3))
                y.square().mean().backward(); optimizers[index].step()
                if index: learner.learn([example(step + 1)])
                streams[index] = torch.get_rng_state().clone()
                outputs.append(y.detach())
        assert torch.equal(streams[0], streams[1]) and torch.equal(*outputs)
        assert CE.equal_tree(baseline.state_dict(), candidate.state_dict())
        assert CE.equal_tree(optimizers[0].state_dict(), optimizers[1].state_dict())
    assert any(not torch.equal(v, learner.model.state_dict()[k]) for k, v in changed.items())


def test_shadow_options_do_not_invalidate_existing_acting_configuration():
    from mimo_bridge.train_mimo import acting_configuration
    old = SimpleNamespace(n_envs=8, intention="on")
    new = SimpleNamespace(**vars(old), latent_body_hierarchy="off", latent_planner="off",
        latent_body_levels="arm,hand,index", latent_body_config="", latent_body_output="",
        latent_body_resume="", latent_body_goal="", latent_body_candidates=None, latent_body_radius=None)
    assert acting_configuration(old) == acting_configuration(new)


def test_identity_comparison_detects_added_optimizer_state_and_record_lengths(tmp_path):
    from mimo_bridge.tools.latent_body_identity import same, statistics
    baseline = {"state": {0: {"exp_avg": torch.zeros(3)}}}
    changed = copy.deepcopy(baseline); changed["state"][1] = {"exp_avg": torch.zeros(3)}
    assert not same(baseline, changed)
    assert not same([1, 2], [1, 2, 3])
    path = tmp_path / "stats.jsonl"
    path.write_text('{"timing_ms": 1, "predicted_time": 2}\n', encoding="utf-8")
    assert statistics(path, ["timing_ms"]) == [{"predicted_time": 2}]
