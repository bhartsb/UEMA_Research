# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy

import pytest
import torch

from lem.latent_body.learning import ShadowLearner
from lem.latent_body.metrics import acceptance, paired_gain, prediction_ranking
from lem.planner.latent_candidates import search
from lem.planner.latent_rollout import realize
from mimo_bridge.tests.latent_body_fixtures import config, example, order, tree


def test_root_intervention_changes_child_intention_and_predicted_consequence():
    learner = ShadowLearner(tree(), config())
    batch = learner.batch([example()])
    roots = {root: torch.zeros(1, 2) for root in learner.tree.roots}
    a, za = learner.model.decode(batch.state, batch.history, roots)
    roots["arm_L"][0, 0] = 1
    b, zb = learner.model.decode(batch.state, batch.history, roots)
    assert not torch.equal(za["hand_L"], zb["hand_L"])
    assert not torch.equal(za["index_L"], zb["index_L"])
    assert not torch.equal(a["index_L"].change, b["index_L"].change)
    # Lateral context carries current state, so another root's intention cannot bypass ownership.
    assert torch.equal(a["arm_R"].change, b["arm_R"].change)


def test_shadow_search_keeps_best_candidate_and_preserves_global_rng_and_weights():
    learner = ShadowLearner(tree(), config())
    batch = learner.batch([example()])
    before = copy.deepcopy(learner.model.state_dict())
    rng = torch.get_rng_state().clone()
    choice = search(learner.model, batch.state, batch.history,
                    lambda p: p["index_L"].change.square().mean(-1),
                    candidates=7, radius=.5, generator=learner.generator)
    assert choice.scores[choice.index] == choice.scores.min()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, learner.model.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in learner.model.parameters())
    output = realize(choice, learner.tree, example().start, order(example().start), config()["scales"])
    original = order(example().start)
    for i, name in enumerate(learner.tree.handle_names):
        if name not in learner.tree.owners:
            assert torch.equal(output[i], original[i])
    assert torch.equal(output[:, 7:9], original[:, 7:9])
    assert torch.allclose(output[:, 3:7].norm(dim=-1), torch.ones(26))


def test_nonfinite_candidate_scores_are_refused():
    learner = ShadowLearner(tree(), config())
    batch = learner.batch([example()])
    with pytest.raises(ValueError, match="finite"):
        search(learner.model, batch.state, batch.history,
               lambda p: torch.full((3,), float("nan")), candidates=3, radius=1., generator=learner.generator)


def test_rank_reversal_and_ties_cannot_pass_as_physical_improvement():
    assert prediction_ranking([1, 2, 3], [3, 2, 1], noise=0)["wrong_pairs"] == 3
    tied = paired_gain([1] * 8, [1] * 8, noise=0)
    assert tied["wins"] == 0 and tied["one_sided_sign_p"] == 1
    wins = paired_gain([1] * 8, [0] * 8, noise=0)
    assert wins["one_sided_sign_p"] == 1 / 256
    verdict = acceptance(quality="plumbing", off_identity=True, comparisons=[wins], alpha=.05,
        held_out_groups=["a", "b"], minimum_groups=2, substrate_valid=True, physical_interventions=True)
    assert not verdict["eligible_for_review"] and not verdict["control_authority"]
    verdict = acceptance(quality="post_repair", off_identity=True, comparisons=[wins], alpha=.05,
        held_out_groups=["a", "b"], minimum_groups=2, substrate_valid=True, physical_interventions=True)
    assert not verdict["eligible_for_review"]  # all eight proposal gates, not only goal improvement


def test_synergy_projection_keeps_the_live_synergy_state_and_nonpilot_commands():
    from mimo_bridge.tests.test_motor_synergy import _syn
    from lem.planner.latent_rollout import through_synergy
    synergy = _syn(k=2)
    before = copy.deepcopy(synergy.state_dict())
    centre = synergy.centre_mean.clone()
    ex = example()
    output = through_synergy(synergy, tree(), ex.start, ex.executed, order(ex.start))
    assert all(torch.equal(v, synergy.state_dict()[k]) for k, v in before.items())
    assert torch.equal(centre, synergy.centre_mean)
    for i, name in enumerate(tree().handle_names):
        if name not in tree().owners:
            assert torch.equal(output[i], order(ex.start)[i])


def test_collapsed_intentions_fail_the_representation_control_without_updating_state():
    from lem.latent_body.evaluation import representation_probe
    from mimo_bridge import copied_execution as CE
    learner = ShadowLearner(tree(), config())
    # Plant the failure the proposal names: a decoder that cannot read intentions.
    with torch.no_grad():
        for part in learner.model.parts.values():
            part.decoder[0].weight[:, -learner.model.latent_dim:].zero_()
    before = learner.state_dict()
    result = representation_probe(learner, [example(), example(2)])
    assert all(value == 0 for value in result["predicted_action_sensitivity"].values())
    assert result["controls"]["posterior"] == result["controls"]["zero"]
    assert CE.equal_tree(before, learner.state_dict())
