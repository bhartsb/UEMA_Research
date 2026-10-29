# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Physical body forecasts cannot replace visual eye aiming."""
from types import SimpleNamespace

import torch

from lem.model.intention_generator import IntentionGenerator
from lem.model.physical_goal import PhysicalTrustRegion
from mimo_bridge.lem_model import LemLite


def fixture():
    modules = [torch.nn.Linear(1, 1) for _ in range(3)]
    config = {'body': {'n_handles': 2}, 'model': {'d_model': 1},
              '_active_modalities': ['vision', 'proprio'],
              'training': {'intention': {'n_steps': 20, 'lr': .1, 'strength_reg': .01}}}
    generator = IntentionGenerator(config, *modules)
    model = SimpleNamespace(intention_gen=generator, GAZE_IDX=1)
    motor = torch.zeros(2, 2, 9)
    motor[..., 6] = 1
    motor[..., 7:] = .5
    current = {'vision': torch.zeros(2, 1), 'proprio': torch.zeros(2, 1)}
    goal = {'vision': torch.full((2, 1), .4), 'proprio': torch.full((2, 1), -20.)}
    decision = {'source_mask': {}, 'context': None, 'achieved': None, 'dev': {}}
    return model, motor, current, goal, decision


def test_eye_stage_grades_visual_goal_and_retains_every_body_field(monkeypatch):
    model, motor, current, goal, decision = fixture()
    def transition(_fm, _enc, _heads, _state, action, **kwargs):
        # Conflicting proprioception must not take over the eye-only stage.
        return SimpleNamespace(tokens={'vision': action[:, 1, 3:4],
                                      'proprio': action[:, 1, 3:4]})
    monkeypatch.setattr('lem.model.predictive_state.transition', transition)
    selected, metadata = LemLite._plan_visual_gaze(
        model, current, goal, motor, decision, None)
    assert (selected[:, 1, 3] > .35).all()
    assert metadata['last_goal_distance'] < metadata['first_goal_distance']
    assert metadata['goal_kind'] == 'visual_memory_gaze_only'
    torch.testing.assert_close(selected[:, 0], motor[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(selected[:, 1, [0, 1, 2, 7, 8]], motor[:, 1, [0, 1, 2, 7, 8]])
    assert not selected.requires_grad

    space = PhysicalTrustRegion(None, selected, ['arm', 'gaze'])
    space.init()
    with torch.no_grad():
        space.rotation[:, 1].add_(10)
        space.strength[:, 1].add_(10)
    candidate, strength, _ = space.assemble()
    torch.testing.assert_close(candidate[:, 1], selected[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(strength, candidate[..., 8:9], rtol=0, atol=0)
    # A spurious body forecast can depend on eyes, but cannot change their choice.
    candidate[:, 1, 3:7].sum().backward()
    assert space.rotation.grad[:, 1].abs().sum() == 0


def test_missing_or_synthetic_visual_goal_retains_gaze_without_running_optimizer():
    model, motor, current, goal, decision = fixture()
    model.intention_gen = SimpleNamespace(generate_action=lambda *a, **kw: (_ for _ in ()).throw(AssertionError()))
    for source in ['noise', 'synthetic']:
        decision['source_mask'] = {'vision': source}
        chosen, metadata = LemLite._plan_visual_gaze(model, current, goal, motor, decision, None)
        assert chosen is motor
        assert metadata['stopped_by'] == 'no_real_visual_goal'
    decision['source_mask'] = {}
    chosen, _ = LemLite._plan_visual_gaze(model, current, {}, motor, decision, None)
    assert chosen is motor
