# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The visual correction is part of the prediction, including for regularization."""
from types import SimpleNamespace

import torch

from lem.training.latent_regularizer import LatentRegularizer
from mimo_bridge.lem_model import LemLite


def test_canceling_visual_components_are_not_penalized_as_a_bad_prediction():
    reg = LatentRegularizer(d_model=4)
    model = SimpleNamespace(latent_regularizer=reg, next_frame_loss=True)
    base = torch.full((2, 4), 10., requires_grad=True)
    correction = torch.full((2, 4), -9., requires_grad=True)
    actual = base + correction
    loss = LemLite._latent_prediction_penalty(
        model, {'vision': base}, ({}, {'vision': actual}), {'vision': 'real'})
    assert reg.penalty({'vision': base}) > 60
    assert loss == 0
    loss.backward()
    assert torch.equal(base.grad, torch.zeros_like(base))
    assert torch.equal(correction.grad, torch.zeros_like(correction))


def test_outlying_complete_prediction_trains_both_paths_and_excludes_noise():
    reg = LatentRegularizer(d_model=4)
    model = SimpleNamespace(latent_regularizer=reg, next_frame_loss=True)
    base = torch.full((2, 4), 10., requires_grad=True)
    correction = torch.full((2, 4), -7., requires_grad=True)
    noise = torch.full((2, 4), 100., requires_grad=True)
    loss = LemLite._latent_prediction_penalty(model, {'vision': base},
        ({}, {'vision': base + correction, 'audio': noise}),
        {'vision': 'real', 'audio': 'synthetic'})
    assert torch.isclose(loss, torch.tensor(1.))
    loss.backward()
    assert torch.equal(base.grad, correction.grad)
    assert bool((correction.grad > 0).all())
    assert noise.grad is None


def test_first_frame_skips_a_missing_transition_but_legacy_path_is_preserved():
    reg = LatentRegularizer(d_model=4)
    model = SimpleNamespace(latent_regularizer=reg, next_frame_loss=True)
    predicted = {'vision': torch.full((2, 4), 4., requires_grad=True)}
    assert LemLite._latent_prediction_penalty(model, predicted, None, {'vision': 'real'}) is None
    model.next_frame_loss = False
    assert LemLite._latent_prediction_penalty(model, predicted, None, {'vision': 'real'}) == reg.penalty(predicted)
