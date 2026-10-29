# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import pytest
import torch

from lem.model.contact_wellness import contact_response, current_need_value


def skin(pressures, areas=None):
    x = torch.zeros(1, len(pressures), 8)
    x[0, :, 0] = torch.tensor(pressures)
    x[0, :, 3] = torch.tensor(areas or [1.] * len(pressures))
    return x


def test_gentle_contact_cannot_conceal_full_local_discomfort():
    one = contact_response(skin([1.]))
    mixed = contact_response(skin([1.] + [.05] * 63))
    assert one['pain'].item() == mixed['pain'].item() == 1.
    assert one['comfort'].item() == mixed['comfort'].item() == 0.
    assert mixed['local_pleasant'][0, 1:].min() > 0


def test_subdivision_preserves_response_but_sensitivity_changes_it():
    x = skin([.05, .4], [2., 1.])
    divided = skin([.05] * 10 + [.4], [.2] * 10 + [1.])
    for k in ('comfort', 'pain', 'pleasant'):
        torch.testing.assert_close(contact_response(x)[k], contact_response(divided)[k])
    sens = torch.ones(1, 2, 2)
    sens[0, 1, 0] = 2
    assert contact_response(x, sens)['pain'] > contact_response(x)['pain']
    sens[0, 0, 1] = 1.5
    assert contact_response(x, sens)['pleasant'] > contact_response(x)['pleasant']


def test_empty_padding_and_order_cannot_change_feelings():
    x = skin([.05, .25])
    padded = torch.cat([x, torch.zeros(1, 400, 8)], 1)
    for k in ('comfort', 'pain', 'pleasant'):
        torch.testing.assert_close(contact_response(x)[k], contact_response(padded)[k])
        torch.testing.assert_close(contact_response(x)[k], contact_response(x.flip(1))[k])
    assert contact_response(skin([0.]))['comfort'].item() == .5


def test_bad_sensor_values_are_not_passes():
    x = skin([float('nan')])
    with pytest.raises(ValueError, match='nonfinite'):
        contact_response(x)


def test_known_bad_contact_never_becomes_good_from_prediction_success():
    args = dict(pain=torch.ones(1, 1), pleasant=torch.ones(1, 1),
                capacity=torch.ones(1, 10), current_capacity=torch.zeros(1, 10),
                curiosity_satisfaction=torch.ones(1, 1), curiosity_need=torch.ones(1, 1))
    assert current_need_value(**args).item() == -1
    args['pain'] = torch.zeros(1, 1)
    assert current_need_value(**args).item() == 1


def test_recovery_is_more_valuable_when_tired_and_gradients_reach_contact():
    zero, one = torch.zeros(1, 1), torch.ones(1, 1)
    tired = current_need_value(zero, zero, one, zero)
    fresh = current_need_value(zero, zero, one, one)
    assert tired > fresh
    x = skin([.4]).requires_grad_()
    contact_response(x)['comfort'].sum().backward()
    assert x.grad[0, 0, 0] < 0
