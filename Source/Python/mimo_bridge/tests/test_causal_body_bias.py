# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Causal forecast calibration cannot read a current or future outcome."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_probe():
    directory = Path(__file__).resolve().parents[4] / 'docs' / 'gain_of_function_v3'
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location('private_causal_bias_probe', directory/'probe_causal_body_bias.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))


def prior_order():
    return dict(step=106, order_began=100,
        achieved_cm=torch.full((1, 26, 3), 0.5), expected_cm=torch.full((1, 26, 3), 0.2),
        achieved_deg=torch.full((1, 26, 3), 5.0), expected_deg=torch.full((1, 26, 3), 3.0))


def test_correction_uses_a_completed_past_order_and_copies_its_readings():
    probe = load_probe()
    record = prior_order()
    model = SimpleNamespace(step_count=111, efference_tenure_steps=5, last_efference_tenure=record)
    correction, receipt = probe.completed_residual(model)
    assert receipt['age_steps'] == 5 and receipt['order_began'] == 100
    torch.testing.assert_close(correction['position_cm'], torch.full((1, 26, 3), .3))
    torch.testing.assert_close(correction['rotation_deg'], torch.full((1, 26, 3), 2.0))
    correction['position_cm'].zero_()
    assert record['achieved_cm'].eq(.5).all() and record['expected_cm'].eq(.2).all()


@pytest.mark.parametrize('changes', [
    {'step_count': 105},  # the recorded outcome lies in the future
    {'step_count': 114},  # residual is too old for the declared test
    {'efference_tenure_steps': 4},  # incompatible prediction horizon
])
def test_calibration_rejects_future_stale_or_wrong_horizon_data(changes):
    probe = load_probe()
    values = dict(step_count=111, efference_tenure_steps=5, last_efference_tenure=prior_order())
    values.update(changes)
    with pytest.raises(AssertionError):
        probe.completed_residual(SimpleNamespace(**values))
