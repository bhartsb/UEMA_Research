# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A readable runtime file must not make partial restoration look successful."""
from pathlib import Path

import pytest
import torch

from mimo_bridge.lem_model import LemLite


@pytest.fixture
def snapshot(tmp_path):
    torch.set_num_threads(1)
    model = LemLite(device='cpu', imagination=False, no_vision=True,
                    checkpoint_dir=str(tmp_path / 'checkpoints'))
    model._vis_adv_ema = 0.375
    model._nonfinite_steps = 17
    path = Path(model.save_checkpoint(7))
    return model, path


@pytest.mark.parametrize('field', ['vis_adv_ema', 'nonfinite_steps'])
def test_readable_but_invalid_runtime_refuses_resume(snapshot, field):
    model, path = snapshot
    state = torch.load(path / 'runtime_state.pt', weights_only=False, map_location='cpu')
    state[field] = 'not a number'
    torch.save(state, path / 'runtime_state.pt')
    with pytest.raises(RuntimeError, match='could not be fully restored'):
        model.load_checkpoint(path)


def test_valid_runtime_restores_early_and_later_fields(snapshot):
    model, path = snapshot
    model._vis_adv_ema = -4.0
    model._nonfinite_steps = 0
    result = model.load_checkpoint(path)
    assert result['step'] == 7
    assert model._vis_adv_ema == 0.375
    assert model._nonfinite_steps == 17
