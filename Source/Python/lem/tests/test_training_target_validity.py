# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Next-frame source validity must determine the prediction lesson's gradient."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lem.source_mask import training_source_masks
from lem.training.loss_manager import LossManager


def manager():
    return LossManager({
        "_active_modalities": ["vision", "motor"],
        "_prediction_head_modalities": ["vision"],
        "model": {"d_model": 2, "wellness": {"loss_weight": 0.1},
                  "motor_controller": {"output_per_handle": 9},
                  "gaze_head": {"enabled": True, "loss_weight": 1.0}},
        "body": {"n_handles": 1},
        "modalities": {m: {"loss_weight": 1.0} for m in ("vision", "motor")},
        "training": {"loss": {"source_scales": {
            "real": 1.0, "stale": 0.3, "synthetic": 0.0}}},
    })


@pytest.mark.parametrize("current,target,scale", [
    (["real"], ["synthetic"], 0.0),
    (["synthetic"], ["real"], 1.0),
    (["real"], ["stale"], 0.3),
    (["stale"], ["real"], 1.0),
    (["real", "real"], ["real", "synthetic"], 0.0),
    (["real", "real"], ["real", "stale"], 0.3),
    ([0], [2], 0.0),
    ([2], [0], 1.0),
    ([0], [1], 0.3),
    ([0, "real"], ["real", 2], 0.0),
])
def test_prediction_gradient_follows_next_frame(current, target, scale):
    before = {"source_masks": [{"vision": value} for value in current]}
    after = {"source_masks": [{"vision": value} for value in target]}
    current_mask, target_mask = training_source_masks(before, after, ["vision", "motor"])
    prediction = torch.ones(len(target), 2, requires_grad=True)
    loss = manager().compute(
        {"vision": prediction}, {"vision": torch.zeros_like(prediction)}, current_mask,
        prediction_source_mask=target_mask)
    loss.backward()
    assert loss.item() == pytest.approx(scale)
    assert torch.allclose(prediction.grad, torch.full_like(prediction, 2 * scale / prediction.numel()))


def test_target_mask_does_not_relabel_current_gaze():
    current, target = training_source_masks(
        {"source_masks": [{"vision": "real"}]},
        {"source_masks": [{"vision": "synthetic"}]}, ["vision"])
    prediction = torch.ones(1, 2, requires_grad=True)
    gaze = torch.ones(1, 2, requires_grad=True)
    loss = manager().compute(
        {"vision": prediction}, {"vision": torch.zeros_like(prediction)}, current,
        prediction_source_mask=target, gaze_out=gaze, gaze_target=torch.zeros_like(gaze))
    loss.backward()
    assert prediction.grad.count_nonzero() == 0
    assert torch.equal(gaze.grad, torch.ones_like(gaze))


def test_default_preserves_callers_with_same_frame_masks():
    prediction = torch.ones(1, 2, requires_grad=True)
    loss = manager().compute(
        {"vision": prediction}, {"vision": torch.zeros_like(prediction)}, {"vision": "stale"})
    loss.backward()
    assert loss.item() == pytest.approx(0.3)
    assert torch.allclose(prediction.grad, torch.full_like(prediction, 0.3))
