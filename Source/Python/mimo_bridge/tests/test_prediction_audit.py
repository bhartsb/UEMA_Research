# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import torch
from mimo_bridge.prediction_audit import paired_errors, accumulate, summarize


def test_small_predicted_change_does_not_pass_when_world_moves():
    before = torch.zeros(2, 4)
    actual = torch.ones(2, 4)
    sums = {}
    accumulate(sums, {'body': paired_errors(before, before, actual)})
    assert summarize(sums)['modalities']['body']['improvement_over_persistence'] == 0
    sums = {}
    accumulate(sums, {'body': paired_errors(actual, before, actual, before)})
    result = summarize(sums)['modalities']['body']
    assert result['improvement_over_persistence'] == 1
    assert result['improvement_over_shuffled'] == 1


def test_motionless_and_nonfinite_inputs_cannot_claim_success():
    still = torch.zeros(1, 4)
    sums = {}
    accumulate(sums, {'body': paired_errors(still, still, still)})
    assert summarize(sums)['modalities']['body']['improvement_over_persistence'] is None
    assert paired_errors(still+float('nan'), still, still) is None


def test_window_uses_error_sums_not_mean_of_ratios():
    sums = {}
    for effect in (1., 10.):
        before = torch.zeros(1, 1)
        actual = before+effect
        accumulate(sums, {'body': paired_errors(actual-1, before, actual)})
    assert abs(summarize(sums)['modalities']['body']['improvement_over_persistence'] - (1-2/101)) < 1e-7


def test_sensitivity_is_how_far_the_order_alone_moves_the_prediction():
    """DESIGN.md section 3: sensitivity = ||pred(real) - pred(shuffled)|| /
    ||pred(real) - current||. 0 = swapping the order leaves the prediction where it was."""
    current = torch.zeros(2, 4)
    pred = torch.ones(2, 4)                    # predicted change of 1 per element
    actual = torch.ones(2, 4)
    sums = {}
    accumulate(sums, {'body': paired_errors(pred, current, actual, pred.clone())})
    assert summarize(sums)['modalities']['body']['sensitivity'] == 0.0
    sums = {}
    accumulate(sums, {'body': paired_errors(pred, current, actual, pred - 0.5)})
    assert abs(summarize(sums)['modalities']['body']['sensitivity'] - 0.5) < 1e-9
    # a window pools the squares before the one square root: shifts 0 and 1 against
    # changes 1 and 1 read sqrt(1/2), not the mean of 0 and 1
    sums = {}
    accumulate(sums, {'body': paired_errors(pred, current, actual, pred.clone())})
    accumulate(sums, {'body': paired_errors(pred, current, actual, pred - 1.0)})
    assert abs(summarize(sums)['modalities']['body']['sensitivity'] - 0.5 ** 0.5) < 1e-9
    # no control this step -> no sensitivity, and the key is absent rather than 0
    sums = {}
    accumulate(sums, {'body': paired_errors(pred, current, actual)})
    assert 'sensitivity' not in summarize(sums)['modalities']['body']
    # a prediction that does not move from the present has no ratio to give
    sums = {}
    accumulate(sums, {'body': paired_errors(current, current, actual, current + 1.0)})
    assert summarize(sums)['modalities']['body']['sensitivity'] is None
