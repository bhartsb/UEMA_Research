# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Independent adversarial checks for the development-only coupled readout."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from lem.model.causal_readout_v2 import (
    LocalCausalReadout, group_scales, grouped_objective, join_senses,
    layout_for, split_senses,
)
from mimo_bridge.tests.test_causal_readout_v2 import (
    inputs, randomize_outputs, synthetic_rows,
)
from mimo_bridge.tools import causal_readout_v2 as study


def test_future_order_of_one_handle_can_change_other_body_predictions():
    """A coupled torso command must be representable in another limb's future."""
    torch.manual_seed(121)
    model = LocalCausalReadout(4, 3, 3, 2, hidden=12)
    randomize_outputs(model)
    x = inputs(1, 3, 2)
    changed = deepcopy(x)
    changed['actions'][:, :, 0, 0] = 70.
    a, _ = split_senses(model(**x), 3, 2)
    b, _ = split_senses(model(**changed), 3, 2)
    assert not torch.equal(a[:, :, 1:], b[:, :, 1:]), (
        'Other body handles cannot respond to this prospective order at all; '
        'shared output weights alone do not model mechanical coupling.'
    )
    assert torch.equal(model(**x, blind=True), model(**changed, blind=True))


def test_shared_body_coupling_respects_handle_permutation_and_site_transfer():
    torch.manual_seed(122)
    model = LocalCausalReadout(4, 3, 3, 2, hidden=12)
    randomize_outputs(model)
    x = inputs(2, 3, 2)
    x['sensed'].normal_()
    x['achieved'][..., :3].normal_()
    x['actions'][..., :3].normal_(0, 25)
    x['site_position'].normal_()
    permutation = torch.tensor([2, 0, 1])
    body, skin = split_senses(x['sensed'], 3, 2)
    changed = deepcopy(x)
    changed['sensed'] = join_senses(body[:, :, permutation], skin)
    changed['achieved'] = x['achieved'][:, :, permutation]
    changed['actions'] = x['actions'][:, :, permutation]
    original_body, original_skin = split_senses(model(**x), 3, 2)
    actual_body, actual_skin = split_senses(model(**changed), 3, 2)
    torch.testing.assert_close(actual_body, original_body[:, :, permutation], atol=3e-7, rtol=2e-6)
    torch.testing.assert_close(actual_skin, original_skin, atol=3e-7, rtol=2e-6)


def test_future_targets_cannot_change_forward_inputs_or_fixed_fit_normalization():
    rows = synthetic_rows()
    norm = study.normalization(rows[:1])
    original_inputs, original_labels = study.batch(rows[2:], norm)
    changed = deepcopy(rows)
    for row in changed[1:]:
        row['target'] = torch.randn_like(row['target']) * 1.e5
    changed_norm = study.normalization(changed[:1])
    changed_inputs, changed_labels = study.batch(changed[2:], changed_norm)
    for name in original_inputs:
        assert torch.equal(original_inputs[name], changed_inputs[name]), name
    for name in ('scale', 'contrast_scale', 'sensed_scale', 'encoded_scale', 'region_scale'):
        assert torch.equal(norm[name], changed_norm[name]), name
    assert not torch.equal(original_labels, changed_labels)


def test_uniform_skin_duplication_does_not_change_normalized_loss():
    torch.manual_seed(123)
    h, s = 2, 2
    y = torch.randn(3, 4, 5, 9*h+10*s)
    p = y + torch.randn_like(y)*.2
    origin = torch.randn(3, 9*h+10*s)
    def duplicate(values):
        body, skin = split_senses(values, h, s)
        return join_senses(body, skin.repeat_interleave(2, dim=-2))
    rows = [dict(target=y[i], start=origin[i]) for i in range(len(y))]
    bigger = [dict(target=duplicate(y[i]), start=duplicate(origin[i])) for i in range(len(y))]
    scale, contrast, _ = group_scales(rows, layout_for(h, s))
    scale2, contrast2, _ = group_scales(bigger, layout_for(h, 2*s))
    loss, terms = grouped_objective((p-origin[:, None, None])/scale,
        (y-origin[:, None, None])/scale, layout_for(h,s), contrast, True)
    loss2, terms2 = grouped_objective((duplicate(p)-duplicate(origin)[:, None, None])/scale2,
        (duplicate(y)-duplicate(origin)[:, None, None])/scale2,
        layout_for(h, 2*s), contrast2, True)
    torch.testing.assert_close(loss, loss2)
    for key in terms:
        torch.testing.assert_close(terms[key], terms2[key])


@pytest.mark.parametrize('coverage', ['empty', 'missing', 'extra'])
def test_restored_auxiliary_requires_complete_exact_source_manifest(tmp_path, monkeypatch, coverage):
    model = LocalCausalReadout(4, 3, 2, 2, hidden=8)
    code = study.source_record()
    if coverage == 'empty':
        code.clear()
    elif coverage == 'missing':
        code.pop(next(iter(code)))
    else:
        extra = tmp_path/'extra.py'
        extra.write_text('not part of the declared model')
        code[str(extra.resolve())] = dict(sha256=study.file_hash(extra), source=extra.read_bytes())
    encoder = tmp_path/'encoder.pt'
    encoder.write_bytes(b'isolated encoder fixture')
    path = tmp_path/'head.pt'
    torch.save(dict(schema=study.SCHEMA, architecture=dict(encoded_width=4, region_width=3,
        handles=2, sites=2, hidden=8), model=model.state_dict(),
        source=dict(code=code, encoder_path=str(encoder), encoder_file_sha256=study.file_hash(encoder))), path)
    monkeypatch.setattr(study, 'restore_encoder', lambda _: object())
    with pytest.raises(ValueError, match='source|coverage|manifest|code'):
        study.restore_readout(path)


def test_nonfinite_backward_refuses_optimizer_update(tmp_path, monkeypatch):
    class PoisonBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()
        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, torch.nan)
    original_objective = study.grouped_objective
    def objective(*args, **kwargs):
        loss, terms = original_objective(*args, **kwargs)
        assert torch.isfinite(loss)
        return PoisonBackward.apply(loss), terms
    monkeypatch.setattr(study, 'grouped_objective', objective)
    steps = []
    original_step = torch.optim.AdamW.step
    def step(optimizer, *args, **kwargs):
        steps.append(True)
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    args = SimpleNamespace(out=tmp_path, fit_worlds=[1,2], evaluation_worlds=[3,4],
        states=1, hidden=8, seed=4, budgets=[1], batch_decisions=1)
    with pytest.raises((ValueError, RuntimeError), match='non.?finite|nan|Non-finite'):
        study.run_factorial(synthetic_rows(), args, {'constructed': True})
    assert not steps, 'A finite loss must not allow invalid backward gradients into Adam.'
    assert not list(tmp_path.glob('*.pt'))
