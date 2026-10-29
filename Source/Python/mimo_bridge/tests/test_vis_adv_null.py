# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Qualification of the visual-agency contrast and its TEMPORAL null
(2026-09-09 review: the first committed null rolled order and outcome together,
which by relabeling is exactly minus the real statistic - it contained the
signal. The repaired null scores current orders against a PAST event's
outcomes. This file is the required known-association / no-association
qualification, plus a regression pinning the broken form's identity.)"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimo_bridge.lem_model import eligible_contrast  # noqa: E402

B, D = 8, 16
SL = slice(0, D)


def _orthogonal_pairs(gen):
    """Per-body order representations q and PERFECTLY matching outcomes delta:
    body i's outcome is its own order vector (orthogonal rows), so the true
    association is maximal and impostor similarity is ~0."""
    q = torch.eye(B, D)
    delta = q.clone()
    return q, delta


def test_known_association_real_positive_null_zero():
    gen = torch.Generator().manual_seed(0)
    q, delta = _orthogonal_pairs(gen)
    el = torch.ones(B, dtype=torch.bool)
    real = eligible_contrast(q, delta, el, SL)
    assert real is not None and real > 0.9
    # the null as USED: an EMA over many events. Single events scatter +-0.17
    # at this B and D; the qualification is that the null DISTRIBUTION centers
    # on zero and the real statistic sits far outside it.
    nulls = []
    for _ in range(200):
        past_delta = torch.randn(B, D, generator=gen)   # unrelated past outcomes
        v = eligible_contrast(q, past_delta, el, SL)
        assert v is not None
        nulls.append(v)
    nulls = torch.tensor(nulls)
    assert abs(float(nulls.mean())) < 0.02, float(nulls.mean())
    assert real > float(nulls.abs().max()) + 0.5


def test_no_association_both_near_zero():
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(B, D, generator=gen)
    delta = torch.randn(B, D, generator=gen)          # unrelated to q
    past = torch.randn(B, D, generator=gen)
    el = torch.ones(B, dtype=torch.bool)
    real = eligible_contrast(q, delta, el, SL)
    null = eligible_contrast(q, past, el, SL)
    assert abs(real) < 0.25 and abs(null) < 0.25      # noise scale at B=8, D=16


def test_broken_null_identity_regression():
    """The defect this file exists to keep out: rolling q and delta TOGETHER
    (scoring S(q_j, delta_i) - S(q_j, delta_j) over pairs) equals MINUS the
    real statistic under full eligibility - proven here on the fixture, so any
    future 'null' that reproduces this identity is the broken form."""
    q, delta = _orthogonal_pairs(torch.Generator().manual_seed(2))
    el = torch.ones(B, dtype=torch.bool)
    real = eligible_contrast(q, delta, el, SL)
    broken_vals = []
    for k in range(1, B):
        qr = torch.roll(q, k, dims=0)
        at_n = torch.nn.functional.cosine_similarity(qr[:, SL], delta[:, SL], dim=-1)
        cn = torch.nn.functional.cosine_similarity(
            qr[:, SL], torch.roll(delta, k, dims=0)[:, SL], dim=-1)
        broken_vals.append(at_n - cn)
    broken = float(torch.cat(broken_vals).mean())
    assert abs(broken + real) < 1e-6, (broken, real)


def test_partial_eligibility_excludes_ineligible_rows():
    """An ineligible body with a huge fake 'match' must not enter the mean."""
    q, delta = _orthogonal_pairs(torch.Generator().manual_seed(3))
    delta[0] = q[0] * 100.0            # body 0: absurd signal
    el = torch.ones(B, dtype=torch.bool)
    el[0] = False
    with_out = eligible_contrast(q, delta, el, SL)
    delta0 = delta.clone(); delta0[0] = 0.0
    without = eligible_contrast(q, delta0, el, SL)
    assert with_out is not None and without is not None
    assert abs(with_out - without) < 1e-6   # row 0 never contributed


def test_single_body_returns_none():
    q = torch.randn(1, D)
    assert eligible_contrast(q, q.clone(), torch.ones(1, dtype=torch.bool), SL) is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
