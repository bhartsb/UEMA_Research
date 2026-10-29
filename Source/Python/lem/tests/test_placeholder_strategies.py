# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What a placeholder actually feeds the encoders, and who owns the buffer it returns.

Nothing under Source/Python exercised lem/placeholders/strategies.py: no test named
GaussianStrategy, so neither the distribution the clamp produces nor the reuse of the
pre-allocated buffer had ever been executed by a test.

The distribution is the point. master_config.yaml asks intermind for N(0.0, 0.01) and
GaussianStrategy clamps the result to the sensor range [0, 1], which removes the whole
negative half. What the encoders receive is max(0, N(0, std)): about half the numbers
exactly 0.0 and the rest small positives around a fixed offset of std/sqrt(2*pi). These
tests pin that, so the day the clamp changes it is a decision someone made rather than a
number that moved. Whether to change it is an operator decision - it alters the synthetic
input distribution mid-lineage, and intermind and the audio placeholder feed the
transformer's attention every step even though their loss weight is 0.0.
"""
import math

import torch

from lem.placeholders.strategies import GaussianStrategy


def test_a_zero_mean_gaussian_placeholder_is_rectified_by_the_sensor_clamp():
    torch.manual_seed(0)
    std = 0.01
    g = GaussianStrategy([10000], mean=0.0, std=std)
    x = g.generate(1).clone()
    n = x.numel()

    zero_share = float((x == 0).float().mean())
    assert 0.45 < zero_share < 0.55, (
        f"{zero_share:.4f} of the numbers are exactly 0.0; a true N(0, {std}) draw would "
        f"put 0.0000 there, and half-rectified is what the clamp produces")

    # the surviving half carries a fixed positive offset: E[max(0, N(0, s))] = s/sqrt(2pi)
    expected_mean = std / math.sqrt(2 * math.pi)
    expected_sd = std * math.sqrt(0.5 - 1 / (2 * math.pi))
    standard_error = expected_sd / math.sqrt(n)
    assert abs(float(x.mean()) - expected_mean) < 4 * standard_error, (
        f"sample mean {float(x.mean()):.6f} against the rectified-normal mean "
        f"{expected_mean:.6f} (standard error {standard_error:.6f})")
    assert abs(float(x.mean())) > 4 * standard_error, (
        "the channel's mean is indistinguishable from 0, so the clamp no longer "
        "rectifies - the synthetic input distribution changed")
    assert float(x.min()) == 0.0 and float(x.max()) > 0.0


def test_a_placeholder_that_fits_inside_the_sensor_range_is_not_rectified():
    """Vision asks for mean 0.5, std 0.15. Both ends of [0, 1] are 3.33 standard
    deviations away, so the clamp touches under 0.1% of the draw at each end instead of
    the 50% it removes from a mean-0 one, and the sample mean stays on 0.5."""
    torch.manual_seed(0)
    g = GaussianStrategy([10000], mean=0.5, std=0.15)
    x = g.generate(1).clone()
    assert float((x == 0).float().mean()) < 0.005
    assert float((x == 1).float().mean()) < 0.005
    assert abs(float(x.mean()) - 0.5) < 0.01


def test_generate_hands_back_the_same_buffer_every_call():
    """A caller that keeps the tensor across steps sees it refilled underneath. Nothing
    retains it today, which is why this is a written contract rather than a repair."""
    torch.manual_seed(0)
    g = GaussianStrategy([64], mean=0.5, std=0.15)
    first = g.generate(2)
    second = g.generate(2)
    assert first is second, "the pre-allocated buffer contract changed"
    held = g.generate(2)
    snapshot = held.clone()
    g.generate(2)
    assert not torch.equal(held, snapshot), (
        "a tensor held across two generate() calls did not change, so the buffer is no "
        "longer reused in place and the clone warning in the docstring is stale")
    assert GaussianStrategy.__doc__ is not None
    assert "clone" in GaussianStrategy.__doc__, (
        "the buffer-reuse contract must stay written where a caller will read it")



# ---------------------------------------------------------------- audit P012 (2026-09-18)
def test_the_clip_range_comes_from_the_config_and_defaults_to_the_sensor_range():
    """The default is the behaviour that runs today AND the config can now say otherwise.

    Measured on the H83 checkpoint (audit P012), recorded here and not asserted because
    it is a property of the TRAINED weights: the intermind encoder turns this clipped
    draw into a token that moves 0.0006 against a token size of 1.0, while an untrained
    encoder on the same draw moves 0.5950 and removing the clip's fixed offset alone
    restores 0.2988. A random linear map plus LayerNorm does not reproduce the flatness
    (clipped 0.29 of the free draw's spread), so no synthetic threshold stands in for it.
    """
    from mimo_bridge.lem_model import build_config
    from lem.placeholders.system import PlaceholderSystem
    torch.manual_seed(0)
    cfg = build_config({"model": {"context_length": 8}})
    ps = PlaceholderSystem(cfg, device="cpu")
    g = ps._generators["intermind"]
    assert g.clip == (0.0, 1.0)
    x = GaussianStrategy([10000], mean=g.mean, std=g.std, clip=g.clip).generate(1)
    zeros = float((x == 0).float().mean())
    assert 0.45 <= zeros <= 0.55
    expect = g.std / math.sqrt(2 * math.pi)
    se = g.std / math.sqrt(10000)
    assert abs(float(x.mean()) - expect) < 4 * se
    cfg2 = build_config({"model": {"context_length": 8},
                         "modalities": {"intermind": {"placeholder": {"clip": None}}}})
    g2 = PlaceholderSystem(cfg2, device="cpu")._generators["intermind"]
    assert g2.clip is None
    y = g2.generate(1)
    assert float((y == 0).float().mean()) == 0.0
    assert abs(float(y.mean())) < 4 * g2.std / math.sqrt(y.numel())

