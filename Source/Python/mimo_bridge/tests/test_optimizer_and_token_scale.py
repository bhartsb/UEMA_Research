# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Two properties that were violated for the whole project and never checked.

1. A REAL ENCODER TOKEN HAS RMS 1.0000.

   predictive_state.py documents that invariant and the contract overseer relies on it,
   but the shared output LayerNorm carried a learnable gain AND bias, so the invariant
   was false by construction. Measured across H31's eight hourly checkpoints:

       ||output_norm.bias||   3.21  6.58  9.63  11.60  13.76  17.63  20.62  22.02
       share of token energy  10.9% 28.5% 43.7% 51.8%  60.1%  70.1%  74.9%  76.4%

   Three quarters of every token had become one CONSTANT vector - identical across all
   six modalities and all 64 timesteps - and it was still climbing when the run ended.
   The gradient beats weight decay on that tensor by 3-19x, so decay was never going to
   hold it. Nothing is lost by removing the affine: all six encoders already end in their
   own norm, so per-modality scale stays learnable.

2. WEIGHT DECAY BELONGS ON MATRICES, NOT ON NORMS AND BIASES.

   The model built one flat AdamW group with weight_decay 0.01 over all 3.97M parameters.
   A 1-D parameter has no redundant directions to shrink; decaying it just drags a
   calibration toward an arbitrary origin.

   It mattered more than usual here. The forward model's 18 LayerNorm affine tensors -
   2,304 numbers, 0.34% of it - are sufficient to produce the whole drift toward copying
   the present: transplanting the hour-8 affines onto the hour-1 model reproduces 159% of
   the copy-R^2 rise and 158% of the history-sensitivity loss, and reverting them on the
   hour-8 model recovers 68% and 54%.
"""
import pytest
import torch


def _model():
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", seed=0, intention=False, imagination=False)


def test_the_shared_output_norm_has_no_learnable_affine():
    m = _model()
    n = m.encoder_registry.output_norm
    assert n is not None, "the shared output norm is gone entirely - different problem"
    assert n.elementwise_affine is False, (
        "the shared output LayerNorm is affine again; its bias ran to 76.4% of token "
        "energy over eight hours the last time it was")
    assert sum(p.numel() for p in n.parameters()) == 0


def test_a_token_leaves_the_registry_at_rms_one():
    """The invariant predictive_state.py documents, checked on output rather than asserted."""
    m = _model()
    n = m.encoder_registry.output_norm
    # LayerNorm divides by sqrt(var + eps). At input variance comparable to eps
    # (1e-5) it deliberately under-normalises, so these scales stay in the range a
    # real encoder produces rather than probing that corner.
    for scale, offset in ((1.0, 0.0), (7.5, 3.0), (0.3, -11.0), (40.0, 250.0)):
        torch.manual_seed(0)
        x = torch.randn(64, 1, m.d_model) * scale + offset
        y = n(x)
        rms = float(y.pow(2).mean()) ** 0.5
        assert abs(rms - 1.0) < 2e-3, (
            f"token RMS {rms:.5f}, not 1.0, for input scale {scale} offset {offset}")
        assert abs(float(y.mean())) < 2e-3, (
            "token carries a constant offset - the thing that reached 76.4% of "
            "token energy on H31")


def test_the_old_affine_behaviour_is_still_reachable():
    """Reproducing an old experiment must remain possible."""
    from lem.model.encoder_registry import EncoderRegistry
    m = _model()
    cfg = {k: v for k, v in m.config.items()}
    cfg["encoder_defaults"] = {**cfg.get("encoder_defaults", {}),
                               "output_layernorm_affine": True}
    reg = EncoderRegistry(cfg)
    assert reg.output_norm.elementwise_affine is True


def test_norms_and_biases_are_exempt_from_weight_decay():
    m = _model()
    groups = m.optimizer.param_groups
    assert len(groups) == 2, f"expected a decayed and an undecayed group, got {len(groups)}"
    decayed = [g for g in groups if g["weight_decay"] > 0]
    undecayed = [g for g in groups if g["weight_decay"] == 0]
    assert len(decayed) == 1 and len(undecayed) == 1
    assert all(p.dim() >= 2 for p in decayed[0]["params"]), (
        "a 1-D parameter is being decayed - LayerNorm gains and biases must not be")
    assert all(p.dim() < 2 for p in undecayed[0]["params"])
    assert undecayed[0]["params"], "nothing is exempt, so the split did nothing"


def test_every_parameter_is_in_exactly_one_group():
    """A split that drops parameters would silently stop training them."""
    m = _model()
    seen = [p for g in m.optimizer.param_groups for p in g["params"]]
    ids = [id(p) for p in seen]
    assert len(ids) == len(set(ids)), "a parameter appears in more than one group"
    assert len(seen) == len(m.params), (
        f"optimizer holds {len(seen)} tensors but the model listed {len(m.params)}")


def test_the_split_is_by_shape_not_by_a_list_of_names():
    """A name list goes stale the first time a module is added."""
    import inspect
    from mimo_bridge import lem_model
    src = inspect.getsource(lem_model.LemLite.__init__)
    assert "p.dim() >= 2" in src and "p.dim() < 2" in src, (
        "the no-decay split is not shape-driven; a new module would silently be decayed")
