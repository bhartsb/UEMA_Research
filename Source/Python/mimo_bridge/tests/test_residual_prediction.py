# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Residual prediction: off by default, identical when off, and impossible to half-apply.

The case for it, measured on H34's final checkpoint (1.0M steps, the first model whose
motor row carried the action in the encoder's units):

  * persistence is nearly as good as the model on every sense - vision 0.0629 against
    0.0491, touch 0.1091 against 0.1029, body position a tie - so most of the target is
    reproducible by copying the present, and the action's marginal value is a few percent
    of the loss
  * a two-layer MLP with 1,547 training examples extracts 9.2% from the executed order in
    predicting the CHANGE in the body-position token, while the 3.5M model after a million
    steps moves its prediction 0.0018 when the commanded rotation is swapped
  * the motor token's gradient is 23x smaller than proprioception's

Predicting next MINUS now removes the copyable part, so what the action explains becomes
the whole task rather than a few percent of it.

The danger is not the idea, it is the surface: fourteen production sites turn the
transformer's output into a prediction. A residual applied to some and not others trains
the model one way and measures it another - the shape of the copy-degeneration and of the
centimetre units defect, each of which cost runs. So the heads RAISE when the flag is on
and a caller does not pass the current tokens, and these tests hold that guarantee.
"""
import inspect

import pytest
import torch

from lem.model import prediction_heads as PH
from lem.model import predictive_state as PS
from mimo_bridge.lem_model import LemLite
from mimo_bridge.tools import causal_test as CT


def _model(residual):
    ov = {"model": {"residual_prediction": True}} if residual else None
    return LemLite(device="cpu", seed=0, intention=False, imagination=False, overrides=ov)


# ----------------------------------------------------------------- default and equality
def test_off_by_default():
    assert _model(False).prediction_heads.residual is False
    assert _model(True).prediction_heads.residual is True


def test_with_the_flag_off_nothing_changes():
    """A flag whose off-state alters a single number would invalidate every past run."""
    m = _model(False)
    torch.manual_seed(0)
    last = torch.randn(3, m.n_mod, m.d_model)
    got = m.prediction_heads(last)
    for name, head in m.prediction_heads.heads.items():
        torch.testing.assert_close(got[name], head(last))
    # and passing current anyway is ignored when off
    cur = {k: torch.randn(3, m.d_model) for k in m.prediction_heads.head_modalities}
    also = m.prediction_heads(last, current=cur, who="test")
    for k in got:
        torch.testing.assert_close(got[k], also[k])


def test_with_the_flag_on_the_head_predicts_the_change():
    m = _model(True)
    torch.manual_seed(0)
    last = torch.randn(3, m.n_mod, m.d_model)
    cur = {k: torch.randn(3, m.d_model) for k in m.prediction_heads.head_modalities}
    got = m.prediction_heads(last, current=cur, who="test")
    for name, head in m.prediction_heads.heads.items():
        torch.testing.assert_close(got[name] - cur[name], head(last))


# ----------------------------------------------------------------- the guarantee
def test_a_caller_that_forgets_the_current_tokens_RAISES():
    m = _model(True)
    last = torch.randn(2, m.n_mod, m.d_model)
    with pytest.raises(RuntimeError, match="without the current tokens"):
        m.prediction_heads(last)
    with pytest.raises(RuntimeError, match="no current token was given"):
        m.prediction_heads(last, current={}, who="test")


def test_the_error_names_the_call_site():
    """"Something raised" is not enough to find a missed site in fourteen."""
    m = _model(True)
    last = torch.randn(2, m.n_mod, m.d_model)
    try:
        m.prediction_heads(last, who="the site I forgot")
    except RuntimeError as e:
        assert "the site I forgot" in str(e)
    else:
        pytest.fail("did not raise")


def test_forward_stacked_is_guarded_too():
    """loss_manager and the imagination chain use the stacked form; an unguarded second
    entry point would be a hole in exactly the surface this is protecting."""
    m = _model(True)
    last = torch.randn(2, m.n_mod, m.d_model)
    with pytest.raises(RuntimeError):
        m.prediction_heads.forward_stacked(last)


def test_every_production_site_passes_the_current_tokens():
    """Source-level, so a new site added later is caught by review rather than by a run."""
    from mimo_bridge import lem_model as LM
    for mod, needle in ((LM, 'who="lem_model live"'),
                        (LM, 'who="lem_model delayed next-frame lesson"'),
                        (LM, 'who="lem_model replay"'),
                        (PS, 'who="predictive_state.transition"'),
                        (CT, 'who="causal_test._predict"')):
        assert needle in inspect.getsource(mod), f"{needle} is missing"


def test_the_trainer_exposes_it_and_defaults_off():
    from mimo_bridge import train_mimo as TM
    src = inspect.getsource(TM)
    assert '"--residual-prediction"' in src
    assert 'choices=["on", "off"], default="off"' in src
    assert 'residual_prediction=True' in src, "the flag must reach the model config"


# ----------------------------------------------------------------- it actually trains
def test_a_live_step_and_the_delayed_lesson_run_with_it_on(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    m = LemLite(device="cpu", seed=0, intention=False, imagination=True, next_frame_loss=True,
                overrides={"model": {"residual_prediction": True},
                           "training": {"steps_per_phase": 3000}})
    m.enable_replay(1, size_per_env=200)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "res"), tag="_res", vision=False, seed=9)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9)); dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(24):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        assert res.loss == res.loss, "loss went NaN with the residual on"
        assert res.next_pe_per_modality, "the delayed lesson produced nothing"
    finally:
        src.close()


def test_the_causal_test_builds_the_model_the_way_it_was_trained(tmp_path):
    """A residual checkpoint evaluated in a non-residual model reads the CHANGE as if it
    were the absolute next state. Measured before this was fixed: 23-43x the cost of
    persistence - a wrong answer that looks like a catastrophic experimental result rather
    than a bug, and the supervisor's own causal test would have produced it."""
    import json
    import mimo_bridge.train_mimo as TM
    from mimo_bridge.tools import causal_test as CT
    # it must be read from the run directory, not guessed
    run = tmp_path / "R"; (run / "checkpoints" / "ckpt_step_5").mkdir(parents=True)
    (run / "args.json").write_text(json.dumps({
        "residual_prediction": "on", "context_length": 64, "action_tenure_loss": "on",
        "top_down_gate": "on", "smooth_actions": 0.65}), encoding="utf-8")
    a, _p, found = CT._cfg_from_run(str(run / "checkpoints" / "ckpt_step_5"))
    assert found and a.get("residual_prediction") == "on"
    # Since 2026-09-22 the model is built by train_mimo.build_learner from the run's own
    # namespace; a constructor spy pins that each of these reaches the model.
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
            raise RuntimeError("spy stop")

    old = TM.LemLite
    TM.LemLite = Spy
    try:
        with pytest.raises(RuntimeError):
            CT._build_model(str(run / "checkpoints" / "ckpt_step_5"), "cpu", 0)
    finally:
        TM.LemLite = old
    assert seen["overrides"]["model"]["residual_prediction"] is True, (
        "the causal test must take residual_prediction from the run's args.json")
    assert seen["overrides"]["model"]["context_length"] == 64
    assert seen["action_tenure_loss"] is True, (
        "same rule for the tenure heads: build the model the way it was trained")
    assert seen["top_down_gate"] is True, (
        "review F5: dropping the gate discards saved td_gates and collects un-gated "
        "tokens against gated training targets")
    assert seen["smooth_actions"] == 0.65, (
        "review F5: unsmoothed commands change the motion regime under test")
