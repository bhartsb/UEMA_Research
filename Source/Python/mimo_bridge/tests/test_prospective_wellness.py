# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Qualification of the prospective-wellness planning term (2026-09-09 plan).

What must hold before the term may run anywhere:
  off is off      weight 0 or no score_fn = the loss is bit-identical to baseline
  right lever     the term's gradient reaches the action ONLY through predicted
                  sensory tokens - a judge that reads only the motor token moves
                  nothing (the wireheading lever is closed by the detach)
  right direction the term pulls predictions toward higher judged wellness
  plumbing        causal_test._build_model forwards the flag (the class that has
                  produced FOUR silent probe/run divergences this month)
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lem.model.intention_generator import (  # noqa: E402
    IntentionGenerator, prospective_wellness_loss, _freeze_params, _restore_params)


def _score_mean_of_vision(sens, z_m):
    """Toy judge: wellness = mean of the vision token (higher vision = happier)."""
    return sens["vision"].mean(dim=-1, keepdim=True).sigmoid()


def test_off_is_exactly_off():
    preds = {"vision": torch.randn(2, 8, requires_grad=True),
             "motor": torch.randn(2, 8)}
    base = torch.tensor(1.5)
    with_term = base + prospective_wellness_loss(preds, _score_mean_of_vision, 0.0)
    assert float(with_term) == float(base)


def test_gradient_flows_through_sensory_predictions():
    v = torch.zeros(2, 8, requires_grad=True)
    preds = {"vision": v, "motor": torch.randn(2, 8)}
    loss = prospective_wellness_loss(preds, _score_mean_of_vision, 1.0)
    loss.backward()
    assert v.grad is not None and float(v.grad.abs().sum()) > 0
    # and the pull is toward HIGHER judged wellness: loss = 1 - w, so the
    # gradient on vision must point negative (descending raises the mean)
    assert float(v.grad.sum()) < 0


def test_motor_lever_is_dead():
    """A judge that reads ONLY the motor token must move the action not at all:
    the motor token is detached inside the term (adversarial pre-review: an
    attached motor token is a direct flatter-the-judge lever)."""
    m = torch.randn(2, 8, requires_grad=True)
    preds = {"vision": torch.randn(2, 8), "motor": m}

    def motor_only_judge(sens, z_m):
        return z_m.mean(dim=-1, keepdim=True).sigmoid()

    loss = prospective_wellness_loss(preds, motor_only_judge, 1.0)
    if loss.requires_grad:
        loss.backward()
        assert m.grad is None or float(m.grad.abs().sum()) == 0.0
    else:
        # even stronger: with only the motor token feeding the judge, the term
        # has NO gradient graph at all - the lever is severed, not just damped
        assert m.grad is None


def test_generator_defaults_are_inert():
    cfg = {"training": {"intention": {}}, "body": {"n_handles": 24},
           "model": {"d_model": 8}, "_active_modalities": ["vision", "motor"]}
    gen = IntentionGenerator(cfg, torch.nn.Identity(), torch.nn.Identity(),
                             torch.nn.Identity())
    assert gen._wellness_score_fn is None
    assert gen._wellness_w == 0.0
    assert gen._wellness_freeze == []


def test_generator_reads_weight_from_config():
    cfg = {"training": {"intention": {"prospective_wellness_w": 0.25}},
           "body": {"n_handles": 24}, "model": {"d_model": 8},
           "_active_modalities": ["vision", "motor"]}
    gen = IntentionGenerator(cfg, torch.nn.Identity(), torch.nn.Identity(),
                             torch.nn.Identity())
    assert gen._wellness_w == 0.25


def test_build_model_forwards_the_flag(tmp_path, monkeypatch):
    """Constructor spy on causal_test._build_model: the run's args request the
    term and the LemLite constructor must receive it (plumbing-hole class)."""
    from mimo_bridge.tools import causal_test as ct
    run = tmp_path / "R1"
    ckpt = run / "checkpoints" / "ckpt_step_10"
    ckpt.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps({
        "context_length": 4, "residual_prediction": "on", "intention": "on",
        "prospective_wellness": "add", "prospective_wellness_w": 0.2}),
        encoding="utf-8")

    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
            raise RuntimeError("spy stop")

    import mimo_bridge.lem_model as LM
    import mimo_bridge.train_mimo as TM
    monkeypatch.setattr(LM, "LemLite", Spy)
    # since 2026-09-22 _build_model builds through train_mimo.build_learner, which
    # calls the LemLite name train_mimo imported - the spy must sit there too
    monkeypatch.setattr(TM, "LemLite", Spy)
    with pytest.raises(RuntimeError):
        ct._build_model(str(ckpt), "cpu", 0, intention=True)
    assert seen.get("prospective_wellness") == "add"
    assert seen.get("prospective_wellness_w") == 0.2
    # and the A/B override path reaches the constructor too
    seen.clear()
    with pytest.raises(RuntimeError):
        ct._build_model(str(ckpt), "cpu", 0, intention=True,
                        arg_overrides={"prospective_wellness": "off"})
    assert seen.get("prospective_wellness") == "off"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_freeze_restore_preserves_prior_flags():
    """09-09 review counterexample: the old restore set every param trainable.
    A deliberately frozen parameter must stay frozen through a planning cycle."""
    m = torch.nn.Linear(4, 4)
    m.bias.requires_grad_(False)
    prior = _freeze_params([m])
    assert not any(p.requires_grad for p in m.parameters())
    _restore_params(prior)
    assert m.weight.requires_grad and not m.bias.requires_grad


def test_explicit_zero_weight_survives_build(tmp_path, monkeypatch):
    """09-09 review: `or 0.15` promoted an explicit zero weight to 0.15 -
    a zero-weight control arm would silently run at 0.15."""
    import json as _json
    from mimo_bridge.tools import causal_test as ct
    run = tmp_path / "R0"
    ckpt = run / "checkpoints" / "c1"
    ckpt.mkdir(parents=True)
    (run / "args.json").write_text(_json.dumps({
        "context_length": 4, "residual_prediction": "on", "intention": "on",
        "prospective_wellness": "add", "prospective_wellness_w": 0.0}),
        encoding="utf-8")
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
            raise RuntimeError("spy stop")

    import mimo_bridge.lem_model as LM
    import mimo_bridge.train_mimo as TM
    monkeypatch.setattr(LM, "LemLite", Spy)
    # since 2026-09-22 _build_model builds through train_mimo.build_learner, which
    # calls the LemLite name train_mimo imported - the spy must sit there too
    monkeypatch.setattr(TM, "LemLite", Spy)
    with pytest.raises(RuntimeError):
        ct._build_model(str(ckpt), "cpu", 0, intention=True)
    assert seen.get("prospective_wellness_w") == 0.0


def test_term_scores_bodily_columns_only():
    """09-09 review counterexample: averaging all five judge outputs let
    prediction-success and coherence outvote worse touch/capacity/posture.
    The wired closure must select tactile_comfort, capability, motor_state."""
    import inspect
    import mimo_bridge.lem_model as LM
    src = inspect.getsource(LM.LemLite.__init__)
    assert "[:, [1, 2, 4]]" in src, (
        "the prospective-wellness closure no longer selects the bodily columns")
    # and the column indices still mean what the selection assumes
    from lem.model.wellness import WellnessEvaluators
    names = WellnessEvaluators.EVALUATOR_NAMES
    assert [names[i] for i in (1, 2, 4)] == [
        "tactile_comfort", "capability", "motor_state"]


def test_goal_mechanism_flags_reach_the_constructor(tmp_path, monkeypatch):
    """The two 09-09 completion mechanisms ride the same plumbing as every
    other flag - forwarded at birth (the four-strikes plumbing-hole class)."""
    import json as _json
    from mimo_bridge.tools import causal_test as ct
    run = tmp_path / "R2"
    ckpt = run / "checkpoints" / "c1"
    ckpt.mkdir(parents=True)
    (run / "args.json").write_text(_json.dumps({
        "context_length": 4, "residual_prediction": "on", "intention": "on",
        "goal_gaze_hold": "on", "goal_revalue": "on"}), encoding="utf-8")
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
            raise RuntimeError("spy stop")

    import mimo_bridge.lem_model as LM
    import mimo_bridge.train_mimo as TM
    monkeypatch.setattr(LM, "LemLite", Spy)
    # since 2026-09-22 _build_model builds through train_mimo.build_learner, which
    # calls the LemLite name train_mimo imported - the spy must sit there too
    monkeypatch.setattr(TM, "LemLite", Spy)
    with pytest.raises(RuntimeError):
        ct._build_model(str(ckpt), "cpu", 0, intention=True)
    assert seen.get("goal_gaze_hold") == "on"
    assert seen.get("goal_revalue") == "on"


def test_gaze_hold_and_revalue_wiring_present():
    """Source-level pins for the two completion mechanisms: the hold consumes
    and expires at the emission site; revaluation feeds SELECTION only and
    keeps stored scalars untouched."""
    import inspect
    import mimo_bridge.lem_model as LM
    step_src = inspect.getsource(LM.LemLite.step)
    assert "self._gaze_hold = (gaze_out," in step_src        # set on plan execute
    assert "gaze_out = _tgt" in step_src                      # consumed between plans
    assert "wl_stack = self._revalue_ring(wl_stack)" in step_src
    rev_src = inspect.getsource(LM.LemLite._revalue_ring)
    assert "wj[:, [1, 4]]" in rev_src                         # touch + posture columns
    assert "_last_capability" in rev_src                      # PRESENT body capacity


def test_revalue_ring_prefers_current_judge_ordering():
    """Behavioral check of the revaluation math on a stub self: two stored
    candidates whose STORED scalars say A > B, while the current judge says
    B's tokens feel better - selection values must flip the order."""
    import types
    import mimo_bridge.lem_model as LM

    class Judge:
        def score_predicted_wellness(self, sens, zm):
            # candidate identity travels in the vision token's first element
            v = sens["vision"][:, 0]
            out = torch.zeros(v.shape[0], 5)
            out[:, 1] = v          # tactile_comfort column follows the marker
            out[:, 4] = v
            return out

        def _formula_capability(self, cap, B, device):
            return torch.zeros(B, 1)

    self_stub = types.SimpleNamespace(
        goal_ring=[({"vision": torch.tensor([[0.1]]), "motor": torch.tensor([[0.0]])},
                    torch.tensor([9.0])),          # stored scalar says A is great
                   ({"vision": torch.tensor([[0.9]]), "motor": torch.tensor([[0.0]])},
                    torch.tensor([1.0]))],         # stored scalar says B is poor
        active=["vision", "motor"], d_model=1,
        wellness=Judge(), _last_capability=None)
    wl = torch.stack([w for _, w in self_stub.goal_ring])
    out = LM.LemLite._revalue_ring(self_stub, wl)
    assert out.shape == wl.shape
    assert float(out[1, 0]) > float(out[0, 0]), (
        "revaluation must follow the CURRENT judge, not the stored scalar")
