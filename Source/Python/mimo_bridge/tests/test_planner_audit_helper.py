# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Counterfactual instrumentation must not change the acting search or its random stream."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from lem.model.intention_generator import IntentionGenerator
from mimo_bridge.lem_model import LemLite


def test_counterfactual_audit_restores_rng_gradients_weights_and_chosen_action(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[4]
    source = root / "docs/gain_of_function_v3/audit_planner_decisions.py"
    spec = importlib.util.spec_from_file_location("audit_under_test", source)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    old_source = tmp_path / "previous_planner.py"
    old_source.write_text('''from lem.model.intention_generator import IntentionGenerator as Current
class IntentionGenerator:
    def generate_action(self, current, goal, motor, **kw):
        for p in self._forward_model.parameters():
            p.grad.add_(123.0)
        return Current.generate_action(self, current, goal, motor, **kw)
''', encoding="utf-8")
    modules = [torch.nn.Linear(1, 1) for _ in range(3)]
    config = {"body": {"n_handles": 2}, "model": {"d_model": 1},
              "_active_modalities": ["touch", "motor"],
              "training": {"intention": {"n_steps": 2, "lr": 0.1, "strength_reg": 0.01,
                                           "effort_per_body": True}}}
    planner = IntentionGenerator(config, *modules)
    motor = torch.zeros(2, 2, 9)
    motor[..., 0] = 0.01
    motor[..., 6] = 1
    motor[..., 7:] = 0.5
    state = {"touch": torch.zeros(2, 1)}
    params = [p for m in modules for p in m.parameters()]
    for p in params:
        p.grad = torch.full_like(p, 3.0)
    versions = lambda: tuple((id(p), p._version) for p in params)
    achieved = torch.zeros(2, 2, 10)
    achieved[..., 6] = 1
    decision = {"parameter_versions": versions(), "smooth_prev": None,
                "smoothing": 0.35, "dev": {"rom_limit": 1.0, "strength_limit": 1.0}}
    model = SimpleNamespace(intention_gen=planner, params=params, device=torch.device("cpu"),
        forward_model=modules[0], encoder_registry=modules[1], prediction_heads=modules[2],
        _plan_parameter_versions=versions, GAZE_IDX=1, smooth_source="proposal",
        _last_plan_inputs={"decision": decision, "cur": state, "goal": state, "motor": motor, "max_steps": 1})

    def transition(_fm, _enc, _heads, _state, action, **kw):
        torch.rand(())  # consume RNG without changing prediction; instrumentation must restore it
        return SimpleNamespace(tokens={"touch": action[:, 0, :1] + 0.1 * action[:, 1, 3:4]})

    monkeypatch.setattr("lem.model.predictive_state.transition", transition)

    def execution(raw, limits, handle_state=None):
        return LemLite.planned_execution(raw, limits, handle_state, previous=None, smoothing=0.35,
                                        keep_gaze_idx=1)

    kwargs = dict(execution_transform=execution, dev_limits=decision["dev"],
                  achieved_state=achieved, source_mask={"touch": "real"})
    rng = torch.get_rng_state().clone()
    expected, _ = planner.generate_action(state, state, motor, **kwargs)
    after_expected = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    before_weights = [p.detach().clone() for p in params]
    audit.install(model, tmp_path, old_source, max_events=1)
    actual, _ = planner.generate_action(state, state, motor, **kwargs)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), after_expected)
    assert all(torch.equal(p, old) for p, old in zip(params, before_weights))
    assert all(torch.equal(p.grad, torch.full_like(p, 3.0)) for p in params)
    assert (tmp_path / "matched_decisions.jsonl").is_file()
