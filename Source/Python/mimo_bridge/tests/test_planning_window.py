# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Report executed plans even when a later decision yields no command."""
from types import SimpleNamespace

from mimo_bridge.train_mimo import planning_window_summary, record_planning_decision


def test_plan_followed_by_no_action_preserves_both_decisions_in_window():
    acc = {}
    old = {}
    model = SimpleNamespace(
        _last_plan_inputs={"plan0": object(), "outcome": {"plan_length": 1, "stopped_by": "horizon"}},
        last_intention={"step": 20, "plan_length": 1, "reached_body": True, "descent": [{"first_loss": 2.0}]},
    )
    record_planning_decision(acc, model, old, 20, latched=True, source="planner")
    previous = model._last_plan_inputs
    model._last_plan_inputs = {"plan0": None, "outcome": {"plan_length": 0, "stopped_by": "converged"}}
    model.last_intention = None
    record_planning_decision(acc, model, previous, 25, latched=True, source="reflex_noise")
    latest_plan, window = planning_window_summary(acc, 20, 30)
    assert latest_plan["step"] == 20 and latest_plan["reached_body"] is True
    assert window["latest_decision"]["step"] == 25
    assert window["latest_decision"]["stopped_by"] == "converged"
    assert window["latest_decision"]["selected_plan"] is False
    assert window["decisions"] == 2
    assert window["selected_plans"] == window["latched_plans"] == window["no_action_decisions"] == 1
    assert (window["first_step"], window["stop_step"]) == (20, 30)
    # A later mutation or clearing by the live actor cannot change the archived plan.
    latest_plan["descent"][0]["first_loss"] = 9.0
    assert acc["planning"]["latest_plan"]["descent"][0]["first_loss"] == 2.0


def test_unlatched_or_overridden_plan_does_not_count_as_acting():
    for latched, source, override in ((False, "planner", False), (True, "self_choice", False),
                                      (True, "planner", True)):
        model = SimpleNamespace(_last_plan_inputs={"plan0": object()},
                                last_intention={"step": 20, "reached_body": None})
        acc = {}
        record_planning_decision(acc, model, None, 20, latched=latched, source=source,
                                 externally_overridden=override)
        last, window = planning_window_summary(acc, 20, 30)
        assert window["selected_plans"] == 1 and window["latched_plans"] == 0
        assert last["command_selected_for_body"] is False


def test_old_decision_cannot_be_counted_again_or_carried_into_next_window():
    inputs = {"plan0": object()}
    model = SimpleNamespace(_last_plan_inputs=inputs, last_intention={"step": 20})
    acc = {}
    record_planning_decision(acc, model, inputs, 21, latched=False, source="reflex")
    last, window = planning_window_summary(acc, 21, 30)
    assert last is None and window["decisions"] == 0


def test_no_action_is_recorded_without_inventing_an_execution():
    model = SimpleNamespace(_last_plan_inputs={"plan0": None, "outcome": {"stopped_by": "scale_gate"}},
                            last_intention=None)
    acc = {}
    record_planning_decision(acc, model, None, 30, latched=True, source="reflex_noise")
    last, window = planning_window_summary(acc, 30, 40)
    assert last is None
    assert window["decisions"] == window["no_action_decisions"] == 1
    assert window["latched_plans"] == 0
    assert window["latest_decision"]["stopped_by"] == "scale_gate"
