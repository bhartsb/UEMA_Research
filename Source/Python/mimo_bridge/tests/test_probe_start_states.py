# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A probe never loses a body whose planner makes no plan (2026-09-25).

The scheduled eye test (docs/gain_of_function_v3/probe_saved_gaze.py) failed twice at checkpoint
400,350 with "All eight bodies must complete both gaze conditions": body 2 produced no command
arm and no error. Body 2's harm reading was over the integrity gate line at every one of its 200
remembered moments, so the goal ranking gave it its least harmful moment, the present one, and,
run alone, its planner measured a distance of 0 to that goal and stopped as converged without an
order at every planning event (every 10 model steps). The harness asked for a moment with a plan
within 30 model steps, found none, and moved on to body 3 without a word.
Record: docs/gain_of_function_v4/agents/gaze_body2_2026-09-25.md.

These tests use a stand-in for the model's planning record (LemLite._last_plan_inputs); no model,
no world.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.tools import gof_probe_gaze as G
from mimo_bridge.tools import gof_probe_harness as H

SAVED_STEP = 400350      # the failing checkpoint; its first model step is a planning step
EVERY = 10               # H84's intention_every


class Planner:
    """The model side of a search: the step counter and the record LemLite leaves on a planning
    event (step % EVERY == 0). plans_at: model steps after the saved moment whose plan returns an
    order; every other planning event stops as converged with the present moment as its goal."""

    def __init__(self, plans_at=(), planner=True, start=SAVED_STEP):
        self.step_count = start
        self.start = start
        self.intention_gen = object() if planner else None
        self._last_plan_inputs = None
        self.plans_at = set(plans_at)

    def step(self):
        step = self.step_count
        self.step_count += 1                       # LemLite.step counts before it plans
        if self.intention_gen is None or step % EVERY:
            return
        cur = {"vision": torch.ones(1, 4), "audio": torch.ones(1, 4), "motor": torch.zeros(1, 4)}
        if step - self.start in self.plans_at:
            goal = {"vision": torch.zeros(1, 4), "audio": torch.zeros(1, 4), "motor": torch.zeros(1, 4)}
            plan0 = torch.ones(1, 3, 9)
            outcome = dict(plan_length=1, stopped_by="horizon", converged=False, scale_exit=False)
        else:
            goal = {k: v.clone() for k, v in cur.items()}          # the goal is the present moment
            plan0 = None
            outcome = dict(plan_length=0, stopped_by="converged", converged=True, scale_exit=False)
        self._last_plan_inputs = dict(cur=cur, goal=goal, motor=torch.zeros(1, 3, 9), plan0=plan0,
                                      outcome=outcome, max_steps=1,
                                      decision=dict(source_mask={"vision": "real", "audio": "real"}))


class Searcher(H.Session):
    """A Session holding only what collect_starts reads and start_states calls."""

    def __init__(self, model, body=0):
        self.model, self.body, self.closed, self.acted = model, body, False, 0

    def observe(self):
        self.model.step()

    def snapshot(self):
        return dict(model_step=self.model.step_count - 1)

    def policy_act(self):
        self.acted += 1

    def warm(self, n_steps):
        pass

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------- one body's search
def test_the_measured_body_gives_no_planned_moment_and_the_search_says_why():
    s = Searcher(Planner())
    assert s.collect_starts(1, 0, want_plan=True, max_steps=30) == []
    events = s.last_search["planning_events"]
    assert [e["step_offset"] for e in events] == [0, 10, 20]
    assert [e["step"] for e in events] == [400350, 400360, 400370]
    assert all(not e["planned"] and e["stopped_by"] == "converged" and e["goal_distance_now"] == 0.0
               for e in events)
    assert s.last_search["model_steps"] == 30 and s.acted == 30


def test_with_the_fallback_it_starts_at_its_first_planning_event_and_says_so():
    s = Searcher(Planner())
    (snap,) = s.collect_starts(1, 0, want_plan=True, max_steps=30, plan_fallback=True)
    assert snap["model_step"] == SAVED_STEP and snap["step_offset"] == 0 and snap["plan0"] is None
    p = snap["planner"]
    assert p["rule"] == "no_plan_in_window" and p["consulted"] and not p["planned"]
    assert p["stopped_by"] == "converged" and p["goal_distance_now"] == 0.0
    assert [e["step_offset"] for e in p["search"]] == [0, 10, 20]       # the whole window was searched


def test_a_plan_later_in_the_window_still_wins_over_the_fallback():
    s = Searcher(Planner(plans_at={10}))
    (snap,) = s.collect_starts(1, 0, want_plan=True, max_steps=30, plan_fallback=True)
    assert snap["step_offset"] == 10 and torch.is_tensor(snap["plan0"])
    assert snap["planner"]["rule"] == "planned" and snap["planner"]["planned"]
    assert snap["planner"]["goal_distance_now"] == 2.0                 # vision 1 + audio 1, motor left out
    assert [e["step_offset"] for e in snap["planner"]["search"]] == [0, 10]
    assert s.acted == 11                         # the search stopped after the plan's own step


def test_without_a_plan_request_any_moment_is_a_start_and_no_planner_is_recorded():
    s = Searcher(Planner(planner=False))
    (snap,) = s.collect_starts(1, 0, want_plan=True, max_steps=30)
    assert snap["step_offset"] == 0 and snap["plan0"] is None
    assert snap["planner"]["rule"] == "no_plan_required" and snap["planner"]["consulted"] is False


def test_the_goal_distance_is_the_planners_own_first_check():
    model = SimpleNamespace(step_count=11)
    cur = {"vision": torch.ones(1, 4), "audio": 5 * torch.ones(1, 4), "motor": torch.ones(1, 4)}
    goal = {"vision": torch.zeros(1, 4), "audio": torch.zeros(1, 4), "motor": torch.zeros(1, 4)}
    placeholder = dict(decision=dict(source_mask={"audio": "placeholder"}), cur=cur, goal=goal,
                       plan0=None, outcome=dict(plan_length=0, stopped_by="converged", converged=True))
    d = H.planner_decision(model, placeholder, 3)
    assert d["step"] == 10 and d["step_offset"] == 3
    assert d["goal_distance_now"] == 1.0            # vision only: audio is not graded real, motor is the action
    nothing_real = dict(placeholder, decision=dict(source_mask={"vision": "stale", "audio": "stale"}))
    assert H.planner_decision(model, nothing_real, 3)["goal_distance_now"] == 1.0 + 25.0 + 1.0
    assert H.planner_decision(model, None, 3) is None


# ---------------------------------------------------------------------------- start_states
@pytest.fixture
def episode_of_three(monkeypatch):
    """Three saved bodies; body 1 never plans (the measured case), bodies 0 and 2 plan 10 steps in."""
    made = []

    def session(frozen, *, episode=None, body=0, babble=0.0, **world):
        assert episode is not None and world == dict(gaze_reflex=False, saccades=False)
        s = Searcher(Planner(plans_at=() if body == 1 else {10}), body)
        made.append(s)
        return s

    monkeypatch.setattr(H, "load_episode", lambda frozen: (dict(worlds=[{}, {}, {}]), "episode note"))
    monkeypatch.setattr(H, "Session", session)
    frozen = SimpleNamespace(args=SimpleNamespace(babble_noise=0.05), info=dict(checkpoint="ckpt"))
    return frozen, made


def test_a_body_without_a_start_is_an_error_naming_it_never_a_silent_skip(episode_of_three):
    frozen, made = episode_of_three
    starts = H.start_states(frozen, 3, want_plan=True, spacing=10, gaze_reflex=False, saccades=False)
    first_session, first = next(starts)
    assert first_session.body == 0 and first["step_offset"] == 10
    with pytest.raises(H.NoStartState) as error:
        next(starts)
    assert str(error.value) == (
        "body 1 gave no start state in 30 model steps. The planner was consulted 3 times: 0 made a plan, 3 "
        "stopped without one (converged 3 times); its goal sat 0, 0, 0 from the present moment (under 0.01 the "
        "planner stops as converged). start_states never moves on without a body (START_WITHOUT_A_PLAN).")
    assert all(s.closed for s in made) and [s.body for s in made] == [0, 1]


def test_with_the_fallback_every_body_starts_and_the_planless_one_is_marked(episode_of_three):
    frozen, made = episode_of_three
    got = [(s.body, snap) for s, snap in H.start_states(frozen, 3, want_plan=True, spacing=10, plan_fallback=True,
                                                          gaze_reflex=False, saccades=False)]
    assert [b for b, _ in got] == [0, 1, 2]
    rules = {b: snap["planner"]["rule"] for b, snap in got}
    assert rules == {0: "planned", 1: "no_plan_in_window", 2: "planned"}
    assert got[1][1]["plan0"] is None and got[1][1]["step_offset"] == 0
    assert all(torch.is_tensor(got[i][1]["plan0"]) and got[i][1]["step_offset"] == 10 for i in (0, 2))
    assert all(s.closed for s in made)


def test_a_fresh_world_short_of_starts_is_an_error(monkeypatch):
    monkeypatch.setattr(H, "load_episode", lambda frozen: (None, "no companion"))
    monkeypatch.setattr(H, "Session", lambda frozen, **kw: Searcher(Planner()))
    frozen = SimpleNamespace(args=SimpleNamespace(babble_noise=0.0), info=dict(checkpoint="ckpt"))
    with pytest.raises(H.NoStartState, match=r"^the fresh world gave 0 of 2 start states in 400 model steps\. "
                                             r"The planner was consulted 40 times: 0 made a plan, 40 stopped "
                                             r"without one \(converged 40 times\); its goal sat 0, 0, 0, 0, 0, 0, "
                                             r"0, 0, and 32 more from"):
        list(H.start_states(frozen, 2, want_plan=True, spacing=10, warm=0))
    with pytest.raises(H.NoStartState, match=r"^the fresh world gave 1 of 2 start states"):
        list(H.start_states(frozen, 2, want_plan=True, spacing=10, warm=0, plan_fallback=True))


# ---------------------------------------------------------------------------- the eye probe
def test_the_eye_probe_asks_for_the_fallback_and_records_each_start(monkeypatch):
    seen = {}
    planner = dict(consulted=True, rule="no_plan_in_window", planned=False, stopped_by="converged",
                   goal_distance_now=0.0, search=[])
    session = SimpleNamespace(start="episode companion, body 2 of 8 at step 400350", episode_body=dict(body=2))

    def starts(frozen, n, **kw):
        seen.update(kw, n=n)
        yield session, dict(step_offset=0, plan0=None, planner=planner)

    monkeypatch.setattr(G.H, "start_states", starts)
    monkeypatch.setattr(G.H, "real_senses", lambda model: ["vision"])
    monkeypatch.setattr(G.H, "frozen_config", lambda frozen, s: {})
    monkeypatch.setattr(G, "_arms_from", lambda s, snap, depth, senses: dict(mom=None, eyes0=np.zeros(2), arms={}))
    monkeypatch.setattr(G, "_pair_noise", lambda s, snap, depth: {})
    monkeypatch.setattr(G, "_score_state", lambda res, senses, depth: {})
    monkeypatch.setattr(G, "_wellness_selection", lambda s, snap, res: {})
    monkeypatch.setattr(G, "_head_turned", lambda s, snap, depth, senses: {})
    cf = G.counterfactual(SimpleNamespace(model=None), states=8, warm=0, spacing=10, depth=3, seed=11, repeats=1)
    assert seen["want_plan"] is True and seen["plan_fallback"] is True and seen["n"] == 8
    assert seen["gaze_reflex"] is False and seen["saccades"] is False
    (state,) = cf["states"]
    assert state["body"] == 2 and state["step_offset"] == 0 and state["start_planner"] is planner


def test_the_planner_reading_says_why_there_was_no_plan(monkeypatch):
    monkeypatch.setattr(G.H, "predicted_wellness", lambda model, mom, order: 1.0)
    session = SimpleNamespace(model=None, restore=lambda snap: None, hold_order=lambda: np.zeros((1, 24, 9)),
                              exec_order=lambda body, g: body)
    res = dict(mom=None, eyes0=np.zeros(2), arms={n: dict(target_deg=[0.0, 0.0]) for n in ("left", "right", "up", "down")})
    snap = dict(plan0=None, planner=dict(consulted=True, stopped_by="converged", goal_distance_now=0.0))
    out = G._wellness_selection(session, snap, res)
    assert out["planner"].startswith("no plan at this moment: the planner stopped as converged, its goal 0 from")
    assert "agrees_horizontal" not in out and out["better_horizontal"] == "tie"
    old = G._wellness_selection(session, dict(plan0=None), res)        # a start without the record
    assert old["planner"] == "no plan at this moment (planner off in this build, or no plan made)"
