# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Regression checks for the diagnostic's fivefold time and context errors."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from mimo_bridge.tools.probe_clock import ProbeClock
from mimo_bridge.tools.consequence_localization import _decision_state


def test_simulator_steps_are_not_model_steps():
    c = ProbeClock(.01, 5, 5)
    assert [c.steps_for(s) for s in (.05, .25, 1.)] == [5, 25, 100]
    assert c.model_step_seconds == .05
    assert c.held_order_seconds == .25
    # The previous diagnostic labeled ONE simulator step as 50 ms.
    with pytest.raises(AssertionError):
        c.verify_elapsed(2., 2.01, .05)
    assert c.verify_elapsed(2., 2.05, .05) == pytest.approx(.05)


def test_schedule_duration_uses_model_clock():
    c = ProbeClock(.01, 5, 5)
    for duration in (1., 2., 3., 5.):
        steps = int(duration / c.model_step_seconds)
        assert steps * .05 == pytest.approx(duration)
    with pytest.raises(ValueError):
        ProbeClock(.03, 5, 5).steps_for(.05)


def test_prediction_context_ends_before_current_frame():
    pre = torch.arange(12.).reshape(1, 3, 4)
    live = SimpleNamespace(last_context=pre, context_buffer=pre + 100,
        _last_plan_inputs={'cur': {'tactile': torch.ones(1, 4)}},
        last_tokens={'tactile': torch.full((1, 4), 2.)},
        last_achieved=torch.zeros(1, 24, 10), _last_source_mask={'tactile': 'real'})
    acting = _decision_state(live)
    learning = _decision_state(live, raw=True)
    assert torch.equal(acting.context, pre)
    assert not torch.equal(acting.context, live.context_buffer)
    assert torch.equal(acting.tokens['tactile'], torch.ones(1, 4))
    assert torch.equal(learning.tokens['tactile'], torch.full((1, 4), 2.))
    pre.add_(3)
    assert not torch.equal(acting.context, pre), 'snapshot must survive later mutation'


def test_elapsed_clock_matches_real_source(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.tools.sensor_action_contract_check import hold_current
    source = SingleEnvSource(scene='crib', vision=False, run_dir=str(tmp_path), seed=37)
    try:
        frame = source.reset()[0]
        c = ProbeClock(float(source.env.dt), 5, 5)
        start = float(frame.sim_time)
        command = hold_current(frame.bones)
        for _ in range(c.steps_for(.05)):
            frame = source.step(command, np.zeros((1, 2)), render=False)[0]
        assert c.verify_elapsed(start, float(frame.sim_time), .05) == pytest.approx(.05)
    finally:
        source.close()


@pytest.mark.parametrize('with_pose', [False, True])
@pytest.mark.parametrize('smoothing', [0., .35, 1.])
def test_planned_command_matches_live_execution_and_gradient(with_pose, smoothing):
    from mimo_bridge.lem_model import LemLite
    raw = torch.zeros(1, 24, 9)
    raw[..., :3] = .1
    raw[..., 6] = 1.
    raw[..., 7:] = .4
    raw.requires_grad_(True)
    prev = raw.detach().clone()
    prev[..., :3] = -10.
    pose = torch.zeros(1, 24, 10) if with_pose else None
    dev = {'rom_limit': 1., 'strength_limit': .3, 'handle_mask': [True] * 23 + [False]}
    actual = LemLite.execution_transform(None, raw, dev, handle_state=pose)
    actual = LemLite.smooth_blend(actual, prev, smoothing)
    predicted = LemLite.planned_execution(raw, dev, pose, previous=prev, smoothing=smoothing)
    assert torch.allclose(predicted, actual)
    predicted[0, 0, 0].backward()
    blend = smoothing if smoothing > 0 else 1.
    if with_pose:
        # The reach bound is straight-through: the VALUE is the clamp's (asserted
        # bit-identical above) and the SLOPE is tanh's, so the planner keeps a
        # gradient on coordinates the clamp had flattened to exactly zero. The slope
        # is therefore 200 * sech^2(pos) * sech^2(step/reach), not a flat 200.
        reach = 40. * dev['rom_limit']
        pos = torch.tensor(.1)
        step = torch.tensor((.1 * 200. - 0.))
        expected = (200. * float(1 - torch.tanh(pos) ** 2)
                    * float(1 - torch.tanh(step / reach) ** 2) * blend)
    else:
        expected = 200. * blend
    assert float(raw.grad[0, 0, 0]) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize('smoothing', [0., .35, 1.])
def test_positional_gradient_survives_past_the_reach_limit(smoothing):
    """A saturated candidate must still be steerable, without changing what the body does.

    Both clamps in the planner's differentiated path returned gradient exactly 0.0 on
    every saturated coordinate, so positional descent died the round a coordinate hit
    the limit - and half of this run's recorded position channels start outside
    [-1, 1]. The bound is now straight-through: forward value from the clamp, backward
    slope from tanh. Measured at a 40 cm reach, asking for 50 cm: gradient 52.7 where
    the clamp gave 0.0, and the executed position still exactly 40 cm.

    THE GAP THIS TEST USED TO DECLARE IS CLOSED (audit P118, 2026-09-19): tanh reaches
    1.0 in float32 at |pos| >= 9.010840, so a raw coordinate at 568 normalised units had
    a 0.0 slope here too. The controller's position channel is now bounded at its source
    (lem/model/motor_controller.py, straight-through clamp), so the planner's seed never
    exceeds 1.0; test_position_head_is_bounded_and_the_executed_order_does_not_change
    below pins it.
    """
    from mimo_bridge.lem_model import LemLite
    dev = {'rom_limit': 1., 'strength_limit': .3, 'handle_mask': [True] * 24}
    pose = torch.zeros(1, 24, 10)
    prev = torch.zeros(1, 24, 9)
    reach = 40. * dev['rom_limit']
    for norm_pos in (0.25, 0.5, 2.0):          # 50, 100 and 400 cm asked, against 40 cm of reach
        raw = torch.zeros(1, 24, 9, requires_grad=True)
        with torch.no_grad():
            raw[..., :3] = norm_pos
            raw[..., 6] = 1.
            raw[..., 7:] = .4
        planned = LemLite.planned_execution(raw, dev, pose, previous=prev, smoothing=smoothing)
        planned[0, 0, 0].backward()
        assert float(raw.grad[0, 0, 0]) > 0., (
            f"no gradient at {norm_pos * 200:.0f} cm - the planner cannot steer a saturated handle")
        # and the body still performs exactly what it performed before this change
        body = LemLite.execution_transform(None, raw.detach(), dev, handle_state=pose)
        body = LemLite.smooth_blend(body, prev, smoothing)
        assert torch.allclose(planned, body), "the planner and the body parted company"
        assert abs(float(body[0, 0, 0])) <= reach + 1e-4


def test_planning_snapshot_survives_smoother_and_context_updates():
    from mimo_bridge.lem_model import LemLite
    p = torch.nn.Parameter(torch.ones(1))
    stub = SimpleNamespace(last_context=torch.ones(1, 3, 4),
        context_buffer=torch.zeros(1, 3, 4), _prev_mfa=torch.ones(1, 24, 9),
        last_achieved=torch.ones(1, 24, 10), _last_source_mask={'tactile': 'real'},
        _last_dev={'rom_limit': 1.}, smooth_actions=.35, params=[p],
        device=torch.device('cpu'), GAZE_IDX=LemLite.GAZE_IDX)     # the eyes' row the decision keeps as sent
    stub._plan_parameter_versions = lambda: LemLite._plan_parameter_versions(stub)
    stub._plan_settings = lambda: (20, .01, .1, False)
    saved = LemLite._capture_plan_state(stub)
    stub._prev_mfa.mul_(10)
    stub.last_context.mul_(20)
    stub.last_achieved.mul_(30)
    assert torch.equal(saved['smooth_prev'], torch.ones(1, 24, 9))
    assert torch.equal(saved['context'], torch.ones(1, 3, 4))
    assert torch.equal(saved['achieved'], torch.ones(1, 24, 10))
    assert saved['parameter_versions'] == stub._plan_parameter_versions()
    with torch.no_grad():
        p.add_(1)
    assert saved['parameter_versions'] != stub._plan_parameter_versions()


def test_replan_reproduces_action_after_live_buffers_change():
    from mimo_bridge.lem_model import LemLite
    m = LemLite(device='cpu', intention=True, imagination=False,
                overrides={'model': {'context_length': 4}}, smooth_actions=.35)
    m.intention_gen._n_steps = 2
    cur = {mod: torch.nn.functional.layer_norm(torch.randn(1, m.d_model), (m.d_model,))
           for mod in m.active}
    goal = {mod: -v for mod, v in cur.items()}
    raw = torch.zeros(1, 24, 9)
    raw[..., 6] = 1.
    raw[..., 7:] = .2
    m.last_context = torch.zeros(1, (m.ctx_len - 1) * m.n_mod, m.d_model)
    m.last_achieved = torch.zeros(1, 24, 10)
    m.last_achieved[..., 6] = 1.
    m._prev_mfa = raw.clone()
    m._last_dev = {'rom_limit': 1., 'strength_limit': 1.}
    m._last_source_mask = {mod: 'real' for mod in m.active}
    decision = m._capture_plan_state()
    plan, meta = m._plan_chain(cur, goal, raw, max_steps=1, decision=decision)
    assert plan and meta['valid']
    m._last_plan_inputs = dict(cur=cur, goal=goal, motor=raw, max_steps=1,
                               decision=decision, plan0=plan[0].detach().clone())
    m.last_context.add_(10)
    m.last_achieved.add_(1)
    m._prev_mfa.add_(20)
    state_before = torch.get_rng_state().clone()
    repeated, metadata = m.replan_last(m.intention_gen._wellness_w)
    assert repeated is not None, metadata
    assert torch.equal(repeated, plan[0])
    assert torch.equal(torch.get_rng_state(), state_before), 'replay changed the live RNG'
    with torch.no_grad():
        m.params[0].add_(.001)
    repeated, metadata = m.replan_last(0.)
    assert repeated is None and 'parameters changed' in metadata['error']
    m.reset_transition_state()
    repeated, metadata = m.replan_last(0.)
    assert repeated is None and 'no planning event' in metadata['error']



# ---------------------------------------------------------------- audit P123 (2026-09-18)
class _RecordingAdam(torch.optim.Adam):
    """Adam that keeps the gradient it was handed each round, per parameter group."""
    rounds: list = []

    def __init__(self, params, **kw):
        params = list(params)
        object.__setattr__(self, "_watch", params)
        super().__init__(params, **kw)

    def step(self, *a, **k):
        names = ("pos", "rot", "radius", "strength")
        _RecordingAdam.rounds.append(
            {n: (None if p.grad is None else float(p.grad.abs().max()))
             for n, p in zip(names, self._watch)})
        return super().step(*a, **k)


def _saturated_planner():
    """A built planner whose candidate starts OUTSIDE the reach envelope.

    An in-reach candidate proves nothing: the old plain clamp passes gradient
    normally there. At 1.5 normalised units - a 300 cm target against 40 cm of
    reach - the plain clamp returned exactly 0.0 on all 20 rounds.
    """
    from mimo_bridge.lem_model import LemLite
    torch.manual_seed(1)
    m = LemLite(device='cpu', intention=True, imagination=False,
                overrides={'model': {'context_length': 4}}, smooth_actions=.35)
    cur = {mod: torch.nn.functional.layer_norm(torch.randn(1, m.d_model), (m.d_model,))
           for mod in m.active}
    goal = {mod: -v for mod, v in cur.items()}
    raw = torch.zeros(1, 24, 9)
    raw[..., 6] = 1.
    raw[..., 7:] = .2
    raw[..., :3] = 1.5
    m.last_context = torch.zeros(1, (m.ctx_len - 1) * m.n_mod, m.d_model)
    m.last_achieved = torch.zeros(1, 24, 10)
    m.last_achieved[..., 6] = 1.
    m._prev_mfa = raw.clone()
    m._last_dev = {'rom_limit': 1., 'strength_limit': 1.}
    m._last_source_mask = {mod: 'real' for mod in m.active}
    return m, cur, goal, raw


def test_the_planner_keeps_a_position_gradient_through_the_executable_form(monkeypatch):
    """The composed loop, not its pieces: a detach anywhere between planned_execution
    and normalize_executed_command leaves every other test passing while the position
    gradient is None on all 20 rounds."""
    from lem.model.intention_generator import IntentionGenerator
    m, cur, goal, raw = _saturated_planner()
    seen = []
    orig = IntentionGenerator.generate_action

    def spy(self, *a, **k):
        seen.append(k.get('execution_transform') is not None)
        return orig(self, *a, **k)

    monkeypatch.setattr(IntentionGenerator, 'generate_action', spy)
    _RecordingAdam.rounds = []
    monkeypatch.setattr(torch.optim, 'Adam', _RecordingAdam)

    plan, meta = m._plan_chain(cur, goal, raw, max_steps=1)
    assert plan and meta['valid']
    assert seen == [True], "the planner stopped descending through the executable form"
    n_steps = m._plan_settings()[0]
    assert len(_RecordingAdam.rounds) == n_steps
    pos = [r['pos'] for r in _RecordingAdam.rounds]
    assert all(p is not None and p > 0.0 for p in pos), (
        f"the position gradient died inside the executable-form loop: {pos}")


def test_the_planner_closes_goal_distance_with_the_effort_term_switched_off():
    """With the effort term at zero the only way the number can fall is the predicted
    sensory consequence moving toward the goal. Deterministic at a fixed seed."""
    m, cur, goal, raw = _saturated_planner()
    m.intention_gen._strength_reg = 0.0
    plan, meta = m._plan_chain(cur, goal, raw, max_steps=1)
    assert plan and meta['valid']
    d = meta['descent'][0]
    assert d['first_effort'] == 0.0 and d['last_effort'] == 0.0
    assert d['last_goal_distance'] < d['first_goal_distance'], (
        "with no effort term the descent did not move closer to the goal: "
        f"{d['first_goal_distance']} -> {d['last_goal_distance']}")


def test_position_head_is_bounded_and_the_executed_order_does_not_change():
    """Audit P118 (2026-09-19): the controller's position channel is bounded at its source.

    Before: the position head was a bare linear output. On the H81 explain probe 791 of
    1,560 recorded position numbers lay outside -1..1 (largest 568), every execution path
    clamped them on the way to the body, and the planner - which seeds its candidate from
    the raw tensor - had a gradient of exactly 0.0 on the 580 coordinates at |pos| >=
    9.010840, for all 20 Adam rounds. Four contracts, at the five magnitudes that probe
    recorded:
    1. the value that leaves the controller is inside -1..1, the head still receives a
       gradient through the bound, and the excursion is reported beside it;
    2. the body performs exactly what it performed before (0.0 difference, not approx);
    3. the planner can steer a candidate seeded from the bounded value;
    4. a runaway inner number is still charged by the motor loss, and the charge is
       bounded (the squared form read 69,960 on that probe's step_160 proposal).
    """
    import torch.nn as nn
    from mimo_bridge.lem_model import LemLite
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    ctrl = m.motor_controller
    last = getattr(ctrl, "output_proj", None)
    if last is None:
        last = [x for x in ctrl.mlp if isinstance(x, nn.Linear)][-1]
    dev = {'rom_limit': 1., 'strength_limit': 1., 'handle_mask': [True] * 24}
    pose = torch.zeros(1, 24, 10)
    prev = torch.zeros(1, 24, 9)
    latents = torch.zeros(1, m.n_mod, m.d_model)
    target = torch.zeros(1, 24, 9)
    target[..., 6] = 1.
    charge = {}
    for mag in (0.26, 1.28, 39.75, 121.0, 568.0):
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            bias = last.bias.view(24, 9)
            bias[:, 0:3] = mag
            bias[:, 6] = 1.
        out, _conf, meta = ctrl(latents, pose)
        # 1. bounded value, reported excursion, and a live gradient through the bound
        assert float(out[..., :3].abs().max()) <= 1.0, f"a position of {mag} left the controller unbounded"
        assert float(meta["pos_excursion"].max()) == pytest.approx(max(0., mag - 1.), rel=1e-5)
        ctrl.zero_grad()
        out[..., 0].sum().backward(retain_graph=True)
        assert float(last.bias.grad.view(24, 9)[:, 0].abs().min()) > 0., "the bound cut the head off from its lesson"
        # 2. the body does exactly what it did
        raw_cmd = out.detach().clone()
        raw_cmd[..., :3] = mag
        body_now = LemLite.execution_transform(None, out.detach(), dev, handle_state=pose)
        body_before = LemLite.execution_transform(None, raw_cmd, dev, handle_state=pose)
        assert float((body_now[..., 0:3] - body_before[..., 0:3]).abs().max()) == 0.0, (
            f"bounding {mag} changed the executed centimetres")
        # 3. a candidate seeded from the bounded value is steerable
        seed = out.detach().clone().requires_grad_(True)
        planned = LemLite.planned_execution(seed, dev, pose, previous=prev, smoothing=.35)
        planned[0, 0, 0].backward()
        assert float(seed.grad[0, 0, 0]) > 0., f"the planner cannot steer a candidate seeded at {mag}"
        # 4. the pull-back
        ctrl.zero_grad()
        loss = m.loss_manager._compute_motor_loss(out, target, dev, pos_excursion=meta["pos_excursion"])
        loss.backward()
        charge[mag] = float(loss)
        if mag > 1.:
            assert float(last.bias.grad.view(24, 9)[:, 0].min()) > 0., "no inward pull on a runaway position"
    assert charge[568.0] > charge[1.28] > charge[0.26]
    assert charge[568.0] < 1000., "the excursion charge is unbounded again"
