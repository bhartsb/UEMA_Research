# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The planner runs on its schedule with every module in train mode (audit P113)."""
import torch

from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite


def test_planner_runs_on_its_schedule_with_every_module_in_train_mode(tmp_path):
    """No eval gate: the planner runs with the forward model, the encoder
    registry and the prediction heads all in train mode. Its only guard is
    `len(goal_ring) >= 20 and step % intention_every == 0`, so with
    --intention-every 5 the first two planning events are steps 20 and 25.
    last_intention is NOT cleared between events - it holds the last plan until
    the next planner step replaces it - so the schedule is read from the
    recorded plan inputs, never from last_intention being None."""
    torch.set_num_threads(1)
    m = LemLite(device="cpu", seed=0, intention=True, intention_execute=False,
                imagination=False, no_vision=True,
                overrides={"training": {"steps_per_phase": 3000}})
    assert m.intention_every == 5 and m.goal_ring.maxlen == 200

    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "psched"), tag="_ps",
                          vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        ran, prev = [], None
        for i in range(26):
            m._last_plan_inputs = None          # the previous event's record
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            assert (m.forward_model.training and m.encoder_registry.training
                    and m.prediction_heads.training), f"step {i}: a module left train mode"
            if m._last_plan_inputs is not None:
                ran.append(i)
            if i < 20:
                assert m.last_intention is None, \
                    f"step {i}: a plan exists before the goal ring holds 20 moments"
            frames = src.step(res.motor_for_action, res.gaze_out, None, render=False)
            prev = res.motor_for_action
        assert ran == [20, 25], f"planner ran on {ran}; expected [20, 25]"
        if m.last_intention is not None:
            assert m.last_intention["step"] in ran
            assert m.last_intention["horizon"] == 1
    finally:
        src.close()


# ---------------------------------------------------------------- audit P114 (2026-09-19)
def test_the_bridge_forces_the_planner_constants_the_page_publishes():
    """Audit P114. The only other test of the planner's constants
    (lem/tests/test_intention_generator.py TestConfigParsing) asserts the master yaml's
    50 Adam rounds and 5 plan steps, which no bridge run uses: LemLite.__init__
    overwrites them. These are the values every bridge run plans with, and the ones the
    explainer page restates in 12 sentences, one tour slide and one mini-diagram."""
    import inspect
    import re
    m = LemLite(device="cpu", seed=0, intention=True, intention_execute=False,
                imagination=False)
    ig = m.intention_gen
    assert ig._n_steps == 20, "Adam rounds per planning call"
    assert ig._lr == 0.1, "the planner's Adam learning rate"
    assert ig._strength_reg == 0.01, "the effort charge per unit of squared strength"
    assert m.config["training"]["intention"]["max_plan_steps"] == 1, \
        "fixed-mode plan depth: depth 2 plans against the pose from before the rollout"
    assert m._plan_settings()[:3] == (20, 0.1, 0.01)
    # the ring and the three deques stored beside it must roll over together, or the
    # per-moment curiosity factor and consequence count stop lining up with the moment
    for ring in (m.goal_ring, m._goal_ring_w0, m._goal_ring_cq, m._goal_ring_drive):
        assert ring.maxlen == 200, \
            "the goal ring or one of the three deques stored beside it is no longer 200 moments long"
    step_src = inspect.getsource(LemLite.step)
    fill = re.findall(r"len\(self\.goal_ring\) >= (\d+) and step % self\.intention_every == 0",
                      step_src)
    assert fill == ["20"], "the planner's fill gate moved or was duplicated"
    assert "best_t = wl_stack.argmax(dim=0)" in step_src, \
        "goal selection is no longer the highest stored value per body"


def test_the_planner_aims_at_the_highest_stored_value_and_hands_the_body_one_order(tmp_path):
    """Audit P114/P117, behavioural. 'The planner takes the highest stored value in the
    ring as its goal' and 'the winning order is one row per handle x 9' are page claims
    nothing drove. A ring entry planted at 100.0 - 33 times the 3.02 to 3.06 the real
    moments of this drive store - must become the goal, and the plan's first order must
    have the body's shape."""
    torch.set_num_threads(1)
    m = LemLite(device="cpu", seed=0, intention=True, intention_execute=False,
                imagination=False, no_vision=True, intention_every=5,
                overrides={"training": {"steps_per_phase": 3000}})
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "goalpick"), tag="_gp",
                          vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        prev = None
        for i in range(4):                       # the first step clears the ring (reset_context)
            res = m.step(frames, dev, train=False, step_index=i, executed_action=prev)
            frames = src.step(res.motor_for_action, res.gaze_out, None, render=False)
            prev = res.motor_for_action
        assert getattr(m, "_last_plan_inputs", None) is None, \
            "a plan ran before the ring held 20 moments"
        planted = {mod: torch.full((1, m.d_model), 0.25 + 0.01 * k)
                   for k, mod in enumerate(m.active)}
        for k in range(30):                      # 29 ordinary moments and one planted peak
            toks = planted if k == 7 else {mod: torch.zeros(1, m.d_model) for mod in m.active}
            m.goal_ring.append((toks, torch.tensor([100.0 if k == 7 else 1.0])))
        m.step(frames, dev, train=False, step_index=5, executed_action=prev)
        pi = m._last_plan_inputs
        assert pi is not None, "step 5 is a planner step and the ring holds more than 20"
        for mod in m.active:
            assert torch.equal(pi["goal"][mod], planted[mod]), \
                f"the goal for {mod} is not the ring's highest stored value"
        # measured 2026-09-19 at this seed: a plan was made and its first order was
        # (1, 24, 9). A None here means the scale gate or the 0.01 arrival line dropped
        # the plan, and the shape claim below would then be checked against nothing.
        assert pi["plan0"] is not None, "no order came out of the planner on this drive"
        assert tuple(pi["plan0"].shape) == (1, len(m.handle_names), 9), \
            "the order handed toward the body is not one row per handle x 9"
    finally:
        src.close()


# ---------------------------------------------------------------- H84 items (2026-09-19)
def test_the_pages_latch_arithmetic_matches_a_step_by_step_count_of_the_two_conditions():
    """The page says who can write the order the body executes from two launch flags:
    1 latched order in every (--intention-every / gcd) falls on a planner step, and 1
    plan in every (--motor-hold / gcd) falls on a latch step. H81's page typed the 5-and-5
    case as a fact of every run; H84 runs 10 and 5, where every second latched order has no
    plan and carries exploration noise. This holds schema.planner_schedule to the two
    conditions the code actually evaluates, counted step by step, and first checks that
    those two conditions are still the ones in the code and still share one step counter."""
    import inspect
    import os
    import re
    from mimo_bridge.explainer import schema as SCH
    from mimo_bridge import lem_model as LM
    step_src = inspect.getsource(LM.LemLite.step)
    assert "len(self.goal_ring) >= 20 and step % self.intention_every == 0" in step_src
    assert "if intent_action is None and self.wellness_mode != WellnessMode.DIRECT:" in step_src, \
        "exploration noise is no longer limited to orders the planner did not write"
    tm = open(os.path.join(os.path.dirname(inspect.getfile(LM)), "train_mimo.py"),
              encoding="utf-8").read()
    assert "hold = max(1, args.motor_hold)" in tm
    assert "if step % hold == 0 or held_motor is None:" in tm
    assert re.search(r"model\.step\(frames, dev, train=not args\.eval_only, step_index=step,", tm), \
        "the latch and the planner no longer read one step counter: the arithmetic is void"

    for every, hold in ((10, 5), (5, 5), (7, 5), (1, 5), (3, 6), (4, 6), (6, 4), (12, 5)):
        steps = range(every * hold * 40)
        latches = [s for s in steps if s % hold == 0]
        plans = [s for s in steps if s % every == 0]
        planner_latches = [s for s in latches if s % every == 0]
        latched_plans = [s for s in plans if s % hold == 0]
        sch = SCH.planner_schedule({"intention": "on", "intention_every": every,
                                    "motor_hold": hold})
        assert len(latches) == sch["latches_per_planner_latch"] * len(planner_latches), (every, hold)
        assert len(plans) == sch["plans_per_latched_plan"] * len(latched_plans), (every, hold)

    # what the page then SAYS for the three launches this project has run
    def said(every, hold):
        return SCH.run_fill({}, {"intention": "on", "intention_every": every,
                                 "motor_hold": hold}, "RUNP")

    h84 = said(10, 5)
    assert "every plan lands on a latch step" in h84["{planner_latch_rule}"]
    assert "Of every 2 latched commands 1 falls on a planner step and 1 does not" in h84["{planner_latch_rule}"]
    assert h84["{babble_latch_share}"].startswith("1 of every 2 latched commands in RUNP")
    h81 = said(5, 5)
    assert "every latch step is also a planner step" in h81["{planner_latch_rule}"]
    assert h81["{babble_latch_share}"].startswith("none of the latched commands in RUNP")
    h82 = said(7, 5)
    # lem_model.py's own comment records 29 of 143 plans reaching the body in 1,000 steps at
    # these two values - 1 plan in 5
    assert "1 plan in 5 lands on a latch step" in h82["{planner_latch_rule}"]
    assert "Of every 7 latched commands 1 falls on a planner step and the other 6 do not" \
        in h82["{planner_latch_rule}"]
    assert len([s for s in range(1000) if s % 7 == 0 and s % 5 == 0]) == 29
    # the launcher's own default is what an unrecorded flag falls back to, and the page says so
    assert re.search(r'"--intention-every", type=int, default=(\d+)', tm).group(1) \
        == str(SCH.INTENTION_EVERY_DEFAULT)
    assert "launcher's default" in SCH.run_fill({}, {}, "RUNP")["{intention_every_source}"]
    assert SCH.run_fill({}, {"intention_every": 10}, "RUNP")["{intention_every_source}"] == ""
