# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lem.model.contact_head import PATCH_NAMES
from mimo_bridge.latent_body_runtime import ShadowRuntime, observation
from mimo_bridge.latent_body_shadow import add_arguments, observe_shadow, _RUNTIMES
from mimo_bridge.tests.latent_body_fixtures import config, observation as obs, order, tree
from mimo_bridge.transition import TransitionV1


def arguments(tmp_path):
    parser = argparse.ArgumentParser(); add_arguments(parser)
    args = parser.parse_args([])
    args.latent_body_hierarchy = "shadow"
    args.latent_body_config = str(tmp_path / "config.json")
    args.latent_body_output = str(tmp_path / "shadow")
    args.out = str(tmp_path / "production")
    args.eval_only, args.report_every = False, 1
    (tmp_path / "config.json").write_text(json.dumps(config()), encoding="utf-8")
    return args


def frame(step):
    value = obs(step)
    contact = [0] * len(PATCH_NAMES)
    contact[PATCH_NAMES.index("left_ffknuckle")] = step % 2
    return SimpleNamespace(env_index=0, sim_time=value.sim_time, bones=value.bones.numpy().copy(),
                           meta={"task": {"contact_patch_state": contact}})


def transition(step, source="real"):
    return TransitionV1(env_id=0, episode_id=0, frame_t=step-1, frame_t1=step, duration_s=.05,
        state_t={}, state_t1={}, achieved_t=None, context_t=None, proposed_action=None,
        executed_action=order(obs(step-1)).numpy(), source_mask_t={"proprio": source},
        source_mask_t1={"proprio": "real"})


def test_shadow_records_real_outcomes_learns_and_resumes_without_mutating_inputs(tmp_path):
    args = arguments(tmp_path)
    runtime = ShadowRuntime(args, tree().handle_names)
    for step in range(3):
        physical = frame(step); before = physical.bones.copy()
        runtime.observe([physical], [] if step == 0 else [transition(step)], step, 0, train=True)
        assert np.array_equal(physical.bones, before)
    assert runtime.learner.updates == 1 and runtime.latest.is_file()
    rows = (runtime.directory / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["start"]["frame"] == 1
    assert "ledger" in json.loads(rows[0])
    with pytest.raises(ValueError, match="explicit"):
        ShadowRuntime(args, tree().handle_names)
    args.latent_body_resume = str(runtime.latest)
    resumed = ShadowRuntime(args, tree().handle_names)
    resumed.observe([frame(3)], [transition(3)], 3, 0, train=True)
    assert resumed.learner.updates == 2


def test_shadow_planner_logs_unused_command_and_retains_production_order(tmp_path):
    args = arguments(tmp_path)
    args.latent_planner = "shadow"
    args.latent_body_candidates, args.latent_body_radius = 3, .5
    args.latent_body_goal = str(tmp_path / "goal.json")
    goals = {r: [[0.] * 6 for _ in tree().subtree[r]] for r in tree().roots}
    (tmp_path / "goal.json").write_text(json.dumps(goals), encoding="utf-8")
    runtime = ShadowRuntime(args, tree().handle_names)
    runtime.observe([frame(0)], [], 0, 0, train=False)
    tr = transition(1); before = tr.executed_action.copy()
    runtime.observe([frame(1)], [tr], 1, 0, train=False)
    record = json.loads((runtime.directory / "plans.jsonl").read_text(encoding="utf-8"))
    assert record["authority"] == "none" and not record["physical_execution"]
    assert np.array_equal(before, tr.executed_action) and runtime.learner.updates == 0


def test_bad_skin_or_synthetic_outcomes_cannot_train_the_shadow(tmp_path):
    physical = frame(0); physical.meta["task"]["contact_patch_state"] = []
    with pytest.raises(ValueError, match="contact"):
        observation(physical, tree(), 0, 0)
    runtime = ShadowRuntime(arguments(tmp_path), tree().handle_names)
    runtime.observe([frame(0)], [], 0, 0, train=True)
    with pytest.raises(ValueError, match="Synthetic"):
        runtime.observe([frame(1)], [transition(1, "synthetic")], 1, 0, train=True)
    assert runtime.learner.updates == 0


def test_failure_disables_only_the_shadow_and_writes_a_receipt(tmp_path):
    args = arguments(tmp_path)
    model = torch.nn.Linear(2, 2); model.handle_names = tree().handle_names
    observe_shadow(args, model, [frame(0)], [], 0, 0)
    observe_shadow(args, model, [frame(1)], [transition(1, "synthetic")], 1, 0)
    assert _RUNTIMES[model] is False
    events = [json.loads(line) for line in (tmp_path / "shadow" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "failed" and events[-1]["saved_updates"] == 0
    assert (tmp_path / "shadow" / "latest.pt").is_file()
