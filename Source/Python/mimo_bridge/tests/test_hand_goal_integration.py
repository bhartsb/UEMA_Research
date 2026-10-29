# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The chosen hand state must survive planning, filtering and checkpoint reload."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from lem.model.hand_response import CalibratedHandResponse, content_hash
from mimo_bridge.hand_target_control import HandKinematics
from mimo_bridge.lem_model import LemLite
from test_hand_response import fixture as response_fixture


def planner(monkeypatch):
    from lem.model import predictive_state as PS
    torch.set_num_threads(1)
    model = LemLite(device='cpu', no_vision=True, imagination=False, next_frame_loss=True, intention=True,
        hand_target_mode='wrist_relative_v1', hand_goals='memory', smooth_actions=.35,
        smooth_source='executed', overrides={'model': {'context_length': 2}})
    geometry, receipt, _ = response_fixture()
    geometry['handle_names'] = model.handle_names
    receipt['geometry_sha256'] = content_hash(geometry)
    receipt['sha256'] = content_hash({k: v for k, v in receipt.items() if k != 'sha256'})
    model._hand_response_calibration = dict(receipt, calibration_file_sha256='a'*64)
    model.configure_hand_geometry(geometry, torch.zeros(2, 1))
    command = torch.zeros(2, len(model.handle_names), 9)
    command[..., 6] = 1.; command[..., 8] = .4; command[..., 7] = .2
    command = model.hand_kinematics.assemble_world(command, torch.full((2, 1), .3))
    achieved = torch.cat((command, torch.zeros_like(command[..., :1])), -1)
    model.last_achieved = achieved
    model._hand_current_angles = torch.zeros(2, 1, dtype=torch.float64)
    model._hand_current_velocities = torch.zeros(2, 1, dtype=torch.float64)
    previous = command.clone(); previous[..., :3] *= 200.
    model._smooth_exec_prev = model.hand_kinematics.assemble_world(previous, torch.zeros(2, 1), 1.)
    decision = model._capture_plan_state()
    target = model.hand_kinematics.assemble_world(achieved, torch.full((2, 1), .15))
    decision['hand_goal'] = dict(pose=target, valid=torch.tensor([True, False]),
                                ring_index=[0, -1], horizon_ms=250.)
    def project(self, orders, position_scale=200.):
        return np.array(orders), np.full((len(orders), 1), .3), []
    monkeypatch.setattr(HandKinematics, 'project_world_orders', project)
    def choose(current, goal, motor, **kwargs):
        space = kwargs['action_space']; space.init()
        return space.assemble()[0].detach(), dict(selection='best_evaluated_per_body_v1')
    monkeypatch.setattr(model.intention_gen, 'generate_action', choose)
    seen = []
    def transition(fm, enc, heads, state, executed, **kwargs):
        seen.append(executed.detach().clone())
        return PS.PredictiveState(tokens=state.tokens, context=state.context,
            achieved=state.achieved, domain='real', memory_bias=state.memory_bias)
    monkeypatch.setattr(PS, 'transition', transition)
    monkeypatch.setattr(PS, 'scale_report', lambda *args: {'all_in_band': True})
    monkeypatch.setattr(model.wellness, 'score_predicted_wellness',
                        lambda tokens, motor: torch.ones(motor.shape[0], 5))
    current = {sense: torch.zeros(2, model.config['model']['d_model']) for sense in model.active}
    goal = {sense: torch.ones_like(value) for sense, value in current.items()}
    return model, command, current, goal, decision, seen


def test_final_prediction_and_comparison_use_selected_hand_with_one_filter(monkeypatch):
    model, command, current, goal, decision, seen = planner(monkeypatch)
    model.hand_goals = 'off'
    baseline, before = model._plan_chain(current, goal, command, decision=decision)
    model.hand_goals = 'memory'
    chosen, after = model._plan_chain(current, goal, command, decision=decision)
    stage = after['hand_descent']
    assert stage['selected_iteration'][0] > 0 and stage['selected_iteration'][1] == 0
    assert stage['selected'][0] < stage['first'][0]
    executed = after['depth1']['exec_form']
    expected_joint = .35 * after['hand_joint_goal_raw']
    ref = model.hand_kinematics.assemble_world(executed, expected_joint, 1.)
    torch.testing.assert_close(executed, ref, atol=0, rtol=0)
    nonfingers = [i for i in range(len(model.handle_names)) if i != 1]
    torch.testing.assert_close(executed[:, nonfingers], before['depth1']['exec_form'][:, nonfingers], atol=0, rtol=0)
    torch.testing.assert_close(executed[..., 7:], before['depth1']['exec_form'][..., 7:], atol=0, rtol=0)
    torch.testing.assert_close(executed[1], before['depth1']['exec_form'][1], atol=0, rtol=0)
    encoded = executed.clone(); encoded[..., :3] /= 200.
    torch.testing.assert_close(seen[-1], encoded, atol=0, rtol=0)
    compared = model._causal_execution_candidates(chosen, decision,
        hand_joint_goals=[after['hand_joint_goal_raw']])
    torch.testing.assert_close(compared, executed, atol=0, rtol=0)
    # Explicit planner goals must never be recovered by another inverse solve.
    monkeypatch.setattr(HandKinematics, 'project_world_orders',
                        lambda *a, **kw: pytest.fail('selected hand was re-solved'))
    model._causal_score_candidates(chosen, after['depth1']['state'], decision,
                                  hand_joint_goals=[after['hand_joint_goal_raw']])
    torch.testing.assert_close(seen[-1], encoded, atol=0, rtol=0)


def test_missing_readings_and_legacy_memory_have_explicit_skip_reasons(monkeypatch):
    model, command, current, goal, decision, _ = planner(monkeypatch)
    decision['hand_current_velocities'] = None
    _, metadata = model._plan_chain(current, goal, command, decision=decision)
    assert metadata['hand_descent']['stopped_by'] == 'awaiting_native_joint_readings'
    decision['hand_goal']['valid'].fill_(False)
    _, metadata = model._plan_chain(current, goal, command, decision=decision)
    assert metadata['hand_descent']['stopped_by'] == 'no_valid_memory_goal'


def test_sensory_convergence_retains_existing_exploratory_behavior(monkeypatch):
    model, command, current, _, decision, _ = planner(monkeypatch)
    monkeypatch.setattr(model.intention_gen, 'generate_action',
                        lambda *a, **k: pytest.fail('sensory-converged body command was optimized'))
    plan, metadata = model._plan_chain(current, current, command, decision=decision)
    assert not plan and metadata['converged'] and not metadata.get('descent')
    assert metadata['hand_descent']['stopped_by'] == 'sensory_converged_existing_exploration_retained'
    model.hand_goals = 'off'
    baseline, before = model._plan_chain(current, current, command, decision=decision)
    assert baseline == plan and before['converged']


def test_frozen_response_receipt_binds_source_and_retains_coefficients(tmp_path, monkeypatch):
    model, _, _, _, _, _ = planner(monkeypatch)
    checkpoint = tmp_path/'checkpoint'; checkpoint.mkdir()
    weights = checkpoint/'model.safetensors'; weights.write_bytes(b'preserved model')
    receipt = copy.deepcopy(model._hand_response_calibration)
    receipt.pop('calibration_file_sha256')
    receipt['source_model_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
    receipt['sha256'] = content_hash({k: v for k, v in receipt.items() if k != 'sha256'})
    asset = tmp_path/'calibration.json'; asset.write_text(json.dumps(receipt), encoding='utf-8')
    model.configure_hand_response_calibration(asset)
    model._restore_hand_response_calibration(checkpoint)
    model.configure_hand_geometry(model._hand_geometry)
    saved = copy.deepcopy(model.receipt['hand_response'])
    (checkpoint/'model_receipt.json').write_text(json.dumps({'hand_response': saved}), encoding='utf-8')
    # A later checkpoint carries the same calibration while its learner changes.
    weights.write_bytes(b'further trained model')
    model._hand_response_calibration = None
    model._restore_hand_response_calibration(checkpoint)
    model.configure_hand_geometry(model._hand_geometry)
    torch.testing.assert_close(model.hand_response.coefficients, torch.tensor(receipt['coefficients'], dtype=torch.float64), atol=0, rtol=0)
    assert model.receipt['hand_response'] == saved
    model._hand_response_calibration['calibration_file_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='differs from the saved'):
        model._restore_hand_response_calibration(checkpoint)
    (checkpoint/'model_receipt.json').unlink()
    with pytest.raises(ValueError, match='different checkpoint weights'):
        model._restore_hand_response_calibration(checkpoint)
