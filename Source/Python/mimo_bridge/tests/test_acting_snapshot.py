# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Real inference/continuation tests; constructed receipts are not behavior evidence."""
import copy
import json
import numpy as np
import pytest
import torch

from mimo_bridge.acting_snapshot import (CONTRACT,FrozenActingSnapshot,calibration_for_candidate,
                                        configuration_fingerprint,rng_state)
from mimo_bridge import continuation_state
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite,command_for_encoder
from mimo_bridge.self_intention import fingerprint,perception_fingerprint


def test_unqualified_hand_stage_is_rejected_before_copying_or_acting():
    from types import SimpleNamespace
    learner=SimpleNamespace(hand_goals='memory')
    with pytest.raises(ValueError,match='calibrated hand-goal stage'):
        FrozenActingSnapshot(learner,{}, {},qualification=True)
    with pytest.raises(ValueError,match='calibrated hand-goal stage'):
        FrozenActingSnapshot.from_learner(learner,lambda:None,{}, {})
    actor=FrozenActingSnapshot.__new__(FrozenActingSnapshot);actor.model=learner
    with pytest.raises(ValueError,match='calibrated hand-goal stage'):
        actor.step([],{},step_index=0,executed_action=None)


def make_model(checkpoint=None):
    return LemLite(device='cpu',imagination=False,no_vision=True,next_frame_loss=True,
                   persistent_memory='on',self_state='perceive',intention=True,
                   smooth_actions=.35,checkpoint_dir=None if checkpoint is None else str(checkpoint))


def candidate_receipt(model,frame):
    return calibration_for_candidate(model,dict(minimum_resolution=.9,minimum_capacity=.8,
        maximum_uncertainty=10.,candidate_position_fraction=.02,candidate_rotation_fraction=.03,
        order_steps=2,orders=2,step_seconds=.01,body_function='modulate',
        body_function_profile=frame.meta['body_function']['profile'],motor_contract=frame.meta['motor_contract']),{})


def constructed_pass(model,frame):
    receipt=candidate_receipt(model,frame)
    receipt.update(schema='physical_self_choice_v1',held_out_worlds=['constructed-test-only'],
                   replay_verified=True,retention_pass=True,choice_gain_lower=.1,action_effect_gain_lower=.1)
    return receipt


@pytest.fixture
def world(tmp_path):
    torch.set_num_threads(1)
    source=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path/'world'),age=24,
                          caregiver_seconds=0,body_function='modulate',contact_profile='mechanical_v2')
    yield source
    source.close()


def test_learner_updates_do_not_change_actor_or_its_random_stream(world):
    learner=make_model();frames=world.reset();dev=learner.dev_limits(0)
    learner.step(frames,dev,train=True,step_index=0)
    frames=world.step(np.zeros((1,len(learner.handle_names),9),np.float32))
    actor=FrozenActingSnapshot.from_learner(learner,make_model,constructed_pass(learner,frames[0]),{})
    assert all(not p.requires_grad for m in actor.model._side_modules().values() if m is not None for p in m.parameters())
    fixed=perception_fingerprint(actor.model)
    duplicate=FrozenActingSnapshot.restore(actor.state_dict(),make_model,'cpu')
    executed=np.zeros((1,len(learner.handle_names),9),np.float32)
    executed[...,3:7]=[0,0,0,1];executed[...,8]=.1
    for step in range(1,5):
        before=perception_fingerprint(learner)
        result=learner.step(frames,dev,train=True,step_index=step,executed_action=executed)
        assert np.isfinite(result.loss)
        assert perception_fingerprint(learner)!=before
        random_before=rng_state(learner.device)
        actual=actor.step(frames,dev,step_index=step,executed_action=executed)
        random_after=rng_state(learner.device)
        assert torch.equal(random_before['torch'],random_after['torch'])
        np.testing.assert_array_equal(random_before['numpy'][1],random_after['numpy'][1])
        expected=duplicate.step(frames,dev,step_index=step,executed_action=executed)
        np.testing.assert_array_equal(actual.motor_for_action,expected.motor_for_action)
        assert perception_fingerprint(actor.model)==fixed
        torch.testing.assert_close(actor.model.self_history.events[-1]['action'],
                                   command_for_encoder(torch.from_numpy(executed)).reshape(1,-1),rtol=0,atol=0)
        assert actor.model.self_history is not learner.self_history
        executed=actual.motor_for_action.copy();frames=world.step(executed)


def test_actor_and_learner_resume_the_same_physical_episode(world,tmp_path):
    learner=make_model(tmp_path/'checkpoints');frames=world.reset();dev=learner.dev_limits(0)
    actor=FrozenActingSnapshot.from_learner(learner,make_model,constructed_pass(learner,frames[0]),{})
    executed=None
    for step in range(3):
        learner.step(frames,dev,train=True,step_index=step,executed_action=executed)
        executed=actor.step(frames,dev,step_index=step,executed_action=executed).motor_for_action
        frames=world.step(executed)
    path=learner.save_checkpoint(3)
    continuation_state.save(path,world,learner,dict(step=3,executed=executed,actor=actor.state_dict()),{})
    saved=continuation_state.load(path)
    original_history=copy.deepcopy(actor.model.self_history.state_dict())
    expected_learn=learner.step(frames,dev,train=True,step_index=3,executed_action=executed)
    expected=actor.step(frames,dev,step_index=3,executed_action=executed)
    expected_frame=world.step(expected.motor_for_action)[0]
    new_learner=make_model();new_learner.load_checkpoint(path)
    new_actor=FrozenActingSnapshot.restore(saved['trainer']['actor'],make_model,'cpu')
    torch.testing.assert_close(new_actor.model.self_history.state,original_history['state'],rtol=0,atol=0)
    frames=world.restore_full_state(saved['worlds'])
    continuation_state.restore_learner(new_learner,saved['learner'])
    actual_learn=new_learner.step(frames,dev,train=True,step_index=3,executed_action=executed)
    actual=new_actor.step(frames,dev,step_index=3,executed_action=executed)
    assert actual_learn.loss==expected_learn.loss
    assert perception_fingerprint(new_learner)==perception_fingerprint(learner)
    np.testing.assert_array_equal(actual.motor_for_action,expected.motor_for_action)
    np.testing.assert_array_equal(world.step(actual.motor_for_action)[0].bones,expected_frame.bones)
    assert len(new_learner.optimizer.state)==len(learner.optimizer.state)>0
    for old,new in zip(learner.optimizer.state.values(),new_learner.optimizer.state.values()):
        for key in old:torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)


def test_unqualified_candidate_cannot_be_promoted_by_saving_or_copying(world):
    model=make_model();frame=world.reset()[0];receipt=candidate_receipt(model,frame)
    actor=FrozenActingSnapshot.from_learner(model,make_model,receipt,{},qualification=True)
    assert actor.qualification
    with pytest.raises(ValueError,match='unqualified trial'):
        FrozenActingSnapshot.restore(actor.state_dict(),make_model,'cpu')
    with pytest.raises(ValueError,match='physical choice gains'):
        FrozenActingSnapshot.from_learner(model,make_model,receipt,{})
    with pytest.raises(ValueError,match='exact execution configuration'):
        FrozenActingSnapshot.from_learner(model,make_model,constructed_pass(model,frame),{'motor_hold':10})


def test_episode_reset_removes_actor_intention_and_cross_episode_causes(world):
    model=make_model();frames=world.reset();dev=model.dev_limits(0)
    actor=FrozenActingSnapshot.from_learner(model,make_model,constructed_pass(model,frames[0]),{})
    command=actor.step(frames,dev,step_index=0,executed_action=None).motor_for_action
    frames=world.step(command);actor.step(frames,dev,step_index=1,executed_action=command)
    assert actor.model.self_history.events
    actor.reset_transition_state();frames=world.reset()
    assert actor.model.self_history.intention is None
    actor.step(frames,dev,step_index=2,executed_action=None)
    assert not actor.model.self_history.events


def test_actor_weight_or_body_contract_mutation_is_refused(world):
    model=make_model();frames=world.reset();dev=model.dev_limits(0)
    actor=FrozenActingSnapshot.from_learner(model,make_model,constructed_pass(model,frames[0]),{})
    with torch.no_grad():next(actor.model.encoder_registry.parameters()).add_(.01)
    with pytest.raises(ValueError,match='contract changed'):
        actor.step(frames,dev,step_index=0,executed_action=None)
    with pytest.raises(ValueError,match='qualified weights'):actor.state_dict()


def test_promotion_keeps_candidate_own_history_after_learner_changed(world):
    learner=make_model();frames=world.reset();dev=learner.dev_limits(0)
    original=FrozenActingSnapshot.from_learner(learner,make_model,constructed_pass(learner,frames[0]),{})
    command=original.step(frames,dev,step_index=0,executed_action=None).motor_for_action
    learner.step(frames,dev,train=True,step_index=0)
    frames=world.step(command)
    original.stage(learner,make_model,candidate_receipt(learner,frames[0]))
    original.step(frames,dev,step_index=1,executed_action=command)
    state=original.state_dict()
    continued=FrozenActingSnapshot.restore(state,make_model,'cpu')
    assert continued.pending.qualification and continued.pending.steps==1
    candidate=continued.pending
    carried=candidate.model.self_history.state.clone()
    old_weights=perception_fingerprint(candidate.model)
    learner.step(frames,dev,train=True,step_index=1,executed_action=command)
    assert perception_fingerprint(learner)!=old_weights
    receipt=copy.deepcopy(candidate.receipt)
    receipt.update(schema='physical_self_choice_v1',held_out_worlds=['constructed-test-only'],
                   replay_verified=True,retention_pass=True,choice_gain_lower=.1,action_effect_gain_lower=.1)
    wrong=copy.deepcopy(receipt);wrong['candidate_position_fraction']*=2
    with pytest.raises(ValueError,match='calibrated decision rule'):continued.promote(wrong)
    assert candidate.qualification
    promoted=continued.promote(receipt)
    assert promoted is candidate and not promoted.qualification
    assert perception_fingerprint(promoted.model)==old_weights
    torch.testing.assert_close(promoted.model.self_history.state,carried,rtol=0,atol=0)


def test_unchanged_weights_do_not_allow_execution_mode_changes(world):
    learner=make_model();frames=world.reset()
    actor=FrozenActingSnapshot.from_learner(learner,make_model,constructed_pass(learner,frames[0]),{})
    actor.model.self_state_mode='learn'
    with pytest.raises(ValueError,match='execution configuration changed'):
        actor.step(frames,learner.dev_limits(0),step_index=0,executed_action=None)


def test_unchanged_weights_and_settings_cannot_inherit_a_changed_execution_method(world,monkeypatch):
    learner=make_model();frames=world.reset()
    actor=FrozenActingSnapshot.from_learner(learner,make_model,constructed_pass(learner,frames[0]),{})
    weights=perception_fingerprint(actor.model)
    monkeypatch.setattr(actor.model,'planned_execution',lambda *args,**kwargs:None)
    assert perception_fingerprint(actor.model)==weights
    with pytest.raises(ValueError,match='execution configuration changed'):
        actor.step(frames,learner.dev_limits(0),step_index=0,executed_action=None)


def test_changed_loaded_motor_table_invalidates_an_actor(world,monkeypatch):
    from mimo_bridge import action_decoder
    learner=make_model();frames=world.reset()
    actor=FrozenActingSnapshot.from_learner(learner,make_model,constructed_pass(learner,frames[0]),{})
    weights=perception_fingerprint(actor.model)
    handle=next(iter(action_decoder.HANDLE_CHAINS))
    monkeypatch.setitem(action_decoder.HANDLE_CHAINS,handle,('a different physical chain',))
    assert perception_fingerprint(actor.model)==weights
    with pytest.raises(ValueError,match='execution configuration changed'):
        actor.step(frames,learner.dev_limits(0),step_index=0,executed_action=None)


def test_learning_audits_use_frozen_inputs_future_targets_and_survive_resume(world):
    from lem.model.learning_opportunity import CONTRACT as OPPORTUNITY_CONTRACT
    learner=make_model();frames=world.reset();dev=learner.dev_limits(0)
    receipt=constructed_pass(learner,frames[0])
    receipt['learning_opportunity']=dict(contract=OPPORTUNITY_CONTRACT,audit_every=1,anchor_capacity=2,
        window=2,effect_floors={'proprioception':1e-8},calibration='constructed test budget and floor only')
    actor=FrozenActingSnapshot.from_learner(learner,make_model,receipt,{})
    command=None
    for step in range(3):
        learner.step(frames,dev,train=True,step_index=step,executed_action=command)
        command=actor.step(frames,dev,step_index=step,executed_action=command).motor_for_action
        frames=world.step(command)
        actor.prepare_learning_audit(learner,command,step)
        pending=actor.opportunity.pending
        assert float(pending['inputs']['time'][0])<frames[0].sim_time
        assert not pending['prediction']['proprioception'].requires_grad
    state=actor.state_dict();new=FrozenActingSnapshot.restore(state,make_model,'cpu')
    assert new.opportunity.completed==2 and len(new.opportunity.anchors)==2
    assert new.opportunity.meter.best==actor.opportunity.meter.best
    expected=actor.step(frames,dev,step_index=3,executed_action=command)
    actual=new.step(frames,dev,step_index=3,executed_action=command)
    np.testing.assert_array_equal(expected.motor_for_action,actual.motor_for_action)
    assert new.opportunity.meter.available(0,'proprioception')==actor.opportunity.meter.available(0,'proprioception')
    assert new.opportunity.completed==3


def test_reference_refresh_and_restore_use_new_embeddings_after_warm_inference(world):
    from lem.model.learning_opportunity import CONTRACT as OPPORTUNITY_CONTRACT
    learner=make_model();frames=world.reset();dev=learner.dev_limits(0)
    result=learner.step(frames,dev,train=False,step_index=0)
    receipt=constructed_pass(learner,frames[0])
    receipt['learning_opportunity']=dict(contract=OPPORTUNITY_CONTRACT,audit_every=1,anchor_capacity=2,
        window=2,effect_floors={'proprioception':1e-8},calibration='constructed cache refresh test only')
    actor=FrozenActingSnapshot.from_learner(learner,make_model,receipt,{})
    audit=actor.opportunity;inputs=audit.inputs()
    command=command_for_encoder(torch.as_tensor(result.motor_for_action))
    before=audit.predict(audit.reference,audit.reference_heads,inputs,command)
    assert audit.reference._emb_cache_valid
    with torch.no_grad():
        learner.forward_model.modality_embedding.weight.add_(torch.randn_like(learner.forward_model.modality_embedding.weight))
        learner.forward_model.temporal_embedding.weight.add_(torch.randn_like(learner.forward_model.temporal_embedding.weight))
    learner.forward_model.invalidate_embedding_cache()
    learner.forward_model.eval();learner.prediction_heads.eval()
    # Construct the branch prerequisite; no behavioral learning qualification is
    # claimed by this regression. Native forward predictions remain unmocked.
    audit.meter.retention['cache-test']={'preserved':True,'improved':True}
    audit.prepare(learner,result.motor_for_action,0)
    assert not audit.reference._emb_cache_valid
    expected=audit.predict(learner.forward_model,learner.prediction_heads,inputs,command)
    actual=audit.predict(audit.reference,audit.reference_heads,inputs,command)
    assert any(not torch.equal(before[key],actual[key]) for key in actual)
    for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    saved=audit.state_dict()
    with torch.no_grad():audit.reference.modality_embedding.weight.add_(torch.randn_like(audit.reference.modality_embedding.weight))
    audit.reference.invalidate_embedding_cache()
    changed=audit.predict(audit.reference,audit.reference_heads,inputs,command)
    assert audit.reference._emb_cache_valid
    assert any(not torch.equal(changed[key],expected[key]) for key in changed)
    audit.load_state_dict(saved)
    assert not audit.reference._emb_cache_valid
    restored=audit.predict(audit.reference,audit.reference_heads,inputs,command)
    for key in restored:torch.testing.assert_close(restored[key],expected[key],rtol=0,atol=0)


@pytest.mark.parametrize('choice_mode',['on','shadow'])
def test_production_trainer_uses_actor_and_resumes_both_while_learning(tmp_path,monkeypatch,choice_mode):
    from mimo_bridge.train_mimo import main,parse_args,build_learner,acting_configuration,env_kwargs_for
    # Isolate this real trainer integration from the separate startup audits.
    from mimo_bridge.tools import preflight
    monkeypatch.setattr(preflight,'run_probes',lambda path,env_kw=None:dict(ok=True,checks=[]))
    torch.set_num_threads(1)
    flags=['--out',str(tmp_path/'run'),'--device','cpu','--n-envs','1','--no-vision',
           '--curriculum','off','--age-binding','none','--delay-binding','off','--episode-steps','0',
           '--action-repeat','1','--motor-hold','2','--minutes','0','--max-steps','2','--no-babble',
           '--self-state','perceive','--body-function','modulate','--persistent-memory','on',
           '--imagination','off','--next-frame-loss','on','--contact-profile','mechanical_v2',
           '--replay','off','--report-every','1','--video-every-min','0']
    args=parse_args(flags);learner=build_learner(args,str(tmp_path/'initial'))
    checkpoint=learner.save_checkpoint(0)
    source=SingleEnvSource(**env_kwargs_for(args,None,str(tmp_path/'receipt_world')))
    try:
        receipt=constructed_pass(learner,source.reset()[0])
    finally:source.close()
    from mimo_bridge.decorated_acting_snapshot import loaded_binding
    receipt['decorated_execution_sha256']=loaded_binding(learner)['sha256']
    receipt['acting_configuration_sha256']=configuration_fingerprint(acting_configuration(args))
    receipt_file=tmp_path/'constructed_test_receipt.json';receipt_file.write_text(json.dumps(receipt))
    flags+=['--self-choice',choice_mode,'--self-choice-receipt',str(receipt_file),'--resume',checkpoint]
    if choice_mode=='shadow':flags+=['--checkpoint-storage',str(tmp_path/'external_storage'/'checkpoints')]
    assert main(flags)==0
    saved_checkpoint=(tmp_path/('external_storage' if choice_mode=='shadow' else 'run')/'checkpoints'/'ckpt_step_2')
    saved=continuation_state.load(saved_checkpoint)
    from mimo_bridge.tools.actor_runtime import inner_state_for_report
    saved_actor=inner_state_for_report(saved['trainer']['acting_snapshot'])
    assert saved_actor['steps']==2
    assert saved_actor['born_step']==0
    records=[json.loads(line) for line in (tmp_path/'run'/'replay_training_stats.jsonl').read_text().splitlines()]
    assert all(r['acting_snapshot']['qualified']==(choice_mode=='on') for r in records if r['type']=='mimo')
    assert all(r['acting_snapshot']['applied_to_body']==(choice_mode=='on') for r in records if r['type']=='mimo')
    flags[flags.index('--max-steps')+1]='4'
    flags[flags.index('--out')+1]=str(tmp_path/'continued')
    flags[flags.index('--resume')+1]=str(saved_checkpoint)
    if choice_mode=='shadow':
        flags[flags.index('--checkpoint-storage')+1]=str(tmp_path/'continued_external'/'checkpoints')
    assert main(flags)==0
    continued_root=tmp_path/('continued_external' if choice_mode=='shadow' else 'continued')
    continued=continuation_state.load(continued_root/'checkpoints'/'ckpt_step_4')
    assert json.loads((continued_root/'args.json').read_text())['max_steps']==4
    continued_actor=inner_state_for_report(continued['trainer']['acting_snapshot'])
    assert continued_actor['steps']==4
    assert continued_actor['born_step']==0
    assert continued['worlds'][0]['payload']['sim_time']>saved['worlds'][0]['payload']['sim_time']
