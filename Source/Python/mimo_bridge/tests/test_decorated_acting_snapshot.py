# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Additive code binding and actual native shadow preservation regressions."""
import copy
import json
from types import ModuleType

import numpy as np
import pytest
import torch

from mimo_bridge import execution_contract as old_execution
from mimo_bridge import continuation_state
from mimo_bridge.acting_snapshot import FrozenActingSnapshot,calibration_for_candidate
from mimo_bridge.decorated_acting_snapshot import (DecoratedActingSnapshot,function_binding,
    record_manifest,verify_manifest,wrap_shadow_state)
from mimo_bridge.tests.test_acting_snapshot import world,make_model,candidate_receipt


def test_decorated_inner_change_is_detected_when_legacy_wrapper_code_is_unchanged():
    module=ModuleType('constructed_decorated_module');module.torch=torch
    exec('@torch.no_grad()\ndef action():\n return 1\n',module.__dict__)
    old=old_execution._module_record(module);before=function_binding(module.action)
    replacement=lambda:2
    module.action.__wrapped__.__code__=replacement.__code__
    assert module.action()==2
    assert old_execution._module_record(module)==old
    assert function_binding(module.action)!=before


def test_primitive_closure_and_context_receiver_are_bound():
    def make(value):
        @torch.no_grad()
        def action():return value
        return action
    assert function_binding(make(1))!=function_binding(make(2))
    def function():return torch.is_grad_enabled()
    assert function_binding(torch.no_grad()(function))!=function_binding(torch.enable_grad()(function))


def test_missing_sources_and_changed_recorded_source_are_rejected(world):
    model=make_model();manifest=record_manifest(model)
    missing=copy.deepcopy(manifest);missing['sources']={}
    with pytest.raises(ValueError,match='incomplete'):verify_manifest(model,missing,require_source_files=True)
    changed=copy.deepcopy(manifest);next(iter(changed['sources'].values()))['sha256']='0'*64
    with pytest.raises(ValueError,match='recorded source changed'):verify_manifest(model,changed,require_source_files=True)


def test_shadow_wrapper_preserves_next_inference_and_old_reader_refuses_it(world):
    learner=make_model();frames=world.reset()
    actor=FrozenActingSnapshot.from_learner(learner,make_model,candidate_receipt(learner,frames[0]),{},qualification=True)
    state=actor.state_dict();manifest=record_manifest(actor.model)
    extended=wrap_shadow_state(state,actor.model,manifest)
    with pytest.raises(ValueError,match='incompatible'):FrozenActingSnapshot.restore(extended,make_model,'cpu',allow_qualification=True)
    with pytest.raises(ValueError,match='explicit shadow'):DecoratedActingSnapshot.restore(state,make_model,'cpu',allow_qualification=True)
    continued=DecoratedActingSnapshot.restore(extended,make_model,'cpu',allow_qualification=True)
    dev=learner.dev_limits(0)
    expected=actor.step(frames,dev,step_index=0,executed_action=None)
    actual=continued.step(frames,dev,step_index=0,executed_action=None)
    np.testing.assert_array_equal(expected.motor_for_action,actual.motor_for_action)
    with pytest.raises(ValueError,match='new physical qualification'):continued.promote(state['receipt'])
    with pytest.raises(ValueError,match='replacement'):continued.stage(learner,make_model,{})


def test_qualified_legacy_state_cannot_migrate(world):
    model=make_model();frames=world.reset()
    actor=FrozenActingSnapshot.from_learner(model,make_model,candidate_receipt(model,frames[0]),{},qualification=True)
    state=actor.state_dict();state['qualification']=False
    with pytest.raises(ValueError,match='only an unqualified'):wrap_shadow_state(state,model,record_manifest(model))


def test_native_selection_protects_births_and_requires_explicit_legacy_migration(world):
    from mimo_bridge.tools.actor_runtime import actor_for_run,require_outer_actor
    model=make_model();frames=world.reset();receipt=candidate_receipt(model,frames[0])
    fresh=actor_for_run(model,make_model,{},'shadow',receipt)
    assert isinstance(fresh,DecoratedActingSnapshot)
    restored=actor_for_run(model,make_model,{},'shadow',receipt,saved=fresh.state_dict())
    assert isinstance(restored,DecoratedActingSnapshot)
    legacy=FrozenActingSnapshot.from_learner(model,make_model,receipt,{},qualification=True)
    state=legacy.state_dict()
    old=actor_for_run(model,make_model,{},'shadow',receipt,saved=state)
    assert isinstance(old,FrozenActingSnapshot)
    with pytest.raises(ValueError,match='explicit decorated-code migration'):require_outer_actor(old)
    qualified=copy.deepcopy(state);qualified['qualification']=False
    with pytest.raises(ValueError,match='legacy qualified'):
        actor_for_run(model,make_model,{},'on',receipt,saved=qualified)
    with pytest.raises(ValueError,match='run-mode flag'):
        actor_for_run(model,make_model,{},'on',receipt,saved=state)


def test_current_hybrid_audit_cannot_be_promoted_or_restored_as_qualified(world):
    from lem.model.learning_opportunity import CONTRACT
    from mimo_bridge.tests.test_acting_snapshot import constructed_pass
    model=make_model();frames=world.reset();receipt=constructed_pass(model,frames[0])
    receipt['learning_opportunity']=dict(contract=CONTRACT,audit_every=1,anchor_capacity=2,
        window=2,effect_floors={'proprioception':1e-8},calibration='constructed refusal test')
    inner=FrozenActingSnapshot.from_learner(model,make_model,receipt,{},qualification=True)
    manifest=record_manifest(inner.model);outer=DecoratedActingSnapshot(inner,manifest)
    receipt['decorated_execution_sha256']=manifest['binding']['sha256']
    with pytest.raises(ValueError,match='hybrid opportunity audit'):outer.promote(receipt)
    inner.qualification=False;inner.receipt=receipt
    with pytest.raises(ValueError,match='hybrid opportunity audit'):DecoratedActingSnapshot(inner,manifest)
    inner.qualification=True
    resumed=DecoratedActingSnapshot.restore(outer.state_dict(),make_model,'cpu',allow_qualification=True)
    assert resumed.qualification and resumed.opportunity is not None


def test_native_migration_keeps_pending_audit_episode_and_all_saved_files(world,tmp_path):
    from mimo_bridge.train_mimo import parse_args,build_learner,acting_configuration
    from mimo_bridge.tools.prepare_decorated_shadow import record,prepare
    from mimo_bridge.tools.prepare_local_shadow import state_digest
    from lem.model.learning_opportunity import CONTRACT
    root=tmp_path/'source';root.mkdir()
    args=parse_args(['--out',str(root),'--device','cpu','--no-vision','--persistent-memory','on',
        '--self-state','perceive','--imagination','off','--next-frame-loss','on','--body-function','modulate'])
    model=build_learner(args,str(root/'checkpoints'));factory=lambda:build_learner(args,None)
    frames=world.reset();dev=model.dev_limits(0)
    receipt=calibration_for_candidate(model,candidate_receipt(model,frames[0]),acting_configuration(args))
    receipt['learning_opportunity']=dict(contract=CONTRACT,audit_every=1,anchor_capacity=2,window=2,
        effect_floors={'proprioception':1e-8},calibration='constructed migration test')
    actor=FrozenActingSnapshot.from_learner(model,factory,receipt,acting_configuration(args),qualification=True)
    executed=None
    for step in range(2):
        result=model.step(frames,dev,train=True,step_index=step,executed_action=executed)
        actor.step(frames,dev,step_index=step,executed_action=executed)
        executed=result.motor_for_action;frames=world.step(executed)
        actor.prepare_learning_audit(model,executed,step)
    assert actor.opportunity.completed==1 and actor.opportunity.pending is not None
    source=model.save_checkpoint(2)
    continuation_state.save(source,world,model,dict(step=2,acting_snapshot=actor.state_dict()),dict(self_choice='shadow'))
    (root/'args.json').write_text(json.dumps(dict(vars(args),self_choice='shadow')))
    manifest=tmp_path/'source_manifest.json';record(source,manifest)
    out=tmp_path/'extended/checkpoints/ckpt_step_2'
    report=prepare(source,out,manifest)
    before=continuation_state.load(source);after=continuation_state.load(out)
    outer=after['trainer']['acting_snapshot']
    after['trainer']['acting_snapshot']=outer['actor']
    assert state_digest(before)==state_digest(after)==report['unwrapped_episode_sha256']
    for name,digest in report['native_file_sha256'].items():
        assert continuation_state.file_sha256(out/name)==digest
        assert continuation_state.file_sha256(__import__('pathlib').Path(source)/name)==digest
    resumed=DecoratedActingSnapshot.restore(outer,factory,'cpu',allow_qualification=True)
    assert resumed.opportunity.pending is not None and resumed.opportunity.completed==1
    expected=actor.step(frames,dev,step_index=2,executed_action=executed)
    actual=resumed.step(frames,dev,step_index=2,executed_action=executed)
    np.testing.assert_array_equal(expected.motor_for_action,actual.motor_for_action)
    assert resumed.opportunity.completed==2


def test_cpu_probe_exceptions_preserve_original_rng_and_all_other_values():
    from pathlib import Path
    from mimo_bridge.tools.prepare_decorated_shadow import cpu_proof_episode,restore_cpu_probe_values
    from mimo_bridge.tools.prepare_local_shadow import state_digest
    cuda=[torch.arange(16,dtype=torch.uint8)]
    from mimo_bridge.acting_snapshot import configuration_fingerprint
    description={'dtype':'torch.bfloat16','unchanged_execution':{'code':'same'}}
    actor=dict(configuration={'device':'cuda:0'},receipt={'acting_configuration_sha256':'original',
        'model_execution_description':description,'model_execution_sha256':configuration_fingerprint(description)},
        causal={'cuda_rng':cuda,'remembered':torch.tensor([3.])},random={'cuda':cuda,'torch':torch.tensor([2])},
        opportunity={'pending':{'forecast':torch.tensor([4.])}})
    episode=dict(learner={'cuda_rng':cuda,'adam_related_history':torch.tensor([9.])},
        trainer={'acting_snapshot':{'actor':actor,'manifest':{'immutable':'value'}}},
        worlds=[{'env_kwargs':{'run_dir':'original_scene'},'world':{'caregiver':None},'payload':{'sim_time':7.}}])
    before=state_digest(episode)
    copied,archived=cpu_proof_episode(episode,Path('C:/isolated/checkpoints/ckpt_step_9'))
    assert state_digest(episode)==before
    assert copied['learner']['cuda_rng'] is None
    assert copied['trainer']['acting_snapshot']['actor']['random']['cuda'] is None
    new_receipt=copied['trainer']['acting_snapshot']['actor']['receipt']
    assert new_receipt['model_execution_description']==dict(description,dtype='torch.float32')
    assert new_receipt['model_execution_sha256']==configuration_fingerprint(new_receipt['model_execution_description'])
    assert new_receipt['model_execution_sha256']!=actor['receipt']['model_execution_sha256']
    assert state_digest(restore_cpu_probe_values(copied,archived))==before


def test_recorded_precision_uses_saved_configuration_without_gpu_execution():
    from mimo_bridge.tools.prepare_decorated_shadow import recorded_precision
    config={'model':{'precision':'bf16'}}
    assert recorded_precision(config,{'device':'cuda:0'})==torch.bfloat16
    assert recorded_precision(config,{'device':'cpu'})==torch.float32
    with pytest.raises(ValueError,match='precision'):
        recorded_precision({'model':{'precision':'unregistered'}},{'device':'cuda:0'})


def test_physical_taxel_slice_mapping_is_preserved_in_episode_digest():
    from mimo_bridge.tools.prepare_local_shadow import state_digest
    saved={'body_slices':{'arm':slice(10,20,None),'hand':slice(20,30,1)}}
    assert state_digest(saved)==state_digest(copy.deepcopy(saved))
    for changed in (slice(11,20,None),slice(10,21,None),slice(10,20,2)):
        assert state_digest(saved)!=state_digest({'body_slices':{'arm':changed,'hand':slice(20,30,1)}})
