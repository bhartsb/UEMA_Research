# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import numpy as np
import pytest
import torch
from lem.body_function import BodyFunction,FunctionProfile
from lem.model.self_state import SelfState,SelfHistory,SITE_INPUT
from lem.model.homeostatic_planner import FunctionCriterion,choose_sequence


def test_impairment_loses_a_distinction_and_recovery_restores_it_locally():
    b=BodyFunction(['left','right'])
    raw=np.zeros((2,10),np.float32);raw[:,0]=12345
    alternate=raw.copy();alternate[0,0]+=2
    np.testing.assert_array_equal(b.observe(raw),raw)
    b.advance([1,0],[0,0],0);b.advance([1,0],[0,0],2)
    assert b.observe(raw)[0,0]==b.observe(alternate)[0,0]
    np.testing.assert_array_equal(b.observe(raw)[1],raw[1])
    b.advance([0,0],[0,0],200)
    assert b.observe(raw)[0,0]!=b.observe(alternate)[0,0]
    np.testing.assert_array_equal(raw[:,0],[12345,12345])


def test_sensor_subdivision_and_more_real_regional_sensitivity_are_distinct():
    a=BodyFunction(['a']);b=BodyFunction(['a1','a2','b'])
    for t in (0,.2,1):
        a.advance([.6],[.1],t);b.advance([.6,.6,.9],[.1,.1,.1],t)
    np.testing.assert_allclose(b.quality[:2],a.quality[0])
    assert b.quality[2]<b.quality[0]


def test_positive_stimulation_improves_actual_resolution_with_headroom():
    a=BodyFunction(['a']);b=BodyFunction(['a'])
    for obj in (a,b):obj.advance([1],[0],0);obj.advance([1],[0],1)
    for t in (2,5,10,20):
        a.advance([.5],[0],t);b.advance([.5],[1],t)
    assert 0<b.quality[0]-a.quality[0]<=b.profile.positive_fraction
    assert b.quality[0]<=1


def test_body_function_roundtrip_and_reversed_time_refused():
    b=BodyFunction(['a']);b.advance([.8],[.1],1);b.advance([.8],[.1],2)
    c=BodyFunction(['a']);c.load_state_dict(b.state_dict())
    np.testing.assert_array_equal(b.advance([0],[0],3),c.advance([0],[0],3))
    with pytest.raises(ValueError):c.advance([0],[0],1)
    with pytest.raises(ValueError):c.advance([np.nan],[0],4)


def percept(m,b=2,n=3):
    return dict(sites=torch.randn(b,n,SITE_INPUT),valid=torch.ones(b,n,dtype=torch.bool),
                context=torch.randn(b,m.n_modalities*m.d_model))


def test_self_state_neutral_extension_and_no_cross_body_influence():
    torch.manual_seed(2);m=SelfState(16,2,9);inputs=percept(m)
    a=m.perceive(**inputs);assert a['bias'].count_nonzero()==0
    other=copy.deepcopy(inputs);other['sites'][1]+=100
    b=m.perceive(**other)
    torch.testing.assert_close(a['belief'][0],b['belief'][0],rtol=0,atol=0)
    assert not torch.allclose(a['belief'][1],b['belief'][1])


def test_future_gradient_uses_actions_and_memory_and_rejects_missing_data():
    m=SelfState(16,2,9);inputs=percept(m)
    inputs['previous']=torch.randn(2,16,requires_grad=True)
    actions=torch.randn(2,5,9,requires_grad=True)
    mean,var=m.forecast(m.perceive(**inputs),actions,torch.full((2,5),.05))
    m.consequence_loss(mean,var,torch.zeros_like(mean),torch.ones(2,5,3,dtype=torch.bool)).backward()
    assert actions.grad[:,0].abs().sum()>0
    assert inputs['previous'].grad.abs().sum()>0
    inputs['valid'][0]=False
    with pytest.raises(ValueError):m.perceive(**inputs)


def test_whole_trajectory_credits_a_detour_and_uncertainty_cannot_pick_zero():
    # Three constructed resolution trajectories: stand, immediate relief, detour.
    q=torch.tensor([[.6,.6,.6],[.8,.5,.4],[.55,.9,1.]])
    mean=torch.zeros(1,3,3,1,3);mean[0,:,:,0,0]=q
    v=torch.full_like(mean,.001);valid=torch.ones(1,1,dtype=torch.bool);dt=torch.ones(1,3,3)
    criterion=FunctionCriterion(.8,.1,'constructed test only')
    assert choose_sequence(mean,v,valid,dt,criterion)['index'].item()==2
    assert choose_sequence(mean,100*v,valid,dt,criterion)['index'].item()==-1
    doubled=choose_sequence(mean.repeat(1,1,1,2,1),v.repeat(1,1,1,2,1),valid.repeat(1,2),dt,criterion)
    torch.testing.assert_close(doubled['cost'],choose_sequence(mean,v,valid,dt,criterion)['cost'])


def test_saved_self_history_owns_its_inputs():
    h=SelfHistory(3);h.inputs=dict(sites=torch.ones(1,2,SITE_INPUT));h.relations.append(torch.ones(1,4))
    s=h.state_dict();h.inputs['sites'].zero_()
    r=SelfHistory(3);r.load_state_dict(s,'cpu')
    assert r.inputs['sites'].eq(1).all()


def test_rehearsal_retains_completed_past_without_claiming_new_episode_continuity():
    from lem.model.self_rehearsal import SelfRehearsal
    r=SelfRehearsal(4)
    for i in range(30):
        sites=torch.zeros(1,2,SITE_INPUT)
        r.add([dict(inputs=dict(sites=sites),action=torch.full((1,9),float(i)),
                    dt=torch.tensor([.05]),target=torch.full((1,2,3),i/30),valid=torch.ones(1,2,dtype=torch.bool))])
    assert len(r.common)+len(r.changed)==4
    restored=SelfRehearsal(4);restored.load_state_dict(r.state_dict())
    torch.testing.assert_close(r.sample('cpu')['actions'][0],restored.sample('cpu')['actions'][0])


def test_multiple_minds_cannot_multiply_one_physical_need_or_read_current_answers():
    from lem.model.self_sharing import CompletedSelf,read_completed,unique_physical_function
    a=CompletedSelf('one_body','first',1.,torch.ones(4),('left',),torch.tensor([[.4]]))
    b=CompletedSelf('one_body','second',1.,torch.ones(4)*2,('left',),torch.tensor([[.4]]))
    assert len(unique_physical_function([a,b]))==1
    assert len(read_completed([a,b],2.))==2
    with pytest.raises(ValueError):read_completed([a],1.)


def test_ue_dense_sites_keep_geometry_and_actual_modulated_experience():
    from mimo_bridge.skin_contract import SkinLayout
    from mimo_bridge.datasource import BridgeFrame
    from mimo_bridge.self_state_integration import batch_observation
    n=640;ids=tuple(str(i) for i in range(n));pos=np.random.default_rng(3).normal(size=(n,3)).astype(np.float32)
    layout=SkinLayout(ids,pos,np.ones((n,2),np.float32),'surface_sites')
    frame=BridgeFrame(None,b'',None,None,1.,.01,1,0,np.zeros((n,10)),np.zeros((24,11)),
                      np.zeros(8),((0,0),(0,0)),(0,0),np.zeros(2),0,{})
    func=BodyFunction(ids);func.advance(np.ones(n),np.zeros(n),0)
    fields=np.zeros((n,8),np.float32);fields[:,0]=1e6
    attached=layout.attach(frame,fields,body_function=func,mechanical_history=np.zeros((n,3)))
    sites,valid,now=batch_observation([attached],'cpu')
    assert valid.sum()==n and sites.shape==(1,n,SITE_INPUT)
    np.testing.assert_array_equal(sites[0,:,:3],pos)
    assert attached.meta['physical_touch_audit'][0,0]==1e6
    assert sites[0,0,-3]<1


@pytest.mark.parametrize('seed',[11,31,51])
def test_action_conditioned_self_prediction_learns_a_planted_reversible_effect(seed):
    # Constructed causal plant, not a claim about MIMo. The advantageous command
    # changes with the current bodily condition; an action-independent guess fails.
    torch.set_num_threads(1);torch.manual_seed(seed)
    m=SelfState(16,1,2);opt=torch.optim.Adam(m.parameters(),lr=.004)
    def batch(n):
        s=torch.zeros(n,1,SITE_INPUT);s[:,:,0]=torch.rand(n,1)*2-1
        a=torch.rand(n,1,2)*2-1
        y=torch.zeros(n,1,1,3);y[...,0]=.5+.4*torch.tanh(3*s[:,None,:,0]*a[:,:,:1])
        return dict(sites=s,valid=torch.ones(n,1,dtype=torch.bool),context=torch.zeros(n,16)),a,y
    for _ in range(300):
        x,a,y=batch(48);mu,v=m.forecast(m.perceive(**x),a,torch.full((48,1),.05))
        loss=m.consequence_loss(mu,v,y,torch.ones(48,1,1,dtype=torch.bool))
        opt.zero_grad();loss.backward();opt.step()
    x,a,y=batch(256)
    with torch.no_grad():
        p=m.perceive(**x);mu,_=m.forecast(p,a,torch.full((256,1),.05))
        scrambled,_=m.forecast(p,a.flip(0),torch.full((256,1),.05))
    error=(mu[...,0]-y[...,0]).square().mean()
    blind=(y[...,0]-.5).square().mean();wrong=(scrambled[...,0]-y[...,0]).square().mean()
    assert error<.2*blind and error<.2*wrong


def test_nonfinite_validation_cannot_license_action(tmp_path):
    import json
    from mimo_bridge.self_intention import SelfIntention
    p=tmp_path/'receipt.json'
    r=dict(minimum_resolution=.8,maximum_uncertainty=.1,candidate_position_fraction=.02,
           candidate_rotation_fraction=.03,choice_gain_lower=float('nan'),action_effect_gain_lower=1)
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='finite'):SelfIntention('on',p)


def test_changed_sensory_encoder_with_frozen_self_predictor_withdraws_license(tmp_path):
    import json
    from types import SimpleNamespace
    from mimo_bridge.self_intention import SelfIntention,fingerprint,perception_fingerprint
    from mimo_bridge.action_decoder import MOTOR_CONTRACT
    m=SimpleNamespace(encoder_registry=torch.nn.Linear(2,2),self_state=torch.nn.Linear(2,2))
    m._side_modules=lambda:dict(self_state=m.self_state)
    record=dict(minimum_resolution=.8,maximum_uncertainty=.1,candidate_position_fraction=.02,
                candidate_rotation_fraction=.03,choice_gain_lower=.1,action_effect_gain_lower=.1,
                schema='physical_self_choice_v1',body_function='modulate',held_out_worlds=[4,5,6],
                replay_verified=True,retention_pass=True,body_function_profile={'version':'test'},
                self_weights_sha256=fingerprint(m.self_state),perception_weights_sha256=perception_fingerprint(m))
    record['motor_contract']=MOTOR_CONTRACT
    p=tmp_path/'receipt.json';p.write_text(json.dumps(record))
    chooser=SelfIntention('on',p)
    frames=[SimpleNamespace(meta={'body_function':{'mode':'modulate','profile':{'version':'test'}}})]
    frames[0].meta['motor_contract']=MOTOR_CONTRACT
    assert chooser.validation_matches(m,frames)
    with torch.no_grad():m.encoder_registry.weight.add_(.01)
    assert fingerprint(m.self_state)==record['self_weights_sha256']
    assert not chooser.validation_matches(m,frames)


def test_long_consequences_keep_real_time_and_survive_as_closed_rehearsal():
    from types import SimpleNamespace
    from lem.body_function import CONTRACT,FUNCTION_FIELDS
    from lem.model.self_rehearsal import SelfRehearsal
    from mimo_bridge.self_state_integration import observe
    torch.set_num_threads(1)
    m=SimpleNamespace(device=torch.device('cpu'),active=['sense'],self_state=SelfState(8,1,9),
                      self_history=SelfHistory(),self_rehearsal=SelfRehearsal(8))
    opt=torch.optim.Adam(m.self_state.parameters(),lr=.001)
    tokens={'sense':torch.zeros(1,1,8)}
    for i in range(101):
        m.step_count=i;t=i*.05
        record=dict(contract=CONTRACT,mode='observe',fields=FUNCTION_FIELDS,time=t,
                    values=[[1.,0.,0.]],site_ids=['one'])
        f=SimpleNamespace(meta={'body_function':record},touch_pos=np.zeros((1,3)),sim_time=t,
                          tokenized=SimpleNamespace(touch_impulses=torch.zeros(1,8)))
        loss,_=observe(m,tokens,[f],None if i==0 else torch.zeros(1,1,9))
        opt.zero_grad();loss.backward();opt.step()
    assert 'consequence_error_100' in m.last_self_state
    windows=m.self_rehearsal.common+[v for _,v in m.self_rehearsal.changed]
    long=[w for w in windows if len(w['dt'])==100]
    assert long and abs(float(torch.stack(long[0]['dt']).sum())-5.)<1e-6
    m.self_history.reset()
    assert not m.self_history.events and m.self_rehearsal.seen>0
