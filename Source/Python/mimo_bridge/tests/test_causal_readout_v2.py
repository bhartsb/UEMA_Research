# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lem.model.causal_readout_v2 import (LocalCausalReadout,layout_for,rotation_columns,
    split_senses,join_senses,group_scales,grouped_objective)
from mimo_bridge.causal_packets import tensor_digest,frame_from_payload,fixed_targets
from mimo_bridge.tests.test_causal_packets import payload,packet
from mimo_bridge.tools.causal_readout_v2 import sensed_achieved,encode_window,run_factorial,batch


def inputs(batch_size=8,handles=2,sites=3):
    torch.set_num_threads(1)
    width=9*handles+10*sites
    achieved=torch.zeros(batch_size,6,handles,9);achieved[...,3]=1;achieved[...,7]=1
    action=torch.zeros(batch_size,5,handles,9);action[...,6]=1
    return dict(encoded=torch.zeros(batch_size,6,4),regions=torch.zeros(batch_size,6,4,3),
        region_valid=torch.ones(batch_size,6,4,dtype=torch.bool),sensed=torch.zeros(batch_size,6,width),
        achieved=achieved,site_position=torch.zeros(batch_size,6,sites,3),
        history_dt=torch.tensor([0.]+[.05]*5).expand(batch_size,-1),
        actions=action,durations=torch.full((batch_size,5),.05))


def randomize_outputs(model):
    for layer in (model.body_output,model.skin_output):
        nn.init.normal_(layer.weight,std=.2);nn.init.normal_(layer.bias,std=.1)


def test_decoded_motor_world_position_and_rotation_not_private_geometry():
    frame=frame_from_payload(payload());motor=frame.tokenized.motor_handles
    motor[:,:3]=torch.tensor([.2,-.3,.1]);motor[:,3:7]=torch.tensor([0.,0.,2**-.5,2**-.5])
    before={k:v.tobytes() for k,v in fixed_targets(frame).items()}
    frame.bones[:]=99999 # the auxiliary must not read this simulator-side convenience array
    result=sensed_achieved(frame)
    torch.testing.assert_close(result[0,:3]*200,torch.tensor([40.,-60.,20.]))
    torch.testing.assert_close(result[0,3:],torch.tensor([0.,1.,0.,-1.,0.,0.]),atol=2e-7,rtol=0)
    motor[:,3:7]*=-1
    assert torch.equal(result,sensed_achieved(frame))
    assert before=={k:v.tobytes() for k,v in fixed_targets(frame).items()}


def test_actual_packet_has_achieved_input_missing_from_v1():
    from pathlib import Path
    import json
    from mimo_bridge.causal_packets import load_packet
    source=Path('Saved/mimo_bridge/causal_readout_study_20260913b')
    if not source.exists():pytest.skip('retained native data unavailable in this checkout')
    receipt=json.loads((source/'packet_index.json').read_text())[0]
    frame=frame_from_payload(load_packet(source/'packets'/receipt['path'])['context'][-1])
    actual=sensed_achieved(frame)
    np.testing.assert_allclose(actual[:,:3].numpy()*200,frame.bones[:,4:7],atol=1e-5,rtol=0)
    assert actual.shape==(26,9)


def test_target_layout_and_shared_site_permutation():
    torch.manual_seed(9);model=LocalCausalReadout(4,3,2,3,hidden=12);randomize_outputs(model)
    x=inputs();x['sensed'].normal_();x['site_position'].normal_();x['regions'].normal_()
    y=model(**x);reordered=deepcopy(x);perm=torch.tensor([2,0,1])
    body,skin=split_senses(x['sensed'],2,3)
    reordered['sensed']=join_senses(body,skin[:,:,perm]);reordered['site_position']=x['site_position'][:,:,perm]
    actual=model(**reordered);b,s=split_senses(y,2,3)
    torch.testing.assert_close(actual,join_senses(b,s[:,:,perm]),atol=2e-7,rtol=1e-6)
    # Width is independent of site count: no trained/private site output rows.
    larger=LocalCausalReadout(4,3,2,7,hidden=12)
    assert {k:v.shape for k,v in larger.state_dict().items()}=={k:v.shape for k,v in model.state_dict().items()}


def test_blind_and_pose_ablation_do_not_leak_commands_or_achieved():
    torch.manual_seed(4);x=inputs();model=LocalCausalReadout(4,3,2,3,hidden=12,achieved_input=False)
    randomize_outputs(model);a=model(**x,blind=True)
    x['actions'][...,:3].normal_(0,30);x['achieved'].normal_()
    assert torch.equal(a,model(**x,blind=True))
    on=LocalCausalReadout(4,3,2,3,hidden=12,achieved_input=True);on.load_state_dict(model.state_dict())
    assert not torch.equal(on(**inputs()),on(**x))


def test_group_scales_include_zero_change_sites_and_transfer_units():
    layout=layout_for(2,3);width=48;y=torch.zeros(4,5,width)
    lo,_=layout['touch_kpa'];y[:,:,lo]=torch.arange(4.)[:,None]
    scale,contrast,evidence=group_scales([dict(target=y,start=torch.zeros(width))],layout)
    assert torch.equal(scale[lo:lo+3],scale[lo].expand(3))
    assert torch.equal(contrast[lo:lo+3],contrast[lo].expand(3))
    assert evidence['touch_kpa']['contrast_informative']
    quiet=group_scales([dict(target=torch.zeros_like(y),start=torch.zeros(width))],layout)
    assert not quiet[1].any()


def test_contrast_objective_cannot_be_bought_by_common_motion():
    layout=layout_for(2,3);y=torch.arange(4.).reshape(1,4,1,1).expand(2,4,5,48).clone()
    constant=torch.full_like(y,1.5,requires_grad=True)
    _,parts=grouped_objective(constant,y,layout,torch.ones(48),True)
    _,translated=grouped_objective(constant+100,y+100,layout,torch.ones(48),True)
    assert parts['contrast']==translated['contrast'] and parts['contrast']>0
    _,perfect=grouped_objective(y,y,layout,torch.ones(48),True)
    assert perfect['contrast']==0
    with pytest.raises(ValueError):grouped_objective(y[:,:3],y[:,:3],layout,torch.ones(48),True)


def test_own_encoder_changes_features_never_fixed_targets_or_achieved(monkeypatch):
    from mimo_bridge.causal_packets import MODALITIES
    class Registry(nn.Module):
        def __init__(self,value):super().__init__();self.value=nn.Parameter(torch.tensor(value))
    class Model:
        def __init__(self,value):self.encoder_registry=Registry(value)
        def _batch_frames(self,frames):
            return {k:None for k in MODALITIES},{k:'real' for k in MODALITIES},True,{}
    def spatial(registry,raw,mask):
        return ({k:torch.ones(6,1,3)*registry.value for k in MODALITIES},
                torch.ones(6,2,3)*registry.value,torch.ones(6,2,dtype=torch.bool))
    monkeypatch.setattr('lem.model.spatial.encode_spatial',spatial)
    a,b=Model(1.),Model(9.)
    first=encode_window(a,packet()['context']);second=encode_window(b,packet()['context'])
    assert not torch.equal(first['encoded'],second['encoded'])
    assert not torch.equal(first['regions'],second['regions'])
    for key in ('sensed','achieved','site_position'):
        assert torch.equal(first[key],second[key])
    assert a.encoder_registry.training and b.encoder_registry.training
    assert a.encoder_registry.value.grad is None


def test_planted_command_contrast_learns_body_and_shared_contact_effect():
    torch.manual_seed(44);x=inputs(24,2,2)
    command=torch.tensor([-30.,-10.,10.,30.]).repeat(6)
    x['actions'][:,:,1,0]=command[:,None]
    b,s=split_senses(x['sensed'],2,2);s[:,:,0,0]=.25;s[:,:,1,0]=.75
    x['sensed']=join_senses(b,s)
    target=torch.zeros(24,5,38);time=torch.arange(1,6)*.1
    target[:,:,3]=command[:,None]/200*time
    target[:,:,18]=command[:,None]/200*time*.25
    target[:,:,19]=command[:,None]/200*time*.75
    layout=layout_for(2,2)
    rows=[dict(target=target[i:i+4],start=torch.zeros(38)) for i in range(0,24,4)]
    scale,contrast,_=group_scales(rows,layout)
    model=LocalCausalReadout(4,3,2,2,hidden=16)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(240):
        optimizer.zero_grad()
        loss,_=grouped_objective(model(**x).reshape(6,4,5,38),
            (target/scale).reshape(6,4,5,38),layout,contrast,True)
        loss.backward();optimizer.step()
    check=deepcopy(x);check['actions'][:,:,1,0]*=.7;truth=target*.7
    with torch.no_grad():
        prediction=model(**check)*scale
        error=(prediction[:,:,[3,18,19]]-truth[:,:,[3,18,19]]).square().mean()
        wrong=(prediction.roll(1,0)[:,:,[3,18,19]]-truth[:,:,[3,18,19]]).square().mean()
        effect=truth[:,:,[3,18,19]].square().mean()
    assert error < .25*effect and error < .25*wrong


def fit_head(model,x,target,updates=160):
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(updates):
        optimizer.zero_grad();loss=(model(**x)-target).square().mean();loss.backward();optimizer.step()
    return optimizer


def test_planted_world_coordinate_ambiguity_requires_achieved_input():
    # Identical relative body history/commands at two world translations; one
    # anchored reference and one moving handle have different future outcomes.
    # Only the already-sensed world pose distinguishes them at decision time.
    torch.manual_seed(31);x=inputs(20,2,1);world=torch.linspace(-.3,.3,20)
    x['achieved'][...,0]=world[:,None,None]
    x['sensed'][:,:,3]=.1
    target=torch.zeros(20,5,28);target[:,:,3]=-world[:,None]*torch.arange(1,6)[None]*.1
    on=LocalCausalReadout(4,3,2,1,hidden=16,achieved_input=True)
    off=LocalCausalReadout(4,3,2,1,hidden=16,achieved_input=False);off.load_state_dict(on.state_dict())
    fit_head(on,x,target,260);fit_head(off,x,target,260)
    check=inputs(8,2,1);held=torch.linspace(-.27,.27,8)
    check['achieved'][...,0]=held[:,None,None];check['sensed'][:,:,3]=.1
    truth=-held[:,None]*torch.arange(1,6)[None]*.1
    with torch.no_grad():
        error=(on(**check)[:,:,3]-truth).square().mean()
        blind_error=(off(**check)[:,:,3]-truth).square().mean()
    assert error < truth.square().mean()*.12 and error < blind_error*.12


def test_planted_contact_law_transfers_to_previously_quiet_site_and_preserves_old_adam():
    torch.manual_seed(40);x=inputs(20,2,3)
    body,skin=split_senses(x['sensed'],2,3)
    pressure=torch.linspace(.1,1,20);skin[:,:,0,0]=pressure[:,None]
    x['sensed']=join_senses(body,skin)
    target=torch.zeros(20,5,48);lo=layout_for(2,3)['touch_kpa'][0]
    target[:,:,lo]=pressure[:,None]*torch.arange(1,6)[None]*.1
    old=nn.Linear(2,2);oldopt=torch.optim.AdamW(old.parameters())
    old(torch.ones(1,2)).sum().backward();oldopt.step()
    before=tensor_digest(dict(model=old.state_dict(),adam=oldopt.state_dict()))
    model=LocalCausalReadout(4,3,2,3,hidden=16);fit_head(model,x,target,220)
    check=deepcopy(x);b,s=split_senses(x['sensed'],2,3);s=s.clone();s[:,:,2]=s[:,:,0];s[:,:,0]=0
    check['sensed']=join_senses(b,s)
    with torch.no_grad():
        learnt=model(**x)[:,:,lo];transferred=model(**check)[:,:,lo+2]
        assert torch.allclose(learnt,transferred,atol=1e-6,rtol=1e-5)
        assert (transferred-target[:,:,lo]).square().mean()<target[:,:,lo].square().mean()*.12
        assert model(**check)[:,:,lo].square().mean()<target[:,:,lo].square().mean()*.12
    assert before==tensor_digest(dict(model=old.state_dict(),adam=oldopt.state_dict()))


def synthetic_rows():
    rows=[];layout=layout_for(2,2)
    for world in (1,2,3,4):
        x=inputs(1,2,2);x['actions']=torch.zeros(4,5,2,9);x['actions'][...,6]=1
        x['actions'][...,0]=torch.arange(4.)[:,None,None]
        row={k:v[0].clone() for k,v in x.items() if k not in ('actions','durations')}
        row.update(actions=x['actions'],duration=torch.full((4,5),.05),
            start=torch.zeros(38),target=torch.arange(4.).reshape(4,1,1).expand(4,5,38)*.01,
            world=world,event=0,phase=0,layout=layout,packet_sha256=str(world))
        rows.append(row)
    return rows


def test_complete_factorial_saves_every_model_adam_and_cannot_claim_confirmation(tmp_path):
    rows=synthetic_rows();args=SimpleNamespace(out=tmp_path,fit_worlds=[1,2],evaluation_worlds=[3,4],
        states=1,hidden=8,seed=4,budgets=[1,2],batch_decisions=1)
    result=run_factorial(rows,args,source={'constructed':True})
    assert result['verdict']=='development_only' and not result['production_qualification']
    assert len(result['conditions'])==4 and len(result['artifacts'])==32
    assert all(len(c['development_resolution']['comparisons'])==10 for c in result['conditions'])
    state=torch.load(next(tmp_path.glob('*.pt')),weights_only=False)
    assert state['optimizer']['state'] and state['stage'] in (1,2)
    with pytest.raises(FileExistsError):run_factorial(rows,args,source={'constructed':True})


@pytest.mark.parametrize('mutate',[lambda r:r.pop(),lambda r:r.append(deepcopy(r[0]))])
def test_partial_or_duplicate_development_matrix_refused(tmp_path,mutate):
    rows=synthetic_rows();mutate(rows)
    args=SimpleNamespace(out=tmp_path,fit_worlds=[1,2],evaluation_worlds=[3,4],states=1)
    with pytest.raises(ValueError):run_factorial(rows,args,{})
