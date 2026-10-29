# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from mimo_bridge import causal_packets as cp
from lem.model.causal_readout import CausalReadout, body_shuffle
from mimo_bridge.tools.causal_readout_study import training_scale, metrics,admission


def payload(t=1., covariance=0.):
    from mimo_bridge.frame_encoder import encode_frame
    bones=np.zeros((24,11),np.float32); bones[:,3]=1
    bones[:,4]=np.arange(24); handles=np.zeros((24,10),np.float32); handles[:,6]=1
    patches=np.zeros((2,10),np.float32); patches[:,0]=[1000,2000]
    wire=encode_frame(frame_number=round(t*100),timestamp=t,dt=.01,wall_clock=0,
        gaze_az_deg=0,gaze_el_deg=0,bones=bones,bone_hashes=None,handles=handles,
        energy=[0,0,0],body_sets=None,patches=patches,collisions=[],
        audio_spectral=np.zeros((256,2),np.float32),vision_valid=True)
    moments=np.zeros((2,20),np.float32); moments[:,18]=1; moments[:,3]=covariance
    return dict(frame_bytes=bytes(wire),extended_handles=None,eye_left=np.zeros((16,16,3),np.uint8),
        eye_right=np.zeros((16,16,3),np.uint8),fovea_left=None,fovea_right=None,
        vision_layout=(('wide',16,60.),),sim_time=t,dt=.01,frame=round(t*100),
        patches=patches,bones=bones,vestibular=np.zeros(6),eye_angles=((0,0),(0,0)),
        gaze_deg=(0,0),touch_pos=np.zeros((2,3)),touch_sensitivity=np.ones((2,2)),
        n_contacts=0,ctrl=np.zeros(1),info=dict(skin_function_granularity=dict(surface_moments_root=moments),
        physical_touch_audit=np.full((2,8),999999),private_object_world_pose=[7,8,9]))


def packet():
    context=[cp.sensory_payload(payload(float(t))) for t in np.linspace(.75,1.,6)]
    groups=cp.fixed_targets(cp.frame_from_payload(context[-1]));start=cp.packed_target(groups)
    commands=np.zeros((4,5,24,9),np.float32); commands[...,6]=1
    target=np.broadcast_to(start,(4,5,len(start))).copy()
    return dict(contract=cp.CONTRACT,world_id=1,source_checkpoint_sha256='a'*64,context=context,
        phase=0,order_started=1.,decision=0,split='train',target_layout=cp.target_layout(groups),
        decision_time=1.,commands=commands,durations=np.full((4,5),.05),
        outcome_times=np.broadcast_to(1+np.arange(1,6)*.05,(4,5)).copy(),targets=target,
        start=start,replay_target=target[0].copy(),gaze_handle=23,reflex_off=True,saccades_off=True,
        input_provenance='predecision_raw_senses_and_precommitted_commands')


def test_fixed_units_and_no_hidden_physical_pressure():
    raw=payload(); p=cp.sensory_payload(raw)
    assert 'physical_touch_audit' not in p['info'] and 'private_object_world_pose' not in p['info']
    target=cp.fixed_targets(cp.frame_from_payload(p))
    np.testing.assert_allclose(target['touch_kpa'],[1,2],atol=1e-6)
    np.testing.assert_allclose(target['position_cm'].reshape(24,3)[:,0],np.arange(24),atol=1e-5)


def test_two_separated_contacts_differ_from_one_at_empty_centroid():
    single=cp.fixed_targets(cp.frame_from_payload(payload(covariance=0)))
    separated=cp.fixed_targets(cp.frame_from_payload(payload(covariance=.01)))
    assert np.array_equal(single['touch_first_kpa_cm'],separated['touch_first_kpa_cm'])
    assert not np.array_equal(single['touch_second_kpa_cm2'],separated['touch_second_kpa_cm2'])


def test_quaternion_sign_has_identical_target():
    frame=cp.frame_from_payload(payload()); before=cp.fixed_targets(frame)
    frame.tokenized.proprio[:,:4]*=-1
    np.testing.assert_array_equal(before['rotation_6d'],cp.fixed_targets(frame)['rotation_6d'])


@pytest.mark.parametrize('mutation',[lambda p:p.update(input_provenance='future_executed_reflex'),
    lambda p:p.update(reflex_off=False), lambda p:p['context'][0].update(sim_time=1.1),
    lambda p:p['outcome_times'].__setitem__((0,4),1.2),
    lambda p:p['commands'].__setitem__((1,1,23,0),3.),
    lambda p:p['replay_target'].__setitem__((0,0),999.)])
def test_causal_and_physical_mismatch_rejected(mutation):
    p=packet();mutation(p)
    with pytest.raises(ValueError):cp.validate(p)


def test_immutable_compressed_roundtrip(tmp_path):
    p=packet(); receipt=cp.write_packet(tmp_path,p,reserve_bytes=0)
    loaded=cp.load_packet(tmp_path/receipt['path'])
    assert cp.tensor_digest(p)==cp.tensor_digest(loaded)
    p['targets'][0,0,0]=123
    assert cp.load_packet(tmp_path/receipt['path'])['targets'][0,0,0]!=123


def test_each_snapshot_uses_own_encoder_targets_byte_identical(monkeypatch):
    class Registry(nn.Module):
        def __init__(self,value):super().__init__();self.value=nn.Parameter(torch.tensor(value))
        def encode(self,raw,mask):return {k:torch.ones(6,1,3)*self.value for k in cp.MODALITIES}
    class Model:
        def __init__(self,value):self.encoder_registry=Registry(value)
        def _batch_frames(self,frames):
            return {k:None for k in cp.MODALITIES},{k:'real' for k in cp.MODALITIES},True,{}
    def spatial(registry,raw,mask):
        return registry.encode(raw,mask),torch.ones(6,1,3)*registry.value,torch.ones(6,1,dtype=torch.bool)
    monkeypatch.setattr('lem.model.spatial.encode_spatial',spatial)
    raw=packet()['context']; a,b=Model(1.),Model(9.)
    ea,ya=cp.encode_own_window(a,raw);eb,yb=cp.encode_own_window(b,raw)
    assert not torch.equal(ea,eb)
    assert ya.tobytes()==yb.tobytes()
    assert a.encoder_registry.training and b.encoder_registry.training
    assert a.encoder_registry.value.grad is None and b.encoder_registry.value.grad is None


def test_body_shuffle_keeps_gaze_and_matches_known_body():
    action=torch.arange(4*5*24*9).reshape(4,5,24,9).float(); perm=torch.tensor([1,2,3,0])
    shuffled=body_shuffle(action,23,perm)
    assert torch.equal(shuffled[:,:,:23],action[perm,:,:23])
    assert torch.equal(shuffled[:,:,23],action[:,:,23])


def test_training_scale_does_not_inspect_unseen_world():
    train=[dict(target=torch.ones(4,5,2),start=torch.zeros(2))]
    scale,mask=training_scale(train)
    assert torch.equal(scale,torch.ones(2)) and mask.all()


def test_auxiliary_can_learn_known_actions_without_changing_old_optimizer():
    torch.manual_seed(2);torch.set_num_threads(1)
    old=nn.Linear(2,2); opt=torch.optim.AdamW(old.parameters())
    old(torch.ones(1,2)).sum().backward();opt.step()
    before=cp.tensor_digest(dict(model=old.state_dict(),optimizer=opt.state_dict()))
    head=CausalReadout(2,2,2,hidden=24);auxopt=torch.optim.AdamW(head.parameters(),lr=.01)
    action=torch.randn(48,5,2);duration=torch.full((48,5),.05)
    inputs=(torch.zeros(48,6,2),torch.zeros(48,6,2),torch.full((48,6),.05),action,duration)
    y=action.cumsum(1)*.05
    for _ in range(160):
        auxopt.zero_grad();loss=(head(*inputs)-y).square().mean();loss.backward();auxopt.step()
    with torch.no_grad():
        correct=(head(*inputs)-y).square().mean()
        wrong=(head(*inputs[:3],action.roll(1,0),duration)-y).square().mean()
    assert correct < y.square().mean()*.12 and correct < wrong*.12
    assert before==cp.tensor_digest(dict(model=old.state_dict(),optimizer=opt.state_dict()))
    assert not ({id(p) for g in opt.param_groups for p in g['params']} &
                {id(p) for g in auxopt.param_groups for p in g['params']})


def test_constant_forecasts_cannot_pass_action_effect_gate():
    y=torch.arange(4.).reshape(4,1,1).expand(4,5,1)
    row=dict(world=1,target=y,start=torch.zeros(1),layout={'position_cm':[0,1]},
        sensed=torch.zeros(2,1),history_dt=torch.tensor([0.,.05]),duration=torch.full((4,5),.05))
    reports=metrics([row],torch.full_like(y,1.5),'constructed')
    assert all(r['action_error_removed']==0 and not r['effect_resolved'] for r in reports)


def gate_rows(**changes):
    final=dict(world=1,group='position_cm',horizon_seconds=.25,forecast='action_conditioned_stage_2',
        informative=True,action_effect_rms=1.,action_contrast_error_rms=.4,prediction_rmse=.4,
        persistence_rmse=1.,wrong_action_rmse=1.,constant_motion_rmse=1.,numerical_resolution=1e-7)
    final.update(changes)
    before=dict(final,forecast='action_conditioned_stage_1',prediction_rmse=.41)
    blind=dict(final,forecast='action_blind_stage_2',prediction_rmse=.9)
    return [dict(row,group=group) for group in ('position_cm','rotation_6d','touch_kpa',
        'touch_first_kpa_cm','touch_second_kpa_cm2') for row in
        (final,before,blind,dict(blind,forecast='action_blind_stage_1'))]


@pytest.mark.parametrize('changes',[dict(action_contrast_error_rms=.999),dict(persistence_rmse=.39),
    dict(wrong_action_rmse=.39),dict(constant_motion_rmse=.39),dict(prediction_rmse=.99)])
def test_all_declared_controls_bind_admission(changes):
    report=admission(gate_rows(**changes))
    assert report['verdict']=='failed' and report['comparisons'][0]['status']=='failed'


def test_retention_and_trained_blind_cannot_be_ignored():
    for index in (1,2):
        rows=gate_rows();rows[index]['prediction_rmse']=.1
        assert admission(rows)['verdict']=='failed'


def test_quiet_group_is_inconclusive_not_failed_or_passed():
    report=admission(gate_rows(informative=False))
    assert report['verdict']=='inconclusive' and report['comparisons'][0]['status']=='uninformative'


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_invalid_numeric_cannot_enter_admission(bad):
    with pytest.raises(ValueError):admission(gate_rows(prediction_rmse=bad))


def test_missing_or_duplicate_control_fails_closed():
    rows=gate_rows()
    with pytest.raises(ValueError):admission(rows[:-1])
    with pytest.raises(ValueError):admission(rows+[rows[0]])


def test_complete_constructed_fit_controls_and_restored_forecast(tmp_path):
    from mimo_bridge.tools.causal_readout_study import fit
    torch.manual_seed(71); torch.set_num_threads(1)
    names=['position_cm','rotation_6d','touch_kpa','touch_first_kpa_cm','touch_second_kpa_cm2']
    rows=[]
    for world in (1,2,3,4):
        for event in range(2):
            action=torch.randn(4,5,5)*.2
            rows.append(dict(world=world,phase=event,split='train' if world<3 else 'test',
                encoded=torch.ones(6,2),sensed=torch.zeros(6,5),history_dt=torch.tensor([0]+[.05]*5),
                action=action,duration=torch.full((4,5),.05),target=action.cumsum(1)*.05,
                start=torch.zeros(5),layout={k:[i,i+1] for i,k in enumerate(names)},packet_sha256=str(world)+str(event)))
    args=SimpleNamespace(train_worlds=[1,2],test_worlds=[3,4],fit_seed=19,updates=16,out=tmp_path,
                         checkpoint_sha256='constructed',encoder_sha256='constructed')
    report=fit(rows,args)
    assert report['production_qualification'] is False and report['intention_demonstrated'] is False
    assert len(report['admission']['comparisons'])==10 and len(report['phase_results'])==160
    assert (tmp_path/'action_conditioned.pt').is_file() and (tmp_path/'action_blind.pt').is_file()
