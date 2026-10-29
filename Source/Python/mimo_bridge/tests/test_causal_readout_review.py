# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Independent instrument regressions; none certify behavior or touch live code."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from mimo_bridge import causal_packets as cp
from mimo_bridge.tests.test_causal_packets import payload,packet,gate_rows
from mimo_bridge.tools.causal_readout_study import admission


def full_gate_rows():
    rows=[]
    for world in (1,2):
        for group in ('position_cm','rotation_6d','touch_kpa','touch_first_kpa_cm','touch_second_kpa_cm2'):
            rows.extend(dict(row,world=world,group=group) for row in gate_rows() if row['group']=='position_cm')
    return rows


def test_missing_entire_world_group_cannot_pass_by_other_world_coverage():
    rows=full_gate_rows()
    assert admission(rows)['verdict']=='passed'
    incomplete=[r for r in rows if not (r['world']==2 and r['group']=='rotation_6d')]
    with pytest.raises(ValueError):admission(incomplete)


def test_omitted_declared_world_cannot_disappear_from_admission():
    rows=[r for r in full_gate_rows() if r['world']==1]
    with pytest.raises(ValueError):admission(rows,expected_worlds=[1,2])


def test_gaze_handle_identity_cannot_be_relabelled_to_hide_future_eye_commands():
    row=packet();row['gaze_handle']=0
    row['commands'][1,1,23,0]=.3
    with pytest.raises(ValueError):cp.validate(row)


def test_wire_timestamp_cannot_be_hidden_by_relabelling_transport_time():
    row=packet()
    row['context'][0]['frame_bytes']=payload(1.5)['frame_bytes']
    with pytest.raises(ValueError):cp.validate(row)


def test_target_layout_cannot_relabel_pressure_as_body_position():
    row=packet()
    layout=cp.target_layout(cp.fixed_targets(cp.frame_from_payload(row['context'][-1])))
    layout['position_cm'],layout['touch_kpa']=layout['touch_kpa'],layout['position_cm']
    row['target_layout']=layout
    with pytest.raises(ValueError):cp.validate(row)


def test_world_rotation_columns_and_spatial_moment_units_have_analytic_values():
    raw=payload()
    moments=raw['info']['skin_function_granularity']['surface_moments_root']
    moments[0,:3]=[.01,.02,.03]
    moments[0,3:9]=[.0001,.00002,.00003,.0004,.00005,.0009]
    frame=cp.frame_from_payload(cp.sensory_payload(raw))
    frame.tokenized.proprio[0,:4]=__import__('torch').tensor([0.,0.,2**-.5,2**-.5])
    values=cp.fixed_targets(frame)
    np.testing.assert_allclose(values['rotation_6d'][:6],[0,1,0,-1,0,0],atol=1e-6)
    np.testing.assert_allclose(values['touch_first_kpa_cm'][:3],[1,2,3],rtol=1e-6)
    np.testing.assert_allclose(values['touch_second_kpa_cm2'][:6],[2,2.2,3.3,8,6.5,18],rtol=1e-6)


def test_extended_trunk_positions_follow_native_relative_handle_layout():
    raw=payload()
    bones=np.zeros((26,11),np.float32);bones[:,3]=1
    bones[:,4:7]=[50,-20,35]
    bones[:,4]+=np.arange(26)
    raw['bones']=bones
    handles=np.zeros((26,10),np.float32);handles[:,6]=1
    raw['extended_handles']=handles
    values=cp.fixed_targets(cp.frame_from_payload(cp.sensory_payload(raw)))
    positions=values['position_cm'].reshape(26,3)
    np.testing.assert_allclose(positions[:,0],np.arange(26),atol=1e-5)
    np.testing.assert_array_equal(positions[:,1:],0)


def test_precommitted_schedule_applies_smoother_once_and_finishes_old_order():
    from mimo_bridge.tools.causal_readout_study import commands_for
    from mimo_bridge.tools.self_recovery_experiment import hold
    from mimo_bridge.lem_model import LemLite
    frame=cp.frame_from_payload(cp.sensory_payload(payload()))
    previous=LemLite.gaze_to_handle(hold(frame,.1),np.zeros((1,2)))
    limits={'rom_limit':1.,'strength_limit':.8,'handle_mask':[True]*24}
    new=commands_for(frame,previous,limits,.35,0,151)
    delayed=commands_for(frame,previous,limits,.35,2,151)
    np.testing.assert_array_equal(delayed[:,:3],np.broadcast_to(previous,(4,3,24,9)))
    np.testing.assert_array_equal(delayed[:,3:],new[:,:2])
    np.testing.assert_array_equal(new,np.broadcast_to(new[:,:1],new.shape))
    displacement=new[2,0,:23,:3]-frame.bones[:23,4:7]
    np.testing.assert_allclose(np.linalg.norm(displacement,axis=-1),8*.35,atol=1e-5)
    np.testing.assert_allclose(new[2,0,:23,8],.3*.35+.1*.65,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(new[:,:, :23,3:7],axis=-1),1,atol=1e-6)
    np.testing.assert_array_equal(new[:,:,23],np.broadcast_to(previous[0,23],new[:,:,23].shape))


def indexed_packets(tmp_path):
    args=SimpleNamespace(out=tmp_path,train_worlds=[10,11],test_worlds=[20,21],states=2)
    rows=[]
    for world in args.train_worlds+args.test_worlds:
        for event in range(args.states):
            row=packet();row.update(world_id=world,decision=event,phase=event,
                order_started=row['decision_time']-.05*event,
                split='train' if world in args.train_worlds else 'test')
            saved=cp.write_packet(tmp_path/'packets',row,reserve_bytes=0)
            rows.append(dict(saved,world=world,event=event,phase=event,split=row['split']))
    return args,rows


@pytest.mark.parametrize('mutation',['partial','duplicate','world','event','phase','split','sha','bytes','source'])
def test_complete_packet_index_rejects_substitution_or_selective_samples(tmp_path,mutation):
    from mimo_bridge.tools.causal_readout_study import validate_index
    args,rows=indexed_packets(tmp_path)
    assert validate_index(args,rows,'a'*64)
    bad=deepcopy(rows)
    if mutation=='partial':bad.pop()
    elif mutation=='duplicate':bad.append(deepcopy(bad[0]))
    elif mutation=='world':bad[0]['world']=999
    elif mutation=='event':bad[0]['event']=1
    elif mutation=='phase':bad[0]['phase']=1
    elif mutation=='split':bad[0]['split']='test'
    elif mutation=='sha':bad[0]['sha256']='b'*64
    elif mutation=='bytes':bad[0]['compressed_bytes']+=1
    elif mutation=='source':
        row=cp.load_packet(tmp_path/'packets'/bad[0]['path'])
        row['source_checkpoint_sha256']='b'*64
        bad[0].update(cp.write_packet(tmp_path/'packets',row,reserve_bytes=0))
    with pytest.raises(ValueError):validate_index(args,bad,'a'*64)
