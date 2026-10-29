# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import numpy as np
import pytest
import json
from types import SimpleNamespace
from mimo_bridge.tools.self_recovery_experiment import recovery_cost,comparison_result,hold,recovery_opportunity,action_prediction_result,sequences


def test_local_loss_cannot_hide_among_unaffected_sites():
    x=np.ones((4,100,3));valid=np.ones(100,bool)
    assert recovery_cost(x,valid)==0
    x[:,37,0]=0
    assert recovery_cost(x,valid)==1
    valid[37]=False
    assert recovery_cost(x,valid)==0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.001,1.001])
def test_invalid_function_is_not_a_good_choice(bad):
    x=np.ones((4,2,3));x[0,0,0]=bad
    with pytest.raises(ValueError):recovery_cost(x,np.ones(2,bool))


def test_empty_or_misaligned_observations_fail():
    for x,v in [(np.ones((0,2,3)),np.ones(2,bool)),(np.ones((2,2,3)),np.zeros(2,bool)),
                (np.ones((2,2,3)),np.ones(3,bool)),(np.ones((2,2,3)),np.ones(2,int))]:
        with pytest.raises(ValueError):recovery_cost(x,v)


def test_hold_uses_achieved_position_and_quaternion_in_actual_wire_order():
    f=SimpleNamespace(bones=np.array([[.1,.2,.3,.9,12.,24.,36.]],np.float32))
    cmd=hold(f,.3)
    np.testing.assert_array_equal(cmd[0,0,:3],f.bones[0,4:7])
    np.testing.assert_array_equal(cmd[0,0,3:7],f.bones[0,:4])


def test_choice_comparison_detects_planted_gain_and_rejects_no_gain():
    positive=comparison_result([.1]*8);null=comparison_result([0.]*8)
    assert positive['paired_permutation_p']<=.05
    assert positive['lower']>0
    assert null['paired_permutation_p']==1
    assert 'function deficit' in positive['scale']
    assert not comparison_result([])['valid']


def test_automatic_recovery_and_movement_only_fatigue_cannot_admit_training():
    row=dict(source_skin_sensitivity=.1,available_skin_gain_over_hold=.1,available_gain_over_hold=.1)
    assert recovery_opportunity(row)
    assert json.loads(json.dumps(dict(opportunity=recovery_opportunity(row))))['opportunity'] is True
    for field in row:
        assert not recovery_opportunity(dict(row,**{field:0.}))


def test_prediction_requires_action_effects_not_correct_average():
    q=np.ones((4,5,2,3));q[:,...,0]=np.arange(4)[:,None,None]/3
    valid=np.ones(2,bool)
    assert action_prediction_result(q,q,valid)['action_error_removed']==1
    assert action_prediction_result(q,np.broadcast_to(q.mean(0),q.shape),valid)['action_error_removed']==0
    assert action_prediction_result(q,q[::-1],valid)['action_error_removed']<0
    assert not action_prediction_result(np.ones_like(q),q,valid)['valid']


def test_commands_retain_physical_units_hold_cadence_and_actual_eye_order():
    bones=np.zeros((26,7),np.float32);bones[:,3]=1;bones[:,4]=10
    frame=SimpleNamespace(bones=bones);proposal=hold(frame,.3);proposal[0,0,0]+=20
    result=sequences(frame,proposal,{'strength_limit':1.,'rom_limit':1.},
                     SimpleNamespace(motor_hold=5,smooth_actions=.35),3,10,17).numpy()
    np.testing.assert_allclose(result[0,:5,0,0],17,atol=1e-5)
    np.testing.assert_allclose(result[0,5:,0,0],21.55,atol=1e-5)
    np.testing.assert_allclose(result[1,:,0,0],10,atol=1e-5)
    np.testing.assert_array_equal(result[:,:,23,3:7],np.broadcast_to([0,0,0,1],(3,10,4)))
