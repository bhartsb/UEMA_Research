# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Contact area, taxel sampling and local functioning have separate contracts."""
from copy import deepcopy
from types import SimpleNamespace
import math
import numpy as np
import pytest

from mimo_bridge.contact_footprint import Geometry,VirtualSkinProfile,footprint,aggregate_contact_footprints
from mimo_bridge.virtual_skin import VirtualTaxelSkin
from mimo_bridge.touch_adapter import PatchSpec,ContactInfo


def geom(kind,size,pos=(0,0,0)):
    return Geometry(kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),np.eye(3))


def test_hertz_force_area_and_modulus_bounds_have_correct_si_units():
    sphere=geom(2,[.02,0,0]);plane=geom(0,[1,1,1])
    record=footprint([sphere,plane],np.array([0,0,-.02]),np.array([0,0,1]),10.)
    radius=(3*10*.02/(4e6))**(1/3)
    assert record['area_m2']==pytest.approx(math.pi*radius**2)
    assert record['mean_pressure_pa']*record['area_m2']==pytest.approx(10.)
    assert record['pressure_bounds_pa'][0]<record['mean_pressure_pa']<record['pressure_bounds_pa'][1]
    assert record['method']=='hertz_circular'
    larger=footprint([geom(2,[.04,0,0]),plane],np.array([0,0,-.04]),np.array([0,0,1]),10.)
    assert larger['mean_pressure_pa']<record['mean_pressure_pa']


def test_capsule_side_uses_line_and_end_uses_circular_contact():
    capsule=geom(3,[.02,.05,0]);plane=geom(0,[1,1,1])
    side=footprint([capsule,plane],np.array([.02,0,0]),np.array([1.,0,0]),10.)
    end=footprint([capsule,plane],np.array([0,0,.07]),np.array([0,0,1.]),10.)
    assert side['method']=='finite_hertz_line_ignores_end_effects'
    assert end['method']=='hertz_circular'
    assert side['mean_pressure_pa']<end['mean_pressure_pa']


def test_flat_face_overlap_does_not_depend_on_solver_point_number():
    box=geom(6,[.02,.03,.01]);plane=geom(0,[1,1,1])
    record=footprint([box,plane],np.array([0,0,-.01]),np.array([0,0,1.]),10.)
    assert record['area_m2']==pytest.approx(.04*.06)
    shifted=geom(6,[.02,.03,.01],pos=(.02,0,-.02))
    overlap=footprint([box,shifted],np.array([.01,0,-.01]),np.array([0,0,1.]),10.)
    assert overlap['area_m2']==pytest.approx(.02*.06)


def make_skin(spacing=.005,duplicate=False):
    axis=np.arange(-.04,.040001,spacing);x,y=np.meshgrid(axis,axis)
    points=np.stack([x.ravel(),y.ravel(),np.zeros(x.size)],axis=1)
    if duplicate:points=np.repeat(points,2,axis=0)
    return VirtualTaxelSkin([PatchSpec(1,'hand',len(points),spacing)],{1:points},np.array([[1.,1.]]))


def record(force=30.):
    result=footprint([geom(2,[.005,0,0]),geom(0,[1,1,1])],np.zeros(3),np.array([0.,0,1.]),force)
    result.update(body_id=1,position_body=[0,0,0],tangent_basis_body=result['tangent_basis'],friction_power_w=0.,slip_mm_s=0.)
    return result


def test_physical_force_and_area_are_invariant_to_taxel_resampling():
    r=record();results=[]
    for spacing in (.005,.0025,.00125):
        skin=make_skin(spacing)
        result=skin.advance([r],0.,.01)
        assert skin.normal_force.sum()==pytest.approx(30.)
        assert skin.loaded_area.sum()==pytest.approx(r['area_m2'])
        assert np.sum(skin.physical[:,0]*skin.loaded_area)==pytest.approx(30.)
        results.append(result['physical'][0,0])
    np.testing.assert_allclose(results,results[0],rtol=1e-6)


def test_duplicate_taxels_do_not_change_pressure_or_impair_more_surface():
    ordinary=make_skin();duplicate=make_skin(duplicate=True);r=record()
    for tick in range(20):
        first=ordinary.advance([r],tick*.01,.01);second=duplicate.advance([r],tick*.01,.01)
    np.testing.assert_allclose(first['physical'],second['physical'])
    np.testing.assert_allclose(first['experienced'],second['experienced'])
    np.testing.assert_allclose(first['values'],second['values'])
    assert first['affected_sensor_area_fraction']==pytest.approx(second['affected_sensor_area_fraction'])


def test_local_pressure_modulates_only_loaded_taxels_and_recovers_after_release():
    skin=make_skin();r=record()
    for tick in range(30):result=skin.advance([r],tick*.01,.01)
    loaded=skin.loaded_area>0;quality=skin.function.quality.copy()
    assert (quality[loaded]<1).any() and (quality[~loaded]==1).all()
    assert np.any(skin.experienced[loaded,0]!=skin.physical[loaded,0])
    assert 0<result['affected_sensor_area_fraction'][0]<.1
    for tick in range(30,80):skin.advance([],tick*.01,.01)
    assert (skin.function.quality[loaded]>quality[loaded]).all()
    assert (skin.function.quality[~loaded]==1).all()


def test_unresolved_geometry_cannot_silently_remove_a_force():
    r=footprint([geom(7,[1,1,1]),geom(0,[1,1,1])],np.zeros(3),np.array([0.,0,1.]),1.)
    assert not r['supported']
    with pytest.raises(ValueError,match='cannot disappear'):make_skin().advance([r],0.,.01)


def test_sharp_edge_and_invalid_geometry_are_not_claimed_as_calibrated_contact():
    box=geom(6,[.01,.02,.03]);plane=geom(0,[1,1,1])
    edge=footprint([box,plane],np.array([.01,.02,0]),np.array([1.,1.,0]),1.)
    assert not edge['supported'] and edge['reason']=='sharp_edge_unresolved'
    with pytest.raises(ValueError,match='positive physical dimensions'):geom(2,[0,0,0])
    with pytest.raises(ValueError,match='proper rotation'):
        Geometry(2,np.ones(3),np.zeros(3),np.zeros((3,3)))


def test_long_contact_strip_does_not_load_the_back_of_the_body():
    points=np.array([[0.,0.,0.],[0.,0.,-.02]])
    skin=VirtualTaxelSkin([PatchSpec(1,'hand',2,.002)],{1:points},np.array([[1.,1.]]))
    r=record(10.)
    r.update(method='finite_hertz_line_ignores_end_effects',tangent_axes_m=[.05,.003])
    skin.advance([r],0.,.01)
    assert skin.normal_force[0]==pytest.approx(10.) and skin.normal_force[1]==0.


def test_vectorized_precision_matches_the_declared_body_operator_exactly():
    skin=make_skin();rng=np.random.default_rng(711)
    skin.function.quality=rng.uniform(0,1,len(skin.sample_area));skin.function.quality[0]=1
    skin.advance([record(500.)],0.,.01)
    np.testing.assert_array_equal(skin.experienced,skin.function.observe(skin.physical))


def test_solver_contact_splitting_preserves_one_physical_patch():
    model=SimpleNamespace(geom_type=np.array([2,0]),geom_size=np.array([[.02,0,0],[1,1,1]]))
    data=SimpleNamespace(geom_xpos=np.zeros((2,3)),geom_xmat=np.tile(np.eye(3).ravel(),(2,1)),
                         xmat=np.tile(np.eye(3).ravel(),(2,1)))
    def contact(force):
        return ContactInfo(1,(0,1),force,np.zeros(3),np.zeros(3),np.zeros(3),np.array([0,0,1.]),0.,0.)
    single=aggregate_contact_footprints(model,data,[contact(10.)])[0]
    split=aggregate_contact_footprints(model,data,[contact(2.5)]*4)[0]
    assert single['normal_force_n']==split['normal_force_n']
    assert single['area_m2']==split['area_m2'] and single['mean_pressure_pa']==split['mean_pressure_pa']


@pytest.mark.parametrize('contact_profile',['mechanical_v3','mechanical_v2_local'])
def test_native_virtual_skin_saves_exact_physical_history_and_feeds_native_self(tmp_path,contact_profile):
    import json
    import torch
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite
    torch.set_num_threads(1)
    source=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path/'scene'),age=24,
        caregiver_seconds=0,body_function='modulate',contact_profile=contact_profile)
    try:
        frames=source.reset();frame=frames[0]
        command=np.zeros((1,len(frame.bones),9),np.float32)
        command[0,:,:3]=frame.bones[:,4:7];command[0,:,3:7]=frame.bones[:,:4]
        for _ in range(3):frames=source.step(command)
        record=frames[0].meta['mechanical_contact']
        if contact_profile=='mechanical_v3':assert record['contract']=='virtual_elastic_skin_v1_2026_09_13'
        else:assert 'unchanged v2' in record['area_source']
        json.dumps(record,allow_nan=False)
        assert frames[0].meta['skin_function_granularity']['taxels']>len(frames[0].patches)
        snapshot=source.save_full_state();expected=source.step(command)[0]
        source.restore_full_state(snapshot);actual=source.step(command)[0]
        np.testing.assert_array_equal(actual.patches,expected.patches)
        np.testing.assert_array_equal(actual.bones,expected.bones)
        np.testing.assert_array_equal(actual.meta['body_function']['values'],expected.meta['body_function']['values'])
        model=LemLite(device='cpu',imagination=False,no_vision=True,contact_profile=contact_profile,
            self_state='learn',self_prediction='grounded_residual',checkpoint_dir=str(tmp_path/'checkpoints'))
        step=model.step([actual],model.dev_limits(0),train=True,step_index=0,executed_action=command)
        assert np.isfinite(step.loss) and model.last_self_state['observed_grounding_error']==0
        assert model.last_self_state['function_extent_observed']
        assert model.last_self_state['surface_moments_observed']
        # Native brain, optimizer, local skin history and causal learner memory
        # are published together, then restored before the same next action.
        from mimo_bridge.continuation_state import save,load,restore_learner
        def companion(staging,final):
            save(staging,source,model,dict(step=1),dict(contact_profile=contact_profile),published_path=final)
        checkpoint=model.save_checkpoint(1,companion_writer=companion)
        continuation=load(checkpoint)
        model.load_checkpoint(checkpoint)
        restore_learner(model,continuation['learner'])
        expected=source.step(command)[0]
        source.restore_full_state(continuation['worlds'])
        actual=source.step(command)[0]
        np.testing.assert_array_equal(actual.patches,expected.patches)
        np.testing.assert_array_equal(actual.bones,expected.bones)
        np.testing.assert_array_equal(actual.meta['body_function']['values'],expected.meta['body_function']['values'])
        assert model.self_history.inputs['function_extent'] is not None
    finally:source.close()


def test_explicit_virtual_transition_preserves_history_body_and_old_observation(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.virtual_skin_transition import activate_virtual_skin
    source=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path/'scene'),age=24,
        caregiver_seconds=0,body_function='modulate',contact_profile='mechanical_v2')
    restored=None
    try:
        frames=source.reset();frame=frames[0]
        command=np.zeros((1,len(frame.bones),9),np.float32)
        command[0,:,:3]=frame.bones[:,4:7];command[0,:,3:7]=frame.bones[:,:4]
        for _ in range(3):frames=source.step(command)
        original=dict(config=dict(contact_profile='mechanical_v2',body_function='modulate'),
            worlds=source.save_full_state(),trainer=dict(step=3,frame_meta=[frames[0].meta]),learner=dict(memory=np.arange(4)))
        upgraded=activate_virtual_skin(original)
        np.testing.assert_array_equal(upgraded['learner']['memory'],original['learner']['memory'])
        np.testing.assert_array_equal(upgraded['worlds'][0]['payload']['patches'],original['worlds'][0]['payload']['patches'])
        assert original['worlds'][0]['payload']['info'].get('contact_profile_transition') is None
        restored=SingleEnvSource(**upgraded['worlds'][0]['env_kwargs'])
        before=restored.restore_full_state(upgraded['worlds'])[0]
        np.testing.assert_array_equal(before.patches,frames[0].patches)
        np.testing.assert_array_equal(before.bones,frames[0].bones)
        expected=source.step(command)[0];actual=restored.step(command)[0]
        np.testing.assert_array_equal(actual.bones,expected.bones)
        assert actual.sim_time==expected.sim_time
        assert actual.meta['mechanical_contact']['contract']=='virtual_elastic_skin_v1_2026_09_13'
        assert not actual.meta.get('contact_profile_transition')
        assert actual.meta['skin_function_granularity']['taxels']>len(actual.patches)
        with pytest.raises(ValueError,match='mechanical_v2'):activate_virtual_skin(upgraded)
    finally:
        source.close()
        if restored is not None:restored.close()


def test_a_live_v2_local_run_can_be_carried_onto_v4(tmp_path):
    """The switch the operator actually has to make, end to end.

    Every crib run launches --contact-profile mechanical_v2_local, not
    mechanical_v2, and the transition helper used to accept only mechanical_v2 as
    a source - so the profile the runs are ON could not be carried anywhere. This
    exercises v2_local to mechanical_v4: the saved observation is untouched, the
    boundary frame is annotated, and the restored world steps on the new
    transducer's contract.
    """
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.virtual_skin_transition import activate_graded_impact_skin
    source = SingleEnvSource(scene='crib', vision=False, run_dir=str(tmp_path / 'scene'),
                             age=24, caregiver_seconds=0, body_function='modulate',
                             contact_profile='mechanical_v2_local')
    restored = None
    try:
        frames = source.reset(); frame = frames[0]
        command = np.zeros((1, len(frame.bones), 9), np.float32)
        command[0, :, :3] = frame.bones[:, 4:7]; command[0, :, 3:7] = frame.bones[:, :4]
        for _ in range(3):
            frames = source.step(command)
        original = dict(config=dict(contact_profile='mechanical_v2_local', body_function='modulate'),
                        worlds=source.save_full_state(),
                        trainer=dict(step=3, frame_meta=[frames[0].meta]),
                        learner=dict(memory=np.arange(4)))
        upgraded = activate_graded_impact_skin(original)

        assert upgraded['config']['contact_profile'] == 'mechanical_v4'
        marker = upgraded['worlds'][0]['payload']['info']['contact_profile_transition']
        assert marker['source'] == 'mechanical_v2_local'
        assert marker['target'] == 'mechanical_v4'
        # nothing the run already saw may be rewritten by the switch
        np.testing.assert_array_equal(upgraded['learner']['memory'], original['learner']['memory'])
        np.testing.assert_array_equal(upgraded['worlds'][0]['payload']['patches'],
                                      original['worlds'][0]['payload']['patches'])

        restored = SingleEnvSource(**upgraded['worlds'][0]['env_kwargs'])
        before = restored.restore_full_state(upgraded['worlds'])[0]
        np.testing.assert_array_equal(before.patches, frames[0].patches)
        actual = restored.step(command)[0]
        assert actual.sim_time == source.step(command)[0].sim_time
        assert actual.meta['mechanical_contact']['contract'] == 'virtual_elastic_skin_v1_2026_09_13'
        assert 'virtual elastic geometry' in actual.meta['mechanical_contact']['area_source']
        # carrying it a second time is refused - v4 is not a source anyone can
        # carry FROM, because it already writes the new history contract
        with pytest.raises(ValueError, match="requires an episode on"):
            activate_graded_impact_skin(upgraded)
    finally:
        source.close()
        if restored is not None:
            restored.close()


def test_a_v3_or_v4_episode_is_not_a_transferable_source():
    """v3 and v4 share a history contract, so neither needs - or gets - a boundary."""
    from mimo_bridge.virtual_skin_transition import activate_local_skin
    for profile in ('mechanical_v3', 'mechanical_v4'):
        episode = dict(config=dict(contact_profile=profile, body_function='modulate'),
                       worlds=[], trainer={}, learner={})
        with pytest.raises(ValueError, match='requires an episode on'):
            activate_local_skin(episode, target='mechanical_v2_local')
