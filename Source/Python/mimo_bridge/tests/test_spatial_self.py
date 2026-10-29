# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import torch
import pytest
from lem.model.spatial_self import SpatialSelfContext


def test_spatial_context_starts_neutral_then_uses_located_distinctions():
    module=SpatialSelfContext(2)
    local=torch.tensor([[[1.,0.],[-1.,0.]]])
    spatial=torch.tensor([[[2.,1.],[-2.,-1.]]])
    valid=torch.ones(1,2,dtype=torch.bool)
    assert module(local,spatial,valid).count_nonzero()==0
    with torch.no_grad():
        for part in (module.query,module.key,module.value,module.output):part.weight.copy_(torch.eye(2))
    first=module(local,spatial,valid)
    assert first[0,0,1]>0 and first[0,1,1]<0
    # Same global mean, opposite relation between position and sensed feature.
    changed=spatial.clone();changed[:,:,1]*=-1
    assert module(local,changed,valid)[0,0,1]<0
    torch.testing.assert_close(spatial.mean(1),changed.mean(1))


def test_spatial_context_masks_missing_senses_and_never_pools_other_bodies():
    torch.manual_seed(8);module=SpatialSelfContext(4)
    with torch.no_grad():module.output.weight.copy_(torch.eye(4))
    x=torch.randn(2,3,4);sense=torch.randn(2,5,4);mask=torch.ones(2,5,dtype=torch.bool)
    before=module(x,sense,mask);sense[1]*=100
    torch.testing.assert_close(before[0],module(x,sense,mask)[0],rtol=0,atol=0)
    mask[0]=False
    assert module(x,sense,mask)[0].eq(0).all()
    with pytest.raises(ValueError):module(x,sense,mask[:,:2])


def test_spatial_encoder_preserves_existing_summary_exactly():
    from mimo_bridge.lem_model import LemLite, TOUCH_TOKENS, TOUCH_TOKEN_F, TOUCH_VALID_COL
    from lem.model.spatial import encode_spatial
    torch.set_num_threads(1)
    model=LemLite(device='cpu',imagination=False,no_vision=True)
    # All tactile sites exist even if none currently carries contact.
    # THE ROW'S SIZE IS THE CODE'S, NOT A NUMBER TYPED HERE. It was 12 and the row has been
    # widened twice since, to 14 and then to 19, so this fed a 12-wide row into a 19-wide
    # encoder and the test failed on the shapes rather than on anything it was written to
    # check. The validity flag moves to the new last column on each widening, which is why it
    # is set by name.
    frame={'tactile':torch.randn(2,TOUCH_TOKENS,TOUCH_TOKEN_F)}
    frame['tactile'][...,TOUCH_VALID_COL]=1
    mask={'tactile':'real'}
    old=model.encoder_registry.encode(frame,mask)
    current,regions,valid=encode_spatial(model.encoder_registry,frame,mask)
    torch.testing.assert_close(current['tactile'],old['tactile'],rtol=0,atol=0)
    assert regions.shape==(2,TOUCH_TOKENS,model.d_model) and valid.all()


def test_actual_eye_regions_reach_native_self_forecast_and_saved_history(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge import continuation_state
    torch.set_num_threads(1)
    source=SingleEnvSource(scene='crib',vision=True,run_dir=str(tmp_path/'world'),age=24,
        caregiver_seconds=0,body_function='modulate',contact_profile='mechanical_v2',
        fovea_size=72,ring_size=64)
    try:
        model=LemLite(device='cpu',imagination=False,self_state='learn',self_prediction='grounded_residual')
        frames=source.reset();command=None
        for i in range(3):
            result=model.step(frames,model.dev_limits(0),train=True,step_index=i,executed_action=command)
            command=result.motor_for_action;frames=source.step(command)
        inputs=model.self_history.inputs
        assert inputs['spatial'].shape[1]>32
        assert inputs['spatial_valid'].any() and model.last_self_state['spatial_regions']>32
        assert not inputs['spatial'].requires_grad
        copied=continuation_state.capture_learner(model)
        assert 'spatial' in copied['self_history']['inputs']
        prediction,_=model.self_state.forecast(model.self_state.perceive(**inputs),
            torch.zeros(1,1,len(model.handle_names)*9),torch.full((1,1),.05))
        assert torch.isfinite(prediction).all()
    finally:source.close()
