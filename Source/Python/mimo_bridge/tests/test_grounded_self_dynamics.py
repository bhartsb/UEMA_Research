# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Local grounding, future causality and additive checkpoint preservation."""
import copy
import numpy as np
import pytest
import torch

from lem.model.self_state import SelfState,SITE_INPUT,balanced_function_mean


def observation(batch=3,sites=4,width=16):
    values=torch.zeros(batch,sites,SITE_INPUT)
    values[...,-3]=torch.rand(batch,sites)*.5+.25
    return dict(sites=values,valid=torch.ones(batch,sites,dtype=torch.bool),context=torch.randn(batch,width))


def test_grounded_initialization_keeps_observed_sites_and_explicit_uncertainty():
    module=SelfState(16,1,9,'grounded_residual');inputs=observation()
    p=module.perceive(**inputs)
    assert torch.equal(p['observed'],inputs['sites'][...,-3:])
    assert torch.equal(p['present'],p['observed'])
    assert not torch.equal(p['learned_present'],p['observed'])
    mean,var=module.forecast(p,torch.randn(3,5,9),torch.full((3,5),.05))
    assert torch.equal(mean,p['observed'][:,None].expand_as(mean))
    assert var.eq(1).all()  # Untrained persistence cannot be a confident chooser.


def test_available_sites_cannot_dilute_changed_site_lessons():
    truth=torch.tensor([[[.5,0.,0.],[1.,0.,0.]]])
    error=torch.zeros_like(truth);error[:,0,0]=.5;error[:,1,0]=.1
    first=balanced_function_mean(error,truth,torch.ones_like(truth,dtype=torch.bool))
    truth=torch.cat([truth[:,:1],truth[:,1:].repeat(1,100,1)],1)
    error=torch.cat([error[:,:1],error[:,1:].repeat(1,100,1)],1)
    second=balanced_function_mean(error,truth,torch.ones_like(truth,dtype=torch.bool))
    torch.testing.assert_close(first,second)
    error[:,1:,0]=.9
    assert balanced_function_mean(error,truth,torch.ones_like(truth,dtype=torch.bool))>second


def test_paired_action_lesson_rejects_absent_variation_and_penalizes_common_bias():
    module=SelfState(16,1,2,'grounded_residual')
    target=torch.tensor([[[[.3,.2,.1]]],[[[.7,.1,.2]]]])
    valid=torch.ones(2,1,1,dtype=torch.bool);variance=torch.full_like(target,.02)
    exact=module.paired_consequence_loss(target,variance,target,valid)
    # Correct action differences are insufficient when every prediction is offset.
    biased=module.paired_consequence_loss(target+.1,variance,target,valid)
    swapped=module.paired_consequence_loss(target.flip(0),variance,target,valid)
    assert biased>exact and swapped>exact
    with pytest.raises(ValueError,match='no measured action'):
        module.paired_consequence_loss(target[:1].repeat(2,1,1,1),variance,target[:1].repeat(2,1,1,1),valid)
    with pytest.raises(ValueError,match='identical site validity'):
        module.paired_consequence_loss(target,variance,target,torch.tensor([[[True]],[[False]]]))


def test_action_calibration_is_recorded_and_does_not_divide_by_constant_channel():
    module=SelfState(16,1,2,'grounded_residual')
    commands=torch.tensor([[.001,1.],[.002,1.],[.003,1.]])
    module.grounded.calibrate_actions(commands)
    scaled=(commands-module.grounded.action_center)/module.grounded.action_scale
    torch.testing.assert_close(scaled[:,0].std(unbiased=False),torch.tensor(1.))
    assert scaled[:,1].eq(0).all()
    copied=SelfState(16,1,2,'grounded_residual');copied.load_state_dict(module.state_dict())
    assert torch.equal(copied.grounded.action_scale,module.grounded.action_scale)
    with pytest.raises(ValueError):module.grounded.calibrate_actions(commands*float('nan'))


def test_grounded_current_motor_capacity_keeps_invalid_channels_invalid():
    module=SelfState(16,1,9,'grounded_residual');inputs=observation()
    inputs.update(capacity=torch.full((3,10),.25),capacity_valid=torch.zeros(3,10,dtype=torch.bool))
    inputs['capacity_valid'][:,4]=True;p=module.perceive(**inputs)
    assert p['valid'][:,4:].sum()==3
    assert p['present'][:,8,0].eq(.25).all()
    altered=copy.deepcopy(inputs);altered['sites'][2,...,-3]=.1
    torch.testing.assert_close(module.perceive(**altered)['present'][0],p['present'][0],rtol=0,atol=0)


def test_new_future_learns_local_action_and_duration_without_degrading_healthy_sites():
    torch.set_num_threads(1);torch.manual_seed(713)
    module=SelfState(16,1,2,'grounded_residual')
    for name,param in module.named_parameters():param.requires_grad_(name.startswith('grounded.'))
    old={name:value.clone() for name,value in module.state_dict().items() if not name.startswith('grounded.')}
    optimizer=torch.optim.Adam(module.grounded.parameters(),lr=.004)
    def batch(n):
        inputs=observation(n,2)
        inputs['sites'][:,1,-3]=1.;inputs['sites'][:,0,0]=-1.;inputs['sites'][:,1,0]=1.
        actions=torch.rand(n,1,2)*2-1;durations=.05+.2*torch.rand(n,1)
        target=inputs['sites'][:,:,-3:].clone()[:,None]
        target[:,0,0,0]+=.7*actions[:,0,0]*durations[:,0]
        return inputs,actions,durations,target
    for _ in range(240):
        inputs,actions,durations,target=batch(32);p=module.perceive(**inputs)
        mean,var=module.forecast(p,actions,durations)
        loss=module.learning_loss(mean,var,target,p['valid'][:,None],initial=p['observed'])
        optimizer.zero_grad();loss.backward();optimizer.step()
    inputs,actions,durations,target=batch(128)
    with torch.no_grad():
        p=module.perceive(**inputs);mean,_=module.forecast(p,actions,durations)
        wrong,_=module.forecast(p,actions.flip(0),durations)
        wrong_time,_=module.forecast(p,actions,durations*.1)
    error=(mean[:,:,0,0]-target[:,:,0,0]).square().mean()
    blind=(p['observed'][:,None,:,0]-target[...,0]).square()[:,:,0].mean()
    assert error<.15*blind
    assert (wrong[:,:,0,0]-target[:,:,0,0]).square().mean()>5*error
    assert (wrong_time[:,:,0,0]-target[:,:,0,0]).square().mean()>5*error
    assert (mean[:,:,1,0]-1.).abs().mean()<.02
    assert all(torch.equal(value,module.state_dict()[name]) for name,value in old.items())


def test_native_checkpoint_growth_preserves_every_tensor_and_optimizer_moment(tmp_path):
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.tools.migrate_trunk import modules
    torch.set_num_threads(1)
    options=dict(device='cpu',imagination=False,no_vision=True,self_state='learn')
    old=LemLite(**options,checkpoint_dir=str(tmp_path/'old'))
    # Populate every old parameter, including old self heads, to prevent an
    # unexercised empty optimizer from falsely establishing migration.
    loss=sum(p.square().sum() for group in old.optimizer.param_groups for p in group['params'])
    loss.backward();old.optimizer.step();checkpoint=old.save_checkpoint(1)
    grown=LemLite(**options,self_prediction='grounded_residual',checkpoint_dir=str(tmp_path/'new'))
    result=grown.load_checkpoint(checkpoint);assert not result['reinitialized_components']
    new_modules=modules(grown)
    for name,module in modules(old).items():
        for key,value in module.state_dict().items():assert torch.equal(value,new_modules[name].state_dict()[key])
        old_params=dict(module.named_parameters());new_params=dict(new_modules[name].named_parameters())
        for key,param in old_params.items():
            for state,value in old.optimizer.state[param].items():
                assert torch.equal(value,grown.optimizer.state[new_params[key]][state]) if torch.is_tensor(value) else value==grown.optimizer.state[new_params[key]][state]
    assert not any(param in grown.optimizer.state for param in grown.self_state.grounded.parameters())
    saved=grown.save_checkpoint(1)
    with pytest.raises(ValueError,match='grounded self dynamics'):old.load_checkpoint(saved)
    restored=LemLite(**options,self_prediction='grounded_residual');restored.load_checkpoint(saved)
    for name,value in grown.self_state.state_dict().items():assert torch.equal(value,restored.self_state.state_dict()[name])


def test_native_grounded_learning_preserves_old_updates_and_continues_physical_episode(tmp_path):
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.tools.migrate_trunk import modules
    from mimo_bridge import continuation_state
    torch.set_num_threads(1)
    source=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path/'world'),age=24,
                          caregiver_seconds=0,body_function='modulate',contact_profile='mechanical_v2')
    try:
        options=dict(device='cpu',imagination=False,no_vision=True,self_state='learn',next_frame_loss=True)
        old=LemLite(**options,checkpoint_dir=str(tmp_path/'old'));frames=source.reset()
        checkpoint=old.save_checkpoint(0)
        new=LemLite(**options,self_prediction='grounded_residual',checkpoint_dir=str(tmp_path/'new'))
        new.load_checkpoint(checkpoint);dev=old.dev_limits(0);command=None
        state=continuation_state.capture_learner(old);continuation_state.restore_learner(new,state)
        for step in range(6):
            rng=torch.get_rng_state()
            expected=old.step(frames,dev,train=True,step_index=step,executed_action=command)
            torch.set_rng_state(rng)
            actual=new.step(frames,dev,train=True,step_index=step,executed_action=command)
            np.testing.assert_array_equal(actual.motor_for_action,expected.motor_for_action)
            command=expected.motor_for_action;frames=source.step(command)
        target_modules=modules(new)
        for name,module in modules(old).items():
            if name=='self_state':continue
            for key,value in module.state_dict().items():assert torch.equal(value,target_modules[name].state_dict()[key]),(name,key)
        # observed_grounding_error is a plumbing check, not a reading: in this mode
        # percept['present'] IS the measurement, so it is |x - x| and zero for every
        # possible input. How far the learned belief sits from the measurement is the
        # worst readout row, and the forecast's do-nothing floor is logged beside it.
        assert new.last_self_state['observed_grounding_error']==0
        worst=new.last_self_state['learned_present_worst_site_error']
        assert np.isfinite(worst) and 0<=worst<=1
        assert np.isfinite(new.last_self_state['persistence_error_1'])
        assert new.self_state.grounded.change.weight.abs().sum()>0
        saved=new.save_checkpoint(6)
        continuation_state.save(saved,source,new,dict(step=6,cmd=command),dict(self_prediction='grounded_residual'))
        retained=continuation_state.load(saved)
        expected=new.step(frames,dev,train=False,step_index=6,executed_action=command)
        restored=LemLite(**options,self_prediction='grounded_residual')
        restored.load_checkpoint(saved);continuation_state.restore_learner(restored,retained['learner'])
        actual=restored.step(frames,dev,train=False,step_index=6,executed_action=command)
        np.testing.assert_array_equal(actual.motor_for_action,expected.motor_for_action)
        assert actual.loss==expected.loss
    finally:source.close()


# ---------------------------------------------------------------- audit P203 (2026-09-18)
def test_action_recalibration_normalises_the_command_and_leaves_the_forecast_unchanged():
    torch.manual_seed(4211)
    module=SelfState(16,1,4,'grounded_residual')
    with torch.no_grad():
        module.grounded.command.weight.normal_(0,.3);module.grounded.command.bias.normal_(0,.1)
        module.grounded.change.weight.normal_(0,.1);module.grounded.uncertainty.weight.normal_(0,.1)
        # a deliberately stale calibration: a centre far from the live orders and a
        # scale 100 times narrower than their real spread, plus one constant column
        module.grounded.action_center.copy_(torch.tensor([.5,-.5,.5,0.]))
        module.grounded.action_scale.copy_(torch.tensor([.004,.004,.004,1.]))
    orders=torch.cat([torch.rand(64,3)*2-1,torch.full((64,1),.25)],-1)
    inputs=observation()
    percept=module.perceive(**inputs)
    actions=orders[:len(inputs['sites'])][:,None].expand(-1,3,-1).contiguous()
    durations=torch.full(actions.shape[:2],.05)
    with torch.no_grad():
        before_mean,before_var=module.forecast(percept,actions,durations)
        stale=((orders-module.grounded.action_center)/module.grounded.action_scale).abs().mean()
        record=module.grounded.recalibrate_actions(orders)
        after_mean,after_var=module.forecast(module.perceive(**inputs),actions,durations)
    # 1. every predicted value survives the refresh
    torch.testing.assert_close(after_mean,before_mean,rtol=0,atol=1e-6)
    torch.testing.assert_close(after_var,before_var,rtol=0,atol=1e-4)
    # 2. the command the encoder now sees is normalised
    z=(orders-module.grounded.action_center)/module.grounded.action_scale
    torch.testing.assert_close(z[:,:3].std(0,unbiased=False),torch.ones(3),rtol=0,atol=1e-5)
    assert float(stale)>100 and float(z.abs().mean())<1.
    assert record['normalised_absmean_before']>100 and record['normalised_absmean_after']<1.
    # 3. a command column that never varies keeps scale 1.0 and normalises to zero
    assert module.grounded.action_scale[3]==1. and z[:,3].eq(0).all()
    # 4. a refresh on unusable orders refuses instead of writing junk
    with pytest.raises(ValueError):module.grounded.recalibrate_actions(orders*float('nan'))


# ---------------------------------------------------------------- audit P213 (2026-09-18)
def _both_inputs(batch=3,sites=4,width=16):
    inputs=observation(batch,sites,width)
    inputs['function_extent']=torch.rand(batch,sites,2)
    inputs['surface_moments']=torch.randn(batch,sites,20)
    return inputs


def test_site_bias_is_one_constant_not_two():
    torch.manual_seed(11)
    module=SelfState(16,1,4,'grounded_residual')
    assert module.grounded.surface_encoder.bias.requires_grad is False
    opt=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=1e-3)
    for _ in range(5):
        p=module.perceive(**_both_inputs())
        loss=p['learned_present'].square().mean()+p['belief'].square().mean()
        opt.zero_grad();loss.backward();opt.step()
    assert int(module.grounded.surface_encoder.bias.count_nonzero())==0   # the retired constant never moves
    assert module.grounded.scope_encoder.bias.grad is not None
    assert float(module.grounded.scope_encoder.bias.grad.abs().mean())>0   # the surviving constant learns
    ones=[(n,t) for n,t in module.named_parameters() if t.ndim==1]
    for i,(n1,t1) in enumerate(ones):
        for n2,t2 in ones[i+1:]:
            if t1.shape==t2.shape and t1.count_nonzero() and t2.count_nonzero():
                assert not torch.equal(t1,t2),(n1,n2)


def test_legacy_duplicate_bias_folds_without_moving_the_site_token():
    torch.manual_seed(12)
    a=SelfState(16,1,4,'grounded_residual')
    b_vec=torch.randn(16)*.1
    with torch.no_grad():
        a.grounded.scope_encoder.bias.copy_(b_vec);a.grounded.surface_encoder.bias.copy_(b_vec)
        a.grounded.scope_encoder.weight.normal_(0,.1);a.grounded.surface_encoder.weight.normal_(0,.1)
    inputs=_both_inputs()
    with torch.no_grad():ref=a.perceive(**inputs)['learned_present']
    b=SelfState(16,1,4,'grounded_residual');b.load_state_dict(a.state_dict())
    assert torch.equal(b.grounded.scope_encoder.bias,2*b_vec)
    assert int(b.grounded.surface_encoder.bias.count_nonzero())==0
    with torch.no_grad():torch.testing.assert_close(b.perceive(**inputs)['learned_present'],ref,rtol=0,atol=1e-6)
    assert sorted(dict(a.named_parameters()))==sorted(dict(b.named_parameters()))
    c=SelfState(16,1,4,'grounded_residual');c.load_state_dict(b.state_dict())
    assert torch.equal(c.grounded.scope_encoder.bias,2*b_vec)   # idempotent: never folds twice


# ---------------------------------------------------------------- S7 (2026-09-18): vectorised lesson
def test_balanced_function_mean_matches_the_loop():
    from lem.model.self_state import _balanced_function_mean_loop,STATE_FIELDS
    torch.manual_seed(21)
    for trial in range(6):
        B,S=(3,53) if trial%2 else (2,7)
        values=torch.rand(B,S,STATE_FIELDS,requires_grad=True)
        target=torch.rand(B,S,STATE_FIELDS)
        target[...,0][torch.rand(B,S)<.3]=1.          # exact endpoints: 'unaffected' field-0 sites
        target[...,1:][torch.rand(B,S,STATE_FIELDS-1)<.3]=0.
        valid=torch.rand(B,S,STATE_FIELDS)<.8
        valid[:,0]=True
        initial=None if trial<3 else target[:,0].clone()+(torch.rand(B,STATE_FIELDS)<.5).float()*.2
        a=balanced_function_mean(values,target,valid,initial)
        b=_balanced_function_mean_loop(values,target,valid,initial)
        torch.testing.assert_close(a,b,rtol=0,atol=1e-6)
        ga,=torch.autograd.grad(a,values,retain_graph=True)
        gb,=torch.autograd.grad(b,values)
        torch.testing.assert_close(ga,gb,rtol=0,atol=1e-6)
    with pytest.raises(ValueError):
        balanced_function_mean(torch.zeros(1,2,STATE_FIELDS),torch.zeros(1,2,STATE_FIELDS),torch.zeros(1,2,STATE_FIELDS,dtype=torch.bool))


# ---------------------------------------------------------------- audit P204 (2026-09-18)
LEGACY_HEAD={'action_encoder.weight','action_encoder.bias','transition.weight_ih','transition.weight_hh',
             'transition.bias_ih','transition.bias_hh','future.0.weight','future.0.bias','future.2.weight','future.2.bias'}
RESIDUAL={'residual.weight','residual.bias'}


def _census(perceive_mode_loss):
    torch.manual_seed(5)
    module=SelfState(32,2,6,'grounded_residual')
    inputs=observation(3,4,64)
    p=module.perceive(**inputs)
    actions=torch.rand(3,3,6)*2-1;durations=torch.full((3,3),.05)
    mean,var=module.forecast(p,actions,durations)
    target=(p['observed'][:,None].expand_as(mean)+.01).clamp(0,1)
    loss=module.present_loss(p,p['observed'])+module.learning_loss(mean,var,target,p['valid'][:,None].expand(-1,3,-1),initial=p['observed'])
    if perceive_mode_loss:loss=loss+p['bias'].square().mean()
    loss.backward()
    return {n for n,q in module.named_parameters() if q.requires_grad and q.grad is None}


def test_grounded_residual_learn_mode_leaves_these_weights_without_gradient():
    """The live census behind the page's 262,022: under --self-prediction grounded_residual
    the legacy forecast head is never executed, and under --self-state learn the
    belief-into-perception residual is computed and discarded. (With these minimal
    inputs the capacity, regime, relation and spatial encoders are also unexercised;
    the two groups the page names must be inside the idle set either way.)"""
    idle=_census(perceive_mode_loss=False)
    assert LEGACY_HEAD<=idle and RESIDUAL<=idle
    reached={n for n,_ in SelfState(32,2,6,'grounded_residual').named_parameters()}-idle
    assert any(n.startswith('grounded.') for n in reached) and 'present.weight' in reached
    # what 'perceive' mode does: the residual pair drops out, the legacy head stays idle
    idle2=_census(perceive_mode_loss=True)
    assert LEGACY_HEAD<=idle2 and not (RESIDUAL & idle2)


def test_the_future_evidence_guard_names_the_check_that_failed():
    """The bare sentence 'invalid or absent future self-state evidence' cost 18
    crash-resume cycles on 2026-09-20 (H84 segment 13, 6.4 hours of the run making no
    progress) without saying whether the evidence was missing, mis-shaped, non-finite or
    zero-variance. Each failure now carries its own count."""
    from lem.model.self_state import SelfState
    mean = torch.rand(2, 3, 4); var = torch.rand(2, 3, 4) + 0.1
    target = torch.rand(2, 3, 4); valid = torch.ones(2, 3, dtype=torch.bool)
    SelfState._validate_consequence(mean, var, target, valid)      # healthy: no raise
    for phrase, args in (
            ('no future evidence is marked valid', (mean, var, target, torch.zeros(2, 3, dtype=torch.bool))),
            ('non-finite', (mean * float('nan'), var, target, valid)),
            ('smallest variance', (mean, var * 0, target, valid)),
            ('shapes disagree', (mean, var, target[:, :2], valid))):
        with pytest.raises(ValueError, match='invalid or absent future self-state evidence'):
            SelfState._validate_consequence(*args)
        try:
            SelfState._validate_consequence(*args)
        except ValueError as raised:
            assert phrase in str(raised), (phrase, str(raised))


def test_the_stored_self_percept_cannot_feed_itself_without_bound():
    """The measured cause of the 2026-09-20 halt. perceive() writes its self-percept into
    the relation memory and the next perceive() attends over those entries to build the
    next one, so any gain above 1 in that path compounds every step. On H84 the 128 stored
    percepts ran 1.29 to 1.49 while healthy and 2.1e+34 to 3.8e+35 on the diverged run -
    inputs ordinary throughout (context 4.13 vs 4.27) - until one body's forecast overflowed
    to 945 non-finite values and the guard stopped the run. Re-measured on that same crash
    checkpoint after this bound: 0 non-finite at horizons 1, 5, 20 and 100."""
    from lem.model.self_state import SelfState, SITE_INPUT, BELIEF_LIMIT
    torch.manual_seed(0)
    b, n, d, mods = 2, 6, 16, 3
    m = SelfState(d, mods, action_size=5, prediction_mode='grounded_residual').eval()
    sites = torch.rand(b, n, SITE_INPUT)
    valid = torch.ones(b, n, dtype=torch.bool)
    context = torch.randn(b, mods * d)
    with torch.no_grad():
        ordinary = m.perceive(sites=sites, valid=valid, context=context,
                              relations=torch.randn(b, 4, d))
    assert m.belief_bounded_count == 0, 'ordinary sizes must not be touched'
    assert float(ordinary['belief'].abs().max()) < BELIEF_LIMIT
    with torch.no_grad():
        runaway = m.perceive(sites=sites, valid=valid, context=context,
                             relations=torch.randn(b, 4, d) * 1e35)
    assert m.belief_bounded_count == 1, 'the runaway was not counted'
    assert torch.isfinite(runaway['belief']).all()
    assert float(runaway['belief'].abs().max()) <= BELIEF_LIMIT + 1e-4
    assert torch.isfinite(runaway['local']).all()
    with torch.no_grad():
        mean, var = m.forecast(runaway, torch.randn(b, 5, 5), torch.full((b, 5), 0.05))
    assert torch.isfinite(mean).all() and torch.isfinite(var).all()
    assert (var > 0).all()


def _held_at_the_limit(release, steps=12):
    """Two bodies through self_state_integration.observe() with a hand-built loop that
    recalls the stored percepts 3 times larger - H84's held body recalled 2 to 3 times the
    limit on 2026-09-25 (median 16 to 23 against 8 in each segment). The GRU's part is zero,
    attention is uniform, and the recall passes 3x the stored percept straight through.
    Body 0 starts with 16 stored percepts at the limit, body 1 with 16 zero ones."""
    from types import SimpleNamespace
    from lem.body_function import CONTRACT, FUNCTION_FIELDS
    from lem.model.self_state import SelfHistory, BELIEF_LIMIT
    from lem.model.self_rehearsal import SelfRehearsal
    from mimo_bridge.self_state_integration import observe
    torch.manual_seed(0)
    torch.set_num_threads(1)
    d = 8
    s = SelfState(d, 1, 9)
    with torch.no_grad():
        s.body_context.weight.zero_()
        s.body_context.bias.zero_()
        s.body_context.weight[:, d:] = torch.eye(d)
        s.relation_value.weight.copy_(3 * torch.eye(d))
        s.relation_value.bias.zero_()
        s.relation_query.weight.zero_()
        s.relation_query.bias.zero_()
    history = SelfHistory()
    held = torch.zeros(2, d)
    held[0, 0] = BELIEF_LIMIT
    for _ in range(16):
        history.relations.append(held.clone())
    m = SimpleNamespace(device=torch.device('cpu'), active=['sense'], self_state=s,
                        self_history=history, self_rehearsal=SelfRehearsal(8))
    if release is not None:
        m.self_percept_release = release
    tokens = {'sense': torch.zeros(2, 1, d)}
    for i in range(steps):
        m.step_count = i
        t = i * .05
        frames = [SimpleNamespace(
            meta={'body_function': dict(contract=CONTRACT, mode='observe', fields=FUNCTION_FIELDS,
                                        time=t, values=[[1., 0., 0.]], site_ids=['one'])},
            touch_pos=np.zeros((1, 3)), sim_time=t,
            tokenized=SimpleNamespace(touch_impulses=torch.zeros(1, 8))) for _ in range(2)]
        with torch.no_grad():
            observe(m, tokens, frames, None if i == 0 else torch.zeros(2, 1, 9))
    return m, torch.stack(list(m.self_history.relations)).abs().amax(-1)   # [stored, body]


def test_a_rescaled_self_percept_stays_at_the_limit_without_the_release():
    """What H84 did from 7:41 AM on 2026-09-25 until the 8:11 PM pause, 12.5 hours across 9
    resumes: perceive() rescales the percept to the limit keeping its direction, the stored
    percept is recalled 3x larger, rescaled again and stored again. Every live step is
    rescaled and every stored row of that body sits at the limit. Unchanged by default."""
    from lem.model.self_state import BELIEF_LIMIT
    for release in (None, False):
        m, sizes = _held_at_the_limit(release)
        assert m.self_state.belief_bounded_count >= 12, 'every live step must have been rescaled'
        assert float((sizes[:, 0] - BELIEF_LIMIT).abs().max()) < 1e-4, 'body 0 left the limit'
        assert float(sizes[:, 1].max()) == 0.0, 'body 1 changed'


def test_the_release_starts_a_held_body_again_and_leaves_the_others():
    """Fails on the code before 2026-09-26, where the stored percepts stay at the limit.
    With --self-percept-release on, a body whose restored stored percepts sit at the limit is
    released BEFORE the first live step (second round: the first round released it on that
    step, after it had run over the limit once), so no live step runs over the limit, and the
    other body's stored percepts are untouched. Measured on H84's recorded steps (checkpoints
    407,032 to 415,075, re-run offline): under the limit on all 1,024 replayed steps."""
    from lem.model.self_state import BELIEF_LIMIT
    m, sizes = _held_at_the_limit(True)
    assert float(sizes[:, 0].max()) == 0.0, 'body 0 still holds percepts at the limit'
    assert float(sizes[:, 1].max()) == 0.0, 'body 1 changed'
    assert len(m.self_history.relations) == 16 + 12
    last = m.last_self_state
    assert last['window_restored_bodies'] == 1 and last['window_steps'] == 12
    assert last['window_live_over_limit'] == 0, 'a live step ran over the limit'
    assert last['self_percept_releases'] == 0 and last['window_bounded_calls'] == 0
    assert last['window_belief_size_max'] < BELIEF_LIMIT


def _loop(fixture, cap, release, steps, report_every=100):
    """Drive two bodies through observe() for `steps` steps and return one stats record per
    report, the way train_mimo.flush_stats writes them (self_state_report closes the window)."""
    from types import SimpleNamespace
    from lem.body_function import CONTRACT, FUNCTION_FIELDS
    from lem.model.self_state import SelfHistory
    from lem.model.self_rehearsal import SelfRehearsal
    from mimo_bridge.self_state_integration import observe
    from mimo_bridge.train_mimo import self_state_report
    torch.manual_seed(0)
    torch.set_num_threads(1)
    s, history = fixture()
    s.attention_peak_limit = cap
    m = SimpleNamespace(device=torch.device('cpu'), active=['sense'], self_state=s, self_history=history,
                        self_rehearsal=SelfRehearsal(8), self_percept_release=release)
    tokens = {'sense': torch.zeros(2, 1, s.d_model)}
    records = []
    for i in range(1, steps + 1):
        m.step_count = i
        t = i * .05
        frames = [SimpleNamespace(
            meta={'body_function': dict(contract=CONTRACT, mode='observe', fields=FUNCTION_FIELDS,
                                        time=t, values=[[1., 0., 0.]], site_ids=['one'])},
            touch_pos=np.zeros((1, 3)), sim_time=t,
            tokenized=SimpleNamespace(touch_impulses=torch.zeros(1, 8))) for _ in range(2)]
        with torch.no_grad():
            observe(m, tokens, frames, None if i == 1 else torch.zeros(2, 1, 9))
        if i % report_every == 0:
            records.append(dict(step=i, self_state=self_state_report(m)))
    return m, records


def _stuck_again_fixture(drive=.3):
    """The review's planted body (d932dcbcd, section 4): the recall passes 3x the stored percept
    with attention spread evenly, plus a constant push, so a zeroed memory grows back to the
    limit. Body 0 starts with 128 stored percepts at the limit. Even attention means the 5% cap
    never binds: this body sticks again whatever the cap does, and only the counts can see it."""
    from lem.model.self_state import SelfHistory, BELIEF_LIMIT
    d = 8
    s = SelfState(d, 1, 9)
    with torch.no_grad():
        s.body_context.weight.zero_(); s.body_context.bias.zero_()
        s.body_context.weight[:, d:] = torch.eye(d)
        s.body_context.bias[0] = drive
        s.relation_value.weight.copy_(3 * torch.eye(d)); s.relation_value.bias.zero_()
        s.relation_query.weight.zero_(); s.relation_query.bias.zero_()
    history = SelfHistory()
    held = torch.zeros(2, d); held[0, 0] = BELIEF_LIMIT
    for _ in range(128):
        history.relations.append(held.clone())
    return s, history


def _mirror_fixture(drive=.5, gain=-3., sharp=50.):
    """H84's stuck state in miniature (second round). The recall is a NEGATIVE feedback, -3 x the
    stored percept (H84's recall path: largest eigenvalues -2.1 to -4.5, none with a real part above
    0.02), and the attention picks the stored percept with the largest first number (H84's stuck
    bodies: one stored percept carried 1.000 of the recall). Each new percept is the attended one's
    mirror image, 3 times larger, so the loop climbs to the limit and the limit holds it there."""
    from lem.model.self_state import SelfHistory
    d = 8
    s = SelfState(d, 1, 9)
    with torch.no_grad():
        s.body_context.weight.zero_(); s.body_context.bias.zero_()
        s.body_context.weight[:, d:] = torch.eye(d)
        s.body_context.bias[0] = drive
        s.relation_value.weight.copy_(gain * torch.eye(d)); s.relation_value.bias.zero_()
        s.relation_query.weight.zero_(); s.relation_query.bias.zero_(); s.relation_query.bias[0] = sharp
        s.relation_key.weight.copy_(torch.eye(d)); s.relation_key.bias.zero_()
    return s, SelfHistory()


def test_the_acceptance_check_fails_a_body_that_gets_stuck_again():
    """Review finding (d932dcbcd): the planted body, released and stuck again every 144 steps,
    passed all three first-round lines, which read the last step of each 100. The per-step
    counts see every release, and the declared stop line (2 or more live steps over the limit
    within 1,000 steps) is crossed; the same fixture with no push releases once and passes."""
    from mimo_bridge.tools.self_percept_watch import check
    from lem.model.self_state import ATTENTION_PEAK_LIMIT
    m, records = _loop(_stuck_again_fixture, ATTENTION_PEAK_LIMIT, True, 1100)
    over = [r['self_state']['window_live_over_limit'] for r in records]
    assert sum(over) >= 6, over                                  # about one every 144 steps
    # the first round's reading: the last step of each report was under 8 in every record
    assert all(r['self_state']['belief_size'] < 8 for r in records[1:])
    result = check(records)
    assert result['verdict'] == 'stop' and result['lines']['sticking_again']['crossed'], result
    _, calm = _loop(lambda: _stuck_again_fixture(0.), ATTENTION_PEAK_LIMIT, True, 1100)
    assert check(calm)['verdict'] == 'pass', check(calm)


def test_the_attention_cap_removes_the_mirror_loop():
    """Second round: why the recall amplifies, and the fix. Without the switch the mirror loop
    sticks at the limit; the release alone breaks it and it sticks again (the first round's
    climb back to 7.93 and the replays' re-release every 130 steps at 415,075); with the release
    and the 5% cap the recall stays an average, the loop never reaches the limit and the
    acceptance check passes."""
    from mimo_bridge.tools.self_percept_watch import check
    from lem.model.self_state import ATTENTION_PEAK_LIMIT
    _, off = _loop(_mirror_fixture, None, False, 1000)
    assert sum(r['self_state']['window_live_over_limit'] for r in off) > 500
    _, release_only = _loop(_mirror_fixture, None, True, 1000)
    assert sum(r['self_state']['window_releases'] for r in release_only) >= 5
    assert check(release_only)['verdict'] == 'stop'
    _, fixed = _loop(_mirror_fixture, ATTENTION_PEAK_LIMIT, True, 1000)
    assert sum(r['self_state']['window_live_over_limit'] for r in fixed) == 0
    assert max(r['self_state']['window_belief_size_max'] for r in fixed) < 2
    assert fixed[-1]['self_state']['window_attention_peak_max'] < ATTENTION_PEAK_LIMIT
    assert check(fixed)['verdict'] == 'pass'


def test_the_attention_cap_leaves_attention_under_it_value_for_value():
    """A cap that does not bind changes nothing; one that binds keeps every weight at or under it,
    keeps the ranking, and lets the gradient through; too few stored percepts give the average."""
    from lem.model.self_state import cap_attention
    torch.manual_seed(3)
    scores = torch.randn(3, 128) * 0.1
    s = SelfState(8, 1, 9)
    rel = torch.randn(3, 40, 8) * 0.1
    args = dict(sites=torch.rand(3, 2, 14), valid=torch.ones(3, 2, dtype=torch.bool), context=torch.randn(3, 8),
                relations=rel)
    s.attention_peak_limit = None
    base = s.perceive(**args)
    s.attention_peak_limit = 1.0
    same = s.perceive(**args)
    assert torch.equal(base['belief'], same['belief']) and s.last_attention_capped == 0
    sharp = scores.clone(); sharp[:, 5] += 12.
    sharp.requires_grad_(True)
    capped = cap_attention(sharp, 0.05)
    assert float(capped.max()) <= 0.05 + 1e-4 and torch.equal(capped.argmax(-1), sharp.argmax(-1))
    torch.testing.assert_close(capped.sum(-1), torch.ones(3))
    capped[:, 5].sum().backward()
    assert sharp.grad is not None and float(sharp.grad.abs().sum()) > 0
    few = cap_attention(torch.randn(2, 10), 0.05)
    torch.testing.assert_close(few, torch.full((2, 10), 0.1))


def test_a_release_clears_the_saved_inputs_and_the_held_rehearsal_windows():
    """Coordinator decision (second round): a released body's stuck percepts leave the rehearsal
    windows that hold them (3 of 128 at 415,075) and the saved step inputs that would copy them
    into new windows; healthy windows and the other bodies' rows are untouched."""
    from types import SimpleNamespace
    from collections import deque
    from lem.model.self_state import SelfHistory, BELIEF_LIMIT
    from lem.model.self_rehearsal import SelfRehearsal
    from mimo_bridge.self_state_integration import clear_released
    stuck = torch.ones(2, 4, 3); stuck[0] *= BELIEF_LIMIT             # body 0 at the limit, body 1 healthy
    healthy = torch.ones(2, 4, 3)
    history = SelfHistory()
    history.events = deque([dict(inputs=dict(relations=stuck.clone())) for _ in range(3)], maxlen=128)
    history.inputs = dict(relations=stuck.clone())
    rehearsal = SelfRehearsal(4)
    rehearsal.common = [dict(inputs=dict(relations=stuck.clone())), dict(inputs=dict(relations=healthy.clone()))]
    rehearsal.changed = [(1., dict(inputs=dict(relations=stuck.clone())))]
    m = SimpleNamespace(self_history=history, self_rehearsal=rehearsal)
    events, windows = clear_released(m, torch.tensor([True, False]))
    assert (events, windows) == (4, 2)
    for rel in [e['inputs']['relations'] for e in history.events] + [history.inputs['relations']]:
        assert float(rel[0].abs().max()) == 0.0 and torch.equal(rel[1], healthy[1])
    assert float(rehearsal.common[0]['inputs']['relations'][0].abs().max()) == 0.0
    assert torch.equal(rehearsal.common[1]['inputs']['relations'], healthy)
    assert torch.equal(rehearsal.changed[0][1]['inputs']['relations'][1], stuck[1])


def test_the_release_zeroes_only_the_bodies_that_ran_over():
    """SelfHistory.store: off (None) and on-with-nobody-over are the old append exactly."""
    from lem.model.self_state import SelfHistory
    h = SelfHistory(4)
    for k in range(3):
        h.relations.append(torch.tensor([[8., 0.], [float(k + 1), 1.]]))
    new = torch.tensor([[8., 0.], [5., 5.]])
    assert h.store(new, torch.tensor([True, False])) == 3
    stacked = torch.stack(list(h.relations))
    assert float(stacked[:, 0].abs().max()) == 0.0
    torch.testing.assert_close(stacked[:, 1], torch.tensor([[1., 1.], [2., 1.], [3., 1.], [5., 5.]]))
    assert h.relations.maxlen == 4
    for released in (None, torch.tensor([False, False])):
        kept = SelfHistory(4)
        kept.relations.append(torch.ones(2, 2))
        assert kept.store(new, released) == 0
        assert kept.relations[-1] is new and len(kept.relations) == 2


def test_the_self_percept_release_is_off_unless_asked_for():
    import inspect
    from mimo_bridge import train_mimo as TM
    from mimo_bridge.lem_model import LemLite
    off = TM.parse_args(['--out', 'unused'])
    assert off.self_percept_release == 'off'
    on = TM.parse_args(['--out', 'unused', '--self-percept-release', 'on'])
    assert on.self_percept_release == 'on'
    assert inspect.signature(LemLite.__init__).parameters['self_percept_release'].default == 'off'
    # a behaviour change, so an acting snapshot qualified without it must not inherit a pass
    assert 'self_percept_release' in TM.acting_configuration(on)
