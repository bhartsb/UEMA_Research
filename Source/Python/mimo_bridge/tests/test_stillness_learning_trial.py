# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mimo_bridge.tools.stillness_learning_trial import MatchedOrders,summarize,prune_superseded_intermediates


def options(arm):
    return SimpleNamespace(no_babble=False,babble_noise=.05,babble_scale=1.,babble_bouts='real',
                           motor_hold=5,action_repeat=1,rest_control=arm)


def test_stillness_comparison_keeps_noise_timing_and_preserves_order_clock():
    free=MatchedOrders(options('model'),913);scheduled=MatchedOrders(options('scheduled'),913)
    traces=[]
    for orders in (free,scheduled):
        traces.append([(orders.limits({'exploration_noise':.2},i,.05,True)['exploration_noise'],orders.bout)
                       for i in range(500)])
    assert traces[0]==traces[1]
    assert any(value==0 for value,bout in traces[0]) and any(value>0 for value,bout in traces[0])
    restored=MatchedOrders(options('model'),0);restored.load_state_dict(free.state_dict())
    assert [(free.limits({},i,.05,True),free.bout) for i in range(500,1000)]==[
        (restored.limits({},i,.05,True),restored.bout) for i in range(500,1000)]


def test_only_scheduled_arm_overrides_body_pose_eyes_and_strength_stay_equal():
    achieved=torch.zeros((1,24,9));achieved[...,:3]=.125;achieved[...,6]=1
    model=SimpleNamespace(last_achieved=achieved)
    proposal=np.zeros((1,24,9),np.float32);proposal[...,3]=1;proposal[...,8]=.4
    class World:
        def step(self,command,gaze,render):
            self.command=command.copy();self.gaze=gaze.copy()
            return [SimpleNamespace(meta={'gaze_cmd':[.1,.2]})]
    worlds=[]
    for arm in ('scheduled','model'):
        order=MatchedOrders(options(arm),913);order.bout=False;source=World()
        order.advance(model,SimpleNamespace(motor_for_action=proposal.copy(),gaze_out=np.array([[.1,.2]])),source,0,True)
        worlds.append(source)
    np.testing.assert_array_equal(worlds[0].command[...,:3],np.full((1,24,3),25.))
    np.testing.assert_array_equal(worlds[1].command,proposal)
    np.testing.assert_array_equal(worlds[0].command[...,8],worlds[1].command[...,8])
    np.testing.assert_array_equal(worlds[0].gaze,worlds[1].gaze)


def paired_worlds():
    return [dict(baseline=dict(gaze_relocated_fraction=.1,function_min=1.),
        model={p:dict(gaze_relocated_fraction=.6,function_min=1.) for p in ('end','retention')},
        scheduled={p:dict(gaze_relocated_fraction=.2,function_min=1.) for p in ('end','retention')}) for _ in range(8)]


def test_behavior_gain_needs_baseline_retention_and_viability_not_lower_prediction_loss():
    worlds=paired_worlds();result=summarize(worlds)
    assert result['verdict']=='FREE_BODY_IMPROVES_RETAINED_OBJECT_LOOKING'
    assert result['capability_pass'] is False and result['production_license'] is False
    assert summarize(worlds,smoke=True)['verdict']=='INSTRUMENT_EXECUTED'
    for mutation in ('baseline','retention','viability','noise'):
        rows=deepcopy(worlds)
        for index,row in enumerate(rows):
            if mutation=='baseline':row['baseline']['gaze_relocated_fraction']=.8
            if mutation=='retention':row['model']['retention']['gaze_relocated_fraction']=.2
            if mutation=='viability':row['model']['retention']['function_min']=.99
            if mutation=='noise':row['model']['retention']['gaze_relocated_fraction']=.8 if index%2 else 0.
        assert summarize(rows)['verdict']=='INCONCLUSIVE'


def test_missing_or_single_world_cannot_establish_a_learning_advantage():
    with pytest.raises(ValueError):summarize([])
    assert summarize(paired_worlds()[:1])['verdict']=='INCONCLUSIVE'


def test_pruning_keeps_parent_endpoints_latest_and_intermediate_hashes(tmp_path):
    folder=tmp_path/'copied_arm';root=folder/'checkpoints';root.mkdir(parents=True)
    inventory=[]
    for step,phase in ((1,'warm'),(2,'fitting'),(3,'end'),(4,'retention')):
        path=root/f'ckpt_step_{step}';path.mkdir();(path/'test.txt').write_text(str(step))
        inventory.append(dict(path=str(path),phase=phase,files={'test.txt':{'sha256':str(step)}},pruned=False))
    prune_superseded_intermediates(folder,inventory,root/'ckpt_step_4')
    assert not (root/'ckpt_step_2').exists()
    assert all((root/f'ckpt_step_{i}').exists() for i in (1,3,4))
    assert inventory[1]['pruned'] and inventory[1]['files']['test.txt']['sha256']=='2'
    inventory.append(dict(path=str(tmp_path/'parent_checkpoint'),phase='fitting',pruned=False))
    with pytest.raises(ValueError,match='outside'):prune_superseded_intermediates(folder,inventory,root/'ckpt_step_4')


def test_fit_resume_uses_full_execution_and_only_remaining_learning_and_retention_budget(monkeypatch,tmp_path):
    """The stillness runner keeps its own clock/arm while using shared restore."""
    from pathlib import Path
    import json
    from mimo_bridge.tools import stillness_learning_trial as trial
    args=options('scheduled');args.device='cpu'
    seed=913;original=MatchedOrders(args,seed)
    prior=[]
    for i in range(3):
        limits=original.limits({},i,.01,True)
        prior.append(dict(trial_step=i,noise=limits['exploration_noise'],bout=original.bout,scheduled_hold=not original.bout,
            sim_time=(i+1)*.01,function_min=1.,loss=.1))
    original.held=np.zeros((1,24,9),np.float32);original.held[...,6]=1;original.executed=original.held.copy();original.order=1
    expected_clock=MatchedOrders(args,0);expected_clock.load_state_dict(original.state_dict())
    expected=[(expected_clock.limits({},i,.01,True)['exploration_noise'],expected_clock.bout) for i in range(3,8)]
    episode=dict(worlds=[dict(env_kwargs={})],learner={'old_state':True},trainer=dict(step=103,trial_step=3,phase='fitting',
        orders=original.state_dict(),rows=prior,execution_source_sha256='same source',fit_execution={'complete':True},frame_meta=[{}]))
    used=[];saved=[];physical={'time':.03}
    achieved=torch.zeros((1,24,9));achieved[...,6]=1
    class Model:
        step_count=0;last_achieved=achieved
        def load_checkpoint(self,path):used.append(('load',path));return {'step':103}
        def step(self,frames,limits,**kw):
            assert kw['step_index']==self.step_count
            used.append(('update',self.step_count,limits['exploration_noise']));self.step_count+=1
            return SimpleNamespace(loss=.1,motor_for_action=original.held.copy(),gaze_out=np.zeros((1,2)))
    class Source:
        env=SimpleNamespace(dt=.01)
        def __init__(self,**kw):pass
        def step(self,*a,**kw):
            physical['time']+=.01
            return [SimpleNamespace(sim_time=physical['time'],meta={'gaze_cmd':[0.,0.]})]
        def close(self):used.append(('close',))
    def restore(model,source,state):
        assert state is episode and state['trainer']['fit_execution']['complete']
        model.step_count=state['trainer']['step'];used.append(('full_restore',103))
        return [SimpleNamespace(sim_time=.03,meta={'actual_saved_frame':True})]
    def save(model,source,orders,out,step,index,phase,args,rows):
        saved.append((step,index,phase,deepcopy(rows)));return out/f'endpoint_{phase}'
    monkeypatch.setattr(trial,'settings',lambda *a:args)
    monkeypatch.setattr(trial,'latest_copy',lambda folder,fallback:Path('retained_partial'))
    monkeypatch.setattr(trial.continuation_state,'load',lambda _:episode)
    monkeypatch.setattr(trial,'execution_source_digest',lambda **kw:'same source')
    monkeypatch.setattr(trial,'build_learner',lambda *a:Model())
    monkeypatch.setattr(trial,'verify_loaded',lambda *a:None)
    monkeypatch.setattr(trial,'SingleEnvSource',Source)
    monkeypatch.setattr(trial.copied_execution,'restore_episode',restore)
    monkeypatch.setattr(trial,'targets',lambda _: (np.ones((1,1)),np.array([True])))
    monkeypatch.setattr(trial,'save_copy',save)
    out=tmp_path/'scheduled'
    result=trial.fit_arm(Path('initial'),out,{},seed,6,2,'scheduled',tmp_path)
    assert [row[1] for row in used if row[0]=='update']==list(range(103,108))
    assert saved[0][:3]==(106,6,'end') and saved[1][:3]==(108,8,'retention')
    assert [(row['noise'],row['bout']) for row in saved[-1][3][3:]]==expected
    assert saved[-1][3][:3]==prior and len(saved[-1][3])==8
    receipt=json.loads((out/'fit.json').read_text())
    assert receipt['online_updates']==8 and receipt['learning_steps']==6 and receipt['intervening_retention_learning_steps']==2
    assert set(result)=={'end','retention'} and used[-1]==('close',)


def test_shared_continuation_refuses_legacy_missing_execution_before_resetting_body():
    from mimo_bridge import copied_execution
    source=SimpleNamespace(restore_full_state=lambda _:pytest.fail('Legacy body must remain untouched'))
    with pytest.raises(ValueError,match='Legacy copied episode'):
        copied_execution.restore_episode(None,source,{'trainer':{},'worlds':[]})
