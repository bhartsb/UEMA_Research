# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Scientific budget and speech bookkeeping survive an interrupted copied fit."""
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge import continuation_state, copied_execution
from mimo_bridge.tools import caregiver_fit_continuation as continuation
from mimo_bridge.tools import caregiver_learning_trial as trial
from mimo_bridge.tools.caregiver_fit_workers import CopyResourcesUnavailable


def test_uninterrupted_and_paused_fit_have_identical_learning_world_orders_and_speech(monkeypatch,tmp_path):
    """A small actual Adam learner and stochastic world exercise the fit loop.

    The native LemLite/MuJoCo proof is separate; this fixture isolates forgotten
    counters, duplicate words, partial budgets and RNG consumption at restart.
    """
    parent=tmp_path/'parent';parent.mkdir();(parent/'model.safetensors').write_bytes(b'fixed fixture parent')
    calls=[]
    class Model:
        def __init__(self,out):
            torch.manual_seed(137);np.random.seed(138);random.seed(139)
            self.encoder_registry=torch.nn.Linear(1,1)
            self.optimizer=torch.optim.Adam(self.encoder_registry.parameters(),lr=.01)
            self.step_count=123;self.out=out;self.memory=[];self.previous=0.
            self.noise=torch.Generator().manual_seed(140)
        def step(self,frames,limits,train,step_index,executed_action,executed_order_id):
            assert step_index==self.step_count
            draw=torch.rand((),generator=self.noise)+torch.rand(())+np.random.rand()+random.random()
            target=frames[0].position+draw+self.previous+float(executed_action or 0.)*.001
            prediction=self.encoder_registry(torch.ones(1,1));loss=(prediction-target).square().mean()
            self.optimizer.zero_grad();loss.backward();self.optimizer.step()
            self.step_count+=1;self.previous=float(prediction.detach());self.memory.append((step_index,float(draw),deepcopy(frames[0].meta)))
            calls.append((str(self.out),step_index))
            return SimpleNamespace(loss=float(loss.detach()),command=float(prediction.detach()))
        def save_checkpoint(self,step,companion_writer):
            checkpoint=self.out/'checkpoints'/f'ckpt_step_{step}';checkpoint.mkdir(parents=True,exist_ok=False)
            torch.save(dict(weights=self.encoder_registry.state_dict(),optimizer=self.optimizer.state_dict()),checkpoint/'weights.pt')
            companion_writer(checkpoint,checkpoint);return str(checkpoint)
    class World:
        def __init__(self):
            self.dt=.01;self.position=0.;self.count=0;self.resets=0;self.rng=np.random.default_rng(88)
            self.caregiver=SimpleNamespace(bank=SimpleNamespace(groups={'x':['one','two','reserved']}),state={'events':[]},
                export_events=lambda path:Path(path).write_text(json.dumps(self.caregiver.state['events'])))
    class Source:
        def __init__(self):self.env=World();self.last=None
        def frame(self):
            self.last=SimpleNamespace(position=self.env.position,sim_time=self.env.count*.05,meta={'world_count':self.env.count})
            return [self.last]
        def reset(self):self.env.resets+=1;return self.frame()
        def save_full_state(self):
            return [dict(payload={'sim_time':self.env.count*.05},state=deepcopy((self.env.position,self.env.count,
                self.env.resets,self.env.rng.bit_generator.state,self.env.caregiver.state)),last=deepcopy(self.last))]
        def restore_full_state(self,worlds):
            w=worlds[0];self.env.position,self.env.count,self.env.resets,rng,self.env.caregiver.state=deepcopy(w['state'])
            self.env.rng.bit_generator.state=rng;self.last=deepcopy(w['last']);return [self.last]
        def close(self):pass
    class Orders:
        def __init__(self,*a):
            self.held=None;self.executed=None;self.order=0;self.bout=True;self.until=0;self.rng=np.random.default_rng(44)
        def limits(self,dev,index,dt,train):
            if index>=self.until:self.bout=not self.bout;self.until=index+int(self.rng.integers(2,4))
            return dev
        def advance(self,model,result,source,index,train):
            if self.held is None or index%3==0:self.held=result.command;self.order+=1
            source.env.count+=1;source.env.position+=self.held+source.env.rng.normal()
            self.executed=self.held
            if source.env.count in (1,4):
                word={'time':source.env.count*.05,'key':f'word_{source.env.count}'}
                source.env.caregiver.state['last_utterance']=word
                source.env.caregiver.state['events'].append(word)
            return source.frame()
    def learner(args,checkpoint,out,train):
        out.mkdir(exist_ok=True);model=Model(out)
        if checkpoint!=parent:
            saved=torch.load(checkpoint/'weights.pt',weights_only=False)
            model.encoder_registry.load_state_dict(saved['weights']);model.optimizer.load_state_dict(saved['optimizer'])
            model.step_count=json.loads((checkpoint/'fit_continuation.json').read_text())['step']
        return model,model.step_count
    def capture(model):
        return dict(contract='fixture',continuation=dict(step_count=model.step_count,memory=deepcopy(model.memory),
            previous=model.previous,noise=model.noise.get_state(),torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()))
    def restore(model,state):
        s=state['continuation'];model.step_count=s['step_count'];model.memory=deepcopy(s['memory']);model.previous=s['previous']
        model.noise.set_state(s['noise']);torch.set_rng_state(s['torch']);np.random.set_state(s['numpy']);random.setstate(s['python'])
    monkeypatch.setattr(trial,'learner',learner);monkeypatch.setattr(trial,'source',lambda *a,**k:Source())
    monkeypatch.setattr(trial,'Orders',Orders);monkeypatch.setattr(copied_execution,'capture',capture);monkeypatch.setattr(copied_execution,'restore',restore)
    monkeypatch.setattr(__import__('shutil'),'disk_usage',lambda _:SimpleNamespace(free=20*1024**3))
    args=SimpleNamespace(device='cpu',action_repeat=5);limits=dict(phase_name='fixture')
    left=tmp_path/'uninterrupted';right=tmp_path/'interrupted'
    final_left=trial.fit(args,parent,left,11,6,limits)
    def pause(model,index):
        if index==2:raise CopyResourcesUnavailable('fixture production checkpoint peak')
    with pytest.raises(CopyResourcesUnavailable):trial.fit(args,parent,right,11,6,limits,step_guard=pause)
    partial=Path(json.loads((right/'paused_fit.json').read_text())['checkpoint'])
    original_bytes={p.name:p.read_bytes() for p in partial.iterdir()}
    with pytest.raises(CopyResourcesUnavailable):
        trial.fit(args,parent,right,11,6,limits,step_guard=pause,resume_checkpoint=partial)
    # A second pause before new updates must not overwrite the completed snapshot.
    assert all(p.read_bytes()==original_bytes[p.name] for p in partial.iterdir())
    final_right=trial.fit(args,parent,right,11,6,limits,resume_checkpoint=partial)
    a=torch.load(final_left/'weights.pt',weights_only=False);b=torch.load(final_right/'weights.pt',weights_only=False)
    assert copied_execution.equal_tree(a,b)
    sa=continuation_state.load(final_left);sb=continuation_state.load(final_right)
    assert copied_execution.equal_tree(sa['learner'],sb['learner'])
    assert copied_execution.equal_tree(sa['worlds'],sb['worlds'])
    for name in ('step','trial_step','held','executed','order','bout','until','bout_rng','frame_meta'):
        assert copied_execution.equal_tree(sa['trainer'][name],sb['trainer'][name]),name
    ra=json.loads((left/'fit.json').read_text());rb=json.loads((right/'fit.json').read_text())
    assert ra['source_utterances']==rb['source_utterances']==[[.05,'word_1'],[.2,'word_4']]
    assert rb['online_updates']==6 and rb['completed_before_segment']==2 and rb['encoder_weights_changed']
    assert [step for folder,step in calls if folder==str(right)]==list(range(123,129))
    assert sb['worlds'][0]['state'][2]==1  # no extra reset on either resume
    assert (parent/'model.safetensors').read_bytes()==b'fixed fixture parent'
    assert (right/'paused_fit_history').is_dir()


def test_discovery_rejects_changed_protocol_and_preserves_legacy_partials(tmp_path):
    base=tmp_path/'responsive_0';legacy=base/'checkpoints'/'old';legacy.mkdir(parents=True)
    (legacy/'episode_state.json').write_text('old incomplete bookkeeping')
    assert continuation.latest(base,lambda _:pytest.fail('Legacy partial must not be relabeled')) is None
    current=base/'attempt_002'/'checkpoints'/'new';current.mkdir(parents=True)
    expected=dict(steps=6)
    (current/'fit_continuation.json').write_text(json.dumps(dict(contract=continuation.CONTRACT,binding=expected,episode_sha256='abc',trial_step=2)))
    (current/'episode_state.json').write_text(json.dumps(dict(sha256='abc')))
    assert continuation.latest(base,lambda _:expected)==(base/'attempt_002',current)
    with pytest.raises(ValueError,match='different execution'):continuation.latest(base,lambda _:dict(steps=7))
    (current/'episode_state.json').write_text(json.dumps(dict(sha256='altered')))
    with pytest.raises(ValueError,match='binding changed'):continuation.latest(base,lambda _:expected)
    assert (legacy/'episode_state.json').read_text()=='old incomplete bookkeeping'


def test_shared_episode_writer_preserves_captured_module_buffer_sets_once(monkeypatch,tmp_path):
    captured=dict(contract=copied_execution.CONTRACT,continuation={'saved':True},
        modules={'module':{'':{'plain':{'_non_persistent_buffers_set':{'cache'},'selected':slice(1,8,2)},'buffers':{}}}})
    monkeypatch.setattr(copied_execution,'capture',lambda _:captured)
    source=SimpleNamespace(save_full_state=lambda:[dict(payload={'sim_time':.1,'info':{'actual_frame':True}})])
    copied_execution.save_episode(tmp_path,tmp_path,source,None,dict(step=2),{})
    state=continuation_state.load(tmp_path)
    assert state['trainer']['fit_execution']['modules']['module']['']['plain']['_non_persistent_buffers_set']=={'cache'}
    assert state['trainer']['fit_execution']['modules']['module']['']['plain']['selected']==slice(1,8,2)
    assert state['trainer']['frame_meta']==[{'actual_frame':True}]
    assert state['learner']=={'saved':True}


@pytest.mark.parametrize('value',[
    torch.tensor(2.),torch.tensor([2.]).as_strided((1,),(0,)),
    torch.tensor(2.,dtype=torch.bfloat16),torch.empty(0),torch.empty(0,dtype=torch.bfloat16)])
def test_native_digest_handles_scalar_zero_stride_bfloat16_and_empty_without_value_loss(value):
    from mimo_bridge.tools.prepare_local_shadow import state_digest
    contiguous=value.clone(memory_format=torch.contiguous_format)
    assert state_digest(value)==state_digest(contiguous)
    if value.numel():assert state_digest(value)!=state_digest(value+1)
    assert state_digest(value)!=state_digest(value.reshape(-1,1))


def test_native_digest_and_clone_preserve_module_sets_and_slices():
    from mimo_bridge.tools.prepare_local_shadow import state_digest
    value={'buffers':{'cache','history'},'indices':slice(1,8,2)}
    assert state_digest(value)==state_digest(deepcopy(value))
    assert state_digest(value)!=state_digest(dict(value,indices=slice(1,8,3)))
    assert copied_execution.clone(slice(1,8,2))==slice(1,8,2)
