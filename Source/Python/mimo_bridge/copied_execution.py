# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Complete execution snapshots for frozen copied-model comparisons.

Weights/optimizer are not cloned: callers must freeze learning. Besides the
episode companion, this retains public counters, module buffers and the runtime
state intentionally stored separately by the production checkpoint writer.
Unknown mutable top-level helpers fail closed instead of silently leaking state.
"""
from collections import deque
from dataclasses import is_dataclass
import enum
import hashlib
import copy

import numpy as np
import torch

from . import continuation_state as continuation
from .acting_snapshot import runtime, restore_runtime


CONTRACT='copied_execution_v1'
_OWNED_HELPERS={'placeholders','loss_manager','memory_history','self_history','self_rehearsal',
                'latent_regularizer','movement','curiosity','replay','consequence_lesson'}
# PLAIN TRACKER OBJECTS the gain-of-function flags put on the model under PUBLIC names (their
# learned part rides runtime_state.pt, so the episode companion never copied them), copied whole
# with copy.deepcopy. Added 2026-09-23 (review of the probes, blocker: every probe of a model built
# with --contact-head on or --wellness-rank normalized stopped at its first snapshot with
# "Unserialized copied model state: contact_tracker"). Each class was read before it was listed:
# its state is numbers, numpy arrays, CPU torch tensors, dicts, lists and deques, and the helper
# objects named here, all of which copy.deepcopy reproduces bit for bit (a deque keeps its maxlen).
# _deepcopy_exact walks every captured object and refuses any value outside those kinds by its
# path, so a later edit that puts a module, a parameter, a generator or a file inside a tracker
# fails here by name instead of being copied silently. Anything else still fails closed.
# Added 2026-09-23 (verify-fix-probes): lem.model.next_frame_norm.NextFrameNorm, the running
# persistence error per sense that --next-frame-norm persistence (DESIGN D12, b6) puts on the model
# as LemLite.nf_norm. Measured: a build of H84's arguments with every DESIGN section 2 flag on
# refused its first snapshot with "Unserialized copied model state: nf_norm (NextFrameNorm)" before
# this line. Its class was read: an int horizon, float alpha and cap, a dict of [float, float, int]
# per sense and a window dict of dicts of floats and ints - plain values copy.deepcopy reproduces.
_PLAIN_OBJECTS={
    'lem.model.contact_head.ContactTracker','lem.model.contact_head._Hist',
    'lem.model.wellness_channels.WellnessChannelState','lem.model.wellness_channels.ChannelNormalizer',
    'lem.model.wellness_channels.ControllabilityTracker',
    'lem.model.wellness_channels.LearningProgressTracker','lem.model.wellness_channels.RankShares',
    'lem.model.next_frame_norm.NextFrameNorm',
    'mimo_bridge.vestibular_sense.VestibularNormaliser',
    'mimo_bridge.proprio_token.WithinBody'}
# FIELDS A PLAIN TRACKER COPIES DIFFERENTLY (2026-09-29, H84 segment 41: every scheduled ability probe stopped at
# its first snapshot with "Unserialized copied model state: proprio_within (WithinBody)" once --proprio-token-repair
# within_body was on). WithinBody's class was read: numbers, lists, dicts and CPU tensors, plus _read_graph, this
# step's reading still attached to its autograd graph (set by the gauge and consumed by the lesson inside one step;
# a graph tensor cannot be deep-copied and must never outlive its step), and rows_transform, a configuration
# function (--proprio-heading free) kept by reference.
_TRANSIENT_FIELDS={'mimo_bridge.proprio_token.WithinBody':('_read_graph',)}
_REFERENCE_FIELDS={'mimo_bridge.proprio_token.WithinBody':('rows_transform',)}
_EXACT_LEAVES=(type(None),bool,int,float,complex,str,bytes,np.generic,np.ndarray,torch.device,torch.dtype)


def _qualname(value):
    return type(value).__module__+'.'+type(value).__qualname__


def _deepcopy_exact(value,path,depth=0):
    """Refuse (ValueError naming the path) any value inside a plain tracker that copy.deepcopy
    would not reproduce exactly, or that must never be copied (a module, a parameter)."""
    if depth>32:raise ValueError(f'Unserialized copied model state: {path} nests deeper than 32 levels')
    if isinstance(value,_EXACT_LEAVES):return
    if isinstance(value,torch.nn.Parameter) or isinstance(value,torch.nn.Module):
        raise ValueError(f'Unserialized copied model state: {path} holds a learned {type(value).__name__}')
    if torch.is_tensor(value):return
    if isinstance(value,enum.Enum):return
    if isinstance(value,dict):
        for k,v in value.items():_deepcopy_exact(v,f'{path}[{k!r}]',depth+1)
        return
    if isinstance(value,(list,tuple,set,frozenset,deque)):
        for i,v in enumerate(value):_deepcopy_exact(v,f'{path}[{i}]',depth+1)
        return
    if _qualname(value) in _PLAIN_OBJECTS:
        skip=set(_TRANSIENT_FIELDS.get(_qualname(value),()))|set(_REFERENCE_FIELDS.get(_qualname(value),()))
        for k,v in vars(value).items():
            if k not in skip:_deepcopy_exact(v,f'{path}.{k}',depth+1)
        return
    raise ValueError(f'Unserialized copied model state: {path} ({type(value).__name__})')


def _generator_state(value):
    """A torch.Generator is saved as its state (get_state) and device, never as the object."""
    return dict(device=str(value.device),state=value.get_state().clone())


def _restore_generator(current,saved):
    if isinstance(current,torch.Generator) and str(current.device)==saved['device']:
        current.set_state(saved['state'].clone());return current
    g=torch.Generator(device=saved['device']);g.set_state(saved['state'].clone());return g


_STATIC={'params','optimizer','checkpoints','device','dtype','noise_rng','wellness_mode','hand_kinematics'}
_MODULE_INTERNAL={'_parameters','_buffers','_modules','_backward_pre_hooks','_backward_hooks',
    '_forward_hooks','_forward_hooks_with_kwargs','_forward_hooks_always_called','_forward_pre_hooks',
    '_forward_pre_hooks_with_kwargs','_state_dict_hooks','_state_dict_pre_hooks',
    '_load_state_dict_pre_hooks','_load_state_dict_post_hooks'}


def clone(value,device='cpu'):
    if isinstance(value,(set,frozenset,slice)):return copy.deepcopy(value)
    return continuation.tree(value,device)


class _SavedGenerator(dict):
    """A torch.Generator kept as its device and state (get_state), never as the object."""


class _SavedObject:
    """A plain tracker (_PLAIN_OBJECTS) copied whole; every restore hands out a fresh deep copy,
    so a restored branch never owns the snapshot's own object."""
    __slots__=('value',)

    def __init__(self,value):self.value=value


def _capture_special(value,path):
    """(True, saved) for a generator or a listed plain tracker, else (False, None)."""
    if isinstance(value,torch.Generator):return True,_SavedGenerator(_generator_state(value))
    if value is not None and _qualname(value) in _PLAIN_OBJECTS:
        _deepcopy_exact(value,path)
        return True,_SavedObject(_copy_plain(value))
    return False,None


def _copy_plain(value):
    """A listed tracker copied whole: its transient fields empty, its reference fields shared, the rest deep-copied."""
    name=_qualname(value)
    transient,reference=_TRANSIENT_FIELDS.get(name,()),_REFERENCE_FIELDS.get(name,())
    if not transient and not reference:return copy.deepcopy(value)
    shallow=copy.copy(value)
    for k in transient:setattr(shallow,k,None)
    kept={k:getattr(value,k) for k in reference if hasattr(value,k)}
    for k in kept:setattr(shallow,k,None)
    out=copy.deepcopy(shallow)
    for k,v in kept.items():setattr(out,k,v)
    return out


def _restore_special(current,saved):
    if isinstance(saved,_SavedGenerator):return _restore_generator(current,saved)
    return _copy_plain(saved.value)


def plain_state(value):
    """A listed tracker as nested dicts for strict state comparisons.

    Tracker NaNs mean 'unmeasured'. Keep their locations explicitly so restoring
    this metadata can be tested without allowing NaNs in scientific outputs.
    """
    if value is not None and _qualname(value) in _PLAIN_OBJECTS:
        return {'__class__':_qualname(value),**{k:plain_state(v) for k,v in vars(value).items()}}
    if isinstance(value,dict):return {k:plain_state(v) for k,v in value.items()}
    if isinstance(value,deque):return ('deque',value.maxlen,[plain_state(v) for v in value])
    if isinstance(value,(list,tuple)):return type(value)(plain_state(v) for v in value)
    if torch.is_tensor(value):
        value=value.detach().cpu().clone()
        missing=torch.isnan(value)
        if bool(missing.any()):
            return {'__tracker_unmeasured__':missing,'values':value.masked_fill(missing,0)}
        return value
    if isinstance(value,np.ndarray):
        value=value.copy()
        if np.issubdtype(value.dtype,np.inexact):
            missing=np.isnan(value)
            if missing.any():
                value[missing]=0
                return {'__tracker_unmeasured__':missing,'values':value}
        return value
    if isinstance(value,float) and value!=value:return {'__tracker_unmeasured__':True}
    return copy.deepcopy(value)


def trackers(model, *, include_timing=True):
    """Every listed plain tracker the model holds under a public name, as plain_state."""
    out = {name:plain_state(value) for name,value in vars(model).items()
           if value is not None and _qualname(value) in _PLAIN_OBJECTS}
    if not include_timing and 'wellness_channels' in out:
        # Wall time changes between identical replays; it never selects an order.
        cost=out['wellness_channels'].get('cost', {})
        for key in ('step_ms', 'plan_ms'):cost.pop(key, None)
    return out


def plain_attributes(obj):
    result={}
    for name,value in vars(obj).items():
        # RNN flat-weight lists/weakrefs are aliases of frozen registered
        # parameters, not execution state; copying them detaches learned weights.
        if name in _MODULE_INTERNAL or name.startswith('_flat_weight') or callable(value) or isinstance(value,torch.nn.Module):continue
        if isinstance(value,list) and any(isinstance(v,torch.nn.Module) for v in value):continue
        special,saved=_capture_special(value,f'{type(obj).__name__}.{name}')
        if special:result[name]=saved;continue
        try:result[name]=clone(value)
        except TypeError:
            if isinstance(value,(enum.Enum,torch.dtype)):continue
            raise ValueError(f'Unserialized copied helper state: {type(obj).__name__}.{name}')
    return result


def _restore_plain(owner,key,value,device):
    if isinstance(value,(_SavedGenerator,_SavedObject)):
        setattr(owner,key,_restore_special(getattr(owner,key,None),value))
    else:
        setattr(owner,key,clone(value,device))


def model_modules(model):
    # Inspect every actual root module, including a later-added side component,
    # instead of maintaining another partial named list of learned dependencies.
    return {name:value for name,value in vars(model).items() if isinstance(value,torch.nn.Module)}


def capture(model):
    hand = getattr(model, 'hand_kinematics', None)
    geometry = getattr(model, '_hand_geometry', None)
    if hand is not None:
        from .hand_target_control import HandKinematics
        if (type(hand) is not HandKinematics or hand.geometry != geometry
                or set(vars(hand)) != {'geometry','groups','_projection_model','_projection_data'}
                or hand.groups != HandKinematics(geometry).groups):
            raise ValueError('Unserialized copied model state: hand_kinematics differs from its pure geometry')
    state=continuation.capture_learner(model,include_replay=True)
    public={}; helpers={}; special={}
    for name,value in vars(model).items():
        if name in state['data'] or name in _STATIC or name in _OWNED_HELPERS or name in ('_schemas','_hself'):
            continue
        if isinstance(value,torch.nn.Module):continue
        if name=='self_intention':
            helpers[name]=None if value is None else plain_attributes(value)
            continue
        found,saved=_capture_special(value,name)
        if found:special[name]=saved;continue
        try:public[name]=continuation.tree(value)
        except TypeError:raise ValueError(f'Unserialized copied model state: {name} ({type(value).__name__})')
    modules={}
    for root,module in model_modules(model).items():
        modules[root]={name:dict(plain=plain_attributes(child),
            buffers={k:continuation.tree(v) for k,v in child._buffers.items()})
            for name,child in module.named_modules()}
    # 'special' (plain trackers and generators) is an addition to the v1 contract: a snapshot
    # written before it has none, and restore reads it with a default
    return dict(contract=CONTRACT,continuation=state,runtime=runtime(model),public=public,
                helpers=helpers,modules=modules,special=special)


def restore(model,state):
    if state.get('contract')!=CONTRACT:raise ValueError('Unknown copied execution contract')
    continuation.restore_learner(model,state['continuation'])
    # CPU .to('cpu') is an alias, and latent running statistics update in place.
    # Never let a restored branch own tensors from its immutable saved snapshot.
    restore_runtime(model,continuation.tree(state['runtime']))
    model._schemas=model._schemas[:len(state['runtime']['schemas'])]
    for name,value in state['public'].items():setattr(model,name,continuation.tree(value,model.device))
    # The compiled private IK model is a derived cache, never learned state.
    # Rebuild it lazily from captured geometry instead of copying native pointers.
    if hasattr(model, 'hand_kinematics'):
        from .hand_target_control import HandKinematics
        geometry = state['continuation']['data'].get('_hand_geometry')
        model._hand_geometry = continuation.tree(geometry)
        model.hand_kinematics = None if geometry is None else HandKinematics(geometry)
    for name,value in (state.get('special') or {}).items():
        setattr(model,name,_restore_special(getattr(model,name,None),value))
    for name,values in state['helpers'].items():
        helper=getattr(model,name,None)
        if values is None:
            if helper is not None:raise ValueError('A new intention helper appeared inside a frozen branch')
        elif helper is None:raise ValueError('A saved intention helper disappeared')
        else:
            for key,value in values.items():_restore_plain(helper,key,value,model.device)
    roots=model_modules(model)
    for root,children in state['modules'].items():
        now=dict(roots[root].named_modules())
        if set(now)!=set(children):raise ValueError('Frozen branch changed module structure')
        for name,values in children.items():
            child=now[name]
            for key,value in values['plain'].items():_restore_plain(child,key,value,model.device)
            for key,value in values['buffers'].items():child._buffers[key]=continuation.tree(value,model.device)


def save_episode(staging,final,source,model,trainer,config):
    """One atomic learning boundary, paired with separately saved weights/Adam.

    Capture replay only once. Besides full model execution, keep the actual
    observed frame metadata rather than recollecting or resetting the world.
    """
    execution=capture(model);learner=execution.pop('continuation')
    worlds=source.save_full_state()
    # capture() already copied module metadata (including buffer-name sets).
    # Do not send it through the narrower generic trainer-value serializer or
    # duplicate the large captured tensors a second time.
    trainer=dict(continuation.tree(trainer),fit_execution=execution)
    if 'frame_meta' not in trainer:
        trainer['frame_meta']=[copy.deepcopy(world['payload']['info']) for world in worlds]
    state=dict(version=continuation.VERSION,worlds=worlds,learner=learner,
        trainer=trainer,config=copy.deepcopy(config))
    return continuation.write_episode(staging,state,published_path=final)


def restore_episode(model,source,state):
    """Restore physical observation first and the exact next learner RNG last."""
    trainer=state['trainer']
    if 'fit_execution' not in trainer or 'frame_meta' not in trainer:
        raise ValueError('Legacy copied episode lacks complete execution/frame continuation; preserve it and start a separately declared attempt')
    frames=source.restore_full_state(state['worlds'])
    if len(frames)!=len(trainer['frame_meta']):raise ValueError('Copied observation count changed')
    for frame,meta in zip(frames,trainer['frame_meta']):frame.meta=continuation.tree(meta)
    restore(model,dict(trainer['fit_execution'],continuation=state['learner']))
    if model.step_count!=trainer['step']:raise ValueError('Copied model and world steps disagree')
    return frames


def equal_tree(a,b):
    """Bit-identical scientific observations; NaNs never establish repeatability."""
    if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a,b))
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    if is_dataclass(a):return type(a)==type(b) and equal_tree(vars(a),vars(b))
    return type(a)==type(b) and a==b


def first_difference(a,b,path='execution'):
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        for key in a:
            difference=first_difference(a[key],b[key],path+'.'+str(key))
            if difference:return difference
        return None
    if isinstance(a,(list,tuple)) and type(a)==type(b) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)):
            difference=first_difference(x,y,path+f'[{i}]')
            if difference:return difference
        return None
    return None if equal_tree(a,b) else path


def output_record(model,result):
    """First/later commands and native forecasts plus behaviorally relevant state."""
    return dict(motor=np.array(result.motor_for_action,copy=True),gaze=np.array(result.gaze_out,copy=True),
        step_count=int(model.step_count),predictions=continuation.tree(model.last_predictions),
        native_next=continuation.tree(model._nf_pred),raw_tokens=continuation.tree(model.last_tokens),
        schema_and_latent=runtime(model),intention=continuation.tree(model.last_intention),
        context=continuation.tree(model.context_buffer),smoother=continuation.tree(model._prev_mfa),
        memory=None if model.memory_history is None else continuation.tree(model.memory_history.state_dict()),
        self_history=None if model.self_history is None else continuation.tree(model.self_history.state_dict()),
        noise_rng=model.noise_rng.get_state().clone(),
        # the gain-of-function trackers (contact, controllability, learning progress, the
        # ranking's normaliser), compared whole; empty on a build without them
        trackers=trackers(model, include_timing=False))


def verify_recorded_frames(model,saved,samples,expected=None):
    """Replay actual recorded inputs twice; compare outputs and meaningful state.

    This exercises inference (including the planner), not just a serialization
    round trip. The caller separately verifies repeated physical execution.
    """
    if len(samples)<2:raise ValueError('Complete model replay needs first and later observations')
    passes=[]
    try:
        for _ in range(2):
            restore(model,saved);records=[]
            for sample in samples:
                result=model.step(sample['frames'],sample['dev'],train=False,
                    step_index=model.step_count,executed_action=sample['executed_action'],
                    executed_order_id=sample['executed_order_id'])
                records.append(output_record(model,result))
            passes.append(records)
        if not equal_tree(passes[0],passes[1]):
            raise ValueError('Identical recorded inputs did not reproduce '+str(first_difference(passes[0],passes[1])))
        if expected is not None and not equal_tree(passes[0],expected):
            raise ValueError('Restored inference does not reproduce original '+str(first_difference(passes[0],expected)))
        return dict(contract=CONTRACT,complete_model_replay_exact=True,frames=len(samples),
            checks=['native forecasts','first and later motor/gaze outputs','step counter','noise RNG',
                    'schema/hierarchical self','latent reference','memory and self history','intention','smoother',
                    'gain-of-function trackers'])
    finally:restore(model,saved)
