# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Additive loaded-code binding around an unchanged frozen acting snapshot.

This module is deliberately outside the inner v2 implementation manifest. The
outer state contract cannot be read as an ordinary legacy actor. Only an explicit
shadow migration may introduce it; existing behavioral licenses are not adopted.
"""
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import time

from . import execution_contract as execution
from .acting_snapshot import FrozenActingSnapshot

CONTRACT='decorated_frozen_acting_snapshot_v1'
BINDING='loaded_decorated_functions_v1'


def function_binding(function,ancestry=()):
    """Bind wrapper code AND the function objects it actually calls.

    Repeated aliases do not affect identity. Recursive closure graphs terminate
    at an explicit code identity; interpreter addresses are never serialized.
    """
    receiver=None
    if inspect.ismethod(function):
        receiver=type(function.__self__).__module__+'.'+type(function.__self__).__qualname__
        function=function.__func__
    identity=id(function)
    base=execution._function_record(function)
    if receiver is not None:base['bound_receiver_type']=receiver
    if identity in ancestry:return dict(recursive_code=base['code_sha256'])
    ancestry=(*ancestry,identity)
    wrapped=getattr(function,'__wrapped__',None)
    if inspect.isfunction(wrapped) or inspect.ismethod(wrapped):
        base['wrapped_function']=function_binding(wrapped,ancestry)
    if function.__closure__:
        base['called_closures']={}
        for name,cell in zip(function.__code__.co_freevars,function.__closure__):
            try:value=cell.cell_contents
            except ValueError:
                base['called_closures'][name]={'empty_cell':True};continue
            if inspect.isfunction(value) or inspect.ismethod(value):
                base['called_closures'][name]=function_binding(value,ancestry)
    return base


def loaded_binding(model):
    inner=execution.loaded_execution(model)
    names=set(inner['descriptor']['modules'])|{__name__}
    modules={}
    for name in sorted(names):
        namespace=importlib.import_module(name)
        modules[name]={key:function_binding(fn) for key,fn in execution._functions(namespace,name)}
    bound={}
    for name in ('step','planned_execution','gaze_to_handle','_gaze_from_handle'):
        value=getattr(model,name,None)
        if inspect.isfunction(value) or inspect.ismethod(value):bound[name]=function_binding(value)
    value=dict(contract=BINDING,modules=modules,bound_model=bound,libraries=inner['libraries'])
    return dict(value=value,sha256=hashlib.sha256(execution._bytes(value)).hexdigest())


def record_manifest(model):
    """Record current source bytes for later migration; not a birth-code claim."""
    from .continuation_state import file_sha256
    binding=loaded_binding(model)
    sources={}
    for name in binding['value']['modules']:
        path=Path(inspect.getsourcefile(importlib.import_module(name))).resolve()
        sources[name]=dict(path=str(path),sha256=file_sha256(path))
    return dict(contract=CONTRACT,binding=binding,sources=sources,recorded_at=time.time(),
        scope='Prospective loaded-code binding; legacy unbound history remains explicitly unqualified')


def verify_manifest(model,manifest,*,require_source_files=False):
    if manifest.get('contract')!=CONTRACT:raise ValueError('unknown decorated execution contract')
    if loaded_binding(model)!=manifest.get('binding'):raise ValueError('decorated acting execution changed')
    if set(manifest.get('sources',{}))!=set(manifest['binding']['value']['modules']):
        raise ValueError('decorated source manifest is incomplete')
    if not isinstance(manifest.get('recorded_at'),(int,float)):
        raise ValueError('decorated manifest needs an explicit recording time')
    if require_source_files:
        from .continuation_state import file_sha256
        for name,item in manifest['sources'].items():
            path=Path(inspect.getsourcefile(importlib.import_module(name))).resolve()
            if path!=Path(item['path']).resolve() or file_sha256(path)!=item['sha256']:
                raise ValueError('recorded source changed before explicit migration: '+name)


def require_qualified_opportunity(actor):
    # The existing observer combines a frozen encoder with the continuing
    # learner's predictor and samples the first frame of a held order. It is a
    # compatibility diagnostic, not calibrated complete-learner progress. No
    # aligned replacement has been implemented/validated; a receipt string may
    # not manufacture one. Preserve its shadow evidence, never license it.
    if actor.opportunity is not None:
        raise ValueError('current hybrid opportunity audit cannot qualify action selection')


class DecoratedActingSnapshot:
    def __init__(self,actor,manifest):
        verify_manifest(actor.model,manifest)
        if actor.pending is not None:raise ValueError('nested replacement needs its own explicit binding')
        if not actor.qualification:require_qualified_opportunity(actor)
        if not actor.qualification and actor.receipt.get('decorated_execution_sha256')!=manifest['binding']['sha256']:
            raise ValueError('legacy qualified licenses cannot acquire a decorated binding')
        self.actor=actor;self.manifest=copy.deepcopy(manifest)

    @property
    def model(self):return self.actor.model
    @property
    def receipt(self):return self.actor.receipt
    @property
    def configuration(self):return self.actor.configuration
    @property
    def qualification(self):return self.actor.qualification
    @property
    def born_step(self):return self.actor.born_step
    @property
    def steps(self):return self.actor.steps
    @property
    def last(self):return self.actor.last
    @property
    def opportunity(self):return self.actor.opportunity
    @property
    def pending(self):return self.actor.pending

    def verify_weights(self):
        verify_manifest(self.model,self.manifest);self.actor.verify_weights()

    def step(self,*args,**kwargs):
        self.verify_weights();result=self.actor.step(*args,**kwargs);self.verify_weights();return result

    def prepare_learning_audit(self,*args,**kwargs):
        self.verify_weights();self.actor.prepare_learning_audit(*args,**kwargs);self.verify_weights()

    def reset_transition_state(self):
        self.verify_weights();self.actor.reset_transition_state();self.verify_weights()

    def stage(self,*args,**kwargs):
        raise ValueError('replacement requires an explicit decorated candidate binding')

    def promote(self,receipt):
        self.verify_weights()
        require_qualified_opportunity(self.actor)
        if receipt.get('decorated_execution_sha256')!=self.manifest['binding']['sha256']:
            raise ValueError('new physical qualification must include the decorated execution binding')
        self.actor.promote(receipt);return self

    def state_dict(self):
        self.verify_weights()
        return dict(contract=CONTRACT,manifest=copy.deepcopy(self.manifest),actor=self.actor.state_dict())

    @classmethod
    def restore(cls,state,factory,device,*,allow_qualification=False):
        if state.get('contract')!=CONTRACT:raise ValueError('legacy state requires explicit shadow migration')
        actor=FrozenActingSnapshot.restore(state['actor'],factory,device,allow_qualification=allow_qualification)
        return cls(actor,state['manifest'])


def wrap_shadow_state(state,model,manifest):
    """A structural addition only: never modifies historical actor/audit values."""
    from .acting_snapshot import CONTRACT as INNER,model_execution_fingerprint
    from .self_intention import fingerprint,perception_fingerprint
    if state.get('contract')!=INNER or state.get('qualification') is not True or state.get('pending') is not None:
        raise ValueError('only an unqualified actor without a staged replacement can migrate')
    if model_execution_fingerprint(model)!=state['receipt']['model_execution_sha256']:
        raise ValueError('legacy execution configuration no longer matches the saved shadow')
    if (fingerprint(model.self_state)!=state['receipt']['self_weights_sha256']
            or perception_fingerprint(model)!=state['receipt']['perception_weights_sha256']):
        raise ValueError('manifest model differs from the saved shadow weights')
    verify_manifest(model,manifest,require_source_files=True)
    return dict(contract=CONTRACT,manifest=copy.deepcopy(manifest),actor=state)
