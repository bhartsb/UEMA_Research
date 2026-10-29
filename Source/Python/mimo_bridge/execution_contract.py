# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Identify the Python implementations and libraries actually loaded for acting."""
from functools import lru_cache
import hashlib
import importlib
import inspect
import json
from pathlib import PurePath
from types import CodeType


def _bytes(value):
    return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _value(value):
    """Canonical values, independent of interning, object aliases and hash seed.

    Unsupported objects are never represented by an address-bearing repr. They
    are reported as outside this primitive-constant contract by the caller.
    """
    if value is None:return ['none']
    if value is Ellipsis:return ['ellipsis']
    if value is NotImplemented:return ['notimplemented']
    if isinstance(value,bool):return ['bool',value]
    if isinstance(value,int):return ['int',str(value)]
    if isinstance(value,float):return ['float',value.hex()]
    if isinstance(value,complex):return ['complex',value.real.hex(),value.imag.hex()]
    if isinstance(value,str):return ['str',value]
    if isinstance(value,bytes):return ['bytes',value.hex()]
    if isinstance(value,CodeType):return ['code',_code_record(value)]
    if isinstance(value,PurePath):return ['path',str(value)]
    if isinstance(value,(tuple,list)):
        return [type(value).__name__,[_value(v) for v in value]]
    if isinstance(value,(set,frozenset)):
        return [type(value).__name__,sorted((_value(v) for v in value),key=_bytes)]
    if isinstance(value,dict):
        return ['dict',sorted(([_value(k),_value(v)] for k,v in value.items()),key=_bytes)]
    raise TypeError(type(value).__module__+'.'+type(value).__qualname__)


@lru_cache(maxsize=16384)
def _code_record(code):
    # Marshal contains mutable interpreter interning/reference flags. These
    # public code fields describe execution without that process-local state.
    fields=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals',
            'co_stacksize','co_flags','co_code','co_consts','co_names','co_varnames',
            'co_freevars','co_cellvars','co_name','co_qualname','co_exceptiontable')
    return {name:_value(getattr(code,name)) for name in fields}


def _supported(value):
    try:return dict(value=_value(value))
    except (TypeError,RecursionError) as exc:return dict(unbound_type=str(exc))


def _function_record(function):
    record=dict(code_sha256=_semantic_code_hash(function.__code__),defaults=_supported(function.__defaults__),
                kwdefaults=_supported(function.__kwdefaults__))
    if function.__closure__:
        record['closure']=[_supported(cell.cell_contents) for cell in function.__closure__]
    return record


@lru_cache(maxsize=16384)
def _semantic_code_hash(code):
    return hashlib.sha256(_bytes(_code_record(code))).hexdigest()


def _functions(namespace,prefix):
    for name,value in sorted(vars(namespace).items()):
        if getattr(value,'__module__',None)!=namespace.__name__:continue
        if inspect.isfunction(value):yield prefix+'.'+name,value
        elif callable(value) and hasattr(value,'__wrapped__'):
            original=inspect.unwrap(value)
            if inspect.isfunction(original):yield prefix+'.'+name+'.__wrapped__',original
        elif inspect.isclass(value):
            for key,method in sorted(vars(value).items()):
                if isinstance(method,(staticmethod,classmethod)):method=method.__func__
                if isinstance(method,property):method=method.fget
                if inspect.isfunction(method):yield prefix+'.'+name+'.'+key,method


def _code_digest(entries):
    return hashlib.sha256(_bytes({name:_code_record(code) for name,code in entries})).hexdigest()


def _module_record(namespace):
    functions={name:hashlib.sha256(_bytes(_function_record(fn))).hexdigest()
               for name,fn in _functions(namespace,namespace.__name__)}
    # Bind loaded primitive tables too: motor chains, indices, gains, schema
    # strings and dataclass constructor defaults cannot inherit a prior pass.
    # Mutable runtime caches and module instances are deliberately not serialized.
    constants={name:_supported(value) for name,value in sorted(vars(namespace).items())
               if name.isupper() and not name.startswith('_')}
    return dict(functions=functions,constants=constants)


def loaded_execution(model):
    """Changing files alone does not change running code; reloading/replacing does.

    Binds loaded semantic code, defaults, primitive uppercase module tables and
    directly replaced execution methods. The per-module descriptor identifies
    differences on restore. Nonprimitive objects are explicitly unbound: model
    parameters/configuration have separate contracts; arbitrary monkeypatches to
    third-party internals require new external qualification.
    """
    import numpy
    import torch
    import mujoco
    from .self_intention import perception_modules
    modules={'mimo_bridge.lem_model','mimo_bridge.action_decoder','mimo_bridge.env',
        'mimo_bridge.datasource','mimo_bridge.self_intention','mimo_bridge.acting_snapshot',
        'mimo_bridge.opportunity_integration','mimo_bridge.self_state_integration',
        'mimo_bridge.vision_adapter','mimo_bridge.skin_contract','mimo_bridge.body_layout',
        'lem.model.homeostatic_planner','lem.model.learning_opportunity','lem.body_function',
        'mimo_bridge.execution_contract'}
    modules.update(type(child).__module__ for module in perception_modules(model).values() for child in module.modules()
                   if type(child).__module__.startswith(('lem.','mimo_bridge.')))
    records={name:_module_record(importlib.import_module(name)) for name in sorted(modules)}
    bound={}
    for name in ('step','planned_execution','gaze_to_handle','_gaze_from_handle'):
        method=getattr(model,name,None)
        if inspect.ismethod(method):method=method.__func__
        if inspect.isfunction(method):bound[name]=hashlib.sha256(_bytes(_function_record(method))).hexdigest()
    import sys
    descriptor=dict(modules=records,bound_model=bound)
    return dict(contract='loaded_python_execution_v2',code_sha256=hashlib.sha256(_bytes(descriptor)).hexdigest(),
        descriptor=descriptor,
        libraries=dict(torch=str(torch.__version__),numpy=str(numpy.__version__),
                       mujoco=str(mujoco.__version__),cuda=str(torch.version.cuda),python=sys.version))
