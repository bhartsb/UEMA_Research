# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Independent regression cases for the additive decorated-code contract."""
import functools
from types import ModuleType
import torch

from mimo_bridge.decorated_acting_snapshot import function_binding
from mimo_bridge.execution_contract import _function_record


def test_changing_only_wrapped_method_body_invalidates_new_binding():
    module=ModuleType('independent_wrapper_review')
    module.__dict__['wraps']=functools.wraps
    exec('def decorate(fn):\n @wraps(fn)\n def wrapper(*a,**kw):\n  return fn(*a,**kw)\n return wrapper\n'
         'class Predictor:\n @decorate\n def consequence(self):\n  return 1\n',module.__dict__)
    method=module.Predictor.consequence
    old_descriptor=_function_record(method)
    extended_descriptor=function_binding(method)
    method.__wrapped__.__code__=method.__wrapped__.__code__.replace(co_consts=(None,2))
    assert module.Predictor().consequence()==2
    assert _function_record(method)==old_descriptor  # The original blind spot.
    assert function_binding(method)!=extended_descriptor


def test_actual_closure_function_is_bound_even_when_wrapped_metadata_points_elsewhere():
    def first():return 1
    def decoy():return 9
    def make(actual):
        @functools.wraps(decoy)
        def wrapper():return actual()
        return wrapper
    wrapped=make(first);before=function_binding(wrapped)
    first.__code__=first.__code__.replace(co_consts=(None,2))
    assert wrapped()==2
    assert wrapped.__wrapped__ is decoy
    assert function_binding(wrapped)!=before


def test_primitive_closure_state_changes_binding():
    values=[1]
    def calculate():return values[0]
    before=function_binding(calculate)
    values[0]=2
    assert function_binding(calculate)!=before


def test_recursive_function_closures_terminate_without_object_addresses():
    def make():
        def even(n):return n==0 or odd(n-1)
        def odd(n):return n!=0 and even(n-1)
        return even
    first=make();second=make()
    assert first is not second
    assert function_binding(first)==function_binding(second)


def test_torch_context_factory_type_is_part_of_the_binding():
    def effect(value):return value*value
    disabled=torch.no_grad()(effect)
    enabled=torch.enable_grad()(effect)
    value=torch.tensor(2.,requires_grad=True)
    assert disabled(value).requires_grad is False
    assert enabled(value).requires_grad is True
    assert function_binding(disabled)!=function_binding(enabled)
