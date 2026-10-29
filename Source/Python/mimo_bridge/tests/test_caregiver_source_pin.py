# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from mimo_bridge.tools import caregiver_source_pin as pinning


def fixture():
    original={'lem/model/causal_readout_v2.py':'old','mimo_bridge/lem_model.py':'brain'}
    pin=dict(original_source_inventory=original,saved_sha256=pinning.digest(original),
        changed_module='lem/model/causal_readout_v2.py',current_auxiliary_sha256='reviewed')
    return pin,dict(original,**{'lem/model/causal_readout_v2.py':'reviewed'})


def test_exact_original_inventory_is_reconstructed_for_only_one_reviewed_unused_edit():
    pin,current=fixture()
    assert pinning.verify(pin,current)==pin['saved_sha256']


@pytest.mark.parametrize('change',['learner','added','removed','auxiliary_again','original_receipt'])
def test_any_unreviewed_source_or_binding_change_is_rejected(change):
    pin,current=fixture()
    if change=='learner':current['mimo_bridge/lem_model.py']='different'
    if change=='added':current['lem/model/another.py']='new'
    if change=='removed':del current['mimo_bridge/lem_model.py']
    if change=='auxiliary_again':current['lem/model/causal_readout_v2.py']='another'
    if change=='original_receipt':pin['saved_sha256']='wrong'
    with pytest.raises(ValueError):pinning.verify(pin,current)


def test_auxiliary_import_is_forbidden_instead_of_assumed_unused(monkeypatch):
    monkeypatch.delitem(sys.modules,pinning.MODULE,raising=False)
    monkeypatch.setattr(sys,'meta_path',[pinning.BlockAuxiliary(),*sys.meta_path])
    with pytest.raises(ImportError,match='forbids'):
        importlib.import_module(pinning.MODULE)


def test_copied_inventory_tree_cannot_authorize_actual_changed_execution(tmp_path):
    actual=Path(pinning.__file__).resolve()
    pinning.verify_actual_root(actual.parents[2],actual)
    with pytest.raises(ValueError,match='actual imported'):
        pinning.verify_actual_root(tmp_path/'old_source',actual)


def test_preloaded_auxiliary_alias_is_rejected(tmp_path):
    auxiliary=(tmp_path/'causal_readout_v2.py').resolve()
    with pytest.raises(RuntimeError,match='already loaded'):
        pinning.reject_loaded_auxiliary({'apparently_unrelated':SimpleNamespace(__file__=str(auxiliary))},auxiliary)


def test_future_or_unrelated_pin_contract_is_not_reinterpreted(tmp_path):
    path=tmp_path/'pin.json';path.write_text(json.dumps({'contract':'different_future_contract'}))
    with pytest.raises(ValueError,match='Unrecognized'):
        pinning.install(path)
