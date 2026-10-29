# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""An older copied candidate cannot overwrite the continuing baby's learning."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from lem.model.self_state import SelfState
from mimo_bridge.self_candidate import transplant_grounded


def make_model(*,earned_grounded=False,older_extension=False):
    module=SelfState(16,1,9,'grounded_residual')
    if older_extension:
        del module.grounded.spatial_context
        del module.grounded.site_observation
    old=torch.nn.Linear(3,2)
    optimizer=torch.optim.Adam(list(old.parameters())+list(module.parameters()),lr=.002)
    model=SimpleNamespace(old=old,self_state=module,optimizer=optimizer,body_layout='trunk_v1')
    trainable=list(old.parameters())+[param for name,param in module.named_parameters()
        if earned_grounded or not name.startswith('grounded.')]
    sum(param.square().sum() for param in trainable).backward();optimizer.step();optimizer.zero_grad()
    return model


def named_modules(model):return dict(old=model.old,self_state=model.self_state)


def save_candidate(model,path):
    path.mkdir()
    weights={name+'.'+key:value for name,module in named_modules(model).items()
             for key,value in module.state_dict().items()}
    save_file(weights,str(path/'model.safetensors'))
    torch.save(dict(optimizer=model.optimizer.state_dict()),path/'optimizer.pt')
    (path/'model_receipt.json').write_text(json.dumps(dict(body_layout='trunk_v1',self_prediction='grounded_residual')))
    return path


def old_evidence(model):
    evidence={}
    for name,module in named_modules(model).items():
        for key,param in module.named_parameters():
            if not (name+'.'+key).startswith('self_state.grounded.'):
                evidence[name+'.'+key]=(param.detach().clone(),copy.deepcopy(model.optimizer.state.get(param)))
    return evidence


def assert_evidence_equal(first,second):
    assert first.keys()==second.keys()
    for key,(weight,states) in first.items():
        assert torch.equal(weight,second[key][0]),key
        assert states.keys()==second[key][1].keys()
        for state,value in states.items():assert torch.equal(value,second[key][1][state]),(key,state)


@pytest.mark.parametrize('older_extension',[False,True])
def test_transplant_keeps_latest_old_weights_and_every_old_adam_moment(tmp_path,monkeypatch,older_extension):
    monkeypatch.setattr('mimo_bridge.tools.migrate_trunk.modules',named_modules)
    source=make_model(earned_grounded=True,older_extension=older_extension)
    destination=make_model();before=old_evidence(destination)
    receipt=transplant_grounded(destination,save_candidate(source,tmp_path/'candidate'))
    assert not receipt['production_choice_license'] and not receipt['older_brain_imported']
    assert_evidence_equal(before,old_evidence(destination))
    source_params=dict(source.self_state.grounded.named_parameters())
    for name,param in destination.self_state.grounded.named_parameters():
        if name in source_params:
            expected=source_params[name]
            assert torch.equal(param,expected)
            for key,value in source.optimizer.state[expected].items():
                assert torch.equal(value,destination.optimizer.state[param][key]),(name,key)
        else:assert param not in destination.optimizer.state
    with pytest.raises(ValueError,match='earned destination'):
        transplant_grounded(destination,tmp_path/'candidate')


def test_bad_optimizer_candidate_cannot_partially_replace_weights(tmp_path,monkeypatch):
    monkeypatch.setattr('mimo_bridge.tools.migrate_trunk.modules',named_modules)
    source=make_model(earned_grounded=True);destination=make_model()
    source.optimizer.state[source.self_state.grounded.change.weight]['exp_avg'].fill_(float('nan'))
    path=save_candidate(source,tmp_path/'bad');before=copy.deepcopy(destination.self_state.state_dict())
    old=old_evidence(destination)
    with pytest.raises(ValueError,match='nonfinite'):transplant_grounded(destination,path)
    assert all(torch.equal(value,destination.self_state.state_dict()[name]) for name,value in before.items())
    assert_evidence_equal(old,old_evidence(destination))


def test_full_precision_audit_does_not_invent_lost_pressure_distinctions():
    from mimo_bridge.tools.function_recovery_fixture import pressure_resolution,pressure_distinctions
    profile=dict(minimum_bits=2,maximum_bits=23,pressure_range_pa=100000.)
    pressure=np.array([1.,1.001,1.002],dtype=np.float32)
    bits,width=pressure_resolution(profile,np.ones(3))
    assert (bits==23).all() and (width==0).all()
    report=pressure_distinctions(pressure,pressure,width)
    assert set(report.values())=={3}
    _,width=pressure_resolution(profile,np.zeros(3))
    experienced=np.rint(pressure/width)*width
    report=pressure_distinctions(pressure,experienced,width)
    assert report['distinct_physical_pressures']==3
    assert report['distinct_experienced_pressures']==1
