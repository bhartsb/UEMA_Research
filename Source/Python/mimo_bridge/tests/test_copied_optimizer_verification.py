# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from mimo_bridge.tools.caregiver_learning_trial import verify_loaded


def test_new_neutral_parameters_may_shift_ids_but_must_preserve_named_histories(tmp_path,monkeypatch):
    from mimo_bridge.tools import migrate_trunk
    old=nn.ParameterDict({'a':nn.Parameter(torch.ones(2)), 'z':nn.Parameter(torch.ones(2)*2)})
    optimizer=torch.optim.AdamW(old.parameters());sum(p.square().sum() for p in old.parameters()).backward();optimizer.step()
    torch.save({'optimizer':optimizer.state_dict()},tmp_path/'optimizer.pt')
    save_file({'part.'+k:v for k,v in old.state_dict().items()},str(tmp_path/'model.safetensors'))
    grown=nn.ParameterDict({'a':nn.Parameter(old['a'].detach().clone()),'fresh':nn.Parameter(torch.zeros(2)),
                            'z':nn.Parameter(old['z'].detach().clone())})
    restored=torch.optim.AdamW(grown.parameters())
    for key in ('a','z'):restored.state[grown[key]]=deepcopy(optimizer.state[old[key]])
    model=SimpleNamespace(optimizer=restored)
    monkeypatch.setattr(migrate_trunk,'modules',lambda _:dict(part=grown))
    receipt=dict(reinitialized_components=[],loaded_components=['part'])
    verify_loaded(model,receipt,tmp_path,True)
    restored.state[grown['z']]['exp_avg'].add_(1)
    with pytest.raises(ValueError,match='history changed'):verify_loaded(model,receipt,tmp_path,True)
