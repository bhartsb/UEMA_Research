# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Checkpoint receipts describe saved architecture, not the earlier base model."""
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open

from lem.training.checkpoint import CheckpointManager, _save_receipt


class LaterModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.learned = nn.Parameter(torch.arange(12.).reshape(3, 4))
        self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
        self.register_buffer('history', torch.arange(101.))
        self.register_buffer('cache', torch.zeros(103), persistent=False)


def test_saved_receipt_includes_late_parameters_and_separates_buffers(tmp_path):
    names = ('encoder_registry', 'forward_model', 'prediction_heads', 'motor_controller', 'wellness')
    modules = {name: nn.Linear(2, 2) for name in names}
    late = LaterModule()
    params = [p for mod in [*modules.values(), late] for p in mod.parameters()]
    optimizer = torch.optim.AdamW(params)
    sum(p.square().sum() for p in params).backward()
    optimizer.step()
    old_optimizer = deepcopy(optimizer.state_dict())
    receipt = dict(total_params=6, generated_at='construction time',
                   contract='retain me', encoders_built={'vision': {'d_output': 128}})
    original = deepcopy(receipt)
    manager = CheckpointManager(dict(checkpoint=dict(format='safetensors', keep_last_n=3,
                                                     progressive_resize=False)), str(tmp_path))
    checkpoint = Path(manager.save(9, **modules, optimizer=optimizer, model_receipt=receipt,
                                   side_modules={'self_state': late}))
    saved = json.loads((checkpoint/'model_receipt.json').read_text())
    assert receipt == original
    assert saved['contract'] == 'retain me'
    assert saved['total_params'] == 45  # 5 x (2 x 2 + 2), then 12 learned + 3 frozen.
    inventory = saved['checkpoint_inventory']
    assert inventory['step'] == 9
    side = inventory['components']['self_state']
    assert side['parameters'] == 15 and side['trainable_parameters'] == 12
    assert side['saved_buffer_elements'] == 101
    assert 'cache' not in side['tensors']
    flat = {f'{component}.{key}': tensor for component, info in inventory['components'].items()
            for key, tensor in info['tensors'].items()}
    with safe_open(str(checkpoint/'model.safetensors'), framework='pt') as file:
        assert set(flat) == set(file.keys())
        for key, info in flat.items():
            tensor = file.get_tensor(key)
            assert info['shape'] == list(tensor.shape)
            assert info['dtype'] == str(tensor.dtype)
            assert info['elements'] == tensor.numel()
    after = torch.load(checkpoint/'optimizer.pt', weights_only=False)['optimizer']
    assert after['param_groups'] == old_optimizer['param_groups']
    for key, value in after['state'].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, old_optimizer['state'][key][name])


def test_receipt_counts_shared_parameters_once_and_does_not_count_cache():
    module = LaterModule()
    alias = nn.Module(); alias.shared = module.learned
    # Registration and optimizer membership are different facts.
    optimizer = torch.optim.AdamW([module.learned])
    receipt = _save_receipt({}, {'self': module, 'alias': alias}, optimizer, 3)
    assert receipt['total_params'] == 15
    assert receipt['checkpoint_inventory']['unique_parameter_tensors'] == 2
    assert receipt['checkpoint_inventory']['optimizer_parameters'] == 12


def test_native_self_learning_checkpoint_receipt_counts_the_whole_model(tmp_path):
    from mimo_bridge.lem_model import LemLite
    model = LemLite(device='cpu', imagination=False, checkpoint_dir=str(tmp_path),
                    body_layout='trunk_v1', persistent_memory='on', self_state='learn',
                    self_prediction='grounded_residual', next_frame_loss=True)
    original = deepcopy(model.receipt)
    assert original['total_params'] < model.n_params  # Reproduces the actual defect.
    checkpoint = Path(model.save_checkpoint(12))
    receipt = json.loads((checkpoint/'model_receipt.json').read_text())
    assert model.receipt == original  # No live receipt/configuration mutation.
    assert receipt['total_params'] == model.n_params
    inventory = receipt['checkpoint_inventory']
    assert {'prediction_heads', 'recon_decoders', 'persistent_memory', 'self_state'} <= inventory['components'].keys()
    assert inventory['optimizer_parameters'] == model.n_params
    count = sum(len(info['tensors']) for info in inventory['components'].values())
    with safe_open(str(checkpoint/'model.safetensors'), framework='pt') as file:
        assert count == len(file.keys())
