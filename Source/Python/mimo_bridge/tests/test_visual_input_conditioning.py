# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from lem.model.input_conditioning import InputScaledLinear, VERSION, install, migrate
from mimo_bridge.lem_model import LemLite


def learned_linear():
    torch.manual_seed(71)
    linear=install(torch.nn.Linear(5, 3))
    optimizer=torch.optim.AdamW(linear.parameters(),lr=.001,weight_decay=.02,amsgrad=True)
    x=torch.randn(7,5)*240
    linear(x).square().mean().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    return linear,optimizer,x


def test_install_keeps_parameters_and_random_stream():
    linear=torch.nn.Linear(5,3)
    weight=linear.weight;state=torch.get_rng_state().clone()
    assert install(linear) is linear
    assert linear.weight is weight
    assert torch.equal(state,torch.get_rng_state())


def test_migration_preserves_function_hooks_and_all_adam_moments():
    linear,opt,x=learned_linear()
    seen=[]
    linear.register_forward_hook(lambda m,inp,out:seen.append(inp[0].clone()))
    before=linear(x).detach().clone()
    state=copy.deepcopy(opt.state[linear.weight]);groups=copy.deepcopy(opt.param_groups)
    bias=linear.bias.detach().clone();weight=linear.weight.detach().clone()
    receipt=migrate(linear,261.5,opt)
    assert receipt['migrated']
    torch.testing.assert_close(linear(x),before,atol=4e-5,rtol=1e-5)
    assert torch.equal(seen[0],seen[1])
    assert torch.equal(linear.weight,weight*261.5)
    assert torch.equal(linear.bias,bias)
    for key,power in [('exp_avg',1),('exp_avg_sq',2),('max_exp_avg_sq',2)]:
        torch.testing.assert_close(opt.state[linear.weight][key],state[key]/261.5**power)
    assert torch.equal(opt.state[linear.weight]['step'],state['step'])
    for before_group,after_group in zip(groups,opt.param_groups):
        assert {k:v for k,v in before_group.items() if k!='params'}=={k:v for k,v in after_group.items() if k!='params'}


def test_reload_retains_forward_units_and_migration_cannot_repeat():
    linear,opt,x=learned_linear();migrate(linear,261.5,opt)
    weights=copy.deepcopy(linear.state_dict());moments=copy.deepcopy(opt.state_dict())
    other=install(torch.nn.Linear(5,3));other.load_state_dict(weights,strict=True)
    other_opt=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=.02,amsgrad=True)
    other_opt.load_state_dict(moments)
    assert isinstance(other,InputScaledLinear)
    torch.testing.assert_close(other(x),linear(x),atol=0,rtol=0)
    assert not migrate(other,261.5,other_opt)['migrated']
    assert torch.equal(other.weight,linear.weight)
    with pytest.raises(ValueError,match='cannot silently'):
        migrate(other,200,other_opt)


def test_legacy_checkpoint_load_restores_identity_including_after_scaled_load():
    linear,opt,x=learned_linear()
    old={k:v.clone() for k,v in linear.state_dict().items() if k!='input_scale'}
    original=linear(x).detach().clone()
    migrate(linear,261.5,opt);linear.load_state_dict(old,strict=True)
    assert float(linear.input_scale)==1
    torch.testing.assert_close(linear(x),original,atol=0,rtol=0)


@pytest.mark.parametrize('scale',[0,-2,float('inf'),float('nan')])
def test_invalid_migration_is_refused_before_weights_change(scale):
    linear,opt,x=learned_linear();before=linear.weight.clone()
    with pytest.raises(ValueError):migrate(linear,scale,opt)
    assert torch.equal(before,linear.weight)


def test_repaired_rows_keep_original_inputs_and_outputs():
    from lem.model.liveness import RevivedRows
    linear,opt,x=learned_linear();rows=RevivedRows(5,3)
    rows.mask[1]=True;rows.any_py=True;rows.centre_mean.copy_(torch.randn(5)*30)
    linear.register_forward_hook(lambda mod,inp,out:rows.apply(inp[0],out))
    before=linear(x).detach().clone();migrate(linear,261.5,opt)
    after=linear(x).detach()
    torch.testing.assert_close(after,before,atol=4e-5,rtol=1e-5)
    assert torch.equal(after[:,1],before[:,1])
    # The liveness probe calls forward directly to avoid recursively firing hooks.
    torch.testing.assert_close(rows.apply(x,linear.forward(x)),after)


@pytest.mark.parametrize('requested_scale',[261.5,261.53])
def test_source_hash_and_receipt_guard_the_one_time_checkpoint_migration(tmp_path,requested_scale):
    linear,opt,x=learned_linear()
    model=SimpleNamespace(prediction_heads=SimpleNamespace(heads={'vision':SimpleNamespace(mlp=[linear])}),
                          optimizer=opt,receipt={})
    source=tmp_path/'model.safetensors';source.write_bytes(b'preserved checkpoint bytes')
    spec=dict(version=VERSION,scale=requested_scale,source_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    calibration=tmp_path/'calibration.json';calibration.write_text(json.dumps(spec))
    LemLite.configure_visual_input_calibration(model,calibration)
    before=linear(x).detach().clone()
    LemLite._restore_visual_input_conditioning(model,tmp_path,True)
    assert model.receipt['visual_input_conditioning']['scale']==float(linear.input_scale)
    torch.testing.assert_close(linear(x),before,atol=4e-5,rtol=1e-5)
    (tmp_path/'model_receipt.json').write_text(json.dumps(model.receipt))
    unchanged=linear.weight.detach().clone()
    LemLite._restore_visual_input_conditioning(model,tmp_path,True)
    assert torch.equal(linear.weight,unchanged)
    (tmp_path/'model_receipt.json').unlink()
    with pytest.raises(ValueError,match='matching version/scale'):
        LemLite._restore_visual_input_conditioning(model,tmp_path,True)


def test_wrong_source_checkpoint_cannot_be_calibrated(tmp_path):
    linear,opt,x=learned_linear()
    model=SimpleNamespace(prediction_heads=SimpleNamespace(heads={'vision':SimpleNamespace(mlp=[linear])}),
                          optimizer=opt,receipt={},_visual_input_calibration={
                              'version':VERSION,'scale':261.5,'source_model_sha256':'0'*64})
    (tmp_path/'model.safetensors').write_bytes(b'different checkpoint')
    before=linear.weight.detach().clone()
    with pytest.raises(ValueError,match='different checkpoint'):
        LemLite._restore_visual_input_conditioning(model,tmp_path,True)
    assert torch.equal(linear.weight,before)
