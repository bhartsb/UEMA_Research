# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Append two trunk handles, preserving every existing weight and Adam moment."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from mimo_bridge.tools.causal_test import _build_model

APPEND = {
    'encoder_registry.encoders.proprioception.mlp.0.weight': 1,
    'encoder_registry.encoders.motor.patch_pos': 1,
    'motor_controller.output_proj.weight': 0,
    'motor_controller.output_proj.bias': 0,
    'wellness.grounded.motor_state.net.0.weight': 1,
    'tenure_heads.pose.2.weight': 0,
    'tenure_heads.pose.2.bias': 0,
    'tenure_heads.order_proj.0.weight': 1,
}


def modules(model):
    out = {k: getattr(model, k) for k in ('encoder_registry', 'forward_model', 'prediction_heads',
                                         'motor_controller', 'wellness', 'gaze_head', 'voice_head')}
    out.update(model._side_modules())
    return {k: v for k, v in out.items() if v is not None}


def expanded(name, before, template):
    if before.shape == template.shape:
        return before.clone()
    after = torch.zeros_like(template, device='cpu', dtype=before.dtype)
    if name == 'motor_controller.input_proj.weight':
        # Latents, achieved state, wellness derivative: insert within the input.
        cut = before.shape[1] - 24*9
        if after.shape[1] != before.shape[1] + 2*19:
            raise ValueError('unexpected controller input growth')
        after[:, :cut] = before[:, :cut]
        after[:, cut+20:cut+20+24*9] = before[:, cut:]
    elif name in APPEND:
        axis = APPEND[name]
        if any(a != b for i, (a, b) in enumerate(zip(before.shape, after.shape)) if i != axis):
            raise ValueError(f'unexpected shape change: {name}')
        after[tuple(slice(0, d) for d in before.shape)] = before
    else:
        raise ValueError(f'unreviewed shape change: {name}: {before.shape} -> {template.shape}')
    return after


def schema_growth(state):
    from lem.model.body_schema import DifferentiableBodySchema
    templates = DifferentiableBodySchema(n_handles=26).state_dict()
    result = {}
    for name, value in state.items():
        new = templates[name].clone()
        if value.ndim != new.ndim or any(a > b for a, b in zip(value.shape, new.shape)):
            raise ValueError(f'unsupported body-schema change: {name}')
        new[tuple(slice(0, d) for d in value.shape)] = value
        result[name] = new
    return result


def migrate(checkpoint, out):
    checkpoint, out = Path(checkpoint).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    old, args, _, found = _build_model(str(checkpoint), 'cpu', 0, intention=True)
    if not found or args.get('body_layout', 'legacy') != 'legacy':
        raise ValueError('migration requires a located 24-handle run configuration')
    old.load_checkpoint(str(checkpoint), load_optimizer=True)
    new, _, _, _ = _build_model(str(checkpoint), 'cpu', 0, intention=True,
                                arg_overrides={'body_layout': 'trunk_v1'}, load=False)
    templates = {f'{k}.{n}': v for k, mod in modules(new).items() for n, v in mod.state_dict().items()}
    saved = load_file(str(checkpoint/'model.safetensors'))
    if set(saved) != set(templates):
        raise ValueError(f'model tensor names differ: {set(saved)^set(templates)}')
    grown = {k: expanded(k, v, templates[k]) for k, v in saved.items()}

    # Initial desired orientations are measured from this body's ordinary reset.
    # No trajectory, named task, or roll direction is supplied.
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.train_mimo import parse_args, env_kwargs_for
    from types import SimpleNamespace
    cfg = vars(parse_args(['--out', str(out.parent.parent)]))
    cfg.update(args)
    kw = env_kwargs_for(SimpleNamespace(**cfg), old.dev_limits(old.step_count)['phase_name'],
                        str(out.parent.parent/'migration_scene'))
    kw.update(body_layout='trunk_v1', vision=False, caregiver_mode='off', audio_real=False)
    source = SingleEnvSource(**kw)
    try:
        bones = source.reset()[0].bones[-2:]
    finally:
        source.close()
    bias = grown['motor_controller.output_proj.bias'].view(26,9)
    bias[24:, :3] = torch.from_numpy(bones[:,4:7]/200.)
    bias[24:, 3:7] = torch.from_numpy(bones[:, :4])
    # Use the controller's ordinary initialization: sigmoid(0) = 0.5 strength.
    # A near-zero bias would leave these rows weak because baseline training
    # has no achieved-strength target from which to learn a different bias.
    bias[24:, 8] = 0.

    named_old = {id(p): f'{k}.{n}' for k, mod in modules(old).items() for n,p in mod.named_parameters()}
    named_new = {id(p): f'{k}.{n}' for k, mod in modules(new).items() for n,p in mod.named_parameters()}
    packed = torch.load(checkpoint/'optimizer.pt', map_location='cpu', weights_only=False)
    optimizer = copy.deepcopy(packed['optimizer'])
    moments, states = 0, 0
    if not len(optimizer['param_groups']) == len(old.optimizer.param_groups) == len(new.optimizer.param_groups):
        raise ValueError('optimizer group count changed')
    for stored, og, ng in zip(optimizer['param_groups'], old.optimizer.param_groups, new.optimizer.param_groups):
        on = [named_old[id(p)] for p in og['params']]
        nn = [named_new[id(p)] for p in ng['params']]
        if on != nn or len(stored['params']) != len(on):
            raise ValueError('optimizer parameter ordering changed')
        for index, name in zip(stored['params'], on):
            state = optimizer['state'].get(index, {})
            states += bool(state)
            for key, value in state.items():
                if torch.is_tensor(value) and value.ndim:
                    state[key] = expanded(name, value, templates[name])
                    moments += 1
    packed['optimizer'] = optimizer
    out.mkdir(parents=True)
    for item in checkpoint.iterdir():
        if item.is_file() and item.name not in ('model.safetensors', 'optimizer.pt', 'runtime_state.pt', 'model_receipt.json'):
            shutil.copy2(item, out/item.name)
    save_file(grown, str(out/'model.safetensors'))
    torch.save(packed, out/'optimizer.pt')
    runtime = torch.load(checkpoint/'runtime_state.pt', map_location='cpu', weights_only=False)
    runtime['body_schemas'] = [schema_growth(s) for s in runtime.get('body_schemas', [])]
    torch.save(runtime, out/'runtime_state.pt')
    (out/'model_receipt.json').write_text(json.dumps(new.receipt, indent=2), encoding='utf-8')
    new_args = dict(args, body_layout='trunk_v1')
    (out.parent.parent/'args.json').write_text(json.dumps(new_args, indent=2), encoding='utf-8')
    loaded = new.load_checkpoint(str(out), load_optimizer=True)
    if loaded.get('reinitialized_components'):
        raise RuntimeError(f'migrated checkpoint reinitialized components: {loaded}')
    after = {f'{k}.{n}': v for k, mod in modules(new).items() for n,v in mod.state_dict().items()}
    for name, value in grown.items():
        if not torch.equal(value, after[name].cpu()):
            raise RuntimeError(f'loaded tensor differs: {name}')
    actual = new.optimizer.state_dict()
    for key, state in optimizer['state'].items():
        for name, value in state.items():
            target = actual['state'][key][name]
            if torch.is_tensor(value):
                assert torch.equal(value.cpu(), target.cpu()), (key, name)
            else:
                assert value == target, (key, name)
    receipt = dict(parent=str(checkpoint), checkpoint=str(out), step=old.step_count,
                   layout='trunk_v1', old_rows_preserved=24, added_handles=['lower_trunk','chest'],
                   optimizer_states_preserved=states, optimizer_moment_tensors_preserved=moments,
                   old_weights_preserved=True, body_schema_evidence_preserved=True,
                   hself_preserved=True,
                   changed_shapes={k: [list(v.shape),list(grown[k].shape)] for k,v in saved.items() if v.shape != grown[k].shape},
                   new_motor_initial_strength=0.5,
                   note='New motor tokens change attention pooling; previous outputs are not claimed identical.')
    (out/'trunk_migration.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    print(json.dumps(migrate(args.checkpoint, args.out), indent=2))
