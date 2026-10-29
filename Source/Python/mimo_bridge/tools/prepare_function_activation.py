# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Prepare actual sensor-function experience without losing the current learner.

Keeps every old weight, optimizer moment, physical episode and observed frame.
The new grounded dynamics extension starts neutral; no copied-study checkpoint
can overwrite learning accumulated by the production trainer.
"""
import argparse
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file
from mimo_bridge import continuation_state
from mimo_bridge.function_transition import activate_function
from .prepare_continuity import native_model
from .migrate_trunk import modules
from lem.training.checkpoint import CheckpointManager
from .training_continuation import atomic_json


def prepare(source,out,episode_storage,candidate=None):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    from .training_continuation import validate_checkpoint
    validate_checkpoint(source)
    old,args=native_model(source)
    changes=dict(body_function='modulate',self_state='learn',self_prediction='grounded_residual')
    new,_=native_model(source,changes)
    loaded=new.load_checkpoint(str(source),load_optimizer=True)
    if loaded.get('reinitialized_components'):raise ValueError('activation lost existing modules')
    saved=load_file(str(source/'model.safetensors'))
    actual={f'{name}.{key}':value.detach().cpu() for name,module in modules(new).items() for key,value in module.state_dict().items()}
    for name,value in saved.items():
        if name not in actual or not torch.equal(value,actual[name]):raise ValueError('old tensor changed: '+name)
    old_names={id(p):f'{name}.{key}' for name,module in modules(old).items() for key,p in module.named_parameters()}
    new_params={f'{name}.{key}':p for name,module in modules(new).items() for key,p in module.named_parameters()}
    state=torch.load(source/'optimizer.pt',map_location='cpu',weights_only=False)['optimizer'];count=0
    if len(state['param_groups'])!=len(old.optimizer.param_groups):raise ValueError('optimizer groups changed')
    for group,parameters in zip(state['param_groups'],old.optimizer.param_groups):
        if len(group['params'])!=len(parameters['params']):raise ValueError('old optimizer layout changed')
        for pid,p in zip(group['params'],parameters['params']):
            name=old_names[id(p)];expected=state['state'].get(pid,{})
            received=new.optimizer.state.get(new_params[name],{})
            if set(expected)!=set(received):raise ValueError('optimizer fields changed: '+name)
            for key,value in expected.items():
                equal=torch.equal(value,received[key].cpu()) if torch.is_tensor(value) else value==received[key]
                if not equal:raise ValueError('optimizer evidence changed: '+name+'/'+key)
            count+=bool(expected)
    episode=continuation_state.load(source)
    if episode is None:raise ValueError('activation requires full physical continuation')
    modified=activate_function(episode)
    modified['config'].update(changes)
    extension_receipt=None
    if candidate is not None:
        from mimo_bridge.self_candidate import transplant_grounded
        extension_receipt=transplant_grounded(new,candidate)
    new.checkpoints=CheckpointManager(new.config,str(out.parent))
    new.receipt['episode_continuation']=continuation_state.VERSION
    created=Path(new.save_checkpoint(new.step_count)).resolve()
    if created!=out:raise ValueError('checkpoint name disagrees with actual training step')
    for name in ('runtime_state.pt','meta.json','lineage.json'):
        if (source/name).is_file():
            shutil.copy2(source/name,out/name)
            if (source/name).read_bytes()!=(out/name).read_bytes():raise ValueError('metadata copy failed')
    episode_receipt=continuation_state.write_episode(out,modified,episode_storage)
    config=dict(args,**changes);config['out']=str(out.parent.parent);config['resume']=str(out)
    atomic_json(out.parent.parent/'args.json',config)
    note=dict(version='function_activation_v1',parent=str(source),checkpoint=str(out),step=new.step_count,
        old_model_tensors_exact=len(saved),old_optimizer_states_exact=count,
        old_observations_preserved=True,world_count=len(modified['worlds']),
        episode=episode_receipt,changes=changes,
        grounded_candidate=extension_receipt,
        scope='Actual modulation and continued self learning; no new choice-policy or learned-capability pass.')
    atomic_json(out/'continuity_migration.json',note)
    return note


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--episode-storage',type=Path,required=True)
    p.add_argument('--candidate',type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.checkpoint,a.out,a.episode_storage,a.candidate),indent=2))
