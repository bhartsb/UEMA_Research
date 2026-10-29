# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Add fresh memory parameters while verifying every existing tensor and Adam state."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file
from mimo_bridge.tools.causal_test import _build_model
from mimo_bridge.tools.migrate_trunk import modules
from lem.training.checkpoint import CheckpointManager


def native_model(src,changes=None):
    """Construct the actual parent, including every production training option."""
    from mimo_bridge.train_mimo import parse_args,build_learner
    args=json.loads((src.parent.parent/'args.json').read_text(encoding='utf-8'))
    config=parse_args(['--out',str(src.parent.parent)])
    vars(config).update(args);vars(config).update(changes or {});config.device='cpu'
    return build_learner(config,str(src.parent)),args


def prepare(src,out,extension='memory',episode_storage=None):
    src,out=Path(src).resolve(),Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    if extension=='self':old,args=native_model(src)
    else:
        old,args,_,found=_build_model(str(src),'cpu',0,intention=True,load=False)
        if not found:raise ValueError('requires located parent configuration')
    if extension=='memory':
        if args.get('persistent_memory','off')!='off':raise ValueError('requires pre-memory configuration')
        changes=dict(persistent_memory='on',contact_profile='mechanical_v2',episode_steps=0,minutes=0.)
        component='persistent_memory'
    elif extension=='self':
        if args.get('self_state','off')!='off':raise ValueError('self perception already exists')
        changes=dict(self_state='learn',body_function='observe')
        component='self_state'
    else:raise ValueError('unknown extension')
    from mimo_bridge import continuation_state
    episode=continuation_state.load(src)
    migrated_episode=None
    if episode is not None:
        if extension!='self':raise ValueError('this extension has no physical episode migration')
        migrated_episode=continuation_state.add_observed_self(episode)
    if extension=='self':new,_=native_model(src,changes)
    else:new,_,_,_=_build_model(str(src),'cpu',0,intention=True,arg_overrides=changes,load=False)
    loaded=new.load_checkpoint(str(src),load_optimizer=True)
    if set(loaded.get('reinitialized_components',[])) != {component}:
        raise ValueError(f'only new {component} may initialize: {loaded}')
    saved=load_file(str(src/'model.safetensors'))
    actual={f'{k}.{n}':v.cpu() for k,m in modules(new).items() for n,v in m.state_dict().items()}
    for name,value in saved.items():
        if name not in actual or not torch.equal(value,actual[name]): raise RuntimeError(f'old weights changed: {name}')
    old_names={id(p):f'{k}.{n}' for k,m in modules(old).items() for n,p in m.named_parameters()}
    new_params={f'{k}.{n}':p for k,m in modules(new).items() for n,p in m.named_parameters()}
    opt=torch.load(src/'optimizer.pt',map_location='cpu',weights_only=False)['optimizer']
    states=0
    if len(opt['param_groups'])!=len(old.optimizer.param_groups):raise RuntimeError('old optimizer group count mismatch')
    for group,old_group in zip(opt['param_groups'],old.optimizer.param_groups):
        if len(group['params'])!=len(old_group['params']): raise RuntimeError('old optimizer layout mismatch')
        for pid,p in zip(group['params'],old_group['params']):
            name=old_names[id(p)];expected=opt['state'].get(pid,{})
            restored=new.optimizer.state.get(new_params[name],{})
            if set(restored)!=set(expected): raise RuntimeError(f'optimizer fields changed: {name}')
            for k,v in expected.items():
                a=restored[k]
                if torch.is_tensor(v):
                    if not torch.equal(v,a.cpu()): raise RuntimeError(f'optimizer evidence changed: {name}/{k}')
                elif v!=a: raise RuntimeError(f'optimizer counter changed: {name}/{k}')
            states+=bool(expected)
    new.checkpoints=CheckpointManager(new.config,str(out.parent))
    if migrated_episode is not None:new.receipt['episode_continuation']=continuation_state.VERSION
    saved_path=Path(new.save_checkpoint(new.step_count))
    if saved_path!=out: raise RuntimeError(f'unexpected checkpoint path {saved_path}')
    # Curriculum and lineage live outside runtime_state.pt. Omitting meta.json
    # resets a mature checkpoint to the launch default (neonate).
    preserved_files=[]
    for item in src.iterdir():
        if item.is_file() and item.name not in ('model.safetensors','optimizer.pt','model_receipt.json','continuity_migration.json',
                                               'episode_state.json','episode_state.pt.gz'):
            shutil.copy2(item,out/item.name)
            if item.read_bytes() != (out/item.name).read_bytes(): raise RuntimeError(f'checkpoint metadata changed: {item.name}')
            preserved_files.append(item.name)
    metadata=json.loads((out/'meta.json').read_text(encoding='utf-8'))
    if metadata != json.loads((src/'meta.json').read_text(encoding='utf-8')):
        raise RuntimeError('curriculum/checkpoint metadata changed')
    # Preserve every pre-existing runtime field, including fields this version
    # does not consume. Episodic memory starts empty in the newly reset world.
    shutil.copy2(src/'runtime_state.pt',out/'runtime_state.pt')
    episode_receipt=None
    if migrated_episode is not None:
        episode_receipt=continuation_state.write_episode(out,migrated_episode,episode_storage)
    (out.parent.parent/'args.json').write_text(json.dumps(dict(args,**changes),indent=2),encoding='utf-8')
    note=dict(parent=str(src),checkpoint=str(out),step=new.step_count,
        old_model_tensors_exact=len(saved),optimizer_states_exact=states,
        runtime_state_exact=True,checkpoint_metadata_exact=True,preserved_auxiliary_files=preserved_files,
        episode_migration=episode_receipt,
        curriculum_state=metadata.get('curriculum_state'),
        new_memory_parameters=sum(p.numel() for p in getattr(new,component).parameters()),
        initial_memory_residual_zero=True,changes=changes,
        scope='Architecture initialization and lineage preservation; not learned behavior')
    (out/'continuity_migration.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    print(json.dumps(note,indent=2));return note


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('--out',required=True);a=ap.parse_args();prepare(a.checkpoint,a.out)
