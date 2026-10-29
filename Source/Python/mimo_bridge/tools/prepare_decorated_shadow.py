# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Explicitly extend an unqualified shadow's code contract without retraining.

No production controls. First record the current source/binding manifest, then
prepare a separate checkpoint only while those recorded sources remain exact.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import uuid
import torch

from mimo_bridge import continuation_state
from mimo_bridge.acting_snapshot import model_execution_fingerprint,model_execution_description
from mimo_bridge.acting_snapshot import configuration_fingerprint
from mimo_bridge.self_intention import perception_modules,fingerprint,perception_fingerprint
from mimo_bridge.decorated_acting_snapshot import record_manifest,wrap_shadow_state
from .prepare_continuity import native_model
from .prepare_local_shadow import state_digest
from .training_continuation import atomic_json


def context(source):
    source=Path(source).resolve()
    episode=continuation_state.load(source)
    if episode is None:raise ValueError('full physical episode required')
    meta=json.loads((source/'meta.json').read_text(encoding='utf-8'))
    if (episode['config'].get('self_choice')!='shadow' or meta['step']!=episode['trainer']['step']
            or source.name!=f"ckpt_step_{meta['step']}"):
        raise ValueError('source must be the exact continuing shadow checkpoint')
    actor=episode['trainer'].get('acting_snapshot')
    if actor is None or actor.get('qualification') is not True:
        raise ValueError('only an existing unqualified shadow may acquire the extension')
    if configuration_fingerprint(actor['configuration'])!=actor['receipt']['acting_configuration_sha256']:
        raise ValueError('saved actor configuration differs from its receipt')
    model,args=native_model(source,dict(self_choice='off',self_choice_receipt=''))
    from mimo_bridge.train_mimo import parse_args,acting_configuration
    original_args=parse_args(['--out',str(source.parent.parent)]);vars(original_args).update(args)
    if acting_configuration(original_args)!=actor['configuration']:
        raise ValueError('source launch configuration differs from its frozen actor')
    model.config.clear();model.config.update(copy.deepcopy(actor['model_config']))
    # This CPU reconstruction is never executed. Reproduce the recorded
    # production precision from its configuration, then check the ENTIRE saved
    # execution descriptor. A CPU proof below explicitly changes precision in
    # its separate copy; the production actor identity is never rebound here.
    model.dtype=recorded_precision(actor['model_config'],actor['configuration'])
    if set(perception_modules(model))!=set(actor['modules']):raise ValueError('saved shadow module layout differs')
    for name,module in perception_modules(model).items():module.load_state_dict(actor['modules'][name],strict=True)
    if (fingerprint(model.self_state)!=actor['receipt']['self_weights_sha256']
            or perception_fingerprint(model)!=actor['receipt']['perception_weights_sha256']):
        raise ValueError('saved shadow weights do not match its receipt')
    if model_execution_fingerprint(model)!=actor['receipt']['model_execution_sha256']:
        expected=actor['receipt'].get('model_execution_description',{})
        actual=model_execution_description(model)
        differences=[]
        def diff(a,b,path):
            if isinstance(a,dict) and isinstance(b,dict):
                for key in sorted(set(a)|set(b)):diff(a.get(key),b.get(key),path+'.'+key)
            elif a!=b:differences.append(dict(field=path,saved=a,current=b))
        diff(expected,actual,'execution')
        raise ValueError('current legacy execution differs from the saved shadow; no implicit requalification: '
                         +json.dumps(differences[:12],default=str))
    return model,args,episode


def recorded_precision(config,configuration):
    device=torch.device(configuration.get('device','cpu'))
    if device.type=='cpu':return torch.float32
    if device.type!='cuda':raise ValueError('unsupported recorded actor device')
    precision=config['model'].get('precision','fp32')
    known={'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}
    if precision not in known:raise ValueError('unsupported recorded actor precision')
    return known[precision]


def record(source,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    model,_,_=context(source)
    manifest=record_manifest(model)
    manifest['recorded_from_checkpoint']=str(Path(source).resolve())
    atomic_json(out,manifest)
    return manifest


def cpu_proof_episode(episode,out):
    """Explicit device/recording accommodation of an isolated test copy only."""
    outer=episode['trainer']['acting_snapshot'];actor=outer['actor']
    archived=dict(learner_cuda_rng=episode['learner']['cuda_rng'],
        actor_causal_cuda_rng=actor['causal']['cuda_rng'],actor_random_cuda=actor['random']['cuda'],
        actor_configuration=copy.deepcopy(actor['configuration']),
        actor_configuration_sha256=actor['receipt']['acting_configuration_sha256'],
        actor_execution_description=copy.deepcopy(actor['receipt']['model_execution_description']),
        actor_execution_sha256=actor['receipt']['model_execution_sha256'])
    description=copy.deepcopy(archived['actor_execution_description'])
    if configuration_fingerprint(description)!=archived['actor_execution_sha256']:
        raise ValueError('source execution description does not match its saved digest')
    description['dtype']='torch.float32'
    configuration=dict(actor['configuration'],device='cpu')
    copied=dict(actor,configuration=configuration,
        receipt=dict(actor['receipt'],acting_configuration_sha256=configuration_fingerprint(configuration),
                     model_execution_description=description,model_execution_sha256=configuration_fingerprint(description)),
        causal=dict(actor['causal'],cuda_rng=None),random=dict(actor['random'],cuda=None))
    worlds=[];paths=[]
    for index,world in enumerate(episode['worlds']):
        kw=dict(world['env_kwargs'],run_dir=str(out.parent.parent/'scene_copies'/str(index)))
        body=dict(world['world']);caregiver=body.get('caregiver')
        old_status=None
        if caregiver is not None:
            caregiver=dict(caregiver);events=copy.copy(caregiver['events'])
            old_status=events.status_path;events.status_path=out.parent.parent/f'copied_recording_status_{index}.json'
            caregiver['events']=events;body['caregiver']=caregiver
        paths.append(dict(run_dir=world['env_kwargs']['run_dir'],recording_status=old_status))
        worlds.append(dict(world,env_kwargs=kw,world=body))
    archived['world_paths']=paths
    result=dict(episode,learner=dict(episode['learner'],cuda_rng=None),worlds=worlds,
        trainer=dict(episode['trainer'],acting_snapshot=dict(outer,actor=copied)))
    return result,archived


def restore_cpu_probe_values(episode,archived):
    """Prove that the declared copy exceptions account for every changed value."""
    outer=episode['trainer']['acting_snapshot'];actor=outer['actor']
    actor=dict(actor,configuration=archived['actor_configuration'],
        receipt=dict(actor['receipt'],acting_configuration_sha256=archived['actor_configuration_sha256'],
                     model_execution_description=archived['actor_execution_description'],
                     model_execution_sha256=archived['actor_execution_sha256']),
        causal=dict(actor['causal'],cuda_rng=archived['actor_causal_cuda_rng']),
        random=dict(actor['random'],cuda=archived['actor_random_cuda']))
    worlds=[]
    for world,paths in zip(episode['worlds'],archived['world_paths']):
        body=dict(world['world']);caregiver=body.get('caregiver')
        if caregiver is not None:
            caregiver=dict(caregiver);events=copy.copy(caregiver['events']);events.status_path=paths['recording_status']
            caregiver['events']=events;body['caregiver']=caregiver
        worlds.append(dict(world,env_kwargs=dict(world['env_kwargs'],run_dir=paths['run_dir']),world=body))
    return dict(episode,learner=dict(episode['learner'],cuda_rng=archived['learner_cuda_rng']),worlds=worlds,
        trainer=dict(episode['trainer'],acting_snapshot=dict(outer,actor=actor)))


def prepare(source,out,manifest_path,episode_storage=None,*,cpu_probe=False):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists() or out.name!=source.name:raise ValueError('requires a separate destination with the identical checkpoint step')
    model,args,episode=context(source)
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    # A newer continuing snapshot may migrate under the previously recorded
    # unchanged code, but its exact actor execution must still pass context().
    before=state_digest(episode)
    wrapped=wrap_shadow_state(episode['trainer']['acting_snapshot'],model,manifest)
    migrated=dict(episode,trainer=dict(episode['trainer'],acting_snapshot=wrapped))
    unwrapped=dict(migrated,trainer=dict(migrated['trainer'],acting_snapshot=wrapped['actor']))
    if state_digest(unwrapped)!=before:raise ValueError('migration changed prior episode values')
    archived=None
    if cpu_probe:
        expected=state_digest(migrated)
        migrated,archived=cpu_proof_episode(migrated,out)
        if state_digest(restore_cpu_probe_values(migrated,archived))!=expected:
            raise ValueError('CPU test accommodation changed undeclared episode values')
        model.dtype=torch.float32
        if model_execution_fingerprint(model)!=migrated['trainer']['acting_snapshot']['actor']['receipt']['model_execution_sha256']:
            raise ValueError('isolated CPU execution differs beyond the declared precision accommodation')
    required=('model.safetensors','optimizer.pt','runtime_state.pt','meta.json','model_receipt.json')
    if not all((source/name).is_file() for name in required):raise ValueError('incomplete native source checkpoint')
    out.parent.mkdir(parents=True,exist_ok=True)
    configuration=dict(args,out=str(out.parent.parent),resume=str(out),checkpoint_storage=str(out.parent))
    if cpu_probe:configuration['device']='cpu'
    args_path=out.parent.parent/'args.json'
    if args_path.exists() and json.loads(args_path.read_text(encoding='utf-8'))!=configuration:
        raise ValueError('destination run already has different checkpoint ancestry')
    staging=out.parent/('.staging_decorated_'+uuid.uuid4().hex);staging.mkdir()
    preserved={}
    for item in source.rglob('*'):
        if not item.resolve().is_relative_to(source):raise ValueError('checkpoint contains an external file link')
        relative=item.relative_to(source)
        if item.is_dir():(staging/relative).mkdir(exist_ok=True);continue
        if str(relative) in ('episode_state.json','episode_state.pt.gz'):continue
        target=staging/relative;shutil.copy2(item,target)
        digest=continuation_state.file_sha256(item)
        if continuation_state.file_sha256(target)!=digest:raise ValueError('native checkpoint bytes changed: '+item.name)
        preserved[str(relative)]=digest
    continuation_state.write_episode(staging,migrated,episode_storage,published_path=out)
    note=dict(contract='decorated_shadow_migration_v1',parent=str(source),checkpoint=str(out),
        step=episode['trainer']['step'],native_file_sha256=preserved,unwrapped_episode_sha256=before,
        binding_sha256=manifest['binding']['sha256'],qualification=False,
        old_inner_actor_state_exact=True,old_learner_and_physical_episode_exact=True,
        history_scope='Existing evidence retained; this records code prospectively and does not certify the previously unbound interval',
        runtime_requirement='Outer actor reader required; legacy reader intentionally refuses this state',cpu_probe_only=cpu_probe)
    if cpu_probe:
        note.update(old_inner_actor_state_exact=False,old_learner_and_physical_episode_exact=False,
                    original_values_exact_after_reversing_declared_cpu_accommodation=True)
        torch.save(archived,staging/'cpu_probe_original_values.pt')
        actor=migrated['trainer']['acting_snapshot']['actor']
        cpu_receipt=dict(version='decorated_shadow_cpu_proof_v1',parent=str(source),checkpoint=str(out),
            step=episode['trainer']['step'],world_count=len(episode['worlds']),
            sim_times=[world['payload']['sim_time'] for world in episode['worlds']],
            cpu_probe_only=True,production_choice_license=False,old_episode_exact_except_declared_boundary=True,
            original_episode_value_sha256=before,original_values_sha256=continuation_state.file_sha256(staging/'cpu_probe_original_values.pt'),
            changes=dict(contact_profile=episode['config']['contact_profile'],self_choice='shadow',
                         self_choice_receipt=args['self_choice_receipt']),
            actor_observations=actor['steps'],actor_born_step=actor['born_step'],
            actor_self_weights_sha256=actor['receipt']['self_weights_sha256'],
            actor_perception_weights_sha256=actor['receipt']['perception_weights_sha256'],
            inner_execution_sha256=actor['receipt']['model_execution_sha256'],
            outer_binding_sha256=manifest['binding']['sha256'],outer_contract=wrapped['contract'],
            production_source_manifest=str(Path(manifest_path).resolve()),
            production_source_manifest_sha256=continuation_state.file_sha256(manifest_path),
            original_inner_execution_sha256=archived['actor_execution_sha256'],
            precision_accommodation=dict(original=archived['actor_execution_description']['dtype'],copied='torch.float32'),
            accommodation='CPU-only copy clears archived CUDA RNG fields, changes actor device and precision identity and isolates scene/status paths; original source and every native weight/Adam/runtime file remain exact')
        if (staging/'continuity_migration.json').exists():
            shutil.copy2(staging/'continuity_migration.json',staging/'source_continuity_migration.json')
            preserved['source_continuity_migration.json']=preserved.pop('continuity_migration.json')
            note['auxiliary_original_names']={'source_continuity_migration.json':'continuity_migration.json'}
        atomic_json(staging/'continuity_migration.json',cpu_receipt)
    atomic_json(staging/'decorated_execution_migration.json',note)
    # Publication occurs only after model, optimizer, runtime, episode and receipt
    # all exist. The source and its companion are never overwritten or removed.
    atomic_json(args_path,configuration)
    os.replace(staging,out)
    return note


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['record','prepare'])
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--episode-storage',type=Path)
    parser.add_argument('--cpu-probe',action='store_true',help='Explicit isolated CPU test accommodation; never a production migration')
    args=parser.parse_args()
    if args.operation=='record':result=record(args.checkpoint,args.out)
    else:
        if args.manifest is None:parser.error('prepare requires the previously recorded --manifest')
        result=prepare(args.checkpoint,args.out,args.manifest,args.episode_storage,cpu_probe=args.cpu_probe)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
