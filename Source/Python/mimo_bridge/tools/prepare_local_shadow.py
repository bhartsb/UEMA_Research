# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Prepare local sensing and an observing candidate without rewinding live learning.

The production trainer is never controlled. Every old tensor and Adam state is
verified, runtime/curriculum are copied exactly inside atomic publication, and
only the explicit contact boundary plus shadow configuration change the episode.
"""
import argparse
from collections import deque
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import torch
from safetensors import safe_open

from mimo_bridge import continuation_state
from mimo_bridge.virtual_skin_transition import activate_local_skin
from .prepare_continuity import native_model
from .caregiver_learning_trial import verify_loaded
from .migrate_trunk import modules
from .training_continuation import atomic_json,validate_checkpoint
from lem.training.checkpoint import CheckpointManager


def state_digest(value):
    """Exact value digest, including NaN payloads, without cloning large histories."""
    digest=hashlib.sha256()
    def visit(item):
        digest.update((type(item).__module__+'.'+type(item).__qualname__).encode())
        if type(item).__name__=='EventTape' and type(item).__module__=='mimo_bridge.event_tape':
            # Restore deliberately forks append-only recording writers. Their
            # UUID and storage-status timer are operational, not experience.
            visit(dict(root=item.root,spans=item.spans,count=item.count,dropped=item.dropped,
                       pending=item._pending,buffer=bytes(item._buffer)))
        elif torch.is_tensor(item):
            # A size-one zero-stride view is considered contiguous by PyTorch;
            # contiguous() alone can therefore leave an invalid dtype-view
            # stride. Flatten then force independent contiguous storage. This
            # also covers scalars, empty tensors and bfloat16 without rounding.
            array=item.detach().cpu().reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8).numpy()
            digest.update(str((tuple(item.shape),item.dtype)).encode());digest.update(memoryview(array))
        elif isinstance(item,np.ndarray):
            digest.update(str((item.shape,item.dtype)).encode())
            digest.update(memoryview(np.ascontiguousarray(item)).cast('B'))
        elif isinstance(item,dict):
            for key,child in item.items():visit(key);visit(child)
        elif isinstance(item,(list,tuple,deque)):
            digest.update(str(len(item)).encode())
            if isinstance(item,deque):digest.update(str(item.maxlen).encode())
            for child in item:visit(child)
        elif isinstance(item,(set,frozenset)):
            digest.update(str(len(item)).encode())
            for child in sorted(item,key=state_digest):visit(child)
        elif isinstance(item,slice):visit((item.start,item.stop,item.step))
        elif isinstance(item,(np.generic,)):
            digest.update(item.tobytes())
        elif isinstance(item,slice):
            visit((item.start,item.stop,item.step))
        elif item is None:pass
        elif isinstance(item,float):digest.update(struct.pack('!d',item))
        elif isinstance(item,(str,int,bool,bytes,torch.device,torch.dtype,Path)):
            digest.update(str(item).encode())
        elif isinstance(item,np.random.Generator):visit(item.bit_generator.state)
        elif hasattr(item,'__dict__'):visit(vars(item))
        else:raise TypeError('Unhandled preserved state: '+type(item).__name__)
    visit(value);return digest.hexdigest()


def verify_episode(old,new,changes):
    if len(old['worlds'])!=len(new['worlds']):raise ValueError('physical world count changed')
    if new['config']!=dict(old['config'],**changes):raise ValueError('unexpected continuation configuration change')
    view=dict(new);view.pop('virtual_skin_transition',None);view['config']=old['config']
    view['worlds']=[]
    for before,after in zip(old['worlds'],new['worlds']):
        if after['env_kwargs']!=dict(before['env_kwargs'],contact_profile='mechanical_v2_local'):
            raise ValueError('unexpected world configuration change')
        restored=dict(after,env_kwargs=before['env_kwargs'])
        attrs=dict(after['world']['attrs'])
        for key in ('_virtual_skin','_virtual_skin_transition'):
            if key in before['world']['attrs']:attrs[key]=before['world']['attrs'][key]
            else:attrs.pop(key,None)
        restored['world']=dict(after['world'],attrs=attrs)
        info=dict(after['payload']['info']);info.pop('contact_profile_transition',None)
        if 'contact_profile_transition' in before['payload']['info']:
            info['contact_profile_transition']=before['payload']['info']['contact_profile_transition']
        restored['payload']=dict(after['payload'],info=info);view['worlds'].append(restored)
    trainer=dict(new['trainer'])
    if 'frame_meta' in trainer:
        trainer['frame_meta']=[]
        for before,after in zip(old['trainer']['frame_meta'],new['trainer']['frame_meta']):
            info=dict(after);info.pop('contact_profile_transition',None)
            if 'contact_profile_transition' in before:info['contact_profile_transition']=before['contact_profile_transition']
            trainer['frame_meta'].append(info)
    view['trainer']=trainer
    expected=state_digest(old);actual=state_digest(view)
    if expected!=actual:raise ValueError('local skin migration changed existing episode evidence')
    return expected


def prepare(source,out,calibration,episode_storage,*,cpu_probe=False):
    source=Path(source).resolve();out=Path(out).resolve();calibration=Path(calibration).resolve()
    if out.exists():raise FileExistsError(out)
    validate_checkpoint(source)
    measured=json.loads(calibration.read_text(encoding='utf-8'))
    if measured.get('contact_profile')!='mechanical_v2_local' or measured.get('production_license') is not False:
        raise ValueError('requires a matching unqualified local-skin shadow calibration')
    changes=dict(contact_profile='mechanical_v2_local',self_choice='shadow',self_choice_receipt=str(calibration))
    model,args=native_model(source,changes)
    if args.get('contact_profile')!='mechanical_v2' or args.get('body_function')!='modulate' or args.get('self_choice','off')!='off':
        raise ValueError('source must be the current modulated v2 learner without an acting snapshot')
    loaded=model.load_checkpoint(str(source),load_optimizer=True)
    verify_loaded(model,loaded,source,True)
    if out.name!=f'ckpt_step_{model.step_count}':
        raise ValueError('target checkpoint name must preserve the actual source step')
    actual={f'{name}.{key}':value for name,module in modules(model).items() for key,value in module.state_dict().items()}
    with safe_open(str(source/'model.safetensors'),framework='pt',device='cpu') as saved:
        names=set(saved.keys())
        for name in names:
            if name not in actual or not torch.equal(saved.get_tensor(name),actual[name].cpu()):
                raise ValueError('old trained tensor changed: '+name)
    added=sorted(set(actual)-names)
    for name in added:
        if not name.startswith(('self_state.grounded.scope_encoder.','self_state.grounded.surface_encoder.')) or actual[name].count_nonzero():
            raise ValueError('unexpected or nonneutral new parameter: '+name)
    optimizer=torch.load(source/'optimizer.pt',map_location='cpu',weights_only=False)['optimizer']
    optimizer_count=len(optimizer['state']);del optimizer,actual;gc.collect()
    episode=continuation_state.load(source)
    if episode is None or episode['trainer'].get('acting_snapshot') is not None:
        raise ValueError('requires the complete current episode without an existing actor')
    migrated=activate_local_skin(episode);migrated['config'].update(changes)
    episode_hash=verify_episode(episode,migrated,changes)
    source_times=[world['payload']['sim_time'] for world in episode['worlds']]
    cpu_accommodation=None
    if cpu_probe:
        # A bounded CPU startup check cannot consume a CUDA RNG. The production
        # migration never takes this branch and keeps every RNG byte unchanged.
        cpu_accommodation=state_digest(migrated['learner']['cuda_rng'])
        migrated['learner']['cuda_rng']=None
        for i,world in enumerate(migrated['worlds']):
            world['env_kwargs']['run_dir']=str(out.parent.parent/'scene_copies'/str(i))
            caregiver=world['world'].get('caregiver')
            if caregiver is not None and hasattr(caregiver.get('events'),'status_path'):
                caregiver['events'].status_path=out.parent.parent/f'copied_recording_status_{i}.json'
    del episode;gc.collect()
    model.checkpoints=CheckpointManager(model.config,str(out.parent))
    model.receipt['episode_continuation']=continuation_state.VERSION
    model.receipt['runtime_state_required']=True
    preserved={name:continuation_state.file_sha256(source/name) for name in ('runtime_state.pt','meta.json','lineage.json') if (source/name).is_file()}
    def companions(staging,final):
        for name,digest in preserved.items():
            shutil.copyfile(source/name,staging/name)
            if continuation_state.file_sha256(staging/name)!=digest:raise ValueError('native runtime copy changed '+name)
        continuation_state.write_episode(staging,migrated,episode_storage,published_path=final)
    # Runtime is intentionally copied byte-for-byte. Avoid first serializing a
    # second large, reconstructed runtime only to overwrite it immediately.
    created=Path(model.checkpoints.save(model.step_count,model.encoder_registry,model.forward_model,
        model.prediction_heads,model.motor_controller,model.wellness,model.optimizer,model.receipt,
        gaze_head=model.gaze_head,voice_head=model.voice_head,side_modules=model._side_modules(),
        companion_writer=companions)).resolve()
    if created!=out:raise ValueError('target checkpoint name must preserve the actual source step')
    configuration=dict(args,**changes,out=str(out.parent.parent),resume=str(out),checkpoint_storage=str(out.parent))
    if cpu_probe:configuration['device']='cpu'
    atomic_json(out.parent.parent/'args.json',configuration)
    note=dict(version='local_skin_shadow_activation_v1',parent=str(source),checkpoint=str(out),step=model.step_count,
        old_model_tensors_exact=len(names),old_optimizer_states_exact=optimizer_count,new_neutral_tensors=added,
        old_runtime_files_sha256=preserved,old_episode_evidence_sha256=episode_hash,
        old_episode_exact_except_declared_boundary=True,world_count=len(migrated['worlds']),sim_times=source_times,
        changes=changes,calibration_sha256=continuation_state.file_sha256(calibration),
        cpu_probe_only=cpu_probe,cpu_probe_original_cuda_rng_sha256=cpu_accommodation,
        recording_continuation='Append-only event prefixes/counts exact; independent UUID writers fork without modifying the saved prefix.',
        production_choice_license=False,scope='Local sensing and observing candidate; original learner still chooses every physical command.')
    atomic_json(out/'continuity_migration.json',note)
    return note


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True);parser.add_argument('--episode-storage',type=Path,required=True)
    parser.add_argument('--cpu-probe',action='store_true',help='Only for a separate copied CPU startup test; never production')
    args=parser.parse_args();torch.set_num_threads(1)
    print(json.dumps(prepare(args.checkpoint,args.out,args.calibration,args.episode_storage,cpu_probe=args.cpu_probe),indent=2))


if __name__=='__main__':main()
