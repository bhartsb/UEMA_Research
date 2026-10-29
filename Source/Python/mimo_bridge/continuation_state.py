# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Checkpoint the physical episode and its causal learner state at a loop boundary.

Weights and Adam remain in the existing checkpoint format. This companion retains
the world, experienced history, held orders and random streams; it never attaches
an old episode's memory to a newly reset body. Only trusted local checkpoints load.
"""
from collections import deque
import copy
from dataclasses import is_dataclass, fields
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch

VERSION = 'episode_continuation_v1'
DATA_NAMES = set('context_buffer goal_ring token_ring enc_history enc_history_raw valid_history '
                 'motor_history ctx_history ach_history memory_bias_history'.split())
OBJECT_DATA = ('replay','movement','curiosity','consequence_lesson')
# The collapse-watch writer in LemLite.step appends CPU float32 tokens (body 0's, and
# every body's in _token_ring_bodies). Its history must stay there after a CUDA resume
# too. Context, goal and learning histories, in contrast, are consumed and extended on
# the model's device.
CPU_DATA_NAMES = frozenset({'token_ring', '_token_ring_bodies'})
# Hand-off from the checkpoint's runtime state to the first step, NOT episode state. The
# checkpoint load writes these, and the self-model's tracker (built at the first step, when
# the body count is known) reads them and clears them. The episode companion is restored
# AFTER the checkpoint load, so carrying them would put a spent value back over the one the
# load just wrote: measured 2026-09-20 on the segment-10 resume, the companion's own
# _entity_self_saved of None landed on top of the 34,950 moments of watching the checkpoint
# held, the tracker was built empty, and the run's 2 lived windows read 0 with no warning.
HANDOFF_NAMES = frozenset({'_entity_self_saved', '_entity_self_saved_names'})
REPORT_WINDOW_NAMES = frozenset({'_syn_window', '_syn_z_window', '_syn_live', '_syn_live_regimes', '_syn_following'})

# long_path lives beside the move resolver, which needs it without importing torch; the name
# stays importable from here for every existing caller.
from .storage_locations import long_path, locate  # noqa: E402

# ENV SETTINGS A RESUME MAY CHANGE ON A SAVED EPISODE. A resumed episode rebuilds each world from
# the env_kwargs saved with it (train_mimo builds SingleEnvSource / PoolSource from
# saved_worlds[i]['env_kwargs']), so a setting changed on the resume command never reached the
# worlds. Measured 2026-09-22: segment 31 was launched with --toy-hand-off follow and all 161 of
# its stats records read object_follows_grip false with 0 grips.
# THE MIGRATION IS OFF UNLESS ASKED FOR (2026-09-23, review ops-storage finding 1). DESIGN D2: a
# supervisor resume of the old command on new code changes nothing. Segment 31's own command
# carries --toy-hand-off follow while its worlds ran with the hand-off off, so migrating by default
# switched the hand-off on for 8 of 8 worlds on a plain resume of that command. migrate_env_kwargs
# therefore keeps the saved worlds' settings (env_settings='saved', the default) and only names
# every launch setting that differs; the deploy asks for the launch's values with
# env_settings='launch' (train_mimo flag --resume-env-settings launch). Only settings that leave
# the saved physical state's shape unchanged may migrate, each with the reason:
#   toy_hand_off     read only by Caregiver.__init__; Caregiver.restore replaces caregiver.state,
#                    whose 'grip' entry exists under both values (_grip_step creates it before it
#                    reads the setting).
#   finger_armature  added to model.dof_armature in BridgeEnv.__init__; restore_full_state's
#                    mutable_geometry does not carry dof_armature, so the constructor's value
#                    stands after a restore. Without this entry a return to 0.02 at a resume
#                    would be ignored the same way.
# Hand mode changes command interpretation, not the physical state dimensions.
# The trainer versions new experience and retains old observations separately.
#   parent_relative_targets, reach_projection  the motor interface audit's calf and arm fixes
#                    (limb_target_control.py): they change how an order is interpreted and
#                    recorded, not the physical state; a world saved without them carries no
#                    _limb_target_cache and projects its next order afresh (env.restore_full_state).
#                    The calf's contract is versioned in replay like the hand mode's.
#   position_rows    which position rows the decoder solves (action_decoder.position_handles): read by
#                    ActionDecoder.__init__ only, no physical state; versioned in replay the same way.
#   contact_footprint_guard  read only by BridgeEnv.__init__ into env attributes that
#                    save_full_state does not carry; the restored virtual skin keeps every per-taxel
#                    memory in its saved shape and continues it under the rule the world is built
#                    with (contact_footprint.FootprintGuard; 'off' = the v1 footprints exactly).
#   rubbing_calibration  read by BridgeEnv.__init__; after a restore the env gives the saved skin the
#                    second memory its setting asks for (a saved one is kept, a missing one starts as a
#                    copy of the saved memory, 'v4' drops it); today's memory keeps its saved shape.
#   touch_inside_rays  read by BridgeEnv.__init__ while MIMo's touch is built (touch_inside_rays.py). MIMo's
#                    touch is rebuilt at every launch and never saved, so under 'process' every resume already
#                    re-draws which seam skin points are live; the restored virtual skin keeps its saved taxels.
#   finger_joint_servo  how the decoder servos the finger chains (action_decoder.py, THE FINGER COMMAND'S MEANING):
#                    read by BridgeEnv.__init__ into an env attribute save_full_state does not carry, no physical
#                    state; versioned in replay like the hand mode (limb_target_control.FINGER_SERVO_CONTRACT_BIT).
ENV_KWARGS_MIGRATABLE = ('toy_hand_off', 'finger_armature', 'hand_target_mode',
                         'parent_relative_targets', 'reach_projection', 'position_rows',
                         'contact_footprint_guard', 'rubbing_calibration', 'touch_inside_rays',
                         'finger_joint_servo')
# Settings added after episodes were already saved without them. A saved world that lacks one ran on
# BridgeEnv's default, so a launch asking for that same default changes nothing and is not written
# into the world: an episode saved on the default then still resumes on code that predates the
# setting (train_mimo's env_kwargs_for leaves it out at its default for the same reason). The same
# rule as the servo authority branch's (fix/servo-authority-2026-09-25) and the skin branch's
# (fix/skin-footprint-2026-09-25); merged, the set holds every branch's names.
_ABSENT_IS_DEFAULT = frozenset({'parent_relative_targets', 'reach_projection', 'position_rows',
                                'contact_footprint_guard', 'rubbing_calibration', 'touch_inside_rays',
                                'finger_joint_servo'})
# the two answers to "whose env settings does a resumed world run on": 'saved' (the episode's own,
# DESIGN D2's default) or 'launch' (the resume command's, for the names above)
ENV_SETTINGS_CHOICES = ('saved', 'launch')
# settings that differ between segments by design and are never a changed flag: the saved
# world keeps its own random stream, name and scene folder
_PER_SEGMENT_KWARGS = frozenset({'seed', 'tag', 'run_dir'})


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def tree(value, device='cpu'):
    if torch.is_tensor(value): return value.detach().to(device).clone()
    if isinstance(value,np.ndarray): return value.copy()
    if isinstance(value,dict): return {k:tree(v,device) for k,v in value.items()}
    if isinstance(value,deque):return deque((tree(v,device) for v in value),maxlen=value.maxlen)
    if isinstance(value,list):return [tree(v,device) for v in value]
    if isinstance(value,tuple):return tuple(tree(v,device) for v in value)
    if is_dataclass(value):
        result=copy.copy(value)
        for f in fields(value):object.__setattr__(result,f.name,tree(getattr(value,f.name),device))
        return result
    if value is None or isinstance(value,(str,int,float,bool,np.generic,torch.device)):return copy.deepcopy(value)
    raise TypeError(f'continuation state needs an explicit serializer for {type(value).__name__}')


def tree_ref(value):
    """tree() without copying what a save only reads (2026-10-01, the save-memory repair).

    The same containers are rebuilt and the same types refused, but a dense tensor is passed on as itself
    (detached) and an array as itself, so a save that writes them straight into the companion holds no second
    copy of the learner. A tensor that is a view into a larger storage, or is not dense, is copied exactly as
    tree() copies it, so the companion never carries more than the tensor and loads as tree() would give it. A
    CUDA tensor stays on its device here; the companion writer copies each one to the CPU only for its own write
    (episode_archive) and records it as a CPU tensor, as tree() did. Nothing here is a snapshot: the caller
    finishes writing before the model runs again."""
    if torch.is_tensor(value):
        value=value.detach()
        if (value.layout==torch.strided and value.storage_offset()==0 and value.is_contiguous()
                and value.untyped_storage().nbytes()==value.numel()*value.element_size()):
            return value
        return value.to('cpu').clone()
    # an array in C order is passed on; any other layout is copied as tree() copies it (C order), so the file
    # loads with tree()'s layout
    if isinstance(value,np.ndarray): return value if value.flags.c_contiguous else value.copy()
    if isinstance(value,dict): return {k:tree_ref(v) for k,v in value.items()}
    if isinstance(value,deque):return deque((tree_ref(v) for v in value),maxlen=value.maxlen)
    if isinstance(value,list):return [tree_ref(v) for v in value]
    if isinstance(value,tuple):return tuple(tree_ref(v) for v in value)
    if is_dataclass(value):
        result=copy.copy(value)
        for f in fields(value):object.__setattr__(result,f.name,tree_ref(getattr(value,f.name)))
        return result
    if value is None or isinstance(value,(str,int,float,bool,np.generic,torch.device)):return copy.deepcopy(value)
    raise TypeError(f'continuation state needs an explicit serializer for {type(value).__name__}')


def capture_learner(model, *, include_replay=True, copy_state=True):
    """The learner's continuation state. copy_state=False passes the model's own tensors and arrays on (tree_ref),
    for save(), which writes them before the model next runs; every other caller keeps copies."""
    tree_of=tree if copy_state else tree_ref
    data={}
    for name,value in vars(model).items():
        # learned state already carried in runtime_state.pt, by the model's own writer
        if name in ('_hself','_schemas','_entity_self','_self_value'):continue
        if name in HANDOFF_NAMES:continue
        if name in REPORT_WINDOW_NAMES:continue
        if name.startswith(('_','last_')) or name in DATA_NAMES:
            data[name]=tree_of(value)
    objects={}
    for name in OBJECT_DATA:
        if name=='replay' and not include_replay:continue
        value=getattr(model,name,None)
        if value is not None:objects[name]=tree_of(vars(value))
    return dict(data=data,objects=objects,
        memory=None if model.memory_history is None else model.memory_history.state_dict(),
        self_history=None if model.self_history is None else model.self_history.state_dict(),
        self_rehearsal=None if model.self_rehearsal is None else tree_of(model.self_rehearsal.state_dict()),
        loss_history=tree_of(model.loss_manager._vicreg_history),
        placeholders={k:tree_of(v._buf) for k,v in model.placeholders._generators.items()},
        noise_rng=model.noise_rng.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if model.device.type=='cuda' else None,
        numpy_rng=np.random.get_state(),python_rng=random.getstate())


def restore_learner(model,state):
    for k,v in state['data'].items():
        # companions written before 2026-09-20 carry the hand-off names; skipping them on
        # the way in as well as on the way out keeps an older companion from clearing them
        if k in HANDOFF_NAMES:continue
        if k in REPORT_WINDOW_NAMES:continue
        setattr(model,k,tree(v,'cpu' if k in CPU_DATA_NAMES else model.device))
    for name,data in state['objects'].items():
        obj=getattr(model,name,None)
        if obj is None:raise ValueError(f'continuation requires saved {name}')
        # Replay is deliberately kept on CPU, including when training on CUDA.
        restored=tree(data,'cpu' if name=='replay' else model.device)
        if 'device' in restored:restored['device']=model.device
        obj.__dict__.update(restored)
    if state['memory'] is not None:
        if model.memory_history is None:raise ValueError('continuation requires persistent memory')
        model.memory_history.load_state_dict(state['memory'],model.device)
    if state['self_history'] is not None:
        if model.self_history is None:raise ValueError('continuation requires self perception')
        model.self_history.load_state_dict(state['self_history'],model.device)
    if state.get('self_rehearsal') is not None:
        model.self_rehearsal.load_state_dict(state['self_rehearsal'])
    model.loss_manager._vicreg_history=tree(state['loss_history'],model.device)
    for k,v in state['placeholders'].items():
        model.placeholders._generators[k]._buf=tree(v,model.device)
    model.noise_rng.set_state(state['noise_rng'].cpu())
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng'] is not None:
        if model.device.type!='cuda':raise ValueError('exact continuation requires the saved execution device')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])


def world_blobs(source):
    """(header, pickle bytes) for each world in turn, header = dict(sim_time, nbytes).

    A source with iter_world_blobs (PoolSource: each worker pickles its own world when asked, one worker at a time;
    SingleEnvSource) is asked for them; any other source's save_full_state() list is pickled here, one world at a
    time, as the worker would."""
    from .episode_archive import world_blob
    stream=getattr(source,'iter_world_blobs',None)
    if stream is not None:
        yield from stream()
        return
    worlds=source.save_full_state()
    while worlds:
        world=worlds.pop(0)
        blob=world_blob(world)
        header=dict(sim_time=world['payload']['sim_time'],nbytes=len(blob))
        del world
        yield header,blob
        del blob


def save(path,source,model,trainer_state,config,storage_root=None,published_path=None):
    """Write the physical episode and the learner beside a checkpoint, holding one world at a time.

    Until 2026-10-01 this built the whole state first: every world requested at once (each worker holding a copy
    of its world and that copy's pickle until the trainer read it), all of them unpickled together in the trainer
    (at step 447,350 the eight caregiver event logs alone were 86.5 million Python objects), the learner cloned,
    then a space estimate walked every one of those objects before torch.save pickled them again into memory.
    Now each world goes from its worker straight into the archive as bytes, the learner's tensors and arrays are
    written from the model itself, and torch.load reads the same file as before (episode_archive). Tmp-then-rename
    and the receipt are unchanged; a failed write leaves no temporary file."""
    from .episode_archive import EpisodeArchive
    from lem.training.checkpoint import require_save_bytes,require_save_space
    path=Path(path)
    target=_companion_target(path,storage_root,published_path)
    temp=target.with_suffix(target.suffix+'.tmp')
    directory=long_path(target.parent)
    times=[]
    archive=None
    try:
        with gzip.open(long_path(temp),'wb',compresslevel=1) as f:
            archive=EpisodeArchive(f)
            worlds=[]
            for header,blob in world_blobs(source):
                if not worlds:
                    # one space check for every world, sized from the first (the old save made one check too;
                    # retention's check globs every checkpoint, so it is not repeated per world)
                    require_save_bytes(directory,len(blob)*max(1,int(getattr(source,'n_envs',1) or 1)))
                worlds.append(archive.put_world(blob))
                times.append(header['sim_time'])
                del blob
            learner=capture_learner(model,copy_state=False)
            trainer=tree(trainer_state)
            require_save_space(directory,dict(learner=learner,trainer=trainer,config=config))
            archive.finish(dict(version=VERSION,worlds=worlds,learner=learner,trainer=trainer,config=config))
            del learner
        os.replace(long_path(temp),long_path(target))
    except BaseException:
        if archive is not None:archive.abandon()
        try:
            os.remove(long_path(temp))
        except OSError:
            pass
        raise
    return _publish_receipt(path,target,trainer['step'],times,storage_root,published_path)


def _companion_target(path,storage_root,published_path):
    if storage_root:
        folder=Path(storage_root).resolve();Path(long_path(folder)).mkdir(parents=True,exist_ok=True)
        import uuid
        return folder/((Path(published_path).name if published_path else Path(path).name)+'_'+uuid.uuid4().hex+'.pt.gz')
    return Path(path)/'episode_state.pt.gz'


def _publish_receipt(path,target,step,times,storage_root,published_path):
    path=Path(path)
    located_target=(Path(published_path)/target.name) if published_path and not storage_root else target
    receipt=dict(version=VERSION,sha256=file_sha256(long_path(target)),
                 bytes=os.path.getsize(long_path(target)),step=step,
                 storage_path=str(located_target.resolve()),free_bytes=shutil.disk_usage(long_path(target.parent)).free,
                 times=times)
    receipt_path=path/'episode_state.json';receipt_temp=path/'episode_state.json.tmp'
    receipt_temp.write_text(json.dumps(receipt,indent=2),encoding='utf-8');receipt_temp.replace(receipt_path)
    return receipt


def write_episode(path,state,storage_root=None,published_path=None):
    """Write a new companion and receipt from a state already in memory (the migration tools); never reuse a
    parent's storage pointer. The trainer's own save streams instead (save)."""
    path=Path(path)
    target=_companion_target(path,storage_root,published_path)
    from lem.training.checkpoint import require_save_space
    require_save_space(long_path(target.parent),state)
    temp=target.with_suffix(target.suffix+'.tmp')
    with gzip.open(long_path(temp),'wb',compresslevel=1) as f:torch.save(state,f)
    os.replace(long_path(temp),long_path(target))
    return _publish_receipt(path,target,state['trainer']['step'],[w['payload']['sim_time'] for w in state['worlds']],
                            storage_root,published_path)


def add_observed_self(state):
    """Neutral off -> observe/learn upgrade of an already saved physical episode.

    The newly observed function is truly intact: the old world did not modulate
    sensing. Start the hypothetical impairment history now, without recollecting
    the observation, advancing time or changing old learner/replay/RNG evidence.
    """
    from lem.body_function import BodyFunction,physical_drives
    upgraded=copy.deepcopy(state)
    config=upgraded['config']
    if config.get('self_state','off')!='off' or config.get('body_function','off')!='off':
        raise ValueError('neutral episode upgrade requires self and body function off')
    config.update(self_state='learn',body_function='observe')
    for i,saved in enumerate(upgraded['worlds']):
        if saved['env_kwargs'].get('body_function','off')!='off':
            raise ValueError('episode body-function configuration disagrees with world')
        attrs=saved['world']['attrs'];payload=saved['payload']
        if attrs.get('body_function') is not None:
            raise ValueError('neutral upgrade cannot discard existing physical function')
        pooler=attrs.get('_mechanical_pooler')
        exposure=payload['info'].get('mechanical_contact',{}).get('history')
        sensitivity=payload.get('touch_sensitivity')
        if pooler is None or exposure is None or sensitivity is None:
            raise ValueError('episode upgrade requires located physical skin identity and history')
        function=BodyFunction([spec.body_name for spec in pooler.specs])
        load,stim=physical_drives(payload['patches'][:,0],exposure,sensitivity)
        function.advance(load,stim,float(payload['sim_time']))
        record=function.record();record['mode']='observe'
        record['shadow_values']=copy.deepcopy(record['values'])
        attrs['body_function']=function
        saved['env_kwargs']['body_function']='observe'
        payload['info']['body_function']=record
        # The trainer can have added metadata since the source last emitted it.
        # Add only the new observation, preserving those existing fields too.
        if 'frame_meta' in upgraded['trainer']:
            upgraded['trainer']['frame_meta'][i]['body_function']=copy.deepcopy(record)
    return upgraded


def extend_caregiver_speech(state, directory):
    """Add dialogue to a running legacy adult without changing saved experience.

    Only the speech-library path changes. The existing PCM, physical state,
    histories and learner are retained. New controller state begins on its next
    tick. This does not permit removing/replacing already available utterances.
    """
    from .speech import SpeechBank
    config=state['config'];old_path=config.get('caregiver_speech_dir')
    if config.get('caregiver_mode')!='responsive' or not old_path:
        raise ValueError('speech extension requires an existing responsive adult')
    old=SpeechBank(old_path);new=SpeechBank(directory)
    if old.groups or not new.groups:
        raise ValueError('speech extension requires legacy-to-responsive-exchange libraries')
    for key,clip in old.clips.items():
        if key not in new.clips or old.texts[key]!=new.texts[key] or not np.array_equal(clip,new.clips[key]):
            raise ValueError(f'speech extension changed an existing utterance: {key}')
    # Shallow structural copies are intentional: large immutable payloads and
    # learned tensors are shared, never edited or needlessly duplicated in RAM.
    upgraded=dict(state,config=dict(config,caregiver_speech_dir=str(directory)),worlds=[])
    for world in state['worlds']:
        kw=world['env_kwargs']
        if kw.get('caregiver_speech_dir')!=old_path or kw.get('caregiver_mode')!='responsive':
            raise ValueError('saved world disagrees with the speech migration source')
        if world['world'].get('caregiver') is None:
            raise ValueError('cannot extend absent caregiver state')
        upgraded['worlds'].append(dict(world,env_kwargs=dict(kw,caregiver_speech_dir=str(directory))))
    return upgraded


def _check_env_value(name, value):
    if name == 'hand_target_mode' and value not in ('world', 'wrist_relative_v1'):
        raise ValueError('Unknown hand-target contract')
    if name in ('parent_relative_targets', 'reach_projection'):
        from .limb_target_control import PARENT_RELATIVE_CHOICES, REACH_PROJECTION_CHOICES
        allowed = PARENT_RELATIVE_CHOICES if name == 'parent_relative_targets' else REACH_PROJECTION_CHOICES
        if value not in allowed:
            raise ValueError(f'{name} must be one of {allowed}, not {value!r}')
    if name == 'position_rows':
        from .action_decoder import POSITION_ROW_CHOICES
        if value not in POSITION_ROW_CHOICES:
            raise ValueError(f'position_rows must be one of {POSITION_ROW_CHOICES}, not {value!r}')
    if name == 'toy_hand_off' and value not in ('off', 'follow'):
        raise ValueError(f"toy_hand_off must be 'off' or 'follow', not {value!r}")
    if name == 'finger_armature':
        v = float(value)
        if not np.isfinite(v) or v < 0:
            raise ValueError(f'finger_armature must be a finite value of 0 or more, not {value!r}')
    if name == 'contact_footprint_guard':
        from .contact_footprint import FOOTPRINT_GUARDS
        if value not in FOOTPRINT_GUARDS:
            raise ValueError('contact_footprint_guard must be one of %s, not %r'
                             % (', '.join(FOOTPRINT_GUARDS), value))
    if name == 'rubbing_calibration':
        from lem.skin_mechanics import RUBBING_CALIBRATIONS
        if value not in RUBBING_CALIBRATIONS:
            raise ValueError('rubbing_calibration must be one of %s, not %r'
                             % (', '.join(RUBBING_CALIBRATIONS), value))
    if name == 'touch_inside_rays':
        from .touch_inside_rays import CHOICES
        if value not in CHOICES:
            raise ValueError('touch_inside_rays must be one of %s, not %r' % (', '.join(CHOICES), value))
    if name == 'finger_joint_servo':
        from .action_decoder import FINGER_JOINT_SERVO_CHOICES
        if value not in FINGER_JOINT_SERVO_CHOICES:
            raise ValueError('finger_joint_servo must be one of %s, not %r'
                             % (', '.join(FINGER_JOINT_SERVO_CHOICES), value))


def _released_grip(world):
    """The saved caregiver's grip, let go at the resume. Switching the hand-off OFF while the
    infant holds the toy would otherwise leave the grip reading held forever (the 'off' path of
    _grip_step returns before it can release), and the world report would claim a held toy the
    adult is carrying again. The release is the one the caregiver itself makes after
    RELEASE_STEPS without contact: held False and the adult's hand blending back."""
    saved = world['world'].get('caregiver') if isinstance(world.get('world'), dict) else None
    grip = (saved or {}).get('grip')
    if not grip or not grip.get('held'):
        return None
    from .caregiver import Caregiver
    released = dict(grip, held=False, hold_steps=0, local_pos=None, local_rot=None,
                    blend=Caregiver.HAND_BLEND_STEPS, releases=int(grip.get('releases', 0)) + 1)
    return dict(world, world=dict(world['world'], caregiver=dict(saved, grip=released)))


def _bridge_env_defaults():
    """{keyword: default} of BridgeEnv's constructor; empty when the env cannot be imported."""
    import inspect
    try:
        from .env import BridgeEnv
    except ImportError:
        return {}
    return {n: p.default for n, p in inspect.signature(BridgeEnv.__init__).parameters.items()
            if p.default is not inspect.Parameter.empty}


def migrate_env_kwargs(state, args, names=ENV_KWARGS_MIGRATABLE, launch_kwargs=None, log=None,
                       env_settings='saved'):
    """Name every env setting a resume launches with that differs from the saved worlds; with
    env_settings='launch', also rewrite the ones in `names` to the launch's values.

    env_settings: 'saved' (the default, DESIGN D2) keeps every saved world's settings exactly,
        so a resume of an old command on new code changes no world; the settings in `names` that
        differ are reported as NOT applied, beside the others. 'launch' rewrites the settings in
        `names` to the launch's values (the deploy's choice, train_mimo --resume-env-settings).
    args: the launch's argparse namespace (or a dict); a name it does not carry is left as saved.
    launch_kwargs: the env kwargs the launch would build a fresh world with (env_kwargs_for); when
        given, every OTHER setting that differs from a saved world is reported as NOT applied, so
        a changed flag outside the list is never silently ignored again.
    Returns (upgraded_state, changes). changes lists dict(world, name, old, new) for the settings
    actually rewritten (always empty under 'saved'); old is None when the saved world did not
    carry the setting (BridgeEnv's default was in force). Like extend_caregiver_speech, the
    copies are shallow: the physical state, learner and histories are shared, never edited. The
    upgraded state's config is the saved one, unchanged."""
    import logging
    logger = logging.getLogger('mimo_bridge.continuation_state')
    log = log or logger.info
    if env_settings not in ENV_SETTINGS_CHOICES:
        raise ValueError("env_settings must be one of %s, not %r"
                         % (', '.join(ENV_SETTINGS_CHOICES), env_settings))
    apply = env_settings == 'launch'
    wanted = {}
    for name in names:
        if isinstance(args, dict):
            if name in args:
                wanted[name] = args[name]
        elif hasattr(args, name):
            wanted[name] = getattr(args, name)
    for name, value in wanted.items():
        _check_env_value(name, value)
    defaults = _bridge_env_defaults() if any(n in _ABSENT_IS_DEFAULT for n in wanted) else {}
    changes, worlds = [], []
    for i, world in enumerate(state['worlds']):
        kw = world['env_kwargs']
        new_kw = dict(kw)
        for name, value in wanted.items():
            if name in kw and kw[name] == value:
                continue
            if (name not in kw and name in _ABSENT_IS_DEFAULT and name in defaults
                    and defaults[name] == value):
                continue            # the saved world already ran on this default
            new_kw[name] = value
            changes.append(dict(world=i, name=name, old=kw.get(name), new=value))
            if apply and name == 'toy_hand_off' and value == 'off':
                world = _released_grip(world) or world
        worlds.append(dict(world, env_kwargs=new_kw) if new_kw != kw else world)
    if not apply:
        # nothing is rewritten; what the launch asked for that the worlds do not run is named
        held = changes
        changes = []
        upgraded = state
        for name in sorted({c['name'] for c in held}):
            rows = [c for c in held if c['name'] == name]
            olds = sorted({repr(c['old']) for c in rows})
            logger.warning(
                'Resume: env setting %s is %r on the launch command and %s in %d of %d saved worlds '
                '(None = the world ran on BridgeEnv\'s default); NOT applied: the resumed worlds keep '
                'their saved settings (DESIGN D2). Pass --resume-env-settings launch to apply it.'
                % (name, rows[0]['new'], ', '.join(olds), len(rows), len(state['worlds'])))
    else:
        upgraded = dict(state, worlds=worlds) if changes else state
        for name in sorted({c['name'] for c in changes}):
            rows = [c for c in changes if c['name'] == name]
            olds = sorted({repr(c['old']) for c in rows})
            log('Resume: env setting %s set to %r on %d of %d saved worlds (saved value %s; None = '
                'the world ran on BridgeEnv\'s default); physical state and learner preserved'
                % (name, rows[0]['new'], len(rows), len(state['worlds']), ', '.join(olds)))
    if launch_kwargs is not None:
        judged = {k: v for k, v in launch_kwargs.items()
                  if k not in _PER_SEGMENT_KWARGS and k not in names}
        ignored = {k for w in upgraded['worlds'] for k, v in judged.items()
                   if k in w['env_kwargs'] and w['env_kwargs'][k] != v}
        # a setting the saved world never carried ran on BridgeEnv's default: a launch value
        # that differs from that default is ignored just the same
        absent = {k for w in upgraded['worlds'] for k in judged if k not in w['env_kwargs']}
        if absent:
            defaults = _bridge_env_defaults()
            ignored |= {k for k in absent if k in defaults and defaults[k] != judged[k]}
        ignored = sorted(ignored)
        if ignored:
            logger.warning(
                'Resume: these env settings differ from the saved episode and are NOT applied '
                '(a resumed world keeps its saved settings; only %s can migrate, and only with '
                '--resume-env-settings launch): %s'
                % (', '.join(names), ', '.join(ignored)))
    return upgraded, changes


def load(path):
    if not path:return None
    # THIS function also accepts a moved checkpoint's old C: path (an existing path is unchanged).
    # Its callers do not all: train_mimo calls model.load_checkpoint(args.resume) BEFORE this and
    # that load does not consult the move records, so a trainer must be given the checkpoint's D:
    # path (review ops-storage finding 11, 2026-09-23). The supervisor resolves a moved --resume
    # to its D: path before it validates and relaunches (training_continuation.resolved).
    path=locate(Path(path),verify=False);manifest=path/'episode_state.json'
    if not manifest.exists():
        model_receipt=path/'model_receipt.json'
        if model_receipt.exists() and json.loads(model_receipt.read_text(encoding='utf-8')).get('episode_continuation'):
            raise ValueError('checkpoint requires a completed physical episode companion')
        if (path/'episode_state.pt.gz').exists():raise ValueError('episode snapshot has no completed integrity receipt')
        return None
    receipt=json.loads(manifest.read_text(encoding='utf-8'))
    target=Path(receipt.get('storage_path',str(path/'episode_state.pt.gz')))
    # AN ARCHIVED CHECKPOINT'S RECEIPT NAMES THE C: COMPANION THE ARCHIVER REMOVED (checked
    # 2026-09-22 on D:/.../seg31_toy_and_push/checkpoints/ckpt_step_340526). The move manifests
    # say where it went and must record this receipt's hash; the hash check just below then
    # proves the file itself, so locate does not read the 600 MB twice.
    target=locate(target,expected_sha256=receipt.get('sha256'),verify=False)
    if receipt['version']!=VERSION or file_sha256(long_path(target))!=receipt['sha256']:
        raise ValueError('episode continuation integrity check failed')
    with gzip.open(long_path(target),'rb') as f:state=torch.load(f,map_location='cpu',weights_only=False)
    if state['version']!=VERSION or state['trainer']['step']!=receipt['step']:
        raise ValueError('episode continuation step/version mismatch')
    return state
