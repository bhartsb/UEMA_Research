# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Bounded copied assay of matched-encoder body and spatial-touch forecasts.

No production control, model update, reward, qualification or promotion occurs.
The new auxiliary head owns a separate optimizer and predicts fixed experienced
channels. A held-command replay measures deterministic physical repeatability.
"""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace,MethodType

import numpy as np
import torch

from lem.model.causal_readout import CausalReadout, body_shuffle
from mimo_bridge.causal_packets import (CONTRACT, sensory_payload, fixed_targets,
    packed_target, target_layout, write_packet, load_packet, encode_own_window,
    tensor_digest)
from mimo_bridge.tools.training_continuation import atomic_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def status(out, stage, **fields):
    row = dict(stage=stage, timestamp=time.time(), **fields)
    atomic_json(Path(out)/'status.json', row); print(json.dumps(row), flush=True)


def source_model(checkpoint, out):
    from mimo_bridge.train_mimo import parse_args, build_learner
    from mimo_bridge.tools.caregiver_learning_trial import verify_loaded
    from mimo_bridge.tools.causal_test import _limits_for_run
    args = parse_args(['--out', str(out)])
    vars(args).update(json.loads((checkpoint.parent.parent/'args.json').read_text(encoding='utf-8')))
    args.device = 'cpu'; args.out = str(out); args.n_envs = 1; args.self_choice = 'off'
    args.caregiver_record_dir = ''; args.gaze_reflex = 'off'; args.spontaneous_saccades = 'off'
    model = build_learner(args, str(out/'unused_checkpoints'))
    loaded = model.load_checkpoint(str(checkpoint), load_optimizer=True)
    verify_loaded(model, loaded, checkpoint, True)
    dev, basis = _limits_for_run(str(checkpoint), model)
    from mimo_bridge.body_layout import extend_limits
    dev = extend_limits(dev, 26 if args.body_layout == 'trunk_v1' else 24)
    return model, args, dev, basis


def old_state_digest(model):
    from mimo_bridge.tools.migrate_trunk import modules
    return dict(model=tensor_digest({k: m.state_dict() for k, m in modules(model).items()}),
                optimizer=tensor_digest(model.optimizer.state_dict()))


def commands_for(frame, previous, limits, smoothing, phase, seed):
    """Unlabelled opposing candidate body orders within native physical limits."""
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.tools.self_recovery_experiment import hold
    rng = np.random.default_rng(seed)
    base = hold(frame, min(.3, float(limits['strength_limit'])))
    raw = np.repeat(base, 4, 0); raw[:, :, :3] /= 200
    raw[1] = previous[0]; raw[1, :, :3] /= 200
    delta = rng.normal(size=(len(frame.bones), 3))
    delta /= np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-8)
    raw[2, :, :3] += .04*delta; raw[3, :, :3] -= .04*delta
    q = rng.normal(0, .10, (len(frame.bones), 4))
    raw[2, :, 3:7] += q; raw[3, :, 3:7] -= q
    raw[:, :, 3:7] /= np.maximum(np.linalg.norm(raw[:, :, 3:7], axis=-1, keepdims=True), 1e-8)
    achieved = torch.from_numpy(base.copy()); achieved[:, :, :3] /= 200
    new = LemLite.planned_execution(torch.from_numpy(raw), limits, achieved.expand(4, -1, -1),
        previous=torch.from_numpy(np.repeat(previous, 4, 0)), smoothing=smoothing).numpy()
    new = LemLite.gaze_to_handle(new, np.zeros((4, 2)))
    old = LemLite.gaze_to_handle(np.repeat(previous, 4, 0), np.zeros((4, 2)))
    new[:, LemLite.GAZE_IDX] = old[0, LemLite.GAZE_IDX]
    # Phase=0 starts a new order now. Other phases finish its remaining tenure.
    remaining = 0 if phase == 0 else 5-phase
    return np.stack([old if t < remaining else new for t in range(5)], 1)


def capture(args, model, native, limits, checkpoint_hash):
    from mimo_bridge.train_mimo import env_kwargs_for
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.tools.self_recovery_experiment import hold
    from mimo_bridge.lem_model import LemLite
    index = []; destination = args.out/'packets'
    for wi, seed in enumerate(args.train_worlds+args.test_worlds):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        kw = env_kwargs_for(native, limits['phase_name'], str(args.out/f'world_{seed}'))
        kw.update(seed=seed, run_dir=str(args.out/f'world_{seed}'),
                  caregiver_record_dir='', gaze_reflex=False, spontaneous_saccades=False)
        source = SingleEnvSource(**kw)
        try:
            frame = source.reset()[0]; previous = hold(frame, min(.3, limits['strength_limit']))
            previous = LemLite.gaze_to_handle(previous, np.zeros((1, 2)))
            context = deque(maxlen=6)
            def advance(command, render):
                nonlocal frame
                before = float(frame.sim_time)
                for sub in range(native.action_repeat):
                    frame = source.step(command[None], np.zeros((1, 2)), limits.get('handle_mask'),
                                        render=render and sub == native.action_repeat-1)[0]
                    actual = np.asarray(frame.meta['gaze_cmd'])
                    if not np.allclose(actual, 0, atol=1e-7, rtol=0):
                        raise ValueError('future gaze differs from the precommitted fixed order')
                return float(frame.sim_time)-before
            for _ in range(6):
                advance(previous[0], True); context.append(sensory_payload(source._last_payload))
            for event in range(args.states):
                phase = event % 5
                # Explicitly re-issue the prior body order, then observe its
                # declared age. These are copied excitation trials, not claims
                # about the production scheduler's phase distribution.
                order_started = float(frame.sim_time)
                for _ in range(phase):
                    advance(previous[0], True); context.append(sensory_payload(source._last_payload))
                commands = commands_for(frame, previous, limits, native.smooth_actions, phase, seed*100+event)
                start_groups = fixed_targets(frame); snapshot = source.save_full_state()
                traces = []; durations = []; times = []; selected_context = []
                for branch in list(range(4))+[0, 3]:
                    frame = source.restore_full_state(snapshot)[0]
                    y = []; dts = []; ts = []
                    render = len(traces) == 5
                    for command in commands[branch]:
                        dts.append(advance(command, render)); y.append(packed_target(fixed_targets(frame)))
                        ts.append(float(frame.sim_time))
                        if render: selected_context.append(sensory_payload(source._last_payload))
                    traces.append(np.stack(y)); durations.append(dts); times.append(ts)
                if not np.array_equal(traces[3], traces[5]):
                    raise ValueError('rendered continuation changed a precommitted physical target')
                packet = dict(contract=CONTRACT, world_id=seed, decision=event, phase=phase,
                    source_checkpoint_sha256=checkpoint_hash, context=list(context),
                    order_started=order_started,
                    decision_time=float(context[-1]['sim_time']), commands=commands,
                    durations=np.asarray(durations[:4]), outcome_times=np.asarray(times[:4]),
                    targets=np.stack(traces[:4]), replay_target=traces[4],
                    start=packed_target(start_groups), target_layout=target_layout(start_groups),
                    input_provenance='predecision_raw_senses_and_precommitted_commands',
                    reflex_off=True, saccades_off=True, gaze_handle=23,
                    split='train' if seed in args.train_worlds else 'test')
                receipt = write_packet(destination, packet, args.max_packet_mib*1024**2)
                index.append(dict(**receipt, world=seed, event=event, phase=phase, split=packet['split']))
                atomic_json(args.out/'packet_index.json', index)
                context.extend(selected_context); previous = commands[3, -1][None].copy()
                status(args.out, 'recording_raw_causal_windows', worlds_complete=wi,
                       world=seed, decisions_saved=len(index), bytes_saved=sum(r['compressed_bytes'] for r in index))
        finally: source.close()
    return index


def materialize(model, out, index):
    rows = []
    for i, receipt in enumerate(index):
        packet = load_packet(out/'packets'/receipt['path'])
        encoded, past = encode_own_window(model, packet['context'])
        times = np.array([p['sim_time'] for p in packet['context']])
        history_dt = np.r_[0, np.diff(times)].astype(np.float32)
        from mimo_bridge.lem_model import command_for_encoder
        action = command_for_encoder(torch.from_numpy(packet['commands'])).reshape(4, 5, -1)
        rows.append(dict(world=packet['world_id'], phase=packet['phase'], split=packet['split'],
             encoded=encoded, sensed=torch.from_numpy(past), history_dt=torch.from_numpy(history_dt),
             action=action, duration=torch.tensor(packet['durations'], dtype=torch.float32),
             target=torch.from_numpy(packet['targets']), start=torch.from_numpy(packet['start']),
             layout=packet['target_layout'], packet_sha256=receipt['sha256']))
        if i % 8 == 0: status(out, 'encoding_own_raw_windows', complete=i+1, total=len(index))
    return rows


def validate_index(args,index,checkpoint_hash):
    """Require the complete declared Cartesian sample, with located identities."""
    expected={(world,event) for world in args.train_worlds+args.test_worlds for event in range(args.states)}
    seen=set();paths=set();digests=set()
    for receipt in index:
        name=receipt['path']
        if Path(name).name!=name:raise ValueError('packet index path must name a local immutable packet')
        path=args.out/'packets'/name; packet=load_packet(path)
        identity=(packet['world_id'],packet['decision'])
        if identity not in expected or identity in seen:raise ValueError('missing, repeated or undeclared decision identity')
        if name in paths or receipt['sha256'] in digests:raise ValueError('duplicate packet receipt')
        split='train' if identity[0] in args.train_worlds else 'test'
        if (receipt['sha256']!=name.split('.')[0] or receipt['world']!=identity[0]
            or receipt['event']!=identity[1] or receipt['phase']!=packet['phase']
            or packet['phase']!=identity[1]%5 or receipt['split']!=split or packet['split']!=split
            or packet['source_checkpoint_sha256']!=checkpoint_hash
            or receipt['compressed_bytes']!=path.stat().st_size):
            raise ValueError('packet, receipt or declared source identity differs')
        seen.add(identity);paths.add(name);digests.add(receipt['sha256'])
    if seen!=expected:raise ValueError('partial packet index is not the declared experiment')
    return True


def preserve_encoder(model,native,args,context):
    """Save enough to reconstruct the exact encoder independently of retention."""
    import inspect
    from lem.model.encoder_registry import EncoderRegistry
    from lem.model import spatial,causal_readout
    from mimo_bridge import causal_packets,datasource,frame_encoder,skin_contract,vision_adapter
    from tokenizer import tokenizer as tokenizer_reader
    from mimo_bridge.lem_model import LemLite
    source_paths={Path(m.__file__).resolve() for m in
                  (spatial,causal_readout,causal_packets,datasource,frame_encoder,skin_contract,vision_adapter,tokenizer_reader)}
    source_paths.add(Path(__file__).resolve())
    source_paths.add(Path(inspect.getfile(LemLite)).resolve())
    for module in model.encoder_registry.modules():
        path=Path(inspect.getfile(type(module))).resolve()
        if 'Source' in path.parts:source_paths.add(path)
    from tokenizer import config as tokenizer_config
    from tokenizer.modalities import proprioception,touch,motor,audio
    source_paths.update(Path(m.__file__).resolve() for m in
                        (tokenizer_config,proprioception,touch,motor,audio))
    code={str(path):dict(sha256=file_hash(path),source=path.read_bytes()) for path in source_paths}
    state=dict(schema='standalone_fixed_sensory_encoder_v1',state=model.encoder_registry.state_dict(),
        resolved_config=model.config,active=model.active,contact_profile=model.contact_profile,
        native_configuration=vars(native),source_checkpoint=str(args.checkpoint.resolve()),
        source_checkpoint_sha256=args.checkpoint_sha256,encoder_sha256=args.encoder_sha256,
        source_files=code,torch_version=str(torch.__version__),numpy_version=np.__version__)
    path=args.out/'frozen_encoder.pt';torch.save(state,path)
    receiver=restore_encoder(path)
    original=encode_own_window(model,context);restored=encode_own_window(receiver,context)
    if not torch.equal(original[0],restored[0]) or original[1].tobytes()!=restored[1].tobytes():
        raise ValueError('standalone frozen encoder reconstruction changed raw-window features or fixed targets')
    return dict(path=path.name,sha256=file_hash(path),exact_raw_window_reconstruction=True,
                source_tensor_hash=args.encoder_sha256,source_checkpoint=str(args.checkpoint.resolve()))


def restore_encoder(path):
    """Checked standalone loading: reject changed preprocessing before encoding."""
    from lem.model.encoder_registry import EncoderRegistry
    from mimo_bridge.lem_model import LemLite
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved.get('schema')!='standalone_fixed_sensory_encoder_v1'
        or saved['torch_version']!=str(torch.__version__) or saved['numpy_version']!=np.__version__):
        raise ValueError('standalone encoder requires the recorded implementation environment')
    for name,record in saved['source_files'].items():
        if hashlib.sha256(record['source']).hexdigest()!=record['sha256'] or file_hash(name)!=record['sha256']:
            raise ValueError('recorded encoder/preprocessing implementation changed; explicit revalidation required')
    if not saved['source_files'] or tensor_digest(saved['state'])!=saved['encoder_sha256']:
        raise ValueError('standalone encoder source or weight identity is missing')
    registry=EncoderRegistry(saved['resolved_config']).to('cpu');registry.load_state_dict(saved['state'],strict=True)
    receiver=SimpleNamespace(device=torch.device('cpu'),active=saved['active'],
                             contact_profile=saved['contact_profile'],encoder_registry=registry)
    receiver._batch_frames=MethodType(LemLite._batch_frames,receiver)
    return receiver


def training_scale(rows):
    """Per-target scale from TRAINING trajectories only; zero-effect channels masked."""
    changes = torch.cat([(r['target']-r['start']).reshape(-1, len(r['start'])) for r in rows])
    scale = changes.square().mean(0).sqrt()
    mask = scale > 8*torch.finfo(torch.float32).eps
    return torch.where(mask, scale, torch.ones_like(scale)), mask


def batch(rows, scale, encoded_scale, blind=False, sensed_mean=None, sensed_scale=None):
    encoded = torch.cat([r['encoded'][None].expand(4, -1, -1) for r in rows])/encoded_scale
    sensed = torch.cat([r['sensed'][None].expand(4, -1, -1) for r in rows])
    sensed = (sensed-(0 if sensed_mean is None else sensed_mean))/(scale if sensed_scale is None else sensed_scale)
    hd = torch.cat([r['history_dt'][None].expand(4, -1) for r in rows])
    action = torch.cat([r['action'] for r in rows])
    duration = torch.cat([r['duration'] for r in rows])
    target = torch.cat([(r['target']-r['start'])/scale for r in rows])
    return (encoded, sensed, hd, torch.zeros_like(action) if blind else action, duration), target


def metrics(rows, predicted, label, by_phase=False):
    reports = {}; offset = 0
    for row in rows:
        if 'sensed' not in row or 'history_dt' not in row or row['history_dt'][-1]<=0:
            raise ValueError('constant-motion control requires actual past sensory times')
        p = predicted[offset:offset+4]; offset += 4
        for group, (lo, hi) in row['layout'].items():
            y = row['target'][:, :, lo:hi]; forecast = p[:, :, lo:hi]
            centered = y-y.mean(0, keepdim=True); guessed = forecast-forecast.mean(0, keepdim=True)
            for t in (0, 4):
                key = (row['world'], row.get('phase') if by_phase else None, group, t)
                entry = reports.setdefault(key, dict(effect=[], error=[], rmse=[], persistence=[], wrong=[],motion=[],floor=[]))
                entry['effect'].append(float(centered[:, t].square().mean()))
                entry['error'].append(float((guessed[:, t]-centered[:, t]).square().mean()))
                entry['rmse'].append(float((forecast[:, t]-y[:, t]).square().mean()))
                entry['persistence'].append(float((y[:, t]-row['start'][lo:hi]).square().mean()))
                entry['wrong'].append(float((forecast.roll(1,0)[:, t]-y[:, t]).square().mean()))
                velocity=(row['sensed'][-1,lo:hi]-row['sensed'][-2,lo:hi])/row['history_dt'][-1]
                extrapolated=row['start'][lo:hi]+velocity*row['duration'][:, :t+1].sum(1)[:,None]
                entry['motion'].append(float((extrapolated-y[:,t]).square().mean()))
                floor=8*torch.finfo(torch.float32).eps*y.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
                entry['floor'].append(float(floor))
    output = []
    for (world, phase, group, t), values in reports.items():
        m = {k: float(np.mean(v)) for k, v in values.items()}
        valid = m['effect']**.5 > m['floor'] and m['effect'] > 0
        output.append(dict(world=world, imposed_order_age=phase,group=group, forecast=label, horizon_seconds=.05*(t+1),
            action_effect_rms=m['effect']**.5, prediction_rmse=m['rmse']**.5,
            persistence_rmse=m['persistence']**.5, wrong_action_rmse=m['wrong']**.5,
            constant_motion_rmse=m['motion']**.5, action_contrast_error_rms=m['error']**.5,
            numerical_resolution=m['floor'],informative=valid,
            action_error_removed=1-m['error']/m['effect'] if valid else None,
            effect_resolved=valid and m['error'] < m['effect'],
            scale='Action-error removed: 0=ignores action differences; 1=predicts every difference; negative=worse. RMSE uses the named experienced channel units.'))
    return output


def admission(results,expected_worlds=None):
    """Predeclared physical resolution and controls; favorable loss cannot bypass."""
    keyed={(r['world'],r['group'],r['horizon_seconds'],r['forecast']):r for r in results}
    if len(keyed)!=len(results):raise ValueError('duplicate admission comparison cannot overwrite evidence')
    required={'position_cm','rotation_6d','touch_kpa','touch_first_kpa_cm','touch_second_kpa_cm2'}
    worlds=set(expected_worlds) if expected_worlds is not None else {r['world'] for r in results}
    labels={'action_conditioned_stage_1','action_conditioned_stage_2','action_blind_stage_1','action_blind_stage_2'}
    expected={(w,g,.25,label) for w in worlds for g in required for label in labels}
    actual={key for key in keyed if key[2]==.25}
    if not worlds or expected!=actual:raise ValueError('incomplete or undeclared world/group/stage admission matrix')
    numeric=('action_effect_rms','action_contrast_error_rms','prediction_rmse','persistence_rmse',
             'wrong_action_rmse','constant_motion_rmse','numerical_resolution')
    if any(not np.isfinite(r[k]) or r[k]<0 for r in results for k in numeric):
        raise ValueError('invalid admission measurement')
    decisions=[]
    for key, final in keyed.items():
        world,group,horizon,label=key
        if label!='action_conditioned_stage_2' or horizon!=.25:continue
        before=keyed.get((world,group,horizon,'action_conditioned_stage_1'))
        blind=keyed.get((world,group,horizon,'action_blind_stage_2'))
        if before is None or blind is None:raise ValueError('missing required admission comparator')
        if not final['informative']:
            decisions.append(dict(world=world,group=group,status='uninformative',
                reason='No action distinction above decoded-channel numerical resolution.'))
            continue
        effect=final['action_effect_rms'];floor=final['numerical_resolution']
        if effect<=floor:raise ValueError('uninformative physical effect was labelled informative')
        gates=dict(action_resolution=final['action_contrast_error_rms']<=.5*effect,
            beats_trained_blind=final['prediction_rmse']+floor<blind['prediction_rmse'],
            beats_wrong_action=final['prediction_rmse']+floor<final['wrong_action_rmse'],
            beats_persistence=final['prediction_rmse']+floor<final['persistence_rmse'],
            beats_constant_motion=final['prediction_rmse']+floor<final['constant_motion_rmse'],
            retains_panel=final['prediction_rmse']<=before['prediction_rmse']+.05*effect+floor)
        decisions.append(dict(world=world,group=group,status='passed' if all(gates.values()) else 'failed',
                              gates=gates,retention_allowance=.05*effect+floor))
    informative=[d for d in decisions if d['status']!='uninformative']
    covered=set(d['group'] for d in informative)
    verdict='failed' if any(d['status']=='failed' for d in informative) else (
        'passed' if informative and covered==required and len(informative)==len(decisions) else 'inconclusive')
    return dict(verdict=verdict,comparisons=decisions,missing_informative_groups=sorted(required-covered),
                production_qualification=False,intention_demonstrated=False)


def fit(rows, args):
    train = [r for r in rows if r['split'] == 'train']; test = [r for r in rows if r['split'] == 'test']
    if set(r['world'] for r in train)!=set(args.train_worlds) or set(r['world'] for r in test)!=set(args.test_worlds):
        raise ValueError('recorded world split differs from the pre-capture declaration')
    if any(r['layout']!=rows[0]['layout'] for r in rows):raise ValueError('fixed target semantics changed across packets')
    if not train or len(set(r['world'] for r in test)) < 2:
        raise ValueError('fit needs training worlds and at least two untouched test worlds')
    scale, mask = training_scale(train)
    if not mask.any(): raise ValueError('training worlds contain no sensed changes')
    past = torch.cat([r['sensed'] for r in train])
    sensed_mean = past.mean(0); sensed_scale = past.std(0)
    sensed_scale = torch.where(sensed_scale > 1e-6, sensed_scale, torch.ones_like(sensed_scale))
    encoded_scale = torch.cat([r['encoded'] for r in train]).square().mean(0).sqrt().clamp_min(1e-6)
    example = train[0]; outputs = []; phase_outputs=[]; preservation = {}
    for blind in (False, True):
        torch.manual_seed(args.fit_seed)
        model = CausalReadout(example['encoded'].shape[-1], len(scale), example['action'].shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        # Split learning into two retained phases; fixed old-world test inputs are
        # never fitted. Old TRAINING worlds remain in phase two rehearsal.
        first_worlds = set(args.train_worlds[:max(1, len(args.train_worlds)//2)])
        early = [r for r in train if r['world'] in first_worlds]
        for stage, data in enumerate((early, train)):
            inputs, target = batch(data, scale, encoded_scale, blind, sensed_mean, sensed_scale)
            model.train()
            for update in range(args.updates):
                optimizer.zero_grad(set_to_none=True)
                pred = model(*inputs); loss = (pred[..., mask]-target[..., mask]).square().mean()
                if not torch.isfinite(loss): raise ValueError('nonfinite copied auxiliary learning')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            model.eval()
            with torch.no_grad():
                inputs, _ = batch(test, scale, encoded_scale, blind, sensed_mean, sensed_scale)
                prediction = model(*inputs)*scale + torch.cat([r['start'][None].expand(4,-1) for r in test])[:,None]
            label = ('action_blind' if blind else 'action_conditioned')+f'_stage_{stage+1}'
            outputs += metrics(test, prediction, label)
            phase_outputs += metrics(test,prediction,label,by_phase=True)
            status(args.out, 'fitting_isolated_auxiliary', condition=label, updates=args.updates,
                   final_training_loss=float(loss), test_worlds=len(set(r['world'] for r in test)))
        state = dict(contract=CONTRACT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     architecture=dict(encoded_width=example['encoded'].shape[-1],target_width=len(scale),
                                       action_width=example['action'].shape[-1],hidden=128),
                     target_layout=example['layout'],
                     scale=scale, active=mask, encoded_scale=encoded_scale,
                     sensed_mean=sensed_mean,sensed_scale=sensed_scale,
                     source_checkpoint_sha256=args.checkpoint_sha256,
                     source_encoder_sha256=args.encoder_sha256,
                     frozen_encoder_artifact=getattr(args,'encoder_artifact',None),
                     packet_ids=[r['packet_sha256'] for r in rows],
                     new_weights_only=True, production_qualification=False)
        path = args.out/('action_blind.pt' if blind else 'action_conditioned.pt'); torch.save(state,path)
        reloaded = torch.load(path, map_location='cpu', weights_only=False)
        if tensor_digest(state) != tensor_digest(reloaded): raise ValueError('auxiliary save/resume differs')
        reconstructed=CausalReadout(**reloaded['architecture'])
        reconstructed.load_state_dict(reloaded['model']);reconstructed.eval()
        with torch.no_grad():
            if not torch.equal(model(*inputs),reconstructed(*inputs)):
                raise ValueError('restored auxiliary forecast differs')
        preservation[label] = file_hash(path)
    # This pilot is an instrument result, never an acting license or skill claim.
    return dict(schema='matched_raw_causal_readout_assay_v1', results=outputs,phase_results=phase_outputs,
        admission=admission(outputs,args.test_worlds),
        body_and_spatial_touch_groups=list(example['layout']), new_auxiliary_checkpoints=preservation,
        new_trainable_parameters=sum(p.numel() for p in model.parameters()),
        old_brain_trained=False, live_actions_changed=False, production_qualification=False,
        intention_demonstrated=False,
        interpretation='This tests a separate matched-encoder readout of sensed movement and contact. Useful action selection, correction, retention under a changed body and general gain of function require subsequent closed-loop tests.',
        retention_scope='The identical untouched test-world panel is evaluated before and after the second learning stage; its scores are descriptive and never used for stopping or fitting.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--train-worlds',type=int,nargs='+',default=list(range(93010,93016)))
    parser.add_argument('--test-worlds',type=int,nargs='+',default=list(range(93020,93024)))
    parser.add_argument('--states',type=int,default=8); parser.add_argument('--updates',type=int,default=240)
    parser.add_argument('--fit-seed',type=int,default=619); parser.add_argument('--max-packet-mib',type=int,default=512)
    parser.add_argument('--fit-only',action='store_true')
    parser.add_argument('--fit-output',type=Path,help='Fresh child directory for an explicitly retained retry of fitting the same packets')
    parser.add_argument('--declare-only',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(1)
    if (set(args.train_worlds)&set(args.test_worlds) or len(set(args.train_worlds))<2
        or len(set(args.test_worlds))<2 or len(set(args.train_worlds+args.test_worlds))!=len(args.train_worlds+args.test_worlds)):
        raise ValueError('world split must be disjoint before capture')
    args.out.mkdir(parents=True,exist_ok=True)
    declaration=dict(schema=CONTRACT, source=str(args.checkpoint.resolve()),
        train_worlds=args.train_worlds,test_worlds=args.test_worlds,states_per_world=args.states,
        horizon_seconds=[.05,.1,.15,.2,.25],past_observations=6,candidates=4,
        duplicate_physical_hold=True,imposed_order_ages=list(range(5)),reflex=False,saccades=False,
        updates_per_stage=args.updates,stages=2,fit_seed=args.fit_seed,max_packet_mib=args.max_packet_mib,
        goal='Resolve sensed movement and spatial-contact differences between known body orders.',
        encoder_input='Own frozen global tokens plus existing retinal-grid/local-touch features, fixed raw sensory history and actual intervals. No cached other-snapshot embedding is fed to this new readout.',
        comparison='Same context and gaze; separate equally trained action-blind readout, wrong-action forecasts, persistence, constant-motion extrapolation and identical held-command replay.',
        admission='At 250 ms, action-contrast error RMS must be at most half measured contrast (at least 75% of action variation error removed), and absolute error must beat fitted blind, wrong-action, persistence and constant-motion controls in every informative world/group. Stage-2 error cannot exceed stage-1 error by more than 5% of that physical contrast plus decoded-channel numerical resolution. Quiet groups are uninformative. This is a declared instrument-resolution requirement, not a biological law or an acting license.',
        phase_scope='Prior commands are explicitly re-issued then observed for 0..4 model intervals before the prospective remaining-order schedule. These are imposed copied conditions, not a measured distribution of native scheduler phases.',
        targets='Fixed pre-ANN body-relative position cm, world rotation columns, experienced pressure kPa and its existing spatial first/second moments. No simulator-only object positions or future reflex commands.',
        production_modified=False)
    path=args.out/'declaration.json'
    if path.exists():
        if json.loads(path.read_text())!=declaration: raise ValueError('existing declaration differs')
    else: atomic_json(path,declaration)
    if args.declare_only:
        status(args.out,'declared_waiting_for_renderer',declaration='declaration.json'); return
    fit_out=args.fit_output or args.out
    if args.fit_output:
        if not args.fit_only or args.out.resolve() not in fit_out.resolve().parents:
            raise ValueError('a new fit output must be a child of the existing recorded study')
    if any((fit_out/name).exists() for name in ('action_conditioned.pt','action_blind.pt','report.json','frozen_encoder.pt')):
        raise FileExistsError('existing fit evidence retained; use --fit-only --fit-output with a fresh child directory')
    if not args.fit_only and ((args.out/'packet_index.json').exists() or list((args.out/'packets').glob('*.packet.gz'))):
        raise FileExistsError('partial or completed capture retained; use a new declared capture directory')
    fit_out.mkdir(parents=True,exist_ok=True)
    from mimo_bridge import causal_packets
    from lem.model import causal_readout
    source_hashes={str(Path(p).resolve()):file_hash(p) for p in
                   (__file__,causal_packets.__file__,causal_readout.__file__)}
    version=args.out/'implementation_versions';version.mkdir(exist_ok=True)
    atomic_json(version/(str(time.time_ns())+'.json'),dict(source_hashes=source_hashes,
        declaration_sha256=file_hash(path),fit_only=args.fit_only))
    try:
        status(args.out,'loading_exact_source_copy')
        model,native,limits,basis=source_model(args.checkpoint,args.out)
        before=old_state_digest(model); args.checkpoint_sha256=file_hash(args.checkpoint/'model.safetensors')
        args.encoder_sha256=tensor_digest(model.encoder_registry.state_dict())
        if native.action_repeat != 5 or native.motor_hold != 5:
            raise ValueError('this declared assay requires the actual 5x10 ms and 5-observation tenure')
        if args.fit_only: index=json.loads((args.out/'packet_index.json').read_text())
        else: index=capture(args,model,native,limits,args.checkpoint_sha256)
        validate_index(args,index,args.checkpoint_sha256)
        fit_args=deepcopy(args);fit_args.out=fit_out
        fit_args.encoder_artifact=preserve_encoder(model,native,fit_args,
            load_packet(args.out/'packets'/index[0]['path'])['context'])
        rows=materialize(model,args.out,index)
        report=fit(rows,fit_args)
        after=old_state_digest(model)
        if after!=before: raise ValueError('copied old model or Adam changed during isolated assay')
        report.update(old_model_and_adam_exact=before==after, old_state_hashes=before,
                      packet_count=len(index),packet_bytes=sum(r['compressed_bytes'] for r in index),
                      source_checkpoint_sha256=args.checkpoint_sha256,encoder_sha256=args.encoder_sha256,
                      standalone_encoder=fit_args.encoder_artifact,raw_packet_root=str((args.out/'packets').resolve()),
                      developmental_limits=limits,limits_basis=basis)
        atomic_json(fit_out/'report.json',report); status(args.out,'complete',report=str(fit_out/'report.json'))
    except Exception as exc:
        atomic_json(args.out/'failure.json',dict(type=type(exc).__name__,message=str(exc),timestamp=time.time()))
        status(args.out,'failed',error=str(exc)); raise


if __name__=='__main__': main()
