# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Audit the frozen native action-conditioned sensory predictor on copied worlds.

Forecasts are made BEFORE physics advances, using the very caches used by the
native delayed next-frame lesson. A repeated branch measures the numerical floor.
Later frames of a held command use observed intervening history: they are online
one-transition forecasts, never described as an open-loop long-horizon forecast.
No developmental capability or live-controller adoption follows from this assay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from mimo_bridge import continuation_state
from mimo_bridge.acting_snapshot import runtime, restore_runtime
from mimo_bridge import copied_execution
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite, MOTOR_ACHIEVED_F, command_for_encoder
from mimo_bridge.train_mimo import build_learner, env_kwargs_for
from .causal_test import _limits_for_run
from .developmental_probe import geometry_for_seed, independent_start, body_signature
from .migrate_trunk import modules
from .motor_reachability_probe import groups, proposal, pose
from .self_recovery_experiment import configuration, hold
from .training_continuation import atomic_json

VERSION = 'native_action_prediction_v1'
MODALITIES = ('proprioception', 'tactile', 'vision')


def finite(values, name):
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError(f'{name}: missing or nonfinite observations')
    return x


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, module in sorted(modules(model).items()):
        for key, tensor in sorted(module.named_parameters()):
            digest.update((name + '/' + key).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def forecast(model, executed):
    """Exact native delayed-lesson path, evaluated using only already-observed data."""
    if model._nf_raw is None or model._nf_ctx is None:
        raise ValueError('No observed native next-frame cache')
    raw = dict(model._nf_raw)
    motor = raw.get('motor')
    if not torch.is_tensor(motor) or motor.shape[-1] != MOTOR_ACHIEVED_F + 9:
        raise ValueError('Checkpoint lacks the achieved-state plus outgoing-command motor contract')
    command = torch.as_tensor(executed, device=motor.device, dtype=motor.dtype)
    raw['motor'] = torch.cat([motor[..., :MOTOR_ACHIEVED_F], command_for_encoder(command)], -1)
    with torch.no_grad():
        bias = None
        if model.persistent_memory is not None and model._nf_memory is not None:
            state, _ = model.persistent_memory.encode(**model._nf_memory)
            bias = model.persistent_memory.bias(state)
        if model.self_state_mode == 'perceive' and model.self_history.inputs is not None:
            self_bias = model.self_state.perceive(**model.self_history.inputs)['bias']
            bias = self_bias if bias is None else bias + self_bias
        encoded = model.encoder_registry.encode(raw, model._nf_mask)
        tokens = {m: encoded[m] if m in encoded and model._nf_mask.get(m) == 'real'
                  else model._nf_tokens[m] for m in model.active}
        _, last, _ = model.forward_model(tokens, model._nf_ctx,
                                         **({} if bias is None else {'memory_bias': bias}))
        predicted = model.prediction_heads(last, current={m: v.squeeze(1) if v.dim() == 3 else v
                                                          for m, v in tokens.items()},
                                           who='developmental_representation_probe pre-physics')
    return {m: p.detach().float().cpu().numpy().copy() for m, p in predicted.items() if m in MODALITIES}


def score_event(target, predicted, swapped, current, repeated):
    """Arrays [action, held-step, feature]. Only the FIRST step has shared history."""
    y, p, s, c, r = [finite(v, name) for v, name in
        ((target, 'target'), (predicted, 'prediction'), (swapped, 'swapped'),
         (current, 'persistence'), (repeated, 'repeat'))]
    if y.ndim != 3 or min(y.shape) < 1 or y.shape[0] < 3 or any(x.shape != y.shape for x in (p, s, c)) or r.shape != y.shape[1:]:
        raise ValueError('Need at least three equally observed action branches and a full duplicate branch')
    # The repeated physical outcome, encoded through the restored same history,
    # is the actual measurement noise. A float32 precision floor prevents an
    # exact deterministic replay making an arbitrarily tiny effect count as real.
    replay_mse = float(np.mean((r - y[0]) ** 2))
    numerical_mse = float((8 * np.finfo(np.float32).eps * max(1., np.abs(y).max())) ** 2)
    yc = y[:, 0] - y[:, 0].mean(0, keepdims=True)
    pc = p[:, 0] - p[:, 0].mean(0, keepdims=True)
    effect_mse = float(np.mean(yc ** 2))
    effect_error = float(np.mean((pc - yc) ** 2))
    resolved = effect_mse > max(10 * replay_mse, numerical_mse)
    costs = np.mean((yc[:, None] - pc[None, :]) ** 2, axis=-1)
    tied = np.isclose(costs, costs.min(-1, keepdims=True), rtol=1e-6, atol=numerical_mse)
    accuracy = float(np.mean(np.diag(tied) / tied.sum(-1)))
    correct_first = float(np.mean((p[:, 0] - y[:, 0]) ** 2))
    return dict(resolved=bool(resolved), actions=len(y), replay_mse=replay_mse,
        numerical_mse=numerical_mse, action_effect_mse=effect_mse,
        action_effect_error=effect_error,
        action_effect_error_removed=1 - effect_error / effect_mse if resolved else None,
        correct_first_mse=correct_first,
        swapped_first_mse=float(np.mean((s[:, 0] - y[:, 0]) ** 2)),
        persistence_first_mse=float(np.mean((c[:, 0] - y[:, 0]) ** 2)),
        action_identification_accuracy=accuracy if resolved else None,
        identification_chance=1 / len(y),
        online_correct_mse=float(np.mean((p-y)**2)),
        online_swapped_mse=float(np.mean((s-y)**2)),
        online_persistence_mse=float(np.mean((c-y)**2)))


def world_summary(events):
    if not events:
        raise ValueError('An empty world cannot be evaluated')
    names = set.intersection(*(set(row['modalities']) for row in events))
    answer = {}
    for name in sorted(names):
        rows = [row['modalities'][name] for row in events]
        average = lambda key: float(np.mean([row[key] for row in rows]))
        denominator = average('action_effect_mse')
        resolved = denominator > max(10 * average('replay_mse'), average('numerical_mse'))
        answer[name] = dict(events=len(rows), resolved=bool(resolved),
            action_effect_mse=denominator, repeat_mse=average('replay_mse'),
            action_effect_error_removed=(1-average('action_effect_error')/denominator) if resolved else None,
            swapped_advantage=average('swapped_first_mse')-average('correct_first_mse'),
            persistence_advantage=average('persistence_first_mse')-average('correct_first_mse'),
            action_identification_accuracy=float(np.mean([r['action_identification_accuracy']
                for r in rows if r['resolved']])) if any(r['resolved'] for r in rows) else None,
            resolved_events=sum(r['resolved'] for r in rows), total_events=len(rows))
    return answer


def sign_p(values):
    """One-sided exact sign test; ties count against an advantage, never disappear."""
    values = finite(values, 'independent worlds')
    n, wins = len(values), int(np.sum(values > 0))
    return sum(math.comb(n, k) for k in range(wins, n+1)) / (2 ** n)


def aggregate(worlds, requested_modalities, smoke=False):
    if not worlds:
        raise ValueError('No world evidence')
    family = max(1, 3 * len(requested_modalities))
    answer = {}
    for name in requested_modalities:
        rows = [w['modalities'].get(name) for w in worlds]
        if any(r is None for r in rows):
            answer[name] = dict(verdict='UNMEASURED', reason='No real input for this modality in every world')
            continue
        if not all(r['resolved'] for r in rows):
            answer[name] = dict(verdict='UNRESOLVED_EFFECT', reason='Action effects were not resolved above replay/precision in every world')
            continue
        contrasts = {key: [r[key] for r in rows] for key in
            ('action_effect_error_removed', 'swapped_advantage', 'persistence_advantage')}
        p = {key: sign_p(value) for key, value in contrasts.items()}
        passed = not smoke and len(worlds) >= 3 and all(v <= .05/family for v in p.values())
        answer[name] = dict(verdict='PREDICTION_COMPARISON_PASS' if passed else 'COMPARISON_NOT_PASSED',
            world_values=contrasts, one_sided_sign_p=p, family_size=family, corrected_alpha=.05/family,
            worlds=len(worlds))
    return dict(modalities=answer, capability_pass=False,
        scope='Frozen native sensory action prediction only. No acquisition, retention, purposeful behavior, visual ownership or self/world partition is established.')


def execute(src, command, repeats, render):
    gazes = []
    for i in range(repeats):
        frames = src.step(command, np.zeros((1,2)), render=render and i == repeats-1)
        gazes.append(frames[0].meta['gaze_cmd'])
    executed = LemLite.gaze_to_handle(command, np.mean(gazes, axis=0)[None])
    return frames, executed


def observed_step(model, frames, dev, executed, order):
    with torch.no_grad():
        return model.step(frames, dict(dev, exploration_noise=0.), train=False,
                   step_index=model.step_count, executed_action=executed, executed_order_id=order)


def run_world(a, checkpoint, out, seed, dev, events, warm, render, progress=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_learner(a, str(out/'unused_checkpoints'))
    receipt = model.load_checkpoint(str(checkpoint), load_optimizer=False)
    if receipt.get('reinitialized_components'):
        raise ValueError('Native forecast assay cannot reinitialize checkpoint components')
    if not model.next_frame_loss:
        raise ValueError('Checkpoint has no native next-frame prediction path')
    for module in modules(model).values():
        module.eval()
    before = parameter_digest(model)
    model.reset_transition_state()
    kw = env_kwargs_for(a, dev['phase_name'], str(out))
    kw.update(seed=seed, vision=render, caregiver_mode='off', caregiver_seconds=0., caregiver_record_dir='',
              mobile=False, critters=False, gaze_reflex=False, spontaneous_saccades=False)
    src = SingleEnvSource(**kw)
    try:
        frames = src.reset(); e = src.env
        geometry = geometry_for_seed(seed)
        independent_start(e, geometry, False)
        mid=e._target_id; e._target_id=None
        eye=e.model.camera('eye_left').id
        e.data.mocap_pos[mid]=e.data.cam_xpos[eye]-e.data.cam_xmat[eye].reshape(3,3)[:,2]*geometry['target_distance_m']
        e._mj.mj_forward(e.model,e.data)
        # Refresh the observations after the auditor changes the physical start.
        initial=pose(e,float(dev['strength_limit']))
        frames, executed=execute(src,initial,a.action_repeat,render)
        observed_step(model,frames,dev,executed,1)
        for _ in range(warm):
            frames,executed=execute(src,initial,a.action_repeat,render)
            observed_step(model,frames,dev,executed,1)
        rows=[]; branch_steps=max(1,a.motor_hold)
        signature=body_signature(e,a,dev)
        for event in range(events):
            atomic_json(out.parent/'status.json',dict(progress or {},stage='measuring',pid=os.getpid(),timestamp=time.time(),
                world=seed,event=event,events=events,condition_index=0,condition_count=4,frame=0,
                frames=branch_steps,elapsed_seconds=time.time()-(progress or {}).get('started_at',time.time())))
            world=src.save_full_state()
            execution=copied_execution.capture(model)
            reference=hold(frames[0],float(dev['strength_limit']))
            hgroups=groups(e.decoder.handle_names)
            raw=[proposal(reference,np.zeros((len(hgroups),6)),hgroups,.10)]
            for _ in range(2):
                parameters=rng.normal(0,.65,(len(hgroups),6))
                raw.append(proposal(reference,parameters,hgroups,.10))
            initial_previous=torch.as_tensor(executed).clone()
            results=[]
            for branch in (0,1,2,0):
                frames=src.restore_full_state(world)
                copied_execution.restore(model,execution)
                previous=initial_previous.clone(); records=[]
                for tick in range(branch_steps):
                    # Raw proposal is held for the trained motor tenure. The same
                    # production execution transform and smoother run every tick.
                    commands=[]
                    for candidate in raw:
                        cmd=LemLite.planned_execution(candidate,dev,model.last_achieved,
                            previous=previous,smoothing=a.smooth_actions).cpu().numpy()
                        commands.append(LemLite.gaze_to_handle(cmd,np.zeros((1,2))))
                    command=commands[branch]
                    predicted=forecast(model,command)
                    swapped=forecast(model,commands[(branch+1)%3])
                    current={m:v.detach().float().cpu().numpy().copy() for m,v in model.last_tokens.items() if m in MODALITIES}
                    frames,actual=execute(src,command,a.action_repeat,render)
                    if not np.allclose(command,actual,rtol=0,atol=1e-6):
                        raise ValueError('The executed motor/eye order differs from the forecast order')
                    observed=observed_step(model,frames,dev,actual,event+2)
                    measured={m:v.detach().float().cpu().numpy().copy() for m,v in model.last_tokens.items()
                              if m in MODALITIES and model._last_source_mask.get(m)=='real'}
                    parity={m:float(np.max(np.abs(predicted[m]-model._nf_pred[m].detach().float().cpu().numpy())))
                            for m in predicted}
                    if any(value>1e-6 for value in parity.values()):
                        raise ValueError(f'Pre-physics forecast disagrees with the native delayed lesson: {parity}')
                    records.append(dict(target=measured,predicted=predicted,swapped=swapped,current=current,
                        native_parity_max_abs=parity,command=actual.copy(),qpos=e.data.qpos.copy(),
                        execution_output=copied_execution.output_record(model,observed),
                        time_s=float(frames[0].sim_time)))
                    previous=torch.as_tensor(actual).clone()
                results.append(records)
            # A repeat compares complete physical trajectories, not final pose alone.
            physics_exact=all(np.array_equal(x['qpos'],y['qpos']) and np.array_equal(x['command'],y['command'])
                              for x,y in zip(results[0],results[3]))
            if not physics_exact:
                raise ValueError('Same-action replay did not reproduce the complete physical trajectory')
            if not all(copied_execution.equal_tree(x['execution_output'],y['execution_output'])
                       and copied_execution.equal_tree(x['predicted'],y['predicted'])
                       for x,y in zip(results[0],results[3])):
                torch.save(results,out/'replay_debug.pt')
                raise ValueError('Restored model differs at '+str(copied_execution.first_difference(
                    [x['execution_output'] for x in results[0]],[x['execution_output'] for x in results[3]])))
            actual_mods=set.intersection(*(set(record['target']) for result in results for record in result))
            metrics={}
            for name in sorted(actual_mods):
                arrays={key:np.stack([np.stack([step[key][name].reshape(-1) for step in result])
                                      for result in results[:3]])
                        for key in ('target','predicted','swapped','current')}
                repeat=np.stack([step['target'][name].reshape(-1) for step in results[3]])
                metrics[name]=score_event(**arrays,repeated=repeat)
            row=dict(event=event,world_seed=seed,modalities=metrics,same_action_replay_exact=True,
                complete_model_replay_exact=True,copied_execution_contract=copied_execution.CONTRACT,
                native_forecast_parity_max_abs=max(value for result in results for step in result
                                                  for value in step['native_parity_max_abs'].values()))
            atomic_json(out/f'event_{event:04d}.json',row)
            # Raw encoded outcomes, commands and forecasts permit an independent
            # recomputation. No model weights or live data are written here.
            # Full runtime is compared in memory above. Persist only measurements:
            # repeating that runtime for every tick previously cost ~700 MB/event.
            measured_results=[[{key:value for key,value in record.items() if key!='execution_output'}
                               for record in branch] for branch in results]
            torch.save(measured_results,out/f'event_{event:04d}_branches.pt')
            rows.append(row)
            # Branch zero is the continuation after every event, irrespective of
            # which arm happened to score better; no outcome-dependent sampling.
            executed=results[3][-1]['command']
        if before!=parameter_digest(model):
            raise ValueError('Frozen assay changed learned parameters')
        result=dict(world_seed=seed,geometry=geometry,body_signature=signature,
            modalities=world_summary(rows),events=len(rows),weights_unchanged=True,
            runtime_restored_per_branch=True,same_action_replay_exact=True,
            complete_model_replay_exact=True,copied_execution_contract=copied_execution.CONTRACT,
            rendered_vision=render,native_horizon_seconds=float(e.dt*a.action_repeat),
            held_command_seconds=float(e.dt*a.action_repeat*branch_steps))
        atomic_json(out/'report.json',result)
        return result
    finally:
        src.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--worlds',type=int,default=8)
    parser.add_argument('--steps',type=int,default=4,help='Independent branch start events per world, not gradient steps')
    parser.add_argument('--seed',type=int,default=88213)
    parser.add_argument('--smoke',action='store_true',help='One no-render event per world; instrument scope only')
    args=parser.parse_args()
    if args.worlds<1 or args.steps<1:
        parser.error('worlds and steps must be positive')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    a=configuration(args.checkpoint,args.out)
    from types import SimpleNamespace
    dev,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
    if not dev.get('phase_name'):
        raise ValueError('Missing checkpoint developmental limits')
    events=1 if args.smoke else args.steps
    warm=3 if args.smoke else int(a.context_length)
    declaration=dict(version=VERSION,checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ability_id='action_self_recognition',worlds=args.worlds,steps=events,seeds=list(range(args.seed,args.seed+args.worlds)),
        warm_steps=warm,instrument_smoke=args.smoke,actual_vision=not args.smoke,limits=dev,limits_source=origin,
        primary='First transition of each restored branch: same state/history, distinct executed commands.',
        secondary='Later one-transition forecasts during a held command, conditioned on actual intervening observations.',
        controls=['actual command swapped within this state','assume no sensory change','predict no difference between actions','repeat identical physical command'],
        statistics='Independent worlds; exact one-sided sign test with ties retained and Bonferroni across modalities and three comparisons.',
        effect_resolution='Action-centered target MSE exceeds both 10 times measured replay MSE and the float32 numerical floor.',
        scope='Auditor-selected commands test the native frozen sensory predictor. No learned intention, acquired self-recognition, visual ownership or self/world distinction is inferred.',
        capability_pass=False)
    atomic_json(args.out/'declaration.json',declaration)
    worlds=[];durations=[];started=time.time()
    try:
        for seed in declaration['seeds']:
            folder=args.out/str(seed);folder.mkdir();start=time.time()
            progress=dict(world_index=len(worlds),world_count=args.worlds,step=len(worlds),steps=args.worlds,
                completed_durations_seconds=durations,started_at=started)
            atomic_json(args.out/'status.json',dict(progress,stage='measuring',pid=os.getpid(),timestamp=time.time()))
            worlds.append(run_world(a,args.checkpoint,folder,seed,dev,events,warm,not args.smoke,progress))
            durations.append(time.time()-start)
            progress.update(world_index=len(worlds),step=len(worlds),completed_durations_seconds=durations,
                elapsed_seconds=time.time()-started)
            atomic_json(args.out/'progress.json',progress)
        result=aggregate(worlds,MODALITIES if not args.smoke else MODALITIES[:2],args.smoke)
        result.update(verdict='INSTRUMENT_EXECUTED' if args.smoke else 'NATIVE_ACTION_PREDICTION_MEASURED',
            results=worlds,completed_at=time.time(),instrument_smoke=args.smoke)
        atomic_json(args.out/'report.json',result)
        atomic_json(args.out/'status.json',dict(progress,stage='complete',pid=os.getpid(),timestamp=time.time(),verdict=result['verdict']))
    except Exception as error:
        atomic_json(args.out/'status.json',dict(stage='failed',pid=os.getpid(),timestamp=time.time(),error=str(error)))
        raise


if __name__=='__main__':
    main()
