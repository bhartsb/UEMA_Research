# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Train and test an isolated copy of the real model with the developmental changes.

Production is owned by training_continuation.py. This program never writes its
control files. Choices are recorded before outcomes, paired branches restore the
complete environment, and an identical-command branch measures replay error.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.consequence_learning import skin_observation
from lem.model.contact_wellness import contact_response, current_need_value
from lem.training.checkpoint import CheckpointManager


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def confidence(values):
    from scipy.stats import t
    x = np.asarray(values, float)
    if len(x) < 2 or not np.isfinite(x).all():
        return dict(valid=False, mean=None, lower=None, upper=None)
    margin = float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return dict(valid=True, mean=float(x.mean()), lower=float(x.mean() - margin), upper=float(x.mean() + margin))


def verify_migration(checkpoint, model):
    """Compare every surviving tensor and Adam moment against the actual parent."""
    parent, _, _, _ = _build_model(str(checkpoint), 'cpu', 702, intention=True)
    parent.load_checkpoint(str(checkpoint), load_optimizer=True)
    def named(m):
        modules = {k: getattr(m, k) for k in ('encoder_registry', 'forward_model', 'prediction_heads',
                   'motor_controller', 'wellness', 'gaze_head', 'voice_head')}
        modules.update(m._side_modules())
        return {f'{key}.{n}': p for key, module in modules.items() if module is not None
                for n, p in module.named_parameters()}
    old, new = named(parent), named(model)
    states = 0
    for name, before in old.items():
        after = new[name]
        if not torch.equal(before.detach().cpu(), after.detach().cpu()):
            raise RuntimeError(f'checkpoint weight changed during migration: {name}')
        evidence = parent.optimizer.state.get(before)
        if evidence:
            restored = model.optimizer.state.get(after, {})
            if evidence.keys() != restored.keys():
                raise RuntimeError(f'optimizer state missing for {name}')
            for k, v in evidence.items():
                if torch.is_tensor(v):
                    if not torch.equal(v.cpu(), restored[k].cpu()):
                        raise RuntimeError(f'optimizer {k} differs for {name}')
                elif v != restored[k]:
                    raise RuntimeError(f'optimizer {k} differs for {name}')
            states += 1
    return dict(parent_parameters_matched=len(old), optimizer_states_matched=states,
                includes_true_step_counters=True, fresh_parameters=sorted(set(new) - set(old)))


def environment(out, seed, speech, a, rendering=True):
    from mimo_bridge.train_mimo import parse_args, env_kwargs_for
    from types import SimpleNamespace
    settings = vars(parse_args(["--out", str(out)]))
    settings.update(a)
    kw = env_kwargs_for(SimpleNamespace(**settings), a.get("_trial_phase"), str(out))
    kw.update(scene='crib', run_dir=str(out / f'scene_{seed}'), seed=seed,
              audio_real=True, caregiver_mode='responsive', caregiver_speech_dir=str(speech),
              touch_geometry='surface', mobile=True, critters=True,
              gaze_reflex=rendering, spontaneous_saccades=rendering)
    return SingleEnvSource(**kw)


def advance(source, command, gaze, repeats):
    from mimo_bridge.lem_model import LemLite
    frames = None
    gaze_path = []
    for i in range(repeats):
        frames = source.step(command, gaze, render=i == repeats - 1)
        gaze_path.append([f.meta.get('gaze_cmd', (0., 0.)) for f in frames])
    source.last_executed = LemLite.gaze_to_handle(command, np.asarray(gaze_path).mean(0))
    return frames


def parent_frames(frames, source):
    """Keep the frozen parent's trained body-centre tactile coordinate contract."""
    from dataclasses import replace
    env = source.env
    positions = (env.data.xpos[env._touch_body_ids] - env.data.xpos[env._root_body_id]).astype(np.float32)
    return [replace(f, touch_pos=positions.copy()) for f in frames]


def choices(model, result, dev, seed, horizon, baseline=None):
    """Already executed-form candidates; hold is a real achieved-pose hold."""
    new_policy = torch.as_tensor(result.motor_for_action, device=model.device)
    policy = (new_policy if baseline is None else
              torch.as_tensor(baseline.motor_for_action, device=model.device))
    hold = policy.clone()
    hold[..., :3] = model.last_achieved[..., :3].float() * 200
    hold[..., 3:7] = model.last_achieved[..., 3:7].float()
    hold[:, model.GAZE_IDX] = policy[:, model.GAZE_IDX]
    gen = torch.Generator(device='cpu').manual_seed(seed)
    candidates = [policy, hold]
    # Existing developmental exploration amplitude, applied through the canonical
    # physical transform and smoother. The evaluator never gives privileged geometry.
    for _ in range(2):
        raw = model.last_achieved[..., :9].detach().float().clone()
        q = torch.randn(raw[..., 3:7].shape, generator=gen).to(model.device)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        raw[..., 3:7] = q
        cmd = model.differentiable_execution_transform(raw, dev, model.last_achieved)
        if model.smooth_actions > 0:
            cmd = model.smooth_blend(cmd, model._last_smooth_prev, model.smooth_actions)
        cmd[:, model.GAZE_IDX] = policy[:, model.GAZE_IDX]
        candidates.append(cmd)
    if baseline is not None:
        own = new_policy.clone()
        own[:, model.GAZE_IDX] = policy[:, model.GAZE_IDX]
        candidates.append(own)
    stacked = torch.stack(candidates)
    return stacked[:, :, None].expand(-1, -1, horizon, -1, -1).clone()


def evaluate(model, baseline, out, speech, a, dev, seeds, decisions, horizon, repeats, normal=False):
    if len(seeds) > 1 and not a.get('_evaluation_checkpoint'):
        raise ValueError('multiple worlds require fresh checkpoint construction per world')
    out.mkdir(parents=True, exist_ok=True)
    worlds = []
    for seed in seeds:
        if a.get('_evaluation_checkpoint'):
            # Transition resets deliberately retain learned/adaptive evidence.
            # Independent frozen-world comparisons need an entirely fresh load.
            model, _, _, _ = _build_model(a['_evaluation_checkpoint'], str(model.device), seed, intention=True)
            baseline, _, _, _ = _build_model(a['_parent_checkpoint'], str(baseline.device), seed, intention=True)
        source = environment(out, seed, speech, a, rendering=normal)
        rows = []
        try:
            frames = source.reset()
            model.reset_transition_state()
            baseline.reset_transition_state()
            command = None
            warm_held = None
            # Fill the temporal context and settle the sensors under the baseline.
            for warm in range(model.ctx_len):
                offset = int(a.get('_trial_start_step', 0)) + warm
                model.step(frames, dev, train=False, executed_action=command, step_index=offset)
                old = baseline.step(parent_frames(frames, source), dev, train=False, executed_action=command, step_index=offset)
                if warm_held is None or warm % max(1, int(a.get('motor_hold', 5))) == 0:
                    warm_held = np.array(old.motor_for_action, copy=True)
                command = warm_held
                frames = advance(source, command, old.gaze_out, repeats)
                command = source.last_executed
                model._prev_mfa = torch.as_tensor(command, device=model.device).clone()
                baseline._prev_mfa = torch.as_tensor(command, device=baseline.device).clone()
            for step in range(decisions):
                result = model.step(frames, dev, train=False, executed_action=command,
                                    step_index=int(a.get('_trial_start_step', 0)) + model.ctx_len + step * horizon)
                old = baseline.step(parent_frames(frames, source), dev, train=False, executed_action=command,
                                    step_index=int(a.get('_trial_start_step', 0)) + model.ctx_len + step * horizon)
                candidates = choices(model, result, dev, seed * 1000 + step, horizon, baseline=old)
                durations = torch.full((1, horizon), source.env.dt * repeats, device=model.device)
                scores, _ = model.consequence_lesson.forecast(model, candidates, durations, body_only=True)
                selected = int(scores[:, 0].argmax())
                row = dict(world=seed, decision=step, selected=selected,
                           forecast=scores[:, 0].tolist(), outcomes=[])
                # Persist the actual orders with the pre-outcome decision.
                torch.save(dict(candidates=candidates.cpu(), choice=row), out / f'choice_{seed}_{step}.pt')
                snapshot = source.env.save_full_state()
                initial = skin_observation(frames, model.device)
                branch_frames = []
                for index in list(range(len(candidates))) + [0]:
                    source.env.restore_full_state(snapshot)
                    values, pains = [], []
                    for h in range(horizon):
                        cmd = candidates[index, :, h].detach().cpu().numpy()
                        gaze = model._gaze_from_handle(candidates[index, :, h])
                        observed = advance(source, cmd, gaze, repeats)
                        skin = skin_observation(observed, model.device)
                        response = contact_response(skin['features'], skin['sensitivity'], skin['valid'])
                        values.append(float(current_need_value(response['pain'], response['pleasant'],
                                                skin['capacity'], initial['capacity']).mean()))
                        pains.append(float(response['pain'].max()))
                    row['outcomes'].append(dict(value=float(np.mean(values)), worst_pain=max(pains)))
                    branch_frames.append(observed)
                row['replay_error'] = abs(row['outcomes'][0]['value'] - row['outcomes'][-1]['value'])
                row['pain_replay_error'] = abs(row['outcomes'][0]['worst_pain'] - row['outcomes'][-1]['worst_pain'])
                rows.append(row)
                # Commit the selected action in the world and repeat observation.
                source.env.restore_full_state(snapshot)
                for h in range(horizon):
                    command = candidates[selected, :, h].detach().cpu().numpy()
                    frames = advance(source, command, model._gaze_from_handle(candidates[selected, :, h]), repeats)
                    # Keep sensing at the trained 50 ms cadence while the chosen
                    # body order is held. Skipping these frames changes what the
                    # model's temporal context means at the next decision.
                    if h < horizon - 1:
                        offset = int(a.get('_trial_start_step', 0)) + model.ctx_len + step * horizon + h + 1
                        model.step(frames, dev, train=False, executed_action=source.last_executed, step_index=offset)
                        baseline.step(parent_frames(frames, source), dev, train=False,
                                      executed_action=source.last_executed, step_index=offset)
                    model._prev_mfa = torch.as_tensor(source.last_executed, device=model.device).clone()
                    baseline._prev_mfa = torch.as_tensor(source.last_executed, device=baseline.device).clone()
                command = source.last_executed
                model._prev_mfa = torch.as_tensor(command, device=model.device).clone()
                baseline._prev_mfa = torch.as_tensor(command, device=baseline.device).clone()
            source.env.caregiver.export_events(out / f'caregiver_{seed}.json')
            world = dict(seed=seed, rows=rows,
                benefit=float(np.mean([r['outcomes'][r['selected']]['value'] - r['outcomes'][0]['value'] for r in rows])),
                hold_benefit=float(np.mean([r['outcomes'][r['selected']]['value'] - r['outcomes'][1]['value'] for r in rows])),
                pain_change=float(np.mean([r['outcomes'][r['selected']]['worst_pain'] - r['outcomes'][0]['worst_pain'] for r in rows])),
                replay_error=max(r['replay_error'] for r in rows),
                pain_replay_error=max(r['pain_replay_error'] for r in rows))
            worlds.append(world)
            write_json(out / f'world_{seed}.json', world)
            print(json.dumps({k:v for k,v in world.items() if k != 'rows'}), flush=True)
        finally:
            source.close()
    benefit = confidence([w['benefit'] for w in worlds])
    hold = confidence([w['hold_benefit'] for w in worlds])
    pain = confidence([w['pain_change'] for w in worlds])
    replay = max((w['replay_error'] for w in worlds), default=float('inf'))
    pain_replay = max((w['pain_replay_error'] for w in worlds), default=float('inf'))
    passed = all(x['valid'] for x in (benefit, hold, pain)) and benefit['lower'] > replay and hold['lower'] > replay and pain['upper'] <= pain_replay
    report = dict(gate='PASS' if passed else 'FAIL', benefit_vs_frozen_parent_planner=benefit,
                  benefit_vs_hold=hold, change_in_worst_pain=pain, replay_error=replay, pain_replay_error=pain_replay,
                  interpretation='Isolated copied model; this does not certify production or child-level development.')
    write_json(out / 'behavior_report.json', report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--speech', type=Path, default=Path('Saved/mimo_bridge/caregiver_speech'))
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--worlds', type=int, default=6)
    ap.add_argument('--eval-seed', type=int, default=9201)
    ap.add_argument('--decisions', type=int, default=12)
    args = ap.parse_args(argv)
    if args.steps < 0 or args.worlds < 2 or args.decisions < 1:
        ap.error('need nonnegative steps, at least two worlds and a decision per world')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    overrides = dict(contact_learning='on', consequence_horizon=5, audio='real', audio_freq_bands='on')
    declaration = dict(checkpoint=str(args.checkpoint.resolve()), overrides=overrides,
                       train_steps=args.steps, held_out_seeds=list(range(args.eval_seed, args.eval_seed + args.worlds)),
                       decisions_per_world=args.decisions, production_changes=False,
                       execution_modes=['normal', 'isolated'], comparator='frozen parent planner and achieved-pose hold',
                       gate='World-paired 95% lower bounds beat planner and hold above replay error; upper bound of extra worst pain <= replay error.',
                       consequence='Failed gates retain the copied model and results, never promote its action selector.')
    declaration['cadence'] = 'Body command held for the native motor_hold interval during fitting and testing; both brains observe every action_repeat interval.'
    write_json(args.out / 'declaration.json', declaration)
    repo = Path(__file__).resolve().parents[4]
    sources = sorted((repo / 'Source/Python/lem/model').rglob('*.py')) + sorted((repo / 'Source/Python/mimo_bridge').glob('*.py')) + [Path(__file__)]
    write_json(args.out / 'source_hashes.json', {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    model, a, _, _ = _build_model(str(args.checkpoint), args.device, 701, intention=True, arg_overrides=overrides)
    load = model.load_checkpoint(str(args.checkpoint), load_optimizer=True)
    write_json(args.out / 'load_report.json', load)
    write_json(args.out / 'migration_verified.json', verify_migration(args.checkpoint, model))
    a = dict(a, **overrides)
    a['_trial_start_step'] = int(load['step'])
    write_json(args.out / 'args.json', a)
    model.checkpoints = CheckpointManager(model.config, str(args.out / 'checkpoints'))
    dev, limits_source = _limits_for_run(str(args.checkpoint), model)
    write_json(args.out / 'limits.json', dict(values=dev, source=limits_source))
    a['_trial_phase'] = dev.get('phase_name')
    write_json(args.out / 'args.json', a)
    repeats = max(1, int(a.get('action_repeat') or 5))
    source = environment(args.out, 701, args.speech, a)
    try:
        frames = source.reset()
        command = None
        held_command = None
        previous_order = None
        hold_steps = max(1, int(a.get('motor_hold') or 5))
        for step in range(args.steps):
            result = model.step(frames, dev, train=True, step_index=int(load['step']) + step,
                                executed_action=command, executed_order_id=previous_order)
            if held_command is None or step % hold_steps == 0:
                options = choices(model, result, dev, 701000 + step, 1)
                order = step // hold_steps
                # Include the native policy, physical hold, and exploration with
                # the SAME duration subsequently forecast by the choice test.
                pick = (1, 2, 0, 3)[order % 4]
                held_command = options[pick, :, 0].detach().cpu().numpy().copy()
                previous_order = order
            command = held_command
            frames = advance(source, command, result.gaze_out, repeats)
            command = source.last_executed
            model._prev_mfa = torch.as_tensor(command, device=model.device).clone()
            if step % 20 == 0:
                record = dict(step=step, loss=result.loss, consequence=model.consequence_lesson.stats,
                              sim_time=frames[0].sim_time, timestamp=time.time(), body_order_id=previous_order,
                              held_for_steps=hold_steps)
                write_json(args.out / 'status.json', record)
                print(json.dumps(record), flush=True)
            if (step + 1) % 200 == 0:
                model.save_checkpoint(int(load['step']) + step + 1)
            if (step + 1) % 2000 == 0:
                frames = source.reset()
                model.reset_transition_state()
                command = None
                held_command = None
                previous_order = None
        model.save_checkpoint(int(load['step']) + args.steps)
        write_json(args.out / 'caregiver_events.json', source.env.caregiver.state['events'])
    finally:
        source.close()
    baseline, _, _, _ = _build_model(str(args.checkpoint), args.device, 702, intention=True)
    a['_evaluation_checkpoint'] = str(args.out / 'checkpoints' / f"ckpt_step_{int(load['step']) + args.steps}")
    a['_parent_checkpoint'] = str(args.checkpoint)
    reports = {}
    for mode in ('normal', 'isolated'):
        reports[mode] = evaluate(model, baseline, args.out / mode, args.speech, a, dev,
                        declaration['held_out_seeds'], args.decisions, 5, repeats, normal=mode == 'normal')
    report = dict(gate='PASS' if all(r['gate'] == 'PASS' for r in reports.values()) else 'FAIL',
                  execution_modes=reports, source_checkpoint=str(args.checkpoint.resolve()))
    write_json(args.out / 'behavior_report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
