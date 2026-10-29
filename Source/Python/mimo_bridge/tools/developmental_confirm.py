# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Confirm copied-model choices with newly constructed brains for EACH world.

Episode resets clear transitions but retain adaptive helper evidence. A frozen
weight comparison must also prevent that evidence carrying between test worlds.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import torch
from mimo_bridge.tools.causal_test import _build_model
from mimo_bridge.tools.developmental_trial import evaluate, confidence, write_json


def summarize(worlds):
    benefit = confidence([w['benefit'] for w in worlds])
    hold = confidence([w['hold_benefit'] for w in worlds])
    pain = confidence([w['pain_change'] for w in worlds])
    replay = max(w['replay_error'] for w in worlds)
    pain_replay = max(w['pain_replay_error'] for w in worlds)
    valid = all(c['valid'] for c in (benefit, hold, pain))
    passed = valid and benefit['lower'] > replay and hold['lower'] > replay and pain['upper'] <= pain_replay
    headroom = [sum(max(o['value'] for o in r['outcomes'][:-1]) - r['outcomes'][0]['value']
                    for r in w['rows']) / len(w['rows']) for w in worlds]
    return dict(gate='PASS' if passed else 'FAIL', benefit_vs_parent=benefit, benefit_vs_hold=hold,
                change_in_worst_pain=pain, value_replay_error=replay, pain_replay_error=pain_replay,
                available_candidate_headroom_by_world=headroom)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=9301)
    p.add_argument('--worlds', type=int, default=3)
    p.add_argument('--decisions', type=int, default=6)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.worlds < 2 or args.decisions < 1:
        p.error('at least two worlds and one decision are required')
    original = json.loads((args.study / 'declaration.json').read_text())
    a = json.loads((args.study / 'args.json').read_text())
    trained = args.study / 'checkpoints' / f"ckpt_step_{a['_trial_start_step'] + original['train_steps']}"
    if not (trained / 'runtime_state.pt').exists():
        raise RuntimeError('the complete declared training checkpoint is not ready')
    args.out.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.seed, args.seed + args.worlds))
    declaration = dict(parent=original['checkpoint'], copied_checkpoint=str(trained.resolve()),
                       checkpoint_sha256=hashlib.sha256((trained / 'model.safetensors').read_bytes()).hexdigest(),
                       seeds=seeds, decisions=args.decisions, gate=original['gate'],
                       reset='Construct and load both brains anew for every world and mode.',
                       modes='Normal gaze and isolated gaze; both permit choice of rest.',
                       reason='Episode resets alone retain adaptive helper evidence between worlds.')
    write_json(args.out / 'declaration.json', declaration)
    torch.set_num_threads(2)
    dev = json.loads((args.study / 'limits.json').read_text())['values']
    speech = Path(__file__).resolve().parents[4] / 'Saved/mimo_bridge/caregiver_speech'
    reports = {}
    for mode in ('normal', 'isolated'):
        worlds = []
        for seed in seeds:
            model, _, _, _ = _build_model(str(trained), args.device, seed, intention=True)
            baseline, _, _, _ = _build_model(original['checkpoint'], args.device, seed, intention=True)
            dest = args.out / mode / str(seed)
            evaluate(model, baseline, dest, speech, a, dev, [seed], args.decisions,
                     int(a['consequence_horizon']), int(a['action_repeat']), normal=mode == 'normal')
            worlds.append(json.loads((dest / f'world_{seed}.json').read_text()))
            del model, baseline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        reports[mode] = summarize(worlds)
        write_json(args.out / mode / 'report.json', reports[mode])
    report = dict(gate='PASS' if all(r['gate'] == 'PASS' for r in reports.values()) else 'FAIL', modes=reports)
    write_json(args.out / 'behavior_report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
