# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Evaluate a frozen sequence predictor on independent worlds; never refit it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mimo_bridge.tools.body_sequence_fit import (
    extract_sequence, final_positions, absolute_scores, prediction_gate, world_select_fit)
from mimo_bridge.tools.body_state_prediction import predict, fit, subset, score, effects

MODEL_SHA = 'bfbf23ff6b827ffacf29f28e3727da08bde09738b42c675c014affe40de2345b'
TRAINING = ('body_sequence_study_2026-09-10', 'body_sequence_extra_2026-09-10',
            'body_sequence_extra2_2026-09-10')
SAVED = Path('Saved/mimo_bridge')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def training_data():
    rows, sources = [], []
    for index, name in enumerate(TRAINING):
        folder = SAVED/name
        rr = torch.load(folder/'paired_frames.pt',weights_only=False)
        dec = json.loads((folder/'declaration.json').read_text())
        if len(rr) != dec['worlds']*dec['events_per_world']:
            raise ValueError('Incomplete training source')
        keys = [(r['world'],r['event']) for r in rr]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate training examples')
        sources.append(dict(path=str(folder),data_sha256=sha(folder/'paired_frames.pt'),declaration=dec))
        rows += [dict(r,world=r['world']+100*index) for r in rr if index or r['world']<6]
    if len(rows) != 384:
        raise ValueError('Frozen model requires its original 384 training states')
    return rows,sources


def without_history(data):
    # Every history entry contributes 24 * (delta-position3 + velocity3 +
    # relative-command3 + strength1) = 240 columns, for five entries.
    return dict(data,s=data['s'][:,:-1200])


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    model_path = SAVED/TRAINING[0]/'sequence_models.pt'
    if sha(model_path) != MODEL_SHA:
        raise ValueError('Frozen model changed')
    frozen = torch.load(model_path,weights_only=False)
    rows,sources = training_data()
    data = extract_sequence(rows)
    worlds = sorted(set(data['worlds'].tolist()))
    controls = dict(frozen=frozen,model_sha256=MODEL_SHA,training_sources=sources,shuffled=[])
    # Preserve exact original weights in the control bundle before evaluation.
    torch.save(controls,out/'in_progress.pt')
    for seed in range(700,708):
        controls['shuffled'].append(fit(data,'state_action_kernel',.1,2.,shuffle_seed=seed))
        print(json.dumps(dict(shuffled_control_fitted=seed)),flush=True)
    ablation,hp,cv = world_select_fit(without_history(data),worlds,'state_action_kernel')
    controls['without_history'] = ablation
    controls['without_history_selection'] = dict(lambda_=hp[0],width=hp[1],training_cv=cv)
    torch.save(controls,out/'controls.pt')
    write(out/'prepared.json',dict(model_sha256=MODEL_SHA,controls_sha256=sha(out/'controls.pt'),
        training_sources=sources,ablation=controls['without_history_selection'],
        purpose='Training-only control preparation; no confirmation observations read.'))
    print(json.dumps(controls['without_history_selection']),flush=True)


def validate_confirmation(rows, declaration):
    if not declaration.get('evaluation_only') or not declaration.get('independent_streams'):
        raise ValueError('Fresh confirmation must declare evaluation-only independent streams')
    if declaration['seed_offset'] != 300 or declaration['train_worlds']:
        raise ValueError('Confirmation split or seeds changed')
    if declaration['worlds'] != 6 or declaration['events_per_world'] != 16:
        raise ValueError('Confirmation sample count changed')
    if len(rows) != 96:
        raise ValueError('Incomplete confirmation')
    for world in range(6):
        rr = [r for r in rows if r['world']==world]
        if sorted(r['event'] for r in rr) != list(range(16)):
            raise ValueError('Missing or duplicate evaluation states')
        if not np.allclose(np.diff([r['sim_time'] for r in rr]),.5,atol=1e-7,rtol=0):
            raise ValueError('Wrong trajectory cadence')
        for r in rr:
            history = r['history']
            if len(history) != 5:
                raise ValueError('Missing motion history')
            starts,ends = [],[]
            for h in history:
                start,end = h['command_start_time'],h['outcome_time']
                if not (np.isfinite([start,end,r['sim_time']]).all() and start < r['sim_time']
                        and end <= r['sim_time']+1e-7 and np.isclose(end-start,.05,atol=1e-7)):
                    raise ValueError('History contains future or mis-timed observations')
                starts.append(start); ends.append(end)
            if not np.allclose(np.diff(starts),.05,atol=1e-7) or not np.isclose(ends[-1],r['sim_time'],atol=1e-7):
                raise ValueError('History is not the preceding five command intervals')
            if set(r['physical_paths']) != set(range(5)) or any(len(p)!=5 for p in r['physical_paths'].values()):
                raise ValueError('Missing path or replay')


def describe(pred, data):
    actual = final_positions(data['y'])
    final = final_positions(pred)
    diff = effects(data['y'])
    body_error = (final-actual).reshape(-1,24,3)
    return dict(path_effect_error=score(pred,data['y']),final_effect_error=score(final,actual),
        absolute=absolute_scores(pred,data),
        error_rms_cm_per_body_point=float(body_error.square().sum(-1).mean().sqrt()),
        error_rms_cm_per_wrist=float(body_error[:,[0,6]].square().sum(-1).mean().sqrt()),
        effect_mse_cm2=float(diff.square().mean()),replay_floor_max=float(data['floor'].max()))


def evaluate(data_dir, controls_dir, out):
    out.mkdir(parents=True,exist_ok=False)
    prepared = json.loads((controls_dir/'prepared.json').read_text())
    if sha(controls_dir/'controls.pt') != prepared['controls_sha256']:
        raise ValueError('Control weights changed after preparation')
    bundle = torch.load(controls_dir/'controls.pt',weights_only=False)
    if bundle['model_sha256'] != MODEL_SHA:
        raise ValueError('Candidate changed')
    rows = torch.load(data_dir/'paired_frames.pt',weights_only=False)
    declaration = json.loads((data_dir/'declaration.json').read_text())
    validate_confirmation(rows,declaration)
    data = extract_sequence(rows)
    results,per,absolute,shuffled = {},{},{},{}
    for world in range(6):
        ev = subset(data,data['worlds']==world)
        ps = predict(bundle['frozen']['sequence'],ev)
        pl = predict(bundle['frozen']['linear'],ev)
        pn = predict(bundle['without_history'],without_history(ev))
        ds,dl,dn = describe(ps,ev),describe(pl,ev),describe(pn,ev)
        shuffled[world] = [score(predict(m,ev),ev['y']) for m in bundle['shuffled']]
        per[world] = dict(sequence=ds['path_effect_error'],linear_comparator=dl['path_effect_error'],
                          effect_mse_cm2=ds['effect_mse_cm2'],replay_floor_max=ds['replay_floor_max'])
        absolute[world] = ds['absolute']
        groups = {}
        for label,mask in [('scheduled_rest',ev['resting']),('scheduled_active',~ev['resting'])]:
            if mask.any():
                part = subset(ev,mask)
                groups[label] = dict(states=int(mask.sum()),sequence=describe(ps[mask],part),
                    linear=describe(pl[mask],part),
                    measured_start_speed_rms_cms=float(part['v'].reshape(-1,24,3).square().sum(-1).mean().sqrt()))
        results[world] = dict(sequence=ds,linear=dl,without_history=dn,subsets=groups,
            original_gate_pass=prediction_gate({world:per[world]},{world:absolute[world]},{world:shuffled[world]}),
            final_frame_comparator_pass=ds['final_effect_error'] < min(1.,dl['final_effect_error'])
                and ds['absolute']['model'] < dl['absolute']['model'])
        print(json.dumps(dict(world=world,sequence=ds['path_effect_error'],linear=dl['path_effect_error'],
            without_history=dn['path_effect_error'],gate=results[world]['original_gate_pass'])),flush=True)
    verdict = prediction_gate(per,absolute,shuffled)
    result = dict(confirmation_passed=verdict,
        final_frame_comparator_passed=all(r['final_frame_comparator_pass'] for r in results.values()),
        model_sha256=MODEL_SHA,controls_sha256=prepared['controls_sha256'],
        data_sha256=sha(data_dir/'paired_frames.pt'),collection_declaration=declaration,
        per_world=results,shuffled=shuffled,ablation=prepared['ablation'],
        scale='Action-effect squared-error ratio: 1 predicts no differences, 0 exact. Absolute MSE is cm squared per coordinate.',
        live_brain_changed=False,wellness_choice_improvement='not measured')
    write(out/'confirmation.json',result)
    print(json.dumps(dict(confirmation_passed=verdict,final_frame_comparator_passed=result['final_frame_comparator_passed'])),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('prepare','evaluate'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path)
    ap.add_argument('--controls',type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.mode=='prepare':
        prepare(args.out)
    else:
        if not args.data or not args.controls:
            ap.error('evaluate needs --data and --controls')
        evaluate(args.data,args.controls,args.out)


if __name__=='__main__':
    main()
