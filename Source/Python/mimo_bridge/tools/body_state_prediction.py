# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Offline physical prediction with explicit state/action interactions.

Raw sensory measurements only. Hyperparameters are chosen on training worlds;
the held-out world gate never controls fitting. This does not alter LemLite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

LAMBDAS = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)
WIDTHS = (.25, 1., 4.)


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def extract(rows, horizon):
    """N states, A candidates, H physical body points; no future input fields."""
    states, commands, targets, velocities, floors = [], [], [], [], []
    for row in rows:
        bones = tensor(row['physical_start']['bones'])
        pos, quat, vel = bones[:, 4:7], bones[:, :4], bones[:, 7:10]
        # Canonicalize quaternion signs without changing rotations.
        quat = quat * torch.where(quat[:, 3:] < 0, -1., 1.)
        state = torch.cat([quat, (pos - pos.mean(0)) / 200., vel / 200., bones[:, 10:11]], -1)
        cmds = tensor(row['commands']).clone()
        cmds[..., :3] = (cmds[..., :3] - pos[None]) / 200.
        cmds[..., 3:7] *= torch.where(cmds[..., 6:7] < 0, -1., 1.)
        if 'physical_paths' in row:
            index = round(horizon / .05) - 1
            end = torch.stack([tensor(row['physical_paths'][a][index]['bones'])[:, 4:7] for a in range(4)])
            replay = tensor(row['physical_paths'][4][index]['bones'])[:, 4:7]
            floors.append(float((end[0] - replay).square().sum(-1).mean()))
        elif horizon == .05:
            end = torch.stack([tensor(e['bones'])[:, 4:7] for e in row['physical_end']])
            floors.append(float('nan'))  # Historical raw replay was not retained.
        else:
            raise ValueError('Cannot assign a different horizon to historical endpoints')
        states.append(state.flatten())
        commands.append(cmds.flatten(1))
        targets.append((end - pos[None]).flatten(1))
        velocities.append(vel.flatten())
    if not rows:
        raise ValueError('No physical observations')
    result = dict(s=torch.stack(states), c=torch.stack(commands), y=torch.stack(targets),
                  v=torch.stack(velocities), floor=np.array(floors),
                  worlds=np.array([r['world'] for r in rows]), horizon=horizon)
    if not all(torch.isfinite(result[k]).all() for k in ('s', 'c', 'y', 'v')):
        raise ValueError('Non-finite physical measurements')
    return result


def subset(data, indices):
    indices = np.asarray(indices)
    return {k: v[indices] if k != 'horizon' else v for k, v in data.items()}


def effects(x):
    i, j = torch.triu_indices(x.shape[1], x.shape[1], 1)
    return x[:, i] - x[:, j]


def body_points(x):
    """[..., bodies * 3] -> [-1, bodies, 3], the body count read from the array (24 legacy, 26
    trunk_v1). Until 2026-09-23 this was a fixed 24, which on trunk_v1 records ran each sample's
    rows into the next sample's (fix-following-gauge)."""
    if x.shape[-1] % 3:
        raise ValueError(f'{x.shape[-1]} numbers per record do not make whole body points of 3')
    return x.reshape(-1, x.shape[-1] // 3, 3)


def wrist_rows(x):
    """The two wrists' rows, found by name in the layout that has x's body count."""
    from mimo_bridge.body_layout import names_for_rows
    names = names_for_rows(x.shape[-1] // 3)
    return [names.index('wrist_L'), names.index('wrist_R')]


def score(pred, y):
    if pred.shape != y.shape or y.numel() == 0 or y.shape[1] < 2:
        raise ValueError('Missing or mismatched candidate outcomes')
    if not torch.isfinite(pred).all() or not torch.isfinite(y).all():
        raise ValueError('Non-finite predictions or outcomes')
    true = effects(y)
    den = float(true.square().sum())
    if den <= 0:
        raise ValueError('No physical action effect')
    return float((effects(pred) - true).square().sum()) / den


def normalizer(x):
    flat = x.reshape(-1, x.shape[-1])
    mean, scale = flat.mean(0), flat.std(0, unbiased=False)
    # Constant dimensions are removed rather than amplifying numerical jitter.
    keep = scale > 1e-6
    if not keep.any():
        raise ValueError('No varying features')
    return mean, scale, keep


def normalized(x, norm):
    mean, scale, keep = norm
    return ((x - mean)[..., keep] / scale[keep])


def squared_dist(a, b):
    return torch.cdist(a, b).square() / a.shape[-1]


def positive_median(dist):
    values = dist[dist > 1e-12]
    return float(values.median()) if values.numel() else 1.


def fit(data, family, lam, width, shuffle_seed=None):
    s, c, y = data['s'], data['c'].clone(), data['y']
    if shuffle_seed is not None:
        rng = torch.Generator().manual_seed(shuffle_seed)
        c = torch.stack([arm[torch.randperm(len(arm), generator=rng)] for arm in c])
    sn, cn = normalizer(s), normalizer(c)
    ss, cc = normalized(s, sn), normalized(c, cn).flatten(0, 1)
    ss = ss.repeat_interleave(c.shape[1], 0)
    ds, dc = squared_dist(ss, ss), squared_dist(cc, cc)
    model = dict(sn=sn, cn=cn, s=ss, c=cc, family=family, width=width, lam=lam,
                 ds_scale=positive_median(ds), dc_scale=positive_median(dc), actions=c.shape[1])
    k = kernel(model, ss, cc)
    target = y.flatten(0, 1)
    # A common displacement bias; constant terms cancel in action-effect scoring.
    model['mean'] = target.mean(0)
    model['alpha'] = torch.linalg.solve(k + lam * torch.eye(len(k), dtype=k.dtype), target-model['mean'])
    if not torch.isfinite(model['alpha']).all():
        raise ValueError('Non-finite fit')
    return model


def kernel(model, s, c):
    family, width = model['family'], model['width']
    if family == 'linear':
        return s @ model['s'].T / s.shape[-1] + c @ model['c'].T / c.shape[-1]
    kc = torch.exp(-squared_dist(c, model['c']) / model['dc_scale'])
    if family == 'command_kernel':
        return kc
    if family != 'state_action_kernel':
        raise ValueError(f'Unknown family {family}')
    return kc * torch.exp(-squared_dist(s, model['s']) / (model['ds_scale'] * width))


def predict(model, data):
    n, a = data['c'].shape[:2]
    ss = normalized(data['s'], model['sn']).repeat_interleave(a, 0)
    cc = normalized(data['c'], model['cn']).flatten(0, 1)
    return (kernel(model, ss, cc) @ model['alpha'] + model['mean']).reshape(n, a, -1)


def select(data, family):
    worlds = sorted(set(data['worlds']))
    if len(worlds) < 3:
        raise ValueError('Need three training worlds for model selection')
    grid = []
    for width in (WIDTHS if family == 'state_action_kernel' else (1.,)):
        for lam in LAMBDAS:
            errors = []
            for world in worlds:
                tr = subset(data, data['worlds'] != world)
                va = subset(data, data['worlds'] == world)
                errors.append(score(predict(fit(tr, family, lam, width), va), va['y']))
            grid.append(dict(lam=lam, width=width, mean=float(np.mean(errors)), per_world=errors))
    chosen = min(grid, key=lambda r: r['mean'])
    return fit(data, family, chosen['lam'], chosen['width']), dict(chosen=chosen, grid=grid)


def measurements(model, data, require_replay=True):
    pred, y = predict(model, data), data['y']
    de, error = effects(y), effects(pred) - effects(y)
    physical = float(body_points(de).square().sum(-1).mean())
    floor = data['floor']
    if require_replay and (not np.isfinite(floor).all() or physical <= 10 * float(floor.max())):
        raise ValueError('Physical replay does not resolve the action effects')
    persistence = float(y.square().sum())
    velocity_prediction = data['v'][:, None] * data['horizon']
    return dict(effect_error_ratio=score(pred, y),
        effect_rms_cm_per_body_point=float(np.sqrt(physical)),
        error_rms_cm_per_body_point=float(body_points(error).square().sum(-1).mean().sqrt()),
        effect_rms_cm_per_wrist=float(body_points(de)[:, wrist_rows(de)].square().sum(-1).mean().sqrt()),
        error_rms_cm_per_wrist=float(body_points(error)[:, wrist_rows(error)].square().sum(-1).mean().sqrt()),
        point_error_vs_persistence=float((pred-y).square().sum())/persistence,
        velocity_error_vs_persistence=float((velocity_prediction-y).square().sum())/persistence,
        physical_replay_max_rms_cm=float(np.sqrt(floor.max())) if np.isfinite(floor).all() else None)


def gate(worlds, results, shuffled):
    if not worlds:
        raise ValueError('Empty evaluation cannot pass')
    checks = []
    for w in worlds:
        candidate = results['state_action_kernel']['heldout'][str(w)]['effect_error_ratio']
        controls = [results[k]['heldout'][str(w)]['effect_error_ratio'] for k in ('linear','command_kernel')]
        null = shuffled[str(w)]
        if len(null) != 8 or not np.isfinite([candidate, *controls, *null]).all():
            raise ValueError('Invalid controls cannot pass')
        checks.append(candidate < min(1., *controls, *null))
    return bool(all(checks))


def fingerprint(data):
    digest = hashlib.sha256(str(data['horizon']).encode())
    for key in ('s', 'c', 'y', 'worlds'):
        value = data[key].numpy() if torch.is_tensor(data[key]) else data[key]
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--horizon', type=float, choices=(.05,.25), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--historical', type=Path)
    ap.add_argument('--prepared', type=Path, help='Already fitted training-only models; fingerprint checked')
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    declaration = json.loads((args.data/'declaration.json').read_text())
    rows = torch.load(args.data/'paired_frames.pt', weights_only=False)
    if len(rows) != declaration['worlds'] * declaration['events_per_world']:
        raise ValueError('Incomplete declared collection')
    for world in range(declaration['worlds']):
        rr = [r for r in rows if r['world'] == world]
        if sorted(r['event'] for r in rr) != list(range(declaration['events_per_world'])):
            raise ValueError('Duplicate or missing events')
        expected_stride = declaration['stride_model_steps'] * .05
        if not np.allclose(np.diff([r['sim_time'] for r in rr]), expected_stride, rtol=0,atol=1e-7):
            raise ValueError('Trajectory clock differs from declaration')
    train = [r for r in rows if r['world'] in declaration['train_worlds']]
    test = [r for r in rows if r['world'] in declaration['heldout_worlds']]
    if not train or not test or set(declaration['train_worlds']) & set(declaration['heldout_worlds']):
        raise ValueError('Invalid train/evaluation split')
    if args.historical:
        if args.horizon != .05:
            raise ValueError('Historical collection only valid here at 50 ms')
        old = torch.load(args.historical/'paired_frames.pt', weights_only=False)
        train += [dict(r, world=1000+r['world']) for r in old if r['world'] < 6]
    tr, te = extract(train,args.horizon), extract(test,args.horizon)
    prepared = None
    if args.prepared:
        prepared = torch.load(args.prepared,weights_only=False)
        if prepared['training_fingerprint'] != fingerprint(tr):
            raise ValueError('Prepared model was fitted to different training data')
        if prepared['source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('Fitting source changed after preparing models')
    worlds = sorted(set(te['worlds']))
    out = dict(horizon_seconds=args.horizon, train_states=len(train), test_states=len(test),
               data_sha256=hashlib.sha256((args.data/'paired_frames.pt').read_bytes()).hexdigest(),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results={},
               meaning='Effect squared-error ratio: 1 predicts no action differences; 0 exact. Not a live skill test.')
    for family in ('linear','command_kernel','state_action_kernel'):
        mdl, selection = prepared['models'][family] if prepared else select(tr, family)
        result = dict(selection=selection, train_effect_error_ratio=score(predict(mdl,tr),tr['y']), heldout={})
        for world in worlds:
            result['heldout'][str(world)] = measurements(mdl,subset(te,te['worlds']==world))
        torch.save(mdl,args.out/f'{family}.pt')
        out['results'][family] = result
        print(json.dumps(dict(family=family,selection=selection['chosen'], heldout=result['heldout'])),flush=True)
    chosen = out['results']['state_action_kernel']['selection']['chosen']
    null = {str(w): [] for w in worlds}
    for seed in range(8):
        mdl = fit(tr,'state_action_kernel',chosen['lam'],chosen['width'],shuffle_seed=99100+seed)
        for world in worlds:
            wt = subset(te,te['worlds']==world)
            null[str(world)].append(score(predict(mdl,wt),wt['y']))
    out['shuffled'] = null
    out['gate_passed'] = gate(worlds,out['results'],null)
    out['production_changed'] = False
    (args.out/'result.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(gate_passed=out['gate_passed'],shuffled=null)),flush=True)


if __name__ == '__main__':
    main()
