# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Fit the checkpoint-compatible local dynamics extension on recorded physics.

Old self and brain weights are frozen. Development learning and reserved-world
measurements are separate. Current observed function is not credited as learned
prediction; only actual future changes can earn improvement over persistence.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from lem.model.self_state import SelfState
from mimo_bridge.train_mimo import build_learner
from .self_recovery_experiment import configuration, action_prediction_result, recovery_cost
from .training_continuation import atomic_json


def broadcast(percept,count):
    return {key:value.expand(count,*value.shape[1:]) for key,value in percept.items()}


def predict(module,sample):
    percept=module.perceive(**sample['inputs'])
    mean,variance=module.forecast(broadcast(percept,len(sample['actions'])),sample['actions'],sample['durations'])
    return percept,mean,variance


def measure(module,sample):
    with torch.no_grad():percept,mean,variance=predict(module,sample)
    truth=sample['targets'];valid=sample['valid'][None,None].expand(*truth.shape[:-1])
    initial=sample['start'][None,None].expand_as(truth)
    affected=(truth[...,0]<1-8*torch.finfo(truth.dtype).eps) & valid
    healthy=(~affected) & valid
    result={}
    for name,pred in [('forecast',mean),('persistence',initial)]:
        error=(pred-truth).abs()
        result[name]=dict(affected_mae=float(error[...,0][affected].mean()) if affected.any() else None,
                         healthy_mae=float(error[...,0][healthy].mean()) if healthy.any() else None,
                         worst_error=float(error[valid].max()),all_fields_mae=float(error[valid].mean()))
    result['action_prediction']=action_prediction_result(truth.numpy(),mean.numpy(),sample['valid'])
    costs=[recovery_cost(x,sample['valid']) for x in truth.numpy()]
    predicted=[recovery_cost(x,sample['valid']) for x in mean.numpy()]
    chosen=int(np.argmin(predicted));best=int(np.argmin(costs))
    result['choice']=dict(costs=costs,predicted=predicted,selected=chosen,best=best,
                         regret=costs[chosen]-costs[best],gain_over_planner=costs[0]-costs[chosen],
                         available_over_planner=costs[0]-costs[best])
    result['uncertainty']=dict(one_sigma_coverage=float(((mean-truth).abs()<=variance.sqrt())[valid].float().mean()),
                              mean_sigma=float(variance.sqrt()[valid].mean()))
    result['observed_present_max_error']=float((percept['present'][0]-sample['start'])[sample['valid']].abs().max())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=480);parser.add_argument('--learning-rate',type=float,default=.0003)
    parser.add_argument('--paired-contrast',action='store_true')
    parser.add_argument('--development-only',action='store_true')
    parser.add_argument('--calibrate-actions',action='store_true')
    parser.add_argument('--horizon-steps',type=int,default=100)
    args=parser.parse_args()
    if args.updates<1 or not 0<args.learning_rate<1 or not 1<=args.horizon_steps<=100:parser.error('invalid update budget, horizon or learning rate')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(61973)
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def status(stage,**kw):
        row=dict(stage=stage,pid=os.getpid(),timestamp=time.time(),**kw)
        atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
    study=json.loads((args.study/'declaration.json').read_text());source=json.loads((args.study/'report.json').read_text())
    development=study['development_seeds'];reserved=study['held_out_seeds']
    declaration=dict(parent=source['checkpoint'],development_seeds=development,reserved_seeds=reserved,
        updates=args.updates,learning_rate=args.learning_rate,
        paired_contrast=args.paired_contrast,development_only=args.development_only,calibrate_actions=args.calibrate_actions,
        code_sha256_at_launch=code_hash,
        horizon_steps=args.horizon_steps,
        protocol='Fit only newly appended grounded dynamics. Original self and old brain weights unchanged. All actual commands, duration and previous sensory state come from complete recorded physical branches.',
        criterion='Each reserved world must remove at least 75% of action-effect squared error (prediction error under half the RMS physical action difference), and affected and healthy availability MAE must each be no worse than persistence. This qualifies a forecast component only.',
        criterion_basis='A positive reduction alone retained over 99.8% of action-effect error in the earlier study. Half the measured RMS action separation is the chosen resolution requirement; report actual choice regret separately rather than infer purpose from this scale.',
        no_adoption='No production choice license. Original reserved worlds have been measured before; this is a versioned regression comparison, not untouched confirmation or learned intention.',
        prediction_source='Current physical function remains explicitly observed. A learned recurrent residual predicts future changes from the existing self belief, local encoding, every executed action and elapsed seconds.',
        scale='Availability MAE: 0 exact, 1 full index-range wrong. Skin index 0 minimum 2-bit precision and 1 original sensing; motor index 0 no force gain and 1 original gain.')
    atomic_json(out/'declaration.json',declaration);status('loading')
    config=configuration(Path(source['checkpoint']),out);config.self_prediction='grounded_residual'
    model=build_learner(config,str(out/'checkpoints'));loaded=model.load_checkpoint(source['checkpoint'])
    if loaded['reinitialized_components']:raise RuntimeError('Original checkpoint component did not load')
    module=model.self_state
    if module.prediction_mode!='grounded_residual':raise RuntimeError('Native builder did not select grounded dynamics')
    old={key:value.clone() for key,value in module.state_dict().items() if not key.startswith('grounded.')}
    # Existing optimizer remains untouched; this temporary optimizer trains only
    # new parameters. Its moments are transferred by exact parameter identity
    # back to the copied full checkpoint after the experiment.
    for name,param in module.named_parameters():param.requires_grad_(name.startswith('grounded.'))
    opt=torch.optim.AdamW(module.grounded.parameters(),lr=args.learning_rate,weight_decay=.01)
    def sample_for(seed):
        sample=torch.load(args.study/str(seed)/'branches.pt',map_location='cpu',weights_only=False)
        for key in ('actions','durations','targets'):sample[key]=sample[key][:,:args.horizon_steps]
        return sample
    samples=[sample_for(seed) for seed in development]
    if args.calibrate_actions:
        commands=torch.cat([sample['actions'].reshape(-1,module.action_size) for sample in samples])
        module.grounded.calibrate_actions(commands)
        atomic_json(out/'action_calibration.json',dict(center=module.grounded.action_center.tolist(),
            scale=module.grounded.action_scale.tolist(),samples=len(commands),source='development recorded executed orders only'))
    # Cache only frozen percepts; raw inputs remain saved in the original study.
    cached=[]
    with torch.no_grad():
        for sample in samples:cached.append({key:value.detach() for key,value in module.perceive(**sample['inputs']).items()})
    for update in range(args.updates):
        index=update%len(samples);sample=samples[index];count=len(sample['actions'])
        mean,var=module.forecast(broadcast(cached[index],count),sample['actions'],sample['durations'])
        valid=sample['valid'][None,None].expand(*mean.shape[:-1])
        initial=sample['start'][None].expand(count,-1,-1)
        loss=(module.paired_consequence_loss(mean,var,sample['targets'],valid) if args.paired_contrast
              else module.learning_loss(mean,var,sample['targets'],valid,initial=initial))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite future-function lesson')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(module.grounded.parameters(),1.)
        opt.step()
        if update%16==0 or update+1==args.updates:
            status('fitting_grounded_dynamics',step=update+1,steps=args.updates,loss=float(loss.detach()))
    if not all(torch.equal(value,module.state_dict()[key]) for key,value in old.items()):
        raise RuntimeError('Existing learned self weights changed')
    for param in module.grounded.parameters():model.optimizer.state[param]=deepcopy(opt.state[param])
    for param in module.parameters():param.requires_grad_(True)
    model.step_count+=args.updates
    checkpoint=model.save_checkpoint(model.step_count)
    atomic_json(out/'args.json',dict(vars(config),_copied_grounded_updates=args.updates))
    rows=[]
    tested=development if args.development_only else reserved
    for index,seed in enumerate(tested):
        sample=sample_for(seed)
        row=dict(seed=seed,**measure(module,sample));rows.append(row)
        status('testing_development_branches' if args.development_only else 'testing_reserved_physical_branches',step=index+1,steps=len(tested))
    passed=all(row['action_prediction']['valid'] and row['action_prediction']['action_error_removed']>=.75
               and all(row['forecast'][field]<=row['persistence'][field] for field in ('affected_mae','healthy_mae')
                       if row['persistence'][field] is not None) for row in rows)
    report=dict(verdict=('DEVELOPMENT_FIT_MEASURED' if args.development_only else
                        'PRACTICAL_FUNCTION_FORECAST_GAIN' if passed else 'FUNCTION_FORECAST_REPAIR_REQUIRED'),
                worlds=rows,checkpoint=checkpoint,original_self_weights_exact=True,production_license=False,
                current_grounding='Observed input retained, not learned capability.',completed_at=time.time(),
                code_sha256=code_hash,source_changed_during_run=code_hash!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(out/'report.json',report);status('complete',verdict=report['verdict'],production_license=False)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        if '--out' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--out')+1])
            if folder.exists():
                atomic_json(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
                atomic_json(folder/'status.json',dict(stage='failed',timestamp=time.time(),error=str(error)))
        raise
