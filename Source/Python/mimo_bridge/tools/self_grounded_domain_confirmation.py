# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Frozen domain-separated forecasts on fresh physical branches as they finish."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from mimo_bridge.train_mimo import build_learner
from .self_recovery_experiment import configuration,action_prediction_result
from .self_grounded_dynamics_trial import predict
from .training_continuation import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--branches',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--steps',type=int,default=20)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source=json.loads((args.branches/'declaration.json').read_text());seeds=source['seeds']
    declaration=dict(parent=str(args.checkpoint.resolve()),physical_source=str(args.branches.resolve()),seeds=seeds,steps=args.steps,
        candidate_sha256=hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        criterion='Separately for skin precision and motor force capacity: each of at least six fresh worlds removes75% of action-effect squared error, meaning RMS error below half the measured between-action separation. Raw index/force-gain errors and uncertainty reported. No pooled score may hide a failed domain.',
        rationale='Development findings located futurecontact failure under a working candidate for motor-effort consequences. Domain hypothesis and frozen candidate registered before reading these fresh outcome branches. No fitting occurs.',
        scope='A domain pass certifies that measured forecast component only. A skin-precision index is not wellness and a motor-capacity prediction is not a purposeful action. No production choice license.')
    atomic_json(out/'declaration.json',declaration)
    model=build_learner(configuration(args.checkpoint,out),str(out/'unused_checkpoints'))
    model.load_checkpoint(str(args.checkpoint),load_optimizer=False);rows=[]
    for index,seed in enumerate(seeds):
        while True:
            status=dict(stage='waiting_for_physical_branches',timestamp=time.time(),pid=os.getpid(),step=index,steps=len(seeds),seed=seed)
            atomic_json(out/'status.json',status)
            producer=json.loads((args.branches/'status.json').read_text())
            if producer['stage']=='failed':raise RuntimeError('Physical producer failed; no synthetic replacement branch')
            partial=args.branches/'partial_results.json'
            ready=[] if not partial.exists() else json.loads(partial.read_text())
            if any(row['seed']==seed for row in ready):break
            if producer['stage']=='complete':raise RuntimeError('Producer completed without required branch')
            time.sleep(5)
        sample=torch.load(args.branches/str(seed)/'branches.pt',map_location='cpu',weights_only=False)
        for key in ('actions','durations','targets'):sample[key]=sample[key][:,:args.steps]
        with torch.no_grad():percept,mean,variance=predict(model.self_state,sample)
        row=dict(seed=seed)
        for kind in ('skin','motor'):
            valid=sample['valid'].clone()
            if kind=='skin':valid[-10:]=False
            else:valid[:-10]=False
            action=action_prediction_result(sample['targets'].numpy(),mean.numpy(),valid)
            error=(mean-sample['targets']).abs()[...,valid,0]
            sigma=variance.sqrt()[...,valid,0]
            row[kind]=dict(**action,mean_absolute_error=float(error.mean()),maximum_absolute_error=float(error.max()),
                           one_sigma_coverage=float((error<=sigma).float().mean()),mean_sigma=float(sigma.mean()))
        rows.append(row);atomic_json(out/'partial_results.json',rows)
    verdicts={kind:(len(rows)>=6 and all(row[kind]['valid'] and row[kind]['action_error_removed']>=.75 for row in rows))
              for kind in ('skin','motor')}
    atomic_json(out/'report.json',dict(worlds=rows,domain_passes=verdicts,production_license=False,
        verdict='DOMAIN_FORECASTS_MEASURED',completed_at=time.time()))
    atomic_json(out/'status.json',dict(stage='complete',timestamp=time.time(),pid=os.getpid(),verdict='DOMAIN_FORECASTS_MEASURED'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        out=Path(sys.argv[sys.argv.index('--out')+1])
        if out.exists():
            atomic_json(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
            atomic_json(out/'status.json',dict(stage='failed',timestamp=time.time(),error=str(error)))
        raise
