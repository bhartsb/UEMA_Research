# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Fresh physical worlds for a frozen local-function prediction candidate."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from mimo_bridge.train_mimo import build_learner
from .self_recovery_experiment import configuration,episode
from .self_grounded_dynamics_trial import measure
from .recorded_limits import recorded_limits
from .training_continuation import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed-start',type=int,default=8300);parser.add_argument('--worlds',type=int,default=8)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--limits-from',type=Path,help='Explicit prior declaration containing measured physical limits')
    args=parser.parse_args()
    if args.worlds<1 or args.steps<1:parser.error('world and horizon counts must be positive')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def status(stage,**kw):
        row=dict(stage=stage,timestamp=time.time(),pid=os.getpid(),**kw)
        atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
    seeds=list(range(args.seed_start,args.seed_start+args.worlds))
    declaration=dict(parent=str(args.checkpoint.resolve()),seeds=seeds,candidates=8,steps=args.steps,
        checkpoint_weights_sha256=hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        code_sha256_at_launch=code_hash,
        test='Identical full physical initial state restored before each actual command sequence; repeat-hold exact and source-removal control. Fresh world construction, no fitting.',
        gate='At least six fresh worlds, each removes75% of measured action-effect squared error. Per-world healthy-site availability error below half the measured RMS action-effect separation. Predicted errors, measured choices and recovery opportunity reported separately.',
        healthy_error_basis='Persistence is exactly correct on untouched healthy sites. Requiring literal zero error would reject any anticipatory false positive. The maximum allowed healthy error is derived from the same actual action separation as the forecast-resolution requirement, not chosen after these worlds.',
        excluded='No production chooser license: physical correction, completion, repeated learning/acting and retention are separate tests. No future simulator state enters the candidate.',
        initialization='Current physical function is observed, not a learned achievement. New recurrent module forecasts future changes using full executed sequences.')
    atomic_json(out/'declaration.json',declaration);status('loading')
    config=configuration(args.checkpoint,out);model=build_learner(config,str(out/'unused_checkpoints'))
    loaded=model.load_checkpoint(str(args.checkpoint),load_optimizer=False)
    if loaded['reinitialized_components'] or model.self_state.prediction_mode!='grounded_residual':
        raise ValueError('confirmation requires the complete frozen grounded candidate')
    limits,limits_source=recorded_limits(args.checkpoint,model,args.limits_from)
    declaration.update(limits=limits,limits_source=limits_source)
    atomic_json(out/'declaration.json',declaration)
    baseline=deepcopy(model.self_state).eval()
    with torch.no_grad():baseline.grounded.change.weight.zero_();baseline.grounded.change.bias.zero_()
    rows=[]
    for index,seed in enumerate(seeds):
        folder=out/str(seed);folder.mkdir();status('testing_fresh_physical_worlds',step=index,steps=len(seeds),seed=seed)
        physical,sample=episode(model,config,limits,folder,seed,8,args.steps,config.context_length,args.checkpoint,baseline)
        row=dict(seed=seed,physical=physical,measurement=measure(model.self_state,sample));rows.append(row)
        atomic_json(out/'partial_results.json',rows)
    passed=len(rows)>=6 and all(row['measurement']['action_prediction']['valid']
                and row['measurement']['action_prediction']['action_error_removed']>=.75
                and (row['measurement']['forecast']['healthy_mae'] is None or
                     row['measurement']['forecast']['healthy_mae']<=.5*np.sqrt(row['measurement']['action_prediction']['action_blind_error']))
                for row in rows)
    report=dict(verdict='FRESH_FUNCTION_FORECAST_PASSED' if passed else 'FRESH_FUNCTION_FORECAST_FAILED',
                worlds=rows,production_license=False,completed_at=time.time(),
                code_sha256=code_hash,source_changed_during_run=code_hash!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(out/'report.json',report);status('complete',verdict=report['verdict'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        folder=Path(sys.argv[sys.argv.index('--out')+1])
        if folder.exists():
            atomic_json(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
            atomic_json(folder/'status.json',dict(stage='failed',timestamp=time.time(),error=str(error)))
        raise
