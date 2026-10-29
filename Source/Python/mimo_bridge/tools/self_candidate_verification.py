# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Read-only native checkpoint verification of a candidate-only transplant."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from mimo_bridge.train_mimo import build_learner
from mimo_bridge.self_candidate import transplant_grounded
from .self_recovery_experiment import configuration
from .migrate_trunk import modules
from .training_continuation import atomic_json


def digest(tensor):
    value=tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def evidence(model):
    tensors={};moments={}
    for name,module in modules(model).items():
        for key,value in module.state_dict().items():
            qualified=name+'.'+key
            if not qualified.startswith('self_state.grounded.'):tensors[qualified]=digest(value)
        for key,param in module.named_parameters():
            qualified=name+'.'+key
            if qualified.startswith('self_state.grounded.'):continue
            moments[qualified]={key:digest(value) if torch.is_tensor(value) else value
                for key,value in model.optimizer.state.get(param,{}).items()}
    return dict(tensors=tensors,optimizer=moments)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latest-checkpoint',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    config=configuration(args.latest_checkpoint,args.out);config.self_prediction='grounded_residual'
    model=build_learner(config,str(args.out/'unused_checkpoints'));loaded=model.load_checkpoint(str(args.latest_checkpoint))
    if loaded['reinitialized_components']:raise RuntimeError('latest brain did not preserve every component')
    before=evidence(model);receipt=transplant_grounded(model,args.candidate);after=evidence(model)
    if before!=after:raise AssertionError('candidate transplant changed the latest original learning')
    result=dict(latest_checkpoint=str(args.latest_checkpoint.resolve()),candidate=receipt,
                latest_old_tensors_exact=len(before['tensors']),
                latest_earned_optimizer_states_exact=sum(bool(value) for value in before['optimizer'].values()),
                old_evidence_digest=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
                physical_episode_touched=False,checkpoint_written=False,production_choice_license=False)
    atomic_json(args.out/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
