# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Development diagnosis: can a fitted local readout recover observed function?

Fit only readout coefficients; preserve the entire copied brain. Leave one saved
development world out per fit. These are diagnostic folds, not fresh confirmation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from mimo_bridge.train_mimo import build_learner
from .self_recovery_experiment import configuration
from .training_continuation import atomic_json


def fit_readout(x, y, weights):
    # Standardization and ridge fit use training rows only. Stable double solve.
    x=x.double();y=y.double();weights=weights.double()
    center=(x*weights[:,None]).sum(0)/weights.sum()
    scale=(((x-center).square()*weights[:,None]).sum(0)/weights.sum()).sqrt().clamp_min(1e-6)
    design=torch.cat([(x-center)/scale,torch.ones(len(x),1,dtype=x.dtype)],1)
    penalty=torch.eye(design.shape[1],dtype=x.dtype)*.01;penalty[-1,-1]=0
    coefficients=torch.linalg.solve(design.T@(design*weights[:,None])+penalty,
                                     design.T@(y*weights[:,None]))
    return dict(center=center,scale=scale,coefficients=coefficients)


def predict(readout,x):
    x=x.double();design=torch.cat([(x-readout['center'])/readout['scale'],torch.ones(len(x),1,dtype=x.dtype)],1)
    return (design@readout['coefficients']).clamp(0,1).float()


def errors(pred,truth):
    error=(pred-truth).abs();affected=truth[:,0]<.99;healthy=~affected
    return dict(affected_sites=int(affected.sum()),healthy_sites=int(healthy.sum()),
        affected_mae=float(error[affected,0].mean()) if affected.any() else None,
        healthy_mae=float(error[healthy,0].mean()) if healthy.any() else None,
        all_channels_mae=float(error.mean()),max_error=float(error.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    def status(stage,**kw):atomic_json(args.out/'status.json',dict(stage=stage,pid=os.getpid(),timestamp=time.time(),**kw))
    declaration=json.loads((args.study/'declaration.json').read_text())
    source=json.loads((args.study/'report.json').read_text());seeds=declaration['development_seeds']
    atomic_json(args.out/'declaration.json',dict(question=__doc__,parent=source['checkpoint'],seeds=seeds,
        fitting='Weighted ridge readout, penalty .01 after training-only standardization. Half weight on affected sites and half on unaffected sites within each training world; no trial selection on held-out worlds.',
        contrasts=['unchanged native readout','frozen site encoder','frozen site encoder plus body belief'],
        primary='Affected-site and unaffected-site absolute errors, separately, for every left-out development world.',
        decision='If encoded local information can be recovered, repair the decoder/training. If not, repair encoding. No adoption or new ability is certified.',
        limits='Development diagnostic only. Direct observed function is an existing sensory input, not a future answer. No production change.'))
    status('loading')
    model=build_learner(configuration(Path(source['checkpoint']),args.out),str(args.out/'checkpoints'))
    model.load_checkpoint(source['checkpoint']);module=model.self_state.eval();samples=[]
    original={k:v.clone() for k,v in module.state_dict().items()}
    with torch.no_grad():
        for seed in seeds:
            s=torch.load(args.study/str(seed)/'branches.pt',map_location='cpu',weights_only=False)
            percept=module.perceive(**s['inputs']);mask=percept['valid'][0]
            local=percept['local'][0][mask];belief=percept['belief'][0]
            samples.append(dict(seed=seed,encoded=local-belief,native_features=local,
                native=percept['present'][0][mask],truth=s['start'][mask]))
    reports=[];coefficients={}
    for i,test in enumerate(samples):
        train=[s for j,s in enumerate(samples) if j!=i];ys=[];weights=[]
        for s in train:
            y=s['truth'];affected=y[:,0]<.99;w=torch.zeros(len(y))
            for mask in (affected,~affected):
                if mask.any():w[mask]=1./mask.sum()
            ys.append(y);weights.append(w/w.sum())
        row=dict(seed=test['seed'],native=errors(test['native'],test['truth']))
        for name in ('encoded','native_features'):
            readout=fit_readout(torch.cat([s[name] for s in train]),torch.cat(ys),torch.cat(weights))
            coefficients[f'{test["seed"]}_{name}']=readout
            row[name]=errors(predict(readout,test[name]),test['truth'])
        reports.append(row);status('fitting_local_readouts',step=i+1,steps=len(samples))
    unchanged=all(torch.equal(v,original[k]) for k,v in module.state_dict().items())
    if not unchanged:raise RuntimeError('Diagnostic changed the copied original weights')
    torch.save(coefficients,args.out/'readouts.pt')
    report=dict(worlds=reports,original_self_weights_unchanged=unchanged,scope='Development folds only. No production license.',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),completed_at=time.time())
    atomic_json(args.out/'report.json',report);status('complete',verdict='LOCAL_READOUT_COMPARISON_COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        out=Path(sys.argv[sys.argv.index('--out')+1])
        if out.exists():
            atomic_json(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
            atomic_json(out/'status.json',dict(stage='failed',timestamp=time.time(),error=str(error)))
        raise
