# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Locate loss of local self information using saved development data only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from .self_recovery_experiment import configuration
from .training_continuation import atomic_json
from mimo_bridge.train_mimo import build_learner


def local_error(module,sample):
    with torch.no_grad():
        percept=module.perceive(**sample['inputs'])
        truth=sample['start'][None,:,0];pred=percept['present'][...,0]
        valid=sample['valid'][None]
        i=int(truth.masked_fill(~valid,2).argmin())
        return dict(mean_absolute_error=float((pred-truth).abs()[valid].mean()),
                    worst_site=i,observed_function=float(truth[0,i]),predicted_function=float(pred[0,i]),
                    worst_site_error=float(abs(pred[0,i]-truth[0,i])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    def status(stage,**kw):atomic_json(a.out/'status.json',dict(stage=stage,timestamp=time.time(),pid=__import__('os').getpid(),**kw))
    d=json.loads((a.study/'declaration.json').read_text());r=json.loads((a.study/'report.json').read_text())
    atomic_json(a.out/'declaration.json',dict(question='Can the self representation learn to read a local loss of function already present in its inputs?',
        data='Saved development scenes only. No held-out test scene is used for repair.',updates=120,
        contrasts=['native all-sites present loss','same lesson focused on each sample lowest-function site'],
        decision='If local grounding fails, repair it before more action trials. If it learns here, inspect future/action learning separately.',
        success='Diagnostic comparisons and exact initial snapshots, not a capability or promotion test.',parent=r['checkpoint']))
    status('loading_development_data')
    config=configuration(Path(r['checkpoint']),a.out);model=build_learner(config,str(a.out/'checkpoints'))
    model.load_checkpoint(r['checkpoint']);initial=deepcopy(model.self_state.state_dict())
    samples=[torch.load(a.study/str(seed)/'branches.pt',weights_only=False,map_location='cpu') for seed in d['development_seeds']]
    before=[local_error(model.self_state,s) for s in samples];results={}
    for mode in ('native','affected_site'):
        module=deepcopy(model.self_state);module.load_state_dict(initial)
        # Keep every corresponding Adam moment; only this copied module receives updates.
        params=list(module.parameters());source=list(model.self_state.parameters())
        opt=torch.optim.AdamW(params,lr=model.optimizer.param_groups[0]['lr'],weight_decay=0.)
        for old,new in zip(source,params):
            if old in model.optimizer.state:opt.state[new]=deepcopy(model.optimizer.state[old])
        for step in range(120):
            sample=samples[step%len(samples)];opt.zero_grad(set_to_none=True)
            percept=module.perceive(**sample['inputs']);truth=sample['start'][None]
            mask=sample['valid'][None,:,None].expand_as(truth)
            if mode=='affected_site':
                i=int(truth[...,0].masked_fill(~sample['valid'][None],2).argmin());mask=torch.zeros_like(mask);mask[:,i]=True
            loss=(percept['present']-truth).square()[mask].mean();loss.backward();opt.step()
            if step%10==0:status('diagnosing_local_grounding',arm=mode,step=step+1,steps=120,loss=float(loss.detach()))
        results[mode]=[local_error(module,s) for s in samples]
    report=dict(before=before,after=results,scope='Development-only grounding diagnostic; no live weight change or acquired ability claim.',
        function_scale='Skin index 0=minimum configured precision (2 bits), 1=original sensing; motor availability 0=no gain, 1=original gain. Error 0=exact, 1=full index-range error.',
        completed_at=time.time())
    atomic_json(a.out/'report.json',report);status('complete',verdict='LOCAL_GROUNDING_DIAGNOSTIC_COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        out=Path(sys.argv[sys.argv.index('--out')+1])
        if out.exists():
            atomic_json(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
            atomic_json(out/'status.json',dict(stage='failed',timestamp=time.time(),error=str(error)))
        raise
