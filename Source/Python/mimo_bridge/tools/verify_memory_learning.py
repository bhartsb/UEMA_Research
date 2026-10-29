# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Train and freeze a planted delayed-consequence task; distinguish it from crib skill."""
import argparse
import json
from pathlib import Path
import torch
from lem.model.persistent_memory import PersistentMemory


def examples(n,length,generator):
    x=.03*torch.randn(n,length,8,generator=generator)
    cue=torch.randint(0,2,(n,),generator=generator).float()*2-1
    where=torch.randint(1,5,(n,),generator=generator)
    x[:,:,0:2]=0
    x[torch.arange(n),where,0]=cue;x[torch.arange(n),where,1]=1
    x[:,-8:]=0  # current/direct-context observations contain no answer
    t=torch.arange(length).float().expand(n,-1)*.5
    return x,t,cue


def run(out,steps=1200,seed=31):
    torch.set_num_threads(2);torch.manual_seed(seed)
    m=PersistentMemory(8,1,4,1,hidden_size=32)
    opt=torch.optim.Adam(m.parameters(),lr=.002)
    g=torch.Generator().manual_seed(seed+1)
    first=None
    for i in range(steps):
        length=(24,40,64)[i%3]
        x,t,cue=examples(32,length,g)
        state,_=m.encode(x,t,t[:,-1])
        action=torch.randint(0,2,(32,1),generator=g).float()*2-1
        target=(action[:,0]*cue>0).float()[:,None].expand(-1,5)
        p=m.predict_needs(state,action)
        # A separate old-cue readout also exercises the adapter used by the main
        # transformer, rather than proving only the auxiliary outcome head learns.
        cue_hat=m.bias(state)[:,0,0]
        loss=(p-target).square().mean()+.1*(cue_hat-cue).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if first is None: first=float(loss.detach())
    m.eval();held=torch.Generator().manual_seed(seed+1000);rows=[]
    with torch.no_grad():
        for length in (48,80,120):
            x,t,cue=examples(256,length,held);state,_=m.encode(x,t,t[:,-1])
            def choice(s):
                a=m.predict_needs(s,torch.ones(len(s),1))[:,0]
                b=m.predict_needs(s,-torch.ones(len(s),1))[:,0]
                return torch.where(a>b,1.,-1.)
            actual=choice(state)
            blank=x.clone();blank[:,:,0:2]=0
            empty,_=m.encode(blank,t,t[:,-1])
            scrambled=state[torch.randperm(len(state),generator=held)]
            reset,_=m.encode(x[:,-8:],t[:,-8:],t[:,-1])
            rows.append(dict(minimum_cue_delay_seconds=(length-5)*.5,maximum_cue_delay_seconds=(length-2)*.5,samples=len(cue),
                correct_choices=int((actual==cue).sum()),
                no_cue_correct=int((choice(empty)==cue).sum()),
                shuffled_memory_correct=int((choice(scrambled)==cue).sum()),
                reset_memory_correct=int((choice(reset)==cue).sum()),
                old_cue_recall_correct=int(((m.bias(state)[:,0,0]>0)==(cue>0)).sum()),
                optimal_choices=len(cue),chance_expected=len(cue)/2))
    passed=all(r['correct_choices']==r['samples'] and r['old_cue_recall_correct']==r['samples'] for r in rows)
    result=dict(status='passed' if passed else 'failed',seed=seed,training_updates=steps,
                first_training_loss=first,last_training_loss=float(loss.detach()),held_out=rows,
                criterion='Every held-out deterministic choice and cue recall correct; direct observations have no cue',
                scope='Learned memory mechanism on planted data. These weights are not adopted by the infant; no crib intention claim.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    torch.save(m.state_dict(),out/'planted_memory_only.pt')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--seed',type=int,default=31);a=ap.parse_args();r=run(a.out,a.steps,a.seed)
    raise SystemExit(0 if r['status']=='passed' else 1)
