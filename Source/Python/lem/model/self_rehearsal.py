# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Retain completed self consequences across episodes without inventing continuity."""
import random
from .self_state import map_state


class SelfRehearsal:
    def __init__(self,capacity=128,seed=901):
        if capacity<2:raise ValueError('self rehearsal requires common and changing experience')
        self.capacity=capacity;self.seen=0;self.common=[];self.changed=[]
        self.rng=random.Random(seed)

    def add(self,events):
        if not events:return
        # A complete causal window owns its earlier inputs and executed sequence.
        # Reserve half the budget for uniform experience, half for observed change.
        entry=map_state(dict(inputs=events[0]['inputs'],
            initial_target=events[0].get('initial_target',events[0]['inputs']['sites'][...,-3:]),
            actions=[e['action'] for e in events],dt=[e['dt'] for e in events],
            targets=[e['target'] for e in events],valid=[e['valid'] for e in events]),'cpu')
        self.seen+=1;half=self.capacity//2
        if len(self.common)<half:self.common.append(entry)
        else:
            j=self.rng.randrange(self.seen)
            if j<half:self.common[j]=entry
        change=float((entry['targets'][-1]-entry['initial_target']).abs().amax())
        if change>0:
            self.changed.append((change,entry));self.changed.sort(key=lambda x:x[0],reverse=True)
            del self.changed[self.capacity-half:]

    def sample(self,device):
        entries=self.common+[entry for _,entry in self.changed]
        return None if not entries else map_state(self.rng.choice(entries),device)

    def state_dict(self):
        return dict(capacity=self.capacity,seen=self.seen,common=self.common,changed=self.changed,rng=self.rng.getstate())

    def load_state_dict(self,s):
        if s['capacity']!=self.capacity:raise ValueError('self rehearsal capacity needs explicit migration')
        self.seen=s['seen'];self.common=map_state(s['common'],'cpu');self.changed=map_state(s['changed'],'cpu')
        self.rng.setstate(s['rng'])
