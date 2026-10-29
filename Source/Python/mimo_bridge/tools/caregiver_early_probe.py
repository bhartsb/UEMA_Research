# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Show a completed copied learner's fixed behavior probes before the full trial ends.

Does not write into the original study, change its sample size, or test a winner.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from .caregiver_learning_trial import settings, probe
from .training_continuation import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--speech',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    declaration=json.loads((args.study/'declaration.json').read_text())
    fitted=json.loads((args.study/'responsive_0/fit.json').read_text());seed=declaration['seeds'][0]
    conditions=('follow_visible','follow_hidden','return_visible','return_hidden','cue_correct','cue_wrong')
    atomic_json(args.out/'declaration.json',dict(question='What does the first completed caregiver-trained copy do in its fixed behavior tests?',
        checkpoint=fitted['checkpoint'],conditions=conditions,seed=seed,parent_study=str(args.study),
        device='cpu',primary='Report every fixed orientation measurement, including misses and the absent-object controls.',
        scope='One copied learner after its exposure. No prior-score or replay-trained comparison; no learning benefit, live ability, intention, or word-understanding verdict.',
        decision='Make actual behavior visible while the full paired comparison continues. Do not alter that comparison based on this early view.'))
    results={};durations=[]
    def status(stage,**kw):atomic_json(args.out/'status.json',dict(stage=stage,pid=os.getpid(),timestamp=time.time(),
        step=len(results),steps=len(conditions),completed_durations_seconds=durations,**kw))
    try:
        for kind in conditions:
            status('probing_frozen_copy',arm=kind);start=time.time();out=args.out/kind
            a=settings(Path(declaration['parent']),out,args.speech,'cpu',seed)
            results[kind]=probe(a,Path(fitted['checkpoint']),out,seed,declaration['limits'],
                args.study/'fixtures_0'/(kind+'.json'),declaration['warm_steps'],declaration['probe_steps'],kind)
            durations.append(time.time()-start)
        report=dict(results=results,durations_seconds=durations,scope='One copied learner, descriptive only; original study untouched.',
            completed_at=time.time(),comparison_pending=True)
        atomic_json(args.out/'report.json',report);status('complete',verdict='EARLY_BEHAVIOR_MEASUREMENTS_RECORDED')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
