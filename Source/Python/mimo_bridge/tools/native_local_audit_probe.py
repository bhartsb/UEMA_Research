# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Inspect completed prospective audits across a bounded native continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from mimo_bridge import continuation_state
from .training_continuation import atomic_json,replace_argument,complete_candidates
from .native_proof_role import EXTENSION_CONTRACT,FLAG,validate,is_isolated


def completed_audit_evidence(audit, target):
    """A completed comparison may coexist with the next prospective audit."""
    payload=audit['last']
    if audit['completed']<1 or len(audit['anchors'])<1:
        raise ValueError('prospective audit did not complete and retain its real outcome')
    if payload.get('completed_comparisons')!=audit['completed']:
        raise ValueError('completed audit payload and counter disagree')
    comparisons=payload.get('latest_comparisons',{})
    if set(comparisons)!={'vision','tactile','audio','proprioception'}:
        raise ValueError('audit did not compare every actual sensory modality')
    for modality,row in comparisons.items():
        for key in ('prediction_mse','action_control_mse','persistence_mse','eligible'):
            values=np.asarray(row[key])
            if values.shape!=(8,) or not np.isfinite(values).all():
                raise ValueError('audit lacks finite paired outcomes for every body: '+modality+'/'+key)
    pending=audit['pending']
    if pending is not None:
        # Native preparation uses the current global step before incrementing.
        # This final command's outcome is consumed on the next actor step.
        if pending['step']!=target-1:
            raise ValueError('saved prospective audit is stale or from the future')
        for group in ('prediction','shuffled','reference'):
            # The forward model also emits motor/voice targets. Only the four
            # calibrated real sensory modalities enter this audit's meter.
            if not set(comparisons).issubset(pending[group]):
                raise ValueError('pending audit sensory contract changed')
            for values in pending[group].values():
                values=np.asarray(values)
                if values.shape[0]!=8 or not np.isfinite(values).all():
                    raise ValueError('pending audit lacks finite eight-body predictions')
    return payload, None if pending is None else pending['step']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true',help='Inspect the already saved extension; do not run a trainer.')
    args=parser.parse_args();previous=args.previous_report.resolve();out=args.out.resolve()
    report=json.loads(previous.read_text());last=report['passes'][-1]
    source=Path(last['checkpoint']);initial=int(last['step']);target=initial+2
    folder=out/'pass_1'
    declaration=out/'declaration.json'
    if not args.verify_only:
        out.mkdir(parents=True,exist_ok=False);folder.mkdir()
        atomic_json(declaration,dict(schema=EXTENSION_CONTRACT,out=str(out),source=str(source),
        previous_report=str(previous),previous_report_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
        updates_per_pass=2,passes=1,body_count=8,production_controls_used=False,
        criterion='Global step modulo audit_every schedules preparation; the next observation completes its actual eight-body outcome.'))
    argv=json.loads((previous.parent/'pass_2/command.json').read_text())
    changes={FLAG:declaration,'--resume':source,'--out':folder,'--checkpoint-storage':folder/'checkpoints',
             '--caregiver-record-dir':folder/'recordings','--max-steps':target}
    for key,value in changes.items():argv=replace_argument(argv,key,value)
    validate(argv)
    if not is_isolated(argv):raise ValueError('production inventory does not recognize this isolated audit proof')
    if args.verify_only:
        if json.loads((folder/'command.json').read_text())!=argv:
            raise ValueError('saved command differs from the declared native continuation')
    else:
        atomic_json(folder/'command.json',argv)
        atomic_json(out/'status.json',dict(stage='native_audit_extension',pid=os.getpid(),recognized_isolated=True,timestamp=time.time()))
        with (folder/'console.log').open('w',encoding='utf-8') as stream:
            done=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        if done.returncode:
            atomic_json(out/'status.json',dict(stage='failed',exit_code=done.returncode,timestamp=time.time()))
            raise RuntimeError('native audit extension failed; see '+str(folder/'console.log'))
    candidates=complete_candidates(folder)
    if not candidates or int(candidates[0].name.rsplit('_',1)[1])!=target:
        raise ValueError('audit extension did not save its actual final checkpoint')
    checkpoint=candidates[0];episode=continuation_state.load(checkpoint)
    from .actor_runtime import inner_state_for_report
    actor=inner_state_for_report(episode['trainer']['acting_snapshot']);audit=actor['opportunity']
    times=[w['payload']['sim_time'] for w in episode['worlds']]
    if (actor['steps']!=last['actor_observations']+2 or actor['born_step']!=last['actor_born_step']
            or actor['qualification'] is not True or actor['last'].get('applied_to_body') is not False):
        raise ValueError('observer identity, observation count or command authority changed')
    if len(times)!=8 or any(abs(a-b-.1)>1e-7 for a,b in zip(times,last['sim_times'])):
        raise ValueError('audit extension lost the continuing eight-body clock')
    payload,pending_step=completed_audit_evidence(audit,target)
    atomic_json(out/'report.json',dict(complete=True,checkpoint=str(checkpoint),step=target,
        actor_observations=actor['steps'],actor_born_step=actor['born_step'],sim_times=times,
        completed_comparisons=audit['completed'],retained_anchors=len(audit['anchors']),audit=payload,
        pending_next_audit_step=pending_step,
        scheduling='Preparation uses global training step modulo audit_every; observation count does not schedule it.',
        validator_correction='A completed audit may coexist with a pending next audit. The original validator incorrectly rejected that valid state.',
        actor_unqualified=True,applied_to_body=False,production_choice_license=False,
        production_controls_used=False,scope='Actual completed native prediction/control/persistence audit; not a learned-function pass.'))
    atomic_json(out/'status.json',dict(stage='complete',timestamp=time.time()))


if __name__=='__main__':main()
