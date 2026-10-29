# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Read-only assessment of opportunity timing and prediction-coordinate scope.

The constructed method trace identifies the actual native call interfaces. It
does not fit a predictor, load a checkpoint, step physics or change any gate.
"""
import argparse
from collections import Counter,deque
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from mimo_bridge.opportunity_integration import OpportunityAudit
from .training_continuation import atomic_json


def observed_predictor_contract():
    """Trace native methods with distinct interface owners and target values."""
    calls=[];modalities=('vision','tactile','audio','proprioception')
    class Predictor:
        def __init__(self,owner):self.owner=owner
        def predict_from_action(self,unused,commands,registry,heads,**kw):
            calls.append(dict(predictor=self.owner,heads=heads.owner,encoder=registry.owner,
                              input_value=float(kw['state_tokens']['vision'][0,0])))
            return {m:torch.zeros(2,2) for m in modalities},None
    class Meter:
        floors={m:1. for m in modalities}  # Construction only; never a live calibration.
        retention={}
        def observe_fresh(self,*a,**kw):pass
        def explain(self,*a):return {}
        def available(self,*a):return 0.,None
    model=SimpleNamespace(device=torch.device('cpu'),encoder_registry=SimpleNamespace(owner='frozen_actor'),
        last_tokens={m:torch.ones(2,2) for m in modalities},last_context=torch.ones(2,1,2),
        last_achieved=torch.ones(2,1,9),_memory_bias_current=None,
        _last_source_mask={m:'real' for m in modalities},self_history=SimpleNamespace(time=torch.ones(2)))
    learner=SimpleNamespace(forward_model=Predictor('continuing_learner'),
        prediction_heads=SimpleNamespace(owner='continuing_learner'),
        last_tokens={m:torch.full((2,2),13.) for m in modalities})
    audit=OpportunityAudit.__new__(OpportunityAudit)
    audit.model=model;audit.every=5;audit.capacity=8;audit.pending=None
    audit.reference=Predictor('frozen_reference');audit.reference_heads=SimpleNamespace(owner='frozen_reference')
    audit.anchors=deque(maxlen=8);audit.retention_panel=[];audit.meter=Meter();audit.completed=0;audit.last={}
    audit.healthy=lambda:torch.ones(2,dtype=torch.bool)
    audit.health_record=lambda:dict(constructed=True)
    audit.prepare(learner,torch.zeros(2,1,9),0)
    model.last_tokens={m:torch.full((2,2),7.) for m in modalities}
    model.self_history.time=torch.full((2,),1.05)
    audit.complete()
    target=audit.anchors[0]['target']['vision']
    if len(calls)!=3 or not torch.equal(target,torch.full((2,2),7.)):
        raise ValueError('native audit interface changed; this assessment cannot classify it')
    return dict(trace_kind='native method calls on constructed metadata; no learned or physical result',
        current_prediction=calls[0],wrong_action_prediction=calls[1],reference_prediction=calls[2],
        target_encoder='frozen_actor',target_value=float(target[0,0]),
        continuing_learner_target_value=float(learner.last_tokens['vision'][0,0]))


def assess(configuration,actor,frame_seconds,observed):
    """Classify semantic validity independently of any favorable numeric score."""
    if not actor or actor.get('learner_updates_independent') is not True:
        raise ValueError('located independent actor metadata required')
    receipt=configuration['learning_opportunity']
    every=int(receipt['audit_every']);hold=int(configuration['motor_hold']);repeat=int(configuration['action_repeat'])
    if min(every,hold,repeat)<1 or not math.isfinite(frame_seconds) or frame_seconds<=0:
        raise ValueError('invalid native timing metadata')
    native_seconds=frame_seconds*repeat
    forecast_seconds=float(actor['decision']['horizon_seconds'])
    if not math.isfinite(forecast_seconds) or forecast_seconds<=0:raise ValueError('invalid self-choice horizon')
    current=observed['current_prediction']
    hybrid=(current['predictor']=='continuing_learner' and current['heads']=='continuing_learner'
            and current['encoder']=='frozen_actor' and observed['target_encoder']=='frozen_actor')
    if not hybrid:raise ValueError('unrecognized prediction interface; no automatic validity claim')
    phases=sorted({step%hold for step in range(0,math.lcm(every,hold),every)})
    audit=actor.get('learning_opportunity') or {}
    reasons=Counter(entry['reason'] for body in audit.get('eligibility',[]) for entry in body.values())
    return dict(schema='opportunity_contract_assessment_v1',
        observed_predictor_contract=observed,prediction_scope='current predictor compatibility with frozen actor coordinates',
        complete_learner_retention_certified=False,complete_learner_learning_certified=False,
        production_qualification_supported=False,
        timing=dict(native_observation_seconds=native_seconds,audit_outcome_seconds=native_seconds,
                    audit_spacing_seconds=every*native_seconds,held_order_seconds=hold*native_seconds,
                    self_choice_forecast_seconds=forecast_seconds,order_phases_sampled=phases,
                    only_order_onset=phases==[0],forecast_duration_measured_by_audit=native_seconds==forecast_seconds),
        last_completed_comparisons=audit.get('completed_comparisons'),
        last_health_eligible_bodies=(audit.get('audit_health') or {}).get('available_bodies'),
        last_gate_reasons=dict(reasons),
        interpretation='Zero hybrid credit cannot establish physical forgetting or absence of useful longer-horizon learning. Its causes require matched copied controls.',
        production_controls_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--through-step',type=int)
    args=parser.parse_args();config=json.loads((args.run/'args.json').read_text(encoding='utf-8'))
    calibration=json.loads(Path(config['self_choice_receipt']).read_text(encoding='utf-8'))
    config['learning_opportunity']=calibration['learning_opportunity']
    last=None;count=0
    with (args.run/'replay_training_stats.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue  # A concurrently appended partial line is not evidence.
            if row.get('type')=='mimo' and (args.through_step is None or row['step']<=args.through_step):
                last=row;count+=1
    if last is None:raise ValueError('no complete native observation in requested interval')
    result=assess(config,last['acting_snapshot'],float(last['env_dt_s']),observed_predictor_contract())
    result.update(run=str(args.run.resolve()),last_logged_step=last['step'],last_timestamp=last['timestamp'],
                  logged_snapshots=count,source_scope='Current imported native call contract and located log/argument metadata; not a retrospective loaded-code attestation')
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.out,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
