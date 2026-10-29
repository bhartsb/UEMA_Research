# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Calibrate an observing candidate using one restored, already lived-in body.

No supplied action reaches production. This writes a shadow-only configuration,
never a skill pass or a physical choice license. The selected world's measured
capacity is an observed operating point, not a biological safety threshold.
"""
import argparse
from copy import deepcopy
import gc
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from mimo_bridge import continuation_state
from mimo_bridge import copied_execution
from mimo_bridge.acting_snapshot import calibration_for_candidate
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite,COMMAND_POSITION_RANGE_CM,command_for_encoder
from mimo_bridge.self_intention import perception_fingerprint
from mimo_bridge.train_mimo import build_learner,acting_configuration
from lem.model.learning_opportunity import CONTRACT
from .self_recovery_experiment import configuration,hold
from .stillness_learning_trial import source_integrity
from .causal_test import _limits_for_run
from .training_continuation import atomic_json


def resolution_requirement(profile,physical_separation_pa):
    """Bound combined quantization error below half a measured distinction.

    Two rounded measurements differ from their true contrast by at most one
    quantization step. Use two steps < measured separation, without interpreting
    the q index as an information percentage or changing the physical operator.
    """
    if not math.isfinite(physical_separation_pa) or physical_separation_pa<=0:
        raise ValueError('no measured physical pressure distinction')
    bits=max(profile['minimum_bits'],math.ceil(math.log2(1+2*profile['pressure_range_pa']/physical_separation_pa)))
    while profile['pressure_range_pa']/(2**bits-1)*2>=physical_separation_pa:bits+=1
    if bits>profile['maximum_bits']:raise ValueError('measured distinction exceeds the maximum sensor precision')
    lower=(bits-.5-profile['minimum_bits'])/(profile['maximum_bits']-profile['minimum_bits'])
    q=float(np.nextafter(np.float32(max(lower,0.)),np.float32(1.)))
    if q<=0:q=float(np.finfo(np.float32).eps)
    # Re-evaluate using the same float32 q -> bits operation as live perception.
    while round(np.float32(profile['minimum_bits'])+np.float32(profile['maximum_bits']-profile['minimum_bits'])*np.float32(q))<bits:
        q=float(np.nextafter(np.float32(q),np.float32(1.)))
    return dict(required_bits=bits,minimum_resolution=q,
        maximum_quantization_step_pa=profile['pressure_range_pa']/(2**bits-1),
        measured_pressure_separation_pa=physical_separation_pa,
        scope='Coarsest sampled repeatable action-controlled pressure distinction; finer distinctions need their own precision')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--body',type=int,default=-1,help='Body index; -1 samples the saved body with the lowest valid motor capacity')
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--contact-profile',choices=['mechanical_v2_local','mechanical_v3'],default=None,
        help='Explicit copied v2-to-v3 physical transducer migration; never changes the source checkpoint')
    args=parser.parse_args(argv);args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    from types import SimpleNamespace
    def status(stage,**extra):atomic_json(args.out/'status.json',dict(stage=stage,timestamp=time.time(),**extra))
    status('loading_saved_body')
    try:
        if args.steps<2:raise ValueError('calibration needs several actual consequences')
        source_integrity(args.checkpoint)
        episode=continuation_state.load(args.checkpoint)
        if episode is None:raise ValueError('complete saved body unavailable')
        if args.body==-1:
            available=[]
            for saved in episode['worlds']:
                cap=saved['payload']['info']['motor_function']
                available.append(float(np.asarray(cap['values'])[np.asarray(cap['valid'],bool)].min()))
            args.body=int(np.argmin(available))
        if not 0<=args.body<len(episode['worlds']):raise ValueError('complete saved body unavailable')
        world=episode['worlds'][args.body]
        source_contact_profile=world['env_kwargs']['contact_profile']
        if args.contact_profile and args.contact_profile!=source_contact_profile:
            from mimo_bridge.virtual_skin_transition import activate_local_skin
            # This one-body probe does not need to duplicate the full eight-body
            # learner/replay in order to migrate its preserved physical history.
            subset=dict(config=episode['config'],worlds=[world],trainer={})
            world=activate_local_skin(subset,target=args.contact_profile)['worlds'][0]
        # A calibration probe has its own fresh model history. It cannot attach
        # one body's frame to another's saved transformer/recurrent history.
        del episode;gc.collect()
        a=configuration(args.checkpoint,args.out);a.device='cpu';a.self_choice='off'
        if args.contact_profile:a.contact_profile=args.contact_profile
        model=build_learner(a,None);loaded=model.load_checkpoint(str(args.checkpoint))
        if loaded.get('reinitialized_components'):raise ValueError('calibration changed existing trained components')
        model.reset_transition_state();model.reset_context(1)
        dev,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
        if not dev.get('phase_name'):raise ValueError('actual body-development limits unavailable')
        dev['exploration_noise']=0.
        source=SingleEnvSource(**world['env_kwargs'])
        try:
            frames=source.restore_full_state([world]);initial=source.save_full_state()
            if frames[0].meta['body_function']['mode']!='modulate':raise ValueError('actual modulated body required')
            profile=frames[0].meta['body_function']['profile']
            first=model.step(frames,dev,train=False,step_index=model.step_count,executed_action=None)
            varied=first.motor_for_action.copy();held=hold(frames[0],float(dev['strength_limit']))
            # Match strength/effort so the sampled contrast is an order difference.
            held[...,7:9]=varied[...,7:9]
            causal=copied_execution.capture(model)
            weights=perception_fingerprint(model);step=model.step_count;rows=[];initial_q=np.asarray(frames[0].meta['body_function']['values'])[:,0]
            for name,command in [('hold',held),('repeat_hold',held),('varied',varied)]:
                status('measuring',condition=name)
                frames=source.restore_full_state(initial);copied_execution.restore(model,causal)
                uncertainty=[];pressure=[];capacities=[];samples=[];expected=[];physics_seconds=0.
                for i in range(args.steps):
                    executed=command.copy();gazes=[]
                    for rep in range(a.action_repeat):
                        tick=time.perf_counter()
                        frames=source.step(command,np.zeros((1,2)),render=rep==a.action_repeat-1)
                        physics_seconds+=time.perf_counter()-tick
                        gazes.append(frames[0].meta['gaze_cmd'])
                    executed=LemLite.gaze_to_handle(command,np.mean(gazes,axis=0)[None])
                    if name=='hold':samples.append(dict(frames=deepcopy(frames),dev=deepcopy(dev),
                        executed_action=executed.copy(),executed_order_id=None))
                    result=model.step(frames,dev,train=False,step_index=model.step_count,executed_action=executed)
                    if name=='hold':expected.append(copied_execution.output_record(model,result))
                    pressure.append(np.asarray(frames[0].meta['physical_touch_audit'])[:,:2].copy())
                    cap=frames[0].meta['motor_function'];capacities.append(np.asarray(cap['values'])[np.asarray(cap['valid'],bool)].copy())
                    # This only records a proposed uncertainty operating point;
                    # it does not attest calibrated forecast accuracy.
                    p=model.self_state.perceive(**model.self_history.inputs)
                    encoded=command_for_encoder(torch.from_numpy(executed)).reshape(1,1,-1)
                    _,variance=model.self_state.forecast(p,encoded,torch.tensor([[.01*a.action_repeat]]))
                    uncertainty.append(float(variance[...,0].sqrt().max()))
                rows.append(dict(name=name,bones=frames[0].bones.copy(),pressure=np.asarray(pressure),
                    physics_seconds=physics_seconds,physical_steps=args.steps*a.action_repeat,
                    capacity=np.asarray(capacities),uncertainty=uncertainty,
                    tokens={m:v.detach().float().cpu().numpy() for m,v in model.last_tokens.items()
                            if model._last_source_mask.get(m)=='real' and m!='motor'}))
                if name=='hold':
                    model_replay=copied_execution.verify_recorded_frames(model,causal,samples,expected)
                del samples,expected
            base,repeat,changed=rows
            bone_noise=float(np.max(np.abs(base['bones']-repeat['bones'])))
            pressure_noise=float(np.max(np.abs(base['pressure']-repeat['pressure'])))
            token_noise={m:float(np.mean((repeat['tokens'][m]-v)**2)) for m,v in base['tokens'].items()}
            if bone_noise!=0 or pressure_noise!=0 or any(value!=0 for value in token_noise.values()):
                raise ValueError('same-command physical and sensory replay is not exact')
            effects={m:float(np.mean((changed['tokens'][m]-v)**2)) for m,v in base['tokens'].items()}
            floors={m:token_noise[m]+float(np.finfo(np.float32).eps**2*max(1.,np.mean(v*v))) for m,v in base['tokens'].items()}
            separation=float(np.max(np.abs(changed['pressure']-base['pressure'])))-pressure_noise
            resolution=resolution_requirement(profile,separation)
            # An observed response at this actual availability does not establish
            # what would happen below it. Keep that reference point explicit.
            motor_floor=float(min(row['capacity'].min() for row in rows))
            if not 0<motor_floor<=1:raise ValueError('no positive observed motor operating point')
            if not any(effects[m]>floors[m] for m in effects):raise ValueError('orders produced no resolved sensory distinction')
            observation=dict(schema='shadow_distinguishability_calibration_v1',checkpoint=str(args.checkpoint),
                body=args.body,source_sim_time=initial[0]['payload']['sim_time'],limits=dev,limits_source=origin,
                actual_saved_body=True,pressure_repeat_error_pa=pressure_noise,token_repeat_mse=token_noise,
                physical_replay_exact=True,model_replay=model_replay,
                pressure_requirement=resolution,minimum_sampled_capacity=motor_floor,
                actor_weights_unchanged=weights==perception_fingerprint(model),token_effect_mse=effects,effect_floors=floors,
                original_minimum_resolution_index=float(initial_q.min()),
                source_contact_profile=source_contact_profile,
                sampled_world_step_seconds={row['name']:row['physics_seconds']/row['physical_steps'] for row in rows},
                calibration_scope='One lived-in body and sampled command contrast; shadow observation only. No general safe operating envelope or choice accuracy established.',
                contact_profile=a.contact_profile,production_license=False)
            atomic_json(args.out/'measurements.json',observation)
            calibration=dict(minimum_resolution=resolution['minimum_resolution'],minimum_capacity=motor_floor,
                maximum_uncertainty=float(max(max(row['uncertainty']) for row in rows)+float(np.finfo(np.float32).eps)),
                candidate_position_fraction=min(1.,float(np.max(np.abs(varied[...,:3]-held[...,:3]))/COMMAND_POSITION_RANGE_CM)),
                candidate_rotation_fraction=min(1.,float(np.max(np.abs(varied[...,3:7]-held[...,3:7])))),
                order_steps=a.motor_hold,orders=1,step_seconds=.01*a.action_repeat,
                body_function='modulate',body_function_profile=profile,motor_contract=frames[0].meta['motor_contract'],
                learning_opportunity=dict(contract=CONTRACT,audit_every=max(1,a.motor_hold),window=8,anchor_capacity=8,
                    effect_floors=floors,calibration=str(args.out/'measurements.json')),
                observed_operating_point_only=True,contact_profile=a.contact_profile,
                uncertainty_scope='Maximum uncertainty proposed in this sample; accuracy unqualified',
                production_license=False)
            receipt=calibration_for_candidate(model,calibration,acting_configuration(a))
            atomic_json(args.out/'shadow_calibration.json',calibration)
            atomic_json(args.out/'unqualified_candidate.json',receipt)
            atomic_json(args.out/'report.json',dict(complete=True,production_license=False,measurements=observation,
                calibration=str(args.out/'shadow_calibration.json'),
                live_integration='Bind a new candidate to exact current learner/configuration; observe only. Recalibrate after changing physical contact profile.'))
            status('complete',production_license=False)
            return 0
        finally:source.close()
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':raise SystemExit(main())
