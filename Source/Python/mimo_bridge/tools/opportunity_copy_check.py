# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measure repeatability and run the complete opportunity loop on one model copy.

This produces numerical sensory calibration and integration evidence. It cannot
issue an actor-adoption receipt or claim a developmental skill.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from mimo_bridge.acting_snapshot import FrozenActingSnapshot,calibration_for_candidate
from mimo_bridge.continuation_state import save as save_episode
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite
from mimo_bridge.self_intention import perception_fingerprint
from mimo_bridge.tools.causal_test import _limits_for_run
from mimo_bridge.tools.self_recovery_experiment import configuration,hold
from mimo_bridge.tools.training_continuation import atomic_json
from mimo_bridge.train_mimo import build_learner,env_kwargs_for,acting_configuration
from lem.model.learning_opportunity import CONTRACT


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--branch-steps',type=int,default=5)
    parser.add_argument('--learning-steps',type=int,default=6)
    parser.add_argument('--allow-cold-runtime',action='store_true',
                        help='Explicit integration-only test despite missing/corrupt source working statistics')
    args=parser.parse_args(argv);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    def status(stage,**data):atomic_json(out/'status.json',dict(stage=stage,timestamp=time.time(),**data))
    status('checking_source')
    if args.branch_steps<1 or args.learning_steps<2:raise ValueError('comparison needs physical steps and future learning outcomes')
    torch.set_num_threads(1)
    checkpoint=Path(args.checkpoint);a=configuration(checkpoint,out)
    runtime_check=dict(loaded=False,error=None)
    try:
        if (checkpoint/'runtime_state.pt').exists():
            runtime=torch.load(checkpoint/'runtime_state.pt',map_location='cpu',weights_only=False)
        else:runtime=json.loads((checkpoint/'runtime_state.json').read_text())
        if not isinstance(runtime,dict):raise ValueError('source working statistics are not a dictionary')
        del runtime
        runtime_check['loaded']=True
    except Exception as error:
        runtime_check['error']=str(error)
        atomic_json(out/'source_runtime.json',runtime_check)
        if not args.allow_cold_runtime:
            status('failed',error='Source working statistics unreadable; explicit integration-only override required',
                   source_runtime=runtime_check)
            raise ValueError('Cannot claim preserved source evidence with unreadable working statistics') from error
    a.device='cpu';a.caregiver_record_dir='';a.body_function='modulate';a.self_choice='off'
    a.caregiver_mode='off';a.caregiver_seconds=0.;a.gaze_reflex='off';a.spontaneous_saccades='off'
    a.critters='off';a.mobile='off'
    factory=lambda:build_learner(a,None)
    learner=build_learner(a,str(out/'copied_checkpoints'));learner.load_checkpoint(str(checkpoint));learner.reset_context(1)
    start_step=learner.step_count
    dev,dev_source=_limits_for_run(str(checkpoint),learner);dev['exploration_noise']=0.
    kw=env_kwargs_for(a,dev['phase_name'],str(out));kw['vision']=False;kw['seed']=81073
    source=SingleEnvSource(**kw)
    declaration=dict(scope='numerical repeatability and complete copied-model integration only',production_license=False,
        checkpoint=str(checkpoint),checkpoint_step=start_step,branch_steps=args.branch_steps,
        learning_steps=args.learning_steps,body_function='modulate',vision='not rendered; vision excluded from comparisons',
        source_runtime=runtime_check,source_causal_history_preserved=False,
        fresh_world='A controlled new episode; complete copies resume only from their own saved episode',
        gate='exact same-command physics/tokens; frozen actor weights; continued copied learner updates; saved complete episode',
        no_skill_verdict='No acquired action, restoration or exploration benefit is inferred from this check.')
    atomic_json(out/'declaration.json',declaration)
    try:
        frames=source.reset();initial_world=source.save_full_state()
        calibration=dict(minimum_resolution=1.,minimum_capacity=1.,maximum_uncertainty=100.,
            candidate_position_fraction=0.,candidate_rotation_fraction=0.,order_steps=a.motor_hold,orders=1,
            step_seconds=.01*a.action_repeat,body_function='modulate',body_function_profile=frames[0].meta['body_function']['profile'],
            motor_contract=frames[0].meta['motor_contract'])
        config=acting_configuration(a);receipt=calibration_for_candidate(learner,calibration,config)
        actor=FrozenActingSnapshot.from_learner(learner,factory,receipt,config,qualification=True)
        actor_initial=actor.state_dict();model_hash=perception_fingerprint(learner)
        result=actor.step(frames,dev,step_index=start_step,executed_action=None)
        varied=result.motor_for_action.copy();held=hold(frames[0],dev['strength_limit'])
        rows=[]
        for name,command in [('held',held),('repeat_held',held),('model_proposal',varied)]:
            status('measuring_repeatability',branch=name)
            frames=source.restore_full_state(initial_world)
            # Reuse one actor instance; restoring creates no third live model.
            del actor
            actor=FrozenActingSnapshot.restore(actor_initial,factory,'cpu',allow_qualification=True)
            actor.step(frames,dev,step_index=start_step,executed_action=None)
            for index in range(args.branch_steps):
                gaze=[]
                for _ in range(a.action_repeat):
                    frames=source.step(command);gaze.append(frames[0].meta.get('gaze_cmd') or (0.,0.))
                executed=LemLite.gaze_to_handle(command,np.asarray(gaze,np.float32).mean(0)[None])
                actor.step(frames,dev,step_index=start_step+index+1,executed_action=executed)
            rows.append(dict(name=name,bones=frames[0].bones.copy(),tokens={m:t.detach().cpu().numpy()
                for m,t in actor.model.last_tokens.items() if actor.model._last_source_mask.get(m)=='real' and m!='motor'}))
        repeated=rows[1];baseline=rows[0];changed=rows[2]
        physical_repeat=float(np.max(np.abs(repeated['bones']-baseline['bones'])))
        repeat_errors={m:float(np.mean((repeated['tokens'][m]-v)**2)) for m,v in baseline['tokens'].items()}
        effects={m:float(np.mean((changed['tokens'][m]-v)**2)) for m,v in baseline['tokens'].items()}
        floors={m:repeat_errors[m]+float(np.finfo(np.float32).eps**2*max(1.,np.mean(v*v)))
                for m,v in baseline['tokens'].items()}
        calibration_record=dict(physical_replay_max_difference=physical_repeat,repeated_token_mse=repeat_errors,
            model_proposal_vs_hold_token_mse=effects,effect_floors=floors,
            definition='Identical physical replay MSE plus one float32 relative-rounding-unit squared at observed token scale',
            limitation='One sampled arrangement; floors describe numerical repeatability, not skill or biological operating bounds')
        atomic_json(out/'sensory_calibration.json',calibration_record)
        if physical_repeat!=0 or any(v!=0 for v in repeat_errors.values()):
            raise ValueError('identical physical replay did not reproduce its raw tokens exactly')
        calibration['learning_opportunity']=dict(contract=CONTRACT,audit_every=1,anchor_capacity=2,window=2,
            effect_floors=floors,calibration=str(out/'sensory_calibration.json'))
        del actor
        receipt=calibration_for_candidate(learner,calibration,config)
        actor=FrozenActingSnapshot.from_learner(learner,factory,receipt,config,qualification=True)
        frozen_hash=perception_fingerprint(actor.model);frames=source.restore_full_state(initial_world)
        executed=None;held_command=None;optimizer_before=len(learner.optimizer.state)
        for index in range(args.learning_steps):
            step=start_step+index
            status('learning_in_copy',step=index,steps=args.learning_steps)
            result=learner.step(frames,dev,train=True,step_index=step,executed_action=executed)
            actor.step(frames,dev,step_index=step,executed_action=executed)
            if index%a.motor_hold==0 or held_command is None:held_command=result.motor_for_action.copy()
            gaze=[]
            for _ in range(a.action_repeat):
                frames=source.step(held_command,result.gaze_out);gaze.append(frames[0].meta.get('gaze_cmd') or (0.,0.))
            executed=LemLite.gaze_to_handle(held_command,np.asarray(gaze,np.float32).mean(0)[None])
            actor.prepare_learning_audit(learner,executed,step)
        copied=learner.save_checkpoint(start_step+args.learning_steps)
        save_episode(copied,source,learner,dict(step=start_step+args.learning_steps,executed=executed,
                                             actor=actor.state_dict()),dict(scope='copied integration check'))
        report=dict(declaration,complete=True,calibration=calibration_record,
            learner_weights_changed=perception_fingerprint(learner)!=model_hash,
            actor_weights_unchanged=perception_fingerprint(actor.model)==frozen_hash,
            populated_adam_before=optimizer_before,populated_adam_after=len(learner.optimizer.state),
            completed_comparisons=actor.opportunity.completed,retained_anchors=len(actor.opportunity.anchors),
            opportunity=actor.opportunity.last,checkpoint_saved=copied,
            simulated_seconds=float(frames[0].sim_time),production_license=False)
        if not report['learner_weights_changed'] or not report['actor_weights_unchanged']:
            raise ValueError('copied learning or frozen acting failed its declared integration gate')
        atomic_json(out/'report.json',report);status('complete',report='report.json');print(json.dumps(report),flush=True)
        return 0
    except Exception as error:
        status('failed',error=str(error));raise
    finally:source.close()


if __name__=='__main__':raise SystemExit(main())
