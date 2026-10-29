# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Frozen-body intervention records: relocated, absent and interrupted targets.

Each condition reloads the same brain and restores the identical physical start.
The target coordinates are used by the measuring tool only. Policy inputs are
the actual eyes, skin, body and executed commands, as in the training bridge.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.train_mimo import env_kwargs_for,build_learner
from mimo_bridge.lem_model import LemLite
from mimo_bridge import continuation_state
from mimo_bridge import copied_execution
from mimo_bridge.developmental_observation import derive_trace
from mimo_bridge.developmental_evidence import assess_trial
from .self_recovery_experiment import configuration,hold
from .caregiver_learning_trial import Orders
from .causal_test import _limits_for_run
from .training_continuation import atomic_json

PROBE_VERSION='physical_geometry_v3'
CONDITION_NAMES={'stationary':'present','relocated':'relocated','absent':'removed','interrupted':'interrupted','body_only':'present','free_objects':'present','support':'present'}
PROFILE_ABILITIES={
    'toy':('gaze_control','head_lift_and_look','reach_to_target','deliberate_strike'),
    'body_floor':('roll_over','head_control','weight_shift','prone_prop','sit_balance','crawl'),
    'free_objects':('grasp','hold_under_slip','place_release','bimanual_hold','proto_tool_use'),
    'support':('kneel','pull_to_stand','stand','cruise_first_step'),
}


def geometry_for_seed(seed):
    """Independent measured arrangements, not merely different random labels."""
    rng=np.random.default_rng(seed)
    return dict(target_distance_m=float(rng.uniform(.22,.34)),lateral_m=float(rng.uniform(-.035,.035)),
        vertical_m=float(rng.uniform(-.025,.025)),relocation_m=float(rng.choice([-1,1])*rng.uniform(.06,.10)),
        body_yaw_rad=float(rng.uniform(-.25,.25)),body_translation_xy_m=rng.uniform(-.025,.025,2).tolist())


def descendant_names(model,root_name):
    root=model.body(root_name).id;names=[]
    for body in range(model.nbody):
        parent=body
        while parent>0 and parent!=root:parent=int(model.body_parentid[parent])
        if parent==root:names.append(model.body(body).name)
    return names


def observation_context(e):
    """Actual anatomy and contact wiring; no unmeasured posture/pain limits."""
    return dict(chest_local_axes=np.asarray(e._body_orientation_axes).tolist(),
        geom_body_names={e.model.geom(i).name:e.model.body(int(e.model.geom_bodyid[i])).name or 'world' for i in range(e.model.ngeom)},
        avatar_body_names=descendant_names(e.model,'mimo_location'),support_geom_names=['floor'],
        hand_body_groups={s:descendant_names(e.model,s+'_hand') for s in ('left','right')},
        foot_body_groups={s:descendant_names(e.model,s+'_foot') for s in ('left','right')},
        arm_body_names=[s+'_'+part for s in ('left','right') for part in ('hand','lower_arm','upper_arm')],
        knee_body_names=['left_lower_leg','right_lower_leg'],target_body='target',
        object_body='box_light',free_object_body_names=['box_light','box_medium','box_heavy'],
        contact_min_force_n=0.,contact_noise_scope='MuJoCo solver positive normal force; a physical tolerance must still be declared for qualification.')


def target_line_of_sight(e,camera,target,present,target_geom='target_geom'):
    if not present:return False
    origin=e.data.cam_xpos[camera];direction=np.asarray(target)-origin
    length=np.linalg.norm(direction)
    if not np.isfinite(length) or length<=0:return False
    hit=np.array([-1],dtype=np.int32)
    distance=e._mj.mj_ray(e.model,e.data,origin,direction/length,None,1,int(e.model.cam_bodyid[camera]),hit)
    return bool(distance>=0 and int(hit[0])==e.model.geom(target_geom).id)


def independent_start(e,geometry,body_only):
    root=e.model.body('mimo_location').id;joint=int(e.model.body_jntadr[root])
    if joint<0 or e.model.jnt_type[joint]!=e._mj.mjtJoint.mjJNT_FREE:raise ValueError('body-only translation requires the actual free root')
    address=int(e.model.jnt_qposadr[joint])
    e.data.qpos[address:address+2]+=geometry['body_translation_xy_m']
    from .motor_reachability_probe import multiply_xyzw
    q=e.data.qpos[address+3:address+7][[1,2,3,0]].copy();theta=geometry['body_yaw_rad']/2
    q=multiply_xyzw(np.array([0.,0.,np.sin(theta),np.cos(theta)]),q)
    e.data.qpos[address+3:address+7]=q[[3,0,1,2]]
    if body_only:
        for name in ('box_light','box_medium','box_heavy'):
            body=e.model.body(name).id;joint=int(e.model.body_jntadr[body]);address=int(e.model.jnt_qposadr[joint])
            e.data.qpos[address:address+3]=[3+body*.05,0,1]
    e._mj.mj_forward(e.model,e.data)


def outcome_declaration(ability,threshold,dt,reference=None):
    """An absent physical reference leaves the outcome unassessable by construction."""
    if reference is not None:
        declaration=deepcopy(reference)
        if declaration.get('ability_id')!=ability:raise ValueError('outcome declaration names a different capability')
        return declaration
    criteria=([dict(metric='target_angle_deg',op='le',value=threshold),dict(metric='target_visible',op='eq',value=True)]
        if ability=='gaze_control' else [dict(metric='body_ventral_up_dot',op='le',value=-math.cos(math.radians(30))),
            dict(metric='body_supported',op='eq',value=True),dict(metric='body_rotation_from_start_deg',op='ge',value=150)] if ability=='roll_over' else [])
    return dict(ability_id=ability,criteria=criteria,minimum_dwell_s=.5,maximum_sample_gap_s=dt,
        calibration=dict(repeatability_verified=False,reachability_verified=False,
            receipt='',measurement_noise=0.),scope='Outcome geometry is defined, but its physical reachability is not yet verified.')


def body_signature(e,a,dev):
    digest=hashlib.sha256()
    for name in ('body_mass','body_inertia','jnt_range','dof_damping','geom_friction','actuator_gear','actuator_gainprm','actuator_forcerange'):
        digest.update(np.asarray(getattr(e.model,name)).tobytes())
    return dict(mechanics_sha256=digest.hexdigest(),body_layout=a.body_layout,rom_limit=float(dev['rom_limit']),
        strength_limit=float(dev['strength_limit']),body_function=a.body_function,
        action_repeat=a.action_repeat,motor_hold=a.motor_hold,smoothing=a.smooth_actions,
        gravity=e.model.opt.gravity.tolist(),timestep_s=float(e.model.opt.timestep),
        control_bounds=[e.decoder.ctrl_low.tolist(),e.decoder.ctrl_high.tolist()])


def verified_reference(declaration,signature,seed,condition='present'):
    """A bare true flag or an unrelated anatomy/world cannot license this trial."""
    reference=declaration.get('calibration',{}).get('receipt')
    receipt=Path(reference) if reference else None
    if receipt is None or not receipt.is_file():return False,'No measured physical reference receipt.'
    record=json.loads(receipt.read_text(encoding='utf-8'))
    if record.get('ability_id')!=declaration['ability_id'] or record.get('reachability_verified') is not True or record.get('same_action_replay_exact') is not True:
        return False,'Reference does not verify this ability and repeated physical execution.'
    if record.get('body_signature')!=signature:return False,'Reference mechanics or execution cadence differ.'
    if record.get('world_seed')!=seed:return False,'Reference did not establish reachability in this physical arrangement.'
    path=record.get('reference_traces',{}).get(condition,record.get('reference_trace'))
    if not path:return False,'The executed reference trace is missing.'
    path=Path(path);path=path if path.is_absolute() else receipt.parent/path
    if not path.is_file():return False,'The executed reference trace cannot be read.'
    trace=json.loads(path.read_text(encoding='utf-8'))
    if trace.get('world_seed')!=seed or trace.get('condition')!=condition:
        return False,'Reference trace belongs to another world or intervention condition.'
    if trace.get('arm')!='baseline' or trace.get('reference_action_source')!='scripted_reference':
        return False,'Reference trace has no declared physical-controller provenance.'
    judged=assess_trial(trace,declaration)
    return judged.get('success') is True,'Physical reference assessed against this exact outcome declaration.'


def angle(origin,forward,target):
    delta=np.asarray(target)-origin;length=np.linalg.norm(delta)
    if not np.isfinite(length) or length<=0:raise ValueError('invalid visual target')
    return float(np.degrees(np.arccos(np.clip(np.dot(forward,delta/length),-1,1))))


def longest_dwell(times,mask):
    times=np.asarray(times,dtype=float);mask=np.asarray(mask,dtype=bool)
    if len(times)!=len(mask) or len(times)<2 or not np.all(np.diff(times)>0):
        raise ValueError('invalid measurement clock')
    longest=duration=0.
    for i in range(1,len(times)):
        duration=duration+times[i]-times[i-1] if mask[i] and mask[i-1] else 0.
        longest=max(longest,duration)
    return float(longest)


def measure(e,frame,target,visible,target_geom='target_geom'):
    camera=e.model.camera('eye_left').id;origin=e.data.cam_xpos[camera]
    direction=-e.data.cam_xmat[camera].reshape(3,3)[:,2]
    chest=e.model.body('chest').id;hip=e.model.body('hip').id
    orientation=frame.meta.get('body_orientation',{})
    # The bridge publishes task measurements in metadata directly.
    if not orientation:
        from mimo_bridge.body_layout import chest_orientation
        orientation=chest_orientation(e.data.xmat[chest].reshape(3,3),e._body_orientation_axes)
    hands=np.stack([e.data.xpos[e.model.body(side+'_hand').id] for side in ('left','right')])
    objects={name:e.data.xpos[e.model.body(name).id].copy().tolist() for name in ('box_light','box_medium','box_heavy')}
    contacts=[]
    for i in range(e.data.ncon):
        contact=e.data.contact[i];force=np.zeros(6);e._mj.mj_contactForce(e.model,e.data,i,force)
        contacts.append(dict(geoms=[e.model.geom(int(g)).name for g in (contact.geom1,contact.geom2)],
            position_m=contact.pos.tolist(),force_contact_frame=force.tolist()))
    names=set(descendant_names(e.model,'mimo_location'))|{'box_light','box_medium','box_heavy','target'}
    positions={name:e.data.xpos[e.model.body(name).id].tolist() for name in names}
    quaternions={name:e.data.xquat[e.model.body(name).id].tolist() for name in ('chest','head')}
    e._mj.mj_subtreeVel(e.model,e.data)
    velocity=np.zeros(6);e._mj.mj_objectVelocity(e.model,e.data,e._mj.mjtObj.mjOBJ_BODY,chest,velocity,0)
    attachment={}
    active=getattr(e.data,'eq_active',e.model.eq_active0)
    for name in objects:
        body=e.model.body(name).id
        attachment[name]=any(bool(active[j]) and int(e.model.eq_type[j]) in (int(e._mj.mjtEq.mjEQ_WELD),int(e._mj.mjtEq.mjEQ_CONNECT))
            and body in (int(e.model.eq_obj1id[j]),int(e.model.eq_obj2id[j])) for j in range(e.model.neq))
    return dict(time_s=float(frame.sim_time),target_angle_deg=angle(origin,direction,target),
        target_present=bool(visible),target_visible=target_line_of_sight(e,camera,target,visible,target_geom),
        target_distance_m=float(np.min(np.linalg.norm(hands-target,axis=-1))),
        target_position_xyz_m=np.asarray(target).tolist(),eye_position_xyz_m=origin.tolist(),
        body_ventral_up_dot=orientation.get('front_up_component'),
        body_height_m=float(e.data.xpos[chest,2]),body_roll_deg=orientation.get('roll_degrees'),
        torso_quaternion_wxyz=e.data.xquat[chest].tolist(),
        center_of_mass_xyz_m=e.data.subtree_com[e.model.body('mimo_location').id].tolist(),
        hip_position_xyz_m=e.data.xpos[hip].tolist(),hand_positions_xyz_m=hands.tolist(),
        object_positions=objects,body_positions=positions,body_quaternions_wxyz=quaternions,
        center_of_mass_velocity_m_s=e.data.subtree_linvel[e.model.body('mimo_location').id].tolist(),
        torso_angular_velocity_rad_s=velocity[:3].tolist(),object_attachment_state=attachment,
        external_assistance=bool(np.any(e.data.xfrc_applied)),contacts=contacts,
        function_min=float(np.asarray(frame.meta['body_function']['values'])[:,0].min())
            if frame.meta.get('body_function') else None)


def run_world(a,checkpoint,out,seed,dev,warm,steps,profile,declared_outcome=None,progress=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=build_learner(a,str(out/'checkpoints'))
    loaded=model.load_checkpoint(str(checkpoint),load_optimizer=False)
    if loaded.get('reinitialized_components'):raise ValueError('probe lost checkpoint components')
    model.reset_transition_state()
    kw=env_kwargs_for(a,dev['phase_name'],str(out))
    kw.update(seed=seed,caregiver_mode='off',caregiver_seconds=0,caregiver_record_dir='',
        mobile=False,critters=False,gaze_reflex=False,spontaneous_saccades=False)
    if profile=='support':kw['support_fixture']='rail_v1'
    src=SingleEnvSource(**kw);e=src.env
    try:
        frames=src.reset();target_id=e._target_id;e._target_id=None
        geometry=geometry_for_seed(seed);independent_start(e,geometry,profile=='body_floor')
        support=None
        if profile=='support':
            from mimo_bridge.support_fixture import place
            support=place(e)
        geom=e.model.geom('target_geom').id
        # A stable fixed object and ordinary floor/body are sufficient to test
        # whether object position actually controls an eye/head movement.
        cid=e.model.camera('eye_left').id;origin=e.data.cam_xpos[cid].copy()
        basis=e.data.cam_xmat[cid].reshape(3,3).copy();forward=-basis[:,2];right=basis[:,0]
        center=origin+forward*geometry['target_distance_m']+right*geometry['lateral_m']+basis[:,1]*geometry['vertical_m']
        offset=right*geometry['relocation_m']
        e.data.mocap_pos[target_id]=center
        from .motor_reachability_probe import pose
        initial=pose(e,float(dev['strength_limit']))
        executed=None
        for i in range(warm):
            model.step(frames,dict(dev,exploration_noise=0.),train=False,step_index=model.step_count,
                executed_action=executed,executed_order_id=1)
            eye=[]
            for rep in range(a.action_repeat):
                frames=src.step(initial,np.zeros((1,2)),render=rep==a.action_repeat-1)
                eye.append(frames[0].meta['gaze_cmd'])
            executed=LemLite.gaze_to_handle(initial,np.mean(eye,axis=0)[None])
        world=src.save_full_state();execution=copied_execution.capture(model)
        context=observation_context(e)
        if profile=='free_objects':context['target_body']='box_light'
        if profile=='support':
            context['support_geom_names']+=['pen_wall_n','pen_wall_s','pen_wall_e','pen_wall_w']
            context['support_geom_names']+=['developmental_support_rail','developmental_support_left_post','developmental_support_right_post']
        conditions=['stationary','relocated','absent','interrupted'] if profile=='toy' else ['body_only' if profile=='body_floor' else profile]
        results=[];traces=[];replay_samples=[];replay_outputs=[]
        cases=[(condition,arm) for condition in conditions for arm in ('candidate','baseline')]
        for case_index,(condition,arm) in enumerate(cases):
            frames=src.restore_full_state(world);copied_execution.restore(model,execution)
            orders=Orders(a,seed);orders.executed=executed.copy();orders.order=1
            rows=[];times=[];commands=[];gazes=[];interruption_time=None
            primary=PROFILE_ABILITIES[profile][0]
            chosen_geom='box_light_geom' if profile=='free_objects' else 'developmental_support_rail' if profile=='support' else 'target_geom'
            initial_point=e.data.geom_xpos[e.model.geom(chosen_geom).id] if profile in ('free_objects','support') else center
            initial_measurement=measure(e,frames[0],initial_point,profile!='body_floor',chosen_geom)
            for i in range(steps):
                point=center+offset if condition=='relocated' or (condition=='interrupted' and i>=steps//2) else center
                visible=condition!='absent' and profile=='toy'
                if condition=='interrupted' and i==steps//2:interruption_time=float(frames[0].sim_time)
                e.data.mocap_pos[target_id]=point if visible else center+np.array([0,0,3.])
                e.model.geom_rgba[geom,3]=float(visible)
                e.model.geom_contype[geom]=int(visible);e.model.geom_conaffinity[geom]=int(visible)
                if arm=='baseline':
                    from types import SimpleNamespace
                    result=SimpleNamespace(motor_for_action=initial.copy(),gaze_out=np.zeros((1,2)))
                else:
                    if case_index==0 and i<max(6,a.motor_hold+1):
                        replay_samples.append(dict(frames=deepcopy(frames),dev=dict(dev,exploration_noise=0.),
                            executed_action=orders.executed.copy(),executed_order_id=orders.order))
                    result=model.step(frames,dict(dev,exploration_noise=0.),train=False,
                        step_index=model.step_count,executed_action=orders.executed,executed_order_id=orders.order)
                    if case_index==0 and i<max(6,a.motor_hold+1):
                        replay_outputs.append(copied_execution.output_record(model,result))
                frames=orders.advance(model,result,src,i,False)
                measured_point=e.data.geom_xpos[e.model.geom(chosen_geom).id] if profile in ('free_objects','support') else point
                rows.append(measure(e,frames[0],measured_point,visible or profile in ('free_objects','support'),chosen_geom));times.append(frames[0].sim_time)
                commands.append(orders.executed[0].tolist());gazes.append(np.asarray(result.gaze_out)[0].tolist())
                if i%25==0:
                    atomic_json(out.parent/'status.json',dict(progress or {},stage='measuring',pid=os.getpid(),timestamp=time.time(),
                        world=seed,condition=condition,condition_index=case_index,condition_count=len(cases),
                        arm=arm,frame=i,frames=steps,elapsed_seconds=time.time()-(progress or {}).get('started_at',time.time())))
            threshold=a.fovea_fov/2 if a.fovea_size else a.eye_fov/2
            in_view=np.array([row['target_visible'] and row['target_angle_deg']<=threshold for row in rows])
            summary=dict(condition=CONDITION_NAMES[condition],source_condition=condition,arm=arm,world_seed=seed,scored_seconds=times[-1]-times[0],
                target_center_in_fovea_fraction=float(in_view.mean()),
                longest_foveal_dwell_s=longest_dwell(times,in_view),
                minimum_target_angle_deg=min(row['target_angle_deg'] for row in rows),
                minimum_front_up=min(row['body_ventral_up_dot'] for row in rows),
                maximum_horizontal_displacement_m=max(np.linalg.norm(np.array(row['hip_position_xyz_m'])[:2]-np.array(rows[0]['hip_position_xyz_m'])[:2]) for row in rows))
            derived=derive_trace([initial_measurement]+rows,context)
            measurements={key:[row[key] for row in rows] for key in rows[0]}
            measurements.update({key:value[1:] for key,value in derived['measurements'].items()})
            trace=dict(ability_id=primary,profile=profile,candidate_abilities=PROFILE_ABILITIES[profile],world_seed=seed,pair_id=str(seed),arm=arm,
                phase='acquisition',condition=CONDITION_NAMES[condition],source_condition=condition,time_s=times,model_authored=[arm=='candidate']*steps,
                external_assistance=[row['external_assistance'] for row in rows],task_valid=[True]*steps,
                measurements=measurements,unavailable=derived['unavailable'],observation_context=context,
                reference_action_source='pose_hold' if arm=='baseline' else None,
                interruption_time_s=interruption_time,
                commands=commands,gaze_commands=gazes,summary=summary,
                provenance=dict(checkpoint=str(checkpoint),raw_physics=True,optimizer_updates=0,probe_version=PROBE_VERSION,
                    geometry=geometry,full_learner_and_tracker_restore=True,
                    scheduled_stillness=False,reflex=False,saccades=False,
                    warmup='Achieved-pose hold and neutral eyes; no target-direction instruction'))
            name=condition+'.json' if arm=='candidate' else ('hold_control.json' if condition in ('stationary','body_only','free_objects','support') else condition+'_hold.json')
            atomic_json(out/name,trace);results.append(summary);traces.append((name,trace))
        replay_receipt=copied_execution.verify_recorded_frames(model,execution,replay_samples,replay_outputs)
        atomic_json(out/'model_replay.json',replay_receipt)
        # Same-command full replay is an instrument test, not a behavior claim.
        replay=[]
        for _ in range(2):
            src.restore_full_state(world)
            replay.append([src.step(initial,np.zeros((1,2)),render=False)[0].bones.copy() for _ in range(5)])
        np.testing.assert_array_equal(replay[0],replay[1])
        declaration=outcome_declaration(traces[0][1]['ability_id'],threshold,e.dt*a.action_repeat,declared_outcome)
        declaration['calibration']['repeatability_verified']=True
        declaration['calibration']['measurement_noise']=float(np.max(np.abs(np.asarray(replay[0])-np.asarray(replay[1]))))
        # Physical reachability remains independent of identical replay. A missing
        # reference is never promoted because a body or gaze sensor is available.
        signature=body_signature(e,a,dev)
        declaration['body_signature']=signature
        atomic_json(out/'outcome_declaration.json',declaration)
        for name,trace in traces:
            trial_declaration=deepcopy(declaration)
            if trial_declaration['calibration'].get('reachability_verified'):
                verified,reason=verified_reference(trial_declaration,signature,seed,trace['condition'])
                trial_declaration['calibration']['reachability_verified']=verified
                trial_declaration['calibration']['reference_verification']=reason
            trace['assessment']=assess_trial(trace,trial_declaration)
            trace['outcome_declaration']=trial_declaration
            atomic_json(out/name,trace)
        result=dict(seed=seed,geometry=geometry,support_fixture=support,conditions=results,same_action_replay_exact=True,
            complete_model_replay=replay_receipt,outcome_assessments=[trace['assessment'] for _,trace in traces])
        atomic_json(out/'report.json',result)
        return result
    finally:src.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',choices=list(PROFILE_ABILITIES),default='toy')
    p.add_argument('--worlds',type=int,default=4);p.add_argument('--steps',type=int,default=160)
    p.add_argument('--seed',type=int,default=51321);p.add_argument('--smoke',action='store_true')
    p.add_argument('--outcome-declaration',type=Path,help='Optional predeclared physical outcome with an actual reachability receipt')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if args.worlds<1 or args.steps<2:p.error('invalid measured worlds or duration')
    a=configuration(args.checkpoint,args.out);a.device='cpu';a.self_choice='off'
    # A production acting snapshot must be tested as its own controller, rather
    # than silently substituting a simultaneously updated learner checkpoint.
    source_args=json.loads((args.checkpoint.parent.parent/'args.json').read_text())
    if source_args.get('self_choice')=='on':raise ValueError('use the saved acting snapshot for a live-actor behavior claim')
    from types import SimpleNamespace
    dev,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
    if not dev.get('phase_name'):raise ValueError('missing actual body-development limits')
    warm=3 if args.smoke else int(a.context_length);steps=4 if args.smoke else args.steps
    declared_outcome=None if args.outcome_declaration is None else json.loads(args.outcome_declaration.read_text(encoding='utf-8'))
    declaration=dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        profile=args.profile,worlds=args.worlds,steps=steps,probe_version=PROBE_VERSION,
        warm_steps=warm,seeds=list(range(args.seed,args.seed+args.worlds)),limits=dev,limits_source=origin,
        question='Does relocation, absence or interruption change model-authored looking? Are body-only postural changes visible?',
        scale='Fraction: target center inside actual foveal half-angle; dwell in seconds; body front +1 up, -1 down.',
        scope='Descriptive frozen-checkpoint intervention test. No acquired skill is certified without calibrated reachability, matched comparisons and retention.',
        instrument_smoke=args.smoke,pass_policy='No numerical change alone certifies intention; report all conditions and misses.')
    atomic_json(args.out/'declaration.json',declaration)
    results=[];durations=[];started=time.time()
    try:
        for seed in declaration['seeds']:
            out=args.out/str(seed);out.mkdir();start=time.time()
            progress=dict(world_index=len(results),world_count=args.worlds,completed_durations_seconds=durations,
                started_at=started,step=len(results),steps=args.worlds)
            results.append(run_world(a,args.checkpoint,out,seed,dev,warm,steps,args.profile,declared_outcome,progress));durations.append(time.time()-start)
            progress.update(world_index=len(results),step=len(results),completed_durations_seconds=durations,
                elapsed_seconds=time.time()-started)
            atomic_json(args.out/'progress.json',progress)
            atomic_json(args.out/'status.json',dict(progress,stage='measuring',pid=os.getpid(),timestamp=time.time()))
        atomic_json(args.out/'report.json',dict(verdict='INSTRUMENT_EXECUTED' if args.smoke else 'CONTROLLED_BEHAVIOR_RECORDED',
            results=results,capability_pass=False,scope=declaration['scope'],completed_at=time.time()))
        atomic_json(args.out/'status.json',dict(progress,stage='complete',timestamp=time.time(),
            verdict='CONTROLLED_BEHAVIOR_RECORDED',pid=os.getpid()))
    except Exception as error:
        atomic_json(args.out/'status.json',dict(stage='failed',timestamp=time.time(),error=str(error)));raise


if __name__=='__main__':main()
