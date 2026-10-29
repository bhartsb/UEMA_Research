"""Frozen, same-state physical-goal planning: direct hand space versus synergy.

Each goal is an observed endpoint of an independent physical reference from the
same saved body. Both planners share strength and gaze inside optimization;
execution uses their selected joint goals through the real moving-wrist path.

Goal suites: 'hand' closes and opens the fingers; 'hand_and_body' adds a left-wrist
reach, a supported trunk turn and a left-elbow flex. Every trial reports whether its
witness request moved the requested part beyond where the controller's own command
took it and the mirrored request (the same change reversed, run once as the control)
to the other side (witness_check, in cm or degrees; WITNESS_RULE), and every result
carries the outside review's ranking statistic over the arms of each trial (ranking:
how many in 100 pairs of commands the forecast orders as their actual goal errors are
ordered, per goal and pooled; mimo_bridge/tools/command_ranking.py). Two of the three
arms (OPTIMIZED_ARMS) were found by searching that same forecast.
"""
from __future__ import annotations
import os
if __name__ == "__main__":  # this probe runs on the CPU; importing the probe harness no longer hides the card
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # (2026-09-26, probe_gpu_hiding_2026-09-26.md)
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mimo_bridge.tools import gof_probe_harness as H
from mimo_bridge.tools.commit_guard import commit_reading, verdict
from mimo_bridge.tools.command_ranking import RULES as RANKING_RULES, ranking as rank_commands
from mimo_bridge.hand_target_control import HandTargetControl, HandActionSpace, HandKinematics, CONTRACT
from mimo_bridge.lem_model import LemLite
from lem.model.efference import EFFERENCE_SCALES
from lem.model.physical_goal import pose_error_per_body
from compare_synergy_physical import checksum

GOAL_SUITES={'hand':['curl_witness','extension_witness'],
             'hand_and_body':['curl_witness','extension_witness','wrist_reach_witness',
                              'supported_trunk_turn_witness','elbow_flex_witness']}
# The body witnesses ask for one bound of the planners' own search region (PhysicalTrustRegion): a
# handle's requested turn at most 11 degrees, its requested position at most 1.8 cm per coordinate
# from the controller's request (the measured 250 ms movement scales), so the planners can make the
# request themselves.
TURN_DEG=EFFERENCE_SCALES['rot_tenure_deg']
REACH_CM=EFFERENCE_SCALES['pos_tenure_cm']
ELBOW_JOINT='robot:left_elbow'      # MIMo's elbow hinge: 5 degrees extended to -146 degrees flexed
FINGER_JOINTS=('_knuckle','_middle','_distal')
# what each witness asks for, as the quantity its check measures: the finger joints (degrees), a
# handle position (cm), handle orientations (degrees) or, for the elbow, the elbow joint's own angle
# (degrees, the forearm's requested turn projected on the hinge axis as the request)
WITNESS_PARTS={'curl_witness':('joints',()),'extension_witness':('joints',()),
               'wrist_reach_witness':('position',('wrist_L',)),
               'supported_trunk_turn_witness':('rotation',('lower_trunk','chest')),
               'elbow_flex_witness':('elbow',('lowerarm_L',))}
WITNESS_RULE=("A witness is informative for a body when its request moved the requested part beyond the "
    "controller's endpoint in the requested direction AND further along that direction than the mirrored "
    "request (the same change from the controller's request, reversed) did: the difference is the request's "
    "doing, not a push that happens whatever is asked. A mirror the body cannot follow (a joint at its limit, a "
    "wrist on the mat) stays near the controller's endpoint and does not count against the witness. Each is an "
    "endpoint minus the controller's endpoint, projected on the executed request (the witness's executed "
    "command minus the controller's, after the smoother). For the elbow the measured quantity is the elbow "
    "joint's own angle (flexion positive), so shoulder motion cannot count. requested_raw is the request "
    "before the smoother, requested_executed after it, mirror_requested_executed the mirror's executed request "
    "on the same direction (negative when it asks for the reverse; mirror_request_blocked when it does not), "
    "achieved_beyond_controller the witness's projection and mirror_beyond_controller the mirrored "
    "request's, achieved_share and mirror_share the two over requested_executed (1 = the whole executed "
    "request happened, 0 = the body ended where the controller took it), separation_from_controller the "
    "whole difference between the witness's and the controller's endpoints and moved_from_start the witness "
    "endpoint's distance from the saved start; sizes are root mean squares over the requested parts, in cm "
    "for a position, degrees for a turn or a joint.")


@torch.no_grad()
def forecast_pose(model,current,decision,command):
    """The same candidate transition and held-order forecast used by the planner."""
    from lem.model.predictive_state import PredictiveState,transition
    from lem.model.encoder_registry import normalize_executed_command
    from lem.model.efference import EfferenceHeads
    from lem.model.motor_synergy import quat_mul,rotvec_to_quat
    state=PredictiveState(tokens=current,context=decision['context'],memory_bias=decision.get('memory_bias'))
    order=normalize_executed_command(command)
    generator=model.intention_gen
    predicted=transition(generator._forward_model,generator._encoder_registry,generator._prediction_heads,
        state,order,achieved_state=decision['achieved'],advance_context=False,include_latent=True)
    effects=EfferenceHeads.to_lived(body=model.efference.body(predicted.transition_latent,order,decision['achieved']))
    pose=decision['achieved'].clone()
    pose[...,:3]+=effects['body_tenure_cm']/200.
    pose[...,3:7]=quat_mul(rotvec_to_quat(effects['body_tenure_deg']*np.pi/180.),pose[...,3:7])
    return pose


def filtered(model, raw, joints, decision):
    command=LemLite.planned_execution(raw,decision['dev'],decision['achieved'],
        previous=decision['smooth_prev'],smoothing=decision['smoothing'],
        keep_gaze_idx=decision.get('smooth_keep_gaze'))
    joint_goal=model._smooth_hand_joints(joints,decision.get('hand_previous_joints'),decision['smoothing'])
    return model.hand_kinematics.assemble_world(command,joint_goal,1.).detach(),joint_goal.detach()


def physical(session,snapshot,command,joint_goal,gaze,target=None,target_joints=None,tracking=True):
    session.restore(snapshot)
    env=session.env;control=env.hand_target_control
    env.hand_target_mode='wrist_relative_v1' if tracking else 'world'
    if not tracking:env.hand_target_control=None
    from mimo_bridge.tools.motor_reachability_probe import posture
    started=float(env.data.time);digest=hashlib.sha256();support=[]
    for step in range(25):
        raw=env.step(command[0].numpy(),gaze=gaze,
            hand_joint_goal=joint_goal[0].numpy() if tracking else None,render=False)
        if tracking:
            canonical=raw.info['canonical_motor_targets'];chosen=command[0].numpy()
            np.testing.assert_allclose(canonical[:,:3],chosen[:,:3],atol=2e-5,rtol=1e-6)
            qa=canonical[:,3:7]/np.linalg.norm(canonical[:,3:7],axis=-1,keepdims=True)
            qb=chosen[:,3:7]/np.linalg.norm(chosen[:,3:7],axis=-1,keepdims=True)
            np.testing.assert_allclose(abs((qa*qb).sum(-1)),1.,atol=1e-6)
            np.testing.assert_array_equal(canonical[:,7:],chosen[:,7:])
            np.testing.assert_array_equal(raw.info['hand_joint_goal'],joint_goal[0].numpy())
        if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
            raise RuntimeError('Nonfinite physical state')
        digest.update(env.data.qpos.tobytes());digest.update(env.data.qvel.tobytes())
        digest.update((raw.info['executed_motor_targets'] if tracking else command[0].numpy()).tobytes())
        support.append(posture(env))
    assert np.isclose(float(env.data.time)-started,.25,rtol=0,atol=1e-8)
    pos,quat=session.pose()
    pose=torch.zeros(1,len(pos),10,dtype=torch.float32)
    pose[0,:,:3]=torch.from_numpy(pos/200.);pose[0,:,3:7]=torch.from_numpy(quat)
    actual_joints=torch.from_numpy(env.data.qpos[control._joint_addresses].copy())[None].float()
    result=dict(trajectory_sha256=digest.hexdigest(),pose=pose.tolist(),joint_angles_rad=actual_joints.tolist(),
                qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),seconds=float(env.data.time)-started,
                support=dict(every_step=all(p['supported'] for p in support),records=support))
    if target is not None:
        body=[i for i,name in enumerate(session.handle_names) if name!='gaze']
        result['physical_goal_error']=float(pose_error_per_body(pose[:,:,:3]*200,pose[:,:,3:7],target,body)[0])
        result['hand_joint_error_deg']=float(torch.rad2deg(abs(actual_joints-target_joints)).mean())
        wrists=[session.handle_names.index('wrist_'+s) for s in ('L','R')]
        result['wrist_goal_error_cm']=float((pose[:,wrists,:3]-target[:,wrists,:3]).norm(dim=-1).mean()*200)
        fingers=[i for h in control.geometry['hands'] for i in h['finger_indices']]
        other=[i for i in body if i not in fingers+wrists]
        result['goal_error_by_part']={name:float(pose_error_per_body(pose[...,:3]*200.,pose[...,3:7],target,indices)[0])
            for name,indices in dict(fingers=fingers,wrists=wrists,other_body=other).items()}
    env.hand_target_mode='wrist_relative_v1';env.hand_target_control=control
    return result,pose,actual_joints


def witness_request(session,raw,joints,goal_name,mirror=False):
    """Researcher-chosen request; its actual endpoint becomes the reachable goal. mirror: the same
    change from the controller's request reversed (the control of WITNESS_RULE); its endpoint is never a
    goal. The session must be at the saved start (the trunk and elbow read the body's current axes)."""
    command=raw.clone();research=joints.clone();geometry=session.model._hand_geometry
    sign=-1. if mirror else 1.
    if goal_name in ('curl_witness','extension_witness'):
        for index,joint in enumerate(geometry['joints']):
            if joint['name'].endswith(FINGER_JOINTS):
                target=.95*joint['limits'][1 if goal_name=='curl_witness' else 0]
                research[:,index]=2.*joints[:,index]-target if mirror else target
        recipe='Compatible curl or extension; other command fields retain the controller.'
    elif goal_name=='wrist_reach_witness':
        index=session.handle_names.index('wrist_L')
        command[:,index,2]+=sign*REACH_CM/200.
        recipe=f"Request the left wrist {REACH_CM:g} cm above the controller's own request; all other requests unchanged."
    elif goal_name=='supported_trunk_turn_witness':
        env=session.env;chest=env.model.body('chest').id
        axis=env.data.xmat[chest].reshape(3,3)@env._body_orientation_axes[0]
        turn=Rotation.from_rotvec(sign*axis/np.linalg.norm(axis)*np.deg2rad(TURN_DEG))
        for name in ('lower_trunk','chest'):
            index=session.handle_names.index(name)
            command[0,index,3:7]=torch.as_tensor((turn*Rotation.from_quat(command[0,index,3:7].numpy())).as_quat(),dtype=command.dtype)
        recipe=f'Rotate both trunk requested orientations {TURN_DEG:g} degrees about the current anatomical length axis; positions unchanged. Support is measured at every 10 ms step.'
    elif goal_name=='elbow_flex_witness':
        command=elbow_flex_request(session,command,sign)
        recipe=(f"Flex the left elbow: the controller's forearm and hand requests turned {TURN_DEG:g} degrees about the "
                "elbow hinge's current axis through its anchor, toward flexion, and the requested wrist position "
                f"carried with them (shortened along its own direction to at most {REACH_CM:g} cm per coordinate); "
                "all other requests unchanged.")
    else:raise ValueError(goal_name)
    return command,session.model.hand_kinematics.project_joints(research),recipe


def elbow_flex_request(session,command,sign=1.):
    """The controller's request with the left forearm and hand turned TURN_DEG about the elbow hinge's
    world axis through its anchor, toward flexion (MIMo flexes the elbow toward its lower limit, so
    flexing is turning by minus the angle about the hinge axis); sign -1 turns toward extension (the
    mirrored control). The requested wrist position moves with the hand; a move with a coordinate over
    REACH_CM is shortened along its own direction, so the request stays inside the planners' search
    region around the controller's request without leaving the arc's direction."""
    env=session.env;elbow=env.model.joint(ELBOW_JOINT).id
    axis=np.asarray(env.data.xaxis[elbow],dtype=np.float64)
    anchor=np.asarray(env.data.xanchor[elbow],dtype=np.float64)*100.
    turn=Rotation.from_rotvec(-sign*axis/np.linalg.norm(axis)*np.deg2rad(TURN_DEG))
    command=command.clone()
    for name in ('lowerarm_L','wrist_L'):
        index=session.handle_names.index(name)
        command[0,index,3:7]=torch.as_tensor((turn*Rotation.from_quat(command[0,index,3:7].numpy())).as_quat(),dtype=command.dtype)
    wrist=session.handle_names.index('wrist_L')
    requested=command[0,wrist,:3].numpy().astype(np.float64)*200.
    moved=anchor+turn.apply(requested-anchor)-requested
    largest=float(np.abs(moved).max())
    if largest>REACH_CM:moved=moved*(REACH_CM/largest)
    command[0,wrist,:3]=torch.as_tensor((requested+moved)/200.,dtype=command.dtype)
    return command


def _turn_deg(target_xyzw,base_xyzw):
    """The world-frame turn from base to target orientation, as a rotation vector in degrees."""
    return np.degrees((Rotation.from_quat(np.asarray(target_xyzw,dtype=np.float64))*
                       Rotation.from_quat(np.asarray(base_xyzw,dtype=np.float64)).inv()).as_rotvec())


def witness_check(goal_name,handle_names,finger_joints,start,raw,executed,ends,start_joints,raw_joints,
                  executed_joints,end_joints,elbow=None):
    """Did the witness request move the requested part beyond where the controller's own command took it,
    and further than the mirrored request did? (WITNESS_RULE). start: the saved start pose [H, 7+]
    (positions cm, orientations x y z w); raw: dict(controller=..., witness=...) and executed:
    dict(controller=..., witness=..., mirror=...) of [H, 7+] requests before and after the smoother;
    ends: dict(controller=..., witness=..., mirror=...) endpoint poses in the same units; start_joints
    and the joint dicts: hand joint angles and goals [J] in radians (executed_joints and end_joints with
    'mirror' too); finger_joints: the curl/extension witnesses' joint indices; elbow: for the elbow
    witness, dict(axis=the hinge's world axis at the start, start=, controller=, witness=, mirror=)
    elbow angles in radians."""
    kind,handles=WITNESS_PARTS[goal_name]
    if kind=='joints':
        parts=list(finger_joints)
        def diff(a,b):return np.degrees(np.asarray(a,dtype=np.float64)[parts]-np.asarray(b,dtype=np.float64)[parts])
        requested_raw=diff(raw_joints['witness'],raw_joints['controller'])
        requested=diff(executed_joints['witness'],executed_joints['controller'])
        mirror_request=diff(executed_joints['mirror'],executed_joints['controller']) if 'mirror' in executed_joints else None
        achieved=diff(end_joints['witness'],end_joints['controller'])
        mirrored=diff(end_joints['mirror'],end_joints['controller']) if 'mirror' in end_joints else None
        moved=diff(end_joints['witness'],start_joints)
    elif kind=='elbow':
        # the forearm's requested turn about the flexion direction, against the elbow joint's own angle
        # (MIMo's elbow flexes toward its lower limit: flexing lowers the angle)
        parts=[handle_names.index(name) for name in handles]
        flexion=-np.asarray(elbow['axis'],dtype=np.float64);flexion/=np.linalg.norm(flexion)
        def turn(a,b):return np.array([float(np.dot(_turn_deg(np.asarray(a,dtype=np.float64)[parts[0],3:7],
                                                              np.asarray(b,dtype=np.float64)[parts[0],3:7]),flexion))])
        def flexed(key,base='controller'):return np.array([np.degrees(float(elbow[base])-float(elbow[key]))])
        requested_raw=turn(raw['witness'],raw['controller'])
        requested=turn(executed['witness'],executed['controller'])
        mirror_request=turn(executed['mirror'],executed['controller']) if 'mirror' in executed else None
        achieved=flexed('witness')
        mirrored=flexed('mirror') if 'mirror' in elbow else None
        moved=flexed('witness','start')
    else:
        parts=[handle_names.index(name) for name in handles]
        if kind=='position':
            def diff(a,b):return np.concatenate([np.asarray(a,dtype=np.float64)[i,:3]-np.asarray(b,dtype=np.float64)[i,:3] for i in parts])
        else:
            def diff(a,b):return np.concatenate([_turn_deg(np.asarray(a,dtype=np.float64)[i,3:7],np.asarray(b,dtype=np.float64)[i,3:7]) for i in parts])
        requested_raw=diff(raw['witness'],raw['controller'])
        requested=diff(executed['witness'],executed['controller'])
        mirror_request=diff(executed['mirror'],executed['controller']) if 'mirror' in executed else None
        achieved=diff(ends['witness'],ends['controller'])
        mirrored=diff(ends['mirror'],ends['controller']) if 'mirror' in ends else None
        moved=diff(ends['witness'],start)
    size=lambda v:float(np.linalg.norm(v)/np.sqrt(max(len(parts),1)))
    denominator=float(np.dot(requested,requested))
    along=lambda v:(float(np.dot(v,requested)/denominator) if denominator>0 and v is not None else None)
    share,mirror,asked_back=along(achieved),along(mirrored),along(mirror_request)
    out=dict(kind=kind,parts=('finger joints' if kind=='joints' else 'left elbow joint' if kind=='elbow' else list(handles)),
             unit='cm' if kind=='position' else 'degrees',
             requested_raw=size(requested_raw),requested_executed=size(requested),
             mirror_requested_executed=None if asked_back is None else asked_back*size(requested),
             mirror_request_blocked=None if asked_back is None else bool(asked_back>=0),
             achieved_beyond_controller=None if share is None else share*size(requested),
             mirror_beyond_controller=None if mirror is None else mirror*size(requested),
             achieved_share=share,mirror_share=mirror,separation_from_controller=size(achieved),
             moved_from_start=size(moved),
             informative=bool(share is not None and share>0 and mirror is not None and share>mirror))
    if kind=='elbow':
        out['elbow_start_deg']=float(np.degrees(float(elbow['start'])))
    return out


def witness_summary(checks,bodies):
    """One goal's witness checks over its bodies: how many were informative, how many mirrors did not even
    ask for the reverse move, and the medians over every body."""
    def median(key):
        values=[c[key] for c in checks if c.get(key) is not None]
        return float(np.median(values)) if values else None
    return dict(bodies=len(checks),expected_bodies=bodies,informative_bodies=int(sum(bool(c['informative']) for c in checks)),
                mirror_request_blocked_bodies=int(sum(c.get('mirror_request_blocked') is True for c in checks)),
                unit=checks[0]['unit'] if checks else None,parts=checks[0]['parts'] if checks else None,
                **{key:median(key) for key in ('requested_raw','requested_executed','mirror_requested_executed',
                                              'achieved_beyond_controller','mirror_beyond_controller',
                                              'achieved_share','mirror_share','separation_from_controller',
                                              'moved_from_start')})


def ranking_of(rows,arms,reference,what):
    """The outside review's statistic over the arms of each trial: the forecast's goal error
    (predicted_goal_error) against the actual one (physical_goal_error), bodies as samples, the planners'
    actual goal error against the reference arm deciding case 3."""
    starts=[dict(body=row['body'],names=list(arms),
                 predicted=[row['outcomes'][m]['predicted_goal_error'] for m in arms],
                 actual=[row['outcomes'][m]['physical_goal_error'] for m in arms]) for row in rows]
    return rank_commands(starts,what,planners=('direct','synergy'),reference=reference)


# the arms whose commands were found by searching this same forecast: every pair of a trial contains
# one, so the pairs favor the forecast's own errors (the planner probe's without_chosen has no such arm)
OPTIMIZED_ARMS=('direct','synergy')


def ranking_and_witnesses(rows,goals,arms,reference,bodies):
    """The result's ranking block (pooled over every goal and per goal) and each goal's witness
    summary, from the trial rows."""
    ranking=dict(pooled=ranking_of(rows,arms,reference,'every goal of the suite'),
                 by_goal={name:ranking_of([row for row in rows if row['goal']==name],arms,reference,name)
                          for name in goals},
                 arms=list(arms),reference=reference,optimized_arms=[a for a in arms if a in OPTIMIZED_ARMS],
                 rules=RANKING_RULES)
    witnesses={name:witness_summary([row['witness_check'] for row in rows if row['goal']==name],bodies)
               for name in goals}
    return ranking,witnesses


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bodies',type=int,default=8)
    ap.add_argument('--initializer',choices=['controller','goal_pose'],default='controller')
    ap.add_argument('--compare-execution-contracts',action='store_true')
    ap.add_argument('--goal-suite',choices=['hand','hand_and_body'],default='hand')
    args=ap.parse_args()
    reading=commit_reading()
    if reading is None or not verdict(reading[0],reading[1],12,6):raise RuntimeError(f'Memory guard refused: {reading}')
    args.out.mkdir(parents=True,exist_ok=False);started=time.time()
    projection_timing=dict(calls=0,elapsed_s=0.,maximum_s=0.,failed_calls=0)
    project_world_orders=HandKinematics.project_world_orders
    def measured_projection(self,*positional,**keyword):
        began=time.perf_counter()
        try:return project_world_orders(self,*positional,**keyword)
        except Exception:
            projection_timing['failed_calls']+=1
            raise
        finally:
            elapsed=time.perf_counter()-began
            projection_timing['calls']+=1;projection_timing['elapsed_s']+=elapsed
            projection_timing['maximum_s']=max(projection_timing['maximum_s'],elapsed)
    timing_patch=patch.object(HandKinematics,'project_world_orders',measured_projection)
    timing_patch.start()
    goals=list(GOAL_SUITES[args.goal_suite])
    declaration=dict(checkpoint=str(args.checkpoint),bodies=args.bodies,goals=goals,goal_suite=args.goal_suite,
        contract=CONTRACT,learning=False,production_changed=False,hold_seconds=.25,
        initializer=args.initializer,
        compare_execution_contracts=args.compare_execution_contracts,
        goal_source=('Observed endpoint of a same-state physical reference: compatible curl/extension, a left-wrist reach, a coordinated trunk turn or a left-elbow flex. Every reference retains ordinary force and gaze settings.' if args.goal_suite=='hand_and_body' else
            'Observed endpoint of a same-state physical reference with chosen compatible curl/extension, ordinary wrist order and force settings.'),
        witness_rule=WITNESS_RULE,ranking_rules=RANKING_RULES,
        ranking_forecast="each arm's forecast goal error (the planners' own held-order pose forecast against the goal pose) against its actual goal error 0.25 seconds after the command, over the arms of each trial",
        comparison='Starting controller, direct feasible-hand planner, composed learned-synergy/feasible-hand planner. Strength and gaze are fixed to the controller inside both searches.',
        gaze_scope='Body-stage mechanism comparison: visual gaze optimization is deliberately held fixed in every arm; this does not qualify the combined visual/body policy.',
        hand_memory_scope='The separate calibrated hand-memory choice stage is disabled in this physical-body planner comparison; its learned calibration remains preserved.',
        adoption='Compare common physical-goal error and hand-joint error across independent bodies; no switch is changed by this experiment.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    root=Path(__file__).resolve().parents[2]
    declaration['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [root/'Source/Python/mimo_bridge/hand_target_control.py',
                  root/'Source/Python/mimo_bridge/lem_model.py',
                  root/'Source/Python/lem/model/physical_goal.py',
                  root/'Source/Python/mimo_bridge/env.py',
                  root/'Source/Python/mimo_bridge/tools/gof_probe_harness.py']}
    declaration['seed']=0
    (args.out/'declaration.json').write_text(json.dumps(declaration,indent=2))
    (args.out/'status.json').write_text(json.dumps(dict(state='running',completed=0,
        total=args.bodies*len(goals),started_at=started,elapsed_s=0)))
    frozen=H.load_frozen(args.checkpoint,intention=True,threads=4)
    episode,note=H.load_episode(frozen)
    if episode is None:raise RuntimeError(note)
    model=frozen.model;initial_weights=checksum(model);rows=[]
    if model.synergy is None:raise RuntimeError('Learned synergy required')
    for body in range(args.bodies):
        # Derive the learner's actual present inputs once; no physical warm-up.
        model.hand_target_mode='world';model.hand_goals='off';model.synergy_planner=False
        session=H.Session(frozen,episode=episode,body=body)
        try:
            previous_motor=(None if model._prev_mfa is None else model._prev_mfa.detach().clone())
            previous_hand=getattr(model,'_hand_previous_joints',None)
            previous_hand=None if previous_hand is None else previous_hand.detach().clone()
            generator=model.intention_gen;model.intention_gen=None
            try:session.observe()
            finally:model.intention_gen=generator
            # Observe supplies the current tokens, but its unused body proposal
            # must not become the smoother's previous command for this decision.
            model._prev_mfa=previous_motor
            env=session.env;env.hand_target_mode='wrist_relative_v1'
            env.hand_target_control=HandTargetControl(env.model,env.decoder)
            session.src.env_kwargs['hand_target_mode']='wrist_relative_v1'
            control=env.hand_target_control;kin=control.kinematics;geometry=control.geometry
            if body==0:
                (args.out/'hand_geometry.json').write_text(json.dumps(geometry,indent=2))
            previous=(env.data.qpos[control._joint_addresses].copy()[None] if previous_hand is None
                      else previous_hand)
            model.configure_hand_geometry(geometry,previous)
            model.hand_target_mode='wrist_relative_v1';model.physical_goals='memory'
            raw=torch.from_numpy(session.res.motor_out).float()
            original_raw=raw.clone()
            canonical,initial_joints,_=kin.project_world_orders(raw.numpy())
            raw=torch.from_numpy(canonical).float();initial_joints=torch.from_numpy(initial_joints).float()
            decision=model._capture_plan_state()
            current={k:v.squeeze(1).detach().float().clone() for k,v in model.last_tokens.items()}
            snapshot=session.snapshot()
            initial_joint_angles=env.data.qpos[control._joint_addresses].copy()
            elbow_id=env.model.joint(ELBOW_JOINT).id
            elbow_address=int(env.model.jnt_qposadr[elbow_id])
            initial_elbow=float(env.data.qpos[elbow_address])
            elbow_axis=np.asarray(env.data.xaxis[elbow_id],dtype=np.float64).copy()
            fingers=[i for i,joint in enumerate(geometry['joints']) if joint['name'].endswith(FINGER_JOINTS)]
            joint_dofs=[int(env.model.jnt_dofadr[joint['source_id']]) for joint in geometry['joints']]
            initial_joint_velocity=env.data.qvel[joint_dofs].copy()
            base_command,base_joints=filtered(model,raw,initial_joints,decision)
            gaze=np.asarray(model._gaze_from_handle(raw)).reshape(-1)[:2]
            for goal_name in declaration['goals']:
                trial_started=time.perf_counter();projection_before=dict(projection_timing)
                session.restore(snapshot)
                request,research,recipe=witness_request(session,raw,initial_joints,goal_name)
                witness_command,witness_joints=filtered(model,request,research,decision)
                witness,target,target_joints=physical(session,snapshot,witness_command,witness_joints,gaze)
                replay,_,_=physical(session,snapshot,witness_command,witness_joints,gaze)
                if witness!=replay:raise RuntimeError('Physical witness replay failed')
                witness.update(canonical_command=witness_command.tolist(),hand_joint_goal=witness_joints.tolist(),recipe=recipe)
                # the control of WITNESS_RULE: the same change reversed, from the same start, replayed like
                # every other branch
                session.restore(snapshot)
                mirror_request,mirror_research,_=witness_request(session,raw,initial_joints,goal_name,mirror=True)
                mirror_command,mirror_joints=filtered(model,mirror_request,mirror_research,decision)
                mirrored,_,_=physical(session,snapshot,mirror_command,mirror_joints,gaze)
                mirror_replay,_,_=physical(session,snapshot,mirror_command,mirror_joints,gaze)
                if mirrored!=mirror_replay:raise RuntimeError('Physical mirror replay failed')
                trial_decision=dict(decision,physical_goal=dict(pose=target,valid=torch.ones(1,dtype=torch.bool),
                    horizon_ms=250.,source='same_state_physical_witness'))
                arms={'controller':(base_command,base_joints,{})}
                plan_start=raw
                if args.initializer=='goal_pose':
                    # Analytical inverse-control initializer: request the known
                    # reachable destination, retaining ordinary force and gaze.
                    # The learned planners must improve this reference before
                    # it can support a learned-choice claim.
                    seed=raw.clone();seed[...,:7]=target[...,:7]
                    seed[:,model.GAZE_IDX]=raw[:,model.GAZE_IDX]
                    seed_array,seed_joints,_=kin.project_world_orders(seed.numpy())
                    plan_start=torch.from_numpy(seed_array).float()
                    seed_joints=torch.from_numpy(seed_joints).float()
                    seed_command,seed_joints=filtered(model,plan_start,seed_joints,trial_decision)
                    arms['goal_seed']=(seed_command,seed_joints,{})
                # Fix controls inside the scored graph; never overwrite a chosen
                # strength/gaze afterward and call it the optimizer's action.
                class MatchedControls(HandActionSpace):
                    def assemble(self):
                        command,strength,penalty=super().assemble()
                        command=command.clone();command[...,8:9]=raw[...,8:9]
                        command[:,model.GAZE_IDX]=raw[:,model.GAZE_IDX]
                        return command,raw[...,8:9],penalty
                for mode,synergy in (('direct',False),('synergy',True)):
                    session.restore(snapshot);model.synergy_planner=synergy
                    def held_gaze(cur,goal,motor,decision,execution_transform):
                        return motor,{'stopped_by':'gaze_held_for_matched_body_comparison'}
                    with patch('mimo_bridge.hand_target_control.HandActionSpace',MatchedControls),\
                         patch.object(model,'_plan_visual_gaze',held_gaze),torch.enable_grad():
                        plan,meta=model._plan_chain(current,current,plan_start,max_steps=1,decision=trial_decision)
                    if not plan:raise RuntimeError(f'{mode} planning rejected: {meta}')
                    command,joints=filtered(model,plan[0],meta['hand_joint_goal_raw'],trial_decision)
                    np.testing.assert_array_equal(command[:,:,8],base_command[:,:,8])
                    np.testing.assert_array_equal(command[:,model.GAZE_IDX],base_command[:,model.GAZE_IDX])
                    arms[mode]=(command,joints,meta['descent'])
                outcomes={}
                for mode,(command,joints,descent) in arms.items():
                    outcome,_,_=physical(session,snapshot,command,joints,gaze,target,target_joints)
                    repeated,_,_=physical(session,snapshot,command,joints,gaze,target,target_joints)
                    if outcome!=repeated:raise RuntimeError(mode+' full trajectory replay failed')
                    outcome.update(descent=descent,canonical_command=command.tolist(),hand_joint_goal=joints.tolist())
                    prediction=forecast_pose(model,current,trial_decision,command)
                    body_indices=[i for i,name in enumerate(session.handle_names) if name!='gaze']
                    predicted_error=float(pose_error_per_body(prediction[...,:3]*200,prediction[...,3:7],target,body_indices)[0])
                    if descent:
                        np.testing.assert_allclose(predicted_error,descent[0]['last_goal_distance'],atol=2e-5,rtol=2e-5)
                    outcome.update(predicted_pose=prediction.tolist(),predicted_goal_error=predicted_error)
                    if args.compare_execution_contracts:
                        old,_,_=physical(session,snapshot,command,joints,gaze,target,target_joints,tracking=False)
                        repeated_old,_,_=physical(session,snapshot,command,joints,gaze,target,target_joints,tracking=False)
                        if old!=repeated_old:raise RuntimeError('Old-contract replay failed')
                        outcome['fixed_world_execution']=old
                    outcomes[mode]=outcome
                if args.compare_execution_contracts:
                    original_command=LemLite.planned_execution(original_raw,decision['dev'],decision['achieved'],
                        previous=decision['smooth_prev'],smoothing=decision['smoothing'],keep_gaze_idx=decision.get('smooth_keep_gaze'))
                    old,_,_=physical(session,snapshot,original_command,None,gaze,target,target_joints,tracking=False)
                    repeated_old,_,_=physical(session,snapshot,original_command,None,gaze,target,target_joints,tracking=False)
                    if old!=repeated_old:raise RuntimeError('Original-controller replay failed')
                    old.update(predicted_pose=forecast_pose(model,current,trial_decision,original_command).tolist(),
                               canonical_command=original_command.tolist())
                    original_world=old
                reference='goal_seed' if args.initializer=='goal_pose' else 'controller'
                for mode in ('direct','synergy'):
                    search=outcomes[mode]['descent'][0]
                    outcomes[mode]['prediction_check']=dict(
                        predicted_start=search['per_body']['goal']['start'][0],
                        predicted_selected=search['per_body']['goal']['selected'][0],
                        predicted_change_during_search=(search['per_body']['goal']['selected'][0]-
                                                        search['per_body']['goal']['start'][0]),
                        actual_reference=reference,
                        actual_change_from_reference=(outcomes[mode]['physical_goal_error']-
                                                      outcomes[reference]['physical_goal_error']),
                        interpretation='The predicted change compares the optimizer own first and selected candidates; synergy first projects the supplied proposal through its learned movement map. Actual error compares the separately executed named reference.')
                pose_cm=lambda pose:np.concatenate([np.asarray(pose,dtype=np.float64)[0,:,:3]*200.,
                                                    np.asarray(pose,dtype=np.float64)[0,:,3:7]],-1)
                command_cm=lambda command:np.asarray(command,dtype=np.float64)[0,:,:7]
                check=witness_check(goal_name,session.handle_names,fingers,pose_cm(decision['achieved']),
                    raw=dict(controller=np.concatenate([raw.numpy()[0,:,:3]*200.,raw.numpy()[0,:,3:7]],-1),
                             witness=np.concatenate([request.numpy()[0,:,:3]*200.,request.numpy()[0,:,3:7]],-1)),
                    executed=dict(controller=command_cm(base_command),witness=command_cm(witness_command),
                                  mirror=command_cm(mirror_command)),
                    ends=dict(controller=pose_cm(outcomes['controller']['pose']),witness=pose_cm(witness['pose']),
                              mirror=pose_cm(mirrored['pose'])),
                    start_joints=initial_joint_angles,
                    raw_joints=dict(controller=initial_joints.numpy()[0],witness=research.numpy()[0]),
                    executed_joints=dict(controller=base_joints.numpy()[0],witness=witness_joints.numpy()[0],
                                         mirror=mirror_joints.numpy()[0]),
                    end_joints=dict(controller=np.asarray(outcomes['controller']['joint_angles_rad'])[0],
                                    witness=np.asarray(witness['joint_angles_rad'])[0],
                                    mirror=np.asarray(mirrored['joint_angles_rad'])[0]),
                    elbow=(dict(axis=elbow_axis,start=initial_elbow,
                                controller=outcomes['controller']['qpos'][elbow_address],
                                witness=witness['qpos'][elbow_address],mirror=mirrored['qpos'][elbow_address])
                           if goal_name=='elbow_flex_witness' else None))
                mirror_record=dict(trajectory_sha256=mirrored['trajectory_sha256'],replayed_exactly=True,pose=mirrored['pose'],
                                   joint_angles_rad=mirrored['joint_angles_rad'],
                                   supported_every_step=mirrored['support']['every_step'],
                                   canonical_command=mirror_command.tolist(),hand_joint_goal=mirror_joints.tolist())
                row=dict(body=body,goal=goal_name,witness=witness,mirror_witness=mirror_record,outcomes=outcomes,
                    exact_replay=True,witness_check=check,
                    initial_pose=decision['achieved'].tolist(),
                    initial_hand_joint_angles_rad=initial_joint_angles.tolist(),
                    initial_hand_joint_velocity_rad_s=initial_joint_velocity.tolist(),
                    controller_raw_hand_joint_goal=initial_joints.tolist(),
                    previous_hand_joint_goal=(None if decision.get('hand_previous_joints') is None
                                              else decision['hand_previous_joints'].tolist()),
                    smoothing=decision['smoothing'],gaze_rad=gaze.tolist(),
                    elapsed_s=time.perf_counter()-trial_started,
                    inverse_projection=dict(calls=projection_timing['calls']-projection_before['calls'],
                        elapsed_s=projection_timing['elapsed_s']-projection_before['elapsed_s']))
                if args.compare_execution_contracts:row['original_world_controller']=original_world
                rows.append(row)
                with (args.out/'trials.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                (args.out/'status.json').write_text(json.dumps(dict(state='running',completed=len(rows),
                    total=args.bodies*len(goals),started_at=started,elapsed_s=time.time()-started)))
                print('Completed',body,goal_name,{k:round(v['physical_goal_error'],4) for k,v in outcomes.items()},flush=True)
        finally:session.close()
    if checksum(model)!=initial_weights:raise RuntimeError('Frozen learned weights changed')
    summary={}
    for metric in ('physical_goal_error','hand_joint_error_deg','wrist_goal_error_cm'):
        # Average repeated goals within each body before treating bodies as samples.
        modes=('controller','goal_seed','direct','synergy') if args.initializer=='goal_pose' else ('controller','direct','synergy')
        values={mode:np.array([np.mean([r['outcomes'][mode][metric] for r in rows if r['body']==b])
            for b in range(args.bodies)]) for mode in modes}
        delta=values['synergy']-values['direct'];draws=np.random.default_rng(641).integers(args.bodies,size=(10000,args.bodies))
        summary[metric]=dict(means={k:float(v.mean()) for k,v in values.items()},
            synergy_minus_direct=float(delta.mean()),paired_bootstrap95=np.quantile(delta[draws].mean(-1),[.025,.975]).tolist(),
            bodies_synergy_better=int((delta<0).sum()))
        summary[metric]['against_controller']={}
        for mode in modes[1:]:
            difference=values[mode]-values['controller']
            summary[metric]['against_controller'][mode]=dict(mean_difference=float(difference.mean()),
                paired_bootstrap95=np.quantile(difference[draws].mean(-1),[.025,.975]).tolist(),
                bodies_better=int((difference<0).sum()))
        if args.initializer=='goal_pose':
            summary[metric]['against_goal_seed']={}
            for mode in ('direct','synergy'):
                difference=values[mode]-values['goal_seed']
                summary[metric]['against_goal_seed'][mode]=dict(mean_difference=float(difference.mean()),
                    paired_bootstrap95=np.quantile(difference[draws].mean(-1),[.025,.975]).tolist(),
                    bodies_better=int((difference<0).sum()))
    by_goal={}
    for name in goals:
        by_goal[name]={}
        selected=[row for row in rows if row['goal']==name]
        for group in ('fingers','wrists','other_body'):
            values={mode:np.asarray([row['outcomes'][mode]['goal_error_by_part'][group] for row in selected]) for mode in modes}
            by_goal[name][group]=dict(means={mode:float(v.mean()) for mode,v in values.items()},against_controller={})
            for mode in modes[1:]:
                difference=values[mode]-values['controller']
                by_goal[name][group]['against_controller'][mode]=dict(mean_difference=float(difference.mean()),
                    paired_bootstrap95=np.quantile(difference[draws].mean(-1),[.025,.975]).tolist(),bodies_better=int((difference<0).sum()))
        by_goal[name]['witness_supported_bodies']=sum(row['witness']['support']['every_step'] for row in selected)
        by_goal[name]['witness_separation_from_controller']={group:[row['outcomes']['controller']['goal_error_by_part'][group] for row in selected]
            for group in ('fingers','wrists','other_body')}
    # the outside review's statistic over the arms of every trial, per goal and pooled, and whether
    # each goal's witness request moved the requested part beyond the controller's endpoint
    reference='goal_seed' if args.initializer=='goal_pose' else 'controller'
    ranking,witnesses=ranking_and_witnesses(rows,goals,modes,reference,args.bodies)
    for name in goals:by_goal[name]['witness_check']=witnesses[name]
    result=dict(declaration,summary=summary,by_goal=by_goal,ranking=ranking,weights_unchanged=True,exact_replay=True,
                inverse_projection=projection_timing,elapsed_s=time.time()-started)
    timing_patch.stop()
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    (args.out/'status.json').write_text(json.dumps(dict(state='completed',completed=len(rows),total=len(rows),
        started_at=started,elapsed_s=time.time()-started)))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
