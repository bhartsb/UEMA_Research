# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Source-bound hand response calibration and bounded hand-only goal choice.

All learned UEMA parameters retain their existing meaning. This admitted small
response fit is frozen, checkpointed calibration; its choices alter only feasible
finger goals, retaining the caller's filtered wrist/body/force/gaze commands.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from ..trainer_code import trainer_code_hash  # noqa: F401  (admission_check's code hash; see THE ADMISSION below)

VERSION='hand_joint_response_v1'
FEATURES=['intercept','goal_minus_initial_over_range','quarter_second_velocity_over_range']
# The finger command meaning a calibration's examples were executed under (mimo_bridge/action_decoder.py, THE FINGER
# COMMAND'S MEANING). A receipt without 'command_meaning' was fit on commands whose finger rows the decoder executed by
# servoing each fingertip's pose (--finger-joint-servo off): the admitted hand_response_calibration_380200.json, 28
# commands from checkpoint 380,200. Its coefficients map a hinge's goal, angle and speed to that command's 250 ms
# change, not to the change under the hinge-by-hinge servo; a refit on commands executed with --finger-joint-servo on
# records command_meaning='finger_joint_servo_v1'. With --hand-goals on, LemLite.configure_hand_geometry refuses any
# other pairing in both directions: the setting on with a pose-servo receipt, and the setting off with a joint-servo one.
POSE_SERVO_MEANING='finger_pose_servo'


def command_meaning(receipt):
    return str(receipt.get('command_meaning',POSE_SERVO_MEANING))


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_receipt(receipt,geometry):
    payload={key:value for key,value in receipt.items() if key!='sha256'}
    if receipt.get('version')!=VERSION or receipt.get('sha256')!=content_hash(payload):
        raise ValueError('Hand response calibration version or source hash mismatch')
    if receipt.get('geometry_sha256')!=content_hash(geometry):
        raise ValueError('Hand response calibration belongs to different native geometry')
    if receipt.get('features')!=FEATURES or receipt.get('horizon_ms')!=250:
        raise ValueError('Hand response feature or horizon contract mismatch')
    if receipt.get('choice')!=dict(steps=20,lr=.1,raw_joint_bound_deg=11.,nonfinger_orders='fixed'):
        raise ValueError('Hand response choice settings differ from the physical qualification')
    if receipt.get('train_bodies')!=[0,1,2,3] or receipt.get('test_bodies')!=[4,5,6,7]:
        raise ValueError('Hand response calibration requires the declared disjoint four/four split')
    if receipt.get('joint_names')!=[joint['name'] for joint in geometry['joints']]:
        raise ValueError('Hand response joint ordering changed')
    coefficients=torch.as_tensor(receipt.get('coefficients'),dtype=torch.float64)
    if coefficients.shape!=(len(geometry['joints']),3) or not torch.isfinite(coefficients).all():
        raise ValueError('Invalid hand response coefficients')
    if not receipt.get('source_sha256') or any(len(value)!=64 for value in receipt['source_sha256'].values()):
        raise ValueError('Hand response calibration lacks source receipts')
    return copy.deepcopy(receipt)


class CalibratedHandResponse(nn.Module):
    def __init__(self,geometry,receipt):
        super().__init__()
        self.geometry=copy.deepcopy(geometry)
        self.receipt=validate_receipt(receipt,geometry)
        self.register_buffer('coefficients',torch.tensor(receipt['coefficients'],dtype=torch.float64))
        self.register_buffer('ranges',torch.tensor([j['limits'][1]-j['limits'][0] for j in geometry['joints']],dtype=torch.float64))
        encoded=json.dumps(self.receipt,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.register_buffer('receipt_utf8',torch.tensor(list(encoded),dtype=torch.uint8))
        self.register_load_state_dict_post_hook(self._verify_restored_buffers)

    @property
    def kinematics(self):
        # Snapshot state contains plain immutable geometry only. A search keeps
        # one local helper; no native or unsupported Python helper is checkpointed.
        from mimo_bridge.hand_target_control import HandKinematics
        return HandKinematics(self.geometry)

    @classmethod
    def from_file(cls,geometry,path):
        return cls(geometry,json.loads(Path(path).read_text(encoding='utf-8')))

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        encoded=state_dict.get(prefix+'receipt_utf8')
        if encoded is not None:
            try:
                if encoded.dtype!=torch.uint8 or encoded.ndim!=1:raise ValueError('Receipt must contain UTF-8 bytes')
                receipt=json.loads(bytes(encoded.detach().cpu().tolist()).decode())
                self.receipt=validate_receipt(receipt,self.geometry)
                self.receipt_utf8=torch.empty_like(encoded,device=self.receipt_utf8.device)
            except (ValueError,TypeError,UnicodeError) as exc:
                error_msgs.append(prefix+'invalid hand response receipt: '+str(exc))
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _verify_restored_buffers(self,module,incompatible_keys):
        expected=torch.tensor(self.receipt['coefficients'],device=self.coefficients.device,dtype=self.coefficients.dtype)
        ranges=self.ranges.new_tensor([j['limits'][1]-j['limits'][0] for j in self.geometry['joints']])
        if not torch.equal(expected,self.coefficients) or not torch.equal(ranges,self.ranges):
            raise ValueError('Restored hand response buffers differ from the admitted calibration')

    def predict(self,initial,velocity,goal,*,_kinematics=None):
        if initial.shape!=goal.shape or velocity.shape!=goal.shape or goal.shape[-1]!=len(self.ranges):
            raise ValueError('Hand response needs matching current angles, velocity and goal')
        # Match the qualified fit's arithmetic while retaining genuine gradients
        # into the float32 planner variables through the cast.
        goal=goal.double();initial=initial.to(goal);velocity=velocity.to(goal)
        ranges=self.ranges.to(goal)
        features=torch.stack((torch.ones_like(goal),(goal-initial)/ranges,.25*velocity/ranges),dim=-1)
        change=(features*self.coefficients.to(goal)).sum(-1)
        kin=self.kinematics if _kinematics is None else _kinematics
        return kin.project_joints(initial+change*ranges)

    def local_goal_cost(self,initial,velocity,goal,target_pose,*,_kinematics=None):
        from .physical_goal import pose_error_per_body
        kin=self.kinematics if _kinematics is None else _kinematics
        target=kin.local_encoding(target_pose.to(device=goal.device,dtype=torch.float64))
        template=torch.zeros_like(target);template[...,6]=1.
        predicted=kin.assemble_world(template,self.predict(initial,velocity,goal,_kinematics=kin))
        fingers=[i for hand in kin.geometry['hands'] for i in hand['finger_indices']]
        return pose_error_per_body(predicted[...,:3]*200.,predicted[...,3:7],target,fingers)

    @torch.enable_grad()
    def choose_hand_goal(self,canonical_execution,raw_initial_goal,previous_goal,smoothing,
                         current_angles,current_velocity,target_pose,valid=None):
        """Choose only fingers, returning both raw and already-filtered goals.

        canonical_execution positions are cm; target_pose positions are cm/200.
        Angles are radians and velocity is rad/s, ordered by geometry['joints'].
        The returned joint_goal has already been smoothed: do not filter it again.
        """
        from mimo_bridge.hand_target_control import HandActionSpace
        from .physical_goal import PhysicalTrustRegion
        kin=self.kinematics
        command=canonical_execution.detach().clone()
        raw=raw_initial_goal.detach().to(device=command.device,dtype=torch.float32)
        previous=None if previous_goal is None else previous_goal.detach().to(raw)
        initial=current_angles.detach().to(device=command.device,dtype=torch.float64)
        velocity=current_velocity.detach().to(initial);target=target_pose.detach().to(initial)
        if not 0<=float(smoothing)<=1:raise ValueError('Invalid hand smoothing')
        for value in (command,raw,initial,velocity,target):
            if not torch.isfinite(value).all():raise ValueError('Nonfinite hand-goal input')
        if previous is not None and (previous.shape!=raw.shape or not torch.isfinite(previous).all()):
            raise ValueError('Invalid previous hand goal')
        valid=(torch.ones(command.shape[0],dtype=torch.bool,device=command.device) if valid is None
               else valid.detach().bool().to(command.device))
        if valid.shape!=(command.shape[0],):raise ValueError('One validity flag per body required')
        template=command.clone();template[...,:3]/=200.
        space=HandActionSpace(kin.geometry,template,raw)
        bounded=PhysicalTrustRegion(space,template,kin.geometry['handle_names']);bounded.init()
        if bounded.rotation_deg!=self.receipt['choice']['raw_joint_bound_deg']:
            raise ValueError('Hand goal trust bound differs from its calibration')
        def executed_goal():
            goal=kin.project_joints(bounded.goal_joints())
            if previous is not None and smoothing>0:goal=float(smoothing)*goal+(1.-float(smoothing))*previous
            return kin.project_joints(goal)
        def score():
            goal=executed_goal()
            costs=self.local_goal_cost(initial,velocity,goal,target,_kinematics=kin)
            return torch.where(valid,costs,torch.zeros_like(costs)),goal
        optimizer=torch.optim.Adam([space.joints],lr=.1)
        costs,goal=score();first=costs.detach().clone()
        baseline=kin.assemble_world(command,goal.to(command),position_scale=1.)
        fingers=[i for hand in kin.geometry['hands'] for i in hand['finger_indices']]
        positions_match=torch.isclose(baseline[:,fingers,:3],command[:,fingers,:3],atol=2e-5,rtol=1e-6).all()
        qa=baseline[:,fingers,3:7];qb=command[:,fingers,3:7]
        qa=qa/qa.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        qb=qb/qb.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        rotations_match=torch.isclose(abs((qa*qb).sum(-1)),torch.ones_like(qa[...,0]),atol=1e-6,rtol=0).all()
        if not bool(positions_match&rotations_match):
            raise ValueError('Hand choice baseline does not match its raw goal and smoothing history')
        best=first.clone();best_goal=goal.detach().clone();best_raw=bounded.goal_joints().detach().clone()
        selected=torch.zeros(command.shape[0],dtype=torch.int64,device=command.device)
        trace=[first.cpu().tolist()]
        for iteration in range(1,21):
            optimizer.zero_grad();costs,_=score();costs.sum().backward();optimizer.step()
            with torch.no_grad():
                costs,goal=score()
                if not torch.isfinite(costs).all():raise RuntimeError('Nonfinite hand-response search')
                improved=(costs<best)&valid
                best=torch.where(improved,costs,best)
                best_goal=torch.where(improved[:,None],goal,best_goal)
                best_raw=torch.where(improved[:,None],bounded.goal_joints(),best_raw)
                selected=torch.where(improved,torch.full_like(selected,iteration),selected)
                trace.append(costs.cpu().tolist())
        chosen=kin.assemble_world(command,best_goal.to(command),position_scale=1.)
        chosen=torch.where(valid[:,None,None],chosen,command).detach()
        return dict(command=chosen,joint_goal=best_goal.detach(),raw_joint_goal=best_raw.detach(),
            metadata=dict(version=VERSION,calibration_sha256=self.receipt['sha256'],
                goal='wrist_relative_finger_shape',horizon_ms=250,steps=20,
                raw_joint_bound_deg=bounded.rotation_deg,first=first.cpu().tolist(),selected=best.cpu().tolist(),
                selected_iteration=selected.cpu().tolist(),valid=valid.cpu().tolist(),trace=trace,
                nonfinger_orders='retained exactly from the filtered controller'))


# THE ADMISSION OF A REPLACEMENT CALIBRATION (docs/gain_of_function_v4/finger_command_interface/
# HAND_CALIBRATION_ADMISSION_DESIGN.md). Every H84 checkpoint saved the admitted calibration in model_receipt.json
# (hand_response), and a resume refuses a requested calibration that differs from it (LemLite.
# _restore_hand_response_calibration). Only with --replace-saved-hand-calibration RECORD (train_mimo) may a resume
# replace it, and only when RECORD, the result of the closed-loop hand-choice comparison
# (docs/gain_of_function_v4/finger_command_interface/hand_calibration_admission.py compare), passes admission_check:
# it names the requested calibration's payload and file bytes, the model weights of the checkpoint the launch resumes
# (the calibration's own source_model_sha256 names the same weights), the trainer code that runs the launch, the
# launch's finger rule and command meaning, the calibration's hand geometry, and per-decision differences that pass
# the declared line (admission_line) when recomputed here. The comparison script's own bytes are checked by that
# tool's verify command, not here (the trainer does not read docs/).
ADMISSION_VERSION='hand_calibration_admission_v1'
ADMISSION_ROUTE=('LemLite.step observe -> LemLite._plan_chain hand memory stage -> motor_for_action and '
                 'hand_joint_goal -> actual physics')
ADMISSION_BODIES=[4,5,6,7]           # held out of every fit: hand_response.validate_receipt's test_bodies
ADMISSION_STARTS_PER_BODY=4          # decisions per held-out body (the design note's arithmetic, section 6)
ADMISSION_ALPHA=.05                  # the exact sign-flip line: floor(0.05 x 2^n) of 2^n patterns
# trainer_code_hash(source_python=None) (imported above) is lem/trainer_code.py's: the one definition of the trainer
# code's hash, which the training preflight's configuration also computes (its code_hash), so that an admission record
# made in one checkout of a commit is accepted by a launch from another checkout of the same commit


def signflip_count(values):
    """Exact two-sided sign-flip count over the values: (count, patterns). A pattern of signs counts when the mean of
    (sign x value) lies at least as far from 0 as the measured mean (within 1e-12); the measured pattern and its
    mirror always count. Exact enumeration in blocks of 2^16 patterns, up to 26 values."""
    import itertools
    import numpy as np
    v=np.asarray(values,dtype=float)
    n=len(v)
    if n==0:return None,0
    if n>26:raise ValueError('exact enumeration is limited to 26 values')
    m=abs(float(v.mean()))
    lo=min(n,16)
    base=np.array(list(itertools.product((1.,-1.),repeat=lo)))@v[:lo]
    count=0
    for hi in itertools.product((1.,-1.),repeat=n-lo):
        total=base+(float(np.dot(hi,v[lo:])) if n>lo else 0.)
        count+=int(np.sum(np.abs(total)/n>=m-1e-12))
    return count,2**n


def admission_line(decisions,bodies=ADMISSION_BODIES,starts_per_body=ADMISSION_STARTS_PER_BODY):
    """The declared pass line of the closed-loop hand-choice comparison. decisions: [{'body': b, 'difference': d}]
    with d = hand shape error at the end of the 250 ms hold with the hand stage (the requested calibration) minus the
    same decision without the hand stage (lower = nearer the remembered hand pose; 0 = no difference).
    P1: the mean of d is below 0 and the exact two-sided sign-flip count of the n values is at most
    floor(0.05 x 2^n). P2: every held-out body's mean of d is below 0. complete: exactly starts_per_body finite values
    for each body in bodies. passes = complete and P1 and P2."""
    import math
    values=[d.get('difference') for d in decisions]
    finite=all(type(v) in (int,float) and math.isfinite(v) for v in values)
    counts={}
    for d in decisions:counts[d.get('body')]=counts.get(d.get('body'),0)+1
    complete=bool(finite and counts=={b:int(starts_per_body) for b in bodies})
    out=dict(decisions=len(values),complete=complete,bodies=list(bodies),starts_per_body=int(starts_per_body))
    if not values or not finite:
        return dict(out,mean=None,signflip_count=None,signflip_patterns=None,signflip_line=None,body_means={},
                    P1=False,P2=False,passes=False)
    count,patterns=signflip_count(values)
    line=int(math.floor(ADMISSION_ALPHA*patterns))
    mean=float(sum(values)/len(values))
    body_means={str(b):float(sum(d['difference'] for d in decisions if d.get('body')==b)/counts[b])
                for b in bodies if counts.get(b)}
    p1=bool(complete and mean<0 and count<=line)
    p2=bool(complete and len(body_means)==len(bodies) and all(v<0 for v in body_means.values()))
    return dict(out,mean=mean,signflip_count=count,signflip_patterns=patterns,signflip_line=line,body_means=body_means,
                P1=p1,P2=p2,passes=bool(p1 and p2))


def admission_check(record,receipt,*,calibration_file_sha256,model_sha256,finger_joint_servo,meaning,code_hash):
    """Every binding of a closed-loop admission record the trainer can verify. record: the comparison's result
    (hand_calibration_admission.py compare). receipt: the requested calibration's payload (without the
    calibration_file_sha256 that LemLite.configure_hand_response_calibration adds). calibration_file_sha256: the SHA-256
    of the requested calibration file's bytes. model_sha256: the SHA-256 of the resumed checkpoint's model.safetensors.
    finger_joint_servo and meaning: the launch's rule ('off' or 'on') and the command meaning it executes. code_hash:
    trainer_code_hash() of the code that runs the launch. Raises ValueError naming the first binding that fails;
    returns the record's summary."""
    def need(condition,text):
        if not condition:raise ValueError('--replace-saved-hand-calibration: the admission record '+text)
    need(isinstance(record,dict),'is not a JSON object')
    need(record.get('version')==ADMISSION_VERSION,'is version %r, not %r'%(record.get('version'),ADMISSION_VERSION))
    need(record.get('sha256')==content_hash({k:v for k,v in record.items() if k!='sha256'}),
         'does not match its own content hash (it was changed after it was written)')
    need(record.get('receipt_payload_sha256')==receipt.get('sha256'),
         'names the calibration payload %s, not the requested %s'%(record.get('receipt_payload_sha256'),receipt.get('sha256')))
    need(record.get('receipt_file_sha256')==calibration_file_sha256,
         'names calibration file bytes %s, not the requested file\'s %s'%(record.get('receipt_file_sha256'),
                                                                          calibration_file_sha256))
    need(receipt.get('source_model_sha256')==model_sha256,
         'is for a calibration bound to model weights %s (its source_model_sha256); this checkpoint\'s model.safetensors '
         'is %s'%(receipt.get('source_model_sha256'),model_sha256))
    need(record.get('model_sha256')==model_sha256,
         'was measured on model weights %s; this checkpoint\'s model.safetensors is %s'%(record.get('model_sha256'),
                                                                                       model_sha256))
    need(record.get('code_hash')==code_hash,
         'was measured on trainer code %s; this launch runs %s: run the comparison again on this code'%(
             record.get('code_hash'),code_hash))
    need(record.get('finger_joint_servo')==finger_joint_servo,
         'was measured with --finger-joint-servo %s; this launch runs %s'%(record.get('finger_joint_servo'),
                                                                          finger_joint_servo))
    need(command_meaning(receipt)==meaning and record.get('command_meaning')==meaning,
         'and the calibration declare the command meanings %r and %r; this launch executes %r'%(
             record.get('command_meaning'),command_meaning(receipt),meaning))
    need(record.get('geometry_sha256')==receipt.get('geometry_sha256'),
         'was measured on hand geometry %s; the calibration was fit on %s'%(record.get('geometry_sha256'),
                                                                            receipt.get('geometry_sha256')))
    need(record.get('tested_route')==ADMISSION_ROUTE,'tested another route: %r'%(record.get('tested_route'),))
    need(record.get('held_out_bodies')==ADMISSION_BODIES,'judged bodies %r, not %r'%(record.get('held_out_bodies'),
                                                                                       ADMISSION_BODIES))
    for key in ('weights_unchanged','exact_replay','nonfinger_commands_unchanged','sensory_goals_identical',
                'hand_stage_completed'):
        need(record.get(key) is True,'does not state %s true'%key)
    need(record.get('learning') is False,'does not state learning false')
    need(record.get('validity_problems')==[],'lists validity problems: %r'%(record.get('validity_problems'),))
    line=admission_line(record.get('decisions') or [])
    need(line['passes'],'holds decisions that do not pass the declared line when recomputed (complete %s, P1 %s, '
         'P2 %s, mean %s, sign-flip count %s of %s, line %s)'%(line['complete'],line['P1'],line['P2'],line['mean'],
                                                              line['signflip_count'],line['signflip_patterns'],
                                                              line['signflip_line']))
    recorded=record.get('line') or {}
    need(all(recorded.get(k)==line[k] for k in ('decisions','mean','signflip_count','signflip_line','body_means',
                                               'P1','P2','passes')),
         'states a line that differs from its decisions recomputed')
    need(record.get('verdict')=='pass' and record.get('passed') is True,'does not read verdict pass')
    return dict(version=record['version'],sha256=record['sha256'],verdict=record['verdict'],
                checkpoint=record.get('checkpoint'),model_sha256=model_sha256,code_hash=code_hash,
                finger_joint_servo=finger_joint_servo,geometry_sha256=record.get('geometry_sha256'),
                written=record.get('written'),decisions=line['decisions'],mean_difference=line['mean'],
                signflip_count=line['signflip_count'],signflip_patterns=line['signflip_patterns'],
                signflip_line=line['signflip_line'],body_means=line['body_means'])
