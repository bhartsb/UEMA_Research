# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Bounded physical search for a whole-body roll through the live motor interface.

This is an auditor's reachability calibration. Search scores never enter a learner,
and finding a command sequence does not mean the model learned to roll. Failure of
a bounded search is not proof that rolling is impossible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from mimo_bridge.body_layout import chest_orientation
from mimo_bridge.env import BridgeEnv
from mimo_bridge.lem_model import LemLite
from mimo_bridge.train_mimo import parse_args, env_kwargs_for
from mimo_bridge.tools.causal_test import _limits_for_run
from mimo_bridge.tools.training_continuation import atomic_json


def multiply_xyzw(a, b):
    av, aw, bv, bw = a[:3], a[3], b[:3], b[3]
    return np.r_[aw*bv+bw*av+np.cross(av, bv), aw*bw-np.dot(av, bv)]


def pose(e, strength):
    ids=e.proprio.body_ids
    values=np.zeros((1,len(ids),9),np.float32)
    values[0,:,:3]=e.data.xpos[ids]*100
    values[0,:,3:7]=e.data.xquat[ids][:,[1,2,3,0]]
    values[0,:,8]=strength
    return values


def groups(names):
    return [[i for i,name in enumerate(names) if name.endswith('_L') and not name.startswith(('thigh','calf','foot'))],
            [i for i,name in enumerate(names) if name.endswith('_R') and not name.startswith(('thigh','calf','foot'))],
            [i for i,name in enumerate(names) if name in ('thigh_L','calf_L','foot_L')],
            [i for i,name in enumerate(names) if name in ('thigh_R','calf_R','foot_R')],
            [i for i,name in enumerate(names) if name in ('lower_trunk','chest')],
            [i for i,name in enumerate(names) if name=='head']]


def proposal(reference, parameters, handle_groups, displacement_m):
    """Task-agnostic grouped position/rotation proposals in the native handle units."""
    result=reference.copy()
    for row,indices in zip(parameters,handle_groups):
        vector=np.asarray(row[:3],float)
        angle=float(np.linalg.norm(vector))
        if angle>math.pi:vector*=math.pi/angle;angle=math.pi
        dq=np.r_[vector/max(angle,1e-12)*math.sin(angle/2),math.cos(angle/2)]
        for i in indices:
            result[0,i,3:7]=multiply_xyzw(dq,reference[0,i,3:7])
            result[0,i,:3]+=np.clip(row[3:6],-1,1)*displacement_m*100
    result[:,:,:3]/=200.
    return torch.from_numpy(result)


def posture(e):
    chest=e.model.body('chest').id
    orientation=chest_orientation(e.data.xmat[chest].reshape(3,3),e._body_orientation_axes)
    floor=e.model.geom('floor').id
    support=False;max_force=0.
    avatar=set(int(i) for i in e.proprio.body_ids)
    root=e.model.body('mimo_location').id
    # Every actual descendant counts, not only the tracked handles.
    for body in range(e.model.nbody):
        ancestor=body
        while ancestor>0 and ancestor!=root:ancestor=int(e.model.body_parentid[ancestor])
        if ancestor==root:avatar.add(body)
    for i in range(e.data.ncon):
        c=e.data.contact[i]
        force=np.zeros(6);e._mj.mj_contactForce(e.model,e.data,i,force)
        if int(e.model.geom_bodyid[c.geom1]) in avatar or int(e.model.geom_bodyid[c.geom2]) in avatar:
            max_force=max(max_force,float(force[0]))
        if int(c.geom1)==floor and int(e.model.geom_bodyid[c.geom2]) in avatar or int(c.geom2)==floor and int(e.model.geom_bodyid[c.geom1]) in avatar:
            support=support or force[0]>0
    return dict(time_s=float(e.data.time),front_up=orientation['front_up_component'],
                roll_deg=orientation['roll_degrees'],roll_valid=orientation['valid'],
                supported=bool(support),maximum_contact_normal_force_n=max_force,
                chest_xyz_m=e.data.xpos[chest].tolist(),
                torso_quaternion_wxyz=e.data.xquat[chest].tolist())


def rollout(e,snapshot,a,dev,parameters,phase_seconds,displacement_m):
    e.restore_full_state(snapshot)
    reference=pose(e,float(dev['strength_limit']))
    handle_groups=groups(e.decoder.handle_names)
    previous=torch.from_numpy(reference.copy());held=reference[0].copy()
    dt=e.dt*max(1,a.action_repeat)
    steps=max(1,int(round(phase_seconds/dt)))*len(parameters)
    rows=[posture(e)];commands=[];max_ctrl=0.;saturations=0;controls=0
    initial_roll=rows[0]['roll_deg']
    streak=longest=0.
    for index in range(steps):
        phase=min(len(parameters)-1,int(index*dt/phase_seconds))
        raw=proposal(reference,parameters[phase],handle_groups,displacement_m)
        achieved=torch.from_numpy(pose(e,float(dev['strength_limit'])))
        achieved[...,:3]/=200.
        # The live learner updates the smoother each model frame; the trainer
        # holds only selected proposals for motor_hold model frames.
        previous=LemLite.planned_execution(raw,dev,achieved,previous=previous,smoothing=a.smooth_actions)
        if index%max(1,a.motor_hold)==0:held=previous.numpy()[0].copy()
        for sub in range(max(1,a.action_repeat)):
            e.step(held,gaze=np.zeros(2),handle_mask=dev.get('handle_mask'),render=False)
        row=posture(e);rows.append(row);commands.append(held.tolist())
        if not np.isfinite(e.data.qpos).all() or not np.isfinite(e.data.qvel).all():raise ValueError('nonfinite body state')
        ctrl=np.asarray(e._handed_ctrl);max_ctrl=max(max_ctrl,float(np.abs(ctrl).max()))
        saturations+=int(np.count_nonzero((ctrl<=e.decoder.ctrl_low+1e-8)|(ctrl>=e.decoder.ctrl_high-1e-8)));controls+=len(ctrl)
        roll_change=abs((row['roll_deg']-initial_roll+180)%360-180) if row['roll_valid'] and initial_roll is not None else 0.
        good=row['supported'] and row['front_up']<=-math.cos(math.radians(30)) and roll_change>=150
        before=rows[-2]
        previous_good=before.get('goal',False)
        row['goal']=bool(good)
        streak=streak+row['time_s']-before['time_s'] if good and previous_good else 0.
        longest=max(longest,streak)
    best=min(float(row['front_up']) for row in rows)
    summary=dict(minimum_front_up=best,maximum_roll_from_start_deg=max(
        abs((row['roll_deg']-initial_roll+180)%360-180) for row in rows if row['roll_valid']),
        supported_prone_dwell_s=longest,attained=longest>=.5,maximum_control_abs=max_ctrl,
        control_limit_fraction=saturations/max(1,controls),
        maximum_contact_normal_force_n=max(row['maximum_contact_normal_force_n'] for row in rows),
        final_qpos=e.data.qpos.tolist(),final_qvel=e.data.qvel.tolist())
    return dict(summary=summary,rows=rows,commands=commands,
                score=(2. if summary['attained'] else 0.)+longest-best)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=63119)
    p.add_argument('--population',type=int,default=12)
    p.add_argument('--generations',type=int,default=3)
    p.add_argument('--phase-seconds',type=float,default=1.5)
    p.add_argument('--phases',type=int,default=3)
    p.add_argument('--budget-seconds',type=float,default=300.)
    p.add_argument('--position-range-m',type=float,default=.25)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--initial-sequence',type=Path,help='Replay a previously measured candidate first, without installing it in a learner')
    args=p.parse_args()
    if args.population<2 or args.generations<1 or args.phase_seconds<=0 or args.phases<1 or args.budget_seconds<=0 or args.position_range_m<=0:p.error('invalid bounded search dimensions')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    saved=json.loads((args.checkpoint.parent.parent/'args.json').read_text(encoding='utf-8'))
    a=parse_args(['--out',str(args.out)]);vars(a).update(saved)
    a.device='cpu';a.n_envs=1;a.out=str(args.out)
    dev,source=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
    if not dev.get('phase_name') or 'strength_limit' not in dev:raise ValueError('missing recorded developmental limits')
    kw=env_kwargs_for(a,dev['phase_name'],str(args.out/'world'))
    kw.update(seed=args.seed,caregiver_mode='off',caregiver_seconds=0,caregiver_record_dir='',
              mobile=False,critters=False,gaze_reflex=False,spontaneous_saccades=False,vision=False)
    declaration=dict(kind='physical_reachability_only',learner_present=False,checkpoint=str(args.checkpoint),
        args_sha256=hashlib.sha256((args.checkpoint.parent.parent/'args.json').read_bytes()).hexdigest(),
        limits=dev,limits_source=source,body_age_months=kw.get('age'),body_layout=kw.get('body_layout'),
        body_function_mode=kw.get('body_function'),action_repeat=a.action_repeat,motor_hold=a.motor_hold,
        smoothing=a.smooth_actions,search=vars(args)|{'checkpoint':str(args.checkpoint),'out':str(args.out),'initial_sequence':None if args.initial_sequence is None else str(args.initial_sequence)},
        goal='From supine, change gravity-relative anatomical roll at least 150 degrees and maintain supported front-down within 30 degrees for 0.5 seconds.',
        negative='A bounded search failure is inconclusive; it does not prove rolling is mechanically impossible.',
        disclosure='No learner or task reward is used. The auditor searches commands only to measure physical possibility; no found controller is installed or used as training.')
    atomic_json(args.out/'declaration.json',declaration)
    e=None
    try:
        e=BridgeEnv(**kw)
        e.reset()
        # The copied body-only fixture parks all non-body objects before the
        # scored snapshot; no invisible collider or caregiver moves the body.
        if e._target_id is not None:e.data.mocap_pos[e._target_id]=[0,0,3];e._target_id=None
        for body_name in ('box_light','box_medium','box_heavy'):
            body=e.model.body(body_name).id;joint=int(e.model.body_jntadr[body]);address=int(e.model.jnt_qposadr[joint])
            e.data.qpos[address:address+3]=[3+body*.05,0,1]
        e._mj.mj_forward(e.model,e.data)
        holding=pose(e,float(dev['strength_limit']))[0]
        for _ in range(30):e.step(holding,gaze=np.zeros(2),render=False)
        snapshot=e.save_full_state()
        if e._body_orientation_axes is None:raise ValueError('anatomical roll axes were not calibrated')
        phase_seconds=.1 if args.smoke else args.phase_seconds
        phases=1 if args.smoke else args.phases
        neutral=np.zeros((phases,6,6))
        baseline=rollout(e,snapshot,a,dev,neutral,phase_seconds,args.position_range_m)
        replay=rollout(e,snapshot,a,dev,neutral,phase_seconds,args.position_range_m)
        np.testing.assert_array_equal(baseline['summary']['final_qpos'],replay['summary']['final_qpos'])
        np.testing.assert_array_equal(baseline['summary']['final_qvel'],replay['summary']['final_qvel'])
        atomic_json(args.out/'hold_control.json',baseline)
        if args.smoke:
            atomic_json(args.out/'report.json',dict(verdict='INSTRUMENT_SMOKE',same_action_replay_exact=True,
                baseline=baseline['summary'],capability_pass=False,reachability_verified=False));return
        rng=np.random.default_rng(args.seed);mean=np.zeros((phases,6,6));scale=np.ones_like(mean)
        scale[...,:3]=1.25
        best=None;records=[];started=time.time()
        for generation in range(args.generations):
            population=[]
            for candidate in range(args.population):
                if time.time()-started>=args.budget_seconds:break
                parameters=rng.normal(mean,scale)
                if generation==0 and candidate==0 and args.initial_sequence is not None:
                    parameters=np.asarray(json.loads(args.initial_sequence.read_text(encoding='utf-8'))['parameters'],float)
                    if parameters.shape!=mean.shape or not np.isfinite(parameters).all():raise ValueError('saved command sequence has wrong shape')
                result=rollout(e,snapshot,a,dev,parameters,phase_seconds,args.position_range_m)
                entry=dict(generation=generation,candidate=candidate,parameters=parameters.tolist(),score=result['score'],**result['summary'])
                records.append(entry);population.append((result['score'],parameters))
                if best is None or result['score']>best['score']:
                    best=result|dict(parameters=parameters.tolist());atomic_json(args.out/'best_sequence.json',best)
                atomic_json(args.out/'status.json',dict(stage='measuring',pid=os.getpid(),timestamp=time.time(),
                    completed_candidates=len(records),maximum_candidates=args.generations*args.population,
                    best_roll_degrees=best['summary']['maximum_roll_from_start_deg'],best_front_up=best['summary']['minimum_front_up']))
                print(json.dumps(dict(candidate=len(records),best_front_up=best['summary']['minimum_front_up'],attained=best['summary']['attained'])),flush=True)
                if best['summary']['attained']:break
            if not population or best['summary']['attained'] or time.time()-started>=args.budget_seconds:break
            elites=np.stack([params for _,params in sorted(population,key=lambda item:item[0],reverse=True)[:max(2,len(population)//4)]])
            mean=elites.mean(0);scale=np.maximum(elites.std(0),.12)
        verified=False
        if best is not None:
            repeat=rollout(e,snapshot,a,dev,np.asarray(best['parameters']),phase_seconds,args.position_range_m)
            np.testing.assert_array_equal(best['summary']['final_qpos'],repeat['summary']['final_qpos'])
            np.testing.assert_array_equal(best['summary']['final_qvel'],repeat['summary']['final_qvel'])
            verified=bool(best['summary']['attained'] and not baseline['summary']['attained'])
            atomic_json(args.out/'best_repeat.json',repeat)
        report=dict(verdict='PHYSICAL_ROLL_FOUND' if verified else 'BOUNDED_SEARCH_INCONCLUSIVE',
            reachability_verified=verified,capability_pass=False,same_action_replay_exact=True,
            completed_candidates=len(records),elapsed_search_seconds=time.time()-started,
            hold=baseline['summary'],best=None if best is None else best['summary'],candidates=records,
            model_total_mass_kg=float(e.model.body_mass.sum()),model_timestep_s=float(e.model.opt.timestep),bridge_dt_s=e.dt,
            joint_limits=[dict(name=e.model.joint(j).name,range_rad=e.model.jnt_range[j].tolist()) for j in range(e.model.njnt) if e.model.jnt_limited[j]],
            scope=declaration['disclosure'])
        atomic_json(args.out/'report.json',report)
        atomic_json(args.out/'status.json',dict(stage='complete',timestamp=time.time(),verdict=report['verdict'],completed_candidates=len(records)))
    except Exception as error:
        atomic_json(args.out/'status.json',dict(stage='failed',timestamp=time.time(),error=str(error)));raise
    finally:
        if e is not None:e.close()


if __name__=='__main__':main()
