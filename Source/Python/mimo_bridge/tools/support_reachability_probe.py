# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Bounded native-hand reach diagnostic for an anchored body-relative rail.

Kinematics only proposes reference commands; real MuJoCo execution must actually
produce contact. This auditor never trains a learner or certifies grasp/standing.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mimo_bridge.env import BridgeEnv
from mimo_bridge.lem_model import LemLite
from mimo_bridge.support_fixture import place,RAIL,NAME
from mimo_bridge.train_mimo import parse_args,env_kwargs_for
from .causal_test import _limits_for_run
from .motor_reachability_probe import pose
from .developmental_probe import descendant_names,body_signature
from .training_continuation import atomic_json


def distance_to_rail(point,center,axis,half):
    return float(np.linalg.norm(point-center-axis*np.clip(np.dot(point-center,axis),-half,half)))


def execute(e,snapshot,a,dev,raw,steps):
    e.restore_full_state(snapshot)
    previous=torch.from_numpy(pose(e,float(dev['strength_limit'])))
    hands=set(descendant_names(e.model,'left_hand')+descendant_names(e.model,'right_hand'))
    rail=e.model.geom(RAIL).id;mid=int(e.model.body_mocapid[e.model.body(NAME).id])
    anchor=e.data.mocap_pos[mid].copy();rows=[]
    for _ in range(steps):
        achieved=torch.from_numpy(pose(e,float(dev['strength_limit'])));achieved[...,:3]/=200.
        command=LemLite.planned_execution(raw,dev,achieved,previous=previous,smoothing=a.smooth_actions)
        for _ in range(a.action_repeat):e.step(command.numpy()[0],gaze=np.zeros(2),render=False)
        contacts=[]
        for i in range(e.data.ncon):
            c=e.data.contact[i]
            if rail not in (c.geom1,c.geom2):continue
            other=int(c.geom2 if c.geom1==rail else c.geom1)
            body=e.model.body(int(e.model.geom_bodyid[other])).name
            if body not in hands:continue
            f=np.zeros(6);e._mj.mj_contactForce(e.model,e.data,i,f)
            if f[0]>0:contacts.append(dict(body=body,normal_force_n=float(f[0])))
        rows.append(dict(time_s=float(e.data.time),qpos=e.data.qpos.copy().tolist(),
            contacts=contacts,maximum_control=float(np.abs(e.data.ctrl).max())))
        previous=command
    np.testing.assert_array_equal(anchor,e.data.mocap_pos[mid])
    return dict(rows=rows,hand_contact=any(row['contacts'] for row in rows),
        contact_frames=sum(bool(row['contacts']) for row in rows),
        maximum_hand_normal_force_n=max([c['normal_force_n'] for row in rows for c in row['contacts']] or [0.]),
        anchor_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=76231);p.add_argument('--samples',type=int,default=128)
    p.add_argument('--seconds',type=float,default=2.);p.add_argument('--height-fraction',type=float,default=.65)
    args=p.parse_args()
    if not 1<=args.samples<=512 or not 0<args.seconds<=3:p.error('Bounded samples/duration required')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    a=parse_args(['--out',str(args.out)]);vars(a).update(json.loads((args.checkpoint.parent.parent/'args.json').read_text()))
    dev,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
    kw=env_kwargs_for(a,dev['phase_name'],str(args.out/'world'))
    kw.update(seed=args.seed,vision=False,caregiver_mode='off',caregiver_seconds=0.,caregiver_record_dir='',
        mobile=False,critters=False,gaze_reflex=False,spontaneous_saccades=False,support_fixture='rail_v1')
    e=BridgeEnv(**kw)
    try:
        e.reset()
        if e._target_id is not None:e.data.mocap_pos[e._target_id]=[0,0,3];e._target_id=None
        neutral=pose(e,float(dev['strength_limit']))
        for _ in range(25):e.step(neutral[0],gaze=np.zeros(2),render=False)
        geometry=place(e,args.height_fraction);snapshot=e.save_full_state();initial=e.data.qpos.copy()
        steps=max(2,int(round(args.seconds/(e.dt*a.action_repeat))))
        neutral=pose(e,float(dev['strength_limit']));neutral[...,:3]/=200.;neutral=torch.from_numpy(neutral)
        control=execute(e,snapshot,a,dev,neutral,steps)
        rng=np.random.default_rng(args.seed);proposals=[]
        rail=e.model.geom(RAIL).id;center=e.data.geom_xpos[rail].copy()
        axis=e.data.geom_xmat[rail].reshape(3,3)[:,2].copy();half=e.model.geom_size[rail,1]
        for side in ('L','R'):
            handle=e.decoder.handles[e.decoder.handle_names.index('wrist_'+side)];best=[]
            for _ in range(args.samples):
                e.data.qpos[:]=initial
                for j in handle.joint_ids:
                    lo,hi=e.model.jnt_range[j]
                    e.data.qpos[int(e.model.jnt_qposadr[j])]=rng.uniform(lo,hi)
                e._mj.mj_kinematics(e.model,e.data)
                distance=distance_to_rail(e.data.xpos[handle.body_id],center,axis,half)
                raw=pose(e,float(dev['strength_limit']));raw[...,:3]/=200.
                best.append((abs(distance-(geometry['radius_m']+.02)),torch.from_numpy(raw.copy())))
            for score,raw in sorted(best,key=lambda x:x[0])[:3]:proposals.append((side,score,raw))
        results=[]
        for i,(side,score,raw) in enumerate(proposals):
            row=execute(e,snapshot,a,dev,raw,steps)
            repeated=execute(e,snapshot,a,dev,raw,steps)
            if row!=repeated:raise ValueError('Physical support reference did not repeat exactly')
            row.update(side=side,kinematic_distance_error_m=score,raw_command=raw.numpy().tolist(),same_action_replay_exact=True)
            atomic_json(args.out/f'candidate_{i}.json',row);results.append(row)
        found=any(row['hand_contact'] for row in results)
        avatar=descendant_names(e.model,'mimo_location')
        atomic_json(args.out/'report.json',dict(verdict='HAND_RAIL_CONTACT_FOUND' if found else 'BOUNDED_REACH_INCONCLUSIVE',
            body_signature=body_signature(e,a,dev),world_seed=args.seed,limits=dev,limits_source=origin,geometry=geometry,
            avatar_mass_kg=float(sum(e.model.body_mass[e.model.body(name).id] for name in avatar)),
            hold=control,results=results,hand_contact_reachable=found,
            grip_load_bearing_verified=False,pull_to_stand_verified=False,capability_pass=False,
            learner_present=False,scope='Auditor-commanded physical reach/contact only; anchored support load is tested separately.'))
    finally:e.close()


if __name__=='__main__':main()
