# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measure native finger command channels and inventory uncommanded actuators.

Kinematic finger targets are an auditor's calibration only. They are never a
reward, training label, movement policy or installed controller.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mimo_bridge.env import BridgeEnv
from mimo_bridge.lem_model import LemLite
from mimo_bridge.train_mimo import parse_args,env_kwargs_for
from .causal_test import _limits_for_run
from .motor_reachability_probe import pose
from .training_continuation import atomic_json


def rollout(env,snapshot,a,dev,raw,seconds):
    env.restore_full_state(snapshot)
    previous=torch.from_numpy(pose(env,float(dev['strength_limit'])))
    rows=[];controls=[]
    for _ in range(max(2,int(round(seconds/(env.dt*a.action_repeat))))):
        achieved=torch.from_numpy(pose(env,float(dev['strength_limit'])))
        achieved[...,:3]/=200.
        command=LemLite.planned_execution(raw,dev,achieved,previous=previous,smoothing=a.smooth_actions)
        for _ in range(a.action_repeat):env.step(command.numpy()[0],gaze=np.zeros(2),render=False)
        rows.append(env.data.qpos.copy());controls.append(env.data.ctrl.copy());previous=command
    return np.stack(rows),np.stack(controls)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=1.);args=p.parse_args()
    if not 0<args.seconds<=3:p.error('bounded duration must be in (0,3] seconds')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    a=parse_args(['--out',str(args.out)]);vars(a).update(json.loads((args.checkpoint.parent.parent/'args.json').read_text()))
    dev,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
    kw=env_kwargs_for(a,dev['phase_name'],str(args.out/'world'))
    kw.update(seed=65411,caregiver_mode='off',caregiver_seconds=0.,caregiver_record_dir='',vision=False,
              mobile=False,critters=False,gaze_reflex=False,spontaneous_saccades=False)
    env=BridgeEnv(**kw)
    try:
        env.reset()
        if env._target_id is not None:env.data.mocap_pos[env._target_id]=[0,0,3];env._target_id=None
        neutral=pose(env,float(dev['strength_limit']))
        for _ in range(25):env.step(neutral[0],gaze=np.zeros(2),render=False)
        snapshot=env.save_full_state();initial=env.data.qpos.copy();neutral=pose(env,float(dev['strength_limit']))
        raw=neutral.copy();raw[...,:3]/=200.;raw=torch.from_numpy(raw)
        baseline,_=rollout(env,snapshot,a,dev,raw,args.seconds)
        repeat,_=rollout(env,snapshot,a,dev,raw,args.seconds)
        np.testing.assert_array_equal(baseline,repeat)
        inventory=[];mapped=set();results=[]
        for name,h in zip(env.decoder.handle_names,env.decoder.handles):
            mapped.update(h.act_ids)
            inventory.append(dict(handle=name,body=env.model.body(h.body_id).name,
                joints=[env.model.joint(j).name for j in h.joint_ids],
                actuators=[env.model.actuator(j).name for j in h.act_ids],
                ranges_rad=[env.model.jnt_range[j].tolist() for j in h.joint_ids]))
            if name.split('_')[0] not in ('thumb','index','middle','ring','pinky'):continue
            env.restore_full_state(snapshot)
            addresses=[int(env.model.jnt_qposadr[j]) for j in h.joint_ids]
            wanted=[]
            for j,adr in zip(h.joint_ids,addresses):
                low,high=env.model.jnt_range[j]
                direction=1. if high-initial[adr]>=initial[adr]-low else -1.
                value=np.clip(initial[adr]+direction*.35,low,high)
                env.data.qpos[adr]=value;wanted.append(value)
            env._mj.mj_forward(env.model,env.data)
            desired=neutral.copy();i=env.decoder.handle_names.index(name)
            desired[0,i,:3]=env.data.xpos[h.body_id]*100
            desired[0,i,3:7]=env.data.xquat[h.body_id][[1,2,3,0]]
            desired[...,:3]/=200.
            actual,controls=rollout(env,snapshot,a,dev,torch.from_numpy(desired),args.seconds)
            repeated,_=rollout(env,snapshot,a,dev,torch.from_numpy(desired),args.seconds)
            np.testing.assert_array_equal(actual,repeated)
            delta=actual[-1,addresses]-baseline[-1,addresses]
            target=np.asarray(wanted)-initial[addresses]
            cosine=float(np.dot(delta,target)/(max(np.linalg.norm(delta)*np.linalg.norm(target),1e-15)))
            results.append(dict(handle=name,target_joint_changes_rad=target.tolist(),
                achieved_change_over_hold_rad=delta.tolist(),direction_cosine=cosine,
                maximum_joint_response_over_hold_deg=float(np.rad2deg(np.abs(actual[:,addresses]-baseline[:,addresses])).max()),
                maximum_chain_control=float(np.abs(controls[:,h.act_ids]).max()),
                same_action_replay_exact=True))
        mapped.update(row[0] for row in env.decoder.eye)
        unmapped=[dict(actuator=env.model.actuator(i).name,joint=env.model.joint(int(env.model.actuator_trnid[i,0])).name)
                  for i in range(env.model.nu) if i not in mapped]
        env.restore_full_state(snapshot)
        along=env.data.xmat[env.model.body('chest').id].reshape(3,3)@env._body_orientation_axes[0]
        head=env.data.xpos[env.model.body('head').id]
        feet=np.stack([env.data.xpos[env.model.body(s+'_foot').id] for s in ('left','right')])
        shoulders=np.stack([env.data.xpos[env.model.body(s+'_upper_arm').id] for s in ('left','right')])
        feet_level=float((feet@along).min())
        support=dict(pen_wall_top_height_m=float(env.data.geom_xpos[env.model.geom('pen_wall_n').id,2]+env.model.geom_size[env.model.geom('pen_wall_n').id,2]),
            head_to_foot_longitudinal_span_m=float(head@along-feet_level),
            shoulder_to_foot_longitudinal_spans_m=(shoulders@along-feet_level).tolist(),
            scope='Anatomical span in current pose; not a proven upright posture, handhold reach or pull-to-stand.')
        atomic_json(args.out/'report.json',dict(kind='physical_motor_channel_audit',checkpoint=str(args.checkpoint),
            limits=dev,limits_source=origin,seconds=args.seconds,action_repeat=a.action_repeat,motor_hold=a.motor_hold,
            smoothing=a.smooth_actions,inventory=inventory,finger_results=results,unmapped_actuators=unmapped,
            support_geometry=support,same_action_replay_exact=True,capability_pass=False,
            scope='Independent finger commands and anatomy calibration only. Grasping and load-bearing support remain separate physical/learned behaviors.'))
    finally:env.close()


if __name__=='__main__':main()
