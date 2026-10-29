"""Paired physical test of the actual learner's remembered hand-goal stage.

The frozen source keeps its old memory. Baseline behavior supplies new physical
observations until the unchanged goal ranker selects a pose that was observed.
Two ordinary motor-latch decisions per held-out body are tested off versus on.
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
from mimo_bridge.tools.commit_guard import commit_reading,verdict
from lem.model.physical_goal import pose_error_per_body
from compare_synergy_physical import checksum
from prove_takeover_restart import runtime_fingerprint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def native_hand_readings(session):
    # An old episode payload predates these read-only fields. A snapshot restore
    # can restore that payload again, so read actual native q/qvel every time.
    control=session.env.hand_target_control
    session.frames[0].meta['hand_joint_angles']=session.env.data.qpos[control._joint_addresses].copy()
    session.frames[0].meta['hand_joint_velocities']=session.env.data.qvel[control._joint_dof_addresses].copy()


def baseline_observe(session):
    # Keep ordinary goal selection/pose recording and native joint readbacks.
    # Only the new hand choice is off while collecting private baseline starts.
    model=session.model;original=model._plan_chain
    def without_hand(*args,**kwargs):
        previous=model.hand_goals;model.hand_goals='off'
        try:return original(*args,**kwargs)
        finally:model.hand_goals=previous
    native_hand_readings(session)
    with patch.object(model,'_plan_chain',without_hand):
        return session.observe()


def physical(session,snapshot,command,joints,gaze,target):
    session.restore(snapshot)
    env=session.env;control=env.hand_target_control;kin=control.kinematics
    start=float(env.data.time);trajectory=hashlib.sha256()
    for _ in range(25):
        result=env.step(command[0],hand_joint_goal=joints[0],gaze=gaze,render=False)
        if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
            raise RuntimeError('Nonfinite physical state')
        np.testing.assert_array_equal(result.info['hand_joint_goal'],joints[0])
        trajectory.update(env.data.qpos.tobytes());trajectory.update(env.data.qvel.tobytes())
        trajectory.update(result.info['executed_motor_targets'].tobytes())
    assert np.isclose(float(env.data.time)-start,.25,atol=1e-8,rtol=0)
    pos,quat=session.pose();pose=torch.zeros(1,len(pos),10,dtype=torch.float64)
    pose[0,:,:3]=torch.from_numpy(pos/200.);pose[0,:,3:7]=torch.from_numpy(quat)
    actual=kin.local_encoding(pose);goal=kin.local_encoding(target.double())
    fingers=[i for h in kin.geometry['hands'] for i in h['finger_indices']]
    rotation=Rotation.from_quat(actual[0,fingers,3:7].numpy())*Rotation.from_quat(goal[0,fingers,3:7].numpy()).inv()
    return dict(hand_shape_error=float(pose_error_per_body(actual[...,:3]*200.,actual[...,3:7],goal,fingers)[0]),
        finger_position_error_cm=float(((actual[:,fingers,:3]-goal[:,fingers,:3])*200.).norm(dim=-1).mean()),
        finger_orientation_error_deg=float(np.degrees(rotation.magnitude()).mean()),
        pose=pose.tolist(),joint_angles_rad=env.data.qpos[control._joint_addresses].tolist(),
        qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),trajectory_sha256=trajectory.hexdigest(),seconds=.25)


def compare(session,before):
    model=session.model;branches={};selected_goal=None;goal_tokens=None
    for mode,enabled in (('baseline',False),('hand_memory',True)):
        session.restore(before);model.hand_goals='memory' if enabled else 'off'
        native_hand_readings(session)
        # the memory reads the last chosen goal from _last_plan_inputs: never wipe it (H.fresh_plan_inputs)
        prior=model._last_plan_inputs
        result=session.observe();inputs=copy.deepcopy(H.fresh_plan_inputs(model,prior))
        if inputs is None:raise RuntimeError('The collected motor-latch decision did not plan')
        if goal_tokens is None:goal_tokens=inputs['goal']
        else:
            for sense in goal_tokens:torch.testing.assert_close(inputs['goal'][sense],goal_tokens[sense],atol=0,rtol=0)
        command=np.array(result.motor_for_action,copy=True);joints=np.array(result.hand_joint_goal,copy=True)
        if enabled:
            torch.testing.assert_close(inputs['decision']['hand_current_angles'],
                torch.from_numpy(session.env.data.qpos[session.env.hand_target_control._joint_addresses].copy())[None],atol=0,rtol=0)
            torch.testing.assert_close(inputs['decision']['hand_current_velocities'],
                torch.from_numpy(session.env.data.qvel[session.env.hand_target_control._joint_dof_addresses].copy())[None],atol=0,rtol=0)
            selected_goal=inputs['decision']['hand_goal']
            if not bool(selected_goal['valid'].all()):raise RuntimeError('Chosen remembered pose was not observed')
            descent=copy.deepcopy((model.last_intention or {}).get('hand_descent'))
            if not descent or descent.get('stopped_by')!='completed':raise RuntimeError('Actual hand stage did not complete')
        else:descent=None
        branches[mode]=dict(command=command,joints=joints,gaze=np.asarray(result.gaze_out).reshape(-1)[:2].copy(),
            after_observation=session.snapshot(),descent=descent,
            goal_index=plain((model.last_plan_record or {}).get('goal_ring_index')),
            action_source=model.last_action_source)
        session.restore(before);model.hand_goals='memory' if enabled else 'off'
        native_hand_readings(session)
        repeated=session.observe()
        np.testing.assert_array_equal(repeated.motor_for_action,command)
        np.testing.assert_array_equal(repeated.hand_joint_goal,joints)
        np.testing.assert_array_equal(np.asarray(repeated.gaze_out).reshape(-1)[:2],branches[mode]['gaze'])
    geometry=model._hand_geometry;fingers={i for h in geometry['hands'] for i in h['finger_indices']}
    nonfinger=[i for i in range(session.n_handles) if i not in fingers]
    base=branches['baseline'];chosen=branches['hand_memory']
    np.testing.assert_array_equal(base['command'][:,nonfinger],chosen['command'][:,nonfinger])
    np.testing.assert_array_equal(base['command'][...,7:],chosen['command'][...,7:])
    np.testing.assert_array_equal(base['gaze'],chosen['gaze'])
    outcomes={}
    for mode,branch in branches.items():
        values=physical(session,branch['after_observation'],branch['command'],branch['joints'],branch['gaze'],selected_goal['pose'])
        repeated=physical(session,branch['after_observation'],branch['command'],branch['joints'],branch['gaze'],selected_goal['pose'])
        if values!=repeated:raise RuntimeError('Full physical trajectory did not reproduce exactly')
        outcomes[mode]=dict(values,command=plain(branch['command']),hand_joint_goal=plain(branch['joints']),
            descent=branch['descent'],action_source=branch['action_source'])
    bp=np.asarray(outcomes['baseline']['pose'])[0];hp=np.asarray(outcomes['hand_memory']['pose'])[0]
    consequence={}
    for label,indices in (('wrists',[h['wrist_index'] for h in geometry['hands']]),
                          ('nonfinger_body',[i for i in nonfinger if i!=session.gaze_idx])):
        rotations=Rotation.from_quat(hp[indices,3:7])*Rotation.from_quat(bp[indices,3:7]).inv()
        consequence[label]=dict(mean_position_change_cm=float(np.linalg.norm((hp[indices,:3]-bp[indices,:3])*200.,axis=-1).mean()),
            mean_orientation_change_deg=float(np.degrees(rotations.magnitude()).mean()))
    return dict(goal=plain(selected_goal),outcomes=outcomes,exact_replay=True,
        nonfinger_orders_unchanged=True,nonfinger_path_change=consequence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--maximum-steps',type=int,default=300)
    args=parser.parse_args();reading=commit_reading()
    if reading is None or not verdict(reading[0],reading[1],12,6):raise RuntimeError(f'Memory guard refused: {reading}')
    args.out.mkdir(parents=True,exist_ok=False);started=time.time();fingerprint=runtime_fingerprint()
    overrides=dict(hand_target_mode='wrist_relative_v1',hand_goals='memory',
        hand_response_calibration=str(args.calibration),physical_goals='off',synergy_planner='off')
    declaration=dict(checkpoint=str(args.checkpoint),source_model_sha256=digest(args.checkpoint/'model.safetensors'),
        calibration_file_sha256=digest(args.calibration),runtime_sha256=fingerprint,overrides=overrides,
        calibration_payload_sha256=json.loads(args.calibration.read_text())['sha256'],
        tested_route='LemLite.step observe -> LemLite._plan_chain hand memory stage -> motor_for_action and hand_joint_goal -> actual physics',
        held_out_bodies=[4,5,6,7],starts_per_body=2,
        production_modified=False,learning=False,script_sha256=digest(__file__),script_path=str(Path(__file__).resolve()),
        goal_source='Unchanged goal ranker chooses actual observed memory; legacy missing poses remain missing until naturally replaced.',
        collection='Baseline behavior; first two valid planning decisions at ordinary motor-latch boundaries, at least10model steps apart.',
        execution='Returned motor_for_action and explicit joint goal, held250ms; same selected gaze held in both arms.',
        admission='All eight exact replays, unchanged old weights and nonfinger commands, improved hand shape in all four body means and paired95difference entirely below zero.')
    (args.out/'declaration.json').write_text(json.dumps(declaration,indent=2))
    def status(**values):
        (args.out/'status.json').write_text(json.dumps(dict(dict(state='running',started_at=started,elapsed_s=time.time()-started),**values)))
    status(completed=0,total=8,phase='loading preserved learner')
    frozen=H.load_frozen(args.checkpoint,intention=True,arg_overrides=overrides,threads=4)
    episode,note=H.load_episode(frozen)
    if episode is None:raise RuntimeError(note)
    # Authorized private control-contract migration; retain all eight physical states.
    for world in episode['worlds']:world['env_kwargs']['hand_target_mode']='wrist_relative_v1'
    model=frozen.model;weights=checksum(model);rows=[];collected={}
    for body in declaration['held_out_bodies']:
        model.hand_goals='memory'
        session=H.Session(frozen,episode=episode,body=body,babble=float(frozen.args.babble_noise))
        try:
            # The legacy frame has no native joint fields; add current read-only
            # measurements, without advancing or reconstructing its sensory history.
            native_hand_readings(session)
            count=0;last=-10**9;skipped_converged=0;skipped_unobserved=0
            for offset in range(args.maximum_steps):
                before=(session.snapshot() if session._policy_i%session.hold==0
                        and model.step_count%model.intention_every==0 and offset-last>=10 else None)
                prior=model._last_plan_inputs
                baseline_observe(session)
                inputs=H.fresh_plan_inputs(model,prior)
                goal=None if inputs is None else inputs['decision'].get('hand_goal')
                if before is not None and inputs is not None:
                    if goal is None or not bool(goal['valid'].all()):skipped_unobserved+=1
                    elif inputs.get('plan0') is None:skipped_converged+=1
                if (before is not None and goal is not None and bool(goal['valid'].all())
                        and inputs.get('plan0') is not None and model.last_action_source=='planner'):
                    continuation=session.snapshot()
                    row=compare(session,before);row.update(body=body,start=count,offset=offset,
                        sim_time=float(before['world'][0]['payload']['sim_time']))
                    rows.append(row);count+=1;last=offset
                    with (args.out/'trials.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    print('completed',body,count,{k:v['hand_shape_error'] for k,v in row['outcomes'].items()},flush=True)
                    session.restore(continuation);model.hand_goals='memory'
                status(completed=len(rows),total=8,body=body,model_steps_observed=offset+1,
                    phase='collecting natural remembered goals' if count<2 else 'body completed')
                if count==2:break
                session.policy_act()
            collected[body]=dict(starts=count,steps_observed=offset+1,
                skipped_without_observed_pose=skipped_unobserved,skipped_without_plan=skipped_converged)
        finally:session.close()
    if checksum(model)!=weights:raise RuntimeError('Old learned weights changed')
    if runtime_fingerprint()!=fingerprint:raise RuntimeError('Runtime source changed during comparison')
    summary={};complete=len(rows)==8 and all(v['starts']==2 for v in collected.values())
    for metric in ('hand_shape_error','finger_position_error_cm','finger_orientation_error_deg'):
        values={mode:np.array([np.mean([r['outcomes'][mode][metric] for r in rows if r['body']==b])
            for b in declaration['held_out_bodies']]) for mode in ('baseline','hand_memory')}
        delta=values['hand_memory']-values['baseline'];draws=np.random.default_rng(641).integers(4,size=(10000,4))
        summary[metric]=dict(means={k:float(v.mean()) for k,v in values.items()},
            body_values={k:v.tolist() for k,v in values.items()},difference=float(delta.mean()),
            paired95_difference=np.quantile(delta[draws].mean(-1),[.025,.975]).tolist(),bodies_improved=int((delta<0).sum()))
    main_result=summary['hand_shape_error']
    passed=complete and main_result['bodies_improved']==4 and main_result['paired95_difference'][1]<0
    result=dict(declaration,passed=passed,weights_unchanged=True,exact_replay=True,completed_trials=len(rows),
        nonfinger_commands_unchanged=True,sensory_goals_identical=True,
        collected=collected,summary=summary,paired95_difference=main_result['paired95_difference'],
        elapsed_s=time.time()-started)
    (args.out/'result.json').write_text(json.dumps(result,indent=2));status(state='completed',completed=len(rows),total=8,passed=passed)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
