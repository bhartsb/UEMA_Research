# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Responsive adult exchanges and actual departures, without infant rewards.

Interaction timing is scene design, not a developmental success threshold.
Every random choice lives in the saved adult state. Replays use recorded speech
and actual poses, never this controller or the second child's responses.
"""
import numpy as np
from .caregiver_language import CONTRACT, OBJECT_NAMES
from .caregiver_visual import clear_approach


def draw(s, low, high=None):
    rng=np.random.default_rng();rng.bit_generator.state=s['exchange_rng']
    value=rng.integers(low) if high is None else rng.uniform(low,high)
    s['exchange_rng']=rng.bit_generator.state
    return value


def initialize(c,env):
    s=c.state
    if s.get('interaction_contract')==CONTRACT:return
    if s.get('interaction_contract') is not None:raise ValueError('unknown caregiver interaction state')
    # Add state to the current episode; keep its pose, object phase, speech cursor
    # and all recorded history. A loaded utterance is allowed to finish.
    s.update(interaction_contract=CONTRACT,exchange_rng=np.random.default_rng(env.seed+643771).bit_generator.state,
        adult_phase='shared',adult_visibility=1.,adult_departures=0,adult_returns=0,
        last_variant={},greeting_pending=True,pending_touch_until=0.,face_look_time=0.,
        object_look_time=0.,motion_anchor=env.data.mocap_pos[env._target_id].copy(),motion_checked_at=s['time'],
        motion_until=0.,last_face_attention=False,last_object_attention=False)
    s['departure_at']=s['time']+draw(s,180.,300.)
    s['quiet_at']=s['time']+draw(s,35.,55.)


def say(c,env,group,reason,evidence=None):
    s=c.state;speech=env._audio.speech
    if speech.busy:return False
    choices=[k for k in c.bank.groups[group] if k!=s['last_variant'].get(group)]
    key=choices[int(draw(s,len(choices)))]
    c._event(env,dict(phrase=key,speech_group=group,reason=reason,evidence=evidence or {},
                     interaction_contract=CONTRACT))
    s['last_variant'][group]=key
    # A response interval starts AFTER the spoken words finish.
    s['next_reply']=s['time']+len(speech.clip)/speech.sample_rate+draw(s,2.,4.)
    return True


def looks_at(env, point, bodies):
    m=env.model;d=env.data;cid=m.camera('eye_left').id
    eye=d.cam_xpos[cid];direction=np.asarray(point)-eye;distance=np.linalg.norm(direction)
    if distance<1e-8:return False
    direction/=distance;forward=-d.cam_xmat[cid].reshape(3,3)[:,2]
    if np.dot(forward,direction)<np.cos(np.radians(15)):return False
    # A SHAPE THE INFANT CANNOT SEE DOES NOT BLOCK A LOOK (audit P277, operator 2026-09-19).
    # mj_ray is geometric: measured on mujoco 3.3.0 it skips a fully transparent geom only
    # when that geom carries NO material (matid -1, and only at alpha exactly 0), and all 188
    # geoms of the crib scene carry one, so the engine never skips a hidden shape here. The
    # ray is re-cast from just past every hidden shape (geom_rgba alpha < .05, the line
    # caregiver_visual.mobile_obstructions uses) until it meets a visible shape or a shape of
    # the looked-at body. The looked-at body's OWN hidden shapes still count as that body:
    # the hidden ball (radius 0.025 m) fills the visible ring's hole (inner radius 0.022 m),
    # so a look at the ring's centre lands on it. Measured on run H84 (8 bodies, 204,751
    # sampled ticks with the adult visible and free to answer): under the single-ray rule
    # used until this change the line from the left eye to the adult's head met only hidden
    # toy shapes on 2.9 of 100 ticks (0 to 14.4 across bodies), mostly the switched-off pad's
    # 0.07 m square around the 0.05 m ball; a withdrawn toy lay on that line on 0 ticks.
    origin=np.array(eye,dtype=float);gid=np.array([-1],dtype=np.int32);own=int(m.cam_bodyid[cid])
    for _ in range(2*m.ngeom+2):                 # a shape is met at most twice: entering and leaving
        hit=env._mj.mj_ray(m,d,origin,direction,None,True,own,gid)
        if hit<0 or gid[0]<0:return False
        if int(m.geom_bodyid[gid[0]]) in bodies:return True
        if m.geom_rgba[gid[0],3]>=.05:return False
        origin=origin+direction*(hit+1e-6)
    return False


def contact_now(c,env):
    target=env.model.body('target').id;infant=set(map(int,env._touch_body_ids))
    for contact in env.data.contact:
        a=int(env.model.geom_bodyid[contact.geom1]);b=int(env.model.geom_bodyid[contact.geom2])
        if (a==target and b in infant) or (b==target and a in infant):return True
    return False


def tick(c,env):
    initialize(c,env)
    s=c.state;now=s['time'];dt=env.dt;m=env.model;d=env.data
    eye=d.cam_xpos[m.camera('eye_left').id].copy();shift=eye-s['anchor'];s['anchor']=eye
    s['target_home']=s['target_home']+shift;s['target_goal']=s['target_goal']+shift
    s['head_offset']=clear_approach(env,c,eye);near=eye+s['head_offset']
    head=d.mocap_pos[c.ids['caregiver_head']]
    phase=s['adult_phase'];busy=env._audio.speech.busy
    target=d.mocap_pos[env._target_id]
    touched=contact_now(c,env)
    if touched and not s['last_contact']:s['pending_touch_until']=now+2.
    s['last_contact']=touched
    face_bodies={m.body('caregiver_head').id}
    face_look=s.get('adult_visibility',1.)>0 and looks_at(env,head,face_bodies)
    object_look=s.get('object_present',True) and looks_at(env,target,{m.body('target').id})
    s['face_look_time']=s['face_look_time']+dt if face_look else 0.
    s['object_look_time']=s['object_look_time']+dt if object_look else 0.
    if not face_look:s['last_face_attention']=False
    if not object_look:s['last_object_attention']=False
    # WHOSE MOTION THIS IS. The toy is a MuJoCo mocap body - body('target') has dofnum 0
    # and jntnum 0 - so it has no equation of motion and contact cannot displace it.
    # Measured 2026-09-15: with the adult's carry disabled and every infant actuator
    # driven to full for 1.0 s, the toy's displacement is exactly 0 m. The only writer is
    # the adult's own carry (caregiver.py:_move_and_record), capped at 0.06 m/s: with the
    # infant still, one 0.5 s window measures 0.030000 m of carried path and 0.028023 m of
    # straight-line displacement, 1.87x the 0.015 m line below. The adult's destination
    # does track the infant's eye (line 70), so infant HEAD motion is an indirect cause,
    # and infant contact FREEZES the carry (caregiver.py:260-261) rather than driving it.
    # Subtracting the adult's own carry would remove most move_* utterances and change the
    # speech mix the model hears: a scene-semantics decision, not a repair.
    if now-s['motion_checked_at']>=.5:
        displacement=float(np.linalg.norm(target-s['motion_anchor']))
        if displacement>.015 and s.get('object_present',True):s['motion_until']=now+1.
        s['motion_displacement']=displacement;s['motion_anchor']=target.copy();s['motion_checked_at']=now

    if phase in ('shared','quiet') and now>=max(s['departure_at'],s['next_reply']) and not busy:
        say(c,env,'farewell','adult_departure')
        s['adult_phase']='farewell';phase='farewell'
    if phase=='farewell':
        s['head_goal']=near
        if not env._audio.speech.busy:
            s.update(adult_phase='leaving',visible=False,scene_phase='withdrawing',phase_until=None)
            # Fixed destination for this departure; following a moving child must
            # not keep the adult endlessly chasing its withdrawal endpoint.
            away=s['head_offset'].copy();away[2]=0;away/=np.linalg.norm(away)
            s['departure_head']=eye+away*1.1+[0.,0.,.4]
            s['target_goal']=eye+[0.,.55,.35]
        return
    if phase in ('leaving','away'):
        s['head_goal']=s['departure_head'];s['visible']=False
        if phase=='leaving' and np.linalg.norm(head-s['head_goal'])<.01:
            s['adult_phase']='away';s['adult_away_until']=now+draw(s,30.,60.);s['adult_departures']+=1
        elif phase=='away' and now>=s['adult_away_until']:
            s['adult_phase']='returning'
        return
    if phase=='returning':
        s['head_goal']=near;s['visible']=False
        if np.linalg.norm(head-near)<.02 and not busy:
            s.update(adult_phase='shared',scene_phase='starting',phase_until=None,greeting_pending=False,
                     departure_at=now+draw(s,180.,300.),quiet_at=now+draw(s,35.,55.))
            s['adult_returns']+=1
            say(c,env,'return','adult_return')
        return
    s['head_goal']=near
    if busy:return
    if now<s['next_reply']:return
    if s['greeting_pending']:
        say(c,env,'greeting','begin_exchange');s['greeting_pending']=False;return
    if phase=='quiet':
        # A new contact can start an exchange after the adult's listening pause.
        if now<s['quiet_until'] and not (touched and now<s['pending_touch_until']):return
        s['adult_phase']='shared';s['quiet_at']=now+draw(s,35.,55.)

    object_phase=s.get('scene_phase','starting')
    if object_phase=='starting' or (object_phase=='absent' and now>=s['phase_until']):
        index=s.get('scene_index',0);style=(index+env.seed)%3
        goal=s['target_home']+np.array([(-1 if index%2 else 1)*.07,0,.015])
        c._event(env,dict(target=goal.tolist(),style=style,visible=True,interaction_contract=CONTRACT))
        s.update(scene_index=index+1,scene_phase='presenting',phase_until=None)
        say(c,env,'present_'+OBJECT_NAMES[style],'object_presented',{'style':style})
        return
    if object_phase=='engaged' and now>=s['phase_until']:
        say(c,env,'put_away','object_withdrawal')
        c._event(env,dict(target=(eye+[0.,.55,.35]).tolist(),visible=False,interaction_contract=CONTRACT))
        s.update(scene_phase='withdrawing',phase_until=None)
        return
    name=OBJECT_NAMES[s['style']]
    if touched and now<s['pending_touch_until'] and s.get('object_present',True):
        say(c,env,'touch_'+name,'observed_contact',{'contact':True,'style':s['style']})
        s['pending_touch_until']=0.;return
    if s['face_look_time']>=.4 and not s['last_face_attention']:
        say(c,env,'face_look','visible_face_attention',{'visible_seconds':s['face_look_time']})
        s['last_face_attention']=True;return
    if s['object_look_time']>=.4 and not s['last_object_attention']:
        say(c,env,'look_'+name,'visible_object_attention',{'visible_seconds':s['object_look_time'],'style':s['style']})
        s['last_object_attention']=True;return
    if now<s['motion_until'] and object_look:
        say(c,env,'move_'+name,'observed_object_motion',{'distance_m':s['motion_displacement'],'style':s['style']})
        s['motion_until']=0.;return
    if now>=s['quiet_at']:
        say(c,env,'quiet','quiet_presence');s['adult_phase']='quiet'
        s['quiet_until']=s['next_reply']+draw(s,12.,24.)
