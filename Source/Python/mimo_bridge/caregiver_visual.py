# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A simple adult, visible facial speech and an unobstructed approach to the crib.

The four existing mocap bodies are retained. Added shapes are visual only: no
infant joint, contact surface, actuator or sensory packet is added or removed.
"""
import copy
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 'caregiver_human_v2'
ADULT_BODIES = ('caregiver_head', 'caregiver_torso', 'caregiver_hand_l', 'caregiver_hand_r')
SKIN = '.82 .60 .46 1'
SHIRT = '.27 .49 .52 1'
HAIR = '.16 .085 .055 1'


def _geom(body, name, kind, size, pos='0 0 0', color=SKIN, **kw):
    return ET.SubElement(body, 'geom', name='caregiver_'+name, type=kind,
        size=size, pos=pos, rgba=color, contype='0', conaffinity='0',
        material='caregiver_matte', **kw)


def add_adult(root, world):
    assets=root.find('asset')
    if assets is None:assets=ET.SubElement(root,'asset')
    ET.SubElement(assets,'material',name='caregiver_matte',specular='.08',shininess='.1',emission='.12')
    bodies={name:ET.SubElement(world,'body',name=name,mocap='true',pos='0 0 2') for name in ADULT_BODIES}
    head=bodies['caregiver_head']
    _geom(head,'head_geom','ellipsoid','.086 .072 .112')
    _geom(head,'hair','ellipsoid','.091 .064 .096','0 .027 .032',HAIR)
    _geom(head,'hair_top','ellipsoid','.083 .059 .032','0 .006 .087',HAIR)
    for side in (-1,1):
        tag='l' if side<0 else 'r'
        _geom(head,'ear_'+tag,'ellipsoid','.013 .017 .027',f'{side*.084} .004 -.005')
        _geom(head,'ear_inner_'+tag,'ellipsoid','.006 .010 .017',f'{side*.092} -.002 -.005','.66 .40 .32 1')
        _geom(head,'eye_'+tag,'ellipsoid','.019 .007 .010',f'{side*.031} -.068 .026','.96 .94 .89 1')
        _geom(head,'iris_'+tag,'ellipsoid','.008 .0027 .0075',f'{side*.031} -.075 .026','.23 .13 .075 1')
        _geom(head,'pupil_'+tag,'ellipsoid','.0038 .0015 .0045',f'{side*.031} -.0775 .026','.035 .025 .02 1')
        _geom(head,'glint_'+tag,'sphere','.0018',f'{side*.031-.002} -.079 .029','1 .98 .92 1')
        _geom(head,'lid_'+tag,'ellipsoid','.021 .005 .0037',f'{side*.031} -.069 .035')
        _geom(head,'brow_'+tag,'ellipsoid','.021 .003 .0034',f'{side*.032} -.061 .048',HAIR)
        _geom(head,'hair_side_'+tag,'ellipsoid','.017 .033 .055',f'{side*.072} .016 .046',HAIR)
    _geom(head,'nose_bridge','ellipsoid','.009 .008 .021','0 -.066 .015')
    _geom(head,'nose','ellipsoid','.011 .013 .012','0 -.073 -.004')
    _geom(head,'mouth','ellipsoid','.019 .003 .001','0 -.069 -.044','.16 .045 .035 1')
    _geom(head,'upper_lip','ellipsoid','.023 .004 .0035','0 -.072 -.042','.58 .28 .24 1')
    _geom(head,'lower_lip','ellipsoid','.021 .004 .004','0 -.072 -.047','.69 .36 .30 1')
    torso=bodies['caregiver_torso']
    _geom(torso,'torso_geom','ellipsoid','.18 .095 .23',color=SHIRT)
    _geom(torso,'neck','capsule','.032 .06',color=SKIN)
    for side in ('l','r'):
        _geom(torso,'upperarm_'+side,'capsule','.042 .13',color=SHIRT)
        _geom(torso,'forearm_'+side,'capsule','.026 .13')
    for side in ('l','r'):
        hand=bodies['caregiver_hand_'+side]
        _geom(hand,'hand_'+side+'_geom','ellipsoid','.037 .025 .014')
        for finger in range(4):
            _geom(hand,f'finger_{side}_{finger}','capsule','.006 .015',
                  f'{(finger-1.5)*.014} -.033 0',quat='.70710678 .70710678 0 0')
        _geom(hand,'thumb_'+side,'ellipsoid','.010 .023 .010','-.036 -.008 -.003')


def _quat(matrix):
    q=Rotation.from_matrix(matrix).as_quat()
    return q[[3,0,1,2]]


def facing(eye, head):
    back=np.asarray(head)-eye;back=back/max(np.linalg.norm(back),1e-12)
    right=np.cross(back,[0.,0.,1.])
    if np.linalg.norm(right)<1e-6:right=np.array([1.,0.,0.])
    right/=np.linalg.norm(right)
    return np.column_stack((right,back,np.cross(right,back)))


def approach(eye, mobile_origin):
    """Lean beside the mobile, closer to the infant than the suspended frame."""
    away=np.asarray(eye[:2])-np.asarray(mobile_origin[:2])
    if np.linalg.norm(away)<.04:away=np.array([0.,1.])
    away/=np.linalg.norm(away)
    return np.r_[away*.18,.32]


def mobile_obstructions(env, caregiver, eye, offset):
    """Count obscured face points using the mobile's actual visible shapes."""
    head=eye+offset;rotation=facing(eye,head);m=env.model;d=env.data
    blocked=0
    for local in ((-.031,-.075,.026),(.031,-.075,.026),(0.,-.074,-.044)):
        delta=head+rotation@local-eye;length=np.linalg.norm(delta);direction=delta/length
        for gid in caregiver.mobile_geoms:
            if m.geom_rgba[gid,3]<.05:continue
            distance=env._mj.mju_rayGeom(d.geom_xpos[gid],d.geom_xmat[gid],m.geom_size[gid],
                                        eye,direction,int(m.geom_type[gid]))
            if 0<=distance<length:
                blocked+=1;break
    return blocked


def clear_approach(env, caregiver, eye):
    """The adult steps around an obstructing mobile; the infant's gaze is untouched.

    Recheck twice per simulated second or after appreciable infant displacement.
    Retain a clear current side to avoid a caregiver that darts around the crib.
    """
    s=caregiver.state;now=s.get('time',0.)
    if ('approach_eye' in s and now<s.get('approach_check_at',0.)
            and np.linalg.norm(eye-s['approach_eye'])<.02):
        return np.asarray(s['head_offset']).copy()
    origin=caregiver.mobile_mount if caregiver.mobile_mount is not None else eye+[0.,-.2,0.]
    preferred=approach(eye,origin)
    candidates=[]
    previous=np.asarray(s.get('head_offset',preferred))
    if .15<=np.linalg.norm(previous[:2])<=.35 and .19<=previous[2]<=.34:
        candidates.append(previous)
    candidates.append(preferred)
    for radius,height in ((.18,.32),(.26,.28),(.34,.20)):
        for angle in (45,-45,90,-90,135,-135,180,0):
            a=np.radians(angle);x,y=preferred[:2]/.18*radius
            candidates.append(np.array([x*np.cos(a)-y*np.sin(a),x*np.sin(a)+y*np.cos(a),height]))
    best=preferred;least=4
    for offset in candidates:
        blocked=mobile_obstructions(env,caregiver,eye,offset)
        if blocked<least:least=blocked;best=offset.copy()
        if blocked==0:break
    s.update(approach_eye=eye.copy(),approach_check_at=now+.5,face_points_obstructed=least)
    return best


def speech_opening(speech, dt):
    # The same source samples will be emitted during this physical step. Reading
    # them does not advance the audio cursor or use sound from the environment.
    n=max(0,int(dt*speech.sample_rate+speech.fraction+1e-9))
    samples=speech.clip[speech.cursor:speech.cursor+n]
    if not len(samples):return 0.
    rms=float(np.sqrt(np.mean(samples.astype(float)**2)))
    return float(np.clip((rms-.008)/.12,0,1))


def _segment(env, name, a, b, center, rotation, radius):
    a=rotation.T@(np.asarray(a)-center);b=rotation.T@(np.asarray(b)-center)
    delta=b-a;length=np.linalg.norm(delta);axis=delta/max(length,1e-12)
    q=np.array([1.+axis[2],-axis[1],axis[0],0.])
    q=np.array([0.,1.,0.,0.]) if np.linalg.norm(q)<1e-8 else q/np.linalg.norm(q)
    gid=env.model.geom('caregiver_'+name).id
    # MuJoCo compiles origin-centered geoms as coincident with their body.
    # Dynamic arm transforms must disable that shortcut before kinematics.
    env.model.geom_sameframe[gid]=0
    env.model.geom_pos[gid]=(a+b)*.5
    env.model.geom_quat[gid]=q
    env.model.geom_size[gid,:2]=[radius,max(length*.5,1e-5)]
    env.model.geom_rbound[gid]=radius+max(length*.5,1e-5)


def pose_adult(env, caregiver, recorded=None):
    s=caregiver.state;m=env.model;d=env.data;ids=caregiver.ids
    head=d.mocap_pos[ids['caregiver_head']]
    eye=np.asarray(s['anchor'])
    # A replay follows the recorded adult, including where its face was aimed.
    if recorded is not None:
        if recorded.get('visual_contract')!=VERSION:
            raise ValueError('caregiver replay lacks matching facial/arm motion records')
        for name,pose in recorded['adult_pose'].items():
            d.mocap_pos[ids[name]]=pose['position'];d.mocap_quat[ids[name]]=pose['quaternion']
        eye=np.asarray(recorded['infant_head_actual'])
        opening=float(recorded['mouth_opening'])
        for key in ('adult_phase','adult_visibility','interaction_contract'):
            if key in recorded:s[key]=recorded[key]
    else:
        opening=speech_opening(env._audio.speech,env.dt)
        d.mocap_quat[ids['caregiver_head']]=_quat(facing(eye,head))
        away=head-eye;away[2]=0
        if np.linalg.norm(away)<1e-6:away=np.array([0.,1.,0.])
        away/=np.linalg.norm(away)
        right=np.cross(away,[0.,0.,1.])
        up=np.array([0.,0.,1.])*.94-away*.342
        up/=np.linalg.norm(up)
        back=np.cross(up,right)
        torso_r=np.column_stack((right,back,up))
        torso=head+away*.21+[0.,0.,-.20]
        d.mocap_pos[ids['caregiver_torso']]=torso
        d.mocap_quat[ids['caregiver_torso']]=_quat(torso_r)
        d.mocap_pos[ids['caregiver_hand_l']]=d.mocap_pos[env._target_id]-right*.05+[0.,0.,.01]
        # THE ADULT HAS LET GO. While the infant holds the toy the adult's hand stays where it
        # let go, rather than riding along with the toy; after the infant releases it the
        # hand goes back to the toy over Caregiver.HAND_BLEND_STEPS model steps instead of
        # jumping, so no adult movement appears that the adult did not make.
        grip=s.get('grip') or {}
        if grip.get('adult_hand') is not None and (grip.get('held') or grip.get('blend',0)>0):
            let_go=np.asarray(grip['adult_hand'],dtype=np.float64)
            if grip.get('held'):
                d.mocap_pos[ids['caregiver_hand_l']]=let_go
            else:
                steps=max(1,getattr(caregiver,'HAND_BLEND_STEPS',30))
                w=float(grip['blend'])/steps
                d.mocap_pos[ids['caregiver_hand_l']]=w*let_go+(1-w)*d.mocap_pos[ids['caregiver_hand_l']]
                grip['blend']=int(grip['blend'])-1
        d.mocap_pos[ids['caregiver_hand_r']]=head+right*.22+away*.10+[0.,0.,-.22]
        if s.get('adult_phase') in ('leaving','away','returning') or (s.get('interaction_contract') and not s.get('visible',True)):
            retreat=float(np.clip((np.linalg.norm((head-eye)[:2])-.35)/.30,0,1))
            # Keep the hand beside the torso while returning without an object.
            # Following the withdrawn target here produced a raised, stretched arm.
            if s.get('adult_phase')=='returning' or (s.get('adult_phase') in ('shared','quiet') and not s.get('visible',True)):
                retreat=1.
            rest=torso-right*.19+torso_r@[0.,-.06,-.12]
            d.mocap_pos[ids['caregiver_hand_l']]=(1-retreat)*d.mocap_pos[ids['caregiver_hand_l']]+retreat*rest
        hand_r=np.column_stack((right,away,[0.,0.,1.]))
        for side in ('l','r'):d.mocap_quat[ids['caregiver_hand_'+side]]=_quat(hand_r)
        if s.get('interaction_contract'):
            distance=np.linalg.norm((head-eye)[:2])
            s['adult_visibility']=float(np.clip((.95-distance)/.20,0,1))
    if 'adult_visibility' in s:
        m.geom_rgba[caregiver.adult_geoms,3]=s['adult_visibility']
    s['mouth_opening']=opening
    gap=.011*opening
    m.geom_size[m.geom('caregiver_mouth').id,2]=.001+gap
    m.geom_pos[m.geom('caregiver_upper_lip').id,2]=-.042+gap*.6
    m.geom_pos[m.geom('caregiver_lower_lip').id,2]=-.047-gap*.8
    torso=d.mocap_pos[ids['caregiver_torso']]
    q=d.mocap_quat[ids['caregiver_torso']];torso_r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    head_q=d.mocap_quat[ids['caregiver_head']];head_r=Rotation.from_quat(head_q[[1,2,3,0]]).as_matrix()
    _segment(env,'neck',torso+torso_r@[0.,0.,.17],head+head_r@[0.,.015,-.087],torso,torso_r,.032)
    for side,sign in [('l',-1),('r',1)]:
        shoulder=torso+torso_r@[sign*.16,0.,.13]
        hand=d.mocap_pos[ids['caregiver_hand_'+side]]
        elbow=.5*(shoulder+hand)+torso_r@[sign*.06,.045,-.035]
        _segment(env,'upperarm_'+side,shoulder,elbow,torso,torso_r,.040)
        _segment(env,'forearm_'+side,elbow,hand,torso,torso_r,.026)
    env._audio.speech.position=head+head_r@[0.,-.074,-.044]


def pose_record(env, caregiver):
    return dict(visual_contract=VERSION,mouth_opening=caregiver.state['mouth_opening'],
        **{key:caregiver.state[key] for key in ('adult_phase','adult_visibility','interaction_contract') if key in caregiver.state},
        adult_pose={name:dict(position=env.data.mocap_pos[mid].tolist(),
                             quaternion=env.data.mocap_quat[mid].tolist())
                    for name,mid in caregiver.ids.items()})


def upgrade_legacy_snapshot(env, snap):
    """Extend only the nine old noncolliding adult shapes in a saved episode.

    The adult bodies stay last in the model, with the same body and mocap IDs.
    Every infant/object geometry precedes them and is preserved exactly. Any
    other shape change is rejected, rather than treated as a visual migration.
    """
    old=np.asarray(snap['mutable_geometry']['geom_size'])
    if len(old)==env.model.ngeom:return snap,False
    m=env.model;start=int(m.body_geomadr[m.body('caregiver_head').id])
    if len(old)!=start+9 or snap.get('caregiver') is None:
        raise ValueError('unsupported scene geometry change in episode continuation')
    expected=np.array([[.075,.065,.095],[.012,0,0],[.006,0,0],[.012,0,0],[.006,0,0],
                       [.018,.007,.005],[.15,.08,.22],[.04,.025,.012],[.04,.025,.012]])
    if not np.allclose(old[start:],expected):raise ValueError('legacy adult geometry identity mismatch')
    if np.any(snap['mutable_geometry']['geom_contype'][start:]) or np.any(snap['mutable_geometry']['geom_conaffinity'][start:]):
        raise ValueError('cannot migrate colliding caregiver shapes as visual geometry')
    result=copy.deepcopy(snap)
    for name,array in result['mutable_geometry'].items():
        if name.startswith('geom_'):
            native=np.array(getattr(m,name));native[:start]=array[:start]
            result['mutable_geometry'][name]=native
    # New visual kinematics will be populated after mj_forward; original rows
    # retain the saved instant used by the next decoder call.
    for name in ('geom_xpos','geom_xmat'):
        native=np.array(getattr(env.data,name));native[:start]=result['observed_kinematics'][name][:start]
        result['observed_kinematics'][name]=native
    result['caregiver']['visual_contract']=VERSION
    return result,True
