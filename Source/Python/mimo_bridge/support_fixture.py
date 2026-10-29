# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Optional anchored rail for copied posture/support experiments only.

It is ordinary visible colliding geometry. Placement uses audited body dimensions
before an episode, never a target coordinate supplied to the learner.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


NAME='developmental_support'
RAIL='developmental_support_rail'


def add_to_scene(path):
    tree=ET.parse(path);world=tree.getroot().find('worldbody')
    if world is None:raise ValueError('Support fixture requires a worldbody')
    if world.find(f"body[@name='{NAME}']") is not None:raise ValueError('Duplicate support fixture')
    body=ET.SubElement(world,'body',name=NAME,mocap='true',pos='5 0 1')
    common=dict(type='capsule',rgba='.16 .44 .65 1',friction='1 .01 .001',condim='6',contype='1',conaffinity='1')
    ET.SubElement(body,'geom',name=RAIL,fromto='-.2 0 0 .2 0 0',size='.01',**common)
    for side,x in (('left',-.2),('right',.2)):
        ET.SubElement(body,'geom',name=NAME+'_'+side+'_post',pos=f'{x} 0 -.2',size='.015 .2',**common)
    tree.write(path,encoding='utf-8',xml_declaration=True)


def place(env,height_fraction=.65):
    """Set rail near the shoulders; bounds are geometric hypotheses, not a skill gate."""
    if not .35<=height_fraction<=.85:raise ValueError('Unsupported body-relative rail height')
    model,data=env.model,env.data
    positions=lambda name:data.xpos[model.body(name).id].copy()
    shoulders=np.stack([positions(s+'_upper_arm') for s in ('left','right')])
    feet=np.stack([positions(s+'_foot') for s in ('left','right')])
    along=data.xmat[model.body('chest').id].reshape(3,3)@env._body_orientation_axes[0]
    shoulder_span=float(np.mean(shoulders@along)-np.min(feet@along))
    lengths=[np.linalg.norm(positions(s+'_upper_arm')-positions(s+'_lower_arm'))+
             np.linalg.norm(positions(s+'_lower_arm')-positions(s+'_hand')) for s in ('left','right')]
    hand_length=np.linalg.norm(positions('left_ffdistal')-positions('left_hand'))
    half_span=max(.12,float(np.linalg.norm(shoulders[0]-shoulders[1]))*.85)
    radius=float(np.clip(hand_length*.12,.008,.016))
    height=float(max(.15,height_fraction*shoulder_span))
    horizontal=along.copy();horizontal[2]=0
    if np.linalg.norm(horizontal)<1e-6:raise ValueError('Place support from the declared lying calibration pose')
    horizontal/=np.linalg.norm(horizontal)
    lateral=np.cross(horizontal,[0.,0.,1.]);lateral/=np.linalg.norm(lateral)
    center=shoulders.mean(0)+horizontal*float(np.mean(lengths))*.45;center[2]=height
    body=model.body(NAME).id;mid=int(model.body_mocapid[body]);yaw=float(np.arctan2(lateral[1],lateral[0]))
    data.mocap_pos[mid]=center;data.mocap_quat[mid]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    rail=model.geom(RAIL).id;model.geom_size[rail,:2]=[radius,half_span]
    model.geom_rbound[rail]=radius+half_span
    # Rail's compiled capsule axis is local x. Posts are vertical capsules.
    for side,sign in (('left',-1),('right',1)):
        geom=model.geom(NAME+'_'+side+'_post').id
        model.geom_pos[geom]=[sign*half_span,0,-height/2]
        model.geom_size[geom,:2]=[radius*1.5,height/2]
        model.geom_rbound[geom]=radius*1.5+height/2
    env._mj.mj_forward(model,data)
    return dict(contract='body_relative_support_v1',rail_geom=RAIL,body_name=NAME,
        center_xyz_m=center.tolist(),half_span_m=half_span,radius_m=radius,height_m=height,
        shoulder_to_foot_span_m=shoulder_span,arm_lengths_m=[float(x) for x in lengths],
        hand_length_m=float(hand_length),height_fraction=height_fraction,
        anchored=True,grasp_reach_verified=False,body_lift_verified=False,
        scope='Visible anchored support. Dimensions alone do not prove reaching, gripping or pulling to stand.')
