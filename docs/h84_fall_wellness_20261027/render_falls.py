"""Posed MIMo teaching animations; no mj_step, policy or training checkpoint.
All timelines are illustrative. Stop uncorrected fall at first surface contact.
"""
from pathlib import Path
import time,json,math
import numpy as np
import mujoco
from scipy.optimize import least_squares
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
EV=ROOT/'Saved/h84/fall_wellness_design_20261027/evidence'
start=time.monotonic()
m=mujoco.MjModel.from_binary_path(str(ROOT/'Saved/h84/whole_body_reach_document_20261026/tmp/model.mjb'))
d=mujoco.MjData(m)
for g in range(m.ngeom):
    m.geom_rgba[g,3]=1 if 1<=m.geom_bodyid[g]<=56 else 0
m.geom_rgba[0]=[.78,.84,.87,1];m.geom_matid[0]=-1
m.vis.global_.offwidth=800;m.vis.global_.offheight=640
bodygs=[g for g in range(m.ngeom) if 1<=m.geom_bodyid[g]<=56]
footgs=[g for g in bodygs if m.geom_bodyid[g] in [49,50,51,54,55,56]]
othergs=[g for g in bodygs if g not in footgs]
def qi(n):return int(m.jnt_qposadr[m.joint('robot:'+n).id])
def fw():mujoco.mj_forward(m,d)
def gaps(gs):return [mujoco.mj_geomDistance(m,d,g,0,2,None) for g in gs]
base=d.qpos.copy();base[:3]=[0,0,.4];base[3:7]=[1,0,0,0];base[7:100]=0
base[qi('right_elbow')]=-.25;base[qi('left_elbow')]=-.25
d.qpos[:]=base;fw();base[2]+=.0005-min(gaps(footgs));d.qpos[:]=base;fw()
feet={s:d.xpos[m.body(s+'_foot').id].copy() for s in ['right','left']}
feetR={s:d.xmat[m.body(s+'_foot').id].reshape(3,3).copy() for s in ['right','left']}
def smooth(u):u=np.clip(u,0,1);return u*u*(3-2*u)
def pose_root(angle,x=0,z=None):
    d.qpos[:]=base;d.qpos[3:7]=[math.cos(angle/2),0,math.sin(angle/2),0]
    if z is None:
        pivot=np.array([.065,0,0]);v=base[:3]-pivot
        R=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
        d.qpos[:3]=pivot+R@v
    else:d.qpos[:3]=[x,0,z]
    fw()
    if z is None:
        d.qpos[2]+=max(0,.0005-min(gaps(footgs)));fw()
def ik(side,part,target):
    names=([side+'_'+n for n in ['hip1','hip2','hip3','knee','foot1','foot2','foot3']] if part=='foot' else [side+'_'+n for n in ['shoulder_horizontal','shoulder_ad_ab','shoulder_rotation','elbow','hand1','hand2','hand3']])
    ids=[qi(n) for n in names];js=[m.joint('robot:'+n).id for n in names]
    lo=m.jnt_range[js,0]+1e-6;hi=m.jnt_range[js,1]-1e-6
    if part=='foot':
        d.qpos[qi(side+'_knee')]=-.65;d.qpos[qi(side+'_hip1')]=-.3
    if part=='foot':
        for ix in [1,2,5,6]:lo[ix]=-.025;hi[ix]=.025
    x0=np.clip(d.qpos[ids],lo+1e-7,hi-1e-7)
    b=m.body(side+'_'+part).id
    def f(v):
        d.qpos[ids]=v;fw();r=list((d.xpos[b]-target)*100)
        if part=='foot':r.extend((d.xmat[b].reshape(3,3)-feetR[side]).ravel()*15)
        r.extend((v-x0)*.003);return r
    rr=least_squares(f,x0,bounds=(lo,hi),max_nfev=75)
    d.qpos[ids]=rr.x;fw()
    return float(np.linalg.norm(d.xpos[b]-target))

# Find a collision-free ending for the uncorrected illustration.
lo=0.;hi=1.6
for _ in range(40):
    a=(lo+hi)/2;pose_root(a)
    if min(gaps(othergs))<.0005:hi=a
    else:lo=a
contact_angle=lo
titles={'early':'Early correction: step to widen support','late':'No timely correction: first body contact','descent':'Controlled lowering: downward motion can be useful','brace':'Late response: fingertips approach the floor'}
renderer=mujoco.Renderer(m,height=640,width=800)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
records={}
for case in titles:
    records[case]=[]
    for k in range(61):
        if time.monotonic()-start>240:raise RuntimeError('Rendering240s limit reached')
        t=k*.05;errs=[]
        if case=='late':
            theta=min(contact_angle, .14*smooth((t-.35)/.25)+3.4*max(0,t-.6)**2)
            pose_root(theta)
        elif case=='brace':
            theta=min(1.30,.14*smooth((t-.35)/.25)+3.4*max(0,t-.6)**2)
            pose_root(theta)
            if t>.65:
                u=smooth((t-.65)/.55)
                for s,sg in [('right',-1),('left',1)]:
                    current=d.xpos[m.body(s+'_hand').id].copy()
                    errs.append(ik(s,'hand',current*(1-u)+np.array([.70,sg*.14,.095])*u))
        elif case=='early':
            tilt=.17*smooth((t-.35)/.25)*(1-smooth((t-.7)/.65));shift=.10*smooth((t-.65)/.65)
            pose_root(tilt,x=shift,z=base[2]-.035*smooth((t-.35)/.4))
            for s,sg in [('right',-1),('left',1)]:
                target=feet[s].copy()
                if s=='right':
                    u=np.clip((t-.65)/.55,0,1);target[0]+=.20*smooth(u);target[2]+=.055*math.sin(math.pi*u)
                errs.append(ik(s,'foot',target))
                if t>.45:errs.append(ik(s,'hand',np.array([shift+.12,sg*.16,.47])))
        else:
            u=smooth((t-.35)/1.0);pose_root(.08*u,x=-.045*u,z=base[2]-.09*u)
            for s,sg in [('right',-1),('left',1)]:
                errs.append(ik(s,'foot',feet[s]));errs.append(ik(s,'hand',np.array([.12,sg*.14,.43-.10*u])))
        # Posed rendering must not visibly intersect floor. Record any geometric offset.
        gap=min(gaps(bodygs));lift=max(0,.0003-gap);d.qpos[2]+=lift;fw()
        groundgaps=gaps(bodygs);closest=bodygs[int(np.argmin(groundgaps))]
        rec={'time_s':round(t,2),'qpos':d.qpos.tolist(),'head':d.xpos[m.body('head').id].tolist(),'root':d.qpos[:3].tolist(),'min_floor_gap_m':min(groundgaps),'geometric_floor_lift_m':float(lift),'max_ik_residual_m':max(errs,default=0),'closest_geom':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,closest),'right_hand':d.xpos[m.body('right_hand').id].tolist()}
        records[case].append(rec)
        for view,az,el in [('side',-95,-12),('front',155,-22)]:
            cam=mujoco.MjvCamera();cam.lookat[:]=[.32,0,.30];cam.distance=1.9;cam.azimuth=az;cam.elevation=el
            renderer.update_scene(d,camera=cam)
            im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im)
            dr.rectangle((0,0,800,70),fill='#132e3e');dr.text((14,6),titles[case],font=font,fill='white');dr.text((14,37),f'{t:.2f} s | {view} view | illustrative timing',font=small,fill='white')
            dr.rectangle((0,600,800,640),fill='#132e3e');dr.text((14,610),'POSED EXPLANATION | No physics rollout or learned recovery | No caregiver',font=small,fill='white')
            im.save(OUT/f'{case}_{view}_{k:03d}.webp',quality=83)
renderer.close()
result={'purpose':'Kinematic explanation, no dynamics, forces or learned policy','physics_steps':0,'frames_per_case':61,'views':2,'contact_angle_radians':contact_angle,'elapsed_seconds':time.monotonic()-start,'records':records,'floor_gap_min_m':min(r['min_floor_gap_m'] for rs in records.values() for r in rs),'max_geometric_lift_m':max(r['geometric_floor_lift_m'] for rs in records.values() for r in rs),'max_ik_residual_m':max(r['max_ik_residual_m'] for rs in records.values() for r in rs)}
(EV/'animation_geometry.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
