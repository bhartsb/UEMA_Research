"""Posed MIMo teaching animations; no mj_step, policy or training checkpoint.
All timelines are illustrative. Long block fall and authored recovery; no dynamics.
"""
from pathlib import Path
import time,json,math,sys
from scipy.spatial.transform import Rotation,Slerp
import numpy as np
import mujoco
from scipy.optimize import least_squares
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
EV=ROOT/'Saved/h84/block_fall_design_20261027/evidence'
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
        else:r.extend((d.xmat[b].reshape(3,3)-np.array([[0,0,1],[-1,0,0],[0,-1,0]])).ravel()*8)
        r.extend((v-x0)*.003);return r
    rr=least_squares(f,x0,bounds=(lo,hi),max_nfev=75)
    d.qpos[ids]=rr.x;fw()
    return float(np.linalg.norm(d.xpos[b]-target))

# A static scene prop, repurposing a hidden world box in the cached model.
# Operator confirmed feet; each edge is exactly 0.6096 metres.
m.geom_size[1]=[.3048,.3048,.3048]
m.geom_pos[1]=[-.3048,0,.3048]
m.geom_rgba[1]=[.24,.46,.60,1];m.geom_matid[1]=-1

def shape(t,x,angle,hip=0,knee=0,z=None,arms=None):
    pose_root(angle,x,base[2] if z is None else z)
    for side in ['left','right']:
        d.qpos[qi(side+'_hip1')]=hip;d.qpos[qi(side+'_knee')]=knee
        if arms:
            for name,val in arms.items():d.qpos[qi(side+'_'+name)]=val
    fw()
    if z is None:d.qpos[2]+=.0006-min(gaps(bodygs));fw()
    return (t,d.qpos.copy())

keys=[]
keys.append(shape(0,-.10,0,z=base[2]+.6096))
keys.append(shape(1,-.10,0,z=base[2]+.6096))
# Tip around the front of the feet. The following ballistic-looking interval
# is explicitly an authored path; no impact energy/force is inferred from it.
pivot=np.array([-.035,0,.6096]);r=Rotation.from_euler('y',.30)
q=keys[-1][1].copy();q[:3]=pivot+r.apply(q[:3]-pivot);q[3:7]=[math.cos(.15),0,math.sin(.15),0]
keys.append((1.45,q))
keys.append(shape(1.70,.35,.83,z=.78,arms={'shoulder_horizontal':1.2,'elbow':-1.0}))
keys.append(shape(1.95,.68,1.48,z=.28,arms={'shoulder_horizontal':1.4,'elbow':-1.1}))
keys.append(shape(2.15,.76,1.57,arms={'shoulder_horizontal':1.25,'elbow':-1.15}))
keys.append(shape(3.5,.76,1.57,arms={'shoulder_horizontal':1.25,'elbow':-1.15}))
# Tuck knees, raise chest, then place a foot before extending the legs.
keys.append(shape(5.0,.78,1.25,hip=-1.05,knee=-1.55,arms={'shoulder_horizontal':.7,'elbow':-.3}))
keys.append(shape(6.5,.80,.50,hip=-.48,knee=-1.9,arms={'shoulder_horizontal':.5,'elbow':-.6}))
keys.append(shape(7.5,.82,.10,hip=-.10,knee=-1.75,arms={'shoulder_horizontal':.25,'elbow':-.5}))
shape(9,.84,.14,hip=-.12,knee=-1.85,arms={'shoulder_horizontal':.4,'elbow':-.7})
ik('right','foot',np.array([1.04,feet['right'][1],feet['right'][2]]))
d.qpos[2]+=max(0,.0006-min(gaps(bodygs)));fw();keys.append((9,d.qpos.copy()))
shape(10.5,.94,.15,z=base[2]-.12,arms={'shoulder_horizontal':.35,'elbow':-.55})
for s,xx in [('left',1.04),('right',1.04)]:ik(s,'foot',np.array([xx,feet[s][1],feet[s][2]]))
d.qpos[2]+=max(0,.0006-min(gaps(bodygs)));fw();keys.append((10.5,d.qpos.copy()))
keys.append(shape(12,1.035,0));keys.append(shape(14,1.035,0))

# Bring hands ahead of the shoulders, open and approximately palm-down.
# These geometric targets explain support placement; they do not prescribe a policy.
for i,(t,q) in enumerate(keys):
    if 1.7<=t<=6.5:
        d.qpos[:]=q;fw()
        targetz=(q[2]-.20) if t<2.15 else (.023 if t<=3.5 else .013)
        targetx=q[0]+(.30 if t>=5 else .20)
        for side,sg in [('left',1),('right',-1)]:
            ik(side,'hand',np.array([targetx,sg*.16,targetz]))
        d.qpos[2]+=max(0,.0006-min(gaps(bodygs)));fw()
        keys[i]=(t,d.qpos.copy())

renderer=mujoco.Renderer(m,height=640,width=800)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
records=[]
preview='--preview' in sys.argv
indices=sorted(set(round(k[0]*10) for k in keys)) if preview else range(141)
for k in indices:
    if time.monotonic()-start>240:raise RuntimeError('Document-render 240-second bound reached')
    t=k/10;j=next((j for j in range(len(keys)-1) if keys[j+1][0]>=t),len(keys)-2)
    t0,q0=keys[j];t1,q1=keys[j+1];u=(t-t0)/(t1-t0)
    # Linear time during airborne portion, eased body arrangements elsewhere.
    u=u if 1.45<t<2.15 else smooth(u)
    d.qpos[:]=q0*(1-u)+q1*u
    rots=Rotation.from_quat(np.array([q0[3:7],q1[3:7]])[:,[1,2,3,0]])
    quat=Slerp([0,1],rots)([u]).as_quat()[0];d.qpos[3:7]=quat[[3,0,1,2]];fw()
    # Preserve the forward planted foot while the other leg is brought forward.
    if 9<=t<12:
        ik('right','foot',np.array([1.04,feet['right'][1],feet['right'][2]]))
        if t>=10.5:ik('left','foot',np.array([1.04,feet['left'][1],feet['left'][2]]))
    lift=max(0,.0006-min(gaps(bodygs)));d.qpos[2]+=lift;fw()
    if 9<=t<12:
        ik('right','foot',np.array([1.04,feet['right'][1],feet['right'][2]]))
        if t>=10.5:ik('left','foot',np.array([1.04,feet['left'][1],feet['left'][2]]))
    bg=[mujoco.mj_geomDistance(m,d,g,1,2,None) for g in bodygs]
    correction=0.
    while min(bg)<.00019 and correction<.08:
        delta=max(.00001,min(.003,.0002-min(bg)));correction+=delta
        d.qpos[2 if t<1.55 else 0]+=delta;fw()
        bg=[mujoco.mj_geomDistance(m,d,g,1,2,None) for g in bodygs]
    records.append({'t':t,'qpos':d.qpos.tolist(),'floor_clearance':min(gaps(bodygs)),
                    'block_clearance':min(bg),'floor_lift':lift,'block_separation_adjustment':correction,
                    'right_foot_origin':d.xpos[m.body('right_foot').id].tolist(),
                    'left_foot_origin':d.xpos[m.body('left_foot').id].tolist()})
    for view,az,el in [('side',-95,-10),('front',130,-22)]:
        cam=mujoco.MjvCamera();cam.lookat[:]=[.39,0,.66];cam.distance=3.1;cam.azimuth=az;cam.elevation=el
        renderer.update_scene(d,camera=cam)
        im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im)
        dr.rectangle((0,0,800,67),fill='#132e3e')
        dr.text((12,5),'Block fall and getting up | posed movement study',font=font,fill='white')
        dr.text((12,36),f'{t:.1f} / 14.0 s | {view} view | block: 2 ft (0.61 m) per side',font=small,fill='white')
        dr.rectangle((0,603,800,640),fill='#132e3e')
        dr.text((12,613),'Authored poses and timing; no simulated impact forces or learned get-up',font=small,fill='white')
        im.save(OUT/f'block_{view}_{k:03d}.webp',quality=83)
renderer.close()
report={'physics_steps':0,'frames':len(records),'views':2,'duration_s':14,'block_edge_m':.6096,
        'unit_status':'feet confirmed by operator','elapsed_s':time.monotonic()-start,
        'min_floor_gap':min(x['floor_clearance'] for x in records),
        'min_block_gap':min(x['block_clearance'] for x in records),'records':records}
(EV/('preview_geometry.json' if preview else 'block_geometry.json')).write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'}))

