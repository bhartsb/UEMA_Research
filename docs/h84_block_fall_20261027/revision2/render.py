"""Posed MIMo teaching animations; no mj_step, policy or training checkpoint.
All timelines are illustrative. Long block fall and authored recovery; no dynamics.
"""
from pathlib import Path
import time,json,math,sys
from scipy.spatial.transform import Rotation,Slerp
from scipy.interpolate import PchipInterpolator
import argparse
import numpy as np
import mujoco
from scipy.optimize import least_squares
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
EV=ROOT/'Saved/h84/block_fall_design_20261027/evidence/revision2'
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
def below(body,ancestor):
    while body:
        if body==ancestor:return True
        body=int(m.body_parentid[body])
    return False
handgs={s:[g for g in bodygs if below(int(m.geom_bodyid[g]),m.body(s+'_lower_arm').id)] for s in ['left','right']}
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
ik_seeds={};current_t=0
def ik(side,part,target,targetR=None):
    names=([side+'_'+n for n in ['hip1','hip2','hip3','knee','foot1','foot2','foot3']] if part=='foot' else [side+'_'+n for n in ['shoulder_horizontal','shoulder_ad_ab','shoulder_rotation','elbow','hand1','hand2','hand3']])
    ids=[qi(n) for n in names];js=[m.joint('robot:'+n).id for n in names]
    lo=m.jnt_range[js,0]+1e-6;hi=m.jnt_range[js,1]-1e-6
    if part=='foot':
        for ix in [1,2,5,6]:lo[ix]=-.025;hi[ix]=.025
    x0=np.clip(d.qpos[ids],lo+1e-7,hi-1e-7)
    previous=ik_seeds.get((side,part))
    if previous is not None and 0<current_t-previous[0]<.05:
        x0=np.clip(previous[1],lo+1e-7,hi-1e-7)
    if part=='hand':
        # Bound authored arm/wrist changes to avoid jumping between IK branches.
        speed=.4 if 1.6<current_t<2.4 else .2
        lo=np.maximum(lo,x0-speed);hi=np.minimum(hi,x0+speed)
    b=m.body(side+'_'+part).id
    def f(v):
        d.qpos[ids]=v;fw();r=list((d.xpos[b]-target)*100)
        if part=='foot':r.extend((d.xmat[b].reshape(3,3)-(feetR[side] if targetR is None else targetR)).ravel()*15)
        elif targetR is not False:r.extend((d.xmat[b].reshape(3,3)-np.array([[0,0,1],[-1,0,0],[0,-1,0]])).ravel()*3*(1 if targetR is None else targetR))
        if part=='foot':
            for g in footgs:
                if m.body(m.geom_bodyid[g]).name.startswith(side):
                    r.append(max(0,.001-mujoco.mj_geomDistance(m,d,g,0,2,None))*1000)
        elif target[2]<.30:
            for g in handgs[side]:r.append(max(0,.001-mujoco.mj_geomDistance(m,d,g,0,2,None))*1000)
        r.extend((v-x0)*(.4 if part=='hand' else .003));return r
    rr=least_squares(f,x0,bounds=(lo,hi),max_nfev=65)
    d.qpos[ids]=rr.x;fw()
    ik_seeds[(side,part)]=(current_t,rr.x.copy())
    return float(np.linalg.norm(d.xpos[b]-target))

parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--edge',type=float,default=.2286);args=parser.parse_args()
EDGE=args.edge;FPS=30;DURATION=12
m.geom_size[1]=[EDGE/2]*3;m.geom_pos[1]=[-EDGE/2,0,EDGE/2];m.geom_matid[1]=-1;m.geom_rgba[1]=[.38,.55,.64,1]
m.geom_rgba[0]=[.88,.90,.90,1]

def rotation_y(a):return Rotation.from_euler('y',a).as_matrix()
def qroot(x,z,pitch):
    d.qpos[:]=base;d.qpos[:3]=[x,0,z];d.qpos[3:7]=[math.cos(pitch/2),0,math.sin(pitch/2),0]
def curve(ts,vs,t):return float(PchipInterpolator(ts,vs)(t))
def blendR(a,b,u):return Slerp([0,1],Rotation.from_matrix(np.array([a,b])))([np.clip(u,0,1)]).as_matrix()[0]
def setj(side,vals):
    for n,v in vals.items():d.qpos[qi(side+'_'+n)]=v
def anchor_knee(x=.43,z=.0365):
    fw();b=m.body('left_lower_leg').id
    d.qpos[0]+=x-d.xpos[b,0];d.qpos[2]+=z-d.xpos[b,2];fw()

def fit_leg_clearance(side):
    """Alter the authored limb angles, never translate the body to hide contact."""
    names=[side+'_'+n for n in ['hip1','knee','foot1']]
    ids=[qi(n) for n in names];js=[m.joint('robot:'+n).id for n in names]
    gs=[g for g in bodygs if m.body(m.geom_bodyid[g]).name in [side+'_'+n for n in ['upper_leg','lower_leg','foot','big_toe','toes']]]
    if all(mujoco.mj_geomDistance(m,d,g,o,2,None)>=.0005 for g in gs for o in [0,1]):return
    target=d.qpos[ids].copy()
    def residual(v):
        d.qpos[ids]=v;fw()
        return list((v-target)*.2)+[max(0,.001-mujoco.mj_geomDistance(m,d,g,o,2,None))*2000 for g in gs for o in [0,1]]
    rr=least_squares(residual,target,bounds=(m.jnt_range[js,0]+1e-6,m.jnt_range[js,1]-1e-6),max_nfev=50)
    if np.max(np.abs(residual(rr.x)[3:]),initial=0)>.2:
        for offset in [-.2,-.45]:
            seed=target.copy();seed[0]+=offset
            seed=np.clip(seed,m.jnt_range[js,0]+1e-5,m.jnt_range[js,1]-1e-5)
            other=least_squares(residual,seed,bounds=(m.jnt_range[js,0]+1e-6,m.jnt_range[js,1]-1e-6),max_nfev=80)
            if np.dot(other.fun,other.fun)<np.dot(rr.fun,rr.fun):rr=other
    d.qpos[ids]=rr.x;fw()

# A planted toe governs the initial tip; free flight uses a gravity term.
TAKEOFF=1.6;ANGLE=.60;OMEGA=2*ANGLE/.95
pivot=np.array([0,0,EDGE]);v=np.array([-.065,0,base[2]])
takepos=pivot+rotation_y(ANGLE)@v
takevel=np.array([OMEGA*(-v[0]*math.sin(ANGLE)+v[2]*math.cos(ANGLE)),0,OMEGA*(-v[0]*math.cos(ANGLE)-v[2]*math.sin(ANGLE))])
airtime=(takevel[2]+math.sqrt(takevel[2]**2+2*9.81*(takepos[2]-.20)))/9.81
LAND=TAKEOFF+airtime;LX=takepos[0]+takevel[0]*airtime
HX=LX+.30;KX=LX+.04;FX=KX+.24
print(json.dumps({'edge_m':EDGE,'avatar_height_m':.8266555,'takeoff':TAKEOFF,'landing':LAND,'fall_seconds':airtime,'landing_root_x':LX}),flush=True)

def frame(t):
    global current_t
    current_t=t
    errs=[];anchors=[]
    if t<=TAKEOFF:
        # Accelerate into the tip rather than stopping at the departure pose.
        u=np.clip((t-.65)/.95,0,1);pitch=ANGLE*u*u
        qroot(0,0,pitch);d.qpos[:3]=pivot+rotation_y(pitch)@v
        for s in ['left','right']:
            setj(s,{'elbow':-.25-.65*u,'shoulder_horizontal':-.10*u})
        fw()
    elif t<LAND:
        u=(t-TAKEOFF)/airtime;pitch=ANGLE+(1.35-ANGLE)*u
        qroot(0,0,pitch);d.qpos[:3]=takepos+takevel*(t-TAKEOFF)+np.array([0,0,-.5*9.81*(t-TAKEOFF)**2])
        for s in ['left','right']:setj(s,{'hip1':-1.7*smooth(u),'knee':-2.3*smooth(u),'foot1':-.95*smooth(u),'elbow':-1.0})
        fw()
    else:
        # One continuous posture path; no repeated rest at every illustration key.
        ts=[LAND,LAND+.30,LAND+.65,3.5,4.3,5.5,6.6,7.3,8.5,10.3,12]
        pitch=curve(ts,[1.35,1.45,1.43,1.40,1.15,.16,.14,.35,.22,.03,0],t)
        x=curve(ts,[LX,LX+.12,LX+.14,KX,KX,KX,KX,KX+.05,FX-.07,FX-.01,FX-.01],t)
        z=curve(ts,[.20,.12,.15,.197,.19,.19,.19,.18,.28,base[2]-.002,base[2]-.002],t)
        hip=curve(ts,[-1.7,-.30,-.35,-1.40,-1.15,-.16,-.14,0,-.8,-.06,0],t)
        knee=curve(ts,[-2.3,-1.0,-1.0,-1.7,-1.7,-1.7,-1.7,-1.7,-1.2,-.08,-.04],t)
        ankle=curve(ts,[-.95,-.4,-.5,-.65,-.65,-.65,-.65,-.65,.2,0,0],t)
        qroot(x,z,pitch)
        for s in ['left','right']:setj(s,{'hip1':hip,'knee':knee,'foot1':ankle,'elbow':-.6})
        # Knees and shins stay where support was established. No whole-body
        # floor projection is permitted in this revision.
        if 3.5<=t<=7.3:
            sitback=.6*math.sin(math.pi*np.clip((t-3.5)/2,0,1)) if t<5.5 else 0
            setj('left',{'hip1':(-pitch-sitback if t<6.6 else -pitch+.35*smooth((t-6.6)/.7)),'knee':-1.7-sitback,'foot1':-.65})
            if t<5.5:setj('right',{'hip1':-pitch-sitback,'knee':-1.7-sitback,'foot1':-.65})
            anchor_knee(KX)
            anchors.append('left knee')
        fw()
        if 5.5<=t<6.6:
            # Lift the front foot clear before bringing it over the floor.
            u=smooth((t-5.5)/1.1)
            startfoot=np.array([KX-.190575,feet['right'][1],.0714218])
            dest=np.array([FX,feet['right'][1],feet['right'][2]])
            target=startfoot*(1-u)+dest*u;target[2]+=.022*math.sin(math.pi*u)
            R=blendR(rotation_y(2.35)@feetR['right'],feetR['right'],min(1,u*2.3))
            errs.append(ik('right','foot',target,R))
        elif t>=6.6:
            errs.append(ik('right','foot',np.array([FX,feet['right'][1],feet['right'][2]])))
            anchors.append('right foot')
        if 7.3<t<8.5:
            u=smooth((t-7.3)/1.2);startfoot=np.array([KX-.175,feet['left'][1],.1273]);dest=np.array([FX-.04,feet['left'][1],feet['left'][2]])
            target=startfoot*(1-u)+dest*u;target[2]+=.04*math.sin(math.pi*u)
            R=blendR(rotation_y(2.7)@feetR['left'],feetR['left'],u)
            errs.append(ik('left','foot',target,R))
        elif t>=8.5:
            errs.append(ik('left','foot',np.array([FX-.04,feet['left'][1],feet['left'][2]+.001])))
            anchors.append('left foot')
    # Hands approach together, absorb the pictured fall with elbow bend, then
    # remain anchored while the knees come underneath. Each sample solves them.
    fw()
    if 1.1<t<4.9:
        for s,sg,offset in [('left',1,.025),('right',-1,0)]:
            target=np.array([HX+offset,sg*.145,.012])
            if t<LAND:
                u=smooth((t-1.1)/(LAND-1.1));current=d.xpos[m.body(s+'_hand').id].copy();shoulder=d.xpos[m.body(s+'_upper_arm').id].copy();vec=target-shoulder;reachable=shoulder+vec*min(1,.225/np.linalg.norm(vec));target=current*(1-u)+reachable*u
            elif t>3.9:
                u=smooth((t-3.9)/1.0);target=target*(1-u)+np.array([KX+.12,sg*.14,.29])*u
            errs.append(ik(s,'hand',target,float(smooth((t-1.1)/(LAND-1.1))) if t<LAND else None))
            if LAND<=t<=3.9:anchors.append(s+' hand')
    elif t>=4.9:
        for s,sg in [('left',1),('right',-1)]:
            target=np.array([KX+.12,sg*.14,.29])
            if t<5.9:errs.append(ik(s,'hand',target))
    if t>=4.5:
        u=smooth((t-4.5)/1.4)
        for s in ['left','right']:
            for name in ['shoulder_horizontal','shoulder_ad_ab','shoulder_rotation','elbow','hand1','hand2','hand3']:
                ix=qi(s+'_'+name);d.qpos[ix]=d.qpos[ix]*(1-u)+base[ix]*u
    if TAKEOFF<t<3.5:
        for s in ['left','right']:fit_leg_clearance(s)
    elif 7.3<t<8.5:
        fit_leg_clearance('left')
    d.qpos[qi('head_tilt')]=.12*smooth((t-.7)/.8)*(1-smooth((t-9)/1.5));fw()
    return errs,anchors

renderer=mujoco.Renderer(m,height=640,width=800);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
indices=[0,30,48,round(LAND*30),round((LAND+.3)*30),75,90,105,120,135,150,165,180,198,210,219,234,255,270,309,360] if args.preview else range(361)
records=[]
for k in indices:
    if time.monotonic()-start>240:raise RuntimeError('Render bound240s')
    t=k/FPS;errs,anchors=frame(t)
    ground=gaps(bodygs);block=[mujoco.mj_geomDistance(m,d,g,1,2,None) for g in bodygs]
    rec={'time':t,'qpos':d.qpos.tolist(),'floor_min':min(ground),'block_min':min(block),'ik_error':max(errs,default=0),'anchors':anchors,'root':d.qpos[:3].tolist()}
    for s in ['left','right']:
        for part in ['hand','foot','lower_leg']:rec[s+'_'+part]=d.xpos[m.body(s+'_'+part).id].tolist()
    records.append(rec)
    for view,az,el in [('side',-95,-10),('front',125,-18)]:
        cam=mujoco.MjvCamera();cam.lookat[:]=[.29,0,.46];cam.distance=2.35;cam.azimuth=az;cam.elevation=el
        renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im)
        dr.rectangle((0,0,800,57),fill='#163442');dr.text((14,5),'Fall and recovery | continuous contact study',font=font,fill='white')
        dr.text((14,31),f'{t:4.2f} / 12.00 s | {view} | block {EDGE/.3048:g} ft per side | original MIMo scale',font=small,fill='white')
        dr.rectangle((0,611,800,640),fill='#163442');dr.text((14,618),'Authored movement illustration; not a trained recovery or contact-force result',font=small,fill='white')
        im.save(OUT/f'block_{view}_{k:03d}.webp',quality=85)
renderer.close()
result={'fps':FPS,'frames':len(records),'duration':DURATION,'edge':EDGE,'avatar_height':.8266555,'landing':LAND,'elapsed':time.monotonic()-start,'records':records,'max_ik_error':max(x['ik_error'] for x in records),'min_floor':min(x['floor_min'] for x in records),'min_block':min(x['block_min'] for x in records)}
(EV/('preview.json' if args.preview else 'motion.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))

