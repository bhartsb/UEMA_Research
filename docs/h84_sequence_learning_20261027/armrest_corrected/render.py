"""Document illustration only: static FK poses, no physics steps or learned model.
Produces two synchronized views and signed object-clearance records for every frame.
"""
from pathlib import Path
import json, time, math
import numpy as np
import mujoco
from scipy.optimize import least_squares, minimize
from PIL import Image, ImageDraw, ImageFont

ROOT=Path.cwd()
OUT=Path(__file__).resolve().parent
WORK=ROOT/'Saved/h84/whole_body_reach_document_20261026'
start=time.monotonic()
m=mujoco.MjModel.from_binary_path(str(WORK/'tmp/model.mjb'))
d=mujoco.MjData(m)
m.vis.global_.offwidth=960; m.vis.global_.offheight=800
# Show only the avatar, floor and a plain ball. All caregiver and old scene items hidden.
for g in range(m.ngeom):
    if not (1 <= m.geom_bodyid[g] <= 56): m.geom_rgba[g,3]=0
    else: m.geom_rgba[g,3]=1
m.geom_rgba[0]=[.80,.85,.88,1]
m.geom_matid[0]=-1
ballg=m.geom('target_geom').id; ballb=m.body('target').id
m.geom_rgba[ballg]=[.93,.21,.10,1]; m.geom_size[ballg,0]=.019
ball=np.array([.365,-.115,.370])
m.geom_rgba[m.geom('geom:right_thdistal1').id]=[.79,.35,.25,1]
digit_colors={'ff':[.12,.46,.90,1],'mf':[.95,.66,.08,1],'rf':[.58,.27,.80,1],'lf':[.05,.64,.54,1],'th':[.91,.36,.65,1]}
for g in range(m.ngeom):
    name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
    for digit,color in digit_colors.items():
        if name.startswith('geom:right_'+digit):
            m.geom_rgba[g]=color;m.geom_matid[g]=-1

def bid(name): return m.body(name).id
def qid(name): return int(m.jnt_qposadr[m.joint('robot:'+name).id])
def forward(): mujoco.mj_forward(m,d)
def setball(p): d.mocap_pos[m.body_mocapid[ballb]]=p
def limits(names):
    ids=[m.joint('robot:'+n).id for n in names]
    return m.jnt_range[ids,0]+1e-7,m.jnt_range[ids,1]-1e-7
def solve_pos(names, body, pos, reference=None):
    ids=[qid(n) for n in names]; lo,hi=limits(names)
    x0=np.clip(d.qpos[ids],lo+1e-6,hi-1e-6)
    def fun(x):
        d.qpos[ids]=x; forward()
        return np.r_[(d.xpos[bid(body)]-pos)*100, (x-x0)*.003]
    r=least_squares(fun,x0,bounds=(lo,hi),max_nfev=90,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    d.qpos[ids]=r.x; forward()

arm=['right_shoulder_horizontal','right_shoulder_ad_ab','right_shoulder_rotation','right_elbow','right_hand1','right_hand2','right_hand3']
left=[x.replace('right','left') for x in arm]
fingers=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)[6:] for j in range(16,38)]
handgeoms=[g for g in range(m.ngeom) if 8<=m.geom_bodyid[g]<=24]
fingergeoms=[g for g in handgeoms if m.geom_bodyid[g]>=9]
digit_geoms={k:[g for g in fingergeoms if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g)or'').startswith('geom:right_'+k)] for k in digit_colors}
selfpairs=[(g,h) for i,a in enumerate(digit_geoms) for b in list(digit_geoms)[i+1:] for g in digit_geoms[a] for h in digit_geoms[b]]
def selfdist():return np.array([mujoco.mj_geomDistance(m,d,g,h,1,None) for g,h in selfpairs])
tips=[m.geom('geom:right_'+n+'distal1').id for n in ['ff','mf','rf','lf','th']]
def distances(gs=handgeoms):
    return np.array([mujoco.mj_geomDistance(m,d,g,ballg,2,None) for g in gs])

def align_gaze():
    names=['head_swivel','head_tilt','left_eye_horizontal','left_eye_vertical','right_eye_horizontal','right_eye_vertical']
    qi=[qid(n) for n in names];lo,hi=limits(names);prior=d.qpos[qi].copy()
    def fun(x):
        d.qpos[qi]=x;forward();rs=[]
        for cam in [1,4]:
            toward=ball-d.cam_xpos[cam];toward/=np.linalg.norm(toward)
            rs.extend((-d.cam_xmat[cam].reshape(3,3)[:,2]-toward)*10)
        return np.r_[rs,(x-prior)*.0001]
    rr=least_squares(fun,np.clip(prior,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=35)
    d.qpos[qi]=rr.x;forward()
    return [float(np.degrees(np.arccos(np.clip(np.dot(-d.cam_xmat[c].reshape(3,3)[:,2],(ball-d.cam_xpos[c])/np.linalg.norm(ball-d.cam_xpos[c])),-1,1)))) for c in [1,4]]

font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
renderer=mujoco.Renderer(m,height=640,width=800)
records=[]
def extra_box(scene,pos,size,color):
    g=scene.geoms[scene.ngeom]; mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array(size,float),np.array(pos,float),np.eye(3).ravel(),np.array(color,np.float32));scene.ngeom+=1
def render(cam,idx,t,close=False):
    renderer.update_scene(d,camera=cam)
    # Low seat, side support and a small object stand stay visible in both views.
    extra_box(renderer.scene,[-.025,0,.065],[.13,.14,.065],[.35,.46,.54,1])
    extra_box(renderer.scene,[.08,.18,.125],[.085,.045,.125],[.35,.46,.54,1])
    extra_box(renderer.scene,[.365,-.115,.1755],[.003,.003,.1755],[.65,.71,.73,1])
    extra_box(renderer.scene,[.365,-.115,.012],[.06,.06,.012],[.65,.71,.73,1])
    im=Image.fromarray(renderer.render())
    dr=ImageDraw.Draw(im);dr.rectangle((0,0,800,52),fill='#112837')
    dr.text((18,9),f'{t:.2f} s  |  '+('Hand close-up' if close else ('Side view' if idx=='side' else 'Front view')),font=font,fill='white')
    dr.rectangle((0,606,800,640),fill='#112837');dr.text((18,612),'STAGED MIMo POSE for evaluating model steps',font=small,fill='white')
    return im


EVIDENCE=ROOT/'Saved/h84/sequence_learning_document_20261027/evidence/armrest_corrected'
source=json.loads((WORK/'evidence/pose_geometry.json').read_text())
left_geoms=[g for g in range(m.ngeom) if 27<=m.geom_bodyid[g]<=43]
arm_geoms=[g for g in range(m.ngeom) if 25<=m.geom_bodyid[g]<=43]
types={2:'sphere',3:'capsule',4:'ellipsoid',5:'cylinder',6:'box'}
xml='<mujoco><worldbody><geom name="armrest" type="box" pos=".08 .18 .125" size=".085 .045 .125"/>'
for g in arm_geoms:
 xml+='<geom name="g%d" type="%s" size="%s"/>'%(g,types[int(m.geom_type[g])],' '.join(map(str,m.geom_size[g])))
xml+='</worldbody></mujoco>'
cm=mujoco.MjModel.from_xml_string(xml);cd=mujoco.MjData(cm);mujoco.mj_forward(cm,cd)
def clearance():
 for j,g in enumerate(arm_geoms,1):
  cd.geom_xpos[j]=d.geom_xpos[g]
  cd.geom_xmat[j]=d.geom_xmat[g]
 return {g:float(mujoco.mj_geomDistance(cm,cd,0,j,1,None)) for j,g in enumerate(arm_geoms,1)}
support_names=left+['left_th_swivel','left_th_adduction','left_th_pivot','left_th_middle','left_th_distal']
li=[qid(n) for n in support_names];lower,upper=limits(support_names)
def bottom(g):
 Rg=d.geom_xmat[g].reshape(3,3);sz=m.geom_size[g];kind=int(m.geom_type[g])
 if kind==6:extent=np.dot(np.abs(Rg[2,:]),sz)
 elif kind==3:extent=sz[0]+sz[1]*abs(Rg[2,2])
 elif kind==5:extent=sz[1]*abs(Rg[2,2])+sz[0]*np.sqrt(max(0,1-Rg[2,2]**2))
 else:raise ValueError(kind)
 return d.geom_xpos[g,2]-extent
palm=m.geom('geom:left_hand1').id
rows=[];prior=None
for rec in source['frames']:
 k=rec['frame'];t=rec['time_s'];original=np.array(rec['qpos']);d.qpos[:]=original;setball(ball);forward()
 before=clearance();P=d.xpos[bid('left_hand')].copy()
 initial=original[li].copy() if prior is None else prior.copy()
 lb=lower.copy();ub=upper.copy()
 if prior is not None:lb=np.maximum(lb,prior-.20);ub=np.minimum(ub,prior+.20)
 def support_residual(x):
  d.qpos[li]=x;forward()
  normal=d.xmat[bid('left_hand')].reshape(3,3)[:,1]
  gaps_plane=np.array(list(clearance().values()))
  return np.r_[(normal-[0,0,-1])*1000,(bottom(palm)-.250-.0007)*10000,
      np.minimum(gaps_plane-.0002,0)*10000,
      (d.xpos[bid('left_hand')][:2]-[.065,.175])*10,(x-initial)*.1]
 sol=least_squares(support_residual,np.clip(initial,lb+1e-8,ub-1e-8),bounds=(lb,ub),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
 if prior is None:
  rng=np.random.default_rng(42)
  for attempt in range(4):
   trial=least_squares(support_residual,rng.uniform(lb,ub),bounds=(lb,ub),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   if trial.cost<sol.cost:sol=trial
  print('initial contact solve cost',sol.cost,flush=True)
 d.qpos[li]=sol.x;forward();prior=sol.x.copy()
 assert 0<=bottom(palm)-.250<.0012,(k,'palm hovering',bottom(palm)-.250)
 gaps=clearance();minimum=min(gaps.values());assert minimum>=.0001,(k,minimum)
 assert np.max(np.abs(np.delete(d.qpos-original,li)))==0
 ballgap=float(distances().min());assert ballgap>=-1e-6
 current=d.xpos[bid('left_hand')].copy()
 rows.append({'frame':k,'time_s':t,'before_min_armrest_m':min(before.values()),'after_min_armrest_m':minimum,'minimum_left_hand_armrest_m':min(gaps[g] for g in left_geoms),'wrist_vertical_shift_m':float(current[2]-P[2]),'wrist_position':current.tolist(),'palm_orientation_error':float(np.linalg.norm(d.xmat[bid('left_hand')].reshape(3,3)[:,1]-[0,0,-1])),'qpos':d.qpos.tolist(),'right_hand_ball_min_m':ballgap,'left_geom_clearances_m':{mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g):v for g,v in gaps.items()}})
 for label,az,el in [('side',-95,-12),('front',155,-17)]:
  cam=mujoco.MjvCamera();cam.lookat[:]=[.14,0,.35];cam.distance=1.23;cam.azimuth=az;cam.elevation=el
  render(cam,label,t).save(OUT/f'{label}_{k:03d}.webp',quality=88)
 if k==60:
  for label,az,el in [('palm',-25,15),('thumb',-135,15)]:
   cam=mujoco.MjvCamera();cam.lookat[:]=ball;cam.distance=.21;cam.azimuth=az;cam.elevation=el
   render(cam,label,t,True).save(OUT/f'{label}_{k:03d}.webp',quality=95)
renderer.close()
report={'purpose':'Correct supporting-hand armrest overlap in illustration only','physics_steps':0,'frames':rows,'minimum_before_m':min(x['before_min_armrest_m'] for x in rows),'minimum_after_m':min(x['after_min_armrest_m'] for x in rows),'max_wrist_shift_m':max(x['wrist_vertical_shift_m'] for x in rows),'elapsed_s':time.monotonic()-start,'armrest_height_before_m':.232,'armrest_height_after_m':.250,'unchanged':'All joints except left arm and left thumb, grasping hand, ball, cameras and timing','source':'Saved/h84/whole_body_reach_document_20261026/evidence/pose_geometry.json'}
(EVIDENCE/'clearance.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='frames'}))
