"""Document illustration only: static FK poses, no physics steps or learned model.
Produces two synchronized views and signed object-clearance records for every frame.
"""
from pathlib import Path
import json, time, math
import numpy as np
import mujoco
from scipy.optimize import least_squares, minimize
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
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

base=d.qpos.copy(); base[:3]=[0,0,.215]; base[3:7]=[1,0,0,0]
base[7:100]=0
for side in ['right','left']:
    base[qid(side+'_hip1')]=-1.5; base[qid(side+'_knee')]=-1.35
base[qid('right_elbow')]=-1.0; base[qid('left_elbow')]=-1.0

def bodypose(lean, wrist):
    d.qpos[:]=base
    # Hip stays on the seat. Rotation moves the upper body toward the object.
    d.qpos[3:7]=[math.cos(lean/2),0,math.sin(lean/2),0]
    for side,sgn in [('right',-1),('left',1)]:
        names=[side+'_'+n for n in ['hip1','hip2','hip3','knee','foot1','foot2','foot3']]
        solve_pos(names,side+'_foot',np.array([.205,sgn*.095,.035]))
    solve_pos(left,'left_hand',np.array([.065,.175,.24]))
    lids=[qid(n) for n in left]; llo,lhi=limits(left)
    desired=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    def support(x):
        d.qpos[lids]=x; forward()
        return np.r_[(d.xpos[bid('left_hand')]-[.065,.175,.24])*80,(d.xmat[bid('left_hand')].reshape(3,3)-desired).ravel()]
    lr=least_squares(support,np.clip(d.qpos[lids],llo+1e-6,lhi-1e-6),bounds=(llo,lhi),max_nfev=100)
    d.qpos[lids]=lr.x
    solve_pos(arm,'right_hand',np.array(wrist))
    d.qpos[qid('head_tilt')]=.20
    d.qpos[qid('head_swivel')]=-.12
    d.qpos[qid('right_eye_vertical')]=-.12
    d.qpos[qid('left_eye_vertical')]=-.12
    setball(ball); forward()
    return d.qpos.copy()

# Final pose comes from fit_wrap.py: curvature, opposite-side contact and collision constraints.
final=np.array(json.loads((OUT/'wrap_pose.json').read_text())['qpos'])
d.qpos[:]=final;setball(ball);forward()
names=arm+fingers; ids=[qid(n) for n in names]; lo,hi=limits(names)
selfpairs.extend([(g,h) for g in tips+[m.geom('geom:right_thhub1').id] for h in handgeoms if m.geom_bodyid[h]==8])
def place(x): d.qpos[ids]=x; forward()
def constraint(x): place(x); return np.r_[distances()-.00015,selfdist()-.00008]
# Open the same hand for approach; interpolate joints only after the hand has arrived.
opened=final.copy()
for f in ['ff','mf','rf','lf']:
    for joint in ['knuckle','middle','distal']: opened[qid('right_'+f+'_'+joint)]=0
    opened[qid('right_'+f+'_side')]=0
opened[qid('right_lf_meta')]=0
opened[qid('right_th_swivel')]=.15; opened[qid('right_th_middle')]=0; opened[qid('right_th_distal')]=0
# Approach from the open palm's outward normal, above the sphere, so open fingers do not sweep through it.
d.qpos[:]=final;forward();handR=d.xmat[bid('right_hand')].reshape(3,3).copy();handP=d.xpos[bid('right_hand')].copy()
pre=opened.copy();d.qpos[:]=pre
ai=[qid(n) for n in arm];al,ah=limits(arm);target=handP+handR[:,1]*.05
ref=final[ai].copy()
def prefun(x):
    d.qpos[ai]=x;forward();return np.r_[(d.xpos[bid('right_hand')]-target)*1000,(d.xmat[bid('right_hand')].reshape(3,3)-handR).ravel()*2]
pr=least_squares(prefun,np.clip(ref,al+1e-6,ah-1e-6),bounds=(al,ah),max_nfev=200)
d.qpos[ai]=pr.x;forward();pre=d.qpos.copy();print('pregrasp error',np.linalg.norm(prefun(pr.x)[:3])/1000,flush=True)
poses=[]
for lean,wrist in [(0,[.16,-.13,.37]),(0,[.235,-.13,.40]),(.30,[.27,-.14,.42])]:
    q=bodypose(lean,wrist); q[22:44]=opened[22:44]; poses.append(q)
poses.extend([pre,opened,final,final.copy()])
# Correct fingers only. The arm path is not permitted to jump to evade a collision.
ids=[qid(n) for n in fingers];lo,hi=limits(fingers)
# 0--3 seconds: six 500ms windows. Last window holds the illustrated grasp.
times=[0,.5,1,1.5,2,2.5,3]
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
    extra_box(renderer.scene,[.08,.18,.116],[.085,.045,.116],[.35,.46,.54,1])
    extra_box(renderer.scene,[.365,-.115,.1755],[.003,.003,.1755],[.65,.71,.73,1])
    extra_box(renderer.scene,[.365,-.115,.012],[.06,.06,.012],[.65,.71,.73,1])
    im=Image.fromarray(renderer.render())
    dr=ImageDraw.Draw(im);dr.rectangle((0,0,800,52),fill='#112837')
    dr.text((18,9),f'{t:.2f} s  |  '+('Hand close-up' if close else ('Side view' if idx=='side' else 'Front view')),font=font,fill='white')
    dr.rectangle((0,606,800,640),fill='#112837');dr.text((18,612),'STAGED MIMo POSE | No caregiver | Not a trained-motion recording',font=small,fill='white')
    return im

for k in range(61):
    t=k*.05; seg=min(int(t/.5),5); u=(t-times[seg])/.5; u=u*u*(3-2*u)
    q=poses[seg]*(1-u)+poses[seg+1]*u
    q[3:7]/=np.linalg.norm(q[3:7]);d.qpos[:]=q;setball(ball);forward()
    if seg==3:
        # Cartesian descent with fixed palm orientation; joint interpolation alone bows through the sphere.
        target=handP+handR[:,1]*.05*(1-u)
        cr=least_squares(prefun,np.clip(d.qpos[ai],al+1e-6,ah-1e-6),bounds=(al,ah),max_nfev=100)
        d.qpos[ai]=cr.x;forward()
    for side,bodies in [('right',[49,50,51]),('left',[54,55,56])]:
        fg=[g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]
        for correction in range(2):
            floor_gap=min(mujoco.mj_geomDistance(m,d,g,0,1,None) for g in fg)
            target=d.xpos[bid(side+'_foot')].copy();target[2]-=floor_gap-.0002
            solve_pos([side+'_'+n for n in ['hip1','hip2','hip3','knee','foot1','foot2','foot3']],side+'_foot',target)
    gaze_error=align_gaze()
    # Correct any intermediate hand overlap by reducing finger closure, never by hiding geoms.
    if distances().min()<.00015 or selfdist().min()<.00005:
        x=d.qpos[ids].copy()
        def obj2(z): return np.sum((z-x)**2)
        rr=minimize(obj2,x,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[{'type':'ineq','fun':constraint}],options={'maxiter':100,'ftol':1e-10})
        place(rr.x)
    gaps=distances(); assert gaps.min()>=-.000001,(k,gaps.min());assert selfdist().min()>=-.000001,(k,'finger overlap',selfdist().min())
    rec={'frame':k,'time_s':round(t,2),'gaze_ray_error_degrees':gaze_error,'minimum_hand_object_gap_m':float(gaps.min()),'tip_gaps_m':distances(tips).tolist(),'qpos':d.qpos.tolist(),'ball_position':ball.tolist(),'right_shoulder':d.xpos[bid('right_upper_arm')].tolist(),'right_wrist':d.xpos[bid('right_hand')].tolist(),'left_wrist':d.xpos[bid('left_hand')].tolist()}
    rec['finger_flexion_degrees']={f:[float(np.degrees(d.qpos[qid('right_'+f+'_'+j)])) for j in ['knuckle','middle','distal']] for f in ['ff','mf','rf','lf']}
    contact_vectors=[]
    for g in tips:
        pair=np.zeros(6);mujoco.mj_geomDistance(m,d,g,ballg,2,pair);v=pair[3:]-ball;contact_vectors.append((v/np.linalg.norm(v)).tolist())
    rec['tip_surface_directions']=contact_vectors
    rec['thumb_finger_opposition_degrees']=[float(np.degrees(np.arccos(np.clip(np.dot(contact_vectors[-1],v),-1,1)))) for v in contact_vectors[:4]]
    rec['minimum_interfinger_gap_m']=float(selfdist().min());records.append(rec)
    for label,az,el in [('side',-95,-12),('front',155,-17)]:
        cam=mujoco.MjvCamera();cam.lookat[:]=[.14,0,.35];cam.distance=1.23;cam.azimuth=az;cam.elevation=el
        im=render(cam,label,t);im.save(OUT/f'{label}_{k:03d}.webp',quality=88)
    if k in [0,10,20,30,40,50,60]:
        for label,az,el in [('palm',-25,15),('thumb',-135,15)]:
            cam=mujoco.MjvCamera();cam.lookat[:]=ball;cam.distance=.21;cam.azimuth=az;cam.elevation=el
            render(cam,label,t,True).save(OUT/f'{label}_{k:03d}.webp',quality=95)
renderer.close()
report={'purpose':'Static illustration geometry; no dynamics or trained actor run','model':'5aff7b57135ab6431aae071e6a4a6680.mjb.gz','caregiver_visible':False,'physics_steps':0,'fps':20,'duration_s':3,'moment_duration_s':.5,'sphere_radius_m':.019,'frames':records,'minimum_gap_all_frames_m':min(x['minimum_hand_object_gap_m'] for x in records),'elapsed_s':time.monotonic()-start,'pose_source':'wrap_pose.json generated by fit_wrap.py'}
(WORK/'evidence/pose_geometry.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='frames'}),flush=True)
