"""Geometry validation for the document's prescribed poses; no dynamics or learning."""
from pathlib import Path
import json,time
import numpy as np
import mujoco
from scipy.optimize import minimize_scalar
p=Path(__file__).with_name('render_scene.py'); ns={'__file__':str(p.resolve())}
exec(compile(p.read_text().split('# Final pose comes')[0],str(p),'exec'),ns)
globals().update(ns)
r=json.loads((WORK/'evidence/pose_geometry.json').read_text());frames=r['frames']
extra=[(g,h) for g in fingergeoms for h in handgeoms if m.geom_bodyid[h]==8 and any(x in m.geom(g).name for x in ['middle','distal','thhub'])]
selfpairs.extend(extra)
stand_lo=np.array([.362,-.118,0]);stand_hi=np.array([.368,-.112,.351])
def stand_gap(g):
    center=d.geom_xpos[g];typ=int(m.geom_type[g]);size=m.geom_size[g];rot=d.geom_xmat[g].reshape(3,3)
    if typ in [3,5]:
        a=center-rot[:,2]*size[1];b=center+rot[:,2]*size[1]
        def dist(t):
            z=a+t*(b-a);return np.linalg.norm(np.maximum(np.maximum(stand_lo-z,z-stand_hi),0))
        rr=minimize_scalar(dist,bounds=(0,1),method='bounded',options={'xatol':1e-9})
        return min(dist(0),dist(1),rr.fun)-size[0]
    # Conservative lower bound from oriented box's world AABB; never overstates clearance.
    half=np.abs(rot)@size;lo=center-half;hi=center+half
    return np.linalg.norm(np.maximum(np.maximum(stand_lo-hi,lo-stand_hi),0))
mins={'object_m':1.,'self_m':1.,'stand_m':1.};worst={};samples=0
for k in range(60):
    qa=np.array(frames[k]['qpos']);qb=np.array(frames[k+1]['qpos'])
    for sub in range(10):
        t=sub/10;d.qpos[:]=qa*(1-t)+qb*t;d.qpos[3:7]/=np.linalg.norm(d.qpos[3:7]);setball(ball);forward();samples+=1
        values={'object_m':float(distances().min()),'self_m':float(selfdist().min()),'stand_m':min(stand_gap(g) for g in handgeoms)}
        for key,value in values.items():
            if value<mins[key]:mins[key]=value;worst[key]={'time_s':k*.05+t*.05,'value':value}
            if key=='self_m' and value<=mins[key]:
                pair=selfpairs[int(np.argmin(selfdist()))];worst[key]['pair']=[m.geom(g).name for g in pair]
delta=np.linalg.norm(np.diff([v['right_wrist'] for v in frames],axis=0),axis=1)
result={'purpose':'Document static-pose path validation','sampling_seconds':.005,'samples':samples,'minimum_gaps':mins,'worst':worst,'max_wrist_displacement_per_50ms_mm':float(delta.max()*1000),'max_wrist_displacement_approach_1_5_to_2_s_mm':float(delta[30:40].max()*1000),'max_digit_joint_change_per_50ms_deg':float(np.degrees(np.abs(np.diff([np.array(v['qpos'])[22:44] for v in frames],axis=0))).max()),'final_thumb_finger_opposition_degrees':frames[50]['thumb_finger_opposition_degrees'],'verdict':'PASS' if min(mins.values())>=-1e-6 else 'FAIL','note':'5ms sampled linear joint interpolation, not mathematical continuous collision proof. Stand checked against capsule/cylinder enclosures and conservative palm AABB.'}
(WORK/'evidence/illustration_path_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
