from pathlib import Path
p=Path('docs/h84_whole_body_reach_20261026/render_scene.py');ns={'__file__':str(p.resolve())};exec(compile(p.read_text().split('# Final pose comes')[0],str(p),'exec'),ns);globals().update(ns)
bodypose(.65,[.34,-.115,.375]);R=d.xmat[bid('right_hand')].reshape(3,3).copy();origin=d.xpos[bid('right_hand')].copy()
selfpairs.extend([(g,h) for g in tips+[m.geom('geom:right_thhub1').id] for h in handgeoms if m.geom_bodyid[h]==8])
ids=[qid(n) for n in fingers];lo,hi=limits(fingers)
for f in ['ff','mf','rf','lf']:
 for suf,v in [('knuckle',.5),('middle',1.4),('distal',.8)]:d.qpos[qid('right_'+f+'_'+suf)]=v
d.qpos[qid('right_th_swivel')]=1.5;d.qpos[qid('right_th_adduction')]=-.6;d.qpos[qid('right_th_middle')]=.8;d.qpos[qid('right_th_distal')]=.8
x0=np.r_[np.clip(d.qpos[ids],lo+1e-6,hi-1e-6),[-.004,-.028,.008]]
bounds=list(zip(lo,hi))+[(-.01,.008),(-.04,-.024),(-.015,.025)]
def place(x):
 d.qpos[ids]=x[:-3];setball(origin+R@x[-3:]);forward()
def obj(x):
 place(x);center=x[-3:];tiplocal=np.array([R.T@(d.geom_xpos[g]-origin) for g in tips]);v=tiplocal-center
 # Far sides of the sphere: curved fingers on the negative-palm-normal hemisphere, thumb at its opposite end.
 cost=np.sum((distances(tips)-.00035)**2)*1e6
 cost+=np.sum(np.maximum(v[:4,1]+.009,0)**2)*1e6
 cost+=np.maximum(.009-v[4,0],0)**2*1e6
 for f in ['ff','mf','rf','lf']:
  cost+=max(2.05-sum(d.qpos[qid('right_'+f+'_'+j)] for j in ['knuckle','middle','distal']),0)**2
 cost+=np.sum((center-[-.004,-.028,.008])**2)*100
 return cost
def cons(x):place(x);return np.r_[distances()-.0002,selfdist()-.0001]
def residual(x):
 place(x);center=x[-3:];tiplocal=np.array([R.T@(d.geom_xpos[g]-origin) for g in tips]);v=tiplocal-center
 return np.r_[(distances(tips)-.0004)*150,np.minimum(distances()-.0002,0)*1500,np.minimum(selfdist()-.0001,0)*1500,np.maximum(v[:4,1]+.006,0)*200,np.maximum(v[4,2]+.005,0)*150,[(d.qpos[qid('right_'+f+'_'+j)]-target)*.08 for f in ['ff','mf','rf','lf'] for j,target in [('knuckle',.65),('middle',1.3),('distal',.8)]],(center-[-.004,-.025,.003])*20]
r=least_squares(residual,x0,bounds=np.array(bounds).T,max_nfev=500,diff_step=1e-4,ftol=1e-10,xtol=1e-10,gtol=1e-10);place(r.x)
print(r.success,r.message,obj(r.x),'gaps',distances(tips),'min',distances().min(),selfdist().min(),'center',r.x[-3:],flush=True)
print([(n,round(float(d.qpos[qid(n)]),3)) for n in fingers],flush=True)
# Move arm to put this grasp around the original scene ball, keeping hand orientation.
localball=r.x[-3:].copy(); target=ball-R@localball; armi=[qid(n) for n in arm];alo,ahi=limits(arm)
def armfun(x):
 d.qpos[armi]=x;forward();return np.r_[(d.xpos[bid('right_hand')]+d.xmat[bid('right_hand')].reshape(3,3)@localball-ball)*1000,(d.xmat[bid('right_hand')].reshape(3,3)-R).ravel()*.001]
a=least_squares(armfun,np.clip(d.qpos[armi],alo+1e-6,ahi-1e-6),bounds=(alo,ahi),max_nfev=200);d.qpos[armi]=a.x;setball(ball);forward()
print('arm error',np.linalg.norm(armfun(a.x)[:3])/100,'gaps',distances(tips),distances().min(),selfdist().min(),flush=True)
print('tip vectors',np.array([d.xmat[bid('right_hand')].reshape(3,3).T@(d.geom_xpos[g]-ball) for g in tips]),flush=True)
(OUT/'wrap_pose.json').write_text(json.dumps({'qpos':d.qpos.tolist(),'local_ball':localball.tolist(),'source':'fit_wrap.py static geometry only','min_object_gap_m':float(distances().min()),'min_self_gap_m':float(selfdist().min())},indent=2))
renderer=mujoco.Renderer(m,height=640,width=800)
for name,az,el in [('wrap_palm',-70,-25),('wrap_thumb',80,-25),('wrap_top',-90,-70)]:
 cam=mujoco.MjvCamera();cam.lookat[:]=ball;cam.distance=.20;cam.azimuth=az;cam.elevation=el;renderer.update_scene(d,camera=cam);Image.fromarray(renderer.render()).save(WORK/'evidence'/f'{name}.png')
renderer.close()

