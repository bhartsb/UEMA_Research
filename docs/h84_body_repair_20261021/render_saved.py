"""Draw existing MIMo evidence; no physics stepping, model training or new experiment.

Owner: the overseer. Retain with H84_BODY_REPAIR_EXPLAINED_2026-10-21.html.
Run from repository root: .venv_mimo/Scripts/python.exe -B docs/h84_body_repair_20261021/render_saved.py
"""
from pathlib import Path
import hashlib, importlib.util, json, sys, time, subprocess, types
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COMMIT = 'ebea657cd055f3f6a9466867a0fdf3f1343e326a'
EV = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/whole_body_request/attempts/comparison_483120_a2/evidence')
OBS = ROOT/'Saved/mimo_bridge/experiments/H84_fresh_drivers_20260918_seg41j_1526/observer/world'
MODEL = OBS/'models/5aff7b57135ab6431aae071e6a4a6680.mjb.gz'
HEADER = OBS/'b5/header_29728_1791671301772.json'
sys.path.insert(0,str(ROOT/'Source/Python'))
from mimo_bridge.tools.movement_viewer import commands as C, render as OLD, sources as S
oldpath = ROOT/'Source/Python/mimo_bridge/tools/movement_viewer/render.py'
code_hashes={}
def committed_module(name,path):
    raw=subprocess.check_output(['git','show',COMMIT+':'+path],cwd=ROOT)
    code_hashes[path]=hashlib.sha256(raw).hexdigest()
    mod=types.ModuleType(name);mod.__file__=str(ROOT/path);mod.__package__=name.rpartition('.')[0]
    sys.modules[name]=mod
    exec(compile(raw,str(ROOT/path),'exec'),mod.__dict__)
    return mod
for mod in ['limb_target_control','arm_reach']:
    committed_module('mimo_bridge.'+mod,'Source/Python/mimo_bridge/'+mod+'.py')
G=committed_module('mimo_bridge.tools.movement_viewer.pose_geometry','Source/Python/mimo_bridge/tools/movement_viewer/pose_geometry.py')
R=committed_module('mimo_bridge.tools.movement_viewer.saved_render','Source/Python/mimo_bridge/tools/movement_viewer/render.py')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started=time.monotonic()
    a=np.load(EV/'records.npz',allow_pickle=False)
    header=json.loads(HEADER.read_text())
    model=C.BodyModel(str(MODEL),header)
    body=R._Body(str(MODEL),header,())
    old=OLD._Body(str(MODEL),header,())
    try:
        start=a['start_qpos']; motor=a['original_motor']; mask=a['original_mask']
        hand=a['original_hand_goal']; goal=a['goal_qpos']
        body.set_frame(start); old.set_frame(start)
        notes=[]
        corrected=R._command_pose(body,model,motor,mask,start,hand,notes)
        if corrected is None: raise RuntimeError(notes)
        q_req=start.copy(); q_req[np.asarray(model.qadr)]=hand
        scored=C.scored(model.names)
        targets={int(model.hb[i]):C.quat_xyzw_to_mat(row[3:7]) for i,row in enumerate(motor)
                 if mask[i]!=0 and row[8]>0 and scored[i]}
        frames=old.requested_frames(q_req,targets)
        states={'start':start,'goal':goal,'corrected':corrected['q']}
        for lab,arm in [('old250','original_A_hold250'),('new250','original_B_hold250'),
                        ('old1000','original_A_hold1000'),('new1000','original_B_hold1000')]:
            states[lab]=a['qpos_'+arm][-1]
        goal_frame=body.frame_of(goal)
        geoms=[body.own_geoms(q) for lab,q in states.items() if lab!='goal']
        geoms += [body.reframed(body.own_geoms(goal),goal_frame),old.own_geoms(q_req,frames=frames)]
        body.fit(geoms,[])
        body.cams={v:(look,dist*1.24,az,el) for v,(look,dist,az,el) in body.cams.items()}
        old.cams=body.cams.copy()
        images=[]
        for view in ['front','left','right']:
            for lab,q in states.items():
                if lab=='corrected': continue  # used for common camera bounds only
                mo=a['goal_mocap'] if lab=='goal' else a['start_mocap']
                op=a['goal_objpose'] if lab=='goal' else a['start_objpose']
                images.append(body.draw(lab,view,q,mo,op,out_dir=str(OUT),
                                        frame=goal_frame if lab=='goal' else None))
        manifest={'purpose':'Real saved MIMo states and clearly labelled calculated target drawings for operator explanation',
                  'owner':'the overseer','physics_steps':0,'training_updates':0,'seconds':time.monotonic()-started,
                  'recording':'H84_fresh_drivers_20260918_seg41j_1526','body':5,'command_step':483120,
                  'start_step':483119,'goal_step':int(a['goal_record_step']),
                  'note':'Common camera and size. Start trunk frame for all except goal, which uses its own trunk frame. Isolated upright display, not standing. No ghosts, target rods or caregiver.',
                  'sources':{str(p):sha(p) for p in [EV/'records.npz',EV/'result.json',MODEL,HEADER,oldpath,Path(__file__)]},
                  'committed_renderer':{'commit':COMMIT,'files':code_hashes},
                  'images':{Path(p).name:sha(OUT/Path(p).name) for p in images},'render_notes':notes}
        (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'comparison_images':len(images),'seconds':manifest['seconds'],'goal_step':manifest['goal_step']}))
    finally:
        old.close();body.close()
    exact_image()

def exact_image():
    rid='a4ba45b19cbd';body_id=3;step=494810
    rec=S.by_id(rid)
    data,cols,used,found=OLD.find_command(rec,body_id,step)
    idx=int(np.flatnonzero(data['step']==step)[0]); ci=int(data['chunk'][idx]); row=int(data['row'][idx])
    start=np.array(data['qpos'][idx-1]);end=np.array(data['qpos'][idx+4])
    run_dir=Path('D:/UEMA_Research/external/D/UEMA_preserved/pose_comparison_POSE_A_off_494025/run')
    goal_info=G.remembered_goal(run_dir,step,body_id)
    if 'record_step' not in goal_info: raise RuntimeError(goal_info)
    goal_idx=int(np.flatnonzero(data['step']==goal_info['record_step'])[0])
    remembered=np.array(data['qpos'][goal_idx])
    motor=np.asarray(data['motors'][ci][row],float); mask=np.asarray(data['masks'][ci][row])
    hand=np.asarray(data['goals'][ci][row]) if 'goals' in data else None
    modelpath=Path(rec['bodies'][str(body_id)]['model']);header=data['headers'][0]
    model=C.BodyModel(str(modelpath),header)
    b=R._Body(str(modelpath),header,());o=OLD._Body(str(modelpath),header,())
    try:
        b.set_frame(start);o.set_frame(start);notes=[]
        corr=R._command_pose(b,model,motor,mask,start,hand,notes)
        if corr is None: raise RuntimeError(notes)
        scored=C.scored(model.names)
        targets={int(model.hb[i]):C.quat_xyzw_to_mat(v[3:7]) for i,v in enumerate(motor) if mask[i]!=0 and v[8]>0 and scored[i]}
        q_req=start.copy()
        if hand is not None and hand.size==len(model.qadr): q_req[np.asarray(model.qadr)]=hand
        frames=o.requested_frames(q_req,targets)
        handle_names=['wrist_L','wrist_R','head','lower_trunk']
        hindices=[list(model.names).index(n) for n in handle_names]
        old_points=[motor[i,:3]/100 for i in hindices]
        new_rows=G.corrected_command(b.m,b.d,list(model.names),model.hb,start,motor,mask)['rows']
        new_points=[new_rows[i,:3]/100 for i in hindices]
        memory_data=mujoco.MjData(b.m)
        memory_data.qpos[:]=remembered
        mujoco.mj_forward(b.m,memory_data)
        memory_points=[np.array(memory_data.xpos[int(model.hb[i])]) for i in hindices]
        b.fit([b.own_geoms(start),b.own_geoms(end),b.own_geoms(remembered),b.own_geoms(corr['q']),o.own_geoms(q_req,frames)],old_points+new_points)
        b.cams={v:(look,dist*1.18,az,el) for v,(look,dist,az,el) in b.cams.items()};o.cams=b.cams.copy()
        images=[]
        overlays={}
        def projected(renderer,point):
            cams=renderer.renderer.scene.camera
            cp=(np.array(cams[0].pos)+np.array(cams[1].pos))/2
            f=np.array(cams[0].forward);u=np.array(cams[0].up);r=np.cross(f,u)
            v=renderer.Rf.T@(np.asarray(point)-renderer.o)-cp
            z=float(v@f);near=float(cams[0].frustum_near)
            half_h=(float(cams[0].frustum_top)-float(cams[0].frustum_bottom))/2
            nx=float(v@r)/z*near/(half_h*R.BODY_W/R.BODY_H)
            ny=float(v@u)/z*near/half_h
            return [round((nx+1)*R.BODY_W/2,3),round((1-ny)*R.BODY_H/2,3)]
        def record_overlay(renderer,label,view,positions):
            overlays[view+'_'+label]=[{'name':n,'asked':projected(renderer,old_points[k]),
                 'actual':projected(renderer,positions[int(model.hb[hindices[k]])]),
                 'memory':projected(renderer,memory_points[k]),
                 'new_asked':projected(renderer,new_points[k]),
                 'gap_cm':round(float(np.linalg.norm(old_points[k]-positions[int(model.hb[hindices[k]])])*100),1),
                 'request_shift_cm':round(float(np.linalg.norm(old_points[k]-new_points[k])*100),1)}
                 for k,n in enumerate(handle_names)]
        for view in ['front','left','right']:
            images.append(o.draw('exact_old_request',view,q_req,None,None,frames=frames,out_dir=str(OUT)))
            record_overlay(o,'exact_old_request',view,frames[2])
            for label,q in [('exact_start',start),('exact_actual',end),('exact_corrected',corr['q']),('memory_goal',remembered)]:
                images.append(b.draw(label,view,q,None,None,out_dir=str(OUT)))
                if label!='memory_goal': record_overlay(b,label,view,np.array(b.d.xpos))
        cached=Path('D:/UEMA_Research/external/D/UEMA_movement_viewer/pictures/ea26f7094e53f0f48a8a/manifest.json')
        orig=json.loads(cached.read_text())
        info={'recording':rid,'run':rec['run'],'body':body_id,'command_step':step,
              'start_step':step-1,'end_step':step+4,'kind':'exact attached image: saved states plus calculated target only; no simulated corrected outcome',
              'manifest':str(cached),'manifest_sha256':sha(cached),'source_provenance':orig['provenance'],
              'remembered_goal':goal_info,
              'measures':orig['measures'],'hinge_residuals':corr['residuals'],
              'corrected':corr['corrected'],'notes':notes,'data_keys':list(data),
              'images':{Path(p).name:sha(OUT/Path(p).name) for p in images},
              'overlays':overlays,'overlay_note':'SVG coordinates projected from the same MuJoCo camera and source world positions. Filled numbered circle = original request; open ring = displayed body point. Diamond on corrected drawing = changed request.'}
        manifest=json.loads((OUT/'provenance.json').read_text());manifest['exact_attached_image']=info
        # Read the saved path; do not simulate or estimate missing velocities.
        from mimo_bridge.observer_record import read_chunk
        raw=read_chunk(found[ci][1])
        pg,rg=model.fk(remembered);p0,r0=model.fk(start)
        trace=[]
        for j in range(idx-1,idx+5):
            p,r=model.fk(data['qpos'][j])
            trace.append({'step':int(data['step'][j]),
                'left_wrist_command_gap_cm':float(np.linalg.norm(p[0]-motor[0,:3]/100)*100),
                'left_wrist_from_start_cm':float(np.linalg.norm(p[0]-p0[0])*100),
                'left_wrist_memory_gap_cm':float(np.linalg.norm(p[0]-pg[0])*100),
                'body_15_mean_memory_position_gap_cm':float(np.mean(np.linalg.norm(p[scored]-pg[scored],axis=1))*100),
                'body_15_mean_memory_rotation_gap_deg':float(np.mean(C._angles_deg(rg,r)[scored]))})
        latch=None
        for line in (run_dir/'causal_chain.jsonl').open(encoding='utf-8'):
            item=json.loads(line)
            if item.get('type')=='latch' and item.get('step')==step:latch=item
        if latch is None: raise RuntimeError('exact planning record missing')
        opt=latch['plan']['optimizer'][0]
        joints={}
        for j in range(model.m.njnt):
            name=model.m.joint(j).name
            if 'left_shoulder_horizontal' in name:
                a=int(model.m.jnt_qposadr[j])
                joints[name]={'start_end_deg':np.degrees([start[a],end[a]]).tolist(),
                    'range_deg':np.degrees(model.m.jnt_range[j]).tolist()}
        ai=header['actuator_names'].index('act:left_shoulder_horizontal')
        rr=[int(np.flatnonzero(raw['step']==s)[0]) for s in range(step,step+5)]
        manifest['exact_command_trace']={
            'purpose':'Explain command versus remembered goal from existing exact records; no physics stepping.',
            'physics_steps':0,'training_updates':0,'remembered_goal':goal_info,
            'goal_mapping':'The planner input at step s is observer record s-1. Its memory age is 3, so 494810-3-1=494806. Confirmed in run commit ff005b254 trainer/model/observer call order.',
            'source':'planner','source_run_commit':'ff005b254',
            'original_command_constant_steps':[step,step+4],'replacement_step':step+5,
            'left_wrist_command_xyz_cm':motor[0,:3].tolist(),
            'right_wrist_command_xyz_cm':motor[6,:3].tolist(),
            'floor_z_cm':0.,
            'hand_collision_box_half_sizes_m':[0.0180282,0.00618988,0.018532],
            'floor_interpretation':'Both wrist points are below the floor. The wrist origin is at the centre of a floor-colliding box. This proves endpoints require penetration, not a causal apportionment of the remaining wrist gap.',
            'saved_path':trace,'joint_observations':joints,
            'left_shoulder_horizontal_ctrl':np.asarray(raw['ctrl'][rr,ai],float).tolist(),
            'selected_search_iteration':opt['selected_iteration'][body_id],
            'planner_score':{k:{t:opt['per_body'][k][t][body_id] for t in ['start','selected']} for k in ['goal','effort','objective']},
            'meaning':'Planner score compares predicted sensory encodings with remembered encodings, plus commanded-strength cost. It is not distance to the wrist command point. Saved-pose comparison is a separate physical description.',
            'causal_limit':'Observer has qpos, sampled ctrl and contact summaries, but no qvel or full simulator state at this instant. This read does not divide the 17 cm gap among controller compromise, force, momentum and contact.',
            'source_sha256':{str(p):sha(p) for p in [Path(found[ci][1]),run_dir/'args.json',run_dir/'causal_chain.jsonl']}}
        manifest['sources'][str(Path(__file__))]=sha(Path(__file__))
        (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        (OUT/'handles.js').write_text('window.H84_HANDLES='+json.dumps(overlays,separators=(',',':'))+';\n',encoding='utf-8')
        print(json.dumps({'exact_images':len(images),'residuals':corr['residuals'],'data_keys':list(data)}))
    finally: b.close();o.close()

def joint_detail():
    """Render saved joints, not a simulated or predicted successful repair."""
    started=time.monotonic()
    rec=S.by_id('a4ba45b19cbd')
    data,cols,used,found=OLD.find_command(rec,3,494810)
    idx=int(np.flatnonzero(data['step']==494810)[0])
    header=data['headers'][0]
    b=OLD._Body(rec['bodies']['3']['model'],header,())
    indices={'earlier':int(np.flatnonzero(data['step']==494806)[0]),'start':idx-1,'end':idx+4}
    poses={k:np.array(data['qpos'][i]) for k,i in indices.items()}
    joint_ids=[j for j in range(b.m.njnt) if (b.m.joint(j).name or '').startswith('robot:left_') and any(s in b.m.joint(j).name for s in ('shoulder','elbow','hand1','hand2','hand3'))]
    wrist=int(header['handle_bodies'][header['handle_names'].index('wrist_L')])
    chain=set()
    for j in joint_ids: chain.add(int(b.m.jnt_bodyid[j]))
    for bid in b.own_set:
        parent=bid
        while parent>0:
            if parent in chain:chain.add(bid);break
            parent=int(b.m.body_parentid[parent])
    overlays={};images={};joints=[]
    elbow_only=poses['start'].copy()
    ej=b.m.joint('robot:left_elbow').id
    elbow_only[int(b.m.jnt_qposadr[ej])]=poses['earlier'][int(b.m.jnt_qposadr[ej])]
    poses['elbow_only']=elbow_only
    all_arm=poses['start'].copy()
    for j in joint_ids: all_arm[int(b.m.jnt_qposadr[j])]=poses['earlier'][int(b.m.jnt_qposadr[j])]
    poses['arm_only']=all_arm
    try:
        b.set_frame(poses['start'])
        geoms=[[g for g in b.own_geoms(q) if int(b.m.geom_bodyid[g[0]]) in chain] for q in poses.values()]
        b.fit(geoms,[])
        b.cams={v:(look,dist*1.16,az,el) for v,(look,dist,az,el) in b.cams.items()}
        for j in joint_ids:
            adr=int(b.m.jnt_qposadr[j])
            joints.append({'name':b.m.joint(j).name,'limits_deg':np.rad2deg(b.m.jnt_range[j]).tolist(),
                'angles_deg':{name:float(np.rad2deg(q[adr])) for name,q in poses.items()},
                'path_deg':[float(np.rad2deg(q[adr])) for q in data['qpos'][idx-1:idx+5]]})
        for view in ('front','left','right'):
            for name,q in poses.items():
                fn=b.draw('joint_'+name,view,q,None,None,out_dir=str(OUT));images[fn]=sha(OUT/fn)
                cams=b.renderer.scene.camera
                cp=(np.array(cams[0].pos)+np.array(cams[1].pos))/2
                f=np.array(cams[0].forward);u=np.array(cams[0].up);r=np.cross(f,u)
                near=float(cams[0].frustum_near);hh=(float(cams[0].frustum_top)-float(cams[0].frustum_bottom))/2
                def project(p):
                    v=b.Rf.T@(np.asarray(p)-b.o)-cp;z=float(v@f)
                    return [round((float(v@r)/z*near/(hh*OLD.BODY_W/OLD.BODY_H)+1)*OLD.BODY_W/2,2),round((1-float(v@u)/z*near/hh)*OLD.BODY_H/2,2)]
                overlays[view+'_'+name]={'joints':{b.m.joint(j).name:project(b.d.xanchor[j]) for j in joint_ids},'axes':{b.m.joint(j).name:[project(b.d.xanchor[j]-.035*b.d.xaxis[j]),project(b.d.xanchor[j]+.035*b.d.xaxis[j])] for j in joint_ids},'wrist':project(b.d.xpos[wrist])}
        wrist_positions={}
        for name,q in poses.items():
            b.place(q);wrist_positions[name]=np.array(b.d.xpos[wrist]).tolist()
        ci=int(data['chunk'][idx]);row=int(data['row'][idx])
        motor=np.asarray(data['motors'][ci][row],float)
        rows={name:motor[header['handle_names'].index(name)].tolist() for name in ('upperarm_L','lowerarm_L','wrist_L') if name in header['handle_names']}
        evidence={'purpose':'Original screenshot: actual seven-joint arm positions and shared camera close-ups; not the new rule outcome',
            'owner':'the overseer','recording':'POSE_A_off_494025','body':3,'command_step':494810,
            'steps':{k:int(data['step'][i]) for k,i in indices.items()},'wrist_positions_m':wrist_positions,'calculated_examples':{'elbow_only':'Start body with only elbow set to earlier angle; forward kinematics, no movement prediction','arm_only':'Start body with all seven arm angles set to earlier values; trunk and other joints held at start'},'joint_rows':joints,'command_rows':rows,
            'overlays':overlays,'images':images,'sources':{str(found[ci][1]):sha(Path(found[ci][1]))},
            'physics_steps':0,'training_updates':0,'seconds':time.monotonic()-started,
            'view_note':'Starting chest frame, same camera across states. Actual body was lying down; floor and other bodies hidden. Close-up may crop the trunk.'}
        (OUT/'joint_detail.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
        (OUT/'joint_detail.js').write_text('window.H84_JOINTS='+json.dumps(evidence)+';\n',encoding='utf-8')
        print(json.dumps({'joints':joints,'command_rows':rows,'seconds':evidence['seconds']}))
    finally:b.close()

if __name__=='__main__':
    if '--joint-detail' in sys.argv: joint_detail()
    elif '--exact-only' in sys.argv: exact_image()
    else: main()
