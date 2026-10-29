"""Draw the saved movement of the screenshot's command as frames; no physics stepping, no training.

Reads the preserved recording of run POSE_A_off_494025, body 3: the goal record (model step 494,806) and the six
records from the start (494,809) to the end of the command's hold (494,814). Draws each with the movement viewer's
renderer (forward kinematics of the saved joint positions), writes per-frame left-arm joint angles and the left
hand point's distances, and copies the viewer's own three pictures of the export the operator screenshotted.
Run from the repository root: .venv_mimo/Scripts/python.exe -B docs/h84_body_repair_v2_20261023/render_path.py
"""
from pathlib import Path
import hashlib, json, shutil, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Source/Python'))
from mimo_bridge.tools.movement_viewer import render as OLD, sources as S  # noqa: E402

VIEWER = Path('D:/UEMA_Research/external/D/UEMA_movement_viewer/pictures/ea26f7094e53f0f48a8a')
JOINTS = ['robot:left_shoulder_horizontal', 'robot:left_shoulder_ad_ab', 'robot:left_shoulder_rotation',
          'robot:left_elbow', 'robot:left_hand1', 'robot:left_hand2', 'robot:left_hand3']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    rec = S.by_id('a4ba45b19cbd')
    data, cols, used, found = OLD.find_command(rec, 3, 494810)
    idx = int(np.flatnonzero(data['step'] == 494810)[0])
    gidx = int(np.flatnonzero(data['step'] == 494806)[0])
    header = data['headers'][0]
    b = OLD._Body(rec['bodies']['3']['model'], header, ())
    order = [('goal', gidx)] + [('s%d' % k, idx - 1 + k) for k in range(6)]
    poses = {n: np.array(data['qpos'][i]) for n, i in order}
    steps = {n: int(data['step'][i]) for n, i in order}
    joint_ids = [b.m.joint(n).id for n in JOINTS]
    wrist = int(header['handle_bodies'][header['handle_names'].index('wrist_L')])
    ci = int(data['chunk'][idx]); row = int(data['row'][idx])
    motor = np.asarray(data['motors'][ci][row], float)
    cmd_point = motor[header['handle_names'].index('wrist_L'), :3] / 100.0
    overlays, images, frames = {}, {}, []
    try:
        b.set_frame(poses['s0'])
        b.fit([b.own_geoms(q) for q in poses.values()], [cmd_point])
        b.cams = {v: (look, dist * 1.12, az, el) for v, (look, dist, az, el) in b.cams.items()}
        pts = {}
        for n, q in poses.items():
            b.place(q); pts[n] = np.array(b.d.xpos[wrist])
        for view in ('front', 'left', 'right'):
            for n, q in poses.items():
                fn = b.draw('path_' + n, view, q, None, None, out_dir=str(OUT)); images[fn] = sha(OUT / fn)
                b.place(q)
                cams = b.renderer.scene.camera
                cp = (np.array(cams[0].pos) + np.array(cams[1].pos)) / 2
                f = np.array(cams[0].forward); u = np.array(cams[0].up); r = np.cross(f, u)
                near = float(cams[0].frustum_near)
                hh = (float(cams[0].frustum_top) - float(cams[0].frustum_bottom)) / 2

                def project(p):
                    v = b.Rf.T @ (np.asarray(p) - b.o) - cp; z = float(v @ f)
                    return [round((float(v @ r) / z * near / (hh * OLD.BODY_W / OLD.BODY_H) + 1) * OLD.BODY_W / 2, 2),
                            round((1 - float(v @ u) / z * near / hh) * OLD.BODY_H / 2, 2)]
                overlays[view + '_' + n] = {
                    'hand': project(pts[n]), 'goal_hand': project(pts['goal']), 'command_hand': project(cmd_point),
                    'joints': {jn: project(b.d.xanchor[j]) for jn, j in zip(JOINTS, joint_ids)}}
        for n, q in poses.items():
            frames.append({
                'name': n, 'step': steps[n], 'ms': None if n == 'goal' else (steps[n] - steps['s0']) * 50,
                'hand_gap_to_goal_cm': float(np.linalg.norm(pts[n] - pts['goal']) * 100),
                'hand_gap_to_command_cm': float(np.linalg.norm(pts[n] - cmd_point) * 100),
                'hand_moved_from_start_cm': float(np.linalg.norm(pts[n] - pts['s0']) * 100),
                'angles_deg': {jn: float(np.rad2deg(q[int(b.m.jnt_qposadr[j])])) for jn, j in zip(JOINTS, joint_ids)}})
        limits = {jn: np.rad2deg(b.m.jnt_range[j]).tolist() for jn, j in zip(JOINTS, joint_ids)}
        copied = {}
        for fn in ['body_front_start.png', 'body_front_target.png', 'body_front_t250.png',
                   'body_left_target.png', 'body_right_target.png', 'hand_left_front_target.png']:
            shutil.copy2(VIEWER / fn, OUT / ('viewer_' + fn))
            copied['viewer_' + fn] = {'from': str(VIEWER / fn), 'sha256': sha(VIEWER / fn)}
        evidence = {
            'purpose': 'The saved movement of the screenshot command drawn frame by frame for the operator explanation; '
                       'real saved joint positions through forward kinematics; no physics stepping, no prediction, no training',
            'owner': 'the builder (the operator-requested revision of Oct 23, 2026)',
            'recording': 'POSE_A_off_494025', 'body': 3, 'command_step': 494810,
            'frames': frames, 'joint_limits_deg': limits, 'command_hand_point_world_m': cmd_point.tolist(),
            'overlays': overlays, 'images': images, 'viewer_pictures_copied': copied,
            'sources': {str(found[ci][1]): sha(Path(found[ci][1])), rec['bodies']['3']['model']: sha(rec['bodies']['3']['model'])},
            'renderer': 'Source/Python/mimo_bridge/tools/movement_viewer/render.py (the working tree) ' + sha(ROOT / 'Source/Python/mimo_bridge/tools/movement_viewer/render.py'),
            'physics_steps': 0, 'training_updates': 0, 'seconds': time.monotonic() - started,
            'view_note': 'Starting chest frame, the same camera for every frame and view; the body was lying on its back '
                         'and is drawn upright and alone; the floor is not drawn.'}
        (OUT / 'path_detail.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
        (OUT / 'path_detail.js').write_text('window.H84_PATH=' + json.dumps(evidence) + ';\n', encoding='utf-8')
        print(json.dumps({'frames': frames, 'seconds': evidence['seconds']}, indent=1))
    finally:
        b.close()


if __name__ == '__main__':
    main()
