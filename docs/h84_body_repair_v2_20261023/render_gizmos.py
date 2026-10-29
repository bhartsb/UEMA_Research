"""MuJoCo-drawn marks on the saved movement frames of the screenshot command; no physics stepping, no training.

On each of the seven saved states (the goal record and the six records of the command's hold) this draws, inside the
MuJoCo scene before rendering: the seven left-arm joint axes as short rods at their anchors, colored by the saved
motor control of that joint's actuator at that model step (grey = 0, red = the full control of 1); a blue rod on the
upper arm, the forearm and the hand showing the direction the command asked that part to point; the hand point
(orange), the goal's hand point (green) and the command's requested hand position (blue), joined by lines; and a red
sphere at every saved contact point between the body and the floor at that model step.
Run from the repository root: .venv_mimo/Scripts/python.exe -B docs/h84_body_repair_v2_20261023/render_gizmos.py
"""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Source/Python'))
import mujoco as mj  # noqa: E402
from PIL import Image  # noqa: E402
from mimo_bridge.tools.movement_viewer import render as OLD, sources as S, commands as C  # noqa: E402

JOINTS = ['robot:left_shoulder_horizontal', 'robot:left_shoulder_ad_ab', 'robot:left_shoulder_rotation',
          'robot:left_elbow', 'robot:left_hand1', 'robot:left_hand2', 'robot:left_hand3']
ACT = {'robot:left_shoulder_horizontal': 'act:left_shoulder_horizontal', 'robot:left_shoulder_ad_ab': 'act:left_shoulder_abduction',
       'robot:left_shoulder_rotation': 'act:left_shoulder_internal', 'robot:left_elbow': 'act:left_elbow',
       'robot:left_hand1': 'act:left_wrist_rotation', 'robot:left_hand2': 'act:left_wrist_flexion', 'robot:left_hand3': 'act:left_wrist_ulnar'}
ORANGE, GREEN, BLUE, RED, DARK = [0.76, 0.29, 0.07, 1], [0.03, 0.47, 0.24, 1], [0.09, 0.40, 0.73, 1], [0.95, 0.78, 0.05, 1], [0.09, 0.16, 0.23, 1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    rec = S.by_id('a4ba45b19cbd')
    data, cols, used, found = OLD.find_command(rec, 3, 494810)
    idx = int(np.flatnonzero(data['step'] == 494810)[0]); gidx = int(np.flatnonzero(data['step'] == 494806)[0])
    header = data['headers'][0]
    ci = int(data['chunk'][idx]); row = int(data['row'][idx])
    z = np.load(found[ci][1]); zsteps = z['step']
    ent = {e['code']: e['name'] for e in header['entities']}
    own = {e['code'] for e in header['entities'] if e.get('kind') == 'self'}
    act_index = {n: i for i, n in enumerate(header['actuator_names'])}
    motor = np.asarray(data['motors'][ci][row], float)
    hn = header['handle_names']
    cmd_point = motor[hn.index('wrist_L'), :3] / 100.0
    req = {'upper': motor[hn.index('upperarm_L')], 'fore': motor[hn.index('lowerarm_L')], 'hand': motor[hn.index('wrist_L')]}
    b = OLD._Body(rec['bodies']['3']['model'], header, ())
    m, d = b.m, b.d
    order = [('goal', gidx)] + [('s%d' % k, idx - 1 + k) for k in range(6)]
    poses = {n: np.array(data['qpos'][i]) for n, i in order}
    steps = {n: int(data['step'][i]) for n, i in order}
    jid = {jn: m.joint(jn).id for jn in JOINTS}
    bid = {'upper': m.body('left_upper_arm').id, 'fore': m.body('left_lower_arm').id, 'hand': m.body('left_hand').id, 'knuckle': m.body('left_mfknuckle').id}
    wrist = int(header['handle_bodies'][hn.index('wrist_L')])
    images, record = {}, {}
    try:
        b.set_frame(poses['s0'])
        b.fit([b.own_geoms(q) for q in poses.values()], [cmd_point])
        b.cams = {v: (look, dist * 1.12, az, el) for v, (look, dist, az, el) in b.cams.items()}
        b.place(poses['goal']); goal_hand = np.array(d.xpos[wrist])
        for n, q in poses.items():
            r = int(np.flatnonzero(zsteps == steps[n])[0])
            ctrl = z['ctrl'][r].astype(float)
            cm = np.flatnonzero(z['c_row'] == r)
            floor = [(ent[int(a)] if ent[int(bb)] == 'floor' else ent[int(bb)], float(z['c_fmax'][i]), z['c_pos'][i].astype(float).tolist())
                     for i, (a, bb) in zip(cm, z['c_e'][cm]) if 'floor' in (ent[int(a)], ent[int(bb)]) and (int(a) in own or int(bb) in own)]
            b.place(q)
            hand = np.array(d.xpos[wrist])
            rods = {}
            nxt = {'upper': np.array(d.xanchor[jid['robot:left_elbow']]), 'fore': np.array(d.xpos[bid['hand']]), 'hand': np.array(d.xpos[bid['knuckle']])}
            for part in ('upper', 'fore', 'hand'):
                o = np.array(d.xpos[bid[part]]); v = nxt[part] - o; L = float(np.linalg.norm(v))
                R_c = np.array(d.xmat[bid[part]]).reshape(3, 3); R_t = C.quat_xyzw_to_mat(req[part][3:7])
                local = R_c.T @ (v / L); want = R_t @ local
                ang = float(np.degrees(np.arccos(np.clip(np.dot(v / L, want), -1, 1))))
                rods[part] = {'origin_m': o.tolist(), 'actual_dir': (v / L).tolist(), 'requested_dir': want.tolist(), 'length_m': L, 'angle_between_deg': ang}
            record[n] = {'step': steps[n], 'controls': {jn: float(ctrl[act_index[ACT[jn]]]) for jn in JOINTS},
                         'floor_contacts': [{'part': p, 'force_max_n': f, 'pos_m': pos} for p, f, pos in floor],
                         'requested_direction_rods': rods, 'left_arm_touched_floor': any(p.startswith('left_') and ('arm' in p or 'hand' in p) for p, _, _ in floor)}
            for view in ('front', 'left', 'right'):
                name = b.draw('gizmo_' + n, view, q, None, None, out_dir=str(OUT))
                b.place(q)
                scene = b.renderer.scene; scene.ngeom = 0
                T = lambda p: b.Rf.T @ (np.asarray(p, dtype=float) - b.o)
                for jn in JOINTS:
                    j = jid[jn]; a = np.array(d.xanchor[j]); ax = np.array(d.xaxis[j])
                    c = 0.0 if n == 'goal' else min(1.0, abs(float(ctrl[act_index[ACT[jn]]])))
                    color = [0.35 + 0.60 * c, 0.35 * (1 - c), 0.35 * (1 - c), 1.0]
                    OLD._add_line(mj, scene, T(a - 0.045 * ax), T(a + 0.045 * ax), 0.0035, color)
                    OLD._add_geom(mj, scene, int(mj.mjtGeom.mjGEOM_SPHERE), [0.007, 0, 0], T(a), np.eye(3), DARK)
                if n != 'goal':
                    for part, rd in rods.items():
                        o = np.array(rd['origin_m']); OLD._add_line(mj, scene, T(o), T(o + rd['length_m'] * np.array(rd['requested_dir'])), 0.004, BLUE)
                    OLD._add_line(mj, scene, T(hand), T(cmd_point), 0.0012, BLUE)
                    OLD._add_geom(mj, scene, int(mj.mjtGeom.mjGEOM_SPHERE), [0.012, 0, 0], T(cmd_point), np.eye(3), BLUE)
                    OLD._add_line(mj, scene, T(hand), T(goal_hand), 0.0012, GREEN)
                OLD._add_geom(mj, scene, int(mj.mjtGeom.mjGEOM_SPHERE), [0.012, 0, 0], T(goal_hand), np.eye(3), GREEN)
                OLD._add_geom(mj, scene, int(mj.mjtGeom.mjGEOM_SPHERE), [0.012, 0, 0], T(hand), np.eye(3), ORANGE)
                for p, f, pos in floor:
                    OLD._add_geom(mj, scene, int(mj.mjtGeom.mjGEOM_SPHERE), [0.012, 0, 0], T(pos), np.eye(3), RED)
                overlay = np.array(b.renderer.render())
                base = np.array(Image.open(OUT / name).convert('RGB'))
                Image.fromarray(OLD._blend(base, overlay, alpha=1.0)).save(OUT / name)
                images[name] = sha(OUT / name)
        evidence = {'purpose': __doc__.strip().split('\n')[0], 'owner': 'the builder (the operator-requested revision of Oct 23, 2026)',
                    'recording': 'POSE_A_off_494025', 'body': 3, 'command_step': 494810, 'per_state': record,
                    'marks': {'joint_axis_rod': 'a 9 cm rod through the joint anchor along its axis, grey at control 0 and red at control 1 (the saved ctrl of the joint actuator at that model step; none on the goal)',
                              'requested_direction_rod': 'blue: the part’s long axis (origin to the next joint) turned to the orientation the command requested for that part',
                              'hand_points': 'orange: the hand point now; green: the hand point at the goal; blue sphere: the command’s requested hand position; thin lines join the hand point to both',
                              'floor_contact': 'yellow: every saved contact point between a body part and the floor at that model step (the floor itself is not drawn)'},
                    'images': images, 'sources': {str(found[ci][1]): sha(Path(found[ci][1]))},
                    'physics_steps': 0, 'training_updates': 0, 'seconds': time.monotonic() - started}
        (OUT / 'gizmo_detail.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({n: {'controls': {k.split(':')[1]: round(v, 2) for k, v in r['controls'].items()}, 'floor': [(c['part'], round(c['force_max_n'], 1)) for c in r['floor_contacts']],
                              'left_arm_touched_floor': r['left_arm_touched_floor'], 'rod_angles': {k: round(v['angle_between_deg'], 1) for k, v in r['requested_direction_rods'].items()}}
                          for n, r in record.items()}, indent=1))
        print('seconds', round(evidence['seconds'], 2))
    finally:
        b.close()


if __name__ == '__main__':
    main()
