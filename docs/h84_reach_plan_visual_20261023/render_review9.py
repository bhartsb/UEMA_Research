"""Two additions after review 9 of the visual page: (1) cmp_rods_right_arm.png, the right arm of cmp_rods.png at 3x with
the turn from each blue rod tip (the original command's direction) to its purple rod tip (the corrected command's)
labelled in degrees; (2) b3_rods.png redrawn with one gizmo added: a red ring on body 3's left shoulder, the joint whose
horizontal motor the saved control holds at -1.0, its limit (gizmo_detail.json per_state.s1.controls). Same cameras as
render_plan.py (the same fit on the same saved states). Updates plan_pictures.json's image hashes.
Run from the repository root with .venv_mimo's python."""
import sys, json, math, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import render_plan as RP
R, G, C, mujoco = RP.R, RP.G, RP.C, RP.mujoco


def right_arm_inset(tmp):
    zc = np.load(RP.COMPARISON_REC, allow_pickle=False); header = json.load(open(RP.RS.HEADER, encoding='utf-8'))
    q = dict(target=np.asarray(zc['goal_qpos'], float), start=np.asarray(zc['start_qpos'], float), original_250ms=np.asarray(zc['qpos_original_A_hold250'][5], float), feasible_1000ms=np.asarray(zc['qpos_feasible_A_hold1000'][20], float))
    bm = C.BodyModel(str(RP.RS.MODEL), header); names = list(bm.names); hb = np.asarray(bm.hb)
    body = R._Body(str(RP.RS.MODEL), header, ())
    try:
        body.set_frame(q['start']); hp_t = RP.hand_points(body.m, q['target'])
        body.fit([body.own_geoms(q[k]) for k in ('start', 'target', 'original_250ms', 'feasible_1000ms')], [hp_t['L'], hp_t['R']])
        body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
        RP.render(body, 'cmp_rods_probe', q['start'], tmp)  # sets the renderer's camera for project()
        axes = G.segment_axes(body.m, body.own); dd = mujoco.MjData(body.m); dd.qpos[:] = q['start']; mujoco.mj_forward(body.m, dd)
        orig, corr = np.asarray(zc['original_motor'], float), np.asarray(zc['corrected_original'], float)
        moves, pts = [], []
        for hname in RP.ARM_HANDLES['R']:
            i = names.index(hname); bid = int(hb[i]); axis, length = axes.get(bid, (np.array([0.0, 0.0, 1.0]), 0.08)); o = np.array(dd.xpos[bid], float); k = 1.0 if 'wrist' not in hname else 3.0
            vo = C.quat_xyzw_to_mat(orig[i, 3:7]) @ np.asarray(axis, float); vc = C.quat_xyzw_to_mat(corr[i, 3:7]) @ np.asarray(axis, float)
            a, e, e2 = RP.project(body, o), RP.project(body, o + vo * float(length) * k), RP.project(body, o + vc * float(length) * k)
            deg = math.degrees(math.acos(max(-1.0, min(1.0, float(vo @ vc) / (np.linalg.norm(vo) * np.linalg.norm(vc))))))
            moves.append((e, e2, '%.1f°' % deg)); pts += [a, e, e2]
        hp = RP.project(body, RP.hand_points(body.m, q['start'])['R']); pts.append(hp)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]; cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 34; box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        # the crop's source: the same render with ONLY the right arm's rods and letters drawn (the left arm's letters would
        # otherwise be caught by the crop), plus the hand rings and T squares of draw_marks
        from PIL import Image, ImageDraw
        src = Path(tmp) / 'cmp_rods_right_only.png'
        def rods_right(d, bb):
            for hname in RP.ARM_HANDLES['R']:
                i = names.index(hname); bid = int(hb[i]); axis, length = axes.get(bid, (np.array([0.0, 0.0, 1.0]), 0.08)); o = np.array(dd.xpos[bid], float); k = 1.0 if 'wrist' not in hname else 3.0
                for row, col, w in ((orig, RP.BLUE, 5), (corr, RP.PURPLE, 3)):
                    v = C.quat_xyzw_to_mat(row[i, 3:7]) @ np.asarray(axis, float); e = RP.project(bb, o + v * float(length) * k); a = RP.project(bb, o)
                    if a and e:
                        d.line([a, e], fill=col, width=w); d.ellipse([e[0] - 4, e[1] - 4, e[0] + 4, e[1] + 4], fill=col)
                        if col == RP.BLUE:
                            letter = {'upperarm': 'U', 'lowerarm': 'F', 'wrist': 'H'}[hname.split('_')[0]]
                            Lr = math.hypot(e[0] - a[0], e[1] - a[1]) or 1.0; ux_, uy_ = (e[0] - a[0]) / Lr, (e[1] - a[1]) / Lr
                            vc_ = C.quat_xyzw_to_mat(corr[i, 3:7]) @ np.asarray(axis, float); e2 = RP.project(bb, o + vc_ * float(length) * k); side = 1.0
                            if e2:
                                cross = (e[0] - a[0]) * (e2[1] - a[1]) - (e[1] - a[1]) * (e2[0] - a[0]); side = -1.0 if cross > 0 else 1.0
                            lx, ly = e[0] + ux_ * 16 - uy_ * 12 * side, e[1] + uy_ * 16 + ux_ * 12 * side
                            d.rectangle([lx - 8, ly - 9, lx + 8, ly + 9], fill=(255, 255, 255)); d.text((lx - 6, ly - 9), letter, fill=RP.BLUE, font=RP.FONT)
        p0 = RP.render(body, 'cmp_rods_right_src', q['start'], tmp); RP.draw_marks(p0, body, q['start'], hp_t, out=src, extra=rods_right, label=False)
        info = RP.inset(src, 'cmp_rods_right_arm.png', box, 'right arm: blue, then purple', moves=moves, avoid=[(hp, 8)]); info['source'] = 'the cmp_rods render with only the right arm rods drawn'
        info['turned_deg'] = [m[2] for m in moves]; return info
    finally:
        body.close()


def shoulder_gizmo(tmp, extra=None):
    g = json.load(open(RP.GIZMO, encoding='utf-8')); s1 = g['per_state']['s1']; pdj = json.load(open(RP.PATHD, encoding='utf-8'))
    ctrl = s1['controls']['robot:left_shoulder_horizontal']
    q_t, header, _ = RP.qpos_at(3, 494806, (RP.POSEA,)); q_s = RP.qpos_at(3, 494809, (RP.POSEA,))[0]; q1 = RP.qpos_at(3, 494810, (RP.POSEA,))[0]; q_e = RP.qpos_at(3, 494814, (RP.POSEA,))[0]
    body = R._Body(str(RP.MODEL), header, ())
    try:
        body.set_frame(q_s); hp_t = RP.hand_points(body.m, q_t); hp_s = RP.hand_points(body.m, q_s)
        body.fit([body.own_geoms(q_s), body.own_geoms(q_t), body.own_geoms(q_e)], [hp_t['L'], hp_t['R'], hp_s['L'], hp_s['R']])
        body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
        ch = np.asarray(pdj['command_hand_point_world_m'], float)
        def rods(d, bb):
            for part, r in s1['requested_direction_rods'].items():
                o = np.asarray(r['origin_m'], float); a = RP.project(bb, o); e = RP.project(bb, o + np.asarray(r['requested_dir'], float) * float(r['length_m']) * (1.0 if part != 'hand' else 3.0))
                if a and e:
                    d.line([a, e], fill=RP.BLUE, width=5); d.ellipse([e[0] - 5, e[1] - 5, e[0] + 5, e[1] + 5], fill=RP.BLUE)
                    letter = {'upper': 'U', 'fore': 'F', 'hand': 'H'}[part]; dx, dy = {'upper': (8, -9), 'fore': (10, 6), 'hand': (-26, -42)}[part]
                    d.rectangle([e[0] + dx - 2, e[1] + dy - 1, e[0] + dx + 14, e[1] + dy + 17], fill=(255, 255, 255)); d.text((e[0] + dx, e[1] + dy - 1), letter, fill=RP.BLUE, font=RP.FONT)
            for c in s1.get('floor_contacts', []):
                cp = RP.project(bb, np.asarray(c['pos_m'], float))
                if cp: d.ellipse([cp[0] - 6, cp[1] - 6, cp[0] + 6, cp[1] + 6], fill=RP.YELLOW, outline=(120, 100, 0))
            cpx = RP.project(bb, ch); hpx = RP.project(bb, RP.hand_points(bb.m, q1)['L'])
            if cpx and hpx: d.line([hpx, cpx], fill=RP.BLUE, width=2); d.ellipse([cpx[0] - 7, cpx[1] - 7, cpx[0] + 7, cpx[1] + 7], fill=RP.BLUE, outline=(255, 255, 255), width=2)
            # the added gizmo: the left shoulder (the left upper arm body's origin), ringed red, with the saved control of its horizontal motor
            sh = RP.project(bb, RP.body_points(bb.m, q1, ('left_upper_arm',))['left_upper_arm'])
            if sh:
                d.ellipse([sh[0] - 14, sh[1] - 14, sh[0] + 14, sh[1] + 14], outline=RP.RED, width=4)
                lines_ = ['shoulder motor at %.1f' % ctrl, 'its limit']; w = 7 * max(len(x) for x in lines_) + 4
                x0 = min(sh[0] + 44, R.BODY_W - w - 4); y0 = min(sh[1] + 34, R.BODY_H - 40)  # right of and below the ring, clear of the rods
                d.line([(sh[0] + 10, sh[1] + 10), (x0 - 1, y0 + 9)], fill=RP.RED, width=2)  # the leader meets the box's left edge
                d.rectangle([x0, y0, x0 + w, y0 + 36], fill=(255, 255, 255))
                for k_, tx in enumerate(lines_): d.text((x0 + 1, y0 + 18 * k_), tx, fill=RP.RED, font=RP.FONT_S)
        def rods_plus(d, bb):
            rods(d, bb)
            if extra: extra(d, bb)
        p = RP.render(body, 'b3_rods', q1, tmp, 'front'); D, grav = RP.draw_marks(p, body, q1, hp_t, out=RP.OUT / 'b3_rods.png', extra=rods_plus, label=False)
        return dict(model_step=494810, view='front', D_cm=D, gravity=grav, shoulder_gizmo=dict(joint='robot:left_shoulder_horizontal', control=ctrl))
    finally:
        body.close()


def main():
    tmp = tempfile.mkdtemp(prefix='review9_')
    a = right_arm_inset(tmp); b = shoulder_gizmo(tmp)
    P = json.load(open(RP.OUT / 'plan_pictures.json', encoding='utf-8'))
    P['numbers']['comparison_483120']['pictures']['cmp_rods_right_arm.png'] = a; P['numbers']['body3_POSE_A_off_494025']['b3_rods.png'] = b
    for n in ('cmp_rods_right_arm.png', 'b3_rods.png'): P['images'][n] = RP.sha(RP.OUT / n)
    json.dump(P, open(RP.OUT / 'plan_pictures.json', 'w', encoding='utf-8'), indent=1); print(json.dumps(dict(inset=a, shoulder=b['shoulder_gizmo'])))


if __name__ == '__main__':
    main()
