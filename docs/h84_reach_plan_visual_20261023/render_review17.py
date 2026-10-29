"""After review 17 of the visual page: b3_rods.png redrawn with one more gizmo, the forearm direction the elbow CAN reach
with the upper arm as requested (purple rod from the elbow, with the angle between it and the requested F rod), beside
the red shoulder ring of render_review9.py. The reachable direction is computed, not estimated: the requested upper-arm
orientation and the requested forearm orientation are the command's own rows (the recording a4ba45b19cbd, body 3,
model step 494,810, read as docs/h84_body_repair_20261021/render_saved.py exact_image reads them); the elbow's nearest
angle inside its range comes from limb_target_control.one_axis_goal (the same closed form that produced the 57.8 degree
residual in provenance.json), and the forearm's orientation at that angle from one_axis_child_wxyz. The rod's direction
is that orientation applied to the forearm's segment axis (pose_geometry.segment_axes), the same axis the F rod uses.
Writes the picture and updates plan_pictures.json. Run from the repository root with .venv_mimo's python."""
import sys, json, math, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import render_plan as RP
import render_review9 as R9
from mimo_bridge import limb_target_control as LTC
R, G, C, mujoco = RP.R, RP.G, RP.C, RP.mujoco
RS = RP.RS


def reachable_forearm():
    rid, body_id, step = 'a4ba45b19cbd', 3, 494810
    rec = RS.S.by_id(rid)
    data, cols, used, found = RS.OLD.find_command(rec, body_id, step)
    idx = int(np.flatnonzero(data['step'] == step)[0]); ci = int(data['chunk'][idx]); row = int(data['row'][idx])
    motor = np.asarray(data['motors'][ci][row], float)
    header = data['headers'][0]; modelpath = Path(rec['bodies'][str(body_id)]['model'])
    model = C.BodyModel(str(modelpath), header); names = list(model.names); hb = np.asarray(model.hb); m = model.m if hasattr(model, 'm') else None
    if m is None:
        m = mujoco.MjModel.from_binary_path(str(modelpath)) if str(modelpath).endswith('.mjb') else RP.R._Body(str(modelpath), header, ()).m
    iu, il = names.index('upperarm_L'), names.index('lowerarm_L'); bu, bl = int(hb[iu]), int(hb[il])
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'robot:left_elbow'); assert int(m.jnt_bodyid[jid]) == bl
    axis = np.array(m.jnt_axis[jid], float); ref = float(m.qpos0[int(m.jnt_qposadr[jid])]); lo, hi = [float(x) for x in m.jnt_range[jid]]
    body_quat = np.array(m.body_quat[bl], float)
    q_up, q_fore = motor[iu, 3:7], motor[il, 3:7]
    angle, residual = LTC.one_axis_goal(q_up, q_fore, body_quat, axis, ref, lo, hi)
    child = LTC.one_axis_child_wxyz(LTC.unit_wxyz_from_xyzw(q_up), body_quat, axis, angle, ref)
    seg = G.segment_axes(m, None) if False else None
    own = [int(x) for x in hb]  # every handle body of this recording body
    axes = G.segment_axes(m, own); seg_axis, seg_len = axes[bl]
    w, x, y, z = [float(v) for v in child]; Rm = C.quat_xyzw_to_mat(np.array([x, y, z, w]))
    reach_dir = Rm @ np.asarray(seg_axis, float); reach_dir /= np.linalg.norm(reach_dir)
    req_dir = C.quat_xyzw_to_mat(q_fore) @ np.asarray(seg_axis, float); req_dir /= np.linalg.norm(req_dir)
    return dict(nearest_joint_deg=math.degrees(angle), residual_deg=math.degrees(residual), elbow_range_deg=[math.degrees(lo), math.degrees(hi)],
                reach_dir=reach_dir.tolist(), requested_dir_recomputed=req_dir.tolist(), dir_angle_deg=math.degrees(math.acos(max(-1.0, min(1.0, float(reach_dir @ req_dir))))),
                model=str(modelpath), seg_length_m=float(seg_len))


def main():
    rf = reachable_forearm()
    g = json.load(open(RP.GIZMO, encoding='utf-8')); fore = g['per_state']['s1']['requested_direction_rods']['fore']
    check = math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(fore['requested_dir']) @ np.asarray(rf['requested_dir_recomputed']))))))
    rf['check_requested_dir_vs_gizmo_detail_deg'] = check
    P = json.load(open(RP.OUT / 'plan_pictures.json', encoding='utf-8'))
    prov = json.load(open(RP.ROOT / 'docs/h84_body_repair_20261021/provenance.json', encoding='utf-8'))['exact_attached_image']['hinge_residuals']
    rf['provenance_residual_deg'] = [h['residual_deg'] for h in prov if h.get('child') == 'lowerarm_L'][0]
    tmp = tempfile.mkdtemp(prefix='review17_')
    origin = np.asarray(fore['origin_m'], float); length = float(fore['length_m']); reach = np.asarray(rf['reach_dir'], float); req = np.asarray(fore['requested_dir'], float)

    def extra_rod(d, bb):
        a = RP.project(bb, origin); e = RP.project(bb, origin + reach * length); f = RP.project(bb, origin + req * length)
        if a and e:
            d.line([a, e], fill=RP.PURPLE, width=4); d.ellipse([e[0] - 4, e[1] - 4, e[0] + 4, e[1] + 4], fill=RP.PURPLE)
            if f:
                # the arc from the F rod to the reachable rod, at a third of the rod's length from the elbow
                r = 0.33 * math.hypot(f[0] - a[0], f[1] - a[1]); t0 = math.atan2(f[1] - a[1], f[0] - a[0]); t1 = math.atan2(e[1] - a[1], e[0] - a[0])
                dt = (t1 - t0 + math.pi) % (2 * math.pi) - math.pi; pts = [(a[0] + r * math.cos(t0 + dt * k / 24), a[1] + r * math.sin(t0 + dt * k / 24)) for k in range(25)]
                d.line(pts, fill=RP.PURPLE, width=3)
                tm = t0 + dt / 2; mid = (a[0] + r * math.cos(tm), a[1] + r * math.sin(tm)); text = 'nearest reachable: %.1f° off' % rf['dir_angle_deg']; w_ = 7 * len(text) + 4
                x0, y0 = R.BODY_W - w_ - 6, 296  # the empty right side under the shoulder label, with a leader line to the arc's middle
                d.line([(x0, y0 + 9), mid], fill=RP.PURPLE, width=2)
                d.rectangle([x0, y0, x0 + w_, y0 + 18], fill=(255, 255, 255)); d.text((x0 + 1, y0), text, fill=RP.PURPLE, font=RP.FONT_S)
    b = R9.shoulder_gizmo(tmp, extra=extra_rod)
    b['reachable_forearm'] = rf
    P['numbers']['body3_POSE_A_off_494025']['b3_rods.png'] = b; P['images']['b3_rods.png'] = RP.sha(RP.OUT / 'b3_rods.png')
    json.dump(P, open(RP.OUT / 'plan_pictures.json', 'w', encoding='utf-8'), indent=1); print(json.dumps(rf))


if __name__ == '__main__':
    main()
