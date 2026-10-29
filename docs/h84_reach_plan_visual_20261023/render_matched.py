"""Repair E's own result on the avatar: the matched comparison of October 24 (two runs from the saved state of
model step 496,250, the target-candidate switch off and on, 6 model steps each). For each body the candidate was
selected in (0, 1, 5 and 6), the body at model step 496,255 (250 ms after the decision) under the switch-off run and
under the switch-on run, in one camera frame per body, with T's hand points (the decision's goal pose, the same in
both runs) as green squares and the hand points as orange rings. Gaps in the captions come from the corrected reading
(corrected_reading_result.json); the marks here are drawn from the saved joint positions.

Run with .venv_mimo's python from this folder's parent repository. Writes matched_b{b}_{start,off,on}.png and
matched_pictures.json beside the other pictures."""
import sys, json, tempfile, os
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_plan as RP

OFF = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/target_matched_496250_s2_off_a1/evidence/run/TMATCH_off_496250/observer')
ON = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/target_matched_496250_s2_on_a1/evidence/run/TMATCH_on_496250/observer')
READING = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/target_matched_496250_package_s2_reread/corrected_reading_result.json')
BODIES = (0, 1, 5, 6)
STEP_DECISION, STEP_READ = 496250, 496255


def goal_hand_points(cap_dir):
    f = sorted(cap_dir.glob('captures_cap000496250_*.npz'))[0]
    z = np.load(f, allow_pickle=True); gp = np.asarray(z['cap000496250_goal_pose'], dtype=np.float64)  # fractions of 200 cm, world
    return {b: {'L': gp[b, 0, :3] * 2.0, 'R': gp[b, 6, :3] * 2.0} for b in range(8)}, str(f)


def crop_hand(body, p_on, q_off, q_on, hp_t, b, side):
    """One enlarged crop per hand from the switch-on run's unmarked render at 250 ms: T's hand point (green square), the
    hand point under the switch-off run (grey ring, dashed grey line to T) and under the switch-on run (orange ring, dashed
    orange line to T), each labelled with its gap in cm (the 3-D distance between hand point and T's hand point)."""
    from PIL import Image, ImageDraw
    import math
    h_off, h_on, t = RP.hand_points(body.m, q_off)[side], RP.hand_points(body.m, q_on)[side], hp_t[side]
    d_off, d_on = float(np.linalg.norm(h_off - t) * 100), float(np.linalg.norm(h_on - t) * 100)
    P = [RP.project(body, x) for x in (h_off, h_on, t)]
    cx, cy = sum(q[0] for q in P) / 3, sum(q[1] for q in P) / 3
    half = max(45.0, max(math.hypot(q[0] - cx, q[1] - cy) for q in P) + 24.0); s = 360.0 / (2 * half)
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    im = Image.open(p_on).convert('RGB').crop(box).resize((360, 360), Image.LANCZOS); d = ImageDraw.Draw(im)
    I = lambda q: ((q[0] - box[0]) * s, (q[1] - box[1]) * s)
    po, pn, pt = (I(q) for q in P); lw = max(2, int(round(0.6 * s)))
    RP.dashed(d, po, pt, RP.GREY, width=lw, dash=int(4 * s)); RP.dashed(d, pn, pt, RP.ORANGE, width=lw, dash=int(4 * s))
    sq = 4 * s; d.rectangle([pt[0] - sq, pt[1] - sq, pt[0] + sq, pt[1] + sq], fill=RP.GREEN, outline=(255, 255, 255))
    r = 8 * s; d.ellipse([po[0] - r, po[1] - r, po[0] + r, po[1] + r], outline=RP.GREY, width=max(3, int(round(0.75 * s))))
    d.ellipse([pn[0] - r, pn[1] - r, pn[0] + r, pn[1] + r], outline=RP.ORANGE, width=max(3, int(round(0.75 * s))))
    labels = ['grey ring: switch off, %.1f cm from T' % d_off, 'orange ring: switch on, %.1f cm from T' % d_on, 'green square: T']
    w = 7 * max(len(x) for x in labels) + 6; d.rectangle([4, 356 - 3 * 18 - 4, 4 + w, 356], fill=(255, 255, 255))
    for k, (text, col) in enumerate(zip(labels, (RP.GREY, RP.ORANGE, RP.GREEN))): d.text((7, 356 - 3 * 18 - 2 + 18 * k), text, fill=col, font=RP.FONT_S)
    name = {'L': 'left', 'R': 'right'}[side]; title = '%s hand, %.1fx' % (name, s); col = RP.DARK  # a neutral frame: orange and green borders carry other meanings on the page
    d.rectangle([0, 0, 359, 359], outline=col, width=4); d.rectangle([4, 4, 12 + 8 * len(title), 26], fill=(255, 255, 255)); d.text((8, 6), title, fill=col, font=RP.FONT)
    out = 'matched_b%d_hand_%s.png' % (b, side); im.save(RP.OUT / out)
    return dict(picture=out, source='switch-on run at 250 ms, unmarked render', crop_box_px=list(box), scale=round(s, 2), gap_off_cm=round(d_off, 2), gap_on_cm=round(d_on, 2), labels=labels, on_nearer=d_on < d_off)


def main():
    reading = json.load(open(READING, encoding='utf-8'))
    by_body = {r['body']: r for r in reading['bodies']}
    hp_goal, cap_file = goal_hand_points(OFF / 'model')
    out = {'source_runs': {'off': str(OFF), 'on': str(ON)}, 'reading': str(READING), 'goal_capture': cap_file, 'bodies': {}}
    tmp = tempfile.mkdtemp(prefix='matched_')
    for b in BODIES:
        q_start, header, _ = RP.qpos_at(b, STEP_DECISION, (OFF / 'world',))
        q_off = RP.qpos_at(b, STEP_READ, (OFF / 'world',))[0]; q_on = RP.qpos_at(b, STEP_READ, (ON / 'world',))[0]
        body = RP.R._Body(str(RP.MODEL), header, ()); rec = {}
        try:
            body.set_frame(q_start); hp_t = hp_goal[b]
            body.fit([body.own_geoms(q_start), body.own_geoms(q_off), body.own_geoms(q_on)], [hp_t['L'], hp_t['R']])
            body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
            for name, q in (('start', q_start), ('off', q_off), ('on', q_on)):
                p = RP.render(body, 'matched_b%d_%s' % (b, name), q, tmp)
                D, grav = RP.draw_marks(p, body, q, hp_t, out=RP.OUT / ('matched_b%d_%s.png' % (b, name)), label=False)
                rec[name] = dict(D_cm_by_picture=D, gravity=grav)
                if name == 'on': p_on = p
            rec['hands'] = {side: crop_hand(body, p_on, q_off, q_on, hp_t, b, side) for side in 'LR'}
        finally:
            body.close()
        r = by_body[b]
        rec['reading'] = {k: dict(gap_cm=r[k]['actual_hand_distance_cm'], pose_error=r[k]['actual_pose_error']['sum']) for k in ('off', 'on')}
        rec['candidate'] = dict(selected=r['candidate']['selected'], outcome=r['candidate']['outcome'])
        out['bodies'][str(b)] = rec
    json.dump(out, open(RP.OUT / 'matched_pictures.json', 'w', encoding='utf-8'), indent=1)
    print(json.dumps({b: {k: {s: round(v, 1) for s, v in rec[k]['D_cm_by_picture'].items()} for k in ('start', 'off', 'on')} for b, rec in out['bodies'].items()}))


if __name__ == '__main__':
    main()
