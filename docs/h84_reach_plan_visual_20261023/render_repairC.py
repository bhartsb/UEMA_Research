"""Repair C's capture on the avatar: body 5 at the start of the decision of model step 496,260 (the planner's record
cap000496260 in chunk_000496261_3716_00004.npz), with the hand points the capture already holds: part 4's predicted hand
points for the first version of the command (hollow purple rings) and for the chosen version, the 21st (filled purple
dots), and the real hand points at the end of the sent command (dark dots, end_pose). Repair C would replay every one
of the 21 versions from this start and record its real end; the capture today holds the predictions of two of them and
the real end of one. Writes repairC_capture.png and repairC_capture.json beside the other pictures.
Run from the repository root with .venv_mimo's python."""
import sys, json, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import render_plan as RP
from PIL import ImageDraw

B, STEP = 5, 496260
K = 'cap000496260_'


def cm_of(a):
    a = np.asarray(a, dtype=np.float64); return a if np.abs(a).max() > 3 else a * 200.0


def crop(body, p, q, pt, hand_now, side):
    """The hand enlarged from the unmarked render: T (green square), the hand now (orange ring), version 1's predicted
    hand (hollow purple ring), version 21's (filled purple dot) and the real end of version 21 (dark dot)."""
    from PIL import Image
    import math
    P = {k: RP.project(body, v) for k, v in (('T', pt['T']), ('now', hand_now), ('v1', pt['v1']), ('v21', pt['v21']), ('end', pt['end']))}
    near = [P[k] for k in ('now', 'v1', 'v21', 'end')]  # the crop is centred on the hand's marks; T is drawn if it falls inside, else an arrow points to it
    cx, cy = sum(q_[0] for q_ in near) / 4, sum(q_[1] for q_ in near) / 4
    half = max(45.0, max(math.hypot(q_[0] - cx, q_[1] - cy) for q_ in near) + 24.0); s = 360.0 / (2 * half)
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    im = Image.open(p).convert('RGB').crop(box).resize((360, 360), Image.LANCZOS); d = ImageDraw.Draw(im)
    I = {k: ((v[0] - box[0]) * s, (v[1] - box[1]) * s) for k, v in P.items()}
    sq = 4 * s; t = I['T']
    if 0 <= t[0] < 360 and 0 <= t[1] < 360: d.rectangle([t[0] - sq, t[1] - sq, t[0] + sq, t[1] + sq], fill=RP.GREEN, outline=(255, 255, 255))
    else:
        n0 = I['now']; vx, vy = t[0] - n0[0], t[1] - n0[1]; L = math.hypot(vx, vy) or 1; RP.arrow(d, (n0[0] + vx / L * 9 * s, n0[1] + vy / L * 9 * s), (n0[0] + vx / L * (9 * s + 90), n0[1] + vy / L * (9 * s + 90)), RP.GREEN, 3)
    r = 8 * s; n = I['now']; d.ellipse([n[0] - r, n[1] - r, n[0] + r, n[1] + r], outline=RP.ORANGE, width=max(3, int(round(0.75 * s))))
    r1 = 4 * s; a = I['v1']; d.ellipse([a[0] - r1, a[1] - r1, a[0] + r1, a[1] + r1], outline=RP.PURPLE, width=max(3, int(round(0.75 * s))))
    b = I['v21']; d.ellipse([b[0] - r1, b[1] - r1, b[0] + r1, b[1] + r1], fill=RP.PURPLE, outline=(255, 255, 255))
    r2 = 3 * s; e = I['end']; d.ellipse([e[0] - r2, e[1] - r2, e[0] + r2, e[1] + r2], fill=(40, 40, 40), outline=(255, 255, 255))
    RP.dashed(d, b, e, (40, 40, 40), width=max(2, int(round(0.5 * s))), dash=int(3 * s))
    labels = ['purple ring: version 1 predicted, %.1f cm from T' % (np.linalg.norm(pt['v1'] - pt['T']) * 100), 'purple dot: version 21 predicted, %.1f cm' % (np.linalg.norm(pt['v21'] - pt['T']) * 100),
              'dark dot: real end of version 21, %.1f cm' % (np.linalg.norm(pt['end'] - pt['T']) * 100)] + ([] if 0 <= I['T'][0] < 360 and 0 <= I['T'][1] < 360 else ['green arrow: toward T'])
    nl = len(labels); w = 7 * max(len(x) for x in labels) + 6; d.rectangle([4, 356 - nl * 18 - 4, 4 + w, 356], fill=(255, 255, 255))
    for k, (text, col) in enumerate(zip(labels, (RP.PURPLE, RP.PURPLE, (40, 40, 40), RP.GREEN))): d.text((7, 356 - nl * 18 - 2 + 18 * k), text, fill=col, font=RP.FONT_S)
    name = {'L': 'left', 'R': 'right'}[side]; title = '%s hand, %.1fx' % (name, s)
    d.rectangle([0, 0, 359, 359], outline=RP.PURPLE, width=4); d.rectangle([4, 4, 12 + 8 * len(title), 26], fill=(255, 255, 255)); d.text((8, 6), title, fill=RP.PURPLE, font=RP.FONT)
    out = 'repairC_capture_hand_%s.png' % side; im.save(RP.OUT / out)
    return dict(picture=out, crop_box_px=list(box), scale=round(s, 2), labels=labels)


def main():
    cap = np.load(RP.CAPTURE, allow_pickle=True)
    rows = {'L': 0, 'R': 6}
    pts = {}
    for side, i in rows.items():
        pts[side] = dict(T=cm_of(cap[K + 'goal_pose'][B, i, :3]) / 100.0, v1=np.asarray(cap[K + 'forecast_pose__iteration0_position_cm'][B, i], dtype=np.float64) / 100.0,
                         v21=np.asarray(cap[K + 'forecast_pose__selected_position_cm'][B, i], dtype=np.float64) / 100.0, end=cm_of(cap[K + 'end_pose'][B, i, :3]) / 100.0,
                         start=cm_of(cap[K + 'start_pose'][B, i, :3]) / 100.0)
    q_start, header, f = RP.qpos_at(B, STEP - 1)
    body = RP.R._Body(str(RP.MODEL), header, ()); tmp = tempfile.mkdtemp(prefix='repairC_')
    try:
        body.set_frame(q_start); hp_t = {s: pts[s]['T'] for s in 'LR'}
        body.fit([body.own_geoms(q_start)], [hp_t['L'], hp_t['R'], pts['L']['end'], pts['R']['end']])
        body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
        hands_now = RP.hand_points(body.m, q_start)
        def extra(d, bb):
            for s in 'LR':
                p1, p21, pe = (RP.project(bb, pts[s][k]) for k in ('v1', 'v21', 'end'))
                if p1: d.ellipse([p1[0] - 6, p1[1] - 6, p1[0] + 6, p1[1] + 6], outline=RP.PURPLE, width=3)
                if p21: d.ellipse([p21[0] - 6, p21[1] - 6, p21[0] + 6, p21[1] + 6], fill=RP.PURPLE, outline=(255, 255, 255))
                if pe: d.ellipse([pe[0] - 5, pe[1] - 5, pe[0] + 5, pe[1] + 5], fill=(40, 40, 40), outline=(255, 255, 255))
            W = RP.R.BODY_W; Y = RP.R.BODY_H - 100; d.rectangle([W - 300, Y - 1, W - 6, Y + 87], fill=(255, 255, 255))  # lower right, clear of the body
            d.text((W - 296, Y + 0), 'purple ring: predicted hand, version 1 of 21', fill=RP.PURPLE, font=RP.FONT_S)
            d.text((W - 296, Y + 17), 'purple dot: predicted hand, version 21 (sent)', fill=RP.PURPLE, font=RP.FONT_S)
            d.text((W - 296, Y + 34), 'dark dot: the real hand at the end of version 21', fill=(40, 40, 40), font=RP.FONT_S)
            d.text((W - 296, Y + 51), 'orange ring: the hand now; green square: T', fill=RP.ORANGE, font=RP.FONT_S)
            d.text((W - 296, Y + 68), 'repair C records the real end of all 21', fill=(40, 40, 40), font=RP.FONT_S)
        p = RP.render(body, 'repairC_capture', q_start, tmp)
        D, grav = RP.draw_marks(p, body, q_start, hp_t, out=RP.OUT / 'repairC_capture.png', extra=extra)
        crops = {s: crop(body, p, q_start, pts[s], hands_now[s], s) for s in 'LR'}
    finally:
        body.close()
    num = {s: dict(hand_now_to_T_cm=round(float(np.linalg.norm(hands_now[s] - pts[s]['T']) * 100), 2), v1_predicted_to_T_cm=round(float(np.linalg.norm(pts[s]['v1'] - pts[s]['T']) * 100), 2),
                   v21_predicted_to_T_cm=round(float(np.linalg.norm(pts[s]['v21'] - pts[s]['T']) * 100), 2), real_end_to_T_cm=round(float(np.linalg.norm(pts[s]['end'] - pts[s]['T']) * 100), 2),
                   real_end_to_v21_predicted_cm=round(float(np.linalg.norm(pts[s]['end'] - pts[s]['v21']) * 100), 2), frame_check_start_row_vs_hand_now_cm=round(float(np.linalg.norm(pts[s]['start'] - hands_now[s]) * 100), 2)) for s in 'LR'}
    out = dict(crops=crops, capture=str(RP.CAPTURE), body=B, decision_step=STEP, start_world_record=STEP - 1, start_chunk=str(f), gravity=grav, D_cm=D, numbers=num)
    json.dump(out, open(RP.OUT / 'repairC_capture.json', 'w', encoding='utf-8'), indent=1); print(json.dumps(num))


if __name__ == '__main__':
    main()
