"""Pictures for the visual reach plan. Real saved MIMo states are drawn by the committed renderer (render_saved.py's
modules: the body in its own trunk coordinates, floor not drawn, front view: the picture's right is the infant's left)
with marks added: this pose's hand points (orange rings; the hand body's origin, inside the palm), the target's hand
points (green squares), a dashed line between each pair, and a grey "down" arrow whose length is gravity's share in
the picture plane (the label gives its angle from the line of sight). No physics stepping, no training. Every number
is written to plan_pictures.json; the page prints those numbers.

Target records: the planner labels its target with a model step (three_way_72.json goal_step = plan step minus age);
the diagnosis of October 23 verified that the world record holding that pose is ONE MODEL STEP EARLIER
(analysis/result.json cases: world_record_step_inferred = model_step_inferred - 1), so the target picture and the D
values use goal_step - 1. The start record is the record before the plan step (plan step - 1), as in the diagnosis.

Sources drawn: the run of October 23 (a2) and its parent run (targets); run POSE_A_off_494025 body 3; the baseline of
October 23 (reach_baseline_8start_a1: records.npz, result.json); the two October 21 tests on run seg41j body 5
(limb_control_483120_a1 and comparison_483120_a2: records.npz); the diagnosis's body 5 example (examples.json:
predicted part positions and the fitted arm), drawn here over the real saved states with this file's marks.

Run from the repository root: .venv_mimo/Scripts/python.exe -B docs/h84_reach_plan_visual_20261023/render_plan.py
"""
import sys, os, json, glob, hashlib, importlib.util, tempfile, time, math, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Source/Python'))
spec = importlib.util.spec_from_file_location('render_saved_for_plan', str(ROOT / 'docs/h84_body_repair_20261021/render_saved.py'))
RS = importlib.util.module_from_spec(spec); _cwd = os.getcwd(); spec.loader.exec_module(RS); os.chdir(_cwd)
R, G, C, mujoco = RS.R, RS.G, RS.C, RS.mujoco
from mimo_bridge import observer_record as OR

A2 = Path('D:/UEMA_Research/external/D/UEMA_preserved/s4r_pose_learning_S4R_pose_learning_a2_496250/run/observer/world')
PARENT = Path('D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run/observer/world')
POSEA = Path('D:/UEMA_Research/external/D/UEMA_preserved/pose_comparison_POSE_A_off_494025/run/observer/world')
MODEL = A2 / 'models/5aff7b57135ab6431aae071e6a4a6680.mjb.gz'
LIMB = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/whole_body_request/attempts/limb_control_483120_a1/evidence/records.npz')
COMPARISON_REC = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/whole_body_request/attempts/comparison_483120_a2/evidence/records.npz')
COMPARISON_RES = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/whole_body_request/attempts/comparison_483120_a2/evidence/result.json')
BASELINE = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/reach_baseline_8start_a1/evidence')
EXAMPLES = Path('D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/a2_record_diagnosis_d1/evidence/analysis/examples.json')
CAPTURE = Path('D:/UEMA_Research/external/D/UEMA_preserved/s4r_pose_learning_S4R_pose_learning_a2_496250/run/observer/model/chunk_000496261_3716_00004.npz')  # the planner's record of the decision at model step 496,260
GIZMO = ROOT / 'docs/h84_body_repair_v2_20261023/gizmo_detail.json'
PATHD = ROOT / 'docs/h84_body_repair_v2_20261023/path_detail.json'
THREE_WAY = ROOT / 'docs/h84_body_repair_v2_20261023/three_way_72.json'
ORANGE, GREEN, PURPLE, DARK, GREY, BLUE, YELLOW, RED, FIT = (195, 75, 17), (8, 120, 62), (122, 63, 176), (24, 48, 68), (110, 120, 128), (23, 101, 186), (230, 190, 0), (200, 30, 30), (95, 111, 122)
FONT = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 15); FONT_S = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 13)
ARM_BODIES = {'L': ('left_upper_arm', 'left_lower_arm', 'left_hand'), 'R': ('right_upper_arm', 'right_lower_arm', 'right_hand')}
ARM_HANDLES = {'L': ('upperarm_L', 'lowerarm_L', 'wrist_L'), 'R': ('upperarm_R', 'lowerarm_R', 'wrist_R')}

_chunks = {}
def chunk_for(body, step, bases):
    for base in bases:
        folder = base / ('b%d' % body)
        files = sorted(glob.glob(str(folder / 'chunk_*.npz')))
        cands = [f for f in files if not f.endswith('.part.npz') and int(os.path.basename(f).split('_')[1]) <= step]
        for f in reversed(cands[-3:]):
            if f not in _chunks: _chunks[f] = OR.read_chunk(f)
            c = _chunks[f]; steps = np.asarray(c['step']).reshape(-1).tolist()
            if step in steps: return c, f, steps.index(step)
    raise RuntimeError('no world record for body %d at model step %d' % (body, step))

def qpos_at(body, step, bases=(A2, PARENT)):
    c, f, i = chunk_for(body, step, bases)
    return np.asarray(c['qpos'][i], dtype=np.float64), c['header'], f

def body_id(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0: raise RuntimeError('no body named ' + name)
    return i

def body_points(m, q, names=('left_hand', 'right_hand', 'head')):
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    return {n: np.array(d.xpos[body_id(m, n)], dtype=np.float64) for n in names}

def hand_points(m, q):
    p = body_points(m, q); return {'L': p['left_hand'], 'R': p['right_hand']}

def project(body, point):
    cams = body.renderer.scene.camera
    cp = (np.array(cams[0].pos) + np.array(cams[1].pos)) / 2
    f = np.array(cams[0].forward); u = np.array(cams[0].up); r = np.cross(f, u)
    v = body.Rf.T @ (np.asarray(point, dtype=np.float64) - body.o) - cp
    z = float(v @ f)
    if not z > 0: return None
    near = float(cams[0].frustum_near); half_h = (float(cams[0].frustum_top) - float(cams[0].frustum_bottom)) / 2
    nx = float(v @ r) / z * near / (half_h * R.BODY_W / R.BODY_H); ny = float(v @ u) / z * near / half_h
    return ((nx + 1) * R.BODY_W / 2, (1 - ny) * R.BODY_H / 2)

def cam_axes_world(body):
    cams = body.renderer.scene.camera
    f = np.array(cams[0].forward); u = np.array(cams[0].up); r = np.cross(f, u)
    return body.Rf @ f, body.Rf @ u, body.Rf @ r

def dashed(d, a, b, fill, width=2, dash=7):
    (x0, y0), (x1, y1) = a, b; L = math.hypot(x1 - x0, y1 - y0)
    if L < 1: return
    n = int(L // dash); ux, uy = (x1 - x0) / L, (y1 - y0) / L
    for k in range(0, n, 2):
        d.line([(x0 + ux * k * dash, y0 + uy * k * dash), (x0 + ux * min((k + 1) * dash, L), y0 + uy * min((k + 1) * dash, L))], fill=fill, width=width)

def arrow(d, a, b, fill, width=3, head=10):
    d.line([a, b], fill=fill, width=width); ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (0.5, -0.5):
        d.line([b, (b[0] - head * math.cos(ang + s), b[1] - head * math.sin(ang + s))], fill=fill, width=width)

def plane_share(body, p, v):
    """How much of the world vector v (m), placed at world point p, lies in the picture plane: measured from pixels.
    The three world axes projected at p have lengths l_x, l_y, l_z = k sin(theta_i); for orthonormal axes
    sum(sin^2) = 2, so k = sqrt(sum(l_i^2) / 2) is the full picture-plane scale at p, and the share of v is
    |proj(p + v) - proj(p)| / (k |v| / L). Returns (share, picture-plane direction as a unit pixel vector)."""
    L = 0.08; p0 = project(body, p)
    ls = [np.linalg.norm(np.subtract(project(body, p + e * L), p0)) for e in (np.array([L / L, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))]
    k = math.sqrt(sum(l * l for l in ls) / 2.0) or 1e-9
    pv = np.subtract(project(body, p + v), p0); n = float(np.linalg.norm(pv))
    share = min(1.0, n / (k * float(np.linalg.norm(v)) / L)) if np.linalg.norm(v) > 0 else 0.0
    return share, (pv / n if n > 1e-9 else np.array([0.0, 1.0]))

def gravity_info(body):
    """Gravity (world -z) in the camera: its angle from the line of sight (degrees, from the picture-plane share
    measured in pixels at the head), its picture-plane direction and that share."""
    o = body_points(body.m, getattr(body, '_q_for_gravity', None) if getattr(body, '_q_for_gravity', None) is not None else np.array(body.d.qpos), ('head',))['head']
    share, direction = plane_share(body, o, np.array([0.0, 0.0, -0.08]))
    ang = math.degrees(math.asin(min(1.0, share))); return ang, direction, share

def gravity_arrow(d, body, origin, size, label=True, q=None):
    """Top-left corner: the label (full-size pictures only), then the arrow below it (its length = gravity's share in the
    picture plane)."""
    body._q_for_gravity = q; ang, direction, share = gravity_info(body); L = max(8.0, 44.0 * share)
    if label:
        text = 'down: %.0f\u00b0 from the line of sight' % ang
        d.rectangle([4, 4, 12 + 7 * len(text), 24], fill=(255, 255, 255)); d.text((8, 6), text, fill=GREY, font=FONT_S)
    a = (34, 52) if label else (26, 26); b = (a[0] + direction[0] * L, a[1] + direction[1] * L); arrow(d, a, b, GREY, 3)
    return dict(angle_from_line_of_sight_deg=round(ang, 1), picture_plane_share=round(share, 3))

def draw_marks(png, body, q, target_pts, ring_cm=None, out=None, crop=None, extra=None, label=True, ring_label=True):
    im = Image.open(png).convert('RGB'); d = ImageDraw.Draw(im)
    pts = hand_points(body.m, q); D = {}
    for s in 'LR':
        p, t = project(body, pts[s]), project(body, target_pts[s]); D[s] = float(np.linalg.norm(pts[s] - target_pts[s]) * 100)
        if p and t: dashed(d, p, t, GREEN)
        if t: d.rectangle([t[0] - 6, t[1] - 6, t[0] + 6, t[1] + 6], fill=GREEN, outline=(255, 255, 255))
        if p: d.ellipse([p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8], outline=ORANGE, width=3)
        if ring_cm and t:
            _, _, rw = cam_axes_world(body); off = project(body, target_pts[s] + rw * ring_cm / 100.0); rad = abs(off[0] - t[0]) if off else 0
            d.ellipse([t[0] - rad, t[1] - rad, t[0] + rad, t[1] + rad], outline=(5, 70, 40), width=3)
            if ring_label: d.text((t[0] + rad + 4, t[1] - 9), '2 cm', fill=(5, 70, 40), font=FONT_S)
    if extra: extra(d, body)
    if crop:
        im = im.crop(crop); d = ImageDraw.Draw(im)
    grav = gravity_arrow(d, body, (40, im.size[1] - 70), im.size, label=(label and not crop), q=q)
    im.save(out or png); return D, grav

def render(body, label, q, tmp, view='front'):
    return Path(tmp) / body.draw(label, view, q, None, None, out_dir=str(tmp))

def new_body(model, header, q_frame, qs, dots):
    body = R._Body(str(model), header, ()); body.set_frame(q_frame)
    body.fit([body.own_geoms(q) for q in qs], dots)
    body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
    return body

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inset(src, dst, box, label, moves=None, avoid=()):
    """An enlarged crop; `moves` = [(from_px, to_px, text)] in the source picture's pixels, drawn as arrows with heads
    scaled to the move and the text beside each."""
    im = Image.open(OUT / src).convert('RGB').crop(box); s = 340 / (box[2] - box[0]); im = im.resize((340, int((box[3] - box[1]) * s)), Image.LANCZOS); d = ImageDraw.Draw(im)
    if moves:
        for a, b, text in moves:
            a2 = ((a[0] - box[0]) * s, (a[1] - box[1]) * s); b2 = ((b[0] - box[0]) * s, (b[1] - box[1]) * s); L = math.hypot(b2[0] - a2[0], b2[1] - a2[1])
            if L >= 3: arrow(d, a2, b2, PURPLE, 2, head=max(4, min(10, 0.45 * L)))
            w = 7 * len(text) + 4; clear = 8 * s + 6; W, H = im.size  # the largest ring at the end (the hand ring, radius 8 at 1x) plus a gap
            ux, uy = ((b2[0] - a2[0]) / L, (b2[1] - a2[1]) / L) if L >= 3 else (1.0, 0.0)
            # candidate places: beyond the end ring on the move's far side first, then right, left, above, below; the first one
            # that, once kept inside the picture, covers neither the end ring nor the start ring is used
            cands = [(b2[0] + clear if ux >= 0 else b2[0] - clear - w, b2[1] - 9)] if abs(ux) >= abs(uy) else [(b2[0] - w / 2, b2[1] + clear if uy >= 0 else b2[1] - clear - 17)]
            cands += [(b2[0] + clear, b2[1] - 9), (b2[0] - clear - w, b2[1] - 9), (b2[0] - w / 2, b2[1] - clear - 17), (b2[0] - w / 2, b2[1] + clear)]
            clamp = lambda c: (min(max(4, c[0]), W - w - 4), min(max(4, c[1]), H - 22))
            def dist_box(c, pt):
                dx = max(c[0] - pt[0], 0, pt[0] - (c[0] + w)); dy = max(c[1] - pt[1], 0, pt[1] - (c[1] + 17)); return math.hypot(dx, dy)
            av = [(((pt[0] - box[0]) * s, (pt[1] - box[1]) * s), r * s + 2) for pt, r in avoid if pt]  # every other mark in the crop: (centre, radius) in the inset's pixels
            x0, y0 = clamp(cands[0])
            for c in cands:
                cc = clamp(c)
                if dist_box(cc, b2) > 8 * s + 2 and dist_box(cc, a2) > 6 * s + 2 and all(dist_box(cc, q) > rr for q, rr in av): x0, y0 = cc; break
            d.rectangle([x0, y0, x0 + w, y0 + 17], fill=(255, 255, 255)); d.text((x0 + 1, y0), text, fill=PURPLE, font=FONT_S)
    d.rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], outline=PURPLE, width=4); d.rectangle([4, 4, 12 + 8 * len(label), 26], fill=(255, 255, 255)); d.text((8, 6), label, fill=PURPLE, font=FONT)
    im.save(OUT / dst); return dict(source=src, crop_box_px=list(box), scale=round(s, 2))

def main():
    t0 = time.time(); TW = json.load(open(THREE_WAY)); tmp = Path(tempfile.mkdtemp(prefix='reach_plan_render_'))
    rec = dict(purpose=__doc__.strip(), physics_steps=0, training_updates=0, started=time.strftime('%Y-%m-%d %H:%M:%S'),
               renderer_commit=RS.COMMIT, model=str(MODEL), target_record_rule='goal_step - 1 (the diagnosis: world_record_step_inferred = model_step_inferred - 1)', images={}, numbers={}, gravity={})
    try:
        # ---- A. the eight starts beside their targets (run of October 23)
        pairs = {}
        for b in range(8):
            tr = TW['trajectories'][str(b)][0]; plan, goal = int(tr['plan_step']), int(tr['goal_step']) - 1
            q_s, header, f_s = qpos_at(b, plan - 1); q_t, _, f_t = qpos_at(b, goal)
            body = R._Body(str(MODEL), header, ())
            try:
                body.set_frame(q_s); hp_s, hp_t = hand_points(body.m, q_s), hand_points(body.m, q_t)
                body.fit([body.own_geoms(q_s), body.own_geoms(q_t)], [hp_s['L'], hp_s['R'], hp_t['L'], hp_t['R']])
                body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
                p = render(body, 'start_b%d' % b, q_s, tmp); Ds, grav = draw_marks(p, body, q_s, hp_t, out=Path(tmp) / ('start_b%d.png' % b))
                if b == 1:
                    shutil.copy(Path(tmp) / 'start_b1.png', OUT / 'start_b1.png')
                    p = render(body, 'done_b1', q_t, tmp); _, gdone = draw_marks(p, body, q_t, hp_t, ring_cm=2.0, out=OUT / 'done_b1.png'); pairs.setdefault('done_b1', {})['gravity'] = gdone
                pairs[b] = dict(start_step=plan - 1, target_step=goal, planner_target_label=goal + 1, D_start_cm=Ds, three_way_start_cm=dict(L=tr['L_start'], R=tr['R_start']), gravity=grav, start_chunk=f_s, target_chunk=f_t)
            finally:
                body.close()
        rec['numbers']['pairs'] = pairs
        # ---- B. body 1, every command of the run (same target for all 9 planner commands): a consistent upper-body crop
        b = 1; goal = int(TW['trajectories']['1'][0]['goal_step']) - 1; assert all(int(t['goal_step']) - 1 == goal for t in TW['trajectories']['1'])
        steps = list(range(496259, 496345, 5)); qs = {s: qpos_at(b, s)[0] for s in steps}; q_t, header, _ = qpos_at(b, goal)
        body = R._Body(str(MODEL), header, ()); seq = []
        try:
            body.set_frame(qs[steps[0]]); hp_t = hand_points(body.m, q_t)
            body.fit([body.own_geoms(qs[s]) for s in steps[::3]] + [body.own_geoms(q_t)], [hp_t['L'], hp_t['R']])
            body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
            frames = {s: render(body, 'seq_b1_%d' % s, qs[s], tmp) for s in steps}
            allp = []
            for s in steps:
                bp = body_points(body.m, qs[s]); allp += [project(body, bp[n]) for n in bp]
            allp += [project(body, hp_t['L']), project(body, hp_t['R'])]; allp = [p for p in allp if p]
            crop = (int(max(0, min(p[0] for p in allp) - 70)), int(max(0, min(p[1] for p in allp) - 70)), int(min(R.BODY_W, max(p[0] for p in allp) + 70)), int(min(R.BODY_H, max(p[1] for p in allp) + 70)))
            AUTHOR = {1: 'planner', 2: 'reflex', 3: 'reflex_noise'}  # observer_record.AUTHOR_CODES: the command's author at its latch step
            for k, s in enumerate(steps):
                if k == 0: what = 'before the first command'; author = None
                else:
                    c_, _, i_ = chunk_for(b, s - 4, (A2, PARENT)); author = AUTHOR.get(int(np.asarray(c_['author']).reshape(-1)[i_]), 'other')
                    planner_slot = ((s - 4 - 496260) // 5) % 2 == 0
                    assert planner_slot == (author == 'planner'), (s, author)
                    what = ('end of planner command %d' % ((k + 1) // 2)) if planner_slot else (('end of babble command %d' % (k // 2)) if author == 'reflex_noise' else ("end of part 7's command %d, no noise (drawn with the bout off)" % (k // 2)))
                D, grav = draw_marks(frames[s], body, qs[s], hp_t, out=OUT / ('seq_b1_%d.png' % s), crop=crop); seq.append(dict(step=s, what=what, author=author, D_cm=D, gravity=grav))
            p = render(body, 'seq_b1_target', q_t, tmp); _, gtarget = draw_marks(p, body, q_t, hp_t, out=OUT / 'seq_b1_target.png', crop=crop)
        finally:
            body.close()
        angs = [f['gravity']['angle_from_line_of_sight_deg'] for f in seq]
        rec['numbers']['sequence_body1'] = dict(target_step=goal, planner_target_label=goal + 1, crop_px=list(crop), target_crop_gravity=gtarget, gravity=dict(angle_from_line_of_sight_deg_min=min(angs), angle_from_line_of_sight_deg_max=max(angs), per_frame_in_frames=True), frames=seq)
        # ---- C. body 5's command at 496,280 (the diagnosis example), drawn here from the saved records with this file's marks
        ex = [e for e in json.load(open(EXAMPLES))['examples'] if e['name'] == 'largest_disagreement'][0]; H = ex['handles']
        q_t, header, _ = qpos_at(5, 496106); q_s = qpos_at(5, 496279)[0]; q_e = qpos_at(5, 496284)[0]
        body = R._Body(str(MODEL), header, ()); b5 = {}
        try:
            body.set_frame(q_s); hp_t = hand_points(body.m, q_t); hp_s = hand_points(body.m, q_s)
            body.fit([body.own_geoms(q_s), body.own_geoms(q_t), body.own_geoms(q_e)], [hp_t['L'], hp_t['R'], hp_s['L'], hp_s['R']])
            body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
            P = np.asarray(ex['selected_prediction']['position_cm'], float) / 100.0
            pred = {s: [P[H.index(n)] for n in ARM_HANDLES[s]] for s in 'LR'}
            fit = {s: [np.asarray(ex['fit_selected'][s]['rebuilt_position_cm'][k], float) / 100.0 for k in ('upperarm', 'lowerarm', 'wrist')] for s in 'LR'}
            check = float(np.linalg.norm(np.asarray(ex['start_pose']['position_cm'][H.index('wrist_R')]) / 100.0 - hp_s['R']) * 100)
            real_end = {s: [body_points(body.m, q_e, (n,))[n] for n in ARM_BODIES[s]] for s in 'LR'}
            for name, q in (('b5_T', q_t), ('b5_start', q_s), ('b5_end', q_e)):
                p = render(body, name, q, tmp); D, grav = draw_marks(p, body, q, hp_t, out=OUT / (name + '.png')); b5[name + '.png'] = dict(D_cm=D, gravity=grav)
            def px(pt): return project(body, pt)
            def draw_pred(d, body_, with_fit=True, hollow=False, fit_dots=False):
                for s in 'LR':
                    f = [px(p) for p in fit[s]]; pr = [px(p) for p in pred[s]]
                    if with_fit:
                        for i in range(2): dashed(d, f[i], f[i + 1], FIT, 3, 6)
                        for q in f: d.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], outline=FIT, width=2)
                        for a, c in zip(pr, f): d.line([a, c], fill=RED, width=2)
                    for q in pr:
                        if hollow: d.ellipse([q[0] - 9, q[1] - 9, q[0] + 9, q[1] + 9], outline=PURPLE, width=2)
                        else: d.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], fill=PURPLE, outline=(255, 255, 255))
                    if fit_dots:
                        for q in f: d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=PURPLE, outline=(255, 255, 255))
            p = render(body, 'b5_prediction', q_s, tmp); _, grav = draw_marks(p, body, q_s, hp_t, out=OUT / 'b5_prediction.png', extra=lambda d, bb: draw_pred(d, bb)); b5['b5_prediction.png'] = dict(gravity=grav)
            p = render(body, 'b5_afterA', q_s, tmp); _, grav = draw_marks(p, body, q_s, hp_t, out=OUT / 'b5_afterA.png', extra=lambda d, bb: draw_pred(d, bb, with_fit=False, hollow=True, fit_dots=True)); b5['b5_afterA.png'] = dict(gravity=grav)
            p = render(body, 'b5_prediction_left', q_s, tmp, 'left'); _, grav = draw_marks(p, body, q_s, hp_t, out=OUT / 'b5_prediction_left.png', extra=lambda d, bb: draw_pred(d, bb)); b5['b5_prediction_left.png'] = dict(gravity=grav, view='left')
            p = render(body, 'b5_afterA_left', q_s, tmp, 'left'); _, grav = draw_marks(p, body, q_s, hp_t, out=OUT / 'b5_afterA_left.png', extra=lambda d, bb: draw_pred(d, bb, with_fit=False, hollow=True, fit_dots=True)); b5['b5_afterA_left.png'] = dict(gravity=grav, view='left')
            def draw_afterC(d, body_):
                for s in 'LR':
                    pr = [px(p) for p in pred[s]]; rl = [px(p) for p in real_end[s]]
                    for q in pr: d.ellipse([q[0] - 9, q[1] - 9, q[0] + 9, q[1] + 9], outline=PURPLE, width=2)
                    for q in rl: d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=PURPLE, outline=(255, 255, 255))
            p = render(body, 'b5_afterC', q_e, tmp); _, grav = draw_marks(p, body, q_e, hp_t, out=OUT / 'b5_afterC.png', extra=draw_afterC); b5['b5_afterC.png'] = dict(gravity=grav)
            movesA = {s: [(px(pred[s][i]), px(fit[s][i]), '%.2f cm' % (np.linalg.norm(pred[s][i] - fit[s][i]) * 100)) for i in range(3)] for s in 'LR'}
            movesC = {s: [(px(pred[s][i]), px(real_end[s][i]), '%.2f cm' % (np.linalg.norm(pred[s][i] - real_end[s][i]) * 100)) for i in range(3)] for s in 'LR'}
            avoidA = {s: [(px(q), 9) for q in pred[s]] + [(px(q), 6) for q in fit[s]] + [(px(hp_s[s]), 8)] for s in 'LR'}
            avoidC = {s: [(px(q), 9) for q in pred[s]] + [(px(q), 4) for q in real_end[s]] + [(px(real_end[s][2]), 8)] for s in 'LR'}
            ins = {}
            for s, tag in (('R', 'right'), ('L', 'left')):
                pts = [m[0] for m in movesA[s]] + [m[1] for m in movesA[s]]; x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts); x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
                side = max(85, x1 - x0 + 24, y1 - y0 + 24); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                ins['b5_prediction_%s_arm.png' % tag] = inset('b5_prediction.png', 'b5_prediction_%s_arm.png' % tag, box, '%s arm now' % tag)
                ins['b5_afterA_%s_arm.png' % tag] = inset('b5_afterA.png', 'b5_afterA_%s_arm.png' % tag, box, '%s arm after repair A' % tag, movesA[s], avoid=avoidA[s])
                ptsC = [m[0] for m in movesC[s]] + [m[1] for m in movesC[s]]; x0, y0 = min(p[0] for p in ptsC), min(p[1] for p in ptsC); x1, y1 = max(p[0] for p in ptsC), max(p[1] for p in ptsC)
                side = max(85, x1 - x0 + 24, y1 - y0 + 24); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; boxC = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                ins['b5_afterC_%s_arm.png' % tag] = inset('b5_afterC.png', 'b5_afterC_%s_arm.png' % tag, boxC, '%s arm after repair C' % tag, movesC[s], avoid=avoidC[s])
            b5['insets'] = ins; b5['moves_cm'] = {k: {s: [m[2] for m in v[s]] for s in 'LR'} for k, v in (('A', movesA), ('C', movesC))}
            b5['check_start_wrist_R_vs_world_record_cm'] = round(check, 3); b5['T_step'] = 496106; b5['start_step'] = 496279; b5['end_step'] = 496284
            b5['predicted_hand_D_cm'] = {s: float(np.linalg.norm(pred[s][2] - hp_t[s]) * 100) for s in 'LR'}
        finally:
            body.close()
        rec['numbers']['body5_496280'] = b5
        # ---- D. body 3 of run POSE_A_off_494025: T, start, the command's requested directions (50 ms in), the end; front and left views
        g = json.load(open(GIZMO, encoding='utf-8')); s1 = g['per_state']['s1']; pdj = json.load(open(PATHD, encoding='utf-8'))
        q_t, header, _ = qpos_at(3, 494806, (POSEA,)); q_s = qpos_at(3, 494809, (POSEA,))[0]; q1 = qpos_at(3, 494810, (POSEA,))[0]; q_e = qpos_at(3, 494814, (POSEA,))[0]
        body = R._Body(str(MODEL), header, ()); b3 = {}
        try:
            body.set_frame(q_s); hp_t = hand_points(body.m, q_t); hp_s = hand_points(body.m, q_s)
            body.fit([body.own_geoms(q_s), body.own_geoms(q_t), body.own_geoms(q_e)], [hp_t['L'], hp_t['R'], hp_s['L'], hp_s['R']])
            body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
            ch = np.asarray(pdj['command_hand_point_world_m'], float)
            s5 = g['per_state']['s5']
            def rods(d, bb, st=None):
                st = st or s1
                for part, r in st['requested_direction_rods'].items():
                    o = np.asarray(r['origin_m'], float); a = project(bb, o); e = project(bb, o + np.asarray(r['requested_dir'], float) * float(r['length_m']) * (1.0 if part != 'hand' else 3.0))
                    if a and e:
                        d.line([a, e], fill=BLUE, width=5); d.ellipse([e[0] - 5, e[1] - 5, e[0] + 5, e[1] + 5], fill=BLUE)
                        letter = {'upper': 'U', 'fore': 'F', 'hand': 'H'}[part]; dx, dy = {'upper': (8, -9), 'fore': (10, 6), 'hand': (-26, -30)}[part]
                        d.rectangle([e[0] + dx - 2, e[1] + dy - 1, e[0] + dx + 14, e[1] + dy + 17], fill=(255, 255, 255)); d.text((e[0] + dx, e[1] + dy - 1), letter, fill=BLUE, font=FONT)
                for c in st.get('floor_contacts', []):
                    cp = project(bb, np.asarray(c['pos_m'], float))
                    if cp: d.ellipse([cp[0] - 6, cp[1] - 6, cp[0] + 6, cp[1] + 6], fill=YELLOW, outline=(120, 100, 0))
                if st is s1:
                    cpx = project(bb, ch); hpx = project(bb, hand_points(bb.m, q1)['L'])
                    if cpx and hpx: d.line([hpx, cpx], fill=BLUE, width=2); d.ellipse([cpx[0] - 7, cpx[1] - 7, cpx[0] + 7, cpx[1] + 7], fill=BLUE, outline=(255, 255, 255), width=2)
            rods_end = lambda d, bb: rods(d, bb, s5)
            for name, q, st, view, extra in (('b3_target', q_t, 494806, 'front', None), ('b3_start', q_s, 494809, 'front', None), ('b3_rods', q1, 494810, 'front', rods), ('b3_rods_left', q1, 494810, 'left', rods), ('b3_end', q_e, 494814, 'front', rods_end), ('b3_end_left', q_e, 494814, 'left', rods_end)):
                p = render(body, name, q, tmp, view); D, grav = draw_marks(p, body, q, hp_t, out=OUT / (name + '.png'), extra=extra, label=False); b3[name + '.png'] = dict(model_step=st, view=view, D_cm=D, gravity=grav)
            b3['rods'] = {k: dict(angle_between_deg=v['angle_between_deg'], length_m=v['length_m']) for k, v in s1['requested_direction_rods'].items()}
            b3['rods_end'] = {k: dict(angle_between_deg=v['angle_between_deg'], length_m=v['length_m']) for k, v in s5['requested_direction_rods'].items()}
            he = hand_points(body.m, q_e); render(body, 'b3_probe', q_e, tmp)
            b3['end_D_angle_from_line_of_sight_deg'] = {sd: round(math.degrees(math.asin(plane_share(body, he[sd], hp_t[sd] - he[sd])[0])), 1) for sd in 'LR'}
            b3['end_D_picture_px'] = {sd: round(float(np.linalg.norm(np.subtract(project(body, hp_t[sd]), project(body, he[sd])))), 1) for sd in 'LR'}
            b3['controls'] = s1['controls']; b3['floor_contacts'] = [c['part'] for c in s1['floor_contacts']]
            b3['command_hand_point_cm'] = dict(from_hand_point=float(np.linalg.norm(ch - hand_points(body.m, q1)['L']) * 100), from_T_hand_point=float(np.linalg.norm(ch - hp_t['L']) * 100))
            b3['joint_ranges_deg'] = {n: [round(math.degrees(float(x)), 1) for x in body.m.jnt_range[mujoco.mj_name2id(body.m, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in ('robot:left_elbow', 'robot:right_elbow', 'robot:left_knee', 'robot:right_knee')}
        finally:
            body.close()
        rec['numbers']['body3_POSE_A_off_494025'] = b3
        # ---- E. the baseline of October 23: per body the start, D at 5, 20 and 60 and P at 60, one camera per body
        z = np.load(BASELINE / 'records.npz', allow_pickle=False); res = json.load(open(BASELINE / 'result.json', encoding='utf-8'))
        base = {}
        for b in range(8):
            qD, qP = np.asarray(z['b%d_D_qpos' % b], float), np.asarray(z['b%d_P_qpos' % b], float)
            goal = int(TW['trajectories'][str(b)][0]['goal_step']) - 1; q_t, header, _ = qpos_at(b, goal)
            body = R._Body(str(MODEL), header, ()); pics = {}
            try:
                body.set_frame(qD[0]); hp_t = hand_points(body.m, q_t); hp_s = hand_points(body.m, qD[0])
                body.fit([body.own_geoms(qD[0]), body.own_geoms(q_t), body.own_geoms(qD[60]), body.own_geoms(qP[60])], [hp_t['L'], hp_t['R'], hp_s['L'], hp_s['R']])
                body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
                shown = (('start', qD[0]), ('T', q_t), ('D60', qD[60]), ('P60', qP[60])) + ((('D5', qD[5]), ('D20', qD[20])) if b == 5 else ())
                for name, q in shown:
                    p = render(body, 'base_b%d_%s' % (b, name), q, tmp); D, grav = draw_marks(p, body, q, hp_t, ring_cm=2.0 if name in ('D5', 'D20', 'D60', 'P60') else None, ring_label=False, out=OUT / ('base_b%d_%s.png' % (b, name)), label=False); pics[name] = dict(D_cm=D, gravity=grav)  # the 2 cm 'arrived' ring on every outcome frame (review 23)
                if b == 5:
                    # the search's one starting point: the first command's right-wrist row, read from the planner's record of this
                    # decision (motor_out: part 7's command, fractions of 200 cm, world frame; forecast_pose__iteration0_order: that
                    # command in the smoothed form the body would receive, cm; executed: the sent command, cm; goal_pose: T's rows).
                    # Blue square = the bound, +-1.80 cm per axis, around the raw row; purple square = what one command can pass
                    # through the smoothing, 0.35 x 1.80 = 0.63 cm, around the smoothed row.
                    cap = np.load(CAPTURE, allow_pickle=True); K = 'cap000496260_'; iR = 6
                    def cm_of(a): a = np.asarray(a, dtype=np.float64); return a if np.abs(a).max() > 3 else a * 200.0
                    raw_cm = cm_of(cap[K + 'motor_out'][5, iR, :3]); sm_cm = cm_of(cap[K + 'forecast_pose__iteration0_order'][5, iR, :3]); ex_cm = cm_of(cap[K + 'executed'][5, iR, :3]); tg_cm = cm_of(cap[K + 'goal_pose'][5, iR, :3])
                    raw_m, sm_m, ex_m = raw_cm / 100.0, sm_cm / 100.0, ex_cm / 100.0
                    hR = hand_points(body.m, qD[0])['R']
                    dist = lambda a, b2: round(float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b2, dtype=np.float64))), 2)
                    r2 = lambda v: [round(float(x), 2) for x in v]
                    region_numbers = dict(capture=str(CAPTURE), rows=dict(wrist_L=0, wrist_R=6), units='cm, world frame',
                                          first_command_wrist_R_cm=r2(raw_cm), first_command_smoothed_wrist_R_cm=r2(sm_cm), sent_wrist_R_cm=r2(ex_cm), T_wrist_R_cm=r2(tg_cm),
                                          first_command_to_T_cm=dist(raw_cm, tg_cm), first_command_to_hand_now_cm=dist(raw_cm, hR * 100), smoothed_to_T_cm=dist(sm_cm, tg_cm), smoothed_to_hand_now_cm=dist(sm_cm, hR * 100),
                                          sent_to_T_cm=dist(ex_cm, tg_cm), sent_minus_smoothed_cm=r2(ex_cm - sm_cm), hand_now_to_T_hand_point_cm=dist(hR * 100, hp_t['R'] * 100),
                                          frame_check_T_row_vs_T_hand_point_cm=dist(tg_cm, hp_t['R'] * 100), frame_check_start_row_vs_hand_now_cm=dist(cm_of(cap[K + 'start_pose'][5, iR, :3]), hR * 100))
                    allb = {}
                    for bb in range(8):
                        allb[str(bb)] = {side: dict(raw_to_T=dist(cm_of(cap[K + 'motor_out'][bb, ir, :3]), cm_of(cap[K + 'goal_pose'][bb, ir, :3])), smoothed_to_T=dist(cm_of(cap[K + 'forecast_pose__iteration0_order'][bb, ir, :3]), cm_of(cap[K + 'goal_pose'][bb, ir, :3])), hand_now_to_T=dist(cm_of(cap[K + 'start_pose'][bb, ir, :3]), cm_of(cap[K + 'goal_pose'][bb, ir, :3]))) for side, ir in (('L', 0), ('R', 6))}
                    region_numbers['all_bodies_wrist_rows'] = allb
                    WX, WY = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])  # world axes: the bound and the 0.63 cm offsets are counted per world axis
                    def square(d, bb, centre, half, col, w, P):
                        corners = [P(centre + WX * sx * half + WY * sy * half) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                        if all(corners): d.polygon(corners, outline=col, width=w)
                        return corners
                    def dot(d, pt, r, col, hollow=False, w=2):
                        if pt is None: return
                        if hollow: d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], outline=col, width=w)
                        else: d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=col, outline=(255, 255, 255))
                    DARK = (40, 40, 40)  # the sent row's dot: dark grey, so 'dark dot' is what the picture shows
                    def marks(d, bb, sc, P):
                        """The region's marks at scale sc (picture pixels per 1x pixel) through the projection P (world point -> pixels or None):
                        the line hand -> request, the 1.80 cm square and the request (blue), the smoothed request (purple ring; its 0.63 cm
                        square only when sc >= 2, where it is wider than the ring), the sent row (dark dot), drawn last."""
                        ph, pr, ps, pe = P(hR), P(raw_m), P(sm_m), P(ex_m); lw = max(2, int(round(1.5 * sc)))
                        if ph and pr: d.line([ph, pr], fill=BLUE, width=lw)
                        cb = [P(raw_m + WX * sx * 0.018 + WY * sy * 0.018) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]  # the 1.80 cm square's corners, not drawn (review 30: one bound on the page)
                        cp = square(d, bb, sm_m, 0.0063, PURPLE, max(1, int(round(0.5 * sc))), P) if sc >= 2 else None
                        dot(d, pr, max(4, int(round(2.5 * sc))), BLUE)
                        dot(d, ps, max(3, int(round(1.5 * sc))), PURPLE, hollow=True, w=max(2, int(round(0.5 * sc))))
                        dot(d, pe, max(2, int(round(2.0 * sc))), DARK)
                        return cb, cp
                    def region(d, bb):
                        marks(d, bb, 1.0, lambda pt: project(bb, pt))
                        d.rectangle([R.BODY_W - 236, 30, R.BODY_W - 6, 83], fill=(255, 255, 255))
                        d.text((R.BODY_W - 232, 31), "blue dot: the first command's request", fill=BLUE, font=FONT_S)
                        d.text((R.BODY_W - 232, 48), 'purple ring: that request smoothed', fill=PURPLE, font=FONT_S); d.text((R.BODY_W - 232, 65), 'dark dot: the row the body received', fill=DARK, font=FONT_S)
                    p = render(body, 'base_b5_region', qD[0], tmp); D, grav = draw_marks(p, body, qD[0], hp_t, out=OUT / 'base_b5_region.png', extra=region); pics['region'] = dict(D_cm=D, gravity=grav, boxes_cm=dict(search_bound_half=1.80, smoothing_pass_half=0.63), squares_axes='world x and y', **region_numbers)
                    # the enlargement, drawn at 4x from the unmarked render so every mark is crisp: the hand ring, the request, its smoothed
                    # form with the 0.63 cm square, the sent row, and the direction to T's row
                    hpx = project(body, hR); rpx = project(body, raw_m); tpx = project(body, tg_cm / 100.0); cx, cy = (hpx[0] + rpx[0]) / 2, (hpx[1] + rpx[1]) / 2
                    box = (int(cx - 45), int(cy - 45), int(cx + 45), int(cy + 45)); s = 4.0
                    im = Image.open(p).convert('RGB').crop(box).resize((360, 360), Image.LANCZOS); d = ImageDraw.Draw(im)
                    to_in = lambda pt: ((pt[0] - box[0]) * s, (pt[1] - box[1]) * s)
                    def P4(pt):
                        q4 = project(body, pt); return to_in(q4) if q4 else None
                    h_in = to_in(hpx); d.ellipse([h_in[0] - 32, h_in[1] - 32, h_in[0] + 32, h_in[1] + 32], outline=ORANGE, width=6)
                    cb, cp = marks(d, body, s, P4)
                    c = to_in(rpx); vx, vy = tpx[0] - rpx[0], tpx[1] - rpx[1]; n = math.hypot(vx, vy) or 1; e = (c[0] + vx / n * 120, c[1] + vy / n * 120); arrow(d, c, e, GREEN, 3)
                    top = min(q[1] for q in cb if q); half_px = (max(q[0] for q in cb if q) - min(q[0] for q in cb if q)) / 2
                    text = 'to T, %.1f cm' % region_numbers['first_command_to_T_cm']; w = 7 * len(text) + 4; x0 = min(max(4, c[0] + 10), 356 - w); y0 = max(4, min(top - 22, 338))  # above the blue square, right of the arrow's start
                    d.rectangle([x0, y0, x0 + w, y0 + 18], fill=(255, 255, 255)); d.text((x0 + 1, y0), text, fill=GREEN, font=FONT_S)
                    text2 = 'hand now, %.1f cm from the request' % region_numbers['first_command_to_hand_now_cm']; w2 = 7 * len(text2) + 4; x2 = min(max(4, h_in[0] - w2 / 2), 356 - w2); y2 = max(4, min(h_in[1] + 40, 338))  # below the hand ring
                    d.rectangle([x2, y2, x2 + w2, y2 + 18], fill=(255, 255, 255)); d.text((x2 + 1, y2), text2, fill=ORANGE, font=FONT_S)
                    pe4 = P4(ex_m); text3 = 'sent: %.1f cm from T' % region_numbers['sent_to_T_cm']
                    if pe4:
                        w3 = 7 * len(text3) + 4; x3 = max(4, min(pe4[0] - w3 - 10, 356 - w3)); y3 = max(4, min(pe4[1] - 9, 338))
                        d.rectangle([x3, y3, x3 + w3, y3 + 18], fill=(255, 255, 255)); d.text((x3 + 1, y3), text3, fill=DARK, font=FONT_S)
                    d.rectangle([0, 0, 359, 359], outline=PURPLE, width=4); d.rectangle([4, 4, 150, 26], fill=(255, 255, 255)); d.text((8, 6), 'right hand, 4x', fill=PURPLE, font=FONT); im.save(OUT / 'base_b5_region_hand.png')
                    # the bound itself at 12x, centred on the row the body received: the 0.63 cm square after smoothing, the smoothed
                    # request (purple ring) at its centre and the dark dot on its corner, with the one label that names the bound (review 44)
                    epx = project(body, ex_m); s12 = 12.0; h12 = 180 / s12; box3 = (int(epx[0] - h12), int(epx[1] - h12), int(epx[0] + h12), int(epx[1] + h12))
                    im3 = Image.open(p).convert('RGB').crop(box3).resize((360, 360), Image.LANCZOS); d3 = ImageDraw.Draw(im3)
                    to_in3 = lambda pt: ((pt[0] - box3[0]) * s12, (pt[1] - box3[1]) * s12)
                    def P12(pt):
                        q12 = project(body, pt); return to_in3(q12) if q12 else None
                    cp3 = square(d3, body, sm_m, 0.0063, PURPLE, 4, P12); dot(d3, P12(sm_m), 10, PURPLE, hollow=True, w=4); dot(d3, P12(ex_m), 12, DARK)
                    pr3 = P12(raw_m)
                    if pr3 and 0 <= pr3[0] < 360 and 0 <= pr3[1] < 360: dot(d3, pr3, 14, BLUE)
                    labels3 = [('0.63 cm: the farthest the search can move this row', PURPLE), ('dark dot: the row sent, on the corner', DARK)]
                    y3 = 356 - 2 * 18 - 4; w3 = 7 * max(len(t_) for t_, _ in labels3) + 6; d3.rectangle([4, y3, 4 + w3, 356], fill=(255, 255, 255))
                    for k_, (t_, c_) in enumerate(labels3): d3.text((7, y3 + 2 + 18 * k_), t_, fill=c_, font=FONT_S)
                    d3.rectangle([0, 0, 359, 359], outline=PURPLE, width=4); d3.rectangle([4, 4, 150, 26], fill=(255, 255, 255)); d3.text((8, 6), 'the bound, 12x', fill=PURPLE, font=FONT); im3.save(OUT / 'base_b5_region_bound.png')
                    pics['region_bound'] = dict(crop_box_px=list(box3), scale=s12, centre='the row the body received', labels=[t_ for t_, _ in labels3], square_px_at_12x=[[round(q[0], 1), round(q[1], 1)] for q in cp3] if cp3 else None)
                    pics['region_hand'] = dict(crop_box_px=list(box), scale=s, centre='midpoint of the hand now and the first command\'s request', marks='drawn at 4x from the unmarked render', D_R_cm=D['R'], labels=[text, text2, text3], blue_square_half_px_at_4x=round(half_px, 1), purple_square_px_at_4x=[[round(q[0], 1), round(q[1], 1)] for q in cp] if cp else None, sent_row_px_at_4x=[round(v, 1) for v in P4(ex_m)] if P4(ex_m) else None)
                    # repair E, drawn: the candidate asks the right-wrist row to go to T's row; the smoothing passes 0.35 of that change, so the
                    # row the body would receive if the candidate won this decision is 0.35 x T's row + 0.65 x the previous command's row
                    # (decision__smooth_prev); the sent row (executed) is drawn beside it
                    TEAL = (0, 140, 140)
                    prev_cm = cm_of(cap[K + 'decision__smooth_prev'][5, iR, :3]); cand_cm = 0.35 * tg_cm + 0.65 * prev_cm; cand_m = cand_cm / 100.0
                    allc = {}
                    for bb in range(8):
                        allc[str(bb)] = {side: dict(candidate_smoothed_to_T=dist(0.35 * cm_of(cap[K + 'goal_pose'][bb, ir, :3]) + 0.65 * cm_of(cap[K + 'decision__smooth_prev'][bb, ir, :3]), cm_of(cap[K + 'goal_pose'][bb, ir, :3])), sent_to_T=dist(cm_of(cap[K + 'executed'][bb, ir, :3]), cm_of(cap[K + 'goal_pose'][bb, ir, :3]))) for side, ir in (('L', 0), ('R', 6))}
                    nearer = sum(1 for v in allc.values() for side in 'LR' if v[side]['candidate_smoothed_to_T'] < v[side]['sent_to_T'])
                    cand_numbers = dict(rule='0.35 x T row + 0.65 x previous command row (decision__smooth_prev)', previous_command_wrist_R_cm=r2(prev_cm), candidate_smoothed_wrist_R_cm=r2(cand_cm), candidate_smoothed_to_T_cm=dist(cand_cm, tg_cm), candidate_smoothed_to_hand_now_cm=dist(cand_cm, hR * 100), candidate_smoothed_to_sent_cm=dist(cand_cm, ex_cm), sent_to_T_cm=dist(ex_cm, tg_cm), hand_now_to_T_row_cm=dist(hR * 100, tg_cm), all_bodies_wrist_rows=allc, wrist_rows_where_candidate_nearer_than_sent='%d of 16' % nearer)
                    def cand_marks(d, bb, sc, P):
                        pc, pe, ph = P(cand_m), P(ex_m), P(hR)
                        if ph and pc: d.line([ph, pc], fill=TEAL, width=max(2, int(round(1.5 * sc))))
                        dot(d, pe, max(2, int(round(2.0 * sc))), DARK); dot(d, pc, max(4, int(round(2.5 * sc))), TEAL)
                    def candidate(d, bb):
                        cand_marks(d, bb, 1.0, lambda pt: project(bb, pt))
                        d.rectangle([R.BODY_W - 262, 30, R.BODY_W - 6, 84], fill=(255, 255, 255))
                        d.text((R.BODY_W - 258, 31), 'teal dot: the row if the candidate wins', fill=TEAL, font=FONT_S); d.text((R.BODY_W - 258, 48), "(35 of 100 of the way to T's row)", fill=TEAL, font=FONT_S); d.text((R.BODY_W - 258, 65), 'dark dot: the row that was sent', fill=DARK, font=FONT_S)
                    p2 = render(body, 'base_b5_candidate', qD[0], tmp); D2, grav2 = draw_marks(p2, body, qD[0], hp_t, out=OUT / 'base_b5_candidate.png', extra=candidate); pics['candidate'] = dict(D_cm=D2, gravity=grav2, drawn=True, **cand_numbers)
                    cpx = project(body, cand_m); cx2, cy2 = (hpx[0] + cpx[0]) / 2, (hpx[1] + cpx[1]) / 2; box2 = (int(cx2 - 45), int(cy2 - 45), int(cx2 + 45), int(cy2 + 45))
                    im = Image.open(p2).convert('RGB').crop(box2).resize((360, 360), Image.LANCZOS); d = ImageDraw.Draw(im)
                    to_in2 = lambda pt: ((pt[0] - box2[0]) * s, (pt[1] - box2[1]) * s)
                    def P4b(pt):
                        q4 = project(body, pt); return to_in2(q4) if q4 else None
                    h2 = to_in2(hpx); d.ellipse([h2[0] - 32, h2[1] - 32, h2[0] + 32, h2[1] + 32], outline=ORANGE, width=6)
                    cand_marks(d, body, s, P4b)
                    c2 = to_in2(cpx); vx, vy = tpx[0] - cpx[0], tpx[1] - cpx[1]; n = math.hypot(vx, vy) or 1; e2 = (c2[0] + vx / n * 110, c2[1] + vy / n * 110); arrow(d, c2, e2, GREEN, 3)
                    labels2 = []; placed = []
                    for pt_in, text_, col, mode in ((c2, 'if it wins: %.1f cm to T' % cand_numbers['candidate_smoothed_to_T_cm'], TEAL, 'right'), (to_in2(project(body, ex_m)), 'sent: %.1f cm from T' % cand_numbers['sent_to_T_cm'], DARK, 16), (h2, 'hand now, %.1f cm from the teal dot' % cand_numbers['candidate_smoothed_to_hand_now_cm'], ORANGE, 40)):
                        w_ = 7 * len(text_) + 4
                        if mode == 'right': x_ = min(max(4, pt_in[0] + 14), 356 - w_); y_ = max(4, min(pt_in[1] - 9, 338))  # beside the dot, clear of the line below it
                        else: x_ = min(max(4, pt_in[0] - w_ / 2), 356 - w_); y_ = max(4, min(pt_in[1] + mode, 338))
                        for (px0, py0, pw) in placed:  # keep the boxes apart
                            if abs(y_ - py0) < 20 and x_ < px0 + pw and px0 < x_ + w_: y_ = max(4, min(py0 + 22, 338))
                        d.rectangle([x_, y_, x_ + w_, y_ + 18], fill=(255, 255, 255)); d.text((x_ + 1, y_), text_, fill=col, font=FONT_S); labels2.append(text_); placed.append((x_, y_, w_))
                    d.rectangle([0, 0, 359, 359], outline=TEAL, width=4); d.rectangle([4, 4, 150, 26], fill=(255, 255, 255)); d.text((8, 6), 'right hand, 4x', fill=TEAL, font=FONT); im.save(OUT / 'base_b5_candidate_hand.png')
                    pics['candidate_hand'] = dict(crop_box_px=list(box2), scale=s, drawn=True, centre='midpoint of the hand now and the teal dot', labels=labels2)
            finally:
                body.close()
            s = res['summary'][b]; at = {c: {k: dict(L=v['palm_cm']['L'], R=v['palm_cm']['R'], E=v['E']) for k, v in s[c]['at'].items()} for c in ('P', 'D')}
            flags = {c: {h: dict(arrived=s[c][h]['arrived'], arrival_step=s[c][h]['arrival_step'], stayed=s[c][h]['stayed_after_first_arrival']) for h in ('both', 'left', 'right')} for c in ('P', 'D')}
            base[b] = dict(start_record=s['start_step'] - 1, target_record=goal, pictures=pics, at=at, flags=flags, start_label=s['start_label'])
        rec['numbers']['baseline_8start'] = dict(source=str(BASELINE), row='reach_baseline_8start_a2', written=res['written'], counts=res['counts'], bodies=base)
        # ---- F. run seg41j body 5 (October 21): the comparison test (original and corrected command, the feasible command) and the limb test
        zc = np.load(COMPARISON_REC, allow_pickle=False); header = json.load(open(RS.HEADER, encoding='utf-8'))
        q = dict(target=np.asarray(zc['goal_qpos'], float), start=np.asarray(zc['start_qpos'], float), original_250ms=np.asarray(zc['qpos_original_A_hold250'][5], float),
                 corrected_250ms=np.asarray(zc['qpos_original_B_hold250'][5], float), feasible_target=np.asarray(zc['feasible_target_qpos'], float),
                 feasible_250ms=np.asarray(zc['qpos_feasible_A_hold250'][5], float), feasible_1000ms=np.asarray(zc['qpos_feasible_A_hold1000'][20], float))
        bm = C.BodyModel(str(RS.MODEL), header); names = list(bm.names); hb = np.asarray(bm.hb)
        body = R._Body(str(RS.MODEL), header, ()); cmp_out = {}
        try:
            body.set_frame(q['start']); hp_t = hand_points(body.m, q['target'])
            body.fit([body.own_geoms(q[k]) for k in ('start', 'target', 'original_250ms', 'feasible_1000ms')], [hp_t['L'], hp_t['R']])
            body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
            for k, qq in q.items():
                p = render(body, 'cmp_' + k, qq, tmp); D, grav = draw_marks(p, body, qq, hp_t, out=OUT / ('cmp_%s.png' % k), label=False); cmp_out['cmp_%s.png' % k] = dict(D_cm=D, gravity=grav)
            # the original command's requested arm directions (blue) and the corrected command's (purple), on the start
            axes = G.segment_axes(body.m, body.own); dd = mujoco.MjData(body.m); dd.qpos[:] = q['start']; mujoco.mj_forward(body.m, dd)
            orig, corr = np.asarray(zc['original_motor'], float), np.asarray(zc['corrected_original'], float); rodinfo = {}
            def rods2(d, bb):
                for s in 'LR':
                    for hname in ARM_HANDLES[s]:
                        i = names.index(hname); bid = int(hb[i]); axis, length = axes.get(bid, (np.array([0.0, 0.0, 1.0]), 0.08)); o = np.array(dd.xpos[bid], float)
                        for row, col, w in ((orig, BLUE, 5), (corr, PURPLE, 3)):
                            v = C.quat_xyzw_to_mat(row[i, 3:7]) @ np.asarray(axis, float); e = project(bb, o + v * float(length) * (1.0 if 'wrist' not in hname else 3.0)); a = project(bb, o)
                            if a and e:
                                d.line([a, e], fill=col, width=w); d.ellipse([e[0] - 4, e[1] - 4, e[0] + 4, e[1] + 4], fill=col)
                                if col == BLUE:
                                    letter = {'upperarm': 'U', 'lowerarm': 'F', 'wrist': 'H'}[hname.split('_')[0]]
                                    Lr = math.hypot(e[0] - a[0], e[1] - a[1]) or 1.0; ux_, uy_ = (e[0] - a[0]) / Lr, (e[1] - a[1]) / Lr
                                    vc_ = C.quat_xyzw_to_mat(corr[i, 3:7]) @ np.asarray(axis, float); e2 = project(bb, o + vc_ * float(length) * (1.0 if 'wrist' not in hname else 3.0))
                                    side = 1.0
                                    if e2:  # the side of the blue rod the purple rod's tip lies on: put the letter on the other side
                                        cross = (e[0] - a[0]) * (e2[1] - a[1]) - (e[1] - a[1]) * (e2[0] - a[0]); side = -1.0 if cross > 0 else 1.0
                                    lx, ly = e[0] + ux_ * 16 - uy_ * 12 * side, e[1] + uy_ * 16 + ux_ * 12 * side  # 16 px past the tip, 12 px to the side
                                    d.rectangle([lx - 8, ly - 9, lx + 8, ly + 9], fill=(255, 255, 255)); d.text((lx - 6, ly - 9), letter, fill=BLUE, font=FONT)
                        vo = C.quat_xyzw_to_mat(orig[i, 3:7]) @ np.asarray(axis, float); vc = C.quat_xyzw_to_mat(corr[i, 3:7]) @ np.asarray(axis, float)
                        rodinfo[hname] = dict(turned_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, float(vo @ vc) / (np.linalg.norm(vo) * np.linalg.norm(vc)))))), 1), place_moved_cm=round(float(np.linalg.norm(orig[i, :3] - corr[i, :3])), 1))
            p = render(body, 'cmp_rods', q['start'], tmp); D, grav = draw_marks(p, body, q['start'], hp_t, out=OUT / 'cmp_rods.png', extra=rods2, label=False); cmp_out['cmp_rods.png'] = dict(D_cm=D, gravity=grav, rods=rodinfo, note='the recorded command rows hold positions in cm (render_saved.py exact_image divides them by 100 for metres)')
        finally:
            body.close()
        rr = json.load(open(COMPARISON_RES, encoding='utf-8'))['readings']
        E = {'%s_%s' % (lane, m): dict(hold250=rr[lane][m]['hold250']['goal_E'], hold1000=rr[lane][m].get('hold1000', {}).get('goal_E')) for lane in ('original', 'feasible') for m in ('A', 'B')}
        rec['numbers']['comparison_483120'] = dict(source=str(COMPARISON_REC), pictures=cmp_out, goal_E=E, T_record=483016, start_record=483119, command_step=483120)
        zl = np.load(LIMB, allow_pickle=False); qA = np.asarray(zl['qpos_control_A'], float); q_goal = np.asarray(zl['goal_qpos'], float); q_ct = np.asarray(zl['control_target_qpos'], float)
        body = R._Body(str(RS.MODEL), header, ()); limb = []
        try:
            body.set_frame(qA[0]); hp_t = hand_points(body.m, q_goal)
            body.fit([body.own_geoms(qA[0]), body.own_geoms(qA[5]), body.own_geoms(qA[20]), body.own_geoms(q_goal)], [hp_t['L'], hp_t['R']])
            body.cams = {v: (look, dist * 1.18, az, el) for v, (look, dist, az, el) in body.cams.items()}
            for name, qq, ms in (('limb_control_target', q_ct, None), ('limb_0ms', qA[0], 0), ('limb_250ms', qA[5], 250), ('limb_1000ms', qA[20], 1000)):
                p = render(body, name, qq, tmp); D, grav = draw_marks(p, body, qq, hp_t, out=OUT / (name + '.png'), label=False); limb.append(dict(picture=name + '.png', ms=ms, D_cm=D, gravity=grav))
        finally:
            body.close()
        rec['numbers']['limb_control'] = dict(source=str(LIMB), frames=limb)
        angs = []
        def collect(o):
            if isinstance(o, dict):
                if 'angle_from_line_of_sight_deg' in o: angs.append(o['angle_from_line_of_sight_deg'])
                for v in o.values(): collect(v)
        collect(rec['numbers']); rec['gravity']['angle_from_line_of_sight_deg_min_max'] = [min(angs), max(angs)] if angs else None
        for f in sorted(OUT.glob('*.png')): rec['images'][f.name] = sha(f)
        rec['seconds'] = round(time.time() - t0, 1); rec['status'] = 'complete'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        json.dump(rec, open(OUT / 'plan_pictures.json', 'w'), indent=1)
    print(json.dumps(dict(images=len(rec['images']), seconds=rec.get('seconds'), status=rec.get('status'))))

if __name__ == '__main__':
    main()
