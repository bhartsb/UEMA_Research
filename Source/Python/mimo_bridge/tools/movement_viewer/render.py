# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The pictures of one saved command: left hand, right hand and whole body; start, asked for, got to.

Generalizes docs/h84_finger_shapes_2026-10-17/build.py (the October 17 finger pictures) from one hard-coded command to
any command of any recording (sources.py). No learner, no training update, no physics step: poses come from the saved
joint positions by forward kinematics (mujoco mj_kinematics and mj_comPos) of the recording's saved model.

INPUT: a recording id, a body and the model step of a command's first record. The chunks around it are read
(commands.load_records) and the command found by commands.segment; refused, with the reason, when no command starts at
that model step, when the saved model's structure fingerprint differs from the header of the chunk holding the
command's first record (commands.BodyModel), when any record the drawing uses (the start, the command, the next
command) comes from a chunk recorded with another model, or when the chunks cannot be read.

THE POSES: start = the record before the command's first record (missing at an episode start or a gap); the samples
t050 ... t250 = the command's records, one per 50 ms model step (t250 = the actual result after 250 ms for a 5-step
command); the requested hand = the start pose with its 44 finger joints set to the command's saved target angles
(order_hand_goal), exactly as build.py constructs it.

HANDS (build.py's method): only the hand's own geometry drawn, fingers colored (index blue, middle orange, ring purple,
little green, thumb red, palm gray), every geom moved into the wrist's coordinates so the wrist sits still and finger
differences show, two cameras (front: azimuth -55, elevation 20; side: azimuth 0, elevation -15; the right hand's turned
by 180 degrees about the vertical so its fingers face the camera). Changed at the operator's review of Oct 19, 2026: the
cameras are placed to fit the hand (build.py's stood 0.20 m from the wrist, and the hand filled less than half of the
picture): each view's camera looks at the middle of the drawn hands' bounding boxes from the distance at which they fit
the picture (HAND_NEAREST at the least), and each sample picture carries the requested hand drawn pale (HAND_PALE) over
the actual hand, each in its own wrist's coordinates. _Hands.draw with distance 0.20 and no ghost still draws build.py's
pictures (tests/test_movement_viewer.py compares them with the October 17 pictures).

WHOLE BODY (Oct 19, 2026, replacing one view from above in which the caregiver adult covered the infant): the infant
alone, each body part colored by its group (PART_GROUPS: left arm and hand, right arm and hand, left leg, right leg,
head, trunk); the caregiver, the floor and every object that no hand part touched during the command are left out;
objects a hand part touched are drawn at their saved poses. Every picture of a command is drawn in the coordinates of
the infant's upper body at the start pose (x toward the belly, z toward the head), so the infant stands upright with
its chest toward the front camera whatever way it lay; the way it lay is named in words (_Body.posture). Three cameras
(BODY_VIEWS: from the front, from the infant's left, from its right), each placed once per command so that every pose
drawn fits the picture (_Body.fit).
  ASKED FOR: the start pose with the body part of every powered scored handle (mask not 0, strength above 0;
  commands.scored: not the 10 fingertip rows, which the hand control replaces before the controller reads them, and not
  the eyes' row, which moves no joint) turned to its target orientation about its first joint's anchor, every other
  part following its parent as at the start (its offset and turn relative to the parent kept), the fingers at their 44
  target angles (_Body.requested_frames). The command gives each part its own target, so this body can bend where the
  joints cannot. It follows the asked-for DIRECTIONS only: the command also asks for PLACES (the target positions the
  decoder solves for: the wrists, and the head and the pelvis handle unless the run turned their rows off), drawn as
  dots in the part's color, each joined by a thin line to its part; how far each part of the asked-for body is from its
  place is saved (asked_for_places) and stated beside the pictures.
  GOT TO: the saved pose at each sample, with the asked-for body drawn pale over it and the same dots, each joined to
  its part where it is at that sample.
PALE OVERLAY (hands and body): the asked-for shape is drawn alone in a second picture from the same camera, in one
color (HAND_PALE, BODY_PALE), and blended over the first picture wherever it covers a pixel (PALE_ALPHA); a pale part
that lies where the drawn part lies (within PALE_SAME_M and PALE_SAME_DEG on the hands, BODY_SAME_M and BODY_SAME_DEG on
the whole body) is left out. Drawn in one picture, two
surfaces in nearly one place came out as interleaved stripes and patches that look like a drawing fault (the
independent review of Oct 19, 2026).

OUTPUT, in pictures_dir()/<key>/: hand_<left|right>_<front|side>_<state>.png, body_<front|left|right>_<state>.png
(states: start, target, t050 ... t250), manifest.json: the command's measures (commands.measure), every handle's target
and errors, at the start, at each sample and after the next command (parts), every finger joint's angle at the start, as
requested, at each sample and after the next command (fingers), the contacts, the posture, provenance with the SHA-256 of
each source file, the renderer version and the cost: seconds, this process's memory and the graphics card's memory in
use before and after.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import time
from typing import Dict, List, Optional

import numpy as np

from . import part_path, pictures_dir, replace_file
from . import commands as C
from . import sources as S

RENDER_VERSION = "movement_render_v16"
HAND_SIZE = 720
HAND_NEAREST = 0.08                  # m: the hand cameras' nearest distance to the point they look at (build.py: 0.20)
BODY_W, BODY_H = 480, 560
BODY_BACKGROUND = (240, 243, 242)    # the body pictures' background (the renderer draws it black)
FINGER_COLORS = {"ff": [.12, .48, .87, 1], "mf": [.96, .58, .12, 1], "rf": [.62, .36, .85, 1],
                 "lf": [.12, .68, .55, 1], "th": [.90, .30, .31, 1]}
PALM_COLOR = [.72, .76, .82, 1]
HAND_PALE = [0.93, 0.95, 0.98, 1.0]          # the asked-for hand, blended over a dark picture
BODY_PALE = [0.40, 0.44, 0.48, 1.0]          # the asked-for body, blended over a light picture
PALE_ALPHA = 0.45                            # the blend: 0.45 of the pale picture over 0.55 of the drawn one
PALE_SAME_M, PALE_SAME_DEG = 0.0015, 1.5     # a pale part this close to the drawn one is left out (module doc)
# the whole body is drawn at about 1 px per 2 mm: there a pale part within 4 mm and 3 degrees of the drawn one tinted the
# body without showing a visible difference, so the body's pictures leave those out too
BODY_SAME_M, BODY_SAME_DEG = 0.004, 3.0
MARK_COLOR = (0.86, 0.15, 0.55)              # a probe's test-pose place: pink, unlike every body part's color
LINE_COLOR = (0.35, 0.37, 0.40)              # the same point at a probe's starting line: grey
# the October 17 cameras for the left hand; the right hand's are turned 180 degrees about the vertical (azimuth + 180),
# so its fingers face the camera too (the October 17 angles show the back of the right hand, measured Oct 18, 2026)
HAND_VIEWS = {"left": (("front", -55, 20), ("side", 0, -15)), "right": (("front", 125, 20), ("side", 180, -15))}
# the whole-body cameras in the start's upper-body coordinates (x toward the belly, y toward the infant's left, z toward
# the head): looking at the chest, at the infant's left side, at its right side; 8 degrees from above
BODY_VIEWS = (("front", 180.0, -8.0), ("left", -90.0, -8.0), ("right", 90.0, -8.0))
BODY_VIEW_NAMES = {"front": "from the front", "left": "from its left side", "right": "from its right side"}
# body part groups and their colors (the page's legend): a body belongs to the group of its nearest named ancestor
PART_GROUPS = (("left arm and hand", "left_upper_arm", (0.16, 0.44, 0.86)),
               ("right arm and hand", "right_upper_arm", (0.91, 0.53, 0.12)),
               ("left leg", "left_upper_leg", (0.18, 0.62, 0.36)),
               ("right leg", "right_upper_leg", (0.07, 0.64, 0.71)),
               ("head", "head", (0.56, 0.27, 0.68)))
TRUNK_COLOR = (0.55, 0.43, 0.39)
# the scored handles in words (the body part each one turns), and the joints of the hands in words
PART_NAMES = {"wrist_L": "left wrist", "lowerarm_L": "left forearm", "upperarm_L": "left upper arm",
              "wrist_R": "right wrist", "lowerarm_R": "right forearm", "upperarm_R": "right upper arm",
              "thigh_L": "left thigh", "calf_L": "left lower leg", "foot_L": "left foot",
              "thigh_R": "right thigh", "calf_R": "right lower leg", "foot_R": "right foot",
              "head": "head", "lower_trunk": "hips (lower trunk)", "chest": "chest"}
FINGER_NAMES = (("th", "thumb"), ("ff", "index finger"), ("mf", "middle finger"), ("rf", "ring finger"),
                ("lf", "little finger"))
JOINT_WORDS = {"knuckle": "knuckle", "middle": "middle joint", "distal": "tip joint", "side": "side-to-side",
               "meta": "palm joint", "swivel": "swivel", "adduction": "toward the palm", "pivot": "twist"}


def key_for(rid: str, body: int, step: int, lite: bool = False) -> str:
    """The drawing's folder name: the recording, body, model step, renderer version and (for an automatic pick) the
    smaller LITE picture set."""
    tag = "|lite" if lite else ""
    return hashlib.sha1(("%s|%d|%d|%s%s" % (rid, int(body), int(step), RENDER_VERSION, tag)).encode()).hexdigest()[:20]


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def part_color(group: str):
    return next((c for g, _, c in PART_GROUPS if g == group), TRUNK_COLOR)


def handle_group(name: str) -> str:
    """The body part group a scored handle's part belongs to."""
    if name in ("head",):
        return "head"
    if name in ("lower_trunk", "chest"):
        return "trunk"
    side = "left" if name.endswith("_L") else "right"
    if name.startswith(("thigh", "calf", "foot")):
        return side + " leg"
    return side + " arm and hand"


def gpu_memory_mb() -> Optional[float]:
    """The graphics card's memory in use, MB (nvidia-smi), or None where it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def process_memory_mb() -> Optional[float]:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "private", mi.rss)) / 2 ** 20
    except Exception:
        return None


def find_command(rec: dict, body: int, step: int):
    """(records, command (first, last), next command or None, the chunks read) for the command starting at `step`."""
    b = rec["bodies"].get(str(body))
    if not b:
        raise C.Refused("recording %s has no body %d" % (rec["id"], body))
    chunks = b["chunks"]
    firsts = [int(c[3]) for c in chunks]
    k = max([i for i, f in enumerate(firsts) if f <= step], default=None)
    if k is None:
        raise C.Refused("no saved chunk holds model step %d" % step)
    use = chunks[max(0, k - 1):k + 2]
    data = C.load_records(use)
    cmds = C.segment(data)
    st = np.asarray(data["step"])
    for ci, (a, z) in enumerate(cmds):
        if int(st[a]) == int(step):
            nxt = cmds[ci + 1] if ci + 1 < len(cmds) else None
            return data, (a, z), nxt, use
    raise C.Refused("no command starts at model step %d in body %d of %s" % (step, body, rec["label"]))


def _vis_state(m) -> dict:
    """The model fields the pictures change, to put back afterwards (the model is shared: commands.loaded_model)."""
    return dict(geom_group=np.array(m.geom_group), geom_rgba=np.array(m.geom_rgba), geom_matid=np.array(m.geom_matid),
                body_pos=np.array(m.body_pos), body_quat=np.array(m.body_quat), znear=float(m.vis.map.znear),
                zfar=float(m.vis.map.zfar), ambient=np.array(m.vis.headlight.ambient),
                diffuse=np.array(m.vis.headlight.diffuse), specular=np.array(m.vis.headlight.specular))


def _restore(m, s: dict) -> None:
    m.geom_group[:] = s["geom_group"]
    m.geom_rgba[:] = s["geom_rgba"]
    m.geom_matid[:] = s["geom_matid"]
    m.body_pos[:] = s["body_pos"]
    m.body_quat[:] = s["body_quat"]
    m.vis.map.znear = s["znear"]
    m.vis.map.zfar = s["zfar"]
    m.vis.headlight.ambient[:] = s["ambient"]
    m.vis.headlight.diffuse[:] = s["diffuse"]
    m.vis.headlight.specular[:] = s["specular"]


def _blend(base, pale, alpha: float = PALE_ALPHA):
    """The pale picture (drawn alone on black) over the base picture wherever it covers a pixel."""
    base = np.asarray(base, dtype=np.float32)
    pale = np.asarray(pale, dtype=np.float32)
    covered = (pale.max(-1, keepdims=True) > 0).astype(np.float32) * alpha
    return np.clip(base * (1.0 - covered) + pale * covered, 0, 255).astype(np.uint8)


def _same_place(pos_a, mat_a, pos_b, mat_b, within_m: float = PALE_SAME_M, within_deg: float = PALE_SAME_DEG) -> bool:
    """Whether two placements of one geom lie within within_m metres and within_deg degrees of each other."""
    if float(np.linalg.norm(np.asarray(pos_a) - np.asarray(pos_b))) >= within_m:
        return False
    c = (float(np.trace(np.asarray(mat_a).reshape(3, 3) @ np.asarray(mat_b).reshape(3, 3).T)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) < within_deg


def _add_line(mj, scene, a, b, width: float, rgba) -> None:
    """Append a capsule from point a to point b (nothing when the scene is full or the points coincide)."""
    if scene.ngeom >= scene.maxgeom or float(np.linalg.norm(np.asarray(b) - np.asarray(a))) < 1e-6:
        return
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(g, int(mj.mjtGeom.mjGEOM_CAPSULE), np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                    np.asarray(rgba, dtype=np.float32))
    mj.mjv_connector(g, int(mj.mjtGeom.mjGEOM_CAPSULE), float(width), np.asarray(a, dtype=np.float64),
                     np.asarray(b, dtype=np.float64))
    scene.ngeom += 1


def _add_geom(mj, scene, gtype: int, size, pos, mat, rgba, dataid: int = -1) -> None:
    """Append one geom to a scene (nothing when the scene is full)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(g, int(gtype), np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
                    np.asarray(mat, dtype=np.float64).reshape(-1), np.asarray(rgba, dtype=np.float32))
    g.dataid = int(dataid)
    scene.ngeom += 1


class _Hands:
    """build.py's hand pictures, one renderer for both hands."""

    def __init__(self, model_path: str, size: int = HAND_SIZE):
        import mujoco as mj
        self.mj = mj
        m, self.d = C.loaded_model(model_path)
        self.m = m
        self.saved = _vis_state(m)
        m.vis.global_.offwidth = max(900, size, int(m.vis.global_.offwidth))
        m.vis.global_.offheight = max(900, size, int(m.vis.global_.offheight))
        m.vis.map.znear = .001
        m.vis.map.zfar = 10
        m.vis.headlight.ambient[:] = [.48, .48, .48]
        m.vis.headlight.diffuse[:] = [.7, .7, .7]
        m.vis.headlight.specular[:] = [.15, .15, .15]
        opt = self.opt = mj.MjvOption()
        opt.flags[mj.mjtVisFlag.mjVIS_TENDON] = False
        opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = False
        opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = False
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = 1
        m.geom_group[:] = 5
        self.renderer = mj.Renderer(m, height=size, width=size)

    def close(self):
        self.renderer.close()
        _restore(self.m, self.saved)

    def _wrist_frame(self, wrist: int, q):
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        return d.xmat[wrist].reshape(3, 3).copy(), d.xpos[wrist].copy()

    def draw(self, side: str, states: List[tuple], out_dir: str, views=None, distance: Optional[float] = None,
             ghost=None, ghost_on=()) -> Dict[str, dict]:
        """states: [(label, qpos or None)]; ghost: a qpos whose hand is drawn pale over the pictures whose label is in
        ghost_on (each hand in its own wrist's coordinates). distance None: each view's camera is placed so that every
        drawn hand fits the picture (HAND_NEAREST at the least: _fit); a number: build.py's camera, looking at the
        point 0.5 cm below the wrist's origin from that distance (0.20 draws the October 17 pictures)."""
        mj, m, d = self.mj, self.m, self.d
        wrist = m.body(side + "_hand").id
        bodies = []
        for b in range(m.nbody):
            anc = b
            while anc and anc != wrist:
                anc = int(m.body_parentid[anc])
            if anc == wrist:
                bodies.append(b)
        geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies]
        m.geom_group[:] = 5
        for g in geoms:
            m.geom_group[g] = 0
            bname = m.body(int(m.geom_bodyid[g])).name
            m.geom_rgba[g] = next((c for k, c in FINGER_COLORS.items() if "_" + k in bname), PALM_COLOR)
            m.geom_matid[g] = -1
        pale = None
        if ghost is not None:
            # the ghost hand's geoms in its own wrist's coordinates, once
            Rg, og = self._wrist_frame(wrist, ghost)
            pale = [(g, int(m.geom_type[g]), np.array(m.geom_size[g]), Rg.T @ (np.array(d.geom_xpos[g]) - og),
                     Rg.T @ np.array(d.geom_xmat[g]).reshape(3, 3), int(m.geom_dataid[g])) for g in geoms
                    if int(m.geom_type[g]) != int(mj.mjtGeom.mjGEOM_PLANE)]
        made = {}
        fitted = self._fit(wrist, geoms, [q for _, q in states if q is not None] + ([ghost] if ghost is not None else []))
        for view, az, el in HAND_VIEWS[side]:
            if views is not None and view not in views:
                continue
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            if distance is None:
                look, dist = fitted[view] if side == "left" else fitted[view + "_R"]
                cam.lookat[:] = look
                cam.distance = float(dist)
            else:
                cam.lookat[:] = [0, 0, -.005]
                cam.distance = float(distance)
            cam.azimuth = az
            cam.elevation = el
            for label, q in states:
                if q is None:
                    continue
                R, origin = self._wrist_frame(wrist, q)
                self.renderer.update_scene(d, camera=cam, scene_option=self.opt)
                scene = self.renderer.scene
                scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
                scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = False
                scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = False
                scene.flags[mj.mjtRndFlag.mjRND_FOG] = False
                for n in range(scene.ngeom):
                    geom = scene.geoms[n]
                    geom.pos[:] = R.T @ (geom.pos - origin)
                    geom.mat[:] = R.T @ geom.mat
                scene.nlight = 1
                # the light turns with the camera: the right hand's views are the left's turned 180 degrees about the
                # vertical, so its light is too (x, y -> -x, -y)
                turn = -1.0 if side == "right" else 1.0
                scene.lights[0].pos[:] = [0, .25 * turn, .3]
                scene.lights[0].dir[:] = [0, -.7 * turn, -.7]
                scene.lights[0].ambient[:] = [.4, .4, .4]
                scene.lights[0].diffuse[:] = [.8, .8, .8]
                name = "hand_%s_%s_%s.png" % (side, view, label)
                pixels = self.renderer.render()
                if pale is not None and label in ghost_on:
                    keep = [x for x in pale if not _same_place(x[3], x[4], R.T @ (np.array(d.geom_xpos[x[0]]) - origin),
                                                                R.T @ np.array(d.geom_xmat[x[0]]).reshape(3, 3))]
                    scene.ngeom = 0
                    for g, gtype, size, pos, mat, dataid in keep:
                        _add_geom(mj, scene, gtype, size, pos, mat, HAND_PALE, dataid)
                    if keep:
                        pixels = _blend(pixels, self.renderer.render())
                _save_png(pixels, os.path.join(out_dir, name))
                made[name] = dict(side=side, view=view, state=label, ghost=bool(pale is not None and label in ghost_on))
        return made


    def _fit(self, wrist: int, geoms: List[int], poses: List[np.ndarray]) -> Dict[str, tuple]:
        """Per view of either hand ('<view>' for the left hand's cameras, '<view>_R' for the right's): (look-at point,
        distance) so that every geom of the hand at every pose, in its own wrist's coordinates with its bounding radius,
        fits the square picture with a 6% margin; never nearer than HAND_NEAREST."""
        m, d = self.m, self.d
        pts = []
        for q in poses:
            R, origin = self._wrist_frame(wrist, q)
            for g in geoms:
                pts += [R.T @ (c - origin) for c in _corners(m, g, d.geom_xpos[g], d.geom_xmat[g])]
        P = np.array(pts)
        r = np.zeros(len(P))
        tanh = math.tan(math.radians(float(m.vis.global_.fovy)) / 2.0)
        out = {}
        for side, views in HAND_VIEWS.items():
            for view, az, el in views:
                f, up = _cam_axes(az, el)
                right = np.cross(f, up)
                x, y, z = P @ right, P @ up, P @ f
                x0, x1, y0, y1 = (x - r).min(), (x + r).max(), (y - r).min(), (y + r).max()
                z0, z1 = (z - r).min(), (z + r).max()
                half = max(x1 - x0, y1 - y0) / 2.0 * 1.06
                dist = max(HAND_NEAREST, half / tanh + (z1 - z0) / 2.0)
                look = (x0 + x1) / 2.0 * right + (y0 + y1) / 2.0 * up + (z0 + z1) / 2.0 * f
                out[view if side == "left" else view + "_R"] = (look, dist)
        return out


def _save_png(pixels, path: str, background=None) -> None:
    from PIL import Image
    if background is not None:
        pixels = np.array(pixels)
        pixels[(pixels == 0).all(-1)] = background
    Image.fromarray(pixels).save(path)


def _corners(m, g: int, pos, mat) -> List[np.ndarray]:
    """The 8 corners of geom g's bounding box (m.geom_aabb: center and half sizes in the geom's frame), world."""
    c, h = np.asarray(m.geom_aabb[g][:3], dtype=np.float64), np.asarray(m.geom_aabb[g][3:], dtype=np.float64)
    R = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    return [np.asarray(pos, dtype=np.float64) + R @ (c + h * np.array(sg)) for sg in
            ((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1))]


def _cam_axes(az: float, el: float):
    """(forward, up) of a MuJoCo free camera at this azimuth and elevation (degrees)."""
    a, e = math.radians(az), math.radians(el)
    f = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    up = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    return f, up


class _Body:
    """The whole-body pictures (module doc): the infant alone, in the start's upper-body coordinates, three cameras."""

    def __init__(self, model_path: str, header: dict, touched_objects=()):
        import mujoco as mj
        self.mj = mj
        m, self.d = C.loaded_model(model_path)
        self.m = m
        self.saved = _vis_state(m)
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), BODY_W)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), BODY_H)
        m.vis.map.znear = .005
        self.header = header
        self.own = sorted({int(b) for e in header["entities"] if e["kind"] == "self" for b in e["bodies"]})
        self.own_set = set(self.own)
        self.group = {b: self._group_of(b) for b in self.own}
        self.objects = header.get("objects") or []
        self.mounts = [(k, int(o["root"])) for k, o in enumerate(self.objects) if not o["free"] and int(o["mocap"]) < 0]
        self.touched = {int(g) for k in touched_objects for g in (self.objects[k].get("geoms") or [])}
        self.opt = mj.MjvOption()
        self.opt.flags[mj.mjtVisFlag.mjVIS_TENDON] = False
        self.renderer = mj.Renderer(m, height=BODY_H, width=BODY_W, max_geom=20000)
        self.trunk = m.body("upper_body").id if _has_body(m, "upper_body") else m.body("chest").id
        self.o, self.Rf = np.zeros(3), np.eye(3)
        self.cams: Dict[str, tuple] = {}

    def close(self):
        self.renderer.close()
        _restore(self.m, self.saved)

    def _group_of(self, b: int) -> str:
        m = self.m
        roots = {r: g for g, r, _ in PART_GROUPS}
        while b > 0:
            name = m.body(b).name
            if name in roots:
                return roots[name]
            b = int(m.body_parentid[b])
        return "trunk"

    def place(self, q, mocap=None, objpose=None):
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = q
        if mocap is not None and m.nmocap and len(mocap):
            d.mocap_pos[:] = np.asarray(mocap)[:, :3]
            d.mocap_quat[:] = np.asarray(mocap)[:, 3:7]
        if objpose is not None and len(objpose):
            for k, root in self.mounts:
                m.body_pos[root] = objpose[k, :3]
                m.body_quat[root] = objpose[k, 3:7]
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        mj.mj_camlight(m, d)

    def set_frame(self, q) -> None:
        """The drawing coordinates: the upper body's frame at q (the start pose)."""
        self.place(q)
        self.o = np.array(self.d.xpos[self.trunk])
        self.Rf = np.array(self.d.xmat[self.trunk]).reshape(3, 3)

    def weight_n(self) -> float:
        """The infant's weight, N: the mass of its own bodies times 9.81 m/s2."""
        return float(sum(float(self.m.body_mass[b]) for b in self.own)) * 9.81

    def posture(self) -> str:
        """The way the infant lay at the start, from its upper body's axes against the vertical."""
        belly, left, head = self.Rf[:, 0], self.Rf[:, 1], self.Rf[:, 2]
        if head[2] > 0.7:
            return "sitting or standing up"
        if head[2] < -0.7:
            return "head down"
        if belly[2] > 0.7:
            return "lying on its back"
        if belly[2] < -0.7:
            return "lying on its front"
        if left[2] > 0.7:
            return "lying on its right side"
        if left[2] < -0.7:
            return "lying on its left side"
        return "leaning, between lying and sitting up"

    def requested_frames(self, q_req, targets: Dict[int, np.ndarray]):
        """(body positions and orientations at q_req, the asked-for ones): from the root outward, each own body follows
        its parent as at q_req (offset and turn relative to the parent kept); a body with a target orientation is
        turned to it about its first hinge joint's anchor."""
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = q_req
        mj.mj_kinematics(m, d)
        xp = np.array(d.xpos)
        xm = np.array(d.xmat).reshape(-1, 3, 3)
        gp, gm = xp.copy(), xm.copy()
        for b in self.own:                         # MuJoCo numbers a parent before its children
            p = int(m.body_parentid[b])
            if p in self.own_set:
                gm[b] = gm[p] @ (xm[p].T @ xm[b])
                gp[b] = gp[p] + gm[p] @ (xm[p].T @ (xp[b] - xp[p]))
            if b in targets:
                js = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == b
                      and int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_HINGE)]
                anchor = np.array(d.xanchor[js[0]]) if js else xp[b]
                local = xm[b].T @ (anchor - xp[b])
                held = gp[b] + gm[b] @ local
                gm[b] = np.asarray(targets[b], dtype=np.float64)
                gp[b] = held - gm[b] @ local
        return xp, xm, gp, gm

    def own_geoms(self, q, frames=None) -> List[tuple]:
        """[(geom id, position, orientation)] of the infant's visible geoms at q, or at the asked-for frames, world."""
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        out = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b not in self.own_set or int(m.geom_group[g]) > 2 or float(m.geom_rgba[g, 3]) <= 0:
                continue
            pos, mat = np.array(d.geom_xpos[g]), np.array(d.geom_xmat[g]).reshape(3, 3)
            if frames is not None:
                xp, xm, gp, gm = frames
                pos = gp[b] + gm[b] @ (xm[b].T @ (pos - xp[b]))
                mat = gm[b] @ (xm[b].T @ mat)
            out.append((g, pos, mat))
        return out

    def fit(self, geom_sets: List[List[tuple]], dots: List[np.ndarray]) -> None:
        """One camera per view (BODY_VIEWS) so that every listed geom (the corners of its bounding box) and dot fits."""
        m = self.m
        pts, rad = [], []
        for gs in geom_sets:
            for g, pos, mat in gs:
                for c in _corners(m, g, pos, mat):
                    pts.append(self.Rf.T @ (c - self.o))
                    rad.append(0.0)
        for p in dots:
            pts.append(self.Rf.T @ (np.asarray(p) - self.o))
            rad.append(0.015)
        P, r = np.array(pts), np.array(rad)
        tanh = math.tan(math.radians(float(m.vis.global_.fovy)) / 2.0)
        aspect = BODY_W / BODY_H
        for name, az, el in BODY_VIEWS:
            f, up = _cam_axes(az, el)
            right = np.cross(f, up)
            x, y, z = P @ right, P @ up, P @ f
            x0, x1, y0, y1 = (x - r).min(), (x + r).max(), (y - r).min(), (y + r).max()
            z0, z1 = (z - r).min(), (z + r).max()
            hw, hh = (x1 - x0) / 2.0 * 1.06, (y1 - y0) / 2.0 * 1.06
            dist = max(hh / tanh, hw / (tanh * aspect)) + (z1 - z0) / 2.0
            look = (x0 + x1) / 2.0 * right + (y0 + y1) / 2.0 * up + (z0 + z1) / 2.0 * f
            self.cams[name] = (look, dist, az, el)

    def draw(self, label: str, view: str, q, mocap, objpose, frames=None, ghost: Optional[List[tuple]] = None,
             dots: Optional[List[tuple]] = None, out_dir: str = ".", dot_radius: float = 0.014) -> str:
        """One picture: the infant at q (or at the asked-for frames), the pale asked-for geoms (ghost: own_geoms of the
        asked-for pose) and the target dots [(position, color, body)], each joined by a thin line to its body's origin
        in this picture (no line where body is None). The dots and lines are drawn in a third picture laid over the
        others, so a dot inside the body (a wrist's place lies in the middle of the palm) stays visible."""
        mj, m = self.mj, self.m
        self.place(q, mocap, objpose)
        look, dist, az, el = self.cams[view]
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.renderer.update_scene(self.d, camera=cam, scene_option=self.opt)
        scene = self.renderer.scene
        for flag in (mj.mjtRndFlag.mjRND_SKYBOX, mj.mjtRndFlag.mjRND_SHADOW, mj.mjtRndFlag.mjRND_REFLECTION,
                     mj.mjtRndFlag.mjRND_FOG):
            scene.flags[flag] = False
        moved = {}
        if frames is not None:
            moved = {g: (pos, mat) for g, pos, mat in self.own_geoms(q, frames)}
            self.place(q, mocap, objpose)
        for n in range(scene.ngeom):
            g = scene.geoms[n]
            gid = int(g.objid)
            if int(g.objtype) == int(mj.mjtObj.mjOBJ_GEOM) and int(m.geom_bodyid[gid]) in self.own_set:
                if gid in moved:
                    g.pos[:] = moved[gid][0]
                    g.mat[:] = moved[gid][1]
                c = part_color(self.group[int(m.geom_bodyid[gid])])
                g.rgba[:] = [c[0], c[1], c[2], 1.0]
                g.matid = -1
            elif int(g.objtype) == int(mj.mjtObj.mjOBJ_GEOM) and gid in self.touched:
                pass
            else:
                g.rgba[3] = 0.0                     # the caregiver, the floor, untouched objects, sites: left out
                g.pos[:] = [0.0, 0.0, -50.0]
        pale = [(gid, pos, mat) for gid, pos, mat in ghost or []
                if not _same_place(pos, mat, *(moved.get(gid) or (np.array(self.d.geom_xpos[gid]),
                                                                  np.array(self.d.geom_xmat[gid]).reshape(3, 3))),
                                   within_m=BODY_SAME_M, within_deg=BODY_SAME_DEG)]
        marks = []
        for pos, color, b in dots or []:
            here = None if b is None else (frames[2][b] if frames is not None else np.array(self.d.xpos[b]))
            marks.append((np.asarray(pos, dtype=np.float64), color, here))
        for n in range(scene.ngeom):
            g = scene.geoms[n]
            g.pos[:] = self.Rf.T @ (np.array(g.pos) - self.o)
            g.mat[:] = self.Rf.T @ np.array(g.mat).reshape(3, 3)
        cam.lookat[:] = look
        cam.distance = float(dist)
        cam.azimuth = float(az)
        cam.elevation = float(el)
        mj.mjv_updateCamera(m, self.d, cam, scene)
        scene.nlight = 1
        light = scene.lights[0]
        light.directional = 1
        light.castshadow = 0
        light.pos[:] = [1.0, 0.6, 1.2]
        light.dir[:] = -np.array([1.0, 0.6, 1.2]) / np.linalg.norm([1.0, 0.6, 1.2])
        light.ambient[:] = [0.45, 0.45, 0.45]
        light.diffuse[:] = [0.70, 0.70, 0.70]
        light.specular[:] = [0.10, 0.10, 0.10]
        name = "body_%s_%s.png" % (view, label)
        pixels = np.array(self.renderer.render())
        pixels[(pixels == 0).all(-1)] = BODY_BACKGROUND
        if pale:
            scene.ngeom = 0
            for gid, pos, mat in pale:
                _add_geom(mj, scene, int(m.geom_type[gid]), m.geom_size[gid], self.Rf.T @ (np.asarray(pos) - self.o),
                          self.Rf.T @ np.asarray(mat).reshape(3, 3), BODY_PALE, int(m.geom_dataid[gid]))
            pixels = _blend(pixels, self.renderer.render())
        if marks:
            scene.ngeom = 0
            for pos, color, here in marks:
                at = self.Rf.T @ (pos - self.o)
                _add_geom(mj, scene, int(mj.mjtGeom.mjGEOM_SPHERE), [dot_radius, 0, 0], at, np.eye(3), list(color) + [1.0])
                if here is not None:
                    _add_line(mj, scene, self.Rf.T @ (np.asarray(here) - self.o), at, max(0.0008, dot_radius * 0.18),
                              list(color) + [1.0])
            pixels = _blend(pixels, self.renderer.render(), alpha=1.0)
        _save_png(pixels, os.path.join(out_dir, name))
        return name


def render_command(rid: str, body: int, step: int, out_root: Optional[str] = None, hands_size: int = HAND_SIZE,
                   recording: Optional[dict] = None, lite: bool = False) -> dict:
    """Draw one command (module doc); returns its manifest. Pictures already drawn by this renderer version for the same
    command are reused when their source files are unchanged. LITE (the automatic picks): the front views of both hands
    and the whole body at the start, as requested and at the end only, 9 pictures."""
    t_start = time.perf_counter()
    mem0, gpu0 = process_memory_mb(), gpu_memory_mb()
    rec = recording or S.by_id(rid)
    if rec is None:
        raise C.Refused("unknown recording id %r" % rid)
    key = key_for(rid, body, step, lite)
    out_dir = os.path.join(str(out_root or pictures_dir()), key)
    man_path = os.path.join(out_dir, "manifest.json")
    if os.path.isfile(man_path):
        try:
            with open(man_path, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("version") == RENDER_VERSION and all(
                    os.path.isfile(p) and _sha(p) == h for p, h in old["provenance"]["chunks"].items()):
                old["reused"] = True
                return old
        except (OSError, ValueError, KeyError):
            pass
    data, (a, z), nxt, used = find_command(rec, body, step)
    st = np.asarray(data["step"])
    ep = np.asarray(data["episode"])
    # the model is checked against the header of the chunk holding the command's first record, and every record the
    # drawing uses (the start, the command, the next command) must come from a chunk recorded with that same model
    header = data["headers"][int(data["chunk"][a])]
    span = np.asarray(data["chunk"])[max(0, a - 1):(nxt[1] if nxt is not None else z) + 1]
    other = sorted({str(used[int(c)][0]) for c in span
                    if data["headers"][int(c)].get("model_fingerprint") != header.get("model_fingerprint")})
    if other:
        raise C.Refused("records of this command come from chunks recorded with another model: %s" % ", ".join(other))
    b = rec["bodies"][str(body)]
    model = C.BodyModel(b["model"], header)
    cols = C.measure(data, model, (rec.get("args") or {}).get("position_rows"), keep_from_step=int(st[a]))
    i = int(np.flatnonzero(cols["step"] == int(st[a]))[0])
    q = np.asarray(data["qpos"])
    has_start = bool(cols["has_start"][i])
    q_start = q[a - 1].copy() if has_start else None
    chunk, row = int(data["chunk"][a]), int(data["row"][a])
    motor = np.asarray(data["motors"][chunk][row], dtype=np.float64)
    mask = np.asarray(data["masks"][chunk][row])
    goal = np.asarray(data["goals"][chunk][row], dtype=np.float64) if data["goals"][chunk].size else np.zeros(0)
    has_goal = model.qadr is not None and goal.size == len(model.qadr) and bool(np.all(np.isfinite(goal)))
    q_target = None
    if q_start is not None and has_goal:
        q_target = q_start.copy()
        q_target[np.asarray(model.qadr)] = goal
    nxt_ok = nxt is not None and bool(cols["has_next"][i])
    samples = [(int(round((r - a + 1) * 50)), r) for r in range(a, z + 1)]
    all_samples = list(samples)
    if lite:
        samples = samples[-1:]
    # the saved objects and mocap per record (read again with the same reader, which returns them)
    from mimo_bridge.observer_record import read_chunk
    world = {}
    for ci, c in enumerate(used):
        ch = read_chunk(c[1])
        world[ci] = (ch["mocap"], ch["object_pose"], ch.get("object_flags"))

    def world_at(r: int):
        ci, pi = int(data["chunk"][r]), int(data["pos_in_chunk"][r])
        mo, op, fl = world[ci]
        return mo[pi], op[pi], (fl[pi] if fl is not None else None)

    os.makedirs(out_dir, exist_ok=True)
    images: Dict[str, dict] = {}
    notes: List[str] = []
    if not has_start:
        notes.append("Start: not saved (the record before this command is missing, from another episode or not the "
                     "previous model step)")
    if not has_goal:
        notes.append("Requested hand shape: this command carried no finger target angles")
    hold = (rec.get("args") or {}).get("motor_hold")
    if hold and (z - a + 1) != int(hold):
        notes.append("This command has %d saved model steps, not the run's %d (an episode change, a rest or a gap "
                     "ended it early)" % (z - a + 1, int(hold)))
    if lite:
        notes.append("Automatic drawing: front views at the start, as requested and at the end only. Draw this command "
                     "for every view and every 50 ms sample")
    sample_labels = ["t%03d" % ms for ms, _ in samples]
    hand_states = [("start", q_start), ("target", q_target)] + [(lab, q[r].copy()) for lab, (ms, r) in
                                                                  zip(sample_labels, samples)]
    hands = _Hands(b["model"], size=hands_size)
    try:
        for side in ("left", "right"):
            images.update(hands.draw(side, hand_states, out_dir, views=("front",) if lite else None,
                                     ghost=q_target, ghost_on=sample_labels))
    finally:
        hands.close()
    # the handle targets of the command
    names = model.names
    posmask = C.position_handle_mask(names, (rec.get("args") or {}).get("position_rows"))
    scored = C.scored(names)
    handles = []
    targets: Dict[int, np.ndarray] = {}
    dots_world = []
    for h, name in enumerate(names):
        if h >= motor.shape[0]:
            break
        mk = int(mask[h]) if h < mask.size else -1
        strength = float(motor[h, 8])
        powered = mk != 0 and math.isfinite(strength) and strength > 0
        qv = motor[h, 3:7]
        tR = C.quat_xyzw_to_mat(qv) if np.all(np.isfinite(qv)) and float(np.linalg.norm(qv)) > 1e-9 else None
        if powered and scored[h] and tR is not None:
            targets[int(model.hb[h])] = tR
            if posmask[h] and np.all(np.isfinite(motor[h, 0:3])):
                dots_world.append((motor[h, 0:3] / 100.0, part_color(handle_group(name)), int(model.hb[h])))
        handles.append(dict(index=h, name=name, part=PART_NAMES.get(name), group=handle_group(name) if scored[h] else None,
                            powered=bool(powered), mask=mk, strength=round(strength, 4),
                            scored=bool(scored[h]), position_row=bool(posmask[h] and scored[h]),
                            target_cm=[round(v, 3) for v in motor[h, 0:3]],
                            target_quat_xyzw=[round(v, 5) for v in qv],
                            position_cm=dict(start=_f(cols["pos_s"][i][h]), end=_f(cols["pos_e"][i][h]),
                                             after_next=_f(cols["pos_n"][i][h])),
                            turn_deg=dict(start=_f(cols["rot_s"][i][h]), end=_f(cols["rot_e"][i][h]),
                                          after_next=_f(cols["rot_n"][i][h]))))
    # each powered scored part's error at every sample (the same comparisons as commands.measure)
    keep = [hh for hh in handles if hh["powered"] and hh["scored"]]
    for hh in keep:
        hh["turn_deg"]["samples"] = []
        hh["position_cm"]["samples"] = []
    for ms, r in all_samples:
        xp, xm = model.fk(q[r])
        for hh in keep:
            k = hh["index"]
            Rt = C.quat_xyzw_to_mat(motor[k, 3:7])
            hh["turn_deg"]["samples"].append(_f(C._angles_deg(Rt[None], xm[k][None])[0]))
            hh["position_cm"]["samples"].append(_f(np.linalg.norm(motor[k, 0:3] / 100.0 - xp[k]) * 100.0)
                                                if hh["position_row"] else None)
    # every finger joint at the start, as requested, at each sample and after the next command (degrees)
    fingers = {}
    if model.qadr is not None:
        adr = np.asarray(model.qadr)
        for s, side in (("L", "left"), ("R", "right")):
            idx = model.side_idx.get(s)
            if idx is None:
                continue
            per = []
            for code, fname in FINGER_NAMES:
                js = [k for k in idx if model.joint_names[k].split(":")[-1].split("_")[1] == code]
                joints = []
                for k in js:
                    kind = model.joint_names[k].split(":")[-1].split("_", 2)[2]
                    ad = int(adr[k])
                    joints.append(dict(name=JOINT_WORDS.get(kind, kind), code=model.joint_names[k],
                                       start=_f(math.degrees(q_start[ad])) if q_start is not None else None,
                                       target=_f(math.degrees(goal[k])) if has_goal else None,
                                       samples=[_f(math.degrees(q[r][ad])) for _, r in all_samples],
                                       after_next=_f(math.degrees(q[nxt[1]][ad])) if nxt_ok else None))
                per.append(dict(finger=fname, joints=joints))
            fingers[side] = per
    # the whole body: three views of start, asked for and every sample
    touched_objects = []
    tl, tr_ = np.asarray(data["touch_L"]), np.asarray(data["touch_R"])
    for k in range(tl.shape[1] if tl.ndim == 2 else 0):
        if (tl[a:z + 1, k] | tr_[a:z + 1, k]).any():
            touched_objects.append(k)
    body_ = _Body(b["model"], header, touched_objects)
    posture, weight, asked_for_places = None, None, []
    try:
        ref_q = q_start if q_start is not None else q[a]
        body_.set_frame(ref_q)
        posture = body_.posture()
        q_req = q_target if q_target is not None else ref_q
        frames = body_.requested_frames(q_req, targets)
        ghost = body_.own_geoms(q_req, frames)
        body_.place(ref_q)
        start_xpos = np.array(body_.d.xpos)
        for pos, _, hb in dots_world:
            hh = next(x for x in handles if int(model.hb[x["index"]]) == hb and x["position_row"])
            asked_for_places.append(dict(name=hh["name"], part=hh["part"],
                                         start_cm=_f(np.linalg.norm(pos - start_xpos[hb]) * 100.0) if q_start is not None
                                         else None,
                                         asked_for_body_cm=_f(np.linalg.norm(pos - frames[2][hb]) * 100.0)))
        weight = body_.weight_n()
        sets = [ghost] + [body_.own_geoms(q[r]) for _, r in samples]
        if q_start is not None:
            sets.append(body_.own_geoms(q_start))
        body_.fit(sets, [p for p, _, _ in dots_world])
        views = BODY_VIEWS[:1] if lite else BODY_VIEWS
        for view, _, _ in views:
            if q_start is not None:
                mo, op, _ = world_at(a - 1)
                images[body_.draw("start", view, q_start, mo, op, out_dir=out_dir)] = dict(view=view, state="start")
                images[body_.draw("target", view, q_req, mo, op, frames=frames, dots=dots_world,
                                  out_dir=out_dir)] = dict(view=view, state="target")
            for lab, (ms, r) in zip(sample_labels, samples):
                mo, op, _ = world_at(r)
                images[body_.draw(lab, view, q[r], mo, op, ghost=ghost, dots=dots_world,
                                  out_dir=out_dir)] = dict(view=view, state=lab)
    finally:
        body_.close()
    for name in images:
        images[name]["sha256"] = _sha(os.path.join(out_dir, name))
    # contacts during the command: which object or caregiver part touched which body part, the peak force
    contacts = _contacts(used, data, a, z, header)
    # the objects a hand touched: hold (still touched at the command's end and during the next command), release
    # (touched, then not at the end), rise of the object's root from the start pose (saved contacts and object poses)
    objects_touched = []
    zz = np.asarray(data["obj_z"], dtype=np.float64)
    for k in touched_objects:
        during = tl[a:z + 1, k] | tr_[a:z + 1, k]
        steps_on = [int(v) for v in st[a:z + 1][during]]
        z0 = zz[a - 1, k] if has_start else zz[a, k]
        obj = (header.get("objects") or [{}] * (k + 1))[k]
        objects_touched.append(dict(
            name=str(obj.get("name", k)), free=bool(obj.get("free")),
            hands=("left and right" if tl[a:z + 1, k].any() and tr_[a:z + 1, k].any()
                   else "left" if tl[a:z + 1, k].any() else "right"),
            first_step=steps_on[0], last_step=steps_on[-1], touched_at_end=bool(during[-1]),
            touched_in_next=(bool((tl[nxt[0]:nxt[1] + 1, k] | tr_[nxt[0]:nxt[1] + 1, k]).any()) if nxt_ok else None),
            rise_cm=_f((zz[z, k] - z0) * 100.0) if obj.get("free") else None))
    gpu1, mem1 = gpu_memory_mb(), process_memory_mb()
    model_path = b["model"]
    manifest = dict(
        version=RENDER_VERSION, key=key, recording=rid, label=rec["label"], kind=rec["kind"], run=rec["run"],
        body=int(body), step=int(st[a]), last_step=int(st[z]), start_step=int(st[a - 1]) if has_start else None,
        next_step=int(st[nxt[0]]) if nxt_ok else None,
        episode=int(ep[a]), author=int(cols["author"][i]), time=float(cols["time"][i]),
        rendered_at=time.time(), samples_ms=[ms for ms, _ in samples], sample_steps=[int(st[r]) for _, r in samples],
        all_samples_ms=[ms for ms, _ in all_samples], lite=bool(lite), has_start=has_start, has_hand_goal=bool(has_goal),
        targets_drawn=len(targets), posture=posture, posture_from="start" if q_start is not None else "first sample",
        body_views=[v for v, _, _ in (BODY_VIEWS[:1] if lite else BODY_VIEWS)],
        asked_for_places=asked_for_places, body_weight_n=_f(weight),
        notes=notes, images=images,
        measures={k: dict(start=_f(cols["m_%s_s" % k][i]), end=_f(cols["m_%s_e" % k][i]),
                          after_next=_f(cols["m_%s_n" % k][i]), name=C.MEASURES[k][0], unit=C.MEASURES[k][1])
                  for k in C.MEASURES},
        handles=handles, fingers=fingers,
        contacts=contacts, objects_touched=objects_touched,
        context=dict(rest=bool(cols["rest"][i]), contact_object=bool(cols["contact_object"][i]),
                     contact_caregiver=bool(cols["contact_caregiver"][i]),
                     caregiver_present=bool(cols["care_present"][i]), toy_held=int(cols["toy_held"][i]),
                     hand_touch_left=bool(cols["hand_touch_L"][i]), hand_touch_right=bool(cols["hand_touch_R"][i]),
                     objects_touched_by_a_hand=int(cols["touched"][i]), lift_cm=_f(cols["lift_cm"][i])),
        settings=(rec.get("settings") or {}).get("label"),
        provenance=dict(chunks={c[1]: _sha(c[1]) for c in used},
                        headers={p: _sha(p) for p in sorted({(os.path.dirname(c[1]) + "/" + hd_name(c[1]))
                                                              for c in used})},
                        model=model_path, model_gzip_sha256=_sha(model_path),
                        model_structure_fingerprint=model.fingerprint, renderer=RENDER_VERSION,
                        selection="the command starting at model step %d, body %d, of %s" % (int(st[a]), body,
                                                                                              rec["label"])),
        cost=dict(seconds=round(time.perf_counter() - t_start, 3), process_memory_mb_before=_f(mem0),
                  process_memory_mb_after=_f(mem1), gpu_memory_mb_before=_f(gpu0), gpu_memory_mb_after=_f(gpu1),
                  pictures=len(images), bytes=sum(os.path.getsize(os.path.join(out_dir, n)) for n in images)))
    tmp = part_path(man_path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    replace_file(tmp, man_path)
    manifest["reused"] = False
    return manifest


def hd_name(chunk_path: str) -> str:
    with np.load(chunk_path, allow_pickle=False) as z:
        return str(z["header"])


def _has_body(m, name: str) -> bool:
    try:
        m.body(name)
        return True
    except Exception:
        return False


def _f(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if math.isfinite(x) else None


def _contacts(used: List[list], data: dict, a: int, z: int, header: dict) -> List[dict]:
    """The saved contact pairs between the body and anything else during the command's records: entity names, the
    model steps, the peak normal force (N). Only saved contact records count (the specification: overlap in a picture is
    not a contact)."""
    from mimo_bridge.observer_record import read_chunk
    ents = header.get("entities") or []
    kinds = header.get("entity_kind") or []
    want = {}
    for r in range(a, z + 1):
        want.setdefault(int(data["chunk"][r]), set()).add(int(data["pos_in_chunk"][r]))
    pairs: Dict[tuple, dict] = {}
    for ci, rows in want.items():
        ch = read_chunk(used[ci][1])
        if "c_row" not in ch:
            continue
        for k in range(len(ch["c_row"])):
            if int(ch["c_row"][k]) not in rows:
                continue
            e1, e2 = int(ch["c_e"][k][0]), int(ch["c_e"][k][1])
            k1 = kinds[e1] if e1 < len(kinds) else -1
            k2 = kinds[e2] if e2 < len(kinds) else -1
            if (k1 == 0) == (k2 == 0):
                continue                     # the body against itself, or two other things
            me, other = (e1, e2) if k1 == 0 else (e2, e1)
            key = (me, other)
            f = float(ch["c_fmax"][k])
            step = int(ch["step"][int(ch["c_row"][k])])
            p = pairs.setdefault(key, dict(body_part=ents[me]["name"] if me < len(ents) else str(me),
                                           other=ents[other]["name"] if other < len(ents) else str(other),
                                           other_kind=ents[other]["kind"] if other < len(ents) else "?",
                                           first_step=step, last_step=step, peak_force_n=0.0))
            p["first_step"] = min(p["first_step"], step)
            p["last_step"] = max(p["last_step"], step)
            p["peak_force_n"] = round(max(p["peak_force_n"], f), 3)
    return sorted(pairs.values(), key=lambda p: -p["peak_force_n"])
