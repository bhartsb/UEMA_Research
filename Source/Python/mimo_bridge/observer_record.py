# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The developmental observer's world recorder: God's-eye measurement, never fed to the learner.

Addendum v4, Part 6 (docs/gain_of_function_v4/UEMA_Embodied_Gain_of_Function_Addendum_v4.html,
sections 22-24 and 34; design in docs/gain_of_function_v4/OBSERVER_DESIGN.md): an external
observer may use MuJoCo's full state to MEASURE the avatar, but none of that state reaches UEMA.
This module is the first layer of that observer, the "raw world stream": it runs inside each world
worker (mimo_bridge/datasource.py), READS the simulator after every physics frame the trainer
asked for, and writes compact chunk files under <run>/observer/world/b<body>/. Nothing it
computes is returned to the worker's reply, the learner or the trainer; the analysis
(mimo_bridge/tools/observer/) runs in its own process and reads the files only.

Read-only on the simulation's state. The MuJoCo calls are mj_contactForce, mj_multiRay, mj_ray,
mj_sizeModel and mj_saveModel. All take the model as const and all but mj_multiRay take the data as
const; mj_multiRay (MuJoCo 3.3.0 declares its mjData writable) uses mjData's scratch stack for its
temporaries, which no step reads back (checked 2026-09-25 by an independent review on a small world:
the state and the stack's high-water mark were identical with and without the recorders). Every
array read from mjData is copied before it is kept; no random number is drawn; no attribute of the
world is written; the step's own arguments (the order a worker was handed) are copied, never written.
tests/test_observer_record.py proves the worker's reply to the learner, and the world's full saved
state, are identical with the recorder on and off.

One record per body per model step (the last physics frame of the trainer's burst), plus
per-frame "fast lane" series inside it (addendum section 24: a 50-100 ms transient must not
vanish into an average). Contacts are aggregated over every frame of the burst, so a touch that
begins and ends between two model steps is still recorded.

What a record holds (schema observer_world_v2 since 2026-09-25; the header JSON beside the chunks
names every column and lists the field groups it recorded in `fields`; read a chunk back with
read_chunk, which reads v1 and v2 chunks alike):
  time        step (the trainer's model step), env frame, simulated seconds, world episode
              (increments on reset, restore and rebuild), the frame count of the burst
  provenance  who wrote the order in force (planner, reflex, babble noise, self choice, the acting
              snapshot, the trainer's scheduled rest hold, a probe, a script, random or limp
              baselines), taken at the burst's first frame; whether this step latched a new
              order; the order's sequence number; babble bout on/off; scheduled rest on/off
  body        the full qpos (every joint of the infant and the world), stored as whole multiples
              of 1e-6 m / 1e-6 rad / 1e-7 (quaternions) and delta-coded in time: with the
              world's compiled model (saved once per kinematic structure as
              models/<fingerprint>.mjb.gz, see structure_fingerprint) the analysis recovers
              every body's pose by forward kinematics, exactly as the simulator placed it
  world       mocap positions and quaternions (the moved ball, critters, caregiver parts), each
              object's root pose as simulated (the caregiver moves the mobile's mount by editing
              the model, which forward kinematics from a saved model cannot know), and each
              object's visible / collidable flags (the caregiver presents and withdraws toys by
              editing their transparency and collision masks)
  effort      the actuator command applied (ctrl), for effort measures
  eye movers  whether the world's spontaneous-saccade generator started a saccade, or its
              peripheral-motion reflex steered the eyes, during the burst (gaze_reflex.py): eye
              movements the learner did not command
  gaze        per eye: a ray bundle (centre ray plus a ring at the fovea's half-angle) with the hit
              entity, distance and, for the painted wall panels, the motif cell hit. A ray that
              first meets a geom the renderer does not draw (alpha 0), or the infant's own head
              and eyeballs (the eye cameras sit inside the head's sphere), is cast on past it, so
              the record matches what the eye's camera could see. v2 adds two INNER rings (2 and 6
              degrees off the axis, 8 rays each, the ray_in_* columns) cast the same way: the ring
              at the fovea's 12-degree half-angle cannot tell whether a toy fills most of the
              fovea (a 6 cm toy covers a majority of the nine rays only within about 15 cm)
  contacts    every pair of entities in contact during the burst: first and last frame, peak and
              mean normal force, the last contact point, and (v2) the NORMAL: the force-weighted
              mean over the burst's frames of the contact normals, pointing from the pair's first
              entity (the lower code) to its second, the direction the first pushes the second.
              Positions around an object's centre cannot tell opposing surfaces apart on the 6 cm
              boxes (grasps read 83-126 degrees against R04's 90-degree line); normals can
  fast lane   per frame of the burst: head accelerometer and gyro magnitude, total force from
              outside the body, the largest single contact force, force on the head (v2: 32-bit;
              the 16-bit field stored a fingertip pushed into the head, above 65,504 N, as infinity)
  order       (v2) the order the world was handed at the burst's first frame, whenever an order
              and not raw actuator commands drove the body and no scheduled rest replaced it: the
              motor targets per handle, the handle mask, the explicit hand joint goal, and the eye
              command of every record. Written as one row per change (order_row); the yoked
              baseline control replays a learner's own orders from these rows
  caregiver   whether a caregiver exists and the adult is present, whether its toy / mobile is
              shown, its scene phase, whether the infant touched the toy, and whether the toy is
              being carried by the infant's hand under the scripted hand-off (0 no, 1 left hand,
              2 right hand): with --toy-hand-off follow the toy is a mocap body that FOLLOWS a
              hand once two parts of it have touched the toy for 0.1 s (caregiver.py), so a
              toy "hold" or "carry" is the script's, never evidence of grip

DISK BUDGET. Disk C: carries the checkpoints. The recorder writes nothing while free space is
below reserve_bytes or while its own folder holds budget_bytes; the steps it drops are written
to gaps.jsonl so a gap is never read as "nothing happened". The analysis digests chunks into
episodes and removes the raw chunks it has digested (tools/observer/daemon.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np

OBSERVER_WORLD_VERSION = "observer_world_v2"
# the field groups a v2 recorder writes (header "fields"); a v1 header has none of them, and a
# reader treats a missing group as NOT RECORDED, never as zero
FIELD_CONTACT_NORMALS = "contact_normals"
FIELD_INNER_RINGS = "inner_rings"
FIELD_ORDERS = "orders"
FIELD_FAST_FLOAT32 = "fast_float32"
FIELDS = (FIELD_CONTACT_NORMALS, FIELD_INNER_RINGS, FIELD_ORDERS, FIELD_FAST_FLOAT32)
# the inner rings of the foveal bundle, degrees off the eye's axis, and rays per ring
INNER_RINGS_DEG = (2.0, 6.0)
INNER_RING_RAYS = 8

AUTHOR_CODES = {"unknown": 0, "planner": 1, "reflex": 2, "reflex_noise": 3, "self_choice": 4,
                "acting_snapshot": 5, "external_rest": 6, "probe": 7, "scripted": 8,
                "random_actions": 9, "limp": 10, "yoked": 11}

# Entity kinds. An entity is what a contact or a gaze ray can land on: one infant body part, one
# object (a whole body tree: all of the mobile's parts are "mobile"), or one static world geom.
KIND_SELF, KIND_OBJECT, KIND_CAREGIVER, KIND_FLOOR, KIND_WALL, KIND_ART, KIND_CEILING, \
    KIND_STATIC = range(8)
KIND_NAMES = ["self", "object", "caregiver", "floor", "wall", "art", "ceiling", "static"]
NO_HIT = -1

# The painted panels (mimo_bridge/pen_art.py): motif order left to right as painted. Each panel
# is a plane whose LONG in-plane axis carries the motifs: local x on the north and south panels
# (size 0.72 x 0.12 m), local y on the east and west ones (0.12 x 0.72), because pen_art rotates
# those two textures to fit the axes MuJoCo chose. Which way along that axis the first motif
# sits was MEASURED by rendering each panel from in front at both ends (2026-09-24, crib scene):
# the first motif is at the negative end on north (A), south (circle) and west (cat), and at the
# positive end on east (1, with 8 at the negative end). tools/observer/validate.py re-checks this
# by rendering, so a scene change that flips a panel fails the validation instead of mislabelling.
ART_MOTIFS = {"n": list("ABCDEFGH"), "e": list("12345678"),
              "s": ["circle", "square", "triangle", "star", "diamond", "heart"],
              "w": ["cat", "fish", "bird", "flower", "tree", "sun"]}
ART_FIRST_AT_NEGATIVE_END = {"n": True, "e": False, "s": True, "w": True}

# quantisation of the stored state: 1 micrometre, 1 microradian, 1e-7 of a unit quaternion.
# Stored as whole multiples, so a record is exact to half of one of these.
POS_QUANTUM = 1e-6
ANGLE_QUANTUM = 1e-6
QUAT_QUANTUM = 1e-7

# the MuJoCo geom colour that means "use the material" (MJCF geom/rgba: a geom rgba different
# from this internal default takes precedence over its material)
_DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])

DEFAULT_BUDGET_BYTES = int(1.5 * 2 ** 30)     # per body folder
DEFAULT_RESERVE_BYTES = 12 * 2 ** 30          # free space on the disk that is never used

# a Windows reader holding a file blocks os.replace (mimo_bridge/event_tape.py, measured
# 2026-09-22); status files are retried over these waits before the write is given up
_REPLACE_WAITS_S = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


def body_group(name: str) -> str:
    """Anatomical group of one MIMo body, the observer's own vocabulary (never the learner's).

    hand_R_palm, finger_R_index / middle / ring / little / thumb, arm_R_upper / lower,
    leg_R_upper / lower, foot_R, head, eye, trunk, root. Unknown names return 'other'."""
    n = name.lower()
    side = "R" if n.startswith("right") else "L" if n.startswith("left") else ""
    if n in ("hip", "lower_body", "upper_body", "chest"):
        return "trunk"
    if n == "head":
        return "head"
    if n.endswith("_eye"):
        return "eye"
    if n in ("mimo_location",):
        return "root"
    if not side:
        return "other"
    if n.endswith("upper_arm"):
        return f"arm_{side}_upper"
    if n.endswith("lower_arm"):
        return f"arm_{side}_lower"
    if n.endswith("_hand") or n.endswith("lfmetacarpal"):
        return f"hand_{side}_palm"
    for key, digit in (("_ff", "index"), ("_mf", "middle"), ("_rf", "ring"), ("_lf", "little"),
                       ("_th", "thumb")):
        if key in n:
            return f"finger_{side}_{digit}"
    if n.endswith("upper_leg"):
        return f"leg_{side}_upper"
    if n.endswith("lower_leg"):
        return f"leg_{side}_lower"
    if n.endswith("foot") or n.endswith("toes") or n.endswith("toe"):
        return f"foot_{side}"
    return "other"


def _object_kind(name: str) -> str:
    n = name.lower()
    if n.startswith("caregiver"):
        return "caregiver"
    if n.startswith("box_"):
        return "box"
    if n == "target":
        return "ball"
    if n == "mobile":
        return "mobile"
    if n in ("squeaker", "dragonfly", "bee"):
        return "critter"
    return "decoration"


def qpos_quantum(m) -> np.ndarray:
    """The storage quantum of every qpos entry, by joint type."""
    import mujoco
    q = np.full(int(m.nq), ANGLE_QUANTUM)
    for j in range(int(m.njnt)):
        a, t = int(m.jnt_qposadr[j]), int(m.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            q[a:a + 3] = POS_QUANTUM
            q[a + 3:a + 7] = QUAT_QUANTUM
        elif t == int(mujoco.mjtJoint.mjJNT_BALL):
            q[a:a + 4] = QUAT_QUANTUM
        elif t == int(mujoco.mjtJoint.mjJNT_SLIDE):
            q[a] = POS_QUANTUM
        else:
            q[a] = ANGLE_QUANTUM
    return q


def pose_quantum(n: int) -> np.ndarray:
    """[n, 7] position + quaternion rows, flattened."""
    return np.tile(np.r_[np.full(3, POS_QUANTUM), np.full(4, QUAT_QUANTUM)], n)


def art_cell(ent: dict, ctr: np.ndarray, R: np.ndarray, point: np.ndarray) -> int:
    """The motif index at a world point on a painted panel (the panel's centre ctr and rotation
    R), or -1 when the point is off the painted area."""
    local = R.T @ (np.asarray(point, dtype=np.float64) - ctr)
    axis = int(ent.get("axis", 0))
    hw = float(ent["half_width_m"])
    n = max(1, len(ent["motifs"]))
    u = (local[axis] + hw) / (2.0 * hw)
    if not (0.0 <= u <= 1.0) or abs(local[1 - axis]) > float(ent["half_height_m"]):
        return -1
    k = int(min(n - 1, max(0, math.floor(u * n))))
    return k if ent.get("first_at_negative_end", True) else n - 1 - k


def ring_directions(fwd: np.ndarray, right: np.ndarray, up: np.ndarray, half_angle_rad: float,
                    n: int) -> List[np.ndarray]:
    """n unit directions at half_angle_rad off the axis fwd, the k-th at 2 pi k / n around it from
    `right` toward `up` (a camera's own axes)."""
    out = []
    for k in range(n):
        ang = 2.0 * math.pi * k / max(1, n)
        v = (math.cos(half_angle_rad) * fwd + math.sin(half_angle_rad) * (math.cos(ang) * right
                                                                          + math.sin(ang) * up))
        out.append(v / np.linalg.norm(v))
    return out


def cast_bundle(m, d, pos: np.ndarray, dirs: List[np.ndarray], exclude_body: int, see_through: set,
                drawn, geom_entity: np.ndarray, art: dict, art_cell_at) -> tuple:
    """Every ray of one eye's bundle from pos along dirs: (entity [R] int16, distance [R] float32,
    painted motif cell [R] int8). A ray that first meets a geom the renderer does not draw
    (drawn(g) False) or a body the eye sees through (its own head and eyeballs) is cast on past it,
    up to six times. mj_ray reads the model and data only; mj_multiRay also uses mjData's scratch
    stack for its temporaries, which no step reads back (module docstring)."""
    import mujoco
    vec = np.ascontiguousarray(np.concatenate(dirs), dtype=np.float64)
    nray = len(dirs)
    geomid = np.full(nray, -1, dtype=np.int32)
    dist = np.full(nray, -1.0, dtype=np.float64)
    gid1 = np.zeros(1, dtype=np.int32)
    mujoco.mj_multiRay(m, d, pos, vec, None, 1, int(exclude_body), geomid, dist, nray, 10.0)
    for r in range(nray):
        # a geom the renderer does not draw, or the eye's own head, hides nothing: cast on past it
        tries = 0
        while geomid[r] >= 0 and tries < 6 and (
                int(m.geom_bodyid[int(geomid[r])]) in see_through or not drawn(int(geomid[r]))):
            v3 = vec[3 * r:3 * r + 3]
            start = pos + (dist[r] + 1e-4) * v3
            more = mujoco.mj_ray(m, d, start, v3, None, 1, int(m.geom_bodyid[int(geomid[r])]), gid1)
            tries += 1
            if more < 0 or gid1[0] < 0:
                geomid[r], dist[r] = -1, -1.0
            else:
                geomid[r], dist[r] = int(gid1[0]), dist[r] + 1e-4 + more
    ent = np.where(geomid >= 0, geom_entity[np.maximum(geomid, 0)], NO_HIT).astype(np.int16)
    cell = np.full(nray, -1, dtype=np.int8)
    for r in range(nray):
        g = int(geomid[r])
        if g in art and dist[r] > 0:
            cell[r] = art_cell_at(g, pos + dist[r] * vec[3 * r:3 * r + 3])
    return ent, dist.astype(np.float32), cell


def geom_alpha(m, g: int) -> float:
    """The transparency the renderer draws geom g with (MJCF: a geom rgba other than the internal
    default takes precedence over the material)."""
    rgba = np.asarray(m.geom_rgba[g])
    mat = int(m.geom_matid[g])
    if mat >= 0 and np.array_equal(rgba, _DEFAULT_RGBA):
        return float(m.mat_rgba[mat][3])
    return float(rgba[3])


def structure_fingerprint(m, own: set) -> str:
    """A hash of what forward kinematics of the infant and the free objects reads: the tree,
    every joint, and the placement and shape of the infant's and free objects' bodies and geoms.
    Mocap-driven and mounted objects (whose poses the record stores each step) and the fields
    the caregiver edits at run time are left out."""
    import mujoco
    free_roots = set()
    for j in range(int(m.njnt)):
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            free_roots.add(int(m.body_rootid[int(m.jnt_bodyid[j])]))
    keep_body = np.array([int(b) in own or int(m.body_rootid[b]) in free_roots
                          for b in range(int(m.nbody))])
    keep_geom = keep_body[np.asarray(m.geom_bodyid, dtype=np.int64)]
    h = hashlib.sha256()
    h.update(("observer_structure_v1:%d:%d:%d:%d:%d" % (m.nq, m.nv, m.nbody, m.njnt, m.ngeom)).encode())
    for arr in (m.body_parentid, m.body_rootid, m.body_jntadr, m.body_jntnum,
                np.asarray(m.body_pos)[keep_body], np.asarray(m.body_quat)[keep_body],
                np.asarray(m.body_ipos)[keep_body], np.asarray(m.body_mass)[keep_body],
                m.jnt_type, m.jnt_qposadr, m.jnt_bodyid, m.jnt_pos, m.jnt_axis, m.jnt_range,
                m.geom_type, m.geom_bodyid, np.asarray(m.geom_size)[keep_geom],
                np.asarray(m.geom_pos)[keep_geom], np.asarray(m.geom_quat)[keep_geom],
                m.cam_bodyid, m.cam_pos, m.cam_quat, m.cam_fovy):
        h.update(np.ascontiguousarray(np.asarray(arr)).tobytes())
    return h.hexdigest()


def load_world_model(path: str):
    """A saved world model (.mjb or .mjb.gz) as an MjModel."""
    import gzip
    import mujoco
    if path.endswith(".gz"):
        with open(path, "rb") as f:
            data = gzip.decompress(f.read())
        return mujoco.MjModel.from_binary_path("world.mjb", {"world.mjb": data})
    return mujoco.MjModel.from_binary_path(path)


def encode_series(x: np.ndarray, quantum: np.ndarray, name: str) -> Dict[str, np.ndarray]:
    """Rows [n, k] as whole multiples of quantum, delta-coded in time (first row absolute).
    Falls back to raw float64 when a value is non-finite or a step exceeds int32."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    if len(x) == 0 or not np.isfinite(x).all():
        return {name + "_raw": x}
    Q = np.rint(x / quantum).astype(np.int64)
    d = np.diff(Q, axis=0)
    if d.size and int(np.abs(d).max()) >= 2 ** 31 - 1:
        return {name + "_raw": x}
    return {name + "_q0": Q[0], name + "_dq": d.astype(np.int32)}


def _same_order(a: dict, b: dict) -> bool:
    def same(x, y):
        if x is None or y is None:
            return x is None and y is None
        # NaN equals NaN here: an order may leave a target unset (NaN), and the same order held over
        # its steps is one row, not one per step
        return x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))
    return same(a["motor"], b["motor"]) and same(a.get("mask"), b.get("mask")) \
        and same(a.get("hand_goal"), b.get("hand_goal"))


def order_columns(orders: List[Optional[dict]], n_handles: int = 0) -> Dict[str, np.ndarray]:
    """The v2 order columns of one chunk from each record's order (WorldRecorder.take_order, or
    None): order_on and order_gaze per record, and one order_row / order_motor / order_mask /
    order_hand_goal row per CHANGE (the chunk's first record with an order always starts one, so a
    chunk reads alone). Shapes that differ between rows are padded with NaN (motor, goal) or -1
    (mask), never refused: watching must not stop the world."""
    n = len(orders)
    on = np.asarray([bool(o is not None and o.get("motor") is not None) for o in orders], dtype=bool)
    gaze = np.full((n, 2), np.nan, dtype=np.float32)
    for i, o in enumerate(orders):
        g = None if o is None else o.get("gaze")
        if g is not None and np.asarray(g).size >= 2:
            gaze[i] = np.asarray(g, dtype=np.float32).reshape(-1)[:2]
    changes, prev = [], None
    for i, o in enumerate(orders):
        cur = o if on[i] else None
        if cur is not None and (prev is None or not _same_order(cur, prev)):
            changes.append(i)
        prev = cur
    H = max([int(orders[i]["motor"].shape[0]) for i in changes] + [int(n_handles)])
    J = max([int(orders[i]["hand_goal"].size) for i in changes if orders[i].get("hand_goal") is not None] + [0])
    motor = np.full((len(changes), H, 9), np.nan, dtype=np.float32)
    mask = np.full((len(changes), H), -1, dtype=np.int8)
    goal = np.full((len(changes), J), np.nan, dtype=np.float32)
    for k, i in enumerate(changes):
        o = orders[i]
        mo = o["motor"]
        motor[k, :mo.shape[0]] = mo
        if o.get("mask") is not None:
            mk = o["mask"][:H]
            mask[k, :mk.size] = mk.astype(np.int8)
        if o.get("hand_goal") is not None:
            goal[k, :o["hand_goal"].size] = o["hand_goal"]
    return dict(order_on=on, order_gaze=gaze, order_row=np.asarray(changes, dtype=np.int32),
                order_motor=motor, order_mask=mask, order_hand_goal=goal)


def orders_in_force(chunk: dict) -> Optional[dict]:
    """Per record of one read chunk, the order in force (the v2 'orders' field group): index [n]
    into the chunk's order rows (-1 where no order drove the body), and the rows themselves
    (motor [k, H, 9], mask [k, H], hand_goal [k, J]) and the eye command per record gaze [n, 2].
    None for a chunk recorded without orders (observer_world_v1)."""
    if "order_on" not in chunk:
        return None
    n = len(chunk["order_on"])
    idx = np.full(n, -1, dtype=np.int64)
    rows = np.asarray(chunk["order_row"], dtype=np.int64)
    on = np.asarray(chunk["order_on"], dtype=bool)
    k = -1
    for i in range(n):
        while k + 1 < rows.size and rows[k + 1] <= i:
            k += 1
        idx[i] = k if (on[i] and k >= 0) else -1
    return dict(index=idx, motor=np.asarray(chunk["order_motor"]), mask=np.asarray(chunk["order_mask"]),
                hand_goal=np.asarray(chunk["order_hand_goal"]), gaze=np.asarray(chunk["order_gaze"]))


def decode_series(z, name: str, quantum: np.ndarray) -> Optional[np.ndarray]:
    files = z.files if hasattr(z, "files") else list(z)
    if name + "_raw" in files:
        return np.asarray(z[name + "_raw"], dtype=np.float64)
    if name + "_q0" not in files:
        return None
    Q = np.concatenate([np.asarray(z[name + "_q0"], dtype=np.int64)[None],
                        np.asarray(z[name + "_dq"], dtype=np.int64)]).cumsum(axis=0)
    return Q * np.asarray(quantum, dtype=np.float64)


_HEADERS: Dict[str, dict] = {}


def read_header(path: str) -> dict:
    h = _HEADERS.get(path)
    if h is None:
        with open(path, encoding="utf-8") as f:
            h = json.load(f)
        _HEADERS[path] = h
    return h


def recorded_fields(header: Optional[dict]) -> set:
    """The v2 field groups a recording's header says it wrote (empty for observer_world_v1)."""
    return set((header or {}).get("fields") or ())


def read_chunk(path: str, header: Optional[dict] = None) -> dict:
    """One world chunk back as arrays: qpos [n, nq], mocap [n, nmocap, 7], object_pose [n, K, 7]
    and the stored columns (contacts as c_* columns). The inverse of WorldRecorder.flush. Reads
    observer_world_v1 and v2 chunks: the v2 columns (c_normal, ray_in_*, order_*) exist only in v2
    chunks, and `fields` names the groups the chunk's header says were recorded."""
    with np.load(path, allow_pickle=False) as z:
        if header is None:
            header = read_header(os.path.join(os.path.dirname(path), str(z["header"])))
        out = {k: np.asarray(z[k]) for k in z.files
               if not k.startswith(("qpos_", "mocap_", "objpose_"))}
        n = len(out["step"])
        out["qpos"] = decode_series(z, "qpos", np.asarray(header["qpos_quantum"]))
        mq = decode_series(z, "mocap", pose_quantum(int(header["nmocap"])))
        out["mocap"] = (np.zeros((n, 0, 7)) if mq is None else mq.reshape(n, int(header["nmocap"]), 7))
        K = len(header["objects"])
        oq = decode_series(z, "objpose", pose_quantum(K))
        out["object_pose"] = np.zeros((n, 0, 7)) if oq is None else oq.reshape(n, K, 7)
    for k in ("fast", "ray_dist", "ctrl", "ray_in_dist", "c_normal"):
        if k in out:
            out[k] = out[k].astype(np.float32)
    out["header_name"] = str(out.pop("header"))
    out["header"] = header
    out["fields"] = recorded_fields(header)
    return out


def _lp(path: str) -> str:
    """The extended-length form of a path past Windows' 260-character limit (the refusal is
    reported as "file not found"; storage_locations.long_path, measured in the H84 launch
    smokes of 2026-09-18). Short paths and other platforms pass unchanged."""
    if os.name == "nt" and len(path) >= 240 and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(path)
    return path


def _atomic_json(path: str, value) -> None:
    path = _lp(path)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f)
    for wait in _REPLACE_WAITS_S:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(wait)
    os.replace(tmp, path)


class WorldRecorder:
    """Reads one world after every physics frame of a burst; writes one record per model step.

    Owned by the world worker, never by BridgeEnv (a recorder must not ride a world snapshot or
    change the env_kwargs a continuation is checked against). Every method is defensive: the
    worker wraps each call, and any failure turns the recorder off with its reason written to
    <dir>/b<body>/recorder_error.json - observing must never stop the body."""

    def __init__(self, out_dir: str, body: int, run: str = "", chunk_steps: int = 100,
                 images_every_s: float = 0.0, ring_rays: int = 8, extra: Optional[dict] = None,
                 budget_bytes: int = DEFAULT_BUDGET_BYTES, reserve_bytes: int = DEFAULT_RESERVE_BYTES):
        self.root = out_dir
        self.dir = os.path.join(out_dir, "b%d" % int(body))
        os.makedirs(self.dir, exist_ok=True)
        self.body = int(body)
        self.run = str(run)
        self.chunk_steps = max(1, int(chunk_steps))
        self.images_every_s = float(images_every_s)
        self.ring_rays = max(0, int(ring_rays))
        self.extra = dict(extra or {})
        self.budget_bytes = int(budget_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.pid = os.getpid()
        self.created = time.time()
        self._model_id = None
        self._header = None
        self._header_name = None
        self._rows: List[dict] = []
        self._contacts: List[tuple] = []
        self._chunk_seq = 0
        self._burst = None
        self._episode = 0
        self._last_frame = None
        # THE SACCADE BASELINE (motor interface audit 2026-09-25, F6): the saccade generator's count
        # when the previous recorded burst ended, and the env frame it ended on. A burst's first
        # on_frame call comes AFTER that frame has stepped, so a saccade that started on it is
        # already counted there; the count at the end of the previous burst is the true baseline
        # whenever this burst's first frame directly follows that burst's last. None = unknown
        # (first burst, a world discontinuity, or frames the recorder did not watch).
        self._saccades_seen = None
        self._saccades_seen_frame = None
        self._last_image_t = -1e18
        self.enabled = True
        self.error = None
        self.records = 0
        self.dropped = 0
        self.written_bytes = 0
        self.cost_s = 0.0

    # ------------------------------------------------------------------ header
    def _build_header(self, env) -> dict:
        import mujoco
        m = env.model
        own = set(int(b) for b in env._own_body_ids)
        root_of = [int(m.body_rootid[b]) for b in range(m.nbody)]
        entities: List[dict] = []
        body_entity = np.full(m.nbody, NO_HIT, dtype=np.int32)
        for b in sorted(own):
            name = m.body(b).name
            body_entity[b] = len(entities)
            entities.append(dict(code=len(entities), name=name, kind="self", group=body_group(name),
                                 bodies=[b]))
        objects = []
        for r in sorted(set(root_of)):
            if r == 0 or r in own:
                continue
            bodies = [b for b in range(m.nbody) if root_of[b] == r]
            name = m.body(r).name
            kind = _object_kind(name)
            code = len(entities)
            for b in bodies:
                body_entity[b] = code
            has_free = any(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
                           for j in range(m.njnt) if int(m.jnt_bodyid[j]) in bodies)
            mass = float(sum(float(m.body_mass[b]) for b in bodies))
            geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies]
            extent = 0.0
            for g in geoms:
                extent = max(extent, float(np.max(np.asarray(m.geom_size[g])[:3])))
            entities.append(dict(code=code, name=name, kind="caregiver" if kind == "caregiver"
                                 else "object", group=kind, bodies=bodies,
                                 mocap=int(m.body_mocapid[r]), free=bool(has_free), mass_kg=mass,
                                 extent_m=extent))
            objects.append(dict(entity=code, name=name, kind=kind, root=r, mocap=int(m.body_mocapid[r]),
                                free=bool(has_free), mass_kg=mass, extent_m=extent, geoms=geoms))
        geom_entity = np.full(m.ngeom, NO_HIT, dtype=np.int32)
        art = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if body_entity[b] >= 0:
                geom_entity[g] = body_entity[b]
                continue
            gname = m.geom(g).name or ("geom%d" % g)
            if gname == "floor":
                kind = "floor"
            elif gname.startswith("pen_wall"):
                kind = "wall"
            elif gname.startswith("pen_art_"):
                kind = "art"
            elif "ceiling" in gname:
                kind = "ceiling"
            else:
                kind = "static"
            code = len(entities)
            geom_entity[g] = code
            ent = dict(code=code, name=gname, kind=kind, group=kind, geoms=[g])
            if kind == "art":
                key = gname[len("pen_art_"):]
                sx, sy = float(m.geom_size[g][0]), float(m.geom_size[g][1])
                ent.update(panel=key, axis=0 if sx >= sy else 1, half_width_m=max(sx, sy),
                           half_height_m=min(sx, sy), motifs=ART_MOTIFS.get(key, []),
                           first_at_negative_end=ART_FIRST_AT_NEGATIVE_END.get(key, True))
                art[g] = ent
            entities.append(ent)
        own_mass = float(sum(float(m.body_mass[b]) for b in own))
        joints = []
        for j in range(m.njnt):
            if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE) or int(m.jnt_bodyid[j]) not in own:
                continue
            lo, hi = (float(x) for x in m.jnt_range[j])
            joints.append(dict(name=m.joint(j).name, qpos=int(m.jnt_qposadr[j]), dof=int(m.jnt_dofadr[j]),
                               body=int(m.jnt_bodyid[j]), limited=bool(m.jnt_limited[j]), range=[lo, hi]))
        cams = {}
        for nm in ("eye_left", "eye_right", "fovea_left", "fovea_right"):
            try:
                cams[nm] = int(m.camera(nm).id)
            except Exception:
                pass
        wide = float(m.cam_fovy[cams["eye_left"]]) if "eye_left" in cams else 60.0
        fovea = (float(m.cam_fovy[cams["fovea_left"]]) if "fovea_left" in cams
                 else float(getattr(env, "fovea_fov", 16.0) or 16.0))
        sensors = {}
        for nm in ("vestibular_acc", "vestibular_gyro"):
            try:
                s = m.sensor(nm)
                sensors[nm] = [int(m.sensor_adr[s.id]), int(m.sensor_dim[s.id])]
            except Exception:
                pass
        names_own = {m.body(b).name for b in own}
        head = int(m.body("head").id) if "head" in names_own else None
        eyes = [int(m.body(n).id) for n in ("left_eye", "right_eye") if n in names_own]
        self._env_own = own
        mjb = self._save_model(m)
        header = dict(
            version=OBSERVER_WORLD_VERSION, run=self.run, body=self.body, pid=self.pid,
            created=self.created, scene=getattr(env, "scene_kind", ""),
            env_dt_s=float(env.dt), physics_dt_s=float(m.opt.timestep),
            nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), nmocap=int(m.nmocap), nbody=int(m.nbody),
            model_file=mjb[0], model_fingerprint=mjb[1], model_sha256=mjb[2],
            qpos_quantum=qpos_quantum(m).tolist(),
            infant_mass_kg=own_mass, weight_n=own_mass * 9.81,
            handle_names=list(env.proprio.handle_names),
            handle_bodies=[int(b) for b in env.proprio.body_ids],
            # the skin patches in the contact head's order (the model lane's contact
            # probabilities are indexed by these), each with the body it covers
            patch_names=list(env.patch_names()),
            patch_bodies=[int(sp.body_id) for sp in env.pooler.specs],
            body_names=[m.body(b).name for b in range(m.nbody)],
            actuator_names=[m.actuator(a).name for a in range(m.nu)],
            body_entity=body_entity.tolist(), geom_entity=geom_entity.tolist(),
            entities=entities, objects=objects,
            joints=joints, cameras=cams, wide_fovy_deg=wide, fovea_fovy_deg=fovea,
            ring_rays=self.ring_rays, ring_half_angle_deg=fovea / 2.0,
            # v2: what this recorder writes beyond v1 (a reader treats a group it does not find
            # here as not recorded), and how
            fields=list(FIELDS),
            inner_rings_deg=list(INNER_RINGS_DEG), inner_ring_rays=INNER_RING_RAYS,
            inner_rays="ray_in_ent / ray_in_dist / ray_in_cell [record, eye, ring x ray]: the rings "
                       "of inner_rings_deg in that order, ray k of each at 2 pi k / inner_ring_rays "
                       "around the eye's axis from the camera's right toward its up (the 12-degree "
                       "ring's own layout); cast like the centre ray",
            contact_normal="c_normal [contact row, 3]: the force-weighted mean over the burst's frames "
                           "of MuJoCo's contact normals, unit length, pointing from the pair's first "
                           "entity (c_e[:, 0], the lower code) to its second: the direction the first "
                           "pushes the second (the unweighted mean where every force was zero)",
            orders=("order_row [k]: records whose order differs from the previous record's; "
                    "order_motor [k, handles, 9] float32: the motor targets as handed (position cm, "
                    "orientation quaternion, radius, strength), order_mask [k, handles] int8 (1 "
                    "powered, 0 masked, -1 no mask given), order_hand_goal [k, J] float32 (NaN: none "
                    "given); order_on [record]: an order drove the body (False where raw actuator "
                    "commands did, a scheduled rest replaced it, or none was given); order_gaze "
                    "[record, 2] float32: the eye command handed (NaN: none)"),
            sensors=sensors, head_body=head, eye_bodies=eyes,
            root_body=int(getattr(env, "_root_body_id", 1)),
            art_panels={str(g): e["panel"] for g, e in art.items()},
            author_codes=AUTHOR_CODES, kind_names=KIND_NAMES,
            fast_lane=["head_acc_ms2", "head_gyro_rads", "outside_force_n", "max_contact_force_n",
                       "head_force_n"],
            fast_dtype="float32",
            object_flags={"1": "visible (drawn with alpha above 0)", "2": "collidable"},
            # how gaze_saccade is counted (motor interface audit F6, 2026-09-25): against the saccade
            # generator's count at the end of the previous recorded burst, so a start on a burst's
            # first frame is flagged. Headers without this key came from the recorder that read the
            # count after the first frame had stepped and missed 1 in 5 starts.
            saccade_flag_baseline="previous_burst_end",
            budget_bytes=self.budget_bytes, reserve_bytes=self.reserve_bytes, extra=self.extra)
        # the entity kind of every code, as a small integer array the reader can index
        header["entity_kind"] = [KIND_NAMES.index(e["kind"]) for e in entities]
        return header

    def _save_model(self, m) -> tuple:
        """The compiled world model for forward kinematics and replay: gzip-compressed (the crib's
        20.8 MB, mostly painted textures, is 0.26 MB compressed) and stored once per kinematic
        STRUCTURE. The caregiver edits some model fields while the world runs (the mobile's
        mount, toy and adult transparency, the adult's drawn arms), so the exact bytes differ
        between moments and processes; the file name is a fingerprint of everything forward
        kinematics of the infant and the free objects reads, which those edits never touch.
        Returns (file relative to the world folder, structure fingerprint, exact bytes hash)."""
        import gzip
        import mujoco
        try:
            buf = np.zeros(int(mujoco.mj_sizeModel(m)), dtype=np.uint8)
            mujoco.mj_saveModel(m, None, buf)
            raw = buf.tobytes()
            exact = hashlib.sha256(raw).hexdigest()
            fp = structure_fingerprint(m, set(int(b) for b in self._env_own))
            folder = _lp(os.path.join(self.root, "models"))
            os.makedirs(folder, exist_ok=True)
            name = fp[:32] + ".mjb.gz"           # 128 bits of the fingerprint name the file
            path = _lp(os.path.join(folder, name))
            if not os.path.exists(path):
                tmp = path + ".%d.part" % self.pid
                with open(tmp, "wb") as f:
                    f.write(gzip.compress(raw, 6))
                try:
                    os.replace(tmp, path)
                except OSError:
                    if os.path.exists(tmp):
                        os.remove(tmp)            # another body saved the same structure first
            return "models/" + name, fp, exact
        except Exception as ex:                    # recorded, never fatal: kinematics then lack it
            return "", "unsaved: %r" % (ex,), ""

    def _write_header(self, header: dict) -> None:
        name = "header_%d_%d.json" % (self.pid, int(self.created * 1000) + self._chunk_seq)
        path = os.path.join(self.dir, name)
        _atomic_json(path, header)
        self._header_name = name

    def _ensure_header(self, env) -> None:
        if self._model_id == id(env.model) and self._header is not None:
            return
        if self._rows:
            self.flush()                      # the rows so far belong to the previous header
        m = env.model
        self._model_id = id(m)
        self._header = self._build_header(env)
        h = self._header
        self._geom_entity = np.asarray(h["geom_entity"], dtype=np.int32)
        self._entity_kind = np.asarray(h["entity_kind"], dtype=np.int8)
        self._self_entities = np.asarray([e["kind"] == "self" for e in h["entities"]])
        self._head_entities = np.asarray([e["kind"] == "self" and e["group"] in ("head", "eye")
                                          for e in h["entities"]])
        self._object_roots = [o["root"] for o in h["objects"]]
        self._object_geoms = [np.asarray(o["geoms"], dtype=np.int64) for o in h["objects"]]
        self._art = {int(g): h["entities"][int(self._geom_entity[int(g)])] for g in h["art_panels"]}
        # the eye cameras sit inside the head's sphere (measured on the crib: 6.7 cm from its
        # centre against a 7.6 cm radius), so a ray from an eye exits through the head; an eye
        # cannot see its own head or eyeballs (no mirror here), and the renderer culls those
        # inside faces, so rays pass through them
        self._see_through = {int(b) for b in ([h["head_body"]] if h["head_body"] is not None else [])
                             + list(h["eye_bodies"])}
        self._write_header(h)

    # ------------------------------------------------------------------ per frame
    def mark(self, event: str, env=None) -> None:
        """A discontinuity in the world: reset, restore, rebuild. The next record starts a new
        world episode, so no detector reads motion across a teleport.

        env (the world just after the event, optional): its saccade count and frame become the next
        burst's baseline, so a saccade that starts on the first frame after the event is flagged -
        after a reset the generator always starts one there (its next switch is frame 0). Without
        env the baseline is unknown and the first burst reads the count after its first frame."""
        self._episode += 1
        self._burst = None
        self._saccades_seen = None            # a rebuilt or restored world restarts the count's story
        self._saccades_seen_frame = None
        if env is not None:
            try:                               # a read only; watching must never stop the world
                sac = getattr(env, "saccades", None)
                if sac is not None:
                    self._saccades_seen = int(getattr(sac, "n_saccades", 0))
                    self._saccades_seen_frame = int(getattr(env, "frame", 0))
            except Exception:
                self._saccades_seen = self._saccades_seen_frame = None

    @staticmethod
    def take_order(order, obs: dict) -> Optional[dict]:
        """A COPY of the order a world was handed (v2 field group 'orders'): order is the step's
        (motor, gaze, handle mask, ctrl, hand joint goal) as the worker received them. The motor
        targets are kept only when they drove the body: not where raw actuator commands (ctrl) did,
        and not where a scheduled rest replaced the order with a postural hold (obs['rest']); the eye
        command is kept whenever one was given. The caller's arrays are copied, never written."""
        if order is None:
            return None
        motor, gaze, mask, ctrl, hand_goal = (tuple(order) + (None,) * 5)[:5]
        g = None
        if gaze is not None:
            g = np.array(gaze, dtype=np.float32).reshape(-1)
            g = g[:2] if g.size >= 2 else None
        if motor is None or ctrl is not None or bool(obs.get("rest", False)):
            return dict(motor=None, gaze=g)
        return dict(motor=np.array(motor, dtype=np.float32).reshape(-1, 9),
                    mask=None if mask is None else np.array(mask, dtype=bool).reshape(-1),
                    hand_goal=(None if hand_goal is None
                               else np.array(hand_goal, dtype=np.float32).reshape(-1)),
                    gaze=g)

    def on_frame(self, env, raw, obs: dict, order=None) -> None:
        """One physics frame of a burst. order (optional): the (motor, gaze, mask, ctrl, hand joint
        goal) the world was handed for this frame; the one at the burst's first frame is recorded
        (take_order)."""
        if not self.enabled:
            self._saccades_seen = None        # a paused recorder cannot know which saccades it missed
            self._saccades_seen_frame = None
            return
        t0 = time.perf_counter()
        self._ensure_header(env)
        frame = int(getattr(env, "frame", 0))
        if self._last_frame is not None and frame < self._last_frame:
            self._episode += 1                # the world reset its frame counter
            self._burst = None
            self._saccades_seen = None
            self._saccades_seen_frame = None
        self._last_frame = frame
        rep = int(obs.get("rep", 0))
        reps = max(1, int(obs.get("reps", 1)))
        if self._burst is None or rep == 0:
            # the burst's provenance is the one in force at its FIRST frame
            sac = getattr(env, "saccades", None)
            now = None if sac is None else int(getattr(sac, "n_saccades", 0))
            # THIS FRAME HAS ALREADY STEPPED: a saccade that started on it is in `now`, and reading
            # the baseline here left 1 in 5 saccade starts unflagged (audit F6). The count at the end
            # of the previous recorded burst is the baseline when this frame follows that burst's
            # last frame directly; after a gap (the recorder was off, a step went unobserved, the
            # world was reset) the frames between were not watched and `now` stands, as before.
            base = now
            if (now is not None and self._saccades_seen is not None
                    and self._saccades_seen_frame is not None
                    and frame == self._saccades_seen_frame + 1):
                base = min(self._saccades_seen, now)
            self._burst = dict(contacts={}, head_acc=[], head_gyro=[], support=[], max_force=[],
                               head_force=[], frames=0, obs=dict(obs), latched=False, reflex=False,
                               saccades0=base, order=self.take_order(order, obs))
        self._burst["latched"] = self._burst["latched"] or bool(obs.get("latched", False))
        rx = getattr(env, "reflex", None)
        if rx is not None:
            try:
                self._burst["reflex"] = self._burst["reflex"] or bool(rx.active)
            except Exception:
                pass
        self._frame_accumulate(env)
        if rep >= reps - 1:
            self._finalize(env, raw, obs)
            self._burst = None
            # the next burst's baseline: the generator's count after this burst's last frame
            sac = getattr(env, "saccades", None)
            self._saccades_seen = None if sac is None else int(getattr(sac, "n_saccades", 0))
            self._saccades_seen_frame = frame
        self.cost_s += time.perf_counter() - t0

    def _frame_accumulate(self, env) -> None:
        import mujoco
        m, d = env.model, env.data
        b = self._burst
        b["frames"] += 1
        fi = b["frames"] - 1
        sens = self._header["sensors"]
        if "vestibular_acc" in sens:
            a, n = sens["vestibular_acc"]
            b["head_acc"].append(float(np.linalg.norm(np.asarray(d.sensordata[a:a + n]))))
        if "vestibular_gyro" in sens:
            a, n = sens["vestibular_gyro"]
            b["head_gyro"].append(float(np.linalg.norm(np.asarray(d.sensordata[a:a + n]))))
        force = np.zeros(6)
        support = 0.0
        peak = 0.0
        head = 0.0
        ge = self._geom_entity
        selfmask = self._self_entities
        for i in range(int(d.ncon)):
            c = d.contact[i]
            e1, e2 = int(ge[int(c.geom1)]), int(ge[int(c.geom2)])
            if e1 < 0 or e2 < 0:
                continue
            s1, s2 = bool(selfmask[e1]), bool(selfmask[e2])
            if e1 == e2 and not s1:
                continue                      # one object's own parts (the mobile's arms)
            k1, k2 = int(self._entity_kind[e1]), int(self._entity_kind[e2])
            # static-static and world-world pairs say nothing about the body or an object
            if not (s1 or s2) and k1 != KIND_OBJECT and k2 != KIND_OBJECT \
                    and k1 != KIND_CAREGIVER and k2 != KIND_CAREGIVER:
                continue
            mujoco.mj_contactForce(m, d, i, force)
            fn = float(abs(force[0]))
            key = (e1, e2) if e1 <= e2 else (e2, e1)
            # MuJoCo's normal (the contact frame's first row) points from geom1 to geom2; the pair's
            # normal points from its first entity (the lower code) to its second
            nrm = np.array(c.frame[:3], dtype=np.float64)
            if e1 > e2:
                nrm = -nrm
            rec = b["contacts"].get(key)
            pos = np.asarray(c.pos, dtype=np.float64)
            if rec is None:
                b["contacts"][key] = [fi, fi, fn, fn, 1, pos.copy(), fn * nrm, nrm]
            else:
                rec[1] = fi
                rec[2] = max(rec[2], fn)
                rec[3] += fn
                rec[4] += 1
                rec[5] = pos.copy()
                rec[6] = rec[6] + fn * nrm
                rec[7] = rec[7] + nrm
            peak = max(peak, fn)
            if s1 != s2:                      # the body against anything that is not itself
                support += fn
                if (s1 and self._head_entities[e1]) or (s2 and self._head_entities[e2]):
                    head += fn
        b["support"].append(support)
        b["max_force"].append(peak)
        b["head_force"].append(head)

    # ------------------------------------------------------------------ per model step
    @staticmethod
    def _alpha(m, g: int) -> float:
        """The transparency the renderer draws geom g with (geom_alpha)."""
        return geom_alpha(m, g)

    def _gaze(self, env) -> dict:
        """Per eye: the v1 bundle (the centre ray and the ring at the fovea's half-angle, cast in one
        call exactly as before) and the v2 inner rings (INNER_RINGS_DEG, cast in a second call, so
        the v1 columns keep their values)."""
        m, d = env.model, env.data
        cams = self._header["cameras"]
        out = {}
        half = math.radians(self._header["ring_half_angle_deg"])
        eyes_body = self._header["eye_bodies"]

        def drawn(g: int) -> bool:
            return not (self._alpha(m, g) <= 0.0)

        def cell_at(g: int, point: np.ndarray) -> int:
            return self._art_cell(d, g, point)
        for side, cam_name in (("L", "eye_left"), ("R", "eye_right")):
            cid = cams.get(cam_name)
            if cid is None:
                continue
            pos = np.array(d.cam_xpos[cid], dtype=np.float64)
            R = np.array(d.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
            fwd, right, up = -R[:, 2], R[:, 0], R[:, 1]    # MuJoCo cameras look along -z
            exclude = eyes_body[0 if side == "L" else 1] if len(eyes_body) == 2 else -1
            dirs = [fwd] + ring_directions(fwd, right, up, half, self.ring_rays)
            ent, dist, cell = cast_bundle(m, d, pos, dirs, exclude, self._see_through, drawn,
                                          self._geom_entity, self._art, cell_at)
            inner = [v for deg in INNER_RINGS_DEG
                     for v in ring_directions(fwd, right, up, math.radians(deg), INNER_RING_RAYS)]
            ient, idist, icell = cast_bundle(m, d, pos, inner, exclude, self._see_through, drawn,
                                             self._geom_entity, self._art, cell_at)
            out[side] = dict(ent=ent, dist=dist, cell=cell, in_ent=ient, in_dist=idist, in_cell=icell)
        return out

    def _art_cell(self, d, g: int, point: np.ndarray) -> int:
        """The motif cell a point on a painted panel falls in (-1 off the painted area)."""
        return art_cell(self._art[g], np.asarray(d.geom_xpos[g]),
                        np.asarray(d.geom_xmat[g]).reshape(3, 3), point)

    def _finalize(self, env, raw, obs: dict) -> None:
        m, d = env.model, env.data
        b = self._burst
        start = b.get("obs") or obs
        objpose = np.zeros((len(self._object_roots), 7), dtype=np.float64)
        objflag = np.zeros(len(self._object_roots), dtype=np.uint8)
        for k, root in enumerate(self._object_roots):
            objpose[k, :3] = d.xpos[root]
            objpose[k, 3:] = d.xquat[root]
            gs = self._object_geoms[k]
            if len(gs):
                if max(self._alpha(m, int(g)) for g in gs) > 0.0:
                    objflag[k] |= 1
                if bool((np.asarray(m.geom_contype[gs]) != 0).any()
                        or (np.asarray(m.geom_conaffinity[gs]) != 0).any()):
                    objflag[k] |= 2
        gaze = self._gaze(env)
        cg = getattr(env, "caregiver", None)
        cg_state = {}
        if cg is not None:
            st = getattr(cg, "state", {}) or {}
            grip = st.get("grip") or {}
            side = str(grip.get("side") or "")
            cg_state = dict(present=float(st.get("adult_visibility", 1.0) or 0.0) > 0.0,
                            object_visible=bool(st.get("visible", False)),
                            phase=str(st.get("scene_phase", "")),
                            last_contact=bool(st.get("last_contact", False)),
                            toy_held=(0 if not grip.get("held") else 1 if side.startswith("l")
                                      else 2 if side.startswith("r") else 3))
        R = 1 + self.ring_rays
        RI = len(INNER_RINGS_DEG) * INNER_RING_RAYS

        def _int(v, default):
            try:
                return int(v)
            except (TypeError, ValueError):
                return default
        row = dict(
            step=_int(obs.get("step"), -1), frame=int(getattr(env, "frame", 0)),
            sim_time=float(d.time), episode=int(self._episode), frames=int(b["frames"]),
            author=int(AUTHOR_CODES.get(str(start.get("author") or "unknown"), 0)),
            latched=bool(b["latched"]), order_seq=_int(start.get("order_seq"), -1),
            bout=bool(start.get("bout", False)), rest=bool(start.get("rest", False)),
            qpos=np.array(d.qpos, dtype=np.float64),
            mocap=(np.concatenate([np.asarray(d.mocap_pos), np.asarray(d.mocap_quat)], axis=1)
                   .astype(np.float64) if int(m.nmocap) else np.zeros((0, 7))),
            objpose=objpose, objflag=objflag,
            ctrl=np.array(d.ctrl, dtype=np.float32),
            ray_ent=np.stack([gaze[s]["ent"] if s in gaze else np.full(R, NO_HIT, np.int16)
                              for s in ("L", "R")]),
            ray_dist=np.stack([gaze[s]["dist"] if s in gaze else np.full(R, -1.0, np.float32)
                               for s in ("L", "R")]),
            ray_cell=np.stack([gaze[s]["cell"] if s in gaze else np.full(R, -1, np.int8)
                               for s in ("L", "R")]),
            ray_in_ent=np.stack([gaze[s]["in_ent"] if s in gaze else np.full(RI, NO_HIT, np.int16)
                                 for s in ("L", "R")]),
            ray_in_dist=np.stack([gaze[s]["in_dist"] if s in gaze else np.full(RI, -1.0, np.float32)
                                  for s in ("L", "R")]),
            ray_in_cell=np.stack([gaze[s]["in_cell"] if s in gaze else np.full(RI, -1, np.int8)
                                  for s in ("L", "R")]),
            order=b.get("order"),
            fast=np.array([self._pad(b["head_acc"]), self._pad(b["head_gyro"]), self._pad(b["support"]),
                           self._pad(b["max_force"]), self._pad(b["head_force"])], dtype=np.float32),
            caregiver=cg is not None,
            caregiver_present=bool(cg_state.get("present", False)),
            caregiver_object_visible=bool(cg_state.get("object_visible", False)),
            caregiver_contact=bool(cg_state.get("last_contact", False)),
            caregiver_phase=str(cg_state.get("phase", "")),
            toy_held=int(cg_state.get("toy_held", 0)),
            # the world's own eye movers (gaze_reflex.py): a spontaneous saccade began during
            # this burst, or the peripheral-motion reflex was steering the eyes
            gaze_saccade=(b["saccades0"] is not None and getattr(env, "saccades", None) is not None
                          and int(getattr(env.saccades, "n_saccades", 0)) != b["saccades0"]),
            gaze_reflex=bool(b["reflex"]),
        )
        idx = len(self._rows)
        for (e1, e2), (f0, f1, fmax, fsum, nfr, pos, nw, nu) in b["contacts"].items():
            # the force-weighted mean normal; the unweighted mean where every force was zero
            nrm = nw if float(np.linalg.norm(nw)) > 1e-12 else nu
            ln = float(np.linalg.norm(nrm))
            nrm = nrm / ln if ln > 1e-12 else np.full(3, np.nan)
            self._contacts.append((idx, e1, e2, f0, f1, fmax, fsum / max(1, nfr), nfr,
                                   float(pos[0]), float(pos[1]), float(pos[2]),
                                   float(nrm[0]), float(nrm[1]), float(nrm[2])))
        self._rows.append(row)
        self.records += 1
        self._maybe_image(raw, row)
        if len(self._rows) >= self.chunk_steps:
            self.flush()

    def _pad(self, xs: list, n: int = 8) -> list:
        # the burst is action_repeat frames (5 on H84); rows keep a fixed width of 8 frames,
        # unused slots NaN, so a burst length change does not change the file's schema
        xs = list(xs)[:n]
        return xs + [float("nan")] * (n - len(xs))

    # ------------------------------------------------------------------ disk budget
    def _used_bytes(self) -> int:
        total = 0
        for folder in (self.dir, os.path.join(self.dir, "images")):
            try:
                with os.scandir(folder) as it:
                    for e in it:
                        if e.is_file():
                            total += e.stat().st_size
            except OSError:
                pass
        return total

    def space_refusal(self, need: int) -> Optional[str]:
        """Why writing `need` more bytes now would break the disk budget, or None."""
        try:
            free = shutil.disk_usage(self.dir).free
        except OSError:
            free = None
        if free is not None and free - need < self.reserve_bytes:
            return ("disk has %.1f GiB free; the observer never takes the last %.1f GiB"
                    % (free / 2 ** 30, self.reserve_bytes / 2 ** 30))
        used = self._used_bytes()
        if used + need > self.budget_bytes:
            return ("this body's observer folder holds %.2f GiB of its %.2f GiB budget; the "
                    "analysis has not digested it" % (used / 2 ** 30, self.budget_bytes / 2 ** 30))
        return None

    def _gap(self, first: int, last: int, rows: int, reason: str) -> None:
        self.dropped += rows
        try:
            with open(os.path.join(self.dir, "gaps.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(time=time.time(), pid=self.pid, first_step=first,
                                        last_step=last, rows=rows, reason=reason)) + "\n")
        except OSError:
            pass

    def _maybe_image(self, raw, row) -> None:
        if self.images_every_s <= 0 or raw is None:
            return
        now = time.time()
        if now - self._last_image_t < self.images_every_s:
            return
        img = getattr(raw, "eye_left", None)
        if img is None:
            return
        self._last_image_t = now
        try:
            if self.space_refusal(256 * 1024):
                return
            from PIL import Image
            folder = os.path.join(self.dir, "images")
            os.makedirs(folder, exist_ok=True)
            stem = "step_%09d_frame_%d" % (row["step"], row["frame"])
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(os.path.join(folder, stem + ".jpg"),
                                                                  quality=85)
            with open(os.path.join(folder, stem + ".json"), "w", encoding="utf-8") as f:
                json.dump(dict(step=row["step"], frame=row["frame"], sim_time=row["sim_time"],
                               episode=row["episode"],
                               left_eye_ray_entities=[int(x) for x in row["ray_ent"][0]],
                               left_eye_ray_cells=[int(x) for x in row["ray_cell"][0]],
                               header=self._header_name,
                               purpose="human audit of the gaze geometry only; never read by the learner"), f)
        except Exception:
            pass                              # an audit picture is never worth stopping for

    # ------------------------------------------------------------------ files
    def flush(self) -> Optional[str]:
        if not self._rows:
            return None
        rows, contacts = self._rows, self._contacts
        self._rows, self._contacts = [], []
        first, last = int(rows[0]["step"]), int(rows[-1]["step"])
        # a generous estimate of this chunk's size (measured: under 1 KB per record)
        refusal = self.space_refusal(4096 * len(rows) + 64 * len(contacts))
        if refusal:
            self._gap(first, last, len(rows), refusal)
            self._status(refusal)
            return None
        h = self._header
        a: Dict[str, np.ndarray] = {}
        a["step"] = np.asarray([r["step"] for r in rows], dtype=np.int64)
        a["frame"] = np.asarray([r["frame"] for r in rows], dtype=np.int64)
        a["episode"] = np.asarray([r["episode"] for r in rows], dtype=np.int32)
        a["frames"] = np.asarray([r["frames"] for r in rows], dtype=np.int8)
        a["author"] = np.asarray([r["author"] for r in rows], dtype=np.int8)
        a["order_seq"] = np.asarray([r["order_seq"] for r in rows], dtype=np.int64)
        for key in ("latched", "bout", "rest", "caregiver", "caregiver_present",
                    "caregiver_object_visible", "caregiver_contact", "gaze_saccade", "gaze_reflex"):
            a[key] = np.asarray([r[key] for r in rows], dtype=bool)
        a["toy_held"] = np.asarray([r["toy_held"] for r in rows], dtype=np.int8)
        a["sim_time"] = np.asarray([r["sim_time"] for r in rows], dtype=np.float64)
        a.update(encode_series(np.stack([r["qpos"] for r in rows]), np.asarray(h["qpos_quantum"]), "qpos"))
        nm = int(h["nmocap"])
        if nm:
            a.update(encode_series(np.stack([r["mocap"].reshape(-1) for r in rows]), pose_quantum(nm),
                                   "mocap"))
        K = len(h["objects"])
        if K:
            a.update(encode_series(np.stack([r["objpose"].reshape(-1) for r in rows]), pose_quantum(K),
                                   "objpose"))
            a["object_flags"] = np.stack([r["objflag"] for r in rows]).astype(np.uint8)
        a["ctrl"] = np.stack([r["ctrl"] for r in rows]).astype(np.float16)
        a["ray_ent"] = np.stack([r["ray_ent"] for r in rows]).astype(np.int16)
        a["ray_dist"] = np.stack([r["ray_dist"] for r in rows]).astype(np.float16)
        a["ray_cell"] = np.stack([r["ray_cell"] for r in rows]).astype(np.int8)
        a["ray_in_ent"] = np.stack([r["ray_in_ent"] for r in rows]).astype(np.int16)
        a["ray_in_dist"] = np.stack([r["ray_in_dist"] for r in rows]).astype(np.float16)
        a["ray_in_cell"] = np.stack([r["ray_in_cell"] for r in rows]).astype(np.int8)
        # 32-bit since v2: float16 ends at 65,504, and a fingertip pushed into the head's sphere
        # (validation case b04_fingertip_to_mouth) made a larger contact force, stored as infinity
        a["fast"] = np.stack([r["fast"] for r in rows]).astype(np.float32)
        phases = sorted({r["caregiver_phase"] for r in rows})
        a["caregiver_phase_names"] = np.asarray(phases if phases else [""])
        a["caregiver_phase"] = np.asarray([phases.index(r["caregiver_phase"]) if phases else 0
                                           for r in rows], dtype=np.int8)
        c = np.asarray(contacts, dtype=np.float64).reshape(-1, 14)
        a["c_row"] = c[:, 0].astype(np.int32)
        a["c_e"] = c[:, 1:3].astype(np.int16)
        a["c_frame"] = c[:, 3:5].astype(np.int8)
        a["c_fmax"] = c[:, 5].astype(np.float32)
        a["c_fmean"] = c[:, 6].astype(np.float32)
        a["c_nframes"] = c[:, 7].astype(np.int8)
        a["c_pos"] = c[:, 8:11].astype(np.float32)
        a["c_normal"] = c[:, 11:14].astype(np.float16)
        a.update(order_columns([r.get("order") for r in rows], len(h.get("handle_names") or [])))
        a["header"] = np.asarray(self._header_name)
        name = "chunk_%09d_%d_%05d.npz" % (first, self.pid, self._chunk_seq)
        self._chunk_seq += 1
        path = _lp(os.path.join(self.dir, name))
        tmp = path[:-4] + ".part.npz"
        np.savez_compressed(tmp, **a)
        os.replace(tmp, path)
        try:
            self.written_bytes += os.path.getsize(path)
        except OSError:
            pass
        self._status(None)
        return path

    def _status(self, refusal: Optional[str]) -> None:
        try:
            _atomic_json(os.path.join(self.dir, "status_%d.json" % self.pid), dict(
                time=time.time(), run=self.run, body=self.body, pid=self.pid, records=self.records,
                dropped=self.dropped, written_bytes=self.written_bytes,
                ms_per_record=(1000.0 * self.cost_s / self.records) if self.records else None,
                header=self._header_name, enabled=self.enabled, error=self.error,
                refusal=refusal))
        except OSError:
            pass

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.enabled = False
            self._status(None)

    def fail(self, reason: str) -> None:
        self.enabled = False
        self.error = reason
        try:
            _atomic_json(os.path.join(self.dir, "recorder_error.json"),
                         dict(time=time.time(), pid=self.pid, reason=reason))
        except Exception:
            pass
