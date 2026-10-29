# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""One recording body's commands, and how close the body came to what each command asked for.

INPUT: the body's chunks sorted by model step (sources.py), read with observer_record.read_chunk: per record (one per
model step) the step, episode, author, rest flag, the full qpos, contacts, and the command in force (order_on, order_seq
and the chunk's order rows: order_motor [k, 26, 9], order_mask [k, 26], order_hand_goal [k, 44]).

A COMMAND is a run of consecutive records with the same order_seq, order_on, one episode and consecutive model steps (on
H84 one command is held for 5 model steps, --motor-hold 5). Its START pose is the record just before its first record
(the state the command began from; missing when that record is absent, from another episode or not the previous model
step); its END pose is its last record; AFTER NEXT is the last record of the command that directly follows it.

OPERATION, per command, from forward kinematics of the recording's saved model at each of those poses (mujoco
mj_kinematics: body positions and orientations from the joint positions; no physics step):
  fingers   per hand, the mean over the hand's 22 finger joints of |joint angle - the command's target angle|, degrees
            (the 44 targets of order_hand_goal, mapped to joints by observer.kinematics.hand_goal_qpos; NaN when the
            command carried no finger targets)
  position  per SCORED handle whose position row the decoder solves (the wrists; head and pelvis unless the run used
            --position-rows no_head_pelvis), when powered: the distance from the handle body's origin to the target
            position, cm (the comparison decode() makes for these rows: target cm / 100 - data.xpos[handle body])
  turn      per powered SCORED handle: the angle between the handle body's orientation and the target orientation
            (target quaternion x y z w, action_decoder.quat_xyzw_to_mat), degrees
A handle is POWERED when its mask entry is not 0 and its strength (the 9th number) is above 0, as in decode(). The
SCORED handles are the 15 the controller moves toward the saved target: not the 10 fingertip rows (in every recording
the hand target control replaces the saved fingertip targets, placed on the requested wrist by the learner, with targets
on the actual wrist before the decoder reads them, env.step's hand_target_control.execute; and with the finger joint
servo on the finger joints follow the 44 joint targets instead), and not the eyes' row (gaze: its handle chain moves no
joint; the eyes follow the separate gaze command). Their rows are stored as not scored (NaN).
  objects   per command: which objects a left-hand or right-hand part touched during its records (saved contacts
            only), and the largest rise of a touched FREE object's root (one the body can move: a free joint) from the
            start pose to the end, cm (a lift; NaN when no hand touched a free object). Mounted and mocap objects are
            placed by the world's scripts (the caregiver raises and lowers the mobile's mount), so their rise is never a
            lift. toy_held is the caregiver script's own carry, never a grip.

OUTPUT: one row per command with those values at START, END and AFTER NEXT, the derived per-measure means (MEASURES),
and context: author code, episode, rest, contacts between the body and an object or the caregiver, and the time.
TIME comes from the chunk file times: a chunk is written when its last model step ends, so record i of a chunk of n
records written at t, whose previous chunk of the same process was written at t0, is dated t0 + (t - t0) (i + 1) / n
(between the previous chunk's file time and its own chunk's).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

INDEX_VERSION = "movement_index_v3"

# measure id -> (name on the page, unit, how it is formed from the stored columns)
MEASURES = {
    "fingers_L": ("Left-hand finger shape", "degrees", "mean |finger joint angle - target| over the left hand's 22 joints"),
    "fingers_R": ("Right-hand finger shape", "degrees", "mean |finger joint angle - target| over the right hand's 22 joints"),
    "wrist_L": ("Left wrist position", "cm", "distance from the left wrist to its target position"),
    "wrist_R": ("Right wrist position", "cm", "distance from the right wrist to its target position"),
    "body_turn": ("Whole-body orientation", "degrees",
                  "mean angle between each powered body part's orientation and its target orientation, over the 15 "
                  "parts the controller moves toward a saved target (wrists, arms, legs, feet, head, trunk)"),
}
FINGER_HANDLES = ("thumb", "index", "middle", "ring", "pinky")


def scored(names: List[str]) -> np.ndarray:
    """Per handle, whether it is SCORED (module doc): not a fingertip row, not the eyes' row."""
    return np.asarray([n != "gaze" and n.split("_")[0] not in FINGER_HANDLES for n in names])

# THE DECODER'S DEFINITIONS, copied so the viewer does not import the learner (action_decoder imports lem, which imports
# torch: 620 MB more per process, measured Oct 18, 2026). tests/test_movement_viewer.py holds them equal to
# mimo_bridge/action_decoder.py's POSITION_HANDLES, HEAD_PELVIS_POSITION_HANDLES and quat_xyzw_to_mat.
POSITION_HANDLES = frozenset({"wrist_L", "wrist_R", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
                              "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R", "head", "lower_trunk"})
HEAD_PELVIS_POSITION_HANDLES = frozenset({"head", "lower_trunk"})


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """A target orientation (x, y, z, w) as a rotation matrix: action_decoder.quat_xyzw_to_mat."""
    x, y, z, w = [float(v) for v in q]
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Refused(RuntimeError):
    """The records cannot be read as asked (another model, a missing header): the reason is shown, never hidden."""


# ONE LOADED MODEL PER PROCESS (measured Oct 18, 2026): the H84 world model's MjData reserves a 380 MB arena (the model's
# own memory setting; mjModel.narena cannot be changed from Python), so the index and the pictures share one MjModel and
# one MjData per saved model file, used one after the other. Changes the pictures make to the model are undone after
# each picture set (render.py).
_LOADED: Dict[str, tuple] = {}


def loaded_model(path: str):
    """(MjModel, MjData) of a saved model file, loaded once per process (at most two files kept)."""
    hit = _LOADED.get(path)
    if hit is None:
        import mujoco
        from mimo_bridge.observer_record import load_world_model
        _LOADED.clear()                            # one at a time: a second file would reserve another 380 MB
        m = load_world_model(path)
        hit = _LOADED[path] = (m, mujoco.MjData(m))
    return hit


def release_models() -> None:
    """Free the loaded models (the viewer service calls this when it has been idle)."""
    _LOADED.clear()


class BodyModel:
    """The recording body's saved world model, checked against its recorder header, for forward kinematics."""

    def __init__(self, model_path: str, header: dict, check_fingerprint: bool = True):
        """check_fingerprint False: a model saved beside the records it drew (a probe's own model) is used without
        comparing it with an observer header's fingerprint; its own fingerprint is still computed and reported."""
        import mujoco
        from mimo_bridge.observer_record import structure_fingerprint
        from mimo_bridge.tools.observer.kinematics import hand_goal_qpos
        self.mj = mujoco
        if not model_path or not os.path.isfile(model_path):
            raise Refused("the recording's saved model %s is missing" % model_path)
        self.path = model_path
        self.m, self.d = loaded_model(model_path)
        own = {int(b) for e in header["entities"] if e["kind"] == "self" for b in e["bodies"]}
        self.own = own
        fp = structure_fingerprint(self.m, own)
        if check_fingerprint and fp != header.get("model_fingerprint"):
            raise Refused("the saved model's structure fingerprint %s... is not the header's %s..."
                          % (fp[:12], str(header.get("model_fingerprint"))[:12]))
        if int(self.m.nq) != int(header["nq"]):
            raise Refused("the saved model has %d joint positions, the header %d" % (self.m.nq, header["nq"]))
        self.fingerprint = fp
        self.names = [str(n) for n in header["handle_names"]]
        self.hb = np.asarray(header["handle_bodies"], dtype=np.int64)
        self.qadr = hand_goal_qpos(self.m, self.names, header["handle_bodies"])
        self.side_idx = {}
        if self.qadr is not None:
            jn = []
            for adr in self.qadr:
                j = int(np.flatnonzero(np.asarray(self.m.jnt_qposadr) == adr)[0])
                jn.append(self.m.joint(j).name)
            self.joint_names = jn
            for s, pre in (("L", "left_"), ("R", "right_")):
                self.side_idx[s] = np.asarray([k for k, n in enumerate(jn) if n.split(":")[-1].startswith(pre)])
        else:
            self.joint_names = []

    def fk(self, q: np.ndarray):
        """(handle positions [H, 3] m, handle orientations [H, 3, 3]) at joint positions q."""
        self.d.qpos[:] = q
        self.mj.mj_kinematics(self.m, self.d)
        return (np.asarray(self.d.xpos)[self.hb].copy(),
                np.asarray(self.d.xmat)[self.hb].reshape(-1, 3, 3).copy())


def position_handle_mask(names: List[str], position_rows: Optional[str]) -> np.ndarray:
    """Per handle, whether the decoder solves its position row under the run's --position-rows (action_decoder
    .position_handles: every position handle, without head and pelvis under 'no_head_pelvis')."""
    ph = set(POSITION_HANDLES) - (set(HEAD_PELVIS_POSITION_HANDLES) if position_rows == "no_head_pelvis" else set())
    return np.asarray([n in ph for n in names])


def _quat_mats(motor: np.ndarray) -> np.ndarray:
    out = np.full((motor.shape[0], 3, 3), np.nan)
    for h in range(motor.shape[0]):
        q = motor[h, 3:7]
        if np.all(np.isfinite(q)) and float(np.linalg.norm(q)) > 1e-9:
            out[h] = quat_xyzw_to_mat(q)
    return out


def _angles_deg(Rt: np.ndarray, Rc: np.ndarray) -> np.ndarray:
    """Per handle, the angle of Rt Rc^T in degrees (NaN where Rt is NaN)."""
    tr = np.einsum("hij,hij->h", Rt, Rc)            # trace(Rt Rc^T)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def load_records(chunks: List[list]) -> Dict[str, object]:
    """The records of a list of chunks ([name, path, file time, first step], sorted by model step) concatenated, with the
    per-record pointer to its command's row (chunk number, row) and the per-record time (module doc)."""
    from mimo_bridge.observer_record import read_chunk, orders_in_force
    parts = []
    last_time_of: Dict[int, float] = {}
    per_record_s: Dict[int, List[float]] = {}
    for ci, (name, path, mtime, first) in enumerate(chunks):
        c = read_chunk(path)
        h = c["header"]
        n = len(c["step"])
        pid = int(name.split("_")[2])
        oif = orders_in_force(c)
        kinds = np.asarray(h.get("entity_kind") or [], dtype=np.int64)
        obj = np.zeros(n, dtype=bool)
        care = np.zeros(n, dtype=bool)
        objects = h.get("objects") or []
        K = len(objects)
        obj_of_entity = {int(o["entity"]): k for k, o in enumerate(objects)}
        side_of_entity = {}
        for e_ in h.get("entities") or []:
            g = str(e_.get("group") or "")
            if e_.get("kind") == "self" and (g.startswith(("hand_L", "finger_L")) or g.startswith(("hand_R", "finger_R"))):
                side_of_entity[int(e_["code"])] = 0 if "_L" in g[:8] else 1
        touch = np.zeros((2, n, K), dtype=bool)          # [left, right] hand part against object k, per record
        if "c_row" in c and len(c["c_row"]) and kinds.size:
            e = np.asarray(c["c_e"], dtype=np.int64)
            k1, k2 = kinds[np.clip(e[:, 0], 0, kinds.size - 1)], kinds[np.clip(e[:, 1], 0, kinds.size - 1)]
            selfpair = (k1 == 0) ^ (k2 == 0)
            other = np.where(k1 == 0, k2, k1)
            rows = np.asarray(c["c_row"], dtype=np.int64)
            obj[rows[selfpair & (other == 1)]] = True
            care[rows[selfpair & (other == 2)]] = True
            for r_, a_, b_, ka, kb in zip(rows, e[:, 0], e[:, 1], k1, k2):
                me, it = (a_, b_) if ka == 0 else (b_, a_)
                if (ka == 0) == (kb == 0) or int(kinds[it]) != 1:
                    continue                         # only the body against an object (not the caregiver adult)
                sd, k = side_of_entity.get(int(me)), obj_of_entity.get(int(it))
                if sd is not None and k is not None:
                    touch[sd, int(r_), k] = True
        op = np.asarray(c.get("object_pose", np.zeros((n, 0, 7))))
        obj_z = op[:, :, 2].astype(np.float32) if op.ndim == 3 and op.shape[1] == K else np.full((n, K), np.nan, np.float32)
        t0 = last_time_of.get(pid)
        if t0 is not None and mtime > t0:
            per_record_s.setdefault(pid, []).append((mtime - t0) / n)
        last_time_of[pid] = mtime
        parts.append(dict(
            step=np.asarray(c["step"], dtype=np.int64), episode=np.asarray(c["episode"], dtype=np.int64),
            author=np.asarray(c["author"], dtype=np.int8), rest=np.asarray(c["rest"], dtype=bool),
            bout=np.asarray(c["bout"], dtype=bool), latched=np.asarray(c["latched"], dtype=bool),
            care_present=np.asarray(c.get("caregiver_present", np.zeros(n, bool)), dtype=bool),
            toy_held=np.asarray(c.get("toy_held", np.zeros(n, np.int8)), dtype=np.int8),
            order_on=np.asarray(c.get("order_on", np.zeros(n, bool)), dtype=bool),
            order_seq=np.asarray(c["order_seq"], dtype=np.int64),
            qpos=np.asarray(c["qpos"], dtype=np.float64), obj=obj, care=care,
            touch_L=touch[0], touch_R=touch[1], obj_z=obj_z,
            chunk=np.full(n, ci, dtype=np.int32), pos_in_chunk=np.arange(n, dtype=np.int32),
            row=(oif["index"] if oif is not None else np.full(n, -1, dtype=np.int64)),
            _mtime=mtime, _pid=pid, _t0=t0, _n=n,
            _motor=(oif["motor"] if oif is not None else np.zeros((0, 0, 9), np.float32)),
            _mask=(oif["mask"] if oif is not None else np.zeros((0, 0), np.int8)),
            _goal=(oif["hand_goal"] if oif is not None else np.zeros((0, 0), np.float32)),
            _header=h))
    if not parts:
        raise Refused("no chunks")
    med = {pid: float(np.median(v)) for pid, v in per_record_s.items() if v}
    times = []
    for p in parts:
        n, t, t0 = p["_n"], p["_mtime"], p["_t0"]
        i = np.arange(n)
        if t0 is not None and t > t0:
            times.append(t0 + (t - t0) * (i + 1) / n)
        else:
            dt = med.get(p["_pid"], 0.0)
            times.append(t - dt * (n - 1 - i))
    out: Dict[str, object] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if not k.startswith("_")}
    out["time"] = np.concatenate(times)
    out["motors"] = [p["_motor"] for p in parts]
    out["masks"] = [p["_mask"] for p in parts]
    out["goals"] = [p["_goal"] for p in parts]
    out["headers"] = [p["_header"] for p in parts]
    # the objects the infant can move (a free joint); mounted and mocap objects are placed by the world's scripts (the
    # caregiver raises and lowers the mobile's mount), so their rise is never a lift
    out["obj_free"] = np.asarray([bool(o.get("free")) for o in (parts[0]["_header"].get("objects") or [])], dtype=bool)
    return out


def segment(rec: Dict[str, object]) -> List[tuple]:
    """[(first record, last record)] of every command (module doc)."""
    on = np.asarray(rec["order_on"]) & (np.asarray(rec["order_seq"]) >= 0) & (np.asarray(rec["row"]) >= 0)
    seq, ep, st = np.asarray(rec["order_seq"]), np.asarray(rec["episode"]), np.asarray(rec["step"])
    n = len(on)
    out = []
    i = 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1] and seq[j + 1] == seq[i] and ep[j + 1] == ep[i] and st[j + 1] == st[j] + 1:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def measure(rec: Dict[str, object], model: BodyModel, position_rows: Optional[str],
            keep_from_step: Optional[int] = None) -> Dict[str, np.ndarray]:
    """The command rows (module doc) of the records `rec`; with keep_from_step only commands starting there or later."""
    cmds = segment(rec)
    st, ep = np.asarray(rec["step"]), np.asarray(rec["episode"])
    q = np.asarray(rec["qpos"])
    H = len(model.names)
    posmask = position_handle_mask(model.names, position_rows) & scored(model.names)
    turnmask = scored(model.names)
    fing_idx = {s: [model.names.index("%s_%s" % (f, s)) for f in FINGER_HANDLES if "%s_%s" % (f, s) in model.names]
                for s in ("L", "R")}
    wrist = {s: model.names.index("wrist_" + s) if ("wrist_" + s) in model.names else None for s in ("L", "R")}
    rows: Dict[str, list] = {k: [] for k in (
        "step", "last_step", "n", "chunk", "pos_in_chunk", "time", "author", "episode", "rest", "bout", "latched",
        "care_present", "toy_held", "contact_object", "contact_caregiver", "has_start", "has_next", "next_step",
        "fing_s", "fing_e", "fing_n", "pos_s", "pos_e", "pos_n", "rot_s", "rot_e", "rot_n", "strength", "mask",
        "hand_touch_L", "hand_touch_R", "touched", "lift_cm")}
    fk_cache: Dict[int, tuple] = {}

    def fk(r: int):
        if r not in fk_cache:
            fk_cache[r] = model.fk(q[r])
        return fk_cache[r]

    for ci, (a, b) in enumerate(cmds):
        if keep_from_step is not None and st[a] < keep_from_step:
            continue
        for old in [r for r in fk_cache if r < a - 1]:
            del fk_cache[old]                  # commands are read in record sequence: earlier poses are not read again
        chunk = int(rec["chunk"][a])
        row = int(rec["row"][a])
        motor = np.asarray(rec["motors"][chunk][row], dtype=np.float64)
        Hm = min(H, motor.shape[0])
        mask = np.asarray(rec["masks"][chunk][row])
        goal = np.asarray(rec["goals"][chunk][row], dtype=np.float64) if rec["goals"][chunk].size else np.zeros(0)
        strength = np.full(H, np.nan)
        strength[:Hm] = motor[:Hm, 8]
        mk = np.full(H, -1, dtype=np.int8)
        mk[:min(H, mask.size)] = mask[:min(H, mask.size)]
        powered = (mk != 0) & np.isfinite(strength) & (strength > 0)
        Rt = np.full((H, 3, 3), np.nan)
        Rt[:Hm] = _quat_mats(motor[:Hm])
        tgt = np.full((H, 3), np.nan)
        tgt[:Hm] = motor[:Hm, 0:3] / 100.0
        has_start = a > 0 and ep[a - 1] == ep[a] and st[a - 1] == st[a] - 1
        nxt = cmds[ci + 1] if ci + 1 < len(cmds) else None
        has_next = bool(nxt is not None and nxt[0] == b + 1 and ep[nxt[0]] == ep[b] and st[nxt[0]] == st[b] + 1)
        poses = [a - 1 if has_start else None, b, nxt[1] if has_next else None]
        fingers, posv, rotv = [], [], []
        for r in poses:
            if r is None:
                fingers.append([np.nan, np.nan])
                posv.append(np.full(H, np.nan))
                rotv.append(np.full(H, np.nan))
                continue
            xp, xm = fk(r)
            d = np.linalg.norm(tgt - xp, axis=1) * 100.0
            posv.append(np.where(powered & posmask & np.isfinite(d), d, np.nan))
            ang = _angles_deg(Rt, xm)
            rotv.append(np.where(powered & turnmask & np.isfinite(ang), ang, np.nan))
            fr = []
            for s in ("L", "R"):
                idx = model.side_idx.get(s)
                if model.qadr is None or idx is None or goal.size != len(model.qadr) or not np.all(np.isfinite(goal[idx])):
                    fr.append(np.nan)
                else:
                    adr = np.asarray(model.qadr)[idx]
                    fr.append(float(np.degrees(np.mean(np.abs(q[r][adr] - goal[idx])))))
            fingers.append(fr)
        rows["step"].append(st[a])
        rows["last_step"].append(st[b])
        rows["n"].append(b - a + 1)
        rows["chunk"].append(chunk)
        rows["pos_in_chunk"].append(int(rec["pos_in_chunk"][a]))
        rows["time"].append(float(rec["time"][b]))
        rows["author"].append(int(rec["author"][a]))
        rows["episode"].append(int(ep[a]))
        rows["rest"].append(bool(np.any(rec["rest"][a:b + 1])))
        rows["bout"].append(bool(rec["bout"][a]))
        rows["latched"].append(bool(rec["latched"][a]))
        rows["care_present"].append(bool(np.any(rec["care_present"][a:b + 1])))
        rows["toy_held"].append(int(np.max(rec["toy_held"][a:b + 1])))
        rows["contact_object"].append(bool(np.any(rec["obj"][a:b + 1])))
        rows["contact_caregiver"].append(bool(np.any(rec["care"][a:b + 1])))
        tl = np.asarray(rec["touch_L"][a:b + 1]).any(0)
        tr = np.asarray(rec["touch_R"][a:b + 1]).any(0)
        both = tl | tr
        rows["hand_touch_L"].append(bool(tl.any()))
        rows["hand_touch_R"].append(bool(tr.any()))
        rows["touched"].append(int(both.sum()))
        z = np.asarray(rec["obj_z"], dtype=np.float64)
        z0 = z[a - 1] if has_start else z[a]
        free = np.asarray(rec.get("obj_free", np.ones(both.size, bool)), dtype=bool)[:both.size]
        lifted = both & free
        rise = (z[b] - z0)[lifted] * 100.0 if lifted.any() else np.zeros(0)
        rows["lift_cm"].append(float(np.max(rise)) if rise.size and np.isfinite(rise).any() else np.nan)
        rows["has_start"].append(bool(has_start))
        rows["has_next"].append(has_next)
        rows["next_step"].append(int(st[nxt[0]]) if has_next else -1)
        rows["fing_s"].append(fingers[0])
        rows["fing_e"].append(fingers[1])
        rows["fing_n"].append(fingers[2])
        rows["pos_s"].append(posv[0])
        rows["pos_e"].append(posv[1])
        rows["pos_n"].append(posv[2])
        rows["rot_s"].append(rotv[0])
        rows["rot_e"].append(rotv[1])
        rows["rot_n"].append(rotv[2])
        rows["strength"].append(strength)
        rows["mask"].append(mk)
    out = {}
    dtypes = dict(step=np.int64, last_step=np.int64, n=np.int16, chunk=np.int32, pos_in_chunk=np.int16,
                  time=np.float64, author=np.int8, episode=np.int32, toy_held=np.int8, next_step=np.int64,
                  strength=np.float16, mask=np.int8)
    for k, v in rows.items():
        dt = dtypes.get(k)
        if dt is None:
            dt = np.int8 if k == "touched" else bool if k in ("rest", "bout", "latched", "care_present",
                                                               "contact_object", "contact_caregiver",
                                                               "hand_touch_L", "hand_touch_R",
                               "has_start", "has_next") else np.float32
        if k in ("fing_s", "fing_e", "fing_n"):
            out[k] = np.asarray(v, dtype=dt).reshape(-1, 2)
        elif k in ("pos_s", "pos_e", "pos_n", "rot_s", "rot_e", "rot_n", "strength", "mask"):
            out[k] = np.asarray(v, dtype=dt).reshape(-1, H)
        else:
            out[k] = np.asarray(v, dtype=dt)
    out.update(derived(out, model.names, fing_idx, wrist))
    return out


def derived(cols: Dict[str, np.ndarray], names: List[str], fing_idx: Optional[dict] = None,
            wrist: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """The per-measure columns m_<measure>_<s|e|n> (start, end, after next) of MEASURES."""
    if fing_idx is None:
        fing_idx = {s: [names.index("%s_%s" % (f, s)) for f in FINGER_HANDLES if "%s_%s" % (f, s) in names]
                    for s in ("L", "R")}
    if wrist is None:
        wrist = {s: names.index("wrist_" + s) if ("wrist_" + s) in names else None for s in ("L", "R")}
    out = {}
    n = len(cols["step"])

    def mean_over(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """Per command, the mean of `values` over the handles `keep` marks; NaN when a kept handle has no value at this
        pose (no start pose, no next command) or no handle is kept."""
        v = np.where(keep, values, 0.0)
        cnt = keep.sum(1)
        missing = (keep & ~np.isfinite(values)).any(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.nansum(v, 1) / np.maximum(cnt, 1)
        return np.where((cnt > 0) & ~missing, m, np.nan).astype(np.float32)

    # the handles of the whole-body mean are the SCORED ones with a value at the command's END (powered, with a target;
    # the fingertip and eye rows are stored as not scored); the start and after-next means use the same handles
    rot_keep = np.isfinite(cols["rot_e"].astype(np.float64))
    for when in ("s", "e", "n"):
        f = cols["fing_" + when]
        out["m_fingers_L_" + when] = f[:, 0].astype(np.float32)
        out["m_fingers_R_" + when] = f[:, 1].astype(np.float32)
        p = cols["pos_" + when].astype(np.float64)
        for s in ("L", "R"):
            w = wrist[s]
            out["m_wrist_%s_%s" % (s, when)] = (p[:, w] if w is not None else np.full(n, np.nan)).astype(np.float32)
        out["m_body_turn_" + when] = mean_over(cols["rot_" + when].astype(np.float64), rot_keep)
    return out
