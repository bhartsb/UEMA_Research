# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Per-step measurements of one body over a window of its record: the "track".

Built from decoded world chunks (mimo_bridge.observer_record.read_chunk) that share one header,
with forward kinematics from the saved model (kinematics.py). Every array's first axis is the
record (one per model step, 50 ms on H84). Units: metres, degrees, seconds, newtons.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .kinematics import DIGITS, FINGERS, SIDES, Kinematics

KIND_SELF, KIND_OBJECT, KIND_CAREGIVER, KIND_FLOOR, KIND_WALL, KIND_ART, KIND_CEILING, KIND_STATIC = range(8)
BACKGROUND_KINDS = (KIND_FLOOR, KIND_WALL, KIND_CEILING, KIND_STATIC)


class Entities:
    """What each entity code is (from the world header)."""

    def __init__(self, header: dict):
        ents = header["entities"]
        self.n = len(ents)
        self.names = [e["name"] for e in ents]
        self.kind = np.asarray(header["entity_kind"], dtype=np.int8)
        self.group = [e.get("group", "") for e in ents]
        self.is_self = self.kind == KIND_SELF
        # hand parts: the palm (and little-finger metacarpal) and every finger body, per side
        self.hand = {s: np.asarray([self.is_self[i] and (g.startswith("hand_" + s) or g.startswith("finger_" + s))
                                    for i, g in enumerate(self.group)]) for s in SIDES}
        self.palm = {s: np.asarray([self.is_self[i] and g == "hand_%s_palm" % s for i, g in enumerate(self.group)])
                     for s in SIDES}
        self.digit = {(s, d): np.asarray([self.is_self[i] and g == "finger_%s_%s" % (s, d)
                                          for i, g in enumerate(self.group)])
                      for s in SIDES for d in DIGITS}
        self.fingertip = {(s, d): np.asarray([bool(self.digit[(s, d)][i]) and "distal" in self.names[i]
                                              for i in range(self.n)])
                          for s in SIDES for d in DIGITS}
        self.thumb = {s: self.digit[(s, "thumb")] for s in SIDES}
        self.fingers = {s: np.logical_or.reduce([self.digit[(s, d)] for d in FINGERS]) for s in SIDES}
        self.head = np.asarray([self.is_self[i] and g in ("head", "eye") for i, g in enumerate(self.group)])
        self.arm = {s: np.asarray([self.is_self[i] and g.startswith("arm_" + s) for i, g in enumerate(self.group)])
                    for s in SIDES}
        self.leg = {s: np.asarray([self.is_self[i] and (g.startswith("leg_" + s) or g == "foot_" + s)
                                   for i, g in enumerate(self.group)]) for s in SIDES}
        self.foot = {s: np.asarray([self.is_self[i] and g == "foot_" + s for i, g in enumerate(self.group)])
                     for s in SIDES}
        self.trunk = np.asarray([self.is_self[i] and g in ("trunk", "root") for i, g in enumerate(self.group)])
        self.floor = self.kind == KIND_FLOOR
        self.object_of_entity = {int(o["entity"]): k for k, o in enumerate(header["objects"])}
        self.objects = header["objects"]

    def label(self, e: int) -> str:
        return self.names[int(e)] if 0 <= int(e) < self.n else "none"

    def side_of_hand(self, e: int) -> Optional[str]:
        for s in SIDES:
            if self.hand[s][int(e)]:
                return s
        return None


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _rot_angle_deg(R0: np.ndarray, R1: np.ndarray) -> float:
    R = R0.T @ R1
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    d = abs(float(np.dot(q0, q1)) / max(1e-12, float(np.linalg.norm(q0) * np.linalg.norm(q1))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


class Track:
    """One body's measured steps over a window (chunks sharing one header)."""

    def __init__(self, chunks: List[dict], kin: Kinematics, pid: int = 0, live: Optional[np.ndarray] = None):
        assert chunks, "a track needs at least one chunk"
        h = chunks[0]["header"]
        self.header = h
        self.kin = kin
        self.ent = Entities(h)
        self.pid = int(pid)
        cat = lambda k: np.concatenate([c[k] for c in chunks])
        n_each = [len(c["step"]) for c in chunks]
        self.n = int(sum(n_each))
        for k in ("step", "sim_time", "episode", "frames", "author", "latched", "order_seq", "bout",
                  "rest", "fast", "ray_ent", "ray_dist", "ray_cell", "qpos", "mocap", "object_pose",
                  "ctrl"):
            setattr(self, k, cat(k))
        for k in ("caregiver", "caregiver_present", "caregiver_object_visible", "caregiver_contact",
                  "gaze_saccade", "gaze_reflex"):
            setattr(self, k, cat(k) if k in chunks[0] else np.zeros(self.n, dtype=bool))
        self.toy_held = cat("toy_held") if "toy_held" in chunks[0] else np.zeros(self.n, dtype=np.int8)
        K = len(h["objects"])
        self.object_flags = cat("object_flags") if K else np.zeros((self.n, 0), dtype=np.uint8)
        self.live = np.ones(self.n, dtype=bool) if live is None else np.asarray(live, dtype=bool)
        # the recorder's v2 field groups (observer_record.FIELDS): a group this track's chunks do not
        # all carry is NOT RECORDED here, and every detector that needs it measures nothing rather
        # than zero (observer_world_v1 chunks carry none)
        self.fields = set.intersection(*[set(c.get("fields") or ()) for c in chunks])
        self.has_normals = "contact_normals" in self.fields and all("c_normal" in c for c in chunks)
        self.has_inner_rings = "inner_rings" in self.fields and all("ray_in_ent" in c for c in chunks)
        if self.has_inner_rings:
            self.ray_in_ent = cat("ray_in_ent")
            self.ray_in_dist = cat("ray_in_dist")
            self.ray_in_cell = cat("ray_in_cell")
        else:
            self.ray_in_ent = self.ray_in_dist = self.ray_in_cell = None
        # contacts, with rows re-based onto the concatenated track
        rows, es, fr, fmax, fmean, pos, nrm = [], [], [], [], [], [], []
        base = 0
        for c, n in zip(chunks, n_each):
            rows.append(c["c_row"].astype(np.int64) + base)
            es.append(c["c_e"].astype(np.int64))
            fr.append(c["c_frame"].astype(np.int64))
            fmax.append(c["c_fmax"].astype(np.float64))
            fmean.append(c["c_fmean"].astype(np.float64))
            pos.append(c["c_pos"].astype(np.float64))
            nrm.append(c["c_normal"].astype(np.float64).reshape(-1, 3) if self.has_normals
                       else np.full((len(c["c_row"]), 3), np.nan))
            base += n
        self.c_row = np.concatenate(rows)
        self.c_e = np.concatenate(es).reshape(-1, 2)
        self.c_frame = np.concatenate(fr).reshape(-1, 2)
        self.c_fmax = np.concatenate(fmax)
        self.c_fmean = np.concatenate(fmean)
        self.c_pos = np.concatenate(pos).reshape(-1, 3)
        # each contact row's normal, from its first entity (c_e[:, 0]) to its second; NaN where the
        # chunks recorded none (observer_world_v1)
        self.c_normal = np.concatenate(nrm).reshape(-1, 3)
        # rows are written in record order, so a record's contacts are one slice (contacts_at)
        self._c_sorted = bool(self.c_row.size < 2 or np.all(np.diff(self.c_row) >= 0))
        # THE COMMAND'S HAND JOINT GOAL PER RECORD (observer_attribution_v1, 2026-10-08): the goal each record's
        # command handed the world (observer_world_v2 field group 'orders', radians, in the controller's joint order:
        # kinematics.hand_goal_qpos), NaN where the command gave none; None on a track recorded without the group
        # (observer_world_v1). Read by the evidence join to say whether a command asked for the movement made under it.
        self.has_orders = "orders" in self.fields and all("order_on" in c for c in chunks)
        self.order_hand_goal = None
        if self.has_orders:
            from mimo_bridge.observer_record import orders_in_force
            parts = []
            for c in chunks:
                of = orders_in_force(c)
                g = np.asarray(of["hand_goal"], dtype=np.float64)
                # a chunk with no command rows carries hand_goal of shape [0, J] (or [0]): no goal to place
                if g.ndim >= 2 and g.shape[0]:
                    g = g.reshape(g.shape[0], -1)
                else:
                    g = np.zeros((0, int(np.prod(g.shape[1:])) if g.ndim >= 2 else 0))
                parts.append((np.asarray(of["index"], dtype=np.int64), g))
            width = max([g.shape[1] for _, g in parts] + [0])
            goal = np.full((self.n, width), np.nan)
            base = 0
            for (idx, g), nrec in zip(parts, n_each):
                ok = np.nonzero(idx >= 0)[0]
                if width and ok.size and g.shape[0] and g.shape[1]:
                    goal[base + ok, :g.shape[1]] = g[idx[ok]]
                base += nrec
            self.order_hand_goal = goal
        self._measure()

    # ------------------------------------------------------------------ kinematics
    def _measure(self):
        kin, n = self.kin, self.n
        self.curl = kin.curl_deg(self.qpos)                           # [N, 2, 5]
        self.closure = np.nanmean(self.curl[:, :, :4], axis=-1)       # [N, 2]
        self.wrist = kin.wrist_deg(self.qpos)                          # [N, 2, 3]
        self.swivel = kin.thumb_swivel_deg(self.qpos)                  # [N, 2]
        self.eye_angles = kin.eye_deg(self.qpos)                       # [N, 2, 2]
        keys = ("tips", "palm", "palm_normal", "forearm", "head_c", "head_face", "head_up", "mouth",
                "head_R", "eye_pos", "eye_fwd", "roll_deg", "sit_deg", "trunk_pos", "belly",
                "trunk_lat", "trunk_R", "shoulder_z", "com", "root_pos", "xpos", "xquat")
        acc: Dict[str, list] = {k: [] for k in keys}
        for i in range(n):
            kin.place(self.qpos[i], self.mocap[i] if self.mocap.size else None,
                      self.object_pose[i] if self.object_pose.size else None)
            g = kin.measure()
            for k in keys:
                acc[k].append(g.get(k, np.nan))
        for k in keys:
            setattr(self, k, np.asarray(acc[k], dtype=np.float64))
        K = self.object_pose.shape[1] if self.object_pose.ndim == 3 else 0
        # Use the same reconstructed snapshot as the palms. The recorder's
        # object_pose came from live derived xpos/xquat, which can precede the
        # final integrated qpos. Kinematics.place restores free roots from qpos,
        # mocap roots from saved mocap, and mounted roots from object_pose.
        # Header roots are top-level model bodies (observer_record._build_header).
        roots = np.asarray([int(o["root"]) for o in self.ent.objects], dtype=np.int64)
        if len(roots) != K or (K and (np.any(roots < 0) or np.any(roots >= self.xpos.shape[1]))):
            raise ValueError("recorded object roots do not match reconstructed kinematics")
        self.obj_pos = self.xpos[:, roots, :].copy() if K else np.zeros((n, 0, 3))
        self.obj_quat = self.xquat[:, roots, :].copy() if K else np.zeros((n, 0, 4))
        # time between records: a new world episode or a missing step is not a time step
        self.dt = np.full(n, np.nan)
        if n > 1:
            d = np.diff(self.sim_time)
            ok = (np.diff(self.episode) == 0) & (np.diff(self.step) == 1) & (d > 0)
            self.dt[1:] = np.where(ok, d, np.nan)
        self.continuous = np.isfinite(self.dt)                         # record i follows i-1
        self.palm_speed = self._speed(self.palm)                       # [N, 2] m/s
        self.com_speed = self._speed(self.com[:, None, :])[:, 0]
        self.obj_speed = self._speed(self.obj_pos) if K else np.zeros((n, 0))
        self.obj_vel = self._vel(self.obj_pos) if K else np.zeros((n, 0, 3))
        self.palm_vel = self._vel(self.palm)
        self.obj_ang_speed = np.full((n, K), np.nan)
        self.head_ang_speed = np.full(n, np.nan)          # degrees per second
        self.trunk_ang_speed = np.full(n, np.nan)
        for i in range(1, n):
            if not self.continuous[i]:
                continue
            for k in range(K):
                self.obj_ang_speed[i, k] = math.radians(_quat_angle_deg(self.obj_quat[i - 1, k],
                                                                        self.obj_quat[i, k])) / self.dt[i]
            if np.isfinite(self.head_R[i]).all() and np.isfinite(self.head_R[i - 1]).all():
                self.head_ang_speed[i] = _rot_angle_deg(self.head_R[i - 1], self.head_R[i]) / self.dt[i]
            if np.isfinite(self.trunk_R[i]).all() and np.isfinite(self.trunk_R[i - 1]).all():
                self.trunk_ang_speed[i] = _rot_angle_deg(self.trunk_R[i - 1], self.trunk_R[i]) / self.dt[i]
        # elbows and knees, degrees (0 = straight; MIMo flexes them negative / positive)
        self.limb = {}
        for j in self.header["joints"]:
            nm = j["name"].split(":")[-1]
            if nm in ("right_elbow", "left_elbow", "right_knee", "left_knee"):
                self.limb[nm] = np.degrees(self.qpos[:, int(j["qpos"])])

    def _vel(self, x: np.ndarray) -> np.ndarray:
        v = np.full(x.shape, np.nan)
        if self.n > 1:
            v[1:] = (x[1:] - x[:-1]) / self.dt[1:].reshape((-1,) + (1,) * (x.ndim - 1))
        return v

    def _speed(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self._vel(x), axis=-1)

    def object_spin(self, k: int) -> np.ndarray:
        """[N] rad/s: a free object's own spin; for the mobile, the spin of its hanging part (the
        first body below the ceiling mount)."""
        o = self.ent.objects[k]
        if o["free"]:
            return self.obj_ang_speed[:, k]
        bodies = self.header["entities"][int(o["entity"])].get("bodies") or []
        if len(bodies) < 2:
            return np.full(self.n, np.nan)
        q = self.xquat[:, bodies[1]]
        out = np.full(self.n, np.nan)
        for i in range(1, self.n):
            if self.continuous[i]:
                out[i] = math.radians(_quat_angle_deg(q[i - 1], q[i])) / self.dt[i]
        return out

    def object_body_speed(self, k: int) -> np.ndarray:
        """[N] the fastest of an object's bodies (all the mobile's parts), m/s."""
        bodies = self.header["entities"][int(self.ent.objects[k]["entity"])].get("bodies") or []
        if not bodies:
            return np.full(self.n, np.nan)
        v = self._speed(self.xpos[:, bodies, :])
        v = np.where(np.isfinite(v), v, -np.inf).max(axis=1)
        return np.where(np.isfinite(v), v, np.nan)

    # ------------------------------------------------------------------ contacts
    def contacts_at(self, i: int) -> np.ndarray:
        """Indices into the contact table for record i (a slice when the table is in record
        order, as the recorder writes it; a scan otherwise)."""
        if self._c_sorted:
            a = int(np.searchsorted(self.c_row, i, "left"))
            b = int(np.searchsorted(self.c_row, i, "right"))
            return np.arange(a, b)
        return np.nonzero(self.c_row == i)[0]

    def pair_series(self, mask_a: np.ndarray, mask_b: np.ndarray) -> Dict[str, np.ndarray]:
        """Per record: whether any contact joins an entity in mask_a with one in mask_b, the
        summed peak force, the force-weighted contact point, and (v2 chunks) `normal`: the
        force-weighted sum of the contact normals turned to point from the mask_a part to the mask_b
        part - the direction the mask_a side pushes the mask_b side - as a unit vector (NaN where
        the pair did not touch or the chunks recorded no normals)."""
        a = mask_a[self.c_e[:, 0]] & mask_b[self.c_e[:, 1]]
        b = mask_b[self.c_e[:, 0]] & mask_a[self.c_e[:, 1]]
        sel = np.nonzero(a | b)[0]
        on = np.zeros(self.n, dtype=bool)
        force = np.zeros(self.n)
        pos = np.full((self.n, 3), np.nan)
        wsum = np.zeros(self.n)
        psum = np.zeros((self.n, 3))
        nsum = np.zeros((self.n, 3))
        normals = getattr(self, "c_normal", None)
        for j in sel:
            r = self.c_row[j]
            on[r] = True
            force[r] += self.c_fmax[j]
            w = max(self.c_fmean[j], 1e-6)
            wsum[r] += w
            psum[r] += w * self.c_pos[j]
            if normals is not None:
                nv = normals[j] if a[j] else -normals[j]      # recorded from c_e[:, 0] to c_e[:, 1]
                if np.isfinite(nv).all():
                    nsum[r] += w * nv
        ok = wsum > 0
        pos[ok] = psum[ok] / wsum[ok, None]
        ln = np.linalg.norm(nsum, axis=1)
        normal = np.full((self.n, 3), np.nan)
        good = ln > 1e-12
        normal[good] = nsum[good] / ln[good, None]
        return dict(on=on, force=force, pos=pos, normal=normal)

    def entity_mask(self, codes) -> np.ndarray:
        m = np.zeros(self.ent.n, dtype=bool)
        for c in np.atleast_1d(codes):
            if 0 <= int(c) < self.ent.n:
                m[int(c)] = True
        return m

    def object_mask(self, k: int) -> np.ndarray:
        return self.entity_mask([self.ent.objects[k]["entity"]])

    # ------------------------------------------------------------------ gaze
    def fixated_target(self, eye: int = 0) -> np.ndarray:
        """[N] the target the eye's CENTRE ray lands on (-1 none), with the painted panel's motif
        cell folded in as a separate target id (entity * 16 + cell + 1 + 10**6) for art hits.
        The fixation primitive reads foveal_target, which falls back on the whole bundle where
        this ray lands on a background surface or on nothing."""
        return self.bundle_targets(eye)[:, 0]

    def bundle_targets(self, eye: int = 0) -> np.ndarray:
        """[N, R] the target of every foveal ray of one eye (the centre ray, then the ring at the
        fovea's half-angle; observer_record._gaze): an entity code, or a painted panel's motif
        cell as its own target id (see fixated_target), -1 for a ray that hit nothing."""
        ent = self.ray_ent[:, eye, :].astype(np.int64)
        cell = self.ray_cell[:, eye, :].astype(np.int64)
        art = (ent >= 0) & (self.ent.kind[np.maximum(ent, 0)] == KIND_ART) & (cell >= 0)
        return np.where(art, ent * 16 + cell + 1 + 10 ** 6, ent)

    def bundle_majority(self, eye: int = 0):
        """Per step, the target a MAJORITY of the eye's foveal rays land on (5 of 9 on H84), else
        -1: a painted panel's motif cell when most rays fall in that one cell, else the entity
        most rays hit (a panel counts as one entity when its cells split the rays). Returns
        (target [N], rays on it [N], rays in the bundle R)."""
        fine = self.bundle_targets(eye)
        coarse = self.ray_ent[:, eye, :].astype(np.int64)
        R = fine.shape[1]
        need = R // 2 + 1
        tgt = np.full(self.n, -1, dtype=np.int64)
        cnt = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            for row in (fine[i], coarse[i]):
                hit = row[row >= 0]
                if not hit.size:
                    continue
                vals, counts = np.unique(hit, return_counts=True)
                k = int(np.argmax(counts))
                if counts[k] >= need:
                    tgt[i], cnt[i] = int(vals[k]), int(counts[k])
                    break
        return tgt, cnt, R

    def foveal_target(self, eye: int = 0):
        """Per step, the target the eye's foveal bundle is ON (detector v7): the CENTRE ray's target
        when it is a thing - a body part, an object, the caregiver or a painted motif cell; where
        the centre ray lands on a background surface (floor, wall, ceiling, fixed furniture) or on
        nothing, the target a MAJORITY of the bundle's rays land on (bundle_majority: a large thing
        such as the adult, seen with the centre ray through a gap); otherwise the centre ray's own
        target (the surface, or -1). Returns (target [N], rays of the bundle on it [N], rays R).

        Why the centre ray decides and not the majority (detector v6's rule): the ring of rays sits
        at the fovea's half-angle, 12 degrees out, so a 6 cm toy fills a majority only within about
        15 cm of the eye. Measured 2026-09-24 on 35,800 production body-steps (segments 38 and 39):
        of the eye-steps whose centre ray was on an object, 12.9% had 5 or more rays on it; ball
        fixations fell from 161 to 31 and the scripted look at a box 32 cm away was lost. The
        majority did not make the holds stiller either: two-record holds on body parts, objects,
        the adult and panels turned the eye 5 degrees or more in one step in 155 of 455 majority
        holds (34%) and 206 of 620 centre-ray holds (33%)."""
        fine = self.bundle_targets(eye)
        coarse = self.ray_ent[:, eye, :].astype(np.int64)
        R = fine.shape[1]
        ent0 = coarse[:, 0]
        thing = (ent0 >= 0) & ~np.isin(self.ent.kind[np.maximum(ent0, 0)], BACKGROUND_KINDS)
        majority, _, _ = self.bundle_majority(eye)
        tgt = np.where(thing, fine[:, 0], np.where(majority >= 0, majority, fine[:, 0]))
        # rays on the chosen target: a motif cell counts the rays in that cell, an entity every ray on it
        on = np.where(tgt[:, None] >= 10 ** 6, fine == tgt[:, None], coarse == tgt[:, None])
        cnt = np.where(tgt >= 0, on.sum(axis=1), 0).astype(np.int64)
        return tgt.astype(np.int64), cnt, R

    def inner_rays_on(self, target: int, eye: int = 0) -> Optional[np.ndarray]:
        """[N] how many of the eye's 17 INNER rays - the centre ray and the rings at 2 and 6 degrees
        (observer_world_v2) - land on one target (a motif cell id counts rays in that cell), or None
        on a track recorded without the inner rings. How much of the fovea's middle a target fills: a
        6 cm toy at 30 cm takes the 2-degree ring only while the centre ray is within about 3.7
        degrees of its centre."""
        if not getattr(self, "has_inner_rings", False):
            return None
        ent = np.concatenate([self.ray_ent[:, eye, :1], self.ray_in_ent[:, eye, :]], axis=1).astype(np.int64)
        t = int(target)
        if t >= 10 ** 6:
            cell = np.concatenate([self.ray_cell[:, eye, :1], self.ray_in_cell[:, eye, :]], axis=1).astype(np.int64)
            art = (ent >= 0) & (self.ent.kind[np.maximum(ent, 0)] == KIND_ART) & (cell >= 0)
            return (np.where(art, ent * 16 + cell + 1 + 10 ** 6, ent) == t).sum(axis=1)
        return (ent == t).sum(axis=1)

    def rays_on(self, target: int, eye: int = 0) -> np.ndarray:
        """[N] how many of the eye's foveal rays land on one target (a motif cell id counts only
        rays in that cell; an entity code counts every ray on the entity)."""
        t = int(target)
        if t >= 10 ** 6:
            return (self.bundle_targets(eye) == t).sum(axis=1)
        return (self.ray_ent[:, eye, :].astype(np.int64) == t).sum(axis=1)

    @staticmethod
    def target_entity(t: int) -> int:
        t = int(t)
        return t if t < 10 ** 6 else (t - 10 ** 6 - 1) // 16

    def target_point(self, t: int) -> np.ndarray:
        """[N, 3] the world point a gaze on target t aims at: a body part's origin, an object's
        root (the mobile: the mean of its hanging parts), a motif cell's centre on its panel;
        NaN for the floor, walls, ceiling and other static surfaces, and for a whole painted
        panel (72 cm long: a gaze anywhere along it is on it, so its centre is no aim point)."""
        t = int(t)
        out = np.full((self.n, 3), np.nan)
        e = self.target_entity(t)
        if not 0 <= e < self.ent.n:
            return out
        k = int(self.ent.kind[e])
        info = self.header["entities"][e]
        if k == KIND_SELF:
            bodies = info.get("bodies") or []
            if bodies:
                out[:] = self.xpos[:, int(bodies[0])]
            return out
        if k in (KIND_OBJECT, KIND_CAREGIVER):
            kk = self.ent.object_of_entity.get(e)
            if kk is None:
                return out
            o = self.ent.objects[kk]
            bodies = info.get("bodies") or []
            if not o["free"] and o["kind"] == "mobile" and len(bodies) > 1:
                out[:] = np.nanmean(self.xpos[:, [int(b) for b in bodies[1:]]], axis=1)
            else:
                out[:] = self.obj_pos[:, kk]
            return out
        if k == KIND_ART and t >= 10 ** 6:
            p = self._cell_centre(e, (t - 10 ** 6 - 1) % 16)
            if p is not None:
                out[:] = p
        return out

    def _cell_centre(self, e: int, cell: int) -> Optional[np.ndarray]:
        """A painted panel's motif cell centre in the world (observer_record.art_cell inverted).
        Panels are static: their pose does not depend on the recorded state, so it is read once
        from the compiled model."""
        cache = self.__dict__.setdefault("_cells", {})
        if (e, cell) in cache:
            return cache[(e, cell)]
        info = self.header["entities"][e]
        geoms = info.get("geoms") or []
        motifs = info.get("motifs") or []
        n = len(motifs)
        p = None
        if geoms and n and 0 <= cell < n and self.kin is not None:
            g = int(geoms[0])
            kin = self.kin
            kin.mj.mj_kinematics(kin.m, kin.d)
            ctr = np.asarray(kin.d.geom_xpos[g], dtype=np.float64).copy()
            R = np.asarray(kin.d.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            axis = int(info.get("axis", 0))
            hw = float(info["half_width_m"])
            k = cell if info.get("first_at_negative_end", True) else n - 1 - cell
            local = np.zeros(3)
            local[axis] = -hw + 2.0 * hw * (k + 0.5) / n
            p = ctr + R @ local
        cache[(e, cell)] = p
        return p

    def gaze_error_deg(self, t: int, eye: int = 0) -> np.ndarray:
        """[N] degrees between the eye's centre ray and the direction to target t's point
        (target_point; NaN for a surface)."""
        with np.errstate(invalid="ignore"):
            return _angle_between(self.eye_fwd[:, eye], self.target_point(t) - self.eye_pos[:, eye])

    def target_label(self, t: int) -> str:
        t = int(t)
        if t < 0:
            return "nothing"
        if t >= 10 ** 6:
            e, cell = divmod(t - 10 ** 6 - 1, 16)
            motifs = self.header["entities"][e].get("motifs") or []
            name = motifs[cell] if 0 <= cell < len(motifs) else str(cell)
            return "%s (%s)" % (self.ent.label(e), name)
        return self.ent.label(t)

    def angle_from_gaze(self, point: np.ndarray, eye: int = 0) -> np.ndarray:
        """[N] degrees between an eye's viewing axis and the direction to a point series [N, 3]."""
        return _angle_between(self.eye_fwd[:, eye], np.asarray(point) - self.eye_pos[:, eye])

    def seconds(self, i0: int, i1: int) -> float:
        return float(self.sim_time[i1] - self.sim_time[i0])
