# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What an observer_world_v2 recorder would have written, rebuilt from a v1 recording: FOR MEASURING.

The recorder's v2 field groups (observer_record.FIELDS) exist only in chunks recorded after
2026-09-25. Before any new recording exists, the definitions that read them (R04 and H02 by contact
normals, fixations by the inner rings) are measured on the kept v1 recordings by rebuilding those
fields from what v1 stored: every record's state (qpos, the moved bodies, each object's root pose and
its visible / collidable flags) and the world's compiled model saved beside the chunks.

  contact normals  the record's state is placed in the saved model and MuJoCo's collision detection
                   runs on it (mj_fwdPosition: geometry only, no dynamics). Each contact row the
                   recorder stored gets the normals of that entity pair's contacts at the record, turned
                   to point from the pair's first entity to its second and weighted by penetration
                   depth (the recorder weights by normal force over the burst's frames, which a stored
                   state cannot give). A pair not touching in the record's own frame (it touched only
                   earlier in the burst) gets no normal (NaN).
  inner rings      cast from the record's eye cameras with the recorder's own code
                   (observer_record.cast_bundle), objects drawn or not by their stored visible flag.
                   The v1 bundle (centre ray and 12-degree ring) is cast the same way and compared
                   with the stored one: `agreement` says how far a rebuilt ray can be trusted.

The digester never calls this: a v1 chunk measures nothing that needs a v2 field (features.Track),
so rebuilt and recorded fields are never mixed in a count. Chunks this module fills are marked
chunk["reconstructed"] = True and carry the groups in chunk["fields"] for the measurement only.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional

import numpy as np

from mimo_bridge.observer_record import (FIELD_CONTACT_NORMALS, FIELD_INNER_RINGS, INNER_RING_RAYS,
                                         INNER_RINGS_DEG, art_cell, cast_bundle, geom_alpha,
                                         load_world_model, ring_directions)


class Reconstructor:
    """One saved world model and the header that names it."""

    def __init__(self, header: dict, world_dir: str):
        import mujoco
        self.mj = mujoco
        self.header = header
        self.m = load_world_model(os.path.join(world_dir, header["model_file"]))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.geom_entity = np.asarray(header["geom_entity"], dtype=np.int32)
        ents = header["entities"]
        self.art = {int(g): ents[int(self.geom_entity[int(g)])] for g in header.get("art_panels", {})}
        self.see_through = {int(b) for b in ([header["head_body"]] if header.get("head_body") is not None else [])
                            + list(header.get("eye_bodies") or [])}
        self.cams = dict(header.get("cameras") or {})
        self.objects = header["objects"]
        self.mounts = [(k, int(o["root"])) for k, o in enumerate(self.objects)
                       if not o["free"] and int(o["mocap"]) < 0]
        self.obj_geoms = [np.asarray(o.get("geoms") or [], dtype=np.int64) for o in self.objects]
        self.alpha0 = np.asarray([geom_alpha(m, g) for g in range(m.ngeom)])
        self.contype0 = np.array(m.geom_contype)
        self.conaff0 = np.array(m.geom_conaffinity)
        self.drawn = self.alpha0 > 0.0
        self.half = math.radians(float(header.get("ring_half_angle_deg", 12.0)))
        self.ring_rays = int(header.get("ring_rays", 8))

    # ------------------------------------------------------------------ one record's state
    def place(self, qpos, mocap=None, object_pose=None, object_flags=None) -> None:
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = qpos
        if mocap is not None and m.nmocap and np.asarray(mocap).size:
            d.mocap_pos[:] = np.asarray(mocap)[:, :3]
            d.mocap_quat[:] = np.asarray(mocap)[:, 3:7]
        if object_pose is not None and np.asarray(object_pose).size:
            op = np.asarray(object_pose)
            for k, root in self.mounts:
                m.body_pos[root] = op[k, :3]
                m.body_quat[root] = op[k, 3:7]
        self.drawn = self.alpha0 > 0.0
        m.geom_contype[:] = self.contype0
        m.geom_conaffinity[:] = self.conaff0
        if object_flags is not None:
            for k, gs in enumerate(self.obj_geoms):
                if not len(gs) or k >= len(object_flags):
                    continue
                flag = int(object_flags[k])
                # the caregiver shows and hides objects by transparency and switches their collision
                # off; the saved model holds whatever it was at the moment it was saved. A visible
                # object keeps its saved per-geom transparency when it was visible then (some geoms
                # are never drawn), else every geom counts as drawn
                if flag & 1:
                    shown = self.alpha0[gs] > 0.0
                    self.drawn[gs] = shown if shown.any() else True
                else:
                    self.drawn[gs] = False
                if flag & 2:
                    off = (self.contype0[gs] == 0) & (self.conaff0[gs] == 0)
                    m.geom_contype[gs[off]] = 1
                    m.geom_conaffinity[gs[off]] = 1
                else:
                    m.geom_contype[gs] = 0
                    m.geom_conaffinity[gs] = 0
        mj.mj_fwdPosition(m, d)

    # ------------------------------------------------------------------ contact normals
    def pair_normals(self) -> Dict[tuple, np.ndarray]:
        """{(first entity, second entity): unit normal from the first to the second} over the placed
        state's contacts, weighted by penetration depth."""
        d = self.d
        acc: Dict[tuple, np.ndarray] = {}
        ge = self.geom_entity
        for i in range(int(d.ncon)):
            c = d.contact[i]
            e1, e2 = int(ge[int(c.geom1)]), int(ge[int(c.geom2)])
            if e1 < 0 or e2 < 0 or e1 == e2:
                continue
            n = np.array(c.frame[:3], dtype=np.float64)
            if e1 > e2:
                n = -n
            key = (min(e1, e2), max(e1, e2))
            w = max(-float(c.dist), 1e-6)
            acc[key] = acc.get(key, np.zeros(3)) + w * n
        out = {}
        for k, v in acc.items():
            ln = float(np.linalg.norm(v))
            if ln > 1e-12:
                out[k] = v / ln
        return out

    # ------------------------------------------------------------------ rays
    def bundles(self) -> dict:
        """Per eye 'L'/'R': the v1 bundle and the inner rings at the placed state, as the recorder
        casts them."""
        m, d = self.m, self.d
        out = {}
        eyes = list(self.header.get("eye_bodies") or [])

        def drawn(g: int) -> bool:
            return bool(self.drawn[g])

        def cell_at(g: int, point: np.ndarray) -> int:
            return art_cell(self.art[g], np.asarray(d.geom_xpos[g]), np.asarray(d.geom_xmat[g]).reshape(3, 3), point)
        for side, cam_name in (("L", "eye_left"), ("R", "eye_right")):
            cid = self.cams.get(cam_name)
            if cid is None:
                continue
            pos = np.array(d.cam_xpos[cid], dtype=np.float64)
            R = np.array(d.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
            fwd, right, up = -R[:, 2], R[:, 0], R[:, 1]
            exclude = eyes[0 if side == "L" else 1] if len(eyes) == 2 else -1
            dirs = [fwd] + ring_directions(fwd, right, up, self.half, self.ring_rays)
            v1 = cast_bundle(m, d, pos, dirs, exclude, self.see_through, drawn, self.geom_entity, self.art, cell_at)
            inner = [v for deg in INNER_RINGS_DEG
                     for v in ring_directions(fwd, right, up, math.radians(deg), INNER_RING_RAYS)]
            v2 = cast_bundle(m, d, pos, inner, exclude, self.see_through, drawn, self.geom_entity, self.art, cell_at)
            out[side] = dict(v1=v1, inner=v2)
        return out

    # ------------------------------------------------------------------ a whole chunk
    def fill(self, chunk: dict, normals: bool = True, rings: bool = True) -> dict:
        """The chunk with rebuilt c_normal and ray_in_* columns (and chunk['agreement']: how often
        the rebuilt v1 bundle equals the stored one, per eye). Changes the dict given."""
        n = len(chunk["step"])
        RI = len(INNER_RINGS_DEG) * INNER_RING_RAYS
        c_row = np.asarray(chunk["c_row"], dtype=np.int64)
        c_e = np.asarray(chunk["c_e"], dtype=np.int64).reshape(-1, 2)
        c_normal = np.full((len(c_row), 3), np.nan)
        in_ent = np.full((n, 2, RI), -1, dtype=np.int16)
        in_dist = np.full((n, 2, RI), -1.0, dtype=np.float32)
        in_cell = np.full((n, 2, RI), -1, dtype=np.int8)
        R1 = 1 + self.ring_rays
        v1_ent = np.full((n, 2, R1), -1, dtype=np.int16)       # the v1 bundle rebuilt, for like-for-like
        v1_cell = np.full((n, 2, R1), -1, dtype=np.int8)
        same = np.zeros(2, dtype=np.int64)
        total = np.zeros(2, dtype=np.int64)
        flags = chunk.get("object_flags")
        for i in range(n):
            self.place(chunk["qpos"][i], chunk["mocap"][i] if np.asarray(chunk["mocap"]).size else None,
                       chunk["object_pose"][i] if np.asarray(chunk["object_pose"]).size else None,
                       None if flags is None or not np.asarray(flags).size else flags[i])
            if normals:
                pn = self.pair_normals()
                for j in np.nonzero(c_row == i)[0]:
                    v = pn.get((int(c_e[j, 0]), int(c_e[j, 1])))
                    if v is not None:
                        c_normal[j] = v
            if rings:
                b = self.bundles()
                for ei, side in enumerate(("L", "R")):
                    if side not in b:
                        continue
                    ent1 = b[side]["v1"][0]
                    stored = np.asarray(chunk["ray_ent"][i, ei])
                    total[ei] += stored.size
                    same[ei] += int(np.sum(stored == ent1))
                    in_ent[i, ei], in_dist[i, ei], in_cell[i, ei] = b[side]["inner"]
                    v1_ent[i, ei], v1_cell[i, ei] = ent1, b[side]["v1"][2]
        fields = set(chunk.get("fields") or ())
        if normals:
            chunk["c_normal"] = c_normal
            fields.add(FIELD_CONTACT_NORMALS)
        if rings:
            chunk["ray_in_ent"], chunk["ray_in_dist"], chunk["ray_in_cell"] = in_ent, in_dist, in_cell
            chunk["ray_rebuilt_ent"], chunk["ray_rebuilt_cell"] = v1_ent, v1_cell
            fields.add(FIELD_INNER_RINGS)
            chunk["agreement"] = dict(rays_same=same.tolist(), rays=total.tolist())
        chunk["fields"] = fields
        chunk["reconstructed"] = True
        return chunk


_CACHE: Dict[str, Reconstructor] = {}


def reconstructor_for(header: dict, world_dir: str) -> Reconstructor:
    """One Reconstructor per saved model (by its hash), the most recent only (each holds a physics
    arena of about 0.4 GB)."""
    key = header.get("model_sha256") or os.path.join(os.path.abspath(world_dir), header["model_file"])
    r = _CACHE.get(key)
    if r is None:
        _CACHE.clear()
        r = _CACHE[key] = Reconstructor(header, world_dir)
    return r
