# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Replay links: an episode shown again from its record (addendum sections 28 and 32).

A snippet is the recorded steps of one body around an episode (the raw chunks are removed once
digested, so a snippet is cut while they still exist). A replay is that snippet drawn from the
saved world model by MuJoCo's own renderer, from a fixed third-person camera looking at the
infant, into an animated GIF. Toys and the adult are hidden where the record says they were
hidden; the painted panels come from the model's own textures.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

PAD_STEPS = 20


def cut_snippet(chunks: List[dict], step0: int, step1: int, pad: int = PAD_STEPS) -> Optional[dict]:
    """The records of [step0 - pad, step1 + pad] from decoded chunks (read_chunk), as arrays."""
    keys = ("step", "sim_time", "episode", "qpos", "mocap", "object_pose", "object_flags", "author",
            "caregiver_present", "toy_held")
    rows = {k: [] for k in keys}
    header = None
    hname = ""
    for c in chunks:
        sel = (c["step"] >= step0 - pad) & (c["step"] <= step1 + pad)
        if not sel.any():
            continue
        if header is None:
            header, hname = c["header"], str(c.get("header_name", ""))
        for k in keys:
            if k in c:
                rows[k].append(np.asarray(c[k])[sel])
    if header is None:
        return None
    out = {k: np.concatenate(v) for k, v in rows.items() if v}
    order = np.argsort(out["step"], kind="stable")
    out = {k: v[order] for k, v in out.items()}
    out["header_name"] = np.asarray(hname)
    return dict(arrays=out, header=header)


def save_snippet(path: str, snip: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = dict(snip["arrays"])
    arrays["header_json"] = np.asarray(json.dumps(snip["header"]))
    tmp = path[:-4] + ".part.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_snippet(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files if k != "header_json"}
        header = json.loads(str(z["header_json"]))
    return dict(arrays=arrays, header=header)


def render(snip: dict, world_dir: str, out_gif: str, size=(320, 240), every: int = 2,
           focus_step: Optional[int] = None) -> dict:
    """Draw a snippet to an animated GIF. Returns {'ok': bool, 'frames': n, 'error': ...}."""
    try:
        import mujoco
        from PIL import Image
        from mimo_bridge.observer_record import load_world_model
        h = snip["header"]
        a = snip["arrays"]
        m = load_world_model(os.path.join(world_dir, h["model_file"]))
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, height=int(size[1]), width=int(size[0]))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 0.9, 135.0, -40.0
        objs = h["objects"]
        geoms = [np.asarray(o.get("geoms") or [], dtype=np.int64) for o in objs]
        alpha0 = np.asarray(m.geom_rgba[:, 3]).copy()
        mounts = [(k, int(o["root"])) for k, o in enumerate(objs) if not o["free"] and int(o["mocap"]) < 0]
        frames = []
        root = int(h.get("root_body", 1))
        for i in range(0, len(a["step"]), max(1, int(every))):
            d.qpos[:] = a["qpos"][i]
            if m.nmocap and "mocap" in a and a["mocap"].size:
                d.mocap_pos[:] = a["mocap"][i][:, :3]
                d.mocap_quat[:] = a["mocap"][i][:, 3:7]
            if "object_pose" in a and a["object_pose"].size:
                for k, rb in mounts:
                    m.body_pos[rb] = a["object_pose"][i][k, :3]
                    m.body_quat[rb] = a["object_pose"][i][k, 3:7]
            if "object_flags" in a and a["object_flags"].size:
                for k, gs in enumerate(geoms):
                    if len(gs):
                        m.geom_rgba[gs, 3] = alpha0[gs] if (int(a["object_flags"][i][k]) & 1) else 0.0
            mujoco.mj_kinematics(m, d)
            cam.lookat[:] = d.xpos[root]
            r.update_scene(d, camera=cam)
            img = r.render()
            if focus_step is not None and int(a["step"][i]) >= focus_step:
                img = img.copy()
                img[:4, :, :] = (255, 200, 0)     # a bar on the frames from the act onward
            frames.append(Image.fromarray(img))
        r.close()
        if not frames:
            return dict(ok=False, frames=0, error="no frames")
        os.makedirs(os.path.dirname(out_gif), exist_ok=True)
        frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=int(50 * every),
                       loop=0, optimize=True)
        return dict(ok=True, frames=len(frames))
    except Exception as ex:
        return dict(ok=False, frames=0, error=repr(ex))
