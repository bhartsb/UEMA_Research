# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Write MuJoCo's abstract visualization of the WI-14 fixture states as JSON, the compiled textures as
PNG, and the classic renderer's own measurements, for the Swift eye renderer's tests (WI-29, REQ-B-6).

INPUT. The shipped scene bundle apple/UEMAKit/Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml
(WI-15) and, per state folder beside this file (Fixtures/eyes/step_*/), WI-14's cameras.json: qpos (146
numbers), the 8 mocap poses, each object's geoms with the alpha and contype/conaffinity the export wrote,
the mounted objects' recorded root poses, and the five cameras' recorded world poses.

OPERATION. (1) mujoco.MjModel.from_xml_path(scene.xml). (2) Per state: data.qpos, qvel 0, time, mocap
from cameras.json; model.geom_rgba alpha and geom_contype/conaffinity per object exactly as recorded;
model.body_pos/body_quat of a mounted object's root set to its recorded pose where the export moved it;
mujoco.mj_forward; then data.cam_xpos / cam_xmat of the five cameras compared with the recorded poses
(the difference is written; 1e-6 m is the position quantum of the recording). (3) Per camera:
mujoco.mjv_updateScene(model, data, MjvOption(), None, fixed camera, mjCAT_ALL, scene) with the same
defaults EyeRenderer uses (env.py:603-638), then every mjvGeom (type, dataid, objtype, objid, category,
matid, size, pos, mat, rgba, emission, specular, shininess, reflectance), every mjvLight and the two
mjvGLCamera records are copied into JSON. The geom list is written once (from eye_left); the other
cameras list only the geoms whose pos differs (planes of size 0 are recentred on the camera:
engine_vis_visualize.c mjv_addGeoms). (4) Once: every compiled texture (model.tex_data at tex_adr, rows
tex_height x tex_width x tex_nchannel) is written as a PNG under textures/ with textures.json (type,
size, channels, SHA-256 of the raw bytes, the materials' texture ids per role, texrepeat, texuniform,
rgba, emission, specular, shininess, reflectance; the lights; the visual constants: headlight, map
znear/zfar/haze, rgba haze, quality numslices/numstacks/numquads/shadowsize, stat.extent). (5) The
classic renderer (mjr_render through one mujoco.GLContext, as env.py:590-638) renders each camera at
the state; its image is compared with the WI-14 PNG (mean absolute difference per channel; 0 expected
when the bundle world equals the fixture world); then, for eye_left and ring_left of each state, one
rendering feature at a time is turned off (textures: MjvOption.flags[mjVIS_TEXTURE] 0; shadows:
scene.flags[mjRND_SHADOW] 0; reflections: mjRND_REFLECTION 0; skybox: mjRND_SKYBOX 0; haze: mjRND_HAZE 0;
headlight: model.vis.headlight.active 0) and the mean absolute difference against the full render is
recorded: the size of each feature's contribution, the error budget a renderer without it pays.

OUTPUT. Fixtures/eyes/<state>/abstract_scene.json; Fixtures/eyes/textures/<index>_<name>.png and
textures.json. Printed: one line per state and camera with the classic renderer's difference against
the fixture PNG, and one line per ablation. Exit 0 when every classic-renderer difference against the
fixture PNG is 0.0 in every channel; exit 1 otherwise (the files are still written, with the numbers).

Run from D:/UEMA_Research/Source/Python with a Python that has mujoco 3.3.0, numpy and imageio, after the
free-memory guard (4 GB of available memory requested, the figure WI-29's acceptance command passes):
    python -m mimo_bridge.tools.free_memory_guard --need-gb 4 && python -I <this file>
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
BUNDLE_DIR = os.path.join(REPO_ROOT, "apple", "UEMAKit", "Sources", "UEMAScene", "Resources", "crib_age24_trunk_v1")
SCENE_XML = os.path.join(BUNDLE_DIR, "scene.xml")
FORMAT_SCENE = "uema.eye_fixture.abstract_scene.v1"
FORMAT_TEXTURES = "uema.eye_fixture.textures.v1"
CAMERA_ORDER = ["eye_left", "eye_right", "fovea_left", "fovea_right", "ring_left"]
GEOM_TYPE_NAMES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box",
                   7: "mesh", 8: "sdf"}
TEX_TYPE_NAMES = {0: "2d", 1: "cube", 2: "skybox"}


def f(x) -> Any:
    """numpy scalar or array -> JSON floats (lists)."""
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    return [float(v) for v in a.reshape(-1)]


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(path: str, doc: Dict[str, Any]) -> None:
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")


def set_state(mujoco, m, d, cam_doc: Dict[str, Any]) -> Dict[str, Any]:
    """cameras.json 'state' -> qpos, qvel, time, mocap, object alpha/contype, mounted roots; mj_forward.
    Returns the camera pose check against the recorded poses."""
    st = cam_doc["state"]
    qpos = np.asarray(st["qpos"], dtype=np.float64)
    if qpos.shape != (int(m.nq),):
        raise RuntimeError("qpos %s against nq %d" % (qpos.shape, m.nq))
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    d.time = float(st["sim_time_s"])
    d.mocap_pos[:] = np.asarray(st["mocap_pos_m"], dtype=np.float64)
    d.mocap_quat[:] = np.asarray(st["mocap_quat_wxyz"], dtype=np.float64)
    for obj in st["object_flags"]:
        for g, alpha, (ct, ca) in zip(obj["geoms"], obj["alpha_written"], obj["contype_conaffinity_written"]):
            m.geom_rgba[int(g), 3] = float(alpha)
            m.geom_contype[int(g)] = int(ct)
            m.geom_conaffinity[int(g)] = int(ca)
    for mo in st["mounted_objects"]:
        if mo["root_moved"]:
            root = int(mo["root_body"])
            pose = np.asarray(mo["recorded_pose"], dtype=np.float64)
            m.body_pos[root] = pose[:3]
            m.body_quat[root] = pose[3:]
    mujoco.mj_forward(m, d)
    worst_pos, worst_mat = 0.0, 0.0
    rows = []
    for c in cam_doc["cameras"]:
        cid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA.value, c["name"]))
        dp = float(np.max(np.abs(np.asarray(d.cam_xpos[cid]) - np.asarray(c["pos_m"]))))
        dm = float(np.max(np.abs(np.asarray(d.cam_xmat[cid]).reshape(-1) - np.asarray(c["xmat_row_major"]))))
        worst_pos, worst_mat = max(worst_pos, dp), max(worst_mat, dm)
        rows.append(dict(name=c["name"], id=cid, recorded_id=int(c["id"]), pos_difference_m=dp, xmat_difference=dm))
    return dict(cameras=rows, max_pos_difference_m=worst_pos, max_xmat_difference=worst_mat)


def geom_row(g) -> Dict[str, Any]:
    return dict(type=int(g.type), type_name=GEOM_TYPE_NAMES.get(int(g.type), str(int(g.type))), dataid=int(g.dataid),
                objtype=int(g.objtype), objid=int(g.objid), category=int(g.category), matid=int(g.matid),
                texcoord=int(g.texcoord), segid=int(g.segid), size=f(g.size), pos=f(g.pos), mat=f(g.mat),
                rgba=f(g.rgba), emission=float(g.emission), specular=float(g.specular),
                shininess=float(g.shininess), reflectance=float(g.reflectance))


def light_row(L) -> Dict[str, Any]:
    return dict(pos=f(L.pos), dir=f(L.dir), attenuation=f(L.attenuation), cutoff=float(L.cutoff),
                exponent=float(L.exponent), ambient=f(L.ambient), diffuse=f(L.diffuse), specular=f(L.specular),
                headlight=int(L.headlight), directional=int(L.directional), castshadow=int(L.castshadow),
                bulbradius=float(L.bulbradius))


def glcamera_row(c) -> Dict[str, Any]:
    return dict(pos=f(c.pos), forward=f(c.forward), up=f(c.up), frustum_center=float(c.frustum_center),
                frustum_width=float(c.frustum_width), frustum_bottom=float(c.frustum_bottom),
                frustum_top=float(c.frustum_top), frustum_near=float(c.frustum_near),
                frustum_far=float(c.frustum_far), orthographic=int(c.orthographic))


def update_scene(mujoco, m, d, scn, opt, cam, name: str) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED.value
    cam.fixedcamid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA.value, name))
    mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scn)


def abstract_for_camera(mujoco, m, d, scn, opt, cam, name: str) -> Dict[str, Any]:
    update_scene(mujoco, m, d, scn, opt, cam, name)
    return dict(ngeom=int(scn.ngeom), geoms=[geom_row(scn.geoms[i]) for i in range(int(scn.ngeom))],
                nlight=int(scn.nlight), lights=[light_row(scn.lights[i]) for i in range(int(scn.nlight))],
                gl_cameras=[glcamera_row(scn.camera[0]), glcamera_row(scn.camera[1])],
                flags=[int(x) for x in np.asarray(scn.flags).reshape(-1)])


def model_doc(mujoco, m) -> Dict[str, Any]:
    """The model-level rendering constants and the material, light and camera tables."""
    # mujoco 3.3.0 exposes no mjNTEXROLE constant; the role count is the second extent of mat_texid
    mat_texid = np.asarray(m.mat_texid).reshape(int(m.nmat), -1)
    nrole = int(mat_texid.shape[1])
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB.value) if hasattr(mujoco, "mjtTextureRole") else 1
    materials = []
    for i in range(int(m.nmat)):
        materials.append(dict(id=i, name=str(m.material(i).name), texid_per_role=[int(x) for x in mat_texid[i]],
                              texid_rgb=int(mat_texid[i, rgb_role]),
                              texuniform=int(m.mat_texuniform[i]), texrepeat=f(m.mat_texrepeat[i]),
                              rgba=f(m.mat_rgba[i]), emission=float(m.mat_emission[i]),
                              specular=float(m.mat_specular[i]), shininess=float(m.mat_shininess[i]),
                              reflectance=float(m.mat_reflectance[i])))
    lights = []
    for i in range(int(m.nlight)):
        lights.append(dict(id=i, name=str(m.light(i).name), mode=int(m.light_mode[i]), bodyid=int(m.light_bodyid[i]),
                           targetbodyid=int(m.light_targetbodyid[i]), directional=int(m.light_directional[i]),
                           castshadow=int(m.light_castshadow[i]), active=int(m.light_active[i]),
                           pos=f(m.light_pos[i]), dir=f(m.light_dir[i]), attenuation=f(m.light_attenuation[i]),
                           cutoff=float(m.light_cutoff[i]), exponent=float(m.light_exponent[i]),
                           ambient=f(m.light_ambient[i]), diffuse=f(m.light_diffuse[i]),
                           specular=f(m.light_specular[i]), bulbradius=float(m.light_bulbradius[i])))
    cameras = []
    for i in range(int(m.ncam)):
        cameras.append(dict(id=i, name=str(m.camera(i).name), fovy_deg=float(m.cam_fovy[i]), ipd_m=float(m.cam_ipd[i]),
                            bodyid=int(m.cam_bodyid[i]), mode=int(m.cam_mode[i])))
    skybox = [i for i in range(int(m.ntex)) if int(m.tex_type[i]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX.value)]
    extent = float(m.stat.extent)
    return dict(
        nq=int(m.nq), nbody=int(m.nbody), ngeom=int(m.ngeom), ncam=int(m.ncam), nlight=int(m.nlight),
        nmat=int(m.nmat), ntex=int(m.ntex), ntexdata=int(m.ntexdata), nmocap=int(m.nmocap), ntexrole=nrole,
        stat=dict(extent_m=extent, center_m=f(m.stat.center), meansize_m=float(m.stat.meansize)),
        vis=dict(map=dict(znear=float(m.vis.map.znear), zfar=float(m.vis.map.zfar), haze=float(m.vis.map.haze),
                          alpha=float(m.vis.map.alpha), fogstart=float(m.vis.map.fogstart),
                          fogend=float(m.vis.map.fogend)),
                 znear_m=float(m.vis.map.znear) * extent, zfar_m=float(m.vis.map.zfar) * extent,
                 headlight=dict(ambient=f(m.vis.headlight.ambient), diffuse=f(m.vis.headlight.diffuse),
                                specular=f(m.vis.headlight.specular), active=int(m.vis.headlight.active)),
                 rgba=dict(haze=f(m.vis.rgba.haze), fog=f(m.vis.rgba.fog)),
                 quality=dict(shadowsize=int(m.vis.quality.shadowsize), offsamples=int(m.vis.quality.offsamples),
                              numslices=int(m.vis.quality.numslices), numstacks=int(m.vis.quality.numstacks),
                              numquads=int(m.vis.quality.numquads)),
                 global_=dict(offwidth=int(m.vis.global_.offwidth), offheight=int(m.vis.global_.offheight),
                              fovy=float(m.vis.global_.fovy), ipd=float(m.vis.global_.ipd))),
        materials=materials, lights=lights, cameras=cameras,
        skybox_texture_id=(skybox[0] if skybox else -1),
        mjMAXPLANEGRID=int(mujoco.mjMAXPLANEGRID), mjMAXLIGHT=int(mujoco.mjMAXLIGHT))


def write_textures(mujoco, m, out_dir: str, common: Dict[str, Any]) -> Dict[str, Any]:
    import imageio.v3 as iio
    os.makedirs(out_dir, exist_ok=True)
    data = np.asarray(m.tex_data, dtype=np.uint8)
    rows = []
    for i in range(int(m.ntex)):
        w, h, nch, adr = int(m.tex_width[i]), int(m.tex_height[i]), int(m.tex_nchannel[i]), int(m.tex_adr[i])
        raw = data[adr: adr + w * h * nch]
        name = str(m.texture(i).name) or "unnamed"
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
        fname = "%02d_%s.png" % (i, safe)
        img = raw.reshape(h, w, nch)
        iio.imwrite(os.path.join(out_dir, fname), np.ascontiguousarray(img))
        rows.append(dict(id=i, name=name, type=int(m.tex_type[i]), type_name=TEX_TYPE_NAMES.get(int(m.tex_type[i])),
                         width=w, height=h, nchannel=nch, adr=adr, bytes=int(raw.size), file=fname,
                         sha256_raw=sha256_hex(raw.tobytes()),
                         layout="rows as stored in mjModel.tex_data (row 0 first); cube and skybox textures hold "
                                "six square faces stacked vertically in the order +X, -X, +Y, -Y, +Z, -Z "
                                "(render_context.c mjr_uploadTexture: face i at tex_data + adr + i*3*w*w)"))
    doc = dict(common, format=FORMAT_TEXTURES, textures=rows, model=model_doc(mujoco, m))
    write_json(os.path.join(out_dir, "textures.json"), doc)
    return doc


class ClassicRenderer:
    """env.py:603-638 EyeRenderer on one shared GL context: mjv_updateScene, mjr_render, mjr_readPixels, flipud."""

    def __init__(self, mujoco, m, size: int, gl):
        self.mujoco, self.m, self.size, self.gl = mujoco, m, size, gl
        gl.make_current()
        self.scene = mujoco.MjvScene(m, maxgeom=10000)
        self.option = mujoco.MjvOption()
        self.rect = mujoco.MjrRect(0, 0, size, size)
        self.ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.ctx)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED.value
        self.buf = np.empty((size, size, 3), dtype=np.uint8)

    def render(self, d, name: str, scene_flags_off=(), vis_flags_off=()) -> np.ndarray:
        mj = self.mujoco
        self.gl.make_current()
        self.cam.fixedcamid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_CAMERA.value, name)
        saved = {}
        for fl in vis_flags_off:
            saved[fl] = int(self.option.flags[fl])
            self.option.flags[fl] = 0
        mj.mjv_updateScene(self.m, d, self.option, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        for fl, v in saved.items():
            self.option.flags[fl] = v
        saved_s = {}
        for fl in scene_flags_off:
            saved_s[fl] = int(self.scene.flags[fl])
            self.scene.flags[fl] = 0
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self.ctx)
        mj.mjr_render(self.rect, self.scene, self.ctx)
        mj.mjr_readPixels(self.buf, None, self.rect, self.ctx)
        for fl, v in saved_s.items():
            self.scene.flags[fl] = v
        return np.flipud(self.buf).copy()

    def close(self):
        try:
            self.ctx.free()
        except Exception:
            pass


def mad(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if a.shape != b.shape:
        return dict(shape_a=list(a.shape), shape_b=list(b.shape), comparable=False)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(comparable=True, mean_abs_per_channel=[float(x) for x in diff.reshape(-1, 3).mean(axis=0)],
                max_abs_per_channel=[int(x) for x in diff.reshape(-1, 3).max(axis=0)],
                differing_pixel_share=float((diff.max(axis=2) > 0).mean()), pixels=int(a.shape[0] * a.shape[1]))


def main() -> int:
    import mujoco
    import imageio.v3 as iio
    written_at = _dt.datetime.now(_dt.timezone.utc).astimezone().strftime("%Y-%m-%d %I:%M %p %Z")
    with open(os.path.abspath(__file__), "rb") as fh:
        generator_sha = sha256_hex(fh.read())
    common = dict(written_by="apple/UEMAKit/Tests/UEMASensesTests/Fixtures/eyes/export_abstract_scene.py",
                  generator_sha256=generator_sha, written_at=written_at,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__),
                  scene_xml=os.path.relpath(SCENE_XML, REPO_ROOT).replace("\\", "/"))
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    build_alpha = np.array(m.geom_rgba[:, 3])
    build_ct, build_ca = np.array(m.geom_contype), np.array(m.geom_conaffinity)
    build_body_pos, build_body_quat = np.array(m.body_pos), np.array(m.body_quat)
    print("model: nq %d nbody %d ngeom %d ncam %d nlight %d nmat %d ntex %d extent %.4f m"
          % (m.nq, m.nbody, m.ngeom, m.ncam, m.nlight, m.nmat, m.ntex, m.stat.extent))
    tex_doc = write_textures(mujoco, m, os.path.join(HERE, "textures"), common)
    print("textures: %d written under textures/" % len(tex_doc["textures"]))

    gl = mujoco.GLContext(480, 480)
    renderers = {256: ClassicRenderer(mujoco, m, 256, gl), 288: ClassicRenderer(mujoco, m, 288, gl)}
    scn = mujoco.MjvScene(m, maxgeom=10000)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    all_zero = True
    states = sorted(x for x in os.listdir(HERE) if x.startswith("step_") and os.path.isdir(os.path.join(HERE, x)))
    for st in states:
        folder = os.path.join(HERE, st)
        with open(os.path.join(folder, "cameras.json"), encoding="utf-8") as fh:
            cam_doc = json.load(fh)
        # restore the build's appearance before applying this state's flags
        m.geom_rgba[:, 3] = build_alpha
        m.geom_contype[:] = build_ct
        m.geom_conaffinity[:] = build_ca
        m.body_pos[:] = build_body_pos
        m.body_quat[:] = build_body_quat
        pose_check = set_state(mujoco, m, d, cam_doc)
        per_camera: Dict[str, Any] = {}
        base = None
        for name in CAMERA_ORDER:
            a = abstract_for_camera(mujoco, m, d, scn, opt, cam, name)
            if base is None:
                base = a
                per_camera[name] = dict(gl_cameras=a["gl_cameras"], lights=a["lights"], nlight=a["nlight"],
                                        geom_differences=[], ngeom=a["ngeom"])
            else:
                diffs = []
                if a["ngeom"] != base["ngeom"]:
                    raise RuntimeError("%s %s: ngeom %d against eye_left %d" % (st, name, a["ngeom"], base["ngeom"]))
                for i, (g0, g1) in enumerate(zip(base["geoms"], a["geoms"])):
                    changed = {k: g1[k] for k in g1 if g1[k] != g0[k]}
                    if changed:
                        if set(changed) - {"pos"} or g0["type"] != 0:
                            raise RuntimeError("%s %s geom %d differs in %s (type %s)"
                                               % (st, name, i, sorted(changed), g0["type_name"]))
                        diffs.append(dict(index=i, pos=g1["pos"]))
                per_camera[name] = dict(gl_cameras=a["gl_cameras"], lights=a["lights"], nlight=a["nlight"],
                                        geom_differences=diffs, ngeom=a["ngeom"])
        # classic renderer against the fixture PNG
        checks: Dict[str, Any] = {}
        for c in cam_doc["cameras"]:
            size = int(c["size_px"])
            img = renderers[size].render(d, c["name"])
            ref = np.asarray(iio.imread(os.path.join(folder, c["png"])))
            r = mad(img, ref)
            checks[c["name"]] = dict(classic_vs_fixture_png=r, size_px=size)
            if not r.get("comparable") or max(r["mean_abs_per_channel"]) != 0.0:
                all_zero = False
            print("%s %s: classic renderer against %s: mean abs per channel %s, max %s, differing share %.4f"
                  % (st, c["name"], c["png"], r.get("mean_abs_per_channel"), r.get("max_abs_per_channel"),
                     r.get("differing_pixel_share", -1.0)))
        ablations: Dict[str, Any] = {}
        for name in ("eye_left", "ring_left"):
            full = renderers[256].render(d, name)
            rows = {}
            for label, sflags, vflags in (
                    ("textures_off", (), (mujoco.mjtVisFlag.mjVIS_TEXTURE.value,)),
                    ("shadows_off", (mujoco.mjtRndFlag.mjRND_SHADOW.value,), ()),
                    ("reflections_off", (mujoco.mjtRndFlag.mjRND_REFLECTION.value,), ()),
                    ("skybox_off", (mujoco.mjtRndFlag.mjRND_SKYBOX.value,), ()),
                    ("haze_off", (mujoco.mjtRndFlag.mjRND_HAZE.value,), ()),
                    ("skybox_and_haze_off", (mujoco.mjtRndFlag.mjRND_SKYBOX.value,
                                             mujoco.mjtRndFlag.mjRND_HAZE.value), ()),
                    ("cull_face_off", (mujoco.mjtRndFlag.mjRND_CULL_FACE.value,), ())):
                img = renderers[256].render(d, name, scene_flags_off=sflags, vis_flags_off=vflags)
                rows[label] = mad(img, full)
            active = int(m.vis.headlight.active)
            m.vis.headlight.active = 0
            rows["headlight_off"] = mad(renderers[256].render(d, name), full)
            m.vis.headlight.active = active
            ablations[name] = rows
            for label, r in rows.items():
                print("  %s %s %s: mean abs per channel %s" % (st, name, label, r["mean_abs_per_channel"]))
        doc = dict(common, format=FORMAT_SCENE,
                   state=dict(name=st, step=int(cam_doc["state"]["step"]), source="cameras.json (WI-14)",
                              camera_pose_check=pose_check),
                   model=tex_doc["model"], textures_file="../textures/textures.json",
                   geom_source_camera="eye_left", ngeom=base["ngeom"], geoms=base["geoms"],
                   cameras=per_camera, classic_renderer_checks=checks, feature_ablations=ablations,
                   notes=["geoms: mjv_updateScene's list for eye_left; cameras.<name>.geom_differences lists the "
                          "geoms whose pos differs for that camera (planes of size 0, recentred on the camera); "
                          "every other field is equal across the five cameras (checked here)",
                          "lights: per camera, since light 0 is the headlight at the camera (mjv_makeLights)",
                          "feature_ablations: mean absolute difference per channel between the classic render with "
                          "one feature off and the full classic render, 8-bit steps (0 = identical, 255 = opposite)"])
        write_json(os.path.join(folder, "abstract_scene.json"), doc)
        print("%s: abstract_scene.json written: ngeom %d, nlight %d, camera pose check max pos diff %.2e m, "
              "max xmat diff %.2e" % (st, base["ngeom"], base["nlight"], pose_check["max_pos_difference_m"],
                                      pose_check["max_xmat_difference"]))
    for r in renderers.values():
        r.close()
    gl.free()
    print("PASS: classic renderer reproduces every fixture PNG" if all_zero else
          "FAIL: a classic render differs from its fixture PNG (numbers above and in abstract_scene.json)")
    return 0 if all_zero else 1


if __name__ == "__main__":
    sys.exit(main())
