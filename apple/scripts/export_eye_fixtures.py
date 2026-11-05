# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Export three recorded states of the released run as eye and tactile fixtures for the Swift senses port
(WI-14; REQ-B-6 vision, REQ-B-5 tactile).

INPUT. The released run folder weights/b1_continue_B1K_skip13_492800/run: its args.json (the world's settings:
scene "crib", age 24 months, body_layout "trunk_v1", mobile on, critters on, responsive caregiver, fovea 288 px
over 24 degrees, ring 150 degrees, contact_profile "mechanical_v4", touch_inside_rays "fixed_v1"), and the
developmental observer's world recording of body 0, observer/world/b0/: a header JSON (nq 146, the compiled
model's structure fingerprint, the object list with each object's geoms, the camera ids) and chunk files of 100
records each (mimo_bridge/observer_record.py:445-467 read_chunk). A record holds the full qpos stored as whole
multiples of 1e-6 m / 1e-6 rad / 1e-7 (quaternions) and delta-coded in time (observer_record.py:336-346
encode_series, 417-424 decode_series), the 8 mocap poses, each object's root pose and its visible / collidable
flags, the caregiver's presence and phase, and the simulated time. Three records are chosen as "chunk:row" pairs
(--states, default 0:0,8:74,11:80: the first record of the run with the adult absent, and two with the adult
present during its "presenting" and "engaged" phases).

OPERATION. (1) Build the world with mimo_bridge.env.BridgeEnv from the released args.json through the same
argument mapping the trainer uses (train_mimo.py:1402-1456 env_kwargs_for, transcribed in env_kwargs_from_args;
the caregiver's recording folder is left empty so nothing is written outside the fixture folder). (2) Refuse to
continue unless the built model's structure fingerprint (observer_record.py:299-322 structure_fingerprint)
equals the header's and nq, nv, nbody and nmocap equal the header's: the fixtures must come from the recorded
world. (3) For each chosen record: set data.qpos to the record's qpos, data.qvel to zero (the recording holds no
velocities), data.time to the record's simulated seconds, data.mocap_pos / mocap_quat to the record's mocap
poses; apply the record's object flags the way the caregiver applies them while the run goes (caregiver.py:
304-319: hidden gives geom alpha 0 and contype / conaffinity 0, shown gives alpha 1 where the build left 0 and
the build's alpha elsewhere, collidable restores contype / conaffinity 1 where the build left 0); move a mounted
object's root (mocap -1, no free joint, parent the world body) to the record's root pose when forward kinematics
from the saved model places it elsewhere (the caregiver moves the mobile's mount by editing the model,
observer_record.py:44-46); mujoco.mj_forward. (4) Render the five cameras exactly as the trainer does: wide
eye_left and eye_right through env.render_eyes (EyeRenderer, env.py:603-638: mjv_updateScene, mjr_render on
the classic OpenGL renderer into the offscreen buffer, mjr_readPixels, rows flipped top-down) at eye_size 256
px over 60 degrees; fovea_left and fovea_right through env.render_fovea at 288 px over 24 degrees; ring_left
through env.render_ring at the eye size 256 px over 150 degrees (env.py:1107-1109; args.json's ring_size 128 is
not used: BridgeEnv renders the ring through the wide eye renderer at eye_size). The ring is not patched: the
snap reflex alone reads it (env.py:1494-1502), so N counts the wide and fovea pairs only. (5) Run env.patcher, the
BinocularPatcher of the wide and fovea pairs (vision_adapter.py:136-215; env.py:1306-1311), with the eye
joints' yaw and pitch (env.eye_yaw_pitch), and record the patch count N, the four angle numbers of the first
four rows and digests of the whole output. (6) Compute the tactile patch fields of the same state: MIMo's
TrimeshTouch.get_touch_obs gives the per-point forces, env._contacts the contact list (env.py:2305-2357), and
a fresh TouchPooler (touch_adapter.py:179-346) turns them into [53, 10] fields; the frame pads them with zero
rows to the 256 patch slots of the model's tactile input (`touch` in the code: tokenizer/config.py:51 TOUCH_MAX_IMPULSES;
frame_encoder.py:119-120, 240).

OUTPUT. apple/UEMAKit/Tests/UEMASensesTests/Fixtures/eyes/<state>/ with <state> = step_<model step, 9 digits>:
eye_left.png, eye_right.png (256 x 256), fovea_left.png, fovea_right.png (288 x 288), ring_left.png (256 x 256),
all RGB 8-bit; cameras.json (the state, the five cameras' world poses and fields of view, the renderer, the
model check); patches.json (N, the sources, the gaze, the first four angle rows, digests); tactile_patches.json
(the 10 field names with units, the [53, 10] fields, the [256, 10] padded fields, the patch table, the inputs).
Printed: one line per state with the five image sizes and N, the fingerprint check, the peak memory. Exit 0 only
when every image has its stated size and N equals --expect-n (1,160 = 2 x (256 + 324): 2 eyes x (16 x 16 wide
patches + 18 x 18 fovea patches), vision_adapter.py:173-174).

Run from D:/UEMA_Research/Source/Python after the free-memory guard (the world is one MuJoCo body with
vision; no trainer starts):
    python -m mimo_bridge.tools.free_memory_guard --need-gb 6 && python -I ../../apple/scripts/export_eye_fixtures.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SOURCE_PYTHON = os.path.join(REPO_ROOT, "Source", "Python")
if SOURCE_PYTHON not in sys.path:
    sys.path.insert(0, SOURCE_PYTHON)

FORMAT_CAMERAS = "uema.eye_fixture.cameras.v1"
FORMAT_PATCHES = "uema.eye_fixture.patches.v1"
FORMAT_TACTILE = "uema.eye_fixture.tactile_patches.v1"
RELEASED_RUN = os.path.join(REPO_ROOT, "weights", "b1_continue_B1K_skip13_492800", "run")
DEFAULT_OUT = os.path.join(REPO_ROOT, "apple", "UEMAKit", "Tests", "UEMASensesTests", "Fixtures", "eyes")
DEFAULT_STATES = "0:0,8:74,11:80"
EXPECTED_N = 1160
# the five cameras and the png each is written to; sizes come from the built world (eye_size, fovea_size)
CAMERA_FILES = (("eye_left", "eye_left.png"), ("eye_right", "eye_right.png"),
                ("fovea_left", "fovea_left.png"), ("fovea_right", "fovea_right.png"),
                ("ring_left", "ring_left.png"))
# touch_adapter.py:6-16: the ten numbers of one tactile patch
TACTILE_FIELDS = [
    ("normal_pressure", "Pa", "sum of normal force on the patch / patch contact area"),
    ("shear", "Pa", "|sum of tangential force on the patch| / patch contact area"),
    ("slip_rate", "mm/s", "contact point movement on the skin since the last update / dt (0 here: one update)"),
    ("contact_area", "mm2", "touched skin points x one point's area"),
    ("vibration_5hz", "N^2", "Goertzel power of the patch's normal-force history at 5 Hz"),
    ("vibration_10hz", "N^2", "Goertzel power at 10 Hz"),
    ("vibration_15hz", "N^2", "Goertzel power at 15 Hz"),
    ("vibration_20hz", "N^2", "Goertzel power at 20 Hz"),
    ("stretch_a", "ratio", "1.0 (MIMo has no soft skin)"),
    ("stretch_b", "ratio", "1.0 (MIMo has no soft skin)"),
]
TOUCH_SLOTS = 256   # tokenizer/config.py:51 TOUCH_MAX_IMPULSES
# the observer's object flag bits (header object_flags: 1 visible, 2 collidable)
FLAG_VISIBLE = 1
FLAG_COLLIDABLE = 2


def parse_states(text: str) -> List[Tuple[int, int]]:
    """INPUT: "chunk:row,chunk:row,..." with chunk the index into the body's chunk files sorted by name and row
    the record index inside that chunk (0 to 99). OPERATION: split and convert. OUTPUT: the list of (chunk, row)
    pairs in the given order; a malformed or negative entry raises ValueError."""
    out: List[Tuple[int, int]] = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if ":" not in part:
            raise ValueError("state %r is not chunk:row" % part)
        a, b = part.split(":", 1)
        chunk, row = int(a), int(b)
        if chunk < 0 or row < 0:
            raise ValueError("state %r has a negative index" % part)
        out.append((chunk, row))
    if not out:
        raise ValueError("no states given")
    return out


def pad_to_slots(fields: np.ndarray, slots: int = TOUCH_SLOTS) -> np.ndarray:
    """INPUT: fields [n, 10] float32 with n at most `slots`. OPERATION: append zero rows (the frame's unused patch
    slots, frame_encoder.py:240). OUTPUT: [slots, 10] float32; more than `slots` rows raises ValueError
    (frame_encoder.py:119-120 refuses them the same way)."""
    f = np.asarray(fields, dtype=np.float32)
    if f.ndim != 2 or f.shape[1] != len(TACTILE_FIELDS):
        raise ValueError("fields must be [n, 10], got %s" % (f.shape,))
    if f.shape[0] > slots:
        raise ValueError("%d patches > %d slots" % (f.shape[0], slots))
    out = np.zeros((slots, f.shape[1]), dtype=np.float32)
    out[:f.shape[0]] = f
    return out


def env_kwargs_from_args(a: Dict[str, Any], scene_dir: str, speech_dir: str, tag: str) -> Dict[str, Any]:
    """INPUT: the released args.json as a dict, the scene folder to write the world's XML into, the caregiver speech
    folder on this machine, a scene file tag. OPERATION: the trainer's mapping from arguments to BridgeEnv keyword
    arguments, transcribed from train_mimo.py:1402-1456 env_kwargs_for for the released run's phase "mature"
    (scene_binding "phase" gives PHASE_SCENE["mature"] = "crib", train_mimo.py:93-94; age_binding "fixed:24"
    gives age 24.0; delay_binding "off" leaves the delays at 0). The caregiver's recording folder is set empty.
    OUTPUT: the keyword dict for BridgeEnv."""
    scene = str(a["scene"])
    kw: Dict[str, Any] = dict(
        scene=scene, run_dir=scene_dir, vision=not bool(a["no_vision"]), render_every=int(a["render_every"]),
        touch_scale_mult=float(a["touch_scale_mult"]), seed=int(a["seed"]), tag=tag,
        kp=float(a["decoder_kp"]), joint_damping_scale=float(a["joint_damping_scale"]),
        hinge_armature=float(a["hinge_armature"]), finger_armature=float(a.get("finger_armature", 0.0004)),
        hand_target_mode=a.get("hand_target_mode", "world"),
        fovea_size=int(a["fovea_size"]), fovea_fov=float(a["fovea_fov"]),
        caregiver_seconds=float(a["caregiver_seconds"]), skin_profile=a.get("skin_profile", ""),
        contact_profile=a.get("contact_profile", "legacy"), body_function=a.get("body_function", "off"),
        touch_geometry=a.get("touch_geometry", "body"), caregiver_mode=a.get("caregiver_mode", "off"),
        caregiver_speech_dir=speech_dir, caregiver_replay=a.get("caregiver_replay", ""),
        caregiver_record_dir="",
        caregiver_present_seconds=float(a.get("caregiver_present_seconds", 60.0)),
        caregiver_absent_seconds=float(a.get("caregiver_absent_seconds", 15.0)),
        toy_hand_off=a.get("toy_hand_off", "off"), body_layout=a.get("body_layout", "legacy"),
        gaze_reflex=(a["gaze_reflex"] == "on"), spontaneous_saccades=a["spontaneous_saccades"] != "off",
        cmd_horizon=max(1, int(a["action_repeat"]) * max(1, int(a["motor_hold"]))),
        mobile=a["mobile"] == "on", ring_fov=float(a["ring_fov"]), ring_size=int(a["ring_size"]),
        critters=a["critters"] == "on", audio_real=a["audio"] == "real",
        gaze_kp=float(a["gaze_kp"]), gaze_kd=float(a["gaze_kd"]))
    binding = str(a["age_binding"])
    if binding.startswith("fixed:"):
        kw["age"] = float(binding.split(":", 1)[1])
    elif binding == "curriculum":
        kw["age"] = 24.0   # PHASE_AGE["mature"], train_mimo.py:88
    if a.get("delay_binding", "off") == "on":
        raise ValueError("delay_binding on is not the released configuration (sensory delay must stay 0)")
    for name, default in (("parent_relative_targets", "off"), ("reach_projection", "off"), ("position_rows", "all")):
        value = a.get(name, default) or default
        if value != default:
            kw[name] = value
    if a.get("contact_footprint_guard", "off") != "off":
        kw["contact_footprint_guard"] = a["contact_footprint_guard"]
    if a.get("rubbing_calibration", "v4") != "v4":
        kw["rubbing_calibration"] = a["rubbing_calibration"]
    if a.get("touch_inside_rays", "process") != "process":
        kw["touch_inside_rays"] = a["touch_inside_rays"]
    if (a.get("finger_joint_servo", "off") or "off") != "off":
        kw["finger_joint_servo"] = a["finger_joint_servo"]
    return kw


def speech_dir_here(recorded: str) -> str:
    """INPUT: the speech folder path as the released args.json names it on the training machine. OPERATION:
    re-root it at this checkout (mimo_bridge.repo_paths.localize), else fall back to the checkout's copy under
    Saved/mimo_bridge. OUTPUT: an existing folder path; raises FileNotFoundError when neither exists."""
    from mimo_bridge.repo_paths import localize
    candidates = [localize(recorded), os.path.join(REPO_ROOT, "Saved", "mimo_bridge",
                                                   os.path.basename(recorded.replace("\\", "/").rstrip("/")))]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("caregiver speech folder not found at %s" % candidates)


def apply_object_flags(model, objects: Sequence[Dict[str, Any]], flags: Sequence[int],
                       build_alpha: Dict[int, float], build_collide: Dict[int, Tuple[int, int]]) -> List[Dict[str, Any]]:
    """INPUT: the MjModel, the header's object list (each with its geom ids), the record's flag byte per object
    (bit 1 visible, bit 2 collidable), and the alpha and (contype, conaffinity) every geom had when the world was
    built. OPERATION: caregiver.py:304-319's rule per geom: hidden -> geom_rgba alpha 0; shown -> the build's
    alpha, or 1.0 where the build left 0 (toys are built invisible, caregiver.py:149, 158); not collidable ->
    contype and conaffinity 0; collidable -> the build's values, or 1 where the build left 0. OUTPUT: one row
    per object with the flag read and the alpha and contype written, for cameras.json."""
    applied = []
    for k, obj in enumerate(objects):
        flag = int(flags[k])
        visible = bool(flag & FLAG_VISIBLE)
        collidable = bool(flag & FLAG_COLLIDABLE)
        alphas, cons = [], []
        for g in obj["geoms"]:
            g = int(g)
            if visible:
                alpha = build_alpha[g] if build_alpha[g] > 0 else 1.0
            else:
                alpha = 0.0
            model.geom_rgba[g, 3] = alpha
            ct, ca = build_collide[g]
            if collidable:
                ct, ca = (ct if ct else 1), (ca if ca else 1)
            else:
                ct, ca = 0, 0
            model.geom_contype[g] = ct
            model.geom_conaffinity[g] = ca
            alphas.append(float(alpha))
            cons.append([int(ct), int(ca)])
        applied.append(dict(name=obj["name"], flag=flag, visible=visible, collidable=collidable,
                            geoms=[int(g) for g in obj["geoms"]], alpha_written=alphas, contype_conaffinity_written=cons))
    return applied


def place_mounted_objects(mujoco, model, data, objects: Sequence[Dict[str, Any]], object_pose: np.ndarray,
                          tol_m: float = 1e-5) -> List[Dict[str, Any]]:
    """INPUT: the model and data (data.qpos and mocap already set), the header's objects, the record's root poses
    [K, 7] (x, y, z, qw, qx, qy, qz). OPERATION: mj_forward; for an object without a mocap body and without a
    free joint whose root's parent is the world body, compare forward kinematics' root position with the record;
    beyond tol_m, write the record's position and quaternion into model.body_pos / body_quat of the root and
    mj_forward again (the caregiver moves the mobile's mount by editing the model, which a saved model cannot
    know; observer_record.py:44-46). OUTPUT: one row per mounted object: the difference before, whether the
    root was moved, and the difference after."""
    mujoco.mj_forward(model, data)
    rows = []
    for k, obj in enumerate(objects):
        if int(obj["mocap"]) >= 0 or bool(obj["free"]):
            continue
        root = int(obj["root"])
        rec = np.asarray(object_pose[k], dtype=np.float64)
        before = float(np.linalg.norm(np.asarray(data.xpos[root]) - rec[:3]))
        moved = False
        if before > tol_m and int(model.body_parentid[root]) == 0:
            model.body_pos[root] = rec[:3]
            model.body_quat[root] = rec[3:]
            mujoco.mj_forward(model, data)
            moved = True
        after = float(np.linalg.norm(np.asarray(data.xpos[root]) - rec[:3]))
        rows.append(dict(name=obj["name"], root_body=root, parent_body=int(model.body_parentid[root]),
                         position_difference_before_m=before, root_moved=moved, position_difference_after_m=after,
                         recorded_pose=rec.tolist()))
    return rows


def png_size(path: str) -> Tuple[int, int]:
    """INPUT: a PNG path. OPERATION: read the IHDR chunk's width and height (bytes 16 to 24, big-endian).
    OUTPUT: (width, height) in pixels."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError("%s is not a PNG" % path)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def build_world(args_json: Dict[str, Any], scene_dir: str):
    """INPUT: the released args.json dict and a scene folder. OPERATION: env_kwargs_from_args, then BridgeEnv.
    OUTPUT: (env, kwargs); the caller closes the env."""
    from mimo_bridge.env import BridgeEnv
    kw = env_kwargs_from_args(args_json, scene_dir, speech_dir_here(str(args_json["caregiver_speech_dir"])),
                              tag="_eyefix")
    return BridgeEnv(**kw), kw


def check_model_against_header(env, header: Dict[str, Any]) -> Dict[str, Any]:
    """INPUT: the built env and the recording's header. OPERATION: structure_fingerprint of the built model
    over the infant's own bodies (env._own_body_ids, env.py:1201-1203) against header model_fingerprint; nq, nv,
    nbody, nmocap against the header. OUTPUT: the comparison rows and `match` (all equal)."""
    from mimo_bridge.observer_record import structure_fingerprint
    m = env.model
    fp = structure_fingerprint(m, set(int(b) for b in env._own_body_ids))
    counts = dict(nq=int(m.nq), nv=int(m.nv), nbody=int(m.nbody), nmocap=int(m.nmocap))
    header_counts = {k: int(header[k]) for k in counts}
    header_fp = str(header.get("model_fingerprint", ""))   # absent in an observer_world_v1 header: refused as a mismatch
    match = fp == header_fp and counts == header_counts
    return dict(structure_fingerprint=fp, header_fingerprint=header_fp, built=counts,
                header=header_counts, match=bool(match))


def camera_rows(env, out_dir: str) -> List[Dict[str, Any]]:
    """INPUT: the env at the state (mj_forward done) and the state folder. OPERATION: for each of the five
    cameras read its id, body, field of view (model.cam_fovy) and world pose (data.cam_xpos, data.cam_xmat).
    OUTPUT: one row per camera with its png name and size."""
    m, d = env.model, env.data
    rows = []
    for name, png in CAMERA_FILES:
        cid = int(m.camera(name).id)
        size = env.fovea_size if name.startswith("fovea") else env.eye_size
        rows.append(dict(name=name, id=cid, body_id=int(m.cam_bodyid[cid]), body_name=str(m.body(int(m.cam_bodyid[cid])).name),
                         size_px=int(size), fovy_deg=float(m.cam_fovy[cid]),
                         pos_m=[float(x) for x in d.cam_xpos[cid]],
                         xmat_row_major=[float(x) for x in np.asarray(d.cam_xmat[cid]).reshape(-1)],
                         png=png, path=os.path.join(out_dir, png).replace("\\", "/")))
    return rows


def render_state(env) -> Dict[str, np.ndarray]:
    """INPUT: the env at the state. OPERATION: env.render_eyes, env.render_fovea, env.render_ring (env.py:
    2200-2210, 1494-1502). OUTPUT: camera name -> (H, W, 3) uint8 RGB, rows top-down."""
    left, right = env.render_eyes()
    fl, fr = env.render_fovea()
    ring = env.render_ring()
    if fl is None or fr is None or ring is None:
        raise RuntimeError("the world has no fovea or ring camera: fovea_size %s ring_size %s"
                           % (env.fovea_size, env.ring_size))
    return {"eye_left": left, "eye_right": right, "fovea_left": fl, "fovea_right": fr, "ring_left": ring}


def patch_state(env, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """INPUT: the env and its five rendered images. OPERATION: env.patcher (BinocularPatcher over the wide and
    fovea pairs, vision_adapter.py:136-215) with the eyes' yaw and pitch from the eye joints (env.eye_yaw_pitch,
    env.py:2212-2220). OUTPUT: the patches.json document body (without the state block)."""
    (ly, lp), (ry, rp) = env.eye_yaw_pitch()
    sources = {}
    for name, size, fovy in env.vision_layout:
        if name == "wide":
            sources[name] = (images["eye_left"], images["eye_right"])
        elif name == "fovea":
            sources[name] = (images["fovea_left"], images["fovea_right"])
        else:
            raise RuntimeError("unknown vision source %r" % name)
    patches, angles = env.patcher(sources, (ly, lp), (ry, rp))
    n = int(patches.shape[0])
    if n != env.patcher.n_total or angles.shape != (n, 4) or patches.shape[1] != 768:
        raise RuntimeError("patcher shapes %s %s against n_total %d" % (patches.shape, angles.shape, env.patcher.n_total))
    src_rows = []
    for name, size, fovy, base, span, _rays in env.patcher.sources:
        src_rows.append(dict(name=name, cameras=["eye_left", "eye_right"] if name == "wide" else ["fovea_left", "fovea_right"],
                             size_px=int(size), fovy_deg=float(fovy), patches_per_eye=int(base.shape[0]),
                             span_rad=float(span), px_per_deg=float(size / fovy)))
    return dict(
        n_total=n, n_per_eye=int(env.patcher.n_per_eye), patch_px=int(env.patcher.patch),
        expected_n_rule="2 eyes x (wide 256/16 squared = 256 patches + fovea 288/16 squared = 324 patches) = 1,160",
        describe=env.patcher.describe(), sources=src_rows,
        row_order="per source in the order listed, left and right eye interleaved (row 0 left patch 0, row 1 right "
                  "patch 0, ...), patches row-major over the image's 16 x 16 grid; a patch row is 768 numbers = "
                  "16 rows x 16 columns x RGB, each pixel / 255 (vision_adapter.py:104-115, 187-215)",
        gaze=dict(left_yaw_pitch_rad=[float(ly), float(lp)], right_yaw_pitch_rad=[float(ry), float(rp)],
                  source="env.eye_yaw_pitch: the eye joints' qpos (right yaw negated), env.py:2212-2220"),
        angle_columns=["azimuth_rad (positive = eye-left)", "elevation_rad", "eye (-1 left, +1 right)", "span_rad"],
        first_angle_rows=[[float(x) for x in row] for row in angles[:4]],
        first_patch_row_head=[float(x) for x in patches[0, :12]],
        patches_shape=[n, int(patches.shape[1])], angles_shape=[n, 4],
        patches_sha256=hashlib.sha256(np.ascontiguousarray(patches, dtype=np.float32).tobytes()).hexdigest(),
        angles_sha256=hashlib.sha256(np.ascontiguousarray(angles, dtype=np.float32).tobytes()).hexdigest(),
        digest_rule="sha256 over the contiguous float32 bytes, row-major")


def tactile_state(env) -> Dict[str, Any]:
    """INPUT: the env at the state (mj_forward done; qvel zero). OPERATION: MIMo's `touch` module (TrimeshTouch) reads the contacts
    (TrimeshTouch.get_touch_obs -> touch.sensor_outputs: body id -> [points, 3] force in the body frame, N);
    env._contacts builds the ContactInfo list (env.py:2305-2357); a fresh TouchPooler with the world's patch
    specs and dt (touch_adapter.py:186-210) computes the [53, 10] fields in one update (touch_adapter.py:
    214-346); pad_to_slots gives the frame's [256, 10]. OUTPUT: the tactile_patches.json document body."""
    from mimo_bridge.touch_adapter import TouchPooler
    if env.data is not env.env.data:
        raise RuntimeError("BridgeEnv.data is not MIMo's data: env.touch would read another state")
    env.touch.get_touch_obs()
    point_forces = env.touch.sensor_outputs
    contacts = env._contacts()
    pooler = TouchPooler(env.pooler.specs, env.dt, vib_window=env.pooler.vib_window)
    fields = pooler.update(point_forces, contacts)
    if fields.shape != (pooler.n_patches, len(TACTILE_FIELDS)):
        raise RuntimeError("fields shape %s" % (fields.shape,))
    padded = pad_to_slots(fields)
    patch_rows = [dict(row=i, body_id=int(s.body_id), body_name=str(s.body_name), scale_m=float(s.scale_m),
                       point_area_mm2=float(pooler.point_area_mm2[i]),
                       sensor_count=int(np.asarray(point_forces[int(s.body_id)]).shape[0]))
                  for i, s in enumerate(pooler.specs)]
    forces = {}
    for body_id, arr in point_forces.items():
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        idx = np.flatnonzero(np.abs(arr).max(axis=1) > 0.0)
        if idx.size:
            forces[str(int(body_id))] = [[int(i), float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])] for i in idx]
    contact_rows = []
    for c in contacts:
        contact_rows.append(dict(
            body_id=int(c.body_id), other_body_name=str(c.other_body_name),
            pair_key=[int(x) for x in c.pair_key], normal_force_n=float(c.normal_force),
            tangential_body_n=[float(x) for x in c.tangential_body], pos_body_m=[float(x) for x in c.pos_body],
            pos_world_m=[float(x) for x in c.pos_world], normal_world=[float(x) for x in c.normal_world],
            normal_body=None if c.normal_body is None else [float(x) for x in c.normal_body],
            penetration_m=float(c.penetration_m), rel_normal_speed_m_s=float(c.rel_normal_speed)))
    return dict(
        dt_s=float(env.dt), vib_window=int(pooler.vib_window), n_patches=int(pooler.n_patches), slots=TOUCH_SLOTS,
        field_names=[f[0] for f in TACTILE_FIELDS], field_units=[f[1] for f in TACTILE_FIELDS],
        field_rules=[f[2] for f in TACTILE_FIELDS],
        fields=[[float(x) for x in row] for row in fields],
        fields_padded=[[float(x) for x in row] for row in padded],
        padding_rule="rows %d to %d are zero: the frame's unused patch slots (tokenizer/config.py:51; "
                     "frame_encoder.py:119-120, 240)" % (int(pooler.n_patches), TOUCH_SLOTS - 1),
        patches=patch_rows,
        inputs=dict(point_forces_body_frame_n=forces,
                    point_forces_rule="body id -> [point index, fx, fy, fz] for every skin point with a non-zero "
                                      "force (MIMo TrimeshTouch.get_touch_obs; the skin layout is "
                                      "skin_layout_age24_trunk_v1.json); points not listed carry zero force",
                    contacts=contact_rows, n_contacts=len(contact_rows)),
        computation="TouchPooler.update(point_forces, contacts) of a fresh pooler: one update, so the slip rate "
                    "is 0 and the vibration bands come from a history holding one sample; qvel is zero (the "
                    "recording holds no velocities). The trainer's mechanical_v4 profile overwrites columns 0 to 3 "
                    "with its virtual skin's aggregate (env.py:3121-3124); these are the TouchPooler's own fields, "
                    "the quantity WI-28 reproduces")


def write_json(path: str, doc: Dict[str, Any]) -> int:
    """INPUT: a path and a JSON-serialisable dict. OPERATION: json.dumps (compact, NaN refused) plus a newline.
    OUTPUT: bytes written."""
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.write("\n")
    return len(text) + 1


def export_state(env, mujoco, header: Dict[str, Any], chunk: Dict[str, Any], chunk_path: str, row: int,
                 out_root: str, build_alpha: Dict[int, float], build_collide: Dict[int, Tuple[int, int]],
                 common: Dict[str, Any]) -> Dict[str, Any]:
    """INPUT: the built env, the header, one read chunk and a row in it, the output root, the build's geom
    appearance, the common document block. OPERATION: steps (3) to (6) of the module docstring, then the five
    PNGs and three JSON files. OUTPUT: a summary row (state name, image sizes, N, n_contacts)."""
    import imageio.v3 as iio
    m, d = env.model, env.data
    step = int(chunk["step"][row])
    name = "step_%09d" % step
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    qpos = np.asarray(chunk["qpos"][row], dtype=np.float64)
    if qpos.shape != (int(m.nq),):
        raise RuntimeError("qpos %s against nq %d" % (qpos.shape, m.nq))
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    d.time = float(chunk["sim_time"][row])
    mocap = np.asarray(chunk["mocap"][row], dtype=np.float64)
    if mocap.shape != (int(m.nmocap), 7):
        raise RuntimeError("mocap %s against nmocap %d" % (mocap.shape, m.nmocap))
    d.mocap_pos[:] = mocap[:, :3]
    d.mocap_quat[:] = mocap[:, 3:]
    flags_applied = apply_object_flags(m, header["objects"], chunk["object_flags"][row], build_alpha, build_collide)
    mounted = place_mounted_objects(mujoco, m, d, header["objects"], np.asarray(chunk["object_pose"][row]))
    images = render_state(env)
    sizes = {}
    for cam, png in CAMERA_FILES:
        img = np.ascontiguousarray(images[cam], dtype=np.uint8)
        iio.imwrite(os.path.join(out_dir, png), img)
        sizes[cam] = [int(img.shape[1]), int(img.shape[0])]
    patches = patch_state(env, images)
    tactile = tactile_state(env)
    phase_names = [str(x) for x in np.asarray(chunk["caregiver_phase_names"]).reshape(-1)]
    phase = int(chunk["caregiver_phase"][row])
    state = dict(
        name=name, step=step, frame=int(chunk["frame"][row]), episode=int(chunk["episode"][row]),
        sim_time_s=float(chunk["sim_time"][row]), chunk_file=os.path.basename(chunk_path), row=int(row),
        header_file=chunk["header_name"], caregiver_present=bool(chunk["caregiver_present"][row]),
        caregiver_phase=phase_names[phase] if 0 <= phase < len(phase_names) else str(phase),
        toy_held=int(chunk["toy_held"][row]),
        qpos=[float(x) for x in qpos], qvel="zero (the recording holds no velocities)",
        mocap_pos_m=mocap[:, :3].tolist(), mocap_quat_wxyz=mocap[:, 3:].tolist(),
        object_flags=flags_applied, mounted_objects=mounted,
        object_pose_check=[dict(name=o["name"], root_body=int(o["root"]),
                                position_difference_m=float(np.linalg.norm(
                                    np.asarray(d.xpos[int(o["root"])]) - np.asarray(chunk["object_pose"][row][k][:3]))))
                           for k, o in enumerate(header["objects"])])
    scene = env._renderer_obj().scene
    renderer = dict(
        call="EyeRenderer.render (env.py:625-638): mjv_updateScene(model, data, option, None, fixed camera, "
             "mjCAT_ALL, scene); mjr_setBuffer(mjFB_OFFSCREEN); mjr_render(rect, scene, context); "
             "mjr_readPixels; rows flipped top-down (mjr_readPixels gives bottom-up rows)",
        library="MuJoCo classic OpenGL renderer, mujoco %s, one mujoco.GLContext per process (env.py:590-601)"
                % mujoco.__version__,
        shadows=int(scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW.value]),
        reflections=int(scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION.value]),
        offscreen_buffer_px=[int(m.vis.global_.offwidth), int(m.vis.global_.offheight)],
        font_scale="mjFONTSCALE_150", max_geom=int(scene.maxgeom))
    cameras = dict(common, format=FORMAT_CAMERAS, state=state, cameras=camera_rows(env, out_dir), renderer=renderer,
                   image_format="PNG, RGB 8-bit, rows top-down, columns left-to-right")
    sizes_json = write_json(os.path.join(out_dir, "cameras.json"), cameras)
    write_json(os.path.join(out_dir, "patches.json"),
               dict(common, format=FORMAT_PATCHES, state=dict(name=name, step=step, chunk_file=state["chunk_file"],
                                                              row=int(row)), **patches))
    write_json(os.path.join(out_dir, "tactile_patches.json"),
               dict(common, format=FORMAT_TACTILE, state=dict(name=name, step=step, chunk_file=state["chunk_file"],
                                                              row=int(row)), **tactile))
    return dict(name=name, step=step, sizes=sizes, n_total=int(patches["n_total"]),
                n_contacts=int(tactile["inputs"]["n_contacts"]), caregiver_present=state["caregiver_present"],
                cameras_json_bytes=sizes_json, out_dir=out_dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--release-run", default=RELEASED_RUN, help="the released run folder (args.json, observer/)")
    parser.add_argument("--body", type=int, default=0, help="the observer body folder b<body> (default 0)")
    parser.add_argument("--states", default=DEFAULT_STATES, help="chunk:row pairs, comma separated")
    parser.add_argument("--out", default=DEFAULT_OUT, help="the fixture root (default: the UEMASensesTests folder)")
    parser.add_argument("--scene-dir", default="", help="scene folder for the build (default: a temporary folder, "
                                                         "removed afterwards)")
    parser.add_argument("--expect-n", type=int, default=EXPECTED_N, help="the patch count the run must give")
    args = parser.parse_args(argv)
    import mujoco
    from mimo_bridge.observer_record import read_chunk, read_header
    states = parse_states(args.states)
    with open(os.path.join(args.release_run, "args.json"), encoding="utf-8") as f:
        args_json = json.load(f)
    body_dir = os.path.join(args.release_run, "observer", "world", "b%d" % args.body)
    chunk_files = sorted(glob.glob(os.path.join(body_dir, "chunk_*.npz")))
    if not chunk_files:
        print("REFUSED: no chunk files under %s" % body_dir)
        return 2
    for chunk_i, row in states:
        if chunk_i >= len(chunk_files):
            print("REFUSED: chunk %d asked, %d chunk files under %s" % (chunk_i, len(chunk_files), body_dir))
            return 2
    written_at = _dt.datetime.now(_dt.timezone.utc).astimezone().strftime("%Y-%m-%d %I:%M %p %Z")
    scene_dir = args.scene_dir or tempfile.mkdtemp(prefix="eye_fixtures_")
    env, kw = build_world(args_json, scene_dir)
    summary: List[Dict[str, Any]] = []
    try:
        first = read_chunk(chunk_files[0])
        header = first["header"]
        check = check_model_against_header(env, header)
        print("model check: built fingerprint %s, header %s, counts %s against %s -> %s"
              % (check["structure_fingerprint"][:16], check["header_fingerprint"][:16], check["built"],
                 check["header"], "MATCH" if check["match"] else "MISMATCH"))
        if not check["match"]:
            print("REFUSED: the built world is not the recorded world; nothing written")
            return 1
        m = env.model
        build_alpha = {int(g): float(m.geom_rgba[g, 3]) for g in range(int(m.ngeom))}
        build_collide = {int(g): (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(int(m.ngeom))}
        import imageio
        common = dict(
            written_by="apple/scripts/export_eye_fixtures.py", written_at=written_at,
            versions=dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                          imageio=imageio.__version__),
            release_run=dict(folder=os.path.relpath(args.release_run, REPO_ROOT).replace("\\", "/"),
                             run=str(header.get("run", "")), body=int(args.body), header_file=first["header_name"]),
            world=dict(env_kwargs={k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v))
                                   for k, v in kw.items() if k not in ("run_dir", "caregiver_speech_dir")},
                       eye_size_px=int(env.eye_size), fovea_size_px=int(env.fovea_size), fovea_fov_deg=float(env.fovea_fov),
                       ring_size_px=int(env.ring_size), ring_fov_deg=float(env.ring_fov),
                       vision_layout=[[str(n), int(s), float(f)] for n, s, f in env.vision_layout],
                       scene_xml=os.path.basename(env.scene_path), model_check=check,
                       deviations_from_release=["caregiver_record_dir empty (no recording folder is written); "
                                                "the world's structure is checked against the recording's "
                                                "fingerprint above"]))
        chunks: Dict[int, Dict[str, Any]] = {0: first}
        for chunk_i, row in states:
            if chunk_i not in chunks:
                chunks[chunk_i] = read_chunk(chunk_files[chunk_i], header)
            chunk = chunks[chunk_i]
            if row >= len(chunk["step"]):
                print("REFUSED: row %d asked, chunk %s holds %d records" % (row, chunk_files[chunk_i], len(chunk["step"])))
                return 2
            summary.append(export_state(env, mujoco, header, chunk, chunk_files[chunk_i], row, args.out,
                                        build_alpha, build_collide, common))
    finally:
        env.close()
        if not args.scene_dir:
            shutil.rmtree(scene_dir, ignore_errors=True)
    ok = True
    for s in summary:
        sizes = s["sizes"]
        line = ", ".join("%s %dx%d" % (cam, sizes[cam][0], sizes[cam][1]) for cam, _ in CAMERA_FILES)
        print("state %s: %s; N=%d; contacts %d; adult present %s -> %s"
              % (s["name"], line, s["n_total"], s["n_contacts"], s["caregiver_present"], s["out_dir"]))
        for cam, png in CAMERA_FILES:
            want = env.fovea_size if cam.startswith("fovea") else env.eye_size
            disk = png_size(os.path.join(s["out_dir"], png))
            if disk != (want, want) or sizes[cam] != [want, want]:
                print("FAIL: %s/%s is %s, expected %dx%d" % (s["name"], png, disk, want, want))
                ok = False
        if s["n_total"] != args.expect_n:
            print("FAIL: %s N=%d, expected %d" % (s["name"], s["n_total"], args.expect_n))
            ok = False
    try:
        import psutil
        peak = psutil.Process().memory_info()
        print("peak working set %s MB" % f"{int(getattr(peak, 'peak_wset', peak.rss) / 2 ** 20):,}")
    except Exception as ex:   # the memory line is a report, never a gate
        print("peak working set unread: %r" % (ex,))
    if ok:
        print("PASS: %d states, 5 PNGs each at the stated sizes, N=%d" % (len(summary), args.expect_n))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
