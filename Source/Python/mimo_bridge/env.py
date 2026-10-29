# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Make and step one MIMo env for the bridge.

What this file does, in plain words:
  * writes its OWN scene XML (v2 five-finger body) that points at MIMo's model files by
    absolute path, so nothing from MIMo is copied and MIMo's own scenes are untouched;
  * optionally "grows" MIMo to an age (months) with MIMo's growth module, done under a
    cross-process lock because MIMo writes fixed-name temp files inside its own tree;
  * builds the MIMoEnv subclass with touch (v2 scales), proprioception, vestibular and
    the MOTOR delay, and NO MIMo vision (we render the two eye cameras ourselves, see
    Phase 0). A SENSORY delay is REFUSED: MIMo applies it inside _get_obs and _collect
    below reads live MuJoCo data instead, so it reached nothing - sensory_delay 4 against
    0 left proprioception, touch and vestibular identical on 12 of 12 steps, maximum
    difference 0.000e+00 (measured 2026-09-16), while the same probe with motor_delay 3
    against 0 differed on 11 of 12. Rather than keep a knob that silently does nothing,
    the constructor raises on a nonzero sensory_delay; delivering one would mean queueing
    the sense arrays in BridgeEnv.step and is a separate operator decision;
  * after each step hands back raw arrays: touch patches + collisions (touch_adapter),
    23 bones (proprio_adapter), eye images, vestibular, time.

Scenes:
  reach    MIMo hangs in the air (upper body welded), gravity on, a small ball floats in
           front of the right hand (mocap body; new random spot on every reset)
  selfbody MIMo sits on the floor (base welded), legs bent, both hands free; good for
           self-touch
  floor    MIMo stands free on the floor and falls / rolls (lots of contact)
"""
from __future__ import annotations

import logging
import os
import time
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mimo_bridge.touch_adapter import PatchSpec, ContactInfo, TouchPooler, build_collisions, bone_hash, region_of
from mimo_bridge.proprio_adapter import ProprioAdapter, ProprioReadout, BONE_MAP
from mimo_bridge.vestibular_adapter import read_vestibular
from mimo_bridge.pen_art import make_pen_textures, make_floor_texture, make_ceiling_texture
from mimo_bridge.action_decoder import ActionDecoder
from mimo_bridge.vision_adapter import BinocularPatcher, EyePatcher, acuity

SCENES = ("reach", "selfbody", "floor", "crib")


# ---------------------------------------------------------------------------- paths
def mimo_assets_dir() -> str:
    import mimo_infant
    return os.path.join(os.path.dirname(mimo_infant.__file__), "simulation", "assets").replace("\\", "/")


def mimo_commit() -> str:
    """git commit of the installed MIMo checkout (for the report)."""
    import mimo_infant, subprocess
    root = os.path.dirname(os.path.dirname(mimo_infant.__file__))
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------- scene XML
_SITTING_V2 = {   # taken from MIMo's SITTING_POSITION where the v2 joint exists (radians)
    "robot:hip_lean1": 0.039088, "robot:hip_rot1": 0.113112, "robot:hip_bend1": 0.5323,
    "robot:hip_bend2": 0.5323,
    "robot:left_shoulder_horizontal": 0.683242, "robot:left_shoulder_ad_ab": 0.3747,
    "robot:left_shoulder_rotation": -0.62714, "robot:left_elbow": -0.756016, "robot:left_hand1": 0.28278,
    "robot:right_hip1": -1.51997, "robot:right_hip2": -0.397578, "robot:right_hip3": 0.0976615,
    "robot:right_knee": -1.85479, "robot:right_foot1": -0.585865, "robot:right_foot2": -0.358165,
    "robot:left_hip1": -1.23961, "robot:left_hip2": -0.8901, "robot:left_hip3": 0.7156,
    "robot:left_knee": -2.531, "robot:left_foot1": -0.63562, "robot:left_foot2": 0.5411,
    "robot:left_foot3": 0.366514, "robot:left_toes": 0.24424,
}


# A newborn's resting posture, lying on its back. Physiological flexion: what is left over
# from being folded up in the womb, held by muscle tone rather than chosen. Elbows bent
# about a right angle, shoulders drawn in and turned inward, hips flexed and opened out
# (the frog-leg position), knees bent. Ours lies like a plank instead, because no scene but
# selfbody ever set a joint angle - every joint sits at the model's neutral STANDING pose.
#
# These numbers come from infant anatomy, NOT from tuning. I first went looking for the
# flexion that put the hands nearest the line of sight and that was the wrong question: it
# makes the starting posture a way of engineering hand regard instead of letting it be
# discovered. Where the hands land here is an observation, not a target.
_NEWBORN_SUPINE = {
    "elbow": -90.0,               # bent a right angle
    "shoulder_ad_ab": 25.0,       # drawn in, a little away from the body
    "shoulder_horizontal": 20.0,
    "shoulder_rotation": -30.0,   # turned inward
    "hip1": -70.0,                # hips flexed
    "hip2": -30.0,                # and opened out: frog legs
    "hip3": 15.0,
    "knee": -85.0,                # knees bent
    "foot1": -10.0,
}


def newborn_supine_qpos() -> dict:
    """The resting posture as MIMo joint names -> radians, both sides."""
    import math
    out = {}
    for side in ("right", "left"):
        for joint, deg in _NEWBORN_SUPINE.items():
            out[f"robot:{side}_{joint}"] = math.radians(deg)
    return out


def lite_meta_xml(out_dir: str) -> str:
    """Write a generated copy of MIMo's meta v2 XML with the huge face/cloth textures replaced
    by tiny flat colours, and return its path.

    Why: MIMo's head, sleeve and pants textures are 2500 x 15000 cube maps, 1 GB of texture
    data per loaded model, so one MIMo env costs about 3 GB of RAM and 16 workers do not fit
    in 32 GB (found in E5). The geometry, joints, actuators, sensors and equality constraints
    are untouched; only texture files are dropped (colour stays rgb1). This file is generated
    at run time from MIMo's file, in the bridge's run folder, never committed.
    """
    import xml.etree.ElementTree as ET
    src = os.path.join(mimo_assets_dir(), "mimo", "MIMo_metav2.xml")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"MIMo_metav2_lite_p{os.getpid()}.xml")   # one file per process: no races
    tree = ET.parse(src)
    root = tree.getroot()
    for tex in root.iter("texture"):
        if any(k.startswith("file") for k in tex.attrib):
            for k in list(tex.attrib):
                if k.startswith("file"):
                    del tex.attrib[k]
            tex.set("builtin", "flat")
            tex.set("width", "16")
            tex.set("height", "16")
            if tex.get("type") == "cube":
                tex.set("height", "96")      # cube textures want 6 square faces stacked
            tex.attrib.setdefault("rgb1", "0.95 0.8 0.5")
    tree.write(dst, encoding="unicode")
    return dst.replace("\\", "/")


def fovea_model_xml(out_dir: str, fovea_fov_deg: float, ring_fov_deg: float = 0.0) -> str:
    """Write a copy of MIMo's body XML with a narrow-field camera added inside each eye.

    Why a second camera instead of a bigger render. Acuity is pixels per DEGREE, so the
    cheap way to see detail is to spend the pixels over less world: 192 px across 16
    degrees is 12 px/deg, where the wide 256 px across 60 degrees is 4.27. Reaching the
    same acuity with one uniform camera would need 720 x 720 and 4050 patches per eye
    against 400. This is what a fovea is for, and it is what the UE avatar's configured
    pipeline already does (panorama 2520x1560 over 210 degrees plus a 512 px fovea over 5).

    The fovea camera is BORESIGHTED with the wide one - same position, same orientation,
    narrower field - so it inherits the eye's gaze and there is no rotation convention to
    get wrong. No toe-in is needed at this field: the ball sits 25-35 cm out, which puts it
    4.7 to 5.6 degrees off each eye's axis, inside a 16 degree fovea in BOTH eyes.
    """
    import xml.etree.ElementTree as ET
    src = os.path.join(mimo_assets_dir(), "mimo", "MIMo_modelv2.xml")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"MIMo_modelv2_fovea{fovea_fov_deg:g}"
                                f"{f'_ring{ring_fov_deg:g}' if ring_fov_deg > 0 else ''}"
                                f"_p{os.getpid()}.xml")
    tree = ET.parse(src)
    added = []
    for body in tree.getroot().iter("body"):
        for cam in list(body.findall("camera")):
            if cam.get("name") in ("eye_left", "eye_right") and fovea_fov_deg > 0:
                fov = ET.SubElement(body, "camera")
                fov.attrib.update(cam.attrib)
                fov.set("name", cam.get("name").replace("eye_", "fovea_"))
                fov.set("fovy", f"{fovea_fov_deg:g}")
                added.append(fov.get("name"))
            if cam.get("name") == "eye_left" and ring_fov_deg > 0:
                # the far-periphery ring: rod-like coarse motion field, read ONLY by the
                # snap reflex (left eye only, like the reflex's wide detector)
                ring = ET.SubElement(body, "camera")
                ring.attrib.update(cam.attrib)
                ring.set("name", "ring_left")
                ring.set("fovy", f"{ring_fov_deg:g}")
                added.append("ring_left")
    want = (["fovea_left", "fovea_right"] if fovea_fov_deg > 0 else []) \
        + (["ring_left"] if ring_fov_deg > 0 else [])
    if sorted(added) != sorted(want):
        raise RuntimeError(f"expected cameras {want}, added {added}")
    tree.write(dst, encoding="unicode")
    return dst.replace("\\", "/")


# THE MOBILE (2026-09-02). A damped pendulum from the nursery ceiling, placed by
# MEASUREMENT (probe suite v2, 3,552 live wrist positions: z p50 0.115, p75 0.164,
# p90 0.206 m; x/y medians +0.06/0.00) and three physics probes:
#   - a four-arm frame at z = 0.45 m, free to SPIN, with five bright shapes
#     (sphere/box/capsule/ellipsoid/sphere - no red: the target ball is red)
#     dangling to element undersides of 0.15-0.19 m: inside the band real wrists
#     cross a quarter of the time (p75..p90), and clear of the belly (a lower
#     single-bob try rested ON the torso: 1000/1000 contact steps);
#   - damping 0.015: a strike swings visibly for seconds and settles (0.01 swung
#     for minutes - the consequence outlived the 10 s goal memory and became
#     ambient motion; 0.08 died in 0.65 s); rest-contact measured 0/500 limp steps.
# Mass 20 g: light enough that a hand visibly launches it.
MOBILE_XML = '''
    <body name="mobile" pos="0.06 0.0 1.05">
      <joint name="mobile_swing" type="ball" damping="0.015"/>
      <geom name="mobile_string" type="capsule" fromto="0 0 0  0 0 -0.60" size="0.004"
            mass="0.005" rgba="0.92 0.92 0.86 1" condim="3"/>
      <body name="mobile_frame" pos="0 0 -0.60">
        <joint name="mobile_spin" type="hinge" axis="0 0 1" damping="0.010"/>
        <geom name="mobile_bar_x" type="capsule" fromto="-0.12 0 0  0.12 0 0" size="0.006"
              mass="0.010" rgba="0.85 0.85 0.80 1" condim="3"/>
        <geom name="mobile_bar_y" type="capsule" fromto="0 -0.12 0  0 0.12 0" size="0.006"
              mass="0.010" rgba="0.85 0.85 0.80 1" condim="3"/>
        <body name="mobile_el_e" pos="0.12 0 0">
          <joint name="mobile_j_e" type="ball" damping="0.004"/>
          <geom name="mobile_str_e" type="capsule" fromto="0 0 0  0 0 -0.20" size="0.003" mass="0.002" rgba="0.9 0.9 0.85 1" condim="3"/>
          <geom name="mobile_el_e_g" type="sphere" pos="0 0 -0.235" size="0.035" mass="0.012" rgba="0.15 0.45 0.95 1" condim="6"/>
        </body>
        <body name="mobile_el_w" pos="-0.12 0 0">
          <joint name="mobile_j_w" type="ball" damping="0.004"/>
          <geom name="mobile_str_w" type="capsule" fromto="0 0 0  0 0 -0.20" size="0.003" mass="0.002" rgba="0.9 0.9 0.85 1" condim="3"/>
          <geom name="mobile_el_w_g" type="box" pos="0 0 -0.23" size="0.03 0.03 0.03" mass="0.012" rgba="0.95 0.80 0.10 1" condim="6"/>
        </body>
        <body name="mobile_el_n" pos="0 0.12 0">
          <joint name="mobile_j_n" type="ball" damping="0.004"/>
          <geom name="mobile_str_n" type="capsule" fromto="0 0 0  0 0 -0.20" size="0.003" mass="0.002" rgba="0.9 0.9 0.85 1" condim="3"/>
          <geom name="mobile_el_n_g" type="capsule" fromto="0 0 -0.20  0 0 -0.27" size="0.022" mass="0.012" rgba="0.20 0.75 0.35 1" condim="6"/>
        </body>
        <body name="mobile_el_s" pos="0 -0.12 0">
          <joint name="mobile_j_s" type="ball" damping="0.004"/>
          <geom name="mobile_str_s" type="capsule" fromto="0 0 0  0 0 -0.20" size="0.003" mass="0.002" rgba="0.9 0.9 0.85 1" condim="3"/>
          <geom name="mobile_el_s_g" type="ellipsoid" pos="0 0 -0.235" size="0.045 0.025 0.030" mass="0.012" rgba="0.95 0.95 0.95 1" condim="6"/>
        </body>
        <body name="mobile_el_c" pos="0 0 0">
          <joint name="mobile_j_c" type="ball" damping="0.004"/>
          <geom name="mobile_str_c" type="capsule" fromto="0 0 0  0 0 -0.21" size="0.003" mass="0.002" rgba="0.9 0.9 0.85 1" condim="3"/>
          <geom name="mobile_el_c_g" type="sphere" pos="0 0 -0.255" size="0.045" mass="0.015" rgba="0.55 0.25 0.85 1" condim="6"/>
        </body>
      </body>
    </body>
'''

# The attention critters (2026-09-05, operator plan): three observability objects.
#  * squeaker - a kick panel placed just past the feet at reset (position measured from
#    the foot bodies, so it survives age growth). Mocap = kinematic: kicking cannot
#    shove it, the env animates a hop as its response. Contingent: it moves only when
#    struck. Collides (condim 6) so the kick is felt.
#  * dragonfly - hover-then-dart flyer; the ~half-second hover before each dart is the
#    anticipation cue. No collisions (contype/conaffinity 0): a visual mover.
#  * bee - slow jittery roamer, wings flickering at 20 Hz (operator-approved; real bee
#    wingbeat ~200 Hz would alias at our ~50 Hz render rate).
# Wing size check (plan A9), measured at 0.45 m viewing distance: dragonfly wingspan
# 6 cm ~ 7.6 deg ~ 32 px in the wide eye; bee wings 2.4 cm ~ 3 deg ~ 13 px - visible.
CRITTERS_XML = '''
    <body name="squeaker" mocap="true" pos="-0.45 0 0.10">
      <geom name="squeaker_base" type="cylinder" size="0.062 0.012" rgba="0.16 0.35 0.75 1" condim="6"/>
      <geom name="squeaker_geom" type="ellipsoid" pos="0 0 0.038" size="0.048 0.048 0.034"
            rgba="0.92 0.15 0.12 1" condim="6"/>
    </body>
    <body name="dragonfly" mocap="true" pos="0.30 0.30 0.50">
      <geom name="dragonfly_thorax" type="ellipsoid" pos="0 0 0" size="0.013 0.009 0.009"
            rgba="0.05 0.45 0.42 1" contype="0" conaffinity="0"/>
      <geom name="dragonfly_abdomen" type="capsule" fromto="-0.060 0 0.001  -0.012 0 0" size="0.0035"
            rgba="0.10 0.60 0.85 1" contype="0" conaffinity="0"/>
      <geom name="dragonfly_head" type="sphere" pos="0.017 0 0.002" size="0.0065"
            rgba="0.08 0.35 0.35 1" contype="0" conaffinity="0"/>
      <geom name="dragonfly_eye_l" type="sphere" pos="0.020 0.0045 0.005" size="0.0038"
            rgba="0.02 0.08 0.10 1" contype="0" conaffinity="0"/>
      <geom name="dragonfly_eye_r" type="sphere" pos="0.020 -0.0045 0.005" size="0.0038"
            rgba="0.02 0.08 0.10 1" contype="0" conaffinity="0"/>
      <geom name="dragonfly_wing_fl" type="ellipsoid" pos="0.005 0.034 0.006" size="0.010 0.034 0.0007"
            euler="6 0 10" rgba="0.82 0.90 0.98 0.75" contype="0" conaffinity="0"/>
      <geom name="dragonfly_wing_fr" type="ellipsoid" pos="0.005 -0.034 0.006" size="0.010 0.034 0.0007"
            euler="-6 0 -10" rgba="0.82 0.90 0.98 0.75" contype="0" conaffinity="0"/>
      <geom name="dragonfly_wing_hl" type="ellipsoid" pos="-0.010 0.032 0.005" size="0.011 0.031 0.0007"
            euler="6 0 -7" rgba="0.82 0.90 0.98 0.75" contype="0" conaffinity="0"/>
      <geom name="dragonfly_wing_hr" type="ellipsoid" pos="-0.010 -0.032 0.005" size="0.011 0.031 0.0007"
            euler="-6 0 7" rgba="0.82 0.90 0.98 0.75" contype="0" conaffinity="0"/>
    </body>
    <body name="bouquet" pos="0.42 0.40 0.14">
      <geom name="table_top" type="cylinder" pos="0 0 -0.012" size="0.10 0.012"
            rgba="0.60 0.42 0.26 1" contype="0" conaffinity="0"/>
      <geom name="table_leg1" type="capsule" fromto="-0.06 -0.06 -0.02  -0.07 -0.07 -0.14" size="0.008"
            rgba="0.50 0.34 0.20 1" contype="0" conaffinity="0"/>
      <geom name="table_leg2" type="capsule" fromto="0.06 -0.06 -0.02  0.07 -0.07 -0.14" size="0.008"
            rgba="0.50 0.34 0.20 1" contype="0" conaffinity="0"/>
      <geom name="table_leg3" type="capsule" fromto="0.06 0.06 -0.02  0.07 0.07 -0.14" size="0.008"
            rgba="0.50 0.34 0.20 1" contype="0" conaffinity="0"/>
      <geom name="table_leg4" type="capsule" fromto="-0.06 0.06 -0.02  -0.07 0.07 -0.14" size="0.008"
            rgba="0.50 0.34 0.20 1" contype="0" conaffinity="0"/>
      <geom name="vase" type="cylinder" pos="0 0 0.05" size="0.035 0.05" rgba="0.55 0.65 0.75 1"
            contype="0" conaffinity="0"/>
      <geom name="stem1" type="capsule" fromto="0 0 0.09  -0.05 -0.04 0.24" size="0.0035"
            rgba="0.20 0.55 0.20 1" contype="0" conaffinity="0"/>
      <geom name="stem2" type="capsule" fromto="0 0 0.09  0.05 0.03 0.28" size="0.0035"
            rgba="0.22 0.60 0.22 1" contype="0" conaffinity="0"/>
      <geom name="stem3" type="capsule" fromto="0 0 0.09  0.00 0.06 0.21" size="0.0035"
            rgba="0.18 0.50 0.18 1" contype="0" conaffinity="0"/>
      <geom name="stem4" type="capsule" fromto="0 0 0.09  -0.03 0.05 0.30" size="0.0035"
            rgba="0.20 0.55 0.20 1" contype="0" conaffinity="0"/>
      <geom name="flower1" type="ellipsoid" pos="-0.05 -0.04 0.25" size="0.030 0.030 0.014"
            rgba="0.95 0.20 0.25 1" contype="0" conaffinity="0"/>
      <geom name="flower2" type="ellipsoid" pos="0.05 0.03 0.29" size="0.028 0.028 0.013"
            rgba="0.98 0.85 0.15 1" contype="0" conaffinity="0"/>
      <geom name="flower3" type="ellipsoid" pos="0.00 0.06 0.22" size="0.026 0.026 0.012"
            rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom name="flower4" type="ellipsoid" pos="-0.03 0.05 0.31" size="0.028 0.028 0.013"
            rgba="0.70 0.35 0.85 1" contype="0" conaffinity="0"/>
      <geom name="fcenter1" type="sphere" pos="-0.05 -0.04 0.258" size="0.009"
            rgba="0.45 0.30 0.10 1" contype="0" conaffinity="0"/>
      <geom name="fcenter2" type="sphere" pos="0.05 0.03 0.297" size="0.009"
            rgba="0.45 0.30 0.10 1" contype="0" conaffinity="0"/>
      <geom name="fcenter3" type="sphere" pos="0.00 0.06 0.227" size="0.008"
            rgba="0.45 0.30 0.10 1" contype="0" conaffinity="0"/>
      <geom name="fcenter4" type="sphere" pos="-0.03 0.05 0.317" size="0.009"
            rgba="0.45 0.30 0.10 1" contype="0" conaffinity="0"/>
    </body>
    <body name="bee" mocap="true" pos="-0.30 -0.30 0.45">
      <geom name="bee_body" type="ellipsoid" size="0.017 0.011 0.011" rgba="0.98 0.80 0.12 1"
            contype="0" conaffinity="0"/>
      <geom name="bee_stripe1" type="ellipsoid" pos="0.005 0 0" size="0.0035 0.0113 0.0113"
            rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
      <geom name="bee_stripe2" type="ellipsoid" pos="-0.004 0 0" size="0.0035 0.0113 0.0113"
            rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
      <geom name="bee_head" type="sphere" pos="0.019 0 0.002" size="0.006"
            rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
      <geom name="bee_wing_l" type="ellipsoid" pos="0.002 0.018 0.012" size="0.019 0.011 0.0008"
            euler="35 0 20" rgba="0.95 0.95 1.0 0.95" contype="0" conaffinity="0"/>
      <geom name="bee_wing_r" type="ellipsoid" pos="0.002 -0.018 0.012" size="0.019 0.011 0.0008"
            euler="-35 0 -20" rgba="0.95 0.95 1.0 0.95" contype="0" conaffinity="0"/>
    </body>
'''


def scene_xml(kind: str, eye_size: int = 256, meta_path: Optional[str] = None,
              art_dir: Optional[str] = None, body_path: Optional[str] = None,
              mobile: bool = False, critters: bool = False) -> str:
    """Return the XML text for one of our scenes (v2 body via absolute includes)."""
    if kind not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}")
    A = mimo_assets_dir()
    META = (meta_path or f"{A}/mimo/MIMo_metav2.xml").replace("\\", "/")
    BODY = (body_path or f"{A}/mimo/MIMo_modelv2.xml").replace("\\", "/")
    off = max(480, eye_size)   # watch_cam videos render at 480
    # gain-of-function props (plan section 9, "room + 3 objects"): a fixed shelf at hand height
    # with three SAME-SIZE boxes of very different weight (0.05 / 0.4 / 2.0 kg), colours
    # white / gray / dark red so vision can tell them apart. Every scene gets a "watch_cam"
    # that follows MIMo for the periodic videos.
    props = '''
    <body name="shelf" pos="0.22 -0.05 0.40">
      <geom name="shelf_geom" type="box" size="0.12 0.16 0.01" rgba="0.45 0.3 0.2 1" condim="3"/>
    </body>
    <body name="box_light" pos="0.20 0.06 0.44">
      <freejoint/><geom name="box_light_geom" type="box" size="0.03 0.03 0.03" mass="0.05" rgba="0.95 0.95 0.95 1" condim="6"/>
    </body>
    <body name="box_medium" pos="0.24 -0.05 0.44">
      <freejoint/><geom name="box_medium_geom" type="box" size="0.03 0.03 0.03" mass="0.4" rgba="0.5 0.5 0.5 1" condim="6"/>
    </body>
    <body name="box_heavy" pos="0.20 -0.16 0.44">
      <freejoint/><geom name="box_heavy_geom" type="box" size="0.03 0.03 0.03" mass="2.0" rgba="0.45 0.1 0.1 1" condim="6"/>
    </body>
    <camera name="watch_cam" mode="targetbody" target="upper_body" pos="0.9 -0.9 0.9"/>'''
    mimo_euler = "0 0 0"
    mimo_quat = None
    if kind == "reach":
        gravity = "0 0 -9.81"
        equality = '<equality><weld body1="upper_body"/></equality>'
        mimo_pos = "0 0 0.6"
        extra = props + '''
    <body name="target" mocap="true" pos="0.25 -0.15 0.55">
      <geom name="target_geom" type="sphere" material="matgeom" size="0.025" condim="6" rgba="1 0.2 0.2 1"/>
    </body>'''
    elif kind == "selfbody":
        gravity = "0 0 -9.81"
        equality = '<equality><weld body1="mimo_location"/></equality>'
        mimo_pos = "0 0 0.2"
        extra = props
    elif kind == "crib":
        # a newborn's world: lying on its back on the floor, free body (it CAN roll over),
        # the ball hangs above its face like a mobile, boxes beside it on the floor
        gravity = "0 0 -9.81"
        equality = ""
        mimo_pos = "0 0 0.1"
        # Supine, taken from MIMo's own roll_over.py rather than guessed. Ours was
        # euler="90 0 0" with a comment claiming face-up; measured, the eyes ended up
        # pitched -48 degrees, looking at the FLOOR. That put the mobile hanging at 45 cm
        # "above its face" behind its head, and is a large part of why the ball sat 60-120
        # degrees off the eye axis in every run. MuJoCo body quat is w x y z.
        mimo_euler = None
        mimo_quat = "0 0.7071068 0 0.7071068"
        extra = '''
    <body name="target" mocap="true" pos="0.05 0.0 0.45">
      <geom name="target_geom" type="sphere" material="matgeom" size="0.025" condim="6" rgba="1 0.2 0.2 1"/>
    </body>
    <body name="box_light" pos="0.05 0.30 0.04">
      <freejoint/><geom name="box_light_geom" type="box" size="0.03 0.03 0.03" mass="0.05" rgba="0.95 0.95 0.95 1" condim="6"/>
    </body>
    <body name="box_medium" pos="0.15 0.30 0.04">
      <freejoint/><geom name="box_medium_geom" type="box" size="0.03 0.03 0.03" mass="0.4" rgba="0.5 0.5 0.5 1" condim="6"/>
    </body>
    <body name="box_heavy" pos="0.25 0.30 0.04">
      <freejoint/><geom name="box_heavy_geom" type="box" size="0.03 0.03 0.03" mass="2.0" rgba="0.45 0.1 0.1 1" condim="6"/>
    </body>
    <geom name="pen_wall_n" type="box" pos="0 0.7 0.12" size="0.72 0.02 0.12" rgba="0.93 0.90 0.84 1" condim="3"/>
    <geom name="pen_wall_s" type="box" pos="0 -0.7 0.12" size="0.72 0.02 0.12" rgba="0.93 0.90 0.84 1" condim="3"/>
    <geom name="pen_wall_e" type="box" pos="0.7 0 0.12" size="0.02 0.72 0.12" rgba="0.93 0.90 0.84 1" condim="3"/>
    <geom name="pen_wall_w" type="box" pos="-0.7 0 0.12" size="0.02 0.72 0.12" rgba="0.93 0.90 0.84 1" condim="3"/>
{WALL_ART_GEOMS}
    <camera name="watch_cam" mode="targetbody" target="upper_body" pos="1.0 -1.0 0.9"/>'''
        if mobile:
            extra = MOBILE_XML + extra
        if critters:
            extra = extra + CRITTERS_XML
    else:  # floor
        gravity = "0 0 -9.81"
        equality = ""
        mimo_pos = "0 0 0.4"
        extra = props
    mimo_orient = (f'quat="{mimo_quat}"' if mimo_quat else f'euler="{mimo_euler}"')
    _xml = f'''<mujoco model="MIMo_bridge_{kind}">
  <compiler inertiafromgeom="true" angle="degree" meshdir="{A}/meshes" texturedir="{A}"/>
  <option gravity="{gravity}" timestep="0.005" iterations="50" tolerance="1e-10" solver="Newton" jacobian="dense" cone="elliptic" impratio="1.0"/>
  <size nconmax="1000" njmax="5000" nstack="10000000"/>
  <visual>
    <map force="0.1" zfar="30" znear="0.005"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
    <global offwidth="{off}" offheight="{off}"/>
  </visual>
  <default><geom material="matgeom" condim="3" solimp="0.95 0.99 0.001 0.5 1" solref="0.005 1"/></default>
  <asset>
{{PEN_ART_ASSETS}}
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    <material name="matgeom" texture="texgeom" texuniform="true" rgba="0.8 0.6 .4 1"/>
  </asset>
  <include file="{META}"/>
  {equality}
  <worldbody>
    <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="{{FLOORMAT}}" condim="3"/>
    <light directional="false" diffuse=".4 .4 .4" specular="0 0 0" pos="0 0 10" dir="0 0 -1" castshadow="false"/>
    <light mode="targetbodycom" target="upper_body" directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 5.0" dir="0 0 -1"/>
    <body name="mimo_location" pos="{mimo_pos}" {mimo_orient}>
      <freejoint name="mimo_orientation"/>
      <include file="{BODY}"/>
    </body>{extra}
  </worldbody>
</mujoco>
'''
    # Playpen art. The walls carry letters, numbers, shapes and living things; the floor
    # carries a patterned play mat. A blank surface is perfectly predictable and teaches
    # the eyes nothing, and a blank FLOOR is worse than a blank wall - the floor is what
    # slides when the baby moves, so without pattern there is no optical flow to learn
    # self-motion from. The art is drawn bold: the eyes are 256x256 per eye, chopped into
    # 16-pixel patches, so fine detail would not survive.
    #
    # The images ride on visual-only PLANE geoms inset just inside each wall rather than
    # on the wall boxes themselves: MuJoCo maps a 2D texture cleanly onto a plane (this is
    # how the floor works) but not onto a box face, where it renders black or as a single
    # stretched crop. The boxes keep doing the physics; the planes only draw.
    assets, art_geoms, floormat = "", "", "matplane"
    if art_dir:
        try:
            fp = make_floor_texture(art_dir).replace("\\", "/")
            assets += (f'    <texture name="playmat_tex" type="2d" file="{fp}"/>\n'
                       f'    <material name="playmat" texture="playmat_tex" texrepeat="4 4"'
                       f' texuniform="true" reflectance="0.05" specular="0.05" shininess="0.05"/>\n')
            floormat = "playmat"
            # the ceiling: a supine baby looks straight UP, so this is most of its
            # world. review E20: only the crib template carries {WALL_ART_GEOMS};
            # building the geom for other scenes silently dropped it in the replace.
            if kind == "crib":
                cp = make_ceiling_texture(art_dir).replace("\\", "/")
                assets += (f'    <texture name="ceil_tex" type="2d" file="{cp}"/>\n'
                           f'    <material name="ceilmat" texture="ceil_tex" texrepeat="1 1"'
                           f' emission="0.75" specular="0.03" shininess="0.03"/>\n')
                art_geoms += ('    <geom name="nursery_ceiling" type="plane" pos="0 0 1.05" '
                              'zaxis="0 0 -1" size="2.2 2.2 0.01" material="ceilmat" '
                              'contype="0" conaffinity="0" group="0"/>\n')
        except Exception as ex:
            logging.getLogger("mimo_bridge.env").warning("floor art unavailable: %s", ex)
    if kind == "crib" and art_dir:
        try:
            paths = make_pen_textures(art_dir)
            #      wall: (position just inside the box, orientation, half-extent)
            # zaxis points the plane's face INTO the pen; deriving euler angles by hand
            # put two panels upright in the middle of the floor
            place = {"n": ("0 0.677 0.12", "0 -1 0", "0.72 0.12 0.01"),
                     "s": ("0 -0.677 0.12", "0 1 0", "0.72 0.12 0.01"),
                     "e": ("0.677 0 0.12", "-1 0 0", "0.12 0.72 0.01"),
                     "w": ("-0.677 0 0.12", "1 0 0", "0.12 0.72 0.01")}
            rows = []
            for k in ("n", "s", "e", "w"):
                f = paths[k].replace("\\", "/")
                rows.append(f'    <texture name="pentex_{k}" type="2d" file="{f}"/>')
                rows.append(f'    <material name="penmat_{k}" texture="pentex_{k}" '
                            f'texrepeat="1 1" emission="0.35" specular="0.05" shininess="0.05"/>')
                pos, zax, size = place[k]
                art_geoms += (f'    <geom name="pen_art_{k}" type="plane" pos="{pos}" zaxis="{zax}" '
                              f'size="{size}" material="penmat_{k}" contype="0" conaffinity="0" '
                              f'group="0"/>\n')
            assets += "\n".join(rows)
        except Exception as ex:                      # art is a nicety, never a blocker
            logging.getLogger("mimo_bridge.env").warning("pen art unavailable: %s", ex)
    return (_xml.replace("{PEN_ART_ASSETS}", assets).replace("{FLOORMAT}", floormat)
                .replace("{WALL_ART_GEOMS}", art_geoms))



def write_scene(kind: str, out_dir: str, eye_size: int = 256, tag: str = "", lite_textures: bool = True,
                fovea_fov: float = 0.0, mobile: bool = False, ring_fov: float = 0.0,
                critters: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    meta = lite_meta_xml(out_dir) if lite_textures else None
    body = fovea_model_xml(out_dir, fovea_fov, ring_fov) if (fovea_fov > 0 or ring_fov > 0) else None
    path = os.path.join(out_dir, f"mimo_bridge_{kind}{tag}.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(scene_xml(kind, eye_size, meta, art_dir=os.path.join(out_dir, "pen_art"), body_path=body,
                          mobile=mobile, critters=critters))
    return path


# ---------------------------------------------------------------------------- growth (age)
class _DirLock:
    """Cross-process lock made of an atomically created directory (works on Windows)."""

    def __init__(self, path: str, timeout_s: float = 600.0):
        self.path, self.timeout = path, timeout_s

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                os.mkdir(self.path)
                return self
            except FileExistsError:
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"growth lock {self.path} held too long")
                time.sleep(0.05)

    def __exit__(self, *a):
        try:
            os.rmdir(self.path)
        except OSError:
            pass


def grow_scene(scene_path: str, age_months: float, out_path: str) -> str:
    """Make a copy of scene_path with MIMo grown to age_months (0..24), saved at out_path.

    Copy the XML include tree into this run before calling MIMo's growth utility.
    That utility writes temporary siblings of its inputs. Keeping those inputs
    local prevents concurrent workers from touching another checkout's models.
    """
    from mimo_infant.growth.growth import adjust_mimo_to_age
    import tempfile
    from pathlib import Path
    import xml.etree.ElementTree as ET
    destination=Path(out_path).resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='growth_',dir=destination.parent) as folder:
        folder=Path(folder);copied={}
        def local_copy(path):
            path=Path(path).resolve()
            if path in copied:return copied[path]
            target=folder/(str(len(copied))+'_'+path.name)
            copied[path]=target
            tree=ET.parse(path)
            for include in tree.getroot().iter('include'):
                source=Path(include.attrib['file'])
                if not source.is_absolute():source=path.parent/source
                include.set('file',str(local_copy(source)).replace('\\','/'))
            tree.write(target,encoding='utf-8',xml_declaration=True)
            return target
        local_scene=local_copy(scene_path)
        tmp=adjust_mimo_to_age(float(age_months),str(local_scene),'v2',None,create_log=False)
        # Only a newly generated file in our own temporary directory can move.
        if not Path(tmp).resolve().is_relative_to(folder.resolve()):
            raise ValueError('growth output escaped its private directory')
        shutil.move(tmp,destination)
    return out_path


# ---------------------------------------------------------------------------- eye renderer
_GL_CONTEXT = None     # one OpenGL context per worker process, kept alive across env rebuilds


def _gl_context(width: int, height: int):
    """The process-wide GL context. mujoco.Renderer makes a brand-new GLFW window + context
    every time it is built; after a few body rebuilds in 20 worker processes the Intel iGPU
    driver ran out ("Shadow framebuffer is not complete, error 0x8cd6", 8-hour run, 12:08).
    Creating the context once per process and only rebuilding the MuJoCo render context per
    model avoids that."""
    global _GL_CONTEXT
    import mujoco
    if _GL_CONTEXT is None:
        _GL_CONTEXT = mujoco.GLContext(max(width, 64), max(height, 64))
    return _GL_CONTEXT


class EyeRenderer:
    """Renders a named camera to RGB, same calls as mujoco.Renderer but on the shared context."""

    def __init__(self, model, height: int, width: int, max_geom: int = 10000):
        import mujoco
        self._mj = mujoco
        if width > model.vis.global_.offwidth or height > model.vis.global_.offheight:
            raise ValueError("eye size larger than the scene's offscreen buffer (visual/global offwidth/offheight)")
        self.model = model
        self.width, self.height = width, height
        self.gl = _gl_context(width, height)
        self.gl.make_current()
        self.scene = mujoco.MjvScene(model, maxgeom=max_geom)
        self.option = mujoco.MjvOption()
        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.ctx)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._buf = np.empty((height, width, 3), dtype=np.uint8)
        self._no_fx = False   # set True to render without shadows/reflections (watch cam videos)

    def render(self, data, camera_name: str) -> np.ndarray:
        mj = self._mj
        self.gl.make_current()
        self.cam.fixedcamid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA.value, camera_name)
        mj.mjv_updateScene(self.model, data, self.option, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        if self._no_fx:
            # shadows and floor reflections produced ghost copies and dark rectangles in the
            # watch videos (several render contexts share one GL context); plain shading is clean
            self.scene.flags[mj.mjtRndFlag.mjRND_SHADOW.value] = 0
            self.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION.value] = 0
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self.ctx)   # re-bind: other renderers share this GL context
        mj.mjr_render(self.rect, self.scene, self.ctx)
        mj.mjr_readPixels(self._buf, None, self.rect, self.ctx)
        return np.flipud(self._buf).copy()          # mjr_readPixels gives bottom-up rows

    def close(self):
        try:
            self.ctx.free()
        except Exception:
            pass


# ---------------------------------------------------------------------------- MIMoEnv subclass
def _make_env_class():
    """Built lazily so importing this module does not import MIMo (tests without MIMo)."""
    from mimo_infant.simulation.envs.mimo_env import MIMoEnv

    class BridgeMIMoEnv(MIMoEnv):
        """MIMoEnv with dummy goal methods (like MIMoDummyEnv) and the delay workaround."""

        def __init__(self, *a, **k):
            # MIMo bug workaround: with sensory_delay > 0 the obs history list is created
            # AFTER the first _get_obs call, so construction fails. Create it first.
            self._obs_history = []
            self._action_history = []
            super().__init__(*a, **k)

        def reset_model(self):
            self.set_state(self.init_qpos, self.init_qvel)
            return self._get_obs()

        def _single_mujoco_step(self):
            observer=getattr(self,'_bridge_contact_observer',None)
            before=self.data.qvel.copy() if observer is not None else None
            super()._single_mujoco_step()
            if observer is not None: observer(before)

        def sample_goal(self):
            return np.zeros(1)

        def get_achieved_goal(self):
            return np.zeros(1)

        def is_success(self, achieved_goal, desired_goal):
            return False

        def is_failure(self, achieved_goal, desired_goal):
            return False

        def is_truncated(self):
            return False

        def compute_reward(self, achieved_goal, desired_goal, info):
            return 0.0

    return BridgeMIMoEnv


# ---------------------------------------------------------------------------- step result
@dataclass
class RawStep:
    frame: int
    sim_time: float
    dt: float
    patches: np.ndarray                 # (n_patches, 10)
    collisions: List[dict]
    proprio: ProprioReadout
    # (VESTIBULAR_DIM,) float32 in vestibular_adapter.VESTIBULAR_FIELDS order: head
    # accelerometer and gyro, neck angles/speeds/efforts, trunk up-direction, what carries
    # the body's weight, effort, joint-limit load, fatigue capacity, validity. Until
    # 2026-09-22 this was MIMo's raw six numbers (accelerometer m/s2, gyro rad/s), which a
    # payload saved before then still carries: vestibular_adapter.pad_legacy_vestibular
    # widens it.
    vestibular: np.ndarray
    eye_left: Optional[np.ndarray]      # (H, W, 3) uint8 or None when not rendered this step
    eye_right: Optional[np.ndarray]
    eye_angles: Tuple[Tuple[float, float], Tuple[float, float]]   # (yaw, pitch) per eye, rad
    gaze_az_el_deg: Tuple[float, float]
    n_contacts: int
    ctrl: np.ndarray
    info: dict = field(default_factory=dict)
    # root-relative positions of the skin bodies, one row per tactile patch, metres.
    # This is the geometry the touch tokens carry: Perception_Space.pdf section 1 says
    # "touch dimensions are structured by the body surface geometry", and until
    # 2026-08-27 the pipeline threw that structure away - 256 anonymous slots, where a
    # contact shifted by ONE patch moved the token as far as teleporting it to a random
    # body site (measured ratio 1.10).
    touch_pos: Optional[np.ndarray] = None     # (n_patches, 3) float32, root-relative m
    # per patch: [share of this step's force that came from the body's OWN parts,
    # how many own parts are on it over 4]. Both 0 for the floor, the toy and the adult.
    touch_self: Optional[np.ndarray] = None    # (n_patches, 2) float32
    # per patch, from its strongest contact: the direction the push came from (3, unit,
    # pointing into the body), how deep it is (cm/10) and how fast it arrived (cm/s / 500).
    # This is what the collision entries carried and nothing read.
    touch_contact: Optional[np.ndarray] = None  # (n_patches, 5) float32
    touch_sensitivity: Optional[np.ndarray] = None  # pain and pleasant-touch tissue gains
    fovea_left: Optional[np.ndarray] = None    # narrow-field pair, None when no fovea / not rendered
    fovea_right: Optional[np.ndarray] = None


def _is_finger_joint(name: str) -> bool:
    """MIMo's finger and thumb hinges: robot:<side>_{ff,mf,rf,lf,th}_<part>."""
    return any(f"_{digit}_" in name for digit in ("ff", "mf", "rf", "lf", "th"))


# ---------------------------------------------------------------------------- contact report
# A CONTACT ONSET (DESIGN.md section 8): a pair of bodies that was apart for at least this many
# env frames (10 ms each, so 5 = 50 ms = one model step) and is touching now. Shorter gaps are
# the same contact chattering on and off, not a new touch. The contract fixes 5; the measured
# gap distribution it has to separate is in docs/gain_of_function_v3/agents/env-senses.md
# ("Re-contact gaps").
ONSET_ABSENT_FRAMES = 5
# "In view": the contact point within this many degrees of either eye's axis. The env's own
# line: hand_in_view in _collect uses the same 30 degrees (half of the 60 degree wide field).
IN_VIEW_DEG = 30.0
# SELF-CONTACT THAT IS POSTURE, NOT ACTION: body pairs that touch whenever the body lies at rest
# under zero command, WHATEVER pose it came to rest in. An onset of one of these pairs is marked
# structural=True so a success measure can leave it out (contact_head.py skips it; train_mimo's
# self_contact_events.structural_share counts it). Nothing else reads this set.
#
# THE RULE (review env-senses MINOR 5, 2026-09-23): a pair is structural when it touched in EVERY
# one of 7 measured rests. MEASURED 2026-09-23 (fix-env, scratchpad fixenv/measure_press_rest.py,
# the H84 segment 31 world from kw_seg31.json: age 24 months, mechanical_v4, body_function
# modulate, trunk_v1, finger armature 0.0004, caregiver responsive, mobile and critters on; vision
# off, 1 body, CPU): rest 1 = 30 s of zero command after a reset; rests 2 to 7 = 3 s of random
# control, then 7 s of zero command, the body resting in whatever pose the movement left it in.
# Counted over each rest's last 5 s (500 frames, after the body settled):
#   left_hand|left_thhub, right_hand|right_thhub   touching in 7 of 7 rests (62-100% of the frames)
#   left and right upper arm on the upper body     2 of 7 (a pose, not the hand's own shape)
#   every other pair                               0 of 7
# So the thumb hub resting in its own palm is the only self-touch the relaxed body always has.
# The earlier list (2026-09-22: every pair seen in the first 10 s after one reset, no cut-off)
# also held four right-hand finger pairs, so a right index tip touching the middle finger - the
# body ladder's finger-to-finger rung - was never scored on the right hand and always on the
# left. Measured here, those pairs touch only while the hand settles after that one reset (right
# index on middle finger 12% of the reset rest after its first 2 s, never in the other 6 rests),
# the same as a left-hand finger pair does (left knuckle-on-thumb onsets happened at zero command
# too). MIMo gives the index and middle fingers the same collision bits on both hands
# (MIMo_modelv2.xml lines 94-123 and 222-251); the difference was the reset pose, not the physics.
# Under movement (random control and babble, 1,099 self-touch onsets) the new rule marks 118
# structural against 136 under the earlier list: 18 finger-pair onsets are now scored.
STRUCTURAL_MEASUREMENT: dict = {
    "date": "2026-09-23",
    "source": ("fix-env scratchpad fixenv/measure_press_rest.py -> press_rest.json; H84 seg31 world, "
               "zero command, 7 rests (one after a reset, six after 3 s of random control), each "
               "rest's last 500 env frames"),
    "rule": "structural = touching on at least one settled frame in every one of the 7 rests",
    "rests_touching_of_7": {
        ("left_hand", "left_thhub"): 7,
        ("right_hand", "right_thhub"): 7,
        ("right_upper_arm", "upper_body"): 2,
        ("left_upper_arm", "upper_body"): 2,
        ("right_ffdistal", "right_mfmiddle"): 0,
        ("right_ffknuckle", "right_thdistal"): 0,
        ("right_lfmiddle", "right_rfknuckle"): 0,
        ("right_lfknuckle", "right_rfknuckle"): 0,
        ("right_ffdistal", "right_mfdistal"): 0,
    },
    # share of each rest's settled frames touching (rest 1 = after the reset, then rests 2-7)
    "share_of_settled_frames": {
        ("left_hand", "left_thhub"): (1.0, 0.702, 1.0, 1.0, 1.0, 0.898, 1.0),
        ("right_hand", "right_thhub"): (1.0, 1.0, 1.0, 1.0, 1.0, 0.624, 1.0),
        ("right_upper_arm", "upper_body"): (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ("left_upper_arm", "upper_body"): (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.364),
    },
    # the list this replaced: share of the first 1,000 frames after one reset (env-senses agent)
    "earlier_2026_09_22_share_of_first_10_s": {
        ("left_hand", "left_thhub"): 0.841,
        ("right_hand", "right_thhub"): 0.796,
        ("right_ffdistal", "right_mfmiddle"): 0.336,
        ("right_lfmiddle", "right_rfknuckle"): 0.005,
        ("right_lfknuckle", "right_rfknuckle"): 0.002,
        ("right_ffdistal", "right_mfdistal"): 0.001,
    },
}
STRUCTURAL_SELF_PAIRS: frozenset = frozenset(
    pair for pair, rests in STRUCTURAL_MEASUREMENT["rests_touching_of_7"].items() if rests == 7)


HAND_PRESS_LINE_KPA = 2.0     # the old hand_pressing line (2026-08-24), kept so all three read alike
# THE WORLD-PRESS ONSET (review env-senses MINOR 4). hand_pressing_world is a STATE, and it is
# mostly a limp hand lying against something. MEASURED 2026-09-23 (fix-env, scratchpad
# fixenv/measure_press_rest.py; the H84 segment 31 world from kw_seg31.json, 1 body, CPU, vision
# off, caregiver recording off), the outside side's own footprint pressure over 2 kPa:
#   at rest (zero command; the 30 s after a reset less its first 2 s, and 6 rests of 7 s after
#   random movement, 5,800 frames): 42 of every 100 frames (the reset rest alone 72 of 100, the
#   left hand against the light box); the pressure's median 0.9 kPa, 99th percentile 88 kPa;
#   random control (uniform -0.5..0.5 held 5 frames, 4,800 frames): 9 of 100, 99th pct 174 kPa;
#   the trainer's motor babble through the decoder (3,000 frames): 6 of 100, 99th pct 13 kPa.
# A line cannot separate them: a line above the still body's 99th percentile (88 kPa) would
# leave out nearly every press the babbling body makes (its 99th percentile is 13 kPa).
# Onsets per 10 s of body time (1,000 frames, both hands), by how long the hand must have
# been at or below the line first:
#   5 frames (the contact onset's rule):   rest 7.8 (45 onsets), random 20.0 (96), babble 6.3 (19)
#   25 frames (one order's tenure on H84): rest 1.6 (9),  random 11.5 (55), babble 4.0 (12)
# With 5 frames a still body starts as many presses as the babbling one (a hand on a box chatters
# across the line); with a whole tenure below, babble starts 2.5 times and random control 7 times
# as many as rest. Raising the line did not separate them further (25 frames at 5, 10 and 20 kPa:
# 7, 5 and 5 onsets in the 5,800 rest frames, 11, 10 and 8 in the 3,000 babble frames), so the
# line stays the old 2 kPa and the tenure rule is the second key (hand_press_world_onsets_tenure).
WORLD_PRESS_ONSET_LINE_KPA = HAND_PRESS_LINE_KPA


def split_hand_press(pressure_pa, own_share) -> dict:
    """The hand press split by WHAT is pressed (DESIGN.md D9 and section 8).

    hand_press_kpa / hand_pressing fire on the body pressing ITSELF: in a CPU probe of
    2026-09-22 (docs/gain_of_function_v3/agents/understand-touch-selfcontact.md), 399 of 414
    pressing steps under random commands were pure self-contact, the thumb in its own palm
    most of all. Here each hand patch's pressure (Pa) is divided by the share of that patch's
    contact force that came from the body's own parts (0 = all from outside, 1 = all from the
    body itself; the touch pooler's last_self column 0), so a patch pressed half by the floor
    and half by a finger gives half its pressure to each. Same 2 kPa line as the old key.
    Returns hand_press_world_kpa, hand_pressing_world, hand_press_self_kpa, hand_pressing_self."""
    p = np.asarray(pressure_pa, dtype=np.float64).reshape(-1)
    s = np.clip(np.asarray(own_share, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if p.size == 0:
        return {}
    world = float((p * (1.0 - s)).max() / 1000.0)
    own = float((p * s).max() / 1000.0)
    return dict(hand_press_world_kpa=world, hand_pressing_world=bool(world > HAND_PRESS_LINE_KPA),
                hand_press_self_kpa=own, hand_pressing_self=bool(own > HAND_PRESS_LINE_KPA))


HAND_PATCH_KEYS = ("hand", "distal", "middle", "knuckle", "metacarpal", "thbase", "thhub")


def footprint_hand_press(records, hand_side: Dict[int, str], own_body_ids, geom_bodyid) -> dict:
    """Each side's OWN pressure on each hand patch, from the virtual skin's contact records.

    split_hand_press gives each side the patch's pressure times that side's share of the force.
    The patch's pressure is ALL the force over ALL the loaded area, so each side's force is
    spread over the other side's footprint too (review env-senses MINOR 3: on hand patches
    pressed by both at once, the self side read a median 19% of its own pressure in the first
    6 s of rest after a reset; reproduced by fix-env on 2026-09-23, 90 frames). Here
    each side's pressure is that side's force over that side's own footprint area, per patch:
    the records (contact_footprint.aggregate_contact_footprints: one per body and shape pair,
    carrying its normal force in N and its footprint area in m2) are grouped by the patch they
    press and by whether the other shape's body is the infant's own (the same body tree test
    as ContactInfo.other_is_self).

    records: VirtualTaxelSkin.contact_records. hand_side: {body id of a hand patch: "left" |
    "right"}. Returns {hand: (world kPa, self kPa)}, the largest over that hand's patches, 0.0
    when nothing on that side presses the hand."""
    per: Dict[tuple, list] = {}
    for r in records or ():
        b = int(r["body_id"])
        if b not in hand_side:
            continue
        area = float(r.get("area_m2") or 0.0)
        force = float(r.get("normal_force_n") or 0.0)
        if area <= 0.0 or force <= 0.0:
            continue
        g1, g2 = (int(g) for g in r["pair"])
        other = int(geom_bodyid[g2]) if int(geom_bodyid[g1]) == b else int(geom_bodyid[g1])
        side = "self" if other in own_body_ids else "world"
        acc = per.setdefault((b, side), [0.0, 0.0])
        acc[0] += force
        acc[1] += area
    out = {}
    for hand in ("left", "right"):
        vals = {"world": 0.0, "self": 0.0}
        for (b, side), (f, a) in per.items():
            if hand_side[b] == hand:
                vals[side] = max(vals[side], f / a / 1000.0)
        out[hand] = (vals["world"], vals["self"])
    return out


def press_onsets(below: Optional[Dict[str, int]], world_kpa: Dict[str, float], line_kpa: float,
                 quiet_frames: Sequence[int] = (ONSET_ABSENT_FRAMES,)) -> Tuple[Dict[str, int], Tuple[int, ...]]:
    """One frame of the world-press onset tracker. An ONSET is a hand whose outside-side
    pressure rises above `line_kpa` after at least q frames at or below it, counted once for
    each q in `quiet_frames` (ONSET_ABSENT_FRAMES = 5 is the contact onset's rule). below:
    {hand: frames in a row at or below the line} or None when tracking has not started (the
    first frame after a reset, or after restoring a snapshot saved without it); that first frame
    reports no onset, because how long a hand had been below the line before it is unknown.
    Returns (the new tracker, onsets per entry of quiet_frames)."""
    starting = below is None
    new = {} if starting else dict(below)
    counts = [0] * len(quiet_frames)
    for hand, kpa in world_kpa.items():
        prev = new.get(hand)
        if kpa > line_kpa:
            if not starting and prev is not None:
                for i, q in enumerate(quiet_frames):
                    if prev >= q:
                        counts[i] += 1
            new[hand] = 0
        else:
            new[hand] = (prev or 0) + 1
    return new, tuple(counts)


def contact_region(body_name: str) -> str:
    """The skin region a body belongs to (touch_adapter.region_of); the eyes, which have no
    skin patch, belong to the head."""
    region = region_of(body_name)
    if not region and body_name.endswith("_eye"):
        return "head"
    return region


def merge_burst_contact_report(tasks: Sequence[dict]) -> dict:
    """Combine the contact-event fields of ONE model step's env frames, oldest first.

    The trainer steps each world action_repeat (5) env frames per model step and gives the
    model the LAST frame's task dict (train_mimo.py, the burst merge after src.step). An onset
    is an event on one env frame, so an onset on frames 1 to 4 of the five would never be
    seen. This returns the keys to lay over the last frame's dict:
      self_contact_onsets        every frame's onsets, concatenated in frame order
      contact_onsets_world       summed over the frames
      contact_patch_state_burst  per skin patch, the union over the frames (bit 1 world,
                                 bit 2 self), beside the last frame's contact_patch_state
      hand_press_world_onsets, hand_press_world_onsets_tenure
                                 summed over the frames (added 2026-09-23, fix-env)
    Keys absent from every frame are absent from the result (absent = not measured)."""
    out: dict = {}
    onsets: list = []
    world = 0
    seen = False
    state = None
    press_keys = ("hand_press_world_onsets", "hand_press_world_onsets_tenure")
    press_sums: dict = {}
    for t in tasks:
        if not t:
            continue
        if "self_contact_onsets" in t:
            onsets.extend(t.get("self_contact_onsets") or [])
            seen = True
        if "contact_onsets_world" in t:
            world += int(t.get("contact_onsets_world") or 0)
            seen = True
        for k in press_keys:
            if k in t:
                press_sums[k] = press_sums.get(k, 0) + int(t.get(k) or 0)
        s = t.get("contact_patch_state")
        if s is not None:
            a = np.asarray(s, dtype=np.int64)
            state = a if state is None or state.shape != a.shape else (state | a)
    if seen:
        out["self_contact_onsets"] = onsets
        out["contact_onsets_world"] = world
    if state is not None:
        out["contact_patch_state_burst"] = state.tolist()
    out.update(press_sums)
    return out


# ---------------------------------------------------------------------------- the env
class BridgeEnv:
    """One MIMo body + all adapters. Not a gym env; step() takes the model's action."""

    def __init__(self, scene: str = "reach", run_dir: Optional[str] = None, age: Optional[float] = None,
                 sensory_delay: int = 0, motor_delay: int = 0, vision: bool = True, eye_size: int = 256,  # noqa: E501 - sensory_delay must stay 0; see the check in __init__
                 render_every: int = 1, touch_scale_mult: float = 1.0, seed: int = 0, tag: str = "",
                 kp: float = 8.0, kd: float = 0.3, vib_window: int = 100,
                 joint_damping_scale: float = 3.0, hinge_armature: float = 0.02,
                 fovea_size: int = 0, fovea_fov: float = 16.0,
                 caregiver_seconds: float = 0.0, gaze_reflex: bool = True,
                 spontaneous_saccades: bool = True, cmd_horizon: int = 5,
                 gaze_kp: float = 80.0, gaze_kd: float = 3.0,
                 mobile: bool = False, ring_fov: float = 0.0, ring_size: int = 128,
                 critters: bool = False, audio_real: bool = False,
                 caregiver_mode: str = "off", caregiver_speech_dir: str = "",
                 caregiver_replay: str = "", caregiver_record_dir: str = "", body_layout: str = 'legacy', skin_profile: str = "",
                 touch_geometry: str = "body", contact_profile: str = 'legacy',
                 body_function: str = 'off', caregiver_present_seconds: float = 60.,
                 caregiver_absent_seconds: float = 15., support_fixture: str = 'off',
                 toy_hand_off: str = 'off', finger_armature: float = 0.0004,
                 effort_power: str = 'no_gear', hand_target_mode: str = 'world',
                 parent_relative_targets: str = 'off', reach_projection: str = 'off',
                 position_rows: str = 'all',
                 contact_footprint_guard: str = 'off', rubbing_calibration: str = 'v4',
                 touch_inside_rays: str = 'process', finger_joint_servo: str = 'off'):
        import mujoco
        from mimo_infant.simulation.envs.mimo_env import (DEFAULT_TOUCH_PARAMS_V2, DEFAULT_PROPRIOCEPTION_PARAMS,
                                                           DEFAULT_VESTIBULAR_PARAMS)
        from .limb_target_control import PARENT_RELATIVE_CHOICES, REACH_PROJECTION_CHOICES
        from .action_decoder import FINGER_JOINT_SERVO_CHOICES
        self._mj = mujoco
        if hand_target_mode not in ('world', 'wrist_relative_v1'):
            raise ValueError('Unknown hand target mode')
        # --finger-joint-servo (action_decoder.py, THE FINGER COMMAND'S MEANING): 'off' = the decoder servos each finger
        # handle's pose, as always; 'on' = each finger hinge is servoed toward its own latched joint goal. A joint goal
        # exists only under wrist_relative_v1, so 'on' needs that mode.
        if finger_joint_servo not in FINGER_JOINT_SERVO_CHOICES:
            raise ValueError('finger_joint_servo must be one of %s, not %r'
                             % (FINGER_JOINT_SERVO_CHOICES, finger_joint_servo))
        if finger_joint_servo == 'on' and hand_target_mode != 'wrist_relative_v1':
            raise ValueError("finger_joint_servo 'on' servos the hand's joint goal, which exists only under "
                             "hand_target_mode wrist_relative_v1")
        self.finger_joint_servo = finger_joint_servo
        self.hand_target_mode = hand_target_mode
        self._hand_target_cache = None
        self.hand_target_control = None
        # the motor interface audit's calf and arm fixes (limb_target_control.py); 'off' = none
        if parent_relative_targets not in PARENT_RELATIVE_CHOICES:
            raise ValueError('Unknown parent-relative target mode')
        if reach_projection not in REACH_PROJECTION_CHOICES:
            raise ValueError('Unknown reach projection mode')
        self.parent_relative_targets = parent_relative_targets
        self.reach_projection = reach_projection
        self._limb_target_cache = None
        self.limb_target_control = None
        # which position rows the decoder solves (action_decoder.POSITION_ROW_CHOICES; 'all' = today's)
        from .action_decoder import POSITION_ROW_CHOICES
        if position_rows not in POSITION_ROW_CHOICES:
            raise ValueError('Unknown position rows setting')
        self.position_rows = position_rows
        if contact_profile not in ('legacy', 'mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
            raise ValueError('unknown contact profile')
        self.contact_profile = contact_profile
        # The footprint guard (contact_footprint.FootprintGuard; 'off' = the v1 footprints exactly).
        # Kept on the env, outside save_full_state: a resumed world runs the setting it is BUILT with
        # (its saved env_kwargs, or the launch's under --resume-env-settings launch), while the skin's
        # saved per-taxel memories continue unchanged.
        from .contact_footprint import FOOTPRINT_GUARDS
        if contact_footprint_guard not in FOOTPRINT_GUARDS:
            raise ValueError('unknown contact footprint guard %r' % (contact_footprint_guard,))
        if contact_footprint_guard != 'off' and contact_profile not in ('mechanical_v3', 'mechanical_v4'):
            raise ValueError('the contact footprint guard acts only on the per-contact footprints of '
                             'mechanical_v3 and mechanical_v4; under %s it would change nothing' % contact_profile)
        self.contact_footprint_guard = contact_footprint_guard
        self._footprint_guard = FOOTPRINT_GUARDS[contact_footprint_guard]
        # What the guard changed since this world was built (not saved: it restarts at each launch), and
        # how many of those changes broke the guard's rule (its stop rule, checked on every change).
        self._footprint_guard_counts = dict(rescored_flat_overlaps=0, pressure_ceiling_contacts=0,
                                            largest_unguarded_pressure_pa=0.0, rule_violations=0,
                                            last_violation=None)
        # The ordinary-contact rubbing recalibration (lem/skin_mechanics.RUBBING_CALIBRATIONS; 'v4' = today's
        # memories only). Like the guard it lives on the env, outside save_full_state: a resumed world
        # runs the setting it is BUILT with. The skin's second memory is saved with the skin.
        from lem.skin_mechanics import RUBBING_CALIBRATIONS
        if rubbing_calibration not in RUBBING_CALIBRATIONS:
            raise ValueError('unknown rubbing calibration %r' % (rubbing_calibration,))
        if rubbing_calibration != 'v4' and contact_profile not in ('mechanical_v2_local', 'mechanical_v3',
                                                                    'mechanical_v4'):
            raise ValueError('the rubbing calibration acts on the virtual skin of mechanical_v2_local, v3 and '
                             'v4; under %s it would change nothing' % contact_profile)
        self.rubbing_calibration = rubbing_calibration
        self._rubbing_report = None       # the calibration report's running sums (since this world was built)
        # Which directions MIMo's skin inside test uses while this world's touch is built (touch_inside_rays.py:
        # 'process' = the directions this process drew, today's behaviour; 'fixed_v1' = one fixed set, so the
        # world's live skin points no longer depend on the process). Lives on the env like the guard: MIMo's touch
        # is rebuilt at every launch and never saved.
        from .touch_inside_rays import CHOICES as _INSIDE_RAY_CHOICES
        if touch_inside_rays not in _INSIDE_RAY_CHOICES:
            raise ValueError('unknown touch inside rays %r' % (touch_inside_rays,))
        self.touch_inside_rays = touch_inside_rays
        if body_function not in ('off','observe','modulate'):
            raise ValueError('body_function must be off, observe or modulate')
        self.body_function_mode = body_function
        self.body_function = None
        self._virtual_skin = None
        self.scene_kind = scene
        self.age = age
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.vision = vision
        self.eye_size = eye_size
        self.fovea_size = int(fovea_size)
        self.fovea_fov = float(fovea_fov)
        self.render_every = max(1, int(render_every))
        run_dir = run_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                          "Saved", "mimo_bridge", "scenes")
        # a saved world names the development machine's folder (D:/UEMA_scratch/...); the scene files written here are
        # also opened by MuJoCo's own C code, which repo_paths' Python-level re-rooting does not reach
        from .repo_paths import localize
        run_dir = os.path.abspath(localize(run_dir))
        self.mobile = bool(mobile)
        self.ring_fov = float(ring_fov)
        # ring renders through the wide renderer: its size IS the eye size (the
        # ring_size arg is kept for compatibility but the shared context decides)
        self.ring_size = int(eye_size) if ring_fov > 0 else 0
        self.critters = bool(critters)
        base_path = write_scene(scene, run_dir, eye_size, tag=tag,
                                fovea_fov=self.fovea_fov if self.fovea_size > 0 else 0.0,
                                mobile=self.mobile, ring_fov=self.ring_fov,
                                critters=self.critters)
        if caregiver_mode != "off":
            if scene != "crib":
                raise ValueError("responsive caregiving currently requires the crib scene")
            from .caregiver import add_caregiver_to_scene
            add_caregiver_to_scene(base_path)
        if age is not None:
            grown = os.path.join(run_dir, f"mimo_bridge_{scene}{tag}_age{age:g}.xml")
            base_path = grow_scene(base_path, age, grown)
        if support_fixture not in ('off','rail_v1'):
            raise ValueError('Unknown support fixture')
        if support_fixture=='rail_v1':
            from .support_fixture import add_to_scene
            add_to_scene(base_path)
        self.scene_path = base_path

        touch_params = {"scales": {k: v * touch_scale_mult for k, v in DEFAULT_TOUCH_PARAMS_V2["scales"].items()},
                        "touch_function": "force_vector", "response_function": "spread_linear"}
        proprio_params = dict(DEFAULT_PROPRIOCEPTION_PARAMS)
        vest_params = {"sensors": list(DEFAULT_VESTIBULAR_PARAMS["sensors"])}
        EnvCls = _make_env_class()
        initial_qpos = (dict(_SITTING_V2) if scene == "selfbody"
                        else newborn_supine_qpos() if scene == "crib" else None)
        # motor_delay takes effect: MIMo applies it in _set_action, which runs inside
        # env.step. sensory_delay does NOT: MIMo applies it in _get_obs, whose return
        # value no sense path here consumes, so the encoders receive undelayed senses at
        # every setting. Measured 2026-09-16 with identical seed and identical control:
        # sensory_delay 4 against 0 leaves proprioception, touch and vestibular identical
        # on 12 of 12 steps, max |difference| 0.000e+00 on every field, while the same
        # probe with motor_delay 3 against 0 differs on 11 of 12 - so that zero is a real
        # zero, not a blind probe. Refuse the setting rather than accept one that does
        # nothing: a knob that silently no-ops is worse than no knob, and two curriculum
        # transitions used to differ only in it and rebuilt every body for no effect.
        if int(sensory_delay) != 0:
            raise ValueError(
                "sensory_delay reaches nothing in this bridge: MIMo applies it in "
                "_get_obs and _collect reads live MuJoCo sensors, so any value leaves "
                "the senses identical (measured: 4 against 0, max difference 0.000e+00 "
                "on 12 of 12 steps). Delivering a sensory delay means queueing the sense "
                "arrays in BridgeEnv.step; until then only 0 is honest. motor_delay does "
                "take effect and is unrestricted.")
        from .touch_inside_rays import inside_rays, check_reached
        with inside_rays(self.touch_inside_rays) as inside_use:   # MIMo's TrimeshTouch picks its skin points here
            self.env = EnvCls(model_path=base_path, initial_qpos=initial_qpos, frame_skip=2,
                              sensory_delay=sensory_delay, motor_delay=motor_delay,
                              proprio_params=proprio_params, touch_params=touch_params,
                              vision_params=None, vestibular_params=vest_params,
                              goals_in_observation=False, done_active=False, age=None)
        check_reached(inside_use, self.env.touch)
        self._inside_use = inside_use
        self.model, self.data = self.env.model, self.env.data
        self.dt = float(self.env.dt)
        # adapters
        self.touch = self.env.touch
        # Keep the physical interpolation footprint fixed when numerical sensor
        # spacing changes. Otherwise denser sampling alone raises pressure.
        self.touch.force_spread_scales = {b:float(s)/touch_scale_mult for b,s in self.touch.sensor_scales.items()}
        from types import MethodType
        from .contact_mechanics import conservative_spread_linear
        self.touch.response_function = MethodType(conservative_spread_linear,self.touch)
        specs = [PatchSpec(body_id=int(b), body_name=self.model.body(b).name,
                           n_points=int(self.touch.get_sensor_count(b)), scale_m=float(self.touch.sensor_scales[b]))
                 for b in self.touch.sensing_bodies()]
        self.pooler = TouchPooler(specs, self.dt, vib_window=vib_window)
        from lem.skin_mechanics import MechanicalExposure
        self.pooler.mechanical_exposure = MechanicalExposure(len(self.pooler.specs))
        self._mechanical_pooler = TouchPooler(specs,float(self.model.opt.timestep),vib_window=4)
        self._mechanical_sample_time = None
        # What the layout switch did in this world's build, and the skin layout it built (touch_inside_rays.py).
        # Said once per world when the switch is on, so a launch log shows every world's directions and layout.
        from .touch_inside_rays import skin_layout
        points, digest = skin_layout(self.touch, [s.body_id for s in self.pooler.specs])
        self.touch_inside_rays_report = dict(self._inside_use.report(), skin_points=points, skin_digest=digest)
        if self.touch_inside_rays != 'process':
            logging.getLogger("mimo_bridge.env").warning(
                "touch inside rays %s (world %s): %d inside tests ran with directions sha256 %s...; skin layout %d "
                "points, digest %s", self.touch_inside_rays, tag or "-", self._inside_use.calls,
                self.touch_inside_rays_report["directions_sha256"][:8], points, digest)
        if self.contact_profile in ('mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
            self.env._bridge_contact_observer = self._mechanical_substep
        self.body_hash = {int(b): bone_hash(self.model.body(b).name) for b in self.touch.sensing_bodies()}
        # for the touch point-cloud: patch row order = pooler.specs order
        self._touch_body_ids = np.array([sp.body_id for sp in self.pooler.specs], dtype=np.int64)
        # MuJoCo's own body tree answers "is this part of me": every part of the infant
        # shares one root body. Contacts with the floor, the adult, the toys and the
        # critters have a different root, so this is exact rather than name matching.
        try:
            own_root = int(self.model.body_rootid[int(self.model.body("hip").id)])
            self._own_body_ids = {int(b) for b in range(self.model.nbody)
                                  if int(self.model.body_rootid[b]) == own_root}
        except Exception:
            self._own_body_ids = set()
        # names by body id and the two eye cameras, for the contact report (_contact_report)
        self._body_names = [self.model.body(b).name for b in range(self.model.nbody)]
        self._eye_cam_ids = [int(self.model.camera(n).id) for n in ("eye_left", "eye_right")]
        # onset tracker: pair key -> last env frame the pair was touching; None = not started
        # (the first collect after a reset or an old-snapshot restore starts it and reports
        # no onsets, because how long a pair touching then had been touching is unknown)
        self._contact_last_seen = None
        self._contact_track_start = 0
        # world-press onset tracker (_hand_press_footprint): hand -> frames in a row at or below
        # WORLD_PRESS_ONSET_LINE_KPA; None = not started, restarted like the contact tracker
        self._press_below = None
        from mimo_bridge.touch_adapter import region_of, SKIN_REGIONS
        self._skin_region_names = [r for r, _ in SKIN_REGIONS]
        self._patch_region = np.array(
            [self._skin_region_names.index(region_of(sp.body_name))
             if region_of(sp.body_name) in self._skin_region_names else -1
             for sp in self.pooler.specs], dtype=np.int64)
        self._root_body_id = 1
        for cand in ("robot:hip", "hip", "robot:torso", "torso"):
            try:
                self._root_body_id = int(self.model.body(cand).id)
                break
            except Exception:
                continue
        self.body_layout = body_layout
        # effort_power (proprio_adapter.EFFORT_POWER_MODES): how the effort that drains the
        # fatigue model is computed. The default no_gear is every run's formula to date; with_gear
        # is mechanical power in watts. Measured on the segment 31 body in the fix-env log.
        self.effort_power = effort_power
        self.proprio = ProprioAdapter(self.model, self.data, body_layout=body_layout,
                                      effort_power=effort_power)
        # Physics-level smoothing (measured 2026-08-21). Our PD runs once per 10 ms env
        # step while physics substeps inside it, so velocity feedback acts DELAYED: raising
        # kd made the buzz worse (0.55 -> 0.71 direction reversals per step). Joint damping
        # and rotor/tendon armature are applied by the integrator every substep instead:
        # damping x3 + 0.02 armature on hinges (never the free root) took reversals from
        # 0.438 to 0.042 and jerk from 4.6 to 0.38 while keeping real motion, which then
        # allows kp 8 for authority without ringing.
        if joint_damping_scale != 1.0:
            self.model.dof_damping[:] = np.asarray(self.model.dof_damping) * joint_damping_scale
        if hinge_armature > 0.0:
            # NOT the eyes. The buzz this fixes was measured in the LIMBS, whose rotor
            # inertia is of the same order as the number added. An eye joint's native
            # armature is 0.0002, so adding 0.02 multiplies it by 101 while its actuator
            # stiffness is 0.00065 - the eyeball becomes almost immovable. Measured: with
            # the flat addition the eye reaches 0.69 degrees in 0.3 s against a 30 degree
            # command; excluded, it reaches 19.4. Every run from H10 (2026-08-21) to H17
            # had frozen eyes, which is why gaze went nowhere no matter what commanded it.
            #
            # NOT THE FINGERS EITHER - they get finger_armature below. The same flat 0.02
            # froze the fingers from 2026-08-21 to 2026-09-22 the way it froze the eyes: a
            # finger joint's native armature is 0.0001 and its link inertia 2e-8 to 4e-6
            # kg m2, so 0.02 made each finger about 200 times harder to turn, against motors
            # of 0.003 to 0.20 N m. Measured on the H84 crib body, full torque on the index
            # knuckle for one model step (0.05 s) moved it 0.2 degrees; through the servo a
            # finger reached 29% of a 20 degree target in one motor hold (0.25 s). On the
            # live body the finger joints did move, but as the swinging hand flung them:
            # command against the next move correlated at a median of 0.18 for the fingers
            # against 0.46 to 0.89 for the wrists and elbows.
            for j in range(self.model.njnt):
                if int(self.model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                    continue
                name = self.model.joint(j).name
                if "_eye_" in name or _is_finger_joint(name):
                    continue
                self.model.dof_armature[int(self.model.jnt_dofadr[j])] += hinge_armature
        if finger_armature > 0.0:
            # THE FINGERS' OWN ANTI-BUZZ AMOUNT, measured 2026-09-22 by sweeping the total
            # finger armature with the decoder's servo in the loop (right hand, strength
            # 0.5, a 20 degree target per finger chain), share of the target reached in one
            # motor hold (0.25 s) and velocity reversals per 10 ms step while holding:
            #   total 0.0001 (native):  1.05 reached, 0.176 reversals - buzzes
            #   total 0.0005 (this):    1.18 reached, 0.032 reversals
            #   total 0.0020:           0.91 reached, 0.021 reversals
            #   total 0.0201 (before):  0.29 reached, 0.006 reversals - cannot follow
            # 0.032 is under the 0.042 the limbs were accepted at on 2026-08-21.
            for j in range(self.model.njnt):
                if (int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                        and _is_finger_joint(self.model.joint(j).name)):
                    self.model.dof_armature[int(self.model.jnt_dofadr[j])] += finger_armature
        self._jit_prev = None
        self._jit_ring = []
        # Eye servo gains from the saccadic main sequence, measured by sweep: kp 2.0 gave
        # 27 deg/s peak and 700 ms to settle, so the eye never arrived before the next
        # saccade. kp 80 / kd 3.0 gives 110 deg/s, 130 ms, no overshoot and no ringing.
        # Peak speed saturates near 110 deg/s at any gain - MIMo's eye actuator is
        # force-limited, so the human 300-400 deg/s is out of reach without changing the
        # body. Small amplitudes carry the rest.
        self.decoder = ActionDecoder(self.model, kp=kp, kd=kd,
                                     body_layout=body_layout,
                                     ctrl_bounds=(self.env.action_space.low, self.env.action_space.high),
                                     gaze_kp=gaze_kp, gaze_kd=gaze_kd, position_rows=position_rows)
        if self.hand_target_mode == 'wrist_relative_v1':
            from .hand_target_control import HandTargetControl
            self.hand_target_control = HandTargetControl(self.model, self.decoder)
        if parent_relative_targets != 'off' or reach_projection != 'off':
            from .limb_target_control import LimbTargetControl
            self.limb_target_control = LimbTargetControl(self.model, self.decoder,
                                                         parent_relative=parent_relative_targets,
                                                         reach_projection=reach_projection)
        wide_fovy = float(self.model.cam_fovy[self.model.camera("eye_left").id])
        self.vision_layout = [("wide", eye_size, wide_fovy)]
        if self.fovea_size > 0:
            self.vision_layout.append(("fovea", self.fovea_size,
                                       float(self.model.cam_fovy[self.model.camera("fovea_left").id])))
        self.patcher = BinocularPatcher(self.vision_layout)
        self._renderer = None
        self._fovea_renderer = None
        self._eye_joint_qpos = {}
        for nm in ("robot:left_eye_horizontal", "robot:left_eye_vertical",
                   "robot:right_eye_horizontal", "robot:right_eye_vertical"):
            try:
                j = self.model.joint(nm).id
                self._eye_joint_qpos[nm] = int(self.model.jnt_qposadr[j])
            except Exception:
                pass
        # Snap-to-motion gaze reflex (endowed, permanent; see gaze_reflex.py). Runs on the
        # wide left-eye render in _collect and blends into the gaze command in step().
        # The eye joint dof addresses give the efference-copy speed for saccade suppression.
        self.reflex = None
        # the brainstem's own jump-and-hold eye movement (see SpontaneousSaccades)
        self.saccades = None
        if spontaneous_saccades:
            from mimo_bridge.gaze_reflex import SpontaneousSaccades
            self.saccades = SpontaneousSaccades(dt=self.dt, seed=seed + 101)
        self._last_gaze_cmd = (0.0, 0.0)
        self._eye_joint_dof = []
        if gaze_reflex and self.vision:
            from mimo_bridge.gaze_reflex import GazeReflex
            for nm in ("robot:left_eye_horizontal", "robot:left_eye_vertical",
                       "robot:right_eye_horizontal", "robot:right_eye_vertical"):
                try:
                    self._eye_joint_dof.append(int(self.model.jnt_dofadr[self.model.joint(nm).id]))
                except Exception:
                    pass
            self.reflex = GazeReflex(size=eye_size, fovy_deg=wide_fovy, dt=self.dt,
                                     mask_deg=self.fovea_fov if self.fovea_size > 0 else 16.0,
                                     ring_size=self.ring_size, ring_fovy=self.ring_fov)
        self._target_id = None
        if scene in ("reach", "crib"):
            self._target_id = int(self.model.body("target").mocapid[0])
        self._target_vel = np.zeros(3)   # slow drift of the ball (learnable, unlike teleporting)
        self._audio = None               # crib hearing: spectrum-space synthesis (lane C)
        if audio_real:
            from mimo_bridge.audio_adapter import CribAudio
            self._audio = CribAudio(self.dt)
        self._cr = None                  # attention-critter state (squeaker, dragonfly, bee)
        if self.critters:
            try:
                _g = self.model.geom
                cr = {
                    "sq_id": int(self.model.body("squeaker").mocapid[0]),
                    "sq_geom": int(_g("squeaker_geom").id),
                    "df_id": int(self.model.body("dragonfly").mocapid[0]),
                    "bee_id": int(self.model.body("bee").mocapid[0]),
                    "df_wings": [int(_g(n).id) for n in
                                 ("dragonfly_wing_fl", "dragonfly_wing_fr",
                                  "dragonfly_wing_hl", "dragonfly_wing_hr")],
                    "bee_wings": [int(_g(n).id) for n in ("bee_wing_l", "bee_wing_r")],
                    "bee_geoms": [int(_g(n).id) for n in
                                  ("bee_body", "bee_stripe1", "bee_stripe2", "bee_head",
                                   "bee_wing_l", "bee_wing_r")],
                    "df_geoms": [int(_g(n).id) for n in
                                 ("dragonfly_thorax", "dragonfly_abdomen", "dragonfly_head",
                                  "dragonfly_eye_l", "dragonfly_eye_r",
                                  "dragonfly_wing_fl", "dragonfly_wing_fr",
                                  "dragonfly_wing_hl", "dragonfly_wing_hr")],
                    "rng": np.random.default_rng(seed + 202),
                    "sq_home": None, "sq_squash_t0": -10**9, "sq_cool_until": -1,
                    "sq_hits": 0, "sq_hits_foot": 0,
                    "df_mode": "hover", "df_until": 0, "df_wp": None,
                    "df_bez": None, "df_prev": None,
                    "df_darts": 0, "df_dart_frame": -10**9,
                    "bee_wp": None, "bee_anchor": None, "bee_prev": None,
                    "bee_pause_until": -1, "bee_next_pause": 0,
                }
                # exact baselines, restored every frame outside an animation
                cr["sq_size0"] = np.array(self.model.geom_size[cr["sq_geom"]])
                cr["sq_rgba0"] = np.array(self.model.geom_rgba[cr["sq_geom"]])
                cr["wing_pos0"] = {g: np.array(self.model.geom_pos[g])
                                   for g in cr["df_wings"] + cr["bee_wings"]}
                self._cr = cr
            except Exception:
                self._cr = None          # scene has no critter bodies (non-crib)
        # weight boxes (gain-of-function props): geom id -> class name
        self._box_geoms = {}
        for cls in ("light", "medium", "heavy"):
            try:
                self._box_geoms[int(self.model.geom(f"box_{cls}_geom").id)] = cls
            except Exception:
                pass
        self._mimo_body_set = set(int(b) for b in self.touch.sensing_bodies())
        # video recording state
        self._video = None      # dict(frames=[], path=str, n=int, size=int) while recording
        self.frame = 0
        self._last_eyes = (None, None)
        self._last_fovea = (None, None)
        # The caregiver. A real baby does not chase a toy that has rolled behind it - an
        # adult puts it back where the baby can see it. Ours had no adult: the ball is
        # leashed to where it spawned while the body slides off across the pen, so it sat
        # 60 to 120 degrees off the eye axis in every run since H12 and was within 10
        # degrees on 0.88% of body-steps. Nothing about reaching can be learned for a target
        # that is never in view. After this many seconds of the ball being outside the wide
        # field, it is placed back in front of the eyes. It is a fact about the world, not a
        # reward: the model is told nothing and scored on nothing.
        self.caregiver_seconds = float(caregiver_seconds)
        self._unseen_s = 0.0
        self._rescues = 0
        try:
            self._mobile_geoms = {int(self.model.geom(i2).id)
                                  for i2 in range(self.model.ngeom)
                                  if (self.model.geom(i2).name or "").startswith("mobile_")}
            self._mobile_speed_geom = int(self.model.geom("mobile_el_c_g").id)
            if not self._mobile_geoms:
                self._mobile_geoms = None
            # the hand shapes, for the reach gauge below: the same body-name test the
            # mobile's own hand-strike flag uses, so "a hand" means one thing in both
            self._hand_geoms = [int(g) for g in range(self.model.ngeom)
                                if any(k in (self.model.body(int(self.model.geom_bodyid[g])).name or "")
                                       for k in ("hand", "distal", "middle", "knuckle",
                                                 "metacarpal", "thbase", "thhub"))] or None
        except Exception:
            self._mobile_geoms = None
            self._mobile_speed_geom = None
            self._hand_geoms = None
        self._mobile_prev = None
        # "eyes on ball" must be EARNED. The mobile is deliberately placed inside the
        # visual field at every reset and every caregiver rescue, so a raw under-10-deg
        # count partly measures my own placement, not the model looking. After any
        # placement this lock engages if the ball is already near the gaze; the eyes must
        # first be clearly off it (>15 deg) and then come back under 10 deg by their own
        # movement before a look counts. Ball drift through a static gaze still counts -
        # small, leashed - and is noted where the number is displayed.
        self._onball_lock = True
        # window measure of command following: last W commanded rotations and orientations
        self._cmd_window = 10
        self._cmd_hist = []      # list of (w_cmd [23,3], xquat_before [23,4])
        # Score each command over its TENURE - the steps it was actually in force - not a
        # fixed 250 ms. With action_repeat 5 a command lives 50 ms and is then replaced;
        # grading it 250 ms later measured how well the body tracks an order that was
        # withdrawn 24 steps ago, and read 0.04 in live training against 0.87 in the
        # held-command probe. The trainer passes its action_repeat here.
        self._cmd_horizon = max(1, int(cmd_horizon))
        self._cmd_delay = []     # (motor, xpos, xquat) waiting to be scored at the horizon
        from .skin_contract import regional_sensitivity
        if touch_geometry not in ("body", "surface"):
            raise ValueError("touch_geometry must be body or surface")
        self.touch_geometry = touch_geometry
        self.touch_sensitivity = regional_sensitivity(self.patch_names(), skin_profile)
        if self.contact_profile in ('mechanical_v2_local','mechanical_v3','mechanical_v4'):self._initialize_virtual_skin()
        if self.body_function_mode != 'off':
            from lem.body_function import BodyFunction
            self.body_function = BodyFunction(self.patch_names())
        self.caregiver = None
        if caregiver_mode != "off":
            from .caregiver import Caregiver
            self.caregiver = Caregiver(self, caregiver_mode, caregiver_speech_dir,
                                       caregiver_replay or None, record_dir=caregiver_record_dir or None,
                                       status_dir=run_dir,present_seconds=caregiver_present_seconds,
                                       absent_seconds=caregiver_absent_seconds,
                                       toy_hand_off=toy_hand_off)
            self.caregiver_seconds = 0.0
        self.reset()
        # the per-frame trackers as this world was built (_FRAME_TRACKERS): what a snapshot saved
        # without them restores to, never saved itself
        self._built_trackers = self._frame_tracker_state()

    # ------------------------------------------------------------------ helpers
    @property
    def n_patches(self) -> int:
        return self.pooler.n_patches

    def patch_names(self) -> List[str]:
        return [s.body_name for s in self.pooler.specs]

    def _renderer_obj(self):
        if self._renderer is None:
            self._renderer = EyeRenderer(self.model, height=self.eye_size, width=self.eye_size)
        return self._renderer

    def _fovea_renderer_obj(self):
        if self._fovea_renderer is None:
            # built AFTER the wide one: the process-wide GL context takes its size from the
            # first caller, and a later request must not be larger than it
            self._renderer_obj()
            self._fovea_renderer = EyeRenderer(self.model, height=self.fovea_size, width=self.fovea_size)
        return self._fovea_renderer

    def render_ring(self) -> Optional[np.ndarray]:
        """The far-periphery ring: left eye only, read only by the reflex. Rendered
        through the SAME wide-eye renderer (no extra GL context: a third context per
        worker put 8 workers over the driver's limit - GL 0x505 at makeContext,
        measured at the H67 launch). The image is eye_size px over ring_fov degrees;
        the reflex pools it to 16 px cells, coarseness comes from the optics."""
        if self.ring_size <= 0:
            return None
        return self._renderer_obj().render(self.data, "ring_left")

    # ------------------------------------------------------------------ critters
    def _angle_from_eye_deg(self, pos) -> float:
        """Angle (deg) between the left eye camera's forward axis and a world point."""
        cid = self.model.camera("eye_left").id
        fwd = -np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[:, 2]
        v = np.asarray(pos) - np.asarray(self.data.cam_xpos[cid])
        n = np.linalg.norm(v)
        if n < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(fwd, v / n), -1, 1))))

    def _body_anchor(self) -> np.ndarray:
        """The airspace anchor, measured from the BODY (never assumed): the point
        over the chest, between the head and the hips. The old waypoint box put
        every flight over the feet-side rail (operator caught it; my head/feet
        sign assumption was never verified against a camera)."""
        try:
            head = np.asarray(self.data.xpos[self.model.body("head").id])
            hip = np.asarray(self.data.xpos[self.model.body("hip").id])
            c = 0.5 * (head + hip)
        except Exception:
            c = np.array([0.05, 0.0, 0.1])
        c[2] = 0.0
        return c

    def _critter_waypoint(self, rng) -> np.ndarray:
        """Airspace over the face and chest, measured from the body."""
        c = self._body_anchor()
        return np.array([c[0] + rng.uniform(-0.18, 0.22), c[1] + rng.uniform(-0.22, 0.22),
                         rng.uniform(0.35, 0.60)])

    @staticmethod
    def _heading_quat(v) -> np.ndarray:
        """Quaternion (w,x,y,z) aligning the body's +x with velocity v: yaw then pitch."""
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])
        u = np.asarray(v, float) / n
        yaw = float(np.arctan2(u[1], u[0]))
        pitch = float(-np.arcsin(np.clip(u[2], -1.0, 1.0)))
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])

    def _critters_reset(self):
        cr = self._cr
        if not cr:
            return
        d, m = self.data, self.model
        # the kick panel sits just past the feet, wherever this age's feet are
        try:
            feet = np.stack([d.xpos[m.body("left_foot").id], d.xpos[m.body("right_foot").id]])
            home = feet.mean(axis=0)
            home[0] -= 0.10          # feet point toward -x in the crib (head +x)
            home[2] = 0.10
        except Exception:
            home = np.array([-0.45, 0.0, 0.10])
        cr["sq_home"] = home
        d.mocap_pos[cr["sq_id"]] = home
        cr["sq_squash_t0"] = -10**9
        cr["sq_cool_until"] = -1
        cr["last_moment"] = ""
        rng = cr["rng"]
        cr["df_wp"] = self._critter_waypoint(rng)
        d.mocap_pos[cr["df_id"]] = cr["df_wp"]
        cr["df_mode"], cr["df_until"] = "hover", self.frame + int(rng.uniform(2.0, 5.0) / self.dt)
        cr["df_bez"], cr["df_prev"] = None, None
        cr["bee_wp"] = self._critter_waypoint(rng)
        cr["bee_anchor"] = cr["bee_wp"].copy()
        d.mocap_pos[cr["bee_id"]] = cr["bee_wp"]
        cr["bee_prev"] = None
        # flower head world positions, harvested from the bouquet's geoms
        try:
            self._mj.mj_forward(m, d)
            cr["flowers"] = [np.asarray(d.geom_xpos[m.geom(f"flower{i}").id]).copy()
                             for i in (1, 2, 3, 4)]
        except Exception:
            cr["flowers"] = []
        cr["bee_state"] = "roam"          # roam | to_flower | landed
        cr["bee_state_until"] = 0
        cr["bee_flower_i"] = 0
        # one visitor at a time (operator): bee first, then gaps, then the dragonfly
        cr["visitor"] = "bee"
        cr["next_visitor"] = "dragonfly"
        cr["visitor_until"] = self.frame + int(rng.uniform(25.0, 40.0) / self.dt)
        cr["df_state"] = "patrol"
        cr["df_theta"] = 0.0
        cr["df_spd"] = 1.1
        self._critter_show("bee", True)
        self._critter_show("dragonfly", False)

    def _critter_show(self, name: str, on: bool) -> None:
        """A visitor leaves the scene entirely: geoms invisible, body parked far
        below the pen (an invisible mover must not feed the motion reflex)."""
        cr = self._cr
        if not cr:
            return
        gids = cr["bee_geoms"] if name == "bee" else cr["df_geoms"]
        for g in gids:
            self.model.geom_rgba[g][3] = (0.95 if on else 0.0)
        if not on:
            self.data.mocap_pos[cr["bee_id" if name == "bee" else "df_id"]] = \
                np.array([2.0, 2.0, -1.0])

    def _critters_tick(self):
        cr = self._cr
        if not cr:
            return
        d, m, rng, f = self.data, self.model, cr["rng"], self.frame
        t = f * self.dt
        # squeaker: contingent hop on FRESH body contact, with a 1 s cooldown.
        # review 2026-09-06: with no freshness gate a foot RESTING on the dome
        # re-triggered every cooldown forever - a 1 Hz ambient squeak, inflated hit
        # counts, and a broken "sounds when struck" contingency. Rising edge only.
        touching = False
        hit = hit_foot = False
        for c in range(int(d.ncon)):
            g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
            if cr["sq_geom"] not in (g1, g2):
                continue
            other = g2 if g1 == cr["sq_geom"] else g1
            nm = self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_GEOM, other) or ""
            if nm.startswith("geom:"):
                touching = True
                if f >= cr["sq_cool_until"] and not cr.get("sq_touch_prev"):
                    hit = True
                    hit_foot = hit_foot or ("foot" in nm or "lower_leg" in nm)
        cr["sq_touch_prev"] = touching
        if hit:
            cr["sq_hits"] += 1
            cr["sq_hits_foot"] += int(hit_foot)
            cr["sq_squash_t0"] = f
            cr["sq_cool_until"] = f + int(1.0 / self.dt)
            cr["last_moment"] = "squeaker"
            if self._audio is not None and cr["sq_home"] is not None:
                self._audio.squeak(cr["sq_home"])
        # squeaker response: the dome SQUASHES and flushes bright, then rebounds -
        # a pressed button, not a teleporting block
        u = (f - cr["sq_squash_t0"]) * self.dt / 0.30
        if 0.0 <= u <= 1.0:
            k = float(np.sin(np.pi * u))
            m.geom_size[cr["sq_geom"]] = cr["sq_size0"] * np.array([1 + 0.30 * k, 1 + 0.30 * k, 1 - 0.62 * k])
            m.geom_rgba[cr["sq_geom"]][:3] = (cr["sq_rgba0"][:3] * (1 - k)
                                              + np.array([1.0, 0.55, 0.35]) * k)
        else:
            m.geom_size[cr["sq_geom"]] = cr["sq_size0"]
            m.geom_rgba[cr["sq_geom"]][:3] = cr["sq_rgba0"][:3]
        if cr["sq_home"] is not None:
            d.mocap_pos[cr["sq_id"]] = cr["sq_home"]
        # ---- one visitor at a time (operator): bee OR dragonfly, gaps between
        if f >= cr["visitor_until"]:
            if cr["visitor"] is None:
                cr["visitor"] = cr["next_visitor"]
                cr["next_visitor"] = "bee" if cr["visitor"] == "dragonfly" else "dragonfly"
                cr["visitor_until"] = f + int(rng.uniform(25.0, 40.0) / self.dt)
                self._critter_show(cr["visitor"], True)
                if cr["visitor"] == "dragonfly":
                    cr["df_prev"] = None
                    cr["df_state"], cr["df_theta"] = "patrol", float(rng.uniform(0, 6.28))
                else:
                    cr["bee_prev"] = None
                    cr["bee_state"] = "roam"
                    cr["bee_anchor"] = self._critter_waypoint(rng)
            else:
                self._critter_show(cr["visitor"], False)
                cr["visitor"] = None
                cr["visitor_until"] = f + int(rng.uniform(6.0, 12.0) / self.dt)
        # ---- dragonfly: territorial PATROL (researched: fast repeated circuit with
        # brief rock-still hover pauses at lookout spots - fast, hover, fast)
        if cr["visitor"] == "dragonfly":
            c = self._body_anchor()
            if cr["df_state"] == "patrol":
                spd = cr.get("df_spd", 1.1)
                cr["df_theta"] += spd * self.dt / 0.32          # angular speed = v / r
                th = cr["df_theta"]
                df_pos = np.array([c[0] + 0.32 * np.cos(th),
                                   c[1] + 0.26 * np.sin(th),
                                   0.48 + 0.05 * np.sin(2 * th)])
                # a lookout pause about twice per lap, at consistent route points
                if (th % np.pi) < spd * self.dt / 0.32 and rng.random() < 0.65:
                    cr["df_state"] = "hover"
                    cr["df_hover_until"] = f + int(rng.uniform(1.5, 3.5) / self.dt)
                    cr["df_hover_pos"] = df_pos.copy()
                    cr["df_darts"] += 1          # one fast patrol segment completed
            else:
                fig = 0.003 * np.array([np.sin(2 * np.pi * 0.4 * t),
                                        np.sin(2 * np.pi * 0.7 * t),
                                        0.4 * np.sin(2 * np.pi * 0.5 * t)])
                df_pos = cr["df_hover_pos"] + fig
                if f >= cr["df_hover_until"]:
                    cr["df_state"] = "patrol"
                    cr["df_spd"] = float(rng.uniform(0.9, 1.5))
            if cr["df_prev"] is not None:
                d.mocap_quat[cr["df_id"]] = self._heading_quat(df_pos - cr["df_prev"])
            d.mocap_pos[cr["df_id"]] = df_pos
            cr["df_prev"] = df_pos.copy()
        # ---- bee: commutes between the bouquet's flowers - fly in, LAND (silent,
        # wings still), wait, take off, next flower; short roams between rounds
        if cr["visitor"] == "bee":
            state = cr["bee_state"]
            if state == "roam":
                anchor = cr["bee_anchor"]
                loop = np.array([0.035 * np.sin(2 * np.pi * 0.7 * t),
                                 0.035 * np.sin(2 * np.pi * 0.5 * t + 1.1),
                                 0.020 * np.sin(2 * np.pi * 0.9 * t + 0.5)])
                bee_pos = anchor + loop
                if f >= cr.get("bee_state_until", 0) and cr["flowers"]:
                    cr["bee_state"] = "to_flower"
                    cr["bee_flower_i"] = int(rng.integers(0, len(cr["flowers"])))
            elif state == "to_flower":
                tgt = cr["flowers"][cr["bee_flower_i"]] + np.array([0, 0, 0.012])
                p = np.asarray(d.mocap_pos[cr["bee_id"]])
                to = tgt - p
                dist = float(np.linalg.norm(to))
                if dist < 0.015:
                    cr["bee_state"] = "landed"
                    cr["bee_state_until"] = f + int(rng.uniform(2.0, 4.0) / self.dt)
                    bee_pos = tgt
                else:
                    wob = 0.008 * np.array([np.sin(2 * np.pi * 1.3 * t),
                                            np.sin(2 * np.pi * 1.1 * t + 1), 0.0])
                    bee_pos = p + 0.14 * self.dt * to / max(dist, 1e-6) + wob * self.dt
            else:  # landed: perched on the flower, wings still, buzz silent
                bee_pos = cr["flowers"][cr["bee_flower_i"]] + np.array([0, 0, 0.012])
                if f >= cr["bee_state_until"]:
                    visited = cr.get("bee_visits", 0) + 1
                    cr["bee_visits"] = visited
                    if visited % 4 == 0:         # after a few flowers, a short roam
                        cr["bee_state"] = "roam"
                        cr["bee_anchor"] = self._critter_waypoint(rng)
                        cr["bee_state_until"] = f + int(rng.uniform(4.0, 8.0) / self.dt)
                    else:
                        cr["bee_state"] = "to_flower"
                        cr["bee_flower_i"] = int((cr["bee_flower_i"]
                                                  + 1 + rng.integers(0, len(cr["flowers"]) - 1))
                                                 % len(cr["flowers"]))
            if cr["bee_prev"] is not None and cr["bee_state"] != "landed":
                d.mocap_quat[cr["bee_id"]] = self._heading_quat(bee_pos - cr["bee_prev"])
            d.mocap_pos[cr["bee_id"]] = bee_pos
            cr["bee_prev"] = bee_pos.copy()
        # wings: hard-contrast alpha flicker + a 1.5 mm bob for glint. Bee pairs beat
        # together at the operator-approved 20 Hz; the dragonfly's fore and hind pairs
        # ALTERNATE, now at 18 Hz (operator: "wings should be beating pretty fast" -
        # real dragonflies run ~30 Hz, 18 is the fastest that still reads at our
        # ~50 Hz render rate while staying distinct from the bee's 20)
        # wings: shimmer between SOLID and BRIGHT (never vanish) - but ONLY for the
        # visitor that is present (the flicker used to overwrite the hide-alpha every
        # step, resurrecting the parked critter), and a LANDED bee's wings fold still
        if cr["visitor"] == "bee":
            if cr.get("bee_state") == "landed":
                for g in cr["bee_wings"]:
                    m.geom_rgba[g][3] = 0.85
                    m.geom_pos[g] = cr["wing_pos0"][g]
            else:
                bee_on = np.sin(2 * np.pi * 20.0 * t) > 0
                for g in cr["bee_wings"]:
                    m.geom_rgba[g][3] = 0.98 if bee_on else 0.45
                    m.geom_pos[g] = cr["wing_pos0"][g] + np.array([0, 0, 0.002 if bee_on else -0.002])
        if cr["visitor"] == "dragonfly":
            df_phase = np.sin(2 * np.pi * 18.0 * t)
            for i, g in enumerate(cr["df_wings"]):
                fore = i < 2
                on = df_phase > 0 if fore else df_phase <= 0
                m.geom_rgba[g][3] = 0.92 if on else 0.55
                m.geom_pos[g] = cr["wing_pos0"][g] + np.array([0, 0, 0.003 if on else -0.003])

    # ------------------------------------------------------------------ video
    def start_video(self, path: str, n_frames: int, size: int = 480, every: int = 10):
        """Record the watch camera over n_frames env steps, keeping every `every`-th frame,
        and save an mp4 at `path` when full. Default: 3000 steps (30 SIM seconds), every 10th
        -> 300 video frames at 10 fps = a 30 s clip that plays sim time at real speed.
        If a recording is already going, the new request is ignored (no restarts)."""
        if self._video is not None:
            return
        self._video = dict(frames=[], path=path, n=int(n_frames), every=max(1, int(every)), size=int(size), k=0)

    # ---------------------------------------------------------------- state branching
    def snapshot_state(self) -> dict:
        """Everything needed to put this body back exactly where it is.

        Exists for the matched passive control: to ask "did the ACTION do that, or would
        gravity and contact have done it anyway?" you have to run the SAME initial state
        twice. Comparing against a separately-collected limp trajectory compares two
        different state distributions and answers a different question.
        """
        d = self.data
        return {"qpos": np.array(d.qpos), "qvel": np.array(d.qvel),
                "act": np.array(d.act) if d.act.size else None,
                "ctrl": np.array(d.ctrl),
                "mocap_pos": np.array(d.mocap_pos), "mocap_quat": np.array(d.mocap_quat),
                "time": float(d.time), "frame": int(self.frame),
                "target_vel": None if getattr(self, "_target_vel", None) is None
                              else np.array(self._target_vel),
                "last_gaze_cmd": tuple(self._last_gaze_cmd)}

    def restore_state(self, snap: dict) -> None:
        """Put the body back. mj_forward re-derives every dependent quantity."""
        d = self.data
        d.qpos[:] = snap["qpos"]
        d.qvel[:] = snap["qvel"]
        if snap["act"] is not None and d.act.size:
            d.act[:] = snap["act"]
        d.ctrl[:] = snap["ctrl"]
        d.mocap_pos[:] = snap["mocap_pos"]
        d.mocap_quat[:] = snap["mocap_quat"]
        d.time = snap["time"]
        self.frame = snap["frame"]
        if snap["target_vel"] is not None:
            self._target_vel = np.array(snap["target_vel"])
        self._last_gaze_cmd = tuple(snap["last_gaze_cmd"])
        self._mj.mj_forward(self.model, self.data)

    # helper attributes carried by save_full_state beyond snapshot_state's physics:
    # plain data (dicts/lists/arrays), safe to deepcopy. Caregiver timer/rescue
    # count and the earned-look lock added 2026-09-09 (battery audit: they
    # survived restore and changed later behavior).
    _FULL_STATE_ATTRS = ("_hand_prev", "_cmd_delay", "_target_home", "_cr", "_hand_target_cache",
                         "_handed_ctrl", "ctrl", "_unseen_s", "_rescues",
                         "_onball_lock", "_mechanical_pooler", "_mechanical_sample_time", "body_function", "_virtual_skin", "_virtual_skin_transition",
                         "_native_actuator_gain", "_native_actuator_force_range",
                         "_body_orientation_axes", "_body_roll_max",
                         "_contact_last_seen", "_contact_track_start", "_press_below")
    _FULL_STATE_POOLER = ("_hist", "_last_contact_pos", "_contact_first_step",
                          "_step", "last_fields", "friction_power_density", "mechanical_exposure")
    # THE PER-FRAME TRACKERS _collect AND step READ (2026-09-26, docs/gain_of_function_v4/agents/
    # env_full_state_trackers_2026-09-26.md). Until then save_full_state left them out, so a restored
    # world read the values it held before the restore: the trainer's resume the values of the world
    # it had just built, and a paired branch the values the previous branch left behind.
    #   _mobile_touch_prev, _chime_cool  the mobile's chime gate (_mobile_chime), heard by the model:
    #                                    a branch that chimed blocked the chime in the next branch of
    #                                    the same arm (gaze_probe_replay.md), and a resume inside a
    #                                    touch chimed again
    #   _mobile_prev                     the mobile's speed on the first frame (mobile_speed_cms)
    #   _jit_prev, _jit_ring             the jitter report (the last 100 frames)
    #   _chest_R0                        the roll report's starting orientation (roll_deg)
    #   _cmd_hist                        the 10-order command-following window (window_cos)
    # _roll_max IS LEFT OUT ON PURPOSE (coordinator's decision of 2026-09-26 on the review's semantic
    # finding, env_full_state_trackers_review_2026-09-26.md). It is each body's largest roll_deg since
    # the world was built or reset, and its readers count per training segment: through roll_max_deg
    # it feeds the trainer's roll_deg_max and rolled_over_fraction (train_mimo.py:3187-3188, written at
    # 2174-2175), which the dashboard (dashboard.py:1710-1712), the overseer's "rolled over this
    # episode" line (overseer.py:509-513) and the explainer's posture rows (build_explainer.py:437-440,
    # 1718-1722) read. Carried, the share of bodies "rolled over" could only rise across resumes
    # (H84 segment ..._183858 read 0.485 at its first record; carried it would have read at least
    # 0.75). No model input and no stop rule reads it. So a restore leaves it as the world holds it,
    # as every restore did before 2026-09-26, and the trainer's resume (a newly built world) starts it
    # at 0 each segment. A probe session carries it itself (gof_probe_harness.Session).
    # A snapshot saved before 2026-09-26 carries none of them; restore_full_state then puts back the
    # values this world had when it was built (_built_trackers), which is what every restore did on
    # a freshly built world, and the same values however often that snapshot is restored.
    _FRAME_TRACKERS = ("_mobile_touch_prev", "_chime_cool", "_mobile_prev", "_jit_prev", "_jit_ring",
                       "_chest_R0", "_cmd_hist")

    def _frame_tracker_state(self) -> dict:
        """Copies of the _FRAME_TRACKERS this world holds now (one not yet created is left out)."""
        import copy
        return {n: copy.deepcopy(getattr(self, n)) for n in self._FRAME_TRACKERS if hasattr(self, n)}

    def save_full_state(self) -> dict:
        """snapshot_state PLUS every stateful helper, for paired branch
        comparisons (first_action_benefit). Follow-up review 2026-09-08: only
        physics was restored between branches, so RNG streams, touch history,
        reflex/saccade clocks, audio transients and the moving world diverged
        and imposed a repeatability floor on the comparison. Covered here:
        physics+mocap+time (snapshot_state), the env RNG stream, the touch
        pooler's vibration history / slip anchors / contact ages, gaze-reflex
        and spontaneous-saccade internals (their own RNGs included), the audio
        synthesizer's transients and clock, the critter state, the mobile's
        home, the hand-speed history, the command-response delay queue, and
        since 2026-09-26 the per-frame trackers (_FRAME_TRACKERS: the mobile's
        chime gate, the mobile speed, jitter, the roll's starting orientation
        and the command window).
        NOT covered: renderer/video buffers (no bearing on physics or sensing),
        the reports counted since this world was built (the footprint
        guard's and the rubbing calibration's), and the roll maximum
        (_roll_max), which the trainer's "rolled over" share counts per
        segment (see _FRAME_TRACKERS)."""
        import copy
        snap = self.snapshot_state()
        state_spec = self._mj.mjtState.mjSTATE_INTEGRATION
        integration = np.zeros(self._mj.mj_stateSize(self.model, state_spec))
        self._mj.mj_getState(self.model, self.data, integration, state_spec)
        snap['integration_state'] = integration
        snap['env_histories'] = {name: copy.deepcopy(getattr(self.env, name)) for name in
                                 ('_obs_history', '_action_history')}
        snap['actuation_control'] = np.array(self.env.actuation_model.control_input)
        # mj_step leaves these at its last evaluated substep. Running mj_forward
        # on restore refreshes them to a different instant; the decoder then sees
        # different poses/anchors before the next integration. Preserve its inputs.
        snap['observed_kinematics'] = {name: np.array(getattr(self.data, name)) for name in
            ('xpos', 'xquat', 'xmat', 'xanchor', 'xaxis', 'geom_xpos', 'geom_xmat',
             'cam_xpos', 'cam_xmat', 'site_xpos', 'site_xmat', 'cvel')}
        snap["rng"] = copy.deepcopy(self.rng.bit_generator.state)
        snap["attrs"] = {n: copy.deepcopy(getattr(self, n, None))
                         for n in self._FULL_STATE_ATTRS}
        snap["attrs"].update(self._frame_tracker_state())
        if getattr(self, "limb_target_control", None) is not None:
            # the order's projected arm rows and knee goals, kept only when the limb fixes are on
            # so a snapshot of a world without them is the one it has always been
            snap["attrs"]["_limb_target_cache"] = copy.deepcopy(self._limb_target_cache)
        snap["pooler"] = {n: copy.deepcopy(getattr(self.pooler, n, None))
                          for n in self._FULL_STATE_POOLER}
        for obj_name in ("reflex", "saccades", "_audio"):
            obj = getattr(self, obj_name, None)
            snap["obj_" + obj_name] = None if obj is None else copy.deepcopy(obj.__dict__)
        # fatigue lives on the proprio adapter, which itself holds MuJoCo handles
        # (never deepcopy those) - snapshot only its pure-data sub-state
        snap["fatigue"] = copy.deepcopy(self.proprio.fatigue.__dict__)
        snap["proprio_vel6"] = np.array(self.proprio._vel6)
        # Capture visibility, collision masks and moving mounts as well as physics.
        snap["caregiver"] = None if self.caregiver is None else self.caregiver.snapshot()
        snap["mutable_geometry"] = {name: np.array(getattr(self.model, name)) for name in
            ("geom_rgba", "geom_contype", "geom_conaffinity", "geom_size", "geom_pos", "geom_quat", "geom_rbound", "geom_friction", "geom_solref", "body_pos",
             "actuator_gainprm", "actuator_forcerange")}
        return snap

    def restore_full_state(self, snap: dict) -> None:
        """Inverse of save_full_state. Deep-copies on the way OUT of the snap as
        well, so one saved state can be restored many times (branch comparisons)
        without later branches inheriting earlier branches' mutations."""
        import copy
        migrated_visuals=False
        if self.caregiver is not None:
            from .caregiver_visual import upgrade_legacy_snapshot
            snap,migrated_visuals=upgrade_legacy_snapshot(self,snap)
            if migrated_visuals:
                logging.getLogger("mimo_bridge.env").warning(
                    "Continuing saved episode with caregiver_human_v2 visuals; infant, objects, motors and integration state preserved")
        for name, value in snap.get("mutable_geometry", {}).items():
            getattr(self.model, name)[:] = value
        if self.caregiver is not None and snap.get("caregiver") is not None:
            self.caregiver.restore(snap["caregiver"])
        self.restore_state(snap)
        if 'integration_state' in snap:
            self._mj.mj_setState(self.model, self.data, snap['integration_state'],
                                 self._mj.mjtState.mjSTATE_INTEGRATION)
            self._mj.mj_forward(self.model, self.data)
            # mj_forward computes an acceleration; restore the saved solver warm start
            # afterward so the next step begins with the same integration state.
            self._mj.mj_setState(self.model, self.data, snap['integration_state'],
                                 self._mj.mjtState.mjSTATE_INTEGRATION)
        for name, value in snap.get('env_histories', {}).items():
            setattr(self.env, name, copy.deepcopy(value))
        if 'actuation_control' in snap:
            self.env.actuation_model.control_input = np.array(snap['actuation_control'])
        if migrated_visuals:
            start=int(self.model.body_geomadr[self.model.body('caregiver_head').id])
            for name in ('geom_xpos','geom_xmat'):
                snap['observed_kinematics'][name][start:]=getattr(self.data,name)[start:]
        for name, value in snap.get('observed_kinematics', {}).items():
            getattr(self.data, name)[:] = value
        self.rng.bit_generator.state = copy.deepcopy(snap["rng"])
        for n, v in snap["attrs"].items():
            setattr(self, n, copy.deepcopy(v))
        if '_hand_target_cache' not in snap['attrs'] or self.hand_target_mode == 'world':
            self._hand_target_cache = None
        if '_limb_target_cache' not in snap['attrs'] or getattr(self, 'limb_target_control', None) is None:
            # a world saved without the limb fixes (or restored with them off) projects afresh
            self._limb_target_cache = None
        if "_contact_last_seen" not in snap["attrs"]:
            # a state saved before the contact report existed (every checkpoint up to
            # ckpt_step_344400): its pairs' histories are unknown, so the tracker restarts
            # on the next collect rather than calling every touching pair a new onset
            self._contact_last_seen = None
        if "_press_below" not in snap["attrs"]:
            # the same for the world-press onset tracker (added 2026-09-23): restart, no onset
            self._press_below = None
        missing = [n for n in self._FRAME_TRACKERS if n not in snap["attrs"]]
        if missing:
            # a state saved before 2026-09-26 carries no per-frame tracker: each goes back to the
            # value this world had when it was built. A freshly built world already holds those, so
            # a trainer's resume continues exactly as before this change; a world restored many
            # times gets the same values every time instead of the previous branch's.
            built = getattr(self, "_built_trackers", None) or {}
            for n in missing:
                if n in built:
                    setattr(self, n, copy.deepcopy(built[n]))
                elif n in vars(self):
                    delattr(self, n)
            if not getattr(self, "_warned_trackers_migrated", False):
                self._warned_trackers_migrated = True
                logging.getLogger("mimo_bridge.env").warning(
                    "Restored a world saved without its per-frame trackers (saved before 2026-09-26): "
                    "%s start from this world as built, as every earlier resume did", ", ".join(missing))
        for n, v in snap["pooler"].items():
            setattr(self.pooler, n, copy.deepcopy(v))
        for obj_name in ("reflex", "saccades", "_audio"):
            obj = getattr(self, obj_name, None)
            saved = snap.get("obj_" + obj_name)
            if obj is not None and saved is not None:
                obj.__dict__.clear()
                obj.__dict__.update(copy.deepcopy(saved))
        if snap.get("fatigue") is not None:
            self.proprio.fatigue.__dict__.clear()
            self.proprio.fatigue.__dict__.update(copy.deepcopy(snap["fatigue"]))
        if snap.get("proprio_vel6") is not None:
            self.proprio._vel6 = np.array(snap["proprio_vel6"])
        if self.contact_profile in ('mechanical_v2_local','mechanical_v3','mechanical_v4') and self._virtual_skin is None:
            marker=getattr(self,'_virtual_skin_transition',None)
            if not marker or marker.get('after_time')!=float(self.data.time):
                raise ValueError('virtual skin requires an explicit continuing-episode transition')
            self._initialize_virtual_skin();self._virtual_skin.inherit_body_function(self.body_function)
            for i,spec in enumerate(self.pooler.specs):
                self._virtual_skin.exposure.values[self._virtual_skin.slices[spec.body_id]]=self.pooler.mechanical_exposure.values[i]
        # the saved skin carries its second memory only if it was saved under a calibration; the world
        # runs the calibration it is built with (a new second memory starts from the saved memory)
        self._apply_rubbing_calibration()

    def _apply_rubbing_calibration(self):
        """Give the virtual skin the second memory this world's rubbing calibration asks for, or none."""
        skin=getattr(self,'_virtual_skin',None)
        if skin is None:
            return
        from lem.skin_mechanics import RUBBING_CALIBRATIONS
        skin.set_pain_profile(RUBBING_CALIBRATIONS[getattr(self,'rubbing_calibration','v4')])

    def _video_tick(self, video_control=None):
        v = self._video
        if v is None:
            return
        v["k"] += 1
        if self._audio is not None:
            # the video's ear: per-step SOURCE states, rendered clean + muxed at save
            # (bins-as-sines sounded like noise; sources sound like the scene)
            _sc = getattr(self._audio, "last_sources", None)
            if _sc is not None:
                v.setdefault("asrc", []).append(_sc)
        if v["k"] % v["every"] == 0:
            try:
                r = self._renderer_obj()
                if not hasattr(self, "_watch") or self._watch is None:
                    self._watch = EyeRenderer(self.model, height=v["size"], width=v["size"])
                    self._watch._no_fx = True
                from .video_control import annotate, LABELS, caregiver_caption
                control = video_control if video_control in LABELS else 'unrecorded'
                caregiver=caregiver_caption(self)
                v["frames"].append(annotate(self._watch.render(self.data, "watch_cam"), control,caregiver))
                # what the baby sees (left eye, follows its own gaze commands)
                # Annotation is applied only to this saved observer copy.
                v.setdefault("eye_frames", []).append(annotate(r.render(self.data, "eye_left"), control,caregiver))
                v.setdefault('control_frames', []).append(dict(
                    frame=len(v['frames'])-1, simulation_seconds=float(self.data.time),
                    body_control=control,caregiver=caregiver))
            except Exception:
                self._video = None
                return
        if v["k"] >= v["n"]:
            try:
                import imageio.v2 as imageio
                import json
                from .video_control import VERSION
                os.makedirs(os.path.dirname(v["path"]), exist_ok=True)
                fps = 1.0 / (self.dt * v['every'])
                imageio.mimwrite(v["path"], v["frames"], fps=fps, quality=6)
                if v.get("eye_frames"):
                    imageio.mimwrite(v["path"].replace(".mp4", "_eye.mp4"), v["eye_frames"], fps=fps, quality=6)
                with open(v['path'].replace('.mp4', '_control.json'), 'w', encoding='utf-8') as stream:
                    json.dump(dict(version=VERSION, fps=fps, frames=v['control_frames'],
                                   observer_overlay=True, sensory_input_modified=False), stream, indent=2)
                if v.get("asrc"):
                    # the scene's sound, rendered from the recorded sources and muxed
                    # into both clips; the wav stays beside them either way
                    from mimo_bridge.audio_adapter import render_track, write_wav, mux_wav_into_mp4
                    wav = v["path"].replace(".mp4", "_heard.wav")
                    write_wav(wav, render_track(v["asrc"], self.dt,
                                                samples=self._audio.sample_waves()))
                    mux_wav_into_mp4(v["path"], wav)
                    if v.get("eye_frames"):
                        mux_wav_into_mp4(v["path"].replace(".mp4", "_eye.mp4"), wav)
            except Exception:
                pass
            self._video = None

    def live_frame(self, path: str, size: int = 480, control=None, keep: str = "") -> float:
        """Render ONE watch-camera frame and put it at `path`, replacing what was there.

        NOTHING IS KEPT: the same file is overwritten every time, so a viewer always reads
        the newest picture and no history piles up on the disk. The write goes to a
        neighbouring .part file and is renamed, so a reader never sees half a picture.
        Returns the milliseconds the whole thing took, which is what decides whether a
        moving live view is affordable beside eight bodies rendering their own eyes.
        """
        import time as _time
        started = _time.perf_counter()
        from .video_control import annotate, LABELS, caregiver_caption
        if getattr(self, "_watch", None) is None:
            self._watch = EyeRenderer(self.model, height=int(size), width=int(size))
            self._watch._no_fx = True
        frame = self._watch.render(self.data, "watch_cam")
        frame = annotate(frame, control if control in LABELS else "unrecorded",
                         caregiver_caption(self))
        import imageio.v2 as imageio
        # the temporary name keeps the .jpg suffix: imageio picks its writer from the
        # extension and refuses a bare ".part"
        root, ext = os.path.splitext(str(path))
        part = root + ".part" + (ext or ".jpg")
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        imageio.imwrite(part, frame, quality=7)
        os.replace(part, str(path))
        if keep:
            # RECORDING: the only time a frame is kept. The live view itself stores
            # nothing - it overwrites one file - so a viewer can watch for hours without
            # filling the disk, and pressing record is what starts keeping pictures.
            try:
                os.makedirs(os.path.dirname(str(keep)) or ".", exist_ok=True)
                imageio.imwrite(str(keep), frame, quality=7)
            except Exception:
                pass
        return (_time.perf_counter() - started) * 1000.0

    def live_state(self, path: str) -> float:
        """Publish THIS body's physical state for an outside viewer, replacing what is at
        `path`. Positions, speeds, the mocap bodies (the adult's hands and head, the toy)
        and the simulation clock - a few kB, no pictures. A viewer process owning its own
        copy of the same scene can put the body exactly here and draw it from any camera
        angle it likes, which costs this run nothing but the write. Returns milliseconds.
        """
        import time as _time
        started = _time.perf_counter()
        d = self.data
        # The sensor layout never moves in body coordinates, so it is published ONCE beside
        # the state: body id, body name, and that body's MIMo touch points. A viewer can
        # then draw the skin's sensing points on the body it is already drawing.
        sensors = os.path.join(os.path.dirname(str(path)) or ".", "sensors.npz")
        if not os.path.exists(sensors):
            try:
                ids, names, pts, counts = [], [], [], []
                for spec in self.pooler.specs:
                    local = np.asarray(self.touch.sensor_positions[spec.body_id], dtype=np.float32)
                    ids.append(int(spec.body_id)); names.append(str(spec.body_name))
                    counts.append(int(local.shape[0])); pts.append(local)
                os.makedirs(os.path.dirname(sensors) or ".", exist_ok=True)
                np.savez(sensors + ".part.npz", body_ids=np.asarray(ids, dtype=np.int64),
                         body_names=np.asarray(names, dtype=object),
                         counts=np.asarray(counts, dtype=np.int64),
                         points=np.concatenate(pts, 0) if pts else np.zeros((0, 3), np.float32))
                os.replace(sensors + ".part.npz", sensors)
            except Exception:
                pass
        # np.savez appends .npz unless the name already ends with it, so the temporary
        # name carries it and the rename lands exactly on `path`
        part = str(path) + ".part.npz"
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        np.savez(part, qpos=np.asarray(d.qpos), qvel=np.asarray(d.qvel),
                 mocap_pos=np.asarray(d.mocap_pos), mocap_quat=np.asarray(d.mocap_quat),
                 ctrl=np.asarray(d.ctrl), time=np.asarray([float(d.time)]),
                 geom_rgba=np.asarray(self.model.geom_rgba),
                 geom_contype=np.asarray(self.model.geom_contype),
                 # the viewer builds ITS copy from the very scene this body is living in,
                 # so a scene edit cannot leave the window drawing a different world
                 scene=np.array(str(getattr(self, "scene_path", "")), dtype=object))
        os.replace(part, str(path))
        return (_time.perf_counter() - started) * 1000.0

    @property
    def video_recording(self) -> bool:
        return self._video is not None

    def save_views(self, path: str, size: int = 480):
        """One composite picture of the WHOLE scene: top-down plus four sides (N, E, S, W),
        laid out in a 2x3 grid together with the baby's own eye view. Saved as PNG."""
        import mujoco
        r = getattr(self, "_watch", None)
        if r is None:
            r = self._watch = EyeRenderer(self.model, height=size, width=size)
            r._no_fx = True
        mj = self._mj
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        chest = self.model.body("chest").id
        cam.lookat[:] = self.data.xpos[chest]
        cam.distance = 2.2
        tiles = []
        for name, az, el in (("top", 90, -89), ("north", 90, -15), ("east", 0, -15), ("south", -90, -15), ("west", 180, -15)):
            cam.azimuth, cam.elevation = az, el
            r.gl.make_current()
            mj.mjv_updateScene(self.model, self.data, r.option, None, cam, mj.mjtCatBit.mjCAT_ALL.value, r.scene)
            r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW.value] = 0
            r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION.value] = 0
            mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, r.ctx)
            mj.mjr_render(r.rect, r.scene, r.ctx)
            mj.mjr_readPixels(r._buf, None, r.rect, r.ctx)
            tiles.append(np.flipud(r._buf).copy())
        eye = self._renderer_obj().render(self.data, "eye_left")
        # pad the eye view to tile size
        pad = np.zeros_like(tiles[0]); h, w = eye.shape[:2]
        pad[:h, :w] = eye
        tiles.append(pad)
        grid = np.concatenate([np.concatenate(tiles[0:3], axis=1), np.concatenate(tiles[3:6], axis=1)], axis=0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.imwrite(path, grid)
        except Exception:
            pass

    def render_eyes(self) -> Tuple[np.ndarray, np.ndarray]:
        r = self._renderer_obj()
        left = r.render(self.data, "eye_left")
        right = r.render(self.data, "eye_right")
        return left, right

    def render_fovea(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.fovea_size <= 0:
            return None, None
        r = self._fovea_renderer_obj()
        return r.render(self.data, "fovea_left"), r.render(self.data, "fovea_right")

    def eye_yaw_pitch(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        q = self.data.qpos
        g = self._eye_joint_qpos
        l = (float(q[g["robot:left_eye_horizontal"]]) if "robot:left_eye_horizontal" in g else 0.0,
             float(q[g["robot:left_eye_vertical"]]) if "robot:left_eye_vertical" in g else 0.0)
        r = (-float(q[g["robot:right_eye_horizontal"]]) if "robot:right_eye_horizontal" in g else 0.0,
             float(q[g["robot:right_eye_vertical"]]) if "robot:right_eye_vertical" in g else 0.0)
        return l, r

    def randomize_target(self):
        """Put the ball 25 to 35 cm in front of the eyes, a little to the right and below the
        line of sight, so it is both visible and reachable with the right arm."""
        if self._target_id is None:
            return
        self._mj.mj_forward(self.model, self.data)
        if self.scene_kind == "crib":
            # a mobile the baby can actually reach: measured from the body, not guessed.
            # Reach = shoulder-to-fingertip segment lengths from the model itself; the ball is
            # hung above the RIGHT SHOULDER at 55 to 80 percent of that reach (audit finding:
            # the old face-relative placement left it 35-48 cm away, untouchable at every age).
            m = self.model
            reach = (np.linalg.norm(m.body_pos[m.body("right_lower_arm").id])
                     + np.linalg.norm(m.body_pos[m.body("right_hand").id])
                     + np.linalg.norm(m.body_pos[m.body("right_ffdistal").id]) + 0.03)
            sh = np.asarray(self.data.xpos[m.body("right_upper_arm").id])
            # Both visible AND reachable, which is what this function's docstring promises
            # and what the crib branch stopped doing. An earlier audit moved the mobile from
            # face-relative to shoulder-relative to make it reachable, and that made it
            # invisible: hung above the shoulder it sits about 60 degrees off the eye axis
            # of a baby looking straight up (measured over 12 resets). Reachable but unseen
            # is no better than seen but unreachable - a mobile has to be both.
            #
            # Solved by sampling rather than by trigonometry, so it cannot be subtly wrong:
            # propose points around the line of sight, keep the first that is inside the
            # eyes' field AND inside arm's reach of the shoulder.
            cid = m.camera("eye_left").id
            eye = np.asarray(self.data.cam_xpos[cid])
            fwd = -np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[:, 2]
            side = np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[:, 0]
            up = np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[:, 1]
            # The two constraints only overlap CLOSE IN. The eyes and the shoulder are about
            # 13 cm apart, and the arm is about 23 cm, so a mobile hung 25 cm straight above
            # the face is 26 cm from the shoulder - just out of reach. Hung 15 cm up it is
            # about 16 cm from the shoulder and still nearly overhead. Sampling the whole
            # range and keeping the most central point that the arm can actually get to
            # finds that overlap without me having to reason it out per body size or age.
            # 0.80 of reach, not all of it: the first version picked the most central point
            # inside reach and that landed at 25.9 cm against a 26.2 cm arm - touchable only
            # at full extension, which is not a mobile a baby bats at.
            best, best_off = None, 1e9
            for frac in (0.75, 0.85, 0.95, 1.0):
                # loosen only as far as needed. A fixed fraction is wrong at every age but
                # one: the body GROWS through the curriculum, so an arm length that leaves
                # room to spare at 24 months leaves none at 1 month, and the loop finds the
                # most comfortable reach that is still in view for whatever body this is.
                for _ in range(400):
                    d = self.rng.uniform(0.10, 0.26)               # along the line of sight
                    a = np.radians(self.rng.uniform(0.0, 28.0))    # how far off it
                    th = self.rng.uniform(0.0, 2 * np.pi)
                    p = eye + fwd * d * np.cos(a) + (side * np.cos(th) + up * np.sin(th)) * d * np.sin(a)
                    if p[2] < 0.12:
                        continue                                    # below the floor, not clamped
                    v = p - eye
                    off = float(np.degrees(np.arccos(np.clip(
                        np.dot(fwd, v / max(np.linalg.norm(v), 1e-9)), -1, 1))))
                    if np.linalg.norm(p - sh) <= frac * reach and off < best_off:
                        best, best_off = p, off
                        if off < 10.0:                              # inside the fovea: enough
                            break
                if best is not None and best_off < 28.0:
                    break
            if best is None:                                    # nothing reachable in view
                best = sh + np.array([0.0, 0.0, reach * 0.8])   # fall back to the old spot
                best[2] = max(0.12, best[2])
            self.data.mocap_pos[self._target_id] = best
            self._target_home = np.array(best)
            self._onball_lock = True          # placement is never an earned look
            return
        cam = self.model.camera("eye_left").id
        R = np.asarray(self.data.cam_xmat[cam]).reshape(3, 3)
        eye = np.asarray(self.data.cam_xpos[cam])
        forward, right, up = -R[:, 2], R[:, 0], R[:, 1]
        dist = self.rng.uniform(0.22, 0.30)
        side = self.rng.uniform(-0.03, 0.12)
        down = self.rng.uniform(0.02, 0.12)
        self.data.mocap_pos[self._target_id] = eye + dist * forward + side * right - down * up
        self._target_home = np.array(self.data.mocap_pos[self._target_id])
        # review 2026-09-06: this branch returned without re-arming the earned-look
        # lock, so after the first unlock every placement here counted as an "earned"
        # look - the exact gift the lock exists to exclude (the crib branch had it)
        self._onball_lock = True

    # ------------------------------------------------------------------ contacts
    def _contacts(self, qvel_for_velocity=None, qvel_for_power=None) -> List[ContactInfo]:
        m, d, mj = self.model, self.data, self._mj
        touch = self.touch
        out: List[ContactInfo] = []
        forces = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            sensing = [b for b in (b1, b2) if touch.has_sensors(b)]
            if not sensing:
                continue
            mj.mj_contactForce(m, d, i, forces)
            fn = float(forces[0])
            if abs(fn) < 1e-9:
                continue
            ft = np.array([forces[1], forces[2]])
            frame = np.asarray(c.frame).reshape(3, 3)   # rows: normal, t1, t2 (world)
            f_world = frame[0] * fn + frame[1] * ft[0] + frame[2] * ft[1]   # force on geom2 from geom1
            pos_world = np.asarray(c.pos, dtype=np.float64).copy()
            pen = max(0.0, -float(c.dist))
            for b in sensing:
                # normal pointing INTO the sensing body (MIMo convention)
                n_into = frame[0] if b == b2 else -frame[0]
                # tangential force acting on the sensing body, in its body frame
                f_on_b = f_world if b == b2 else -f_world
                tang_world = f_on_b - np.dot(f_on_b, n_into) * n_into
                R_b = np.asarray(d.xmat[b]).reshape(3, 3)
                tang_body = R_b.T @ tang_world
                pos_body = touch.get_contact_position_relative(i, b)
                # Jacobians give velocity AT this contact, including rotation.
                # Body-centre translation alone misses a swinging limb's impact.
                j1 = np.zeros((3,m.nv)); j2 = np.zeros((3,m.nv))
                mj.mj_jac(m,d,j1,None,pos_world,b1)
                mj.mj_jac(m,d,j2,None,pos_world,b2)
                jac = j1-j2
                rel = jac @ (d.qvel if qvel_for_velocity is None else qvel_for_velocity)
                if b != b2: rel = -rel
                speed = max(0.0, float(np.dot(rel, n_into)))
                tangent_velocity = rel - np.dot(rel,n_into)*n_into
                tangent_speed = float(np.linalg.norm(tangent_velocity))
                rel_power=rel if qvel_for_power is None else jac@qvel_for_power*(1 if b==b2 else -1)
                friction_power = max(0.,float(np.dot(tang_world,rel_power)))
                other = b1 if b == b2 else b2
                out.append(ContactInfo(other_is_self=(int(other) in getattr(self, "_own_body_ids", ())),
                                       body_id=b, pair_key=(int(c.geom1), int(c.geom2)),
                                       normal_force=abs(fn), tangential_body=tang_body, pos_body=np.asarray(pos_body),
                                       pos_world=pos_world, normal_world=n_into,
                                       normal_body=R_b.T @ n_into, penetration_m=pen,
                                       rel_normal_speed=speed, other_body_name=m.body(other).name,
                                       rel_tangent_speed=tangent_speed, friction_power_w=friction_power))
        return out

    def _mechanical_substep(self, previous_velocity):
        """Capture resolved contact load every physics tick, before observation thinning."""
        self.touch.get_touch_obs()
        contacts=self._contacts(previous_velocity,.5*(previous_velocity+self.data.qvel))
        if self.contact_profile in ('mechanical_v2_local','mechanical_v3','mechanical_v4') and self._virtual_skin is not None:
            self._advance_virtual_skin(contacts)
            self._mechanical_sample_time=float(self.data.time)
            return
        fields=self._mechanical_pooler.update(self.touch.sensor_outputs,contacts,include_vibration=False)
        self.pooler.mechanical_exposure.update(fields,float(self.model.opt.timestep),
                                               self._mechanical_pooler.friction_power_density)
        self._mechanical_sample_time=float(self.data.time)

    def _initialize_virtual_skin(self):
        from .virtual_skin import VirtualTaxelSkin
        self._virtual_skin=VirtualTaxelSkin(self.pooler.specs,self.touch.sensor_positions,self.touch_sensitivity)
        self._skin_unmapped_contacts=0

    def _advance_virtual_skin(self,contacts):
        from .contact_footprint import aggregate_contact_footprints,tangent_basis
        skin=self._virtual_skin
        if skin.last_time is not None and skin.last_time==float(self.data.time):return
        if self.contact_profile=='mechanical_v2_local':
            reference=self._mechanical_pooler.update(self.touch.sensor_outputs,contacts,include_vibration=False)
            skin.advance_support(reference,self.touch.sensor_outputs,self._mechanical_pooler.friction_power_density,
                float(self.data.time),float(self.model.opt.timestep),aggregate=False)
            return
        guard=getattr(self,'_footprint_guard',None)
        records=aggregate_contact_footprints(self.model,self.data,contacts,skin.profile,guard=guard)
        # MIMo gives the two eyes touch sensors, so _contacts admits them, but the bridge's
        # touch map (touch_adapter.py: 53 skin bodies, the eyes are placeholders above
        # SKIN_SCALE_MAX) has no patch for left_eye or right_eye: a hand or object on an eye reached skin.advance with no slice and
        # raised KeyError, ending the run inside a worker (H84 smokes, 2026-09-18, body 46
        # = right_eye, at steps 8 and 10 of two runs). Eye contacts are counted here and
        # dropped from the skin, which is what the touch map already does with them.
        mapped=[r for r in records if r['body_id'] in skin.slices]
        if len(mapped)!=len(records):
            self._skin_unmapped_contacts+=len(records)-len(mapped)
            records=mapped
        unsupported=[r for r in records if not r['supported']]
        if unsupported:
            # Preserve measured force through unsupported geometry. The fallback
            # remains explicitly a sensor-support estimate and cannot qualify a
            # physical pressure claim. It is never disguised as resolved tissue.
            reference=self._mechanical_pooler.update(self.touch.sensor_outputs,contacts,include_vibration=False)
            totals={b:sum(r['normal_force_n'] for r in records if r['body_id']==b) for b in {r['body_id'] for r in unsupported}}
            specs={s.body_id:s for s in self.pooler.specs}
            for record in unsupported:
                b=record['body_id'];index=self.pooler.body_to_patch[b]
                area=max(float(reference[index,3])*1e-6,specs[b].point_area_m2)*record['normal_force_n']/totals[b]
                basis=tangent_basis(record['normal_world'])
                record.update(supported=True,geometry_supported=False,method='legacy_sensor_support_fallback',
                    area_m2=area,mean_pressure_pa=record['normal_force_n']/area,pressure_bounds_pa=None,
                    tangent_axes_m=[np.sqrt(area/np.pi)]*2,polygon_tangent=None,
                    tangent_basis_body=(np.asarray(self.data.xmat[b]).reshape(3,3).T@basis).tolist())
                if guard is not None:
                    from .contact_footprint import bound_mean_pressure
                    bound_mean_pressure(record,guard)
        if guard is not None:self._count_footprint_guard(records)
        skin.advance(records,float(self.data.time),float(self.model.opt.timestep),aggregate=False)

    def _count_footprint_guard(self,records):
        """Add this physics tick's guarded records to the counts since the world was built, and check every
        one against the guard's rule (contact_footprint.check_guard_note; review 2026-09-25, finding 1: the
        guard is judged on the quantity it changes). A check that cannot run counts as a violation."""
        from .contact_footprint import check_guard_note
        counts=self._footprint_guard_counts
        for record in records:
            note=record.get('footprint_guard')
            if not note:continue
            if note.get('rescored'):counts['rescored_flat_overlaps']+=1
            if 'area_before_ceiling_m2' in note:counts['pressure_ceiling_contacts']+=1
            try:why=check_guard_note(record,self._footprint_guard)
            except Exception as error:why='the rule check failed: %r'%(error,)
            if why is not None:
                counts['rule_violations']+=1
                counts['last_violation']='body %s at %.3f s: %s'%(record.get('body_id'),float(self.data.time),why)
            unguarded=note.get('flat_overlap_area_m2',note.get('area_before_ceiling_m2'))
            if unguarded:
                counts['largest_unguarded_pressure_pa']=max(counts['largest_unguarded_pressure_pa'],
                                                            float(record['normal_force_n'])/float(unguarded))

    def footprint_guard_report(self):
        """The guard's setting, what it changed at the last physics tick and since the world was built;
        None when the guard is off (no key is then added anywhere)."""
        guard=getattr(self,'_footprint_guard',None)
        if guard is None:return None
        last=[r.get('footprint_guard') or {} for r in (self._virtual_skin.contact_records if self._virtual_skin else [])]
        return dict(setting=guard.name,edge_rounding_m=guard.edge_rounding_m,pressure_ceiling_pa=guard.pressure_ceiling_pa,
                    last_tick=dict(rescored_flat_overlaps=sum(1 for n in last if n.get('rescored')),
                                   pressure_ceiling_contacts=sum(1 for n in last if 'area_before_ceiling_m2' in n)),
                    since_built=dict(self._footprint_guard_counts))

    # the rubbing calibration report: 100 bins of body pain from 0 to 1, and a detrend horizon of 50 env
    # frames = the goal ring's 10 moments of 5 frames (RING_DETREND_STEPS; H84's action_repeat)
    RUBBING_REPORT_BINS = 100
    RUBBING_REPORT_DETREND_FRAMES = 50
    # SUDDEN DEPOSITS are judged apart from ordinary contact (review 2026-09-25, finding 3), with the
    # definition the baseline used on the planning records (docs/gain_of_function_v4/
    # skin_footprint_measurements.py deposits): the worst patch's load up by more than 10 against its load
    # one planning interval (10 model steps = 50 frames) earlier; the episode lasts until the load is back
    # within 2 of that earlier level. IMPACT frames: the worst patch's acute part at least load 2.36, the
    # peak of the 30 cm fall's median drop (737.5 kPa; scale_anchors.json fall_response.mechanical_v4).
    RUBBING_REPORT_DEPOSIT_JUMP = 10.0
    RUBBING_REPORT_DEPOSIT_SETTLE = 2.0
    RUBBING_REPORT_IMPACT_ACUTE = 737502.1 / 219400.0 - 1.0
    HARM_LINE = 0.92

    def _pain_memory(self):
        """The virtual skin's memory that feeds pain: the recalibrated one under 'excess_shear_v1',
        else today's (which alone drives the skin's local function either way)."""
        skin=self._virtual_skin
        pain=getattr(skin,'pain_exposure',None)
        return pain if (pain is not None and getattr(self,'rubbing_calibration','v4')=='excess_shear_v1') else skin.exposure

    def rubbing_calibration_report(self, patches):
        """Both rubbing rules on this frame's skin: body pain under each (the model's formula on the frame's
        pressure and each rule's per-patch memories, worst patch), the worst patch's load parts, and running
        sums since this world was built, per rule: histograms of body pain over all frames, over ordinary
        frames (outside sudden-deposit episodes) and over impact frames; frames over the harm line split by
        kind; the sudden-deposit episodes; the load parts; and the spread of pain about its own 10-moment
        running average (what the goal ranking's normaliser divides integrity by). None when no calibration
        is set."""
        skin=getattr(self,'_virtual_skin',None)
        if getattr(self,'rubbing_calibration','v4')=='v4' or skin is None or getattr(skin,'pain_exposure',None) is None:
            return None
        from lem.skin_mechanics import PROFILE, WIDE_RANGE_PROFILE, load_parts_np, pain_np
        shape = WIDE_RANGE_PROFILE if self.contact_profile == 'mechanical_v4' else PROFILE
        memories = {'v4': skin.exposure, 'excess_shear_v1': skin.pain_exposure}
        pressure = np.asarray(patches, dtype=np.float64)[:, 0]
        gain = np.asarray(self.touch_sensitivity, dtype=np.float64)[:, 0]
        nb = self.RUBBING_REPORT_BINS
        R = self._rubbing_report
        if R is None:
            R = self._rubbing_report = {'frames': 0, 'rules': {k: dict(
                histogram=[0] * nb, ordinary_histogram=[0] * nb, impact_histogram=[0] * nb,
                over_line=dict(all=0, ordinary=0, impact=0, episode=0),
                episodes=dict(count=0, frames=0, largest_load=0.0, largest=[]),
                parts_sum=[0.0, 0.0, 0.0], detrended=dict(n=0, sum=0.0, sumsq=0.0, avg=0.0, weight=0.0),
                recent=[], episode=None) for k in memories}}
        a = 1.0 / self.RUBBING_REPORT_DETREND_FRAMES
        frame = R['frames']
        pain_now, parts_now = {}, {}
        for rule, mem in memories.items():
            st = R['rules'][rule]
            history = np.stack([mem.values[skin.slices[spec.body_id]].max(0) for spec in self.pooler.specs])
            parts = load_parts_np(pressure, history, gain, shape)
            local = pain_np(parts.sum(-1), shape.aversion_shape)
            worst = int(np.argmax(local))
            p = float(local[worst])
            load = float(parts[worst].sum())
            pain_now[rule] = round(p, 6)
            parts_now[rule] = [round(float(x), 6) for x in parts[worst]]
            # a sudden deposit: up by more than the jump on the load one planning interval earlier
            before = st['recent'][0] if len(st['recent']) >= self.RUBBING_REPORT_DETREND_FRAMES else None
            ep = st['episode']
            if ep is None and before is not None and load - before > self.RUBBING_REPORT_DEPOSIT_JUMP:
                ep = st['episode'] = dict(start_frame=frame, before=round(before, 4), frames=0, max_load=load)
            if ep is not None:
                if load > ep['before'] + self.RUBBING_REPORT_DEPOSIT_SETTLE:
                    ep['frames'] += 1
                    ep['max_load'] = max(ep['max_load'], load)
                else:
                    self._close_rubbing_episode(st)
                    ep = None
            st['recent'].append(load)
            if len(st['recent']) > self.RUBBING_REPORT_DETREND_FRAMES:
                del st['recent'][0]
            b = min(int(p * nb), nb - 1)
            st['histogram'][b] += 1
            impact = float(parts[worst, 0]) >= self.RUBBING_REPORT_IMPACT_ACUTE
            if ep is None:
                st['ordinary_histogram'][b] += 1
            if impact:
                st['impact_histogram'][b] += 1
            if p > self.HARM_LINE:
                o = st['over_line']
                o['all'] += 1
                o['episode' if ep is not None else 'ordinary'] += 1
                o['impact'] += int(impact)
            for j in range(3):
                st['parts_sum'][j] += float(parts[worst, j])
            d = st['detrended']
            if d['weight'] > 0:
                dev = p - d['avg'] / d['weight']
                d['n'] += 1
                d['sum'] += dev
                d['sumsq'] += dev * dev
            d['avg'] = (1 - a) * d['avg'] + a * p
            d['weight'] = (1 - a) * d['weight'] + a
        R['frames'] += 1
        pain_rule = 'excess_shear_v1' if self.rubbing_calibration == 'excess_shear_v1' else 'v4'

        def summary(st):
            ep = st['episode']
            episodes = dict(st['episodes'], open=None if ep is None else dict(ep, max_load=round(ep['max_load'], 4)))
            return dict(histogram=list(st['histogram']), ordinary_histogram=list(st['ordinary_histogram']),
                        impact_histogram=list(st['impact_histogram']), over_line=dict(st['over_line']),
                        episodes=episodes, parts_sum=[round(x, 4) for x in st['parts_sum']],
                        detrended=dict(n=st['detrended']['n'], sum=round(st['detrended']['sum'], 6),
                                       sumsq=round(st['detrended']['sumsq'], 6)))
        return dict(setting=self.rubbing_calibration, pain_rule=pain_rule,
                    other_rule='v4' if pain_rule != 'v4' else 'excess_shear_v1',
                    pain=pain_now, worst_patch_load_parts=parts_now,
                    since_built=dict(frames=R['frames'], bins=nb, detrend_frames=self.RUBBING_REPORT_DETREND_FRAMES,
                                     deposit_jump_load=self.RUBBING_REPORT_DEPOSIT_JUMP,
                                     deposit_settle_load=self.RUBBING_REPORT_DEPOSIT_SETTLE,
                                     impact_acute_load=round(self.RUBBING_REPORT_IMPACT_ACUTE, 4),
                                     harm_line=self.HARM_LINE,
                                     rules={k: summary(v) for k, v in R['rules'].items()}))

    @staticmethod
    def _close_rubbing_episode(st):
        """Count a finished sudden-deposit episode; keep the five with the largest load."""
        ep = st['episode']
        st['episode'] = None
        e = st['episodes']
        e['count'] += 1
        e['frames'] += ep['frames']
        e['largest_load'] = round(max(e['largest_load'], ep['max_load']), 4)
        e['largest'] = sorted(e['largest'] + [dict(ep, max_load=round(ep['max_load'], 4))],
                              key=lambda x: -x['max_load'])[:5]

    def _view_deg(self, point) -> float:
        """Degrees between a world point and the nearer eye's axis (MuJoCo cameras look
        along their -z), the same camera code hand_view_deg uses; 180 when undefined."""
        best = 180.0
        p = np.asarray(point, dtype=np.float64)
        for cid in self._eye_cam_ids:
            fwd = -np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[:, 2]
            v = p - np.asarray(self.data.cam_xpos[cid])
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                best = min(best, float(np.degrees(np.arccos(np.clip(np.dot(fwd, v / n), -1.0, 1.0)))))
        return best

    def _contact_report(self, contacts: Sequence[ContactInfo], task: dict) -> None:
        """The contact events of DESIGN.md section 8, written into `task`.

          contact_patch_state   per skin patch (pooler order): 0 nothing touching, 1 something
                                outside the body, 2 the body's own part, 3 both
          self_contact_onsets   one entry per pair of the body's own parts that started
                                touching on this frame (apart >= ONSET_ABSENT_FRAMES frames,
                                touching now): pair [bodyA, bodyB] and their regions in name
                                order, structural (a pair that touches at rest under zero
                                command, STRUCTURAL_SELF_PAIRS), in_view (the contact point
                                within IN_VIEW_DEG of either eye's axis) and view_deg
          contact_onsets_world  how many (own body, outside body) pairs started touching

        Pairs are BODY pairs, not shape pairs: a hand has several shapes and one touch can
        involve two of them at once. A self-touch is felt on both sides, so it arrives as two
        records with the same pair; it is one onset."""
        names = self._body_names
        frame = int(self.frame)
        seen = self._contact_last_seen
        starting = seen is None
        if starting:
            seen = self._contact_last_seen = {}
            self._contact_track_start = frame
        state = np.zeros(self.n_patches, dtype=np.int64)
        self_pairs: Dict[tuple, ContactInfo] = {}
        world_pairs = set()
        b2p = self.pooler.body_to_patch
        for c in contacts:
            body = int(c.body_id)
            own_side = names[body]
            other = str(c.other_body_name)
            p = b2p.get(body)
            if c.other_is_self:
                if p is not None:
                    state[p] |= 2
                key = ("s", own_side, other) if own_side <= other else ("s", other, own_side)
                best = self_pairs.get(key)
                if best is None or float(c.normal_force) > float(best.normal_force):
                    self_pairs[key] = c
            else:
                if p is not None:
                    state[p] |= 1
                world_pairs.add(("w", own_side, other))
        onsets = []
        world_onsets = 0
        if not starting:
            # a pair never seen since tracking began has been apart since then
            never = self._contact_track_start - 1
            for key, c in self_pairs.items():
                if frame - seen.get(key, never) > ONSET_ABSENT_FRAMES:
                    a, b = key[1], key[2]
                    deg = self._view_deg(c.pos_world)
                    onsets.append(dict(pair=[a, b], regions=[contact_region(a), contact_region(b)],
                                       structural=bool((a, b) in STRUCTURAL_SELF_PAIRS),
                                       in_view=bool(deg < IN_VIEW_DEG), view_deg=round(deg, 2)))
            for key in world_pairs:
                if frame - seen.get(key, never) > ONSET_ABSENT_FRAMES:
                    world_onsets += 1
        for key in self_pairs:
            seen[key] = frame
        for key in world_pairs:
            seen[key] = frame
        task["contact_patch_state"] = state.tolist()
        task["self_contact_onsets"] = onsets
        task["contact_onsets_world"] = int(world_onsets)

    def _hand_press_side(self) -> Dict[int, str]:
        """{body id of each hand and finger skin patch: "left" | "right"}, the same patches the
        hand_press_kpa key reads (names containing HAND_PATCH_KEYS). Derived from the model, so
        it is cached per model and never saved."""
        cache = getattr(self, "_hand_press_side_cache", None)
        if cache is None or cache[0] != id(self.model):
            side = {int(sp.body_id): ("left" if sp.body_name.startswith("left_") else "right")
                    for sp in self.pooler.specs if any(k in sp.body_name for k in HAND_PATCH_KEYS)}
            cache = self._hand_press_side_cache = (id(self.model), side)
        return cache[1]

    def _hand_press_footprint(self, task: dict) -> None:
        """Each side's own footprint pressure on the hands and the onset of a world press, as NEW
        task keys (review env-senses MINOR 3 and 4); hand_press_kpa, hand_pressing and the split
        keys are left exactly as they were.

          hand_press_world_footprint_kpa   largest over the hand patches of the outside side's
                                           force over its own footprint area, kPa
          hand_press_self_footprint_kpa    the same for the body's own parts
          hand_pressing_world_footprint    the first above HAND_PRESS_LINE_KPA (2 kPa, the old
          hand_pressing_self_footprint     key's line, so all of them read alike)
          hand_press_world_onsets          how many hands (0-2) had their outside-side pressure
                                           rise above WORLD_PRESS_ONSET_LINE_KPA on this frame
                                           after at least ONSET_ABSENT_FRAMES (5) frames at or
                                           below it (the contact onset's rule)
          hand_press_world_onsets_tenure   the same after at least one order's tenure below the
                                           line (the env's cmd_horizon: 25 frames = 250 ms on
                                           H84); the key the measurement favours, see
                                           WORLD_PRESS_ONSET_LINE_KPA

        Only the footprint skins (mechanical_v3 and mechanical_v4) carry per-contact areas; under
        any other contact profile the keys are absent (unmeasured, never a stand-in value)."""
        if self.contact_profile not in ('mechanical_v3', 'mechanical_v4') or self._virtual_skin is None:
            return
        try:
            side = self._hand_press_side()
            hands = footprint_hand_press(self._virtual_skin.contact_records, side,
                                         getattr(self, "_own_body_ids", ()), self.model.geom_bodyid)
        except Exception as error:      # a new report must never stop the world; say so once
            if not getattr(self, "_hand_press_footprint_warned", False):
                self._hand_press_footprint_warned = True
                logging.getLogger("mimo_bridge.env").warning(
                    "hand press footprint report unavailable (keys left out): %s", error)
            self._press_below = None    # an unmeasured frame breaks the run of frames below
            return
        world = max(v[0] for v in hands.values())
        own = max(v[1] for v in hands.values())
        task["hand_press_world_footprint_kpa"] = float(world)
        task["hand_press_self_footprint_kpa"] = float(own)
        task["hand_pressing_world_footprint"] = bool(world > HAND_PRESS_LINE_KPA)
        task["hand_pressing_self_footprint"] = bool(own > HAND_PRESS_LINE_KPA)
        self._press_below, (onsets, onsets_tenure) = press_onsets(
            self._press_below, {hand: v[0] for hand, v in hands.items()}, WORLD_PRESS_ONSET_LINE_KPA,
            (ONSET_ABSENT_FRAMES, max(ONSET_ABSENT_FRAMES,
                                      int(getattr(self, "_cmd_horizon", ONSET_ABSENT_FRAMES)))))
        task["hand_press_world_onsets"] = int(onsets)
        task["hand_press_world_onsets_tenure"] = int(onsets_tenure)

    # ------------------------------------------------------------------ step / reset
    def _start_body_function(self):
        """The per-patch body function as reset() starts it: intact, no clock (None when it is off)."""
        if self.body_function_mode != 'off':
            from lem.body_function import BodyFunction
            self.body_function = BodyFunction(self.patch_names())

    def _start_touch_state(self):
        """The touch state as reset() starts it: the touch poolers (vibration histories, slip anchors,
        contact ages), the per-patch mechanical exposure, and under a local contact profile the virtual
        skin (its own clock, its exposure memories and its per-taxel function) with this world's rubbing
        calibration."""
        self.pooler = TouchPooler(self.pooler.specs, self.dt, vib_window=self.pooler.vib_window)
        from lem.skin_mechanics import MechanicalExposure
        self.pooler.mechanical_exposure = MechanicalExposure(len(self.pooler.specs),pain_sensitivity=self.touch_sensitivity[:,0])
        self._mechanical_pooler = TouchPooler(self.pooler.specs,float(self.model.opt.timestep),vib_window=4)
        self._mechanical_sample_time = None
        if self.contact_profile in ('mechanical_v2_local','mechanical_v3','mechanical_v4'):
            self._initialize_virtual_skin()
            self._apply_rubbing_calibration()

    def restart_clock(self, time: float = 0.0) -> None:
        """Start the world's clock over at `time` for a new scripted start (observer validation starts
        every case at 0), and with it everything that keeps its own clock beside the physics, as reset()
        starts it: the touch state (_start_touch_state: the virtual skin's clock, exposure memories and
        per-taxel function, the touch poolers, the mechanical exposure) and the body function that
        mirrors the skin. Physics, fatigue and the world's movers are the caller's.
        Why (observer candidate, 2026-09-26): validate set data.time back to 0 at every case, but the
        virtual skin (contact_profile mechanical_v2_local and later; H84 runs mechanical_v4) kept its
        clock from reset()'s last collect, so the first physics step of the first driven case stopped
        on "virtual skin time reversed". Only the clock set back would leave the skin's memories and
        function at a later instant than the clock; a new start carries none of them.
        A world put back to a SAVED instant uses restore_full_state instead, which puts the skin back
        with the rest (restore_state alone restores physics only)."""
        time = float(time)
        if not np.isfinite(time) or time < 0:
            raise ValueError('the world clock must restart at a finite time >= 0, got %r' % time)
        self.data.time = time
        self._start_body_function()
        self._start_touch_state()

    def reset(self) -> RawStep:
        self._hand_target_cache = None
        self._limb_target_cache = None
        self._virtual_skin=None
        if getattr(self,'_native_actuator_gain',None) is not None:
            self.model.actuator_gainprm[:]=self._native_actuator_gain
            self.model.actuator_forcerange[:]=self._native_actuator_force_range
        self.env.reset()
        self._start_body_function()
        if self.scene_kind == "crib":
            # Let the body settle before anything looks at it, the way MIMo's own roll_over
            # env does (100 steps with zero control after placing). Placed and immediately
            # observed, the first frames show a body mid-drop rather than one lying on the
            # floor, and the ball is then hung relative to a pose that is about to change.
            self.data.ctrl[:] = 0
            for _ in range(100):
                self._mj.mj_step(self.model, self.data)
        self._start_touch_state()
        self.proprio.reset()
        self._cmd_hist = []
        # review E7: these spanned the teleport - the first cmd_horizon slow scores
        # paired pre-teleport commands with post-teleport poses, one garbage jitter
        # sample per reset, and the caregiver timer kept counting across worlds
        self._cmd_delay = []
        self._jit_prev = None
        self._jit_ring = []
        self._unseen_s = 0.0
        # the frame counter must be zeroed BEFORE anything computes a frame-keyed
        # deadline. _critters_reset set visitor_until = self.frame + 25-40s while
        # self.frame still held the OLD episode's ~10,000 - a deadline past the end of
        # every 10,000-frame episode, so after the first episode the scheduler never
        # fired again: bee locked all episode, every episode, dragonfly gone (measured
        # 2026-09-06: dragonfly in 10 of 306 live records, dart counter frozen at its
        # first-episode value; probes without resets looked healthy). Same defect class
        # the reflex reset fixed one line below.
        self.frame = 0
        # review 2026-09-06, three more teleport stragglers: the hand tracker's first
        # post-reset delta read the TELEPORT as thousands of cm/s of "self-caused
        # change" (one fabricated anticipation target per episode); the efference copy
        # carried the dead episode's last eye command; the ball kept its old drift.
        self._hand_prev = {}
        self._last_gaze_cmd = (0.0, 0.0)
        self._target_vel = np.zeros(3)
        # contact onsets are frame-keyed like the reflex deadlines below: a teleport is not a
        # touch, and last-seen frames from the old episode would outrun the zeroed counter
        self._contact_last_seen = None
        # the world-press onset tracker likewise: a teleport is not a press
        self._press_below = None
        if self.reflex is not None:
            self.reflex.reset()        # frame-keyed deadlines die at reset otherwise (measured)
        if self.saccades is not None:
            self.saccades.reset()
        self._critters_reset()
        self.randomize_target()
        self._chest_R0 = None
        self._roll_max = 0.0
        self._body_orientation_axes = None
        self._body_roll_max = 0.0
        self._mobile_prev = None
        # the mobile's chime gate (_mobile_chime) is frame-keyed like the deadlines above: a cooldown
        # set in the old episode outlived the zeroed frame counter and silenced the chime for up to
        # that many frames of the new one. These are the values the gate read before it was ever set.
        self._mobile_touch_prev = False
        self._chime_cool = -1
        self._mj.mj_forward(self.model, self.data)   # so the first frame already shows the moved ball
        self.ctrl = np.zeros(self.model.nu)
        if self.caregiver is not None:
            from .speech import SpeechStream
            self._audio.speech = SpeechStream()
            self.caregiver.reset(self)
            self._mj.mj_forward(self.model, self.data)
        return self._collect(force_render=True)

    def _apply_motor_function(self):
        # Scaling an actuator's fixed force gain changes its available force,
        # rather than changing the requested target or merely reporting fatigue.
        if np.any(self.model.actuator_gaintype!=0) or np.any(self.model.actuator_biastype!=0):
            raise ValueError('motor-function reference requires validated fixed-gain, bias-free actuators')
        if getattr(self,'_native_actuator_gain',None) is None:
            self._native_actuator_gain=self.model.actuator_gainprm.copy()
            self._native_actuator_force_range=self.model.actuator_forcerange.copy()
        factor=np.ones(self.model.nu)
        known=self.proprio.act_set>=0
        factor[known]=self.proprio.fatigue.cap[self.proprio.act_set[known]]
        self.model.actuator_gainprm[:]=self._native_actuator_gain
        self.model.actuator_gainprm[:,0]*=factor
        self.model.actuator_forcerange[:]=self._native_actuator_force_range*factor[:,None]

    def step(self, motor: Optional[np.ndarray] = None, gaze: Optional[Sequence[float]] = None,
             handle_mask: Optional[Sequence[bool]] = None, ctrl: Optional[np.ndarray] = None,
             render: Optional[bool] = None, video_control: Optional[str] = None,
             hand_joint_goal: Optional[np.ndarray] = None) -> RawStep:
        """motor: (24, 9) in the model's action units (pos cm, quat, radius, strength [0,1]); row 23 = gaze.
        Or pass ctrl directly (nu,) in [-1, 1] (random-action baselines)."""
        requested_motor = None if motor is None else np.asarray(motor, dtype=np.float64).copy()
        decoded_hand = ctrl is None and motor is not None and self.hand_target_control is not None
        if hand_joint_goal is not None and (self.hand_target_control is None or motor is None or ctrl is not None):
            raise ValueError('Explicit hand goals require wrist_relative_v1 and motor decoding')
        if decoded_hand:
            motor, self._hand_target_cache = self.hand_target_control.execute(
                self.data, requested_motor, self._hand_target_cache, hand_joint_goal)
        decoded_limbs = ctrl is None and motor is not None and self.limb_target_control is not None
        limb_ledger = None
        if decoded_limbs:
            # the calf goal regenerated around the thigh where it is, the arm rows projected into
            # reach once per order (limb_target_control.py); the ledger is what the learner records
            hand_goal = (None if not decoded_hand or self._hand_target_cache is None else
                         (self.hand_target_control.kinematics, self._hand_target_cache['joint_goal']))
            motor, self._limb_target_cache, limb_ledger = self.limb_target_control.execute(
                self.data, motor, requested_motor, self._limb_target_cache,
                handle_mask=handle_mask, hand=hand_goal)
        if self.saccades is not None:
            gaze = self.saccades.apply(gaze, self.frame,
                                       reflex_active=bool(self.reflex is not None and self.reflex.active))
        if self.reflex is not None:
            gaze = self.reflex.apply(gaze, self.eye_yaw_pitch())   # blend a pending snap; passthrough otherwise
        # the efference copy: the oculomotor command actually issued this step, saccade and
        # reflex included, reported back so the model's motor token can carry it
        if gaze is not None:
            g = np.asarray(gaze, dtype=np.float64).reshape(-1)
            self._last_gaze_cmd = (float(g[0]), float(g[1]))
        if ctrl is None:
            if motor is None:
                ctrl = np.zeros(self.model.nu)
            else:
                # --finger-joint-servo on: the finger chains are servoed toward the latched joint goal itself (the
                # decoder's finger_goals); off: the decoder is called exactly as before, with no goal
                finger = ({} if self.finger_joint_servo != 'on' or not decoded_hand or self._hand_target_cache is None
                          else {'finger_goals': self.hand_target_control.goals_by_joint(
                              self._hand_target_cache['joint_goal'])})
                ctrl = self.decoder.decode(self.data, np.asarray(motor, dtype=np.float64), gaze, handle_mask,
                                           **finger)
        self.ctrl = np.asarray(ctrl, dtype=np.float64)
        if self.body_function_mode == 'modulate':
            self._apply_motor_function()
        if self._target_id is not None and self.caregiver is None:
            # random-walk drift with a leash: pulled back toward its spawn point and never
            # more than 12 cm from it, so the ball stays reachable (overseer finding 16:00:
            # an unleashed walk drifted out of reach and was touched in 0% of checks)
            home = getattr(self, "_target_home", None)
            if home is None:
                home = self._target_home = np.array(self.data.mocap_pos[self._target_id])
            p0 = np.array(self.data.mocap_pos[self._target_id])
            self._target_vel = 0.98 * self._target_vel + 0.02 * self.rng.normal(0, 0.04, 3) + 0.004 * (home - p0)
            p1 = p0 + self._target_vel * self.dt
            d = np.linalg.norm(p1 - home)
            if d > 0.12:
                p1 = home + (p1 - home) * (0.12 / d)
            self.data.mocap_pos[self._target_id] = p1
        self._critters_tick()
        if self.caregiver is not None:
            self.caregiver.tick(self)
        # remember handle-body orientations and angular velocities to measure how much of the
        # commanded rotation happened (displacement) and whether the body ACCELERATED that way
        xq_before = np.array(self.data.xquat[self.proprio.body_ids])
        xp_before = np.array(self.data.xpos[self.proprio.body_ids])
        w_before = self._handle_angvel()
        # What MuJoCo is HANDED, captured at the moment of handing. Distinct from
        # data.ctrl afterwards: MIMo's actuation model rewrites it during the substeps -
        # measured, 18 of 90 actuators (shoulders and finger distals) settle to a
        # different value than they were given. Recording only the post-step value and
        # calling it "applied" would misattribute MIMo's actuation model to the decoder.
        self._handed_ctrl = np.array(self.ctrl)
        self.env.step(self.ctrl)   # MIMo does the physics (frame_skip substeps) and touch
        self.frame += 1
        out = self._collect(want_render=render)
        self._video_tick(video_control)
        out.info["gaze_cmd"] = self._last_gaze_cmd     # efference copy for the model side
        # the decoded joint-space control MuJoCo was actually given, so the ledger can be
        # checked end to end rather than only at the training-loop boundary
        out.info["decoded_ctrl"] = self.ctrl.copy()          # the decoder's output
        out.info["handed_ctrl"] = np.array(getattr(self, "_handed_ctrl", self.ctrl))
        out.info["settled_ctrl"] = np.array(self.data.ctrl)   # after MIMo's actuation model
        out.info["applied_ctrl"] = out.info["handed_ctrl"]    # what "applied" should mean
        if decoded_hand:
            # World target at each 10 ms decoder call, before the body moves.
            # Learned/planned canonical targets stay distinct from this readback.
            out.info['requested_motor_targets'] = requested_motor
            out.info['executed_motor_targets'] = np.asarray(motor).copy()
            if self._hand_target_cache is not None:
                out.info['canonical_motor_targets'] = self._hand_target_cache['canonical'].copy()
                out.info['hand_joint_goal'] = self._hand_target_cache['joint_goal'].copy()
                out.info['hand_joint_error_deg'] = np.degrees(np.abs(
                    self.data.qpos[self.hand_target_control._joint_addresses] -
                    self._hand_target_cache['joint_goal']))
                out.info['hand_target_projection'] = self._hand_target_cache['projection']
        if decoded_limbs:
            out.info['requested_motor_targets'] = requested_motor
            out.info['executed_motor_targets'] = np.asarray(motor).copy()
            out.info['ledger_motor_targets'] = limb_ledger
            out.info['limb_target_contract'] = self.limb_target_control.contract
            import copy
            out.info['limb_target_report'] = copy.deepcopy(self._limb_target_cache['report'])
        if motor is not None:
            out.info["cmd_response"] = self._command_response(np.asarray(motor, dtype=np.float64), xp_before, xq_before, w_before, handle_mask=handle_mask)
            # ...and the same comparison at a horizon where a limb can actually move. The
            # one-step version scores a command against 10 ms of physics, which is not
            # enough for anything with mass to respond: on identical commands, measured,
            # one step reads +0.29 where 250 ms reads +0.77, and in training - 23 handles
            # competing, smoothed, with babble on top - the one-step number went NEGATIVE
            # (-0.13) and read as a body that ignores its instructions. It does not. It is
            # slow. Commands are parked with the pose they were issued against and scored
            # when they come of age.
            # An order entered on this frame is scored after the physics of exactly
            # cmd_horizon frames, this one included (25 frames = 250 ms = one order's
            # tenure on H84). Until 2026-09-23 the queue waited for cmd_horizon + 1 entries,
            # so every order was scored over 26 frames (260 ms): its own 25 and the first
            # frame of the next order. A queue saved before that fix holds one entry more
            # than it should; the surplus oldest entries are dropped here.
            self._cmd_delay.append((np.asarray(motor, dtype=np.float64).copy(),
                                    xp_before.copy(), xq_before.copy()))
            if len(self._cmd_delay) >= self._cmd_horizon:
                del self._cmd_delay[:len(self._cmd_delay) - self._cmd_horizon]
                mo, xp0, xq0 = self._cmd_delay.pop(0)
                out.info["cmd_response_slow"] = self._command_response(
                    mo, xp0, xq0, w_before, handle_mask=handle_mask, record_window=False)
        return out

    def _handle_angvel(self) -> np.ndarray:
        v = np.zeros(6)
        out = np.zeros((len(self.proprio.body_ids), 3))
        for i, b in enumerate(self.proprio.body_ids):
            self._mj.mj_objectVelocity(self.model, self.data, self._mj.mjtObj.mjOBJ_BODY, int(b), v, 0)
            out[i] = v[:3]
        return out

    def _command_response(self, motor: np.ndarray, xp_before: np.ndarray, xq_before: np.ndarray, w_before: np.ndarray, handle_mask=None, record_window=True) -> dict:
        """How well did the body follow? For each handle with strength > 0: the rotation the
        model asked for (target vs orientation before the step) and the rotation that actually
        happened during the step; report the mean cosine between the two and the mean ratio of
        their sizes. cos ~ 1 = the body turns the way it was told (E4 check)."""
        from mimo_bridge.action_decoder import quat_xyzw_to_mat, rotmat_log, POSITION_HANDLES
        cos_list, ratio_list, acc_list = [], [], []
        w_after = self._handle_angvel()
        pos_cos, cmd_mags = [], []
        # THE SAME POSITION CHECK ON THE ROWS WHOSE POSITION THE DECODER READS (motor interface
        # audit 2026-09-25, F4). pos_cos scores every powered row, the 11 orientation-only body rows
        # included, whose ordered position reaches no actuator; with --executed-fields-only on the
        # ledger holds their achieved position at the order's latch, so their score would read how
        # far the part moved away from where it was. pos_cos_position_rows keeps only the rows whose
        # position this world's decoder SOLVES (decoder.use_position: POSITION_HANDLES under the
        # decoder's default, without the head and the pelvis under --position-rows no_head_pelvis,
        # audit F3); the trainer reports it as cmd_response_pos_cos under that flag. Always computed,
        # read by nothing else: today's pos_cos is unchanged.
        pos_cos_rows = []
        names = self.proprio.handle_names
        _dec = getattr(self, "decoder", None)
        live_rows = ({h for h, u in zip(_dec.handle_names, _dec.use_position) if u}
                     if _dec is not None and hasattr(_dec, "use_position") else set(POSITION_HANDLES))
        # the eyes' row, found by NAME in the active body layout (row 23 in both layouts). Until
        # 2026-09-23 this was the LAST row, which is the eyes only in the 24-row legacy layout:
        # under trunk_v1 (26 rows, the trunk after the eyes; every run from H71 on) the chest
        # was never scored and the eyes' head-relative order was scored against the eye
        # body's world rotation, in both loops below
        gaze_idx = self.proprio.handle_names.index("gaze")
        for i, b in enumerate(self.proprio.body_ids):
            if i == gaze_idx:
                # the gaze handle drives no joints in this pathway (dedicated eye servo);
                # scoring its commanded orientation against the EYE BODY's world rotation -
                # which is mostly head motion - filled the mean with garbage rows
                continue
            if motor[i, 8] <= 0 or (handle_mask is not None and not handle_mask[i]):
                continue
            q0 = xq_before[i]
            R0 = quat_xyzw_to_mat([q0[1], q0[2], q0[3], q0[0]])
            q1 = self.data.xquat[b]
            R1 = quat_xyzw_to_mat([q1[1], q1[2], q1[3], q1[0]])
            Rt = quat_xyzw_to_mat(motor[i, 3:7])
            w_cmd = rotmat_log(Rt @ R0.T)
            w_got = rotmat_log(R1 @ R0.T)
            nc, ng = np.linalg.norm(w_cmd), np.linalg.norm(w_got)
            if nc < 1e-4:
                continue
            cos_list.append(float(np.dot(w_cmd, w_got) / (nc * max(ng, 1e-9))) if ng > 1e-6 else 0.0)
            cmd_mags.append(nc)
            # position twin (review fix 5): commanded position step vs achieved displacement,
            # same clip as the decoder - without this a broken position pathway is invisible
            e_cmd = np.clip(motor[i, 0:3] / 100.0 - xp_before[i], -0.5, 0.5)
            npc = float(np.linalg.norm(e_cmd))
            if npc > 1e-3:
                dpos = np.asarray(self.data.xpos[b], dtype=np.float64) - xp_before[i]
                ndp = float(np.linalg.norm(dpos))
                pos_cos.append(float(np.dot(e_cmd, dpos) / (npc * ndp)) if ndp > 1e-5 else 0.0)
                if names[i] in live_rows:
                    pos_cos_rows.append(pos_cos[-1])
            ratio_list.append(float(ng / nc))
            dw = w_after[i] - w_before[i]          # change of angular velocity during the step
            nd = np.linalg.norm(dw)
            acc_list.append(float(np.dot(w_cmd, dw) / (nc * nd)) if nd > 1e-6 else 0.0)
        # window measure: over the last W steps, does the SUM of what was asked line up with the
        # rotation that actually happened? (one-step numbers are mostly momentum; this is the
        # fair "commands correlate with the following change" check, E4)
        w_cmd_all = np.zeros((len(self.proprio.body_ids), 3))
        for i, b in enumerate(self.proprio.body_ids):
            if i == gaze_idx:
                # review E6: the per-step loop skips the gaze handle (head-relative
                # command vs eye-body world rotation is garbage); the window loop
                # did not, so every window sum carried one garbage row
                continue
            if motor[i, 8] <= 0 or (handle_mask is not None and not handle_mask[i]):
                continue
            q0 = xq_before[i]
            R0 = quat_xyzw_to_mat([q0[1], q0[2], q0[3], q0[0]])
            w_cmd_all[i] = rotmat_log(quat_xyzw_to_mat(motor[i, 3:7]) @ R0.T)
        # review E6: this function runs twice per step once the slow pipe fills
        # (fresh + 25-step-old scorer); both used to append, so the "last 10"
        # window spanned 5 steps of interleaved fresh and stale commands
        if record_window:
            self._cmd_hist.append((w_cmd_all, np.array(xq_before)))
            if len(self._cmd_hist) > self._cmd_window:
                self._cmd_hist.pop(0)
        win_cos = []
        if len(self._cmd_hist) == self._cmd_window:
            asked = np.sum([h[0] for h in self._cmd_hist], axis=0)          # [23,3]
            xq_start = self._cmd_hist[0][1]
            for i, b in enumerate(self.proprio.body_ids):
                na = np.linalg.norm(asked[i])
                if na < 1e-3:
                    continue
                qs = xq_start[i]
                Rs = quat_xyzw_to_mat([qs[1], qs[2], qs[3], qs[0]])
                q1 = self.data.xquat[b]
                R1 = quat_xyzw_to_mat([q1[1], q1[2], q1[3], q1[0]])
                got = rotmat_log(R1 @ Rs.T)
                ng = np.linalg.norm(got)
                win_cos.append(float(np.dot(asked[i], got) / (na * ng)) if ng > 1e-4 else 0.0)
        win = float(np.mean(win_cos)) if win_cos else 0.0
        if not cos_list:
            return {"cos": 0.0, "ratio": 0.0, "acc_cos": 0.0, "window_cos": win, "n": 0,
                "pos_cos": float(np.mean(pos_cos)) if pos_cos else 0.0,
                "pos_cos_position_rows": float(np.mean(pos_cos_rows)) if pos_cos_rows else 0.0,
                "cmd_mag": float(np.mean(cmd_mags)) if cmd_mags else 0.0}
        return {"cos": float(np.mean(cos_list)), "ratio": float(np.mean(ratio_list)),
                "acc_cos": float(np.mean(acc_list)), "window_cos": win, "n": len(cos_list),
                "pos_cos": float(np.mean(pos_cos)) if pos_cos else 0.0,
                "pos_cos_position_rows": float(np.mean(pos_cos_rows)) if pos_cos_rows else 0.0,
                "cmd_mag": float(np.mean(cmd_mags)) if cmd_mags else 0.0}

    def _mobile_chime(self, touching: bool, where) -> None:
        """The mobile's strike VOICE (operator decision 2026-09-06): a 12 g element can never reach
        the body-scale impact gate (ceiling ~0.004 Ns vs 0.15), so batting was silent - the
        hear-what-I-did half of the certified deliberate act was missing. Rising edge + 0.3 s
        cooldown, struck element's own position (`where`) so the sound comes from where the eyes saw
        the hit. The gate's two trackers are saved with the world (_FRAME_TRACKERS)."""
        if self._audio is not None and touching:
            if (not getattr(self, "_mobile_touch_prev", False)
                    and self.frame >= getattr(self, "_chime_cool", -1)):
                try:
                    self._audio.chime(where)
                    self._chime_cool = self.frame + int(0.3 / self.dt)
                except Exception:
                    pass
        self._mobile_touch_prev = touching

    def _collect(self, force_render: bool = False, want_render: Optional[bool] = None) -> RawStep:
        d = self.data
        # MIMo already computed touch in env.step()/reset(); sensor_outputs is per body
        if self._mechanical_sample_time is None:
            self.touch.get_touch_obs()
        point_forces = self.touch.sensor_outputs
        contacts = self._contacts()
        patches = self.pooler.update(point_forces, contacts)
        virtual=None
        if self.contact_profile in ('mechanical_v2_local','mechanical_v3','mechanical_v4'):
            self._advance_virtual_skin(contacts);virtual=self._virtual_skin.aggregate()
            patches[:,:4]=virtual['physical'][:,:4]
            # the memory that feeds pain: today's, unless the recalibrated rubbing is switched on
            pain_memory=self._pain_memory()
            for i,spec in enumerate(self.pooler.specs):
                self.pooler.mechanical_exposure.values[i]=pain_memory.values[self._virtual_skin.slices[spec.body_id]].max(0)
        elif self.contact_profile != 'mechanical_v2' or self._mechanical_sample_time is None:
            self.pooler.mechanical_exposure.update(patches,self.dt,self.pooler.friction_power_density)
        exposure=self.pooler.mechanical_exposure.values.copy()
        cols = build_collisions(contacts, self.pooler, self.body_hash, self.dt)
        pro = self.proprio.read(self.dt)
        # THE VESTIBULAR / INTERNAL-STATE VECTOR, every frame and whatever the flags say
        # (DESIGN.md section 7). A resumed world is rebuilt from the constructor arguments it
        # was saved with, so a constructor switch would never reach it; whether the model
        # USES the vector is decided model-side. Read after proprio.read so the fatigue
        # capacities it reports are this frame's.
        vest = read_vestibular(self, contacts)
        # Rendering used to run on a fixed modulus, which did not line up with the frames
        # the model actually looks at: with render_every=2 and action_repeat=5 the eyes were
        # drawn 5 times per 10 frames, the model saw 2 of those frames, and only ONE of the
        # two carried a fresh image - the other reused the previous vision token. The
        # trainer now says which frames it will read (want_render), so every render is used
        # and every model step gets a fresh image.
        eyes = (None, None)
        fovea = (None, None)
        do_render = (force_render or self.frame % self.render_every == 0) if want_render is None \
            else bool(want_render or force_render)
        if self.vision and do_render:
            eyes = self.render_eyes()
            if self.reflex is not None:
                # detection wants the previous RENDERED frame, so observe before overwriting;
                # eye speed (efference copy) gates out self-motion smear
                eye_speed = max((abs(float(d.qvel[a])) for a in self._eye_joint_dof), default=0.0)
                self.reflex.observe(eyes[0], self.eye_yaw_pitch(), eye_speed, self.frame,
                                    ring_img=self.render_ring())
            self._last_eyes = eyes
            fovea = self.render_fovea()
            self._last_fovea = fovea
        # task readout (MIMo's own Reach criterion: hand touches the target ball)
        task = {}
        # infant milestone: rolling. Angle between the chest's current orientation and its
        # orientation at reset (0 = as placed; > 60 deg = rolled to the side; > 120 deg = rolled
        # over). The body is free in the crib, so this is something it can DO.
        chest = self.model.body("chest").id
        R_now = np.asarray(d.xmat[chest]).reshape(3, 3)
        if getattr(self, "_chest_R0", None) is None:
            self._chest_R0 = R_now.copy()
        from mimo_bridge.action_decoder import rotmat_log
        # SELF-TOUCH, as plain counts a person can read. Until 2026-09-21 the fact that a
        # contact was the body's own part was computed and thrown away one line later, so
        # "how often does this body touch itself" had no answer at all.
        _self = getattr(self.pooler, "last_self", None)
        if _self is not None and len(_self):
            _share = np.asarray(_self)[:, 0]
            task["self_contact_patches"] = int((_share > 0.0).sum())
            task["self_contact_pairs"] = int(getattr(self.pooler, "last_self_pairs", 0))
            _total = float(getattr(self.pooler, "last_total_force", 0.0))
            task["self_contact_force_share"] = float(
                getattr(self.pooler, "last_self_force", 0.0) / _total) if _total > 0 else 0.0
            # WHERE it is being touched, by the region names a person uses. Four numbers a
            # region: patches in contact, its share of the body's touch force, and the same
            # two counted only for the body touching itself.
            _regions = getattr(self, "_patch_region", None)
            _fields = getattr(self.pooler, "last_fields", None)
            if _regions is not None and _fields is not None and len(_fields) == len(_regions):
                _press = np.asarray(_fields)[:, 0]
                _touched = _press > 0.0
                _sum = float(_press.sum())
                region_report = {}
                for index, name in enumerate(self._skin_region_names):
                    rows = _regions == index
                    if not rows.any():
                        continue
                    region_report[name] = dict(
                        patches=int((_touched & rows).sum()),
                        force_share=float(_press[rows].sum() / _sum) if _sum > 0 else 0.0,
                        self_patches=int(((_share > 0.0) & rows).sum()))
                task["skin_regions"] = region_report
        task["roll_deg"] = float(np.degrees(np.linalg.norm(rotmat_log(R_now @ self._chest_R0.T))))
        self._roll_max = max(getattr(self, "_roll_max", 0.0), task["roll_deg"])
        task["roll_max_deg"] = self._roll_max
        # Keep the legacy general-rotation field for historical comparability.
        # Its name did not make it a roll measurement: yaw alone increased it.
        from .body_layout import chest_axes,chest_orientation
        if self.scene_kind=='crib' and getattr(self,'_body_orientation_axes',None) is None:
            self._body_orientation_axes=chest_axes(R_now,d.xpos[self.model.body('head').id],
                                                  d.xpos[self.model.body('hip').id])
        orientation=chest_orientation(R_now,getattr(self,'_body_orientation_axes',None))
        if orientation['valid']:
            self._body_roll_max=max(getattr(self,'_body_roll_max',0.),abs(orientation['roll_degrees']))
        task['body_orientation']=dict(**orientation,maximum_roll_degrees=self._body_roll_max)
        # which weight box is MIMo touching right now (with any skin body)?
        if self._box_geoms:
            touched_cls = set()
            touched_by_hand = set()
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                for g_box, g_other in ((g1, g2), (g2, g1)):
                    cls = self._box_geoms.get(g_box)
                    if cls and int(self.model.geom_bodyid[g_other]) in self._mimo_body_set:
                        touched_cls.add(cls)
                        # hands only: "engagement" must be by hand or finger, not the
                        # torso lying against a box (found auditing the 90% box-contact result)
                        bname = self.model.body(int(self.model.geom_bodyid[g_other])).name
                        if "hand" in bname or any(k in bname for k in ("distal", "middle", "knuckle", "metacarpal", "thbase", "thhub")):
                            touched_by_hand.add(cls)
            task["box_touch"] = sorted(touched_cls)
            task["box_touch_hand"] = sorted(touched_by_hand)
        # jitter at PHYSICS timescale: fraction of joint velocities that reversed sign
        # since the last env step (the 50 ms smoothness EMA aliases a fast buzz away)
        qv = np.asarray(self.data.qvel[:40], dtype=np.float64)
        if self._jit_prev is not None:
            self._jit_ring.append(float(np.mean(np.sign(qv) != np.sign(self._jit_prev))))
            if len(self._jit_ring) > 100:
                self._jit_ring.pop(0)
        self._jit_prev = qv.copy()
        if self._jit_ring:
            task["jitter"] = float(np.mean(self._jit_ring))
        # The observations of 2026-08-24: do the arms ever CROSS the body, and do the
        # hands ever PRESS on anything? Both were absent from every video, both have
        # mechanical causes (mid-range-preferring wellness; a movement drive that punishes
        # the stillness a press requires), and neither was measured anywhere. Midline
        # crossing is computed against the shoulder line, so it is valid whatever way the
        # body is lying; a press is a hand patch above 2 kPa.
        try:
            pl = np.asarray(self.data.xpos[self.model.body("left_upper_arm").id])
            pr = np.asarray(self.data.xpos[self.model.body("right_upper_arm").id])
            lat = pr - pl
            n = np.linalg.norm(lat)
            if n > 1e-6:
                lat = lat / n
                mid = 0.5 * (pl + pr)
                lh = float(np.dot(np.asarray(self.data.xpos[self.model.body("left_hand").id]) - mid, lat))
                rh = float(np.dot(np.asarray(self.data.xpos[self.model.body("right_hand").id]) - mid, lat))
                # left hand belongs on the negative side, right on the positive
                task["hands_crossed"] = bool(lh > 0.03 or rh < -0.03)
            hp = [i2 for i2, nm in enumerate(self.patch_names())
                  if any(k in nm for k in ("hand", "distal", "middle", "knuckle",
                                           "metacarpal", "thbase", "thhub"))]
            if hp:
                task["hand_press_kpa"] = float(patches[hp, 0].max() / 1000.0)
                task["hand_pressing"] = bool(task["hand_press_kpa"] > 2.0)
                # the same press split into world and self (split_hand_press); the old keys
                # above stay exactly as they were
                # (absent when the pooler has no share for these rows: unmeasured, never
                # silently "all world")
                _own = getattr(self.pooler, "last_self", None)
                if _own is not None and len(_own) == len(patches):
                    task.update(split_hand_press(patches[hp, 0],
                                                 np.asarray(_own, dtype=np.float64)[hp, 0]))
        except Exception:
            pass
        self._hand_press_footprint(task)
        self._contact_report(contacts, task)
        # Is a hand where the eyes can see it? Measurement only, nothing feeds this back.
        # The point is to test a premise: a moving limb should be the MOST predictable thing
        # in the visual field, because the model has both the command that moved it and the
        # proprioception that reports it. If that is true, vision prediction error should be
        # lower on steps where a hand is in view than on steps where it is not.
        try:
            best = 180.0
            for cname in ("eye_left", "eye_right"):
                cid = self.model.camera(cname).id
                fwd = -np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[:, 2]
                origin = np.asarray(self.data.cam_xpos[cid])
                for hb in ("right_hand", "left_hand"):
                    to_hand = np.asarray(self.data.xpos[self.model.body(hb).id]) - origin
                    nrm = np.linalg.norm(to_hand)
                    if nrm > 1e-6:
                        best = min(best, float(np.degrees(np.arccos(
                            np.clip(np.dot(fwd, to_hand / nrm), -1, 1)))))
            # cheapest detector for the whole "the visual field is empty" class of bug:
            # the plank-posture defect (b52f68e4) left the eyes pitched -48 deg into the
            # floor for seven runs and nothing in any log would have shown it
            cid = self.model.camera("eye_left").id
            task["eye_pitch_deg"] = float(np.degrees(np.arcsin(np.clip(
                -np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[2, 2], -1, 1))))
            task["hand_view_deg"] = best
            task["hand_in_view"] = bool(best < 30.0)                 # the 60 deg wide field
            task["hand_in_fovea"] = bool(best < 0.5 * self.fovea_fov if self.fovea_size > 0 else False)
        except Exception:
            pass
        if self.reflex is not None:
            task["reflex_snaps"] = self.reflex.n_snaps      # cumulative; rate = delta/time
            task["reflex_active"] = self.reflex.active
            if self.ring_size > 0:
                task["ring_snaps"] = self.reflex.n_snaps_ring   # cumulative, subset of reflex_snaps
        if self._cr:
            cr = self._cr
            task["squeaker_hits"] = cr["sq_hits"]               # cumulative
            task["squeaker_hits_foot"] = cr["sq_hits_foot"]
            # the lift-head-and-look milestone (operator, 2026-09-06): a supine baby
            # cannot see the pedal at its feet (measured ~90 deg off the resting gaze
            # axis); bringing it into the wide cone REQUIRES lifting the head. Logged
            # every frame; the first success cuts a moment video.
            try:
                if cr.get("sq_home") is not None:
                    vdeg = self._angle_from_eye_deg(cr["sq_home"])
                    task["squeaker_view_deg"] = vdeg
                    jt = self.model.joint("robot:head_tilt").id
                    task["head_tilt_deg"] = float(np.degrees(
                        self.data.qpos[int(self.model.jnt_qposadr[jt])]))
                    seen = bool(vdeg < 30.0)
                    task["squeaker_seen"] = seen
                    if seen and not cr.get("headlook_latch"):
                        cr["headlook_latch"] = True
                        cr["headlook_events"] = cr.get("headlook_events", 0) + 1
                        cr["last_moment"] = "headlook"
                    elif not seen and vdeg > 45.0:
                        cr["headlook_latch"] = False    # re-arm after looking away
                    task["headlook_events"] = cr.get("headlook_events", 0)
            except Exception:
                pass
            task["critter_visitor"] = cr.get("visitor") or "none"
            if cr.get("visitor") == "dragonfly":
                task["dragonfly_deg"] = self._angle_from_eye_deg(self.data.mocap_pos[cr["df_id"]])
                task["dragonfly_hovering"] = cr.get("df_state") == "hover"
            if cr.get("visitor") == "bee":
                task["bee_deg"] = self._angle_from_eye_deg(self.data.mocap_pos[cr["bee_id"]])
                task["bee_landed"] = cr.get("bee_state") == "landed"
            task["dragonfly_darts"] = cr["df_darts"]            # cumulative
            if cr.get("last_moment"):
                task["moment"] = cr.pop("last_moment") or ""    # one-shot event for video capture
        if self.saccades is not None:
            task["spont_saccades"] = self.saccades.n_saccades
        if self._target_id is not None:
            hand = self.data.xpos[self.model.body("right_hand").id]
            ball = self.data.mocap_pos[self._target_id]
            task["hand_ball_dist_m"] = float(np.linalg.norm(hand - ball))
            # eyes-on-ball: is either eye camera actually POINTED at the ball? The gaze
            # training loss cannot answer this (its target is the eyes' own current angles,
            # a self-echo); this measures looking directly. MuJoCo cameras view along -z.
            try:
                best = 180.0
                for cname in ("eye_left", "eye_right"):
                    cid = self.model.camera(cname).id
                    fwd = -np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[:, 2]
                    to_ball = ball - np.asarray(self.data.cam_xpos[cid])
                    n = np.linalg.norm(to_ball)
                    if n > 1e-6:
                        ang = float(np.degrees(np.arccos(np.clip(np.dot(fwd, to_ball / n), -1, 1))))
                        best = min(best, ang)
                task["gaze_ball_deg"] = best
                task["gaze_on_ball"] = bool(best < 10.0)
                if self._onball_lock and best > 15.0:
                    self._onball_lock = False
                task["gaze_on_ball_earned"] = bool(best < 10.0 and not self._onball_lock)
                if self.caregiver_seconds > 0.0:
                    self._unseen_s = 0.0 if best < 30.0 else self._unseen_s + self.dt
                    if self._unseen_s >= self.caregiver_seconds:
                        self.randomize_target()          # an adult puts the toy back in view
                        self._unseen_s = 0.0
                        self._rescues += 1
                task["ball_unseen_s"] = self._unseen_s
                task["caregiver_rescues"] = self._rescues
            except Exception:
                pass
            tgt_geom = self.model.geom("target_geom").id
            touched, touched_hand = False, False
            for ci in range(self.data.ncon):
                c = self.data.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 != tgt_geom and g2 != tgt_geom:
                    continue
                touched = True
                other = g2 if g1 == tgt_geom else g1
                bname = self.model.body(int(self.model.geom_bodyid[other])).name
                # review fix 4: 'touched' used to count ANY contact (floor, torso, face),
                # which is why a limp body scored like a trained one
                if "hand" in bname or any(k in bname for k in ("distal", "middle", "knuckle", "metacarpal", "thbase", "thhub")):
                    touched_hand = True
            task["ball_touched"] = bool(touched)
            task["ball_touched_hand"] = bool(touched_hand)
        if self._mobile_geoms is not None:
            mb = np.asarray(d.geom_xpos[self._mobile_speed_geom], dtype=np.float64)
            if self._mobile_prev is not None:
                task["mobile_speed_cms"] = float(
                    np.linalg.norm(mb - self._mobile_prev) / max(self.dt, 1e-9) * 100.0)
            self._mobile_prev = mb.copy()
            _mt, _mh = False, False
            for ci in range(d.ncon):
                c = d.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                _in1, _in2 = g1 in self._mobile_geoms, g2 in self._mobile_geoms
                if not (_in1 or _in2):
                    continue
                if _in1 and _in2:
                    continue                      # shapes brushing their own frame
                _mt = True
                other = g2 if _in1 else g1
                bname = self.model.body(int(self.model.geom_bodyid[other])).name
                if "hand" in bname or any(k in bname for k in
                                          ("distal", "middle", "knuckle", "metacarpal",
                                           "thbase", "thhub")):
                    _mh = True
                _mpos = np.asarray(d.geom_xpos[g1 if _in1 else g2], dtype=np.float64)
            task["mobile_touched"] = bool(_mt)
            task["mobile_touched_hand"] = bool(_mh)
            # HOW CLOSE THE HANDS GET, not just whether they arrived. Hand strikes on the
            # mobile read 0 for whole runs, and 0 cannot say whether the hand came within a
            # finger's width or never left the floor. Surface-to-surface: centre distance
            # less each shape's bounding radius, floored at 0, so a touch reads 0.000 m.
            # Measured 2026-09-20 in a fresh crib scene: 0.190 m with the body at rest, and
            # 0.052 m at the best of 172 commanded postures.
            if self._hand_geoms:
                # SURFACE TO SURFACE, EXACTLY. This used to subtract each shape's BOUNDING
                # SPHERE radius from the distance between centres. A bounding sphere holds
                # the whole shape, so that number is a LOWER BOUND on the real gap and can
                # only ever under-report it - and for the mobile it under-reports enormously:
                # the string is a capsule 4 mm thick and 0.60 m long, whose bounding sphere
                # has a radius of 0.304 m, so any hand centre within 0.335 m of the string's
                # middle read a gap of exactly zero. On the live segment 24 of 43 windows
                # reported a closest approach of exactly 0.000 m and NOT ONE of those windows
                # recorded a hand touching the mobile. A gauge built to say whether the mobile
                # is within reach was answering that it was already being held.
                # The bound is kept as a cheap sort key: because it can only under-report, a
                # pair whose bound is further than the best exact distance found so far cannot
                # win, so the walk below stops there. Typically two or three exact calls.
                _hg = list(self._hand_geoms)
                _mg = sorted(self._mobile_geoms)
                _hp = np.asarray(d.geom_xpos[_hg], dtype=np.float64)
                _mp = np.asarray(d.geom_xpos[_mg], dtype=np.float64)
                _hr = np.asarray(self.model.geom_rbound[_hg], dtype=np.float64)
                _mr = np.asarray(self.model.geom_rbound[_mg], dtype=np.float64)
                _sep = (np.linalg.norm(_hp[:, None, :] - _mp[None, :, :], axis=-1)
                        - _hr[:, None] - _mr[None, :])
                _order = np.argsort(_sep, axis=None)
                _best = float('inf')
                for _flat in _order:
                    _i, _j = int(_flat) // len(_mg), int(_flat) % len(_mg)
                    if float(_sep[_i, _j]) >= _best:
                        break               # every remaining pair is further away than this
                    _exact = float(self._mj.mj_geomDistance(
                        self.model, d, int(_hg[_i]), int(_mg[_j]), max(_best, 2.0), None))
                    _best = min(_best, _exact)
                task["mobile_hand_gap_m"] = float(max(0.0, _best))
            self._mobile_chime(_mt, _mpos if _mt else None)
        # anticipation ground truth (gaze plan B11): the gaze-command-space direction of
        # the loudest REAL self-caused change on this frame - the moving mobile if it is
        # above twice the ambient floor, else the faster-moving hand. Sign convention
        # (-az_offset, +el_offset) is the one the interventional gaze probe calibrated
        # on this body (2026-09-05: fovea-on-hand 0.00 -> 0.13 under it).
        task["antic_ok"] = False
        try:
            # hand-position history advances EVERY frame, regardless of which target
            # wins (external review 2026-09-08 finding 3: updating it only in the
            # hand branch left it stale through mobile-priority intervals, and the
            # first hand frame after one divided the ACCUMULATED displacement by a
            # single frame's dt - an isolated replay read 1,000 cm/s for a hand
            # whose true last-frame speed was zero, steering the teacher wrong)
            if not hasattr(self, "_hand_prev"):
                self._hand_prev = {}
            hand_speed = {}
            for hb in ("right_hand", "left_hand"):
                hp = np.asarray(d.xpos[self.model.body(hb).id], dtype=np.float64)
                pv = self._hand_prev.get(hb)
                self._hand_prev[hb] = hp.copy()
                if pv is not None:
                    hand_speed[hb] = (hp, float(np.linalg.norm(hp - pv)
                                                / max(self.dt, 1e-9) * 100.0))
            tgt_pos, best_sp = None, 0.0
            if task.get("mobile_speed_cms", 0.0) > 4.0 and self._mobile_speed_geom is not None:
                tgt_pos = np.asarray(d.geom_xpos[self._mobile_speed_geom])
                best_sp = task["mobile_speed_cms"]
            else:
                for hb, (hp, sp) in hand_speed.items():
                    if sp > 5.0 and sp > best_sp:
                        tgt_pos, best_sp = hp, sp
            if tgt_pos is not None:
                cid = self.model.camera("eye_left").id
                R = np.asarray(d.cam_xmat[cid]).reshape(3, 3)
                fwd, right, up = -R[:, 2], R[:, 0], R[:, 1]
                v = tgt_pos - np.asarray(d.cam_xpos[cid])
                n = np.linalg.norm(v)
                if n > 1e-6:
                    v = v / n
                    az_off = float(np.arctan2(np.dot(v, right), np.dot(v, fwd)))
                    el_off = float(np.arctan2(np.dot(v, up),
                                              np.hypot(np.dot(v, right), np.dot(v, fwd))))
                    (ly, lp), (ry, rp) = self.eye_yaw_pitch()
                    yaw, pitch = 0.5 * (ly + ry), 0.5 * (lp + rp)
                    task["antic_az"] = float(np.clip(yaw - az_off, -0.785, 0.785))
                    task["antic_el"] = float(np.clip(pitch + el_off, -0.820, 0.576))
                    task["antic_ok"] = True
                    task["antic_speed_cms"] = best_sp
        except Exception:
            pass
        info = {"task": task}
        if self._audio is not None:
            # crib hearing (lane C): impacts from this frame's collision list (the touch
            # pipeline's own dicts: point_cm, impulse_ns), the bee's continuous buzz,
            # squeaks pushed by the squeaker hit. Rendered at the head every step.
            try:
                # only NEW, forceful contacts make a sound. Measured (2026-09-06,
                # collision_stats.py): a resting body carries ~9 persistent contacts
                # per step and the old every-step feed fired 355 thumps/second - THE
                # machine gun. Fresh contacts run 58/s but their impulse p90 is
                # 0.078; real strikes land above 0.2. Gate: frame_count <= 1 AND
                # impulse > 0.15.
                for c0 in (cols or []):
                    if int(c0.get("frame_count", 99)) <= 1:
                        _imp = float(np.linalg.norm(np.asarray(c0["impulse_ns"], float)))
                        if _imp > 0.15:
                            self._audio.impact(np.asarray(c0["point_cm"], float) / 100.0, _imp)
                cida = self.model.camera("eye_left").id
                Ra = np.asarray(d.cam_xmat[cida]).reshape(3, 3)
                # only the PRESENT visitor sounds - and a landed bee is silent
                # (bees stop buzzing when perched: the silence IS the landing cue)
                cr0 = self._cr or {}
                bee = (np.asarray(d.mocap_pos[cr0["bee_id"]])
                       if cr0.get("visitor") == "bee" and cr0.get("bee_state") != "landed"
                       else None)
                dfly = (np.asarray(d.mocap_pos[cr0["df_id"]])
                        if cr0.get("visitor") == "dragonfly" else None)
                # dt is passed LIVE, as it is to proprio.read and build_collisions above.
                # CribAudio stored the construction-time dt; hearing was the only sense in
                # this method that did not re-read env.dt, so a step length changed after
                # construction advanced the speech cursor at the construction rate.
                a_spec, a_meta = self._audio.render(np.asarray(d.cam_xpos[cida]),
                                                    Ra[:, 0], -Ra[:, 2], buzz_pos=bee,
                                                    flutter_pos=dfly, dt=self.dt)
                info["audio_spec"] = a_spec
                info["audio_meta"] = a_meta
            except Exception as error:
                if self.caregiver is not None:
                    raise RuntimeError("caregiver hearing failed; speech cannot be replaced by silent input") from error
                logging.getLogger("mimo_bridge.env").warning("crib hearing unavailable: %s", error)
        ang = self.eye_yaw_pitch()
        gaze_az = 0.5 * (ang[0][0] + ang[1][0])
        gaze_el = 0.5 * (ang[0][1] + ang[1][1])
        tpos = (d.xpos[self._touch_body_ids] - d.xpos[self._root_body_id]).astype(np.float32)
        from .skin_contract import measured_contact_centres
        if self.touch_geometry == "surface":
            tpos = measured_contact_centres(self._touch_body_ids, tpos,
                                            d.xpos[self._root_body_id], contacts)
            tpos = (tpos @ np.asarray(d.xmat[self._root_body_id]).reshape(3, 3)).astype(np.float32)
            info['skin_geometry'] = 'contact_centres_root_axes'
        else:
            info['skin_geometry'] = 'body_centres_world_axes_at_root'
        if self.caregiver is not None:
            info["caregiver"] = {"mode": self.caregiver.mode,
                                  "replies": self.caregiver.state["replies"],
                                  "object_style": self.caregiver.state["style"],
                                  "events": len(self.caregiver.state["events"])}
            info['caregiver'].update(self.caregiver.diagnostics(self))
            tape = self.caregiver.state['events']
            if hasattr(tape, 'manifest'):
                info['caregiver']['recording'] = dict(path=str(tape.manifest),
                    free_bytes=tape.free_bytes, dropped=tape.dropped, error=tape.error)
        info['body_layout'] = self.body_layout
        from .action_decoder import MOTOR_CONTRACT
        info['motor_contract'] = MOTOR_CONTRACT
        if getattr(self, 'finger_joint_servo', 'off') == 'on':
            # --finger-joint-servo on: the finger command means each hinge's own goal (action_decoder.py); reported
            # only then, so a world on the default reports exactly what it always has. The transition's action
            # contract and replay's contract bit are read from it (limb_target_control.action_contract_parts,
            # moment_contract).
            from .action_decoder import FINGER_SERVO_CONTRACT
            info['finger_servo'] = FINGER_SERVO_CONTRACT
            info['motor_contract'] = MOTOR_CONTRACT + '+' + FINGER_SERVO_CONTRACT
        if getattr(self, 'position_rows', 'all') != 'all':
            # the decoder solves no position for the head and pelvis (audit F3); reported only then, so a
            # world on the default reports exactly what it always has
            info['position_rows'] = self.position_rows
        _limbs = getattr(self, 'limb_target_control', None)
        if _limbs is not None and _limbs.calves:
            # the knees as the learner's planner needs them to imagine the calf rows this world records
            # (limb_target_control.calf_rows_torch); only when the calves are projected
            info['limb_target_geometry'] = {'calf': _limbs.calf_geometry()}
        if self.body_function is not None:
            sets=self.proprio.act_set
            present=np.array([np.any(sets==i) for i in range(len(self.proprio.fatigue.cap))])
            availability=(self.proprio.fatigue.cap if self.body_function_mode=='modulate'
                          else np.ones_like(self.proprio.fatigue.cap))
            info['motor_function']=dict(contract='motor_function_v1',values=availability.tolist(),
                valid=present.tolist(),mode=self.body_function_mode,
                source='fixed actuator gain availability; fatigue rates are engineering assumptions')
        if self.body_function is not None:
            from lem.body_function import physical_drives
            if virtual is None:
                load,stim = physical_drives(patches[:,0],exposure,self.touch_sensitivity)
                self.body_function.advance(load,stim,float(d.time))
            else:
                values=virtual['values'];load,stim=values[:,1],values[:,2]
                self.body_function.quality=values[:,0].astype(float)
                self.body_function.last_load=load.copy();self.body_function.last_stimulation=stim.copy()
                self.body_function.last_time=float(d.time)
            info['body_function'] = self.body_function.record()
            info['body_function']['mode'] = self.body_function_mode
            if self.body_function_mode == 'modulate':
                # Physical fields remain available only as external audit evidence.
                # All learner touch routes receive the same experienced observation.
                info['physical_touch_audit'] = patches.copy()
                if virtual is None:patches = self.body_function.observe(patches)
                else:patches[:,:2]=virtual['experienced'][:,:2]
            else:
                # Observation-only calculation has not changed actual sensor function.
                info['body_function']['shadow_values'] = info['body_function']['values']
                info['body_function']['values'] = np.stack([
                    np.ones(len(load)),load,stim],axis=-1).tolist()
        if self.contact_profile in ('mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
            from lem.skin_mechanics import MECHANICAL_CONTRACT
            info['mechanical_contact'] = dict(contract=MECHANICAL_CONTRACT,
                force_distribution='normalized geometric weights conserve full force vector',
                history=exposure.tolist(), area_source='active sensor support area; not resolved tissue footprint',
                sampling_seconds=float(self.model.opt.timestep), pressure_peak_pa=float(exposure[:,0].max()),
                pressure_exposure_pa_s=float(exposure[:,1].max()),
                rubbing_work_j_m2=float(exposure[:,2].max()))
            if virtual is not None:
                from .contact_footprint import CONTRACT
                if self.contact_profile in ('mechanical_v3','mechanical_v4'):
                    info['mechanical_contact'].update(contract=CONTRACT,area_source='per-contact virtual elastic geometry; unresolved geometry explicitly falls back to sensor support',
                        virtual_skin_profile=virtual['profile'],supported_contacts=virtual['supported_contacts'],
                        unresolved_geometry_contacts=virtual['unsupported_contacts'],
                        contact_footprints=self._virtual_skin.contact_records)
                    guard_report=self.footprint_guard_report()
                    if guard_report is not None:info['mechanical_contact']['footprint_guard']=guard_report
                else:
                    info['mechanical_contact'].update(area_source='unchanged v2 active sensor support area; not a resolved tissue footprint',
                        localization='existing active taxels receive unchanged v2 body-row pressure and shear')
                rubbing_report=self.rubbing_calibration_report(patches)
                if rubbing_report is not None:info['mechanical_contact']['rubbing_calibration']=rubbing_report
                info['skin_function_granularity']=dict(taxels=virtual['taxels'],
                    affected_sensor_area_fraction=virtual['affected_sensor_area_fraction'],
                    area_weighted_quality=virtual['area_weighted_quality'],body_summary='worst local taxel; not uniform whole-body impairment')
                moments=[];indices=np.triu_indices(3)
                root=np.asarray(d.xmat[self._root_body_id]).reshape(3,3)
                for spec,body_moments in zip(self.pooler.specs,virtual['surface_moments_body']):
                    row=np.asarray(body_moments).copy();rotation=root.T@np.asarray(d.xmat[spec.body_id]).reshape(3,3)
                    origin=root.T@(d.xpos[spec.body_id]-d.xpos[self._root_body_id])
                    for offset,flag in ((0,18),(9,19)):
                        if row[flag]:
                            row[offset:offset+3]=rotation@row[offset:offset+3]+origin
                            covariance=np.zeros((3,3));covariance[indices]=row[offset+3:offset+9]
                            covariance=covariance+covariance.T-np.diag(np.diag(covariance))
                            row[offset+3:offset+9]=(rotation@covariance@rotation.T)[indices]
                    moments.append(row.tolist())
                info['skin_function_granularity'].update(surface_moments_root=moments,
                    surface_moment_fields='sensed pressure centroid xyz m, covariance upper triangle m2; affected-skin centroid/covariance; two validity flags',
                    surface_scope='within-body first and second spatial moments; separated contacts may remain ambiguous beyond covariance')
                info['skin_function_granularity']['mode']=self.body_function_mode
                if self.body_function_mode!='modulate':
                    info['skin_function_granularity'].update(shadow_affected_sensor_area_fraction=virtual['affected_sensor_area_fraction'],
                        shadow_area_weighted_quality=virtual['area_weighted_quality'],
                        affected_sensor_area_fraction=[0.]*len(self.pooler.specs),area_weighted_quality=[1.]*len(self.pooler.specs))
                    for row in info['skin_function_granularity']['surface_moments_root']:
                        row[9:18]=[0.]*9;row[19]=0.
                if 'body_function' in info:info['body_function']['scope']='per-taxel pressure/shear operator; body rows summarize worst local precision'
        info['physical_state_finite'] = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
                                            and np.isfinite(patches).all())
        info['hand_target_contract'] = ('world_v0' if self.hand_target_control is None
                                        else self.hand_target_control.geometry['contract'])
        if self.hand_target_control is not None:
            info['hand_geometry'] = self.hand_target_control.geometry
            info['hand_joint_angles'] = np.array(d.qpos[self.hand_target_control._joint_addresses])
            info['hand_joint_velocities'] = np.array(d.qvel[self.hand_target_control._joint_dof_addresses])
        return RawStep(frame=self.frame, sim_time=float(d.time), dt=self.dt, patches=patches, collisions=cols,
                       proprio=pro, vestibular=vest, eye_left=eyes[0], eye_right=eyes[1],
                       fovea_left=fovea[0], fovea_right=fovea[1], eye_angles=ang,
                       gaze_az_el_deg=(float(np.degrees(gaze_az)), float(np.degrees(gaze_el))),
                       n_contacts=int(d.ncon), ctrl=self.ctrl.copy(), info=info,
                       touch_pos=tpos, touch_self=getattr(self.pooler, "last_self", None),
                       touch_contact=getattr(self.pooler, "last_contact", None),
                       touch_sensitivity=self.touch_sensitivity)

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        if getattr(self, 'caregiver', None) is not None:
            self.caregiver.close_recording()
        try:
            self.env.close()
        except Exception:
            pass
        if getattr(self, "_watch", None) is not None:
            try:
                self._watch.close()
            except Exception:
                pass
            self._watch = None
        # drop the big MuJoCo objects so the memory goes back right away (rebuilds)
        self.env = None
        self.model = None
        self.data = None
        self.touch = None
        self.proprio = None
        self.decoder = None
