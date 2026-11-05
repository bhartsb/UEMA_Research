# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The eye and tactile fixtures (apple/scripts/export_eye_fixtures.py, WI-14) hold what the Swift senses port needs.

Without a world: the state parser, the 256-slot padding rule (tokenizer/config.py:51; frame_encoder.py:119-120,
240), the object-flag rule on a three-geom MuJoCo model (caregiver.py:304-319), the released args.json mapped to
BridgeEnv arguments (train_mimo.py:1402-1456) and the PNG size reader. With the fixtures on disk (skipped with the
command that writes them when they are absent): three states, five PNGs each at 256 / 288 / 256 px, N 1,160 with
four angle rows of four numbers, tactile fields [53, 10] inside [256, 10] with zero padding, the model check
MATCH, every object's root within 1e-6 m of the recorded pose (the mobile's mount moved to it), the three
caregiver phases absent / presenting / engaged. Run from Source/Python:
    python -m pytest ../../apple/scripts/tests/test_export_eye_fixtures.py -p no:cacheprovider -q
"""
import json
import os
import struct
import sys
import zlib

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import export_eye_fixtures as eef  # noqa: E402


def test_parse_states_reads_chunk_row_pairs_in_order():
    assert eef.parse_states("0:0,8:74, 11:80") == [(0, 0), (8, 74), (11, 80)]
    assert eef.parse_states(eef.DEFAULT_STATES) == [(0, 0), (8, 74), (11, 80)]
    with pytest.raises(ValueError):
        eef.parse_states("3")
    with pytest.raises(ValueError):
        eef.parse_states("0:-1")
    with pytest.raises(ValueError):
        eef.parse_states(" ")


def test_pad_to_slots_appends_zero_rows_to_256():
    fields = np.arange(53 * 10, dtype=np.float32).reshape(53, 10)
    padded = eef.pad_to_slots(fields)
    assert padded.shape == (256, 10) and padded.dtype == np.float32
    assert np.array_equal(padded[:53], fields) and not padded[53:].any()
    with pytest.raises(ValueError):
        eef.pad_to_slots(np.zeros((257, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        eef.pad_to_slots(np.zeros((5, 9), dtype=np.float32))


def test_object_flags_follow_the_caregiver_rule():
    """caregiver.py:304-319: hidden -> alpha 0 and no collision; shown -> the build's alpha or 1 where the build
    left 0; collidable -> the build's contype / conaffinity or 1 where the build left 0."""
    mujoco = pytest.importorskip("mujoco")
    xml = """
    <mujoco><worldbody>
      <body name="toy"><geom name="a" type="sphere" size="0.01" rgba="1 0 0 0" contype="0" conaffinity="0"/>
                       <geom name="b" type="sphere" size="0.01" rgba="0 1 0 0.5" contype="1" conaffinity="1"/></body>
      <body name="other"><geom name="c" type="sphere" size="0.01" rgba="0 0 1 1" contype="3" conaffinity="2"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    objects = [dict(name="toy", geoms=[0, 1]), dict(name="other", geoms=[2])]
    build_alpha = {g: float(m.geom_rgba[g, 3]) for g in range(3)}
    build_collide = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(3)}
    rows = eef.apply_object_flags(m, objects, [3, 0], build_alpha, build_collide)
    assert [r["visible"] for r in rows] == [True, False] and [r["collidable"] for r in rows] == [True, False]
    assert m.geom_rgba[0, 3] == 1.0 and m.geom_rgba[1, 3] == 0.5 and m.geom_rgba[2, 3] == 0.0
    assert (int(m.geom_contype[0]), int(m.geom_conaffinity[0])) == (1, 1)
    assert (int(m.geom_contype[1]), int(m.geom_conaffinity[1])) == (1, 1)
    assert (int(m.geom_contype[2]), int(m.geom_conaffinity[2])) == (0, 0)
    rows = eef.apply_object_flags(m, objects, [1, 2], build_alpha, build_collide)
    assert (int(m.geom_contype[0]), int(m.geom_conaffinity[0])) == (0, 0) and m.geom_rgba[0, 3] == 1.0
    assert (int(m.geom_contype[2]), int(m.geom_conaffinity[2])) == (3, 2) and m.geom_rgba[2, 3] == 0.0


def test_released_args_map_to_the_recorded_world_settings():
    path = os.path.join(eef.RELEASED_RUN, "args.json")
    if not os.path.exists(path):
        pytest.skip("no released args.json at %s" % path)
    with open(path, encoding="utf-8") as f:
        a = json.load(f)
    kw = eef.env_kwargs_from_args(a, "scenes", "speech", "_t")
    assert kw["scene"] == "crib" and kw["age"] == 24.0 and kw["body_layout"] == "trunk_v1"
    assert kw["fovea_size"] == 288 and kw["fovea_fov"] == 24.0 and kw["ring_fov"] == 150.0
    assert kw["mobile"] is True and kw["critters"] is True and kw["audio_real"] is True
    assert kw["caregiver_mode"] == "responsive" and kw["caregiver_speech_dir"] == "speech"
    assert kw["caregiver_record_dir"] == "" and kw["run_dir"] == "scenes" and kw["tag"] == "_t"
    assert kw["contact_profile"] == "mechanical_v4" and kw["touch_inside_rays"] == "fixed_v1"
    assert kw["hand_target_mode"] == "wrist_relative_v1" and kw["finger_joint_servo"] == "on"
    assert kw["cmd_horizon"] == 25 and kw["render_every"] == 2 and kw["vision"] is True
    # settings at their defaults are left out, as train_mimo.env_kwargs_for leaves them (continuation_state)
    for name in ("parent_relative_targets", "reach_projection", "position_rows", "contact_footprint_guard",
                 "rubbing_calibration"):
        assert name not in kw


def test_mounted_objects_move_to_the_recorded_root_pose():
    """place_mounted_objects: a body hung from the world without a joint is moved to the record when forward
    kinematics places it beyond 1e-5 m; a free body and a mocap body are left alone."""
    mujoco = pytest.importorskip("mujoco")
    xml = """
    <mujoco><worldbody>
      <body name="mount" pos="0 0 1"><geom type="sphere" size="0.01"/></body>
      <body name="box" pos="0.1 0 0"><freejoint/><geom type="sphere" size="0.01"/></body>
      <body name="puppet" pos="0.2 0 0" mocap="true"><geom type="sphere" size="0.01"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    objects = [dict(name="mount", root=1, mocap=-1, free=False), dict(name="box", root=2, mocap=-1, free=True),
               dict(name="puppet", root=3, mocap=0, free=False)]
    pose = np.array([[0.06, 0.0, 1.75, 1.0, 0.0, 0.0, 0.0], [0.1, 0, 0, 1, 0, 0, 0], [0.2, 0, 0, 1, 0, 0, 0]])
    rows = eef.place_mounted_objects(mujoco, m, d, objects, pose)
    assert [r["name"] for r in rows] == ["mount"] and rows[0]["root_moved"] is True
    assert abs(rows[0]["position_difference_before_m"] - 0.7524) < 1e-3 and rows[0]["position_difference_after_m"] < 1e-9
    assert np.allclose(d.xpos[1], [0.06, 0.0, 1.75])
    again = eef.place_mounted_objects(mujoco, m, d, objects, pose)
    assert again[0]["root_moved"] is False and again[0]["position_difference_before_m"] < 1e-9


def _png_bytes(width: int, height: int) -> bytes:
    """A minimal valid RGB PNG of the given size (one IDAT of zero rows)."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x00" * (3 * width) for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_png_size_reads_the_ihdr(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(_png_bytes(288, 256))
    assert eef.png_size(str(p)) == (288, 256)
    q = tmp_path / "y.png"
    q.write_bytes(b"not a png")
    with pytest.raises(ValueError):
        eef.png_size(str(q))


@pytest.fixture(scope="module")
def state_dirs():
    if not os.path.isdir(eef.DEFAULT_OUT):
        pytest.skip("no %s: run python ../../apple/scripts/export_eye_fixtures.py from Source/Python first"
                    % eef.DEFAULT_OUT)
    # pose folders only: the vision tests keep a sibling "textures" folder here (the SceneKit renderer's inputs)
    dirs = sorted(d for d in os.listdir(eef.DEFAULT_OUT)
                  if d.startswith("step_") and os.path.isdir(os.path.join(eef.DEFAULT_OUT, d)))
    if not dirs:
        pytest.skip("no state folders under %s" % eef.DEFAULT_OUT)
    return [os.path.join(eef.DEFAULT_OUT, d) for d in dirs]


def _load(d, name):
    with open(os.path.join(d, name), encoding="utf-8") as f:
        return json.load(f)


def test_three_states_with_five_pngs_at_the_stated_sizes(state_dirs):
    assert len(state_dirs) == 3
    for d in state_dirs:
        cams = _load(d, "cameras.json")
        assert cams["format"] == eef.FORMAT_CAMERAS and cams["world"]["model_check"]["match"] is True
        assert cams["world"]["model_check"]["built"]["nq"] == 146
        assert [c["name"] for c in cams["cameras"]] == [c for c, _ in eef.CAMERA_FILES]
        for c in cams["cameras"]:
            want = 288 if c["name"].startswith("fovea") else 256
            assert c["size_px"] == want and eef.png_size(os.path.join(d, c["png"])) == (want, want)
            assert c["fovy_deg"] == {"eye": 60.0, "fovea": 24.0, "ring": 150.0}[c["name"].split("_")[0]]
            assert len(c["pos_m"]) == 3 and len(c["xmat_row_major"]) == 9
        assert len(cams["state"]["qpos"]) == 146 and os.path.basename(d) == cams["state"]["name"]
        assert cams["state"]["name"] == "step_%09d" % cams["state"]["step"]


def test_states_are_rebuilt_at_the_recorded_poses(state_dirs):
    """Every object's root sits within 1e-6 m of the record (quantum 1e-6 m, observer_record.py:336-346); the
    mobile's mount was moved there; the three states are the adult absent, presenting, engaged."""
    phases = []
    for d in state_dirs:
        st = _load(d, "cameras.json")["state"]
        assert max(o["position_difference_m"] for o in st["object_pose_check"]) <= 1e-6
        mounted = {r["name"]: r for r in st["mounted_objects"]}
        assert mounted["mobile"]["root_moved"] is True and mounted["mobile"]["position_difference_after_m"] <= 1e-6
        assert st["qvel"].startswith("zero") and len(st["mocap_pos_m"]) == 8 and len(st["mocap_quat_wxyz"]) == 8
        assert len(st["object_flags"]) == 13 and all(len(o["geoms"]) == len(o["alpha_written"]) for o in st["object_flags"])
        phases.append((st["caregiver_present"], st["caregiver_phase"]))
    assert phases == [(False, "absent"), (True, "presenting"), (True, "engaged")]


def test_patch_count_is_1160_with_four_angle_rows(state_dirs):
    for d in state_dirs:
        p = _load(d, "patches.json")
        assert p["format"] == eef.FORMAT_PATCHES
        assert p["n_total"] == eef.EXPECTED_N == 2 * p["n_per_eye"]
        assert [s["patches_per_eye"] for s in p["sources"]] == [256, 324]
        assert p["patches_shape"] == [1160, 768] and p["angles_shape"] == [1160, 4]
        rows = p["first_angle_rows"]
        assert len(rows) == 4 and all(len(r) == 4 for r in rows)
        assert [r[2] for r in rows] == [-1.0, 1.0, -1.0, 1.0]           # eyes interleaved
        # the angle table is float32 (vision_adapter.py:205), the source span float64: one float32 step at
        # 0.065 rad is 2^-27 = 7.5e-9 rad, so 1e-6 rad allows 134 of them
        assert all(abs(r[3] - p["sources"][0]["span_rad"]) < 1e-6 for r in rows)
        head = p["first_patch_row_head"]
        assert len(head) == 12 and all(0.0 <= x <= 1.0 for x in head)   # pixel / 255
        assert len(p["patches_sha256"]) == 64 and len(p["angles_sha256"]) == 64


def test_tactile_fields_are_53_rows_inside_256_slots(state_dirs):
    for d in state_dirs:
        t = _load(d, "tactile_patches.json")
        assert t["format"] == eef.FORMAT_TACTILE
        fields = np.asarray(t["fields"], dtype=np.float32)
        padded = np.asarray(t["fields_padded"], dtype=np.float32)
        assert fields.shape == (53, 10) and padded.shape == (256, 10) and t["n_patches"] == 53
        assert np.array_equal(padded[:53], fields) and not padded[53:].any()
        assert np.isfinite(fields).all() and (fields[:, 8:] == 1.0).all() and (fields[:, :4] >= 0).all()
        assert [r["row"] for r in t["patches"]] == list(range(53))
        body_ids = [r["body_id"] for r in t["patches"]]
        assert body_ids == sorted(body_ids) and len(set(body_ids)) == 53   # TouchPooler sorts by body id
        assert t["inputs"]["n_contacts"] == len(t["inputs"]["contacts"])
        assert len(t["field_names"]) == len(t["field_units"]) == 10
