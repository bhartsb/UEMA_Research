# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tests for apple/scripts/export_scene_bundle.py (WI-15).

Run from D:/UEMA_Research/Source/Python with the repository's interpreter:
    python -m pytest ../../apple/scripts/test_export_scene_bundle.py -p no:cacheprovider -q
The export test loads three MjModels (the fresh-folder bundle, the trainer's reference scene and
the test's own fresh copy); the tests that need the shipped weights folder or the mimo_infant
package skip, naming the missing piece, where either is absent.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import export_scene_bundle as esb  # noqa: E402

EXPECTED_NQ = 146          # observer header of the shipped run, world b0: "nq": 146
EXPECTED_FILES = ("scene.xml", "MIMo_metav2_lite.xml", "MIMo_modelv2_grown24_fovea.xml",
                  "play_mat.png", "ceiling.png", "pen_wall_n.png", "pen_wall_s.png", "pen_wall_e.png", "pen_wall_w.png")
# pen_art.py:25 (walls 2048 x 342, "6:1, matching the 1.44 m x 0.24 m wall panels"; the east and west
# panels are turned on their side), :167 (play mat 1024) and :215 (ceiling 1024)
EXPECTED_TEXTURE_SIZES = {"play_mat.png": [1024, 1024], "ceiling.png": [1024, 1024],
                          "pen_wall_n.png": [2048, 342], "pen_wall_s.png": [2048, 342],
                          "pen_wall_e.png": [342, 2048], "pen_wall_w.png": [342, 2048]}
PATH_ATTRIBUTE = re.compile(r'(?:file|meshdir|texturedir|assetdir)="([^"]*)"')
DRIVE_OR_ROOT = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/])|\.\.[\\/]")
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40

needs_weights = pytest.mark.skipif(not esb.RELEASED_ARGS.is_file(),
                                   reason=f"{esb.RELEASED_ARGS} is absent (fetch_weights.py)")


def released_args() -> dict:
    return json.loads(esb.RELEASED_ARGS.read_text(encoding="utf-8"))


@needs_weights
def test_released_configuration_is_read_from_args_json():
    cfg = esb.configuration_from_args_json(esb.RELEASED_ARGS)
    assert cfg == esb.CribReleaseConfiguration()


@needs_weights
@pytest.mark.parametrize("key, value, named", [
    ("scene", "reach", "scene"),
    ("body_layout", "legacy", "body_layout"),
    ("age_binding", "fixed:12", "age_months"),
    ("age_binding", "phase", "age_binding"),
])
def test_configuration_rejects_a_value_the_bundle_name_does_not_promise(tmp_path, key, value, named):
    args = released_args()
    args[key] = value
    path = tmp_path / "args.json"
    path.write_text(json.dumps(args), encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        esb.configuration_from_args_json(path)
    assert named in str(failure.value)


def test_configuration_names_a_missing_key(tmp_path):
    path = tmp_path / "args.json"
    path.write_text(json.dumps({"scene": "crib"}), encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        esb.configuration_from_args_json(path)
    assert "age_binding" in str(failure.value) and "ring_fov" in str(failure.value)


def test_absolute_path_pattern_catches_drive_root_and_parent_steps(tmp_path):
    xml = tmp_path / "t.xml"
    xml.write_text('<mujoco><a f="C:/x/y.png"/><b f="/x.png"/><c f="../x.png"/><d f="x.png"/><e f="sub/x.png"/></mujoco>',
                   encoding="utf-8")
    found = esb.absolute_path_attributes(xml)
    assert found == ["a@f=C:/x/y.png", "b@f=/x.png", "c@f=../x.png"]


def test_find_includes_needs_one_meta_and_one_model_include():
    import xml.etree.ElementTree as ET

    both = ET.fromstring('<mujoco><include file="C:/a/MIMo_metav2_lite_p1.xml"/>'
                         '<worldbody><body><include file="C:/a/MIMo_modelv2_fovea.xml"/></body></worldbody></mujoco>')
    found = esb.find_includes(both)
    assert found["meta"].get("file").endswith("MIMo_metav2_lite_p1.xml")
    assert found["model"].get("file").endswith("MIMo_modelv2_fovea.xml")
    with pytest.raises(ValueError) as failure:
        esb.find_includes(ET.fromstring('<mujoco><include file="MIMo_metav2.xml"/></mujoco>'))
    assert "['meta']" in str(failure.value)


def test_git_head_resolves_loose_ref_packed_ref_detached_head_and_worktree_file(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (repo / ".git" / "refs" / "heads" / "main").write_text(COMMIT_A + "\n", encoding="utf-8")
    assert esb.git_head(repo) == COMMIT_A

    (repo / ".git" / "refs" / "heads" / "main").unlink()
    (repo / ".git" / "packed-refs").write_text(f"# pack-refs with: peeled\n{COMMIT_B} refs/heads/main\n", encoding="utf-8")
    assert esb.git_head(repo) == COMMIT_B

    (repo / ".git" / "HEAD").write_text(COMMIT_A + "\n", encoding="utf-8")
    assert esb.git_head(repo) == COMMIT_A

    # a worktree: .git is a file naming its private git folder, whose commondir names the shared one
    worktree = tmp_path / "wt"
    private = repo / ".git" / "worktrees" / "wt"
    private.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {private}\n", encoding="utf-8")
    (private / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (private / "commondir").write_text("../..\n", encoding="utf-8")
    assert esb.git_head(worktree) == COMMIT_B

    (repo / ".git" / "HEAD").write_text("ref: refs/heads/absent\n", encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        esb.git_head(repo)
    assert "refs/heads/absent" in str(failure.value)


def test_compare_models_names_the_differing_fields_only():
    mujoco = pytest.importorskip("mujoco")
    a = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b"><geom name="g" size="0.1"/></body></worldbody></mujoco>')
    b = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b"><geom name="g" size="0.2"/></body></worldbody></mujoco>')
    fields = ("nbody", "ngeom", "geom_size", "body_mass", "names")
    assert esb.compare_models(a, a, fields) == []
    assert esb.compare_models(a, b, fields) == ["geom_size", "body_mass"]


def test_captured_mujoco_warnings_collects_routed_messages_and_restores_the_hook():
    import math

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="hinge"/><geom size="0.1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    data.qpos[0] = math.nan            # mj_step reports mjWARN_BADQPOS through mju_warning on its first occurrence
    assert mujoco.get_mju_user_warning() is None
    with esb.captured_mujoco_warnings() as seen:
        assert mujoco.get_mju_user_warning() is not None
        mujoco.mj_step(model, data)
    assert len(seen) == 1 and seen[0].startswith("Nan, Inf or huge value in QPOS"), seen
    assert data.warning[int(mujoco.mjtWarning.mjWARN_BADQPOS)].number == 1
    assert mujoco.get_mju_user_warning() is None
    with esb.captured_mujoco_warnings() as again:
        pass
    assert again == []


@needs_weights
def test_export_writes_relative_bundle_that_loads_with_nq_146(tmp_path):
    pytest.importorskip("mimo_infant")
    import mujoco

    out = tmp_path / "bundle"
    out.mkdir()
    foreign = out / "foreign.txt"
    foreign.write_text("left alone by the exporter", encoding="utf-8")
    manifest = esb.export(out, tmp_path / "work", esb.CribReleaseConfiguration())

    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(EXPECTED_FILES + ("bundle.json", "foreign.txt"))
    assert foreign.read_text(encoding="utf-8") == "left alone by the exporter"
    for name in ("scene.xml", "MIMo_metav2_lite.xml", "MIMo_modelv2_grown24_fovea.xml"):
        text = (out / name).read_text(encoding="utf-8")
        assert esb.absolute_path_attributes(out / name) == [], name
        for value in PATH_ATTRIBUTE.findall(text):
            assert not DRIVE_OR_ROOT.search(value), (name, value)
        assert "<mesh" not in text, name
    assert "MIMo_metav2_lite.xml" in (out / "scene.xml").read_text(encoding="utf-8")
    assert "MIMo_modelv2_grown24_fovea.xml" in (out / "scene.xml").read_text(encoding="utf-8")

    # the manifest hashes name the files on disk; no machine path is written into it
    written = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert written["model_counts"] == manifest["model_counts"]
    for rec in written["files"]:
        assert rec["bytes"] == (out / rec["name"]).stat().st_size
        assert rec["sha256"] == esb.sha256_of(out / rec["name"])
    assert [rec["name"] for rec in written["files"]] == list(EXPECTED_FILES)
    assert written["play_mat_size_px"] == [1024, 1024]
    assert written["texture_sizes_px"] == EXPECTED_TEXTURE_SIZES
    assert written["not_shipped"]["mimo_texture_png_files"] == 25
    assert written["not_shipped"]["mesh_files_referenced"] == 0
    assert written["reference_check"]["equal"] is True
    assert written["observer_header_match"] == {"compared": ["nq", "nv", "nu", "nmocap", "nbody"], "equal": True}
    assert written["fresh_folder_load"]["mju_user_warning_messages"] == []
    for source in written["sources"].values():
        assert source["path"].startswith("mimo_infant/"), source
        assert not DRIVE_OR_ROOT.search(source["path"]), source
    assert re.fullmatch(r"[0-9a-f]{40}", written["generator"]["repository_commit"])

    # a fresh folder holding only the bundle's files loads
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    for name in EXPECTED_FILES:
        shutil.copyfile(out / name, fresh / name)
    model = mujoco.MjModel.from_xml_path(str(fresh / "scene.xml"))
    assert model.nq == EXPECTED_NQ
    assert model.nq == written["model_counts"]["nq"]
    for name in ("fovea_left", "fovea_right", "ring_left", "eye_left", "eye_right", "watch_cam"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0, name
    for name in ("mobile", "target", "mimo_location"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name
