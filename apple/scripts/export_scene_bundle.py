# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Export the released crib world as a self-contained MuJoCo scene bundle for the Apple port (WI-15).

INPUT
    The released training configuration, read from the shipped run's args.json
    (weights/b1_continue_B1K_skip13_492800/run/args.json): scene "crib", mobile "on",
    critters "on", caregiver_mode "responsive", age_binding "fixed:24", body_layout
    "trunk_v1", fovea_fov 24.0 degrees with fovea_size 288 px, ring_fov 150.0 degrees,
    and the trainer's default eye_size of 256 px (mimo_bridge/env.py:989; train_mimo.py's
    env_kwargs_for passes no eye_size). The MIMo body files come from the installed
    mimo_infant package (MIMo_metav2.xml, 41,064 bytes; MIMo_modelv2.xml, 38,486 bytes).

OPERATION
    1. mimo_bridge.env.write_scene("crib", ...) writes the crib wrapper (mobile, critters,
       pen walls, target ball, three boxes), the lite copy of MIMo_metav2.xml (every texture
       file attribute replaced by a 16 x 16 flat builtin texture, 16 x 96 for cube textures,
       env.py:129-139: 25 PNG files, 3,309,114 bytes, are not shipped), the body copy with the
       fovea_left, fovea_right and ring_left cameras (env.py:144), and the six pen-art PNGs
       (play_mat.png 1024 x 1024 px and ceiling.png 1024 x 1024 px, pen_art.py:167 and :215;
       pen_wall_n/s.png 2048 x 342 px and pen_wall_e/w.png 342 x 2048 px, pen_art.py:25).
    2. mimo_bridge.caregiver.add_caregiver_to_scene adds the adult's visual bodies and the
       pad and ring geoms on the target body (the trainer's step at env.py:1115-1119).
    3. Growth to 24 months runs twice from the same growth parameters
       (mimo_infant.growth.growth.get_growth_params(24.0, "v2")):
       (a) mimo_bridge.env.grow_scene, the trainer's own call (env.py:1120-1122), gives the
           REFERENCE: one self-contained file with the grown body and motor gears inlined;
       (b) mimo_infant.growth.scene.create_growth_scene(..., long_format=False) on a copy
           of the same scene gives the SHIPPED layout: scene.xml plus two include files,
           MIMo_metav2_lite.xml (grown motor gears) and MIMo_modelv2_grown24_fovea.xml
           (grown geom sizes, positions, masses; body, joint and site positions).
    4. Every absolute path is made relative: the two <include file="..."> attributes become
       the include file names, the six <texture file="..."> attributes become PNG names, and
       the compiler's meshdir and texturedir attributes are removed (0 mesh elements exist).
    5. Checks, each one a failure of the export when it does not hold:
       - no attribute value in the three XML files holds a drive letter, a leading slash or
         a parent-folder step;
       - the bundle, copied into a fresh temporary folder, loads with
         mujoco.MjModel.from_xml_path, no message reaches MuJoCo's mju_user_warning hook
         while it loads, and its nq, nv, nu, nmocap and nbody equal the shipped observer
         header's (nq 146, nv 136, nu 90, nmocap 8, nbody 76 in
         weights/b1_continue_B1K_skip13_492800/run/observer/world/b0/header_*.json);
       - the bundle's MjModel equals the reference MjModel of step 3(a) on the fields listed
         in REFERENCE_FIELDS (sizes, positions, masses, gears, ranges, names), exactly.

OUTPUT
    apple/UEMAKit/Sources/UEMAScene/Resources/crib_age24_trunk_v1/ with scene.xml,
    MIMo_metav2_lite.xml, MIMo_modelv2_grown24_fovea.xml, play_mat.png, ceiling.png,
    pen_wall_n.png, pen_wall_s.png, pen_wall_e.png, pen_wall_w.png and bundle.json, the
    manifest: for every file its name, bytes and SHA-256; the configuration and its source;
    the source file hashes (paths relative to the mimo_infant package, never a machine path);
    the model counts; the six texture sizes in pixels; the warnings captured during the
    fresh-folder load; the header comparison; the reference comparison.

    body_layout "trunk_v1" is part of the bundle's name because the released args.json names
    it; it changes no scene element (env.py reads it for the proprioception adapter only,
    env.py:1230-1236).

USAGE (cwd D:/UEMA_Research/Source/Python, the repository's interpreter)
    python ../../apple/scripts/export_scene_bundle.py [--out DIR] [--work DIR] [--keep-work]
    Exit code 0 when every check holds; the last printed line reads the fresh-folder nq.
    Exit code 1 when a check fails; the failing check is printed on stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_PYTHON = REPO_ROOT / "Source" / "Python"
if str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

BUNDLE_NAME = "crib_age24_trunk_v1"
DEFAULT_OUT = REPO_ROOT / "apple" / "UEMAKit" / "Sources" / "UEMAScene" / "Resources" / BUNDLE_NAME
RELEASED_ARGS = REPO_ROOT / "weights" / "b1_continue_B1K_skip13_492800" / "run" / "args.json"
OBSERVER_HEADER_DIR = REPO_ROOT / "weights" / "b1_continue_B1K_skip13_492800" / "run" / "observer" / "world" / "b0"

META_NAME = "MIMo_metav2_lite.xml"
BODY_NAME = "MIMo_modelv2_grown24_fovea.xml"
SCENE_NAME = "scene.xml"
MANIFEST_NAME = "bundle.json"
TEXTURE_NAMES = ("play_mat.png", "ceiling.png", "pen_wall_n.png", "pen_wall_s.png", "pen_wall_e.png", "pen_wall_w.png")
# Every file the exporter writes; these, and only these, are replaced in the output folder.
BUNDLE_FILE_NAMES = (SCENE_NAME, META_NAME, BODY_NAME) + TEXTURE_NAMES + (MANIFEST_NAME,)

# args.json keys the configuration is read from; a missing key is reported by name.
ARGS_JSON_KEYS = ("scene", "mobile", "critters", "caregiver_mode", "age_binding", "body_layout",
                  "fovea_fov", "fovea_size", "ring_fov")

# Observer header counts the bundle must reproduce (header_*.json of the shipped run, body 0).
HEADER_COUNT_FIELDS = ("nq", "nv", "nu", "nmocap", "nbody")

# MjModel fields compared exactly between the shipped three-file bundle and the trainer's own
# single-file grown scene. Scalars are counts; arrays are compared element for element.
REFERENCE_FIELDS = (
    "nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsite", "ncam", "nlight", "nmocap", "nsensor",
    "ntex", "nmat", "neq", "ntendon", "nuser_body",
    "body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_mocapid",
    "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range",
    "geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat", "geom_rgba", "geom_contype",
    "geom_conaffinity", "geom_condim", "geom_matid",
    "site_bodyid", "site_pos", "cam_bodyid", "cam_pos", "cam_quat", "cam_fovy",
    "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_ctrlrange", "actuator_forcerange",
    "sensor_type", "sensor_objid", "dof_damping", "dof_armature", "qpos0", "tex_height", "tex_width",
    "names",
)

ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/])|\.\.[\\/]")
MIMO_LICENSE_COMMENT = re.compile(r"^\s*(<!--.*?-->)", re.DOTALL)


@dataclass(frozen=True)
class CribReleaseConfiguration:
    """The scene arguments of the released run, each with the args.json key it is read from.

    scene, mobile, critters, caregiver_mode, age, body_layout, fovea_fov, fovea_size and ring_fov
    are read from args.json; eye_size is MimoEnv's default (env.py:989) since the trainer passes
    none; lite_textures is write_scene's default (env.py:512) and the trainer passes none.
    """
    scene: str = "crib"
    mobile: bool = True
    critters: bool = True
    caregiver_mode: str = "responsive"
    age_months: float = 24.0
    body_layout: str = "trunk_v1"
    fovea_fov_deg: float = 24.0
    fovea_size_px: int = 288
    ring_fov_deg: float = 150.0
    eye_size_px: int = 256
    lite_textures: bool = True
    tag: str = ""


def configuration_from_args_json(path: Path) -> CribReleaseConfiguration:
    """Read the released args.json and return the scene arguments it fixes.

    Input: path to args.json. Operation: each key of ARGS_JSON_KEYS is read (a missing key
    raises ValueError naming it) and checked against the values the bundle name promises
    (crib, age 24, trunk_v1); a mismatch raises ValueError naming the key.
    Output: a CribReleaseConfiguration.
    """
    args = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ARGS_JSON_KEYS if k not in args]
    if missing:
        raise ValueError(f"{path} lacks the keys {missing}")
    binding = str(args["age_binding"])
    if not binding.startswith("fixed:"):
        raise ValueError(f"age_binding is {binding!r}; the bundle is built for a fixed age")
    cfg = CribReleaseConfiguration(
        scene=str(args["scene"]),
        mobile=args["mobile"] == "on",
        critters=args["critters"] == "on",
        caregiver_mode=str(args["caregiver_mode"]),
        age_months=float(binding.split(":", 1)[1]),
        body_layout=str(args["body_layout"]),
        fovea_fov_deg=float(args["fovea_fov"]),
        fovea_size_px=int(args["fovea_size"]),
        ring_fov_deg=float(args["ring_fov"]),
        tag=str(args.get("tag", "")),
    )
    expected = {"scene": "crib", "age_months": 24.0, "body_layout": "trunk_v1"}
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"args.json {key} is {getattr(cfg, key)!r}, the bundle {BUNDLE_NAME} needs {value!r}")
    return cfg


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path) -> Dict[str, object]:
    """Return {"name", "bytes", "sha256"} for one file."""
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path)}


def mimo_license_comment(path: Path) -> str:
    """Return the leading XML comment of a MIMo source file (its MIT notice), or "" when absent.

    xml.etree drops comments when it parses, so the notice is re-attached to each derived file.
    """
    m = MIMO_LICENSE_COMMENT.match(path.read_text(encoding="utf-8"))
    return m.group(1) if m else ""


def write_xml(root: ET.Element, path: Path, leading_comment: str = "") -> None:
    """Write an element tree as UTF-8 with an XML declaration and an optional leading comment."""
    body = ET.tostring(root, encoding="unicode")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        if leading_comment:
            f.write(leading_comment + "\n")
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")


def absolute_path_attributes(path: Path) -> List[str]:
    """Return "tag@attr=value" for every attribute value that is an absolute or parent-relative path."""
    found = []
    for elem in ET.parse(path).getroot().iter():
        for key, value in elem.attrib.items():
            if ABSOLUTE_PATH.search(value):
                found.append(f"{elem.tag}@{key}={value}")
    return found


def find_includes(root: ET.Element) -> Dict[str, ET.Element]:
    """Return the scene's two include elements keyed "meta" and "model" by file name, as MIMo's growth does."""
    out: Dict[str, ET.Element] = {}
    for inc in root.iter("include"):
        name = os.path.basename(inc.get("file", "")).lower()
        if "meta" in name:
            out["meta"] = inc
        elif "model" in name:
            out["model"] = inc
    if set(out) != {"meta", "model"}:
        raise ValueError(f"scene needs one meta and one model include, found {sorted(out)}")
    return out


def write_base_scene(cfg: CribReleaseConfiguration, work: Path) -> Path:
    """Run the trainer's scene writer and caregiver step; return the ungrown scene path.

    Input: the configuration and a work folder. Operation: env.write_scene (lite meta, fovea
    and ring body copy, pen art, crib wrapper), then caregiver.add_caregiver_to_scene.
    Output: the path of mimo_bridge_crib.xml inside work, whose includes are absolute.
    """
    from mimo_bridge.caregiver import add_caregiver_to_scene
    from mimo_bridge.env import write_scene

    base = write_scene(cfg.scene, str(work), cfg.eye_size_px, tag=cfg.tag, lite_textures=cfg.lite_textures,
                       fovea_fov=cfg.fovea_fov_deg if cfg.fovea_size_px > 0 else 0.0,
                       mobile=cfg.mobile, ring_fov=cfg.ring_fov_deg, critters=cfg.critters)
    if cfg.caregiver_mode != "off":
        add_caregiver_to_scene(base)
    return Path(base)


def grow_reference(base: Path, cfg: CribReleaseConfiguration, work: Path) -> Path:
    """The trainer's own growth call (env.py:1120-1122): one self-contained grown scene file."""
    from mimo_bridge.env import grow_scene

    out = work / "reference" / f"mimo_bridge_{cfg.scene}{cfg.tag}_age{cfg.age_months:g}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    return Path(grow_scene(str(base), cfg.age_months, str(out)))


def grow_split(base: Path, cfg: CribReleaseConfiguration, work: Path) -> Dict[str, Path]:
    """Grow the same scene with its meta and body kept as separate include files.

    Input: the ungrown scene (absolute includes). Operation: the two include files are copied
    beside a copy of the scene under the shipped names, the scene's includes are rewritten to
    those names, then MIMo's create_growth_scene(long_format=False) applies the age-24 growth
    parameters and writes "_temp" copies with relative includes. Output: paths keyed "scene",
    "meta", "model" of the grown files.
    """
    from mimo_infant.growth.growth import get_growth_params
    from mimo_infant.growth.scene import create_growth_scene

    split = work / "split"
    split.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(base)
    inc = find_includes(tree.getroot())
    shutil.copyfile(inc["meta"].get("file"), split / META_NAME)
    shutil.copyfile(inc["model"].get("file"), split / BODY_NAME)
    inc["meta"].set("file", META_NAME)
    inc["model"].set("file", BODY_NAME)
    scene_copy = split / "scene_base.xml"
    tree.write(scene_copy, encoding="utf-8", xml_declaration=True)
    params = get_growth_params(cfg.age_months, "v2", None)
    grown_scene = Path(create_growth_scene(params, str(scene_copy), long_format=False))
    grown_inc = find_includes(ET.parse(grown_scene).getroot())
    return {"scene": grown_scene,
            "meta": split / grown_inc["meta"].get("file"),
            "model": split / grown_inc["model"].get("file")}


def mimo_assets_dir() -> Path:
    """Return the installed mimo_infant package's assets/mimo folder (MIMo_metav2.xml, MIMo_modelv2.xml)."""
    import mimo_infant

    return Path(mimo_infant.__file__).resolve().parent / "simulation" / "assets" / "mimo"


def assemble_bundle(grown: Dict[str, Path], pen_art: Path, out: Path, header_comment: str) -> List[Path]:
    """Copy the grown files into the bundle folder with every path made relative; return the file list.

    Only the files named in BUNDLE_FILE_NAMES are replaced; a file of another name in `out`
    is left as it is.
    """
    assets = mimo_assets_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILE_NAMES:
        stale = out / name
        if stale.is_file():
            stale.unlink()
    scene_root = ET.parse(grown["scene"]).getroot()
    inc = find_includes(scene_root)
    inc["meta"].set("file", META_NAME)
    inc["model"].set("file", BODY_NAME)
    compiler = scene_root.find("compiler")
    if compiler is None:
        raise ValueError("scene has no <compiler> element")
    for key in ("meshdir", "texturedir", "assetdir"):
        compiler.attrib.pop(key, None)
    if list(scene_root.iter("mesh")):
        raise ValueError("the scene references a mesh; none may ship")
    texture_files = []
    for tex in scene_root.iter("texture"):
        for key in list(tex.attrib):
            if key.startswith("file"):
                name = os.path.basename(tex.get(key))
                tex.set(key, name)
                texture_files.append(name)
    if sorted(texture_files) != sorted(TEXTURE_NAMES):
        raise ValueError(f"scene textures {sorted(texture_files)} differ from {sorted(TEXTURE_NAMES)}")
    write_xml(scene_root, out / SCENE_NAME, header_comment)
    for key, name, source in (("meta", META_NAME, assets / "MIMo_metav2.xml"),
                              ("model", BODY_NAME, assets / "MIMo_modelv2.xml")):
        root = ET.parse(grown[key]).getroot()
        if list(root.iter("mesh")):
            raise ValueError(f"{name} references a mesh; none may ship")
        for tex in root.iter("texture"):
            if any(k.startswith("file") for k in tex.attrib):
                raise ValueError(f"{name} still references a texture file")
        write_xml(root, out / name, mimo_license_comment(source))
    for name in TEXTURE_NAMES:
        shutil.copyfile(pen_art / name, out / name)
    return [out / SCENE_NAME, out / META_NAME, out / BODY_NAME] + [out / n for n in TEXTURE_NAMES]


def model_counts(model) -> Dict[str, int]:
    """Return the five size fields the observer header records (nq, nv, nu, nmocap, nbody) plus the
    joint, geom, camera, texture and material counts (njnt, ngeom, ncam, ntex, nmat) of an MjModel."""
    return {k: int(getattr(model, k)) for k in ("nq", "nv", "nu", "nmocap", "nbody", "njnt", "ngeom", "ncam", "ntex", "nmat")}


@contextlib.contextmanager
def captured_mujoco_warnings() -> Iterator[List[str]]:
    """Collect every message MuJoCo routes to its mju_user_warning hook while the block runs.

    Input: none. Operation: mujoco.set_mju_user_warning installs a collector; the hook is
    removed again when the block ends, whatever happened inside. Output: the list of messages,
    filled in place (empty when no message was routed). MuJoCo 3.3.0's Python bindings expose
    no other channel for the text of a compile warning (MjSpec has no warning attribute), so
    this list is the warning evidence the manifest records.
    """
    import mujoco

    messages: List[str] = []
    mujoco.set_mju_user_warning(lambda message: messages.append(str(message)))
    try:
        yield messages
    finally:
        mujoco.set_mju_user_warning(None)


def load_in_fresh_folder(bundle: Path):
    """Copy the bundle's files into a new temporary folder and load scene.xml there with MuJoCo."""
    import mujoco

    with tempfile.TemporaryDirectory(prefix="fresh_bundle_") as fresh:
        for p in bundle.iterdir():
            if p.is_file():
                shutil.copyfile(p, Path(fresh) / p.name)
        return mujoco.MjModel.from_xml_path(str(Path(fresh) / SCENE_NAME))


def compare_models(a, b, fields=REFERENCE_FIELDS) -> List[str]:
    """Return the names of fields in `fields` on which MjModels a and b differ (empty means equal)."""
    import numpy as np

    differing = []
    for name in fields:
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, (int, float, bytes)):
            same = va == vb
        else:
            same = np.array_equal(np.asarray(va), np.asarray(vb))
        if not same:
            differing.append(name)
    return differing


def observer_header() -> Optional[Dict[str, object]]:
    """Read the shipped observer world header (body 0); None when the weights folder is absent.

    A header that lacks one of HEADER_COUNT_FIELDS raises ValueError naming the field.
    """
    if not OBSERVER_HEADER_DIR.is_dir():
        return None
    headers = sorted(OBSERVER_HEADER_DIR.glob("header_*.json"))
    if not headers:
        return None
    data = json.loads(headers[0].read_text(encoding="utf-8"))
    missing = [k for k in HEADER_COUNT_FIELDS if k not in data]
    if missing:
        raise ValueError(f"{headers[0]} lacks the counts {missing}")
    return {"path": str(headers[0].relative_to(REPO_ROOT)).replace("\\", "/"),
            **{k: int(data[k]) for k in HEADER_COUNT_FIELDS}}


def git_head(repo: Path) -> str:
    """Return the commit the repository is at, read from its git files without running git.

    Input: the checkout's root. Operation: `.git` is a folder, or a file reading
    "gitdir: <path>" for a worktree or submodule (the path is taken relative to the checkout
    when it is not absolute); HEAD is a commit hash, or "ref: <name>" resolved through the
    loose ref file or packed-refs, in the checkout's own git folder and then in the common
    folder a worktree's "commondir" file names. Output: the 40-hex commit; a ref that
    resolves nowhere raises ValueError naming it.
    """
    git_dir = repo / ".git"
    if git_dir.is_file():
        pointer = git_dir.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir:"):
            raise ValueError(f"{git_dir} is a file that does not read 'gitdir: <path>'")
        git_dir = Path(pointer[len("gitdir:"):].strip())
        if not git_dir.is_absolute():
            git_dir = (repo / git_dir).resolve()
    if not git_dir.is_dir():
        raise ValueError(f"{repo} has no git folder")
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_name = head[len("ref: "):]
    folders = [git_dir]
    commondir = git_dir / "commondir"
    if commondir.is_file():
        folders.append((git_dir / commondir.read_text(encoding="utf-8").strip()).resolve())
    for folder in folders:
        ref = folder / ref_name
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        packed = folder / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref_name):
                    return line.split(" ", 1)[0]
    raise ValueError(f"{ref_name} resolves to no commit under {git_dir}")


def export(out: Path, work: Path, cfg: CribReleaseConfiguration) -> Dict[str, object]:
    """Build, check and write the bundle; return the manifest that was written to bundle.json.

    Input: the bundle folder, a work folder for the generated scene, the configuration.
    Operation: steps 1-5 of the module docstring; a failed check raises ValueError naming it.
    Output: the manifest dictionary, also written as bundle.json in the bundle folder.
    """
    import mujoco
    import mimo_infant

    base = write_base_scene(cfg, work)
    reference_path = grow_reference(base, cfg, work)
    grown = grow_split(base, cfg, work)
    header_comment = (
        "<!-- Generated by apple/scripts/export_scene_bundle.py; do not edit by hand (bundle.json holds every other file's SHA-256).\n"
        f"     Released configuration of weights/b1_continue_B1K_skip13_492800/run/args.json: scene {cfg.scene}, mobile "
        f"{'on' if cfg.mobile else 'off'}, critters {'on' if cfg.critters else 'off'}, caregiver {cfg.caregiver_mode}, age "
        f"{cfg.age_months:g} months, fovea {cfg.fovea_fov_deg:g} degrees at {cfg.fovea_size_px} px, ring {cfg.ring_fov_deg:g} "
        f"degrees, eyes {cfg.eye_size_px} px, lite textures. -->")
    files = assemble_bundle(grown, work / "pen_art", out, header_comment)

    problems = []
    for xml_file in files[:3]:
        problems += [f"{xml_file.name}: {p}" for p in absolute_path_attributes(xml_file)]
    if problems:
        raise ValueError("absolute paths remain: " + "; ".join(problems))

    with captured_mujoco_warnings() as load_warnings:
        fresh_model = load_in_fresh_folder(out)
    if load_warnings:
        raise ValueError("MuJoCo warned while loading the bundle: " + "; ".join(load_warnings))
    counts = model_counts(fresh_model)
    reference_model = mujoco.MjModel.from_xml_path(str(reference_path))
    differing = compare_models(fresh_model, reference_model)
    if differing:
        raise ValueError("bundle differs from the trainer's grown scene on: " + ", ".join(differing))
    header = observer_header()
    header_match = None
    if header is not None:
        mismatched = {k: (counts[k], header[k]) for k in HEADER_COUNT_FIELDS if counts[k] != header[k]}
        if mismatched:
            raise ValueError(f"bundle counts differ from the observer header: {mismatched}")
        header_match = {"compared": list(HEADER_COUNT_FIELDS), "equal": True}

    assets = mimo_assets_dir()
    package_root = Path(mimo_infant.__file__).resolve().parent.parent
    source_meta = ET.parse(assets / "MIMo_metav2.xml").getroot()
    dropped = sorted({v for t in source_meta.iter("texture") for k, v in t.attrib.items() if k.startswith("file")})
    dropped_bytes = sum((assets.parent / v).stat().st_size for v in dropped if (assets.parent / v).exists())

    from PIL import Image
    texture_sizes = {}
    for name in TEXTURE_NAMES:
        with Image.open(out / name) as im:
            texture_sizes[name] = list(im.size)

    def source_record(name: str) -> Dict[str, object]:
        path = assets / name
        return {**file_record(path), "path": str(path.relative_to(package_root)).replace("\\", "/")}

    manifest = {
        "bundle": BUNDLE_NAME,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": {
            "script": "apple/scripts/export_scene_bundle.py",
            "repository_commit": git_head(REPO_ROOT),
            "python": sys.version.split()[0],
            "mujoco": mujoco.__version__,
            "mimo_infant": str(getattr(mimo_infant, "__version__", "unknown")),
        },
        "configuration": {**asdict(cfg), "source": str(RELEASED_ARGS.relative_to(REPO_ROOT)).replace("\\", "/"),
                          "note": "body_layout changes no scene element (env.py:1230-1236 reads it for the proprioception adapter only)"},
        "sources": {
            "MIMo_metav2.xml": source_record("MIMo_metav2.xml"),
            "MIMo_modelv2.xml": source_record("MIMo_modelv2.xml"),
        },
        "not_shipped": {
            "mimo_texture_png_files": len(dropped), "mimo_texture_png_bytes": dropped_bytes,
            "mesh_files_referenced": 0,
        },
        "files": [file_record(p) for p in files],
        "scene_file": SCENE_NAME,
        "includes": [META_NAME, BODY_NAME],
        "textures": list(TEXTURE_NAMES),
        "play_mat_size_px": texture_sizes["play_mat.png"],
        "texture_sizes_px": texture_sizes,
        "model_counts": counts,
        "observer_header": header,
        "observer_header_match": header_match,
        "reference_check": {
            "reference": "mimo_bridge.env.grow_scene output (the trainer's own single-file grown scene, env.py:1120-1122)",
            "fields_compared": list(REFERENCE_FIELDS),
            "equal": True,
        },
        "fresh_folder_load": {
            "nq": counts["nq"], "loaded": True,
            "mju_user_warning_messages": list(load_warnings),
            "warning_channel": "mujoco.set_mju_user_warning during MjModel.from_xml_path; the 3.3.0 Python bindings expose no other compile-warning text",
        },
    }
    with open(out / MANIFEST_NAME, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry: export the bundle and print the checks; 0 on success, 1 on a failed check."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"bundle folder (default {DEFAULT_OUT})")
    ap.add_argument("--work", type=Path, default=None, help="work folder for the generated scene (default: a temporary folder)")
    ap.add_argument("--keep-work", action="store_true", help="keep the work folder")
    ap.add_argument("--args-json", type=Path, default=RELEASED_ARGS, help="the released run's args.json")
    a = ap.parse_args(argv)
    work = a.work if a.work is not None else Path(tempfile.mkdtemp(prefix="scene_bundle_work_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        cfg = configuration_from_args_json(a.args_json)
        manifest = export(a.out.resolve(), work.resolve(), cfg)
    except (ValueError, OSError) as failure:
        print(f"export failed: {failure}", file=sys.stderr)
        return 1
    finally:
        if a.work is None and not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    print(f"bundle written: {a.out.resolve()}")
    for rec in manifest["files"]:
        print(f"  {rec['name']}: {rec['bytes']} bytes sha256 {rec['sha256'][:16]}")
    print(f"reference check: equal on {len(REFERENCE_FIELDS)} MjModel fields against the trainer's grown scene")
    header = manifest["observer_header"]
    if header is not None:
        print(f"observer header {header['path']}: " + ", ".join(f"{k} {header[k]}" for k in HEADER_COUNT_FIELDS) + " - all equal")
    counts = manifest["model_counts"]
    sizes = manifest["texture_sizes_px"]
    print("textures px: " + ", ".join(f"{n} {w}x{h}" for n, (w, h) in sizes.items()))
    print(f"load warnings: {len(manifest['fresh_folder_load']['mju_user_warning_messages'])}")
    print(f"fresh folder load: nq {counts['nq']} nv {counts['nv']} nu {counts['nu']} nmocap {counts['nmocap']} nbody {counts['nbody']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
