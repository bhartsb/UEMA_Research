# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Build the crib world once and write its skin layout as JSON for the Swift tactile port (WI-13, REQ-B-5).

INPUT. The released training configuration's world (weights/b1_continue_B1K_skip13_492800/run/args.json: scene
"crib", age_binding "fixed:24" (24 months), body_layout "trunk_v1", contact_profile "mechanical_v4",
touch_inside_rays "fixed_v1", touch_scale_mult 1.0), built by mimo_bridge.env.BridgeEnv with vision off in a
temporary scene folder. While it is built, MIMo's TrimeshTouch.add_body
(third_party/MIMo/mimo/mimo_infant/touch/touch.py:868-939) spreads one sensor mesh per colliding geom of each
sensing body (touch.py:986-1024), moves its vertices into the body frame (touch.py:890) and
marks a vertex inactive when it lies inside another geom of the same body (touch.py:917-925); the inside test uses
the bridge's 10 fixed ray directions (mimo_bridge/touch_inside_rays.py:38-53, FIXED_V1), so the layout is the one
every released world was built with: 4,683 live points, digest 0581069917fae467 (touch_inside_rays.py:24-25).

OPERATION. For every skin body (the sensing bodies TouchPooler keeps, touch_adapter.py:190: sensor spacing under
SKIN_SCALE_MAX 0.5 m; the two eye bodies have one placeholder point each at a larger spacing and are listed under
placeholder_bodies, outside the digest) in id order and every one of its submeshes (one per geom,
touch._submeshes), copy: every vertex's body-frame position in metres (float64, active and inactive, since MIMo's
breadth-first search walks inactive vertices too, touch.py:1276-1293), the active mask (touch._active_subvertices),
the vertex's index in the body's output arrays (touch._vertex_to_sensor_idx, -1 when inactive) and the vertex's
neighbour list from trimesh's vertex_adjacency_graph (touch.py:1222-1235: the mesh's edges). Per body: its sensor
spacing in metres (touch.sensor_scales = touch_params scales x touch_scale_mult, env.py:1130) and the spacing the
per-contact search uses (touch.force_spread_scales = sensor_scales / touch_scale_mult, env.py:1170; the "scale" of
contact_mechanics.conservative_spread_linear, contact_mechanics.py:67; the two are equal at touch_scale_mult 1.0) and
its sensor count. The layout digest is mimo_bridge.touch_inside_rays.skin_layout's: the first 16 hex characters of
the sha256 of the live points' float64 body-frame positions, bodies in id order. The script re-reads its own JSON
text, rebuilds the live positions from it and refuses to finish unless the rebuilt digest equals the live world's.

OUTPUT. apple/UEMAKit/Sources/UEMAScene/Resources/skin_layout_age24_trunk_v1.json (format "uema.skin_layout.v1",
keys documented in build_layout). Printed: point count, digest, body count, vertex count and the output path.
Exit 0 only when the count is 4,683 and the digest 0581069917fae467 (or the values passed with --expect-*).

Run from D:/UEMA_Research/Source/Python after the free-memory guard (6 GB is the port's build rule for one world; the
run that wrote the shipped file passed it with 7,031,312,384 bytes available):
    python -m mimo_bridge.tools.free_memory_guard --need-gb 6 && python ../../apple/scripts/export_skin_layout.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SOURCE_PYTHON = os.path.join(REPO_ROOT, "Source", "Python")
if SOURCE_PYTHON not in sys.path:
    sys.path.insert(0, SOURCE_PYTHON)

FORMAT = "uema.skin_layout.v1"
DEFAULT_OUT = os.path.join(REPO_ROOT, "apple", "UEMAKit", "Sources", "UEMAScene", "Resources",
                           "skin_layout_age24_trunk_v1.json")
# The released world (weights/b1_continue_B1K_skip13_492800/run/args.json) and the test that measured its layout
# (mimo_bridge/tests/test_touch_inside_rays.py:157-199).
WORLD = dict(scene="crib", age_months=24.0, body_layout="trunk_v1", contact_profile="mechanical_v4",
             touch_inside_rays="fixed_v1", touch_scale_mult=1.0)
EXPECTED_POINTS = 4683
EXPECTED_DIGEST = "0581069917fae467"
# The per-contact search the Swift port reproduces (contact_mechanics.py:67-76: scale = force_spread_scales[body],
# touch.get_sensors_within_distance(pos, body, 3 * scale), touch.py:1297-1358 with distance_margin 1.5).
SEARCH = dict(scale="force_spread_scale_m of the body (bodies[].force_spread_scale_m)",
              distance_limit="3 * force_spread_scale_m", first_search_margin=1.5,
              weight="max(1 - distance / (2 * force_spread_scale_m), 0), normalised to sum 1; when no point is in "
                     "range, or every point in range lies at 2 * force_spread_scale_m or further (all weights 0), "
                     "the single nearest live point gets weight 1 (contact_mechanics.py:71-74)",
              source="Source/Python/mimo_bridge/contact_mechanics.py:60-76; "
                     "third_party/MIMo/mimo/mimo_infant/touch/touch.py:1238-1358")


def digest_of_positions(positions_by_body: List[np.ndarray]) -> Tuple[int, str]:
    """(point count, digest) of live positions, bodies already in id order: the rule of
    mimo_bridge.touch_inside_rays.skin_layout (sha256 of the contiguous float64 bytes, first 16 hex characters)."""
    pos = [np.ascontiguousarray(np.asarray(p, dtype=np.float64)) for p in positions_by_body]
    return int(sum(len(p) for p in pos)), hashlib.sha256(b"".join(p.tobytes() for p in pos)).hexdigest()[:16]


def colliding_geoms(model, body_id: int) -> List[int]:
    """The geom ids TrimeshTouch.add_body meshes for a body, in its order (touch.py:884-889: the body's geoms,
    skipping those with contype 0 and conaffinity 0)."""
    start = int(model.body_geomadr[body_id])
    end = start + int(model.body_geomnum[body_id])
    out = []
    for geom_id in range(start, end):
        if int(model.geom(geom_id).contype.item()) == 0 and int(model.geom(geom_id).conaffinity.item()) == 0:
            continue
        out.append(int(geom_id))
    return out


def neighbour_lists(mesh) -> List[List[int]]:
    """Each vertex's sorted neighbour vertex indices from trimesh's vertex_adjacency_graph (the graph
    TrimeshTouch._get_mesh_adjacency_graph returns, touch.py:1222-1235); a point cloud or a one-vertex mesh has no
    edges and gets empty lists."""
    from trimesh import PointCloud
    n = int(len(mesh.vertices))
    if isinstance(mesh, PointCloud) or n <= 1 or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        return [[] for _ in range(n)]
    graph = mesh.vertex_adjacency_graph
    return [sorted(int(v) for v in graph.adj[i]) if i in graph else [] for i in range(n)]


def submesh_row(model, mesh, geom_id: int, mask: np.ndarray, smap: np.ndarray, sub: int, body_id: int
                ) -> Tuple[Dict[str, Any], np.ndarray]:
    """INPUT: one of a body's sensor meshes (touch._submeshes[body_id][sub]), the geom it was spread over, its
    active mask (touch._active_subvertices[body_id][sub]) and its vertex-to-sensor map
    (touch._vertex_to_sensor_idx[body_id][sub]). OPERATION: copy every vertex's body-frame position, the mask, the
    output index (-1 when inactive) and the neighbour lists; refuse mismatched array shapes. OUTPUT: (the submesh
    row of the JSON document, the live vertices' float64 positions in output order)."""
    import mujoco
    from trimesh import PointCloud
    verts = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64))
    mask = np.asarray(mask, dtype=bool)
    smap = np.asarray(smap, dtype=np.int64)
    if verts.shape != (len(mask), 3) or smap.shape != (len(mask),):
        raise RuntimeError("body %d submesh %d: shapes %s %s %s" % (body_id, sub, verts.shape, mask.shape, smap.shape))
    geom = model.geom(geom_id)
    row = dict(
        submesh=sub, geom_id=int(geom_id), geom_name=str(geom.name),
        geom_type=mujoco.mjtGeom(int(geom.type.item())).name, geom_size_m=[float(x) for x in geom.size],
        is_point_cloud=bool(isinstance(mesh, PointCloud)),
        vertex_count=int(len(mask)), active_count=int(mask.sum()),
        positions_m=verts.tolist(), active=mask.tolist(), sensor_index=np.where(mask, smap, -1).astype(np.int64).tolist(),
        neighbours=neighbour_lists(mesh))
    return row, verts[mask]


def body_row(model, touch, body_id: int, pooler_row: int, point_offset: int) -> Tuple[Dict[str, Any], np.ndarray]:
    """INPUT: the built touch, one skin body's id, its row in the pooler (patch order) and the number of live points
    of the bodies before it. OPERATION: one submesh_row per colliding geom, in add_body's order; refuse a body whose
    geom count differs from its submesh count, and one whose rebuilt live positions differ from
    touch.sensor_positions[body_id] or whose count differs from touch.get_sensor_count. OUTPUT: (the body row of the
    JSON document, the body's live float64 positions in output order)."""
    submeshes = touch._submeshes[body_id]
    geoms = colliding_geoms(model, body_id)
    if len(geoms) != len(submeshes):
        raise RuntimeError("body %d: %d colliding geoms but %d submeshes" % (body_id, len(geoms), len(submeshes)))
    rows: List[Dict[str, Any]] = []
    live_parts: List[np.ndarray] = []
    for sub, (mesh, geom_id) in enumerate(zip(submeshes, geoms)):
        row, live_part = submesh_row(model, mesh, geom_id, touch._active_subvertices[body_id][sub],
                                     touch._vertex_to_sensor_idx[body_id][sub], sub, body_id)
        rows.append(row)
        live_parts.append(live_part)
    live = np.concatenate(live_parts) if live_parts else np.zeros((0, 3))
    if not np.array_equal(live, np.asarray(touch.sensor_positions[body_id], dtype=np.float64)):
        raise RuntimeError("body %d: rebuilt live positions differ from touch.sensor_positions" % body_id)
    if int(touch.get_sensor_count(body_id)) != len(live):
        raise RuntimeError("body %d: sensor count %d, live vertices %d"
                           % (body_id, touch.get_sensor_count(body_id), len(live)))
    return dict(
        body_id=int(body_id), body_name=str(model.body(body_id).name), pooler_row=int(pooler_row),
        scale_m=float(touch.sensor_scales[body_id]),
        force_spread_scale_m=float(getattr(touch, "force_spread_scales", touch.sensor_scales)[body_id]),
        sensor_count=int(len(live)), point_offset=int(point_offset), submeshes=rows), live


def placeholder_rows(model, touch, skin_body_ids) -> List[Dict[str, Any]]:
    """The sensing bodies that are not skin bodies (the eyes, touch_adapter.py:190): id, name, spacing, count and
    their points, in id order; these stay outside the digest."""
    skin = set(int(b) for b in skin_body_ids)
    return [dict(body_id=int(b), body_name=str(model.body(b).name), scale_m=float(touch.sensor_scales[b]),
                 sensor_count=int(touch.get_sensor_count(b)),
                 positions_m=np.asarray(touch.sensor_positions[b], dtype=np.float64).tolist())
            for b in sorted(int(b) for b in touch.sensing_bodies()) if int(b) not in skin]


def build_layout(env, written_at: str) -> Dict[str, Any]:
    """INPUT: a built BridgeEnv (its touch is MIMo's TrimeshTouch, its model MuJoCo's). OPERATION: one body_row per
    skin body in id order (the pooler's bodies, touch_adapter.py:190-191), the points rows in digest order, the digest
    of the live positions, refused unless it equals the world's own report (env.touch_inside_rays_report). OUTPUT:
    the JSON document as a dict with keys format, written_by, written_at, versions, world, search, units,
    digest_rule, layout_digest, point_count, vertex_count, body_count, bodies (list, id order), points (list of
    [body_id, submesh, vertex, x_m, y_m, z_m] in digest order) and placeholder_bodies (the eyes)."""
    import mujoco
    import networkx
    import trimesh
    from mimo_bridge import touch_inside_rays as tir

    touch = env.touch
    model = env.model
    pooler_row = {int(s.body_id): i for i, s in enumerate(env.pooler.specs)}
    bodies: List[Dict[str, Any]] = []
    points: List[List[Any]] = []
    live_by_body: List[np.ndarray] = []
    offset = 0
    for body_id in sorted(pooler_row):
        row, live = body_row(model, touch, body_id, pooler_row[body_id], offset)
        for sub in row["submeshes"]:
            for v in np.flatnonzero(sub["active"]):
                points.append([body_id, sub["submesh"], int(v)] + sub["positions_m"][v])
        bodies.append(row)
        live_by_body.append(live)
        offset += len(live)
    count, digest = digest_of_positions(live_by_body)
    report = dict(env.touch_inside_rays_report)
    if (count, digest) != (report["skin_points"], report["skin_digest"]):
        raise RuntimeError("digest here (%d, %s) differs from the world's report %s" % (count, digest, report))
    return dict(
        format=FORMAT, written_by="apple/scripts/export_skin_layout.py", written_at=written_at,
        versions=dict(python=platform.python_version(), numpy=np.__version__, trimesh=trimesh.__version__,
                      mujoco=mujoco.__version__, networkx=networkx.__version__),
        world=dict(WORLD, directions_sha256=report["directions_sha256"], inside_tests=int(report["inside_tests"]),
                   shape_pairs=int(tir.shapes_to_test(touch)), nq=int(model.nq), nbody=int(model.nbody),
                   scene_xml=os.path.basename(env.scene_path),
                   scene_additions="vision, caregiver, mobile and critters off for this build (BridgeEnv defaults; "
                                   "the released run had mobile on, critters on, caregiver_mode responsive): they "
                                   "add joints (nq %d here against 146 in the released run's observer headers, "
                                   "weights/b1_continue_B1K_skip13_492800/run/observer/world/b0/header_*.json) and "
                                   "no skin; the digest is the released worlds' "
                                   "(mimo_bridge/touch_inside_rays.py:24-25)" % int(model.nq)),
        search=SEARCH,
        units=dict(positions_m="metres in the body frame (MIMo geom_pos_to_body at the build pose, touch.py:890)",
                   scale_m="metres between neighbouring sensor points (touch_params scales x touch_scale_mult)"),
        digest_rule="first 16 hex characters of sha256 over the live points' float64 [x, y, z] bytes, bodies in "
                    "id order, points in each body's output order (mimo_bridge/touch_inside_rays.py:126-131)",
        layout_digest=digest, point_count=count,
        vertex_count=int(sum(s["vertex_count"] for b in bodies for s in b["submeshes"])), body_count=len(bodies),
        bodies=bodies, points=points,
        placeholder_bodies=dict(rule="sensing bodies whose scale_m is at least SKIN_SCALE_MAX 0.5 m form no patch "
                                     "(mimo_bridge/touch_adapter.py:68, 190) and are outside the digest",
                                bodies=placeholder_rows(model, touch, pooler_row)))


def digest_from_document(doc: Dict[str, Any]) -> Tuple[int, str]:
    """Recompute (point count, digest) from a loaded layout document's points rows alone, grouped by body in the
    rows' order; a Swift reader that keeps these float64 values byte-exact reproduces the same digest."""
    by_body: Dict[int, List[List[float]]] = {}
    order: List[int] = []
    for row in doc["points"]:
        b = int(row[0])
        if b not in by_body:
            by_body[b] = []
            order.append(b)
        elif b != order[-1]:
            raise ValueError("points rows of body %d are split by another body's rows" % b)
        by_body[b].append([float(row[3]), float(row[4]), float(row[5])])
    if order != sorted(order):
        raise ValueError("points rows are not grouped in body id order")
    return digest_of_positions([np.asarray(by_body[b], dtype=np.float64).reshape(-1, 3) for b in order])


def build_world(run_dir: str):
    """The released crib world with vision off (mimo_bridge.env.BridgeEnv); the caller closes it."""
    from mimo_bridge.env import BridgeEnv
    return BridgeEnv(scene=WORLD["scene"], vision=False, run_dir=run_dir, age=WORLD["age_months"],
                     body_layout=WORLD["body_layout"], contact_profile=WORLD["contact_profile"],
                     touch_inside_rays=WORLD["touch_inside_rays"], touch_scale_mult=WORLD["touch_scale_mult"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="JSON path to write (default: the UEMAScene resource)")
    parser.add_argument("--run-dir", default="", help="scene folder for the build (default: a temporary folder, "
                                                       "removed afterwards)")
    parser.add_argument("--expect-points", type=int, default=EXPECTED_POINTS)
    parser.add_argument("--expect-digest", default=EXPECTED_DIGEST)
    args = parser.parse_args(argv)
    written_at = _dt.datetime.now(_dt.timezone.utc).astimezone().strftime("%Y-%m-%d %I:%M %p %Z")
    run_dir = args.run_dir or tempfile.mkdtemp(prefix="skin_layout_")
    try:
        env = build_world(run_dir)
        try:
            doc = build_layout(env, written_at)
        finally:
            env.close()
    finally:
        if not args.run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False)
    count, digest = digest_from_document(json.loads(text))
    if (count, digest) != (doc["point_count"], doc["layout_digest"]):
        print("REFUSED: the JSON text rebuilds (%d, %s), the world gave (%d, %s)"
              % (count, digest, doc["point_count"], doc["layout_digest"]))
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.write("\n")
    print("skin layout: %d live points, digest %s, %d bodies, %d vertices (live and inactive), %d bytes -> %s"
          % (count, digest, doc["body_count"], doc["vertex_count"], len(text) + 1, args.out))
    if count != args.expect_points or digest != args.expect_digest:
        print("FAIL: expected %d points with digest %s" % (args.expect_points, args.expect_digest))
        return 1
    print("PASS: %d points, digest %s" % (args.expect_points, args.expect_digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
