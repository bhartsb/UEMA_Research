# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The written skin layout (apple/scripts/export_skin_layout.py, WI-13) holds what the Swift tactile port needs.

These tests read the JSON the script wrote (skin_layout_age24_trunk_v1.json) and check it without building a world:
the digest rule of mimo_bridge/touch_inside_rays.py:126-131 recomputed from the file's points rows gives the file's
own digest, 0581069917fae467 over 4,683 points (touch_inside_rays.py:24-25); every body's submesh tables agree
with the points rows; the neighbour lists are symmetric (an undirected mesh graph, touch.py:1222-1235); every live
vertex carries its output index in order and every inactive vertex carries -1. A missing file skips the tests
with the command that writes it. Run from Source/Python:
    python -m pytest ../../apple/scripts/tests/test_export_skin_layout.py -p no:cacheprovider -q
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import export_skin_layout as esl  # noqa: E402


@pytest.fixture(scope="module")
def doc():
    if not os.path.exists(esl.DEFAULT_OUT):
        pytest.skip("no %s: run python ../../apple/scripts/export_skin_layout.py from Source/Python first"
                    % esl.DEFAULT_OUT)
    with open(esl.DEFAULT_OUT, encoding="utf-8") as f:
        return json.load(f)


def test_the_file_rebuilds_the_released_layout_digest(doc):
    assert doc["format"] == esl.FORMAT
    assert (doc["point_count"], doc["layout_digest"]) == (esl.EXPECTED_POINTS, esl.EXPECTED_DIGEST)
    assert esl.digest_from_document(doc) == (esl.EXPECTED_POINTS, esl.EXPECTED_DIGEST)
    assert len(doc["points"]) == esl.EXPECTED_POINTS
    assert doc["world"]["touch_inside_rays"] == "fixed_v1" and doc["world"]["age_months"] == 24.0
    assert doc["world"]["inside_tests"] == doc["world"]["shape_pairs"] == 58   # test_touch_inside_rays.py:197
    # the two eye bodies: one placeholder point each at a spacing of at least 0.5 m (touch_adapter.py:68, 190)
    eyes = doc["placeholder_bodies"]["bodies"]
    assert len(eyes) == 2 and all(e["scale_m"] >= 0.5 and e["sensor_count"] == 1 for e in eyes)
    assert not ({e["body_id"] for e in eyes} & {b["body_id"] for b in doc["bodies"]})


def test_bodies_submeshes_and_points_rows_agree(doc):
    ids = [b["body_id"] for b in doc["bodies"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) == doc["body_count"]
    assert sorted(b["pooler_row"] for b in doc["bodies"]) == list(range(len(ids)))
    rows = iter(doc["points"])
    offset = 0
    vertex_total = 0
    for body in doc["bodies"]:
        assert body["point_offset"] == offset and body["scale_m"] > 0 and body["force_spread_scale_m"] > 0
        live_in_body = 0
        for sub_i, sub in enumerate(body["submeshes"]):
            n = sub["vertex_count"]
            assert sub["submesh"] == sub_i
            assert len(sub["positions_m"]) == len(sub["active"]) == len(sub["sensor_index"]) == len(sub["neighbours"]) == n
            assert sub["active_count"] == sum(sub["active"])
            for v in range(n):
                if sub["active"][v]:
                    assert sub["sensor_index"][v] == live_in_body
                    row = next(rows)
                    assert row[:3] == [body["body_id"], sub_i, v]
                    assert row[3:] == sub["positions_m"][v]
                    live_in_body += 1
                else:
                    assert sub["sensor_index"][v] == -1
            vertex_total += n
        assert live_in_body == body["sensor_count"]
        offset += live_in_body
    assert offset == doc["point_count"] and vertex_total == doc["vertex_count"]
    assert next(rows, None) is None


def test_neighbour_lists_form_an_undirected_graph_within_each_submesh(doc):
    edges = 0
    for body in doc["bodies"]:
        for sub in body["submeshes"]:
            n = sub["vertex_count"]
            for v, nb in enumerate(sub["neighbours"]):
                assert nb == sorted(set(nb)) and v not in nb
                for u in nb:
                    assert 0 <= u < n and v in sub["neighbours"][u]
                edges += len(nb)
            if sub["is_point_cloud"] or n == 1:
                assert all(len(nb) == 0 for nb in sub["neighbours"])
    assert edges > 0


def test_digest_rule_matches_touch_inside_rays():
    """The script's digest rule is touch_inside_rays.skin_layout's on the same arrays."""
    from mimo_bridge import touch_inside_rays as tir
    rng = np.random.default_rng(3)
    arrays = {7: rng.normal(size=(5, 3)), 2: rng.normal(size=(4, 3))}
    fake = type("T", (), {"sensor_positions": arrays})()
    assert tir.skin_layout(fake, [7, 2]) == esl.digest_of_positions([arrays[2], arrays[7]])


def test_neighbour_lists_are_the_mesh_edges_and_empty_for_point_clouds():
    """touch.py:1222-1235: the breadth-first search walks trimesh's vertex_adjacency_graph, the mesh's edges; a point
    cloud (a one-point body) has no edges (touch.py:1268-1272)."""
    import trimesh
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    lists = esl.neighbour_lists(box)
    assert len(lists) == len(box.vertices) == 8
    edges = {tuple(sorted(e)) for e in box.edges_unique.tolist()}
    assert {(v, u) for v, nb in enumerate(lists) for u in nb if v < u} == edges
    assert all(nb == sorted(set(nb)) and v not in nb for v, nb in enumerate(lists))
    assert esl.neighbour_lists(trimesh.PointCloud(np.zeros((1, 3)))) == [[]]
    assert esl.neighbour_lists(trimesh.PointCloud(np.zeros((3, 3)))) == [[], [], []]


def test_digest_from_document_rebuilds_the_digest_and_refuses_unordered_rows():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(3, 3)), rng.normal(size=(2, 3))
    rows = [[4, 0, i] + a[i].tolist() for i in range(3)] + [[9, 0, i] + b[i].tolist() for i in range(2)]
    assert esl.digest_from_document({"points": rows}) == esl.digest_of_positions([a, b])
    with pytest.raises(ValueError, match="body id order"):
        esl.digest_from_document({"points": rows[3:] + rows[:3]})
    with pytest.raises(ValueError, match="split"):
        esl.digest_from_document({"points": rows[:1] + rows[3:] + rows[1:3]})


def test_colliding_geoms_skips_non_colliding_geoms():
    """touch.py:884-889: a geom with contype 0 and conaffinity 0 gets no sensor mesh."""
    class Geom:
        def __init__(self, c, a):
            self.contype = np.array(c)
            self.conaffinity = np.array(a)

    class Model:
        body_geomadr = [0, 2]
        body_geomnum = [2, 3]
        _geoms = [Geom(1, 1), Geom(0, 0), Geom(0, 1), Geom(0, 0), Geom(1, 0)]

        def geom(self, i):
            return self._geoms[i]

    assert esl.colliding_geoms(Model(), 0) == [0]
    assert esl.colliding_geoms(Model(), 1) == [2, 4]
