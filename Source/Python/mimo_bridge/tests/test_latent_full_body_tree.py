# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The whole-body tree (Oct 1, 2026): ownership, the kinematic chain read from MIMo's own model, and the skin map.

Each test names the defect it catches; tests read MIMo_modelv2.xml (no model compiled, no world)."""
import os

import pytest

from lem.latent_body.tree import (BodyNode, BodyTree, PILOT_HANDLES, body_parents_from_xml, body_tree,
                                  check_kinematic_order, from_manifest, full_body_tree, handle_parents, skin_owner,
                                  skin_owner_from_chain, arm_hand_tree)
from lem.model.contact_head import PATCH_NAMES
from mimo_bridge.body_layout import names_for

NAMES = names_for("trunk_v1")


def model_xml():
    try:
        import mimo_infant
    except ImportError:
        pytest.skip("MIMo is not installed")
    path = os.path.join(os.path.dirname(mimo_infant.__file__), "simulation", "assets", "mimo", "MIMo_modelv2.xml")
    if not os.path.exists(path):
        pytest.skip("MIMo_modelv2.xml not found")
    return path


def test_every_handle_is_owned_by_exactly_one_part():
    tree = full_body_tree(NAMES)
    assert sorted(tree.owners) == sorted(NAMES) and len(NAMES) == 26
    assert len(tree.nodes) == 24
    assert tree.roots == ("trunk", "thigh_L", "thigh_R")
    assert tree.owners["gaze"] == "gaze" and tree.by_name["gaze"].parent == "head"
    assert tree.by_name["head"].parent == "trunk" and tree.by_name["chest"].parent == "trunk"
    assert tree.by_name["arm_L"].parent == "chest" and tree.by_name["hand_L"].parent == "arm_L"
    for d in ("thumb", "index", "middle", "ring", "pinky"):
        assert tree.by_name[f"{d}_R"].parent == "hand_R" and tree.owners[f"{d}_R"] == f"{d}_R"
    assert tree.by_name["foot_L"].parent == "calf_L" and tree.by_name["calf_L"].parent == "thigh_L"
    assert tree.by_name["index_L"].peers == ("thumb_L", "middle_L")
    assert from_manifest(tree.manifest()).fingerprint == tree.fingerprint
    assert body_tree(NAMES, "full_body").fingerprint == tree.fingerprint


def test_pilot_trees_keep_their_fingerprints_and_refuse_the_eyes():
    # a manifest field added for the eyes must not change an earlier tree's fingerprint (saved hierarchies bind it)
    import hashlib
    import json
    from dataclasses import asdict
    pilot = arm_hand_tree(NAMES)
    assert "eyes" not in pilot.manifest()
    assert body_tree(NAMES, "arm,hand,index").fingerprint == pilot.fingerprint
    earlier = {"handles": list(NAMES), "nodes": [asdict(n) for n in pilot.nodes]}   # the manifest before Oct 1
    assert pilot.fingerprint == hashlib.sha256(json.dumps(earlier, sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError):
        BodyTree(NAMES, [BodyNode("a", None, ("gaze",))])
    assert set(pilot.owners) == set(PILOT_HANDLES)
    with pytest.raises(ValueError):
        body_tree(NAMES, "full_body,arm")


def test_the_kinematic_chain_is_mimos_own():
    parents = handle_parents(body_parents_from_xml(model_xml()), NAMES)
    assert parents["thigh_L"] is None and parents["lower_trunk"] is None          # both hang from the free hip
    assert parents["head"] == "lower_trunk" and parents["chest"] == "lower_trunk"  # through the upper body
    assert parents["gaze"] == "head" and parents["upperarm_R"] == "chest"
    assert parents["index_L"] == "wrist_L" and parents["foot_R"] == "calf_R"
    assert check_kinematic_order(full_body_tree(NAMES), parents)
    assert check_kinematic_order(arm_hand_tree(NAMES), parents)


@pytest.mark.parametrize("defect", ["head_under_chest", "legs_under_trunk", "foot_under_thigh", "digit_under_arm"])
def test_planted_chain_defects_are_refused(defect):
    parents = handle_parents(body_parents_from_xml(model_xml()), NAMES)
    tree = full_body_tree(NAMES)
    nodes = list(tree.nodes)
    edit = {"head_under_chest": ("head", "chest"), "legs_under_trunk": ("thigh_L", "trunk"),
            "foot_under_thigh": ("foot_L", "thigh_L"), "digit_under_arm": ("index_L", "arm_L")}[defect]
    nodes = [BodyNode(n.name, edit[1] if n.name == edit[0] else n.parent, n.handles,
                      () if n.name == edit[0] or edit[0] in n.peers else n.peers) for n in nodes]
    if defect == "head_under_chest":
        nodes = [BodyNode(n.name, n.parent, n.handles, tuple(p for p in n.peers if p != "head")) for n in nodes]
    key = {n.name: (depth(nodes, n), k) for k, n in enumerate(nodes)}
    nodes = sorted(nodes, key=lambda n: key[n.name])
    planted = BodyTree(NAMES, nodes, eyes=True)
    with pytest.raises(ValueError, match="kinematic chain"):
        check_kinematic_order(planted, parents)


def depth(nodes, node):
    by = {n.name: n for n in nodes}
    d = 0
    while node.parent is not None:
        node, d = by[node.parent], d + 1
    return d


def test_every_skin_patch_maps_to_the_handle_whose_joints_move_it():
    parents = body_parents_from_xml(model_xml())
    owners = {name: skin_owner(name) for name in PATCH_NAMES}
    assert len(PATCH_NAMES) == 53 and None not in owners.values()
    for name in PATCH_NAMES:
        assert owners[name] == skin_owner_from_chain(name, parents, NAMES), name
    assert owners["hip"] == owners["lower_body"] == owners["upper_body"] == "lower_trunk"
    assert owners["chest"] == "chest" and owners["head"] == "head"
    assert owners["left_big_toe"] == owners["left_toes"] == owners["left_foot"] == "foot_L"
    assert owners["right_lower_leg"] == "calf_R" and owners["right_upper_leg"] == "thigh_R"
    assert owners["left_ffknuckle"] == "index_L" and owners["right_thbase"] == "thumb_R"
    covered = set(owners.values())
    assert covered == set(NAMES) - {"gaze"}                          # the eyes carry no skin


def test_planted_skin_defect_is_caught_by_the_chain_rule():
    parents = body_parents_from_xml(model_xml())
    assert skin_owner_from_chain("left_toes", parents, NAMES) == "foot_L"
    # a toe mapped to the calf, or the upper body mapped to the chest, disagrees with the chain
    assert skin_owner_from_chain("left_toes", parents, NAMES) != "calf_L"
    assert skin_owner_from_chain("upper_body", parents, NAMES) != "chest"
    assert skin_owner_from_chain("right_lfmetacarpal", parents, NAMES) == "pinky_R"
