# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import pytest

from lem.latent_body.tree import BodyNode, BodyTree, arm_hand_tree, from_manifest, skin_owner
from mimo_bridge.body_layout import names_for


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_arm_hand_index_ownership_and_skin_are_explicit(layout):
    tree = arm_hand_tree(names_for(layout))
    assert len(tree.owners) == 16
    assert tree.owners["index_L"] == "index_L"
    assert tree.owners["wrist_L"] == "hand_L"
    assert tree.owners["thumb_L"] == "hand_L"
    assert "gaze" not in tree.owners and "chest" not in tree.owners
    index = tree.handle_names.index("index_L")
    assert index in tree.subtree["hand_L"] and index in tree.subtree["arm_L"]
    assert index not in tree.owned["hand_L"]
    assert skin_owner("left_ffknuckle") == "index_L"
    assert skin_owner("right_thhub") == "thumb_R"
    assert skin_owner("left_lower_arm") == "lowerarm_L"
    # every patch has an owner since the whole-body tree (Oct 1, 2026); the pilot's observations still set contact
    # bits only for the handles its tree owns (latent_body_runtime.observation)
    assert skin_owner("left_foot") == "foot_L" and "foot_L" not in tree.owners
    assert from_manifest(tree.manifest()).fingerprint == tree.fingerprint


def test_optional_thumb_has_a_peer_context_link_and_its_own_handles():
    tree = arm_hand_tree(names_for("trunk_v1"), ("arm", "hand", "index", "thumb"))
    assert tree.by_name["index_L"].peers == ("thumb_L",)
    assert tree.by_name["thumb_L"].peers == ("index_L",)
    assert tree.owners["thumb_L"] == "thumb_L"
    assert tree.by_name["arm_L"].peers == ("arm_R",)


@pytest.mark.parametrize("nodes", [
    [BodyNode("a", "b", ("index_L",)), BodyNode("b", "a", ("thumb_L",))],
    [BodyNode("a", None, ("index_L",)), BodyNode("b", "a", ("index_L",))],
    [BodyNode("a", None, ("gaze",))],
    [BodyNode("a", None, ("index_L",), ("missing",))],
    [BodyNode("a", None, ("index_L",), ("a",))],
])
def test_mapping_rejects_cycles_duplicate_ownership_eyes_and_bad_links(nodes):
    with pytest.raises(ValueError):
        BodyTree(names_for("trunk_v1"), nodes)


def test_unknown_level_or_missing_required_handle_is_refused():
    with pytest.raises(ValueError):
        arm_hand_tree(names_for("legacy"), ("arm", "hand", "knee"))
    with pytest.raises(ValueError):
        arm_hand_tree(tuple(n for n in names_for("legacy") if n != "index_L"))
