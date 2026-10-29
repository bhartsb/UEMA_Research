# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Explicit functional ownership, with separately declared peer context links."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class BodyNode:
    name: str
    parent: str | None
    handles: tuple[str, ...]
    peers: tuple[str, ...] = ()


class BodyTree:
    def __init__(self, handle_names, nodes, *, eyes=False):
        # EYES (whole-body tree, Oct 1, 2026). The gaze handle is the eyes' orientation relative to the head, read
        # through its own servo path (action_decoder: empty joint chain; lem_model._gaze_from_handle: yaw and pitch,
        # clamped to the eye ranges; position, radius and strength ignored). A tree owns it only when it says so
        # (eyes=True), and a learner holding it must declare the whole-body row convention (trajectory.BodyRows),
        # which treats it as orientation only and converts its command to the eyes' physical convention.
        self.eyes = bool(eyes)
        self.handle_names = tuple(handle_names)
        self.nodes = tuple(nodes)
        self.by_name = {node.name: node for node in self.nodes}
        if not self.nodes or len(self.by_name) != len(self.nodes):
            raise ValueError("Body node names must be nonempty and unique")
        if len(set(self.handle_names)) != len(self.handle_names):
            raise ValueError("Duplicate handle name")
        owners = {}
        visited = set()
        for node in self.nodes:
            if not node.name or "." in node.name:
                raise ValueError("Node names must be nonempty module keys without dots")
            if node.parent is not None and node.parent not in visited:
                raise ValueError("Parents must precede children; cycles are forbidden")
            visited.add(node.name)
            for handle in node.handles:
                if handle not in self.handle_names or (handle == "gaze" and not self.eyes) or handle in owners:
                    raise ValueError(f"Invalid or multiply owned handle: {handle}")
                owners[handle] = node.name
            if len(set(node.peers)) != len(node.peers):
                raise ValueError("Duplicate lateral link")
            for peer in node.peers:
                if peer == node.name or peer not in self.by_name:
                    raise ValueError("Unknown or self lateral link")
                if self.by_name[peer].parent != node.parent:
                    raise ValueError("Lateral links join peers at the same ownership level")
        self.owners = owners
        self.roots = tuple(n.name for n in self.nodes if n.parent is None)
        self.children = {n.name: tuple(c.name for c in self.nodes if c.parent == n.name)
                         for n in self.nodes}
        self.owned = {n.name: tuple(self.handle_names.index(h) for h in n.handles)
                      for n in self.nodes}
        self.subtree = {}
        for n in reversed(self.nodes):
            ids = list(self.owned[n.name])
            for child in self.children[n.name]:
                ids.extend(self.subtree[child])
            self.subtree[n.name] = tuple(sorted(ids))
            if not ids:
                raise ValueError(f"Empty subtree: {n.name}")
        if any(not self.owned[r] for r in self.roots):
            raise ValueError("Every root owns a handle (its first owned handle anchors the chain's positions)")

    def manifest(self):
        out = {"handles": list(self.handle_names), "nodes": [asdict(n) for n in self.nodes]}
        if self.eyes:          # absent for every earlier tree, so their fingerprints are unchanged
            out["eyes"] = True
        return out

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.manifest(), sort_keys=True).encode()).hexdigest()


DIGITS = ("thumb", "index", "middle", "ring", "pinky")
# the handles of the arm-and-hand pilot (arm_hand_tree); a learner whose tree owns any other handle must declare the
# whole-body row convention (trajectory.BodyRows)
PILOT_HANDLES = frozenset(f"{h}_{s}" for s in ("L", "R") for h in ("upperarm", "lowerarm", "wrist", *DIGITS))


def arm_hand_tree(handle_names, levels=("arm", "hand", "index"), sides=("L", "R")):
    """Two arm/hand/index chains; optional digit modules remain children of the hand.

    The hand owns its wrist and any digit without a separate module. The arm owns
    upper/lower arm handles. This functional map does not give wrist joints
    anatomical ownership of fingers. Other body parts stay outside this pilot.
    """
    levels, sides = tuple(levels), tuple(sides)
    digits = DIGITS
    if (not {"arm", "hand", "index"}.issubset(levels)
            or set(levels) - {"arm", "hand", *digits} or len(set(levels)) != len(levels)):
        raise ValueError("Pilot levels require arm,hand,index, with optional named digits")
    if not sides or set(sides) - {"L", "R"} or len(set(sides)) != len(sides):
        raise ValueError("Sides must be unique L and/or R")
    nodes = []
    for side in sides:
        other = "R" if side == "L" else "L"
        nodes.append(BodyNode(f"arm_{side}", None,
                              (f"upperarm_{side}", f"lowerarm_{side}"),
                              (f"arm_{other}",) if other in sides else ()))
        nodes.append(BodyNode(f"hand_{side}", f"arm_{side}",
                              (f"wrist_{side}",) + tuple(f"{d}_{side}" for d in digits if d not in levels)))
        for digit in digits:
            if digit in levels:
                partner = "thumb" if digit == "index" else "index" if digit == "thumb" else None
                peers = (f"{partner}_{side}",) if partner in levels else ()
                nodes.append(BodyNode(f"{digit}_{side}", f"hand_{side}", (f"{digit}_{side}",), peers))
    return BodyTree(handle_names, nodes)


# THE WHOLE BODY (operator, Oct 1, 2026: "Let's just do the entire body"). The chain is MIMo's own (MIMo_modelv2.xml;
# check_kinematic_order verifies a tree against any compiled model's body_parentid or the XML):
#   hip                      the free root: no handle, no motor
#     lower_body             lower_trunk   hip_lean1, hip_rot1, hip_bend1
#       upper_body           (no handle)   hip_lean2/rot2/bend2, held to the pelvis hinges by fixed tendons, no motor
#         chest              chest         chest_lean, chest_rot
#           upper_arm        upperarm_X    three shoulder hinges
#             lower_arm      lowerarm_X    elbow
#               hand         wrist_X       hand1-3
#                 five digit chains, the distal body tracked (thumb_X ... pinky_X)
#         head               head          head_swivel, head_tilt, head_tilt_side
#           left_eye         gaze          (both eyes yoked by the eye servo; the right eye is not a handle)
#     upper_leg              thigh_X       hip1-3
#       lower_leg            calf_X        knee
#         foot               foot_X        foot1-3 (toes and big toe: no handle, no motor)
# So the thighs hang from the free hip, beside the pelvis, not from it, and the head hangs from the upper body,
# beside the chest, not from it. Parts follow that chain: three roots that all push on the free hip (trunk owning
# lower_trunk, thigh_L, thigh_R; peers of each other), chest and head as peers under the trunk (both carried by
# the upper body), the arms as peers under the chest, each hand under its arm, the five digits as their own parts
# under the hand (neighbouring digits are peers), the eyes under the head, and calf then foot under each thigh.
# The arm part keeps the upper arm and forearm together because the decoder solves the wrist, forearm and upper
# arm of one side as one stacked group (action_decoder.arm_groups); every leg segment is its own handle chain in
# the decoder, so each is its own part.
FULL_BODY = "full_body"


def full_body_tree(handle_names):
    names = tuple(handle_names)
    nodes = [BodyNode("trunk", None, ("lower_trunk",), ("thigh_L", "thigh_R")),
             BodyNode("thigh_L", None, ("thigh_L",), ("thigh_R", "trunk")),
             BodyNode("thigh_R", None, ("thigh_R",), ("thigh_L", "trunk")),
             BodyNode("chest", "trunk", ("chest",), ("head",)),
             BodyNode("head", "trunk", ("head",), ("chest",))]
    for side in ("L", "R"):
        other = "R" if side == "L" else "L"
        nodes.append(BodyNode(f"arm_{side}", "chest", (f"upperarm_{side}", f"lowerarm_{side}"), (f"arm_{other}",)))
    for side in ("L", "R"):
        nodes.append(BodyNode(f"hand_{side}", f"arm_{side}", (f"wrist_{side}",)))
    for side in ("L", "R"):
        for k, digit in enumerate(DIGITS):
            neighbours = tuple(f"{DIGITS[j]}_{side}" for j in (k - 1, k + 1) if 0 <= j < len(DIGITS))
            nodes.append(BodyNode(f"{digit}_{side}", f"hand_{side}", (f"{digit}_{side}",), neighbours))
    nodes.append(BodyNode("gaze", "head", ("gaze",)))
    for side in ("L", "R"):
        nodes.append(BodyNode(f"calf_{side}", f"thigh_{side}", (f"calf_{side}",)))
    for side in ("L", "R"):
        nodes.append(BodyNode(f"foot_{side}", f"calf_{side}", (f"foot_{side}",)))
    tree = BodyTree(names, nodes, eyes=True)
    if set(tree.owners) != set(names):
        raise ValueError("The whole-body tree must own every handle of the layout: missing %s"
                         % sorted(set(names) - set(tree.owners)))
    return tree


def body_tree(handle_names, levels):
    """--latent-body-levels: "full_body" (every handle) or the pilot's arm,hand,index[,digits]."""
    levels = tuple(levels.split(",")) if isinstance(levels, str) else tuple(levels)
    if levels == (FULL_BODY,):
        return full_body_tree(handle_names)
    if FULL_BODY in levels:
        raise ValueError("full_body is a level set of its own")
    return arm_hand_tree(handle_names, levels)


# ----------------------------------------------------------------------------- the chain, read from the model
def tracked_bodies(handle_names):
    """{handle: the MIMo body whose pose its bone row reports} (proprio_adapter.BONE_MAP, body_layout.TRUNK_BODIES)."""
    from mimo_bridge.body_layout import TRUNK_BODIES
    from mimo_bridge.proprio_adapter import BONE_MAP
    table = {**BONE_MAP, **TRUNK_BODIES}
    return {h: table[h] for h in handle_names}


def body_parents_from_model(model):
    """{body name: parent body name or None} from a compiled MuJoCo model (body_parentid; the world is None)."""
    out = {}
    for b in range(1, model.nbody):
        p = int(model.body_parentid[b])
        out[model.body(b).name] = model.body(p).name if p > 0 else None
    return out


def body_parents_from_xml(path):
    """The same map from MIMo's body file (MIMo_modelv2.xml), without compiling a model."""
    import xml.etree.ElementTree as ET
    out = {}

    def walk(element, parent):
        for body in element.findall("body"):
            out[body.get("name")] = parent
            walk(body, body.get("name"))
    walk(ET.parse(path).getroot(), None)
    return out


def handle_parents(body_parent, handle_names):
    """{handle: the nearest tracked ancestor's handle, or None} following the body chain."""
    bodies = tracked_bodies(handle_names)
    handle_of = {b: h for h, b in bodies.items()}
    out = {}
    for h, body in bodies.items():
        if body not in body_parent:
            raise ValueError(f"Body {body} of handle {h} is not in the model")
        b = body_parent[body]
        while b is not None and b not in handle_of:
            b = body_parent.get(b)
        out[h] = handle_of.get(b)
    return out


def check_kinematic_order(tree, parents):
    """Ownership follows the body chain: a handle whose nearest tracked ancestor is held by this tree is owned by
    the same part or by that part's parent; a handle with no tracked ancestor in the tree is owned by a root; and
    every non-root part owns a handle carried by a handle of its parent part. `parents`: handle_parents()."""
    problems = []
    for handle, part in tree.owners.items():
        node, above = tree.by_name[part], parents.get(handle)
        if above is None or above not in tree.owners:
            if node.parent is not None:
                problems.append(f"{handle} has no tracked ancestor in the tree but {part} is not a root")
        elif tree.owners[above] not in (part, node.parent):
            problems.append(f"{handle} is carried by {above} ({tree.owners[above]}), not by {part} or its parent")
    for node in tree.nodes:
        if node.parent is not None and not any(
                parents.get(h) in tree.owners and tree.owners[parents[h]] == node.parent for h in node.handles):
            problems.append(f"{node.name} owns no handle carried by its parent part {node.parent}")
    if problems:
        raise ValueError("Body tree does not follow the kinematic chain: " + "; ".join(problems))
    return True


# ----------------------------------------------------------------------------- skin
# WHOLE-BODY SKIN (Oct 1, 2026). Every sensing body maps to the handle whose joints move it: its own handle, the
# digit whose chain it lies on (the knuckle, middle, metacarpal, base and hub segments of a digit are moved by that
# digit's command; action_decoder.HANDLE_CHAINS), or else its nearest tracked ancestor (toes and big toe: the foot;
# the upper body: the pelvis, whose hinges it follows through MIMo's fixed tendons, limb_target_control._Chest).
# The free hip has no ancestor and no motor; its skin goes to lower_trunk, the tracked body hinged to it, the root of
# the trunk chain. The eyes carry no skin (no eye body is in contact_head.PATCH_NAMES).
_SIDE_SKIN = {"upper_arm": "upperarm", "lower_arm": "lowerarm", "hand": "wrist",
              "upper_leg": "thigh", "lower_leg": "calf", "foot": "foot", "toes": "foot", "big_toe": "foot"}
_TRUNK_SKIN = {"hip": "lower_trunk", "lower_body": "lower_trunk", "upper_body": "lower_trunk",
               "chest": "chest", "head": "head"}
SKINLESS = frozenset({"gaze"})


def skin_owner(body_name):
    """MIMo sensing body -> handle, including the proximal skin of each digit."""
    if body_name in _TRUNK_SKIN:
        return _TRUNK_SKIN[body_name]
    for prefix, side in (("left_", "L"), ("right_", "R")):
        if body_name.startswith(prefix):
            local = body_name[len(prefix):]
            if local in _SIDE_SKIN:
                return f"{_SIDE_SKIN[local]}_{side}"
            for short, digit in (("ff", "index"), ("mf", "middle"), ("rf", "ring"),
                                 ("lf", "pinky"), ("th", "thumb")):
                if local.startswith(short) and local[len(short):] in {
                        "knuckle", "middle", "distal", "metacarpal", "base", "hub"}:
                    return f"{digit}_{side}"
    return None


def skin_owner_from_chain(body, body_parent, handle_names):
    """The rule skin_owner states, computed from the body chain alone (to check skin_owner against the model)."""
    bodies = tracked_bodies(handle_names)
    handle_of = {b: h for h, b in bodies.items()}
    wrists = {bodies[h]: h for h in handle_names if h.startswith("wrist_")}
    # a body on a digit's chain (between a tracked distal body and its wrist) belongs to that digit
    for h in handle_names:
        if h.split("_")[0] in DIGITS:
            b = bodies[h]
            while b is not None and b not in wrists:
                if b == body:
                    return h
                b = body_parent.get(b)
    b = body
    while b is not None:
        if b in handle_of:
            return handle_of[b]
        b = body_parent.get(b)
    # no tracked ancestor: the free root; its tracked child in the trunk chain
    children = [h for h, tb in bodies.items() if body_parent.get(tb) == body and h == "lower_trunk"]
    return children[0] if children else None


def from_manifest(manifest):
    return BodyTree(manifest["handles"], [BodyNode(n["name"], n["parent"], tuple(n["handles"]),
                                                  tuple(n["peers"])) for n in manifest["nodes"]],
                    eyes=bool(manifest.get("eyes", False)))


def bind_mujoco(tree, model, patch_names):
    """Resolve names against the actual body without moving it or constructing a world.

    Digit handles cover the chain from their distal body up to (excluding) the
    wrist. Other handles own joints on their named body; the gaze handle owns both
    eyes' joints (the eye servo drives them together). The exported mapping
    names joint/sensor membership; observations keep the bridge's physical units.
    The tree's ownership is checked against the model's body chain, and every skin
    patch's owner against the chain rule (skin_owner_from_chain).
    """
    from mimo_bridge.proprio_adapter import BONE_MAP
    parents = body_parents_from_model(model)
    check_kinematic_order(tree, handle_parents(parents, tree.handle_names))
    for name in patch_names:
        named, chained = skin_owner(name), skin_owner_from_chain(name, parents, tree.handle_names)
        if (named in tree.owners or chained in tree.owners) and named != chained:
            raise ValueError(f"Skin patch {name}: skin_owner and the body chain disagree")
    bodies_of = tracked_bodies(tree.handle_names)
    bindings = {}
    for handle, owner in tree.owners.items():
        body = int(model.body(bodies_of[handle]).id)
        bodies = [body]
        if handle.split("_")[0] in DIGITS:
            wrist = int(model.body(BONE_MAP["wrist_" + handle[-1]]).id)
            parent = int(model.body_parentid[body])
            while parent != wrist:
                if parent == 0:
                    raise ValueError("Digit ancestry does not reach the declared wrist")
                bodies.append(parent)
                parent = int(model.body_parentid[parent])
        if handle == "gaze":
            bodies.append(int(model.body("right_eye").id))
        joints = []
        for b in reversed(bodies):
            first, count = int(model.body_jntadr[b]), int(model.body_jntnum[b])
            joints.extend(model.joint(j).name for j in range(first, first + count))
        patches = tuple(i for i, name in enumerate(patch_names) if skin_owner(name) == handle)
        if not patches and handle not in SKINLESS:
            raise ValueError(f"No skin sensor mapped to {handle}")
        bindings[handle] = dict(owner=owner, body=bodies_of[handle], joints=joints,
                                skin_indices=patches, skin_names=[patch_names[i] for i in patches])
    return bindings
