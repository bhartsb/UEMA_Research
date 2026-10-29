# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Physical, completed transitions; no learned tokens or proposed actions as targets."""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math

import torch

from .hierarchy import STATE_DIM, CHANGE_DIM, ACTION_DIM


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone()


def quaternion(q):
    norm = q.norm(dim=-1, keepdim=True)
    if not torch.isfinite(q).all() or (norm <= 1e-8).any():
        raise ValueError("Invalid physical quaternion")
    q = q / norm
    # q and -q name the same orientation, including a half turn (w == 0).
    pivot = torch.where(q[..., 3:4] != 0, q[..., 3:4],
                        q.gather(-1, q.abs().argmax(-1, keepdim=True)))
    return torch.where(pivot < 0, -q, q)


def rotation_delta(start, end):
    from lem.model.motor_synergy import quat_mul
    start, end = quaternion(start), quaternion(end)
    inverse = torch.cat((-start[..., :3], start[..., 3:]), -1)
    relative = quaternion(quat_mul(end, inverse))
    length = relative[..., :3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, relative[..., 3:4])
    factor = torch.where(length > 1e-7, angle / length.clamp_min(1e-7), 2.0)
    return relative[..., :3] * factor


@dataclass
class Observation:
    env_id: int
    episode_id: int
    frame: int
    sim_time: float
    bones: torch.Tensor  # [H,11]: xyzw, world cm, world cm/s, capacity
    contact: torch.Tensor  # [H]: any contact on this handle's mapped skin

    def __post_init__(self):
        self.bones, self.contact = tensor(self.bones), tensor(self.contact)
        if any(type(v) is not int for v in (self.env_id, self.episode_id, self.frame)):
            raise ValueError("Body, episode and frame identifiers must be integers")

    def validate(self, count):
        if (self.bones.shape != (count, 11) or self.contact.shape != (count,)
                or not math.isfinite(self.sim_time)):
            raise ValueError("Observation layout or time does not match")
        if not torch.isfinite(self.bones).all() or not torch.isfinite(self.contact).all():
            raise ValueError("Non-finite physical observation")
        if ((self.contact != 0) & (self.contact != 1)).any():
            raise ValueError("Contact must be an observed binary event")
        quaternion(self.bones[:, :4])

    def state_dict(self):
        return dict(env_id=self.env_id, episode_id=self.episode_id, frame=self.frame,
                    sim_time=self.sim_time, bones=self.bones.clone(), contact=self.contact.clone())

    @classmethod
    def from_dict(cls, value):
        return cls(value["env_id"], value["episode_id"], value["frame"],
                   float(value["sim_time"]), tensor(value["bones"]), tensor(value["contact"]))


@dataclass
class Example:
    history: Observation
    start: Observation
    end: Observation
    executed: torch.Tensor  # [H,9], canonical issued target in cm, xyzw, radius, strength
    action_contract: str

    def __post_init__(self):
        self.executed = tensor(self.executed)

    def validate(self, tree, *, duration_s, action_contract, horizon_steps=1):
        """horizon_steps (HOLD HORIZON, Oct 1, 2026; config key horizon_steps, absent = 1): the end lies that many
        model steps after the start (one held order, hold_example); the history stays one model step before it."""
        for obs in (self.history, self.start, self.end):
            obs.validate(len(tree.handle_names))
        if self.action_contract != action_contract:
            raise ValueError("Action contract changed; do not mix command meanings")
        for (a, b), steps in (((self.history, self.start), 1), ((self.start, self.end), horizon_steps)):
            if (a.env_id != b.env_id or a.episode_id != b.episode_id or b.frame != a.frame + steps
                    or not math.isclose(b.sim_time - a.sim_time, steps * duration_s, rel_tol=1e-6, abs_tol=1e-9)):
                raise ValueError("History crosses a body, episode, frame gap or duration boundary "
                                 f"(the end must lie {steps} model step(s) after the start)")
        if self.executed.shape != (len(tree.handle_names), 9) or not torch.isfinite(self.executed).all():
            raise ValueError("Invalid executed command")
        quaternion(self.executed[:, 3:7])


@dataclass
class LocalBatch:
    state: dict
    history: dict
    action: dict
    change: dict
    contact: dict
    # WHOLE-BODY LESSON WEIGHTS (BodyRows, Oct 1, 2026): per part, one weight per change / action / contact number,
    # shared by every example. None (every earlier configuration) = every number counts equally, as before.
    change_weight: dict | None = None
    action_weight: dict | None = None
    contact_weight: dict | None = None
    consistency_weight: dict | None = None


def validate_scales(scales):
    expected = {"state": STATE_DIM, "change": CHANGE_DIM, "action": ACTION_DIM}
    if set(scales) != set(expected):
        raise ValueError("Supply measured state, change and action scales")
    for key, dim in expected.items():
        v = tensor(scales[key])
        if v.shape != (dim,) or not torch.isfinite(v).all() or (v <= 0).any():
            raise ValueError(f"{key} scales must contain {dim} positive finite numbers")


HAND_FRAMES = ("recorded", "actual_wrist")
DIGITS = ("thumb", "index", "middle", "ring", "pinky")


def quat_rotate(q, v):
    from lem.model.motor_synergy import quat_mul
    pure = torch.cat((v, torch.zeros_like(v[..., :1])), -1)
    conj = torch.cat((-q[..., :3], q[..., 3:]), -1)
    return quat_mul(quat_mul(q, pure), conj)[..., :3]


def move_digits(command, handle_names, source, target):
    """Carry each side's digit rows from wrist pose `source` [H,7 cm+xyzw rows] to wrist pose `target`.

    Under wrist_relative_joint_v1 the recorded digit rows are the finger goal placed on the REQUESTED wrist, and
    execution regenerates the same goal around the ACTUAL wrist before every physics frame. Measured Sep 30, 2026
    on 2,000 H84 seg41 rows: the recorded digit target sat a median 14.15 cm from the finger, the re-anchored one
    0.85 cm, and the re-anchored target matched the first physics frame's row within a median 0.12 cm.
    """
    from lem.model.motor_synergy import quat_mul
    out = command.clone()
    for side in ("L", "R"):
        name = "wrist_" + side
        if name not in handle_names:
            continue
        w = handle_names.index(name)
        ids = [handle_names.index(f"{d}_{side}") for d in DIGITS if f"{d}_{side}" in handle_names]
        if not ids:
            continue
        qs, qt = quaternion(source[..., w, 3:7]), quaternion(target[..., w, 3:7])
        qs_inv = torch.cat((-qs[..., :3], qs[..., 3:]), -1)
        qsi = qs_inv.unsqueeze(-2).expand(*qs_inv.shape[:-1], len(ids), 4)
        qtt = qt.unsqueeze(-2).expand(*qt.shape[:-1], len(ids), 4)
        rel_p = quat_rotate(qsi, command[..., ids, :3] - source[..., w:w + 1, :3])
        out[..., ids, :3] = quat_rotate(qtt, rel_p) + target[..., w:w + 1, :3]
        rel_q = quat_mul(qsi, quaternion(command[..., ids, 3:7]))
        out[..., ids, 3:7] = quaternion(quat_mul(qtt, rel_q))
    return out


def actual_wrist_frame(executed, start, handle_names):
    """The executed command with digit rows re-anchored from the requested wrist to the wrist at the start."""
    actual = torch.cat((start.bones[..., 4:7], start.bones[..., :4]), -1)
    return move_digits(executed, handle_names, executed[..., :7], actual)


def check_hand_frame(hand_frame, action_contract):
    if hand_frame not in HAND_FRAMES:
        raise ValueError("Unknown hand action frame")
    if hand_frame == "actual_wrist" and "wrist_relative_joint_v1" not in str(action_contract).split("+"):
        raise ValueError("The actual-wrist hand frame applies only to the wrist_relative_joint_v1 contract")


# DIGITS IN THEIR WRIST'S FRAME (index-level repair, Sep 30, 2026; docs/gain_of_function_v4/agents/latent_index_2026-09-30.md).
# digit_frame "world" (absent = the earlier meaning): every row is a world-frame quantity. 98-99% of a finger's
# one-step world movement is the arm and wrist carrying it, so the index code learned that carried movement (a
# linear read-out of the index code gives the wrist's own step with R^2 0.47-0.62) and the index level predicted
# movement its finger command cannot make. digit_frame "wrist": every digit row, in every part that holds it, is the
# digit relative to its own side's wrist, in the wrist's coordinates: position (wrist to digit), orientation
# (wrist^-1 * digit), velocity (difference of world velocities), one-step change (of that relative pose) and command
# (target minus digit, turn of the relative orientation). A digit's world pose follows from its wrist row, which
# every part above the digit level holds (end_positions).
DIGIT_FRAMES = ("world", "wrist")


def digit_wrists(handle_names):
    """{digit row: its side's wrist row} for every digit handle whose wrist is in the layout."""
    out = {}
    for side in ("L", "R"):
        if "wrist_" + side not in handle_names:
            continue
        w = handle_names.index("wrist_" + side)
        for d in DIGITS:
            if f"{d}_{side}" in handle_names:
                out[handle_names.index(f"{d}_{side}")] = w
    return out


@functools.lru_cache(maxsize=16)
def _digits(handle_names):
    return digit_wrists(handle_names)


def check_digit_frame(digit_frame, hand_frame, digit_scales=None):
    if digit_frame not in DIGIT_FRAMES:
        raise ValueError("Unknown digit frame")
    if digit_frame == "wrist" and hand_frame != "actual_wrist":
        raise ValueError("The wrist digit frame requires finger commands on the actual wrist "
                         "(hand_action_frame actual_wrist)")
    if digit_scales is not None:
        validate_scales(digit_scales)


# ============================================================================ THE WHOLE BODY (Oct 1, 2026)
# Config key "body_rows" (absent = every earlier meaning). Every row is in the frame its command contract gives
# (BODY_ROWS_VERSION "contract_v1", read from the action contract the data was recorded under):
#   digits      relative to their wrist (wrist_relative_joint_v1; digit_frame "wrist" is required with it)
#   gaze        the eyes relative to the head: the bone row's orientation already is (proprio_adapter, B8), its
#               position and velocity are made head-relative here; its command row is converted to the eyes'
#               physical convention (EYE_ROW_CONVENTIONS) and only its turn is learned
#   calves      relative to their thigh only when the contract holds parent_relative_calf_v1 (the knee goal on the
#               thigh's ORDERED row, regenerated around the actual thigh every 10 ms frame, limb_target_control);
#               their commands are then carried from the ordered to the actual thigh, as the fingers' are
#   chest       parent_relative_chest_v1 is refused: its goal sits on the upper body the pelvis's ORDERED row implies,
#               a body no bone row reports, so no recorded row can re-anchor it
#   all others  world rows, as their world targets are (wrist_relative_joint_v1 and world_v0)
# Learned command numbers (action weights): the position of a handle whose position an executor reads
# (action_decoder.position_handles under the contract: the wrists, the digits, and the head and the pelvis unless
# no_head_pelvis_position_v1) and every handle's turn. Radius is read by nothing and the hierarchy issues no
# strength (handle_targets keeps the controller's), so neither is learned as a command; both stay encoder inputs.
# Learned change numbers: all, except the eyes' position (rigid in the head). Learned contact numbers: handles with
# skin (not the eyes) whose contact the data recorded (contact_recorded; the shadows before Oct 1 recorded the arm
# and hand skin only, latent_body_runtime.observation).
# part_scales: each handle's rows are scaled by its group's measured root mean squares (PART_GROUPS), so a small
# part is not drowned by a large one in its own lessons.
# future_rows: which subtree rows a part's future lesson grades (FUTURE_ROWS; see body_lesson_weights).
BODY_ROWS_VERSION = "contract_v1"
EYE_ROW_CONVENTIONS = ("legacy", "physical")
PART_GROUPS = {"trunk": ("lower_trunk", "chest"), "head": ("head",), "gaze": ("gaze",),
               "arm": ("upperarm_L", "lowerarm_L", "upperarm_R", "lowerarm_R"), "hand": ("wrist_L", "wrist_R"),
               "digit": tuple(f"{d}_{s}" for s in ("L", "R") for d in DIGITS),
               "leg": tuple(f"{p}_{s}" for s in ("L", "R") for p in ("thigh", "calf", "foot"))}
GROUP_OF = {h: g for g, members in PART_GROUPS.items() for h in members}
FUTURE_ROWS = ("subtree", "own")
CALF_PAIRS = (("calf_L", "thigh_L"), ("calf_R", "thigh_R"))
# CARRIED ROWS (Oct 1, 2026). Each of these bodies turns about hinges at (or, the foot, 1.5 cm from) its own origin
# (MIMo_modelv2.xml), its position is read by no executor (action_decoder.position_handles), and its tracked parent
# is owned by another part, so its origin moves only when an ancestor turns: measured on 3,000 seg41_v2 rows the share
# of its one-step world displacement left after the parent's carry is 0.0 for the upper arms and calves and 0.5-0.9%
# for the feet (agents/latent_full_body_2026-10-01/carry_shares.json). In world rows that displacement taught the part
# its parent's movement (a small whole-body fit: the foot code read the calf's world step with R^2 0.49-0.55, the calf
# code the thigh's with 0.25-0.26). As carried rows their position is relative to the parent (constant up to the
# chest's tendon-coupled upper body) and not graded; their orientation stays the world orientation they are commanded
# in. Their world end position follows from the parent's (end_positions). The chest sits on the upper body, which the
# pelvis's hinges set through MIMo's fixed tendons; the head is carried too when its position row is read by nothing
# (no_head_pelvis_position_v1).
CARRIED = (("chest", "lower_trunk"), ("upperarm_L", "chest"), ("upperarm_R", "chest"), ("calf_L", "thigh_L"),
           ("calf_R", "thigh_R"), ("foot_L", "calf_L"), ("foot_R", "calf_R"))


@dataclass(frozen=True)
class BodyRows:
    contract: str
    eye_row_convention: str
    relative: tuple            # ((child handle, parent handle, "rigid" | "eye"), ...) besides the digits
    reanchor: tuple            # ((child handle, parent handle), ...) commands carried from ordered to actual parent
    position_read: frozenset   # handles whose command position an executor reads
    contact_recorded: frozenset | None
    part_scales: tuple         # ((group, key, values), ...)
    future_rows: str

    @property
    def positionless(self):
        """Handles whose position change is not graded: the eyes (rigid in the head) and the carried rows."""
        return frozenset(c for c, _, kind in self.relative if kind in KEEP_ORIENTATION)

    def scale(self, handle, key):
        group = GROUP_OF.get(handle)
        for g, k, values in self.part_scales:
            if g == group and k == key:
                return tensor(values)
        raise ValueError(f"No measured {key} scale for {handle} (group {group})")


def body_rows(config):
    """The configuration's BodyRows, or None when it declares none (every earlier configuration)."""
    spec = config.get("body_rows")
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - {"version", "eye_row_convention", "part_scales",
                                                    "contact_recorded", "future_rows"}:
        raise ValueError("body_rows holds version, eye_row_convention, part_scales, optional contact_recorded "
                         "and future_rows")
    if spec.get("version") != BODY_ROWS_VERSION:
        raise ValueError(f"Unknown body_rows version {spec.get('version')!r}")
    if spec.get("eye_row_convention") not in EYE_ROW_CONVENTIONS:
        raise ValueError("Declare the eye row convention the data was recorded under (legacy or physical)")
    future_rows = spec.get("future_rows", "subtree")
    if future_rows not in FUTURE_ROWS:
        raise ValueError(f"future_rows must be one of {FUTURE_ROWS}")
    contract = str(config["action_contract"])
    parts = contract.split("+")
    if "parent_relative_chest_v1" in parts:
        raise ValueError("parent_relative_chest_v1 places the chest goal on the upper body the pelvis's ordered row "
                         "implies; no bone row reports that body, so the chest command cannot be re-anchored")
    if "wrist_relative_joint_v1" in parts:
        if config.get("digit_frame") != "wrist" or config.get("hand_action_frame") != "actual_wrist":
            raise ValueError("Under wrist_relative_joint_v1 the whole-body rows need digit_frame wrist and "
                             "hand_action_frame actual_wrist")
    elif config.get("digit_frame", "world") != "world":
        raise ValueError("Digit rows are wrist-relative only under the wrist_relative_joint_v1 contract")
    if config.get("digit_scales") is not None:
        raise ValueError("Under body_rows digit_scales is not used: the digit group's scales are part_scales['digit']")
    from mimo_bridge.action_decoder import position_handles
    read = position_handles("no_head_pelvis" if "no_head_pelvis_position_v1" in parts else "all")
    calf = "parent_relative_calf_v1" in parts
    scales = spec.get("part_scales")
    if not isinstance(scales, dict) or set(scales) != set(PART_GROUPS):
        raise ValueError(f"part_scales needs measured scales for every group {sorted(PART_GROUPS)}")
    for group, value in scales.items():
        validate_scales(value)
    recorded = spec.get("contact_recorded")
    carried = tuple((c, p, "carried") for c, p in CARRIED if not (calf and c.startswith("calf_")))
    if "no_head_pelvis_position_v1" in parts:
        carried += (("head", "lower_trunk", "carried"),)
    return BodyRows(contract=contract, eye_row_convention=spec["eye_row_convention"],
                    relative=(("gaze", "head", "eye"),) + carried
                    + (tuple((c, p, "rigid") for c, p in CALF_PAIRS) if calf else ()),
                    reanchor=CALF_PAIRS if calf else (),
                    position_read=frozenset(read),
                    contact_recorded=None if recorded is None else frozenset(recorded),
                    part_scales=tuple((g, k, tuple(float(x) for x in v[k])) for g, v in sorted(scales.items())
                                      for k in ("state", "change", "action")),
                    future_rows=future_rows)


def row_options(config):
    """The row convention a configuration declares; absent keys keep the earlier meaning."""
    return dict(digit_frame=config.get("digit_frame", "world"), digit_scales=config.get("digit_scales"),
                body=body_rows(config))


def row_scales(handle_names, ids, scales, key, digit_scales=None, body=None):
    """[len(ids), dim]: each row's measured scale. DIGIT SCALES (index-level repair, Sep 30, 2026): the pooled
    action position scale is 5.75 cm (arm targets lie about 10 cm from the arm) while a finger's target lies 0.85 cm
    from the finger, so the index command's position scored 0.0145 against 1.0 for the whole "command zero" guess.
    digit_scales, when measured, scale the digit rows on their own. Under body_rows every row has its group's scale
    (PART_GROUPS)."""
    if body is not None:
        return torch.stack([body.scale(handle_names[i], key) for i in ids]) if len(ids) else \
            tensor(scales[key]).expand(0, -1)
    base = tensor(scales[key]).expand(len(ids), -1).clone()
    if digit_scales is not None:
        digits = _digits(tuple(handle_names))
        for k, i in enumerate(ids):
            if i in digits:
                base[k] = tensor(digit_scales[key])
    return base


def _inverse(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


# RELATIVE ROW KINDS. "rigid": the whole pose relative to the parent (digits on their wrist; calves on their thigh
# under parent_relative_calf_v1). "eye": position relative to the head, the orientation kept as the bone row reports it
# (already head-relative). "carried": position relative to the parent, the orientation kept in the world, as its world
# command is (see CARRIED below).
KEEP_ORIENTATION = ("eye", "carried")


@functools.lru_cache(maxsize=64)
def _pairs(handle_names, digit_frame, relative):
    """{child row: (parent row, kind)} of every row expressed relative to another."""
    out = {}
    if digit_frame == "wrist":
        out.update({d: (w, "rigid") for d, w in digit_wrists(handle_names).items()})
    for child, parent, kind in relative:
        if child in handle_names and parent in handle_names:
            out[handle_names.index(child)] = (handle_names.index(parent), kind)
    return out


def _relative_split(handle_names, ids, digit_frame, body=None):
    """(positions in ids, child rows, their parent rows, kinds) of the rows expressed relative to a parent."""
    if digit_frame not in DIGIT_FRAMES:
        raise ValueError("Unknown digit frame")
    if digit_frame == "world" and body is None:
        return [], [], [], []
    if handle_names is None:
        raise ValueError("Relative rows need the handle names")
    pairs = _pairs(tuple(handle_names), digit_frame, () if body is None else body.relative)
    rows = [k for k, i in enumerate(ids) if i in pairs]
    return rows, [ids[k] for k in rows], [pairs[ids[k]][0] for k in rows], [pairs[ids[k]][1] for k in rows]


def _digit_split(handle_names, ids, digit_frame):
    rows, child, parent, _ = _relative_split(handle_names, ids, digit_frame)
    return rows, child, parent


def eye_yaw_pitch(q):
    """Yaw (about z) and pitch (about y) of xyzw rows written as Rz(yaw) Ry(pitch) (lem_model._gaze_from_handle)."""
    x, y, z, w = q.unbind(-1)
    return (torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
            torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0)))


def eye_quaternion(yaw, pitch):
    """Rz(yaw) Ry(pitch) as xyzw (lem_model.LemLite.gaze_to_handle)."""
    ca, sa, ce, se = torch.cos(yaw / 2), torch.sin(yaw / 2), torch.cos(pitch / 2), torch.sin(pitch / 2)
    return torch.stack((-sa * se, ca * se, sa * ce, ca * ce), -1)


def eye_row_physical(q, convention):
    """An eye command row <-> the eyes' physical head-relative rotation (the same map both ways).

    Under 'legacy' (every H84 segment: args eye_row_convention legacy) the row's pitch is plus the elevation sent to
    the vertical eye joint, whose physical rotation about y is minus that angle (the left eye's vertical axis is
    0 -1 0, MIMo_modelv2.xml), so the physical rotation is Rz(yaw) Ry(-pitch). Measured Oct 1, 2026 on 6,000 seg41_v2
    rows: the eye's one-step pitch change correlates 0.551 with (-commanded pitch - pitch) and -0.332 with
    (+commanded pitch - pitch); its yaw change 0.515 with (commanded yaw - yaw). Under 'physical' the row already is
    the physical rotation."""
    if convention not in EYE_ROW_CONVENTIONS:
        raise ValueError("Unknown eye row convention")
    q = quaternion(q)
    if convention == "physical":
        return q
    yaw, pitch = eye_yaw_pitch(q)
    return quaternion(eye_quaternion(yaw, -pitch))


def relative_pose(bones, child, parent, kinds=None):
    """Child poses relative to their parents: [n,3] cm in parent coordinates, [n,4] xyzw, [n,3] cm/s. The orientation
    of an "eye" row (already head-relative in the bone rows) or a "carried" row (a world orientation command) is kept."""
    from lem.model.motor_synergy import quat_mul
    inv = _inverse(quaternion(bones[..., parent, :4]))
    pos = quat_rotate(inv, bones[..., child, 4:7] - bones[..., parent, 4:7])
    rel = quaternion(quat_mul(inv, quaternion(bones[..., child, :4])))
    keep = [k in KEEP_ORIENTATION for k in (kinds or [])]
    if any(keep):
        rel = torch.where(torch.tensor(keep)[:, None], quaternion(bones[..., child, :4]), rel)
    vel = quat_rotate(inv, bones[..., child, 7:10] - bones[..., parent, 7:10])
    return pos, rel, vel


def wrist_relative(bones, digit_ids, wrist_ids):
    """Digit poses relative to their wrists: [n,3] cm in wrist coordinates, [n,4] xyzw, [n,3] cm/s."""
    return relative_pose(bones, digit_ids, wrist_ids)


def physical_state(obs, ids, anchor, digit_frame="world", handle_names=None, body=None):
    ids = list(ids)
    b = obs.bones[..., ids, :]
    out = torch.cat((b[..., 4:7] - obs.bones[..., anchor:anchor + 1, 4:7], quaternion(b[..., :4]),
                     b[..., 7:11], obs.contact[..., ids, None]), -1)
    rows, child, parent, eye = _relative_split(handle_names, ids, digit_frame, body)
    if rows:
        pos, rel, vel = relative_pose(obs.bones, child, parent, eye)
        out[..., rows, :3], out[..., rows, 3:7], out[..., rows, 7:10] = pos, rel, vel
    return out


def change_rows(start, end, ids, handle_names=None, digit_frame="world", body=None):
    """[len(ids),6] one-step change, cm displacement and rotation vector, in the declared row convention."""
    ids = list(ids)
    a, b = start.bones[..., ids, :], end.bones[..., ids, :]
    change = torch.cat((b[..., 4:7] - a[..., 4:7], rotation_delta(a[..., :4], b[..., :4])), -1)
    rows, child, parent, eye = _relative_split(handle_names, ids, digit_frame, body)
    if rows:
        p0, q0, _ = relative_pose(start.bones, child, parent, eye)
        p1, q1, _ = relative_pose(end.bones, child, parent, eye)
        change[..., rows, :] = torch.cat((p1 - p0, rotation_delta(q0, q1)), -1)
    return change


def action_rows(start, command, ids, handle_names=None, digit_frame="world", body=None):
    """[len(ids),8] command relative to the start: target displacement, turn, radius, strength."""
    from lem.model.motor_synergy import quat_mul
    ids = list(ids)
    before, cmd = start.bones[..., ids, :], command[..., ids, :]
    action = torch.cat((cmd[..., :3] - before[..., 4:7], rotation_delta(before[..., :4], cmd[..., 3:7]),
                        cmd[..., 7:9]), -1)
    rows, child, parent, eye = _relative_split(handle_names, ids, digit_frame, body)
    if rows:
        inv = _inverse(quaternion(start.bones[..., parent, :4]))
        _, q0, _ = relative_pose(start.bones, child, parent, eye)
        target = quaternion(quat_mul(inv, quaternion(command[..., child, 3:7])))
        carried = [k == "carried" for k in eye]
        if any(carried):        # a world orientation command, as recorded
            target = torch.where(torch.tensor(carried)[:, None], quaternion(command[..., child, 3:7]), target)
        eyes = [k == "eye" for k in eye]
        if any(eyes):
            target = torch.where(torch.tensor(eyes)[:, None],
                                 eye_row_physical(command[..., child, 3:7], body.eye_row_convention), target)
        action[..., rows, :3] = quat_rotate(inv, command[..., child, :3] - start.bones[..., child, 4:7])
        action[..., rows, 3:6] = rotation_delta(q0, target)
    return action


def digit_vectors_to_world(start, ids, vectors, handle_names=None, digit_frame="world", body=None):
    """[len(ids),6] (displacement, turn) in the row convention -> world coordinates. In the wrist frame a turn w of
    the relative orientation is the world turn R_wrist w: rotation_delta(W^-1 A, W^-1 B) = W^-1 (B A^-1) W. An eye
    row's turn stays head-relative (its bone row's orientation is head-relative); its displacement is turned into
    world coordinates like any relative row's."""
    ids = list(ids)
    out = vectors.clone()
    rows, child, parent, eye = _relative_split(handle_names, ids, digit_frame, body)
    if rows:
        qw = quaternion(start.bones[..., parent, :4])
        out[..., rows, :3] = quat_rotate(qw, vectors[..., rows, :3])
        turned = quat_rotate(qw, vectors[..., rows, 3:6])
        keep = [k in KEEP_ORIENTATION for k in eye]
        if any(keep):
            turned = torch.where(torch.tensor(keep)[:, None], vectors[..., rows, 3:6], turned)
        out[..., rows, 3:6] = turned
    return out


def end_positions(start, ids, change, handle_names=None, digit_frame="world", body=None):
    """Predicted world end positions [..., len(ids), 3] from change rows [..., len(ids), 6] in physical units.
    A relative row needs its parent among ids (every part above the child level holds it)."""
    from lem.model.motor_synergy import quat_mul, rotvec_to_quat
    ids = list(ids)
    out = start.bones[ids, 4:7] + change[..., :3]
    rows, child, parent, kinds = _relative_split(handle_names, ids, digit_frame, body)
    if not rows:
        return out
    if any(w not in ids for w in parent):
        raise ValueError("A wrist-frame digit's world position needs its wrist row (a relative row needs its "
                         "parent row)")
    lead = change.shape[:-2]
    of = {c: (p, k) for c, p, k in zip(child, parent, kinds)}
    pos, turn = {}, {}

    def orientation(i):
        """World end orientation of row i: a world or carried row turns by its own world turn; a rigid row's turn is
        of its pose relative to its parent."""
        if i not in turn:
            k = ids.index(i)
            q0 = quaternion(start.bones[i, :4]).expand(*lead, 4)
            if i not in of or of[i][1] == "carried":
                turn[i] = quat_mul(rotvec_to_quat(change[..., k, 3:6]), q0)
            elif of[i][1] == "rigid":
                _, rel0, _ = relative_pose(start.bones, [i], [of[i][0]])
                turn[i] = quat_mul(orientation(of[i][0]), quat_mul(rotvec_to_quat(change[..., k, 3:6]),
                                                                   rel0[0].expand(*lead, 4)))
            else:
                raise ValueError("An eye row carries no other row")
        return turn[i]

    def position(i):
        if i not in pos:
            k = ids.index(i)
            if i not in of:
                pos[i] = out[..., k, :]
            else:
                p, kind = of[i]
                rel_s, _, _ = relative_pose(start.bones, [i], [p], [kind])
                pos[i] = position(p) + quat_rotate(orientation(p), rel_s[0] + change[..., k, :3])
        return pos[i]
    return torch.stack([position(i) for i in ids], -2)


def move_children(command, handle_names, pairs, source, target):
    """Carry each (child, parent) row pair's command from parent pose `source` to parent pose `target` [H,7]."""
    from lem.model.motor_synergy import quat_mul
    out = command.clone()
    for child, parent in pairs:
        if child not in handle_names or parent not in handle_names:
            continue
        c, p = handle_names.index(child), handle_names.index(parent)
        qs, qt = quaternion(source[..., p, 3:7]), quaternion(target[..., p, 3:7])
        qs_inv = _inverse(qs)
        out[..., c, :3] = quat_rotate(qt, quat_rotate(qs_inv, command[..., c, :3] - source[..., p, :3])) + target[..., p, :3]
        out[..., c, 3:7] = quaternion(quat_mul(qt, quat_mul(qs_inv, quaternion(command[..., c, 3:7]))))
    return out


def executed_rows(executed, start, handle_names, hand_frame="recorded", body=None):
    """The executed command in the frames the hierarchy learns it in: digit rows on the actual wrist
    (hand_frame actual_wrist), calf rows on the actual thigh (BodyRows.reanchor)."""
    out = executed if hand_frame == "recorded" else actual_wrist_frame(executed, start, handle_names)
    if body is not None and body.reanchor:
        actual = torch.cat((start.bones[..., 4:7], start.bones[..., :4]), -1)
        out = move_children(out, handle_names, body.reanchor, out[..., :7].clone(), actual)
    return out


def body_lesson_weights(tree, body):
    """{part: (change [rows*6], action [owned*8], contact [rows])} weights of BodyRows (module comment above).

    future_rows "own": a part's future lesson grades only the rows of the handles it owns; its predictions of its
    descendants' rows are trained by the cross-level consistency lesson against the descendants' own predictions.
    "subtree" grades every row of the subtree, as before."""
    from .tree import SKINLESS
    names = tree.handle_names
    out = {}
    for node in tree.nodes:
        ids, own = tree.subtree[node.name], set(tree.owned[node.name])
        change = torch.ones(len(ids), CHANGE_DIM)
        for k, i in enumerate(ids):
            if names[i] in body.positionless:
                change[k, :3] = 0
            if body.future_rows == "own" and i not in own:
                change[k] = 0
        action = torch.zeros(len(tree.owned[node.name]), ACTION_DIM)
        for k, i in enumerate(tree.owned[node.name]):
            action[k, 3:6] = 1
            if names[i] in body.position_read:
                action[k, :3] = 1
        contact = torch.tensor([float(names[i] not in SKINLESS and (body.contact_recorded is None
                                                                    or names[i] in body.contact_recorded))
                                for i in ids])
        out[node.name] = (change.flatten(), action.flatten(), contact)
    return out


def predicted_rows(tree, predictions, node, body=None):
    """[..., len(subtree), 6] the hierarchy's predicted change of a part's subtree (normalized rows). With
    future_rows "own" each row comes from the part that owns its handle (the part whose future lesson grades it);
    otherwise from the part itself, as before."""
    ids = tree.subtree[node]
    own = predictions[node].change
    if body is None or body.future_rows != "own":
        return own.reshape(*own.shape[:-1], len(ids), CHANGE_DIM)
    rows = []
    for i in ids:
        part = tree.owners[tree.handle_names[i]]
        k = tree.subtree[part].index(i)
        rows.append(predictions[part].change.reshape(*own.shape[:-1], -1, CHANGE_DIM)[..., k, :])
    return torch.stack(rows, -2)


def predicted_world_steps(tree, predictions, start, scales, digit_frame="world", digit_scales=None, body=None):
    """{handle row: [..., 3] predicted one-step WORLD displacement, cm} for every handle of the tree, each predicted by
    the part that owns it (future_rows own; trajectory.predicted_rows) and placed in the world through its parents'
    predicted poses (end_positions): the frame-free reading of where a code moves the body."""
    out = {}
    for root in tree.roots:
        ids = list(tree.subtree[root])
        change = predicted_rows(tree, predictions, root, body) * row_scales(
            tree.handle_names, ids, scales, "change", digit_scales, body)
        end = end_positions(start, ids, change, tree.handle_names, digit_frame, body)
        for k, i in enumerate(ids):
            out[i] = end[..., k, :] - start.bones[i, 4:7]
    return out


def consistency_weights(tree, body):
    """{part: [rows*6]} the change numbers the cross-level consistency lesson compares (the eyes' position is not)."""
    names = tree.handle_names
    out = {}
    for node in tree.nodes:
        w = torch.ones(len(tree.subtree[node.name]), CHANGE_DIM)
        for k, i in enumerate(tree.subtree[node.name]):
            if names[i] in body.positionless:
                w[k, :3] = 0
        out[node.name] = w.flatten()
    return out


def _rows_of(tree):
    """Every handle the tree holds, and each node's rows within that list (one computation per example)."""
    every = sorted({i for ids in tree.subtree.values() for i in ids})
    where = {h: k for k, h in enumerate(every)}
    return every, {n.name: ([where[i] for i in tree.subtree[n.name]], [where[i] for i in tree.owned[n.name]])
                   for n in tree.nodes}


def _anchor(tree, node):
    # One root frame for the entire arm chain preserves parent/child geometry.
    root = node
    while root.parent is not None:
        root = tree.by_name[root.parent]
    return tree.owned[root.name][0]


def state_inputs(tree, history, current, scales, digit_frame="world", digit_scales=None, body=None):
    """Planner inputs never contain future observations or posterior intentions."""
    state, delta = {}, {}
    every, rows = _rows_of(tree)
    cache = {}
    for node in tree.nodes:
        scale = row_scales(tree.handle_names, tree.subtree[node.name], scales, "state", digit_scales, body)
        anchor = _anchor(tree, node)
        if anchor not in cache:
            cache[anchor] = (physical_state(current, every, anchor, digit_frame, tree.handle_names, body),
                             physical_state(history, every, anchor, digit_frame, tree.handle_names, body))
        now, old = (v[..., rows[node.name][0], :] for v in cache[anchor])
        one = now.dim() == 2                      # a single observation, or a stack of them (make_batch)
        state[node.name] = (now / scale).flatten(-2).unsqueeze(0) if one else (now / scale).flatten(-2)
        delta[node.name] = ((now - old) / scale).flatten(-2).unsqueeze(0) if one else ((now - old) / scale).flatten(-2)
    return state, delta


def check_body_tree(tree, body):
    """A tree beyond the arm-and-hand pilot needs the whole-body row convention, and the eyes need it always."""
    from .tree import PILOT_HANDLES
    if body is None and (set(tree.owners) - PILOT_HANDLES):
        raise ValueError("A tree owning %s needs the whole-body row convention (config key body_rows)"
                         % sorted(set(tree.owners) - PILOT_HANDLES))


def make_batch(tree, examples, *, scales, duration_s, action_contract, hand_frame="recorded",
               digit_frame="world", digit_scales=None, body=None, horizon_steps=1):
    validate_scales(scales)
    check_horizon(horizon_steps)
    check_hand_frame(hand_frame, action_contract)
    check_digit_frame(digit_frame, hand_frame, digit_scales)
    check_body_tree(tree, body)
    if body is not None and body.contract != action_contract:
        raise ValueError("The whole-body rows were declared for another action contract")
    if not examples:
        raise ValueError("Empty trajectory batch")
    names = tree.handle_names
    sizes = {n.name: (row_scales(names, tree.subtree[n.name], scales, "change", digit_scales, body),
                      row_scales(names, tree.owned[n.name], scales, "action", digit_scales, body)) for n in tree.nodes}
    every, rows = _rows_of(tree)
    for ex in examples:
        ex.validate(tree, duration_s=duration_s, action_contract=action_contract, horizon_steps=horizon_steps)
    # EVERY EXAMPLE AT ONCE (whole body, Oct 1, 2026): the row functions take any leading batch dimension, so the
    # examples are stacked and each handle's rows computed once for the whole batch (the same numbers as one example
    # at a time; the whole body's 26 handles took 13 ms per example one at a time)
    history, start, end = (_Stack.of([getattr(ex, k) for ex in examples]) for k in ("history", "start", "end"))
    state, recent = state_inputs(tree, history, start, scales, digit_frame, digit_scales, body)
    executed = executed_rows(torch.stack([ex.executed for ex in examples]), start, names, hand_frame, body)
    changes = change_rows(start, end, every, names, digit_frame, body)
    actions = action_rows(start, executed, every, names, digit_frame, body)
    fields = {k: {} for k in ("state", "history", "action", "change", "contact")}
    for n in tree.nodes:
        ids = list(tree.subtree[n.name])
        fields["state"][n.name] = state[n.name]
        fields["history"][n.name] = recent[n.name]
        fields["change"][n.name] = (changes[:, rows[n.name][0]] / sizes[n.name][0]).flatten(1)
        fields["action"][n.name] = (actions[:, rows[n.name][1]] / sizes[n.name][1]).flatten(1)
        fields["contact"][n.name] = end.contact[:, ids]
    batch = LocalBatch(**fields)
    if body is not None:
        weights = body_lesson_weights(tree, body)
        batch.change_weight = {n: w[0] for n, w in weights.items()}
        batch.action_weight = {n: w[1] for n, w in weights.items()}
        batch.contact_weight = {n: w[2] for n, w in weights.items()}
        batch.consistency_weight = consistency_weights(tree, body)
    return batch


@dataclass
class _Stack:
    """Observations stacked along a leading example dimension (the row functions read .bones and .contact only)."""
    bones: torch.Tensor     # [N, H, 11]
    contact: torch.Tensor   # [N, H]

    @classmethod
    def of(cls, observations):
        return cls(torch.stack([o.bones for o in observations]), torch.stack([o.contact for o in observations]))


# ============================================================================ THE HOLD HORIZON (Oct 1, 2026)
# Orders are held 5 model steps (0.25 s; every complete run in the seg41 recording is 5 rows: 4,352 per body). A
# one-step hierarchy predicts 50 ms and the planner comparison stretched that step over the hold by a measured factor
# (about 4.2). With horizon_steps H (config key; absent = 1, every earlier meaning) the hierarchy learns and predicts
# the change over the whole hold directly: each example spans the H consecutive recorded rows of one body that executed
# one order (hold_example; HoldRuns builds them from a recording). The history stays one model step before the start,
# so the planner inputs are the same as before. The live shadow builds one-step examples and refuses H != 1.
def check_horizon(horizon_steps):
    if type(horizon_steps) is not int or horizon_steps < 1:
        raise ValueError("horizon_steps must be a positive integer number of model steps")


def _order_rows(handle_names):
    """The rows that must be identical through one held order: every row but the eyes' (the recorded eye row is each
    step's mean eye command, saccades and the reflex included, so it changes within a hold)."""
    return [i for i, n in enumerate(handle_names) if n != "gaze"]


def hold_example(rows, handle_names):
    """One Example spanning consecutive one-step Examples `rows` of one body that executed one order: the first row's
    history and start, the last row's end, the common order. Its eye row is the mean over the steps (position, radius
    and strength averaged; orientation the normalized mean of the steps' quaternions, signs aligned to the first): the
    mean eye command over the hold, as each step's row is the mean over that step."""
    if len(rows) < 2:
        raise ValueError("A hold example spans at least two model steps")
    keep = _order_rows(tuple(handle_names))
    first = rows[0]
    for a, b in zip(rows, rows[1:]):
        if (a.start.env_id != b.start.env_id or a.start.episode_id != b.start.episode_id
                or b.start.frame != a.start.frame + 1 or a.end.frame != b.start.frame
                or not torch.equal(a.end.bones, b.start.bones) or a.action_contract != b.action_contract):
            raise ValueError("A hold example needs consecutive rows of one body and episode")
        if not torch.equal(a.executed[keep], b.executed[keep]):
            raise ValueError("A hold example needs one executed order (every row but the eyes' identical)")
    executed = first.executed.clone()
    if "gaze" in handle_names:
        g = list(handle_names).index("gaze")
        stack = torch.stack([r.executed[g] for r in rows])
        q = quaternion(stack[:, 3:7])
        q = torch.where(((q * q[:1]).sum(-1, keepdim=True) < 0), -q, q)
        executed[g] = stack.mean(0)
        executed[g, 3:7] = quaternion(q.mean(0))
    return Example(first.history, first.start, rows[-1].end, executed, first.action_contract)


class HoldRuns:
    """Hold-length examples from a recording (bodies interleaved, each body in time order): every run of exactly
    `steps` consecutive rows of one body with one order (hold_example) gives one example. A run cut short by a frame gap,
    an episode change, a start that is not the previous row's end, or a new order gives none; a run reaching `steps` rows
    is emitted and the next row starts a new run. `lengths` counts every closed run's length per body (the measurement of
    what the recording yields), `breaks` the rows that broke a run without a new order."""
    def __init__(self, handle_names, steps):
        check_horizon(steps)
        if steps < 2:
            raise ValueError("A hold horizon is at least 2 model steps")
        self.names, self.steps = tuple(handle_names), steps
        self.keep = _order_rows(self.names)
        self.runs, self.lengths, self.emitted = {}, {}, {}
        self.breaks = {}       # body -> rows whose start is not the previous row's end (frame or bones): a run break

    def _close(self, body):
        run = self.runs.pop(body, None)
        if run:
            counts = self.lengths.setdefault(body, {})
            counts[len(run)] = counts.get(len(run), 0) + 1

    def add(self, ex):
        """Feed one one-step Example; returns a hold Example when this row completes a run, else None."""
        body = ex.start.env_id
        run = self.runs.get(body)
        if run:
            last = run[-1]
            joined = (ex.start.episode_id == last.start.episode_id and ex.start.frame == last.start.frame + 1
                      and ex.start.frame == last.end.frame and torch.equal(ex.start.bones, last.end.bones))
            if not joined:
                self.breaks[body] = self.breaks.get(body, 0) + 1
            if not joined or not torch.equal(ex.executed[self.keep], last.executed[self.keep]):
                self._close(body)
                run = None
        run = (run or []) + [ex]
        self.runs[body] = run
        if len(run) == self.steps:
            self._close(body)
            self.emitted[body] = self.emitted.get(body, 0) + 1
            return hold_example(run, self.names)
        return None

    def finish(self):
        """Close the open runs (cut by the end of the recording); returns {body: {run length: count}}."""
        for body in list(self.runs):
            self._close(body)
        return {b: dict(sorted(v.items())) for b, v in sorted(self.lengths.items())}


class TrajectoryExtractor:
    """Two prior observations per body, serialized independently of production replay."""
    def __init__(self, tree, *, duration_s, action_contract):
        if not math.isfinite(duration_s) or duration_s <= 0 or not action_contract:
            raise ValueError("A measured duration and explicit action contract are required")
        self.tree, self.duration_s, self.action_contract = tree, duration_s, action_contract
        self.previous = {}
        self.dropped = 0

    def observe(self, obs, executed=None, *, action_contract=None, frame_t=None):
        obs.validate(len(self.tree.handle_names))
        prior = self.previous.get(obs.env_id, [])
        if prior and (obs.episode_id != prior[-1].episode_id or obs.frame != prior[-1].frame + 1
                or not math.isclose(obs.sim_time - prior[-1].sim_time, self.duration_s,
                                    rel_tol=1e-6, abs_tol=1e-9)):
            prior = []
            self.dropped += 1
        result = None
        if executed is not None:
            if action_contract != self.action_contract:
                raise ValueError("Action contract changed; begin a separately declared dataset")
            if prior and frame_t != prior[-1].frame:
                raise ValueError("Executed action is not aligned with its starting observation")
            if len(prior) == 2:
                result = Example(prior[0], prior[1], obs, tensor(executed), action_contract)
                result.validate(self.tree, duration_s=self.duration_s, action_contract=self.action_contract)
        # A missing command discards that transition, but physical history remains useful.
        self.previous[obs.env_id] = (prior + [obs])[-2:]
        return result

    def state_dict(self):
        return {"previous": {k: [o.state_dict() for o in v] for k, v in self.previous.items()},
                "dropped": self.dropped}

    def load_state_dict(self, state):
        previous = {int(k): [Observation.from_dict(o) for o in v] for k, v in state["previous"].items()}
        for key, observations in previous.items():
            if not 1 <= len(observations) <= 2:
                raise ValueError("Invalid saved history length")
            for o in observations:
                o.validate(len(self.tree.handle_names))
                if o.env_id != key:
                    raise ValueError("Saved history belongs to another body")
            if len(observations) == 2:
                a, b = observations
                if (a.episode_id != b.episode_id or b.frame != a.frame + 1
                        or not math.isclose(b.sim_time - a.sim_time, self.duration_s,
                                            rel_tol=1e-6, abs_tol=1e-9)):
                    raise ValueError("Saved history crosses an observation boundary")
        if type(state["dropped"]) is not int or state["dropped"] < 0:
            raise ValueError("Invalid saved boundary count")
        self.previous, self.dropped = previous, int(state["dropped"])
