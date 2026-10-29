# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""TransitionV1 — one completed transition, as a type.

The plan drew this dataclass and then the implementation plumbed its fields as loose call
arguments and replay columns. That worked, but it left the contract as a convention rather
than a thing: nothing could be handed a transition, nothing could validate one, and the
invariant that matters — that the action is the one which carried state_t to state_t1 —
lived only in comments.

The invariant, stated once, here:

    state_t + action_t  ->  state_t1

`action_t` is the OUTGOING action: the order selected from state_t and actually executed
across the interval. It is not the order that produced state_t (that one has already
finished acting), and it is not the raw controller proposal (developmental limits,
exploration noise and the motor hold all change it before the body sees it).

A transition cannot be built when state_t is observed, because the action does not exist
yet. It is completed one step later, which is what `Pending.complete()` expresses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# what an executed order's numbers mean: world targets (world_v0), or '+'-joined parts in this order:
# finger rows as a feasible joint goal on the recorded wrist (wrist_relative_joint_v1,
# hand_target_control.py); calf rows as a knee goal on the recorded thigh (parent_relative_calf_v1)
# and the chest row as its two joints' goal on the upper body the recorded pelvis implies
# (parent_relative_chest_v1; both limb_target_control.py); the head's and the pelvis's position
# numbers read by nothing (no_head_pelvis_position_v1, action_decoder.position_handles); the finger goal
# servoed hinge by hinge (finger_joint_servo_v1, action_decoder.FINGER_SERVO_CONTRACT, only with the
# wrist-relative goal). The arm reach projection keeps world targets and adds no part.
ACTION_CONTRACT_PARTS = ('wrist_relative_joint_v1', 'parent_relative_calf_v1', 'parent_relative_chest_v1',
                         'no_head_pelvis_position_v1', 'finger_joint_servo_v1')


def known_action_contract(contract) -> bool:
    if contract == 'world_v0':
        return True
    parts = str(contract).split('+')
    order = [ACTION_CONTRACT_PARTS.index(p) for p in parts if p in ACTION_CONTRACT_PARTS]
    return (len(order) == len(parts) and order == sorted(set(order))
            and ('parent_relative_chest_v1' not in parts or 'parent_relative_calf_v1' in parts)
            and ('finger_joint_servo_v1' not in parts or 'wrist_relative_joint_v1' in parts))


@dataclass(frozen=True)
class TransitionV1:
    """One completed transition. Frozen: a transition is a record, not a workspace."""

    env_id: int
    episode_id: int
    frame_t: int
    frame_t1: int
    duration_s: float

    state_t: Dict[str, Any]
    state_t1: Dict[str, Any]
    achieved_t: Any                       # [n_handles, 10] the pose at t
    context_t: Any                        # the temporal context as of t, before t was added

    proposed_action: Any                  # raw controller output
    executed_action: Any                  # after limits, noise and the hold: what the body got
    decoded_ctrl: Optional[Any] = None    # joint-space control MuJoCo received
    gaze_substeps: Optional[Any] = None   # [n_substeps, 2] every eye command in the interval
    order_id: Optional[int] = None        # serial of the latched order (trainer-stamped)
    action_contract: str = 'world_v0'
    # For wrist-relative hands, executed_action is the issued canonical goal;
    # these are its actual expanded world commands on every physics substep.
    motor_substeps: Optional[Any] = None
    hand_joint_goal: Optional[Any] = None

    source_mask_t: Dict[str, str] = field(default_factory=dict)
    source_mask_t1: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Validation belongs HERE, not only in Pending.complete(). A reviewer built a
        # TransitionV1 directly with frame_t1 < frame_t, a negative duration and
        # executed_action=None, and construction succeeded - so calling the type
        # "self-validating" was false. Every construction path is checked now.
        self.validate()

    def gaze_summary(self) -> Dict[str, Any]:
        """Start, end, mean and integrated angular displacement of the eyes.

        Gaze is not held: the saccade generator and the peripheral-motion reflex can move
        the eyes on any substep, and the image is rendered on the last one. A single mean
        is a summary that may correspond to no command ever issued, so the sequence is
        kept and the summary derived from it rather than replacing it.
        """
        g = self.gaze_substeps
        if g is None or len(g) == 0:
            return {"start": None, "end": None, "mean": None, "path_deg": 0.0, "n": 0}
        g = np.asarray(g, dtype=np.float64)
        steps = np.linalg.norm(np.diff(g, axis=0), axis=-1) if len(g) > 1 else np.zeros(1)
        return {"start": g[0].tolist(), "end": g[-1].tolist(),
                "mean": g.mean(axis=0).tolist(),
                "path_deg": float(np.degrees(steps.sum())), "n": int(len(g))}

    def validate(self) -> None:
        """Raise if this record cannot be what it claims to be.

        Checks the states, the context, the gaze path and the controls too - the first
        version checked only frame order, duration and the action array, which a reviewer
        pointed out leaves most of the record unvalidated.
        """
        if self.frame_t1 <= self.frame_t:
            raise ValueError(f"frame_t1 {self.frame_t1} does not follow frame_t {self.frame_t}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.executed_action is None:
            raise ValueError("executed_action is None: this transition has no cause")
        ex = np.asarray(self.executed_action)
        if ex.ndim < 2 or ex.shape[-1] != 9:
            raise ValueError(f"executed_action must end in 9 features, got {ex.shape}")
        if not np.all(np.isfinite(ex)):
            raise ValueError("executed_action contains non-finite values")
        if not known_action_contract(self.action_contract):
            raise ValueError('Unknown motor action contract')
        parts = self.action_contract.split('+')
        if 'wrist_relative_joint_v1' in parts:
            if self.motor_substeps is None or self.hand_joint_goal is None:
                raise ValueError('Wrist-relative action needs its joint goal and physical substeps')
        if any(p.startswith('parent_relative_') for p in parts) and self.motor_substeps is None:
            # the calf and chest rows are joint goals on their parents' ordered poses; the rows the
            # physics received, regenerated around the parents every frame, are the executed ones
            raise ValueError('Parent-relative action needs its physical substeps')
        if self.motor_substeps is not None:
            substeps = np.asarray(self.motor_substeps)
            if substeps.ndim != 3 or substeps.shape[1:] != ex.shape or not len(substeps):
                raise ValueError('Motor substeps must contain every handle command')
            if not np.isfinite(substeps).all():
                raise ValueError('Non-finite motor substeps')
        if self.hand_joint_goal is not None:
            joints = np.asarray(self.hand_joint_goal)
            if joints.ndim != 1 or not len(joints) or not np.isfinite(joints).all():
                raise ValueError('Invalid hand joint goal')
        if self.state_t is None or self.state_t1 is None:
            raise ValueError("a transition needs both of its states")
        if self.achieved_t is not None:
            a = np.asarray(self.achieved_t)
            if a.shape[-1] < 10:
                raise ValueError(f"achieved_t must carry 10 features, got {a.shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError("achieved_t contains non-finite values")
        if self.gaze_substeps is not None:
            g = np.asarray(self.gaze_substeps)
            if g.shape[-1] != 2:
                raise ValueError(f"gaze_substeps must end in (az, el), got {g.shape}")
            if not np.all(np.isfinite(g)):
                raise ValueError("gaze_substeps contains non-finite values")
        if self.decoded_ctrl is not None:
            c = np.asarray(self.decoded_ctrl)
            if not np.all(np.isfinite(c)):
                raise ValueError("decoded_ctrl contains non-finite values")
        for name, m in (("source_mask_t", self.source_mask_t),
                        ("source_mask_t1", self.source_mask_t1)):
            for k, v in (m or {}).items():
                sv = str(getattr(v, "value", v))
                if sv not in ("real", "synthetic", "stale", "disabled"):
                    raise ValueError(f"{name}[{k}] is {sv!r}, not a known source")


@dataclass
class Pending:
    """A transition waiting for its cause.

    state_t is known the moment it is observed; the action that carries it forward is not
    chosen until afterwards, and is not KNOWN to have been executed until the body has run
    it. Holding the half-built record makes that ordering explicit instead of leaving it
    implicit in the order of statements in a training loop.
    """

    env_id: int
    episode_id: int
    frame_t: int
    state_t: Dict[str, Any]
    achieved_t: Any
    context_t: Any
    source_mask_t: Dict[str, str] = field(default_factory=dict)
    proposed_action: Optional[Any] = None

    def complete(self, *, executed_action, state_t1, frame_t1, duration_s,
                 source_mask_t1=None, decoded_ctrl=None,
                 gaze_substeps=None, episode_id=None, order_id=None,
                 action_contract='world_v0', motor_substeps=None, hand_joint_goal=None) -> TransitionV1:
        """Close the record. Refuses to close one that spans an episode boundary."""
        if episode_id is not None and episode_id != self.episode_id:
            raise ValueError(
                f"transition would span an episode boundary ({self.episode_id} -> "
                f"{episode_id}): an action cannot carry a body across a teleport")
        tr = TransitionV1(
            env_id=self.env_id, episode_id=self.episode_id,
            frame_t=self.frame_t, frame_t1=frame_t1, duration_s=duration_s,
            state_t=self.state_t, state_t1=state_t1,
            achieved_t=self.achieved_t, context_t=self.context_t,
            proposed_action=self.proposed_action, executed_action=executed_action,
            decoded_ctrl=decoded_ctrl, gaze_substeps=gaze_substeps,
            order_id=order_id,
            action_contract=action_contract, motor_substeps=motor_substeps, hand_joint_goal=hand_joint_goal,
            source_mask_t=dict(self.source_mask_t),
            source_mask_t1=dict(source_mask_t1 or {}))
        tr.validate()
        return tr
