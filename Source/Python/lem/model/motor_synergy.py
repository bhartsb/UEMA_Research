# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Motor synergy: a few numbers of intention become a coordinated whole-body posture change.

WHY THIS EXISTS (docs/gain_of_function_v3/DESIGN.md, decision D11; the addendum's section 5).
The body follows a whole-body order that is a posture it really reached at 0.625 (-1 against,
0 unrelated, +1 exact; physics alone, tools/whole_body_following.py) and one made of 25
independent turns at 0.365. Babble is 25 independent turns, and the planner moves 234 numbers
per body with nothing tying one handle's turn to another's. Live following read 0.14 on segment
31. This module gives both a compact intention z instead: a small nonlinear network turns z into
a change of every body handle's target, relative to where the body is, drawn from the space of
changes the body has really made.

  S(z, pose)   z [B, k] and the current achieved pose -> per handle the target change: a turn
               (rotation vector, radians, world frame) for each of the 25 body handles and a
               move (centimetres, world frame) for each of the 14 handles whose position the
               decoder reads. Never the eyes: they have their own servo.
  S'(z, pose)  = S(z, pose) - S(0, pose), taken before the bound, so z = 0 asks for no change
               (every handle held where it is). The last layer starts small but not zero, so
               the planner has a gradient through S' from its first step.
  bound        each handle's turn is limited to the angle its own joints can produce and each
               move to the distance its own joints can carry it (REACH below, measured from the
               body's joint ranges), softly: a change well inside the reach passes nearly
               unchanged, one near the reach flattens against it.
  E(change, pose)  the hindsight encoder: from an order's begin pose and the change the body
               ACHIEVED over the order's tenure, the z that S' should turn into that change.

THE LESSON (hindsight, not a policy). Every order that ran its full tenure is a sample: its
begin pose, the change achieved, how well the body followed the order and how much it moved.
E encodes the achieved change, S' must reproduce it through the k-number bottleneck, and a small
L2 on z keeps z near unit scale so noise drawn in z has a known size. Each sample is weighted by
following x movement (the order's direction match, below zero counted as zero, times the size of
the change in measured spreads), so orders the body did not follow and orders that barely moved
the body - trivially reconstructed by z = 0 - do not set what S learns. Nothing here is a reward
and nothing chooses behaviour: the planner chooses z by the forward model's predictions, babble
draws z at random.

LIVENESS (DESIGN section 10). The pose input is the achieved pose, which carries a large part
shared by every moment (the body lies in the same crib, mostly on its back): it is centred by a
running mean over lived moments (kept in runtime_state.pt) and then LayerNormed (rule 1). No
pooling: every handle reaches the output through the flat projection (rule 3). The window
reports the live share of every hidden layer (rule 2).

Every constant carries its measurement or its arithmetic; an unmeasured one is None and the
module refuses to build.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SYNERGY_VERSION = "motor_synergy_v1"

# THE DEFAULT SIZE OF z (--synergy-dim): the number of principal components carrying 90% of the
# achieved 250 ms posture changes, bounded to 8-32. MEASURED 2026-09-22 (b5 scratch b5_replay.py)
# on the live policy's own record: the replay diary saved with ckpt_step_344400, 8 bodies x 6,000
# moments of H84 segment 31, 9,592 orders that ran their whole 5-moment tenure; each change is the
# 25 body handles' turns and the 14 position handles' moves in the change scales below (117
# numbers). Uncentred second moment (S'(0) = 0, so z = 0 reconstructs nothing): 90% needs 35
# components (32 carry 0.890; 8 carry 0.600, 16 carry 0.776), bounded to 32; weighted as the lesson
# weights (following x movement) 90% needs 31 (32 carry 0.906). The CPU run on the checkpoint's
# world is in b5-synergy.md beside it. Trained on the same record (b5_fit.py capacity, 3,000 lesson
# batches on the first 80% in time, graded on the last 20%), the share of the change reproduced
# through z: k 8 0.70, k 16 0.79, k 32 0.83.
SYNERGY_DIM: Optional[int] = 32
SYNERGY_DIM_MEASUREMENT: dict = {
    "date": "2026-09-22", "source": "b5 scratch b5_replay.py / b5_fit.py capacity on the replay "
    "diary of ckpt_step_344400 (8 bodies, 9,592 full-tenure orders)",
    "k90_uncentred": 35, "share_at_32": 0.890, "k90_weighted": 31, "share_at_32_weighted": 0.906,
    "held_out_share": {"8": 0.70, "16": 0.79, "32": 0.83}}

# THE CHANGE SCALES: the root mean square, per axis, of the achieved 250 ms change of a body handle
# (degrees for turns over the 25 body handles, centimetres for moves over the 14 position handles),
# over the same 9,592 orders. The network's outputs and its loss are in these units, so one unit
# is one typical 250 ms change. (b1's efference tenure scales, 1 body on the CPU: 11.0 degrees and
# 1.80 cm over all 25 handles' positions, most of which move less than the hands.)
CHANGE_SCALE_DEG: Optional[float] = 11.03
CHANGE_SCALE_CM: Optional[float] = 2.825
CHANGE_SCALE_MEASUREMENT: dict = {
    "date": "2026-09-22", "source": "b5 scratch b5_replay.py on the replay diary of ckpt_step_344400",
    "orders": 9592, "turn_deg_rms_per_axis": 11.03, "move_cm_rms_per_axis": 2.825}

# EACH HANDLE'S REACH: the largest turn (degrees, capped at 180) and the largest move (centimetres,
# per axis, the decoder's own clip) its own joints can produce, from the body's joint ranges. The
# position reach is ActionDecoder._measure_reach's; the turn reach is the widest angle between any
# two orientations of the handle's body over every corner of its joints' ranges. MEASURED
# 2026-09-22 (b5 scratch b5_reach.py) on the crib body at 24 months, trunk_v1, the segment 31 world.
ROT_REACH_DEG: Optional[Dict[str, float]] = {
    "wrist_L": 180.0, "thumb_L": 180.0, "index_L": 170.94, "middle_L": 170.94, "ring_L": 170.94,
    "pinky_L": 179.93, "wrist_R": 180.0, "thumb_R": 180.0, "index_R": 170.94, "middle_R": 170.94,
    "ring_R": 170.94, "pinky_R": 179.93, "head": 173.88, "upperarm_L": 170.38, "lowerarm_L": 151.0,
    "upperarm_R": 170.38, "lowerarm_R": 151.0, "thigh_L": 178.45, "calf_L": 149.0, "thigh_R": 178.45,
    "calf_R": 149.0, "foot_L": 127.46, "foot_R": 127.46, "lower_trunk": 64.73, "chest": 40.12}
POS_REACH_CM: Optional[Dict[str, float]] = {
    "wrist_L": 42.85, "thumb_L": 5.54, "index_L": 5.48, "middle_L": 6.09, "ring_L": 5.48,
    "pinky_L": 6.8, "wrist_R": 42.85, "thumb_R": 5.54, "index_R": 5.48, "middle_R": 6.09,
    "ring_R": 5.48, "pinky_R": 6.8, "head": 4.81, "lower_trunk": 5.58}
REACH_MEASUREMENT: dict = {
    "date": "2026-09-22", "source": "b5 scratch b5_reach.py: BridgeEnv crib, age 24, trunk_v1; "
    "ActionDecoder._measure_reach for moves, the same corner walk on orientation for turns"}

# THE L2 ON z, as a weight on mean(z^2) beside the lesson's mean squared error in change scales. Its
# job is to keep z near unit scale (so noise drawn in z has a known size). MEASURED 2026-09-22 (b5
# scratch b5_fit.py capacity, k 32, 3,000 lesson batches on the replay diary's orders): the root
# mean square of the encoded z reads 1.19 at 0.001, 0.96 at 0.002, 0.83 at 0.003, 0.68 at 0.005,
# 0.53 at 0.01, 0.20 at 0.1, while the held-out explained share stays 0.82-0.83 at every one of them;
# 0.002 is the weight whose z sits nearest 1.
Z_L2: Optional[float] = 0.002
Z_L2_MEASUREMENT: dict = {
    "date": "2026-09-22", "source": "b5 scratch b5_fit.py capacity (k 32, 3,000 batches)",
    "z_rms": {"0.001": 1.19, "0.002": 0.96, "0.003": 0.83, "0.005": 0.68, "0.01": 0.53, "0.1": 0.20},
    "held_out_share": {"0.001": 0.824, "0.002": 0.826, "0.01": 0.833, "0.1": 0.828}}

# THE LAST LAYER'S STARTING SIZE: its weights are drawn at this multiple of the default
# initialisation, so a fresh S' asks for small changes and the planner still has a gradient.
# Measured on the module as built (b5 scratch b5_fit.py): a fresh S' at z ~ N(0, I) asks 0.00012
# rad of turn and 0.0018 cm of move per handle, against 11 degrees (0.19 rad) and 2.8 cm for one
# typical 250 ms change, so a fresh synergy holds the body where it is. Adam's first steps are
# lr-sized whatever the start, so the lesson grows it: emulated on the replay diary's orders (b5_fit
# online, 8 new samples + 56 from the ring per lesson), the share of a change reproduced BEFORE it
# is learned from reached 0.38 after 1,000 samples (125 lessons, about 10 minutes of the live run's
# 8 bodies) and 0.71-0.76 after 5,000-9,000.
LAST_LAYER_INIT = 0.01

# the hindsight samples kept for the lesson, and how many are graded per order ending. Arithmetic:
# 8 bodies finish 8 orders every 5 moments, so a sample stays in a 1,024 ring for 128 order endings
# (640 moments, about 9 minutes live) and each lesson of 64 draws 56 of up to 1,016 older ones: every
# sample is graded about 7 times after its first. Measured cost: 3,000 lessons of 64 took 27-33 s on
# 4 CPU threads (about 10 ms per lesson, one lesson every 5 moments).
BANK_SIZE = 1024
LESSON_BATCH = 64

# the running mean that centres the pose input: an exponential average over lived moments with
# this horizon, divided by the weight it has accumulated (contact_head.CENTRE_HORIZON_STEPS, the
# same arithmetic: the shared part is found within a few moments; 2,000 moments follows the slow
# drift of the body's usual posture)
CENTRE_HORIZON_STEPS = 2000

# DESIGN section 10 rule 2: a unit is live when its input rose above this line
LIVE_LINE = -3.0

# the mean length of a 3-D standard normal vector, 2 sqrt(2 / pi): the babble's quaternion wobble
# draws each axis at 10 x exploration_noise radians, so its mean turn is 10 x ns x this
CHI3_MEAN = 2.0 * math.sqrt(2.0 / math.pi)


def unmeasured() -> List[str]:
    """The constants the module cannot build without, by name."""
    out = []
    for name in ("SYNERGY_DIM", "CHANGE_SCALE_DEG", "CHANGE_SCALE_CM", "ROT_REACH_DEG",
                 "POS_REACH_CM", "Z_L2"):
        if globals()[name] is None:
            out.append(name)
    return out


# ------------------------------------------------------------------------ rotation helpers
def rotvec_to_quat(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] rotation vector (radians) -> [..., 4] unit quaternion x y z w. Smooth at zero."""
    ang = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-12)
    half = 0.5 * ang
    k = torch.where(ang > 1e-4, torch.sin(half) / ang, 0.5 - ang * ang / 48.0)
    return torch.cat([v * k, torch.cos(half)], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a * b for x y z w quaternions: the turn b, then the turn a (both world frame)."""
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dim=-1)


def turn_between_rad(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """The world-frame turn carrying q0 to q1, rotation vector in radians (efference's form)."""
    from lem.model.efference import rotation_change_deg
    return rotation_change_deg(q0, q1) * (math.pi / 180.0)


# ------------------------------------------------------------------------ the module
class MotorSynergy(nn.Module):
    """S, S' and the hindsight encoder E.

    handle_names: the body layout's names, in order (trunk_v1: 26 with the eyes at gaze_idx).
    position_names: the handles whose position the decoder reads (action_decoder.POSITION_HANDLES).
    k: the size of z. Outputs are packed [B, n_out] = 25 turns x 3, then 14 moves x 3, in
    change-scale units inside the network and in radians and centimetres outside it."""

    LIVE_LAYERS = ("S1", "S2", "E1", "E2")

    def __init__(self, handle_names: Sequence[str], gaze_idx: int, position_names: Sequence[str],
                 k: int, hidden: int = 256, pose_embed: int = 32):
        super().__init__()
        missing = unmeasured()
        if missing:
            raise ValueError("lem/model/motor_synergy.py holds unmeasured constants %s; the synergy "
                             "cannot be built on a guess" % ", ".join(missing))
        self.handle_names = list(handle_names)
        self.gaze_idx = int(gaze_idx)
        self.k = int(k)
        H = len(self.handle_names)
        self.rot_idx = [i for i in range(H) if i != self.gaze_idx]
        pos_set = set(position_names)
        self.pos_idx = [i for i in range(H) if i != self.gaze_idx and self.handle_names[i] in pos_set]
        self.n_rot, self.n_pos = len(self.rot_idx), len(self.pos_idx)
        self.n_out = 3 * (self.n_rot + self.n_pos)
        rr = [float(ROT_REACH_DEG[self.handle_names[i]]) * math.pi / 180.0 for i in self.rot_idx]
        pr = [float(POS_REACH_CM[self.handle_names[i]]) for i in self.pos_idx]
        # the reach in the network's own units (change scales), per output channel
        self.register_buffer("rot_reach_rad", torch.tensor(rr, dtype=torch.float32), persistent=False)
        self.register_buffer("pos_reach_cm", torch.tensor(pr, dtype=torch.float32), persistent=False)
        self.rot_scale_rad = float(CHANGE_SCALE_DEG) * math.pi / 180.0
        self.pos_scale_cm = float(CHANGE_SCALE_CM)
        self.pose_f = 7 * self.n_rot
        # the pose input's running mean over lived moments (runtime_state.pt), and its weight
        self.register_buffer("centre_mean", torch.zeros(self.pose_f), persistent=False)
        self.centre_weight = 0.0
        self.pose_norm = nn.LayerNorm(self.pose_f)
        self.pose_proj = nn.Linear(self.pose_f, pose_embed)
        self.S1 = nn.Linear(self.k + pose_embed, hidden)
        self.S2 = nn.Linear(hidden, hidden)
        self.S3 = nn.Linear(hidden, self.n_out)
        self.E1 = nn.Linear(self.n_out + pose_embed, hidden)
        self.E2 = nn.Linear(hidden, hidden)
        self.E3 = nn.Linear(hidden, self.k)
        with torch.no_grad():
            self.S3.weight.mul_(float(LAST_LAYER_INIT))
            self.S3.bias.zero_()          # cancels in S' anyway

    # -------------------------------------------------------------- pose
    def pose_features(self, achieved: torch.Tensor) -> torch.Tensor:
        """[B, H, >=7] achieved handle state (position / 200 cm, world quaternion x y z w) ->
        [B, 7 x 25]: each body handle's position relative to the body handles' mean (cm / 10)
        and its quaternion with w >= 0 (q and -q are one orientation)."""
        a = achieved[:, self.rot_idx].float()
        pos = a[..., 0:3] * 200.0
        pos = (pos - pos.mean(dim=1, keepdim=True)) / 10.0
        q = a[..., 3:7]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q = torch.where(q[..., 3:4] < 0, -q, q)
        return torch.cat([pos, q], dim=-1).reshape(a.shape[0], -1)

    @torch.no_grad()
    def update_centre(self, achieved: torch.Tensor) -> None:
        """One lived moment of the pose input's running mean (all bodies pooled)."""
        a = 1.0 / float(CENTRE_HORIZON_STEPS)
        x = self.pose_features(achieved).mean(dim=0).to(self.centre_mean.dtype)
        w_old = float(self.centre_weight)
        w_new = (1.0 - a) * w_old + a
        self.centre_mean.mul_((1.0 - a) * w_old / w_new).add_(x * (a / w_new))
        self.centre_weight = w_new

    def pose_embedding(self, achieved: torch.Tensor) -> torch.Tensor:
        return self.pose_proj(self.pose_norm(self.pose_features(achieved) - self.centre_mean))

    # -------------------------------------------------------------- S and S'
    def _s_raw(self, z: torch.Tensor, pe: torch.Tensor, pre: Optional[dict] = None) -> torch.Tensor:
        h1p = self.S1(torch.cat([z.float(), pe], dim=-1))
        h1 = F.gelu(h1p)
        h2p = self.S2(h1)
        out = self.S3(F.gelu(h2p))
        if pre is not None:
            pre["S1"], pre["S2"] = h1p, h2p
        return out

    def units(self, z: torch.Tensor, achieved: torch.Tensor,
              residual: Optional[torch.Tensor] = None, pre: Optional[dict] = None) -> torch.Tensor:
        """S'(z) - before the bound, in change-scale units [B, n_out]; S(0) subtracted."""
        pe = self.pose_embedding(achieved)
        u = self._s_raw(z, pe, pre) - self._s_raw(torch.zeros_like(z), pe)
        if residual is not None:
            u = u + residual
        return u

    def bound(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Change-scale units -> (turns [B, 25, 3] radians, moves [B, 14, 3] cm), each softly
        limited to its handle's reach: a turn's LENGTH flattens against the angle the handle's
        joints can produce, a move's every axis against the distance they can carry it (the
        decoder clips each axis the same way)."""
        B = u.shape[0]
        ur = u[:, :3 * self.n_rot].reshape(B, self.n_rot, 3) * self.rot_scale_rad
        up = u[:, 3 * self.n_rot:].reshape(B, self.n_pos, 3) * self.pos_scale_cm
        R = self.rot_reach_rad.view(1, -1, 1)
        n = torch.sqrt((ur * ur).sum(-1, keepdim=True) + 1e-12)
        rot = R * torch.tanh(n / R) * ur / n
        P = self.pos_reach_cm.view(1, -1, 1)
        pos = P * torch.tanh(up / P)
        return rot, pos

    def delta(self, z: torch.Tensor, achieved: torch.Tensor,
              residual: Optional[torch.Tensor] = None, pre: Optional[dict] = None):
        """S'(z, pose) in lived units: (turns [B, 25, 3] rad, moves [B, 14, 3] cm)."""
        return self.bound(self.units(z, achieved, residual, pre))

    def to_units(self, rot_rad: torch.Tensor, pos_cm: torch.Tensor) -> torch.Tensor:
        """Lived changes -> the packed change-scale vector [B, n_out] (no bound applied)."""
        B = rot_rad.shape[0]
        return torch.cat([(rot_rad / self.rot_scale_rad).reshape(B, -1),
                          (pos_cm / self.pos_scale_cm).reshape(B, -1)], dim=-1)

    def from_bounded(self, rot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.to_units(rot, pos)

    # -------------------------------------------------------------- E
    def encode(self, change_units: torch.Tensor, achieved: torch.Tensor,
               pre: Optional[dict] = None) -> torch.Tensor:
        """The hindsight encoder: [B, n_out] achieved change (change-scale units) + begin pose ->
        z_hat [B, k]."""
        pe = self.pose_embedding(achieved)
        h1p = self.E1(torch.cat([change_units.float(), pe], dim=-1))
        h2p = self.E2(F.gelu(h1p))
        if pre is not None:
            pre["E1"], pre["E2"] = h1p, h2p
        return self.E3(F.gelu(h2p))

    # -------------------------------------------------------------- orders
    def order_from_delta(self, rot: torch.Tensor, pos: torch.Tensor, achieved: torch.Tensor,
                         template: torch.Tensor, strength: Optional[torch.Tensor] = None,
                         gaze_quat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """A raw-form order [B, H, 9] (position / 200 cm absolute world, quaternion, radius,
        strength - the motor controller's output form) asking every body handle for its current
        pose changed by (rot, pos): target orientation = the turn applied to the current one,
        target position = current + move for the position handles, current for the rest (the
        decoder never reads those). The eyes' row, radius and, unless given, strength come from
        `template`; gaze_quat replaces the eyes' orientation."""
        a = achieved.float()
        out = template.float().clone()
        q_cur = a[:, self.rot_idx, 3:7]
        q_cur = q_cur / q_cur.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out[:, self.rot_idx, 3:7] = quat_mul(rotvec_to_quat(rot), q_cur)
        out[:, self.rot_idx, 0:3] = a[:, self.rot_idx, 0:3]
        out[:, self.pos_idx, 0:3] = a[:, self.pos_idx, 0:3] + pos / 200.0
        if strength is not None:
            out[..., 8:9] = strength
        if gaze_quat is not None:
            out[:, self.gaze_idx, 3:7] = gaze_quat
        return out

    def perturb_order(self, order_cm: torch.Tensor, rot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """An execution-form order (cm) with every body handle's target turned by `rot` and the
        position handles' targets moved by `pos` (babble's form: the change is applied to the
        order, not to the pose). The eyes' row is untouched."""
        out = order_cm.clone()
        q = out[:, self.rot_idx, 3:7]
        out[:, self.rot_idx, 3:7] = quat_mul(rotvec_to_quat(rot.to(out.dtype)), q)
        out[:, self.pos_idx, 0:3] = out[:, self.pos_idx, 0:3] + pos.to(out.dtype)
        return out

    def achieved_change(self, ach0: torch.Tensor, ach1: torch.Tensor):
        """What the body did between two achieved poses, on the live channels:
        (turns [B, 25, 3] rad, moves [B, 14, 3] cm)."""
        rot = turn_between_rad(ach0[:, self.rot_idx, 3:7], ach1[:, self.rot_idx, 3:7])
        pos = (ach1[:, self.pos_idx, 0:3].float() - ach0[:, self.pos_idx, 0:3].float()) * 200.0
        return rot, pos

    def asked_turn_rad(self, order_cm: torch.Tensor, achieved: torch.Tensor,
                       strength_min: float = 0.0) -> torch.Tensor:
        """[B] the turn an order asks of the body handles relative to where they are, mean over
        handles with strength above strength_min and an asked turn above 1e-4 rad - the
        environment's cmd_mag at the moment the order is issued (env._command_response)."""
        q0 = achieved[:, self.rot_idx, 3:7].float()
        q1 = order_cm[:, self.rot_idx, 3:7].float()
        q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ang = turn_between_rad(q0, q1).norm(dim=-1)                       # [B, 25]
        ok = (order_cm[:, self.rot_idx, 8].float() > strength_min) & (ang > 1e-4)
        s = (ang * ok).sum(-1)
        n = ok.sum(-1).clamp_min(1)
        return s / n


def following_cos(order_cm: torch.Tensor, ach0: torch.Tensor, ach1: torch.Tensor,
                  rot_idx: Sequence[int]) -> torch.Tensor:
    """[B] how well the body followed an order over its tenure: per body handle with strength
    above 0 and an asked turn above 1e-4 rad, the cosine between the turn asked (target against
    the begin pose) and the turn achieved (end pose against the begin pose), averaged - the
    environment's own 250 ms gauge (env._command_response, cmd_response_slow.cos) computed from
    the model's own records. -1 = against, 0 = unrelated, +1 = exactly the asked direction. A body
    with no scorable handle reads 0."""
    q0 = ach0[:, rot_idx, 3:7].float()
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    qt = order_cm[:, rot_idx, 3:7].float()
    qt = qt / qt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q1 = ach1[:, rot_idx, 3:7].float()
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    asked = turn_between_rad(q0, qt)
    got = turn_between_rad(q0, q1)
    na, ng = asked.norm(dim=-1), got.norm(dim=-1)
    cos = (asked * got).sum(-1) / (na * ng).clamp_min(1e-12)
    cos = torch.where(ng > 1e-6, cos, torch.zeros_like(cos))
    ok = (order_cm[:, rot_idx, 8].float() > 0) & (na >= 1e-4)
    return (cos * ok).sum(-1) / ok.sum(-1).clamp_min(1)


def participation_ratio(z: torch.Tensor) -> Optional[float]:
    """The effective number of dimensions z uses: (sum of the covariance's eigenvalues)^2 / sum
    of their squares. 1 = every sample lies on one line, k = spread evenly over all k."""
    if z.shape[0] < 2:
        return None
    c = torch.cov(z.float().T)
    ev = torch.linalg.eigvalsh(c).clamp_min(0.0)
    s = float(ev.sum())
    return None if s <= 0 else float(s * s / float((ev * ev).sum()))


@torch.no_grad()
def following_counts(order_cm, ach0, ach1, rot_idx):
    """Per-part following and its exact random-axis control, on the same realized turn.

    Targets are at the ordered angular distance from the starting pose. The control draws
    a uniform axis on the sphere, so its dot product with the realized axis is uniform on
    [-1, 1]. Quaternion distance uses the absolute dot product, including rotations whose
    alternative quaternion representation has the opposite sign. Zero motion closes no gap.
    Returned arrays contain only finite, nonzero orders to powered body parts.
    """
    q0, qt, q1 = (v[:, rot_idx, 3:7].double() for v in (ach0, order_cm, ach1))
    valid = torch.isfinite(order_cm[:, rot_idx, 8]) & (order_cm[:, rot_idx, 8] > 0)
    for q in (q0, qt, q1):
        valid &= torch.isfinite(q).all(-1) & (q.norm(dim=-1) > 1e-6)
    q0, qt, q1 = (F.normalize(q, dim=-1) for q in (q0, qt, q1))

    def distance(a, b):
        # Quaternion chords retain precision near zero and make identical poses exactly still.
        b = torch.where(((a * b).sum(-1) < 0)[..., None], -b, b)
        return 4 * torch.atan2((a - b).norm(dim=-1), (a + b).norm(dim=-1))

    asked, moved, left = distance(q0, qt), distance(q0, q1), distance(q1, qt)
    valid &= asked >= 1e-4  # same scorable-order floor as following_cos
    ca, cg = (asked / 2).cos(), (moved / 2).cos()
    product = (asked / 2).sin() * (moved / 2).sin()
    safe = product.clamp_min(1e-15)
    chance = ((1 - (ca - ca * cg) / safe) / 2).clamp(0, 1)
    chance += ((1 + (-ca - ca * cg) / safe) / 2).clamp(0, 1)
    chance = torch.where(product > 1e-15, chance, torch.zeros_like(chance))
    return {"closer": (left < asked)[valid], "chance": chance[valid],
            "ordered_deg": torch.rad2deg(asked[valid]),
            "moved_deg": torch.rad2deg(moved[valid]),
            "remaining_deg": torch.rad2deg(left[valid])}


# ------------------------------------------------------------------------ the planner's space
# WHY THE PLANNER HAS NO RESIDUAL BY DEFAULT (--synergy-residual-l2 unset). MEASURED 2026-09-23 (b5
# scratch b5_planner.py): ckpt_step_344400, 1 body in segment 31's world, a synergy fitted on the
# replay diary's orders; at each of 8 planning events the SAME decision was planned six ways. The
# summed squared distance to the goal moved, from the descent's start to its end:
#   raw order (234 numbers per body, today)   0.7118 -> 0.7102   first action asks 1.88 rad per handle
#   z only                                     0.7109 -> 0.7105   asks 0.34 rad
#   z + residual at l2 0.001 / 0.01 / 0.1 / 1  0.7109 -> 0.7104-0.7105 at every weight, residual
#                                              length 0.23-0.26 change scales per body
# The goal distance moves by at most 0.23% under ANY order, because this checkpoint's forward model
# hardly depends on the order at all (DESIGN section 0: improvement over another body's order
# 0.0000-0.0008), so no weight for the residual can be measured against what it buys, and it buys
# nothing measurable here. Planning on z reached the raw planner's distance to within 0.0003 while
# asking the body for 0.34 rad instead of 1.88. The residual stays available behind the flag; its
# weight is to be measured once the forward model's action sensitivity is repaired.
RESIDUAL_MEASUREMENT: dict = {
    "date": "2026-09-23", "source": "b5 scratch b5_planner.py fit, 8 planning events, 1 body",
    "goal_distance_start_end": {"raw": [0.7118, 0.7102], "z": [0.7109, 0.7105],
                                "residual_any_l2": [0.7109, 0.7104]},
    "asked_turn_rad": {"raw": 1.88, "z": 0.34}, "plan_ms_cpu": {"raw": 6783, "z": 6721}}


class SynergyActionSpace:
    """What the planner's Adam descends on when --motor-synergy is on: z [B, k] (starting at the
    encoded controller proposal), the existing strength logits [B, H, 1], the eyes' orientation [B, 4] (the eyes are
    not in the synergy; the planner keeps aiming them itself) and, when residual_l2 is given, a
    per-handle residual [B, n_out] in change-scale units added to S'(z) before the bound, charged
    residual_l2 x its summed square per body. assemble() returns the raw-form order the rest of
    the planner already knows how to execute, transform, smooth and predict."""

    def __init__(self, synergy: MotorSynergy, achieved: torch.Tensor, current_motor: torch.Tensor,
                 residual_l2: Optional[float] = None, observe=None):
        self.syn = synergy
        self.ach = achieved.detach().float()
        self.template = current_motor.detach().float()
        self.residual_l2 = None if residual_l2 is None else float(residual_l2)
        self.observe = observe
        self.z = self.strength = self.gaze = self.res = None

    def frozen_modules(self) -> List[nn.Module]:
        return [self.syn]

    def init(self) -> List[torch.Tensor]:
        B = self.template.shape[0]
        dev = self.template.device

        def _logit(x):
            x = x.clamp(1e-4, 1 - 1e-4)
            return torch.log(x / (1 - x))
        # Use the learned inverse to start from the same proposal as the raw planner.
        # This remains a projection into the learned movement space, not an exact inverse.
        # A fresh decoder still needs training; --synergy-planner off permits that bootstrap.
        with torch.no_grad():
            rot, pos = self.syn.achieved_change(self.ach, self.template)
            pre = {} if self.observe is not None else None
            initial = self.syn.encode(self.syn.to_units(rot, pos), self.ach, pre=pre)
            if self.observe is not None:self.observe(pre)
        if not bool(torch.isfinite(initial).all()):
            raise ValueError("motor synergy encoded a non-finite planner starting point")
        self.z = initial.detach().clone().requires_grad_(True)
        self.strength = _logit(self.template[:, :, 8:9]).clone().requires_grad_(True)
        self.gaze = self.template[:, self.syn.gaze_idx, 3:7].clone().requires_grad_(True)
        params = [self.z, self.strength, self.gaze]
        if self.residual_l2 is not None:
            self.res = torch.zeros(B, self.syn.n_out, device=dev, requires_grad=True)
            params.append(self.res)
        return params

    def assemble(self):
        """(raw-form order [B, H, 9], bounded strength [B, H, 1], extra penalty scalar)."""
        pre = {} if self.observe is not None else None
        rot, pos = self.syn.delta(self.z, self.ach, residual=self.res, pre=pre)
        if self.observe is not None:self.observe(pre)
        s = torch.sigmoid(self.strength)
        order = self.syn.order_from_delta(rot, pos, self.ach, self.template, strength=s,
                                          gaze_quat=F.normalize(self.gaze, dim=-1))
        pen = None
        if self.res is not None:
            pen = self.residual_l2 * (self.res ** 2).sum(-1).mean()
        return order, s, pen

    def summary(self) -> dict:
        with torch.no_grad():
            out = {"z_norm": float(self.z.norm(dim=-1).mean())}
            if self.res is not None:
                out["residual_norm"] = float(self.res.norm(dim=-1).mean())
            return out

    def penalty_per_body(self):
        """Residual cost for selecting each body's best evaluated command."""
        if self.res is None:
            return torch.zeros(self.template.shape[0], device=self.template.device)
        return self.residual_l2 * self.res.square().sum(-1)
