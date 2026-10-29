# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Efference heads: the order the body was sent, given directly, predicts what it does.

WHY THIS EXISTS (docs/gain_of_function_v3/DESIGN.md, decision D5). Over all 344,400 steps of
run H84, knowing the executed order improved the next-moment prediction of every sense by
0.0000 to 0.0008 on a scale where 0 means the order was no use and 1 means the prediction was
exact. The order reaches the forward model only as nine of nineteen columns per handle, pooled
into one motor token. These heads give the order to the prediction a second, direct way in:
the core's latent state AND the executed order go into small networks, and what comes out
is in units a person can picture.

  body head  - for every handle, how far it moves (centimetres, x/y/z in the world) and how
               far it turns (degrees, as a rotation vector) over the next moment (50 ms) and
               over the order's whole tenure (motor_hold moments, 250 ms on H84);
  gaze head  - how far the eyes turn (degrees of azimuth and elevation) over 1, 2 and 3
               moments (50, 100 and 150 ms), the horizon the addendum asks for in section 6;
  gaze copy  - a path from the eye command (with the head's and the eye's orientation) into
               VISION's next-moment prediction, so what the eyes will see is predicted with the
               explicit gaze command, not only through the pooled motor token.

THIS IS VERSION 2 (efference_v2, 2026-09-23, lane-b1-efference; the review of the first version is
docs/gain_of_function_v3/reviews/b1-causal-efference.md). What changed and why:

  THE LATENT IS CENTRED, THEN NORMALISED (review MAJOR 1, DESIGN section 10 rule 1). The core's
  latent carries a part shared by every moment (norm 66 of 75 on ckpt_step_344400; 17.1 of the
  first version's 17.9 pooled output). A running mean of the latent over lived training moments
  (kept in runtime_state.pt) is subtracted from each of the six tokens and each token is then
  LayerNormed, so the attention pools and the layers after them read what varies between moments.
  A LayerNorm alone does not remove a direction every moment shares.
  EVERY TOKEN HAS A PATH (rule 3): beside each attention pool a zero-initialised projection of the
  MEAN of the six normalised tokens is added, so a pool that settles on one token cannot hide the
  others. The first version's pools went from spread 0.86 and 0.91 to 0.23 and 0.31 (1 = equal
  weight on all six, 0 = all weight on one) in 100 head-only steps on real moments.
  THE POOLS ARE BOUNDED (TokenPool, POOL_SPREAD_FLOOR): measured on the checkpoint's own moments,
  centring and the mean path did NOT stop an ordinary attention pool from sharpening onto single
  tokens (0.96 to 0.27), so each pool compares its query with the tokens by cosine times a fixed
  bound that keeps every head at least half as spread as an even weighting.
  LIVENESS (rule 2): every window reports each hidden layer's share of units whose input rose above
  -3 and each pool's spread, and the share of each input that was shared by every moment of the
  window (the control that shows the centring working).

  THE BODY HEAD SEES THE ASKED CHANGE (review MAJOR 2). How far a part moves depends on where it
  was told to go MINUS where it is. The first version got only the absolute target (a vector of
  length 6.28 of which 5.70 was the same every moment) and learned to ignore it (the order's share
  of its first layer's input variance fell from 0.20 to 0.12). It now also gets, per handle, the
  asked position change and asked turn relative to the achieved pose, in the tenure divisors
  (1.80 cm and 11.0 degrees: one unit of input = one typical order-long move), and the strength.
  The absolute order stays beside it, centred by its own running mean.

  THE GAZE HEAD HAS ITS OWN LAYERS (review MINOR 5). The first version retrained the vestigial
  GazeHead as a 50 ms change head; several lem/* paths read that module's output as the ABSOLUTE
  direction the eyes should point. From v2 on the GazeHead is never touched; a checkpoint whose
  GazeHead the first version retrained keeps the receipt stamp gaze_head_repurposed = True, and
  lem.training.checkpoint refuses to load such a GazeHead into any caller that reads it as a
  direction.
  HEAD POSE (review MAJOR 4, addendum section 6: "Include head pose"): the gaze head and the gaze
  copy read the head's and the eye's achieved world orientation (the achieved pose's "head" and
  "gaze" rows, always present); with --vestibular-sense on the gaze head also reads the head's
  accelerometer (gravity) and gyro, through zero-initialised columns.

GRADIENT. The next-moment body outputs, the 50 ms gaze outputs and the gaze copy are graded on the
latent the delayed next-frame lesson has just computed with gradient, so their error trains the
core, the encoders and the heads together. The longer horizons (the tenure, 100 ms and 150 ms)
finish after the optimizer has already stepped, so they are graded on the stored latent with the
gradient stopped: they train only these heads. The loss SUMS the horizons with declared weights
(HORIZON_WEIGHTS), so the part that reaches the core does not shrink on the steps the longer
horizons are also graded (review MINOR 4); the efference block reports that part on its own.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

EFFERENCE_VERSION = "efference_v2"
GAZE_HORIZONS = (1, 2, 3)          # moments; x 50 ms at --action-repeat 5

# THE DIVISORS, MEASURED (b1-causal-efference, 2026-09-22). Each is the root-mean-square, per
# axis, of the change it divides, over 248 model steps of ONE body in H84 segment 31's own world
# (args.json of the segment, crib, caregiver, mobile, critters, babble 0.05), acting on the
# frozen weights of ckpt_step_344400 on the CPU, the trainer's latch rule (a new order every 5
# model steps). Script and raw arrays: the b1 scratchpad (b1_measure_scales.py, b1_scales.json);
# the numbers are copied into docs/gain_of_function_v3/agents/b1-causal-efference.md. With these
# divisors one measured spread is one unit of error on that data. Excluding 20, 30 or 40
# settling steps after the world's reset instead of 12 moves every divisor by 5% or less
# (0.452 -> 0.430 cm, 2.73 -> 2.62 degrees, 3.23 -> 3.19 degrees). The 25 body
# handles are pooled (the eyes' handle is the gaze head's). One body on frozen weights moves as
# the eight live bodies do only as far as the same policy and world make it; the deploy's first
# windows carry the live do-nothing errors (efference.*_do_nothing_*) that check it.
# For scale: 99 in 100 one-moment moves stayed under 1.93 cm per axis and 99 in 100 order-long
# moves under 7.33 cm and 42.2 degrees per axis; 99 in 100 eye moves under 8.3, 15.7 and 22.1
# degrees at 50, 100 and 150 ms.
EFFERENCE_SCALES: Dict[str, float] = {
    "pos_next_cm": 0.452,     # handle position change over one moment (50 ms), cm per axis
    "rot_next_deg": 2.73,     # handle rotation change over one moment, degrees per axis
    "pos_tenure_cm": 1.80,    # over the order's tenure (5 moments, 250 ms)
    "rot_tenure_deg": 11.0,
    "gaze_deg_1": 3.23,       # eye azimuth/elevation change over 1, 2, 3 moments, degrees per axis
    "gaze_deg_2": 6.22,
    "gaze_deg_3": 8.86,
}

# THE ASKED CHANGE'S DIVISORS: the tenure divisors above (the task's rule, lane-b1-efference), so one
# unit of asked change is one typical ACHIEVED order-long move. What the orders really ask is larger
# than what the body achieves. MEASURED 2026-09-23 on the lesson inputs of ckpt_step_344400 (120
# lived moments, 2 bodies, 25 body handles; docs/gain_of_function_v3/measurements/
# efference_liveness_ckpt_step_344400.json "fixture"): the asked change reads 4.36 units of position
# (7.8 cm) and 1.64 units of turn (18 degrees) root mean square per axis, 99 in 100 under 10.0 and
# 4.7 units (b4 measured 9.30 cm and 21.0 degrees on its own orders); the reach clamp bounds the
# position at 40 cm per axis (22 units). The body head reads it through a plain linear layer.
ASK_POS_CM = EFFERENCE_SCALES["pos_tenure_cm"]
ASK_ROT_DEG = EFFERENCE_SCALES["rot_tenure_deg"]
ASK_F = 7            # per handle: asked position change (3), asked turn (3), strength (1)
POSE_F = 8           # the head's and the eye's achieved world orientation, x y z w each
GAZE_ORDER_F = 11    # the eyes' order row (9) and the command's azimuth and elevation (2)
HEAD_MOTION_F = 6    # --vestibular-sense on: head accelerometer x y z (g), head gyro x y z (scale)
HEAD_MOTION_FIELDS = ("head_acc_x", "head_acc_y", "head_acc_z",
                      "head_gyro_x", "head_gyro_y", "head_gyro_z")

# THE LOSS'S HORIZON WEIGHTS (review MINOR 4). Each graded horizon counts once: the body's next
# moment and its tenure, the eyes at 50, 100 and 150 ms. The first version took the MEAN of the
# horizons graded that step, so the next-moment part (the only part that trains the core) was
# halved on tenure steps and cut to a third once the eyes' longer horizons existed, and nothing
# said so. With a sum, the part that reaches the core is the same on every step: weight x (body next
# + eyes at 50 ms), reported as loss_body_core / loss_gaze_core.
HORIZON_WEIGHTS: Dict[str, float] = {"body_next": 1.0, "body_tenure": 1.0,
                                     "gaze_1": 1.0, "gaze_2": 1.0, "gaze_3": 1.0}

# THE RUNNING MEANS THAT CENTRE THE INPUTS: an exponential average over lived training moments with
# this horizon, divided by the weight it has accumulated (corrected for starting at zero; the first
# lived moment sets the mean to itself). Arithmetic, the contact head's (lem/model/contact_head.py
# CENTRE_HORIZON_STEPS): the part the centre removes is shared by every moment (DESIGN section 10:
# norm 66 of 75 in the core latent), so any horizon longer than a few moments finds it; 2,000
# moments (100 s) follows the slow drift of the weights and moves at most 1 - exp(-200/2000) = 9.5%
# while one goal ring's 200 moments are lived. The control that catches it lying is the window's
# input_shared_share (the share of each centred input that every moment of the window shared:
# 0 = nothing shared, 1 = every moment identical).
CENTRE_HORIZON_STEPS = 2000

# DESIGN section 10 rule 2: a unit is live when its input rose above this line on the window's
# samples (GELU of -3 is -0.004; below it a unit returns about nothing and passes no gradient)
LIVE_LINE = -3.0

# THE LESSON'S TIME on the graphics card is read with the card synchronised, on one step in this
# many (a synchronisation costs the overlap of the kernels queued before it); on the CPU every step
EFFERENCE_TIMING_EVERY_CUDA = 10


def rotation_change_deg(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """The turn that carries orientation q0 to q1, as a rotation vector in DEGREES.

    q0, q1: [..., 4] unit quaternions, x y z w (the achieved-pose layout). The result is the
    axis of q1 * conj(q0) times its angle, in the world frame; its length is the angle turned
    (0 = no turn, 180 = the largest possible). q and -q are the same orientation, so the shorter
    of the two turns is always taken."""
    q0 = q0.float()
    q1 = q1.float()
    x0, y0, z0, w0 = q0.unbind(-1)
    x1, y1, z1, w1 = q1.unbind(-1)
    w = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
    x = -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0
    y = -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0
    z = -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0
    v = torch.stack([x, y, z], dim=-1)
    sign = torch.where(w < 0, -1.0, 1.0).unsqueeze(-1)
    v = v * sign
    w = w.abs()
    n = v.norm(dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(n, w.unsqueeze(-1))
    # below 1e-9 the axis is undefined and sin(ang/2) = ang/2, so the vector is 2 v
    axis_angle = torch.where(n > 1e-9, v / n.clamp_min(1e-12) * ang, 2.0 * v)
    return axis_angle * (180.0 / math.pi)


def pose_change(ach0: torch.Tensor, ach1: torch.Tensor, pos_unit_cm: float = 200.0):
    """How far every handle moved and turned between two achieved poses.

    ach0, ach1: [B, n_handles, >=7] achieved handle state - positions divided by pos_unit_cm
    (the tokenizer's 200 cm), then an x y z w world quaternion. Returns (cm [B, n, 3],
    degrees [B, n, 3])."""
    cm = (ach1[..., 0:3].float() - ach0[..., 0:3].float()) * float(pos_unit_cm)
    deg = rotation_change_deg(ach0[..., 3:7], ach1[..., 3:7])
    return cm, deg


def gaze_order_features(order_enc: torch.Tensor, gaze_idx: int,
                        convention: str = "legacy") -> torch.Tensor:
    """The gaze handle's row of an order, plus the command's azimuth and elevation in radians.

    order_enc: [B, n_handles, 9] in encoder units. The eyes are handle `gaze_idx`; its x y z w
    quaternion is the eye command (LemLite.gaze_to_handle writes it). The two angles are the
    same numbers the eye servo is sent, so the head sees the command in the eye's own terms,
    the explicit gaze-command copy the addendum's section 6 asks for. Returns [B, 11].

    convention: the model's eye row convention (mimo_bridge.gaze_reflex.EYE_ROW_CONVENTIONS). Under
    'physical' the row's pitch is MINUS the elevation sent (the eye's physical rotation, motor
    interface audit 2026-09-25, F1), so the elevation is read back as minus the pitch and the two
    angles stay the numbers the servo is sent."""
    from mimo_bridge.gaze_reflex import eye_pitch_sign
    row = order_enc[:, gaze_idx, :].float()
    q = row[:, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    el = pitch if eye_pitch_sign(convention) > 0 else -pitch
    return torch.cat([row, yaw.unsqueeze(-1), el.unsqueeze(-1)], dim=-1)


def asked_change(order_enc: torch.Tensor, achieved: torch.Tensor, gaze_idx: int) -> torch.Tensor:
    """[B, H, 7] what the order asks of every handle relative to where it is (review MAJOR 2).

    order_enc: [B, H, 9] encoder units (positions / 200 cm, world quaternion, radius, strength);
    achieved: [B, H, >=7] achieved pose (positions / 200 cm, world quaternion). Per handle: the
    asked position change in cm / ASK_POS_CM, the asked turn (degrees as a rotation vector) /
    ASK_ROT_DEG, and the strength (0-1). The eyes' row is zero: its order row is a joint-space eye
    command and its achieved row the eye body's world orientation (161 degrees apart on the probes'
    measurement), so their difference is not a change; the eyes are the gaze head's."""
    o = order_enc.float()
    a = achieved.float()
    dpos = (o[..., 0:3] - a[..., 0:3]) * 200.0 / ASK_POS_CM
    q1 = o[..., 3:7] / o[..., 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q0 = a[..., 3:7] / a[..., 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    drot = rotation_change_deg(q0, q1) / ASK_ROT_DEG
    out = torch.cat([dpos, drot, o[..., 8:9]], dim=-1)
    if 0 <= int(gaze_idx) < out.shape[1]:
        keep = torch.ones(out.shape[1], 1, device=out.device, dtype=out.dtype)
        keep[int(gaze_idx)] = 0.0
        out = out * keep
    return out


def eye_row_quaternion(az: torch.Tensor, el: torch.Tensor, convention: str = "legacy") -> torch.Tensor:
    """[B, 4] x y z w: the eyes' row of a command for the azimuth and elevation (radians) the eye servo is sent,
    exactly as LemLite.gaze_to_handle writes the executed ledger (the row's pitch is plus the elevation under
    'legacy', minus it under 'physical')."""
    from mimo_bridge.gaze_reflex import eye_pitch_sign
    pitch = el if eye_pitch_sign(convention) > 0 else -el
    ca, sa = torch.cos(az / 2.0), torch.sin(az / 2.0)
    ce, se = torch.cos(pitch / 2.0), torch.sin(pitch / 2.0)
    return torch.stack([-sa * se, ca * se, sa * ce, ca * ce], dim=-1)


def size_matched_other_order(order_cm: torch.Tensor, achieved: torch.Tensor, gaze_idx: int,
                             eye_deg: Optional[torch.Tensor] = None, convention: str = "legacy",
                             shift: int = 1) -> torch.Tensor:
    """ANOTHER body's command, size-matched to this body: the swap test's and the side heads' size-matched control
    (2026-10-02, docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md). [B, H, 9], execution units.

    THE 25 BODY ROWS are exactly lem/model/contact_head.shuffled_order (the contact head's control): body b-shift's
    command taken as a change from body b-shift's own achieved pose and applied to body b's pose - position = this
    pose + (other target - other pose) in cm, orientation = (other target x inverse of the other pose) x this pose -
    so every asked move and turn is the other body's own; radius and strength as they stand.

    THE EYES' ROW. shuffled_order keeps the other body's eye row as it stands (the contact head reads no eye row). The
    row is an eye-in-head joint target - the azimuth and elevation the eye servo is sent (LemLite.gaze_to_handle) -
    and the turn it asks is that target minus where the eyes are, so another body's target asks this body's eyes a
    turn the other body was never asked. With eye_deg ([B, 2] degrees: each body's realised eye azimuth and elevation
    at the state the command starts from, LemLite._eye_angles_deg, which reads the eye joints in the servo's own
    coordinates, action_decoder.EYE_JOINTS and env.eye_yaw_pitch) the row is rebuilt the same way as the body rows:
    the other body's asked eye turn (its eye command minus its eyes' angles) added to this body's eyes' angles,
    clamped to the eye joints' ranges (gaze_reflex.AZ_RANGE, EL_RANGE) and written back with gaze_to_handle's
    quaternion; its other seven numbers stay the other body's. Without eye_deg it stays the other body's as it
    stands (shuffled_order's), and the caller counts it.

    order_cm: [B, H, 9] position cm, quaternion x y z w, radius, strength; achieved: [B, H, >=7] positions / 200 cm,
    quaternion x y z w."""
    from lem.model.contact_head import shuffled_order
    out = shuffled_order(order_cm, achieved, gaze_idx, shift=shift)
    gi = int(gaze_idx)
    if eye_deg is None or not torch.is_tensor(eye_deg) or tuple(eye_deg.shape) != (out.shape[0], 2) \
            or not (0 <= gi < out.shape[1]):
        return out
    from mimo_bridge.gaze_reflex import AZ_RANGE, EL_RANGE
    cmd = gaze_order_features(order_cm.float(), gi, convention)[:, 9:11]           # [B, 2] az, el sent, radians
    eye = eye_deg.detach().to(device=out.device, dtype=torch.float32) * (math.pi / 180.0)
    asked = cmd.roll(shift, dims=0) - eye.roll(shift, dims=0)                       # the other body's asked eye turn
    az = (eye[:, 0] + asked[:, 0]).clamp(*AZ_RANGE)
    el = (eye[:, 1] + asked[:, 1]).clamp(*EL_RANGE)
    out[:, gi, 3:7] = eye_row_quaternion(az, el, convention).to(out.dtype)
    return out


def head_eye_pose(achieved: torch.Tensor, head_idx: int, gaze_idx: int) -> torch.Tensor:
    """[B, 8] the head's and the eye's achieved world orientation (x y z w each, unit length,
    the sign chosen so w >= 0: q and -q are one orientation). DESIGN D5 as amended 2026-09-23:
    the addendum's "Include head pose"."""
    a = achieved.float()
    out = []
    for i in (int(head_idx), int(gaze_idx)):
        q = a[:, i, 3:7]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q = torch.where(q[:, 3:4] < 0, -q, q)
        out.append(q)
    return torch.cat(out, dim=-1)


def head_motion_from_vestibular(vec: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """[B, 6] the head's accelerometer (g) and gyro (scale units) out of the vestibular vector
    ([B, 29], mimo_bridge.vestibular_adapter.VESTIBULAR_FIELDS order), or None."""
    if not torch.is_tensor(vec) or vec.dim() != 2:
        return None
    from mimo_bridge.vestibular_adapter import FIELD_INDEX
    idx = [FIELD_INDEX[n] for n in HEAD_MOTION_FIELDS]
    if vec.shape[1] <= max(idx):
        return None
    return vec[:, idx].float()


# THE POOLS' SPREAD FLOOR (DESIGN section 10 rules 2-4, review of b1 MAJOR 1). MEASURED 2026-09-23
# (lane-b1-efference, liveness_fixture.py on the ckpt_step_344400 fixture, 100 head-only Adam steps
# at the checkpoint's learning rate): with the latent centred and LayerNormed and the all-token path
# beside it, an ordinary attention pool STILL sharpened onto single tokens - per head, the body pool
# from 0.96 to 0.27 of the even-spread entropy and the gaze pool from 0.96 to 0.54 (the first
# version's pools, 0.77 to 0.23 and 0.90 to 0.36 in the same run). Centring removes the shared part
# the pools read; it does not stop a query from growing onto one token. So each pool here compares
# its query with the tokens by COSINE (both unit length per head) times a fixed bound, which caps
# how far any head can concentrate: with the bound below, the most concentrated weighting a head can
# reach over six tokens (one token at the largest similarity, the other five at the smallest) still
# keeps half the entropy of an even spread and at most 77 of every 100 parts of its weight on one
# token. The line is the majority line rule 4 draws for live units, applied to entropy; the window
# reports the spread every head actually keeps (pool_spread), which is the control. MEASURED WITH THE
# BOUND (same run): body and gaze pools 0.99 and 0.99 at the start, 0.85 and 0.85 after the 100 steps;
# with the bound lifted to 20 (the control) 0.17 and 0.21; every hidden layer stayed live (share 1.0)
# in every case, so live share alone cannot see a pool lock
# (docs/gain_of_function_v3/measurements/efference_liveness_ckpt_step_344400.json).
POOL_SPREAD_FLOOR = 0.5


def logit_bound(n_tokens: int, floor: float = POOL_SPREAD_FLOOR) -> float:
    """The largest cosine scale t at which softmax([t, -t, ..., -t]) over n_tokens keeps `floor` of
    the even-spread entropy (bisection; the entropy share falls as t grows). 1.4225 for six tokens."""
    n = max(2, int(n_tokens))

    def share(t):
        z = torch.tensor([t] + [-t] * (n - 1), dtype=torch.float64)
        p = torch.softmax(z, 0)
        return float(-(p * p.log()).sum() / math.log(n))
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if share(mid) > floor:
            lo = mid
        else:
            hi = mid
    return lo


class TokenPool(nn.Module):
    """A learned query pools the tokens by bounded (cosine) attention: per head, the query and each
    token's key are made unit length and their product is scaled by logit_bound(n_tokens), so no
    head can give more than about three quarters of its weight to one of six tokens (POOL_SPREAD_
    FLOOR says why). Returns the pooled vector [B, d] and the per-head weights [B, heads, 1, n]."""

    def __init__(self, d_model: int, n_heads: int, n_tokens: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must divide into the pool's heads")
        self.n_heads = int(n_heads)
        self.dh = int(d_model) // self.n_heads
        self.bound = float(logit_bound(n_tokens))
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.qnorm = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        B, n, d = x.shape
        q = self.wq(self.qnorm(self.query)).reshape(1, self.n_heads, 1, self.dh)
        k = self.wk(x).reshape(B, n, self.n_heads, self.dh).transpose(1, 2)       # [B, h, n, dh]
        v = self.wv(x).reshape(B, n, self.n_heads, self.dh).transpose(1, 2)
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        w = torch.softmax(self.bound * (q * k).sum(-1, keepdim=True).transpose(-1, -2), dim=-1)
        out = (w.to(v.dtype) @ v).reshape(B, self.n_heads * self.dh)               # [B, d]
        return self.wo(out), w                                                      # w [B, h, 1, n]


class EfferenceHeads(nn.Module):
    """The action-conditioned heads (version 2). Body and gaze outputs are in units of
    EFFERENCE_SCALES (to_lived gives centimetres and degrees); the gaze copy's output is a change
    of vision's next-moment token.

    latent:   [B, n_mod, d_model] - the forward model's last-step tokens for a state whose motor
              token already carries the order (the delayed next-frame lesson's nf_last).
    order:    [B, n_handles, 9] - the executed order in ENCODER units (command_for_encoder).
    achieved: [B, n_handles, >=7] - the achieved pose of the state the order starts from.
    """

    GAZE_ORDER_F = GAZE_ORDER_F
    # the hidden layers whose live share the window reports
    LIVE_LAYERS = ("body_mlp.0", "body_mlp.2", "gaze_in", "vision_in")
    POOLS = ("body_pool", "gaze_pool")

    def __init__(self, d_model: int, n_handles: int, gaze_idx: int, head_idx: int, n_mod: int,
                 vision_idx: Optional[int] = None, hidden: int = 256, n_heads: int = 4,
                 gaze_hidden: int = 128, vision_hidden: int = 128,
                 eye_row_convention: str = "legacy"):
        super().__init__()
        self.d_model = int(d_model)
        self.n_handles = int(n_handles)
        self.gaze_idx = int(gaze_idx)
        self.head_idx = int(head_idx)
        self.n_mod = int(n_mod)
        self.vision_idx = None if vision_idx is None else int(vision_idx)
        # how the order's eye row carries the elevation sent (gaze_order_features); a plain setting,
        # not a buffer, so the saved tensors are the same under either convention
        self.eye_row_convention = str(eye_row_convention)
        # the running means (not in the state dict: running statistics, saved in runtime_state.pt)
        self.register_buffer("centre_mean", torch.zeros(self.n_mod, self.d_model), persistent=False)
        self.centre_weight = 0.0
        self.register_buffer("order_centre", torch.zeros(self.n_handles * 9), persistent=False)
        self.order_centre_weight = 0.0
        self.lat_norm = nn.LayerNorm(self.d_model)
        # ---- body: a learned query pools the normalised tokens (bounded attention, TokenPool); the
        # zero-initialised mean path beside it reaches all six tokens whatever the pool attends to
        self.body_pool = TokenPool(self.d_model, n_heads, self.n_mod)
        self.body_all = nn.Linear(self.d_model, self.d_model)
        self.body_mlp = nn.Sequential(
            nn.Linear(self.d_model + self.n_handles * 9 + self.n_handles * ASK_F, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.n_handles * 12))
        # ---- gaze: its own pool and layers (the vestigial GazeHead is not touched)
        self.gaze_pool = TokenPool(self.d_model, n_heads, self.n_mod)
        self.gaze_all = nn.Linear(self.d_model, self.d_model)
        self.gaze_in = nn.Linear(self.d_model + GAZE_ORDER_F + POSE_F, gaze_hidden)
        self.gaze_motion = nn.Linear(HEAD_MOTION_F, gaze_hidden, bias=False)
        self.gaze_out = nn.Linear(gaze_hidden, 2 * len(GAZE_HORIZONS))
        # ---- the gaze copy into vision's next-moment prediction
        self.vision_in = nn.Linear(self.d_model + GAZE_ORDER_F + POSE_F, vision_hidden)
        self.vision_out = nn.Linear(vision_hidden, self.d_model)
        from .activation_compensation import VisualActivationCompensation
        self.vision_activation = VisualActivationCompensation(
            self.d_model + GAZE_ORDER_F + POSE_F, vision_hidden, self.d_model)
        with torch.no_grad():
            for lin in (self.body_all, self.gaze_all, self.vision_out):
                lin.weight.zero_()
                lin.bias.zero_()
            self.gaze_motion.weight.zero_()
        self._win_reset()

    # ------------------------------------------------------------------ running statistics
    def _win_reset(self):
        # window accumulators (plain tensors, dicts and numbers; read and reset by window_liveness)
        self.live_max: Dict[str, torch.Tensor] = {}
        self.live_lo: Dict[str, torch.Tensor] = {}
        self.live_hi: Dict[str, torch.Tensor] = {}
        self.spread_sum: Dict[str, list] = {}
        self.shared_sum: Dict[str, list] = {}

    @torch.no_grad()
    def ensure_centres(self, latent: torch.Tensor, order: Optional[torch.Tensor] = None) -> None:
        """The first lived moment ever sets each running mean to itself (a zero centre would hand
        the LayerNorm the shared part it exists to remove). No-op once set."""
        if self.centre_weight <= 0.0 and latent is not None:
            self.centre_mean.copy_(latent.detach().float().mean(dim=0).to(self.centre_mean.dtype))
            self.centre_weight = 1.0 / float(CENTRE_HORIZON_STEPS)
        if order is not None and self.order_centre_weight <= 0.0:
            self.order_centre.copy_(order.detach().float().reshape(order.shape[0], -1).mean(0)
                                    .to(self.order_centre.dtype))
            self.order_centre_weight = 1.0 / float(CENTRE_HORIZON_STEPS)

    @torch.no_grad()
    def update_centres(self, latent: torch.Tensor, order: Optional[torch.Tensor] = None) -> None:
        """One lived training moment of the running means (all bodies pooled)."""
        a = 1.0 / float(CENTRE_HORIZON_STEPS)
        for buf, attr, x in (("centre_mean", "centre_weight", latent),
                             ("order_centre", "order_centre_weight", order)):
            if x is None:
                continue
            b = getattr(self, buf)
            v = x.detach().float().reshape(x.shape[0], *b.shape).mean(dim=0).to(b.dtype)
            w_old = float(getattr(self, attr))
            w_new = (1.0 - a) * w_old + a
            b.mul_((1.0 - a) * w_old / w_new).add_(v * (a / w_new))
            setattr(self, attr, w_new)

    def runtime_state(self) -> dict:
        return {"version": EFFERENCE_VERSION,
                "centre_mean": self.centre_mean.detach().cpu(),
                "centre_weight": float(self.centre_weight),
                "order_centre": self.order_centre.detach().cpu(),
                "order_centre_weight": float(self.order_centre_weight)}

    @torch.no_grad()
    def load_runtime_state(self, rt: Optional[dict]) -> bool:
        """Restore the running means saved beside a checkpoint; False (cold start: the first
        lived moment sets them) when none were saved or their shapes differ."""
        if not rt:
            return False
        ok = True
        for buf, attr in (("centre_mean", "centre_weight"), ("order_centre", "order_centre_weight")):
            t = rt.get(buf)
            b = getattr(self, buf)
            if torch.is_tensor(t) and tuple(t.shape) == tuple(b.shape):
                b.copy_(t.to(b))
                setattr(self, attr, float(rt.get(attr, 0.0)))
            else:
                ok = False
        return ok

    # ------------------------------------------------------------------ the inputs
    def normalise(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, n_mod, d] centred by the running mean, then LayerNormed per token."""
        return self.lat_norm(latent.float() - self.centre_mean.to(latent.device))

    def _order_input(self, order: torch.Tensor) -> torch.Tensor:
        B = order.shape[0]
        return order.reshape(B, -1).float() - self.order_centre.to(order.device)

    @torch.no_grad()
    def _note_live(self, name: str, pre: torch.Tensor) -> None:
        p = pre.detach().float().reshape(-1, pre.shape[-1])
        m = p.max(dim=0).values
        prev = self.live_max.get(name)
        self.live_max[name] = m if prev is None or prev.shape != m.shape else torch.maximum(prev, m)
        # each unit's GELU output range over the window (every hidden layer here is read through GELU): the -3 line
        # alone cannot see a unit held at one active value (window_liveness's liveness_varying)
        if not hasattr(self, "live_lo"):                  # an object restored from before the field existed
            self.live_lo, self.live_hi = {}, {}
        post = F.gelu(p)
        lo, hi = post.min(dim=0).values, post.max(dim=0).values
        plo = self.live_lo.get(name)
        same = plo is not None and plo.shape == lo.shape
        self.live_lo[name] = torch.minimum(plo, lo) if same else lo
        self.live_hi[name] = torch.maximum(self.live_hi[name], hi) if same else hi

    # The window accumulators stay on the device (no read per step: the lesson makes ONE device
    # read per step for all its gauges); window_liveness reads them once per report window.
    @torch.no_grad()
    def _note_shared(self, name: str, x: torch.Tensor) -> None:
        x = x.detach().float().reshape(x.shape[0], -1)
        s = self.shared_sum.get(name)
        vsum, nsum = x.sum(dim=0), x.norm(dim=-1).sum()
        if s is None or s[0].shape != vsum.shape:
            self.shared_sum[name] = [vsum, nsum, int(x.shape[0])]
        else:
            s[0] = s[0] + vsum
            s[1] = s[1] + nsum
            s[2] += int(x.shape[0])

    @torch.no_grad()
    def _note_spread(self, name: str, w: torch.Tensor) -> None:
        """w: [B, heads, 1, n_keys] per-head attention weights. The spread of each head is its
        entropy as a share of the even-spread maximum (1 = equal weight on every key, 0 = all
        weight on one); a pool whose four heads each read ONE different key reads 0 here even
        though the heads' average looks spread (the motor pool's measured failure)."""
        p = w.detach().float().clamp_min(1e-12)
        n = p.shape[-1]
        ent = -(p * p.log()).sum(dim=-1) / math.log(max(n, 2))
        s = self.spread_sum.get(name)
        if s is None:
            self.spread_sum[name] = [ent.sum(), int(ent.numel())]
        else:
            s[0] = s[0] + ent.sum()
            s[1] += int(ent.numel())

    def _pool(self, x, pool, all_path, record: Optional[str]):
        out, w = pool(x)
        pooled = out + all_path(x.mean(dim=1))
        if record is not None:
            self._note_spread(record, w)
        return pooled

    # ------------------------------------------------------------------ the heads
    def body(self, latent: torch.Tensor, order: torch.Tensor, achieved: torch.Tensor,
             record: bool = False) -> torch.Tensor:
        """[B, n_handles, 12]: next pos(3) rot(3), tenure pos(3) rot(3), in scale units."""
        B = latent.shape[0]
        x = self.normalise(latent)
        pooled = self._pool(x, self.body_pool, self.body_all, "body_pool" if record else None)
        o = self._order_input(order)
        ask = asked_change(order, achieved, self.gaze_idx).reshape(B, -1)
        inp = torch.cat([pooled, o.to(pooled.dtype), ask.to(pooled.dtype)], dim=-1)
        l0, a0, l1, a1, l2 = self.body_mlp
        h0 = l0(inp)
        h1 = l1(a0(h0))
        out = l2(a1(h1))
        if record:
            self._note_live("body_mlp.0", h0)
            self._note_live("body_mlp.2", h1)
            self._note_shared("latent", latent.float() - self.centre_mean.to(latent.device))
            self._note_shared("order", o)
        return out.reshape(B, self.n_handles, 12)

    def gaze(self, latent: torch.Tensor, order: torch.Tensor, achieved: torch.Tensor,
             head_motion: Optional[torch.Tensor] = None, record: bool = False) -> torch.Tensor:
        """[B, 3, 2]: eye azimuth/elevation change at 1, 2, 3 moments, in scale units.
        head_motion: [B, 6] (head_motion_from_vestibular) or None (the columns read zero)."""
        x = self.normalise(latent)
        pooled = self._pool(x, self.gaze_pool, self.gaze_all, "gaze_pool" if record else None)
        feats = gaze_order_features(order, self.gaze_idx,
                                    getattr(self, "eye_row_convention", "legacy")).to(pooled.dtype)
        pose = head_eye_pose(achieved, self.head_idx, self.gaze_idx).to(pooled.dtype)
        h = self.gaze_in(torch.cat([pooled, feats, pose], dim=-1))
        if head_motion is not None:
            h = h + self.gaze_motion(head_motion.to(h.dtype))
        if record:
            self._note_live("gaze_in", h)
        return self.gaze_out(F.gelu(h)).reshape(-1, len(GAZE_HORIZONS), 2)

    def vision_delta(self, latent: torch.Tensor, order: torch.Tensor, achieved: torch.Tensor,
                     record: bool = False) -> Optional[torch.Tensor]:
        """[B, d_model]: the gaze copy's change of vision's next-moment token (zero at the start:
        the output layer is zero-initialised, so switching it on changes no prediction). Reads the
        latent's vision token (centred, normalised), the eye command and the head's and the eye's
        orientation. None when the build has no vision."""
        if self.vision_idx is None:
            return None
        x = self.normalise(latent)[:, self.vision_idx]
        feats = gaze_order_features(order, self.gaze_idx,
                                    getattr(self, "eye_row_convention", "legacy")).to(x.dtype)
        pose = head_eye_pose(achieved, self.head_idx, self.gaze_idx).to(x.dtype)
        inputs = torch.cat([x, feats, pose], dim=-1)
        output, h = self.vision_activation.forward_with_preactivation(inputs,self.vision_in,self.vision_out)
        if record:
            self._note_live("vision_in", h)
        return output

    # ------------------------------------------------------------------ reports
    def window_liveness(self) -> dict:
        """{liveness: {layer: live share 0..1}, liveness_varying: {layer: varying share 0..1},
        pool_spread: {pool: 0..1}, input_shared_share: {input: 0..1}} over the window's recorded
        samples; a fresh window starts. Absent keys = nothing recorded. liveness_varying: the share
        of a layer's units whose GELU output ranged over the window by more than one bf16 step at
        the unit's own size and the widest output below the -3 line (lem/model/liveness.py, A UNIT
        THAT VARIES): the -3 line alone reads a unit held at one active value live."""
        from .liveness import varying_floor
        live = {k: round(float((v > LIVE_LINE).float().mean()), 4) for k, v in self.live_max.items()}
        varying = {}
        for k, lo in getattr(self, "live_lo", {}).items():
            hi = self.live_hi[k]
            varying[k] = round(float(((hi - lo) > varying_floor(lo, hi, "gelu").to(hi)).float().mean()), 4)
        spread = {k: round(float(s[0]) / s[1], 4) for k, s in self.spread_sum.items() if s[1] > 0}
        shared = {}
        for k, (vsum, nsum, n) in self.shared_sum.items():
            nsum = float(nsum)
            if n > 0 and nsum > 0:
                shared[k] = round(float((vsum / n).norm()) / (nsum / n), 4)
        self._win_reset()
        return {"liveness": live, "liveness_varying": varying, "pool_spread": spread, "input_shared_share": shared}

    @staticmethod
    def to_lived(body: Optional[torch.Tensor] = None, gaze: Optional[torch.Tensor] = None,
                 scales: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Scale units -> centimetres and degrees."""
        s = EFFERENCE_SCALES if scales is None else scales
        out = {}
        if body is not None:
            b = body.float()
            out["body_next_cm"] = b[..., 0:3] * s["pos_next_cm"]
            out["body_next_deg"] = b[..., 3:6] * s["rot_next_deg"]
            out["body_tenure_cm"] = b[..., 6:9] * s["pos_tenure_cm"]
            out["body_tenure_deg"] = b[..., 9:12] * s["rot_tenure_deg"]
        if gaze is not None:
            g = gaze.float()
            k = torch.tensor([s["gaze_deg_%d" % h] for h in GAZE_HORIZONS],
                             dtype=g.dtype, device=g.device).view(1, 3, 1)
            out["gaze_deg"] = g * k
        return out


class GazeCopy:
    """The gaze copy, attached to the prediction heads as `_efference_gaze_copy` (a plain callable
    attribute: never a registered submodule, so the prediction heads' saved tensors do not change).

    Every place that turns a latent into a next-moment prediction with a known order calls it:
    the delayed next-frame lesson, the paired shuffled-order audit and replay (lem_model.py), and
    lem.model.predictive_state.transition (the planner, its Adam loop, the imagination chain, the
    causal-chain predictions and the probes' rollouts). It adds the path's output to the vision
    prediction; before the first lived moment has set the latent's centre it adds nothing."""

    def __init__(self, heads: EfferenceHeads, vision_key: str = "vision"):
        self.heads = heads
        self.vision_key = vision_key

    def __call__(self, preds: Dict[str, torch.Tensor], latent: torch.Tensor,
                 order_enc: Optional[torch.Tensor], achieved: Optional[torch.Tensor],
                 mask: Optional[torch.Tensor] = None, record: bool = False):
        if (self.vision_key not in preds or order_enc is None or achieved is None
                or latent is None or self.heads.centre_weight <= 0.0
                or self.heads.vision_idx is None):
            return preds
        if order_enc.shape[0] != latent.shape[0] or achieved.shape[0] != latent.shape[0]:
            return preds
        delta = self.heads.vision_delta(latent, order_enc, achieved, record=record)
        if delta is None:
            return preds
        p = preds[self.vision_key]
        delta = delta.reshape(p.shape).to(p.dtype)
        if mask is not None:
            delta = delta * mask.reshape(-1, *([1] * (delta.dim() - 1))).to(delta.dtype)
        out = dict(preds)
        out[self.vision_key] = p + delta
        return out


def gaze_copy_of(prediction_heads) -> Optional[GazeCopy]:
    """The gaze copy attached to these prediction heads (or their compiled original), or None."""
    for obj in (prediction_heads, getattr(prediction_heads, "_orig_mod", None)):
        gc = getattr(obj, "_efference_gaze_copy", None) if obj is not None else None
        if gc is not None:
            return gc
    return None


def attach_gaze_copy(prediction_heads, heads: EfferenceHeads) -> GazeCopy:
    """Attach the gaze copy to the prediction heads without registering it as a submodule."""
    gc = GazeCopy(heads)
    target = getattr(prediction_heads, "_orig_mod", prediction_heads)
    object.__setattr__(target, "_efference_gaze_copy", gc)
    return gc


def scaled_error(pred: torch.Tensor, target: torch.Tensor, scale: float,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The heads' loss on one group: the error in units of the group's measured spread,
    squared up to one spread and linear beyond it (Huber, beta = 1 spread; half the square
    below, |e| - 0.5 above - loss_manager's excursion charge is the precedent).

    pred is in scale units, target in lived units (cm or degrees). mask: optional boolean over
    the handle axis ([n]) selecting which handles count.

    WHY NOT A PLAIN SQUARE (measured 2026-09-22, b1 scratchpad b1_scales.npz): the first
    transition after a world reset moved every handle 27.7 cm in 50 ms, 61 measured spreads per
    axis; squared, that one step outweighs about 3,700 ordinary steps and takes the whole clipped
    gradient. Every transition after the settling moved 0.3-2.5 cm. Linear beyond one spread, the
    jump counts about 61 times an ordinary step instead."""
    diff = pred.float() - target.float() / float(scale)
    if mask is not None:
        diff = diff[:, mask]
    return torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), beta=1.0)


class LessonTimer:
    """The efference lesson's own time per step (the machine block's efference_ms; review
    MINOR 6). On the CPU every step is timed; on the graphics card one step in
    EFFERENCE_TIMING_EVERY_CUDA, with the card synchronised before and after (the kernels are
    asynchronous, so an unsynchronised clock would time only their launch). What it times: the
    lesson's forward passes and gauges; its share of the backward pass cannot be separated from
    the rest of the step's backward and stays inside model_ms."""

    def __init__(self):
        self.t0 = None
        self.cuda = False

    def start(self, device, step: int) -> None:
        self.cuda = getattr(device, "type", str(device)) == "cuda"
        if self.cuda and int(step) % EFFERENCE_TIMING_EVERY_CUDA != 0:
            self.t0 = None
            return
        if self.cuda:
            torch.cuda.synchronize(device)
        self.t0 = time.perf_counter()

    def stop(self, device, acc: list) -> None:
        """acc: [sum_ms, n] (mutated)."""
        if self.t0 is None:
            return
        if self.cuda:
            torch.cuda.synchronize(device)
        acc[0] += 1000.0 * (time.perf_counter() - self.t0)
        acc[1] += 1
        self.t0 = None


# =================================================================================== the model side
# The functions below are LemLite's efference lesson, window report and checkpoint rules. They live
# here (lane-b1-efference, 2026-09-23) so mimo_bridge/lem_model.py keeps only thin calls; `model` is
# the LemLite instance.
EFFERENCE_SEED_OFFSET = 3_000_003     # the heads' own random stream: run seed + this (contact head
                                      # + 4_000_004, synergy + 5_000_005)


def make_heads(d_model: int, n_handles: int, gaze_idx: int, head_idx: int, n_mod: int,
               vision_idx: Optional[int], seed: int,
               eye_row_convention: str = "legacy") -> EfferenceHeads:
    """The heads at their seeded initial weights, built inside a forked random stream: the run's own
    stream is left exactly where a build without them leaves it, and a fresh start at load time
    rebuilds the very same initial weights. eye_row_convention: the model's (a setting, no weight)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + EFFERENCE_SEED_OFFSET)
        return EfferenceHeads(d_model, n_handles, gaze_idx, head_idx, n_mod, vision_idx=vision_idx,
                              eye_row_convention=eye_row_convention)


def stamp(model) -> dict:
    """The receipt's record of what the heads were built and trained under (a load compares it)."""
    return {"version": EFFERENCE_VERSION, "scales": dict(EFFERENCE_SCALES),
            "weight": float(model.efference_weight),
            "tenure_steps": int(model.efference_tenure_steps),
            "horizon_weights": dict(HORIZON_WEIGHTS),
            "asked_change_scales": {"pos_cm": ASK_POS_CM, "rot_deg": ASK_ROT_DEG},
            "centre_horizon_steps": CENTRE_HORIZON_STEPS,
            "pools": {"kind": "bounded cosine attention with a zero-initialised mean path",
                      "spread_floor": POOL_SPREAD_FLOOR,
                      "bound": round(float(model.efference.body_pool.bound), 6)},
            "gaze_copy_into_vision": model.efference.vision_idx is not None,
            "gaze_head_inputs": (["pooled_latent", "eye_command", "head_orientation",
                                  "eye_orientation"]
                                 + (["head_accelerometer", "head_gyro"]
                                    if getattr(model, "vestibular_sense", False) else [])),
            # False from v2 on: the vestigial GazeHead is never trained. A checkpoint whose GazeHead
            # the first version DID retrain keeps True through every later save (load_checkpoint),
            # and lem.training.checkpoint refuses to hand such a GazeHead to a caller that reads it
            # as the absolute direction of the eyes.
            "gaze_head_repurposed": False}


def head_motion_of(model, raw: Optional[dict]) -> Optional[torch.Tensor]:
    """[B, 6] head accelerometer and gyro from a cached raw frame (None when the vestibular sense
    is off or the frame has no vector)."""
    if not getattr(model, "vestibular_sense", False) or not isinstance(raw, dict):
        return None
    return head_motion_from_vestibular(raw.get("vestibular"))


def added_readings() -> bool:
    """mimo_bridge.prediction_audit.ADDED_READINGS: whether the 2026-10-02 readings (here the side heads'
    size-matched control) are computed; True in every run."""
    from mimo_bridge import prediction_audit
    return bool(prediction_audit.ADDED_READINGS)


def size_matched_order(model, exec_t: torch.Tensor, ach0: torch.Tensor, B: int):
    """(command in ENCODER units, eye row kept as it stands) for the side heads' size-matched control: another
    body's command size-matched to this body (size_matched_other_order), the eyes' row rebuilt from the eyes'
    angles at the state the command starts from (model._eff_eye_prev, the previous model step's, which the lesson
    refreshes only at its end). The second value is True when those angles were missing and the eyes' row stayed
    the other body's as it stands."""
    from mimo_bridge.lem_model import command_for_encoder
    eye = getattr(model, "_eff_eye_prev", None)
    have = torch.is_tensor(eye) and tuple(eye.shape) == (int(B), 2)
    sm = size_matched_other_order(exec_t.float(), ach0, model.GAZE_IDX, eye_deg=eye if have else None,
                                  convention=getattr(model, "eye_row_convention", "legacy"))
    return command_for_encoder(sm), not have


def apply_gaze_copy(prediction_heads, preds, latent, order_enc, achieved, mask=None, record=False):
    """preds with the gaze copy added to vision (unchanged when the build has no gaze copy)."""
    gc = gaze_copy_of(prediction_heads)
    if gc is None:
        return preds
    return gc(preds, latent, order_enc, achieved, mask=mask, record=record)


def _win_add(model, key: str, value) -> None:
    s = model.efference_window.setdefault(key, [0.0, 0])
    s[0] += float(value)
    s[1] += 1


@torch.no_grad()
def _gauge_vision(model, before, after, target, shuffled, size_matched=None) -> None:
    """The gaze copy's own reading: vision's next-moment error without the copy, with it, with ANOTHER
    body's absolute eye command through it, and with another body's eye command size-matched to this
    body's eyes (size_matched_other_order) through it (mean squared error per token dimension). Kept on
    the device until the lesson's one read of the step (efference_pending)."""
    t = target.detach().float()
    g = {"vision_err_without_copy": (before.detach().float() - t).pow(2).mean(),
         "vision_err_with_copy": (after.detach().float() - t).pow(2).mean()}
    if shuffled is not None:
        g["vision_err_copy_shuffled"] = (shuffled.detach().float() - t).pow(2).mean()
    if size_matched is not None:
        g["vision_err_copy_size_matched"] = (size_matched.detach().float() - t).pow(2).mean()
    model.efference_pending = g


def lesson_gaze_copy(model, preds, latent, exec_t, target_vision, B):
    """The delayed next-frame lesson's predictions with the gaze copy added (lem_model.step, the
    hoisted S_t-1 + A_t-1 forward): the order A_t-1, the pose of S_t-1 (model._nf_ach, not yet
    refreshed). A lived moment: the first one ever sets the latent's centre. Also fills the copy's
    window reading when the vision target is real (target_vision None otherwise)."""
    heads = model.efference
    gc = gaze_copy_of(model.prediction_heads)
    ach0 = getattr(model, "_nf_ach", None)
    if (gc is None or heads is None or latent is None or exec_t is None or ach0 is None
            or ach0.shape[0] != B or latent.shape[0] != B or exec_t.shape[0] != B):
        return preds
    from mimo_bridge.lem_model import command_for_encoder
    order = command_for_encoder(exec_t.float())
    heads.ensure_centres(latent, order)
    out = gc(preds, latent, order, ach0, record=True)
    if target_vision is not None and "vision" in preds and out is not preds:
        with torch.no_grad():
            sh = sm = None
            if B > 1:
                sh = gc(preds, latent.detach(), order.roll(1, dims=0), ach0)["vision"]
                if added_readings():
                    # the size-matched control (2026-10-02): a measurement pass the liveness repair does not watch
                    from lem.model.liveness import unwatched
                    sm_order, _kept = size_matched_order(model, exec_t, ach0, B)
                    with unwatched(model):
                        sm = gc(preds, latent.detach(), sm_order, ach0)["vision"]
            _gauge_vision(model, preds["vision"], out["vision"], target_vision, sh, sm)
    return out


def efference_lesson(model, latent, exec_t, order_id, handle_state, frames, step, B, train=True):
    """Grade the efference heads on what the body and the eyes really did (LemLite.step's loss
    section; see LemLite._efference_lesson for the arguments).

    Five gradings, each the error in units of its measured spread (EFFERENCE_SCALES), squared up
    to one spread and linear beyond it (scaled_error), SUMMED with HORIZON_WEIGHTS:
      body, next moment   pose(t) - pose(t-1)          gradient reaches the core
      gaze, 50 ms         eye(t) - eye(t-1)            gradient reaches the core
      gaze, 100/150 ms    eye(t) - eye(t-2), eye(t) - eye(t-3), from the latents stored
                          one and two steps ago        heads only
      body, the tenure    pose where the order ended - pose where it began, graded the
                          step the next order appears  heads only
    THE EXPECTATION OF A TENURE (review MINOR 1) is the head's tenure output computed where the
    order began, with the weights of that moment, stored detached; the re-forward at the order's
    end serves only the loss. Returns {"efference_body": weighted loss, "efference_gaze": weighted
    loss}, either key absent on a step that graded nothing of it."""
    timer = LessonTimer()
    timer.start(model.device, step)
    try:
        return _efference_lesson(model, latent, exec_t, order_id, handle_state, frames, step, B,
                                 train)
    finally:
        timer.stop(model.device, model.efference_timing)


def _efference_lesson(model, latent, exec_t, order_id, handle_state, frames, step, B, train):
    from mimo_bridge.lem_model import command_for_encoder
    S = EFFERENCE_SCALES
    heads = model.efference
    eye_now = model._eye_angles_deg(frames, B)
    gi = model.GAZE_IDX
    H = len(model.handle_names)
    mask = torch.ones(H, dtype=torch.bool, device=model.device)
    mask[gi] = False
    terms: Dict[str, torch.Tensor] = {}
    gauges: Dict[str, torch.Tensor] = {}

    def dist(a, b):
        return (a.float() - b.float())[:, mask].norm(dim=-1).mean()

    def size(a):
        return a.float()[:, mask].norm(dim=-1).mean()

    ach0 = model._nf_ach
    lp_on = model.wellness_channels is not None and model.wellness_channels.lp is not None
    ok = (latent is not None and exec_t is not None and ach0 is not None
          and ach0.shape[0] == B and latent.shape[0] == B)
    if ok:
        order = command_for_encoder(exec_t.float())
        heads.ensure_centres(latent, order)
        hm = head_motion_of(model, model._nf_raw)
        # ---- body, the next moment (gradient into the core)
        pb = heads.body(latent, order, ach0, record=True)
        cm_t, deg_t = pose_change(ach0, handle_state)
        terms["body_next"] = 0.5 * (scaled_error(pb[..., 0:3], cm_t, S["pos_next_cm"], mask)
                                    + scaled_error(pb[..., 3:6], deg_t, S["rot_next_deg"], mask))
        pb_sh = None
        pb_sm = sm_order = None
        with torch.no_grad():
            lv = EfferenceHeads.to_lived(body=pb.detach())
            gauges["body_cm_err_next"] = dist(lv["body_next_cm"], cm_t)
            gauges["body_cm_do_nothing_next"] = size(cm_t)
            gauges["body_deg_err_next"] = dist(lv["body_next_deg"], deg_t)
            gauges["body_deg_do_nothing_next"] = size(deg_t)
            if lp_on:
                # PER BODY, for learning progress (DESIGN D7): the head's error and the do-nothing
                # error in centimetres, mean over the body handles
                model._eff_lp["efference_body"] = (
                    (lv["body_next_cm"].float() - cm_t.float())[:, mask].norm(dim=-1).mean(-1),
                    cm_t.float()[:, mask].norm(dim=-1).mean(-1))
            if B > 1:
                # ANOTHER BODY'S ORDER with THIS body's latent and pose: its asked change is taken
                # from this body's pose, so only the order differs
                pb_sh = heads.body(latent.detach(), order.roll(1, dims=0), ach0)
                ls = EfferenceHeads.to_lived(body=pb_sh)
                gauges["body_cm_shuffled_next"] = dist(ls["body_next_cm"], cm_t)
                gauges["body_deg_shuffled_next"] = dist(ls["body_next_deg"], deg_t)
                # THE SIZE-MATCHED CONTROL (2026-10-02, docs/gain_of_function_v4/agents/
                # swap_test_phases_2026-10-02.md): another body's order as a change from its own pose
                # applied to this body's pose (size_matched_other_order), so the asked moves and turns
                # are a real order's size; the absolute roll above stays for one segment beside it and
                # still feeds controllability's gate in force (its recorded-only shadow gate reads this
                # one). A measurement pass: the liveness repair does not watch it.
                if added_readings():
                    from lem.model.liveness import unwatched
                    sm_order, eye_kept = size_matched_order(model, exec_t, ach0, B)
                    with unwatched(model):
                        pb_sm = heads.body(latent.detach(), sm_order, ach0)
                    lm = EfferenceHeads.to_lived(body=pb_sm)
                    gauges["body_cm_size_matched_next"] = dist(lm["body_next_cm"], cm_t)
                    gauges["body_deg_size_matched_next"] = dist(lm["body_next_deg"], deg_t)
                    gauges["size_matched_eye_as_stands"] = torch.tensor(float(eye_kept), device=model.device)
        # ---- gaze, 50 ms (gradient into the core)
        if (eye_now is not None and model._eff_eye_prev is not None
                and model._eff_eye_prev.shape[0] == B):
            pg = heads.gaze(latent, order, ach0, hm, record=True)
            t1 = eye_now - model._eff_eye_prev
            terms["gaze_1"] = scaled_error(pg[:, 0], t1, S["gaze_deg_1"])
            with torch.no_grad():
                lg = EfferenceHeads.to_lived(gaze=pg.detach())["gaze_deg"]
                gauges["gaze_deg_err_1"] = (lg[:, 0] - t1).norm(dim=-1).mean()
                gauges["gaze_deg_do_nothing_1"] = t1.norm(dim=-1).mean()
                if lp_on:
                    model._eff_lp["efference_gaze"] = ((lg[:, 0] - t1).norm(dim=-1),
                                                       t1.norm(dim=-1))
                if B > 1:
                    sg = EfferenceHeads.to_lived(gaze=heads.gaze(
                        latent.detach(), order.roll(1, dims=0), ach0, hm))["gaze_deg"]
                    gauges["gaze_deg_shuffled_1"] = (sg[:, 0] - t1).norm(dim=-1).mean()
                    if sm_order is not None:
                        # the size-matched control: the gaze head reads only the eyes' row, rebuilt
                        # as the other body's asked eye turn applied to this body's eyes
                        from lem.model.liveness import unwatched
                        with unwatched(model):
                            smg = heads.gaze(latent.detach(), sm_order, ach0, hm)
                        smg = EfferenceHeads.to_lived(gaze=smg)["gaze_deg"]
                        gauges["gaze_deg_size_matched_1"] = (smg[:, 0] - t1).norm(dim=-1).mean()
        # ---- gaze, 100 and 150 ms, from the stored latents (heads only)
        if eye_now is not None:
            for r in model._eff_hist:
                h = int(step) - int(r["step"])
                if h not in (2, 3) or r.get("eye") is None or r["latent"].shape[0] != B:
                    continue
                ph = heads.gaze(r["latent"], r["order"], r["ach"], r.get("hm"))[:, h - 1]
                th = eye_now - r["eye"]
                terms["gaze_%d" % h] = scaled_error(ph, th, S["gaze_deg_%d" % h])
                with torch.no_grad():
                    k = torch.tensor([S["gaze_deg_%d" % h]], device=model.device)
                    gauges["gaze_deg_err_%d" % h] = (ph.detach().float() * k - th).norm(dim=-1).mean()
                    gauges["gaze_deg_do_nothing_%d" % h] = th.norm(dim=-1).mean()
                    if B > 1:
                        sh = heads.gaze(r["latent"], r["order"].roll(1, dims=0), r["ach"],
                                        r.get("hm"))[:, h - 1]
                        gauges["gaze_deg_shuffled_%d" % h] = (sh.float() * k - th).norm(dim=-1).mean()
                        osm = r.get("order_size_matched")
                        if torch.is_tensor(osm) and osm.shape[0] == B:
                            # the size-matched order formed where this record began (its eyes' angles)
                            from lem.model.liveness import unwatched
                            with unwatched(model):
                                shm = heads.gaze(r["latent"], osm, r["ach"], r.get("hm"))[:, h - 1]
                            gauges["gaze_deg_size_matched_%d" % h] = (shm.float() * k - th).norm(dim=-1).mean()
        # ---- body, the order's tenure (heads only), graded when the next order appears
        if order_id is not None:
            changed = (model._eff_prev_order_id is None
                       or int(order_id) != model._eff_prev_order_id)
        else:
            limbs = torch.cat([exec_t[:, :gi], exec_t[:, gi + 1:]], dim=1)
            prev = model._eff_prev_order
            changed = (prev is None or prev.shape != exec_t.shape or not torch.equal(
                torch.cat([prev[:, :gi], prev[:, gi + 1:]], dim=1), limbs))
        if changed:
            p = model._eff_tenure
            if (p is not None and p["ach"].shape[0] == B
                    and (int(step) - 1) - int(p["step"]) == model.efference_tenure_steps):
                pt = heads.body(p["latent"], p["order"], p["ach"])
                cm_T, deg_T = pose_change(p["ach"], ach0)
                terms["body_tenure"] = 0.5 * (
                    scaled_error(pt[..., 6:9], cm_T, S["pos_tenure_cm"], mask)
                    + scaled_error(pt[..., 9:12], deg_T, S["rot_tenure_deg"], mask))
                with torch.no_grad():
                    # PER BODY, for controllability (DESIGN D6): what the order was expected to do
                    # WHEN IT WAS SENT - the tenure outputs stored where it began, from that
                    # moment's weights (review MINOR 1) - beside what the body did over its tenure.
                    e = p["expected"].float()
                    exp_cm = e[..., 0:3] * S["pos_tenure_cm"]
                    exp_deg = e[..., 3:6] * S["rot_tenure_deg"]
                    rec = {"step": int(step), "order_began": int(p["step"]),
                           "expected_cm": exp_cm, "expected_deg": exp_deg,
                           "achieved_cm": cm_T.float(), "achieved_deg": deg_T.float(),
                           "order": p["order"],
                           "expected_formed": "where the order began, before any of its outcome"}
                    gauges["body_cm_err_latch"] = dist(exp_cm, cm_T)
                    gauges["body_cm_do_nothing_latch"] = size(cm_T)
                    gauges["body_deg_err_latch"] = dist(exp_deg, deg_T)
                    gauges["body_deg_do_nothing_latch"] = size(deg_T)
                    es = p.get("expected_shuffled")
                    if es is not None and es.shape[0] == B:
                        es = es.float()
                        sh_cm = es[..., 0:3] * S["pos_tenure_cm"]
                        sh_deg = es[..., 3:6] * S["rot_tenure_deg"]
                        # THE SHUFFLED-ORDER EXPECTATION: this body's latent and pose with ANOTHER
                        # body's order, formed at the same moment (controllability's control)
                        rec["expected_shuffled_cm"], rec["expected_shuffled_deg"] = sh_cm, sh_deg
                        gauges["body_cm_shuffled_latch"] = dist(sh_cm, cm_T)
                        gauges["body_deg_shuffled_latch"] = dist(sh_deg, deg_T)
                    esm = p.get("expected_size_matched")
                    if torch.is_tensor(esm) and esm.shape[0] == B:
                        # THE SIZE-MATCHED EXPECTATION (2026-10-02): this body's latent and pose with another
                        # body's order size-matched to it, formed where the order began. Recorded beside the
                        # shuffled one; controllability's gate in force still reads expected_shuffled_*
                        # (unchanged), and its recorded-only shadow gate reads this one
                        # (wellness_channels.SHADOW_CONTROL_KIND, 2026-10-03)
                        esm = esm.float()
                        sm_cm = esm[..., 0:3] * S["pos_tenure_cm"]
                        sm_deg = esm[..., 3:6] * S["rot_tenure_deg"]
                        rec["expected_size_matched_cm"], rec["expected_size_matched_deg"] = sm_cm, sm_deg
                        gauges["body_cm_size_matched_latch"] = dist(sm_cm, cm_T)
                        gauges["body_deg_size_matched_latch"] = dist(sm_deg, deg_T)
                    model.last_efference_tenure = rec
            model._eff_tenure = {"latent": latent.detach(), "order": order.detach(),
                                 "ach": ach0.detach().clone(), "step": int(step) - 1,
                                 "expected": pb[..., 6:12].detach(),
                                 "expected_shuffled": (None if pb_sh is None
                                                       else pb_sh[..., 6:12].detach()),
                                 "expected_size_matched": (None if pb_sm is None
                                                           else pb_sm[..., 6:12].detach())}
        model._eff_prev_order_id = None if order_id is None else int(order_id)
        model._eff_prev_order = exec_t.detach().clone()
        # this state's latent, for the 100 and 150 ms gradings of the next two steps
        model._eff_hist.append({"latent": latent.detach(), "order": order.detach(),
                                "ach": ach0.detach().clone(),
                                "hm": None if hm is None else hm.detach(),
                                "eye": model._eff_eye_prev, "step": int(step) - 1,
                                "order_size_matched": None if sm_order is None else sm_order.detach()})
        if train:
            # a lived training moment: the running means move (after they were used, so no
            # prediction saw its own moment in its centre)
            heads.update_centres(latent, order)
    model._eff_eye_prev = None if eye_now is None else eye_now.detach()
    out = {}
    w = model.efference_weight
    HW = HORIZON_WEIGHTS
    body = [HW[k] * terms[k] for k in ("body_next", "body_tenure") if k in terms]
    gaze = [HW[k] * terms[k] for k in ("gaze_1", "gaze_2", "gaze_3") if k in terms]
    zero = torch.zeros((), device=model.device)
    if body:
        out["efference_body"] = w * torch.stack(body).sum()
        gauges["loss_body"] = out["efference_body"].detach()
        # the part whose gradient reaches the core (the next-moment term); the tenure term trains
        # only the head (review MINOR 4)
        gauges["loss_body_core"] = ((w * HW["body_next"] * terms["body_next"]).detach()
                                    if "body_next" in terms else zero)
    if gaze:
        out["efference_gaze"] = w * torch.stack(gaze).sum()
        gauges["loss_gaze"] = out["efference_gaze"].detach()
        gauges["loss_gaze_core"] = ((w * HW["gaze_1"] * terms["gaze_1"]).detach()
                                    if "gaze_1" in terms else zero)
    # the gaze copy's reading of this step (lesson_gaze_copy), read together with the rest
    pend = getattr(model, "efference_pending", None)
    if pend:
        gauges.update(pend)
        model.efference_pending = {}
    if gauges:
        keys = list(gauges)
        vals = torch.stack([gauges[k].float().reshape(()) for k in keys]).tolist()   # one read
        for k, v in zip(keys, vals):
            _win_add(model, k, v)
    return out


def window_summary(model) -> dict:
    """The `efference` block of the stats record (DESIGN.md section 3), and a fresh window.

    Every error is a mean distance, in centimetres or degrees, between the change the head
    predicted and the change that happened. Beside each: do_nothing = the mean size of the change
    itself (the error of predicting "nothing moves": the floor a working head must beat),
    shuffled = the error when another body's order is handed to the head with this body's latent
    and pose (the direct order input's null: equal to err means the head ignores the order it is
    given). next = one moment (50 ms); latch = the order's whole tenure, judged on the expectation
    formed where the order began; gaze lists are 50, 100, 150 ms. loss_* are the weighted loss
    contributions (window means); loss_body_core / loss_gaze_core are the parts whose gradient
    reaches the core (the next-moment body term and the 50 ms gaze term), the rest trains only the
    heads. gaze_copy: vision's next-moment error without the gaze copy, with it, and with another
    body's eye command through it (mean squared error per token dimension), and share_removed =
    1 - with / without (0 = the copy removes nothing of vision's error, 1 = all of it),
    share_removed_shuffled the same with another body's eye command (the control: share_removed
    above it means the copy uses THIS body's command). liveness / pool_spread /
    input_shared_share: DESIGN section 10 (lem.model.efference.EfferenceHeads.window_liveness).

    THE TWO CONTROLS (2026-10-02, docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md):
    shuffled = another body's ABSOLUTE order (its targets as they stand, chosen for its own pose); a head
    whose output grows with the asked change (target minus pose) errs more with it without using the
    order's effect. size_matched = another body's order as a change from its own pose applied to this
    body's pose, the eyes' row as its asked eye turn applied to this body's eyes (size_matched_other_order):
    body_cm_size_matched_next/latch, body_deg_size_matched_next/latch, gaze_deg_size_matched (50, 100, 150
    ms) and gaze_copy.vision_err_copy_size_matched / share_removed_size_matched, on the same graded model
    steps as the shuffled keys. size_matched_eye_as_stands = the share of graded model steps whose eyes'
    row kept the other body's row as it stands (the eyes' angles at the start were missing): 0 in a run
    whose frames carry eye angles. The shuffled keys stay beside the size-matched ones for one segment, the
    first on this code, so the change can be read across; controllability's gate in force still reads the
    shuffled expectation, and its recorded-only shadow gate (2026-10-03) the size-matched one. `controls`
    in the block repeats these two definitions."""
    W = model.efference_window

    def m(k, nd=4):
        s = W.get(k)
        return None if not s or s[1] <= 0 else round(s[0] / s[1], nd)

    out = {k: m(k) for k in (
        "body_cm_err_next", "body_cm_do_nothing_next", "body_cm_shuffled_next",
        "body_cm_err_latch", "body_cm_do_nothing_latch", "body_cm_shuffled_latch",
        "body_deg_err_latch", "body_deg_do_nothing_latch", "body_deg_shuffled_latch",
        "body_deg_err_next", "body_deg_do_nothing_next", "body_deg_shuffled_next",
        "loss_body", "loss_gaze", "loss_body_core", "loss_gaze_core",
        "body_cm_size_matched_next", "body_deg_size_matched_next",
        "body_cm_size_matched_latch", "body_deg_size_matched_latch", "size_matched_eye_as_stands")}
    for name in ("err", "do_nothing", "shuffled", "size_matched"):
        out["gaze_deg_" + name] = [m("gaze_deg_%s_%d" % (name, h)) for h in GAZE_HORIZONS]
    out["events_next"] = int((W.get("body_cm_err_next") or [0, 0])[1])
    out["events_latch"] = int((W.get("body_cm_err_latch") or [0, 0])[1])
    wo, wi, ws, wm = (m(k, 6) for k in ("vision_err_without_copy", "vision_err_with_copy",
                                        "vision_err_copy_shuffled", "vision_err_copy_size_matched"))
    out["gaze_copy"] = {
        "vision_err_without_copy": wo, "vision_err_with_copy": wi, "vision_err_copy_shuffled": ws,
        "share_removed": (None if wo is None or wi is None or wo <= 0 else round(1.0 - wi / wo, 4)),
        "share_removed_shuffled": (None if wo is None or ws is None or wo <= 0
                                   else round(1.0 - ws / wo, 4)),
        "events": int((W.get("vision_err_with_copy") or [0, 0])[1]),
        "vision_err_copy_size_matched": wm,
        "share_removed_size_matched": (None if wo is None or wm is None or wo <= 0
                                       else round(1.0 - wm / wo, 4))}
    out.update(model.efference.window_liveness())
    out["version"] = EFFERENCE_VERSION
    out["controls"] = dict(EFFERENCE_CONTROLS)
    model.efference_window = {}
    return out


# what the efference block's two controls are (window_summary's `controls`)
EFFERENCE_CONTROLS = {
    "shuffled": ("another body's ABSOLUTE command with this body's latent and pose: its targets as they stand, "
                 "chosen for the other body's pose (the original control; kept beside size_matched for one "
                 "segment, the first on this code, then retired; controllability's evidence gate in force "
                 "still reads it)"),
    "size_matched": ("another body's command as a change from its own pose applied to this body's pose, the eyes' "
                     "row as its asked eye turn applied to this body's eyes (lem/model/efference.py "
                     "size_matched_other_order), with this body's latent and pose (controllability's "
                     "recorded-only shadow gate reads it)"),
}


def lesson_ms(model) -> Optional[float]:
    """The machine block's efference_ms: mean milliseconds of the lesson's forward passes and
    gauges per timed step this window (LessonTimer), and a fresh window; None when none was timed."""
    t = model.efference_timing
    val = None if t[1] <= 0 else round(t[0] / t[1], 2)
    model.efference_timing = [0.0, 0]
    return val


# ------------------------------------------------------------------------------ checkpoints
def _output_rows(heads: EfferenceHeads, ks: Dict[str, float]):
    """(layer, per-output-row factor) for the two output layers, from per-divisor factors."""
    H = heads.n_handles
    rows_body = torch.ones(H, 12)
    for c0, key in ((0, "pos_next_cm"), (3, "rot_next_deg"), (6, "pos_tenure_cm"),
                    (9, "rot_tenure_deg")):
        rows_body[:, c0:c0 + 3] = ks[key]
    rows_gaze = torch.tensor([ks["gaze_deg_%d" % h] for h in GAZE_HORIZONS for _ in (0, 1)])
    return [(heads.body_mlp[-1], rows_body.reshape(-1)), (heads.gaze_out, rows_gaze)]


def migrate_scales(model, saved_scales: dict, load_optimizer: bool = True) -> bool:
    """Carry the heads across a change of their divisors, exactly: each output row (weight and
    bias) times k = old / new, Adam's averages times k and k squared (the precedent is
    LemLite._migrate_touch_push_scale). True when something moved."""
    ks = {k: float(saved_scales.get(k, v)) / float(v) for k, v in EFFERENCE_SCALES.items()}
    if all(abs(k - 1.0) < 1e-9 for k in ks.values()):
        return False
    with torch.no_grad():
        for lin, k in _output_rows(model.efference, ks):
            k = k.to(lin.weight.device, lin.weight.dtype)
            for p, kk in ((lin.weight, k.view(-1, 1)), (lin.bias, k)):
                p.mul_(kk)
                st = model.optimizer.state.get(p) if load_optimizer else None
                if st:
                    if torch.is_tensor(st.get("exp_avg")):
                        st["exp_avg"].mul_(kk)
                    if torch.is_tensor(st.get("exp_avg_sq")):
                        st["exp_avg_sq"].mul_(kk * kk)
    return True


def reset_tenure_rows(model, load_optimizer: bool = True) -> int:
    """Put the body head's tenure outputs (outputs 6-11 of every handle in the last layer) back to
    their seeded initial weights and drop their Adam averages. Used when a checkpoint's heads were
    trained on orders of another length (review MINOR 3): what those rows meant is not what they
    would mean now. Returns the number of rows reset."""
    heads = model.efference
    fresh = make_heads(heads.d_model, heads.n_handles, heads.gaze_idx, heads.head_idx, heads.n_mod,
                       heads.vision_idx, model.efference_seed)
    last, f_last = heads.body_mlp[-1], fresh.body_mlp[-1]
    rows = torch.tensor([h * 12 + c for h in range(heads.n_handles) for c in range(6, 12)],
                        device=last.weight.device)
    with torch.no_grad():
        last.weight[rows] = f_last.weight.to(last.weight)[rows]
        last.bias[rows] = f_last.bias.to(last.bias)[rows]
        if load_optimizer:
            for p in (last.weight, last.bias):
                st = model.optimizer.state.get(p)
                if st:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if torch.is_tensor(st.get(key)):
                            st[key][rows] = 0.0
    return int(rows.numel())


def reconcile(model, saved: Optional[dict], load_optimizer: bool = True, log=None) -> dict:
    """A same-version checkpoint's stamp against this build: divisors migrated exactly; a different
    order length (tenure_steps, --motor-hold) resets the tenure rows and says so; a different loss
    weight is reported (it changes how hard the heads are taught, not what they output)."""
    done = {}
    if not saved:
        return done
    if saved.get("scales") and migrate_scales(model, saved["scales"], load_optimizer):
        done["scales_migrated"] = dict(saved["scales"])
        if log is not None:
            log.warning("efference heads: this checkpoint's heads were trained on divisors %s; the "
                        "code now divides by %s, so each output row was multiplied by old/new and "
                        "its optimiser averages moved to match. Every prediction in cm and degrees "
                        "reads as it did.", saved["scales"], dict(EFFERENCE_SCALES))
    t_old = saved.get("tenure_steps")
    if t_old is not None and int(t_old) != int(model.efference_tenure_steps):
        n = reset_tenure_rows(model, load_optimizer)
        done["tenure_rows_reset"] = {"saved": int(t_old), "now": int(model.efference_tenure_steps),
                                     "rows": n}
        if log is not None:
            log.warning("efference heads: this checkpoint's body head learned the change over orders "
                        "of %d moments; this run holds each order %d moments (--motor-hold), so its "
                        "%d tenure output rows were put back to their initial weights with no Adam "
                        "history. The next-moment rows and the gaze head are kept.",
                        int(t_old), int(model.efference_tenure_steps), n)
    w_old = saved.get("weight")
    if w_old is not None and abs(float(w_old) - float(model.efference_weight)) > 1e-12:
        done["weight_changed"] = {"saved": float(w_old), "now": float(model.efference_weight)}
        if log is not None:
            log.warning("efference heads: trained at loss weight %g, now taught at %g "
                        "(--efference-weight); what the heads output is unchanged.",
                        float(w_old), float(model.efference_weight))
    return done


def fresh_start(model, path: str, load_optimizer: bool, log=None) -> dict:
    """A checkpoint whose heads are ANOTHER VERSION (efference_v1): the documented fresh start.

    WHY NOT A MIGRATION: version 1 read the raw latent, the absolute order only and the retrained
    GazeHead; version 2 reads the centred, normalised latent, the asked change and its own gaze
    layers. No v1 weight means the same thing in v2 (the first layers' inputs are other
    quantities), and on 2026-09-23 no lineage checkpoint carries v1 heads (the deploy that switches
    them on had not happened; only test runs trained them).

    WHAT HAPPENS: the heads go back to their seeded initial weights (the loader may have copied the
    v1 tensors whose shapes happened to match), their running means start cold, and every OTHER
    Adam moment is carried by position around the replaced block: in each optimizer group the saved
    parameters before and after the heads' block keep their moments, the v1 block's moments are
    dropped and the v2 parameters start with none. Raises when that accounting does not add up."""
    import os
    heads = model.efference
    fresh = make_heads(heads.d_model, heads.n_handles, heads.gaze_idx, heads.head_idx, heads.n_mod,
                       heads.vision_idx, model.efference_seed).to(heads.centre_mean.device)
    with torch.no_grad():
        heads.load_state_dict(fresh.state_dict())
        heads.centre_mean.zero_()
        heads.order_centre.zero_()
    heads.centre_weight = 0.0
    heads.order_centre_weight = 0.0
    report = {"fresh_start": EFFERENCE_VERSION, "adam_states_carried": None}
    opt_path = os.path.join(path, "optimizer.pt")
    if load_optimizer and os.path.isfile(opt_path):
        import safetensors.torch as st_
        saved = st_.load_file(os.path.join(path, "model.safetensors"))
        saved_opt = torch.load(opt_path, weights_only=False,
                               map_location=heads.centre_mean.device)["optimizer"]
        report.update(_carry_around_block(model, saved_opt, saved, "efference_heads."))
    if log is not None:
        log.warning("efference heads: this checkpoint's heads are an older version; they start "
                    "fresh as %s (their initial weights, no Adam history, the latent's centre from "
                    "the first lived moment). Every other weight loaded unchanged and %s saved Adam "
                    "moments were carried.", EFFERENCE_VERSION, report.get("adam_states_carried"))
    return report


def _carry_around_block(model, saved_opt: dict, saved: dict, prefix: str) -> dict:
    """Load the saved optimizer state with one module's parameter block replaced (fresh_start)."""
    opt = model.optimizer
    names = {}
    for comp, mod in _named_components(model).items():
        for n, p in mod.named_parameters():
            names[id(p)] = f"{comp}.{n}"
    block = [k for k in saved if k.startswith(prefix)]
    n_block = {0: sum(1 for k in block if saved[k].dim() >= 2),
               1: sum(1 for k in block if saved[k].dim() < 2)}
    cur = opt.state_dict()
    s_groups = saved_opt.get("param_groups") or []
    if len(s_groups) != len(cur["param_groups"]) or len(s_groups) != 2:
        raise ValueError("the checkpoint's optimizer has %d groups, this one %d (two expected: "
                         "decayed matrices, then everything 1-D)"
                         % (len(s_groups), len(cur["param_groups"])))
    st = saved_opt.get("state") or {}
    new_state, carried, dropped = {}, 0, 0
    for gi, (sg, cg, gobj) in enumerate(zip(s_groups, cur["param_groups"], opt.param_groups)):
        cls = []
        for p in gobj["params"]:
            n = names.get(id(p))
            if n is None:
                raise ValueError("a parameter in the optimizer has no qualified name")
            cls.append("block" if n.startswith(prefix) else ("saved" if n in saved else "new"))
        if "block" in cls:
            first = cls.index("block")
            last = len(cls) - 1 - cls[::-1].index("block")
            if any(c != "block" for c in cls[first:last + 1]):
                raise ValueError("the heads' parameters are not one block in optimizer group %d" % gi)
        else:
            first = last = len(cls)
        before = [i for i, c in enumerate(cls) if c == "saved" and i < first]
        after = [i for i, c in enumerate(cls) if c == "saved" and i > last]
        s_idx = list(sg["params"])
        nb = n_block[gi]
        if len(before) + nb + len(after) != len(s_idx):
            raise ValueError("optimizer group %d: the checkpoint lists %d parameters, but %d before "
                             "the heads + %d saved heads' tensors + %d after make %d"
                             % (gi, len(s_idx), len(before), nb, len(after),
                                len(before) + nb + len(after)))
        pairs = list(zip(before, s_idx[:len(before)])) + list(zip(after, s_idx[len(before) + nb:]))
        for ci, si in pairs:
            if si in st:
                new_state[cg["params"][ci]] = st[si]
                carried += 1
        dropped += sum(1 for si in s_idx[len(before):len(before) + nb] if si in st)
    groups = []
    for sg, cg in zip(s_groups, cur["param_groups"]):
        g = dict(sg)
        g["params"] = list(cg["params"])
        groups.append(g)
    opt.load_state_dict({"state": new_state, "param_groups": groups})
    return {"adam_states_saved": len(st), "adam_states_carried": carried,
            "adam_states_dropped_with_old_heads": dropped}


def _named_components(model) -> dict:
    comps = {"encoder_registry": model.encoder_registry, "forward_model": model.forward_model,
             "prediction_heads": model.prediction_heads, "motor_controller": model.motor_controller,
             "wellness": model.wellness}
    comps.update({k: v for k, v in model._side_modules().items() if v is not None})
    for k in ("gaze_head", "voice_head"):
        if getattr(model, k, None) is not None:
            comps[k] = getattr(model, k)
    return {k: getattr(v, "_orig_mod", v) for k, v in comps.items()}
