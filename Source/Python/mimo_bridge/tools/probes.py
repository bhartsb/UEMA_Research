# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Input-side probes: what the encoder can READ, measured on the token, not on the weights.

The dashboard used to report the motor encoder's command-column weight (0.083) as the
sign that the action pathway was alive. It was - and the model still could not see the
action, because 99.6% of the row's input energy sat in three position columns written in
centimetres (see command_for_encoder). Weight x input is what reaches the network, so a
pathway is measured here by moving the INPUT and watching the token:

    diff_action   token distance between two different real orders on the same body
    pos_block     token distance when the command's position block is zeroed
    null          token distance to the null action (identity rotation, zero strength)

On H33's slot-era encoder (cm block unscaled): diff_action 1.6, pos_block 14.6 - the
action was drowned. With commands normalised the same rows give diff_action ~8, and the
position block, now tiny, moves the token little. The ratio diff_action / pos_block is
the reading; below 0.2 the action is not readable, above 1 it is the dominant signal.

Pure functions over an EncoderRegistry so the dashboard, the tests and a hand probe get
the same number from the same rows.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def synthetic_motor_rows(seed: int = 0, n: int = 24, pairs: int = 6):
    """Realistic (achieved, executed-in-cm) rows: achieved as the tokenizer delivers it
    (pos/200, unit quaternion, radius, strength, is_active), executed as
    motor_for_action produces it (pos cm near the achieved position, unit quaternion,
    radius ~0.5, strength ~0.54). Returns a list of (achieved, cmd_a, cmd_b) triples where
    cmd_a and cmd_b are two different orders for the same body state."""
    from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, MOTOR_COMMAND_F
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(pairs):
        ach = torch.zeros(1, n, MOTOR_ACHIEVED_F)
        ach[..., 0:3] = torch.randn(1, n, 3, generator=g) * 0.07
        q = torch.randn(1, n, 4, generator=g); ach[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
        ach[..., 7] = 0.5; ach[..., 8] = 0.5; ach[..., 9] = 1.0
        cmds = []
        for _k in range(2):
            c = torch.zeros(1, n, MOTOR_COMMAND_F)
            c[..., 0:3] = ach[..., 0:3] * 200.0 + torch.randn(1, n, 3, generator=g) * 4.0
            q = torch.randn(1, n, 4, generator=g); c[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
            c[..., 7] = 0.5; c[..., 8] = 0.54
            cmds.append(c)
        out.append((ach, cmds[0], cmds[1]))
    return out


def _null_like(cmd):
    a = torch.zeros_like(cmd)
    a[..., 6] = 1.0
    return a


def action_readability(registry, rows=None, normalise=True) -> dict:
    """Token-distance reading of how much of the motor token is the ACTION.

    normalise=True runs the rows through command_for_encoder (the production path);
    normalise=False feeds raw centimetres, which reproduces the defect on any encoder and
    is how the detector itself is tested.
    """
    from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, command_for_encoder
    rows = rows if rows is not None else synthetic_motor_rows()
    reg = getattr(registry, "_orig_mod", registry)

    def tok(ach, cmd):
        c = command_for_encoder(cmd) if normalise else cmd
        x = torch.cat([ach[..., :MOTOR_ACHIEVED_F], c], dim=-1)
        with torch.no_grad():
            return reg.encode({"motor": x}, {"motor": "real"})["motor"].reshape(-1).float()

    d_act, d_pos, d_null = [], [], []
    for ach, a, b in rows:
        t = tok(ach, a)
        d_act.append(float((tok(ach, b) - t).norm()))
        z = a.clone(); z[..., 0:3] = 0.0
        d_pos.append(float((tok(ach, z) - t).norm()))
        d_null.append(float((tok(ach, _null_like(a)) - t).norm()))
    da, dp, dn = float(np.mean(d_act)), float(np.mean(d_pos)), float(np.mean(d_null))
    return {"diff_action": da, "pos_block": dp, "null": dn,
            "ratio": (da / dp) if dp > 1e-9 else float("inf"),
            "readable": bool(da >= 0.2 * dp)}


# ------------------------------------------------------------------ one body part at a time
# Why this exists (WORKLOG 2026-09-22 20:30, reviews/dashboards.md M9): action_readability above
# changes EVERY handle's order at once, with random full rotations, so it reads "READABLE" when a
# few handles move the token a lot - on ckpt_step_344400 each of the motor encoder's 4 pool heads
# read exactly ONE of 26 handles and 16 handles (the eyes among them) got zero weight, and the
# all-at-once test still read green. Here only ONE handle's ordered orientation is turned, by a
# realistic 20 degrees (the probes' limb order), everything else held, and the token's movement is
# reported per handle as a share of the token's own size.
TURN_DEG = 20.0


def _axis_angle_quat(axis: torch.Tensor, deg: float) -> torch.Tensor:
    """[..., 4] xyzw quaternion of a rotation by `deg` about the unit `axis` [..., 3]."""
    half = math.radians(deg) / 2.0
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.cat([axis * math.sin(half), torch.full_like(axis[..., :1], math.cos(half))], dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of xyzw quaternions (a applied after b in the world frame)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def per_handle_readability(registry, rows, turn_deg: float = TURN_DEG, axes: int = 3,
                           seed: int = 0, normalise: bool = True) -> dict:
    """How far ONE handle's `turn_deg` order turn moves the motor token, handle by handle.

    rows: a list of (achieved [1, n, 10], order [1, n, 9]) in the encoder's input units (the
    achieved half as the tokenizer delivers it; the order in execution units, cm, as the ledger
    keeps it). For every row and handle h, the order's quaternion of handle h alone is turned by
    `turn_deg` about `axes` random world axes; the token distance from the unturned order is
    averaged. Returns {"per_handle": [mean distance per handle], "token_size": mean token norm,
    "share": [per-handle distance / token size], "all_at_once": distance when every handle is
    turned together (the old test's form, for comparison)}."""
    from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, command_for_encoder
    reg = getattr(registry, "_orig_mod", registry)
    g = torch.Generator().manual_seed(int(seed))

    def tok(ach, cmd):
        c = command_for_encoder(cmd) if normalise else cmd
        x = torch.cat([ach[..., :MOTOR_ACHIEVED_F], c], dim=-1)
        with torch.no_grad():
            return reg.encode({"motor": x}, {"motor": "real"})["motor"].reshape(-1).double()

    n = int(rows[0][1].shape[1])
    per = np.zeros(n)
    sizes, whole = [], []
    count = 0
    for ach, cmd in rows:
        base = tok(ach, cmd)
        sizes.append(float(base.norm()))
        for _ in range(int(axes)):
            axis = torch.randn(n, 3, generator=g)
            rot = _axis_angle_quat(axis, turn_deg)
            turned_all = cmd.clone()
            turned_all[0, :, 3:7] = _quat_mul(rot, cmd[0, :, 3:7])
            whole.append(float((tok(ach, turned_all) - base).norm()))
            for h in range(n):
                one = cmd.clone()
                one[0, h, 3:7] = turned_all[0, h, 3:7]
                per[h] += float((tok(ach, one) - base).norm())
            count += 1
    per = per / max(1, count)
    size = float(np.mean(sizes))
    return {"per_handle": per.tolist(), "token_size": size,
            "share": (per / max(size, 1e-30)).tolist(), "all_at_once": float(np.mean(whole)),
            "rows": len(rows), "axes": int(axes), "turn_deg": float(turn_deg)}
