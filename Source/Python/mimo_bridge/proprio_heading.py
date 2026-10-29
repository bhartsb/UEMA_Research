# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A proprioceptive input that ignores the direction the body lies in the crib (--proprio-heading free, default off).

WHY (docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md). The proprioceptive input carries each bone's
WORLD quaternion and its world-axis position (relative to the left wrist) and velocity. Turning the whole body
about the vertical - lying the other way round in the crib - changes every one of those numbers while the posture
is the same. Measured on H84: each body sits in a crib corner at a heading that holds within 4-8 degrees for
minutes; the bodies at one heading share one token time-mean (cosine 0.99-1.00 between them, 0.63-0.85 across
headings), and the encoder goes deaf to the posture of the bodies at some headings (5 of 8 since the Sept 23
deploy). The heading is a nuisance the input hands the encoder as a large constant.

WHAT THIS DOES. Before the proprioception encoder reads a row (26 bones x 12 numbers), every bone's quaternion,
position and velocity are turned about the world vertical by minus the body's heading psi:
  - psi = the lower trunk's rotation about the vertical: its twist about world z, 2 atan2(q_z, q_w) of its world
    quaternion (x, y, z, w). A rotation of the whole body by alpha about the vertical adds exactly alpha to psi,
    so the turned row is the same for every heading (tested to float32 rounding). Gravity-relative orientation is
    kept: lying on the back, the side or the front, and every tilt, read as before.
  - the gaze row's quaternion is the eye relative to the head already (proprio_adapter.ProprioAdapter.read) and
    is left; its position and velocity are turned like every row's;
  - fatigue and the row index are left.
The twist is undefined only for the lower trunk turned half a revolution about a horizontal axis (upside down);
there psi is 0. On H84's rings the twist's two numbers have a size of 0.53-0.70 for every body (1 = upright, 0 =
upside down), far from it.

WHERE. A forward pre-hook on the proprioception encoder module, so every path that encodes this sense reads the
same input: the live step, the delayed next-moment lesson's re-encoding, the tenure matching, the paired audit,
the within-body lesson's window, and every probe that calls the encoder. Nothing else reads the change: the
movement drive, the body schema and the replay's stored poses keep the world-frame numbers.

AT THE SWITCH. This changes what the encoder reads, so the token changes for every body whose heading is not 0
and the core, the memory and the goals see a different proprioception token; the load logs the switch and stamps
the receipt. A checkpoint trained with the input turned is refused by a build without it (its encoder would read
world-frame rows it never saw). The cost is measured on saved checkpoints (proprio_token_measurements.py heading).

With the flag off LemLite imports nothing from here, installs no hook and stamps nothing: today's model.
"""
from __future__ import annotations

import math

import torch

VERSION = "proprio_heading_v1"
MODES = ("off", "free")
N_FIELDS = 12
TRUNK_ROW = 24          # body_layout.names_for("trunk_v1"): "lower_trunk"
GAZE_ROW = 23           # "gaze": its quaternion is head-relative


def heading(q: torch.Tensor) -> torch.Tensor:
    """The twist about world z of quaternions q [..., 4] (x, y, z, w), in radians (0 where it is undefined)."""
    return 2.0 * torch.atan2(q[..., 2], q[..., 3])


def turn_quaternion(q: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """(rotation about world z by -psi) x q, for q [..., 4] (x, y, z, w) and psi [...] broadcast over q's rows."""
    c, s = torch.cos(-psi / 2.0), torch.sin(-psi / 2.0)
    x, y, z, w = q.unbind(-1)
    # (0, 0, s, c) x (x, y, z, w), Hamilton product in (x, y, z, w) order
    return torch.stack([c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z], dim=-1)


def turn_vector(v: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """v [..., 3] turned about world z by -psi."""
    c, s = torch.cos(psi), torch.sin(psi)
    x, y, z = v.unbind(-1)
    return torch.stack([c * x + s * y, -s * x + c * y, z], dim=-1)


def heading_free(x: torch.Tensor) -> torch.Tensor:
    """The row [..., n_bones x 12] with every bone turned by minus the lower trunk's heading (a new tensor)."""
    F = x.shape[-1]
    if F % N_FIELDS or F // N_FIELDS <= TRUNK_ROW:
        raise ValueError("--proprio-heading free needs the trunk_v1 body's 26-row input, not %d numbers" % F)
    r = x.reshape(*x.shape[:-1], F // N_FIELDS, N_FIELDS)
    psi = heading(r[..., TRUNK_ROW, 0:4]).unsqueeze(-1)                  # [..., 1]: one heading per row set
    q = turn_quaternion(r[..., 0:4], psi)
    keep_gaze = torch.zeros(r.shape[-2], dtype=torch.bool, device=x.device)
    keep_gaze[GAZE_ROW] = True
    q = torch.where(keep_gaze.unsqueeze(-1), r[..., 0:4], q)
    out = torch.cat([q, turn_vector(r[..., 4:7], psi), turn_vector(r[..., 7:10], psi), r[..., 10:]], dim=-1)
    return out.reshape(x.shape)


def install(encoder: torch.nn.Module):
    """The forward pre-hook on the proprioception encoder; returns its handle (keep it on a PUBLIC attribute: the
    episode companion serialises every _* attribute and cannot serialise a hook handle)."""
    def _hook(_module, args):
        if not args or not torch.is_tensor(args[0]):
            return None
        return (heading_free(args[0]),) + tuple(args[1:])
    return encoder.register_forward_pre_hook(_hook)


def stamp() -> dict:
    return dict(version=VERSION, mode="free", reference="lower_trunk twist about world z", turned="quaternion, "
                "position, velocity of every row (the gaze row's quaternion is head-relative and left)")


def switch_message(path: str) -> str:
    return ("proprioception heading: this build turns every proprioceptive row by minus the body's heading before "
            "the encoder reads it, and %s was trained on world-frame rows: the proprioception token changes now "
            "for every body whose heading is not 0 and the encoder, the core and the memory relearn it "
            "(docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md, the cost measured there)" % path)
