# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The asked change as an explicit input of the motor encoder (--motor-asked-change, default off).

WHAT WAS MEASURED (docs/gain_of_function_v4/agents/motor_encoder_asked_change_2026-10-03.md, section 1; checkpoint
455,800 of H84, 2,000 recorded encoder inputs; registry rows motor_encoder_asked_change_stages_455800 and
motor_encoder_asked_change_mechanism_455800). The motor encoder receives, per handle, the achieved pose (10 numbers,
positions / 200 cm) and the executed command (9 numbers, positions / 200 cm). What the body's servo acts on is neither:
it is the CHANGE the command asks for, target minus pose, times the strength (mimo_bridge/action_decoder.py: the
rotation still needed, the wanted move, each joint's control multiplied by the handle's strength). The token does not
carry that change:
  * a linear read-out that has to carry over to another body finds the asked position change in each handle's own 19
    input numbers (R^2 0.9999) and in no later stage (each handle's per-handle network output -3.67, the pooled vector
    0.005, the token 0.0045); on the same bodies the per-handle output still holds it (0.75), the backbone output 0.19,
    the token 0.04;
  * the asked turn is not linearly available even at the input (-0.08 from the two absolute orientations, 0.998 from
    the command's orientation relative to the achieved one);
  * each of the 4 pooling heads puts 99.75 to 99.97 of every 100 parts of its weight on one handle (the logit gap to
    the second handle is 2,200 to 21,000), so the pooled vector reads at most 4 of the 26 handles, and which 4 changes
    from input to input;
  * the pooled vector is 9,972 per number, and the registry's output LayerNorm divides each row by its standard
    deviation over the 192 numbers (9,846 on average): a path added to the pooled vector moves the token only when its
    output is of that size.

WHAT THIS MODULE ADDS, with the flag on (off builds nothing: an old command builds the model it always built):

  INPUT.   The motor encoder's own input rows [B, n_handles, 19], at every call of the encoder (one forward hook on
           the encoder module, so the live pass, the context back-fill, the delayed lesson, the swap test, the hold
           loss, replay, the imagination chain, the planner's transition and the probes all pass through it).
  FEATURES, per handle (F_ASK = 10 numbers; features() below):
           0-2  asked position change: (command position - achieved position) x 200 cm, clamped to the reach clamp
                of 40 cm per axis, in units of POSITION_UNIT_CM;
           3-5  asked turn: the rotation vector that carries the achieved orientation to the commanded one, in units
                of TURN_UNIT_DEG (lem.model.efference.rotation_change_deg, the form the efference body head, the
                contact head and the size-matched swap already use);
           6    strength (0 to 1);   7  radius;   8  1 when the handle's row asks something (below), else 0;
           9    1 on the eyes' row, else 0.
           The eyes' row is in its own convention: its command is an eye-in-head joint target, not a world
           orientation, so it has no asked change against the achieved row; numbers 3 and 4 hold the command's
           azimuth and elevation (lem.model.efference.gaze_order_features, the numbers the eye servo is sent) in units
           of EYE_UNIT_RAD, numbers 0-2 and 5 are 0.
           WHEN A HANDLE'S ROW ASKS SOMETHING. A body handle: its command quaternion has length above 0.5 (a command
           is there), its achieved quaternion has length above 0.5 (a pose is there) and its strength is above 0: the
           limb servo skips a handle whose strength is 0 (mimo_bridge/action_decoder.py decode: `if strength <= 0.0:
           continue`), so such a command asks no change whatever its target. The eyes' row: its command quaternion
           has length above 0.5 (the eye servo is sent the two angles whatever the row's strength). Every feature of
           a handle that asks nothing is 0.
           So: a row without a command (the live pass writes zeros into the 9 command columns: the command is not
           chosen yet) gives a path output of exactly 0, and the live pass's token is the token of a build without
           this module, at every model step. The hold loss's do-nothing row (position 0, identity rotation, strength
           0) asks nothing of the 25 body handles; its eyes' row is the command "both angles 0".
  PATH.    z[b, i] = [a[b, i], net(a[b, i])]: the handle's 10 features and RANK more from a small network shared by
           the handles (10 -> HIDDEN -> RANK, GELU); u[b] = up(concat over the 26 handles of z[b, i]) / sqrt(N), N the
           count of numbers `up` reads (output_gain below): every handle enters through its own (10 + RANK) columns
           of `up`, so no attention weight decides which handle is read. `up` starts at zero (weight and bias).
  OUTPUT.  The encoder's output v[b] (the pooled vector, before the registry's LayerNorm) becomes
           v[b] + s[b] x u[b], s[b] = the standard deviation of v[b] over its 192 numbers (no gradient). The
           LayerNorm then gives LayerNorm(v + s u) = LayerNorm(LayerNorm(v) + u) up to its eps: one unit of u is one
           unit of the token before its final normalisation, whatever the pooled vector's size. With `up` at zero,
           s x u is exactly 0 and the token is bit for bit the token without the path.
           THE SUM IS FORMED IN FLOAT32 and handed to the LayerNorm in float32. Under the run's bf16 autocast the
           encoder's output is bf16: its numbers of size 4,096 to 16,383 lie on a grid of 32 to 64, which is 0.0045
           of the row's spread on average and at most 0.031 (MEASURED on the 2,000 recorded rows, mechanism.json
           bf16_step_over_row_sd). Added in bf16, a path output under half such a step (0.002 to 0.003 token units)
           would be rounded away and a larger one moved to that grid. The LayerNorm computes in float32 under autocast
           whatever it is given (the card's policy for layer_norm), so converting v to float32 changes no value: with
           a zero path the token keeps its bits (the load's switch check reads this under the run's autocast).

LIVENESS (DESIGN.md section 10). The path has no attention and no pooling: rule 3 holds by construction. The 10 raw
features reach `up` directly, so a hidden unit that stops responding cannot remove a handle's asked change from the
path. The hidden layer is an nn.Sequential Linear + GELU + Linear named motor_asked_change.net.0: the whole-model
monitor of lem/model/liveness.py reports its live share and varying share in the `liveness` stats block (the hourly
watcher's stop rule reads every layer listed there), and with --liveness-repair on the layer is one of the repair's
layers (LemLite passes LIVENESS_LAYER to LivenessRepair). window_summary() adds the path's own gauges, read on
the delayed lesson's own calls (arm_gauge), so reporting adds no pass through a layer the repair watches.

GRADIENT. THE PATH LEARNS ONLY FROM ROWS WHOSE POSE AND COMMAND BELONG TO THE SAME MODEL STEP: the delayed
next-model-step lesson (and the efference and contact heads' lessons on its latent), its re-study in replay, and the
hold loss (the pose at the hold's first model step with the hold's command). Lessons on the live pass's token
(VICReg, the motor controller's input) reach nothing here: that row has no command and the path's output for it is 0
times its weights. EVERY IMAGINED ROW (transition(), so the imagination chain, the planner's descent and plan chain,
the causal record and the probes' roll-outs; lem.model.encoder_registry.imagined_motor_token) passes through the path
with its weights detached (frozen() below): the row keeps its gradient, the weights get none. Why: an imagined row's
pose is the pose the roll-out started from, 1 and 2 model steps old at depths 2 and 3 of the imagination chain (a
handle moves 0.45 cm and turns 2.7 degrees per axis in one model step, lem/model/efference.py EFFERENCE_SCALES,
against an asked change of 11.9 cm and 20.2 degrees), and the chain's loss has no per-sense normalisation. The body
copy of Oct 1 (a network reading the same asked change, added to the touch and proprioception predictions) learned
from that route too: per update the chain pushed its touch part 2.1 times as hard as the lived lesson, and its touch
prediction got worse on lived model steps (planner_repair_2026-09-30.md sections 11 and 12); its repair was this
rule. For this path the same harm is not measured; the rule is taken from that measurement.

Every constant carries its measurement.
"""
from __future__ import annotations

import contextlib
import math
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ASKED_CHANGE_VERSION = "motor_asked_change_v1"
MODES = ("off", "on")
SIDE_NAME = "motor_asked_change"
# the hidden layer's name for lem/model/liveness.py (layer_parts resolves it on the model)
LIVENESS_LAYER = "motor_asked_change.net.0"

# THE MOTOR ROW'S LAYOUT (mimo_bridge/lem_model.py MOTOR_ACHIEVED_F, MOTOR_COMMAND_F): achieved pose 0-9 (position 3,
# orientation x y z w, radius, strength, active), command 10-18 (position 3, orientation x y z w, radius, strength)
ACHIEVED_F = 10
COMMAND_F = 9
ROW_F = ACHIEVED_F + COMMAND_F
POSITION_RANGE_CM = 200.0          # tokenizer.config.TokenizerConfig.position_range_cm (encoder_registry.py)

# THE UNITS, MEASURED on the 2,000 recorded transitions of checkpoint 455,800 (8 bodies x 250 model steps;
# D:/UEMA_Research/external/D/UEMA_scratch/motor_encoder_asked_change/2026-10-03/stages_455800/stages.json, input_statistics; Oct 3, 2026):
# the asked position change is 11.89 cm per axis (root mean square; 99 of 100 under 26.5 cm), the asked turn 20.16
# degrees per axis (99 of 100 under 60.2), the eye command 0.340 rad azimuth and 0.220 rad elevation. One unit of each
# feature is that size rounded, so every feature the path reads has a root mean square near 1 (strength: mean 0.47;
# radius: mean 0.51). The control that catches a unit lying: window_summary's features_rms.
POSITION_UNIT_CM = 12.0
TURN_UNIT_DEG = 20.0
EYE_UNIT_RAD = 0.3
# THE REACH CLAMP: the trainer clamps a commanded position step to 40 cm x rom_limit per axis around the pose
# (mimo_bridge/lem_model.py: reach = 40.0 * rom_limit; rom_limit is 1 in the mature phase H84 runs). An asked position
# change beyond it is never executed; the hold loss's do-nothing row (position 0, strength 0) asks a move to the world
# origin, which this clamp bounds.
POSITION_CLAMP_CM = 40.0
PRESENT_NORM = 0.5                 # a row holds a command (a pose) when its quaternion's length is above this
F_ASK = 10
# THE SHARED PER-HANDLE NETWORK. The servo's control is strength x (the rotation still needed, the wanted move): six
# products per handle. RANK = 16 leaves room for those six and ten more; HIDDEN = 64 is 4 x RANK. Neither was tuned.
# The controls that catch the sizes lying: the trial's token read-out (the token's R^2 for the asked change, read-out 2)
# and window_summary's handle_reach. The 10 raw features reach `up` beside the network's output, so the path carries
# the asked change linearly whatever the network does.
HIDDEN = 64
RANK = 16
# THE OUTPUT'S GAIN: u = up(z) / sqrt(N), N = n_handles x (F_ASK + RANK) numbers read by `up` (676 on H84's 26
# handles; gain 1/26). Why: Adam moves every weight of `up` by about the learning rate per update whatever the
# gradient's size, and on one row all of a number's N weights can move the same way, so u can change by up to
# learning rate x sum|z| per update. MEASURED on the 2,000 recorded rows (Oct 3, 2026, three seeds of the shared
# network): sum|z| = 236 to 240 on average (largest 347), |z| = 14.3. Without the gain that bound is 3e-4 x 238 =
# 0.071 token units per update (1 token unit after 14 updates of noise); with it 0.0027 per update, the size of the
# gaze copy's bound (128 hidden units of about 0.5: 3e-4 x 64 = 0.019). One token unit then needs at least 365
# updates that all push the same way. The control: window_summary's output_rms per report.
def output_gain(n_handles: int, rank: int = RANK) -> float:
    return 1.0 / math.sqrt(float(n_handles * (F_ASK + rank)))


SEED_OFFSET = 8_000_008            # the module's own random stream: the run's seed + this
GAUGE_CALLS = 8                    # the newest lesson calls the window's gauges read (8 calls x 8 bodies = 64 rows)
GAUGE_ROWS = 512                   # and at most this many of their body rows
LIVE_LINE = -3.0                   # DESIGN section 10: a unit is live when its input rose above this


def split_row(rows: torch.Tensor):
    """(achieved [.., 10], command [.., 9]) of motor rows [.., 19]."""
    if rows.shape[-1] != ROW_F:
        raise ValueError("the asked-change path reads motor rows of %d numbers (achieved pose 10, command 9), not %d"
                         % (ROW_F, rows.shape[-1]))
    return rows[..., :ACHIEVED_F], rows[..., ACHIEVED_F:]


def features(rows: torch.Tensor, gaze_idx: int, eye_row_convention: str = "legacy") -> torch.Tensor:
    """[B, H, F_ASK] float32: what each handle's command asks relative to where the handle is (the module text).

    rows: [B, H, 19] the motor encoder's input. Differentiable in the command and the pose (the planner's descent
    passes through it); a row without a command or without a pose gives zeros with zero gradient."""
    from lem.model.efference import rotation_change_deg, gaze_order_features
    ach, cmd = split_row(rows.float())
    B, H = rows.shape[0], rows.shape[1]
    qa, qc = ach[..., 3:7], cmd[..., 3:7]
    has_cmd = qc.norm(dim=-1) > PRESENT_NORM                              # [B, H] a command is there
    has_pose = qa.norm(dim=-1) > PRESENT_NORM
    # a body handle asks something when a command and a pose are there and its strength is above 0 (the limb servo
    # skips a handle whose strength is 0); the eyes' row when its command is there (the module text)
    asks = has_cmd & has_pose & (cmd[..., 8] > 0)
    has_eye = 0 <= int(gaze_idx) < H
    eye = torch.zeros(H, dtype=torch.bool, device=rows.device)
    if has_eye:
        eye[int(gaze_idx)] = True
    present = torch.where(eye.unsqueeze(0), has_cmd, asks)
    pf = present.to(ach.dtype).unsqueeze(-1)
    # an absent quaternion (length 0) is replaced by the identity before the turn is formed, so the turn of a handle
    # that asks nothing is 0 with a finite gradient (atan2 has none at 0, 0); such a handle's body features are
    # multiplied by 0 below. The eyes' row forms its own two angles from its command alone.
    ident = torch.zeros(4, dtype=ach.dtype, device=rows.device)
    ident[3] = 1.0
    both = asks.unsqueeze(-1)
    qa = torch.where(both, qa, ident)
    qc_body = torch.where(both, qc, ident)
    qc_safe = torch.where(has_cmd.unsqueeze(-1), qc, ident)
    dpos = ((cmd[..., 0:3] - ach[..., 0:3]) * POSITION_RANGE_CM).clamp(-POSITION_CLAMP_CM, POSITION_CLAMP_CM)
    dpos = dpos / POSITION_UNIT_CM
    q0 = qa / qa.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q1 = qc_body / qc_body.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    turn = rotation_change_deg(q0, q1) / TURN_UNIT_DEG
    body = torch.cat([dpos, turn], dim=-1)
    eyef = eye.to(ach.dtype).view(1, H, 1)
    if has_eye:
        # the eyes' row: the command's azimuth and elevation, the numbers the eye servo is sent
        order = torch.cat([cmd[..., 0:3], qc_safe, cmd[..., 7:9]], dim=-1)
        az_el = gaze_order_features(order, int(gaze_idx), eye_row_convention)[:, 9:11] / EYE_UNIT_RAD   # [B, 2]
        gaze = torch.zeros(B, H, 6, dtype=ach.dtype, device=rows.device)
        gaze[:, int(gaze_idx), 3:5] = az_el
        body = body * (1.0 - eyef) + gaze * eyef
    out = torch.cat([body, cmd[..., 8:9], cmd[..., 7:8], torch.ones_like(pf), eyef.expand(B, H, 1)], dim=-1)
    return out * pf


class MotorAskedChange(nn.Module):
    """The asked-change path beside the motor encoder (the module text). forward(rows) returns u [B, 1, d] in token
    units; attach(encoder) adds s x u to the encoder's output."""

    def __init__(self, n_handles: int, gaze_idx: int, d_model: int, eye_row_convention: str = "legacy",
                 hidden: int = HIDDEN, rank: int = RANK):
        super().__init__()
        self.n_handles = int(n_handles)
        self.gaze_idx = int(gaze_idx)
        self.d_model = int(d_model)
        self.eye_row_convention = str(eye_row_convention)
        self.rank = int(rank)
        self.net = nn.Sequential(nn.Linear(F_ASK, int(hidden)), nn.GELU(), nn.Linear(int(hidden), self.rank))
        self.up = nn.Linear(self.n_handles * (F_ASK + self.rank), self.d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gain = output_gain(self.n_handles, self.rank)
        # the window's gauges: the newest lesson calls' features and hidden-layer inputs (kept when arm_gauge() was
        # called before the encoder's call; read once per report) and the unit s summed on the device over every call
        self.gauge_calls: deque = deque(maxlen=GAUGE_CALLS)
        self.gauge_armed = False
        self.unit_sum = None
        self.unit_n = 0
        self.attached = False
        self.bypass = False               # tests and the switch check: the hook adds nothing while True
        self.frozen_depth = 0             # above 0 (inside frozen()): the path's weights enter detached
        self._detach = lambda: None

    # ------------------------------------------------------------------ the path
    def features(self, rows: torch.Tensor) -> torch.Tensor:
        return features(rows, self.gaze_idx, self.eye_row_convention)

    @contextlib.contextmanager
    def frozen(self):
        """Inside: the path computes the same numbers with its weights detached, so no lesson reaches them; the row
        (the command, the pose) keeps its gradient. Used for every imagined row (the module text, GRADIENT). Nests;
        an exception inside leaves the count where it was."""
        self.frozen_depth += 1
        try:
            yield self
        finally:
            self.frozen_depth -= 1

    def _hidden(self, a: torch.Tensor):
        """(h, g): the hidden layer's input to its GELU (as the liveness repair routes it, when it does) and the
        network's output, for features a [B, H, F_ASK]."""
        x = a.to(self.net[0].weight.dtype)
        if self.frozen_depth:
            # the hidden layer is called as a module (so the liveness monitor's and repair's hooks on it run as in
            # every other call) with its weight and bias swapped for detached ones; the output layer of the network
            # has no hook and is applied directly
            from torch.func import functional_call
            lin = self.net[0]
            h = functional_call(lin, {"weight": lin.weight.detach(), "bias": lin.bias.detach()}, (x,))
            return h, F.linear(self.net[1](h), self.net[2].weight.detach(), self.net[2].bias.detach())
        h = self.net[0](x)
        return h, self.net[2](self.net[1](h))

    def handle_features(self, rows: torch.Tensor) -> torch.Tensor:
        """[B, H, F_ASK + RANK]: what `up` reads, handle by handle (a handle's block depends on that handle's row
        only)."""
        a = self.features(rows)
        if a.shape[1] != self.n_handles:
            raise ValueError("the asked-change path was built for %d handles, the row has %d"
                             % (self.n_handles, a.shape[1]))
        h, g = self._hidden(a)
        if self.gauge_armed:
            # the delayed lesson's call (arm_gauge): its features and hidden-layer inputs, for the window's gauges
            self.gauge_calls.append((a.detach().float(), h.detach().float()))
            self.gauge_armed = False
        return torch.cat([a.to(g.dtype), g], dim=-1) * a[..., 8:9].to(g.dtype)

    def output(self, z: torch.Tensor) -> torch.Tensor:
        """u [B, d_model] from the handles' features z [B, H, F_ASK + RANK]: exactly 0 for a row no handle of which
        holds a command."""
        flat = z.reshape(z.shape[0], -1)
        if self.frozen_depth:
            u = F.linear(flat, self.up.weight.detach(), self.up.bias.detach()) * self.gain
        else:
            u = self.up(flat) * self.gain
        any_cmd = (z[..., 8].amax(dim=1) > 0).to(u.dtype).unsqueeze(-1)       # [B, 1]: any handle holds a command
        return u * any_cmd

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """u [B, 1, d_model], in units of the token before its final LayerNorm; exactly 0 for a row without a
        command."""
        return self.output(self.handle_features(rows)).unsqueeze(1)

    # ------------------------------------------------------------------ the hook
    def attach(self, encoder: nn.Module) -> None:
        """One forward hook on the motor encoder module: its output v [B, 1, d] becomes v + s x u (the module
        text). Every call of the encoder passes through it."""
        if self.attached:
            raise RuntimeError("the asked-change path is attached already")
        first = getattr(encoder, "patch_mlp", [None])[0]
        if getattr(first, "in_features", None) != ROW_F:
            raise ValueError("--motor-asked-change needs the motor encoder that reads 19 numbers per handle "
                             "(achieved pose 10, command 9); this one reads %s" % getattr(first, "in_features", None))

        def hook(mod, args, out):
            if self.bypass:
                return None
            x = args[0]
            u = self(x)
            # the sum is formed in float32 at least (the module text, OUTPUT): added in the run's bf16 a small
            # output would be rounded away at the pooled vector's size
            dt = torch.promote_types(out.dtype, torch.float32)
            v = out.to(dt)
            s = v.detach().std(dim=-1, keepdim=True, unbiased=False)
            with torch.no_grad():
                self.unit_sum = s.mean() if self.unit_sum is None else self.unit_sum + s.mean()
                self.unit_n += 1
            return v + s * u.to(dt)
        handle = encoder.register_forward_hook(hook)
        # the encoder names its path, so a caller that encodes an imagined row can freeze it
        # (lem.model.encoder_registry.imagined_motor_token). Written into the instance's own dict: a plain
        # reference, not a registered submodule, so the encoder's state dict and its checkpoint keys are unchanged
        encoder.__dict__["asked_change_path"] = self

        def remove():
            handle.remove()
            encoder.__dict__.pop("asked_change_path", None)
        self._detach = remove
        self.attached = True

    def detach(self) -> None:
        self._detach()
        self._detach = lambda: None
        self.attached = False

    # ------------------------------------------------------------------ records
    def stamp(self) -> dict:
        return {"version": ASKED_CHANGE_VERSION, "mode": "on", "handles": self.n_handles, "features": F_ASK,
                "hidden": int(self.net[0].out_features), "rank": self.rank,
                "units": {"position_cm": POSITION_UNIT_CM, "turn_deg": TURN_UNIT_DEG, "eye_rad": EYE_UNIT_RAD,
                          "position_clamp_cm": POSITION_CLAMP_CM},
                "eye_row_convention": self.eye_row_convention, "output_gain": self.gain,
                "output_unit": "the standard deviation of the row's pooled vector over its numbers",
                "liveness_layer": LIVENESS_LAYER}

    def arm_gauge(self) -> None:
        """The next call of the path keeps its features and hidden-layer inputs for the window's gauges. LemLite.step
        calls this right before the delayed lesson encodes its motor row, so the gauges read the lesson's own
        commanded rows and add no forward pass (a pass of their own would join the liveness repair's watching
        window)."""
        self.gauge_armed = True

    @torch.no_grad()
    def window_summary(self) -> dict:
        """The `motor_asked_change` stats block; a fresh window starts. Read on the newest lesson calls the gauge kept
        (arm_gauge; at most GAUGE_CALLS calls, the body rows of them that hold a command), with the weights as they
        are at the report and no forward pass through any hooked layer:
          rows               body rows the gauges read; 0 = no lesson call in the window (the fields after
                             output_weight_rms are then absent);
          encoder_output_sd  mean over the window's encoder calls of s, the standard deviation of the pooled vector
                             over its numbers (the unit u is multiplied by);
          output_weight_rms  root mean square of `up`'s weights (0 = the switch);
          output_rms         root mean square per number of u on those rows, in token units: 0 = the path adds
                             nothing, 1 = as large as the pooled vector after standardising;
          handle_reach       how evenly the handles share the path's output: per handle the length of its own part
                             of u, as shares; spread = their entropy over log(n_handles) (1 = even, 0 = one handle),
                             largest_share, handles_under_1_in_1000; absent while `up` is still zero;
          features_rms       root mean square of each feature group over the commanded handles (the units' control:
                             near 1 for position and turn);
          liveness           {"net.0": share of the hidden units whose input rose above -3 on those rows} and
                             liveness_varying (the share whose GELU output range exceeds lem/model/liveness.py's
                             floor)."""
        out = {"version": ASKED_CHANGE_VERSION, "rows": 0}
        if self.unit_n:
            out["encoder_output_sd"] = round(float(self.unit_sum) / self.unit_n, 3)
            out["encoder_calls"] = int(self.unit_n)
        self.unit_sum, self.unit_n = None, 0
        calls = [c for c in self.gauge_calls if c[0].shape[1] == self.n_handles]
        self.gauge_calls.clear()
        out["output_weight_rms"] = float(self.up.weight.float().pow(2).mean().sqrt())
        if not calls:
            return out
        a = torch.cat([c[0] for c in calls], dim=0)
        h = torch.cat([c[1] for c in calls], dim=0)
        keep = a[..., 8].amax(dim=1) > 0
        if not bool(keep.any()):
            return out
        a, h = a[keep][-GAUGE_ROWS:], h[keep][-GAUGE_ROWS:]
        g = F.linear(F.gelu(h), self.net[2].weight.float(), self.net[2].bias.float())
        z = torch.cat([a, g], dim=-1) * a[..., 8:9]
        n, H = z.shape[0], z.shape[1]
        u = F.linear(z.reshape(n, -1), self.up.weight.float(), self.up.bias.float()) * self.gain
        out["rows"] = int(n)
        out["output_rms"] = float(u.pow(2).mean().sqrt())
        cmd = a[..., 8] > 0
        body = cmd.clone()
        if 0 <= self.gaze_idx < H:
            body[:, self.gaze_idx] = False

        def group_rms(t, mask):
            m = mask.unsqueeze(-1).expand_as(t)
            return round(float(t[m].pow(2).mean().sqrt()), 4) if bool(m.any()) else None
        out["features_rms"] = {"position": group_rms(a[..., 0:3], body), "turn": group_rms(a[..., 3:6], body),
                               "strength": group_rms(a[..., 6:7], cmd), "radius": group_rms(a[..., 7:8], cmd)}
        if 0 <= self.gaze_idx < H:
            out["features_rms"]["eye"] = round(float(a[:, self.gaze_idx, 3:5].pow(2).mean().sqrt()), 4)
        W = self.up.weight.float().reshape(self.d_model, H, -1)               # [d, H, F + R]
        part = torch.einsum("dhf,nhf->nhd", W, z)                             # each handle's own part of u
        size = part.norm(dim=-1).mean(dim=0)                                  # [H]
        if float(size.sum()) > 0:
            share = size / size.sum()
            ent = -(share * share.clamp_min(1e-12).log()).sum() / math.log(max(H, 2))
            out["handle_reach"] = {"spread": round(float(ent), 4), "largest_share": round(float(share.max()), 4),
                                   "handles_under_1_in_1000": int((share < 1e-3).sum())}
        p = h.reshape(-1, h.shape[-1])[cmd.reshape(-1)]
        out["liveness"] = {"net.0": round(float((p > LIVE_LINE).any(dim=0).float().mean()), 4)}
        if p.shape[0] > 1:
            from lem.model.liveness import varying_floor
            post = F.gelu(p)
            lo, hi = post.min(dim=0).values, post.max(dim=0).values
            out["liveness_varying"] = {"net.0": round(float(((hi - lo) > varying_floor(lo, hi, "gelu").to(hi))
                                                             .float().mean()), 4)}
        return out


def build(n_handles: int, gaze_idx: int, d_model: int, seed: int, eye_row_convention: str = "legacy") -> MotorAskedChange:
    """The path, built in its own random stream (the run's stream is left where a build without it leaves it)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + SEED_OFFSET)
        return MotorAskedChange(n_handles, gaze_idx, d_model, eye_row_convention=eye_row_convention)


@torch.no_grad()
def switch_check(model, rows: int = 64, seed: int = 0) -> dict:
    """What switching the path on does to the motor token, measured on `rows` probe rows: the largest absolute change
    of the token with the path against without it, in float32 and, on the graphics card, under the run's autocast.
    With `up` at zero both are exactly 0.0. Probe rows: poses and commands drawn at the recorded sizes (positions 33 cm
    root mean square, an asked change of 12 cm per axis, unit quaternions), from a generator of its own."""
    path = getattr(model, "motor_asked_change", None)
    if path is None:
        raise ValueError("the model was built without --motor-asked-change on")
    reg = getattr(model.encoder_registry, "_orig_mod", model.encoder_registry)
    dev = next(path.parameters()).device
    g = torch.Generator().manual_seed(int(seed) + SEED_OFFSET)
    H = path.n_handles
    x = torch.zeros(rows, H, ROW_F)
    x[..., 0:3] = torch.randn(rows, H, 3, generator=g) * (33.0 / POSITION_RANGE_CM)
    q = torch.randn(rows, H, 4, generator=g)
    x[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    x[..., 7:10] = torch.tensor([0.5, 0.5, 1.0])
    x[..., 10:13] = x[..., 0:3] + torch.randn(rows, H, 3, generator=g) * (POSITION_UNIT_CM / POSITION_RANGE_CM)
    q = torch.randn(rows, H, 4, generator=g)
    x[..., 13:17] = q / q.norm(dim=-1, keepdim=True)
    x[..., 17:19] = torch.rand(rows, H, 2, generator=g)
    x = x.to(dev)
    saved = (path.unit_sum, path.unit_n)
    out = {"rows": int(rows), "output_weight_max_abs": float(path.up.weight.abs().max()),
           "output_bias_max_abs": float(path.up.bias.abs().max())}

    def token(bypass, amp):
        path.bypass = bypass
        try:
            with torch.amp.autocast("cuda", enabled=amp, dtype=getattr(model, "dtype", torch.float32)):
                return reg.encode({"motor": x}, {"motor": "real"})["motor"].float()
        finally:
            path.bypass = False
    modes = [("float32", False)] + ([("autocast", True)] if getattr(model, "use_amp", False) else [])
    for name, amp in modes:
        on, off = token(False, amp), token(True, amp)
        out[name] = {"change_max_abs": float((on - off).abs().max()), "token_rms": float(off.pow(2).mean().sqrt()),
                     "bit_identical": bool(torch.equal(on, off))}
    path.unit_sum, path.unit_n = saved
    out["exact"] = all(out[name]["bit_identical"] for name, _ in modes)
    return out
