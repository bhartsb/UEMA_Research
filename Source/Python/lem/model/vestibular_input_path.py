# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The vestibular encoder's linear input path (--vestibular-input-path, default off).

WHAT WAS MEASURED (docs/gain_of_function_v4/agents/vestibular_token_2026-10-04.md; registry rows
vestibular_token_stage_readouts, vestibular_token_lived_readouts and vestibular_token_path_bench; Oct 4, 2026, on the
replayed inputs of H84's 8 bodies, 1,670 consecutive model steps each, and on checkpoint 479,350). The read-out is the
trainer's own probe: each body's change of head tilt within a report of 100 model steps, read by a ridge regression
fitted on four bodies and scored on the other four (0 = no better than predicting no change, 1 = exact, the trainer's
line 0.25).
  * The encoder's input (28 numbers) reads head tilt at 0.56. A freshly initialised encoder of the same shape passes
    it to its token at 0.41. The trained encoder does not: saved at 461,631 its third hidden layer still read 0.40
    and its last layer 0.00 (in float32 the token still held the tilt at 0.000005 of the size of its main change,
    under the graphics card's bf16 rounding); saved at 479,350 its last layer and token hold it at no size (0.00
    from the 8 principal components that carry all of the token's change, negative beyond).
  * The trained token's change within a report sits in 2 directions (0.97 of it; a fresh encoder: 0.20) that follow
    the neck angles; 27 of the 28 input fields cannot be read back from it (a fresh encoder's token: 24 of 28 read
    above 0.5). Its change between two model steps is 0.02 per number against a fresh encoder's 0.58.
  * No other sense's token and no slot of the transformer's output carries head tilt in that form (0.09 or less).
  * WHY: two lessons reach the encoder without passing the transformer. VICReg's term falls when the 8 bodies'
    tokens differ at one model step. The next-model-step lesson's `+ current` residual adds the squared change of
    the token between two model steps to the loss (lem/model/next_frame_norm.py, THE ENCODER'S SHARE). No term falls
    when a body's token follows that body's own input over time. On the bench these two lessons alone took a fresh
    encoder's read-out from 0.41 to 0.06 in 20,000 updates, and left today's encoder at 0.01 or less.

WHAT THIS MODULE ADDS, with the flag on (off builds nothing: an old command builds the model it always built):

  INPUT.   The vestibular encoder's own input rows x [B, 28], at every call of the encoder (one forward hook on the
           encoder module): the 28 fields of the vestibular vector without its validity flag (head accelerometer in g,
           gyro, neck angles, speeds and efforts, the trunk's up direction, support shares and total, effort, the
           share of joints at a limit, motor capacity), each minus its running mean, divided by its running spread
           and clipped to +-4 (mimo_bridge/vestibular_sense.VestibularNormaliser).
  PATH.    u[b] = up(x[b]) x GAIN: one Linear(28 -> 192) without bias, its weight zero at the start. No hidden layer,
           no normalisation, no saturation: a change of an input field by d changes u by d x that field's column x
           GAIN, the same for a difference between two bodies and for one body's own change over time. So whatever
           VICReg's term gains from telling the bodies apart through an input field, each body's own change of that
           field passes into the token at the same size.
  OUTPUT.  The encoder's output v[b] (the last layer's output, before the registry's LayerNorm) becomes
           v[b] + s[b] x u[b], s[b] = the standard deviation of v[b] over its 192 numbers (no gradient; 18.09 at
           479,350). The LayerNorm then gives LayerNorm(v + s u) = LayerNorm(LayerNorm(v) + u) up to its eps: one
           unit of u is one unit of the token before its final normalisation. With `up` at zero, s x u is exactly 0
           and the token is bit for bit the token without the path. THE SUM IS FORMED IN FLOAT32 (v is bf16 under the
           run's autocast, on a grid of up to 0.25 at numbers of 32 to 64, which is 0.014 token units; the LayerNorm
           computes in float32 whatever it is given, so converting v to float32 changes no value).

THE TWO WIRINGS (the flag's two on values; the same path, the same tokens, the same switch). The token is
LayerNorm(the encoder's part v / s + the path's part u), and what differs is what the delayed lesson's re-encoding
of the previous row trains (LemLite.step):
  "on"                       neither part: the previous row's vestibular token enters the delayed lesson without
                             gradient, so every lesson that trains the encoder's part trains the path's part too.
  "on_delayed_encoder_only"  the encoder's own layers and not the path: the path's weight is detached there
                             (frozen()) and the encoder's layers keep the delayed lesson's gradient. THE EXCEPTION.
WHY "on" IS THE WIRING WITHOUT THE EXCEPTION (the coordinator's decision of Oct 4, 2026). The prediction heads' first
all-slot paths ran away in production because one lesson trained the old head and not the path while both lessons
graded their sum. "on_delayed_encoder_only" has that shape, and its cost is measured: the encoder's part learns to
work against the path's. The independent review's long bench (docs/gain_of_function_v4/agents/
vestibular_input_path_review_2026-10-04.md, sections 7 and 8; path version 1, 240,000 updates, 16 reports read by
no update) read, for the exception, the cancelling levelling at 70 to 83 of 100 parts of the path's change with the
read-out at 0.46, while the path's weight (6.5) and fixed vector (0.15) kept growing; and for the wiring without
it a read-out of 0.43 to 0.56 with no opposition. The exception stays selectable as the second value. (A third
form, the delayed lesson's token-change term training the path's weight too, fails the target: on this module's
bench its read-out stayed under the acceptance line of 0.25.)

WHICH LESSONS TRAIN WHICH PART, MEASURED per lesson term on a run with every lesson live (tests/
test_vestibular_input_path.py holds both tables; the summed absolute gradient over 10 model steps on the path's
weight and on the encoder's weights):
  * BOTH parts, in both wirings: the live row's lessons that reach the encoders, vicreg (3.19 and 70.1 under
    "on_delayed_encoder_only") and recon_dec, the reconstruction decoders' term (0.037 and 0.89).
  * NEITHER PART under "on"; THE ENCODER'S PART AND NOT THE PATH'S under "on_delayed_encoder_only": every lesson
    that reads the delayed row's re-encoding: next_frame.vestibular (837 on the encoder under the exception, the
    largest gradient it gets), the other senses' next_frame terms (6.7 to 15.9), contact_head (2.07),
    efference_gaze (1.24), latent_reg (0.15), efference_body (0.024).
  * NEITHER part, in both wirings: the imagination chain (an imagined row's vestibular token is the prediction
    head's output through the registry's output LayerNorm; no vestibular row is encoded there, and the same holds
    for the planner's rows), the hold-end lessons (tenure, tenure_tok), replay, and the motor, wellness,
    persistent-memory and synergy terms.
  * No lesson trains the path alone, in either wiring.

THE BOUND, in two parts (path version 2).
  THE WEIGHT IS HELD AT THE CAP. On every live call of a training model step, before u is formed: when the largest
  row's plain output up(x) x GAIN has a root mean square above OUTPUT_CAP, the whole weight is multiplied by
  OUTPUT_CAP / that size (in place, no gradient). So the weight cannot grow past what the cap lets through, the path
  stays its linear map, and its direction turns at the rate the optimizer gives it. WHY: version 1 capped the
  output per row only, and the cap's factor carries no gradient, so VICReg's term went on enlarging the weight
  under it: the review measured its root mean square at 0.20 at 4,000 updates, 1.40 at 20,000 and 3.39 at 60,000.
  With the hold the encoder bench reads 0.07 to 0.105 from 4,000 through 60,000 updates in both wirings, the
  weight scaled back at 1 in 100 of the updates or fewer.
  THE PER-ROW CAP stays as the bound on every other call (the delayed row, a row of a model step that does not
  train, a measuring call): a row whose u has a root mean square above OUTPUT_CAP is multiplied by OUTPUT_CAP /
  that size (no gradient through the factor). No row's output is ever above OUTPUT_CAP token units per number, so
  of the token before its final normalisation at most OUTPUT_CAP^2 / (1 + OUTPUT_CAP^2) of the variance is the
  path's (0.083 at 0.3). With the weight held it acts on 1 in 100 rows or fewer.

WHAT EACH WIRING MEASURED (row vestibular_token_path_bench, the run on path version 2: the encoder alone under the
two lessons that reach it directly, rows of model steps 478,030 to 479,350, 60,000 updates, 5 reports read by no
update). The change slope: over each body's change from its own mean within a report, the summed product of the
two parts' changes over the path's summed squared change (-1 = the encoder's part takes the path's change out of
the sum, 0 = unrelated).
  "on":                       head tilt on unread rows 0.43 to 0.54 (line 0.25); the change slope -0.02 to +0.20
                              throughout; the token's change between two model steps 0.14 to 0.22 token units
                              against 0.08 to 0.13 under the exception (the encoder's own layers no longer get the
                              term that holds the token still).
  "on_delayed_encoder_only":  head tilt on unread rows 0.46 to 0.53; the change slope 0.01 at 4,000 updates, -0.20
                              at 10,000, -0.34 at 40,000, -0.46 at 60,000: the encoder's part works against the
                              path's, and more at the later readings.
window_summary reports the slope per report (change_slope) and the watch holds it to a standing line.

WHAT IT DOES NOT DO. It adds no lesson. "on" takes the delayed row's lessons off the encoder's own layers;
"on_delayed_encoder_only" changes no lesson of theirs. It puts a linear map of all 28 input fields into the token, the fast ones
(gyro, neck speeds, efforts) included, so the token changes more between model steps than today's 0.02 to 0.05
(production's other senses: 0.08 to 0.21).

Every constant carries its measurement.
"""
from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_PATH_VERSION = "vestibular_input_path_v2"   # v2 (Oct 4, 2026): the weight held at the cap; the two wirings
# THE FLAG'S VALUES. "off" builds nothing. The two others build the same path and differ in ONE thing, what the delayed
# lesson's re-encoding of the previous row trains (the module text, WHICH LESSONS TRAIN WHICH PART):
#   "on"                       neither: the previous row's vestibular token enters the delayed lesson without
#                              gradient, so every lesson that trains the encoder's part trains the path's part too;
#   "on_delayed_encoder_only"  the encoder's own layers, and not the path's weight (the exception).
WIRINGS = ("on", "on_delayed_encoder_only")
MODES = ("off",) + WIRINGS
SIDE_NAME = "vestibular_input_path"

# THE OUTPUT'S GAIN: u = up(x) / sqrt(n_inputs). Why (the asked-change path's rule, lem/model/motor_asked_change.py
# output_gain): Adam moves every weight of `up` by about the learning rate per update whatever the gradient's size,
# and on one row all of a number's n_inputs weights can move the same way, so u can change by up to
# learning rate x sum|x| x GAIN per update. The inputs are centred and scaled to a running spread of 1 and clipped at
# 4: MEASURED on the replayed rows with checkpoint 479,350's statistics (bench.json summed_input_size, Oct 4, 2026):
# sum|x| over the 28 fields is 22.8 on average (99th percentile 42.5, largest 54.7), so without the gain the bound is
# 3e-4 x 22.8 = 0.0068 token units per update; with it 0.0013. The bench's measured growth was 14 of every 100 parts
# of that bound after 100 updates and 6.5 after 3,000. The control: window_summary's output_rms per report (the
# production watch's line M6, docs/gain_of_function_v4/vestibular_input_path_watch.py).
def output_gain(n_inputs: int) -> float:
    return 1.0 / math.sqrt(float(n_inputs))


# THE OUTPUT'S CAP, token units (root mean square of u over one row's 192 numbers; 1.0 would be as large as the
# encoder's own standardised output). 0.3: at most 0.083 of the token's variance before its final normalisation.
# WHY 0.3, MEASURED Oct 4, 2026 (rows vestibular_token_path_bench and vestibular_token_path_live_bench; the record
# docs/gain_of_function_v4/agents/vestibular_token_2026-10-04.md):
#   * without a bound the output does not stop where the path's purpose is met. With checkpoint 479,350's whole
#     model updated on the graphics card (path version 1, the cap at 1.0, never reached), the read-out of head tilt
#     from the token reached 0.53 at 2,000 updates, at a mean output of 0.13, and stayed there; the output went on
#     to 0.22 at 3,000 (largest row 0.65), rising by 0.047 per 500 updates. On the encoder alone it read 0.52 at
#     6,000 updates and 0.79 at 20,000.
#   * with the weight held at 0.3 (path version 2, the encoder alone, 60,000 updates, both wirings) the largest row
#     sits at 0.300, the mean output at 0.08 to 0.14, and the read-out on unread rows at 0.43 to 0.54 (acceptance
#     line 0.25).
# So the cap sits above the output at which the read-out is reached and bounds what the lessons add past it.
# The controls: window_summary's weight_held_share (the share of live calls at which the weight was scaled back) and
# capped_share (the share of rows the per-row cap scaled down; 0 = none, 1 = every row).
OUTPUT_CAP = 0.3
SEED_OFFSET = 8_000_029            # the module's own random stream (nothing is drawn today: `up` starts at zero)
# the field groups window_summary reports (index ranges of the 28 encoder fields,
# mimo_bridge/vestibular_sense.ENCODER_FIELDS)
FIELD_GROUPS = {"head_accelerometer": (0, 3), "head_gyro": (3, 6), "neck_angles": (6, 9), "neck_speeds": (9, 12),
                "neck_efforts": (12, 15), "trunk_up": (15, 18), "support_shares": (18, 22),
                "support_total": (22, 23), "effort": (23, 25), "limit_share": (25, 26), "motor_capacity": (26, 28)}


class VestibularInputPath(nn.Module):
    """The linear path beside the vestibular encoder (the module text). forward(x) returns u [B, d] in token units;
    attach(encoder) adds s x u to the encoder's output."""

    def __init__(self, n_inputs: int, d_model: int, cap: float = OUTPUT_CAP, wiring: str = "on"):
        super().__init__()
        if wiring not in WIRINGS:
            raise ValueError("the vestibular input path's wiring is one of %s, not %r" % (WIRINGS, wiring))
        self.n_inputs = int(n_inputs)
        self.d_model = int(d_model)
        self.cap = float(cap)             # the largest root mean square of u over one row's numbers
        self.wiring = wiring
        self.delayed_detached = wiring == "on"            # LemLite.step reads it (the delayed lesson)
        self.live_n = 0                   # the window's live calls, and those at which the weight was scaled back
        self.held_n = 0
        self.hold_enabled = True          # LemLite.step sets it to `train` at every model step
        self.up = nn.Linear(self.n_inputs, self.d_model, bias=False)
        nn.init.zeros_(self.up.weight)
        self.gain = output_gain(self.n_inputs)
        self.unit_sum = None              # the window's gauges, summed on the device (no read until the report)
        self.unit_n = 0
        self.out_sq_sum = None
        self.out_n = 0
        self.capped_sum = None            # rows the cap scaled down, and rows seen, summed on the device
        self.rows_n = 0
        self.parts = None                 # the live rows' sums for change_slope (window_summary), on the device
        self.attached = False
        self.bypass = False               # tests and the switch check: the hook adds nothing while True
        self.frozen_depth = 0             # above 0 (inside frozen()): the path's weight enters detached
        self._detach = lambda: None

    @contextlib.contextmanager
    def frozen(self):
        """Inside: the path computes the same numbers with its weight detached, so no lesson reaches it; the row keeps
        its gradient. Nests; an exception inside leaves the count where it was."""
        self.frozen_depth += 1
        try:
            yield self
        finally:
            self.frozen_depth -= 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """u [B, d_model] in units of the token before its final LayerNorm, for encoder inputs x [B, n_inputs]."""
        if x.shape[-1] != self.n_inputs:
            raise ValueError("the vestibular input path was built for %d input fields, the row has %d"
                             % (self.n_inputs, x.shape[-1]))
        w = self.up.weight
        if self.frozen_depth:
            w = w.detach()
        u = F.linear(x.to(w.dtype), w) * self.gain
        # the bound (module text): rows above the cap are scaled back to it; under it the factor is exactly 1
        size = u.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        with torch.no_grad():
            over = (size > self.cap).float().sum()
            self.capped_sum = over if self.capped_sum is None else self.capped_sum + over
            self.rows_n += int(size.shape[0])
        return u * (self.cap / size.clamp_min(self.cap)).to(u.dtype)

    @torch.no_grad()
    def _hold_weight(self, x: torch.Tensor) -> None:
        """The live row's call, before u is formed (module text, THE BOUND): when the largest row's plain output
        up(x) x GAIN has a root mean square above the cap, the weight is multiplied in place by cap / that size, so
        the weight itself cannot grow past what the cap lets through. Computed in float32 with autocast off: a
        bf16 copy of the weight made here, under no_grad, would be the copy autocast keeps for the rest of its block,
        and the live row's lesson would then reach no gradient (row autocast_cache_gradient_audit_479350)."""
        w = self.up.weight
        with torch.autocast(device_type=w.device.type, enabled=False):
            size = float((F.linear(x.float(), w.float()) * self.gain).pow(2).mean(dim=-1).sqrt().max())
        self.live_n += 1
        if size > self.cap:
            w.mul_(self.cap / size)
            self.held_n += 1

    @torch.no_grad()
    def _two_parts(self, own: torch.Tensor, u: torch.Tensor) -> None:
        """One live call's rows join the window's sums: own [B, d] = the encoder's last-layer output over its standard
        deviation (its part of the sum, in token units), u [B, d] = the path's part. Per body the sums over the
        window's live calls of own, of u, and of the products own x u, u x u, own x own (float64, on the device)."""
        own, u = own.double(), u.double()
        p = self.parts
        if p is None or p["own"].shape != own.shape:
            p = self.parts = dict(n=0, own=torch.zeros_like(own), u=torch.zeros_like(u),
                                  own_u=own.new_zeros(()), u_u=own.new_zeros(()), own_own=own.new_zeros(()))
        p["n"] += 1
        p["own"] += own
        p["u"] += u
        p["own_u"] += (own * u).sum()
        p["u_u"] += (u * u).sum()
        p["own_own"] += (own * own).sum()

    # ------------------------------------------------------------------ the hook
    def attach(self, encoder: nn.Module) -> None:
        """One forward hook on the vestibular encoder module: its output v [B, 1, d] becomes v + s x u."""
        if self.attached:
            raise RuntimeError("the vestibular input path is attached already")
        first = getattr(encoder, "mlp", [None])[0]
        if getattr(first, "in_features", None) != self.n_inputs:
            raise ValueError("--vestibular-input-path needs the vestibular encoder that reads %d fields; this one "
                             "reads %s" % (self.n_inputs, getattr(first, "in_features", None)))

        def hook(mod, args, out):
            if self.bypass:
                return None
            x = args[0].reshape(-1, self.n_inputs)
            live = self.frozen_depth == 0 and torch.is_grad_enabled()      # the live row's call
            if live and self.hold_enabled:
                self._hold_weight(x)
            u = self(x).reshape(out.shape[0], -1, self.d_model)
            # the sum is formed in float32 at least (the module text, OUTPUT)
            dt = torch.promote_types(out.dtype, torch.float32)
            v = out.to(dt)
            s = v.detach().std(dim=-1, keepdim=True, unbiased=False)
            if live and out.shape[0] * self.d_model == out.numel():
                # the live row's call (one token per body): the two parts of the sum join the window's sums
                self._two_parts((v.detach() / s).reshape(-1, self.d_model), u.detach().reshape(-1, self.d_model))
            with torch.no_grad():
                self.unit_sum = s.mean() if self.unit_sum is None else self.unit_sum + s.mean()
                self.unit_n += 1
                q = u.detach().float().pow(2).mean()
                self.out_sq_sum = q if self.out_sq_sum is None else self.out_sq_sum + q
                self.out_n += 1
            return v + s * u.to(dt)
        handle = encoder.register_forward_hook(hook)

        def remove():
            handle.remove()
        self._detach = remove
        self.attached = True

    def detach(self) -> None:
        self._detach()
        self._detach = lambda: None
        self.attached = False

    # ------------------------------------------------------------------ records
    def stamp(self) -> dict:
        return {"version": INPUT_PATH_VERSION, "mode": self.wiring, "inputs": self.n_inputs,
                "output_gain": self.gain, "output_cap": self.cap,
                "delayed_lesson": ("trains neither the encoder's own layers nor the path (the previous row's "
                                   "vestibular token enters it without gradient)" if self.delayed_detached else
                                   "trains the encoder's own layers and not the path (the path's weight is detached "
                                   "inside its re-encoding)"),
                "output_unit": "the standard deviation of the row's last-layer output over its numbers",
                "bias": False, "learns_from": "the live row only (frozen inside the delayed lesson's re-encoding)"}

    @torch.no_grad()
    def window_summary(self) -> dict:
        """The `vestibular_input_path` stats block; a fresh window starts.
          encoder_output_sd  mean over the window's encoder calls of s, the standard deviation of the encoder's
                             last-layer output over its 192 numbers (the unit u is multiplied by; 18.1 on H84);
          output_rms         root mean square per number of u over the window's encoder calls, in token units: 0 =
                             the path adds nothing, 1 = as large as the encoder's own output after standardising;
          output_weight_rms  root mean square of `up`'s weight (0 = the switch);
          field_reach        per group of input fields, the root mean square of its columns of `up` x GAIN: the
                             change of u per number when one field of the group changes by one running spread;
          capped_share       the share of the window's rows whose output the cap scaled down (0 = none; the cap is
                             OUTPUT_CAP token units per number);
          output_cap         that cap, token units;
          change_slope       whether the encoder's own part works against the path's (module text, WHAT THE EXCEPTION
                             COSTS): over the window's live rows, each body's change from its own mean over the
                             window, the summed product of the encoder's part's change and the path's change over the
                             path's summed squared change. -1 = the encoder's part takes the path's change out of the
                             sum, 0 = unrelated, +1 = it doubles it. With own_change and path_change, the root mean
                             square per number of each part's change (token units), and live_calls, the window's
                             live calls the three are computed from (at least 2)."""
        out = {"version": INPUT_PATH_VERSION}
        if self.unit_n:
            out["encoder_output_sd"] = round(float(self.unit_sum) / self.unit_n, 4)
            out["encoder_calls"] = int(self.unit_n)
        if self.out_n:
            out["output_rms"] = float(torch.sqrt(self.out_sq_sum / self.out_n))
        if self.rows_n:
            out["capped_share"] = float(self.capped_sum) / self.rows_n
        out["output_cap"] = self.cap
        out["wiring"] = self.wiring
        if self.live_n:
            # the share of the window's live calls at which the weight was scaled back to the cap (0 = never)
            out["weight_held_share"] = self.held_n / float(self.live_n)
        self.live_n, self.held_n = 0, 0
        p, self.parts = self.parts, None
        if p is not None and p["n"] >= 2:
            n = p["n"]
            cov = float(p["own_u"] - (p["own"] * p["u"]).sum() / n)
            var_u = float(p["u_u"] - (p["u"] * p["u"]).sum() / n)
            var_o = float(p["own_own"] - (p["own"] * p["own"]).sum() / n)
            numbers = float(n * p["own"].numel())
            out["live_calls"] = int(n)
            out["own_change"] = max(var_o, 0.0) ** 0.5 / numbers ** 0.5
            out["path_change"] = max(var_u, 0.0) ** 0.5 / numbers ** 0.5
            out["change_slope"] = (cov / var_u) if var_u > 0.0 else None
        self.unit_sum, self.unit_n, self.out_sq_sum, self.out_n = None, 0, None, 0
        self.capped_sum, self.rows_n = None, 0
        w = self.up.weight.float()
        out["output_weight_rms"] = float(w.pow(2).mean().sqrt())
        if self.n_inputs == 28:
            out["field_reach"] = {k: float((w[:, a:b] * self.gain).pow(2).mean().sqrt())
                                  for k, (a, b) in FIELD_GROUPS.items()}
        return out


def build(n_inputs: int, d_model: int, seed: int = 0, wiring: str = "on") -> VestibularInputPath:
    """The path, built in its own random stream (the run's stream is left where a build without it leaves it)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + SEED_OFFSET)
        return VestibularInputPath(n_inputs, d_model, wiring=wiring)


@torch.no_grad()
def switch_check(model, rows: int = 64, seed: int = 0) -> dict:
    """What switching the path on does to the vestibular token, measured on `rows` probe rows: the largest absolute
    change of the token with the path against without it, in float32 and, on the graphics card, under the run's
    autocast. With `up` at zero both are exactly 0.0. Probe rows: the normaliser's contract (each field centred, unit
    spread, clipped at 4), from a generator of its own."""
    path = getattr(model, "vestibular_input_path", None)
    if path is None:
        raise ValueError("the model was built without --vestibular-input-path on")
    reg = getattr(model.encoder_registry, "_orig_mod", model.encoder_registry)
    dev = path.up.weight.device
    g = torch.Generator().manual_seed(int(seed) + SEED_OFFSET)
    x = torch.randn(int(rows), path.n_inputs, generator=g).clamp(-4.0, 4.0).to(dev)
    saved = (path.unit_sum, path.unit_n, path.out_sq_sum, path.out_n, path.capped_sum, path.rows_n)
    out = {"rows": int(rows), "output_weight_max_abs": float(path.up.weight.abs().max())}

    def token(bypass, amp):
        path.bypass = bypass
        try:
            with torch.amp.autocast("cuda", enabled=amp, dtype=getattr(model, "dtype", torch.float32)):
                return reg.encode({"vestibular": x}, {"vestibular": "real"})["vestibular"].float()
        finally:
            path.bypass = False
    modes = [("float32", False)] + ([("autocast", True)] if getattr(model, "use_amp", False) else [])
    for name, amp in modes:
        on, off = token(False, amp), token(True, amp)
        out[name] = {"change_max_abs": float((on - off).abs().max()), "token_rms": float(off.pow(2).mean().sqrt()),
                     "bit_identical": bool(torch.equal(on, off))}
    path.unit_sum, path.unit_n, path.out_sq_sum, path.out_n, path.capped_sum, path.rows_n = saved
    out["exact"] = all(out[name]["bit_identical"] for name, _ in modes)
    return out
