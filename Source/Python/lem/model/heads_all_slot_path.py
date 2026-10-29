# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""An all-slot path beside every prediction head's attention (--heads-all-slot-path, default off).

WHAT WAS MEASURED (docs/gain_of_function_v4/restart_candidate/RESTART_CANDIDATE_HEADS_2026-10-04.md, sections 2 and 3;
checkpoints 476,501, 474,875 and 455,800 of H84; registry rows prediction_heads_state_capture and
prediction_heads_state_readings). A prediction head (lem/model/prediction_heads.py) reads the 6 newest tokens of the
transformer (one per slot: vision, tactile, audio, proprioception, motor, vestibular; 192 numbers each), to which the
persistent memory's bias has been added after the transformer's final LayerNorm. One learned query scores the 6 slots,
a softmax turns the scores into weights, the weighted values go through a 2-layer network, and the result is added to
the sense's current token. Three things stop a head from following its input:
  * THE POOL READS ONE SLOT. The memory's bias is the same vector at every model step (its constant) and is 20 to 50
    times the tokens' own size on four slots, so each attention head's best slot leads by tens to hundreds of logits,
    the softmax gives it all the weight and passes no gradient to the query and the keys;
  * WHAT THE FIRST LAYER READS IS THE CONSTANT'S IMAGE. The attended values carry the constant through the value and
    output projections; the part that differs between model steps is a small share of it and, at that size, below the
    bf16 rounding step of the graphics card's arithmetic for the heads that attend the large slots;
  * THE REVIVED FIRST-LAYER UNITS DIVIDE IT AWAY. The units the liveness repair revived read LayerNorm(input - running
    centre); the centre lags the input's shared part, so the LayerNorm divides what differs between model steps (and
    the gradient going back) by the lag's size.
The liveness design (docs/gain_of_function_v3/DESIGN.md section 10) already holds the rules for both faults and the
repair applied them to the encoders' pools and to dead layers, not to the heads' pools: rule 1, a layer reads its input
CENTRED by a running mean and then LayerNormed (a shared direction killed the pose head); rule 3, beside a pool that
reads one element there is a path every element reaches (lem/model/liveness.py AllElementPath, beside the motor and
tactile encoders' pools).

WHAT THIS MODULE ADDS, with the flag on (off builds nothing: an old command builds the model it always built):

  INPUT.   The tensor every prediction head is called with: last_step [B, 6, 192], the transformer's final LayerNorm
           output at the newest model step plus the memory bias. And the memory's constant c [6, 192]: the part of
           that bias every body and model step shares, read from the memory at every call
           (lem/model/motor_input_fold.effective_constant: the readout's ordinary bias times the return's factor),
           without gradient.
  STEP 1.  x = last_step - c, in float32: the transformer's own output (about 5 per number) with the memory's varying
           part (1e-6 to 5e-5 per number on H84, its state being saturated).
  STEP 2.  x - centre, per slot: centre [6, 192] is a running mean of x over the calls that train the heads (the
           horizon of the liveness repair's centres, CENTRE_HORIZON_CALLS). The transformer's output is mostly a
           vector every model step shares (4.7 to 5.4 of its 5.3 per number at 415,075); without the centre the first
           layer's weights would read a constant and act as a bias with a lever of the learning rate times the summed
           input (rule 1's fault).
  STEP 3.  u_j = LayerNorm_h(x_j - centre_j) per slot j and head h (each head has its own LayerNorm gain and bias),
           then v_j = down_h u_j: one 192 -> SLOT_RANK weight matrix per head, the same matrix for the 6 slots, no
           bias (32 numbers per slot on H84).
  STEP 4.  out_h = up_h(GELU(lin_h(concat_j v_j))) x GAIN: every slot enters through its own SLOT_RANK columns of
           lin_h (6 x 32 = 192 -> HIDDEN), so no attention weight decides which slot is read and no softmax can stop
           passing gradient. up_h starts at zero (weight and bias).
  OUTPUT.  PredictionHeads adds out_h to head h's prediction, in float32, after the head's own output and the current
           token (lem/model/prediction_heads.py). With up_h at zero the sum is the prediction of a build without the
           path, bit for bit (the load's switch check reads this under the run's autocast).

THE OLD HEAD STAYS as it is: its weights, its attention and its revived rows compute what they computed, so the switch
changes no prediction, and the two senses whose heads follow their input today (proprioception, audio) keep that. The
old pool keeps reading one slot; what the HEAD reads is the pool and the path together, reported per head as
head_read (the liveness repair's token_read for the encoders' pools: how evenly the head's output answers to the 6
slots).

GRADIENT (version 2, Oct 4, 2026). The path learns from EVERY lesson that trains the head it sits beside: the delayed
next-model-step lesson, its re-study in replay, and the lessons graded on imagined rows (lem.model.predictive_state.
transition: the imagination chain). Inside transition the path runs within imagined(): its weights take the row's
gradient, its running centre does not move (the centre is a mean over lived rows) and no gauge row is kept. The
planner's descent trains nothing: the intention generator freezes the path's weights with the heads' (requires_grad
off), and frozen() remains for the measurement passes that must leave the weights without gradient.
WHY NOT DETACHED ON IMAGINED ROWS, as version 1 had it (a rule taken from the body copy of Oct 1,
docs/gain_of_function_v4/agents/planner_repair_2026-09-30.md sections 11 and 12, without a measurement on this path).
MEASURED Oct 4 (the record's section 13; registry rows heads_all_slot_path_runaway_read, heads_all_slot_path_lesson_forces,
heads_all_slot_path_runaway_reproduction): the imagination chain and the lived lesson grade the same sum, old head +
path, and ask different fixed vectors of it. The lived lesson compares the sum plus the current token with the next
token. The chain passes the sum plus the current token through the encoders' output LayerNorm first and feeds it back
in for 3 depths, at weight 0.5 with no per-sense normalisation. Before the switch the old tactile head's fixed vector
sat between the two requests (0.030 token units per number, 4.3 of 100 of the lived lesson's error). In version 1 the
lived lesson alone trained the path, and a path can output a fixed vector at no cost; it cancelled the old head's on
lived rows. Its output was added on imagined rows too, so the chain's request was unmet again; the chain could not
move the path, so it moved the old head's fixed vector further; the lived lesson moved the path's opposite vector
further. No pair of values meets both lessons that way: in 350 model steps the old head's fixed vector reached 2.78 and
the path's 2.68 with cosine -0.9998, their sum 0.11, and the tactile error rose from 1.03 to 1.85 times the no-change
error. With both parts trained by both lessons the two requests meet at one weighted value of the sum, as they did
for the old head alone.
The memory's constant enters without gradient, so the path sends the memory's bias only what it sends every reader of
last_step: its request on x. A change of the constant does not change x (it is subtracted again at the next call).

Every constant carries its measurement or its arithmetic.
"""
from __future__ import annotations

import contextlib
import math
from collections import deque
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ALL_SLOT_PATH_VERSION = "heads_all_slot_path_v2"
# WHAT AN IMAGINED ROW DOES TO THE PATH'S WEIGHTS (v2, Oct 4, 2026; the record's section 13). "learn": every lesson
# that trains the head a path sits beside trains the path too. "detached" was v1's rule: the imagination chain's loss
# reached the old heads' weights and not the paths', while the paths' output was added to the chain's predictions.
# MEASURED on production (run H84_fresh_drivers_20260918_seg41n_0856, model steps 479,350 to 479,700) and on the CPU
# (registry rows heads_all_slot_path_runaway_read and heads_all_slot_path_runaway_reproduction): the chain and the
# lived lesson grade the same sum (old head + path) and ask different fixed vectors of it (the chain's prediction
# passes the encoders' output LayerNorm, the lived lesson's does not). The path, trained by the lived lesson alone,
# cancelled the old tactile head's fixed vector on lived rows; its output being added on imagined rows too, the
# chain's request was then unmet, its gradient moved the old head's fixed vector further (it could not move the
# path), the lived lesson moved the path's opposite vector further, and no pair of values met both: the old head's
# fixed output went from 0.030 to 2.78 token units per number in 350 model steps and the path's to 2.68 with cosine
# -0.9998 between them. With "learn" both parts receive both lessons' gradient on their sum, as the old head alone
# did before the switch, and a pair of values that meets the two lessons' weighted request exists.
IMAGINED_ROWS = "learn"
MODES = ("off", "on")
SIDE_NAME = "heads_all_slot_path"
SEED_OFFSET = 9_000_009            # the module's own random stream: the run's seed + this
# THE HIDDEN WIDTH is the model's width (192 on H84): the head's own first layer is that wide (lem_model.py overrides
# the config's expansion to 1), so the path can hold what a head's own network holds. Not tuned. The control that
# catches the size lying: the bench's held-out error at half and at twice the width (the record's section 4).
HIDDEN_PER_D_MODEL = 1
# THE NUMBERS KEPT PER SLOT before the first layer. MEASURED (registry row prediction_heads_all_slot_path_bench, second
# round, Oct 4, 2026: checkpoint 479,350's transformer held fixed, 2,800 training windows of 4 bodies, read on 2,100
# windows of 3 other bodies, three seeds, two folds; error = prediction error / no-change error, 1 = no better than
# predicting no change): with every slot's 192 numbers read directly (6 x 192 x 192 = 221,184 first-layer weights per
# head) the path fits its training windows and its held-out error rises from optimizer update 250 on (tactile 1.003
# and 1.002 at update 250, 1.055 and 1.086 at update 1,000; vestibular 1.027 and 1.017, then 1.196 and 1.217). With 32
# numbers per slot (6,144 + 36,864 weights) the same readings are tactile 0.997 and 0.995, then 0.994 and 1.013;
# vestibular 1.005 and 1.006, then 1.070 and 1.092; proprioception's own-command reading is the same within its seed
# spread (0.274 and 0.273 against 0.285 and 0.284 at update 250). The seed spread of the errors is 0.001 to 0.015.
SLOT_RANK = 32
# THE RUNNING CENTRE'S HORIZON, in calls that train the heads: lem/model/liveness.py CENTRE_HORIZON_STEPS (2,000) and
# its arithmetic (the part the centre removes is shared by every model step, so any horizon longer than a few model
# steps finds it; 2,000 follows the slow drift of the weights). The control: window_summary's centre_lag.
CENTRE_HORIZON_CALLS = 2000
GAUGE_CALLS = 8                    # the newest lesson calls the window's gauges read (8 calls x 8 bodies = 64 rows)
GAUGE_ROWS = 64
LIVE_LINE = -3.0                   # DESIGN section 10: a unit is live when its input rose above this


def output_gain(hidden: int) -> float:
    """THE OUTPUT'S GAIN: out = up(GELU(h)) / sqrt(hidden). Why: Adam moves every weight of `up` by about the
    learning rate per optimizer update whatever the gradient's size, and on one row all of a number's `hidden`
    weights can move the same way, so the output can change by up to learning rate x sum|GELU(h)| per update:
    3e-4 x 192 x 0.41 (the mean of |GELU| of a standard normal input is 0.41) = 0.024 token units per update without
    the gain, 0.0017 with it. The real change of a token from one model step to the next is 0.02 to 0.3 per number
    (the readings' real_change), so without the gain 1 to 13 updates of noise are as large as the whole target. The
    same rule as lem/model/motor_asked_change.output_gain. The control: window_summary's output_rms per report."""
    return 1.0 / math.sqrt(float(hidden))


def liveness_layer(head: str) -> str:
    """The hidden layer's name for lem/model/liveness.py (layer_parts resolves it on the model)."""
    return "%s.paths.%s.net.0" % (SIDE_NAME, head)


class _HeadPath(nn.Module):
    def __init__(self, n_slots: int, d_model: int, hidden: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank, bias=False)          # one matrix for the 6 slots (STEP 3)
        self.net = nn.Sequential(nn.Linear(n_slots * rank, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)


class HeadsAllSlotPath(nn.Module):
    """The all-slot paths of every prediction head (the module text). forward(last_step) returns {head: [B, d]} in
    token units, float32; attach(prediction_heads) makes PredictionHeads add them."""

    def __init__(self, head_modalities: Sequence[str], n_slots: int, d_model: int, hidden: Optional[int] = None,
                 rank: Optional[int] = None):
        super().__init__()
        self.head_modalities = list(head_modalities)
        self.n_slots = int(n_slots)
        self.d_model = int(d_model)
        self.hidden = int(hidden) if hidden else HIDDEN_PER_D_MODEL * self.d_model
        self.rank = min(int(rank) if rank else SLOT_RANK, self.d_model)
        self.paths = nn.ModuleDict({m: _HeadPath(self.n_slots, self.d_model, self.hidden, self.rank)
                                    for m in self.head_modalities})
        self.gain = output_gain(self.hidden)
        # the running centre of x = last_step - constant, per slot; saved with the weights (a checkpoint of the path
        # carries it), never trained
        self.register_buffer("centre_mean", torch.zeros(self.n_slots, self.d_model))
        self.register_buffer("centre_weight", torch.zeros((), dtype=torch.float64))
        self.constant_fn: Optional[Callable[[], Optional[torch.Tensor]]] = None
        self.measuring_fn: Optional[Callable[[], bool]] = None
        self.gauge_calls: deque = deque(maxlen=GAUGE_CALLS)
        self.gauge_armed = False
        self.attached = False
        self.bypass = False               # tests and the switch check: PredictionHeads adds nothing while True
        self.frozen_depth = 0             # above 0 (inside frozen()): the path's weights enter detached
        self.imagined_depth = 0           # above 0 (inside imagined()): an imagined row; the centre stays where it is
        # what an imagined row does to the path's weights (IMAGINED_ROWS above): "learn" or "detached"
        self.imagined_rows = IMAGINED_ROWS
        self._detach = lambda: None

    # ------------------------------------------------------------------ the path
    @contextlib.contextmanager
    def frozen(self):
        """Inside: the path computes the same numbers with its weights detached and its centre unmoved, so nothing
        graded inside reaches them; the row keeps its gradient. Used by the measurement passes that take a gradient
        on the rows (the stats block's lesson_gradient); version 1 used it for every imagined row. Nests."""
        self.frozen_depth += 1
        try:
            yield self
        finally:
            self.frozen_depth -= 1

    @contextlib.contextmanager
    def imagined(self):
        """Inside: the call is an imagined row (lem.model.predictive_state.transition: the imagination chain, the
        planner's descent and plan chain, the probes' roll-outs). With imagined_rows = "learn" (the rule since v2, see
        IMAGINED_ROWS) the path computes with its own weights, so a lesson graded on an imagined row trains them exactly
        as it trains the weights of the head the path sits beside; the running centre does not move (it is a mean
        over lived rows only) and no gauge row is kept. With "detached" this is frozen(). Nests."""
        if self.imagined_rows == "detached":
            with self.frozen():
                yield self
            return
        self.imagined_depth += 1
        try:
            yield self
        finally:
            self.imagined_depth -= 1

    def constant(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        c = self.constant_fn() if self.constant_fn is not None else None
        if c is None:
            return None
        return c.detach().to(device=like.device, dtype=torch.float32).reshape(self.n_slots, self.d_model)

    def centred_input(self, last_step: torch.Tensor) -> torch.Tensor:
        """x = last_step - constant, float32 [B, n_slots, d] (STEP 1)."""
        if last_step.dim() != 3 or last_step.shape[1] != self.n_slots or last_step.shape[2] != self.d_model:
            raise ValueError("the all-slot path was built for %d slots of %d numbers; the heads were called with %s"
                             % (self.n_slots, self.d_model, tuple(last_step.shape)))
        x = last_step.float()
        c = self.constant(x)
        return x if c is None else x - c

    @torch.no_grad()
    def update_centre(self, x: torch.Tensor) -> None:
        """One call's rows (pooled) into the running mean; the first call sets it (lem/model/liveness.py
        RevivedRows.update_centre's arithmetic: an exponential average divided by the weight it has accumulated)."""
        a = 1.0 / float(CENTRE_HORIZON_CALLS)
        m = x.detach().float().mean(dim=0).to(self.centre_mean)
        w_old = float(self.centre_weight)
        w_new = (1.0 - a) * w_old + a
        self.centre_mean.mul_((1.0 - a) * w_old / w_new).add_(m * (a / w_new))
        self.centre_weight.fill_(w_new)

    def _learning_call(self) -> bool:
        """A call that trains the heads: gradient on, training mode, weights not frozen (the planner's descent
        freezes them through requires_grad, an imagined row through frozen()), and not a measurement pass (the
        liveness repair's unwatched block)."""
        if self.frozen_depth or self.imagined_depth or not torch.is_grad_enabled() or not self.training:
            return False
        if self.measuring_fn is not None and self.measuring_fn():
            return False
        first = self.paths[self.head_modalities[0]].net[0].weight
        return bool(first.requires_grad)

    def head_output(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """out [B, d] for head `name` from x [B, n_slots, d] (STEPS 2 to 4), float32."""
        p = self.paths[name]
        xc = x - self.centre_mean
        if self.frozen_depth:
            from torch.func import functional_call
            u = F.layer_norm(xc, (self.d_model,), p.norm.weight.detach(), p.norm.bias.detach(), p.norm.eps)
            u = F.linear(u, p.down.weight.detach())
            lin = p.net[0]
            # the hidden layer is called as a module (so the liveness monitor's and repair's hooks on it run as in
            # every other call) with its weight and bias swapped for detached ones
            h = functional_call(lin, {"weight": lin.weight.detach(), "bias": lin.bias.detach()},
                                (u.reshape(u.shape[0], -1),))
            return F.linear(p.net[1](h), p.net[2].weight.detach(), p.net[2].bias.detach()) * self.gain
        u = p.down(p.norm(xc))
        return p.net(u.reshape(u.shape[0], -1)) * self.gain

    @torch.no_grad()
    def hidden_input(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """The hidden layer's input [B, hidden] for head `name` from x [B, n_slots, d] (the first layer's own weights,
        no hook): what the output layer's growth bound is measured from."""
        p = self.paths[name]
        u = F.linear(F.layer_norm(x - self.centre_mean, (self.d_model,), p.norm.weight, p.norm.bias, p.norm.eps),
                     p.down.weight)
        return F.linear(u.reshape(u.shape[0], -1), p.net[0].weight, p.net[0].bias)

    def forward(self, last_step: torch.Tensor) -> Dict[str, torch.Tensor]:
        """{head: [B, d]} float32, computed outside the run's autocast (float32 throughout: the path's inputs are of
        size 1 after the LayerNorm, so bf16 would cost nothing here either; float32 keeps the switch check exact and
        the centre's subtraction free of the constant's rounding)."""
        dev = last_step.device.type
        with torch.autocast(device_type=dev, enabled=False):
            x = self.centred_input(last_step)
            if self._learning_call():
                self.update_centre(x)
            if self.gauge_armed:
                self.gauge_calls.append(x.detach().clone())
                self.gauge_armed = False
            return {m: self.head_output(m, x) for m in self.head_modalities}

    # ------------------------------------------------------------------ the attachment
    def attach(self, prediction_heads: nn.Module, constant_fn=None, measuring_fn=None) -> None:
        """PredictionHeads reads the path from its instance dict (a plain reference, not a registered submodule: the
        heads' state dict and checkpoint keys are unchanged) and adds its output to every head's prediction."""
        if self.attached:
            raise RuntimeError("the all-slot path is attached already")
        heads = getattr(prediction_heads, "_orig_mod", prediction_heads)
        if list(heads.head_modalities) != self.head_modalities:
            raise ValueError("the all-slot path was built for heads %s; the model's heads are %s"
                             % (self.head_modalities, list(heads.head_modalities)))
        heads.__dict__["all_slot_path"] = self
        self.constant_fn = constant_fn
        self.measuring_fn = measuring_fn

        def remove():
            heads.__dict__.pop("all_slot_path", None)
        self._detach = remove
        self.attached = True

    def detach(self) -> None:
        self._detach()
        self._detach = lambda: None
        self.attached = False

    # ------------------------------------------------------------------ records
    def stamp(self) -> dict:
        return {"version": ALL_SLOT_PATH_VERSION, "mode": "on", "imagined_rows": self.imagined_rows,
                "heads": list(self.head_modalities),
                "slots": self.n_slots, "hidden": self.hidden, "slot_rank": self.rank, "output_gain": self.gain,
                "centre_horizon_calls": CENTRE_HORIZON_CALLS,
                "liveness_layers": [liveness_layer(m) for m in self.head_modalities]}

    def arm_gauge(self) -> None:
        """The next call keeps its input x for the window's gauges. LemLite.step calls this right before the delayed
        lesson's call of the heads, so the gauges read the lesson's own rows and add no pass through a hooked layer."""
        self.gauge_armed = True

    def gauge_rows(self) -> Optional[torch.Tensor]:
        calls = [c for c in self.gauge_calls]
        if not calls:
            return None
        return torch.cat(calls, dim=0)[-GAUGE_ROWS:]

    @torch.no_grad()
    def window_summary(self) -> dict:
        """The `heads_all_slot_path` stats block; a fresh window starts. Read on the newest lesson calls the gauge
        kept (arm_gauge), with the weights as they are at the report and no pass through any hooked layer:
          rows            body rows the gauges read; 0 = no lesson call in the window (the per-head fields are absent);
          centre_weight   the running centre's accumulated weight (1 = the horizon is filled);
          input_rms       root mean square per number of x = last_step - constant, per slot;
          moment_rms      root mean square per number of what differs between the rows (x minus the rows' mean);
          centre_lag      per slot, the length of (the rows' mean of x - centre) over the mean length of a row's own
                          departure from the rows' mean: 0 = the centre sits on the rows' mean, 1 = the offset is as
                          large as what differs between rows;
          per head h:
            output_weight_rms  root mean square of up_h's weights (0 = the switch);
            output_rms         root mean square per number of the path's output on those rows, token units;
            output_fixed_rms   root mean square per number of the MEAN of that output over the rows (192 numbers: the
                               part every row shares); 0 = the path adds no fixed vector, equal to output_rms = it
                               adds nothing else;
            slot_reach         how evenly the path's output answers to the 6 slots: per slot, the root mean square
                               change of the output when that slot's row is replaced by the next body's, as shares;
                               spread = their entropy over log(6) (1 = even, 0 = one slot); absent while up_h is zero;
            live               share of the hidden units whose input rose above -3 on those rows;
            varying            share whose GELU output range exceeds lem/model/liveness.py's floor."""
        out = {"version": ALL_SLOT_PATH_VERSION, "rows": 0, "centre_weight": round(float(self.centre_weight), 6)}
        x = self.gauge_rows()
        self.gauge_calls.clear()
        heads = {}
        for m in self.head_modalities:
            heads[m] = {"output_weight_rms": float(self.paths[m].net[2].weight.float().pow(2).mean().sqrt())}
        out["heads"] = heads
        if x is None or x.shape[0] < 2:
            return out
        n = x.shape[0]
        out["rows"] = int(n)
        mean = x.mean(dim=0)
        dev = x - mean
        out["input_rms"] = [round(float(x[:, j].pow(2).mean().sqrt()), 4) for j in range(self.n_slots)]
        out["moment_rms"] = [round(float(dev[:, j].pow(2).mean().sqrt()), 4) for j in range(self.n_slots)]
        lag = (mean - self.centre_mean).norm(dim=-1)
        own = dev.norm(dim=-1).mean(dim=0).clamp_min(1e-12)
        out["centre_lag"] = [round(float(v), 4) for v in (lag / own)]
        from lem.model.liveness import varying_floor
        xc = x - self.centre_mean
        for m in self.head_modalities:
            p = self.paths[m]
            u = F.linear(F.layer_norm(xc, (self.d_model,), p.norm.weight.float(), p.norm.bias.float(), p.norm.eps),
                         p.down.weight.float())

            def run(uu):
                h = F.linear(uu.reshape(n, -1), p.net[0].weight.float(), p.net[0].bias.float())
                return h, F.linear(F.gelu(h), p.net[2].weight.float(), p.net[2].bias.float()) * self.gain
            h, o = run(u)
            H = heads[m]
            H["output_rms"] = float(o.pow(2).mean().sqrt())
            H["output_fixed_rms"] = float(o.mean(dim=0).pow(2).mean().sqrt())
            H["live"] = round(float((h > LIVE_LINE).any(dim=0).float().mean()), 4)
            post = F.gelu(h)
            lo, hi = post.min(dim=0).values, post.max(dim=0).values
            H["varying"] = round(float(((hi - lo) > varying_floor(lo, hi, "gelu").to(hi)).float().mean()), 4)
            if H["output_weight_rms"] > 0:
                change = []
                for j in range(self.n_slots):
                    us = u.clone()
                    us[:, j] = u.roll(1, dims=0)[:, j]
                    change.append(float((run(us)[1] - o).pow(2).mean().sqrt()))
                total = sum(change)
                if total > 0:
                    share = [c / total for c in change]
                    ent = -sum(s * math.log(max(s, 1e-12)) for s in share) / math.log(self.n_slots)
                    H["slot_reach"] = {"spread": round(ent, 4), "largest_share": round(max(share), 4),
                                       "shares": [round(s, 4) for s in share]}
        return out


def build(head_modalities: Sequence[str], n_slots: int, d_model: int, seed: int) -> HeadsAllSlotPath:
    """The paths, built in their own random stream (the run's stream is left where a build without them leaves it)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + SEED_OFFSET)
        return HeadsAllSlotPath(head_modalities, n_slots, d_model)


@torch.no_grad()
def head_read(prediction_heads, x_last: torch.Tensor, current: Dict[str, torch.Tensor]) -> dict:
    """How evenly each head's WHOLE prediction (its attention pool and the all-slot path together) answers to the 6
    slots, on rows x_last [n, 6, d] (the tensor the heads are called with): per slot, the root mean square change of
    the prediction when that slot's row is replaced by the next body's row; as shares, their entropy over log(6)
    (spread: 1 = even, 0 = one slot) and the largest share. The caller runs it inside the liveness repair's unwatched
    block and under the run's autocast, so the old head's part is computed as the run computes it."""
    heads = getattr(prediction_heads, "_orig_mod", prediction_heads)
    n, S = x_last.shape[0], x_last.shape[1]
    base = heads(x_last, current=current, who="heads_all_slot_path.head_read")
    change = {m: [] for m in base}
    for j in range(S):
        xs = x_last.clone()
        xs[:, j] = x_last.roll(1, dims=0)[:, j]
        got = heads(xs, current=current, who="heads_all_slot_path.head_read")
        for m in base:
            change[m].append(float((got[m].float() - base[m].float()).pow(2).mean().sqrt()))
    out = {}
    for m, ch in change.items():
        total = sum(ch)
        if not total > 0:
            out[m] = {"spread": 0.0, "largest_share": None, "shares": None, "rows": int(n)}
            continue
        share = [c / total for c in ch]
        ent = -sum(s * math.log(max(s, 1e-12)) for s in share) / math.log(S)
        out[m] = {"spread": round(ent, 4), "largest_share": round(max(share), 4),
                  "shares": [round(s, 4) for s in share], "rows": int(n)}
    return out


@torch.no_grad()
def fixed_parts(prediction_heads, x_last: torch.Tensor, current: Dict[str, torch.Tensor]) -> dict:
    """THE TWO FIXED VECTORS of each head, on rows x_last [n, 6, d] (the tensor the heads are called with) with their
    current tokens. INPUT: the old head's output on each row (its prediction with the path bypassed, minus the current
    token when the heads predict the change) and the path's output on each row. OPERATION: the mean of each over the
    rows (192 numbers per head: what the part adds to every row alike). OUTPUT per head: old_rms and path_rms (root
    mean square per number of the two mean vectors, token units), sum_rms (of the mean of old + path, the fixed part
    of what the lessons grade) and cosine (between the two mean vectors: -1 = the path's fixed vector is the old
    head's reversed, 0 = unrelated, 1 = the same direction).
    WHY IT IS READ (the record's section 13): in the stopped run of Oct 4 the old tactile head's fixed vector and the
    tactile path's grew against each other (2.78 and 2.68 token units per number, cosine -0.9998) while their sum
    stayed at 0.11; the path's output_rms alone reads the size, these fields read the pair. The caller runs it inside
    the liveness repair's unwatched block and under the run's autocast; nothing learns and no centre moves (no_grad)."""
    heads = getattr(prediction_heads, "_orig_mod", prediction_heads)
    path = heads.__dict__.get("all_slot_path")
    if path is None:
        return {}
    path.bypass = True
    try:
        old = heads(x_last, current=current, who="heads_all_slot_path.fixed_parts")
    finally:
        path.bypass = False
    add = path(x_last)
    residual = bool(getattr(heads, "residual", False))
    out = {}
    for m in path.head_modalities:
        o = old[m].float()
        if residual:
            o = o - current[m].float().reshape(o.shape)
        ov, pv = o.mean(dim=0), add[m].float().mean(dim=0)
        on, pn = float(ov.norm()), float(pv.norm())
        out[m] = {"old_rms": float(ov.pow(2).mean().sqrt()), "path_rms": float(pv.pow(2).mean().sqrt()),
                  "sum_rms": float((ov + pv).pow(2).mean().sqrt()),
                  "cosine": (round(float(ov @ pv) / (on * pn), 4) if on > 0 and pn > 0 else None),
                  "rows": int(x_last.shape[0])}
    return out


@torch.no_grad()
def switch_check(model, rows: int = 64, seed: int = 0) -> dict:
    """What switching the path on does to the predictions, measured on `rows` probe rows: the largest absolute change
    of each head's prediction with the path against without it, in float32 and, when the run computes under autocast,
    under the run's autocast. With every up_h at zero both are exactly 0.0. Probe rows: tokens of the size the heads
    read (the memory's constant plus a standard normal draw times 5), from a generator of its own; the liveness
    repair's watching windows and running centres are not touched (unwatched), nor is the path's centre (no_grad)."""
    path = getattr(model, SIDE_NAME, None)
    if path is None:
        raise ValueError("the model was built without --heads-all-slot-path on")
    heads = getattr(model.prediction_heads, "_orig_mod", model.prediction_heads)
    dev = next(path.parameters()).device
    g = torch.Generator().manual_seed(int(seed) + SEED_OFFSET)
    x = torch.randn(rows, path.n_slots, path.d_model, generator=g) * 5.0
    c = path.constant(x)
    x = x.to(dev) + (0.0 if c is None else c.to(dev))
    cur = {m: torch.randn(rows, path.d_model, generator=g).to(dev) for m in model.active}
    out = {"rows": int(rows),
           "output_weight_max_abs": max(float(p.net[2].weight.abs().max()) for p in path.paths.values()),
           "output_bias_max_abs": max(float(p.net[2].bias.abs().max()) for p in path.paths.values())}
    from lem.model import liveness as LV

    def preds(bypass, amp):
        path.bypass = bypass
        try:
            with LV.unwatched(model), torch.amp.autocast("cuda", enabled=amp, dtype=getattr(model, "dtype", torch.float32)):
                got = heads(x, current=cur, who="heads_all_slot_path.switch_check")
                return {m: v.float() for m, v in got.items()}
        finally:
            path.bypass = False
    modes = [("float32", False)] + ([("autocast", True)] if getattr(model, "use_amp", False) else [])
    for name, amp in modes:
        on, off = preds(False, amp), preds(True, amp)
        out[name] = {"change_max_abs": max(float((on[m] - off[m]).abs().max()) for m in on),
                     "bit_identical": all(bool(torch.equal(on[m], off[m])) for m in on),
                     "same_dtype": True}
    out["exact"] = all(out[name]["bit_identical"] for name, _ in modes)
    return out
