# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Learned recurrent/event memory beyond the transformer's direct observation window.

History stores observed inputs only. A recurrent encoder is re-run with gradients
through retained events; its evicted prefix is a detached recurrent carry, as in
truncated backpropagation. Query attention can retrieve an old event. Residual
readout starts at zero, so installation preserves the parent's computation.

THE CENTRED REPAIR (--persistent-memory-repair centred, default "off" = the memory every
earlier build computed; docs/gain_of_function_v4/agents/persistent_memory_2026-09-25.md).
Measured on H84's 226 saved checkpoints (docs/gain_of_function_v3/measurements/
persistent_memory_lineage_2026-09-25.json, weights and saved memory histories, no model):
  - the events the memory reads are mostly the part every moment and body shares (0.66-0.80
    root mean square per number against 0.38-0.48 for the moment-specific part), and the input
    projection turns that shared part into pre-activations of 6.8-16.7 (tanh passes 0.99 at
    2.65), so 92-99.9% of its outputs sat at +-1 from the first checkpoint on;
  - the GRU's candidate has a recurrent block with a gain above 1 from the start (largest
    singular value 1.7 at update 3,764, 4.5 at 344,400, 5.0 now), so units hold their own sign
    (a latch), and its median pre-activation grew from 1.6 to 5.0 before the Sept 23 deploy and
    to 8.9 within 15,000 updates after it, where tanh passes a gradient of 1e-7: from update
    379,324 every body's 192 state numbers are one fixed pattern (spread between bodies 7e-7)
    and the GRU no longer learns;
  - the readout turns the fixed +-1 state into a bias whose constant moves about 190 times
    faster than an ordinary bias (the Adam step the saved moments imply moves it 165-196 times
    further through the readout's weights than through its own bias vector): 5.7 to 147.
The repair keeps what every head reads at the switch and removes the mechanisms:
  1. THE CONSTANT BECOMES AN ORDINARY BIAS, COMPUTED IN FP32. At the switch the readout's bias takes
     the bias the heads read before it, computed the way production computes it (the readout under the
     trainer's autocast: bf16 on the graphics card, rounded as there), per number the value one of the
     bodies reads (the median over the bodies), and the readout's weight is set to 0. From then on the
     weight multiplies the CENTRED state (the state minus its running mean over lived moments) and the
     whole bias is computed in fp32 outside autocast, so the heads' inputs after the switch equal those
     before it bit for bit, and a small varying part is not rounded onto the constant's bf16 grid (at a
     constant of 147 per number that grid is 0.25 to 4.0 apart). The needs forecast reads
     needs_gain * centred state (needs_gain starts at 0) with its old state columns' share folded into
     its first layer's bias.
  2. THE INPUT'S SHARED PART IS REMOVED: the input projection reads the events minus their running
     mean and its pre-activation is divided by its root mean square over the 96 units.
  3. THE CANDIDATE'S TWO PARTS HAVE SIZES NO GRADIENT CAN CHANGE: the input part W_in [x; 1] and the
     recurrent part W_hn [h; 1] are each multiplied, inside the graph, by target / sqrt(tr(W M W^T) / 96),
     M the running second moment of what the part reads (the codes [x; 1], the previous states [h; 1];
     200 lived calls, no gradient). Over the data the part's root mean square is then the target (1.0 for
     the input part, 0.5 for the recurrent part) whatever the stored rows' size, and since the loss does
     not depend on that size its gradient has no outward part (Euler's theorem for a function unchanged
     by scaling W): a lesson cannot grow the stored rows and shrink the scale, which cut the candidate's
     learning rate 5.5 to 12 times within 4,000 updates under the first version's running-size cap (the
     review's stress test). The switch multiplies the stored rows by their scales once, so they start at
     their targets and Adam's step of about the learning rate per stored number is one per effective
     number (H84's candidate bias of about 30 held its units at +-1: stored that large, the rows would
     learn about 30 times slower). The candidate's root mean square is then at most 1.5: at most
     (1.5 / 8.4)^2 = 3.2 in 100 of its values can pass 8.4, the frozen regime (Chebyshev). The fused GRU
     call (cuDNN on the card) is kept: the effective rows are passed to it.
  4. A REQUEST EVERY BODY SHARES AND THAT PERSISTS GOES TO THE ORDINARY BIAS, NOT TO THE STATE. A lesson's
     request is its gradient on what it reads from the memory: the memory bias (bias(), per call site: the lived
     moment, the delayed lesson, the tenure lesson, replay) and the needs forecast's state slot (predict_needs).
     From each request the running mean over that site's backward calls (REQUEST_CENTRE_HORIZON) of the request's
     mean over the rows (the bodies) is removed before it reaches the readout's weight and the state; the
     readout's ordinary bias still receives the whole request (backward only: the forward is the readout's own
     F.linear, and no body reads another's events). What the Sept 23 run-away was - a lesson asking every body's
     memory for the same thing, update after update - then moves only the ordinary bias; what every body is asked
     alike at one moment and not the next, a signal the worlds meet at the same time, still teaches the memory.
     The centre is taken on the request, before the readout's weight, because that weight grows as it learns: a
     running mean taken on the state's gradient (the first draft of this version) lags a request that grows with
     the weight, and in the stress test that lag let a constant request through as if nothing were centred. The
     centre is the running mean corrected for its lag (twice the running mean less the running mean of the
     running mean), so a request that changes at a steady rate - as one does while the ordinary bias learns what
     it asks for - is followed without lag, and it is exact for a request that does not change: Adam divides a
     gradient by its own running size, so a residual of one sign at every update, however small, moves the
     weights at a large share of their full step (a plain running mean trails a steady change by the horizon
     times its rate).
     Version 2 removed the state gradient's mean over the bodies at every moment; the second review measured that
     a lesson built on a signal every body meets at the same moment then cannot be learned and pulls the bodies
     together (spread between bodies 0.18 to 0.034 in 2,000 updates), and that 5 in 100 of H84's memory input
     varies that way (touch 18 in 100, motor 14). The first call of a site removes its own shared part whole (the
     running mean starts from it).
  Measured: docs/gain_of_function_v4/persistent_memory_stress.py -> docs/gain_of_function_v3/measurements/
  persistent_memory_stress_2026-09-25.json (version 2) and persistent_memory_stress_v3_2026-09-25.json (this
  version and its controls, with the synchronized lesson): the record's section 5.
The state the memory hands out (encode's first result, what the replay ring stores, what
bias() and predict_needs() read) is the centred state; the ring's saved states are
converted at the switch (state - state_ref), after a check that no written entry's bias
would change by more than the line, so replay reads the same constant.
The switch happens at the start of the first training step after a load that holds a lived
moment (or at the lived call of an empty history) (LemLite._persistent_memory_switch). It is
planned first (prepare_switch: every reading and every refusal, nothing written) and then
applied (apply_switch: copies only), so a refusal or an error leaves the memory as it was. It
refuses when the saved state is not one constant, restarts readout.weight's Adam moments,
moves the liveness repair's centre of the needs forecast's revived units, and is recorded in
the checkpoint (repair_switched), the receipt and the stats stream. A build without the flag
refuses such a checkpoint and names the only way back: the checkpoint saved before the switch.
"""
from __future__ import annotations
from collections import deque
import math
import torch
from torch import nn
import torch.nn.functional as F

MEMORY_REPAIR_CHOICES = ("off", "centred")
# v2 (2026-09-25, after the independent review of v1): the fold takes production's own (bf16) number and the
# bias is computed in fp32; the candidate's sizes are set inside the graph (no outward gradient) instead of by
# a running-size cap; the state's gradient has its mean over the bodies removed; the switch is planned before
# anything is written and checks the replay ring. No production checkpoint was ever switched under v1.
# v3 (2026-09-25, after the second review): each lesson's request on the memory loses the running mean over its
# call site's updates of its mean over the bodies, taken before the readout's weight (not that mean at every
# moment, and not on the state's gradient); the encoder's Adam moments restart at the switch; the ring check reads
# production's arithmetic; the goal reading (the next-moment error with and without the memory's varying part).
# v4 (2026-09-26, after the third review and the coordinator's scale-drift survey): the folded constant returns to
# its pre-deploy size per sense slot along a geometric schedule after the switch and is capped there (THE CONSTANT'S
# RETURN below); the goal reading gains the other-body arms. A checkpoint switched by v3 would resume here without
# the return's buffers, so it is refused by version like the others (LemLite.load_checkpoint).
MEMORY_REPAIR_VERSION = "persistent_memory_centred_v4"
SWITCHED_KEY = "repair_switched"
# every buffer and parameter enable_repair() adds (a build without the repair has none of them)
REPAIR_BUFFERS = (SWITCHED_KEY, "state_ref", "event_centre", "event_centre_weight", "state_centre",
                  "state_centre_weight", "codes_moment", "hidden_moment", "moment_weight", "bias_at_switch",
                  "scales_at_switch", "request_centre", "request_centre_weight", "needs_request_centre",
                  "needs_request_centre_weight", "constant_start", "constant_target", "constant_return_updates",
                  "updates_since_switch")
REPAIR_PARAMETERS = ("needs_gain",)
# the running means' horizon in lived model steps: lem/model/liveness.py CENTRE_HORIZON_STEPS and
# lem/model/contact_head.py use 2,000 for the same job (the part a centre removes is shared by every
# moment, so any horizon longer than a few moments finds it; 2,000 follows the slow drift of the weights)
CENTRE_HORIZON_STEPS = 2000
# the floor under a root mean square before dividing by it (float32 values of order 1)
RMS_FLOOR = 1e-6
# THE CANDIDATE'S TWO PARTS' SIZES (root mean square over the 96 units and the data the part reads). The
# input part 1.0: the scale of a LayerNorm-GRU's normalised pre-activation. The recurrent part half of that:
# the lineage's own reading when the bodies' states still differed (update 67,934: recurrent 1.35 against
# input 2.57, 0.53 of it, bodies 0.19 apart; 344,400: 0.66 of it, 0.08 apart; frozen from 380,200: 0.89 of it,
# 7e-7 apart). On the 78 saved checkpoints since the freeze, the repaired memory at once held 1-13 in 100 of
# its numbers above 0.99 with the recurrent part at half the input's, 6-32 at the input's size and 18-47
# uncapped (persistent_memory_lineage_2026-09-25.json, repair_at_switch).
CANDIDATE_INPUT_RMS = 1.0
CANDIDATE_RECURRENT_RMS = 0.5
# the horizon of the second moments that set the candidate's scales, in lived calls. One call's reading is
# already an average over every (event, body) - 131 x 8 = 1,048 rows of 97 numbers on H84 - so the horizon
# only has to follow the drift of the codes and the states.
SCALE_HORIZON_STEPS = 200
# THE REQUEST'S CENTRE: the memory bias's call sites whose shared request is tracked apart (their requests mean
# different things; the needs forecast's slot has its own), and the horizon of the running mean in that site's
# backward calls (one per update for the lived moment, the delayed lesson and the needs forecast; one per ended
# tenure for "tenure"; one per replay batch for "replay"). A shared request that persists longer than the horizon
# goes to the ordinary bias; one that changes faster still teaches the memory. One model step is 0.05 s of
# simulated time on H84, so 200 updates are 10 s: a signal every world meets at once and that changes within
# 10 s is learned. The first call of a site removes its shared part whole; a request that appears later leaks
# while the centre settles on it (for about the horizon), and the lag-corrected centre then overshoots by as
# much the other way, so the leak's signed sum over the calls is 0.
REQUEST_SITES = ("lived", "delayed", "tenure", "replay", "other")
REQUEST_CENTRE_HORIZON = 200
# THE SWITCH'S EXACTNESS LINE: the largest change of any bias number the fold may make, as a share of the
# largest bias number, above a floor of 2e-5 (the frozen vision audit's start limit, scaled with the largest
# value above 16: docs/gain_of_function_v3/audit_visual_regression_2026_09_24.py). It holds three readings:
# the fp32 fold against every body's fp32 bias (the saved state must be one constant; H84 at 411,150: 0.00027
# against a line of 0.00078; at 344,400, bodies 0.08 apart, refused), production's own bias against every
# body's (bf16 on the card; a body whose number lies across a rounding boundary from the others' would move
# by one bf16 step), and every written replay-ring entry's fp32 bias against the fold.
SWITCH_LIMIT = 2e-5
SWITCH_LIMIT_SCALE_FROM = 16.0
# THE CONSTANT'S RETURN (2026-09-26). The scale-drift survey (branch fix/self-percept-runaway-2026-09-26,
# docs/gain_of_function_v4/measurements/self_percept_runaway/scale_survey_results.json) found one layer that grew:
# this memory bias, added after the transformer's final LayerNorm (whose output is about 5 per slot). Its size per
# sense slot (root mean square over the 8 bodies and 192 numbers) at 344,400, the last checkpoint before the Sept 23
# deploy: the targets below (the sixth slot was the intermind placeholder then and is the vestibular sense now); at
# 415,075: touch 268, proprioception 208, motor 128, vision 106, audio 5.6, vestibular 6.7. The switch folds that
# constant whole, so every head reads what it read before; from the first update after it, each slot's constant is
# scaled inside the graph to at most R(t) = start x (target / start) ^ min(1, t / T) (t: model updates since the
# switch; start: the slot's size at the switch), so it shrinks geometrically to its pre-deploy size over T updates
# and no gradient can grow it past the cap afterwards; a slot at or under its target is never shrunk and is capped
# at its target. T has no default: the trainer's --persistent-memory-constant-return names it at the launch that
# switches (0 keeps the constant as folded), because the heads as they are cannot absorb any measured part of the
# return without training (docs/gain_of_function_v4/persistent_memory_constant_return.py on 415,075: moving every
# constant 2 in 100 of the way raised vision's next-moment error by a factor of about 3,400), so the length is set
# from a training measurement (the switch trial's return arm; the record's section 12).
PRE_DEPLOY_BIAS_RMS = dict(vision=5.50, tactile=6.11, audio=5.80, proprioception=5.10, motor=5.17,
                           vestibular=6.18, intermind=6.18)
TANH_099 = 2.647                    # |tanh(a)| > 0.99
TANH_1EM7 = 8.4                     # 1 - tanh(8.4) = 1e-7: the frozen regime
# the gauges the repaired memory adds up on the device at every lived call and reads once per report
# (gauge_report): the stats record's persistent_memory block carries the report window's mean of each
GAUGES = ("state_share_above_0p99", "state_spread_between_bodies", "state_spread_over_window",
          "centred_state_rms", "varying_bias_rms", "bias_spread_between_bodies", "constant_bias_rms",
          "constant_bias_moved_rms", "needs_gain_mean_abs", "candidate_rms", "candidate_share_above_tanh_0p99",
          "candidate_share_above_tanh_1em7", "update_gate_share_above_0p99", "candidate_input_scale",
          "candidate_recurrent_scale", "candidate_input_stored_size", "candidate_recurrent_stored_size",
          "candidate_input_rows_norm", "candidate_recurrent_rows_norm")


def _rms_units(x: torch.Tensor) -> torch.Tensor:
    """x divided by its root mean square over the last axis (per body and event)."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_FLOOR ** 2)


def switch_limit(largest: float) -> float:
    return SWITCH_LIMIT * max(1.0, float(largest) / SWITCH_LIMIT_SCALE_FROM)


class _BodyCentredGrad(torch.autograd.Function):
    """Identity forward; backward removes the gradient's mean over the bodies (the first axis), so the
    memory's state is trained only on what the lessons ask differently of different bodies. One body:
    unchanged."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        if g.shape[0] < 2:
            return g
        return g - g.mean(0, keepdim=True)


def body_centred_grad(x: torch.Tensor) -> torch.Tensor:
    """Version 2's centring (a measurement control since version 3)."""
    return _BodyCentredGrad.apply(x)


def _shared_rows(rows: torch.Tensor) -> torch.Tensor:
    """The rows' mean over the first axis, and exactly the rows' common value wherever every row holds the same
    number (a mean of identical float numbers can round away from them when their count is not a power of 2). On
    the device, no read."""
    m = rows.mean(0)
    return torch.where((rows == rows[:1]).all(0), rows[0], m)


class _RunningCentre(torch.autograd.Function):
    """Identity forward. Backward removes from every row's gradient the RUNNING MEAN over this key's backward calls
    of the gradient's mean over the rows (PersistentMemory._request_centre_update): the needs forecast's state slot
    ("needs"), and the first draft of version 3 on the state itself ("state", a measurement control)."""

    @staticmethod
    def forward(ctx, x, module, key):
        ctx.module = module
        ctx.key = key
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        centre = ctx.module._request_centre_update(ctx.key, _shared_rows(g.detach().float().reshape(-1, g.shape[-1])))
        return g - centre.to(g), None, None


class _CentredReadout(torch.autograd.Function):
    """The repaired readout y = W s + b: forward F.linear (the numbers nn.Linear computes). Backward: the ordinary
    bias b receives every row's request whole; the weight W and the state s receive each row's request minus the
    RUNNING MEAN over this call site's backward calls of the request's mean over the rows
    (PersistentMemory._request_centre_update). The centre is taken before W, so W's growth does not change what it
    has to follow: a request every body shares and that persists moves only b."""

    @staticmethod
    def forward(ctx, state, weight, bias, module, site):
        ctx.save_for_backward(state, weight)
        ctx.module = module
        ctx.site = site
        return F.linear(state, weight, bias)

    @staticmethod
    def backward(ctx, g):
        state, weight = ctx.saved_tensors
        rows = g.reshape(-1, g.shape[-1])
        centre = ctx.module._request_centre_update(ctx.site, _shared_rows(rows.detach().float()))
        gc = rows - centre.to(rows)
        grad_state = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_state = (gc @ weight.to(gc)).reshape(state.shape).to(state.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = (gc.T @ state.reshape(-1, state.shape[-1]).to(gc)).to(weight.dtype)
        if ctx.needs_input_grad[2]:
            grad_bias = rows.sum(0)
        return grad_state, grad_weight, grad_bias, None, None


def _with_ones(x: torch.Tensor) -> torch.Tensor:
    """[..., n] -> [..., n + 1] with a column of ones (the part's bias as a weight column)."""
    return torch.cat([x, torch.ones_like(x[..., :1])], -1)


def _second_moment(x: torch.Tensor) -> torch.Tensor:
    """E[[x;1][x;1]^T] over every row of x ([..., n]) -> [n + 1, n + 1], fp32, no gradient."""
    z = _with_ones(x.detach().float().reshape(-1, x.shape[-1]))
    return (z.T @ z) / float(z.shape[0])


class PersistentMemory(nn.Module):
    def __init__(self, input_size, n_modalities, d_model, action_size, hidden_size=96, repair="off"):
        super().__init__()
        self.input_size=int(input_size);self.hidden_size=int(hidden_size)
        self.n_modalities=n_modalities;self.d_model=d_model
        self.input_projection=nn.Sequential(nn.Linear(input_size+2,hidden_size),nn.Tanh())
        self.recurrent=nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.query=nn.Linear(input_size,hidden_size)
        self.key=nn.Linear(hidden_size,hidden_size,bias=False)
        self.readout=nn.Linear(2*hidden_size,n_modalities*d_model)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
        self.outcome=nn.Sequential(nn.Linear(2*hidden_size+action_size,hidden_size),nn.GELU(),nn.Linear(hidden_size,5),nn.Sigmoid())
        # --persistent-memory-repair (default "off": nothing below is built and the state dict is
        # the one every earlier build saved). The refusal hook is registered in every build: it acts
        # only on a checkpoint whose memory was switched, which no build without the flag writes.
        self.repair = "off"
        self.repair_switched_py = False
        self._event_w = 0.0
        self._state_w = 0.0
        self._moment_w = 0.0
        # the gauges' window: device sums of GAUGES over the lived calls since the last report (no read per
        # step); gauge_report() reads them once and starts a new window
        self._gauge_sum = None
        self._gauge_slot_sum = None
        self._gauge_n = 0
        self._return_on = False             # the constant's return is on (a mirror of its buffer: no read per call)
        self.last_repair = {}
        # MEASUREMENT CONTROLS, never changed in training: the stress test
        # (docs/gain_of_function_v4/persistent_memory_stress.py) turns each off to show what it does.
        # _centring: "request" (this version: each request's running mean over its site's calls of its mean over
        # the bodies removed before the readout's weight), "state" (the first draft of this version: the same
        # running mean taken on the state's gradient), "bodies" (version 2: the state gradient's mean over the
        # bodies removed at every moment), "off"; _lag_corrected: False is the plain running mean as the centre
        self._scales_in_graph = True
        self._centring = "request"
        self._lag_corrected = True
        self.request_centre_horizon = REQUEST_CENTRE_HORIZON
        self._rc_w = [0.0] * len(REQUEST_SITES)
        self._nrc_w = 0.0
        self._state_rc = None
        self._src_w = 0.0
        self._register_load_state_dict_pre_hook(PersistentMemory._check_repair_switch, with_module=True)
        self.register_load_state_dict_post_hook(PersistentMemory._sync_repair)
        if repair != "off":
            self.enable_repair(repair)

    # --------------------------------------------------------------- --persistent-memory-repair
    def enable_repair(self, mode: str = "centred") -> None:
        """Build the repair's parts (before the optimizer is built: needs_gain is a parameter). The
        switch itself happens at the first lived step (prepare_switch, apply_switch)."""
        if mode not in MEMORY_REPAIR_CHOICES or mode == "off":
            raise ValueError("persistent memory repair must be one of %s" % (MEMORY_REPAIR_CHOICES[1:],))
        S = 2 * self.hidden_size
        H = self.hidden_size
        dev = self.readout.weight.device
        self.register_buffer(SWITCHED_KEY, torch.zeros((), dtype=torch.bool, device=dev))
        self.register_buffer("state_ref", torch.zeros(S, device=dev))
        self.register_buffer("event_centre", torch.zeros(self.input_size, device=dev))
        self.register_buffer("event_centre_weight", torch.zeros((), dtype=torch.float64, device=dev))
        self.register_buffer("state_centre", torch.zeros(S, device=dev))
        self.register_buffer("state_centre_weight", torch.zeros((), dtype=torch.float64, device=dev))
        # the second moments of what the candidate's two parts read ([x; 1] and [h; 1]) and their weight
        self.register_buffer("codes_moment", torch.eye(H + 1, device=dev))
        self.register_buffer("hidden_moment", torch.eye(H + 1, device=dev))
        self.register_buffer("moment_weight", torch.zeros((), dtype=torch.float64, device=dev))
        # the readout's bias and the candidate's two scales at the switch (the gauges' references)
        self.register_buffer("bias_at_switch", torch.zeros(self.n_modalities * self.d_model, device=dev))
        self.register_buffer("scales_at_switch", torch.zeros(2, device=dev))
        # per call site, the running mean over its backward calls of the request's mean over the bodies and the
        # running mean of that running mean (the centre is twice the first less the second): the memory bias's
        # sites, and the needs forecast's state slot
        self.register_buffer("request_centre", torch.zeros(len(REQUEST_SITES), 2, self.n_modalities * self.d_model,
                                                           device=dev))
        self.register_buffer("request_centre_weight", torch.zeros(len(REQUEST_SITES), dtype=torch.float64,
                                                                  device=dev))
        self.register_buffer("needs_request_centre", torch.zeros(2, S, device=dev))
        self.register_buffer("needs_request_centre_weight", torch.zeros((), dtype=torch.float64, device=dev))
        # the constant's return (THE CONSTANT'S RETURN above): per slot its size at the switch and its target, the
        # schedule's length (0 = no return: the constant stays as folded) and the model updates since the switch
        self.register_buffer("constant_start", torch.zeros(self.n_modalities, device=dev))
        self.register_buffer("constant_target", torch.zeros(self.n_modalities, device=dev))
        self.register_buffer("constant_return_updates", torch.zeros((), device=dev))
        self.register_buffer("updates_since_switch", torch.zeros((), dtype=torch.float64, device=dev))
        # what the next switch writes into them (LemLite.set_constant_return names the targets by sense slot)
        self._return_targets = torch.zeros(self.n_modalities)
        self._return_updates = 0
        self.needs_gain = nn.Parameter(torch.zeros(S, device=dev))
        self.repair = mode

    @property
    def repaired(self) -> bool:
        return self.repair != "off" and self.repair_switched_py

    def set_constant_return(self, targets, updates: int) -> None:
        """The constant's return the next switch writes: per sense slot its target size (0 = no cap for that slot) and
        the schedule's length in model updates (0 = the constant stays as folded)."""
        t = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)
        if t.numel() != self.n_modalities or bool((t < 0).any()) or int(updates) < 0:
            raise ValueError("the constant's return needs one target of 0 or more per sense slot and a length >= 0")
        self._return_targets = t
        self._return_updates = int(updates)

    def constant_factor(self) -> torch.Tensor:
        """[n_modalities]: what multiplies each slot's constant (the readout's ordinary bias) in the memory bias,
        inside the graph: min(1, R(t) / its size now) while the return is on and at least one update has passed since
        the switch; exactly 1 otherwise (at the switch every head reads the folded constant as production did)."""
        b = self.readout.bias.float().reshape(self.n_modalities, self.d_model)
        r = b.square().mean(-1).sqrt()
        start, target = self.constant_start.float(), self.constant_target.float()
        T = self.constant_return_updates.float()
        t = self.updates_since_switch.float()
        progress = (t / T.clamp_min(1.0)).clamp(0.0, 1.0)
        R = start * (target / start.clamp_min(RMS_FLOOR)).clamp_min(RMS_FLOOR) ** progress
        cap = torch.where(target < start, R, target)            # a slot at or under its target: capped at it
        on = (T > 0) & (t > 0) & (target > 0) & (start > 0)
        return torch.where(on, torch.minimum(torch.ones_like(r), cap / r.clamp_min(RMS_FLOOR)), torch.ones_like(r))

    def constant_cap(self) -> torch.Tensor:
        """[n_modalities]: each slot's cap R(t) now (its size at the switch before the first update; 0 when off)."""
        with torch.no_grad():
            start, target = self.constant_start.float(), self.constant_target.float()
            T = float(self.constant_return_updates)
            if T <= 0:
                return torch.zeros_like(start)
            p = min(1.0, float(self.updates_since_switch) / T)
            R = start * (target / start.clamp_min(RMS_FLOOR)).clamp_min(RMS_FLOOR) ** p
            return torch.where(target < start, R, target)

    @torch.no_grad()
    def advance_constant_return(self) -> None:
        """One model update has ended (LemLite.step, after its optimizer step): the schedule's clock moves. On the
        device, no read; nothing before the switch."""
        if self.repaired:
            self.updates_since_switch.add_(1.0)

    @staticmethod
    def _check_repair_switch(module, state_dict, prefix, local_metadata, strict, missing_keys,
                             unexpected_keys, error_msgs):
        """A switched memory's readout bias holds the folded constant and its readout weight multiplies
        the centred state; a build without the repair would read the saturated state through it.
        ValueError on purpose: the checkpoint loader turns a RuntimeError into a silent
        reinitialisation of the memory. (LemLite.load_checkpoint refuses earlier, from the receipt,
        with the pre-switch checkpoint's path.)"""
        flag = state_dict.get(prefix + SWITCHED_KEY)
        if flag is not None and bool(flag) and module.repair == "off":
            raise ValueError(
                "this checkpoint's persistent memory was switched to --persistent-memory-repair centred "
                "(its readout bias holds the folded constant and its readout weight reads the centred "
                "state); a build without the repair would compute another memory bias. Relaunch with "
                "--persistent-memory-repair centred, or go back to the checkpoint saved before the switch "
                "(the receipt's persistent_memory_repair.switch.pre_switch_checkpoint), which discards "
                "everything learned since the switch. Code from before the repair was merged does not "
                "refuse: it trains another memory and restarts every Adam moment of the model.")
        if flag is not None and bool(flag) and prefix + "codes_moment" not in state_dict:
            raise ValueError("this checkpoint's persistent memory was switched by the first version of the "
                             "repair (persistent_memory_centred_v1), whose running statistics this build "
                             "does not have; go back to the checkpoint saved before the switch")

    @staticmethod
    def _sync_repair(module, incompatible_keys) -> None:
        if module.repair == "off":
            return
        module.repair_switched_py = bool(getattr(module, SWITCHED_KEY))
        module._event_w = float(module.event_centre_weight)
        module._state_w = float(module.state_centre_weight)
        module._moment_w = float(module.moment_weight)
        module._rc_w = [float(x) for x in module.request_centre_weight.tolist()]
        module._nrc_w = float(module.needs_request_centre_weight)
        module._gauge_sum = None
        module._gauge_slot_sum = None
        module._gauge_n = 0
        module._return_on = bool(module.repair_switched_py and float(module.constant_return_updates) > 0)

    # ------------------------------------------------------------------ the running statistics
    @staticmethod
    def _ema(buf: torch.Tensor, value: torch.Tensor, w_old: float, horizon: int) -> float:
        """buf <- the start-up corrected running mean with value (liveness.RevivedRows.update_centre's
        arithmetic: the first call sets it). On the device, no read. Returns the new weight."""
        a = 1.0 / float(horizon)
        w_new = (1.0 - a) * w_old + a
        buf.mul_((1.0 - a) * w_old / w_new).add_(value.detach().to(buf) * (a / w_new))
        return w_new

    @torch.no_grad()
    def _request_centre_update(self, key: str, shared: torch.Tensor) -> torch.Tensor:
        """One backward call's shared request (its mean over the rows) into its key's running mean and the running
        mean of that (start-up corrected: the first call sets both); returns the centre, twice the running mean
        less the running mean of the running mean (a steadily changing request is followed without lag). Keys: a
        REQUEST_SITES site (the memory bias; an unknown site counts as "other"), "needs" (the needs forecast's
        state slot), "state" (the first draft's centre on the state's gradient, a measurement control). Called
        inside backward, on the device, no read."""
        a = 1.0 / float(self.request_centre_horizon)
        i = None
        if key == "needs":
            pair, w_old = self.needs_request_centre, self._nrc_w
        elif key == "state":
            if self._state_rc is None or tuple(self._state_rc.shape) != (2,) + tuple(shared.shape):
                self._state_rc = torch.zeros((2,) + tuple(shared.shape), dtype=torch.float32, device=shared.device)
                self._src_w = 0.0
            pair, w_old = self._state_rc, self._src_w
        else:
            i = REQUEST_SITES.index(key) if key in REQUEST_SITES else REQUEST_SITES.index("other")
            pair, w_old = self.request_centre[i], self._rc_w[i]
        w_new = (1.0 - a) * w_old + a
        k = a / w_new
        level, level_mean = pair[0], pair[1]
        # each running mean in its delta form, m + k (x - m): the same number as m (1 - k) + x k, and exact when
        # the request does not change. A request every body shares at every update must leave exactly 0 behind:
        # Adam divides a gradient by its own running size, so a rounding residual of the same sign at every update
        # would move the memory's weights at full-size steps (the stress test's constant push, with the product
        # form: bodies 0.27 to 0.013 apart). 2 x - x is exact in float arithmetic.
        level.add_((shared.to(level) - level) * k)
        level_mean.add_((level - level_mean) * k)
        if key == "needs":
            self._nrc_w = w_new
            self.needs_request_centre_weight.fill_(w_new)
        elif key == "state":
            self._src_w = w_new
        else:
            self._rc_w[i] = w_new
            self.request_centre_weight[i] = w_new
        return (2.0 * level - level_mean) if self._lag_corrected else level

    def _centre_state_grad(self, raw: torch.Tensor) -> torch.Tensor:
        """The measurement controls that centre the state's own gradient ("request", this version, centres each
        request where it is read: bias() and predict_needs())."""
        if self._centring == "bodies":
            return body_centred_grad(raw)
        if self._centring == "state":
            return _RunningCentre.apply(raw, self, "state")
        return raw

    @torch.no_grad()
    def _update_centre(self, which: str, value: torch.Tensor) -> None:
        """One lived call's values (mean over the bodies) into the events' or the state's running mean."""
        if which == "event":
            m = value.detach().float().reshape(-1, self.event_centre.shape[0]).mean(0)
            self._event_w = self._ema(self.event_centre, m, self._event_w, CENTRE_HORIZON_STEPS)
            self.event_centre_weight.fill_(self._event_w)
        else:
            m = value.detach().float().reshape(-1, self.state_centre.shape[0]).mean(0)
            self._state_w = self._ema(self.state_centre, m, self._state_w, CENTRE_HORIZON_STEPS)
            self.state_centre_weight.fill_(self._state_w)

    @torch.no_grad()
    def _update_moments(self, x: torch.Tensor, hprev: torch.Tensor) -> None:
        """One lived call's codes and previous states into the candidate's second moments. OUT OF PLACE: the
        step's graphs hold the moments the scales were computed from (their backward reads them), so the new
        moments are new tensors and the buffers are reassigned."""
        a = 1.0 / float(SCALE_HORIZON_STEPS)
        w_new = (1.0 - a) * self._moment_w + a
        keep, take = (1.0 - a) * self._moment_w / w_new, a / w_new
        self.codes_moment = self.codes_moment * keep + _second_moment(x).to(self.codes_moment) * take
        self.hidden_moment = self.hidden_moment * keep + _second_moment(hprev).to(self.hidden_moment) * take
        self._moment_w = w_new
        self.moment_weight.fill_(w_new)

    # ------------------------------------------------------------------ the candidate
    def _parts(self):
        """The candidate's two stored parts as matrices with their bias as the last column: [H, H + 1]."""
        g = self.recurrent
        H = self.hidden_size
        Wu = torch.cat([g.weight_ih_l0[2 * H:], g.bias_ih_l0[2 * H:, None]], 1)
        Wv = torch.cat([g.weight_hh_l0[2 * H:], g.bias_hh_l0[2 * H:, None]], 1)
        return Wu, Wv

    def candidate_scales(self, codes_moment=None, hidden_moment=None):
        """(c_u, c_v, stored size u, stored size v), IN THE GRAPH: the scales that put the two parts' root
        mean square over the data at CANDIDATE_INPUT_RMS and CANDIDATE_RECURRENT_RMS, from the parts' stored
        sizes sqrt(tr(W M W^T) / H). The moments are the running ones unless given (the switch's plan)."""
        Mx = self.codes_moment if codes_moment is None else codes_moment
        Mh = self.hidden_moment if hidden_moment is None else hidden_moment
        Wu, Wv = self._parts()
        H = float(self.hidden_size)
        su = torch.sqrt(((Wu @ Mx.to(Wu)) * Wu).sum() / H + RMS_FLOOR ** 2)
        sv = torch.sqrt(((Wv @ Mh.to(Wv)) * Wv).sum() / H + RMS_FLOOR ** 2)
        cu, cv = CANDIDATE_INPUT_RMS / su, CANDIDATE_RECURRENT_RMS / sv
        if not self._scales_in_graph:                 # measurement control only (the review's plain numbers)
            cu, cv = cu.detach(), cv.detach()
        return cu, cv, su, sv

    @torch.no_grad()
    def _rescale_candidate_rows(self, cu: float, cv: float) -> None:
        """Multiply the stored candidate rows by their scales, so the stored parts' sizes equal their targets
        (the scales then read 1): the effective rows, and every output, are unchanged (the scale divides by
        the stored size), and Adam's step of about the learning rate per stored number is then a step of about
        the learning rate per effective number. Without it a part stored at 30 times its target (H84's
        candidate bias held its units at +-1) learns 30 times slower. The switch does this once."""
        g = self.recurrent
        H = self.hidden_size
        g.weight_ih_l0[2 * H:].mul_(float(cu))
        g.bias_ih_l0[2 * H:].mul_(float(cu))
        g.weight_hh_l0[2 * H:].mul_(float(cv))
        g.bias_hh_l0[2 * H:].mul_(float(cv))

    def _effective_gru(self, codes_moment=None, hidden_moment=None):
        """The GRU's weights with the candidate's rows scaled (the gates untouched): (w_ih, w_hh, b_ih,
        b_hh, c_u, c_v)."""
        g = self.recurrent
        H = self.hidden_size
        cu, cv, _su, _sv = self.candidate_scales(codes_moment, hidden_moment)
        w_ih = torch.cat([g.weight_ih_l0[:2 * H], g.weight_ih_l0[2 * H:] * cu], 0)
        b_ih = torch.cat([g.bias_ih_l0[:2 * H], g.bias_ih_l0[2 * H:] * cu], 0)
        w_hh = torch.cat([g.weight_hh_l0[:2 * H], g.weight_hh_l0[2 * H:] * cv], 0)
        b_hh = torch.cat([g.bias_hh_l0[:2 * H], g.bias_hh_l0[2 * H:] * cv], 0)
        return w_ih, w_hh, b_ih, b_hh, cu, cv

    def _gru_scaled(self, x: torch.Tensor, h0, codes_moment=None, hidden_moment=None) -> tuple:
        """The GRU with its candidate rows scaled, in one fused call. (states [B, T, H], last [B, H])."""
        H = self.hidden_size
        B = x.shape[0]
        w_ih, w_hh, b_ih, b_hh, _cu, _cv = self._effective_gru(codes_moment, hidden_moment)
        hx = x.new_zeros(1, B, H) if h0 is None else h0.reshape(1, B, H).to(x)
        states, last = torch._VF.gru(x, hx, (w_ih, w_hh, b_ih, b_hh), True, 1, 0.0, self.training, False, True)
        return states, last[-1]

    @torch.no_grad()
    def candidate_parts(self, x: torch.Tensor, states: torch.Tensor, h0, codes_moment=None, hidden_moment=None):
        """The EFFECTIVE candidate's input part c_u (W_in x + b_in), recurrent part r * c_v (W_hn h + b_hn) and
        update gate z at every retained event (h the states the scaled GRU produced, batched: no loop over the
        events), with the previous states: (u, v, z, hprev)."""
        H = self.hidden_size
        B = x.shape[0]
        w_ih, w_hh, b_ih, b_hh, _cu, _cv = self._effective_gru(codes_moment, hidden_moment)
        h_first = x.new_zeros(B, 1, H) if h0 is None else h0.reshape(B, 1, H).to(x)
        hprev = torch.cat([h_first, states[:, :-1]], 1)
        gi = F.linear(x, w_ih, b_ih)
        gh = F.linear(hprev, w_hh, b_hh)
        r = torch.sigmoid(gi[..., :H] + gh[..., :H])
        z = torch.sigmoid(gi[..., H:2 * H] + gh[..., H:2 * H])
        return gi[..., 2 * H:], r * gh[..., 2 * H:], z, hprev

    # ------------------------------------------------------------------ the repaired encoder
    def _input_codes(self, events, delta, age, event_centre=None):
        """The repaired input layer: the events minus their running mean, the pre-activation divided by
        its root mean square over the units. (centred events, codes x)."""
        centre = self.event_centre if event_centre is None else event_centre
        ec = events.float() - centre
        pre = self.input_projection[0](torch.cat([ec, torch.log1p(delta.clamp_min(0)).unsqueeze(-1),
                                                  torch.log1p(age).unsqueeze(-1)], -1))
        return ec, torch.tanh(_rms_units(pre))

    def _encode_centred(self, events, delta, age, carry, event_centre=None, codes_moment=None,
                        hidden_moment=None):
        """The repaired encoder: (raw state [B, 2H], GRU states [B, T, H], codes x [B, T, H])."""
        ec, x = self._input_codes(events, delta, age, event_centre)
        states, last = self._gru_scaled(x, None if carry is None else carry.float(), codes_moment, hidden_moment)
        q = self.query(ec[:, -1]).unsqueeze(1)
        scores = (q * self.key(states)).sum(-1) / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=1)
        retrieved = (weights.unsqueeze(-1) * states).sum(1)
        return torch.cat([last, retrieved], -1), states, x

    def encode(self, events, times, now, carry=None, carry_time=None, observe=False):
        """(state [B, 2H], GRU states [B, T, H]). observe=True marks the one lived call per model step
        (memory_integration.observe): under the repair it moves the running means and moments and adds the
        gauges to their window on the device (no read). Without the repair, or before its switch, the memory
        every earlier build computed."""
        if not self.repaired:
            return self.encode_legacy(events, times, now, carry, carry_time)
        delta, age = self._check_and_time(events, times, now, carry, carry_time)
        with torch.autocast(device_type=events.device.type, enabled=False):
            raw, states, x = self._encode_centred(events, delta, age, carry)
            held = self._centre_state_grad(raw)            # the measurement controls only (see _centring)
            view = held - self.state_centre
            if observe:
                self._observe_lived(events, raw, view, x, states, carry)
        return view, states

    @torch.no_grad()
    def _observe_lived(self, events, raw, view, x, states, carry) -> None:
        """The lived call: this moment's gauges (with the running statistics as they were used) added to the
        window's device sums; then the events' and the state's running means and the candidate's second
        moments move. Nothing is read from the device."""
        u, v, z, hprev = self.candidate_parts(x, states, carry)
        cu, cv, su, sv = self.candidate_scales()
        Wu, Wv = self._parts()                      # the stored candidate rows (their size: no gradient grows it)
        a = u + v
        vd = view.detach().float()
        varying = F.linear(vd, self.readout.weight.float())
        vals = torch.stack([
            (raw.abs() > 0.99).float().mean(),
            (raw - raw.mean(0, keepdim=True)).square().mean().sqrt(),
            (states - states.mean(1, keepdim=True)).square().mean().sqrt(),
            vd.square().mean().sqrt(),
            varying.square().mean().sqrt(),
            (varying - varying.mean(0, keepdim=True)).square().mean().sqrt(),
            self.readout.bias.float().square().mean().sqrt(),
            (self.readout.bias.float() - self.bias_at_switch).square().mean().sqrt(),
            self.needs_gain.float().abs().mean(),
            a.square().mean().sqrt(),
            (a.abs() > TANH_099).float().mean(),
            (a.abs() > TANH_1EM7).float().mean(),
            (z > 0.99).float().mean(),
            cu.reshape(()), cv.reshape(()), su.reshape(()), sv.reshape(()),
            Wu.norm(), Wv.norm()]).float()
        self._gauge_sum = vals if self._gauge_sum is None else self._gauge_sum + vals
        per_slot = varying.reshape(varying.shape[0], self.n_modalities, self.d_model).square().mean((0, 2)).sqrt()
        self._gauge_slot_sum = per_slot if self._gauge_slot_sum is None else self._gauge_slot_sum + per_slot
        self._gauge_n += 1
        self._update_centre("event", events[:, -1].float())
        self._update_centre("state", raw)
        self._update_moments(x, hprev)

    def gauge_report(self) -> dict:
        """The window's mean of every gauge since the last report (one read from the device), and a new
        window. {} before the switch, or when no lived call happened since the last report."""
        if self._gauge_sum is None or self._gauge_n == 0:
            return {}
        vals = (self._gauge_sum / float(self._gauge_n)).tolist()
        out = dict(zip(GAUGES, vals))
        out["repair"] = self.repair
        out["gauge_calls"] = int(self._gauge_n)
        at_switch = self.scales_at_switch.tolist()
        out["candidate_input_scale_at_switch"] = at_switch[0]
        out["candidate_recurrent_scale_at_switch"] = at_switch[1]
        # the constant's return, per sense slot (in the memory's slot order), read once per report: the stored
        # constant's size, what the heads read of it (after the cap), the cap now, the varying part's size (the
        # window's mean over the lived calls), and the updates since the switch
        with torch.no_grad():
            b = self.readout.bias.float().reshape(self.n_modalities, self.d_model)
            r = b.square().mean(-1).sqrt()
            f = self.constant_factor() if self._return_on else torch.ones_like(r)
            out["constant_rms_per_slot"] = [float(x) for x in r.tolist()]
            out["constant_effective_rms_per_slot"] = [float(x) for x in (r * f).tolist()]
            out["constant_cap_per_slot"] = [float(x) for x in self.constant_cap().tolist()]
            out["constant_return_updates"] = int(float(self.constant_return_updates))
            out["updates_since_switch"] = int(float(self.updates_since_switch))
            if self._gauge_slot_sum is not None:
                out["varying_bias_rms_per_slot"] = [float(x) for x in (self._gauge_slot_sum / float(self._gauge_n)).tolist()]
        self._gauge_slot_sum = None
        self._gauge_sum = None
        self._gauge_n = 0
        self.last_repair = out
        return out

    def _check_and_time(self, events, times, now, carry, carry_time):
        if events.ndim != 3 or events.shape[-1] != self.input_size or events.shape[1] == 0:
            raise ValueError('memory needs a nonempty body/time/feature sequence')
        if times.shape != events.shape[:2] or now.shape != events.shape[:1]:
            raise ValueError('memory timestamps must match each body and event')
        if not torch.isfinite(events).all() or not torch.isfinite(times).all() or not torch.isfinite(now).all():
            raise ValueError('nonfinite memory input')
        if (times[:,1:] < times[:,:-1]).any() or (times > now[:,None]+1e-5).any():
            raise ValueError('memory cannot read unordered or future events')
        initial_gap = torch.zeros_like(times[:,:1])
        if carry_time is not None:
            if carry is None or carry_time.shape != now.shape or not torch.isfinite(carry_time).all() or (carry_time > times[:,0]).any():
                raise ValueError('invalid recurrent prefix timestamp')
            initial_gap = times[:,:1] - carry_time[:,None]
        delta=torch.cat([initial_gap,times[:,1:]-times[:,:-1]],1)
        age=(now[:,None]-times).clamp_min(0)
        return delta, age

    # ------------------------------------------------------------------ the switch
    @torch.no_grad()
    def production_bias(self, states: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """The bias the unswitched memory hands the heads for these states, as production computes it: the
        readout under the caller's autocast (bf16 on the graphics card inside LemLite.step), as fp32."""
        out = [self.readout(states[i:i + chunk].to(self.readout.weight.device)).float()
               for i in range(0, int(states.shape[0]), chunk)]
        return torch.cat(out, 0)

    @torch.no_grad()
    def prepare_switch(self, snapshot: dict) -> dict:
        """Every reading and every refusal of the switch, nothing written: the plan apply_switch writes.
        snapshot: MemoryHistory.snapshot() of the lived moment. Call it where production computes the memory
        bias (inside LemLite.step's autocast): the fold takes the bias production computed there.
        Refuses (ValueError) when the fold would change a bias number by more than the line (switch_limit):
        in fp32 against every body's fp32 bias (the saved state must be one constant for every body), and
        against every body's production bias."""
        if self.repair == "off":
            raise ValueError("the memory repair is off")
        if self.repair_switched_py:
            raise ValueError("the memory was switched already")
        keys = ("events", "times", "now", "carry", "carry_time")
        snap = {k: snapshot.get(k) for k in keys}
        old_state, _ = self.encode_legacy(**snap)            # fp32 (the old encoder turns autocast off)
        old_state = old_state.float()
        prod = self.readout(old_state)                        # production's bias (the caller's autocast)
        prod_dtype = str(prod.dtype).replace("torch.", "")
        prod = prod.float()
        with torch.autocast(device_type=self.readout.weight.device.type, enabled=False):
            return self._plan_fp32(snap, old_state, prod, prod_dtype)

    def _plan_fp32(self, snap, old_state, prod, prod_dtype) -> dict:
        S = old_state.shape[-1]
        H = self.hidden_size
        s_ref = old_state.mean(0)
        old_bias = self.readout(old_state)
        largest = float(old_bias.abs().max())
        limit = switch_limit(largest)
        change32 = float(((self.readout.bias + self.readout.weight @ s_ref).unsqueeze(0) - old_bias).abs().max())
        # THE FOLD TAKES PRODUCTION'S OWN NUMBER: per number the median over the bodies of the bias production
        # computed (a body's value; every body's where they agree)
        fold = prod.median(dim=0).values if prod.shape[0] > 1 else prod[0].clone()
        prod_change = (fold.unsqueeze(0) - prod).abs()
        differ = (prod.max(0).values - prod.min(0).values) > 0
        record = dict(version=MEMORY_REPAIR_VERSION, mode=self.repair, bodies=int(old_state.shape[0]),
                      old_state_share_above_0p99=float((old_state.abs() > 0.99).float().mean()),
                      old_state_spread_between_bodies=float((old_state - s_ref).square().mean().sqrt()),
                      readout_bias_rms_before=float(self.readout.bias.square().mean().sqrt()),
                      readout_weight_norm_before=float(self.readout.weight.norm()),
                      bias_change_max_abs=change32, bias_largest_abs=largest, bias_change_limit=limit,
                      production_bias_dtype=prod_dtype,
                      production_minus_fp32_max_abs=float((prod - old_bias).abs().max()),
                      production_minus_fp32_rms=float((prod - old_bias).square().mean().sqrt()),
                      production_change_max_abs=float(prod_change.max()),
                      production_numbers_changed=int((prod_change > 0).sum()),
                      production_numbers=int(prod_change.numel()),
                      production_numbers_where_bodies_differ=int(differ.sum()),
                      constant_bias_rms=float(fold.square().mean().sqrt()))
        if not change32 <= limit:
            raise ValueError(
                "persistent memory repair refused: folding the saturated state's bias into the readout "
                "bias would change a bias number by %.3g (line %.3g): the saved memory state is not one "
                "constant (spread between bodies %.3g)" % (change32, limit, record["old_state_spread_between_bodies"]))
        if not record["production_change_max_abs"] <= limit:
            raise ValueError(
                "persistent memory repair refused: the bodies' %s memory biases differ at %d of %d numbers "
                "(largest %.3g, line %.3g): one shared bias would move some body's number by a rounding step"
                % (prod_dtype, record["production_numbers_where_bodies_differ"], int(differ.numel()),
                   record["production_change_max_abs"], limit))
        # the needs forecast's state slot: its old share joins its first layer's bias
        W1 = self.outcome[0].weight
        fold_needs = W1[:, :S].float() @ s_ref
        record["needs_fold_rms"] = float(fold_needs.square().mean().sqrt())
        # the running statistics start from this moment: the events' mean over the retained history, the
        # candidate's moments from the snapshot's codes and states (the states' moment in two passes: the
        # scaled GRU's states depend on it), the state's mean from the repaired encoder's reading of it
        ev = snap["events"].float()
        a_c = 1.0 / float(CENTRE_HORIZON_STEPS)
        event_centre = ev.reshape(-1, ev.shape[-1]).mean(0)
        delta, age = self._check_and_time(snap["events"], snap["times"], snap["now"], snap["carry"], snap["carry_time"])
        _ec, x = self._input_codes(snap["events"], delta, age, event_centre)
        codes_moment = _second_moment(x)
        B = x.shape[0]
        carry = snap["carry"]
        h0 = x.new_zeros(B, H) if carry is None else carry.reshape(B, H).float()
        hidden_moment = _second_moment(h0)                    # pass 0: the saved carry alone
        for _pass in range(2):
            raw, states, x = self._encode_centred(snap["events"], delta, age, carry, event_centre,
                                                  codes_moment, hidden_moment)
            _u, _v, _z, hprev = self.candidate_parts(x, states, carry, codes_moment, hidden_moment)
            hidden_moment = _second_moment(hprev)
        raw, states, x = self._encode_centred(snap["events"], delta, age, carry, event_centre, codes_moment,
                                              hidden_moment)
        u, v, z, _hp = self.candidate_parts(x, states, carry, codes_moment, hidden_moment)
        cu, cv, su, sv = self.candidate_scales(codes_moment, hidden_moment)
        eff = u + v
        # the stored candidate rows are multiplied by these scales at the switch (_rescale_candidate_rows), so
        # the scales read 1 from then on: the gauges' reference and the scale's stop line start there
        scales_after = torch.stack([(CANDIDATE_INPUT_RMS / (su * cu)).reshape(()),
                                    (CANDIDATE_RECURRENT_RMS / (sv * cv)).reshape(())]).float()
        record.update(candidate_input_stored_size=float(su), candidate_recurrent_stored_size=float(sv),
                      candidate_input_rows_rescaled_by=float(cu), candidate_recurrent_rows_rescaled_by=float(cv),
                      candidate_input_rows_norm_after_switch=float(self._parts()[0].norm() * cu),
                      candidate_recurrent_rows_norm_after_switch=float(self._parts()[1].norm() * cv),
                      candidate_rms=float(eff.square().mean().sqrt()),
                      candidate_input_part_rms=float(u.square().mean().sqrt()),
                      candidate_recurrent_part_rms=float(v.square().mean().sqrt()),
                      candidate_share_above_tanh_0p99=float((eff.abs() > TANH_099).float().mean()),
                      candidate_share_above_tanh_1em7=float((eff.abs() > TANH_1EM7).float().mean()),
                      update_gate_share_above_0p99=float((z > 0.99).float().mean()),
                      input_codes_share_above_0p99=float((x.abs() > 0.99).float().mean()),
                      new_state_share_above_0p99=float((raw.abs() > 0.99).float().mean()),
                      new_state_spread_between_bodies=float((raw - raw.mean(0, keepdim=True)).square().mean().sqrt()),
                      new_states_spread_over_window=float((states - states.mean(1, keepdim=True)).square().mean().sqrt()))
        return dict(record=record, s_ref=s_ref, fold=fold, fold_needs=fold_needs, event_centre=event_centre,
                    event_w=1.0 - (1.0 - a_c) ** int(ev.shape[1]), codes_moment=codes_moment,
                    hidden_moment=hidden_moment, moment_w=1.0 / float(SCALE_HORIZON_STEPS),
                    state_centre=raw.float().mean(0), state_w=a_c, rows_scale=(float(cu), float(cv)),
                    scales=scales_after, limit=limit, largest=largest)

    @torch.no_grad()
    def apply_switch(self, plan: dict) -> dict:
        """Write the plan: copies only, so nothing can fail between the first write and the flag."""
        if self.repair_switched_py:
            raise ValueError("the memory was switched already")
        self.readout.bias.copy_(plan["fold"])
        self.readout.weight.zero_()
        self.outcome[0].bias.add_(plan["fold_needs"].to(self.outcome[0].bias))
        self._rescale_candidate_rows(*plan["rows_scale"])
        self.state_ref.copy_(plan["s_ref"])
        self.needs_gain.zero_()
        self.event_centre.copy_(plan["event_centre"])
        self._event_w = float(plan["event_w"])
        self.event_centre_weight.fill_(self._event_w)
        self.codes_moment.copy_(plan["codes_moment"])
        self.hidden_moment.copy_(plan["hidden_moment"])
        self._moment_w = float(plan["moment_w"])
        self.moment_weight.fill_(self._moment_w)
        self.state_centre.copy_(plan["state_centre"])
        self._state_w = float(plan["state_w"])
        self.state_centre_weight.fill_(self._state_w)
        self.bias_at_switch.copy_(plan["fold"])
        self.scales_at_switch.copy_(plan["scales"])
        start = plan["fold"].float().reshape(self.n_modalities, self.d_model).square().mean(-1).sqrt()
        self.constant_start.copy_(start)
        self.constant_target.copy_(self._return_targets.to(self.constant_target))
        self.constant_return_updates.fill_(float(self._return_updates))
        self.updates_since_switch.zero_()
        self._return_on = self._return_updates > 0
        plan["record"]["constant_return"] = dict(updates=int(self._return_updates),
                                                 start_per_slot=[float(x) for x in start.tolist()],
                                                 target_per_slot=[float(x) for x in self._return_targets.tolist()])
        self._gauge_sum = None
        self._gauge_slot_sum = None
        self._gauge_n = 0
        getattr(self, SWITCHED_KEY).fill_(True)
        self.repair_switched_py = True
        return plan["record"]

    def switch(self, snapshot: dict) -> dict:
        """prepare_switch then apply_switch (the memory alone; LemLite._persistent_memory_switch also checks
        the replay ring and moves the optimizer, the ring and the liveness centre between the two)."""
        return self.apply_switch(self.prepare_switch(snapshot))

    def encode_legacy(self, events, times, now, carry=None, carry_time=None):
        """The memory every build before the repair computed (and the repaired build before its switch)."""
        if events.ndim != 3 or events.shape[-1] != self.input_size or events.shape[1] == 0:
            raise ValueError('memory needs a nonempty body/time/feature sequence')
        if times.shape != events.shape[:2] or now.shape != events.shape[:1]:
            raise ValueError('memory timestamps must match each body and event')
        if not torch.isfinite(events).all() or not torch.isfinite(times).all() or not torch.isfinite(now).all():
            raise ValueError('nonfinite memory input')
        if (times[:,1:] < times[:,:-1]).any() or (times > now[:,None]+1e-5).any():
            raise ValueError('memory cannot read unordered or future events')
        # Time is simulation time, never computer runtime. Inter-event gaps inform
        # recurrence, and attention explicitly receives elapsed age.
        initial_gap = torch.zeros_like(times[:,:1])
        if carry_time is not None:
            if carry is None or carry_time.shape != now.shape or not torch.isfinite(carry_time).all() or (carry_time > times[:,0]).any():
                raise ValueError('invalid recurrent prefix timestamp')
            initial_gap = times[:,:1] - carry_time[:,None]
        delta=torch.cat([initial_gap,times[:,1:]-times[:,:-1]],1)
        age=(now[:,None]-times).clamp_min(0)
        # cuDNN recurrent precision support differs from attention autocast. Keep
        # this bounded state in float32 across acting, replay and checkpoints.
        with torch.autocast(device_type=events.device.type,enabled=False):
            x=self.input_projection(torch.cat([events.float(),torch.log1p(delta.clamp_min(0)).unsqueeze(-1),
                                              torch.log1p(age).unsqueeze(-1)],-1))
            states,last=self.recurrent(x,None if carry is None else carry.float())
            q=self.query(events[:,-1].float()).unsqueeze(1)
            scores=(q*self.key(states)).sum(-1)/math.sqrt(self.hidden_size)
            weights=torch.softmax(scores,dim=1)
            retrieved=(weights.unsqueeze(-1)*states).sum(1)
        return torch.cat([last[-1],retrieved],-1), states

    def bias(self, state, site=None):
        """The memory bias [B, n_modalities, d_model]. site: the call site (REQUEST_SITES: lived, delayed, tenure,
        replay; None counts as "other") whose shared request the repaired readout's backward tracks."""
        if self.repaired:
            # after the repair's switch, state is the centred state and readout.bias holds the constant at
            # production's own (bf16 on the card) value: computed in fp32 outside the trainer's autocast, the
            # heads read that value exactly at the switch and a small varying part on top of it unrounded; from the
            # first update after it each slot's constant is scaled to at most its return schedule's cap
            with torch.autocast(device_type=state.device.type, enabled=False):
                const = self.readout.bias
                if self._return_on:
                    const = (const.float().reshape(self.n_modalities, self.d_model)
                             * self.constant_factor().unsqueeze(-1)).reshape(-1)
                if self._centring == "request":
                    out = _CentredReadout.apply(state.float(), self.readout.weight, const, self,
                                                site or "other")
                else:
                    out = F.linear(state.float(), self.readout.weight, const)
                return out.reshape(-1, self.n_modalities, self.d_model)
        return self.readout(state).reshape(-1,self.n_modalities,self.d_model)

    def predict_needs(self,state,executed):
        if self.repaired:
            # the centred state reaches the forecast through needs_gain (0 at the switch); the needs lesson's
            # request on that slot loses its running shared part before it reaches needs_gain and the state
            state = self.needs_gain * state
            if self._centring == "request":
                state = _RunningCentre.apply(state, self, "needs")
        return self.outcome(torch.cat([state,executed.reshape(len(state),-1)],-1))


class MemoryHistory:
    """Bounded salient samples plus a recurrent prefix.

    WHICH moment inside an interval each body keeps is its own (the torch.where in
    observe). The sampling boundary and the eviction are SHARED across bodies: the
    commit test is .all(), so nothing is written until every body has passed
    interval_s of its own simulated time, and bodies on different clocks are sampled
    at the slowest one's pace - driven with body 0 at 0.05 s and body 1 at 0.01 s per
    call, body 0 received one event per 2.50 of its own seconds against interval_s
    0.5. Every world in H83 advances the same 0.05 s per model step, so that
    stretching does not happen there; it would begin the day two worlds advance at
    different rates, and no log line would report it.

    WHAT COUNTS AS THE LARGEST CHANGE is an unweighted mean square over every feature
    channel, so a block's claim on what is kept follows its channel count and its
    numerical spread rather than any stated importance. Measured over H83's 128 stored
    events (1,016 body-steps, 1,140 channels): the six senses hold 128 channels each
    and supply motor 36.95%, audio 31.52%, vision 16.34%, proprioception 9.85%,
    tactile 1.07% and intermind 0.00% of the score, while the 234-channel executed
    order supplies 3.57%, the goal summary 0.63% and the five measured needs 0.07%.

    Every interval retains its largest observed feature change; short events are
    eligible even if they end before the sampling boundary. Current observation is
    appended for the query, never a later outcome. Capacity and sampling period are
    memory-budget settings, not developmental success criteria.
    """
    def __init__(self, capacity=128, interval_s=.5):
        if capacity < 2 or interval_s <= 0: raise ValueError('invalid memory history budget')
        self.capacity=int(capacity);self.interval_s=float(interval_s);self.reset()

    def reset(self):
        self.events=deque();self.times=deque();self.carry=None;self.carry_time=None
        self.last_time=None;self.previous=None;self.pending=None;self.pending_time=None
        self.priority=None;self.boundary=None

    def observe(self,features,now,module):
        features=features.detach().float();now=now.detach().float()
        if self.previous is not None and (features.shape != self.previous.shape or (now < self.last_time).any()):
            raise ValueError('memory body layout or clock changed without an episode reset')
        if self.previous is None:
            self.events.append(features.clone());self.times.append(now.clone());self.boundary=now.clone()
            self.previous=features.clone();self.last_time=now.clone()
            return
        score=(features-self.previous).square().mean(-1)
        if self.pending is None:
            self.pending=features.clone();self.pending_time=now.clone();self.priority=score.clone()
        else:
            choose=score > self.priority
            self.pending=torch.where(choose[:,None],features,self.pending)
            self.pending_time=torch.where(choose,now,self.pending_time)
            self.priority=torch.maximum(score,self.priority)
        if bool(((now-self.boundary)>=self.interval_s-1e-5).all()):
            if len(self.events) == self.capacity:
                # Advance only the evicted prefix. It never contains a later input.
                old=self.events.popleft();old_time=self.times.popleft()
                with torch.no_grad():
                    _,states=module.encode(old[:,None],old_time[:,None],old_time,self.carry,self.carry_time)
                self.carry=states[:,-1].unsqueeze(0).detach()
                self.carry_time=old_time.clone()
            self.events.append(self.pending.clone());self.times.append(self.pending_time.clone())
            self.pending=self.pending_time=self.priority=None;self.boundary=now.clone()
        self.previous=features.clone();self.last_time=now.clone()

    def snapshot(self,features=None,now=None):
        if not self.events: return None
        features=self.previous if features is None else features.detach().float()
        now=self.last_time if now is None else now.detach().float()
        ev=list(self.events);ts=list(self.times)
        # Include the salient event in the open interval before current query.
        if self.pending is not None:
            ev.append(self.pending);ts.append(self.pending_time)
        ev.append(features);ts.append(now)
        return dict(events=torch.stack(ev,1).detach(),times=torch.stack(ts,1).detach(),
                    now=now.clone(),carry=None if self.carry is None else self.carry.clone(),
                    carry_time=None if self.carry_time is None else self.carry_time.clone())

    def state_dict(self):
        return dict(capacity=self.capacity,interval_s=self.interval_s,
                    **{k: ([v.detach().cpu() for v in value] if isinstance(value,deque)
                           else value.detach().cpu() if torch.is_tensor(value) else value)
                       for k,value in self.__dict__.items() if k not in ('capacity','interval_s')})

    def load_state_dict(self,state,device):
        if state['capacity'] != self.capacity or state['interval_s'] != self.interval_s:
            raise ValueError('memory history budget changed without migration')
        for k in ('events','times'):
            setattr(self,k,deque(v.to(device) for v in state[k]))
        for k in ('carry','carry_time','last_time','previous','pending','pending_time','priority','boundary'):
            v=state.get(k);setattr(self,k,None if v is None else v.to(device))
