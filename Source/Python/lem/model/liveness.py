# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Liveness: which learned parts respond, the revival of units that stopped responding, and a path
every element reaches beside the two pools that read one element.

WHAT WAS FOUND (docs/gain_of_function_v3/DESIGN.md section 10; the lead's sweep of ckpt_step_344400,
2026-09-22, measurements/liveness_sweep_ckpt_step_344400.json: 24 lived moments, 4 bodies, CPU).
A hidden unit is LIVE when the input its GELU receives rises above LIVE_LINE (-3) on some sample:
GELU(-3) = -0.004 and its slope there is -0.012, so below the line a unit passes almost nothing
forward and almost no gradient back. Nine layers had fewer than half their units live
(SWEEP_2026_09_22 below; the 250 ms pose head had 0 of 256). The pose head died against a part of
the core latent that every moment shares (norm 66 of the latent's 75): its 256 rows all lean against
that direction, which alone puts every unit's input near -17. The motor encoder's pool read one of
its 26 handles per attention head (spread 0.001 of an even spread; 16 handles, the eyes among them,
got no weight) and the touch encoder's pool one contact row (0.091).

WHAT THIS MODULE DOES, behind --liveness-repair (lem_model.LemLite liveness_repair; default "off"
builds none of it, so an old command builds the model it always built):
  "report"  the whole-model LivenessMonitor only: read-only hooks, the `liveness` stats block.
  "on"      the monitor, plus
    1. RevivedRows, one per layer in REPAIR_LAYERS. After the flag comes on, each such layer's units
       are watched over successive REVIVE_MIN_SAMPLES windows of the layer's input; each unit whose input never
       rose above LIVE_LINE (with --liveness-rule varying, also each unit whose output did not vary: WHICH UNITS
       THE REVIVAL CHOOSES) is revived between two training steps, including a previously repaired
       unit that has become silent again: its outgoing weights (the
       next layer's columns) are set to zero after their mean contribution over the watched samples
       is moved into the next layer's bias. The change is bounded by the discarded units' small
       variation, measured at each switch; from then on the unit's input comes from fresh weights reading the
       layer's input CENTRED by a running mean and then LayerNormed (DESIGN section 10 rule 1: a
       shared direction killed the pose head, and a LayerNorm alone does not remove one). The
       layer's other units keep reading the input as before. Their optimiser moments are reset;
       the outgoing columns' second moment is seeded from the live columns of the same row (see
       RevivedRows.revive for the arithmetic). With --liveness-rule none no unit is revived and no
       window is watched: the units revived before keep their routing, centres and learning.
    2. AllElementPath beside the motor encoder's pool and the touch encoder's pool (rule 3): every
       handle (every contact row) enters the token through its own weights; the path's output layer
       starts at zero, so the first step's tokens are the ones the pool alone gives, bit for bit.

The monitor, every SAMPLE_EVERY-th model step: for each Linear followed by GELU or ReLU inside an
nn.Sequential (the sweep's rule and naming), which units rose above LIVE_LINE; for each
nn.MultiheadAttention, the spread of its attention (entropy as a share of the even-spread maximum,
0 = it reads one input, 1 = every input evenly), recomputed from the module's own inputs and
weights on one body per call so the model's own numbers are untouched. For the two repaired pools
it also reports how evenly the TOKEN reads the elements (attention and the all-element path
together): the share of the token's sensitivity that each element carries, as the same entropy
share (`<pool>.token_read`).

Every constant carries its measurement or its arithmetic.
"""
from __future__ import annotations

import contextlib
import math
import time
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lem.model.attention_cap import head_scores

LIVENESS_VERSION = "liveness_repair_v3"
MODES = ("off", "report", "on")

# DESIGN section 10 rule 2: a unit is live when its input rose above this line on the window's
# samples (GELU(-3) = -0.004, slope -0.012; below it a unit passes almost nothing either way)
LIVE_LINE = -3.0
# the section 10 line for a LAYER: fewer than half its units live = reported as not responding
LAYER_LINE = 0.5

# A UNIT THAT VARIES (2026-09-26, beside the -3 line; the window block's layers_varying and not_varying). The -3 line
# cannot see a unit held at ONE ACTIVE VALUE: its input sits above -3 on every sample, so it reads live, although it
# passes the same number forward whatever the moment. Measured on H84 (docs/gain_of_function_v4/agents/
# memory_deploy_readings_2026-09-26.md, the replay construction at 415,025): the vision and touch prediction heads read
# 100 of 100 units live by the -3 line while 2 and 0.5 of 100 varied from moment to moment (100 of 100 at 344,400,
# before the memory bias grew from 5-6 to 105-267 per number). A unit VARIES in a window when its output's range over
# the window's samples exceeds its floor: the larger of (a) one bf16 step at the unit's own largest output (production
# computes in bf16: a unit held at 40 moves in steps of 0.25, so a smaller spread is rounding, not response) and (b)
# the widest output a unit under the -3 line can give (GELU: |GELU(-3)| = 0.00405, its outputs lie in (GELU(-3), 0];
# ReLU: 0). The gauge is read-only. The revival chooses by this measure only under --liveness-rule varying (WHICH
# UNITS THE REVIVAL CHOOSES, below); by default it still chooses by the -3 line.
def activation_kind(act) -> Optional[str]:
    """'gelu', 'gelu_tanh' or 'relu' for the activation after a monitored Linear; None otherwise."""
    if isinstance(act, nn.ReLU):
        return "relu"
    if isinstance(act, nn.GELU):
        return "gelu_tanh" if getattr(act, "approximate", "none") == "tanh" else "gelu"
    return None


def activate(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return F.relu(x)
    return F.gelu(x, approximate="tanh" if kind == "gelu_tanh" else "none")


def varying_floor(lo: torch.Tensor, hi: torch.Tensor, kind: str) -> torch.Tensor:
    """Per unit, the smallest output range that counts as varying (see A UNIT THAT VARIES): the larger of one bf16
    step at the unit's largest |output| (2 ** (exponent - 8) for a value m x 2 ** exponent, m in [0.5, 1); bf16 keeps
    8 significant bits) and the widest output below the -3 line."""
    top = torch.maximum(lo.abs(), hi.abs()).float()
    _, e = torch.frexp(top)
    step = torch.pow(2.0, (e - 8).float())
    dead = 0.0 if kind == "relu" else abs(float(activate(torch.tensor(LIVE_LINE), kind)))
    return torch.clamp(step, min=dead)


# WHICH UNITS THE REVIVAL CHOOSES (--liveness-rule, 2026-09-26; the operator's adopted review: "Alive must mean the unit
# actually varies/responds, not merely that its activation sits above a threshold").
#   "live_line" (the default, the rule every run before this switch trained with): a unit is revived when its input
#       never rose above LIVE_LINE over the watching window. Bit-identical to the code before the switch.
#   "varying": a unit is revived when its output does not VARY over the watching window (A UNIT THAT VARIES: its
#       output range over the window's rows is at most varying_floor, the gauge's layers_varying measure), or when its
#       input never rose above LIVE_LINE. A unit under the line always has a range under |GELU(-3)| = 0.00405, the
#       floor's smallest value, so the second clause adds nothing the first misses; it is kept so the old choice is a
#       subset of the new one by construction. A unit held at one ACTIVE value (the vision and touch prediction heads
#       at 415,025: 100 of 100 above the line, 43 and 4 of 100 varying) is now revived - but only when its drawn
#       replacement would itself vary on the watched inputs (A HELD UNIT IS REVIVED ONLY WHEN ITS REPLACEMENT VARIES,
#       in LivenessRepair.revive); otherwise it keeps its weights and the record counts it.
#   "none" (2026-09-26, for the restart after the memory repair's trial): NO UNIT IS REVIVED and no window is watched.
#       Every saved part is built, loaded, routed and trained exactly as under the other rules: the revived units'
#       rows, their running centres (which keep following the input, so a unit held still by a lagging centre can
#       come back) and their own Adam ages, and the all-element paths. The monitor reports as in every mode (layers,
#       layers_varying, not_varying, pools). Why not --liveness-repair report: report builds none of these parts, so
#       LemLite.load_checkpoint refuses a checkpoint trained in mode "on" (H84's), and a forced load would drop the
#       routing of every revived unit and both all-element paths. The memory trial's clean arm reached this state by
#       keeping mode "on" and setting the watching window to 2 ** 63 - 1 rows (docs/gain_of_function_v4/
#       persistent_memory_trial_trainer.py, not in production); this is that state as a production switch.
# The watching window (REVIVE_MIN_SAMPLES rows, restarted at every load) and the function-preserving switch (the mean
# contribution folded into the next layer's bias, fresh Adam ages) are the same under live_line and varying.
REVIVE_RULES = ("live_line", "varying", "none")
DEFAULT_REVIVE_RULE = "live_line"

# THE LAYERS THE REPAIR MAY REVIVE: every layer the lead's sweep found under LAYER_LINE (the share
# of units live on 24 lived moments of ckpt_step_344400, 2026-09-22).
SWEEP_2026_09_22 = {
    "tenure_heads.pose.0": 0.0,
    "wellness.approximate.net.0": 0.051,
    "wellness.approximate.net.2": 0.367,
    "motor_controller.residual_blocks.0.0": 0.047,
    "recon_decoders.vision.0": 0.096,
    "recon_decoders.tactile.0": 0.049,
    "recon_decoders.audio.0": 0.375,
    "recon_decoders.proprioception.0": 0.169,
    "persistent_memory.outcome.0": 0.365,
}
# Additional failures measured across segment 32's reports through step 359800.
# New repair paths begin unused; saved weights and existing repair paths survive the load.
# Evidence: Saved/gof_takeover_20260923/health_repair/before_replay_training_stats.jsonl.
REPAIR_LAYERS: Tuple[str, ...] = tuple(SWEEP_2026_09_22) + (
    "prediction_heads.heads.vision.mlp.0",
    "prediction_heads.heads.tactile.mlp.0",
    "prediction_heads.heads.vestibular.mlp.0",
    "recon_decoders.vestibular.0",
    "motor_controller.confidence_head.0",
    "contact_head.ctx1",
)

# THE WATCHING WINDOW before a layer's dead units are chosen: rows of the layer's input seen since
# the flag came on (every call counts, as in the sweep). None = unmeasured, and "on" refuses to build.
REVIVE_MIN_SAMPLES: Optional[int] = 800
REVIVE_MEASUREMENT: dict = {
    "source": "docs/gain_of_function_v3/measurements/liveness_m1b_inputs.json",
    "body_moments": 800, "bodies": 8,
    "choice": "800 observed input rows before repair; the longest common measured window",
    "limitation": "rare responding units continued to appear; this is not a proved plateau",
}

# THE RUNNING MEAN THAT CENTRES A REVIVED UNIT'S INPUT: an exponential average over the inputs the
# layer is trained on, with this horizon, divided by the weight it has accumulated (corrected for
# starting at zero) - lem/model/contact_head.py CENTRE_HORIZON_STEPS and its arithmetic: the part the
# centre removes is shared by every moment, so any horizon longer than a few moments finds it; 2,000
# follows the slow drift of the weights.
CENTRE_HORIZON_STEPS = 2000

# HOW OFTEN THE MONITOR SAMPLES: every k-th model step, k the first of SAMPLE_PERIODS that shares no
# factor with the order's hold (--motor-hold) or the planner's period (--intention-every), so the
# sampled steps fall on every phase of an order's tenure (the tenure pose head is read only at order
# boundaries) and on planning steps in their own proportion; a period that divided the hold would
# sample one phase only (the rule of lem/model/contact_head.control_period). On H84 (hold 5, planner
# every 10) k = 7. SAMPLE_EVERY overrides the rule when set (tests, probes). The cost per sampled
# step is SAMPLE_MEASUREMENT.
SAMPLE_PERIODS = (7, 11, 13, 17, 19, 23)
SAMPLE_EVERY: Optional[int] = None
SAMPLE_MEASUREMENT: dict = {
    "source": "docs/gain_of_function_v3/measurements/liveness_m1b_inputs.json",
    "hook_median_ms": 100.28484981739894, "step_median_ms": 938.3833408355713,
    "attention_query_rows": 64, "maximum_entropy_share_error": 0.013771004974842072,
    "device": "cpu", "gpu_cost": None,
}


def sample_period(motor_hold: int, plan_every: int = 1) -> int:
    for p in SAMPLE_PERIODS:
        if math.gcd(p, max(1, int(motor_hold))) == 1 and math.gcd(p, max(1, int(plan_every))) == 1:
            return p
    return SAMPLE_PERIODS[-1]
# attention spread is recomputed on at most this many calls of each attention module per sampled
# step (the planner calls the core about 20 times on a planning step; the first calls of a step are
# the lived moment and the next-frame lesson), each on one body (rotating over the sampled steps)
ATTENTION_CALLS_PER_STEP = 2
# and on at most this many query rows of the call, evenly spaced. MEASURED 2026-09-23 (scratch
# m1_inputs.py, ckpt_step_344400, CPU, 1 body, learning off): recomputing every row cost a median
# 305 ms per moment against a 1,294 ms step, most of it the core's bridge attention (768 queries x 768
# keys x 8 heads per call). The selected rows' weights are exactly the full computation's rows
# (test_liveness.py); how far their mean spread sits from every row's is measured on the checkpoint
# (scratch m1b, recorded in agents/lane-liveness-repair.md)
ATTENTION_ROWS = 64
# the repaired pools' token_read spread is measured on up to this many stored pool inputs per window
# with this many random probe directions each (Hutchinson estimate of the per-element sensitivity)
READ_INPUTS_PER_WINDOW = 8
READ_PROBES = 4
# the revival measures each layer pair's output (the next layer's, on the newest watched inputs)
# before and after it changes anything, on at most this many rows: the function-preservation record
PROBE_ROWS = 64

# THE ALL-ELEMENT PATHS: each element (a handle, a contact row) is LayerNormed, projected to RANK
# numbers by weights shared across elements, and the concatenation reaches the token through weights
# of its own per element. RANK = the width of the row the element arrived as (19 numbers per motor
# row: 10 achieved + 9 ordered; 19 per touch row, TOUCH_TOKEN_F), so the path can carry as many
# independent numbers per element as the element's own row holds, and no more. The control that
# catches the size lying is the token_read spread and the per-handle readability of the motor token
# (measurements/per_handle_readability.py) with the path against without it.
POOL_PATHS = ("motor", "tactile")
PATH_RANK = {"motor": 19, "tactile": 19}


def unmeasured() -> List[str]:
    """The constants the repair needs that have no measurement yet."""
    out = []
    if REVIVE_MIN_SAMPLES is None:
        out.append("REVIVE_MIN_SAMPLES")
    if not SAMPLE_MEASUREMENT:
        out.append("SAMPLE_MEASUREMENT")
    return out


def _key(name: str) -> str:
    """A layer name as a ModuleDict key (no dots allowed there)."""
    return name.replace(".", "__")


def unwatched(model):
    """The model's LivenessRepair.unwatched() block, or a block that does nothing when the model has no
    liveness monitor (--liveness-repair off). For forward passes that only measure."""
    lv = getattr(model, "liveness", None)
    if lv is None or not hasattr(lv, "unwatched"):
        return contextlib.nullcontext()
    return lv.unwatched()


def resolve(model, name: str) -> Optional[nn.Module]:
    """The module a sweep name points at on a LemLite (first part an attribute, the rest a path of
    child names), or None when this build has no such part."""
    head, _, rest = name.partition(".")
    root = getattr(model, head, None)
    if not isinstance(root, nn.Module):
        return None
    try:
        return root.get_submodule(rest) if rest else root
    except AttributeError:
        return None


def layer_parts(model, name: str):
    """(linear, activation, next_linear) for a sweep layer name 'seq.i' whose Sequential holds
    Linear, GELU/ReLU, Linear at i, i+1, i+2; None when the build lacks the layer. Raises when the
    layer exists with another structure (a revival there would not be function-preserving)."""
    if name == "contact_head.ctx1":
        head = getattr(model, "contact_head", None)
        if head is None:
            return None
        return head.ctx_lat, nn.GELU(), head.ctx2
    parent_name, _, idx = name.rpartition(".")
    seq = resolve(model, parent_name)
    if seq is None:
        return None
    kids = list(seq.children())
    i = int(idx)
    if not (isinstance(seq, nn.Sequential) and i + 2 < len(kids) and isinstance(kids[i], nn.Linear)
            and isinstance(kids[i + 1], (nn.GELU, nn.ReLU)) and isinstance(kids[i + 2], nn.Linear)):
        raise ValueError("liveness repair: %s is not Linear, GELU/ReLU, Linear inside an nn.Sequential; "
                         "reviving it would not keep its output" % name)
    return kids[i], kids[i + 1], kids[i + 2]


# ------------------------------------------------------------------------------ revived rows
class RevivedRows(nn.Module):
    """The revived units of one Linear layer (in_f -> out_f).

    mask[j] says unit j is revived: its input is lin(norm(x - centre))[j] instead of the layer's own
    row. done says the layer's first observation window has ended; later windows keep watching every unit. Both
    are in the checkpoint. The centre is a running statistic and rides runtime_state.pt."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.in_f, self.out_f = int(in_f), int(out_f)
        self.register_buffer("mask", torch.zeros(self.out_f, dtype=torch.bool))
        self.register_buffer("done", torch.zeros((), dtype=torch.bool))
        # Adam stores one age per tensor. Reset rows need their own bias-correction age,
        # otherwise their first updates use a mature tensor's uncorrected moments.
        self.register_buffer("adam_birth_weight", torch.zeros(self.out_f, dtype=torch.float64))
        self.register_buffer("adam_birth_bias", torch.zeros(self.out_f, dtype=torch.float64))
        self.adam_correction_py = False
        self.register_buffer("centre_mean", torch.zeros(self.in_f), persistent=False)
        self.centre_weight = 0.0
        # Python copies of `done` and "any unit revived", so no call reads a device tensor (a read
        # would make the card wait for the host on every forward of the layer)
        self.done_py = False
        self.any_py = False
        self.norm = nn.LayerNorm(self.in_f)
        self.lin = nn.Linear(self.in_f, self.out_f)
        # the watching window before the pass (plain tensors; reset at every load)
        self.seen_max: Optional[torch.Tensor] = None
        self.act_sum: Optional[torch.Tensor] = None
        self.seen_n = 0
        # --liveness-rule varying: each unit's smallest and largest OUTPUT over the window (A UNIT THAT VARIES); not
        # kept under the default rule, whose window records exactly what it recorded before the switch
        self.track_range = False
        self.seen_lo: Optional[torch.Tensor] = None
        self.seen_hi: Optional[torch.Tensor] = None
        # the newest inputs the layer received while watched (at most PROBE_ROWS rows): the revival
        # measures the layer pair's output on them before and after it changes anything
        self.probe_x: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update_centre(self, x: torch.Tensor) -> None:
        """One call's inputs (all rows pooled) into the running mean; the first call sets it."""
        a = 1.0 / float(CENTRE_HORIZON_STEPS)
        m = x.detach().float().reshape(-1, self.in_f).mean(dim=0).to(self.centre_mean)
        w_old = float(self.centre_weight)
        w_new = (1.0 - a) * w_old + a
        self.centre_mean.mul_((1.0 - a) * w_old / w_new).add_(m * (a / w_new))
        self.centre_weight = w_new

    @torch.no_grad()
    def watch(self, x: torch.Tensor, pre: torch.Tensor, act: nn.Module) -> None:
        """One call's pre-activations into the watching window: the largest per unit and the sum of
        the unit's output (for the fold into the next layer's bias); its inputs as the probe."""
        p = pre.detach().float().reshape(-1, self.out_f)
        mx = p.max(dim=0).values
        a = act(p)
        s = a.sum(dim=0)
        self.seen_max = mx if self.seen_max is None else torch.maximum(self.seen_max, mx)
        self.act_sum = s if self.act_sum is None else self.act_sum + s
        if self.track_range:
            lo, hi = a.min(dim=0).values, a.max(dim=0).values
            self.seen_lo = lo if self.seen_lo is None else torch.minimum(self.seen_lo, lo)
            self.seen_hi = hi if self.seen_hi is None else torch.maximum(self.seen_hi, hi)
        self.seen_n += int(p.shape[0])
        self.probe_x = x.detach().float().reshape(-1, self.in_f)[:PROBE_ROWS].clone()

    def apply(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """The layer's output with the revived units' inputs replaced (unchanged when none is)."""
        if not self.any_py:
            return pre
        xc = x.float() - self.centre_mean
        z = self.lin(self.norm(xc)).to(pre.dtype)
        return torch.where(self.mask, z, pre)


def _adam_rows(optimizer, param, idx, dim, seed_sq_from_rest: bool):
    """Reset the Adam moments of `param` along `dim` at `idx`: first moment 0; second moment 0, or,
    when seed_sq_from_rest, the mean of the second moment over the untouched positions of the same
    row (see RevivedRows.revive). Returns what was done, for the record."""
    if optimizer is None:
        return "no optimizer"
    st = optimizer.state.get(param)
    if not st:
        return "no saved moments"
    ea, es = st.get("exp_avg"), st.get("exp_avg_sq")
    if not (torch.is_tensor(ea) and torch.is_tensor(es)):
        return "no saved moments"
    sel = [slice(None)] * es.dim()
    sel[dim] = idx
    ea[tuple(sel)] = 0.0
    if seed_sq_from_rest:
        keep = torch.ones(es.shape[dim], dtype=torch.bool, device=es.device)
        keep[idx] = False
        if bool(keep.any()):
            rest = es.index_select(dim, keep.nonzero().flatten())
            seed = rest.mean(dim=dim, keepdim=True)
            es[tuple(sel)] = seed.expand_as(es[tuple(sel)])
            return "first moment 0, second moment seeded from the live columns"
        # every column is revived: no live column to learn a scale from, so the tensor's whole
        # history is dropped and Adam starts it fresh with its own bias correction
        optimizer.state.pop(param, None)
        return "every column revived: moments dropped, fresh start with bias correction"
    es[tuple(sel)] = 0.0
    return "moments 0"


def install_row_adam(optimizer, rows):
    """Give reset incoming rows fresh Adam ages without changing other rows' history.

    AdamW still computes gradients, moments and every untouched update. For reset rows,
    its post-step hook uses those same new moments with the row's own age. This includes
    epsilon and decoupled weight decay, and leaves a skipped fused-AMP step untouched.
    Handles live only in the returned closure, keeping execution snapshots serializable.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('liveness row restart requires AdamW')
    # Probe restoration replaces buffer tensors. Resolve their names at each step,
    # so a hook never keeps reading a discarded pre-restore buffer.
    owners = {id(p): (row, birth) for row in rows for p, birth in
              ((row.lin.weight, 'adam_birth_weight'), (row.lin.bias, 'adam_birth_bias'))}
    pending = []

    @torch.no_grad()
    def before(opt, args, kwargs):
        pending.clear()
        for group in opt.param_groups:
            for p in group['params']:
                owned = owners.get(id(p))
                if owned is None or p.grad is None or not owned[0].adam_correction_py:
                    continue
                birth = getattr(owned[0], owned[1])
                state = opt.state.get(p) or {}
                step = float(state.get('step', 0))
                # Frozen probes may load weights without optimizer history. If such a
                # model later trains with a fresh optimizer, that tensor is fresh too.
                birth[birth > step] = 0
                idx = (birth > 0).nonzero().flatten()
                if idx.numel():
                    pending.append((p, group, idx, birth[idx].clone(), p[idx].clone(),
                                    step))

    @torch.no_grad()
    def after(opt, args, kwargs):
        for p, group, idx, birth, old, previous_step in pending:
            state = opt.state[p]
            current_step = float(state['step'])
            if current_step == previous_step:
                continue  # fused AMP skipped the update
            age = (current_step - birth).clamp_min(1)
            shape = (-1,) + (1,) * (p.ndim - 1)
            beta1, beta2 = group['betas']
            bc1 = (1 - beta1 ** age).reshape(shape).to(p)
            bc2 = (1 - beta2 ** age).sqrt().reshape(shape).to(p)
            variance = state['max_exp_avg_sq'] if group.get('amsgrad') else state['exp_avg_sq']
            denom = variance[idx].sqrt() / bc2 + group['eps']
            p[idx] = old * (1 - group['lr'] * group['weight_decay']) - (
                group['lr'] * state['exp_avg'][idx] / bc1 / denom)
        pending.clear()

    pre = optimizer.register_step_pre_hook(before)
    post = optimizer.register_step_post_hook(after)
    return lambda: (pre.remove(), post.remove())


# ---------------------------------------------------------------------------- the pool paths
class AllElementPath(nn.Module):
    """Beside an attention pool over N elements of width d: LayerNorm each element, project it to
    `rank` numbers (weights shared by the elements), and map the concatenation (each element through
    its own weights) to the pool's output width. The output layer starts at zero."""

    def __init__(self, n_elements: int, d: int, rank: int):
        super().__init__()
        self.n, self.d, self.rank = int(n_elements), int(d), int(rank)
        self.norm = nn.LayerNorm(self.d)
        self.down = nn.Linear(self.d, self.rank, bias=False)
        self.up = nn.Linear(self.n * self.rank, self.d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, N, d] (the pool's keys) -> [B, 1, d]."""
        B, N, d = x.shape
        if N != self.n or d != self.d:
            raise ValueError("all-element path built for %d elements of width %d got %d of width %d"
                             % (self.n, self.d, N, d))
        z = self.down(self.norm(x.float()))
        return self.up(z.reshape(B, 1, N * self.rank))


# ------------------------------------------------------------------------ attention weights
def attention_weights(mha: nn.MultiheadAttention, args: tuple, kwargs: dict, body: int = 0,
                      max_rows: Optional[int] = None):
    """The attention weights [heads, Lq, Lk] an nn.MultiheadAttention call computed for one batch
    element (index `body` modulo the batch), recomputed from its inputs and its own projection
    weights (the call itself is not touched). None when the call uses a feature this does not
    reproduce (bias_k, add_zero_attn). max_rows: at most this many query rows, evenly spaced (their
    mean entropy estimates the mean over every row at a fraction of the cost)."""
    if mha.bias_k is not None or mha.add_zero_attn:
        return None
    names = ("query", "key", "value", "key_padding_mask", "need_weights", "attn_mask",
             "average_attn_weights", "is_causal")
    got = dict(zip(names, args))
    got.update(kwargs)
    q, k = got["query"], got["key"]
    if not mha.batch_first:
        q, k = q.transpose(0, 1), k.transpose(0, 1)
    b = int(body) % q.shape[0]
    q, k = q[b:b + 1].float(), k[b:b + 1].float()
    E, H = mha.embed_dim, mha.num_heads
    hd = E // H
    if mha._qkv_same_embed_dim:
        Wq, Wk = mha.in_proj_weight[:E], mha.in_proj_weight[E:2 * E]
    else:
        Wq, Wk = mha.q_proj_weight, mha.k_proj_weight
    bq = bk = None
    if mha.in_proj_bias is not None:
        bq, bk = mha.in_proj_bias[:E], mha.in_proj_bias[E:2 * E]
    rows = None
    if max_rows is not None and q.shape[1] > int(max_rows):
        rows = torch.linspace(0, q.shape[1] - 1, int(max_rows), device=q.device).round().long()
        q = q[:, rows]
    Q = F.linear(q, Wq.float(), None if bq is None else bq.float())
    K = F.linear(k, Wk.float(), None if bk is None else bk.float())
    Lq, Lk = Q.shape[1], K.shape[1]
    Q = Q.view(1, Lq, H, hd).transpose(1, 2)
    K = K.view(1, Lk, H, hd).transpose(1, 2)
    # the module's own scores: the dot product over sqrt(hd), or bound x cosine for a module that carries an
    # attention cap (ATTENTION_CAP, lem/model/attention_cap.py head_scores)
    s = head_scores(mha, Q, K)                              # [1, H, Lq, Lk]
    am = got.get("attn_mask")
    if am is None and got.get("is_causal"):
        Lq_all = got["query"].shape[1 if mha.batch_first else 0]
        am = torch.triu(torch.ones(Lq_all, Lk, dtype=torch.bool, device=s.device), diagonal=1)
    if am is not None:
        if am.dim() == 3:
            am = am.view(-1, H, am.shape[-2], Lk)[b:b + 1]
        if rows is not None:
            am = am[..., rows, :]
        s = s.masked_fill(am, float("-inf")) if am.dtype == torch.bool else s + am.float()
    kpm = got.get("key_padding_mask")
    if kpm is not None:
        kb = kpm[b:b + 1].view(1, 1, 1, Lk)
        s = s.masked_fill(kb, float("-inf")) if kb.dtype == torch.bool else s + kb.float()
    return torch.softmax(s, dim=-1)[0]


def entropy_share(w: torch.Tensor) -> torch.Tensor:
    """Mean over rows of the entropy of each row of weights over its last axis, as a share of the
    even-spread maximum log(n) (the sweep's definition: n = every key, masked or not)."""
    n = w.shape[-1]
    wc = w.clamp_min(1e-12)
    ent = -(w * wc.log()).sum(dim=-1)
    return torch.nan_to_num(ent / math.log(max(n, 2)), nan=0.0).mean()


# --------------------------------------------------------------------------------- the module
class LivenessRepair(nn.Module):
    """The monitor (every mode but "off") and, in mode "on", the revived rows and the pool paths.

    Held on the model as LemLite.liveness (a public name: the episode companion does not copy it;
    its learned part is a side module "liveness_repair" in the checkpoint, its centres ride
    runtime_state.pt). Plain attributes only (tensors, numbers, strings, lists, dicts), so the
    probes' execution snapshot can copy it; hook handles are held inside closures."""

    def __init__(self, model, mode: str, *, seed: int = 0, sample_every: Optional[int] = None,
                 revive_min_samples: Optional[int] = None, layers: Sequence[str] = REPAIR_LAYERS,
                 pools: Sequence[str] = POOL_PATHS, revive_rule: str = DEFAULT_REVIVE_RULE):
        super().__init__()
        if mode not in ("report", "on"):
            raise ValueError("liveness repair mode must be report or on (off builds nothing)")
        if revive_rule not in REVIVE_RULES:
            raise ValueError("liveness revive rule must be one of %s" % ", ".join(REVIVE_RULES))
        if revive_rule != DEFAULT_REVIVE_RULE and mode != "on":
            raise ValueError("liveness revive rule %s chooses which units the repair revives; it needs "
                             "--liveness-repair on (mode %s builds no revived units and revives nothing)"
                             % (revive_rule, mode))
        self.mode = str(mode)
        self.revive_rule = str(revive_rule)
        self.seed = int(seed)
        if sample_every is None:
            sample_every = SAMPLE_EVERY or sample_period(getattr(model, "efference_tenure_steps", 1),
                                                         getattr(model, "intention_every", 1))
        self.sample_every = int(sample_every)
        self.revive_min_samples = int(REVIVE_MIN_SAMPLES if revive_min_samples is None
                                      else revive_min_samples)
        if self.sample_every < 1 or self.revive_min_samples < 1:
            raise ValueError("liveness: sample_every and revive_min_samples must be at least 1")
        self.rows = nn.ModuleDict()
        self.paths = nn.ModuleDict()
        self.layer_names: List[str] = []      # the REPAIR_LAYERS this build has
        self.absent_layers: List[str] = []    # the ones it lacks (feature off)
        self.revived: Dict[str, List[int]] = {}
        self.revived_at_step: Dict[str, int] = {}
        self.revival_record: Dict[str, dict] = {}
        # --liveness-rule varying: per layer, the sums over every window that ended since the last stats report
        # (THE WINDOWS IN WHICH THE CHECK KEEPS HELD UNITS, in revive); the stats block's revival_windows
        self.window_tally: Dict[str, Dict[str, int]] = {}
        # A saved 'done' mask proves an old observation window, not a fresh one after resume.
        self.completed_windows: Dict[str, int] = {}
        if self.mode == "on":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + 6_000_006)
                for name in layers:
                    parts = layer_parts(model, name)
                    if parts is None:
                        self.absent_layers.append(name)
                        continue
                    lin = parts[0]
                    self.layer_names.append(name)
                    in_f = lin.in_features
                    if name == "contact_head.ctx1":
                        in_f += model.contact_head.ctx_ord.in_features
                    self.rows[_key(name)] = RevivedRows(in_f, lin.out_features)
                    self.rows[_key(name)].track_range = self.revive_rule == "varying"
                for enc_name in pools:
                    enc = model.encoder_registry.encoders[enc_name] \
                        if enc_name in model.encoder_registry.encoders else None
                    if enc is None or not hasattr(enc, "pool_attn"):
                        continue
                    n = self._elements(model, enc_name)
                    self.paths[enc_name] = AllElementPath(n, enc.pool_attn.embed_dim, PATH_RANK[enc_name])
        # ---- the monitor's window (plain types)
        self.active = False
        self.step_seen = -1
        self.attached = False
        self.attached_ids: List[int] = []
        self.linear_names: List[str] = []
        self.attention_names: List[str] = []
        self.win_live: Dict[str, torch.Tensor] = {}
        # each monitored unit's smallest and largest OUTPUT (after its activation) over the window: A UNIT THAT VARIES
        self.win_lo: Dict[str, torch.Tensor] = {}
        self.win_hi: Dict[str, torch.Tensor] = {}
        self.act_kinds: Dict[str, str] = {}
        self.win_rows: Dict[str, int] = {}
        self.win_att: Dict[str, torch.Tensor] = {}
        self.win_att_n: Dict[str, int] = {}
        self.step_att_calls: Dict[str, int] = {}
        self.read_inputs: Dict[str, List[torch.Tensor]] = {}
        self.read_stored_step: Dict[str, int] = {}
        self.sampled_steps = 0
        self.sampled_bodies = 0
        self.hook_s = 0.0
        self.window_steps = 0
        # MEASUREMENT PASSES ARE NOT WATCHED (2026-10-02, docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md).
        # While this count is above 0 (inside unwatched()), the repaired layers still route their revived units
        # (RevivedRows.apply), so a measurement reads the production output, but no row joins a watching window,
        # no running centre moves and the monitor records nothing. Without it every extra no-gradient forward pass
        # a gauge adds through a repaired layer (the prediction heads' vision, tactile and vestibular first layers
        # are three) would add its rows to that layer's window of REVIVE_MIN_SAMPLES rows, change when the window
        # fills and what its fold into the next layer's bias carries, and so change the weights. A plain integer:
        # never saved, never in the checkpoint.
        self.unwatched_depth = 0
        self._detach_hooks = lambda: None

    def add_layer(self, model, name: str, seed: int) -> bool:
        """One more repair layer, added after construction and before attach_repair (2026-10-03, the asked-change
        path's hidden layer, --motor-asked-change on). Its revived rows are built in a random stream of their own
        (seed), so every layer and pool path built above keeps the random draws it has without this layer: a build
        with the extra layer starts from the same weights everywhere else. Mode "on" only (mode "report" builds no
        rows; the monitor reports the layer by its own rule in every mode). False when the model lacks the layer."""
        if self.mode != "on":
            return False
        parts = layer_parts(model, name)
        if parts is None:
            self.absent_layers.append(name)
            return False
        if name in self.layer_names:
            raise ValueError("liveness: %s is one of the repair's layers already" % name)
        lin = parts[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            rows = RevivedRows(lin.in_features, lin.out_features)
        rows.track_range = self.revive_rule == "varying"
        self.layer_names.append(name)
        self.rows[_key(name)] = rows
        return True

    @staticmethod
    def _elements(model, enc_name: str) -> int:
        """How many elements the encoder's pool reads: the body's handles for motor, the touch
        token's rows for touch (lem_model.TOUCH_TOKENS)."""
        if enc_name == "motor":
            return len(model.handle_names)
        from mimo_bridge.lem_model import TOUCH_TOKENS
        return int(TOUCH_TOKENS)

    def stamp(self) -> dict:
        """What the checkpoint's receipt records: the mode, the rules and what was revived. The revive rule is
        named only when it is not the default, so a default-rule receipt is the receipt written before the switch."""
        out = {"version": LIVENESS_VERSION, "mode": self.mode, "live_line": LIVE_LINE,
               "layers": list(self.layer_names), "absent_layers": list(self.absent_layers),
               "revive_min_samples": self.revive_min_samples, "sample_every": self.sample_every,
               "centre_horizon_steps": CENTRE_HORIZON_STEPS,
               "paths": {k: {"elements": p.n, "rank": p.rank} for k, p in self.paths.items()},
               "revived": {k: len(v) for k, v in self.revived.items()},
               "revived_at_step": dict(self.revived_at_step)}
        if self.revive_rule != DEFAULT_REVIVE_RULE:
            out["revive_rule"] = self.revive_rule
        return out

    # -------------------------------------------------------------------------- the hooks
    def attach_repair(self, model) -> None:
        """Mode "on": the hooks that route the revived units and add the pool paths. Registered
        before the monitor's hooks, so the monitor sees the repaired outputs."""
        handles = []
        # --liveness-rule none: no window ever ends, so nothing is watched (the routing and the centres run as always)
        watched = self.revive_rule != "none"
        for name in self.layer_names:
            lin, act, _nxt = layer_parts(model, name)
            rows = self.rows[_key(name)]

            if name == "contact_head.ctx1":
                kind = activation_kind(act)
                if kind is not None:
                    self.act_kinds[name] = kind

                def contact_hook(mod, inp, out, rows=rows, act=act, name=name):
                    x = torch.cat((inp[1], inp[2]), dim=-1)
                    result = rows.apply(x, out)
                    measuring = getattr(self, "unwatched_depth", 0) > 0
                    if watched and not measuring:
                        rows.watch(x, result, act)
                    if torch.is_grad_enabled() and mod.training and not measuring:
                        rows.update_centre(x)
                    self._linear_hook(name)(mod, inp, result)
                    return result
                handles.append(model.contact_head.ctx_pre.register_forward_hook(contact_hook))
                continue

            def hook(mod, inp, out, rows=rows, act=act):
                x = inp[0]
                result = rows.apply(x, out)
                measuring = getattr(self, "unwatched_depth", 0) > 0
                # Watch the values actually sent to the activation, not the replaced old rows.
                if watched and not measuring:
                    rows.watch(x, result, act)
                if torch.is_grad_enabled() and mod.training and mod.weight.requires_grad and not measuring:
                    rows.update_centre(x)
                return result
            handles.append(lin.register_forward_hook(hook))
        for enc_name, path in self.paths.items():
            mha = model.encoder_registry.encoders[enc_name].pool_attn

            def phook(mod, args, kwargs, out, path=path):
                key = args[1] if len(args) > 1 else kwargs["key"]
                o, w = out
                return (o + path(key).to(o.dtype), w)
            handles.append(mha.register_forward_hook(phook, with_kwargs=True))
        prev = self._detach_hooks
        self._detach_hooks = lambda: (prev(), [h.remove() for h in handles])

    def attach_monitor(self, model) -> None:
        """Hook every Linear followed by GELU/ReLU inside an nn.Sequential and every attention
        module of every module the model holds (not this one), named as the lead's sweep named them.
        Called again at each window's end, it hooks modules built since (never twice)."""
        handles = []
        seen = set(self.attached_ids)
        for attr, root in list(vars(model).items()):
            if not isinstance(root, nn.Module) or root is self:
                continue
            for sub, mod in root.named_modules():
                name = attr + ("." + sub if sub else "")
                if isinstance(mod, nn.Sequential):
                    kids = list(mod.children())
                    for i in range(len(kids) - 1):
                        lin = kids[i]
                        if (isinstance(lin, nn.Linear) and isinstance(kids[i + 1], (nn.GELU, nn.ReLU))
                                and id(lin) not in seen):
                            seen.add(id(lin))
                            key = f"{name}.{i}"
                            self.linear_names.append(key)
                            kind = activation_kind(kids[i + 1])
                            if kind is not None:
                                self.act_kinds[key] = kind
                            handles.append(lin.register_forward_hook(self._linear_hook(key)))
                if isinstance(mod, nn.MultiheadAttention) and id(mod) not in seen:
                    seen.add(id(mod))
                    self.attention_names.append(name)
                    pool_of = None
                    for enc_name in POOL_PATHS:
                        if name == f"encoder_registry.encoders.{enc_name}.pool_attn":
                            pool_of = enc_name
                    handles.append(mod.register_forward_hook(self._attention_hook(name, pool_of),
                                                             with_kwargs=True))
        self.attached_ids = sorted(seen)
        self.attached = True
        prev = self._detach_hooks
        self._detach_hooks = lambda: (prev(), [h.remove() for h in handles])

    def _linear_hook(self, key: str):
        def hook(mod, inp, out):
            if not self.active or getattr(self, "unwatched_depth", 0) > 0:
                return None
            t0 = time.perf_counter()
            o = out.detach()
            now = (o > LIVE_LINE).reshape(-1, o.shape[-1]).any(dim=0)
            prev = self.win_live.get(key)
            self.win_live[key] = now if prev is None or prev.shape != now.shape else (prev | now)
            kind = self.act_kinds.get(key)
            if kind is not None:
                # the unit's output range over the window (stays on the device: read once at the window's end)
                post = activate(o.float().reshape(-1, o.shape[-1]), kind)
                lo, hi = post.min(dim=0).values, post.max(dim=0).values
                plo, phi = self.win_lo.get(key), self.win_hi.get(key)
                same = plo is not None and plo.shape == lo.shape
                self.win_lo[key] = torch.minimum(plo, lo) if same else lo
                self.win_hi[key] = torch.maximum(phi, hi) if same else hi
            self.win_rows[key] = self.win_rows.get(key, 0) + int(o.numel() // o.shape[-1])
            self.hook_s += time.perf_counter() - t0
            return None
        return hook

    def _attention_hook(self, key: str, pool_of: Optional[str]):
        def hook(mod, args, kwargs, out):
            if not self.active or getattr(self, "unwatched_depth", 0) > 0:
                return None
            t0 = time.perf_counter()
            n = self.step_att_calls.get(key, 0)
            if n < ATTENTION_CALLS_PER_STEP:
                self.step_att_calls[key] = n + 1
                with torch.no_grad():
                    # which batch element: spread over the batch across sampled steps and calls (a
                    # core block's batch is bodies x moments or bodies x senses, not bodies alone)
                    w = attention_weights(mod, args, kwargs, body=self.sampled_steps * 7919 + 104729 * n,
                                          max_rows=ATTENTION_ROWS)
                if w is not None:
                    e = entropy_share(w).detach()
                    self.win_att[key] = e if key not in self.win_att else self.win_att[key] + e
                    self.win_att_n[key] = self.win_att_n.get(key, 0) + 1
                if (pool_of is not None and self.read_stored_step.get(pool_of) != self.step_seen
                        and len(self.read_inputs.get(pool_of, [])) < READ_INPUTS_PER_WINDOW):
                    k = args[1] if len(args) > 1 else kwargs["key"]
                    self.read_inputs.setdefault(pool_of, []).append(k.detach().float().clone())
                    self.read_stored_step[pool_of] = self.step_seen
            self.hook_s += time.perf_counter() - t0
            return None
        return hook

    def detach(self) -> None:
        """Remove every hook this object registered (tests, tools)."""
        self._detach_hooks()
        self._detach_hooks = lambda: None
        self.attached = False
        self.attached_ids = []

    @contextlib.contextmanager
    def unwatched(self):
        """Forward passes inside this block are measurements: revived units are routed as always, but no row
        joins a watching window, no running centre moves and the monitor records nothing (unwatched_depth)."""
        self.unwatched_depth = getattr(self, "unwatched_depth", 0) + 1
        try:
            yield
        finally:
            self.unwatched_depth -= 1

    # ---------------------------------------------------------------------- per step
    def begin_step(self, model, step: int) -> None:
        self.attach_optimizer(model)
        if not self.attached:
            self.attach_monitor(model)
        self.step_seen = int(step)
        self.window_steps += 1
        self.active = (int(step) % self.sample_every) == 0
        self.step_att_calls = {}

    def end_step(self, model, step: int, batch: int, train: bool) -> None:
        if self.active:
            self.sampled_steps += 1
            self.sampled_bodies += int(batch)
        self.active = False
        if self.mode == "on" and train:
            self.attach_optimizer(model)
            if self.revive_rule == "none":
                return                           # no unit is ever revived (WHICH UNITS THE REVIVAL CHOOSES)
            for name in self.layer_names:
                rows = self.rows[_key(name)]
                if rows.seen_n >= self.revive_min_samples:
                    self.revive(model, name, int(step))

    def attach_optimizer(self, model):
        opt = getattr(model, 'optimizer', None)
        # Registration belongs to the optimizer, which probe snapshots do not replace.
        # A restored module attribute must not register a duplicate pair of hooks.
        owner = getattr(opt, '_liveness_repair_owner', None)
        if self.mode == 'on' and opt is not None and owner != id(self):
            if owner is not None:
                raise RuntimeError('another liveness repair already owns this optimizer')
            remove = install_row_adam(opt, self.rows.values())
            opt._liveness_repair_owner = id(self)
            def detach_optimizer():
                remove()
                if getattr(opt, '_liveness_repair_owner', None) == id(self):
                    del opt._liveness_repair_owner
            previous = self._detach_hooks
            self._detach_hooks = lambda: (previous(), detach_optimizer())

    # ------------------------------------------------------------------------ the revival
    @staticmethod
    @torch.no_grad()
    def _replacement_varies(rows: RevivedRows, px: Optional[torch.Tensor], w: torch.Tensor, b: torch.Tensor,
                            act: nn.Module) -> torch.Tensor:
        """Per drawn replacement (rows of w, b): would it VARY (A UNIT THAT VARIES) on the watched inputs px, reading
        them as a revived unit reads them (RevivedRows.apply: LayerNorm of the input less the running centre)? False
        for every replacement when fewer than two inputs were kept."""
        if px is None or px.shape[0] < 2:
            return torch.zeros(w.shape[0], dtype=torch.bool)
        x = px.float().to(rows.norm.weight.device)
        z = rows.norm(x - rows.centre_mean.to(x)).float()
        post = act(z @ w.to(z).T + b.to(z))
        lo, hi = post.min(0).values, post.max(0).values
        return ((hi - lo) > varying_floor(lo, hi, activation_kind(act) or "gelu").to(hi)).cpu()

    @torch.no_grad()
    def revive(self, model, name: str, step: int) -> dict:
        """End one observation window; revive units silent throughout that window.

        A unit that responded during startup can fall silent later. The original
        one-pass implementation then left it silent forever: on H84's 140-step
        check only 10 of 256 motor units were repaired, while 234 later stopped
        responding. A later collapse of an already repaired unit must also be repaired.
        Responding units retain all their learned weights.

        Adam: the revived units' own new weights live in this module and start with fresh moments
        and their own bias correction. The layer's old rows for those units are set to zero with
        zero moments (they no longer reach the output, so they receive no gradient). The next
        layer's columns for those units are set to zero; their first moment is set to 0 and their
        second moment is SEEDED with the mean second moment of the same row's live columns. Why not
        0: the second-moment correction uses the tensor's shared step count (hundreds of thousands),
        so a zeroed second moment gets no bias correction, and Adam's first updates on those columns
        would be (1 - 0.9^t) / sqrt(1 - 0.999^t) times the learning rate: 3.2 at t = 1, 6.5 at
        t = 10, 3.3 at t = 100, 1.3 at t = 1,000 - several times the step of a live column, for
        hundreds of steps, on weights that start at zero. When every column of the next layer is
        revived (the pose head: 256 of 256) there is no live column; that tensor's moments are
        dropped and Adam starts it fresh with a bias correction of its own."""
        if self.revive_rule == "none":
            raise RuntimeError("--liveness-rule none revives no unit: the saved revived units are kept as they are")
        rows = self.rows[_key(name)]
        lin, act, nxt = layer_parts(model, name)
        opt = getattr(model, "optimizer", None)
        dead = (rows.seen_max <= LIVE_LINE) if rows.seen_max is not None else \
            torch.zeros(rows.out_f, dtype=torch.bool, device=rows.mask.device)
        spread = None
        drawn = None
        held_still = 0
        held_found = 0
        if self.revive_rule == "varying" and rows.seen_lo is not None:
            # WHICH UNITS THE REVIVAL CHOOSES: also every unit whose output range over the window is at most its floor
            below_line = dead
            spread = rows.seen_hi - rows.seen_lo
            still = spread <= varying_floor(rows.seen_lo, rows.seen_hi, activation_kind(act) or "gelu").to(spread)
            dead = below_line | still
            held = dead & ~below_line
            held_found = int(held.sum())
            if held_found:
                # A HELD UNIT IS REVIVED ONLY WHEN ITS REPLACEMENT VARIES. A revived unit reads LayerNorm(input -
                # running centre); when the layer's input barely varies against its shared part, or the centre lags a
                # fast shift of that part, the replacement is held too, and reviving would discard the unit's weights
                # at every window for nothing (415,025, the vision head: its input 12,884 per number against a spread
                # of 8.8, the saved centre 42 spreads from the input's mean; 29 of 111 fresh replacements varied, 0 of
                # 191 with the memory bias removed; agents/liveness_revive_varying_2026-09-26.md). The replacements are
                # drawn here as the revival draws them and judged by the same measure on the watched inputs; a unit
                # under the -3 line is revived as the default rule revives it.
                cand = dead.nonzero().flatten()
                again = bool((dead.to(rows.mask.device) & rows.mask).any())
                g = torch.Generator(device="cpu").manual_seed(self.seed + zlib.crc32(name.encode())
                                                             + (int(step) if again else 0))
                bound = 1.0 / math.sqrt(rows.in_f)
                w_c = (torch.rand(cand.numel(), rows.in_f, generator=g) * 2 - 1) * bound
                b_c = (torch.rand(cand.numel(), generator=g) * 2 - 1) * bound
                keep = below_line.cpu()[cand.cpu()] | self._replacement_varies(rows, rows.probe_x, w_c, b_c, act)
                chosen = torch.zeros_like(dead)
                chosen[cand[keep.to(cand.device)]] = True
                held_still = int((held & ~chosen).sum())
                dead = chosen
                drawn = (w_c[keep], b_c[keep])
        dead = dead.to(rows.mask.device)
        idx = dead.nonzero().flatten()
        rec = {"step": int(step), "samples": int(rows.seen_n), "units": int(rows.out_f),
               "dead": int(idx.numel()),
               "repaired_again": int((dead & rows.mask).sum()),
               "live_share_before": round(1.0 - float(dead.float().mean()), 4)}
        if spread is not None:
            rec["rule"] = self.revive_rule
            rec["below_line"] = int(below_line.sum())
            rec["held_above_line"] = int((dead & ~below_line.to(dead.device)).sum())
            rec["held_kept_replacement_still"] = held_still
        px = rows.probe_x
        contact = model.contact_head if name == "contact_head.ctx1" else None

        def pair_output():
            # the next layer's output on the newest watched inputs, computed without the hooks
            x = px.to(lin.weight)
            if contact is None:
                raw = lin.forward(x)
            else:
                raw = (F.linear(x[:, :lin.in_features], lin.weight, lin.bias)
                       + F.linear(x[:, lin.in_features:], contact.ctx_ord.weight, contact.ctx_ord.bias))
            pre = rows.apply(x, raw)
            return F.linear(act(pre), nxt.weight, nxt.bias).float()
        y_before = pair_output() if px is not None else None
        if idx.numel() > 0:
            dev = nxt.weight.device
            idx_d = idx.to(dev)
            mean_act = (rows.act_sum / max(rows.seen_n, 1)).to(dev, nxt.weight.dtype)
            # the dead units' mean contribution moves into the next layer's bias, so removing their
            # outgoing weights leaves the layer's output as it was up to the per-sample variation
            fold = nxt.weight[:, idx_d] @ mean_act[idx_d]
            rec["fold_max_abs"] = float(fold.abs().max())
            # THE LARGEST CHANGE THE SWITCH CAN MAKE on a watched input: a dead unit's output lies in
            # (GELU(-3), 0] = (-0.00405, 0] (ReLU: exactly 0), so its distance from its own mean is
            # under 0.00405, and a row of the next layer moves by at most 0.00405 x the sum of its
            # weights on the dead units
            floor = 0.0 if isinstance(act, nn.ReLU) else abs(float(F.gelu(torch.tensor(LIVE_LINE))))
            if spread is None:
                rec["output_change_bound"] = float(floor * nxt.weight[:, idx_d].abs().sum(dim=1).max())
            else:
                # under --liveness-rule varying a revived unit may be held at an active value: its distance from its
                # own mean is under its output range over the window (at most one bf16 step at its size, or under
                # 0.00405 below the line), so a row of the next layer moves by at most the sum of |weight| x range
                rng_d = spread.to(dev, torch.float32)[idx_d]
                rec["output_change_bound"] = float((nxt.weight[:, idx_d].abs().float() @ rng_d).max())
            if nxt.bias is not None:
                nxt.bias.add_(fold)
            nxt.weight[:, idx_d] = 0.0
            rec["next_weight_moments"] = _adam_rows(opt, nxt.weight, idx_d, 1, True)
            # the old rows: no longer read (the revived path replaces them), zero with zero moments
            if contact is None:
                lin.weight[idx_d, :] = 0.0
                rec["old_weight_moments"] = _adam_rows(opt, lin.weight, idx_d, 0, False)
                if lin.bias is not None:
                    lin.bias[idx_d] = 0.0
                    _adam_rows(opt, lin.bias, idx_d, 0, False)
            else:
                rec["old_weight_moments"] = "retained for the unchanged contact-context skip path"
            # fresh incoming weights for the revived units, from a generator seeded by the run's seed
            # and the layer's name (the run's own random stream is not drawn from)
            if drawn is None:
                g = torch.Generator(device="cpu").manual_seed(self.seed + zlib.crc32(name.encode())
                                                             + (int(step) if rec["repaired_again"] else 0))
                bound = 1.0 / math.sqrt(rows.in_f)
                w_new = (torch.rand(idx.numel(), rows.in_f, generator=g) * 2 - 1) * bound
                b_new = (torch.rand(idx.numel(), generator=g) * 2 - 1) * bound
            else:
                w_new, b_new = drawn             # the replacements judged above (--liveness-rule varying)
            ri = idx.to(rows.lin.weight.device)
            rows.lin.weight[ri] = w_new.to(rows.lin.weight)
            rows.lin.bias[ri] = b_new.to(rows.lin.bias)
            for p, birth in ((rows.lin.weight, rows.adam_birth_weight),
                             (rows.lin.bias, rows.adam_birth_bias)):
                birth[ri] = float((opt.state.get(p) or {}).get('step', 0)) if opt else 0
                _adam_rows(opt, p, ri, 0, False)
            rows.adam_correction_py = bool(rows.adam_birth_weight.any() or rows.adam_birth_bias.any())
            rec['incoming_adam_age'] = 'restarted per repaired row; other rows retain their history'
            rows.mask[ri] = True
            rows.any_py = True
            rec["centre_weight"] = round(float(rows.centre_weight), 6)
        if y_before is not None:
            # FUNCTION PRESERVATION, measured at the switch: the largest change of the pair's output
            # against the size of that output, on the same inputs
            y_after = pair_output()
            rec["output_change_max"] = float((y_after - y_before).abs().max())
            rec["output_size_mean"] = float(y_before.abs().mean())
            rec["probe_rows"] = int(px.shape[0])
            with torch.no_grad():
                x = px.to(lin.weight)
                raw = lin.forward(x) if contact is None else (
                    F.linear(x[:, :lin.in_features], lin.weight, lin.bias)
                    + F.linear(x[:, lin.in_features:], contact.ctx_ord.weight, contact.ctx_ord.bias))
                pre_after = rows.apply(x, raw)
            alive = (pre_after > LIVE_LINE).any(0)
            rec["live_share_after_on_probe"] = round(float(alive.float().mean()), 4)
            if idx.numel() > 0:
                rec["revived_live_share_on_probe"] = round(
                    float(alive[idx.to(alive.device)].float().mean()), 4)
            if spread is not None and px.shape[0] > 1:
                # the same units' variation on the same watched inputs after the switch (A UNIT THAT VARIES)
                post = act(pre_after.float())
                lo_a, hi_a = post.min(0).values, post.max(0).values
                moves = (hi_a - lo_a) > varying_floor(lo_a, hi_a, activation_kind(act) or "gelu").to(hi_a)
                rec["varying_share_after_on_probe"] = round(float(moves.float().mean()), 4)
                if idx.numel() > 0:
                    rec["revived_varying_share_on_probe"] = round(
                        float(moves[idx.to(moves.device)].float().mean()), 4)
        rows.done.fill_(True)
        rows.done_py = True
        rows.seen_max = rows.act_sum = None
        rows.seen_lo = rows.seen_hi = None
        rows.probe_x = None
        rows.seen_n = 0
        self.completed_windows[name] = self.completed_windows.get(name, 0) + 1
        self.revived[name] = [int(i) for i in rows.mask.nonzero().flatten().tolist()]
        if idx.numel():
            self.revived_at_step[name] = int(step)
            self.revival_record[name] = rec
        elif held_found and not (self.revival_record.get(name) or {}).get("dead"):
            # THE WINDOWS IN WHICH THE CHECK KEEPS HELD UNITS (--liveness-rule varying; Lane E review 2026-09-26, MAJOR).
            # A window that finds held units and revives none (every replacement would be held too, or the newest
            # call left fewer than 2 rows to judge them on) is the reading the deploy watch needs: its record reaches
            # the stats like a revival's. A revival's record from an earlier window of the same report is kept.
            self.revival_record[name] = rec
        if self.revive_rule == "varying":
            # every window of the report, summed, whatever its record: a later window replaces an earlier record
            n_probe = 0 if px is None else int(px.shape[0])
            t = self.window_tally.setdefault(name, {"windows": 0, "revived": 0, "below_line": 0,
                                                    "held_above_line": 0, "held_kept_replacement_still": 0,
                                                    "probe_under_2_rows": 0})
            t["windows"] += 1
            t["revived"] += int(idx.numel())
            t["below_line"] += int(rec.get("below_line", 0))
            t["held_above_line"] += int(rec.get("held_above_line", 0))
            t["held_kept_replacement_still"] += int(held_still)
            # the replacement check judges on the newest call's rows only (RevivedRows.watch): with fewer than 2 it
            # cannot see a replacement vary, so every held unit of that window is kept
            t["probe_under_2_rows"] += int(n_probe < 2)
        if isinstance(getattr(model, "receipt", None), dict):
            model.receipt["liveness_repair"] = self.stamp()
        return rec

    def after_load(self, model, saved: Optional[dict] = None) -> None:
        """After a checkpoint load: the revived lists from the loaded masks; the watching windows
        restart (the samples before a save are not in the checkpoint). saved: the loaded checkpoint's
        receipt block (its liveness_repair stamp). The step at which each layer was last revived is carried
        from it: before 2026-09-26 every resume started revived_at_step empty, so the next receipt kept only
        the revivals since the resume (415,075's names one layer of the 15 it had revived)."""
        self.completed_windows = {}
        for name in self.layer_names:
            rows = self.rows[_key(name)]
            rows.seen_max = rows.act_sum = rows.probe_x = None
            rows.seen_lo = rows.seen_hi = None
            rows.seen_n = 0
            rows.done_py = bool(rows.done)
            rows.any_py = bool(rows.mask.any())
            rows.adam_correction_py = bool(rows.adam_birth_weight.any() or rows.adam_birth_bias.any())
            if rows.done_py:
                self.revived[name] = [int(i) for i in rows.mask.nonzero().flatten().tolist()]
        at = (saved or {}).get("revived_at_step") if isinstance(saved, dict) else None
        for name, s in (at.items() if isinstance(at, dict) else ()):
            if name in self.layer_names and self.revived.get(name) and isinstance(s, int) and not isinstance(s, bool):
                self.revived_at_step[name] = int(s)
        if isinstance(getattr(model, "receipt", None), dict):
            model.receipt["liveness_repair"] = self.stamp()

    # ---------------------------------------------------------------- runtime statistics
    def runtime_state(self) -> dict:
        return {k: {"centre_mean": r.centre_mean.detach().cpu(), "centre_weight": float(r.centre_weight)}
                for k, r in self.rows.items()}

    def load_runtime_state(self, state: Optional[dict]) -> List[str]:
        got = []
        for k, r in self.rows.items():
            s = (state or {}).get(k) or {}
            cm = s.get("centre_mean")
            if torch.is_tensor(cm) and tuple(cm.shape) == tuple(r.centre_mean.shape):
                r.centre_mean.copy_(cm.to(r.centre_mean))
                r.centre_weight = float(s.get("centre_weight", 0.0))
                got.append(k.replace("__", "."))
        return got

    # --------------------------------------------------------------------- the window
    def token_read(self, model, enc_name: str) -> Optional[dict]:
        """How evenly the pooled token of `enc_name` reads its elements, on the window's stored pool
        inputs: per element, the squared size of the token's change per unit change of that
        element (a Hutchinson estimate with READ_PROBES random directions), as shares; the spread
        is their entropy as a share of the even-spread maximum. Also the share of the token's size
        that comes through the all-element path (0 without one)."""
        xs = self.read_inputs.get(enc_name) or []
        if not xs:
            return None
        enc = model.encoder_registry.encoders[enc_name]
        g = torch.Generator(device="cpu").manual_seed(12345)
        infl, path_share = [], []
        was = self.active
        self.active = False
        try:
            for x0 in xs:
                x = x0.to(enc.pool_query.device).requires_grad_(True)
                with torch.enable_grad():
                    B = x.shape[0]
                    q = enc.pool_norm(enc.pool_query.expand(B, -1, -1))
                    out, _ = enc.pool_attn(q, x, x)
                    acc = torch.zeros(x.shape[:2], device=x.device)
                    for _ in range(READ_PROBES):
                        v = torch.randn(out.shape, generator=g).to(out)
                        gr, = torch.autograd.grad((out * v).sum(), x, retain_graph=True)
                        acc = acc + gr.float().pow(2).sum(-1)
                infl.append((acc / acc.sum(-1, keepdim=True).clamp_min(1e-30)).detach())
                if enc_name in self.paths:
                    with torch.no_grad():
                        p = self.paths[enc_name](x.detach())
                        path_share.append(float((p.float().norm(dim=-1) /
                                                 out.detach().float().norm(dim=-1).clamp_min(1e-12)).mean()))
        finally:
            self.active = was
        share = torch.cat(infl).mean(0)                        # [N]
        return {"spread": round(float(entropy_share(share.unsqueeze(0))), 4),
                "largest_element_share": round(float(share.max()), 4),
                "elements_under_1_in_1000": int((share < 1e-3).sum()),
                "path_size_share": round(sum(path_share) / len(path_share), 4) if path_share else 0.0,
                "inputs": len(xs)}

    def window_summary(self, model) -> dict:
        """The top-level `liveness` stats block (field contract 2026-09-23): per hidden layer the
        share of units whose input rose above LIVE_LINE on the window's samples; per attention module
        its spread (and each repaired pool's token_read); the line; how many body-moments were
        sampled; which layers the repair has revived. Resets the window."""
        layers = {k: round(float(v.float().mean()), 4) for k, v in self.win_live.items()}
        # A UNIT THAT VARIES: the share of each layer's units whose output range over the window exceeds its floor
        varying = {}
        for k, lo in self.win_lo.items():
            hi = self.win_hi[k]
            varying[k] = round(float(((hi - lo) > varying_floor(lo, hi, self.act_kinds[k]).to(hi)).float().mean()), 4)
        pools = {k: round(float(self.win_att[k]) / max(self.win_att_n.get(k, 0), 1), 4)
                 for k in self.win_att}
        reads = {}
        for enc_name in POOL_PATHS:
            r = self.token_read(model, enc_name)
            if r is not None:
                reads[enc_name] = r
                pools[f"encoder_registry.encoders.{enc_name}.token_read"] = r["spread"]
        out = {"mode": self.mode, "layers": layers, "pools": pools, "line": LIVE_LINE,
               "layer_line": LAYER_LINE, "samples": int(self.sampled_bodies),
               "sampled_steps": int(self.sampled_steps), "window_steps": int(self.window_steps),
               "sample_every": self.sample_every,
               "hook_ms_per_step": round(1000.0 * self.hook_s / max(self.window_steps, 1), 3),
               "below_line": sorted(k for k, v in layers.items() if v < LAYER_LINE),
               "layers_varying": varying,
               "not_varying": sorted(k for k, v in varying.items() if v < LAYER_LINE),
               "varying_rule": ("the share of a layer's units whose output range over the window's samples exceeds "
                                "one bf16 step at the unit's own size and the widest output below the -3 line; "
                                "not_varying lists layers under half"),
               "token_read": reads,
               "revived": sorted(k for k, v in self.revived.items() if v),
               "revived_units": {k: len(v) for k, v in self.revived.items()},
               "revived_at_step": dict(self.revived_at_step)}
        if self.revive_rule != DEFAULT_REVIVE_RULE:
            out["revive_rule"] = self.revive_rule
        if self.mode == "on":
            if self.revive_rule == "none":
                # no window is watched and none ends (needed None), so no layer's first repair check ever comes: ready
                # stays False, as in the memory trial's clean arm (a window of 2 ** 63 - 1 rows). The H84 hourly watcher
                # skips its liveness stop for a repair layer whose `ready` is not True, so under this rule it judges
                # only the layers outside the repair, as it did for that arm (agents/liveness_revive_varying_2026-09-26.md
                # section 6)
                out["watching"] = {n: {"samples": 0, "needed": None, "ready": False} for n in self.layer_names}
            else:
                out["watching"] = {n: {"samples": int(self.rows[_key(n)].seen_n),
                                       "needed": self.revive_min_samples,
                                       "ready": self.completed_windows.get(n, 0) > 0}
                                   for n in self.layer_names}
            out["revival"] = {k: dict(v) for k, v in self.revival_record.items()}
            self.revival_record = {}
            if self.revive_rule == "varying":
                out["revival_windows"] = {k: dict(v) for k, v in self.window_tally.items()}
                self.window_tally = {}
        self.win_live, self.win_rows, self.win_att, self.win_att_n = {}, {}, {}, {}
        self.win_lo, self.win_hi = {}, {}
        self.read_inputs, self.read_stored_step = {}, {}
        self.sampled_steps = self.sampled_bodies = self.window_steps = 0
        self.hook_s = 0.0
        # modules built since the last window (the entity self tracker is built at the first step)
        self.attach_monitor(model)
        return out
