# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""ATTENTION_CAP: bounded attention scores in the attention modules where the predictor's route merges the body
handles' command changes and where it switches which element it reads (--attention-cap off|on, default off;
--attention-cap-bound B). Record: docs/gain_of_function_v4/attention_cap/ATTENTION_CAP_2026-10-05.md.

WHAT WAS MEASURED (checkpoint 481,800 of run H84_fresh_drivers_20260918_seg41o_1431, Oct 5, 2026; rows
investigate_handle_merge_location_481800 and planner_search_mechanism_controls; records
docs/gain_of_function_v4/coordinator/HANDLE_DIRECTION_LOCATION_2026-10-05.md on branch
investigate/wellness-gradient-motor-pool-2026-10-05 and PLANNER_SEARCH_REPAIR_2026-10-05.md section 1.2 on branch
planner/search-2026-10-05):
  - the transformer's cross-modal attention (ThreeStageBlock.cm_attn: per model step each of the 6 sense slots' queries
    reads the 6 slots, 8 heads of 24 numbers on H84) puts weight 1.000 on one slot with every head at every query of
    block 0 (entropy share 0.0010; 0 = one key, 1 = even); the motor slot's change reaches the proprioception slot
    through one head, whose value-to-output map sends 93 of 100 parts of its squared singular values along one
    direction, so the 26 handles' predicted changes come to lie along one line (mean absolute cosine between two
    handles' changes 0.23 in the motor token, 0.84 after block 0's cross-modal attention, 0.87 in the prediction,
    0.45 in the real changes);
  - a softmax with a weight of 1.000 passes no gradient to its scores (its derivative with respect to every score is
    0), so no lesson can move such a head to another key;
  - under one-handle command turns of 20 degrees, 27 of 1,200 turns at block 0's motor query and 30 of 1,200 at block
    1's vestibular query move a head's weights by more than half (total variation above 0.5);
  - the motor encoder's attention (its 2 attention layers among the 26 handles and its 4-head pool,
    lem/model/encoders/cross_patch.py) changes which handle a head reads under a small command change; holding those 3
    modules' weights removes 14 of 15 parts of the planner's selected command's change under a token change of 1 in
    100,000 (18.5 degrees on the average body handle on 481,800 with nothing held).

WHAT THIS MODULE DOES. Every covered module is built as CappedMultiheadAttention, a torch.nn.MultiheadAttention whose
forward is nn.MultiheadAttention's own, unchanged, while its bound is None (the default: every build with the switch
off or absent). With a bound B > 0 (apply(), called by LemLite under --attention-cap on) each head scores key j for
query i as

    s_ij = B x cos(q_i, k_j) = B x (q_i / max(|q_i|, EPS)) . (k_j / max(|k_j|, EPS))

instead of q_i . k_j / sqrt(head size), where q_i and k_j are the head's query and key vectors after the module's own
in-projection (its weights and biases, unchanged). The softmax over the keys, the weighted sum of the values and the
output projection are the module's own arithmetic. A score lies in [-B, B], so with n keys no key can take more than
1 / (1 + (n - 1) exp(-2B)) of a head's weight (largest_weight) and the softmax always passes gradient to the scores
(least_gradient_share). The scores and the softmax are computed in float32 (float64 when the module computes in
float64) with autocast switched off for them; the weighted sum of the values uses the values' own dtype.

WHAT IT DOES NOT CHANGE. No parameter, buffer or random draw is added: the state_dict's keys, shapes and values are
those of a build without the cap, so a checkpoint loads with the switch on or off, and the existing in-projection and
out-projection weights are what trains. B is a fixed number (not learned). Not covered (their scores stay the dot
product): the transformer's causal-temporal attention (ct_attn) and bridge (cmb_attn), the tactile encoder (built from
the same class, never given a bound), the prediction heads' attention, the intermind encoder and the gaze head.

TOOLS THAT RECOMPUTE ATTENTION WEIGHTS from a module's inputs (the liveness monitor, lem/model/liveness.py
attention_weights; the memory deploy's readings, mimo_bridge/tools/memory_deploy_readings.py attention_entropy) take
their scores from head_scores(), so they read the bounded weights of a capped module and, for every other module, the
same expression they computed before. lem/fused_kernels.py refuses to replace a capped module by its fused
dot-product attention (the UE-side inference graphs; not used by the MIMo trainer).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = "attention_cap_v1"
MODES = ("off", "on")
# THE COSINE'S EPSILON: a query or key vector of length under EPS is divided by EPS instead of its length (its cosine
# with anything is then under its length / EPS), so a zero vector gives a score of 0 and a finite gradient. For a
# vector longer than EPS the score is the exact cosine. MEASURED on checkpoint 481,800 (row attention_cap_bound_481800,
# result.json "shortest_vector", 16 states x 79 commands): the shortest in-projected query or key vector of a covered
# module is 1.005 long (block 2's cross-modal keys); the motor encoder's are 9.8 or longer. EPS changes no score of
# those inputs.
EPS = 1e-6

# THE BOUND THE SWITCH USES WHEN --attention-cap-bound IS NOT GIVEN (--attention-cap on alone). MEASURED on checkpoint
# 481,800 before any training by row attention_cap_bound_481800, whose rule (declared in the row before any reading,
# its release clause added before the registered run) picked 3: of the bounds 1, 1.5, 2, 2.5, 3, 4 and 5, the largest
# at which no key can take more than 99 in 100 of a head's weight among 6 keys (B at most 3.1), the covered modules'
# switching turns fall to a tenth of today's or fewer, and the planner search's change under a token change of 1 in
# 100,000 is at most a tenth of today's. At 3: 219 switching turns against 5,866 without the cap (none at the
# cross-modal queries), the planner's change 0.86 degrees against 18.4 (111 of 128 pairs under 0.1 degrees).
MEASURED_BOUND: Optional[float] = 3.0
MEASUREMENT: dict = {
    "row": "attention_cap_bound_481800",
    "result": "D:/UEMA_Research/external/D/UEMA_scratch/attention_cap/bound_481800/result.json",
    "run": "Oct 5, 2026 5:36 PM to 6:25 PM PDT, commit e7c08f066, outcome PASS (production's test runner)",
    "checkpoint": "H84_fresh_drivers_20260918_seg41o_1431/checkpoints/ckpt_step_481800",
    "picked": 3.0,
    "record": "docs/gain_of_function_v4/attention_cap/ATTENTION_CAP_2026-10-05.md",
}


def largest_weight(n_keys: int, bound: float) -> float:
    """The largest weight one key can take in a head's softmax over n_keys keys when every score lies in [-bound, bound]:
    one key at +bound, the other n - 1 at -bound: 1 / (1 + (n - 1) exp(-2 bound))."""
    n = int(n_keys)
    if n < 1:
        raise ValueError("a softmax needs at least one key")
    return 1.0 / (1.0 + (n - 1) * math.exp(-2.0 * float(bound)))


def least_gradient_share(n_keys: int, bound: float) -> float:
    """sum_j a_j (1 - a_j) at that most concentrated weighting (one key at +bound, n - 1 at -bound): the least share of
    a lesson's gradient the softmax can pass to the scores (0 = none, as at a weight of 1.000)."""
    n = int(n_keys)
    a1 = largest_weight(n, bound)
    ak = (1.0 - a1) / (n - 1) if n > 1 else 0.0
    return a1 * (1.0 - a1) + (n - 1) * ak * (1.0 - ak)


def check_switch(mode, bound) -> Optional[float]:
    """The bound a build applies (a float > 0), or None for the switch off. mode "off" (the default) with a bound
    given is refused (the bound would do nothing); mode "on" takes `bound`, else MEASURED_BOUND, and refuses when
    neither exists or the number is not finite and above 0."""
    mode = "off" if mode is None else str(mode)
    if mode not in MODES:
        raise ValueError("--attention-cap must be 'off' or 'on', not %r" % (mode,))
    if mode == "off":
        if bound is not None:
            raise ValueError("--attention-cap-bound %r was given with --attention-cap off: the bound acts only with "
                             "--attention-cap on" % (bound,))
        return None
    if bound is None:
        bound = MEASURED_BOUND
        if bound is None:
            raise ValueError("--attention-cap on needs --attention-cap-bound: lem/model/attention_cap.py MEASURED_BOUND "
                             "is not measured yet (row attention_cap_bound_481800 measures it)")
    bound = float(bound)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("--attention-cap-bound must be a finite number above 0, not %r" % (bound,))
    return bound


# ------------------------------------------------------------------------------------------------- the scores
def cosine_scores(q: torch.Tensor, k: torch.Tensor, bound: float) -> torch.Tensor:
    """q [..., L, d], k [..., S, d] -> [..., L, S] = bound x cos(q_i, k_j), computed in float32 (float64 for float64
    inputs) with autocast switched off."""
    dtype = torch.promote_types(q.dtype, torch.float32)
    with torch.autocast(device_type=q.device.type, enabled=False):
        qn = F.normalize(q.to(dtype), dim=-1, eps=EPS)
        kn = F.normalize(k.to(dtype), dim=-1, eps=EPS)
        return float(bound) * torch.matmul(qn, kn.transpose(-1, -2))


def head_scores(mha: nn.Module, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """The scores the softmax of attention module `mha` reads, from its in-projected head vectors Q [..., L, dh] and
    K [..., S, dh]: the dot product over the square root of the head size (the expression the readings computed before
    the cap, unchanged) when the module carries no bound; bound x cosine when it does."""
    bound = getattr(mha, "attention_cap_bound", None)
    if bound is None:
        return (Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    return cosine_scores(Q, K, bound).to(torch.promote_types(Q.dtype, K.dtype))


def _masked(scores: torch.Tensor, attn_mask, key_padding_mask, B: int, H: int) -> torch.Tensor:
    """nn.MultiheadAttention's two masks on scores [B, H, L, S]: True (bool) = the key is not read; a float mask is
    added to the scores."""
    L, S = scores.shape[-2], scores.shape[-1]
    if attn_mask is not None:
        m = attn_mask
        if m.dim() == 3:                                        # (B * H, L, S)
            m = m.reshape(B, H, L, S)
        scores = scores.masked_fill(m, float("-inf")) if m.dtype == torch.bool else scores + m.to(scores.dtype)
    if key_padding_mask is not None:
        kp = key_padding_mask.reshape(B, 1, 1, S)
        scores = scores.masked_fill(kp, float("-inf")) if kp.dtype == torch.bool else scores + kp.to(scores.dtype)
    return scores


def capped_forward(mha: "CappedMultiheadAttention", query, key, value, key_padding_mask=None, need_weights=True,
                   attn_mask=None, average_attn_weights=True, is_causal=False, bound: float = None):
    """nn.MultiheadAttention's forward with every head's scores bound x cos(query, key) (see the module text); the same
    inputs and the same returns: (output, weights averaged over the heads [B, L, S] | per head [B, H, L, S] | None)."""
    if is_causal and attn_mask is None:
        raise RuntimeError("Need attn_mask if specifying the is_causal hint (nn.MultiheadAttention's rule)")
    if query.dim() == 2:                                        # an unbatched call: (L, E)
        kp = None if key_padding_mask is None else key_padding_mask.unsqueeze(0)
        out, w = capped_forward(mha, query.unsqueeze(0 if mha.batch_first else 1),
                                key.unsqueeze(0 if mha.batch_first else 1),
                                value.unsqueeze(0 if mha.batch_first else 1), kp, need_weights, attn_mask,
                                average_attn_weights, is_causal, bound)
        out = out.squeeze(0 if mha.batch_first else 1)
        return out, (None if w is None else w.squeeze(0))
    if not mha.batch_first:
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    E, H = mha.embed_dim, mha.num_heads
    dh = E // H
    W, b = mha.in_proj_weight, mha.in_proj_bias
    bq = bk = bv = None
    if b is not None:
        bq, bk, bv = b[:E], b[E:2 * E], b[2 * E:]
    B, L, S = query.shape[0], query.shape[1], key.shape[1]
    q = F.linear(query, W[:E], bq).reshape(B, L, H, dh).transpose(1, 2)            # [B, H, L, dh]
    k = F.linear(key, W[E:2 * E], bk).reshape(B, S, H, dh).transpose(1, 2)         # [B, H, S, dh]
    v = F.linear(value, W[2 * E:], bv).reshape(B, S, H, dh).transpose(1, 2)        # [B, H, S, dh]
    with torch.autocast(device_type=q.device.type, enabled=False):
        s = _masked(cosine_scores(q, k, bound), attn_mask, key_padding_mask, B, H)
        w = torch.softmax(s, dim=-1)                                                # [B, H, L, S]
    a = w
    if mha.training and float(mha.dropout) > 0.0:
        a = F.dropout(a, p=float(mha.dropout))
    o = torch.matmul(a.to(v.dtype), v)                                              # [B, H, L, dh]
    o = o.transpose(1, 2).reshape(B, L, E)
    out = F.linear(o, mha.out_proj.weight, mha.out_proj.bias)
    if not mha.batch_first:
        out = out.transpose(0, 1)
    if not need_weights:
        return out, None
    return out, (w.mean(dim=1) if average_attn_weights else w)


class CappedMultiheadAttention(nn.MultiheadAttention):
    """torch.nn.MultiheadAttention with an optional score bound (attention_cap_bound, a plain attribute, None by
    default). None: forward IS nn.MultiheadAttention.forward, the same call with the same arguments. A float B > 0:
    capped_forward (bound x cosine scores). Adds no parameter and no buffer, so the state_dict is nn.MultiheadAttention's
    and forward hooks registered on the module see every call as before."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_cap_bound: Optional[float] = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        bound = self.attention_cap_bound
        if bound is None:
            return super().forward(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights,
                                   attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                                   is_causal=is_causal)
        return capped_forward(self, query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights,
                              attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal,
                              bound=bound)


# --------------------------------------------------------------------------------------- the covered modules
def _unwrap(module):
    return getattr(module, "_orig_mod", module)


def targets(forward_model, motor_encoder, n_handles: Optional[int] = None) -> List[Tuple[str, nn.Module, int]]:
    """[(name, module, keys per query)] for the covered modules, in a fixed order: the transformer's cross-modal
    attention of every block (keys = the sense slots of one model step), then the motor encoder's attention layers and
    its pool (keys = the body handles, n_handles; None when not given). Refuses a build where one is missing or is not
    a CappedMultiheadAttention that the cap can compute (packed in-projection, no bias_k, no zero attention)."""
    fm = _unwrap(forward_model)
    enc = _unwrap(motor_encoder)
    out = []
    n_slots = int(getattr(fm, "_n_modalities", 0))
    for k, blk in enumerate(fm.blocks):
        if not getattr(blk, "enable_cross_modal", False) or not hasattr(blk, "cm_attn"):
            raise ValueError("--attention-cap on needs the transformer's cross-modal stage; block %d has none" % k)
        out.append(("forward_model.blocks.%d.cm_attn" % k, blk.cm_attn, n_slots))
    if hasattr(enc, "layers") and hasattr(enc, "pool_attn"):
        for i, layer in enumerate(enc.layers):
            out.append(("encoder_registry.encoders.motor.layers.%d.attn" % i, layer["attn"], n_handles))
        out.append(("encoder_registry.encoders.motor.pool_attn", enc.pool_attn, n_handles))
    else:
        raise ValueError("--attention-cap on needs the motor encoder's attention layers and pool "
                         "(lem/model/encoders/cross_patch.py); this build's motor encoder is %s" % type(enc).__name__)
    for name, mod, _n in out:
        if not isinstance(mod, CappedMultiheadAttention):
            raise ValueError("%s is %s, not a CappedMultiheadAttention: the cap cannot be applied to it"
                             % (name, type(mod).__name__))
        if not mod._qkv_same_embed_dim or mod.in_proj_weight is None:
            raise ValueError("%s has separate query, key and value projections; the cap reads the packed one" % name)
        if mod.bias_k is not None or mod.add_zero_attn:
            raise ValueError("%s uses bias_k or add_zero_attn, which the cap does not compute" % name)
    return out


def model_targets(model) -> List[Tuple[str, nn.Module, int]]:
    """targets() of a LemLite (or any object with forward_model, encoder_registry and handle_names)."""
    reg = _unwrap(model.encoder_registry)
    if "motor" not in reg.encoders:
        raise ValueError("--attention-cap on covers the motor encoder; this build has no motor sense")
    return targets(model.forward_model, reg.encoders["motor"], len(getattr(model, "handle_names", []) or []) or None)


def set_bound(modules, bound: Optional[float]) -> None:
    """Give every module of `modules` (an iterable of modules or of (name, module, ...) tuples) the bound (None takes
    the cap away: nn.MultiheadAttention's own forward again)."""
    b = None if bound is None else float(bound)
    for m in modules:
        mod = m[1] if isinstance(m, tuple) else m
        mod.attention_cap_bound = b


def apply(model, bound: Optional[float]) -> List[str]:
    """The switch: set `bound` on every covered module of a LemLite (None = off). Returns the covered names."""
    found = model_targets(model)
    set_bound(found, bound)
    return [name for name, _m, _n in found]


def bounds_in_force(model) -> Dict[str, Optional[float]]:
    """{covered module name: its bound now (None = no cap)}; {} for a build without the covered modules."""
    try:
        found = model_targets(model)
    except ValueError:
        return {}
    return {name: getattr(mod, "attention_cap_bound", None) for name, mod, _n in found}


def stamp(model, bound: float) -> dict:
    """What the receipt records for a build with the cap on: the version, the bound, every covered module with its
    keys per query and the largest weight one key can take at the bound, and the modules left out."""
    found = model_targets(model)
    return {"version": VERSION, "bound": float(bound), "eps": EPS,
            "modules": {name: {"keys": None if n is None else int(n),
                               "largest_weight": None if n is None else round(largest_weight(n, bound), 6)}
                        for name, _m, n in found},
            "not_covered": ["forward_model.blocks.*.ct_attn", "forward_model.blocks.*.cmb_attn",
                            "encoder_registry.encoders.tactile", "prediction_heads"]}


def capped(module) -> bool:
    """True when `module` is an attention module that carries a bound now."""
    return getattr(module, "attention_cap_bound", None) is not None
