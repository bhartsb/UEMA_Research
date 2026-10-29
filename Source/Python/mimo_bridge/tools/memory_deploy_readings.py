# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The memory deploy's readings beyond vision, per body: attention spread, unit liveness, and what the memory's
variation is worth to each sense's prediction.

    python -m mimo_bridge.tools.regression_pack compare --before <checkpoint | step> --after <checkpoint | step>
        (the before/after comparison: the regression pack's own command carries these readings beside its six)
    python -m mimo_bridge.tools.memory_deploy_readings noise --checkpoints C1 C2 ... --cache DIR [--out FILE]
    python -m mimo_bridge.tools.memory_deploy_readings goal --checkpoints C1 C2 ... [--slots 48] --out FILE

WHY (the operator's decision of 2026-09-25 7:47 PM PDT, docs/gain_of_function_v3/WORKLOG.md): when the persistent
memory repair goes live, compare beyond vision - vision variability, proprioception and vestibular sensitivity,
action-conditioned prediction, attention entropy, head and unit liveness, planner candidate separation. Several
improving together marks a central pathology. The regression pack (regression_pack.py) and the memory repair's
watch readings already read most of these; its record's NEXT 3b and NEXT 6
(docs/gain_of_function_v4/agents/persistent_memory_2026-09-25.md on fix/persistent-memory-2026-09-25-v3) name
three that nothing measured: attention entropy per body, unit liveness per body, and the goal line's scale (how
much the memory lowered each sense's next-moment error before it saturated). This module measures them.

WHERE AND HOW (every reading on a saved checkpoint, nothing trained, the graphics card hidden). The regression
pack's model part: the checkpoint's model built on the CPU (gof_probe_harness.load_frozen), its replay ring (the
last 300 s of lived moments of all 8 bodies), SLOTS window ends per body spread over the ring, each window
built exactly as the replay lesson builds it (regression_pack._replay_inputs, fidelity-tested against
LemLite.replay_update). A window counts when its moments belong to one episode. On each window, fp32:
  - the forward model (the memory's bias is added after the final LayerNorm, so the blocks never see it);
  - the heads that read the latent (the forward model's newest tokens with the memory's bias): the prediction
    heads with the gaze copy, the efference body and gaze heads, the tenure pose head, the motor controller (with
    the wellness slope the replay lesson rebuilds from the stored pose and limit), and the memory's own needs
    forecast (it reads the memory's state).

1 ATTENTION SPREAD (entropy): for every attention call, each query row's weights over its keys, recomputed from
  the module's own inputs and projection weights (lem/model/liveness.py attention_weights' arithmetic, every row,
  every batch element); the entropy of the row as a share of the even-spread maximum log(number of keys, masked
  keys included: production's definition), averaged over the attention heads and rows that belong to one window.
  0 = every row reads one input, 1 = every row reads all its inputs evenly. Parts: each block's three stages
  (named by the explainer's schema.ATTENTION_STAGES), each prediction head's attention, the body head's and the
  gaze head's pools. Production's live gauge (the stats record's liveness.pools) samples 64 rows of 2 calls per
  sampled step of the lived step, pooled over bodies; this reads every row of the replay construction per body.
2 UNIT LIVENESS: a unit is one output of a Linear followed by GELU or ReLU inside an nn.Sequential
  (production's liveness naming), read after the activation at the window's newest moment (for a block's
  feed-forward units: the newest moment's six token positions). A unit is LIVE for a body when its output's
  standard deviation over that body's windows, at the position where it varies most, exceeds its NOISE FLOOR:
  the larger of (a) the root mean square rounding production's bf16 arithmetic puts on that unit's own output
  (x - bf16(x), over every window and position) and (b) the largest spread a unit can show while its input stays
  under production's liveness line of -3 (half of |GELU(-3)|: 0.00202; 0 for ReLU). Why (a): production computes
  every head in bf16 (the receipt's precision), so a variation under the rounding at the unit's own size is not
  carried (a unit pinned at 40 moves in steps of 0.25). Why (b): a unit under that line passes almost nothing
  forward or back (lem/model/liveness.py), and its own rounding is tiny, so (a) alone would count its millionths.
  (A first design took the layer's root mean square rounding for (b); one unit pinned at 100,000 then raised the
  floor of every other unit and a layer of 399 moving units read dead: test_memory_deploy_readings.)
  Production's rule - the unit's input rose above -3 on some moment (lem/model/liveness.py LIVE_LINE) - is read
  beside it; it cannot see a unit held at one active value, which is what a constant memory bias does.
  Parts: each forward-model block (its feed-forward units), each prediction head, the body head, the tenure pose
  head, the motor controller, the memory's needs forecast.
3 THE MEMORY'S WORTH (the goal reading): each sense's next-moment error with the memory's bias as each body's
  own stored state gives it, against the error with the memory's VARYING PART taken out - the bias of the mean
  state over every sampled window of every body, which is the constant the repair folds (for the memory's linear
  readout: the readout weight times the state's deviation from its mean, removed) - through the heads and the
  gaze copy. ln(error without / error with): 0 = the variation helps no prediction, above 0 = it lowers the
  error (share removed = 1 - with / without). Beside it the same with only what differs BETWEEN bodies at the
  same moment taken out (the bodies' mean state per moment; NEXT 6's construction). The repair's own goal
  reading (LemLite._memory_goal_predictions on the v3 branch) takes out the same varying part on the delayed
  lesson, live; this is the replay construction on the saved ring, so a saturated memory reads 0 here by
  construction and an early checkpoint gives the scale the goal line lacked.

WITH AND WITHOUT BODY 2 (the operator's decision: body 2, numbered from 0, face down since Sept 19, is excluded
from learning-performance conclusions): every pooled reading is given over all 8 bodies and without body 2, and
the comparison's verdict is read without it. Per-body values are in the JSON.

LINES: each reading's no-change line is the largest change between any two checkpoints with no change of code or
flags between them (the regression pack's rule; `noise` writes NOISE_FILE). A change beyond it in the better
direction (all three families read higher = better) is IMPROVED, in the worse REGRESSED, else within noise.
"""
from __future__ import annotations

import os
if __name__ == "__main__":  # only when run as a program: importing never changes the devices (2026-09-26)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

READINGS_VERSION = "memory_deploy_readings_v2"
REPO = Path(__file__).resolve().parents[4]
NOISE_FILE = REPO / "docs" / "gain_of_function_v3" / "measurements" / "memory_deploy_readings_noise_2026-09-26.json"
EXCLUDED_BODY = 2
# the logits of one chunk of attention recomputation: 2^24 floats, 64 MB
ENTROPY_CHUNK = 2 ** 24
# which input an attention reads most is kept for calls with at most this many keys
KEY_SHARE_MAX = 8
BOOT_DRAWS = 2000
# window ends per body these readings sample: their no-change lines (NOISE_FILE) were measured on 24, so they keep it
# when the regression pack's own model sample grows (regression_pack.MODEL_SLOTS, 1,280 since pack v2)
SLOTS = 24
# THE REPAIR'S DECLARED GOAL LINE (persistent_memory_2026-09-25.md NEXT 3, GOAL, declared 9:30 PM PDT on
# fix/persistent-memory-2026-09-25-v3): at 10,000 updates after the switch the median of ln(error without the
# memory's varying part / error with it) must exceed that sense's report-to-report noise for at least one sense
GOAL_LINE_LN = {"vision": 0.295, "tactile": 0.0062, "audio": 0.027, "proprioception": 0.0070}
# the explainer's names of the forward model's three attention stages, as fix/one-name-per-part-2026-09-25
# (commit 10ebaeb04) defines schema.ATTENTION_STAGES; schema's own dict is used whenever it has one
_STAGES_FALLBACK = {"cm_attn": "stage 1 - senses read each other",
                    "ct_attn": "stage 2 - each sense reads its past",
                    "cmb_attn": "stage 3 - everything reads everything"}
FAMILIES = ("attention", "units", "goal", "scale")
FAMILY_MEANING = {
    "attention": ("attention spread: the entropy of each attention row as a share of the even spread",
                  "0 = every row reads one input; 1 = every row reads all its inputs evenly"),
    "units": ("unit liveness: the share of the part's units whose output varies over the body's moments beyond "
              "its bf16 rounding floor", "0 = no unit responds to the moment; 1 = every unit does"),
    "goal": ("the memory's worth: ln(next-moment error with the memory's varying part taken out / error with it)",
             "0 = the memory's variation helps no prediction; above 0 = it lowers the error"),
    "scale": ("the memory bias's size and what it does to the heads, on the checkpoint's own last 128 recorded "
              "moments (the scale survey's construction)", "memory bias: root mean square per number of each sense "
              "slot's bias (tokens are about 5 per number; lower = better); top slot: the share of a head's attention on "
              "the slot it reads most (1 / 6 = even, 1 = one slot; lower = better); between bodies: how far a head's "
              "output differs between bodies, as a share of how far that sense's own tokens differ (0 = the same "
              "output for every body, a constant head; higher = better)"),
}
# which way each family improves (the scale family's parts carry their own)
BETTER = {"attention": +1, "units": +1, "goal": +1}


def attention_stages() -> Dict[str, str]:
    """schema.ATTENTION_STAGES when the explainer's schema has it, else the same three names."""
    try:
        from mimo_bridge.explainer import schema
    except ImportError:
        return dict(_STAGES_FALLBACK)
    got = getattr(schema, "ATTENTION_STAGES", None)
    return dict(got) if got else dict(_STAGES_FALLBACK)


def part_label(name: str) -> str:
    """The part's name in the words the dashboard and the explainer use for it."""
    m = re.fullmatch(r"forward_model\.blocks\.(\d+)\.(cm_attn|ct_attn|cmb_attn)", name)
    if m:
        return "forward model, block %d, %s" % (int(m.group(1)) + 1, attention_stages()[m.group(2)])
    m = re.fullmatch(r"forward_model\.blocks\.(\d+)", name)
    if m:
        return "forward model, block %d: feed-forward network" % (int(m.group(1)) + 1)
    m = re.fullmatch(r"prediction_heads\.heads\.([a-z_]+)\.attn", name)
    if m:
        return "%s prediction head's attention (which of the forward model's newest tokens it reads)" % m.group(1)
    m = re.fullmatch(r"prediction_heads\.heads\.([a-z_]+)", name)
    if m:
        return "%s prediction head" % m.group(1)
    m = re.fullmatch(r"memory_bias\.([a-z_]+)", name)
    if m:
        return "the memory bias on the %s slot" % m.group(1)
    m = re.fullmatch(r"top_slot\.([a-z_]+)", name)
    if m:
        return "%s prediction head: its attention's share on the slot it reads most" % m.group(1)
    m = re.fullmatch(r"between_bodies\.([a-z_]+)", name)
    if m:
        return ("%s prediction head: how far its output differs between bodies, as a share of how far the bodies' "
                "own %s tokens differ" % (m.group(1), m.group(1)))
    return {"efference.body_pool": "body head's pool (which of the forward model's tokens it reads)",
            "efference.gaze_pool": "gaze head's pool (which of the forward model's tokens it reads)",
            "efference.body_mlp": "body head",
            "tenure_heads.pose": "the tenure pose head",
            "motor_controller": "the motor controller",
            "persistent_memory.outcome": "the memory's needs forecast"}.get(name, name)


# ============================================================================================ what is hooked
UNIT_GROUPS = ((r"forward_model\.blocks\.(\d+)\.ffn\.\d+", lambda m: "forward_model.blocks.%s" % m.group(1)),
               (r"prediction_heads\.heads\.([a-z_]+)\.mlp\.\d+", lambda m: "prediction_heads.heads.%s" % m.group(1)),
               (r"efference\.body_mlp\.\d+", lambda m: "efference.body_mlp"),
               (r"tenure_heads\.pose\.\d+", lambda m: "tenure_heads.pose"),
               (r"motor_controller\.(?:residual_blocks\.\d+|confidence_head|recursive_block|mlp)\.\d+",
                lambda m: "motor_controller"),
               (r"persistent_memory\.outcome\.\d+", lambda m: "persistent_memory.outcome"))
ATTENTION_PARTS = (r"forward_model\.blocks\.\d+\.(?:cm_attn|ct_attn|cmb_attn)", r"prediction_heads\.heads\.[a-z_]+\.attn")
POOL_PARTS = (r"efference\.(?:body_pool|gaze_pool)",)
BLOCK_FFN = r"forward_model\.blocks\.\d+\.ffn\.\d+"


def unit_group(name: str) -> Optional[str]:
    for pattern, group in UNIT_GROUPS:
        m = re.fullmatch(pattern, name)
        if m:
            return group(m)
    return None


def _named_modules(model):
    """(name, module) of every module under the model's attributes, named as lem/model/liveness.py names them
    (first attribute, then the child path); the liveness module itself is left out."""
    import torch.nn as nn
    own = getattr(model, "liveness", None)
    for attr, root in list(vars(model).items()):
        if not isinstance(root, nn.Module) or root is own:
            continue
        for sub, mod in root.named_modules():
            yield attr + ("." + sub if sub else ""), mod


def attention_entropy(mha, args: tuple, kwargs: dict, chunk: int = ENTROPY_CHUNK, with_keys: bool = False):
    """Per batch element of one nn.MultiheadAttention call: the mean over its attention heads and query rows of
    each row's entropy as a share of log(number of keys) - lem/model/liveness.py attention_weights recomputed for
    every element and every row at once, then entropy_share. None for a call this does not reproduce.
    with_keys: also each element's mean weight on each key (over heads and rows) when the call has at most
    KEY_SHARE_MAX keys (a prediction head's attention and stage 1 read the six senses), else None."""
    import torch
    import torch.nn.functional as F
    from lem.model.attention_cap import head_scores
    if mha.bias_k is not None or mha.add_zero_attn:
        return None
    names = ("query", "key", "value", "key_padding_mask", "need_weights", "attn_mask",
             "average_attn_weights", "is_causal")
    got = dict(zip(names, args))
    got.update(kwargs)
    q, k = got["query"], got["key"]
    if not mha.batch_first:
        q, k = q.transpose(0, 1), k.transpose(0, 1)
    E, H = mha.embed_dim, mha.num_heads
    hd = E // H
    if mha._qkv_same_embed_dim:
        Wq, Wk = mha.in_proj_weight[:E], mha.in_proj_weight[E:2 * E]
    else:
        Wq, Wk = mha.q_proj_weight, mha.k_proj_weight
    bq = bk = None
    if mha.in_proj_bias is not None:
        bq, bk = mha.in_proj_bias[:E].float(), mha.in_proj_bias[E:2 * E].float()
    N, Lq, Lk = q.shape[0], q.shape[1], k.shape[1]
    am = got.get("attn_mask")
    if am is None and got.get("is_causal"):
        am = torch.triu(torch.ones(Lq, Lk, dtype=torch.bool, device=q.device), diagonal=1)
    kpm = got.get("key_padding_mask")
    per = max(1, int(chunk // max(1, H * Lq * Lk)))
    out, shares = [], []
    for s in range(0, N, per):
        qs, ks = q[s:s + per].float(), k[s:s + per].float()
        c = qs.shape[0]
        Q = F.linear(qs, Wq.float(), bq).view(c, Lq, H, hd).transpose(1, 2)
        K = F.linear(ks, Wk.float(), bk).view(c, Lk, H, hd).transpose(1, 2)
        # the module's own scores: the dot product over sqrt(hd), or bound x cosine for a module that carries an
        # attention cap (ATTENTION_CAP, lem/model/attention_cap.py head_scores)
        sc = head_scores(mha, Q, K)                                          # [c, H, Lq, Lk]
        if am is not None:
            a = am.view(-1, H, Lq, Lk)[s:s + per] if am.dim() == 3 else am
            sc = sc.masked_fill(a, float("-inf")) if a.dtype == torch.bool else sc + a.float()
        if kpm is not None:
            kb = kpm[s:s + per].view(c, 1, 1, Lk)
            sc = sc.masked_fill(kb, float("-inf")) if kb.dtype == torch.bool else sc + kb.float()
        w = torch.softmax(sc, dim=-1)
        ent = -(w * w.clamp_min(1e-12).log()).sum(dim=-1) / math.log(max(Lk, 2))
        out.append(torch.nan_to_num(ent, nan=0.0).mean(dim=(1, 2)))
        if with_keys and Lk <= KEY_SHARE_MAX:
            shares.append(w.mean(dim=(1, 2)))
    if not with_keys:
        return torch.cat(out)
    return torch.cat(out), (torch.cat(shares) if shares else None)


def pool_entropy(w):
    """A TokenPool's weights [B, heads, 1, n] (lem/model/efference.py): per batch element, the mean over its heads
    of the weights' entropy as a share of log(n) (the efference heads' own _note_spread arithmetic)."""
    p = w.detach().float().clamp_min(1e-12)
    n = p.shape[-1]
    ent = -(w.detach().float() * p.log()).sum(dim=-1) / math.log(max(n, 2))
    return ent.reshape(ent.shape[0], -1).mean(dim=1)


class Capture:
    """Forward hooks on the parts this reads, registered after the model's own (the liveness repair's revived
    units replace a layer's output in a hook registered when the model is built, so these see the repaired
    output). Records only between begin() and end(), and only each part's FIRST call there."""

    def __init__(self, model, n_mod: int):
        import torch.nn as nn
        from lem.model.efference import TokenPool
        self.n_mod = int(n_mod)
        self.on = False
        self.calls: Dict[str, int] = {}
        self.problems: List[str] = []
        self.handles = []
        self.act: Dict[str, nn.Module] = {}
        self.group: Dict[str, str] = {}
        self.units: Dict[str, list] = {}
        self.att: Dict[str, list] = {}
        self._pass_units: Dict[str, object] = {}
        self._pass_att: Dict[str, object] = {}
        seen = set()
        for name, mod in _named_modules(model):
            if isinstance(mod, nn.Sequential):
                kids = list(mod.children())
                for i in range(len(kids) - 1):
                    key = "%s.%d" % (name, i)
                    if (isinstance(kids[i], nn.Linear) and isinstance(kids[i + 1], (nn.GELU, nn.ReLU))
                            and id(kids[i]) not in seen and unit_group(key)):
                        seen.add(id(kids[i]))
                        self.act[key], self.group[key] = kids[i + 1], unit_group(key)
                        self.handles.append(kids[i].register_forward_hook(self._unit_hook(key)))
            if isinstance(mod, nn.MultiheadAttention) and id(mod) not in seen and any(
                    re.fullmatch(p, name) for p in ATTENTION_PARTS):
                seen.add(id(mod))
                self.handles.append(mod.register_forward_hook(self._attention_hook(name), with_kwargs=True))
            if isinstance(mod, TokenPool) and id(mod) not in seen and any(re.fullmatch(p, name) for p in POOL_PARTS):
                seen.add(id(mod))
                self.handles.append(mod.register_forward_hook(self._pool_hook(name)))

    def _unit_hook(self, key):
        def hook(mod, inp, out):
            if self.on:
                self.calls[key] = self.calls.get(key, 0) + 1
                if key not in self._pass_units:
                    self._pass_units[key] = out.detach().float().clone()
            return None
        return hook

    def _attention_hook(self, key):
        def hook(mod, args, kwargs, out):
            if self.on:
                self.calls[key] = self.calls.get(key, 0) + 1
                if key not in self._pass_att:
                    import torch
                    with torch.no_grad():
                        self._pass_att[key] = attention_entropy(mod, args, kwargs, with_keys=True)
            return None
        return hook

    def _pool_hook(self, key):
        def hook(mod, args, out):
            if self.on:
                self.calls[key] = self.calls.get(key, 0) + 1
                if key not in self._pass_att:
                    w = out[1].detach().float()
                    self._pass_att[key] = (pool_entropy(w), w.reshape(w.shape[0], -1, w.shape[-1]).mean(dim=1))
            return None
        return hook

    def begin(self) -> None:
        self._pass_units, self._pass_att = {}, {}
        self.on = True

    def pause(self) -> None:
        self.on = False

    def end(self, R: int, keep, bodies) -> None:
        """Keep the pass's records of the windows kept (keep: [R] bool), per window, each with its body
        (bodies: [R]); a part whose rows do not map onto the windows is named in `problems` and left out."""
        self.on = False
        kb = bodies[keep]
        for key, out in self._pass_units.items():
            if out.shape[0] != R:
                self.problems.append("%s: %d rows for %d windows" % (key, out.shape[0], R))
                continue
            x = out[:, -self.n_mod:, :] if re.fullmatch(BLOCK_FFN, key) else out.reshape(R, -1, out.shape[-1])
            self.units.setdefault(key, []).append((x[keep], kb))
        for key, got in self._pass_att.items():
            e, ks = got if isinstance(got, tuple) else (got, None)
            if e is None or e.shape[0] % R:
                self.problems.append("%s: %s batch elements for %d windows" % (key, None if e is None else e.shape[0], R))
                continue
            ks = None if ks is None else ks.reshape(R, -1, ks.shape[-1]).mean(dim=1)[keep]
            self.att.setdefault(key, []).append((e.reshape(R, -1).mean(dim=1)[keep], kb, ks))
        self._pass_units, self._pass_att = {}, {}

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================================================ the arithmetic
def unit_liveness(pre, act, bodies, n_bodies: int, live_line: float = -3.0) -> dict:
    """pre: [N, P, U] a layer's pre-activations at the windows' newest moment (N windows, P positions, U units);
    act: the layer's activation; bodies: [N] each window's body. Per body: how many units are live (the output's
    standard deviation over the body's windows, at the position where it is largest, above the unit's noise
    floor), how many rose above production's -3 line on some window and position, and the median spread."""
    import torch
    post = act(pre.float())
    r = post - post.to(torch.bfloat16).float()
    r_unit = r.pow(2).mean(dim=(0, 1)).sqrt()                            # [U]
    ceiling = dead_ceiling(act, live_line)
    floor = torch.clamp(r_unit, min=ceiling)
    per = []
    for b in range(int(n_bodies)):
        sel = bodies == b
        n = int(sel.sum())
        if n < 2:
            per.append(dict(windows=n, live=None, live_line=None, spread_median=None))
            continue
        spread = post[sel].std(dim=0).max(dim=0).values                 # [U]
        per.append(dict(windows=n, live=int((spread > floor).sum()),
                        live_line=int((pre[sel] > live_line).flatten(0, 1).any(dim=0).sum()),
                        spread_median=float(spread.median())))
    return dict(units=int(pre.shape[-1]), positions=int(pre.shape[1]), dead_ceiling=ceiling,
                rounding_unit_median=float(r_unit.median()), rounding_unit_max=float(r_unit.max()),
                units_floored_by_rounding=int((r_unit > ceiling).sum()), per_body=per)


def dead_ceiling(act, live_line: float = -3.0) -> float:
    """The largest standard deviation a unit's output can show while its input stays below production's liveness
    line: half the width of the output range below the line - for GELU half of |GELU(-3)| = 0.00202 (outputs lie
    in (GELU(-3), 0]: lem/model/liveness.py), for ReLU 0 (the output is exactly 0 there)."""
    import torch
    import torch.nn as nn
    if isinstance(act, nn.ReLU):
        return 0.0
    if isinstance(act, nn.GELU):
        return abs(float(act(torch.tensor(float(live_line))))) / 2.0
    raise TypeError("unit liveness reads GELU or ReLU units, not %s" % type(act).__name__)


def _rom_rows(ring: dict, backs: Sequence[int], B: int):
    """The development limit each window's target moment lived under (the replay lesson reads rom at the
    window's last moment), rows slot-major; None when the ring does not keep it."""
    import torch
    rom = ring.get("rom")
    if rom is None:
        return None
    size, ptr = int(ring["size"]), int(ring["ptr"])
    vals = []
    for back in backs:
        last = (ptr - 1 - int(back)) % size
        vals += [float(rom[b, last]) for b in range(B)]
    return torch.tensor(vals, dtype=torch.float32)


def recorded_part(model, ckpt) -> dict:
    """THE SCALE SURVEY'S CONSTRUCTION (docs/gain_of_function_v4/measurements/self_percept_runaway/scale_survey.py on
    fix/self-percept-runaway-2026-09-26, which found the memory readout's bias growing from 5.7 to 154 root mean
    square between 344,400 and 415,075), reused: the checkpoint's runtime_state.pt keeps every body's raw encoder
    tokens for the last 128 model steps (the self history: the forward model's whole context) and the persistent
    memory's event history; the forward model is re-run on them for each body's newest moment, with the memory's
    bias rebuilt from its history and without it. Per body: the memory bias's size per sense slot; each prediction
    head's attention over the slots (head-averaged weights) and its output (the residual base removed), with and
    without the memory bias; the sense's own newest token. The check the survey reads: the recorded `internal`
    (the newest output averaged over the slots) against the re-run's. The live forward model reads the tokens after
    the top-down gate; the history keeps the raw tokens (the survey's known difference)."""
    import torch
    rt = torch.load(str(Path(ckpt) / "runtime_state.pt"), map_location="cpu", weights_only=False)
    H, PMH = rt.get("self_history"), rt.get("persistent_memory_history")
    del rt
    gc.collect()
    if not H or not H.get("events"):
        return dict(error="runtime_state.pt keeps no self history")
    fm = getattr(model.forward_model, "_orig_mod", model.forward_model)
    heads = getattr(model.prediction_heads, "_orig_mod", model.prediction_heads)
    pm = getattr(model, "persistent_memory", None)
    mods = list(model.active)
    n, d, ctx = len(mods), int(model.d_model), int(model.ctx_len)
    frames = [e["inputs"]["context"] for e in list(H["events"])[1:]] + [H["context"]]
    frames = [f.float().reshape(f.shape[0], n, d) for f in frames]
    if len(frames) < ctx:
        return dict(error="only %d recorded moments; the context needs %d" % (len(frames), ctx))
    frames = frames[-ctx:]
    B = int(frames[-1].shape[0])
    emb = fm.modality_embedding.weight.float()
    context_buffer = torch.cat([f + emb[None] for f in frames[:-1]], 1)
    tokens = {m: frames[-1][:, i:i + 1] for i, m in enumerate(mods)}
    cur_raw = {m: frames[-1][:, i] for i, m in enumerate(mods)}
    out = dict(construction="the scale survey's: each body's newest recorded moment with the 127 before it as context",
               bodies=B, slots=mods, residual_prediction=bool(getattr(heads, "residual", False)))
    bias = None
    with torch.no_grad():
        if pm is not None and PMH is not None:
            from lem.model.persistent_memory import MemoryHistory
            mh = MemoryHistory(PMH["capacity"], PMH["interval_s"])
            mh.load_state_dict(PMH, "cpu")
            state, _ = pm.encode(**mh.snapshot())
            bias = pm.bias(state).float()
            out["memory_bias_rms"] = bias.square().mean(dim=-1).sqrt().tolist()        # [B][slot]
        _, last, _ = fm(tokens, context_buffer, **({} if bias is None else {"memory_bias": bias}))
        _, last_nb, _ = fm(tokens, context_buffer)
        stored = H.get("internal")
        if stored is not None:
            diff = last.float().mean(1) - stored.float().reshape(B, -1)
            out["internal_rerun_minus_recorded_rms"] = float(diff.square().mean().sqrt())
            out["internal_recorded_rms"] = float(stored.float().square().mean().sqrt())
        seen = {}
        hooks = []
        for m, hd in getattr(heads, "heads", {}).items():
            if hasattr(hd, "attn"):
                def cap(mod, inp, o, m=m):
                    if isinstance(o, tuple) and o[1] is not None:
                        seen[m] = o[1].detach().float().reshape(o[1].shape[0], -1)
                hooks.append(hd.attn.register_forward_hook(cap))
        try:
            p_b = heads(last, current=cur_raw, who="memory_deploy_readings recorded")
            att_b = dict(seen)
            p_nb = heads(last_nb, current=cur_raw, who="memory_deploy_readings recorded")
            att_nb = dict(seen)
        finally:
            for h in hooks:
                h.remove()
    hr = {}
    for m in p_b:
        base = cur_raw[m].reshape(B, -1) if (out["residual_prediction"] and m in cur_raw) else 0.0
        hr[m] = dict(out=(p_b[m].float() - base).tolist(), out_without_memory_bias=(p_nb[m].float() - base).tolist(),
                     token=cur_raw[m].reshape(B, -1).tolist() if m in cur_raw else None,
                     share=att_b[m].tolist() if m in att_b else None,
                     share_without_memory_bias=att_nb[m].tolist() if m in att_nb else None)
    out["heads"] = hr
    return out


def load_model(ckpt):
    """The checkpoint's model on the CPU as gof_probe_harness.load_frozen builds it, except that a sensory encoder the
    load had to start afresh is accepted: these readings never run one (the ring stores every sense's token; only
    the motor token is re-encoded), and H84's touch encoder read 12 numbers per contact row before Sept 23 and 19
    since, so load_frozen refuses every earlier checkpoint. Returns (model, the encoders started afresh)."""
    from mimo_bridge.tools import causal_test as CT
    from mimo_bridge.tools import gof_probe_harness as H
    try:
        fr = H.load_frozen(str(ckpt))
        return fr.model, []
    except H.RefusedCheckpoint as refused:
        if "reinitialised" not in str(refused):
            raise
    info = H.resolve_checkpoint(str(ckpt))
    saved, _, _ = CT._cfg_from_run(info["checkpoint"])
    model, _a, _src, _found = CT._build_model(info["checkpoint"], "cpu", 0,
                                              intention=str(saved.get("intention", "off")) == "on", load=False)
    loaded = model.load_checkpoint(info["checkpoint"], load_optimizer=False)
    fresh = list(loaded.get("reinitialized_components") or [])
    unused = {"encoder:" + m for m in model.active if m != "motor"}
    if set(fresh) - unused:
        raise H.RefusedCheckpoint("%s: the load started afresh %s, which these readings run"
                                  % (info["checkpoint"], sorted(set(fresh) - unused)))
    return model, fresh


def checkpoint_part(model, ring: dict, n_slots: int = None, ckpt=None, newest: Optional[int] = None) -> dict:
    """The families on one checkpoint's ring (see the module note), per window and per body; with `ckpt`, also the
    scale family on its last 128 recorded moments (recorded_part). newest: only windows wholly inside the ring's
    newest `newest` moments (the after side of a switch reads only what was lived after it)."""
    import torch
    from lem.model.efference import apply_gaze_copy
    from mimo_bridge.lem_model import command_for_encoder
    from mimo_bridge.tools import regression_pack as RP
    t0 = time.time()
    n_slots = SLOTS if n_slots is None else int(n_slots)
    L = int(model.ctx_len)
    mods = list(ring["mods"])
    fm = getattr(model.forward_model, "_orig_mod", model.forward_model)
    if list(getattr(fm, "_active_modalities", mods)) != mods:
        raise ValueError("the ring's senses %s are not the forward model's %s" % (mods, list(fm._active_modalities)))
    n_mod = len(mods)
    B = int(ring["tokens"].shape[0])
    heads = model.prediction_heads
    head_mods = list(getattr(heads, "_orig_mod", heads).head_modalities)
    backs = RP.model_slots(ring, L, n_slots, newest)
    per_batch = max(1, RP.MAX_BATCH // B)
    pm = getattr(model, "persistent_memory", None)
    ms = ring.get("memory_states")
    has_memory = pm is not None and ms is not None
    # the memory's mean state over every sampled window of every body: the reference the varying part is taken
    # out against (the constant the repair folds)
    mean_state = None
    if has_memory:
        tot, cnt = None, 0
        for i in range(0, len(backs), per_batch):
            w = RP.gather_windows(ring, backs[i:i + per_batch], L)
            st = w["memory_state"][w["same_episode"]]
            tot = st.sum(dim=0) if tot is None else tot + st.sum(dim=0)
            cnt += int(st.shape[0])
        mean_state = tot / max(1, cnt) if tot is not None else None
        has_memory = mean_state is not None and cnt > 0
    roms = _rom_rows(ring, backs, B)
    cap = Capture(model, n_mod)
    senses = [m for m in head_mods if m in mods]
    S = len(backs)
    shape = (S, B, len(senses))
    sums = {k: np.zeros(shape) for k in ("own", "mean", "between", "unchanged")}
    rows = np.zeros(shape)
    body_w, slot_w = [], []
    mem = dict(varying_sq=0.0, between_sq=0.0, latent_sq=0.0, numbers=0, state_between_sq=0.0,
               state_time_sq=0.0, state_numbers=0)
    try:
        with torch.no_grad():
            for i in range(0, S, per_batch):
                sl = list(range(i, min(S, i + per_batch)))
                w = RP.gather_windows(ring, [backs[j] for j in sl], L)
                R = int(w["tokens"].shape[0])
                order = w["executed"][:, L - 1]
                lab = w["has_executed"][:, L - 1]
                keep = w["same_episode"]
                inp = RP._replay_inputs(model, w, order)
                cur = {m: v.squeeze(1) for m, v in inp["cur"].items()}
                order_enc = command_for_encoder(order.float())
                ach = w["achieved"][:, L - 1]
                cap.begin()
                seq, last0, _ = model.forward_model(inp["cur"], inp["ctx"])   # the blocks never see the bias
                base = last0 if inp["self_bias"] is None else last0 + inp["self_bias"]
                mb = inp["memory_bias"]
                last_own = base if mb is None else base + mb
                preds_own = heads(last_own, current=cur, who="memory_deploy_readings")
                preds_own = apply_gaze_copy(heads, preds_own, last_own, order_enc, ach, mask=lab)
                eff = getattr(model, "efference", None)
                if isinstance(eff, torch.nn.Module):
                    eff.body(last_own, order_enc, ach)
                    eff.gaze(last_own, order_enc, ach)
                th = getattr(model, "tenure_heads", None)
                if th is not None and "pose" in th:
                    th["pose"](last_own.reshape(R, -1))
                mc = getattr(model, "motor_controller", None)
                if mc is not None:
                    latents = torch.cat([seq[:, :-n_mod], last_own], dim=1)
                    wgrad = None
                    if roms is not None and hasattr(getattr(model, "wellness", None), "motor_command_gradient"):
                        wcmd = ach[:, :, :9].detach().clone().float()
                        wcmd[..., 0:3] = wcmd[..., 0:3] - wcmd[..., 0:3].mean(dim=1, keepdim=True)
                        with torch.enable_grad():
                            wcmd.requires_grad_(True)
                            wgrad = model.wellness.motor_command_gradient(
                                wcmd, rom_limit=roms[sl[0] * B:(sl[-1] + 1) * B]).detach()
                    mc(latents, ach, wellness_grad=wgrad)
                if has_memory and hasattr(pm, "predict_needs"):
                    pm.predict_needs(w["memory_state"], order_enc)
                cap.end(R, keep, w["body"])
                body_w += [int(b) for b in w["body"][keep]]
                slot_w += [sl[0] + int(s) for s in w["slot"][keep]]
                # 3 the goal: the memory's varying part taken out, through the heads and the gaze copy
                variants = {"own": preds_own}
                if has_memory:
                    st = w["memory_state"]
                    b_mean = pm.bias(mean_state.to(st).expand_as(st).contiguous())
                    b_betw = pm.bias(st.reshape(R // B, B, -1).mean(dim=1, keepdim=True).expand(R // B, B, -1)
                                     .reshape(R, -1).contiguous())
                    for name, bias in (("mean", b_mean), ("between", b_betw)):
                        lat = base + bias
                        p = heads(lat, current=cur, who="memory_deploy_readings")
                        variants[name] = apply_gaze_copy(heads, p, lat, order_enc, ach, mask=lab)
                    k_ = keep.view(R, 1, 1)
                    mem["varying_sq"] += float(((mb - b_mean).float().square() * k_).sum())
                    mem["between_sq"] += float(((mb - b_betw).float().square() * k_).sum())
                    mem["latent_sq"] += float((base.float().square() * k_).sum())
                    mem["numbers"] += int(keep.sum()) * mb.shape[1] * mb.shape[2]
                    sk = st[keep]
                    grp = st.reshape(R // B, B, -1)
                    dev_b = (grp - grp.mean(dim=1, keepdim=True)).reshape(R, -1)[keep]
                    mem["state_between_sq"] += float(dev_b.square().sum())
                    mem["state_time_sq"] += float((sk - mean_state).square().sum())
                    mem["state_numbers"] += int(sk.numel())
                for si, m in enumerate(senses):
                    k = mods.index(m)
                    ok = keep & lab & w["valid"][:, L, k] & w["valid"][:, L - 1, k]
                    tgt = inp["nxt"][m].float()
                    errs = {name: (p[m].float() - tgt).square().mean(dim=-1) for name, p in variants.items()}
                    errs["unchanged"] = (cur[m].float() - tgt).square().mean(dim=-1)
                    for r in range(R):
                        if not bool(ok[r]):
                            continue
                        s_, b_ = sl[0] + int(w["slot"][r]), int(w["body"][r])
                        rows[s_, b_, si] += 1
                        for name, e in errs.items():
                            sums[name][s_, b_, si] += float(e[r])
                del w, inp, seq, last0, base, last_own, preds_own, variants
    finally:
        cap.detach()
    units = {}
    for key, parts in cap.units.items():
        pre, bodies = torch.cat([p for p, _ in parts]), torch.cat([b for _, b in parts])
        units[key] = dict(group=cap.group[key], **unit_liveness(pre, cap.act[key], bodies, B))
    attention = {key: dict(values=[float(x) for x in torch.cat([p[0] for p in parts])],
                           bodies=[int(x) for x in torch.cat([p[1] for p in parts])],
                           key_share=(None if any(p[2] is None for p in parts)
                                      else torch.cat([p[2] for p in parts]).tolist()),
                           keys=(list(mods) if key.endswith((".attn", "cm_attn", "_pool")) else None))
                 for key, parts in cap.att.items()}
    n = max(1, mem["numbers"])
    ns = max(1, mem["state_numbers"])
    memory = None if not has_memory else dict(
        varying_part_rms=math.sqrt(mem["varying_sq"] / n), between_part_rms=math.sqrt(mem["between_sq"] / n),
        latent_rms=math.sqrt(mem["latent_sq"] / n), state_between_rms=math.sqrt(mem["state_between_sq"] / ns),
        state_over_time_rms=math.sqrt(mem["state_time_sq"] / ns),
        meaning=("root mean square per number: the memory bias's varying part (own minus the mean state's bias) "
                 "and its between-body part, against the latent's own size (the forward model's newest tokens "
                 "after the final LayerNorm); the stored state's spread between bodies at a moment and over "
                 "every sampled moment (states lie in -1..1)"))
    recorded = None
    if ckpt is not None:
        try:
            recorded = recorded_part(model, ckpt)
        except Exception as error:                     # noqa: BLE001 - recorded, the ring's readings stand
            recorded = dict(error="%s: %s" % (type(error).__name__, error))
    return dict(version=READINGS_VERSION, n_slots=int(n_slots), newest=None if newest is None else int(newest),
                slots=[int(x) for x in backs], bodies=B,
                recorded=recorded,
                windows=len(body_w), window_body=body_w, window_slot=slot_w, attention=attention, units=units,
                goal=dict(senses=senses, sums={k: v.tolist() for k, v in sums.items()}, rows=rows.tolist(),
                          memory=memory, has_memory=bool(has_memory)),
                calls=dict(cap.calls), problems=list(cap.problems),
                construction="the replay lesson's (regression_pack._replay_inputs), fp32 on the CPU, no optimizer",
                seconds=round(time.time() - t0, 1))


# ================================================================================================ pooling
def _keep(n: int, exclude: Iterable[int]) -> List[int]:
    ex = set(int(b) for b in exclude)
    return [b for b in range(int(n)) if b not in ex]


def goal_values(part: dict, exclude: Iterable[int] = (), slots: Optional[Sequence[int]] = None) -> dict:
    """Per sense over the bodies kept (and the slots given, default all): ln(error without the varying part /
    error with it), the share of the error the variation removed, the same for the between-body part, and the
    model's error against the unchanged token's."""
    g = part["goal"]
    keep = _keep(part["bodies"], exclude)
    sums = {k: np.asarray(v) for k, v in g["sums"].items()}
    rows = np.asarray(g["rows"])
    sl = slice(None) if slots is None else np.asarray(slots, dtype=int)
    out = {}
    for si, m in enumerate(g["senses"]):
        s = {k: float(v[sl][:, keep, si].sum()) for k, v in sums.items()}
        n = int(rows[sl][:, keep, si].sum())
        own = s["own"]
        row = dict(rows=n, err_own=own / n if n else None)
        if g.get("has_memory") and n and own > 0 and s["mean"] > 0:
            row.update(ln=math.log(s["mean"] / own), share=1.0 - own / s["mean"],
                       ln_between=math.log(s["between"] / own) if s["between"] > 0 else None,
                       share_between=1.0 - own / s["between"] if s["between"] > 0 else None)
        else:
            row.update(ln=None, share=None, ln_between=None, share_between=None)
        row["error_vs_unchanged"] = own / s["unchanged"] if (n and s["unchanged"] > 0) else None
        out[m] = row
    return out


def goal_interval(part: dict, exclude: Iterable[int] = (), draws: int = BOOT_DRAWS, seed: int = 0) -> dict:
    """Per sense, the 5th and 95th percentiles of ln(without / with) over `draws` resamplings of the window slots
    (each slot keeps all its bodies: they lived the same moment)."""
    S = len(part["slots"])
    if S < 2 or not part["goal"].get("has_memory"):
        return {}
    rng = np.random.default_rng(seed)
    vals = {m: [] for m in part["goal"]["senses"]}
    for _ in range(int(draws)):
        pick = rng.integers(0, S, S)
        for m, row in goal_values(part, exclude, pick).items():
            if row["ln"] is not None:
                vals[m].append(row["ln"])
    return {m: ([float(np.percentile(v, 5)), float(np.percentile(v, 95))] if v else None) for m, v in vals.items()}


def pooled(part: dict, exclude: Iterable[int] = ()) -> Dict[str, dict]:
    """Every reading over the bodies kept: {key: dict(family, part, value, [beside])}. Attention: the mean over
    the kept bodies' windows; units: live units over units x kept bodies (each body with 2 or more windows);
    goal: ln(without / with) from the kept bodies' summed errors."""
    keep = set(_keep(part["bodies"], exclude))
    out = {}
    for name, rec in sorted(part["attention"].items()):
        v = np.asarray(rec["values"], dtype=float)[np.isin(np.asarray(rec["bodies"], dtype=int), list(keep))]
        row = dict(family="attention", part=name, value=float(v.mean()) if v.size else None)
        ks = rec.get("key_share")
        if ks is not None and rec.get("keys") and v.size:
            share = np.asarray(ks, dtype=float)[np.isin(np.asarray(rec["bodies"], dtype=int), list(keep))].mean(axis=0)
            top = int(np.argmax(share))
            row["beside"] = dict(reads_most=rec["keys"][top] if top < len(rec["keys"]) else top,
                                 its_weight=float(share[top]))
        out["attention|" + name] = row
    groups: Dict[str, List[int]] = {}
    for key, u in part["units"].items():
        g = groups.setdefault(u["group"], [0, 0, 0])
        for b, pb in enumerate(u["per_body"]):
            if b in keep and pb["live"] is not None:
                g[0] += pb["live"]
                g[1] += pb["live_line"]
                g[2] += u["units"]
    for grp, (live, line, total) in sorted(groups.items()):
        out["units|" + grp] = dict(family="units", part=grp, value=(live / total) if total else None,
                                   beside=dict(live_by_minus3_line=(line / total) if total else None))
    for m, row in goal_values(part, exclude).items():
        out["goal|" + m] = dict(family="goal", part=m, value=row["ln"],
                                beside=dict(share=row["share"], ln_between=row["ln_between"], rows=row["rows"],
                                            error_vs_unchanged=row["error_vs_unchanged"]))
    out.update(scale_values(part.get("recorded"), exclude))
    for row in out.values():
        row.setdefault("better", BETTER.get(row["family"], +1))
    return out


def _between(rows, keep):
    """Root mean square over the kept bodies of each body's vector minus their mean; None for fewer than 2."""
    x = np.asarray([rows[b] for b in keep], dtype=float)
    if x.shape[0] < 2:
        return None
    return float(np.sqrt(np.square(x - x.mean(axis=0, keepdims=True)).mean()))


def scale_values(rec: Optional[dict], exclude: Iterable[int] = ()) -> Dict[str, dict]:
    """The scale family over the bodies kept: each sense slot's memory bias (root mean square per number over the
    kept bodies; lower = better), each prediction head's share of attention on the slot it reads most (the mean over
    kept bodies of each body's largest slot share; lower = better) and how far its output differs between bodies
    as a share of how far that sense's own newest tokens differ (higher = better)."""
    if not rec or rec.get("error"):
        return {}
    keep = _keep(rec["bodies"], exclude)
    out = {}
    for i, slot in enumerate(rec.get("slots") or []):
        if rec.get("memory_bias_rms") is not None:
            v = [rec["memory_bias_rms"][b][i] for b in keep]
            out["scale|memory_bias." + slot] = dict(family="scale", part="memory_bias." + slot, better=-1,
                                                    value=float(np.sqrt(np.mean(np.square(v)))) if v else None)
    for m, h in (rec.get("heads") or {}).items():
        if h.get("share") is not None:
            sh = np.asarray([h["share"][b] for b in keep], dtype=float)
            nb = np.asarray([h["share_without_memory_bias"][b] for b in keep], dtype=float)
            top = int(sh.mean(axis=0).argmax())
            out["scale|top_slot." + m] = dict(
                family="scale", part="top_slot." + m, better=-1, value=float(sh.max(axis=1).mean()),
                beside=dict(reads_most=rec["slots"][top] if top < len(rec["slots"]) else top,
                            its_weight=float(sh.mean(axis=0)[top]), without_memory_bias=float(nb.max(axis=1).mean())))
        if h.get("token") is not None:
            spread, tok = _between(h["out"], keep), _between(h["token"], keep)
            nb = _between(h["out_without_memory_bias"], keep)
            out["scale|between_bodies." + m] = dict(
                family="scale", part="between_bodies." + m, better=+1,
                value=(spread / tok) if (spread is not None and tok) else None,
                beside=dict(output_spread=spread, token_spread=tok,
                            without_memory_bias=(nb / tok) if (nb is not None and tok) else None,
                            output_rms=float(np.sqrt(np.square(np.asarray([h["out"][b] for b in keep])).mean()))))
    return out


def per_body(part: dict) -> List[dict]:
    """Each body's own value of every reading."""
    return [dict(body=b, **{k: v["value"] for k, v in pooled(part, [x for x in range(part["bodies"]) if x != b]).items()})
            for b in range(part["bodies"])]


# ======================================================================================== lines and compare
def load_lines(path: Optional[Path] = None) -> dict:
    p = Path(path) if path else NOISE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("lines") or {}
    except (OSError, ValueError):
        return {}


def lines_from(parts: Sequence[dict]) -> dict:
    """Each reading's largest change between any two of the no-change checkpoints' parts, with and without body 2."""
    lines = {}
    for variant, ex in (("all", ()), ("without_body_2", (EXCLUDED_BODY,))):
        vals = [pooled(p, ex) for p in parts]
        for key in sorted(set().union(*[set(v) for v in vals])) if vals else []:
            xs = [v[key]["value"] for v in vals if key in v and v[key]["value"] is not None]
            d = [abs(a - b) for i, a in enumerate(xs) for b in xs[i + 1:]]
            lines.setdefault(key, {})[variant] = max(d) if d else None
    return lines


def _order(key: str):
    """Readings in the model's order: by family, a block's stages as the block runs them (1, 2, 3), numbers
    as numbers."""
    fam, _, part = key.partition("|")
    stage = {"cm_attn": "1", "ct_attn": "2", "cmb_attn": "3"}
    parts = [stage.get(x, x) for x in part.split(".")]
    return (FAMILIES.index(fam) if fam in FAMILIES else 9, [(0, int(x), "") if x.isdigit() else (1, 0, x) for x in parts])


def verdict(change: Optional[float], line: Optional[float], better: int = +1) -> str:
    """IMPROVED / REGRESSED / within noise: the change against its no-change line, in the reading's better
    direction (+1: higher is better, -1: lower is better)."""
    if change is None:
        return "unmeasured: a reading is missing"
    if line is None:
        return "no verdict: no noise line measured for this reading"
    good = float(better) * change
    if good > float(line):
        return "IMPROVED"
    if good < -float(line):
        return "REGRESSED"
    return "within noise"


def compare_parts(before: Optional[dict], after: Optional[dict], lines: Optional[dict] = None) -> dict:
    """Every reading before and after, with and without body 2, its change, its no-change line and its verdict
    (read without body 2); the readings that moved beyond their lines, by family."""
    lines = load_lines() if lines is None else lines
    if not before or not after:
        return dict(version=READINGS_VERSION, note="a side carries no memory deploy readings", rows={})
    pb = {v: pooled(before, ex) for v, ex in (("all", ()), ("without_body_2", (EXCLUDED_BODY,)))}
    pa = {v: pooled(after, ex) for v, ex in (("all", ()), ("without_body_2", (EXCLUDED_BODY,)))}
    rows = {}
    for key in sorted(set(pb["all"]) | set(pa["all"]), key=_order):
        row = dict(family=(pb["all"].get(key) or pa["all"].get(key))["family"], label=part_label(key.split("|", 1)[1]))
        reads = [((p_.get(key) or {}).get("beside") or {}) for p_ in (pb["without_body_2"], pa["without_body_2"])]
        if any("reads_most" in r for r in reads):
            row["reads_most"] = [(r.get("reads_most"), r.get("its_weight")) for r in reads]
        for variant in ("all", "without_body_2"):
            b = (pb[variant].get(key) or {}).get("value")
            a = (pa[variant].get(key) or {}).get("value")
            ch = None if (a is None or b is None) else a - b
            line = (lines.get(key) or {}).get(variant)
            better = (pb[variant].get(key) or pa[variant].get(key) or {}).get("better", +1)
            row[variant] = dict(before=b, after=a, change=ch, line=line, verdict=verdict(ch, line, better))
        rows[key] = row
    moved = {f: dict(improved=[], regressed=[]) for f in FAMILIES}
    for key, row in rows.items():
        v = row["without_body_2"]["verdict"]
        if v == "IMPROVED":
            moved[row["family"]]["improved"].append(key)
        elif v == "REGRESSED":
            moved[row["family"]]["regressed"].append(key)
    return dict(version=READINGS_VERSION, rows=rows, moved=moved,
                improved=sum(len(m["improved"]) for m in moved.values()),
                regressed=sum(len(m["regressed"]) for m in moved.values()),
                per_body=dict(before=per_body(before), after=per_body(after)),
                memory=dict(before=before["goal"].get("memory"), after=after["goal"].get("memory")),
                problems=dict(before=before.get("problems"), after=after.get("problems")))


def _fmt(family: str, v) -> str:
    if v is None:
        return "unmeasured"
    return ("%+.4f" % v) if family == "goal" else ("%.3f" % v)


def table(rep: dict) -> str:
    """The comparison as one short table per family (the verdict read without body 2)."""
    if not rep.get("rows"):
        return "\nMemory deploy readings: %s\n" % rep.get("note", "none")
    out = ["", "Memory deploy readings beyond vision (%s). Body 2 is numbered from 0 (the dashboard's body 3); the "
               "verdict is read without it. A change counts only beyond the largest change measured between two "
               "no-change checkpoints." % rep["version"],
           "Moved beyond their lines, without body 2: %d improved, %d regressed. The operator's rule: several "
           "improving together marks a central pathology." % (rep["improved"], rep["regressed"])]
    for fam in FAMILIES:
        what, scale = FAMILY_MEANING[fam]
        out += ["", "%s (%s)" % (what, scale), "",
                "| part | before (without body 2 / all 8) | after (without body 2 / all 8) | change without body 2 | "
                "no-change line | verdict |", "|---|---|---|---|---|---|"]
        for key, row in rep["rows"].items():
            if row["family"] != fam:
                continue
            w, a = row["without_body_2"], row["all"]
            label = row["label"]
            if row.get("reads_most"):
                (b_in, b_w), (a_in, a_w) = row["reads_most"]
                label += " - the input it reads most: %s" % " before, ".join(
                    "unmeasured" if x is None else "%s (%.0f%% of its weight)" % (x, 100 * (y or 0))
                    for x, y in ((b_in, b_w), (a_in, a_w))) + " after"
            out.append("| %s | %s / %s | %s / %s | %s | %s | %s |" % (
                label, _fmt(fam, w["before"]), _fmt(fam, a["before"]), _fmt(fam, w["after"]),
                _fmt(fam, a["after"]), _fmt("goal", w["change"]),
                "unmeasured" if w["line"] is None else "%.4f" % w["line"], w["verdict"]))
    return "\n".join(out) + "\n"


# ============================================================================================ the goal job
def goal_report(results: List[dict]) -> str:
    """The measured goal as one table: per checkpoint and sense, the share of the next-moment error the memory's
    variation removed, with its interval, beside the repair's declared goal line."""
    out = ["What the persistent memory's variation was worth to each sense's next-moment prediction (%s). "
           "Construction: the replay lesson's, on each checkpoint's own ring; 'without' = the memory's bias of the "
           "mean state over every sampled window of every body (the varying part taken out). Share removed: "
           "0%% = the variation helps no prediction, 100%% = it removes the whole error. Interval: 5th-95th "
           "percentile over resampled moments. Body 2 left out." % READINGS_VERSION, "",
           "| checkpoint | memory spread between bodies / over time (state, -1..1) | varying part of the bias / the "
           "latent (per number) | sense | share removed (interval of ln) | between-body part only | the repair's "
           "goal line (ln) |", "|---|---|---|---|---|---|---|"]
    for r in results:
        if r.get("error"):
            out.append("| %s | failed: %s | | | | | |" % (r.get("step"), r["error"]))
            continue
        mem = r["memory"] or {}
        for m, row in r["without_body_2"].items():
            iv = (r.get("interval_without_body_2") or {}).get(m)
            out.append("| %s | %s | %s | %s | %s (%s) | %s | %s |" % (
                format(r["step"], ","),
                "%.3f / %.3f" % (mem.get("state_between_rms", float("nan")), mem.get("state_over_time_rms", float("nan"))),
                "%.4f / %.3f" % (mem.get("varying_part_rms", float("nan")), mem.get("latent_rms", float("nan"))),
                m, "unmeasured" if row["share"] is None else "%.2f%%" % (100 * row["share"]),
                "unmeasured" if not iv else "%+.4f to %+.4f" % tuple(iv),
                "unmeasured" if row["share_between"] is None else "%.2f%%" % (100 * row["share_between"]),
                "%.4f" % GOAL_LINE_LN[m] if m in GOAL_LINE_LN else "not graded"))
    return "\n".join(out) + "\n"


def goal(checkpoints: Sequence[str], n_slots: int, draws: int = BOOT_DRAWS) -> List[dict]:
    """The three families on each checkpoint (one at a time, released before the next), the goal summarised."""
    import torch
    from mimo_bridge.tools import gof_probe_harness as H
    from mimo_bridge.tools import regression_pack as RP
    torch.set_num_threads(4)
    results = []
    for ck in checkpoints:
        row = dict(checkpoint=str(ck))
        t0 = time.time()
        try:
            row["step"] = RP.checkpoint_step(ck)
            ring = RP.load_ring(ck)
            row["companion_load_s"] = ring["load_seconds"]
            t1 = time.time()
            model, fresh = load_model(ck)
            row["model_load_s"] = round(time.time() - t1, 1)
            row["encoders_started_afresh_unused_here"] = fresh
            part = checkpoint_part(model, ring, n_slots, ckpt=ck)
            del model, ring
            gc.collect()
            row.update(part=part, memory=part["goal"]["memory"],
                       all=goal_values(part), without_body_2=goal_values(part, [EXCLUDED_BODY]),
                       interval_all=goal_interval(part, (), draws),
                       interval_without_body_2=goal_interval(part, [EXCLUDED_BODY], draws),
                       readings_all=pooled(part), readings_without_body_2=pooled(part, [EXCLUDED_BODY]))
        except Exception as error:                     # noqa: BLE001 - one checkpoint's failure is recorded
            row["error"] = "%s: %s" % (type(error).__name__, error)
        row["seconds"] = round(time.time() - t0, 1)
        row["peak_commit_gb"] = RP._peak_commit_gb()
        results.append(row)
        print(json.dumps({k: row.get(k) for k in ("step", "seconds", "peak_commit_gb", "error")}), flush=True)
    return results


# ================================================================================================ the CLI
def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                   encoding="utf-8")
    tmp.replace(path)


def main(argv=None) -> int:
    from mimo_bridge.tools import regression_pack as RP
    ap = argparse.ArgumentParser(prog="python -m mimo_bridge.tools.memory_deploy_readings",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("noise", help="the no-change lines of every reading")
    n.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint folders or bare steps of the lineage")
    n.add_argument("--cache", required=True, help="the regression pack's checkpoint cache folder to use")
    n.add_argument("--out", default=str(NOISE_FILE))
    n.add_argument("--slots", type=int, default=None)
    n.add_argument("--saved", default=None)
    g = sub.add_parser("goal", help="what the memory's variation was worth, on named checkpoints")
    g.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint folders or bare steps of the lineage")
    g.add_argument("--slots", type=int, default=None)
    g.add_argument("--draws", type=int, default=BOOT_DRAWS)
    g.add_argument("--out", required=True)
    g.add_argument("--saved", default=None)
    a = ap.parse_args(argv)
    saved = RP.saved_root(a.saved)
    slots = SLOTS if a.slots is None else int(a.slots)
    cks = [RP.parse_side(t, RP.WINDOW_STEPS, saved)["checkpoint"] for t in a.checkpoints]
    if a.cmd == "goal":
        res = goal(cks, slots, a.draws)
        out = Path(a.out)
        _write(out, dict(version=READINGS_VERSION, generated=time.strftime("%Y-%m-%d %H:%M:%S"), slots=slots,
                         goal_line_ln=GOAL_LINE_LN, checkpoints=res))
        text = goal_report(res)
        out.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        return 0
    parts, costs = [], []
    for c in cks:
        ck = RP.checkpoint_readings(c, model=True, n_slots=slots, cache_dir=Path(a.cache), internals=True,
                                    internal_slots=slots)
        parts.append(ck["memory_deploy"])
        costs.append(dict(step=ck["step"], checkpoint=c, seconds=ck.get("seconds"), part_seconds=ck["memory_deploy"].get("seconds"),
                          peak_commit_gb=ck.get("peak_commit_gb")))
    lines = lines_from(parts)
    _write(Path(a.out), dict(version=READINGS_VERSION, generated=time.strftime("%Y-%m-%d %H:%M:%S"), slots=slots,
                             checkpoints=costs, lines=lines,
                             readings=[dict(step=c["step"], all=pooled(p), without_body_2=pooled(p, [EXCLUDED_BODY]))
                                       for c, p in zip(costs, parts)],
                             rule=("a reading's line is the largest change between any two no-change checkpoints; "
                                   "a larger change in the better direction is IMPROVED")))
    print(json.dumps({k: v.get("without_body_2") for k, v in lines.items()}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
