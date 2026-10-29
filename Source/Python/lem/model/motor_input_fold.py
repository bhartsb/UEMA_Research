# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The motor controller reads the newest tokens without the persistent memory's constant (--motor-input-fold).

WHAT WAS FOUND (docs/gain_of_function_v4/agents/memory_motor_layer_fall_2026-09-26.md, section 8). The persistent
memory adds its bias to the six newest latent tokens after the transformer's final LayerNorm (lem/model/
forward_model.py). After the memory's switch (--persistent-memory-repair centred) the bias's part that every body and
moment shares is the readout's ordinary bias times the return's factor (PersistentMemory.bias): 153 per number on the
six newest tokens of H84 (106-268 on the touch, body-position, motor and vision slots), against 5.1-5.3 per number for
the tokens' own means. The motor controller reads those tokens laid end to end (MotorController.forward, `pooled`),
through its input projection (the first residual layer's input) and its confidence head's first layer.

A weight that reads a constant input is a bias with a lever. Adam moves each weight by about its learning rate per
optimizer step whatever the size of the input the weight reads, so the 1,152 weights of one unit that read the
constant can move the unit's mean input by up to 3e-4 x (the sum of the constant's sizes) = 3e-4 x 110,400 = 33 per
step, against 0.0003 through the unit's own bias. The saved optimizer states of both arms of the memory trial imply
35-87 (in length over the 128 units) per optimizer step, 0.26-0.62 of the part of the input that differs between
moments. That lever carried 0.93-1.11 of every save-to-save change of the first residual layer's lag in the switch's
length-0 arm and in its memory-off control, and a reversal of that growth 2.3-2.5 times the largest seen would kill half
of the layer's units with the liveness revival off (the restart's setting; the record, 8e).

THE FOLD (this module; "off", the default, builds nothing and imports nothing into the model). From the fold on, the
controller reads the newest tokens minus the memory's effective constant c (PersistentMemory.bias's `const`: the
readout's ordinary bias times the return's factor, read at every call and detached), and at the fold its image moves
into the two biases that read those tokens:
    input projection        x = W_tok (tokens - c) + W_rest [handles; slope] + (b + W_tok c)
    confidence head layer 0 its own units: bias += W0 c; its revived units (lem/model/liveness.py RevivedRows, which
                            read LayerNorm(input - running centre)): the running centre moves by -c.
Every output is unchanged at the fold, to float32 rounding (the record's offline reading: 5.4e-7 of the input's largest
value; the fold refuses above EXACT_LINE and changes nothing). The first residual layer's revived units read the input
projection's output, which the fold leaves unchanged, so their running centre needs nothing. The tokens' own means
(5.1-5.3 per number) remain on the weights, so the lever shrinks from 108,000-113,000 to 4,370-4,660 (23-26 times), and
a later change of the constant (its bias's own Adam steps, or a return) no longer moves the controller's input: the
constant enters the tokens and leaves them again at every call. The constant is the same for every body and moment
(its body-specific part: 2e-5), so the controller loses no information. Under the trainer's bf16 arithmetic the part
that differs between moments survives 0.986-0.990 before the fold and after it; the folded bias (about 807 per number)
is rounded to bf16's step of 4 there, which moves the shared part of the input once by 14-18 in length (0.10-0.14
moment parts; the revived units' running centre absorbs it).

WHAT IS NOT CHANGED. The controller's lessons still send their request on the newest tokens to the memory as before
(c is read detached, as the record's proposal specifies): replay's motor lesson reaches the constant's bias through the
tokens; the motor lesson's re-run and the lived call read detached latents. The fold changes what the controller
reads, not what it asks of the memory. No parameter is added and nothing is drawn from any random stream, so the
optimizer, its moments and every other module are as they were; the fold writes the two biases and the centre in place,
under torch.no_grad, between two steps' graphs.

WHEN THE FOLD HAPPENS (MotorInputFold.at_step_start, called by LemLite.step right after the memory's switch, before
anything in the step reads the memory or runs the controller): at the first TRAINING step at which the memory is
switched and the controller not yet folded. So: at the memory's own switch step (the switch happens at the start of the
first training step after a load whose history holds a lived moment); for a checkpoint whose memory was switched
without the fold, at the first training step after the load that turns the flag on; for a switch inside an empty
history's lived call (memory_integration.observe), at the next step's start. Until then the controller reads the tokens
as production does and the hooks change nothing. An evaluation step (train False) never folds.

A RESUME. The folded flag, the step of the fold and the constant at the fold are buffers of this module, saved as the
side module "motor_input_fold" in model.safetensors (LemLite._side_modules); the folded biases are the controller's own
parameters; the confidence head's revived centre rides runtime_state.pt; the episode companion carries this module's
buffers and plain mirror. LemLite.load_checkpoint calls before_load before any tensor is read: a folded checkpoint is
refused by a build without the flag (it would count the constant twice: in the tokens and in the biases), and a build
with the flag starts not folded and takes the saved state only from the checkpoint's own side module. The fold never
repeats: the flag is written with the fold.

EXACTNESS AND THE RECORD. The fold is planned first (every new value computed, its effect measured on the controller's
last lived inputs, the lag caches LemLite keeps for the motor lesson, or on the constant alone when there are none) and
then written (copies only). Its record goes into the receipt (motor_input_fold) and, when the memory switched in the
same step, into the switch's own record, which the trainer writes into the stats stream (type persistent_memory_switch).
"""
from __future__ import annotations

import json
import logging
import math
import os
import weakref
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

FOLD_VERSION = "motor_input_fold_v1"
MODES = ("off", "on")
# THE FOLD'S EXACTNESS LINE: the largest change the fold may make to a folded layer's pre-activation, in float32, as a
# share of that pre-activation's largest value on the same inputs (the coordinator's line, Sun Sep 27 2026; the record's
# offline reading on the length-0 arm's 7 saves: 4.6e-7 to 5.4e-7). Above it the fold refuses and writes nothing.
EXACT_LINE = 1e-5
# the Adam learning rate the lever reading is quoted at (the trainer's lr; the record's section 8b)
LEVER_LR = 3e-4
CONFIDENCE_ROWS_KEY = "motor_controller__confidence_head__0"
SIDE_NAME = "motor_input_fold"


def effective_constant(pm) -> torch.Tensor:
    """[n_modalities * d_model], float32, detached: the part of the memory bias every body and moment shares, exactly as
    PersistentMemory.bias adds it (the readout's ordinary bias, times the return's factor per slot while a return runs),
    computed as bias() computes it: in float32 outside the trainer's autocast."""
    with torch.no_grad(), torch.autocast(device_type=pm.readout.bias.device.type, enabled=False):
        const = pm.readout.bias
        if pm._return_on:
            const = (const.float().reshape(pm.n_modalities, pm.d_model)
                     * pm.constant_factor().unsqueeze(-1)).reshape(-1)
        return const.detach().float()


def first_layer(mc) -> nn.Linear:
    """The controller's first Linear on [tokens ; handle state ; wellness slope]: the input projection (adaptive and
    recursive controllers) or the plain MLP's first layer."""
    mc = getattr(mc, "_orig_mod", mc)
    lin = getattr(mc, "input_proj", None)
    if lin is None:
        lin = mc.mlp[0]
    if not isinstance(lin, nn.Linear):
        raise ValueError("--motor-input-fold: the motor controller's first layer is not a Linear")
    return lin


class MotorInputFold(nn.Module):
    """The fold's state (buffers, saved as the side module "motor_input_fold") and its two input hooks."""

    def __init__(self, n_tokens: int):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.register_buffer("folded", torch.zeros((), dtype=torch.bool))
        self.register_buffer("fold_step", torch.full((), -1, dtype=torch.int64))
        self.register_buffer("constant_at_fold", torch.zeros(self.n_tokens))
        self.folded_py = False           # a mirror of `folded`: the hooks read no device tensor
        self.record = None               # the fold's record (this process's, or the loaded checkpoint's)
        self.saved_block = None          # the loaded checkpoint's receipt block (before_load), carried into the receipt
        self._model = None               # weakref to the LemLite (callable: the episode companion skips it)
        self._detach = lambda: None      # removes the hooks (callable: the companion skips it)
        self.register_load_state_dict_post_hook(MotorInputFold._after_state_load)

    # ------------------------------------------------------------------------------------------------ the state
    @staticmethod
    def _after_state_load(module, incompatible_keys) -> None:
        module.folded_py = bool(module.folded)
        block = module.saved_block if isinstance(module.saved_block, dict) else {}
        module.record = block.get("fold") if module.folded_py else None
        module.stamp()

    def model(self):
        m = self._model() if self._model is not None else None
        if m is None:
            raise RuntimeError("--motor-input-fold: the model this fold belongs to is gone")
        return m

    def stamp(self) -> dict:
        """The receipt block, written into the model's receipt."""
        out = {"version": FOLD_VERSION, "mode": "on", "folded": bool(self.folded_py),
               "folded_at_step": (int(self.fold_step) if self.folded_py else None),
               "readers": ["motor_controller first layer (token columns)", "motor_controller.confidence_head.0"],
               "exact_line": EXACT_LINE}
        if self.folded_py and self.record is not None:
            out["fold"] = self.record
        m = self._model() if self._model is not None else None
        if m is not None and isinstance(getattr(m, "receipt", None), dict):
            m.receipt["motor_input_fold"] = out
        return out

    @torch.no_grad()
    def reset(self) -> None:
        """Not folded (before a load: the checkpoint's own side module restores a saved fold)."""
        self.folded.fill_(False)
        self.fold_step.fill_(-1)
        self.constant_at_fold.zero_()
        self.folded_py = False
        self.record = None
        self.stamp()

    # ------------------------------------------------------------------------------------------------ the hooks
    def install(self, model) -> None:
        """The two forward pre-hooks: from the fold on, the controller's first layer reads its token columns minus
        the memory's effective constant, and the confidence head reads the tokens minus it. Before the fold they
        return None (the call's own inputs, untouched)."""
        self._model = weakref.ref(model)
        pm = model.persistent_memory
        mc = getattr(model.motor_controller, "_orig_mod", model.motor_controller)
        n = self.n_tokens
        fold = self

        def first(mod, args):
            if not fold.folded_py:
                return None
            x = args[0]
            c = effective_constant(pm).to(x.device)
            dt = torch.promote_types(x.dtype, torch.float32)
            y = torch.cat((x[..., :n].to(dt) - c.to(dt), x[..., n:].to(dt)), dim=-1)
            return (y,) + tuple(args[1:])

        def confidence(mod, args):
            if not fold.folded_py:
                return None
            x = args[0]
            c = effective_constant(pm).to(x.device)
            dt = torch.promote_types(x.dtype, torch.float32)
            return (x.to(dt) - c.to(dt),) + tuple(args[1:])

        handles = [first_layer(mc).register_forward_pre_hook(first),
                   mc.confidence_head.register_forward_pre_hook(confidence)]
        prev = self._detach
        self._detach = lambda: (prev(), [h.remove() for h in handles])

    def detach_hooks(self) -> None:
        self._detach()
        self._detach = lambda: None

    # ------------------------------------------------------------------------------------------------ the fold
    def at_step_start(self, model, train: bool) -> Optional[dict]:
        """LemLite.step, right after the memory's switch: fold when training, not yet folded, and the memory is
        switched. Returns the record when it folded, else None."""
        if not train or self.folded_py:
            return None
        pm = model.persistent_memory
        if pm is None or not pm.repaired:
            return None
        return self.fold(model)

    @torch.no_grad()
    def plan(self, model) -> dict:
        """Every new value and every reading of the fold, nothing written."""
        pm = model.persistent_memory
        mc = getattr(model.motor_controller, "_orig_mod", model.motor_controller)
        lin = first_layer(mc)
        conf = mc.confidence_head[0]
        n = self.n_tokens
        dev = lin.weight.device
        c = effective_constant(pm).to(dev)
        if c.numel() != n or lin.in_features < n or conf.in_features != n:
            raise ValueError("--motor-input-fold: the memory's constant has %d numbers; the controller reads %d token "
                             "numbers (first layer %d inputs, confidence head %d)"
                             % (c.numel(), n, lin.in_features, conf.in_features))
        W = lin.weight.detach()
        W0 = conf.weight.detach()
        image = (W[:, :n].double() @ c.double()).float()
        image0 = (W0.double() @ c.double()).float()
        new_b = lin.bias.detach().float() + image
        new_b0 = conf.bias.detach().float() + image0
        lv = getattr(model, "liveness", None)
        rows = lv.rows[CONFIDENCE_ROWS_KEY] if (lv is not None and getattr(lv, "mode", "off") == "on"
                                                 and CONFIDENCE_ROWS_KEY in lv.rows) else None
        # the revived units read LayerNorm(input - centre): the centre moves with the input, always (a centre that has
        # not learned yet, weight 0, is still read by the call that first sets it, so it moves too)
        new_centre = (rows.centre_mean.detach().float() - c.to(rows.centre_mean.device)) if rows is not None else None
        # the probe: the controller's last lived inputs (the lag caches the motor lesson re-runs), or the constant alone
        lat = getattr(model, "_lag_lat", None)
        hs, wg = getattr(model, "_lag_hs", None), getattr(model, "_lag_wg", None)
        if torch.is_tensor(lat) and lat.dim() == 3 and lat.shape[1] * lat.shape[2] >= n:
            B = lat.shape[0]
            tok = lat[:, -(n // lat.shape[2]):, :].reshape(B, -1).float().to(dev)
            rest_dim = lin.in_features - n
            rest = torch.zeros(B, rest_dim, device=dev)
            parts = [t.reshape(B, -1).float().to(dev) for t in (hs, wg) if torch.is_tensor(t) and t.shape[0] == B]
            got = torch.cat(parts, -1) if parts else rest
            if got.shape[1] == rest_dim:
                rest = got
            probe = "the controller's last lived inputs (the motor lesson's lag caches), %d rows" % B
        else:
            tok = c.unsqueeze(0)
            rest = torch.zeros(1, lin.in_features - n, device=dev)
            probe = "the constant alone (no lived input kept yet), 1 row"
        with torch.autocast(device_type=dev.type, enabled=False):
            x_before = F.linear(torch.cat([tok, rest], -1), W.float(), lin.bias.float())
            x_after = F.linear(torch.cat([tok - c, rest], -1), W.float(), new_b)
            p_before = F.linear(tok, W0.float(), conf.bias.float())
            p_after = F.linear(tok - c, W0.float(), new_b0)
            if rows is not None and rows.any_py:
                p_before = rows.apply(tok, p_before)
                z = rows.lin(rows.norm((tok - c) - new_centre)).float()
                p_after = torch.where(rows.mask, z, p_after)
        dx = float((x_after - x_before).abs().max())
        vx = float(x_before.abs().max())
        dp = float((p_after - p_before).abs().max())
        vp = float(p_before.abs().max())
        mean_before = tok.mean(0)
        mean_after = (tok - c).mean(0)
        rec = dict(version=FOLD_VERSION, probe=probe, line=EXACT_LINE,
                   constant_rms_per_slot=[round(float(v), 4) for v in
                                          c.reshape(-1, pm.d_model).square().mean(-1).sqrt().tolist()],
                   constant_image_length_first_layer=round(float(image.norm()), 3),
                   constant_image_length_confidence=round(float(image0.norm()), 3),
                   first_layer_largest_change=dx, first_layer_largest_value=vx,
                   first_layer_change_share=(dx / vx) if vx > 0 else 0.0,
                   confidence_largest_change=dp, confidence_largest_value=vp,
                   confidence_change_share=(dp / vp) if vp > 0 else 0.0,
                   confidence_revived_units=(int(rows.mask.sum()) if rows is not None else 0),
                   confidence_revived_centre_moved=rows is not None,
                   confidence_revived_centre_weight=(round(float(rows.centre_weight), 6) if rows is not None else None),
                   lever_sum_abs_mean_input_before=round(float(mean_before.abs().sum()), 1),
                   lever_sum_abs_mean_input_after=round(float(mean_after.abs().sum()), 1),
                   lever_largest_step_per_unit_before=round(LEVER_LR * float(mean_before.abs().sum()), 3),
                   lever_largest_step_per_unit_after=round(LEVER_LR * float(mean_after.abs().sum()), 3))
        if torch.is_autocast_enabled(dev.type):
            # the caller's own arithmetic (bf16 on the card), as a reading with no line: the folded bias is rounded to
            # bf16's step there (4 at the image's size of about 800 per number)
            xa16 = F.linear(torch.cat([tok, rest], -1), W, lin.bias).float()
            xb16 = F.linear(torch.cat([tok - c, rest], -1), W, new_b.to(lin.bias.dtype)).float()
            torch.clear_autocast_cache()
            rec["production_arithmetic_mean_shift_length"] = round(float((xb16 - xa16).mean(0).norm()), 3)
            rec["production_arithmetic_largest_change"] = float((xb16 - xa16).abs().max())
        return dict(record=rec, c=c, new_b=new_b, new_b0=new_b0, rows=rows, new_centre=new_centre)

    @torch.no_grad()
    def fold(self, model) -> dict:
        """Plan, refuse above the line, then write (copies only): the two biases, the revived centre, the flag."""
        if self.folded_py:
            raise ValueError("--motor-input-fold: the controller's input is folded already")
        p = self.plan(model)
        rec = p["record"]
        worst = max(rec["first_layer_change_share"], rec["confidence_change_share"])
        if not worst <= EXACT_LINE:
            raise ValueError("--motor-input-fold refused: folding the memory's constant into the controller's biases "
                             "would change a pre-activation by %.3g of its largest value (line %.3g); nothing was "
                             "changed" % (worst, EXACT_LINE))
        mc = getattr(model.motor_controller, "_orig_mod", model.motor_controller)
        lin = first_layer(mc)
        conf = mc.confidence_head[0]
        lin.bias.copy_(p["new_b"].to(lin.bias))
        conf.bias.copy_(p["new_b0"].to(conf.bias))
        if p["new_centre"] is not None:
            p["rows"].centre_mean.copy_(p["new_centre"].to(p["rows"].centre_mean))
        self.constant_at_fold.copy_(p["c"].to(self.constant_at_fold))
        step = int(getattr(model, "step_count", 0))
        self.fold_step.fill_(step)
        self.folded.fill_(True)
        self.folded_py = True
        # a step's autocast keeps bf16 copies of the parameters it cast; the biases changed in place
        torch.clear_autocast_cache()
        rec["folded_at_step"] = step
        self.record = rec
        self.stamp()
        sw = getattr(model, "persistent_memory_switch_unwritten", None)
        if isinstance(sw, dict):
            # the memory switched in this same step: the fold rides the switch's record into the stats stream
            sw["motor_input_fold"] = rec
        log.warning("motor input fold at step %d: the memory's constant (%s per slot) taken out of the controller's "
                    "input, its image (length %.1f) moved into the input projection's bias and the confidence head's "
                    "(%d revived units, centre moved: %s); largest change %.3g and %.3g of the largest value (line "
                    "%.3g) on %s; the lever's largest step per unit %.2f -> %.2f",
                    step, rec["constant_rms_per_slot"], rec["constant_image_length_first_layer"],
                    rec["confidence_revived_units"], rec["confidence_revived_centre_moved"],
                    rec["first_layer_change_share"], rec["confidence_change_share"], EXACT_LINE, rec["probe"],
                    rec["lever_largest_step_per_unit_before"], rec["lever_largest_step_per_unit_after"])
        return rec


# ---------------------------------------------------------------------------------------------------- the model side
def build(model, mode: str) -> Optional[MotorInputFold]:
    """LemLite's constructor, with the flag: the module and its hooks (no parameter, no random draw); None when off."""
    if mode not in MODES:
        raise ValueError("--motor-input-fold must be one of %s, not %r" % (MODES, mode))
    if mode == "off":
        return None
    pm = getattr(model, "persistent_memory", None)
    if pm is None or getattr(pm, "repair", "off") == "off":
        raise ValueError("--motor-input-fold on takes the persistent memory's constant out of the motor controller's "
                         "input; the constant is an ordinary bias only after the repaired memory's switch: it needs "
                         "--persistent-memory on and --persistent-memory-repair centred")
    mc = getattr(model.motor_controller, "_orig_mod", model.motor_controller)
    n = int(mc._d_model) * int(mc._n_active)
    if n != int(pm.n_modalities) * int(pm.d_model):
        raise ValueError("--motor-input-fold: the controller reads %d token numbers, the memory's bias has %d"
                         % (n, int(pm.n_modalities) * int(pm.d_model)))
    fold = MotorInputFold(n).to(pm.readout.bias.device)
    fold.install(model)
    fold.stamp()
    return fold


def _saved_block(path) -> Optional[dict]:
    rp = os.path.join(str(path), "model_receipt.json")
    if not os.path.exists(rp):
        return None
    with open(rp, encoding="utf-8") as f:
        block = json.load(f).get("motor_input_fold")
    return block if isinstance(block, dict) else None


def before_load(model, path) -> None:
    """LemLite.load_checkpoint, before any tensor is read. A folded checkpoint is refused by a build without the fold;
    a build with it starts not folded and takes a saved fold only from the checkpoint's own side module."""
    saved = _saved_block(path)
    fold = getattr(model, "motor_input_fold", None)
    folded = bool(saved and saved.get("folded"))
    if folded and fold is None:
        raise ValueError(
            "checkpoint %s has a folded motor controller (--motor-input-fold on, folded at step %s): its input "
            "projection's and confidence head's biases hold the persistent memory's constant, which it no longer "
            "reads from its input; a build without the fold would count the constant twice. Resume with "
            "--motor-input-fold on" % (path, (saved or {}).get("folded_at_step")))
    if fold is None:
        return
    if saved is not None and saved.get("version") not in (None, FOLD_VERSION):
        raise ValueError("checkpoint %s was folded by %s; this build is %s" % (path, saved.get("version"), FOLD_VERSION))
    if folded:
        wp = os.path.join(str(path), "model.safetensors")
        ok = False
        if os.path.exists(wp):
            from safetensors import safe_open
            with safe_open(wp, framework="pt", device="cpu") as f:
                key = SIDE_NAME + ".folded"
                ok = key in f.keys() and bool(f.get_tensor(key))
        if not ok:
            raise ValueError("checkpoint %s says its motor controller is folded, but its weights hold no folded "
                             "%s side module" % (path, SIDE_NAME))
    fold.reset()
    fold.saved_block = saved
