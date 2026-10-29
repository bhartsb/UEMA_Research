# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The LEM model, lite size, driven by bridge frames (Mode A).

This is the model-side half of the bridge. It builds the SAME model classes the big UE
run uses (EncoderRegistry, ForwardModel, PredictionHeads, MotorController,
WellnessEvaluators, GazeHead, VoiceHead, LossManager, imagination, latent regularizer,
curiosity) from master_config.yaml plus the 8 GB "lite" overrides of lem/live_test_lite.py,
and runs one training step per env step over a BATCH of B MIMo envs.

Model-side facts this relies on (all verified in the plan check):
  * every active modality must be in the frame dict each step (ForwardModel indexes all);
  * audio has no MIMo source -> data_source "synthetic" (placeholder fills it, loss scale 0);
  * intermind is synthetic already; voice is disabled in master_config;
  * placeholders must be filled with batch_size = B (live_test_lite's 32 is a bug);
  * MotorController.forward returns (out, confidence, meta) (3-tuple);
  * lite d_model 128 needs forward_model.n_heads that divides 128 (master has 12).

Nothing in lem/ is modified; this file only composes it.
"""
from __future__ import annotations

import json
import copy
import logging
import math
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from lem.config.loader import load_and_resolve
from lem.config.resolver import resolve, _deep_merge
from lem.model.encoder_registry import EncoderRegistry
from lem.model.forward_model import ForwardModel
from lem.model.prediction_heads import PredictionHeads, GazeHead, VoiceHead
from lem.model.motor_controller import MotorController
from lem.model.wellness import (APPROXIMATE_TERMS, WellnessEvaluators, WellnessMode,
                               MotorStateFormula)
from lem.model.receipt import emit_model_receipt
from lem.placeholders.system import PlaceholderSystem
from lem.training.loss_manager import LossManager
from lem.training.harness import _get_developmental_phase
from lem.training.checkpoint import CheckpointManager
from lem.training.latent_regularizer import LatentRegularizer
from lem.live_test import apply_developmental_constraints, _run_imagination_compiled
from lem.source_mask import to_str as _src_to_str
from lem.model.body_schema import DifferentiableBodySchema
from mimo_bridge.replay import ReplayMemory

# The motor token carries BOTH halves of the transition contract:
#   slots 0..9   achieved handle state (pos, quat, radius, strength, active) - what the
#                body became. This is what the token held before 2026-08-25.
#   slots 10..18 executed command - the order actually in force over the transition,
#                after developmental limits, exploration noise and the motor hold.
# Widened rather than replaced so an old checkpoint migrates bit-identically: the first
# ten columns keep their trained weights and the nine new ones start at zero.
MOTOR_ACHIEVED_F = 10
MOTOR_COMMAND_F = 9
MOTOR_INPUT_F = MOTOR_ACHIEVED_F + MOTOR_COMMAND_F      # 19
# THE UNITS OF THE COMMAND HALF (2026-08-27). The achieved half of a motor row comes off
# the tokenizer wire with positions divided by position_range_cm (200). The executed
# order (motor_for_action) carries positions in raw centimetres, and for every run since
# the ledger existed it went into the same row unscaled: 99.6% of the row's input energy
# was the centimetre block, rotation 0.1%, strength 0.0%. The motor token therefore
# encoded WHERE THE HANDLES ARE, not what they were told to do - two completely
# different orders moved it by 1.6 while zeroing the block moved it by 14.6 - and no
# causal test could ever resolve the action. Every command entering the encoder goes
# through command_for_encoder(); the ledger keeps execution units (cm) for the body.
COMMAND_POSITION_RANGE_CM = 200.0   # == tokenizer.config.TokenizerConfig.position_range_cm
# One unit of the achieved-pose half's position block is COMMAND_POSITION_RANGE_CM
# centimetres, so it is that many hundredths of a metre. The body map's object half needs
# world metres, because that is what the world reports an outside thing's place in.
HANDLE_POS_TO_METRES = COMMAND_POSITION_RANGE_CM / 100.0    # 2.0
# THE TOUCH POINT-CLOUD (2026-08-27). Touch used to reach the encoder as 256 anonymous
# patch slots: measured on H32, shifting one contact by ONE patch moved the token as far
# as teleporting it to a random body site (ratio 1.10), so predicting the NEXT touch
# token was rote memorisation of patch adjacencies - next-state touch error rose 70%
# while touch reconstruction improved 54% and contact statistics stayed flat. A contact
# is now a token of (root-relative 3D position, features): a sliding contact is one
# token whose coordinate changes smoothly, so there is no boundary to cross. This is
# how vision (angular coordinates) and proprioception (continuous bone values) already
# work - the two senses whose prediction improves - and it is what Perception_Space.pdf
# section 1 asserts: "touch dimensions are structured by the body surface geometry".
TOUCH_TOKENS = 32       # max simultaneous contacts kept (measured mean active: 5.7)
TOUCH_TOKEN_F = 19      # [x,y,z, pressure, shear, slip, area, vib x4, self share, self parts,
                        #  push direction x3, depth, arrival speed, valid]
TOUCH_POS_SCALE = 0.5   # metres; the body spans ~0.4 m from the root
# WHERE THE VALIDITY FLAG SITS, by name rather than by number. The row has been widened
# twice (12 -> 14 -> 19) and the flag moves to the new last column each time, so every reader
# that wrote 11 or 13 went on reading a different quantity in silence: three builder tests
# were reading the self-touch share as the validity flag and asserting on it.
TOUCH_VALID_COL = TOUCH_TOKEN_F - 1
# the push's depth and arrival speed, the two columns just before the validity flag
TOUCH_DEPTH_COL = TOUCH_VALID_COL - 2
TOUCH_ARRIVAL_COL = TOUCH_VALID_COL - 1
# HOW MANY CONTROLLED READINGS THE SELF-TOUCH GATE JUDGES, and how stale one may be. The
# control runs every 20 model steps, so 32 readings span about 640 steps. A reading's own
# spread was measured at 0.0169 over the 154 readings of two segments, so 32 of them give a
# standard error of 0.0030 and the gate's line, twice that, is 0.0060 - against an average
# that measured -0.0001. Nothing under a real effect of that size can open it by luck.
SELF_TOUCH_READINGS = 32
SELF_TOUCH_STALE_STEPS = 2000       # 100 control periods: a gate is never held open by a
                                    # measurement that has stopped arriving
from mimo_bridge.movement_drive import MovementDrive
# ONE copy of the eye joint ranges. gaze_reflex owns them (it is the body-side circuit
# that writes eye commands); they were also typed as bare literals at three sites here,
# so the two copies could drift apart with nothing to catch it. gaze_reflex pulls in
# vision_adapter, which imports only math/typing/numpy, so this costs no heavy chain.
from mimo_bridge.gaze_reflex import AZ_RANGE, EL_RANGE, EYE_ROW_CONVENTIONS, eye_pitch_sign
from lem.model.intention_generator import IntentionGenerator, plan_multistep
from lem.model.body_sets import HANDLE_NAMES
from tokenizer.config import MOD_AUDIO, MOD_PROPRIOCEPTION, MOD_MOTOR, MOD_TOUCH, MOD_VOICE, MOD_VISION

log = logging.getLogger("mimo_bridge.lem")


def traceback_short() -> str:
    """The last line of the exception being handled, for a one-line log entry."""
    import traceback
    lines = traceback.format_exc().strip().splitlines()
    return lines[-1] if lines else "no exception"

LEM_DIR = Path(__file__).resolve().parent.parent / "lem"
CONFIG_PATH = str(LEM_DIR / "master_config.yaml")

# lem/live_test_lite.py LITE_OVERRIDES + the fixes the plan check found + MIMo data sources
def command_for_encoder(cmd):
    """An executed order in the units the motor encoder's ACHIEVED half already uses.

    cmd: [..., 9] = pos cm(3), quat xyzw(4), radius, strength - motor_for_action, the
    ledger's executed action, or a planner candidate after the execution transform.
    Returns the same tensor with position / COMMAND_POSITION_RANGE_CM; rotation, radius
    and strength are already dimensionless and pass through. This is the ONLY place a
    command's units change on the way to the encoder; the body and the ledger keep cm.
    """
    from lem.model.encoder_registry import normalize_executed_command
    return normalize_executed_command(cmd, COMMAND_POSITION_RANGE_CM)


def self_touch_rows(token):
    """Which rows of a TOUCH TOKEN are the body touching ITSELF.

    Column 11 is the share of that contact's force that came from the body's own parts and
    TOUCH_VALID_COL, the last column, is the valid flag; a row counts when it is a real
    contact carrying any self-force. The flag moves every time the row is widened, which is
    why it is named rather than numbered here. Returns a boolean mask, or None when this is not a touch token of this
    build's width - the encoder's OUTPUT is d_model wide and its columns mean nothing
    here, which is the mistake this signature exists to prevent.
    """
    if not torch.is_tensor(token) or token.ndim < 2 or token.shape[-1] != TOUCH_TOKEN_F:
        return None
    return (token[..., 11] > 0.0) & (token[..., TOUCH_VALID_COL] > 0.5)


def self_touching_bodies(token):
    """Which BODIES of a [B, rows, TOUCH_TOKEN_F] touch token had their own parts on them."""
    rows = self_touch_rows(token)
    if rows is None or rows.ndim < 2:
        return None
    return rows.any(dim=-1)


def widen_touch_columns(old):
    """Move a narrower touch row, or anything indexed BY touch column, to this build's width.

    ONE function for both, because they have to agree or the encoder reads the wrong
    quantity. The row is widened by putting the old columns where they still belong: the
    validity flag was the old last column and this build's is TOUCH_VALID_COL, so it moves
    there; the column it used to sit in now carries a different quantity and starts at zero;
    every new column starts at zero.

    The same permutation applies to the tactile encoder's first projection, whose columns ARE
    the row's columns. Copying that projection left-aligned instead left the encoder reading
    its trained validity weight on a new column. Measured on the live checkpoints
    (encoder_registry.encoders.tactile.patch_mlp.0.weight): at 12 wide on H84 segment 22 the
    validity column 11 carried |w| mean 0.145; after the 14-wide resume of segment 23 that
    0.145 sat on column 11, which at 14 wide is the self-touch share and is supposed to start
    at exactly zero, while column 13, the validity flag the tokens actually use, started at
    0.000. The same happened again at 19 wide on segment 29.

    old: [..., n] with n < TOUCH_TOKEN_F. Returns a new [..., TOUCH_TOKEN_F] tensor.
    """
    n_old = int(old.shape[-1])
    out = torch.zeros(*old.shape[:-1], TOUCH_TOKEN_F, dtype=old.dtype, device=old.device)
    out[..., :n_old - 1] = old[..., :n_old - 1]
    out[..., TOUCH_VALID_COL] = old[..., n_old - 1]
    return out


def fit_tactile(frame):
    """Make a stored touch token this build's width, without changing what it says.

    The touch token gained two self-touch columns on 2026-09-21 and five push columns the
    same day. A resume brings back replayed moments, the delayed lesson's caches and the
    imagination histories that were stored at the old width, and a narrower tensor cannot
    enter the widened encoder. Pad with zeros - which is exactly "nothing of this was
    recorded for this moment" - carry the validity flag to the column this build reads it
    from, and never touch a tensor that already fits. Returns the same dict object when
    nothing is needed, so the live path pays nothing.
    """
    t = frame.get("tactile") if isinstance(frame, dict) else None
    if not torch.is_tensor(t) or t.shape[-1] == TOUCH_TOKEN_F:
        return frame
    out = dict(frame)
    if t.shape[-1] < TOUCH_TOKEN_F:
        out["tactile"] = widen_touch_columns(t)
    else:
        out["tactile"] = t[..., :TOUCH_TOKEN_F]
    return out


def touch_point_cloud(impulses, touch_pos, touch_self=None, touch_contact=None):
    """(256, >=8) slots + (n_patches, 3) positions -> (TOUCH_TOKENS, TOUCH_TOKEN_F) contacts.

    Rows: active patches only (any nonzero feature), sorted by pressure descending so
    the ordering is a property of the CONTACT and not of the slot index, top
    TOUCH_TOKENS kept, zero-padded, valid flag last. Slot i corresponds to patch row i
    (tokenize_touch fills patches[:n] in adapter order). With touch_pos missing (an old
    pickle, the UE transport) the coordinates are zero and the valid flag still marks
    real rows - degraded to a set of features, never wrong geometry.

    2026-09-14: tokenize_touch now delivers 10 fields (principal stretches added);
    this 12-wide bridge token keeps the first 8 until the bridge encoder is widened -
    the slice below is deliberate, not truncation-by-accident like the one it names.
    """
    out = torch.zeros(TOUCH_TOKENS, TOUCH_TOKEN_F, dtype=impulses.dtype)
    n = impulses.shape[0] if touch_pos is None else min(impulses.shape[0], len(touch_pos))
    act = torch.nonzero(impulses[:n].abs().sum(dim=-1) > 0, as_tuple=False).flatten()
    if act.numel() == 0:
        return out
    order = torch.argsort(impulses[act, 0], descending=True)
    act = act[order][:TOUCH_TOKENS]
    k = act.numel()
    if touch_pos is not None:
        tp = torch.as_tensor(np.asarray(touch_pos), dtype=impulses.dtype)
        out[:k, 0:3] = tp[act] / TOUCH_POS_SCALE
    out[:k, 3:11] = impulses[act][:, :8]
    # SELF-TOUCH (2026-09-21). Columns 11 and 12 say how much of this contact's force came
    # from the body's OWN parts and how many own parts are on it. They are the only thing
    # in the touch sense that separates a hand landing on its own thigh from the floor
    # pressing on it, and they read zero for everything that is not the body. Kept OUT of
    # the 3:11 slice wellness and the movement gain read, so neither changes meaning.
    if touch_self is not None:
        ts = torch.as_tensor(np.asarray(touch_self), dtype=impulses.dtype)
        if ts.ndim == 2 and ts.shape[1] >= 2 and ts.shape[0] > int(act.max()):
            out[:k, 11:13] = ts[act][:, :2]
    # THE PUSH ITSELF (2026-09-21): which way it came from, how deep it is and how fast
    # it arrived. The felt columns say how hard and how much it rubbed; none of them say
    # direction. This is the information the bridge's collision entries carried while
    # nothing read them, delivered on the skin site it happened to.
    if touch_contact is not None:
        tc = torch.as_tensor(np.asarray(touch_contact), dtype=impulses.dtype)
        if tc.ndim == 2 and tc.shape[1] >= 5 and tc.shape[0] > int(act.max()):
            out[:k, 13:18] = tc[act][:, :5]
    out[:k, 18] = 1.0
    return out


# THESE ARE THE DEFAULTS, NOT EVERY RUN'S SIZE. Until 2026-09-17 this dict WAS the model's
# size: 128 wide, 2 core blocks, 2 vision Swin blocks, one attention layer per sense
# encoder. Since then train_mimo.py's size flags (--d-model --core-blocks
# --vision-swin-layers --vision-width --encoder-layers --encoder-width) override it per run
# through size_overrides(), and a flag left at 0 keeps the value written here. H84
# (2026-09-18) runs 192 wide with 4 core blocks and 4 vision Swin blocks. Every comment
# inside this dict describes the DEFAULT build unless it names another run.
LITE_OVERRIDES = {
    "hardware": {"multi_gpu": False, "capture_device": "cuda:0", "training_device": "cuda:0",
                 "torch_compile": {"enabled": False}, "async_backward": False},
    "model": {
        "d_model": 128,
        "context_length": 4,
        "forward_model": {"n_layers": 2, "n_heads": 8},       # 8 divides 128 (master's 12 does not)
        "motor_controller": {"n_layers": 2, "hidden_dim": 128},
        "prediction_heads": {"hidden_expansion": 1},
    },
    # live_test_lite resolves the config twice; the first pass (d_model 768) fills every
    # encoder's d_internal / per_patch_hidden / hidden_dims / conv_channels with 768 and
    # the second pass keeps them, so "lite" encoders stay 768 wide inside (10 M params
    # each, 130 ms per step on this laptop). The bridge makes them 128 wide for real.
    "modalities": {
        # angular_pos_inputs 4: (azimuth, elevation, which eye, patch span). The eye number
        # is what makes stereo representable; the span number separates a coarse peripheral
        # patch from a sharp foveal one pointing the same way. patch_subsample is a cap, not
        # a target - 1160 patches (wide 256 + fovea 288, both eyes) sits well under it.
        # n_swin_layers 2: with one block, two patches are compared only if they fall in the
        # same group of 7 (the windows never shift), or at the final pooling step, which is
        # order-blind. At reaching distance the same object sits about 2.5 patches apart
        # between the eyes in the wide view and about 7 apart in the sharp centre, so a
        # second block re-groups after the first has mixed: 1.5 ms measured.
        # WHAT THAT SECOND BLOCK DOES NOT BUY, AT THIS DICT'S 2 BLOCKS (measured 2026-09-15 by
        # perturbing one patch and reading which outputs move; a run launched with
        # --vision-swin-layers N has N blocks and the paragraph after this one applies):
        # the windows partition the FLAT patch list, so one
        # patch reaches another only within 10 sequence slots. Interleaved eyes put a
        # disparity of d patches about 2d slots apart, so the wide view's 2.5-patch pair
        # (5 slots) does meet inside a window and the fovea's 7-patch pair (14 slots) still
        # never does - the sharp centre remains unable to match its own stereo pair. One
        # image ROW is 32 slots in the wide source and 36 in the fovea, both beyond 10, so a
        # patch never attends to the patch directly above or below it in any block; vertical
        # neighbours meet only at the final pooling query, one weighted sum over all patches,
        # which cannot compute the difference between two named patches. Making the windows
        # 2-D over the (eye, source, row, column) grid would fix both; it changes what the
        # trained vision weights mean, so it needs a recorded checkpoint migration, and a
        # receptive-field test (perturb one patch, assert which outputs change) to pin it.
        # AT 4 BLOCKS (--vision-swin-layers 4, which H84 runs; measured the same way on
        # 2026-09-19, and computed for any depth by explainer/schema.py swin_dependency,
        # which tests/test_build_explainer.py holds to this encoder by that perturbation):
        # one patch reaches another within 17 slots, so the fovea's 14-slot stereo pair DOES
        # meet (634 of 634 pairs), while a 32- or 36-slot image row is still out of reach and
        # vertical neighbours still meet only at the pooling query. One thing both
        # measurements add: the shifted blocks roll the padded list with torch.roll and no
        # mask, so the list closes into a ring and its last 2 patches share a window with its
        # first 3 - the fovea's last corner with the wide view's first.
        "vision": {"encoder": {"n_swin_layers": 2, "patch_subsample": 4096, "d_internal": 128,
                               "angular_pos_hidden": 128, "angular_pos_inputs": 4}},
        "tactile": {"pipeline": {"n_sensors": TOUCH_TOKENS, "features_per_sensor": TOUCH_TOKEN_F},
                    "encoder": {"n_cross_attn_layers": 1, "d_internal": 128, "per_patch_hidden": 128,
                                "per_patch_input_dim": TOUCH_TOKEN_F,
                                # geometry IS the identity now; a per-slot embedding would
                                # reintroduce the anonymous-slot disease this removes
                                "patch_pos_embedding": False}},
        "motor": {"encoder": {"n_cross_attn_layers": 1, "d_internal": 128, "per_patch_hidden": 128,
                              "per_patch_input_dim": MOTOR_INPUT_F}},
        "intermind": {"encoder": {"n_self_attn_layers": 1, "d_internal": 128}},
        "audio": {"data_source": "synthetic", "encoder": {"d_internal": 128, "conv_channels": [128, 128, 128]}},
        "proprioception": {"encoder": {"d_internal": 128, "hidden_dims": [128, 128]}},
    },
    "training": {"batch_size": 1},
}


def core_shape_mismatch(keys, shape_of, *, n_blocks: int, d_model: int) -> Optional[str]:
    """Name the way a checkpoint's core disagrees with the model built to load it, or None.

    A pure function over the checkpoint's tensor names and a shape getter, so the
    loader's guard can be tested without a model or a file. It refuses two things
    the loader used to accept silently:

    BLOCK COUNT. Saved 2 blocks, built 8: strict=False load reported 150 missing
    keys and raised nothing; six blocks kept their random init and the run trained
    a different brain behind one warning line. Built 2 from saved 8: six trained
    blocks discarded, same silence. Measured 2026-09-16 against the real config.

    WIDTH. The encoders' width is checked earlier through the receipt, the core's
    was not: 52 of 54 core tensors would be re-randomised under a log warning.

    A checkpoint with no forward_model tensors at all is not this guard's problem
    and returns None; the loader's own missing-key path handles it.
    """
    keys = list(keys)
    core = [k for k in keys if k.startswith("forward_model.")]
    if not core:
        return None
    saved_blocks = {int(k.split(".")[2]) for k in core
                    if k.startswith("forward_model.blocks.") and k.split(".")[2].isdigit()}
    n_saved = len(saved_blocks)
    if saved_blocks and n_saved != n_blocks:
        if n_saved < n_blocks:
            what = (f"the {n_blocks - n_saved} extra blocks would keep their random "
                    f"initialisation and the run would train a different brain")
        else:
            what = f"{n_saved - n_blocks} trained blocks would be discarded"
        return (f"checkpoint core has {n_saved} core blocks but this model was built with "
                f"{n_blocks}; loading would raise nothing and {what}. Build the model at the "
                f"checkpoint's depth, or grow the checkpoint with an explicit migration "
                f"(zero-initialised residual projections, Adam moments carried by name).")
    width_key = "forward_model.modality_embedding.weight"
    if width_key in keys:
        saved_d = int(shape_of(width_key)[1])
        if saved_d != d_model:
            return (f"checkpoint core is {saved_d} wide but this model was built {d_model} wide; "
                    f"52 of 54 core tensors would be re-randomised behind a log warning. A width "
                    f"change is a fresh start, not a resume.")
    return None


def build_config(extra_overrides: Optional[dict] = None, no_vision: bool = False) -> dict:
    config = load_and_resolve(CONFIG_PATH, skip_validation=True)
    config = _deep_merge(config, LITE_OVERRIDES)
    if extra_overrides:
        config = _deep_merge(config, extra_overrides)
    if no_vision:
        config["modalities"]["vision"]["enabled"] = False
    return resolve(config)


@dataclass
class StepResult:
    # n_handles = len(model.handle_names), set by body_layout.names_for: 24 under
    # 'legacy', 26 under 'trunk_v1'. Written symbolically because the literal went
    # stale: these two lines said 24 and replay_update said 23 while the running body
    # had 26.
    motor_out: np.ndarray            # [B, n_handles, 9] raw controller output (normalized units; the 'gaze' handle is the eyes)
    motor_for_action: np.ndarray     # [B, n_handles, 9] after developmental constraints + noise (cm, quat, radius, strength)
    gaze_out: Optional[np.ndarray]   # [B, 2] radians
    loss: float
    breakdown: Dict[str, float]
    pe_per_modality: Dict[str, float]
    mean_pe: float
    wellness: Dict[str, float]
    total_wellness: float
    motor_confidence: float
    grad_norm: float
    curiosity_drive: float
    source_mask: Dict[str, str]
    timings_ms: Dict[str, float]
    enc_stats: Dict[str, float] = field(default_factory=dict)
    imagination_loss: Optional[float] = None
    wellness_grad_norm: float = 0.0      # mean |d(motor_state score)/d(achieved centroid-relative handle pose)| reaching the controller
    wellness_align_err: Optional[float] = None   # |approximate - grounded|, the imagination trust gauge
    # proof, in the run's own record, that the executed action reached the model:
    # a run where it never did must not look like one where it always did
    exec_seen: int = 0
    nf_aligned: int = 0
    recon_pe: float = float('nan')      # current-frame reconstruction: diagnostic only
    causal_pe: float = float('nan')     # delayed S_t-1 + A_t-1 -> S_t: the real one
    next_pe_per_modality: dict = None
    recon_dec_per_modality: dict = None   # the reconstruction DECODERS' error per real sense
    tenure_pe_per_modality: dict = None   # with-order error at the order's END (250 ms class)
    tenure_gap_per_modality: dict = None  # do-nothing error minus with-order error: the order's worth
    tenure_tok_per_modality: dict = None  # token-reach error per sense (encoder-gradient term)
    # False when no sense was graded 'real' this step: the causal error does not exist,
    # so curiosity_drive is 0.0 and the goal ring got no boost. mean_pe still carries the
    # reconstruction fallback because the curriculum's mastery window reads it.
    curiosity_measured: bool = True
    hand_joint_goal: Optional[np.ndarray] = None
    hand_target_contract: str = 'world_v0'




def eligible_contrast(q, delta, el, sl):
    """Mean over jointly-eligible (i, rolled-j) pairs of
    S(q_i, delta_i) - S(q_i, delta_j) on the slice sl.

    The ONE estimator behind both the real visual agency advantage (delta from
    the SAME event as q - true pairing present) and its measured null (delta
    from a PAST banked event - the time shift breaks any true pairing while
    keeping the estimator's form and data distributions). Factored out and
    unit-tested after the 2026-09-09 review proved the first null (rolling q
    and delta together) was algebraically MINUS the real statistic.
    Returns None when no jointly-eligible pair exists (e.g. B == 1)."""
    import torch.nn.functional as _F
    import torch as _t
    B = q.shape[0]
    vals = []
    a_t = _F.cosine_similarity(q[:, sl], delta[:, sl], dim=-1)
    for k in range(1, B):
        m2 = el & _t.roll(el, k, dims=0)
        if m2.any():
            c = _F.cosine_similarity(q[:, sl], _t.roll(delta, k, dims=0)[:, sl], dim=-1)
            vals.append((a_t - c)[m2])
    return float(_t.cat(vals).mean()) if vals else None

SELF_OBJECTS = ("toy", "mobile")
"""The outside things the body map may take into the self (UEMA 11.2). A thing qualifies when
the world reports WHERE IT IS every moment, because that is what lets the map see it move with
the body's commands: the caregiver's toy and the overhead mobile both do (caregiver.diagnostics
reports object_position and mobile_position in world metres). The caregiver itself is left out -
it moves under its own schedule, and a thing that moves on its own while the body happens to be
moving is exactly the false positive this must not make.

WHAT THE WORLD ALLOWS, per thing, and why the reading has to say so. The toy is a MuJoCo mocap
body (<body name="target" mocap="true"> in the crib scene): MuJoCo writes a mocap body's place
from the outside every step and never integrates it, so the infant's hand cannot displace the
toy at all. Its whole path is the adult's, moving it toward the infant at 0.06 m/s. So a toy
match can only ever be the adult happening to carry it the way a hand was going, never the
infant carrying it. The mobile hangs on a ball joint and does swing when struck. Each thing's
report carries object_physics_driven / mobile_physics_driven saying which it is, and
incorporation_stats passes that through, so no reading is quoted as incorporation when the
world never gave the body a way to earn it."""


def object_state_for(meta: dict):
    """Where each outside thing is this moment, and whether it is out at all, from one body's
    own report. Returns ([n_objects, 3] world metres, [n_objects] 1 where out). A report that
    does not carry a thing reads as away, never as at the origin."""
    cg = (meta or {}).get("caregiver") or {}
    pos, here = [], []
    for name, key, flag in (("toy", "object_position", "object_present"),
                            ("mobile", "mobile_position", "mobile_present")):
        p = cg.get(key)
        ok = bool(cg.get(flag)) and isinstance(p, (list, tuple)) and len(p) == 3
        pos.append([float(v) for v in p] if ok else [0.0, 0.0, 0.0])
        here.append(1.0 if ok else 0.0)
    return pos, here


def object_physics_driven_for(meta: dict) -> dict:
    """Whether the body could move each outside thing at all, from the world's own report.
    A mocap body is written from outside every step and physics never moves it, so no amount
    of reaching can displace it; a thing hanging on a joint does swing when it is struck. The
    toy is the exception once the adult hands it over (--toy-hand-off follow): it is still a
    mocap body, but while the infant grips it the adult makes it go where the infant's hand
    goes, so the body CAN move it, and it is reported that way. A report that does not say
    reads False, which is the careful reading."""
    cg = (meta or {}).get("caregiver") or {}
    return {"toy": bool(cg.get("object_physics_driven", False)
                        or cg.get("object_follows_grip", False)),
            "mobile": bool(cg.get("mobile_physics_driven", False))}


def consequence_kinds(task: dict, world_only: bool = False) -> float:
    """How many kinds of self-relevant consequence this moment carried, 0 to 3, read
    from the world's own per-step task flags (env.py fills them every model step).

    contact by a hand: the mobile, the ball or the box touched by a hand, or a hand
    pressing on something (hand_press_kpa above 2 kPa);
    an object moved by the body: the mobile, the ball or the box touched by ANY body
    part - a kick counts, a caregiver-carried toy brushing a still body does too, which
    is why the developmental tests, not this term, decide whether a behaviour is aimed;
    a view gained: the ball centred in the eye (gaze_on_ball) or the hand in the fovea.
    This is access to a consequence, not a skill label: no test name, no score enters.

    world_only (--consequence-self-contact exclude, DESIGN D9): the hand press counts only
    when the world, not the body's own parts, pressed the hand (hand_pressing_world). The
    old flag fired on SELF-contact - measured under babble on the segment 31 world, 69.6% of
    frames against 11.8% for the world part (env-senses.md) - so the thumb resting in its own
    palm paid 0.5 per moment into the goal value. A report without the split key counts no
    hand press at all under this rule: paying the old flag would pay the self-contact the
    rule exists to exclude."""
    if not task:
        return 0.0
    press = task.get("hand_pressing_world") if world_only else task.get("hand_pressing")
    contact = bool(task.get("mobile_touched_hand") or task.get("ball_touched_hand")
                   or task.get("box_touch_hand") or press)
    moved = bool(task.get("mobile_touched") or task.get("ball_touched") or task.get("box_touch"))
    view = bool(task.get("gaze_on_ball") or task.get("hand_in_fovea"))
    return float(contact) + float(moved) + float(view)


class LemLite:
    """Lite LEM model + one-step train/infer over a batch of bridge frames."""

    # A MEASUREMENT'S PROBE (docs/gain_of_function_v4/proprio_token_measurements.py grade; review finding 3): an
    # object with a `terms` dict and before_backward(model, loss) / after_clip(model, grad_norm). None in every
    # run: each site below that records a term is then a no-op, and no number of the step changes.
    lesson_probe = None

    def _probe_term(self, name: str, term) -> None:
        """Record one term of this step's loss, with its graph, for the probe (nothing without one)."""
        if self.lesson_probe is not None and term is not None:
            self.lesson_probe.terms[name] = term

    def __init__(self, device: str = "cuda:0", overrides: Optional[dict] = None, no_vision: bool = False,
                 wellness_mode: str = "baseline", motor_state_formula: str = "preference",
                 imagination: Optional[bool] = None, latent_reg: Optional[bool] = None,
                 curiosity: Optional[bool] = None, seed: int = 0, lr: Optional[float] = None,
                 checkpoint_dir: Optional[str] = None, think_harder: bool = False,
                 intention: bool = False, intention_every: int = 5, intention_execute: bool = True,
                 action_tenure_loss: bool = False, action_tenure_weight: float = 0.5,
                 action_tenure_alpha: float = 0.04,
                 action_tenure_token_reach: bool = False,
                 action_tenure_token_weight: float = 0.25,
                 action_tenure_token_alpha: float = 0.005,
                 agency_drive: bool = False, agency_weight: float = 0.1,
                 agency_prior: float = 0.05,
                 plan_horizon: str = "fixed",
                 vision_gate: str = "hand",
                 movement_drive: bool = False, movement_sensor_mod: bool = False,
                 movement_param_mod: bool = False, movement_mref: float = 12.0,
                 movement_bonus: float = 0.35, movement_penalty: float = 0.5,
                 step_seconds: float = 0.05,
                 next_frame_loss: bool = False, smooth_actions: float = 0.0,
                 body_schema_updates: bool = False,
                 top_down_gate: bool = False,
                 curiosity_goal: bool = False, anticipation_gaze: bool = False,
                 anticipation_gaze_w: float = 0.1,
                 goal_locality: bool = False,
                 goal_gaze_hold: str = "off",
                 goal_revalue: str = "off",
                 planner_effort_per_body: bool = False,
                 consequence_weight: float = 0.0,
                 refresh_action_calibration: bool = False,
                 prospective_wellness: str = "off",
                 prospective_wellness_w: float = 0.15,
                 hierarchical_self: str = "off",
                 wellness_coherence: str = "extent",
                 # how much a fully self-touching moment may add to wellness once its gate
                 # opens; 0.0 keeps the term at exactly zero (the default until earned)
                 self_touch_wellness: float = 0.0,
                 entity_self: str = "off", entity_self_teacher: str = "wellness",
                 self_value_weight: float = 0.0, self_value_level: str = "l2",
                 self_value_gate_windows: int = 3,
                 hself_slots: int = 64, contact_learning: bool = False,
                 consequence_horizon: int = 5, body_layout: str = 'legacy',
                 contact_profile: str = 'legacy', persistent_memory: str = 'off',
                 self_state: str = 'off', self_choice: str = 'off', self_choice_receipt: str = '',
                 self_prediction: str = 'legacy', self_percept_release: str = 'off',
                 # --proprio-token-repair (mimo_bridge/proprio_token.py, version 6): "off" builds the model the
                 # previous code built; "within_body" adds a lesson that keeps every body's proprioception token
                 # moving with its own posture (its response per unit of posture movement lifted to the line from
                 # the moment it reads under DEAF; its fatigue answer held at the live bodies' level). strength
                 # None = proprio_token.STRENGTH; the line and DEAF are measured (None refuses the lesson while they
                 # are unmeasured); skip = the bodies (numbered from 1) the lesson never pushes
                 proprio_token_repair: str = "off", proprio_token_strength: Optional[float] = None,
                 proprio_token_line: Optional[float] = None, proprio_token_skip=None,
                 proprio_token_deaf: Optional[float] = None,
                 # --proprio-heading (mimo_bridge/proprio_heading.py): "off" = today's world-frame input; "free"
                 # turns every proprioceptive row by minus the body's heading before the encoder reads it
                 proprio_heading: str = "off",
                 # gain-of-function build (docs/gain_of_function_v3/DESIGN.md D5, D13). Both
                 # default OFF so an old launch line builds the model it always built.
                 efference_heads: bool = False, efference_weight: float = 0.1,
                 efference_tenure_steps: int = 5, causal_chain: bool = False,
                 # DESIGN.md D4 (--vestibular-sense, default off): the intermind placeholder's
                 # input slot carries the body's vestibular / internal-state vector instead
                 # (mimo_bridge/vestibular_sense.py). The two settings below act only with it on:
                 # the sense's weight in the next-moment lesson (None = the measured
                 # vestibular_sense.LESSON_WEIGHT) and whether wellness's grounded
                 # prediction-error term counts it ("exclude" keeps the raw sum, DESIGN D8)
                 vestibular_sense: bool = False,
                 vestibular_lesson_weight: Optional[float] = None,
                 vestibular_wellness_pe: str = "exclude",
                 # --vestibular-encoder-repair (vestibular_sense.repair_encoder): "off" builds the model
                 # the previous code built and loads a checkpoint as it did; "rescale" divides the
                 # vestibular encoder's last layer (and its Adam moments) by a power of two at the load
                 # when its bias's spread is above 1,000 (down to 9.2-18.3), so every token stays and
                 # learning can move it again; "rescale_restart" divides the same and restarts the
                 # encoder's Adam state instead (the deploy's mode)
                 vestibular_encoder_repair: str = "off",
                 # DESIGN.md D6-D9 (lem/model/wellness_channels.py). Every default reproduces
                 # the model the previous code built: no channel is measured, the goal value
                 # is the raw sum, curiosity multiplies, the hand press counts self-contact.
                 controllability: bool = False, learning_progress: bool = False,
                 wellness_rank: str = "raw", wellness_weights=None,
                 curiosity_mode: str = "multiply",
                 consequence_self_contact: str = "count",
                 # DESIGN.md D10 (lem/model/contact_head.py): the per-skin-patch contact head and
                 # the contact_control wellness channel. Off and 0.0 build the model the previous
                 # code built. contact_head_weight None = the measured CONTACT_HEAD_WEIGHT.
                 contact_head: bool = False, contact_head_weight: Optional[float] = None,
                 contact_control_wellness: float = 0.0,
                 # DESIGN.md D11 (lem/model/motor_synergy.py): the motor synergy, babble drawn in
                 # its intention space, the planner's residual beside it, and where the smoother
                 # takes its previous order from. Off / None / "proposal" build the model the
                 # previous code built. synergy_tenure_steps is the order's hold (--motor-hold).
                 motor_synergy: bool = False, synergy_dim: Optional[int] = None,
                 synergy_babble: bool = False, synergy_residual_l2: Optional[float] = None,
                 synergy_tenure_steps: int = 5, smooth_source: str = "proposal",
                 synergy_planner: bool = True,
                 # DESIGN.md D12 (lem/model/next_frame_norm.py): each sense's next-moment error
                 # measured against its own running "nothing changes" error. "off" builds the model
                 # the previous code built.
                 next_frame_norm: str = "off", liveness_repair: str = "off",
                 liveness_rule: str = "live_line",
                 physical_goals: str = "off", hand_target_mode: str = "world",
                 hand_goals: str = "off",
                 # MOTOR INTERFACE AUDIT 2026-09-25 (docs/gain_of_function_v4/agents/
                 # motor_interface_audit.md, F1 and F4). Both defaults build the model the previous
                 # code built, so a resume of an old command on this code changes nothing.
                 eye_row_convention: str = "legacy", executed_fields_only: str = "off",
                 # --persistent-memory-repair (lem/model/persistent_memory.py): "off" builds the memory
                 # the previous code built; "centred" switches it, function-preservingly, to a memory
                 # whose state cannot saturate and whose constant is an ordinary bias
                 persistent_memory_repair: str = "off",
                 # --persistent-memory-constant-return: the length in model updates of the folded constant's return
                 # to its pre-deploy size per sense slot after the switch (0: it stays as folded); read at the switch
                 persistent_memory_constant_return: int = 0,
                 # --motor-input-fold (lem/model/motor_input_fold.py): "off" builds nothing; "on" takes the switched
                 # memory's constant out of the motor controller's input, its image folded into the biases
                 motor_input_fold: str = "off",
                 # --motor-asked-change (lem/model/motor_asked_change.py): "off" builds nothing; "on" gives the motor
                 # encoder each handle's asked change (command minus achieved pose) through a path with its own
                 # weights per handle, added to the encoder's output in units of that output's own spread; the
                 # path's output layer starts at zero
                 motor_asked_change: str = "off",
                 # --heads-all-slot-path (lem/model/heads_all_slot_path.py): "off" builds nothing; "on" adds, beside
                 # every prediction head's attention, a path that reads all 6 newest tokens without the memory's
                 # constant, centred and LayerNormed, each slot through its own weights; its output layer starts
                 # at zero
                 heads_all_slot_path: str = "off",
                 # --attention-cap / --attention-cap-bound (ATTENTION_CAP, lem/model/attention_cap.py): "off" leaves
                 # every attention module's scores as they are; "on" scores each key as the bound times the cosine
                 # of query and key in the transformer's cross-modal attention (every block) and the motor encoder's
                 # attention layers and pool (None = the module's MEASURED_BOUND)
                 attention_cap: str = "off", attention_cap_bound: Optional[float] = None,
                 # --vestibular-input-path (lem/model/vestibular_input_path.py): "off" builds nothing; "on" adds one
                 # Linear(28 -> 192) from the vestibular encoder's own input to its output, in units of that output's
                 # own spread, its weight zero at the start; it learns from the live row only
                 vestibular_input_path: str = "off"):
        from .body_layout import names_for, config_overrides
        # PUBLIC names on purpose: the episode companion (continuation_state.capture_learner) carries
        # every _* and last_* attribute across a resume and would put a saved value back over the
        # launch flag; these two come from the launch command only.
        # eye_row_convention (F1): which sign the pitch of the eyes' order row gives the elevation
        # sent to the eye servo; gaze_reflex.EYE_ROW_CONVENTIONS says what each one means.
        if eye_row_convention not in EYE_ROW_CONVENTIONS:
            raise ValueError("eye_row_convention must be one of %s, not %r"
                             % (EYE_ROW_CONVENTIONS, eye_row_convention))
        self.eye_row_convention = str(eye_row_convention)
        # executed_fields_only (F4): the order the ledger records and the planner imagines holds only
        # what the decoder reads - no position for the handles whose position the decoder ignores
        # (their achieved position instead) and no radius (LemLite.executed_fields_only).
        if executed_fields_only not in ("off", "on"):
            raise ValueError("executed_fields_only must be 'off' or 'on', not %r" % (executed_fields_only,))
        self.executed_fields_only_on = executed_fields_only == "on"
        # what the last checkpoint load found changed in these two settings, {name: [saved, now]}, or
        # None (load_checkpoint writes it; public, so never episode state)
        self.motor_interface_change = None
        # what that load did to the saved averages over the order's fields at the change
        # (_convert_order_averages), or None when nothing changed
        self.motor_interface_conversion = None
        self.body_layout = body_layout
        if physical_goals not in ('off', 'memory'):
            raise ValueError('Unknown physical-goal mode')
        if hand_target_mode not in ('world', 'wrist_relative_v1'):
            raise ValueError('Unknown hand-target mode')
        self.physical_goals = physical_goals
        self.hand_target_mode = hand_target_mode
        if hand_goals not in ('off', 'memory'):
            raise ValueError('Unknown hand-goal mode')
        if hand_goals != 'off' and hand_target_mode != 'wrist_relative_v1':
            raise ValueError('Hand goals require wrist-relative execution')
        self.hand_goals = hand_goals
        self.hand_response = None
        self._hand_response_calibration = None
        self._hand_current_angles = None
        self._hand_current_velocities = None
        self._hand_native_seen = False
        self.hand_kinematics = None
        self._hand_geometry = None
        self._hand_previous_joints = None
        # the knees the worlds project calf orders onto (--parent-relative-targets calf or calf_chest;
        # limb_target_control.calf_rows_torch): the planner imagines the calf rows the body receives.
        # Public, so an episode companion never carries it: the trainer sets it at every launch.
        self.calf_projection = None
        if contact_profile not in ('legacy','mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
            raise ValueError('unknown contact profile')
        self.contact_profile = contact_profile
        if contact_learning and contact_profile in ('mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
            raise ValueError('Experimental contact learning has no pressure-history forecast; use contact_learning off with mechanical_v2')
        self.handle_names = names_for(body_layout)
        # THE RUN'S OWN GAZE INDEX, not the class constant. GAZE_IDX below is computed
        # once at class-definition time from the legacy 24-name list; it is 23 under
        # 'trunk_v1' only because names_for appends the trunk names AFTER gaze. A layout
        # that inserts a handle before gaze would read and write a different handle's
        # quaternion with no error. Every instance site reads this attribute; the static
        # gaze_to_handle has no instance and keeps the class constant's documented
        # 24-name assumption.
        if "gaze" not in self.handle_names:
            raise ValueError(f"body layout {body_layout!r} has no 'gaze' handle")
        self.GAZE_IDX = self.handle_names.index("gaze")
        overrides = _deep_merge(dict(overrides or {}), config_overrides(body_layout))
        # THE VESTIBULAR SENSE (DESIGN.md D4). With the flag on, intermind is switched off and a
        # "vestibular" modality is added, which the resolver places in intermind's slot (index 5
        # on every bridge run) with an encoder built like proprioception's at this run's width.
        # With it off nothing is added to the config at all, so the resolved config, the model
        # and every number it produces are the ones the previous code produced.
        self.vestibular_sense = bool(vestibular_sense)
        if self.vestibular_sense:
            from .vestibular_sense import modality_overrides
            overrides = _deep_merge(dict(overrides or {}),
                                    modality_overrides(overrides, LITE_OVERRIDES))
        # think-harder: the motor controller checks its own confidence and runs more
        # thinking blocks when unsure (adaptive mode, UEMA spec; max_depth 4 at lite size).
        # Switching an old fixed-mode snapshot to adaptive keeps everything EXCEPT the
        # motor MLP (fresh blocks) - resume such runs with load_optimizer=False.
        # When the flip is on, the copy (reconstruction) lesson must become a small guard,
        # not a co-equal teacher: the SAME prediction heads cannot both reproduce the
        # current moment and produce the next one. Running both at full weight made the
        # model chase the immediate data - measured 2026-08-21: frozen-weight error 25x
        # worse than live (1.67 vs 0.02) while a pre-flip brain froze cleanly at 0.026.
        if next_frame_loss:
            # Act 2: the prediction heads are future-only. The reconstruction job moves to
            # separate discardable decoder heads (see self.recon_decoders) with a weight
            # that follows a schedule instead of a constant.
            guard = {"modalities": {m: {"loss_weight": 0.0} for m in
                                    ("vision", "tactile", "proprioception", "audio", "intermind")
                                    + (("vestibular",) if self.vestibular_sense else ())}}
            overrides = _deep_merge(dict(overrides or {}), guard)
        if think_harder:
            overrides = _deep_merge(dict(overrides or {}),
                                    {"model": {"motor_controller": {"adaptive": True, "max_depth": 4}}})
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = build_config(overrides, no_vision=no_vision)
        cfg = self.config
        self.d_model = cfg["model"]["d_model"]
        self.active = list(cfg["_active_modalities"])
        if self.vestibular_sense:
            from .vestibular_sense import check_layout
            check_layout(self.active)
        self.n_mod = len(self.active)
        self.ctx_len = cfg["model"]["context_length"]
        self.pred_mods = list(cfg["_prediction_head_modalities"])
        prec = cfg["model"].get("precision", "fp32")
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(prec, torch.float32)
        if self.device.type != "cuda":
            self.dtype = torch.float32
        self.use_amp = self.dtype != torch.float32 and self.device.type == "cuda"

        d = self.device
        self.encoder_registry = EncoderRegistry(cfg).to(d)
        self.forward_model = ForwardModel(cfg).to(d)
        self.prediction_heads = PredictionHeads(cfg).to(d)
        self.motor_controller = MotorController(cfg).to(d)
        self.wellness_mode = WellnessMode(wellness_mode)
        self.wellness = WellnessEvaluators(cfg, wellness_mode=self.wellness_mode,
                                          motor_state_formula=MotorStateFormula(motor_state_formula)).to(d)
        gaze_cfg = cfg["model"].get("gaze_head", {})
        self.gaze_head = None
        if gaze_cfg.get("enabled", False):
            self.gaze_head = GazeHead(d_model=self.d_model, hidden_dim=gaze_cfg["hidden_dim"], n_heads=gaze_cfg["n_heads"],
                                      azimuth_range=gaze_cfg["azimuth_range"], elevation_range=gaze_cfg["elevation_range"]).to(d)
        voice_cfg = cfg["model"].get("voice_head", {})
        self.voice_head = None
        if voice_cfg.get("enabled", False):
            self.voice_head = VoiceHead(d_model=self.d_model, hidden_dim=voice_cfg["hidden_dim"], n_heads=voice_cfg["n_heads"],
                                        output_dim=voice_cfg.get("output_dim", 80)).to(d)
        self.placeholders = PlaceholderSystem(cfg, device=str(d))
        self.loss_manager = LossManager(cfg, wellness_mode=wellness_mode)
        # Act 2: reconstruction decoders - small, separate, discardable (the tactile
        # supplement's design). They certify the embedding still CONTAINS the present;
        # the prediction heads never see this loss, so there is no now-vs-next conflict.
        self.recon_decoders = torch.nn.ModuleDict({
            m: torch.nn.Sequential(torch.nn.Linear(self.d_model, 2 * self.d_model),
                                   torch.nn.GELU(),
                                   torch.nn.Linear(2 * self.d_model, self.d_model)).to(d)
            for m in self.pred_mods}) if next_frame_loss else None
        # Act 3: per-modality top-down gates - g = 0.5*sigmoid(wÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â·[surprise, bias]) blends
        # the forward model's prediction of NOW into perception (bounded at 0.5; learning
        # is always graded on raw encoded reality, never the corrected version)
        self.top_down_gate = bool(top_down_gate)
        self.td_gates = torch.nn.ModuleDict({
            m: torch.nn.Linear(2, 1).to(d) for m in self.pred_mods}) if top_down_gate else None
        # ACTION-TENURE LOSS (2026-08-31, default off). The action's marginal
        # contribution, graded at the order's END rather than 50 ms out. One head per
        # real sense predicts the CHANGE across the whole tenure from the state where
        # the order began; forward() runs once with the order and once with "do
        # nothing" (stop-gradient), and a hinge requires the order branch to win by
        # alpha. Ignoring the command then costs loss directly - the thing H36-H45
        # showed the base objective never charges for.
        if action_tenure_loss and not next_frame_loss:
            raise ValueError(
                "action_tenure_loss requires next_frame_loss: the tenure lesson rides "
                "the next-frame caches (_nf_tokens/_nf_ctx) and is silently absent "
                "without them")
        if agency_drive and not intention:
            raise ValueError(
                "agency_drive requires intention: the drive pays into the goal ring, "
                "which only exists on the intention path")
        if agency_drive and not (action_tenure_loss and action_tenure_token_reach):
            raise ValueError(
                "agency_drive requires the order-to-outcome matching term "
                "(action_tenure_loss + action_tenure_token_reach): its payments ARE "
                "the match outcomes")
        self.action_tenure = bool(action_tenure_loss)
        self.tenure_weight = float(action_tenure_weight)
        self.tenure_alpha = float(action_tenure_alpha)
        # 2026-09-01: the head predicts the tenure's ACHIEVED-POSE change, not token
        # change. Measured on H46's final checkpoint: the order's explainable share of
        # the 250 ms token change is ~0 (-1.1/-2.3/+0.9 pts) while its share of the
        # physical pose change is +4.4 pts - the encoding discards the consequences, so
        # a token-space hinge collects in a currency the tokens do not carry.
        self.tenure_heads = torch.nn.ModuleDict({
            "pose": torch.nn.Sequential(
                torch.nn.Linear(self.n_mod * self.d_model, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, len(self.handle_names) * 3)).to(d)
        }) if self.action_tenure else None
        # TOKEN-REACH (2026-09-01). The encoders discard the order's consequences
        # (token-space share ~0 against +4.4 physical). These heads predict the
        # tenure's TOKEN change from a begin-frame RE-ENCODED LIVE at grading time -
        # the next-frame lesson's trick - so the gradient reaches the encoders and can
        # teach them to retain what the order explains. Plain error, no hinge: nothing
        # is demanded that has not been measured.
        self.tenure_token_reach = bool(action_tenure_token_reach) and self.action_tenure
        self.tenure_token_weight = float(action_tenure_token_weight)
        self.tenure_token_alpha = float(action_tenure_token_alpha)
        # ORDER-TO-OUTCOME MATCHING (2026-09-01, bench-validated). A projection of the
        # order must pick out its own tenure's token change among recent decoys.
        # Bench: discrimination 0.00 -> 1.00 in 300 encoder steps with variance held,
        # where both hinge forms collapsed (H49/H50's failure, reproduced on the bed).
        if self.tenure_token_reach and self.tenure_heads is not None:
            self.tenure_heads["order_proj"] = torch.nn.Sequential(
                torch.nn.Linear(len(self.handle_names) * 9, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 3 * self.d_model)).to(d)
        self._tenure_bank: deque = deque(maxlen=7)   # recent (proj, delta) decoys, detached
        self._tenure_bank_el: deque = deque(maxlen=7)  # the matching hand-visibility masks
        self.last_tenure_tok: Dict[str, float] = {}
        # AGENCY DRIVE (2026-09-02): sensing your own effect pays wellness. Bounded
        # per-body payment from the match outcome; hand-watching bootstrap fades by
        # touch-parity displacement (no clocks); payments enter the goal-ring wellness.
        self.agency_drive = bool(agency_drive) and self.tenure_token_reach
        self.agency_weight = float(agency_weight)
        self.agency_prior_w0 = float(agency_prior)
        # CONSEQUENCE ACCESS in the goal-ring value (2026-09-18, tree revision W2/W1):
        # weight per kind of self-relevant consequence the world measured this moment
        # (contact by a hand, an object moved by the body, a view gained). 0 = off.
        self.consequence_weight = float(consequence_weight)
        self.last_consequence: dict = {}
        self.refresh_action_calibration = bool(refresh_action_calibration)   # P203, resume-time
        self._agency_pay_ema = None          # [B] per-body payment memory
        self._vis_adv_ema = 0.0              # vision slice's agency evidence, live
        # eligible-only since 2026-09-08 (bodies with hand in view, vs eligible
        # impostors); _vis_adv_elig_ema is the eligible FRACTION, split out so the
        # admitted-body count can never masquerade as recognition skill
        self._vis_adv_elig_ema = 0.0
        self._vis_adv_null_ema = 0.0     # mispaired-order control: the gauge's own noise
        self._touch_adv_ema = 0.0            # touch's - the displacement anchor
        self._hand_motion_ema = 1e-3
        # review M16a: measured advantage-noise scale (touch-adv EMA std 0.0042 over
        # H60, 0.0056 over H61's first hour). Below this the anchor has no established
        # evidence to displace against; the ratio floors here instead of at 1e-6.
        self._agency_adv_floor = 0.005
        self._last_agency_vec = None         # [B] this step's payment (tests/diagnosis)
        self.last_agency: Dict[str, float] = {}
        self._tenure_pending = None      # {"tokens", "ctx", "ach", "order"} where the live order began
        self._tenure_prev_order = None   # to detect the moment an order ends (fallback)
        self._tenure_prev_order_id = None    # the trainer-stamped serial, when given
        self.last_tenure_pe: Dict[str, float] = {}
        self.last_tenure_gap: Dict[str, float] = {}
        self._td_err_ema: Optional[torch.Tensor] = None
        # review M23: slow baseline for the gate's persistent-surprise shut-off
        self._td_err_slow: Optional[torch.Tensor] = None
        self.last_gate: Dict[str, float] = {}   # review T10: per-sense mean gate, logged
        # which pairing produced this step's wellness prediction_error, and which of the
        # two quantities named "coherence" produced this step's coherence column
        self.last_pe_source: Optional[str] = None
        self.last_coherence_source: Optional[str] = None
        # whether the causal error existed this step (a sense graded "real"): the
        # curiosity drive is 0.0 and no goal-ring boost is applied when it did not
        self.curiosity_measured = False
        # review M2: measured still-body pose-noise floor for the tenure hinge ratio.
        # 5-step displacement mse of a limp crib body: mean 3.1e-7, p90 4.3e-7 (39
        # windows, CPU probe 2026-09-02); live-babble windows measure 1.6e-5 (37x
        # above). Below this floor a window carries no motion signal and the hinge
        # ratio is noise-over-noise (mw=1e-8/m0=1e-10 adds +99 at d/dmw=1e10).
        self._tenure_m0_floor = 5e-7
        # intention: picture a goal, imagine what candidate commands would do (Adam descent
        # in action space through the frozen forward model), act on the best plan's first
        # step. The goal is the best-feeling recent moment (highest grounded wellness in
        # the ring) - grounded in wellness, no task reward anywhere. No new weights.
        # PLAN-HORIZON LICENSE (2026-09-02). "fixed": max_plan_steps from config
        # (the old hand-set constant). "earned": depth d unlocks while the
        # imagination chain's depth-d prediction BEATS the do-nothing baseline
        # (win-rate EMA > 0.5 - the canonical null, same standard as the tenure
        # hinge); horizon = 1 + contiguous licensed depths. Demonstrated skill,
        # no clock, logged per depth, persisted across resumes.
        if vision_gate not in ("hand", "self-motion"):
            raise ValueError(f"vision_gate must be 'hand' or 'self-motion', got {vision_gate!r}")
        # SELF-MOTION GATE (2026-09-03). "hand": vision's matching slice needs the
        # hand in view at both tenure ends (the H55 protection). "self-motion":
        # ...or the body's OWN mobile moving above the measured ambient floor
        # (2.0 cm/s; spawn drift settles to 1.5, strikes 6-54) - measured H64
        # bottleneck: the hand-only gate opened on 2.75% of events while the
        # struck mobile swung in view all night.
        self.vision_gate_mode = vision_gate
        self._selfmotion_floor_cms = 2.0
        if plan_horizon not in ("fixed", "earned"):
            raise ValueError(f"plan_horizon must be 'fixed' or 'earned', got {plan_horizon!r}")
        self.plan_horizon_mode = plan_horizon
        self._plan_win_ema: Optional[list] = None    # per-depth win-rate EMAs
        self._plan_guard_trips = 0   # P125: rollouts the 5% divergence guard trimmed; process-local, like contract.exec_seen
        self.last_plan_license: Dict[str, object] = {}
        self.think_harder = bool(think_harder)
        self.intention_enabled = bool(intention)
        self.intention_every = max(1, int(intention_every))
        self.intention_execute = bool(intention_execute)
        self.intention_gen = None
        self.goal_ring: deque = deque(maxlen=200)
        self._goal_ring_pose: deque = deque(maxlen=200)
        # parallel to goal_ring: each moment's PRE-adjustment bodily wellness
        # (before movement/agency/curiosity ride on top), so revaluation can
        # replace the bodily part while PRESERVING those drive contributions
        # (09-09 follow-up review: revaluation replaced the whole stored value,
        # silently deleting curiosity and agency from the ranking)
        self._goal_ring_w0: deque = deque(maxlen=200)
        self._goal_ring_cq: deque = deque(maxlen=200)     # kinds of consequence per stored moment [B]
        self._goal_ring_drive: deque = deque(maxlen=200)  # P119: the curiosity factor per stored moment
        # --wellness-rank normalized (DESIGN D8): each stored moment's channel values [C, B], in
        # the order of rank_channel_names, each stored as its distance from the body's running
        # average (lem/model/wellness_channels.detrend_moment; the last rows, pain and harm, raw).
        # Empty and never appended under the default raw ranking.
        self._goal_ring_ch: deque = deque(maxlen=200)
        # the list the ring's rows are in (rides the episode companion WITH the ring, so a resume
        # into a build with another list restarts the ring instead of misreading its rows) and the
        # bodies' running averages the stored distances are taken from
        self._goal_ring_ch_names: Optional[List[str]] = None
        self._rank_detrend: dict = {}
        # the channels THIS build ranks: a PUBLIC name, so the episode companion never carries it
        # over the list the run's own flags set (review of b3, major 2)
        self.rank_channel_names: Optional[List[str]] = None
        # goal persistence (09-09 review completion): hold the PLANNED gaze
        # target for the whole action interval instead of letting the per-step
        # emission replace it; reflex and saccades still interrupt in the env
        self.goal_gaze_hold = goal_gaze_hold == "on"
        self._gaze_hold = None            # (gaze target, expiry step)
        # memory revaluation (09-09 review completion): candidates re-scored
        # at SELECTION time - touch+posture by the current judge on the stored
        # tokens, capacity from the body's PRESENT capability features (the
        # snapshot carries the OLD body; reusing it preserves the staleness)
        self.goal_revalue = goal_revalue == "on"
        self._last_capability = None
        # goal locality + commitment (operator decision 2026-09-06, measured basis in
        # overseer_questions q8: goals were chosen 1.6-1.9x beyond the 14-step
        # lived-motion radius, so no plan could arrive; ring census: wellness spread
        # inside the reach ball is 5.5-6.9x noise, so a best-felt-within-reach rule
        # picks real differences). Constant-free: the reach radius and the arrival
        # line are the model's OWN running measurements, so ambition grows with
        # earned reach and nothing is operator-tuned.
        #   _gl_reach_ema   [B] distance to the ring entry 14 steps back (the world's
        #                   own 14-step displacement, EMA 0.95) = the reach ball
        #   _gl_instant_ema [B] one-step displacement (EMA 0.98) = the arrival line
        #   _gl_held_*      the committed goal per env: keep it while it stays inside
        #                   the ball and unarrived and younger than the ring lifetime
        #                   (200 steps); re-pick only when one of those fails - the
        #                   census showed an uncommitted reachable-best flips every
        #                   event, and a goal that flips cannot anchor a journey.
        self.goal_locality = bool(goal_locality)
        self._gl_reach_ema: Optional[torch.Tensor] = None
        self._gl_instant_ema: Optional[torch.Tensor] = None
        self._gl_held_tok: Optional[Dict[str, torch.Tensor]] = None   # {m: [B,d]}
        self._gl_held_born: Optional[torch.Tensor] = None             # [B] step index
        self._gl_held_mask: Optional[torch.Tensor] = None             # [B] bool
        # the held goal's age in steps when it was picked [B], and where the goal of the latest
        # pick sits now (ring index [B], -1 = its moment has left the ring; age in steps [B])
        self._gl_held_age0: Optional[torch.Tensor] = None
        self._gl_goal_index: Optional[List[int]] = None
        self._gl_goal_age: Optional[List[int]] = None
        self._gl_arrivals = 0
        self._gl_fallbacks = 0
        self.last_intention: Optional[dict] = None
        self.last_scale_report: Optional[dict] = None
        self._last_dev: Optional[dict] = None
        self._last_source_mask: Optional[dict] = None
        # lagged motor grading (review fix 1): the previous step's latents/handle state,
        # so the PREVIOUS command is graded against the CURRENT (post-action) pose.
        # Grading command t against frame t's own pose taught the controller to copy its
        # input - trained-in holding still - because frame t's pose is the result of command t-1.
        self._lag_lat: Optional[torch.Tensor] = None
        self._lag_last: Optional[torch.Tensor] = None
        self._lag_hs: Optional[torch.Tensor] = None
        self._lag_wg: Optional[torch.Tensor] = None
        # movement-hunger drive (US11537850B2 wellness modulation; see movement_drive.py)
        # step_seconds is the RUN's model-step length (train_mimo._transition_seconds =
        # 0.005 * 2 * action_repeat). MovementDrive turns it into the stillness debt's
        # rise (dt/150) and fall (dt/30), so its own 0.05 default means "one model step
        # is 50 ms" - true only at --action-repeat 5. At --action-repeat 2 a step is
        # 0.020 s and full stillness debt is 60 body seconds, at 10 it is 300, while the
        # hardcoded constant reads 150 in every case. The default here keeps every
        # existing caller bit-identical.
        self.step_seconds = float(step_seconds)
        self.movement = (MovementDrive(1, d, dt=self.step_seconds, m_ref_cm_s=movement_mref,
                                       bonus=movement_bonus, penalty=movement_penalty)
                         if movement_drive else None)
        # next-frame loss (the flip): the PREVIOUS moment's tokens are re-run through the
        # forward model and its heads must produce THIS moment's tokens - the future was
        # never in the bag, so no pack/unpack shortcut exists. The current-frame
        # reconstruction stays as a background guard; this becomes the main teacher.
        self.next_frame_loss = bool(next_frame_loss)
        self.body_schema_updates = bool(body_schema_updates)
        # curiosity-goal (2026-09-05 gaze plan B10): the moderate-PE drive reaches the
        # planner's felt-state scoring - multiplicative, (1 + drive). The implemented
        # drive clamps to [0, 1] (measured), so goldilocks moments feel up to TWICE as
        # good AS GOALS and too-easy/too-hard moments keep their plain felt value (no
        # boost, never a penalty). Decision-time only: it scales the goal ring's stored
        # wellness copy, which nothing but goal selection ever reads. No constant to
        # measure.
        self.curiosity_goal = bool(curiosity_goal)
        # anticipation-gaze (B11): imagination aims, reality grades. A supervised term
        # pulls handle 23's gaze toward where REAL self-caused change sat (env-computed
        # direction of the struck mobile or the moving hand, from the recorded frames).
        # Default weight matches the spec's own gaze-head loss weight (UEMA 6.4: 0.1);
        # overridable per attempt (dwell_boost 2026-09-08: at 0.1 the raw loss sat flat
        # at median 0.148 rad^2 = a 22-degree miss vs the 12-degree fovea half-width,
        # outranked by vicreg/tenure/imagination). The launch kill criterion
        # (fovea-on-hand/mobile rates) owns the outcome.
        self.anticipation_gaze = bool(anticipation_gaze)
        self.anticipation_gaze_w = float(anticipation_gaze_w)
        self._nf_tokens: Optional[dict] = None      # detached embeddings (stale/synthetic fill)
        self._nf_raw: Optional[dict] = None         # previous RAW frame for the live re-encode
        self._nf_mask: Optional[dict] = None
        self._nf_ctx: Optional[torch.Tensor] = None
        self._nf_pred: Optional[dict] = None        # prediction of the CURRENT moment (act 3)
        # review M4/M13: cached beside the other _nf_* state (they were created only by
        # reset/first-step, so external readers crashed on a fresh model)
        self._nf_ach: Optional[torch.Tensor] = None      # previous step's achieved pose
        self._nf_handvis: Optional[torch.Tensor] = None  # previous step's hand-in-view flags
        self.last_next_pe: Dict[str, float] = {}
        # THE DEVELOPMENTAL OBSERVER'S MODEL LANE (addendum v4 section 24; train_mimo
        # --observer record; mimo_bridge/observer_model_lane.py). Off by default, and then nothing
        # below is computed. On, this moment's next-moment error per BODY and the contact head's
        # probabilities are kept for the lane's writer to copy out. They are read by the observer's
        # files only, never by any lesson, and their names start with neither "_" nor "last_", so
        # an episode save (continuation_state.capture_learner) does not carry them.
        self.observer_per_body_errors = False
        self.observer_moment: Optional[dict] = None
        self.observer_contact: Optional[dict] = None
        self.last_prediction_audit = {}
        # the same audit per body (prediction_audit.paired_rows): device tensors the trainer sums over a report.
        # A PUBLIC name: the episode companion, which carries every _* and last_* attribute, does not keep it
        self.prediction_audit_rows = {}
        # the swap test graded on the hold's first model step (prediction_audit.audit_blocks "hold_first"; the stats
        # record's prediction_audit_hold_first block), per body on the device; public for the same reason
        self.prediction_audit_hold_first_rows = {}
        # THE COMPARISON RECORDS (mimo_bridge/comparison_records.py, Oct 10, 2026). Public names, so the episode
        # companion does not carry them; plain numbers, dicts, tensors and arrays, so a copied execution
        # (copied_execution.capture) copies them; read by the trainer's record writers only, never by a lesson:
        #   clip_norms       {group: gradient norm before the clip} of the newest clip (_clip_training_gradients)
        #   gaze_copy_now    this model step's vision prediction before and after the gaze copy (record b)
        #   replay_slot_step [ring size] the model step each replay slot holds (enable_replay; record c)
        # The spike trigger (record d) is not an attribute: comparison_records.spike_of(model) finds the one the trainer
        # attached, so a copy of the learner has none.
        self.clip_norms = None
        self.gaze_copy_now = None
        self.replay_slot_step = None
        # THE EXECUTED-COMMAND STREAM'S RUNNING PER-COLUMN MEAN AND MEAN SQUARE (prediction_audit.
        # command_spread_update: {mean, sq: float64 [26 x 9], weight, steps}), read only by the swap test's shifted-own
        # arm for each column's standard deviation. Not a parameter, never trained, no gradient; it starts with "_" so
        # the episode companion carries it across a resume (plain tensors and numbers, which the companion can save)
        self._command_spread = None
        self.last_recon_dec: Dict[str, float] = {}   # decoders' reconstruction error per real sense
        # EFFERENCE-HEAD WORKING STATE (lem/model/efference.py). Plain tensors, dicts and lists
        # only: continuation_state.capture_learner serialises every _* and last_* attribute into
        # the episode companion and raises on anything else. All of it dies at a reset or
        # teleport (reset_transition_state / reset_context), never the window sums.
        self._eff_hist: deque = deque(maxlen=3)   # recent states' {latent, order, eye, step}, gradient stopped
        self._eff_eye_prev: Optional[torch.Tensor] = None   # [B, 2] eye az/el of the previous step, degrees
        self._eff_tenure: Optional[dict] = None   # where the order in force began: {latent, order, ach, step}
        self._eff_prev_order_id: Optional[int] = None
        self._eff_prev_order: Optional[torch.Tensor] = None
        # per-window [sum, count] of every efference gauge, and the lesson's own time [sum ms, n].
        # PUBLIC names on purpose (review MINOR 8): continuation_state saves every _* attribute into
        # the episode companion, so a window sum with a _ name rode the hourly save and the first
        # block after a resume mixed moments from before it; every other block starts fresh
        self.efference_window: Dict[str, list] = {}
        self.efference_timing: list = [0.0, 0]
        self.efference_pending: dict = {}         # the gaze copy's gauges until the lesson reads them
        # the newest graded tenure, per body: {step, order_began, expected_cm/deg [B, H, 3] (formed
        # where the order began), achieved_cm/deg [B, H, 3], order, expected_shuffled_cm/deg (another
        # body's order, B > 1)}; None until the first order of the right length ends
        self.last_efference_tenure: Optional[dict] = None
        # CAUSAL-CHAIN WORKING STATE. last_plan_record is this step's planning event (None on a
        # step without one, cleared at the start of every step); _transition_epoch counts resets
        # and teleports so a prediction is never paired with a moment from a different body
        # history. (The body-path order before the smoother, kept for the retired plan-share
        # gauge, is gone: the recorder compares the plan's imagined order with the order sent.)
        self.last_plan_record: Optional[dict] = None
        # the NEWEST planning event since the last reset, whatever step it was made at (it carries
        # its own "step"): for readers that show the latest plan (the explainer's snapshot). Public:
        # never episode state, so a resume starts without one until the next plan
        self.newest_plan_record: Optional[dict] = None
        self._causal_failures = 0
        self._transition_epoch = 0
        # smooth_actions in (0,1]: executed commands are blended with the previous
        # executed command (position/strength lerp, quaternion nlerp) so motion comes in
        # strokes instead of 50 ms twitches. 0 = off. The LOSS still grades the raw
        # command; only the body's input is smoothed (like muscle low-pass filtering).
        self.smooth_actions = float(smooth_actions)
        self._prev_mfa: Optional[torch.Tensor] = None
        self.movement_sensor_mod = bool(movement_sensor_mod)
        self.movement_param_mod = bool(movement_param_mod)
        self._base_lrs: Optional[list] = None   # filled after optimizer creation
        if intention:
            it = cfg.setdefault("training", {}).setdefault("intention", {})
            # effort penalty scaling: legacy sums strength over the whole batch
            # against an AVERAGED prediction loss (8 bodies = 8x relative effort
            # penalty, external review 2026-09-08 finding 4); per-body averages it.
            # Flag-gated: flips only by explicit declared attempt.
            it["effort_per_body"] = bool(planner_effort_per_body)
            it["prospective_wellness_w"] = (float(prospective_wellness_w)
                                              if prospective_wellness == "add" else 0.0)
            # ONE. transition() carries the same achieved pose at every depth, because a
            # predicted proprioception token cannot be decoded back to the ten physical
            # handle fields the motor token needs - so at depth 2 the candidate action is
            # encoded against the pose from BEFORE the rollout. Capping the signature
            # default was not enough: this config value is passed explicitly by the
            # production caller and said 2.
            #
            # Raise it when an achieved-state predictor exists AND has passed the causal
            # test, not before.
            it.update({"enabled": True, "n_steps": 20, "lr": 0.1, "strength_reg": 0.01,
                       "max_plan_steps": 1})
            self.intention_gen = IntentionGenerator(cfg, self.forward_model,
                                                    self.encoder_registry, self.prediction_heads)
            # prospective wellness (2026-09-09 plan; component selection and
            # statistic corrected same day by external review): the planner's
            # Adam loop may add weight x (1 - predicted BODILY wellness of the
            # imagined next state). BODILY columns only - tactile_comfort,
            # capability, motor_state (EVALUATOR_NAMES indices 1, 2, 4): the
            # first version averaged all five judge outputs, so a future scoring
            # high on "predictable" and "coherent" could outweigh worse touch,
            # capacity and posture - the pain-anticipation defect INSIDE the
            # term (review counterexample). Motor token detached in the term
            # (wireheading lever closed); evaluator params frozen during
            # planning. Judge accuracy at adoption time: wellness_align_err
            # 0.037 MEAN ABSOLUTE error = 3.7 points per component on the 0-100
            # scale (the earlier "0.19 RMS" misread it as MSE). The score fn is
            # wired for EVERY build so replan_last() can run paired A/B on
            # checkpoints trained without the term; the weight alone is gated.
            self.intention_gen._wellness_score_fn = (
                lambda sens, z_m: self.wellness.score_predicted_wellness(
                    sens, z_m)[:, [1, 2, 4]])
            self.intention_gen._wellness_freeze = [self.wellness]
        self.receipt = emit_model_receipt(cfg, self.encoder_registry, self.forward_model, self.motor_controller,
                                          self.wellness, gaze_head=self.gaze_head, voice_head=self.voice_head)
        from lem.model.contact_wellness import WELLNESS_CONTRACT
        self.receipt['wellness_contract'] = WELLNESS_CONTRACT
        self.receipt['contact_learning'] = bool(contact_learning)
        self.receipt['body_layout'] = self.body_layout
        # the core's depth was never recorded anywhere - H83's 133 args carry no depth key -
        # so a resume could not even tell what it was resuming; the loader now refuses a
        # depth mismatch, and this is the number it refuses against
        self.receipt['core_blocks'] = len(self.forward_model.blocks)
        # the proprioception row-index scale the encoder was trained under; the loader
        # refuses a checkpoint stamped differently (or not at all) instead of feeding
        # weights trained on a raw 0..25 ramp an input divided by 25
        from tokenizer.config import TokenizerConfig as _TC
        self.receipt['proprio_index_scale'] = float(_TC().proprio_index_range)
        from mimo_bridge.touch_adapter import PUSH_DEPTH_SCALE_CM, PUSH_ARRIVAL_SCALE_CMS
        self.receipt['touch_push_scale'] = [float(PUSH_DEPTH_SCALE_CM),
                                            float(PUSH_ARRIVAL_SCALE_CMS)]
        self.receipt['handle_names'] = list(self.handle_names)
        if self.vestibular_sense:
            # the field list, the scales and the input bound the encoder was trained under; a
            # checkpoint stamped differently is refused at load (vestibular_sense.stamp_mismatch)
            from .vestibular_sense import stamp as _vest_stamp
            self.receipt['vestibular_sense'] = _vest_stamp()
        # what the last load did to move an intermind checkpoint onto the vestibular slot (None
        # when nothing was moved). Public name on purpose: continuation_state captures only
        # _* and last_* attributes, and this belongs to the load, not to the episode.
        self.vestibular_switch = None
        # the running mean and spread that centre the vestibular input (runtime_state.pt), the
        # sense's declared weight in the next-moment lesson, and whether wellness's grounded
        # prediction-error term counts it; None / 1.0 / nothing with the flag off
        self.vestibular_norm = None
        self.vestibular_lesson_weight = 1.0
        if self.vestibular_sense:
            from .vestibular_sense import configure as _vest_configure
            _vest_configure(self, vestibular_lesson_weight, vestibular_wellness_pe)
        elif vestibular_lesson_weight is not None or vestibular_wellness_pe != "exclude":
            raise ValueError("--vestibular-lesson-weight and --vestibular-wellness-pe act on the "
                             "vestibular sense; they need --vestibular-sense on")
        # --vestibular-encoder-repair (vestibular_sense.repair_encoder; docs/gain_of_function_v4/agents/
        # vestibular_encoder_2026-09-25.md). "off" imports nothing, stamps nothing and changes nothing. PUBLIC
        # names: the episode companion carries every _* and last_* attribute and must never put an old record
        # back.
        if vestibular_encoder_repair != "off":
            from .vestibular_sense import REPAIR_MODES as _VR_MODES, repair_stamp as _vr_stamp
            if vestibular_encoder_repair not in _VR_MODES:
                raise ValueError("--vestibular-encoder-repair must be one of %s, not %r"
                                 % (_VR_MODES, vestibular_encoder_repair))
            if not self.vestibular_sense:
                raise ValueError("--vestibular-encoder-repair %s divides the vestibular encoder's last "
                                 "layer; it needs --vestibular-sense on" % vestibular_encoder_repair)
            self.receipt['vestibular_encoder_repair'] = _vr_stamp(vestibular_encoder_repair)
            # mode rescale_restart_fp32: this encoder computes in float32 whatever the run's autocast (review 2,
            # finding 4: in bf16 the revived weights' part is rounded away); nothing is saved, so every build
            # with the mode applies it
            from .vestibular_sense import apply_encoder_precision as _vr_precision
            _vr_precision(self, vestibular_encoder_repair)
        self.vestibular_encoder_repair = str(vestibular_encoder_repair)
        # what the last load's switch did (None when it divided nothing)
        self.vestibular_encoder_repair_record = None
        # the revival's lesson ceiling this step (vestibular_sense.revival_ceiling): the factor the sense's
        # weight was multiplied by, live and in the last replay step; None when it did not apply
        self.vestibular_revival_live = None
        self.vestibular_revival_replay = None
        params = (list(self.encoder_registry.parameters()) + list(self.forward_model.parameters()) +
                  list(self.prediction_heads.parameters()) + list(self.motor_controller.parameters()) +
                  list(self.wellness.parameters()))
        if self.gaze_head is not None:
            params += list(self.gaze_head.parameters())
        if self.voice_head is not None:
            params += list(self.voice_head.parameters())
        if self.recon_decoders is not None:
            params += list(self.recon_decoders.parameters())
        if self.td_gates is not None:
            params += list(self.td_gates.parameters())
        if self.tenure_heads is not None:
            # 2026-09-01: MISSING for the first five tenure runs (H47-H52). The pose
            # head and the order projection sat frozen at random init - the pose branch
            # "worked" because the trunk learned to satisfy a fixed random readout, and
            # the matching term could not work at all. The checkpoint-registration trap
            # has an optimizer-registration cousin; the test now guards both.
            params += list(self.tenure_heads.parameters())
        self.contact_consequences = None
        self.consequence_lesson = None
        if persistent_memory not in ('off','on'):
            raise ValueError('persistent_memory must be off or on')
        self.persistent_memory = None
        self.memory_history = None
        self._memory_state = self._memory_bias_current = self._nf_memory = None
        self.last_memory = {}
        self.memory_bias_history = deque(maxlen=int(cfg['training'].get('imagination',{}).get('steps',3))+1)
        from lem.model.persistent_memory import MEMORY_REPAIR_CHOICES, MEMORY_REPAIR_VERSION
        if persistent_memory_repair not in MEMORY_REPAIR_CHOICES:
            raise ValueError('persistent_memory_repair must be one of %s' % (MEMORY_REPAIR_CHOICES,))
        if persistent_memory_repair != 'off' and persistent_memory != 'on':
            raise ValueError('--persistent-memory-repair %s needs --persistent-memory on' % persistent_memory_repair)
        # the switch's record (LemLite._persistent_memory_switch, or the loaded checkpoint's receipt); PUBLIC
        # names so the episode companion, which carries every _* and last_* attribute, never puts an old
        # record back. _unwritten: the switch of this process, until the trainer has put it in the stats.
        self.persistent_memory_switch = None
        self.persistent_memory_switch_unwritten = None
        # --persistent-memory-repair, the goal reading: per graded sense and body, the delayed lesson's squared
        # error with and without the memory's varying part, summed on the device over a report (memory_report)
        self.memory_goal_window = None
        # the trainer's --replay-batch (train_mimo sets it): the switch's ring check reads the stored states in
        # calls of that size, as replay does
        self.replay_batch = None
        # the checkpoint folder this process loaded (load_checkpoint): the switch's way back
        self.loaded_from = None
        if persistent_memory == 'on':
            from lem.model.persistent_memory import PersistentMemory, MemoryHistory
            inputs = (self.n_mod+1)*self.d_model + 10 + len(self.handle_names)*9
            self.persistent_memory = PersistentMemory(inputs,self.n_mod,self.d_model,len(self.handle_names)*9,
                                                      repair=persistent_memory_repair).to(d)
            if persistent_memory_repair != 'off':
                # the constant's return the switch writes: each sense slot's pre-deploy size, by name
                # (lem/model/persistent_memory.py, THE CONSTANT'S RETURN)
                from lem.model.persistent_memory import PRE_DEPLOY_BIAS_RMS
                self.persistent_memory.set_constant_return([PRE_DEPLOY_BIAS_RMS.get(m, 0.0) for m in self.active],
                                                           int(persistent_memory_constant_return or 0))
            self.memory_history = MemoryHistory()
            params += list(self.persistent_memory.parameters())
        self.receipt['persistent_memory'] = persistent_memory
        if persistent_memory_repair != 'off':
            self.receipt['persistent_memory_repair'] = {'version': MEMORY_REPAIR_VERSION,
                                                        'mode': persistent_memory_repair}
        if self_state not in ('off','learn','perceive'):
            raise ValueError('self_state must be off, learn or perceive')
        self.self_state_mode = self_state
        if self_prediction not in ('legacy','grounded_residual'):
            raise ValueError('unknown self prediction mode')
        if self_state == 'off' and self_prediction != 'legacy':
            raise ValueError('grounded self prediction requires self-state learning')
        self.self_prediction_mode = self_prediction
        # off (default) = every run before 2026-09-26; on = a body whose stored self-percept
        # had to be rescaled, or sits at the limit in a restored memory, starts its stored
        # percepts again from zero, and no stored percept carries more than
        # ATTENTION_PEAK_LIMIT of a body's recall (lem/model/self_state.py SelfHistory.store,
        # ATTENTION_PEAK_LIMIT; self_state_integration.observe)
        if self_percept_release not in ('off', 'on'):
            raise ValueError('self_percept_release must be off or on')
        self.self_percept_release = self_percept_release == 'on'
        self.self_state = self.self_history = None
        self.self_rehearsal = None
        self._self_percept = None
        self.last_self_state = {}
        if self_state != 'off':
            from lem.model.self_state import SelfState, SelfHistory
            from lem.model.self_rehearsal import SelfRehearsal
            self.self_state = SelfState(self.d_model,self.n_mod,len(self.handle_names)*9,self_prediction).to(d)
            if self.self_percept_release:
                from lem.model.self_state import ATTENTION_PEAK_LIMIT
                self.self_state.attention_peak_limit = ATTENTION_PEAK_LIMIT
            self.self_history = SelfHistory()
            self.self_rehearsal = SelfRehearsal()
            params += list(self.self_state.parameters())
        self.receipt['self_state'] = self_state
        self.receipt['self_prediction'] = self_prediction
        self.receipt['runtime_state_required'] = True
        # learned-parameter count per component, for the explainer diagram; every
        # optimizer-registered module appears exactly once, so the values sum to
        # the checkpoint inventory's learned total
        def _np(mod):
            return sum(p.numel() for p in mod.parameters()) if mod is not None else 0
        _enc = {m: _np(e) for m, e in self.encoder_registry.encoders.items()}
        _comp = {('enc_' + m): n for m, n in _enc.items()}
        _shared = _np(self.encoder_registry) - sum(_enc.values())
        if _shared:
            _comp['enc_shared'] = _shared
        _comp.update(forward_model=_np(self.forward_model),
                     prediction_heads=_np(self.prediction_heads),
                     motor_controller=_np(self.motor_controller),
                     wellness=_np(self.wellness),
                     recon_decoders=_np(self.recon_decoders),
                     td_gates=_np(self.td_gates),
                     tenure_heads=_np(self.tenure_heads),
                     gaze_head=_np(self.gaze_head),
                     voice_head=_np(self.voice_head),
                     persistent_memory=_np(self.persistent_memory),
                     self_state=_np(self.self_state))
        self.receipt['component_params'] = {k: int(v) for k, v in _comp.items() if v}
        self.self_intention = None
        if self_choice != 'off':
            if self.self_state is None:raise ValueError('self choice needs learned self perception')
            from .self_intention import SelfIntention
            self.self_intention = SelfIntention(self_choice,self_choice_receipt)
        self.receipt['contact_profile'] = contact_profile
        if contact_profile in ('mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
            from lem.skin_mechanics import MECHANICAL_CONTRACT
            self.receipt['wellness_contract'] = MECHANICAL_CONTRACT
            if contact_profile in ('mechanical_v3','mechanical_v4'):
                from .contact_footprint import CONTRACT
                self.receipt['wellness_contract']=CONTRACT
        log.info('wellness contract: %s; contact learning: %s; persistent memory: %s',
                 self.receipt['wellness_contract'], contact_learning, persistent_memory)
        if contact_learning:
            from lem.model.contact_consequences import ContactConsequences
            from mimo_bridge.consequence_learning import ConsequenceLesson
            # the skin field width comes from the tokenizer that produces it, never
            # a literal: it went 8 -> 10 with the skin-stretch channels and this
            # head silently kept 8 until every live call raised a shape error
            from tokenizer.modalities.touch import _FIELD_NORMS
            self.contact_consequences = ContactConsequences(
                self.d_model, n_handles=len(self.handle_names),
                n_features=len(_FIELD_NORMS)).to(d)
            self.consequence_lesson = ConsequenceLesson(consequence_horizon)
            params += list(self.contact_consequences.parameters())
        # EFFERENCE HEADS (DESIGN.md D5, 2026-09-22, --efference-heads, default off). The order
        # the body was sent, given DIRECTLY beside the core's latent, predicts each handle's
        # movement in centimetres and turn in degrees over the next moment and over the order's
        # tenure, and the eyes' turn in degrees at 50, 100 and 150 ms (lem/model/efference.py).
        # APPENDED LAST, after every module that existed before it, so a checkpoint saved
        # without it migrates its Adam moments by name and only these parameters start fresh;
        # registered in _side_modules so it saves and loads under its own name.
        self.efference = None
        self.efference_weight = float(efference_weight)
        self.efference_tenure_steps = max(1, int(efference_tenure_steps))
        self.efference_seed = int(seed)
        self.efference_load = None     # what the last load did to the heads (public: never episode state)
        if efference_heads:
            if not next_frame_loss:
                raise ValueError("efference_heads requires next_frame_loss: the heads read the "
                                 "latent the delayed next-frame lesson computes for S_t-1 + A_t-1")
            from lem.model import efference as EF
            if not all(v > 0.0 for v in EF.EFFERENCE_SCALES.values()):
                raise ValueError("lem/model/efference.py EFFERENCE_SCALES holds an unmeasured "
                                 "(zero) divisor; the loss cannot be normalised")
            # version 2 (review of b1, 2026-09-23): the latent centred and normalised, the asked
            # change beside the order, the gaze head's own layers with the head's orientation, and
            # the gaze copy into vision's next-moment prediction (attached to the prediction heads
            # as a plain callable, never a submodule, so their saved tensors do not change). Built in
            # its own random stream: the run's stream is left where a build without it leaves it
            self.efference = EF.make_heads(
                self.d_model, len(self.handle_names), self.GAZE_IDX, self.handle_names.index("head"),
                self.n_mod, self.active.index("vision") if "vision" in self.active else None,
                self.efference_seed, eye_row_convention=self.eye_row_convention).to(d)
            EF.attach_gaze_copy(self.prediction_heads, self.efference)
            params += list(self.efference.parameters())
            self.receipt['efference_heads'] = EF.stamp(self)
            self.receipt['component_params']['efference_heads'] = int(
                sum(p.numel() for p in self.efference.parameters()))
        # CAUSAL CHAIN (DESIGN.md D13, --causal-chain, default off). Every planning event keeps
        # its goal, its chosen plan's predicted next state and predicted wellness, and the same
        # for alternatives scored from the SAME captured state (hold, the reflex controller's
        # proposal, two seeded perturbations); the trainer's CausalChainRecorder
        # (mimo_bridge/causal_chain.py) stamps the latch and resolves every latched order's
        # prediction against what happened. No trainable parameter; no draw from any global
        # random stream (the perturbations use a generator seeded from (seed, step)).
        self.causal_chain = bool(causal_chain)
        self._causal_seed = int(seed)
        # CONTACT HEAD (DESIGN.md D10, --contact-head, default off; lem/model/contact_head.py). Per
        # skin patch, from the core's latent for the moment before AND the order given directly,
        # the probability that the patch is touched in the next moment and, given a touch, that an
        # own part or an outside thing is among what touches it. Graded on the world's own
        # per-patch contact report; its next-moment gradient trains the core as well as the head.
        # APPENDED LAST, so a checkpoint without it migrates every Adam moment by name and only
        # these parameters start fresh; registered in _side_modules under "contact_head".
        # Its running statistics (the latent's centre, the patches' base rates) ride
        # runtime_state.pt; its per-body history (the previous moment's patch state, the
        # habituation memory) is episode state, cleared at every reset.
        self.contact_head = None
        self.contact_tracker = None
        self.contact_head_weight = None
        self.contact_control_weight = float(contact_control_wellness)
        # b3's hook: set before _init_wellness_channels (the last call in __init__), so the channel
        # joins the normalised ranking's channel list
        self.contact_control_enabled = False
        self._contact_prev_state = None      # [B, 53] the previous moment's patch states (CPU ints)
        self._contact_prev_valid = None      # [B] whether that report existed
        self._contact_habit = None           # per body {pair: [steps]} - the habituation memory
        self._contact_value_state = None     # per body [sum, weight] of the ranking's running value
        self._contact_control_value = None   # [B] this moment's contact_control value (b3's hook)
        self.last_contact_control = None     # this moment's scoring, per body (plain types)
        if contact_head:
            if not next_frame_loss:
                raise ValueError("contact_head requires next_frame_loss: the head reads the latent "
                                 "the delayed next-frame lesson computes for S_t-1 + A_t-1")
            from lem.model import contact_head as CH
            from .action_decoder import POSITION_HANDLES
            self.contact_head_weight = float(CH.CONTACT_HEAD_WEIGHT if contact_head_weight is None
                                             else contact_head_weight)
            _ch_body = [n for i, n in enumerate(self.handle_names) if i != self.GAZE_IDX]
            # built on the CPU inside a forked random stream: the head's initial weights come from
            # the run's seed and the run's own stream is left exactly where a build without the
            # head leaves it
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed) + 4_000_004)
                self.contact_head = CH.ContactHead(
                    self.d_model, self.n_mod, len(_ch_body),
                    order_mask=CH.order_input_mask(_ch_body, POSITION_HANDLES))
            self.contact_head = self.contact_head.to(d)
            params += list(self.contact_head.parameters())
            self.contact_tracker = CH.ContactTracker()
            self.contact_control_period = CH.control_period(self.efference_tenure_steps)
            self.receipt['contact_head'] = {
                "version": CH.CONTACT_HEAD_VERSION, "patch_names": list(CH.PATCH_NAMES),
                "order_scales": {"pos_cm": CH.ORDER_POS_CM, "rot_deg": CH.ORDER_ROT_DEG},
                "order_inputs": int(self.contact_head.order_cols.numel()),
                "order_input_rule": CH.ORDER_INPUT_RULE,
                "sizes": {"hidden": CH.HEAD_HIDDEN, "patch_hidden": CH.HEAD_PATCH_HIDDEN,
                          "patch_embed": CH.HEAD_PATCH_EMBED},
                "weight": self.contact_head_weight, "line_quantile": CH.LINE_QUANTILE,
                "line_horizon_steps": CH.LINE_HORIZON_STEPS,
                "centre_horizon_steps": CH.CENTRE_HORIZON_STEPS,
                "base_rate_horizon_steps": CH.BASE_RATE_HORIZON_STEPS,
                "control_period": self.contact_control_period,
                "gate": {"readings": CH.CONTACT_GATE_READINGS, "open_noise": CH.CONTACT_GATE_OPEN_NOISE,
                         "close_noise": CH.CONTACT_GATE_CLOSE_NOISE,
                         "stale_steps": CH.CONTACT_GATE_STALE_STEPS},
                "value": CH.CONTACT_VALUE, "value_orders": CH.CONTACT_VALUE_ORDERS,
                "contact_control_weight": self.contact_control_weight}
            self.receipt['component_params']['contact_head'] = int(
                sum(p.numel() for p in self.contact_head.parameters()))
        if self.contact_control_weight > 0.0:
            if self.contact_head is None:
                raise ValueError("--contact-control-wellness pays onsets the contact head predicted; "
                                 "it needs --contact-head on")
            self.contact_control_enabled = True
        elif self.contact_control_weight < 0.0:
            raise ValueError("--contact-control-wellness is a weight of 0 or more")
        # MOTOR SYNERGY (DESIGN.md D11, --motor-synergy, default off; lem/model/motor_synergy.py). A
        # compact intention z becomes a coordinated change of every body handle's target relative
        # to where the body is; trained on hindsight (the changes the body really made over each
        # order's tenure, weighted by following x movement); the planner descends on z and, with
        # --synergy-babble on, babble draws z. APPENDED LAST, after the contact head, so a
        # checkpoint without it migrates every Adam moment by name and only these parameters start
        # fresh; registered in _side_modules under "motor_synergy".
        if smooth_source not in ("proposal", "executed"):
            raise ValueError("smooth_source must be 'proposal' or 'executed'")
        self.smooth_source = str(smooth_source)
        self.synergy = None
        self.motor_synergy = bool(motor_synergy)
        self.synergy_planner = bool(synergy_planner)
        self.synergy_babble = bool(synergy_babble)
        self.synergy_residual_l2 = None if synergy_residual_l2 is None else float(synergy_residual_l2)
        self.synergy_tenure_steps = max(1, int(synergy_tenure_steps))
        self._syn_pending = None           # the order in force: {ach [B,H,10], order [B,H,9], step}
        self._syn_prev_ach = None          # the previous moment's achieved pose [B,H,10]
        self._syn_prev_order_id = None
        self._syn_prev_order = None
        self._syn_execution = None
        self._syn_latched_source = "unknown"
        self._syn_bank = None              # recent hindsight samples: {ach, x, w} ring tensors, n, ptr
        self._syn_window: Dict[str, list] = {}
        self._syn_z_window: List[torch.Tensor] = []
        self._syn_live: Dict[str, torch.Tensor] = {}
        self._syn_live_regimes = {}
        self._syn_following = {}
        self._synergy_sigma = None         # babble's scale on z, recalibrated every babble moment
        self._smooth_exec_prev = None      # --smooth-source executed: the order in force until now
        if self.synergy_babble and not self.motor_synergy:
            raise ValueError("--synergy-babble draws babble in the synergy's intention space; it "
                             "needs --motor-synergy on")
        if self.synergy_residual_l2 is not None:
            if not self.synergy_planner:
                raise ValueError("--synergy-residual-l2 needs --synergy-planner on")
            if not self.motor_synergy:
                raise ValueError("--synergy-residual-l2 is the planner's residual beside the synergy; "
                                 "it needs --motor-synergy on")
            if self.synergy_residual_l2 <= 0.0:
                raise ValueError("--synergy-residual-l2 must be above 0 (a free residual would let "
                                 "the planner leave the synergy at no cost)")
        if self.motor_synergy:
            from lem.model import motor_synergy as MS
            from .action_decoder import POSITION_HANDLES
            if MS.unmeasured():
                raise ValueError("lem/model/motor_synergy.py holds unmeasured constants %s"
                                 % ", ".join(MS.unmeasured()))
            k = int(MS.SYNERGY_DIM if synergy_dim is None else synergy_dim)
            if k < 1:
                raise ValueError("--synergy-dim must be at least 1")
            # built inside a forked random stream: the module's initial weights come from the run's
            # seed and the run's own stream is left where a build without it leaves it
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed) + 5_000_005)
                self.synergy = MS.MotorSynergy(self.handle_names, self.GAZE_IDX,
                                               sorted(POSITION_HANDLES), k)
            self.synergy = self.synergy.to(d)
            params += list(self.synergy.parameters())
            self._synergy_seed = int(seed)
            self.receipt['motor_synergy'] = self._synergy_stamp()
            self.receipt['component_params']['motor_synergy'] = int(
                sum(p.numel() for p in self.synergy.parameters()))
        if self.smooth_source != "proposal":
            self.receipt['smooth_source'] = self.smooth_source
        # the two motor interface settings are stamped only when they differ from the old model's,
        # so a default build writes the receipt the previous code wrote (load_checkpoint warns when
        # a checkpoint was trained under the other setting: what the weights learned about the eye
        # row's pitch, or about the fields F4 replaces, then reads differently)
        if self.eye_row_convention != "legacy":
            self.receipt['eye_row_convention'] = self.eye_row_convention
        if self.executed_fields_only_on:
            self.receipt['executed_fields_only'] = True
        # NEXT-FRAME NORMALISATION (DESIGN.md D12, --next-frame-norm persistence, default off;
        # lem/model/next_frame_norm.py). No trainable parameter: each sense's error in the delayed
        # next-frame lesson (and in replay's re-study of it) is weighted by the mean running
        # persistence error over that sense's own, so every sense is pushed to beat "nothing
        # changes" equally. PUBLIC name: its running averages ride runtime_state.pt, never the
        # episode companion. None when off, and nothing it touches runs.
        if next_frame_norm not in ("off", "persistence"):
            raise ValueError("next_frame_norm must be 'off' or 'persistence', got %r" % (next_frame_norm,))
        self.nf_norm = None
        if next_frame_norm == "persistence":
            if not next_frame_loss:
                raise ValueError("--next-frame-norm persistence weights the delayed next-frame lesson, "
                                 "which runs only with --next-frame-loss on")
            from lem.model import next_frame_norm as NFN
            self.nf_norm = NFN.NextFrameNorm()
            self.receipt['next_frame_norm'] = {
                "version": NFN.NEXT_FRAME_NORM_VERSION, "mode": "persistence",
                "horizon_steps": self.nf_norm.horizon, "weight_cap": self.nf_norm.cap,
                "zero_guard": NFN.ZERO_GUARD, "replay": True}
        # PROPRIOCEPTION TOKEN REPAIR (--proprio-token-repair, default off; mimo_bridge/proprio_token.py;
        # docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md). No trainable parameter: a lesson
        # that keeps every body's proprioception token moving with its own posture. "off" imports nothing,
        # stamps nothing and adds no term. PUBLIC names: the window rides runtime_state.pt, never the
        # episode companion.
        if proprio_token_repair not in ("off", "within_body"):
            raise ValueError("--proprio-token-repair must be 'off' or 'within_body', not %r"
                             % (proprio_token_repair,))
        if proprio_token_repair == "off" and (proprio_token_strength is not None or proprio_token_line is not None
                                              or proprio_token_skip is not None or proprio_token_deaf is not None):
            raise ValueError("--proprio-token-strength, --proprio-token-line, --proprio-token-deaf and "
                             "--proprio-token-skip set the within-body lesson; they need --proprio-token-repair "
                             "within_body")
        self.proprio_token_repair = str(proprio_token_repair)
        self.proprio_within = None
        self._pw_pending = None
        if proprio_token_repair == "within_body":
            if "proprioception" not in self.active:
                raise ValueError("--proprio-token-repair within_body needs the proprioception sense")
            from .proprio_token import WithinBody
            self.proprio_within = WithinBody(line=proprio_token_line, strength=proprio_token_strength,
                                             skip=proprio_token_skip, deaf=proprio_token_deaf)
            self.receipt['proprio_token_repair'] = self.proprio_within.stamp()
        # HEADING-FREE PROPRIOCEPTION (--proprio-heading, default off; mimo_bridge/proprio_heading.py). A forward
        # pre-hook on the proprioception encoder turns every row by minus the body's heading, on every path that
        # encodes the sense. "off" imports nothing, installs nothing and stamps nothing. PUBLIC names: a hook
        # handle cannot ride the episode companion.
        if proprio_heading not in ("off", "free"):
            raise ValueError("--proprio-heading must be 'off' or 'free', not %r" % (proprio_heading,))
        self.proprio_heading = str(proprio_heading)
        self.proprio_heading_hook = None
        self.proprio_heading_switch = False
        if proprio_heading == "free":
            if "proprioception" not in self.active or body_layout != "trunk_v1":
                raise ValueError("--proprio-heading free needs the proprioception sense of the trunk_v1 body "
                                 "(its heading is read from the lower trunk's row)")
            from . import proprio_heading as _PH
            _reg = getattr(self.encoder_registry, "_orig_mod", self.encoder_registry)
            self.proprio_heading_hook = _PH.install(_reg.encoders["proprioception"])
            self.receipt['proprio_heading'] = _PH.stamp()
            if self.proprio_within is not None:
                # review P2-5: the lesson reads its posture from the turned rows the encoder reads
                self.proprio_within.rows_transform = _PH.heading_free
        # ATTENTION_CAP (--attention-cap, default off; lem/model/attention_cap.py; docs/gain_of_function_v4/
        # attention_cap/ATTENTION_CAP_2026-10-05.md). "off" sets no bound and stamps nothing: every attention module
        # computes nn.MultiheadAttention's own arithmetic, bit for bit. "on": every head of the transformer's
        # cross-modal attention (all blocks) and of the motor encoder's 2 attention layers and its pool scores a key as
        # the bound times the cosine of its query and key. No parameter, buffer or random draw is added, so the state
        # dict, the random streams and the optimizer's parameter list are those of a build without it, and a checkpoint
        # loads either way (load_checkpoint writes a change of the cap down, attention_cap_change). The receipt and the
        # model config carry the cap when it is on (the config is part of the acting snapshot's execution
        # description). PUBLIC names: never episode state.
        from lem.model import attention_cap as _ACAP
        self.attention_cap_bound = _ACAP.check_switch(attention_cap, attention_cap_bound)
        self.attention_cap_change = None
        if self.attention_cap_bound is not None:
            _ACAP.apply(self, self.attention_cap_bound)
            self.receipt['attention_cap'] = _ACAP.stamp(self, self.attention_cap_bound)
            self.config['model']['attention_cap'] = {'version': _ACAP.VERSION, 'bound': self.attention_cap_bound}
        # THE ASKED CHANGE AS AN INPUT OF THE MOTOR ENCODER (--motor-asked-change, default off; lem/model/
        # motor_asked_change.py; docs/gain_of_function_v4/agents/motor_encoder_asked_change_2026-10-03.md). "off"
        # imports nothing, builds nothing, sets no attribute and stamps nothing. "on": per handle, the change the
        # command asks (command position minus achieved position, the turn from the achieved to the commanded
        # orientation, strength, radius; the eyes' row as its azimuth and elevation) enters the motor token through
        # a path with its own weights for every handle, added to the motor encoder's output by one forward hook on
        # the encoder module (so every producer of a motor token passes through it) in units of that output's
        # standard deviation over its numbers. The path's output layer starts at zero: the first tokens are the
        # tokens of a build without it, bit for bit. THE PATH LEARNS ONLY FROM ROWS WHOSE POSE AND COMMAND BELONG TO
        # THE SAME MODEL STEP (the delayed lesson, replay, the hold loss): every imagined row (transition(), the
        # imagination chain, the planner, the probes: lem.model.encoder_registry.imagined_motor_token) passes
        # through it with its weights detached. Built BEFORE the liveness repair (which takes the path's hidden
        # layer as one of its layers), in its own random stream; its parameters are APPENDED LAST, after the liveness
        # repair's, so a checkpoint without it migrates every Adam moment by name and only these parameters start
        # fresh; registered in _side_modules under "motor_asked_change". PUBLIC names: never episode state.
        if motor_asked_change not in ("off", "on"):
            raise ValueError("--motor-asked-change must be 'off' or 'on', not %r" % (motor_asked_change,))
        if motor_asked_change == "on":
            from lem.model import motor_asked_change as _MAC
            if "motor" not in self.active:
                raise ValueError("--motor-asked-change on adds a path beside the motor encoder; this build has no "
                                 "motor sense")
            self.motor_asked_change = _MAC.build(len(self.handle_names), self.GAZE_IDX, self.d_model, int(seed),
                                                 eye_row_convention=self.eye_row_convention).to(d)
            _mac_reg = getattr(self.encoder_registry, "_orig_mod", self.encoder_registry)
            self.motor_asked_change.attach(_mac_reg.encoders["motor"])
            # what the last load's switch check read (None when the loaded checkpoint already held the path)
            self.motor_asked_change_switch = None
            if self.intention_gen is not None:
                # frozen inside the planner's descent with the encoder registry it sits beside: planning must
                # leave no gradient on the path's weights (the list is the descent's extra frozen modules)
                self.intention_gen._wellness_freeze.append(self.motor_asked_change)
            self.receipt['motor_asked_change'] = self.motor_asked_change.stamp()
            self.receipt['component_params']['motor_asked_change'] = int(
                sum(p.numel() for p in self.motor_asked_change.parameters()))
        # AN ALL-SLOT PATH BESIDE EVERY PREDICTION HEAD'S ATTENTION (--heads-all-slot-path, default off; lem/model/
        # heads_all_slot_path.py; docs/gain_of_function_v4/restart_candidate/RESTART_CANDIDATE_HEADS_2026-10-04.md).
        # "off" imports nothing, builds nothing, sets no attribute and stamps nothing. "on": per head, the 6 newest
        # tokens minus the memory's constant, centred by a running mean and LayerNormed, enter a 2-layer network
        # through weights of their own per slot; the result is added to the head's prediction (PredictionHeads reads
        # the path from its instance dict, so every caller of the heads passes through it). Each output layer starts
        # at zero: the first predictions are those of a build without the path, bit for bit. THE PATH LEARNS FROM
        # EVERY LESSON THAT TRAINS THE HEAD IT SITS BESIDE (version 2: the delayed lesson, replay, and the imagination
        # chain's rows through lem.model.predictive_state.transition; its running centre moves on lived rows only;
        # the module text, GRADIENT, holds the measurement of Oct 4 that ended version 1's rule of detached weights
        # on imagined rows). Built BEFORE the liveness repair (which takes each
        # path's hidden layer as one of its layers), in its own random stream; its parameters are APPENDED LAST, after
        # the asked-change path's, so a checkpoint without it migrates every Adam moment by name and only these
        # parameters start fresh; registered in _side_modules under "heads_all_slot_path". PUBLIC names: never
        # episode state.
        if heads_all_slot_path not in ("off", "on"):
            raise ValueError("--heads-all-slot-path must be 'off' or 'on', not %r" % (heads_all_slot_path,))
        if heads_all_slot_path == "on":
            from lem.model import heads_all_slot_path as _ASP
            _asp_heads = getattr(self.prediction_heads, "_orig_mod", self.prediction_heads)
            self.heads_all_slot_path = _ASP.build(list(_asp_heads.head_modalities), len(self.active), self.d_model,
                                                  int(seed)).to(d)

            def _asp_constant():
                # the memory's constant as its bias adds it now (None without a persistent memory)
                pm = getattr(self, "persistent_memory", None)
                if pm is None:
                    return None
                from lem.model.motor_input_fold import effective_constant
                return effective_constant(pm)

            def _asp_measuring():
                lv = getattr(self, "liveness", None)
                return bool(lv is not None and getattr(lv, "unwatched_depth", 0) > 0)
            self.heads_all_slot_path.attach(self.prediction_heads, constant_fn=_asp_constant,
                                            measuring_fn=_asp_measuring)
            # what the last load's switch check read (None when the loaded checkpoint already held the path)
            self.heads_all_slot_path_switch = None
            # the newest lesson calls' rows for the head_read gauge (heads_all_slot_path_report); a PUBLIC name: never episode state
            self.heads_all_slot_path_gauge = deque(maxlen=4)
            if self.intention_gen is not None:
                # frozen inside the planner's descent with the prediction heads it sits beside: planning must leave
                # no gradient on the path's weights (the list is the descent's extra frozen modules)
                self.intention_gen._wellness_freeze.append(self.heads_all_slot_path)
            self.receipt['heads_all_slot_path'] = self.heads_all_slot_path.stamp()
            self.receipt['component_params']['heads_all_slot_path'] = int(
                sum(p.numel() for p in self.heads_all_slot_path.parameters()))
        # THE VESTIBULAR ENCODER'S LINEAR INPUT PATH (--vestibular-input-path, default off; lem/model/
        # vestibular_input_path.py; docs/gain_of_function_v4/agents/vestibular_token_2026-10-04.md). "off" imports
        # nothing, builds nothing, sets no attribute and stamps nothing. "on": u = Linear(28 -> 192, no bias)(the
        # encoder's own input row) x 1/sqrt(28) is added to the encoder's last-layer output in units of that output's
        # standard deviation over its 192 numbers, by one forward hook on the encoder module (so every producer of a
        # vestibular token passes through it). The weight starts at zero: the first tokens are the tokens of a build
        # without it, bit for bit. THE PATH LEARNS FROM THE LIVE ROW ONLY: the delayed lesson's re-encoding of the
        # previous row passes through it with its weight detached (step(), `_vip.frozen()`). Its parameter is
        # APPENDED LAST, after the asked-change path's and the heads' all-slot paths', so a checkpoint without it
        # migrates every Adam moment by name and only this parameter starts fresh; registered in _side_modules
        # under "vestibular_input_path". PUBLIC names: never episode state.
        if vestibular_input_path not in ("off", "on", "on_delayed_encoder_only"):
            raise ValueError("--vestibular-input-path must be 'off', 'on' or 'on_delayed_encoder_only', not %r"
                             % (vestibular_input_path,))
        if vestibular_input_path != "off":
            # "on": the delayed lesson trains neither the encoder's own layers nor the path (step(), the delayed
            # lesson). "on_delayed_encoder_only": it trains the encoder's own layers and not the path. The same path,
            # the same tokens at the switch.
            from lem.model import vestibular_input_path as _VIP
            if "vestibular" not in self.active:
                raise ValueError("--vestibular-input-path on adds a path beside the vestibular encoder; it needs "
                                 "--vestibular-sense on")
            _vip_reg = getattr(self.encoder_registry, "_orig_mod", self.encoder_registry)
            _vip_enc = _vip_reg.encoders["vestibular"]
            self.vestibular_input_path = _VIP.build(int(_vip_enc.mlp[0].in_features), self.d_model, int(seed),
                                                    wiring=vestibular_input_path).to(d)
            self.vestibular_input_path.attach(_vip_enc)
            # what the last load's switch check read (None when the loaded checkpoint already held the path)
            self.vestibular_input_path_switch = None
            if self.intention_gen is not None:
                # frozen inside the planner's descent with the encoder registry it sits beside
                self.intention_gen._wellness_freeze.append(self.vestibular_input_path)
            self.receipt['vestibular_input_path'] = self.vestibular_input_path.stamp()
            self.receipt['component_params']['vestibular_input_path'] = int(
                sum(p.numel() for p in self.vestibular_input_path.parameters()))
        # LIVENESS (DESIGN.md section 10, --liveness-repair, default off; lem/model/liveness.py).
        # "report": read-only hooks and the top-level liveness stats block. "on": also the revived
        # rows of the nine dead layers and the all-element paths beside the motor and touch pools,
        # APPENDED LAST so a checkpoint without them migrates every Adam moment by name; registered
        # in _side_modules under "liveness_repair". A public name: the episode companion skips it.
        self.liveness = None
        if liveness_repair not in ("off", "report", "on"):
            raise ValueError("liveness_repair must be off, report or on")
        if str(liveness_rule) != "live_line" and liveness_repair != "on":
            raise ValueError("--liveness-rule %s chooses which units the liveness repair revives; it needs "
                             "--liveness-repair on (the revived units and the all-element paths are built only in "
                             "mode on; --liveness-rule none keeps them and revives no unit)" % liveness_rule)
        if liveness_repair != "off":
            from lem.model import liveness as LV
            if LV.unmeasured():
                raise ValueError("lem/model/liveness.py holds unmeasured constants %s"
                                 % ", ".join(LV.unmeasured()))
            # --liveness-rule: which units the repair revives (lem/model/liveness.py WHICH UNITS THE REVIVAL
            # CHOOSES); the default "live_line" is the rule before the switch, bit for bit
            self.liveness = LV.LivenessRepair(self, liveness_repair, seed=int(seed),
                                              revive_rule=str(liveness_rule))
            if getattr(self, "motor_asked_change", None) is not None:
                # --motor-asked-change on: the path's hidden layer is one of the repair's layers (the monitor
                # reports it in every mode by its own rule). Added in a random stream of its own, so the other
                # layers' revived rows and the pool paths are built from the same random draws as without it
                from lem.model import motor_asked_change as _MAC_LV
                self.liveness.add_layer(self, _MAC_LV.LIVENESS_LAYER, int(seed) + _MAC_LV.SEED_OFFSET + 1)
            if getattr(self, "heads_all_slot_path", None) is not None:
                # --heads-all-slot-path on: each path's hidden layer is one of the repair's layers, each added in a
                # random stream of its own (after the asked-change path's, so every earlier layer keeps its draws)
                from lem.model import heads_all_slot_path as _ASP_LV
                for _i, _m in enumerate(self.heads_all_slot_path.head_modalities):
                    self.liveness.add_layer(self, _ASP_LV.liveness_layer(_m),
                                            int(seed) + _ASP_LV.SEED_OFFSET + 1 + _i)
            self.liveness = self.liveness.to(d)
            if liveness_repair == "on":
                params += list(self.liveness.parameters())
                self.liveness.attach_repair(self)
                self.receipt['component_params']['liveness_repair'] = int(
                    sum(p.numel() for p in self.liveness.parameters()))
            self.receipt['liveness_repair'] = self.liveness.stamp()
        if getattr(self, "motor_asked_change", None) is not None:
            # appended last (see the note where it is built)
            params += list(self.motor_asked_change.parameters())
        if getattr(self, "heads_all_slot_path", None) is not None:
            # appended after the asked-change path's (see the note where it is built)
            params += list(self.heads_all_slot_path.parameters())
        if getattr(self, "vestibular_input_path", None) is not None:
            # appended after the heads' all-slot paths' (see the note where it is built)
            params += list(self.vestibular_input_path.parameters())
        if motor_input_fold != "off":
            # --motor-input-fold (lem/model/motor_input_fold.py): no parameter, no random draw; its state saves as the
            # side module "motor_input_fold". Off: no attribute, nothing built
            from lem.model import motor_input_fold as _MIF
            self.motor_input_fold = _MIF.build(self, motor_input_fold)
        self.params = params
        self.n_params = sum(p.numel() for p in params)
        tr = cfg["training"]
        # TWO PARAMETER GROUPS, not one. Weight decay belongs on matrices, not on
        # LayerNorm gains, biases or embedding tables: a 1-D parameter has no
        # redundant directions to shrink, so decaying it just pulls a calibration
        # toward an arbitrary origin. Excluding them is standard practice and this
        # model was not doing it - one flat AdamW group, weight_decay 0.01 on all
        # 3.97M parameters.
        #
        # It matters here more than usual. Measured across H31's eight hourly
        # checkpoints, the 18 LayerNorm affine tensors in the forward model - 2,304
        # numbers, 0.34% of it - carry the drift toward copying the present:
        # transplanting the hour-8 affines onto the hour-1 model reproduces 159% of
        # the copy-R^2 rise and 158% of the history-sensitivity loss, and reverting
        # them on the hour-8 model recovers 68% and 54%. They are sufficient to
        # produce the whole effect.
        #
        # Selection is by SHAPE, not by a list of names. A name list goes stale the
        # first time a module is added; p.dim() < 2 catches every norm gain, every
        # bias and every 1-D buffer-like parameter that exists now or later.
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": tr["weight_decay"]},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr or tr["lr"], betas=tuple(tr["betas"]), fused=(d.type == "cuda"))
        log.info("optimizer: %d decayed tensors (%d params), %d undecayed (%d params)",
                 len(decay), sum(p.numel() for p in decay),
                 len(no_decay), sum(p.numel() for p in no_decay))
        self.grad_clip = tr["grad_clip"]
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]   # config truth, pre-resume

        # optional systems (defaults from config, like live_test.py)
        im = tr.get("imagination", {})
        self.imagination_enabled = im.get("enabled", True) if imagination is None else imagination
        self.imagination_steps = int(im.get("steps", 3))
        self.imagination_every = int(im.get("every_n_steps", 1))
        if self.imagination_enabled and self.ctx_len < self.imagination_steps + 1:
            # live_test.py does the same: not enough context for the chain
            self.imagination_steps = max(1, self.ctx_len - 1)
        self._exec_seen = 0        # steps that received a TransitionV1 executed action
        self._nf_aligned = 0       # next-frame lessons conditioned on the OUTGOING action
        self.enc_history: deque = deque(maxlen=self.imagination_steps + 1)
        self.memory_bias_history = deque(maxlen=self.imagination_steps + 1)
        self.enc_history_raw: deque = deque(maxlen=self.imagination_steps + 1)  # M8 targets
        self._nonfinite_steps = 0            # T7: skipped optimizer steps, cumulative
        # review P2/P3 (2026-09-02): the imagination chain used to start from
        # S_{t-3} with the CURRENT context - its own targets sat at attendable
        # positions inside that context, so two of three graded depths were
        # minimizable by copying, and the motor token carried the FUTURE pose.
        # These histories snapshot context and pose in step with the frames.
        self.ctx_history: deque = deque(maxlen=self.imagination_steps + 1)
        self.ach_history: deque = deque(maxlen=self.imagination_steps + 1)
        # the source mask that was true at each depth, so a synthetic target is
        # excluded at depth 2 even if it was real at depth 0 (TransitionV1.valid)
        self.valid_history: deque = deque(maxlen=self.imagination_steps + 1)
        self.motor_history: deque = deque(maxlen=self.imagination_steps)
        lr_cfg = tr.get("latent_reg", {})
        lr_on = lr_cfg.get("enabled", True) if latent_reg is None else latent_reg
        self.latent_regularizer = LatentRegularizer(d_model=self.d_model, sigma_threshold=lr_cfg.get("sigma_threshold", 2.0),
                                                    momentum=lr_cfg.get("momentum", 0.99)).to(d) if lr_on else None
        self.latent_reg_weight = lr_cfg.get("loss_weight", 0.01)
        cu = tr.get("curiosity", {})
        cu_on = cu.get("enabled", True) if curiosity is None else curiosity
        self.curiosity = None
        if cu_on:
            from lem.model.curiosity import CuriositySystem
            self.curiosity = CuriositySystem(pe_min=cu.get("pe_min", 0.1), pe_max=cu.get("pe_max", 0.4))
        self.step_count = 0
        # collapse watch: last 200 encoder tokens per modality (cpu, fp32) -> temporal std
        self.token_ring: Dict[str, deque] = {m: deque(maxlen=200) for m in self.active}
        # THE COLLAPSE WATCH ON EVERY BODY (token_temporal_std_bodies; docs/gain_of_function_v4/agents/
        # proprio_token_2026-09-25.md): the same samples as token_ring for all the batch's bodies. token_ring keeps
        # body 0 only, and body 0 was never deaf while 2 of 8 bodies' proprioception tokens were, for five days.
        # A reading, never a lesson: detached CPU copies. The leading underscore makes the episode companion carry
        # it across a resume as token_ring is carried, so the first report after a resume reads the full window.
        self._token_ring_bodies: Dict[str, deque] = {m: deque(maxlen=200) for m in self.active}
        self.context_buffer: Optional[torch.Tensor] = None
        self.last_vision_token: Optional[torch.Tensor] = None
        # the moment's encoded tokens and the achieved handle state, kept for probes that
        # need to re-run the transition with a substituted action (tools/causal_test.py)
        self.last_tokens: Optional[Dict[str, torch.Tensor]] = None
        self.last_achieved: Optional[torch.Tensor] = None
        # explainer stashes (mimo_bridge/explainer): what the live forward ACTUALLY
        # consumed and per-body outputs that otherwise die inside step(). All detached
        # references, no clones. last_context is the PRE-shift buffer - the context
        # update at the end of step() rebinds context_buffer to a new tensor, so this
        # reference keeps the exact sequence the forward attended to. last_tokens_fed
        # is the post-gate token set (== raw when the top-down gate is closed or off);
        # last_tokens above stays RAW, the measured-reality convention every probe uses.
        self.last_context: Optional[torch.Tensor] = None
        self.last_tokens_fed: Optional[Dict[str, torch.Tensor]] = None
        self.last_voice: Optional[torch.Tensor] = None
        self.last_predictions: Optional[Dict[str, torch.Tensor]] = None
        self.last_wellness_grounded: Optional[Dict[str, torch.Tensor]] = None
        self.checkpoints = CheckpointManager(cfg, checkpoint_dir) if checkpoint_dir else None
        # body schema (UEMA spec section 11): learns which sensory dimensions are "me" by
        # correlating motor commands with proprioception; its coherence score feeds wellness
        # (previously unwired in the bridge - the coherence input fell back to a cosine)
        self.body_schema = DifferentiableBodySchema(n_handles=len(self.handle_names),
                                                    object_names=SELF_OBJECTS).to(d)
        self._schemas: List[DifferentiableBodySchema] = [self.body_schema]   # grown to B lazily
        # what the world says the body can move, filled from each step's report; empty until
        # the first step, and an unknown thing reads as one the body cannot move
        self._object_physics_driven: dict = {}
        # THE SELF-TOUCH GATE'S RUN OF CONTROLLED READINGS, built once here so it exists from
        # the first step of a fresh run or a resume and survives every episode boundary. It
        # holds the last SELF_TOUCH_READINGS contrasts and the step the newest arrived at.
        from collections import deque as _deque
        self._self_touch_ring = _deque(maxlen=SELF_TOUCH_READINGS)
        self._self_touch_ring_step = 0
        self.last_self_touch_gate_why = "no controlled reading yet"
        self.last_self_touch_gate_mean = None
        self.last_self_touch_gate_line = None
        # why the imagination-trust gauge could not be measured, when it could not
        self.last_align_err_skipped: Optional[str] = None
        self.replay: Optional[ReplayMemory] = None    # created by enable_replay()
        self._replay_motor_rebuilt = 0   # replay steps that rebuilt the motor token
        self._replay_motor_unlabeled = 0  # E15: windows whose step-from moment had no action
        self._replay_ctx_embedded = 0    # replay steps whose context got the modality embedding
        # active senses that reached the token dict with no encoder output this step
        self._no_encoder_output: List[str] = []
        # per-depth horizon-license gradings skipped because no sense was graded real:
        # a frozen license is a planner pinned at horizon 1, and it used to be silent
        self._plan_depths_ungraded = 0
        self.noise_rng = torch.Generator(device="cpu").manual_seed(seed)
        # UEMA 11.4 Levels 2/3, DARK: observes fused tokens + wellness, logs membership,
        # feeds NOTHING (no loss, no goal, no wellness term). EMA buffers only.
        self._hself = None
        if hierarchical_self == "dark":
            from mimo_bridge.hierarchical_self import HierarchicalSelfDark
            self._hself = HierarchicalSelfDark(sorted(self.active), self.d_model,
                                               n_slots=int(hself_slots)).to(d)
        # WHICH READING OF THE BODY MAP WELLNESS CARRIES (see the wellness call in step)
        self.wellness_coherence = str(wellness_coherence)
        # how much a fully self-touching moment may add to wellness, once the gate opens.
        # 0.0 keeps the term at exactly zero, which is the default until the evidence is in
        self.self_touch_wellness_weight = float(self_touch_wellness)
        # A TERM THE PLANNER CANNOT SEE. The planner scores an IMAGINED moment with the
        # approximate evaluator, which has one output per name in APPROXIMATE_TERMS and is
        # trained to copy exactly those. self_touch is not among them. So while its weight is
        # above zero the body's felt value carries it and every imagined moment's value does
        # not, which means the planner cannot aim at it and cannot avoid it. That is a real
        # choice - widening the evaluator changes its shape and so its checkpoint - and it is
        # said out loud at launch rather than discovered later from a flat number.
        if self.self_touch_wellness_weight > 0.0 and "self_touch" not in APPROXIMATE_TERMS:
            log.warning(
                "wellness: the self-touch term is paying at weight %.3f and the approximate "
                "evaluator does not predict it (it predicts %s). The body's own felt value "
                "carries the term; the value of every IMAGINED moment does not, so the "
                "planner can neither aim at self-touch nor avoid it, and the gauge of how far "
                "the two evaluators agree leaves it out. Widen the approximate evaluator to "
                "%d outputs and add the name to lem.model.wellness.APPROXIMATE_TERMS to make "
                "it visible.", self.self_touch_wellness_weight,
                ", ".join(APPROXIMATE_TERMS), len(APPROXIMATE_TERMS) + 1)
        self.last_self_touch_gate = 0.0
        # UEMA 11.4 Levels 2/3 over NAMED THINGS: the caregiver, each toy, the mobile, the
        # critters, the infant's own hand in view. One presence signal each, read from the
        # world, so a thing can be pointed at and can survive the encoders changing - which
        # the hierarchical self above cannot (2026-09-19 review). Also dark.
        self._entity_self = None
        self._self_value = None
        self._entity_presence = None
        self.entity_self_mode = str(entity_self)
        self.self_value_weight = float(self_value_weight)
        self._entity_self_cfg = dict(teacher=str(entity_self_teacher), level=str(self_value_level),
                                     gate_windows=int(self_value_gate_windows))
        if self.entity_self_mode != "off":
            # The tracker keeps one running average per body, so it is built at the first step,
            # when the body count is known from the frames themselves. The reader needs only
            # the names and is built here. It adds nothing until its own gate opens: only
            # entities whose link passed the evidence test are read, and the real reader must
            # track wellness better than a shuffled copy of itself for a run of windows. A
            # weight of 0 keeps it watching whatever the gate says.
            from mimo_bridge.entity_presence import ENTITY_NAMES
            from mimo_bridge.self_value_reader import SelfValueReader
            self._self_value = SelfValueReader(ENTITY_NAMES, weight=self.self_value_weight,
                                               level=str(self_value_level),
                                               gate_windows=int(self_value_gate_windows)).to(d)
            log.warning("entity self: %s over %d named things, teacher=%s; reader weight %.3f (%s)",
                        self.entity_self_mode, len(ENTITY_NAMES), entity_self_teacher,
                        self.self_value_weight,
                        "live once its gate opens" if self.self_value_weight > 0 else "watching only")
        self._init_wellness_channels(controllability, learning_progress, wellness_rank,
                                     self._with_contact_control_weight(wellness_weights),
                                     curiosity_mode, consequence_self_contact, contact_learning)
        log.info("LemLite: %d params, active=%s, d_model=%d, ctx=%d, amp=%s, imagination=%s, wellness=%s",
                 self.n_params, self.active, self.d_model, self.ctx_len, self.use_amp, self.imagination_enabled,
                 self.wellness_mode.value)

    # ------------------------------------------------------------------ helpers
    def dev_limits(self, step: int) -> dict:
        from .body_layout import extend_limits
        return extend_limits(_get_developmental_phase(self.config, step), len(self.handle_names))

    def _gl_clear(self):
        """Goal-locality state dies with the ring (same staleness rules: a held goal
        from an old world or an old batch shape is a defect, not a memory)."""
        self._gl_held_tok = None
        self._gl_held_born = None
        self._gl_held_mask = None
        self._gl_reach_ema = None
        self._gl_instant_ema = None
        self._gl_held_age0 = None
        self._gl_goal_index = None
        self._gl_goal_age = None

    def _goal_position(self, best_t: torch.Tensor, where=None):
        """([ring index per body], [age in model steps per body]) of the goal this planning event
        aims at: the ranking's pick, or goal locality's (where = its index and age lists)."""
        if where is not None:
            return [int(v) for v in where[0]], [int(v) for v in where[1]]
        T = len(self.goal_ring)
        idx = [int(v) for v in best_t.reshape(-1).tolist()]
        return idx, [T - 1 - i for i in idx]

    def _gl_select(self, cur: Dict[str, torch.Tensor], wl_stack: torch.Tensor,
                   tok_stacks: Dict[str, torch.Tensor], best_t_global: torch.Tensor,
                   step: int, allowed: Optional[torch.Tensor] = None):
        """Goal locality + commitment: best-felt WITHIN SELF-MEASURED REACH, held
        until arrival, escape, or old age.

        Constant-free by design: the reach ball R is the model's own running measure
        of how far the world drifts in 14 steps (distance to the ring entry 14 back,
        EMA), and the arrival line is its running one-step drift (EMA) - both grow
        or shrink with the body's earned motion, nothing operator-tuned. Commitment:
        the census (2026-09-06) measured an uncommitted reachable-best flipping every
        plan event, and a goal that flips every 250 ms cannot anchor a journey - so
        the pick is KEPT while it stays inside the ball, unarrived, and younger than
        the ring lifetime (200 steps).

        Returns (goal_state {m: [B,d]}, info dict). Falls back to the global
        best-felt pick for any env whose ball is empty (census: never observed; the
        counter proves it stays that way).

        allowed [T, B] bool (--wellness-rank normalized): the moments the ranking allows. A
        moment over the harm line is never allowed, so when every moment inside the ball is
        over it the ball counts as empty and the global pick - which respects the gate -
        stands (review of b3: the largest-value pick over an all-minus-infinity ball returned
        ring index 0, the oldest moment, over the line and possibly outside the ball).
        """
        T, B = wl_stack.shape
        # distances current -> every ring entry, mean over senses: [T, B]
        dists = torch.stack([
            ((tok_stacks[m] - cur[m].unsqueeze(0)) ** 2).mean(-1)
            for m in self.active]).mean(0)
        # self-measured scales (need a 15-deep ring for the 14-step radius)
        if T >= 15:
            r_now = dists[T - 1 - 14]                                    # [B]
            self._gl_reach_ema = (r_now if self._gl_reach_ema is None
                                  else 0.95 * self._gl_reach_ema + 0.05 * r_now)
        inst_now = torch.stack([
            ((tok_stacks[m][-1] - tok_stacks[m][-2]) ** 2).mean(-1)
            for m in self.active]).mean(0)                               # [B]
        self._gl_instant_ema = (inst_now if self._gl_instant_ema is None
                                else 0.98 * self._gl_instant_ema + 0.02 * inst_now)
        if self._gl_reach_ema is None:
            return None, {"on": True, "warming": True}
        R = self._gl_reach_ema
        # ---- keep or drop the held goal, per env ----
        arrived_n = 0
        if self._gl_held_mask is None or self._gl_held_mask.shape[0] != B:
            self._gl_held_mask = torch.zeros(B, dtype=torch.bool, device=wl_stack.device)
        if self._gl_held_tok is not None:
            d_held = torch.stack([
                ((self._gl_held_tok[m] - cur[m]) ** 2).mean(-1)
                for m in self.active]).mean(0)                           # [B]
            arrived = self._gl_held_mask & (d_held < self._gl_instant_ema)
            escaped = self._gl_held_mask & (d_held > R)
            aged = self._gl_held_mask & ((step - self._gl_held_born) > 200)
            arrived_n = int(arrived.sum())
            self._gl_arrivals += arrived_n
            self._gl_held_mask = self._gl_held_mask & ~(arrived | escaped | aged)
        # ---- fresh pick for envs without a held goal ----
        need = ~self._gl_held_mask                                        # [B]
        inside = dists <= R.unsqueeze(0)                                  # [T, B]
        if allowed is not None:
            inside = inside & allowed
        none_inside = ~inside.any(dim=0)                                  # [B]
        self._gl_fallbacks += int((need & none_inside).sum())
        wl_local = wl_stack.masked_fill(~inside, float("-inf"))
        best_local = wl_local.argmax(dim=0)                               # [B]
        pick = torch.where(none_inside, best_t_global, best_local)        # [B]
        picked_tok = {m: tok_stacks[m].gather(
            0, pick.view(1, -1, 1).expand(1, -1, tok_stacks[m].shape[-1])
        ).squeeze(0) for m in self.active}                                # {m: [B,d]}
        age_pick = (T - 1 - pick).long()                                   # [B]
        if self._gl_held_tok is None:
            self._gl_held_tok = {m: v.clone() for m, v in picked_tok.items()}
            self._gl_held_born = torch.full((B,), step, dtype=torch.long,
                                            device=wl_stack.device)
            self._gl_held_age0 = age_pick.clone()
        else:
            for m in self.active:
                self._gl_held_tok[m] = torch.where(need.view(-1, 1),
                                                   picked_tok[m],
                                                   self._gl_held_tok[m])
            self._gl_held_born = torch.where(need,
                                             torch.full_like(self._gl_held_born, step),
                                             self._gl_held_born)
            _a0 = self._gl_held_age0
            if _a0 is None or _a0.shape != age_pick.shape:
                _a0 = age_pick.clone()
            self._gl_held_age0 = torch.where(need, age_pick, _a0)
        self._gl_held_mask = torch.ones_like(self._gl_held_mask)
        # where each body's goal sits now: its age grows by one a step while it is held
        _age_now = (self._gl_held_age0 + (step - self._gl_held_born)).tolist()
        self._gl_goal_age = [int(a) for a in _age_now]
        self._gl_goal_index = [int(T - 1 - a) if a <= T - 1 else -1 for a in self._gl_goal_age]
        info = {"on": True,
                "R": round(float(R.mean()), 4),
                "instant": round(float(self._gl_instant_ema.mean()), 4),
                "held_age_mean": round(float((step - self._gl_held_born).float()
                                             .mean()), 1),
                "new_picks": int(need.sum()), "arrived_now": arrived_n,
                "arrivals_total": int(self._gl_arrivals),
                "fallbacks_total": int(self._gl_fallbacks)}
        return {m: v for m, v in self._gl_held_tok.items()}, info

    def reset_transition_state(self):
        """Forget every lesson that would otherwise span a simulator teleport.

        An episode reset or a body rebuild moves the body discontinuously. Anything still
        holding pre-teleport state pairs a post-teleport observation with a pre-teleport
        cause: the delayed next-frame lesson would predict across the jump, the imagination
        chain would roll out from a body that no longer exists, and the context buffer would
        carry the old episode's history into the new one. Replay already refuses to build a
        window across the boundary; this is the model-side equivalent, and the training loop
        must additionally drop its own held action and executed-action carry.
        """
        self._nf_tokens = self._nf_ctx = self._nf_raw = self._nf_mask = None
        if getattr(self, 'proprio_within', None) is not None:
            # --proprio-token-repair: the window must not span the teleport
            self.proprio_within.reset()
        if getattr(self,'memory_history',None) is not None:
            self.memory_history.reset()
        if getattr(self,'self_history',None) is not None:
            self.self_history.reset()
        self._memory_state = self._memory_bias_current = self._nf_memory = None
        if hasattr(self,'memory_bias_history'): self.memory_bias_history.clear()
        self._nf_pred = None
        # the delayed error belongs to the body that just vanished. Leaving it means the
        # next two steps report a pre-teleport number as a measurement, and curiosity's
        # drive is computed from it.
        self.last_next_pe = {}
        self.last_causal_pe = float("nan")
        self._lag_lat = self._lag_last = self._lag_hs = None
        self._tenure_pending = None
        self._tenure_prev_order = None
        self._tenure_prev_order_id = None
        self._tenure_bank.clear()
        self._tenure_bank_el.clear()
        self._agency_pay_ema = None
        self._nf_ach = None
        self._nf_handvis = None
        # review P8: the planner kept steering toward pre-teleport moments for up to
        # 200 steps after a reset or body rebuild - goals from a body that no longer
        # exists, scored by wellness measured under the old body
        self.goal_ring.clear()
        self._goal_ring_pose.clear()
        self._hand_previous_joints = None
        if getattr(self, "consequence_lesson", None) is not None:
            self.consequence_lesson.reset()
        self._goal_ring_w0.clear()
        self._goal_ring_cq.clear()
        self._goal_ring_drive.clear()
        self._goal_ring_ch.clear()
        self._rank_detrend = {}   # the ring's running averages belong to the same history
        self._gaze_hold = None
        self._gl_clear()          # a held goal from the old world is the same defect
        self.last_intention = None
        self.last_tenure_pe = {}
        self._last_plan_inputs = None
        self.last_tenure_gap = {}
        self.last_tenure_tok = {}
        self.last_agency = {}
        self.enc_history.clear()
        self.enc_history_raw.clear()
        self.valid_history.clear()
        self.motor_history.clear()
        self.ctx_history.clear()
        self.ach_history.clear()
        self.last_vision_token = None
        self.last_tokens = self.last_achieved = None
        self.last_self_touch_audit = None
        # The self-touch gate's run of controlled readings is NOT cleared here. It is built
        # in __init__ and it is evidence gathered across episodes: an episode boundary is not a
        # reason for the gate to forget what it measured. Built here, it was missing on the
        # first step after a resume, which is where H84 crashed at 18:22 on 2026-09-21.
        self.last_self_touch_controlled = None
        self.last_context = self.last_tokens_fed = None
        self.last_predictions = self.last_wellness_grounded = None
        self.last_voice = None
        self._forget_efference_and_causal()
        for sc in self._schemas:
            if hasattr(sc, "reset_prev"):
                sc.reset_prev()
        if self.replay is not None:
            self.replay.mark_reset()
        if self.context_buffer is not None:
            self.reset_context(self.context_buffer.shape[0])

    def reset_context(self, batch: int):
        # a batch-size change invalidates every per-body tensor the agency path
        # holds (review R10): banked decoys and the pay EMA would crash the next event.
        # review 2026-09-06: the EMA is nulled ONLY on an actual batch-size change -
        # the unconditional null wiped the value runtime_state.pt had just restored
        # (every resume's first step lands here), cold-starting agency payments each
        # segment, the exact M16b class the save exists to prevent.
        self._tenure_bank.clear()
        self._tenure_bank_el.clear()
        if (self._agency_pay_ema is not None
                and self._agency_pay_ema.shape[0] != batch):
            self._agency_pay_ema = None
        self._tenure_pending = None
        self._tenure_prev_order = None
        self._tenure_prev_order_id = None
        self.context_buffer = torch.zeros(batch, (self.ctx_len - 1) * self.n_mod, self.d_model, device=self.device)
        if getattr(self,'self_history',None) is not None:
            self.self_history.reset()
        if getattr(self,'memory_history',None) is not None:
            self.memory_history.reset()
            self._memory_state = self._memory_bias_current = self._nf_memory = None
            self.memory_bias_history.clear()
        self._lag_lat = self._lag_hs = self._lag_wg = self._lag_last = None
        # a batch-size change invalidates every per-body explainer stash too
        self.last_context = self.last_tokens_fed = None
        self.last_predictions = self.last_wellness_grounded = None
        self.last_voice = None
        self._nf_tokens = self._nf_ctx = self._nf_raw = self._nf_mask = None
        self._nf_pred = None
        self._nf_ach = None
        self._nf_handvis = None
        self._td_err_ema = None
        self._td_err_slow = None
        self._prev_mfa = None
        self._last_smooth_prev = None
        self._last_plan_inputs = None
        # review M5: a batch-size change invalidates the ring's per-body wellness and
        # any plan aimed at it - mixed [B_old]/[B_new] entries crash the next
        # torch.stack in goal selection (the R10 crash class, on the path R10 missed)
        self.goal_ring.clear()
        self._goal_ring_pose.clear()
        self._hand_previous_joints = None
        if getattr(self, "consequence_lesson", None) is not None:
            self.consequence_lesson.reset()
        self._goal_ring_w0.clear()
        self._goal_ring_cq.clear()
        self._goal_ring_drive.clear()
        self._goal_ring_ch.clear()
        self._rank_detrend = {}   # the ring's running averages belong to the same history
        self._gaze_hold = None
        self._gl_clear()          # held tensors carry the old B too
        self.last_intention = None
        if self.movement is not None:
            self.movement.reset()
        self.last_vision_token = None
        self.enc_history.clear()
        self.enc_history_raw.clear()
        self.valid_history.clear()
        self.motor_history.clear()
        self.ctx_history.clear()
        self.ach_history.clear()
        self._forget_efference_and_causal()

    def _forget_efference_and_causal(self):
        """Drop every efference and causal-chain quantity that belongs to one body history.

        Called by reset_transition_state and reset_context, the two places a body's history
        ends (a teleport, a rebuild, a batch-size change, every resume). The window sums are
        kept: they are a count of what was measured, not a pairing across the boundary.
        _transition_epoch rises by one, and the trainer's recorder refuses to pair any
        prediction made under an older epoch with a moment observed under this one."""
        self._eff_hist.clear()
        self._eff_eye_prev = None
        self._eff_tenure = None
        self._eff_prev_order_id = None
        self._eff_prev_order = None
        self.last_efference_tenure = None
        self.last_plan_record = None
        self._hand_chosen_plan = None
        self.newest_plan_record = None
        self._transition_epoch = int(self._transition_epoch) + 1
        # controllability's order in progress spans the boundary too (its averages are kept);
        # getattr: this runs from reset_context inside the first step of a fresh build as well
        if getattr(self, "wellness_channels", None) is not None:
            self.wellness_channels.forget_pending()
        self._eff_lp = {}
        # the contact head's per-body history (DESIGN D10): the previous moment's patch states
        # and the habituation memory belong to the body that just ended; the base rates and the
        # latent's centre are statistics over many histories and are kept
        self._contact_prev_state = None
        self._contact_prev_valid = None
        self._contact_habit = None
        self._contact_value_state = None
        # the motor synergy's order in progress and the pose it began at belong to the body that
        # just ended (its ring of graded samples and babble's sigma are kept: each sample is a
        # finished change of a body that existed), and so does the order the smoother blends with
        self._syn_pending = None
        self._syn_prev_ach = None
        self._syn_prev_order_id = None
        self._syn_prev_order = None
        self._syn_execution = None
        self._syn_latched_source = "unknown"
        self._smooth_exec_prev = None

    def _batch_frames(self, frames: Sequence) -> (dict, Dict[str, str], bool, dict):
        """Mirror of lem.live_test.build_frame_from_tokenized for a batch of BridgeFrames."""
        d = self.device
        B = len(frames)
        frame: dict = {}
        mask: Dict[str, str] = {}
        tfs = [f.tokenized for f in frames]
        bits = tfs[0].metadata.modality_valid_mask
        extras = {}
        has_vision = "vision" in self.active and all(f.vision_patches is not None for f in frames)
        if "vision" in self.active:
            if has_vision:
                def _stack(items):
                    if torch.is_tensor(items[0]):
                        return torch.stack([t.to(d, non_blocking=True) for t in items])
                    return torch.from_numpy(np.stack(items)).to(d, non_blocking=True)
                p = _stack([f.vision_patches for f in frames])
                if p.dim() == 4:                     # [B, N, 16, 16, 3] from the model tokenizer -> [B, N, 768]
                    p = p.reshape(p.shape[0], p.shape[1], -1)
                elif p.dim() == 5:
                    p = p.reshape(p.shape[0], p.shape[1], -1)
                a = _stack([f.vision_angles for f in frames])
                frame["vision"] = {"patches": p.float(), "angular_pos": a.float()}
                mask["vision"] = "real"
            else:
                mask["vision"] = "stale"
        if "proprioception" in self.active and (bits & MOD_PROPRIOCEPTION):
            frame["proprioception"] = torch.stack([t.proprio.reshape(-1) for t in tfs]).to(d)
            mask["proprioception"] = "real"
        if "motor" in self.active and (bits & MOD_MOTOR):
            frame["motor"] = torch.stack([t.motor_handles for t in tfs]).to(d)
            mask["motor"] = "real"
            extras["capability"] = torch.stack([t.motor_capability for t in tfs]).to(d)
        if "tactile" in self.active and (bits & MOD_TOUCH):
            from .skin_contract import frame_skin_features, batch_skin_features
            frame["tactile"] = torch.stack(
                [touch_point_cloud(frame_skin_features(f2), getattr(f2, "touch_pos", None),
                                   getattr(f2, "touch_self", None),
                                   getattr(f2, "touch_contact", None))
                 for f2, t in zip(frames, tfs)]).to(d)
            mask["tactile"] = "real"
            # Wellness sees ALL measured patches, before the encoder's contact limit.
            all_features = batch_skin_features(frames, d)
            gains = torch.ones(*all_features.shape[:2], 2, device=d)
            for i, f2 in enumerate(frames):
                g = getattr(f2, "touch_sensitivity", None)
                if g is not None:
                    gains[i, :len(g)] = torch.as_tensor(g, device=d)
            extras["raw_tactile"] = {"features": all_features, "sensitivity": gains}
            if self.contact_profile in ('mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'):
                history = torch.zeros(*all_features.shape[:2],3,device=d)
                from lem.skin_mechanics import MECHANICAL_CONTRACT
                if self.contact_profile in ('mechanical_v3','mechanical_v4'):
                    from .contact_footprint import CONTRACT as MECHANICAL_CONTRACT
                for i,f2 in enumerate(frames):
                    record = (f2.meta or {}).get('mechanical_contact')
                    if not record or record.get('contract') != MECHANICAL_CONTRACT:
                        marker=(f2.meta or {}).get('contact_profile_transition',{})
                        from lem.skin_mechanics import MECHANICAL_CONTRACT as LEGACY_MECHANICAL_CONTRACT
                        # The one frame at a profile switch still carries the OLD
                        # contract. Accept it only when the episode's own marker
                        # names this exact switch - comparing to the live profile,
                        # not to a literal, so v4 is admitted on the same terms
                        # v3 always was and neither can admit the other's frame.
                        boundary=(self.contact_profile in ('mechanical_v3','mechanical_v4') and record
                            and record.get('contract')==LEGACY_MECHANICAL_CONTRACT
                            and marker.get('version')=='virtual_skin_boundary_v1'
                            and marker.get('source') in ('mechanical_v2','mechanical_v2_local')
                            and marker.get('target')==self.contact_profile
                            and marker.get('after_time')==f2.sim_time)
                        if not boundary:raise ValueError('mechanical contact history is absent or has the wrong contract')
                    values = torch.as_tensor(record['history'],device=d,dtype=torch.float32)
                    history[i,:len(values)] = values
                extras['raw_tactile'].update(profile=self.contact_profile,history=history)
        if "voice" in self.active and (bits & MOD_VOICE) and tfs[0].voice is not None:
            frame["voice"] = torch.stack([t.voice for t in tfs]).to(d)
            mask["voice"] = "real"
        if ("audio" in self.active
                and any(getattr(t, "audio_spectral", None) is not None for t in tfs)):
            # crib hearing (gaze plan lane C): the env synthesizes the avatar's own
            # 256-bin stereo contract into the frame bytes; spatial = (ILD, ITD) norms.
            # A body whose frame carries no audio this step hears SILENCE (zeros = the
            # -80 dB floor), never crashes the batch: bodies reset independently and a
            # transient env guard can drop one frame's audio (measured: body 6, step 45
            # of the first H67 launch).
            _zs = torch.zeros(256, 2)
            _zm = torch.zeros(5)
            frame["audio"] = {
                "spectral": torch.stack([t.audio_spectral if getattr(t, "audio_spectral", None) is not None
                                         else _zs for t in tfs]).to(d),
                "spatial": torch.stack([(t.audio_metadata if getattr(t, "audio_metadata", None) is not None
                                         else _zm)[:2] for t in tfs]).to(d)}
            mask["audio"] = "real"
        if "vestibular" in self.active:
            # THE VESTIBULAR SENSE (DESIGN.md D4): the 29-number vector every frame carries
            # (BridgeFrame.vestibular), clipped to the measured input bound, its 28 fields without
            # the validity flag centred and scaled by the running mean and spread (a field the
            # frame does not carry reads as the mean). Real only when EVERY body's frame is a
            # real read; the first frame after resuming an older checkpoint is the saved payload
            # padded from MIMo's six numbers (valid 0), and a whole batch of those is "stale":
            # still encoded and fed forward, graded by no lesson, stored by replay as not valid
            # (mimo_bridge/vestibular_sense.py). The statistics move in step(), on training
            # steps only.
            if getattr(self, "vestibular_norm", None) is None:
                from .vestibular_sense import VestibularNormaliser
                self.vestibular_norm = VestibularNormaliser()
            _vv, _vok, _ = self.vestibular_norm.batch(frames)
            frame["vestibular"] = torch.from_numpy(_vv).to(d)
            mask["vestibular"] = "real" if bool(_vok.all()) else "stale"
        # gaze target (radians) from the frame header, like live_test
        extras["gaze_target"] = torch.tensor([[math.radians(t.metadata.gaze_azimuth_deg),
                                               math.radians(t.metadata.gaze_elevation_deg)] for t in tfs],
                                             dtype=torch.float32, device=d)
        return frame, mask, has_vision, extras

    # ------------------------------------------------------------------ the step
    def step(self, frames: Sequence, dev_limits: dict, train: bool = True, step_index: Optional[int] = None,
             random_actions: bool = False, executed_action: Optional[np.ndarray] = None,
             executed_order_id: Optional[int] = None,
             executed_ctrl_gaze: Optional[np.ndarray] = None) -> StepResult:
        """executed_action: [B, n_handles, 9] the order that was ACTUALLY in force while the
        body produced `frames` - after developmental limits, exploration noise and the
        motor hold. TransitionV1's action field.

        It is not motor_out. motor_out is this step's fresh proposal, which under
        motor_hold=5 is the executed order on one step in five, and even then without the
        limits and noise that were applied to it before the body saw it. Training the
        forward model, the imagination chain or the body schema on motor_out teaches
        `state -> next state` while labelling it `state + action -> next state`.

        None means the caller has not adopted the ledger yet; the old behaviour is kept
        so single-env probes and tests keep working, and the run manifest records which
        path was used.

        executed_ctrl_gaze: [B, 2] the model's gaze command over the same transition
        (azimuth, elevation in radians: res.gaze_out as the world received it, before its saccade
        generator and gaze reflex; that is _gaze_from_handle of the controller's eye row, or of the
        planner's or the self-choice's chosen row when they author the order: the deploy branch's
        third review, finding B). The ledger's eye row is written from the gaze the world SENT; the
        replay ring keeps this beside it (ReplayMemory.ctrl_gaze), so the row splits into the
        command's part, which keeps its numbers at an F1 switch, and the world's. No lesson reads its
        numbers; under --eye-row-convention physical its presence marks the moments replay studies
        (_eye_contract_bit: the fourth review's fix round)."""
        t0 = time.perf_counter()
        B = len(frames)
        # the comparison records' readings of this model step start empty (record b and c)
        self.clip_norms = None
        self.gaze_copy_now = None
        self._hand_chosen_plan = None
        if self.hand_target_mode != 'world' and self.smooth_source == 'executed':
            goals = [f.meta.get('hand_joint_goal') for f in frames]
            if all(g is not None for g in goals):
                self._hand_previous_joints = torch.as_tensor(np.stack(goals), device=self.device, dtype=torch.float32)
        # this step's planning event only (review MINOR 8: it outlived its step until the next
        # planning step, and a reader without the recorder's step check took it for this step's)
        self.last_plan_record = None
        exec_t = None
        if executed_action is not None:
            exec_t = torch.as_tensor(np.asarray(executed_action, dtype=np.float32),
                                     device=self.device)
            if exec_t.ndim == 2:
                exec_t = exec_t.unsqueeze(0)
            assert exec_t.shape[0] == B, f"executed_action batch {exec_t.shape[0]} != {B}"
        self._exec_seen = self._exec_seen + 1 if exec_t is not None else self._exec_seen
        if self.context_buffer is None or self.context_buffer.shape[0] != B:
            self.reset_context(B)
        self._hand_current_angles = self._hand_current_velocities = None
        if self.hand_goals != 'off':
            readings = [(f.meta.get('hand_joint_angles'), f.meta.get('hand_joint_velocities'))
                        for f in frames]
            if all(q is not None and v is not None for q, v in readings):
                angles, velocities = (np.stack([pair[i] for pair in readings]) for i in (0, 1))
                expected = (B, len(self._hand_geometry['joints']))
                if (angles.shape != expected or velocities.shape != expected
                        or not np.isfinite(angles).all() or not np.isfinite(velocities).all()):
                    raise ValueError('Hand goals require finite native joint angles and velocities')
                self._hand_current_angles = torch.as_tensor(angles, device=self.device, dtype=torch.float64)
                self._hand_current_velocities = torch.as_tensor(velocities, device=self.device, dtype=torch.float64)
                self._hand_native_seen = True
            elif (self._hand_native_seen or any(v is not None for q, v in readings)
                  or (any(q is not None for q, v in readings) and not all(q is not None for q, v in readings))):
                raise ValueError('Hand joint readout disappeared or is incomplete across bodies')
        if self.smooth_source == "executed":
            # --smooth-source executed: the order the body was really given over the interval that
            # just ended is what a new order is smoothed against (and what the planner imagines
            # smoothing against); None after a reset or on the first moment
            self._smooth_exec_prev = (None if exec_t is None
                                      else exec_t.detach().float().clone())
        step = self.step_count if step_index is None else step_index
        if self.liveness is not None:
            # the liveness report samples this step or not (lem/model/liveness.py SAMPLE_EVERY)
            self.liveness.begin_step(self, step)
        self._last_dev = dict(dev_limits)
        # threaded to the intention objective so it never optimises toward noise
        self._last_source_mask = None
        frame, source_mask, has_vision, extras = self._batch_frames(frames)
        if getattr(self, "vestibular_input_path", None) is not None:
            # --vestibular-input-path: the path's weight is held at its cap on training steps only (a step that does
            # not train never moves a weight)
            self.vestibular_input_path.hold_enabled = bool(train)
        if train and getattr(self, "vestibular_norm", None) is not None:
            # the vestibular input's running mean and spread take this moment's real reads
            # (after it was centred with the statistics before it)
            self.vestibular_norm.commit()
        _real_audio = frame.get("audio") if source_mask.get("audio") == "real" else None
        source_mask.update(self.placeholders.fill_frame(frame, batch_size=B))
        if _real_audio is not None:
            # real crib hearing outranks any configured placeholder generator
            frame["audio"] = _real_audio
            source_mask["audio"] = "real"
        # the audio placeholder yields one tensor [B,256,2]; the audio encoder wants the
        # tokenizer's dict form {"spectral": [B,256,2], "spatial": [B,2]} (model-side gap)
        if "audio" in frame and torch.is_tensor(frame["audio"]):
            frame["audio"] = {"spectral": frame["audio"], "spatial": torch.zeros(B, 2, device=self.device)}
        for mod in self.active:
            source_mask.setdefault(mod, "real")
        if not has_vision and "vision" in self.active and self.last_vision_token is not None:
            source_mask["vision"] = "stale"
        timings = {}
        d = self.device
        handle_state = frame.get("motor")
        if handle_state is None:
            handle_state = torch.zeros(B, len(self.handle_names), 10, device=d)
        # handle_state stays the 10-wide ACHIEVED tensor: the motor controller, the motor
        # loss target, the replay echo and the reach probe all read that layout. Only the
        # ENCODER input is widened.
        #
        # The command columns hold ONE thing: the action that carries this state to the
        # next one. The live pass does not know it - the model has not chosen it yet - so
        # it writes zeros, a defined "not yet known".
        #
        # Measured 2026-08-25: writing the ARRIVAL action here instead, while the delayed
        # next-frame lesson wrote the OUTGOING action into the same columns, gave those
        # nine columns two contradictory meanings and training diverged - proprioception
        # error rose 0.169 -> 0.740 over 900 steps and vision 0.313 -> 0.685, reproducibly,
        # from a checkpoint that converges to 0.001 with the conflict removed. One slot,
        # one meaning.
        #
        # The action reaches the model where the prediction it conditions actually happens:
        # the delayed next-frame lesson swaps it into the cached previous frame, and replay
        # rebuilds the motor token from [achieved, outgoing action].
        # BACK-FILL THE PREVIOUS CONTEXT SLOT (B9 fix, 2026-08-26). The action arriving
        # now is the OUTGOING action of the PREVIOUS moment - one slot, one meaning, so
        # it belongs in the previous moment's motor slot, not this one's. Until now the
        # live context carried [achieved, ZEROS] in every one of its 63 motor slots
        # while replay_update rebuilt the same slots from [achieved, executed]: two
        # different context distributions training one set of weights. Measured on H31's
        # final checkpoint, same current token, only the context motor slots changed:
        # zero-command context predicted at 0.851/0.662/0.231 (vision/tactile/proprio),
        # real-action context at 0.251/0.009/0.015 - 3.4x/71x/16x better. The next-frame
        # lesson, the one place the action trains the model on-policy, always ran in the
        # context replay never produces. This is the same rebuild replay does, done live.
        if (exec_t is not None and self.context_buffer is not None
                and self.last_achieved is not None
                and self.last_achieved.shape[0] == B and "motor" in self.active):
            with torch.no_grad():
                mi = self.active.index("motor")
                x_bf = torch.cat([self.last_achieved[..., :MOTOR_ACHIEVED_F],
                                  command_for_encoder(exec_t)], dim=-1)
                z_bf = self.encoder_registry.encode({"motor": x_bf},
                                                    {"motor": "real"})["motor"].reshape(B, self.d_model)
                _fm = getattr(self.forward_model, "_orig_mod", self.forward_model)
                if hasattr(_fm, "_refresh_embedding_cache"):
                    _fm._refresh_embedding_cache()
                _emb = getattr(_fm, "_mod_emb_expanded", None)
                if _emb is not None:
                    z_bf = z_bf + _emb.reshape(-1, self.d_model)[mi].to(z_bf.dtype)
                # newest frame occupies the last n_mod slots in modality order
                self.context_buffer[:, -(self.n_mod - mi), :] = z_bf
        if "motor" in frame and frame["motor"].shape[-1] == MOTOR_ACHIEVED_F:
            frame["motor"] = torch.cat(
                [frame["motor"], torch.zeros(B, frame["motor"].shape[1], MOTOR_COMMAND_F,
                                             device=frame["motor"].device,
                                             dtype=frame["motor"].dtype)], dim=-1)
        mv = None
        if self.movement is not None and "proprioception" in frame:
            # The schema is not merely untrained - its statistic has no lag bank, no sign,
            # and no passive or actuator-disabled comparator, so what it measures is not
            # self-causation. With zero evidence it still emits membership 0.076 and
            # coherence 0.848, which reads as a confident "not self". While
            # body_schema_updates is off, the channel is BYPASSED rather than given a
            # neutral number: substituting 0.5 would be inventing evidence too.
            if not self.body_schema_updates:
                sf = None                      # movement drive runs unmodulated
            else:
                try:
                    sf = float(torch.stack([sc.self_membership().mean()
                                            for sc in self._schemas[:B]]).mean())
                except Exception:
                    sf = float(self.body_schema.coherence_score())
            mv = self.movement.update(frame["proprioception"].view(B, -1, 12),
                                      self_factor=1.0 if sf is None else sf)
            if self.movement_sensor_mod:
                # Stage 2: stillness numbs the body senses (floored), movement sharpens them
                g = mv["gain"]
                if "tactile" in frame:
                    if frame["tactile"].shape[-1] == TOUCH_TOKEN_F:
                        t2 = frame["tactile"].clone()
                        # gain on the SENSE, not the geometry: scaling a coordinate
                        # would move the contact, not numb it
                        t2[..., 3:11] = t2[..., 3:11] * g.view(B, 1, 1)
                        frame["tactile"] = t2
                    else:
                        frame["tactile"] = frame["tactile"] * g.view(B, 1, 1)
                pro = frame["proprioception"].view(B, -1, 12).clone()
                pro[:, :, 4:11] = pro[:, :, 4:11] * g.view(B, 1, 1)   # pos/vel/fatigue only:
                frame["proprioception"] = pro.view(B, -1)             # never scale quat/index

        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.dtype):
            self_spatial=self_spatial_valid=None
            if self.self_prediction_mode=='grounded_residual':
                from lem.model.spatial import encode_spatial
                tokens,self_spatial,self_spatial_valid=encode_spatial(
                    self.encoder_registry,frame,source_mask,self.forward_model.modality_embedding)
                # The additional consequence lesson cannot rewrite its own
                # sensory targets by collapsing the established visual encoder.
                self_spatial=self_spatial.detach().float()
            else:
                tokens = self.encoder_registry.encode(fit_tactile(frame), source_mask)
            if "vision" in self.active:
                if source_mask.get("vision") == "stale" and self.last_vision_token is not None:
                    tokens["vision"] = self.last_vision_token
                elif "vision" in tokens:
                    self.last_vision_token = tokens["vision"].detach()
            zero = torch.zeros(B, 1, self.d_model, device=d)
            # A SENSE WITH NO ENCODER OUTPUT IS NOT MEASURED REALITY. The line below
            # substitutes a zeros token for it, and the source mask above already
            # defaulted it to "real", so it would join VICReg's pool: a zeros token has
            # std 0, so its variance term is the constant relu(1.0 - 0) = 1.0 with no
            # gradient, and compute_vicreg averages over senses - measured with one such
            # sense among three, vision's anti-collapse gradient fell to 0.53x and
            # tactile's to 0.57x. Not reachable on the MIMo producer (the frame encoder
            # always sets proprioception|motor|touch, missing vision is labelled "stale",
            # audio and intermind are placeholder-filled "synthetic"), so this list is
            # empty in every run here and nothing changes numerically.
            self._no_encoder_output = [m for m in self.active if m not in tokens]
            tokens = {m: tokens.get(m, zero) for m in self.active}
            # No side-channel gaze injection any more: the eyes are handle 23, so the
            # command is IN the motor modality and the achieved eye pose is IN
            # proprioception and the motor echo, the same as every limb.
            # Act 3 (gated): the hill. The forward model predicted THIS moment a step ago;
            # where a sense's report deviates from that expectation, a small learned gate
            # blends the expectation in (bounded 0.5). Persistent surprise forces raw
            # reality back through (gate zeroed while a sense's error EMA stays high).
            # review P4/P5 (2026-09-02): the next-frame forward runs HERE, before the
            # gate and wellness, so _nf_pred is the prediction of THIS frame during the
            # step that consumes it. It used to be produced in the loss section and
            # consumed one step late: wellness paired the previous frame's prediction
            # with the current frame (prediction error PLUS one frame of world change),
            # and the gate corrected the current percept with the previous frame's
            # expectation. The loss section reuses this stashed forward.
            nf_live = None
            # the latent of S_t-1 + A_t-1 the delayed lesson computes, WITH gradient: the
            # efference heads read it (and only when --efference-heads is on)
            nf_latent = None
            goal_variants = None      # --persistent-memory-repair's goal reading (below)
            if (self.persistent_memory is not None and self.persistent_memory.repair != 'off'
                    and not self.persistent_memory.repair_switched_py
                    and self.memory_history.previous is not None):
                # --persistent-memory-repair: the switch comes before anything this step reads the memory.
                # Its fold rewrites the readout's weight in place, and a step whose graph already held that
                # weight (the delayed lesson's bias below) would find it modified at backward. The snapshot is
                # the newest lived moment (the one the delayed lesson re-encodes as _nf_memory).
                self._persistent_memory_switch(self.memory_history.snapshot())
            if getattr(self, "motor_input_fold", None) is not None:
                # --motor-input-fold: the switched memory's constant leaves the controller's input before anything
                # this step runs the controller (lem/model/motor_input_fold.py, WHEN THE FOLD HAPPENS)
                self.motor_input_fold.at_step_start(self, train)
            nf_memory_state = nf_memory_bias = None
            if self.persistent_memory is not None and self._nf_memory is not None:
                nf_memory_state,_ = self.persistent_memory.encode(**self._nf_memory)
                nf_memory_bias = self.persistent_memory.bias(nf_memory_state, site="delayed")
            if self.self_state_mode == 'perceive' and self.self_history.inputs is not None:
                with torch.autocast(device_type=self.device.type,enabled=False):
                    old_self_bias = self.self_state.perceive(**self.self_history.inputs)['bias']
                nf_memory_bias = old_self_bias if nf_memory_bias is None else nf_memory_bias+old_self_bias
            if self.next_frame_loss and self._nf_raw is not None:
                nf_raw = self._nf_raw
                if exec_t is not None and torch.is_tensor(nf_raw.get("motor")):
                    mt = nf_raw["motor"]
                    if mt.shape[-1] == MOTOR_INPUT_F and mt.shape[0] == B:
                        nf_raw = dict(nf_raw)
                        nf_raw["motor"] = torch.cat(
                            [mt[..., :MOTOR_ACHIEVED_F],
                             command_for_encoder(exec_t.to(mt.dtype))], dim=-1)
                        self._nf_aligned = self._nf_aligned + 1
                        if getattr(self, "motor_asked_change", None) is not None:
                            # --motor-asked-change: the window's gauges read this call's commanded rows
                            # (lem/model/motor_asked_change.py arm_gauge); nothing is computed here
                            self.motor_asked_change.arm_gauge()
                _vip = getattr(self, "vestibular_input_path", None)
                if _vip is None:
                    nf_in = self.encoder_registry.encode(fit_tactile(nf_raw), self._nf_mask)
                else:
                    # --vestibular-input-path: the previous row passes through the path with its weight detached
                    # (the same numbers): the lesson's `+ current` residual adds the token's squared change between
                    # two model steps to the loss, and its gradient on the path's weight would shrink every column
                    # whose input field changes between model steps (lem/model/vestibular_input_path.py, GRADIENT)
                    with _vip.frozen():
                        nf_in = self.encoder_registry.encode(fit_tactile(nf_raw), self._nf_mask)
                    if _vip.delayed_detached and torch.is_tensor(nf_in.get("vestibular")):
                        # --vestibular-input-path on: the previous row's vestibular token enters the
                        # delayed lesson without gradient (the same numbers), so that lesson trains neither the
                        # encoder's own layers nor the path, and every lesson that trains one part trains the other
                        nf_in = dict(nf_in)
                        nf_in["vestibular"] = nf_in["vestibular"].detach()
                nf_in = {m: (nf_in[m] if (m in nf_in and self._nf_mask.get(m) == "real")
                             else self._nf_tokens[m]) for m in self.active}
                _l, nf_last, _c = self.forward_model(nf_in, self._nf_ctx,
                    **({} if nf_memory_bias is None else {'memory_bias':nf_memory_bias}))
                if getattr(self, "heads_all_slot_path", None) is not None:
                    # --heads-all-slot-path: the window's gauges read this call's rows (arm_gauge; nothing is
                    # computed here), and the head_read and lesson-gradient gauges keep them with their current
                    # tokens and with the lesson's targets (this model step's raw tokens of the senses that are
                    # real now and were real at the model step predicted from)
                    self.heads_all_slot_path.arm_gauge()
                    self.heads_all_slot_path_gauge.append((
                        nf_last.detach(),
                        {m: (v.squeeze(1) if v.dim() == 3 else v).detach() for m, v in nf_in.items()},
                        {m: tokens[m].squeeze(1).detach() for m in self.heads_all_slot_path.head_modalities
                         if m in tokens and source_mask.get(m) == "real" and self._nf_mask.get(m) == "real"
                         and tokens[m].shape[0] == nf_last.shape[0]}))
                nf_preds = self.prediction_heads(
                    nf_last,
                    current={m: v.squeeze(1) if v.dim() == 3 else v for m, v in nf_in.items()},
                    who="lem_model delayed next-frame lesson")
                _nf_preds_before_copy = nf_preds
                if self.efference is not None:
                    # the gaze copy into vision's next-moment prediction (lem/model/efference.py,
                    # DESIGN D5 as amended 2026-09-23); zero at the start, so nothing moves on switch
                    from lem.model.efference import lesson_gaze_copy
                    nf_preds = lesson_gaze_copy(
                        self, nf_preds, nf_last, exec_t,
                        (tokens["vision"].squeeze(1).detach() if "vision" in tokens
                         and source_mask.get("vision") == "real"
                         and self._nf_mask.get("vision") == "real" else None), B)
                from . import comparison_records as _CR
                if _CR.RECORDING:
                    # record b: vision's prediction without the gaze copy's term and with it (references to tensors
                    # the step already holds, detached: no copy, no arithmetic); lesson_gaze_copy returns a new
                    # dict when it adds the copy and the same dict when it adds nothing. Behind the records' guard:
                    # a failure is counted (comparison_records.record_failed), never raised
                    try:
                        if "vision" in _nf_preds_before_copy and "vision" in nf_preds:
                            self.gaze_copy_now = dict(without=_nf_preds_before_copy["vision"].detach(),
                                                      with_copy=nf_preds["vision"].detach(),
                                                      applied=nf_preds is not _nf_preds_before_copy)
                    except Exception as _ex:
                        self.gaze_copy_now = None
                        _CR.record_failed(self, "gaze_copy", _ex)
                nf_live = (nf_in, nf_preds)
                nf_latent = nf_last
                self._nf_pred = {m: v.detach() for m, v in nf_preds.items()}
                if (nf_memory_state is not None and self.persistent_memory is not None
                        and self.persistent_memory.repaired):
                    # --persistent-memory-repair, THE GOAL READING: the same predictions with the memory's
                    # varying part (the readout weight times the centred state) taken out of the latent, and
                    # with another body's and a second other body's in its place (the information check and
                    # its control); the constant, the tokens and the context stay. Graded beside the lesson.
                    goal_variants = self._memory_goal_variants(
                        nf_in, nf_last, nf_memory_state, exec_t, B)
            tokens_raw = {m: tokens[m] for m in tokens}
            self_state_loss = self_state_bias = None
            if self.self_state is not None:
                from .self_state_integration import observe as observe_self
                self_state_loss,self_state_bias = observe_self(self,tokens_raw,frames,exec_t,
                    spatial=self_spatial,spatial_valid=self_spatial_valid)
            memory_snapshot = current_needs = None
            self._memory_bias_current = None
            if self.persistent_memory is not None:
                from .memory_integration import observe
                self._memory_state,self._memory_bias_current,memory_snapshot,current_needs = observe(
                    self,tokens_raw,source_mask,extras,frames,exec_t)
                self.last_memory = dict(events=len(self.memory_history.events),
                    capacity=int(self.memory_history.capacity),
                    interval_seconds=float(self.memory_history.interval_s),
                    window_seconds=float(self.memory_history.capacity * self.memory_history.interval_s),
                    encoded_positions=int(memory_snapshot['events'].shape[1]),
                    oldest_retained_age_seconds=float((memory_snapshot['now']-memory_snapshot['times'][:,0]).mean()),
                    has_recurrent_prefix=self.memory_history.carry is not None,
                    bias_rms=float(self._memory_bias_current.detach().square().mean().sqrt()))
                # (--persistent-memory-repair: the repaired memory adds its gauges up on the device at every
                # lived call; the stats record reads the report window's means once: memory_report)
            if self_state_bias is not None and self.self_state_mode == 'perceive':
                self._memory_bias_current = (self_state_bias if self._memory_bias_current is None
                                             else self._memory_bias_current+self_state_bias)
            if self.top_down_gate and self._nf_pred is not None:
                if self._td_err_ema is None or self._td_err_ema.shape[0] != B:
                    self._td_err_ema = torch.zeros(B, len(self.pred_mods), device=d)
                if self._td_err_slow is None or self._td_err_slow.shape[0] != B:
                    self._td_err_slow = torch.zeros(B, len(self.pred_mods), device=d)
                self.last_gate = {}
                for mi, m in enumerate(self.pred_mods):
                    if m not in self._nf_pred or source_mask.get(m) != "real":
                        continue
                    zp = self._nf_pred[m]
                    if zp.shape[0] != B:
                        continue
                    err = (tokens[m].squeeze(1) - zp).pow(2).mean(-1, keepdim=True)     # [B,1]
                    with torch.no_grad():
                        _e = err.squeeze(-1)
                        self._td_err_ema[:, mi] = 0.95 * self._td_err_ema[:, mi] + 0.05 * _e
                        # slow baseline seeded from the first sample, then 0.999
                        _seed = self._td_err_slow[:, mi] == 0
                        self._td_err_slow[:, mi] = torch.where(
                            _seed, _e, 0.999 * self._td_err_slow[:, mi] + 0.001 * _e)
                    g = 0.5 * torch.sigmoid(self.td_gates[m](
                        torch.cat([err.detach(), torch.ones_like(err)], dim=-1)))        # [B,1]
                    # review M23: the old reference was 4x the CURRENT batch error -
                    # under SUSTAINED surprise fast and current move together, the
                    # ratio sits near 1, and the gate stayed open exactly when its
                    # comment said it must close. The reference is now a slow (0.999)
                    # baseline; K=2.0 measured: fast(0.95)/slow(0.999) never exceeded
                    # 1.48 in 1,035 stats windows across H59/H60/H61.
                    g = g * (self._td_err_ema[:, mi:mi+1]
                             <= 2.0 * self._td_err_slow[:, mi:mi+1].clamp(min=1e-9)).float()
                    self.last_gate[m] = float(g.detach().mean())   # review T10: log the blend
                    tokens[m] = tokens[m] + (g.unsqueeze(-1) * (zp.unsqueeze(1) - tokens[m]))
            timings["encode"] = (time.perf_counter() - t0) * 1000
            # review M9: the collapse watch exists to catch encoder collapse, and the
            # causal probes read last_tokens as measured reality - both must see the
            # RAW encodings. A collapsed encoder plus a varying blended-in prediction
            # still shows temporal std, masking exactly what the ring watches for.
            if self.step_count % 5 == 0:
                for m in self.active:
                    self.token_ring[m].append(tokens_raw[m].detach().float().squeeze(1)[0].cpu())
                    # the same sample for every body (token_temporal_std_bodies); a companion saved before a
                    # sense existed has no deque for it
                    self._token_ring_bodies.setdefault(m, deque(maxlen=200)).append(
                        tokens_raw[m].detach().float().squeeze(1).cpu())
            self.last_tokens = {m: tokens_raw[m].squeeze(1).detach() for m in tokens_raw}
            self._last_source_mask = dict(source_mask)
            self.last_achieved = handle_state.detach()
            if self.replay is not None:
                # review M7: gated tokens carried up to g<=0.5 of the model's own
                # same-step prediction into every stored replay target
                from .limb_target_control import moment_contract   # finger | calf | chest | rows bits
                # the model's gaze command beside the ledger (motor fixes deploy, F1)
                _cg = self._ctrl_gaze_for_ring(executed_ctrl_gaze, exec_t, B)
                try:
                    _slots = getattr(self, "replay_slot_step", None)
                    if _slots is not None and len(_slots) == self.replay.size:
                        # the comparison records' map (record c): the slot this add writes holds model step `step`
                        _slots[self.replay.ptr % self.replay.size] = int(step)
                except Exception as _ex:            # the records' guard: counted, never raised
                    from . import comparison_records as _CR
                    _CR.record_failed(self, "replay_slot_map", _ex)
                self.replay.add(tokens_raw, handle_state, dev_limits,
                                source_mask=source_mask, executed=exec_t,
                                # the eye row's bit: under 'physical', a moment whose order arrives with the
                                # model's gaze command (the fourth review's fix round: replay studies only them)
                                action_contract=moment_contract(f.meta for f in frames) | self._eye_contract_bit(_cg),
                                **({} if self.self_state_mode != 'perceive' else {'self_state':self._self_percept['belief']}),
                                **({} if self._memory_state is None else {'memory_state':self._memory_state}),
                                **_cg)
            if self.imagination_enabled:
                self.memory_bias_history.append(None if self._memory_bias_current is None else self._memory_bias_current.detach())
                self.enc_history.append({m: tokens[m].squeeze(1).detach() for m in tokens})
                # review M8: targets must be raw encodings; grading the chain against
                # the gated blend is the same self-grading channel as M7, at weight 0.5
                self.enc_history_raw.append({m: tokens_raw[m].squeeze(1).detach()
                                             for m in tokens_raw})
                self.valid_history.append(dict(source_mask))
                self.ctx_history.append(None if self.context_buffer is None
                                        else self.context_buffer.detach().clone())
                self.ach_history.append(handle_state.detach().clone())
            if self.latent_regularizer is not None:
                # review M11: the penalty is applied to raw-trained predictions; the
                # reference statistics must come from the same (raw) distribution
                self.latent_regularizer.update(
                    {m: tokens_raw[m].squeeze(1).detach() for m in tokens_raw}, valid=source_mask)
            latents, last_step, ctx_update = self.forward_model(tokens, self.context_buffer,
                **({} if self._memory_bias_current is None else {'memory_bias':self._memory_bias_current}))
            # ONE PAIRING, ONE DICT - and that dict is tokens_raw. The residual base on
            # the first line and the target on the second must remain the SAME tensor:
            # predictions[m] - targets[m] is then exactly the head's own output, and `pe`
            # below is gate-independent (measured: max deviation 1.192e-07 in fp32, pe
            # identical to 9 decimal places across g = 0.00 / 0.25 / 0.50). Changing only
            # ONE of them would INTRODUCE gate dependence rather than remove it - the
            # difference would become -g*(prediction_of_now - raw).
            # Both moved from `tokens` to tokens_raw (2026-09-18, audit row P042) so the
            # live prediction heads read what every neighbouring lesson reads: next-frame,
            # Judge B, vicreg, replay, the imagination chain. Two things this closes:
            #   - with --residual-prediction off (its default), grading the heads against
            #     the gated blend is the M7/M8 self-grading channel: the error shrinks by
            #     (1 - g)^2 - measured 0.250 of the g=0 error at g=0.5 - with nothing
            #     learned. The only thing holding that channel shut was the loss_weight
            #     0.0 guard that --next-frame-loss installs.
            #   - `predictions` feed latent_regularizer.penalty below, whose reference
            #     statistics are collected from tokens_raw (M11). With a gated residual
            #     base that penalty's gradient reached td_gates - the M10/E3 defect class
            #     already removed from vicreg. Measured: sum|grad| on the td_gate Linear
            #     parameters 0.00598 with the gated base, exactly 0 with tokens_raw.
            # The acting path is untouched: last_step above still comes from the GATED
            # tokens, so the latents the motor controller reads do not move.
            predictions = self.prediction_heads(
                last_step,
                current={m: tokens_raw[m].squeeze(1) for m in tokens_raw},
                who="lem_model live")
            targets = {m: tokens_raw[m].squeeze(1).detach()
                       for m in self.pred_mods if m in tokens_raw}
            # review E5: the BASELINE guard left wellness._rom_limit at 1.0 through
            # every phase - the one wellness term with a gradient path to the motor
            # command was flat (gradient identically zero) for the whole neonate phase
            # of a fresh run. Measured: rom 0.05 scores 0.1819 with 83.1% of
            # coordinates in the boundary zone; stuck-1.0 scored 1.0000 with 0%.
            self.wellness.set_developmental_limits(dev_limits["rom_limit"])
            # AUDIT FIX: the motor_state formula divides |pos|/rom expecting ROM-space
            # coordinates; world positions made it a constant pull toward the world
            # origin. Body-relative displacement (handle minus body centroid) is the
            # quantity with a meaningful magnitude and a live gradient at every ROM.
            mc = handle_state[:, :, :9].detach().clone().float()
            mc[..., 0:3] = mc[..., 0:3] - mc[..., 0:3].mean(dim=1, keepdim=True)
            motor_commands = mc.requires_grad_(True)
            # Grounded prediction-error wellness was scored from `predictions` vs
            # `targets` - CURRENT-frame reconstruction. Spec 7.1 makes this score the
            # region where predictions SUCCEED, and the authoritative prediction is the
            # delayed causal lesson S_{t-1} + A_{t-1} -> S_t, not agreement with the
            # frame the model is looking at. The dashboard and curiosity were corrected
            # earlier; Judge A was still reading reconstruction.
            _wp, _wt = predictions, targets
            # WHICH PAIRING PRODUCED THIS STEP'S prediction_error, recorded rather than
            # left to be inferred. The authoritative pairing is the delayed causal
            # prediction (S_{t-1} + A_{t-1} -> S_t) against raw reality. On the first
            # step of a run, of a resume, and after every reset_transition_state /
            # reset_context, _nf_pred is None and the same-frame pair below is used
            # instead - and with residual_prediction on, prediction = head_output +
            # current token while target = current token, so the term evaluates
            # 1 - clamp(mean(head_output^2)): whether the heads predicted NO CHANGE, a
            # different quantity from prediction accuracy. Measured in this run's own
            # record: explain/step_0.json has causal_pe null yet prediction_error 0.98257.
            # Handing wellness nothing on those steps would change the wellness sum, the
            # wellness gradient and therefore the first command after every reset, so the
            # value is left alone and the source is published instead.
            self.last_pe_source = ("same-frame reconstruction pair: the delayed causal "
                                   "lesson has not run this step, so this scores whether "
                                   "the heads predicted no change")
            # review M20: _nf_pred present means the hoist ran THIS step; last_next_pe
            # is refreshed later the same step, so requiring it cost one post-reset
            # step of correct pairing
            if self._nf_pred is not None:
                _shared = [m for m in self._nf_pred
                           if m in tokens and source_mask.get(m) == "real"
                           and self._nf_pred[m].shape[0] == B
                           # vestibular only when predicted from a real moment (see nf_terms)
                           and (m != "vestibular" or (self._nf_mask or {}).get(m) == "real")]
                if _shared:
                    # review P4: _nf_pred is now the prediction of THIS frame; the
                    # target is the RAW encoding of this frame, not the gated blend
                    _wp = {m: self._nf_pred[m] for m in _shared}
                    _wt = {m: tokens_raw[m].squeeze(1).detach() for m in _shared}
                    self.last_pe_source = ("delayed causal prediction vs raw tokens")
            wellness_out = self.wellness(_wp, _wt, latents,
                                         source_mask=source_mask,
                                         # Judge B learns in the SAME space it is
                                         # scored in during planning: encoder tokens,
                                         # not transformer latents. review M6: RAW and
                                         # DETACHED - undetached gated tokens let Judge
                                         # B's alignment loss train the encoders and
                                         # td_gates toward wellness-predictability,
                                         # which nothing documents as intended, and
                                         # trained it on a different distribution than
                                         # planning scores it on.
                                         canonical_tokens={m: tokens_raw[m].squeeze(1).detach()
                                                           for m in tokens_raw},
                                         capability_features=(
                                             self._stash_capability(extras.get("capability"))),
                                         motor_commands=motor_commands,
                                         # None while the schema is unvalidated: its
                                         # 0.848 coherence is emitted with zero evidence
                                         # and would enter the wellness sum as if earned.
                                         # wellness falls back to its own coherence term.
                                         # WHICH QUANTITY, AND PER BODY. --wellness-coherence
                                         # picks it and last_coherence_source names it in the
                                         # snapshot. "extent" is the share of everything the
                                         # body map watches that it calls its own - its 26
                                         # handles AND any outside thing taken in - which is
                                         # what the project's goal asks wellness to reward;
                                         # "certainty" is how SURE the map is, which on run
                                         # H84 equalled 1 - 2 x the BODY's own share
                                         # (correlation -1.0000, largest gap 1.0e-05; against
                                         # extent, which now counts the outside things too,
                                         # 0.9991 and 0.0322) and so paid for the perceived self
                                         # SHRINKING while carrying 53 of every 100 parts of
                                         # wellness's movement. Each body now sends its own
                                         # number; the mean over bodies used to be shared by
                                         # all of them. With no body map running, wellness
                                         # falls back to the cosine likeness of prediction to
                                         # arrival, a third quantity, and the source says so.
                                         body_schema_coherence=(
                                             self._schema_coherence(B)
                                             if self.body_schema_updates else None),
                                         # the wellness formula reads pressure at column 0
                                         # and vib at 4..7: hand it the FEATURE columns of
                                         # the point-cloud layout, where they sit at 3..10
                                         raw_tactile=(extras["raw_tactile"] if "raw_tactile" in extras else
                                                      frame["tactile"][..., 3:11]
                                                      if "tactile" in frame
                                                      and frame["tactile"].shape[-1] == TOUCH_TOKEN_F
                                                      else frame.get("tactile")))
            self.last_coherence_source = ("body_schema_" + self.wellness_coherence
                                          if self.body_schema_updates else "prediction_cosine")
            # SELF-TOUCH AS A WELLNESS TERM (2026-09-21), with the gate that keeps it
            # empty until it is earned. An animal that cannot predict its own limbs is in
            # danger, and a person who loses feeling in a hand loses well-being with it, so
            # the body is paid for touch on its own parts - but only once the prediction of
            # that touch has beaten its shuffled-action control, and never for force or
            # duration, which is what resting one hand on the other would maximise.
            if 'tactile' in frame and frame['tactile'].shape[-1] == TOUCH_TOKEN_F:
                wellness_out['grounded']['self_touch'] = self._self_touch_wellness(
                    frame['tactile'], B)
            total_wellness = torch.cat(list(wellness_out["grounded"].values()), dim=-1).sum()
            try:
                wellness_grad = torch.autograd.grad(total_wellness, motor_commands, create_graph=True,
                                                    retain_graph=True, allow_unused=True)[0]
                if wellness_grad is None:
                    wellness_grad = torch.zeros_like(motor_commands)
            except RuntimeError:
                wellness_grad = torch.zeros_like(motor_commands)
            motor_out, motor_conf, motor_meta = self.motor_controller(latents, handle_state, wellness_grad=wellness_grad)
            # AUDIT FIX (critical): the lag caches were refreshed HERE, before their
            # consumers ran - so the "lagged" motor grading and gaze de-echo used the
            # CURRENT frame and both de-echoes were silently no-ops in every run. The
            # caches are read into locals now and refreshed only after the loss.
            prev_lat, prev_hs, prev_wg, prev_last = self._lag_lat, self._lag_hs, self._lag_wg, self._lag_last
            if motor_out.isnan().any() or motor_out.isinf().any():
                motor_out = torch.nan_to_num(motor_out.detach())
            forward_wellness_loss = None
            if self.wellness_mode == WellnessMode.FORWARD:
                pred_tok, z_motor = self.forward_model.predict_from_action(
                    last_step, motor_out, self.encoder_registry, self.prediction_heads,
                    achieved_state=self.last_achieved,
                    state_tokens={m: tokens[m] for m in tokens},
                    context_buffer=self.context_buffer,
                    **({} if self._memory_bias_current is None else {'memory_bias':self._memory_bias_current}))
                forward_wellness_loss = -self.wellness.score_predicted_wellness(pred_tok, z_motor).sum()
            # The eyes are handle 23: the gaze command is that handle's orientation,
            # decided by the same controller (and the same intention descent) as every
            # limb. The old gaze_head stays in the module purely so checkpoints load;
            # nothing reads it.
            gaze_out = self._gaze_from_handle(motor_out)
            if self.goal_gaze_hold and self._gaze_hold is not None:
                _tgt, _until = self._gaze_hold
                if step < _until:
                    gaze_out = _tgt   # the selected goal keeps the eyes
                else:
                    self._gaze_hold = None
            # Audit P262: no lesson reaches the voice head in this bridge (voice_target=None
            # below), so its forward pass builds no autograd graph. Safe HERE only: the bridge
            # never compiles (LITE_OVERRIDES disables torch_compile) and never runs
            # phase_rebuild's warm-up. The asynchronous lem pipeline's call must stay
            # gradient-tracked together with that warm-up (lem/training_step.py, step 10b).
            voice_out = None
            if self.voice_head is not None:
                with torch.no_grad():
                    voice_out = self.voice_head(last_step)
            if self.imagination_enabled:
                # review P1: transition()'s contract is ENCODER units; exec_t is the
                # ledger's execution-form (cm). Every other consumer normalizes at its
                # boundary - this one fed raw centimetres to the imagination chain at
                # weight 0.5 in every run.
                self.motor_history.append(command_for_encoder(exec_t).detach()
                                          if exec_t is not None else motor_out.detach())
            imagination_loss = None
            if (self.imagination_enabled and step % self.imagination_every == 0
                    and len(self.enc_history) >= self.imagination_steps + 1
                    and len(self.motor_history) >= self.imagination_steps):
                # review P2/P3: the chain starts at enc_history[0] = S_{t-3}, so it
                # gets the context and pose AS OF that frame - not the current
                # context (which contains the chain's own targets at attendable
                # positions) and not the current pose (which is from the future of
                # every transition being graded).
                _pure_depth = []
                # eval and probe runs never reach backward, so the chain was building
                # three transformer passes' worth of graph every step and discarding
                # it. The loss is still computed and reported; an outer no_grad still
                # wins, so this never turns gradients ON anywhere.
                with torch.set_grad_enabled(train and torch.is_grad_enabled()):
                    imagination_loss, _ = _run_imagination_compiled(
                        self.enc_history, self.motor_history, self.imagination_steps, self.forward_model,
                        self.prediction_heads, self.encoder_registry,
                        (self.ctx_history[0] if len(self.ctx_history) == self.imagination_steps + 1
                         else self.context_buffer),
                        self.active, self.n_mod,
                        latent_regularizer=self.latent_regularizer, latent_reg_weight=self.latent_reg_weight,
                        valid_history=list(self.valid_history),
                        target_history=(list(self.enc_history_raw)
                                        if len(self.enc_history_raw) == len(self.enc_history)
                                        else None),
                        pure_out=_pure_depth,
                        achieved_state=(self.ach_history[0]
                                        if len(self.ach_history) == self.imagination_steps + 1
                                        else handle_state.detach()),
                        **({} if self._memory_bias_current is None else
                           {'memory_bias':self.memory_bias_history[0]}))
                # horizon license: per-depth, did the chain beat do-nothing on the
                # SAME graded senses? (targets and baseline both from the raw
                # history, mirroring the chain's own validity filter)
                if _pure_depth and len(self.enc_history_raw) == self.imagination_steps + 1:
                    with torch.no_grad():
                        base0 = self.enc_history_raw[0]
                        for _si, _pl in enumerate(_pure_depth):
                            if _pl is None:      # depth had no graded real sense:
                                continue         # the slot holds so indices stay depths
                            _tgt = self.enc_history_raw[_si + 1]
                            _vh = (self.valid_history[_si + 1]
                                   if len(self.valid_history) > _si + 1 else {})
                            # source masks travel in two encodings (strings here,
                            # lem/source_mask's ints elsewhere) and to_str is the
                            # declared bridge. The imagination chain uses it on both of
                            # its validity reads (live_test.py:272 and :314); this was
                            # the one reader in this file that did a bare string
                            # compare, and it also had no missing-key default where the
                            # chain defaults to "real". Behaviour-identical today (every
                            # producer writes strings and lem_model.py setdefaults every
                            # active modality before the dict is stored).
                            _mods = [m for m in self.pred_mods
                                     if m != "motor" and m in _tgt and m in base0
                                     and _src_to_str(_vh.get(m, "real")) == "real"]
                            if not _mods:
                                # a frozen license pins the planner at horizon 1; count
                                # it rather than skipping in silence
                                self._plan_depths_ungraded += 1
                                continue
                            _pers = float(sum(F.mse_loss(base0[m], _tgt[m])
                                              for m in _mods))
                            _win = 1.0 if _pl < _pers else 0.0
                            if self._plan_win_ema is None:
                                self._plan_win_ema = [0.0] * self.imagination_steps
                            elif len(self._plan_win_ema) < self.imagination_steps:
                                # a resume that DEEPENED the chain (hail-mary lever):
                                # keep the earned depths, open the new ones at 0
                                self._plan_win_ema += [0.0] * (self.imagination_steps
                                                               - len(self._plan_win_ema))
                            self._plan_win_ema[_si] = (0.99 * self._plan_win_ema[_si]
                                                       + 0.01 * _win)
                    if self._plan_win_ema is not None:
                        _lic = 0
                        for _w in self._plan_win_ema:
                            if _w > 0.5:
                                _lic += 1
                            else:
                                break
                        self.last_plan_license = {
                            "win_emas": [round(float(w), 4) for w in self._plan_win_ema],
                            "licensed": _lic,
                            "depths_ungraded": int(self._plan_depths_ungraded),
                            "divergence_trips": int(getattr(self, "_plan_guard_trips", 0)),
                            "mode": self.plan_horizon_mode}
            # `pe` is the prediction heads applied to THIS moment, scored against THIS
            # moment. With --next-frame-loss the heads are trained only to produce the NEXT
            # moment, so this is the distance between the model's guess of what comes next
            # and the present: PREDICTED CHANGE, not reconstruction. It is small when the
            # model predicts persistence and grows when it predicts movement - H34 read it
            # rising the moment the action became readable, and every reader that took it
            # for reconstruction reported a fault (2026-08-27). Reconstruction proper is the
            # decoders' error, last_recon_dec below.
            with torch.no_grad():
                pe = {m: float(F.mse_loss(predictions[m].float(), targets[m].float()).item())
                      for m in predictions if m in targets}
            # curiosity / curriculum PE: real senses only (synthetic ones are noise and would
            # pin the curiosity drive at "overwhelmed"; same fix applied in lem/live_test.py)
            real_pe = [v for m, v in pe.items() if source_mask.get(m) != "synthetic"]
            recon_mean_pe = sum(real_pe) / len(real_pe) if real_pe else 0.0
            nf_loss = None
            self.last_prediction_audit = {}
            self.prediction_audit_rows = {}
            self.prediction_audit_hold_first_rows = {}
            if nf_live is not None:
                # Act 2 (forward hoisted above the gate - review P4). The previous
                # moment's RAW senses were re-encoded LIVE with the exec_t swap
                # (TEMPORAL ALIGNMENT, 2026-08-25), so the grade reaches the encoders
                # and the lesson is S_{t-1} + A_{t-1} -> S_t.
                nf_in, nf_preds = nf_live
                # AUDIT FIX: synthetic senses are fresh noise and a stale-vision target
                # is the model's own input - both excluded. Targets are tokens_RAW: the
                # gate now blends the SAME-step prediction into `tokens`, and grading
                # against the blend would let the lesson partially grade itself.
                nf_terms = {m: F.mse_loss(nf_preds[m], tokens_raw[m].squeeze(1).detach())
                            for m in self.pred_mods
                            if m in nf_preds and m in tokens_raw
                            and source_mask.get(m) == "real"
                            # the vestibular sense only from a real moment: after a stale one the
                            # prediction was made from the padded vector, another quantity
                            and (m != "vestibular" or self._nf_mask.get(m) == "real")}
                # review M22: a step where no graded sense is "real" left the stack empty
                nf_loss = (torch.stack(list(nf_terms.values())).sum() if nf_terms else None)
                if "vestibular" in nf_terms and self.vestibular_lesson_weight != 1.0:
                    # the sense's declared weight in this lesson (--vestibular-lesson-weight):
                    # the sum above counted it once; last_next_pe below keeps its unweighted error
                    nf_loss = nf_loss + (self.vestibular_lesson_weight - 1.0) * nf_terms["vestibular"]
                if self.lesson_probe is not None:
                    for _m, _t in nf_terms.items():
                        self._probe_term("next_frame." + _m, _t * (self.vestibular_lesson_weight
                                                                   if _m == "vestibular" else 1.0))
                self.last_next_pe = {m: float(v.detach()) for m, v in nf_terms.items()}
                if goal_variants is not None and nf_terms:
                    self._accumulate_memory_goal(nf_preds, goal_variants, tokens_raw, list(nf_terms))
                if self.observer_per_body_errors and nf_terms:
                    # the same squared error the lesson averages, kept per body: one body's slip
                    # must not vanish into the mean over eight (addendum v4 section 24)
                    from . import comparison_records as _CR
                    with torch.no_grad():
                        _per = torch.stack([((nf_preds[m].detach().float()
                                              - tokens_raw[m].squeeze(1).detach().float()) ** 2
                                             ).mean(-1) for m in nf_terms])
                    _moment = None
                    if _CR.RECORDING:
                        # records a and b (comparison_records.lane_moment): the persistence error of the same
                        # transitions and vision's error without the gaze copy's term, read with the per-body errors
                        # above in one transfer. Behind the records' guard: a failure is counted and the step's own
                        # per-body errors below stand
                        try:
                            _moment = _CR.lane_moment(step, nf_terms.keys(), _per, nf_in, tokens_raw,
                                                      self.gaze_copy_now, B)
                        except Exception as _ex:
                            _moment = None
                            _CR.record_failed(self, "lane", _ex)
                    # (a write; tests/test_observer_record.py checks that no line of this file reads the copy)
                    self.observer_moment = {"step": int(step), "senses": list(nf_terms.keys()),
                                            "next_pe_per_body": _per.cpu().numpy()} if _moment is None else _moment
                from .prediction_audit import paired_errors, paired_rows, audit_blocks
                from . import prediction_audit as _pa
                shuffled_preds = {}
                # WHICH SWAP-TEST BLOCKS GRADE THIS TRANSITION (prediction_audit.audit_blocks, 2026-10-02):
                # "every_20" = the existing block (model steps that are multiples of 20, the condition this
                # site always had), "hold_first" = the hold's first model step, derived from the trainer's
                # latch rule and --motor-hold (efference_tenure_steps is --motor-hold, set unconditionally)
                _audit = (audit_blocks(step, self.efference_tenure_steps)
                          if (B > 1 and exec_t is not None
                              and self.config['model'].get('dropout', 0.) == 0.) else ())
                if not _pa.ADDED_READINGS:
                    _audit = tuple(b for b in _audit if b == "every_20")
                # A paired intervention on the command alone. Sample periodically;
                # leave the learning graph and the production command unchanged.
                if "every_20" in _audit:
                    with torch.no_grad():
                        shuffled_raw = dict(nf_raw)
                        mt = nf_raw.get('motor')
                        if torch.is_tensor(mt) and mt.shape[-1] == MOTOR_INPUT_F:
                            shuffled_raw['motor'] = torch.cat([
                                mt[..., :MOTOR_ACHIEVED_F],
                                command_for_encoder(exec_t.roll(1, dims=0).to(mt.dtype))], dim=-1)
                            shuffled_in = self.encoder_registry.encode(fit_tactile(shuffled_raw), self._nf_mask)
                            shuffled_in = {m: (shuffled_in[m] if self._nf_mask.get(m) == 'real'
                                              else self._nf_tokens[m]) for m in self.active}
                            _, shuffled_last, _ = self.forward_model(shuffled_in, self._nf_ctx,
                                **({} if nf_memory_bias is None else {'memory_bias':nf_memory_bias.detach()}))
                            shuffled_preds = self.prediction_heads(shuffled_last,
                                current={m: v.squeeze(1) for m, v in shuffled_in.items()},
                                who='paired shuffled-action audit')
                            if self.efference is not None:
                                # the same gaze copy, with the other body's order: the audit then
                                # compares two predictions formed the same way
                                from lem.model.efference import apply_gaze_copy
                                shuffled_preds = apply_gaze_copy(
                                    self.prediction_heads, shuffled_preds, shuffled_last,
                                    command_for_encoder(exec_t.roll(1, dims=0)), self._nf_ach)
                # THE ADDED ARMS AND THE SECOND PHASE (2026-10-02, docs/gain_of_function_v4/agents/
                # swap_test_phases_2026-10-02.md): another body's command size-matched to this body, this
                # body's own command shifted by one standard deviation per column, and on the hold's first
                # model step the absolute roll too. No gradient, and outside every liveness watching window
                # (_swap_test_arms), so no loss, weight, command or random stream of the step changes
                _arms = (self._swap_test_arms(nf_raw, nf_in, nf_memory_bias, exec_t, step, B,
                                              absolute=("hold_first" in _audit and "every_20" not in _audit))
                         if _audit and _pa.ADDED_READINGS else {})
                _first_abs = shuffled_preds if "every_20" in _audit else _arms.get("shuffled", {})
                for m in nf_terms:
                    if self._nf_mask.get(m) == 'real':
                        _sm = _arms.get("size_matched", {}).get(m)
                        _so = _arms.get("shifted_own", {}).get(m)
                        _every = "every_20" in _audit
                        self.last_prediction_audit[m] = paired_errors(
                            nf_preds[m], nf_in[m].squeeze(1), tokens_raw[m].squeeze(1),
                            shuffled_preds.get(m), size_matched=_sm if _every else None,
                            shifted_own=_so if _every else None)
                        # the same per body, on the device (the stats record's prediction_audit_per_body)
                        self.prediction_audit_rows[m] = paired_rows(
                            nf_preds[m], nf_in[m].squeeze(1), tokens_raw[m].squeeze(1),
                            shuffled_preds.get(m), size_matched=_sm if _every else None,
                            shifted_own=_so if _every else None)
                        if "hold_first" in _audit:
                            # the stats record's prediction_audit_hold_first block (per body, on the device)
                            self.prediction_audit_hold_first_rows[m] = paired_rows(
                                nf_preds[m], nf_in[m].squeeze(1), tokens_raw[m].squeeze(1),
                                _first_abs.get(m), size_matched=_sm, shifted_own=_so)
                from . import comparison_records as _CR
                _spike = _CR.spike_of(self) if _CR.RECORDING else None
                if _spike is not None and _spike.enabled:
                    # RECORD D, THE SPIKE TRIGGER: this model step's per-body vision prediction and persistence errors
                    # (the audit rows just computed, read from the device once); at a crossing of the spike line the
                    # inputs of this transition are copied to the processor (no gradient, no arithmetic). Behind the
                    # records' guard: a failure is counted (VisionSpike.fail), the trigger observes the next model
                    # step, and nothing reaches the lesson.
                    try:
                        _vr = self.prediction_audit_rows.get("vision")
                        if isinstance(_vr, dict) and torch.is_tensor(_vr.get("prediction_mse")):
                            _vals = torch.stack([_vr["prediction_mse"], _vr["persistence_mse"],
                                                 _vr["transitions"]]).float().cpu().numpy()
                            if _spike.observe(step, _vals[0], _vals[2], _vals[1]):
                                _gcn = self.gaze_copy_now or {}
                                _spike.keep(step, _CR.cpu_tree(dict(
                                    errors=_vals[0], persistence=_vals[1], graded=_vals[2],
                                    prev_tokens=nf_in, target_tokens=tokens_raw, context=self._nf_ctx,
                                    memory_bias=nf_memory_bias, command=exec_t, pose=self._nf_ach,
                                    prediction={m: nf_preds[m] for m in nf_terms},
                                    vision_without_copy=_gcn.get("without"), gaze_copy_applied=_gcn.get("applied"),
                                    raw_prev=nf_raw, raw_now=frame, prev_mask=dict(self._nf_mask),
                                    now_mask=dict(source_mask))))
                            _spike.judge(step)
                    except Exception as _ex:
                        _spike.fail(_ex)
                # SELF-TOUCH, scored against the bodies that were NOT touching themselves.
                # The tactile error is dominated by the floor, which the body rests on every
                # step, and the prediction here is the touch EMBEDDING, one vector per body,
                # so there is no way to look only at the rows carrying self-contact. This
                # comment used to claim the comparison was restricted to those rows; it never
                # was. What isolates self-touch is the CONTRAST: the same measurement on the
                # other bodies, subtracted, so whatever the floor contributes cancels.
                if 'tactile' in nf_terms and self._nf_mask.get('tactile') == 'real':
                    _tt = nf_raw.get('tactile')
                    if torch.is_tensor(_tt) and _tt.shape[-1] != TOUCH_TOKEN_F:
                        _tt = fit_tactile({'tactile': _tt})['tactile']
                    # The control is computed every 20 steps; the audit runs every step.
                    # The GATE must read a measurement that had its control, so the newest
                    # controlled one is kept separately - otherwise the gate sees "no
                    # control" on 19 steps in 20 and can never open.
                    self.last_self_touch_audit = self._self_touch_audit(
                        nf_preds.get('tactile'),
                        tokens_raw.get('tactile', torch.zeros(0)).squeeze(1)
                        if torch.is_tensor(tokens_raw.get('tactile')) else None,
                        shuffled_preds.get('tactile'), _tt)
                    _a = self.last_self_touch_audit or {}
                    if _a.get("improvement_over_shuffled") is not None:
                        self.last_self_touch_controlled = self.last_self_touch_audit
                    # the gate judges the CONTRAST against the bodies that were not touching
                    # themselves, whose null is zero, over a run of readings
                    if _a.get("improvement_over_other_bodies") is not None:
                        self._self_touch_ring.append(
                            float(_a["improvement_over_other_bodies"]))
                        self._self_touch_ring_step = self.step_count
                if self.nf_norm is not None and nf_terms:
                    # --next-frame-norm persistence (DESIGN D12, lem/model/next_frame_norm.py): the
                    # same per-sense errors, each weighted by the mean running persistence error over
                    # that sense's own, so every sense is pushed to beat "nothing changes" equally and
                    # the lesson keeps its size. The readings are the audit's persistence errors just
                    # measured (a sense whose previous moment was not a real reading has none); the
                    # weights use the running averages from BEFORE this step's readings join them. No
                    # device read is added: the audit and last_next_pe above already read these floats.
                    _nfr = {m: a["persistence_mse"] for m, a in self.last_prediction_audit.items()
                            if a is not None and m in nf_terms}
                    # a sense predicted FROM a moment that was not a real reading of it (the padded
                    # vestibular vector after a resume) carries no weight this step: its error is the
                    # jump from the stand-in, not a prediction
                    _nfw, _nfi = self.nf_norm.weights(
                        list(nf_terms), _nfr,
                        input_not_real=[m for m in nf_terms if self._nf_mask.get(m) != "real"])
                    _nfw_gauge = _nfw
                    if self.vestibular_encoder_repair != "off":
                        # the revival's ceiling (vestibular_sense.REVIVAL_CEILING): while the sense's weight is at
                        # its cap, its term is held to the mean of the other graded senses' terms; the errors are
                        # the floats last_next_pe already read. It judges the term with the sense's lesson weight
                        # in it, and the window's shares read the weights the lesson carries (third review,
                        # section D: at a lesson weight of 2.0 the optimizer saw 33 in 100, the gauge 20)
                        from .vestibular_sense import (revival_ceiling as _vr_ceiling,
                                                       lesson_weights_used as _vr_used)
                        _nfw, self.vestibular_revival_live = _vr_ceiling(
                            _nfw, self.last_next_pe, _nfi.get("capped"),
                            lesson_weight=self.vestibular_lesson_weight)
                        _nfw_gauge = _vr_used(_nfw, self.vestibular_lesson_weight)
                    # the weight reaches the prediction head and the core; the path into the
                    # encoder through the head's residual `+ current` keeps the plain weight, so a
                    # quiet sense's encoder is not pushed harder to stop its token changing
                    # (measured 2026-09-23, next_frame_norm.py "THE ENCODER'S SHARE")
                    from lem.model.next_frame_norm import weighted_term as _nf_wt
                    _nf_res = bool(getattr(self.prediction_heads, "residual", False))
                    _nf_parts = [
                        _nf_wt(nf_preds[m], nf_in[m].squeeze(1) if nf_in[m].dim() == 3 else nf_in[m],
                               tokens_raw[m].squeeze(1).detach(), _nfw[m], _nf_res)
                        * (self.vestibular_lesson_weight if m == "vestibular" else 1.0)
                        for m in nf_terms]
                    nf_loss = torch.stack(_nf_parts).sum()
                    if self.lesson_probe is not None:
                        # the normalised parts replace the plain ones recorded above
                        for _m, _t in zip(nf_terms, _nf_parts):
                            self._probe_term("next_frame." + _m, _t)
                    self.nf_norm.update(_nfr)
                    self.nf_norm.add_live(self.last_next_pe, _nfw_gauge, _nfi)
            from . import prediction_audit as _pa
            if exec_t is not None and _pa.ADDED_READINGS:
                # THE EXECUTED-COMMAND STREAM'S PER-COLUMN SPREAD (the swap test's shifted-own arm,
                # prediction_audit.command_spread_update): this model step's executed command joins the
                # running mean and mean square after the audit above read the spread of the model steps
                # before it. Measurement state only: no gradient, nothing trains it, nothing else reads it
                from .prediction_audit import command_spread_update
                self._command_spread = command_spread_update(self._command_spread, exec_t)
            # `pe` above compares prediction heads against the CURRENT tokens: it is a
            # reconstruction diagnostic, not evidence about what the action did. The
            # causal quantity is last_next_pe, produced by the delayed lesson
            # S_{t-1} + A_{t-1} -> S_t, graded JUST ABOVE (review M21: it used to be
            # graded below this read, so curiosity and causal_pe ran one step stale).
            nxt = [v for k, v in (self.last_next_pe or {}).items()
                   if source_mask.get(k) != "synthetic"]
            self.last_recon_pe = recon_mean_pe
            self.last_causal_pe = (sum(nxt) / len(nxt)) if nxt else float("nan")
            # mean_pe is the CURRICULUM's input (train_mimo.py hands it to
            # CurriculumManager.step, which feeds MasteryTracker's 50,000-step window and
            # the phase progress report). A NaN there poisons that window's mean and
            # variance for 50,000 steps, so the reconstruction fallback stays on THIS
            # quantity and is named as a fallback.
            mean_pe = (sum(nxt) / len(nxt)) if nxt else recon_mean_pe
            # ONE CURIOSITY QUANTITY. The drive is computed from the causal error alone.
            # It used to be computed from mean_pe, which silently substitutes the
            # reconstruction diagnostic when no sense was graded real - so on exactly
            # those steps a drive derived from reconstruction was reported under the
            # causal error's name while the goal ring applied no boost at all. Both now
            # read the same number and the same flag. compute_drive is never handed the
            # unmeasured value: compute_drive(nan) returns 1.0, its PEAK.
            self.curiosity_measured = bool(nxt) and self.curiosity is not None
            drive = (self.curiosity.compute_drive(self.last_causal_pe)
                     if self.curiosity_measured else 0.0)
            # no measurement -> no modulation. 1.0 is the identity multiplier here, not a
            # drive value, which is why it is a separate name from `drive`.
            drive_mult = drive if self.curiosity_measured else 1.0
            scaled_total = drive_mult * total_wellness if self.wellness_mode == WellnessMode.DIRECT else None
            scaled_forward = drive_mult * forward_wellness_loss if forward_wellness_loss is not None else None
            latent_pen = self._latent_prediction_penalty(predictions, nf_live, source_mask)
            # review fix 1: re-run the controller on the PREVIOUS frame's inputs (params get
            # gradients; the cached latents are detached) and grade THAT command against the
            # CURRENT frame's achieved pose - the pose its action actually produced.
            if prev_lat is not None and prev_lat.shape[0] == B:
                motor_for_loss, _lc, _lm = self.motor_controller(prev_lat, prev_hs,
                                                                 wellness_grad=prev_wg)
            else:
                motor_for_loss = motor_out              # first step only: same-frame fallback
                _lm = motor_meta
            motor_target = handle_state[:, :, :9].clone()
            motor_target[..., 7:9] = motor_for_loss[..., 7:9].detach()   # de-echo: no self-copy teacher
            loss, bd = self.loss_manager.compute(
                predictions, targets, source_mask, motor_for_loss, motor_target, wellness_out, dev_limits, step,
                gaze_out=None,
                # Spec 6.4: gaze is "an active motor decision rather than a passive property
                # of the vision system". The target here WAS the eyes' own measured angles,
                # which makes it exactly that passive property - every eye position satisfies
                # it, so the loss is a lock that holds the eyes wherever they are. Measured:
                # the ball sat 60-120 deg off axis in every run, and a hand was inside the
                # 60 deg field on 0 of 400 steps. The spec's own supervision for gaze is
                # mocap/eye-tracking during imitation, which "provides acceleration, not a
                # prerequisite" - we have none, so gaze carries no regression target at all.
                # Aiming lives where the spec puts it: the head is handle 12, already inside
                # the wellness/intention loop.
                gaze_target=None, voice_out=voice_out, voice_target=None,
                # audit P118: the controller's position channel is bounded at its source;
                # the loss charges the excursion the controller reports instead
                motor_pos_excursion=(_lm or {}).get("pos_excursion"),
                # review M10/E3: vicreg's variance floor could be satisfied by gate
                # modulation instead of encoder variance (gradient reached td_gates),
                # and synthetic placeholder tokens were regularised toward Gaussian
                # statistics - the exact thing the replay path excludes.
                encoder_outputs={m: tokens_raw[m].squeeze(1) for m in tokens_raw
                                 if source_mask.get(m) == "real"
                                 and m not in self._no_encoder_output},
                return_breakdown=True,
                total_wellness=scaled_total, forward_wellness_loss=scaled_forward,
                imagination_loss=imagination_loss, latent_reg_loss=latent_pen)
            if nf_loss is not None:
                loss = loss + nf_loss
                bd["next_frame"] = nf_loss
            if self.efference is not None:
                # the heads' own lesson and gauges; the loss terms are already weighted
                _eff = self._efference_lesson(nf_latent, exec_t, executed_order_id,
                                              handle_state, frames, step, B, train=train)
                for _k, _v in _eff.items():
                    loss = loss + _v
                    bd[_k] = _v
                    self._probe_term(_k, _v)
            if self.contact_head is not None:
                # the contact head's lesson, the onset success signal and this moment's
                # contact_control value (set before the goal ring reads it below); the loss
                # term is already weighted
                for _k, _v in self._contact_lesson(nf_latent, nf_live, nf_memory_bias, exec_t,
                                                   frames, step, B).items():
                    loss = loss + _v
                    bd[_k] = _v
                    self._probe_term(_k, _v)
            if self.synergy is not None:
                # the motor synergy's hindsight lesson (DESIGN D11): only the synergy learns from
                # it (its inputs are recorded poses and changes); the term is already weighted.
                # Full precision: it grades changes of a few degrees and centimetres, the size the
                # card's bf16 would round at the third significant figure
                with torch.amp.autocast("cuda", enabled=False):
                    _syn_terms = self._synergy_lesson(exec_t, executed_order_id, handle_state,
                                                      step, B, train)
                for _k, _v in _syn_terms.items():
                    loss = loss + _v
                    bd[_k] = _v
                    self._probe_term(_k, _v)
            if self.proprio_within is not None:
                # --proprio-token-repair within_body (mimo_bridge/proprio_token.py, version 5): each
                # body's proprioception token must move with its own posture (its response per unit of
                # posture movement at least the line) over the last 300 s. NOT added to the loss: its
                # gradient joins the proprioception encoder's AFTER the clip below, at a set share of that
                # encoder's own gradient size (review finding 1)
                self._pw_pending = self._proprio_within_lesson(frame, source_mask, train)
            if self.wellness_channels is not None:
                # learning progress and controllability (DESIGN D6, D7): measurements only, no
                # gradient, no draw from any random stream
                self._observe_wellness_channels(step, exec_t, executed_order_id, handle_state,
                                                nf_live, tokens_raw, source_mask, B)
            if self.anticipation_gaze:
                # anticipation-gaze (B11, 2026-09-05): reality grades. Pull handle 23's
                # gaze toward where REAL self-caused change sits on this frame - the
                # env-measured direction of the struck mobile or the moving hand
                # (antic_* task fields). Masked to frames that HAVE a consequence.
                # Weight matches the spec's gaze-head loss (UEMA 6.4, 0.1). This trains
                # dwell-on-consequence; whether gaze arrives BEFORE an event is judged
                # by the dragonfly-hover metric, never by this loss.
                _tg, _mk = [], []
                for f in frames:
                    tk = (f.meta.get("task") or {})
                    _mk.append(1.0 if tk.get("antic_ok") else 0.0)
                    _tg.append((float(tk.get("antic_az") or 0.0),
                                float(tk.get("antic_el") or 0.0)))
                if any(_mk):
                    _tgt = torch.tensor(_tg, device=d, dtype=torch.float32)
                    _msk = torch.tensor(_mk, device=d, dtype=torch.float32)
                    _pred = self._gaze_from_handle_t(motor_out)
                    antic = (((_pred - _tgt) ** 2).sum(-1) * _msk).sum() / _msk.sum()
                    loss = loss + self.anticipation_gaze_w * antic
                    if self.lesson_probe is not None:
                        self._probe_term("anticipation_gaze", self.anticipation_gaze_w * antic)
                    # THE CONTRIBUTION, like every LossManager term (loss_manager stores
                    # the already-weighted value for prediction, motor, wellness, vicreg,
                    # imagination and latent_reg). Stored raw, this bar was drawn at 10x
                    # the share it actually taught: 0.569440 against 0.056944 on H81's
                    # step 380.
                    bd["anticipation_gaze"] = self.anticipation_gaze_w * antic
            if self.recon_decoders is not None:
                # reconstruction on SEPARATE decoders: scheduled weight - strong scaffold
                # while prediction is poor, fading to a smoke-detector floor as it sharpens
                npe = [v for v in self.last_next_pe.values()]
                w_copy = float(min(0.3, max(0.02, 0.5 * (sum(npe) / len(npe)))) if npe else 0.3)
                rd_terms = {m: F.mse_loss(self.recon_decoders[m](last_step[:, mi]),
                                          tokens_raw[m].squeeze(1).detach())
                            for mi, m in enumerate(self.active) if m in self.recon_decoders
                            and source_mask.get(m) == "real"}
                rl = list(rd_terms.values())
                # the real reconstruction, per sense: what the fault rule must read
                self.last_recon_dec = {m: float(v.detach()) for m, v in rd_terms.items()}
                if rl:
                    recon = torch.stack(rl).mean()
                    loss = loss + w_copy * recon
                    if self.lesson_probe is not None:
                        self._probe_term("recon_dec", w_copy * recon)
                    # THE CONTRIBUTION, not the pre-weight error. w_copy is scheduled and
                    # ran 0.02 to 0.3 across H81's twenty snapshots, so the raw value was
                    # drawn at 3.3x to 50x its share - at step 380 recon was drawn at
                    # 13.916830 against a real contribution of 0.278337, taller than
                    # VICReg's 0.959120 while teaching a third as much.
                    bd["recon"] = w_copy * recon
                    # a schedule value, not a lesson: it is not in the sum, so it is not
                    # in the breakdown. The decoders' own per-sense error is in
                    # last_recon_dec and reaches the run record as recon_dec.
                    self.last_recon_weight = float(w_copy)
            # ACTION-TENURE LESSON. Fires only at the step where a NEW order first
            # appears: the outgoing order changed, so the previous order's tenure just
            # ended, and the next-frame cache (_nf_tokens/_nf_ctx, the PREVIOUS step)
            # holds both ends of it - the state where the pending order began was cached
            # at its first step, and the cache right now is the state where it ended.
            if (self.action_tenure and exec_t is not None
                    and self._nf_tokens is not None and self._nf_ctx is not None):
                # REVIEW R1 (2026-09-02). The trainer rewrites handle 23 (gaze)
                # from the eye efference copy every step - gaze is deliberately not
                # held - so whole-action equality saw "a new order" every step and the
                # tenure lessons in H53-H58 ran at the 1-step horizon they were built
                # to escape. The bench held its commands bit-identical, which is why
                # offline and live diverged. Boundaries are the LIMBS' order changing.
                _gi = self.GAZE_IDX
                if executed_order_id is not None:
                    # deferred item 1 (2026-09-02): the trainer stamps each latched
                    # order with a serial, so a bit-identically RE-ISSUED order - which
                    # float equality cannot tell from a held one - still ends the
                    # previous tenure. Equality below stays as the fallback for
                    # callers without ids.
                    changed = (self._tenure_prev_order_id is None
                               or int(executed_order_id) != self._tenure_prev_order_id)
                else:
                    _cur_limbs = torch.cat([exec_t[:, :_gi], exec_t[:, _gi + 1:]], dim=1)
                    _prev_limbs = (None if self._tenure_prev_order is None
                                   or self._tenure_prev_order.shape != exec_t.shape
                                   else torch.cat([self._tenure_prev_order[:, :_gi],
                                                   self._tenure_prev_order[:, _gi + 1:]], dim=1))
                    changed = (_prev_limbs is None
                               or not torch.equal(_prev_limbs, _cur_limbs))
                if changed:
                    p = self._tenure_pending
                    if (p is not None and p["tokens"] is not None
                            and self._nf_ach is not None
                            and self._nf_ach.shape[0] == B
                            and p["ach"].shape[0] == B):
                        # physical target: where the limbs ended up, minus where they
                        # began, in the tokenizer's own units (positions / 200 cm) -
                        # both ends from the _nf-aligned pose cache, the same moments
                        # the token windows use (review R3)
                        tgt = (self._nf_ach[..., :3] - p["ach"][..., :3]
                               ).detach().reshape(B, -1)
                        ach9 = p["ach"]
                        row = torch.cat([ach9[..., :MOTOR_ACHIEVED_F],
                                         command_for_encoder(p["order"])], dim=-1)
                        no = torch.zeros_like(p["order"])
                        no[..., 6] = 1.0        # identity rotation, zero strength: "do nothing"
                        row0 = torch.cat([ach9[..., :MOTOR_ACHIEVED_F],
                                          command_for_encoder(no)], dim=-1)

                        def _tenure_forward(motor_row, grad):
                            ctxm = torch.enable_grad() if grad else torch.no_grad()
                            with ctxm:
                                z = self.encoder_registry.encode(
                                    {"motor": motor_row}, {"motor": "real"})["motor"]
                                z = z.reshape(B, 1, self.d_model)
                                tk = {m: (z if m == "motor"
                                          else p["tokens"][m].reshape(B, 1, self.d_model))
                                      for m in self.active if m == "motor" or m in p["tokens"]}
                                memory_kw = {}
                                if p.get('memory') is not None:
                                    remembered, _ = self.persistent_memory.encode(**p['memory'])
                                    memory_kw['memory_bias'] = self.persistent_memory.bias(remembered, site="tenure")
                                _l2, last2, _c2 = self.forward_model(tk, p["ctx"], **memory_kw)
                                return self.tenure_heads["pose"](last2.reshape(B, -1))
                        pred_with = _tenure_forward(row, grad=train)
                        pred_null = _tenure_forward(row0, grad=False)
                        mw = F.mse_loss(pred_with, tgt)
                        m0 = F.mse_loss(pred_null, tgt).detach()   # STOP-GRAD baseline
                        # SCALE-FREE hinge (2026-09-01). The absolute form scaled with
                        # the error pie: H47's pose error fell 10x in an hour and the
                        # fine fell with it, to ~0.002% of total loss - a pressure the
                        # optimizer cannot feel. The ratio form demands the same alpha
                        # regardless of how small the errors get.
                        # review M2: below the measured still-body pose noise the
                        # ratio is noise-over-noise and its gradient (1/m0) dominates
                        # the clipped step. No motion signal -> no hinge; mw remains.
                        if float(m0) >= self._tenure_m0_floor:
                            t_term = mw + torch.relu(mw / m0
                                                     - (1.0 - self.tenure_alpha))
                        else:
                            t_term = mw
                        self.last_tenure_pe = {"pose": float(mw.detach())}
                        self.last_tenure_gap = {"pose": float(m0 - mw.detach())}
                        if train:
                            loss = loss + self.tenure_weight * t_term
                            if self.lesson_probe is not None:
                                self._probe_term("tenure", self.tenure_weight * t_term)
                            # the contribution, not the pre-weight hinge (weight 0.5 in
                            # this run), so the panel's bars are one scale
                            bd["tenure"] = self.tenure_weight * t_term
                        if (self.tenure_token_reach and train and p.get("raw") is not None
                                and p.get("mask") is not None
                                and self._nf_raw is not None and self._nf_mask is not None):
                            # Both ends of the change re-encoded LIVE so the matching
                            # gradient reaches the encoders through begin AND end.
                            # review M14: encode() is per-modality, so the old motor-row
                            # swap could not influence the sensory slices - it was dead
                            # weight, and each call also encoded audio/intermind/motor
                            # for nothing (twice per boundary step). The order reaches
                            # this loss through order_proj alone. Only the three senses
                            # are encoded, gated on the CACHED frames' masks (the
                            # current step's staleness says nothing about two cached
                            # frames).
                            def _live_tokens(raw, mask):
                                r2 = {m: raw[m] for m in ("vision", "tactile", "proprioception")
                                      if m in raw}
                                m2 = {m: mask[m] for m in r2 if m in mask}
                                e2 = self.encoder_registry.encode(fit_tactile(r2), m2)
                                return {m: e2[m].reshape(B, self.d_model)
                                        for m in r2 if m in e2 and mask.get(m) == "real"}
                            tb_l = _live_tokens(p["raw"], p["mask"])
                            te_l = _live_tokens(self._nf_raw, self._nf_mask)
                            common = [m for m in ("vision", "tactile", "proprioception")
                                      if m in tb_l and m in te_l]
                            if common:
                                # PER BODY (2026-09-01). The first wiring pooled the
                                # eight bodies' projections and deltas into one pair -
                                # matching a MEAN order to a MEAN outcome, where the
                                # signal mostly cancels; the bench that validated this
                                # objective runs one body. Each body now carries its
                                # own 1-of-K contrast against its own banked history.
                                z = torch.zeros(B, self.d_model, device=self.device)
                                # HAND-REGARD GATE (2026-09-02). Vision's slice takes
                                # part only for bodies whose hand was in the field of
                                # view at BOTH ends of the tenure: on hand-out-of-view
                                # tenures there is physically nothing for the order to
                                # have changed in the image, and demanding traces there
                                # eroded vision's causal error ~5x over six H55 hours.
                                # Visual self-attribution is learned while watching the
                                # hand - the infant's own curriculum - not deferred.
                                # review M13: the flags must describe the frames the
                                # delta is measured on - the CACHED end frame (previous
                                # step), not this step's frames (env computes the flag
                                # at the render substep, one frame ahead of the cache).
                                hv_end = (self._nf_handvis
                                          if self._nf_handvis is not None
                                          and self._nf_handvis.shape[0] == B
                                          else torch.zeros(B, dtype=torch.bool,
                                                           device=self.device))
                                hv = (p.get("hand_vis", hv_end) & hv_end).float().unsqueeze(1)
                                parts = []
                                for m in ("vision", "tactile", "proprioception"):
                                    dpart = (te_l[m] - tb_l[m]) if m in common else z
                                    if m == "vision":
                                        dpart = dpart * hv
                                    # SLICE NORMALIZATION (2026-09-02). H54's matching
                                    # crushed vision's tenure-change 5x (0.263 -> 0.079
                                    # on the bed measure): a near-zero-signal slice is
                                    # noise in the lineup, and silencing it cleans the
                                    # contrast. Unit-normalizing each slice per body
                                    # renormalizes a quieted slice right back, so
                                    # suppression buys nothing.
                                    dpart = dpart / (dpart.std(dim=-1, keepdim=True) + 1e-6)
                                    parts.append(dpart)
                                delta = torch.cat(parts, dim=-1)             # [B, 384]
                                q = self.tenure_heads["order_proj"](
                                    command_for_encoder(p["order"]).reshape(B, -1))  # [B, 384]
                                sims = [F.cosine_similarity(q, delta, dim=-1) / 0.2]
                                # SAME-INSTANT CROSS-BODY DECOYS: body b must beat the
                                # other bodies' outcomes from this very event - decoys
                                # with zero staleness (the freshness control passed,
                                # 0.282 vs 0.113, but fresh decoys close the question
                                # structurally). The stale bank stays as a supplement
                                # and for single-body rigs.
                                if B > 1:
                                    for shift in range(1, B):
                                        sims.append(F.cosine_similarity(
                                            q, torch.roll(delta.detach(), shift, dims=0),
                                            dim=-1) / 0.2)
                                for dd in self._tenure_bank:
                                    sims.append(F.cosine_similarity(q, dd, dim=-1) / 0.2)
                                if len(sims) >= 4:
                                    logits = torch.stack(sims, dim=1)        # [B, 1+K]
                                    match_loss = F.cross_entropy(
                                        logits, torch.zeros(B, dtype=torch.long, device=logits.device))
                                    loss = loss + self.tenure_token_weight * match_loss
                                    if self.lesson_probe is not None:
                                        self._probe_term("tenure_tok", self.tenure_token_weight * match_loss)
                                    # the contribution, not the pre-weight
                                    # cross-entropy (weight 0.25 in this run)
                                    bd["tenure_tok"] = self.tenure_token_weight * match_loss
                                    with torch.no_grad():
                                        self.last_tenure_tok = {
                                            "match_loss": float(match_loss.detach()),
                                            "rank1": float((torch.argmax(logits, dim=1) == 0).float().mean()),
                                            "n_decoys": float(len(sims) - 1),
                                            "vis_gate_open": float(hv.mean())}
                                        if self.agency_drive:
                                            pay = (torch.argmax(logits, dim=1) == 0).float()
                                            # slice advantages vs the decoys: the live
                                            # evidence per sense. review M15: at B==1
                                            # there are no cross-body decoys and the
                                            # EMAs never updated - the banked stale
                                            # decoys serve as the comparison set on
                                            # single-body rigs.
                                            _decoy_sets = None
                                            if B > 1:
                                                _decoy_sets = [torch.roll(delta, k, dims=0)
                                                               for k in range(1, B)]
                                            elif len(self._tenure_bank) > 0:
                                                _decoy_sets = list(self._tenure_bank)
                                            if _decoy_sets is not None:
                                                for m_i, m_name in ((0, "vis"), (1, "touch")):
                                                    sl = slice(m_i * self.d_model, (m_i + 1) * self.d_model)
                                                    a_t = F.cosine_similarity(q[:, sl], delta[:, sl], dim=-1)
                                                    if m_name == "vis":
                                                        # ELIGIBILITY SPLIT (follow-up review
                                                        # 2026-09-08): the old mean ran over ALL
                                                        # bodies, but hand-out-of-view bodies carry
                                                        # a ZEROED vision slice - so the gauge moved
                                                        # with the eligible-body COUNT at fixed
                                                        # per-body skill. Recognition is now scored
                                                        # only on eligible bodies against eligible
                                                        # impostors; the eligible fraction is its
                                                        # own gauge (vis_adv_elig).
                                                        _el = hv.squeeze(1) > 0.5
                                                        self._vis_adv_elig_ema = (0.99 * self._vis_adv_elig_ema
                                                                                  + 0.01 * float(_el.float().mean()))
                                                        adv = eligible_contrast(q, delta, _el, sl)
                                                        if adv is not None:
                                                            self._vis_adv_ema = 0.99 * self._vis_adv_ema + 0.01 * adv
                                                        elif _el.any() and B == 1 and _decoy_sets:
                                                            # single-body rig: stale bank rows carry
                                                            # no eligibility flag - old semantics,
                                                            # stated limitation
                                                            _c = torch.stack([
                                                                F.cosine_similarity(q[:, sl], dd[:, sl], dim=-1)
                                                                for dd in _decoy_sets]).mean(0)
                                                            self._vis_adv_ema = (0.99 * self._vis_adv_ema
                                                                                 + 0.01 * float((a_t - _c)[_el].mean()))
                                                        # MEASURED NULL, TEMPORAL MISPAIRING (2026-09-09
                                                        # review: the first null rolled order and outcome
                                                        # TOGETHER, which by relabeling is exactly MINUS
                                                        # the real advantage - it contained the signal).
                                                        # Now: the identical estimator scores this event's
                                                        # orders against the OLDEST BANKED event's outcomes
                                                        # (joint eligibility) - a time shift no true
                                                        # order->outcome association survives, while the
                                                        # distributions and estimator form are preserved.
                                                        # Qualified on synthetic known-association and
                                                        # no-association fixtures (test_vis_adv_null.py).
                                                        if len(self._tenure_bank) > 0 and len(self._tenure_bank_el) > 0:
                                                            _db = self._tenure_bank[0]
                                                            _elb = self._tenure_bank_el[0]
                                                            if _db.shape[0] == B and _elb.shape[0] == B:
                                                                nul = eligible_contrast(
                                                                    q, _db, _el & _elb, sl)
                                                                if nul is not None:
                                                                    self._vis_adv_null_ema = (
                                                                        0.99 * self._vis_adv_null_ema
                                                                        + 0.01 * nul)
                                                    else:
                                                        a_d = torch.stack([
                                                            F.cosine_similarity(q[:, sl], dd[:, sl], dim=-1)
                                                            for dd in _decoy_sets]).mean(0)
                                                        adv = float((a_t - a_d).mean())
                                                        self._touch_adv_ema = 0.99 * self._touch_adv_ema + 0.01 * adv
                                            if self._agency_pay_ema is None or self._agency_pay_ema.shape[0] != B:
                                                self._agency_pay_ema = pay
                                            else:
                                                self._agency_pay_ema = 0.9 * self._agency_pay_ema + 0.1 * pay
                                # review E22: q was stored and never read
                                self._tenure_bank.append(delta.detach())
                                # eligibility stored ALONGSIDE the delta, same site, so
                                # bank and bank_el can never desynchronize - the temporal
                                # null needs the past event's own hand-visibility mask
                                self._tenure_bank_el.append((hv.squeeze(1) > 0.5).detach())
                    # the order that just appeared begins its tenure at the PREVIOUS
                    # step's state - exactly what the next-frame cache holds now.
                    # review M3b: the trainer rewrites handle 23 (gaze) every step, so
                    # the stored gaze command was only in force for the tenure's FIRST
                    # interval; the pose head and order_proj conditioned on it as if
                    # held. Stored neutralized, like the null row (identity rotation,
                    # zero strength). review M13: the begin visibility flag comes from
                    # the same cached frame as the begin state, not this step's frames.
                    _ord = exec_t.detach().clone()
                    _ord[:, self.GAZE_IDX] = 0.0
                    _ord[:, self.GAZE_IDX, 6] = 1.0
                    self._tenure_pending = {
                        "memory": self._nf_memory,
                        "tokens": {m: v.detach() for m, v in self._nf_tokens.items()},
                        "ctx": (None if self._nf_ctx is None else self._nf_ctx.detach()),
                        "ach": (self._nf_ach if self._nf_ach is not None
                                else torch.zeros(B, len(self.handle_names), 10, device=d)),
                        "order": _ord,
                        "raw": (self._nf_raw if self.tenure_token_reach else None),
                        "mask": (dict(self._nf_mask) if self.tenure_token_reach
                                 and self._nf_mask is not None else None),
                        "hand_vis": (self._nf_handvis
                                     if self._nf_handvis is not None
                                     and self._nf_handvis.shape[0] == B
                                     else torch.zeros(B, dtype=torch.bool, device=d))}
                    self._tenure_prev_order = exec_t.detach().clone()
                    self._tenure_prev_order_id = (None if executed_order_id is None
                                                  else int(executed_order_id))
            if self.consequence_lesson is not None:
                contact_loss = self.consequence_lesson.update(
                    self, frames, frame, source_mask, self.context_buffer, exec_t, train, drive)
                if contact_loss is not None:
                    loss = loss + contact_loss
                    self._probe_term("contact_consequences", contact_loss)
                bd["contact_consequences"] = dict(self.consequence_lesson.stats)
            if nf_memory_state is not None and exec_t is not None:
                predicted_needs = self.persistent_memory.predict_needs(nf_memory_state,command_for_encoder(exec_t))
                valid_needs = self._memory_needs_valid.float()
                memory_loss = ((predicted_needs-current_needs).square()*valid_needs).sum()/valid_needs.sum().clamp_min(1)
                loss = loss + .1*memory_loss
                if self.lesson_probe is not None:
                    self._probe_term("persistent_memory", .1 * memory_loss)
                self.last_memory['next_bodily_state_error'] = float(memory_loss.detach())
                # ONLY the loss term goes in the breakdown - the next-bodily-state needs
                # prediction, .1 x its MSE, the weight at which it enters the sum above.
                # The other four last_memory fields are diagnostics - a count, an age in
                # seconds, a bool and a token magnitude - and the viewer draws every
                # breakdown entry as a lesson share, so on H81 step_380 a count of 39
                # remembered events became the largest bar on the teaching-signal panel
                # and set the scale every real lesson was measured against. They are
                # already shown, with their own labels, in the memory panel.
                bd['persistent_memory'] = .1 * memory_loss.detach()
            if self_state_loss is not None:
                loss = loss + self_state_loss
                self._probe_term("self_state", self_state_loss)
                bd['self_state_loss'] = self_state_loss.detach()
                # A completed internal state becomes available on the next tick.
                self.self_history.internal = last_step.detach().float().mean(1)
            # review M1: the agency motion gate (below, in the intention section)
            # needs the PREVIOUS step's pose; the refresh just below overwrites
            # _nf_ach with THIS step's. Captured here, before the refresh.
            _ach_prev = self._nf_ach
            if self.next_frame_loss:
                self._nf_memory = memory_snapshot
                self._nf_tokens = {m: tokens_raw[m].detach() for m in tokens_raw}
                self._nf_raw = {k: (v if not torch.is_tensor(v) else v.detach())
                                for k, v in frame.items()}
                if isinstance(self._nf_raw.get("vision"), dict):
                    self._nf_raw["vision"] = {k2: v2.detach() for k2, v2 in self._nf_raw["vision"].items()}
                self._nf_mask = dict(source_mask)
                self._nf_ctx = self.context_buffer.detach().clone()
                # the pose that belongs to the SAME moment as these caches - captured
                # before next step's refresh, so tenure pose windows align with token
                # windows (review R3: they were each one step late)
                self._nf_ach = (self.last_achieved.detach().clone()
                                if self.last_achieved is not None else None)
                # review M13: the visibility flags of THIS frame, cached beside the
                # frame itself, so next step's tenure gate pairs flags with the frames
                # the delta is actually measured on
                if self.vision_gate_mode == "self-motion":
                    self._nf_handvis = torch.tensor(
                        [bool((f.meta.get("task") or {}).get("hand_in_view"))
                         or (float((f.meta.get("task") or {}).get("mobile_speed_cms") or 0.0)
                             > self._selfmotion_floor_cms)
                         for f in frames], device=d)
                else:
                    self._nf_handvis = torch.tensor(
                        [bool((f.meta.get("task") or {}).get("hand_in_view"))
                         for f in frames], device=d)
            # refresh the lag caches for NEXT step's grading (after all consumers)
            self._lag_lat = latents.detach()
            self._lag_last = last_step.detach()
            self._lag_hs = handle_state.detach()
            self._lag_wg = None if wellness_grad is None else wellness_grad.detach()
            # explainer stashes. last_context BEFORE the shift below rebinds
            # context_buffer; `tokens` here is what the forward at this step's core
            # call consumed (gate-blended when the gate opened). References only.
            self.last_context = self.context_buffer
            self.last_tokens_fed = {m: tokens[m].detach() for m in tokens}
            self.last_predictions = {m: v.detach() for m, v in predictions.items()}
            self.last_wellness_grounded = {k: v.detach()
                                           for k, v in wellness_out["grounded"].items()}
            self.last_voice = None if voice_out is None else voice_out.detach()
            if self._hself is not None:
                # dark observation only: raw tokens + measured wellness in, nothing out.
                # FP32: this tracker reports cosines and correlations to five decimals
                # and compares them against fixed lines (novelty 0.9), so it does not
                # inherit the transformer's bf16 autocast. Measured effect of the old
                # regime over 6,400 unit-norm 768-wide slots: 0.00054 maximum absolute
                # cosine error against a decision line the real frames sat 0.21 or more
                # away from, so no admission decision changed - this is precision
                # hygiene for a reported statistic, not a repair of a wrong result.
                with torch.amp.autocast("cuda", enabled=False):
                    self._hself.observe(self.last_tokens, wellness_out["grounded"])
            if self.entity_self_mode != "off":
                # the same kind of watching over NAMED things, whose presence the world
                # reports rather than a likeness between token patterns
                self._observe_entity_self(frames, wellness_out)
        timings["forward"] = (time.perf_counter() - t0) * 1000 - timings["encode"]
        grad_norm = 0.0
        if train:
            # review T7: one non-finite loss used to reach optimizer.step(), turn every
            # parameter NaN, and the run kept stepping while the hourly save archived
            # the poisoned weights (keep_last_n then deletes the last healthy one).
            # The trainer only counted nan_steps. Skip the step loudly instead.
            if not bool(torch.isfinite(loss.detach())):
                self._nonfinite_steps += 1
                log.error("non-finite loss at step %d - optimizer step skipped (#%d)",
                          step, self._nonfinite_steps)
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad(set_to_none=True)
                if self.lesson_probe is not None:
                    # a measurement's probe (docs/gain_of_function_v4/proprio_token_measurements.py grade):
                    # every term of this step, each with its graph, before the backward frees it
                    self.lesson_probe.before_backward(self, loss)
                loss.backward()
                grad_norm = self._clip_training_gradients()
                if self.lesson_probe is not None:
                    self.lesson_probe.after_clip(self, grad_norm)
                if not math.isfinite(grad_norm):
                    self._nonfinite_steps += 1
                    log.error("non-finite grad norm at step %d - optimizer step skipped (#%d)",
                              step, self._nonfinite_steps)
                    self.optimizer.zero_grad(set_to_none=True)
                    grad_norm = 0.0
                else:
                    if self._pw_pending is not None:
                        # the within-body lesson, after the clip (--proprio-token-repair within_body)
                        _reg_pw = getattr(self.encoder_registry, "_orig_mod", self.encoder_registry)
                        self.proprio_within.apply(self._pw_pending,
                                                  list(_reg_pw.encoders["proprioception"].parameters()),
                                                  self.optimizer)
                    if mv is not None and self.movement_param_mod:
                        for gi, group in enumerate(self.optimizer.param_groups):
                            group["lr"] = self._base_lrs[gi] * mv["lr_mult"]
                    self.optimizer.step()
            self._pw_pending = None
            # the expanded-embedding cache is stale the moment the weights move; the
            # method to say so existed and was called by nobody, so no_grad frames read
            # embeddings from before the update until the next gradient-enabled forward
            _fm_inv = getattr(self.forward_model, "_orig_mod", self.forward_model)
            if hasattr(_fm_inv, "invalidate_embedding_cache"):
                _fm_inv.invalidate_embedding_cache()
            if self.persistent_memory is not None and self.persistent_memory.repaired:
                # --persistent-memory-repair: one model update ended; the constant's return schedule moves
                self.persistent_memory.advance_constant_return()
        timings["backward"] = (time.perf_counter() - t0) * 1000 - timings["encode"] - timings["forward"]
        # context update
        if self.ctx_len > 1:
            self.context_buffer = torch.cat([self.context_buffer[:, self.n_mod:, :], ctx_update.detach()], dim=1)
        self.step_count += 1
        # body schema update: correlate this step's commands with the proprioception of every
        # body in the batch (cheap EMA update; live_test does the single-env equivalent)
        with torch.no_grad():
            # AUDIT FIX: one shared schema mixed every body's motion into a single
            # prev-state buffer (cross-env deltas) - each env now has its own schema
            while len(self._schemas) < B:
                self._schemas.append(DifferentiableBodySchema(
                    n_handles=len(self.handle_names), object_names=SELF_OBJECTS).to(self.device))
            # DISABLED 2026-08-25 (plan item 7). The statistic correlates the MAGNITUDE of
            # the frame-to-frame command change with the magnitude of the proprioceptive
            # change. Under motor_hold=5 the command delta is ~0 on four steps out of five
            # while the body keeps moving, and the strongest response can arrive several
            # frames after the command changed. It has no lag bank, no sign, and no
            # passive/actuator-disabled comparator, so what it measures is not self-causation.
            # It also starts at membership 0.076 / coherence 0.848 with zero evidence, which
            # reads as a confident "not self" that has not been earned. Self-membership is
            # reported as unknown until it beats the shuffled, cross-body and passive
            # controls. Set body_schema_updates=True to re-enable for development.
            if self.body_schema_updates:
                for b in range(B):
                    pro = frames[b].tokenized.proprio
                    if pro is not None and pro.shape[0] >= 1:
                        act_b = exec_t[b] if exec_t is not None else motor_out[b].detach()
                        # UEMA 11.2: the toy and the mobile are watched the way a handle is.
                        # A toy the hand is holding travels with every arm command and its
                        # score climbs - the self extending; let go, the score falls back at
                        # the map's own rate, which is the self contracting.
                        _pos, _here = object_state_for(frames[b].meta)
                        self._object_physics_driven = object_physics_driven_for(frames[b].meta)
                        # WHERE THE HANDLES ARE, IN WORLD METRES, so a handle's movement and
                        # a thing's movement are one unit in one frame. The achieved-pose half
                        # of the motor row carries world centimetres divided by 200 (the
                        # contract assay checks motor_handles[:, :3] * 200 == bones world cm),
                        # so 200 cm / 100 = 2 metres per unit. Proprioception cannot be used:
                        # it is measured from the left wrist and divided by the same 200.
                        _mh = frames[b].tokenized.motor_handles
                        _hp = (None if _mh is None else
                               _mh[:, :3].to(self.device).float() * HANDLE_POS_TO_METRES)
                        self._schemas[b].update(
                            act_b, pro.to(self.device),
                            object_state=torch.tensor(_pos, device=self.device, dtype=torch.float32),
                            object_present=torch.tensor(_here, device=self.device, dtype=torch.float32),
                            handle_world_pos=_hp)
        # intention: every Nth step, plan toward the best-feeling remembered moment and
        # (if executing) let the plan's first action drive the body instead of the reflex.
        intent_action = None
        # WHO AUTHORED THE ORDER, stamped where it is authored. This used to be inferred
        # after the fact from "is intent_action non-None", and SelfIntention.choose()
        # writes into that same variable - so an order the self-model's candidate
        # evaluation produced was tallied as "planner" in exec_src, the very tally the
        # note below cites as evidence that babble never acted. Initialised here, before
        # any branch, so the label can never be left over from the previous step.
        self.last_action_source = "reflex"
        if self.intention_gen is not None:
            with torch.no_grad():
                wball = torch.cat(list(wellness_out["grounded"].values()), dim=-1).sum(-1)   # [B]
                _w0 = wball.detach().float().clone()   # bodily value BEFORE drives
                if mv is not None:
                    wball = wball + mv["wellness"]         # moving memories feel better
                if self.agency_drive:
                    ag = (self.agency_weight * self._agency_pay_ema
                          if self._agency_pay_ema is not None
                          else torch.zeros(B, device=self.device))
                    # hand-watching bootstrap: pays only for a MOVING hand in the
                    # fovea, and fades as vision's live agency evidence reaches
                    # touch's (displacement, not a clock)
                    # review M19: a missing key means the flag was never computed
                    # (env exception path, or a rig that fills no task meta) - the
                    # bootstrap goes dark; the count makes that visible in the log.
                    _fov_missing = sum(1 for f in frames
                                       if "hand_in_fovea" not in (f.meta.get("task") or {}))
                    fov = torch.tensor([1.0 if (f.meta.get("task") or {}).get("hand_in_fovea")
                                        else 0.0 for f in frames], device=self.device)
                    # review M1 (2026-09-02, second relocation): R2 moved the read from
                    # last_achieved to _nf_ach - but the delayed-lesson refresh above
                    # had ALREADY overwritten _nf_ach with this step's pose, so motion
                    # still measured 0.0 forever (probe: the EMA followed the
                    # zero-decay curve to nine digits over 41 steps). _ach_prev is
                    # captured before the refresh. review M18: per-BODY motion over the
                    # HAND handles (0..11) only - a cross-body all-handle mean let a
                    # still body in a moving batch collect the prior with a parked
                    # hand, the degenerate equilibrium the gate exists to block.
                    if _ach_prev is not None and _ach_prev.shape[0] == B:
                        hm_vec = (handle_state[:, :12, :3].detach()
                                  - _ach_prev[:, :12, :3]).norm(dim=-1).mean(dim=1)   # [B]
                    else:
                        hm_vec = torch.zeros(B, device=self.device)
                    hm_now = float(hm_vec.mean())
                    self._hand_motion_ema = 0.99 * self._hand_motion_ema + 0.01 * max(hm_now, 1e-6)
                    motion = (hm_vec / (2 * self._hand_motion_ema)).clamp(max=1.0)     # [B]
                    # review M16a/T3: the raw ratio was a step function on the noise
                    # SIGNS of two EMAs (measured on both rails live: H60 end 0.0, H61
                    # start 0.05). Both advantages clamp at 0 and the anchor floors at
                    # the measured advantage-noise scale (_agency_adv_floor).
                    _va = max(self._vis_adv_ema, 0.0)
                    _ta = max(self._touch_adv_ema, 0.0)
                    displaced = min(1.0, max(0.0, 1.0 - _va / max(_ta, self._agency_adv_floor)))
                    w_prior = self.agency_prior_w0 * displaced
                    ag = ag + w_prior * motion * fov
                    wball = wball + ag
                    self._last_agency_vec = ag.detach()
                    self.last_agency = {"pay": float(ag.mean()),
                                        "prior_w": float(w_prior),
                                        "vis_adv": float(self._vis_adv_ema),
                                        "vis_adv_elig": float(self._vis_adv_elig_ema),
                                        "vis_adv_null": float(self._vis_adv_null_ema),
                                        "touch_adv": float(self._touch_adv_ema),
                                        "motion": float(motion.mean()),
                                        "fovea": float(fov.mean()),
                                        "fov_missing": float(_fov_missing)}
                if self.consequence_weight > 0.0:
                    # CONSEQUENCE ACCESS (2026-09-18, tree revision W2/W1). Until now the goal
                    # ring held only moments the body had lived, ranked by five terms that read
                    # no contact, no moved object and no gained view; a first touch of the
                    # mobile scored no higher than lying still, so it could never become the
                    # goal (H83: mobile struck by a hand on 0.0 of steps over 13,600 steps).
                    # Per body, one unit per KIND of self-relevant consequence the world
                    # measured this moment (consequence_kinds), times the launch weight.
                    # Weight measured against H83's own ring: stored values spanned
                    # 3.993-4.410 (a 0.42 spread), so 0.5 per kind lets one consequence
                    # moment outrank every still moment in the ring; whether the planner then
                    # picks it is read directly off last_consequence['goal_rate'] below.
                    cq = torch.tensor([consequence_kinds(f.meta.get("task") or {},
                                                         world_only=self.consequence_world_only)
                                       for f in frames],
                                      device=self.device, dtype=torch.float32)          # [B]
                    wball = wball + self.consequence_weight * cq
                    self.last_consequence.update(kinds_mean=float(cq.mean()),
                                                 rate=float((cq > 0).float().mean()),
                                                 weight=self.consequence_weight)
                    if self.consequence_world_only:
                        # what the rule removed, read live: the same moments under the old
                        # count-every-press rule, and how many reports lacked the split key
                        _cq_old = torch.tensor([consequence_kinds(f.meta.get("task") or {})
                                                for f in frames], dtype=torch.float32)
                        self.last_consequence.update(
                            world_only=1.0, kinds_mean_count_rule=float(_cq_old.mean()),
                            rate_count_rule=float((_cq_old > 0).float().mean()),
                            world_key_missing=float(sum(
                                1 for f in frames
                                if "hand_pressing_world" not in (f.meta.get("task") or {}))))
                else:
                    cq = torch.zeros(B, device=self.device)
                if self._self_value is not None and self._entity_presence is not None:
                    # THE READER (UEMA 11.4's "the gradient maintains them", made narrow). A
                    # remembered moment is worth more when a thing with a PROVEN link to the
                    # body's wellness was present in it, so the planner aims at such moments
                    # and works to keep the thing. Exactly zero until the reader's own gate
                    # opens, so a run with it wired in but unproven values moments as before.
                    wball = wball + self._self_value.value(self._entity_presence)
                if self.curiosity_goal and self.curiosity_measured:
                    # decision-time curiosity (B10): this moment's felt value AS A GOAL
                    # is scaled by the goldilocks drive. THE SAME drive the stat, the
                    # explainer and the consequence lesson read - this block used to
                    # recompute its own copy from the same dict with the same filter, so
                    # the two agreed on every step except the ones where no sense was
                    # graded real, which is exactly where they mattered. The delayed
                    # errors are graded EARLIER in this same step() call (review M21, see
                    # the note above the grading), so neither is one step stale; with
                    # --next-frame-loss OFF the delayed lesson does not run and
                    # last_next_pe carries nothing to read.
                    if self.curiosity_mode == "additive":
                        # --curiosity-mode additive (DESIGN D8): the drive ADDS its table weight
                        # times itself (0.25 x a drive of 0 to 1) instead of multiplying the whole
                        # felt value; measured on segment 31 the multiplier decided the goal in 5
                        # of 5 snapshots (understand-wellness-planner.md section 3)
                        wball = wball + float(self.rank_weights.get("curiosity", 0.0)) * float(drive)
                    else:
                        wball = wball * (1.0 + float(drive))
                _chv = None
                if self.wellness_rank_mode == "normalized":
                    # THIS MOMENT'S CHANNELS (DESIGN section 4), kept beside the raw value so the
                    # goal choice can rank them normalised (lem/model/wellness_channels.py); the
                    # ring stores each as its distance from the body's running average
                    _chv = self._rank_store_moment(B, wellness_out, extras, frame, mv, cq, drive,
                                                   frames)
                from lem.model.physical_goal import align_pose_memory
                align_pose_memory(self._goal_ring_pose, len(self.goal_ring))
                self._goal_ring_pose.append(handle_state.detach().float().clone())
                self.goal_ring.append(({m: tokens[m].squeeze(1).detach().float() for m in tokens},
                                       wball.detach().float()))
                self._goal_ring_w0.append(_w0)
                self._goal_ring_cq.append(cq.detach().float())
                # the MULTIPLIER per stored moment (the explainer divides by 1 + this); under the
                # additive mode nothing was multiplied, so 0.0
                self._goal_ring_drive.append(float(drive) if (self.curiosity_goal and self.curiosity_measured
                                                              and self.curiosity_mode == "multiply") else 0.0)
                if _chv is not None:
                    self._goal_ring_ch.append(_chv.detach().float())
                if self.consequence_lesson is not None:
                    self.consequence_lesson.record_goal(drive)
            if len(self.goal_ring) >= 20 and step % self.intention_every == 0:
                self.last_intention = None      # never let an old plan masquerade as this step's
                wl_stack = torch.stack([w for _, w in self.goal_ring])          # [T, B]
                if self.consequence_lesson is not None:
                    wl_stack = self.consequence_lesson.revalue_goals(drive)
                    if wl_stack.shape[0] != len(self.goal_ring):
                        raise RuntimeError("bodily memory values lost alignment with sensory memories")
                elif self.goal_revalue:
                    wl_stack = self._revalue_ring(wl_stack)
                _rank_info = None
                if self.wellness_rank_mode == "normalized":
                    # the goal is the best NORMALISED, prioritised moment, never one over the harm
                    # line while any moment is under it (DESIGN section 4); the stored raw values
                    # stay in the ring untouched
                    wl_stack, _rank_info = self._normalized_goal_values(wl_stack)
                best_t = wl_stack.argmax(dim=0)                                  # [B]
                if len(self._goal_ring_cq) == len(self.goal_ring):
                    # the direct reading of the consequence term: did a consequence moment WIN
                    _cq_win = torch.stack(list(self._goal_ring_cq)).gather(0, best_t.view(1, -1)).squeeze(0)
                    self.last_consequence.update(goal_rate=float((_cq_win > 0).float().mean()))
                current_state = {m: tokens[m].squeeze(1).detach().float() for m in self.active}
                tok_stacks = {m: torch.stack([t[m] for t, _ in self.goal_ring])
                              for m in self.active}                              # [T, B, d]
                _gl_info, goal_state = None, None
                if self.goal_locality:
                    goal_state, _gl_info = self._gl_select(
                        current_state, wl_stack, tok_stacks, best_t, step,
                        allowed=(torch.isfinite(wl_stack) if _rank_info is not None else None))
                _goal_where = None
                if goal_state is not None and getattr(self, "_gl_goal_index", None) is not None:
                    _goal_where = (self._gl_goal_index, self._gl_goal_age)
                if goal_state is None:   # locality off, or its reach measure warming
                    goal_state = {m: tok_stacks[m].gather(
                        0, best_t.view(1, -1, 1).expand(1, -1, tok_stacks[m].shape[-1])
                    ).squeeze(0) for m in self.active}
                # WHERE THE GOAL SAT IN THE RING (the b3 review's blocker, read directly): ring
                # index (0 = oldest; -1 = a goal held by goal locality whose moment has left the
                # ring) and its age in model steps, per body
                _goal_idx, _goal_age = self._goal_position(best_t, _goal_where)
                if _rank_info is not None:
                    self.wellness_channels.shares.add_goal(
                        [max(0, i) for i in _goal_idx], len(self.goal_ring))
                # live_test's reference call is wrapped in no_grad and never fires (eval
                # gate); the generator NEEDS gradients for its Adam loop - enable_grad here
                if self.plan_horizon_mode == "earned":
                    _lic = 0
                    for _w in (self._plan_win_ema or []):
                        if _w > 0.5:
                            _lic += 1
                        else:
                            break
                    _max_steps = max(1, min(1 + _lic, self.imagination_steps))
                else:
                    _max_steps = int(self.config["training"]["intention"]["max_plan_steps"])
                # paired-A/B hook (09-09 review requirement 3): the exact inputs of
                # this decision, so replan_last() can re-run it under a different
                # prospective-wellness weight from IDENTICAL state
                self._last_plan_inputs = {
                    "cur": {m: v.detach().clone() for m, v in current_state.items()},
                    "goal": {m: v.detach().clone() for m, v in goal_state.items()},
                    "motor": motor_out.detach().float().clone(),
                    "max_steps": _max_steps,
                    "decision": self._capture_plan_state()}
                if self.physical_goals == 'memory' or self.hand_goals == 'memory':
                    from lem.model.physical_goal import align_pose_memory, selected_pose
                    align_pose_memory(self._goal_ring_pose, len(self.goal_ring))
                    pose, valid = selected_pose(self._goal_ring_pose, _goal_idx, handle_state)
                    physical_memory = {
                        'pose': pose, 'valid': valid, 'ring_index': list(_goal_idx),
                        'horizon_ms': 1000. * self.step_seconds * self.efference_tenure_steps}
                    if self.physical_goals == 'memory':
                        self._last_plan_inputs['decision']['physical_goal'] = physical_memory
                    if self.hand_goals == 'memory':
                        self._last_plan_inputs['decision']['hand_goal'] = physical_memory
                with torch.enable_grad():
                    plan, imeta = self._plan_chain(
                        current_state, goal_state, motor_out.detach().float(),
                        max_steps=_max_steps, decision=self._last_plan_inputs['decision'])
                self._last_plan_inputs["outcome"] = {
                    key: imeta.get(key) for key in
                    ("plan_length", "stopped_by", "converged", "scale_exit")}
                self._last_plan_inputs["plan0"] = (
                    plan[0].detach().float().clone() if plan else None)
                self._hand_chosen_plan = imeta.get('hand_joint_goal_raw') if plan else None
                if self.synergy is not None and self.synergy_planner and plan:
                    # the first action's intention: how long z ended, and the residual beside it
                    _ds = ((imeta.get("descent") or [{}])[0].get("synergy") or {})
                    if _ds.get("z_norm") is not None:
                        self._syn_add("plan_z_norm", _ds["z_norm"])
                    if _ds.get("residual_norm") is not None:
                        self._syn_add("residual_norm", _ds["residual_norm"])
                if self.causal_chain:
                    # the planning event's record (goal, prediction, predicted wellness for
                    # the chosen plan and its alternatives from the same captured state).
                    # Instrumentation must not end a run: a failure is counted into the
                    # causal_chain block (failures) and logged, never silent.
                    self.last_plan_record = None
                    if plan:
                        try:
                            self.last_plan_record = self._causal_plan_record(
                                step=step, best_t=best_t, wl_stack=wl_stack,
                                goal_state=goal_state, motor_out=motor_out,
                                plan0=plan[0].detach().float(), imeta=imeta,
                                decision=self._last_plan_inputs["decision"],
                                exec_t=exec_t, max_steps=_max_steps,
                                locality=_gl_info is not None, rank_info=_rank_info)
                            self.newest_plan_record = self.last_plan_record
                        except Exception:
                            self._causal_failures += 1
                            log.warning("causal chain: the planning record at step %d failed "
                                        "(%d so far): %s", step, self._causal_failures,
                                        traceback_short())
                if plan:
                    if self.intention_execute:
                        intent_action = plan[0].detach().float()
                        self.last_action_source = "planner"
                        # the plan's gaze rides inside intent_action as handle 23; the
                        # ordinary extraction below the executor picks it up
                        gaze_out = self._gaze_from_handle(intent_action.to(self.device))
                        if self.goal_gaze_hold:
                            self._gaze_hold = (gaze_out,
                                               step + self.intention_every)
                    _sm = float(imeta.get("start_mse", float("nan")))
                    _fm = float(imeta.get("final_mse", float("nan")))
                    self.last_intention = {"plan_length": imeta.get("plan_length"),
                                           "descent": imeta.get("descent"),
                                           "gaze_descent": imeta.get("gaze_descent"),
                                           "hand_descent": imeta.get("hand_descent"),
                                           "horizon": _max_steps,
                                           "converged": bool(imeta.get("converged")),
                                           "final_mse": _fm, "start_mse": _sm,
                                           # P125: every depth's distance and the rule that stopped
                                           # the rollout, so the page can say more than start == final
                                           "mse_by_depth": imeta.get("mse_by_depth"),
                                           "stopped_by": imeta.get("stopped_by"),
                                           "closer_than_start": bool(_fm < _sm),
                                           # the operator's scale: 0 = the plan never
                                           # moved toward its goal, 100 = it arrived -
                                           # and it measures the IMAGINED rollout's
                                           # convergence, never the executed action's
                                           # effect. At horizon 1 the loop body runs once
                                           # and start_mse and final_mse are the same
                                           # first-iteration number, the distance BEFORE
                                           # any action existed, so the ratio is 0 by
                                           # construction and measures nothing: None
                                           # rather than a measured-looking 0.0. Measured
                                           # on H81: all five horizon-1 windows read
                                           # exactly 0.0.
                                           "gap_closed_pct": (100.0 * (1.0 - _fm / _sm)
                                                              if _sm == _sm and _sm > 1e-9
                                                              and _fm == _fm
                                                              and _max_steps > 1 else None),
                                           "goal_locality": _gl_info,
                                           # THE LAUNCH FLAG, not whether this plan
                                           # reached a limb. A plan is computed on
                                           # step % intention_every == 0 and reaches the
                                           # body only when the trainer latches on
                                           # step % motor_hold == 0; at H82/H83's
                                           # intention_every 7 with motor_hold 5 that is
                                           # 29 of 143 plans in 1,000 steps (20.3%),
                                           # while the old field name "executed" read
                                           # True on all 143. reached_body stays None
                                           # until the trainer stamps it at the latch.
                                           "intention_execute_flag": bool(self.intention_execute),
                                           "reached_body": None, "step": step,
                                           # the goal the plan aimed at: ring index per body (0 =
                                           # oldest of the ring's moments) and its age in steps
                                           "goal_ring_index": _goal_idx,
                                           "goal_age_steps": _goal_age}
        # action for the body: developmental constraints + exploration noise (like live_test)
        if self.self_intention is not None:
            if step % int(self.self_intention.receipt['order_steps']) == 0:
                selected = self.self_intention.choose(self,
                    motor_out.detach().float() if intent_action is None else intent_action,
                    dev_limits,handle_state,step,frames)
                self.last_self_state['choice'] = self.self_intention.last
                if selected is not None:
                    intent_action = selected
                    # the self-model's candidate evaluation authored this order, not the
                    # Adam planner
                    self.last_action_source = "self_choice"
                    gaze_out = self._gaze_from_handle(selected)
        with torch.no_grad():
            src_cmd = motor_out if intent_action is None else intent_action
            # last_action_source was set at each authoring site above; the trainer
            # tallies it over LATCHED (executed) steps, because with motor_hold ==
            # intention_every every executed order was a planner-step order and babble
            # structurally never acted (measured 181/181 on H81, 2026-09-14). The tally
            # can now carry a fourth key, "self_choice"; train_mimo's write creates a
            # bucket on first use.
            mfa = self.execution_transform(src_cmd, dev_limits)
            # review fix 3: apply_developmental_constraints scales WORLD positions by
            # rom_limit, dragging every target toward the world origin (~the floor) in any
            # phase with rom<1. Redo positions body-relative: full 200 cm denorm, then clamp
            # the STEP (command minus current pose) to the phase's reach fraction of ~40 cm
            # PER AXIS. torch.clamp with scalar bounds is elementwise, so x, y and z are
            # each bounded independently and the largest permitted single-step
            # displacement is 40*sqrt(3) = 69.28 cm at rom_limit 1.0, not 40 cm. Any line
            # that quotes 40 cm as a DISTANCE bound is quoting a number 1.73x too small.
            cur_cm = handle_state[:, :, 0:3].detach().float() * 200.0
            des_cm = src_cmd.detach().float()[:, :, 0:3].clamp(-1.0, 1.0) * 200.0
            reach = 40.0 * float(dev_limits.get("rom_limit", 1.0))
            mfa[..., 0:3] = cur_cm + (des_cm - cur_cm).clamp(-reach, reach)
            if intent_action is None and self.wellness_mode != WellnessMode.DIRECT:
                ns = dev_limits.get("exploration_noise", 0.0)
                if mv is not None and self.movement_param_mod:
                    ns = ns * mv["noise_mult"]             # Stage 3: stillness -> fidgeting
                synergy_babbled = False
                if ns > 0 and self.synergy_babble:
                    # --synergy-babble on (DESIGN D11): the noise is ONE coordinated whole-body
                    # change drawn in the synergy's intention space, sized so each body handle
                    # turns as far on average as the independent wobble below turns it
                    self.last_action_source = "reflex_noise"
                    coordinated = self._synergy_babble(mfa, handle_state, ns, cur_cm, reach)
                    if coordinated is not None:
                        mfa = coordinated
                        mfa[..., 8].clamp_(0.0, dev_limits["strength_limit"])
                        synergy_babbled = True
                if ns > 0 and not synergy_babbled:
                    self.last_action_source = "reflex_noise"
                    # live_test adds `ns` to the cm positions and strength. MIMo's decoder reads
                    # ORIENTATION, so we also wobble the target quaternion by a small random
                    # rotation with per-axis std = 10 * ns rad. A guess, written down; it is
                    # what makes "motor babbling" visible on MIMo. Since the babble floor
                    # (curriculum.py BABBLE_FLOOR = 0.05), ns is 0.05 in EVERY phase, so that
                    # std is 0.50 rad (29 degrees) everywhere - not the 0.1 rad this comment
                    # used to quote for neonate.
                    mfa = self._independent_babble(mfa, ns, cur_cm, reach,
                                                   dev_limits["strength_limit"])
                    # THE EYES GET NO BABBLE FROM THIS. The wobble above also turns handle 23's
                    # row, but the eyes are driven by gaze_out, which was taken from motor_out
                    # (or the plan) BEFORE this block, and the trainer's ledger then overwrites
                    # row 23 of the executed order with the eyes' realised angle. So the wobble on
                    # row 23 reaches neither the eye servo nor the model. The eyes' own random
                    # movement is the body-side saccade generator (gaze_reflex.py). This comment
                    # used to say the wobble IS the eyes' babble (understand-motor-path-energy.md
                    # section 5); it never was.
            if random_actions:
                # random-action baseline (E4): random quats, random strength inside the limit
                q = torch.randn(mfa.shape[0], mfa.shape[1], 4, generator=self.noise_rng)
                mfa[..., 3:7] = (q / q.norm(dim=-1, keepdim=True)).to(mfa.device)
                mfa[..., 8] = torch.rand(mfa.shape[0], mfa.shape[1], generator=self.noise_rng).to(mfa.device) * dev_limits["strength_limit"]
        hand_joints = None
        if self.hand_target_mode != 'world':
            if self.hand_kinematics is None:
                raise RuntimeError('Hand geometry must be configured before choosing actions')
            if intent_action is not None and self.last_action_source == 'planner' and self._hand_chosen_plan is not None:
                hand_joints = self._hand_chosen_plan.to(mfa)
            else:
                _, joint_array, _ = self.hand_kinematics.project_world_orders(mfa.cpu().numpy(), position_scale=1.)
                hand_joints = torch.as_tensor(joint_array, device=mfa.device, dtype=mfa.dtype)
            hand_joints = self._smooth_hand_joints(hand_joints, self._hand_previous_joints, self.smooth_actions)
        if self.smooth_actions > 0.0 and self.smooth_source == "executed":
            # --smooth-source executed (DESIGN D11): the new order is blended with the order the
            # body was really given over the interval that just ended (exec_t, the trainer's
            # ledger), not with a running average of this model's proposals - four of every five
            # of which were never executed (the order is latched every --motor-hold moments). The
            # eyes' row is passed through: the eyes are sent the new order's gaze.
            with torch.no_grad():
                _sp = self._smooth_exec_prev
                self._last_smooth_prev = (_sp.to(mfa.device).clone()
                                          if _sp is not None and _sp.shape == mfa.shape else None)
                mfa = self.smooth_blend(mfa, self._last_smooth_prev, self.smooth_actions,
                                        keep_row=self.GAZE_IDX)
        elif self.smooth_actions > 0.0:
            with torch.no_grad():
                # stash the smoother's input state BEFORE updating it, so an offline
                # probe can put a replanned command through the IDENTICAL executed-order
                # pipeline (09-09 follow-up review: paired branches skipped smoothing,
                # so the compared orders were not the orders the body would receive)
                self._last_smooth_prev = (self._prev_mfa.clone()
                                          if self._prev_mfa is not None
                                          and self._prev_mfa.shape == mfa.shape else None)
                mfa = self.smooth_blend(mfa, self._last_smooth_prev, self.smooth_actions)
                self._prev_mfa = mfa.clone()
        else:
            self._last_smooth_prev = None
        if hand_joints is not None:
            mfa = self.hand_kinematics.assemble_world(mfa, hand_joints, position_scale=1.)
            if self.smooth_source != 'executed':
                self._hand_previous_joints = hand_joints.detach().clone()
                self._prev_mfa = mfa.detach().clone()
        if self.executed_fields_only_on:
            # --executed-fields-only on (motor interface audit 2026-09-25, F4): the order the ledger
            # records is the order the decoder reads - the achieved position on every row whose
            # position the decoder ignores, radius 0 - formed LAST, after the smoother and the hand
            # assembly, as planned_execution forms it for the planner. The smoother's memory keeps
            # this form wherever it kept the order above, so the next blend starts from it. The rows
            # kept are the decoder's live position rows under the learner's position_rows (the
            # decoder's own setting, train_mimo --position-rows; 'all' when the learner has none).
            with torch.no_grad():
                mfa = self.executed_fields_only(mfa, handle_state,
                                                getattr(self, "position_rows", "all") or "all")
            if self.smooth_source != "executed" and (self.smooth_actions > 0.0 or hand_joints is not None):
                self._prev_mfa = mfa.detach().clone()
        if self.synergy is not None:
            # the turn this moment's order asks of the body handles at the moment it is formed (the
            # environment's cmd_mag at issue), by author: the direct check that babble in z asks as
            # much as the independent wobble did
            with torch.no_grad():
                _ask = float(self.synergy.asked_turn_rad(mfa, handle_state).mean())
            self._syn_add("cmd_mag_all", _ask)
            self._syn_add("cmd_mag_" + str(self.last_action_source), _ask)
        def _f(x):
            return float(x.detach().float().mean().item()) if torch.is_tensor(x) else float(x)
        bdf = {}
        for k, v in (bd or {}).items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    bdf[f"{k}.{kk}"] = _f(vv)
            else:
                bdf[k] = _f(v)
        wl = {k: _f(v) for k, v in wellness_out["grounded"].items()}
        # How far the approximate scorer is from the grounded one it is trained to copy.
        # This is the number that decides whether imagination can be trusted: planning and
        # the wellness safety exit both score IMAGINED states with the approximate
        # evaluator, and it is only worth listening to once it agrees with the authority.
        # Measured on the H17 brain the day the grounded formulas landed: 0.277 mean
        # absolute error on a 0-1 scale, off by 0.5 on capability and coherence, which is
        # why _plan_chain still skips the safety exit in baseline mode.
        # ONLY THE TERMS THE APPROXIMATE EVALUATOR PREDICTS. It has one output per name in
        # APPROXIMATE_TERMS and is trained to copy exactly those; a grounded term added since
        # (self_touch is the first) is not among them. Taking every grounded value made this a
        # 6-wide minus 5-wide subtraction, which raised, and the bare except turned the gauge
        # off without a word: it read None on all 49 records of the live run against 0.0134
        # and 0.0174 on the two segments before self_touch existed.
        align_err = None
        ap = wellness_out.get("approximate")
        if ap is not None:
            grounded = wellness_out["grounded"]
            cols = [grounded[n] for n in APPROXIMATE_TERMS if n in grounded]
            if len(cols) == int(ap.shape[-1]):
                align_err = _f((ap.float() - torch.cat(cols, dim=-1).float()).abs().mean())
            else:
                self.last_align_err_skipped = (
                    "the approximate evaluator predicts %d terms and %d of the terms it is "
                    "trained on are present" % (int(ap.shape[-1]), len(cols)))
        if mv is not None:
            wl["movement"] = float(mv["wellness"].mean())
        if self.liveness is not None:
            # closes the sampled step; with --liveness-repair on, a layer whose watching window is
            # full has its dead units revived HERE, between two training steps
            self.liveness.end_step(self, step, B, train)
        timings["total"] = (time.perf_counter() - t0) * 1000
        return StepResult(exec_seen=int(self._exec_seen), nf_aligned=int(self._nf_aligned),
                          recon_pe=float(getattr(self, 'last_recon_pe', float('nan'))),
                          causal_pe=float(getattr(self, 'last_causal_pe', float('nan'))),
                          next_pe_per_modality=dict(self.last_next_pe or {}),
                          recon_dec_per_modality=dict(getattr(self, 'last_recon_dec', None) or {}),
                          tenure_pe_per_modality=dict(self.last_tenure_pe or {}),
                          tenure_gap_per_modality=dict(self.last_tenure_gap or {}),
                          tenure_tok_per_modality=dict(getattr(self, 'last_tenure_tok', None) or {}),
                          motor_out=motor_out.detach().float().cpu().numpy(), motor_for_action=mfa.cpu().numpy(),
                          hand_joint_goal=None if hand_joints is None else hand_joints.detach().cpu().numpy(),
                          hand_target_contract='world_v0' if hand_joints is None else 'wrist_relative_joint_v1',
                          gaze_out=(None if gaze_out is None else
                                    gaze_out if isinstance(gaze_out, np.ndarray) else
                                    gaze_out.detach().float().cpu().numpy()),
                          loss=float(loss.detach().float().item()), breakdown=bdf, pe_per_modality=pe, mean_pe=mean_pe,
                          wellness=wl, wellness_align_err=align_err,
                          total_wellness=_f(total_wellness), motor_confidence=_f(motor_conf),
                          grad_norm=grad_norm, curiosity_drive=float(drive),
                          curiosity_measured=bool(self.curiosity_measured),
                          source_mask=dict(source_mask),
                          timings_ms=timings, imagination_loss=None if imagination_loss is None else _f(imagination_loss),
                          wellness_grad_norm=float(wellness_grad.detach().float().abs().mean().item()))

    def token_temporal_std(self) -> Dict[str, float]:
        """Mean over dims of the std of each modality's token over the last ~1000 steps.
        Near 0 = the encoder emits the same token all the time (collapse)."""
        out = {}
        for m, ring in self.token_ring.items():
            if len(ring) >= 8:
                out[m] = float(torch.stack(list(ring)).std(dim=0).mean().item())
        return out

    def token_temporal_std_bodies(self) -> Dict[str, List[float]]:
        """The collapse watch on EVERY body: per sense, one number per body - the standard deviation of each of
        its token's numbers over the last 200 stored samples (every 5th model step, about 1,000 steps: the
        window of token_temporal_std, whose body-0 reading is entry 0 here), averaged over the numbers. 0 = the
        body's token is the same every moment. Samples of another body count (a batch-size change) start a new
        window. A reading only: no gradient, no random draw."""
        out: Dict[str, List[float]] = {}
        for m, ring in getattr(self, "_token_ring_bodies", {}).items():
            if len(ring) < 8:
                continue
            n = ring[-1].shape[0]
            same = [t for t in ring if t.shape[0] == n]
            if len(same) < 8:
                continue
            x = torch.stack([t.detach().float().cpu() for t in same])        # [samples, bodies, numbers]
            out[m] = [float(v) for v in x.std(dim=0).mean(dim=-1)]
        return out

    def _proprio_within_lesson(self, frame, source_mask, train: bool):
        """--proprio-token-repair within_body (mimo_bridge/proprio_token.py, version 6): this moment's raw
        proprioceptive input joins the window on the collapse watch's own sampling steps (step_count % 5, as
        token_ring), then, when training, the lesson on the window re-encoded through the live encoder and the
        registry's output LayerNorm (encoder_registry._encode_one, the path every proprioception token takes):
        each engaged body's shortfall of its response per unit of posture movement under the line (engaged under
        DEAF, released at the line), pushed through the posture-only rows, and from a body's first push on its
        fatigue answer held at the live bodies' largest at the start (the hold). The same path through saved encoder
        weights (torch.func.functional_call) serves the held-out reading. None when the window is not full, neither
        the push nor the hold acts or nothing trains. The step applies it after the clip (WithinBody.apply)."""
        pw = self.proprio_within
        x = frame.get("proprioception") if isinstance(frame, dict) else None
        pw.observe(x, self.step_count, real=torch.is_tensor(x) and source_mask.get("proprioception") == "real")
        if not train:
            return None
        reg = getattr(self.encoder_registry, "_orig_mod", self.encoder_registry)
        enc = reg.encoders["proprioception"]

        def _encode(xx):
            return reg._encode_one("proprioception", xx, enc).reshape(xx.shape[0], -1)

        def _encode_with(state):
            def run(d):
                return torch.func.functional_call(enc, state, (d,))

            def f(xx):
                return reg._encode_one("proprioception", xx, run).reshape(xx.shape[0], -1)
            return f

        def _weights_now():
            return {k: v.detach().clone() for k, v in enc.state_dict().items()}

        return pw.lesson(_encode, _encode_with, _weights_now)

    def enable_replay(self, n_envs: int, size_per_env: int = 6000):
        self.replay = ReplayMemory(self.active, self.d_model, n_handles=len(self.handle_names),
                                   size_per_env=size_per_env, n_envs=n_envs,
                                   self_size=self.d_model if self.self_state_mode == 'perceive' else 0,
                                   memory_size=0 if self.persistent_memory is None else 2*self.persistent_memory.hidden_size)
        # the comparison records' map of ring slots to model steps (record c): -1 = no entry added by this process
        self.replay_slot_step = np.full(int(size_per_env), -1, dtype=np.int64)

    def replay_loss_only(self, batch: int = 48, seed: int = 0) -> Optional[float]:
        """The replay loss with NO optimizer step and NO weight change.

        Exists because a test that measures replay_update twice measures the second run
        against weights the first run already moved: the reviewer showed that structure
        producing two different losses from IDENTICAL action data, so it could not tell a
        live action pathway from a dead one.
        """
        if self.replay is None:
            return None
        st = torch.get_rng_state()
        torch.manual_seed(seed)
        try:
            # LemLite is not an nn.Module: snapshot the parameter list it optimises
            snap = [p.detach().clone() for p in self.params]
            opt = None
            if getattr(self, "optimizer", None) is not None:
                import copy as _copy
                opt = _copy.deepcopy(self.optimizer.state_dict())
            r = self.replay_update(batch=batch)
            with torch.no_grad():
                for p, q in zip(self.params, snap):
                    p.copy_(q)
            if opt is not None:
                self.optimizer.load_state_dict(opt)
            return None if r is None else float(r["replay_loss"])
        finally:
            torch.set_rng_state(st)

    def replay_update(self, batch: int = 48) -> Optional[dict]:
        """One re-study step from the diary: forward model + heads graded on the real next
        moment, motor controller graded on the real NEXT achieved pose. The five SENSORY
        encoders are not trained here (their tokens were stored detached); the MOTOR encoder
        IS, because the motor token is rebuilt below instead of replayed - measured: all 16
        of the motor encoder's parameter tensors take gradient and every one of its
        parameters gets a non-zero entry (284,928 on the 24-handle test body, 285,184 on
        trunk_v1's 26), while the other five encoders get exactly none; AdamW then moves
        the motor weights by about the learning rate (3.0068e-04 at lr 3.000e-04). The
        |grad| SUM is not a property of this code - the loss is a mean over windows drawn
        at random from the ring, so the same build gave 25.505 at batch 6 and 19.896 at
        batch 48 on one CPU ring, and 20.370 to 26.123 across four ring draws at batch 6.
        Quote the counts and the weight move, never the sum (audit P411).

        The motor token is REBUILT here rather than replayed from storage. The stored token
        was encoded live from [achieved(S_i), arrival action A_{i-1}] - the action that had
        already finished acting when S_i was observed. What conditions the step S_i -> S_i+1
        is the OUTGOING action A_i, which the ring back-fills one add later. Replaying the
        stored token would train the forward model on the wrong cause; a test asserts that
        changing only the stored action changes this update.
        """
        if self.replay is None:
            return None
        from .limb_target_control import replay_contract   # None, or the finger | calf | chest | rows | eye bits
        # THE EYE ROW'S CONTRACT (the fourth review's fix round, the coordinator's decision): under
        # --eye-row-convention physical replay studies only windows whose every step-from moment was recorded
        # under that convention with the model's gaze command beside its order (_eye_contract_bit), so after
        # an F1 switch it studies nothing until a whole window of post-switch moments exists - its ctx_len
        # step-from moments back-filled with the command, and its target, never the newest moment: ctx_len + 2
        # post-switch updates, 130 on H84 (tests/test_eye_replay_contract.py); until then sample_windows returns
        # None and so does this - and then post-switch windows only; no window reads a moment stored before the
        # switch, whose stored memory and self states were encoded from old-convention rows. Under 'legacy' the
        # contract is the one of before, bit for bit. The finger servo's bit (--finger-joint-servo on, set on the
        # learner by train_mimo.build_learner) works the same way; off adds nothing.
        _contract = replay_contract(self.hand_target_mode, getattr(self, 'parent_relative_targets', 'off'),
                                    getattr(self, 'position_rows', 'all'),
                                    getattr(self, 'eye_row_convention', 'legacy'),
                                    getattr(self, 'finger_joint_servo', 'off') or 'off')
        w = self.replay.sample_windows(batch, self.ctx_len, self.device,
            **({} if _contract is None else {'action_contract': _contract}))
        if w is None:
            return None
        # n_handles = len(self.handle_names) (26 under this run's trunk_v1 layout); the
        # literal that stood here said 23
        toks, ach = w["tokens"], w["achieved"]              # [R, ctx+1, n_mod, d], [R, ctx+1, n_handles, 10]
        Rn = toks.shape[0]
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.dtype):
            # context = the first ctx-1 moments, current = moment ctx-1, target = moment ctx
            # The live context buffer stores forward_model's ctx_update, which is the
            # frame AFTER modality embeddings are added (forward_model.py:314, and its
            # docstring says so). Replay stored raw encoder tokens and reshaped them
            # straight in, so replay fed the transformer a context one embedding short -
            # and that embedding has RMS 0.99, the same order as the tokens themselves.
            # Two different context distributions training the same weights.
            _fm = getattr(self.forward_model, "_orig_mod", self.forward_model)
            # The buffer is a lazily refreshed cache: it holds ZEROS until the first
            # forward pass populates it, and goes stale after every optimizer step.
            # Reading it without refreshing silently added a zero embedding on the first
            # replay call - caught by the control test that asserts identical data gives
            # identical loss, which is the whole reason that control exists.
            if hasattr(_fm, "_refresh_embedding_cache"):
                _fm._refresh_embedding_cache()
            ctx = toks[:, :self.ctx_len - 1]                       # [R, ctx-1, n_mod, d]
            _emb = getattr(_fm, "_mod_emb_expanded", None)
            if _emb is None or _emb.shape[-2] != ctx.shape[2]:
                # Skipping this in silence restores the measured defect the comment above
                # documents. The two neighbouring integrity checks in this method raise
                # for a missing memory (:'replay is missing the memory...') and a missing
                # self perception; this one used to have no else branch at all, so a
                # modality added or dropped between a checkpoint and a resume would send
                # replay back to a different context distribution with no log line.
                raise RuntimeError(
                    "replay context has no modality embedding to add "
                    f"(cache {None if _emb is None else tuple(_emb.shape)}, "
                    f"context modalities {ctx.shape[2]}) - replay would train the "
                    "transformer on a different context distribution than the live step")
            ctx = ctx + _emb.to(ctx.dtype).unsqueeze(1)            # broadcast over time
            self._replay_ctx_embedded = self._replay_ctx_embedded + 1
            ctx = ctx.reshape(Rn, -1, self.d_model)
            cur = {m: toks[:, self.ctx_len - 1, k].unsqueeze(1) for k, m in enumerate(self.replay.mods)}
            nxt = {m: toks[:, self.ctx_len, k] for k, m in enumerate(self.replay.mods)}
            # rebuild the motor token from [achieved, OUTGOING action] for the moment being
            # stepped from, and for every context moment behind it
            # review E15: this gated on .all() - one unlabeled step-from moment
            # anywhere skipped the rebuild for every window in the batch, and the
            # whole batch trained on stored motor tokens with zero command columns,
            # uncounted. Per-window now; unlabeled windows keep their stored token.
            _lab = w["has_executed"][:, self.ctx_len - 1]                # [R]
            if "motor" in self.replay.mods:
                # COUNTED FIRST, outside the any() gate. With no labelled window in the
                # batch the rebuild below is skipped for every window, each one trains
                # the forward model on the stored motor token - the arrival action, which
                # this method's docstring calls the wrong cause - and the old form
                # recorded zero unlabeled windows, so that pass was indistinguishable in
                # its own return value from a fully labelled one. Measured on a ring with
                # no executed orders: replay_motor_rebuilt 0, replay_motor_unlabeled 0,
                # and one optimizer step moving parameters by about the learning rate
                # (3.000e-04).
                self._replay_motor_unlabeled += int((~_lab).sum())
            if "motor" in self.replay.mods and bool(_lab.any()):
                mi = self.replay.mods.index("motor")
                # Through the REGISTRY, not the encoder module: the registry applies the
                # output LayerNorm every real token gets. Calling the module directly is
                # exactly what predict_from_action does, and that produced tokens 1,500x
                # the scale of real ones.
                def _enc(a9, act):
                    x = torch.cat([a9[..., :MOTOR_ACHIEVED_F], command_for_encoder(act)], dim=-1)
                    flat = x.reshape(-1, x.shape[-2], MOTOR_INPUT_F)
                    return self.encoder_registry.encode(
                        {"motor": flat}, {"motor": "real"})["motor"].reshape(-1, self.d_model)

                # NOT detached, unlike the context rebuild below: this is the step-from
                # moment's own token, and training the motor encoder from it is the
                # intended behaviour (the live delayed next-frame lesson re-encodes the
                # same way). It is safe against the shared-LayerNorm hazard the context
                # rebuild's comment names ONLY because the registry's output norm is
                # non-affine: encoder_registry.py reads output_layernorm_affine, default
                # False, and measured here it is LayerNorm((128,), elementwise_affine=
                # False) with 0 parameters. Setting that flag True to reload an old
                # checkpoint would let replay train a parameter shared by all six
                # modalities from a motor-only gradient while the live path does not.
                cur["motor"] = torch.where(
                    _lab.view(Rn, 1, 1),
                    _enc(ach[:, self.ctx_len - 1],
                         w["executed"][:, self.ctx_len - 1]).reshape(Rn, 1, self.d_model),
                    cur["motor"])
                if self.ctx_len > 1:
                    ctx = ctx.reshape(Rn, self.ctx_len - 1, len(self.replay.mods), self.d_model)
                    ok = w["has_executed"][:, :self.ctx_len - 1]
                    # These past tokens are detached below. Building and retaining
                    # their backward graph can consume gigabytes at longer contexts
                    # without training any parameter. Encode that history without it.
                    with torch.no_grad():
                        reb = _enc(ach[:, :self.ctx_len - 1],
                                   w["executed"][:, :self.ctx_len - 1]).reshape(
                                       Rn, self.ctx_len - 1, self.d_model)
                    # The embedding was added to every slot above; this write replaces the
                    # motor slot, so the embedding must be re-added or the motor token
                    # becomes the ONLY token in the sequence without its modality tag.
                    # Measured before this line existed: max|ctx_motor - reb| = 0.0 exactly
                    # while all five other slots matched token+embedding to 1e-7.
                    if _emb is not None:
                        reb = reb + _emb.to(reb.dtype)[:, mi].unsqueeze(1)
                    # detached for the same reason the live context is detached: a context
                    # slot must not push gradient into the encoders, or replay trains the
                    # motor encoder and the SHARED output LayerNorm from a motor-only
                    # signal while live does not. Measured: without the detach gradient
                    # reaches the motor encoder and the shared output LayerNorm; with it,
                    # neither - and every other encoder gets exactly none either way (the
                    # |grad| sums are ring-draw numbers, not properties of the code).
                    ctx[:, :, mi] = torch.where(ok.unsqueeze(-1), reb.detach(), ctx[:, :, mi])
                    ctx = ctx.reshape(Rn, -1, self.d_model)
                self._replay_motor_rebuilt = self._replay_motor_rebuilt + 1
            replay_memory_bias = None
            if self.persistent_memory is not None:
                if w.get('memory_state') is None:
                    raise ValueError('replay is missing the memory belonging to its sampled decision')
                replay_memory_bias = self.persistent_memory.bias(w['memory_state'], site="replay")
            if self.self_state_mode == 'perceive':
                if w.get('self_state') is None:
                    raise ValueError('replay lacks the self perception belonging to its decision')
                self_bias = self.self_state.residual(w['self_state']).reshape(-1,self.n_mod,self.d_model)
                replay_memory_bias = self_bias if replay_memory_bias is None else replay_memory_bias+self_bias
            latents, last_step, _ = self.forward_model(cur, ctx,
                **({} if replay_memory_bias is None else {'memory_bias':replay_memory_bias}))
            preds = self.prediction_heads(last_step, current={m: v.squeeze(1) if v.dim() == 3 else v for m, v in cur.items()}, who="lem_model replay")
            if self.efference is not None and "motor" in self.replay.mods and bool(_lab.any()):
                # the gaze copy the live lesson adds to vision, on the windows whose order is known
                from lem.model.efference import apply_gaze_copy
                preds = apply_gaze_copy(
                    self.prediction_heads, preds, last_step,
                    command_for_encoder(w["executed"][:, self.ctx_len - 1].float()),
                    ach[:, self.ctx_len - 1].float(), mask=_lab)
            # Only REAL targets are graded. A synthetic channel is a fresh Gaussian draw
            # each moment: grading against it teaches the heads to predict noise.
            # PER-SAMPLE masking on the TARGET moment, and only that moment: a channel
            # that went stale mid-window is still graded, because what is graded is its
            # state where the target sits. This replaced a batch-global AND over the whole
            # window, which threw a whole modality away because one window somewhere held
            # a stale frame.
            vmask = w["valid"]                                   # [R, ctx+1, n_mod]
            graded, terms = [], []
            for k, m in enumerate(self.replay.mods):
                if m not in preds or m == "motor" or m not in self.pred_mods:
                    continue
                keep = vmask[:, self.ctx_len, k]                 # target moment only
                if not bool(keep.any()):
                    continue
                se = (preds[m] - nxt[m]).pow(2).mean(dim=-1)     # [R]
                terms.append((se * keep.float()).sum() / keep.float().sum().clamp(min=1.0))
                graded.append(m)
            _nfw = None
            if self.nf_norm is not None and terms:
                # --next-frame-norm persistence: the re-study of the same next-moment lesson carries
                # the same per-sense weights as the live lesson (the running averages only; replay
                # adds no reading of its own - its tokens were stored by older encoder weights)
                _nfw, _nfi_r = self.nf_norm.weights(graded, None)
                _vr_factor = None
                if self.vestibular_encoder_repair != "off":
                    # the revival's ceiling in the re-study too (vestibular_sense.REVIVAL_CEILING): the ring still
                    # holds dead windows; the factor stays on the device until the read after the step below
                    # the sense's lesson weight weighs its re-study as it weighs the live lesson (third review,
                    # section D; production's replay leaves it out, and with the flag off it still does)
                    from .vestibular_sense import revival_ceiling_terms as _vr_ceiling_terms
                    _weighted, _vr_factor = _vr_ceiling_terms(graded, terms, _nfw, _nfi_r.get("capped"),
                                                              lesson_weight=self.vestibular_lesson_weight)
                    loss = torch.stack(_weighted).sum()
                else:
                    loss = torch.stack([t * _nfw[m] for t, m in zip(terms, graded)]).sum()
            else:
                loss = (torch.stack(terms).sum() if terms
                        else torch.zeros((), device=self.device))
            # motor: command computed from the remembered moment, graded on what the body
            # REALLY was one moment later (graded on the measured next frame)
            handle_state = ach[:, self.ctx_len - 1]
            # DIVERGENCE CLOSED (audit P166, 2026-09-18). The live step hands the controller
            # the wellness slope at that moment's pose, both when ACTING and when
            # GRADING. Replay passed nothing, so MotorController.forward substituted zeros
            # into 234 of the controller's 1,262 input features - 18.54% of the input
            # vector - and re-study left those columns at exactly 0.0 gradient while the
            # live path trained them on real values (H83's own record: mean |wellness
            # slope| 0.000426 to 0.004861 over 137 windows, above zero in all 137).
            # NOTHING NEW IS STORED. The slope is a parameter-free function of the
            # achieved pose and that window's rom limit, both already in the ring, so it
            # is rebuilt here the way the live step builds it: same centring on the
            # handle centroid, same formula, and the window's OWN limit rather than
            # today's phase. Measured: the rebuilt slope on the ring's fp16 pose differs
            # from the live float32 value by at most 3.763e-03 against a mean slope size
            # of 0.1346 - the same treatment every other stored quantity gets.
            wcmd = handle_state[:, :, :9].detach().clone().float()
            wcmd[..., 0:3] = wcmd[..., 0:3] - wcmd[..., 0:3].mean(dim=1, keepdim=True)
            wcmd = wcmd.requires_grad_(True)
            replay_wgrad = self.wellness.motor_command_gradient(
                wcmd, rom_limit=torch.as_tensor(w["rom"]).flatten()).detach()
            # detached: the live copy carries create_graph=True but reaches no trainable
            # parameter (0 of 286 optimizer tensors), so detaching trains exactly what
            # the live path trains and builds no second-order graph here.
            motor_out, _c, _meta = self.motor_controller(latents, handle_state,
                                                         wellness_grad=replay_wgrad)
            # same units as the live motor loss: `achieved` stored the very handle_state
            # tensor the tokenizer already normalized - no rescale here
            target9 = ach[:, self.ctx_len, :, :9].clone()
            target9[..., 7:9] = motor_out[..., 7:9].detach()   # de-echo (same as live loss)
            # per-WINDOW limits: each remembered moment is graded under the phase it
            # lived (review 2026-09-06; the mean blended phases across the ring)
            dev = {"rom_limit": torch.as_tensor(w["rom"]).flatten(),
                   "strength_limit": torch.as_tensor(w["strength"]).flatten()}
            # DELIBERATE DIVERGENCE FROM THE LIVE MOTOR TERM, and all three parts of it
            # are exactly zero in effect today. Live (lem/training/loss_manager.py:256-270)
            # multiplies this same function by the modality's loss_weight and by the source
            # scale, and passes the curriculum's handle_mask in `dev`. Here it is added raw
            # and `dev` carries only rom_limit and strength_limit. Measured: motor loss_weight
            # 1.0, source scale for 'real' 1.0, and every one of the six curriculum phases
            # lists tuple(ALL_HANDLE_INDICES), so the mask is all-True. The day any of those
            # three changes, replay and live become two distributions training one set of
            # weights - the hazard this file documents elsewhere - so change them together.
            # (named, then added: the same sum; the comparison records report the term, record c)
            motor_term = self.loss_manager._compute_motor_loss(
                motor_out, target9, dev, pos_excursion=_meta.get("pos_excursion"))
            loss = loss + motor_term
            # review 2026-09-06: the replay VICReg term graded STORED tokens - fp16
            # ring contents rebuilt as leaf tensors with no path to any parameter.
            # A loss on constants trains nothing; it only inflated replay_loss with a
            # data-dependent offset and burned compute. Removed. (Live VICReg on the
            # encoders is untouched; if replay-side anti-collapse is ever wanted it
            # must grade re-encoded tokens, not buffer contents.)
        gn = float("nan")               # (read only by the comparison records when the loss is not finite)
        _loss_finite = bool(torch.isfinite(loss.detach()))
        if not _loss_finite:
            self._nonfinite_steps += 1
            log.error("non-finite replay loss - optimizer step skipped (#%d)",
                      self._nonfinite_steps)
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = self._clip_training_gradients()
            if not math.isfinite(gn):
                self._nonfinite_steps += 1
                log.error("non-finite replay grad norm - optimizer step skipped (#%d)",
                          self._nonfinite_steps)
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                if hasattr(_fm, "invalidate_embedding_cache"):
                    _fm.invalidate_embedding_cache()
        from . import comparison_records as _CR
        _term_values = None
        if _nfw is not None:
            # the re-study's per-sense errors for the next_frame_norm block (one read; the read of
            # the loss on the next line synchronises the device anyway)
            if _vr_factor is not None:
                self.vestibular_revival_replay = float(_vr_factor)
                _nfw = dict(_nfw, vestibular=_nfw["vestibular"] * self.vestibular_revival_replay)
            elif self.vestibular_encoder_repair != "off":
                self.vestibular_revival_replay = None
            if self.vestibular_encoder_repair != "off":
                # the replay shares read the weight the re-study carries, the lesson weight in it
                from .vestibular_sense import lesson_weights_used as _vr_used
                _nfw = _vr_used(_nfw, self.vestibular_lesson_weight)
            # the re-study's per-sense errors read once from the device (kept for the comparison records' pass line)
            _term_values = torch.stack(terms).detach().float().tolist()
            self.nf_norm.add_replay(dict(zip(graded, _term_values)), _nfw)
        out = {"replay_loss": float(loss.detach().float().item()), "replay_batch": Rn,
               "replay_graded_mods": len(graded),
               "replay_motor_rebuilt": int(self._replay_motor_rebuilt),
               "replay_motor_unlabeled": int(self._replay_motor_unlabeled),
               "replay_ctx_embedded": int(self._replay_ctx_embedded),
               "replay_clean_frac": self.replay.clean_fraction(),
               "replay_fill_frac": self.replay.fill_fraction()}
        if _CR.RECORDING:
            # THE COMPARISON RECORDS' PASS LINE (record c, comparison_records.pass_record): numbers this pass already
            # computed, read without gradient; the windows' bodies and first slots as sample_windows chose them (no
            # draw of its own). Behind the records' guard: a failure is counted and the line reads record None
            try:
                out["record"] = _CR.pass_record(
                    loss_value=out["replay_loss"], batch=Rn, graded=graded, terms=terms, term_values=_term_values,
                    weights=_nfw, motor_term=motor_term, grad_norm=gn, loss_finite=_loss_finite,
                    clip_norms=self.clip_norms, contract=_contract, windows=w, ring=self.replay,
                    slot_steps=self.replay_slot_step, ctx_len=self.ctx_len)
            except Exception as _ex:
                out["record"] = None
                _CR.record_failed(self, "replay_pass", _ex)
        return out

    GAZE_IDX = HANDLE_NAMES.index("gaze")

    @staticmethod
    def gaze_to_handle(motor9: np.ndarray, gaze_az_el: np.ndarray,
                       convention: str = "legacy") -> np.ndarray:
        """Write an ACTUAL eye command back into handle 23's orientation slots.

        The inverse of _gaze_from_handle. The order that moved the eyes is not the one the
        controller proposed: a spontaneous saccade or the peripheral-motion reflex can
        override it, and the servo clamps it. env.py records the result in
        out.info["gaze_cmd"], which until now was written and read by nothing. This puts
        it into the executed action so the forward model is conditioned on the eye
        movement that actually happened, not the one that was asked for.

        motor9: [B, n_handles, 9]; gaze_az_el: [B, 2] radians. Returns a copy.

        convention (gaze_reflex.EYE_ROW_CONVENTIONS; the model's own is model.eye_row_convention):
        'legacy' writes the row's pitch = plus the elevation, 'physical' = minus it, which is the
        eye's physical head-relative rotation that proprioception reports (motor interface audit
        2026-09-25, F1). A caller that writes the ledger for a model passes that model's setting.
        """
        s = eye_pitch_sign(convention)
        out = np.array(motor9, dtype=np.float32, copy=True)
        az = np.asarray(gaze_az_el, dtype=np.float32).reshape(-1, 2)[:, 0]
        el = np.asarray(gaze_az_el, dtype=np.float32).reshape(-1, 2)[:, 1]
        pitch = el if s > 0 else -el
        ca, sa = np.cos(az / 2.0), np.sin(az / 2.0)
        ce, se = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        gi = LemLite.GAZE_IDX
        out[:, gi, 3] = -sa * se      # x
        out[:, gi, 4] = ca * se       # y
        out[:, gi, 5] = sa * ce       # z
        out[:, gi, 6] = ca * ce       # w
        return out

    def _gaze_from_handle_t(self, motor_out: torch.Tensor) -> torch.Tensor:
        """Differentiable twin of _gaze_from_handle: same quaternion -> (az, el) math
        WITHOUT detach, so the anticipation-gaze loss can reach the controller through
        handle 23. Used only inside losses; acting still goes through the detached one."""
        q = motor_out[:, self.GAZE_IDX, 3:7].float()
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x, y, z, w = q.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
        # the elevation the row asks the servo for, in this model's eye row convention
        el = pitch if eye_pitch_sign(getattr(self, "eye_row_convention", "legacy")) > 0 else -pitch
        # gaze_reflex.AZ_RANGE / EL_RANGE, not a second copy of the numbers
        return torch.stack([yaw.clamp(*AZ_RANGE), el.clamp(*EL_RANGE)], dim=-1)

    def _gaze_from_handle(self, motor_out: torch.Tensor) -> np.ndarray:
        """Handle 23's orientation -> (azimuth, elevation) for the eye servo.

        Yaw and pitch of the commanded quaternion, clamped to the real eye ranges. The
        handle's position, radius and strength are ignored - an eyeball has none of them
        to command - which mirrors how the decoder treats it (empty joint chain). The
        elevation is the pitch (eye_row_convention 'legacy') or minus it ('physical': the row
        is the eye's physical head-relative rotation, whose pitch is minus the vertical joint).
        """
        q = motor_out[:, self.GAZE_IDX, 3:7].detach().float()
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x, y, z, w = q.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
        az = yaw.clamp(*AZ_RANGE)          # gaze_reflex owns the eye joint ranges
        el = pitch if eye_pitch_sign(getattr(self, "eye_row_convention", "legacy")) > 0 else -pitch
        el = el.clamp(*EL_RANGE)
        return torch.stack([az, el], dim=-1).cpu().numpy()

    @staticmethod
    def smooth_blend(cmd, prev, a, keep_row=None):
        """THE executed-command smoother (muscle low-pass, review E16 hemisphere
        alignment) - ONE implementation shared by the live path and offline
        probes, so an offline branch and the body receive the same order by
        construction (09-09 follow-up review: the paired probe re-implemented
        nothing and skipped this, comparing orders the body never gets).

        keep_row (--smooth-source executed passes the eyes' handle): that row is passed through
        unblended, because the eyes are sent the new order's gaze, never a blend (the eye servo
        reads res.gaze_out, taken from the order before this smoother)."""
        if a <= 0.0 or prev is None or prev.shape != cmd.shape:
            return cmd
        out = cmd.clone()
        out[..., 0:3] = a * cmd[..., 0:3] + (1 - a) * prev[..., 0:3]
        # q and -q are the same rotation; align hemispheres before blending
        pq = prev[..., 3:7]
        dot = (cmd[..., 3:7] * pq).sum(-1, keepdim=True)
        pq = torch.where(dot < 0, -pq, pq)
        q = a * cmd[..., 3:7] + (1 - a) * pq
        out[..., 3:7] = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out[..., 8] = a * cmd[..., 8] + (1 - a) * prev[..., 8]
        if keep_row is not None:
            out[..., keep_row, :] = cmd[..., keep_row, :]
        return out

    @staticmethod
    def gaze_row_as_sent(cmd, gaze_idx, convention: str = "legacy"):
        """The eyes' row of an order rewritten as the command the eye servo is sent: the row's
        yaw and elevation clamped to the eye joints' ranges (as _gaze_from_handle does for
        res.gaze_out) and written back as the quaternion LemLite.gaze_to_handle writes into the
        executed ledger, both in `convention` (the model's eye_row_convention). Differentiable (the
        clamp's gradient is zero beyond the eye's range, where the eye cannot go). Used by
        planned_execution, so the gaze the planner imagines is the gaze the eyes are commanded."""
        s = eye_pitch_sign(convention)
        q = cmd[..., gaze_idx, 3:7]
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x, y, z, w = q.unbind(-1)
        az = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).clamp(*AZ_RANGE)
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
        el = (pitch if s > 0 else -pitch).clamp(*EL_RANGE)     # the elevation the servo is sent
        pitch = el if s > 0 else -el                            # and the row that says it
        ca, sa = torch.cos(az / 2.0), torch.sin(az / 2.0)
        ce, se = torch.cos(pitch / 2.0), torch.sin(pitch / 2.0)
        row = torch.cat([cmd[..., gaze_idx, 0:3],
                         torch.stack([-sa * se, ca * se, sa * ce, ca * ce], dim=-1),
                         cmd[..., gaze_idx, 7:9]], dim=-1)
        out = cmd.clone()
        out[..., gaze_idx, :] = row
        return out

    @staticmethod
    def executed_fields_only(cmd, handle_state, position_rows: str = "all"):
        """The order as the decoder reads it (motor interface audit 2026-09-25, F4; applied when
        --executed-fields-only on). action_decoder.decode reads a position only for the rows it
        solves a position for and reads no radius (column 7) at all, so the ledger used to store 36
        position numbers (11 orientation-only body rows and the eyes' row) and 26 radii that reach no
        actuator, fed them to the motor encoder and let the planner optimise them. Here every other
        row carries the part's ACHIEVED position (cm) and every radius is 0; nothing else changes.
        Replacing them leaves the decoder's control exactly as it was
        (tests/test_executed_fields_only.py).

        position_rows: the decoder's own setting (action_decoder.POSITION_ROW_CHOICES; the learner's
        position_rows, train_mimo --position-rows). The rows kept are the decoder's LIVE position rows,
        action_decoder.position_handles(position_rows), never the fixed POSITION_HANDLES list: under
        'no_head_pelvis' (audit F3, the operator's decision of 2026-09-25 9:09 AM PDT) the decoder
        solves no position for the head and the pelvis, so their position asks are replaced by the
        achieved position here too, or the record and the contact head's input would keep asks no
        actuator performs. F3 and F4 read the same setting, so they agree on the live rows.

        cmd: [B, H, 9] in execution units (position cm); handle_state: [B, H, >=3] the achieved pose,
        positions in units of COMMAND_POSITION_RANGE_CM, or None (then cmd is returned unchanged).
        Out of place: a gradient reaches the live fields and none reaches the replaced ones."""
        if handle_state is None:
            return cmd
        from mimo_bridge.action_decoder import position_handles
        from mimo_bridge.body_layout import names_for_rows
        live_rows = position_handles(position_rows or "all")
        live = torch.tensor([n in live_rows for n in names_for_rows(cmd.shape[-2])],
                            device=cmd.device)[:, None]
        cur = (handle_state[..., 0:3].detach().to(device=cmd.device, dtype=cmd.dtype)
               * COMMAND_POSITION_RANGE_CM)
        pos = torch.where(live, cmd[..., 0:3], cur)
        return torch.cat([pos, cmd[..., 3:7], torch.zeros_like(cmd[..., 7:8]), cmd[..., 8:]], dim=-1)

    def configure_visual_input_calibration(self, path):
        from lem.model.input_conditioning import VERSION
        import hashlib
        raw = Path(path).read_bytes()
        calibration = json.loads(raw)
        if (calibration.get('version') != VERSION
                or not math.isfinite(float(calibration.get('scale', 0)))
                or float(calibration['scale']) < 1
                or len(str(calibration.get('source_model_sha256', ''))) != 64):
            raise ValueError('Invalid visual input calibration')
        self._visual_input_calibration = dict(calibration,
            calibration_sha256=hashlib.sha256(raw).hexdigest())

    def _restore_visual_input_conditioning(self, checkpoint, load_optimizer):
        from lem.model.input_conditioning import VERSION, migrate
        import hashlib
        if 'vision' not in self.prediction_heads.heads:
            if getattr(self, '_visual_input_calibration', None):
                raise ValueError('A visual calibration requires a vision head')
            return
        linear = self.prediction_heads.heads['vision'].mlp[0]
        receipt_path = Path(checkpoint) / 'model_receipt.json'
        saved = (json.loads(receipt_path.read_text(encoding='utf-8'))
                 if receipt_path.exists() else {}).get('visual_input_conditioning')
        scale = float(linear.input_scale)
        if scale != 1:
            if not saved or saved.get('version') != VERSION or float(saved.get('scale', 0)) != scale:
                raise ValueError('Conditioned visual weights lack their matching version/scale receipt')
            self.receipt['visual_input_conditioning'] = saved
        elif saved and float(saved.get('scale', 1)) != 1:
            raise ValueError('Visual conditioning receipt disagrees with saved weights')
        requested = getattr(self, '_visual_input_calibration', None)
        if requested is None:
            return
        requested_scale = float(torch.tensor(requested['scale'], dtype=linear.input_scale.dtype))
        if scale != 1:
            if (requested_scale != scale
                    or requested['source_model_sha256'] != saved.get('source_model_sha256')):
                raise ValueError('Requested visual calibration differs from the learned checkpoint')
            return
        source = Path(checkpoint) / 'model.safetensors'
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != requested['source_model_sha256']:
            raise ValueError('Visual calibration was measured on different checkpoint weights')
        migration = migrate(linear, requested['scale'], self.optimizer if load_optimizer else None)
        self.receipt['visual_input_conditioning'] = dict(requested, scale=migration['scale'], migration=migration,
            forward='first vision Linear reads x/scale; retained weight is W*scale; liveness input unchanged')

    def configure_visual_activation_calibration(self, path):
        from lem.model.activation_compensation import VERSION
        import hashlib
        raw = Path(path).read_bytes()
        calibration = json.loads(raw)
        if (calibration.get('version') != VERSION
                or len(str(calibration.get('source_model_sha256', ''))) != 64
                or not isinstance(calibration.get('shift'), list)
                or not isinstance(calibration.get('gain'), list)):
            raise ValueError('Invalid visual activation calibration')
        self._visual_activation_calibration = dict(calibration,
            calibration_sha256=hashlib.sha256(raw).hexdigest())

    def _restore_visual_activation_calibration(self, checkpoint):
        from lem.model.activation_compensation import VERSION
        import hashlib
        requested = getattr(self, '_visual_activation_calibration', None)
        if self.efference is None:
            if requested:
                raise ValueError('Visual activation calibration requires efference heads')
            return
        module = self.efference.vision_activation
        receipt_path = Path(checkpoint) / 'model_receipt.json'
        saved = (json.loads(receipt_path.read_text(encoding='utf-8'))
                 if receipt_path.exists() else {}).get('visual_activation_compensation')
        if module.enabled_py:
            if (not saved or saved.get('version') != VERSION
                    or not torch.equal(torch.as_tensor(saved.get('shift', []), device=module.shift.device,
                        dtype=module.shift.dtype), module.shift)
                    or not torch.equal(torch.as_tensor(saved.get('gain', []), device=module.gain.device,
                        dtype=module.gain.dtype), module.gain)):
                raise ValueError('Visual activation buffers lack their matching calibration receipt')
            if requested and (requested['calibration_sha256'] != saved.get('calibration_sha256')
                    or requested['source_model_sha256'] != saved.get('source_model_sha256')):
                raise ValueError('Requested activation calibration differs from the learned checkpoint')
            self.receipt['visual_activation_compensation'] = saved
            return
        if saved:
            raise ValueError('Activation calibration receipt disagrees with saved weights')
        if requested is None:
            return
        digest = hashlib.sha256()
        with (Path(checkpoint)/'model.safetensors').open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != requested['source_model_sha256']:
            raise ValueError('Activation calibration was measured on different checkpoint weights')
        module.configure(self.efference.vision_in, self.efference.vision_out,
                         requested['shift'], requested['gain'])
        self.receipt['visual_activation_compensation'] = dict(requested,
            shift=module.shift.cpu().tolist(), gain=module.gain.cpu().tolist(),
            forward='measured shifted/scaled GELU plus frozen old-minus-new correction; all learned tensors retained')

    @torch.no_grad()
    def _persistent_memory_switch(self, snapshot) -> dict:
        """--persistent-memory-repair: the function-preserving switch, at the start of the first training
        step after a load whose history holds a lived moment (LemLite.step, before the delayed lesson reads
        the memory, inside the step's autocast, so the fold takes the bias production computes there), or at
        an empty history's first lived call (memory_integration.observe); the lem/model/persistent_memory.py
        module note. PLANNED, THEN WRITTEN: every reading and every refusal comes first (the memory's own
        plan, PersistentMemory.prepare_switch, and the replay ring's check below); only then is anything
        changed, and the changes are copies, so a refusal or an error leaves every store as it was. Here,
        what the memory cannot reach:
          - the replay ring's written states must each give the fold's bias within the switch's line in
            fp32 (the readout weight times the stored state's distance from the reference state); a ring
            that still holds moments from before every body's state was the same is refused (at 380,550,
            0.149 per state number, against 1.1e-6 at 411,150). The saved states then become centred
            states of the saturated reference, state - state_ref, so replay's bias from them is the same
            constant; the ring stores centred states from now on;
          - readout.weight now multiplies the CENTRED state and starts at 0: its Adam moments (averages
            of gradients taken against a fixed +-1 state) are dropped, so Adam restarts the tensor with
            its own bias correction (the liveness repair does the same for a tensor revived whole);
          - the needs forecast's state slot now reads needs_gain * centred state (0 at the switch) where
            it read the saturated state; units the liveness repair revived read that slot through their
            own running centre, which moves with the slot, so their input is unchanged.
        The record names the checkpoint this process loaded (pre_switch_checkpoint): after the switch it is
        the only way back (the receipt carries the record into every later checkpoint; the trainer writes it
        into the stats stream). Returns the switch's record."""
        pm = self.persistent_memory
        plan = pm.prepare_switch(snapshot)
        rec = plan["record"]
        ref = plan["s_ref"]
        S = int(ref.shape[0])
        limit = float(plan["limit"])
        ring = getattr(self, "replay", None)
        order = None
        if ring is not None and getattr(ring, "memory_states", None) is not None:
            written = int(min(int(ring.count), int(ring.size)))
            if written:
                order = torch.tensor([(int(ring.ptr) - 1 - k) % int(ring.size) for k in range(written)])
                kept = ring.memory_states[:, order].reshape(-1, S)
                # IN REPLAY'S OWN ARITHMETIC (the second review, finding 5): replay computes each stored
                # moment's bias as production computes every memory bias, the readout inside the step's
                # autocast (bf16 on the graphics card), in calls of the trainer's --replay-batch; the same here,
                # against the fold. In bf16 every stored state number above 0.998 in size rounds to the same
                # +-1, so a stored state a few millionths from the reference reads the fold exactly.
                fold = plan["fold"]
                batch = int(getattr(self, "replay_batch", None) or 48)
                worst, above = 0.0, 0
                for i in range(0, int(kept.shape[0]), batch):
                    d = (pm.production_bias(kept[i:i + batch]) - fold).abs().amax(-1)
                    worst = max(worst, float(d.max()))
                    above += int((d > limit).sum())
                # beside it, read with no line: the fp32 distance version 2 judged
                W = pm.readout.weight.detach().float()
                fp32 = 0.0
                with torch.autocast(device_type=W.device.type, enabled=False):
                    for i in range(0, int(kept.shape[0]), 4096):
                        fp32 = max(fp32, float(((kept[i:i + 4096].to(W.device) - ref) @ W.T).abs().max()))
                rec["ring_states_checked"] = int(kept.shape[0])
                rec["ring_check_batch"] = batch
                rec["ring_bias_change_max_abs"] = worst
                rec["ring_states_above_line"] = above
                rec["ring_bias_change_fp32_max_abs"] = fp32
                rec["ring_state_distance_max_abs"] = float((kept - ref.cpu()).abs().max())
                if above:
                    raise ValueError(
                        "persistent memory repair refused: %d of %d written replay moments hold a memory state "
                        "whose bias differs from the fold by more than the line (largest %.3g, line %.3g); "
                        "replay of them would change at the switch. Switch after the ring has aged them out"
                        % (above, int(kept.shape[0]), worst, limit))
        # ---- every check passed: the writes (copies only)
        opt = getattr(self, "optimizer", None)          # a frozen acting copy may carry none
        dropped = opt is not None and opt.state.pop(pm.readout.weight, None) is not None
        rec["readout_weight_adam"] = ("dropped: Adam restarts it with its own bias correction" if dropped
                                      else "no saved moments")
        # THE ENCODER'S MOMENTS TOO (the second review, finding 7): built while the saturated tanh passed about
        # 1e-7 of every gradient, they hold about 1e-11 and nothing else; with the step count past 400,000 Adam's
        # bias correction is off, so a gradient growing from 0 after the switch met them with steps 3 to 5 times
        # the learning rate. Dropped, Adam restarts each tensor with its own bias correction.
        g = pm.recurrent
        encoder = (pm.input_projection[0].weight, pm.input_projection[0].bias, g.weight_ih_l0, g.weight_hh_l0,
                   g.bias_ih_l0, g.bias_hh_l0, pm.query.weight, pm.query.bias, pm.key.weight)
        n_enc = 0 if opt is None else sum(opt.state.pop(p, None) is not None for p in encoder)
        rec["encoder_adam"] = "%d of %d tensors dropped: Adam restarts them with its own bias correction" % (
            n_enc, len(encoder))
        if order is not None:
            ring.memory_states.sub_(ref.to(ring.memory_states))
            rec["ring_states_converted"] = int(ring.memory_states.shape[0]) * int(order.numel())
        lv = getattr(self, "liveness", None)
        key = "persistent_memory__outcome__0"
        if lv is not None and getattr(lv, "mode", "off") == "on" and key in lv.rows:
            rows = lv.rows[key]
            moved = float(rows.centre_weight) > 0.0
            if moved:
                rows.centre_mean[:S] -= ref.to(rows.centre_mean)
            rec["needs_units_revived"] = int(rows.mask.sum())
            rec["needs_revived_centre_moved"] = moved
        pm.apply_switch(plan)
        # the plan read the readout inside the caller's autocast, which keeps bf16 copies of the weight and bias
        # it cast; the switch changed both in place (the second review, question 2): forget those copies
        torch.clear_autocast_cache()
        rec["switched_at_step"] = int(self.step_count)
        rec["pre_switch_checkpoint"] = getattr(self, "loaded_from", None)
        ret = rec.get("constant_return") or {}
        names = list(getattr(self, "active", None) or [])
        for k in ("start_per_slot", "target_per_slot"):             # the constant's return, by sense slot name
            if isinstance(ret.get(k), list) and len(ret[k]) == len(names):
                ret[k] = dict(zip(names, ret[k]))
        self.persistent_memory_switch = rec
        # the trainer writes it into the stats stream once (train_mimo, after the step) and clears this
        self.persistent_memory_switch_unwritten = rec
        self.receipt["persistent_memory_repair"] = dict(self.receipt.get("persistent_memory_repair") or {},
                                                        switch=rec)
        log.warning("persistent memory repair switched at step %d: constant bias %.4g RMS folded into the "
                    "readout bias at production's own %s value (largest change: %.3g in fp32, %.3g in production's "
                    "numbers, line %.3g); the saved state was %.4f above 0.99, the repaired state reads %.4f above "
                    "0.99 with spread %.4g between bodies; the way back is the checkpoint %s",
                    self.step_count, rec["constant_bias_rms"], rec["production_bias_dtype"],
                    rec["bias_change_max_abs"], rec["production_change_max_abs"], rec["bias_change_limit"],
                    rec["old_state_share_above_0p99"], rec["new_state_share_above_0p99"],
                    rec["new_state_spread_between_bodies"], rec["pre_switch_checkpoint"])
        return rec

    def memory_report(self) -> dict:
        """The stats record's persistent_memory block: last_memory and, with --persistent-memory-repair, the
        report window's means of the repaired memory's gauges (PersistentMemory.gauge_report, one read from
        the device per report), the step of the switch and the goal reading (memory_goal_report)."""
        out = dict(self.last_memory)
        pm = self.persistent_memory
        if pm is not None and pm.repair != 'off':
            gauges = pm.gauge_report()
            for k in [k for k, v in gauges.items() if k.endswith('_per_slot') and isinstance(v, list)]:
                if len(gauges[k]) == len(self.active):          # the memory's slots are the forward model's
                    gauges[k] = dict(zip(self.active, gauges[k]))
            out.update(gauges)
            out['repair_switched'] = bool(pm.repair_switched_py)
            sw = self.persistent_memory_switch
            if sw is not None:
                out['switched_at_step'] = sw.get('switched_at_step')
            goal = self.memory_goal_report()
            if goal:
                out['goal'] = goal
        return out

    # the goal reading's arms (persistent_memory_deploy_watch.py GOAL, redeclared 2026-09-26): the varying part taken
    # out; another body's in its place (body b reads body b + 1's, modulo the batch); a second other body's (b + 2)
    GOAL_VARIANTS = (("without", 0), ("other", 1), ("other2", 2))

    @torch.no_grad()
    def _memory_goal_predictions(self, nf_in, nf_last, memory_state, exec_t, B, donor_shift: int = 0):
        """The delayed lesson's predictions (the heads and the gaze copy) on its latent with the memory's varying
        part changed: nf_last minus readout.weight @ centred state (donor_shift 0: taken out), plus, for donor_shift
        k > 0, body (b + k) mod B's varying part in body b's place; the constant and everything else stay.
        A reading, not a lesson: the liveness monitor is paused and the revived rows' watching windows are put
        back afterwards, so the extra pass through the heads leaves no trace."""
        pm = self.persistent_memory
        varying = F.linear(memory_state.detach().float(), pm.readout.weight.detach().float())
        change = -varying
        if donor_shift:
            change = change + torch.roll(varying, shifts=-int(donor_shift), dims=0)
        latent = nf_last.detach() + change.reshape(nf_last.shape).to(nf_last.dtype)
        lv = getattr(self, "liveness", None)
        rows = list(getattr(lv, "rows", {}).values()) if lv is not None else []
        was = getattr(lv, "active", False) if lv is not None else False
        kept = [(r, r.seen_max, r.act_sum, r.seen_n, r.probe_x) for r in rows]
        try:
            if lv is not None:
                lv.active = False
            preds = self.prediction_heads(
                latent, current={m: v.squeeze(1) if v.dim() == 3 else v for m, v in nf_in.items()},
                who="lem_model memory goal reading")
            if self.efference is not None and exec_t is not None:
                from lem.model.efference import gaze_copy_of
                gc = gaze_copy_of(self.prediction_heads)
                ach0 = getattr(self, "_nf_ach", None)
                if (gc is not None and ach0 is not None and ach0.shape[0] == B and latent.shape[0] == B
                        and exec_t.shape[0] == B):
                    preds = gc(preds, latent, command_for_encoder(exec_t.float()), ach0)
        finally:
            if lv is not None:
                lv.active = was
            for r, a, b, c, d in kept:
                r.seen_max, r.act_sum, r.seen_n, r.probe_x = a, b, c, d
        return {m: v.detach() for m, v in preds.items()}

    @torch.no_grad()
    def _memory_goal_variants(self, nf_in, nf_last, memory_state, exec_t, B) -> dict:
        """{arm: predictions} for the goal reading's arms (GOAL_VARIANTS) the batch allows: another body's varying
        part needs 2 bodies, a second other body's 3."""
        return {name: self._memory_goal_predictions(nf_in, nf_last, memory_state, exec_t, B, donor_shift=k)
                for name, k in self.GOAL_VARIANTS if k < B}

    @torch.no_grad()
    def _accumulate_memory_goal(self, preds, variants, tokens_raw, senses) -> None:
        """One step's per-body squared errors of the delayed lesson's graded senses, with the memory's own varying
        part and under each arm of variants ({arm: {sense: prediction}}), into the report's device sums (no read)."""
        window = self.memory_goal_window if self.memory_goal_window is not None else {}
        for m in senses:
            if m not in preds or m not in tokens_raw or any(m not in v for v in variants.values()):
                continue
            target = tokens_raw[m].squeeze(1).detach().float()

            def err(p):
                return (p.detach().float() - target).square().reshape(target.shape[0], -1).mean(-1)
            errors = {"with": err(preds[m]), **{name: err(v[m]) for name, v in variants.items()}}
            acc = window.get(m)
            if acc is None or acc["with"].shape != errors["with"].shape or set(acc) != set(errors) | {"steps"}:
                window[m] = dict({k: e.clone() for k, e in errors.items()}, steps=1)
            else:
                for k, e in errors.items():
                    acc[k] += e
                acc["steps"] += 1
        self.memory_goal_window = window

    def memory_goal_report(self) -> dict:
        """The report window's goal reading, and a new window. Per graded sense, the delayed lesson's mean squared
        error with the memory's own varying part and under each arm, and three PAIRED readings (the same
        predictions on the same moments, differing only in the varying part):
          ln_without_over_with   ln(error without the varying part / with the body's own): the gain the repair is
                                 for (0 = the varying part changes nothing; above 0 = it lowers the error);
          ln_other_over_with     ln(error with another body's varying part / with the body's own): above 0 when the
                                 memory carries something about this body that another body's does not;
          ln_other2_over_other   ln(error with a second other body's / with the first other body's): the same paired
                                 quantity with no own memory in it (the control; the goal line's margin);
        over all bodies and without the bodies left out of learning-performance conclusions
        (prediction_audit.EXCLUDED_BODIES), and per body."""
        window, self.memory_goal_window = self.memory_goal_window, None
        if not window:
            return {}
        from .prediction_audit import EXCLUDED_BODIES
        pairs = (("ln_without_over_with", "without", "with"), ("ln_other_over_with", "other", "with"),
                 ("ln_other2_over_other", "other2", "other"))
        out = {}
        for m, acc in window.items():
            n = int(acc["steps"])
            e = {k: [float(x) for x in v.tolist()] for k, v in acc.items() if k != "steps"}
            nb = len(e["with"])
            keep = [b for b in range(nb) if b not in EXCLUDED_BODIES]

            def ln(a, b):
                return math.log(a / b) if a > 0 and b > 0 else None

            def pooled(idx):
                mean = {k: sum(v[b] for b in idx) / (n * max(1, len(idx))) for k, v in e.items()}
                row = {"error_" + k: v for k, v in mean.items()}
                for name, top, bottom in pairs:
                    if top in mean and bottom in mean:
                        row[name] = ln(mean[top], mean[bottom])
                return row
            per_body = {"per_body_" + name: [ln(e[top][b], e[bottom][b]) for b in range(nb)]
                        for name, top, bottom in pairs if top in e and bottom in e}
            out[m] = dict(steps=n, all_bodies=pooled(list(range(nb))),
                          **{"without_body_" + "_".join(str(b) for b in EXCLUDED_BODIES): pooled(keep)},
                          **per_body)
        return out

    def _latent_prediction_penalty(self, predictions, nf_live, source_mask):
        # Regularize the same complete prediction the next-frame lesson grades.
        # With eye efference enabled, the base head and its learned eye correction
        # sum to the visual prediction. Penalizing the base alone pulls the two
        # lessons in different directions. The delayed lesson already knows the
        # executed order; the current acting prediction does not yet know it.
        # At startup there is no previous transition to regularize.
        if self.latent_regularizer is None:
            return None
        if self.next_frame_loss:
            if nf_live is None:
                return None
            predictions = nf_live[1]
        return self.latent_regularizer.penalty(predictions, valid=source_mask)

    def _clip_training_gradients(self):
        # A learning-only extension must not change the existing brain's update
        # indirectly by changing the denominator of shared gradient clipping.
        # THE EFFERENCE HEADS (2026-09-22) are clipped as their own group for the same reason: the
        # heads' own gradient size must not shrink every other module's step. Their gradient INTO
        # the core is part of the core's own norm, by design (DESIGN D5). Since efference_v2 the
        # gaze head has its own layers and the vestigial GazeHead gets no gradient at all.
        extra = []
        # each group's name, beside it, for the comparison records' clip_norms (record c); names only, no arithmetic
        names = []
        if self.self_state_mode == 'learn':
            extra.append(list(self.self_state.parameters()))
            names.append("self_state")
        if getattr(self, "efference", None) is not None:
            extra.append(list(self.efference.parameters()))
            names.append("efference")
        # THE CONTACT HEAD (2026-09-22) likewise: its own parameters are clipped as their own
        # group; its gradient into the core is part of the core's norm (DESIGN D10)
        if getattr(self, "contact_head", None) is not None:
            extra.append(list(self.contact_head.parameters()))
            names.append("contact_head")
        # THE MOTOR SYNERGY (2026-09-22) learns only from its own hindsight lesson, never into the
        # core, so it is its own group too: its gradient size cannot shrink any other module's step
        if getattr(self, "synergy", None) is not None:
            extra.append(list(self.synergy.parameters()))
            names.append("synergy")
        # THE LIVENESS REPAIR (2026-09-23): the revived units' new weights and the pools' all-element
        # paths are their own group too, so their gradient size cannot shrink the existing weights' step
        if getattr(self, "liveness", None) is not None and self.liveness.mode == "on":
            extra.append(list(self.liveness.parameters()))
            names.append("liveness")
        # THE ASKED-CHANGE PATH (2026-10-03, --motor-asked-change): its own group too, so the size of its gradient
        # cannot shrink the existing weights' step; at the switch the existing weights' first update is then the
        # update of a build without the path
        if getattr(self, "motor_asked_change", None) is not None:
            extra.append(list(self.motor_asked_change.parameters()))
            names.append("motor_asked_change")
        # THE HEADS' ALL-SLOT PATHS (2026-10-04, --heads-all-slot-path): their own group too, for the same reason
        if getattr(self, "heads_all_slot_path", None) is not None:
            extra.append(list(self.heads_all_slot_path.parameters()))
            names.append("heads_all_slot_path")
        # THE VESTIBULAR INPUT PATH (2026-10-04, --vestibular-input-path): its own group too, for the same reason
        if getattr(self, "vestibular_input_path", None) is not None:
            extra.append(list(self.vestibular_input_path.parameters()))
            names.append("vestibular_input_path")
        if extra:
            ids = {id(p) for group in extra for p in group}
            original = [p for p in self.params if id(p) not in ids]
            # (the explainer page quotes this line: tests/test_build_explainer.py anchors)
            norm = float(torch.nn.utils.clip_grad_norm_(original,self.grad_clip))
            norms = {"core": norm}
            # every group is clipped whatever its name: the loop runs over the groups, never over the names
            for gi, group in enumerate(extra):
                group_norm = float(torch.nn.utils.clip_grad_norm_(group, self.grad_clip))
                norms[names[gi] if gi < len(names) else "group_%d" % gi] = group_norm
                if not math.isfinite(group_norm):
                    # The optimizer's skip guard must see failures in every group.
                    norm = float("nan")
            self.clip_norms = norms
            return norm
        norm = float(torch.nn.utils.clip_grad_norm_(self.params,self.grad_clip))
        self.clip_norms = {"core": norm}
        return norm

    @staticmethod
    def differentiable_execution_transform(cmd, dev_limits, handle_state=None):
        """The same transform, WITHOUT detaching, so it can live inside the Adam loop.

        execution_transform() calls .detach(), which is right for the body path and fatal
        for the planner: the optimizer could not differentiate through it, so it kept
        descending on the RAW candidate. Measured with strength limited to 0.02, the two
        transition() calls in one plan saw:

            action inside the intention optimizer : 0.90 strength
            action transformed for the body       : 0.02 strength

        The optimizer was choosing an action by the consequences of a command the body
        cannot perform - a 45x difference in the quantity being optimised.

        Exploration noise is deliberately absent: it is stochastic and cannot be part of
        a transform the optimizer differentiates through. The ledger records the
        post-noise command, so the planned and executed actions are both recorded and
        are not claimed to be identical.
        """
        rom = float(dev_limits.get("rom_limit", 1.0))
        smax = float(dev_limits.get("strength_limit", 1.0))
        pos, rot = cmd[..., 0:3], cmd[..., 3:7]
        rad, stre = cmd[..., 7:8], cmd[..., 8:9]
        if handle_state is not None:
            # STRAIGHT THROUGH, so the value is the clamp's and the slope is tanh's.
            # Both clamps sat inside the Adam loop's differentiated path and returned
            # gradient exactly 0.0 on every saturated coordinate, so the planner's
            # positional descent was dead from the round a coordinate saturated - and
            # half of this run's recorded position channels start outside [-1, 1].
            # Writing tanh straight into the forward value instead would change the
            # order the body performs on every step, on a frozen lineage; here the
            # forward result is bit-identical to the clamp and only the backward pass
            # differs, so nothing the body does changes and round 20 still carries
            # information. execution_transform keeps the plain clamp and agrees with
            # this function value for value.
            cur_cm = handle_state[..., 0:3].to(cmd.dtype) * 200.0
            reach = 40.0 * rom
            des_hard = pos.clamp(-1.0, 1.0) * 200.0
            des_soft = torch.tanh(pos) * 200.0
            des_cm = des_soft + (des_hard - des_soft).detach()
            step = des_cm - cur_cm
            step_hard = step.clamp(-reach, reach)
            step_soft = reach * torch.tanh(step / reach)
            pos = cur_cm + step_soft + (step_hard - step_soft).detach()
        else:
            pos = pos.clamp(-1.0, 1.0) * rom * COMMAND_POSITION_RANGE_CM
        stre = stre.clamp(0.0, smax)
        mask = dev_limits.get('handle_mask')
        if mask is not None:
            mask = torch.as_tensor(mask, device=cmd.device, dtype=cmd.dtype)[None, :, None]
            rad, stre = rad * mask, stre * mask
            # Channels 3:7 are a TARGET QUATERNION in xyzw order, so a masked handle
            # HOLDS IDENTITY (0,0,0,1) - what apply_developmental_constraints does on the
            # body path. Multiplying the quaternion by the mask left (0,0,0,0), which is
            # not a rotation: the planner then descended on an order the body would never
            # be given, the exact split differentiable_execution_transform exists to
            # close. Every curriculum phase's mask is all-True today (all six phases,
            # checked), so nothing changes numerically in any run here.
            identity = torch.tensor([0.0, 0.0, 0.0, 1.0],
                                    dtype=cmd.dtype, device=cmd.device)
            rot = rot * mask + (1.0 - mask) * identity
            if handle_state is None:
                pos = pos * mask
        return torch.cat([pos, rot, rad, stre], dim=-1)

    def execution_transform(self, cmd, dev_limits, handle_state=None):
        """The deterministic part of what happens to a command before the body gets it.

        The planner imagined a RAW candidate and the body executed
        transform(candidate) - developmental range-of-motion limits, normalised-position
        to world-centimetre conversion, body-relative reach clamping and strength
        clamping. Those are not the same tensor, so the planner was scoring an action
        nobody would ever perform.

        Exploration noise is deliberately NOT here: it is stochastic and cannot be part
        of a deterministic scoring transform. It is applied after this, and the ledger
        records the post-noise command, so what is imagined is the deterministic body
        command and what is recorded is what the body actually got.

        TWO BRANCHES, AND THE BODY PATH TAKES THE ONE WITHOUT handle_state. Called with
        no handle_state this returns an ABSOLUTE position of rom_limit x 200 cm -
        measured 10.00 cm at the neonate rom 0.05 and 200.00 cm at the mature rom 1.0 -
        because apply_developmental_constraints denormalises without a pose to be
        relative to. The live body path calls it that way and then applies the
        40 x rom_limit reach cap itself, on the lines right after the call
        (measured 2.00 cm at rom 0.05 and 40.00 cm at rom 1.0). The planner supplies
        handle_state, through differentiable_execution_transform, and gets the reach cap
        from inside this function. The two forms are bit-identical once both are applied;
        a probe that calls this with handle_state is therefore NOT measuring the body
        path.
        """
        out = apply_developmental_constraints(cmd.detach().float().clone(), dev_limits)
        if handle_state is not None:
            cur_cm = handle_state[:, :, 0:3].detach().float() * 200.0
            des_cm = cmd.detach().float()[:, :, 0:3].clamp(-1.0, 1.0) * 200.0
            reach = 40.0 * float(dev_limits.get("rom_limit", 1.0))
            out[..., 0:3] = cur_cm + (des_cm - cur_cm).clamp(-reach, reach)
        out[..., 8].clamp_(0.0, float(dev_limits.get("strength_limit", 1.0)))
        return out

    def _stash_capability(self, cap):
        self._last_capability = cap
        return cap

    @torch.no_grad()
    def _revalue_ring(self, wl_stack: torch.Tensor) -> torch.Tensor:
        """Re-felt candidate values for goal SELECTION (09-09 review completion:
        'a remembered pleasant moment must be reconsidered under present needs').
        Touch and posture come from the CURRENT judge scoring each stored token
        snapshot (EVALUATOR_NAMES columns 1 and 4); the body's PRESENT capacity
        deficit MULTIPLIES each candidate's judged capability (column 2), so a
        depleted body is drawn toward remembered fresh-body states while a fresh
        body ranks by comfort alone - present need can flip the winner. Each
        moment's DRIVE contributions (movement, agency, curiosity) are preserved
        as the additive offset they contributed at storage (wl_stack minus the
        stored pre-adjustment bodily value). Only the ORDERING feeds the argmax,
        so the changed scale is harmless; the stored scalars are untouched
        (reporting and non-revalued paths keep them)."""
        T = len(self.goal_ring)
        if T == 0:
            return wl_stack
        B = wl_stack.shape[1]
        toks = {m: torch.stack([t[m] for t, _ in self.goal_ring]).reshape(T * B, -1)
                for m in self.active}
        zm = toks.get("motor")
        if zm is None:
            zm = torch.zeros(T * B, self.d_model, device=wl_stack.device)
        sens = {m: v for m, v in toks.items() if m != "motor"}
        wj = self.wellness.score_predicted_wellness(sens, zm)      # [T*B, 5]
        touch_post = wj[:, [1, 4]].mean(-1).reshape(T, B)
        cap_now = self.wellness._formula_capability(
            self._last_capability, B, wl_stack.device).view(1, B)
        # PRESENT NEED conditions the ranking (09-09 feasibility review
        # counterexample: adding cap_now equally to every candidate could never
        # change which goal wins). The interaction: the more depleted the body
        # is NOW (1 - cap_now, 0=fresh..1=spent), the more a candidate whose
        # remembered state the CURRENT judge scores as high-capability (a
        # fresh, rested body - column 2) is worth seeking; a fresh body ranks
        # purely by judged comfort+posture. All three columns share the 0-1
        # scale, so the deficit multiplier moves a candidate by at most one
        # component's full range - the same footing as the other components.
        # unknown capacity (no wellness call yet this segment) must mean
        # "rank by comfort", not "act depleted": the formula returns 0 for a
        # missing stash, which would read as a fully spent body
        if self._last_capability is None:
            core = touch_post
        else:
            need = (1.0 - cap_now).clamp(0.0, 1.0)                 # [1, B]
            cap_of_candidate = wj[:, 2].reshape(T, B)
            core = touch_post + need * cap_of_candidate
        # PRESERVE the drive contributions (09-09 follow-up review: replacing
        # the whole stored value deleted curiosity, agency and movement from
        # the ranking). Each moment's drives ride on top as the same additive
        # offset they contributed at storage: stored value minus the stored
        # PRE-adjustment bodily value. (Curiosity is multiplicative at storage;
        # its effect is carried here as the additive offset it produced then.)
        w0 = getattr(self, "_goal_ring_w0", None)
        if w0 is not None and len(w0) == T:
            drives = wl_stack - torch.stack(list(w0)).to(wl_stack.device)
            return core + drives
        return core

    def replan_last(self, wellness_w: float):
        """Replay a decision with its saved context, pose, limits and smoother.

        Weights must still be the same: this is a counterfactual at a frozen
        decision, not a way to evaluate a past decision under later training.
        The caller must still enforce raw-action pairing before comparing arms.
        """
        pi = getattr(self, "_last_plan_inputs", None)
        if pi is None:
            return None, {"error": "no planning event stored"}
        decision = pi.get('decision')
        if decision is None:
            return None, {'error': 'decision predates captured execution state'}
        if decision['parameter_versions'] != self._plan_parameter_versions():
            return None, {'error': 'model parameters changed since decision'}
        if decision['planner_settings'] != self._plan_settings():
            return None, {'error': 'planner settings changed since decision'}
        old_w = self.intention_gen._wellness_w
        self.intention_gen._wellness_w = float(wellness_w)
        try:
            devices = [self.device] if self.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices), torch.enable_grad():
                torch.set_rng_state(decision['torch_rng'])
                if devices:
                    torch.cuda.set_rng_state(decision['cuda_rng'], self.device)
                plan, meta = self._plan_chain(pi["cur"], pi["goal"], pi["motor"],
                                              max_steps=pi["max_steps"], decision=decision)
        finally:
            self.intention_gen._wellness_w = old_w
        return (plan[0].detach().float() if plan else None), meta

    def _plan_parameter_versions(self):
        return tuple((id(p), p._version) for p in self.params)

    def _plan_settings(self):
        # a learner without a planner (--intention off) has no planner settings, and the
        # self-choice actor still captures its decision state (self_intention.py): None, not a
        # crash (tests/test_acting_snapshot.py failed on this at ecfc9060f)
        if self.intention_gen is None:
            return None
        return tuple(getattr(self.intention_gen, k) for k in
                     ('_n_steps', '_lr', '_strength_reg', '_effort_per_body'))

    def configure_hand_response_calibration(self, path):
        import hashlib
        from lem.model.hand_response import VERSION, content_hash
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        if (payload.get('version') != VERSION or payload.get('sha256') !=
                content_hash({k: v for k, v in payload.items() if k != 'sha256'})):
            raise ValueError('Invalid hand-response calibration')
        self._hand_response_calibration = dict(payload,
            calibration_file_sha256=hashlib.sha256(raw).hexdigest())

    def _restore_hand_response_calibration(self, checkpoint):
        from .continuation_state import file_sha256
        receipt_path = Path(checkpoint) / 'model_receipt.json'
        saved_receipt = (json.loads(receipt_path.read_text(encoding='utf-8'))
                         if receipt_path.exists() else {})
        saved = saved_receipt.get('hand_response')
        requested = self._hand_response_calibration
        # --replace-saved-hand-calibration RECORD (train_mimo; docs/gain_of_function_v4/finger_command_interface/
        # HAND_CALIBRATION_ADMISSION_DESIGN.md, part A): absent or '' (every launch before it), each branch below
        # computes what it computed before; set, a requested calibration that differs from the saved one replaces it
        # only through _replace_saved_hand_calibration, and in the other two cases there is nothing to replace
        replacing = getattr(self, 'replace_saved_hand_calibration', '') or ''
        if saved:
            if requested is not None and requested != saved:
                if not replacing:
                    raise ValueError('Requested hand calibration differs from the saved checkpoint')
                self._replace_saved_hand_calibration(checkpoint, saved, requested, saved_receipt)
            else:
                if replacing:
                    log.warning("--replace-saved-hand-calibration: %s already saved the requested hand calibration "
                                "(or none was requested); nothing to replace, the saved one stands", checkpoint)
                self._hand_response_calibration = copy.deepcopy(saved)
        elif requested is not None:
            if replacing:
                log.warning("--replace-saved-hand-calibration: %s saved no hand calibration; nothing to replace, the "
                            "requested one is checked against the checkpoint's weights as before", checkpoint)
            expected = requested.get('source_model_sha256')
            if expected != file_sha256(Path(checkpoint) / 'model.safetensors'):
                raise ValueError('Hand calibration was measured on different checkpoint weights')
        if self._hand_response_calibration is not None:
            self.receipt['hand_response'] = copy.deepcopy(self._hand_response_calibration)
        elif self.hand_goals != 'off':
            raise ValueError('Hand goals need a source-bound learned response calibration')
        if (saved_receipt.get('hand_calibration_replacements')
                and getattr(self, 'hand_calibration_replacement', None) is None):
            # a later resume keeps the record of every earlier replacement in its receipt; only a replacement writes
            # this key, so a checkpoint saved before one changes nothing here
            self.receipt['hand_calibration_replacements'] = copy.deepcopy(
                saved_receipt['hand_calibration_replacements'])

    def _replace_saved_hand_calibration(self, checkpoint, saved, requested, saved_receipt):
        """--replace-saved-hand-calibration RECORD (train_mimo; HAND_CALIBRATION_ADMISSION_DESIGN.md, part A).

        The checkpoint saved the calibration `saved` (model_receipt.json hand_response) and the launch requests another,
        `requested` (--hand-response-calibration). The requested one replaces the saved one only when RECORD, the
        closed-loop hand-choice comparison's result, passes lem.model.hand_response.admission_check against: the
        requested payload and file bytes; the SHA-256 of this checkpoint's model.safetensors (which the requested
        calibration's source_model_sha256 must also name); the trainer code that runs this launch (trainer_code_hash);
        this launch's --finger-joint-servo rule and the command meaning it executes; the calibration's hand geometry;
        and its per-decision differences recomputed against the declared line. Otherwise ValueError names the binding
        that failed. The replacement's record (what replaced what, under which record, at which launch and time) goes
        into the receipt and, from train_mimo, into the stats stream; configure_hand_geometry checks the world's hand
        geometry against it and keeps it in force after the episode companion restores the saved calibration."""
        import hashlib
        from .continuation_state import file_sha256
        from .action_decoder import FINGER_SERVO_CONTRACT
        from lem.model.hand_response import POSE_SERVO_MEANING, admission_check, command_meaning, trainer_code_hash
        record_path = Path(self.replace_saved_hand_calibration)
        if self.hand_target_mode != 'wrist_relative_v1':
            raise ValueError("--replace-saved-hand-calibration needs --hand-target-mode wrist_relative_v1: the "
                             "replacement is checked against the world's hand geometry when the hand path is set up")
        raw = record_path.read_bytes()
        record = json.loads(raw)
        rule = getattr(self, 'finger_joint_servo', 'off') or 'off'
        meaning = FINGER_SERVO_CONTRACT if rule == 'on' else POSE_SERVO_MEANING
        model_sha256 = file_sha256(Path(checkpoint) / 'model.safetensors')
        summary = admission_check(record, {k: v for k, v in requested.items() if k != 'calibration_file_sha256'},
                                  calibration_file_sha256=requested.get('calibration_file_sha256'),
                                  model_sha256=model_sha256, finger_joint_servo=rule, meaning=meaning,
                                  code_hash=trainer_code_hash())
        now = time.time()
        self.hand_calibration_replacement = dict(
            setting='--replace-saved-hand-calibration', checkpoint=str(checkpoint), model_step=int(self.step_count),
            model_sha256=model_sha256,
            replaced=dict(payload_sha256=saved.get('sha256'), calibration_file_sha256=saved.get('calibration_file_sha256'),
                          command_meaning=command_meaning(saved), source_model_sha256=saved.get('source_model_sha256')),
            replaced_by=dict(payload_sha256=requested.get('sha256'),
                             calibration_file_sha256=requested.get('calibration_file_sha256'),
                             command_meaning=command_meaning(requested),
                             source_model_sha256=requested.get('source_model_sha256'),
                             geometry_sha256=requested.get('geometry_sha256')),
            admission=dict(summary, path=str(record_path), file_sha256=hashlib.sha256(raw).hexdigest()),
            launch=dict(finger_joint_servo=rule, hand_goals=self.hand_goals, hand_target_mode=self.hand_target_mode),
            at=now, at_text=time.strftime('%b %d, %Y %I:%M:%S %p', time.localtime(now)),
            world_geometry_sha256=None, calibration=copy.deepcopy(requested))
        self._hand_response_calibration = copy.deepcopy(requested)
        self.receipt['hand_calibration_replacements'] = (
            copy.deepcopy(list(saved_receipt.get('hand_calibration_replacements') or []))
            + [self.hand_calibration_replacement_report()])
        log.warning("--replace-saved-hand-calibration: at %s (model step %d) the saved hand calibration %s (%s) is "
                    "replaced by %s (%s) under the admission record %s (%d decisions, mean difference %.4g)",
                    checkpoint, self.step_count, saved.get('sha256'), command_meaning(saved), requested.get('sha256'),
                    command_meaning(requested), record_path, summary['decisions'], summary['mean_difference'])

    def hand_calibration_replacement_report(self):
        """This launch's replacement record without the replacing calibration's own body (the receipt holds that under
        hand_response), or None when the launch replaced nothing."""
        record = getattr(self, 'hand_calibration_replacement', None)
        return None if record is None else {k: copy.deepcopy(v) for k, v in record.items() if k != 'calibration'}

    def configure_calf_projection(self, geometry):
        """The knees the worlds project calf orders onto (limb_target_control.planner_calf_geometry), or
        None: with them the planner's executed order (planned_execution) carries the calf rows the
        body receives."""
        if geometry is None:
            self.calf_projection = None
            return
        from .limb_target_control import check_calf_geometry
        self.calf_projection = check_calf_geometry(copy.deepcopy(geometry), self.handle_names)

    def configure_hand_geometry(self, geometry, previous_joints=None):
        from .hand_target_control import HandKinematics
        if list(geometry['handle_names']) != list(self.handle_names):
            raise ValueError('Hand geometry does not match the learner body layout')
        self._hand_geometry = copy.deepcopy(geometry)
        self.hand_kinematics = HandKinematics(self._hand_geometry)
        if getattr(self, 'hand_calibration_replacement', None) is not None:
            # --replace-saved-hand-calibration (set only by _replace_saved_hand_calibration at this launch's load)
            self._keep_hand_calibration_replacement(geometry)
        if self._hand_response_calibration is not None:
            from lem.model.hand_response import CalibratedHandResponse
            payload = {k: v for k, v in self._hand_response_calibration.items()
                       if k != 'calibration_file_sha256'}
            self.hand_response = CalibratedHandResponse(geometry, payload).to(self.device)
            self.receipt['hand_response'] = copy.deepcopy(self._hand_response_calibration)
        if self.hand_goals != 'off' and self.hand_response is None:
            raise ValueError('Hand goals require a calibrated hand response')
        if self.hand_goals != 'off':
            # --finger-joint-servo (set by train_mimo.build_learner; action_decoder.py, THE FINGER COMMAND'S MEANING):
            # the hand stage's fixed response predicts each hinge's 250 ms change from coefficients fit on executed
            # commands, so they must have been fit under the rule the world executes, in both directions: 'on' needs a
            # receipt declaring finger_joint_servo_v1, 'off' (production's rule) one fit under the pose servo (every
            # receipt without the key, the admitted one included, so production's launches pass as before)
            from lem.model.hand_response import POSE_SERVO_MEANING, command_meaning
            from .action_decoder import FINGER_SERVO_CONTRACT
            rule = getattr(self, 'finger_joint_servo', 'off') or 'off'
            expected = FINGER_SERVO_CONTRACT if rule == 'on' else POSE_SERVO_MEANING
            meaning = command_meaning(self.hand_response.receipt)
            if meaning != expected:
                raise ValueError("--finger-joint-servo %s with --hand-goals %s: the hand response calibration was "
                                 "fit on commands executed as '%s', not '%s'; refit it on commands executed under "
                                 "this setting" % (rule, self.hand_goals, meaning, expected))
        if previous_joints is not None:
            self._hand_previous_joints = torch.as_tensor(previous_joints, device=self.device, dtype=torch.float32)

    def _keep_hand_calibration_replacement(self, geometry):
        """The replacement's world geometry check, and the replacement kept in force. The world's hand geometry
        (content_hash of the geometry the worlds report) must be the one the replacing calibration was fit on and its
        admission record measured on. The episode companion restores the checkpoint's own calibration after the load
        (continuation_state.capture_learner carries every _* attribute, _hand_response_calibration among them), so the
        replacing calibration is put back here, before the hand stage's predictor is built from it."""
        from lem.model.hand_response import content_hash
        record = self.hand_calibration_replacement
        world = content_hash(geometry)
        if world != record['replaced_by']['geometry_sha256'] or world != record['admission']['geometry_sha256']:
            raise ValueError("--replace-saved-hand-calibration: the world's hand geometry (content hash %s) is not the "
                             "geometry the replacing calibration was fit on (%s) and admitted on (%s)"
                             % (world, record['replaced_by']['geometry_sha256'], record['admission']['geometry_sha256']))
        self._hand_response_calibration = copy.deepcopy(record['calibration'])
        record['world_geometry_sha256'] = world
        replacements = self.receipt.setdefault('hand_calibration_replacements', [])
        if replacements and replacements[-1].get('at') == record['at']:
            replacements[-1] = self.hand_calibration_replacement_report()
        else:
            replacements.append(self.hand_calibration_replacement_report())

    def _smooth_hand_joints(self, joints, previous, smoothing):
        joints = self.hand_kinematics.project_joints(joints)
        if previous is not None and previous.shape == joints.shape and smoothing > 0:
            joints = smoothing * joints + (1. - smoothing) * previous.to(joints)
        return self.hand_kinematics.project_joints(joints)

    def _capture_plan_state(self):
        """Own the mutable inputs read while a first action is optimized."""
        ctx = self.last_context if self.last_context is not None else self.context_buffer
        executed = getattr(self, "smooth_source", "proposal") == "executed"
        # --smooth-source executed: the smoother's previous order is the one the body was given,
        # and the eyes' row is neither blended nor left in a form the servo would not be sent
        prev = self._smooth_exec_prev if executed else self._prev_mfa
        d = {'version': 1,
             'memory_bias': None if getattr(self,'_memory_bias_current',None) is None else self._memory_bias_current.detach().clone(),
             'context': None if ctx is None else ctx.detach().clone(),
             'achieved': None if self.last_achieved is None else self.last_achieved.detach().clone(),
             'source_mask': dict(self._last_source_mask or {}),
             'dev': copy.deepcopy(self._last_dev or self.dev_limits(0)),
             'smooth_prev': None if prev is None else prev.detach().clone(),
             'smoothing': float(self.smooth_actions),
             # getattr: a caller's stand-in model without hand state (the planner tests' stubs) reads
             # as "no hand state", as every LemLite without hand geometry does
             'hand_previous_joints': (None if getattr(self, '_hand_previous_joints', None) is None
                                      else self._hand_previous_joints.detach().clone()),
             'hand_current_angles': (None if getattr(self, '_hand_current_angles', None) is None
                                     else self._hand_current_angles.detach().clone()),
             'hand_current_velocities': (None if getattr(self, '_hand_current_velocities', None) is None
                                         else self._hand_current_velocities.detach().clone()),
             'parameter_versions': self._plan_parameter_versions(),
             'planner_settings': self._plan_settings(),
             'torch_rng': torch.get_rng_state().clone(),
             'cuda_rng': (torch.cuda.get_rng_state(self.device).clone()
                          if self.device.type == 'cuda' else None)}
        # The eye request bypasses body smoothing in both modes. Reflexes and
        # spontaneous saccades can subsequently modify it in the environment.
        d['smooth_keep_gaze'] = int(self.GAZE_IDX)
        # the motor interface settings the live order is formed under (audit 2026-09-25, F1 and F4),
        # so every reader of this decision - the planner, the causal chain's alternatives, a replan,
        # the self-choice candidates - forms its orders the way the live step forms the body's
        d['eye_row_convention'] = str(getattr(self, 'eye_row_convention', 'legacy'))
        d['executed_fields_only'] = bool(getattr(self, 'executed_fields_only_on', False))
        # the decoder's live position rows F4 keeps (train_mimo --position-rows, audit F3)
        d['position_rows'] = str(getattr(self, 'position_rows', 'all') or 'all')
        # ...and inside the decision's limits too (a copy; the live limits are untouched): a probe that
        # forms orders with LemLite.planned_execution(raw, decision['dev'], ...) and no settings of its
        # own (docs/gain_of_function_v3 and v4 probe scripts) then forms them as the live step does
        d['dev'] = dict(d['dev'], eye_row_convention=d['eye_row_convention'],
                        executed_fields_only=d['executed_fields_only'],
                        position_rows=d['position_rows'])
        # the calf rows the worlds record (--parent-relative-targets; limb_target_control.calf_rows_torch):
        # only when the launch projects the calves, so a decision without them is exactly today's
        if getattr(self, 'calf_projection', None):
            d['calf_projection'] = self.calf_projection
            d['dev']['calf_projection'] = self.calf_projection
        return d

    @staticmethod
    def planned_execution(cmd, dev_limits, handle_state=None, *, previous=None, smoothing=0.,
                          keep_gaze_idx=None, eye_row_convention=None, executed_fields=None,
                          position_rows=None, calf_projection=None):
        """Differentiable command actually presented to the body after filtering.

        Planning formerly evaluated a full new command although the live actor
        mixed it with its previous proposal (0.35 new / 0.65 previous in H67).
        Keep the filter in the gradient path and apply it exactly once.

        keep_gaze_idx (--smooth-source executed): the eyes' row is written as the command the eye
        servo is sent (gaze_row_as_sent) and is not blended, so the gaze the planner imagines is
        the gaze the eyes get. The live smoother blends every row and the eyes are sent the
        unblended order's gaze, which is the mismatch this closes (understand-motor-path-energy.md
        section 5).

        eye_row_convention: the model's (gaze_reflex.EYE_ROW_CONVENTIONS), for the eyes' row as sent.
        executed_fields (--executed-fields-only on, audit F4): after the blend, the fields the
        decoder never reads are replaced as the live step replaces them (executed_fields_only), so
        the planner imagines the order the ledger records. After, not before: under --smooth-source
        proposal a replacement before the blend would leave 0.35 x achieved + 0.65 x an earlier free
        value where the ledger holds the achieved position.
        position_rows: the decoder's setting whose live position rows executed_fields keeps
        (action_decoder.position_handles; audit F3's --position-rows).
        Any of these three left at None is read from dev_limits (a planning decision's 'dev' carries
        them, _capture_plan_state), and without it is the old default: 'legacy', off, 'all'.
        calf_projection (--parent-relative-targets calf or calf_chest; limb_target_control.
        calf_rows_torch): the calf rows become the knee goal on the thigh's row, as the worlds record
        them, after everything above, since the worlds project the order they receive. None is read
        from dev_limits; absent there, no projection (today's order). F4's replacement comes first;
        the two commute (the projection rewrites only the calf rows' quaternions and passes their
        positions, radius and strength through; F4 touches only positions and radii).
        """
        if eye_row_convention is None:
            eye_row_convention = (dev_limits or {}).get('eye_row_convention', 'legacy')
        if executed_fields is None:
            executed_fields = bool((dev_limits or {}).get('executed_fields_only', False))
        if position_rows is None:
            position_rows = (dev_limits or {}).get('position_rows', 'all') or 'all'
        if calf_projection is None:
            calf_projection = (dev_limits or {}).get('calf_projection')
        command = LemLite.differentiable_execution_transform(cmd, dev_limits,
                                                            handle_state=handle_state)
        if keep_gaze_idx is not None:
            command = LemLite.gaze_row_as_sent(command, int(keep_gaze_idx),
                                               convention=eye_row_convention)
        blended = LemLite.smooth_blend(command, previous, smoothing, keep_row=keep_gaze_idx)
        if executed_fields:
            blended = LemLite.executed_fields_only(blended, handle_state, position_rows)
        if calf_projection:
            from .limb_target_control import calf_rows_torch
            blended = calf_rows_torch(blended, calf_projection)
        return blended

    def _plan_visual_gaze(self, current_state, goal_state, motor, decision, execution_transform):
        """Aim the eyes at the visual memory before searching for a body pose."""
        source = (decision.get('source_mask') or {}).get('vision', 'real')
        if ('vision' not in current_state or 'vision' not in goal_state
                or str(getattr(source, 'value', source)) != 'real'):
            return motor, {'stopped_by': 'no_real_visual_goal'}
        from lem.model.physical_goal import GazeOnlyActionSpace
        action, metadata = self.intention_gen.generate_action(
            current_state, {'vision': goal_state['vision']}, motor,
            source_mask=decision.get('source_mask'),
            execution_transform=execution_transform, dev_limits=decision['dev'],
            achieved_state=decision['achieved'], context_buffer=decision['context'],
            memory_bias=decision.get('memory_bias'),
            action_space=GazeOnlyActionSpace(motor, self.GAZE_IDX))
        metadata['goal_kind'] = 'visual_memory_gaze_only'
        return action, metadata

    def _plan_chain(self, current_state, goal_state, current_motor, max_steps: int = 1,
                    decision=None):
        """Bridge-side chain planner.

        The caller supplies the licensed horizon. Optimization and rollout use
        the same saved decision state and command filter. Later depths still
        carry the initial achieved pose: predicted proprioception tokens cannot
        yet be decoded into physical handle fields. A prediction license does
        not establish accurate future pose or improved behavior.
        """
        # review 2026-09-06: the old belt-and-braces clamp here (min(max_steps, 1))
        # silently discarded the earned-horizon license - plans stayed length 1 for
        # the license's whole life while last_intention.horizon logged 2-3. The
        # license IS the validation the clamp demanded: a depth opens only after
        # imagination beats do-nothing at that depth (win rate > 0.5), and its own
        # ceiling (1 + licensed depths, at most 3) arrives via max_steps. Fixed mode
        # still passes 1. One owner for the horizon: the caller.
        max_steps = max(1, int(max_steps))
        decision = self._capture_plan_state() if decision is None else decision
        plan_context = decision['context']
        plan_memory_bias = decision.get('memory_bias')
        plan_achieved = decision['achieved']
        plan_dev = decision['dev']
        physical_cost = None
        hand_goal = decision.get('hand_goal') if self.hand_goals == 'memory' else None
        hand_goal_active = hand_goal is not None and bool(hand_goal['valid'].any())
        physical_goal = decision.get('physical_goal')
        if physical_goal is not None and bool(physical_goal['valid'].any()):
            from lem.model.physical_goal import PhysicalGoalCost
            if self.efference is None:
                raise RuntimeError('Physical goals require the learned body head')
            physical_cost = PhysicalGoalCost(self.efference, physical_goal['pose'],
                                             physical_goal['valid'], self.handle_names)
            # The head forecasts the full held order. Later sensory depths carry
            # the original physical pose; do not present those as future body states.
            max_steps = 1
        smooth_prev = decision['smooth_prev']
        # from the CAPTURED decision, never the live model: a replan has to
        # reproduce its action, and self._last_source_mask moves under it
        source_mask = decision.get('source_mask') or {}
        plan, cur = [], {m: v for m, v in current_state.items()}
        motor = current_motor
        if self.hand_target_mode != 'world' and self.hand_kinematics is None:
            raise RuntimeError('Configure hand geometry before planning')
        if self.hand_target_mode != 'world':
            max_steps = 1
        _pstate = _preal = None
        meta = {"plan_length": 0, "converged": False, "final_mse": float("nan"),
                "start_mse": float("nan"), "scale_exit": False, "valid": True,
                # EVERY depth's distance, not just the start and the best (P125). final_mse
                # keeps only the CLOSEST depth, so start == final said "no depth was
                # closer" and nothing said how much further the rollout went, nor
                # which rule stopped it. Measured on H83: 137 of 137 planning events
                # stopped at 2 of a licensed 3 with final_mse == start_mse.
                "mse_by_depth": [], "stopped_by": "horizon"}
        if self.hand_goals == 'memory' and not hand_goal_active:
            meta['hand_descent'] = {'stopped_by': 'no_valid_memory_goal'}
        last_mse = float("inf")
        for _ in range(max_steps):
            # the SAME quantity the Adam loop minimises: intention_generator's filtered
            # set (no motor - it is the action - and no synthetic sense) and its SUM
            # reduction (_dict_mse, intention_generator.py:92-103)
            _dm = [m for m in cur if m != "motor" and m in goal_state
                   and str(source_mask.get(m, "real")) == "real"]
            mse = float(torch.stack([F.mse_loss(cur[m], goal_state[m])
                                     for m in (_dm or list(cur))]).sum())
            meta["mse_by_depth"].append(round(mse, 6))
            if meta["plan_length"] == 0 and meta["start_mse"] != meta["start_mse"]:
                meta["start_mse"] = mse       # where the plan STARTED from its goal -
                                              # the denominator of "how much of the gap
                                              # did this plan close" (operator scale)
            if mse < last_mse:
                meta["final_mse"] = mse     # report the best distance reached, not the blow-up
            sensory_converged = mse < 0.01 and physical_cost is None
            if sensory_converged:
                meta["converged"] = True
                meta["stopped_by"] = "converged"
                if self.hand_goals == 'memory':
                    meta['hand_descent'] = {'stopped_by': 'sensory_converged_existing_exploration_retained'}
                break
            # divergence guard: chaining predict_from_action on its own outputs can leave the
            # learned manifold and explode (measured: goal distance 2.2e6 at depth 4 without
            # this). Extend the plan only while imagination is APPROACHING the goal.
            if len(plan) >= 2 and mse > last_mse * 1.05:
                meta["stopped_by"] = "diverged"
                self._plan_guard_trips += 1
                break              # depth 2 minimum: the guard may only trim beyond it
            last_mse = min(last_mse, mse)
            hand_space = None
            if self.hand_target_mode != 'world':
                from .hand_target_control import HandActionSpace
                canonical, joints, _ = self.hand_kinematics.project_world_orders(motor.detach().cpu().numpy())
                motor = torch.as_tensor(canonical, device=motor.device, dtype=motor.dtype)
                base_space = (self._synergy_space(plan_achieved, motor)
                              if self.synergy is not None and self.synergy_planner else None)
                hand_space = HandActionSpace(self._hand_geometry, motor,
                    torch.as_tensor(joints, device=motor.device, dtype=motor.dtype), base_space=base_space)
            if physical_cost is not None:
                # Different units and horizons need separate searches: visual
                # aiming uses the existing next-frame prediction; body choice
                # uses the held-order physical forecast. Neither trades its
                # goal against an arbitrary weighting of the other quantity.
                initial_hand = None if hand_space is None else hand_space.initial.detach()
                def eye_execution(cmd, dev_limits, handle_state=None):
                    executed = self.planned_execution(cmd, dev_limits, handle_state,
                        previous=smooth_prev, smoothing=decision['smoothing'],
                        keep_gaze_idx=decision.get('smooth_keep_gaze'),
                        eye_row_convention=decision.get('eye_row_convention', 'legacy'),
                        executed_fields=decision.get('executed_fields_only', False),
                        position_rows=decision.get('position_rows', 'all'))
                    if initial_hand is not None:
                        fixed_joints = self._smooth_hand_joints(initial_hand,
                            decision.get('hand_previous_joints'), decision['smoothing'])
                        executed = self.hand_kinematics.assemble_world(
                            executed, fixed_joints, position_scale=1.)
                    return executed
                motor, eye_meta = self._plan_visual_gaze(
                    cur, goal_state, motor, decision, eye_execution)
                meta['gaze_descent'] = eye_meta
                if hand_space is not None:
                    from lem.model.physical_goal import PhysicalTrustRegion
                    hand_space = PhysicalTrustRegion(hand_space,motor,self.handle_names)
            hand_goal_override = None
            def candidate_execution(cmd, dev_limits, handle_state=None):
                executed = self.planned_execution(cmd, dev_limits, handle_state,
                                              previous=smooth_prev,
                                              smoothing=decision['smoothing'],
                                              keep_gaze_idx=decision.get('smooth_keep_gaze'),
                                              eye_row_convention=decision.get('eye_row_convention', 'legacy'),
                                              executed_fields=decision.get('executed_fields_only', False),
                                              position_rows=decision.get('position_rows', 'all'))
                if hand_space is not None:
                    raw_joints = (hand_space.goal_joints() if hand_goal_override is None
                                  else hand_goal_override)
                    joints = self._smooth_hand_joints(raw_joints,
                        decision.get('hand_previous_joints'), decision['smoothing'])
                    executed = self.hand_kinematics.assemble_world(executed, joints, position_scale=1.)
                return executed
            # --motor-synergy on: the descent runs on the synergy's intention z (and the strength
            # logits and the eyes), the order built from the pose this depth starts at
            _space = None
            if self.synergy is not None and self.synergy_planner and hand_space is None:
                _space = self._synergy_space(_pstate.achieved if _pstate is not None
                                             else plan_achieved, motor)
            if hand_space is not None:
                _space = hand_space
            elif physical_cost is not None:
                from lem.model.physical_goal import PhysicalTrustRegion
                _space = PhysicalTrustRegion(_space,motor,self.handle_names)
            # THE REAL STATE. This passed neither, so generate_action built its
            # PredictiveState with context=None and achieved=None: the action was chosen
            # against a fabricated history (the current frame repeated across every
            # context position) and a zero achieved pose, and only then scored against
            # the real ones. Choosing on one state and scoring on another is the same
            # defect as choosing on one route and scoring on another.
            action, adam_meta = self.intention_gen.generate_action(
                cur, goal_state, motor, source_mask=decision['source_mask'],
                physical_goal=physical_cost,
                execution_transform=candidate_execution,
                dev_limits=plan_dev,
                achieved_state=(_pstate.achieved if _pstate is not None
                                else plan_achieved),
                # review 2026-09-06: the planner ran on the POST-shift buffer, which
                # already contains frame t - concatenated with the frame-t tokens the
                # forward model saw [t-2, t-1, t, t]: the current frame duplicated, a
                # layout no training path produces, and slot t-3 lost. last_context is
                # the pre-shift buffer stashed for exactly this pairing.
                context_buffer=(_pstate.context if _pstate is not None
                                else plan_context),
                **({} if plan_memory_bias is None else {'memory_bias':plan_memory_bias}),
                **({} if _space is None else {'action_space': _space}))
            if hand_goal_active:
                if hand_space is None or self.hand_response is None:
                    raise RuntimeError('Hand goal search lacks its execution or response model')
                if (decision.get('hand_current_angles') is None
                        or decision.get('hand_current_velocities') is None):
                    meta['hand_descent'] = {'stopped_by': 'awaiting_native_joint_readings'}
                else:
                    if abs(float(hand_goal['horizon_ms']) - 250.) > 1e-6:
                        raise ValueError('Hand response was qualified for 250 ms orders')
                    baseline = candidate_execution(action, plan_dev, handle_state=plan_achieved)
                    chosen = self.hand_response.choose_hand_goal(
                        baseline, hand_space.goal_joints(), decision.get('hand_previous_joints'),
                        decision['smoothing'], decision['hand_current_angles'],
                        decision['hand_current_velocities'], hand_goal['pose'], valid=hand_goal['valid'])
                    hand_goal_override = chosen['raw_joint_goal'].to(action)
                    action = self.hand_kinematics.assemble_world(action, hand_goal_override).detach()
                    executed = candidate_execution(action, plan_dev, handle_state=plan_achieved)
                    if not torch.allclose(executed, chosen['command'].to(executed), atol=2e-5, rtol=1e-6):
                        raise RuntimeError('Chosen hand goal differs from final planned execution')
                    meta['hand_descent'] = dict(chosen['metadata'],
                        stage='after_sensory_planner_before_final_prediction',
                        ring_index=hand_goal.get('ring_index'), stopped_by='completed')
            plan.append(action)
            if hand_space is not None and len(plan) == 1:
                meta['hand_joint_goal_raw'] = (hand_space.goal_joints() if hand_goal_override is None
                                             else hand_goal_override).detach().clone()
            meta["plan_length"] = len(plan)
            # RECORD THE ADAM DESCENT. generate_action returns what its own rounds did -
            # the loss it started and ended at, split into the two halves that make it up
            # (goal_distance, the summed squared distance to the desired sensory
            # consequence, and effort, strength_reg * sum(strength^2)) - and this caller
            # bound it to a name nothing read, so the plan record carried only
            # start_mse/final_mse from the rollout AFTER the descent. Those two are equal
            # at horizon 1 by construction, which is why gap_closed_pct read exactly 0.0
            # on all 137 of H83's planning events: the number meant to say how much of the
            # gap a plan closed never measured the descent at all. Keeping the halves
            # apart matters - a descent that only shrank the commanded strength would
            # otherwise report the same "improvement" as one that got closer to the goal.
            descent = {
                "depth": len(plan),
                **{key: (adam_meta or {}).get(key) for key in
                   ('n_steps', 'first_loss', 'last_loss', 'first_goal_distance', 'last_goal_distance',
                    'first_effort', 'last_effort', 'strength_mean', 'selection', 'selected_iteration',
                    'per_body', 'updates', 'stopped_by', 'goal_kind')},
                **({} if not adam_meta or 'synergy' not in adam_meta else {'synergy': adam_meta['synergy']}),
            }
            if adam_meta is not None:
                meta.setdefault('descent', []).append(descent)
            with torch.no_grad():
                fm = getattr(self.forward_model, "_orig_mod", self.forward_model)
                tok = {m: cur[m].unsqueeze(1) for m in cur}
                # CANONICAL TRANSITION. This used to call predict_from_action WITHOUT
                # state tokens, which encoded the motor token past the registry LayerNorm
                # and ran the heads on a hand-edited last_step without rerunning the
                # stack. That is what put tokens 1,500-3,400x off scale into the wellness
                # scorer, railed it to 0.0/1.0, and cut every plan to length 1.
                # CHAIN the state. The planner used to hand self.context_buffer and
                # self.last_achieved to EVERY imagined depth, so depth 2 and 3 predicted
                # from the real context of depth 0 - "predict S2 using C0 again" rather
                # than a rollout. The returned PredictiveState carries its own advanced
                # context and pose, so each depth starts where the last one ended.
                from lem.model.predictive_state import PredictiveState, transition, scale_report
                if _pstate is None:
                    _pstate = PredictiveState(
                        tokens={m: (cur[m].squeeze(1) if cur[m].dim() == 3 else cur[m])
                                for m in cur},
                        # pre-shift buffer: pairs with the frame-t tokens (see above)
                        context=plan_context,
                        achieved=plan_achieved, domain="real", memory_bias=plan_memory_bias)
                    _preal = _pstate
                # the same transform the OPTIMIZER descended through, so the action
                # scored here is the one it was choosing, not a re-derived cousin
                _exec_form = candidate_execution(
                    action, plan_dev, handle_state=_pstate.achieved).detach()
                # _exec_form is in EXECUTION units (cm): the same normalisation every other
                # command gets on its way into the motor encoder. Differentiable (a division).
                _nxt = transition(self.forward_model, self.encoder_registry,
                                  self.prediction_heads, _pstate, command_for_encoder(_exec_form),
                                  advance_context=True)
                # RUNTIME SCALE GATE. scale_report existed and was called only by a unit
                # test on hand-made tensors, so "the gate says no" was true of the probe
                # and not of the planner. A rollout whose tokens have left the band the
                # scorer was trained on is stopped here, with the reason recorded.
                _rep = scale_report(_preal, _nxt)
                if not _rep["all_in_band"]:
                    # FAIL CLOSED. This used to set the flag, break, and return the plan
                    # with the action already appended - measured: plan length 1,
                    # scale_exit true, motor RMS ratio 20.46, and the caller executed it
                    # because it only checks whether the plan is non-empty. A rollout
                    # whose tokens have left the domain the scorer was trained on must
                    # not reach the body at all.
                    meta["scale_exit"] = True
                    meta["stopped_by"] = "scale_gate"
                    meta["scale_report"] = {k: round(v, 2) for k, v in _rep["ratio"].items()}
                    meta["valid"] = False
                    self.last_scale_report = meta["scale_report"]
                    plan = []
                    meta["plan_length"] = 0
                    break
                _pstate = _nxt
                smooth_prev = _exec_form.detach()
                preds = {m: v for m, v in _nxt.tokens.items() if m != "motor"}
                z_motor = _nxt.tokens.get("motor")
                well = self.wellness.score_predicted_wellness(preds, z_motor)
                if len(plan) == 1:
                    # THE DEPTH-1 PREDICTION AND ITS PREDICTED WELLNESS, kept (references,
                    # no copy, no compute) so the causal chain can log what the planner
                    # expected its first action to do. They were computed and dropped here.
                    meta["depth1"] = {"tokens": {m: v.detach() for m, v in preds.items()},
                                      "well": well.detach(), "exec_form": _exec_form,
                                      "state": _preal}
                # prediction heads return sensory tokens; the imagined motor token comes
                # back separately as z_motor - keep any other missing sense from before
                cur = {m: (preds[m].detach() if m in preds else
                           (z_motor.detach() if m == "motor" else cur[m]))
                       for m in cur}
                motor = action.detach()
            # The safety exit: stop extending a plan whose imagined outcome is dire. It was
            # skipped in baseline mode because the approximate scorer was untrained noise
            # back then, and gating plans on noise chopped every plan to length 1 (H4 hour
            # 1). The scorer now tracks the grounded one to 0.043 on a 0-1 scale, so the
            # skip is retired. In practice this changes almost nothing - the lowest
            # imagined wellness ever measured is 0.377, far above the 0.05 trigger - which
            # is exactly why it was not worth the attention it got.
            if float(well.min()) < 0.05:
                meta["stopped_by"] = "low_predicted_wellness"
                break
        return plan, meta

    # ------------------------------------------------------------------ efference heads
    def _eye_angles_deg(self, frames, B: int) -> Optional[torch.Tensor]:
        """[B, 2] the eyes' realised azimuth and elevation, degrees, the mean of the two eyes
        (BridgeFrame.eye_angles, read from the eye joints; the right eye's horizontal angle is
        already sign-flipped into the left eye's convention by env.eye_yaw_pitch). None when
        any frame does not carry them (the UE transport, an old pickle)."""
        vals = []
        for f in frames[:B]:
            ea = getattr(f, "eye_angles", None)
            if ea is None:
                return None
            vals.append(np.asarray(ea, dtype=np.float64).reshape(-1, 2).mean(axis=0))
        if len(vals) != B:
            return None
        return torch.tensor(np.degrees(np.stack(vals)), dtype=torch.float32, device=self.device)

    def _efference_lesson(self, latent, exec_t, order_id, handle_state, frames, step, B,
                          train: bool = True):
        """Grade the efference heads on what the body and the eyes really did.

        latent: the delayed next-frame lesson's last-step tokens for S_t-1 + A_t-1, WITH
        gradient; exec_t: A_t-1 in centimetres; handle_state: the achieved pose of S_t;
        self._nf_ach (not yet refreshed this step): the achieved pose of S_t-1; train: whether
        this is a training step (only a lived training moment moves the heads' running means).
        The lesson itself is lem.model.efference.efference_lesson (efference_v2, lane-b1-efference
        2026-09-23); it returns {"efference_body": weighted loss, "efference_gaze": weighted
        loss}, either key absent on a step that graded nothing of it."""
        from lem.model.efference import efference_lesson
        return efference_lesson(self, latent, exec_t, order_id, handle_state, frames, step, B,
                                train=train)

    def efference_window_summary(self) -> Optional[dict]:
        """The `efference` block of the stats record, and a fresh window; None when off. Its
        fields are described in lem.model.efference.window_summary."""
        if self.efference is None:
            return None
        from lem.model.efference import window_summary
        return window_summary(self)

    @property
    def _eff_window(self) -> Dict[str, list]:
        """The efference window sums under their old name, for readers written before they moved to
        the public efference_window (mimo_bridge/explainer/capture.py). A property is not an
        instance attribute, so the episode companion never saves it (review MINOR 8); an older
        companion that carries the old attribute restores into the setter, which drops it: a window
        begins after the resume."""
        return self.efference_window

    @_eff_window.setter
    def _eff_window(self, value) -> None:
        pass

    def efference_lesson_ms(self) -> Optional[float]:
        """The machine block's efference_ms (lem.model.efference.lesson_ms); None when off."""
        if self.efference is None:
            return None
        from lem.model.efference import lesson_ms
        return lesson_ms(self)

    # ------------------------------------------------------------------ the contact head (D10)
    @staticmethod
    def _contact_report(frames, B: int):
        """This moment's contact report per body, read from the world's task dict: the patch
        states of the burst that produced it ([B, 53] ints, the union over the burst's env frames
        when the trainer merged it, else the last frame's), whether each body reported them, the
        self-contact onsets (None = not reported) and the world's own count of outside onsets."""
        from lem.model.contact_head import N_PATCHES
        cur = np.zeros((B, N_PATCHES), dtype=np.int64)
        ok = np.zeros(B, dtype=bool)
        onsets, wrep, mismatch = [], [], 0
        for b, f in enumerate(frames[:B]):
            t = (getattr(f, "meta", None) or {}).get("task") or {}
            s = t.get("contact_patch_state_burst", t.get("contact_patch_state"))
            if s is not None:
                if len(s) == N_PATCHES:
                    cur[b] = np.asarray(s, dtype=np.int64)
                    ok[b] = True
                else:
                    mismatch += 1
            onsets.append(t.get("self_contact_onsets") if "self_contact_onsets" in t else None)
            wrep.append(t.get("contact_onsets_world"))
        return cur, ok, onsets, wrep, mismatch

    @staticmethod
    def _contact_prev_input(prev: np.ndarray, device) -> torch.Tensor:
        """[B, 53] patch states -> [B, 53, 2] floats (an outside thing touching, an own part)."""
        return torch.from_numpy(np.stack([(prev & 1) > 0, (prev & 2) > 0], axis=-1)
                                .astype(np.float32)).to(device)

    def _contact_lesson(self, latent, nf_live, nf_memory_bias, exec_t, frames, step, B):
        """Grade the contact head, score this moment's onsets, and set its contact_control value.

        latent: the delayed next-frame lesson's latent for S_t-1 + A_t-1 (WITH gradient); exec_t:
        A_t-1 in centimetres; self._nf_ach (not yet refreshed): the achieved pose of S_t-1;
        self._contact_prev_state: the patch states at S_t-1. The target and the onsets are the
        world's report on the burst A_t-1 drove, which produced this moment's frames: the
        prediction was formed from nothing later than the moment before the burst.

        Returns {"contact_head": weighted loss} (absent when nothing was graded). Always sets
        self._contact_control_value ([B], NaN for a body that could not be scored, which the
        normalised ranking reads as the channel's centre) and refreshes the previous-moment
        cache, so a moment that cannot be graded still serves as the next one's past."""
        from lem.model import contact_head as CH
        dev = self.device
        head, tr = self.contact_head, self.contact_tracker
        cur, cur_ok, onsets, wrep, mismatch = self._contact_report(frames, B)
        tr.win["patch_count_mismatch"] += int(mismatch)
        self._contact_control_value = torch.full((B,), float("nan"), device=dev)
        self.last_contact_control = None
        prev = self._contact_prev_state
        prev_ok = self._contact_prev_valid
        out = {}
        ready = (latent is not None and exec_t is not None and prev is not None
                 and prev_ok is not None and self._nf_ach is not None
                 and int(prev.shape[0]) == B and int(latent.shape[0]) == B
                 and int(self._nf_ach.shape[0]) == B and int(exec_t.shape[0]) == B)
        if ready:
            prev_np = (prev.detach().cpu().numpy() if torch.is_tensor(prev)
                       else np.asarray(prev)).astype(np.int64)
            prev_okn = (prev_ok.detach().cpu().numpy() if torch.is_tensor(prev_ok)
                        else np.asarray(prev_ok)).astype(bool)
            valid = cur_ok & prev_okn
            prev_in = self._contact_prev_input(prev_np, dev)
            ofeat = CH.order_features(exec_t, self._nf_ach, self.GAZE_IDX)
            if head.centre_weight <= 0.0:
                # the first lived moment ever: the centre starts as that moment, never as zero
                # (a zero centre would hand the LayerNorm the shared part it exists to remove)
                head.update_centre(latent)
            logits, pre = head(latent, ofeat, prev_in, return_pre=True)
            if bool(valid.any()):
                # targets and mask stay on the CPU: the loss counts its masks there
                terms = CH.ContactHead.loss(logits, CH.state_bits(torch.from_numpy(cur)),
                                            torch.from_numpy(valid))
                if terms:
                    out["contact_head"] = self.contact_head_weight * torch.stack(
                        list(terms.values())).mean()
            with torch.no_grad():
                tr.add_liveness(pre)
                q = CH.ContactHead.probabilities(logits.detach())
                qc = None
                if B > 1 and int(step) % int(self.contact_control_period) == 0:
                    qc = self._contact_control_forward(nf_live, nf_memory_bias, exec_t, prev_in, B)
                # the head's one device read of the step: this moment's probabilities (and the
                # control's); the window's liveness is read once per report
                q_np = q.float().cpu().numpy()
                qc_np = None if qc is None else qc.float().cpu().numpy()
                if self.observer_per_body_errors:
                    # the head's probabilities for THIS moment (formed from the moment before and
                    # the order in force) beside each patch's base rate of starting, BEFORE this
                    # moment updates them: what the observer compares an observed onset against
                    self.observer_contact = {
                        "step": int(step), "q": q_np.astype(np.float16),
                        "base_rate_start": np.asarray(tr.rates()[:, 0, :], dtype=np.float16),
                        "valid": np.asarray(cur_ok, dtype=bool)}
                # a lived moment: the latent's running mean moves (after it was used, so the
                # prediction never saw its own moment in its centre)
                head.update_centre(latent)
            if self._contact_habit is None or len(self._contact_habit) != B:
                self._contact_habit = [{} for _ in range(B)]
            pb = CH.state_bits(torch.from_numpy(prev_np)).numpy()
            cb = CH.state_bits(torch.from_numpy(cur)).numpy()
            # the payment is read only by the normalised ranking's contact_control channel
            # (review of b4, minor 6: paid_mean used to report payments that went nowhere)
            pay, detail = tr.score(int(step), q_np, pb, cb, valid, onsets, self._contact_habit,
                                   wrep, qc_np,
                                   pay_on=bool(self.contact_control_enabled
                                               and getattr(self, "wellness_rank_mode", "raw")
                                               == "normalized"))
            # what the ranking stores: the per-body running average of the payments (review of
            # b4, M2; contact_head.CONTACT_VALUE), NaN for a body not scored this moment
            self._contact_value_state, val_np = CH.running_value(
                self._contact_value_state, pay, valid, self.efference_tenure_steps)
            tr.note_value(val_np)
            self._contact_control_value = torch.tensor(val_np, dtype=torch.float32, device=dev)
            self.last_contact_control = {"step": int(step), "valid": [bool(v) for v in valid],
                                         "control": qc is not None, **detail}
        # this moment becomes the next one's past
        self._contact_prev_state = torch.from_numpy(cur)
        self._contact_prev_valid = torch.from_numpy(cur_ok)
        return out

    @torch.no_grad()
    def _swap_test_arms(self, nf_raw, nf_in, nf_memory_bias, exec_t, step: int, B: int,
                        absolute: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """The swap test's added arms (2026-10-02, docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md):
        {arm: {sense: predicted next token [B, d_model]}} for
          size_matched  another body's command size-matched to this body (lem/model/efference.py
                        size_matched_other_order: lem/model/contact_head.shuffled_order for the 25 body rows; the
                        eyes' row as the other body's asked eye turn applied to this body's eyes, whose angles at
                        the start are _eff_eye_prev, kept with --efference-heads on; without them the eyes' row is
                        the other body's as it stands);
          shifted_own   this body's own command plus one standard deviation of Gaussian noise per column
                        (prediction_audit.shifted_own_order; the deviations from _command_spread; absent before
                        prediction_audit.COMMAND_SPREAD_MIN_STEPS model steps of stream);
          shuffled      only with absolute=True (the hold's first model step when the existing block does not also
                        grade it): another body's absolute command, as the existing block builds it.
        Each arm rebuilds only the motor token - the cached previous model step's achieved pose with the arm's
        command in the 9 command columns, encoded by the motor encoder - and runs the transformer with this body's
        other tokens (nf_in, the delayed lesson's), context and memory bias, then the prediction heads and the gaze
        copy with the arm's eyes' row, as the existing block does. Inside liveness.unwatched: the repaired layers
        route their revived units as always, but no row joins a watching window and the monitor records nothing,
        so no weight of the step can depend on these passes. {} when the cached motor row is not this build's width
        or was not a real reading (the existing block's swap does not apply then either)."""
        from lem.model.liveness import unwatched
        from lem.model.efference import size_matched_other_order, apply_gaze_copy
        from .prediction_audit import command_spread_std, shifted_own_order
        mt = nf_raw.get('motor') if isinstance(nf_raw, dict) else None
        if (not torch.is_tensor(mt) or mt.shape[-1] != MOTOR_INPUT_F or mt.shape[0] != B
                or (self._nf_mask or {}).get('motor') != 'real'
                or self._nf_ach is None or self._nf_ach.shape[0] != B):
            return {}
        eye = getattr(self, "_eff_eye_prev", None)
        if not (torch.is_tensor(eye) and tuple(eye.shape) == (B, 2)):
            eye = None
        cmds = {}
        if absolute:
            cmds["shuffled"] = exec_t.roll(1, dims=0)
        cmds["size_matched"] = size_matched_other_order(exec_t, self._nf_ach, self.GAZE_IDX, eye_deg=eye,
                                                        convention=self.eye_row_convention).to(exec_t.dtype)
        std = command_spread_std(self._command_spread)
        if std is not None:
            cmds["shifted_own"] = shifted_own_order(exec_t, std, step, self.efference_seed)
        out = {}
        kw = {} if nf_memory_bias is None else {"memory_bias": nf_memory_bias.detach()}
        with unwatched(self):
            for arm, cmd in cmds.items():
                row = torch.cat([mt[..., :MOTOR_ACHIEVED_F], command_for_encoder(cmd.to(mt.dtype))], dim=-1)
                z = self.encoder_registry.encode({"motor": row}, {"motor": "real"})["motor"]
                toks = {m: (z.reshape(B, 1, self.d_model) if m == "motor"
                            else nf_in[m].detach().reshape(B, 1, self.d_model)) for m in self.active}
                _seq, last, _ctx = self.forward_model(toks, self._nf_ctx, **kw)
                preds = self.prediction_heads(
                    last, current={m: v.reshape(B, self.d_model) for m, v in toks.items()},
                    who="swap test arm " + arm)
                if self.efference is not None:
                    preds = apply_gaze_copy(self.prediction_heads, preds, last, command_for_encoder(cmd),
                                            self._nf_ach)
                out[arm] = preds
        return out

    @torch.no_grad()
    def _contact_control_forward(self, nf_live, nf_memory_bias, exec_t, prev_in, B):
        """The contact probabilities with ANOTHER body's order (the next body's, as a change from
        its own pose applied to this body's pose, so its asked moves and turns are exactly the size
        of a real order: contact_head.shuffled_order, review of b4 minor 2) given both ways the head
        gets it: rebuilt into the motor token the core reads, and handed to the head directly.
        Every other sense's token, the context and the memory bias are this body's own. The null of
        "the prediction depended on this body's order". No gradient; the centre does not move. None
        when the cached motor row is not the width this build encodes."""
        from lem.model import contact_head as CH
        if nf_live is None or self._nf_raw is None:
            return None
        nf_in, _ = nf_live
        mt = self._nf_raw.get("motor")
        if not (torch.is_tensor(mt) and mt.shape[-1] == MOTOR_INPUT_F and mt.shape[0] == B):
            return None
        rolled = CH.shuffled_order(exec_t, self._nf_ach, self.GAZE_IDX).to(exec_t.dtype)
        row = torch.cat([mt[..., :MOTOR_ACHIEVED_F], command_for_encoder(rolled.to(mt.dtype))], dim=-1)
        z = self.encoder_registry.encode({"motor": row}, {"motor": "real"})["motor"]
        toks = {m: (z.reshape(B, 1, self.d_model) if m == "motor"
                    else nf_in[m].detach().reshape(B, 1, self.d_model)) for m in self.active}
        kw = {} if nf_memory_bias is None else {"memory_bias": nf_memory_bias.detach()}
        _seq, last, _ctx = self.forward_model(toks, self._nf_ctx, **kw)
        lg = self.contact_head(last, CH.order_features(rolled, self._nf_ach, self.GAZE_IDX), prev_in)
        return CH.ContactHead.probabilities(lg)

    def contact_control_window_summary(self) -> Optional[dict]:
        """The `contact_control` block of the stats record, and a fresh window; None when
        --contact-head is off. Every count is over the window and all bodies.

        onsets_self: self-touches that began (the world's report, pair level), structural pairs
        (touching at rest under zero command) apart in onsets_self_structural and never scored;
        scorable_self: of those, onsets with a side on a skin patch no own part touched the moment
        before (a patch already touched cannot show a new touch) and whose patches have a base rate
        and a line (unlined_self: waiting for a line); predicted_self: scorable onsets where, on
        every scorable side, the head's probability of an own part touching minus the patch's base
        rate of STARTING such a touch exceeded THAT PATCH'S line (the 95th percentile of the same
        difference over its untouched moments; lines: how many patches have one), formed the moment
        before; success_self = predicted / scorable (0 = never foreseen, 1 = every one),
        success_self_both the same over onsets with both sides scorable. onsets_world /
        success_world: patches an outside thing began to touch (off over the whole previous burst).
        false_alarm_*: share of patch-moments that stayed untouched where the patch's line was
        crossed anyway (the design gives 5 in 100 on every patch; false_alarm_by_region per body
        region). onset_auc_self / onset_auc_world: on patch-moments whose bit was off the moment
        before, how often one that became touched scored above one that stayed untouched (0.5 = a
        coin flip, 1 = every time); control_onset_auc_*: the same for each patch's base rate of
        starting that touch - the head knows something the patch's own history does not only when
        it scores above its control. success_control: success_self when ANOTHER body's order, as
        a change of the same size applied to this body's pose, goes through the core and the head
        (every control_period-th moment), success_paired: this body's own order on those same
        onsets (compare these two, never success_self with success_control).
        would_pay_mean: per body-moment, 2 per foreseen self-touch pair and 1 per body-moment with
        a foreseen outside touch, times the habituation factor 1 / (1 + onsets of that pair or
        patch in the last 200 moments) (habituation_mean: that factor averaged over the paid
        events, 1 = a new pair, lower = repeated); paid_mean: what the payments were with each
        kind's own gate applied, 0 when nothing reads them (the channel off or the ranking raw);
        value_mean: what the ranking stored, the per-body running average of the payments.
        gate_self_* / gate_world_*: one evidence gate per kind of touch; a reading per control
        moment is (foreseen with own order) - (foreseen with another body's), averaged over that
        moment's onsets of the kind; over the last 32 readings the gate opens above 3 times their
        own standard error and closes below 1; it shuts when its newest reading is over 200
        moments old, and at every load (gate_open: either is open; gate_why: both texts).
        head_bce / control_bce: cross-entropy of the head and of the base-rate control
        (each patch's rate given whether it was touched the moment before), mean over the three
        bits; lower is better, equal = the head knows no more than the base rates. head_auc: how
        often a touched patch-moment scored above an untouched one over ALL patch-moments (touches
        that simply continue included, so it measures persistence more than foresight), per bit
        in head_auc_bits beside control_auc_bits. liveness: share of each layer's units whose input
        rose above -3 (DESIGN section 10); pool_spread: none, the head pools nothing.
        channel.rank_winner_margin_share: this window's share of the goal ranking's winning margins
        that contact_control carried (wellness_rank.winner_margin_share, read before that block
        resets), beside channel.rank_weight_share, the share its weight would give it."""
        if self.contact_head is None:
            return None
        live = self.contact_tracker.liveness_summary()
        out = self.contact_tracker.summary(liveness=live)
        out["control_period"] = int(self.contact_control_period)
        in_ranking = bool(self.contact_control_enabled
                          and getattr(self, "wellness_rank_mode", "raw") == "normalized")
        out["channel"] = {"weight": float(self.contact_control_weight), "in_ranking": in_ranking}
        wc = getattr(self, "wellness_channels", None)
        shares = getattr(wc, "shares", None) if wc is not None else None
        if in_ranking and shares is not None:
            # the ranking's own tally so far this window (its block, written after this one, resets
            # it); the weights in force give the share contact_control's weight alone would earn
            wms = (shares.summary() or {}).get("winner_margin_share") or {}
            out["channel"]["rank_winner_margin_share"] = wms.get("contact_control")
            try:
                w_now = self._rank_weights_now()
                names = [n for n in self._rank_channel_names if n in w_now]
                tot = sum(abs(float(w_now[n])) for n in names)
                out["channel"]["rank_weight_share"] = (
                    None if tot <= 0 else round(abs(float(w_now.get("contact_control", 0.0))) / tot, 4))
            except (AttributeError, TypeError):
                out["channel"]["rank_weight_share"] = None
        return out

    @torch.no_grad()
    def predict_contacts(self, state=None, order_cm=None, patch_state=None) -> Optional[dict]:
        """What the contact head expects `order_cm` to do to the skin in the next moment.

        state: a PredictiveState (default: live_predictive_state(), the last lived moment);
        order_cm: [B, n_handles, 9] (or [n_handles, 9]) an order in EXECUTION units (position cm,
        quaternion, radius, strength), the eyes in the gaze handle; patch_state: [B, 53] ints
        (0 none, 1 outside, 2 own part, 3 both) the skin at `state` (default: the last lived
        moment's report). Returns CPU tensors [B, 53]: p_contact (anything touches the patch),
        p_self (an own part touches it), p_world (an outside thing touches it), plus patch_names
        and horizon_ms. None when there is no state or no patch state."""
        if self.contact_head is None:
            raise RuntimeError("predict_contacts needs --contact-head on")
        from lem.model import contact_head as CH
        state = self.live_predictive_state() if state is None else state
        if state is None or order_cm is None:
            return None
        order = torch.as_tensor(np.asarray(order_cm, dtype=np.float32), device=self.device)
        if order.ndim == 2:
            order = order.unsqueeze(0)
        Bn = int(order.shape[0])
        ps = self._contact_prev_state if patch_state is None else patch_state
        if ps is None:
            return None
        ps = (ps.detach().cpu().numpy() if torch.is_tensor(ps) else np.asarray(ps)).astype(np.int64)
        if ps.ndim == 1:
            ps = ps[None]
        if ps.shape != (Bn, CH.N_PATCHES):
            raise ValueError(f"patch_state must be [{Bn}, {CH.N_PATCHES}], got {tuple(ps.shape)}")
        order_enc = command_for_encoder(order)
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.dtype):
            latent = self._latent_for_order(state, order_enc)
            lg = self.contact_head(latent, CH.order_features(order, state.achieved, self.GAZE_IDX),
                                   self._contact_prev_input(ps, self.device))
        p = CH.ContactHead.probabilities(lg).float().cpu()
        return {"p_contact": p[..., 0], "p_self": p[..., 1], "p_world": p[..., 2],
                "patch_names": list(CH.PATCH_NAMES), "horizon_ms": 1000.0 * float(self.step_seconds)}

    # ------------------------------------------------------------------ predictions for probes
    def live_predictive_state(self):
        """The state the model is in right now, on the LEARNING route: the RAW encoder tokens of
        the last step (never the top-down-gated blend), the context that step's forward
        attended to (before the shift), its achieved pose and its memory bias. None before the
        first step or right after a reset."""
        from lem.model.predictive_state import PredictiveState
        if self.last_tokens is None or self.last_achieved is None:
            return None
        hm = None
        if getattr(self, "efference", None) is not None:
            # the head's motion of the same moment (the cached raw frame is this step's after
            # step() returns), read by the efference gaze head under --vestibular-sense on
            from lem.model.efference import head_motion_of
            hm = head_motion_of(self, self._nf_raw)
        return PredictiveState(tokens=dict(self.last_tokens),
                               context=(self.last_context if self.last_context is not None
                                        else self.context_buffer),
                               achieved=self.last_achieved, domain="real",
                               memory_bias=self._memory_bias_current, head_motion=hm)

    def _latent_for_order(self, state, order_enc):
        """The forward model's last-step tokens for `state` with the motor token rebuilt from
        the state's achieved pose and `order_enc` (encoder units) - the same route transition()
        and the delayed next-frame lesson take."""
        from lem.model.encoder_registry import build_motor_input, imagined_motor_token
        reg = self.encoder_registry
        Bn = order_enc.shape[0]
        # an imagined row (a reading before execution): with --motor-asked-change on the path's weights are detached
        z = imagined_motor_token(reg, build_motor_input(reg, order_enc, achieved=state.achieved)
                                 ).reshape(Bn, 1, self.d_model)
        toks = {m: (z if m == "motor" else state.tokens[m].reshape(Bn, 1, self.d_model))
                for m in self.active}
        kw = {} if state.memory_bias is None else {"memory_bias": state.memory_bias}
        _seq, last, _ctx = self.forward_model(toks, state.context, **kw)
        return last

    @torch.no_grad()
    def predict_efference(self, order_cm, state=None) -> Optional[dict]:
        """What the efference heads expect `order_cm` to do, in centimetres and degrees.

        order_cm: [B, n_handles, 9] (or [n_handles, 9]) an order in EXECUTION units - the form
        the body receives (position cm, quaternion, radius, strength), with the eyes' command
        in the gaze handle's quaternion (LemLite.gaze_to_handle). state: a PredictiveState
        (default: live_predictive_state()). Returns tensors on the CPU:
          body_next_cm / body_next_deg      [B, n_handles, 3]  the next 50 ms
          body_tenure_cm / body_tenure_deg  [B, n_handles, 3]  the order's tenure
          gaze_deg                          [B, 3, 2]          az/el change at 50/100/150 ms
        plus handle_names and the horizons in milliseconds. None when there is no state."""
        if self.efference is None:
            raise RuntimeError("predict_efference needs --efference-heads on")
        from lem.model.efference import EfferenceHeads, GAZE_HORIZONS
        state = self.live_predictive_state() if state is None else state
        if state is None:
            return None
        order = torch.as_tensor(np.asarray(order_cm, dtype=np.float32), device=self.device)
        if order.ndim == 2:
            order = order.unsqueeze(0)
        order_enc = command_for_encoder(order)
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.dtype):
            latent = self._latent_for_order(state, order_enc)
            # efference_v2: the heads read the pose the order starts from (the asked change, the
            # head's and the eye's orientation) and, with --vestibular-sense on, the head's motion
            ach = state.achieved.to(latent.device)
            if ach.shape[0] != latent.shape[0]:
                ach = ach.expand(latent.shape[0], *ach.shape[1:])
            hm = getattr(state, "head_motion", None)
            body = self.efference.body(latent, order_enc, ach)
            gaze = self.efference.gaze(latent, order_enc, ach,
                                       None if hm is None or hm.shape[0] != latent.shape[0] else hm)
        out = {k: v.float().cpu() for k, v in EfferenceHeads.to_lived(body, gaze).items()}
        out["handle_names"] = list(self.handle_names)
        ms = 1000.0 * float(self.step_seconds)
        out["horizons_ms"] = {"next": ms, "tenure": ms * self.efference_tenure_steps,
                              "gaze": [ms * h for h in GAZE_HORIZONS]}
        return out

    @torch.no_grad()
    def predict_before_execution(self, order_cm, gaze_az_el=None, depth: int = 1) -> Optional[dict]:
        """The model's prediction of what the order about to be sent will do, BEFORE it is sent.

        order_cm: [B, n_handles, 9] the order in execution units (the trainer's send_motor).
        gaze_az_el: [B, 2] radians, the eye command issued with it (written into the gaze
        handle; the saccade generator and the reflex may still change what the eyes do, and
        the order the model is later told it executed carries the realised command).
        depth: how many moments to roll the held order forward (the trainer passes its
        motor_hold, so the last depth is the order's end).

        LEARNING ROUTE: the raw tokens of the step just taken, the context that step attended
        to, its achieved pose and memory bias (live_predictive_state), through transition()
        - whose head outputs pass the registry's output normalisation - with the context
        advancing one moment per depth and the achieved pose held at the latch pose (a
        predicted proprioception token cannot be decoded back into a pose). No gradient, no
        draw from any random stream.

        Returns {"depths": {1: {"tokens": {sense: [B, d]}, "judge": [B, 5]}, depth: {...}},
        "tokens_now", "judge_now" (the approximate evaluator on the present), "epoch" (the
        reset counter) and "B"}, or None when the model has no present state."""
        from lem.model.predictive_state import transition
        state = self.live_predictive_state()
        if state is None:
            return None
        o = np.asarray(order_cm, dtype=np.float32)
        if o.ndim == 2:
            o = o[None]
        if gaze_az_el is not None:
            o = LemLite.gaze_to_handle(o, np.asarray(gaze_az_el, dtype=np.float32),
                                       convention=getattr(self, "eye_row_convention", "legacy"))
        cmd = command_for_encoder(torch.as_tensor(o, device=self.device))
        depth = max(1, int(depth))
        out = {}
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.dtype):
            st = state
            for k in range(1, depth + 1):
                st = transition(self.forward_model, self.encoder_registry, self.prediction_heads,
                                st, cmd, advance_context=True)
                if k == 1 or k == depth:
                    preds = {m: v.detach().float() for m, v in st.tokens.items() if m != "motor"}
                    out[k] = {"tokens": preds, "judge": self.wellness.score_predicted_wellness(
                        preds, st.tokens["motor"].float()).float()}
            now = {m: v.detach().float() for m, v in state.tokens.items() if m != "motor"}
            judge_now = self.wellness.score_predicted_wellness(
                now, state.tokens["motor"].float()).float()
        return {"depths": out, "tokens_now": now, "judge_now": judge_now,
                "epoch": int(self._transition_epoch), "B": int(cmd.shape[0])}

    # ------------------------------------------------------------------ the causal chain
    @torch.no_grad()
    def _independent_babble(self, order, ns, cur_cm, reach, strength_limit):
        """Apply the production independent exploration, before body smoothing.

        All quaternion components must read the original rotation. In-place
        component writes previously changed the source used by later components.
        The paired physical probe calls this same path.
        """
        out = order.clone()
        noise = torch.randn(out.shape, generator=self.noise_rng).to(out.device) * ns
        out[..., :3] += noise[..., :3] * 100.0
        out[..., :3] = cur_cm + (out[..., :3] - cur_cm).clamp(-reach, reach)
        out[..., 8] = (out[..., 8] + noise[..., 8].abs()).clamp(0.0, strength_limit)
        rv = torch.randn(out.shape[0], out.shape[1], 3,
                         generator=self.noise_rng).to(out.device) * (10.0 * ns)
        angle = rv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dq = torch.cat([rv / angle * torch.sin(angle / 2), torch.cos(angle / 2)], dim=-1)
        x1, y1, z1, w1 = dq.unbind(-1)
        x2, y2, z2, w2 = out[..., 3:7].unbind(-1)
        out[..., 3:7] = torch.stack([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2], dim=-1)
        return out

    def _causal_perturb(self, action: torch.Tensor, gen: torch.Generator, ns: float = 0.05):
        """One seeded perturbation of a planner action (raw form: position in command units
        of 200 cm, unit quaternion, radius, strength), at the babble's own sizes: the run
        executes exploration noise 0.05 (--babble-noise, lem_model babble block), which is 5 cm
        per axis of position, strength + |N(0, 0.05)|, and a rotation wobble of 0.5 rad per
        axis. Drawn from `gen` only; the global random streams are never touched."""
        out = action.detach().float().clone()
        B, H, _ = out.shape
        noise = torch.randn(out.shape, generator=gen).to(out.device) * ns
        out[..., 0:3] = out[..., 0:3] + noise[..., 0:3] * (100.0 / COMMAND_POSITION_RANGE_CM)
        out[..., 8] = (out[..., 8] + noise[..., 8].abs()).clamp(0.0, 1.0)
        rv = torch.randn(B, H, 3, generator=gen).to(out.device) * (10.0 * ns)
        ang = rv.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        dq = torch.cat([rv / ang * torch.sin(ang / 2), torch.cos(ang / 2)], dim=-1)
        x1, y1, z1, w1 = dq.unbind(-1)
        x2, y2, z2, w2 = out[..., 3:7].unbind(-1)
        out[..., 3:7] = torch.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dim=-1)
        return out

    CAUSAL_ALTERNATIVES = ("hold", "reflex", "perturb_1", "perturb_2")

    @torch.no_grad()
    def _causal_execution_candidates(self, cands, decision, *, hand_joint_goals=None):
        """Apply the live command contract to comparison actions.

        A planned hand candidate supplies its explicit pre-filter joint goal.
        Other alternatives use the ordinary controller's projection after its
        developmental limits, then the same single joint/body filter.
        """
        K = len(cands)
        def rep(t):
            return None if t is None else t.repeat(K, *([1] * (t.dim() - 1)))
        allc = torch.cat(cands, dim=0)
        achieved = rep(decision.get('achieved'))
        execution = self.planned_execution(allc, decision['dev'], handle_state=achieved,
            previous=rep(decision.get('smooth_prev')), smoothing=decision.get('smoothing', 0.),
            keep_gaze_idx=decision.get('smooth_keep_gaze'),
            eye_row_convention=decision.get('eye_row_convention', 'legacy'),
            executed_fields=decision.get('executed_fields_only', False),
            position_rows=decision.get('position_rows', 'all'))
        if self.hand_target_mode != 'world':
            if hand_joint_goals is None:
                raw = self.differentiable_execution_transform(allc, decision['dev'], handle_state=achieved)
                _, joints, _ = self.hand_kinematics.project_world_orders(raw.detach().cpu().numpy(), position_scale=1.)
                joints = torch.as_tensor(joints, device=execution.device, dtype=execution.dtype)
            else:
                if len(hand_joint_goals) != K:
                    raise ValueError('Each hand comparison needs its explicit raw joint goal')
                joints = torch.cat(hand_joint_goals).to(execution)
            joints = self._smooth_hand_joints(joints, rep(decision.get('hand_previous_joints')),
                                             decision.get('smoothing', 0.))
            execution = self.hand_kinematics.assemble_world(execution, joints, position_scale=1.)
        return execution

    @torch.no_grad()
    def _causal_score_candidates(self, cands, st0, decision, *, hand_joint_goals=None):
        """The alternatives' route of a planning record: each raw candidate [B, H, 9] through the
        decision's own execution transform and smoother, transition() from the depth-1 state the
        planner started from, and the approximate evaluator. Returns (predicted tokens {sense:
        [K*B, d]}, the evaluator's five terms [K*B, 5]). A test scores the planner's own first
        action this way and requires the planner's own predicted wellness (the route check the
        review of b1 measured at a difference of 0.0)."""
        from lem.model.predictive_state import PredictiveState, transition
        K = len(cands)
        ach = st0.achieved
        def rep(t):
            return None if t is None else t.repeat(K, *([1] * (t.dim() - 1)))
        exec_c = self._causal_execution_candidates(cands, decision, hand_joint_goals=hand_joint_goals)
        stK = PredictiveState(tokens={m: rep(v) for m, v in st0.tokens.items()},
                              context=rep(st0.context), achieved=rep(ach), domain="real",
                              memory_bias=rep(st0.memory_bias),
                              head_motion=rep(getattr(st0, "head_motion", None)))
        nxt = transition(self.forward_model, self.encoder_registry, self.prediction_heads,
                         stK, command_for_encoder(exec_c), advance_context=True)
        preds = {m: v for m, v in nxt.tokens.items() if m != "motor"}
        return preds, self.wellness.score_predicted_wellness(preds, nxt.tokens["motor"]).float()

    @torch.no_grad()
    def _causal_plan_record(self, *, step, best_t, wl_stack, goal_state, motor_out, plan0,
                            imeta, decision, exec_t, max_steps, locality, rank_info=None):
        """One planning event: the goal, the chosen action's predicted next state and predicted
        wellness, the same for four alternatives scored from the SAME captured state, and the
        chosen action's rank.

        THE ROUTE IS THE PLANNER'S OWN: the top-down-gated tokens it descends from, the
        decision's captured context, pose, memory bias and smoother state, the same
        execution transform, transition() at full precision, and the approximate evaluator
        (Judge B, five terms: prediction_error, tactile_comfort, capability, coherence,
        motor_state, each 0-1). The chosen action's numbers are the planner's own depth-1
        values (_plan_chain meta["depth1"]); the alternatives are:
          hold       every handle's target = where it is now, same strength as the plan, the
                     eyes held at the last realised eye command
          reflex     the motor controller's own proposal this step
          perturb_1/2  the plan plus babble-sized noise from a generator seeded by
                     (the run's seed, this step)
        rank 1 = the chosen action has the highest predicted wellness sum of the five."""
        from lem.model.predictive_state import PredictiveState, transition
        d1 = imeta.get("depth1")
        if d1 is None:
            return {"step": int(step), "error": "the planner kept no depth-1 prediction"}
        B = plan0.shape[0]
        st0 = d1["state"]
        ach = st0.achieved
        gi = self.GAZE_IDX
        hold = plan0.clone()
        if ach is not None:
            hold[..., 0:3] = ach[..., 0:3].float()
            q = ach[..., 3:7].float()
            hold[..., 3:7] = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        if exec_t is not None and exec_t.shape[0] == B:
            hold[:, gi, 3:7] = exec_t[:, gi, 3:7].float()
        else:
            hold[:, gi] = plan0[:, gi]
        gen = torch.Generator(device="cpu")
        gen.manual_seed((int(self._causal_seed) * 1_000_003 + int(step)) % (2 ** 62))
        cands = [hold, motor_out.detach().float(), self._causal_perturb(plan0, gen),
                 self._causal_perturb(plan0, gen)]
        K = len(cands)
        preds, well_alt = self._causal_score_candidates(cands, st0, decision)
        well_ch = d1["well"].float()
        sums = torch.cat([well_ch.sum(-1).unsqueeze(0), well_alt.sum(-1).reshape(K, B)], dim=0)
        from mimo_bridge.causal_chain import rank_of_chosen
        rank = rank_of_chosen(sums[0], sums[1:])
        smask = decision.get("source_mask") or {}
        real = [m for m in preds if m in goal_state and str(smask.get(m, "real")) == "real"]
        gd_alt = torch.zeros(K, B, device=sums.device)
        gd_ch = torch.zeros(B, device=sums.device)
        for m in real:
            g = goal_state[m].float()
            gd_alt = gd_alt + ((preds[m].float().reshape(K, B, -1) - g.unsqueeze(0)) ** 2).mean(-1)
            gd_ch = gd_ch + ((d1["tokens"][m].float() - g) ** 2).mean(-1)
        # the goal's value and what it is made of, at the moment it was chosen
        T = len(self.goal_ring)
        bt = best_t.view(-1).long()
        ranked = wl_stack.gather(0, bt.view(1, -1)).squeeze(0).float()
        stored = torch.stack([w for _, w in self.goal_ring]).gather(0, bt.view(1, -1)).squeeze(0)

        def ring(dq):
            if len(dq) != T:
                return None
            return torch.stack([torch.as_tensor(x, device=ranked.device, dtype=torch.float32)
                                .expand(B) for x in dq]).gather(0, bt.view(1, -1)).squeeze(0)
        w0, cq, drv = ring(self._goal_ring_w0), ring(self._goal_ring_cq), ring(self._goal_ring_drive)
        cons = None if cq is None else cq * float(self.consequence_weight)
        other = (None if w0 is None or cons is None or drv is None
                 else stored / (1.0 + drv) - w0 - cons)
        cur = st0.tokens
        exf = d1["exec_form"].float()
        body = [i for i in range(len(self.handle_names)) if i != gi]
        step_cm = ((exf[:, body, 0:3] - ach[:, body, 0:3].float() * COMMAND_POSITION_RANGE_CM)
                   .norm(dim=-1).mean(-1)) if ach is not None else None
        # THE SHARE OF THE PLANNED STEP THE REACH CLAMP REMOVED (review MINOR 2), per body: the
        # plan's position step (its target, clipped to the world range, minus where each body
        # handle is) against the same step after the per-axis reach clamp; 0 = the clamp removed
        # nothing, 1 = it removed the whole step. Handles the plan asked to stay put are left out.
        clamp_share = None
        if ach is not None:
            _cur_cm = ach[:, body, 0:3].float() * COMMAND_POSITION_RANGE_CM
            _raw = plan0[:, body, 0:3].float().clamp(-1.0, 1.0) * COMMAND_POSITION_RANGE_CM - _cur_cm
            _reach = 40.0 * float((decision.get("dev") or {}).get("rom_limit", 1.0))
            _rn = _raw.norm(dim=-1)
            _cn = _raw.clamp(-_reach, _reach).norm(dim=-1)
            _asked = _rn > 1e-6
            clamp_share = torch.where(_asked, 1.0 - _cn / _rn.clamp_min(1e-6),
                                      torch.zeros_like(_rn)).sum(-1) / _asked.sum(-1).clamp_min(1)
        gz = torch.as_tensor(self._gaze_from_handle(plan0), device=ranked.device)

        def L(t, nd=4):
            return None if t is None else [round(float(v), nd) for v in t.reshape(-1).tolist()]
        def scores(t):
            # Preserve the values used to compare alternatives. Display rounding
            # can turn a real ordering into a tie and contradict chosen_rank.
            return [float(v) for v in t.detach().reshape(-1).tolist()]
        rec = {"step": int(step), "bodies": int(B), "horizon": int(max_steps),
               "optimizer": imeta.get("descent"),
               "hand_optimizer": imeta.get('hand_descent'),
               "optimizer_stage_order": ['sensory_or_physical_body', 'wrist_relative_hand', 'final_prediction'],
               "sensory_converged": bool(imeta.get('sensory_converged')),
               "eye_optimizer": ([dict(imeta['gaze_descent'], depth=1)]
                   if (imeta.get('gaze_descent') or {}).get('selection') else []),
               "eye_optimizer_status": ((imeta.get('gaze_descent') or {}).get('stopped_by')),
               "route": "acting: the top-down-gated tokens the planner descends from",
               "goal": {"ring_index": [int(v) for v in bt.tolist()],
                        "age_steps": [int(T - 1 - v) for v in bt.tolist()], "ring_size": int(T),
                        "value_ranked": L(ranked), "value_stored": L(stored), "bodily": L(w0),
                        "consequence": L(cons),
                        "curiosity_factor": None if drv is None else L(1.0 + drv),
                        "other_drives": L(other), "locality": bool(locality)},
               "plan0": {"pos_step_cm": L(step_cm, 3),
                         "strength_mean": L(plan0[:, body, 8].mean(-1), 3),
                         "gaze_az_deg": L(gz[:, 0] * (180.0 / math.pi), 2),
                         "gaze_el_deg": L(gz[:, 1] * (180.0 / math.pi), 2)},
               "pred_change": {m: L((d1["tokens"][m].float() - cur[m].float()).pow(2).mean(-1).sqrt())
                               for m in d1["tokens"] if m in cur},
               "pred_tokens": {m: v.detach() for m, v in d1["tokens"].items()},
               # the order the planner imagined the body receiving (execution units, a tensor kept
               # out of the JSON line like pred_tokens): the recorder measures how far the order
               # really sent was from it (plan_exec_distance_cm, review MINOR 2)
               "exec_form": exf.detach(),
               "reach_clamp_removed_share": L(clamp_share),
               "pred_wellness_terms_chosen": L(well_ch.mean(0)),
               "pred_wellness": dict({"chosen": scores(sums[0])},
                                     **{n: scores(sums[1 + i]) for i, n in enumerate(self.CAUSAL_ALTERNATIVES)}),
               "goal_distance": dict({"chosen": scores(gd_ch)},
                                     **{n: scores(gd_alt[i]) for i, n in enumerate(self.CAUSAL_ALTERNATIVES)}),
               "chosen_rank": [int(v) for v in rank.tolist()],
               "alternatives": list(self.CAUSAL_ALTERNATIVES)}
        if rank_info is not None:
            # THE NORMALISED RANKING OF THIS GOAL (DESIGN D8): per channel, what the chosen moment
            # contributed (weight x its distance from the channel's centre in scales) and its lead
            # over the runner-up; the channels the approximate evaluator cannot see in an imagined
            # moment are named, so no reader takes Judge B's sums for the whole value
            from lem.model.wellness_channels import JUDGE_B_VIEW, UNRANKED_ROWS

            def L(t, nd=4):       # JSON has no NaN: an unmeasured value is written as null
                return None if t is None else [round(float(v), nd) if math.isfinite(float(v))
                                               else None for v in t.reshape(-1).tolist()]
            cb = rank_info["contrib"]                                    # [T, C, B]
            names = rank_info["names"]
            wr = wl_stack.float()
            second = wr.clone()
            second.scatter_(0, bt.view(1, -1), float("-inf"))
            j = second.argmax(dim=0)
            has2 = torch.isfinite(second.gather(0, j.view(1, -1)).squeeze(0))
            chosen_c = cb.gather(0, bt.view(1, 1, -1).expand(1, cb.shape[1], -1)).squeeze(0)
            second_c = cb.gather(0, j.view(1, 1, -1).expand(1, cb.shape[1], -1)).squeeze(0)
            _wcs = self.wellness_channels
            ch_now = (_wcs.last_raw if getattr(_wcs, "last_raw", None) is not None
                      else (self._goal_ring_ch[-1] if len(self._goal_ring_ch) else None))
            rec["wellness_rank"] = {
                "contrib_chosen": {n: L(chosen_c[i], 3) for i, n in enumerate(names)
                                   if n not in UNRANKED_ROWS},
                "lead_over_runner_up": {n: L(torch.where(has2, chosen_c[i] - second_c[i],
                                                         torch.full_like(chosen_c[i], float("nan"))), 3)
                                        for i, n in enumerate(names) if n not in UNRANKED_ROWS},
                "gated_share": L(rank_info["gated"].float().mean(0), 3),
                "all_gated": [bool(v) for v in rank_info["all_gated"].tolist()],
                "channels_now": (None if ch_now is None else
                                 {n: L(ch_now[i], 4) for i, n in enumerate(names)}),
                "planner_blind": [n for n in names if JUDGE_B_VIEW.get(n) == "no"]}
        return rec

    # ------------------------------------------------------------------ the motor synergy (D11)
    def _synergy_stamp(self) -> dict:
        """What the synergy was built and trained under, for the receipt: a load compares it."""
        from lem.model import motor_synergy as MS
        return {"version": MS.SYNERGY_VERSION, "k": int(self.synergy.k),
                "change_scale_deg": float(MS.CHANGE_SCALE_DEG),
                "change_scale_cm": float(MS.CHANGE_SCALE_CM),
                "rot_reach_deg": {n: float(MS.ROT_REACH_DEG[n]) for n in self.handle_names
                                  if n in MS.ROT_REACH_DEG},
                "pos_reach_cm": {n: float(MS.POS_REACH_CM[n]) for n in self.handle_names
                                 if n in MS.POS_REACH_CM},
                "z_l2": float(MS.Z_L2), "tenure_steps": int(self.synergy_tenure_steps),
                "planner": bool(self.synergy_planner), "babble": bool(self.synergy_babble),
                "planner_start": "encoded_proposal",
                "residual_l2": self.synergy_residual_l2}

    def _syn_add(self, key: str, value, count: int = 1) -> None:
        s = self._syn_window.setdefault(key, [0.0, 0])
        s[0] += float(value)
        s[1] += int(count)

    def _synergy_lesson(self, exec_t, order_id, handle_state, step, B, train):
        """The hindsight lesson (DESIGN D11). Called every moment with the order that was in force
        while the body produced this moment (exec_t, centimetres) and the pose it produced
        (handle_state). An order is followed from the moment it is first seen: its begin pose is
        the previous moment's pose. When it has been in force for its whole tenure
        (synergy_tenure_steps moments, the trainer's motor hold) and is still the order in force,
        it becomes one sample per body: the begin pose, the change the body achieved over the
        tenure on the live channels (every body handle's turn, the position handles' moves), how
        well the body followed the order (following_cos, -1 to +1) and how much it moved (the
        change's root mean square in measured spreads).

        Each graded order is first predicted by the CURRENT synergy (encode, then S') before it is
        learned from, so the window's explained share is measured on changes the synergy has not
        yet trained on. It then joins a ring of recent samples, and the lesson grades a batch of
        up to LESSON_BATCH: the new samples plus a draw from the ring (a generator seeded from the
        run's seed and the step, so no global random stream moves). The loss is the samples'
        mean squared error in change scales, weighted by following (below zero counted as zero) x
        movement, plus Z_L2 x the weighted mean of z^2. Only the synergy learns from it: its inputs
        are the recorded poses and changes, never the core.

        Returns {"synergy": loss} on a moment that graded something, else {}."""
        from lem.model import motor_synergy as MS
        syn = self.synergy
        out = {}
        # handle_state is the 10-wide ACHIEVED state (step() widens only the encoder's copy)
        ach_now = handle_state.detach().float()
        prev = self._syn_prev_ach
        if exec_t is not None and prev is not None and prev.shape == ach_now.shape:
            if order_id is not None:
                changed = self._syn_prev_order_id is None or int(order_id) != self._syn_prev_order_id
                # A rest override can replace targets without changing the trainer's latch ID.
                # Such an interval is not a full tenure of the original order.
                if not changed and self._syn_prev_order is not None:
                    idx = syn.rot_idx
                    changed = not torch.equal(self._syn_prev_order[:, idx], exec_t[:, idx])
            else:
                gi = self.GAZE_IDX
                po = self._syn_prev_order
                changed = (po is None or po.shape != exec_t.shape or not torch.equal(
                    torch.cat([po[:, :gi], po[:, gi + 1:]], dim=1),
                    torch.cat([exec_t[:, :gi], exec_t[:, gi + 1:]], dim=1)))
            p = self._syn_pending
            if changed:
                # the order just seen began at the previous moment's pose; an order still pending
                # here ended before its tenure was full (a reset, or a re-latch) and is dropped
                execution = self._syn_execution or {}
                source = (execution.get("source") if execution.get("order_id") == order_id else None)
                self._syn_pending = {"ach": prev.clone(), "order": exec_t.detach().float().clone(),
                                     "step": int(step) - 1,
                                     "source": source or "unknown"}
            elif (p is not None and p["ach"].shape[0] == B
                  and int(step) - int(p["step"]) == self.synergy_tenure_steps):
                self._syn_pending = None
                out = self._synergy_grade(p, ach_now, step, B, train)
        self._syn_prev_ach = ach_now.clone()
        self._syn_prev_order_id = None if order_id is None else int(order_id)
        self._syn_prev_order = None if exec_t is None else exec_t.detach().clone()
        # the pose input's running mean learns from every lived moment
        if train:
            syn.update_centre(ach_now)
        return out

    def _synergy_grade(self, p, ach_now, step, B, train):
        from lem.model import motor_synergy as MS
        syn = self.synergy
        d = ach_now.device
        count = MS.following_counts(p["order"], p["ach"], ach_now, syn.rot_idx)
        author = p.get("source") or "unknown"
        self._syn_following.setdefault(author, []).append({k: v.cpu() for k, v in count.items()})
        rot, pos = syn.achieved_change(p["ach"], ach_now)
        x = syn.to_units(rot, pos)                                             # [B, n_out]
        cos = MS.following_cos(p["order"], p["ach"], ach_now, syn.rot_idx)    # [B]
        move = x.pow(2).mean(-1).sqrt()                                        # [B] RMS in spreads
        w = cos.clamp_min(0.0) * move
        ok = torch.isfinite(x).all(-1) & torch.isfinite(w)
        x = torch.where(ok.view(-1, 1), x, torch.zeros_like(x))
        w = torch.where(ok, w, torch.zeros_like(w))
        with torch.no_grad():
            # predicted BEFORE this sample is learned from: the window's explained share
            zh = syn.encode(x, p["ach"])
            xh = syn.from_bounded(*syn.delta(zh, p["ach"]))
            err = (xh - x).pow(2).sum(-1)
            tot = x.pow(2).sum(-1)
            vals = torch.stack([(w * err).sum(), (w * tot).sum(), err.sum(), tot.sum(),
                                cos.sum(), move.sum(), (w > 0).float().sum(), ok.float().sum()])
            v = vals.tolist()                                                  # one read
            self._syn_add("w_err", v[0], 0)
            self._syn_add("w_tot", v[1], 0)
            self._syn_add("err", v[2], 0)
            self._syn_add("tot", v[3], 0)
            self._syn_add("following", v[4], B)
            self._syn_add("movement", v[5], B)
            self._syn_add("weighted", v[6], B)
            self._syn_add("orders", v[7], 0)
            self._syn_z_window.append(zh[ok].detach().float().cpu())
            if len(self._syn_z_window) > 400:          # a window of 100 moments holds at most 20
                self._syn_z_window = self._syn_z_window[-400:]
            # the ring of recent samples
            H = ach_now.shape[1]
            bk = self._syn_bank
            if bk is None or bk["x"].shape[1] != x.shape[1] or bk["ach"].shape[1] != H:
                bk = self._syn_bank = {
                    "ach": torch.zeros(MS.BANK_SIZE, H, MOTOR_ACHIEVED_F, device=d),
                    "x": torch.zeros(MS.BANK_SIZE, x.shape[1], device=d),
                    "w": torch.zeros(MS.BANK_SIZE, device=d), "n": 0, "ptr": 0}
            keep = ok.nonzero().flatten()
            for b in keep.tolist():
                i = int(bk["ptr"])
                bk["ach"][i] = p["ach"][b]
                bk["x"][i] = x[b]
                bk["w"][i] = w[b]
                bk["ptr"] = (i + 1) % MS.BANK_SIZE
                bk["n"] = min(int(bk["n"]) + 1, MS.BANK_SIZE)
        n_new = int(ok.sum())
        n_old = int(self._syn_bank["n"]) - n_new
        take = max(0, min(MS.LESSON_BATCH - n_new, n_old))
        ach_b, x_b, w_b = p["ach"][ok], x[ok], w[ok]
        if take > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed((int(self._synergy_seed) * 1_000_033 + int(step)) % (2 ** 62))
            # draw only among the samples stored BEFORE this moment's
            older = [(int(self._syn_bank["ptr"]) - n_new - 1 - j) % MS.BANK_SIZE for j in range(n_old)]
            pick = torch.tensor(older)[torch.randperm(n_old, generator=gen)[:take]].to(d)
            ach_b = torch.cat([ach_b, self._syn_bank["ach"][pick]], 0)
            x_b = torch.cat([x_b, self._syn_bank["x"][pick]], 0)
            w_b = torch.cat([w_b, self._syn_bank["w"][pick]], 0)
        W = float(w_b.sum())
        if W <= 0.0 or not train:
            return {}
        pre = {}
        zb = syn.encode(x_b, ach_b, pre=pre)
        xhb = syn.from_bounded(*syn.delta(zb, ach_b, pre=pre))
        rec = (w_b * (xhb - x_b).pow(2).mean(-1)).sum() / W
        zl2 = float(MS.Z_L2) * (w_b * zb.pow(2).mean(-1)).sum() / W
        loss = rec + zl2
        with torch.no_grad():
            for name, t in pre.items():
                now = (t.detach().float() > MS.LIVE_LINE).reshape(-1, t.shape[-1]).any(dim=0)
                prevl = self._syn_live.get(name)
                self._syn_live[name] = now if prevl is None or prevl.shape != now.shape else (prevl | now)
            lv = torch.stack([loss.detach(), rec.detach(), zb.detach().pow(2).mean()]).tolist()
            self._syn_add("lesson_loss", lv[0])
            self._syn_add("lesson_rec", lv[1])
            self._syn_add("lesson_z2", lv[2])
            self._syn_add("lesson_batch", x_b.shape[0])
        return {"synergy": loss}

    def _synergy_babble(self, mfa, handle_state, ns, cur_cm, reach):
        """Babble drawn in the synergy's intention space (--synergy-babble on). z = sigma x u with
        u drawn from the babble's own generator (noise_rng); S'(z, current pose) is a coordinated
        change of every body handle, applied to the order the way today's babble wobble is (the
        turn on each target orientation, the move on each position target), the reach re-clamped
        after. SIGMA IS CALIBRATED ON EVERY BABBLE MOMENT so the mean turn per body handle equals
        today's independent wobble's: 10 x ns radians per axis, mean length 10 x ns x 2 sqrt(2/pi)
        (0.798 rad at ns 0.05) - the babble definition's own number, so no new constant. Up to
        six fixed-point corrections sigma <- sigma x target / measured on the same u (each one
        pass of the synergy over B rows). The ends 1e-6 and 1e6 are numerical rails only, and a
        moment that sits on one is counted (babble_at_rail). How large sigma gets: a FRESH synergy
        (last layer at LAST_LAYER_INIT) asks about 1e-4 rad per handle at unit z, so sigma starts
        near 5,000 (measured 2026-09-22 on the module as built: 0.175 rad at sigma 1,000, 1.44 at
        10,000); it falls as the lesson grows the synergy's output toward the achieved changes.
        The strength wobble (+|N(0, ns)|) is kept as today."""
        from lem.model import motor_synergy as MS
        syn = self.synergy
        B = mfa.shape[0]
        u = torch.randn(B, syn.k, generator=self.noise_rng).to(mfa.device)
        target = 10.0 * float(ns) * MS.CHI3_MEAN
        sigma = float(self._synergy_sigma) if self._synergy_sigma else 1.0
        lo, hi = 1e-6, 1e6
        rail = False
        for _ in range(6):
            rot, pos = syn.delta(sigma * u, handle_state)
            m = float(rot.norm(dim=-1).mean())
            if not math.isfinite(m) or not bool(torch.isfinite(pos).all()):
                self._syn_add("babble_nonfinite", 1)
                return None  # caller uses independent exploration for this moment
            if m <= 0.0:
                sigma, rail = hi, True
                break
            r = target / m
            if abs(r - 1.0) < 0.005:
                break
            new = min(hi, max(lo, sigma * r))
            rail = new in (lo, hi)
            sigma = new
        pre = {}
        rot, pos = syn.delta(sigma * u, handle_state, pre=pre)
        self._synergy_observe_liveness('babble', pre)
        if not bool(torch.isfinite(rot).all() & torch.isfinite(pos).all()):
            self._syn_add("babble_nonfinite", 1)
            return None
        self._synergy_sigma = sigma
        m = rot.norm(dim=-1).mean()
        mfa = syn.perturb_order(mfa, rot, pos)
        mfa[..., 0:3] = cur_cm + (mfa[..., 0:3] - cur_cm).clamp(-reach, reach)
        s_noise = torch.randn(B, mfa.shape[1], generator=self.noise_rng).to(mfa.device) * ns
        mfa[..., 8] += s_noise.abs()
        vals = torch.stack([m, pos.norm(dim=-1).mean()]).tolist()
        self._syn_add("babble_sigma", sigma)
        self._syn_add("babble_turn_rad", vals[0])
        self._syn_add("babble_move_cm", vals[1])
        self._syn_add("babble_target_rad", target)
        self._syn_add("babble_at_rail", 1.0 if rail else 0.0)
        return mfa

    @torch.no_grad()
    def _synergy_observe_liveness(self, regime, pre):
        from lem.model.motor_synergy import LIVE_LINE
        seen = self._syn_live_regimes.setdefault(regime, {})
        for name, value in pre.items():
            alive = (value.detach() > LIVE_LINE).reshape(-1, value.shape[-1]).any(0)
            seen[name] = alive if name not in seen else seen[name] | alive

    def _synergy_space(self, achieved, current_motor):
        """The planner's descent space under --motor-synergy on (motor_synergy.SynergyActionSpace)."""
        from lem.model.motor_synergy import SynergyActionSpace
        return SynergyActionSpace(self.synergy, achieved, current_motor,
                                  residual_l2=self.synergy_residual_l2,
                                  observe=lambda pre: self._synergy_observe_liveness('planner', pre))

    def synergy_window_summary(self) -> Optional[dict]:
        """The `synergy` block of the stats record, and a fresh window. None when off.

        explained_share: of the achieved posture change of the orders graded this window, the share
          the synergy reproduced through its k numbers BEFORE learning from them (1 - weighted
          squared error / weighted squared change; 0 = no better than asking for no change, 1 = the
          whole change), weighted by following x movement as the lesson is; _unweighted beside it.
        effective_dim: the participation ratio of the window's encoded z (1 = every order used one
          direction of z, k = all k equally).
        z_rms: the encoded z's root mean square (the L2 keeps it near 1; babble's sigma is read
          against it).
        following_mean: the graded orders' direction match (-1 against, 0 unrelated, +1 exact) and
          movement_mean their change in measured spreads (1 = one typical 250 ms change).
        babble_*: babble moments in z - sigma, the mean turn per body handle it produced (radians)
          against today's wobble's (target), the mean move (cm) and the share on a rail.
        plan_z_norm / residual_norm: the planner's z and residual length at the end of its descent.
        cmd_mag_rad: the turn every order the model formed asked of the body handles at the moment
          it was formed (the environment's cmd_mag at issue), all moments; _by_author per author.
        liveness: share of each hidden layer's units whose input rose above -3 on the window's
          lesson samples (DESIGN section 10 rule 2)."""
        if self.synergy is None:
            return None
        from lem.model import motor_synergy as MS
        W = self._syn_window

        def m(key, nd=4):
            s = W.get(key)
            return None if not s or s[1] <= 0 else round(s[0] / s[1], nd)

        def tot(key):
            s = W.get(key)
            return 0.0 if not s else float(s[0])

        def share(num, den, nd=4):
            a, b = tot(num), tot(den)
            return None if b <= 0 else round(1.0 - a / b, nd)
        zs = [z for z in self._syn_z_window if z.shape[0]]
        zc = torch.cat(zs, 0) if zs else torch.zeros(0, self.synergy.k)
        pr = MS.participation_ratio(zc) if zc.shape[0] >= 2 else None
        out = {"dim": int(self.synergy.k), "version": MS.SYNERGY_VERSION,
               "orders": int(tot("orders")),
               "lesson_loss": m("lesson_loss", 5), "lesson_rec": m("lesson_rec", 5),
               "lesson_z2": m("lesson_z2", 4), "lesson_batch": m("lesson_batch", 1),
               "explained_share": share("w_err", "w_tot"),
               "explained_share_unweighted": share("err", "tot"),
               "effective_dim": None if pr is None else round(pr, 3),
               "z_rms": None if zc.shape[0] == 0 else round(float(zc.pow(2).mean().sqrt()), 4),
               "following_mean": m("following"), "movement_mean": m("movement"),
               "weighted_share": m("weighted"),
               "babble_sigma": m("babble_sigma"), "babble_turn_rad": m("babble_turn_rad"),
               "babble_target_rad": m("babble_target_rad"), "babble_move_cm": m("babble_move_cm", 3),
               "babble_at_rail": m("babble_at_rail"),
               "babble_nonfinite": int(tot("babble_nonfinite")),
               "babble_steps": int((W.get("babble_sigma") or [0, 0])[1]),
               "plans": int((W.get("plan_z_norm") or [0, 0])[1]),
               "plan_z_norm": m("plan_z_norm"), "residual_norm": m("residual_norm"),
               "cmd_mag_rad": m("cmd_mag_all"),
               "cmd_mag_rad_by_author": {k[8:]: m(k) for k in sorted(W) if k.startswith("cmd_mag_")
                                         and k != "cmd_mag_all"} or None,
               "bank": 0 if self._syn_bank is None else int(self._syn_bank["n"]),
               "liveness": ({n: round(float(v.float().mean()), 4) for n, v in self._syn_live.items()}
                            or None)}
        def summarize_counts(rows):
            data = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
            n = data["closer"].numel()
            if not n:
                return {"part_orders": 0, "closer": 0, "chance_count": 0.0,
                        "closer_per_100": None, "above_chance_per_100": None}
            closer, chance = int(data["closer"].sum()), float(data["chance"].sum())
            asked = float(data["ordered_deg"].sum())
            return {"part_orders": n, "closer": closer, "chance_count": chance,
                    "closer_per_100": 100.0 * closer / n,
                    "chance_per_100": 100.0 * chance / n,
                    "above_chance_per_100": 100.0 * (closer - chance) / n,
                    "ordered_deg_median": float(data["ordered_deg"].quantile(0.5)),
                    "moved_deg_median": float(data["moved_deg"].quantile(0.5)),
                    "distance_closed_pct": 100 * (1 - float(data["remaining_deg"].sum()) / asked)}
        out["following_by_author"] = {author: summarize_counts(rows)
                                      for author, rows in self._syn_following.items() if rows}
        out["command_size_scope"] = "formed proposals; following_by_author grades full executed tenures"
        out["liveness_by_regime"] = {
            regime: {name: round(float(value.float().mean()), 4) for name, value in seen.items()}
            for regime, seen in dict(self._syn_live_regimes, lesson=self._syn_live).items() if seen}
        self._syn_window = {}
        self._syn_z_window = []
        self._syn_live = {}
        self._syn_live_regimes = {}
        self._syn_following = {}
        return out

    def _migrate_synergy_scales(self, path: str, load_optimizer: bool = True) -> None:
        """Carry the synergy across a change of its change scales, exactly; name a reach change.

        The network works in change-scale units and the receipt stamps the scales it was trained
        under. If the code's scales differ, the same change in radians and centimetres is u' = u x
        old / new units: the last layer's rows (weight and bias) are multiplied by old / new, and
        the encoder's first layer, which READS changes in those units, has its change columns
        multiplied by new / old; Adam's averages move by the same factor and its square (the
        precedent is _migrate_efference_scales). A changed reach cannot be carried exactly (the
        bound's shape moves) and is logged by handle."""
        if getattr(self, "synergy", None) is None:
            return
        rp = os.path.join(path, "model_receipt.json")
        if not os.path.isfile(rp):
            return
        with open(rp, encoding="utf-8") as fh:
            saved = json.load(fh).get("motor_synergy") or {}
        if not saved:
            return
        from lem.model import motor_synergy as MS
        if int(saved.get("k", self.synergy.k)) != int(self.synergy.k):
            raise ValueError("checkpoint synergy has k = %s but this build has %d; its z would mean "
                             "something else. Resume with --synergy-dim %s"
                             % (saved.get("k"), self.synergy.k, saved.get("k")))
        k_rot = float(saved.get("change_scale_deg", MS.CHANGE_SCALE_DEG)) / float(MS.CHANGE_SCALE_DEG)
        k_pos = float(saved.get("change_scale_cm", MS.CHANGE_SCALE_CM)) / float(MS.CHANGE_SCALE_CM)
        moved = [n for n, v in (saved.get("rot_reach_deg") or {}).items()
                 if n in MS.ROT_REACH_DEG and abs(float(v) - float(MS.ROT_REACH_DEG[n])) > 1e-9]
        moved += [n + " (move)" for n, v in (saved.get("pos_reach_cm") or {}).items()
                  if n in MS.POS_REACH_CM and abs(float(v) - float(MS.POS_REACH_CM[n])) > 1e-9]
        if moved:
            log.warning("motor synergy: the checkpoint's synergy was bounded by other reaches for %s; "
                        "its changes near those reaches now flatten at the new ones", ", ".join(moved))
        if abs(k_rot - 1.0) < 1e-9 and abs(k_pos - 1.0) < 1e-9:
            return
        syn = self.synergy
        n_r = 3 * syn.n_rot
        rows = torch.ones(syn.n_out)
        rows[:n_r] = k_rot
        rows[n_r:] = k_pos
        cols = torch.ones(syn.E1.weight.shape[1])
        cols[:n_r] = 1.0 / k_rot
        cols[n_r:syn.n_out] = 1.0 / k_pos
        with torch.no_grad():
            for p, kk in ((syn.S3.weight, rows.view(-1, 1)), (syn.S3.bias, rows),
                          (syn.E1.weight, cols.view(1, -1))):
                kk = kk.to(p.device, p.dtype)
                p.mul_(kk)
                # a weight multiplied by kk has its gradient divided by kk (the output it feeds is
                # read kk times larger, or its input arrives kk times smaller), so its running
                # averages follow the gradient: exp_avg / kk, exp_avg_sq / kk^2
                st = self.optimizer.state.get(p) if load_optimizer else None
                if st:
                    if torch.is_tensor(st.get("exp_avg")):
                        st["exp_avg"].div_(kk)
                    if torch.is_tensor(st.get("exp_avg_sq")):
                        st["exp_avg_sq"].div_(kk * kk)
        log.warning("motor synergy: the checkpoint's synergy worked in change scales %.4g deg / %.4g "
                    "cm; the code now uses %.4g / %.4g, so its last layer and its encoder's first "
                    "layer were rescaled and their optimiser averages moved to match. Every change "
                    "in radians and centimetres reads as it did.",
                    float(saved.get("change_scale_deg")), float(saved.get("change_scale_cm")),
                    MS.CHANGE_SCALE_DEG, MS.CHANGE_SCALE_CM)

    @torch.no_grad()
    def _schema_coherence(self, B: int) -> torch.Tensor:
        """The number the coherence term of wellness carries, one per body.

        "extent" is the share of everything that body's own map watches that it calls its
        own, 0 = none and 1 = all - its 26 handles AND any outside thing taken into the self
        (UEMA 11.2), so 28 things on the crib runs. That is the quantity the objective asks
        wellness to reward, since a thing taken into the self raises it. "certainty" is how
        sure the map is either way, which on run H84 equalled 1 - 2 x the BODY's own share and
        so rewarded the perceived self shrinking."""
        pick = (lambda sc: sc.self_extent()) if self.wellness_coherence == "extent" \
            else (lambda sc: sc.coherence_score())
        # ONE NUMBER PER BODY, ALWAYS. The maps are grown to the batch later in the step, so
        # a batch wider than the maps in hand - a fresh start before the first update, or a
        # resume of a checkpoint saved with fewer bodies under a larger --n-envs - returned
        # fewer numbers than bodies and the wellness term raised on the shape.
        while len(self._schemas) < B:
            self._schemas.append(DifferentiableBodySchema(
                n_handles=len(self.handle_names), object_names=SELF_OBJECTS).to(self.device))
        return torch.stack([pick(sc) for sc in self._schemas[:B]]).reshape(-1)

    @torch.no_grad()
    def _observe_entity_self(self, frames, wellness_out) -> None:
        """One moment for the named-entity levels, and one moment of evidence for the reader's
        gate. The presence of each named thing is read from the world's own per-body report
        (entity_presence), never from the model, so it survives the encoders changing."""
        from mimo_bridge.entity_presence import ENTITY_NAMES, entity_presence_batch
        pres = entity_presence_batch([f.meta.get("task") for f in frames],
                                     [f.meta.get("caregiver") for f in frames]).to(self.device)
        self._entity_presence = pres
        B = pres.shape[0]
        if self._entity_self is not None and self._entity_self.n_bodies != B:
            log.warning("entity self: the body count changed from %d to %d, so its running "
                        "averages restart", self._entity_self.n_bodies, B)
            self._entity_self = None
            # AND THE READER GIVES BACK WHAT IT WAS READING. Its links, its pass list and its
            # gate were all earned from the averages just thrown away, so leaving them in
            # place would go on adding value to the planner's goals from evidence that no
            # longer exists anywhere. The resume path already does this; this path did not.
            if self._self_value is not None:
                self._self_value.forget("the body count changed, so the tracker's running "
                                        "averages restarted and the links it had proven were "
                                        "thrown away with them")
        if self._entity_self is None:
            from mimo_bridge.entity_self import EntitySelfDark
            self._entity_self = EntitySelfDark(
                ENTITY_NAMES, n_bodies=B, teacher=self._entity_self_cfg["teacher"]).to(self.device)
            log.warning("entity self: watching %d named things across %d bodies, learning from "
                        "%s. A link needs %s windows of %d moments at the minutes level before "
                        "it can count.", len(ENTITY_NAMES), B, self._entity_self.teacher,
                        self._entity_self.first_evidence_tick(), self._entity_self.window("l2"))
            saved = getattr(self, "_entity_self_saved", None)
            if saved:
                names = getattr(self, "_entity_self_saved_names", None)
                if names is not None and list(names) != list(ENTITY_NAMES):
                    log.warning("entity self: the checkpoint watched a different list of things "
                                "(%s), so nothing is carried over", names)
                else:
                    self._entity_self.load_state_dict(
                        {k: v.to(self.device) for k, v in saved.items()}, strict=False)
                    log.warning("entity self: carried %d moments of watching across the resume",
                                int(self._entity_self.frames.item()))
                self._entity_self_saved = None
        wellness = torch.cat(list(wellness_out["grounded"].values()), dim=-1).sum(-1).detach()
        if self._entity_self.teacher == "ability":
            # WHAT THE PROJECT'S GOAL ASKS FOR rather than what the body feels: how many kinds
            # of consequence the world measured this moment - a hand contact, an object moved,
            # a view gained (consequence_kinds). 0 to 3.
            # (--consequence-self-contact exclude applies here too: one definition of a
            # consequence per run)
            teach = torch.tensor([consequence_kinds(f.meta.get("task") or {},
                                                    world_only=getattr(self, "consequence_world_only",
                                                                       False))
                                  for f in frames],
                                 device=self.device, dtype=torch.float32)
        else:
            teach = wellness
        self._entity_self.observe(pres, teach)
        if self._self_value is not None:
            self._self_value.compare(pres, wellness)
            w = self._entity_self.window(self._self_value.level)
            if int(self._entity_self.frames.item()) % w == 0:
                self._self_value.window_tick(self._entity_self)

    @torch.no_grad()
    def incorporation_stats(self) -> Optional[dict]:
        """How much of the body, and of the outside things, the body map calls its own.

        The share of the body is what the coherence term of wellness carries. The per-object
        numbers are UEMA 11.2's object incorporation, measured. A thing with no evidence at
        all reads 0.0759 and a thing carried as surely as a limb reads 0.9241, which are the
        two ends of this scale; above 0.5 means the thing kept more than half the hand's
        distance in the hand's own direction. A toy rises while it is held and falls when it
        is let go.

        Each thing also carries what it scored, what the same score reads with the hand moving
        the opposite way, and the difference, so a reading can be told from its own noise."""
        if not (self.body_schema_updates and self._schemas):
            return None
        sc = self._schemas
        out = {"handles_mine": round(float(torch.stack([s.self_membership().mean() for s in sc]).mean()), 5),
               "extent": round(float(torch.stack([s.self_extent() for s in sc]).mean()), 5),
               "certainty": round(float(torch.stack([s.coherence_score() for s in sc]).mean()), 5),
               "carried_by_wellness": self.wellness_coherence}
        if sc[0].n_objects:
            m = torch.stack([s.object_membership() for s in sc]).mean(0)
            out["objects"] = {n: round(float(v), 5) for n, v in zip(sc[0].object_names, m.tolist())}
            out["objects_per_body"] = {
                n: [round(float(s.object_membership()[i]), 5) for s in sc]
                for i, n in enumerate(sc[0].object_names)}
            ev = [s.object_evidence() for s in sc]
            out["objects_evidence"] = {
                n: {k: round(float(sum(e[n][k] for e in ev)) / len(ev), 5)
                    for k in ("match", "control", "earned")}
                for n in sc[0].object_names}
            # what the world lets the body do to each thing at all: a mocap body is placed by
            # the adult every step and physics never moves it, so the body cannot carry it
            out["objects_physics_driven"] = {
                n: bool(self._object_physics_driven.get(n, False)) for n in sc[0].object_names}
        return out

    @torch.no_grad()
    def entity_self_stats(self) -> Optional[dict]:
        """What the named-entity levels have found, and what the reader is doing with it."""
        if self._entity_self is None:
            return None
        out = dict(self._entity_self.stats() or {})
        if self._self_value is not None:
            out["reader"] = self._self_value.last_report or self._self_value.report()
        return out

    # ------------------------------------------------------------------ wellness channels (B3)
    def _init_wellness_channels(self, controllability, learning_progress, wellness_rank,
                                wellness_weights, curiosity_mode, consequence_self_contact,
                                contact_learning) -> None:
        """Controllability, learning progress, the normalised ranking and the world-only
        consequence (DESIGN D6-D9, lem/model/wellness_channels.py). Called last in __init__, so a
        later agent's channel (contact_control: set self.contact_control_enabled earlier in
        __init__) is seen when the channel list is fixed."""
        from lem.model import wellness_channels as WC
        if wellness_rank not in ("raw", "normalized"):
            raise ValueError(f"wellness_rank must be 'raw' or 'normalized', got {wellness_rank!r}")
        if curiosity_mode not in ("multiply", "additive"):
            raise ValueError(f"curiosity_mode must be 'multiply' or 'additive', got {curiosity_mode!r}")
        if consequence_self_contact not in ("count", "exclude"):
            raise ValueError("consequence_self_contact must be 'count' or 'exclude', got "
                             f"{consequence_self_contact!r}")
        self.wellness_rank_mode = str(wellness_rank)
        self.curiosity_mode = str(curiosity_mode)
        self.consequence_world_only = consequence_self_contact == "exclude"
        self.rank_weights = WC.parse_weights(wellness_weights)
        self.controllability_on = bool(controllability)
        self.learning_progress_on = bool(learning_progress)
        # PUBLIC name on purpose: its state rides runtime_state.pt, never the episode companion
        self.wellness_channels = None
        self._eff_lp: dict = {}
        self._rank_ring_cleared = 0
        normalized = self.wellness_rank_mode == "normalized"
        if normalized:
            if not self.intention_enabled:
                raise ValueError("--wellness-rank normalized ranks the goal ring, which exists only "
                                 "with --intention on")
            if self.goal_revalue or contact_learning:
                raise ValueError("--wellness-rank normalized ranks the stored channels; --goal-revalue "
                                 "and --contact-learning re-score the ring their own way, and the two "
                                 "rankings have no defined combination")
            if self.curiosity_goal and self.curiosity_mode != "additive":
                raise ValueError("--wellness-rank normalized needs --curiosity-mode additive: the "
                                 "normalised value is centred on 0, and multiplying it by (1 + drive) "
                                 "would make a below-average moment WORSE the more curious the "
                                 "moment was")
            if WC.INTEGRITY_GATE_LINE is None:
                raise ValueError("lem/model/wellness_channels.py INTEGRITY_GATE_LINE is unmeasured")
            if WC.RING_DETREND_STEPS is None:
                raise ValueError("lem/model/wellness_channels.py RING_DETREND_STEPS is unmeasured: "
                                 "without it the ranking follows each channel's drift and picks "
                                 "the goal by its age in the ring")
            self.rank_channel_names = self._rank_channels_active()
            unmeasured = [c for c in self.rank_channel_names
                          if c not in WC.UNRANKED_ROWS and self.rank_weights.get(c, 0.0) != 0.0
                          and WC.FLOORS.get(c) is None]
            if unmeasured:
                raise ValueError("--wellness-rank normalized: no measured floor for %s "
                                 "(lem/model/wellness_channels.py FLOORS)" % ", ".join(unmeasured))
        control = lp = None
        if self.controllability_on:
            need = ("command",) + (("efference",) if self.efference is not None else ())
            if any(WC.CONTROL_FLOOR.get(k) is None for k in need):
                raise ValueError("--controllability: lem/model/wellness_channels.py CONTROL_FLOOR "
                                 "is unmeasured for %s" % ", ".join(need))
            control = WC.ControllabilityTracker(
                len(self.handle_names), self.GAZE_IDX,
                hold=self.efference_tenure_steps, floors=dict(WC.CONTROL_FLOOR),
                expect_efference=self.efference is not None)
        if self.learning_progress_on:
            if not self.next_frame_loss:
                raise ValueError("--learning-progress reads the delayed next-frame lesson's errors, "
                                 "which run only with --next-frame-loss on")
            lp = WC.LearningProgressTracker()
        if normalized or control is not None or lp is not None:
            self.wellness_channels = WC.WellnessChannelState(normalized, control, lp)
        if (self.wellness_channels is not None or self.consequence_world_only
                or self.curiosity_mode != "multiply"):
            # what this build's goal value was made of, stamped beside the weights it trained
            # (absent under every default, so a default build's receipt is unchanged)
            self.receipt['wellness_channels'] = {
                "version": WC.WELLNESS_CHANNELS_VERSION, "rank": self.wellness_rank_mode,
                "channels": list(self.rank_channel_names or []),
                "weights": dict(self.rank_weights),
                "floors": dict(WC.FLOORS), "gate_line": WC.INTEGRITY_GATE_LINE,
                "gate_reads": WC.HARM_ROW, "detrend_steps": WC.RING_DETREND_STEPS,
                "control_gate_noise": {"open": WC.CONTROL_GATE_OPEN_NOISE,
                                       "close": WC.CONTROL_GATE_CLOSE_NOISE},
                "control_kind": "shuffled order", "lp_holdout_steps": WC.LP_HOLDOUT_STEPS,
                "control_floor": dict(WC.CONTROL_FLOOR), "controllability": self.controllability_on,
                "learning_progress": self.learning_progress_on,
                "curiosity_mode": self.curiosity_mode,
                "consequence_self_contact": consequence_self_contact}
            log.warning("wellness channels: rank %s%s; controllability %s; learning progress %s; "
                        "curiosity %s; consequence hand press %s",
                        self.wellness_rank_mode,
                        (" over %s" % ", ".join(self.rank_channel_names or [])) if normalized else "",
                        "on" if control is not None else "off", "on" if lp is not None else "off",
                        self.curiosity_mode,
                        "world only" if self.consequence_world_only else "counts self-contact")

    def _with_contact_control_weight(self, wellness_weights):
        """--wellness-weights with contact_control set to --contact-control-wellness. With the
        channel on, naming contact_control in --wellness-weights with another value is refused:
        one weight, two sources. With the channel off, naming it at all is refused (review of b4,
        minor 7: such a weight was silently ignored while the receipt stamped it), and the weights
        carry contact_control 0, so the receipt stamps the weight in force; no ranking reads a
        channel the run does not have, so every other value is what it was."""
        from lem.model import wellness_channels as WC
        named = (("contact_control" in wellness_weights) if isinstance(wellness_weights, dict)
                 else any(p.split("=", 1)[0].strip() == "contact_control"
                          for p in str(wellness_weights or "").split(",") if "=" in p))
        if not getattr(self, "contact_control_enabled", False):
            if named:
                raise ValueError("--wellness-weights names contact_control, but that channel is off: "
                                 "--contact-control-wellness switches it on and gives its weight "
                                 "(with --contact-head on)")
            w = WC.parse_weights(wellness_weights)
            w["contact_control"] = 0.0
            return w
        w = WC.parse_weights(wellness_weights)
        if named and w["contact_control"] != self.contact_control_weight:
            raise ValueError("contact_control's weight is given twice: --contact-control-wellness %g "
                             "and --wellness-weights contact_control=%g" % (
                                 self.contact_control_weight, w["contact_control"]))
        w["contact_control"] = float(self.contact_control_weight)
        return w

    def _rank_channels_active(self) -> List[str]:
        """The channels this run's goal value is made of, in report order, then the unranked rows
        (pain, and harm = what the gate reads)."""
        from lem.model.wellness_channels import CHANNEL_ORDER, UNRANKED_ROWS
        have = {"integrity", "coherence", "prediction_error", "tactile_comfort", "capability",
                "efficiency"}
        if self.controllability_on:
            have.add("controllability")
        if self.learning_progress_on:
            have.add("learning_progress")
        if getattr(self, "contact_control_enabled", False):
            have.add("contact_control")
        if self.consequence_weight > 0.0:
            have.add("consequence")
        if self._self_value is not None:
            have.add("self_value")
        if self.agency_drive:
            have.add("agency")
        if self.movement is not None:
            have.add("movement")
        if self.curiosity_goal:
            have.add("curiosity")
        return [c for c in CHANNEL_ORDER if c in have] + list(UNRANKED_ROWS)

    # a companion saved by the first b3 build carries the channel list under this name; the list
    # a build ranks is ALWAYS the one its own flags set (rank_channel_names), so a restored value is
    # ignored, and readers written for that build still find the list here
    @property
    def _rank_channel_names(self) -> Optional[List[str]]:
        return self.rank_channel_names

    @_rank_channel_names.setter
    def _rank_channel_names(self, _value) -> None:
        pass

    @torch.no_grad()
    def _channel_pain(self, extras: dict, frame: dict, B: int) -> torch.Tensor:
        """[B] the pain wellness scores inside tactile_comfort, 0 (no modelled aversion anywhere on
        the skin) to 1 (an extreme load): lem/model/contact_wellness.contact_response on exactly the
        skin features the wellness call scored. A moment with no touch sense measured has no pain
        reading: NaN (unmeasured - the ranking counts it as the channel's centre and the gate
        cannot judge it), never 0, which would read as the best integrity a body can have."""
        from lem.model.contact_wellness import contact_response
        rt = extras.get("raw_tactile") if isinstance(extras, dict) else None
        if rt is None and "tactile" in frame and frame["tactile"].shape[-1] == TOUCH_TOKEN_F:
            rt = frame["tactile"][..., 3:11]
        if rt is None:
            return torch.full((B,), float("nan"), device=self.device)
        if isinstance(rt, dict):
            cr = contact_response(rt["features"], rt.get("sensitivity"), rt.get("valid"),
                                  profile=rt.get("profile", "legacy"), history=rt.get("history"))
        else:
            cr = contact_response(rt)
        return cr["pain"].reshape(-1).detach().float()

    @torch.no_grad()
    def _observe_wellness_channels(self, step, exec_t, order_id, handle_state, nf_live,
                                   tokens_raw, source_mask, B) -> None:
        """One moment for learning progress and controllability (called in the loss section, after
        the efference lesson, before this step's order is authored). Every error it needs is read
        from the model's device in ONE copy."""
        wc = self.wellness_channels
        if wc is None:
            return
        _t0 = time.perf_counter()
        wc.last_B = int(B)
        if wc.lp is not None:
            wc.lp.tick_holdout()
            keys, rows = [], []
            if nf_live is not None and self._nf_mask is not None:
                nf_in, nf_preds = nf_live
                for m in self.pred_mods:
                    if (m in nf_preds and m in tokens_raw and m in nf_in
                            and source_mask.get(m) == "real" and self._nf_mask.get(m) == "real"):
                        act = tokens_raw[m].reshape(B, -1).float()
                        keys.append(m)
                        rows.append((nf_preds[m].detach().reshape(B, -1).float() - act).pow(2).mean(-1))
                        rows.append((nf_in[m].detach().reshape(B, -1).float() - act).pow(2).mean(-1))
            for k, (err, dn) in (self._eff_lp or {}).items():
                if err.shape[0] == B:
                    keys.append(k)
                    rows.append(err.detach().float().reshape(-1).to(self.device))
                    rows.append(dn.detach().float().reshape(-1).to(self.device))
            self._eff_lp = {}
            if rows:
                errs = torch.stack(rows).cpu()                         # [2K, B], one read
                for j, k in enumerate(keys):
                    wc.lp.update(k, errs[2 * j], errs[2 * j + 1])
            wc.lp.tick_window(B)
        if wc.control is not None:
            wc.control.step(int(step), exec_t, order_id, handle_state,
                            getattr(self, "last_action_source", None),
                            self.last_efference_tenure)
        wc.add_cost(1000.0 * (time.perf_counter() - _t0), new_step=True)

    @torch.no_grad()
    def _wellness_channel_vector(self, B, wellness_out, pain, mv, ag, cq, drive, frames):
        """[C, B] this moment's channel values in rank_channel_names order (NaN = unmeasured),
        before the ring's detrend."""
        from lem.model.wellness_channels import PAIN_ROW, HARM_ROW
        names = self.rank_channel_names
        g = wellness_out["grounded"]
        dev = self.device
        nan = torch.full((B,), float("nan"), device=dev)

        def col(key):
            v = g.get(key)
            return nan if v is None else v.detach().float().reshape(-1).to(dev)
        vals = {"coherence": col("coherence"), "prediction_error": col("prediction_error"),
                "tactile_comfort": col("tactile_comfort"), "capability": col("capability")}
        ms = col("motor_state")
        p = pain.to(dev) if pain is not None else nan
        # the worse of the two harms the body can report: skin load (pain) and a limb pressed
        # into the end of its reach (1 - motor_state, the boundary penalty's mean). An unmeasured
        # pain leaves both unmeasured (torch.maximum carries the NaN): the limit part alone is a
        # mean over 69 coordinates and would read as a nearly harmless moment
        harm = torch.maximum(p, 1.0 - torch.nan_to_num(ms, nan=1.0))
        vals["integrity"] = 1.0 - harm
        wc = self.wellness_channels
        if "controllability" in names:
            vals["controllability"] = wc.control.value(B, dev)
        if "learning_progress" in names:
            vals["learning_progress"] = wc.lp.channel(B).to(dev)
        if "contact_control" in names:
            cc = getattr(self, "_contact_control_value", None)
            vals["contact_control"] = (cc.detach().float().reshape(-1).to(dev)
                                       if torch.is_tensor(cc) and cc.numel() == B else nan)
        if "consequence" in names:
            vals["consequence"] = cq.detach().float().reshape(-1).to(dev)
        if "self_value" in names:
            sv = self._self_value
            if (self._entity_presence is not None and sv is not None
                    and bool(sv.gate_open.item())):
                vals["self_value"] = sv.raw_value(self._entity_presence).float().reshape(-1).to(dev)
            else:
                vals["self_value"] = torch.zeros(B, device=dev)
        if "agency" in names:
            vals["agency"] = (ag.detach().float().reshape(-1).to(dev) if torch.is_tensor(ag)
                              else nan)
        if "movement" in names:
            vals["movement"] = (mv["wellness"].detach().float().reshape(-1).to(dev)
                                if mv is not None else nan)
        if "curiosity" in names:
            vals["curiosity"] = (torch.full((B,), float(drive), device=dev)
                                 if self.curiosity_measured else nan)
        # EFFICIENCY, logged at weight 0 (energy is the addendum's item 11): minus the body's
        # effort this moment, mechanical power over 66 W plus summed actuator torque over 63 N m
        # (the vestibular vector's fields 23 and 24, env-senses' measured 99th percentiles), so 0 =
        # no effort and -2 = both at that 99th percentile. NaN on a padded legacy read.
        eff = []
        for f in frames[:B]:
            v = getattr(f, "vestibular", None)
            if v is None or len(v) < 29 or float(v[28]) < 0.5:
                eff.append(float("nan"))
            else:
                eff.append(-(float(v[23]) + float(v[24])))
        vals["efficiency"] = torch.tensor(eff, device=dev, dtype=torch.float32)
        vals[PAIN_ROW] = p
        vals[HARM_ROW] = harm
        return torch.stack([vals.get(n, nan) for n in names]).float()

    @torch.no_grad()
    def _rank_store_moment(self, B, wellness_out, extras, frame, mv, cq, drive, frames):
        """This moment's channels as the goal ring stores them ([C, B] on the processor): the raw
        vector is read from the device ONCE; its window means are reported raw; each ranked
        channel is stored as its distance from the body's running average (wellness_channels.
        detrend_moment, RING_DETREND_STEPS); the normaliser learns the stored values' centre and
        spread. A ring restored with no channel record, or recorded under another channel list,
        restarts here (planning waits 20 moments, 1 s, as after a reset)."""
        from lem.model import wellness_channels as WC
        _t0 = time.perf_counter()
        wc = self.wellness_channels
        names = self.rank_channel_names
        raw = self._wellness_channel_vector(
            B, wellness_out, self._channel_pain(extras, frame, B), mv,
            self._last_agency_vec if self.agency_drive else None, cq, drive, frames)
        raw = raw.detach().float().cpu()                                  # the one read
        wc.last_raw = raw
        for _ci, _cn in enumerate(names):
            wc.add_raw(_cn, raw[_ci])
        if (len(self._goal_ring_ch) != len(self.goal_ring)
                or self._goal_ring_ch_names != list(names)
                or (len(self._goal_ring_ch) and self._goal_ring_ch[-1].shape != raw.shape)):
            if len(self.goal_ring) or len(self._goal_ring_ch):
                log.warning("wellness rank: the goal ring's %d moments carry no channel record "
                            "for this run's %d channels (%s); the ring restarts",
                            len(self.goal_ring), int(raw.shape[0]), ", ".join(names))
                self._rank_ring_cleared += 1
            self.goal_ring.clear()
            self._goal_ring_pose.clear()
            self._goal_ring_w0.clear()
            self._goal_ring_cq.clear()
            self._goal_ring_drive.clear()
            self._goal_ring_ch.clear()
            self._rank_detrend = {}
            self._gl_clear()
            self._goal_ring_ch_names = list(names)
        stored = WC.detrend_moment(self._rank_detrend, names, raw, WC.RING_DETREND_STEPS)
        for _ci, _cn in enumerate(names):
            if _cn not in WC.UNRANKED_ROWS:
                wc.normalizer.update(_cn, stored[_ci])
        wc.add_cost(1000.0 * (time.perf_counter() - _t0))
        return stored

    def _rank_weights_now(self) -> Dict[str, float]:
        """The weights the ranking uses at this planning event: the table, with controllability
        at 0 while its evidence gate is shut (lem/model/wellness_channels.CONTROL_GATE_READINGS)."""
        w = dict(self.rank_weights)
        wc = self.wellness_channels
        if wc is not None and wc.control is not None and not wc.control.gate()[0]:
            w["controllability"] = 0.0
        return w

    @torch.no_grad()
    def _normalized_goal_values(self, wl_stack: torch.Tensor):
        """[T, B] the normalised, prioritised value of every remembered moment, -inf where its
        harm is over the integrity gate line (unless every moment is: then the least harmful
        keeps its value); plus the parts for the causal-chain record. Ranked on the processor
        (the ring is kept there) and handed back on the model's device."""
        from lem.model import wellness_channels as WC
        _t0 = time.perf_counter()
        ch = torch.stack([t.detach().float().cpu() for t in self._goal_ring_ch])   # [T, C, B]
        if ch.shape[0] != wl_stack.shape[0] or ch.shape[2] != wl_stack.shape[1]:
            raise RuntimeError("the goal ring's channel record lost alignment with its moments")
        names = self.rank_channel_names
        ranked, contrib, gated, all_gated = WC.rank_values(
            ch, names, self._rank_weights_now(), self.wellness_channels.normalizer, WC.FLOORS,
            WC.INTEGRITY_GATE_LINE)
        self.wellness_channels.shares.add(contrib, ranked, gated, all_gated, names)
        self.wellness_channels.add_cost(1000.0 * (time.perf_counter() - _t0), planning=True)
        dev = wl_stack.device
        return ranked.to(dev), {"contrib": contrib.to(dev), "gated": gated.to(dev),
                                "all_gated": all_gated.to(dev), "names": list(names)}

    def next_frame_norm_window_summary(self) -> Optional[dict]:
        """The `next_frame_norm` block (DESIGN section 3), and a fresh window; None when
        --next-frame-norm is off (lem/model/next_frame_norm.NextFrameNorm.summary says what each
        field means)."""
        if getattr(self, "nf_norm", None) is None:
            return None
        return self.nf_norm.summary()

    def controllability_window_summary(self) -> Optional[dict]:
        """The `controllability` block (DESIGN section 3); None when --controllability is off.
        value_mean / per_body: the running agreement per body over its last few measured orders
        (lem/model/wellness_channels.CONTROL_EMA_ORDERS), -1 = the opposite effect, 0 = unrelated
        or none where one was expected, +1 = exactly the expected effect. measured_share: orders
        whose expected and commanded effects were both above the body's measured drift while
        holding still (the rest are stillness, never counted). agreement_mean: this window's
        measured orders. control_mean: the same orders scored against what the model would have
        expected from THIS body's state with ANOTHER body's order (the shuffled-order control) -
        the level agreement reaches with no link to the order sent; difference_mean: agreement -
        control, -2 to +2, 0 = the order made no difference. by_author: agreement by who authored
        the order. source: efference (the body head's expectation) or command (the order's own
        commanded change, when the head is off).
        Since 2026-10-03, recorded only (no weight, ranking or loss reads them):
        size_matched_control_mean / size_matched_difference_mean: the same two readings with the
        size-matched control (another body's asked moves and turns, measured from its own pose,
        applied to this body's pose) in place of the shuffled-order one; shadow_gate_open, _mean,
        _standard_error, _readings_above_zero, _readings_held: the evidence gate's rule applied to
        that control's readings (the shadow gate), beside gate_standard_error,
        gate_readings_above_zero and gate_readings_held for the gate in force; gate_commands,
        gate_open_commands, shadow_gate_commands, shadow_gate_open_commands, gates_same_commands,
        gates_both_open_commands, gate_only_open_commands, shadow_gate_only_open_commands: per
        window, the graded commands at which each gate took a reading and those at which it stood
        open. gate_definitions in the block defines each (lem/model/wellness_channels.py
        CONTROL_GATE_DEFINITIONS)."""
        wc = self.wellness_channels
        if wc is None or wc.control is None:
            return None
        return wc.control.summary(int(getattr(wc, "last_B", 1) or 1))

    def learning_progress_window_summary(self) -> Optional[dict]:
        """The `learning_progress` block; None when --learning-progress is off. per_sense[s] =
        (r_slow - r_fast) / r_slow averaged over the window and the bodies, where r = the model's
        next-moment error / the error of predicting no change (efference heads: head error / the
        do-nothing error): +1 = the error relative to doing nothing fell to zero lately, 0 = no
        change, -1 = it doubled or worse. value_mean / per_body: the channel, the mean over senses.
        ratio_fast / ratio_slow: the two ratios now (0 = perfect, 1 = no better than doing
        nothing)."""
        wc = self.wellness_channels
        if wc is None or wc.lp is None:
            return None
        return wc.lp.summary(int(getattr(wc, "last_B", 1) or 1))

    def wellness_rank_window_summary(self) -> Optional[dict]:
        """The `wellness_rank` block; None under the raw ranking. centre / scale: each channel's
        running mean and spread over per-moment values; a channel's contribution is weight x
        (value - centre) / max(scale, floor). variance_share / winner_margin_share: how much of the
        spread of the candidates' values, and of the winner's lead over the runner-up, each channel
        carried (each set adds to 1; a channel near 1 decides alone; a margin share below 0 pushed
        against the winner, above 1 pushed for it by more than the whole lead). integrity_gated_share: the
        candidates whose harm (skin pain or the limit part) was over the line (gate_line) and could
        not be chosen. raw: each channel's window mean of its per-moment values (before the ring's
        detrend; centre and scale are of the STORED distances from each body's running average).
        weight_share_by_channel: each channel's share of the weights in force (|w| / sum |w|),
        the size a channel would carry if every channel spread alike - compare with its
        winner_margin_share. goal_age: where the chosen goals sat in the ring (RankShares).
        planner_blind: channels the approximate evaluator cannot score in an IMAGINED moment
        (remembered moments carry them all)."""
        wc = self.wellness_channels
        if wc is None or wc.normalizer is None:
            return None
        from lem.model import wellness_channels as WC
        names = [n for n in (self.rank_channel_names or []) if n not in WC.UNRANKED_ROWS]
        cs = {n: wc.normalizer.centre_scale(n) for n in names}

        def r(x, nd=5):
            return None if x is None or not math.isfinite(x) else round(float(x), nd)
        w_now = self._rank_weights_now()
        out = {"mode": self.wellness_rank_mode, "curiosity_mode": self.curiosity_mode,
               "weights": {n: self.rank_weights.get(n, 0.0) for n in names},
               # the weights in force at the end of the window: controllability reads 0 while
               # its evidence gate is shut (the controllability block says why)
               "weights_in_force": {n: w_now.get(n, 0.0) for n in names},
               "centre": {n: r(cs[n][0]) for n in names},
               "scale": {n: r(cs[n][1]) for n in names},
               "floors": {n: WC.FLOORS.get(n) for n in names},
               "gate_line": WC.INTEGRITY_GATE_LINE,
               "detrend_steps": WC.RING_DETREND_STEPS}
        _wsum = sum(abs(float(w_now.get(n, 0.0))) for n in names)
        out["weight_share_by_channel"] = ({n: round(abs(float(w_now.get(n, 0.0))) / _wsum, 4)
                                           for n in names} if _wsum > 0 else None)
        out.update(wc.shares.summary())
        out["raw"] = wc.raw_summary()
        out["ring_cleared"] = int(self._rank_ring_cleared)
        out["planner_blind"] = [n for n in names if WC.JUDGE_B_VIEW.get(n) == "no"]
        out["planner_partly"] = [n for n in names if str(WC.JUDGE_B_VIEW.get(n, "")).startswith("partly")]
        wc.shares.reset()
        return out

    def _lp_hold_out_fresh(self, res: dict, reinit: Sequence[str] = ()) -> List[str]:
        """Leave out of learning progress, for LP_HOLDOUT_STEPS model steps, every key whose
        network this load started fresh: an encoder the loader reinitialised or was told to
        (encoder:<sense>), the vestibular encoder of an intermind -> vestibular switch, and both
        efference heads when they were not in the checkpoint. Returns the keys."""
        wc = getattr(self, "wellness_channels", None)
        if wc is None or wc.lp is None:
            return []
        from lem.model import wellness_channels as WC
        fresh = set()
        for r in list((res or {}).get("reinitialized_components") or []) + list(reinit or ()):
            r = str(r)
            if r.startswith("encoder:") and r.split(":", 1)[1] in self.pred_mods:
                fresh.add(r.split(":", 1)[1])
            elif r == "efference_heads":
                fresh.update(("efference_body", "efference_gaze"))
        if getattr(self, "vestibular_switch", None) is not None and "vestibular" in self.pred_mods:
            fresh.add("vestibular")
        keys = sorted(fresh)
        if keys:
            if WC.LP_HOLDOUT_STEPS is None:
                raise ValueError("--learning-progress: lem/model/wellness_channels.py LP_HOLDOUT_STEPS "
                                 "is unmeasured, and this load starts %s fresh" % ", ".join(keys))
            wc.lp.set_holdout(keys, WC.LP_HOLDOUT_STEPS)
            log.warning("learning progress: %s started fresh at this load and sit%s out for %d "
                        "model steps, so a new network's first fall from a random start is not "
                        "counted as progress", ", ".join(keys), "s" if len(keys) == 1 else "",
                        WC.LP_HOLDOUT_STEPS)
        return keys

    def wellness_channels_cost_window(self) -> Optional[dict]:
        """Wall time of the wellness channels' own code since the last call (None when no channel
        flag is on): per_step_ms per model step, planning_ms per ranking of the ring. For the
        machine block (machine.wellness_channels_ms), so a proof can compare flags on and off."""
        wc = getattr(self, "wellness_channels", None)
        return None if wc is None else wc.cost_window()

    # ------------------------------------------------------------------ checkpoints
    def _side_modules(self) -> dict:
        """Trained parameters that live on the model rather than in a named component.

        These were never passed to CheckpointManager.save(), so they appeared in 0 of the
        274 tensors a checkpoint holds and were rebuilt from a fresh random init at every
        resume. recon_decoders is a real cost: the reconstruction loss restarts at
        0.87-0.98 and took 16,600 steps on H31 and 64,300 on H28 to return within 10% of
        its floor. td_gates is worse than a cost - the top-down gate feeds the token that
        next_pe is graded against, so re-randomising it silently changes the measurement.
        """
        side = {"recon_decoders": self.recon_decoders, "td_gates": self.td_gates}
        if self.tenure_heads is not None:
            side["tenure_heads"] = self.tenure_heads
        if self.contact_consequences is not None:
            side["contact_consequences"] = self.contact_consequences
        if self.persistent_memory is not None:
            side['persistent_memory'] = self.persistent_memory
        if self.self_state is not None:
            side['self_state'] = self.self_state
        if getattr(self, "efference", None) is not None:
            side["efference_heads"] = self.efference
        if getattr(self, "contact_head", None) is not None:
            side["contact_head"] = self.contact_head
        if getattr(self, "synergy", None) is not None:
            side["motor_synergy"] = self.synergy
        if getattr(self, "liveness", None) is not None and self.liveness.mode == "on":
            side["liveness_repair"] = self.liveness
        if getattr(self, "motor_input_fold", None) is not None:
            side["motor_input_fold"] = self.motor_input_fold
        if getattr(self, "motor_asked_change", None) is not None:
            side["motor_asked_change"] = self.motor_asked_change
        if getattr(self, "heads_all_slot_path", None) is not None:
            side["heads_all_slot_path"] = self.heads_all_slot_path
        if getattr(self, "vestibular_input_path", None) is not None:
            side["vestibular_input_path"] = self.vestibular_input_path
        return side

    def heads_all_slot_path_report(self) -> Optional[dict]:
        """The `heads_all_slot_path` stats block (--heads-all-slot-path on; None without the path): the path's own
        window (lem/model/heads_all_slot_path.py window_summary) and head_read, how evenly each head's whole
        prediction (its attention pool and the path together) answers to the 6 slots on the newest lesson rows
        (at most 4 calls). fixed, per head: the root mean square per number of the old head's output and of the path's
        output, each averaged over those rows first (the vector the part adds to every row alike), of their sum, and
        the cosine between the two (lem/model/heads_all_slot_path.fixed_parts). head_read and fixed run the heads
        without gradient, inside the liveness repair's unwatched block
        (no row joins a watching window, no running centre moves) and under the run's autocast, so the old head's
        part is computed as the run computes it.
        lesson_gradient, per head: how large the next-model-step lesson's gradient is on the transformer's output.
        INPUT: the same lesson rows (the tensor the heads were called with, 6 slots x 192 numbers per row), their
        current tokens and the lesson's targets. OPERATION: per row, the derivative of that row's mean squared
        error (the mean over the sense's 192 numbers of (the head's prediction - the target) squared; the old head
        and the path together, before the gaze copy) with respect to each of the row's 6 x 192 numbers; rms = the
        root mean square of those derivatives over the rows and numbers. OUTPUT: {head: {rms, rows}}. 0 = the
        lesson asks nothing of the transformer through this head. Measured through the old heads alone at
        checkpoint 479,350: tactile 3e-10, vestibular 6e-10, proprioception 4e-5 (the record's section 2). Only
        the rows' own gradient is computed (torch.autograd.grad on the rows): no parameter's .grad is written, no
        running centre moves (unwatched; the path frozen), no random number is drawn."""
        path = getattr(self, "heads_all_slot_path", None)
        if path is None:
            return None
        out = path.window_summary()
        calls = list(self.heads_all_slot_path_gauge)
        self.heads_all_slot_path_gauge.clear()
        if calls:
            from lem.model import heads_all_slot_path as _ASP
            from lem.model import liveness as _LV
            x = torch.cat([c[0] for c in calls], dim=0)
            cur = {m: torch.cat([c[1][m] for c in calls], dim=0) for m in calls[0][1]}
            if x.shape[0] >= 2:
                with torch.no_grad(), _LV.unwatched(self), \
                        torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.dtype):
                    out["head_read"] = _ASP.head_read(self.prediction_heads, x, cur)
                    # the old head's and the path's fixed vectors on the same rows (version 2: fixed_parts)
                    out["fixed"] = _ASP.fixed_parts(self.prediction_heads, x, cur)
            heads = getattr(self.prediction_heads, "_orig_mod", self.prediction_heads)
            sums = {}
            with torch.enable_grad(), _LV.unwatched(self), path.frozen(), \
                    torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.dtype):
                for c in calls:
                    targets = c[2] if len(c) > 2 else {}
                    if not targets:
                        continue
                    xg = c[0].clone().requires_grad_(True)
                    preds = heads(xg, current=c[1], who="heads_all_slot_path.lesson_gradient")
                    for m, tgt in targets.items():
                        if m not in preds or preds[m].shape != tgt.shape:
                            continue
                        row_loss = (preds[m].float() - tgt.float()).pow(2).mean(dim=-1).sum()
                        g, = torch.autograd.grad(row_loss, xg, retain_graph=True)
                        a = sums.setdefault(m, [0.0, 0, 0])
                        a[0] += float(g.float().pow(2).sum())
                        a[1] += int(g.numel())
                        a[2] += int(g.shape[0])
            if sums:
                out["lesson_gradient"] = {m: {"rms": (a[0] / a[1]) ** 0.5, "rows": a[2]} for m, a in sums.items()}
        return out

    def hself_stats(self) -> Optional[dict]:
        """Level 2/3 self-model dark-mode observation stats, or None when off."""
        return None if self._hself is None else self._hself.stats()

    def _self_touch_wellness(self, tactile, B):
        """The self-touch term, per body, already multiplied by its gate.

        WHAT IT PAYS FOR: a body's own parts ARRIVING on it. The value is how many different
        own parts are on the body's skin, on the column the touch adapter already fills as a
        count over four and capped, taken as the largest over the body's valid contact rows.
        What reaches wellness is how much that ROSE since the last step, never the level, so
        pressing one hand on the other and resting there pays once and then pays nothing for
        as long as it is held. Two separate faults are what that rule fixes:

        PAYING FOR DURATION. This was the level, scored every step, so holding a hand in place
        paid the same amount every step it was held - which the previous description of this
        term said it never did.

        PAYING FOR TOUCHING NOTHING ELSE. This was a SHARE of all contact force, so it FELL
        whenever the body also touched the floor, a toy, the mobile or the adult. Its largest
        value was a body curled onto itself, off the floor, touching nothing in the world -
        the same shape as the capability term the review of 2026-09-19 found paying for not
        moving. A count of the body's own parts does not fall when the world is touched too.

        The gate is _self_touch_gate: a run of controlled readings standing above their own
        spread, not one reading above zero. With the weight at 0, the default, this returns
        zeros and the wellness sum is unchanged term for term from before the term existed.

        This has no path to the action gradient - it is built from the environment's own
        numbers - so it moves the value the goal ring and the named-entity teacher read, and
        not the action.

        SUPERSEDED (2026-09-22, DESIGN D10) by the contact_control channel (--contact-head on
        with --contact-control-wellness W): that channel pays a self-touch only when the contact
        head predicted it on both touching parts before it began, weights it 2 to 1 over an
        outside touch, habituates per pair and never pays a pair touching at rest. This term
        still pays an arrival whether or not it was foreseen; it stays working so an old launch
        line means what it meant, at the 0.0 H84 runs.
        """
        weight = float(getattr(self, "self_touch_wellness_weight", 0.0))
        gate = self._self_touch_gate()
        self.last_self_touch_gate = gate
        valid = (tactile[..., TOUCH_VALID_COL] > 0.5).to(tactile.dtype)
        # column 12 is how many different own parts are on that patch, over 4 and capped
        parts = (tactile[..., 12] * valid).max(-1).values.reshape(B, 1)
        prev = getattr(self, "_self_touch_prev_parts", None)
        if prev is None or prev.shape != parts.shape or prev.device != parts.device:
            prev = torch.zeros_like(parts)
        arrived = (parts - prev).clamp_min(0.0)
        self._self_touch_prev_parts = parts.detach()
        self.last_self_touch_parts = float(parts.mean())
        return arrived * (weight * gate)

    @torch.no_grad()
    def _self_touch_audit(self, prediction, target, shuffled, token=None):
        """How well the next-moment prediction anticipated the body touching ITSELF.

        WHAT IS COMPARED, exactly. The prediction and its target are the next moment's touch
        EMBEDDING, one vector per body, not the touch token: the encoder's output columns mean
        nothing row by row, so there is no way in this space to look only at the rows that
        carry self-contact. Scoring the whole embedding for the bodies that were touching
        themselves therefore measures the ordinary touch prediction for those bodies, and the
        floor the body rests on every step is in it. The call site used to say the comparison
        was restricted to the contacts whose self-share is above zero. It never was.

        SO THE NUMBER THAT COUNTS IS A CONTRAST. The same improvement is measured on the
        bodies that were NOT touching themselves, and the difference is reported. Whatever the
        floor contributes, it contributes to both, so it cancels. A contrast at zero says
        knowing the action helped a self-touching body exactly as much as any other body,
        which is the reading "self-touch in particular is not being anticipated". Its null is
        zero by construction, which is what lets the gate judge it against its own spread.

        Returns the squared error on the self-touching bodies, the same for the
        shuffled-action control, the share of the control that the real action recovered, the
        same three for the bodies that were not touching themselves, and the contrast between
        the two shares. None when no contact this step was the body's own, which is most steps
        and is itself worth reporting, and the contrast is None when every body was touching
        itself, because then there is nothing to contrast against.
        """
        bodies = self_touching_bodies(token)
        if bodies is None or not bool(bodies.any()):
            return None
        if not torch.is_tensor(prediction) or not torch.is_tensor(target):
            return None
        if prediction.shape != target.shape or prediction.shape[0] != bodies.shape[0]:
            return None

        def share(mask):
            """How much of the shuffled control's error knowing the real action recovered."""
            if not bool(mask.any()) or not (torch.is_tensor(shuffled)
                                            and shuffled.shape == target.shape):
                return None, None, None
            real = float(((prediction[mask] - target[mask]) ** 2).mean())
            null = float(((shuffled[mask] - target[mask]) ** 2).mean())
            return real, null, ((null - real) / null if null > 0 else None)

        real, null, got = share(bodies)
        other = ~bodies
        o_real, o_null, o_got = share(other)
        out = {"bodies": int(bodies.sum()), "other_bodies": int(other.sum()),
               "rows": int(self_touch_rows(token).sum()),
               "prediction_mse": (float(((prediction[bodies] - target[bodies]) ** 2).mean())
                                  if real is None else real),
               "shuffled_mse": null, "improvement_over_shuffled": got,
               "other_prediction_mse": o_real, "other_shuffled_mse": o_null,
               "other_improvement_over_shuffled": o_got,
               "improvement_over_other_bodies": (None if got is None or o_got is None
                                                 else got - o_got)}
        return out

    @torch.no_grad()
    def _self_touch_gate(self) -> float:
        """Whether the self-touch term may pay anything, from a RUN of controlled readings.

        This was one reading above zero. The reading's own spread is far wider than the
        improvement that was cited for it: over the 154 controlled readings of two segments its
        average was -0.0001 and its spread 0.0169, and 85 of the 154 sat above zero, which is
        55 in 100 against a coin's 50. One lucky reading held the gate open until the next
        control, because nothing ever cleared the stored one.

        What is judged now is the CONTRAST from the audit, whose null is zero by construction,
        over the last SELF_TOUCH_READINGS controlled readings. The gate opens when the average
        of those readings stands more than twice its own standard error above zero, which is a
        line drawn from the readings themselves rather than chosen. It shuts again the moment
        the average falls back, and it shuts when the newest reading is older than
        SELF_TOUCH_STALE_STEPS, so a gate can never be held open by a measurement that has
        stopped arriving.
        """
        weight = float(getattr(self, "self_touch_wellness_weight", 0.0))
        if weight <= 0.0:
            self.last_self_touch_gate_why = "the run gives this term a weight of 0"
            return 0.0
        ring = self._self_touch_ring
        age = self.step_count - int(self._self_touch_ring_step)
        if len(ring) < SELF_TOUCH_READINGS:
            self.last_self_touch_gate_why = (
                "%d of the %d controlled readings needed" % (len(ring), SELF_TOUCH_READINGS))
            return 0.0
        if age > SELF_TOUCH_STALE_STEPS:
            self.last_self_touch_gate_why = (
                "the newest controlled reading is %d steps old, past the %d allowed"
                % (age, SELF_TOUCH_STALE_STEPS))
            return 0.0
        vals = list(ring)
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        err = (var / n) ** 0.5
        self.last_self_touch_gate_mean = mean
        self.last_self_touch_gate_line = 2.0 * err
        if mean > 2.0 * err:
            self.last_self_touch_gate_why = (
                "over %d controlled readings the prediction helped a self-touching body "
                "%.4f more than it helped the others, against a line of %.4f drawn at twice "
                "the readings' own error" % (n, mean, 2.0 * err))
            return 1.0
        self.last_self_touch_gate_why = (
            "over %d controlled readings the prediction helped a self-touching body %.4f "
            "more than it helped the others, under the %.4f it must clear, which is twice "
            "the readings' own error" % (n, mean, 2.0 * err))
        return 0.0

    def _migrate_touch_push_scale(self, path: str, load_optimizer: bool = True) -> None:
        """Carry the tactile encoder across the change of the push's two divisors, exactly.

        A touch row's depth column is the contact's depth over PUSH_DEPTH_SCALE_CM and its
        arrival column the closing speed over PUSH_ARRIVAL_SCALE_CMS. On 2026-09-21 those went
        from 10 cm and 500 cm/s to 0.1 cm and 2 cm/s, measured from this body's own contacts,
        so the same contact now reads k times larger in each column: k = old divisor over new,
        100 for depth and 250 for speed. The first layer's weights on those two columns are
        divided by the same k, which leaves every token the encoder produces exactly as it
        was, and the optimiser's running averages for those weights are moved to match - the
        gradient of a weight scales with its input, so the first moment is multiplied by k and
        the second by k squared, which keeps each update the size it would have been.

        A checkpoint that does not record its scale was saved before the change and used the
        old divisors. One that records the scale in force needs nothing."""
        import json as _json
        from mimo_bridge.touch_adapter import (LEGACY_PUSH_SCALE, PUSH_ARRIVAL_SCALE_CMS,
                                               PUSH_DEPTH_SCALE_CM)
        saved = list(LEGACY_PUSH_SCALE)
        receipt = os.path.join(path, "model_receipt.json")
        if os.path.isfile(receipt):
            try:
                with open(receipt, encoding="utf-8") as fh:
                    saved = list(_json.load(fh).get("touch_push_scale", LEGACY_PUSH_SCALE))
            except (OSError, ValueError):
                saved = list(LEGACY_PUSH_SCALE)
        now = [float(PUSH_DEPTH_SCALE_CM), float(PUSH_ARRIVAL_SCALE_CMS)]
        ks = [float(a) / float(b) for a, b in zip(saved, now)]
        if all(abs(k - 1.0) < 1e-9 for k in ks):
            return
        enc = self.encoder_registry.encoders.get("tactile") if hasattr(
            self.encoder_registry.encoders, "get") else self.encoder_registry.encoders["tactile"]
        if enc is None or not hasattr(enc, "patch_mlp"):
            return
        w = enc.patch_mlp[0].weight
        if w.shape[1] != TOUCH_TOKEN_F:
            log.warning("touch push scale: the tactile encoder's first layer is %d wide, not "
                        "the token's %d; nothing migrated", w.shape[1], TOUCH_TOKEN_F)
            return
        state = self.optimizer.state.get(w) if (load_optimizer and getattr(
            self, "optimizer", None) is not None) else None
        with torch.no_grad():
            for col, k in ((TOUCH_DEPTH_COL, ks[0]), (TOUCH_ARRIVAL_COL, ks[1])):
                w[:, col].div_(k)
                if state:
                    if torch.is_tensor(state.get("exp_avg")):
                        state["exp_avg"][:, col].mul_(k)
                    if torch.is_tensor(state.get("exp_avg_sq")):
                        state["exp_avg_sq"][:, col].mul_(k * k)
        log.warning("touch push scale: this checkpoint's weights were trained on depth over "
                    "%g cm and arrival over %g cm/s; the code now divides by %g cm and %g cm/s, "
                    "so the tactile encoder's weights on those two columns were divided by %g "
                    "and %g and their optimiser averages moved to match. Every token reads "
                    "exactly as it did.", saved[0], saved[1], now[0], now[1], ks[0], ks[1])

    def _migrate_efference_scales(self, path: str, load_optimizer: bool = True) -> None:
        """Reconcile a checkpoint's efference stamp of THIS version with this build
        (lem.model.efference.reconcile): divisors migrated exactly (each output row times old/new,
        Adam's averages times k and k squared - the precedent is _migrate_touch_push_scale); a
        different order length (--motor-hold) puts the tenure rows back to their initial weights
        and says so (review MINOR 3); a different loss weight is reported. A checkpoint of another
        efference version is not reconciled here: load_checkpoint starts its heads fresh."""
        if getattr(self, "efference", None) is None:
            return
        rp = os.path.join(path, "model_receipt.json")
        if not os.path.isfile(rp):
            return
        with open(rp, encoding="utf-8") as fh:
            saved = json.load(fh).get("efference_heads") or {}
        from lem.model import efference as EF
        if not saved or saved.get("version", EF.EFFERENCE_VERSION) != EF.EFFERENCE_VERSION:
            return
        done = EF.reconcile(self, saved, load_optimizer, log=log)
        if done:
            self.efference_load = dict(self.efference_load or {}, **done)

    def _migrate_contact_order_scales(self, path: str, load_optimizer: bool = True) -> None:
        """Carry the contact head across a change of its order divisors, exactly.

        The head reads each body handle's asked position change over ORDER_POS_CM and asked turn
        over ORDER_ROT_DEG (lem/model/contact_head.order_features), and the receipt stamps the
        divisors it was trained under. If the code's differ, the same order now reads k = old /
        new times larger in those columns, so ctx_ord's weights on them are divided by k and
        Adam's running averages moved to match - a weight's gradient scales with its input, so
        exp_avg x k and exp_avg_sq x k^2 - the precedent is _migrate_touch_push_scale. A
        checkpoint without the stamp has no head to migrate."""
        if getattr(self, "contact_head", None) is None:
            return
        rp = os.path.join(path, "model_receipt.json")
        if not os.path.isfile(rp):
            return
        with open(rp, encoding="utf-8") as fh:
            saved = (json.load(fh).get("contact_head") or {}).get("order_scales")
        if not saved:
            return
        from lem.model import contact_head as CH
        k_pos = float(saved.get("pos_cm", CH.ORDER_POS_CM)) / float(CH.ORDER_POS_CM)
        k_rot = float(saved.get("rot_deg", CH.ORDER_ROT_DEG)) / float(CH.ORDER_ROT_DEG)
        if abs(k_pos - 1.0) < 1e-9 and abs(k_rot - 1.0) < 1e-9:
            return
        lin = self.contact_head.ctx_ord
        k = self.contact_head.order_scale_factors(k_pos, k_rot).to(lin.weight.device, lin.weight.dtype)
        with torch.no_grad():
            lin.weight.div_(k)
            st = self.optimizer.state.get(lin.weight) if load_optimizer else None
            if st:
                if torch.is_tensor(st.get("exp_avg")):
                    st["exp_avg"].mul_(k)
                if torch.is_tensor(st.get("exp_avg_sq")):
                    st["exp_avg_sq"].mul_(k * k)
        log.warning("contact head: this checkpoint's head read orders over %s; the code now divides "
                    "by %.4g cm and %.4g degrees, so its order weights were divided by %.4g "
                    "(position) and %.4g (turn) and their optimiser averages moved to match. Every "
                    "prediction reads as it did.", saved, CH.ORDER_POS_CM, CH.ORDER_ROT_DEG,
                    k_pos, k_rot)

    def _vestibular_encoder_repair_on_load(self, path: str, load_optimizer: bool = True) -> None:
        """--vestibular-encoder-repair: the load-time switch (mimo_bridge/vestibular_sense.py, "the
        encoder repair"), once per lineage.

        With "rescale" or "rescale_restart", after the weights, the optimizer and the other scale
        migrations are in place: a checkpoint whose receipt records no switch, and whose vestibular encoder's
        last-layer bias has a spread above vestibular_sense.REPAIR_SWITCH_LINE (1,000), has that layer
        divided by a power of two (to a spread of 9.2-18.3) and its Adam moments moved with it ("rescale"),
        or every encoder tensor's Adam state dropped so Adam restarts it ("rescale_restart")
        (vestibular_sense.repair_encoder); the tokens stay the same and the record goes into this build's
        receipt, so the next checkpoint carries it. The next load divides nothing: the receipt says so, and
        the spread is then far below the line even where a build without this branch dropped the record
        (review finding 7). The runtime part follows the runtime state's restore
        (_vestibular_encoder_repair_after_runtime). A checkpoint that records a switch is left as it is. With "off" nothing is divided and nothing is stamped; a checkpoint that
        records a switch keeps its record in this build's receipt (its weights are ordinary weights and
        load unchanged: removing the flag does not undo a switch, only the pre-switch checkpoint does)."""
        saved = None
        rp = os.path.join(path, "model_receipt.json")
        if os.path.isfile(rp):
            try:
                with open(rp, encoding="utf-8") as fh:
                    saved = json.load(fh).get("vestibular_encoder_repair")
            except (OSError, ValueError):
                saved = None
        done = saved.get("switched") if isinstance(saved, dict) else None
        self.vestibular_encoder_repair_record = None
        if self.vestibular_encoder_repair == "off":
            if done:
                # the switch's record rides on; the build that carried it says it ran with the flag off
                self.receipt["vestibular_encoder_repair"] = dict(saved, carried_by_build_mode="off")
                log.warning("vestibular encoder repair: this checkpoint's vestibular encoder was divided at "
                            "step %s (by %.6g); this build (--vestibular-encoder-repair off) divides nothing "
                            "and keeps the record. Removing the flag does not undo the switch; the "
                            "checkpoint before it does.", done.get("step"), float(done.get("divided_by", 0.0)))
            return
        if done:
            self.receipt["vestibular_encoder_repair"] = dict(self.receipt["vestibular_encoder_repair"],
                                                             switched=done)
            log.info("vestibular encoder repair: already switched at step %s (divided by %.6g); nothing "
                     "divided at this load", done.get("step"), float(done.get("divided_by", 0.0)))
            return
        from .vestibular_sense import repair_encoder, repair_message
        rec = repair_encoder(self, load_optimizer=load_optimizer,
                             restart_moments=self.vestibular_encoder_repair in ("rescale_restart",
                                                                                "rescale_restart_fp32"))
        rec["step"] = int(self.step_count)
        rec["checkpoint"] = str(path)
        # the mode the switch ran: the kill lines take the revival's declared cost only from the pause measurement's
        # arm of this mode (third review, section E)
        rec["mode"] = str(self.vestibular_encoder_repair)
        if rec["switched"]:
            self.vestibular_encoder_repair_record = rec
            self.receipt["vestibular_encoder_repair"] = dict(self.receipt["vestibular_encoder_repair"],
                                                             switched=rec)
            log.warning("%s", repair_message(rec, int(self.step_count)))
        else:
            self.receipt["vestibular_encoder_repair"] = dict(self.receipt["vestibular_encoder_repair"],
                                                             not_needed=rec)
            spread, floor = float(rec["bias_spread_before"]), float(rec["spread_floor"])
            (log.warning if spread > 2.0 * floor else log.info)("vestibular encoder repair: %s", rec["why"])

    def _vestibular_encoder_repair_after_runtime(self) -> None:
        """--vestibular-encoder-repair, the second part of a switch, after the runtime state is restored:
        the next-moment normalisation drops this sense's running average and learning progress holds the
        key out for vestibular_sense.REPAIR_LP_HOLDOUT_STEPS model steps (vestibular_sense.
        repair_runtime_restart), both kept in the switch's record. Nothing at a load that did not switch."""
        rec = getattr(self, "vestibular_encoder_repair_record", None)
        if not rec:
            return
        from .vestibular_sense import repair_runtime_restart
        done = repair_runtime_restart(self)
        rec["runtime_restart"] = done
        sw = (self.receipt.get("vestibular_encoder_repair") or {}).get("switched")
        if isinstance(sw, dict):
            sw["runtime_restart"] = done
        lp = done.get("learning_progress") or {}
        steps = [x for x in (lp.get("channel_step_per_body") or []) if x is not None]
        log.warning("vestibular encoder repair: the next-moment normalisation's running average of this sense "
                    "was restarted (%s) and learning progress holds it out for %s model steps; its channel "
                    "moved at the switch by %s per body",
                    (done.get("next_frame_norm") or {}).get("dropped_readings", "no")
                    if isinstance(done.get("next_frame_norm"), dict) else done.get("next_frame_norm", "no"),
                    lp.get("held_out_steps", "no"),
                    ("%+.4f to %+.4f" % (min(steps), max(steps))) if steps else "nothing measured")

    def _restore_widened_touch_encoder(self, path: str, result: dict) -> None:
        """Put the tactile encoder's learning back after the touch token grew.

        The saved tensors are read, never rewritten: a checkpoint is evidence. Every
        tensor whose shape still matches is copied straight in; the one projection that
        takes the token as input is moved to the new column layout by widen_touch_columns,
        the SAME function that moves a stored touch token, so the first step after the
        resume produces the tokens the last step before it produced.

        THE PERMUTATION IS THE WHOLE POINT, and it was missing until 2026-09-21. This copied
        old columns 0..n-1 into new columns 0..n-1 while the tokens had their validity flag
        carried to the new last column, so the encoder's trained "is this row a real contact"
        weight ended up on a brand-new column and the column the tokens use for validity
        started at zero. Measured on the live checkpoints of run H84
        (encoder_registry.encoders.tactile.patch_mlp.0.weight, |w| mean per column): segment
        22 at 12 wide had 0.145 on validity column 11; segment 23 at 14 wide had that same
        0.145 on column 11, which is now the self-touch share and is meant to open at exactly
        zero, and 0.000 on column 13, the validity flag. The 19-wide resume of segment 29
        repeated it: segment 28's last checkpoint carried 0.035 on its validity column 13,
        and that 0.035 opened segment 29's column 13, which at 19 wide is the contact's
        depth, while column 18, the validity flag, opened at zero for the second time.
        """
        # the loader names encoders "encoder:tactile"; match on the sense, not the spelling
        if not any("tactile" in str(x) for x in result.get("reinitialized_components", [])):
            return
        try:
            import safetensors.torch as _st
            saved = _st.load_file(os.path.join(path, "model.safetensors"))
        except (ImportError, OSError, ValueError):
            return
        prefix = "encoder_registry.encoders.tactile."
        live = dict(self.encoder_registry.encoders["tactile"].named_parameters())
        live.update(dict(self.encoder_registry.encoders["tactile"].named_buffers()))
        copied, widened, skipped = 0, [], []
        with torch.no_grad():
            for name, tensor in live.items():
                old = saved.get(prefix + name)
                if old is None:
                    skipped.append(name)
                    continue
                old = old.to(tensor.dtype).to(tensor.device)
                if tuple(old.shape) == tuple(tensor.shape):
                    tensor.copy_(old)
                    copied += 1
                elif (old.ndim == tensor.ndim == 2 and old.shape[0] == tensor.shape[0]
                      and old.shape[1] < tensor.shape[1]):
                    tensor.zero_()
                    if tensor.shape[1] == TOUCH_TOKEN_F:
                        # this projection's columns ARE the token's columns, so they move the
                        # way the token's columns move and not left-aligned
                        tensor.copy_(widen_touch_columns(old))
                    else:
                        tensor[:, :old.shape[1]].copy_(old)
                    widened.append(f"{name} {old.shape[1]}->{tensor.shape[1]}")
                else:
                    skipped.append(name)
        # Adam's moments were saved at the old width too. Widening the parameter without
        # them leaves the optimizer holding a 12-column history for a 14-column weight,
        # which torch refuses outright ("params, grads, exp_avgs ... must have same dtype,
        # device, and layout"). Move them the SAME way the weight moved - a moment belongs
        # to its own column - so what was learned about the older columns keeps its momentum
        # and the new columns start with none.
        moments = 0
        if widened and getattr(self, "optimizer", None) is not None:
            with torch.no_grad():
                for name, tensor in live.items():
                    state = self.optimizer.state.get(tensor)
                    if not state:
                        continue
                    for key, value in list(state.items()):
                        if (torch.is_tensor(value) and value.ndim == tensor.ndim
                                and value.shape != tensor.shape
                                and value.shape[0] == tensor.shape[0]
                                and value.shape[-1] < tensor.shape[-1]):
                            value = value.to(tensor.dtype).to(tensor.device)
                            if tensor.shape[-1] == TOUCH_TOKEN_F:
                                grown = widen_touch_columns(value)
                            else:
                                grown = torch.zeros_like(tensor)
                                grown[..., :value.shape[-1]] = value
                            state[key] = grown
                            moments += 1
        log.warning("touch sense widened to %d columns: the tactile encoder kept %d of its "
                    "saved tensors, widened %s with the new columns at zero, and carried "
                    "%d optimizer moments across%s",
                    TOUCH_TOKEN_F, copied, ", ".join(widened) or "none", moments,
                    "" if not skipped else f"; could not place {len(skipped)}")

    def _motor_interface_changes(self, saved_receipt: dict) -> dict:
        """{setting: [saved, now]} for the two motor interface settings (audit 2026-09-25, F1 and
        F4) that differ between a checkpoint's receipt and this build; a receipt without the stamp
        was trained under the old default (legacy, off)."""
        changes = {}
        eye_saved = str(saved_receipt.get('eye_row_convention', 'legacy'))
        if eye_saved != self.eye_row_convention:
            changes['eye_row_convention'] = [eye_saved, self.eye_row_convention]
        fields_saved = bool(saved_receipt.get('executed_fields_only', False))
        if fields_saved != self.executed_fields_only_on:
            changes['executed_fields_only'] = [fields_saved, self.executed_fields_only_on]
        return changes

    def _convert_order_averages(self, efference_restored: bool, self_state_loaded: bool) -> Optional[dict]:
        """Convert the saved averages over the order's 234 fields at a motor interface switch.

        WHY (motor fixes review 2026-09-25, finding 2). The efference body head reads every order
        field minus a running mean over 2,000 lived moments (lem/model/efference.py, order_centre,
        saved in runtime_state.pt). Measured on the replay ring of checkpoint 401,750 (8 bodies):
        the ledger's radius column has mean 0.500 and standard deviation 0.288. Under
        --executed-fields-only on every radius is 0, so without a conversion the head's centred
        input carries -0.50 on all 26 radius columns, 1.7 of their standard deviations, on every
        moment; the mean follows at 1/2,000 a moment and has removed 1 - exp(-400/2000) = 18% of
        that offset by the time the deploy's kill lines are judged (step N + 400).

        WHAT IS CONVERTED, only where the conversion is exact: executed_fields_only switched OFF ->
        ON sets every radius entry to 0 (every order now carries radius 0, and command_for_encoder
        passes the radius through, so their mean is exactly 0).

        THE EYE ROW IS NOT CONVERTED (the deploy branch's review, finding 3a; this code negated the
        eye row's x and y entries until then, on the first review's advice). At an
        --eye-row-convention switch the weights are the same, so the controller writes the same eye
        row; _gaze_from_handle now sends the servo MINUS its pitch, and gaze_to_handle writes the
        ledger's pitch as minus the elevation sent - the controller's row again (the review's probe:
        equal to the controller's row to 6e-8 under both conventions); what flips is the eye's
        movement. That holds for the command's SHARE of the ledger row only: the ledger is written from
        the gaze the world sent, which adds the saccade generator's offset, clips to the eye's range and,
        while a reflex snap steers, sends only 0.3 of that with 0.7 of the snap's target
        (motor_fixes_eye_row_split.world_sent: the world's arithmetic; the fourth review's findings 2 and
        10). So after the switch the ledger row keeps the command's share (all of the command with no
        snap, 0.3 of it during a snap) and writes the offset's and the snap target's shares mirrored,
        the clip aside. Left as saved, the average is off for new orders by the switch's change to their
        average (negated, by more). The eye entries are
        left as saved (CENTRE_LEFT_AS_SAVED: since the fourth review's fix round nothing moves them after
        the restore either). What does change is the meaning of the rows STORED before the switch: the
        replay ring and the short histories are rewritten by convert_stored_eye_rows.
        LEFT AS SAVED, and named in the record: the eye row's entries at a convention switch (above),
        executed_fields_only ON -> OFF (the mean of the free radius numbers the orders will carry
        again is not known; the running mean moves 1/2,000 of the way a moment), and the positions of
        the rows whose position the decoder ignores (under the flag they follow the achieved pose,
        whose mean the average does not hold; the review measured their mean moving 0.004 encoder
        units, 0.9 cm).

        WHICH AVERAGES: the efference heads' order centre when THIS load restored it from the
        checkpoint (a cold centre is set by the first lived moment, already in the new form), and
        the grounded self dynamics' command calibration (self_state.grounded.action_center, the
        same 234 fields in encoder units) when this load read it from the checkpoint. The
        calibration's scale needs no change: a radius of 0 minus a centre of 0 reads 0 whatever the
        scale. On H84 checkpoint 401,750 that calibration is the identity (centre 0, scale 1), so the
        conversion leaves it unchanged.

        Returns the record (also self.motor_interface_conversion), or None when no setting changed."""
        change = self.motor_interface_change or {}
        if not change:
            self.motor_interface_conversion = None
            return None
        eye = "eye_row_convention" in change
        fields = change.get("executed_fields_only")
        radius_to_zero = bool(fields) and not bool(fields[0]) and bool(fields[1])
        radius_left = bool(fields) and bool(fields[0]) and not bool(fields[1])
        H, gi = len(self.handle_names), int(self.GAZE_IDX)
        radius = torch.tensor([h * 9 + 7 for h in range(H)], dtype=torch.long)
        eye_xy = torch.tensor([gi * 9 + 3, gi * 9 + 4], dtype=torch.long)
        targets = []
        eff = getattr(self, "efference", None)
        if eff is not None and efference_restored and float(eff.order_centre_weight) > 0.0:
            targets.append(("efference.order_centre", eff.order_centre))
        grounded = getattr(getattr(self, "self_state", None), "grounded", None)
        centre = getattr(grounded, "action_center", None)
        if self_state_loaded and torch.is_tensor(centre) and centre.numel() == H * 9:
            targets.append(("self_state.grounded.action_center", centre))
        record = {"changes": dict(change), "averages": {}}
        for name, buf in targets:
            with torch.no_grad():
                before = buf.detach().clone()
                if radius_to_zero:
                    buf[radius.to(buf.device)] = 0.0
                moved = (buf.detach() - before).abs()
            record["averages"][name] = {
                "radius_set_to_0": bool(radius_to_zero),
                "largest_change": round(float(moved.max()), 6),
                "radius_mean_before": round(float(before[radius.to(buf.device)].mean()), 6),
                "eye_row_x_y": [round(float(v), 6) for v in before[eye_xy.to(buf.device)].tolist()]}
        record["left_as_saved"] = (
            ["the positions of the rows whose position the decoder ignores (not exact)"]
            + (["the eye row's entries: at a convention switch the controller writes the same row and only the "
                "command sent to the eye servo flips; the ledger row, written from the gaze the world sent, keeps "
                "the command's share of it (all of the command with no reflex snap, 0.3 of it while a snap "
                "steers) and writes the saccade offset's and the snap target's shares mirrored, the clip at the "
                "eye's range aside; " + LemLite.CENTRE_LEFT_AS_SAVED + " (order_centre_eye_shift)"]
               if eye else [])
            + (["every radius entry: executed_fields_only switched off, and the free radii's mean is "
                "not known (the running mean moves 1/2,000 of the way a moment)"] if radius_left else []))
        if not targets:
            record["left_as_saved"].append("no saved average over the order was restored by this load")
        self.motor_interface_conversion = record
        log.warning("motor interface switch: saved averages over the order's fields converted where exact "
                    "(%s): %s; left as saved: %s", ", ".join(sorted(change)),
                    {k: {kk: v[kk] for kk in ("radius_set_to_0", "largest_change")}
                     for k, v in record["averages"].items()} or "none restored",
                    "; ".join(record["left_as_saved"]))
        return record

    @staticmethod
    def mirror_eye_rows(orders, gaze_idx: int):
        """Orders [..., H, 9] (execution or encoder units) or flattened [..., H * 9] with the eye row's
        quaternion x and y negated: the row gaze_to_handle writes for the same eye command in the other
        eye row convention (exact, and its own inverse). Out of place for tensors and arrays."""
        if orders is None:
            return None
        gi = int(gaze_idx)
        out = orders.clone() if torch.is_tensor(orders) else np.array(orders, copy=True)
        if out.shape[-1] == 9:
            out[..., gi, 3:5] = -out[..., gi, 3:5]
        else:
            out[..., gi * 9 + 3:gi * 9 + 5] = -out[..., gi * 9 + 3:gi * 9 + 5]
        return out

    def _ctrl_gaze_for_ring(self, executed_ctrl_gaze, exec_t, B) -> dict:
        """The replay ring's ctrl_gaze argument: {} without a command or an order, or when the command is
        not [B, 2] (a recording must never stop training: it is left out and said once)."""
        if executed_ctrl_gaze is None or exec_t is None:
            return {}
        g = np.asarray(executed_ctrl_gaze, dtype=np.float32)
        if g.size != int(B) * 2:
            if not getattr(self, "_ctrl_gaze_shape_said", False):
                log.warning("replay: the model's gaze command has shape %s for %d bodies; it is not "
                            "recorded beside the order (ReplayMemory.has_ctrl_gaze stays False)",
                            tuple(g.shape), int(B))
                self._ctrl_gaze_shape_said = True
            return {}
        return {"ctrl_gaze": torch.as_tensor(g).reshape(int(B), 2)}

    def _eye_contract_bit(self, ctrl_kwargs) -> int:
        """replay's eye row bit (limb_target_control.EYE_ROW_CONTRACT_BIT) for the order back-filled at this add:
        set under --eye-row-convention physical when the model's gaze command arrives with it (ctrl_kwargs:
        _ctrl_gaze_for_ring's {} or {'ctrl_gaze': ...}), else 0. Under 'legacy' it is always 0, so the stored
        contract is the one of before, bit for bit (the fourth review's fix round). At an F1 switch the trainer
        drops the command of the last transition before it with the rewrite of that transition's ledger
        (train_mimo, after switch_stored_eye_rows), so the first order back-filled after the switch, executed
        before it, carries no bit: every moment carrying it was recorded after the switch."""
        if not ctrl_kwargs or getattr(self, "eye_row_convention", "legacy") != "physical":
            return 0
        from .limb_target_control import EYE_ROW_CONTRACT_BIT
        return int(EYE_ROW_CONTRACT_BIT)

    # THE TWO RUNNING AVERAGES OF THE EYE ROW AT AN F1 SWITCH (the efference order centre's eye entries and, with
    # the memory repair, its running mean of events' eye columns) are LEFT AS SAVED since the fourth review's fix
    # round. Until then the switch moved both by the change the replay ring's recorded moments gave for new
    # orders' average (LemLite.ring_eye_row_shift, removed), which needed 6,000 steps of the model's gaze command
    # recorded in the ring on merged code before the switch - training with the eye bug live, which the restart
    # rule forbids (the coordinator's decision: the deploy needs no refilled ring) - and whose row formula
    # ignored the gaze reflex's blend while a snap steers (the fourth review's finding 2).
    CENTRE_LEFT_AS_SAVED = (
        "left as saved at the switch (the fourth review's fix round, the coordinator's decision): the change the "
        "switch makes to new orders' average eye row needs, for every stored order, the model's gaze command and "
        "the reflex snap's and the saccade generator's state at each physics step, which the replay ring does not "
        "hold (at the deploy it holds no command: filling it would mean training with the eye bug live). New "
        "orders' average then differs from this centre by the switch's change to it, which the deploy stop "
        "measures without training (motor_fixes_eye_row_split.py, switch_change_of_new_orders_mean); the running "
        "mean follows new orders at 1/2,000 a step (37 of every 100 parts of that difference left after 2,000 "
        "steps, 5 after 6,000), and the switch-jump job's second body-head reading is taken with the centre left "
        "as saved")

    @classmethod
    def _centre_left_as_saved(cls, centre, cols) -> dict:
        """The record of a running average's eye entries `cols` left as saved at an F1 switch."""
        return dict(shift=None, entries=[int(c) for c in cols],
                    centre=[float(v) for v in centre[cols].tolist()], left_as_saved=cls.CENTRE_LEFT_AS_SAVED)

    def convert_stored_eye_rows(self) -> dict:
        """At an --eye-row-convention switch, rewrite every STORED ledger eye row in the new convention.

        WHY (the deploy branch's review, finding 3b). A ledger eye row is the row of the eye command
        actually sent, paired with the outcome that command produced. After the switch the heads read
        rows in the new convention: a row stored before it would be read as the MIRRORED elevation of
        what it sent - for replay, 6,000 moments per body of vision's gaze copy taught a relation that
        is wrong in both conventions, beside live moments that teach the right one. Rewriting the
        stored row as the new convention's row for the same command is exact (negate the quaternion's
        x and y; the command, the outcome and every other field stay).

        WHAT IS REWRITTEN (restored from the checkpoint and its episode companion before this runs),
        each with what reads it (STORED_EYE_ROW_READERS): the replay ring's executed orders; the
        efference heads' 100/150 ms gaze history, the order in its tenure and the previous order; the
        imagination chain's recent orders; the grounded self's recorded actions and its rehearsal
        windows; the executed-source smoother's memory; the persistent memory's events (its history's
        retained events, the previous and the pending moment, and the learner's two snapshots of it:
        the delayed lesson's _nf_memory and the tenure's _tenure_pending["memory"]) - their order
        columns hold the ledger row in encoder units (memory_integration.eye_row_columns; the deploy
        branch's second review, finding 1). The caller rewrites the trainer's own ledger of the last
        transition (prev_executed). NOT rewritten, because they are ENCODED and cannot be
        (ENCODED_EYE_ROW_CARRIERS): the live context's motor tokens, the replay ring's stored memory
        states, the memory's recurrent carry, and the imagination chain's saved contexts and memory
        biases; each ages out.

        THE TWO RUNNING AVERAGES OF THE EYE ROW, by one method (the coordinator's decision after the four F1
        follow-ups): the efference order centre's eye entries and, when the memory repair's running mean of
        events is present and set, its eye columns, each LEFT AS SAVED (the fourth review's fix round:
        CENTRE_LEFT_AS_SAVED says why and what it leaves). Recorded in
        motor_interface_conversion["order_centre_eye_shift"] and ["event_centre_eye_shift"].

        THE REPLAY RING'S MOMENTS STORED BEFORE THE SWITCH are rewritten here too (exact), but under
        'physical' replay no longer studies them (replay_update's eye row contract): their stored memory and
        self states were encoded from old-convention rows, which no rewrite reaches.

        Returns {store: rows rewritten}."""
        gi = int(self.GAZE_IDX)
        done = {}
        memory = self._memory_event_stores()          # refuses an unknown memory layout before any rewrite
        eff = getattr(self, "efference", None)
        conv = getattr(self, "motor_interface_conversion", None)
        if eff is not None and float(getattr(eff, "order_centre_weight", 0.0)) > 0.0:
            rec = self._centre_left_as_saved(eff.order_centre, [gi * 9 + 3, gi * 9 + 4])
            if isinstance(conv, dict):
                conv["order_centre_eye_shift"] = rec
            log.warning("motor interface switch (eye row convention): the order centre's eye entries %s: %s",
                        rec["centre"], rec["left_as_saved"])
        rp = getattr(self, "replay", None)
        if rp is not None and torch.is_tensor(getattr(rp, "executed", None)):
            with torch.no_grad():
                rp.executed[..., gi, 3:5] = -rp.executed[..., gi, 3:5]
            n = rp.count if rp.count < rp.size else rp.size
            done["replay.executed"] = int(rp.executed.shape[0]) * int(n)
        hist = getattr(self, "_eff_hist", None)
        if hist:
            for r in hist:
                if torch.is_tensor(r.get("order")):
                    r["order"] = self.mirror_eye_rows(r["order"], gi)
            done["efference.gaze_history"] = len(hist)
        ten = getattr(self, "_eff_tenure", None)
        if isinstance(ten, dict) and torch.is_tensor(ten.get("order")):
            ten["order"] = self.mirror_eye_rows(ten["order"], gi)
            done["efference.tenure_order"] = 1
        if torch.is_tensor(getattr(self, "_eff_prev_order", None)):
            self._eff_prev_order = self.mirror_eye_rows(self._eff_prev_order, gi)
            done["efference.previous_order"] = 1
        mh = getattr(self, "motor_history", None)
        if mh:
            items = [self.mirror_eye_rows(x, gi) if torch.is_tensor(x) else x for x in mh]
            mh.clear()
            mh.extend(items)
            done["imagination.motor_history"] = len(items)
        if torch.is_tensor(getattr(self, "_smooth_exec_prev", None)):
            self._smooth_exec_prev = self.mirror_eye_rows(self._smooth_exec_prev, gi)
            done["smoother.executed_previous"] = 1
        sh = getattr(self, "self_history", None)
        if sh is not None and getattr(sh, "events", None):
            for e in sh.events:
                if torch.is_tensor(e.get("action")):
                    e["action"] = self.mirror_eye_rows(e["action"], gi)
            done["self_history.actions"] = len(sh.events)
        sr = getattr(self, "self_rehearsal", None)
        if sr is not None:
            n = 0
            for entry in list(getattr(sr, "common", [])) + [e for _, e in getattr(sr, "changed", [])]:
                acts = entry.get("actions") if isinstance(entry, dict) else None
                if isinstance(acts, list):
                    entry["actions"] = [self.mirror_eye_rows(x, gi) if torch.is_tensor(x) else x for x in acts]
                    n += 1
            if n:
                done["self_rehearsal.windows"] = n
        done.update(self._convert_memory_eye_rows(memory))
        log.warning("motor interface switch (eye row convention): stored ledger eye rows rewritten in the "
                    "new convention (quaternion x and y negated; the command sent and its outcome are "
                    "unchanged): %s; encoded, not rewritten (they age out): %s", done or "none stored",
                    ", ".join(sorted(self.ENCODED_EYE_ROW_CARRIERS)))
        return done

    # what reads each store convert_stored_eye_rows rewrites (the deploy branch's second review, point 7)
    STORED_EYE_ROW_READERS = {
        "replay.executed": "the replay lesson: the transformer's motor token (rebuilt from the achieved pose "
                           "and the order) and vision's gaze copy, which reads the eye row in the convention; "
                           "under 'physical' replay studies no window over a moment stored before the switch "
                           "(the eye row's replay contract), so the rewrite keeps the ring consistent for its "
                           "other readers (the switch-jump job, a later switch)",
        "efference.gaze_history": "the gaze head's 100 and 150 ms lesson and its shuffled control, which read "
                                  "the eye row in the convention",
        "efference.tenure_order": "the body head's tenure lesson, once per body when the order in force ends; "
                                  "the body head reads the row without the convention and its target leaves "
                                  "the eyes out",
        "efference.previous_order": "the efference lesson's test for a new order, which leaves the eye row out: "
                                    "the rewrite changes nothing it computes",
        "imagination.motor_history": "the imagination lesson's 3-step chain: the transformer's motor token",
        "smoother.executed_previous": "the executed-source smoother and the planner's copy of it, which pass "
                                      "the eye row through unblended: the rewrite changes nothing sent",
        "self_history.actions": "the grounded self's consequence forecast over 1, 5, 20 and 100 recorded moments",
        "self_rehearsal.windows": "the grounded self's rehearsal of stored windows",
        "persistent_memory.history": "the persistent memory's encoder (its bias into the transformer, and the "
                                     "state its needs forecast reads)",
        "persistent_memory._nf_memory": "the delayed lesson's memory bias and the needs forecast",
        "persistent_memory._tenure_pending": "the tenure lesson's memory bias",
        "trainer.prev_executed": "the next step's ledger: every lesson of that step and the stores it adds to "
                                 "(the command recorded beside it is dropped at the switch: that order, executed "
                                 "before it, stays out of the eye row's replay contract)",
    }
    # the stores that hold an eye row ENCODED, which no exact rewrite reaches (they age out)
    ENCODED_EYE_ROW_CARRIERS = {
        "context_buffer": "the live context's motor tokens: turn over in 128 steps",
        "replay.memory_states": "the replay ring's stored memory states, applied as replay's memory bias: "
                                "turn over with the ring (6,000 steps)",
        "persistent_memory.carry": "the memory's recurrent carry, which folds in every evicted event: an old "
                                   "row's share fades only through the recurrent update, one evicted event "
                                   "every 10 steps once the history is full",
        # the imagination chain's own copies (the deploy branch's third review, point G), carried by the episode
        # companion (continuation_state.DATA_NAMES), at most imagination steps + 1 (4) moments long
        "imagination.ctx_history": "the imagination chain's saved live contexts, whose motor tokens encoded the "
                                   "old rows: the chain's starting context, turns over in 4 steps",
        "imagination.memory_bias_history": "the imagination chain's saved memory biases, encoded from the "
                                           "memory's events before the switch: turns over in 4 steps",
    }

    def _memory_event_stores(self):
        """The persistent memory's stores of events, checked, or None without the memory: (history,
        [(record key, snapshot dict)], the eye row's two columns). Refuses (ValueError) an event whose width
        is not the memory's input width: its columns would not be the order's, and rewriting them would
        corrupt another block - checked before convert_stored_eye_rows rewrites anything."""
        mh = getattr(self, "memory_history", None)
        pm = getattr(self, "persistent_memory", None)
        if mh is None or pm is None:
            return None
        from .memory_integration import eye_row_columns
        width = int(pm.input_size)
        tp = getattr(self, "_tenure_pending", None)
        snaps = [(key, snap) for key, snap in (
            ("persistent_memory._nf_memory", getattr(self, "_nf_memory", None)),
            ("persistent_memory._tenure_pending", tp.get("memory") if isinstance(tp, dict) else None))
            if isinstance(snap, dict) and torch.is_tensor(snap.get("events"))]
        tensors = list(mh.events) + [getattr(mh, "previous", None), getattr(mh, "pending", None)]
        tensors += [snap["events"] for _, snap in snaps]
        for x in tensors:
            if torch.is_tensor(x) and x.shape[-1] != width:
                raise ValueError("a persistent memory event holds %d columns where the memory reads %d: its "
                                 "eye row columns are not known, so the switch cannot rewrite them"
                                 % (int(x.shape[-1]), width))
        return mh, snaps, eye_row_columns(self)

    def _convert_memory_eye_rows(self, stores) -> dict:
        """The persistent memory's stored events at an eye row convention switch (stores:
        _memory_event_stores): the eye row's x and y columns of every retained event, of the previous and
        the pending moment, and of the learner's two snapshots of the history (dicts whose events are
        [B, T, features]), each rewritten once and out of place, so a snapshot held under both names is
        not negated twice. The repair's running mean of events is left as saved (CENTRE_LEFT_AS_SAVED); its
        record goes to motor_interface_conversion["event_centre_eye_shift"]."""
        if stores is None:
            return {}
        mh, snaps, cols = stores

        def mirrored(x):
            out = x.clone()
            out[..., cols] = -out[..., cols]
            return out
        done = {}
        n = 0
        if mh.events:
            items = [mirrored(e) for e in mh.events]
            mh.events.clear()
            mh.events.extend(items)
            n += len(items)
        for name in ("previous", "pending"):
            v = getattr(mh, name, None)
            if torch.is_tensor(v):
                setattr(mh, name, mirrored(v))
                n += 1
        if n:
            done["persistent_memory.history"] = n
        seen = set()
        for key, snap in snaps:
            if id(snap) not in seen:
                seen.add(id(snap))
                snap["events"] = mirrored(snap["events"])
            done[key] = int(snap["events"].shape[1])
        # THE REPAIRED MEMORY'S RUNNING MEAN OF EVENTS (--persistent-memory-repair centred,
        # fix/persistent-memory-2026-09-25: the encoder reads every event minus event_centre, and the mean moves
        # 1/2,000 of the way toward every lived step's event). By the order centre's method (the coordinator's
        # decision after the four F1 follow-ups: one method for both centres), which since the fourth review's
        # fix round is to leave its eye columns as saved (CENTRE_LEFT_AS_SAVED): new events differ from it by the
        # switch's change to their average, and the rewritten stored events (mirrored whole) by twice their old
        # average, until they age out of the 64 s history. Absent (no repair, or a centre never set): nothing.
        pm = getattr(self, "persistent_memory", None)
        centre = getattr(pm, "event_centre", None)
        weight = getattr(pm, "event_centre_weight", None)
        if (torch.is_tensor(centre) and centre.shape[-1] == int(pm.input_size)
                and float(weight if weight is not None else 0.0) > 0.0):
            rec = self._centre_left_as_saved(centre, list(cols))
            conv = getattr(self, "motor_interface_conversion", None)
            if isinstance(conv, dict):
                conv["event_centre_eye_shift"] = rec
            log.warning("motor interface switch (eye row convention): the repaired memory's running mean of "
                        "events, eye columns %s: %s", rec["centre"], rec["left_as_saved"])
        return done

    def load_checkpoint(self, path: str, load_optimizer: bool = True,
                        reinit: Sequence[str] = ()) -> dict:
        """Load every module + optimizer from a snapshot folder (the model's own CheckpointManager
        format). Returns its result dict: step, loaded_components, reinitialized_components,
        curriculum_state (phase + phase_step) if the snapshot carries it."""
        # a checkpoint with no receipt was trained under the old defaults (legacy, off); a receipt
        # replaces this below. Never left from an earlier load of this object: the conversion of
        # the saved averages at the end of this load reads it. (Placed before receipt_path so that the
        # memory repair's own lines after _eff_fresh merge beside it without a conflict: the deploy
        # branch's third review, finding F.)
        self.motor_interface_change = self._motor_interface_changes({}) or None
        if not os.path.exists(os.path.join(path, 'model_receipt.json')) and self.motor_interface_change:
            # THE SWITCH'S TRIGGER IS THE RECEIPT'S STAMP (the deploy branch's second review, point 7). A
            # checkpoint without its receipt says nothing about the settings its weights and its stored
            # eye rows were trained under; read as the old defaults, a checkpoint trained under 'physical'
            # would have its stored rows mirrored BACK. The supervisor never resumes from one
            # (training_continuation.complete_candidates requires the receipt); a resume by hand could.
            raise ValueError(
                'checkpoint %s has no model_receipt.json, so the motor interface settings its weights and '
                'its stored eye rows were trained under are unknown; this launch asks for %s, which would be '
                'read as a switch from the old defaults (legacy, off) and would rewrite stored eye rows that '
                'may already be in the new convention. Resume from a complete checkpoint (it carries the '
                'receipt), or launch with --eye-row-convention legacy --executed-fields-only off'
                % (path, {k: v[1] for k, v in self.motor_interface_change.items()}))
        receipt_path = os.path.join(path, 'model_receipt.json')
        _eff_fresh = False       # the checkpoint's efference heads are another version
        _pm_saved_switch = None  # --persistent-memory-repair: the saved switch record, carried below
        _liveness_saved = None   # the checkpoint's liveness_repair stamp (its revived_at_step is carried)
        _mac_saved = None        # the checkpoint's motor_asked_change stamp (--motor-asked-change)
        _asp_saved = None        # the checkpoint's heads_all_slot_path stamp (--heads-all-slot-path)
        _acap_saved = None       # the checkpoint's attention_cap stamp (--attention-cap; None = trained without it)
        _vip_saved = None        # the checkpoint's vestibular_input_path stamp (--vestibular-input-path)
        if os.path.exists(receipt_path):
            with open(receipt_path, encoding='utf-8') as receipt_file:
                saved_layout = json.load(receipt_file).get('body_layout', 'legacy')
            if saved_layout != self.body_layout:
                raise ValueError(f'Checkpoint body layout {saved_layout} does not match {self.body_layout}; '
                                 'use the explicit trunk migration and matching --body-layout')
            with open(receipt_path, encoding='utf-8') as receipt_file:
                saved_receipt = json.load(receipt_file)
            if saved_receipt.get('persistent_memory') == 'on' and self.persistent_memory is None:
                raise ValueError('checkpoint has learned persistent memory; enable it instead of discarding it')
            # A SWITCHED MEMORY (--persistent-memory-repair centred; lem/model/persistent_memory.py): its
            # readout bias holds the folded constant and its readout weight reads the centred state. A build
            # without the repair would compute another memory bias; the only way back is the checkpoint
            # saved before the switch (the switch record names it).
            _pm_rep = saved_receipt.get('persistent_memory_repair') or {}
            _pm_saved_switch = _pm_rep.get('switch')
            if _pm_saved_switch and (self.persistent_memory is None or self.persistent_memory.repair == 'off'):
                raise ValueError(
                    'checkpoint has a switched persistent memory (--persistent-memory-repair centred, switched '
                    'at step %s); a build without the repair would compute another memory bias. Resume with '
                    '--persistent-memory-repair centred, or go back to the checkpoint saved before the switch '
                    '(%s), which discards everything learned since the switch. Code from before the repair '
                    'was merged does not refuse: it trains another memory and restarts every Adam moment of '
                    'the model.' % (_pm_saved_switch.get('switched_at_step'),
                                    _pm_saved_switch.get('pre_switch_checkpoint')
                                    or 'none recorded: the run that switched started without a checkpoint'))
            if _pm_saved_switch and _pm_rep.get('version') != self.receipt.get(
                    'persistent_memory_repair', {}).get('version'):
                raise ValueError('checkpoint\'s persistent memory was switched by repair version %s; this build '
                                 'is %s, whose running statistics differ. Go back to the checkpoint saved before '
                                 'the switch (%s)' % (_pm_rep.get('version'),
                                                      self.receipt['persistent_memory_repair'].get('version'),
                                                      _pm_saved_switch.get('pre_switch_checkpoint')))
            if saved_receipt.get('self_state','off') != 'off' and self.self_state is None:
                raise ValueError('checkpoint has learned self perception; enable it instead of discarding it')
            if saved_receipt.get('self_prediction','legacy') != 'legacy' and self.self_prediction_mode == 'legacy':
                raise ValueError('checkpoint has learned grounded self dynamics; retain --self-prediction grounded_residual')
            if saved_receipt.get('efference_heads') and getattr(self, 'efference', None) is None:
                raise ValueError('checkpoint has learned efference heads; resume with '
                                 '--efference-heads on instead of discarding what they learned')
            if saved_receipt.get('efference_heads') and getattr(self, 'efference', None) is not None:
                # another version starts its heads fresh after the load (lem.model.efference.
                # fresh_start); a GazeHead the first version retrained stays marked in every
                # later receipt, so no caller ever reads it as the eyes' direction
                from lem.model.efference import EFFERENCE_VERSION as _EV
                _eff_saved = saved_receipt['efference_heads']
                _eff_fresh = _eff_saved.get('version', _EV) != _EV
                # (a first-version stamp without the field reused the GazeHead whenever one existed)
                if _eff_saved.get('gaze_head_repurposed',
                                  _eff_saved.get('version') == 'efference_v1'):
                    self.receipt['efference_heads']['gaze_head_repurposed'] = True
            if saved_receipt.get('contact_head') and getattr(self, 'contact_head', None) is None:
                raise ValueError('checkpoint has a learned contact head; resume with --contact-head on '
                                 'instead of discarding what it learned')
            if saved_receipt.get('motor_synergy') and getattr(self, 'synergy', None) is None:
                raise ValueError('checkpoint has a learned motor synergy; resume with --motor-synergy on '
                                 'instead of discarding what it learned')
            _liveness_saved = saved_receipt.get('liveness_repair')
            if ((saved_receipt.get('liveness_repair') or {}).get('mode') == 'on'
                    and getattr(getattr(self, 'liveness', None), 'mode', 'off') != 'on'):
                raise ValueError('checkpoint has learned liveness repair parts (revived units that read '
                                 'their input through the repair, and the motor and touch pools\' '
                                 'all-element paths); resume with --liveness-repair on instead of '
                                 'discarding what they learned (add --liveness-rule none to keep them and '
                                 'revive no unit)')
            # --motor-asked-change: a checkpoint trained with the asked-change path computes its commanded motor
            # tokens through it; a build without the path would hand the core other tokens for every commanded
            # row, and a path of another version reads other features
            _mac_saved = saved_receipt.get('motor_asked_change')
            if _mac_saved and getattr(self, 'motor_asked_change', None) is None:
                raise ValueError('checkpoint has a learned asked-change path beside its motor encoder '
                                 '(--motor-asked-change on); resume with --motor-asked-change on instead of '
                                 'discarding what it learned: a build without it computes other motor tokens for '
                                 'every row that holds a command')
            if _mac_saved and getattr(self, 'motor_asked_change', None) is not None:
                from lem.model.motor_asked_change import ASKED_CHANGE_VERSION as _MAC_VERSION
                if _mac_saved.get('version') != _MAC_VERSION:
                    raise ValueError('checkpoint\'s asked-change path is %s; this build is %s, whose features or '
                                     'units differ, so the saved weights would read other inputs here'
                                     % (_mac_saved.get('version'), _MAC_VERSION))
            # --heads-all-slot-path: a checkpoint trained with the paths adds their output to every prediction; a
            # build without them would predict without that part
            _asp_saved = saved_receipt.get('heads_all_slot_path')
            if _asp_saved and getattr(self, 'heads_all_slot_path', None) is None:
                raise ValueError('checkpoint has learned all-slot paths beside its prediction heads '
                                 '(--heads-all-slot-path on); resume with --heads-all-slot-path on instead of '
                                 'discarding what they learned: a build without them computes other predictions')
            if _asp_saved and getattr(self, 'heads_all_slot_path', None) is not None:
                from lem.model.heads_all_slot_path import ALL_SLOT_PATH_VERSION as _ASP_VERSION
                if _asp_saved.get('version') != _ASP_VERSION:
                    raise ValueError('checkpoint\'s all-slot path is %s; this build is %s, whose inputs differ, '
                                     'so the saved weights would read other numbers here'
                                     % (_asp_saved.get('version'), _ASP_VERSION))
            _acap_saved = saved_receipt.get('attention_cap')
            # --vestibular-input-path: a checkpoint trained with the path computes its vestibular tokens through it;
            # a build without it would hand the core other tokens at every model step
            _vip_saved = saved_receipt.get('vestibular_input_path')
            if _vip_saved and getattr(self, 'vestibular_input_path', None) is None:
                raise ValueError('checkpoint has a learned input path beside its vestibular encoder '
                                 '(--vestibular-input-path on); resume with --vestibular-input-path on instead of '
                                 'discarding what it learned: a build without it computes other vestibular tokens')
            if _vip_saved and getattr(self, 'vestibular_input_path', None) is not None:
                from lem.model.vestibular_input_path import INPUT_PATH_VERSION as _VIP_VERSION
                if _vip_saved.get('version') != _VIP_VERSION:
                    raise ValueError('checkpoint\'s vestibular input path is %s; this build is %s, so the saved '
                                     'weight would be applied in another way here'
                                     % (_vip_saved.get('version'), _VIP_VERSION))
                # the output's cap is part of how the saved weight is applied: a weight trained under one cap gives
                # other tokens under another (a stamp saved before the cap existed carries none and is not judged)
                _vip_cap = _vip_saved.get('output_cap')
                if _vip_cap is not None and float(_vip_cap) != float(self.vestibular_input_path.cap):
                    raise ValueError('checkpoint\'s vestibular input path was trained with its output capped at %s '
                                     'token units; this build caps it at %s, so the saved weight would give other '
                                     'vestibular tokens here'
                                     % (_vip_cap, self.vestibular_input_path.cap))
            if saved_receipt.get('contact_head') and getattr(self, 'contact_head', None) is not None:
                from lem.model.contact_head import PATCH_NAMES as _CH_PATCHES
                _saved_patches = list(saved_receipt['contact_head'].get('patch_names') or [])
                if _saved_patches != list(_CH_PATCHES):
                    raise ValueError('checkpoint contact head was trained on another skin patch list '
                                     '(%d patches); its patch identities would mean other patches '
                                     'here' % len(_saved_patches))
                from lem.model.contact_head import CONTACT_HEAD_VERSION as _CH_VERSION
                _saved_ver = saved_receipt['contact_head'].get('version')
                if _saved_ver != _CH_VERSION:
                    raise ValueError('checkpoint contact head is %s; this build is %s, whose order '
                                     'input leaves out the position requests of handles the body '
                                     'only turns, so the saved order weights mean other inputs here'
                                     % (_saved_ver, _CH_VERSION))
            if saved_receipt.get('vestibular_sense') and not getattr(self, 'vestibular_sense', False):
                raise ValueError('checkpoint has a learned vestibular sense in its sixth input slot; '
                                 'resume with --vestibular-sense on instead of putting the '
                                 'intermind placeholder back over it')
            if getattr(self, 'vestibular_sense', False):
                from .vestibular_sense import stamp_mismatch, check_saved_layout
                _why = stamp_mismatch(saved_receipt.get('vestibular_sense'))
                if _why:
                    raise ValueError(_why)
                check_saved_layout(saved_receipt.get('active_modalities'), self.active)
            if saved_receipt.get('runtime_state_required') and not os.path.isfile(os.path.join(path,'runtime_state.pt')):
                raise ValueError('checkpoint is missing required runtime evidence; use the preceding complete checkpoint')
            # --proprio-heading: weights trained on the turned rows read world-frame rows wrongly, and the reverse
            # (mimo_bridge/proprio_heading.py). A build without the flag refuses a checkpoint trained with it; a
            # build with it on a checkpoint trained without it IS the switch, logged and stamped.
            _ph_saved = (saved_receipt.get('proprio_heading') or {}).get('mode', 'off')
            if _ph_saved != 'off' and getattr(self, 'proprio_heading', 'off') == 'off':
                raise ValueError('checkpoint was trained on heading-free proprioception (--proprio-heading %s); '
                                 'resume with --proprio-heading %s: this build would hand its encoder '
                                 'world-frame rows it never read' % (_ph_saved, _ph_saved))
            self.proprio_heading_switch = bool(_ph_saved == 'off'
                                               and getattr(self, 'proprio_heading', 'off') != 'off')
            if self.proprio_heading_switch:
                from .proprio_heading import switch_message as _ph_msg
                self.receipt['proprio_heading'] = dict(self.receipt['proprio_heading'], switched_at_load=str(path))
                log.warning("%s", _ph_msg(path))
            # --proprio-token-repair: loading a checkpoint trained without this version of the lesson into a build
            # with it IS the switch; the trainer writes it into the stats stream, where the hourly watcher's kill
            # lines start from it (docs/gain_of_function_v4/proprio_repair_watch.py). Nothing else changes here.
            self.proprio_token_repair_switch = None
            if getattr(self, 'proprio_within', None) is not None:
                _pt_saved = (saved_receipt.get('proprio_token_repair') or {}).get('version')
                _pt_stamp = self.proprio_within.stamp()
                if _pt_saved != _pt_stamp['version']:
                    self.proprio_token_repair_switch = dict(_pt_stamp, from_version=_pt_saved, checkpoint=str(path))
                    self.receipt['proprio_token_repair'] = dict(self.receipt['proprio_token_repair'],
                                                                switched_at_load=str(path))
                    log.warning("--proprio-token-repair within_body switched on at %s (the checkpoint was trained "
                                "with %s): the lesson starts when the window is full", path,
                                _pt_saved or "no within-body lesson")
            # P021: the proprioception row index used to reach the encoder raw, 0..25;
            # it now arrives divided by TokenizerConfig.proprio_index_range. Weights
            # trained under one scale read under the other move the proprioception
            # token by 0.4892 mean absolute (measured on H83's weights), so a mismatch
            # is refused by name. A checkpoint with no stamp predates the change.
            from tokenizer.config import TokenizerConfig as _TC
            _cur = float(_TC().proprio_index_range)
            _saved = float(saved_receipt.get('proprio_index_scale', 1.0))
            if _saved != _cur:
                raise ValueError(
                    f'checkpoint proprioception row-index scale is {_saved:g} but this code '
                    f'divides by {_cur:g}; its encoder would read a body sense on a different '
                    f'scale than it was trained on. Either start fresh, or migrate: multiply '
                    f'encoder_registry.encoders.proprioception.mlp.0.weight[:, 11::12] by '
                    f'{_cur/_saved:g} (the first layer is bias-free, so the two rescalings '
                    f'cancel to 1.5e-06) and scale its Adam moments by the inverse.')
            # THE MOTOR INTERFACE SETTINGS THE WEIGHTS WERE TRAINED UNDER (audit 2026-09-25, F1 and
            # F4). A change is allowed - it is how the deploy switches them on, and how a revert
            # switches them off - but never silent: after it, what the weights learned about the
            # eyes' order row (its pitch sign) or about the order's dead fields reads differently
            # until it is relearned. A checkpoint with no stamp was trained under the old defaults.
            _changes = self._motor_interface_changes(saved_receipt)
            self.motor_interface_change = _changes or None
            if _changes:
                # (the deploy branch's second review, point 7: this line said every stored replay moment
                # keeps the old form, which under F1 is false - the trainer rewrites the stored eye rows)
                log.warning('motor interface settings differ from the checkpoint (saved -> now): %s. '
                            'The eye row pitch sign (F1) or the order fields the decoder never reads '
                            '(F4) now mean something other than what the weights learned. At an F1 '
                            'switch the trainer rewrites every stored ledger eye row in the new '
                            'convention (train_mimo.switch_stored_eye_rows); under F4 the replay moments '
                            'stored before this resume keep their free dead fields until they age out '
                            '(6,000 steps) (docs/gain_of_function_v4/agents/motor_fixes_eye_fields_2026-09-25.md)',
                            _changes)
        # ATTENTION_CAP (--attention-cap): a build with the cap and one without hold the same parameters and buffers,
        # so a checkpoint loads either way (the operator's requirement of Oct 5, 2026). A difference between the cap the
        # checkpoint was saved under (its receipt's stamp; none = without the cap) and this build's is written down
        # (attention_cap_change; the trainer puts it into the stats stream) and logged, never refused.
        _acap_now = self.receipt.get('attention_cap')
        self.attention_cap_change = (None if _acap_saved == _acap_now else
                                     dict(saved=_acap_saved, now=_acap_now, checkpoint=str(path)))
        if self.attention_cap_change is not None:
            log.warning('attention cap differs from the checkpoint (saved -> now): bound %s -> %s at %s. The '
                        'transformer\'s cross-modal attention and the motor encoder\'s attention now score their keys '
                        'by another rule than the one these weights were last trained under '
                        '(lem/model/attention_cap.py)', (_acap_saved or {}).get('bound'),
                        (_acap_now or {}).get('bound'), path)
        vest_switch = False
        weights_path = os.path.join(path,'model.safetensors')
        if os.path.exists(weights_path):
            from safetensors import safe_open
            with safe_open(weights_path,framework='pt',device='cpu') as sf:
                key='forward_model.temporal_embedding.weight'
                if key in sf.keys() and sf.get_slice(key).get_shape()[0] != self.ctx_len:
                    raise ValueError('checkpoint attention length differs; use the explicit context migration')
                # The context-length guard above existed; these two did not. Measured
                # 2026-09-16: a checkpoint saved with 2 core blocks loaded into a model
                # built with 8 gave 150 missing keys, NO exception, and six blocks kept
                # their fresh random weights - the run would have trained a different
                # brain with one warning line as the only trace. Reversed, six trained
                # blocks were silently dropped. A width change is caught earlier by the
                # receipt's d_output check for the ENCODERS, but nothing checked the
                # core, whose 52 of 54 tensors would then be re-randomised with a log
                # warning. Refuse both, by name, before a tensor is read.
                problem = core_shape_mismatch(
                    sf.keys(), lambda k: tuple(sf.get_slice(k).get_shape()),
                    n_blocks=len(self.forward_model.blocks), d_model=self.d_model)
                if problem:
                    raise ValueError(problem)
                # THE VESTIBULAR SLOT, read from the tensors themselves so a checkpoint with no
                # receipt is judged too: a learned vestibular encoder is never loaded into the
                # intermind build, and an intermind checkpoint loaded into the vestibular build
                # is moved by name (vestibular_sense.migrate_checkpoint) below.
                _keys = list(sf.keys())
                if (not getattr(self, 'vestibular_sense', False)
                        and any(k.startswith('encoder_registry.encoders.vestibular.') for k in _keys)):
                    raise ValueError('checkpoint has a learned vestibular encoder; resume with '
                                     '--vestibular-sense on')
                if getattr(self, 'vestibular_sense', False):
                    from .vestibular_sense import switch_needed
                    vest_switch = switch_needed(_keys, self.active)
        # a held goal from before the load belongs to a brain that no longer exists,
        # and a mid-process load can move the step counter backwards (a negative age
        # would never expire the goal)
        self._gl_clear()
        mgr = self.checkpoints or CheckpointManager(self.config, os.path.dirname(os.path.abspath(path)))
        # load_optimizer=False for architecture-changing resumes (fixed -> adaptive motor
        # controller): the param groups differ, and a stale state that happens to match in
        # count would defer a shape crash to the first optimizer.step()
        # THE TOUCH SENSE GREW TWO COLUMNS (2026-09-21): self-touch share and how many own
        # parts are on a contact. Only one saved tensor depends on the width - the tactile
        # encoder's first projection - and the loader would otherwise report that encoder
        # reinitialised and throw its learning away. Copy what was learned about the eleven
        # older columns and start the two new ones at exactly zero, so the first step after
        # the resume produces the same tokens as the last step before it.
        # an intermind -> vestibular switch loads the optimizer itself, by parameter name: the
        # loader's by-position load would pair saved moments with the wrong parameters
        if _eff_fresh and vest_switch:
            raise ValueError('this checkpoint needs two optimizer migrations at once (older efference '
                             'heads and the intermind -> vestibular switch); resume it once with '
                             '--vestibular-sense as it was saved, then switch')
        # --motor-input-fold: a folded checkpoint refuses a build without the fold; with it, the fold's saved state
        # comes only from the checkpoint's own side module (lem/model/motor_input_fold.py, A RESUME)
        from lem.model.motor_input_fold import before_load as _mif_before_load
        _mif_before_load(self, path)
        # older efference heads: their fresh start loads the optimizer itself (every other Adam
        # moment carried around the replaced block), like the vestibular switch below
        res = mgr.load(path, self.encoder_registry, self.forward_model, self.prediction_heads,
                       self.motor_controller, self.wellness,
                       optimizer=(self.optimizer if (load_optimizer and not vest_switch
                                                     and not _eff_fresh) else None),
                       gaze_head=self.gaze_head, voice_head=self.voice_head, map_location=self.device,
                       side_modules=self._side_modules(),
                       skip_encoders=[r.split(':', 1)[1] for r in reinit if r.startswith('encoder:')],
                       allow_retrained_gaze_head=True)
        if _eff_fresh:
            from lem.model.efference import fresh_start
            self.efference_load = fresh_start(self, path, load_optimizer, log=log)
            if "efference_heads" not in res.setdefault("reinitialized_components", []):
                res["reinitialized_components"].append("efference_heads")
            if "efference_heads" in (res.get("loaded_components") or []):
                res["loaded_components"].remove("efference_heads")
        self.vestibular_switch = None
        if vest_switch:
            from .vestibular_sense import migrate_checkpoint, load_message
            self.vestibular_switch = migrate_checkpoint(self, path, load_optimizer=load_optimizer)
            # worded for the load that read no optimizer too (review b2-vestibular finding 11)
            log.warning("%s", load_message(self.vestibular_switch))
        # the touch token grew two columns on 2026-09-21; put the tactile encoder's
        # learning back rather than letting a width change throw it away
        self._restore_widened_touch_encoder(path, res)
        self._migrate_touch_push_scale(path, load_optimizer)
        self._migrate_efference_scales(path, load_optimizer)
        self._migrate_contact_order_scales(path, load_optimizer)
        self._migrate_synergy_scales(path, load_optimizer)
        self.step_count = int(res.get("step", 0))
        self.loaded_from = os.path.abspath(path)
        if self.persistent_memory is not None and self.persistent_memory.repaired:
            # the switch stays named in every later receipt (and the trainer's launch record)
            self.persistent_memory_switch = _pm_saved_switch or {
                'note': 'the memory is switched (its repair_switched buffer) but the receipt holds no switch record'}
            self.receipt['persistent_memory_repair'] = dict(self.receipt.get('persistent_memory_repair') or {},
                                                            switch=self.persistent_memory_switch)
        # review M16b + items 3/5: restore the working statistics saved beside the
        # checkpoint (absent in older snapshots: cold start, exactly the old behavior)
        _rt_pt = os.path.join(path, "runtime_state.pt")
        _rt_js = os.path.join(path, "runtime_state.json")
        rt = None
        _eff_centres_restored = False     # set when THIS load restored the heads' running means
        try:
            if os.path.isfile(_rt_pt):
                rt = torch.load(_rt_pt, weights_only=False, map_location="cpu")
            elif os.path.isfile(_rt_js):
                with open(_rt_js) as f:
                    rt = json.load(f)
        except (OSError, ValueError, TypeError, RuntimeError,EOFError,pickle.UnpicklingError) as e:
            raise RuntimeError(f"checkpoint runtime state is corrupt; use the preceding complete checkpoint: {path}") from e
        if rt is not None:
            if rt.get('self_rehearsal') is not None:
                if self.self_rehearsal is None:raise ValueError('checkpoint has retained self experience')
                self.self_rehearsal.load_state_dict(rt['self_rehearsal'])
            try:
                self._vis_adv_ema = float(rt.get("vis_adv_ema", 0.0))
                self._vis_adv_elig_ema = float(rt.get("vis_adv_elig_ema", 0.0))
                # key versioned: _tmis = temporal-mispairing null (2026-09-09). Snapshots
                # from the broken same-event null used "vis_adv_null_ema"; that value
                # was minus-the-signal and must NOT seed the repaired gauge.
                self._vis_adv_null_ema = float(rt.get("vis_adv_null_tmis_ema", 0.0))
                self._touch_adv_ema = float(rt.get("touch_adv_ema", 0.0))
                self._hand_motion_ema = float(rt.get("hand_motion_ema", 1e-3))
                _pw = rt.get("plan_win_ema")
                if _pw is not None:
                    _pw = [float(w) for w in _pw]
                    if len(_pw) == self.imagination_steps:
                        self._plan_win_ema = _pw
                    else:
                        # a resume that changed the chain depth: the earned overlap
                        # migrates, new depths open at 0 (same doctrine as the
                        # hierarchical-self slot migration - earned evidence is never
                        # discarded for a config change). Before this, ANY depth change
                        # cold-started all six EMAs: ~69 consecutive wins per depth to
                        # re-cross 0.5, paid for nothing.
                        _mig = _pw[:self.imagination_steps]
                        _mig += [0.0] * (self.imagination_steps - len(_mig))
                        self._plan_win_ema = _mig
                        log.warning(
                            "plan license: migrated win EMAs across chain-depth change "
                            "(%d -> %d depths, %d earned kept)",
                            len(_pw), self.imagination_steps,
                            min(len(_pw), self.imagination_steps))
                _pe = rt.get("agency_pay_ema")
                self._agency_pay_ema = (
                    None if _pe is None
                    else (_pe.to(self.device) if torch.is_tensor(_pe)
                          else torch.tensor(_pe, dtype=torch.float32, device=self.device)))
                _lr = rt.get("latreg")
                if _lr and self.latent_regularizer is not None \
                        and _lr["running_mean"].shape == self.latent_regularizer.running_mean.shape:
                    self.latent_regularizer.running_mean = _lr["running_mean"].to(self.device)
                    self.latent_regularizer.running_var = _lr["running_var"].to(self.device)
                    self.latent_regularizer._initialized = bool(_lr.get("initialized", True))
                _vh = rt.get("vicreg_history")
                if _vh and hasattr(self.loss_manager, "_vicreg_history"):
                    from collections import deque as _dq
                    _w = int(getattr(self.loss_manager, "_vicreg_temporal_window", 0)) or None
                    self.loss_manager._vicreg_history = {
                        m: _dq((t.to(self.device) for t in ts), maxlen=_w)
                        for m, ts in _vh.items()}
                _bs = rt.get("body_schemas")
                if _bs and self.body_schema_updates:
                    while len(self._schemas) < len(_bs):
                        self._schemas.append(DifferentiableBodySchema(
                            n_handles=len(self.handle_names), object_names=SELF_OBJECTS).to(self.device))
                    for sc, sd in zip(self._schemas, _bs):
                        sc.load_state_dict({k: v.to(self.device) for k, v in sd.items()},
                                           strict=False)
                _hs = rt.get("hself")
                if _hs and self._hself is not None:
                    self._hself.load_state_dict(
                        {k: v.to(self.device) for k, v in _hs.items()}, strict=False)
                # the entity levels: the tracker is built at the first step, when the body
                # count is known, so its saved state waits here until then
                self._entity_self_saved = rt.get("entity_self")
                self._entity_self_saved_names = rt.get("entity_self_names")
                _sv = rt.get("self_value")
                if _sv and self._self_value is not None:
                    try:
                        self._self_value.load_state_dict(
                            {k: v.to(self.device) for k, v in _sv.items()}, strict=False)
                        # a resume never inherits an open gate: the reader must show again,
                        # on the moments after the resume, that it beats its own control
                        self._self_value.gate_open.fill_(False)
                        self._self_value.gate_streak.zero_()
                        self._self_value.cmp_n.zero_()
                        self._self_value.cmp_sums.zero_()
                    except (RuntimeError, ValueError) as exc:
                        log.warning("self value reader: saved state not loaded (%s); it starts "
                                    "with its gate shut", exc)
                if rt.get("nonfinite_steps") is not None:
                    # saved since M16b but never restored: the lifetime skipped-step
                    # count reset every resume (review 2026-09-06)
                    self._nonfinite_steps = int(rt["nonfinite_steps"])
                _wcs = rt.get("wellness_channels")
                if getattr(self, "wellness_channels", None) is not None:
                    # a checkpoint saved before these channels existed starts them cold
                    _got = self.wellness_channels.load_state_dict(_wcs) if _wcs else []
                    log.info("wellness channels: restored %s from the checkpoint%s",
                             ", ".join(_got) or "nothing",
                             "" if _got else " (none saved: they start cold)")
                _chs = rt.get("contact_head")
                if getattr(self, "contact_head", None) is not None:
                    # a checkpoint saved before the head existed starts it cold: the centre from
                    # the first lived moment, the base rates after 100 moments per patch
                    _cm = (_chs or {}).get("centre_mean")
                    if torch.is_tensor(_cm) and tuple(_cm.shape) == tuple(self.contact_head.centre_mean.shape):
                        self.contact_head.centre_mean.copy_(_cm.to(self.contact_head.centre_mean))
                        self.contact_head.centre_weight = float(_chs.get("centre_weight", 0.0))
                    _trk = self.contact_tracker.load_state_dict((_chs or {}).get("tracker"))
                    log.info("contact head: running statistics %s; both evidence gates start shut "
                             "and must be proved again on readings taken after this load",
                             "restored" if _trk else "start cold (none saved for these patches)")
                _sys = rt.get("motor_synergy")
                if getattr(self, "synergy", None) is not None:
                    # a checkpoint saved before the synergy existed starts its pose centre cold (the
                    # first lived moment sets it) and babble's sigma from 1 (recalibrated on the
                    # first babble moment)
                    _cm = (_sys or {}).get("centre_mean")
                    if torch.is_tensor(_cm) and tuple(_cm.shape) == tuple(self.synergy.centre_mean.shape):
                        self.synergy.centre_mean.copy_(_cm.to(self.synergy.centre_mean))
                        self.synergy.centre_weight = float(_sys.get("centre_weight", 0.0))
                    if (_sys or {}).get("babble_sigma") is not None:
                        self._synergy_sigma = float(_sys["babble_sigma"])
                    log.info("motor synergy: running statistics %s",
                             "restored" if torch.is_tensor(_cm) else "start cold (none saved)")
                if getattr(self, "nf_norm", None) is not None:
                    # a checkpoint saved before the normalisation existed starts it cold: each sense's
                    # divisor is its first real reading until the running average builds up
                    _nfs = self.nf_norm.load_state_dict(rt.get("next_frame_norm"))
                    log.info("next-frame normalisation: running persistence errors %s",
                             ("restored for " + ", ".join(_nfs)) if _nfs
                             else "start cold (none saved)")
                if getattr(self, "liveness", None) is not None and self.liveness.mode == "on":
                    # the revived units' input centres (lem/model/liveness.py); a checkpoint saved
                    # before the repair starts them cold (the first trained call sets each)
                    _lvg = self.liveness.load_runtime_state(rt.get("liveness_repair"))
                    log.info("liveness repair: input centres %s",
                             ("restored for " + ", ".join(_lvg)) if _lvg else "start cold (none saved)")
                if getattr(self, "efference", None) is not None and not _eff_fresh:
                    # the efference heads' running means (the latent's and the order's centres);
                    # a checkpoint saved before them starts cold: the first lived moment sets them
                    _efs = rt.get("efference")
                    _eff_centres_restored = bool(self.efference.load_runtime_state(_efs))
                    log.info("efference heads: running means %s",
                             "restored" if _eff_centres_restored
                             else "start cold (none saved)")
                if getattr(self, "vestibular_norm", None) is not None:
                    # the vestibular input's running mean and spread (a checkpoint saved before
                    # the sense, or before the normaliser, starts them cold)
                    log.info("vestibular sense: running input mean and spread %s",
                             self.vestibular_norm.load_state_dict(rt.get("vestibular_norm")))
                if getattr(self, "proprio_within", None) is not None:
                    # --proprio-token-repair: the window of raw proprioceptive inputs, the start and the
                    # attributed lift (a checkpoint saved without the flag starts the window empty: the lesson waits
                    # for its 1,200 samples, 6,000 model steps)
                    log.info("proprioception token repair: the window %s",
                             self.proprio_within.load_state_dict(rt.get("proprio_token_repair"), self.device))
                log.info("runtime state restored: vis_adv %.5f touch_adv %.5f latreg %s "
                         "schemas %s", self._vis_adv_ema, self._touch_adv_ema,
                         "yes" if rt.get("latreg") else "no",
                         len(rt.get("body_schemas") or []))
            except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                # Continuing here would keep only the fields preceding the failed
                # one and cold-start the rest of the saved learning state. Explicit
                # compatible migrations above remain allowed; a failed load does not.
                raise RuntimeError(
                    f"checkpoint runtime state could not be fully restored; "
                    f"discard this model instance and use a complete compatible checkpoint: {path}"
                ) from e
        # AFTER every saved average is restored (the efference heads' order centre from
        # runtime_state.pt above, the grounded self's command calibration from the weights): the
        # averages over the order's fields are converted at a change of --eye-row-convention or
        # --executed-fields-only, where the conversion is exact (motor fixes review 2026-09-25,
        # finding 2)
        self._convert_order_averages(
            efference_restored=_eff_centres_restored,
            self_state_loaded="self_state" in (res.get("loaded_components") or []))
        # LEARNING PROGRESS sits out for every network this load started fresh (review of b2,
        # finding 5): its fall from a random start is not progress (lem/model/wellness_channels.py
        # LP_HOLDOUT_STEPS)
        self._lp_hold_out_fresh(res, reinit)
        # --vestibular-encoder-repair: the load-time switch (after the weights, the optimizer and the step
        # count are in place; nothing restored above depends on the divided layer), then the restart of this
        # sense's running averages, which the runtime state restored just above from the dead token. Placed
        # here, not beside the step count, where the persistent-memory repair adds its own lines.
        self._vestibular_encoder_repair_on_load(path, load_optimizer)
        self._vestibular_encoder_repair_after_runtime()
        # P203: refresh the grounded self dynamics' command calibration from the resumed
        # run's own stored executed orders, rescaling the command encoder so every
        # predicted value is unchanged (self_state.py recalibrate_actions). Opt-in flag.
        if getattr(self,'refresh_action_calibration',False) and self.self_state is not None                 and self.self_state.prediction_mode=='grounded_residual':
            stored=list(self.self_history.events) if self.self_history is not None else []
            if len(stored)<2:
                raise ValueError('action calibration refresh needs the resumed run stored executed orders')
            orders=torch.cat([e['action'].float().reshape(-1,self.self_state.action_size)
                              for e in stored],0).to(self.device)
            record=self.self_state.grounded.recalibrate_actions(orders)
            # Adam's moments live in the OLD parameterisation. dL/dW'[:,i] = (s_i/s'_i)
            # dL/dW[:,i], so carry the moments across by the same factor or the first
            # steps after the refresh are taken in the units the weights just left.
            factor=record.pop('scale_factor').to(self.device)
            param=self.self_state.grounded.command.weight
            state=self.optimizer.state.get(param)
            if state:
                n=self.self_state.action_size
                for key,power in (('exp_avg',1),('exp_avg_sq',2)):
                    if key in state:
                        state[key][:,:n]=state[key][:,:n]/factor.pow(power)
            import datetime as _dt
            self.receipt['self_action_calibration']=dict(record,at_step=self.step_count,
                source="the resumed run's own stored executed orders",
                measured_at=_dt.datetime.now().astimezone().isoformat())
            log.warning('self action calibration refreshed on %d stored executed orders: '
                        'normalised command absolute mean %.4f -> %.4f',
                        record['samples'],record['normalised_absmean_before'],
                        record['normalised_absmean_after'])
        # A reinitialised encoder must not inherit the optimizer moments of the weights it
        # replaced: Adam would take its first steps in the old weights' directions.
        for r in reinit:
            if r.startswith("encoder:"):
                _n = r.split(":", 1)[1]
                enc = self.encoder_registry.encoders[_n] if _n in self.encoder_registry.encoders else None
                for p in (enc.parameters() if enc is not None else ()):
                    self.optimizer.state.pop(p, None)
        # AUDIT FIX: a Stage-3-modulated lr saved into the snapshot must not become the
        # new base (it ratcheted across resumes); restore the config lr after every load
        if self._base_lrs is not None:
            for gi, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self._base_lrs[gi]
        if getattr(self, "motor_asked_change", None) is not None:
            # --motor-asked-change on a checkpoint trained without the path IS the switch. Its check, on probe rows:
            # the motor token with the path against without it must be the same number (the output layer is zero),
            # in float32 and under the run's autocast; the trainer writes the record into the stats stream. Before
            # the liveness windows restart below, so the probe rows join no watching window.
            from lem.model import motor_asked_change as _MAC
            self.motor_asked_change_switch = None
            _mac_stamp = self.motor_asked_change.stamp()
            if _mac_saved:
                # a later resume: the switch stays named in every receipt
                if _mac_saved.get('switch'):
                    _mac_stamp['switch'] = _mac_saved['switch']
            else:
                _mac_check = _MAC.switch_check(self)
                if not _mac_check["exact"]:
                    raise RuntimeError('--motor-asked-change: the checkpoint carries no stamp of the asked-change '
                                       'path, yet the path changes the motor token on the probe rows (%s): its '
                                       'output layer is not zero, so the switch would not keep the tokens'
                                       % ({k: v for k, v in _mac_check.items() if isinstance(v, dict)},))
                self.motor_asked_change_switch = dict(_mac_check, version=_MAC.ASKED_CHANGE_VERSION,
                                                      checkpoint=str(path), at_step=int(self.step_count))
                _mac_stamp['switch'] = {'at_step': int(self.step_count), 'checkpoint': str(path)}
                log.warning("--motor-asked-change switched on at %s (step %d): the path's output layer starts at "
                            "zero; the motor token on %d probe rows is unchanged (%s)", path, self.step_count,
                            _mac_check["rows"], ", ".join(k for k in ("float32", "autocast") if k in _mac_check))
            self.receipt['motor_asked_change'] = _mac_stamp
        if getattr(self, "heads_all_slot_path", None) is not None:
            # --heads-all-slot-path on a checkpoint trained without the paths IS the switch. Its check, on probe
            # rows: every head's prediction with the paths against without them must be the same number (the output
            # layers are zero), in float32 and under the run's autocast; the trainer writes the record into the
            # stats stream. The check runs inside the liveness repair's unwatched block and moves no centre.
            from lem.model import heads_all_slot_path as _ASP
            self.heads_all_slot_path_switch = None
            _asp_stamp = self.heads_all_slot_path.stamp()
            if _asp_saved:
                # a later resume: the switch stays named in every receipt
                if _asp_saved.get('switch'):
                    _asp_stamp['switch'] = _asp_saved['switch']
            else:
                _asp_check = _ASP.switch_check(self)
                if not _asp_check["exact"]:
                    raise RuntimeError('--heads-all-slot-path: the checkpoint carries no stamp of the all-slot '
                                       'paths, yet they change the predictions on the probe rows (%s): an output '
                                       'layer is not zero, so the switch would not keep the predictions'
                                       % ({k: v for k, v in _asp_check.items() if isinstance(v, dict)},))
                self.heads_all_slot_path_switch = dict(_asp_check, version=_ASP.ALL_SLOT_PATH_VERSION,
                                                       checkpoint=str(path), at_step=int(self.step_count))
                _asp_stamp['switch'] = {'at_step': int(self.step_count), 'checkpoint': str(path)}
                log.warning("--heads-all-slot-path switched on at %s (step %d): every path's output layer starts "
                            "at zero; the predictions on %d probe rows are unchanged (%s)", path, self.step_count,
                            _asp_check["rows"], ", ".join(k for k in ("float32", "autocast") if k in _asp_check))
            self.receipt['heads_all_slot_path'] = _asp_stamp
        if getattr(self, "vestibular_input_path", None) is not None:
            # --vestibular-input-path on a checkpoint trained without the path IS the switch. Its check, on probe
            # rows: the vestibular token with the path against without it must be the same number (the weight is
            # zero), in float32 and under the run's autocast; the trainer writes the record into the stats stream
            from lem.model import vestibular_input_path as _VIP
            self.vestibular_input_path_switch = None
            _vip_stamp = self.vestibular_input_path.stamp()
            if _vip_saved:
                # a later resume: the switch stays named in every receipt
                if _vip_saved.get('switch'):
                    _vip_stamp['switch'] = _vip_saved['switch']
                # the wiring may change between two resumes (the two values compute the same tokens from the same
                # weight and differ only in what the delayed lesson trains); the change is logged and stamped
                _vip_was = _vip_saved.get('mode')
                if _vip_saved.get('wiring_changes'):
                    _vip_stamp['wiring_changes'] = list(_vip_saved['wiring_changes'])
                if _vip_was and _vip_was != _vip_stamp['mode']:
                    _vip_stamp['wiring_changes'] = list(_vip_stamp.get('wiring_changes') or []) + [
                        {'at_step': int(self.step_count), 'from': _vip_was, 'to': _vip_stamp['mode']}]
                    log.warning("--vestibular-input-path: the checkpoint was trained with the wiring %s; this run "
                                "continues with %s from step %d (the same path and weight; the delayed lesson's "
                                "gradient differs)", _vip_was, _vip_stamp['mode'], self.step_count)
            else:
                _vip_check = _VIP.switch_check(self)
                if not _vip_check["exact"]:
                    raise RuntimeError('--vestibular-input-path: the checkpoint carries no stamp of the input path, '
                                       'yet the path changes the vestibular token on the probe rows (%s): its weight '
                                       'is not zero, so the switch would not keep the tokens'
                                       % ({k: v for k, v in _vip_check.items() if isinstance(v, dict)},))
                self.vestibular_input_path_switch = dict(_vip_check, version=_VIP.INPUT_PATH_VERSION,
                                                         checkpoint=str(path), at_step=int(self.step_count))
                _vip_stamp['switch'] = {'at_step': int(self.step_count), 'checkpoint': str(path)}
                log.warning("--vestibular-input-path switched on at %s (step %d): the path's weight starts at zero; "
                            "the vestibular token on %d probe rows is unchanged (%s)", path, self.step_count,
                            _vip_check["rows"], ", ".join(k for k in ("float32", "autocast") if k in _vip_check))
            self.receipt['vestibular_input_path'] = _vip_stamp
        if getattr(self, "liveness", None) is not None:
            # which units the loaded checkpoint had revived (and at which step, from its receipt); the watching
            # windows restart
            self.liveness.after_load(self, saved=_liveness_saved)
        self._restore_visual_input_conditioning(path, load_optimizer)
        self._restore_visual_activation_calibration(path)
        self._restore_hand_response_calibration(path)
        log.info("resumed from %s at step %d (loaded %s, reinitialized %s)", path, self.step_count,
                 res.get("loaded_components"), res.get("reinitialized_components"))
        # A shape clash inside a component makes CheckpointManager reinitialize the WHOLE
        # component, not the clashing tensor. For vision that means a checkpoint saved
        # before the eye/span positional inputs existed would quietly throw away the entire
        # trained vision encoder and keep going, which reads in the log as one word. Refuse
        # instead, and say the command that fixes it.
        if "encoder:vision" in (res.get("reinitialized_components") or []):
            raise RuntimeError(
                f"the vision encoder was NOT loaded from {path} - every weight in it is "
                f"random now. This snapshot predates the binocular positional inputs "
                f"(azimuth, elevation, which eye, patch span). Widen it first:\n"
                f"    python -m mimo_bridge.tools.grow_vision {path} --inputs 4\n"
                f"and resume from the folder that writes.")
        return res

    def save_checkpoint(self, step: int, curriculum_state: Optional[dict] = None, companion_writer=None) -> Optional[str]:
        if self.checkpoints is None:
            return None
        # review M16b + deferred items 3/5 (2026-09-02): the displacement fade is
        # documented "no clocks" - with the EMAs reset at every resume it re-armed
        # the full bootstrap every segment (measured: H60 end prior_w 0.0 -> H61
        # first record 0.05). The latent regularizer's reference statistics and the
        # vicreg history are body-independent and re-estimated for nothing each
        # segment; the body schema's EMA evidence silently reset the day the
        # feature is re-enabled. All of it rides runtime_state.pt beside the
        # checkpoint; a missing file just means a cold start.
        def write_companions(path,final_path):
            try:
                rt = {"vis_adv_ema": float(self._vis_adv_ema),
                      "vis_adv_elig_ema": float(self._vis_adv_elig_ema),
                      "vis_adv_null_tmis_ema": float(self._vis_adv_null_ema),
                      "touch_adv_ema": float(self._touch_adv_ema),
                      "hand_motion_ema": float(self._hand_motion_ema),
                      "plan_win_ema": (None if self._plan_win_ema is None
                                       else [float(w) for w in self._plan_win_ema]),
                      "agency_pay_ema": (None if self._agency_pay_ema is None
                                         else self._agency_pay_ema.detach().cpu()),
                      "nonfinite_steps": int(self._nonfinite_steps)}
                if self.latent_regularizer is not None:
                    rt["latreg"] = {
                        "running_mean": self.latent_regularizer.running_mean.detach().cpu(),
                        "running_var": self.latent_regularizer.running_var.detach().cpu(),
                        "initialized": bool(self.latent_regularizer._initialized)}
                _vh = getattr(self.loss_manager, "_vicreg_history", None)
                if _vh:
                    rt["vicreg_history"] = {m: [t.detach().cpu() for t in dq]
                                            for m, dq in _vh.items()}
                if self.body_schema_updates and self._schemas:
                    rt["body_schemas"] = [
                        {k: v.detach().cpu() for k, v in sc.state_dict().items()}
                        for sc in self._schemas]
                if self._hself is not None:
                    rt["hself"] = {k: v.detach().cpu()
                                   for k, v in self._hself.state_dict().items()}
                if self._entity_self is not None:
                    rt["entity_self"] = {k: v.detach().cpu()
                                         for k, v in self._entity_self.state_dict().items()}
                    rt["entity_self_names"] = list(self._entity_self.entity_names)
                    rt["entity_self_teacher"] = self._entity_self.teacher
                if self._self_value is not None:
                    rt["self_value"] = {k: v.detach().cpu()
                                        for k, v in self._self_value.state_dict().items()}
                if getattr(self, "wellness_channels", None) is not None:
                    # the normaliser's centres and scales by channel name, controllability's and
                    # learning progress's running averages (lem/model/wellness_channels.py)
                    rt["wellness_channels"] = self.wellness_channels.state_dict()
                if getattr(self, "contact_head", None) is not None:
                    # the contact head's running statistics (lem/model/contact_head.py): the
                    # latent's centre and the patches' base rates
                    rt["contact_head"] = {
                        "centre_mean": self.contact_head.centre_mean.detach().cpu(),
                        "centre_weight": float(self.contact_head.centre_weight),
                        "tracker": self.contact_tracker.state_dict()}
                if getattr(self, "efference", None) is not None:
                    # the efference heads' running means (lem/model/efference.py): the latent's and
                    # the order's centres over lived training moments
                    rt["efference"] = self.efference.runtime_state()
                if getattr(self, "synergy", None) is not None:
                    # the motor synergy's running statistics (lem/model/motor_synergy.py): the pose
                    # input's centre, and babble's current sigma
                    rt["motor_synergy"] = {
                        "centre_mean": self.synergy.centre_mean.detach().cpu(),
                        "centre_weight": float(self.synergy.centre_weight),
                        "babble_sigma": (None if self._synergy_sigma is None
                                         else float(self._synergy_sigma))}
                if getattr(self, "nf_norm", None) is not None:
                    # each sense's running persistence error, by name (lem/model/next_frame_norm.py)
                    rt["next_frame_norm"] = self.nf_norm.state_dict()
                if getattr(self, "liveness", None) is not None and self.liveness.mode == "on":
                    # the revived units' input centres (lem/model/liveness.py)
                    rt["liveness_repair"] = self.liveness.runtime_state()
                if getattr(self, "vestibular_norm", None) is not None:
                    # the vestibular input's running mean and spread (mimo_bridge/
                    # vestibular_sense.VestibularNormaliser)
                    rt["vestibular_norm"] = self.vestibular_norm.state_dict()
                if getattr(self, "proprio_within", None) is not None:
                    # --proprio-token-repair: the window of raw proprioceptive inputs per body
                    rt["proprio_token_repair"] = self.proprio_within.state_dict()
                if self.memory_history is not None:
                    # Restore only WITH the corresponding physical world. A reset
                    # world must not inherit a previous episode's transition state.
                    rt['persistent_memory_history'] = self.memory_history.state_dict()
                if self.self_history is not None:
                    rt['self_history'] = self.self_history.state_dict()
                    rt['self_rehearsal'] = self.self_rehearsal.state_dict()
                from lem.training.checkpoint import require_save_space
                require_save_space(path,rt)
                torch.save(rt, os.path.join(str(path), "runtime_state.pt"))
                # released before the episode companion is written: rt held CPU copies of the memory and
                # self histories (runtime_state.pt was 488 MB at step 447,350) through the whole companion
                # write until 2026-10-01 (docs/gain_of_function_v4/agents/save_memory_2026-10-01.md)
                del rt
            except OSError as e:
                raise RuntimeError('runtime state was not saved; checkpoint remains unpublished') from e
            if companion_writer is not None:companion_writer(path,final_path)
        path = self.checkpoints.save(step, self.encoder_registry, self.forward_model, self.prediction_heads,
                                     self.motor_controller, self.wellness, self.optimizer, self.receipt,
                                     gaze_head=self.gaze_head, voice_head=self.voice_head,
                                     curriculum_state=curriculum_state,
                                     side_modules=self._side_modules(),companion_writer=write_companions)
        # CheckpointManager writes model.safetensors (every module: encoders, forward model,
        # prediction heads, motor controller, wellness, gaze head, voice head) + optimizer.pt
        # + meta.json. That IS the full snapshot of the model; nothing else is added (disk
        # on this laptop is tight: ~27 GB free; a lite snapshot is ~37 MB, a 768-wide one ~0.6 GB).
        return path
