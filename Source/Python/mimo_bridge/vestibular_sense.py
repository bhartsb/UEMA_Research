# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The vestibular sense in the model's sixth input slot (docs/gain_of_function_v3/DESIGN.md D4).

WHAT CHANGES, in plain words. The model reads six input tokens every moment. The sixth
("intermind") was a placeholder for a future social sense. Its encoder was fed fresh Gaussian
noise every moment, but it had LEARNED TO IGNORE THAT NOISE: on ckpt_step_344400 its token moved
by 0.0000049 from moment to moment against a token size of 13.86 (0 = the same token every
moment, 13.86 = a change as large as the token itself; review b2-vestibular finding 1, and audit
P012 measured the same on H83). So the slot carried a LEARNED CONSTANT, and every part that reads
it (the forward model, the motor controller, the self-model) was trained with that constant in
it. Behind --vestibular-sense on (default off, so an old launch line builds the model it always
built) that slot carries the body's vestibular / internal-state vector instead
(mimo_bridge/vestibular_adapter.py, 29 numbers per moment): which way gravity pulls in the head
and how fast the head turns, the neck's angles, speeds and efforts, which way is up for the
trunk, what share of the body's support each part carries, mechanical effort, joint-limit load,
fatigue, and a validity flag.

THE SWITCH CHANGES NOTHING THE BODY IS TOLD AT FIRST (function-preserving, since 2026-09-23;
before that fix the orders changed by 11.7 in every 100 parts of their size at the first moment,
review b2-vestibular finding 1). On a load from an intermind checkpoint:
  - the new encoder's last layer starts at weight 0 and bias = the old encoder's mean output
    before the registry's output normalisation (averaged over 64 draws of the placeholder noise
    it was trained on), so its token IS the old constant. On H84 that output had a size of
    1,031,290 (spread 74,425), and the LayerNorm then divided the weights' part of the token and
    every gradient by 74,425: the token never moved (--vestibular-encoder-repair, "the encoder
    repair" below, divides the layer back down without changing a token);
  - the vestibular prediction head's last layer starts at 0, so it predicts "no change" (the
    heads add the current token: --residual-prediction on; without it the bias is the old
    token), which keeps the top-down gate's blend, the planner's goal distance and every
    lesson on this sense at exactly zero until the token starts to vary;
  - the persistent memory's input weights on this slot's columns start at 0: the memory was
    fed zeros for a placeholder sense (memory_integration.observe), so those weights never
    multiplied anything;
  - wellness's grounded prediction-error term leaves this sense out by default
    (--vestibular-wellness-pe exclude), so the raw wellness sum is the one the old model
    computed (DESIGN D8).
The varying part then grows through training: every lesson that reads the core's latent sends a
gradient to the token, and once it varies the next-moment lesson and VICReg grade it.

WHAT THE SLOT KEEPS AND WHAT STARTS FRESH on a resume from an intermind checkpoint:
  kept     the forward model (its modality-embedding row 5 included), every other sense's
           encoder, head, gate and decoder, the motor controller, the wellness evaluators, and
           every Adam moment of those parameters (the persistent memory's two input matrices
           keep theirs; only this slot's columns of the weights are set to 0);
  mapped   the slot's own prediction head, top-down gate and reconstruction decoder: their
           saved "intermind" tensors move to the "vestibular" names (same shapes, same position
           in the model). None of them ever learned anything - measured on ckpt_step_344400:
           the head's Adam second moment is exactly 0 after 344,400 steps (every gradient it got
           was 0; only weight decay shrank its matrices), and the gate and decoder have no Adam
           state at all. Their Adam states are DROPPED so they start fresh with Adam's start-up
           correction on: a carried state holds a step count of 344,400 and moments of exactly
           0, which switched the correction off and moved the head's weights a median 1.8 times
           the learning rate per step against 0.24 for a fresh encoder (review finding 3);
  fresh    the encoder: the old one read five rows of noise (a self-attention pool); the new
           one reads 28 numbers (the vector without its validity flag), centred and scaled per
           field by a running mean and spread (VestibularNormaliser below), through a small MLP
           like the proprioception encoder at the run's --encoder-width and --encoder-layers,
           with a bias on every layer. Its Adam moments start at zero.

THE INPUT IS CENTRED AND SCALED (reviews/env-senses.md MAJOR 1). Measured over 1,200 frames of the
segment 31 world, 87 of every 100 parts of the raw vector's energy is the part every frame shares
(valid is always 1, trunk_up_x averages 0.9985, the head accelerometer 0.93 g, support 1.09 body
weights, the capacities 0.61 and 0.85); a fresh encoder fed the raw vector had 48-52 of every 100
first-layer units whose offset was more than twice their variation, the condition that preceded
the tenure pose head's death (DESIGN section 10). Centred and scaled per field: 0 of every 100.
So each of the 28 fields is centred by a running mean and divided by a running spread, both
updated on REAL frames only (per body), pooled over bodies, over a 2,000-moment horizon, and saved
in runtime_state.pt. The validity flag is a mask, never an input. A field the frame does not carry
(the padded legacy frame's 22 non-MIMo fields, a frame with no vector, a non-finite value) is
filled with the running mean, which is 0 after centring.

THE LOSS. The sense is "real" and its configured loss weight is 1.0, like vision and
proprioception. Under --next-frame-loss every sense's same-moment route is weighted 0.0 (the
prediction heads predict the NEXT moment, lem_model.py); the vestibular sense gets the same
guard and is graded where every other real sense is graded: the delayed next-moment lesson
(at --vestibular-lesson-weight, summed with the other senses), the imagination chain (0.5),
re-study from the replay diary, VICReg and the top-down gate. The approximate evaluator keeps its
five-token input (lem.model.wellness.APPROXIMATE_EXCLUDED).

A FRAME WHOSE VECTOR IS NOT A REAL READ (the first frame after resuming a checkpoint saved before
2026-09-22 carries MIMo's six old numbers, padded, with valid 0; a transport without the vector)
is marked "stale" for the whole batch: it is still encoded and fed forward (the forward model
needs every slot), but it trains no encoder weight, replay stores it as not valid, and it moves
no running statistic. The next-moment lesson grades this sense only when BOTH the moment it
predicts from and the moment it predicts are real: a prediction made from the padded vector is a
prediction from a different quantity (review finding 7; before the fix the moment after a stale
frame was graded, vestibular error 0.95 at that step). Any stale batch other than the first
frame after a resume is logged once per reason.

WHAT A RESUME ACROSS THE SWITCH CLEARS (migrate_learner_state; the episode companion carries
state keyed by sense name): the imagination histories (4 moments), the delayed lesson's cached
previous moment (1 moment), the pending order-end lesson, the goal ring (200 moments; planning
waits for 20), and the placeholder's generator buffer. The replay diary keeps every stored moment
for the other senses; its stored slot-5 tokens are the old noise, and every one of them is marked
not valid for slot 5, so none can grade the vestibular head. The context buffer keeps its 127
past moments; their slot-5 tokens are the old noise and roll out one per moment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
from collections import deque
from typing import Dict, Optional, Sequence

import numpy as np

from mimo_bridge import vestibular_adapter as _VA
from mimo_bridge.vestibular_adapter import (EFFORT_HOLDING_SCALE_NM, EFFORT_POWER_SCALE_W,
                                            FIELD_INDEX, GRAVITY_MS2, GYRO_SCALE_RADS,
                                            LEGACY_VESTIBULAR_DIM, LIMIT_BAND_RAD,
                                            NECK_VEL_SCALE_RADS, VESTIBULAR_DIM,
                                            VESTIBULAR_FIELDS, pad_legacy_vestibular)

log = logging.getLogger("mimo_bridge.vestibular_sense")

MODALITY = "vestibular"
REPLACED = "intermind"
# v2 (2026-09-23): the encoder reads the 28 fields without the validity flag, centred and scaled
# by the running normaliser, and is built with a bias on every layer. A v1 checkpoint (29 raw
# inputs) is refused by stamp_mismatch; none was ever trained past a proof run.
VERSION = "vestibular_sense_v2"

# THE ENCODER'S INPUT: every field but the validity flag, which is a mask (reviews/env-senses.md
# MAJOR 1: `valid` was 1.000 on every real frame with spread 0, a constant input column).
ENCODER_FIELDS = tuple(f for f in VESTIBULAR_FIELDS if f != "valid")
ENCODER_DIM = len(ENCODER_FIELDS)
_ENC_IDX = np.array([FIELD_INDEX[f] for f in ENCODER_FIELDS], dtype=np.int64)
# the six MIMo numbers a padded legacy frame carries (vestibular_adapter.pad_legacy_vestibular)
LEGACY_FIELDS = ("head_acc_x", "head_acc_y", "head_acc_z", "head_gyro_x", "head_gyro_y",
                 "head_gyro_z")

# THE RUNNING MEAN AND SPREAD that centre and scale the input: an exponential average over lived
# moments (all bodies pooled, real frames only), divided by the weight it has accumulated
# (corrected for starting at zero). 2,000 moments (100 s at 20 moments a second) is the horizon
# of the other two running centres in this build (lem/model/contact_head.CENTRE_HORIZON_STEPS and
# the wellness normaliser): a step change in the pooled input moves the centre half way in
# ln 2 x 2,000 = 1,386 moments (69 s), so the part every frame shares (87 of every 100 parts of
# the raw vector's energy) is removed within the first minutes, while one posture held for a few
# seconds by one of 8 bodies moves the centre by under 1 in 100 of its offset.
NORM_HORIZON_STEPS = 2000
# The spread a field is divided by never falls below this, in the field's own units (every field is
# scaled so its measured 99th percentile reads about 1, or has a natural range of 0-1 or -1..1).
# It only stops a field that has not varied at all over the horizon from turning float noise into
# a full-scale input. The control that catches it lying is the stats block's input_spread_min
# (the smallest running spread of any field, with its name) and input_clipped_share.
SPREAD_FLOOR = 1e-3
# THE BOUND ON A CENTRED AND SCALED VALUE: 4 running spreads. The same bound, for the same reason,
# as INPUT_BOUND on the raw vector: one impact frame must not put a value ten times every other
# into the first layer. The control is the stats block's input_clipped_share (the share of
# normalised values the bound changed in the window).
NORMALISED_BOUND = 4.0

# THE INPUT BOUND, in the vector's own units (each field's scale, so 1.0 = one g for the head
# accelerometer, one scale unit for the others). Every value is clipped to [-4, 4] before the
# encoder reads it. MEASURED (b2-vestibular, 2026-09-22 19:20, scratchpad
# b2/angles_measurement.json; the H84 segment 31 world, one body, CPU, vision off):
#   babble (the run's noise 0.05, 3,000 frames)     1 frame in 3,000 had a value beyond 4
#                                                   (head_acc_x 4.45 g); 1.1 in 100,000 values
#   lying still after a reset (1,204 frames)        none beyond 2 (largest 1.12, support_total)
#   placed on a side and settling (600 frames)      0.3 to 0.7 frames in 100 (support_total 6.9)
#   torque rolls (600 frames)                       7 frames in 100 (head_acc 10.9 g, support 11)
#   10 cm drops (300 frames)                        1 frame in 150 each (41.2 g, 18.2 body weights)
# so a babbling or resting body is left untouched and what is cut is the impact tail. Unclipped,
# one impact frame puts a value ten times every other into a bias-free first layer, and the
# encoder's token for that moment is the accelerometer column alone. The control that would
# catch this bound lying is the stats block's clipped_share: the share of real values the bound
# changed in the window (about 0.00001 under babble; rising means the body lives in impacts).
INPUT_BOUND = 4.0

# the settings of each field's scale, stamped into the receipt: a checkpoint trained on one
# scale refuses a build that divides by another (the encoder would read a different quantity)
SCALES = {"head_acc_g_per_ms2": 1.0 / GRAVITY_MS2, "head_gyro_rads": GYRO_SCALE_RADS,
          "neck_vel_rads": NECK_VEL_SCALE_RADS, "effort_power_w": EFFORT_POWER_SCALE_W,
          "effort_holding_nm": EFFORT_HOLDING_SCALE_NM}

# REFERENCE DIRECTIONS for the stats block's angles (never inputs to the model). MEASURED
# (b2-vestibular, 2026-09-22 19:20, scratchpad b2/angles_measurement.json, the same world).
# CHEST_AXES: rows = the trunk's along (hip to head), front (belly) and side axes, written in the
# chest body's own frame. The env calibrates exactly these at the first frame after every reset
# (body_layout.chest_axes on the supine rest pose) and reports body_orientation from them; read
# after four resets over two seeds they differed by at most 0.00026. With the vector's trunk_up
# (world-up seen from the chest) they reproduce the env's body_orientation on every one of 5,704
# frames (babble, sides, face down, rolls, drops) to within 0.09 degrees of roll and 0.18 of
# pitch, on a scale where 0 = lying on the back and 180 = face down.
CHEST_AXES = ((0.034652343256803005, -0.00025664831740062954, 0.9993993942555969),
              (0.9993994202175853, -0.00010939053001065678, -0.034652372248754934),
              (0.0001182183024615323, 0.9999999610826759, 0.0002527035414150351))
# HEAD_GRAVITY_REST: which way gravity pulls in the head frame lying still on the back - the mean
# accelerometer direction over 804 still frames after four resets (1.000 g); 99 in 100 of those
# frames lay within 3.6 degrees of it, so head_tilt_deg reads 0 to 4 for a still supine body.
HEAD_GRAVITY_REST = (0.995596957974032, -0.05333441058950089, 0.07708526396091515)


def adapter_definition() -> dict:
    """Every constant of vestibular_adapter that decides what a field MEANS, read (never written)
    from that module: the field list, the neck joints and motors, the support groups, one g, the
    joint-limit band, the four measured scales and the legacy width. A DEFINITION_VERSION constant,
    if the adapter's owner adds one when a field's computation changes (for example the fatigue
    capacities once FatigueModel's missing actuator gear is corrected), is included too."""
    return {"fields": list(_VA.VESTIBULAR_FIELDS), "neck_joints": list(_VA.NECK_JOINTS),
            "neck_actuators": list(_VA.NECK_ACTUATORS),
            "support_groups": list(_VA.SUPPORT_GROUPS), "gravity_ms2": float(_VA.GRAVITY_MS2),
            "limit_band_rad": float(_VA.LIMIT_BAND_RAD),
            "gyro_scale_rads": float(_VA.GYRO_SCALE_RADS),
            "neck_vel_scale_rads": float(_VA.NECK_VEL_SCALE_RADS),
            "effort_power_scale_w": float(_VA.EFFORT_POWER_SCALE_W),
            "effort_holding_scale_nm": float(_VA.EFFORT_HOLDING_SCALE_NM),
            "legacy_dim": int(_VA.LEGACY_VESTIBULAR_DIM),
            "definition_version": getattr(_VA, "DEFINITION_VERSION", None)}


def adapter_sha() -> str:
    """A short hash of adapter_definition(): any change to what a field means changes it."""
    blob = json.dumps(adapter_definition(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def stamp() -> dict:
    """The receipt's record of what the sense reads (model_receipt.json 'vestibular_sense')."""
    return {"version": VERSION, "modality": MODALITY, "replaces": REPLACED,
            "fields": list(VESTIBULAR_FIELDS), "dim": int(VESTIBULAR_DIM),
            "encoder_fields": list(ENCODER_FIELDS),
            "scales": dict(SCALES), "input_bound": float(INPUT_BOUND),
            "normaliser": {"horizon_steps": int(NORM_HORIZON_STEPS),
                           "spread_floor": float(SPREAD_FLOOR),
                           "bound": float(NORMALISED_BOUND)},
            "adapter_sha": adapter_sha(), "adapter_definition": adapter_definition()}


def stamp_mismatch(saved: Optional[dict]) -> Optional[str]:
    """Why a checkpoint's vestibular stamp cannot be resumed by this code, or None."""
    if not saved:
        return None
    now = stamp()
    if saved.get("version") != now["version"]:
        return (f"the checkpoint's vestibular sense was built by {saved.get('version')!r} and this "
                f"code is {now['version']!r} (v2 reads 28 centred and scaled fields without the "
                f"validity flag; v1 read the 29 raw numbers); its encoder would read a different "
                f"quantity. Write an explicit migration of the vestibular encoder's first layer "
                f"before resuming.")
    for key in ("fields", "dim", "encoder_fields", "input_bound", "normaliser"):
        if saved.get(key) != now[key]:
            return (f"the checkpoint's vestibular sense was trained with {key} = {saved.get(key)!r} "
                    f"and this code reads {now[key]!r}; its encoder would read a different "
                    f"quantity. Write an explicit migration of the vestibular encoder's first layer "
                    f"(and its Adam moments), as _migrate_touch_push_scale does for touch, before "
                    f"resuming.")
    s_old, s_new = saved.get("scales") or {}, now["scales"]
    for k, v in s_new.items():
        if k not in s_old or abs(float(s_old[k]) - float(v)) > 1e-12 * max(1.0, abs(float(v))):
            return (f"the checkpoint's vestibular sense was trained with scale {k} = "
                    f"{s_old.get(k)!r} and this code uses {v!r}; write an explicit migration of the "
                    f"vestibular encoder's first layer before resuming.")
    if saved.get("adapter_sha") != now["adapter_sha"]:
        old_def = saved.get("adapter_definition") or {}
        changed = sorted(k for k in set(old_def) | set(now["adapter_definition"])
                         if old_def.get(k) != now["adapter_definition"].get(k))
        return (f"the checkpoint's vestibular sense was trained on vestibular_adapter constants "
                f"{saved.get('adapter_sha')!r} and this code has {now['adapter_sha']!r} (changed: "
                f"{', '.join(changed) or 'unknown'}); at least one field now means something else. "
                f"Write an explicit migration of the vestibular encoder's first layer before "
                f"resuming.")
    return None


# ------------------------------------------------------------------------------ the config
def modality_overrides(overrides: Optional[dict], lite_overrides: dict) -> dict:
    """Config overrides that put the vestibular sense in the intermind slot.

    The encoder is built exactly like the proprioception encoder - the other flat-vector body
    sense - at the width and depth this run gives proprioception (train_mimo.size_overrides:
    --encoder-width, --encoder-layers; lite default 128 wide, two hidden layers). lem/config/
    resolver.py places the modality directly after intermind and gives it its input width."""
    lite_p = lite_overrides["modalities"]["proprioception"]["encoder"]
    ov_p = (((overrides or {}).get("modalities") or {}).get("proprioception") or {}).get("encoder") or {}
    hidden = [int(h) for h in ov_p.get("hidden_dims", lite_p["hidden_dims"])]
    d_int = int(ov_p.get("d_internal", lite_p["d_internal"]))
    return {"modalities": {
        REPLACED: {"enabled": False},
        MODALITY: {
            "enabled": True,
            "data_source": "real",
            "loss_weight": 1.0,
            # one read per model step (the last env frame of each 5-frame burst)
            "rate_hz": 20,
            # the 28 fields without the validity flag, centred and scaled (VestibularNormaliser)
            "pipeline": {"format": "flat_vector", "input_dim": int(ENCODER_DIM),
                         "fields": list(ENCODER_FIELDS)},
            # bias on: the switch from an intermind checkpoint sets the last layer's weight to 0
            # and its bias to the old encoder's constant (migrate_checkpoint), which needs a bias;
            # every other encoder is bias-free (master_config encoder_defaults.bias false)
            "encoder": {"type": "mlp_project", "hidden_dims": hidden, "d_internal": d_int,
                        "bias": True},
            # never used: the sense is real, so the placeholder system builds no generator for it
            "placeholder": {"strategy": "gaussian", "mean": 0.0, "std": 0.1, "clip": [0.0, 1.0]},
            "loss_ramp": {"enabled": False, "from_weight": 0.1, "to_weight": 1.0,
                          "over_steps": 1000},
        }}}


# THE SENSE'S WEIGHT IN THE NEXT-MOMENT LESSON (--vestibular-lesson-weight; review b2-vestibular
# finding 2). The lesson adds each graded sense's error with weight 1; this is the vestibular
# term's multiplier. MEASUREMENT: see LESSON_WEIGHT_MEASUREMENT. A placeholder since Sept 23 (the lane's
# after-fix share run never ran); the rule that sets it is lesson_weight_rule below (the third review's fix
# round, 2026-09-26), applied on the repaired run: until that reading exists the rule's own default, 1.0.
LESSON_WEIGHT = 1.0
LESSON_WEIGHT_MEASUREMENT: Dict[str, object] = {}

WELLNESS_PE_CHOICES = ("exclude", "include")


def configure(model, lesson_weight: Optional[float], wellness_pe: str) -> None:
    """The vestibular build's own settings on the model (LemLite.__init__, flag on only)."""
    if wellness_pe not in WELLNESS_PE_CHOICES:
        raise ValueError(f"--vestibular-wellness-pe must be one of {WELLNESS_PE_CHOICES}, "
                         f"got {wellness_pe!r}")
    w = LESSON_WEIGHT if lesson_weight is None else float(lesson_weight)
    if not math.isfinite(w) or w < 0.0:
        raise ValueError(f"--vestibular-lesson-weight must be 0 or more, got {lesson_weight!r}")
    model.vestibular_lesson_weight = w
    model.vestibular_norm = VestibularNormaliser()
    # "exclude": wellness's grounded prediction-error term averages the other real senses only,
    # so the raw wellness sum is the one the intermind model computed (DESIGN D8); "include":
    # the sense's error counts there too, which scores moving moments lower than resting ones
    # through this sense alone (review b2-vestibular finding 4)
    model.wellness.pe_excluded = (MODALITY,) if wellness_pe == "exclude" else ()
    st = model.receipt.get("vestibular_sense")
    if isinstance(st, dict):
        st["lesson_weight"] = w
        st["wellness_pe"] = wellness_pe


def check_layout(active: Sequence[str]) -> None:
    """The sense must sit in intermind's place and intermind must be gone."""
    if MODALITY not in active or REPLACED in active:
        raise ValueError(f"--vestibular-sense on expects '{MODALITY}' active and '{REPLACED}' "
                         f"inactive; the resolved modalities are {list(active)}")


# ------------------------------------------------------------------------------ the frames
def frame_vectors(frames) -> tuple:
    """The batch's vectors as the encoder reads them.

    Returns (vectors [B, VESTIBULAR_DIM] float32, clipped to +-INPUT_BOUND; valid [B] bool, True
    where the frame carries a real read; clipped [B] int, how many of that body's values the
    bound changed). A legacy (6,) vector is padded (vestibular_adapter.pad_legacy_vestibular,
    valid 0); a frame with no vector reads as zeros with valid 0."""
    B = len(frames)
    out = np.zeros((B, VESTIBULAR_DIM), dtype=np.float32)
    for i, f in enumerate(frames):
        v = getattr(f, "vestibular", None)
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        if a.size != VESTIBULAR_DIM:
            a = pad_legacy_vestibular(a)
        out[i] = a
    valid = (out[:, FIELD_INDEX["valid"]] > 0.5) & np.isfinite(out).all(axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=INPUT_BOUND, neginf=-INPUT_BOUND)
    clipped = (np.abs(out) > INPUT_BOUND).sum(axis=1).astype(np.int64)
    np.clip(out, -INPUT_BOUND, INPUT_BOUND, out=out)
    return out, valid, clipped


# why a body's frame is not a real read (the key of the once-per-reason log line)
STALE_LEGACY = "a padded legacy vector (the first frame after resuming a checkpoint saved before 2026-09-22)"
STALE_NONE = "no vestibular vector on the frame"
STALE_NONFINITE = "a non-finite value in the vector"
STALE_INVALID = "a full-width vector whose valid field is 0"


def read_frames(frames) -> tuple:
    """The batch's raw vectors with what each frame does NOT carry marked as missing.

    Returns (raw [B, VESTIBULAR_DIM] float64, NaN where the frame carries no reading of that field;
    valid [B] bool, a real read; why [B] str, '' for a real read or the STALE_* reason). A legacy
    (6,) vector carries only its six MIMo numbers (on the live read's scales); a frame with no
    vector carries nothing; a non-finite value is missing. Any other width is a broken contract
    and raises (vestibular_adapter.pad_legacy_vestibular)."""
    B = len(frames)
    raw = np.full((B, VESTIBULAR_DIM), np.nan, dtype=np.float64)
    valid = np.zeros(B, dtype=bool)
    why = [""] * B
    for i, f in enumerate(frames):
        v = getattr(f, "vestibular", None)
        if v is None:
            why[i] = STALE_NONE
            continue
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        if a.size != VESTIBULAR_DIM:
            p = pad_legacy_vestibular(a).astype(np.float64)       # raises on a width not 6 or 29
            raw[i, :LEGACY_VESTIBULAR_DIM] = p[:LEGACY_VESTIBULAR_DIM]
            why[i] = STALE_LEGACY
            continue
        fin = np.isfinite(a)
        raw[i, fin] = a[fin]
        if not fin.all():
            why[i] = STALE_NONFINITE
        elif a[FIELD_INDEX["valid"]] <= 0.5:
            why[i] = STALE_INVALID
        else:
            valid[i] = True
    return raw, valid, why


class VestibularNormaliser:
    """Centres and scales each of the 28 encoder fields by a running mean and spread.

    Lives on the model as `vestibular_norm` (a public name: continuation_state captures only _* and
    last_* attributes, and this belongs to the model's input function, saved in runtime_state.pt
    under 'vestibular_norm'). Used in two calls per model step:
      batch(frames)  -> (input [B, 28] float32, valid [B], why [B]) with the CURRENT statistics;
                        the real rows are held until commit()
      commit()       -> fold the held real rows into the statistics (the model calls it only on a
                        training step, so a frozen probe never moves them)
    Before the first real frame every input is 0 (nothing to centre against yet)."""

    def __init__(self, horizon_steps: int = NORM_HORIZON_STEPS):
        self.horizon_steps = int(horizon_steps)
        self.mean = np.zeros(ENCODER_DIM, dtype=np.float64)
        self.var = np.zeros(ENCODER_DIM, dtype=np.float64)
        self.weight = 0.0            # 0 = no real frame yet; tends to 1 as the horizon fills
        self.n_real = 0              # real body-moments folded in, lifetime
        self._held: Optional[np.ndarray] = None
        self.last_input: Optional[np.ndarray] = None       # [B, 28] what the encoder read last
        self.last_valid: Optional[np.ndarray] = None
        self.logged: set = set()     # stale reasons already logged once
        self.window = {"values": 0, "clipped": 0}          # the stats window's clipped share

    # ---------------------------------------------------------------- the transform
    def spread(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, SPREAD_FLOOR ** 2))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x [B, 28] in field units, NaN = not carried -> [B, 28] centred, scaled, bounded; a
        missing field reads 0 (the running mean)."""
        if self.weight <= 0.0:
            z = np.zeros_like(x, dtype=np.float64)
        else:
            z = (x - self.mean) / self.spread()
            z = np.where(np.isfinite(z), z, 0.0)
        self.window["values"] += int(z.size)
        self.window["clipped"] += int((np.abs(z) > NORMALISED_BOUND).sum())
        return np.clip(z, -NORMALISED_BOUND, NORMALISED_BOUND)

    def batch(self, frames) -> tuple:
        raw, valid, why = read_frames(frames)
        # the raw bound first: an impact frame (a 41 g drop) must not reach the statistics either
        x = np.clip(raw[:, _ENC_IDX], -INPUT_BOUND, INPUT_BOUND)
        z = self.transform(x).astype(np.float32)
        self._held = x[valid].copy() if valid.any() else None
        self.last_input, self.last_valid = z, valid
        for reason in sorted({w for w in why if w and w != STALE_LEGACY}):
            if reason not in self.logged:
                self.logged.add(reason)
                log.warning("vestibular sense: %d of %d bodies' frames are not a real read (%s); "
                            "the batch is marked stale: encoded, graded by nothing, no running "
                            "statistic moves. Logged once per reason; the stats block's "
                            "frames_invalid and real_steps count every such moment.",
                            sum(1 for w in why if w == reason), len(why), reason)
        return z, valid, why

    def commit(self) -> None:
        rows, self._held = self._held, None
        if rows is None or not len(rows):
            return
        a = 1.0 / float(self.horizon_steps)
        w_new = (1.0 - a) * self.weight + a
        keep = (1.0 - a) * self.weight / w_new
        self.mean = self.mean * keep + rows.mean(axis=0) * (a / w_new)
        d2 = ((rows - self.mean) ** 2).mean(axis=0)
        self.var = self.var * keep + d2 * (a / w_new)
        self.weight = w_new
        self.n_real += int(len(rows))

    # ---------------------------------------------------------------- runtime_state.pt
    def state_dict(self) -> dict:
        import torch
        return {"version": VERSION, "fields": list(ENCODER_FIELDS),
                "horizon_steps": int(self.horizon_steps),
                "mean": torch.from_numpy(self.mean.copy()), "var": torch.from_numpy(self.var.copy()),
                "weight": float(self.weight), "n_real": int(self.n_real)}

    def load_state_dict(self, sd: Optional[dict]) -> str:
        """Restore; returns what happened, in words, for the load's log line."""
        if not sd:
            return "none saved: the running mean and spread start cold (the first real frame sets them)"
        if list(sd.get("fields") or []) != list(ENCODER_FIELDS) or sd.get("version") != VERSION:
            return (f"saved for another field list or version ({sd.get('version')!r}): they start "
                    f"cold")
        self.mean = np.asarray(sd["mean"], dtype=np.float64).reshape(ENCODER_DIM).copy()
        self.var = np.asarray(sd["var"], dtype=np.float64).reshape(ENCODER_DIM).copy()
        self.weight = float(sd.get("weight", 0.0))
        self.n_real = int(sd.get("n_real", 0))
        return (f"restored from {self.n_real} real body-moments (the {self.horizon_steps}-moment "
                f"horizon {100.0 * self.weight:.0f}% filled)")

    def window_summary(self) -> dict:
        """The stats block's normaliser readings for this window; starts a new window."""
        sp = self.spread()
        i = int(np.argmin(sp))
        out = {"input_centre_filled_share": round(float(self.weight), 4),
               "input_real_body_moments": int(self.n_real),
               "input_spread_min": {"field": ENCODER_FIELDS[i], "spread": float(f"{sp[i]:.4g}"),
                                    "floor": SPREAD_FLOOR},
               "input_clipped_share": (round(self.window["clipped"] / self.window["values"], 6)
                                       if self.window["values"] else None)}
        self.window = {"values": 0, "clipped": 0}
        return out


# ------------------------------------------------------------------------------ checkpoints
ENC_OLD = f"encoder_registry.encoders.{REPLACED}."
ENC_NEW = f"encoder_registry.encoders.{MODALITY}."


def _rename(name: str, old: str, new: str) -> str:
    """Replace the first path segment equal to `old` with `new` (module paths only)."""
    segs = name.split(".")
    for i, s in enumerate(segs):
        if s == old:
            segs[i] = new
            return ".".join(segs)
    return name


def saved_name(current: str) -> Optional[str]:
    """The name a current parameter or buffer had in an intermind checkpoint; None for the
    vestibular encoder, which has no counterpart there."""
    if current.startswith(ENC_NEW):
        return None
    return _rename(current, MODALITY, REPLACED)


def switch_needed(saved_keys, active: Sequence[str]) -> bool:
    """True when this build reads the vestibular sense and the checkpoint's slot held intermind."""
    keys = list(saved_keys)
    return (MODALITY in active
            and any(k.startswith(ENC_OLD) for k in keys)
            and not any(k.startswith(ENC_NEW) for k in keys))


def check_saved_layout(saved_active: Optional[Sequence[str]], active: Sequence[str]) -> None:
    """The checkpoint's modality list, with intermind renamed, must be this build's list."""
    if not saved_active:
        return
    want = [MODALITY if m == REPLACED else m for m in saved_active]
    if list(want) != list(active):
        raise ValueError(f"the checkpoint's senses {list(saved_active)} do not map onto this "
                         f"build's {list(active)} by putting '{MODALITY}' in '{REPLACED}''s place")


def _components(model) -> dict:
    comps = {"encoder_registry": model.encoder_registry, "forward_model": model.forward_model,
             "prediction_heads": model.prediction_heads, "motor_controller": model.motor_controller,
             "wellness": model.wellness}
    comps.update({k: v for k, v in model._side_modules().items() if v is not None})
    if getattr(model, "gaze_head", None) is not None:
        comps["gaze_head"] = model.gaze_head
    if getattr(model, "voice_head", None) is not None:
        comps["voice_head"] = model.voice_head
    return {k: getattr(v, "_orig_mod", v) for k, v in comps.items()}


def migrate_weights(model, saved: Dict[str, "object"]) -> dict:
    """Copy every saved intermind-named tensor outside the encoder onto its vestibular name.

    saved: the checkpoint's flat {component.key: tensor} (model.safetensors). Raises when a
    saved intermind tensor has no counterpart of the same shape, or when a vestibular tensor
    outside the encoder is left without one - so nothing is silently left at its random init."""
    import torch
    current = {}
    for comp, mod in _components(model).items():
        for k, t in mod.state_dict(keep_vars=True).items():
            current[f"{comp}.{k}"] = t
    copied, dropped = [], []
    with torch.no_grad():
        for k in sorted(saved):
            if REPLACED not in k.split("."):
                continue
            if k.startswith(ENC_OLD):
                dropped.append(k)
                continue
            target = _rename(k, REPLACED, MODALITY)
            t = current.get(target)
            if t is None:
                raise ValueError(f"saved tensor {k} has no '{MODALITY}' counterpart in this build")
            src = saved[k]
            if tuple(t.shape) != tuple(src.shape):
                raise ValueError(f"saved tensor {k} is {tuple(src.shape)} but {target} is "
                                 f"{tuple(t.shape)}")
            t.copy_(src.to(device=t.device, dtype=t.dtype))
            copied.append(target)
    left = [n for n in current if MODALITY in n.split(".") and not n.startswith(ENC_NEW)
            and n not in copied]
    if left:
        raise ValueError(f"these {MODALITY} tensors had no saved {REPLACED} tensor to start "
                         f"from: {left}")
    fresh = [n for n in current if n.startswith(ENC_NEW)]
    return {"mapped_tensors": len(copied), "mapped": sorted({n.split(".")[0] for n in copied}),
            "dropped_encoder_tensors": len(dropped), "fresh_encoder_tensors": len(fresh)}


def migrate_optimizer(model, saved_opt: dict, saved: Dict[str, "object"]) -> dict:
    """Carry every saved Adam moment onto this build's optimizer by parameter NAME.

    The checkpoint's optimizer lists parameters by position. This build's parameter list is the
    checkpoint's with one block changed: the intermind encoder's parameters (where the
    vestibular encoder's now sit) - every other parameter keeps its position, and the slot's head,
    gate and decoder keep theirs under a new name. The saved positions are therefore reproduced
    exactly by rebuilding the intermind encoder from this build's own resolved config (resolve()
    fills every modality's encoder settings, enabled or not) and splicing its parameters in. Every
    saved moment is then checked against the shape of the saved tensor its name says it belongs
    to before anything is loaded; one mismatch raises and nothing is changed."""
    import torch
    from lem.model.encoders import ENCODER_FACTORIES
    opt = model.optimizer
    comps = _components(model)
    id2name = {}
    for c, mod in comps.items():
        for n, p in mod.named_parameters():
            id2name[id(p)] = f"{c}.{n}"
    ecfg = model.config["modalities"][REPLACED]["encoder"]
    old_enc = ENCODER_FACTORIES[ecfg["type"]](copy.deepcopy(ecfg))
    old_named = [(ENC_OLD + n, p.dim(), tuple(p.shape)) for n, p in old_enc.named_parameters()]
    saved_enc = {k for k in saved if k.startswith(ENC_OLD)}
    extra = saved_enc - {n for n, _, _ in old_named}
    missing = {n for n, _, _ in old_named} - saved_enc
    if extra or missing:
        raise ValueError(f"the rebuilt {REPLACED} encoder does not match the checkpoint's: "
                         f"saved only {sorted(extra)}, rebuilt only {sorted(missing)}")
    for n, _, shp in old_named:
        if tuple(saved[n].shape) != shp:
            raise ValueError(f"{n}: saved {tuple(saved[n].shape)}, rebuilt {shp}")
    old_list, inserted = [], False
    for p in model.params:
        n = id2name.get(id(p))
        if n is None:
            raise ValueError("a parameter in the optimizer has no qualified name")
        if n.startswith(ENC_NEW):
            if not inserted:
                old_list.extend((m, d) for m, d, _ in old_named)
                inserted = True
            continue
        if saved_name(n) not in saved:
            # A MODULE THE CHECKPOINT NEVER HAD (appended since it was saved: the efference
            # heads, the contact head). It held no position in the saved optimizer and starts with
            # fresh moments. Counting it here made every build that switched the slot AND added
            # such a module refuse the checkpoint - measured 2026-09-22 (b4): --vestibular-sense on
            # with --efference-heads on could not load ckpt_step_344400 ("groups hold [224, 243]
            # parameters; the layout rebuilt here holds [232, 252]", the 8 + 9 efference tensors).
            continue
        old_list.append((saved_name(n), p.dim()))
    if not inserted:
        raise ValueError(f"this build has no {MODALITY} encoder parameters to replace")
    # the optimizer's two groups: weight-decayed matrices, then everything 1-D (LemLite.__init__)
    old_groups = [[n for n, d in old_list if d >= 2], [n for n, d in old_list if d < 2]]
    s_groups = saved_opt.get("param_groups") or []
    if len(s_groups) != len(old_groups) or any(
            len(sg["params"]) != len(og) for sg, og in zip(s_groups, old_groups)):
        raise ValueError(f"the checkpoint's optimizer groups hold "
                         f"{[len(g['params']) for g in s_groups]} parameters; the {REPLACED} "
                         f"layout rebuilt here holds {[len(g) for g in old_groups]}")
    name_to_idx = {}
    for sg, og in zip(s_groups, old_groups):
        for idx, n in zip(sg["params"], og):
            name_to_idx[n] = idx
    st = saved_opt.get("state") or {}
    bad = []
    for n, idx in name_to_idx.items():
        s = st.get(idx)
        ea = None if s is None else s.get("exp_avg")
        if torch.is_tensor(ea) and n in saved and tuple(ea.shape) != tuple(saved[n].shape):
            bad.append(n)
    if bad:
        raise ValueError(f"{len(bad)} saved Adam moments do not have the shape of the tensor their "
                         f"position names (first: {bad[0]}); the saved positions are not the "
                         f"layout rebuilt here")
    cur = opt.state_dict()
    new_state, fresh = {}, []
    for cg, gobj in zip(cur["param_groups"], opt.param_groups):
        for cur_idx, p in zip(cg["params"], gobj["params"]):
            n = id2name[id(p)]
            on = saved_name(n)
            if on is None or on not in saved:
                # the new encoder, or a module appended since the checkpoint was saved
                fresh.append(n)
                continue
            sidx = name_to_idx.get(on)
            if sidx is None:
                raise ValueError(f"{n} ({on} in the checkpoint) has no saved position")
            if sidx in st:
                new_state[cur_idx] = st[sidx]
    dropped = sum(1 for n, idx in name_to_idx.items() if n.startswith(ENC_OLD) and idx in st)
    if len(new_state) + dropped != len(st):
        raise ValueError(f"{len(st)} saved Adam states: {len(new_state)} carried and {dropped} "
                         f"dropped with the {REPLACED} encoder do not add up")
    groups = []
    for sg, cg in zip(s_groups, cur["param_groups"]):
        g = dict(sg)
        g["params"] = list(cg["params"])
        groups.append(g)
    opt.load_state_dict({"state": new_state, "param_groups": groups})
    return {"adam_states_saved": len(st), "adam_states_carried": len(new_state),
            "adam_states_dropped_with_old_encoder": dropped,
            "fresh_parameters": len(fresh)}


# how many draws of the placeholder noise the old encoder's constant is averaged over. The old
# token moved by 0.0000049 against its size of 13.86 from moment to moment on ckpt_step_344400
# (review b2-vestibular, probe.json), so any draw count reproduces it to that size; 64 draws make
# the average's own spread 8 times smaller again, and the switch report records the measured
# spread of the draws (old_token_spread) beside the token size.
OLD_CONSTANT_DRAWS = 64


def old_encoder_constant(model, saved: Dict[str, "object"]) -> tuple:
    """The intermind encoder's mean output BEFORE the registry's output normalisation, on the
    placeholder noise it was trained on, from the checkpoint's own saved tensors.

    Returns (c [d_model] tensor, spread: the mean distance of one draw's output from c, token_size:
    the norm of the normalised token). The draws come from a private generator, so no random
    stream of the run moves."""
    import torch
    from lem.model.encoders import ENCODER_FACTORIES
    from lem.placeholders.system import PlaceholderSystem
    cfg = model.config
    mcfg = cfg["modalities"][REPLACED]
    ecfg = mcfg["encoder"]
    old = ENCODER_FACTORIES[ecfg["type"]](copy.deepcopy(ecfg))
    old.load_state_dict({k[len(ENC_OLD):]: v for k, v in saved.items() if k.startswith(ENC_OLD)},
                        strict=True)
    old = old.to(model.device).eval()
    pc = mcfg.get("placeholder") or {}
    shape = PlaceholderSystem._resolve_shape(REPLACED, mcfg.get("pipeline", {}), ecfg, cfg["body"],
                                             cfg["model"]["d_model"])
    g = torch.Generator().manual_seed(0)
    x = torch.randn(OLD_CONSTANT_DRAWS, *shape, generator=g) * float(pc.get("std", 0.1)) + float(
        pc.get("mean", 0.0))
    clip = pc.get("clip", (0.0, 1.0))
    if clip is not None:
        x = x.clamp(float(clip[0]), float(clip[1]))
    with torch.no_grad():
        out = old(x.to(model.device)).reshape(OLD_CONSTANT_DRAWS, -1).float()
        c = out.mean(dim=0)
        spread = float((out - c).norm(dim=-1).mean())
        norm = model.encoder_registry.output_norm
        tok = norm(c) if norm is not None else c
    return c, spread, float(tok.norm())


def preserve_function(model, saved: Dict[str, "object"]) -> dict:
    """Start the switched slot so the model computes what the intermind model computed.

    The old slot held a learned constant (the module docstring). Set: the new encoder's last layer
    to weight 0 and bias = that constant (before output normalisation), so the vestibular token IS
    the old token; the vestibular prediction head's last layer to 0 (under residual prediction
    it then predicts "no change"; without it, weight 0 and bias = the old token); the persistent
    memory's input weights on this slot's columns to 0 (the memory read zeros for a placeholder
    sense). Returns what was set, by tensor name, for the switch report and the load proof."""
    import torch
    c, spread, tok_size = old_encoder_constant(model, saved)
    enc = model.encoder_registry.encoders[MODALITY]
    last = enc.mlp[-1]
    if getattr(last, "bias", None) is None:
        raise ValueError("the vestibular encoder's last layer has no bias; modality_overrides "
                         "builds it with bias on so the switch can start it at the old constant")
    set_names = []
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(c.to(last.bias))
        idx = len(enc.mlp) - 1
        set_names += [f"{ENC_NEW}mlp.{idx}.weight", f"{ENC_NEW}mlp.{idx}.bias"]
        head = model.prediction_heads.heads[MODALITY]
        hl = head.mlp[-1]
        hl.weight.zero_()
        if getattr(model.prediction_heads, "residual", False):
            hl.bias.zero_()
        else:
            norm = model.encoder_registry.output_norm
            hl.bias.copy_((norm(c) if norm is not None else c).to(hl.bias))
        hidx = len(head.mlp) - 1
        set_names += [f"prediction_heads.heads.{MODALITY}.mlp.{hidx}.weight",
                      f"prediction_heads.heads.{MODALITY}.mlp.{hidx}.bias"]
        pm = getattr(model, "persistent_memory", None)
        mem_cols = None
        if pm is not None:
            k = model.active.index(MODALITY)
            d = int(model.d_model)
            mem_cols = [k * d, (k + 1) * d]
            pm.input_projection[0].weight[:, k * d:(k + 1) * d] = 0.0
            pm.query.weight[:, k * d:(k + 1) * d] = 0.0
            set_names += ["persistent_memory.input_projection.0.weight",
                          "persistent_memory.query.weight"]
    return {"old_constant_draws": OLD_CONSTANT_DRAWS,
            "old_token_spread": float(f"{spread:.3g}"), "old_token_size": round(tok_size, 3),
            "residual_prediction": bool(getattr(model.prediction_heads, "residual", False)),
            "memory_slot_columns_zeroed": mem_cols, "tensors_set": set_names}


def drop_slot_adam_states(model) -> list:
    """Drop the carried Adam states of the slot's prediction head, top-down gate and reconstruction
    decoder: their moments are exactly 0 with a step count of 344,400, which switches Adam's
    start-up correction off (review b2-vestibular finding 3). Returns the dropped names."""
    mods = [("prediction_heads.heads." + MODALITY, model.prediction_heads.heads[MODALITY])]
    if getattr(model, "td_gates", None) is not None and MODALITY in model.td_gates:
        mods.append(("td_gates." + MODALITY, model.td_gates[MODALITY]))
    if getattr(model, "recon_decoders", None) is not None and MODALITY in model.recon_decoders:
        mods.append(("recon_decoders." + MODALITY, model.recon_decoders[MODALITY]))
    dropped = []
    for prefix, mod in mods:
        for n, p in mod.named_parameters():
            if model.optimizer.state.pop(p, None) is not None:
                dropped.append(f"{prefix}.{n}")
    return dropped


def migrate_checkpoint(model, path: str, load_optimizer: bool = True) -> dict:
    """The weights and (if asked) the optimizer of an intermind checkpoint, moved onto this
    vestibular build, started so the switch changes nothing the model computes (preserve_function).
    Called by LemLite.load_checkpoint after CheckpointManager.load, which loads everything whose
    name is unchanged and leaves the optimizer to this function."""
    import os
    import torch
    import safetensors.torch as st_
    saved = st_.load_file(os.path.join(path, "model.safetensors"))
    report = {"switch": f"{REPLACED} -> {MODALITY}", "slot": int(model.active.index(MODALITY))}
    report.update(migrate_weights(model, saved))
    report["function_preserving"] = preserve_function(model, saved)
    opt_path = os.path.join(path, "optimizer.pt")
    report["optimizer_read"] = bool(load_optimizer and os.path.isfile(opt_path))
    report["optimizer_file_found"] = bool(os.path.isfile(opt_path))
    if report["optimizer_read"]:
        saved_opt = torch.load(opt_path, weights_only=False, map_location=model.device)
        report.update(migrate_optimizer(model, saved_opt.get("optimizer") or {}, saved))
        names = drop_slot_adam_states(model)
        report["adam_states_dropped_fresh"] = len(names)
        report["adam_states_dropped_fresh_names"] = names
    else:
        report["adam_states_carried"] = None
    return report


def load_message(report: dict) -> str:
    """The load's log line, in words, including the case where no optimizer was read."""
    fp = report.get("function_preserving") or {}
    head = (f"vestibular sense: the checkpoint's sixth input slot held the intermind placeholder "
            f"(a learned constant token, size {fp.get('old_token_size')}, moving "
            f"{fp.get('old_token_spread')} between noise draws); its prediction head, top-down gate "
            f"and reconstruction decoder moved to the vestibular names "
            f"({report.get('mapped_tensors', 0)} tensors); its encoder was replaced by a fresh "
            f"vestibular encoder ({report.get('dropped_encoder_tensors', 0)} tensors dropped, "
            f"{report.get('fresh_encoder_tensors', 0)} new) whose last layer starts at the old "
            f"constant, the vestibular head starts predicting no change, and the persistent "
            f"memory's weights on this slot's columns {fp.get('memory_slot_columns_zeroed')} start "
            f"at 0, so the first step computes what the old model computed. ")
    if report.get("optimizer_read"):
        return head + (f"{report.get('adam_states_carried')} of {report.get('adam_states_saved')} "
                       f"saved Adam states carried by name, "
                       f"{report.get('adam_states_dropped_with_old_encoder')} dropped with the old "
                       f"encoder, {report.get('adam_states_dropped_fresh')} dropped from the slot's "
                       f"head, gate and decoder (they start fresh). Every other weight loaded "
                       f"unchanged.")
    if report.get("optimizer_file_found"):
        return head + ("NO OPTIMIZER STATE WAS READ (the load asked for none): every parameter "
                       "starts with fresh Adam moments. Every other weight loaded unchanged.")
    return head + ("NO OPTIMIZER STATE WAS READ: optimizer.pt is ABSENT from the checkpoint, so "
                   "every parameter starts with fresh Adam moments. Every other weight loaded "
                   "unchanged.")


# ------------------------------------------------------------------------------ the encoder repair
# --vestibular-encoder-repair (docs/gain_of_function_v4/agents/vestibular_encoder_2026-09-25.md; its independent
# review docs/gain_of_function_v4/agents/vestibular_encoder_review_2026-09-25.md).
#
# WHAT WENT WRONG. preserve_function above started the new encoder's last layer at weight 0 and
# bias = the old intermind encoder's mean output BEFORE the registry's output LayerNorm. On H84 that
# output had grown to a size of 1,031,290 (spread 74,425 across its 192 numbers; read from the saved
# weights of all 96 checkpoints since the deploy, vestibular_encoder_records.py). The LayerNorm divides
# its input by that spread, so everything the layer's weights add reaches the token divided by 74,425
# (the token varied by at most 3.6e-7 per number, float32's own rounding of it), and every gradient the
# encoder receives is divided by 74,425 too: at least 94 of every 100 gradient numbers of every encoder
# tensor sat below Adam's eps, so each update was 0.03 to 1.1 in 100 of a normal Adam step, and the
# bias itself could not move at all (an update of about 4e-6 per number is below float32's spacing of
# 0.0078 at 74,000). The fixture of the deploy's own test built a constant of size about 14.
#
# THE REPAIR. Divide the last layer's weight and bias by one number s, a POWER OF TWO. The layer's output
# y = W h + b becomes y / s, and the registry's LayerNorm (no gain or bias: encoder_registry.py) reads
# (y - mean) / sqrt(var + eps): dividing y by s leaves that unchanged except for eps, so every token stays
# what it was. Dividing by a power of two changes no rounding: in float32 and in bf16 (the precision H84's
# encoders run in on the graphics card, under autocast) W / s and b / s are the same numbers with a smaller
# exponent, and so is every product and sum the layer forms from them. With s = spread / 10, not a power of
# two, the independent review measured the switch moving the bf16 token by up to 0.0103 per number, 500
# times the check line, while the float64 and float32 checks read 2e-7 and 7e-7 (its finding 3). What changes
# is what learning can do: the weights' part and every gradient are no longer divided by 74,425 but by the
# new spread. Nothing else in the encoder changes; the load then restarts the next-moment normalisation's
# and learning progress's averages of this sense (LemLite._vestibular_encoder_repair_after_runtime).
# TWO WAYS TO CARRY ADAM ACROSS THE SWITCH (both keep every weight and token the same):
#   rescale          the divided layer's moments follow the division exactly (first x s, second x s^2);
#                    the hidden layers keep theirs. Exact only for moments accumulated in the dead state
#                    (H84: every encoder tensor's step count is the checkpoint's step - 344,400). Adam's own
#                    arithmetic for carried moments near 0 meeting a gradient that switches on: 3.16
#                    learning rates on the first update and 6.57 on the 12th for a steady gradient, 7.26 for
#                    one growing 10 in 100 per update (review finding 6, scratchpad vrev/adam_ratio.py);
#                    measured on the real checkpoints 411,150 and 347,791: hidden-layer steps of 7.0 and 7.1.
#   rescale_restart  every tensor of the encoder restarts Adam (its saved state is dropped, so Adam's
#                    start-up correction applies from the next update), as the deploy did for the slot's
#                    head, gate and decoder (review b2-vestibular finding 3). A fresh state takes steps of 1
#                    learning rate on a steady gradient and 3.6-4.0 on one growing 1-5 in 100 per update;
#                    measured on the same checkpoints: 2.0 and 1.9. What is dropped is 66,750 updates of
#                    gradients 74,425 times too small. The deploy's mode (review verdict; the pause
#                    measurement's rule MODE re-checks it on the pause's own checkpoint).
#   rescale_restart_fp32  rescale_restart, and this encoder computed in float32 from then on (review 2,
#                    finding 4). The division changes no number, so it keeps the weights' part of the token as
#                    small beside the bias as it was: 5.9e-6 per number against a bias of about 55 after the
#                    switch, a ten-millionth. bf16, the precision the card runs the encoders in, keeps 8
#                    significant bits (a step of 0.25 at 55), so there the weights' part is rounded away and the
#                    token is one value in every body: the anti-collapse lesson's gradient is then exactly 0 on
#                    all 14 encoder tensors (measured by the review on 411,150 and 415,000 under CPU autocast
#                    bf16; in float32, where the weights' part is 1 to 3 steps of 3.8e-6, the same lesson spread
#                    the token to 0.19 between the bodies in 100 updates). Computed in float32 the encoder's token
#                    moves at the switch by the bf16 rounding of the dead token the model has run on (at most
#                    0.0064 per number, 0.0019 root mean square, the same in every body: the first review's
#                    reading at 411,150); the pause measurement reads what that does to every head (EM). The
#                    graphics card's own reading decides between this mode and rescale_restart (rule MODE).
REPAIR_MODES = ("off", "rescale", "rescale_restart", "rescale_restart_fp32")
REPAIR_FP32_MODES = ("rescale_restart_fp32",)
REPAIR_VERSION = "vestibular_encoder_repair_v3"
# THE SPREAD THE BIAS IS DIVIDED DOWN TO. The LayerNorm's eps is the one place the division shows: it scales
# the token by 1 / sqrt(1 + eps / spread^2), a change of eps / (2 spread^2) of the token. That is at most
# half a float32 step (2^-24 = 6.0e-8 of a value) for a spread of sqrt(eps / 2^-23) or more: 9.16 at the
# registry's eps of 1e-5 (repair_spread_floor reads the eps from the norm itself). s is the LARGEST power of
# two that keeps the spread at or above that floor, so the spread after the switch lies between 9.16 and
# 18.3. On H84 (spread 74,425.2): s = 4,096, spread 18.17, eps term 1.5e-8 of a value. The next power, 8,192,
# gives a spread of 9.08 and an eps term of 6.1e-8, just past half a float32 step (the review measured it at
# 2.4e-7 per number in float32 and bf16 alike, all of it the eps term). The learning it restores: every
# update moves the token 4,096 times further than today (the review measured one update at a spread of 10
# moving it 7,442 times further); at 18.17 it is 1.8 times slower than at 10, which the pause measurement's
# rule LS reads on the real checkpoints.
REPAIR_EPS_STEP = 2.0 ** -24          # half a float32 step, as a share of a value
# THE SWITCH LINE: the switch divides only a bias whose spread is above 1,000 (review finding 7: "once per
# lineage" rested on the receipt's record alone, which a build without this branch drops, and a copy of a
# switched checkpoint whose spread had drifted to 10.01 switched again: Adam restarted the encoder). The
# dead state is 74,425 (96 of 96 post-deploy checkpoints), 74 times above the line; right after the switch
# the spread is 18.2; a bias number moves by about one learning rate (3e-4) per update and by at most 7.3
# (Adam's largest step, above), so a switched bias needs more than (1,000 - 18) / (7.3 x 3e-4) = 448,000
# updates in one direction to reach the line (the whole lineage has run 411,150); a freshly built encoder's
# bias has a spread of about 0.04 (the default initialisation, uniform within 1 / sqrt(192)). Between the
# eps floor and this line nothing is divided and the load says so with the spread (an encoder held down by
# a smaller constant is named, not switched).
REPAIR_SWITCH_LINE = 1000.0
# THE LINE THE SWITCH'S OWN CHECK HOLDS THE TOKEN TO: about ten float32 units in the last place of the
# token's largest value, 2e-5 for values up to 16 (the vision audit's start limit,
# docs/gain_of_function_v4/agents/vision_regression.md section 5a). It is checked in float64 (the switch's
# own arithmetic), float32 and bf16 under autocast (the precisions the run computes the token in), each
# before against after: a change past the line means the layer is not followed by a scale-free
# normalisation, and the switch restores every value and refuses.
REPAIR_CHECK_LINE = 2e-5
REPAIR_CHECK_SCALE = 16.0
REPAIR_CHECK_PRECISIONS = ("float64", "float32", "bfloat16")
# the probe the check reads: 64 draws of the normaliser's contract (each field centred, unit spread,
# bounded at NORMALISED_BOUND) from a private generator, so no random stream of the run moves
REPAIR_PROBE_DRAWS = 64
# LEARNING PROGRESS AND THE NEXT-MOMENT NORMALISATION AT THE SWITCH (review findings 5 and 1). Both hold
# running averages of this sense taken on the dead token: learning progress's persistence averages are
# 1.2e-20 (200-update) and 1.1e-13 (2,000-update) at 411,150, the next-moment normalisation's 1.1e-13 over
# 66,723 readings. Left in place, the first update in which the token moves sends learning progress's
# vestibular term from -1 to +0.66..+0.77 at once (every body's channel up 0.25) and keeps the sense's
# next-moment weight at its cap of 30.7 for 80 to 3,600 updates while the old average catches up (20 to 88
# of every 100 parts of that lesson on this sense). So the switch restarts both: the next-moment average of
# this sense is dropped (its divisor is then its own readings from the switch on) and learning progress
# holds the key out for REPAIR_LP_HOLDOUT_STEPS model steps (the tracker's own set_holdout, which drops
# the dead averages), so its term returns measured on the revived token. 2,000 = learning progress's slow
# horizon (LP_SLOW_STEPS): its slow average is then filled from the revived token alone.
REPAIR_LP_HOLDOUT_STEPS = 2000
# THE REVIVAL'S SHARE OF THE NEXT-MOMENT LESSON (review 2, finding 2). A deploy rehearsal (the first review's dead
# micro checkpoint, 2 bodies, 825 steps) showed the revived token changing per moment far more slowly than a live
# sense (persistence 8.7e-7 to 8.5e-5 against the cap's floor), so the sense's weight sat at its cap of 30.7 on
# every step, while its head, trained on the dead token, met a token drifting away from it: its term took 61 of
# every 100 parts of the next-moment lesson in the first report and 95 of replay's (on dead windows the ring still
# holds), and the other senses' error ratios moved in that report (body position 2.19 against a control's 0.85).
# So while this sense's weight is at its cap, its term (weight x error) is held to at most the mean of the other
# graded senses' terms, live and in replay: its share of either lesson is then at most its equal share, 1 / (1 +
# the other graded senses), 20 in 100 on H84's four. The term is scaled, not clipped, so its gradient keeps its
# direction at the reduced size. Only with --vestibular-encoder-repair on; once the sense's running persistence
# reaches the floor its weight leaves the cap and the plain normalisation applies again.
REVIVAL_CEILING = "the mean of the other graded senses' terms, while this sense's weight is at its cap"


def repair_stamp(mode: str) -> dict:
    """The receipt's record of the repair setting (model_receipt.json 'vestibular_encoder_repair')."""
    return {"version": REPAIR_VERSION, "mode": str(mode), "divisor": "the largest power of two keeping the "
            "spread at or above the eps floor", "eps_step": float(REPAIR_EPS_STEP),
            "switch_line": float(REPAIR_SWITCH_LINE), "check_line": float(REPAIR_CHECK_LINE),
            "check_scale": float(REPAIR_CHECK_SCALE), "check_precisions": list(REPAIR_CHECK_PRECISIONS),
            "lp_holdout_steps": int(REPAIR_LP_HOLDOUT_STEPS), "revival_ceiling": REVIVAL_CEILING,
            "encoder_precision": "float32" if str(mode) in REPAIR_FP32_MODES else "the run's"}


def _fp32_forward(self, x):
    """The vestibular encoder's forward outside autocast, in float32 (mode rescale_restart_fp32): bound to the
    module instance, so a deep copy rebinds it to the copy and the state dict is unchanged."""
    import torch
    dev = x.device.type if torch.is_tensor(x) else "cpu"
    with torch.autocast(device_type=dev, enabled=False):
        return type(self).forward(self, x.float() if torch.is_tensor(x) else x)


def apply_encoder_precision(model, mode: str) -> bool:
    """With a float32 mode, make the vestibular encoder compute in float32 whatever the run's autocast
    (the registry's output LayerNorm already runs in float32 under autocast). Returns whether it was applied.
    Nothing is saved: the build with the mode applies it at every construction."""
    import types
    if str(mode) not in REPAIR_FP32_MODES:
        return False
    enc = model.encoder_registry.encoders[MODALITY]
    enc.forward = types.MethodType(_fp32_forward, enc)
    enc.vestibular_repair_fp32 = True
    return True


def revival_ceiling(weights: dict, errors: dict, capped, lesson_weight: float = 1.0) -> tuple:
    """The live lesson's weights with this sense's held to REVIVAL_CEILING: (weights, factor), factor None when
    the ceiling does not apply this step (the sense ungraded or not at its cap), else the factor its weight was
    multiplied by (1.0 when its term was already at or under the ceiling). errors: the plain per-sense errors
    the weights multiply (floats already read). lesson_weight: the sense's own weight in the lesson
    (--vestibular-lesson-weight), which the live lesson multiplies in after these weights (third review, section
    D): the ceiling judges the term the lesson carries, lesson_weight x weight x error, so the sense's share is
    held to its equal share at any lesson weight. The returned weights leave the lesson weight out, as the live
    lesson applies it itself; lesson_weights_used gives the weights the lesson carries in total."""
    if MODALITY not in weights or MODALITY not in set(capped or ()):
        return weights, None
    others = [float(weights[m]) * float(errors[m]) for m in weights
              if m != MODALITY and m in errors and math.isfinite(float(errors[m]))]
    tv = float(lesson_weight) * float(weights[MODALITY]) * float(errors.get(MODALITY, float("nan")))
    if not others or not math.isfinite(tv):
        return weights, None
    ceiling = sum(others) / len(others)
    if not (tv > ceiling >= 0.0):
        return weights, 1.0
    factor = ceiling / tv
    out = dict(weights)
    out[MODALITY] = float(weights[MODALITY]) * factor
    return out, factor


def lesson_weights_used(weights: dict, lesson_weight: float = 1.0) -> dict:
    """The weight each sense's term carries in the lesson in total: `weights` (after the ceiling) with this sense's
    multiplied by its lesson weight. What the share gauge (next_frame_norm share_normalized) must read (third
    review, section D: at a lesson weight of 2.0 the optimizer saw 33 in 100 while the gauge said 20)."""
    if MODALITY not in weights or float(lesson_weight) == 1.0:
        return weights
    return dict(weights, **{MODALITY: float(weights[MODALITY]) * float(lesson_weight)})


def revival_ceiling_terms(senses, terms, weights: dict, capped, lesson_weight: float = 1.0):
    """The replay lesson's weighted terms with this sense's held to REVIVAL_CEILING, on the device (no read back
    to the processor): (weighted terms in the order of `senses`, the factor tensor or None). This sense's term
    carries its lesson weight here (with the repair on, replay weights the sense as the live lesson does, and the
    ceiling judges the weighted term)."""
    import torch
    lw = float(lesson_weight)
    weighted = [t * weights[m] * lw if (m == MODALITY and lw != 1.0) else t * weights[m]
                for t, m in zip(terms, senses)]
    senses = list(senses)
    if MODALITY not in senses or MODALITY not in set(capped or ()) or len(senses) < 2:
        return weighted, None
    k = senses.index(MODALITY)
    others = torch.stack([w for i, w in enumerate(weighted) if i != k]).detach()
    tv = weighted[k].detach()
    factor = torch.clamp(others.mean() / tv.clamp_min(1e-30), max=1.0)
    weighted[k] = weighted[k] * factor
    return weighted, factor


# THE LESSON WEIGHT'S RULE (the Sept 23 balance-sense lane: branch fix/gof-v3-model-b2 at 57eac94f8,
# docs/gain_of_function_v3/agents/lane-b2-vestibular.md, "DECLARED BEFORE THE AFTER-FIX SHARE RUN (04:06)"). The weight
# stays 1.0 if the sense's share of the next-moment lesson is no larger than an equal share per graded sense (m = 1 /
# the number of graded senses: 20 in 100 with vision, touch, hearing, proprioception and this sense); otherwise it is
# the weight that brings the share to that equal share, w = m (1 - s) / ((1 - m) s), s the measured share. The lane
# declared it for a 2-body run of the plain sum; H84 trains the persistence-normalised lesson, so it is applied to that
# lesson's shares (next_frame_norm share_normalized and replay_share_normalized, which with the repair on read the
# weight the lesson carries) on the repaired run, 8 bodies, over the 20 full reports ending 10,000 updates after the
# switch, from the reports in which the sense's weight never sat at its cap (while it does, the ceiling holds the share
# at m by construction). The larger of the live and replay shares is s: one weight governs both lessons, and the rule
# then gives the sense at most its equal share in either. The kill-line watcher prints the weight it gives
# (docs/gain_of_function_v4/vestibular_repair_watch.py, reading "the lesson weight"); the coordinator sets it at the
# watch's end (--vestibular-lesson-weight) and LESSON_WEIGHT_MEASUREMENT then names the reading. One decision rides on
# it, the operator's: under the normalisation a sense predicted worse than the others takes more of the lesson by
# design, and this rule takes that back for this sense.
def lesson_weight_rule(share, graded, weight_in_use=1.0) -> Optional[float]:
    """The lesson weight the rule gives from the sense's measured share `share` of the lesson (a fraction), read at
    the weight `weight_in_use`, with `graded` graded senses: the weight in use when the share is at most the equal
    share; else the weight that brings the share to it (w0 m (1 - s) / ((1 - m) s): at w0 = 1.0 the lane's formula).
    None when the share, the count or the weight cannot be read."""
    try:
        s, n, w0 = float(share), int(graded), float(weight_in_use)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(s) and 0.0 < s < 1.0 and n >= 2 and math.isfinite(w0) and w0 > 0.0):
        return None
    m = 1.0 / n
    if s <= m:
        return w0
    return w0 * m * (1.0 - s) / ((1.0 - m) * s)


def _spread64(t) -> float:
    """The standard deviation across a tensor's numbers (the population one, as the LayerNorm reads it)."""
    return float(t.detach().double().std(unbiased=False))


def repair_spread_floor(eps: float) -> float:
    """The smallest spread at which the LayerNorm's eps changes a token by at most half a float32 step:
    eps / (2 spread^2) <= 2^-24, so spread >= sqrt(eps / 2^-23). 9.16 at eps 1e-5."""
    return math.sqrt(float(eps) / (2.0 * REPAIR_EPS_STEP))


def repair_divisor(spread: float, eps: float) -> float:
    """The largest power of two s with spread / s at or above repair_spread_floor(eps); 1.0 when the spread is
    already below twice the floor (nothing to divide)."""
    floor = repair_spread_floor(eps)
    if not spread >= 2.0 * floor:
        return 1.0
    return float(2.0 ** math.floor(math.log2(spread / floor)))


def _probe_tokens(model, probe, precision: str = "float64"):
    """The vestibular token of each probe row from the parameters as they are now, in one precision:
    'float64' (the switch's own arithmetic), 'float32', or 'bfloat16' (autocast on the parameters' device
    type, as the run computes it: the Linear layers in bf16, the LayerNorms in float32). The encoder's own
    layers (dropout left out: it is identity at evaluation) and the registry's output norm. Returns float64."""
    import contextlib
    import torch
    import torch.nn.functional as F
    enc = model.encoder_registry.encoders[MODALITY]
    norm = model.encoder_registry.output_norm
    if precision == "float64":
        h = probe.double()
        for mod in enc.mlp.children():
            if isinstance(mod, torch.nn.Linear):
                h = h @ mod.weight.double().T + (mod.bias.double() if mod.bias is not None else 0.0)
            elif isinstance(mod, torch.nn.LayerNorm):
                h = F.layer_norm(h, mod.normalized_shape,
                                 None if mod.weight is None else mod.weight.double(),
                                 None if mod.bias is None else mod.bias.double(), mod.eps)
            elif isinstance(mod, torch.nn.Dropout):
                continue
            else:
                h = mod(h)
        return F.layer_norm(h, norm.normalized_shape,
                            None if norm.weight is None else norm.weight.double(),
                            None if norm.bias is None else norm.bias.double(), norm.eps)
    dev = next(enc.parameters()).device
    ctx = (torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if precision == "bfloat16"
           else contextlib.nullcontext())
    with ctx:
        h = probe.float()
        for mod in enc.mlp.children():
            if isinstance(mod, torch.nn.Dropout):
                continue
            h = mod(h)
        tok = norm(h)
    return tok.double()


def repair_encoder(model, load_optimizer: bool = True, restart_moments: bool = False) -> dict:
    """Divide the vestibular encoder's last layer by s = the largest power of two keeping its bias's spread
    at or above the eps floor (repair_divisor), when that spread is above REPAIR_SWITCH_LINE, and move that
    layer's Adam moments with it (first times s, second times s squared); with restart_moments (mode
    rescale_restart) every tensor of the encoder instead drops its Adam state and restarts.
    Function-preserving: in float64, float32 and bf16 the token changes by at most REPAIR_CHECK_LINE x
    max(1, largest / REPAIR_CHECK_SCALE) on the probe, or every value is put back and ValueError is raised.
    A bias whose spread is at most the switch line is left alone. Returns the switch's record (what was
    divided, by what, and what the check read in each precision)."""
    import torch
    enc = model.encoder_registry.encoders[MODALITY]
    last = enc.mlp[-1]
    if not isinstance(last, torch.nn.Linear) or last.bias is None:
        raise ValueError("the vestibular encoder's last layer is not a Linear with a bias; "
                         "--vestibular-encoder-repair has nothing to divide")
    norm = model.encoder_registry.output_norm
    if not isinstance(norm, torch.nn.LayerNorm) or norm.weight is not None or norm.bias is not None:
        raise ValueError("no LayerNorm without gain or bias follows the encoders (encoder_defaults."
                         "output_layernorm off, or made affine): dividing the last layer would change "
                         "every vestibular token")
    W, b = last.weight, last.bias
    spread = _spread64(b)
    floor = repair_spread_floor(norm.eps)
    rec = {"version": REPAIR_VERSION, "switch_line": float(REPAIR_SWITCH_LINE), "spread_floor": floor,
           "bias_spread_before": spread, "bias_size_before": float(b.detach().double().norm()),
           "bias_mean_before": float(b.detach().double().mean()),
           "weight_size_before": float(W.detach().double().norm())}
    if not spread > REPAIR_SWITCH_LINE:
        rec.update(switched=False, why=(
            "the bias's spread %.6g is not above the switch line %g: nothing is divided%s"
            % (spread, REPAIR_SWITCH_LINE,
               "" if spread <= 2.0 * floor else (
                   " (the weights' part and every gradient still reach the token divided by %.4g, %.4g "
                   "times more than at the floor %.3g)" % (spread, spread / floor, floor)))))
        return rec
    s = repair_divisor(spread, norm.eps)
    g = torch.Generator().manual_seed(0)
    probe = torch.randn(REPAIR_PROBE_DRAWS, enc.mlp[0].in_features,
                        generator=g).clamp(-NORMALISED_BOUND, NORMALISED_BOUND)
    probe = probe.to(W.device)
    opt = getattr(model, "optimizer", None) if load_optimizer else None
    params = (("mlp.%d.weight" % (len(enc.mlp) - 1), W), ("mlp.%d.bias" % (len(enc.mlp) - 1), b))
    with torch.no_grad():
        before = {p: _probe_tokens(model, probe, p) for p in REPAIR_CHECK_PRECISIONS}
        kept = {n: p.detach().clone() for n, p in params}
        kept_state = {}
        for n, p in params:
            st = opt.state.get(p) if opt is not None else None
            if st:
                kept_state[n] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in st.items()}
        moved = []
        for n, p in params:
            p.copy_((p.detach().double() / s).to(p.dtype))
            st = opt.state.get(p) if opt is not None else None
            if st and not restart_moments:
                for key, factor in (("exp_avg", s), ("exp_avg_sq", s * s)):
                    v = st.get(key)
                    if torch.is_tensor(v):
                        v.copy_((v.double() * factor).to(v.dtype))
                moved.append(n)
        after = {p: _probe_tokens(model, probe, p) for p in REPAIR_CHECK_PRECISIONS}
        checks = {}
        for p in REPAIR_CHECK_PRECISIONS:
            largest = float(before[p].abs().max())
            line = REPAIR_CHECK_LINE * max(1.0, largest / REPAIR_CHECK_SCALE)
            change = float((after[p] - before[p]).abs().max())
            checks[p] = dict(change_max_abs=change, largest=largest, line=line, within=change <= line)
        bad = [p for p, c in checks.items() if not c["within"]]
        if bad:
            for n, p in params:
                p.copy_(kept[n])
                if n in kept_state:
                    opt.state[p] = kept_state[n]
            raise ValueError("vestibular encoder repair refused: dividing the last layer by %.6g changed the "
                             "token past its line in %s (%s); every value was put back"
                             % (s, ", ".join(bad), "; ".join("%s %.3g against %.3g" % (
                                 p, checks[p]["change_max_abs"], checks[p]["line"]) for p in bad)))
        restarted = []
        if restart_moments and opt is not None:
            # every tensor of the encoder restarts Adam: its state is dropped, so the next update applies
            # Adam's start-up correction from step 1
            for n, p in enc.named_parameters():
                if opt.state.pop(p, None) is not None:
                    restarted.append(n)
    rec.update(switched=True, divided_by=float(s), bias_spread_after=_spread64(b),
               bias_size_after=float(b.detach().double().norm()),
               weight_size_after=float(W.detach().double().norm()),
               moments_moved=moved, moments_restarted=restarted,
               moment_factors=({"exp_avg": float(s), "exp_avg_sq": float(s * s)} if moved else None),
               optimizer_loaded=opt is not None,
               probe_rows=int(REPAIR_PROBE_DRAWS), checks=checks,
               probe_token_change_max_abs=max(c["change_max_abs"] for c in checks.values()),
               check_line=min(c["line"] for c in checks.values()),
               layernorm_eps_share_predicted=float(norm.eps) / (2.0 * _spread64(b) ** 2))
    return rec


def repair_message(rec: dict, step: int) -> str:
    """The load's log line for a switch that happened."""
    if rec.get("moments_restarted"):
        adam = ("the encoder's %d Adam states were dropped, so Adam restarts them with its start-up "
                "correction" % len(rec["moments_restarted"]))
    elif rec.get("moments_moved"):
        adam = ("%d Adam moments moved with them (first x %.6g, second x %.6g)"
                % (len(rec["moments_moved"]), rec["moment_factors"]["exp_avg"],
                   rec["moment_factors"]["exp_avg_sq"]))
    elif not rec.get("optimizer_loaded"):
        adam = "no Adam state was moved or restarted: this load read no optimizer"
    else:
        adam = "no Adam state was moved or restarted: the encoder's last layer had none saved"
    checks = rec.get("checks") or {}
    return ("vestibular encoder repair switched at step %d: the last layer's bias had a spread of %.6g "
            "(size %.6g), so the registry's LayerNorm divided the weights' part of the token and every "
            "gradient by that much; weight and bias were divided by %.6g, a power of two (bias spread now "
            "%.6g), and %s. The token changed by at most %s on the %d-row probe (line %.3g)."
            % (step, rec["bias_spread_before"], rec["bias_size_before"], rec["divided_by"],
               rec["bias_spread_after"], adam,
               ", ".join("%.3g in %s" % (c["change_max_abs"], p) for p, c in checks.items()) or "unread",
               rec["probe_rows"], rec["check_line"]))


def repair_runtime_restart(model) -> dict:
    """After the runtime state is restored at a load that switched (LemLite.
    _vestibular_encoder_repair_after_runtime): restart this sense's next-moment normalisation average and hold
    it out of learning progress (the REPAIR_LP_HOLDOUT_STEPS note above). Returns what was done, with the
    learning-progress channel's step at the switch per body (the vestibular term's leaving, measured from the
    tracker's own state before and after)."""
    out = {}
    nf = getattr(model, "nf_norm", None)
    if nf is not None:
        old = nf.stats.pop(MODALITY, None)
        out["next_frame_norm"] = (
            "no running average of this sense was held" if old is None else
            dict(dropped_average=(float(old[0]) / float(old[1]) if old[1] > 0 else None),
                 dropped_readings=int(old[2]),
                 now="the sense's divisor is its own readings from the switch on"))
    wc = getattr(model, "wellness_channels", None)
    lp = getattr(wc, "lp", None) if wc is not None else None
    if lp is not None:
        before = _lp_channel(lp)
        lp.set_holdout([MODALITY], REPAIR_LP_HOLDOUT_STEPS)
        after = _lp_channel(lp)
        step = None
        if before is not None and after is not None:
            step = [None if not (math.isfinite(a) and math.isfinite(b)) else round(a - b, 6)
                    for a, b in zip(after, before)]
        out["learning_progress"] = dict(held_out_steps=int(REPAIR_LP_HOLDOUT_STEPS),
                                        channel_before=before, channel_after=after,
                                        channel_step_per_body=step)
    return out


def _lp_channel(lp) -> Optional[list]:
    """Learning progress's channel per body from the tracker's own state (LearningProgressTracker.channel),
    or None when the tracker holds no body."""
    for s in (lp.state or {}).values():
        B = int(s["mf"].shape[0])
        return [float(x) for x in lp.channel(B).tolist()]
    return None


# ------------------------------------------------------------------------------ the episode
def _is_old_companion(learner: dict) -> bool:
    data, objects = learner.get("data") or {}, learner.get("objects") or {}
    rep = objects.get("replay") or {}
    return (REPLACED in (data.get("token_ring") or {})
            or REPLACED in (rep.get("mods") or [])
            or REPLACED in (learner.get("placeholders") or {}))


def migrate_learner_state(learner: dict, model) -> Optional[dict]:
    """Make an intermind episode companion restorable into a vestibular build (in place).

    learner: continuation_state's saved learner dict (episode_resume['learner']), BEFORE
    continuation_state.restore_learner. Returns what was done, or None when nothing was needed
    (the build has no vestibular sense, or the companion already has it)."""
    if MODALITY not in getattr(model, "active", ()) or not _is_old_companion(learner):
        return None
    import torch
    data = learner.setdefault("data", {})
    objects = learner.setdefault("objects", {})
    cleared = []
    # per-moment histories whose entries are {sense: token} dicts, and the ones kept aligned
    # with them by length: emptied together (the imagination chain needs 4 moments again)
    for name in ("enc_history", "enc_history_raw", "valid_history", "motor_history",
                 "ctx_history", "ach_history", "memory_bias_history",
                 "goal_ring", "_goal_ring_w0", "_goal_ring_cq", "_goal_ring_drive"):
        v = data.get(name)
        if isinstance(v, deque):
            if len(v):
                cleared.append(f"{name} ({len(v)})")
            data[name] = deque(maxlen=v.maxlen)
    # one cached moment keyed by sense name: the delayed lesson skips one step, the pending
    # order-end lesson skips one order
    for name in ("_nf_raw", "_nf_tokens", "_nf_mask", "_nf_pred", "_tenure_pending",
                 "_last_plan_inputs", "_last_source_mask", "last_tokens", "last_tokens_fed",
                 "last_predictions", "_gl_held_tok"):
        if data.get(name) is not None:
            cleared.append(name)
            data[name] = None
    # goal locality's held goal and its distance scales die with the goal ring (LemLite._gl_clear)
    for name in ("_gl_held_born", "_gl_held_mask", "_gl_reach_ema", "_gl_instant_ema"):
        if name in data:
            data[name] = None
    # the collapse watch: the other senses keep their history, the new sense starts empty
    tr = data.get("token_ring")
    if isinstance(tr, dict):
        data["token_ring"] = {m: tr.get(m, deque(maxlen=200)) for m in model.active}
    # the top-down gate's error averages, by prediction-head position: the slot's column was
    # never updated for a placeholder (the gate skips a sense that is not real); zeroed so the
    # vestibular gate's slow baseline seeds from its own first reading
    k_pred = model.pred_mods.index(MODALITY)
    for name in ("_td_err_ema", "_td_err_slow"):
        t = data.get(name)
        if torch.is_tensor(t) and t.dim() == 2 and t.shape[1] == len(model.pred_mods):
            t = t.clone()
            t[:, k_pred] = 0.0
            data[name] = t
    # a consequence lesson holds values aligned with the goal ring just emptied: the model's own
    # fresh one is kept instead of the saved one
    if objects.pop("consequence_lesson", None) is not None:
        cleared.append("consequence_lesson")
    report = {"cleared": cleared}
    rep = objects.get("replay")
    if rep is not None and REPLACED in (rep.get("mods") or []):
        k = rep["mods"].index(REPLACED)
        renamed = [MODALITY if m == REPLACED else m for m in rep["mods"]]
        if renamed != list(model.active):
            raise ValueError(f"the saved replay ring's senses {rep['mods']} do not map onto this "
                             f"build's {list(model.active)}")
        rep["mods"] = renamed
        val = rep.get("valid")
        n_stored = int(min(int(rep.get("count", 0)), int(rep.get("size", 0) or 0))) * int(
            rep.get("n_envs", 1))
        if torch.is_tensor(val):
            report["replay_slot_valid_before"] = int(val[..., k].sum())
            val = val.clone()
            val[..., k] = False       # no stored moment may grade the vestibular head
            rep["valid"] = val
        report["replay_moments_kept_for_other_senses"] = n_stored
        report["replay_slot"] = k
    ph = learner.get("placeholders")
    if isinstance(ph, dict) and ph.pop(REPLACED, None) is not None:
        cleared.append(f"placeholder generator '{REPLACED}'")
    ctx = data.get("context_buffer")
    if torch.is_tensor(ctx):
        report["context_moments_with_old_slot"] = int(ctx.shape[1] // max(1, model.n_mod))
    return report


# ------------------------------------------------------------------------------ the stats block
def trunk_roll_pitch_deg(trunk_up: np.ndarray) -> tuple:
    """[B, 3] world-up seen from the chest -> (roll, pitch) in degrees, body_orientation's
    convention: roll 0 = lying on the back, +-90 = on a side, +-180 = face down; pitch 0 = the
    trunk horizontal, +90 = head straight up."""
    up = np.asarray(trunk_up, dtype=np.float64).reshape(-1, 3)
    up = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    along, front, side = (np.asarray(CHEST_AXES) @ up.T)
    return (np.degrees(np.arctan2(side, front)),
            np.degrees(np.arcsin(np.clip(along, -1.0, 1.0))))


def head_tilt_deg(acc: np.ndarray) -> np.ndarray:
    """[B, 3] head accelerometer -> the angle between where gravity pulls in the head now and
    where it pulls lying still on the back, in degrees: 0 = that resting orientation, 90 = a
    quarter turn (face to one side, or the head upright as when sitting), 180 = upside down from
    it (face down). Movement adds its own acceleration, so a window mean is the reading."""
    a = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(a @ np.asarray(HEAD_GRAVITY_REST), -1.0, 1.0)))


def balance_labels(raw: np.ndarray) -> tuple:
    """[B, VESTIBULAR_DIM] raw vestibular vectors -> ([B, 5] balance state, [B] defined), the balance probe's
    quantities: where gravity pulls in the head (the head accelerometer's unit direction, 3 numbers; head_tilt_deg
    is its angle from the resting direction) and the trunk's roll as its sine and cosine (trunk_roll_pitch_deg's
    roll is their angle: 0 = on the back, +-90 = on a side, +-180 = face down). Direction vectors, not the angles
    themselves, because the angles fold and wrap where the bodies lie: |roll| folds at 0, where a body lying on
    its back sits, and the signed roll jumps from +180 to -180 where a body lying face down sits, so no linear
    read of a token can follow either there. Undefined when the accelerometer reads nothing or the trunk stands
    straight up or down (its roll has no direction)."""
    raw = np.asarray(raw, dtype=np.float64).reshape(len(raw), -1)
    acc = raw[:, 0:3]
    na = np.linalg.norm(acc, axis=1)
    head = acc / np.maximum(na, 1e-9)[:, None]
    up = raw[:, 15:18]
    up = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    along, front, side = np.asarray(CHEST_AXES) @ up.T
    nr = np.hypot(side, front)
    roll = np.stack([side, front], 1) / np.maximum(nr, 1e-9)[:, None]
    ok = (na > 1e-6) & (nr > 1e-3) & np.isfinite(acc).all(1) & np.isfinite(raw[:, 15:18]).all(1)
    return np.nan_to_num(np.concatenate([head, roll], 1)), ok


class BalanceProbe:
    """Does the vestibular token carry the body's balance state? (--vestibular-encoder-repair's gauge; review
    finding 4: every other line judged that the token VARIES, which the anti-collapse lesson alone produces
    whatever the input means.)

    A linear read, fitted by ridge regression, from the raw vestibular token (192 numbers) to the balance
    state of the same frame (balance_labels): HEAD_TILT, where gravity pulls in the head (3 numbers), and
    TRUNK_ROLL, the trunk's roll as its sine and cosine (2 numbers); each read as one R squared over its numbers.
    WITHIN EACH BODY: the token and the state are taken as each body's change from its own mean over the report
    (100 steps), so the read is how a body's token follows that body's own turning. What differs between bodies
    is left out, for two reasons measured on H84's layout (body 2 lies face down, the other seven on their backs;
    scratchpad vfix/probe_h84_layout.py, 2026-09-25): a read across bodies of a token that carries the state
    exactly scored the trunk's roll at -2.4 when the bodies wobbled 10 degrees, because no fit on the bodies lying
    on their backs can place the one lying face down; and any token that only tells the bodies apart fits every
    body's steady posture (8 bodies, 192 numbers), so an across-body read cannot tell balance from identity.
    HELD OUT: fitted on the even-numbered bodies and scored on the odd ones, then the other way round (one encoder
    maps the same change to the same token change in every body). CONTROL: the same fit and score with each
    body's token paired with the previous body's state at the same moment (the body-shuffled pairing): the
    bodies move independently, so what that scores is not balance information in the token. Read as R squared
    on the held-out bodies: 1 = every held-out change predicted exactly, 0 = no better than predicting no
    change, below 0 = worse. A constant (dead) token predicts no change: 0. Needs at least two bodies.
    Sufficient statistics accumulate on the device every step (one 192 x 192 product per body group and each
    body's sums; no read back to the processor) and are read at the report, over the last HORIZON_REPORTS
    reports."""

    LABELS = {"head_tilt": (0, 1, 2), "trunk_roll": (3, 4)}
    # 10 reports of 100 steps x 8 bodies: 4,000 moments per fit, 21 per coefficient (192 of them)
    HORIZON_REPORTS = 10
    # the ridge's strength as a share of the fit's mean diagonal: the token's 192 numbers sum to zero (the
    # registry's LayerNorm), so the plain normal equations are singular; 1e-3 of the mean diagonal makes the
    # solve well posed while shrinking a coefficient of a direction the data holds by at most 1 part in 1,000
    RIDGE_SHARE = 1e-3

    def __init__(self):
        self.done = deque(maxlen=self.HORIZON_REPORTS)
        self.cur = None

    def add(self, tok, labels, valid) -> None:
        """tok [B, d] (the device's), labels [B, 5] (numpy, balance_labels), valid [B] bool (numpy)."""
        import torch
        B = int(tok.shape[0])
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        if B < 2 or valid.size != B:
            return
        both = valid & np.roll(valid, 1)              # a body and the body whose state it borrows (control)
        if not both.any():
            return
        dev = tok.device
        x = tok.detach().reshape(B, -1).double()
        y = torch.as_tensor(np.asarray(labels, dtype=np.float64).reshape(B, -1), device=dev)
        ys = y.roll(1, dims=0)                        # body b paired with body b - 1's state
        if self.cur is None or self.cur["B"] != B:
            if self.cur is not None:
                self.flush()
            D, L = x.shape[1], y.shape[1]
            z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)      # noqa: E731
            self.cur = dict(B=B, XtX=z(2, D, D), Xty=z(2, 2, D, L), yy=z(2, 2, L), sx=z(B, D), sy=z(2, B, L),
                            nb=z(B))
        c = self.cur
        w = torch.as_tensor(both.astype(np.float64), device=dev)[:, None]
        par = np.arange(B) % 2
        for g in (0, 1):
            m = torch.as_tensor((both & (par == g)).astype(np.float64), device=dev)[:, None]
            xg = x * m
            c["XtX"][g] += xg.T @ xg
            for p, yy in ((0, y), (1, ys)):
                c["Xty"][g, p] += xg.T @ yy
                c["yy"][g, p] += (yy * yy * m).sum(0)
        c["sx"] += x * w
        c["sy"][0] += y * w
        c["sy"][1] += ys * w
        c["nb"] += w[:, 0]

    def flush(self) -> None:
        """Close the report: its statistics become each body's changes from its own mean over the report."""
        c, self.cur = self.cur, None
        if c is None:
            return
        import torch
        nb = c["nb"]
        inv = torch.where(nb > 0, 1.0 / nb.clamp_min(1.0), torch.zeros_like(nb))
        XtX, Xty, yy = c["XtX"].clone(), c["Xty"].clone(), c["yy"].clone()
        n = torch.zeros(2, dtype=torch.float64, device=nb.device)
        par = torch.arange(c["B"], device=nb.device) % 2
        for g in (0, 1):
            k = (par == g).double()[:, None] * inv[:, None]          # 1 / n_b for this group's bodies
            sx = c["sx"]
            XtX[g] -= (sx * k).T @ sx
            for p in (0, 1):
                sy = c["sy"][p]
                Xty[g, p] -= (sx * k).T @ sy
                yy[g, p] -= (sy * sy * k).sum(0)
            n[g] = (nb * (par == g).double()).sum()
        self.done.append(dict(XtX=XtX, Xty=Xty, yy=yy.clamp_min(0.0), n=n))

    def read(self) -> Optional[dict]:
        """R squared on the held-out bodies' changes over the last HORIZON_REPORTS reports, for the true
        pairing and the body-shuffled one, per quantity; None before any report holds a moment."""
        import torch
        if not self.done:
            return None
        s = {k: sum(c[k] for c in self.done) for k in ("XtX", "Xty", "yy", "n")}
        n = s["n"]
        if not bool((n > 1).all()):
            return dict(samples=int(n.sum()), r2=None, r2_shuffled=None,
                        why="one of the two body groups holds no moment yet")
        D = s["XtX"].shape[-1]
        eye = torch.eye(D, dtype=torch.float64, device=n.device)
        out = {}
        for p, name in ((0, "r2"), (1, "r2_shuffled")):
            sse = torch.zeros(s["yy"].shape[-1], dtype=torch.float64, device=n.device)
            sst = torch.zeros_like(sse)
            for g in (0, 1):
                t = 1 - g
                A = s["XtX"][g]
                lam = self.RIDGE_SHARE * float(torch.diagonal(A).mean()) + 1e-12
                beta = torch.linalg.solve(A + lam * eye, s["Xty"][g, p])                   # [D, L]
                sse += s["yy"][t, p] - 2.0 * (beta * s["Xty"][t, p]).sum(0) + (beta * (s["XtX"][t] @ beta)).sum(0)
                sst += s["yy"][t, p]
            vals = {}
            for lab, comps in self.LABELS.items():
                idx = list(comps)
                tot = float(sst[idx].sum())
                vals[lab] = (None if not tot > 1e-12 * float(n.sum()) else
                             round(1.0 - float(sse[idx].sum()) / tot, 4))
            out[name] = vals
        within = s["yy"][:, 0].sum(0) / float(n.sum())
        out.update(samples=int(n.sum()), reports=len(self.done),
                   within_body_turn_deg={lab: round(math.degrees(math.sqrt(max(0.0, float(within[list(c)].sum())))), 2)
                                         for lab, c in self.LABELS.items()})
        return out


class VestibularWindow:
    """The `vestibular` block of the stats record (DESIGN.md section 3), one report window.

    add() after every model step with the frames the model just read; summary() at the flush,
    which also starts a new window. Everything is a window mean over the real reads."""

    SCALE = {
        "pred_err": ("mean squared distance between the predicted and the actual next vestibular "
                     "token; 0 = exact, persist_err = the score of predicting no change"),
        "improvement_over_persistence": ("1 - pred_err / persist_err: 0 = no better than "
                                         "'nothing changes', 1 = exact, negative = worse"),
        "head_tilt_deg": ("angle of the head from its lying-still-on-the-back orientation, by "
                          "gravity: 0 = that orientation, 90 = a quarter turn, 180 = upside down"),
        "trunk_roll_deg": ("how far the trunk has rolled from lying on its back, either way: "
                           "0 = on the back, 90 = on a side, 180 = face down"),
        "trunk_pitch_deg": "0 = trunk horizontal, 90 = head straight up, -90 = head straight down",
        "per_body": ("the same three per body; trunk_roll_deg there is signed (+ and - are the two "
                     "sides) and is the direction average over the window"),
        "support_trunk": "share of outside support carried by the trunk (0-1)",
        "limit_load": ("share of motor-driven joints within %.1f degrees of a limit (0-1)"
                       % math.degrees(LIMIT_BAND_RAD)),
        "clipped_share": ("share of real raw values the input bound changed (0 = none). The bound "
                          "is 4 in each field's own unit: 4 g for the head accelerometer, 4 body "
                          "weights for support_total, and 4 times the measured 99th percentile "
                          "for the gyro, neck speeds and the two efforts"),
        "lesson_share": ("the vestibular term's share of the next-moment lesson over the window, "
                         "at its declared weight (0 = none of the lesson, 1 = all of it; an "
                         "equal share among five graded senses is 0.2)"),
        "input_clipped_share": ("share of the encoder's centred and scaled inputs that the bound "
                                "of 4 running spreads changed (0 = none)"),
        "input_spread_min": ("the smallest running spread of any field, in that field's unit, "
                             "beside the floor it never falls below"),
        "input_centre_filled_share": ("how much of the running mean's 2,000-moment horizon real "
                                      "frames have filled (0 = no real frame yet, 1 = full)"),
        "liveness": ("per hidden layer of the encoder, the share of its units whose input rose "
                     "above -3 on some moment of the window (0 = a dead layer, 1 = every unit "
                     "took part)"),
    }

    # --vestibular-encoder-repair's gauges: in the record only when the flag is on (the block's own
    # "scale" entry says what each reads), so a default build writes the record the previous code wrote
    REPAIR_SCALE = {
        "token_between_bodies": ("window mean over steps of the spread of the raw vestibular token between the "
                                 "bodies, root mean square over its 192 numbers (0 = every body gets the same "
                                 "token; the five other senses read 0.15-0.53 in the memory's saved events)"),
        "token_change": ("window mean of the root mean square change per number of each body's raw vestibular "
                         "token from one moment to the next, over pairs of real moments (0 = it never changes)"),
        "bias_spread": ("the spread across its 192 numbers of the encoder's last-layer bias at the window's "
                        "end: what the registry's LayerNorm divides the weights' part by (74,425 before the "
                        "repair, 9.2-18.3 right after it: 18.17 on H84)"),
        "weight_rms": "root mean square of the encoder's last-layer weight at the window's end",
        "switch": "the load's switch: the step, what it divided by, and the probe's token change",
        "balance_probe": ("how much of the body's balance state the token carries (BalanceProbe): a linear read, "
                          "from each body's token change, of that body's change of head tilt (where gravity pulls "
                          "in the head) and of trunk roll (its sine and cosine), each change taken from the "
                          "body's own mean over its report, fitted on the even-numbered bodies and scored on the "
                          "odd ones and the other way round, over the last 10 reports; r2 1 = every change "
                          "predicted exactly, 0 = no better than predicting no change, below 0 = worse; "
                          "r2_shuffled = the same read with each body's token paired with another body's state "
                          "(the control); a dead token scores 0; within_body_turn_deg = how far the bodies "
                          "turned within a report (root mean square, degrees): the size of what is read"),
        "revival_ceiling": ("the next-moment lesson's ceiling on this sense while its weight is at its cap "
                            "(REVIVAL_CEILING): live and replay, the steps it applied to (the weight at its cap), "
                            "the steps it held the term down, and the mean factor the weight was multiplied by on "
                            "those (1 = not held down; 0.1 = the term would have been 10 times the other senses' "
                            "mean); the ceiling judges the term with the sense's lesson weight in it"),
        "encoder_precision": "the precision the encoder computes in: float32 with rescale_restart_fp32, else the run's",
        "lesson_weight": ("the sense's own weight in the next-moment lesson (--vestibular-lesson-weight; 1 = counted "
                          "like every other sense), live and in replay; next_frame_norm's shares read it"),
    }

    def __init__(self):
        self._prev_tok = None        # the previous step's raw vestibular token (repair gauges), kept
        self._probe = None           # across windows: a pair of moments straddles the flush, and the
        self._reset()                # balance probe reads the last 10 reports

    def _reset(self):
        self.rep_between = None      # device sums (no read until the flush) of the repair gauges
        self.rep_change = None
        self.rep_n = 0
        self.rep_change_n = 0
        self.rev = {k: [0, 0, 0.0] for k in ("live", "replay")}    # applied, held down, sum of factors held
        self.steps = 0
        self.real_steps = 0
        self.frames_invalid = 0
        self.values = 0
        self.clipped = 0
        self.f_sum = np.zeros(VESTIBULAR_DIM, dtype=np.float64)
        self.f_n = 0
        self.tilt = None
        self.rs = self.rc = self.pitch = self.body_n = None
        self.abs_roll = 0.0          # sum over real body-steps of |roll|, for the scalar
        self.pe = [0.0, 0.0, 0]
        self.live = {}               # layer name -> [units] bool, ever above LIVE_LINE
        self.ls = [0.0, 0.0]         # the sense's weighted term, and every graded term, summed

    # DESIGN section 10: a unit is live on a moment when its input (the layer's output after its
    # LayerNorm, which is what GELU reads) exceeds -3
    LIVE_LINE = -3.0

    def _add_liveness(self, model) -> None:
        """Run the encoder's hidden layers on what it just read and mark the live units."""
        norm = getattr(model, "vestibular_norm", None)
        x = None if norm is None else norm.last_input
        if x is None or MODALITY not in getattr(model, "active", ()):
            return
        import torch
        enc = model.encoder_registry.encoders[MODALITY]
        mods = list(enc.mlp.children())
        p = next(enc.parameters())
        h = torch.as_tensor(x, dtype=p.dtype, device=p.device)
        lin_i = 0
        with torch.no_grad():
            for i, mod in enumerate(mods):
                h = mod(h)
                if isinstance(mod, torch.nn.Linear):
                    lin_i = i
                if isinstance(mod, (torch.nn.LayerNorm, getattr(torch.nn, "RMSNorm", ()))):
                    name = f"{ENC_NEW}mlp.{lin_i}"
                    now = (h.float() > self.LIVE_LINE).reshape(-1, h.shape[-1]).any(dim=0).cpu()
                    prev = self.live.get(name)
                    self.live[name] = now if prev is None or prev.shape != now.shape else (prev | now)

    def add(self, frames, model) -> None:
        B = len(frames)
        raw = np.zeros((B, VESTIBULAR_DIM), dtype=np.float64)
        for i, f in enumerate(frames):
            v = getattr(f, "vestibular", None)
            if v is not None:
                a = np.asarray(v, dtype=np.float32).reshape(-1)
                raw[i] = a if a.size == VESTIBULAR_DIM else pad_legacy_vestibular(a)
        ok = (raw[:, FIELD_INDEX["valid"]] > 0.5) & np.isfinite(raw).all(axis=1)
        self.steps += 1
        mask = getattr(model, "_last_source_mask", None) or {}
        self.real_steps += int(str(mask.get(MODALITY)) == "real")
        self.frames_invalid += int((~ok).sum())
        if self.tilt is None or len(self.tilt) != B:
            self.tilt = np.zeros(B)
            self.rs, self.rc, self.pitch, self.body_n = (np.zeros(B), np.zeros(B), np.zeros(B),
                                                        np.zeros(B))
        if ok.any():
            R = raw[ok]
            self.values += R.size
            self.clipped += int((np.abs(R) > INPUT_BOUND).sum())
            self.f_sum += R.sum(axis=0)
            self.f_n += int(ok.sum())
            idx = np.nonzero(ok)[0]
            tilt = head_tilt_deg(R[:, 0:3])
            roll, pitch = trunk_roll_pitch_deg(R[:, 15:18])
            self.tilt[idx] += tilt
            self.rs[idx] += np.sin(np.radians(roll))
            self.rc[idx] += np.cos(np.radians(roll))
            self.abs_roll += float(np.abs(roll).sum())
            self.pitch[idx] += pitch
            self.body_n[idx] += 1
        audit = getattr(model, "last_prediction_audit", None) or {}
        a = audit.get(MODALITY)
        if a:
            n = int(a.get("transitions", 0) or 0)
            self.pe[0] += n * float(a["prediction_mse"])
            self.pe[1] += n * float(a["persistence_mse"])
            self.pe[2] += n
            # its share of the next-moment lesson (review finding 2: the deploy reads it against
            # the share before the switch); the audit's prediction error is the lesson's term
            v = float(getattr(model, "vestibular_lesson_weight", 1.0)) * float(a["prediction_mse"])
            self.ls[0] += v
            self.ls[1] += v + sum(float(x["prediction_mse"]) for k, x in audit.items()
                                  if k != MODALITY and x)
        self._add_liveness(model)
        self._norm = getattr(model, "vestibular_norm", None)
        if getattr(model, "vestibular_encoder_repair", "off") != "off":
            self._add_repair(model, raw, ok)

    def _add_repair(self, model, raw=None, ok=None) -> None:
        """The repair's two token gauges and the balance probe for this step, summed on the device (read at
        the flush)."""
        tok = (getattr(model, "last_tokens", None) or {}).get(MODALITY)
        self._rep_model = model
        if tok is None:
            self._prev_tok = None
            return
        real = str((getattr(model, "_last_source_mask", None) or {}).get(MODALITY)) == "real"
        t = tok.detach().float().reshape(tok.shape[0], -1)
        if real and t.shape[0] >= 2:
            v = t.std(0, unbiased=False).square().mean().sqrt()
            self.rep_between = v if self.rep_between is None else self.rep_between + v
            self.rep_n += 1
        prev = self._prev_tok
        if real and prev is not None and prev.shape == t.shape:
            c = (t - prev).square().mean(-1).sqrt().mean()
            self.rep_change = c if self.rep_change is None else self.rep_change + c
            self.rep_change_n += 1
        self._prev_tok = t.clone() if real else None
        for key, attr in (("live", "vestibular_revival_live"), ("replay", "vestibular_revival_replay")):
            f = getattr(model, attr, None)
            if f is not None:
                c = self.rev[key]
                c[0] += 1
                if float(f) < 1.0:
                    c[1] += 1
                    c[2] += float(f)
                if key == "replay":
                    # each re-study is counted once: replay runs every 2 live steps and its factor stays set
                    # until the next one (third review, minor: "steps applied" in replay read twice the replays)
                    model.vestibular_revival_replay = None
        if real and raw is not None and ok is not None and len(raw) == t.shape[0]:
            labels, defined = balance_labels(raw)
            if self._probe is None:
                self._probe = BalanceProbe()
            self._probe.add(t, labels, np.asarray(ok, dtype=bool) & defined)

    def _repair_summary(self) -> dict:
        model = self._rep_model
        out = {"token_between_bodies": (round(float(self.rep_between) / self.rep_n, 8)
                                        if self.rep_n else None),
               "token_change": (round(float(self.rep_change) / self.rep_change_n, 8)
                                if self.rep_change_n else None)}
        try:
            last = model.encoder_registry.encoders[MODALITY].mlp[-1]
            out["bias_spread"] = float("%.6g" % _spread64(last.bias))
            out["weight_rms"] = float("%.6g" % float(last.weight.detach().double().square().mean().sqrt()))
        except (AttributeError, KeyError, IndexError, TypeError):
            out["bias_spread"] = out["weight_rms"] = None
        rec = getattr(model, "vestibular_encoder_repair_record", None)
        out["switch"] = (None if not rec else {k: rec.get(k) for k in (
            "step", "divided_by", "bias_spread_before", "bias_spread_after", "probe_token_change_max_abs",
            "check_line")})
        if self._probe is not None:
            self._probe.flush()
            out["balance_probe"] = self._probe.read()
        else:
            out["balance_probe"] = None
        out["revival_ceiling"] = {k: {"steps_applied": c[0], "steps_held_down": c[1],
                                      "factor_mean_held": (round(c[2] / c[1], 6) if c[1] else None)}
                                  for k, c in self.rev.items()}
        enc = None
        try:
            enc = model.encoder_registry.encoders[MODALITY]
        except (AttributeError, KeyError, TypeError):
            pass
        out["encoder_precision"] = ("float32" if getattr(enc, "vestibular_repair_fp32", False) else "the run's")
        lw = getattr(model, "vestibular_lesson_weight", None)
        out["lesson_weight"] = float(lw) if isinstance(lw, (int, float)) else None
        out["scale"] = self.REPAIR_SCALE
        return out

    def summary(self) -> dict:
        def r(x, nd=4):
            return None if x is None or not math.isfinite(float(x)) else round(float(x), nd)
        n = self.body_n if self.body_n is not None else np.zeros(0)
        per = lambda s: [r(v / c, 1) if c else None for v, c in zip(s, n)]   # noqa: E731
        fm = (self.f_sum / self.f_n) if self.f_n else None
        pe, pp, pn = self.pe
        tot = float(n.sum()) if n.size else 0.0
        out = {
            "steps": int(self.steps), "real_steps": int(self.real_steps),
            "frames_invalid": int(self.frames_invalid),
            "pred_err": r(pe / pn, 6) if pn else None,
            "persist_err": r(pp / pn, 6) if pn else None,
            "improvement_over_persistence": (r(1.0 - pe / pp, 4) if pn and pp > 1e-12 else None),
            # scalars over every real body-step of the window (the watcher and the dashboard read
            # these); the per-body readings are under per_body
            "head_tilt_deg": r(float(self.tilt.sum()) / tot, 1) if tot else None,
            "trunk_roll_deg": r(self.abs_roll / tot, 1) if tot else None,
            "trunk_pitch_deg": r(float(self.pitch.sum()) / tot, 1) if tot else None,
            "per_body": (None if self.tilt is None else {
                "head_tilt_deg": per(self.tilt),
                "trunk_roll_deg": [r(math.degrees(math.atan2(s, c)), 1) if k else None
                                   for s, c, k in zip(self.rs, self.rc, n)],
                "trunk_pitch_deg": per(self.pitch)}),
            "support_trunk": r(fm[FIELD_INDEX["support_trunk"]]) if fm is not None else None,
            "limit_load": r(fm[FIELD_INDEX["limit_load"]]) if fm is not None else None,
            "clipped_share": (r(self.clipped / self.values, 6) if self.values else None),
            "lesson_share": (r(self.ls[0] / self.ls[1], 4) if self.ls[1] > 0 else None),
            "fields_mean": ({name: r(fm[i]) for i, name in enumerate(VESTIBULAR_FIELDS)}
                            if fm is not None else None),
            # DESIGN section 10 rule 2: the encoder's live share per hidden layer; it has no
            # attention pool, so pool_spread is empty
            "liveness": ({k: round(float(v.float().mean()), 4) for k, v in self.live.items()}
                         if self.live else None),
            "pool_spread": {},
            "scale": self.SCALE,
        }
        norm = getattr(self, "_norm", None)
        out.update(norm.window_summary() if norm is not None else
                   {"input_centre_filled_share": None, "input_real_body_moments": None,
                    "input_spread_min": None, "input_clipped_share": None})
        if getattr(self, "_rep_model", None) is not None:
            out["repair"] = self._repair_summary()
        self._reset()
        return out
