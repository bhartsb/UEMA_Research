# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Contact head: which skin patches will be touched in the next moment, and by what.

WHY THIS EXISTS (docs/gain_of_function_v3/DESIGN.md, decision D10). MIMo's body can touch
itself: 1,410 pairs of its own parts can collide, both sides feel the touch, and the order that
moved the limb is known. Until now nothing predicted either side of such a touch: the touch
encoder pools every contact row into one vector, and the only self-touch wellness term paid for
own parts ARRIVING, whether or not the arrival was expected (understand-touch-selfcontact.md).
The addendum (section 2) asks for the opposite: weight SUCCESSFUL PREDICTION AND CONTROL of
self-contact above ordinary contact, and never pay for contact merely happening.

THE HEAD. For each of the 53 skin patches (one per sensing body, pooler order, PATCH_NAMES) it
gives three probabilities for the next moment (the 50 ms burst the order drives):
  contact  - anything touches the patch
  self     - given contact, one of the body's own parts is among what touches it
  world    - given contact, something outside the body is among what touches it
so P(own part touches the patch) = contact x self and P(outside thing touches it) = contact x
world. Its inputs are the core's latent for the moment before (the delayed next-frame lesson's
latent for S_t-1 + A_t-1), the executed order GIVEN DIRECTLY (each body handle's asked position
change in cm, asked turn in degrees and strength, relative to where the handle was), and each
patch's own contact state at that moment. The next-moment grading trains the core as well as
the head (DESIGN D10: the efference form of b1's heads).

LIVENESS (DESIGN section 10). The core latent carries a part shared by every moment (norm 66 of
75 on ckpt_step_344400) that pinned every unit of the old tenure pose head below zero. This head
subtracts a running mean of the latent (updated only on lived moments, kept in runtime_state.pt)
and then LayerNorms it (rule 1); every patch has its own path to the output (rule 3); the window
reports the share of each layer's units whose input rose above -3 (rule 2).

THE SUCCESS SIGNAL AND THE PAY live in ContactTracker below. Every constant carries its
measurement or its arithmetic.

VERSION 2 (lane-b4-contact, 2026-09-23, the adversarial review docs/gain_of_function_v3/reviews/
b4-self-contact.md): each patch judges an onset against ITS OWN line (the head's 95th percentile on
that patch's untouched moments), self-touch and outside touch each have their own evidence gate,
an outside touch pays once per body-moment, the shuffled-order control is size-matched, the order
input drops the position requests of handles the body turns but never places, the ranking reads a
per-body running average of the payments, and the window reports the onset AUCs beside their
base-rate control.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# v2: the order input drops the position columns of rotation-only handles (ORDER_INPUT_RULE) and the
# divisors are re-measured on executed orders; a v1 head cannot load into v2 (no run ever saved one:
# checked 2026-09-23, no args.json or checkpoint receipt under Saved/mimo_bridge/experiments names it)
CONTACT_HEAD_VERSION = "contact_head_v2"

# THE 53 SKIN PATCHES, in the touch pooler's order (MIMo sensing bodies sorted by body id; the two
# eyes are not skin). MEASURED 2026-09-22 (b4 scratch patch_names.py -> patch_names.json) from
# BridgeEnv.patch_names() on the H84 segment 31 world (age 24, trunk_v1) and on a legacy age-1
# body: the same 53 names in the same order. The world's contact_patch_state is in this order, and
# a self-contact onset names its two bodies, which this list maps to patches. The live check that
# catches it lying: an onset side whose body is not in the list is counted (unmapped_sides), and a
# report of another length is refused per body (patch_count_mismatch).
PATCH_NAMES: Tuple[str, ...] = (
    "hip", "lower_body", "upper_body", "chest", "right_upper_arm", "right_lower_arm",
    "right_hand", "right_ffknuckle", "right_ffmiddle", "right_ffdistal", "right_mfknuckle",
    "right_mfmiddle", "right_mfdistal", "right_rfknuckle", "right_rfmiddle", "right_rfdistal",
    "right_lfmetacarpal", "right_lfknuckle", "right_lfmiddle", "right_lfdistal", "right_thbase",
    "right_thhub", "right_thdistal", "left_upper_arm", "left_lower_arm", "left_hand",
    "left_ffknuckle", "left_ffmiddle", "left_ffdistal", "left_mfknuckle", "left_mfmiddle",
    "left_mfdistal", "left_rfknuckle", "left_rfmiddle", "left_rfdistal", "left_lfmetacarpal",
    "left_lfknuckle", "left_lfmiddle", "left_lfdistal", "left_thbase", "left_thhub",
    "left_thdistal", "head", "right_upper_leg", "right_lower_leg", "right_foot", "right_big_toe",
    "right_toes", "left_upper_leg", "left_lower_leg", "left_foot", "left_toes", "left_big_toe")
N_PATCHES = len(PATCH_NAMES)
PATCH_INDEX: Dict[str, int] = {n: i for i, n in enumerate(PATCH_NAMES)}


def _region(name: str) -> str:
    """The skin region of a patch, by the prefixes of mimo_bridge/touch_adapter.SKIN_REGIONS (the
    names the world's skin_regions and self_contact_onsets report; a test holds the two equal). The
    window reports false alarms per region (review of b4, finding M1)."""
    for region, prefixes in (
            ("head", ("head",)), ("trunk", ("chest", "upper_body", "lower_body", "hip")),
            ("left hand", ("left_hand", "left_ff", "left_mf", "left_rf", "left_lf", "left_th")),
            ("right hand", ("right_hand", "right_ff", "right_mf", "right_rf", "right_lf", "right_th")),
            ("left lower arm", ("left_lower_arm",)), ("right lower arm", ("right_lower_arm",)),
            ("left upper arm", ("left_upper_arm",)), ("right upper arm", ("right_upper_arm",)),
            ("left upper leg", ("left_upper_leg",)), ("right upper leg", ("right_upper_leg",)),
            ("left lower leg", ("left_lower_leg",)), ("right lower leg", ("right_lower_leg",)),
            ("left foot", ("left_foot", "left_toes", "left_big_toe")),
            ("right foot", ("right_foot", "right_toes", "right_big_toe"))):
        if any(name == p or name.startswith(p) for p in prefixes):
            return region
    return ""


PATCH_REGIONS: Tuple[str, ...] = tuple(_region(n) for n in PATCH_NAMES)
REGIONS: Tuple[str, ...] = tuple(dict.fromkeys(PATCH_REGIONS))

# the three bits every patch-moment carries: 0 anything, 1 an own part, 2 an outside thing
BITS = ("any", "self", "world")
ORDER_F = 7      # per body handle: position change asked (3), turn asked (3), strength (1)

# THE ORDER'S SCALES (the divisors of the order features): the root mean square, per axis, of
# what an EXECUTED order asked of each body handle relative to where it was, so a typical order
# reads about 1. RE-MEASURED 2026-09-23 by lane-b4-contact (review of b4, minor 3: the first
# numbers, 9.3 cm and 21.0 degrees, mixed in the 136 shuffled-order control rows of 1,094, whose
# turns are 26% larger): b4's recorded run m1 (ckpt_step_344400, CPU, 2 bodies in segment 31's own
# world, TRAINING, the run's own orders - planner every 10 moments, babble 0.05 in bouts - 480
# moments), executed rows only (958 body-orders; scratchpad lane_b/b4c/divisors_m1.py ->
# divisors_m1.json): position 9.24 cm over the 14 handles whose position the body executes
# (13,412 handle-orders; median |axis| 5.2 cm, 99th percentile 21.8 cm), turn 20.2 degrees over all
# 25 body handles (23,950; median 10.6, 99th percentile 63.5).
ORDER_POS_CM = 9.24
ORDER_ROT_DEG = 20.2
ORDER_SCALE_MEASUREMENT = {
    "date": "2026-09-23", "source": "lane-b4-contact divisors_m1.py on b4 run m1, executed rows only",
    "pos_cm_rms_per_axis": 9.24, "rot_deg_rms_per_axis": 20.25, "handle_orders_position": 13412,
    "handle_orders_rotation": 23950, "control_rows_left_out": 136,
}
# WHICH ORDER COLUMNS THE HEAD READS (review of b4, minor 3). 11 of the 25 body handles are turned
# and never placed: the action decoder reads a position only for the handles in
# mimo_bridge/action_decoder.POSITION_HANDLES (14 on the trunk_v1 body: both wrists, the ten
# fingers, the head, the lower trunk). The position fields of the other 11 still carried 9.07 cm
# root mean square of asked moves the body never executes (divisors_m1.json) - 33 of the head's 175
# order inputs, noise by construction. The head reads position columns only for position handles:
# 25 x 7 - 11 x 3 = 142 inputs on the H84 body.
ORDER_INPUT_RULE = "position columns only for action_decoder.POSITION_HANDLES"


def order_input_mask(body_handle_names: Sequence[str], position_handles) -> torch.Tensor:
    """[n_body_handles, ORDER_F] bool: the order columns the head reads (ORDER_INPUT_RULE)."""
    pos = set(position_handles)
    mask = torch.ones(len(body_handle_names), ORDER_F, dtype=torch.bool)
    for i, n in enumerate(body_handle_names):
        if n not in pos:
            mask[i, 0:3] = False
    return mask


# THE RUNNING MEAN THAT CENTRES THE LATENT: an exponential average over lived moments with this
# horizon, divided by the weight it has accumulated (corrected for starting at zero). Arithmetic:
# the part of the latent the centre removes is shared by every moment (DESIGN section 10: norm 66
# of 75), so any horizon longer than a few moments finds it; 2,000 moments (100 s) is the horizon
# the wellness normaliser uses for the same reason (it moves at most 1 - exp(-200/2000) = 9.5%
# while one goal ring's 200 moments are lived), and it follows the slow drift of the weights.
CENTRE_HORIZON_STEPS = 2000

# BASE RATES per patch, pooled over bodies: the share of moments on which each bit was on, given
# whether it was on the moment before (so the base rate of STARTING a contact is the rate with the
# bit off before). The same 2,000-moment exponential average, corrected for starting at zero.
BASE_RATE_HORIZON_STEPS = 2000
# A base rate is used only once it rests on this many OBSERVATIONS - one per body per moment, so at
# 8 bodies a rate is used after 13 moments and at 2 bodies after 50 (review of b4, minor 10: the
# name said moments). The standard error of a rate r from n independent observations is
# sqrt(r (1 - r) / n), at most 0.05 on the 0-1 scale at n = 100; the bodies live separate worlds,
# so their observations of one moment are independent of each other.
BASE_RATE_MIN_OBSERVATIONS = 100
BASE_RATE_MIN_MOMENTS = BASE_RATE_MIN_OBSERVATIONS      # the first name; it counted observations
# THE PSEUDO-COUNT (review of b4, minor 9): every rate is (on + half an observation) / (all + one
# observation), half an observation each way - so a patch never seen to start a touch reads a small
# rate, never 0. With a rate of exactly 0 (clipped to 1e-6) one onset there cost the base-rate
# control 13.8 nats of cross-entropy and flattered the head in the head-against-control comparison.
# In the exponential average one observation is one moment's weight, 1 / BASE_RATE_HORIZON_STEPS;
# at a full horizon the smallest rate is 0.5 / 2001 = 0.00025 (8.3 nats for an onset).
BASE_RATE_PSEUDO = 0.5

# THE ONSET LINE, per skin patch and per bit (own part, outside thing). REPLACES the one margin per
# bit of version 1 (0.114 own part, 0.176 outside), which gave 5 false alarms in 100 pooled over the
# patches but 0 in 100 on 29 of the 53 patches and 53 in 100 on the right upper arm (review of b4,
# finding M1, rv_margin.py on m1's moments 250-480): an onset there crossed the line on half the
# moments nothing touched it, so paying it paid touch merely happening. Now an onset side counts as
# FORESEEN when the head's probability that an own part (or an outside thing) touches the patch,
# minus the patch's base rate of starting such a touch, exceeded THAT PATCH'S line: the
# LINE_QUANTILE point of the same difference over that patch's own untouched moments (untouched
# before and after, all bodies pooled). By construction the head crosses its line on 5 in 100 of a
# patch's untouched moments - the design's false-alarm rate on every patch alike; the window reports
# the rate each line actually gave, per bit and per body region (false_alarm_*), the control that
# shows when a line trails the head.
LINE_QUANTILE = 0.95
# The difference lives on -1..+1; LINE_BINS bins of 0.005 (the measured lines of version 1, 0.11 to
# 0.18, sit 20-40 bins above 0). A line is the upper edge of the bin where the running share
# reaches LINE_QUANTILE, so at most 5 in 100 untouched moments lie above it.
LINE_BINS = 400
LINE_LO, LINE_HI = -1.0, 1.0
# THE LINE'S HORIZON, in moments: an exponential average of each patch's per-moment histogram.
# Arithmetic: the head is still learning (on m1 its any-contact AUC rose 0.79, 0.83, 0.86, 0.89 over
# four 100-moment windows), so the line must follow it within a few hundred moments; at 2 bodies a
# 500-moment horizon holds about 1,000 untouched observations of a patch that is untouched most of
# the time (8 bodies: 4,000), where the 95th percentile rests on about 50 observations above it and
# the false-alarm rate it gives has a standard error of sqrt(0.05 x 0.95 / 1000) = 0.7 in 100.
LINE_HORIZON_STEPS = 500
# A line is used only once it rests on this many untouched observations of its patch: the number of
# n observations above a true 95th percentile is binomial, 0.05 n +- sqrt(0.05 x 0.95 n), so at
# n = 400 a line gives 5 +- 1.1 false alarms in 100 (one standard deviation). At 2 bodies that is
# 200 moments of a patch untouched throughout, at 8 bodies 50. Before it a side is counted, not
# scored (as before a base rate exists).
LINE_MIN_OBSERVATIONS = 400

# THE EVIDENCE GATES of the contact_control channel, one per kind of touch (review of b4, finding
# M1: the first build pooled self-touch and outside-touch readings in one gate - 70 of its 157
# readings on m1 came from outside touches - and then paid self-touches 2 each, so foreseeing the
# floor could open the pay for self-touch). A head that knows nothing of the order still marks some
# onsets foreseen (5 in 100 untouched moments cross a line by construction), and paying those would
# pay touch merely happening. So each kind pays only while foreseeing THAT kind beats the
# SHUFFLED-ORDER control (the same state, another body's order through the core and the head).
# ONE READING PER CONTROL MOMENT per kind: the mean, over that moment's scored onsets of the kind
# (all bodies), of (foreseen with the body's own order) - (foreseen with another body's order), -1
# to +1. The bodies of one moment are scored against each other's orders, so their readings are
# not independent; lane-b3-wellness measured that pooling such readings opens a two-standard-error
# gate on noise on 19.8 of 100 planning events against 1.1 for one reading per event with two lines
# (scratchpad lane_b/gate_null.py). Over the last CONTACT_GATE_READINGS readings the gate OPENS when
# the mean stands above CONTACT_GATE_OPEN_NOISE standard errors and, once open, CLOSES when it falls
# under CONTACT_GATE_CLOSE_NOISE (the lead's rule for this gate and controllability's). The null of
# these exact readings is measured by lane-b4-contact (lane-b4-contact.md, "gate null"). The onsets
# are scored and reported whatever the gates say.
CONTACT_GATE_READINGS = 32
CONTACT_GATE_OPEN_NOISE = 3.0
CONTACT_GATE_CLOSE_NOISE = 1.0
GATE_KINDS = ("self", "world")
# A GATE SHUTS WHEN ITS EVIDENCE STOPS ARRIVING (review of b4, minor 1): when its newest reading is
# more than this many moments old, the gate shuts and its readings are dropped, so it must be proved
# again on fresh readings (the self-touch term's SELF_TOUCH_STALE_STEPS rule). A gate also starts
# shut at every load and never inherits saved readings (the entity reader's rule): at one body no
# control runs, and a gate loaded open would pay for ever. Arithmetic for the length: a reading
# licenses paying for the moments the planner chooses among, the goal ring's last 200 moments
# (HABITUATION_STEPS below); a reading older than the whole ring vouches for none of them. At 2
# bodies a reading of either kind arrives on nearly every control moment (m1: 14 to 15 control
# moments per 100 with 6 to 23 scored self-touch onsets and 7 to 20 outside ones), so a normal gap is
# 7 moments; the gaps measured on the lane's own run are in lane-b4-contact.md.
CONTACT_GATE_STALE_STEPS = 200

# WHAT THE GOAL RANKING READS (review of b4, finding M2). The per-moment payment is 0 on about 94 of
# 100 body-moments and up to 2 on the rest, so under the ranking's mean-and-spread scaling one
# foreseen self-touch lifted its moment by 0.5 x (2 - 0.028) / 0.178 = 5.5 of the ranking's units -
# more than integrity, weight 2, moves when its value moves 2.8 of its own spreads: the channel won
# by the shape of its numbers, which DESIGN D8 exists to stop. The value the ranking stores for a
# body's moment is set by CONTACT_VALUE (chosen by measurement, lane-b4-contact.md "M2").
CONTACT_VALUE = "running"
# The running average, per body, of the payments over the last few orders (the review's first fix,
# controllability's form): an exponential average over CONTACT_VALUE_ORDERS orders of the body's
# order hold (5 moments on H84) - 25 moments, 1.25 s - corrected for starting at zero. Arithmetic
# (controllability's CONTROL_EMA_ORDERS): the goal ring holds 40 orders, so an average over 5 gives
# a ring about 8 independent readings.
CONTACT_VALUE_ORDERS = 5

# HABITUATION (the task's own rule, DESIGN D10): a paid onset is multiplied by 1 / (1 + the onsets
# of the same pair in the last HABITUATION_STEPS moments), so a touch-release loop stops paying.
# 200 moments = 10 s = the goal ring's length: a loop the planner could aim at from one ring pays
# less every time it repeats inside it.
HABITUATION_STEPS = 200
# the addendum's section 2: successful prediction and control of SELF-contact is weighted above
# ordinary outside contact; the task fixes the ratio at 2 to 1
SELF_ONSET_PAY = 2.0
WORLD_ONSET_PAY = 1.0
# ONE OUTSIDE TOUCH PER BODY-MOMENT (review of b4, minor 5). A self-touch onset is one pair of the
# body's own parts; an outside onset was counted per skin PATCH, and on m1 a moment with one had 1.41
# patches starting at once, up to 5 - one roll onto the floor paid up to 5 against 2 for a foreseen
# self-touch. The skin report carries one outside bit per patch and cannot tell the floor from a toy,
# so the patches that began touching outside things in one moment are one outside onset for the
# pay: WORLD_ONSET_PAY once, times the habituation factor of the least repeated of its foreseen
# patches. The success counts stay per patch (a patch-level rate of foresight).

# THE LOSS WEIGHT: the head's loss is the mean of its three masked cross-entropies times this. The
# declared weight of each efference head (--efference-weight 0.1, DESIGN D12: the new heads carry
# declared weights; balancing the whole loss is B6's). MEASURED at 0.1 (b4 scratch m1, moments
# 20-480, 2 bodies, TRAINING on ckpt_step_344400): the head is a median 2.2% of the positive loss
# mass (10th-90th percentile 1.3-3.4%), a contribution of 0.033 against the efference body head's
# 0.053 (3.5% of the mass) in the same steps.
CONTACT_HEAD_WEIGHT = 0.1
CONTACT_HEAD_WEIGHT_MEASUREMENT: dict = {
    "date": "2026-09-22", "source": "b4 scratch b4_run.py m1 / b4_analyse.py",
    "share_of_positive_loss_median": 0.022, "share_p10_p90": [0.013, 0.034],
    "efference_body_share_median": 0.035}

# THE HEAD'S SIZES (review of b4, minor 14: the first build's 256 / 64 / 32 carried no measurement):
# the width of its two context layers, of each patch's layer, and of each patch's learned identity.
# CHOSEN BY MEASUREMENT - see lane-b4-contact.md "head sizes" for the fit that set them.
HEAD_HIDDEN = 256
HEAD_PATCH_HIDDEN = 64
HEAD_PATCH_EMBED = 32

# THE SHUFFLED-ORDER CONTROL (another body's order through the core AND the head, no gradient)
# costs one forward of the core, so it runs on every CONTROL_PERIOD-th moment only. The period is
# the smallest of these that shares no factor with the order's hold (--motor-hold, 5 on H84), so
# the control samples every moment of an order's tenure equally often (7 and 5: all 5 phases in
# every 35 moments) - a period that divided the hold would sample one phase only.
CONTROL_PERIODS = (7, 11, 13, 17, 19)


def control_period(motor_hold: int) -> int:
    for p in CONTROL_PERIODS:
        if math.gcd(p, max(1, int(motor_hold))) == 1:
            return p
    return CONTROL_PERIODS[-1] + 4


# DESIGN section 10 rule 2: a unit is live when its input rose above this line on the window's
# samples (GELU of -3 is -0.004; below it a unit returns about nothing and passes no gradient)
LIVE_LINE = -3.0

# AUC resolution: scores are binned on [0, 1]; ties inside one bin count half
AUC_BINS = 1000


# ---------------------------------------------------------------------------------- the head
def rotation_change_deg(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """The turn from q0 to q1 as a rotation vector in degrees (lem/model/efference.py)."""
    from lem.model.efference import rotation_change_deg as _r
    return _r(q0, q1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b of [..., 4] x y z w quaternions."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def shuffled_order(order_cm: torch.Tensor, achieved: torch.Tensor, gaze_idx: int,
                   shift: int = 1) -> torch.Tensor:
    """ANOTHER body's order, size-matched to this body (review of b4, minor 2): the other body's
    order taken as a CHANGE from the other body's own pose, then applied to this body's pose.

    The first control rolled the absolute order across bodies, so its asked changes were measured
    from the wrong pose: turns 26% larger and moves 11% larger than the bodies' own orders
    (rv_orders.py; 25.6 against 20.2 degrees in divisors_m1.json), and a head that learned "bigger
    order, more touch" beat the control without knowing which touch its order causes. Here every
    body handle's asked position change (cm) and asked turn (the world-frame rotation carrying the
    other body's pose to its order, lem/model/efference.rotation_change_deg's convention) is exactly
    the other body's, so order_features of the result equals order_features of the other body's own
    order row for row. Radius, strength and the eyes' row (an eye-in-head command, already relative
    to the body) are the other body's as they stand.

    order_cm: [B, H, 9] execution units; achieved: [B, H, >=7] (positions / 200 cm, x y z w)."""
    order_cm = order_cm.float()
    ach = achieved.float()
    other = order_cm.roll(shift, dims=0)
    ach_o = ach.roll(shift, dims=0)
    out = other.clone()
    out[..., 0:3] = ach[..., 0:3] * 200.0 + (other[..., 0:3] - ach_o[..., 0:3] * 200.0)
    q1 = other[..., 3:7] / other[..., 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q0o = ach_o[..., 3:7] / ach_o[..., 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q0 = ach[..., 3:7] / ach[..., 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    conj = q0o * torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=q0o.dtype, device=q0o.device)
    out[..., 3:7] = _quat_mul(_quat_mul(q1, conj), q0)
    gi = int(gaze_idx)
    out[:, gi] = other[:, gi]
    return out


def order_features(order_cm: torch.Tensor, achieved: torch.Tensor, gaze_idx: int) -> torch.Tensor:
    """The order relative to where the body was, per body handle (the eyes' handle left out).

    order_cm: [B, H, 9] execution units - position cm (world), quaternion x y z w, radius,
    strength. achieved: [B, H, >=7] achieved handle state - position / 200 cm (world), quaternion.
    Returns [B, H - 1, 7]: the position change asked (cm / ORDER_POS_CM), the turn asked (degrees
    as a rotation vector / ORDER_ROT_DEG) and the strength (0-1)."""
    order_cm = order_cm.float()
    ach = achieved.float()
    H = order_cm.shape[1]
    body = [i for i in range(H) if i != int(gaze_idx)]
    dpos = (order_cm[..., 0:3] - ach[..., 0:3] * 200.0) / ORDER_POS_CM
    q1 = order_cm[..., 3:7]
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q0 = ach[..., 3:7]
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    drot = rotation_change_deg(q0, q1) / ORDER_ROT_DEG
    return torch.cat([dpos, drot, order_cm[..., 8:9]], dim=-1)[:, body]


def state_bits(state: torch.Tensor) -> torch.Tensor:
    """[..., P] contact_patch_state ints (0 none, 1 outside, 2 own part, 3 both) -> [..., P, 3]
    booleans for the three BITS (any, self, world)."""
    s = state.long()
    return torch.stack([s > 0, (s & 2) > 0, (s & 1) > 0], dim=-1)


class ContactContextInput(nn.Module):
    """Named observation point for the summed context inputs; unchanged without a repair hook."""

    def forward(self, pre, latent_features, order_features):
        return pre


class ContactHead(nn.Module):
    """Per skin patch: logits for contact, own part given contact, outside thing given contact.

    latent: [B, n_mod, d] the core's last-step tokens for a state whose motor token carries the
    order (the delayed next-frame lesson's latent). order_feat: [B, H - 1, 7] (order_features);
    the head reads the columns `order_mask` keeps ([H - 1, 7] bool, order_input_mask; None = every
    column). prev_state: [B, P, 2] float - each patch's (outside, own part) contact at that
    state."""

    def __init__(self, d_model: int, n_mod: int, n_body_handles: int, n_patches: int = N_PATCHES,
                 hidden: int = HEAD_HIDDEN, patch_hidden: int = HEAD_PATCH_HIDDEN,
                 patch_embed: int = HEAD_PATCH_EMBED, order_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.d_model, self.n_mod, self.n_patches = int(d_model), int(n_mod), int(n_patches)
        # the running mean of the latent over lived moments, already corrected for starting at
        # zero, and the weight it has accumulated (a Python number, so no step reads the device).
        # Not in the state dict: it is a running statistic, saved in runtime_state.pt.
        self.register_buffer("centre_mean", torch.zeros(n_mod, d_model), persistent=False)
        self.centre_weight = 0.0
        mask = (torch.ones(int(n_body_handles), ORDER_F, dtype=torch.bool) if order_mask is None
                else torch.as_tensor(order_mask, dtype=torch.bool))
        if tuple(mask.shape) != (int(n_body_handles), ORDER_F):
            raise ValueError("order_mask must be [%d, %d], got %s" % (n_body_handles, ORDER_F,
                                                                     tuple(mask.shape)))
        # the flattened order columns the head reads (not in the state dict: the receipt stamps the
        # rule, and a head built under another rule has another input width)
        self.register_buffer("order_cols", mask.reshape(-1).nonzero().reshape(-1), persistent=False)
        self.n_order_rows = int(n_body_handles)
        self.lat_norm = nn.LayerNorm(n_mod * d_model)
        self.ctx_lat = nn.Linear(n_mod * d_model, hidden)
        self.ctx_ord = nn.Linear(int(self.order_cols.numel()), hidden)
        self.ctx_pre = ContactContextInput()
        self.ctx2 = nn.Linear(hidden, hidden)
        self.patch_embed = nn.Embedding(n_patches, patch_embed)
        self.p_ctx = nn.Linear(hidden, patch_hidden)
        self.p_emb = nn.Linear(patch_embed, patch_hidden, bias=False)
        self.p_state = nn.Linear(2, patch_hidden, bias=False)
        self.p_out = nn.Linear(patch_hidden, 3)

    # the layers whose live share the window reports, with their unit counts
    LIVE_LAYERS = ("ctx1", "ctx2", "patch")

    @torch.no_grad()
    def update_centre(self, latent: torch.Tensor) -> None:
        """One lived moment of the running mean (all bodies pooled). Call only on real moments.
        The first moment ever sets the mean to itself (the correction for starting at zero).
        An element whose new mean is not finite keeps its old value (review of b4, minor 4: one
        NaN latent used to poison the centre for good - every later loss non-finite, every
        optimizer step skipped, the poisoned centre saved into runtime_state.pt); torch.where, so
        the check reads nothing from the device."""
        a = 1.0 / float(CENTRE_HORIZON_STEPS)
        x = latent.detach().float().mean(dim=0).to(self.centre_mean.dtype)
        w_old = float(self.centre_weight)
        w_new = (1.0 - a) * w_old + a
        new = self.centre_mean * ((1.0 - a) * w_old / w_new) + x * (a / w_new)
        self.centre_mean.copy_(torch.where(torch.isfinite(new), new, self.centre_mean))
        self.centre_weight = w_new

    def order_scale_factors(self, k_pos: float, k_rot: float) -> torch.Tensor:
        """[1, inputs of ctx_ord]: k_pos on the position columns the head reads, k_rot on the turn
        columns, 1 on strength - the factor a change of divisors multiplies each input by."""
        n_body = self.n_order_rows
        k = torch.ones(n_body, ORDER_F)
        k[:, 0:3] = float(k_pos)
        k[:, 3:6] = float(k_rot)
        return k.reshape(-1)[self.order_cols.cpu()].reshape(1, -1)

    def forward(self, latent: torch.Tensor, order_feat: torch.Tensor, prev_state: torch.Tensor,
                return_pre: bool = False):
        B = latent.shape[0]
        x = self.lat_norm((latent.float() - self.centre_mean).reshape(B, -1))
        of = order_feat.reshape(B, -1).float().index_select(1, self.order_cols)
        original_pre = self.ctx_lat(x) + self.ctx_ord(of)
        h1_pre = self.ctx_pre(original_pre, x, of)
        h1 = F.gelu(h1_pre)
        h2_pre = self.ctx2(h1)
        # Keep the existing skip contribution when a silent context unit is revived.
        # Its new path starts through a zeroed ctx2 column, preserving the old output.
        h2 = F.gelu(original_pre) + F.gelu(h2_pre)
        hp_pre = (self.p_ctx(h2).unsqueeze(1) + self.p_emb(self.patch_embed.weight).unsqueeze(0)
                  + self.p_state(prev_state.float()))
        logits = self.p_out(F.gelu(hp_pre))                              # [B, P, 3]
        if return_pre:
            return logits, {"ctx1": h1_pre, "ctx2": h2_pre, "patch": hp_pre}
        return logits

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        """[B, P, 3] logits -> [B, P, 3] probabilities of the three BITS: anything, an own part,
        an outside thing touching the patch (contact x the conditional)."""
        p = torch.sigmoid(logits.float())
        pc = p[..., 0]
        return torch.stack([pc, pc * p[..., 1], pc * p[..., 2]], dim=-1)

    @staticmethod
    def loss(logits: torch.Tensor, target_bits: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Binary cross-entropy, masked. target_bits [B, P, 3] bool; valid [B] bool (a body whose
        report or previous state is missing, or which crossed a reset, is not graded).
          contact            every patch of a valid body
          own | contact      only patches that were touched (the conditional is undefined elsewhere)
          outside | contact  the same
        Returns each term (a mean over its mask; absent when its mask is empty). The two masks'
        counts are taken where the inputs live: the trainer hands the targets and `valid` from the
        CPU, so grading reads nothing back from the card (review of b4, minor 12: the first build
        read both counts from the device)."""
        lg = logits.float()
        yc = target_bits.bool()
        vc = valid.bool().view(-1, 1).expand(-1, yc.shape[1])
        n = int(vc.sum())
        nt = int((vc & yc[..., 0]).sum())
        y = yc.to(lg.device).float()
        vb = vc.to(lg.device).float()
        out = {}
        if n > 0:
            out["contact"] = (F.binary_cross_entropy_with_logits(lg[..., 0], y[..., 0],
                                                                 reduction="none") * vb).sum() / n
            touched = vb * y[..., 0]
            if nt > 0:
                for k, name in ((1, "self"), (2, "world")):
                    out[name] = (F.binary_cross_entropy_with_logits(lg[..., k], y[..., k],
                                                                    reduction="none") * touched).sum() / nt
        return out


# ------------------------------------------------------------------------------ the tracker
def _key_self(a: str, b: str) -> str:
    return "s:" + "|".join(sorted((str(a), str(b))))


def _key_world(patch: int) -> str:
    return "w:" + PATCH_NAMES[int(patch)]


def habituation_factor(habit: Dict[str, List[int]], key: str, step: int) -> float:
    """1 / (1 + onsets of `key` in the HABITUATION_STEPS moments before `step`)."""
    n = sum(1 for s in habit.get(key, ()) if step - HABITUATION_STEPS <= int(s) < step)
    return 1.0 / (1.0 + n)


def record_onset(habit: Dict[str, List[int]], key: str, step: int) -> None:
    lst = habit.setdefault(key, [])
    lst.append(int(step))
    while lst and int(lst[0]) < step - HABITUATION_STEPS:
        lst.pop(0)


def prune_habit(habit: Dict[str, List[int]], step: int) -> None:
    for k in list(habit):
        lst = [int(s) for s in habit[k] if int(s) >= step - HABITUATION_STEPS]
        if lst:
            habit[k] = lst
        else:
            del habit[k]


class _Hist:
    """Positive / negative score histograms for an AUC (probability of a positive scoring above a
    negative; ties inside one bin count half)."""

    def __init__(self):
        self.pos = np.zeros(AUC_BINS, dtype=np.int64)
        self.neg = np.zeros(AUC_BINS, dtype=np.int64)

    def add(self, score: np.ndarray, label: np.ndarray) -> None:
        if score.size == 0:
            return
        b = np.clip((score * AUC_BINS).astype(np.int64), 0, AUC_BINS - 1)
        self.pos += np.bincount(b[label], minlength=AUC_BINS)
        self.neg += np.bincount(b[~label], minlength=AUC_BINS)

    def auc(self) -> Optional[float]:
        P, N = int(self.pos.sum()), int(self.neg.sum())
        if P == 0 or N == 0:
            return None
        below = np.concatenate([[0], np.cumsum(self.neg)[:-1]])
        return float((self.pos * (below + 0.5 * self.neg)).sum() / (P * N))


def _bce(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))


# ONE KIND OF TOUCH'S EVIDENCE GATE, as plain data (a dict of a deque, a bool, an int and a string)
# so every copy of the tracker - the episode's runtime state, the probe harness's exact snapshot
# (mimo_bridge/copied_execution: only listed classes and plain values may sit inside a tracker) -
# carries it without a class of its own. GATE_KINDS; CONTACT_GATE_READINGS says why, and
# CONTACT_GATE_STALE_STEPS when it shuts. A reading is the mean, over one control moment's scored
# onsets of the kind, of (foreseen with the body's own order) - (foreseen with another body's
# order). Never saved: every load starts it shut and empty.
GATE_NOUN = {"self": "self-touches", "world": "outside touches"}


def new_gate(kind: str) -> dict:
    return {"kind": str(kind), "readings": deque(maxlen=CONTACT_GATE_READINGS), "open": False,
            "last_step": None, "stale_note": None}


def gate_stats(g: dict):
    """(n, mean, standard error, readings above 0, readings below 0) of the last
    CONTACT_GATE_READINGS readings; the last four None until there are that many."""
    n = len(g["readings"])
    if n < CONTACT_GATE_READINGS:
        return n, None, None, None, None
    vals = list(g["readings"])
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    return (n, mean, math.sqrt(var / n), sum(1 for v in vals if v > 0),
            sum(1 for v in vals if v < 0))


def gate_add(g: dict, value: float, step: int) -> None:
    """One control moment's reading; the gate opens above CONTACT_GATE_OPEN_NOISE standard errors
    and, once open, closes under CONTACT_GATE_CLOSE_NOISE."""
    g["readings"].append(float(value))
    g["last_step"] = int(step)
    g["stale_note"] = None
    _n, mean, se, _a, _b = gate_stats(g)
    if mean is None:
        g["open"] = False
    elif g["open"]:
        g["open"] = not (mean < CONTACT_GATE_CLOSE_NOISE * se)
    else:
        g["open"] = bool(mean > CONTACT_GATE_OPEN_NOISE * se)


def gate_expire(g: dict, step: int) -> bool:
    """Shut and drop the readings when the newest is older than CONTACT_GATE_STALE_STEPS."""
    if g["last_step"] is None or int(step) - int(g["last_step"]) <= CONTACT_GATE_STALE_STEPS:
        return False
    age = int(step) - int(g["last_step"])
    g["readings"].clear()
    g["open"] = False
    g["last_step"] = None
    g["stale_note"] = ("its newest reading was %d moments old, past the %d allowed, so it shut and "
                       "dropped its readings" % (age, CONTACT_GATE_STALE_STEPS))
    return True


def gate_why(g: dict) -> str:
    noun = GATE_NOUN.get(g["kind"], g["kind"])
    n, mean, se, above, below = gate_stats(g)
    if mean is None:
        txt = ("%s: %d of the %d control moments the gate judges have been measured (a control "
               "moment counts when one of the %s began on it); it is shut until all %d are"
               % (noun, n, CONTACT_GATE_READINGS, noun, CONTACT_GATE_READINGS))
        return txt + ("; " + g["stale_note"] if g["stale_note"] else "")
    if se > 0:
        size = "%.1f times its own measurement noise" % (mean / se)
    else:
        size = ("above zero on every reading alike" if mean > 0 else
                "zero or below on every reading alike")
    return ("%s: on %d of the last %d control moments the bodies' own orders foresaw more of the %s "
            "that began than another body's order did, and on %d fewer (with no link to the order "
            "the two counts are about equal). The average advantage is %s; the gate opens above %g "
            "times and, once open, closes below %g time%s. It is %s."
            % (noun, above, n, noun, below, size, CONTACT_GATE_OPEN_NOISE, CONTACT_GATE_CLOSE_NOISE,
               "" if CONTACT_GATE_CLOSE_NOISE == 1 else "s", "open" if g["open"] else "shut"))


def running_value(state: Optional[list], pay: np.ndarray, valid: np.ndarray, hold: int):
    """What the goal ranking stores for each body this moment (CONTACT_VALUE).

    state: [sum, weight] per-body float64 arrays [B] of an exponential average over
    CONTACT_VALUE_ORDERS orders of `hold` moments (None or another body count: starts empty);
    pay: [B] this moment's payment (0 while the kind's gate is shut); valid [B]: bodies scored this
    moment (the others keep their average and read NaN = the ranking's centre). Returns (new state,
    value [B] float64). Under CONTACT_VALUE "payment" the value is the payment itself."""
    pay = np.asarray(pay, dtype=np.float64).reshape(-1)
    ok = np.asarray(valid, dtype=bool).reshape(-1)
    B = pay.shape[0]
    if CONTACT_VALUE == "payment":
        return state, np.where(ok, pay, np.nan)
    a = 1.0 / float(max(1, int(CONTACT_VALUE_ORDERS) * max(1, int(hold))))
    if (not isinstance(state, (list, tuple)) or len(state) != 2
            or np.asarray(state[0]).shape != (B,)):
        state = [np.zeros(B), np.zeros(B)]
    m = np.asarray(state[0], dtype=np.float64)
    w = np.asarray(state[1], dtype=np.float64)
    m = np.where(ok, (1.0 - a) * m + a * pay, m)
    w = np.where(ok, (1.0 - a) * w + a, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(ok & (w > 0), m / np.maximum(w, 1e-300), np.nan)
    return [m, w], v


class ContactTracker:
    """Base rates, the per-patch onset lines, the onset success signal, the two evidence gates,
    the pay, and the window's report.

    Plain numpy; one call per model step with the head's probabilities already on the CPU. Held
    under a PUBLIC name on the model (LemLite.contact_tracker) so the episode companion does not
    copy it; its base rates and lines ride runtime_state.pt (state_dict), its gates never do."""

    def __init__(self, n_patches: int = N_PATCHES):
        self.P = int(n_patches)
        self.alpha = 1.0 / float(BASE_RATE_HORIZON_STEPS)
        # [bit, previous state off/on, patch]: exponential mean numerator, accumulated weight, count
        self.m = np.zeros((3, 2, self.P), dtype=np.float64)
        self.w = np.zeros((3, 2, self.P), dtype=np.float64)
        self.n = np.zeros((3, 2, self.P), dtype=np.int64)
        # the onset lines: [own part, outside] x patch, an exponential histogram of (probability -
        # base rate of starting) over the patch's untouched moments, its weight and its count
        self.line_alpha = 1.0 / float(LINE_HORIZON_STEPS)
        self.lh = np.zeros((2, self.P, LINE_BINS), dtype=np.float64)
        self.lw = np.zeros((2, self.P), dtype=np.float64)
        self.ln = np.zeros((2, self.P), dtype=np.int64)
        self.gates: Dict[str, dict] = {k: new_gate(k) for k in GATE_KINDS}
        self._region_idx = np.array([REGIONS.index(r) for r in PATCH_REGIONS], dtype=np.int64)
        # per layer of the head, which units rose above LIVE_LINE this window (device tensors,
        # OR-ed each moment and read once, at the report)
        self.live: Dict[str, torch.Tensor] = {}
        self._win_reset()

    def gate(self, kind: Optional[str] = None) -> Tuple[bool, Optional[float], Optional[float], str]:
        """(open, mean reading, the line in force, why) of one kind's gate; kind None = the two
        together (open when either is; the mean and line are None)."""
        if kind is None:
            return (any(g["open"] for g in self.gates.values()), None, None,
                    " ".join(gate_why(self.gates[k]) for k in GATE_KINDS))
        g = self.gates[kind]
        _n, mean, se, _a, _b = gate_stats(g)
        line = None if se is None else (CONTACT_GATE_CLOSE_NOISE if g["open"]
                                        else CONTACT_GATE_OPEN_NOISE) * se
        return bool(g["open"]), mean, line, gate_why(g)

    @torch.no_grad()
    def add_liveness(self, pre: Dict[str, torch.Tensor]) -> None:
        for name, x in pre.items():
            now = (x.detach().float() > LIVE_LINE).reshape(-1, x.shape[-1]).any(dim=0)
            prev = self.live.get(name)
            self.live[name] = now if prev is None or prev.shape != now.shape else (prev | now)

    def liveness_summary(self) -> Optional[dict]:
        """Share of each layer's units whose input rose above LIVE_LINE on this window's samples
        (0 = a dead layer, 1 = every unit took part); reset for the next window."""
        if not self.live:
            return None
        out = {k: round(float(v.float().mean()), 4) for k, v in self.live.items()}
        self.live = {}
        return out

    # ---------------------------------------------------------------- base rates
    def rates(self) -> np.ndarray:
        """[3, 2, P] the base rate of each bit given the previous moment's bit, with half an
        observation each way (BASE_RATE_PSEUDO); NaN until it rests on BASE_RATE_MIN_OBSERVATIONS
        observations."""
        a = self.alpha
        r = (self.m + BASE_RATE_PSEUDO * a) / (self.w + 2.0 * BASE_RATE_PSEUDO * a)
        return np.where(self.n >= BASE_RATE_MIN_OBSERVATIONS, r, np.nan)

    def update_rates(self, prev_bits: np.ndarray, cur_bits: np.ndarray, valid: np.ndarray) -> None:
        """prev_bits, cur_bits [B, P, 3] bool; valid [B] bool. Pooled over valid bodies; each
        (bit, previous state, patch) cell moves once per moment by the mean of its bodies."""
        if not valid.any():
            return
        pv = prev_bits[valid]
        cv = cur_bits[valid].astype(np.float64)
        a = self.alpha
        for k in range(3):
            for s in (0, 1):
                sel = (pv[..., k] == bool(s))                       # [Bv, P]
                cnt = sel.sum(axis=0)
                has = cnt > 0
                if not has.any():
                    continue
                mean = np.where(has, (cv[..., k] * sel).sum(axis=0) / np.maximum(cnt, 1), 0.0)
                self.m[k, s] = np.where(has, (1 - a) * self.m[k, s] + a * mean, self.m[k, s])
                self.w[k, s] = np.where(has, (1 - a) * self.w[k, s] + a, self.w[k, s])
                self.n[k, s] += cnt

    # ---------------------------------------------------------------- the onset lines
    def lines(self) -> np.ndarray:
        """[2, P] each patch's line for the own-part bit (row 0) and the outside bit (row 1): the
        LINE_QUANTILE point of (probability - base rate of starting) over its untouched moments,
        NaN until it rests on LINE_MIN_OBSERVATIONS observations."""
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf = np.cumsum(self.lh, axis=-1) / np.maximum(self.lw, 1e-300)[..., None]
        reach = cdf >= LINE_QUANTILE - 1e-12
        idx = reach.argmax(axis=-1)
        edge = LINE_LO + (idx + 1) * (LINE_HI - LINE_LO) / LINE_BINS
        ok = (self.ln >= LINE_MIN_OBSERVATIONS) & (self.lw > 0) & reach.any(axis=-1)
        return np.where(ok, edge, np.nan)

    def update_lines(self, q: np.ndarray, prev_bits: np.ndarray, cur_bits: np.ndarray,
                     valid: np.ndarray, off_rate: np.ndarray) -> None:
        """Every untouched patch-moment (bit off before and after, a base rate to compare with)
        adds (probability - base rate of starting) to its patch's histogram; each (bit, patch) moves
        once per moment by the distribution of its bodies' values."""
        valid = np.asarray(valid, dtype=bool)
        if not valid.any():
            return
        a = self.line_alpha
        P = self.P
        pidx = np.arange(P)[None, :]
        for j, k in enumerate((1, 2)):
            rate = off_rate[k]
            neg = (~prev_bits[valid][..., k]) & (~cur_bits[valid][..., k]) & np.isfinite(rate)[None, :]
            if not neg.any():
                continue
            x = np.nan_to_num(q[valid][..., k].astype(np.float64) - rate[None, :])
            bins = np.clip(((x - LINE_LO) / (LINE_HI - LINE_LO) * LINE_BINS).astype(np.int64),
                           0, LINE_BINS - 1)
            flat = (np.broadcast_to(pidx, neg.shape)[neg] * LINE_BINS + bins[neg])
            C = np.bincount(flat, minlength=P * LINE_BINS).reshape(P, LINE_BINS).astype(np.float64)
            cnt = C.sum(axis=1)
            has = cnt > 0
            self.lh[j, has] = (1 - a) * self.lh[j, has] + a * (C[has] / cnt[has, None])
            self.lw[j, has] = (1 - a) * self.lw[j, has] + a
            self.ln[j, has] += cnt[has].astype(np.int64)

    # ---------------------------------------------------------------- one moment
    def _win_reset(self):
        self.win = dict(steps=0, body_moments=0, scored=0,
                        onsets_self=0, onsets_self_structural=0, scorable_self=0,
                        both_sides_self=0, predicted_self=0, predicted_both_self=0,
                        onsets_world=0, scorable_world=0, onsets_world_report=0, predicted_world=0,
                        unmapped_sides=0, unscorable_self=0, unlined_self=0, unlined_world=0,
                        neg_self=0, fa_self=0, neg_world=0, fa_world=0,
                        ctl_steps=0, ctl_scorable_self=0, ctl_pred_self=0, ctl_real_pred_self=0,
                        ctl_onsets_world=0, ctl_pred_world=0, ctl_real_pred_world=0,
                        paid=0.0, would_pay=0.0, would_pay_self=0.0, would_pay_world=0.0,
                        paid_moments=0, world_events=0, hab_sum=0.0, hab_n=0,
                        patch_count_mismatch=0, value_sum=0.0, value_n=0,
                        gate_readings={k: 0 for k in GATE_KINDS},
                        gate_open_steps={k: 0 for k in GATE_KINDS},
                        stale_shuts={k: 0 for k in GATE_KINDS})
        self.bce = {k: [0.0, 0.0, 0] for k in BITS}          # head sum, control sum, count
        self.hist_head = {k: _Hist() for k in BITS}
        self.hist_ctl = {k: _Hist() for k in BITS}
        # onsets only: patch-moments whose bit was off the moment before (review of b4, M3)
        self.hist_onset_head = {k: _Hist() for k in GATE_KINDS}
        self.hist_onset_ctl = {k: _Hist() for k in GATE_KINDS}
        # false alarms per body region: [untouched patch-moments, over the line] per kind
        self.fa_region = {k: np.zeros((2, len(REGIONS)), dtype=np.int64) for k in GATE_KINDS}

    def note_value(self, value: np.ndarray) -> None:
        """This moment's contact_control values as the ranking stores them (NaN = unscored)."""
        v = np.asarray(value, dtype=np.float64).reshape(-1)
        v = v[np.isfinite(v)]
        self.win["value_sum"] += float(v.sum())
        self.win["value_n"] += int(v.size)

    def score(self, step: int, q: np.ndarray, prev_bits: np.ndarray, cur_bits: np.ndarray,
              valid: np.ndarray, self_onsets: Sequence[Optional[list]], habit: List[dict],
              world_report: Sequence[Optional[int]] = (), q_control: Optional[np.ndarray] = None,
              pay_on: bool = True) -> Tuple[np.ndarray, dict]:
        """Score this moment and return (pay [B], per-body detail).

        q: [B, P, 3] the head's probabilities of the three bits, formed from the moment BEFORE
        (S_t-1 and the order in force); prev_bits / cur_bits [B, P, 3] bool: the bits at S_t-1 and
        in the burst that produced S_t; valid [B]; self_onsets[b]: the world's self_contact_onsets
        for body b (None = not reported); habit[b]: body b's habituation memory (mutated);
        world_report[b]: the world's contact_onsets_world; q_control: the same probabilities with
        another body's size-matched order through the core and the head (None on steps without
        the control); pay_on: False when nothing reads the payment (the channel is off, or the
        ranking is raw) - the onsets are scored and the would-pay reported all the same.
        Uses the base rates and lines from BEFORE this moment and then updates them."""
        B = q.shape[0]
        R = self.rates()                                          # [3, 2, P]
        off_rate = R[:, 0, :]                                     # base rate of starting, [3, P]
        L = self.lines()                                          # [2, P]
        line = {1: L[0], 2: L[1]}
        pay = np.zeros(B, dtype=np.float64)
        detail = []
        w = self.win
        w["steps"] += 1
        # a gate whose evidence stopped arriving shuts before this moment is judged
        for kind, g in self.gates.items():
            if gate_expire(g, step):
                w["stale_shuts"][kind] += 1
        # the gates as they stood BEFORE this moment: a moment's own readings never open its own pay
        g_open = {k: bool(self.gates[k]["open"]) for k in GATE_KINDS}
        for k in GATE_KINDS:
            w["gate_open_steps"][k] += int(g_open[k])
        readings = {k: [] for k in GATE_KINDS}
        for b in range(B):
            d = {"pay": 0.0, "self_predicted": [], "world_predicted": []}
            detail.append(d)
            if not bool(valid[b]):
                continue
            w["body_moments"] += 1
            pb, cb, qb = prev_bits[b], cur_bits[b], q[b]
            qc = None if q_control is None else q_control[b]
            h = habit[b]
            # ---- the head and the base-rate control on every patch whose control exists
            ctl_p = np.where(pb, R[:, 1, :].T, R[:, 0, :].T)       # [P, 3] rate given previous bit
            ok = np.isfinite(ctl_p)
            for k, name in enumerate(BITS):
                sel = ok[:, k]
                if sel.any():
                    y = cb[sel, k]
                    hq, cq = qb[sel, k], ctl_p[sel, k]
                    bh = self.bce[name]
                    bh[0] += float(_bce(hq, y).sum())
                    bh[1] += float(_bce(cq, y).sum())
                    bh[2] += int(sel.sum())
                    self.hist_head[name].add(hq, y)
                    self.hist_ctl[name].add(cq, y)
            # ---- onsets only (M3): patch-moments whose bit was off the moment before, the head
            # against each patch's base rate of starting that touch on the same patch-moments
            for k, name in ((1, "self"), (2, "world")):
                sel = (~pb[:, k]) & np.isfinite(off_rate[k])
                if sel.any():
                    y = cb[sel, k]
                    self.hist_onset_head[name].add(qb[sel, k], y)
                    self.hist_onset_ctl[name].add(off_rate[k][sel], y)
            w["scored"] += 1
            # ---- false alarms: patch-moments that stayed off, where the patch's line was crossed
            for k, name in ((1, "self"), (2, "world")):
                neg = (~pb[:, k]) & (~cb[:, k]) & np.isfinite(off_rate[k]) & np.isfinite(line[k])
                if neg.any():
                    over = neg & (np.nan_to_num(qb[:, k] - off_rate[k] - line[k], nan=-1.0) > 0)
                    w["neg_" + name] += int(neg.sum())
                    w["fa_" + name] += int(over.sum())
                    fr = self.fa_region[name]
                    np.add.at(fr[0], self._region_idx[neg], 1)
                    np.add.at(fr[1], self._region_idx[over], 1)
            # ---- self onsets (pair level, the world's report), paid per pair
            ons = self_onsets[b] if b < len(self_onsets) else None
            this_keys = []
            self_pay = 0.0
            for o in (ons or ()):
                pair = list(o.get("pair") or ())
                if len(pair) != 2:
                    continue
                if o.get("structural"):
                    w["onsets_self_structural"] += 1
                    this_keys.append(_key_self(*pair))
                    continue
                w["onsets_self"] += 1
                this_keys.append(_key_self(*pair))
                sides, unlined = [], False
                for name in pair:
                    p = PATCH_INDEX.get(str(name))
                    if p is None:
                        if not str(name).endswith("_eye"):
                            w["unmapped_sides"] += 1
                        continue
                    if pb[p, 1] or not np.isfinite(off_rate[1, p]):
                        continue          # already touched by an own part, or no base rate yet
                    if not np.isfinite(line[1][p]):
                        unlined = True    # no line of its own yet: the onset waits for it
                        continue
                    sides.append(p)
                if unlined:
                    w["unlined_self"] += 1
                    continue
                if not sides:
                    w["unscorable_self"] += 1
                    continue
                w["scorable_self"] += 1
                both = len(sides) == 2
                w["both_sides_self"] += int(both)
                hit = all(qb[p, 1] > off_rate[1, p] + line[1][p] for p in sides)
                if hit:
                    w["predicted_self"] += 1
                    w["predicted_both_self"] += int(both)
                    f = habituation_factor(h, _key_self(*pair), step)
                    d["self_predicted"].append([str(pair[0]), str(pair[1]), round(f, 4)])
                    self_pay += SELF_ONSET_PAY * f
                    w["hab_sum"] += f
                    w["hab_n"] += 1
                if qc is not None:
                    hit_c = all(qc[p, 1] > off_rate[1, p] + line[1][p] for p in sides)
                    w["ctl_scorable_self"] += 1
                    w["ctl_real_pred_self"] += int(hit)
                    w["ctl_pred_self"] += int(hit_c)
                    readings["self"].append(float(hit) - float(hit_c))
            # ---- outside onsets (patch level: the outside bit off over the whole previous burst),
            # scored per patch, paid once per body-moment (WORLD_ONSET_PAY)
            onw = np.nonzero((~pb[:, 2]) & cb[:, 2])[0]
            world_f = 0.0
            for p in onw.tolist():
                w["onsets_world"] += 1
                this_keys.append(_key_world(p))
                if not np.isfinite(off_rate[2, p]):
                    continue                      # no base rate yet: counted, not scored
                if not np.isfinite(line[2][p]):
                    w["unlined_world"] += 1
                    continue
                w["scorable_world"] += 1
                hit = bool(qb[p, 2] > off_rate[2, p] + line[2][p])
                if hit:
                    w["predicted_world"] += 1
                    f = habituation_factor(h, _key_world(p), step)
                    d["world_predicted"].append([PATCH_NAMES[p], round(f, 4)])
                    world_f = max(world_f, f)
                if qc is not None:
                    hit_c = bool(qc[p, 2] > off_rate[2, p] + line[2][p])
                    w["ctl_onsets_world"] += 1
                    w["ctl_real_pred_world"] += int(hit)
                    w["ctl_pred_world"] += int(hit_c)
                    readings["world"].append(float(hit) - float(hit_c))
            world_pay = WORLD_ONSET_PAY * world_f
            if world_f > 0.0:
                w["world_events"] += 1
                w["hab_sum"] += world_f
                w["hab_n"] += 1
            if b < len(world_report) and world_report[b] is not None:
                w["onsets_world_report"] += int(world_report[b])
            # every onset of this moment is remembered for habituation, paid or not
            prune_habit(h, step)
            for key in this_keys:
                record_onset(h, key, step)
            # what the foreseen onsets would pay, and what the channel carries: each kind only while
            # its own evidence gate is open, nothing when no one reads the payment
            would = self_pay + world_pay
            d["would_pay_self"] = round(float(self_pay), 4)
            d["would_pay_world"] = round(float(world_pay), 4)
            d["would_pay"] = round(float(would), 4)
            w["would_pay"] += float(would)
            w["would_pay_self"] += float(self_pay)
            w["would_pay_world"] += float(world_pay)
            paid = ((self_pay if g_open["self"] else 0.0)
                    + (world_pay if g_open["world"] else 0.0)) if pay_on else 0.0
            pay[b] = paid
            d["pay"] = round(float(paid), 4)
            w["paid"] += float(paid)
            w["paid_moments"] += int(paid > 0)
        if q_control is not None:
            w["ctl_steps"] += 1
        # ONE reading per kind per control moment: the mean over its scored onsets of that kind
        for kind in GATE_KINDS:
            if readings[kind]:
                gate_add(self.gates[kind], sum(readings[kind]) / len(readings[kind]), step)
                w["gate_readings"][kind] += 1
        # the lines and the base rates learn from this moment, after it was judged
        self.update_lines(q, prev_bits, cur_bits, valid, off_rate)
        self.update_rates(prev_bits, cur_bits, valid)
        return pay, {"bodies": detail, "gate_open": bool(g_open["self"] or g_open["world"]),
                     "gate_self_open": g_open["self"], "gate_world_open": g_open["world"]}

    # ---------------------------------------------------------------- the report
    def summary(self, liveness: Optional[dict] = None) -> dict:
        w = self.win

        def ratio(a, b, nd=4):
            return None if not b else round(a / b, nd)

        def auc(hist):
            v = hist.auc()
            return None if v is None else round(v, 4)
        bce_h = {k: ratio(v[0], v[2], 5) for k, v in self.bce.items()}
        bce_c = {k: ratio(v[1], v[2], 5) for k, v in self.bce.items()}
        auc_h = {k: auc(self.hist_head[k]) for k in BITS}
        auc_c = {k: auc(self.hist_ctl[k]) for k in BITS}
        hb = [v for v in bce_h.values() if v is not None]
        cb = [v for v in bce_c.values() if v is not None]
        L = self.lines()
        lines = {}
        for j, kind in enumerate(GATE_KINDS):
            ok = np.isfinite(L[j])
            lines[kind] = {"patches_ready": int(ok.sum()), "patches": int(self.P),
                           "median": None if not ok.any() else round(float(np.median(L[j][ok])), 4),
                           "largest": None if not ok.any() else round(float(L[j][ok].max()), 4)}
        fa_region = {}
        for ri, region in enumerate(REGIONS):
            fa_region[region] = {k: ratio(int(self.fa_region[k][1, ri]), int(self.fa_region[k][0, ri]))
                                 for k in GATE_KINDS}
        out = {
            "onsets_self": int(w["onsets_self"]),
            "onsets_self_structural": int(w["onsets_self_structural"]),
            "scorable_self": int(w["scorable_self"]),
            "both_sides_self": int(w["both_sides_self"]),
            "predicted_self": int(w["predicted_self"]),
            "predicted_both_self": int(w["predicted_both_self"]),
            "onsets_world": int(w["onsets_world"]),
            "scorable_world": int(w["scorable_world"]),
            "onsets_world_report": int(w["onsets_world_report"]),
            "predicted_world": int(w["predicted_world"]),
            "success_self": ratio(w["predicted_self"], w["scorable_self"]),
            "success_self_both": ratio(w["predicted_both_self"], w["both_sides_self"]),
            "success_world": ratio(w["predicted_world"], w["scorable_world"]),
            "false_alarm_self": ratio(w["fa_self"], w["neg_self"], 5),
            "false_alarm_world": ratio(w["fa_world"], w["neg_world"], 5),
            "false_alarm_by_region": fa_region,
            "success_control": ratio(w["ctl_pred_self"], w["ctl_scorable_self"]),
            "success_control_world": ratio(w["ctl_pred_world"], w["ctl_onsets_world"]),
            "success_paired": ratio(w["ctl_real_pred_self"], w["ctl_scorable_self"]),
            "success_paired_world": ratio(w["ctl_real_pred_world"], w["ctl_onsets_world"]),
            "control_steps": int(w["ctl_steps"]),
            "control_onsets_self": int(w["ctl_scorable_self"]),
            "control_onsets_world": int(w["ctl_onsets_world"]),
            "paid_mean": ratio(w["paid"], w["body_moments"], 5),
            "would_pay_mean": ratio(w["would_pay"], w["body_moments"], 5),
            "would_pay_self_mean": ratio(w["would_pay_self"], w["body_moments"], 5),
            "would_pay_world_mean": ratio(w["would_pay_world"], w["body_moments"], 5),
            "paid_moments_share": ratio(w["paid_moments"], w["body_moments"]),
            "world_pay_events": int(w["world_events"]),
            "value_mean": ratio(w["value_sum"], w["value_n"], 5),
            "habituation_mean": ratio(w["hab_sum"], w["hab_n"]),
            "head_bce": None if not hb else round(sum(hb) / len(hb), 5),
            "control_bce": None if not cb else round(sum(cb) / len(cb), 5),
            "head_auc": auc_h["any"],
            "head_bce_bits": bce_h, "control_bce_bits": bce_c,
            "head_auc_bits": auc_h, "control_auc_bits": auc_c,
            "onset_auc_self": auc(self.hist_onset_head["self"]),
            "onset_auc_world": auc(self.hist_onset_head["world"]),
            "control_onset_auc_self": auc(self.hist_onset_ctl["self"]),
            "control_onset_auc_world": auc(self.hist_onset_ctl["world"]),
            "unscorable_self": int(w["unscorable_self"]),
            "unlined_self": int(w["unlined_self"]),
            "unlined_world": int(w["unlined_world"]),
            "unmapped_sides": int(w["unmapped_sides"]),
            "patch_count_mismatch": int(w["patch_count_mismatch"]),
            "body_moments": int(w["body_moments"]), "steps": int(w["steps"]),
            "line_quantile": LINE_QUANTILE, "lines": lines,
            "base_rate_min_observations": BASE_RATE_MIN_OBSERVATIONS,
            "habituation_steps": HABITUATION_STEPS,
        }
        for kind in GATE_KINDS:
            g_open, g_mean, g_line, g_why = self.gate(kind)
            out.update({"gate_%s_open" % kind: bool(g_open),
                        "gate_%s_mean" % kind: None if g_mean is None else round(g_mean, 5),
                        "gate_%s_line" % kind: None if g_line is None else round(g_line, 5),
                        "gate_%s_why" % kind: g_why,
                        "gate_%s_readings" % kind: int(w["gate_readings"][kind]),
                        "gate_%s_open_share" % kind: ratio(w["gate_open_steps"][kind], w["steps"]),
                        "gate_%s_stale_shuts" % kind: int(w["stale_shuts"][kind])})
        out["gate_open"] = bool(out["gate_self_open"] or out["gate_world_open"])
        out["gate_why"] = self.gate()[3]
        if liveness is not None:
            out["liveness"] = liveness
        # the head pools nothing: every patch has its own path to the output (DESIGN section 10)
        out["pool_spread"] = {}
        self._win_reset()
        return out

    # ---------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        """The base rates and the onset lines. The gates are never saved: a load starts them shut
        and empty (review of b4, minor 1)."""
        return {"m": self.m.copy(), "w": self.w.copy(), "n": self.n.copy(),
                "lh": self.lh.astype(np.float32), "lw": self.lw.copy(), "ln": self.ln.copy(),
                "patch_names": list(PATCH_NAMES), "version": CONTACT_HEAD_VERSION}

    def load_state_dict(self, sd: dict) -> bool:
        """Restore the base rates (and the lines when saved in this layout); the gates stay shut and
        empty whatever the saved state held. False when nothing could be restored."""
        if not sd or list(sd.get("patch_names") or []) != list(PATCH_NAMES):
            return False
        m, w_, n = (np.asarray(sd.get(k)) for k in ("m", "w", "n"))
        if m.shape != self.m.shape or w_.shape != self.w.shape or n.shape != self.n.shape:
            return False
        self.m, self.w, self.n = m.astype(np.float64), w_.astype(np.float64), n.astype(np.int64)
        lh, lw, ln = (sd.get(k) for k in ("lh", "lw", "ln"))
        if (lh is not None and lw is not None and ln is not None
                and np.asarray(lh).shape == self.lh.shape and np.asarray(lw).shape == self.lw.shape
                and np.asarray(ln).shape == self.ln.shape):
            self.lh = np.asarray(lh, dtype=np.float64)
            self.lw = np.asarray(lw, dtype=np.float64)
            self.ln = np.asarray(ln, dtype=np.int64)
        self.gates = {k: new_gate(k) for k in GATE_KINDS}
        return True
