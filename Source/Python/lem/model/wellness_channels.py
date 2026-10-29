# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Wellness channels: controllability, learning progress, and the normalised, prioritised ranking.

docs/gain_of_function_v3/DESIGN.md decisions D6 (controllability), D7 (learning progress), D8 and
section 4 (the ranking), D9 (consequence counts world contact only - lem_model.consequence_kinds).

WHAT THIS CHANGES AND WHAT IT DOES NOT. The planner chooses a GOAL: the remembered moment (one of
the last 200 in the goal ring, 10 s at 50 ms a moment) with the best value, per body. Until now
that value was a plain sum - five grounded terms, the movement drive, the agency payment, 0.5 per
kind of consequence, the reader - multiplied by (1 + curiosity drive). Measured on H84 segment 31
(understand-wellness-planner.md section 3): the consequence term carried 0.962 of the movement of
that sum, and the curiosity multiplier decided the winning goal in 5 of 5 snapshots. One channel
won because its numbers were larger.

Under --wellness-rank normalized each channel of a remembered moment is centred and scaled by its
own running statistics over per-moment values, weighted by the addendum's priority order
(integrity > controllability > learning progress > the rest > efficiency), and summed:

    value = sum_c w_c * (s_c - centre_c) / max(scale_c, floor_c)      if the moment's harm is
                                                                        at or under the gate line
    value = -inf                                                     otherwise, unless every
                                                                        remembered moment is over
                                                                        it: then the least harmful

s_c is what the ring STORES for channel c: the moment's value minus that body's running average of
the channel at the moment it was stored (RING_DETREND_STEPS). A slow drift - skin load building up
over a minute, fatigue, a learning curve - moves the value and its average together, so it gives
every moment of a ring nearly the same stored value and cannot make the oldest or the newest moment
the goal (review of b3, 2026-09-22: 42 of 44 goals were the oldest tenth of the ring before this).
Harm = max(skin pain, the limit part 1 - motor_state), the worse of the two harms integrity reads;
the gate reads it raw, never detrended.

The raw grounded sum is NOT touched: it still feeds wellness_grad (an input of the motor
controller) and the named-entity teacher, and the goal ring still stores the raw value beside the
channel vector. Only the ranking that picks a goal changes.

CONTROLLABILITY (--controllability on): for every latched order that ran its full tenure, the
effect the model EXPECTED from it (the efference body head's tenure prediction, lem/model/
efference.py; the commanded change itself when the head is off) against the effect the body
ACHIEVED over the tenure. Agreement = direction cosine x (smaller magnitude / larger magnitude),
-1 (the opposite effect) to +1 (exactly the expected effect); 0 = unrelated, or nothing where
something was expected. Counted only when both the expected and the commanded effect are larger
than the body's own drift while holding still (CONTROL_FLOOR), so stillness is never counted as
control. A SHUFFLED-ORDER control beside it (DESIGN D6): what the model would have expected from
THIS body's state had it sent ANOTHER body's order (the efference head on this body's latent with
the other order; for the command source, the other body's target orientation taken from this
body's own starting pose), against this body's result. An expectation that ignores the order is
the same with either order, so it scores exactly 0 above its control. The channel enters the goal
ranking only while agreement - control stands above 3 times its own measurement noise, and leaves
it when it falls under 1 time (CONTROL_GATE_OPEN_NOISE / CONTROL_GATE_CLOSE_NOISE); until then it
is measured and reported only. Beside that gate a SHADOW GATE (2026-10-03, recorded only) applies
the same rule to the size-matched control - another body's asked moves and turns, measured from its
own pose, applied to this body's pose - on the same graded commands; the weight, the ranking and
every loss read the first gate only (SHADOW_CONTROL_KIND says why both are recorded).

LEARNING PROGRESS (--learning-progress on): per body, per sense, the ratio r = the model's
next-moment error / the error of predicting that nothing changes (persistence); for the efference
heads, head error / the do-nothing error. Fast and slow running averages of the model error and of
the persistence error, each corrected for starting at zero; r_fast and r_slow are the ratios of
those averages; progress = (r_slow - r_fast) / r_slow, clipped to [-1, 1]: +1 = the error relative
to doing nothing has fallen to zero lately, 0 = no change, -1 = it has doubled or worse. The ratio
cancels the world becoming uniformly easier or harder (the caregiver leaving makes both errors
fall together). It does NOT cancel a change in what kind of change the world makes: MEASURED on
the frozen weights of ckpt_step_344400 (b3 measurement 2, 2 bodies, a fresh world settling for 16
s, where no learning can happen) the channel read -0.016, -0.055 and -0.071 over three 100-moment
windows, proprioception alone -0.20; a value inside that band is not evidence of learning. No
same-moment control exists for it yet (docs/gain_of_function_v3/agents/b3-wellness.md, open
issues). A sense or head whose network was freshly started at the load (the vestibular encoder,
the efference heads) is left out for LP_HOLDOUT_STEPS model steps (50), so its first moments from
a random start are not counted as progress (review of b2, finding 5; measured without it, a fresh
key had the largest progress of all keys on 1.2 of 100 steps, all of them in moments 38-73).

Every constant below carries its measurement; the per-constant notes say where it was measured.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# v2 (lane-b3-wellness, 2026-09-23): the ring stores each channel as its distance from the body's
# running average; the gate reads harm (pain and the limit part); controllability's control is the
# shuffled ORDER with a two-line gate; a freshly started sense sits out of learning progress
WELLNESS_CHANNELS_VERSION = "wellness_channels_v2"

# THE CHANNELS, in report order. Each exists in a run only when its source does (see
# LemLite._rank_channels_active): integrity, coherence, prediction_error, tactile_comfort,
# capability and efficiency always; controllability and learning_progress with their flags;
# contact_control when B4's channel is on; consequence with --consequence-w > 0; self_value when
# the entity self's reader exists; agency with --agency-drive; movement with --movement-drive;
# curiosity with --curiosity-goal on.
CHANNEL_ORDER = ("integrity", "controllability", "learning_progress", "coherence",
                 "contact_control", "consequence", "self_value", "agency",
                 "prediction_error", "tactile_comfort", "capability", "movement",
                 "curiosity", "efficiency")

# the priority order of the addendum (section 3.4): integrity > controllability > learning
# progress > efficiency, the rest in between (DESIGN.md section 4). Efficiency is logged at 0:
# energy is item 11 of the addendum, after controllability works.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "integrity": 2.0, "controllability": 1.0, "learning_progress": 0.75, "coherence": 0.5,
    "contact_control": 0.5, "consequence": 0.5, "self_value": 0.5, "agency": 0.25,
    "prediction_error": 0.25, "tactile_comfort": 0.25, "capability": 0.25, "movement": 0.25,
    "curiosity": 0.25, "efficiency": 0.0}

# The rows stored beside the channels in each remembered moment. Neither is weighted, normalised or
# detrended. PAIN_ROW is the skin pain alone (reported); HARM_ROW = max(pain, 1 - motor_state), the
# worse of the two harms the integrity channel reads (DESIGN section 4), is what the gate reads.
PAIN_ROW = "pain"
HARM_ROW = "harm"
UNRANKED_ROWS = (PAIN_ROW, HARM_ROW)

# WHICH CHANNELS THE PLANNER CAN SEE IN AN IMAGINED MOMENT. Remembered moments carry every
# channel (they happened, so each was measured). An IMAGINED moment - the planner's rollout, the
# causal chain's alternatives, prospective wellness - is scored by the approximate evaluator
# (Judge B), which predicts only lem.model.wellness.APPROXIMATE_TERMS. Everything else is blind.
JUDGE_B_VIEW: Dict[str, str] = {
    "integrity": "partly: its pain part only through tactile_comfort = (0.5 + 0.35 pleasant)(1 - "
                 "pain), its limit part through motor_state",
    "coherence": "yes", "prediction_error": "yes", "tactile_comfort": "yes", "capability": "yes",
    "controllability": "no", "learning_progress": "no", "contact_control": "no",
    "consequence": "no", "self_value": "no", "agency": "no", "movement": "no",
    "curiosity": "no", "efficiency": "no"}

# NORMALISER HORIZON, in model steps. Arithmetic, not a fit: a goal is chosen among the last 200
# moments (the ring), and every candidate is judged on the SAME centre and scale at that instant.
# With a 2,000-step horizon the scale can move at most 1 - exp(-200/2000) = 9.5% while a moment
# sits in the ring, so a candidate's standing does not depend on when it happened to be stored.
NORMALISER_HORIZON_STEPS = 2000

# FLOORS: the smallest scale a channel is divided by, in the channel's own units: its
# MOMENT-TO-MOMENT spread, the standard deviation of the change between successive moments over
# the square root of 2 (for independent moments this equals the ordinary spread; a slow drift,
# such as skin exposure building up over a minute, does not inflate it). A channel whose running
# spread falls below the jitter it shows from one moment to the next is never divided by less,
# so a stuck value or a shut gate cannot turn its last digits into the deciding vote.
# MEASURED 2026-09-22 by b3-wellness on ckpt_step_344400, CPU, the segment 31 world, the run's own
# orders (planner every 10 moments, babble 0.05 in bouts), TRAINING as the live run does, every b3
# flag on with the efference heads and the causal chain (b3 measurement 3, scratchpad
# b3/b3_run.py m3 -> run_m3.pt, b3_analyse_m2.py m3): 2 bodies, moments 20-240, the mean of the two
# bodies' values. The same quantities on FROZEN weights (measurements 1 and 2) read 2.4 times
# larger for integrity (0.0477), 3.6 times for prediction_error (0.0532) and 5.1 times for
# curiosity (0.187): a model that does not learn online predicts worse and more erratically, so
# frozen-weight jitter is not the live condition. Numbers in docs/gain_of_function_v3/agents/b3-wellness.md, section "Floors".
# None = unmeasured, and a normalised ranking refuses to start with an unmeasured floor on a
# weighted channel.
FLOORS: Dict[str, Optional[float]] = {c: None for c in CHANNEL_ORDER}
FLOORS.update({
    "integrity": 0.0196,          # 0-1; pain dominates it (1 - pain almost everywhere)
    "learning_progress": 0.00035, # -1 to +1 (its spread over the run 0.0045)
    "coherence": 0.0054,          # 0-1
    "consequence": 0.512,         # kinds 0-3 under the world-only rule
    "agency": 0.0022,             # the agency payment, 0 to about 0.15
    "prediction_error": 0.0148,   # 0-1
    "tactile_comfort": 0.0222,    # 0-0.85
    "capability": 0.00052,        # 0-1; a slow drift (its spread over the run 0.062)
    "movement": 0.0073,           # 0-1
    "curiosity": 0.0366,          # the drive, 0-1; see the note below
    "efficiency": 0.272,          # minus (power / 66 W + holding / 63 N m)
    # CONTROLLABILITY had no measured order in measurement 3 (the fresh efference head expected
    # less than the holding drift on every order), so its floor comes from measurement 2 (frozen
    # weights, 2 bodies, the commanded-rotation source): the running agreement over 5 orders,
    # -1 to +1, per body 0.0037 and 0.0039 (its spread over that run 0.017)
    "controllability": 0.0038,
    # the reader's value is the proven link of each counted thing present, -1 to +1 each; it read
    # 0 on every moment measured (its gate shut on all 782 windows of segments 30-31), so its
    # floor is the one counted link segment 31 ended with: the caregiver's, 0.118 (the last mimo
    # record of H84_fresh_drivers_20260918_seg31_toy_and_push, entity_self.reader.link)
    "self_value": 0.118,
    # CONTACT_CONTROL (b4-self-contact, lem/model/contact_head.py): 2 per self-touch and 1 per
    # outside touch the contact head foresaw, times the habituation factor, 0 while its evidence gate
    # is shut. MEASURED 2026-09-22 (b4 scratch b4_run.py m1 -> b4_analyse.py m1 250 measured):
    # ckpt_step_344400, CPU, 2 bodies, TRAINING, moments 100-480, the value the channel would carry
    # with its gate open (the gate stayed shut in that run, so the carried value was 0 throughout):
    # moment-to-moment spread 0.184 and 0.168 per body, mean 0.176; the value averaged 0.028 and was
    # above 0 on 6 of 100 body-moments
    "contact_control": 0.176,
})
# NOTE ON CURIOSITY: the drive is computed from the batch's MEAN next-moment error, so eight live
# bodies average eight errors and jitter less than the two bodies measured here. Where the live
# running spread sits under the floor the floor governs and the channel counts for less than its
# weight; the wellness_rank block prints scale beside floor per channel, which shows it.

# THE INTEGRITY GATE LINE, on the harm scale (HARM_ROW = max(skin pain, 1 - motor_state); pain as
# lem/model/contact_wellness.contact_response scores it: 0 = no modelled aversion anywhere on the
# skin, approaching 1 = an extreme load; mechanical_v4: load / (1 + load), load 1 = one
# first-discomfort reference of 219.4 kPa). A moment over it is never a goal while any remembered
# moment is under it: an impact, a fall, a limb crushed - not ordinary life.
# RE-MEASURED 2026-09-23 by lane-b3-wellness on the population the deploy resumes (the review of
# b3, minor 2: the first line, 0.87, was the largest pain of two CPU runs within 50 s of a reset,
# before the skin's 60 s load had built up): ckpt_step_344400's own 8-body episode, restored from
# its companion (4.8 hours of body time in), frozen weights, the run's own orders, moments 20-600,
# 4,640 body-moments (scratchpad lane_b/ep_run.py asbuilt8 -> ep_pain.py): pain median 0.725,
# 95th percentile 0.866, 99th 0.901. The 0.87 line gated 4.8 of every 100 of those ordinary
# moments (one body lies at a median 0.85 for minutes) and up to 12 of every 100 candidates in a
# 100-moment window. Ordinary pain runs continuously up to 0.912 (37 moments between 0.90 and
# 0.92); above it the next moment is 0.932, then 16 moments up to 0.999 in two short events (one
# body for 15 moments, one for 2) - an empty stretch from 0.912 to 0.932. The line sits in that
# gap: it gates the 16 event moments (0.34 in 100) and no moment of the continuous ordinary part.
# The limit part never approaches it: 1 - motor_state is a mean over 69 coordinates, median
# 0.026 and largest 0.078 on the same moments, so a single limb at the end of its reach cannot
# reach any line on this scale (see lane-b3-wellness.md; the gate reads it as DESIGN section 4
# says, and it is harmless there).
INTEGRITY_GATE_LINE: Optional[float] = 0.92

# CONTROLLABILITY FLOOR: how far the body moves over one order's 250 ms while it is TOLD TO HOLD
# STILL (target = where each handle is when the order begins, the model's own strengths), as the
# length of the effect vector in units of the measured per-axis tenure spreads below. An order
# expected, or commanded, to do less than that cannot be told apart from holding still, so it is
# never counted as control. MEASURED 2026-09-22 by b3-wellness on ckpt_step_344400 (CPU, frozen
# weights, the segment 31 world): 59 hold orders interleaved with 60 of the model's own orders
# (scratchpad b3/b3_measure_world.py drift part, b3_analyse_world.py / b3_analyse_rot.py); the
# line is the 95th percentile of the hold drift (5 in 100 hold orders drifted further).
#   efference: every body handle's position and rotation change (the body head's output) -
#              hold drift median 3.50, 95th percentile 6.67, largest 10.97; the model's own orders
#              achieved median 13.0
#   command:   every body handle's rotation change only (the order's commanded rotation, DESIGN
#              D6) - hold drift median 2.65, 95th percentile 4.91, largest 6.66; the model's own
#              orders COMMANDED 41-53 (median 45) and achieved median 9.2
CONTROL_FLOOR: Dict[str, Optional[float]] = {"efference": 6.67, "command": 4.91}

# The per-axis spread of the change over one order's tenure (5 moments, 250 ms), measured by
# b1-causal-efference on ckpt_step_344400 (lem/model/efference.py EFFERENCE_SCALES, 2026-09-22):
# one unit of each is one measured spread, so centimetres and degrees count alike in one effect
# vector.
TENURE_CM = 1.80
TENURE_DEG = 11.0

# CONTROLLABILITY's running average per body, over measured orders. Arithmetic: the goal ring
# holds 40 orders (200 moments / 5); an average over the last 5 orders (1.25 s) gives a ring about
# 8 independent readings, so the channel can tell moments of one ring apart while each reading
# averages 5 orders.
CONTROL_EMA_ORDERS = 5

# CONTROLLABILITY'S EVIDENCE GATE. The channel enters the goal ranking only while the real
# agreement beats its SHUFFLED-ORDER control (the same body's state and result, another body's
# order). One reading per graded ORDER: the mean of (agreement - control) over the bodies measured
# on it. Over the last CONTROL_GATE_READINGS orders the mean reading is compared with its own
# standard error (the readings' own measurement noise). Without the gate the channel adds noise at
# weight 1.0: MEASURED on ckpt_step_344400 (b3 measurement 2, 2 bodies, 126 body-orders) the
# commanded-rotation agreement was 0.011 against a control of 0.005-0.017, both inside their
# spread (agreement sd 0.053 per body-order). 32 orders = 160 moments = 8 s of a run.
# ONE READING PER ORDER, NOT PER BODY-ORDER, AND TWO LINES, NOT ONE (review of b3, minor 1; the
# control that catches the lines lying, lane-b3-wellness.md "gate with two lines", scratchpad
# lane_b/gate_null.py, 100,000 planning events, 8 bodies, readings with no control at all):
#   the first build (32 body-order readings, one line at twice the standard error) stood open on
#   2.7 of 100 planning events with independent readings and on 19.8 of 100 when half of each
#   reading's spread was shared inside its order - the bodies of one order are scored against
#   each other's orders, so their readings are not independent and a standard error over them
#   reads too small; two lines on the same pooled readings still stood open on 17.3 of 100.
#   AS BUILT (one reading per order, two lines, this class's own _update_gate): open on 1.1 of 100
#   planning events in both cases (1.08 independent, 1.10 shared), in about one opening per 1,000
#   planning events that lasts about 11 of them (the closing line holds it until the noise falls
#   under 1 standard error): about 5 planning events an hour at H84's 450 an hour, against 12
#   (independent) or 89 (shared) for the first build.
# The gate OPENS only above CONTROL_GATE_OPEN_NOISE standard errors and, once open, CLOSES only
# below CONTROL_GATE_CLOSE_NOISE (the lead's rule for this gate and b4's contact gate). A resume
# re-proves the gate: saved readings are kept, the gate starts shut and opens only when they clear
# the opening line.
# Measured and reported whatever the gate says; a single body has no control and never opens it.
CONTROL_GATE_READINGS = 32
CONTROL_GATE_OPEN_NOISE = 3.0
CONTROL_GATE_CLOSE_NOISE = 1.0

# THE SHADOW GATE (2026-10-03; the operator's decision of Oct 3: the restart segment reads both gates on the same
# commands for one segment, and only then is the gate switched). The gate above reads the ABSOLUTE control: the
# body head's expectation from another body's command as it stands (last_efference_tenure expected_shuffled_cm /
# _deg), targets chosen for the other body's pose. A body head whose output grows with the asked change forms a
# larger, worse expectation from such a command, the control's agreement falls, and the gate's difference grows
# without command use (docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md, section 8, finding 1).
# The shadow gate is the same gate fed by the SIZE-MATCHED control (lem/model/efference.size_matched_other_order:
# the other body's asked moves and turns, measured from its own pose, applied to this body's pose;
# last_efference_tenure expected_size_matched_cm / _deg). Per graded command (the bodies' commands latched on one
# model step, graded together): one reading = the mean over the measured bodies of (agreement with this body's
# expectation - agreement with the size-matched expectation), on the same measured bodies as the gate above; its
# own last CONTROL_GATE_READINGS readings; the same two lines (ControllabilityTracker._two_lines, one function for
# both gates).
# RECORDED ONLY. The stats block is its one reader (summary()): the channel's weight, the goal ranking and every
# loss read the gate above (LemLite._rank_weights_now reads gate()), so training is unchanged by it
# (docs/gain_of_function_v4/swap_test_phases/lockstep_vs_production.py). Its readings ride the tracker's saved
# state under their own control kind and follow the gate's own load rule: the readings come back, the shadow gate
# starts shut and reopens only when they clear the opening line; a state saved before it existed loads with the
# shadow gate empty and shut.
SHADOW_CONTROL_KIND = "size-matched command"

# What the two controls, the two gates and their counts are: the text the stats block carries beside the numbers
# (summary()'s gate_definitions), so a reader of a stats record needs no other file to say what each number is
CONTROL_GATE_DEFINITIONS: Dict[str, str] = {
    "reading": "one per graded command (the bodies' commands latched on one model step are graded together when "
               "the next commands appear): the mean, over the bodies measured on it, of the agreement with this "
               "body's own expectation minus the agreement with the control's expectation; -2 to +2, 0 = the body "
               "head's expectation matched what the body did no better with this body's command than with the "
               "control command",
    "gate": "the evidence gate in force. Its control is the ABSOLUTE one: the body head's expectation from another "
            "body's command as it stands, targets chosen for that body's pose (control_mean; difference_mean = "
            "agreement minus it). Over its last %d readings it opens when their mean is above %g of their standard "
            "errors and, once open, shuts when the mean is under %g; while it is shut the controllability weight "
            "in the goal ranking is 0. gate_mean and gate_standard_error are that mean and standard error, "
            "gate_readings_above_zero the readings above 0 (half of the readings held = no advantage), "
            "gate_readings_held how many of the %d readings exist (the gate stays shut until all do)"
            % (CONTROL_GATE_READINGS, CONTROL_GATE_OPEN_NOISE, CONTROL_GATE_CLOSE_NOISE, CONTROL_GATE_READINGS),
    "shadow_gate": "recorded only: the same rule on the same graded commands and the same measured bodies, with "
                   "the SIZE-MATCHED control in place of the absolute one: the body head's expectation from another "
                   "body's asked moves and turns, measured from that body's own pose and applied to this body's "
                   "pose (size_matched_control_mean; size_matched_difference_mean = agreement minus it). "
                   "shadow_gate_mean, shadow_gate_standard_error, shadow_gate_readings_above_zero, "
                   "shadow_gate_readings_held and shadow_gate_open read as the gate's. This block is its one "
                   "reader: the controllability weight, the goal ranking and every loss read the gate in force",
    "counts": "per report window, in graded commands. gate_commands / shadow_gate_commands: the commands at which "
              "that gate took a reading; gate_open_commands / shadow_gate_open_commands: those after whose reading "
              "it stood open. gates_same_commands: the commands at which both gates took a reading, of which "
              "gates_both_open_commands had both open, gate_only_open_commands the gate in force open and the "
              "shadow gate shut, shadow_gate_only_open_commands the reverse; on the rest both were shut",
}

# LEARNING PROGRESS horizons, in moments. Arithmetic: the fast average spans one goal ring (200
# moments, 10 s); the slow one spans 2,000 moments (100 s), longer than one full caregiver cycle
# (60 s present + 15 s absent = 1,500 moments on H84), so the slow average is never one phase of
# the world schedule.
LP_FAST_STEPS = 200
LP_SLOW_STEPS = 2000

# LEARNING-PROGRESS HOLD-OUT after a network is freshly started at a load, in model steps (review of
# b2, finding 5: "the new sense's own learning curve will dominate the learning-progress channel at
# the deploy"). MEASURED 2026-09-23 by lane-b3-wellness (scratchpad lane_b/ep_run.py lpfresh ->
# run_lpfresh.pt, lp_holdout.py -> run_lpfresh_holdout.json): ckpt_step_344400 resumed with the
# deploy's fresh networks (the vestibular encoder, both efference heads), one body of its 8-body
# episode, TRAINING, 900 moments, learning progress on with NO hold-out; the replay then lets the
# fresh keys in after N steps. "Dominates" = a fresh key's |progress| is larger than every trained
# sense's at the same step; keys that behaved alike would each be largest equally often, 3 fresh
# of 7 keys = 43 steps in 100.
#   N = 0:  a fresh key was largest on 1.2 of 100 steps (11 of 880: the gaze head, moments 38-73,
#           at 0.009-0.023 against the trained senses' 0.007-0.020); the vestibular sense never;
#           the fresh keys' largest |progress| 0.052 against the trained senses' 95th percentile
#           0.151 (largest 0.162). Its error as a share of persistence's fell from 1.30 to 1.06
#           over moments 0-450 and rose to 1.63 by moment 900; proprioception, a trained sense,
#           fell from 1.80 to 1.17 in the same run.
#   N = 25: 1.1 of 100.  N = 50 to 600: 0 of 100 (the fresh keys' largest |progress| 0.063 at 50,
#           at most 0.10 at 150-300, always under the trained senses' 95th percentile 0.15).
# 50 is the smallest hold-out with no step dominated; it costs 50 moments (2.5 s) in which the
# fresh keys are not counted. One body, one run, before b6's loss balance: the deploy's first
# windows (learning_progress.per_sense) show whether 8 bodies and the balanced loss change it.
LP_HOLDOUT_STEPS: Optional[int] = 50

# THE RING'S DETREND HORIZON, in moments (the blocker of the b3 review: the first build picked the
# oldest moment of the ring as the goal in 42 of 44 choices on its proof run). Each ranked channel
# is stored as its distance from the body's running average over about this many moments. The
# arithmetic: a drift with time constant T keeps tau / (T - tau) of its slope in the stored value
# (the skin's 60 s load build-up, T = 1200 moments: 1 in 100 kept; learning progress's 200-moment
# average: 5 in 100 kept), and a change of state stays in the stored value for about tau moments
# (0.5 s, two orders' tenure). A shorter horizon removes more drift and keeps less of what a
# moment holds; no horizon does both, so it was CHOSEN BY MEASUREMENT (lane-b3-wellness.md, "the
# blocker"; scratchpad lane_b/ring_analyse.py and choose_horizon.json replay the module's own
# ranking; a choice blind to age puts 20 in 100 goals in the two end tenths of the ring), with the
# harm line below:
#   resumed 8-body episode of ckpt_step_344400, 320 goals (sampling error about 4 in 100): first
#     build 29-30 in 100 at the ends; horizon 5: 20-22, 10: 23-24, 15: 22, 20: 24-25, 30: 26-27.
#     Of every 100 parts of the ranking's spread inside a ring, the parts a straight line in age
#     accounts for (half a part for an input with no age in it): first build 26-31; horizon 5: 2-3,
#     10: 5-7, 20: 11-13. What it keeps of the first build's ranking apart from age: of 100 goals,
#     how many lie in the best tenth of the first build's ranking once its straight line in age is
#     removed (a random choice: 10; the first build itself: 97-98; scratchpad lane_b/
#     content_count.py): horizon 5: 52-54, 10: 68-74, 15: 77-78, 20: 78-88, 30: 89-97 (the first
#     number with controllability's gate open, the second shut)
#   b3's proof run (2 bodies, training, a fresh world whose skin load built up), 44 goals: first
#     build 96 in 100 in the oldest tenth; horizon 5: 25 at the ends, 10: 16, 20: 21, 30: 30
#   a channel rising and falling over 60 s (synthetic, 4,480 goals, its own age-blind count 19-21):
#     first build 72 at the ends; horizon 5: 22, 10: 22, 15: 24-25, 20: 25-26, 50: 30
# 10 keeps the strongest drift measured within 2 in 100 of its age-blind count and the deploy's
# population within its sampling error, while keeping 68-74 goals in 100 inside the first build's
# age-removed best tenth; 5 is more age-blind and keeps 52-54; 15-20 keep 77-88 and leave 2-4 in 100
# more at the ends of the strong drift. The lead can trade either way with these numbers.
# LIVE, the fixed code on the resumed 8-body episode (every b3 flag, frozen, 600 moments, 320 goals
# in full rings; lane-b3-wellness.md "Fix 4"): 8.4 in 100 goals in the oldest tenth and 12.2 in the
# newest, 20.6 together, against 31.9 for the first build on the same episode.
# Taking learning progress, controllability and capability out of the ranking instead (the
# review's fix 2) left the ends at 34 in 100 on the 8-body episode and 98 in 100 oldest on b3's
# run, and adding it to horizon 10 or 20 left the 8-body age parts unchanged (8 and 15 in 100): the
# drift that picks the goal by age is integrity's (the skin load), not theirs.
RING_DETREND_STEPS: Optional[int] = 10


def parse_weights(spec) -> Dict[str, float]:
    """DEFAULT_WEIGHTS updated by a --wellness-weights string "name=w,name=w" (or a dict)."""
    w = dict(DEFAULT_WEIGHTS)
    if not spec:
        return w
    items = spec.items() if isinstance(spec, dict) else (
        tuple(p.split("=", 1)) for p in str(spec).split(",") if p.strip())
    for item in items:
        if len(item) != 2:
            raise ValueError(f"--wellness-weights expects name=weight pairs, got {item!r}")
        name, val = item[0].strip(), float(item[1])
        if name not in DEFAULT_WEIGHTS:
            raise ValueError(f"--wellness-weights: unknown channel {name!r}; the channels are "
                             f"{', '.join(CHANNEL_ORDER)}")
        if not math.isfinite(val):
            raise ValueError(f"--wellness-weights: {name} must be a finite number")
        w[name] = val
    return w


# ---------------------------------------------------------------------------- the normaliser
class ChannelNormalizer:
    """Running centre and scale per channel NAME over per-moment values, all bodies pooled.

    Exponential averages of the mean and of the mean square with horizon NORMALISER_HORIZON_STEPS
    model steps, each divided by the weight the average has accumulated (corrected for starting at
    zero), kept in Python floats (float64) so a channel near 0.97 with a spread of 0.003 keeps its
    variance. Mean and standard deviation, not median and MAD: both are exact under an
    exponential average and the channels are bounded (mostly 0-1, consequence 0-3)."""

    def __init__(self, horizon: int = NORMALISER_HORIZON_STEPS):
        self.alpha = 1.0 / float(max(1, int(horizon)))
        self.stats: Dict[str, List[float]] = {}    # name -> [m1, m2, wsum, n]

    def update(self, name: str, values: torch.Tensor) -> None:
        v = values.detach().float().reshape(-1)
        v = v[torch.isfinite(v)]
        if v.numel() == 0:
            return
        x1 = float(v.double().mean())
        x2 = float((v.double() ** 2).mean())
        a = self.alpha
        s = self.stats.setdefault(name, [0.0, 0.0, 0.0, 0])
        s[0] = (1.0 - a) * s[0] + a * x1
        s[1] = (1.0 - a) * s[1] + a * x2
        s[2] = (1.0 - a) * s[2] + a
        s[3] = int(s[3]) + int(v.numel())

    def centre_scale(self, name: str) -> Tuple[float, float]:
        s = self.stats.get(name)
        if not s or s[2] <= 0.0:
            return float("nan"), float("nan")
        c = s[0] / s[2]
        var = max(s[1] / s[2] - c * c, 0.0)
        return c, math.sqrt(var)

    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "stats": {k: list(v) for k, v in self.stats.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.stats = {str(k): [float(v[0]), float(v[1]), float(v[2]), int(v[3])]
                      for k, v in (sd or {}).get("stats", {}).items()}


@torch.no_grad()
def detrend_moment(state: Dict[str, list], names: Sequence[str], raw: torch.Tensor,
                   horizon: Optional[int]) -> torch.Tensor:
    """What the goal ring stores for one moment: [C, B] in `names` order, on raw's device.

    Each ranked channel is stored as its value minus THIS BODY's running average of the channel
    over the moments before (an exponential average with `horizon` moments, corrected for starting
    at zero), so a drift that moves the value and its average together gives every moment of a ring
    the same stored value (RING_DETREND_STEPS says why and how the horizon was measured). The
    unranked rows (pain, harm) are stored raw: the gate judges the harm itself. A body's first
    moment has no average yet and stores NaN (= the channel's centre in the ranking); an unmeasured
    value stores NaN and leaves the average unchanged. horizon None stores every row raw.

    state: {name: [sum, weight]} per-body CPU tensors [B], updated in place. It belongs to the
    bodies' history (it rides the episode companion beside the ring and is cleared with it)."""
    out = raw.detach().float().clone()
    if horizon is None:
        return out
    a = 1.0 / float(max(1, int(horizon)))
    x = out.cpu()
    B = x.shape[1]
    for ci, name in enumerate(names):
        if name in UNRANKED_ROWS:
            continue
        v = x[ci].clone()             # x[ci] is overwritten below; the average learns the value
        s = state.get(name)
        if (not isinstance(s, (list, tuple)) or len(s) != 2 or not torch.is_tensor(s[0])
                or s[0].shape[0] != B):
            s = [torch.zeros(B), torch.zeros(B)]
        m, w = s[0].detach().float().cpu(), s[1].detach().float().cpu()
        avg = torch.where(w > 0, m / w.clamp_min(1e-12), torch.full_like(m, float("nan")))
        x[ci] = v - avg                                  # NaN where either is unmeasured
        ok = torch.isfinite(v)
        m = torch.where(ok, (1.0 - a) * m + a * v, m)
        w = torch.where(ok, (1.0 - a) * w + a, w)
        state[name] = [m, w]
    return x.to(out.device)


def rank_values(ch: torch.Tensor, names: Sequence[str], weights: Dict[str, float],
                norm: ChannelNormalizer, floors: Dict[str, Optional[float]],
                gate_line: Optional[float]):
    """The normalised ranking of remembered moments.

    ch: [T, C, B] per-moment channel values in `names` order, as the ring stores them; the row
    named HARM_ROW (or, in a list without it, PAIN_ROW) is the gate's, raw. A value that is not
    finite (a channel unmeasured at that moment) contributes 0, which is the channel's centre, and
    an unmeasured harm never gates. Returns (ranked [T, B] with -inf on gated moments, contrib
    [T, C, B] the weighted normalised contributions (0 on the unranked rows), gated [T, B] bool,
    all_gated [B] bool). A body whose every candidate is gated is given its least harmful moment
    (ties by value): a planner must aim somewhere, and "least over the line" is the gate's own
    order."""
    T, C, B = ch.shape
    contrib = torch.zeros(T, C, B, dtype=torch.float32, device=ch.device)
    for ci, name in enumerate(names):
        if name in UNRANKED_ROWS:
            continue
        w = float(weights.get(name, 0.0))
        if w == 0.0:
            continue
        c, s = norm.centre_scale(name)
        if not math.isfinite(c):
            continue
        f = floors.get(name)
        den = max(s, float(f)) if f is not None else s
        if not (den > 0.0):
            continue
        v = ch[:, ci].float()
        contrib[:, ci] = torch.where(torch.isfinite(v), w * (v - c) / den, torch.zeros_like(v))
    value = contrib.sum(dim=1)                                           # [T, B]
    gate_row = HARM_ROW if HARM_ROW in names else PAIN_ROW
    if gate_row in names and gate_line is not None:
        pain = ch[:, list(names).index(gate_row)].float()
        gated = torch.isfinite(pain) & (pain > float(gate_line))
    else:
        pain = None
        gated = torch.zeros(T, B, dtype=torch.bool, device=ch.device)
    ranked = value.masked_fill(gated, float("-inf"))
    all_gated = gated.all(dim=0)
    if bool(all_gated.any()) and pain is not None:
        least = pain.masked_fill(~torch.isfinite(pain), float("inf"))
        is_least = least == least.min(dim=0, keepdim=True).values
        fallback = value.masked_fill(~is_least, float("-inf"))
        ranked = torch.where(all_gated.view(1, B), fallback, ranked)
    return ranked, contrib, gated, all_gated


class RankShares:
    """How much of the ranking each channel carries, accumulated over planning events.

    variance_share[c]: over each body's non-gated candidates at a planning event, the covariance
      of channel c's contribution with the total value, divided by the total's variance, summed
      over bodies and events before dividing. The shares add to 1; a channel with 0.9 decides
      90 parts in 100 of how the candidates are spread.
    winner_margin_share[c]: the chosen moment's lead over the runner-up, split by channel,
      summed over events before dividing: which channels made the winner win. The shares add to
      1; a channel can push against the winner (below 0) while others push for it by more than
      the whole lead (above 1).
    integrity_gated_share: candidates the gate removed / candidates seen.
    all_gated_share: body-events whose every candidate was over the gate line.
    goal_age: how far back in the ring each body's goal sat (add_goal), the direct reading of the
      b3 review's blocker: mean_share_back 0 = the newest moment of the ring, 1 = the oldest;
      oldest_tenth_share / newest_tenth_share = goals whose share back was at least 0.9 / at most
      0.1. A choice that ignores age puts 10 in 100 goals in each of the two end tenths (20 in 100
      together); the as-built ranking put 42 of 44 in the oldest tenth on b3's proof run."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cov: Dict[str, float] = {}
        self.var = 0.0
        self.margin_parts: Dict[str, float] = {}
        self.margin = 0.0
        self.candidates = 0
        self.gated = 0
        self.body_events = 0
        self.all_gated = 0
        self.events = 0
        self.goals = 0
        self.goal_back = 0.0
        self.goal_oldest = 0
        self.goal_newest = 0

    def add_goal(self, index: Sequence[int], T: int) -> None:
        """Count where each body's chosen goal sat: index into a ring of T moments (0 = oldest,
        T - 1 = newest), one per body."""
        if T < 2:
            return
        for i in index:
            back = (T - 1 - int(i)) / float(T - 1)       # 0 = newest, 1 = oldest
            self.goals += 1
            self.goal_back += back
            self.goal_oldest += int(back >= 0.9)
            self.goal_newest += int(back <= 0.1)

    @torch.no_grad()
    def add(self, contrib: torch.Tensor, ranked: torch.Tensor, gated: torch.Tensor,
            all_gated: torch.Tensor, names: Sequence[str]) -> None:
        T, C, B = contrib.shape
        self.events += 1
        self.candidates += int(T * B)
        self.gated += int(gated.sum())
        self.body_events += int(B)
        self.all_gated += int(all_gated.sum())
        total = contrib.sum(dim=1)
        for b in range(B):
            ok = torch.isfinite(ranked[:, b])
            n = int(ok.sum())
            if n < 2:
                continue
            u = contrib[ok][:, :, b].double()                  # [n, C]
            V = total[ok, b].double()                          # [n]
            dV = V - V.mean()
            self.var += float((dV * dV).mean())
            du = u - u.mean(dim=0, keepdim=True)
            cov = (du * dV.unsqueeze(1)).mean(dim=0)
            for ci, name in enumerate(names):
                if name not in UNRANKED_ROWS:
                    self.cov[name] = self.cov.get(name, 0.0) + float(cov[ci])
            r = ranked[:, b].double().masked_fill(~ok, float("-inf"))
            top = torch.topk(r, 2)
            i, j = int(top.indices[0]), int(top.indices[1])
            d = (contrib[i, :, b] - contrib[j, :, b]).double()
            self.margin += float(d.sum())
            for ci, name in enumerate(names):
                if name not in UNRANKED_ROWS:
                    self.margin_parts[name] = self.margin_parts.get(name, 0.0) + float(d[ci])

    def summary(self, nd: int = 4) -> dict:
        def share(parts, den):
            if not parts or abs(den) < 1e-12:
                return None
            return {k: round(v / den, nd) for k, v in parts.items()}
        return {"variance_share": share(self.cov, self.var),
                "winner_margin_share": share(self.margin_parts, self.margin),
                "integrity_gated_share": (round(self.gated / self.candidates, nd)
                                          if self.candidates else None),
                "all_gated_share": (round(self.all_gated / self.body_events, nd)
                                    if self.body_events else None),
                "planning_events": int(self.events),
                "goal_age": ({"mean_share_back": round(self.goal_back / self.goals, nd),
                              "oldest_tenth_share": round(self.goal_oldest / self.goals, nd),
                              "newest_tenth_share": round(self.goal_newest / self.goals, nd),
                              "events": int(self.goals)} if self.goals else None)}


class WellnessChannelState:
    """Everything the channels keep across steps, in one object the model holds under a PUBLIC
    name (LemLite.wellness_channels): continuation_state.capture_learner copies only _* and last_*
    attributes into the episode companion, and this state rides runtime_state.pt beside the
    checkpoint instead (key 'wellness_channels'), read back with defaults - a checkpoint without
    it starts cold. The per-window sums are not saved: they are a count of this window."""

    def __init__(self, normalized: bool, control: Optional["ControllabilityTracker"],
                 lp: Optional["LearningProgressTracker"]):
        self.normalizer = ChannelNormalizer() if normalized else None
        self.control = control
        self.lp = lp
        self.shares = RankShares() if normalized else None
        self.raw_win: Dict[str, List[float]] = {}
        self.last_B = 1        # the batch the report's per-body lists are cut to
        # this moment's channel values BEFORE the ring's detrend ([C, B], CPU, in the model's
        # rank_channel_names order); the ring holds the stored form. Readers (the explainer) that
        # want the values themselves read this
        self.last_raw: Optional[torch.Tensor] = None
        self._cost_reset()

    def _cost_reset(self) -> None:
        self.cost = {"step_ms": 0.0, "steps": 0, "plan_ms": 0.0, "plans": 0}

    def add_cost(self, ms: float, planning: bool = False, new_step: bool = False) -> None:
        """Wall time the channels' own code took (perf_counter around it; on the graphics card it
        includes the waits its reads of card memory cause). The per-step work is timed in more
        than one place per model step; new_step=True marks the one call made once per step
        (LemLite._observe_wellness_channels), so per_step_ms is per MODEL STEP, not per call."""
        if planning:
            self.cost["plan_ms"] += float(ms)
            self.cost["plans"] += 1
        else:
            self.cost["step_ms"] += float(ms)
            self.cost["steps"] += int(bool(new_step))

    def cost_window(self) -> Optional[dict]:
        """{per_step_ms, planning_ms, steps, planning_events} since the last call, and a fresh
        window. per_step_ms = the per-moment work (learning progress, controllability, the
        channel vector, the normaliser) averaged over the window's steps; planning_ms = the ranking
        of the ring, averaged over the planning events."""
        c = self.cost
        out = {"per_step_ms": round(c["step_ms"] / c["steps"], 3) if c["steps"] else None,
               "planning_ms": round(c["plan_ms"] / c["plans"], 3) if c["plans"] else None,
               "steps": int(c["steps"]), "planning_events": int(c["plans"])}
        self._cost_reset()
        return out

    def forget_pending(self) -> None:
        if self.control is not None:
            self.control.forget_pending()

    def add_raw(self, name: str, values: torch.Tensor) -> None:
        v = values.detach().float().reshape(-1)
        v = v[torch.isfinite(v)]
        if v.numel():
            s = self.raw_win.setdefault(name, [0.0, 0])
            s[0] += float(v.double().sum())
            s[1] += int(v.numel())

    def raw_summary(self, nd: int = 5) -> Dict[str, float]:
        out = {k: round(s[0] / s[1], nd) for k, s in self.raw_win.items() if s[1]}
        self.raw_win = {}
        return out

    def state_dict(self) -> dict:
        return {"version": WELLNESS_CHANNELS_VERSION,
                "normalizer": None if self.normalizer is None else self.normalizer.state_dict(),
                "controllability": None if self.control is None else self.control.state_dict(),
                "learning_progress": None if self.lp is None else self.lp.state_dict()}

    def load_state_dict(self, sd: dict) -> List[str]:
        """Restore what the saved state has for the parts this build runs; returns their names."""
        got = []
        sd = sd or {}
        if self.normalizer is not None and sd.get("normalizer"):
            self.normalizer.load_state_dict(sd["normalizer"])
            got.append("normalizer")
        if self.control is not None and sd.get("controllability"):
            self.control.load_state_dict(sd["controllability"])
            got.append("controllability")
        if self.lp is not None and sd.get("learning_progress"):
            self.lp.load_state_dict(sd["learning_progress"])
            got.append("learning_progress")
        return got


# ---------------------------------------------------------------------------- controllability
def effect_vector(cm: torch.Tensor, deg: torch.Tensor, pos_mask: torch.Tensor,
                  rot_mask: torch.Tensor) -> torch.Tensor:
    """[B, K] one body's effect over a tenure: the selected handles' position change (cm) over
    TENURE_CM and rotation (degrees, rotation vector) over TENURE_DEG, flattened."""
    parts = []
    if bool(pos_mask.any()):
        parts.append((cm[:, pos_mask].float() / TENURE_CM).reshape(cm.shape[0], -1))
    if bool(rot_mask.any()):
        parts.append((deg[:, rot_mask].float() / TENURE_DEG).reshape(deg.shape[0], -1))
    return torch.cat(parts, dim=-1)


def agreement(expected: torch.Tensor, achieved: torch.Tensor) -> torch.Tensor:
    """[B] direction cosine x (smaller / larger magnitude). -1 = the opposite effect, 0 =
    unrelated or nothing where something was expected, +1 = exactly the expected effect."""
    ne = expected.norm(dim=-1)
    na = achieved.norm(dim=-1)
    cos = (expected * achieved).sum(-1) / (ne * na).clamp_min(1e-12)
    ratio = torch.minimum(ne, na) / torch.maximum(ne, na).clamp_min(1e-12)
    return torch.where((ne > 0) & (na > 0), cos * ratio, torch.zeros_like(cos))


class ControllabilityTracker:
    """Per body, per latched order: did the body do what was expected of the order?

    step() is called once per model step with the order in force during the burst that produced
    this step's frames (exec_t, execution units: position cm, quaternion x y z w, radius,
    strength), its serial number, this step's achieved pose [B, H, >=7] (positions / 200 cm,
    world quaternion) and the author of the order latched at the PREVIOUS step. An order is graded
    the step the next one appears, when it ran exactly `hold` moments."""

    def __init__(self, n_handles: int, gaze_idx: int, hold: int,
                 floors: Dict[str, Optional[float]], ema_orders: int = CONTROL_EMA_ORDERS,
                 expect_efference: bool = False):
        self.hold = max(1, int(hold))
        # with the efference heads on, every order is judged against the HEAD's expectation; an
        # order whose head record is missing is skipped and counted, never judged by its command
        # instead, so one window never mixes the two sources under one label
        self.expect_efference = bool(expect_efference)
        self.gaze_idx = int(gaze_idx)
        body = torch.ones(int(n_handles), dtype=torch.bool)
        body[self.gaze_idx] = False
        self.body_mask = body                  # every handle but the eyes
        self.no_mask = torch.zeros(int(n_handles), dtype=torch.bool)
        self.floors = dict(floors)
        self.alpha = 1.0 / float(max(1, int(ema_orders)))
        self.ema_m: Optional[torch.Tensor] = None      # [B]
        self.ema_w: Optional[torch.Tensor] = None      # [B]
        from collections import deque
        self.gate_readings = deque(maxlen=CONTROL_GATE_READINGS)   # per order: mean agreement - control
        # the gate's state: it opens above the opening line and stays open until the closing line
        self.gate_is_open = False
        # THE SHADOW GATE (SHADOW_CONTROL_KIND; recorded only): per graded command, mean agreement - the
        # size-matched control, and its own open / shut state under the same two lines. Read by summary() alone
        self.shadow_gate_readings = deque(maxlen=CONTROL_GATE_READINGS)
        self.shadow_gate_is_open = False
        self.pending: Optional[dict] = None
        self.prev_ach: Optional[torch.Tensor] = None
        self.prev_id: Optional[int] = None
        self.prev_order: Optional[torch.Tensor] = None
        self.last: dict = {}
        self._win_reset()

    def _win_reset(self):
        self.win = {"orders": 0, "measured": 0, "agree": 0.0, "control": 0.0, "control_n": 0,
                    "diff": 0.0, "command_agree": 0.0, "command_n": 0, "by_author": {},
                    "source": None, "tenure_mismatch": 0, "below_floor": 0,
                    "efference_missing": 0,
                    # the size-matched control on the measured body-commands, and the two gates' counts
                    # in graded commands (CONTROL_GATE_DEFINITIONS["counts"])
                    "size_matched": 0.0, "size_matched_diff": 0.0, "size_matched_n": 0,
                    "gate_commands": 0, "gate_open_commands": 0,
                    "shadow_gate_commands": 0, "shadow_gate_open_commands": 0,
                    "gates_same_commands": 0, "gates_both_open_commands": 0,
                    "gate_only_open_commands": 0, "shadow_gate_only_open_commands": 0}

    def forget_pending(self):
        """A reset or teleport: no order may be graded across it (the averages are kept)."""
        self.pending = None
        self.prev_ach = None
        self.prev_id = None
        self.prev_order = None

    @staticmethod
    def _readings_stats(readings):
        """(n, mean, standard error, readings above zero) of one gate's readings (the gate in force
        and the shadow gate share this arithmetic); mean, standard error and the count are None
        until all CONTROL_GATE_READINGS readings exist."""
        n = len(readings)
        if n < CONTROL_GATE_READINGS:
            return n, None, None, None
        vals = list(readings)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        return n, mean, math.sqrt(var / n), sum(1 for v in vals if v > 0)

    @staticmethod
    def _two_lines(is_open: bool, mean: Optional[float], se: Optional[float]) -> bool:
        """The two-line rule both gates use: the state after one reading, from the state before it
        and the readings' mean and standard error. Shut until all readings exist; a shut gate opens
        above CONTROL_GATE_OPEN_NOISE standard errors; an open gate shuts under
        CONTROL_GATE_CLOSE_NOISE."""
        if mean is None:
            return False
        if is_open:
            return not (mean < CONTROL_GATE_CLOSE_NOISE * se)
        return bool(mean > CONTROL_GATE_OPEN_NOISE * se)

    def _gate_stats(self):
        """(n, mean, standard error, readings above zero) of the last CONTROL_GATE_READINGS."""
        return self._readings_stats(self.gate_readings)

    def _update_gate(self) -> None:
        """Once per graded order: open above CONTROL_GATE_OPEN_NOISE standard errors, close under
        CONTROL_GATE_CLOSE_NOISE (the two lines; see CONTROL_GATE_READINGS)."""
        n, mean, se, _ = self._gate_stats()
        self.gate_is_open = self._two_lines(self.gate_is_open, mean, se)

    def _shadow_gate_stats(self):
        """(n, mean, standard error, readings above zero) of the shadow gate's last
        CONTROL_GATE_READINGS readings (agreement - the size-matched control)."""
        return self._readings_stats(self.shadow_gate_readings)

    def _update_shadow_gate(self) -> None:
        """Once per graded command that carried the size-matched expectation: the same two lines on
        the shadow gate's own readings. Changes shadow_gate_is_open and nothing else."""
        n, mean, se, _ = self._shadow_gate_stats()
        self.shadow_gate_is_open = self._two_lines(self.shadow_gate_is_open, mean, se)

    def shadow_gate(self) -> Tuple[bool, Optional[float], Optional[float], Optional[int], int]:
        """(open, mean of agreement - the size-matched control, its standard error, readings above
        zero, readings held) of the shadow gate. For the stats block and tests: no weight, ranking
        or loss may read it (gate() is the gate in force)."""
        n, mean, se, above = self._shadow_gate_stats()
        return bool(self.shadow_gate_is_open), mean, se, above, n

    def gate(self) -> Tuple[bool, Optional[float], Optional[float], str]:
        """(open, mean of agreement - control, the line in force, why) over the last
        CONTROL_GATE_READINGS measured body-orders. The line in force is the opening line while
        the gate is shut and the closing line while it is open."""
        n, mean, se, above = self._gate_stats()
        if mean is None:
            return False, None, None, ("%d of the %d orders the gate judges have been measured "
                                       "beside the shuffled-order control" % (n, CONTROL_GATE_READINGS))
        ok = bool(self.gate_is_open)
        line = (CONTROL_GATE_CLOSE_NOISE if ok else CONTROL_GATE_OPEN_NOISE) * se
        times = (mean / se) if se > 0 else (float("inf") if mean > 0 else 0.0)
        return ok, mean, line, (
            "%d of the last %d orders agreed better with what the model expected from the order "
            "it sent than with what it would have expected from another body's order (a coin flip "
            "gives %d of %d). The average advantage is %.1f times its own measurement noise; the "
            "gate opens above %g times and, once open, closes below %g time%s. It is %s."
            % (above, n, n // 2, n, times, CONTROL_GATE_OPEN_NOISE, CONTROL_GATE_CLOSE_NOISE,
               "" if CONTROL_GATE_CLOSE_NOISE == 1 else "s", "open" if ok else "shut"))

    def value(self, B: int, device) -> torch.Tensor:
        """[B] the running controllability per body; NaN where no order has been measured."""
        if self.ema_m is None or self.ema_m.shape[0] != B:
            return torch.full((B,), float("nan"), device=device)
        v = self.ema_m / self.ema_w.clamp_min(1e-12)
        return torch.where(self.ema_w > 0, v, torch.full_like(v, float("nan"))).to(device)

    @torch.no_grad()
    def step(self, step: int, exec_t: Optional[torch.Tensor], order_id: Optional[int],
             ach: torch.Tensor, author_prev: Optional[str], efference_tenure: Optional[dict]):
        from lem.model.efference import pose_change, rotation_change_deg
        # one copy of this moment's pose and order to the processor; everything below runs there
        ach = ach.detach().float().cpu()
        B = ach.shape[0]
        if exec_t is None:
            self.prev_ach = ach
            return None
        exec_t = exec_t.detach().float().cpu()
        gi = self.gaze_idx
        if order_id is not None:
            changed = self.prev_id is None or int(order_id) != self.prev_id
        else:
            limbs = torch.cat([exec_t[:, :gi], exec_t[:, gi + 1:]], dim=1)
            prev = self.prev_order
            changed = (prev is None or prev.shape != exec_t.shape or not torch.equal(
                torch.cat([prev[:, :gi], prev[:, gi + 1:]], dim=1), limbs))
        graded = None
        if changed:
            p = self.pending
            if (p is not None and self.prev_ach is not None and p["ach"].shape[0] == B
                    and self.prev_ach.shape[0] == B):
                if (int(step) - 1) - int(p["step"]) == self.hold:
                    graded = self._grade(p, self.prev_ach, efference_tenure, int(step), B,
                                         pose_change, rotation_change_deg)
                else:
                    self.win["tenure_mismatch"] += 1
            if self.prev_ach is not None and self.prev_ach.shape[0] == B:
                self.pending = {"ach": self.prev_ach.clone(), "order": exec_t.clone(),
                                "step": int(step) - 1, "author": author_prev or "unknown"}
            else:
                self.pending = None
        self.prev_id = None if order_id is None else int(order_id)
        self.prev_order = exec_t.clone()
        self.prev_ach = ach
        return graded

    def _grade(self, p, end_ach, eff, step, B, pose_change, rotation_change_deg):
        begin = p["ach"]
        order = p["order"]
        ach_cm, ach_deg = pose_change(begin, end_ach)
        cmd_cm = order[..., 0:3] - begin[..., 0:3] * 200.0
        cmd_deg = rotation_change_deg(begin[..., 3:7], order[..., 3:7])
        # the command's effect is its commanded ROTATION of every body handle (DESIGN D6): the
        # position an order names is clipped by the decoder to the handle's reach, so it is not
        # what the body is asked to do (measured on ckpt_step_344400, 136 of the model's own
        # orders: the named position sat a median 18.6 cm per axis from the hand where the order
        # began, root mean square over the 14 position handles, while the hands moved a median
        # 1.7 cm over the order); the achieved effect on the same handles
        C_vec = effect_vector(cmd_cm, cmd_deg, self.no_mask, self.body_mask)
        A_cmd = effect_vector(ach_cm, ach_deg, self.no_mask, self.body_mask)
        f_cmd = self.floors.get("command")
        cmd_ok = C_vec.norm(dim=-1) > float(f_cmd)
        use_eff = (eff is not None and int(eff.get("step", -1)) == int(step)
                   and int(eff.get("order_began", -2)) == int(p["step"])
                   and torch.is_tensor(eff.get("expected_cm"))
                   and eff["expected_cm"].shape[0] == B)
        if self.expect_efference and not use_eff:
            # an order the gauge could not judge still counts among the orders: measured_share is
            # a share of ORDERS, and this one was not measured
            self.win["efference_missing"] += 1
            self.win["orders"] += int(B)
            return None
        E_shuf = None
        if use_eff:
            E = effect_vector(eff["expected_cm"].detach().float().cpu(),
                              eff["expected_deg"].detach().float().cpu(),
                              self.body_mask, self.body_mask)
            A = effect_vector(ach_cm, ach_deg, self.body_mask, self.body_mask)
            if (B > 1 and torch.is_tensor(eff.get("expected_shuffled_cm"))
                    and eff["expected_shuffled_cm"].shape[0] == B):
                # the head's expectation from THIS body's latent with ANOTHER body's order
                E_shuf = effect_vector(eff["expected_shuffled_cm"].detach().float().cpu(),
                                       eff["expected_shuffled_deg"].detach().float().cpu(),
                                       self.body_mask, self.body_mask)
            f_exp = self.floors.get("efference")
            source = "efference"
        else:
            E, A, f_exp, source = C_vec, A_cmd, f_cmd, "command"
            if B > 1:
                # ANOTHER body's order taken from THIS body's own starting pose: the turn from
                # where this body's handles were to the orientations the other order names
                sh_deg = rotation_change_deg(begin[..., 3:7], order.roll(1, dims=0)[..., 3:7])
                E_shuf = effect_vector(cmd_cm, sh_deg, self.no_mask, self.body_mask)
        exp_ok = E.norm(dim=-1) > float(f_exp)
        measured = cmd_ok & exp_ok
        agr = agreement(E, A)
        ctl = None
        if E_shuf is not None:
            # the shuffled-ORDER control on the same measured orders: an expectation that ignores
            # the order is identical under both orders, so it reads exactly 0 here
            ctl = torch.where(measured, agreement(E_shuf, A), torch.full_like(agr, float("nan")))
            if bool(measured.any()):
                # ONE reading per order: the mean over the bodies measured on it. The bodies of one
                # order share the orders the control is built from, so their readings are not
                # independent, and a standard error over body-orders would read too small
                # (lane-b3-wellness.md, "gate with two lines": with half of each reading's spread
                # shared inside an order, pooled body-orders stood open on 17 of 100 planning
                # events with no control at all)
                self.gate_readings.append(float((agr - ctl)[measured].mean()))
                self._update_gate()
        # THE SHADOW GATE'S READING (recorded only; SHADOW_CONTROL_KIND): the same reading with the
        # size-matched expectation in place of the shuffled one, on the same measured bodies. It
        # changes shadow_gate_readings, shadow_gate_is_open and the window's counts, nothing else
        ctl_sm = None
        sm_cm = sm_deg = None
        if use_eff and B > 1:
            sm_cm, sm_deg = eff.get("expected_size_matched_cm"), eff.get("expected_size_matched_deg")
        if (torch.is_tensor(sm_cm) and torch.is_tensor(sm_deg) and sm_cm.shape[0] == B
                and sm_deg.shape[0] == B):
            # the head's expectation from THIS body's latent and pose with another body's command
            # size-matched to this body (efference.size_matched_other_order)
            E_sm = effect_vector(sm_cm.detach().float().cpu(), sm_deg.detach().float().cpu(),
                                 self.body_mask, self.body_mask)
            ctl_sm = torch.where(measured, agreement(E_sm, A), torch.full_like(agr, float("nan")))
            if bool(measured.any()):
                self.shadow_gate_readings.append(float((agr - ctl_sm)[measured].mean()))
                self._update_shadow_gate()
        cmd_agr = agreement(C_vec, A_cmd)
        # the running average, measured orders only
        if self.ema_m is None or self.ema_m.shape[0] != B:
            self.ema_m = torch.zeros(B)
            self.ema_w = torch.zeros(B)
        a = self.alpha
        m = measured
        self.ema_m = torch.where(m, (1 - a) * self.ema_m + a * agr, self.ema_m)
        self.ema_w = torch.where(m, (1 - a) * self.ema_w + a, self.ema_w)
        w = self.win
        w["orders"] += int(B)
        w["measured"] += int(measured.sum())
        w["below_floor"] += int((~measured).sum())
        w["source"] = source
        if bool(measured.any()):
            w["agree"] += float(agr[measured].sum())
            ca = cmd_agr[measured & cmd_ok]
            w["command_agree"] += float(ca.sum())
            w["command_n"] += int(ca.numel())
            if ctl is not None:
                w["control"] += float(ctl[measured].sum())
                w["diff"] += float((agr - ctl)[measured].sum())
                w["control_n"] += int(measured.sum())
            if ctl_sm is not None:
                w["size_matched"] += float(ctl_sm[measured].sum())
                w["size_matched_diff"] += float((agr - ctl_sm)[measured].sum())
                w["size_matched_n"] += int(measured.sum())
            # THE TWO GATES' COUNTS, in graded commands (CONTROL_GATE_DEFINITIONS["counts"]): a gate
            # took a reading at this command when its control existed (measured bodies exist here),
            # and its state is the one that reading left it in
            took, took_sm = ctl is not None, ctl_sm is not None
            if took:
                w["gate_commands"] += 1
                w["gate_open_commands"] += int(self.gate_is_open)
            if took_sm:
                w["shadow_gate_commands"] += 1
                w["shadow_gate_open_commands"] += int(self.shadow_gate_is_open)
            if took and took_sm:
                w["gates_same_commands"] += 1
                w["gates_both_open_commands"] += int(self.gate_is_open and self.shadow_gate_is_open)
                w["gate_only_open_commands"] += int(self.gate_is_open
                                                    and not self.shadow_gate_is_open)
                w["shadow_gate_only_open_commands"] += int(self.shadow_gate_is_open
                                                           and not self.gate_is_open)
            s = w["by_author"].setdefault(str(p["author"]), [0.0, 0])
            s[0] += float(agr[measured].sum())
            s[1] += int(measured.sum())
        self.last = {"step": int(step), "source": source, "order_began": int(p["step"]),
                     "agreement": [round(float(x), 4) if bool(mm) else None
                                   for x, mm in zip(agr.tolist(), measured.tolist())],
                     "control": (None if ctl is None else
                                 [round(float(x), 4) if bool(mm) else None
                                  for x, mm in zip(ctl.tolist(), measured.tolist())]),
                     "author": str(p["author"])}
        return self.last

    def summary(self, B: int) -> dict:
        """The window's block. control_mean: the SHUFFLED-ORDER control (this body's state and
        result, another body's order), on the same measured orders as agreement_mean;
        difference_mean: agreement - control on those orders (-2 to +2; 0 = the order made no
        difference to what was expected); measured_share: measured orders / orders.
        Since 2026-10-03 the block also carries the size-matched control and the shadow gate fed by
        it, recorded only, and both gates' counts in graded commands; gate_definitions
        (CONTROL_GATE_DEFINITIONS) says what each of those keys is."""
        w = self.win
        v = self.value(B, "cpu")
        fin = torch.isfinite(v)
        out = {"value_mean": round(float(v[fin].mean()), 4) if bool(fin.any()) else None,
               "per_body": [None if not math.isfinite(x) else round(x, 4) for x in v.tolist()],
               "measured_share": (round(w["measured"] / w["orders"], 4) if w["orders"] else None),
               "agreement_mean": (round(w["agree"] / w["measured"], 4) if w["measured"] else None),
               "control_mean": (round(w["control"] / w["control_n"], 4) if w["control_n"] else None),
               "difference_mean": (round(w["diff"] / w["control_n"], 4) if w["control_n"] else None),
               "control_kind": "shuffled order",
               "command_agreement_mean": (round(w["command_agree"] / w["command_n"], 4)
                                          if w["command_n"] else None),
               "by_author": {k: round(s[0] / s[1], 4) for k, s in w["by_author"].items() if s[1]},
               "by_author_orders": {k: int(s[1]) for k, s in w["by_author"].items()},
               "orders": int(w["orders"]), "source": w["source"],
               "tenure_mismatch": int(w["tenure_mismatch"]),
               "efference_missing": int(w["efference_missing"]),
               "floors": dict(self.floors)}
        g_open, g_mean, g_line, g_why = self.gate()
        out.update({"gate_open": bool(g_open),
                    "gate_mean": None if g_mean is None else round(g_mean, 5),
                    "gate_line": None if g_line is None else round(g_line, 5),
                    "gate_lines_noise": {"open": CONTROL_GATE_OPEN_NOISE,
                                         "close": CONTROL_GATE_CLOSE_NOISE},
                    "gate_why": g_why})
        # THE SIZE-MATCHED CONTROL, THE SHADOW GATE AND BOTH GATES' COUNTS (recorded only; every key
        # is defined in gate_definitions). The gate in force gets its standard error and its
        # readings above zero beside the shadow gate's, so the two are read like for like
        g_n, _g_mean, g_se, g_above = self._gate_stats()
        s_open, s_mean, s_se, s_above, s_n = self.shadow_gate()
        out.update({
            "gate_standard_error": None if g_se is None else round(g_se, 5),
            "gate_readings_above_zero": None if g_above is None else int(g_above),
            "gate_readings_held": int(g_n),
            "size_matched_control_mean": (round(w["size_matched"] / w["size_matched_n"], 4)
                                          if w["size_matched_n"] else None),
            "size_matched_difference_mean": (round(w["size_matched_diff"] / w["size_matched_n"], 4)
                                             if w["size_matched_n"] else None),
            "shadow_control_kind": SHADOW_CONTROL_KIND,
            "shadow_gate_open": bool(s_open),
            "shadow_gate_mean": None if s_mean is None else round(s_mean, 5),
            "shadow_gate_standard_error": None if s_se is None else round(s_se, 5),
            "shadow_gate_readings_above_zero": None if s_above is None else int(s_above),
            "shadow_gate_readings_held": int(s_n),
            "gate_commands": int(w["gate_commands"]),
            "gate_open_commands": int(w["gate_open_commands"]),
            "shadow_gate_commands": int(w["shadow_gate_commands"]),
            "shadow_gate_open_commands": int(w["shadow_gate_open_commands"]),
            "gates_same_commands": int(w["gates_same_commands"]),
            "gates_both_open_commands": int(w["gates_both_open_commands"]),
            "gate_only_open_commands": int(w["gate_only_open_commands"]),
            "shadow_gate_only_open_commands": int(w["shadow_gate_only_open_commands"]),
            "gate_definitions": dict(CONTROL_GATE_DEFINITIONS)})
        self._win_reset()
        return out

    def state_dict(self) -> dict:
        return {"ema_m": None if self.ema_m is None else self.ema_m.clone(),
                "ema_w": None if self.ema_w is None else self.ema_w.clone(),
                "alpha": self.alpha, "gate_readings": [float(v) for v in self.gate_readings],
                "control_kind": "shuffled order",
                # the shadow gate's readings under their own control kind (a build without the
                # shadow gate ignores both keys)
                "shadow_gate_readings": [float(v) for v in self.shadow_gate_readings],
                "shadow_control_kind": SHADOW_CONTROL_KIND}

    def load_state_dict(self, sd: dict) -> None:
        """The running averages come back; the gate's readings come back only when they were taken
        against the shuffled-ORDER control (a v1 checkpoint's readings used another body's
        expectation, the control the b3 review showed an order-blind head beats). The gate starts
        shut either way and reopens only when the readings clear the opening line.
        The shadow gate follows the same rule on its own keys: its readings come back only when
        they were taken against the size-matched control (shadow_control_kind), it starts shut and
        reopens only when they clear the opening line. A state saved before the shadow gate existed
        carries neither key: the shadow gate loads empty and shut."""
        sd = sd or {}
        m, w = sd.get("ema_m"), sd.get("ema_w")
        if torch.is_tensor(m) and torch.is_tensor(w) and m.shape == w.shape:
            self.ema_m, self.ema_w = m.float().cpu(), w.float().cpu()
        self.gate_readings.clear()
        if sd.get("control_kind") == "shuffled order":
            for v in (sd.get("gate_readings") or [])[-CONTROL_GATE_READINGS:]:
                self.gate_readings.append(float(v))
        self.gate_is_open = False
        self._update_gate()
        self.shadow_gate_readings.clear()
        if sd.get("shadow_control_kind") == SHADOW_CONTROL_KIND:
            for v in (sd.get("shadow_gate_readings") or [])[-CONTROL_GATE_READINGS:]:
                self.shadow_gate_readings.append(float(v))
        self.shadow_gate_is_open = False
        self._update_shadow_gate()


# ---------------------------------------------------------------------------- learning progress
class LearningProgressTracker:
    """Per body, per sense (and per efference head): r = model error / persistence error.

    Fast and slow exponential averages of each error separately, each divided by the weight it has
    accumulated (corrected for starting at zero). r = the ratio of the averages, NOT the average of
    per-moment ratios: a moment where a sense did not change has a persistence error near 0 and its
    own ratio is unbounded (measured: see b3-wellness.md, "Learning progress"), while the ratio of
    averages is always the share of doing-nothing's error the model still makes over the window."""

    def __init__(self, fast: int = LP_FAST_STEPS, slow: int = LP_SLOW_STEPS):
        self.af = 1.0 / float(max(1, int(fast)))
        self.as_ = 1.0 / float(max(1, int(slow)))
        self.state: Dict[str, Dict[str, torch.Tensor]] = {}
        self.win: Dict[str, List[float]] = {}
        # keys left out after their network was freshly started: key -> model steps still to sit
        # out (set_holdout; LP_HOLDOUT_STEPS)
        self.holdout: Dict[str, int] = {}
        # this step's (model error, persistence error) per key, held-out keys included - for
        # measurement only, never saved
        self.last_errors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def set_holdout(self, keys: Sequence[str], steps: int) -> None:
        """Leave `keys` out for `steps` model steps (a network freshly started at a load). Any
        averages saved for such a key belong to the network that was replaced and are dropped."""
        for k in keys:
            self.state.pop(str(k), None)
            self.holdout[str(k)] = max(int(steps), int(self.holdout.get(str(k), 0)))

    def tick_holdout(self) -> None:
        """A model step begins (called once per step, before that step's updates): a key with
        steps still to sit out sits out this one; a key with none left is counted from this one
        on. set_holdout(keys, N) therefore leaves the keys out of exactly N steps."""
        self.last_errors = {}
        for k in list(self.holdout):
            if self.holdout[k] <= 0:
                del self.holdout[k]
            else:
                self.holdout[k] -= 1

    def update(self, key: str, model_err: torch.Tensor, persist_err: torch.Tensor,
               valid: Optional[torch.Tensor] = None) -> None:
        m = model_err.detach().float().cpu().reshape(-1)
        p = persist_err.detach().float().cpu().reshape(-1)
        self.last_errors[key] = (m, p)
        if key in self.holdout:
            return
        B = m.shape[0]
        ok = torch.isfinite(m) & torch.isfinite(p)
        if valid is not None:
            ok = ok & valid.detach().cpu().reshape(-1).bool()
        s = self.state.get(key)
        if s is None or s["mf"].shape[0] != B:
            z = torch.zeros(B)
            s = {k: z.clone() for k in ("mf", "pf", "wf", "ms", "ps", "ws")}
            self.state[key] = s
        for pre, a in (("f", self.af), ("s", self.as_)):
            s["m" + pre] = torch.where(ok, (1 - a) * s["m" + pre] + a * m, s["m" + pre])
            s["p" + pre] = torch.where(ok, (1 - a) * s["p" + pre] + a * p, s["p" + pre])
            s["w" + pre] = torch.where(ok, (1 - a) * s["w" + pre] + a, s["w" + pre])

    def ratios(self, key: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        s = self.state.get(key)
        if s is None:
            return None, None
        nan = torch.full_like(s["mf"], float("nan"))
        rf = torch.where((s["wf"] > 0) & (s["pf"] > 0), s["mf"] / s["pf"].clamp_min(1e-30), nan)
        rs = torch.where((s["ws"] > 0) & (s["ps"] > 0), s["ms"] / s["ps"].clamp_min(1e-30), nan)
        return rf, rs

    def progress(self, key: str) -> Optional[torch.Tensor]:
        rf, rs = self.ratios(key)
        if rf is None:
            return None
        pr = (rs - rf) / rs.clamp_min(1e-30)
        pr = torch.where(torch.isfinite(pr) & (rs > 0), pr.clamp(-1.0, 1.0),
                         torch.full_like(pr, float("nan")))
        return pr

    def channel(self, B: int) -> torch.Tensor:
        """[B] the learning-progress channel: the mean over senses (and heads) with a value."""
        vals = [self.progress(k) for k in self.state]
        vals = [v for v in vals if v is not None and v.shape[0] == B]
        if not vals:
            return torch.full((B,), float("nan"))
        st = torch.stack(vals)
        fin = torch.isfinite(st)
        n = fin.sum(0)
        tot = torch.where(fin, st, torch.zeros_like(st)).sum(0)
        return torch.where(n > 0, tot / n.clamp_min(1), torch.full((B,), float("nan")))

    def tick_window(self, B: int) -> None:
        """Add this step's channel and per-key values to the report window."""
        ch = self.channel(B)
        for k in list(self.state) + ["__channel__"]:
            v = ch if k == "__channel__" else self.progress(k)
            if v is None or v.shape[0] != B:
                continue
            s = self.win.setdefault(k, [torch.zeros(B), torch.zeros(B)])
            if s[0].shape[0] != B:
                s[0], s[1] = torch.zeros(B), torch.zeros(B)
            fin = torch.isfinite(v)
            s[0] += torch.where(fin, v, torch.zeros_like(v))
            s[1] += fin.float()

    def summary(self, B: int) -> dict:
        def m(key):
            s = self.win.get(key)
            if not s or float(s[1].sum()) <= 0:
                return None
            return round(float(s[0].sum() / s[1].sum()), 5)
        chw = self.win.get("__channel__")
        per_body = None
        if chw is not None and chw[0].shape[0] == B:
            per_body = [None if float(n) <= 0 else round(float(t / n), 5)
                        for t, n in zip(chw[0].tolist(), chw[1].tolist())]
        rfast, rslow = {}, {}
        for k in self.state:
            rf, rs = self.ratios(k)
            if rf is not None and bool(torch.isfinite(rf).any()):
                rfast[k] = round(float(rf[torch.isfinite(rf)].mean()), 5)
            if rs is not None and bool(torch.isfinite(rs).any()):
                rslow[k] = round(float(rs[torch.isfinite(rs)].mean()), 5)
        out = {"value_mean": m("__channel__"), "per_body": per_body,
               "per_sense": {k: m(k) for k in self.state},
               "ratio_fast": rfast or None, "ratio_slow": rslow or None,
               "horizons_steps": {"fast": int(round(1 / self.af)), "slow": int(round(1 / self.as_))},
               # senses and heads still sitting out after a fresh start: model steps left
               "held_out_steps": dict(self.holdout) or None}
        self.win = {}
        return out

    def state_dict(self) -> dict:
        return {"state": {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in self.state.items()},
                "af": self.af, "as": self.as_, "holdout": dict(self.holdout)}

    def load_state_dict(self, sd: dict) -> None:
        st = (sd or {}).get("state") or {}
        self.state = {str(k): {kk: vv.float().cpu() for kk, vv in v.items()}
                      for k, v in st.items() if isinstance(v, dict)
                      and all(torch.is_tensor(vv) for vv in v.values())}
        self.holdout = {str(k): int(v) for k, v in ((sd or {}).get("holdout") or {}).items()
                        if int(v) > 0}
