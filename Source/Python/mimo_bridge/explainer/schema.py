# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Snapshot schema: constants, field documentation, and the validator.

Every block documents itself the way the dashboard does: `t` (title), `w` (what it
is), `when` (when it runs), `h` (how it is computed - naming the exact source),
`f` (what would fool it). The viewer renders these as popovers; [[term]] markers
inside any text become clickable words opening that term's own popover ("term.*"
entries below). The tests hold the shapes.

Schema v2 (2026-09-13) adds five optional blocks - memory, self, hself, voice,
caregiver - as scalar summaries; v1 snapshots without them stay valid.

Schema v3 (2026-09-22, the gain-of-function build) adds optional blocks - causal_chain,
efference, vestibular, wellness_channels, contact_control, synergy - and motor.order_sent /
motor.rest_window. Every one is read from an attribute the model already caches during its
step (capture.py never runs an extra forward pass and never calls a *_window_summary(), which
would reset the stats window). Each is absent, or a labelled absence, when its launch flag is
off; v1 and v2 snapshots stay valid and the viewer draws what they carry.
"""
from __future__ import annotations

import base64
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 3
ACCEPTED_VERSIONS = (1, 2, 3)

# blocks that must exist in every snapshot (attention/senses may be absent only with
# an explicit reason string in their place - e.g. stale vision, first step)
REQUIRED_BLOCKS = ("header", "model", "tokens", "predictions", "attention",
                   "motor", "drives", "intention", "losses", "senses")

REQUIRED_HEADER = ("schema_version", "run", "step", "timestamp", "phase",
                   "source_mask", "n_envs", "body", "weights")

# Schema v3's blocks (the gain-of-function build). A v3 snapshot carries every one, as a dict or
# as an "absent: <reason>" string; a v1 or v2 snapshot carries none and still validates.
V3_BLOCKS = ("causal_chain", "efference", "vestibular", "wellness_channels",
             "contact_control", "synergy")

# what `weights` may say - see the capture contract in __init__.py
WEIGHTS_LABELS = ("post_update", "frozen")

# The run DOC's "this run", "this probe" and "the audited run" sentences were written
# about. DOC is written once and shipped on every page, whichever run the page is built
# for, so the viewer prints this in every popup that uses one of those phrases.
# THE WELLNESS TERM tactile_comfort (lem/model/wellness.py _formula_tactile_comfort), in one line from
# lem/model/contact_wellness.py contact_response: comfort = (0.5 + 0.35 x pleasant) x (1 - pain), the
# pleasant part a hill in pressure that peaks at TACTILE_PLEASANT_PEAK (0.05 of 100 kPa) and rises with
# texture, the pain part the worst local discomfort. Shared by the explainer and (held equal by a test)
# the development dashboard's glossary; test_sense_names.py holds the numbers to the code
TACTILE_COMFORT_DEFINITION = ("the fixed skin-comfort score: 50% without contact or remaining load history, "
                              "up to 85% from area-weighted pleasant contact near 5 kPa, reduced by "
                              "the largest skin-patch aversion")

DOC_WRITTEN_FOR = {
    "run": "H81_decorated_shadow_explainprobe_20260913d",
    "what": "a 400-step --eval-only probe of the H81 weights: 2 bodies, 20 snapshots; "
            "values quoted over a whole training run are H82_dephased_babble_20260914's",
}

DOC: Dict[str, Dict[str, str]] = {
    # ================= glossary: terms any popup may link to =================
    "term.token": {
        "t": "token",
        "w": "One fixed-length list of numbers describing one sense at one "
             "[[model-step]]. In this model a token is {d_model} numbers. Every sense, "
             "however big its raw input, is squeezed to exactly this size before "
             "the [[transformer]] reads it.",
    },
    "term.model-step": {
        "t": "model step",
        "w": "One update of the model's loop: one set of sense inputs read, one command sent to the "
             "body (the code and the stats files call it `step`). The physics simulation advances "
             "{action_repeat} physics steps of {env_step_ms} ms per model step in this run "
             "(the --action-repeat launch flag sets the {action_repeat}; the "
             "{env_step_ms} ms is {env_step_source}), so one model step is "
             "{moment_ms} ms of body time; in the crib scene each 10 ms env "
             "step is itself 2 MuJoCo integration steps of 5 ms. Every "
             "model step, each sense contributes one [[token]] to the sequence.",
    },
    "term.hself-measurements": {
        "t": "hierarchical self: the measurements behind the warnings",
        "w": "- **A table that is about old model steps.** A pattern is stored once. If the "
             "encoders change afterwards, today's model steps stop matching the stored "
             "patterns, and every score then describes leftovers. Two live rows in the "
             "inspector panel show which case a run is in: the share of model steps that belong "
             "to a slot, and how alike a model step and its nearest slot are. On run H84 on "
             "2026-09-19 every slot had been opened in the first 4,268 model steps, before the "
             "encoders had trained. Against 4,800 of the run's own recorded model steps none "
             "belonged to a slot: 0 of 4,800. The nearest slot's pattern was nowhere near the "
             "match rule's line on any of them (how alike they were is a [[cosine]]: not yet "
             "measured as a count).\n"
             "- **A match rule that is strict for a wriggling body.** On run H84 two model steps "
             "of one body, one step apart, matched on every sense in 70 of 100 pairs. A "
             "hundred steps apart they matched in under 1 of 100, and the motor sense "
             "differed most. So a table opened on today's model steps fills up again within "
             "minutes of body time.\n"
             "- **The slots-in-use count is not a count of kinds of model step.** Two slots can "
             "store the same pattern. Until 2026-09-19 every model step pulled on every slot, "
             "and slots could be dragged onto each other: in a test with one common and one "
             "rare type of model step, 9 of 11 slots ended on one pattern and the rare type "
             "kept no window of its own. Now a model step changes only the slot it belongs to, "
             "and the same test ends with 2 slots, no copies, and the rare slot keeps every "
             "window it lived. The inspector panel counts copies live.\n"
             "- **The average link size on its own.** Fake signals score almost as high as "
             "the real one. On run H81 the 8 controls' average link reached 89% of the real "
             "signal's at the minutes level, and at the hours level the controls' was 131% of "
             "the real one's - LARGER. That is why a slot is judged against its own controls "
             "and never by link size.\n"
             "- **A 0 that was forced.** Before enough windows have passed, no slot could "
             "possibly count. On run H83 the hours level had lived 3 windows and printed 0 "
             "slots, while the best slot's standing against its 8 controls stood at 62.5 in "
             "100 against the 63.6 in 100 that 3 windows allow at most. The standing is a "
             "running average of how many of its controls the slot beat in each window, out "
             "of every 100; it starts at 50 and closes a tenth of the gap to 100 in a window "
             "where the slot beats every control, so even a slot that beat all 8 in all 3 "
             "windows reads 63.6, not 100. "
             "That 0 could not have been anything else. The page now prints -1 with the "
             "model steps still to go.\n"
             "- **Counts by luck.** The bar is set so that a table with no real link shows "
             "a counted slot 135 times in 100,000. Measured on test tables where the model step's "
             "slot changes every model step: 0 of 384 slots, and 0 of 12,800 on a 6,400-slot "
             "table; a planted link was still counted in 6 of 6 tries. Measured where "
             "presence itself drifts slowly (2 slots, about 4 windows per drift): 2 of 64 "
             "tables with no link showed a counted slot, 3.1 in 100, which is 23 times the "
             "setting. So under slow drift a counted slot is luck more often than the "
             "setting says.\n"
             "- **Two bars are shown and one is used.** The table's own middle standing "
             "and spread are printed beside the bar (rank_mean, rank_sd, line_empirical). "
             "They are not used as the bar: tried on a test with a planted link, that bar "
             "counted nothing in 6 of 6 tries, because a real link lifts the middle.\n"
             "- **The 'slots live' row.** It counts every slot holding more than a tenth "
             "of an even share of presence (an even share on 6,400 slots is 1 part in 6,400, "
             "so the line is 1 part in 64,000). On run H81 the "
             "quietest slot cleared that by 1.5 in 100, so the row read 6,400 of 6,400, its "
             "ceiling, and said nothing. Read the margin row beside it.\n"
             "- **'Every model step looks new.'** On run H83 all 6,400 slots were opened inside "
             "4,360 model steps (218 s of body time), the table had then been full for 484,088 "
             "model steps, and the rule called 100 of every 100 checked model steps new (968,230 "
             "of 968,176 counted). So the count of model steps refused for lack of a free slot "
             "cannot say how big the table should be.\n"
             "- **Changing the number of slots at a restart.** A bigger table keeps every "
             "slot. A SMALLER table drops the slots above the new number, evidence and "
             "all; the log line 'hself: migrated' says how many filled slots were dropped "
             "and how many of them had counted as evidence at each level. A changed set of "
             "senses starts the table over. Check which line the log shows.",
    },
    "term.entity": {
        "t": "entity (a named thing)",
        "w": "One named thing the self-model watches: the caregiver, a toy, the mobile, the "
             "squeaker, a critter, or the baby's own hand in view. Each has its own "
             "[[entity-presence|presence]] signal, so it can be pointed at - unlike a "
             "[[slot]], which stores a whole model step. mimo_bridge/entity_presence.py fixes "
             "the list and the rule for each one.",
    },
    "term.entity-presence": {
        "t": "presence of a named thing",
        "w": "How available a named thing is to the baby right now, from 0 (not there) to 1 "
             "(there, and squarely in view or in hand). It is read from the world's own "
             "report, not from the model, so it does not change when the [[encoder|encoders]] "
             "do. A missing reading is 0, never a guess. This is availability, not value: a "
             "thing can be present and unhelpful.",
    },
    "term.self-value-reader": {
        "t": "the reader (self value)",
        "w": "The part that turns a PROVEN link into something the body acts to keep. For each "
             "remembered model step it adds weight x the sum over proven things of (link x "
             "presence then), so the planner aims at model steps that held those things. It adds "
             "exactly 0 until its gate opens: something must have passed the evidence test, "
             "and the real reader must track [[wellness]] better than a shuffled copy of "
             "itself for a run of windows. It changes which goal is ranked first and nothing else: no loss reads "
             "it and it enters no lesson. "
             "mimo_bridge/self_value_reader.py.",
    },
    "term.slot": {
        "t": "slot (hierarchical self)",
        "w": "One line of the [[hierarchical-self|hierarchical self]]'s table. A slot stores "
             "one pattern - what all the senses reported at one [[model-step]], as one long list "
             "of numbers - and the running scores kept for that pattern. A slot is opened "
             "when a model step matches no stored pattern and a slot is free. The code calls the "
             "stored pattern the slot's centre.",
    },
    "term.hierarchical-self": {
        "t": "hierarchical self",
        "w": "The part that keeps score of which kinds of model step go together with "
             "[[wellness]] rising or falling over minutes and over hours. It has no trained "
             "weights and changes nothing the model does: it only watches. Its box on the "
             "model diagram is 'hierarchical self (dark)'.",
    },
    "term.presence": {
        "t": "presence (of a slot)",
        "w": "How much of the current [[model-step]] one [[slot]] holds: a share between 0 and "
             "1. All the slots' shares add up to 1. A slot whose stored pattern is closer to "
             "the model step gets a bigger share. With thousands of slots every share is small.",
    },
    "term.wellness-link": {
        "t": "wellness link (the code calls it membership)",
        "w": "A score from -1 to +1 for one [[slot]] at one level: how closely the change in "
             "the slot's [[presence]] moves together with the change in [[wellness]] inside "
             "a [[evidence-window|window]]. +1 = they always move together, 0 = no link, -1 = "
             "they always move opposite ways. A link COUNTS as evidence only after it has "
             "kept beating its [[sign-flipped-control|controls]].",
    },
    "term.evidence-window": {
        "t": "window (hierarchical self)",
        "w": "A stretch of model steps the [[hierarchical-self|hierarchical self]] averages "
             "over: 10,000 [[model-step|model steps]] at the minutes level and 100,000 at the hours "
             "level. Once per window every slot's link is compared with its controls; each "
             "such comparison is one window lived by that slot.",
    },
    "term.sign-flipped-control": {
        "t": "control (sign-flipped wellness)",
        "w": "A fake [[wellness]] signal used to measure luck. It is the real signal with "
             "its sign flipped at random times - once per [[evidence-window|window]], on a "
             "fixed schedule. Flipping destroys any true link but keeps the signal's size "
             "and slowness, so whatever link a [[slot]] shows with a control is luck. Each "
             "level keeps 8 controls.",
    },
    "term.cosine": {
        "t": "cosine",
        "w": "A score of how alike two lists of numbers are, judged by their direction and "
             "not their size. 1 = the same pattern, 0 = nothing in common, -1 = exact "
             "opposites.",
    },
    "term.transformer": {
        "t": "transformer",
        "w": "A sequence model built on [[attention]]: it reads a lineup of "
             "tokens and lets each token gather information from the others, "
             "layer by layer, before anything is predicted from the result.",
    },
    "term.attention": {
        "t": "attention",
        "w": "The core operation of a transformer. Each token computes a "
             "relevance weight for every other token (weights are 0 to 1 and "
             "sum to 1 per token - a fixed budget) and takes the weighted mix. "
             "The heatmaps on this page are those weights.",
    },
    "term.encoder": {
        "t": "encoder",
        "w": "A small network that turns one sense's raw input (pixels, "
             "contacts, joint states, a sound spectrum) into a [[token]].",
    },
    "term.layernorm": {
        "t": "LayerNorm",
        "w": "A normalization layer: it subtracts a token's own average and "
             "divides by the token's own spread, so what comes out is centred "
             "and has a standard overall size. This model has two kinds. The "
             "shared one in front of the core has NO learned gain or bias, so "
             "it holds every sense's token at the same size - [[rms|RMS]] 1, "
             "measured within 1 part in 100,000 of 1 over this run's six senses and 20 "
             "snapshots, a small constant in the divisor keeping it off "
             "exactly 1 - and no sense can shout over the others. Centring is "
             "the other half of the same job: a token's average measured "
             "under 1 part in a million of the token's size here, so no sense arrives with a standing offset "
             "either. The core's {core_layernorms} each have a learned gain "
             "and bias ({core_layernorm_numbers} numbers in all), so they can rescale and shift "
             "what they produce and they hold no token at a fixed size.",
    },
    "term.mlp": {
        "t": "MLP",
        "w": "A multi-layer perceptron: stacked fully-connected layers with a "
             "bend (nonlinearity) between them. The plainest neural network.",
    },
    "term.gru": {
        "t": "GRU (recurrent network)",
        "w": "A gated recurrent unit: a network that reads a sequence one item "
             "at a time, carrying a running summary vector forward as it goes.",
    },
    "term.loss": {
        "t": "loss",
        "w": "A number measuring how wrong the model was at one job. Training "
             "means adjusting weights to shrink losses. Each loss term is one "
             "lesson; a term at zero taught nothing. The STEP's loss is the "
             "plain sum of those terms, and they are not on a common scale: "
             "one of them - self perception's forecast - is a likelihood "
             "rather than an error, and it goes negative whenever that "
             "forecast is confident, so the total can sit below zero (this "
             "run's 20 snapshots run from below zero to tens of thousands, "
             "almost all of the top of that range the motor lesson). A loss "
             "has no lived unit: this page draws each lesson as its share of "
             "the step's total, never as its raw value.",
    },
    "term.gradient": {
        "t": "gradient",
        "w": "The derivatives of one number with respect to each of the numbers it is computed from, "
             "found by backpropagation: for each input, how much the output changes per unit change "
             "of that input. Training follows the gradient of the loss with respect to the weights "
             "downhill. The planner follows the gradient of its [[intention-distance|score]] with "
             "respect to the COMMAND's numbers, through the frozen [[forward-model|forward model]], "
             "and changes the command, never the weights. [[wellness]] sets which goal-ring entry is "
             "the goal; it is not in the score unless --prospective-wellness add puts a "
             "predicted-wellness part into it. A separate wellness gradient - the derivatives of the "
             "six fixed wellness formulas' sum with respect to the body's measured handle state (its "
             "first 9 numbers per handle, positions taken relative to the body's centre) - is handed "
             "to the motor controller as an extra input.",
    },
    "term.adam": {
        "t": "Adam",
        "w": "A [[gradient]]-following optimizer. For each number it changes it keeps two running "
             "averages - of the derivative and of the squared derivative - and changes the number by "
             "step size x (average derivative) / (square root of the average squared derivative + a "
             "small constant), against the derivative's sign. On the first update every number with a "
             "nonzero derivative moves by the step size, whatever its derivative's scale; later updates scale with how "
             "steady its derivative has been. The planner runs it with step size one tenth on the "
             "command's numbers, 20 updates per planning event, from fresh averages each time; "
             "training runs AdamW (Adam with weight decay) on the model's weights.",
    },
    "term.quaternion": {
        "t": "quaternion",
        "w": "Four numbers describing a 3-D orientation, used instead of three "
             "angles because angle triples have blind spots (gimbal lock). In "
             "this code a quaternion is always written x, y, z, w; the physics "
             "engine's own sequence is w first, and the body reader flips it on "
             "the way in.",
    },
    "term.rms": {
        "t": "RMS",
        "w": "Root mean square: one number for 'how big overall' a list of "
             "numbers is. 0 = all zeros.",
    },
    "term.mse": {
        "t": "Mean squared error: why square and average?",
        "w": "Take two equal-length lists of numbers. Subtract each matching pair, square each difference, add the squares and divide by the number of pairs. For errors +2 and -2, the ordinary average is zero, but the mean squared error is 4: both errors still count. Squaring also makes an error twice as large count four times as much. Averaging prevents a longer list alone from creating a larger total.\n\nZero means the compared lists match exactly. The units are squared units of the compared numbers. Learned sensory coordinates have no direct unit such as cm or kPa.",
        "h": "lem/model/wellness.py: _formula_prediction_error uses squared sensory-coordinate differences. lem/training/loss_manager.py: _compute_wellness_loss instead compares predicted and measured wellness scores.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "Always identify the lists and averaging rule. The planner adds one MSE per compared sense in its search objective; its diagnostic distance averages sense errors instead. These numbers are not interchangeable."
    },
    "term.intention-distance": {
        "t": "the planner's score",
        "w": "The one number per body that the planner's 20 updates lower and its keep-the-lowest "
             "step compares, computed for each candidate command from the forward model's "
             "[[predicted-outcome|predicted tokens]]:\n\n"
             "- the goal part: for every sense measured at that model step except motor (a stale or "
             "synthetic sense is left out), the mean over the token's {d_model} numbers of (predicted "
             "- goal)^2, added over those senses. Tokens are held at size 1, so two unrelated tokens "
             "differ by about 2 per sense;\n"
             "- the strength part: one hundredth of the sum over the {n_handles} handles of strength^2, "
             "on the command's strength before the execution filter's limit - from 0 (no strength) to "
             "{n_handles} hundredths (every handle at full strength). With --planner-effort-per-body "
             "off it is summed over all {n_bodies} bodies while the goal part is averaged over them, so "
             "it counts {n_bodies} times over; with it on it is averaged too. {effort_run}\n\n"
             "{score_extras_sentence} The planner follows the score's [[gradient]] with respect to the "
             "command. Measured on H83 at 8 bodies with --planner-effort-per-body off, the strength "
             "part made up 56 of every 100 parts of the score and the goal part 44.",
        "h": "lem/model/intention_generator.py generate_action: goal = the sum over the compared "
             "senses of ((predicted - desired) ** 2).reshape(B, -1).mean(-1); effort = strength_reg x "
             "the sum of strength squared per body, times B unless effort_per_body; total = goal + "
             "effort + the synergy penalty + the prospective-wellness part; the Adam step lowers "
             "total.mean() over the bodies, and the kept command is each body's lowest total",
        "f": "- reading a lower score as predicted tokens nearer the goal tokens: the strength part can "
             "fall while the goal part does not, and the planner panel's live values give both parts "
             "for the newest planning event.\n"
             "- reading it as the plan record's distance: that one is this model step's tokens against "
             "the goal tokens, with no strength part.",
    },
    "term.causal_pe": {
        "t": "Delayed sensory prediction error used by curiosity",
        "w": "The model re-encodes the previous raw sensor readings and the command actually executed, using its current weights. It predicts the current sensory-number lists and compares them with the current raw encoder outputs. For each eligible sense, average the squared coordinate differences across bodies and coordinates. Then average the available sense errors equally. [[mse|Why square and average]].\n\nThe delayed lesson records errors for live-measured senses; vestibular additionally requires a live reading in the previous frame. Curiosity averages available recorded errors whose current sense is not marked synthetic. Vestibular can therefore count here while being excluded from the fixed prediction-accuracy wellness score.\n\nThe chart multiplies this mean error by 100. The sensor lists are normalized toward unit root-mean-square, making an all-zero prediction a reference near 100%; this display is not a separately measured zero-prediction baseline. Zero error means an exact match before display rounding.",
        "h": "mimo_bridge/lem_model.py: nf_terms and last_next_pe record unweighted per-sense F.mse_loss; last_causal_pe averages the available records. The current-weight prediction is recomputed, not retrieved as a forecast saved at the previous model step.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "With no eligible delayed-error record, causal_pe is unavailable and curiosity drive is zero. The curiosity band chooses how this error maps to drive; the band does not establish learnability. [[drives-pairing|How the fixed wellness comparison and its fallback differ]]."
    },
    "term.handle": {
        "t": "handle",
        "w": "One of the body's named control points: both wrists and all "
             "ten fingers, the head, upper and lower arms, thighs, calves, "
             "feet, the eyes as the gaze handle - and, on the trunk body "
             "layout, the lower trunk and chest. The run's exact list and "
             "sequence is handle_names in its model receipt; the motor boxes on "
             "the architecture tab read their counts from it. A command gives "
             "each handle 9 numbers: a target position (3, cm) that only the "
             "twelve hand handles - both wrists and all ten finger tips - "
             "actually drive, through a small inverse-kinematics step; a "
             "target orientation ([[quaternion]], 4); a radius, which nothing "
             "reads; and a strength 0-1 that scales how hard the joints push "
             "toward the target (a [[pd-controller]] per joint). For the head, "
             "arms, legs, feet and trunk the position numbers are computed and "
             "scored by the posture evaluator, then dropped by the body, which "
             "turns those handles by orientation alone. The gaze handle is the "
             "further exception: its position, radius and strength are all "
             "ignored, and only its orientation is used, read out as one "
             "azimuth-and-elevation pair and driven by {eye_servo_clause} "
             "{servo_timing} Added up, 63 of a command's 234 "
             "numbers - 26.9% - reach no actuator at all: every handle's "
             "radius (26 numbers), the position triple on the 12 handles "
             "whose position is not used (36: the arm and leg segments, the "
             "chest - whose two hinges are anchored exactly at its own "
             "origin, so no joint could close a position error - and gaze), "
             "and the gaze handle's strength (1). Since 2026-09-18 the head "
             "and the lower trunk DO use their position numbers: their hinges "
             "have a lever from anchor to body origin, so 'head up 5 cm' "
             "becomes neck extension. "
             "They are still scaled and handed back to the model as the cause "
             "of the next frame, so the lesson that measures what an action "
             "did is conditioned on them.",
        "h": "mimo_bridge/action_decoder.py: POSITION_HANDLES names the fourteen "
             "handles whose position is used (the twelve of the hands, the "
             "head and the lower trunk), HANDLE_CHAINS['gaze'] is empty "
             "on purpose, and the eye servo runs on --gaze-kp / --gaze-kd, the "
             "limb servo on --decoder-kp (schema.servo_fill reads all three "
             "from the run's args.json)",
    },
    "term.pd-controller": {
        "t": "PD controller",
        "w": "A simple mechanical controller: push a joint toward its target "
             "(proportional) while resisting fast motion (damping). When a "
             "command asks for more torque than the actuators can deliver, the "
             "whole handle's set of joint pushes is scaled down together "
             "rather than each joint being cut at its own limit - so a strong "
             "command arrives weaker but keeps the ratios between its joints, "
             "which are the part that says what movement it is. The fingers' own joint inertia is set apart from the limbs' ([[finger-armature|why the fingers could not move until 2026-09-22]]).",
        "h": "mimo_bridge/action_decoder.py: the per-joint push, then a "
             "direction-preserving rescale per handle chain and arm group, "
             "then a clip as the final guard",
    },
    "term.executed-command": {
        "t": "executed command",
        "w": "The command the body was actually GIVEN (the code calls it `executed_action`; the snapshot's "
             "fields say `order`): the held command after "
             "growth-stage limits, the reach clamp, [[babble]] noise - added "
             "only on a model step the planner did not author, which is "
             "{babble_latch_share} - and a "
             "smoothing blend of 35% of the new command with 65% of the previous one ([[smooth-source|which previous one]] is --smooth-source), with the eye command the world really used written "
             "into the gaze [[handle]]'s slot - averaged over the interval's "
             "sub-steps (--action-repeat, {action_repeat} in {run_name}), because the "
             "eyes are not held and a saccade or the motion reflex can move "
             "them on any sub-step, so the stored eye command may be a value "
             "never issued on its own. It is the command written into the motor token the next frame is "
             "predicted from - never the controller's raw proposal. It is "
             "the command ISSUED: not the pose the body reached, and not the "
             "joint-level actuator command, which the trainer's ledger "
             "carries but the replay ring never stores.",
        "when": "Every [[model-step]]. In a scheduled rest window (--rest-control "
                "scheduled with --babble-bouts real; {scheduled_rest_note}) "
                "the command sent to the body, and therefore the "
                "command recorded, is a postural hold built from the pose the "
                "body is in right now - not the held command. Only the strength "
                "survives from the held command there; the position and the "
                "orientation are rewritten with the body's own achieved pose, "
                "every single step. The H81 probe, launched that way, spent 25% "
                "to 42% of its steps in rest windows.",
        "f": "reading command-following, the motor [[token]] or a replay "
             "window across rest and bout steps as one set of numbers. Inside "
             "a rest window the command IS the pose, so obeying it means "
             "nothing: this run measured the match between the direction "
             "commanded and the direction moved as close to 'exactly against "
             "it' in rest windows and close to 'sideways to it, unrelated' in "
             "bouts (the match is an averaged [[cosine]]: not yet measured as "
             "a count of commands)",
    },
    "term.babble": {
        "t": "babble noise",
        "w": "Random motor noise added to commands - an infant's wiggling. "
             "It bootstraps learning: without self-caused movement there is "
             "nothing from which to learn what actions do. It is added ONLY on "
             "reflex steps - steps where the planner produced no command - so "
             "when --motor-hold equals --intention-every a "
             "command the body receives carries babble only when the planner "
             "produced nothing (the goal ring still filling, an empty plan, or "
             "the scale gate refusing a rollout): on the H81 probe, launched "
             "with both flags at 5, the code's own tally counted 181 of 181 "
             "executed commands written by the planner, so none of them carried "
             "noise. {planner_latch_rule} {exec_source_tally} --babble-bouts also "
             "switches it off entirely during rest windows - 25% to 42% of "
             "the H81 probe's steps. The numbers it adds are not small. At this "
             "run's --babble-noise 0.05 the position offset is 5.00 cm per axis "
             "(standard deviation), 3.99 cm on average; strength gains 4% of "
             "full strength on average; and the target orientation is turned 46 degrees on "
             "average - a bigger turn than the 28.6-degree shoulder command "
             "the joint controller's gains were tuned against.",
    },
    "term.motor-hold": {
        "t": "motor hold",
        "w": "The trainer latches a new command for the body on every model step whose number is a "
             "multiple of --motor-hold ({motor_hold} in {run_name}) and sends that same command on every "
             "model step until the next latch: {motor_hold} model steps, {hold_ms} ms of body time, "
             "about the quarter second a limb needs to respond. A plan or a motor controller command "
             "made on a model step that is not a latch step never reaches the body. Inside a "
             "scheduled rest window the held command's target position and orientation are rewritten "
             "on every model step with the body's current pose, so what is held there is the "
             "strength, not the target.",
    },
    "term.wellness": {
        "t": "Wellness: measured scores, predicted scores and goal ranking",
        "w": "Wellness is a set of numbers used to evaluate sensor predictions and bodily conditions. It is not one measurement of health or ability. This implementation has three separate calculations:\n\n- **Measured scores:** six fixed formulas calculate [[eval-prediction-error|encoded prediction accuracy]], [[eval-tactile-comfort|tactile comfort]], [[eval-capability|remaining muscle capacity]], [[eval-coherence|body-map co-movement]], [[eval-motor-state|handle-position boundary score]] and [[eval-self-touch|self-contact increase]].\n- **Predicted scores:** a [[approximate-evaluator|learned wellness predictor]] estimates the first five scores from encoded sensory and motor numbers. It has no self-touch output.\n- **Goal ranking:** a separate calculation combines enabled [[wellness-ranking|wellness channels]] to select one remembered sensory state from the goal ring. {wellness_rank_run}\n\nThe measured scores also supply training targets for the learned predictor. The motor controller receives the [[wellness-gradient|derivative of the measured sum with respect to handle positions]].",
        "h": "lem/model/wellness.py: WellnessEvaluators.forward and APPROXIMATE_TERMS. mimo_bridge/lem_model.py: _self_touch_wellness, _wellness_channel_vector, _normalized_goal_values and the total_wellness gradient in step.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "A score has meaning only with its formula: tactile comfort has a maximum of 85%, while self-touch has a configurable maximum. Matching a remembered sensory state does not guarantee the same wellness later: capacity, contact history and prediction-error history can differ. [[wellness-spec|How this implementation compares with UEMA 3.5.]]"
    },
    "term.eval-self-touch": {
        "t": "Self-touch: an increase in the largest own-part contact count",
        "w": "For each valid skin patch, the tactile input gives the number of the body's own parts touching it, divided by four and capped at one. Keep the largest patch value. Subtract the previous model step's largest value and discard negative differences. Multiply the remaining increase by the self-touch weight and by its evidence gate. {self_touch_weight}\n\nA held contact with an unchanged count adds zero after its first increase. The maximum contribution is the configured weight. This is a change in the largest count: it does not count every new contact anywhere on the body.",
        "h": "mimo_bridge/lem_model.py: _self_touch_wellness reads tactile column 12 and TOUCH_VALID_COL, calculates max, positive difference, weight and _self_touch_gate.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The gate uses the last 32 controlled self-touch readings: mean advantage must exceed twice its standard error, and the newest reading must be no more than 2,000 model steps old. This older term is separate from [[contact-control|predicted-contact scoring]]. It has no learned-predictor output and no direct channel in the normalized-ranking channel list."
    },
    "term.eval-prediction-error": {
        "t": "Prediction accuracy: comparing encoded sensory numbers",
        "w": "**What is compared?** Each sense encoder converts its sensor readings into a list of {d_model} learned numbers, called a [[token]]. For example, the vision list is calculated from image pixels; the proprioception list is calculated from handle orientations, positions, velocities and capacities. The prediction heads produce a list of the same size for each predicted sense. These lists contain learned features; an individual number has no fixed meaning such as a finger angle or a pixel brightness.\n\n**Calculation.** For each body, subtract each measured number from the matching predicted number. Square every difference, average within each sense, then average the eligible senses equally. The wellness score is 1 minus that average, clipped between 0 and 1.\n\n**Why square and average?** Squaring prevents positive and negative errors from cancelling and gives larger errors more influence. Averaging makes the result an error per number, rather than a total that grows with the list length. Averaging the sense results gives each included sense equal weight.\n\n**Reading the bar.** An exact score of one means every compared number matched. An exact zero means the mean squared error reached one or more; larger errors all receive zero. Displayed percentages are rounded, so a displayed 100% need not be an exact match. A moving body can achieve an exact match if its encoded readings are predicted exactly. This is encoded prediction accuracy, not a measurement of stillness or physical health.",
        "h": "lem/model/wellness.py: WellnessEvaluators._formula_prediction_error. The source calculates (pred_stack - tgt_stack).pow(2).mean(dim=-1).mean(dim=0), then 1 minus the clipped result. See [[drives-pairing|which model steps are compared]] and [[mse|mean squared error]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "Only matching predicted/target senses marked real are included, after the configured exclusions. Vestibular is excluded by default from this wellness score even when it is measured. Motor has no next-sense prediction head. With no eligible pair the code returns 100% as a fallback; that value is not evidence of prediction accuracy. The training loss can use different sense weights and scaling."
    },
    "term.contact-concentration": {
        "t": "why one patch reads a huge pressure",
        "w": "A tactile number is a force divided by an area, and the two halves "
             "come from different places. The force is what MuJoCo's contact "
             "solver reports. The area is NOT the footprint the body is lying "
             "on: it is the number of MIMo skin points that felt anything, times "
             "the size of ONE point. Resting, the head sums 40.3 N over 14 points "
             "of 169 mm2 each, so 2,366 mm2, giving 17,036 Pa. A fingertip point "
             "is 4 mm2 - 42 times smaller - so the same body landing on a "
             "fingertip divides 266 N by 72 mm2 and reads 3,701,230 Pa, 217 times "
             "the resting number. This run's body carries 4,683 points across 53 "
             "parts, and one point's size runs from 4 to 1,444 mm2.",
        "h": "mimo_bridge/touch_adapter.py TouchPooler.update: pressure = "
             "normal_sum / (n_active_points * point_area). Measured by "
             "mimo_bridge/tools/contact_concentration.py, which splits every "
             "peak into its force and its divisor",
        "f": "reading peak patch pressure as how hard a landing was. It does "
             "not measure that, and it does not merely measure it weakly - it "
             "points the WRONG WAY. Across 30 measured drop peaks the "
             "correlation between impact force and this number is BELOW ZERO "
             "(a correlation: not yet measured as a count): "
             "the harder the landing, the lower it tends to read. Splitting the "
             "spread exactly (pressure is force divided by area, so the two "
             "shares must sum to 100 percent), which part happened to touch "
             "accounts for 120.5 percent of it and how hard the landing was for "
             "MINUS 20.5 percent. The worst single reading in the whole set, "
             "3,730,075 Pa, was a fingertip carrying 254 N. The largest force in "
             "the whole set, 3,221 N on the head, read 1,361,526 Pa - less than "
             "half as much. A 1,861 N hip landing read 99,275 Pa. The 361-fold "
             "spread in sensor point size that causes this is MIMo's own "
             "deliberate spacing, not a defect, but it means this number cannot "
             "be used as impact severity in any form",
    },

    "term.contact-profile": {
        "t": "Contact profile: turning skin measurements into aversion",
        "w": "{contact_profile_run} The profile selects both the pressure measurement path and the aversion formula. In mechanical_v4, pressure uses the resolved contact footprint. Each patch's aversion uses three loads: peak pressure, sustained pressure exposure and rubbing work. Add their scaled contributions to get L, then calculate L/(1+L). [[mechanical-load|Exact inputs, scales and history]].\n\nLegacy instead uses a pressure-only ramp. Earlier mechanical profiles use an exponential conversion that approaches one faster. The v2_local path divides force by active sensor area; v3 and v4 use the resolved contact footprint. These differences mean scores from different profiles should not be treated as the same measurement.",
        "h": "lem/skin_mechanics.py: MechanicalProfile, mechanical_load and aversion_from_load. mimo_bridge/virtual_skin.py: advance_support and advance. env.py records mechanical_contact.area_source.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "These are engineering formulas. Their numerical references do not make the output a calibrated human pain rating. [[eval-tactile-comfort|How aversion reduces comfort]], [[contact-concentration|contact-area effects]], [[fall-response|what fall comparisons measure]]."
    },
    "term.touch-pressure-scale": {
        "t": "Pressure units: 1 stored unit means 100 kPa",
        "w": "The tactile pressure number is pressure in pascals divided by 100,000. Thus 1 means 100 kPa, and 2.194 means 219.4 kPa. One is a storage reference, not a ceiling; the number can exceed one.\n\nPressure is force divided by contact area. The same force on a smaller area produces greater pressure. Neither the stored number nor a pressure value by itself is a wellness score: the selected contact profile converts pressure and its history into skin aversion.",
        "h": "lem/model/contact_wellness.py: contact_response multiplies stored pressure by 100,000 to recover pascals. lem/skin_mechanics.py: pressure_reference_pa. [[contact-profile|How aversion is calculated]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },

    "term.fall-response": {
        "t": "What the fall comparison checks",
        "w": "The fall test drops the simulated body from several heights and records contact pressure, skin aversion and tactile comfort. It compares pairs of falls to ask how often the higher fall produces a worse score. Trials starting inside furniture are excluded.\n\nA higher fall can land on a different body part and contact area. Local pressure therefore need not increase as drop height increases. This test checks how well a particular score separates those impacts; the score itself does not measure fall height.",
        "h": "mimo_bridge/tools/fall_response.py and impact_grading.py; historical measurements in Saved/mimo_bridge/scale_anchors.json and scale_anchors_v3.json. [[contact-profile|Name the profile]] when comparing results.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "Tactile comfort ranges from zero to 85% under its formula, not 100%. Whole-body impact force and local skin pressure measure different things; a force measurement is not automatically an input to wellness."
    },
    "term.eval-tactile-comfort": {
        "t": "Tactile comfort: pleasant contact reduced by skin load",
        "w": "Tactile comfort is {tactile_comfort_definition}.\n\n**Inputs:** each valid skin patch supplies pressure, contact area and four vibration values. The mechanical contact profiles also use contact-load history and regional sensitivity. These are numerical simulator measurements, not predicted sensory tokens.\n\n**Calculation:** calculate a pleasant-contact value for each contacting patch, then average those values using contact area as the weight. Separately, calculate a load-aversion value per patch and keep the largest. Comfort = (one half + 35% times pleasant-contact average) times (1 minus largest aversion).\n\n**Scale:** no contact or remaining load history gives 50%. The highest possible result is 85%. Maximum aversion gives zero, even if other patches have pleasant contact. A small patch with high aversion therefore cannot be hidden by many comfortable patches.",
        "h": "lem/model/contact_wellness.py: contact_response. The pressure preference is a bell curve over log pressure, centred at 5 kPa with width six tenths of a base-ten logarithmic unit. Vibration scales that curve from half to full strength; regional gains and clipping are applied before the area-weighted average. Each active patch missing an area gets fallback weight 1; other patches keep their measured area weights. missing_area counts these fallbacks.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "{contact_profile_run} [[mechanical-load|The mechanical_v4 calculation]] adds scaled peak pressure, sustained exposure and rubbing work, then converts their total L to L/(1+L). Stored loads decay, so aversion can remain after contact ends. Other [[contact-profile|profiles]] differ. This formula does not itself generate a withdrawal command. [[contact-concentration|Contact-area effects]]; [[touch-pressure-scale|pressure units]]."
    },
    "term.eval-capability": {
        "t": "Capacity: remaining strength in the ten muscle groups",
        "w": "The simulator maintains one capacity number for each of ten [[muscle-groups|muscle groups]]. Each number is limited to 0 through 1: 1 means full configured capacity; 0 means depleted capacity. This wellness score is their arithmetic mean: add the ten capacities and divide by ten. Every group has equal weight.\n\nEffort reduces these simulated capacities and recovery increases them. The score makes depleted capacity reduce the measured wellness sum and supplies the capability channel used in goal ranking.",
        "h": "lem/model/wellness.py: _formula_capability reads capability_features[:, :10], averages and clips. mimo_bridge/proprio_adapter.py: FatigueModel.step maintains the capacity numbers.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "A missing capacity vector returns zero, which is indistinguishable in this score from all ten groups being depleted. The recorded input is needed to distinguish them. This is a simulated capacity model, not a direct measurement of biological fatigue. With body_function=modulate, these capacities also scale actuator gain and force limits, changing the available body strength."
    },
    "term.muscle-groups": {
        "t": "Ten muscle-group capacity numbers",
        "w": "The groups are left arm, right arm, left hand, right hand, spine, head, left leg, right leg, left foot and right foot. Each has one capacity number.\n\nOn each physics update, capacity gains elapsed seconds times 5% of its missing capacity, and loses elapsed seconds times 2% times the group effort divided by its effort reference. The result is clipped between zero and full capacity. The wellness capability score averages the ten results.",
        "h": "lem/model/body_sets.py: BodySet. mimo_bridge/proprio_adapter.py: FatigueModel.step; effort_per_set supplies group effort.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The effort_power setting matters. With no_gear, effort sums absolute actuator force times joint speed without actuator gearing; it is not mechanical power in watts. With with_gear, the calculation includes gearing. Do not label the no_gear reading as watts or interpret the configured rates as measured infant physiology."
    },
    "term.eval-coherence": {
        "t": "Coherence: body-map co-movement",
        "w": "This score uses the body map. For each handle, the map compares how much its command changed with how much each proprioception measurement changed during the same model step. Large changes occurring together increase a running score. The map converts that score into a number between zero and one. This measures co-movement; it does not establish that the command caused the movement.\n\nWith the extent setting, wellness averages these numbers for the handles and tracked outside objects. Outside objects are scored by how closely their movement follows handle movement, after a mismatched-handle comparison. {wellness_coherence_choice}\n\nThe wellness calculation reads the existing map before the current model step updates it. It does not compare sensory predictions with measurements when the body map is enabled.",
        "h": "lem/model/body_schema.py: update normalizes command-change length and absolute proprioception changes by running root-mean-square, averages their products, and applies sigmoid(temperature * (score - one half)). At temperature 5, scores 0 and 1 give about 7.6% and 92.4%. self_extent averages handle and object memberships. mimo_bridge/lem_model.py: _schema_coherence. The default formula is sigmoid(5 x (score - one half)). It uses no time-lag bank, no sign and no passive-movement comparison for handles. In the synthetic checks in test_body_schema_co_movement.py, unrelated random changes read 80 of 100 when commands change every model step and 30 of 100 when commands change every fifth model step. A response delayed by one model step reads 8 of 100, the same as movement without command changes. These examples explain why this number cannot establish causal control.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "This page rescales the extent display between those two endpoints. The certainty alternative averages 2 * abs(membership - one half), rewarding either extreme. With body-map updates off, the fallback instead averages the cosine similarity of predicted and measured sensory lists. These are different measurements. The October 3 operator decision remains: call the current measure co-movement and replace its input after the motor-encoder repair; this explanation does not implement that replacement."
    },
    "term.eval-motor-state": {
        "t": "Motor state: handle positions relative to a configured boundary",
        "w": "**Input:** each handle position minus the mean position of all handles, with each coordinate divided by 200 cm. These are measured handle positions, not proposed target positions or joint-limit angles.\n\n**Boundary formula:** the coordinate limit is 42 cm times the current developmental range fraction. A coordinate inside the first 90% of that limit has no penalty. Across the final 10%, its penalty rises as the square of its progress through that band; at and beyond the limit its penalty is one. Average the penalties across all handle coordinates and subtract from one.\n\n**Use:** a higher score means fewer/lower coordinate penalties. The derivative of this formula is an input to the motor controller. It is not a measurement that every joint is within its physical limit.",
        "h": "mimo_bridge/lem_model.py: step constructs centroid-relative motor_commands from handle_state. lem/model/wellness.py: _formula_motor_state_boundary uses BODY_REACH = 21/100 in 200 cm units, multiplied by the developmental ROM fraction. The alternative preference formula subtracts the mean absolute normalized coordinate from one and clips.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The displayed score saturates beyond the boundary, but the implemented backward calculation keeps an inward derivative there. [[wellness-gradient|How that derivative reaches the controller.]] The boundary is a configured Cartesian envelope, not the individual joints' mechanical limits."
    },
    "term.eval-movement": {
        "t": "Movement: a speed bonus and an inactivity penalty",
        "w": "Start at 50%. Add a bonus that grows with handle speed relative to the body, then multiply that bonus by movement smoothness and the body map's average co-movement number. Subtract a penalty for accumulated inactivity. Clip the result between zero and one. This is an extra goal-ranking channel, separate from the six fixed scores.\n\nInactivity increases while speed stays below a quarter of the reference speed. It reaches its maximum after 150 seconds starting from zero. Movement reduces it; 30 seconds at reference speed clears a full amount. The penalty uses the square of this stored inactivity amount.",
        "h": "mimo_bridge/movement_drive.py: MovementDrive.update. score = one half + bonus*tanh(speed/reference)*self_factor*smoothness - penalty*debt**2. Local speed subtracts the mean handle velocity; smoothness uses a running average of successive world-velocity direction agreement.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The bonus and penalty are configured weights, not measurements of how strongly the planner will move. The co-movement factor is shared across bodies and does not establish causal control. [[eval-coherence|How that factor is measured.]]"
    },
    "term.goal-ring": {
        "t": "goal ring",
        "w": "The planner's memory of the last 200 model steps - {goal_ring_s} seconds of body time "
             "at the {moment_ms} ms model step of {run_name}: one [[goal-ring-entry|entry]] per model "
             "step, holding every body's values, written at the end of every model step and dropped "
             "200 model steps later. An entry holds that model step's tokens and its wellness values "
             "(a stored value, and under --wellness-rank normalized one value per wellness channel). On "
             "a planner step the planner gives every entry a ranking number - the stored value under "
             "--wellness-rank raw, the [[wellness-ranking|normalised ranking]] under normalized - and "
             "the tokens of the highest-ranked entry are each body's goal. Planning starts once the "
             "ring holds 20 entries; an episode reset or a rebuilt body (LemLite."
             "reset_transition_state) empties it.",
    },
    "term.forward-model": {
        "t": "forward model, and what 'frozen' means",
        "w": "The prediction stack run as a simulator, once per candidate command: the motor "
             "[[encoder]] turns the command and the body's measured handle state into the motor "
             "[[token]]; the [[transformer]] core runs on this model step's tokens of the other "
             "senses with that motor token and the context of the previous {context_minus_1} model "
             "steps; and the prediction heads write the next model step's tokens, {d_model} numbers "
             "for every sense except motor, each passed through the shared output "
             "LayerNorm{gaze_copy_clause}.{memory_bias_clause} FROZEN IS NOT A SECOND COPY: the planner uses the very "
             "weights the run is training. For the length of ONE planning call - 20 Adam updates - it "
             "switches off gradient collection on every weight along that path: the {n_senses_word} "
             "encoders, the core, the prediction heads, the gaze copy's heads, the synergy's and the "
             "physical goal's modules when they are in the search, and the [[wellness]] scorer when the "
             "prospective-wellness part is on. It records each weight's setting first and restores "
             "exactly that setting when the call ends, so a weight frozen on purpose stays frozen and a "
             "trainable one becomes trainable again. Gradient still FLOWS THROUGH the frozen weights - "
             "that is how the derivative of the score with respect to the command is computed - and is "
             "stored only on the command's numbers, the only numbers that change. The imagination "
             "chain runs the same path the other way: its command is the one the body executed, and "
             "the weights are what its error trains.",
        "h": "lem/model/intention_generator.py: _freeze_params sets "
             "requires_grad False and returns the prior flags, "
             "_restore_params puts them back, both inside generate_action; "
             "the frozen list is forward_model, encoder_registry, "
             "prediction_heads plus _wellness_freeze, the gaze copy's heads and the action space's and "
             "physical goal's frozen_modules(); one run is lem/model/predictive_state.py transition. "
             "Size of the core being frozen in {run_name}: {n_blocks} blocks, {core_learned} learned "
             "numbers ({core_saved} saved, the difference being the blocks' causal masks, "
             "{core_mask_cells} true/false cells); the whole frozen path also carries the "
             "{encoders_params} encoder numbers and the {heads_params} in the prediction heads.",
        "f": "reading 'frozen' as a separate, older model kept aside - the planner uses the newest "
             "weights on every call, so a bad update reaches the planner at the next planning event. "
             "Second trap: reading it as 'no gradient' - gradient flows through the weights and is "
             "stored only on the command",
    },
    "term.source-label": {
        "t": "source labels: real / stale / synthetic",
        "w": "How a sense arrived this frame. real = measured this step. "
             "stale = the previous value reused, because no new reading "
             "arrived - reachable in the code, but this run never produces "
             "one: the trainer renders the eyes on the last physics sub-step "
             "of every model step, and all 20 of its snapshots label vision real. "
             "synthetic = generated placeholder noise. The loss "
             "manager would weigh a stale lesson at 30% of a real one's "
             "weight, but every lesson these runs grade picks only the senses "
             "labelled real, so a stale sense teaches nothing either; "
             "synthetic weighs zero everywhere. In all three cases the token "
             "still enters the [[transformer]]'s input.",
    },
    "term.collapse": {
        "t": "encoder collapse",
        "w": "The failure where an encoder outputs the same token no matter "
             "what the body senses. Its watch-number is the token's variation "
             "over time: zero = collapsed.",
        "h": "the standard deviation of each token dimension over the last 200 "
             "stored samples, averaged over the {d_model} dimensions. Samples are "
             "taken every 5th [[model-step]] from BODY 0 only, so the window is "
             "about 1,000 model steps - {collapse_window_s} seconds of body time - for one "
             "body, not a batch statistic. It reads RAW encoder tokens on "
             "purpose, so a top-down gate blending predictions into the tokens "
             "cannot hide a collapse",
        "f": "reading it as a whole-run number: a sense flat for {collapse_window_s} seconds and "
             "one flat all run read the same here, and a collapse in the body "
             "it does not sample is invisible. The third and largest limit: "
             "the 1,000-step window also contains the encoder's own weights "
             "moving, and for a flat encoder that movement is nearly the whole "
             "number - measured as a percentage of a token's size (1.0), the "
             "intermind encoder reads 4.2% in this run, while "
             "the same encoder with its weights held still moves 0.06% and "
             "two checkpoints 2,268 steps apart pooled give 8.1%. An encoder "
             "that has genuinely stopped answering its input cannot read near "
             "zero here.",
    },
    "term.fovea": {
        "t": "fovea",
        "w": "The sharp center of vision. Here: a second 288-pixel camera per "
             "eye covering the middle 24 degrees, like the human fovea inside "
             "the wide 60-degree view.",
    },
    "term.frequency-band": {
        "t": "frequency band",
        "w": "A slice of pitch range. Audio reports how strong the sound is "
             "in each of 256 slices as a number from 0 to 1 on a logarithmic "
             "scale, written here as a percentage of it: 100% is a summed level of 1, "
             "every eighth of the scale (12.5%) below it is a "
             "tenfold drop in level - so 50% is a thousand times the level of "
             "12.5% - and the scale bottoms out at 0%, a level a hundred million "
             "times smaller than the top; anything quieter also reads 0%. (The "
             "code adds a guard of one billionth before the logarithm, which lifts "
             "a level sitting exactly on that bottom to 0.5% of the scale and changes "
             "nothing measurable from 25% up.) The code computes it as 10 x "
             "log10(level) with a floor of -80, which looks like decibels, but "
             "the level it logs is amplitude-like - each source's gain times "
             "its spectral shape, with speech added as the square root of its "
             "power - so for speech one eighth of the scale is 20 decibels of sound "
             "pressure, and for the synthesized sources, whose gains are chosen "
             "numbers, no decibel value applies. Slices run from 40 Hz (a "
             "deep hum) to 8,000 Hz (a hiss), spaced like octaves, separately "
             "per ear.",
    },
    "term.principal-stretch": {
        "t": "principal stretch",
        "w": "How much a small patch of skin is stretched along its two surface "
             "directions, as a share of its resting length: 100% = resting length, 120% = stretched 20%, 80% = "
             "compressed by a fifth. On the UE avatar it is computed from the mesh's "
             "deformation against the rest pose, so skin near a bending joint "
             "registers stretch during movement even with no contact. In the "
             "MIMo crib there is no soft skin: both stretch numbers are the "
             "constant 1.0, and the bridge's tactile token keeps only the first "
             "8 of the patch's 10 fields, so neither number reaches the "
             "[[encoder]]. The channel is real only on the UE avatar.",
    },

    # ---------------- inspector panel fields ----------------
    "header.weights": {
        "t": "weights label",
        "w": "Says whether the attention maps were drawn with the exact weights "
             "the model used this step. 'frozen' means yes: an eval probe never "
             "updates weights. 'post_update' means the weights had already "
             "taken this step's one learning update before the maps were "
             "redrawn.",
        "h": "'frozen' in --eval-only probes; 'post_update' in live training, "
             "where optimizer.step() runs inside model.step()",
        "f": "treating a post_update map as exact - one update at the "
             "configured learning rate (three ten-thousandths, the model "
             "config's training.lr) shifts it; how far is not measured",
    },
    "tokens.raw": {
        "t": "raw encoder tokens",
        "w": "Each sense's [[encoder]] output for this frame: one [[token]] per "
             "sense, with nothing blended in. All grading uses these.",
        "h": "model.last_tokens; the shared output [[layernorm|LayerNorm]] "
             "holds each token at [[rms|RMS]] 1.0",
        "f": "these are not always what the core read - with the top-down gate "
             "open, the core read tokens.fed",
    },
    "tokens.fed": {
        "t": "tokens the core consumed",
        "w": "The tokens the [[transformer]] actually read this step. Usually "
             "the same as the raw tokens; when the top-down gate is open, up to "
             "half of the model's own guess of NOW is mixed in.",
        "h": "model.last_tokens_fed - equals raw when --top-down-gate is off or "
             "closed; the gate slams shut on persistent surprise (fast error "
             "average above twice its slow baseline)",
        "f": "grading predictions against these would let the model grade "
             "itself against its own guess - grading always uses raw tokens",
    },
    "predictions.of_now": {
        "t": "prediction of this frame",
        "w": "One step ago, knowing its own [[executed-command]], the model "
             "guessed what THIS model step would look like. Here that guess meets "
             "what really arrived. The per-sense error averages into "
             "[[causal_pe]].",
        "h": "the delayed next-frame lesson: the previous frame is re-encoded "
             "with the executed command in the motor channel, run through the "
             "whole stack, and graded against this frame's raw tokens",
        "f": "absent on the first step after a reset or with --next-frame-loss "
             "off - the block then says why",
    },
    "predictions.of_next": {
        "t": "prediction of the next frame",
        "w": "The model's guess about the NEXT model step, made just now. Nothing "
             "has arrived yet to grade it.",
        "h": "prediction heads on this step's output (model.last_predictions); "
             "stored absolute even under --residual-prediction (the heads add "
             "the current token internally)",
        "f": "scoring it against THIS frame measures predicted change, not "
             "correctness",
    },
    "attention": {
        "t": "attention maps",
        "w": "Where the [[transformer]] looked while computing this step. "
             "Bright cell = much weight. Each row is one token doing the "
             "looking, and each row's weights add up to 1 - a fixed budget of "
             "[[attention]], spent across the columns.",
        "h": "one no-gradient re-forward of the same tokens and context the "
             "live step used, with capture_attention='all'; heads averaged; "
             "stored 8-bit",
        "f": "in live-training snapshots the weights had already taken one "
             "update (see the weights label); stage 3 stores the current "
             "frame's rows plus summaries, not its full matrix; and the "
             "re-forward reproduces the attention WEIGHTS exactly while its "
             "OUTPUT tokens do not match the live step's - the "
             "persistent-memory bias is not handed to it, and that bias is "
             "added after the final LayerNorm, which is why the weights are "
             "unaffected",
    },
    "attention.stage1": {
        "t": "stage 1 - senses read each other",
        "w": "At one single [[model-step]]: how much each sense's token read each "
             "other sense. Row = the sense doing the reading.",
        "h": "stage-1 cross-modal attention, reshaped [model steps, senses, "
             "senses]; rows sum to 1",
        "f": "columns are not normalized - only rows are budgets",
    },
    "attention.stage2": {
        "t": "stage 2 - each sense reads its past",
        "w": "Within one sense: how much this model step read each earlier model step. "
             "This is the model's short-term memory in use, and it is the only "
             "one of the three stages that carries a mask.",
        "h": "stage-2 temporal attention [senses, model steps, model steps] under the "
             "causal mask - no token may read the future, so the upper "
             "triangle is zero",
        "f": "a row that spends almost its whole budget of 1.0 on its own "
             "newest model step is memory not being used; a row that spreads the "
             "budget back over earlier model steps is memory in use. For the first "
             "full window after any reset or body rebuild ({context_length} model steps = "
             "{window_s} s at this run's settings) the older columns are empty "
             "tokens the window has not refilled, so weight there is attention "
             "paid to padding.",
    },
    "attention.stage3": {
        "t": "stage 3 - everything reads everything",
        "w": "The stage where any sense at any model step may read any other - an "
             "earlier model step or a LATER one, because this stage carries no "
             "mask. Shown: what the current frame read, a sense-pair mean, "
             "mean attention by time distance in both directions, and the "
             "share of the whole attention budget spent on earlier model steps, "
             "on the same model step and on later model steps.",
        "h": "stage-3 bridge attention: current-frame rows [senses, sequence]; "
             "a sense-pair mean; stage3_lag and stage3_lag_forward, the mean "
             "weight at each time distance looking back and looking forward; "
             "and stage3_direction_share, three numbers over one total that "
             "add to 1",
        "f": "reading the two curves as shares: each point averages a "
             "different number of cells, so the curves do not add up to "
             "anything - the three-way share under the chart is the quantity "
             "that does. Fine detail is only in the current-frame rows. A "
             "snapshot written before 2026-09-16 carries the backward curve "
             "alone, and the chart's heading then says so.",
    },
    "motor.prev_executed": {
        "t": "previously executed command",
        "w": "The [[executed-command]] in force during the previous interval - "
             "after limits and [[babble]] noise. It caused the frames this "
             "snapshot shows, and it is what the model's prediction was graded "
             "against. On a scheduled-rest step (a run launched with "
             "--babble-bouts real and --rest-control scheduled, as this one "
             "was) its position and rotation are overwritten with the body's "
             "OWN achieved pose - an active postural hold - so on those steps "
             "the cause the model is handed reads 'stay exactly where you "
             "are'.",
        "h": "the trainer's transition record (prev_executed); the gaze slot "
             "carries the eye command the world really used; the rest-step "
             "substitution is train_mimo.py's scheduled_hold branch, and the "
             "share of steps that were NOT rest is the babble_duty stat",
        "f": "this is not the command in force now - that is the held command",
    },
    "motor.held_order": {
        "t": "command about to be issued",
        "w": "The command the body is being given for the coming interval. One "
             "command stays in force for several steps ([[motor-hold]]).",
        "h": "the trainer's held_motor after this step's decision, with "
             "order_seq and order_age (steps since the command last changed)",
        "f": "computing command age as step-modulo-hold goes wrong after resets; "
             "this counter follows actual command changes",
    },
    "drives.prediction_error_source": {
        "t": "Which sensor readings produced this accuracy score?",
        "w": "The recorded label identifies the two lists used for [[eval-prediction-error|prediction accuracy]]. Delayed causal prediction vs raw tokens means the previous sensor readings and executed command were used to predict the current encoded readings. Same-frame reconstruction pair means the fallback used current predictions and current encoded readings. With residual prediction enabled, the fallback measures the size of a predicted change. [[drives-pairing|Read the timing and fallback calculation.]]",
        "h": "mimo_bridge/lem_model.py: LemLite.step sets last_pe_source at the wellness call. explainer/capture.py copies it into drives.prediction_error_source.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "drives.wellness": {
        "t": "Measured wellness scores for the selected body",
        "w": "These bars show fixed-formula outputs for the selected body at this recorded model step. Click a score name to read its inputs, calculation and limits. The prediction_error bar is higher when encoded predictions match encoded measurements more closely. The tactile_comfort bar has a maximum of 85% and reads 50% without contact or remaining load history. The self_touch bar has the configured weight as its maximum. {self_touch_weight}\n\nThese scores are separate from the learned predictor outputs and from the number used to rank remembered goals. [[wellness|The three calculations.]]",
        "h": "explainer/capture.py: drives.wellness_body from last_wellness_grounded. mimo_bridge/lem_model.py adds self_touch to the five scores from WellnessEvaluators.forward.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "incorporation": {
        "t": "the body map's co-movement numbers (UEMA 11.2 calls this incorporation)",
        "w": "The body map (the code calls it `body_schema`) keeps one co-movement number per "
             "thing it watches: one for each of the body's 26 "
             "[[handle|handles]] and one for each outside thing on its list - "
             "the toy and the mobile on the crib runs. A HANDLE'S NUMBER counts how often the size "
             "of that handle's command change and the size of proprioception's change are large in "
             "the same model step ([[eval-coherence|the computation, step for step]]). It does not "
             "say the command caused the change, and nothing is subtracted from it: on synthetic "
             "inputs two unrelated random signals read 30 to 80 of 100 on the sigmoid's own "
             "0-to-100 output, and a body that answers every command one model step late reads 8 of "
             "100. AN OUTSIDE THING'S NUMBER COMES FROM GOING "
             "WHERE A HAND WENT, AS FAR AS IT WENT, IN THE SAME MODEL STEP: the likeness of the two "
             "directions times the smaller travel over the larger, so a toy "
             "struck away scores its distance ratio and a toy carried with the "
             "hand scores near 1.\n\n"
             "THE TWO ENDS OF THIS SCALE ARE NOT 0 AND 1. The map runs each "
             "score through sigmoid(5 x (score - one half)), so a score of 0 "
             "reads 7.6% of the 0-to-1 range and a score of 1 "
             "reads 92.4%. The page therefore writes every co-movement "
             "number as a percentage of the way between those two ends: 0% = a score of 0 (a "
             "handle whose command change and proprioception's change never fell in the same model "
             "step; a thing that never travelled with a hand), 100% = a score of 1 (a handle whose "
             "two change sizes matched on every model step; a thing that travelled exactly with a "
             "hand). For an outside thing, past the "
             "middle (50%) means the thing kept more than "
             "half the hand's distance in the hand's own direction. The "
             "body-wide rows sit on that same scale: 'handles' averages the 26 handles' numbers and "
             "'extent' averages every number, the outside things' included. The "
             "'certainty' row is a different "
             "quantity again - the average distance of each number from the "
             "middle of the range - which runs from 0% when "
             "every number sits exactly on the middle to 100% when every number is "
             "at one end of the scale or the other.\n\n"
             "EACH OUTSIDE THING IS SHOWN WITH ITS CONTROL. The match column is what "
             "the thing scored against the best-matching handle; the control "
             "column is the same score with that handle's direction reversed, "
             "which cannot be carrying; earned is the match minus the control, "
             "floored at 0, and it is what the map converts into the thing's "
             "number. Without the subtraction the live toy would have "
             "counted: its match read 14% of a toy carried exactly with the hand, "
             "and its control - which cannot be carrying - read MORE, 17%. A handle's number has "
             "no such control.\n\n"
             "CAN THE BODY MOVE THIS THING AT ALL? The last column reads the "
             "world's own report for the snapshot's model step. The toy is a MuJoCo "
             "mocap body: MuJoCo writes a mocap body's place from outside every "
             "step and never integrates it, so no push can displace it. Without "
             "the [[toy-hand-off|toy hand-off]] its whole path is the adult's, "
             "and a toy score can only be the adult happening to carry the toy "
             "the way a hand was going. With --toy-hand-off follow the adult "
             "lets go while the infant grips: two parts of one hand on the toy "
             "for 0.1 s and the toy goes where that hand goes, until no part of "
             "that hand has touched it for 0.1 s - so a hand CAN carry it, and "
             "the world reports the toy as movable by the body. The mobile "
             "hangs on a ball joint and does swing when it is struck.",
        "when": "Every model step the body map is updated (--body-schema-updates "
                "on, which the crib runs set); the panel reads the snapshot, "
                "and the same numbers go into every stats record as "
                "'incorporation'.",
        "h": "mimo_bridge/lem_model.py incorporation_stats(), which reads "
             "lem/model/body_schema.py: self_membership() for the handles (the stats key "
             "handles_mine is their mean), "
             "object_membership() for the outside things, object_evidence() for "
             "the match/control/earned columns, self_extent() for the extent "
             "row and coherence_score() for the certainty row. The last column "
             "is object_physics_driven / mobile_physics_driven, straight from "
             "the world's per-body report. The synthetic readings are held by "
             "mimo_bridge/tests/test_body_schema_co_movement.py",
        "f": "reading a handle's number as how much of its body the model owns or controls - it "
             "counts command changes and proprioception changes that fall in the same model step, "
             "with no lag, no sign and no comparison against movement without a command. Second "
             "trap: reading a toy's number as the infant having carried the toy "
             "while the world gives the infant no way to move it - "
             "that is what the last column is for. Third trap: reading the "
             "map's raw number (7.6% of the 0-to-1 range) as 'almost none': "
             "that is what a score of 0 reads, so it is the floor of the scale "
             "(0% on this page) and not a small amount of co-movement",
    },
    "intention": {
        "t": "the plan",
        "w": "The newest planning event's record (model.last_intention): the plan's length in "
             "commands; its licensed depth; the distance from that model step's tokens to the goal "
             "tokens - for every sense measured then except motor, the mean over the token's numbers "
             "of the squared difference, added over those senses and averaged over the bodies, so two "
             "unrelated tokens add about 2 per sense; the predicted distance after each command but the "
             "last; the rule that stopped the plan; and, per depth, the 20 updates' own record: the "
             "[[intention-distance|score]]'s goal part and strength part for the starting command and "
             "for the kept one, and the update each body kept. Where the goal entry sits in the "
             "[[goal-ring]] and its stored value come from the capture. The intention box on the "
             "architecture tab opens the whole computation with its diagram.",
        "when": "On a planner step - a model step whose number is a multiple of --intention-every - "
                "once the goal ring holds 20 entries; between planner steps the record keeps the "
                "newest planning event's.",
        "h": "model.last_intention, written in LemLite.step from LemLite._plan_chain's record; its "
             "descent block is generate_action's own record of the 20 updates (first_goal_distance, "
             "last_goal_distance, first_effort, last_effort, selected_iteration); a plan the size "
             "check drops leaves no record for that planning event",
        "f": "- The plan record's distances are not the score the 20 updates lower: they leave out the "
             "strength part.\n"
             "- A one-command plan (every plan under --hand-target-mode wrist_relative_v1) records no "
             "predicted distance after its command, so whether it came nearer the goal cannot be read "
             "from the distances; the descent rows give the score's goal part before and after the 20 "
             "updates.\n"
             "- On a plan of two commands or more, a closest predicted distance equal to the start "
             "means no predicted depth came nearer the goal tokens - measured on H83 that was 137 of "
             "137 planning events, each stopping at 2 of a licensed 3.\n"
             "- 'No plan at this model step' is normal: most model steps are not planner steps.",
    },
    "losses": {
        "t": "live-step loss breakdown",
        "w": "This step's teaching signal, split into terms. Each term is one "
             "[[loss|lesson]]; a term at zero taught nothing this step.",
        "h": "StepResult.breakdown, live step only. The motor term is computed "
             "from the motor controller run a second time on the PREVIOUS "
             "model step's stored inputs (motor_for_loss in "
             "mimo_bridge/lem_model.py), graded against the pose reached now",
        "f": "replay updates (re-study of remembered model steps) run separately "
             "and are not in this panel; and reading the motor bar against "
             "the command panel's 'proposed now' column - that column is this "
             "model step's new proposal, a different output from the one the bar "
             "grades",
    },
    "senses.eyes": {
        "t": "eye images",
        "w": "The exact pictures the vision encoder consumed this step: a wide "
             "view and a sharp center ([[fovea]]) for each eye.",
        "h": "frames[body].vision_patches re-assembled into images; absent only "
             "when vision was reused from the previous step "
             "([[source-label|stale]]), which does not happen in these MIMo "
             "runs: the trainer renders on the LAST sub-step of every "
             "model step, and the fresh-vision share reads 100 of every 100 steps in every stats "
             "window of both this run and the training run behind it (100 = "
             "every step got a newly rendered picture, 0 = none did)",
        "f": "absent does not mean blind - a stale step reuses the previous "
             "vision token",
    },
    "tiles": {
        "t": "the five verdict tiles",
        "w": "The questions this project exists to answer, in order of "
             "importance, each with the one number that answers it. The tiles "
             "read the run-replay series, so when this page's charts come from "
             "a different run than its snapshots, the tiles describe the run "
             "the CHARTS came from - the line above them says which. One "
             "sentence is the exception: tile 5 ends with a line read from a "
             "snapshot - the one selected on the inspector tab, the newest one "
             "on the run tab - and that line names its step and, when the "
             "charts are another run's, the run it belongs to.",
        "h": "viewer.js renderTiles: each trend is the mean of the LAST quarter "
             "of that series' points against the mean of its FIRST quarter, "
             "inside the current process segment (a window never straddles a "
             "resume seam), with a quarter capped at 100 points",
        "f": "a log holding no resume seam gives one window spanning every "
             "record in it, so a configuration change inside that span is "
             "averaged away; and a tile whose series is missing is left out "
             "entirely, which is not the same as that series reading zero",
    },

    # ---------------- architecture diagram boxes ----------------
    "node.sense.vision": {
        "t": "vision",
        "w": "The eyes. Four pictures every [[model-step]]: a wide view (256 px, "
             "60 degrees) and a sharp center ([[fovea]]: 288 px, 24 degrees) "
             "for each eye, as raw pixels cut into 16x16 patches - every "
             "colour value 0 to 1 (the camera's 0-255 divided by 255). Each "
             "patch carries four tags: where it looks (azimuth and elevation, "
             "in radians), which eye saw it (-1 left, +1 right), and how much "
             "world one patch covers (3.75 degrees in the wide view, 1.33 in "
             "the sharp centre). A fifth camera - a 150-degree far-periphery "
             "motion detector - exists, but it feeds only the reflex that "
             "snaps the eyes toward sudden movement; the model never sees its "
             "pixels.",
        "when": "Every [[model-step]]. The eyes are rendered on the last physics "
                "sub-step of each model step, so every model step the model reads "
                "carries a fresh image: this run's launch check found 10 fresh "
                "images on the 10 frames the model read, and all 20 of its "
                "snapshots label vision [[source-label|real]]. --render-every "
                "is never consulted while the trainer says which frames it "
                "will read.",
        "h": "env.py builds wide+fovea per eye (BinocularPatcher); the motion "
             "ring is gaze_reflex.py - frame difference, median-subtracted, "
             "with habituation",
        "f": "no VISION channel encodes motion - visual movement has to be "
             "found by comparing model steps inside the transformer. The other "
             "senses do measure it directly: proprioception carries each part's "
             "velocity in cm/s, and tactile carries a slip speed and four "
             "vibration strengths",
    },
    "node.sense.tactile": {
        "t": "tactile",
        "w": "The skin's sense of touch. 4,683 tiny pressure points on this run's 24-month body "
             "- 5,330 on the ungrown newborn body, because the points are laid "
             "out by the body's size and grow with it - are pooled into one "
             "reading per skin part (53 parts, the same count at every age). "
             "Every part with any nonzero reading becomes a tactile row: "
             "pressure, rub, slip or contact area now, or a vibration "
             "strength, which stays nonzero for up to one second after a touch "
             "ends because it is measured over the last second of pressing "
             "force. The rows are sorted by pressure, highest first, and cut "
             "at 32. So a row is not always a part being pressed: on the H81 "
             "probe's 12 embedded snapshots 94 of the 190 rows - 49 in every "
             "100, where 0 would mean every row is a part pressed now - had "
             "pressure, rub, slip and area all at zero and only that leftover "
             "vibration. Each row is 19 "
             "numbers: where the part sits relative to the hip (3), pressure, "
             "sideways rub (shear), slip speed, contact area, four vibration "
             "strengths (5-20 Hz), how much of the force came from the body's "
             "OWN parts (0% = none of it, 100% = all of it) and how many own parts "
             "are on it over 4, then the push itself - which way it came from "
             "(3 numbers, a unit direction pointing into the body), how deep "
             "the contact is and how fast it arrived - and a validity flag "
             "last.",
        "when": "Every [[model-step]]. Measured over this run's own 400 model steps: "
                "7.5 skin parts in contact on average, with no 50-model-step "
                "stretch averaging below 5.2 or above 8.5; 32 is the cap.",
        "h": "touch_adapter.py pools skin points per body part and keeps each "
             "part's last 100 pressing-force samples (1.00 s at the 0.01 s env "
             "step) for the vibration strengths; lem_model.touch_point_cloud "
             "admits every part whose row has any nonzero field, sorts the "
             "admitted rows by pressure, highest first, and keeps the first 32 "
             "of the 53",
        "f": "32 caps how many PARTS are reported at once, not the skin's "
             "resolution. Past 32, rows are dropped from the low-pressure end: "
             "the zero-pressure rows go first, then the gentlest presses. "
             "Second trap: reading the row count as the number of parts in "
             "contact - on the H81 probe 49 of every 100 rows were parts whose "
             "touch had already ended",
    },
    "node.sense.audio": {
        "t": "audio",
        "w": "What the two ears hear. All sound arrives as a spectrum: how strong the sound "
             "is in each of 256 [[frequency-band|frequency bands]] (40 Hz to 8 "
             "kHz), separately for the left and right ear, as a number from 0% "
             "(silence) to 100% (the top of the scale), each eighth of the scale a tenfold step "
             "in level. Both ears get "
             "the SAME spectrum shape at two different loudnesses - the far "
             "ear is quieter by a fixed head-shadow factor, 65% of the near "
             "ear's level for a source straight out to one side. The far ear also "
             "hears LATER, and that is not in the spectrum at all: the "
             "arrival-time difference travels as one extra number in "
             "milliseconds (see the arrow into the encoder), and only the "
             "caregiver's speech clips carry a real per-ear delay in their "
             "waveform. This world's sounds: the "
             "caregiver's speech, body impacts, the squeaker toy, the mobile's "
             "chime, the bee and the dragonfly. The model cannot hear its own "
             "voice: nothing carries the voice head's output into the world "
             "yet.",
        "when": "Every [[model-step]] - audio is never stale.",
        "h": "audio_adapter.py: 256 log-spaced bins; speech is a cached "
             "waveform - in this run 76 clips of 0.96 s to 3.04 s, "
             "rendered ahead of time by a text-to-speech voice, not a human "
             "recording - put through an FFT (64 ms window); every other "
             "source is "
             "synthesized directly in spectrum space as one shape times a "
             "per-ear gain, with the distance taken from the head centre for "
             "both ears; two ears 7 cm apart, so the arrival-time difference "
             "reaches 0.204 ms at most (7 cm at 343 m/s) and is reported as "
             "its own number rather than as anything in the spectrum",
        "f": "a 16-band side input exists because convolution+pooling erases "
             "absolute pitch - measured: the same pattern at three pitches "
             "produced identical tokens before it was added",
    },
    "node.sense.proprioception": {
        "t": "proprioception",
        "w": "The body's sense of where its own parts are and how they move. For each tracked body part - one row per [[handle]], so 26 on "
             "the trunk body (25 bones plus the eye direction; the legacy "
             "body has 24): its orientation as a [[quaternion]] (4 numbers), "
             "its position measured from the LEFT WRIST, which is row 0 (3), "
             "its velocity as a full 3-D vector - direction and size, not a "
             "bare speed - and one freshness number from its "
             "[[muscle-groups|muscle group]]. 11 numbers per part.",
        "when": "Every [[model-step]].",
        "h": "proprio_adapter.py reads a world [[quaternion]], a world "
             "position in cm, MuJoCo's linear velocity vector in cm/s and the "
             "body-set capacity; the tokenizer then subtracts ROW 0's position "
             "from every row and divides by 200 cm. Row 0 is wrist_L, the "
             "first name in BONE_MAP. Tactile measures from the hip instead, so "
             "the two senses do not share an origin. The eye row's orientation "
             "is head-relative.",
        "f": "row 0 is the left hand, the most mobile part in the set, so the "
             "origin moves: when that one hand moves and nothing else does, "
             "every part's position numbers change together. A trunk origin - "
             "the hip, which tactile already uses, or the lower-trunk and chest "
             "rows this body layout added - would hold still. Separately, the "
             "UE avatar's richer version adds skin strain - each skin patch's "
             "[[principal-stretch]] while moving - routed into TACTILE there; on "
             "this branch the tokenizer delivers all 10 tactile fields and the bridge's 19-wide tactile row keeps the first 8 of them as its felt columns (touch_point_cloud in mimo_bridge/lem_model.py; the other 11 columns are the part's place, the two self-touch numbers, the push and the validity flag), and on MIMo those two stretch "
             "channels are a constant zero anyway - MIMo has no soft skin",
    },
    "node.sense.motor": {
        "t": "motor",
        "w": "The model's own last [[executed-command]], fed back as a sense - "
             "the held command after limits, [[babble]] noise and smoothing, "
             "with the eye command the world really used. Feeding it back puts the "
             "command beside the pose the body reached in one input row, the only place the "
             "transformer reads the command.",
        "when": "Every [[model-step]]. The current model step's slot carries the "
                "achieved pose with the command half zeroed - it is filled in "
                "the next step, once the command has actually acted.",
        "h": "train_mimo hands prev_executed to the model; the context's "
             "previous motor slot is back-filled with [achieved pose, "
             "executed command]. The achieved pose's positions are world "
             "centimetres divided by 200. Proprioception subtracts row 0 (the "
             "left wrist) first, so the same handle's position reaches the two "
             "encoders in two origins in one model step; measured on a live crib "
             "frame the two blocks differ by 23 cm on their largest coordinate, "
             "against a block that spans 63 cm end to end.",
        "f": "it is NOT motor_out - that is this step's fresh proposal, which "
             "can reach the body only on a latch model step - one model step in "
             "{motor_hold} in {run_name} - and then only when no plan was made "
             "on that model step, after limits and noise",
    },
    "node.sense.intermind": {
        "t": "intermind",
        "w": "A reserved channel for this model's OTHER MINDS - the planned "
             "split of the one brain into cooperating controllers for body "
             "regions (posture, each arm, each hand, gaze...). Each mind will "
             "expose a learned summary of its [[transformer]]'s last hidden "
             "layer for the others to read in the same [[model-step]] - like "
             "feeling a sibling's intent directly, not like mail. Today it is "
             "a placeholder: faint random noise whose lessons weigh zero.",
        "when": "Placeholder noise arrives every model step. Plan (spec v3.5): "
                "first the single mind reads its own summary ('self-reading', "
                "at a tenth of full loss weight); actual splitting waits for single-mind "
                "mastery - prediction error steadily under 5% of the all-zeros "
                "answer's error across a 50,000-step window.",
        "h": "ManyMinds_Spec_v2 section 2.3 (readable internal state, not "
             "messages); current stub: gaussian placeholder, loss weight 0, "
             "excluded from wellness",
        "f": "What reaches attention is not moving noise. The encoder learned "
             "to turn this draw into one fixed direction: with its weights held "
             "still the token moves 0.06% of a token's size, while "
             "every sense the world does feed moves 12% to 65% of it. An untrained "
             "encoder on the same draw moves 59.5%, so the flatness was "
             "learned, not built in. The cause is the clip at 0: it leaves a "
             "fixed positive offset of 0.4% of the input's 0-to-1 range on every number of every row, and removing "
             "that offset alone restores the token's movement to 29.9% "
             "(audit P012; the clip range is now a config field, default the "
             "[0, 1] that runs today).",
    },
    "node.enc.vision": {
        "t": "vision encoder",
        "w": "Turns the four eye pictures - {vision_patches} patches of 16x16 pixels, "
             "left and right eye rows alternating - into one [[token]]. A "
             "small Swin [[transformer]], {vision_width} numbers wide inside: "
             "[[attention]] inside windows of 7 "
             "patches, {vision_layers} layers deep with every second layer's windows shifted "
             "by half a window. The windows are cut from the patch LIST, not "
             "from the picture. {vision_reach}",
        "when": "Every [[model-step]] - vision is fresh on every one of them.",
        "h": "lem/model/encoders/swin_patch.py. No reach figure is typed "
             "here: schema.swin_dependency repeats the encoder's own window "
             "arithmetic for the layer count in the run's model receipt, and "
             "tests/test_build_explainer.py holds that arithmetic to the real "
             "encoder by perturbing one patch and reading which outputs "
             "move. Measured that way on 2026-09-19 on the 1,160-patch list: "
             "10 places at 2 layers, 17 at 4.",
    },
    "node.enc.tactile": {
        "t": "tactile encoder",
        "w": "Turns the up-to-32 tactile rows into one [[token]]: each row "
             "attends to the others, then one learned query pools them. "
             "Geometry is the identity - a row means 'where it is', never a "
             "numbered slot.",
        "when": "Every [[model-step]].",
        "h": "lem/model/encoders/cross_patch.py. Built for {run_name}: "
             "{tactile_depth} over the rows, {tactile_width} numbers wide "
             "inside, then the pooling query - counted from that run's model "
             "receipt. The --encoder-layers and --encoder-width launch flags "
             "set both.",
    },
    "node.enc.audio": {
        "t": "audio encoder",
        "w": "Two branches, one [[token]]. One branch reads the stereo "
             "spectrum (256 [[frequency-band|bands]] x 2 ears) with a stack of "
             "1-D convolutions along the frequency axis, and adds a 16-band "
             "summary of the same spectrum so absolute pitch survives "
             "(convolutions alone are pitch-blind - measured); that 16-band "
             "path starts at exactly zero weight, so it does nothing at all "
             "until training earns it. The other branch reads the two binaural "
             "numbers - the loudness difference between the ears and the "
             "arrival-time difference. A learned gate then mixes the two "
             "branches channel by channel, and the mix is a see-saw: every bit "
             "of a channel given to the two direction numbers is taken from "
             "the 512 spectrum numbers.",
        "when": "Every [[model-step]].",
        "h": "lem/model/encoders/conv1d_stack.py. Built for {run_name}: "
             "{audio_depth} in the spectrum branch, {audio_width} numbers "
             "wide inside - counted from that run's model receipt. "
             "--encoder-layers L builds 2 + L of them.",
    },
    "node.enc.proprioception": {
        "t": "proprioception encoder",
        "w": "Turns the 26 body-part rows into one [[token]]. Unlike the tactile and "
             "motor encoders, this one never compares one part with another: the 26 "
             "rows are laid end to end into one 312-number list and "
             "{proprio_mlp} reads the whole list at once. Which row "
             "a number came from survives only as the row-index number each "
             "row carries - now on a 0 to 1 scale like its neighbours; raw, "
             "that one channel was 99.16% of what this encoder saw.",
        "when": "Every [[model-step]].",
    },
    "node.enc.motor": {
        "t": "motor encoder",
        "w": "Turns the [[executed-command]] plus the achieved pose into one "
             "[[token]], so the model's own action enters the sequence like "
             "any other sense.",
        "when": "Every [[model-step]].",
        "h": "lem/model/encoders/cross_patch.py, the class the tactile encoder "
             "uses. Built for {run_name}: {motor_depth} over the handle rows, "
             "{motor_width} numbers wide inside, then the pooling query - "
             "counted from that run's model receipt.",
    },
    "node.enc.intermind": {
        "t": "intermind encoder",
        "w": "Turns the placeholder rows into one [[token]] with one "
             "self-[[attention]] layer and a learned pooling query. No lesson "
             "of its own touches it: its prediction weight is 0, and the "
             "anti-collapse and reconstruction lessons skip it. Its token "
             "still enters the shared core with every other token, so every "
             "OTHER sense's error trains this encoder too - measured, an error "
             "placed on the vision output alone leaves 4% as much "
             "[[gradient]] here as it leaves in the vision encoder (4.2 to "
             "4.5 per cent on three fresh starts).",
        "when": "Every [[model-step]] (on noise, for now).",
        "h": "lem/model/encoders/self_attention.py. Built for {run_name}: "
             "{intermind_depth}, {intermind_width} numbers wide inside - "
             "counted from that run's model receipt. The --encoder-layers and "
             "--encoder-width launch flags leave this encoder alone, so its "
             "inside width can differ from the token width it projects to.",
    },
    "node.ln_out": {
        "t": "shared output LayerNorm",
        "w": "One shared [[layernorm|normalization layer]] every encoder's "
             "token passes through. It centres the token and divides by the "
             "token's own spread, so every sense leaves at the same size and "
             "no sense can shout louder than the others.",
        "when": "Every [[model-step]], at the end of each encoder's own call - one shared module run {n_senses_word} times, not a stage that sees the {n_senses_word} tokens together.",
        "h": "lem/model/encoder_registry.py: nn.LayerNorm({d_model}, "
             "elementwise_affine=False) unless the run set "
             "encoder_defaults.output_layernorm_affine, so by default it holds "
             "zero learned numbers; a run that switched the gain and bias on "
             "shows their count at this box's corner and in the live rows "
             "below, and that run's tokens are not held at RMS 1.00. It "
             "pins a token at [[rms|RMS]] 1.00 only while that encoder's own "
             "spread (the standard deviation of its output before the norm) "
             "stays above about two hundredths; measured on the module itself, "
             "a spread of 1 gives a token 100% of full size, a spread of one "
             "hundredth gives 95%, one thousandth gives 30%, and an all-zero "
             "token gives 0%.",
        "f": "reading 'RMS 1' as unconditional. The one case where it stops "
             "holding is a collapsing encoder - exactly what the collapse "
             "chart watches for - so a small token is not always a small "
             "signal. Tokens produced any other way once reached the wellness "
             "scorer 1,500x off scale.",
    },
    "node.td_gate": {
        "t": "top-down gate",
        "w": "The one place where the model's own expectation changes what it "
             "perceives. Per sense it mixes the model's prediction of THIS "
             "model step into the encoder's [[token]]: the core reads (1 - g) of "
             "the encoder plus g of the guess. The page writes g as parts of "
             "the token out of 100: it runs from 0 in 100 (the core reads "
             "the encoder alone) to 50 in 100 (half the token is the model's own "
             "guess), and 50 is a hard ceiling - g is half a sigmoid, so at "
             "least half of every token is always the measurement.",
        "when": "Every [[model-step]], between the shared LayerNorm and the token "
                "assembly, for the {n_heads_word} senses that have a prediction head, "
                "and only while that sense arrived [[source-label|real]] this "
                "step.",
        "h": "mimo_bridge/lem_model.py: g = half of sigmoid(td_gates[sense]"
             "([surprise, 1])), then token = token + g x (prediction - token). "
             "g is forced to 0 whenever the fast error average rises above "
             "twice its slow baseline. Click the box for this snapshot's blend "
             "per sense, solved from the raw and fed tokens the page carries. "
             "Measured over the 19 of this run's 20 snapshots that carry a "
             "gate reading: audio's blend ran 10 to 28 parts in 100 against the "
             "ceiling of 50, so between a tenth and just over a quarter of the "
             "audio token the core read was the model's own guess; vision, "
             "tactile and proprioception stayed at 0 in 100 on every one of the 19.",
        "f": "grading any lesson against the gated token. An error measured "
             "against a target that already holds g of the prediction shrinks "
             "by (1 - g) x (1 - g) with nothing learned. Every lesson reads the "
             "RAW token (tokens.raw); only behaviour reads this one.",
    },
    "node.assembly": {
        "t": "token assembly",
        "w": "Lines the tokens up for the [[transformer]]: this frame's {n_senses_word} tokens plus the last {context_minus_1} remembered frames, one [[token]] per "
             "sense per [[model-step]] - {seq_len} tokens covering {window_s} s of body time "
             "at this run's settings. Two learned tags are added to each token "
             "saying WHICH sense it is and WHICH model step - the transformer's "
             "only way to tell them apart.",
        "when": "Every [[model-step]].",
        "h": "the model step tag is the token's PLACE IN THE WINDOW, not a clock "
             "reading: the newest slot always means now and the oldest always "
             "means {window_s} s ago, so every token's tag changes as the window "
             "slides. The sense tag is stamped once, while the frame is "
             "current, then frozen in the buffer with it.",
        "f": "after an episode reset or a body rebuild the window refills from "
             "zeros, so for the next {context_minus_1} steps ({window_s} s of body time) up to {seq_older} "
             "of the {seq_len} slots are empty tokens. And a caller that supplies no "
             "remembered frames at all gets the current model step copied into all "
             "{context_length} slots - a past made identical to the present; this run's "
             "trainer always supplies the real buffer.",
    },
    "node.core": {
        "t": "the core (the transformer blocks) and its two size figures",
        "w": "Each dashed frame holds ONE block, with weights of its own, named as the defect register and the "
             "code name it (`blocks.0` is block 0, the 1st of {n_blocks}). Block 0 is drawn at full size: its "
             "three [[attention]] stages and its feed-forward network are boxes. Each later block is drawn as "
             "one row of four cells - stage 1, stage 2, stage 3 and FFN, the same four parts as block 0 with "
             "their own weights. The blocks run in series: each block's output is the next block's input. The "
             "model diagram carries two figures and they count different things: the one ON a frame is that "
             "block alone, the one UNDER all the frames is the whole core.",
        "when": "Every [[model-step]], more than once: on the live pass; on the "
                "delayed next-frame lesson's pass over the PREVIOUS model step with "
                "the command the body executed; once per depth of the imagination "
                "chain; on a planning model step inside each of the planner's 20 "
                "rounds; with --causal-chain on, once for the chosen plan's four "
                "alternatives and up to --motor-hold times for the prediction "
                "made before each latched command is sent; and in each re-study "
                "pass.",
        "h": "this run's checkpoint. ON each frame: {core_block_learned} learned numbers - "
             "{core_stage_learned} for each of the three attention stages and {core_ffn_learned} for "
             "the feed-forward network, so the four corner counts inside block 0's "
             "frame add up to its figure (the cells of the later blocks carry no corner count). UNDER all the frames: "
             "{core_saved} saved numbers - the {n_blocks} blocks' {core_blocks_learned}, the {core_tag_tables} of "
             "the two learned tag tables (which sense, which [[model-step]]), the "
             "{core_final_norm} of the final [[layernorm|LayerNorm]], and {core_mask_cells} stored but "
             "never learned, which are the true/false cells of the {n_blocks} causal "
             "masks that stop stage 2 reading the future.",
        "f": "reading the whole-core figure as one block's size, or as "
             "learned weights. Take the masks out and the core holds {core_learned} "
             "learned numbers, which is the count the final-LayerNorm popup "
             "measures its {core_layernorm_numbers} gain-and-bias numbers against",
    },
    "node.block_s1": {
        "t": "stage 1 - senses read each other",
        "w": "At each [[model-step]] separately, the senses read each other with "
             "[[attention]]: tactile can check what vision saw at the same "
             "instant. First of the three stages inside a block; the arrows "
             "show the sequence.",
        "when": "Every [[model-step]], inside each of the blocks in turn.",
        "h": "each block's stages run on the same {seq_len} tokens; the block count "
             "drawn on the dashed group comes from the run's own model "
             "receipt. The block count is the --core-blocks launch setting ({n_blocks} in {run_name}); a launch that leaves it at 0 builds the 2-block default of LITE_OVERRIDES in mimo_bridge/lem_model.py.",
        "f": "'at each model step separately' says where the attention may LOOK, "
             "not what the tokens carry. From block 1 (the 2nd) onward the {n_senses_word} tokens it lines up have already had other model steps mixed into them by block "
             "0's unmasked stage 3.",
    },
    "node.block_s2": {
        "t": "stage 2 - each sense reads its past",
        "w": "Each sense reads its own earlier [[model-step|model steps]], under a "
             "hard rule that nothing may read the future. This is where "
             "short-term memory lives, and it is the only masked stage of the "
             "three.",
        "when": "Every [[model-step]], after stage 1.",
        "h": "a fixed {context_length}x{context_length} true/false mask blocks every column later than "
             "its row ({mask_blocked_one} blocked cells of the {mask_cells_one}). Measured on this "
             "run's trained weights, the forbidden half of the map holds "
             "exactly 0 of every 100 parts of every row's budget.",
        "f": "the rule says where this stage may LOOK, not what the tokens "
             "carry: from block 1 (the 2nd) onward a row forbidden to attend to later "
             "model steps is reading tokens that block 0's unmasked stage 3 has "
             "already mixed later model steps into.",
    },
    "node.block_s3": {
        "t": "stage 3 - everything reads everything",
        "w": "Every [[token]] may read every other: any sense at any "
             "[[model-step]] in the window, earlier or later. This stage carries "
             "no mask at all - stage 2 is the only masked one.",
        "when": "Every [[model-step]], after stage 2.",
        "h": "full attention across all {seq_len} tokens at once, with no mask "
             "argument at all (lem/model/forward_model.py). Measured on this "
             "run's trained weights: 65 of every 100 units of this stage's "
             "looking budget land on model steps LATER than the token doing the "
             "looking, where 0 of 100 would be a strictly backward-looking "
             "stage.",
        "f": "reading that forward-in-time weight as the model seeing its own "
             "future. The window ENDS at now, and every output reads only the newest model step's {n_senses_word} tokens, so there is no later model step inside "
             "the window to leak.",
    },
    "node.block_ffn": {
        "t": "feed-forward network",
        "w": "A two-layer [[mlp]] applied to each token on its own, after the "
             "attention stages. Attention gathers information; this layer "
             "processes it.",
        "when": "Every [[model-step]], closing each block; then the next block "
                "repeats all four parts.",
        "h": "{d_model} numbers in, widened to {ffn_width}, a GELU bend, then back to {d_model} - "
             "this run's ffn_expansion is {ffn_expansion}",
        "f": "reading any of the four parts as a replacement. Each one is a "
             "residual step: its output is ADDED to the token that entered it, "
             "so nothing the token already carried is lost.",
    },
    "node.final_norm": {
        "t": "final LayerNorm",
        "w": "The core's last [[layernorm|normalization layer]]. Unlike the "
             "shared one at the front it has a learned gain and a learned bias "
             "({core_final_norm} numbers), so it does NOT hold tokens at a fixed size.",
        "when": "Every [[model-step]], after the last block.",
        "h": "nn.LayerNorm({d_model}) with the default learned scale. With "
             "persistent memory on - this run - the memory's [{n_senses} x {d_model}] bias is added to the newest {n_senses_word} tokens AFTER this layer, so what the "
             "heads, decoders, motor controller, wellness and voice head read "
             "is this layer's output plus that bias.",
        "f": "treating it as the same type of layer as the shared LayerNorm at "
             "the front. This run's core holds {core_layernorms} LayerNorms with {core_layernorm_numbers} gain-and-bias numbers, {core_layernorm_pct} of its {core_learned} learned numbers. On the 2-block, 128-wide core every run built before the size became a launch setting (2026-09-17), the nine LayerNorm gains and biases - 2,304 of "
             "its 679,168 learned numbers, 0.3% - were measured to carry the "
             "whole drift toward copying the present: moving just those 2,304 "
             "numbers from the hour-8 model onto the hour-1 model reproduced "
             "159% of the drift, where 0% would be none of it.",
    },
    "node.heads": {
        "t": "prediction heads",
        "w": "One small readout for every sense EXCEPT motor - {n_heads_word} in {run_name}: {head_list}. Each outputs what the NEXT "
             "model step's [[token]] should be. Motor has no head on purpose: the "
             "command is an input the model was just handed, so a motor head would "
             "score well by copying it and leave nothing to show what the action "
             "explained. Their error against what really arrives - knowing the "
             "[[executed-command]] - is [[causal_pe]], the model's headline error "
             "and its main lesson. Only senses labelled real are graded: {graded_sentence}",
        "when": "Every [[model-step]]: a fresh guess of next, and the grading of "
                "the previous guess against what just arrived.",
    },
    # THE TWO REPAIR PATHS DRAWN INSIDE THEIR HOST BOXES and the liveness repair the model diagram lists as not drawn
    # (the decided design of the Architecture tab, Oct 7, 2026, D-E4). Each entry opens from a line of the model
    # diagram (viewer.js HOSTED_DOC and the not-drawn list's DOCKEY), is written as input, computation and output, and
    # is keyed outside "node.*": every "node.*" key is a box the component manifest must map.
    "path.motor_asked_change": {
        "t": "the motor encoder's asked-change path (MOTOR_ENCODER_REPAIR)",
        "w": "A repair path built inside the motor encoder when a run is launched with --motor-asked-change on "
             "(the repair MOTOR_ENCODER_REPAIR). The encoder receives each body handle's achieved pose and its "
             "command, but what the body's servo acts on is the change the command asks for: the command minus "
             "the achieved pose, times the handle's strength. Measured on checkpoint 455,800 of H84, each of the "
             "encoder's 4 pooling heads put 99.75 to 99.97 of every 100 parts of its weight on one handle, so the "
             "pooled vector read at most 4 of the 26 handles. This path hands every handle's asked change to the "
             "[[token]] through weights of its own; no attention weight chooses which handle is read. The line in "
             "the motor encoder's box gives the path's count of learned numbers from the run's receipt; the box's "
             "corner count is the encoder's own module.",
        "when": "At every call of the motor encoder (one forward hook): the live pass, the delayed "
                "next-model-step lesson, re-study in replay, the imagination chain and the planner's candidate "
                "commands. On the live pass the 9 command numbers are zeros (no command is chosen yet), so the "
                "path's output is exactly 0 there and the live token is the token of a build without the path.",
        "h": "lem/model/motor_asked_change.py.\n"
             "- INPUT: the motor encoder's own input rows, 19 numbers per handle: the achieved pose (10) and the "
             "executed command (9), positions divided by 200 cm.\n"
             "- COMPUTATION, per handle: 10 features - the asked position change (command position minus achieved "
             "position, clamped at 40 cm per axis), the asked turn (the rotation that carries the achieved "
             "orientation to the commanded one), the strength, the radius, a 1 when the handle's row asks "
             "something (a command, a pose and a strength above 0) and a 1 on the eyes' row; a small network "
             "shared by the handles (10 numbers in, GELU) adds more numbers; every handle's numbers enter one "
             "output projection through columns of their own, and the sum is divided by the square root of the "
             "count of numbers the projection reads. The projection starts at zero.\n"
             "- OUTPUT: the encoder's pooled vector v becomes v + s x u, s the standard deviation of v over its "
             "{d_model} numbers (no gradient), u the path's output, summed in float32 before the shared output "
             "LayerNorm: one unit of u is one unit of the token. With the projection at zero the token is bit for "
             "bit the token without the path.",
        "f": "reading the path as a lesson of the live token: it learns only from rows whose pose and command "
             "belong to the same [[model-step]] - the delayed next-model-step lesson, its re-study in replay and "
             "the hold loss - and the token the motor controller reads on the live pass carries 0 from it.",
    },
    "path.heads_all_slot_path": {
        "t": "the prediction heads' all-slot path (PREDICTOR_REPAIR)",
        "w": "A repair path built beside every prediction head's attention when a run is launched with "
             "--heads-all-slot-path on (the repair PREDICTOR_REPAIR). A head reads the 6 newest tokens, one per "
             "sense slot, through one learned query and a softmax. On H84 each head put all of its weight on one "
             "slot: the persistent memory's bias, the same vector at every model step, is 20 to 50 times the "
             "tokens' own size on four slots, and a softmax at full weight passes no [[gradient]] to its query "
             "and keys. The path reads every slot through weights of its own; no attention weight chooses which "
             "slot is read. The line in the prediction heads' box gives the path's count of learned numbers from "
             "the run's receipt; the box's corner count is the heads' own module.",
        "when": "At every call of the prediction heads: the live pass, the delayed next-model-step lesson, "
                "re-study in replay and the imagination chain. The planner's search runs it with its weights "
                "frozen.",
        "h": "lem/model/heads_all_slot_path.py.\n"
             "- INPUT: the tensor every head is called with - the transformer's final LayerNorm output at the "
             "newest model step plus the memory's bias, 6 slots x {d_model} numbers - and the memory's constant "
             "c, the part of that bias every body and model step shares, read from the memory at each call "
             "without gradient.\n"
             "- COMPUTATION: x = the input minus c, in float32; x minus a running mean of x per slot, taken over "
             "the calls that train the heads; per slot and per head a LayerNorm and one matrix shared by the 6 "
             "slots (32 numbers out per slot on H84); the 6 results joined, a hidden layer with GELU and an "
             "output layer that starts at zero.\n"
             "- OUTPUT: added to each head's own prediction, in float32. With the output layer at zero the "
             "prediction is bit for bit the prediction without the path; the old head keeps its weights and its "
             "attention.",
        "f": "reading the old head's attention as what the head reads: the old pool still reads one slot, and the "
             "head's output is the pool and the path together.",
    },
    "part.liveness_repair": {
        "t": "the liveness repair (LIVENESS_REPAIR)",
        "w": "Learned numbers a run builds with --liveness-repair on to bring units that stopped responding back "
             "into use. A hidden unit counts as live when its GELU passes a signal forward on some sample; a unit "
             "whose input never reaches the liveness sweep's line (`LIVE_LINE` in lem/model/liveness.py) passes "
             "almost nothing forward and almost no [[gradient]] back. The repair's numbers sit in 15 layers of "
             "other parts and beside two encoders' pools, so the model diagram draws no box for them and lists "
             "their count among the parts it does not draw.",
        "when": "A unit is revived between two training steps when its input did not reach that line over a "
                "window of 800 input rows of its layer; with --liveness-rule none no unit is revived and the "
                "units revived before keep their routing. The two pool paths run at every call of the motor and "
                "tactile encoders.",
        "h": "lem/model/liveness.py.\n"
             "- REVIVED ROWS in 15 layers: the 9 the sweep of Sep 22, 2026 found with fewer than half their units "
             "live (the first layer of the command-to-outcome pose head, two layers of the wellness predictor, the "
             "motor controller's first residual block, four reconstruction decoders and the persistent memory's "
             "outcome layer) and 6 added from segment 32's reports (the vision, tactile and vestibular prediction "
             "heads' first layers, the vestibular reconstruction decoder, the motor controller's confidence head "
             "and the contact head's first layer). A revived unit's outgoing weights are set to zero after their "
             "mean contribution is moved into the next layer's bias; from then on the unit reads the layer's "
             "input centred by a running mean and LayerNormed, through fresh weights.\n"
             "- AN ALL-ELEMENT PATH beside the motor encoder's pool and the tactile encoder's pool: every handle "
             "(every contact row) enters the token through weights of its own; the path's output layer starts at "
             "zero.",
        "f": "reading its count as the size of one part: it is the sum of the fresh weights and LayerNorms of "
             "every revived row in those 15 layers and of the two pool paths.",
    },
    "node.tenure_heads": {
        "t": "command-to-outcome heads (tenure heads)",
        "w": "Two small networks that ask, at each command's end, whether the command "
             "explains what followed. One reads the core's output and predicts the "
             "body's pose change over that command's life. The other turns the "
             "[[executed-command]] itself into a sketch of the token change it "
             "should have caused, and that sketch must pick its OWN outcome out of "
             "a lineup of decoy outcomes; the share it gets right is rank1, the "
             "first number on this page.",
        "when": "Only when a command ends - one command is held for several "
                "[[model-step|model steps]] - so in a run that grades no command-ending "
                "event these weights receive no gradient at all.",
        "h": "lem_model.py: the pose head is {token_row} numbers in, 256 hidden, 3 numbers "
             "out per [[handle]]; the command projection is 9 numbers per handle in, "
             "256 hidden, {order_proj_out} out (vision, tactile and proprioception, {d_model} each). "
             "Their scores are the run records' tenure_pe, tenure_gap and "
             "tenure_tok - the run tab's first two charts.",
        "f": "presence is not learning: these weights are saved and registered in "
             "the optimizer whether or not one command-ending event was ever graded. "
             "Read tenure_events before reading rank1.",
    },
    "node.recon": {
        "t": "reconstruction decoders",
        "w": "{n_heads_word_cap} small readouts - one per sense except motor - that "
             "re-produce the CURRENT frame from the transformer's output, a "
             "guard on whether the token still holds the present. Each reads "
             "only ITS OWN sense's {d_model} numbers, and they never mix with "
             "prediction. Read their scores against the same two ends: the "
             "[[encoder]] tokens are held at [[rms|RMS]] 1, so the page writes "
             "each score as a percentage of the error a decoder that always "
             "answered with zeros would make: 100% = no better than zeros, 0% = "
             "perfect. Measured over this probe's 20 snapshots, proprioception ran "
             "90%, tactile 140%, vision 211% and audio 1,458% - three of the "
             "four graded senses re-describe the present worse than a decoder "
             "that says nothing at all.",
        "when": "Every [[model-step]] during training (their lesson weight "
                "follows a schedule, with a floor at 2% of full weight).",
        "h": "one decoder is {d_model} numbers in, {recon_hidden} hidden, {d_model} out; {n_heads_word} of them hold {recon_params} weights. The target is that sense's raw encoder "
             "token, detached",
    },
    "node.motor": {
        "t": "motor controller",
        "w": "An [[mlp]] that reads the transformer's pooled output, the "
             "body's current [[handle]] state, and the [[gradient]] of "
             "[[wellness]] with respect to that current pose written in the "
             "shape of a command - fed as an extra INPUT - and proposes 9 "
             "numbers per handle: a target position, a target orientation "
             "([[quaternion]]), a radius (the body reads it nowhere) and a "
             "strength 0-1. The position numbers are NOT centimetres here. "
             "They are held inside -1 to +1, the whole range the body can be "
             "asked for: -1 and +1 are 200 cm either side of the WORLD's origin "
             "on each axis - the same world frame the achieved pose is measured "
             "in, not a frame centred on the body, whose own middle sat 12 to 25 "
             "cm from that origin across the H81 probe's 12 embedded snapshots - "
             "and they become centimetres only downstream - see the line into "
             "the world. The network's inner number is unbounded "
             "and is reported beside the command as an excursion, which the "
             "motor lesson charges with a bounded cost. Before this bound "
             "(September 19, 2026) the inner number WAS the proposal: across "
             "the H81 probe's 20 snapshots it ran from -472 to +568, 791 of "
             "those 1,560 numbers lay outside -1..1, and every one was "
             "pinned to the edge before the body saw it, so the body's "
             "commands did not change when the bound arrived.",
        "when": "Every [[model-step]]. Its proposal reaches the body only on "
                "command-boundary steps ([[motor-hold]]), and only when no plan "
                "was made that step. {controller_authorship} "
                "{exec_source_tally} On the H81 probe, launched with both "
                "flags at 5 so that the planner wrote every executed command, "
                "the plan that replaced this controller's proposal closed "
                "0.0% of its own distance "
                "to its goal on 16 of the 19 snapshots that carried a plan "
                "(0% = never moved toward it, 100% = arrived), and 1.7%, 5.1% "
                "and 5.3% on the other three.",
        "h": "lem/model/motor_controller.py; think-harder mode adds thinking "
             "depth when its confidence is low",
    },
    "node.gaze": {
        "t": "gaze",
        "w": "Where to look. The eyes are the gaze [[handle]]: its orientation "
             "is "
             "read as one azimuth+elevation pair, and that single command "
             "turns BOTH eyes together (the right eye's horizontal axis is "
             "mirrored). No separate gaze network is in use - the same "
             "controller and "
             "planner move the eyes. {gaze_head_use} Two body-side "
             "circuits then act ON that issued command, in this sequence: a "
             "spontaneous-saccade generator (on in this run) adds a wandering "
             "offset, re-drawn every 0.3 s to 0.6 s with a median size of 5 "
             "degrees and a 30-degree cap; then, while a snap is live, a "
             "peripheral-motion reflex pulls the command 70% of the way to "
             "its target. The reflex watches the wide 60-degree field and a "
             "coarse 150-degree ring around it, with the central 24 degrees - "
             "the fovea's own territory - masked out of detection. Neither "
             "circuit acts before the command is issued; both rewrite the issued "
             "command.",
        "when": "Every [[model-step]]: the limbs hold one command for {motor_hold} "
                "model steps ({hold_ms} ms of body time in {run_name}), "
                "the eyes do not (--goal-gaze-hold is off in this run; with "
                "it on the planner's target holds for a whole action "
                "interval).",
        "h": "_gaze_from_handle in mimo_bridge/lem_model.py reads the gaze "
             "handle's orientation out of the motor controller's proposal; "
             "the GazeHead (lem/model/prediction_heads.py) never sets the "
             "gaze - it is loaded and, unless --efference-heads on reuses its "
             "layers as the efference gaze head (lem/model/efference.py), "
             "never called",
    },
    "node.wellness": {
        "t": "Wellness calculations and their uses",
        "w": "The wellness box contains fixed score formulas and a learned predictor of five scores. [[wellness|Read the three distinct uses of wellness.]]\n\n**Fixed scores, per body:** [[eval-prediction-error|prediction accuracy]], [[eval-tactile-comfort|tactile comfort]], [[eval-capability|capacity]], [[eval-coherence|co-movement]], [[eval-motor-state|position boundary score]], and [[eval-self-touch|self-contact increase]]. Each link gives the actual inputs, calculation and scale.\n\n**Learned prediction:** current encoded readings train a network to reproduce five fixed scores. Predicted encoded readings let that network estimate the same five scores for a candidate command.\n\n**Connections:** the fixed sum supplies a [[wellness-gradient|position derivative]] to the motor controller. Enabled channels rank [[wellness-ranking|remembered goals]]. Predicted scores enter command search only when prospective wellness is enabled with a positive weight.",
        "h": "lem/model/wellness.py: forward and score_predicted_wellness. lem/training/loss_manager.py: _compute_wellness_loss. mimo_bridge/lem_model.py: step and _plan_chain.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The learned predictor does not predict the full goal-ranking score. [[planner-blind|Which channels it cannot predict.]]",
        "when": "Fixed scores are calculated each model step. The learned predictor changes its weights during training through its matching loss; evaluation-only runs keep those weights fixed."
    },
    "node.curiosity": {
        "t": "Curiosity: a preference for a configured prediction-error range",
        "w": "Input: [[causal_pe|prediction error]], averaged across the batch. Calculation: a triangle over the configured error range. The drive is zero outside the range and at either edge, rises to one at the midpoint, then falls to zero. The sub-diagram shows this run's range.\n\nWhen curiosity-goal is on, [[curiosity-mode|the ranking mode]] determines how this drive changes remembered-goal value. Direct and forward wellness modes can also scale the wellness lesson by it. The fixed wellness formulas do not read it.",
        "h": "lem/model/curiosity.py: CuriositySystem. mimo_bridge/lem_model.py: goal-value assembly and _wellness_channel_vector.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The drive is a configured function of error, not a test of learnability. It cannot establish whether repeating a command will improve prediction. [[learning-progress|The separate prediction-error trend]] measures something different.",
        "when": "Calculated every model step. Enabled goal ranking reads it when storing a new model step; direct and forward modes also use it in the wellness lesson."
    },
    "node.intention": {'t': 'intention planner', 'w': "The planner computes one [[candidate-command|command]] per body every 10 model steps (one model step = one 50 ms model update). A command is 26 body handles x 9 numbers: target position x, y, z (in units of 200 cm from the world origin), target orientation (a [[quaternion]], 4 numbers), radius (0 to 1) and strength (0 to 1, a share of full strength). Under --hand-target-mode wrist_relative_v1, as B1K_skip13_492800 was launched, the planner also sets a target angle for every finger joint, and the finger handles' rows of the command are computed from those angles.\n\n**Input 1, this model step's tokens.** One [[token]] per sense, 192 numbers each: that sense's [[encoder]] output for this model step's sensor readings, blended under --top-down-gate on (as B1K_skip13_492800 was launched) with up to half of the model's own prediction of this model step: token + g x (prediction - token), g from 0 to one half. With them go the tokens of the previous 127 model steps (the [[forward-model|forward model]]'s context, 128 model steps with this one) and the body's measured handle state (26 handles x 10 numbers), which the motor encoder reads beside the command. Under --persistent-memory on the persistent memory's bias is added to the core's newest output tokens before the prediction heads read them.\n\n**Input 2, the goal.** The [[goal-ring]] holds the last 200 model steps. Each [[goal-ring-entry|entry]] holds that model step's tokens and its wellness values: a stored value - the sum of the six wellness formulas, the movement score, the agency payment under --agency-drive on, the consequence term under a --consequence-w above 0 (that weight once for each consequence the world measured at that model step) and the reader's value under --entity-self with a --self-value-weight above 0; under --curiosity-goal on the value is then multiplied by (1 + the curiosity drive) under --curiosity-mode multiply, or raised by the drive x its table weight under additive - and, under --wellness-rank normalized, one value per wellness channel. Each entry gets one ranking number. Under --wellness-rank raw it is the stored value. Under --wellness-rank normalized it is the sum over the channels of weight x (the entry's channel value - the channel's running mean) / max(the channel's running spread, its measured floor) ([[wellness-ranking|the ranking]]): the entry stores each channel as that model step's value minus the body's running average of the channel over about the previous 10 model steps; the running mean and spread are taken over the stored values of about the last 2,000 model steps, all bodies together; controllability's weight is 0 while its evidence gate is shut; and under --curiosity-goal on the curiosity drive (0 to 1, the same number for every body) is one of the channels, at its own weight, whatever --curiosity-mode says. An entry whose harm reading (the larger of skin pain and 1 - motor state) is over 92% of the way from no modelled aversion to an extreme load cannot be the goal while any entry of that body is under that line; when every one is over it, the least harmful is. The goal = the tokens of the highest-ranked entry, one entry per body. B1K_skip13_492800 was launched with --wellness-rank normalized. B1K_skip13_492800 was launched with --curiosity-mode additive: the drive is added.\n\n**Input 3, the starting command.** The motor controller's command at this model step, one per body.\n\n**Computation, 20 times** (lem/model/intention_generator.py generate_action):\n\n1. The [[execution-filter|execution filter]] converts the command into the command the body would be given (LemLite.planned_execution): each handle's step from where it is now is clamped to 40 cm x the growth phase's reach limit on each axis and its strength to the phase's strength limit; position, orientation and strength are blended 35 parts this command with 65 parts the previous model step's blended command (--smooth-source proposal), B1K_skip13_492800's --smooth-actions 0.35; the eyes' row and the radius are not blended; and the positions are divided by 200, the motor encoder's units.\n2. The motor encoder turns that command and the measured handle state into the motor token, and the forward model runs once on this model step's tokens with that motor token and the context (lem/model/predictive_state.py transition): the [[predicted-outcome|predicted tokens]] of the next model step, 192 numbers for every sense except motor; under --efference-heads on the gaze copy adds the eye command's part to the vision prediction.\n3. The [[intention-distance|score]] = for every sense measured at this model step except motor (a stale or synthetic sense is left out), the mean over its 192 numbers of (predicted - goal)^2, added over those senses; plus one hundredth of the sum over the 26 handles of strength^2. B1K_skip13_492800 was launched with --planner-effort-per-body on: the strength part is averaged over the bodies, like the goal part. \n4. Backpropagation through the frozen forward model gives the derivative of the score with respect to each command number (the [[gradient]]).\n5. One [[adam|Adam]] step, step size one tenth, changes every command number and every finger joint angle against that derivative.\n\nThe model's weights do not change: they take no gradient during the search, and their settings are restored after it. Of the 21 commands scored (the starting command and the 20 updates) the lowest-scoring is kept, per body.\n\n**After the search.** Under --hand-goals memory, as B1K_skip13_492800 was launched, a goal entry whose recorded hand pose is valid gets a second search over the finger joint targets only ([[hand-goal|the finger search]]): 20 Adam steps of step size one tenth, each angle held within 11 degrees of the first search's, scored by how far the finger poses a fitted hand-response model predicts 250 ms ahead lie from the hand pose recorded with the entry; of the 21 scored the lowest is kept. The forward model then runs once more on the kept command; if the size ([[rms|RMS]]) of any sense's predicted tokens is below half or above twice the size of that sense's tokens at this model step ([[scale-check|the size check]]), the whole plan is dropped and no planner command is sent. Every plan is one command long under --hand-target-mode wrist_relative_v1, as B1K_skip13_492800 was launched, whatever depth is licensed.\n\n**Output.** The plan's first command replaces the motor controller's command, goes through the same execution filter on its way to the body, and reaches the body only on a model step where a new command is latched: every 5 model steps, held for 250 ms ([[motor-hold]]). With no plan kept the body gets the motor controller's command, with [[babble]] noise when the noise is on.", 'when': "On a model step whose number is a multiple of 10 - the --intention-every flag; 10 is the value B1K_skip13_492800 was launched with - once the goal ring holds at least 20 entries. A new command is latched for the body every 5 model steps ([[motor-hold]]) and held for 250 ms of body time, so a plan made on a model step that is not a latch step is computed and never reaches the body. B1K_skip13_492800 was launched with --intention-every 10 and --motor-hold 5: every plan lands on a latch step and its first command becomes the executed command there. Of every 2 latched commands 1 falls on a planner step and 1 does not; on that one the body receives the motor controller's command, with [[babble]] noise added whenever the noise is on at that model step. Counted by the trainer at each latch and read from the stats log of B1K_skip13_492800 (13 report windows, steps 492,900 to 494,025): the planner wrote 51 of every 100 (50 to 60 across the windows); the motor controller with babble noise added wrote 30 of every 100 (20 to 45 across the windows); the motor controller with no noise added (the noise was 0 at that model step: a rest window of the babble-bout schedule, or a launch with babble off) wrote 19 of every 100 (5 to 30 across the windows). 0 of every 100 would mean that author never wrote an executed command, 100 that it wrote them all. B1K_skip13_492800 was launched with --causal-chain on, so the trainer's causal-chain recorder stamps each plan's reached_body at the latch: true when the plan's first command became the body's command at that model step, false when it was computed and never sent.", 'h': "lem/model/intention_generator.py generate_action runs the 20 updates and keeps the lowest-scoring command per body; mimo_bridge/lem_model.py LemLite.step ranks the goal ring (lem/model/wellness_channels.py rank_values under --wellness-rank normalized) and calls LemLite._plan_chain, which forms every command with LemLite.planned_execution, runs the size check (lem/model/predictive_state.py scale_report) and, under --hand-goals memory, the finger search (lem/model/hand_response.py choose_hand_goal). All of it runs inside model.step() in the trainer process, on the GPU. A depth beyond the first is licensed one at a time: depth k + 1 is open while the imagination chain's running average of wins against predicting no change at depth k stays above 50 in 100 (0 = it never wins, 100 = it always does), and the first depth to fall to 50 or below closes every depth above it. The cap is --imagination-steps, 3 in B1K_skip13_492800. A plan's later depths start from the predicted tokens but carry the body pose from BEFORE the plan - a predicted proprioception token cannot be turned back into handle positions - so their pose is 1 to 2 model steps old. B1K_skip13_492800 was launched with --hand-target-mode wrist_relative_v1, so every plan it makes is one command long, whatever depth is licensed. B1K_skip13_492800 was launched with --hand-goals memory, so its plans set the finger targets by the second search, toward the hand pose recorded with the goal entry.", 'f': "- The kept command is the one with the lowest predicted SCORE, and the score has two parts: a command can score lower by carrying less strength while its predicted tokens come no nearer the goal tokens. The planner panel's live values give both parts, for the starting command and the kept one, of the newest planning event.\n- 'Frozen' is not a second, older model: the search runs the very weights the run is training, with their gradient switched off for the length of the search.\n- The plan record's distances (this model step's tokens against the goal tokens, and the predicted distance after each command but the last) are not the score: they leave out the strength part, and a one-command plan records no predicted distance at all.\n- Under --wellness-rank normalized the goal is the entry ranked highest against the channels' running means, which need not be the entry with the highest stored value.\n- 'No plan at this model step' is normal: the planner runs on 1 model step in 10, and a plan the size check drops leaves no plan record.", 'plain': 'The planner in the inspected MIMo bridge chooses a movement by comparing predictions. It has no network weights of its own. It uses the learned encoders, forward model and prediction heads.\n\n1. Start with the motor controller’s command and a remembered goal chosen using wellness.\n2. Predict the result of that command. Compare the prediction with the goal.\n3. Adjust the command numbers to reduce the difference, then predict again. The network weights stay fixed during this search.\n4. Pass the selected command to the movement system. The command must pass the existing checks and reach a command-update time before the body uses it.\n\nWellness helps choose the remembered goal. The ordinary search tries to match that goal. A separate prospective-wellness option can also rate predicted results by wellness; the recorded setting below says whether that option is enabled.\n\nThe wider UEMA design also specifies a learned CrossSimulationAttention module for comparing imagined sequences (section 10.4). That module exists in source but is not used by this bridge path. This box describes the bridge search, not every possible UEMA intention mechanism.'},
    "node.imagination": {
        "t": "imagination",
        "w": "A chain run inside the live step: take the oldest [[model-step]] the "
             "chain keeps ({imagination_back}), predict the next one, "
             "feed that prediction back in as if it were a real [[token]], "
             "predict again - once per depth - and grade every depth against "
             "the RAW tokens that really arrived. Only the senses measured "
             "at that model step are graded and fed back; one that was not "
             "measured is put back as it was observed, so its noise cannot "
             "compound. It reads no stored ring of old model steps - that is the "
             "replay memory box, a different mechanism.",
        "when": "Every [[model-step]] once enough history exists. No command leaves "
                "this chain, so it puts no force on the body itself. "
                "{imagination_depth_use} Here the [[forward-model|"
                "forward model]] is the thing being TRAINED - not run frozen "
                "the way the planner runs it - and the command fed in at each "
                "depth is the one the body really executed, read back from "
                "the ledger, not a candidate being improved.",
        "h": "the chain itself in lem/live_test.py (_run_imagination_compiled: "
             "the feed-back, the re-anchoring, the grading); one hop in "
             "lem/model/predictive_state.py (transition: the motor token, the "
             "full transformer pass, the shared [[layernorm|LayerNorm]] that "
             "lets a prediction re-enter as a token); the stored histories, "
             "the call and the depth licence in mimo_bridge/lem_model.py - "
             "trainer process",
        "f": "the body does not move through the chain: every depth carries "
             "the real measured pose from the chain's first frame inside its "
             "command token, because a predicted proprioception token cannot be "
             "turned back into handle positions. And its error reaches the "
             "forward model, the prediction heads and the motor [[encoder]] "
             "only - never the {n_heads_word} sense encoders, because it reads stored tokens that were detached when they were stored",
    },
    "node.replay": {
        "t": "replay memory",
        "w": "Every few live [[model-step|model steps]] the model re-studies a batch "
             "of remembered windows: the [[forward-model]], the prediction "
             "heads, the motor controller and the persistent memory's bias "
             "readout learn again from stored experience. {n_heads_word_cap} of the {n_senses_word} [[encoder|encoders]] - {non_motor_list} - are excluded, because their tokens were stored "
             "detached. The MOTOR encoder is not excluded: re-study rebuilds "
             "each window's motor [[token]] from the achieved pose plus the "
             "[[executed-command]], so that encoder's {enc_motor_params} weights - "
             "{enc_motor_pct} of this model's {total_params} - are trained from memory too.",
        "when": "One pass every --replay-every live [[model-step|model steps]] ({replay_every} in "
                "this run), over up to --replay-batch remembered windows "
                "({replay_batch} in this run). The batch size is a ceiling, not a "
                "count: a window that would cross an episode reset is thrown "
                "away, and a pass that collects fewer than {replay_floor} usable windows "
                "is skipped and leaves no record at all. Under "
                "--fill-ring-first the pass waits "
                "until max(4 x batch, 8 x context) stored model steps carry the "
                "command that left them - {replay_need} model steps, about {replay_need_s} s of body "
                "time, at a batch of {replay_batch} and a context of {context_length}. An --eval-only "
                "run never builds the ring and takes 0 re-study passes; the "
                "box on the diagram says which case this run is.",
        "h": "mimo_bridge/replay.py holds the ring in CPU memory; "
             "LemLite.replay_update in mimo_bridge/lem_model.py does the "
             "re-study, called from the trainer's own loop in train_mimo.py "
             "AFTER the live step of the same [[model-step]] - a second optimizer "
             "step inside one model step, not part of model.step(), which only "
             "WRITES to the ring. lem/replay_train.py is the UE-side trainer "
             "and is NOT on this page's path: it builds a different optimizer, "
             "so checking any claim here against that file confirms the wrong "
             "code",
        "f": "its updates are invisible in the per-step loss panel by "
             "construction; its own loss is charted in the run tab",
    },
    "node.curriculum": {
        "t": "curriculum",
        "w": "The growth schedule. Six phases - neonate, early infant, late "
             "infant, toddler, child, mature - raise how hard the body may push "
             "(2% of full force at neonate, 100% at mature) and how far one "
             "command may move a [[handle]] (2 cm to 40 cm), shorten the wait "
             "before a command acts (30 ms to 0) - senses are never delayed, see "
             "the note below - and set the simulated age the world workers regrow "
             "the body at (1 month to 24 months). [[babble]] noise is the one "
             "thing the phases do NOT change: the table falls from a 1 cm random offset "
             "to none, and a floor lifts every phase to a random offset of 5 cm "
             "of position and 29 degrees of rotation on a command, one standard "
             "deviation.",
        "when": "The --steps-per-phase launch flag sets the pace; the "
                "sub-diagram below prints this run's own value, and the newest "
                "snapshot's header says which phase this page's model is in.",
        "h": "lem/training/curriculum.py (PHASE_ORDER, PHASE_DEFINITIONS and "
             "limits_for, which applies the babble floor) for the limits, "
             "train_mimo.py's PHASE_AGE and PHASE_DELAYS for the age and the "
             "delays, driven by the trainer's main loop; a phase change that "
             "changes the body rebuilds the world workers",
        "f": "reading the six boxes as six eras of comparable length - the "
             "whole ramp is five times --steps-per-phase [[model-step|model steps]] and "
             "every step after it is spent in the last phase. And the dropping "
             "is not total: the model steps already in the replay ring survive the "
             "rebuild and keep being re-studied.",
    },
    "node.memory": {
        "t": "persistent memory",
        "w": "Memory beyond the transformer's window. It keeps striking "
             "events - the biggest change in each half second, up to 128 of "
             "them, which is 64 s of body time - summarizes that window in "
             "sequence with a [[gru|recurrent network]], weights the window by "
             "how closely each running summary matches the present model step, "
             "and adds what it recalls to the {n_senses_word} tokens of the newest model step "
             "AFTER the transformer has finished. {memory_bias_vs_token} A "
             "second trained head forecasts "
             "the body's next five bodily needs.",
        "when": "Observes every [[model-step]]. The transformer never reads the "
                "recall on the step it is made - the prediction heads, the "
                "motor controller and the voice head "
                "do. It reaches the tokens the transformer reads one step "
                "later, and only through the top-down gate at that gate's own "
                "weight: measured here audio's blend 14 to 28 parts of the "
                "token in 100 where the gate's ceiling is 50, the other three "
                "gated senses below 1 part in 100,000. "
                "It trains through the retained events.",
        "h": "lem/model/persistent_memory.py + mimo_bridge/memory_integration"
             ".py, inside model.step() in the trainer process; the readout is "
             "created at zero, so installing the memory on an existing "
             "checkpoint changes nothing until it has trained - in the H81 "
             "lineage all 148,224 readout numbers are nonzero. Re-study "
             "re-uses the recall stored with each remembered model step, so only "
             "that readout ({memory_readout} numbers in this run) learns there: {memory_not_restudied} of the memory's {memory_params} "
             "numbers get nothing from re-study. Each step the recurrent "
             "network encodes more than the ring: MemoryHistory.snapshot() "
             "appends the open half second's biggest change so far and then "
             "the present model step, so it reads 130 positions with a full ring "
             "and 129 on the step right after a half second closes",
        "f": "it stores what was observed, never later outcomes - it cannot "
             "leak the future",
    },
    "node.self_state": {'t': 'self perception', 'w': "Self perception builds a changing state of this body from skin readings, motor capacity, sensory context and recent history. Its learned networks combine these inputs over time. A separate future-function decoder uses that state and timed commands to predict changes in sensing quality, load, stimulation and motor capacity.\n\nLearning compares these forecasts with later measurements and updates the self networks. The learning arrow returns here; the report goes to the researcher. In grounded_residual mode, measured current function anchors the forecast.\n\nThe learn setting teaches these networks. The perceive setting also adds a learned adjustment to the forward model's output, which affects motor control and predictions. The independent self-choice setting is off, shadow or on: off builds no chooser; shadow compares with a frozen copy and discards its commands; on uses a qualified frozen actor. The overview above explains each route and its activation requirements.", 'when': "Observes every [[model-step]]; forecast lessons replay the last 1, 5 and 20 executed commands (100 every 20th model step, once 100 are stored) - one command here is one model step's [[executed-command|executed command]], 0.05 s of simulated body time at the 5 physics sub-steps of B1K_skip13_492800; stored windows are re-studied every 5th model step. Those lessons move its weights only while the run is training: an --eval-only run computes and reports them and updates nothing, which is what the inspector's 'weights: frozen' label means.", 'h': 'lem/model/self_state.py + mimo_bridge/self_state_integration.py (grounded_residual), inside model.step() in the trainer process', 'f': 'its forecasts are graded only against measured function, never against its own beliefs. 565,062 of its 1,994,895 weights receive no gradient under the flags B1K_skip13_492800 was launched with. 342,726 are the retired legacy forecast head - kept so older checkpoints load, never executed while --self-prediction is grounded_residual; unmeasured: the run folder holds one checkpoint (step 494,025), so there is no earlier one to compare it with. 222,336 are the belief-into-perception bias - computed every model step and discarded while --self-state is learn; measured on the checkpoint at step 494,025: 0 of those 222,336 numbers are nonzero, so they still hold the zero they were created with. 1,429,833 weights are reached and trained (audit P204).', 'plain': "Self perception builds a changing state of this body from skin readings, motor capacity, sensory context and recent history. Its learned networks combine these inputs over time. A separate future-function decoder uses that state and timed commands to predict changes in sensing quality, load, stimulation and motor capacity.\n\nLearning compares these forecasts with later measurements and updates the self networks. The learning arrow returns here; the report goes to the researcher. In grounded_residual mode, measured current function anchors the forecast.\n\nThe learn setting teaches these networks. The perceive setting also adds a learned adjustment to the forward model's output, which affects motor control and predictions. The independent self-choice setting is off, shadow or on: off builds no chooser; shadow compares with a frozen copy and discards its commands; on uses a qualified frozen actor. The overview above explains each route and its activation requirements."},
    "bp.grade.self_state": {
        "t": "grading the self-state forecast",
        "w": "This is the lesson that is NOT graded against tokens. Its "
             "target is not a [[token]] and not the model's own belief: it "
             "is the simulator's own body function - sensing resolution, "
             "mechanical load and stimulation per skin site - together with "
             "the actuator availability MuJoCo is running with. Both are "
             "read off the frame the world worker sent, both detached, and a "
             "frame whose body-function clock does not match its own "
             "simulated time is refused.",
        "h": "observe() in mimo_bridge/self_state_integration.py builds the "
             "target (the line that starts observed=torch.cat) from "
             "meta['body_function']['values'] (lem/body_function.py "
             "BodyFunction.features()) and meta['motor_function']['values']; "
             "the same function stores it detached as each event's target "
             "(history.events.append, target=observed.detach()) and reads it "
             "back to grade the rollout (torch.stack of each event's target)",
        "f": "a self-model graded against its own prediction can score well "
             "by agreeing with itself - measuring the target instead is what "
             "rules that out here. The [[causal_pe|sense lessons]] under "
             "'the world answers' are graded the other way, against the raw "
             "arriving tokens.",
    },
    "edge.world.self_state": {
        "t": "world -> self perception: what flows here",
        "w": "Two measurements, straight from the simulator, every "
             "[[model-step]]. One row per skin site (53 on this body) carrying "
             "three fractions: sensing resolution (100% = the pressure "
             "reading is returned untouched, 0% = quantised to 2 bits of a "
             "100 kPa range), mechanical load now, and positive stimulation "
             "now. Plus one row per actuator group (10 here) carrying that "
             "group's available force (100% = the body's native force, 0% = "
             "none). 63 rows per body in all, and the forecast and its "
             "grading cover every one of them. Beside them the frame's simulated time, whose gap since the "
             "previous model step is an input, and, with --self-prediction grounded_residual, each skin "
             "site's measured function extent (area-weighted quality and the affected share of its sensor "
             "area) when the frame carries it. The skin sites' own positions and skin features arrive on the "
             "line from the tactile sense.",
        "h": "env.py fills meta['body_function'] from lem/body_function.py's "
             "BodyFunction.features() and meta['motor_function'] from the "
             "actuator gain availability; observe() in "
             "mimo_bridge/self_state_integration.py stacks the two into one "
             "target tensor (the line that starts observed=torch.cat), takes now from each frame's sim_time "
             "(delta = now - history.time) and, under grounded_residual, function_extent from "
             "meta['skin_function_granularity']",
        "f": "the 10 actuator rows are given three fields to match the skin "
             "rows, but only the first carries a measurement - the other two "
             "are constant zeros, 20 of the 189 graded numbers per body per "
             "model step, and they are marked valid, so the lesson averages them "
             "in alongside genuinely-zero skin rows",
    },
    "edge.ln_out.self_state": {
        "t": "tokens -> self perception: what flows here",
        "w": "The {n_senses_word} sense [[token|tokens]] of this [[model-step]], {token_row} numbers "
             "in all, taken BEFORE the top-down gate blends any prediction "
             "into them. With --self-prediction grounded_residual the same encoder call also hands over "
             "the vision and tactile encoders' local features - one row per vision region and per tactile "
             "contact row (up to 32), after the shared LayerNorm, with the sense's modality embedding added. Both arrive "
             "DETACHED, so no gradient from the self-perception lesson can reach "
             "the encoders. The core's own output reaches self perception one model step late, on its "
             "own line from the final LayerNorm.",
        "h": "observe() in mimo_bridge/self_state_integration.py builds the "
             "context from tokens_raw (the line that starts "
             "context=torch.cat) and adds spatial and spatial_valid to its inputs; "
             "lem/model/spatial.py encode_spatial produces them (regions = "
             "registry.output_norm(regions), sites = registry.output_norm(sites)) and "
             "LemLite.step in mimo_bridge/lem_model.py detaches them",
        "f": "the running state these feed is a tanh-bounded {d_model}-number "
             "vector: measured 2026-09-15 on the H81 eval probe's step-400 "
             "checkpoint it reaches both rails, the tanh's -1 and +1 ends, at a "
             "root-mean-square of 87% of the bound. Whether "
             "that saturation limits the forecast is unmeasured.",
    },
    "edge.motor.self_state": {
        "t": "executed command -> self perception: what flows here",
        "w": "The [[executed-command|command the body actually carried out]] "
             "over the interval that just ended (dashed: it arrives one [[model-step]] after the controller "
             "wrote it) - one row per [[handle]] ({n_handles} on this body) with 9 numbers each, "
             "flattened - stored alongside the "
             "measurement that followed it, up to 128 deep. The forecast's "
             "own recurrent cell reads one of these per command, in the sequence "
             "they happened. Detached: this lesson cannot push back on the "
             "motor controller.",
        "h": "observe() in mimo_bridge/self_state_integration.py "
             "(command_for_encoder of the executed command, appended to the "
             "event store by history.events.append); "
             "GroundedDynamics.forward in lem/model/self_state.py steps the "
             "command-history GRU once per stored command",
    },
    "edge.self_state.loss": {
        "t": "self perception -> the step's loss: what flows here",
        "w": "One number, added straight into the step's total [[loss]]: the present-belief lesson, plus the mean of the forecast lessons over the horizons run this [[model-step]], plus a re-study lesson every 5th model step. The belief-into-perception path is enabled only in 'perceive' mode. The separate self-choice setting may also read self forecasts; learn mode alone does not decide that setting.",
        "h": "LemLite.step in mimo_bridge/lem_model.py (loss = loss + "
             "self_state_loss, recorded in the breakdown as "
             "self_state_loss); the 'perceive' path is gated in the same "
             "function by the test self.self_state_mode == 'perceive'",
        "f": "this number is not a forecast error. It is squared error PLUS "
             "an uncertainty term (half of error/variance plus log "
             "variance), every body, measured field and rare changed row "
             "weighted equally. Measured 2026-09-15 on the H81 eval probe's "
             "step-400 checkpoint over its own 128 stored commands, the "
             "accuracy half came out positive and under three millionths of "
             "the uncertainty half's size, which was negative - so "
             "self_state_loss falling is the module's stated "
             "confidence shrinking toward the code's variance floor, not the "
             "forecast getting closer to the measurement. The stated spread "
             "is not calibrated either: over 109 twenty-command windows it is "
             "8.3 times the module's own error 0.05 s ahead and 3.0 times at "
             "1.00 s ahead, and it shrinks 49% across that horizon while the "
             "error grows 52%.",
    },
    "node.voice": {
        "t": "voice head",
        "w": "A readout of the transformer's output that emits 80 numbers each "
             "[[model-step]], each held between 0 and 1 by a sigmoid - built for the "
             "voice-box goal. 80 is the width of the action slot the "
             "shared-memory buffer reserves (lem/action/writer.py: float32[80] "
             "at offset 856), not a count of vocal-tract controls: the only "
             "vocal-tract parameter set this repo defines is 56 numbers - 11 "
             "named parameters, a wrapped synth clock and 44 tract diameters "
             "(tokenizer/modalities/voice.py) - so 24 of the 80 have no named "
             "meaning anywhere in the tree. It is the one box on this diagram whose weights "
             "no lesson reaches: no target is ever given for the voice and its "
             "configured loss weight is zero, so a gradient of exactly zero "
             "arrives and its numbers still hold the values they were first "
             "given. Voice is not implemented in the MIMo world yet: nothing "
             "turns these controls into sound, so the model cannot hear "
             "itself. The caregiver could not answer a sound in any case - it "
             "reads gaze direction and contact and reads no ear signal at all "
             "- so a spoken exchange needs two things built, not one: this "
             "head wired out through BioFold (the physical vocal-tract "
             "simulation), and an audio input wired into the caregiver.",
        "when": "Computed every [[model-step]] inside model.step() in the trainer "
                "process, in the same pass that produces the tokens it reads. "
                "The result is kept for this page and nothing else reads it: "
                "no loss, no planner, no world.",
        "h": "VoiceHead in lem/model/prediction_heads.py: ONE learned query "
             "vector - [[layernorm|LayerNormed]], so that normalization acts "
             "on the query and not on the senses - reads the newest frame's {n_senses_word} sense rows through 4-head [[attention]], and the single "
             "{d_model}-number summary it returns goes through an [[mlp]] down to 80 numbers "
             "whose outputs a sigmoid squeezes into 0 to 1.",
        "f": "reading the box as a working voice. Its numbers sit in the "
             "optimizer's parameter list but receive a gradient of exactly "
             "zero, and the controls' narrow spread around the middle of their "
             "0-to-1 range (50%) is what an "
             "untaught sigmoid readout produces - not a measurement of the "
             "model or of the world. Reading the 80 as 80 articulators is the "
             "same mistake: 24 of them are reserved wire space.",
    },
    "node.world": {
        "t": "the world",
        "w": "The simulated crib: the infant body, the caregiver, the toys, "
             "the mobile, the flying critters - physics, rendering and sound. "
             "It is NOT part of the model and has no learned weights. The "
             "model's commands act here, and the next model step's senses are "
             "measured here.",
        "when": "Steps in lock-step with the model: one [[model-step]] = "
                "{action_repeat} physics steps of {env_step_ms} ms = {moment_ms} "
                "ms of body time (the --action-repeat flag sets the "
                "{action_repeat}); in the crib scene each 10 ms step is "
                "itself 2 MuJoCo integration steps of 5 ms.",
        "h": "mimo_bridge/env.py on the MuJoCo engine - one world worker "
             "process per body; the trainer exchanges frames and commands with "
             "them over pipes",
    },
    "node.hself": {'t': 'legacy whole-moment self tracker', 'w': 'This box is the legacy whole-moment tracker, an earlier attempt at levels 2 and 3 of hierarchical self. It groups complete sensory moments and tracks their links with wellness over long windows. It stores statistics, not trained neural weights. Dark mode observes and reports; off does not build the tracker. The reviewed October 15 run has it off.\n\nUEMA v3.6 section 11.4 describes the named-things tracker as the replacement for these levels. Level 1 is the body schema, including tracked objects. The legacy flag named hierarchical_self is not the whole self architecture.', 'when': 'Off in the reviewed October 15 run. When enabled, the legacy tracker observes every model step; its configured windows are described below. These are not operating windows of an active tracker in the displayed run.', 'h': "When this tracker is enabled: 1. **Store kinds of model steps.** It keeps a table of 6,400 [[slot|slots]]. A slot holds one pattern: the six tokens of one model step, laid end to end. A model step that matches no slot gets a free slot. When the table is full, nothing new is stored.\n2. **Match.** Patterns are compared with a [[cosine]], a likeness score: 1 = the same, 0 = nothing in common. A model step BELONGS to a slot when every sense scores nine tenths or more against it. Only such model steps change a slot's pattern, by one part in a thousand each.\n3. **Share out the model step.** Every slot gets a share of each model step, its [[presence]]. Closer patterns get bigger shares; the shares add up to 1.\n4. **Look for a link.** It keeps two running averages of a slot's presence: a quick one (about the last 50 s at Level 2) and a slow one (the whole window). Presence is UP when the quick one is above the slow one, DOWN when below. Wellness is tracked the same way (both averaged over the 8 bodies). Every model step it multiplies the two: both up or both down gives a plus, one up and one down gives a minus. The average of the pluses and minuses, scaled to run from -1 to +1, is the [[wellness-link|wellness link]]: +1 = always together, 0 = no pattern, -1 = always opposite. Example: a slot holds 'the caregiver's face is close'. If presence and wellness are both up during visits and both down in between, the pluses pile up and the link climbs toward +1.\n5. **Rule out luck.** The link is compared with 8 [[sign-flipped-control|controls]]: copies of wellness with the sign flipped at random times, which cannot have a true link. A slot COUNTS only if it keeps beating its controls, window after window.\n\n**Right now.** It is switched off in B1K_skip13_492800, so there is nothing to report.\n\n**Why it sits outside the model.** Two reasons from its code. The design defines these levels as running scores, not trained weights, like Level 1. And it is unproven, so it may touch no lesson, goal or wellness term until a separate switch is certified. It does read the model's tokens, so when training changes the [[encoder|encoders]] (the parts that make the tokens), stored patterns stop matching.\n\n**The code.** mimo_bridge/hierarchical_self.py.", 'f': "- **Out-of-date slots.** A pattern is stored once. When the encoders change, today's model steps stop matching and every score is about old model steps. The inspector panel's first rows show it.\n- **Windows counted by the clock.** A slot gains a window every window, even if no model step belongs to it again, so a count can describe a pattern seen for seconds.\n- **The sixth slot changed meaning.** With --vestibular-sense on, the slot the intermind placeholder's noise held carries the vestibular input, so a table filled before the switch stored noise in that part of every pattern.\n- **Link size by itself.** Controls score almost as high as the real link, and the slow averages start at zero, so early links all read positive. Only the count means anything.\n- **A link is not a cause.** Moving together does not show that a pattern HELPS.\n- **A restart with fewer slots.** The dropped slots take their evidence with them. The log line 'hself: migrated' says how many.\n\n[[hself-measurements|Measurements behind these warnings]].", 'plain': 'This box is the legacy whole-moment tracker, an earlier attempt at levels 2 and 3 of hierarchical self. It groups complete sensory moments and tracks their links with wellness over long windows. It stores statistics, not trained neural weights. Dark mode observes and reports; off does not build the tracker. The reviewed October 15 run has it off.\n\nUEMA v3.6 section 11.4 describes the named-things tracker as the replacement for these levels. Level 1 is the body schema, including tracked objects. The legacy flag named hierarchical_self is not the whole self architecture.'},
    "node.entity_self": {'t': 'hierarchical self: named things, levels 2 and 3', 'w': "This is the named-things mechanism for levels 2 and 3 of hierarchical self in UEMA v3.6 section 11.4. It receives supplied availability of each named thing and the body's wellness or a selected ability measure. It tracks their associations over shorter and longer windows. Names come from world metadata, not learned visual recognition; experience teaches the association.\n\nA separate reader compares learned links with the same links assigned to the wrong things. After repeated evidence checks pass, it adds a signed value to remembered-goal scores. While the gate is closed it adds zero. The observer's dark mode does not by itself disable this reader.\n\nThe reviewed October 15 run uses entity_self=dark with a reader weight of 0.1. The body schema, learned bodily self state, named relations and prediction reliability serve distinct roles within the wider self.", 'when': 'Observed every model step when entity_self=dark. The reviewed October 15 run enables this observer and a separate self-value reader weight of 0.1. The reader contributes only after its evidence checks pass.', 'h': "1. **Read the world.** mimo_bridge/entity_presence.py turns the world's own per-body report into one number from 0 to 1 per named thing: the adult's visibility, whether a toy is out, how far off the eye a critter or the baby's own hand sits. A missing reading is 0.\n2. **Correlate, per body.** Each body's presence is compared with THAT body's [[wellness]] - or, when the run asks for it, with how many kinds of consequence the body caused that model step (a contact, an object moved, a view gained). The older table averaged all the bodies into one number first.\n3. **Correct the start.** Both running averages are divided by how much of them has filled, so early links do not all read positive. Without that, all 6,400 of H84's hours-level links read positive on a table that had lived 1.4 windows.\n4. **Count a window only when it said something.** A thing whose presence never moved inside a window gains no window from it, in either direction.\n5. **Rule out luck**, with the same 8 [[sign-flipped-control|controls]] and the same pass mark as the older table.\n\n**The code.** mimo_bridge/entity_self.py and self_value_reader.py.", 'f': "- **A name is not a thing.** The list is fixed by hand from what the crib reports. Something with no look or feel - money, a promise - has no presence signal here, so this cannot reach it.\n- **A link is not a cause.** Moving together does not show that a thing HELPS. The world's own comings and goings give a check: the adult leaves every 60 seconds of body time, so wellness with and without it must agree with the link's sign.\n- **The teacher can be the wrong signal.** Learning from wellness inherits whatever raises wellness; the run's own choice is named in the live rows.\n- **The reader could act on noise.** That is what its shuffled control is for, and why its gate stays shut until the real reader beats it.", 'plain': "This is the named-things mechanism for levels 2 and 3 of hierarchical self in UEMA v3.6 section 11.4. It receives supplied availability of each named thing and the body's wellness or a selected ability measure. It tracks their associations over shorter and longer windows. Names come from world metadata, not learned visual recognition; experience teaches the association.\n\nA separate reader compares learned links with the same links assigned to the wrong things. After repeated evidence checks pass, it adds a signed value to remembered-goal scores. While the gate is closed it adds zero. The observer's dark mode does not by itself disable this reader.\n\nThe reviewed October 15 run uses entity_self=dark with a reader weight of 0.1. The body schema, learned bodily self state, named relations and prediction reliability serve distinct roles within the wider self."},
    "node.caregiver": {
        "t": "caregiver (part of the world)",
        "w": "An adult in the world - not part of the model. The world picks "
             "their comings and goings at random. While present they bring "
             "and remove toys, speak short lines rendered ahead of time by a "
             "text-to-speech voice, go quiet for a stretch, and ANSWER WHAT "
             "THE BABY DOES: a look held on their face for 0.4 s, a look held "
             "on the toy for 0.4 s, any touch of the toy, or the toy shifting "
             "1.5 cm in half a second while the baby is watching it. The toy "
             "is a mocap body with no degrees of freedom, so a push cannot move "
             "it - the push moves only the baby's own hand. The adult carries "
             "it, at up to 6 cm/s, except while the baby grips it under the "
             "[[toy-hand-off|toy hand-off]] (--toy-hand-off follow, "
             "2026-09-21): then it goes where the gripping hand goes. "
             "{toy_hand_off_run} The adult keeps the toy at a fixed "
             "offset from the baby's left eye, so when the baby moves its head "
             "the adult moves the toy after it. On run H84 (8 bodies, 27,176 s "
             "of body time, 688 'the toy is moving' lines) 78 of 100 of those "
             "lines followed the baby moving its own head more than 1.5 cm in "
             "the half second before, 11 of 100 the adult carrying the toy in "
             "or out, and 10 of 100 came with the head moving 1.5 cm or less; "
             "every one needed the baby's look on the toy. They have no ears "
             "of any kind, so they "
             "cannot answer the baby's sounds, and the model has no voice in "
             "this world yet either.",
        "when": "The two launch flags time the TOY, not the adult: a toy "
                "stays in reach 60 s, then 15 s pass before the next one "
                "(this training family). The adult's own presence is a random "
                "draw of 180-300 s, and its absence is 18 s of walking out, a "
                "random 30-60 s wait and 18 s of walking back - 67 to 97 s in "
                "all, measured once end to end at 91 s. While it is away, "
                "nothing the baby does is answered, and the toy and the "
                "mobile leave the crib with it. Their speech reaches the ears "
                "with real direction cues: the far ear hears it quieter and "
                "later.",
        "h": "mimo_bridge/caregiver_interaction.py (the exchange: the "
             "departures, the four answered events - both look tests judged "
             "against the LEFT EYE camera only, inside a 15 degree half-angle "
             "cone with a clear line of sight, which a shape the baby cannot "
             "see (a switched-off toy shape, a withdrawn toy) does not block "
             "- and the quiet stretch) + "
             "caregiver_language.py (greeting, return, farewell, quiet, "
             "put_away, face_look and present/touch/look/move phrases per "
             "toy) + caregiver.py (movement, toys, the mobile, recording) + "
             "caregiver_visual.py (body, face, mouth, and the speech source "
             "pinned to the visible mouth) + speech.py (waveform to the "
             "ears), ticking inside each world worker process once per 0.01 s "
             "of body time (env.py) - not the trainer",
        "f": "the world decides these comings and goings, not the model - "
             "never read an absence as the model's choice; and a touch is "
             "answered only if the adult is free to speak within 2.0 s of the "
             "contact starting, while a look or a moving toy waits out the "
             "adult's turn and is answered when it stops speaking",
    },

    # ================= the planner's data and steps (its detail diagram) =========
    # Two keys were renamed on 2026-10-02 (the style guide "ONE WORD PER THING"): the goal-ring entry and the
    # candidate command. viewer.js DOC_ALIAS opens the moved entries from links and saved addresses
    # written with the old keys.
    "term.goal-ring-entry": {
        "t": "A remembered sensory state and its recorded scores",
        "w": "While the intention planner is enabled, each model step adds one entry per body to a ring holding the newest 200 entries. It records the {n_senses_word} sensory lists, {d_model} numbers each, after any top-down blend; a raw value; the channel record when normalized ranking is enabled; and the measured handle pose.\n\nThe raw value combines the fixed wellness scores with enabled movement, agency, consequence, named-entity and curiosity contributions. The normalized ranking instead uses stored channel changes and measured harm. [[wellness-ranking|How those changes are ranked.]]\n\nChoosing an entry supplies a remembered sensory target. The planner searches for a new command from the current state; the entry is not an instruction to repeat its old command.",
        "h": "mimo_bridge/lem_model.py: goal_ring.append, _rank_store_moment, _goal_ring_ch and _goal_ring_pose. [[goal-ring|Goal ring]] and [[wellness|fixed scores]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "term.candidate-command": {
        "t": "candidate command",
        "w": "The command the planner's search changes: one row per [[handle]] ({n_handles} in "
             "{run_name}) of 9 numbers - target position x, y, z (in units of 200 cm from the world "
             "origin), target orientation ([[quaternion]], 4 numbers), radius (0 to 1) and strength "
             "(0 to 1). {hand_mode_sentence} It starts as the motor controller's command at that model "
             "step. Over the search's 20 updates the Adam step changes {search_space_words}. Every "
             "command the search scores is scored in the form the body would be given it "
             "([[execution-filter|the execution filter]]); the command it keeps is the raw one, and "
             "on its way to the body it goes through the same filter again. One of the 9 numbers "
             "changes nothing the body does: this world's action decoder reads position, orientation "
             "and strength and never reads radius. The gaze row of the executed command is rewritten "
             "afterwards with the eye command the world used. History: until the motor controller's "
             "position numbers were bounded at their source (September 19, 2026), 580 of the H81 "
             "probe's 1,560 position numbers - 37 of every 100 - started where the filter's clamp has "
             "a slope of exactly 0, so the search could not change them; seeded from the bounded "
             "value, all 1,560 measured a slope above 0, and the clamp now passes the slope of a tanh.",
    },
    "term.execution-filter": {
        "t": "execution filter",
        "w": "The steps that turn a command into the command the body is given, applied inside "
             "every one of the planner's 20 updates (LemLite.planned_execution) and again on the live "
             "path to the body:\n\n"
             "{execution_filter_steps}\n\n"
             "[[babble|Babble]] noise is not in the filter: it is random, and it is added only to a "
             "command the planner did not write.",
        "h": "LemLite.planned_execution (mimo_bridge/lem_model.py): differentiable_execution_transform "
             "(the clamps, written so that a clamped number still passes the slope of a tanh to the "
             "search), gaze_row_as_sent, smooth_blend and, under --executed-fields-only on, "
             "executed_fields_only; the division by 200 is normalize_executed_command "
             "(lem/model/encoder_registry.py), called inside generate_action",
        "f": "reading the planner's kept command as what the body executes: the body gets the "
             "filtered form, blended with the command before it",
    },
    "term.hand-goal": {
        "t": "the finger search",
        "w": "Under --hand-goals memory, which needs --hand-target-mode wrist_relative_v1: once the "
             "first search has kept a command, a goal entry whose recorded hand pose is valid gets a "
             "second search over the finger joint target angles only. 20 Adam steps of step size one "
             "tenth change those angles, each held within {finger_bound} of the angle the first search "
             "left and inside the joint's range. Each candidate is scored by the distance between the "
             "finger poses that a fitted linear hand-response model predicts 250 ms ahead - from each "
             "joint's present angle, its speed and the target, blended with the previous targets as "
             "the execution filter blends them - and the hand pose recorded with the goal entry. Of the "
             "21 scored (the first search's angles and the 20 steps) the lowest is kept; every other "
             "number of the command stays as the first search left it. {hand_goals_run}",
        "h": "lem/model/hand_response.py CalibratedHandResponse.choose_hand_goal: predicted angle = "
             "present angle + range x (coefficients . [1, (target - present) / range, a quarter of "
             "the speed / range]); the coefficients are a frozen calibration checked against its "
             "receipt (version, source hashes, the four-and-four body split it was fitted and tested "
             "on); called from LemLite._plan_chain",
    },
    "term.scale-check": {
        "t": "the size check",
        "w": "After each kept command the forward model runs once more on it (LemLite._plan_chain, "
             "the transition with the context advanced). For every sense, motor included, the size "
             "([[rms|RMS]], over all bodies' numbers together) of the predicted tokens is divided by "
             "the size of the same sense's tokens at this model step. If any ratio is below one half "
             "or above two, the whole plan is dropped - every depth, every body - and no planner "
             "command is sent: the body gets the motor controller's command, with [[babble]] noise "
             "when the noise is on. Real tokens leave the shared output [[layernorm|LayerNorm]] at "
             "size 1, so the band is half to twice a real token's size.",
        "h": "lem/model/predictive_state.py scale_report(real, imagined): one ratio per sense, in band "
             "from one half to two; LemLite._plan_chain empties the plan when any is out of band and "
             "keeps the ratios in last_scale_report, which the model holds until its process ends",
        "f": "reading an empty report as a check that never runs: it runs after every kept command, "
             "and a report exists only after a drop",
    },
    "term.predicted-outcome": {
        "t": "predicted tokens",
        "w": "What one run of the frozen [[forward-model]] gives for a candidate command: for each of "
             "the {n_bodies} bodies, one {d_model}-number token per sense except motor, for the next "
             "model step - the prediction heads' output passed through the shared output LayerNorm, "
             "so each is held at size 1 like a real token - plus the motor token the command itself "
             "was encoded to. Only senses measured live and not motor are compared with the goal "
             "tokens: the motor token is the command itself, and a stale or synthetic sense "
             "([[source-label|source label]]) carries no measurement to aim at. The 20 updates push "
             "down the sum of those senses' mean squared differences plus one hundredth of every "
             "handle's squared strength ([[intention-distance|the score]]). Two unrelated tokens "
             "differ by about 2 per compared sense, so with four compared senses the goal part reads "
             "about 8 for unrelated tokens and 0 for identical ones.",
    },

    # Goal selection, observed wellness and prediction-error learning are separate paths.
    "loop.overview": {
        "t": "A remembered goal, a new command, a measured outcome",
        "w": "The goal ring stores recent sensory readings and wellness information, not a list of successful commands. Wellness ranks the remembered states. The planner starts from the current state and searches for a new command whose predicted senses approach the chosen state. After execution, new senses and measured wellness return through two paths: prediction lessons and new goal memories.",
        "h": "lem_model.py: goal-ring storage and selection in LemLite.step; _plan_chain; _normalized_goal_values. intention_generator.py: generate_action. causal_chain.py: _grade. The card labels read the selected run configuration.",
        "f": "A high historical wellness value does not prove that repeating a pose restores that value. Reaching a sensory goal, predicting wellness and measuring wellness are different operations."
    },
    "loop.goal": {
        "t": "What is remembered and selected?",
        "w": "For each body, the goal ring holds up to 200 recent model steps of encoded sensory readings and their wellness values. It also stores the corresponding body-handle poses. With normalized ranking, stored wellness-channel deviations are scaled by current channel statistics, weighted and restricted by measured harm. The highest remaining score selects a remembered state. Optional goal-locality or revaluation settings can change selection. This selects a target state; it does not recover the old command that preceded it.",
        "h": "lem_model.py: goal_ring and companion deques in __init__; goal-ring append and best_t selection in step; _rank_store_moment and _normalized_goal_values."
    },
    "loop.search": {
        "t": "How is the new command chosen?",
        "w": "The search receives the current sensory tokens, remembered goal tokens and current motor-controller proposal. It changes candidate command values while holding model weights fixed. Its score adds squared differences between predicted and goal sensory tokens, effort, and any enabled constraints. Predicted future wellness enters this score only when prospective wellness is enabled. With hand goals from memory, a second search uses the fixed hand-response fit to choose finger goals for the remembered hand pose; the final command is predicted again.",
        "h": "intention_generator.py: generate_action. lem_model.py: _capture_plan_state and _plan_chain. hand_response.py: choose_hand_goal.",
        "f": "The hand search has its own objective. Improvement recorded before that search does not by itself certify the final executed command."
    },
    "loop.execute": {
        "t": "Which command reaches the body?",
        "w": "When enabled for execution, a planner command replaces the motor-controller proposal on a planning model step. Other model steps use the controller with the configured exploration. Smoothing, rest rules and the command hold can alter which command is sent. The body decoder converts that final command into actuator controls. The actual executed command is returned to the model for the delayed prediction lesson. Gaze has its own execution timing.",
        "h": "lem_model.py: intent_action, motor_for_action and action-source selection. train_mimo.py: held_motor, hand_goal and env.step. env.py: command decoding."
    },
    "loop.outcome": {
        "t": "Does the outcome recover the remembered wellness?",
        "w": "Wellness is measured again from the new sensed state and recent experience. Its channels include bodily conditions and history-dependent measurements, so the same pose need not have the same value later. When causal-chain recording is enabled, predicted and measured changes are compared for the five wellness terms that have predictions, after the next model step and the command hold. This is a recorded measurement, does not directly train the chosen command. It is not a comparison against the full score that ranked the old goal.",
        "h": "causal_chain.py: grounded_terms, _grade and _finish. lem_model.py: _rank_store_moment. wellness_channels.py: channel normalization and rank_values.",
        "f": "The five-term comparison does not cover every goal-ranking channel. Learning progress depends on recent prediction improvement; returning to old sensory readings need not reproduce that improvement."
    },
    "loop.memory": {
        "t": "How does measured wellness affect the next goal?",
        "w": "The new sensory readings and measured wellness enter the goal ring. With normalized ranking, running channel statistics also change and help rank later goals. The diagnostic predicted-versus-measured wellness record does not directly subtract value from the old selected goal after a failed attempt.",
        "h": "lem_model.py: _rank_store_moment, goal-ring append, _normalized_goal_values. causal_chain.py: _grade records errors and wellness changes."
    },
    "loop.learning": {
        "t": "How does the outcome change the model?",
        "w": "The previous sensed state and the command actually executed are used to predict the senses arriving now. Training reduces differences between those predictions and the measured targets. Separate lessons train other outputs, including a learned copy of five measured wellness terms. The command search changes command variables, not model weights. A recorded wellness improvement does not itself update a command-selection policy. Evaluation-only runs retain their weights.",
        "h": "lem_model.py: delayed next-frame lesson, main backward/optimizer step, replay_update. loss_manager.py: approximate_wellness alignment. intention_generator.py: parameter freeze/restore."
    },
    "bp.loop": {
        "t": "How prediction errors change weights",
        "w": "At the start of a model step, incoming senses describe the previous executed command. The model re-runs that previous state and command, compares the prediction with the incoming raw sensed targets, and adds the other enabled lessons. During training, backward computation produces gradients and AdamW updates connected weights. The next command is then selected and executed; its outcome arrives on the next model step. Evaluation-only runs omit weight updates. Replay, when enabled for training, takes a separate optimizer step."
    },
    "bp.forward": {
        "t": "Read the outcome and predict it again",
        "w": "The arriving senses come from the previous executed command. The delayed route re-encodes the previous raw state with that command and predicts the arriving senses. Targets use current raw encodings detached from the gradient. A separate current-state forward pass supplies current outputs and the controller proposal. During training, the weight update occurs before planning the next command. {imagination_forward}. On a planning step the frozen forward model runs inside the planner's 20 rounds.\n\n{learning_update_run}",
        "h": "lem_model.py: LemLite.step delayed raw re-encoding, current forward pass and optimizer step."
    },
    "bp.grade": {
        "t": "Compare predictions with sensed outcomes",
        "w": "The prediction and its target must refer to the same command and time. Valid arriving senses grade the delayed predictions. For the motor lesson, the controller is run a second time on the PREVIOUS model step's stored inputs; that output is compared with the pose reached now. The controller proposal made from the current inputs is a different output. Other enabled lessons have their own targets, weights and schedules. During training, their combined loss is used for the main backward computation; it is not a wellness score or a measure of an acquired ability. The instance inspector shows the recorded loss terms for the selected snapshot. {learning_update_run}",
        "h": "lem_model.py: next-frame prediction loss, loss_manager.compute and additional side losses.",
        "f": "See [[next-frame-norm|next-step error scaling]] for the selected run's normalization setting."
    },
    "bp.backward": {
        "t": "Compute gradients along connected learning paths",
        "w": "Backward computation calculates how connected weights contributed to the loss. Through the core, it follows transformer blocks from the last block to the first; this model stacks {n_blocks} such blocks. A detached target, a frozen parameter or a disconnected side path receives no gradient from that lesson. Different lessons reach different encoders and heads; not every encoder receives every lesson. The live main gradient groups are clipped before the update. The within-body proprioception repair adds its own encoder gradient after the main clipping step.",
        "h": "lem_model.py: loss.backward, _clip_training_gradients and proprio_within_body_loss gradient injection."
    },
    "bp.update": {
        "t": "Update weights during training",
        "w": "AdamW uses each connected parameter gradient and its saved optimizer state to update model weights. Evaluation-only runs skip this update. Planning happens afterward, with model weights fixed while candidate command values are optimized. A weight update can improve later predictions; the body measurements decide whether the resulting commands improve function.\n\n{learning_update_run}",
        "h": "lem_model.py: optimizer.step before intention planning; intention_generator.py: generate_action freezes model parameters."
    },
    "bp.replay": {
        "t": "Replay: a separate update from remembered transitions",
        "w": "When training and replay are enabled, every {replay_every} model steps the sampler requests {replay_batch} windows of {replay_window} model steps from individual body episodes. Fewer than {replay_floor} usable windows skips the pass. Replay predicts stored next-sense tokens and the achieved next pose. It trains the core, prediction heads, controller, memory bias and current motor encoder. The other sense encoders do not receive this replay gradient: their remembered tokens are detached. This pass has its own loss, backward computation and optimizer step. Evaluation-only runs do not start replay.\n\n{learning_replay_run}",
        "h": "replay.py: sample_windows. lem_model.py: replay_update. Replay configuration and any recorded pass come from the selected run."
    },
    "edge.bp.forward.grade": {
        "t": "Prediction and measured target",
        "w": "Predictions remain attached to their learning computation. The raw sensed targets are detached. One model step spans {moment_ms} ms of body time in {run_name}. The simulator is outside that computation: no gradient passes through the physical world."
    },
    "edge.bp.grade.backward": {
        "t": "Combined loss enters backward computation",
        "w": "The enabled lessons contribute weighted terms to the main loss. Backward computation follows their connected parameter paths. Separate gradient injection and replay updates are described beside this loop."
    },
    "edge.bp.backward.update": {
        "t": "Gradients enter the weight update",
        "w": "Gradients are clipped by the live training groups before AdamW updates their parameters. The within-body proprioception correction adds its own encoder gradient after main clipping. A missing gradient is different from a measured zero gradient."
    },
    "edge.bp.update.forward": {
        "t": "Execution and the next sensed outcome",
        "w": "After the live update, selection produces the next command. The trainer applies its smoothing and hold rules and the body executes. On the next model step, incoming senses supply the next measured targets. The next forward computation reads the current weights; no gradient crosses the simulator."
    },
    "edge.bp.replay.update": {
        "t": "Replay takes its own optimizer step",
        "w": "Replay samples stored transitions from a ring of {replay_size} model steps per body and takes a separate AdamW step when enabled and enough same-episode windows are available. Its sensory tokens are detached, while motor inputs are re-encoded with the current motor encoder and the command actually executed."
    },

    "term.drives-pairing": {
        "t": "Which sensor readings does prediction accuracy compare?",
        "w": "**Usual delayed comparison.** At the current model step, the code re-encodes the previous raw sensor readings together with the command actually executed. The forward model and prediction heads calculate the sensory-number lists for the current model step. Those predicted lists are compared with the current sensor readings after encoding, before the top-down gate mixes predictions into perception. The word delayed describes which command is being checked; this prediction is recomputed during the current model step.\n\n**Fallback comparison.** If that delayed prediction is unavailable, the code compares the current forward pass with the current encoded readings. With residual prediction enabled, the prediction is current numbers plus a predicted change. Subtracting the current numbers leaves only that predicted change. A high score then means a small predicted change, not an accurate forecast of a later reading.",
        "h": "mimo_bridge/lem_model.py: LemLite.step builds nf_preds, stores _nf_pred, selects _wp/_wt and writes last_pe_source. The snapshot field drives.prediction_error_source records the selected pairing.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "Use the recorded pairing, not a rule that every resume must use the fallback. Prediction history and reset state determine which pair is available. A cleared or missing transition and a run without the delayed lesson can use the fallback."
    },
    "term.wellness-gradient": {
        "t": "The wellness derivative supplied to the motor controller",
        "w": "For each body, the controller receives {n_handles} rows of nine numbers alongside the encoded state and measured handle state. These numbers are derivatives: how changing a coordinate in the current measured handle-state tensor changes the sum of the fixed wellness scores.\n\nOnly the motor-state formula depends on that tensor. With the boundary setting, its nonzero derivatives concern position coordinates near or beyond the configured boundary and point toward lower position penalties. The other wellness scores do not provide a pressure-withdrawal or capacity-recovery direction through this input.\n\nThe controller uses these numbers as inputs to its neural network. Supplying a derivative does not force the output command to follow that direction.",
        "h": "mimo_bridge/lem_model.py: step builds motor_commands, sums grounded scores, calls torch.autograd.grad and passes wellness_grad to motor_controller. lem/model/wellness.py: motor_command_gradient and _formula_motor_state_boundary.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "UEMA 3.5 section 7.4 describes a derivative with respect to motor commands that drives command generation. The current measured-pose derivative provides a narrower input. [[wellness-spec|Code and specification comparison.]] The alternative preference setting supplies a position derivative even away from the boundary."
    },
    "term.wellness-spec": {'t': 'Wellness code compared with UEMA development specification 3.6', 'w': "**Matches the specification:** several evaluation scores; a learned predictor trained to copy measured scores; body-map extent selected instead of certainty; and planning toward a desired encoded sensory state.\n\n**Implementation has changed:** sections 7.2-7.3 show four scores, while this code has five fixed main scores plus self-touch. Fixed formulas replace the schematic learned evaluator. Capacity is implemented; the note calling it a zero stub is outdated. Goal ranking now uses additional channels, recent-history subtraction, scaling and evidence gates.\n\n**Important remaining difference:** sections 7.4 and 8 describe wellness changes directing movement, including withdrawal from harmful contact. The current fixed-score derivative supplied to the motor controller comes only from the handle-position formula. Skin discomfort, capacity and body-map changes do not supply their own command direction through that derivative. The learned predictor can score imagined results where the optional prospective path is enabled.\n\nThe source code defines the behavior described on this page. The specification's thermal-contact examples and proposed automatic grip recovery should not be read as descriptions of implemented pathways.", 'h': 'docs/UEMA/UEMA_Spec_v3_6.txt: sections 7, 8, 9, 10 and 11. Source comparison: lem/model/wellness.py, lem/model/wellness_channels.py, lem/model/contact_wellness.py and mimo_bridge/lem_model.py.', 'h_label': 'Source and calculation details', 'f_label': 'Limits and settings', 'f': 'This documentation correction does not change training. [[wellness-gradient|Implemented motor input]], [[approximate-evaluator|learned score predictor]], and [[wellness-ranking|implemented ranking]].'},

    "term.mechanical-load": {
        "t": "Mechanical skin load: peak, duration and rubbing",
        "w": "Mechanical_v4 calculates these contributions for each skin patch:\n\n1. **Acute pressure:** take the larger of current pressure and stored peak pressure. Multiply by regional sensitivity, divide by 219.4 kPa, subtract one and keep only a positive result.\n2. **Sustained exposure:** multiply accumulated pressure exposure by sensitivity and divide by 219.4 kPa times 60 seconds.\n3. **Rubbing:** multiply stored rubbing work per area by sensitivity and divide by 2,194 joules per square metre.\n\nAdd the three contributions to get L. Aversion = L/(1+L): L=0 gives 0, L=1 gives 50%, and L=9 gives 90%. The largest patch aversion reduces the body's tactile-comfort score.",
        "h": "lem/skin_mechanics.py: mechanical_load, MechanicalExposure.update and MechanicalProfile. The selected rubbing rule uses measured friction work when shear exceeds its onset threshold.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "Stored peak pressure decays with a 100 ms time constant. Sustained exposure starts accumulating above a quarter of the sensitivity-adjusted pressure reference; it grows continuously, not only after a minute. Exposure uses a 60-second loaded time constant and a 10-second recovery time constant. Rubbing memory also decays over 10 seconds. A time constant describes exponential change, not a fixed time after which the memory suddenly vanishes."
    },


    # ================= arrow data contracts (click any line) =================
    "edge.world.sense_vision": {
        "t": "world → vision: what flows here",
        "w": "Four pictures every [[model-step]], sent as raw bytes: a wide view "
             "(256x256 px, 60 degrees) and a [[fovea]] (288x288 px, 24 "
             "degrees) per eye, every colour 0-255. The trainer cuts them into "
             "1,160 patches of 16x16 pixels - 580 per eye - and divides by "
             "255, so what leaves this line is 0 to 1, with four tags per "
             "patch. One fresh set of four pictures per model step: the arrow is "
             "never empty. No [[gradient]] crosses it - the world is a "
             "separate process with no learned weights.",
    },
    "edge.world.sense_tactile": {
        "t": "world → tactile: what flows here",
        "w": "One pooled reading per skin part every [[model-step]]: 53 rows x 10 "
             "numbers - pressure in pascals, sideways rub in pascals, slip "
             "speed in mm/s, contact area in mm2, four vibration strengths "
             "(the wobble in the pressing force at 5, 10, 15 and 20 Hz, "
             "measured over the last second) and two skin-stretch numbers MIMo "
             "leaves at rest. They are pooled from the 4,683 sensor points of "
             "this run's 24-month body. With skin function on (this run "
             "launched with --body-function modulate) the pressure and rub "
             "that travel are the SENSED values: wherever a skin sample's own "
             "function has fallen below full they are rounded to a coarser "
             "step, and the undegraded physical values are kept beside them "
             "for audit only, never handed to the model. No [[gradient]] "
             "crosses.",
    },
    "edge.world.sense_audio": {
        "t": "world → audio: what flows here",
        "w": "514 numbers every model step. First the stereo spectrum: the sound "
             "level in 256 [[frequency-band|frequency bands]] for each ear, "
             "512 numbers from 0% (silence; the scale bottoms out at a level "
             "100,000,000 times smaller than its top) to 100% of the scale (the top, a summed "
             "level of 1), each eighth of the scale a tenfold step. It is a level, not a "
             "measured sound power: what is added up per band is each source's "
             "gain times its spectral shape, and speech joins as the square "
             "root of its power. Then two direction numbers on their own "
             "path into the encoder, not folded into the spectrum: how much "
             "louder one ear is, in decibels (the encoder is handed it "
             "divided by 20), and how much earlier that ear hears it, in "
             "milliseconds - never more than 0.204 ms, the time sound takes to "
             "cross the 7 cm between the ears. So direction is handed over, "
             "not inferred. THREE further numbers are written into every frame "
             "and then dropped before the encoder: the angle to the loudest "
             "source, a front-versus-back confidence, and the left ear's "
             "overall loudness. No [[gradient]] crosses. Audio is never "
             "stale.",
    },
    "edge.world.sense_proprioception": {
        "t": "world → proprioception: what flows here",
        "w": "One row per [[handle]] every [[model-step]] - 26 on this trunk body: "
             "orientation [[quaternion]] (4, unit length), position in world "
             "centimetres (3), velocity vector in cm/s (3; up to 59 cm/s in a "
             "40-step probe of this crib) and muscle freshness 0-1 (1). The "
             "tokenizer then rewrites every position as (this part - ROW 0) "
             "divided by 200 cm, and row 0 is the LEFT WRIST - so the numbers "
             "the model reads are left-hand-relative, while tactile's are "
             "hip-relative. No [[gradient]] crosses.",
    },
    "edge.world.sense_motor": {
        "t": "world → motor: what flows here",
        "w": "The achieved pose - what the body actually did, read from "
             "physics - every [[model-step]]: one row per [[handle]] (26) x 10 "
             "numbers, being the position in world centimetres (3) - the SAME "
             "26 bodies proprioception carries, but measured from the world "
             "origin, where proprioception measures them from row 0, the left "
             "wrist; the two blocks therefore differ by the left wrist's own "
             "position on every row - the "
             "orientation [[quaternion]] (4), the radius and strength that "
             "were COMMANDED (the body has no achieved counterpart for those) "
             "and an is-active flag. The trainer completes this row with the "
             "[[executed-command]] (the ledger's half), so the commanded half and the "
             "achieved half of a handle sit in one row. No [[gradient]] crosses. "
             "World centimetres is the frame the body acts in: the executed "
             "command in this same row is world centimetres divided by 200, and "
             "the reach clamp reads this position back as the handle's current "
             "world place before limiting the step, so both halves of one row "
             "have to share it.",
    },
    "edge.world.sense_intermind": {
        "t": "world → intermind: what flows here",
        "w": "Nothing. The world produces no intermind signal; a placeholder "
             "generator fills the slot every [[model-step]] with 5 rows x {intermind_row} "
             "numbers drawn from a bell curve centred at 0 with a spread of "
             "one hundredth and clipped at 0 - so half of them are exactly 0 and the "
             "rest sit under five hundredths. Its lessons weigh zero.",
    },
    "edge.sense_vision.enc": {
        "t": "vision → its encoder: the data",
        "w": "One matrix per body, 1,160 rows x 772 numbers, every "
             "[[model-step]]: 768 pixel values 0 to 1 (a 16x16 patch, 3 colours) "
             "plus 4 tags - where the patch looks (azimuth and elevation in "
             "radians), which eye saw it (-1 left, +1 right), and how much "
             "world one patch covers (3.75 degrees in the wide "
             "view, 1.33 degrees in the sharp centre; the tag itself is "
             "written in radians). That last "
             "tag is the only thing separating a coarse wide patch from a "
             "sharp foveal one pointing the same way. Left and right eye rows "
             "alternate so the encoder's windows straddle both. The "
             "[[gradient]] of the step's losses reaches the encoder's weights "
             "through this line but stops at these numbers: they are "
             "measurements, not something the model can change.",
    },
    "edge.sense_tactile.enc": {
        "t": "tactile → its encoder: the data",
        "w": "Up to 32 rows x 19 numbers, every [[model-step]] - one row for every "
             "skin part with any nonzero reading, which includes a part whose "
             "touch ended within the last second and whose vibration strengths "
             "are still nonzero; sorted by pressure, highest first, and cut at "
             "32 of the 53 skin parts. Per row: where the "
             "part sits relative to the HIP (3, in metres divided by half a metre), "
             "then pressure, rub, slip, area and four vibrations, each divided "
             "by its own constant - pressure by 100,000 Pa, rub by 50,000 Pa, "
              "slip by 200 mm/s, area by 20,000 mm2, each vibration by 10 - then "
             "the two SELF-TOUCH numbers (2026-09-21): how much of this contact's "
             "force came from the body's OWN parts, from 0 (none of it) to 1 "
             "(all of it), and how many different own parts are on it, over 4. "
             "Both read 0 for the floor, the toy and the adult, so these two are "
             "the only thing in the tactile rows that tells a hand landing on its "
             "own thigh from the floor pressing on it. Then THE PUSH ITSELF: which "
             "way it came from (3 numbers, a unit direction pointing into the body), "
             "how deep the contact is (centimetres divided by a tenth of a centimetre, so 1.0 = one millimetre) and how fast it arrived (centimetres a second divided by 2, so 1.0 = 2 cm/s). Those divisors are the 99th percentile of what the crib body produces (measured 2026-09-21, touch_adapter.py PUSH_DEPTH_SCALE_CM and PUSH_ARRIVAL_SCALE_CMS): the deepest contact measured, 0.312 cm, reads 3.1 and the fastest arrival, 14.0 cm/s, reads 7.0. Until 2026-09-21 they were divided by 10 cm and 500 cm/s, where the same contacts read about 3% of one divisor. The felt numbers above say how hard "
             "and how much it rubbed; these say which way and how sudden, and until "
             "2026-09-21 they were assembled for the model and read by nobody. Last "
             "comes a validity "
             "flag. Those divisors are what turn a number here "
             "back into a pascal. The area channel is the one that saturates: "
             "in a 40-step probe of this crib the largest touching part "
             "reached the whole 20,000 mm2 - a reading of exactly 1.0 - on 20 "
             "of the 40 steps. Unused rows are zeros with the validity flag "
             "at 0.",
    },
    "edge.sense_audio.enc": {
        "t": "audio → its encoder: the data",
        "w": "514 numbers every [[model-step]]: a 256 x 2 stereo spectrum (0 to 1, "
             "40 Hz to 8 kHz on log-spaced bins) AND two binaural numbers - "
             "the loudness difference between the ears in decibels divided by "
             "20, and the arrival-time difference in milliseconds, which "
             "reaches 0.204 ms at most (the 7 cm ear gap at 343 m/s). Those "
             "two are the only channel carrying arrival time into the model. "
             "The 16-band pitch summary is NOT on this line: the encoder "
             "builds it from the same spectrum.",
    },
    "edge.sense_proprioception.enc": {
        "t": "proprioception → its encoder: the data",
        "w": "One row per [[handle]] every [[model-step]] - 26 rows x 12 numbers: "
             "the orientation [[quaternion]] (4), the position as (this part - "
             "the LEFT WRIST, which is row 0) divided by 200 cm (3), the "
             "velocity vector in cm/s divided by 500 (3), muscle freshness 0 "
             "to 1, and the row's own index divided by 25 so it lands between 0 "
             "and 1 like every other number in the row. It used to arrive raw, "
             "0 to 25: on this body that one channel carried 99.16% of the 312 "
             "numbers' total squared size and never changed between frames, so "
             "the encoder's first layer - which has no bias of its own - was "
             "reading a proprioception input that was 99 parts fixed ramp and 1 part body. "
             "Divided by 25 it is 15.83%. The encoder reads all 26 rows laid end "
             "to end as one 312-number list.",
    },
    "edge.sense_motor.enc": {
        "t": "motor sense → its encoder: the data",
        "w": "One 19-number row per [[handle]] every [[model-step]] - 26 rows: the "
             "achieved half (10 - position divided by 200 cm, orientation "
             "[[quaternion]], radius, strength 0-1, an is-active flag) and the "
             "[[executed-command|executed command]] half (9 - the same units). "
             "The current model step's command half is zeros until the next step "
             "fills it - one slot, one meaning.",
    },
    "edge.sense_intermind.enc": {
        "t": "intermind → its encoder: the data",
        "w": "5 placeholder rows of noise (one per future mind slot), each {intermind_row} "
             "numbers at this model size, every [[model-step]]. The numbers come "
             "from a bell curve centred at 0 with a spread of one hundredth, clipped at "
             "0, so half of them are exactly 0 and the rest sit under five hundredths. No "
             "lesson of its own grades this token - but the [[gradient]] of "
             "the other senses' lessons still reaches this encoder through the "
             "shared core.",
    },
    "edge.enc.ln_out": {
        "t": "encoder → shared LayerNorm: the data",
        "w": "One [[token]] per sense: a single list of {d_model} numbers per body "
             "([{n_bodies} x 1 x {d_model}] with this run's {n_bodies} bodies) - and already held at "
             "[[rms|RMS]] 1.0. The shared [[layernorm|LayerNorm]] runs INSIDE "
             "each encoder call, so no un-normalised token ever travels on a "
             "line; this box marks where that happens, not a separate step. "
             "Measured across the six senses and the bodies of the H81 probe: "
             "within 0.12% of that size on every token.",
    },
    "edge.ln_out.assembly": {
        "t": "LayerNorm → assembly: the data",
        "w": "The same {n_senses_word} tokens, now all at the same overall size "
             "([[rms|RMS]] 1.00) so no sense can shout over the others. This "
             "line is drawn only when --top-down-gate is off; with the gate on "
             "the tokens go through it first.",
        "h": "{n_senses_word} [1 x {d_model}] tensors, float32 even under this run's bfloat16 "
             "precision (a LayerNorm is computed in float32 under automatic "
             "mixed precision); unitless internal features; every one carries "
             "gradient back into its encoder; once per [[model-step]].",
    },
    "edge.ln_out.td_gate": {
        "t": "LayerNorm → top-down gate: the data",
        "w": "The {n_senses_word} raw encoder tokens, each at the same overall size "
             "([[rms|RMS]] 1.00). These are the tokens every lesson is graded "
             "against, whatever the gate does next.",
        "h": "{n_senses_word} [1 x {d_model}] tensors, float32 even under this run's bfloat16 "
             "precision (a LayerNorm is computed in float32 under automatic "
             "mixed precision); unitless internal features; "
             "gradient crosses into every encoder; once per [[model-step]].",
    },
    "edge.td_gate.assembly": {
        "t": "top-down gate → assembly: the data",
        "w": "The tokens the core will actually read. A sense whose gate is shut arrives untouched; a sense whose gate is open arrives as (1 - g) of the "
             "encoder plus g of the model's guess of NOW, with g at most half the token.",
        "h": "{n_senses_word} [1 x {d_model}] tensors, float32; gradient crosses, into the "
             "encoders through the (1 - g) part and into the gate's own 15 "
             "learned numbers through g - the prediction itself is detached, "
             "so nothing trains by agreeing with it. Once per [[model-step]]. The "
             "blend changes the token's overall size very little because the "
             "prediction points nearly the same way as the token: measured "
             "across this run's 19 snapshots, the blended audio token's RMS "
             "stayed between 94.6% and 101.3% of the raw token's size.",
    },
    "edge.assembly.block_s1": {
        "t": "assembly → the core: the data",
        "w": "The full sequence: {seq_len} tokens = {n_senses} senses x {context_length} [[model-step|"
             "model steps]] ({window_s} seconds of body time), each token carrying its "
             "sense tag and model step tag. This exact sequence flows through "
             "every stage of every block.",
        "h": "one [1 x {seq_len} x {d_model}] tensor, float32, unitless internal features, "
             "once per [[model-step]]. Only the newest 6 of the {seq_len} carry gradient "
             "back to the encoders; the other {seq_older} were stored with their "
             "gradient cut off (context_buffer holds ctx_update.detach()), so "
             "nothing this step teaches reaches an encoder through a "
             "remembered frame.",
    },
    "edge.block_s1.block_s2": {
        "t": "stage 1 → stage 2: the data",
        "w": "The same {seq_len}-token sequence: each token has now mixed in what "
             "the OTHER SENSES carried at its own model step. Added, not replaced "
             "- the token's own content survives every stage.",
        "h": "one [1 x {seq_len} x {d_model}] tensor. The arrow carries float32: the "
             "attention multiply itself runs in bfloat16 under this run's "
             "--precision, and adding the token back promotes the result to "
             "float32 again. Gradient crosses, in every one of the {n_blocks} blocks.",
    },
    "edge.block_s2.block_s3": {
        "t": "stage 2 → stage 3: the data",
        "w": "The same sequence again: each token has now also mixed in its "
             "own sense's earlier model steps, never later ones. Added, not "
             "replaced.",
        "h": "one [1 x {seq_len} x {d_model}] tensor, float32 on the arrow (bfloat16 "
             "inside the attention multiply), gradient crossing, once per "
             "[[model-step]]. In block 1 (the 2nd) and later the 'never later ones' is about where "
             "this stage looked, not about what those tokens hold: block 0's "
             "stage 3 already mixed later model steps into them.",
    },
    "edge.block_s3.block_ffn": {
        "t": "stage 3 → FFN: the data",
        "w": "The same sequence: each token has now read every other token - "
             "any sense, any model step in the window, earlier or later, since "
             "this stage has no mask. The FFN then digests each token on its "
             "own.",
        "h": "one [1 x {seq_len} x {d_model}] tensor, float32 on the arrow (bfloat16 "
             "inside the attention multiply), gradient crossing, once per "
             "[[model-step]].",
    },
    "edge.block_ffn.block_s1": {
        "t": "one block → the next: the data",
        "w": "The same {seq_len}-token sequence after one full block; the next block "
             "repeats all four parts on it at a deeper level.",
        "h": "one [1 x {seq_len} x {d_model}] tensor, float32, gradient crossing, once per "
             "[[model-step]]. Nothing is dropped between blocks - the sequence "
             "keeps its full length and its tags.",
    },
    "edge.block_ffn.final_norm": {
        "t": "the core → final LayerNorm: the data",
        "w": "The finished {seq_len}-token sequence, about to be read by every "
             "output.",
        "h": "one [1 x {seq_len} x {d_model}] tensor, float32, gradient crossing, once per "
             "[[model-step]]. Only the newest {n_senses_word} of the {seq_len} are ever read by an output; the other {seq_older} exist so those {n_senses_word} could look back at them.",
    },
    "edge.final_norm.heads": {
        "t": "core output → prediction heads: the data",
        "w": "The newest frame's {n_senses_word} tokens [{n_senses} x {d_model}]. Each of the {n_heads_word} heads reads all {n_senses_word} and writes its own sense's guess of the NEXT "
             "model step's token.",
    },
    "edge.final_norm.recon": {
        "t": "core output → recon decoders: the data",
        "w": "One token each, not all {n_senses_word}: decoder m reads the newest frame's "
             "m-th token [1 x {d_model}] and must re-produce that same sense's "
             "present from it - the guard.",
    },
    "edge.final_norm.motor": {
        "t": "core output → motor controller: the data",
        "w": "The {n_senses_word} newest tokens, flattened: {token_row} numbers. They are {token_row} of "
             "the controller's {motor_in} inputs - the motor sense's achieved handle state "
             "({n_handles} x 10) arrives on its own line from the motor sense, and the [[wellness]] "
             "slope ({n_handles} x 9) arrives on its own line from the wellness "
             "box.",
        "h": "the receipt's motor_controller._input_dim carries the exact "
             "count",
    },
    "edge.ln_out.wellness": {
        "t": "Encoder lists supplied to the learned wellness predictor",
        "w": "The learned wellness predictor reads current raw encoder lists, before top-down blending: vision, tactile, audio, proprioception and motor, each with {d_model} numbers. It places the five lists end to end and returns five predicted wellness scores. The source passes detached lists, so the matching loss updates the predictor without updating the encoders.\n\nThe fixed formulas have different inputs: a predicted-versus-measured token comparison, skin measurements, capacities, body-map statistics and measured positions. They do not all read one common transformer output.",
        "h": "mimo_bridge/lem_model.py: canonical_tokens argument to WellnessEvaluators.forward. lem/model/wellness.py: _sensory_indices and approximate_input assembly. [[approximate-evaluator|The learned predictor.]]",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "edge.wellness.motor": {
        "t": "Wellness to motor controller: position derivatives",
        "w": "The data is {n_handles} rows of nine derivatives per body, calculated at the current measured handle state. Only the motor-state position formula contributes a nonzero derivative. The motor controller receives this tensor as an additional input; it does not receive a pressure-withdrawal derivative from tactile comfort. [[wellness-gradient|The exact computation and its use.]]",
        "h": "mimo_bridge/lem_model.py: total_wellness, wellness_grad and motor_controller in LemLite.step.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "edge.motor.gaze": {
        "t": "motor → gaze: the data",
        "w": "The gaze [[handle]]'s row of the proposal - its orientation "
             "[[quaternion]] becomes one azimuth+elevation pair, clamped to "
             "the real eye ranges (azimuth ±45°, elevation "
             "-47° to +33°). That pair is then offset by the "
             "spontaneous saccade, blended with any live reflex snap, and "
             "finally turned into four actuator commands - one per eye joint, "
             "with the right eye's horizontal axis mirrored.",
    },
    "edge.motor.world": {
        "t": "motor → the world: the data",
        "w": "The [[executed-command]]: one row per [[handle]] x 9 numbers, "
             "latched every {motor_hold} model steps and held for {hold_ms} ms of "
             "body time - {hold_env_steps} env steps of {env_step_ms} ms in {run_name}. "
             "What the body receives is not the proposal. "
             "Between the controller and the actuators the command passes, in "
             "this sequence: the growth-stage strength limit; positions clamped "
             "to -1..1 and multiplied by 200 cm; the reach clamp, which "
             "rewrites each target as the handle's current position plus a "
             "step of at most 40 cm; [[babble]] noise ONLY on steps the "
             "planner did not author ({babble_latch_share}); a smoothing blend of "
             "35% new command with 65% previous command; and, on a scheduled rest "
             "window (--rest-control scheduled with --babble-bouts real: 25% "
             "to 42% of the H81 probe's steps; {scheduled_rest_note}), a substitution of every "
             "handle's position and rotation by the body's own achieved pose, "
             "so it holds its posture instead of being driven. The eyes are "
             "not in this hold: their command is sent fresh every step and "
             "written back into the gaze handle afterwards. The decoder then "
             "turns the command into 90 actuator numbers, rescales each handle's "
             "group as a unit if it exceeds its rail, and clips; MuJoCo's own "
             "actuation model then rewrites 18 of those 90 during the "
             "substeps.",
    },
    "edge.intention.motor": {
        "t": "planner → motor: the data",
        "w": "The plan's first command: {n_handles} handle rows x 9 numbers per body (the kept "
             "[[candidate-command|candidate command]]). The motor controller never reads it: it "
             "REPLACES that controller's command downstream, on the way through the "
             "[[execution-filter|execution filter]] to the body. The controller's command, where each planner "
             "search starts, runs the other way on a line of its own, out of the controller's bottom edge. "
             "{planner_line_rule}",
    },
    "edge.wellness.intention": {
        "t": "Wellness supplied to remembered-goal selection",
        "w": "While the intention planner is enabled, each body's current sensory state is stored with a raw value. With normalized ranking enabled, it also stores each enabled channel's change from recent history, plus measured harm.\n\nRaw ranking uses the combined stored value, including configured extra contributions. Normalized ranking compares and weights the channel changes. The chosen entry supplies a sensory target for command search. [[goal-ring-entry|What the entry holds]]; [[wellness-ranking|how ranking works]]. {wellness_rank_run}",
        "h": "mimo_bridge/lem_model.py: goal_ring.append, _rank_store_moment and _normalized_goal_values.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "edge.curiosity.intention": {
        "t": "Curiosity supplied to remembered-goal selection",
        "w": "One drive number between zero and one, shared by the bodies, is calculated each model step from batch prediction error. It affects remembered-goal value only when curiosity-goal is on. Normalized ranking adds it as a separately scaled, weighted channel; raw ranking uses the configured additive or multiplicative rule. [[curiosity-mode|Exact rules]]. {curiosity_mode_run}",
        "h": "lem/model/curiosity.py: CuriositySystem.compute_drive. mimo_bridge/lem_model.py: _wellness_channel_vector and goal-ring assembly.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "edge.memory.final_norm": {
        "t": "memory → the core's output: the data",
        "w": "The recalled bias: one [{n_senses} x {d_model}] addition to the newest [[model-step]]'s {n_senses_word} finished tokens, applied AFTER the core's last "
             "LayerNorm. It never passes through an attention stage, so the "
             "memory cannot be reasoned over across the window - it shifts the "
             "finished representation of NOW. What reads it: the prediction "
             "heads, the reconstruction decoders, the voice head, the "
             "[[wellness]] scorer, and the motor controller - for which those {n_senses_word} tokens are the WHOLE latent input, {token_row} of the {motor_in} numbers "
             "it reads, so the recall is not diluted across the window before "
             "it reaches the body's commands. It is never zero once the "
             "readout has trained: there is no gate and no emptiness check "
             "between the memory and this line. {memory_bias_vs_token}",
        "h": "lem/model/forward_model.py applies final_norm first, then adds memory_bias to the last {n_senses_word} tokens only. Probed: with a bias "
             "supplied, tokens 0..761 change by exactly zero and only the newest {n_senses_word} move. float32; gradient crosses into the memory's own "
             "{memory_params} learned numbers, and its inputs are detached, so nothing "
             "flows back to the encoders.",
    },
    "edge.memory.ring_gru": {
        "t": "event ring → the recurrent network: the data",
        "w": "Every retained event, oldest first - up to 130 rows with a full "
             "ring: the 128 kept events, the biggest change so far inside the "
             "open half second, and the present model step. Each row is the "
             "event's {memory_event} numbers plus TWO time numbers, so the network "
             "reads {memory_event_timed} numbers per row: the seconds of body time since the "
             "previous kept event, and the event's age in seconds of body time "
             "right now. Both pass through log(1 + seconds), so 0 s reads "
             "zero, and the half second of one sampling interval and the 64 s "
             "age of the oldest event in a full ring - 128 times apart in "
             "seconds - read about ten times apart after that squeeze. The whole window is "
             "read again from its first row on every call, and with the "
             "delayed next-frame lesson on that is twice per [[model-step]]: once "
             "on the previous model step's stored window and once on this "
             "model step's. Nothing is carried from one read to the next except "
             "the summary of events already pushed out of the ring.",
        "h": "lem/model/persistent_memory.py encode(): delta is the gap "
             "between neighbouring retained timestamps, age is now minus each "
             "timestamp, both log1p-compressed and joined to the event before "
             "input_projection, a Linear of input_size + 2; self.recurrent "
             "then runs over every row. The two calls per model step are "
             "mimo_bridge/lem_model.py step() on self._nf_memory and "
             "mimo_bridge/memory_integration.py observe() on this model step's "
             "snapshot; a command boundary while training adds two more "
             "inside the tenure lesson",
        "f": "reading 'GRU summary' as a running summary that takes in one "
             "new event per model step. Only the summary of EVICTED events is "
             "carried forward, and that carry is detached, so no lesson "
             "reaches an event once it has left the ring",
    },
    "edge.ln_out.memory": {
        "t": "shared LayerNorm → persistent memory: the raw tokens",
        "w": "What one stored event IS: {memory_event} numbers, read every [[model-step]] "
             "and detached so no lesson flows back along this line. This line carries the first part: the {n_senses_word} senses' RAW [[token|tokens]] ({n_senses} x {d_model} = {token_row} - what the senses "
             "reported, before the top-down gate rewrites anything). The rest arrives on lines of its own: 5 bodily "
             "needs between 0 and 1 (pain, pleasure, muscle capacity, squashed "
             "pressure, squashed rub) from the tactile and the motor sense, with 5 flags saying which of those five were "
             "really measured; the {n_handles} x 9 numbers of the [[executed-command]]; and "
             "{d_model} numbers summarising the goal the planner last aimed at. The event's time stamp, kept "
             "beside those numbers, comes from the world.",
        "h": "mimo_bridge/memory_integration.py observe(): the tokens come in "
             "as tokens_raw, and the whole vector is detached before the "
             "memory sees it",
        "f": "reading the goal's {d_model} numbers as an instruction from outside - it is the "
             "model's own last plan, averaged over its senses",
    },
    "edge.memory.needs": {
        "t": "memory → needs forecast: the data",
        "w": "The memory's SECOND trained output. From the previous model step's "
             "summary of the memory (192 numbers) plus the command just "
             "executed (26 [[handle|handles]] x 9), it forecasts five numbers "
             "between 0 and 1 for the body's next pain, pleasure, muscle "
             "capacity, pressure and rub. It is graded against what those five "
             "actually measured a model step later, at a tenth of full [[loss|lesson]] weight. "
             "The inspector shows the result as the needs-forecast error, a "
             "mean of squares whose square root is the typical miss, where 0 "
             "means all five came out exactly as forecast: this run measured a "
             "typical miss of 10% to 24% of each need's 0-to-1 range.",
        "h": "lem/model/persistent_memory.py predict_needs: 426 numbers in, 96 "
             "hidden, 5 out through a sigmoid; graded in "
             "mimo_bridge/lem_model.py and reported as "
             "persistent_memory.next_bodily_state_error",
        "f": "this is the only memory number the audited run shows moving; the "
             "recall itself is graded by nothing of its own",
    },
    "edge.ln_out.replay": {
        "t": "tokens → replay memory: the data",
        "w": "Every [[model-step]], one row is written into the ring: the {n_senses_word} raw [[token|tokens]] that left the shared output "
             "[[layernorm|LayerNorm]] ({n_senses} x {d_model}, kept at half precision), a "
             "[[source-label|real/stale/synthetic]] flag per sense, and the growth "
             "phase's reach and strength limits. Three more parts of the row arrive on lines of their own: the "
             "achieved pose of every [[handle]] ({n_handles} x 10) from the motor sense, the persistent memory's "
             "state from the memory, and the [[executed-command]] from the motor controller. That command is not "
             "known when the row is written - it is filled in one model step later, once the command that left here "
             "has been chosen. Everything is stored DETACHED, which is why {n_heads_word} of the {n_senses_word} [[encoder|encoders]] cannot learn from re-study. "
             "No [[gradient]] crosses this line.",
        "h": "mimo_bridge/replay.py: the ring holds {replay_size} model steps per body in "
             "CPU memory; model.step() writes to it, and an --eval-only run "
             "never builds it at all",
    },
    "edge.replay.final_norm": {
        "t": "Replay memory to weights: a separate training update",
        "w": "When replay is enabled during training, a replay pass uses remembered transitions to calculate its own loss and parameter gradients, then calls AdamW. It updates the core, prediction heads, motor controller, persistent-memory bias readout and motor encoder. It has a separate loss from the live model step. {learning_replay_run}",
        "h": "train_mimo.py: replay creation and replay_update schedule. lem_model.py: replay_update.",
        "f": "A configured replay pass can be skipped when there are too few usable windows. The selected snapshot and replay chart show recorded passes; configuration alone does not prove a pass ran."
    },
    "edge.ln_out.imagination": {
        "t": "tokens → imagination: the data",
        "w": "The last few [[model-step|model steps]] of [[token|tokens]] - one "
             "{d_model}-number row per sense per body, one model step more than the "
             "chain is deep. They are kept TWICE. The set the [[transformer]] "
             "actually read that model step, which can carry up to half of the "
             "model's own expectation blended in by the top-down gate, is "
             "where the chain STARTS (with the gate on, that set arrives on its own line from the "
             "top-down gate; with it off, it is this line's set); the untouched measured set on this "
             "line is what every "
             "depth is GRADED against, so the chain cannot score by agreeing "
             "with itself. Each model step's [[source-label|source labels]] ride "
             "along, so a sense that was not measured is neither graded nor "
             "fed back.",
        "h": "enc_history (started from) and enc_history_raw (graded against) "
             "in mimo_bridge/lem_model.py, appended every step and detached",
        "f": "the two sets are not the same starting point when a gate is "
             "open. The do-nothing baseline this chain must beat, which is "
             "what licenses the planner's horizon, is built from the RAW "
             "tokens while the chain starts from the blended ones - in this "
             "run audio's blend ran 10 to 28 parts of the token in 100, so on audio the two "
             "sides of that comparison begin from different states",
    },
    "edge.motor.imagination": {
        "t": "executed command → imagination: the data",
        "w": "The [[executed-command|command the body really carried out]] at each "
             "of those [[model-step|model steps]] - after growth limits, the reach "
             "clamp, [[babble]] noise and smoothing - NOT the controller's "
             "proposal. One row per [[handle]] x 9 numbers, in encoder units "
             "(positions already divided by 200 cm). Each hop re-encodes it "
             "together with the body's measured pose into one motor "
             "[[token]], which is why the motor [[encoder]] is the one encoder "
             "this lesson reaches.",
        "h": "motor_history in mimo_bridge/lem_model.py; the per-hop re-encode "
             "is in lem/model/predictive_state.py",
    },
    "edge.imagination.intention": {
        "t": "imagination → planner: the horizon licence",
        "w": "Not a tensor - a permission. At each depth the chain is compared "
             "with assuming nothing changed, and a slow running average of how "
             "often it WINS decides how many model steps ahead the planner may "
             "aim: depth k opens only while that average stays above 50 wins "
             "in 100, where 0 means the chain never beats doing nothing and 100 means it "
             "always does. The depths count upward, so the first one to fall "
             "to 50 closes every depth above it and the horizon drops to 1 "
             "in a single step. The cap is --imagination-steps, {imagination_cap}.",
        "h": "the win averages and the licensed depth in "
             "mimo_bridge/lem_model.py, reported per snapshot in the plan "
             "popup; the comparison itself is in lem/live_test.py",
        "f": "reading a licence of 3 as three accurate model steps. It says only "
             "that the chain beat doing nothing more than half the time; how "
             "close it came to the real next model steps is a separate number",
    },
    "edge.final_norm.voice": {
        "t": "core output → voice head: the data",
        "w": "The newest frame's {n_senses_word} rows, one per sense - [{n_bodies} bodies x {n_senses} x {d_model}] "
             "in this run - crossing once per [[model-step]]. They are the "
             "final [[layernorm]]'s output with the persistent memory's "
             "recalled bias already added: the memory adds that bias to the "
             "newest frame AFTER the final LayerNorm, so it rides this line "
             "too. Same model step as the core's own pass, not the model step before - "
             "the line is drawn dashed only because it is routed around the "
             "boxes. Out of the head come 80 numbers per body, each held "
             "between 0 and 1 by a sigmoid - the reserved width of the "
             "shared-memory action slot, wider than the 56-number vocal-tract "
             "parameter set this repo defines; in this world they make no sound, "
             "and nothing but this page reads them.",
    },
    "edge.wellness.hself": {
        "t": "Fixed-score sum supplied to the hierarchical observer",
        "w": "Add the six fixed scores for each body, then average those sums across the {n_bodies} bodies. The hierarchical observer compares changes in this one number with changes in its slot-presence measurements. This averaging loses both the individual score and body identities. Movement and the normalized ranking are not included.",
        "h": "mimo_bridge/lem_model.py: hierarchical observer call in step; last_wellness_grounded values are summed then averaged.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The sum does not have a zero-to-one range. Tactile comfort tops out at 85%, and self-touch can add its configured weight. [[wellness-link|How the observer measures an association.]]"
    },
    "edge.ln_out.hself": {
        "t": "shared LayerNorm → hierarchical self: the data",
        "w": "This [[model-step]]'s RAW [[token|tokens]], one {d_model}-number row per "
             "active sense, straight off the shared "
             "[[layernorm|LayerNorm]] - the same set every lesson is graded "
             "against, NOT the gate-blended set the core reads. The observer "
             "lays them end to end into one long row and scales it to length "
             "1 before matching it against its slot centres. Read inside the "
             "same step, read-only; nothing flows back.",
        "h": "mimo_bridge/lem_model.py hands the observer model.last_tokens, "
             "which is built from tokens_raw, together with this step's six "
             "grounded [[wellness]] scores",
        "f": "the line is drawn solid on purpose. The page's legend reads a "
             "dashed line as influence between steps, and this read happens "
             "inside the same step() call as the forward pass",
    },
    "edge.curriculum.motor": {
        "t": "curriculum → motor: the data",
        "w": "The phase's limits, applied to EVERY command before the body gets "
             "it: the step away from the current pose is clamped to 40 cm x "
             "rom_limit, the strength channel is clamped to strength_limit, and "
             "a handle mask says which [[handle|handles]] may move. No gradient "
             "flows back along this line.",
        "when": "Every [[model-step]], not only at a phase change.",
        "h": "lem_model.py execution_transform plus the reach clamp; the mask is "
             "built from a 24-name list in lem/training/curriculum.py and padded "
             "with True to the body's handle count by body_layout.extend_limits "
             "(26 on this run's trunk layout), so every handle is active in every "
             "phase today.",
        "f": "reading it as the path [[babble]] noise takes to the body: the "
             "noise is added only on a model step where the planner produced no "
             "plan, so with --motor-hold equal to --intention-every every "
             "latched command is the planner's and the noise never acts.",
    },
    "edge.curriculum.world": {
        "t": "curriculum → the world: the data",
        "w": "At a phase change only, and only what actually changed: the "
             "body's age in months, and the MOTOR delay in env steps of 10 ms. "
             "Each world worker rebuilds its MuJoCo body at that age. There is "
             "no sensory delay to send: MIMo applies one inside _get_obs and "
             "this bridge reads live MuJoCo sensors, so it reached nothing "
             "(measured: 4 steps against 0 left tactile, proprioception and "
             "vestibular identical on 12 of 12 steps, a maximum difference of "
             "exactly zero), and BridgeEnv now refuses a nonzero setting.",
        "when": "Once per phase change, and nothing is sent when the new phase "
                "changes none of them.",
        "h": "train_mimo.py env_kwargs_for plus the phase-change branch; the "
             "worker regrows the body through MIMo's adjust_mimo_to_age (growth "
             "model v2) in mimo_bridge/env.py",
        "f": "assuming a phase change always rebuilds: with --age-binding none "
             "and --delay-binding off nothing differs, so nothing is sent.",
    },

    # ---------------- the input check of Oct 15, 2026 ----------------
    # Each line below was drawn after its data flow was read in the code; the claim ledger in
    # docs/explainer_inputs_2026-10-15/ quotes the code line behind every statement here.
    "edge.sense_motor.motor": {
        "t": "motor sense → motor controller: the achieved handle state",
        "w": "The motor sense's reading of where the body IS, taken without the motor encoder: {n_handles} "
             "[[handle|handles]] x 10 numbers per body - position x, y, z in centimetres divided by 200, "
             "orientation as 4 numbers, radius, strength and an active flag - from this [[model-step]]'s frame. "
             "The motor controller lays the {n_handles} x 10 numbers end to end and joins them to the core's "
             "{token_row} numbers and the [[wellness]] slope's {n_handles} x 9 numbers: one row of {motor_in} "
             "inputs. The same reading is the motor lesson's target: the controller re-run on the previous "
             "model step's inputs is graded against this model step's achieved position and orientation "
             "(the first 7 numbers per handle; the target's radius and strength columns are the controller's "
             "own output, detached).",
        "h": "mimo_bridge/lem_model.py LemLite.step: handle_state = frame.get(\"motor\") (the motor sense's "
             "10-wide achieved rows), self.motor_controller(latents, handle_state, wellness_grad=...), and "
             "motor_target = handle_state[:, :, :9]; lem/model/motor_controller.py forward: flat_handles = "
             "handle_state.reshape(B, -1) and motor_input = torch.cat([pooled, flat_handles, flat_wgrad])",
        "f": "reading the controller's input as the motor TOKEN: the token is the encoder's output from the "
             "19-number row (achieved 10 + command 9); the controller reads the 10 achieved numbers directly, "
             "and the command half is not on this line.",
    },
    "edge.sense_motor.wellness": {
        "t": "motor sense → wellness: handle positions and capacities",
        "w": "Two of the six fixed wellness scores read the motor sense without the encoder. The "
             "handle-position score (motor_state) takes the {n_handles} achieved positions, subtracts their "
             "mean, so each is a displacement from the middle of the body's handles, and scores them against "
             "the phase's reach limit; its derivative with respect to these positions is the wellness slope "
             "the motor controller reads. The capability score is the mean of the first 10 of the motor "
             "sense's capability numbers (the body-set capacities, each 0 to 1).",
        "h": "mimo_bridge/lem_model.py LemLite.step: mc = handle_state[:, :, :9], mc[..., 0:3] minus its mean "
             "over the handles, motor_commands = mc.requires_grad_(True), and wellness_grad = "
             "torch.autograd.grad(total_wellness, motor_commands); _batch_frames: extras[\"capability\"] = the "
             "frames' motor_capability; lem/model/wellness.py _formula_capability: cf[:, :10].mean(...)",
    },
    "edge.sense_tactile.wellness": {
        "t": "tactile sense → wellness: the skin's raw readings",
        "w": "Two of the six fixed wellness scores read the tactile sense without the encoder. Skin comfort "
             "(tactile_comfort) takes the features of EVERY measured skin patch - pressure, shear, slip, area "
             "and vibration, before the encoder's contact limit - with each patch's sensitivity, and keeps the "
             "worst local discomfort. Self-contact (self_touch) takes the tactile point cloud and pays for an "
             "increase, since the previous model step, in the count of the body's own parts on its skin, "
             "multiplied by its gate. Self-contact is also multiplied by the --self-touch-wellness weight; at weight "
             "0 it adds 0 at every model step ({self_touch_weight_run}). Under --contact-profile mechanical_v2 to "
             "mechanical_v4, skin comfort also reads each patch's 3-number mechanical contact history from the frame's "
             "mechanical_contact record ({contact_profile_run}).",
        "h": "mimo_bridge/lem_model.py _batch_frames: extras[\"raw_tactile\"] = {features: "
             "batch_skin_features(frames), sensitivity}; LemLite.step hands it to WellnessEvaluators as "
             "raw_tactile and sets wellness_out['grounded']['self_touch'] = _self_touch_wellness(frame['tactile']); "
             "lem/model/wellness.py: tactile_comfort = _formula_tactile_comfort(raw_tactile) "
             "(lem/model/contact_wellness.py contact_response)",
    },
    "edge.sense_tactile.memory": {
        "t": "tactile sense → persistent memory: four measured needs",
        "w": "Four of the five measured needs stored with each memory event come from the tactile sense's raw "
             "skin features: pain and pleasant touch (contact_response's two readings), and the largest "
             "pressure and the largest shear over the skin patches, which arrive already divided by 100,000 Pa and "
             "by 50,000 Pa, each squashed by x / (1 + x). A flag beside "
             "each says it was measured. The same five needs are the target of the memory's needs forecast, "
             "graded at a tenth of full weight.",
        "h": "mimo_bridge/memory_integration.py measured_needs: r = contact_response(raw['features'], ...), "
             "pain, pleasant = r['pain'], r['pleasant'], pressure = raw['features'][..., 0].amax(1), shear = "
             "raw['features'][..., 1].amax(1), values = torch.cat([pain, pleasant, capacity, pressure/(1+pressure), "
             "shear/(1+shear)]); mimo_bridge/lem_model.py: memory_loss from predicted_needs - current_needs",
    },
    "edge.sense_motor.memory": {
        "t": "motor sense → persistent memory: the capacity need",
        "w": "The fifth measured need of each memory event: the mean of the first 10 of the motor sense's "
             "capability numbers, clipped to 0 to 1 - the same quantity as wellness's capability score. Its "
             "flag says it was measured.",
        "h": "mimo_bridge/memory_integration.py measured_needs: cap = extras.get('capability'), capacity = "
             "cap[:, :10].float().mean(-1, keepdim=True).clamp(0, 1)",
    },
    "edge.sense_tactile.self_state": {
        "t": "tactile sense → self perception: per-site skin fields",
        "w": "One row per skin site per body: the site's position (3 numbers) and the first 8 skin features of "
             "that site (pressure, shear, slip, area, vibration), joined to the world's three body-function "
             "fractions of the same site into one 14-number row. Self perception refuses a frame without "
             "them.",
        "h": "mimo_bridge/self_state_integration.py batch_observation: touch = frame_skin_features(f), pos = "
             "f.touch_pos, sites[i, :len(q)] = torch.cat([pos, touch[:len(q), :8], q], -1)",
    },
    "edge.world.memory": {
        "t": "the world → persistent memory: the time stamp",
        "w": "The frame's simulated time in seconds, one number per body per [[model-step]]. The memory stamps "
             "each event with it; the recurrent network then reads, for every retained event, log(1 + the "
             "seconds since the previous kept event) and log(1 + the event's age now in seconds).",
        "h": "mimo_bridge/memory_integration.py observe: now = torch.tensor([f.sim_time for f in frames]) and "
             "model.memory_history.observe(features, now, ...)",
    },
    "edge.world.contact_head": {
        "t": "the world → contact head: the contact report",
        "w": "The world's report for one [[model-step]]: per body, the state of each of the 53 skin patches "
             "(0 nothing touching, 1 something outside the body, 2 one of the body's own parts, 3 both) - the "
             "union over the model step's 5 env frames when the trainer merged them, else the last frame's - "
             "with the self-contact onsets and the count of onsets with outside things. The contact head reads the PREVIOUS model step's patch states as "
             "two flags per patch (an outside thing, an own part) beside the core's state and the executed "
             "command, and its lesson grades the probabilities it formed against THIS model step's states. "
             "The onsets are scored by the contact-control [[wellness]] channel.",
        "h": "mimo_bridge/env.py _contact_report (contact_patch_state, self_contact_onsets, "
             "contact_onsets_world) and merge_burst_contact_report (contact_patch_state_burst); "
             "mimo_bridge/lem_model.py _contact_report, _contact_prev_input and _contact_lesson: head(latent, "
             "ofeat, prev_in) graded by ContactHead.loss against state_bits(cur)",
    },
    "edge.td_gate.intention": {
        "t": "top-down gate → planner: the gated tokens",
        "w": "This [[model-step]]'s {n_senses} tokens of {d_model} numbers per body AFTER the top-down gate: "
             "each sense's raw token with a share g of (the delayed re-run's prediction minus that token) added, "
             "g from none to one half. With the planner built (--intention on), every model step the [[goal-ring]] "
             "stores them with the model step's raw goal value; on a planning step the same set is the planner's "
             "current state, and the goal is the stored set of the chosen goal ring entry. The planner's "
             "search therefore compares gated tokens, while every lesson is graded against the raw ones. Under "
             "--hand-goals memory the hand stage's goal is also the chosen entry's stored handle pose (the motor "
             "sense's achieved numbers), on its own line from the motor sense.",
        "h": "mimo_bridge/lem_model.py LemLite.step: tokens[m] = tokens[m] + g * (zp - tokens[m]) with g = half of "
             "sigmoid(td_gates[m](...)); goal_ring.append(({m: tokens[m] ...}, wball)); current_state = "
             "{m: tokens[m] ...}; the goal is gathered from the goal ring's stacked tokens",
    },
    "edge.ln_out.intention": {
        "t": "shared LayerNorm → planner: the tokens",
        "w": "This [[model-step]]'s {n_senses} raw tokens of {d_model} numbers per body (no top-down gate in "
             "this drawing, so the tokens the core reads are the shared LayerNorm's). With the planner built "
             "(--intention on), every model step the [[goal-ring]] stores them with the model step's raw goal "
             "value; on a planning step the same set is the planner's current state, and the goal is the stored "
             "set of the chosen goal ring entry. Under --hand-goals memory the hand stage's goal is also the chosen "
             "entry's stored handle pose (the motor sense's achieved numbers), on its own line from the motor sense.",
        "h": "mimo_bridge/lem_model.py LemLite.step: goal_ring.append(({m: tokens[m] ...}, wball)) and "
             "current_state = {m: tokens[m] ...}; with --top-down-gate off tokens is the encoders' output",
    },
    "edge.td_gate.imagination": {
        "t": "top-down gate → imagination: where the chain starts",
        "w": "The tokens after the top-down gate, kept every [[model-step]] in the imagination chain's history "
             "(one model step more than the chain is deep). The chain STARTS from the oldest kept set and "
             "feeds its own predictions forward; a sense that was not a real reading at a depth is put back "
             "to the kept set of that depth. Every depth is graded against the RAW set, which arrives on the "
             "line from the shared LayerNorm.",
        "h": "mimo_bridge/lem_model.py: enc_history.append({m: tokens[m] ...}) after the gate, "
             "enc_history_raw.append({m: tokens_raw[m] ...}); lem/live_test.py _run_imagination_compiled: "
             "imagined = dict(encoder_output_history[0]) and target = target_history[s + 1]",
    },
    "edge.memory.replay": {
        "t": "persistent memory → replay memory: the memory state",
        "w": "Every [[model-step]] the persistent memory's state (the recurrent network's encoding of the "
             "event window, per body) is written into the replay ring beside that model step's row. A "
             "re-study update reads the state stored with each window's last input model step and turns it "
             "into the memory bias the core's re-run adds after its final LayerNorm.",
        "h": "mimo_bridge/lem_model.py LemLite.step: replay.add(..., memory_state=self._memory_state); "
             "mimo_bridge/replay.py: memory_states[:, i] = memory_state, read back as "
             "memory_states[env, idx[-2]]; lem_model.py replay_update: persistent_memory.bias(w['memory_state'], "
             "site=\"replay\")",
    },
    "edge.motor.enc_motor": {
        "t": "executed command → motor encoder: the command half of the motor row",
        "w": "The [[executed-command]] of the interval that just ended ({n_handles} rows x 9 numbers: position "
             "in centimetres divided by 200, orientation 4, radius, strength) fills the command half of the "
             "motor row [achieved 10 + command 9]. Three uses: the context back-fill re-encodes the previous "
             "model step's motor slot from [previous achieved pose, this command]; the delayed next-model-step "
             "lesson re-encodes the previous model step's row with it, so that lesson is S(t-1) + A(t-1) -> "
             "S(t); and re-study rebuilds each window's motor token from [achieved, executed command]. The "
             "live pass's own row carries zeros in the command half.",
        "h": "mimo_bridge/lem_model.py LemLite.step: x_bf = torch.cat([last_achieved[..., :MOTOR_ACHIEVED_F], "
             "command_for_encoder(exec_t)]) written into context_buffer; nf_raw[\"motor\"] = "
             "torch.cat([mt[..., :MOTOR_ACHIEVED_F], command_for_encoder(exec_t)]); the live row's "
             "torch.zeros(..., MOTOR_COMMAND_F); command_for_encoder divides the position by "
             "COMMAND_POSITION_RANGE_CM = 200",
    },
    "edge.motor.memory": {
        "t": "executed command → persistent memory: the command columns",
        "w": "The [[executed-command]] of the interval that just ended, {n_handles} x 9 numbers per body with "
             "the position divided by 200, stored as each memory event's command columns (zeros when no "
             "command was supplied). The needs forecast also reads it: the memory's previous state plus this "
             "command predicts the five needs.",
        "h": "mimo_bridge/memory_integration.py observe: action = command_for_encoder(executed).reshape(b, -1); "
             "mimo_bridge/lem_model.py: persistent_memory.predict_needs(nf_memory_state, "
             "command_for_encoder(exec_t))",
    },
    "edge.intention.memory": {
        "t": "planner → persistent memory: the last goal",
        "w": "The goal the planner last aimed at: the chosen goal ring entry's tokens from the most recent "
             "planning step, averaged over the senses into {d_model} numbers per body, stored as each memory "
             "event's goal columns; zeros before the first plan.",
        "h": "mimo_bridge/memory_integration.py observe: goal = model._last_plan_inputs, goal_summary = "
             "torch.stack(goal tokens).mean(0); mimo_bridge/lem_model.py stores \"goal\" in _last_plan_inputs "
             "on each planning step",
        "f": "reading it as an instruction from outside: it is the model's own last plan.",
    },
    "edge.final_norm.self_state": {
        "t": "the core's output → self perception: the internal state",
        "w": "The final LayerNorm's {n_senses} newest tokens, with persistent memory's bias added, averaged "
             "over the senses into {d_model} numbers per body. It is stored after this [[model-step]]'s self "
             "perception lesson and read as the 'internal' input of the next model step's perception. "
             "Detached: no self-perception lesson reaches the core through it.",
        "h": "mimo_bridge/lem_model.py LemLite.step: self.self_history.internal = "
             "last_step.detach().float().mean(1); mimo_bridge/self_state_integration.py observe: "
             "internal=history.internal; lem/model/forward_model.py: final_norm, then memory_bias added to the "
             "newest tokens, last_step = sequence[:, -n_modalities:]",
    },
    "edge.ln_out.heads": {
        "t": "shared LayerNorm → prediction heads: the residual base",
        "w": "With --residual-prediction on, each prediction head outputs a CHANGE, and its sense's raw token "
             "({d_model} numbers per body) is added to it, and, under --heads-all-slot-path on, the all-slot "
             "path's output too: a head that outputs zeros, with that path also at zero, predicts that nothing "
             "changes. The live pass adds this [[model-step]]'s raw tokens. The delayed next-model-step lesson "
             "adds the previous model step's tokens, re-encoded this model step from the cached readings with "
             "the executed command in the motor row; that sum is the prediction the heads' three lines out "
             "carry.",
        "h": "lem/model/prediction_heads.py PredictionHeads.forward: out = out + _residual_base(current, ...) "
             "when residual is on; mimo_bridge/lem_model.py: current = tokens_raw (live) and current = nf_in "
             "(the delayed lesson); train_mimo.py --residual-prediction on sets model.residual_prediction",
    },
    # the independent review of Oct 15, 2026 (S1, m8, m12): the planner's starting command and stored pose, the replay
    # row's pose and command, the world's consequence count
    "edge.motor.intention": {
        "t": "motor controller → planner: where each search starts",
        "w": "This [[model-step]]'s motor controller output, {n_handles} [[handle|handles]] x 9 numbers per body, "
             "handed to the planner on a planning step as the command its search starts from: the 20 Adam updates "
             "change a copy of it, and the [[candidate-command|candidate commands]] the search keeps begin there. The "
             "controller's own command is also recorded with the plan's inputs, so a replanning of the same "
             "decision starts from the same command.",
        "h": "mimo_bridge/lem_model.py LemLite.step: _plan_chain(current_state, goal_state, "
             "motor_out.detach().float(), ...) and _last_plan_inputs['motor']; _plan_chain sets motor = current_motor",
    },
    "edge.sense_motor.intention": {
        "t": "motor sense → planner: the goal's stored pose",
        "w": "Under --hand-goals memory or --physical-goals memory: every model step the [[goal-ring]] stores the "
             "motor sense's achieved handle state ({n_handles} x 10 numbers per body) beside the entry's tokens, and "
             "on a planning step the chosen entry's stored pose becomes the goal's pose (this model step's achieved "
             "state stands in for a body whose entry holds none). Under --hand-goals memory the hand stage aims the "
             "fingers at that pose; under --physical-goals memory the body head's search does.",
        "h": "mimo_bridge/lem_model.py LemLite.step: _goal_ring_pose.append(handle_state ...) and "
             "selected_pose(self._goal_ring_pose, _goal_idx, handle_state) into the decision's hand_goal and "
             "physical_goal; lem/model/physical_goal.py selected_pose",
    },
    "edge.world.intention": {
        "t": "the world → planner: the consequence count",
        "w": "Under --consequence-w above 0: per body, the number of kinds of self-relevant consequence the world's "
             "task report shows this [[model-step]], 0 to 3 (a hand's contact with the mobile, the ball or the box or a "
             "hand press; an object moved by any body part; the ball centred in the eye or the hand in the fovea), "
             "times the --consequence-w weight, added to the value the [[goal-ring]] stores with the model step.",
        "h": "mimo_bridge/lem_model.py: consequence_kinds(f.meta['task']) and wball = wball + consequence_weight * cq "
             "in LemLite.step; train_mimo.py --consequence-w, whose default is 0, sets consequence_weight",
    },
    "edge.motor.replay": {
        "t": "executed command → replay memory: the row's command",
        "w": "The [[executed-command]] of the interval that just ended ({n_handles} x 9 numbers per body), written "
             "into the replay ring beside the row of the model step it was sent from; re-study rebuilds each window's "
             "motor token from the stored achieved pose and this command.",
        "h": "mimo_bridge/lem_model.py LemLite.step: replay.add(..., executed=exec_t, ...); replay_update rebuilds the "
             "motor token from [achieved, executed]",
    },
    "edge.sense_motor.replay": {
        "t": "motor sense → replay memory: the row's achieved pose",
        "w": "The motor sense's achieved handle state, {n_handles} x 10 numbers per body, stored in each replay row: "
             "re-study rebuilds the motor token from it with the executed command, and grades the motor controller "
             "against the next row's pose.",
        "h": "mimo_bridge/lem_model.py LemLite.step: replay.add(tokens_raw, handle_state, ...); replay_update reads "
             "w['achieved']",
    },

    # ---------------- drawn text that is not a box or a line ----------------
    "diagram.numbered_chain": {
        "t": "the numbered chain, and what it leaves out",
        "w": "The circles number one [[model-step]]'s LIVE pass in the sequence the "
             "program runs it: the senses arrive, ONE call encodes all {n_senses_word} together (which is why every sense box carries the first two "
             "numbers, not vision alone), the persistent memory reads those "
             "raw tokens, the top-down gate may blend the model's own guess of "
             "NOW into them, the tokens are assembled, the blocks run, the "
             "final [[layernorm|LayerNorm]], the prediction heads, the motor "
             "controller that writes the command the body executes, and the "
             "world. The memory and the gate are numbered only in a run that "
             "has them. The shared output LayerNorm carries no number of its "
             "own: it runs INSIDE each encoder's call and comes back with it, "
             "so numbering it would make one operation carry two numbers.",
        "when": "The live pass is not the only pass. Before it, the delayed "
                "next-frame lesson re-runs the encoders, the core and the "
                "heads on the PREVIOUS frame with the command that was actually "
                "executed - that pass, not this one, produces [[causal_pe]]. "
                "The imagination chain runs the core again once per depth, and "
                "re-study runs it again on remembered model steps.",
        "f": "counting the numbered boxes as everything the model does in a "
             "model step: the pass that teaches it most is unnumbered and runs "
             "over five of the same boxes",
    },
    "diagram.world_step": {
        "t": "the loop closes in the world",
        "w": "The last numbered step is not part of the model. The command the "
             "motor controller wrote - after growth limits, the reach clamp, "
             "[[babble]] noise, smoothing and, in a rest window, the postural "
             "substitution - is executed by the body in the simulated crib. "
             "What that command changes comes back through the senses as the "
             "next frame, and that frame is what grades the guess made from "
             "the frame before it.",
        "f": "reading this step as the prediction heads acting. They write the "
             "next model step's [[token|tokens]] and reach no actuator; the command "
             "comes from the motor controller, or from the planner when the "
             "planner ran that model step",
    },

    # ================= the gain-of-function build (2026-09-22) ==================
    # docs/gain_of_function_v3/DESIGN.md decisions D4-D13. Every feature below sits behind a
    # launch flag whose default reproduces the run before the build; each entry says which.
    "term.causal-chain": {
        "t": "the causal chain (a record of intention to result)",
        "w": "For every planning event and every command sent to the body, one "
             "record of five links: the INTENTION (the remembered model step the "
             "planner aimed at, and what it was worth), the PREDICTION (the "
             "next model step and the command's end, as the model expected them "
             "BEFORE the command was sent), the PREDICTED WELLNESS (the "
             "[[approximate-evaluator|learned copy's]] five-term score of that "
             "prediction, for the chosen plan and for four "
             "[[alternatives|alternatives]] scored from the same state), the "
             "ACTION (whether the plan's first command reached the body, and how "
             "much of it survived the smoother), and the RESULT (the raw tokens "
             "that arrived one model step later and at the command's end, graded "
             "against the prediction and against [[persistence]]). "
             "{causal_chain_run}",
        "when": "A planning event is recorded on the planner's own model step; a "
                "prediction is made on every latch model step (one model step in "
                "--motor-hold), before the command is sent; each is graded one "
                "model step later and again when the command ends.",
        "h": "mimo_bridge/lem_model.py _causal_plan_record (the planning event, "
             "last_plan_record) and predict_before_execution (the prediction); "
             "mimo_bridge/causal_chain.py CausalChainRecorder (the latch, the "
             "grading, <run>/causal_chain.jsonl one line per resolved command, "
             "and the stats record's causal_chain block)",
        "f": "reading a good prediction as a good plan. The planner descends "
             "on distance to a remembered model step, not on predicted wellness, so "
             "'the chosen plan scored best' measures whether the two agree, not "
             "whether the planner tried; and a prediction is worth having only "
             "while it sits below its [[persistence]] line",
    },
    "term.efference-heads": {
        "t": "efference heads",
        "w": "Two small networks handed the executed command DIRECTLY, beside the core's "
             "state, that predict what the command will do in lived units: every handle's "
             "move in cm and turn in degrees over the next 50 ms and over the command's 250 "
             "ms, and the eyes' turn in degrees at 50, 100 and 150 ms. The body head's "
             "command-long prediction is what [[controllability]] judges the body against. "
             "{efference_run}",
        "h": "lem/model/efference.py; the architecture tab's 'efference heads' box, when "
             "the run built them",
    },
    "term.persistence": {
        "t": "persistence (the 'nothing changes' line)",
        "w": "The error of predicting that the next model step will be exactly "
             "this one: every token, every handle and the eyes held where they "
             "are. It is the floor every prediction on this page is set "
             "against. A prediction below it has learned something about how "
             "the model step changes; one above it would have been better replaced "
             "by 'nothing changes'. Beating it is an admission line, not the "
             "goal: the goal is an error of 0.",
        "h": "for tokens, the squared distance between this model step's raw token "
             "and the next one's (causal_chain.py persist_err_*, "
             "prediction_audit's persistence_mse); for the body and the eyes "
             "the size of the change itself (efference do_nothing_*)",
    },
    "term.approximate-evaluator": {
        "t": "Learned wellness predictor: five estimated scores",
        "w": "**Inputs during training:** the {d_model}-number encoder lists for vision, tactile, audio, proprioception and motor, placed end to end. Vestibular and intermind are excluded. The inputs are detached, so this matching loss changes the predictor's weights without changing those encoders.\n\n**Calculation and target:** a neural network with two hidden layers produces five numbers between zero and one. Its training loss squares each estimate-minus-measured-score difference and averages over all five outputs and all bodies. The targets are the measured prediction-accuracy, tactile-comfort, capacity, coherence and motor-state scores.\n\n**Use during planning:** predicted sensory lists and the encoded candidate command are supplied in the same input sequence. The five outputs estimate those same five measured scores for the predicted outcome.",
        "h": "lem/model/wellness.py: ApproximateEvaluator, APPROXIMATE_TERMS, APPROXIMATE_EXCLUDED and score_predicted_wellness. lem/training/loss_manager.py: _compute_wellness_loss. The bridge supplies canonical_tokens from detached raw encoder outputs.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The network has no self-touch output and does not predict the extra channels used in [[wellness-ranking|goal ranking]]. Its output scale allows a tactile-comfort prediction above the formula's 85% maximum; training targets, rather than an output constraint, teach that limit. [[planner-blind|Missing predicted channels.]]"
    },
    "term.alternatives": {
        "t": "the four alternatives of a planning event",
        "w": "Four commands scored from the SAME captured state as the planner's "
             "chosen one, so their predicted wellness can be ranked against "
             "it: hold (every handle's target where it is now, the plan's "
             "strength, the eyes held), the motor controller's own proposal "
             "this model step, and two perturbations of the plan at the babble's "
             "own size (5 cm per axis of position, a turn of 29 degrees per "
             "axis, drawn from a generator seeded by the run's "
             "seed and the step). Rank 1 = no alternative was predicted "
             "higher; rank 5 = all four were. Picking at random among five "
             "would put the chosen one best 1 time in 5.",
        "h": "LemLite.CAUSAL_ALTERNATIVES and _causal_perturb in "
             "mimo_bridge/lem_model.py; rank_of_chosen in "
             "mimo_bridge/causal_chain.py",
    },
    "term.shuffled-command-control": {
        "t": "the other-body command control (absolute and size-matched)",
        "w": "The same prediction made with ANOTHER body's command in place of "
             "this body's, everything else unchanged: the line a prediction must "
             "beat to show it used the command, since a part that ignores the "
             "command it is given scores the same with either. The command is "
             "built one of two ways. ABSOLUTE: the other body's 26 x 9 numbers as "
             "they stand, targets chosen for the other body's pose, so the change "
             "they ask of this body is larger than any body was asked, and a part "
             "whose output grows with the asked change errs more with it without "
             "using the command's effect. SIZE-MATCHED: the other body's asked "
             "moves and turns, measured from its own pose, applied to this body's "
             "pose (lem/model/contact_head.shuffled_order), and the eyes' row as "
             "the other body's asked eye turn applied to this body's eyes "
             "(lem/model/efference.size_matched_other_order). The contact head's "
             "gates (success_control beside success_paired) use the size-matched "
             "command. Since Oct 2, 2026 the swap test (prediction_audit) and the "
             "efference heads report both, *_size_matched beside the absolute "
             "shuffled_* readings, which stay for one segment and are then retired; "
             "controllability's evidence gate (control_mean) and the self-touch "
             "audit still read the absolute command. Since Oct 3, 2026 "
             "controllability also records the size-matched control "
             "(size_matched_control_mean, size_matched_difference_mean) and a "
             "shadow gate fed by it under the evidence gate's own rule "
             "(shadow_gate_open; per report window gate_commands and "
             "shadow_gate_commands count the graded commands each gate took a "
             "reading at, gate_open_commands and shadow_gate_open_commands those "
             "after which it stood open); the controllability weight in the goal "
             "ranking is set by the gate fed by the absolute command, and the "
             "shadow gate is recorded only. The swap test has a third control: "
             "this body's own command plus one standard deviation of each of its "
             "numbers over the executed-command stream (improvement_over_shifted_own). "
             "It needs two or more bodies.",
    },
    "term.evidence-gate": {
        "t": "Evidence gates: when a measured signal can affect selection",
        "w": "A gate makes a signal contribute zero until its measured advantage over a control is large enough. The rules differ by signal; there is no single threshold for all wellness channels.\n\n- **Controllability:** 32 controlled command events; opens above three standard errors, closes below one.\n- **Contact control:** separate self-contact and outside-contact gates use the same three/open, one/close thresholds over 32 events. The gate shuts if its newest controlled reading is more than 200 model steps old. A current reading cannot open its own payment.\n- **Older self-touch score:** 32 controlled readings; mean advantage must exceed two standard errors, with evidence no more than 2,000 model steps old.\n\nA standard error measures uncertainty in an average, calculated from the spread and number of readings. The named-entity reader has its own controlled-admission checks.",
        "h": "lem/model/wellness_channels.py: ControllabilityTracker._two_lines, _update_gate and gate. lem/model/contact_head.py: gate_add, gate_expire, CONTACT_GATE_STALE_STEPS and ContactTracker.score. mimo_bridge/lem_model.py: _self_touch_gate. [[eval-self-touch|Older self-touch score]] and [[contact-control|contact control]] are different mechanisms.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "A shut gate means its rule currently excludes that contribution. It does not mean the measurement is absent or the module is broken."
    },
    "term.controllability": {
        "t": "Controllability: predicted command effects compared with measured effects",
        "w": "For a qualifying command hold, compare the predicted and measured changes in handle position and orientation. Agreement combines direction match with the smaller change size divided by the larger. An exact match scores 1; an opposite direction gives a negative score. Commands or predictions below the measured drift thresholds do not qualify.\n\nThe control comparison predicts this body's effect using another body's command. The channel can influence goal ranking after the real-command advantage passes its evidence gate: 32 qualifying readings, opening above three standard errors and closing below one.",
        "h": "lem/model/wellness_channels.py: ControllabilityTracker. An exponential running average with a five-event horizon supplies the channel value; older events retain decreasing weight; the evidence history uses 32 command events pooled across bodies. [[efference-heads|The heads that predict command effects]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The size-matched command comparison is recorded separately and does not currently decide this gate. [[shuffled-command-control|What the comparison changes.]] [[evidence-gate|Why gates differ.]] With efference prediction disabled, the fallback compares commanded rotation with measured rotation instead."
    },
    "term.learning-progress": {
        "t": "Learning progress: recent prediction error versus longer-term error",
        "w": "For each body and eligible sense or efference head, track two kinds of error: the model's squared prediction error, and the error from predicting no change. Average each error separately over a fast window of about 200 model steps and a slow window of about 2,000.\n\nDivide the fast model-error average by the fast no-change-error average; do the same for the slow averages. The channel is (slow ratio minus fast ratio) divided by the slow ratio, clipped between -1 and +1, then averaged over available senses and heads. Positive means the recent relative error is lower.",
        "h": "lem/model/wellness_channels.py: LearningProgressTracker.update and progress. Each ratio is a ratio of two error averages, not an average of per-step ratios.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "Changing the difficulty of the observations can change this reading even with fixed model weights. It is a comparison of recorded prediction errors, not a direct count of abilities acquired. It is different from the [[eval-prediction-error|fixed prediction-accuracy score]]."
    },
    "term.wellness-ranking": {
        "t": "Goal ranking: weighted changes in wellness channels",
        "w": "The normalized ranking chooses among remembered sensory states in three steps:\n\n1. **Store each channel's change.** Subtract that body's running average over about {detrend_steps} model steps from the current channel value. A capacity of 80% can therefore rank differently after a recent 90% than after a recent 70%.\n2. **Put changes on comparable scales.** Subtract each channel's running centre and divide by its running spread, or its minimum scale if the spread is smaller. These statistics combine bodies over about {normaliser_steps} model steps.\n3. **Multiply by the configured weights and add.** Missing measurements add zero. [[integrity|The harm rule]] excludes excessively harmful memories when a safer one is available.\n\nThus a high rank means a large favorable weighted change relative to recent history. It does not necessarily mean the highest absolute wellness or predict that a command will improve wellness now. {wellness_rank_run}",
        "h": "lem/model/wellness_channels.py: detrend_moment, ChannelNormalizer and rank_values. mimo_bridge/lem_model.py: _rank_store_moment and _normalized_goal_values. The raw fixed-score sum is separate.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "[[channel-floor|Minimum scales]] keep nearly constant channels from receiving enormous scores. Weights express priorities; the channel table shows their actual contributions. [[controllability|Controllability]] has zero weight while its gate is shut. [[learning-progress|Learning progress]], [[contact-control|contact control]], [[consequence-world-only|consequences]], [[efficiency-channel|effort]] and [[curiosity-mode|curiosity]] each have their own inputs. [[planner-blind|Which channels the learned predictor cannot forecast.]] [[eval-movement|Movement channel]]."
    },
    "term.integrity": {
        "t": "Integrity: skin load and handle-position penalties",
        "w": "Harm is the larger of the highest skin-patch aversion and one minus the motor-state score. Integrity is one minus harm. Skin aversion uses the worst patch; the motor-state score averages position penalties across handles and coordinates. [[eval-tactile-comfort|Skin formula]]; [[eval-motor-state|position formula]].\n\nA remembered entry with measured harm above {harm_line_pct} is excluded while that body has an entry at or below the limit. If every measured entry exceeds the limit, the least harmful remains available, with ties settled by rank.",
        "h": "mimo_bridge/lem_model.py: _wellness_channel_vector. lem/model/wellness_channels.py: rank_values.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "An unavailable harm reading is not automatically excluded. This rule is based on these two numerical scores; the position score is not a physical joint-limit detector."
    },
    "term.channel-floor": {
        "t": "a channel's floor",
        "w": "The smallest spread a channel is ever divided by: its measured "
             "jitter from one model step to the next. Without it, a channel stuck near one "
             "value (a shut gate, a slow drift) would be divided by nearly "
             "nothing and its last digits would decide the goal. Measured on "
             "ckpt_step_344400 at 2 bodies while training (b3-wellness "
             "measurement 3), each as a percentage of the channel's own 0-to-1 "
             "range: integrity 2%, prediction error 1.5%, tactile comfort 2.2%, "
             "coherence 0.5%, curiosity 3.7%; consequence's floor is about one "
             "type of consequence (each type pays one half), and "
             "contact_control's 18% of what one foreseen outside touch pays.",
        "h": "FLOORS in lem/model/wellness_channels.py, each with its source",
    },
    "term.planner-blind": {
        "t": "What the learned wellness predictor can forecast",
        "w": "The learned predictor has five outputs: prediction accuracy, tactile comfort, capacity, coherence and motor state. It has no outputs for self-touch, controllability, learning progress, contact control, consequence, named-entity value, agency, movement, curiosity or efficiency. It does not predict the complete normalized goal-ranking score.\n\nGoal selection uses recorded channels. Command search then tries to match the chosen sensory target. Predicted wellness enters that search objective only when prospective wellness is enabled with a positive weight; recording predicted scores does not itself enable that objective.",
        "h": "lem/model/wellness.py: APPROXIMATE_TERMS. lem/model/wellness_channels.py: JUDGE_B_VIEW. mimo_bridge/lem_model.py: _plan_chain.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "Integrity is only partly represented through tactile-comfort and motor-state estimates. Neither output directly predicts the combined harm gate."
    },
    "term.curiosity-mode": {
        "t": "How curiosity affects remembered-goal value",
        "w": "With normalized ranking, curiosity is its own weighted channel when curiosity-goal is enabled. It goes through the same history subtraction and scaling as the other channels. The multiply/additive setting does not multiply this normalized ranking.\n\nWith raw ranking, multiply changes the stored raw value to value times (1 + drive). Additive instead adds drive times its configured weight. {curiosity_mode_run}",
        "h": "mimo_bridge/lem_model.py: goal-ring value assembly, _wellness_channel_vector and _normalized_goal_values. lem/model/curiosity.py: CuriositySystem.compute_drive.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The drive is a triangle-shaped function of prediction error. Being inside its chosen error range is not proof that the observations are learnable. The curiosity box shows the configured range."
    },
    "term.consequence-world-only": {
        "t": "Consequence: recorded types of effect on the surroundings",
        "w": "The raw goal-value contribution adds the configured consequence weight once for each recorded kind: hand pressure, object movement or a newly gained view. Normalized ranking stores the channel and applies its own scaling and weight.\n\nWith consequence-self-contact set to exclude, the hand-pressure kind counts pressure on something outside the body. With include, self-contact can count too. {consequence_run}",
        "h": "mimo_bridge/lem_model.py: consequence_kinds and _wellness_channel_vector. env.py: split_hand_press. [[hand-press-split|World and self-contact measurements]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "term.hand-press-split": {
        "t": "the hand press, split into world and self",
        "w": "Every world step reports the firmest press on a hand patch twice: "
             "the part pressed by something outside the body and the part "
             "pressed by the body itself, each as the share of model steps of a body "
             "over the 2 kPa line and the mean firmest press in kPa. Measured "
             "under babble on the crib body (env-senses, 2026-09-22): the old "
             "undivided flag fired on 70 of every 100 frames, the world part on "
             "12, the self part on 66.",
        "h": "env.py split_hand_press; the stats record's hand_press block",
        "f": "the 2 kPa line predates the mechanical_v4 skin (a hand at rest "
             "reads 1.5 kPa at the median, a babbling one 117 kPa); re-anchoring "
             "it is an open question",
    },
    "term.efficiency-channel": {
        "t": "Efficiency: negative scaled effort, default weight zero",
        "w": "Add vestibular fields 23 and 24, then negate the sum. These fields contain scaled mechanical power and summed motor torque. Zero means no measured effort; more effort gives a more negative value. At one reference unit in each field, the score is -2.\n\nIts default ranking weight is zero, so recording this channel alone does not change goal choice. A weight override can enable it.",
        "h": "mimo_bridge/lem_model.py: _wellness_channel_vector reads vestibular columns 23 and 24. lem/model/wellness_channels.py: DEFAULT_WEIGHTS. [[vestibular-vector|Field definitions and reference scales]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "term.vestibular-vector": {
        "t": "the vestibular / internal-state vector (29 numbers)",
        "w": "What the body feels of its own load and balance, every world "
             "step, in these groups (each scaled so the crib body's 99th "
             "percentile reads about 1):\n"
             "- **Head accelerometer** (3): which way gravity pulls in the head "
             "plus linear acceleration, in g (one g = gravity alone).\n"
             "- **Head gyro** (3): how fast the head turns, scaled so 115 "
             "degrees a second reads one.\n"
             "- **Neck** (9): the three neck joints' angles (from one end of the "
             "joint's range to the other), speeds (scaled so 86 degrees a "
             "second reads one) and motor efforts (from full strength one way "
             "to full strength the other).\n"
             "- **Which way is up for the trunk** (3): world-up seen from the "
             "chest, a unit vector.\n"
             "- **Support** (5): the share of outside contact force carried by "
             "the head, the trunk, the arms and the legs (they sum to 1 when "
             "anything outside touches the body), and the total in body "
             "weights (one body weight lying still, 3.6 body weights at the "
             "99th percentile of babbling, 18 body weights on a 10 cm drop).\n"
             "- **Effort** (2): mechanical power (scaled so 66 W reads one) and "
             "holding load, the summed motor torque a pose costs (scaled so "
             "63 N m reads one).\n"
             "- **Limit load** (1): out of every 100 motor-driven joints, how "
             "many sit within 2 degrees of a limit (4 at rest, 12 at the 99th "
             "percentile of babbling).\n"
             "- **Fatigue** (2): the lowest and the mean muscle-group capacity "
             "(1 = fresh, 0 = spent).\n"
             "- **Valid** (1): 1 on a real read, 0 on a vector padded from the "
             "six numbers MIMo reported before 2026-09-22.",
        "h": "mimo_bridge/vestibular_adapter.py read_vestibular, "
             "VESTIBULAR_FIELDS and the measured scales (MEASUREMENT: 18,450 "
             "frames of babble, lying on each side, rolls and drops); the model "
             "clips every value to -4..+4 (vestibular_sense.INPUT_BOUND), which "
             "cuts only impacts",
        "f": "the world reads this vector on every run; the MODEL reads it only "
             "under --vestibular-sense on, in the slot the intermind "
             "placeholder held",
    },
    "term.contact-onset": {
        "t": "a contact onset",
        "w": "A touch that BEGINS: a skin patch whose contact bit was off over "
             "the whole previous burst and is on now. For self-contact the "
             "world reports each new PAIR of own parts that came together "
             "(absent for 5 or more world steps and present now), with both "
             "sides named, whether the pair touches at rest anyway (a thumb "
             "against its palm - 'structural', never scored or paid) and "
             "whether the contact point sat within 30 degrees of an eye's "
             "axis. Measured under babble (env-senses): 2,049 self onsets in "
             "15,000 frames - 68 in every 100 model steps - of which 12 or 13 "
             "in every 100 were structural and fewer than 1 in 100 in view.",
        "h": "env.py _contact_report: self_contact_onsets, "
             "contact_onsets_world, contact_patch_state (per patch 0 none, 1 "
             "outside, 2 own part, 3 both)",
    },
    "term.contact-control": {
        "t": "Contact control: a score for predicted new contacts",
        "w": "A contact can add value when it begins and the contact head predicted that onset above its patch-specific base rate and threshold. Self-contact adds 2 times its repetition discount for each qualifying pair. Every side the tracker can score must pass; it may have only one eligible side. Outside contact adds at most 1 times the largest qualifying discount for that body and model step.\n\nRepeated contacts with the same pair or patch receive less value: divide by one plus the number of matching onsets in the preceding 200 model steps. Both scores are zero unless their separate command-control evidence gates were already open. {contact_control_run}",
        "h": "lem/model/contact_head.py: ContactTracker.score, self/world gates and habituation. [[contact-onset|What counts as an onset]], [[habituation|repetition discount]], and [[evidence-gate|gate thresholds]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings",
        "f": "The displayed would_pay value assumes an open gate; paid is the value actually allowed through. Predicting a contact is not by itself evidence that the planner deliberately chose it."
    },
    "term.habituation": {
        "t": "habituation",
        "w": "A paid touch is multiplied by 1 / (1 + the onsets of the same "
             "pair in the last 200 model steps, 10 s), so the first touch of a "
             "pair pays in full and a touch-release loop pays less every time "
             "it repeats: 1 repeat halves it, 3 repeats quarter it.",
        "h": "HABITUATION_STEPS and habituation_factor in "
             "lem/model/contact_head.py",
    },
    "term.liveness": {
        "t": "liveness (is a layer doing anything?)",
        "w": "Two measurements of whether a part of the network can respond "
             "and learn. LIVE UNITS: out of every 100 hidden units of a layer, how "
             "many left the flat low end of their activation function (GELU, the "
             "curve that turns a unit's input into its output) on some lived "
             "model step. In that flat end a change in the unit's input moves its "
             "output by less than 1.2% of the change, and the lesson's correction "
             "reaches the unit's weights shrunk by the same share; for a large "
             "input the output moves as much as the input (100%). 0 = no unit "
             "crossed that response threshold on the sampled inputs; 100 = every unit "
             "did. This does not test unseen inputs or establish a learned ability. LOCKED ATTENTION: out of every "
             "100 parts of an attention pool's attention, how many its single "
             "largest input takes: 100 = each measured reading puts all attention on one input. "
             "Measured by the lead on ckpt_step_344400 over 24 lived model steps "
             "(2026-09-22): 0 of 100 units live in the tenure pose head, 5 in "
             "the wellness predictor's first layer, 5 in block 1 of the motor "
             "controller, and the motor encoder's pool attention gives 100 "
             "of 100 parts to one input (each of its 4 heads reads one of 26 "
             "handles).",
        "h": "docs/gain_of_function_v3/measurements/liveness_sweep.py and "
             "liveness_sweep_ckpt_step_344400.json; the lineage table in "
             "measurements/liveness_over_lineage/README.txt; the Probes tab "
             "prints both. The line is an input value in the code "
             "(GELU_LIVE_INPUT in tools/gof_probe_harness.py, LIVE_LINE in "
             "lem/model/contact_head.py and motor_synergy.py): the input where "
             "GELU's output moves by 1.2% of a change (gof_probe_harness.gelu_slope)",
        "f": "reading a checked wire as a working part: the earlier checks "
             "confirmed the wiring and averaged the outputs, and the dead pose "
             "head's 0.22% share of the pose change was read as 'still "
             "learning'",
    },
    "term.captured-share": {
        "t": "captured share (the probes' headline)",
        "w": "Change ONLY the command, from the same lived model step, and measure "
             "two differences: how much the model's PREDICTION changed, and "
             "how much the REAL outcome changed (both relative to holding "
             "still from that model step). Captured share = (predicted change . "
             "real change) / (real change . real change): 0 = the prediction "
             "ignores the command, 1 = it foresees the whole change the command "
             "made, below 0 = it predicts the opposite. It is the direct test "
             "of whether the prediction changes with the command the way the outcome does.",
        "h": "mimo_bridge/tools/gof_probe_harness.py; each probe writes "
             "Saved/mimo_bridge/gof_probes/<run>/<checkpoint>/<probe>.json and "
             "a line in index.jsonl",
    },
    "term.toy-hand-off": {
        "t": "the toy hand-off",
        "w": "Since 2026-09-21 (--toy-hand-off follow) the adult lets go of "
             "the toy while the infant grips it: once two different parts of "
             "one hand have touched it for 0.1 s in a row (10 world steps of "
             "10 ms), the toy keeps its place and turn relative to that hand, "
             "until no part of that hand has touched it for 0.1 s; then the "
             "adult's own carry takes over from wherever the infant left it. "
             "The toy stays a mocap body throughout. With 'off' the toy moves "
             "only in the adult's hand and no push can move it.",
        "h": "mimo_bridge/caregiver.py Caregiver._grip_step (GRIP_PARTS 2, "
             "GRIP_STEPS 10, RELEASE_STEPS 10); the world reports "
             "object_follows_grip, object_held_by_infant and object_grips",
        "f": "trusting the launch flag over the world's report: until "
             "2026-09-22 a resumed segment kept its saved world's settings "
             "(fixed in commit bdbfb7895), so segment 31, launched with "
             "follow, ran with the hand-off off",
    },
    "term.finger-armature": {
        "t": "finger armature (why the fingers could not move until 2026-09-22)",
        "w": "Armature is extra rotor inertia MuJoCo adds to a joint every "
             "physics substep; it damps the buzz a servo that acts once per "
             "10 ms step produces. From 2026-08-21 a flat armature was added "
             "to every hinge that was 200 times a finger joint's own and "
             "larger than a finger link's own inertia, so each finger was "
             "about 200 times harder to turn against its small motor: "
             "through the servo a finger reached 29 in 100 parts of a 20 "
             "degree target in one motor hold (0.25 s). Since 2026-09-22 the "
             "fingers get their own --finger-armature (4 times a finger "
             "joint's own, so 5 times in all), measured to reach 118 in 100 parts of the target "
             "(a slight overshoot) with 3 velocity reversals in every 100 "
             "steps. {finger_armature_run}",
        "h": "mimo_bridge/env.py BridgeEnv: --hinge-armature 0.02 skips the "
             "eyes and the fingers; --finger-armature 0.0004 is added to the "
             "finger joints alone (their own armature is a quarter of that; "
             "their motors give 3 to 200 thousandths of a newton metre)",
        "f": "reading any hand measurement taken before 2026-09-22 as a "
             "measurement of hand control: the fingers moved only as the "
             "swinging hand flung them (a command and the next move went "
             "together far less for the fingers than for the wrists and "
             "elbows: a correlation, not yet measured as a count)",
    },
    "term.smooth-source": {
        "t": "what the smoother blends with (--smooth-source)",
        "w": "Every command the body is sent is blended with a previous command so "
             "motion comes in strokes: 35% new, 65% previous at H84's "
             "--smooth-actions 0.35. Under --smooth-source proposal (the "
             "default) 'previous' is a running average of the controller's "
             "PROPOSALS, four of every five of which were never executed "
             "(a command is latched once every --motor-hold model steps). Under "
             "--smooth-source executed it is the command the body was really "
             "given over the interval that just ended, and the eyes' row is "
             "passed through unblended. {smooth_source_run}",
        "h": "LemLite.smooth_blend and the --smooth-source branch of "
             "LemLite.step (DESIGN.md D11)",
    },
    "term.next-frame-norm": {
        "t": "next-frame loss normalised by persistence (--next-frame-norm)",
        "w": "DESIGN.md D12: divide each sense's next-model-step error by its own "
             "running [[persistence]] error, so every sense is pushed to beat "
             "'nothing changes' equally. Measured on segment 31, the next-frame lesson was 8.7 of every 100 parts of the whole loss; audio carried 67 of every 100 parts of that lesson and proprioception about 4. {next_frame_norm_run}",
        "h": "the loss-balance build agent (b6-loss-balance); the stats "
             "record's next_frame_norm block when it lands",
    },
    # THE EFFERENCE BOX AND ITS GAZE-COPY LINE. Until 2026-09-25 viewer.js replaced this entry and
    # added edge.efference.heads at load (commit aebef83c1), so a reader saw text that none of the
    # suite's checks of this map ever read. Both live here; the headless page check fails a page
    # whose script changes the embedded map (tools/check_explainer_page.py, doc_changed).
    "node.efference": {
        "t": "efference heads (what the command will do: cm, degrees and the next view)",
        "w": "Two small networks, the body head and the gaze head, that are "
             "handed the executed command DIRECTLY, beside the core's state, and "
             "predict what it will do in units a person can picture. The body "
             "head: for every [[handle]], how far it moves (cm) and turns "
             "(degrees) over the next model step (50 ms) and over the command's "
             "tenure (250 ms). The gaze head: how far the eyes turn (degrees of "
             "azimuth and elevation) over 50, 100 and 150 ms. A third path, the "
             "gaze copy, adds the eye command's expected effect to the "
             "prediction heads' next-model-step vision token (the line marked 'eye "
             "command → vision'), so the predicted next view depends on "
             "where the eye command points. The core itself receives the "
             "command only as 9 of 19 columns per handle pooled into one motor "
             "token; over H84's 344,400 steps before the heads existed, knowing "
             "the command improved the next-model-step prediction by 0% to 0.08% of "
             "the way from no use to exact. The heads give the command a second, "
             "direct way in.",
        "when": "{efference_run} Every [[model-step]] with a previous model step: the "
                "next-model-step body outputs and the 50 ms gaze outputs are graded "
                "on the delayed lesson's latent WITH gradient, so their error "
                "trains the core too; the tenure, 100 ms and 150 ms outputs are "
                "graded later on stored latents and train only the heads. The "
                "gaze copy has no lesson of its own: its output is added to the "
                "vision prediction before the delayed next-frame lesson grades "
                "that prediction, and, where they run, before re-study (--replay "
                "on, on the remembered windows whose command is known) and the "
                "imagination chain (--imagination) grade it; each of these "
                "trains it. The planner's action search makes the same addition "
                "with the gaze copy's weights held still. "
                "{physical_goals_run}",
        "h": "lem/model/efference.py EfferenceHeads and GazeCopy; loss = Huber "
             "in units of each horizon's measured spread (0.452 cm and 2.73 "
             "degrees per model step, 1.80 cm and 11.0 degrees per tenure, 3.23 "
             "degrees, 6.22 degrees and 8.86 degrees of eye turn), each head "
             "at --efference-weight 0.1 unless set. Two [[wellness]] channels "
             "read the heads: under --controllability on the "
             "[[controllability]] channel scores the body head's command-long "
             "prediction against what the body did, and under "
             "--learning-progress on each head's next-model-step error against "
             "doing nothing joins the [[learning-progress|learning progress]] "
             "channel. Under --wellness-rank normalized both are among the "
             "values that decide which remembered model step the planner aims at "
             "(the dashed line into the wellness box; controllability only "
             "while its [[evidence-gate|evidence gate]] is open). "
             "{efference_architecture}",
        "f": "reading the error without its do-nothing and control lines: "
             "with the heads freshly built the errors sat on do-nothing (1.32 "
             "cm against 1.32 cm per model step; 4.54 degrees against 4.13 degrees "
             "at 50 ms), and an error equal to the size-matched control's means "
             "the head ignores the command it is handed (the absolute control's "
             "line, kept for one segment, sits higher for any head whose output "
             "grows with the asked change). And reading a more accurate body "
             "head as a better choice of command: the planner's action search "
             "calls the body head only under --physical-goals memory (the solid "
             "line into the planner's box). Under --wellness-rank normalized "
             "with --controllability or --learning-progress on, the heads' "
             "readings change which remembered model step becomes the goal (the "
             "dashed line), whether or not the commands chosen for it get better.",
    },
    "edge.motor.efference": {
        "t": "executed command -> efference heads: the data",
        "w": "The command the body was really given, one row per [[handle]] x 9 "
             "numbers in encoder units, handed to both heads directly (the "
             "gaze head reads the eye command's azimuth and elevation beside "
             "the gaze row). This is the explicit copy of the command the "
             "addendum's section 6 asks for.",
        "when": "Every [[model-step]] the heads are graded: the command executed over the interval that just "
                "ended (the trainer's prev_executed), so the line is dashed.",
    },
    "edge.final_norm.efference": {
        "t": "core state -> efference heads: the data",
        "w": "The core's newest-model-step tokens for the PREVIOUS model step with the "
             "executed command in its motor token - the latent the delayed "
             "next-frame lesson computes - pooled by each head's own learned "
             "query. Gradient crosses into the core on the next-model-step "
             "outputs.",
        "when": "Every [[model-step]] with a previous model step.",
    },
    "edge.efference.heads": {
        "t": "eye command -> vision prediction (the gaze copy): the data",
        "w": "The gaze copy's output: one change of {d_model} numbers per body, "
             "added to the prediction heads' next-model-step vision token, so the "
             "predicted view depends on where the eyes were told to look. It is "
             "computed from three inputs: the core's vision token for the "
             "model step the command starts from (centred and LayerNormed), the eyes' "
             "row of the command with the eye command's azimuth and elevation, "
             "and the head's and the eyes' orientation in the world. Its output "
             "layer starts at zero, so switching the heads on changed no "
             "prediction until a lesson moved it. {gaze_copy_repair_run}",
        "when": "Every prediction of the next model step made with a known command: "
                "the delayed next-frame lesson, the paired shuffled-command audit, "
                "re-study of remembered windows whose command is known, and every "
                "imagined step - the planner's action search, the imagination "
                "chain, the causal chain's predictions and the probes' rollouts. "
                "The planner holds its weights still; the lessons train it.",
        "h": "lem/model/efference.py GazeCopy and EfferenceHeads.vision_delta, "
             "attached to the prediction heads by attach_gaze_copy; the "
             "addition is made in lem/model/predictive_state.py transition and "
             "in LemLite's delayed lesson, audit and re-study. Drawn when the "
             "run's receipt records gaze_copy_into_vision, which the heads "
             "write whenever the model has vision.",
        "f": "reading a smaller vision error as proof that the copy uses the "
             "eye command: the stats record's efference.gaze_copy block sets "
             "vision's next-model-step error with this body's eye command beside "
             "the error with another body's eye command through the copy, "
             "size-matched to this body's eyes (vision_err_copy_size_matched: the "
             "other body's asked eye turn added to this body's eye angles) and, for "
             "one segment, as it stands (vision_err_copy_shuffled), and the copy "
             "uses this body's command only while the first is smaller than the "
             "size-matched one.",
    },
    "edge.efference.wellness": {
        "t": "efference heads -> wellness channels: the data",
        "w": "Two readings of the heads that two [[wellness]] channels are "
             "built from; the channels are stored with every remembered model step "
             "in the [[goal-ring]] and, under --wellness-rank normalized, "
             "decide which of those model steps the planner aims at. "
             "[[controllability|Controllability]]: the body head's command-long "
             "expectation - how far every body handle would move and turn over "
             "the command's tenure, predicted when the command was sent - against "
             "what the body did over that tenure; the channel is a running "
             "average of that agreement in which the newest measured command "
             "counts one fifth and older ones count less: once many commands are "
             "measured, the last five carry 67 of every 100 parts of it. "
             "[[learning-progress|Learning progress]]: the body head's "
             "next-model-step error in cm and the gaze head's in degrees, each set "
             "against the error of predicting no change, join the average over "
             "every graded sense. {efference_channels_run}",
        "when": "Controllability: once per command, when the next command appears. "
                "Learning progress: every [[model-step]]. The planner ranks the "
                "stored values at each planning event.",
        "h": "lem/model/wellness_channels.py ControllabilityTracker (it reads "
             "LemLite.last_efference_tenure, which lem/model/efference.py "
             "efference_lesson writes) and LearningProgressTracker (the keys "
             "efference_body and efference_gaze); LemLite._wellness_channel_vector "
             "stores both with every goal ring entry and "
             "_normalized_goal_values ranks them. Dashed: the expectation is "
             "formed when a command is sent and judged when it ends, and the "
             "stored values are ranked at later planning events. Drawn when the "
             "run ranks normalized with --controllability on or "
             "--learning-progress on at a weight other than 0.",
        "f": "reading the line as the heads choosing commands: these channels only "
             "rank which remembered model step becomes the goal, controllability "
             "counts only while its [[evidence-gate|evidence gate]] is open "
             "(weight 0 while it is shut), and learning progress has no control "
             "of its own ([[learning-progress|its entry]] says what it reads on "
             "weights that do not learn)",
    },
    "edge.efference.intention": {
        "t": "body head -> the planner's action search: the data",
        "w": "Every remembered model step keeps the body's measured pose beside it, "
             "and under --physical-goals memory the planner uses it. When the "
             "goal the planner chose has a stored pose, the planner first "
             "searches for the eye command whose predicted view comes closest "
             "to the remembered view, then searches for a body command with the "
             "eyes held at that command: for each candidate command it calls the "
             "body head for the move and turn of every body handle over the "
             "command's tenure, adds them to the pose the command starts from, and "
             "scores how far that predicted pose lands from the stored one, in "
             "measured tenure spreads (1.80 cm and 11.0 degrees per axis). A "
             "candidate may move each body handle at most 1.80 cm along each "
             "axis and turn it at most 11.0 degrees from the starting command (a "
             "finger: each joint at most 11.0 degrees), and the plan is one "
             "command long. {physical_goals_run}",
        "when": "At a planning event (one model step in --intention-every) whose "
                "goal carries a stored pose; the body head's weights are held "
                "still while the planner searches.",
        "h": "lem/model/physical_goal.py PhysicalGoalCost (pose_error_per_body) "
             "and PhysicalTrustRegion, built in LemLite._plan_chain from the "
             "physical goal LemLite.step stores only under --physical-goals "
             "memory; lem/model/intention_generator.py generate_action scores "
             "each candidate with it. Solid: the body head is called inside the "
             "planning step.",
        "f": "reading a plan that follows the body head as a plan that reaches "
             "the pose: the head's own command-long error is the efference "
             "block's body_cm_err_latch, set beside the do-nothing error on the "
             "run tab's efference body head chart",
    },
    "node.contact_head": {
        "t": "contact head (which skin patches will be touched, and by what)",
        "w": "For each of the 53 skin patches, three probabilities for the next "
             "model step: that anything touches it, that an own part is among what "
             "touches it, and that an outside thing is. It reads the core's "
             "state for the model step before, the executed command given directly "
             "(each handle's asked move over 9.3 cm, turn over 21 degrees, "
             "and strength) and each patch's own contact state. Its "
             "predictions feed [[contact-control|contact_control]]: a touch "
             "counts as foreseen only when the probability beat the patch's "
             "base rate of starting that touch by a measured margin.",
        "when": "{contact_head_run} Every [[model-step]]; every 7th model step it also "
                "predicts with another body's command (the "
                "[[shuffled-command-control|control]]).",
        "h": "lem/model/contact_head.py ContactHead (graded by cross-entropy "
             "on the world's per-patch report, at --contact-head-weight 0.1 "
             "unless set) and ContactTracker; "
             "the core's state is centred by a running mean and LayerNormed "
             "before the head reads it ([[liveness]] rule 1)",
        "f": "reading the head without the base-rate control: measured over "
             "480 model steps at 2 bodies the head ranked a patch touched at a model step "
             "above an untouched one in 79, then 89, of every 100 pairs, while "
             "the control stood at 91, then 95 (50 = a coin flip, 100 = every "
             "pair)",
    },
    "edge.motor.contact_head": {
        "t": "executed command -> contact head: the data",
        "w": "Per body handle: the asked position change (scaled so 9.3 cm "
             "reads one), the asked turn (scaled so 21 degrees reads one) "
             "relative to where the handle was, and the strength - 7 numbers "
             "per handle.",
        "when": "Every [[model-step]] the head is graded: the command executed over the interval that just "
                "ended (the trainer's prev_executed), so the line is dashed.",
    },
    "edge.heads.td_gate": {
        "t": "prediction of now -> top-down gate: the data",
        "w": "The delayed re-run's prediction of THIS [[model-step]]'s token for each predicted sense (vision, "
             "tactile, audio, proprioception, vestibular): {d_model} numbers per sense per body, detached. The "
             "encoders, the core and the prediction heads run again on the previous model step's readings with the "
             "executed command in, and their output is the prediction the gate blends into the token the core "
             "reads: token + g x (prediction - token), with g at most one half. Vision's prediction includes the "
             "efference heads' gaze copy.",
        "when": "Every [[model-step]] with a previous model step.",
        "h": "mimo_bridge/lem_model.py: self._nf_pred, stored by the delayed next-frame lesson and read by the "
             "top-down gate block (zp = self._nf_pred[m]).",
    },
    "edge.heads.wellness": {
        "t": "prediction of now -> wellness: the data",
        "w": "The same delayed prediction of this model step's tokens, {d_model} numbers per predicted sense per "
             "body. Two [[wellness]] channels read it: prediction error (1 minus the squared difference between "
             "the prediction and the raw tokens that arrived) and learning progress (each sense's error set "
             "against the error of predicting no change).",
        "when": "Every [[model-step]] with a previous model step.",
        "h": "mimo_bridge/lem_model.py: the predictions handed to the wellness evaluation are self._nf_pred; the "
             "learning-progress rows are built from the same delayed predictions.",
    },
    "edge.heads.curiosity": {
        "t": "prediction error -> curiosity drive: the data",
        "w": "One number for all bodies: the mean, over the senses graded real, of the squared difference between "
             "the delayed prediction of this model step's tokens and the raw tokens that arrived. The curiosity "
             "drive is computed from that one number over the --curiosity-band range (the curiosity box's entry "
             "gives the rule).",
        "when": "Every [[model-step]] with a previous model step.",
        "h": "mimo_bridge/lem_model.py: self.last_causal_pe from the delayed lesson's per-sense errors, then "
             "self.curiosity.compute_drive(self.last_causal_pe).",
    },
    "edge.gaze.world": {
        "t": "eye command -> the world: the data",
        "w": "One azimuth and one elevation per body, in radians, sent to the eye servo on every physics step of "
             "the next model step; unlike the body's command it is not held. In the world the spontaneous saccade "
             "generator and the gaze reflex can still rewrite it before the eye joints move.",
        "when": "Every [[model-step]].",
        "h": "mimo_bridge/lem_model.py: gaze_out = self._gaze_from_handle(motor_out); mimo_bridge/train_mimo.py: "
             "src.step(send_motor, res.gaze_out, ...) on each env step; mimo_bridge/env.py: the saccade generator "
             "and the gaze reflex.",
    },
    "edge.contact_head.wellness": {
        "t": "contact head -> the contact-control channel: the data",
        "w": "Per body, a running average of payments for touch onsets the contact head predicted from the "
             "previous model step (empty while nothing has been scored). It is stored with every goal ring entry "
             "as the contact-control [[wellness]] channel and ranked when the planner's goal is picked, at the "
             "weight --contact-control-wellness sets.",
        "when": "Every [[model-step]] the head is graded.",
        "h": "mimo_bridge/lem_model.py: self._contact_control_value, read into the contact_control ranking channel.",
    },
    "edge.world.caregiver": {
        "t": "the world -> the caregiver: the data",
        "w": "What the caregiver reads of the world on every physics step: where the infant's left eye looks (at "
             "the adult's face or at the toy, held for 0.4 s to count), whether the infant touches the toy, how far "
             "the toy moved over the last 0.5 s, and which hand parts grip the toy.",
        "when": "Every physics step, inside the world's own step.",
        "h": "mimo_bridge/caregiver_interaction.py (looks_at, the toy contact and movement checks) and "
             "mimo_bridge/caregiver.py (the grip check).",
    },
    "edge.caregiver.world": {
        "t": "the caregiver -> the world: the data",
        "w": "What the caregiver changes in the world: its recorded speech (which the audio sense hears), the "
             "adult's head and hand poses, the toy's pose and visibility, and the mobile's height. It comes and "
             "goes on the --caregiver-present-seconds and --caregiver-absent-seconds schedule.",
        "when": "Every physics step, inside the world's own step.",
        "h": "mimo_bridge/env.py: BridgeEnv.step runs the caregiver; mimo_bridge/caregiver.py.",
    },
    "edge.world.entity_self": {
        "t": "the world's presence report -> named things: the data",
        "w": "Per body, from the world's frame report: which of the named things are present now, among them "
             "whether the adult is present and how visible the adult is (the caregiver's report). The named-entity "
             "self learns how each one's presence goes with changes in wellness.",
        "when": "Every [[model-step]].",
        "h": "mimo_bridge/lem_model.py: the entity presence assembly, which reads frame.meta['caregiver'] "
             "(adult_present, adult_visibility).",
    },
    "edge.final_norm.contact_head": {
        "t": "core state -> contact head: the data",
        "w": "The delayed lesson's latent for the previous model step and its "
             "command, minus its running mean over lived model steps, then "
             "LayerNormed. Gradient crosses into the core.",
        "when": "Every [[model-step]] with a previous model step.",
    },
    "edge.wellness.entity_self": {
        "t": "Fixed wellness sum supplied to named-entity learning",
        "w": "Each body supplies its own sum of fixed wellness scores as a training signal for associations between an entity's measured presence and changes in that sum. A configured alternative uses the number of recorded consequence kinds. This is a measured training signal, not the learned wellness predictor or normalized goal rank.",
        "h": "mimo_bridge/lem_model.py: named-entity teacher assembly and entity-self update. [[wellness|The fixed scores]] and [[wellness-link|associations with wellness changes]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "edge.entity_self.intention": {'t': 'tested named associations to remembered-goal value', 'w': 'The separate SelfValueReader adds signed associations to remembered-goal values only after its repeated evidence gate passes. An enabled reader weight is not evidence that the gate is open. Its value is zero while closed.', 'plain': 'The separate SelfValueReader adds signed associations to remembered-goal values only after its repeated evidence gate passes. An enabled reader weight is not evidence that the gate is open. Its value is zero while closed.', 'h': 'mimo_bridge/lem_model.py; self_state_integration.py; latent_body_shadow.py; self_value_reader.py.'},
    "node.synergy": {
        "t": "motor synergy: learning a movement map",
        "w": "This small network learns to describe completed body movements with a compact intention. "
             "It trains on the changes between the body's starting and ending poses, weighted by "
             "movement and command following. Learning can continue while both control switches are off.",
        "when": "{synergy_run} The planner uses the map only with --synergy-planner on; exploration "
                "uses it only with --synergy-babble on. Each switch needs a paired physical comparison "
                "before activation under the takeover criteria.",
        "h": "MotorSynergy in lem/model/motor_synergy.py; LemLite._synergy_lesson and _synergy_grade. "
             "Only the movement map's weights learn from this lesson. See RESTART_CRITERIA.md.",
        "f": "Reproducing a movement the body already made does not prove that using the map will "
             "make the body follow commands better.",
    },
    "node.synergy_examples": {
        "t": "completed movements: the synergy's training examples",
        "w": "The body's pose before and after a completed command, plus how far it moved and how "
             "well it followed that command. These examples train the movement map even when the "
             "map has no control over the body.",
        "when": "After an executed command completes its tenure.",
        "h": "LemLite._synergy_lesson and _synergy_grade; the map keeps its own movement bank.",
    },
    "edge.world.synergy_examples": {
        "t": "physical movement becomes a training example",
        "w": "Observed starting and ending body poses over the completed command. The dashed line "
             "crosses model steps: these are measured outcomes, not predicted movements.",
        "when": "After the command's physical execution.",
    },
    "edge.synergy_examples.synergy": {
        "t": "completed movements train the synergy map",
        "w": "The achieved pose change is encoded into a compact intention and reconstructed. "
             "The reconstruction lesson updates only the synergy map. It does not send a command "
             "to the body or update the existing planner.",
        "when": "During training, including when synergy planning and exploration are off.",
    },
    "edge.synergy.intention": {
        "t": "synergy -> planner: the data",
        "w": "The planner's action space: it adjusts z (dim numbers per body) "
             "and the synergy turns z into the candidate command it scores.",
        "when": "Every planning [[model-step]], inside each of the planner's rounds.",
    },
    "edge.synergy.world": {
        "t": "synergy exploration reaches the body",
        "w": "With synergy exploration enabled, the map can replace independent random turns "
             "with a coordinated body movement. This happens only on a command without a planner "
             "action, with exploration noise above zero, outside direct wellness control. "
             "The command still passes through the body's reach and strength limits.",
        "when": "When --synergy-babble is on and an eligible exploratory command is executed.",
        "h": "LemLite._synergy_babble and the exploration branch after execution_transform.",
    },
    "node.sense.vestibular": {
        "t": "vestibular (balance, neck, support, effort)",
        "w": "The body's own load and balance as a sense: the "
             "[[vestibular-vector|29-number vector]] - head gravity and "
             "turning, the neck, which way is up for the trunk, what carries "
             "the body's weight, effort, joints at their limits, fatigue. "
             "Behind --vestibular-sense on it takes the sixth input slot, "
             "where the intermind placeholder's Gaussian noise used to go.",
        "when": "Every [[model-step]]. The first frame after resuming a checkpoint "
                "saved before 2026-09-22 carries the old six numbers padded "
                "(valid 0) and is marked stale for the whole batch: encoded "
                "and fed forward, graded by nothing.",
        "h": "mimo_bridge/vestibular_adapter.py (the world side) and "
             "mimo_bridge/vestibular_sense.py (the slot, the checkpoint "
             "switch, the stats block)",
        "f": "reading its first hours of error as the sense's quality: the "
             "encoder starts fresh at the switch, so its improvement over "
             "[[persistence]] starts below 0",
    },
    "node.enc.vestibular": {
        "t": "vestibular encoder",
        "w": "Turns the 29 numbers into one [[token]] with a small [[mlp]], "
             "the same form as the proprioception encoder, at the run's "
             "--encoder-width. Every value is clipped to -4..+4 first, which "
             "cuts only impacts (a 10 cm drop reads 41 g for one frame).",
        "when": "Every [[model-step]].",
        "h": "vestibular_sense.modality_overrides (mlp_project encoder); "
             "started fresh at the switch, while the slot's prediction head, "
             "gate and decoder carry over under the new name",
    },
    "edge.world.sense_vestibular": {
        "t": "world -> vestibular: what flows here",
        "w": "29 numbers every world step, read from live MuJoCo on the frame "
             "the model reads: the head's accelerometer and gyro, the neck "
             "joints, the chest's up direction, the outside contact forces by "
             "body region, the motors' power and torque, the joints near a "
             "limit and the fatigue capacities. No [[gradient]] crosses.",
    },
    "edge.sense_vestibular.enc": {
        "t": "vestibular -> its encoder: the data",
        "w": "The 29-number vector in its field sequence, scaled (one g, 115 "
             "degrees a second of head turn, 86 degrees a second of neck turn, "
             "66 W, 63 N m ... each read as one) and clipped to four times "
             "those sizes either way, every [[model-step]].",
    },
    "causal_chain": {
        "t": "the causal chain, this planning event",
        "w": "The five links of one planning event of the [[causal-chain|causal chain]], read from the model at the snapshot's model step: what the planner aimed at, what it expected "
             "its first command to do, the predicted wellness of that command "
             "against four [[alternatives]], whether the command reached the "
             "body, and - from the run's causal_chain.jsonl, for events whose "
             "command has ended - what actually happened, graded against the "
             "prediction and against [[persistence]].",
        "when": "A planning event is recorded on planner model steps only (1 in "
                "--intention-every); the snapshot shows the newest one and "
                "names its step.",
        "h": "snapshot causal_chain block from LemLite.newest_plan_record and "
             "last_intention; the resolved events from <run>/causal_chain.jsonl",
    },
    "efference": {
        "t": "what the command was expected to do, against what it did",
        "w": "The newest command that ran its whole tenure: per handle, the move "
             "(cm) and turn (degrees) the efference body head expected when "
             "the command was sent, beside the move and turn the body made. The "
             "expectation was formed from the state where the command began, "
             "before any of the outcome existed.",
        "h": "LemLite.last_efference_tenure (expected_cm, expected_deg, "
             "achieved_cm, achieved_deg) and the window's running sums",
    },
    "vestibular": {
        "t": "balance, neck, support and effort, this model step",
        "w": "The [[vestibular-vector|vestibular vector]] of the snapshot's "
             "body at this model step, field by field in its own units. The world "
             "reads it on every run; the model reads it only under "
             "--vestibular-sense on.",
        "h": "BridgeFrame.vestibular of the captured body",
    },
    "wellness_channels": {
        "t": "Wellness channels: inputs to remembered-goal ranking",
        "w": "For the latest stored model step, this table shows each channel's recorded change, the running centre and scale used to compare changes, its weight, and its signed contribution to the ranking score. A positive contribution raises the score; a negative one lowers it. These are not the raw fixed-score bars.\n\n[[wellness-ranking|The three calculation steps]] explain why a channel's current raw value alone cannot tell you which remembered goal wins. The nearby panels show [[controllability|command-effect agreement]], [[learning-progress|error trends]] and the [[integrity|harm exclusion rule]].",
        "h": "explainer/capture.py: wellness_channels_block; explainer/viewer.js: wellnessChannelsPanel.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "contact_control": {
        "t": "foreseen touches, this model step",
        "w": "The self-touches and outside touches that began at this model step "
             "and were [[contact-control|foreseen]] by the contact head the "
             "model step before, what each would pay after "
             "[[habituation]], and whether the channel's evidence gate lets it "
             "pay.",
        "h": "LemLite.last_contact_control (per body: self_predicted, "
             "world_predicted, would_pay, pay, and the gate)",
    },
    "synergy": {
        "t": "the motor synergy, this model step",
        "w": "The synergy's size of z and babble's scale on z at the "
             "snapshot's model step; its learning is in the run tab's synergy "
             "chart.",
    },
    "probes": {
        "t": "the gain-of-function probes",
        "w": "Controlled tests run on a frozen checkpoint, from its own lived "
             "model steps: change ONLY one command and measure whether the model's "
             "prediction moves the way the real outcome does "
             "([[captured-share|captured share]]). Item 7: gaze. Item 8: "
             "simple body control (finger, wrist, elbow, shoulder, knee). Item "
             "9: an arm brought into view and self-touch, binding vision, "
             "proprioception and tactile. Item 10: head, neck and posture. Each result "
             "carries its floor (what knowing nothing scores), its goal (what "
             "working control would score), the distance between, and its "
             "controls.",
        "when": "On each checkpoint the probe watcher is pointed at (CPU, one "
                "suite at a time); the before-baseline ran on "
                "ckpt_step_344400 on 2026-09-22.",
        "h": "mimo_bridge/tools/gof_probe_*.py; Saved/mimo_bridge/gof_probes/"
             "index.jsonl (DESIGN.md section 9)",
    },
    "liveness": {
        "t": "dead and locked parts of the network",
        "w": "Which layers can respond and learn, measured on real lived "
             "model steps ([[liveness]]). The table traces each part across the "
             "lineage's checkpoints, so it says when a part died, not only "
             "that it is dead now.",
        "h": "docs/gain_of_function_v3/measurements/: the sweep of "
             "ckpt_step_344400 and liveness_over_lineage/README.txt (seven "
             "checkpoints, 24 lived model steps each)",
        "f": "rows marked with a * came from checkpoints that load only with a "
             "fresh tactile encoder, so their inner state differs; the motor "
             "pool does not read the tactile input and its numbers are exact",
    },

    'node.self_learning': {'t': 'self learning', 'w': 'Self learning is a training calculation, not another neural network. It compares earlier forecasts with later body measurements, then updates the weights of the self-perception and forecast networks. The return arrow shows those weight changes. A separate report carries measurements to the researcher. Evaluation computes measurements while keeping weights fixed.', 'h': 'mimo_bridge/lem_model.py: self_state_loss, _memory_bias_current; self_state_integration.py observe; self_intention.py choose.', 'plain': 'Self learning is a training calculation, not another neural network. It compares earlier forecasts with later body measurements, then updates the weights of the self-perception and forecast networks. The return arrow shows those weight changes. A separate report carries measurements to the researcher. Evaluation computes measurements while keeping weights fixed.'},
    'edge.self_state.self_learning': {'t': 'self forecasts to learning', 'w': 'Compare self forecasts with measured bodily function. Self learning uses the resulting prediction errors to update the same self networks; the outgoing return arrow shows that update.', 'plain': 'Compare self forecasts with measured bodily function. Self learning uses the resulting prediction errors to update the same self networks; the outgoing return arrow shows that update.', 'h': 'mimo_bridge/lem_model.py; self_state_integration.py; latent_body_shadow.py; self_value_reader.py.'},
    'edge.self_state.final_norm': {'t': 'self perception → forward-model output', 'w': 'With self-state set to perceive, the learned body estimate becomes an added adjustment to the forward model’s output. The motor controller and prediction heads can then use that adjusted output. The planner also receives the adjustment for its predictions. This route is absent in learn mode.', 'h': 'mimo_bridge/lem_model.py: self_state_loss, _memory_bias_current; self_state_integration.py observe; self_intention.py choose.'},

    'edge.self_learning.self_report': {'t': 'learning measurements to researcher', 'w': 'Computed self-prediction errors and function measurements go to the report. Reports do not issue body commands.', 'plain': 'Computed self-prediction errors and function measurements go to the report. Reports do not issue body commands.', 'h': 'mimo_bridge/lem_model.py; self_state_integration.py; latent_body_shadow.py; self_value_reader.py.'},
    'edge.self_learning.self_state': {'t': 'learning updates self perception', 'w': 'Prediction errors update the weights of the self-perception and future-function networks during training. This is a learning return path, not a command. In evaluation the weights stay fixed.', 'plain': 'Prediction errors update the weights of the self-perception and future-function networks during training. This is a learning return path, not a command. In evaluation the weights stay fixed.', 'h': 'mimo_bridge/lem_model.py; self_state_integration.py; latent_body_shadow.py; self_value_reader.py.'},
    'edge.world.latent_body': {'t': 'experienced movement to latent-body', 'w': 'The shadow observer reads completed state/command/result transitions and teaches its separate hierarchy. The source is the execution history, not a proposed command treated as already executed. No decoded command returns to production in shadow mode.', 'plain': 'The shadow observer reads completed state/command/result transitions and teaches its separate hierarchy. The source is the execution history, not a proposed command treated as already executed. No decoded command returns to production in shadow mode.', 'h': 'mimo_bridge/lem_model.py; self_state_integration.py; latent_body_shadow.py; self_value_reader.py.'},
    'guide.self_architecture': {'t': 'Self perception, movement and wellness', 'html': '<section class="self-guide" id="self-guide">\n<header>\n  <p class="sg-kicker">UEMA architecture review &bull; October 28, 2026</p>\n  <h2>How UEMA connects perception, self and movement</h2>\n  <p>MIMo is the simulated body. UEMA is the learning system that controls it. The aim is for UEMA to use its own experience to coordinate its body, eyes and fingers: lean toward a ball, reach, and grasp it.</p>\n  <p>UEMA\'s aim is broader than making the body move. Its design connects what the body senses, what may happen next, how that affects the body, and which movement follows. Through experience, these connections should build a continuing model of self: what belongs to this body, what it can control, what matters to it, and how reliable its predictions are.</p>\n</header>\n<section class="sg-spec-loop" aria-labelledby="sg-spec-title">\n  <h3 id="sg-spec-title">The intended loop in the UEMA specification</h3>\n  <p>A <strong>command</strong> is a set of numbers sent toward the body\'s motors. <strong>Wellness</strong> describes several aspects of the body\'s condition, such as sensing, available strength and prediction accuracy. UEMA keeps the following jobs distinct, so the movement can depend on predicted consequences for this body.</p>\n  <ol class="sg-five-stages" aria-label="UEMA\'s five connected jobs">\n    <li><strong>1. Perceive</strong><span>Read the body\'s senses. Encoders turn those readings into compact sets of numbers called tokens.</span></li>\n    <li><strong>2. Predict</strong><span>The forward model uses recent tokens and candidate commands to predict what would follow.</span></li>\n    <li><strong>3. Evaluate self</strong><span>Measure wellness from real experience. A learned evaluator predicts wellness for imagined outcomes.</span></li>\n    <li><strong>4. Form intention</strong><span>Select a desired consequence. Change candidate command numbers and compare their predicted consequences.</span></li>\n    <li><strong>5. Act</strong><span>Send the selected command. The motors change the body, and its senses provide the next readings.</span></li>\n  </ol>\n  <p class="sg-feedback"><strong>The loop returns through experience:</strong> actual readings are compared with earlier predictions. Their differences teach the prediction networks. The next choice uses the new readings and the learned predictions.</p>\n  <p><strong>Real experience and imagination stay separate.</strong> The spec calls real sensor readings the <em>Grounded Register</em> and encoded or predicted tokens the <em>Predictive Register</em>. Real outcomes teach the predicted wellness evaluator. An imagined safe landing must never be used as evidence that a safe landing occurred.</p>\n  <p>The spec calls the third job <em>feel myself</em>. Its computational role is to make the body\'s own condition matter to movement. The aim of conscious experience concerns this whole continuing loop; it is not assigned to one box merely because the box is named <code>SelfState</code>.</p>\n  <details><summary>Which specification, and which implementation?</summary><p>This review uses the complete September development <strong>UEMA v3.6</strong>. Its sections 1&ndash;2 describe the five jobs; sections 7&ndash;10 connect wellness and intention; sections 11 and 23 describe self. The community site\'s February release v3.5 is a different document line. A matching <a data-spec-link href="UEMA/UEMA_Spec_v3_6.html">v3.6 HTML edition</a> now preserves the development text.</p><p>The spec contains dated implementation notes beside its design. Some conflict with later passages: section 29 still says the body schema is missing, while sections 11.1&ndash;11.2 record its implementation. This guide uses the spec for intended roles and inspected source plus saved settings for implemented behavior. The detailed map below shows that implementation; this five-job diagram explains the design.</p></details>\n</section>\n<nav class="sg-tabs" aria-label="Self architecture topics">\n  <button type="button" data-self-topic="perception" aria-pressed="true">1. Self perception</button>\n  <button type="button" data-self-topic="parts" aria-pressed="false">2. The other self parts</button>\n  <button type="button" data-self-topic="choices" aria-pressed="false">3. From perception to choice</button>\n  <button type="button" data-self-topic="feelings" aria-pressed="false">4. Feelings and learning</button>\n</nav>\n<div class="sg-topic" data-self-panel="perception">\n  <h3>The learned self-perception component</h3>\n  <p><strong>Self perception</strong>, named <code>SelfState</code> in the code, learns a changing representation of bodily condition. It supplies one part of the wider self described above. The body schema tracks control, and the longer-term self mechanisms track relations with wellness.</p>\n  <p>The self-perception network receives skin readings, motor capacity and recent sensory history. Motor capacity means how much of its normal motor force is available. Sensory history includes what its senses, such as vision and touch, have recently reported. It keeps a learned internal state: numbers that carry what recent experience says about this body.</p><p>A command is a set of numbers sent toward the motors. A timed command sequence is a list of those commands and how long each lasts. A forecast network uses the self state and that sequence to predict changes in bodily function. These self forecasts concern sensing quality and motor capacity; the separate movement predictions describe where body parts will move.</p>\n  <div class="sg-scroll" tabindex="0" aria-label="Self-perception learning diagram; scroll sideways on a small screen">\n  <svg class="sg-map" viewBox="0 0 1120 410" role="img" aria-labelledby="sg-loop-title sg-loop-desc">\n  <title id="sg-loop-title">Self perception learns from the difference between a prediction and a later measurement</title>\n  <desc id="sg-loop-desc">Body readings enter self perception. Its state and timed commands enter body forecasts. Later measurements are compared with forecasts. Learning updates the self networks, completing the loop. A separate report goes to the researcher.</desc>\n  <defs><marker id="sg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#2c6682"/></marker></defs>\n  <g class="sg-box"><rect x="20" y="55" width="270" height="100" rx="8"/><text x="155" y="82">Body readings and history</text><text class="sg-small" x="155" y="108">Skin, motor capacity and context</text><text class="sg-small" x="155" y="132">and previous internal state</text></g>\n  <g class="sg-box sg-neural"><rect x="425" y="55" width="270" height="100" rx="8"/><text x="560" y="82">Self perception</text><text class="sg-small" x="560" y="108">A learned state of this body</text><text class="sg-small" x="560" y="132">with memory of recent changes</text></g>\n  <g class="sg-box sg-neural"><rect x="830" y="55" width="270" height="100" rx="8"/><text x="965" y="82">Body forecasts</text><text class="sg-small" x="965" y="108">Predict sensing quality, load,</text><text class="sg-small" x="965" y="132">stimulation and motor capacity</text></g>\n  <path class="sg-arrow" d="M290 105 H425"/><text class="sg-label" x="357" y="94">readings</text>\n  <path class="sg-arrow" d="M695 105 H830"/><text class="sg-label" x="762" y="94">learned state</text>\n  <path class="sg-arrow" d="M965 20 V55"/><text class="sg-label" x="945" y="15">timed commands</text>\n  <g class="sg-box"><rect x="830" y="270" width="270" height="100" rx="8"/><text x="965" y="298">Compare with later readings</text><text class="sg-small" x="965" y="324">After the command has run:</text><text class="sg-small" x="965" y="348">how far was the forecast off?</text></g>\n  <path class="sg-arrow" d="M965 155 V270"/><text class="sg-label" x="918" y="218">forecast</text>\n  <path class="sg-arrow" d="M1100 220 H1040 V270"/><text class="sg-label" x="1035" y="208">actual readings</text>\n  <g class="sg-box"><rect x="425" y="270" width="270" height="100" rx="8"/><text x="560" y="298">Self learning</text><text class="sg-small" x="560" y="324">Use the prediction error</text><text class="sg-small" x="560" y="348">to update these networks</text></g>\n  <path class="sg-arrow" d="M830 320 H695"/><text class="sg-label" x="762" y="309">error</text>\n  <path class="sg-arrow" d="M560 270 V155"/><text class="sg-label" x="460" y="215">update self weights</text>\n  <path class="sg-arrow" d="M620 270 V180 H875 V155"/><text class="sg-label" x="765" y="174">update forecast weights</text>\n  <g class="sg-box"><rect x="20" y="270" width="270" height="100" rx="8"/><text x="155" y="298">Report for the researcher</text><text class="sg-small" x="155" y="324">Predictions, errors and</text><text class="sg-small" x="155" y="348">measured bodily function</text></g>\n  <path class="sg-arrow" d="M425 320 H290"/><text class="sg-label" x="357" y="309">measurements</text>\n  </svg></div>\n  <p class="sg-caption"><strong>Read the loop:</strong> readings &rarr; self state &rarr; forecast &rarr; later measurement &rarr; learning &rarr; changed network weights. A weight is a stored number that learning changes. The report ends at the researcher; the learning arrow returns to the networks.</p>\n  <p><strong>This is more than a checklist.</strong> The learned state combines inputs over time and relates them to earlier self states. The displayed condition fields are only some of its outputs. In the recorded <code>grounded_residual</code> setting, current function is anchored to measured function; the learned forecast predicts how it will change.</p>\n  <h3>How self perception can affect movement</h3><p>The <strong>forward model</strong> combines recent sensory information. Its <strong>prediction heads</strong> turn the combined information into predictions. The <strong>motor controller</strong> is a learned network that proposes commands. The <strong>planner</strong> compares command predictions before choosing a command.</p>\n  <div class="sg-route"><strong>Route A &mdash; change the information used for movement</strong><p>Self state &rarr; learned adjustment to the forward model\'s output &rarr; motor controller, prediction heads and planner predictions.</p><p><code>self_state=perceive</code> enables this route. <code>learn</code> teaches self perception without adding this adjustment.</p></div>\n  <div class="sg-route"><strong>Route B &mdash; compare predicted bodily consequences</strong><p>Timed candidate commands &rarr; body forecasts &rarr; self-choice comparison &rarr; selected commands.</p><p><code>self_choice=on</code> enables qualified command use. <code>shadow</code> runs the comparison but discards its commands. <code>off</code> builds no chooser. These settings are independent of Route A.</p></div>\n  <p class="sg-run" data-self-run-summary></p>\n  <details><summary>The exact inputs, outputs and learning operation</summary>\n    <p><code>SelfState</code> contains small encoders, a recurrent network that carries recent state, learned attention over prior self states, and current/future function decoders. Each skin site\'s current function has three fields: sensing resolution, mechanical load and stimulation. Motor groups contribute available capacity. The future predictor reads the initial state, commands and their actual durations; it also predicts uncertainty.</p>\n    <p>In grounded-residual prediction, predicted function equals initial measured function plus elapsed time times a learned rate of change, limited to the allowed range. Later observed function supplies the learning target. Self-state input copies are detached from the main model\'s learning graph. Its errors train the self networks; in learn mode their gradients are clipped separately from the main model\'s gradients.</p>\n    <p>Self learning is an optimizer operation, not another neural decision maker. During evaluation, weights stay fixed. The report is output data, not a model of conscious speech.</p>\n  </details>\n</div>\n<div class="sg-topic" data-self-panel="parts" hidden>\n  <h3>How the self extends across body, objects and time</h3>\n  <p>A hierarchy has smaller parts inside larger parts. UEMA uses several different hierarchies. One groups body parts; another groups experiences across time. The name alone does not explain what a part learns.</p>\n  <div class="sg-part-grid">\n  <article><h4>1. Body schema: what responds to this body\'s commands?</h4><p><strong>Input:</strong> motor commands, sensed changes, and reported movements of tracked objects.</p><p><strong>Operation:</strong> track command/sense agreement. For an outside object, compare its direction and distance with a body part\'s movement, against the same comparison with that movement reversed.</p><p><strong>Output:</strong> membership in the body map, including tracked outside objects. The average membership contributes to wellness as the extent of self.</p><p>This is <strong>level 1 of hierarchical self</strong>. It stores running statistics. A held object needs a stable reported identity and position; following a hand is distinct from proving that the hand physically controls it.</p></article>\n  <article><h4>2. Named things: what matters over longer periods?</h4><p><strong>Input:</strong> supplied availability of each named thing, and this body\'s wellness or the selected ability measure.</p><p><strong>Operation:</strong> track whether availability and that measure change together over shorter and longer windows. A separate reader checks learned links against the same links attached to the wrong things.</p><p><strong>Output:</strong> after repeated evidence checks pass, the reader changes the value of remembered goals.</p><p>This is the current <strong>levels 2 and 3 mechanism</strong> described in v3.6 section 11.4. World metadata supplies identity and availability; experience supplies the association. The names do not tell the system which things are good. The observer can be <code>dark</code> while its separate reader is enabled.</p></article>\n  <article><h4>3. Legacy whole-moment tracker</h4><p><strong>Input:</strong> whole sensory moments and wellness.</p><p><strong>Operation:</strong> group similar moments and track their links with wellness over longer windows.</p><p><strong>Output:</strong> association tables and reports. Its <code>dark</code> mode observes without choosing commands.</p><p>This older attempt at levels 2 and 3 remains in the code as <code>hierarchical_self</code>, but it is <strong>off in the reviewed October 15 run</strong>. The spec records why changing sensory patterns made those slots unreliable, and specifies named things instead. The flag\'s name must not be mistaken for the whole self architecture.</p></article>\n  <article class="sg-neural-card"><h4>4. Latent-body: learn how body parts move together</h4><p><strong>Input:</strong> a body part\'s state, nearby parts\' states, the command and what happened next.</p><p><strong>Operation:</strong> train small neural networks to describe and predict movement using compact learned codes.</p><p><strong>Output:</strong> predicted movement/contact, a reconstructed command, and codes for smaller body parts.</p><p>Its lessons match experienced outcomes and agreement between body levels. They do not maximize wellness.</p></article>\n  </div>\n  <div class="sg-route"><strong>The named-things route has an output</strong><p>Named availability + wellness &rarr; association tracker &rarr; evidence reader &rarr; remembered-goal value &rarr; intention planning.</p><p>The final link is conditional. An enabled reader weight does not show that its evidence checks have passed. When the checks are closed, it adds zero.</p></div>\n  <div class="sg-route"><strong>Latent-body has a different destination</strong><p>Experienced movement &rarr; latent-body learning &rarr; saved movement hierarchy and prediction reports.</p><p>In the reviewed October 15 run it is <code>shadow</code>, with its planner <code>off</code>. It learns beside the acting system and supplies no production command. Separate comparison tools can test its decoded commands in copied worlds.</p></div>\n  <p><strong>These parts serve different jobs within a wider self.</strong> The body schema tracks immediate control. Named associations track what accompanies wellness over time. Self perception learns bodily state and command consequences. Latent-body learns how parts move together. None of their names alone describes the whole UEMA self.</p>\n  <details><summary>Why the older diagram omitted parts</summary><p>The detailed diagram hid disabled components, so the old hierarchical-self table disappeared. It also omitted latent-body, even though the saved launch arguments record shadow learning. The revised diagram shows the disabled table and the separate latent-body branch. The later causal-loop work\'s learned body head belongs to the forward/efference path; it is not this latent-body hierarchy.</p></details>\n</div>\n<div class="sg-topic" data-self-panel="choices" hidden>\n  <h3>The inspected planner searches through learned predictions</h3>\n  <p>A command is the set of numbers sent toward the body\'s motors. The <strong>motor controller</strong> is a learned network that proposes one. The <strong>forward model and prediction heads</strong> predict what a command will cause. The <strong>intention planner</strong> changes the command numbers, predicts again, and compares the result with a goal. During that search, the learned network weights stay fixed.</p>\n  <p>The ordinary planner chooses a goal from remembered sensory moments ranked using wellness. It then searches for a command whose prediction matches that goal. A separate <code>prospective_wellness</code> setting can add predicted wellness to the search; it is off in the reviewed October 15 run. The self-choice route below is another distinct option.</p>\n  <p>This is UEMA\'s <strong>model inversion</strong>: start with a desired predicted consequence and search for a command that produces it. The search itself need not own a network; it uses the learned forward model. Section 10.4 also specifies a learned attention module to compare several imagined sequences. That module exists in the source as <code>CrossSimulationAttention</code>, but is not used by the inspected MIMo bridge path. Saying that all of UEMA\'s intention system has no networks would be incorrect.</p>\n  <h3>Exactly what self-choice does</h3>\n  <p><strong>Self-choice is one optional implementation experiment.</strong> UEMA\'s spec calls for bodily consequences to influence commands. It does not require this particular six-candidate chooser. Its activation checks answer whether that implementation is useful; they do not define the whole solution.</p>\n  <p>It forecasts bodily function for six command sequences: the proposed sequence, a hold at the current pose, the remaining commands from the previous plan, and three sampled variations. It checks uncertainty and compares expected loss of sensing or motor capacity over time.</p>\n  <details><summary>The additional learning-opportunity comparison</summary><p>This optional comparison asks where recent experience has improved prediction while older predictions remain accurate. It first checks that predictions using the real command beat predictions with shuffled commands and predictions that simply repeat the present reading. It then scores each candidate\'s predicted sensory change against holding still, weighted by that measured learning progress. Only candidates whose predicted bodily function stays within the configured range can receive this preference.</p><p>The implementation uses the first command\'s sensory prediction, not a forecast of the whole movement\'s learning benefit. Its score is a reason to try an experience. Repeated physical ability tests must measure whether that experience actually teaches reaching or grasping.</p></details>\n  <p>This comparison uses learned forecasts, but its thresholds and comparison rule are supplied by code. The comparison rule is fixed; its predictions are learned. Its purpose is to test whether predictions of this body\'s consequences improve its choices.</p>\n  <div class="sg-table-scroll"><table><thead><tr><th>Setting</th><th>What runs</th><th>What reaches MIMo</th></tr></thead><tbody>\n  <tr><td><code>off</code></td><td>No self chooser is built. Self perception can still learn.</td><td>No command from self-choice.</td></tr>\n  <tr><td><code>shadow</code></td><td>A copy with fixed network weights computes the comparison and reports it.</td><td>The trainer discards that copy\'s commands. The ordinary actor continues.</td></tr>\n  <tr><td><code>on</code></td><td>A tested copy with fixed network weights uses the self-choice route. A separate learner can keep learning.</td><td>That tested copy\'s commands, through the execution path.</td></tr>\n  <tr><td>Internal <code>qualification</code></td><td>A trial mode inside copied/frozen actors; not a trainer command-line choice.</td><td>The caller controls trial execution. Shadow discards commands; copied-world qualification may execute them there.</td></tr>\n  </tbody></table></div>\n  <div class="sg-route"><strong>A concrete comparison</strong><p>Imagine two candidate movements: lean farther while keeping a hand planted, or shift that hand and then lean. If the first predicts a loss of sensing or motor capacity and the second predicts less loss, the self-choice rule can prefer the second when its forecasts are reliable enough to separate them. If neither predicts a function loss, that comparison alone does not explain why one better grasps the ball. Matching a remembered grasp goal and trying an experience that may teach something are separate reasons for choosing a command.</p><p>This illustrates the comparison rule, rather than reporting a measured reach.</p></div><h3>When was this intended to be turned on?</h3>\n  <p>The recorded plan was: first make bodily forecasts accurate, then compare forecast-guided commands with the existing actor and control comparisons on matched starting worlds. Enable the tested configuration after it improves useful choice, keeps earlier abilities, and works in held-out worlds &mdash; worlds not used to fit the model.</p>\n  <p><strong>There is a specific current blocker.</strong> The preserved branch uses remembered hand goals. The frozen self-choice actor rejects that hand-goal stage because it does not yet include its calibrated behavior in the frozen actor. The fixed copy must preserve the hand-goal settings and state, then pass physical comparisons before the switch can work with this branch.</p>\n  <p>If this self-choice experiment is selected when model work resumes, its next step is to complete the fixed actor\'s hand-goal support, verify preservation of its learned state, and run the matched physical comparison. Selecting a different implementation of UEMA\'s intended loop would need its own explicit design. Model work remains stopped for this documentation review.</p>\n  <details><summary>What the activation receipt checks</summary><p>The receipt must bind the exact self weights, other perception/action weights, body-function settings and command timing. It requires physical action effects, useful choice/acquisition gains, replay verification, retention and held-out worlds. The newer receipt also requires twice the prediction-error upper bound to be below the measured separation between command effects. This asks whether the forecast is accurate enough to tell the choices apart. It is more specific than a low average prediction error.</p></details>\n  <h3>A reach example: follow the information</h3>\n  <ol class="sg-story">\n    <li><strong>Before leaning:</strong> read current skin contact, available motor capacity and sensory context. Self perception updates its internal state.</li>\n    <li><strong>Before sending a command:</strong> the planner predicts consequences of candidate command numbers. A self-choice comparison, when enabled, also forecasts their bodily-function consequences.</li>\n    <li><strong>While the body moves:</strong> actual contact and movement change the next sensory readings. Those readings, not a desired success label, become the next prediction targets.</li>\n    <li><strong>After contact changes:</strong> compare earlier forecasts with what happened. Update the prediction networks. At the next permitted command update, planning can use the changed readings and learned predictions.</li>\n  </ol>\n  <p>Which coordinated command to use must come from learned predictions and choice. The four steps above describe the flow of information; they do not supply a fixed recipe of lean, reach and close fingers.</p>\n  <h3>Looking ahead does not mean waiting to react</h3><p>The <strong>history window</strong> is how much of the past the model reads. The <strong>planning horizon</strong> is how far into the future it predicts. The <strong>command duration</strong> is how long an order is applied before it can change. The <strong>observation interval</strong> is how often new readings arrive. These are four different times.</p><p>For the requested three-second reach, the design is to predict the whole movement, execute a short part, observe again, and revise what remains. A half-second phase in an illustration need not make the body blind to changes for half a second. The reviewed October 15 run\'s short-horizon settings are recorded separately in the detailed map; they do not implement the later three-second design.</p>\n  <details><summary>Why future readings must not leak into an earlier prediction</summary><p>To predict what happens just before a fall, the model may use everything sensed before that moment. It may imagine what comes next, but it may not read the later landing from the training example. That would teach it to use information unavailable when it must act.</p><p>The current forward model blocks later times in each sense\'s time-attention path. Its cross-sense attention path can mix all times supplied to it. Predicting the newest next step from a past-only input window is valid. Training a prediction at every position in a longer recorded sequence requires a past-only input for each position, or a time mask on every path that mixes times. The decisive check is to alter later observations and confirm that earlier predictions remain unchanged. This protects prediction-based planning; it does not prevent the planner from imagining a future.</p></details>\n</div>\n<div class="sg-topic" data-self-panel="feelings" hidden>\n  <h3>Knowing a condition and choosing a movement are different lessons</h3>\n  <p>Suppose the hand is sore. Three questions follow: <strong>What is happening now?</strong> <strong>What would this movement do?</strong> <strong>Which movement should happen?</strong> Training a network to answer one question does not automatically teach the other two.</p>\n  <div class="sg-table-scroll"><table><thead><tr><th>Job</th><th>What teaches it</th><th>What comes out</th></tr></thead><tbody>\n  <tr><td>Self perception and body prediction</td><td>Match current or later measured bodily function.</td><td>A body state and predicted consequences.</td></tr>\n  <tr><td>Learned wellness prediction</td><td>Match the wellness terms calculated from real experience.</td><td>Predicted wellness terms. The present approximate evaluator predicts five terms.</td></tr>\n  <tr><td>Planning</td><td>No planner weights change during search. Search changes command numbers using predicted outcomes and the selected comparison rule.</td><td>A selected command.</td></tr>\n  <tr><td>Teaching a command network</td><td>Change its weights using a stated lesson: achieved pose, planner commands, or a future-wellness objective are different choices.</td><td>A learned command proposal. Its dependence on prediction, wellness and intention determines its role in the architecture.</td></tr>\n  </tbody></table></div>\n  <p><strong>UEMA does not ban learned motor networks.</strong> Sections 9 and 27 explicitly include them. Section 1 rejects collapsing perception, prediction, self evaluation, intention and action into one direct observation-to-command policy; it also rejects a permanent reinforcement-learning substitute. Section 2 says wellness should guide which action is chosen. Therefore, a learned network that receives information about wellness is not automatically a violation.</p><p><strong>A gradient</strong> tells how changing one input would change a computed value. Searching with a gradient changes the command being considered. Learning with a gradient changes stored network weights. The spec describes both learned motor networks and command search, but does not fully specify every motor-network weight update. Any proposed lesson must name which weights change and show how prediction, grounded wellness and intention remain involved.</p><p><strong>Intrinsic</strong> means that the drive comes from the system\'s own condition, rather than task points such as "receive points for this ball." An intrinsic signal can feed either UEMA\'s prediction-and-intention route or a reward-trained action policy. Its source alone does not make the two architectures equivalent. Learning to predict wellness also needs a path by which that prediction affects commands.</p>\n  <p>An <strong>MLP</strong> is a small layered neural network. It can predict bodily condition, reconstruct motion, or produce commands. The important distinction is <strong>what its lesson changes it to do</strong>. Latent-body and a wellness-seeking command network can both use MLPs while learning different jobs.</p>\n  <details><summary>What the reviewed motor network actually learns</summary><p>The reviewed October 15 run uses <code>wellness_mode=baseline</code>. The motor network\'s movement lesson compares an earlier output with the pose the body later reached, with output-limit penalties. It also receives a wellness gradient from the present handle-position formula. Skin discomfort, lost capacity and body-map changes do not provide their own command direction through that gradient. Receiving that position gradient is different from training weights to maximize future wellness.</p><p>The loss manager also defines <code>direct</code> and <code>forward</code> wellness objectives. Direct replaces the pose-matching term with negative current wellness; forward adds negative predicted wellness alongside its pose term. Forward explicitly scores the controller output through learned predictions. The current-wellness input used by direct comes from a detached achieved pose in this bridge, so its name alone does not establish a learning path into command weights. The reviewed October 15 run uses baseline.</p><p>The current measured-wellness path uses formulas. A separate learned approximate evaluator copies five of their results and can evaluate predicted tokens. Additional goal-ranking channels are not all included in those five forecasts. Neither this evaluator nor <code>SelfState</code> alone is the full wellness-and-intention mechanism.</p></details>\n  <h3>What humans suggest we should keep separate</h3>\n  <p>Human brains do have circuits for movement and action planning. Brain imaging finds both areas linked to particular body parts and a network associated with whole-body action planning. <strong>Design implication:</strong> having movement networks is compatible with human biology; the study does not prescribe UEMA\'s exact networks or learning rule. <a href="https://pubmed.ncbi.nlm.nih.gov/37076628/">Gordon and colleagues, 2023</a>.</p>\n  <p>People also sense their internal body condition. This is called <strong>interoception</strong>. In a heartbeat study, activity in part of the insula was related to how accurately people sensed their heartbeat. <strong>Design implication:</strong> keep a learned body-state representation that can affect later perception and choice. <a href="https://pubmed.ncbi.nlm.nih.gov/14730305/">Critchley and colleagues, 2004</a>.</p>\n  <p>Pain intensity and how unpleasant pain feels can change separately. Physical pain and social rejection also have distinguishable brain activity patterns. <strong>Design implication:</strong> keep damage, lost function, unpleasantness and social context distinguishable before combining them for choice. One skin-pressure number should not stand for all of them. <a href="https://pubmed.ncbi.nlm.nih.gov/9252330/">Rainville and colleagues, 1997</a>; <a href="https://pubmed.ncbi.nlm.nih.gov/25400102/">Woo and colleagues, 2014</a>.</p>\n  <h3>Reflex, learned movement, and conscious self</h3>\n  <p>A <strong>reflex</strong> is a rapid response that does not require deliberate planning. The inspected code has a built-in gaze response: peripheral visual motion supplies an eye-orienting target that is blended into the gaze command. Its rule has no trainable weights. The reviewed run enables it.</p><p>UEMA section 9 instead aims for fast and careful responses to develop within one prediction-and-wellness mechanism, using different amounts of computation. The gaze rule is an implementation difference to evaluate, not evidence that the spec requires a permanent reflex layer. A fast learned movement is not automatically a separate reflex.</p>\n  <div class="sg-route"><strong>Reflex route, separate from intention planning</strong><p>Peripheral visual motion &rarr; gaze-reflex rule &rarr; eye target &rarr; gaze command.</p><p>This is a built-in response rule. It does not explain the full reach-and-grasp choice.</p></div>\n  <h3>UEMA\'s self also includes predictions about its own limits</h3>\n  <p>Section 23 goes beyond a body-state list. It asks the system to predict what will happen, how wrong that prediction may be, and how imagined wellness may differ from later real wellness. In the future Many Minds design, each subordinate retains its own prediction-and-wellness machinery for a mastered skill and can report an unexpected event. Predicting those reports adds another form of self knowledge.</p><p>The present self-function network already learns uncertainty: an estimate fitted to the size of its bodily-function forecast errors. That is one part of this idea. The inspected bridge does not expose the full set of section 23\'s own-error, imagined-versus-real wellness-gap and subordinate-alert predictions. These are distinct connections to map and test, not new names for the existing self-state numbers.</p>\n  <p><strong>UEMA already proposes emotional mechanisms.</strong> Section 22 describes stress as prediction errors that remain despite attempts to resolve them, and grief as the sustained consequences of losing something on which learned predictions and wellness depended. These are the spec\'s hypotheses about emergent experience. They do not require a programmer to label a state "fear" or "grief."</p><p>To evaluate that account, follow a concrete loss: a useful thing becomes unavailable; predictions that depended on it fail; attempts to restore the expected outcome have consequences; effort and capacity change; learning changes later predictions and choices. The named-thing associations and bodily self forecasts cover parts of this chain. The full sequence needs evidence linking those changes over time. A brief prediction error, by itself, does not distinguish surprise from the proposed persistent distress.</p>\n  <h3>A body perspective can develop through experience</h3><p>A small child and an adult have different reach, strength, memories and expectations. For UEMA, the useful design is to carry current capacities and learned history into the self state. Then the same event can predict different consequences as the body and experience change. Age alone should not prescribe a feeling or a movement.</p><h3>The design direction I recommend</h3>\n  <ol class="sg-story">\n    <li><strong>Preserve the whole UEMA loop.</strong> Keep bodily state, control of body and objects, remembered relations, and prediction reliability connected to intention. Teach forecasts from the consequences that actually follow commands.</li>\n    <li><strong>Keep wellness learning grounded in experience.</strong> Predict bodily condition and its consequences before ranking movements. Persistent damage and short pain events need distinct inputs and timescales, so a brief successful catch is not treated like the fall it prevented.</li>\n    <li><strong>Make perception usable by intention.</strong> Compare the consequences of a full timed movement, including a temporary cost that prevents a worse outcome. Predict ahead, execute part, read the body again and revise the remaining commands. The optional self-choice experiment is one candidate implementation; it is not the definition of UEMA.</li>\n    <li><strong>State every command-learning lesson openly.</strong> Name its inputs, time horizon, teacher and exact weight update. Check whether real and imagined consequences still guide action. A reward-trained replacement or a network that only copies planner commands changes the design; it must not enter under the name of a prediction repair.</li>\n    <li><strong>Give emotional context a measurable account.</strong> Before adding a new preference, specify what persists in self state, which observations update it, and how its predicted consequences affect choice. Test changed context with the same bodily condition, and changed bodily condition with the same context. This separates context learning from relabelling physical pain.</li>\n  </ol>\n  <div class="sg-route"><strong>Three designs, with different commitments</strong><p><strong>UEMA prediction and search:</strong> experience teaches body and wellness forecasts; the motor network proposes commands; intention compares predicted consequences while changing command numbers. The design retains the separate five jobs.</p><p><strong>Future-wellness command training:</strong> future wellness also supplies a lesson that changes command-network weights. Its compatibility depends on the exact objective and which prediction/self/intention paths remain necessary. Neither the word "intrinsic" nor the word "MLP" settles that question.</p><p><strong>Copying planner commands into a shortcut:</strong> commands selected by the planner become targets for another command network. This is a form of policy distillation. Section 26 instead consolidates mastered skills through Many Minds, retaining prediction and wellness in each subordinate. Section 20\'s imitation stream teaches sensory consequences of demonstrated movement; that is a different lesson.</p></div><p>The operator has opened the policy restriction for careful evaluation. That permits comparison of an explicit revision; it does not make the alternatives part of the existing spec. The default design direction here is to preserve perception, prediction, grounded wellness and intention, and name any proposed departure before adopting it.</p>\n</div>\n<details class="sg-evidence"><summary>Code, saved settings and review scope</summary>\n<p>This explanation was checked against the repository on October 28. The reviewed explainer measurements are from <code>B1K_skip13_492800</code>, generated October 15. They are not a live training display. The later preserved <code>TUNBL_u_496250_a2</code> branch also records self learning, self-choice off and latent-body shadow; it uses remembered hand goals.</p>\n<p>For the reviewed October 15 run, its original <code>args.json</code> confirms <code>latent_body_hierarchy=shadow</code>, <code>latent_planner=off</code>, <code>gaze_reflex=on</code>, <code>hierarchical_self=off</code>, <code>entity_self=dark</code>, and a separate self-value reader weight of 0.1. Those omitted settings are shown from an explicitly named source; the saved measurement blocks are preserved.</p>\n<ul>\n<li><code>lem/model/self_state.py</code>; <code>mimo_bridge/self_state_integration.py</code>; <code>mimo_bridge/lem_model.py</code>: learned state, timed forecasts, separate self learning and conditional perception output.</li>\n<li><code>mimo_bridge/self_intention.py</code>; <code>acting_snapshot.py</code>; <code>train_mimo.py</code>: self-choice modes, evidence requirements and hand-goal incompatibility.</li>\n<li><code>lem/model/intention_generator.py</code>; <code>motor_controller.py</code>; <code>wellness.py</code>; <code>loss_manager.py</code>: command search, motor lessons, grounded and approximate wellness, alternative training objectives.</li>\n<li><code>lem/model/body_schema.py</code>; <code>mimo_bridge/hierarchical_self.py</code>; <code>entity_self.py</code>; <code>entity_presence.py</code>; <code>self_value_reader.py</code>: the three association mechanisms and conditional goal-value output.</li>\n<li><code>lem/latent_body/model.py</code>, <code>hierarchy.py</code>, <code>losses.py</code>; <code>mimo_bridge/latent_body_shadow.py</code>, <code>latent_body_clone.py</code>; <code>gaze_reflex.py</code>: movement hierarchy, copied-world command tests and direct eye response.</li>\n<li><code>docs/experiment_review_2026-09-13/after_predictor_implementation_sequence.md</code>: intended development, comparison and adoption order.</li>\n</ul>\n<p>Review record: <code>Saved/h84/explainer_connections_20261028/evidence/</code>. This work changes explanation and diagrams. The stopped model work is not resumed.</p>\n</details>\n</section>\n', 'reviewed_run': 'B1K_skip13_492800', 'reviewed_args': {'latent_body_hierarchy': 'shadow', 'latent_body_levels': 'arm,hand,index', 'latent_planner': 'off', 'gaze_reflex': 'on'}, 'reviewed_args_path': 'D:\\UEMA_Research\\external\\D\\UEMA_scratch\\b1_continue_492800\\run\\B1K_skip13_492800\\args.json', 'reviewed_args_sha256': 'fef8accf15c722155b26ffd922d19755bf7ba9f150eb3c10a5e258d68b760413'},
    'node.latent_body': {'t': 'latent-body movement hierarchy', 'h': 'See the code sources in the self architecture overview.', 'plain': 'Latent-body is a separate learned hierarchy of body parts. Its small networks receive part states, nearby-part information, commands and observed outcomes. They learn compact movement codes, predict movement/contact, reconstruct experienced commands and coordinate codes across parent and child parts.\n\nIts lessons match observed outcomes and agreement across levels; they do not maximize wellness. In shadow mode it trains independently and saves its learned state but supplies no production command. The latent planner is separately off in the reviewed October 15 run. Copied-world comparison tools can execute its decoded commands in test worlds.', 'w': 'Latent-body is a separate learned hierarchy of body parts. Its small networks receive part states, nearby-part information, commands and observed outcomes. They learn compact movement codes, predict movement/contact, reconstruct experienced commands and coordinate codes across parent and child parts.\n\nIts lessons match observed outcomes and agreement across levels; they do not maximize wellness. In shadow mode it trains independently and saves its learned state but supplies no production command. The latent planner is separately off in the reviewed October 15 run. Copied-world comparison tools can execute its decoded commands in test worlds.'},
    'node.self_choice': {'t': 'self-choice comparison', 'h': 'See the code sources in the self architecture overview.', 'plain': "Self-choice compares six timed command sequences using learned forecasts of bodily function: the proposed plan, a hold, the remaining commands of the previous plan, and three sampled variations. It checks uncertainty and predicted deficits in sensing or motor capacity over time; an optional learning-opportunity comparison can rank candidates with no predicted function deficit. That additional score uses first-command sensory changes relative to holding still, weighted by measured prediction progress and retention checks; it is not a whole-movement skill forecast. The comparison rule is fixed; the forecasts are learned.\n\nOff builds no chooser. Shadow runs a frozen comparison and discards its commands. On uses the qualified frozen actor's commands. Qualification is an internal trial mode, not a trainer CLI choice. The preserved remembered-hand-goal branch cannot simply enable this: the current frozen actor rejects that hand-goal stage until its contract is extended and qualified.", 'w': "Self-choice compares six timed command sequences using learned forecasts of bodily function: the proposed plan, a hold, the remaining commands of the previous plan, and three sampled variations. It checks uncertainty and predicted deficits in sensing or motor capacity over time; an optional learning-opportunity comparison can rank candidates with no predicted function deficit. That additional score uses first-command sensory changes relative to holding still, weighted by measured prediction progress and retention checks; it is not a whole-movement skill forecast. The comparison rule is fixed; the forecasts are learned.\n\nOff builds no chooser. Shadow runs a frozen comparison and discards its commands. On uses the qualified frozen actor's commands. Qualification is an internal trial mode, not a trainer CLI choice. The preserved remembered-hand-goal branch cannot simply enable this: the current frozen actor rejects that hand-goal stage until its contract is extended and qualified."},
    'node.self_report': {'t': 'self report', 'h': 'See the code sources in the self architecture overview.', 'plain': 'This report sends measured bodily function, self forecasts and prediction errors to the researcher. No movement command leaves this box. The separate self-learning calculation updates the self networks; that is the return arrow in the diagram.', 'w': 'This report sends measured bodily function, self forecasts and prediction errors to the researcher. No movement command leaves this box. The separate self-learning calculation updates the self networks; that is the return arrow in the diagram.'},
}


ENV_STEP_FALLBACK_S = 0.005 * 2
"""The crib scene's MuJoCo timestep times BridgeEnv's frame_skip - the same product as
train_mimo.ENV_DT_ASSUMED_S. Used ONLY for a receipt written before the trainer began
recording the env step it measured (env_dt_s, model_step_s)."""


def moment_clock(receipt: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """The length of one env step and one model step for THIS page, from one place.

    The receipt's measured env_dt_s and model_step_s when the run recorded them; else the
    scene's own 0.005 s x 2 times this run's --action-repeat (launcher default 5).
    'measured' says which of the two the page is printing."""
    reps = max(1, int(args.get("action_repeat") or 5))
    env_s = float(receipt.get("env_dt_s") or ENV_STEP_FALLBACK_S)
    step_s = float(receipt.get("model_step_s") or env_s * reps)
    return {"env_step_s": env_s, "action_repeat": reps, "model_step_s": step_s,
            "measured": bool(receipt.get("env_dt_s"))}


def moment_fill(receipt: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, str]:
    """The {placeholders} the DOC prose carries for the model step's length, filled for one run."""
    c = moment_clock(receipt, args)
    hold = max(1, int(args.get("motor_hold") or 5))
    source = ("the env step this run measured on its own body and wrote into its receipt"
              if c["measured"] else
              "the crib scene's 0.005 s MuJoCo timestep times 2, assumed because this run's "
              "receipt was written before the trainer recorded the step it measured")
    return {"{env_step_ms}": f"{c['env_step_s'] * 1000:g}",
            "{env_step_source}": source,
            "{action_repeat}": str(c["action_repeat"]),
            "{moment_ms}": f"{c['model_step_s'] * 1000:g}",
            "{motor_hold}": str(hold),
            "{hold_ms}": f"{c['model_step_s'] * hold * 1000:g}"}


LITE_SHAPE_DEFAULTS = {"d_model": 128, "core_blocks": 2, "context_length": 4,
                       "ffn_expansion": 4, "vision_layers": 2, "encoder_layers": 1}
"""What an unset (0) size flag builds: mimo_bridge/lem_model.py LITE_OVERRIDES and the help
text of --d-model, --core-blocks, --context-length, --vision-swin-layers and
--encoder-layers in train_mimo.py. Used ONLY when neither the receipt nor args.json
records the built size; model_shape()['source'] says which of the three each number came
from, so a page never prints a default as if the run had recorded it."""


def _inventory_tensors(receipt: Dict[str, Any], component: str) -> Dict[str, Any]:
    comp = (((receipt.get("checkpoint_inventory") or {}).get("components") or {})
            .get(component) or {})
    return comp.get("tensors") or {}


def gaze_head_reused(receipt: Dict[str, Any]) -> bool:
    """True when this run's efference gaze head trains the GazeHead's own layers. The receipt's
    efference_heads stamp says so in gaze_head_repurposed (LemLite writes it: False when the heads
    were built with a gaze pool of their own, which b1 minor 5's third remedy would make the rule);
    a stamp without the field predates it, and every such build reused the layers."""
    st = receipt.get("efference_heads")
    if not st:
        return False
    return bool(st.get("gaze_head_repurposed", True)) if isinstance(st, dict) else True


def clip_groups(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """The gradient-clipping sets LemLite._clip_training_gradients builds for THIS run, from its
    receipt: everything else in one set, then one set each for self perception (only in 'learn'
    mode), the efference heads together with the GazeHead layers they repurpose, the contact head
    and the motor synergy - each only when this run built it. An earlier page said "split in two"
    for every run; with --efference-heads on that was false (review b1, minor 5). 'known' is False
    when the receipt's inventory lacks a count, and the page then says so instead of a number."""
    inv = receipt.get("checkpoint_inventory") or {}
    comps = inv.get("components") or {}
    opt_n = int(inv.get("optimizer_parameters") or 0)
    opt_t = int(inv.get("optimizer_parameter_tensors") or 0)

    def counts(*names: str) -> Optional[Tuple[int, int]]:
        n = t = 0
        for nm in names:
            c = comps.get(nm) or {}
            if not c.get("parameters"):
                return None
            n, t = n + int(c["parameters"]), t + int(c.get("parameter_tensors") or 0)
        return n, t

    groups: List[Dict[str, Any]] = []
    if str(receipt.get("self_state") or "") == "learn":
        groups.append({"who": "self perception's", "counts": counts("self_state")})
    if receipt.get("efference_heads"):
        # the code clips the GazeHead's parameters in this set whenever a GazeHead exists, reused
        # or not; a build without a GazeHead clips the heads alone
        with_gaze = "gaze_head" in comps and gaze_head_reused(receipt)
        who = ("the efference heads' (with the GazeHead layers they repurpose)" if gaze_head_reused(receipt)
               else "the efference heads' (with the unused GazeHead's layers, which the code clips "
                    "in the same set)" if with_gaze else "the efference heads'")
        groups.append({"who": who, "counts": (counts("efference_heads", "gaze_head") if with_gaze
                                              else counts("efference_heads"))})
    if receipt.get("contact_head"):
        groups.append({"who": "the contact head's", "counts": counts("contact_head")})
    if receipt.get("motor_synergy"):
        groups.append({"who": "the motor synergy's", "counts": counts("motor_synergy")})
    if (receipt.get("liveness_repair") or {}).get("mode") == "on":
        groups.append({"who": "the liveness repair's", "counts": counts("liveness_repair")})
    known = bool(opt_n and opt_t) and all(g["counts"] is not None for g in groups)
    main_n = opt_n - sum(g["counts"][0] for g in groups) if known else None
    main_t = opt_t - sum(g["counts"][1] for g in groups) if known else None
    return {"groups": groups, "known": known, "main_numbers": main_n, "main_tensors": main_t,
            "n_sets": 1 + len(groups)}


def clip_sets_text(receipt: Dict[str, Any]) -> str:
    """One clause naming this run's clipping sets and their sizes, for the backpropagation, weight
    update and gradient-line popups."""
    c = clip_groups(receipt)
    if c["n_sets"] == 1:
        return "once, over every optimized number together"
    head = f"{_word(c['n_sets'])} times, over {_word(c['n_sets'])} separate sets"
    if not c["known"]:
        return (head + ": the rest of the model, then " + _and_list([g["who"] for g in c["groups"]])
                + " own numbers (sizes this run's receipt does not carry)")
    parts = [f"the rest of the model's {c['main_numbers']:,} numbers in {c['main_tensors']:,} tensors"]
    parts += [f"{g['who']} {g['counts'][0]:,} numbers in {g['counts'][1]:,} tensors"
              for g in c["groups"]]
    return head + ": " + _and_list(parts)


def idle_weights_text(receipt: Dict[str, Any], run_name: str) -> Dict[str, str]:
    """Which optimized weights no lesson reaches in THIS run. The voice head never has a lesson.
    The GazeHead has none unless --efference-heads on repurposes its layers as the efference gaze
    head, whose 50 ms eye-turn lesson then trains them - so "the gaze head's weights never move"
    is true only for a run without the heads (review b1, minor 5)."""
    cp = receipt.get("component_params") or {}
    inv = receipt.get("checkpoint_inventory") or {}
    total = int(receipt.get("total_params") or inv.get("optimizer_parameters") or 0)
    voice = cp.get("voice_head") if (receipt.get("voice_head") or {}).get("enabled", "voice_head" in cp) else None
    gaze = cp.get("gaze_head")
    eff = gaze_head_reused(receipt)
    idle = [("the voice head's", voice)] if voice else []
    if gaze and not eff:
        idle.append(("the gaze head's", gaze))
    moving = (f" The gaze head's {int(gaze):,} numbers DO move in {run_name}: --efference-heads on "
              f"reuses its attention and layers as the efference gaze head, and that head's 50 ms "
              f"eye-turn lesson trains them." if gaze and eff else "")
    if not idle:
        sentence = "Every optimized weight in " + run_name + " has a lesson that can reach it." + moving
        heads = "no head"
    else:
        n_idle = sum(int(v) for _, v in idle)
        share = f"{100.0 * n_idle / total:.2f}%" if total else "a share this run's receipt does not carry"
        names = _and_list([f"{w} {int(v):,}" for w, v in idle])
        verb = "either one" if len(idle) > 1 else "them"
        sentence = (f"{n_idle:,} weights never move at all: {names} sit in the optimizer's list, "
                    f"but no lesson grades {verb} and "
                    f"{'both' if len(idle) > 1 else 'they'} measured a gradient of exactly zero on "
                    f"every live step and every re-study step - {share} of the model's "
                    f"{total:,} optimized numbers." + moving)
        heads = _and_list([f"{w} {int(v):,}" for w, v in idle])
    use = (f"(An older gaze-head network rides in the checkpoint; in {run_name} --efference-heads on "
           f"reuses its attention and layers as the efference gaze head, which predicts how far the "
           f"eyes will turn and trains on it - it never chooses where to look.)" if eff else
           f"(An older gaze-head network still rides in the checkpoint so old checkpoints load; "
           f"nothing reads its output - the footer line counts it as unused. In {run_name} the "
           f"efference gaze head has layers of its own, so the GazeHead's are not reused.)"
           if receipt.get("efference_heads") else
           "(An older gaze-head network still rides in the checkpoint so old checkpoints load; "
           "nothing reads its output - the footer line counts it as unused. A run launched with "
           "--efference-heads on reuses its layers as the efference gaze head, and there they "
           "train.)")
    return {"{idle_weights_sentence}": sentence, "{no_gradient_heads}": heads,
            "{gaze_head_use}": use}


# The gate line's measurement as written beside INTEGRITY_GATE_LINE in lem/model/wellness_channels.py
# when the page quotes it; a line with no entry here is sent to that comment instead of being
# described from memory.
_GATE_LINE_SOURCES = {
    0.87: ("the largest pain of any model step the checkpoint lived through in b3-wellness's runs (980 "
           "model steps; routine pain sat at 77%, give or take 4%, once the skin's sustained load had "
           "built up)"),
}


def integrity_gate_fill() -> Dict[str, str]:
    """The integrity gate as THIS tree's lem/model/wellness_channels.py holds it, read when the page
    is built, never typed (verify-fix-explainer, 2026-09-23): which reading it gates on (skin pain,
    or - where the module has a HARM_ROW - the worse of skin pain and a limb pressed into the end of
    its reach), where the line sits, and what rank_values does when every remembered model step of a
    body is over it: it gives that body its least harmful model step, so "never chosen" holds only
    while some model step is under the line (review b3-wellness, minor: "Two user-visible sentences
    state what the code does not do")."""
    try:
        from lem.model import wellness_channels as WCM
    except Exception as exc:                                  # a page is still built without it
        return {"{integrity_gate_text}": (
            "It is also a GATE on which remembered model steps may become a goal; its line could not "
            f"be read from lem/model/wellness_channels.py when this page was built "
            f"({type(exc).__name__}).")}
    line = getattr(WCM, "INTEGRITY_GATE_LINE", None)
    harm = hasattr(WCM, "HARM_ROW")
    what = ("harm reading - the worse of skin pain and a limb pressed into the end of its reach, "
            "on the pain scale -" if harm else "skin pain reading")
    short = "harm" if harm else "pain"
    if line is None:
        return {"{integrity_gate_text}": (
            f"The {what} can also be a GATE, but this tree's INTEGRITY_GATE_LINE is unset, so no "
            "remembered model step is gated.")}
    at = f"{round(100.0 * float(line), 1):g}%"
    source = _GATE_LINE_SOURCES.get(round(float(line), 4))
    why = (f"The line sits at {at} because that is {source}." if source else
           "How that line was measured is written beside INTEGRITY_GATE_LINE in "
           "lem/model/wellness_channels.py.")
    return {"{integrity_gate_text}": (
        f"The {what} is also a GATE: a remembered model step whose {short} is over {at} of the way from "
        "no modelled aversion to an extreme load is never chosen as a goal while any remembered "
        "model step of that body is under the line. When every one is over it, the least harmful is "
        "chosen, because a planner must aim somewhere; the wellness_rank block counts how often "
        f"that happens (all_gated_share). {why}")}


def model_shape(receipt: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """The size of the model THIS page describes, from one place (the 2026-09-19 H84 sync).

    The size is a launch setting (--d-model --core-blocks --vision-swin-layers
    --vision-width --encoder-layers --encoder-width --context-length), so no DOC string,
    chart caption or viewer.js label may type it. Every number here is read from the
    run's model receipt; args.json is the fallback for a receipt that does not carry the
    field, LITE_SHAPE_DEFAULTS the fallback for a flag left at 0. 'source' names which.
    Derived sizes use the constructing code's own formula:
      memory_event   (n_senses + 1) x d_model + 10 + n_handles x 9 - the
                     PersistentMemory(inputs, ...) line in mimo_bridge/lem_model.py; the
                     recurrent network reads 2 time numbers more (memory_event_timed)
      motor_in       the receipt's motor_controller._input_dim, else
                     n_senses x d_model + n_handles x 10 + n_handles x 9
      wellness_in    the receipt's wellness.approximate_input_dim, else d_model for every
                     sense except intermind
      intermind_row  d_model + 3 (lem/config/receipt.py _embed_input_dim)
      recon_hidden   2 x d_model, and order_proj_out 3 x d_model (mimo_bridge/lem_model.py)
    The core's learned-number counts are summed from the receipt's per-tensor inventory
    with the rule viewer.js subParam uses (every non-buffer tensor under a key prefix).
    A receipt without an inventory gets torch's own module sizes instead - an attention
    stage is 4d^2 + 4d plus its LayerNorm's 2d, the feed-forward network is
    2 x d x (e x d) + e x d + d plus its LayerNorm's 2d - and 'core_counts_from' says
    which of the two the page printed."""
    enc = receipt.get("encoders_built") or {}
    mods = list(receipt.get("active_modalities") or enc.keys())
    fm = receipt.get("forward_model") or {}
    src: Dict[str, str] = {}

    def pick(name: str, from_receipt: Any, arg_key: str) -> int:
        if from_receipt:
            src[name] = "receipt"
            return int(from_receipt)
        if args.get(arg_key):
            src[name] = "args"
            return int(args[arg_key])
        src[name] = "default"
        return int(LITE_SHAPE_DEFAULTS[name])

    first = enc.get(mods[0]) if mods and isinstance(enc.get(mods[0]), dict) else None
    if first is None:
        first = next((v for v in enc.values() if isinstance(v, dict)), {})
    d = pick("d_model", first.get("d_output"), "d_model")
    n_senses = int(fm.get("_n_modalities") or len(mods) or 6)
    n_blocks = pick("core_blocks", receipt.get("core_blocks") or fm.get("n_layers"),
                    "core_blocks")
    seq = int(fm.get("_seq_length") or 0)
    ctx = pick("context_length", seq // n_senses if seq and n_senses else 0,
               "context_length")
    seq = seq or ctx * n_senses
    n_handles = len(receipt.get("handle_names") or []) or 24
    order_numbers = n_handles * int((receipt.get("motor_controller") or {})
                                    .get("output_per_handle") or 9)

    fmt = _inventory_tensors(receipt, "forward_model")
    regt = _inventory_tensors(receipt, "encoder_registry")

    def learned(tensors: Dict[str, Any], *prefixes: str) -> int:
        return sum(int(v.get("elements") or 0) for k, v in tensors.items()
                   if v.get("kind") != "buffer" and any(k.startswith(p) for p in prefixes))

    ffn_w0 = (fmt.get("blocks.0.ffn.0.weight") or {}).get("shape") or []
    if len(ffn_w0) == 2 and d and ffn_w0[1] == d:
        ffn_e = int(ffn_w0[0]) // d
        src["ffn_expansion"] = "receipt"
    else:
        ffn_e = pick("ffn_expansion", fm.get("ffn_expansion"), "ffn_expansion")
    if fmt:
        stage = learned(fmt, "blocks.0.cm_")
        ffn = learned(fmt, "blocks.0.ffn")
        block = learned(fmt, "blocks.0.")
        blocks_all = learned(fmt, "blocks.")
        tags = learned(fmt, "modality_embedding", "temporal_embedding")
        fnorm = learned(fmt, "final_norm")
        masks = sum(int(v.get("elements") or 0) for v in fmt.values()
                    if v.get("kind") == "buffer")
        core_learned = learned(fmt, "")
        lnorm_numbers = sum(int(v.get("elements") or 0) for k, v in fmt.items()
                            if v.get("kind") != "buffer" and "norm" in k)
        counts_from = "checkpoint_inventory"
    else:
        stage = 4 * d * d + 4 * d + 2 * d
        ffn = 2 * d * (ffn_e * d) + ffn_e * d + d + 2 * d
        block = 3 * stage + ffn
        blocks_all = n_blocks * block
        tags = n_senses * d + ctx * d
        fnorm = 2 * d
        masks = n_blocks * ctx * ctx
        core_learned = blocks_all + tags + fnorm
        lnorm_numbers = (4 * n_blocks + 1) * 2 * d
        counts_from = "formula"
    vis_layers = len({k.split(".")[3] for k in regt
                      if k.startswith("encoders.vision.blocks.")})
    enc_layers = len({k.split(".")[3] for k in regt
                      if k.startswith("encoders.tactile.layers.")})
    cp = receipt.get("component_params") or {}
    # the optimizer's two weight-decay groups (split by tensor dimension, as
    # mimo_bridge/lem_model.py splits them by p.dim()) and the two gradient-clipping sets
    # (self perception apart from everything else). Published only when the inventory's
    # own tensors add up to the optimizer total it records - a partial inventory gives None.
    inv = receipt.get("checkpoint_inventory") or {}
    comps = inv.get("components") or {}
    dec_t = dec_n = flat_t = flat_n = 0
    for c in comps.values():
        for v in (c.get("tensors") or {}).values():
            if v.get("kind") == "buffer":
                continue
            if len(v.get("shape") or []) >= 2:
                dec_t, dec_n = dec_t + 1, dec_n + int(v.get("elements") or 0)
            else:
                flat_t, flat_n = flat_t + 1, flat_n + int(v.get("elements") or 0)
    opt_n = int(inv.get("optimizer_parameters") or 0)
    opt_t = int(inv.get("optimizer_parameter_tensors") or 0)
    groups = None
    if opt_n and dec_n + flat_n == opt_n:
        groups = {"decay_tensors": dec_t, "decay_numbers": dec_n,
                  "no_decay_tensors": flat_t, "no_decay_numbers": flat_n}
    ss = comps.get("self_state") or {}
    clip = None
    groups_c = clip_groups(receipt)
    if opt_n and opt_t and groups_c["known"]:
        clip = {"main_numbers": groups_c["main_numbers"],
                "main_tensors": groups_c["main_tensors"],
                "self_state_numbers": int(ss.get("parameters") or 0),
                "self_state_tensors": int(ss.get("parameter_tensors") or 0),
                "groups": groups_c["groups"]}
    memt = _inventory_tensors(receipt, "persistent_memory")
    mem_readout = learned(memt, "readout") if memt else None
    sensory = [m for m in mods if m != "intermind"]
    embed = (regt.get("encoders.intermind.embed.weight") or {}).get("shape") or []
    return {
        "d_model": d, "n_senses": n_senses, "token_row": n_senses * d,
        "core_blocks": n_blocks, "context_length": ctx, "seq_len": seq,
        "n_heads": int(fm.get("n_heads") or 0),
        "ffn_expansion": ffn_e, "ffn_width": ffn_e * d,
        "n_handles": n_handles, "order_numbers": order_numbers,
        "recon_hidden": 2 * d, "order_proj_out": 3 * d,
        "wellness_in": int((receipt.get("wellness") or {}).get("approximate_input_dim")
                           or len(sensory) * d),
        "motor_in": int((receipt.get("motor_controller") or {}).get("_input_dim")
                        or n_senses * d + n_handles * 10 + order_numbers),
        "memory_event": (n_senses + 1) * d + 10 + order_numbers,
        "memory_event_timed": (n_senses + 1) * d + 10 + order_numbers + 2,
        "intermind_row": int(embed[1]) if len(embed) == 2 else d + 3,
        "vision_layers": pick("vision_layers", vis_layers, "vision_swin_layers"),
        "vision_width": int((enc.get("vision") or {}).get("d_internal")
                            or args.get("vision_width") or d),
        "encoder_layers": pick("encoder_layers", enc_layers, "encoder_layers"),
        "encoder_width": int((enc.get("tactile") or {}).get("d_internal")
                             or args.get("encoder_width") or d),
        "core_stage_learned": stage, "core_ffn_learned": ffn, "core_block_learned": block,
        "core_blocks_learned": blocks_all, "core_tag_tables": tags,
        "core_final_norm": fnorm, "core_mask_cells": masks,
        "core_learned": core_learned, "core_saved": core_learned + masks,
        "core_layernorms": 4 * n_blocks + 1, "core_layernorm_numbers": lnorm_numbers,
        "core_counts_from": counts_from,
        "total_params": int(receipt.get("total_params") or 0),
        "component_params": {k: int(v) for k, v in cp.items()},
        "optimizer_groups": groups, "clip_sets": clip, "memory_readout": mem_readout,
        "source": src,
    }


SHAPE_NOT_RECORDED = "a count this run's receipt does not carry"


def shape_fill(receipt: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, str]:
    """The {placeholders} the DOC prose and the chart captions carry for the model's size,
    filled for one run. Counts print with thousands separators, the way the prose wrote
    them; a share prints as a percentage of the receipt's total_params. A count the
    receipt does not carry prints SHAPE_NOT_RECORDED, never another run's number."""
    s = model_shape(receipt, args)
    cp, total = s["component_params"], s["total_params"]

    def n(v: Any) -> str:
        return f"{int(v):,}"

    def comp(*keys: str) -> str:
        vals = [cp.get(k) for k in keys]
        return n(sum(vals)) if vals and all(v is not None for v in vals) \
            else SHAPE_NOT_RECORDED

    def pct(*keys: str) -> str:
        vals = [cp.get(k) for k in keys]
        if not total or not vals or any(v is None for v in vals):
            return "a share this run's receipt does not carry"
        return f"{100.0 * sum(vals) / total:.2f}%"

    enc_keys = [k for k in cp if k.startswith("enc_")]
    g, c, mr = s["optimizer_groups"], s["clip_sets"], s["memory_readout"]
    ctx = s["context_length"]
    step_s = moment_clock(receipt, args)["model_step_s"]
    rb = int(args.get("replay_batch") or 48)
    need = max(4 * rb, 8 * ctx)               # train_mimo.py's --fill-ring-first gate
    return {
        "{d_model}": str(s["d_model"]), "{n_blocks}": str(s["core_blocks"]),
        "{token_row}": n(s["token_row"]),
        "{context_length}": n(ctx), "{context_minus_1}": n(max(0, ctx - 1)),
        "{seq_len}": n(s["seq_len"]),
        "{seq_older}": n(max(0, s["seq_len"] - s["n_senses"])),
        "{window_s}": f"{ctx * step_s:.2f}",
        "{mask_cells_one}": n(ctx * ctx), "{mask_blocked_one}": n(ctx * (ctx - 1) // 2),
        "{ffn_expansion}": str(s["ffn_expansion"]), "{ffn_width}": n(s["ffn_width"]),
        "{recon_hidden}": n(s["recon_hidden"]), "{order_proj_out}": n(s["order_proj_out"]),
        "{wellness_in}": n(s["wellness_in"]), "{motor_in}": n(s["motor_in"]),
        "{memory_event}": n(s["memory_event"]),
        "{memory_event_timed}": n(s["memory_event_timed"]),
        "{intermind_row}": n(s["intermind_row"]),
        "{vision_layers}": str(s["vision_layers"]),
        "{encoder_layers}": str(s["encoder_layers"]),
        "{core_stage_learned}": n(s["core_stage_learned"]),
        "{core_ffn_learned}": n(s["core_ffn_learned"]),
        "{core_block_learned}": n(s["core_block_learned"]),
        "{core_blocks_learned}": n(s["core_blocks_learned"]),
        "{core_tag_tables}": n(s["core_tag_tables"]),
        "{core_final_norm}": n(s["core_final_norm"]),
        "{core_mask_cells}": n(s["core_mask_cells"]),
        "{core_learned}": n(s["core_learned"]), "{core_saved}": n(s["core_saved"]),
        "{core_layernorms}": str(s["core_layernorms"]),
        "{core_layernorm_numbers}": n(s["core_layernorm_numbers"]),
        "{core_layernorm_pct}": (
            f"{100.0 * s['core_layernorm_numbers'] / s['core_learned']:.1f}%"
            if s["core_learned"] else "a share this run's receipt does not carry"),
        "{total_params}": n(total) if total else SHAPE_NOT_RECORDED,
        "{enc_motor_params}": comp("enc_motor"), "{enc_motor_pct}": pct("enc_motor"),
        "{enc_vision_params}": comp("enc_vision"),
        "{encoders_params}": comp(*enc_keys),
        "{heads_params}": comp("prediction_heads"),
        "{recon_params}": comp("recon_decoders"),
        "{voice_params}": comp("voice_head"), "{gaze_params}": comp("gaze_head"),
        "{idle_heads_params}": comp("voice_head", "gaze_head"),
        "{idle_heads_pct}": pct("voice_head", "gaze_head"),
        "{memory_params}": comp("persistent_memory"),
        "{self_state_params}": comp("self_state"),
        "{decay_tensors}": n(g["decay_tensors"]) if g else SHAPE_NOT_RECORDED,
        "{decay_numbers}": n(g["decay_numbers"]) if g else SHAPE_NOT_RECORDED,
        "{no_decay_tensors}": n(g["no_decay_tensors"]) if g else SHAPE_NOT_RECORDED,
        "{no_decay_numbers}": n(g["no_decay_numbers"]) if g else SHAPE_NOT_RECORDED,
        "{clip_main_numbers}": n(c["main_numbers"]) if c else SHAPE_NOT_RECORDED,
        "{clip_main_tensors}": n(c["main_tensors"]) if c else SHAPE_NOT_RECORDED,
        "{self_state_tensors}": n(c["self_state_tensors"]) if c else SHAPE_NOT_RECORDED,
        "{memory_readout}": n(mr) if mr else SHAPE_NOT_RECORDED,
        "{memory_not_restudied}": (n(cp["persistent_memory"] - mr)
                                   if mr and cp.get("persistent_memory") else SHAPE_NOT_RECORDED),
        "{replay_every}": str(int(args.get("replay_every") or 4)),
        "{replay_batch}": str(rb), "{replay_floor}": str(max(2, rb // 4)),
        "{replay_window}": n(ctx + 1), "{replay_need}": n(need),
        "{replay_need_s}": f"{need * step_s:.0f}",
    }


# ================================================= H84 items, second pass (2026-09-19)
# Facts the first H84 sync left typed because each needed a measurement or a decision: the
# planner's cadence against the command hold, the imagination depth, the replay ring's size,
# each encoder's own depth and width, and how far one vision patch reaches in the patch
# list. Every one is read from the run's args.json and model receipt, or computed from
# them, when the page is built. A sentence that states one of them NAMES THE RUN it was read
# from ({run_name}) and never says "this run": the viewer adds a note to every popup that
# uses that phrase, saying it means the run DOC was written about.

GOAL_RING_MOMENTS = 200
HSELF_WINDOW_MOMENTS = {"l2": 10_000, "l3": 100_000}
"""The hierarchical self's two evidence windows in model steps: 1 / (1 - momentum) for the class's own
defaults m_l2 = 0.9999 and m_l3 = 0.99999 (mimo_bridge/hierarchical_self.py). A test holds them to
the class, so the page's seconds follow the code and this run's model step length."""
"""LemLite.goal_ring's maxlen (mimo_bridge/lem_model.py): fixed in the model's code, reached
by no flag. tests/test_build_explainer.py holds it to the built model."""

COLLAPSE_WINDOW_STEPS = 1000
"""The collapse watch's window in model steps: LemLite.token_ring keeps 200 samples and one
is stored every 5th model step. tests/test_build_explainer.py holds it to the code."""

REPLAY_SIZE_DEFAULT = 6000
INTENTION_EVERY_DEFAULT = 5
"""train_mimo.py's own defaults for --replay-size and --intention-every, used only for a run
whose args.json does not record the flag. tests/test_build_explainer.py holds both."""

WIDE_VIEW_PX, FOVEA_SIZE_DEFAULT, PATCH_PX = 256, 288, 16
"""The wide view is 256 px in every bridge run (train_mimo.py's eye_layout statement types
("wide", 256) and BridgeEnv's eye_size default is the same); the sharp centre is
--fovea-size (default 288, 0 = none); a patch is 16 px (vision_adapter.PATCH)."""

SWIN_WINDOW = 7
"""lem/master_config.yaml modalities.vision.encoder.window_size. No launch flag reaches it."""

STEREO_DISPARITY_PATCHES = (("wide", 2.5), ("fovea", 7.0))
"""How far apart, in patches, one object sits between the two eyes at reaching distance:
about 2.5 in the wide view and about 7 in the sharp centre (the measured figures in the
LITE_OVERRIDES comment of mimo_bridge/lem_model.py). Camera geometry, not model size."""


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def encoder_depths(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Each encoder's own depth and inside width, COUNTED from the receipt's per-tensor
    inventory. --encoder-layers L does not mean L layers everywhere (train_mimo.py
    size_overrides): tactile and motor get L attention layers, audio a stack of 2 + L
    convolutions, proprioception 1 + L hidden layers and then an output layer, and intermind is
    left alone. 'inventory' is False for a receipt that carries no tensor list, and every
    count is then None - never another run's number."""
    regt = _inventory_tensors(receipt, "encoder_registry")
    enc = receipt.get("encoders_built") or {}

    def containers(prefix: str) -> int:
        return len({k.split(".")[3] for k in regt if k.startswith(prefix)})

    def weights(prefix: str, ndim: int) -> List[List[int]]:
        hits = [(int(k.split(".")[3]), v.get("shape") or []) for k, v in regt.items()
                if k.startswith(prefix) and k.endswith(".weight")
                and k.split(".")[3].isdigit() and len(v.get("shape") or []) == ndim]
        return [s for _, s in sorted(hits)]

    have = bool(regt)
    mlp = weights("encoders.proprioception.mlp.", 2)
    return {
        "inventory": have,
        "vision_layers": containers("encoders.vision.blocks.") if have else None,
        "tactile_attn_layers": containers("encoders.tactile.layers.") if have else None,
        "motor_attn_layers": containers("encoders.motor.layers.") if have else None,
        "intermind_attn_layers": containers("encoders.intermind.layers.") if have else None,
        "audio_conv_layers": len(weights("encoders.audio.spectral_conv.", 3)) if have else None,
        "proprio_linear_layers": len(mlp) if have else None,
        # a hidden layer's width is the first number of its weight's shape; the last
        # linear layer is the output projection and is not a hidden layer
        "proprio_hidden_widths": [int(s[0]) for s in mlp[:-1]] if have else None,
        "width": {m: (int(e["d_internal"]) if isinstance(e, dict) and e.get("d_internal")
                      else None) for m, e in enc.items()},
    }


def vision_patch_layout(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The patch LIST the vision encoder reads, source by source, in BinocularPatcher's
    sequence: the wide view first, then the sharp centre; inside a source the two eyes' rows
    alternate (L0, R0, L1, R1, ...), so one picture row is 2 x columns list places long."""
    fovea = args.get("fovea_size")
    fovea = FOVEA_SIZE_DEFAULT if fovea is None else int(fovea)
    sources = [("wide", WIDE_VIEW_PX // PATCH_PX)]
    if fovea > 0:
        sources.append(("fovea", fovea // PATCH_PX))
    out, start = [], 0
    for name, cols in sources:
        n = 2 * cols * cols
        out.append({"name": name, "cols": cols, "start": start, "stop": start + n,
                    "row_places": 2 * cols})
        start += n
    return out


_SWIN_DEP_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def swin_dependency(n_patches: int, window: int, n_layers: int) -> np.ndarray:
    """dep[j, p] is True when the vision encoder's output at list position j depends on the
    input patch at list position p, after n_layers Swin blocks.

    Pure set arithmetic that repeats lem/model/encoders/swin_patch.py SwinBlock.forward
    step by step: pad the list with fresh zero rows to a multiple of the window; on every
    second block roll it by half a window (torch.roll - cyclic, and no mask hides the
    seam); let every position in a window take the union of what its window-mates depend
    on; roll back; cut the pad rows off. tests/test_build_explainer.py holds this to the
    real encoder by perturbing one patch and reading which outputs move.
    Measured 2026-09-19: 1,160 patches, 4 layers, 9 ms."""
    key = (int(n_patches), int(window), int(n_layers))
    if key in _SWIN_DEP_CACHE:
        return _SWIN_DEP_CACHE[key]
    n, ws, layers = key
    dep = np.eye(n, dtype=bool)
    npad = n + ((-n) % ws)
    shift = ws // 2
    for layer in range(layers):
        d = np.zeros((npad, n), dtype=bool)
        d[:n] = dep                                # a pad row carries no input
        if layer % 2:
            d = np.roll(d, -shift, axis=0)
        d = d.reshape(npad // ws, ws, n)
        d = np.broadcast_to(d.any(axis=1, keepdims=True), d.shape).reshape(npad, n)
        if layer % 2:
            d = np.roll(d, shift, axis=0)
        dep = d[:n].copy()
    _SWIN_DEP_CACHE[key] = dep
    return dep


def vision_reach(receipt: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """How far one vision patch reaches in the patch list at THIS run's layer count.

    'reach' is the largest number of list places between a patch and an output position
    that depends on it, counted round the ring the shifted blocks close the list into.
    'pairs' answers the question the page asks of each type of patch pair - do the two ever
    MEET, meaning some output position depends on both, so the encoder can compare them
    before its position-blind pooling step: 'met' of 'total' pairs that many places apart
    inside one source. 'wrap' is how many patches at the list's end share a shifted window
    with how many at its start."""
    shape = model_shape(receipt, args)
    layers = int(shape["vision_layers"])
    layout = vision_patch_layout(args)
    n, ws = layout[-1]["stop"], SWIN_WINDOW
    dep = swin_dependency(n, ws, layers)
    ring = n + ((-n) % ws)
    js, ps = np.nonzero(dep)
    gap = np.abs(js - ps)
    disparity = dict(STEREO_DISPARITY_PATCHES)
    pairs: Dict[str, Dict[str, int]] = {}
    for src in layout:
        lo, hi = src["start"], src["stop"]
        for kind, places in (("stereo", int(round(2 * disparity[src["name"]]))),
                             ("vertical", int(src["row_places"]))):
            both = (dep[:, lo:hi - places] & dep[:, lo + places:hi]).any(axis=0)
            pairs[f"{src['name']}_{kind}"] = {"places": places, "met": int(both.sum()),
                                              "total": int(both.size)}
    wrap = None
    if layers >= 2:
        seam = [(j + ws // 2) % ring for j in range(ring - ws, ring)]
        tail = [m for m in seam if n - ws <= m < n]
        head = [m for m in seam if m < ws]
        if tail and head:
            wrap = {"last": len(tail), "first": len(head)}
    return {"layers": layers, "layers_source": shape["source"].get("vision_layers"),
            "n_patches": n, "window": ws, "reach": int(np.minimum(gap, ring - gap).max()),
            "pairs": pairs, "wrap": wrap,
            "sources": [{k: s[k] for k in ("name", "cols", "row_places")} for s in layout]}


def _meets(p: Dict[str, int], what: str) -> str:
    if p["met"] == p["total"]:
        return (f"do meet: for every one of the {p['total']:,} {what} some position in the "
                f"encoder reads both patches")
    if p["met"] == 0:
        return (f"do not meet: for none of the {p['total']:,} {what} does any position in "
                f"the encoder read both patches")
    return f"meet on {p['met']:,} of the {p['total']:,} {what}"


def vision_reach_text(v: Dict[str, Any], run_name: str) -> str:
    """The reach paragraph of the vision encoder's popup, for one run's layer count."""
    src = {"receipt": f"the model receipt of {run_name}",
           "args": f"the launch flags of {run_name}",
           "default": "the launcher's default, because neither the receipt nor the launch "
                      f"flags of {run_name} record the depth"}.get(v["layers_source"], run_name)
    pr = v["pairs"]
    out = (f"Computed when this page was built, from the window arithmetic the encoder runs, "
           f"for the {_count(v['layers'], 'layer')} in {src}: one patch can reach another only "
           f"within {v['reach']} places of it in that list of {v['n_patches']:,} (0 would "
           f"mean no patch reaches any other). Alternating eyes put an object's two views "
           f"about 2 places apart for each patch of disparity. The wide view's pairs, about "
           f"2.5 patches of disparity and so {pr['wide_stereo']['places']} places apart, "
           f"{_meets(pr['wide_stereo'], 'such pairs')}")
    if "fovea_stereo" in pr:
        out += (f"; the sharp centre's, about 7 patches and "
                f"{pr['fovea_stereo']['places']} places apart, "
                f"{_meets(pr['fovea_stereo'], 'such pairs')}")
    vert = [pr[k] for k in ("wide_vertical", "fovea_vertical") if k in pr]
    where = (f"{pr['wide_vertical']['places']} places apart in the wide view"
             + (f" and {pr['fovea_vertical']['places']} in the centre"
                if "fovea_vertical" in pr else ""))
    out += f". Up-and-down neighbours are {where}, "
    if all(p["met"] == 0 for p in vert):
        out += ("so no patch ever meets the patch above it inside the layers; they meet only "
                f"at the final pooling step, one weighted sum over all {v['n_patches']:,}")
    elif all(p["met"] == p["total"] for p in vert):
        out += "and at this depth every patch does meet the patch above it inside the layers"
    else:
        out += ("and a patch meets the patch above it on "
                + " and ".join(f"{p['met']:,} of {p['total']:,} pairs" for p in vert)
                + " (wide view first)")
    out += "."
    if v["wrap"]:
        last, first = v["wrap"]["last"], v["wrap"]["first"]
        out += (f" The shifted layers roll the list round on itself and nothing masks the "
                f"seam, so its last {last} patch{'' if last == 1 else 'es'} share a window "
                f"with its first {first}: a pairing of the list's two ends that means nothing "
                f"in the picture.")
    return out


def planner_schedule(args: Dict[str, Any]) -> Dict[str, Any]:
    """Who can write the command the body executes, from the two launch flags that decide it.

    The trainer latches a new command when step % --motor-hold == 0 (train_mimo.py) and the
    planner runs when step % --intention-every == 0 (LemLite.step), on the SAME step
    counter. So of the latched commands gcd / every fall on a planner model step, and of the
    plans gcd / hold fall on a latch model step; exploration noise is added only to a command
    the planner did not write. tests/test_planner_schedule.py holds this arithmetic to a
    step-by-step count of the two conditions."""
    every_raw, hold_raw = args.get("intention_every"), args.get("motor_hold")
    every = max(1, int(every_raw or INTENTION_EVERY_DEFAULT))
    hold = max(1, int(hold_raw or 5))
    g = math.gcd(every, hold)
    return {"on": (args.get("intention") or "off") == "on",
            "execute": not bool(args.get("no_intention_execute")),
            "every": every, "hold": hold,
            "every_recorded": bool(every_raw), "hold_recorded": bool(hold_raw),
            # 1 latched command in this many falls on a planner model step
            "latches_per_planner_latch": every // g,
            # 1 plan in this many falls on a latch model step
            "plans_per_latched_plan": hold // g,
            "regime": ("equal" if every == hold else "every_divides_hold" if hold % every == 0
                       else "hold_divides_every" if every % hold == 0 else "neither")}


def _schedule_text(s: Dict[str, Any], run_name: str) -> Dict[str, str]:
    e, h = s["every"], s["hold"]
    a, b = s["latches_per_planner_latch"], s["plans_per_latched_plan"]
    flags = f"--intention-every {e} and --motor-hold {h}"
    nothing = ("the planner produced nothing (the goal ring still filling, an empty plan, or "
               "the realism check on imagined token size refusing a rollout)")
    # the planner's own sentences (node.intention, edge.intention.motor) in the canonical names
    # (the style guide "ONE WORD PER THING", 2026-10-02): a command, a model step, the size check
    no_plan = ("the planner produced nothing (the goal ring still filling, an empty plan, or the "
               "size check dropping a plan)")
    if not s["on"]:
        rule = (f"{run_name} was launched with --intention off, so the planner never ran: "
                f"every latched command is the motor controller's command, with [[babble]] "
                f"noise added whenever the noise is on at that model step.")
        return {"rule": rule,
                "controller": (f"{run_name} was launched with --intention off, so every "
                               f"executed command is this controller's proposal."),
                "line": (f"{run_name} was launched with --intention off, so nothing travels "
                         f"this line."),
                "babble_share": f"every latched command in {run_name}, whose planner is off",
                "off_note": (f"{run_name} WAS LAUNCHED WITH --intention off: the planner never "
                             f"ran there, and what follows describes a run with it on. ")}
    if not s["execute"]:
        rule = (f"{run_name} was launched with --no-intention-execute: the planner plans and "
                f"its plan is measured, and no plan is ever sent to the body, so every "
                f"latched command is the motor controller's command with [[babble]] noise "
                f"added whenever the noise is on at that model step.")
        return {"rule": rule,
                "controller": (f"{run_name} was launched with --no-intention-execute, so every "
                               f"executed command is this controller's proposal."),
                "line": (f"{run_name} was launched with --no-intention-execute, so the kept command "
                         f"is computed and never travels this line to the body."),
                "babble_share": (f"every latched command in {run_name}, which never sends a "
                                 f"plan to the body"),
                "off_note": ""}
    if a == 1:                       # every latch step is a planner step
        dropped = ("" if b == 1 else
                   f" The plans made on the other {b - 1} of every {b} planner steps fall "
                   f"between latch steps and are dropped.")
        rule = (f"{run_name} was launched with {flags}, so every latch step is also a "
                f"planner step: the plan's first command becomes the executed command on that "
                f"same model step, and a command the body receives carries [[babble]] noise only "
                f"when {no_plan}.{dropped}")
        controller = (f"With {flags} in {run_name}, every command boundary is also a planner "
                      f"model step, so this controller writes an executed command only when "
                      f"{nothing}; otherwise it supplies the planner's starting candidate.")
        line = (f"With {flags} in {run_name}, every latch step IS a planner step, so the kept "
                f"command takes effect on that same model step rather than waiting for a later "
                f"latch, is held for {h} model steps ([[motor-hold]]), and this line, not the "
                f"one from the core output, is what moves the body on every executed command "
                f"the planner produced.")
        share = (f"none of the latched commands in {run_name}, where {flags} make every latch "
                 f"model step a planner model step, unless {nothing}")
    else:
        lands = ("every plan lands on a latch step and its first command becomes the executed "
                 "command there"
                 if b == 1 else
                 f"1 plan in {b} lands on a latch step and its first command becomes the executed "
                 f"command there, and the other {b - 1} are dropped")
        rest = ("1 does not; on that one" if a == 2 else f"the other {a - 1} do not; on those")
        rule = (f"{run_name} was launched with {flags}: {lands}. Of every {a} latched "
                f"commands 1 falls on a planner step and {rest} "
                f"the body receives the motor controller's command, with [[babble]] noise "
                f"added whenever the noise is on at that model step.")
        controller = (f"With {flags} in {run_name}, this controller's proposal is the "
                      f"executed command on {a - 1} of every {a} command boundaries, the ones no "
                      f"planner model step falls on; on the remaining 1 it supplies the "
                      f"planner's starting candidate.")
        line = (f"With {flags} in {run_name}, {lands}; it is held for {h} model steps "
                f"([[motor-hold]]). This line moves the body on 1 of every {a} executed "
                f"commands; the line from the core output, through the motor controller, moves "
                f"it on the other {a - 1}.")
        share = (f"{a - 1} of every {a} latched commands in {run_name}, launched with {flags}")
    return {"rule": rule, "controller": controller, "line": line, "babble_share": share,
            "off_note": ""}


def exec_source_text(summary: Optional[Dict[str, Any]], stats_run: str) -> str:
    """The run's own tally of who wrote its executed commands, as one sentence. The schedule
    is arithmetic on two flags; this is the count the trainer took at each latch
    (exec_source in the stats log), so it is the measurement and the schedule is not."""
    if not summary or not summary.get("windows"):
        return (f"The stats log of {stats_run} carries no tally of who wrote its executed "
                f"commands, so how those shares came out in practice is unmeasured on this page.")
    names = {"planner": "the planner", "self_choice": "the self-model's candidate evaluation",
             "reflex_noise": "the motor controller with babble noise added",
             "reflex": "the motor controller with no noise added (the noise was 0 at that "
                       "model step: a rest window of the babble-bout schedule, or a launch with "
                       "babble off)"}
    parts = [f"{names.get(k, k)} wrote {round(100 * v['mean'])} of every 100 "
             f"({round(100 * v['min'])} to {round(100 * v['max'])} across the windows)"
             for k, v in summary["shares"].items()]
    return (f"Counted by the trainer at each latch and read from the stats log of "
            f"{stats_run} ({summary['windows']:,} report windows, steps "
            f"{summary['first_step']:,} to {summary['last_step']:,}): "
            + "; ".join(parts)
            + ". 0 of every 100 would mean that author never wrote an executed command, 100 "
              "that it wrote them all.")


def planner_fill(receipt: Dict[str, Any], args: Dict[str, Any], run_name: str) -> Dict[str, str]:
    """The {placeholders} of the planner's text (node.intention, its terms and its two input lines),
    filled for one run: the planner text rewritten on 2026-10-02 to state its input, computation
    and output (the style guide "MECHANISM, NEVER METAPHOR"). Each sentence reads the run's own launch flag
    and says so when the run's args.json does not record it (the trainer's default then applies).
    The constants it names - the goal ring's detrend and normaliser horizons, the harm line, the
    finger search's bound - are read from the code that owns them when the page is built."""
    def flag(name: str):
        return args.get(name) if name in args else None

    def said(name: str, value) -> str:
        return f"{run_name} was launched with --{name.replace('_', '-')} {value}"

    def unrecorded(name: str, default: str) -> str:
        return (f"the launch flags of {run_name} do not record --{name.replace('_', '-')}, "
                f"so it ran with the default, {default}")

    n_h = model_shape(receipt, args)["n_handles"]
    try:
        from lem.model import wellness_channels as WCM
        detrend = f"{int(WCM.RING_DETREND_STEPS):,}"
        normaliser = f"{int(WCM.NORMALISER_HORIZON_STEPS):,}"
        harm_line = f"{round(100.0 * float(WCM.INTEGRITY_GATE_LINE), 1):g}%"
    except Exception:                                    # a page is still built without the module
        detrend = normaliser = "a horizon lem/model/wellness_channels.py could not be read for"
        harm_line = "the line in lem/model/wellness_channels.py (INTEGRITY_GATE_LINE)"
    try:
        from lem.model.efference import EFFERENCE_SCALES
        finger_bound = f"{float(EFFERENCE_SCALES['rot_tenure_deg']):g} degrees"
    except Exception:
        finger_bound = "one measured tenure rotation (lem/model/efference.py EFFERENCE_SCALES)"
    htm, hg = flag("hand_target_mode"), flag("hand_goals")
    hand = htm == "wrist_relative_v1"
    synergy_search = flag("motor_synergy") == "on" and (flag("synergy_planner") or "on") == "on"
    # what a command is
    if hand:
        hand_mode = (f"Under --hand-target-mode wrist_relative_v1, as {run_name} was launched, the "
                     f"planner also sets a target angle for every finger joint, and the finger "
                     f"handles' rows of the command are computed from those angles.")
    elif htm:
        hand_mode = (f"{said('hand_target_mode', htm)}: every handle, the fingers included, is set "
                     f"by its own 9 numbers.")
    else:
        hand_mode = (f"{unrecorded('hand_target_mode', 'world')[:1].upper()}"
                     f"{unrecorded('hand_target_mode', 'world')[1:]}: every handle, the fingers "
                     f"included, is set by its own 9 numbers.")
    # the tokens behaviour reads: the top-down gate's blend, or the encoders' output
    if flag("top_down_gate") == "on":
        gate_now = (f", blended under --top-down-gate on (as {run_name} was launched) with up to half "
                    f"of the model's own prediction of this model step: token + g x (prediction - "
                    f"token), g from 0 to one half")
        gate_then = (f", blended under --top-down-gate on (as {run_name} was launched) with up to "
                     f"half of the model's own prediction of that model step")
    else:
        why = (said("top_down_gate", flag("top_down_gate")) if flag("top_down_gate")
               else unrecorded("top_down_gate", "off"))
        gate_now = gate_then = f" ({why}, so these are the encoders' output unchanged)"
    # the execution filter's blend
    s = float(flag("smooth_actions") or 0.0)
    if s > 0:
        new = int(round(100 * s))
        partner = ("the command the body was given over the interval that just ended (--smooth-source "
                   "executed)" if flag("smooth_source") == "executed" else
                   "the previous model step's blended command (--smooth-source proposal)")
        smooth = (f"position, orientation and strength are blended {new} parts this command with "
                  f"{100 - new} parts {partner}, {run_name}'s --smooth-actions {s:g}; the eyes' row "
                  f"and the radius are not blended")
    else:
        smooth = ("no blend: " + (said("smooth_actions", flag("smooth_actions")) if "smooth_actions" in args
                                  else unrecorded("smooth_actions", "0")))
    steps = ["each target position is held within 200 cm of the world origin on each axis and turned "
             "into centimetres (x 200), and its step from where that handle is now is clamped to 40 cm "
             "x the growth phase's reach limit on each axis",
             "strength is clamped to the growth phase's strength limit",
             "the eyes' row is written as the eye servo is sent it",
             smooth]
    if hand:
        steps.append("the finger joint angles are blended the same way and the finger handles' rows "
                     "are computed from them (--hand-target-mode wrist_relative_v1)")
    if flag("executed_fields_only") == "on":
        steps.append("on every row whose position the action decoder ignores the position is replaced "
                     "by the measured one, and every radius by 0 (--executed-fields-only on)")
    steps.append("the positions are divided by 200 again, the motor encoder's units, before the motor "
                 "encoder reads the command")
    filter_steps = "\n".join(f"{i}. {t}" for i, t in enumerate(steps, 1))
    # the score: how its strength part is summed, and what else a run's flags add to it
    if flag("planner_effort_per_body") == "on":
        effort_run = (f"{said('planner_effort_per_body', 'on')}: the strength part is averaged over "
                      f"the bodies, like the goal part.")
    else:
        why = (said("planner_effort_per_body", flag("planner_effort_per_body"))
               if flag("planner_effort_per_body") else unrecorded("planner_effort_per_body", "off"))
        effort_run = f"{why[:1].upper()}{why[1:]}: the strength part is summed over the bodies."
    extra = []
    if synergy_search:
        extra.append(f"Under --motor-synergy on with --synergy-planner on, as {run_name} was "
                     f"launched, the search changes the synergy's numbers - an intention vector, the "
                     f"strength settings and the eyes - and the synergy's penalty is added to the score.")
    if flag("prospective_wellness") == "add":
        extra.append(f"Under --prospective-wellness add, as {run_name} was launched, the score also "
                     f"adds its weight x (1 - the mean of the learned wellness copy's predicted bodily "
                     f"scores for the next model step).")
    if flag("physical_goals") == "memory":
        extra.append(f"Under --physical-goals memory, as {run_name} was launched, a goal entry with a "
                     f"recorded pose replaces the goal part by the distance between the efference body "
                     f"head's 250 ms forecast and that pose, after a first search that sets only the "
                     f"eyes toward the entry's view.")
    if hand:
        space = ("the synergy's numbers and every finger joint angle" if synergy_search else
                 "every command number and every finger joint angle")
    elif synergy_search:
        space = "the synergy's numbers"
    else:
        space = ("every command number - position and orientation directly, radius and strength "
                 "through the logits behind their 0-to-1 values")
    # after the search
    if hg == "memory":
        hand_goal = (f"Under --hand-goals memory, as {run_name} was launched, a goal entry whose "
                     f"recorded hand pose is valid gets a second search over the finger joint targets "
                     f"only ([[hand-goal|the finger search]]): 20 Adam steps of step size one tenth, "
                     f"each angle held within {finger_bound} of the first search's, scored by how far "
                     f"the finger poses a fitted hand-response model predicts 250 ms ahead lie from the "
                     f"hand pose recorded with the entry; of the 21 scored the lowest is kept.")
    elif hg:
        hand_goal = f"{said('hand_goals', hg)}: no second search; the finger targets are the first search's."
    else:
        why = unrecorded("hand_goals", "off")
        hand_goal = f"{why[:1].upper()}{why[1:]}: no second search; the finger targets are the first search's."
    ph = flag("plan_horizon") or "fixed"
    if hand:
        depth = (f"Every plan is one command long under --hand-target-mode wrist_relative_v1, as "
                 f"{run_name} was launched, whatever depth is licensed.")
    elif ph == "earned":
        depth = (f"Under --plan-horizon earned, as {run_name} was launched, the plan repeats the search "
                 f"from the predicted tokens of each kept command, up to 1 + the number of leading "
                 f"depths the imagination chain has licensed (a depth stays licensed while its running "
                 f"average of wins against predicting no change stays above 50 in 100), at most "
                 f"--imagination-steps; it stops early when a predicted distance to the goal tokens "
                 f"grows more than 5% past the smallest so far, when that distance falls under one "
                 f"hundredth, or when the learned wellness copy scores a predicted model step under 5% "
                 f"of its 0-to-1 range.")
    else:
        how = (f"as {run_name} was launched" if "plan_horizon" in args
               else "the default; the launch flags do not record it")
        depth = f"Under --plan-horizon {ph} ({how}) every plan is one command long."
    if not hand and flag("physical_goals") == "memory":
        depth += " Under --physical-goals memory a goal entry with a recorded pose gets a one-command plan."
    locality = (f" Under --goal-locality on, as {run_name} was launched, the goal is the highest-ranked "
                f"entry within the model's running measure of how far its tokens drift in 14 model "
                f"steps, kept until the tokens come within the one-model-step drift, leave that reach "
                f"or the entry is 200 model steps old." if flag("goal_locality") == "on" else "")
    eff = receipt.get("efference_heads")
    gaze_copy = ("; under --efference-heads on the gaze copy adds the eye command's part to the vision "
                 "prediction" if isinstance(eff, dict) and eff.get("gaze_copy_into_vision") else "")
    memory = (" Under --persistent-memory on the persistent memory's bias is added to the core's "
              "newest output tokens before the prediction heads read them."
              if (receipt.get("persistent_memory") or flag("persistent_memory")) == "on" else "")
    return {"{n_handles}": str(n_h), "{hand_mode_sentence}": hand_mode,
            "{gate_blend_now}": gate_now, "{gate_blend_then}": gate_then,
            "{execution_filter_steps}": filter_steps, "{smooth_sentence}": smooth,
            "{effort_run}": effort_run, "{score_extras_sentence}": " ".join(extra),
            "{search_space_words}": space, "{hand_goal_sentence}": hand_goal,
            "{plan_depth_sentence}": depth, "{goal_locality_sentence}": locality,
            "{harm_line_pct}": harm_line, "{detrend_steps}": detrend,
            "{normaliser_steps}": normaliser, "{finger_bound}": finger_bound,
            "{gaze_copy_clause}": gaze_copy, "{memory_bias_clause}": memory}


def hself_first_evidence_windows(k: int, rank_momentum: float = 0.9,
                                  alpha: float = 0.00135) -> Optional[int]:
    """How many windows a PERFECT slot needs before it can count, on a table of k slots - the
    same arithmetic as HierarchicalSelfDark.first_evidence_tick, without importing torch: the
    rank average starts at 0.5 and closes (1 - rank_momentum) of the gap to 1 each window, and
    the bar is 0.5 + z * sqrt(1/12) / sqrt(min(windows, 1 / (1 - rank_momentum))) with z the
    normal quantile of 1 - alpha / k. None when the bar sits above 1.0, the highest standing there
    is (tables of roughly 40,000 slots and more). A test holds this equal to the class."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - alpha / max(int(k), 1))
    for n in range(1, 100000):
        ceiling = 0.5 + 0.5 * (1.0 - rank_momentum ** n)
        neff = min(float(n), 1.0 / (1.0 - rank_momentum))
        if ceiling > 0.5 + z * math.sqrt(1.0 / 12.0) / math.sqrt(neff):
            return n
    return None


def hself_fill(args: Dict[str, Any], step_s: float, run_name: str) -> Dict[str, str]:
    """The hierarchical self popup's numbers for ONE run: its slot count, its two windows in
    minutes of body time, the windows a perfect slot needs on a table that size, and how many
    caregiver visits one window spans."""
    slots = int(args.get("hself_slots") or 64)
    climb = hself_first_evidence_windows(slots)
    l2, l3 = HSELF_WINDOW_MOMENTS["l2"], HSELF_WINDOW_MOMENTS["l3"]
    if climb is None:
        first = (f"With {slots:,} stored patterns judged at once the luck test asks for more than a "
                 f"perfect link can ever show, so nothing can ever count; the table would have to be "
                 f"smaller.")
        need = None
    else:
        need = 2 + climb
        first = (f"Even a perfect link needs {need} windows before it counts (2 for the luck test to warm "
                 f"up, {climb} for a perfect score to climb past the pass mark): {need * l2:,} model steps for "
                 f"Level 2 ({need * l2 * step_s / 3600.0:.1f} hours of body time) and {need * l3:,} for "
                 f"Level 3 ({need * l3 * step_s / 3600.0:.1f} hours).")
    anchor = ""
    try:
        cycle = float(args.get("caregiver_present_seconds") or 0) + float(args.get("caregiver_absent_seconds") or 0)
    except (TypeError, ValueError):
        cycle = 0.0
    if cycle > 0 and (args.get("caregiver_mode") or "off") != "off":
        anchor = (f"For scale: the caregiver comes and goes once every {cycle:g} s, so a Level 2 window "
                  f"spans about {l2 * step_s / cycle:.0f} visits and a Level 3 window about "
                  f"{l3 * step_s / cycle:.0f}.")
    off = str(args.get("hierarchical_self", "off")).lower() == "off"
    entity = str(args.get("entity_self", "off")).lower() != "off"
    notice = ("**Switched off in " + run_name + ".** That run was launched with "
              "--hierarchical-self off, "
              "so no table of whole model steps exists here and none of the numbers below are being "
              "produced" + (", and the named-things panel above it is what watches instead"
                            if entity else "") + ". What follows describes the part as it runs "
              "when it is switched on." + chr(10) * 2) if off else ""
    return {"{hself_off_notice}": notice,
            "{hself_slots}": f"{slots:,}",
            "{hself_l2_window_min}": f"{l2 * step_s / 60.0:.1f}",
            "{hself_l3_window_min}": f"{l3 * step_s / 60.0:.0f}",
            "{hself_first_evidence_text}": first,
            "{hself_l2_first_moments}": "" if need is None else f"{need * l2:,}",
            "{hself_l3_first_moments}": "" if need is None else f"{need * l3:,}",
            "{hself_window_anchor}": anchor,
            # the wall-clock sentence needs the run's measured speed; build_explainer fills it
            # (hself_wall_fill) and a page built without stats leaves it empty
            "{hself_first_evidence_wall}": "",
            "{hself_l2_fast_s}": f"{l2 * step_s / 10.0:.0f}",
            # what this part is doing in the run on the page; build_explainer fills it from
            # the newest snapshot (hself_now_fill)
            "{hself_now}": ("It is switched off in " + run_name + ", so there is nothing to report."
                            if off else f"{run_name}'s snapshots do not record it.")}


def wellness_fill(args: Dict[str, Any], run_name: str) -> Dict[str, str]:
    """The two wellness {placeholders} that a launch flag decides.

    {self_touch_weight}: how much a fully self-touching model step may add to wellness
    (--self-touch-wellness). {wellness_coherence_choice}: which reading of the body map the
    coherence term of wellness carries (--wellness-coherence). A flag the launch did not
    record prints a sentence saying so, never another run's setting.
    """
    w = args.get("self_touch_wellness")
    if not isinstance(w, (int, float)) or isinstance(w, bool):
        touch = (f"The launch flags of {run_name} do not record --self-touch-wellness, so how "
                 f"much a fully self-touching model step may add is not stated on this page.")
    elif float(w) == 0.0:
        touch = (f"{run_name} was launched with --self-touch-wellness 0, so this term reads "
                 f"exactly zero on every step of it and the wellness sum is what it was, term "
                 f"for term, before the term existed.")
    else:
        touch = (f"{run_name} was launched with --self-touch-wellness {float(w):g}, so a "
                 f"model step in which four or more of the body's own parts newly arrive on its "
                 f"skin adds {float(w):g} to a sum whose other five terms reach 5.0 between "
                 f"them - and it adds that only while the gate is open.")
    c = args.get("wellness_coherence")
    if c == "extent":
        coh = (f"{run_name} was launched with --wellness-coherence extent, so extent is the "
               f"reading its wellness carries.")
    elif c == "certainty":
        coh = (f"{run_name} was launched with --wellness-coherence certainty, so its wellness "
               f"carries the OTHER reading described below - the one that rises as the "
               f"handles' co-movement numbers fall.")
    else:
        coh = (f"The launch flags of {run_name} do not record --wellness-coherence, so which "
               f"of the two readings it carried is not stated on this page.")
    return {"{self_touch_weight}": touch, "{wellness_coherence_choice}": coh}


# train_mimo.py's parser defaults for the limb and eye servo stiffness (--decoder-kp, --gaze-kp),
# named on the page only for a launch that did not record them; a test holds both equal to
# train_mimo.py's own add_argument lines. The eye's arrival time was measured at one pair of eye
# gains, noted beside the eye servo in env.py: "kp 80 / kd 3.0 gives 110 deg/s, 130 ms, no
# overshoot and no ringing".
DECODER_KP_DEFAULT = 8.0
GAZE_KP_DEFAULT = 80.0
EYE_SWEEP_KP, EYE_SWEEP_KD, EYE_SWEEP_MS = 80.0, 3.0, 130


def _launch_gain(args: Dict[str, Any], name: str) -> Optional[float]:
    v = args.get(name)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
        return None
    return float(v)


def servo_fill(args: Dict[str, Any], run_name: str) -> Dict[str, str]:
    """The eye and limb servo stiffness the glossary's handle entry states, read from the run's own
    launch settings (--decoder-kp, --gaze-kp, --gaze-kd).

    Until 2026-09-23 the entry typed "a separate eye servo 40 times stiffer than the limb servo
    (stiffness 80 against 2)" and "a limb needs about 250 ms". 2 is ActionDecoder's class default
    and H83's setting; all 50 H84 segments ran --decoder-kp 6 with --gaze-kp 80, 13.3 times, and no
    file measures a limb's arrival time at stiffness 6. A setting the launch did not record prints a
    sentence saying so with train_mimo's default, never another run's number."""
    kp = _launch_gain(args, "decoder_kp")
    gkp = _launch_gain(args, "gaze_kp")
    gkd = _launch_gain(args, "gaze_kd")
    defaults = {"--decoder-kp": f"{DECODER_KP_DEFAULT:g} for the limbs",
                "--gaze-kp": f"{GAZE_KP_DEFAULT:g} for the eye"}
    missing = [flag for flag, v in (("--decoder-kp", kp), ("--gaze-kp", gkp)) if v is None]
    if not missing:
        ratio = f"{gkp / kp:.1f}"
        clause = (f"a separate eye servo {ratio} times stiffer than the limb servo in {run_name} "
                  f"(stiffness {gkp:g} against {kp:g}; a limb's is further scaled by the command's "
                  f"strength, 0-1).")
    else:
        ratio = (f"not stated (the launch of {run_name} did not record "
                 f"{' or '.join(missing)})")
        recorded = (f"; it recorded --gaze-kp {gkp:g}" if gkp else
                    f"; it recorded --decoder-kp {kp:g}" if kp else "")
        clause = (f"a separate eye servo with a stiffness of its own. The launch of {run_name} did "
                  f"not record {' or '.join(missing)}, so how much stiffer the eye servo is than "
                  f"the limb servo there is not stated on this page (train_mimo's default "
                  f"{'is' if len(missing) == 1 else 'values are'} "
                  f"{' and '.join(defaults[m] for m in missing)}{recorded}).")
    sweep = (f"a sweep measured the eye reaching a new target in {EYE_SWEEP_MS} ms with no "
             f"overshoot at eye stiffness {EYE_SWEEP_KP:g} and damping {EYE_SWEEP_KD:g} (the note "
             f"beside the eye servo in env.py)")
    if gkp == EYE_SWEEP_KP and gkd == EYE_SWEEP_KD:
        eye = (f"At {run_name}'s eye stiffness {gkp:g} and damping {gkd:g}, a sweep measured the "
               f"eye reaching a new target in {EYE_SWEEP_MS} ms with no overshoot (the note "
               f"beside the eye servo in env.py).")
    elif gkp and gkd:
        eye = (f"{sweep[:1].upper()}{sweep[1:]}; at {run_name}'s eye stiffness {gkp:g} and damping "
               f"{gkd:g} the eye's arrival time has not been measured.")
    else:
        eye_missing = [f for f, v in (("--gaze-kp", gkp), ("--gaze-kd", gkd)) if v is None]
        eye = (f"{sweep[:1].upper()}{sweep[1:]}; the launch of {run_name} did not record "
               f"{' or '.join(eye_missing)}, so whether that holds there is not stated.")
    tuned = ("when the limb servo was tuned at stiffness 2, a 28.6 degree shoulder turn was met "
             "within 1.4 degrees in 0.6 s (the note at the top of action_decoder.py)")
    if kp == 2.0:
        limb = f"{tuned[:1].upper()}{tuned[1:]}."
    elif kp:
        limb = (f"How long a limb takes to reach its target at stiffness {kp:g} has not been "
                f"measured; {tuned}.")
    else:
        limb = (f"How long a limb takes to reach its target in {run_name} is not stated, because "
                f"its launch did not record --decoder-kp; {tuned}.")
    return {"{limb_kp}": f"{kp:g}" if kp else
            f"not recorded (train_mimo's default is {DECODER_KP_DEFAULT:g})",
            "{gaze_kp}": f"{gkp:g}" if gkp else
            f"not recorded (train_mimo's default is {GAZE_KP_DEFAULT:g})",
            "{eye_limb_kp_ratio}": ratio,
            "{eye_servo_clause}": clause,
            "{servo_timing}": f"{eye} {limb}"}


# ================================================= the gain-of-function build (2026-09-22)
# The page's words for each sense, the approximate evaluator's excluded senses (held equal to
# lem/model/wellness.py APPROXIMATE_EXCLUDED by a test, so this module never imports torch), and
# the default sense list a receipt that names none is assumed to have (lem_model.py builds these
# six unless --vestibular-sense on puts vestibular in the sixth slot).
# THE MODEL REGISTRY'S OWN NAMES (lem/config active_modalities), one word per sense on this page and on
# the development dashboard (the operator's rule: never rename a part); each is defined where the
# page first names it (the sense boxes' popups, the tour's senses step)
SENSE_WORDS = {"vision": "vision", "tactile": "tactile", "audio": "audio",
               "proprioception": "proprioception", "motor": "motor", "intermind": "intermind",
               "vestibular": "vestibular"}
# ONE plain line per sense name, shared by the tour's senses step, the development dashboard's
# Model health card and (held equal by a test) its glossary (tools/development_dashboard_terms.js).
# Each is what the sense's own adapter hands its encoder (the node.sense.* entries give the detail).
SENSE_DEFINITIONS = {
    "vision": "what the two eyes see: a wide view and a sharp centre for each eye, as pixels",
    "tactile": "the skin's sense of touch: up to 32 skin parts at once, each with its pressure, "
               "rub, slip, contact area, vibration and push",
    "audio": "what the two ears hear: how loud each of 256 pitch bands is at each ear, plus two "
             "numbers for the sound's direction",
    "proprioception": "where each tracked body part is, how it is turned and how fast it moves",
    "motor": "the command the body was last given, beside the pose each part reached, read back as "
             "an input",
    "vestibular": "the body's load and balance in 29 numbers: which way gravity pulls on the head "
                  "and how the head turns, the neck, which way is up for the trunk, what carries "
                  "the weight, effort, joints at their limits and fatigue",
    "intermind": "a reserved input for other minds' summaries, placeholder noise for now",
}
APPROXIMATE_EXCLUDED = ("intermind", "vestibular")
DEFAULT_SENSES = ("vision", "tactile", "audio", "proprioception", "motor", "intermind")
_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve")


def _word(n: int) -> str:
    return _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else f"{n:,}"


def _and_list(items: List[str]) -> str:
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def run_senses(receipt: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    """The senses this run carries, in slot sequence: the receipt's own list, else the default six
    with the sixth slot named by --vestibular-sense."""
    mods = list(receipt.get("active_modalities") or [])
    if mods:
        return mods
    out = list(DEFAULT_SENSES)
    if str(args.get("vestibular_sense") or "off") == "on":
        out[out.index("intermind")] = "vestibular"
    return out


# The two goal-ranking channels that read the efference heads, with their table weights (held equal
# to lem/model/wellness_channels.py DEFAULT_WEIGHTS by a test, so this module never imports torch).
EFFERENCE_RANK_CHANNELS = {"controllability": 1.0, "learning_progress": 0.75}


def efference_routes(receipt: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """Where the efference heads' readings reach the planner in this run: ONE rule for the diagram's
    lines out of the efference box (viewer.js reads META.efference_routes), the tour step that
    animates them (walkthrough.py) and the popups' run sentences (gof_fill).

      goal_channels  the channels that rank goals with a reading of the heads in them. Under
                     --wellness-rank normalized: controllability with --controllability on (the body
                     head's command-long expectation against what the body did, lem/model/
                     wellness_channels.py ControllabilityTracker, fed LemLite.last_efference_tenure),
                     and learning progress with --learning-progress on (each head's next-model-step
                     error against doing nothing, the keys efference_body and efference_gaze), each
                     only at a weight other than 0. Under --wellness-rank raw both are measured and
                     reported only: LemLite._wellness_channel_vector is called by the normalised
                     ranking alone.
      action_search  --physical-goals memory with --intention on: the planner's action search scores
                     candidate commands by the body head (lem/model/physical_goal.py PhysicalGoalCost,
                     built in LemLite._plan_chain from the physical goal LemLite.step stores only
                     under memory).

    The receipt's wellness_channels stamp (what the model was built with) wins; a run without it
    is read from its launch flags."""
    out: Dict[str, Any] = {"goal_channels": [], "action_search": False}
    if not receipt.get("efference_heads"):
        return out
    wc = receipt.get("wellness_channels") if isinstance(receipt.get("wellness_channels"), dict) else {}
    if str(wc.get("rank") or args.get("wellness_rank") or "raw") == "normalized":
        ranked = wc.get("channels") if isinstance(wc.get("channels"), list) and wc.get("channels") else None
        weights = dict(EFFERENCE_RANK_CHANNELS)
        if isinstance(wc.get("weights"), dict):
            weights.update(wc["weights"])
        else:
            for part in str(args.get("wellness_weights") or "").split(","):
                name, eq, val = part.partition("=")
                if eq and name.strip() in weights:
                    try:
                        weights[name.strip()] = float(val)
                    except ValueError:
                        pass
        for ch in EFFERENCE_RANK_CHANNELS:
            on = (ch in ranked) if ranked is not None else str(args.get(ch) or "off") == "on"
            if on and float(weights.get(ch) or 0.0) != 0.0:
                out["goal_channels"].append(ch)
    out["action_search"] = (str(args.get("physical_goals") or "off") == "memory"
                            and str(args.get("intention") or "off") == "on")
    return out


def gof_fill(receipt: Dict[str, Any], args: Dict[str, Any], run_name: str,
             source_mask: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The {placeholders} the build's features and the sense counts put into DOC, for one run.

    The sense counts and names come from the run's receipt (vestibular can take the
    sixth slot, where the intermind placeholder was); which senses arrive real comes from the
    newest snapshot's source labels when the page has one, else from the launch flags. Every
    feature sentence reads the run's own launch flag and says 'not recorded' for a flag the run's
    args.json does not carry (a run launched before the flag existed), never another run's."""
    mods = run_senses(receipt, args)
    heads = [m for m in (receipt.get("prediction_head_modalities") or mods) if m != "motor"]
    n, nh = len(mods), len(heads)
    sm = dict(source_mask or {})

    def is_real(m: str) -> bool:
        if m in sm:
            return str(sm[m]) == "real"
        if m == "intermind":
            return False                     # a placeholder generator feeds it in every world
        if m == "audio":
            return str(args.get("audio") or "real") == "real"
        if m == "vision":
            return not bool(args.get("no_vision"))
        return True

    graded = [m for m in heads if is_real(m)]
    unreal = [m for m in heads if not is_real(m)]
    words = lambda ms: [SENSE_WORDS.get(m, m) for m in ms]
    graded_text = (f"{_word(len(graded))} of them in {run_name}: {_and_list(words(graded))}"
                   if graded else f"none of them in {run_name}")
    graded_sentence = (
        f"in {run_name} causal_pe is the mean over {_word(len(graded))} of the {_word(nh)}"
        + (f" ({_and_list(words(unreal))} "
           + ("is a synthetic placeholder" if len(unreal) == 1 else "are synthetic placeholders")
           + " in this world)." if unreal else ", every one of them measured live."))
    judge = [m for m in mods if m not in APPROXIMATE_EXCLUDED]
    left = [m for m in mods if m in APPROXIMATE_EXCLUDED]
    judge_words = [("MOTOR" if m == "motor" else SENSE_WORDS.get(m, m)) for m in judge]
    if "vestibular" in left:
        judge_left = ("Vestibular is left out, as the intermind placeholder was in its "
                      "slot, so the learned copy's input stays the width its weights were "
                      "trained at.")
    elif left:
        judge_left = f"{_and_list(words(left))[:1].upper()}{_and_list(words(left))[1:]} " + \
                     ("is the one left out." if len(left) == 1 else "are left out.")
    else:
        judge_left = "No sense is left out."
    if "intermind" in mods:
        hself_warn = (f"- **The placeholder sense.** One of the {_word(n)} senses, intermind, "
                      f"is placeholder noise. It is one part in {n} of every pattern and "
                      f"changes as the model trains.")
        backward = ("the intermind placeholder's encoder included, and no lesson grades that "
                    "channel at all (measured on a live training step)")
    elif "vestibular" in mods:
        hself_warn = ("- **The sixth slot changed meaning.** With --vestibular-sense on, the slot "
                      "the intermind placeholder's noise held carries the vestibular input, so a "
                      "table filled before the switch stored noise in that part of every "
                      "pattern.")
        backward = ("the vestibular encoder included, which its own next-model-step lesson grades "
                    "as well")
    else:
        hself_warn = (f"- **Every sense in every pattern.** A pattern is the {_word(n)} senses' "
                      f"tokens laid end to end, so when any encoder changes, stored patterns stop "
                      f"matching.")
        backward = "every sense's encoder included"

    def flag(name: str):
        return args.get(name) if name in args else None

    def said(name: str, value) -> str:
        return f"{run_name} was launched with --{name.replace('_', '-')} {value}"

    unrec = lambda name, since: (f"The launch flags of {run_name} do not record "
                                 f"--{name.replace('_', '-')} (the flag arrived on {since}), so "
                                 f"{run_name} ran without it.")
    cp = flag("contact_profile") or receipt.get("contact_profile")
    contact_profile = (f"{said('contact_profile', cp)}." if cp else
                       f"The launch flags of {run_name} do not record --contact-profile, so "
                       f"which path it ran is not stated on this page.")
    th = flag("toy_hand_off")
    if th == "follow":
        toy = (f"{said('toy_hand_off', 'follow')}. Until 2026-09-22 a resumed segment kept its "
               f"saved world's settings, so the flag and the world could disagree; the caregiver "
               f"panel's 'the toy follows a grip' row is the world's own report.")
    elif th:
        toy = f"{said('toy_hand_off', th)}: the toy moves only in the adult's hand."
    else:
        toy = unrec("toy_hand_off", "2026-09-21")
    cc = flag("causal_chain")
    reached = (f"{said('causal_chain', 'on')}, so the trainer's causal-chain recorder stamps "
               f"each plan's reached_body at the latch: true when the plan's first command became "
               f"the body's command at that model step, false when it was computed and never sent."
               if cc == "on" else
               (f"{said('causal_chain', cc)}, so a plan record's reached_body stays empty; its "
                f"intention_execute_flag is only the launch setting." if cc else
                f"{run_name} was launched before --causal-chain existed (2026-09-22), so a plan "
                f"record's reached_body stays empty; its intention_execute_flag is only the "
                f"launch setting."))
    causal_run = (f"{said('causal_chain', 'on')}: its snapshots carry the planning event and "
                  f"its causal_chain.jsonl the resolved commands." if cc == "on" else
                  (f"{said('causal_chain', cc)}: nothing is recorded." if cc else
                   unrec("causal_chain", "2026-09-22")))

    def onoff(name: str, since: str, on_text: str, off_text: str) -> str:
        v = flag(name)
        if v is None:
            return unrec(name, since)
        return f"{said(name, v)}: " + (on_text if str(v) not in ("off", "0", "0.0", "False",
                                                                  "proposal", "raw", "multiply",
                                                                  "count") else off_text)
    wr = flag("wellness_rank")
    wellness_rank = (f"{said('wellness_rank', wr)}." if wr else
                     f"{run_name} ranks goals by the raw sum: its launch flags do not record "
                     f"--wellness-rank (the flag arrived on 2026-09-22).")
    ccw = flag("contact_control_wellness")
    contact_control = (unrec("contact_control_wellness", "2026-09-22") if ccw is None else
                       (f"{said('contact_control_wellness', ccw)}: the channel is scored and "
                        f"reported, and enters the goal ranking at that weight while "
                        f"--wellness-rank is normalized." if float(ccw) > 0 else
                        f"{said('contact_control_wellness', ccw)}: the channel is off."))
    fa = flag("finger_armature")
    finger = (f"{said('finger_armature', fa)}." if fa is not None else
              f"The launch flags of {run_name} do not record --finger-armature, so its fingers "
              f"carried the flat hinge armature that held them nearly still.")
    synergy = onoff("motor_synergy", "2026-09-22",
                    "the planner descends on z and the stats record carries the synergy block.",
                    "the planner moves every handle's command directly, as before.")
    if flag("motor_synergy") == "on" and flag("synergy_planner") == "off":
        synergy = (f"{said('motor_synergy', 'on')} and --synergy-planner off: the map learns "
                   "from achieved movements while the planner still moves each handle directly.")
    if flag("motor_synergy") is None and "motor_synergy" not in receipt:
        synergy = (f"Being built in the gain-of-function build (DESIGN.md D11 and D11a) and not "
                   f"yet in a running segment: {run_name}'s launch flags do not record "
                   f"--motor-synergy.")
    # WHERE THE PLANNER'S ACTION SEARCH READS THE BODY HEAD, AND WHERE THE HEADS REACH THE CHOICE OF
    # COMMAND WITHOUT IT: one rule, efference_routes, the rule the diagram's lines are drawn by.
    # lem_model.LemLite._plan_chain builds lem/model/physical_goal.py PhysicalGoalCost only from a
    # decision's physical_goal, which step() stores only under --physical-goals memory (commit
    # ecfc9060f, 2026-09-24) and only inside the planning block, which exists only with --intention on
    # (so action_search needs both; the review of 2026-09-25, finding 3). Under --wellness-rank
    # normalized the heads' readings are among the channels that rank the goal (goal_channels), so a
    # change in a head's accuracy can change the goal and with it the command, with no action search at
    # all (the review's finding 1: learning progress has no evidence gate).
    pg = flag("physical_goals")
    er = efference_routes(receipt, args)
    ch_words = {"controllability": "controllability", "learning_progress": "learning progress"}
    chans = [ch_words[c] for c in er["goal_channels"]]
    via_goal = (lambda word: (
        f" The heads {word} reach the choice of command through the goal: {_and_list(chans)} "
        f"{'rank' if len(chans) > 1 else 'ranks'} the remembered model step the planner aims at (the dashed "
        f"line into the wellness box), so a change in a head's accuracy can change the goal, and with it "
        f"the command." if chans else ""))
    if er["action_search"]:
        physical_goals = (
            f"{said('physical_goals', pg)}: when the remembered model step the planner aims at has a "
            f"measured pose stored beside it, the planner first searches for the eye command whose "
            f"predicted view comes closest to the remembered view, then scores each candidate command "
            f"by how far the body head's command-long prediction lands from the stored pose, and plans "
            f"one command ahead (lem/model/physical_goal.py PhysicalGoalCost)." + via_goal("also"))
    elif pg == "memory":
        physical_goals = (
            f"{said('physical_goals', pg)}, but its planner is off (--intention off): the physical goal "
            f"is stored only when the planner plans, so no action search runs and none calls the body "
            f"head." + via_goal("still"))
    elif pg is not None:
        physical_goals = (f"{said('physical_goals', pg)}: the planner's action search never calls the "
                          f"body head." + via_goal("still"))
    else:
        physical_goals = (
            f"The launch flags of {run_name} do not record --physical-goals (the flag arrived on "
            f"2026-09-24), so the planner's action search in {run_name} never called the body head."
            + via_goal("still"))
    # which goal-ranking channels read the heads in this run (efference_routes, the dashed line from
    # the efference box into the wellness box)
    wr_flag = flag("wellness_rank")
    if not receipt.get("efference_heads"):
        efference_channels = f"No efference heads were built in {run_name}, so no such line is drawn."
    elif er["goal_channels"]:
        efference_channels = (
            f"In {run_name} the goal ranking reads {_and_list([ch_words[c] for c in er['goal_channels']])} "
            f"(--wellness-rank normalized), so the dashed line is drawn.")
    elif str((receipt.get("wellness_channels") or {}).get("rank") or wr_flag or "raw") != "normalized":
        efference_channels = (
            f"{run_name} ranks goals by the raw sum ("
            + (f"--wellness-rank {wr_flag}" if wr_flag else "its launch does not record --wellness-rank")
            + "), so these channels, where they are on, are measured and reported only, and no line "
              "is drawn.")
    else:
        efference_channels = (
            f"{run_name} ranks goals normalized, but neither --controllability nor --learning-progress "
            f"is on at a weight other than 0, so no line is drawn.")
    # THE GAZE COPY'S ACTIVATION REPAIR (review of 2026-09-25, finding 11). Under
    # --vision-activation-calibration, EfferenceHeads.vision_delta runs through
    # lem/model/activation_compensation.py VisualActivationCompensation: the output it had at the repair
    # comes from a frozen copy of the weights, and the current weights add only their difference from
    # that copy, each through a GELU shifted up and scaled per unit. The model receipt records it as
    # visual_activation_compensation (LemLite._restore_visual_activation_calibration).
    if not receipt.get("efference_heads"):
        gaze_copy_repair = f"No efference heads were built in {run_name}."
    elif receipt.get("visual_activation_compensation"):
        gaze_copy_repair = (
            f"In {run_name} the gaze copy runs with the activation repair measured for it "
            f"(--vision-activation-calibration, lem/model/activation_compensation.py): a frozen copy "
            f"of its weights from the repair keeps producing the output it had then, and its current "
            f"weights add only their difference from that copy, each passed through a GELU shifted up "
            f"per unit, so units that sat in the flat end of the curve respond again, and scaled per "
            f"unit to the sensitivity of the units that already responded.")
    elif flag("vision_activation_calibration"):
        gaze_copy_repair = (f"{run_name} was launched with --vision-activation-calibration, but its model "
                            f"receipt records no repair, so this page cannot say the gaze copy runs with it.")
    else:
        gaze_copy_repair = (f"{run_name} runs the gaze copy's plain path: its launch names no "
                            f"--vision-activation-calibration.")
    # HOW LONG A PLAN IS (review of 2026-09-25, finding 10): lem_model.LemLite._plan_chain sets its
    # length to 1 under any --hand-target-mode other than world, and for a goal with a stored pose
    htm, hg = flag("hand_target_mode"), flag("hand_goals")
    if flag("intention") != "on":
        plan_length = f"{run_name}'s planner is off (--intention {flag('intention') or 'off'})."
    elif htm and htm != "world":
        plan_length = (f"{said('hand_target_mode', htm)}, so every plan it makes is one command long, "
                       f"whatever depth is licensed.")
    elif pg == "memory":
        plan_length = (f"{said('physical_goals', 'memory')}: a plan whose goal carries a stored pose is "
                       f"one command long; any other plan runs to its licensed depth.")
    else:
        plan_length = f"{run_name}'s plans run to their licensed depth."
    if hg == "memory":
        hand_goals = (f"{said('hand_goals', 'memory')}, so its plans set the finger targets by the "
                      f"second search, toward the hand pose recorded with the goal entry.")
    elif hg is not None:
        hand_goals = f"{said('hand_goals', hg)}: the fingers' goals come from the search alone."
    else:
        hand_goals = (f"The launch flags of {run_name} do not record --hand-goals, so the fingers' "
                      f"goals come from the search alone.")
    nfn = flag("next_frame_norm")
    next_frame_norm = (f"{said('next_frame_norm', nfn)}." if nfn else
                       f"Designed in DESIGN.md D12 and not yet built into a running segment: the "
                       f"launch flags of {run_name} do not record --next-frame-norm.")
    return {
        "{n_senses}": str(n), "{n_senses_word}": _word(n),
        "{n_heads}": str(nh), "{n_heads_word}": _word(nh),
        "{n_heads_word_cap}": _word(nh)[:1].upper() + _word(nh)[1:],
        "{head_list}": _and_list(words(heads)),
        "{non_motor_list}": _and_list(words([m for m in mods if m != "motor"])),
        "{graded_text}": graded_text, "{graded_sentence}": graded_sentence,
        "{judge_list}": _and_list(judge_words), "{judge_n}": str(len(judge)),
        "{judge_n_word_cap}": _word(len(judge))[:1].upper() + _word(len(judge))[1:],
        "{judge_left_out}": judge_left,
        "{hself_slot5_warning}": hself_warn, "{slot5_backward_clause}": backward,
        "{contact_profile_run}": contact_profile, "{toy_hand_off_run}": toy,
        "{reached_body_sentence}": reached, "{causal_chain_run}": causal_run,
        "{wellness_rank_run}": wellness_rank,
        "{controllability_run}": onoff("controllability", "2026-09-22",
                                       "the channel is measured every command.",
                                       "the channel is not measured."),
        "{learning_progress_run}": onoff("learning_progress", "2026-09-22",
                                         "the channel is measured every model step.",
                                         "the channel is not measured."),
        "{curiosity_mode_run}": onoff("curiosity_mode", "2026-09-22",
                                      "the drive is added.", "the drive multiplies."),
        "{consequence_run}": onoff("consequence_self_contact", "2026-09-22",
                                   "the hand-press kind counts the world only.",
                                   "the hand-press kind still counts self-contact."),
        "{contact_control_run}": contact_control,
        "{contact_head_run}": onoff("contact_head", "2026-09-22", "the head is built.",
                                    "the head is not built."),
        "{efference_run}": onoff("efference_heads", "2026-09-22", "the heads are built.",
                                 "the heads are not built."),
        "{efference_architecture}": (
            "This version reuses the old unused gaze head's attention and layers."
            if gaze_head_reused(receipt) else
            "The gaze head has its own layers. Version 2 centres the core's state before the "
            "heads read it, hands the body head the asked change relative to the current pose, "
            "and adds the gaze copy's output to the vision prediction. The gaze head and the gaze "
            "copy also read the head's and the eyes' orientation, and under --vestibular-sense on "
            "the gaze head also reads the head's accelerometer and gyro."
            if receipt.get('efference_heads') else
            f"No efference heads were built in {run_name}."),
        "{physical_goals_run}": physical_goals,
        "{efference_channels_run}": efference_channels,
        "{gaze_copy_repair_run}": gaze_copy_repair,
        "{plan_length_run}": plan_length, "{hand_goals_run}": hand_goals,
        "{finger_armature_run}": finger,
        "{smooth_source_run}": onoff("smooth_source", "2026-09-22",
                                     "the smoother blends with the executed command.",
                                     "the smoother blends with the proposals' running average."),
        "{synergy_run}": synergy, "{next_frame_norm_run}": next_frame_norm,
        "{clip_sets_text}": clip_sets_text(receipt),
        **idle_weights_text(receipt, run_name),
        **integrity_gate_fill(),
    }


def run_fill(receipt: Dict[str, Any], args: Dict[str, Any], run_name: str, *,
             stats_run: Optional[str] = None,
             exec_source: Optional[Dict[str, Any]] = None,
             source_mask: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The {placeholders} of the H84 items' second pass, filled for one run. Every key is
    always returned - a fact the run did not record prints a sentence saying so, never
    another run's number - so no placeholder can reach a built page unfilled."""
    c = moment_clock(receipt, args)
    step_s = c["model_step_s"]
    sched = planner_schedule(args)
    st = _schedule_text(sched, run_name)
    hold = sched["hold"]

    # the imagination chain's depth: the launch flag, clamped the way LemLite clamps it
    # (the context window has to hold depth + 1 model steps)
    imag_on = (args.get("imagination") or "config") != "off"
    depth = args.get("imagination_steps")
    depth = int(depth) if isinstance(depth, int) and depth > 0 else None
    ctx = model_shape(receipt, args)["context_length"]
    if depth is not None and ctx and ctx < depth + 1:
        depth = max(1, ctx - 1)
    unrecorded = (f"the launch of {run_name} did not record --imagination-steps, so the depth "
                  f"is the config file's, which this page does not read")
    if not imag_on:
        forward = (f"The imagination chain is switched off in {run_name} (--imagination off) "
                   f"and adds no transition")
        back = f"switched off in {run_name}, so no model step is taken"
        cap = f"and the chain is switched off in {run_name}"
    elif depth is None:
        forward = f"The imagination chain adds one more transition per depth every model step ({unrecorded})"
        back = f"as many steps back as the chain is deep; {unrecorded}"
        cap = f"and {unrecorded}"
    else:
        forward = f"The imagination chain adds {_count(depth, 'more transition')} every model step"
        back = f"{_count(depth, 'step')} back in {run_name}"
        cap = f"{depth} in {run_name}"

    size = args.get("replay_size")
    replay_size = (f"{int(size):,}" if size else
                   f"{REPLAY_SIZE_DEFAULT:,} (the launcher's default; the launch flags of "
                   f"{run_name} do not record --replay-size)")

    # the postural-hold substitution runs only under --babble-bouts real WITH --rest-control
    # scheduled (train_mimo.py's scheduled_hold); H84 was launched with --rest-control
    # choice, and the page described the substitution as a fact of every run
    bouts, rest = args.get("babble_bouts"), args.get("rest_control")
    if bouts is None or rest is None:
        rest_note = (f"the launch flags of {run_name} do not record both --babble-bouts and "
                     f"--rest-control, so whether it is live there is not stated on this page")
    elif bouts == "real" and rest == "scheduled":
        rest_note = f"{run_name} was launched that way, so it is live there"
    else:
        rest_note = (f"{run_name} was launched with --babble-bouts {bouts} and --rest-control "
                     f"{rest}, so it does not happen there")

    d = encoder_depths(receipt)
    shape = model_shape(receipt, args)
    width = d["width"]

    def wide(m: str, fallback: Any) -> str:
        return str(width.get(m) or fallback)

    missing = f"a layer count the model receipt of {run_name} does not carry"
    if d["inventory"] and d["proprio_linear_layers"]:
        k, hw = d["proprio_linear_layers"], d["proprio_hidden_widths"]
        same = len(set(hw)) <= 1
        widths = (f"each {hw[0]} numbers wide" if hw and same else
                  ", ".join(str(x) for x in hw) + " numbers wide in turn" if hw else "")
        proprio = (f"a {k}-layer [[mlp|MLP]] ({_count(k - 1, 'hidden layer')}"
                   + (f", {widths}" if widths else "") + ", then one output layer; counted "
                   f"from the model receipt of {run_name})")
    else:
        proprio = f"an [[mlp|MLP]] with {missing}"

    def depth_of(key: str, word: str) -> str:
        return _count(d[key], word) if d["inventory"] and d[key] else missing

    out = {
        "{run_name}": run_name,
        "{intention_every}": str(sched["every"]),
        "{intention_every_source}": ("" if sched["every_recorded"] else
                                     " (the launcher's default: the launch flags do not "
                                     "record --intention-every)"),
        "{planner_off_note}": st["off_note"],
        "{planner_latch_rule}": st["rule"],
        "{controller_authorship}": st["controller"],
        "{planner_line_rule}": st["line"],
        "{babble_latch_share}": st["babble_share"],
        "{exec_source_tally}": exec_source_text(exec_source, stats_run or run_name),
        "{scheduled_rest_note}": rest_note,
        "{hold_env_steps}": str(hold * c["action_repeat"]),
        "{goal_ring_s}": f"{GOAL_RING_MOMENTS * step_s:g}",
        **hself_fill(args, step_s, run_name),
        "{hself_l2_window_s}": f"{HSELF_WINDOW_MOMENTS['l2'] * step_s:,.0f}",
        "{hself_l3_window_s}": f"{HSELF_WINDOW_MOMENTS['l3'] * step_s:,.0f}",
        "{collapse_window_s}": f"{COLLAPSE_WINDOW_STEPS * step_s:g}",
        "{moment_s}": f"{step_s:g}",
        "{imagination_forward}": forward,
        "{imagination_back}": back,
        # whether the chain's win rate sets the planner's depth in this run: only under --hand-target-mode world (the
        # code check of Oct 15, 2026; lem_model.py _plan_chain sets max_steps = 1 otherwise)
        "{imagination_depth_use}": (
            "It does decide one thing about behaviour: how often each depth beats assuming nothing changed is what "
            "sets how many model steps ahead the planner may aim, and the planner's first step is what the body "
            "executes."
            if (args.get("hand_target_mode") or "world") == "world" else
            "In " + run_name + " it decides nothing about behaviour either: under --hand-target-mode "
            + str(args.get("hand_target_mode")) + " the planner's search looks one model step ahead, so the depth "
            "this chain earns is computed and not used."),
        "{imagination_cap}": cap,
        "{replay_size}": replay_size,
        "{vision_width}": str(shape["vision_width"]),
        "{vision_patches}": f"{vision_patch_layout(args)[-1]['stop']:,}",
        "{vision_reach}": vision_reach_text(vision_reach(receipt, args), run_name),
        "{proprio_mlp}": proprio,
        "{tactile_depth}": depth_of("tactile_attn_layers", "[[attention]] layer"),
        "{motor_depth}": depth_of("motor_attn_layers", "[[attention]] layer"),
        "{audio_depth}": depth_of("audio_conv_layers", "1-D convolution"),
        "{intermind_depth}": depth_of("intermind_attn_layers", "self-[[attention]] layer"),
        **wellness_fill(args, run_name),
        **servo_fill(args, run_name),
        "{tactile_width}": wide("tactile", shape["encoder_width"]),
        "{motor_width}": wide("motor", shape["encoder_width"]),
        "{audio_width}": wide("audio", shape["encoder_width"]),
        "{intermind_width}": wide("intermind", shape["d_model"]),
        **gof_fill(receipt, args, run_name, source_mask),
        **planner_fill(receipt, args, run_name),
    }
    out["{tactile_comfort_definition}"] = TACTILE_COMFORT_DEFINITION
    frozen = args.get("eval_only")
    if frozen is True:
        update = f"{run_name} records --eval-only: model weights are frozen; there is no training update."
        replay = f"{run_name} records --eval-only: the trainer creates no replay sampler and runs no replay updates."
    elif frozen is False:
        update = f"{run_name} records training enabled: the live model step can update model weights."
        if args.get("replay") == "off":
            replay = f"{run_name} records --replay off: no replay updates run."
        elif args.get("replay") == "on":
            replay = (f"{run_name} records training enabled and --replay on: replay updates are enabled, "
                      "subject to the schedule and availability of usable windows.")
        else:
            replay = f"{run_name} does not record --replay; this page cannot identify its replay setting."
    else:
        update = f"{run_name} does not record eval_only; this page cannot identify whether weights were frozen."
        replay = f"{run_name} does not record eval_only; this page cannot identify whether replay training was enabled."
    out["{learning_update_run}"] = update
    out["{learning_replay_run}"] = replay
    # the tactile sense -> wellness entry (the independent review of Oct 15, 2026, S2): the self-contact weight and the
    # contact profile as this run was launched (train_mimo --self-touch-wellness, default 0.0; --contact-profile,
    # default legacy); a run that did not record one says so
    w = args.get("self_touch_wellness")
    if isinstance(w, (int, float)):
        out["{self_touch_weight_run}"] = (
            f"{run_name} was launched with --self-touch-wellness {w:g}, so self-contact adds 0 there"
            if float(w) == 0.0 else
            f"{run_name} was launched with --self-touch-wellness {w:g}, so the gated increase is multiplied by that weight")
    else:
        out["{self_touch_weight_run}"] = (f"{run_name} does not record --self-touch-wellness; train_mimo's default "
                                          "weight is 0, at which self-contact adds 0")
    cp = args.get("contact_profile")
    mech = ("mechanical_v2", "mechanical_v2_local", "mechanical_v3", "mechanical_v4")
    out["{contact_profile_run}"] = (
        f"{run_name} does not record --contact-profile; train_mimo's default is legacy, which carries no history"
        if not cp else
        f"{run_name} was launched with --contact-profile {cp}, so skin comfort reads that history there" if cp in mech
        else f"{run_name} was launched with --contact-profile {cp}, which carries no history")
    return out


def _b64_to_u8(data: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype=np.uint8)


def quantize(arr: np.ndarray) -> Dict[str, Any]:
    """float array -> {shape, min, max, u8(base64)}; decode = min + u8/255*(max-min)."""
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()) if a.size else 0.0, float(a.max()) if a.size else 0.0
    scale = (hi - lo) or 1.0
    u8 = np.round((a - lo) / scale * 255.0).astype(np.uint8)
    return {"shape": list(a.shape), "min": lo, "max": hi,
            "u8": base64.b64encode(u8.tobytes()).decode("ascii")}


def dequantize(q: Dict[str, Any]) -> np.ndarray:
    u8 = _b64_to_u8(q["u8"]).astype(np.float32)
    a = q["min"] + u8 / 255.0 * (q["max"] - q["min"])
    return a.reshape(q["shape"])


def _finite(x: Any, path: str, errs: List[str]) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            _finite(v, f"{path}.{k}", errs)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _finite(v, f"{path}[{i}]", errs)
    elif isinstance(x, float) and not math.isfinite(x):
        errs.append(f"non-finite float at {path}")


def validate(snap: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """(ok, problems). Structural checks only - content invariants (row sums, causal
    mask) are asserted pre-quantization in capture.py and its tests."""
    errs: List[str] = []
    for b in REQUIRED_BLOCKS:
        if b not in snap:
            errs.append(f"missing block: {b}")
    hdr = snap.get("header") or {}
    for k in REQUIRED_HEADER:
        if k not in hdr:
            errs.append(f"missing header field: {k}")
    if hdr.get("schema_version") not in ACCEPTED_VERSIONS:
        errs.append(f"schema_version {hdr.get('schema_version')} not in {ACCEPTED_VERSIONS}")
    if hdr.get("weights") not in WEIGHTS_LABELS:
        errs.append(f"weights label {hdr.get('weights')!r} not in {WEIGHTS_LABELS}")
    if isinstance(hdr.get("schema_version"), int) and hdr["schema_version"] >= 3:
        for b in V3_BLOCKS:
            v = snap.get(b)
            if not (isinstance(v, dict) or (isinstance(v, str) and v.startswith("absent"))):
                errs.append(f"schema v3 block {b} is neither a record nor an 'absent: ...' reason")
    att = snap.get("attention")
    if isinstance(att, dict) and "layers" in att:
        for li, layer in enumerate(att["layers"]):
            for key in ("stage1", "stage2"):
                q = layer.get(key)
                if q is None:
                    continue
                for f in ("shape", "min", "max", "u8"):
                    if f not in q:
                        errs.append(f"attention.layers[{li}].{key} missing {f}")
                try:
                    n = int(np.prod(q["shape"]))
                    if _b64_to_u8(q["u8"]).size != n:
                        errs.append(f"attention.layers[{li}].{key} payload size mismatch")
                except Exception as e:  # malformed base64/shape
                    errs.append(f"attention.layers[{li}].{key} undecodable: {e}")
    _finite({k: v for k, v in snap.items() if k != "senses"}, "snap", errs)
    return (not errs), errs
