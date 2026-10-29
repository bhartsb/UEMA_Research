# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Build the explainer page: one self-contained HTML file with three tabs
(architecture walkthrough, instance inspector, run replay) for one run.

    python -m mimo_bridge.tools.build_explainer [--run DIR] [--watch] [--every 30]
        [--max-snapshots 12] [--out PATH]

Reads what the run already writes (args.json, model_receipt.json,
replay_training_stats.jsonl, explain/*.json) and inlines everything - the page
works from file:// with no server and no network. Writes are atomic (tmp +
os.replace) and skipped when the rendered bytes are unchanged, so --watch on a
30 s cadence does not grind an idle page into the disk. Page budget: 6 MB, and the
builder keeps the page inside it by itself: a snapshot's size follows the model's
(measured 125-450 KB on the 128-wide runs, 668-705 KB on the 192-wide H84), so the
count of embedded snapshots is whatever fits - always the newest and the first, the
rest thinned evenly - and the page says how many exist and how many it embeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimo_bridge.explainer import schema as SCH                    # noqa: E402
from mimo_bridge.explainer.walkthrough import build_steps          # noqa: E402
from mimo_bridge.explainer.current_state import (                  # noqa: E402
    current_state, current_state_html, local_time_label, selected_production_run,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mimo_bridge.build_explainer")

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explainer")
SAVED = os.path.abspath(os.path.join(ROOT, "..", "..", "Saved", "mimo_bridge"))
PAGE_BUDGET_MB = 6.0
SERIES_POINT_CAP = 1200

# The flag values the page may print. viewer.js can only check a drawn number
# against META.args, so any flag a sub-diagram names has to be here or the picture
# states a constant the build cannot falsify. Published twice when the charts come
# from another run (--stats-from): META.args is the --run directory's, META.stats_args
# is the chart run's, and the run tab names the values where the two disagree.
PUBLISHED_ARGS = (
    "scene", "n_envs", "context_length", "residual_prediction",
    "latent_body_hierarchy", "latent_body_levels", "latent_planner", "gaze_reflex",
    "next_frame_loss", "top_down_gate", "intention", "intention_every", "motor_hold",
    "action_repeat", "start_phase", "eval_only",
    # the replay sub-diagram names the ring's capacity, its on/off state and the
    # fill gate; all three were literals in the drawing because the build never
    # carried them
    "replay", "replay_every", "replay_batch", "replay_size", "fill_ring_first",
    "fovea_size", "curriculum",
    # P119: the plan popup prints the most a moment can store WITHOUT the curiosity
    # factor from these three; and the 2026-09-18 launch settings the page names
    "movement_bonus", "agency_weight", "agency_prior", "consequence_w",
    "planner_effort_per_body", "self_choice", "decoder_kp",
    "audio", "caregiver_mode", "caregiver_present_seconds",
    "caregiver_absent_seconds", "hierarchical_self", "hself_slots",
    "persistent_memory", "self_state", "body_layout",
    "imagination", "imagination_steps", "imagination_weight", "steps_per_phase",
    # curiosity_goal decides whether the drive reaches goal choice at all, and
    # wellness_mode whether it scales the wellness lesson - the curiosity
    # sub-diagram states both
    "curiosity_band", "curiosity_goal", "plan_horizon", "wellness_mode",
    # the two flags that decide what the wellness terms ARE: how much a fully
    # self-touching moment may add (the sixth grounded term, 2026-09-21), and which
    # reading of the body map the coherence term carries. body_schema_updates says
    # whether that map runs at all, which is what decides between the two readings
    # and the cosine fallback
    "self_touch_wellness", "wellness_coherence", "body_schema_updates",
    # who writes the order the body executes, and what is done to it on the way:
    # babble reaches the body only on non-planner steps, the rest schedule can
    # replace the order with a postural hold, and smoothing blends it with the
    # previous one
    "babble_noise", "babble_bouts", "rest_control", "smooth_actions",
    "spontaneous_saccades", "contact_learning",
    # the world the body lives in: which contact path the skin's hurt comes from, whether the
    # adult hands the toy over, the joints' armature and damping, the eye servo's gains and the
    # body-function modulation (2026-09-21/22 settings the page names)
    "contact_profile", "toy_hand_off", "finger_armature", "hinge_armature",
    "joint_damping_scale", "gaze_kp", "gaze_kd", "vision_gate", "body_function",
    "self_prediction", "anticipation_gaze", "anticipation_gaze_w",
    # the named-entity self and its reader: the diagram draws their box, the popups name the
    # teacher, the reader's weight, its level and its gate
    "entity_self", "entity_self_teacher", "self_value_weight", "self_value_level",
    "self_value_gate_windows",
    # the gain-of-function build's flags (docs/gain_of_function_v3/DESIGN.md section 2): each
    # decides whether a box, a panel, a slide or a chart describes something that ran
    "causal_chain", "efference_heads", "efference_weight", "vestibular_sense",
    "wellness_rank", "wellness_weights", "controllability", "learning_progress",
    "curiosity_mode", "consequence_self_contact", "contact_head", "contact_head_weight",
    "contact_control_wellness", "motor_synergy", "synergy_dim", "synergy_babble",
    "synergy_planner", "synergy_residual_l2", "smooth_source", "next_frame_norm",
    "liveness_repair", "liveness_rule", "resume_env_settings",
    # Publish the selected run's own values; absent values remain absent in the
    # presence map/current-state panel, rather than borrowing today's defaults.
    # Only flags this checkout's train_mimo parses (test_gof_every_published_flag_is_one_the_
    # trainer_parses). Four repair flags were listed here on 2026-09-27 (commit 520ef50a4) while
    # they existed only on unmerged branches, so the page printed "Not recorded" for settings
    # production's train_mimo cannot take: --heads-memory-constant and --heads-score-bound
    # (fix/prediction-heads-slot-lock-2026-09-26), --gaze-copy-centre (fix/efference-vision-input-
    # 2026-09-26) and --liveness-centre-pace (fix/revived-centre-pace-2026-09-27). Removed
    # 2026-10-02; the merge that adds one of them to train_mimo adds it here and to
    # explainer/current_state.py's REPAIR_SETTINGS.
    "persistent_memory_repair", "persistent_memory_constant_return",
    "motor_input_fold", "self_percept_release", "motor_asked_change",
    # whether the planner's action search reads the efference body head (2026-09-24): the
    # efference box's popup and its live rows name this run's setting
    "physical_goals",
    # the planner's drawing and its plan rows (2026-10-02): whether a command carries finger joint
    # angles and a plan is one command long, whether a second search sets the fingers, how the
    # strength part of the score is summed, and the goal-choice and score options the text names
    "hand_target_mode", "hand_goals", "goal_locality", "prospective_wellness",
    "prospective_wellness_w", "no_intention_execute",
    "executed_fields_only",
    # the repairs drawn inside their host boxes (the decided design of the Architecture tab, Oct 7, 2026, D-E4): the
    # prediction heads' all-slot path (PREDICTOR_REPAIR) beside the motor encoder's asked-change path above, and the
    # attention cap with its bound (ATTENTION_CAP), which the stage-1 and motor-encoder boxes name when it is on
    "heads_all_slot_path", "attention_cap", "attention_cap_bound",
    # whether the curriculum can still change the body (the code check of Oct 15, 2026): a run that starts in the final
    # phase with its age fixed and its delays off can never change phase, so the curriculum box says so and draws no
    # line to the world
    "age_binding", "delay_binding",
)

# chart-level documentation, same t/w/h/f idiom as the dashboard and the schema
DOC_CHARTS: Dict[str, Dict[str, str]] = {
    "chart.causal": {
        "t": "causal next-state error",
        "w": "How wrong the model was about the next model step, given the command it "
             "executed ([[causal_pe]] - click for the full definition). An "
             "error: smaller is better.",
        "h": "mimo records' causal_pe / causal_pe_per_modality - the delayed lesson "
             "(state and action at t-1 predict state at t), graded against raw tokens",
        "f": "reading it without body speed (the chart below): a stiller body is "
             "easier to predict, so a falling error can be a stilling body",
    },
    "chart.speed": {
        "t": "body speed",
        "w": "How fast the body is actually moving - the yardstick every error "
             "chart must be read against.",
        "h": "mimo records' movement.local_speed (cm/s)",
        "f": "comparing errors across runs at different speeds as if speeds were equal",
    },
    "chart.contacts": {
        "t": "skin contacts per body",
        "w": "How many skin contacts one body carries: 0 = touching nothing. The "
             "context for every tactile number on this page.",
        "h": "mimo records' contacts_per_env - each step's contact count averaged "
             "over the bodies, then averaged over the report window",
        "f": "reading it as pressure or as hurt: it counts contacts and says nothing "
             "about how hard any of them presses",
    },
    "chart.stillness_debt": {
        "t": "stillness debt",
        "w": "How overdue the body is to move, 0 to 1. It rises while the body's "
             "local speed stays under a quarter of the reference speed, reaching 1.0 "
             "after 150 seconds, and clears after 30 seconds of moving at the "
             "reference speed.",
        "h": "mimo records' movement.stillness_debt - mimo_bridge/movement_drive.py's "
             "debt, averaged over the bodies and over the report window",
        "f": "reading a low debt as chosen movement: exploration noise moves the body "
             "too, so the debt cannot say who commanded the movement",
    },
    "chart.rank1": {
        "t": "command-to-outcome matching (rank1)",
        "w": "At each command's end (every 250 ms): a projection of the command is compared by "
             "cosine with the sensory change that followed it and with a set of decoy changes - the "
             "other bodies' changes over the same model steps and a bank of recent ones. The chart "
             "is how often the command's OWN change scores highest. "
             "This is the signal the project exists to produce.",
        "h": "mimo records' tenure_tok.rank1 (per event, window-averaged); "
             "vis_gate_open = share of events with the hand in view at both ends",
        "f": "guessing scores 1/(1+decoys), not zero - always read rank1 against "
             "the decoy count",
    },
    "chart.exec": {
        "t": "executed-action contract",
        "w": "The share of steps whose executed command was handed back to the model "
             "as the cause of what it sees. Complete at 100%.",
        "h": "contract.exec_seen divided by steps since this process segment's "
             "launch record (exec_seen is process-local - review T13)",
        "f": "a run where this is near zero can still train - it just cannot learn "
             "what its actions do",
    },
    "chart.license": {
        "t": "depth licence",
        "w": "How far ahead the planner is allowed to aim, and what earned it. "
             "The imagination chain is graded at each depth against assuming "
             "nothing changed since its first frame; the share of model steps it wins "
             "that comparison decides the planner's reach, and the planner's first "
             "step is what the body executes.",
        "h": "mimo records' plan_license.win_emas (a slow running average, one per "
             "chain depth), plan_license.licensed, and the intention record's "
             "horizon and plan_length; produced in mimo_bridge/lem_model.py",
        "f": "the win share is an admission line, not an achievement - it says the "
             "chain beat a do-nothing guess, not that the plans reach anything. "
             "The two sides also do not start from the same tokens: the chain "
             "starts from what the transformer read (with the model's own "
             "expectation blended in where the top-down gate is open, capped at "
             "half) and the do-nothing side from the untouched measured tokens The gap on this chart between the licensed depth and the plan length is the 5% divergence guard trimming the rollout (P125).",
    },
    "chart.collapse": {
        "t": "token temporal std",
        "w": "The [[collapse|collapse watch]]: how much each sense's token changes "
             "over time. Zero would mean the encoder says the same thing every step "
             "no matter what the body senses.",
        "h": "mimo records' token_temporal_std: the standard deviation of each of "
             "the token's numbers over the last 200 stored samples, averaged over "
             "those numbers. One sample is stored every 5th model step and from "
             "body 0 only, so the window is about 1,000 model steps - {collapse_window_s} s of body "
             "time at {action_repeat} physics sub-steps per step - for one body, not the batch",
        "f": "a gate blending predictions into tokens could mask collapse - this "
             "watch reads RAW tokens on purpose (review M9)",
    },
    "chart.collapse_bodies": {
        "t": "the quietest body's token spread",
        "w": "The [[collapse|collapse watch]] read on every body, and for each sense the body whose token "
             "moves least is drawn. 0% = that body's token is the same at every model step: it has "
             "stopped reporting the sense (a deaf body), however lively the other bodies are. Measured on "
             "H84 (Sept 18-25, 10 checkpoints, 8 bodies, 6 windows of 50 s each), the live bodies read: "
             "vision 32-51%, tactile 17-50%, audio 34-48%, proprioception 26-52%, motor 47-68% (5th to "
             "95th percentile); deaf bodies read vision 5-19% (one body, lying face down) and "
             "proprioception 4-20%. The hourly watcher names every body under its sense's line "
             "(mimo_bridge/deaf_senses.py: vision and proprioception 20%, tactile 9%, audio 29%, motor "
             "39%, vestibular 1.6%). The vestibular token reads 0% on every body since Sept 23 (its "
             "encoder's copied constant); the placeholder before it read 2-24%.",
        "h": "mimo records' token_temporal_std_bodies: per sense, one number per body - the standard "
             "deviation of each of that body's token numbers over the same 200 samples as the collapse "
             "watch (every 5th model step, about {collapse_window_s} s), averaged over the numbers; the "
             "record holds all the bodies, this chart draws the lowest",
        "f": "body 0's number is the collapse watch itself; the watch alone missed two deaf bodies for "
             "five days (docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md)",
    },
    "chart.tenure_gap": {
        "t": "what knowing the command is worth",
        "w": "At each command's end: of the pose error a do-nothing command leaves, how much the "
             "[[executed-command]] removes, in per cent - 0% = the command explains nothing, 100% = "
             "the pose change is predicted exactly, below 0% = the command made it worse. The raw "
             "errors behind it have no lived unit, so the page draws only this share.",
        "h": "mimo records' tenure_gap.pose / (tenure_pe.pose + tenure_gap.pose) - the gap is "
             "the do-nothing error minus the with-command error; the command-to-outcome pose head, "
             "scored at command-ending events only",
        "f": "a run with no command-ending events plots nothing at all; and a still "
             "body makes both errors small, so read the gap next to body speed",
    },
    "chart.cmd_follow": {
        "t": "command following",
        "w": "Whether the limbs went where they were told, drawn between three named ends: the "
             "joints turned exactly the way the command pointed, unrelated to it, or exactly "
             "against it. The match is an averaged cosine: not yet measured as a count of "
             "commands. The command size chart says whether real movement was commanded at all.",
        "h": "mimo records' cmd_response_cos (the match one step after the command), "
             "cmd_response_cos_held (the same match one full command interval later, "
             "which the environment code documents as the fair read because a limb "
             "with mass cannot respond inside one step) and cmd_mag_rad (the "
             "commanded turn, stored in radians; the chart converts it to degrees)",
        "f": "reading the one-step match as disobedience: on identical commands "
             "the environment measured a match much nearer 'exactly as commanded' at 250 ms "
             "than at one step (averaged cosines: not yet measured as a count)",
    },
    "chart.recon_dec": {
        "t": "reconstruction decoders (the guard)",
        "w": "Separate readouts re-describe the CURRENT model step from the token - "
             "the proof the token still holds the present. An error, drawn as a "
             "percentage of the error a decoder answering all zeros makes: smaller "
             "is better.",
        "h": "mimo records' recon_dec per sense, from the decoders' own loss",
        "f": "it says nothing about prediction; a model can re-describe the "
             "present perfectly and still predict the next model step badly",
    },
    "chart.pe": {
        "t": "predicted change",
        "w": "How far the guess of the NEXT model step sits from the present one. It "
             "GROWS when the model predicts movement, so rising is not damage and "
             "neither direction is good by itself.",
        "h": "mimo records' pe per sense - the prediction heads' output against the "
             "current token, not against what arrived",
        "f": "reading it as accuracy: the accuracy number is the causal error "
             "chart, which grades the guess against what really arrived",
    },
    "chart.replay": {
        "t": "re-study of remembered model steps",
        "w": "The error on remembered model steps the model re-studies between live "
             "steps, drawn as a percentage of its own first window in the log (a loss "
             "has no lived unit). An error: smaller is better.",
        "h": "mimo records' replay_loss - the replay update's own loss, averaged "
             "over the window. That loss is the five prediction heads graded "
             "against the real next model step plus the motor controller graded "
             "against the real next achieved pose, and nothing else: no "
             "anti-collapse term, no imagination, no delayed next-model-step lesson, "
             "no reconstruction, no wellness, no tenure. Five of the six encoders "
             "are excluded because their tokens were stored detached; the MOTOR "
             "encoder is not, because its token is rebuilt with gradients on",
        "f": "reading it beside the live loss as if they measured the same thing - "
             "they are different sums over different data, and this one does not "
             "appear in the inspector's loss panel, which covers the live step only",
    },
    "chart.loss_terms": {
        "t": "the teaching signal, split by term",
        "w": "Each line is one lesson's share of the window's positive lessons added up, "
             "out of every 100 parts. A loss has no lived unit and the terms share no "
             "scale, so the page draws shares, never raw values; the self-perception "
             "likelihood is negative and stays out of the positive total. A term at "
             "zero taught nothing in that window.",
        "h": "training records' loss_breakdown entries, each over the sum of the positive ones",
        "f": "assuming a term is on because it exists: imagination had zero "
             "records for a whole era of runs while its box was still drawn",
    },
    "chart.wellness": {
        "t": "Measured wellness scores",
        "w": "Each line is a fixed-formula score averaged over a report window, displayed as score times 100. Prediction accuracy, capacity, coherence and motor state range from zero to one. Tactile comfort tops out at 85%; self-touch is limited by its configured weight. The movement line is an additional goal-ranking input, outside the six-score measured sum. [[wellness|Inputs and formulas]].",
        "h": "mimo records: wellness entries, averaged over each reporting window.",
        "f": "These lines are measured scores, not learned-predictor outputs or normalized goal ranks. Higher prediction accuracy means closer encoded predictions; it does not imply less movement.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "chart.curiosity": {
        "t": "Curiosity drive: preference over prediction error",
        "w": "The drive is zero at and outside the configured error-band edges and reaches one at the midpoint. Each point is the report-window average, displayed as a percentage of full drive. The formula measures position within that chosen band, not whether the observations are learnable. [[curiosity-mode|How ranking uses it]].",
        "h": "train_mimo.py accumulates per-step curiosity_drive and divides by the report-window step count.",
        "f": "When curiosity-goal is enabled, normalized ranking uses a separate channel; raw ranking uses its configured additive or multiplicative rule. Direct and forward modes can also scale the wellness lesson.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "chart.agency": {
        "t": "agency drive",
        "w": "Evidence that an outcome was the model's own doing pays wellness. "
             "The payment follows wins in the command-to-outcome match; the "
             "hand-watching prior fades as vision's own evidence catches up with "
             "touch's.",
        "h": "mimo records' agency entries: pay, prior_w, vis_adv and touch_adv",
        "f": "the advantages are measured against decoys, so they say which sense "
             "carries the evidence, not how strong the evidence is in absolute "
             "terms",
    },
    "chart.td_gate": {
        "t": "top-down gate, per sense",
        "w": "How much of the model's own expectation was blended into each "
             "sense's token, in parts of the token out of 100: 0 = the core read "
             "measured reality only, and the blend is capped at 50. Learning is "
             "graded on raw tokens whatever this reads.",
        "h": "mimo records' td_gate per sense, averaged over the window, from the "
             "gate's own opening in the model step",
        "f": "a gate that opens changes what the core acted on without changing "
             "what any lesson was graded against - check the causal error for a "
             "split between the acting path and the learning path",
    },
    "chart.vision_fresh": {
        "t": "fresh vision share",
        "w": "Out of every 100 steps of a report window, how many had their eye pictures "
             "rendered on that step: 100 = every step read a new picture, 0 = every step "
             "re-used the previous vision token.",
        "h": "mimo records' vision_fresh_fraction - train_mimo.py counts, on every "
             "step, whether body 0's frame carried newly rendered vision patches, and "
             "divides by the window's step count",
        "f": "reading it as a statement about the whole batch: only body 0 is counted",
    },
    "chart.throughput": {
        "t": "throughput",
        "w": "Model steps per second: a number about the rig, not about the model.",
        "h": "mimo records' model_steps_per_s",
        "f": "a fall here changes how much experience an hour buys, but says "
             "nothing about what was learned",
    },
    # ---------------- the gain-of-function build (2026-09-22) and the stats fields no chart
    # drew before it (the explainer survey's stale item 17)
    "chart.causal_choice": {
        "t": "causal chain: was the chosen plan the one predicted best?",
        "w": "Out of every 100 (planning event, body) pairs: how many times the "
             "planner's chosen command had the highest predicted wellness among itself "
             "and its [[alternatives]], how many times it was the closest to its goal, "
             "and how many times its first command reached the body. The flat line is "
             "what picking at random among the chosen plan and its alternatives "
             "scores (20 in 100 with four alternatives). Since 2026-09-23 a plan counts "
             "as best only when its lead over the next exceeds the tie band - the "
             "evaluator's own error on a difference between two candidates (1.5% of "
             "the five scores' range) - and as tied inside it (review b1, major 3). "
             "That band has been wider than the spread of the candidates' predicted "
             "wellness in every report window of H84 since then (1,174 of 1,174; median "
             "12.6 times the spread), so every choice reads tied whatever the forward "
             "model predicts: the tied line is no evidence about the forward model. "
             "The planner-side reading of the forward model is the planner recheck's "
             "captured share (the Development page's planner-candidate round). The goal "
             "line is unmeasured: the planner does not aim at predicted wellness, so 100 "
             "is not what a working planner would read.",
        "h": "mimo records' causal_chain block: chosen_best_share, chosen_tied_share, "
             "alternatives_n (the chance line is 1 / (1 + alternatives_n)), "
             "chosen_closest_share, reached_body_share, each drawn x 100; tie_band, "
             "gap_to_best_alt_mean and candidate_spread_mean as per cent of the five "
             "scores' range; plan_exec_distance_cm in cm - mimo_bridge/causal_chain.py. "
             "The band against the spread over the lineage: docs/gain_of_function_v4/"
             "agents/hold_loss_history_2026-10-02.json (ties)",
        "f": "reading a low best line as a failed planner: the planner descends "
             "on distance to its goal, not on predicted wellness, so this measures "
             "whether the two agree. Its own objective is the closest line. Second "
             "trap: reading the tied line as the forward model failing to separate the "
             "candidates - the band, wider than any predicted spread so far, decides it",
    },
    "chart.causal_prediction": {
        "t": "causal chain: the prediction made before the command, against 'nothing changes'",
        "w": "For every command sent to the body, the model's prediction made BEFORE "
             "it was sent, graded against the raw tokens that arrived, as a "
             "percentage of the error of predicting 'nothing changes' "
             "([[persistence]]): 100% = no better than nothing changes, 0% = exact, "
             "above 100% = worse than nothing changes. One line per sense. A model "
             "that foresees what its command does reads well under 100%; the distance "
             "to 0% is how much of the change it still misses.",
        "h": "mimo records' causal_chain.pred_err_* divided by persist_err_*, per "
             "sense, x 100; next = one model step after the latch, latch = the command's end",
        "f": "reading a line below 100% as a model that knows what its command does: "
             "the command-free part of the prediction beats 'nothing changes' too. "
             "The command's own share is the efference charts' shuffled line and the "
             "probes' captured share",
    },
    "chart.causal_wellness": {
        "t": "Predicted versus measured wellness change",
        "w": "Compare the change in the learned predictor’s five-score sum with the change measured by the five corresponding fixed formulas over the command hold. Both changes are divided by five and multiplied by 100 for display. Five is the chart’s reference scale, not the attainable maximum of the fixed sum: tactile comfort tops out at 85%.",
        "h": "mimo records: causal_chain.predicted_wellness_change and realized_wellness_change, each multiplied by 20.",
        "f": "Positive means the sum increased; negative means it decreased. A predicted increase need not occur in the body. These records are separate from the remembered-goal ranking score.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "chart.efference_body": {
        "t": "efference body head: its error against doing nothing and another body's command",
        "w": "The body head's mean error, in centimetres per handle, beside the "
             "lines it must beat: doing nothing (predicting no movement: the size "
             "of the movement itself) and another body's command with this body's "
             "state and pose, size-matched to this body (that body's asked moves "
             "and turns, measured from its own pose, applied to this body's pose). "
             "Below doing nothing = it foresees some of the movement; below the "
             "size-matched line = it uses THIS command. A head that foresees the "
             "whole movement reads 0 cm. For one segment a fourth line shows "
             "another body's absolute command (its targets as they stand), the "
             "original control.",
        "h": "mimo records' efference body_cm_err_*, body_cm_do_nothing_*, "
             "body_cm_size_matched_*, body_cm_shuffled_* (the absolute command); "
             "next = 50 ms, latch = the command's 250 ms (lem/model/efference.py "
             "window_summary)",
        "f": "reading the gap to the absolute line as command use: that command "
             "asks this body for changes chosen for another pose, so any head "
             "whose output grows with the asked change errs more with it; a head "
             "whose error equals the size-matched line ignores the command it is "
             "handed, however far under doing nothing it sits",
    },
    "chart.efference_gaze": {
        "t": "efference gaze head, degrees of eye turn",
        "w": "The gaze head's mean error in degrees at 50, 100 and 150 ms, beside "
             "doing nothing (the size of the eye's own turn) at the same horizons. "
             "A head that foresees the eye's turn exactly reads 0 degrees.",
        "h": "mimo records' efference gaze_deg_err and gaze_deg_do_nothing (lists "
             "of three horizons)",
    },
    "chart.vestibular": {
        "t": "vestibular: prediction against 'nothing changes'",
        "w": "The model's next-model-step error on the vestibular token as a percentage "
             "of the error of [[persistence]] on the same model steps: 100% = no better "
             "than nothing changes, 0% = exact. The sense's encoder starts fresh at "
             "the switch, so the first hours sit above 100%.",
        "h": "mimo records' vestibular pred_err divided by persist_err, x 100",
    },
    "chart.vestibular_posture": {
        "t": "posture from the vestibular vector, degrees",
        "w": "head tilt (0 = the head as it lies still on the back, 90 = a quarter "
             "turn, 180 = upside down), trunk roll (0 = on the back, 90 = on a "
             "side, 180 = face down, mean of its size over bodies) and trunk "
             "pitch.",
        "h": "mimo records' vestibular head_tilt_deg, trunk_roll_deg, "
             "trunk_pitch_deg (mimo_bridge/vestibular_sense.py, reference "
             "directions measured 2026-09-22)",
    },
    "chart.posture": {
        "t": "how far the bodies have rolled",
        "w": "The mean and the largest roll of the chest over the window, in "
             "degrees (0 = lying on the back, 90 = on a side, 180 = face down), "
             "and out of every 100 model steps of a body, how many were rolled past 120 "
             "degrees (0 = none, 100 = every body face down for the whole window).",
        "h": "mimo records' roll_deg_mean, roll_deg_max, rolled_over_fraction x 100 "
             "(each body's roll_max_deg over 120, averaged over bodies and the "
             "window), from the world's body_orientation",
        "f": "reading a rolled body as a skill: a body that rolled once and "
             "stayed face down reads the same as one that rolls on purpose",
    },
    "chart.controllability": {
        "t": "Controllability: predicted versus measured command effects",
        "w": "The chart shows the percentage of commands whose commanded and predicted effects exceeded the measurement floors, the gate state at the window’s end, and the real-command advantage over a shuffled-command control. Agreement combines direction match and relative effect size. [[controllability|Exact calculation and gate]].",
        "h": "mimo records: controllability.measured_share and gate_open, multiplied by 100; difference_mean is the measured-minus-control agreement difference.",
        "f": "The control predicts this body’s effect using another body’s command. The gate allows this channel to affect normalized goal ranking only after its controlled evidence passes the configured rule.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "chart.controllability_gates": {
        "t": "controllability: the gate in force beside the shadow gate",
        "w": "Out of every 100 graded commands a gate judged in a report window, after "
             "how many it stood open. Two gates are judged on the same graded commands "
             "(the bodies' commands latched on one model step are graded together when "
             "the next commands appear). Each takes one reading per graded command: the "
             "mean, over the bodies measured, of the agreement between the body head's "
             "expectation from this body's own command and what the body did, minus the "
             "same agreement for the expectation from a control command. The gate in "
             "force takes another body's absolute command as the control (its targets "
             "as they stand) and sets the controllability weight in the goal ranking; "
             "the shadow gate takes another body's size-matched command (its asked moves "
             "and turns, measured from its own pose, applied to this body's pose) and is "
             "recorded only. Each opens when the mean of its last 32 readings is above 3 "
             "of their standard errors and shuts when it is under 1.",
        "h": "mimo records' controllability gate_open_commands / gate_commands and "
             "shadow_gate_open_commands / shadow_gate_commands, x 100; "
             "gates_same_commands, gates_both_open_commands, gate_only_open_commands, "
             "shadow_gate_only_open_commands, size_matched_control_mean, "
             "size_matched_difference_mean, shadow_gate_mean, shadow_gate_standard_error "
             "and shadow_gate_readings_above_zero are logged beside them, and "
             "gate_definitions in the block defines each (lem/model/wellness_channels.py "
             "ControllabilityTracker.summary, from the code of 2026-10-03)",
        "f": "reading the shadow gate as a change to training: no weight, ranking or "
             "loss reads it. And reading a window in which neither gate took a reading "
             "as a shut gate: the chart draws no point for that window",
    },
    "chart.learning_progress": {
        "t": "Learning progress: fast versus slow relative prediction error",
        "w": "Each point compares the fast model-error/no-change-error ratio with the slow ratio: (slow minus fast) divided by slow, clipped between minus one and one. Display multiplies by 100. Positive means a lower recent relative error. [[learning-progress|The two error averages, window lengths and body/sense aggregation]].",
        "h": "mimo records: learning_progress.value_mean and per_sense; each displayed value is multiplied by 100.",
        "f": "World difficulty can change this reading with frozen weights. A zero trend after improvement does not mean that the improvement was lost.",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "chart.wellness_rank": {
        "t": "which channels made the goal win",
        "w": "Out of every 100 parts of the winning goal's lead over the runner-up, "
             "how many each channel carried, summed over the window's planning "
             "events. They add to 100; one channel near 100 decides alone; below 0 "
             "pushed against the winner. With it, out of every 100 candidates, how "
             "many were over the integrity gate line, and out of every 100 planning "
             "events of a body, how many found EVERY candidate over it. A model step over "
             "the line is never a goal while any candidate of that body is under it; "
             "when all are over it, the least harmful one becomes the goal.",
        "h": "mimo records' wellness_rank.winner_margin_share, integrity_gated_share "
             "and all_gated_share, x 100 (variance_share in the same block); the "
             "fallback is rank_values in lem/model/wellness_channels.py",
    },
    "chart.contact_head": {
        "t": "contact head against its base-rate control",
        "w": "Out of every 100 pairs of a touched and an untouched patch at a model step, how many "
             "the head scores the right way round (50 = a coin flip, 100 = every "
             "pair), for the head and for the base-rate control (each patch's rate "
             "given whether it was touched the model step before). The head's score "
             "depends on the command only where it rises above the control. Since "
             "2026-09-23 the same ranking is also taken over ONSETS only - patches at a model step "
             "whose bit was off the model step before - for own-part and outside touches "
             "apart, where the model step before cannot give the answer away.",
        "h": "mimo records' contact_control onset_auc_self, onset_auc_world and their "
             "control_onset_auc_* when present, else head_auc_bits.any and "
             "control_auc_bits.any (area under the curve), x 100",
    },
    "chart.contact_control": {
        "t": "foreseen touches and what they paid",
        "w": "Out of every 100 self-touches that began and could be scored, how "
             "many the head foresaw with this body's command. The control runs only "
             "on some model steps (every few steps, when there are two or more bodies): "
             "on those model steps the head is asked twice, once with this body's command "
             "and once with another body's command, and the two counts are drawn side "
             "by side over the SAME onsets. The same for outside touches. And what "
             "contact_control carried per 100 model steps of a body against what it would "
             "carry with its gate open (a foreseen self-touch pays 2 and an "
             "outside touch 1, each habituated).",
        "h": "mimo records' contact_control success_self and success_world (every scored "
             "onset); success_paired against success_control and success_paired_world "
             "against success_control_world (the onsets of the control model steps only, "
             "lem/model/contact_head.py summary); paid_mean, would_pay_mean, habituation_mean "
             "(the average pay factor 1 / (1 + repeats)), gate_self_open and gate_world_open "
             "(two gates since 2026-09-23), each x 100",
        "f": "setting success_self beside success_control: they are counted over different "
             "onsets (every scored onset against the control model steps' onsets only), so their "
             "gap is not the command's effect; the same-onset pair is success_paired against "
             "success_control. And the every-onset line alone: at the measured margins a head "
             "that knows nothing of the command still marks about 1 onset in 10 foreseen",
    },
    "chart.self_contact": {
        "t": "self-contact per body",
        "w": "How many of the body's own part pairs touch each other, and how many "
             "skin patches carry an own-part touch, averaged over the bodies and "
             "the window; with the onsets, how many self-touches BEGAN per 100 "
             "model steps of a body.",
        "h": "mimo records' self_contact_pairs, self_contact_patches, "
             "self_contact_events.onsets_per_body_step x 100",
    },
    "chart.self_contact_share": {
        "t": "what share of touch is the body's own",
        "w": "Out of every 100 parts of all contact force, how many come from the "
             "body's own parts (0 = none, 100 = all); and out of every 100 "
             "self-touch onsets, how many happen at rest anyway (structural) and "
             "how many within 30 degrees of an eye's axis (in view).",
        "h": "mimo records' self_contact_force_share, "
             "self_contact_events.structural_share and in_view_share, x 100",
    },
    "chart.hand_press": {
        "t": "hand pressing, world against self",
        "w": "Out of every 100 model steps of a body, how many had a hand patch pressed "
             "over 2 kPa by something outside the body, and how many by the body "
             "itself.",
        "h": "mimo records' hand_press world_share and self_share, x 100 (env.py "
             "split_hand_press)",
    },
    "chart.consequence": {
        "t": "the consequence term",
        "w": "Out of every 100 model steps of a body, how many carried at least one type of "
             "consequence (a hand in contact, an object moved, a view gained); out "
             "of every 100 planning events, how many chose a goal that carried one; "
             "and, under the world-only rule, how many the old count-every-press "
             "rule would have given.",
        "h": "mimo records' consequence rate, goal_rate, rate_count_rule, x 100",
    },
    "chart.incorporation": {
        "t": "the body map's co-movement numbers, over time",
        "w": "Three kinds of line, each a percentage of the way from the map's floor (a score "
             "of 0) to its top (a score of 1). The handles' line: the mean over the body's 26 "
             "handles of how often a handle's command change and proprioception's change are "
             "large in the same model step - co-movement, not proof that the command caused the "
             "change. One line per outside thing: how closely it travelled with a hand, minus "
             "the same score with the hand's direction reversed. The extent line: the mean of "
             "all of those numbers, which is what the coherence term of wellness carries.",
        "h": "mimo records' incorporation handles_mine, extent, objects.*, mapped "
             "from the map's own sigmoid ends (floor and top pinned by "
             "test_explainer_capture's incorporation test); the handle statistic is "
             "lem/model/body_schema.py update() and self_membership()",
        "f": "reading a rising handles line as growing control of the body: the statistic has no "
             "time-lag bank, no sign and no comparison against movement without a command, and "
             "two unrelated random signals read 30 to 80 of 100 on its own output "
             "(mimo_bridge/tests/test_body_schema_co_movement.py)",
    },
    "chart.entity_self": {
        "t": "named things with a proven wellness link",
        "w": "The count at the minutes level and at the hours level; -1 means too "
             "few windows have passed for even a perfect link to count, not 'none'.",
        "h": "mimo records' entity_self l2_counted and l3_counted",
    },
    "chart.wellness_align": {
        "t": "Error in the learned wellness predictions",
        "w": "Subtract each learned wellness estimate from its matching fixed score, take the absolute difference, then average across the five copied scores and bodies. The chart multiplies that mean by 100. Zero means exact agreement; a reading of 10 means an average error of ten percentage points on the score scale. Self-touch is not copied.",
        "h": "mimo records: wellness_align_err, multiplied by 100.",
        "f": "This reporting measure uses absolute differences. The training loss uses squared differences instead. [[approximate-evaluator|The learned predictor’s inputs, outputs and training]].",
        "h_label": "Source and calculation details",
        "f_label": "Limits and settings"
    },
    "chart.prediction_audit": {
        "t": "next-model-step prediction against 'nothing changes' (the paired audit)",
        "w": "Per sense, as a percentage of the way from [[persistence]] (0% = no "
             "better than nothing changes) to the exact next token (100%); below "
             "0% = worse than nothing changes. Measured online, before the update, "
             "on the same model steps for both.",
        "h": "mimo records' prediction_audit.modalities.<sense>."
             "improvement_over_persistence x 100",
    },
    "chart.order_sensitivity": {
        "t": "how much the prediction moves when the command is swapped",
        "w": "Per sense: the change of the predicted next token when another "
             "body's absolute command (its targets as they stand) replaces this "
             "body's, as a percentage of the change it predicts from the present "
             "(0% = the swapped command moves nothing, 100% = as much as the whole "
             "predicted change). The original control, kept for one segment beside "
             "the size-matched one (the swap-test chart).",
        "h": "mimo records' prediction_audit.modalities.<sense>.sensitivity x 100 "
             "(mimo_bridge/prediction_audit.py, 2026-09-22); the size-matched "
             "control's is sensitivity_size_matched",
    },
    "chart.swap_test": {
        "t": "the swap test: own command against another body's size-matched command",
        "w": "Per sense: 1 minus (the next-model-step prediction error with the "
             "body's own executed command) / (the error with another body's "
             "command size-matched to this body in its place: that body's asked "
             "moves and turns, measured from its own pose, applied to this body's "
             "pose), summed over the window's graded transitions, as a percentage: "
             "0% = the control predicts as well (the command is not used), 100% = "
             "the own command predicts exactly, below 0% = the control predicts "
             "better. One line per sense on model steps that are multiples of 20 "
             "(with --motor-hold 5, the last model step of a hold) and one on the "
             "hold's first model step, where the command is new and the context "
             "still holds the previous one.",
        "h": "mimo records' prediction_audit.modalities.<sense>."
             "improvement_over_size_matched and prediction_audit_hold_first.senses."
             "<sense>.all_bodies.improvement_over_size_matched, x 100 "
             "(mimo_bridge/prediction_audit.py, 2026-10-02)",
        "f": "reading 0% as proof that the world ignores the command: the mean "
             "square of the difference between two candidate commands' real next "
             "tokens is 47 to 53 of every 100 parts of the tactile prediction error "
             "and 93 to 102 of proprioception's (checkpoints 437,125 and 450,525; "
             "docs/gain_of_function_v4/coordinator/own_command_analysis_2026-10-02"
             ".html), so 0% says the prediction does not carry the command's effect",
    },
    "chart.skin_regions": {
        "t": "where the bodies are touched",
        "w": "Out of every 100 parts of the contact force, how many land on each "
             "skin region, averaged over the bodies and the window (one body's "
             "regions add to 100).",
        "h": "mimo records' skin_regions.<region>.force_share x 100 (per-body "
             "window means since 2026-09-22; before, the last body at the window's "
             "last step)",
    },
    "chart.world_events": {
        "t": "world events",
        "w": "Out of every 100 model steps of a body, how many had: the hand in view, the "
             "hand in the fovea, the mobile touched by any part, by a hand, the "
             "ball touched, a hand pressing on anything.",
        "h": "mimo records' task_hand_seen, task_hand_fov, task_mobile_hit, "
             "task_mobile_hand, task_ball_touch_rate, task_pressing, x 100",
    },
    "chart.machine_memory": {
        "t": "the machine: memory",
        "w": "Commit charge against its limit, RAM still available and the "
             "trainer's own resident memory, in GiB (1,073,741,824 bytes per GiB). On 2026-09-21 the commit charge "
             "ran out beside the live trainer and its supervisor died "
             "(docs/gain_of_function_v3/WORKLOG.md).",
        "h": "mimo records' machine block (mimo_bridge/machine_stats.py)",
    },
    "chart.machine_time": {
        "t": "the machine: time per step",
        "w": "Milliseconds per model step, split into the model's part, the "
             "world's part and the causal chain's part. Segment 31 ran 1.25 model "
             "steps a second in its first hour and 0.40 model steps a second when "
             "the watcher stopped it at 17:15, after a break at 14:16 with every "
             "phase slower, and nothing recorded why.",
        "h": "mimo records' machine step_ms, model_ms, env_ms, causal_ms",
    },
    "chart.synergy": {
        "t": "motor synergy: does the body follow its commands?",
        "w": "Item 5's measurement is how well the body follows the commands the synergy "
             "writes: the direction match between each graded command and the body's "
             "250 ms turn, drawn between 'exactly as commanded', 'unrelated' and "
             "'exactly against' (an averaged cosine: not yet measured as a count), by "
             "author when the log splits it. Beside it, not instead of it: of the "
             "posture changes the body made, the percentage the synergy reproduced "
             "through its few numbers before learning from them (0% = no better than "
             "asking for no change, 100% = the whole change).",
        "h": "mimo records' synergy following_mean (and following_by_author when "
             "present), explained_share and explained_share_unweighted x 100 "
             "(lem_model.synergy_window_summary)",
        "f": "reading the explained share as control: a network that reproduces the "
             "average posture change explains a large share while controlling nothing",
    },
    "chart.liveness": {
        "t": "response of sampled network layers",
        "w": "Out of every 100 hidden units of each layer, how many left the flat low end of their "
             "activation function (GELU) on the window's samples. In that flat end a change in a "
             "unit's input moves its output by less than 1.2% of the change, and the lesson's "
             "correction reaches it shrunk by the same share (0 = no unit crossed the response threshold "
             "on these samples; 100 = every unit did), beside DESIGN.md section 10's "
             "line of 50. And for each attention pool, how evenly it spreads over the elements it "
             "reads, drawn between 'reads one element' and 'spreads evenly' - an entropy, which "
             "this page does not print as a number.",
        "h": "mimo records' liveness block: layers, pools, line, samples, revived (the field "
             "contract of 2026-09-23; the model lane writes it)",
        "f": "reading a live layer as a working one: live units can still carry nothing the "
             "lesson needs - the probes are the test of that",
    },
    "chart.next_frame_norm": {
        "t": "next-frame loss weights, per sense",
        "w": "Under --next-frame-norm persistence, each sense's next-frame error "
             "is divided by its own running persistence error; the line is the "
             "weight that division gives each sense, as a percentage of the plain "
             "lesson's weight (100% = unchanged).",
        "h": "mimo records' next_frame_norm weight_effective x 100 (DESIGN.md D12)",
    },
}


# ------------------------------------------------------------------ run discovery
def run_dirs() -> List[str]:
    pats = [os.path.join(SAVED, "experiments", "*"), os.path.join(SAVED, "runs", "*")]
    out = []
    for p in pats:
        for d in glob.glob(p):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "args.json")):
                out.append(d)
    return out


def newest_run() -> Optional[str]:
    best, best_t = None, -1.0
    for d in run_dirs():
        s = os.path.join(d, "replay_training_stats.jsonl")
        t = os.path.getmtime(s) if os.path.exists(s) else 0.0
        if t > best_t:
            best, best_t = d, t
    return best


def _load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def find_receipt(run: str, manifests=None):
    """(receipt, note): the newest checkpoint's model receipt, and a sentence the page prints when
    it was not read from the run folder, or when there is none (None when the run folder had it).

    tools/archive_checkpoints.py moves checkpoints off C: to D: and writes a manifest row per move
    (mimo_bridge/storage_locations.py reads them back). After it ran, the run folder's checkpoints/
    held no receipt, and the page drew 35 of its 52 boxes, no efference box among them, with no word
    on why, while the browser check read 0 problems (review of 2026-09-25, finding 14). The receipt is
    now read where the manifest says the checkpoint went, proven against the hash the manifest
    recorded (storage_locations.locate), and a page without one says so where the diagram starts."""
    local = sorted(glob.glob(os.path.join(run, "checkpoints", "ckpt_step_*", "model_receipt.json")),
                   key=lambda p: int(p.split("ckpt_step_")[1].split(os.sep)[0]))
    for path in reversed(local):
        receipt = _load(path)
        if isinstance(receipt, dict) and receipt:
            return receipt, None
    # THE MANIFESTS ARE READ ONLY WHEN THE RUN FOLDER HAS NO RECEIPT: the archiver moves a run's older
    # checkpoints and a live run keeps writing new ones, so a receipt in the run folder is the newest,
    # and a page build never waits on the archive drive for a run that has its own
    from mimo_bridge import storage_locations as SL
    steps = {}
    ck_dir = SL._key(os.path.join(run, "checkpoints"))
    for manifest in (SL.default_manifests() if manifests is None else manifests):
        for key, move in SL.moves_in(manifest).items():
            m = re.fullmatch(r"ckpt_step_(\d+)", os.path.basename(key))
            if m and move.get("kind") == "folder" and os.path.dirname(key) == ck_dir:
                steps.setdefault(int(m.group(1)), os.path.join(run, "checkpoints", os.path.basename(
                    move["source"]), "model_receipt.json"))
    refused = []
    for step in sorted(steps, reverse=True):
        try:
            found = str(SL.locate(steps[step], manifests=manifests))
        except SL.MovedFileError as exc:
            refused.append(str(exc))
            continue
        receipt = _load(found) if os.path.exists(found) else None
        if not isinstance(receipt, dict) or not receipt:
            continue
        return receipt, (f"The model receipt was read from {found}: the checkpoint archive moved "
                         f"checkpoint {step:,} there, and the run folder holds no checkpoint of its own.")
    return {}, ("No model receipt: the run folder holds no checkpoint with a model_receipt.json"
                + (" and no storage manifest names a moved one" if not refused else
                   ", and the moved ones could not be proven (" + "; ".join(refused[:2]) + ")")
                + ", so the diagram draws only what the launch flags name, not the parts the model "
                  "was built with.")


def newest_receipt(run: str) -> Dict[str, Any]:
    return find_receipt(run)[0]


# Parameter tensors the optimizer holds that NO loss term reaches, decided by the
# two launch flags the receipt records (P204). forecast() returns GroundedDynamics before
# the legacy head under 'grounded_residual', and percept['bias'] is consumed only under
# self_state_mode == 'perceive', so 'learn' never grades residual. Measured on H83
# ckpt_step_4097248: present_loss + learning_loss at horizons 1/5/20/100 leave 262,022 of
# 912,143 with grad None, split 162,950 + 99,072.
# The fifth field says how the group was CREATED, which decides what a checkpoint can show
# about it: a zero-created group that no gradient reaches must still be exactly zero; a
# randomly created one must be bit for bit what it was at an earlier checkpoint. Whether
# either holds is MEASURED on the run's own checkpoints at build time (measure_idle_groups)
# and never typed: "still exactly zero after 4,097,248 steps" was H83's measurement and
# was printed on every later run's page.
SELF_STATE_IDLE = [
    ("self_prediction", "grounded_residual",
     ("action_encoder.", "transition.", "future."),
     "the retired legacy forecast head - kept so older checkpoints load, "
     "never executed while --self-prediction is grounded_residual", "random", "legacy_head"),
    ("self_state", "learn", ("residual.",),
     "the belief-into-perception bias - computed every model step and discarded "
     "while --self-state is learn", "zero", "belief_bias"),
]


def _ckpt_files(run: str, name: str) -> List[str]:
    return sorted(glob.glob(os.path.join(run, "checkpoints", "ckpt_step_*", name)),
                  key=lambda p: int(p.split("ckpt_step_")[1].split(os.sep)[0]))


def _ckpt_step(path: str) -> int:
    return int(path.split("ckpt_step_")[1].split(os.sep)[0])


def measure_idle_groups(run: str, receipt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What the run's OWN checkpoints show about the weights no lesson reaches.

    For each idle group: how many of its numbers are nonzero in the newest checkpoint, and
    how many differ from the run folder's first checkpoint. safetensors' safe_open reads
    only the tensors asked for. Measured 2026-09-19 on H84's 58 MB checkpoints: 9 to 14 ms
    for both groups and both checkpoints, against a page build of about 1.5 s - so it runs
    on every build, the 10-minute watch included. None when the run folder holds no
    checkpoint or the file cannot be read; the page then says unmeasured."""
    files = _ckpt_files(run, "model.safetensors")
    if not files:
        return None
    wanted = [(gid, prefixes) for key, mode, prefixes, _w, _i, gid in SELF_STATE_IDLE
              if receipt.get(key) == mode]
    if not wanted:
        return None
    try:
        import numpy as np
        from safetensors import safe_open
        t0 = time.perf_counter()

        def read(path: str, prefixes) -> Dict[str, Any]:
            with safe_open(path, framework="np") as f:
                return {k: f.get_tensor(k) for k in f.keys()
                        if any(k.startswith("self_state." + p) for p in prefixes)}

        newest, first = files[-1], files[0]
        out: Dict[str, Any] = {"step": _ckpt_step(newest),
                               "first_step": (_ckpt_step(first) if first != newest else None),
                               "groups": {}}
        for gid, prefixes in wanted:
            new = read(newest, prefixes)
            if not new:
                continue
            g = {"elements": int(sum(v.size for v in new.values())),
                 "nonzero": int(sum(int(np.count_nonzero(v)) for v in new.values())),
                 "differ_from_first": None}
            if first != newest:
                old = read(first, prefixes)
                if sorted(old) == sorted(new) and all(old[k].shape == new[k].shape for k in new):
                    g["differ_from_first"] = int(sum(
                        int(np.count_nonzero(old[k] != new[k])) for k in new))
            out["groups"][gid] = g
        out["read_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return out if out["groups"] else None
    except Exception as exc:                 # an unreadable checkpoint must not stop the page
        log.warning("idle-weight measurement skipped: %s", exc)
        return None


def landing_token_estimate(run: str) -> Optional[Dict[str, Any]]:
    """An ESTIMATE of the size of the token the recalled memory bias lands on, for a page
    whose snapshots do not record it (every snapshot taken before 2026-09-19).

    That token is the final LayerNorm's output, g * z + b, where z has mean 0 and variance
    1 over the token's numbers. If z is unrelated to g and b from number to number, the
    token's RMS is sqrt(mean(g^2) + mean(b^2)) - read here from the newest checkpoint's two
    final_norm tensors. The 'if' is an assumption, so the page calls this an estimate and
    never a measurement. Checked 2026-09-19 on the one run where the measurement exists:
    0.589 estimated on the H81 probe's checkpoint against 0.59 measured. On H84 it gives
    2.59 while the snapshots measure the bias at 3.01 to 4.03."""
    files = _ckpt_files(run, "model.safetensors")
    if not files:
        return None
    try:
        import numpy as np
        from safetensors import safe_open
        with safe_open(files[-1], framework="np") as f:
            names = set(f.keys())
            gk, bk = "forward_model.final_norm.weight", "forward_model.final_norm.bias"
            if gk not in names or bk not in names:
                return None
            g = f.get_tensor(gk).astype("float64")
            b = f.get_tensor(bk).astype("float64")
        return {"step": _ckpt_step(files[-1]),
                "gain_rms": round(float(np.sqrt((g ** 2).mean())), 4),
                "bias_rms": round(float(np.sqrt((b ** 2).mean())), 4),
                "estimate": round(float(np.sqrt((g ** 2).mean() + (b ** 2).mean())), 4)}
    except Exception as exc:
        log.warning("landing-token estimate skipped: %s", exc)
        return None


def _idle_measured_text(init: str, g: Optional[Dict[str, Any]], m: Optional[Dict[str, Any]]) -> str:
    """One group's measurement as a clause, or the word unmeasured with the reason."""
    if not m or not g:
        return "unmeasured: the run folder held no readable checkpoint when this page was built"
    n, step = g["elements"], m["step"]
    if init == "zero":
        if g["nonzero"] == 0:
            return (f"measured on the checkpoint at step {step:,}: 0 of those {n:,} numbers "
                    f"are nonzero, so they still hold the zero they were created with")
        return (f"measured on the checkpoint at step {step:,}: {g['nonzero']:,} of those "
                f"{n:,} numbers are nonzero, so something has moved them off the zero they "
                f"were created with")
    if g["differ_from_first"] is None:
        return (f"unmeasured: the run folder holds one checkpoint (step {step:,}), so there "
                f"is no earlier one to compare it with")
    if g["differ_from_first"] == 0:
        return (f"measured on the checkpoints at steps {m['first_step']:,} and {step:,}: 0 of "
                f"those {n:,} numbers differ between the two")
    return (f"measured on the checkpoints at steps {m['first_step']:,} and {step:,}: "
            f"{g['differ_from_first']:,} of those {n:,} numbers differ between the two, so "
            f"something has moved them")


def idle_params(receipt: Dict[str, Any],
                measured: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Weights in the optimizer that no lesson reaches in THIS configuration.
    {'self_state': {'elements': n, 'groups': [{'elements': n, 'why': str,
                                               'measured': str}, ...]}}
    'measured' is what the run's own checkpoints show (measure_idle_groups), or a sentence
    saying it is unmeasured and why."""
    tens = (((receipt.get("checkpoint_inventory") or {}).get("components") or {})
            .get("self_state") or {}).get("tensors") or {}
    groups = []
    for key, mode, prefixes, why, init, gid in SELF_STATE_IDLE:
        if receipt.get(key) != mode:
            continue
        n = sum(v.get("elements", 0) for k, v in tens.items()
                if v.get("kind") != "buffer" and any(k.startswith(p) for p in prefixes))
        if n:
            groups.append({"elements": n, "why": why, "id": gid,
                           "measured": _idle_measured_text(
                               init, ((measured or {}).get("groups") or {}).get(gid), measured)})
    if not groups:
        return {}
    return {"self_state": {"elements": sum(g["elements"] for g in groups), "groups": groups}}


def idle_fill(receipt: Dict[str, Any], idle: Dict[str, Any], run_name: str) -> Dict[str, str]:
    """The {self_state_idle} sentence of node.self_state, for one run: this run's own
    counts from its receipt, and what its own checkpoints show about them."""
    total = ((((receipt.get("checkpoint_inventory") or {}).get("components") or {})
              .get("self_state") or {}).get("parameters")
             or (receipt.get("component_params") or {}).get("self_state"))
    ss = (idle or {}).get("self_state")
    if not ss or not total:
        return {"{self_state_idle}": (
            f"How many of its weights no lesson reaches is not stated here: the model "
            f"receipt of {run_name} lists none under its launch flags, or carries no "
            f"per-tensor inventory to count them from.")}
    parts = " ".join(f"{g['elements']:,} are {g['why']}; {g['measured']}."
                     for g in ss["groups"])
    return {"{self_state_idle}": (
        f"{ss['elements']:,} of its {int(total):,} weights receive no gradient under the "
        f"flags {run_name} was launched with. {parts} {int(total) - ss['elements']:,} "
        f"weights are reached and trained (audit P204).")}


def exec_source_summary(mimo: List[dict]) -> Optional[Dict[str, Any]]:
    """Who wrote the orders the body executed, from the trainer's own tally at the latch
    (exec_source: a share per author per report window). Windows after the log's last
    restart only - the rule the verdict tiles use - because one log can hold more than
    one configuration. None when the log carries no tally."""
    seams = resume_seams(mimo)
    cut = seams[-1] if seams else None
    recs = [r for r in mimo if isinstance(r.get("exec_source"), dict)
            and r.get("step") is not None and (cut is None or int(r["step"]) >= cut)]
    if not recs:
        return None
    order = ["planner", "self_choice", "reflex_noise", "reflex"]
    keys = [k for k in order if any(k in r["exec_source"] for r in recs)]
    keys += sorted({k for r in recs for k in r["exec_source"]} - set(keys))
    shares = {}
    for k in keys:
        vals = [float(r["exec_source"].get(k) or 0.0) for r in recs]
        shares[k] = {"mean": round(sum(vals) / len(vals), 4),
                     "min": round(min(vals), 4), "max": round(max(vals), 4)}
    return {"windows": len(recs), "first_step": int(recs[0]["step"]),
            "last_step": int(recs[-1]["step"]), "after_restart_at": cut, "shares": shares}


def component_params(run: str, receipt: Dict[str, Any]):
    """Per-component parameter counts and their basis. Prefer the receipt's
    learned-only counts (newer checkpoints carry them); fall back to counting
    the newest checkpoint's saved tensors by name prefix (includes frozen
    calibration buffers - the basis string says which was used)."""
    if receipt.get("component_params"):
        return receipt["component_params"], "(learned weights, from the model receipt)"
    cands = sorted(glob.glob(os.path.join(run, "checkpoints", "ckpt_step_*",
                                          "model.safetensors")),
                   key=lambda p: int(p.split("ckpt_step_")[1].split(os.sep)[0]))
    if not cands:
        return {}, ""
    try:
        from safetensors import safe_open
        counts: Dict[str, int] = {}
        with safe_open(cands[-1], framework="np") as f:
            for k in f.keys():
                n = 1
                for s in f.get_slice(k).get_shape():
                    n *= s
                parts = k.split(".")
                top = ("enc_" + parts[2]
                       if parts[0] == "encoder_registry" and len(parts) > 2
                       and parts[1] == "encoders" else parts[0])
                counts[top] = counts.get(top, 0) + n
        return counts, "(saved numbers, buffers the optimizer never trains included)"
    except Exception:
        return {}, ""


# ------------------------------------------------------------------ stats -> series
def read_stats(run: str) -> Dict[str, List[dict]]:
    path = os.path.join(run, "replay_training_stats.jsonl")
    out: Dict[str, List[dict]] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                out.setdefault(r.get("type", "?"), []).append(r)
    except FileNotFoundError:
        pass
    return out


def resume_seams(mimo: List[dict]) -> List[int]:
    """ALL restart steps, not just the last one (the dashboard's split_at_resume
    truncates to the final segment - not what a whole-run tab needs). A restart is
    where elapsed_s drops (elapsed counts from process start)."""
    seams = []
    for i in range(1, len(mimo)):
        if (mimo[i].get("elapsed_s") or 0) < (mimo[i - 1].get("elapsed_s") or 0):
            seams.append(int(mimo[i].get("step") or 0))
    return seams


def _get(rec: dict, path: str):
    cur: Any = rec
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur if isinstance(cur, (int, float)) else None


def _plot_value(v) -> Optional[float]:
    """One plotted number, kept to 6 SIGNIFICANT figures; None when it is not finite.

    round(v, 6) kept 6 DECIMAL places whatever the series' own size, so a series
    living near 1e-06 was cut to one or two plotted levels and anything under 5e-07
    was drawn as exactly 0.0. Measured on H84's 679 report windows: tenure_gap.pose -
    the series chart 2's trend is computed from - went from 678 distinct values to 5,
    and 674 of its 679 nonzero values were drawn as 0.0. Six significant figures keep
    678 of 678 and cost 4.49% more bytes in the series block (678,829 -> 709,280).

    A value that is not finite is dropped: json.dumps writes it as the bare word NaN,
    which is not JSON, so the browser's JSON.parse rejects the WHOLE series block and
    the run tab reports "No stats series found" for a log that is full of records.
    """
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return float(f"{v:.6g}")


def _quietest_body(sense: str):
    """A series function: the lowest per-body collapse-watch reading of `sense` in a record
    (token_temporal_std_bodies), or None where the record has none."""
    def f(r):
        vals = ((r.get("token_temporal_std_bodies") or {}).get(sense)) or []
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        return min(vals) if vals else None
    f.field = "token_temporal_std_bodies.%s.min" % sense
    return f


def _series(recs: List[dict], path, label: str) -> Optional[dict]:
    """One plotted line. `path` is a dotted field name, or a function of a record.

    The function form exists so a chart can plot a line that is DERIVED rather
    than logged - specifically the chance line a score must be read against. A
    number with no line to beat is not interpretable, and putting the raw count
    that sets that line on the same axis squashes the score instead (see the
    headline chart below).
    """
    pts = []
    for r in recs:
        v = path(r) if callable(path) else _get(r, path)
        # THE X AXIS IS NOT ALWAYS CALLED "step". The live-step loss records key
        # theirs as training_step, so requiring "step" here returned no series at
        # all for the "Loss terms (live step, window mean)" chart: the chart was
        # declared and never reached the page (15 charts drawn of 16 declared).
        # Measured on H82's 456 records: where a line carries both fields they
        # hold the SAME number, so the two axes line up.
        step = r.get("step", r.get("training_step"))
        v = None if v is None else _plot_value(v)
        if v is not None and step is not None:
            pts.append([int(step), v])
    if len(pts) < 2:
        return None
    pts = _thin(pts)
    return {"label": label, "points": pts,
            "key": getattr(path, "field", label) if callable(path) else path}


def _thin(pts: List[list]) -> List[list]:
    """At most SERIES_POINT_CAP points, taken at an even stride across the WHOLE run,
    with the newest point always kept. A plain pts[:SERIES_POINT_CAP] keeps the first
    1,200 records and drops every later one, so on a long log the line stops short of
    the present while its neighbours on the same chart run to the end."""
    if len(pts) > SERIES_POINT_CAP:
        stride = -(-len(pts) // SERIES_POINT_CAP)
        pts = pts[::stride] + [pts[-1]]
    return pts


def _chance_from_decoys(record):
    """What guessing scores on the matching test: 1/(1+decoys).

    Read from the run's own logged decoy count per window, never assumed, so a
    run that changes its decoy count moves this line with it.
    """
    n = _get(record, "tenure_tok.n_decoys")
    if n is None:
        return None
    try:
        n = float(n)
    except (TypeError, ValueError):
        return None
    return 1.0 / (1.0 + n) if n >= 0 else None


# ITS OWN KEY. The series holds 1/(1+decoys), not the decoy count, so it may not be
# published under "tenure_tok.n_decoys": viewer.js looked that key up for headline
# tile 1 and computed 1/(1 + 0.0667) = 0.94, printing "out of ~0 decoys. Guessing
# would score 94%" and marking a run at rank1 0.20 against a true chance of 0.067 as
# "at or below guessing" (measured on H84's 679 windows).
_chance_from_decoys.field = "tenure_tok.chance"


def _deg(path: str, field: str):
    """A plotted line in DEGREES of a field the log stores in radians: the page shows degrees,
    never radians alone (the writing rules; the survey's stale item 10)."""
    def f(r):
        v = _get(r, path)
        return None if v is None else float(v) * 180.0 / 3.141592653589793
    f.field = field
    return f


def _ratio(num: str, den: str, field: str, factor: float = 1.0):
    """factor x num / den of one record, None when either is missing or den is not above 0
    (factor 100 draws the ratio in per cent)."""
    def f(r):
        a, b = _get(r, num), _get(r, den)
        return None if a is None or b is None or not b > 0 else factor * float(a) / float(b)
    f.field = field
    return f


def _share_removed(r):
    """Of the pose error a do-nothing order leaves at an order's end, the share the executed order
    removes: tenure_gap.pose / (tenure_pe.pose + tenure_gap.pose), because the gap is the do-nothing
    error minus the with-order error. 0 = the order explains nothing, 1 = exact, below 0 = worse.
    The raw error and gap are unitless internals the page does not print (DESIGN.md section 11)."""
    g, e = _get(r, "tenure_gap.pose"), _get(r, "tenure_pe.pose")
    if g is None or e is None or not (float(g) + float(e)) > 0:
        return None
    return float(g) / (float(g) + float(e))


_share_removed.field = "tenure_gap.share_removed"


def _age_blind(r):
    """The goal-age chart's reference line: a choice that ignores a moment's age puts 10 of every
    100 goals in each tenth of the ring. Drawn only where the record carries goal_age."""
    return 10.0 if isinstance((r.get("wellness_rank") or {}).get("goal_age"), dict) else None


_age_blind.field = "wellness_rank.goal_age.age_blind_per100"


def _half_line(r):
    """DESIGN.md section 10's live-share line, 50 of every 100 units, where the record carries the
    whole-model liveness block."""
    return 50.0 if isinstance((r.get("liveness") or {}).get("layers"), dict) else None


_half_line.field = "liveness.line_per100"


def _loss_share(term: str):
    """One lesson's share of the window's POSITIVE lessons added up (training records'
    loss_breakdown), 0 to 1. A loss has no lived unit and the terms share no scale, so the page
    draws each as its part of the positive total; the self-perception likelihood is negative and
    stays out of that total (DESIGN.md section 11)."""
    def f(r):
        bd = r.get("loss_breakdown")
        if not isinstance(bd, dict) or not isinstance(bd.get(term), (int, float)):
            return None
        # Synergy contributes to the summed loss but trains only its own map on detached data.
        tot = sum(float(v) for k, v in bd.items()
                  if k != "synergy" and isinstance(v, (int, float)) and v > 0)
        return float(bd[term]) / tot if tot > 0 else None
    f.field = "loss_breakdown." + term + ".share"
    return f


def _liveness_value(block: str, key: str):
    """A liveness reading by its full key. The keys are dotted code paths
    (encoder_registry.encoders.motor.patch_mlp.0), which the dotted-field lookup _get splits at every
    dot: through 2026-09-25 the layers chart drew only its 50 line on every production page, and the
    pools chart was never drawn."""
    def f(r):
        v = ((r.get("liveness") or {}).get(block) or {}).get(key)
        return v if isinstance(v, (int, float)) else None
    f.field = f"liveness.{block}[{key}]"
    return f


def _scaled(src, factor: float, field: str, lo: float = 0.0):
    """factor x (value - lo) of a field or of a derived line: a share drawn per 100, a ratio in
    per cent, a bounded score as a percentage of the way between its named ends. DESIGN.md
    section 11 (operator 2026-09-22): a person reads counts, real units or a percentage between
    named ends, never a bare score decimal."""
    def f(r):
        v = src(r) if callable(src) else _get(r, src)
        return None if v is None else (float(v) - lo) * factor
    f.field = field
    return f


# the body map's belonging scale runs from sigmoid(T x (0 - 0.5)) to sigmoid(T x (1 - 0.5)) at the
# map's temperature: its floor (no evidence) and its top (carried as surely as a limb), pinned to
# the map's own temperature by test_explainer_capture's incorporation test
BELONG_FLOOR, BELONG_TOP = 0.0759, 0.9241


def _idx(path: str, i: int, field: str):
    """Element i of a list the log stores under `path` (the efference gaze horizons)."""
    def f(r):
        cur: Any = r
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        if isinstance(cur, list) and i < len(cur) and isinstance(cur[i], (int, float)):
            return float(cur[i])
        return None
    f.field = field
    return f


def _chance_of_alternatives(record):
    """What picking at random among the chosen plan and its alternatives scores: 1/(1+n)."""
    n = _get(record, "causal_chain.alternatives_n")
    return None if n is None or n < 0 else 1.0 / (1.0 + float(n))


_chance_of_alternatives.field = "causal_chain.chance"


def _keys_under(recs: List[dict], path: str, cap: int = 12) -> List[str]:
    """The keys a dict-valued field carries anywhere in the log, first seen first (a channel
    list, a region list, a sense list) - never a typed list, so a run's own names are drawn."""
    seen: List[str] = []
    for r in recs:
        cur: Any = r
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            for k in cur:
                if k not in seen:
                    seen.append(k)
    return seen[:cap]


# the per-sense record families the run tab charts
PER_SENSE_FAMILIES =("causal_pe_per_modality", "token_temporal_std", "recon_dec",
                      "pe", "td_gate")
# DRAWING ORDER ONLY - never membership. A sense keeps the same place, and so the same
# line colour, on a page built before the run's first receipt and on one built after.
SENSE_DRAW_ORDER = ("vision", "tactile", "audio", "proprioception", "motor", "intermind",
                    "vestibular")


def _senses_in_log(mimo: List[dict]) -> List[str]:
    """Every sense any per-sense family of the log names. None is left out: a name
    SENSE_DRAW_ORDER does not know is kept and sorted after the ones it does."""
    seen: List[str] = []
    for r in mimo:
        for fam in PER_SENSE_FAMILIES:
            block = r.get(fam)
            if isinstance(block, dict):
                for k in block:
                    if k not in seen:
                        seen.append(k)
    return sorted(seen, key=lambda m: (SENSE_DRAW_ORDER.index(m) if m in SENSE_DRAW_ORDER
                                       else len(SENSE_DRAW_ORDER), m))


def build_series(stats: Dict[str, List[dict]], snap_steps: List[int],
                 modalities: Optional[List[str]] = None,
                 sources: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    mimo = stats.get("mimo", [])
    training = stats.get("training", [])
    launches = sorted(int(r.get("step") or 0) for r in stats.get("launch", []))
    dropped: List[str] = []          # charts declared here that no record could fill
    off: List[str] = []              # charts whose feature block no record carries at all

    def chart(label, log_y, specs, recs=mimo, doc=None, caption=None, trend=None,
              requires=None, unit=None, scale=None, ends=None, relative=False, integer=False):
        # Every chart names the exact field it reads in a popup the reader can open
        # - the page's own footer promises that, and charts shipped without one
        # broke the promise silently. Required here rather than in the signature so
        # the failure names the chart.
        if not doc or doc not in DOC_CHARTS:
            raise ValueError(f"chart {label!r} needs doc= naming an entry in DOC_CHARTS")
        # DESIGN.md section 11: every chart says how a person reads its numbers - a unit printed on
        # every tick and in the hover readout (a count per 100, a percentage between named ends, a
        # real unit), or, for a raw internal with no count form yet, its named ends and no numeric
        # tick at all. The series keep the log's own values; `scale` and `relative` are the view
        # (viewer.js chartView), so the verdict tiles and the trend chips read the logged numbers.
        if not unit and not ends:
            raise ValueError(f"chart {label!r} needs unit= or ends= (DESIGN.md section 11)")
        series = [s for p, lab in specs if (s := _series(recs, p, lab))]
        if not series and requires and not any(
                (requires(r) if callable(requires) else requires in r) for r in recs):
            # THE FEATURE WAS OFF, not the data thin: no record of this log carries the
            # block the chart reads (a run launched before the flag existed, or with it off)
            off.append(label)
            return None
        if not series:
            # NEVER SILENTLY. A declared chart that no record can fill used to vanish
            # with nothing said anywhere - which is how "Loss terms" stayed missing
            # from every page while its records sat in the log under another x name.
            dropped.append(label)
            return None
        out = {"label": label, "log": log_y, "doc": doc, "series": series,
               "caption": caption, "trend": trend}
        if unit:
            # the y axis's tick labels carry it (DESIGN.md section 11: a number a person reads
            # carries its real unit, or is a count per 100, or a percentage)
            out["unit"] = unit
        if scale:
            out["scale"] = scale
        if ends:
            out["ends"] = [[float(v), str(name)] for v, name in ends]
        if relative:
            out["relative"] = True
        if integer:
            out["integer"] = True
        return out

    def _exec_frac(r):
        """Share of this process segment's steps whose executed order was handed
        back to the model (T13): contract.exec_seen over the steps since the last
        launch record at or before this one."""
        c = r.get("contract") or {}
        if "exec_seen" not in c:
            return None
        step = int(c.get("step") or r.get("step") or 0)
        base = 0
        for ls in launches:
            if ls <= step:
                base = ls
        return min(1.0, c["exec_seen"] / max(1, step - base))

    _exec_frac.field = "exec_frac"

    # THE RUN'S OWN SENSES, from the model receipt - never a literal list. The
    # literal ("vision", "tactile", "proprioception") hid the two senses that
    # mattered most in H82: audio, the only sense whose top-down gate is open
    # (0.221641 to 0.303787 across 456 windows, against 0.000001 to 0.000003 for
    # the other three), and intermind, the quietest encoder. A sense with no
    # records for a field simply produces no line, so listing all six is safe.
    # The receipt exists only once a checkpoint does, and the first one is written
    # --checkpoint-every-min minutes after launch (60 unless the launch says
    # otherwise) - until then every run segment has no receipt, and the three-sense
    # fallback that stood here drew 3 of the 6 senses again. The log itself names
    # its senses, so the list is the receipt's order followed by any sense only the
    # log carries.
    mods = tuple(dict.fromkeys(list(modalities or []) + _senses_in_log(mimo)))
    src = sources or {}
    # ORDER = IMPORTANCE. The first charts answer the project's question - does the
    # model learn that its own orders cause what happens next - and each carries a
    # visible caption saying what movement means and which direction is good.
    charts = [
        # The decoy COUNT used to be plotted here as a third line. It is a constant
        # 14 on this run and rank1 lives in 0 to 1, so one linear axis ran to 14 and
        # the headline line this chart exists to show was squeezed into the bottom
        # 7% of the plot. What the reader actually needs from that count is the line
        # rank1 must beat - guessing scores 1/(1+decoys) - so plot THAT instead: it
        # shares rank1's scale, and the gap between the two lines is the finding.
        chart("1 · Does the command explain the outcome? (the headline)", False,
              [("tenure_tok.rank1", "rank1: command picks its OWN outcome"),
               (_chance_from_decoys, "guessing would score this"),
               ("tenure_tok.vis_gate_open", "hand-in-view gate open")], doc="chart.rank1",
              caption="rank1 = at each command's end, how often, out of every 100 command-endings, "
                      "a projection of that command picks out its own outcome among decoy "
                      "outcomes. The flat line is what guessing scores, 100 in (1 + decoys); "
                      "the distance ABOVE it is how well the model tells WHICH change was its "
                      "own doing - the signal this project exists to produce. Beating "
                      "guessing is the admission gate, not the goal; working self-recognition "
                      "would read near 100. Higher is better.",
              trend={"key": "tenure_tok.rank1", "dir": "higher"}, scale=100.0, unit="/100"),
        chart("2 · Is knowing the command worth anything?", False,
              [(_share_removed, "the do-nothing error the command removes")],
              doc="chart.tenure_gap",
              caption="At each command's end: of the pose error a do-nothing command leaves, how "
                      "much the executed command removes, in per cent. 0% = the command explains "
                      "nothing, 100% = the pose change is predicted exactly; below 0% = the "
                      "command made the prediction worse. Higher is better; the distance to "
                      "100% is how much of the change the command still leaves unexplained.",
              trend={"key": "tenure_gap.share_removed", "dir": "higher"}, unit="%", scale=100.0),
        chart("3 · How wrong about the next model step (causal error)", True,
              [("causal_pe", "all real senses")] +
              [(f"causal_pe_per_modality.{m}", m) for m in mods], doc="chart.causal",
              caption="Given the last model step AND the executed command, how far off was "
                      "the prediction of this model step - as a percentage of the error an "
                      "answer of all zeros makes (0% = exact, 100% = no better than "
                      "answering zeros, 200% = an unrelated token). This is an error: "
                      "smaller is better. Always judge it with the body-speed chart below - a "
                      "stiller body is easier to predict, so a falling error can "
                      "just be a stilling body.",
              trend={"key": "causal_pe", "dir": "error"}, scale=100.0, unit="%"),
        # THREE UNITS, THREE AXES. Speed (cm/s), a contact count and a 0-to-1 debt
        # shared one linear axis, so the axis ran to the largest of them: measured on
        # H84's 679 windows the axis ran 0 to 31.1 and stillness debt (0 to 0.002354)
        # occupied 0.01% of the plot height; on H83 it occupied 3.05%.
        chart("4 · Body speed - the context for every error", False,
              [("movement.local_speed", "local speed")], doc="chart.speed",
              caption="How fast the body's parts move relative to the body, in cm/s, "
                      "averaged over each report window. Not good or bad by itself - "
                      "it is the yardstick the error charts must be read against.",
              unit="cm/s"),
        chart("4b · Skin contacts per body", False,
              [("contacts_per_env", "contacts per body")], doc="chart.contacts",
              caption="How many skin contacts one body carries, averaged over the "
                      "bodies and over each report window. 0 = touching nothing.",
              unit="per body"),
        chart("4c · Stillness debt", False,
              [("movement.stillness_debt", "stillness debt")],
              doc="chart.stillness_debt",
              caption="How overdue the body is to move, as a percentage of the full debt: "
                      "0% = it has been moving, 100% = it has been still for 150 seconds or "
                      "more. The movement drive charges wellness by the square of this "
                      "number; --movement-penalty sets the most that charge can be.",
              unit="%", scale=100.0),
        # A COSINE IS DRAWN, NEVER PRINTED (DESIGN.md section 11): its three named ends are the
        # axis, with no numeric tick; the command size, in degrees, is a chart of its own
        chart("5 · Do the limbs obey? (command following)", False,
              [("cmd_response_cos", "direction match one step after the command"),
               ("cmd_response_cos_held", "held-command match (the fairer read)")],
              doc="chart.cmd_follow",
              caption="Where the limbs moved against where the command pointed, averaged over "
                      "each report window and drawn between three named ends: exactly as "
                      "commanded, unrelated to the command, and exactly against it. The match is "
                      "an averaged cosine: not yet measured as a count of commands.",
              trend={"key": "cmd_response_cos_held", "dir": "higher"},
              ends=[(1.0, "exactly as commanded"), (0.0, "unrelated to the command"),
                    (-1.0, "exactly against it")]),
        chart("5b · How big were the commands? (degrees of turn)", False,
              [(_deg("cmd_mag_rad", "cmd_mag_deg"),
                "command size (degrees of turn asked of each joint)")], doc="chart.cmd_follow",
              caption="Whether the commands call for real movement at all: the turn each "
                      "command sets for each joint, in degrees (H23-H28 sat near 3 degrees and "
                      "nothing moved).",
              unit="°"),
        # The chart DECLARES the series its title, popup and trend chip are about.
        # It used to declare only the fresh-vision share and have the contract share
        # spliced in afterwards by a label match: with no contract records it drew
        # fresh vision under this title, and with no fresh-vision field the whole
        # chart vanished and took 679 contract records with it (measured on H84).
        chart("6 · Does the executed command reach the model?", False,
              [(_exec_frac, "executed command reached the model")], doc="chart.exec",
              caption="Out of every 100 steps, how many handed their executed command (after "
                      "limits and noise) back to the model as the cause of what it sees; "
                      "the contract is complete at 100. A run where this is low can still "
                      "train - it just cannot learn what its actions do.",
              trend={"key": "exec_frac", "dir": "higher"}, scale=100.0, unit="/100"),
        chart("7 · Encoder collapse watch", True,
              [(f"token_temporal_std.{m}", m + (f" - {src[m]}" if src.get(m) else ""))
               for m in mods], doc="chart.collapse",
              caption="How much each sense's token varies over time, as a percentage of a "
                      "real token's size. 0% means the "
                      "encoder says the same thing every step regardless of input - "
                      "collapse. Watch for any sense sliding toward zero. Every sense "
                      "this run built is drawn, each labelled with how it arrives: a "
                      "sense labelled synthetic is fed by a placeholder generator, so "
                      "its low variation is expected and says nothing about whether "
                      "that encoder works.",
              scale=100.0, unit="%"),
        chart("7b · The quietest body, per sense", True,
              [(_quietest_body(m), m) for m in mods], doc="chart.collapse_bodies",
              requires=lambda r: "token_temporal_std_bodies" in r,
              caption="The same reading as chart 7 on every body, the lowest drawn: a body far "
                      "below its sense's live range (in the definition) has stopped reporting "
                      "that sense while the others still do.",
              scale=100.0, unit="%"),
        chart("Reconstruction decoders (the guard)", True,
              [(f"recon_dec.{m}", m) for m in mods], doc="chart.recon_dec",
              caption="Separate decoders re-describe the CURRENT model step from the "
                      "token. An error; smaller is better, and the scale is fixed: "
                      "encoder tokens are held at RMS 1.0, so each line is written as a "
                      "percentage of the error a decoder that always output zeros would make "
                      "(100% = no better than zeros, 0% = perfect). A sense sitting above "
                      "100% is being re-described worse than by silence; a sudden blow-up "
                      "against the run's own history is the damage signature.",
              trend={"key": "recon_dec.vision", "dir": "error"}, scale=100.0, unit="%"),
        chart("Predicted change (not reconstruction)", True,
              [(f"pe.{m}", m) for m in mods], doc="chart.pe",
              caption="Distance between the guess of NEXT and the present, as a percentage "
                      "of a real token's squared size (100% = a change as big as the token "
                      "itself). It GROWS "
                      "when the model predicts movement, so rising is not damage - "
                      "H34's best-ever run showed it rising. Neither direction is "
                      "good by itself.",
              scale=100.0, unit="%"),
        chart("Replay memory (re-study)", True,
              [("replay_loss", "replay loss - re-study of remembered model steps")],
              recs=mimo, doc="chart.replay",
              caption="Every few live steps the model re-studies remembered model steps "
                      "in a SECOND pass with its own loss, its own backward and its "
                      "own optimizer step. It re-teaches the forward model, the "
                      "prediction heads, the motor controller, the persistent "
                      "memory's bias readout and the MOTOR ENCODER: that window's "
                      "motor token is rebuilt from the command actually executed, with "
                      "gradients on, so all {enc_motor_params} of the motor encoder's numbers "
                      "are trained here - {enc_motor_pct} of the {total_params} numbers the "
                      "optimizer holds. The other five encoders are not touched; "
                      "their stored tokens were detached. A loss has no lived unit, so the "
                      "line is drawn as a percentage of its own first window in this log "
                      "(100% = where the log began, 0% = no error left). An error on "
                      "remembered data; smaller is better - but it is "
                      "not the live loss and the two are not comparable.",
              trend={"key": "replay_loss", "dir": "error"}, relative=True, unit="%"),
        chart("Loss terms (live step, window mean)", False,
              [(_loss_share("next_frame"), "next_frame"),
               (_loss_share("imagination"), "imagination"),
               (_loss_share("vicreg"), "vicreg"),
               (_loss_share("tenure_tok"), "tenure_tok"),
               (_loss_share("motor"), "motor")], recs=training,
              doc="chart.loss_terms",
              caption="The teaching signal split into its terms, each drawn as its share of "
                      "the window's positive lessons added up, excluding the synergy map's independent "
                      "reconstruction, out of every 100 parts. A "
                      "loss has no lived unit and the terms share no scale - one term can own "
                      "a step: on H81's step 100 the motor term carried 99.996 of every 100 "
                      "parts. The self-perception lesson is a likelihood rather than an "
                      "error, so it is negative, is left out of the positive total, and is "
                      "what carries the step's loss below zero. A term at zero taught "
                      "nothing in that window - check it is on before reasoning about "
                      "it (the H23-H28 lesson).",
              unit="/100"),
        chart("Wellness components", False,
              [(f"wellness.{k}", k) for k in
               ("prediction_error", "tactile_comfort", "capability", "coherence",
                "motor_state", "self_touch", "movement")], doc="chart.wellness",
              caption="Fixed wellness scores averaged over each report window, shown as score times 100. "
                      "Tactile comfort tops out at 85%; self-touch is limited by its configured weight. "
                      "Movement is shown separately from the six fixed scores. Raw remembered value also includes "
                      "enabled movement, agency, consequence, named-entity and curiosity contributions. "
                      "Normalized goal ranking instead combines scaled changes in its enabled channels; "
                      "neither ranking is the learned five-score prediction. Click the information button for the formulas.",
              scale=100.0, unit="%"),
        chart("Curiosity drive (mean over each report window)", False,
              [("curiosity_drive", "drive, mean over the report window")],
              doc="chart.curiosity",
              caption="The configured error-band preference: full drive at the midpoint, zero at and outside "
                      "the edges. Each point averages a report window. This does not establish learnability. "
                      "With curiosity-goal enabled, normalized ranking adds a separate channel; raw ranking "
                      "uses the selected additive or multiplicative rule. Direct and forward modes can also scale the wellness lesson.",
              scale=100.0, unit="%"),
        chart("Agency drive: what it pays", False,
              [("agency.pay", "payment"), ("agency.prior_w", "hand-watching prior")],
              doc="chart.agency",
              caption="Evidence of your own effect pays wellness, drawn as a percentage of "
                      "one evaluator's 0-to-1 range: payment follows match wins; the "
                      "hand-watching prior fades as vision's own evidence catches up to "
                      "touch's.",
              scale=100.0, unit="%"),
        chart("Agency drive: whose evidence (own outcome against the decoys)", False,
              [("agency.vis_adv", "vision advantage"), ("agency.touch_adv", "touch advantage")],
              doc="chart.agency",
              caption="How much closer each command's projection sits to its OWN outcome than "
                      "to the decoys', per sense, drawn against 'no better than a decoy'. "
                      "The advantage is a difference of cosines: not yet measured as a "
                      "count.",
              ends=[(0.0, "no better than a decoy")]),
        chart("Top-down gate (per sense blend)", False,
              [(f"td_gate.{m}", m) for m in mods], doc="chart.td_gate",
              caption="How much of the model's own expectation is blended into each "
                      "sense before the transformer reads it, in parts of the token out of "
                      "100 (0 = raw reality only, capped at 50). The senses do not sit "
                      "together, so read each line's own level rather than assuming the gate "
                      "is shut: across H82's 456 windows audio ran 22 to 30 in 100 while "
                      "vision, tactile and proprioception stayed under 1 part in 100,000. "
                      "Learning is always graded on raw reality regardless, so an "
                      "open gate changes what the model ACTS on, never what it is "
                      "graded against.",
              scale=100.0, unit="/100"),
        chart("Throughput", False, [("model_steps_per_s", "model steps per second")],
              doc="chart.throughput",
              caption="Model steps per second - a number about the rig, not "
                      "the model.",
              unit="model steps/s"),
        chart("Fresh vision share", False,
              [("vision_fresh_fraction", "steps with a newly rendered picture")],
              doc="chart.vision_fresh",
              caption="Out of every 100 steps in each report window, how many had their eyes "
                      "rendered that step: 100 = every step read a new picture, 0 = "
                      "every step re-used the previous vision token. It says whether "
                      "vision was fresh, and nothing about the executed command.",
              scale=100.0, unit="/100"),
    ]
    # ---------------- the gain-of-function build (2026-09-22) and the fields no chart drew
    # before it. Each chart names the block it needs: a run launched without that feature
    # lists the chart under "feature off" instead of drawing an empty axis.
    # the senses the log's own per-sense families name, plus any the build's blocks name (a
    # log whose older families are absent still draws the causal chain's per-sense lines)
    cc_senses = list(dict.fromkeys(
        [m for m in mods if m != "motor"]
        + _keys_under(mimo, "causal_chain.pred_err_next")
        + _keys_under(mimo, "causal_chain.pred_err_latch")
        + _keys_under(mimo, "next_frame_norm.weight_effective")))
    audit_senses = _keys_under(mimo, "prediction_audit.modalities") or cc_senses
    channels = _keys_under(mimo, "wellness_rank.winner_margin_share")
    regions = _keys_under(mimo, "skin_regions")
    lp_keys = _keys_under(mimo, "learning_progress.per_sense")
    # DESIGN.md SECTION 11 (operator 2026-09-22): no bare score decimal anywhere a person reads.
    # Every chart below draws a COUNT per 100, a REAL UNIT (cm, degrees, GB, ms), or a PERCENTAGE
    # between two named ends; a cosine or a correlation is not drawn at all - its caption says
    # "not yet measured as a count" until the trainer logs the count form.
    P = 100.0
    charts += [
        chart("Causal chain: out of every 100 planning events, was the chosen plan the one "
              "predicted best?", False,
              [(_scaled("causal_chain.chosen_best_share", P, "causal_chain.chosen_best_per100"),
                "chosen plan predicted best"),
               (_scaled(_chance_of_alternatives, P, "causal_chain.chance"),
                "picking at random would score this"),
               (_scaled("causal_chain.chosen_closest_share", P, "causal_chain.chosen_closest_per100"),
                "chosen plan closest to its goal"),
               (_scaled("causal_chain.chosen_tied_share", P, "causal_chain.chosen_tied_per100"),
                "chosen plan tied (its lead inside the evaluator's own error band)"),
               (_scaled("causal_chain.reached_body_share", P, "causal_chain.reached_body_per100"),
                "plan's first command reached the body")],
              doc="chart.causal_choice", unit="/100", requires="causal_chain",
              caption="Out of every 100 (planning event, body) pairs: 0 = never, 100 = always. "
                      "The flat line is what picking at random among the chosen plan and its "
                      "alternatives scores. Since 2026-09-23 a plan counts as best only when its "
                      "lead over the next exceeds the tie band (the evaluator's own error on a "
                      "difference between two candidates); inside it the plan is counted as tied. "
                      "While that band is wider than the spread of the candidates' predicted "
                      "wellness (the next chart), every choice reads tied whatever the forward "
                      "model predicts, so the tied line is no evidence about the forward model; "
                      "the planner recheck's captured share is its planner-side reading. The "
                      "planner aims at distance to a remembered model step, so the closest line is its "
                      "own objective and the best line is whether predicted wellness agrees with "
                      "it: its goal line is unmeasured."),
        chart("Causal chain: how far apart the candidates' predicted wellness sat (per cent of "
              "the five scores' full range)", False,
              [(_scaled("causal_chain.gap_to_best_alt_mean", P / 5.0, "causal_chain.gap_pct"),
                "chosen plan's lead over the best alternative"),
               (_scaled("causal_chain.candidate_spread_mean", P / 5.0, "causal_chain.spread_pct"),
                "spread from the lowest candidate to the highest"),
               (_scaled("causal_chain.tie_band", P / 5.0, "causal_chain.tie_band_pct"),
                "the tie band (inside it, candidates count as tied)")],
              doc="chart.causal_choice", unit="%", requires=lambda r: "tie_band" in (r.get("causal_chain") or {}),
              caption="100% = all five scores from their worst to their best. A lead inside the "
                      "tie band is smaller than the evaluator's own error on a difference between "
                      "two candidates; while the band line sits above the spread line, every "
                      "choice is tied whatever the forward model predicts."),
        chart("Causal chain: how far the command sent sat from the command the planner imagined (cm)",
              False,
              [("causal_chain.plan_exec_distance_cm", "distance per handle, window mean")],
              doc="chart.causal_choice", unit="cm",
              requires=lambda r: "plan_exec_distance_cm" in (r.get("causal_chain") or {}),
              caption="0 cm = the body got exactly the command the planner imagined it would get. "
                      "It replaces the plan's share of the executed command, which was the "
                      "smoother's weight by construction."),
        chart("Causal chain: prediction error before the command, in per cent of 'nothing "
              "changes' (one model step later)", False,
              [(_ratio(f"causal_chain.pred_err_next.{m}", f"causal_chain.persist_err_next.{m}",
                       f"causal_chain.ratio_next.{m}", P), m) for m in cc_senses],
              doc="chart.causal_prediction", unit="%", requires="causal_chain",
              caption="100% = no better than predicting nothing changes, 0% = exact, above "
                      "100% = worse than nothing changes. The prediction was made before the "
                      "command was sent and graded on the raw tokens that arrived."),
        chart("Causal chain: prediction error before the command, in per cent of 'nothing "
              "changes' (at the command's end)", False,
              [(_ratio(f"causal_chain.pred_err_latch.{m}", f"causal_chain.persist_err_latch.{m}",
                       f"causal_chain.ratio_latch.{m}", P), m) for m in cc_senses],
              doc="chart.causal_prediction", unit="%", requires="causal_chain",
              caption="The same, graded when the command ended (--motor-hold model steps later)."),
        chart("Causal chain: predicted against realised wellness change (per cent of the "
              "five scores' full range)", False,
              [(_scaled("causal_chain.predicted_wellness_change", P / 5.0,
                        "causal_chain.predicted_wellness_change_pct"), "predicted change"),
               (_scaled("causal_chain.realized_wellness_change", P / 5.0,
                        "causal_chain.realized_wellness_change_pct"), "realised change")],
              doc="chart.causal_wellness", unit="%", requires="causal_chain",
              caption="Over each command's tenure; 100% = all five scores from their worst to "
                      "their best. Whether the prediction tracks the change command by command is "
                      "a correlation: not yet measured as a count."),
        chart("Efference body head: the next 50 ms (cm per handle)", False,
              [("efference.body_cm_err_next", "head's error"),
               ("efference.body_cm_do_nothing_next", "doing nothing (the move itself)"),
               ("efference.body_cm_size_matched_next", "with another body's command, size-matched"),
               ("efference.body_cm_shuffled_next", "with another body's absolute command")],
              doc="chart.efference_body", unit="cm", requires="efference",
              caption="An error in cm. Below doing nothing = the head foresees some of the "
                      "move; below the size-matched line = it uses THIS command. Equal to the "
                      "size-matched line = it ignores the command it is handed. The absolute "
                      "line (kept for one segment) sits higher for any head whose output grows "
                      "with the asked change."),
        chart("Efference body head: the command's 250 ms (cm per handle)", False,
              [("efference.body_cm_err_latch", "head's error"),
               ("efference.body_cm_do_nothing_latch", "doing nothing (the move itself)"),
               ("efference.body_cm_size_matched_latch", "with another body's command, size-matched"),
               ("efference.body_cm_shuffled_latch", "with another body's absolute command")],
              doc="chart.efference_body", unit="cm", requires="efference",
              caption="The same over the command's whole hold; this is the expectation "
                      "controllability judges the body against (its control still reads "
                      "the absolute line)."),
        chart("Efference gaze head (degrees of eye turn)", False,
              [(_idx("efference.gaze_deg_err", i, f"efference.gaze_deg_err.{i}"),
                f"head's error at {50 * (i + 1)} ms") for i in range(3)]
              + [(_idx("efference.gaze_deg_do_nothing", i, f"efference.gaze_deg_do_nothing.{i}"),
                  f"doing nothing at {50 * (i + 1)} ms") for i in range(3)],
              doc="chart.efference_gaze", unit="°", requires="efference",
              caption="An error in degrees; each horizon against the eye's own turn over it."),
        chart("Vestibular: prediction error in per cent of 'nothing changes'", False,
              [(_ratio("vestibular.pred_err", "vestibular.persist_err", "vestibular.ratio", P),
                "next-model-step error")],
              doc="chart.vestibular", unit="%", requires="vestibular",
              caption="100% = no better than nothing changes, 0% = exact. Below 100% = the "
                      "model foresees how its own load and balance change."),
        chart("Posture from the vestibular vector (degrees)", False,
              [("vestibular.head_tilt_deg", "head tilt"),
               ("vestibular.trunk_roll_deg", "trunk roll (size)"),
               ("vestibular.trunk_pitch_deg", "trunk pitch")],
              doc="chart.vestibular_posture", unit="°", requires="vestibular",
              caption="0 = lying on the back; 90 = on a side (roll) or a quarter turn "
                      "(tilt); 180 = face down or upside down."),
        chart("Posture: how far the bodies have rolled (degrees)", False,
              [("roll_deg_mean", "mean roll"), ("roll_deg_max", "largest roll")],
              doc="chart.posture", unit="°",
              caption="0 = lying on the back, 90 = on a side, 180 = face down."),
        chart("Posture: out of every 100 model steps of a body, how many rolled past 120 degrees", False,
              [(_scaled("rolled_over_fraction", P, "rolled_over_per100"), "rolled past 120 degrees")],
              doc="chart.posture", unit="/100",
              caption="0 = no body was past 120 degrees, 100 = every body was, all window."),
        chart("Controllability: out of every 100 commands, how many could be measured", False,
              [(_scaled("controllability.measured_share", P, "controllability.measured_per100"),
                "commands large enough to measure"),
               (_scaled("controllability.gate_open", P, "controllability.gate_open_per100"),
                "evidence gate open (100) or shut (0), window's end")],
              doc="chart.controllability", unit="/100", requires="controllability",
              caption="Whether the body did what was expected is an averaged cosine in the "
                      "log: not yet measured as a count. The channel enters the goal ranking "
                      "only while that agreement beats its control by the evidence gate."),
        chart("Controllability: agreement beyond another body's command (drawn, not counted)", False,
              [("controllability.difference_mean", "agreement minus the shuffled-command control")],
              doc="chart.controllability",
              requires=lambda r: "difference_mean" in (r.get("controllability") or {}),
              caption="Above the line = the expectation formed from THIS command matched the result "
                      "better than the same state paired with another body's command did. A "
                      "difference of two averaged cosines: not yet measured as a count.",
              ends=[(0.0, "no better than another body's command")]),
        # the gate in force beside the recorded-only shadow gate (2026-10-03, lem/model/
        # wellness_channels.py SHADOW_CONTROL_KIND): counts of graded commands, per window
        chart("Controllability: out of every 100 graded commands a gate judged, after how many it "
              "stood open", False,
              [(_ratio("controllability.gate_open_commands", "controllability.gate_commands",
                       "controllability.gate_open_commands_per100", 100.0),
                "the gate in force (control: another body's absolute command)"),
               (_ratio("controllability.shadow_gate_open_commands", "controllability.shadow_gate_commands",
                       "controllability.shadow_gate_open_commands_per100", 100.0),
                "the shadow gate (control: another body's size-matched command; recorded only)")],
              doc="chart.controllability_gates", unit="/100",
              requires=lambda r: "shadow_gate_commands" in (r.get("controllability") or {}),
              caption="0 = the gate stood shut after every graded command it judged in the window, "
                      "100 = open after every one. The controllability weight in the goal ranking "
                      "reads the gate in force; the shadow gate is recorded beside it on the same "
                      "graded commands and changes nothing."),
        chart("Learning progress (per cent)", False,
              [(_scaled("learning_progress.value_mean", P, "learning_progress.value_pct"),
                "the channel (mean over senses)")]
              + [(_scaled(f"learning_progress.per_sense.{k}", P, f"learning_progress.pct.{k}"), k)
                 for k in lp_keys]
              + [(_scaled("learning_progress.control_mean", P, "learning_progress.control_pct"),
                  "the same-model-step control (no learning possible)")],
              doc="chart.learning_progress", unit="%", requires="learning_progress",
              caption="+100% = the error relative to 'nothing changes' fell to zero lately, "
                      "0% = no change, -100% = it doubled. On frozen weights the channel read "
                      "-1.6% to -7.1% from the world's own change; the control line, when the "
                      "log carries it, is that same reading taken where no learning can happen."),
        chart("Which channels made the goal win (out of every 100 parts of the winner's lead)",
              False,
              [(_scaled(f"wellness_rank.winner_margin_share.{c}", P,
                        f"wellness_rank.winner_margin_per100.{c}"), c) for c in channels]
              + [(_scaled("wellness_rank.integrity_gated_share", P,
                          "wellness_rank.integrity_gated_per100"),
                  "candidates over the integrity gate line (of every 100)"),
                 # rank_values gives a body whose EVERY candidate is over the line its least
                 # harmful one, so the gate's "never a goal" holds only outside these events
                 (_scaled("wellness_rank.all_gated_share", P, "wellness_rank.all_gated_per100"),
                  "planning events where every candidate was over it, so the least harmful "
                  "became the goal (of every 100)")],
              doc="chart.wellness_rank", unit="/100", requires="wellness_rank",
              caption="The channels add to 100; one channel near 100 decides alone, below 0 "
                      "pushed against the winner."),
        chart("Goal age: out of every 100 chosen goals, how many sat in the oldest and the newest "
              "tenth of the goal ring", False,
              [(_scaled("wellness_rank.goal_age.oldest_tenth_share", P, "wellness_rank.goal_age.oldest_per100"),
                "in the oldest tenth"),
               (_scaled("wellness_rank.goal_age.newest_tenth_share", P, "wellness_rank.goal_age.newest_per100"),
                "in the newest tenth"),
               (_age_blind, "an age-blind choice would put this many in each")],
              doc="chart.wellness_rank", unit="/100",
              requires=lambda r: "goal_age" in (r.get("wellness_rank") or {}),
              caption="A ranking that follows what the model steps hold puts about 10 of every 100 "
                      "goals in each tenth; one that follows the ring's time axis piles them at "
                      "one end (review b3: 95 of every 100 in the oldest tenth)."),
        chart("Goal age: how far back the chosen goals sat (per cent of the way from the newest "
              "model step of the ring to the oldest)", False,
              [(_scaled("wellness_rank.goal_age.mean_share_back", P, "wellness_rank.goal_age.back_pct"),
                "mean place in the ring")],
              doc="chart.wellness_rank", unit="%",
              requires=lambda r: "goal_age" in (r.get("wellness_rank") or {}),
              caption="0% = the newest model step, 100% = the oldest; an age-blind choice averages 50%."),
        chart("Each channel's weight (out of every 100 parts of all the weights)", False,
              [(_scaled(f"wellness_rank.weight_share_by_channel.{c}", P,
                        f"wellness_rank.weight_per100.{c}"), c)
               for c in _keys_under(mimo, "wellness_rank.weight_share_by_channel")],
              doc="chart.wellness_rank", unit="/100",
              requires=lambda r: "weight_share_by_channel" in (r.get("wellness_rank") or {}),
              caption="Each channel's priority weight over the sum of the weights in force."),
        chart("Contact head on touch ONSETS against its base-rate control (pairs ranked right, "
              "of every 100)", False,
              [(_scaled("contact_control.onset_auc_self", P, "contact_control.onset_self_per100"),
                "own-part onsets: the head"),
               (_scaled("contact_control.control_onset_auc_self", P,
                        "contact_control.onset_self_control_per100"),
                "own-part onsets: the base-rate control"),
               (_scaled("contact_control.onset_auc_world", P, "contact_control.onset_world_per100"),
                "outside onsets: the head"),
               (_scaled("contact_control.control_onset_auc_world", P,
                        "contact_control.onset_world_control_per100"),
                "outside onsets: the base-rate control")],
              doc="chart.contact_head", unit="/100",
              requires=lambda r: "onset_auc_self" in (r.get("contact_control") or {}),
              caption="Over patches at a model step whose contact bit was off the model step before: out of every "
                      "100 (onset, no onset) pairs, how many the head ranks the right way round "
                      "(50 = a coin flip, 100 = every onset above every non-onset), beside the "
                      "base-rate control on the same patches and model steps."),
        chart("Contact head against its base-rate control, every patch at every model step (pairs ranked right, "
              "of every 100)",
              False,
              [(_scaled("contact_control.head_auc_bits.any", P, "contact_control.head_per100"),
                "head"),
               (_scaled("contact_control.control_auc_bits.any", P, "contact_control.control_per100"),
                "base-rate control")],
              doc="chart.contact_head", unit="/100", requires="contact_control",
              caption="50 = a coin flip, 100 = every touched patch above every "
                      "untouched one, at every model step. The head adds information only above the control. Most of "
                      "these pairs are a patch that stays touched, which the control ranks "
                      "right from the model step before; the onset chart above, when the log "
                      "carries it, reads touches that begin."),
        # THE CONTROL IS SET BESIDE THE SAME ONSETS (fix-readout-explainer 2026-09-23). success_self
        # counts every scored onset; success_control only the onsets of the moments the control ran
        # (every control_period-th moment, two or more bodies). The chart drew those two as a pair, so
        # their gap mixed the order's effect with which onsets were counted. success_paired is this
        # body's own order on the control's onsets (ContactTracker.score: ctl_real_pred_self over
        # ctl_scorable_self, the denominator of success_control), and it is the line the control is
        # drawn beside. A log written before success_paired existed draws no pair.
        chart("Foreseen touches (of every 100) and what they paid (per 100 model steps of a body)", False,
              [(_scaled("contact_control.success_self", P, "contact_control.success_self_per100"),
                "self-touches foreseen, own command, every scored onset"),
               (_scaled("contact_control.success_paired", P,
                        "contact_control.success_paired_per100"),
                "self-touches foreseen, own command, on the control's onsets"),
               (_scaled("contact_control.success_control", P,
                        "contact_control.success_control_per100"),
                "the same onsets with another body's command (the control)"),
               (_scaled("contact_control.success_world", P, "contact_control.success_world_per100"),
                "outside touches foreseen, every scored onset"),
               (_scaled("contact_control.success_paired_world", P,
                        "contact_control.success_paired_world_per100"),
                "outside touches foreseen, own command, on the control's onsets"),
               (_scaled("contact_control.success_control_world", P,
                        "contact_control.success_control_world_per100"),
                "the same outside onsets with another body's command (the control)"),
               (_scaled("contact_control.paid_mean", P, "contact_control.paid_per100"),
                "contact_control paid, per 100 model steps of a body"),
               (_scaled("contact_control.would_pay_mean", P, "contact_control.would_pay_per100"),
                "what it would pay with its gate open"),
               (_scaled("contact_control.habituation_mean", P, "contact_control.habituation_per100"),
                "pay factor after habituation (100 = a new pair)"),
               (_scaled("contact_control.gate_self_open", P, "contact_control.gate_self_per100"),
                "own-part gate open (100) or shut (0)"),
               (_scaled("contact_control.gate_world_open", P, "contact_control.gate_world_per100"),
                "outside gate open (100) or shut (0)")],
              doc="chart.contact_control", unit="/100", requires="contact_control",
              caption="0 = never foreseen, 100 = every one. The command shows only where the "
                      "'own command, on the control's onsets' line stands above the control line: "
                      "those two are counted on the same onsets. The 'every scored onset' line "
                      "counts more onsets and is not compared with the control. Pay is 2 per "
                      "foreseen self-touch and 1 per outside touch, habituated: a pair's pay "
                      "factor is 1 / (1 + its repeats), so 100 = a new pair and 50 = one repeat."),
        chart("Self-contact per body", False,
              [("self_contact_pairs", "own part pairs touching"),
               ("self_contact_patches", "patches carrying an own-part touch")],
              doc="chart.self_contact", unit="per body",
              caption="Counts per body, window means; 0 = the body touches none of itself."),
        chart("Self-touches beginning (per 100 model steps of a body)", False,
              [(_scaled("self_contact_events.onsets_per_body_step", P,
                        "self_contact_events.onsets_per100"),
                "self-touches beginning per 100 model steps of a body")],
              doc="chart.self_contact", unit="/100", requires="self_contact_events",
              caption="0 = no own-part touch began, 100 = one began on every model step of every body."),
        chart("What share of touch is the body's own (of every 100)", False,
              [(_scaled("self_contact_force_share", P, "self_contact_force_per100"),
                "own parts' part of the contact force"),
               (_scaled("self_contact_events.structural_share", P,
                        "self_contact_events.structural_per100"),
                "onsets that happen at rest anyway"),
               (_scaled("self_contact_events.in_view_share", P,
                        "self_contact_events.in_view_per100"),
                "onsets within 30 degrees of an eye")],
              doc="chart.self_contact_share", unit="/100",
              caption="Out of every 100: 0 = none, 100 = all."),
        chart("Hand pressing, world against self (of every 100 model steps of a body)", False,
              [(_scaled("hand_press.world_share", P, "hand_press.world_per100"),
                "pressed by something outside"),
               (_scaled("hand_press.self_share", P, "hand_press.self_per100"),
                "pressed by the body itself")],
              doc="chart.hand_press", unit="/100", requires="hand_press",
              caption="Over the 2 kPa line; 0 = never, 100 = every model step."),
        chart("The consequence term (of every 100)", False,
              [(_scaled("consequence.rate", P, "consequence.rate_per100"),
                "model steps with a consequence"),
               (_scaled("consequence.goal_rate", P, "consequence.goal_per100"),
                "chosen goals carrying one"),
               (_scaled("consequence.rate_count_rule", P, "consequence.count_rule_per100"),
                "the old count-every-press rule")],
              doc="chart.consequence", unit="/100",
              caption="0 = never, 100 = always. Before --consequence-self-contact exclude, a "
                      "hand pressing on the body itself counted."),
        chart("The body map's co-movement numbers (per cent of the way from a score of 0 to a score of 1)",
              False,
              [(_scaled("incorporation.handles_mine", P / (BELONG_TOP - BELONG_FLOOR),
                        "incorporation.handles_mine_pct", BELONG_FLOOR),
                "the body's handles"),
               (_scaled("incorporation.extent", P / (BELONG_TOP - BELONG_FLOOR),
                        "incorporation.extent_pct", BELONG_FLOOR),
                "extent (everything watched)")]
              + [(_scaled(f"incorporation.objects.{k}", P / (BELONG_TOP - BELONG_FLOOR),
                          f"incorporation.objects_pct.{k}", BELONG_FLOOR), k)
                 for k in _keys_under(mimo, "incorporation.objects")],
              doc="chart.incorporation", unit="%",
              caption="0% = the map's floor (a score of 0: a handle's command change and "
                      "proprioception's change never fell in the same model step; a thing never "
                      "travelled with a hand), 100% = its top (a score of 1). Co-movement, not "
                      "proof that a command caused a change."),
        chart("Named things with a proven wellness link", False,
              [("entity_self.l2_counted", "minutes level"),
               ("entity_self.l3_counted", "hours level")],
              doc="chart.entity_self", unit="named things", integer=True,
              caption="-1 = too few windows have passed for even a perfect link to count."),
        chart("How far the learned wellness copy sits from the fixed formulas (per cent)", False,
              [(_scaled("wellness_align_err", P, "wellness_align_pct"),
                "mean distance, per cent of the 0-to-1 range")],
              doc="chart.wellness_align", unit="%",
              caption="0% = identical, 100% = every copied score as far off as it can be."),
        chart("Next-model-step prediction against 'nothing changes' (the paired audit, per cent)",
              False,
              [(_scaled(f"prediction_audit.modalities.{m}.improvement_over_persistence", P,
                        f"prediction_audit.improvement_pct.{m}"), m)
               for m in audit_senses],
              doc="chart.prediction_audit", unit="%",
              caption="0% = no better than nothing changes, 100% = exact, below 0% = worse "
                      "than nothing changes."),
        chart("How much the prediction moves when the command is shuffled (per cent)", True,
              [(_scaled(f"prediction_audit.modalities.{m}.sensitivity", P,
                        f"prediction_audit.sensitivity_pct.{m}"), m) for m in audit_senses],
              doc="chart.order_sensitivity", unit="%",
              # the audit's shuffled-order sensitivity arrived with the causal-chain build: an
              # older log carries the audit and not this field, so the chart is "feature off"
              requires=lambda r: any(isinstance(v, dict) and "sensitivity" in v for v in
                                     (((r.get("prediction_audit") or {}).get("modalities"))
                                      or {}).values()),
              caption="0% = another body's absolute command moves the prediction not at all, "
                      "100% = as much as the whole change it predicts from the present."),
        chart("The swap test: own command against another body's size-matched command (per cent)",
              False,
              [(_scaled(f"prediction_audit.modalities.{m}.improvement_over_size_matched", P,
                        f"prediction_audit.size_matched_pct.{m}"), m + ", model steps that are multiples of 20")
               for m in audit_senses]
              + [(_scaled(f"prediction_audit_hold_first.senses.{m}.all_bodies.improvement_over_size_matched", P,
                          f"prediction_audit_hold_first.size_matched_pct.{m}"), m + ", a hold's first model step")
                 for m in audit_senses],
              doc="chart.swap_test", unit="%",
              # written from the first segment on the code of 2026-10-02: an older log is "feature off"
              requires=lambda r: any(isinstance(v, dict) and "improvement_over_size_matched" in v for v in
                                     (((r.get("prediction_audit") or {}).get("modalities")) or {}).values()),
              caption="0% = another body's size-matched command predicts as well (the command is not "
                      "used), 100% = exact, below 0% = the control predicts better. With --motor-hold 5 "
                      "the multiples of 20 are the last model step of a hold."),
        chart("Where the bodies are touched (of every 100 parts of the contact force)", False,
              [(_scaled(f"skin_regions.{k}.force_share", P, f"skin_regions.per100.{k}"), k)
               for k in regions],
              doc="chart.skin_regions", unit="/100",
              caption="One body's regions add to 100."),
        chart("World events (of every 100 model steps of a body)", False,
              [(_scaled(k, P, k + "_per100"), lab) for k, lab in (
                  ("task_hand_seen", "hand in view"), ("task_hand_fov", "hand in the fovea"),
                  ("task_mobile_hit", "mobile touched"),
                  ("task_mobile_hand", "mobile touched by a hand"),
                  ("task_ball_touch_rate", "ball touched"), ("task_pressing", "a hand pressing"))],
              doc="chart.world_events", unit="/100",
              caption="0 = never, 100 = every model step of a body that reports the event."),
        chart("The machine: memory (GiB)", False,
              [("machine.commit_gb", "commit charge"), ("machine.commit_limit_gb", "commit limit"),
               ("machine.ram_available_gb", "RAM available"), ("machine.rss_gb", "trainer resident")],
              doc="chart.machine_memory", unit="GiB", requires="machine",
              caption="Commit charge reaching its limit kills processes."),
        chart("The machine: time per step (ms)", False,
              [("machine.step_ms", "whole step"), ("machine.model_ms", "model"),
               ("machine.env_ms", "world"), ("machine.causal_ms", "causal chain")],
              doc="chart.machine_time", unit="ms", requires="machine",
              caption="Milliseconds per model step."),
        chart("Motor synergy: does the body follow its commands? (item 5, drawn, not counted)", False,
              [("synergy.following_mean", "graded commands' direction match")]
              + [(f"synergy.following_by_author.{k}", "following, commands by " + k)
                 for k in _keys_under(mimo, "synergy.following_by_author")],
              doc="chart.synergy", requires="synergy",
              caption="Item 5's measurement: how closely the body's 250 ms turn follows the "
                      "direction each command asked, drawn between three named ends. An averaged "
                      "cosine in the log: not yet measured as a count of commands.",
              ends=[(1.0, "exactly as commanded"), (0.0, "unrelated"), (-1.0, "exactly against")]),
        chart("Motor synergy: how much of the posture change it reproduces (logged beside item 5)",
              False,
              [(_scaled("synergy.explained_share", P, "synergy.explained_pct"),
                "explained (weighted)"),
               (_scaled("synergy.explained_share_unweighted", P, "synergy.explained_unweighted_pct"),
                "explained (unweighted)")],
              doc="chart.synergy", unit="%", requires="synergy",
              caption="0% = no better than asking for no change, 100% = the whole change, below "
                      "0% = worse than asking for none. It is not the item-5 measure: a network "
                      "that reproduces the average posture change explains a large share while "
                      "controlling nothing (review dashboards, M5)."),
        chart("Next-frame loss weight per sense (per cent of the plain lesson's weight)", False,
              [(_scaled(f"next_frame_norm.weight_effective.{m}", P,
                        f"next_frame_norm.weight_pct.{m}"), m) for m in cc_senses],
              doc="chart.next_frame_norm", unit="%", requires="next_frame_norm",
              caption="100% = the weight the lesson gives without --next-frame-norm."),
        # THE WHOLE-MODEL LIVENESS BLOCK of the field contract (2026-09-23): per layer the share of
        # units whose input rose above -3 on the window's samples, per attention pool its spread
        # each line is labelled with the part's name and its code path beside it (liveness_part_name,
        # the development dashboard's words), never the bare path
        chart("Responding units in selected hidden layers (of 100)", False,
              [(_scaled(_liveness_value("layers", k), P, f"liveness.layers_per100[{k}]"),
                liveness_part_name(k))
               for k in _keys_under(mimo, "liveness.layers", cap=16)]
              + [(_half_line, "the line: half the units (DESIGN.md section 10)")],
              doc="chart.liveness", unit="/100", requires="liveness",
              caption="0 = no unit crossed the response threshold on the sampled inputs; 100 = all did. "
                      "Under 50 triggers a layer warning. Up to 16 layers are displayed; useful ability "
                      "needs separate tests."),
        chart("Liveness: how evenly each attention pool spreads (drawn, not counted)", False,
              [(_liveness_value("pools", k), liveness_part_name(k))
               for k in _keys_under(mimo, "liveness.pools", cap=16)],
              doc="chart.liveness",
              requires=lambda r: isinstance((r.get("liveness") or {}).get("pools"), dict),
              caption="Each pool's attention entropy as a share of its even maximum, drawn "
                      "between its two ends: an entropy is not yet measured as a count. A pool "
                      "at the bottom concentrated on one input per measured reading.",
              ends=[(0.0, "reads one element"), (1.0, "spreads evenly")]),
    ]
    # the depth licence: the imagination chain's per-depth win-against-do-nothing
    # share is what sets how many moments ahead the planner may aim, and the
    # planner's first step is what the body executes. win_emas is a list, so its
    # per-depth series are built here rather than through the dotted-path helper.
    # TWO UNITS, TWO CHARTS (DESIGN.md section 11): the win shares are counts per 100 comparisons,
    # the depths are moments; one axis could not carry both units on its ticks
    def _keep_line(r):
        return 50.0 if isinstance((r.get("plan_license") or {}).get("win_emas"), list) else None
    _keep_line.field = "plan_license.keep_line_per100"
    lic_wins = chart("Depth licence: out of every 100 comparisons with doing nothing, how many "
                     "each depth wins", False,
                     [(_keep_line, "the line that keeps a depth (50)")],
                     doc="chart.license", unit="/100",
                     caption="Each depth of the imagination chain is graded against assuming "
                             "nothing changed. Out of every 100 of those comparisons, how many it "
                             "wins (100 = every time, 50 = the line that keeps a depth, 0 = never) "
                             "decides how far ahead the planner may aim, and the planner's first "
                             "step is what the body does. Winning that comparison is the line for "
                             "keeping a depth, not evidence the plans arrive anywhere. The count "
                             "is a running average over about the last 100 comparisons that "
                             "starts at 0 when a depth first opens (lem_model.py _plan_win_ema), so "
                             "a new depth that wins every comparison still reads under 50 for its "
                             "first 68.")
    lic = chart("Depth licence: how far ahead the planner is allowed to aim", False,
                [("plan_license.licensed", "extra depths earned"),
                 ("intention.horizon", "model steps the planner may aim ahead"),
                 ("intention.plan_length", "steps in the plan it actually made")],
                doc="chart.license", unit="steps", integer=True,
                caption="The depths the win shares above have opened, the model steps the planner "
                        "may aim ahead, and the steps in the plan it actually made.")
    if lic_wins:
        wins: Dict[int, List[List[float]]] = {}
        for r in mimo:
            we = (r.get("plan_license") or {}).get("win_emas")
            if not isinstance(we, list) or r.get("step") is None:
                continue
            for di, v in enumerate(we):
                if isinstance(v, (int, float)):
                    wins.setdefault(di, []).append([int(r["step"]), round(100.0 * float(v), 2)])
        for di in sorted(wins, reverse=True):
            if len(wins[di]) >= 2:
                lic_wins["series"].insert(0, {
                    "label": f"depth {di + 1}: comparisons won, of every 100",
                    "points": _thin(wins[di]),
                    "key": f"plan_license.win_per100.{di}"})
    # right after the collapse watch, wherever that sits; 9 is its slot plus one
    at = next((i + 1 for i, c in enumerate(charts) if c and c["doc"] == "chart.collapse"), 9)
    for c in (lic, lic_wins):
        if c:
            charts.insert(at, c)
    phases = [{"step": int(r.get("step") or 0), "phase": r.get("phase") or ""}
              for r in stats.get("phase_transition", [])]
    if dropped:
        log.warning("%d declared chart(s) not drawn - no record carries two or more "
                    "points for any of their fields: %s", len(dropped), "; ".join(dropped))
    if off:
        log.info("%d chart(s) need a feature no record of this log carries: %s",
                 len(off), "; ".join(off))
    return {
        "charts": [c for c in charts if c],
        "dropped": dropped,
        "off": off,
        "seams": resume_seams(mimo),
        "phases": phases,
        "snapmarks": snap_steps,
        "thinning": f"series thinned to <= {SERIES_POINT_CAP} points each",
    }


# ------------------------------------------------------------------ snapshots
H81_MEMORY_BIAS = ("On run H81's probe it measured a third to a half of that token (over 20 "
                   "snapshots).")


def hself_wall_fill(rows: List[dict], fill: Dict[str, str], run_name: str) -> Dict[str, str]:
    """{hself_first_evidence_wall}: how long the hierarchical self's first countable evidence takes
    in hours and days of RUNNING, at the run's own measured speed (the median model_steps_per_s of
    its last 50 stats records). Empty when the run recorded no speed."""
    rates = [r.get("model_steps_per_s") for r in rows[-50:]
             if isinstance(r.get("model_steps_per_s"), (int, float)) and r.get("model_steps_per_s") > 0]
    try:
        l2 = int(fill["{hself_l2_first_moments}"].replace(",", ""))
        l3 = int(fill["{hself_l3_first_moments}"].replace(",", ""))
    except (KeyError, ValueError):
        return {"{hself_first_evidence_wall}": ""}
    if not rates:
        return {"{hself_first_evidence_wall}": ""}
    rate = sorted(rates)[len(rates) // 2]
    return {"{hself_first_evidence_wall}":
            (f"At the measured {rate:.2f} model steps per second that is about "
             f"{l2 / rate / 3600.0:.0f} hours of running for Level 2 and about "
             f"{l3 / rate / 86400.0:.0f} days for Level 3. ")}


def entity_self_now_fill(run: str, run_name: str) -> Dict[str, str]:
    """{entity_self_now}: what the named-entity levels are doing in the run on the page, from
    its newest snapshot. A run without them says so."""
    newest = None
    for entry in reversed(snapshot_entries(run)):
        newest = _load_snapshot(run, entry)
        if newest is not None:
            break
    es = (newest or {}).get("entity_self")
    if not isinstance(es, dict) or not es.get("names"):
        return {"{entity_self_now}": "%s does not run them (--entity-self off)." % run_name}
    rd = es.get("reader") or {}
    counted = es.get("l2_counted_names") or []
    return {"{entity_self_now}": (
        "In %s: %d named things, learning from %s, %s windows lived of the %s a link needs. "
        "%s The reader is %s."
        % (run_name, len(es["names"]),
           "the body's wellness" if es.get("teacher") == "wellness" else "an ability signal",
           es.get("l2_ticks"), es.get("windows_needed"),
           ("Counted so far: " + ", ".join(counted) + ".") if counted
           else "Nothing has passed the evidence test yet.",
           "adding %s of one evaluator's range per unit of link times presence"
           % _pct_text(rd.get("applied_weight") or 0.0)
           if rd.get("gate_open") else "off, so nothing it finds changes what the body does"))}


def hself_now_fill(run: str, run_name: str) -> Dict[str, str]:
    """{hself_now}: what the hierarchical self is doing in the run on the page, from its newest
    snapshot - slots in use, the share of moments that belong to a slot, how alike a moment and its
    nearest slot are, and the windows lived against the windows needed. Says so when the run's
    snapshots do not carry it. (H84 segments 6 and 7 wrote the two gauges as moments_with_a_row and
    best_row_cos; both names are read.)"""
    newest = None
    for entry in reversed(snapshot_entries(run)):
        newest = _load_snapshot(run, entry)
        if newest is not None:
            break
    hs = (newest or {}).get("hself")
    if isinstance(hs, str) and hs.startswith("absent"):
        # capture.py writes "absent: hierarchical self off" when the run has no table. Saying
        # "the snapshots do not record it" of that would be false: they record that it is off.
        return {"{hself_now}": f"{run_name}'s newest snapshot records the table as switched "
                               "off, so it stored nothing and produced none of these numbers."}
    if not isinstance(hs, dict) or hs.get("slots_filled") is None:
        return {}
    step = int((newest.get("header") or {}).get("step", 0))
    share = hs.get("moments_with_a_slot", hs.get("moments_with_a_row"))
    near = hs.get("best_slot_cos", hs.get("best_row_cos"))
    out = [f"Newest snapshot (model step {step:,}): {int(hs['slots_filled']):,} slots in use."]
    if isinstance(share, (int, float)) and isinstance(near, (int, float)):
        # the likeness is a cosine: the page says where it sits against the match rule's line
        # (nine tenths on every sense), never the cosine itself (DESIGN.md section 11)
        out.append(f"{100 * share:.0f} of 100 model steps belong to a slot; a model step's likeness to its "
                   f"nearest slot sits " + ("above" if near >= 0.9 else "below")
                   + " the nine tenths a match needs (a cosine: not yet measured as a count).")
        if share == 0:
            out.append("So no stored pattern matches what the body lives through today, and every score "
                       "is about old model steps.")
    lived = [(name, hs.get(f"{tag}_ticks"), hs.get(f"{tag}_ticks_needed"), hs.get(f"{tag}_sig"))
             for tag, name in (("l2", "Level 2"), ("l3", "Level 3"))]
    lived = [x for x in lived if x[1] is not None and x[2] is not None]
    if lived:
        out.append(" and ".join(f"{name} has lived {int(n)} of the {int(need)} windows it needs"
                                for name, n, need, _ in lived) + ".")
        counted = [f"{name} counts {int(sig)} slots" for name, _, _, sig in lived
                   if isinstance(sig, (int, float)) and sig >= 0]
        out.append((" and ".join(counted) + ".") if counted else "Neither can count yet.")
    return {"{hself_now}": " ".join(out)}


def memory_bias_fill(run: str, run_name: str) -> Dict[str, str]:
    """{memory_bias_vs_token}: how big the recalled memory bias is against the token it lands on, from
    THIS run's newest snapshot (capture.py records bias_rms_body and landing_token_rms since
    2026-09-19). A run whose snapshots do not carry the pair says so; H81's figure is always named as
    H81's. Measured on H84 at step 128,000: bias 4.34 against a token of 3.69, 1.18 times."""
    newest = None
    for entry in reversed(snapshot_entries(run)):
        newest = _load_snapshot(run, entry)
        if newest is not None:
            break
    mem = (newest or {}).get("memory")
    bias = mem.get("bias_rms_body") if isinstance(mem, dict) else None
    token = mem.get("landing_token_rms") if isinstance(mem, dict) else None
    if isinstance(bias, (int, float)) and isinstance(token, (int, float)) and token > 0:
        hdr = newest.get("header") or {}
        text = (f"Measured on {run_name}'s newest snapshot (step {int(hdr.get('step', 0)):,}, body "
                f"{hdr.get('body', 0)}): the addition is {_pct_text(bias / token)} of the size of the "
                f"token it lands on (under 100% = smaller than the token, 100% = the same size, above "
                f"100% = larger than the token it is added to). {H81_MEMORY_BIAS}")
    else:
        text = (f"{run_name}'s snapshots do not record the size of the token the addition lands on "
                f"(snapshots written before 2026-09-19 do not, and a run with persistent memory off has "
                f"none), so its share of that token cannot be stated for this run. {H81_MEMORY_BIAS}")
    return {"{memory_bias_vs_token}": text}


def snapshot_entries(run: str) -> List[dict]:
    """The run's snapshot index, oldest first: {step, file, bytes, mtime} per snapshot."""
    idx = _load(os.path.join(run, "explain", "index.json")) or {}
    return idx.get("snapshots") or []


def _load_snapshot(run: str, entry: dict) -> Optional[dict]:
    s = _load(os.path.join(run, "explain", entry["file"]))
    return s if s and isinstance(s, dict) and "header" in s else None


def _even_indices(n: int, k: int) -> List[int]:
    """k of n positions spread evenly, the first and the last always among them."""
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [n - 1] if n else []
    return sorted({int(i * (n - 1) / (k - 1) + 0.5) for i in range(k)})


def _embedded_bytes(snap: dict) -> int:
    return len(json.dumps(snap, separators=(",", ":")).replace("</", "<\\/").encode("utf-8")) + 1


def _pick_snapshots(run: str, cap: int, budget_bytes: Optional[float] = None):
    """(snapshots, every step on disk, what limited the count).

    ONE thinning over every snapshot the run has, never a thinning of a thinning: the most
    snapshots that fit both the --max-snapshots cap and the bytes the page has left for
    them, always the newest and the first, the rest spread evenly between the two. A
    snapshot's size follows the model's size (668-705 KB each on the 192-wide H84 against
    125-450 KB on the 128-wide runs), so a COUNT cannot keep the page inside its budget -
    12 of H84's put it at 9.1 MB against 6 MB. The count is chosen from the index's own
    byte sizes, so only the chosen snapshots are loaded (measured on H84: the index's
    figure and the embedded size agree to within 1 byte); the loaded sizes are then
    checked, and one more is dropped if the index under-stated any. The newest is kept
    even when it does not fit on its own; build() then warns.
    The third value is "budget", "max_snapshots", or None when every snapshot is embedded."""
    entries = snapshot_entries(run)
    all_steps = [e["step"] for e in entries]
    n = len(entries)
    if not n:
        return [], all_steps, None

    def on_disk(i: int) -> int:
        b = entries[i].get("bytes")
        if not b:
            try:
                b = os.path.getsize(os.path.join(run, "explain", entries[i]["file"]))
            except OSError:
                b = 0
        return int(b)

    k_cap = min(max(int(cap), 1), n)
    k = k_cap
    if budget_bytes is not None:
        while k > 1 and sum(on_disk(i) for i in _even_indices(n, k)) > budget_bytes:
            k -= 1
    loaded: Dict[int, Optional[dict]] = {}
    while True:
        idx = _even_indices(n, k)
        for i in idx:
            if i not in loaded:
                loaded[i] = _load_snapshot(run, entries[i])
        snaps = [loaded[i] for i in idx if loaded[i] is not None]
        if (budget_bytes is None or k <= 1
                or sum(_embedded_bytes(s) for s in snaps) <= budget_bytes):
            break
        k -= 1
    snaps.sort(key=lambda s: s["header"]["step"])
    limited = "budget" if k < k_cap else ("max_snapshots" if k_cap < n else None)
    return snaps, all_steps, limited


def pick_snapshots(run: str, cap: int, budget_bytes: Optional[float] = None):
    """(snapshots, every step on disk) - see _pick_snapshots."""
    snaps, all_steps, _limited = _pick_snapshots(run, cap, budget_bytes)
    return snaps, all_steps


# ------------------------------------------------------------------ the phase table
def phase_table(env_step_s: float = SCH.ENV_STEP_FALLBACK_S) -> List[Dict[str, Any]]:
    """The six curriculum phases, read from the code that runs them.

    The viewer draws its phase boxes from this list instead of a literal of its own,
    because a literal drifted: three of the six names it drew - "infant", "sitter" and
    "crawler" - exist nowhere in the code. exploration_noise is what limits_for()
    actually returns (the babble floor applied) and table_exploration_noise is the raw
    table value, so the page can say which is which. The age and the two delays live in
    the trainer, whose import pulls the simulator in; a machine without it still gets
    every other column.
    """
    from lem.training.curriculum import PHASE_DEFINITIONS, PHASE_ORDER, limits_for
    try:
        from mimo_bridge.train_mimo import PHASE_AGE, PHASE_DELAYS, PHASE_SCENE
    except Exception:                    # no simulator here: those three columns stay None
        PHASE_AGE, PHASE_DELAYS, PHASE_SCENE = {}, {}, {}
    out: List[Dict[str, Any]] = []
    for name in PHASE_ORDER:
        lim = limits_for(PHASE_DEFINITIONS[name])
        delay = PHASE_DELAYS.get(name)          # (sensory, motor) in env steps of 10 ms
        out.append({
            "name": name,
            "rom_limit": lim["rom_limit"],
            "strength_limit": lim["strength_limit"],
            "exploration_noise": lim["exploration_noise"],
            "table_exploration_noise": PHASE_DEFINITIONS[name].exploration_noise,
            "context_length": lim["context_length"],
            "age_months": PHASE_AGE.get(name),
            # PHASE_DELAYS counts ENV STEPS; their length is the run's measured one
            "sensory_delay_ms": None if delay is None else round(delay[0] * env_step_s * 1000, 3),
            "motor_delay_ms": None if delay is None else round(delay[1] * env_step_s * 1000, 3),
            "scene": PHASE_SCENE.get(name),
        })
    return out


# ------------------------------------------------------------------ the gain-of-function build
DOCS = os.path.abspath(os.path.join(ROOT, "..", "..", "docs"))
PROBE_INDEX = os.path.join(SAVED, "gof_probes", "index.jsonl")
LIVENESS_DIR = os.path.join(DOCS, "gain_of_function_v3", "measurements")
PROBE_CHECKPOINTS_MAX = 8           # the newest this many probed checkpoints of the lineage
CAUSAL_EVENTS_MAX = 8               # the newest this many resolved orders with a plan
CAUSAL_TAIL_BYTES = 400_000         # causal_chain.jsonl is read from its end only

# THE CODE PATH OF EACH COLUMN of measurements/liveness_over_lineage/README.txt, whose column words
# are the measurement's own shorthand. Matched on the table's 344,400 row, where each column reads
# what liveness_sweep_ckpt_step_344400.json records for that one part (live shares 0.000, 0.051,
# 0.047, 0.049 and 0.365; attention entropies 0.001, 0.091 and 0.004, each unique in the sweep)
LIVENESS_PARTS = {
    "pose-head": "tenure_heads.pose.0",
    "wellness-copy": "wellness.approximate.net.0",
    "reflex-blk0": "motor_controller.residual_blocks.0.0",
    "recon-touch": "recon_decoders.tactile.0",
    "memory-out": "persistent_memory.outcome.0",
    "motor-pool": "encoder_registry.encoders.motor.pool_attn",
    "touch-pool": "encoder_registry.encoders.tactile.pool_attn",
    "core-blk0-cross": "forward_model.blocks.0.cm_attn",
}


def liveness_part_name(path: str) -> str:
    """A part the liveness sweep measured, by the name the development dashboard's Model health card
    gives it (development_dashboard._part, the code's own words), with the code path beside it: one
    part has one name on both pages, and no title or bar is a bare code path. A path the dashboard
    has no name for is printed as it is."""
    from mimo_bridge.tools.development_dashboard import _part
    name = _part(path)
    return path if name == path else f"{name} ({path})"


def lineage_prefix(run_name: str) -> str:
    """H84_fresh_drivers_20260918_seg31_toy_and_push -> H84_fresh_drivers_20260918."""
    m = re.match(r"^(.*?)_seg\d+", run_name)
    return m.group(1) if m else run_name


def _pct_text(share: Any) -> str:
    """A share (1 = all) as a per-cent number a person reads (DESIGN.md section 11): whole numbers
    from 10 up, one decimal below, a tiny non-zero one as "under 0.001%", never an exponent."""
    try:
        v = 100.0 * float(share)
    except (TypeError, ValueError):
        return "not recorded"
    if v != v:
        return "not recorded"
    if v != 0.0 and abs(v) < 0.001:
        return ("-" if v < 0 else "") + "under 0.001%"
    return f"{v:.0f}%" if abs(v) >= 10 else f"{round(v, 1):g}%"


def _probe_controls(probe: str, controls: Any) -> List[str]:
    """A probe's controls as plain sentences. A control the probe wrote as a sentence is kept
    as written; the measured ones this page knows are put in words (per cent and cm, never a
    bare decimal); any other is named."""
    out: List[str] = []
    for key, val in (controls or {}).items() if isinstance(controls, dict) else []:
        if isinstance(val, str):
            out.append(val)
        elif key == "ablation" and isinstance(val, dict):
            parts = [f"{_pct_text(v.get('share_of_error_removed') or 0.0)} at {h}"
                     for h, v in val.items() if isinstance(v, dict)]
            out.append("removing the command from the prediction changed its error by "
                       + ", ".join(parts) + " (0% = the command is not used at all)")
        elif key == "instrument_repeat" and isinstance(val, dict):
            out.append(f"the same measurement repeated {val.get('repeats')} times gave "
                       + ("identical numbers" if val.get("exact") else "different numbers"))
        elif key == "random_order" and isinstance(val, dict):
            out.append("a random command instead of the limb's command: it foresaw "
                       f"{_pct_text(val.get('captured_share_tenure') or 0.0)} of the change; the "
                       f"head's error {float(val.get('tenure_error_cm') or 0.0):.2f} cm against doing "
                       f"nothing {float(val.get('do_nothing_error_cm') or 0.0):.2f} cm")
        elif key == "earlier_checkpoint" and isinstance(val, dict):
            out.append(f"the lineage's earlier checkpoint at step {int(val.get('lineage_step') or 0):,}: "
                       f"it foresaw {_pct_text(val.get('captured_share_tenure') or 0.0)} of the change "
                       f"(0% = ignores the command, 100% = foresees the whole change)")
        else:
            out.append(f"a further control, '{key}', is recorded in the probe's own file")
    return out


def read_probes(run: str) -> Dict[str, Any]:
    """The gain-of-function probe results of the page's lineage (DESIGN.md section 9): per probed
    checkpoint, newest first, each probe's newest headline and its controls. Read from
    Saved/mimo_bridge/gof_probes/index.jsonl and each result file; a page built where the index
    does not exist says so."""
    run_name = os.path.basename(os.path.normpath(run))
    prefix = lineage_prefix(run_name)
    out: Dict[str, Any] = {"index": PROBE_INDEX, "lineage": prefix, "checkpoints": []}
    if not os.path.exists(PROBE_INDEX):
        out["missing"] = "no probe has written Saved/mimo_bridge/gof_probes/index.jsonl yet"
        return out
    newest: Dict[tuple, dict] = {}
    try:
        with open(PROBE_INDEX, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not str(r.get("run") or "").startswith(prefix):
                    continue
                key = (os.path.basename(str(r.get("checkpoint") or "")), str(r.get("probe")))
                if key not in newest or str(r.get("time") or "") >= str(newest[key].get("time") or ""):
                    newest[key] = r
    except OSError as exc:
        out["missing"] = f"the probe index could not be read: {exc}"
        return out
    by_ckpt: Dict[str, dict] = {}
    for (ck, probe), r in newest.items():
        c = by_ckpt.setdefault(ck, {"checkpoint": ck, "run": r.get("run"),
                                    "lineage_step": r.get("lineage_step"), "probes": {}})
        detail: Dict[str, Any] = {"item": r.get("item"), "time": r.get("time"),
                                  "headline": r.get("headline") or {}, "controls": []}
        doc = _load(str(r.get("path") or ""))
        if isinstance(doc, dict):
            detail["status"] = doc.get("status")
            detail["controls"] = _probe_controls(probe, doc.get("controls"))
        c["probes"][probe] = detail
    ordered = sorted(by_ckpt.values(), key=lambda c: -int(c.get("lineage_step") or 0))
    out["checkpoints"] = ordered[:PROBE_CHECKPOINTS_MAX]
    out["checkpoints_total"] = len(ordered)
    return out


def read_liveness() -> Dict[str, Any]:
    """The liveness measurements the lead took on 2026-09-22 (docs/gain_of_function_v3/
    measurements/): the lineage table (liveness_over_lineage/README.txt) and the newest
    checkpoint's sweep (liveness_sweep_ckpt_step_344400.json). None of it is typed here."""
    out: Dict[str, Any] = {"dir": LIVENESS_DIR}
    readme = os.path.join(LIVENESS_DIR, "liveness_over_lineage", "README.txt")
    try:
        lines = open(readme, encoding="utf-8").read().splitlines()
    except OSError:
        out["missing"] = "docs/gain_of_function_v3/measurements/liveness_over_lineage/README.txt is absent"
        return out
    head = next((i for i, ln in enumerate(lines) if ln.split()[:1] == ["step"]), None)
    if head is not None:
        cols = lines[head].split()[1:]
        rows = []
        for ln in lines[head + 1:]:
            parts = ln.split()
            if not parts or not parts[0].isdigit():
                break
            rows.append({"step": int(parts[0]), "values": parts[1:1 + len(cols)]})
        out["columns"] = [{"key": c, "path": LIVENESS_PARTS.get(c),
                           "name": liveness_part_name(LIVENESS_PARTS[c]) if c in LIVENESS_PARTS else c}
                          for c in cols]
        out["rows"] = rows
        out["intro"] = " ".join(l.strip() for l in lines[:head] if l.strip())
        out["notes"] = " ".join(l.strip() for l in lines[head + 1 + len(rows):] if l.strip())
    sweeps = sorted(glob.glob(os.path.join(LIVENESS_DIR, "liveness_sweep_ckpt_step_*.json")))
    if sweeps:
        sw = _load(sweeps[-1]) or {}
        # LOCKED = the attention DESIGN.md section 10 lists as locked on one input: entropy under a
        # tenth of an even spread (the section names 0.001 to 0.091 locked and 0.53 open). The page
        # shows its COUNT form (section 11): of every 100 parts of the attention, how many the single
        # largest input takes (mean_max_weight), and how many inputs it could read
        out["sweep"] = {"file": os.path.basename(sweeps[-1]), "moments": sw.get("moments"),
                        "dead_layers": sw.get("dead_layers") or {},
                        "locked_attention": {k: {"top_share": v.get("mean_max_weight"),
                                                 "keys": v.get("keys")}
                                             for k, v in (sw.get("attention") or {}).items()
                                             if isinstance(v, dict)
                                             and (v.get("entropy_share_of_uniform") or 1.0) < 0.1}}
        # each shown part's name with its code path beside it (the bars and the boxes' popup rows)
        shown = list(out["sweep"]["dead_layers"]) + list(out["sweep"]["locked_attention"])
        out["sweep"]["names"] = {k: liveness_part_name(k) for k in shown}
    return out


def read_causal_events(run: str) -> Dict[str, Any]:
    """The newest resolved orders that carried a planning event, from <run>/causal_chain.jsonl
    (mimo_bridge/causal_chain.py writes one line per order whose end was graded). Only the file's
    last CAUSAL_TAIL_BYTES are read. A run launched without --causal-chain has no file."""
    path = os.path.join(run, "causal_chain.jsonl")
    out: Dict[str, Any] = {"file": path, "events": []}
    # THE FILE ROTATES (review b1, minor 7): past its size limit causal_chain.py renames it to
    # causal_chain_<YYYYmmdd_HHMMSS>.jsonl and starts a fresh one, so just after a rotation the
    # current file holds a handful of lines. The newest rotated file is read first, then the
    # current one, and the newest resolved orders of the two are kept.
    rotated = sorted(glob.glob(os.path.join(run, "causal_chain_*.jsonl")))
    sources = rotated[-1:] + ([path] if os.path.exists(path) else [])
    if not sources:
        out["missing"] = "this run wrote no causal_chain.jsonl (it was launched without --causal-chain on)"
        return out
    out["files"] = [os.path.basename(p) for p in sources]
    recs = []
    for src in sources:
        try:
            with open(src, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - CAUSAL_TAIL_BYTES))
                tail = fh.read().decode("utf-8", "replace").splitlines()
        except OSError as exc:
            out.setdefault("unreadable", []).append(f"{os.path.basename(src)}: {exc}")
            continue
        for ln in tail:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("type") == "latch":
                recs.append(r)
    if not recs and out.get("unreadable"):
        out["missing"] = "causal_chain.jsonl could not be read: " + "; ".join(out["unreadable"])
        return out
    with_plan = [r for r in recs if r.get("plan")]
    pick = (with_plan or recs)[-CAUSAL_EVENTS_MAX:]
    keep_plan = ("goal", "plan0", "pred_wellness", "goal_distance", "chosen_rank",
                 "alternatives", "wellness_rank", "horizon", "bodies")
    for r in pick:
        ev = {k: r.get(k) for k in ("step", "order_seq", "source", "bodies", "hold",
                                    "reached_body", "plan_share_of_executed", "judge_now",
                                    "next", "latch")}
        if r.get("plan"):
            ev["plan"] = {k: r["plan"].get(k) for k in keep_plan if k in r["plan"]}
        out["events"].append(ev)
    out["resolved_in_tail"] = len(recs)
    return out


# ------------------------------------------------------------------ page assembly
def _html_text(text: str) -> str:
    import html
    return html.escape(str(text), quote=False)


def _json_block(bid: str, obj: Any) -> str:
    payload = json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")
    return f'<script type="application/json" id="{bid}">{payload}</script>'


def explanation_sources():
    return [os.path.abspath(__file__)] + [os.path.join(PKG, name) for name in
            ("current_state.py", "refresh.py", "schema.py", "walkthrough.py", "viewer.js", "viewer.css")] + [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), name) for name in ELIMINATION_FILES]


# THE ROOT-CAUSE INVESTIGATIONS IN THE ARCHITECTURE TAB (the style guide, "EVERY ROOT-CAUSE INVESTIGATION HAS A LIVE
# PROCESS-OF-ELIMINATION DIAGRAM", the operator's rule of Oct 5, 2026).
# INPUT     the investigation files of this code's checkout (docs/gain_of_function_v4/investigations) and the test
#           registry of the checkout whose Saved/ the page is built from (tools/elimination.py view).
# OPERATION every box computed from the registry's rows and their result files, read at most once per
#           COMPONENT_VIEW_KEEP_S seconds in one process; the page draws them with tools/elimination_diagram.js, the
#           renderer of the dashboard's Tests page, which the build puts into the page with its stylesheet.
# OUTPUT    the page's data-investigations block, or None when the view cannot be computed. The viewer reads GET
#           /api/investigations again when the development dashboard serves the page, so this copy is what a page
#           opened from disk shows, with the time it was read.
ELIMINATION_FILES = ("elimination.py", "elimination_diagram.js", "elimination_diagram.css")
_INVESTIGATION_VIEW: Dict[str, Any] = {}


def investigation_view(data_root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    data_root = os.path.abspath(data_root or os.path.join(SAVED, "..", ".."))
    kept = _INVESTIGATION_VIEW.get(data_root)
    if kept and time.time() - kept[0] < COMPONENT_VIEW_KEEP_S:
        return kept[1]
    try:
        from mimo_bridge.tools import elimination
        view = elimination.view(data_root=data_root)
    except Exception as exc:          # the page is built without them; its panel says that the page holds none
        log.warning("the investigations were not read for the page: %s: %s", type(exc).__name__, exc)
        view = None
    _INVESTIGATION_VIEW[data_root] = (time.time(), view)
    return view


# THE ARCHITECTURE TAB'S CODE AND QA MAP (the recovery directive of Oct 5, 2026, sections 7C and 9).
# INPUT     docs/gain_of_function_v4/component_manifest.json, the QA log and the defect register of this code's
#           checkout, its git history, and the test registry and the active run's launch settings of the checkout
#           whose Saved/ the page is built from (component_manifest.status).
# OPERATION per component, the fields the box popups print (COMPONENT_FIELDS), and by_block, the components of each
#           box's entry key; read at most once per COMPONENT_VIEW_KEEP_S seconds in one process, so the suite's many
#           builds and a watch loop's rebuilds start git at most once a minute.
# OUTPUT    the page's data-components block, or None when the manifest cannot be read (the popup then says so).
#           The viewer reads GET /api/components again when the development dashboard serves the page, so this
#           copy is what a page opened from disk shows, with the time it was read.
COMPONENT_VIEW_KEEP_S = 60.0
COMPONENT_FIELDS = ("id", "kind", "name", "does", "blocks", "flags", "ref", "confirmed", "entry_fresh",
                    "entry_stale_text", "last_code_change_text", "last_qa_text", "qa_stale", "qa_stale_text", "switch",
                    "open_defects", "linked_tests")
_COMPONENT_VIEW: Dict[str, Any] = {}


def component_view(data_root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    data_root = os.path.abspath(data_root or os.path.join(SAVED, "..", ".."))
    kept = _COMPONENT_VIEW.get(data_root)
    if kept and time.time() - kept[0] < COMPONENT_VIEW_KEEP_S:
        return kept[1]
    try:
        from mimo_bridge.tools import component_manifest
        full = component_manifest.status(data_root=data_root)
    except Exception as exc:          # the page is built without the map; its popups say that the page holds none
        log.warning("the component manifest's view was not read for the page: %s: %s", type(exc).__name__, exc)
        view = None
    else:
        check = full.get("check") or {}
        view = dict(read_at_text=full.get("read_at_text"), no_qa_text=full.get("no_qa_text"),
                    active_run=full.get("active_run"), by_block=full.get("by_block") or {},
                    check=dict(ok=check.get("ok"), problems=check.get("problems") or [], sentence=check.get("sentence")),
                    components=[{k: c.get(k) for k in COMPONENT_FIELDS} for c in full.get("components") or []
                                if isinstance(c, dict)],
                    error=full.get("error"))
    _COMPONENT_VIEW[data_root] = (time.time(), view)
    return view


# EACH REPAIR'S SWITCH IN THE RUN THE PAGE DRAWS (the decided design of the Architecture tab, Oct 7, 2026, D-E2: a
# solid orange ring marks a repair at IN TRAINING "whose switch is on in the drawn run").
# INPUT     the component manifest's repair entries of this code's checkout (each names its switch {flag, on}) and the
#           drawn run's own args.json, every flag it was launched with (not only PUBLISHED_ARGS).
# OPERATION per repair: the flag's value in that args.json and whether it is one of the values the manifest names as
#           "on" - component_manifest.switch_state's rule, applied to the drawn run instead of the active run.
# OUTPUT    META.repair_switches {repair id: {flag, value, present, on}}, or {} when the manifest cannot be read. The
#           component view's own switch describes the ACTIVE run (training_continuation.json); a page built for another
#           run (--run) marks the drawn run's switches from this.
def repair_switches(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from mimo_bridge.tools import component_manifest
        known = component_manifest.components(component_manifest.load(os.path.join(ROOT, "..", "..")))
    except Exception as exc:          # the page then marks no repair as switched on in its run, and says so per line
        log.warning("the repairs' switches were not read for the page: %s: %s", type(exc).__name__, exc)
        return {}
    out: Dict[str, Any] = {}
    for cid, entry in known.items():
        switch = entry.get("switch") if isinstance(entry.get("switch"), dict) else None
        if entry.get("kind") != "repair" or not switch:
            continue
        key = str(switch.get("flag") or "").lstrip("-").replace("-", "_")
        present = key in args
        out[cid] = dict(flag=switch.get("flag"), value=args.get(key), present=present,
                        on=bool(present and str(args.get(key)) in [str(v) for v in switch.get("on") or []]))
    return out


def render(run: str, *, max_snapshots: int = 12, watch: bool = False,
           every: float = 30.0, stats_from: Optional[str] = None) -> str:
    args = _load(os.path.join(run, "args.json")) or {}
    receipt, receipt_note = find_receipt(run)
    stats_dir = os.path.abspath(stats_from) if stats_from else os.path.abspath(run)
    same_run = stats_dir == os.path.abspath(run)
    # the charts' own run, when it is a different one: the page has to be able to say
    # which body each half describes, and --intention-every against --motor-hold is the
    # pair that decides who writes the orders the body executes
    stats_args = {} if same_run else (_load(os.path.join(stats_dir, "args.json")) or {})
    stats = read_stats(stats_dir)
    # the snapshots themselves are chosen LAST, once the rest of the page has been measured
    # and the bytes left for them are known; the newest one is read now because the charts
    # label each sense with how it arrived on that snapshot, and the newest is always kept
    entries = snapshot_entries(run)
    all_snap_steps = [e["step"] for e in entries]
    newest = _load_snapshot(run, entries[-1]) if entries else None
    # snapshot markers only when the charts and the snapshots share a step axis -
    # a probe's steps count from 0 and would land nonsense markers on a
    # training run's million-step axis
    series = build_series(stats, all_snap_steps if same_run else [],
                          modalities=receipt.get("active_modalities"),
                          sources=((newest.get("header") or {}).get("source_mask")
                                   if newest else None))
    tour = build_steps(receipt, args)
    # an eval-only probe resumes a training checkpoint: surface that step count
    trained_step = None
    m = re.search(r"ckpt_step_(\d+)", str(args.get("resume") or ""))
    if m and args.get("eval_only"):
        trained_step = int(m.group(1))
    doc = dict(SCH.DOC)
    doc.update(DOC_CHARTS)
    # P112: the prose used to hard-code "both bodies" from the two-body probe while the
    # training runs have 8; every {n_bodies} in a DOC string is the run's own count
    # P406: the moment's length was typed into the glossary ("5 steps of 10 ms", "50 ms").
    # It is now the receipt's MEASURED env step and model step with this run's own
    # --action-repeat and --motor-hold, from schema.moment_clock - the same source the
    # tour and the viewer read.
    clock = SCH.moment_clock(receipt, args)
    fill = SCH.moment_fill(receipt, args)
    # H84 sync (2026-09-19): the model's size is a launch setting (--d-model,
    # --core-blocks, --vision-swin-layers, --encoder-layers, ...), so every token width,
    # block count and learned-number count in the prose is a {placeholder} filled from
    # THIS run's receipt by schema.shape_fill; viewer.js reads the same numbers as
    # META.shape. A typed 128 or "2 blocks" described H81 on every later run's page.
    shape = SCH.model_shape(receipt, args)
    fill.update(SCH.shape_fill(receipt, args))
    # H84 items, second pass (2026-09-19): the planner's cadence against the order hold,
    # the imagination depth, the replay ring's size, each encoder's depth and width and the
    # vision encoder's reach are read or computed from THIS run (schema.run_fill); who wrote
    # the executed orders is the trainer's own tally in the stats log; and whether the
    # weights no lesson reaches have stayed put is measured on the run's own checkpoints.
    run_name = os.path.basename(os.path.normpath(run))
    exec_src = exec_source_summary(stats.get("mimo", []))
    idle_measured = measure_idle_groups(run, receipt)
    idle = idle_params(receipt, idle_measured)
    fill.update(SCH.run_fill(receipt, args, run_name,
                             stats_run=os.path.basename(os.path.normpath(stats_dir)),
                             exec_source=exec_src,
                             source_mask=((newest.get("header") or {}).get("source_mask")
                                          if newest else None)))
    fill.update(idle_fill(receipt, idle, run_name))
    fill.update(memory_bias_fill(run, run_name))
    fill.update(hself_wall_fill(stats.get("mimo", []), fill, run_name))
    fill.update(hself_now_fill(run, run_name))
    fill.update(entity_self_now_fill(run, run_name))
    nb = args.get("n_envs")
    if nb:
        fill["{n_bodies}"] = str(nb)

    def _fill(text: str) -> str:
        for key, val in fill.items():
            text = text.replace(key, val)
        return text
    doc = {k: {kk: (_fill(vv) if isinstance(vv, str) else vv)
               for kk, vv in v.items()} for k, v in doc.items()}
    # a chart caption is page text too: the re-study caption quotes the motor encoder's
    # share of the optimizer's numbers, which is this run's receipt, not a typed figure
    for ch in series.get("charts") or []:
        if isinstance(ch.get("caption"), str):
            ch["caption"] = _fill(ch["caption"])
    css = open(os.path.join(PKG, "viewer.css"), encoding="utf-8").read()
    js = open(os.path.join(PKG, "viewer.js"), encoding="utf-8").read()
    comp, comp_basis = component_params(run, receipt)
    selected_state = current_state(run, args, stats, SAVED, explanation_sources())
    meta = {
        "run": os.path.basename(run), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_label": local_time_label(time.time()),
        "watch": bool(watch), "every": every, "receipt": receipt,
        # where the receipt came from when not the run folder, or why there is none (find_receipt)
        "receipt_note": receipt_note,
        "trained_step": trained_step,
        "stats_source": os.path.basename(stats_dir),
        "component_params": comp, "component_params_basis": comp_basis,
        "idle_params": idle,
        # what the run's own checkpoints show about those idle weights, and what reading
        # them cost this build (None = no readable checkpoint: the page says unmeasured)
        "idle_measured": idle_measured,
        # the trainer's tally of who wrote the executed orders, from the charts' stats log
        "exec_source": exec_src,
        # how far one vision patch reaches at this run's layer count (schema.vision_reach)
        "vision_reach": SCH.vision_reach(receipt, args),
        # an ESTIMATE of the size of the token the recalled memory bias lands on, from the
        # newest checkpoint's final LayerNorm; the memory diagram uses it only when the
        # snapshots do not record the measured size, and calls it an estimate
        "memory_token_estimate": landing_token_estimate(run),
        # the six curriculum phases straight from lem/training/curriculum.py, so the
        # viewer never writes the phase names or their limits down a second time
        "phases": phase_table(clock["env_step_s"]),
        # one env step and one model step in seconds, measured by the run when its receipt
        # carries them; viewer.js reads every moment length from here
        "clock": clock,
        # the model's size from the run's own receipt (schema.model_shape): token width,
        # core blocks, context length and the sizes derived from them. viewer.js reads
        # every one of these from here and types none of them
        "shape": shape,
        # ONE list, PUBLISHED_ARGS, so a flag a sub-diagram names cannot be missing
        # from the page while the drawing prints it as a literal the build cannot
        # falsify (steps_per_phase printed a literal 3,000; the replay ring, the
        # curiosity band and the imagination depth were all literals too).
        "args": {k: args.get(k) for k in PUBLISHED_ARGS},
        "args_recorded": {k: k in args for k in PUBLISHED_ARGS},
        "current_state": selected_state,
        # When the charts come from another run, publish THAT run's flag values as
        # well. META.args describes the --run directory (the snapshots); without
        # this second block the page printed one run's configuration beside another
        # run's history with nothing saying so, and the two disagree - H81 against
        # H82: n_envs 2 vs 8, start_phase mature vs neonate, eval_only true vs
        # false, intention_every 5 vs 7. None when both halves are the same run.
        "stats_args": (None if same_run
                       else {k: stats_args.get(k) for k in PUBLISHED_ARGS}),
        # which snapshots this page carries and what limited the count, so the inspector
        # can say so beside its picker; assemble() fills the embedded half once the
        # snapshots have been chosen
        "n_snapshots_embedded": 0, "n_snapshots_on_disk": len(all_snap_steps),
        "snapshots": {"on_disk": len(all_snap_steps), "embedded": 0, "embedded_steps": [],
                      "limited_by": None, "budget_mb": PAGE_BUDGET_MB,
                      "max_snapshots": max_snapshots},
        # the run the glossary's "this run" sentences were written about; the viewer
        # says so in every popup that uses the phrase, because DOC is written once and
        # shipped on every page, whichever run the page is built for
        "doc_written_for": getattr(SCH, "DOC_WRITTEN_FOR", None),
        # the gain-of-function build (2026-09-22): the probe results of this lineage
        # (Saved/mimo_bridge/gof_probes/index.jsonl, DESIGN.md section 9), the liveness
        # measurements (docs/gain_of_function_v3/measurements/) and the causal chain's
        # resolved orders (<run>/causal_chain.jsonl) - each read when the page is built
        "probes": read_probes(run),
        "liveness": read_liveness(),
        "causal_events": read_causal_events(run),
        # where the efference heads' readings reach the planner in this run (the goal-ranking
        # channels, the action search): viewer.js draws the lines out of the efference box from this,
        # and walkthrough.py builds the tour step from the same schema.efference_routes
        "efference_routes": SCH.efference_routes(receipt, args),
        # each repair's switch in THIS run (repair_switches): the orange marks under the model diagram read it
        "repair_switches": repair_switches(args),
    }
    # the header's figure carries its basis: the diagram's footer prints a second,
    # larger total (saved numbers, frozen buffers included), and a bare number here
    # left the reader with two model sizes and nothing telling them apart.
    # "adjustable", not "learned": total_params counts every registered parameter,
    # and the footer names the ones inside it that no lesson reaches (the voice head
    # and the gaze head here), so calling all of them learned contradicts that line.
    total_params = receipt.get("total_params")
    saved_total = sum(comp.values()) if comp else None
    if total_params and saved_total and saved_total != total_params:
        pcount = (f"{total_params:,} adjustable numbers "
                  f"({saved_total:,} saved, frozen buffers included) · ")
    elif total_params:
        pcount = f"{total_params:,} adjustable numbers · "
    else:
        pcount = ""
    components = component_view()
    investigations = investigation_view()
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    elim_css = open(os.path.join(tools_dir, "elimination_diagram.css"), encoding="utf-8").read()
    elim_js = open(os.path.join(tools_dir, "elimination_diagram.js"), encoding="utf-8").read()

    def assemble(used: List[dict], limited_by: Optional[str]) -> str:
        n_all = len(all_snap_steps)
        meta["n_snapshots_embedded"] = len(used)
        meta["snapshots"].update(embedded=len(used), limited_by=limited_by,
                                 embedded_steps=[s["header"]["step"] for s in used])
        why = {"budget": f" (the first and the newest always, the rest thinned evenly to keep "
                         f"the page inside its {PAGE_BUDGET_MB:g} MB budget)",
               "max_snapshots": f" (the first and the newest always, the rest thinned evenly "
                                f"to the --max-snapshots cap of {max_snapshots})"}
        sub = (f"{meta['run']} · {len(used)} of {n_all} snapshots embedded"
               f"{why.get(limited_by, '')} · "
               f"{pcount}"
               f"generated {meta['generated_label']}"
               # the title line named the snapshot run only, so a page whose Run tab
               # charts came from elsewhere read as one run throughout
               + ("" if same_run else
                  f" · snapshots and flags from this run; the Run tab's charts are "
                  f"{meta['stats_source']}, a different run")
               + (" · live (reloads when idle)" if watch else ""))
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>LEM explainer — {meta['run']}</title>",
            f"<style>{css}\n{elim_css}</style></head><body>",
            f'<header><h1>LEM model explainer</h1><span class="sub">{sub}</span><a href="http://127.0.0.1:8766/" style="margin-left:auto">Development dashboard</a></header>',
            current_state_html(selected_state),
            '<nav class="tabs" id="tabs"></nav>',
            "<main>",
            '<section id="tab-arch">'
            + (f'<p class="note" id="receipt-note">{_html_text(meta["receipt_note"])}</p>'
               if meta.get("receipt_note") else "")
            # the learning circle, then the model diagram's own panel: its heading, the tour of it and the drawing
            # (the decided design of the Architecture tab, Oct 7, 2026, D-E1); viewer.js writes the heading and the
            # list of the parts being worked on into the panel
            + '<div id="learnloop"></div>'
            '<div id="model-diagram"><div id="tour"></div><div id="diagram"></div></div>'
            '<div id="detail" hidden></div></section>',
            '<section id="tab-inspect" class="hidden"><div id="inspector"></div></section>',
            '<section id="tab-run" class="hidden"><div id="runtab"></div></section>',
            '<section id="tab-probes" class="hidden"><div id="probestab"></div></section>',
            "</main>",
            "<footer>every metric popover names the exact field it reads; "
            "the schema lives in mimo_bridge/explainer/schema.py</footer>",
            _json_block("data-meta", meta),
            _json_block("data-snapshots", used),
            _json_block("data-series", series),
            _json_block("data-tour", tour),
            _json_block("data-doc", doc),
            _json_block("data-components", components),
            _json_block("data-investigations", investigations),
            f"<script>{elim_js}</script>",
            f"<script>{js}</script>",
            "</body></html>",
        ]
        return "\n".join(parts)

    # THE PAGE KEEPS ITSELF INSIDE ITS BUDGET. Everything but the snapshots is measured
    # first; the snapshots then get the bytes that are left. 4 KB of slack covers the few
    # bytes by which the header and META differ between the two assemblies.
    rest = len(assemble([], None).encode("utf-8"))
    kept, _steps, limited = _pick_snapshots(run, max_snapshots,
                                            PAGE_BUDGET_MB * 1e6 - rest - 4096)
    if limited == "budget":
        log.info("page budget %.0f MB: %d of the run's %d snapshots embedded "
                 "(--max-snapshots %d)", PAGE_BUDGET_MB, len(kept), len(all_snap_steps),
                 max_snapshots)
    return assemble(kept, limited)


# THE PAGE'S OWN BUILD TIME IS NOT A CHANGE. generated_at is stamped to the second, so two builds of
# identical content a second apart differed in those bytes alone: --watch rewrote an idle page on
# every cycle, and the change-gate test failed whenever its two builds straddled a second (review of
# 2026-09-25, finding 13). The page on disk keeps the time its content was generated.
_BUILD_TIME = re.compile(rb'"generated_at":\s*"([^"]*)"')   # the page's meta block is compact JSON
_BUILD_LABEL = re.compile(rb'"generated_label":\s*"([^"]*)"')


def _same_but_build_time(old: bytes, new: bytes) -> bool:
    a, b = _BUILD_TIME.search(old), _BUILD_TIME.search(new)
    if not (a and b):
        return False
    old = old.replace(a.group(1), b.group(1))
    a, b = _BUILD_LABEL.search(old), _BUILD_LABEL.search(new)
    if a and b:
        old = old.replace(a.group(1), b.group(1))
    return old == new


def write_atomic_if_changed(path: str, content: str) -> bool:
    data = content.encode("utf-8")
    try:
        if os.path.exists(path) and os.path.getsize(path) == len(data):
            with open(path, "rb") as fh:
                old = fh.read()
            if old == data or _same_but_build_time(old, data):
                return False
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return True


def build(run: str, out: Optional[str], max_snapshots: int, watch: bool,
          every: float, stats_from: Optional[str] = None, run_copy: bool = True) -> List[str]:
    """Render the page for `run` and write it. TWO TARGETS by default: <run>/explainer.html and `out` (default
    Saved/mimo_bridge/explainer.html). run_copy=False (--no-run-copy, the output-destination decision of Oct 15,
    2026, arm1-failure-reconciled-output-fence-20261015) writes ONLY `out`, and refuses before rendering an output that
    resolves inside the run folder or the --stats-from run folder under any spelling (refresh.output_inside_run: the
    string test and the file-identity test; an identity that cannot be read refuses): the watcher of 9:49 AM that day
    rewrote <B1K run>/explainer.html inside a protected predecessor folder during an experiment's before/after fence.
    The input runs are then only read."""
    target = out or os.path.join(SAVED, "explainer.html")
    if not run_copy:
        from mimo_bridge.explainer.refresh import inputs_holding
        held = inputs_holding(target, run, stats_from)
        if held:
            raise ValueError("refused: the output %s is inside the input run %s; --no-run-copy keeps the input runs "
                             "read-only" % (target, held[0]))
    page = render(run, max_snapshots=max_snapshots, watch=watch, every=every,
                  stats_from=stats_from)
    mb = len(page.encode("utf-8")) / 1e6
    if mb > PAGE_BUDGET_MB:
        # render() already thinned the snapshots down to the newest one, so what is over
        # budget here is the page WITHOUT its snapshot history - the charts, the text or
        # one snapshot alone - and no snapshot count can fix it
        log.warning("page is %.1f MB, over the %.0f MB budget with the snapshots already "
                    "thinned to the fewest the page keeps - the rest of the page, or the "
                    "newest snapshot alone, is over budget", mb, PAGE_BUDGET_MB)
    targets = ([os.path.join(run, "explainer.html")] if run_copy else []) + [target]
    written = []
    for t in targets:
        if write_atomic_if_changed(t, page):
            written.append(t)
    log.info("page %.2f MB -> %s%s", mb, ", ".join(written) or "(unchanged)",
             "" if written else " [skip: identical bytes]")
    return written


def _sources_stamp(run: str, stats_from: Optional[str] = None):
    stamp = []
    paths = [os.path.join(run, "replay_training_stats.jsonl"),
             os.path.join(run, "explain", "index.json"),
             os.path.join(run, "args.json"),
             os.path.join(run, "control.json"),
             os.path.join(run, "checkpoints"),
             os.path.join(SAVED, "training_continuation.json"),
             os.path.join(SAVED, "supervisor_control.json"),
             # the fourth tab and the causal-chain panel read these; without them in the stamp
             # a new probe result or a newly resolved order never rebuilt the page
             os.path.join(run, "causal_chain.jsonl"),
             PROBE_INDEX,
             os.path.join(LIVENESS_DIR, "liveness_over_lineage", "README.txt")]
    # the code and QA map's files (component_view): a new manifest entry, review or defect rebuilds the page; a commit
    # or a test run does not, and a page the dashboard serves reads those at the moment it is opened
    paths.extend(os.path.join(ROOT, "..", "..", "docs", "gain_of_function_v4", name) for name in
                 ("component_manifest.json", "component_qa_log.jsonl", "defect_register.json"))
    # the investigations (investigation_view): a new or changed investigation file rebuilds the page; a test run that
    # changes a box does not, and a page the dashboard serves reads the boxes at the moment it is opened
    inv_dir = os.path.join(ROOT, "..", "..", "docs", "gain_of_function_v4", "investigations")
    try:
        paths.extend(sorted(os.path.join(inv_dir, n) for n in os.listdir(inv_dir) if n.endswith(".json")))
    except OSError:
        paths.append(inv_dir)
    paths.extend(explanation_sources())
    if stats_from:
        # the charts' own run: without its log in the stamp a --stats-from page never
        # rebuilt when the training run it charts wrote new records
        paths.append(os.path.join(os.path.abspath(stats_from), "replay_training_stats.jsonl"))
    for p in paths:
        try:
            st = os.stat(p)
            stamp.append((p, st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append((p, 0, 0))
    return tuple(stamp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run", default=None,
                    help="run dir (default: supervisor's selected production run, else newest run)")
    ap.add_argument("--watch", action="store_true",
                    help="follow the selected run and rebuild from fresh source on changes; retry failures")
    ap.add_argument("--watch-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--need-gb", type=float, default=2.0,
                    help="memory admission for each watch rebuild (default 2 GiB)")
    ap.add_argument("--every", type=float, default=30.0, help="watch poll seconds")
    ap.add_argument("--max-snapshots", type=int, default=12,
                    help="the most snapshots the page may embed; fewer are embedded when "
                         "that many do not fit the 6 MB page budget")
    ap.add_argument("--out", default=None,
                    help="second output path (default Saved/mimo_bridge/explainer.html)")
    ap.add_argument("--no-run-copy", action="store_true",
                    help="write ONLY the --out page (default Saved/mimo_bridge/explainer.html), never the run's own "
                         "<run>/explainer.html, and refuse an output inside the run folder; --watch passes it to "
                         "every rebuild")
    ap.add_argument("--stats-from", default=None,
                    help="read the run-replay charts from THIS run dir instead of "
                         "--run (use the training run here when --run is a probe; "
                         "snapshot markers are dropped since the step axes differ)")
    a = ap.parse_args(argv)
    def choose_run():
        return os.path.abspath(a.run) if a.run else selected_production_run(SAVED) or newest_run()

    # Watch selection errors belong inside the retry loop, including at startup.
    run = choose_run() if not a.watch else None
    if not a.watch and (not run or not os.path.exists(os.path.join(run, "args.json"))):
        print("no run found (need a directory with args.json)", file=sys.stderr)
        return 2
    from mimo_bridge.explainer.refresh import refresh_loop, output_lock, inputs_holding
    target = os.path.abspath(a.out or os.path.join(SAVED, "explainer.html"))
    if a.no_run_copy:
        # before the watch lock or any page is written: the lock and the receipt sit beside the output. A run that
        # cannot be chosen now (the production record unreadable, no run found) refuses too, so nothing is written
        # beside an output the fence has not judged (the independent recheck of Oct 15, 2026, finding A-m1)
        try:
            first = choose_run()
        except Exception as exc:
            print("refused: --no-run-copy needs the input run to judge the output against, and choosing it failed: "
                  "%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
            return 2
        if not first:
            print("refused: --no-run-copy needs the input run to judge the output against, and none was found",
                  file=sys.stderr)
            return 2
        held = inputs_holding(target, first, a.stats_from)
        if held:
            print("refused: the output %s is inside the input run %s; --no-run-copy keeps the input runs read-only"
                  % (target, held[0]), file=sys.stderr)
            return 2
    if not a.watch:
        log.info("run: %s", run)
        build(run, a.out, a.max_snapshots, a.watch_child, a.every, stats_from=a.stats_from,
              run_copy=not a.no_run_copy)
        return 0
    # Hold an OS lock, not just a PID file: a crashed refresher cannot leave a
    # false owner behind, and two sessions cannot race to replace the main page.
    with output_lock(target + ".refresh.lock"):
        refresh_loop(a, choose_run, _sources_stamp, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
