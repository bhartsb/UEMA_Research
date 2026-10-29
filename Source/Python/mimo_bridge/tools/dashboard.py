# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Legacy training dashboard: one HTML page, rewritten every 30 s.

    python -m mimo_bridge.tools.dashboard            # writes Saved/mimo_bridge/dashboard_legacy.html
    python -m mimo_bridge.tools.dashboard --once

Open Saved/mimo_bridge/dashboard_legacy.html in a browser; it refreshes itself.
The development dashboard owns the dashboard.html entry point.

Every value on the page is computed by this script from the run's own files -
replay_training_stats.jsonl, args.json, preflight.json, checkpoints/*/lineage.json,
causal_test.json - at render time. Nothing numeric is typed into the page by hand;
the one free-text channel (--note) is labelled as an operator note.

What it shows, for every run folder under Saved/mimo_bridge/experiments and runs/:
  * a chip strip with the numbers that decide anything: code provenance, action-ledger
    coverage, causal next-state error over report windows, command following, the
    five-control causal test, and the gate count
  * provenance: which commit the trainer actually loaded, vs the current source HEAD
  * lineage steps vs new steps this run; folder age vs the trainer's own clock
  * causal next-state error and reconstruction error, compared over the first and last
    K-report windows (single-record endpoints can flip the sign of a trend)
  * every gate and setting the run was launched with, and its state
  * behaviour telemetry, labelled as the proxies they are
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os
import re
import subprocess
import time

from mimo_bridge.tools.plain_numbers import clean, fixed, pct, per

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(SRC, "..", ".."))
SAVED = os.path.join(REPO, "Saved", "mimo_bridge")
# The original renderer remains available without overwriting the new live entry.
OUT = os.path.join(SAVED, "dashboard_legacy.html")

# each sense as the model's registry spells it (the operator's rule: never rename a part); the page's
# reading legend defines each from mimo_bridge.explainer.schema.SENSE_DEFINITIONS
SENSE_NAMES = {"vision": "vision", "tactile": "tactile", "proprioception": "proprioception", "audio": "audio",
               "intermind": "intermind", "motor": "motor", "vestibular": "vestibular"}


def sense_names_legend() -> str:
    """One sentence defining every sense name the page uses, in the model's own words."""
    from mimo_bridge.explainer.schema import SENSE_DEFINITIONS
    return ("<b>Sense names</b> are the model's own: "
            + "; ".join(html.escape(f"{m} is {d}") for m, d in SENSE_DEFINITIONS.items()) + ".")
PHASE_WORDS = {"neonate": "newborn", "early_infant": "early infant", "late_infant": "late infant", "toddler": "toddler",
               "child": "child", "mature": "grown"}
# The meta-overseer's goal trend, named for what it is. The discrimination ratio is
# between-body spread over within-body variance: it rises when a body moves LESS, and
# the runs it compares differ in body count, scene, checkpoint and duration. Older
# passes wrote the claim-words "improving"/"not improving"; a direction is all the
# comparison supports, so cached values are rewritten rather than reprinted.
_RATIO = "parallel-body action-signature ratio (uncalibrated proxy)"
TREND_LABEL = {"roll_deg_max": "largest trunk rotation seen",
               "action_signature_ratio": _RATIO, "discrimination": _RATIO}
TREND_WORD = {"improving": "higher than the earliest unmatched run",
              "not improving": "not higher than the earliest unmatched run"}

# Switch gates: things that are on or off. (name, what it does when on.)
GATE_DEFS = [
    ("intention", "chooses actions by optimising imagined outcomes"),
    ("imagination", "rolls the world model forward before acting"),
    ("replay", "re-studies stored model steps from the buffer"),
    ("next_frame_loss", "the causal lesson: state + executed action predicts the NEXT state"),
    ("top_down_gate", "wellness biases which sense gets attention"),
    ("babble", "endowed random motor babble"),
    ("random_actions", "control condition: ignores the model, sends random actions"),
    ("gaze_reflex", "endowed snap-to-motion eye reflex"),
    ("spontaneous_saccades", "endowed random eye movements"),
    ("movement_drive", "stillness builds a debt that pushes the body to move"),
    ("body_schema", "the body map's co-movement number (command changes and proprioception changes in the same "
                    "model step; not causation) multiplies the movement drive's bonus"),
]
# Settings: numeric or named configuration, listed with their value.
SETTING_DEFS = [
    ("motor_hold", "each command stays in force for this many model steps"),
    ("smooth_actions", "low-pass factor on outgoing commands (0 = none)"),
    ("curiosity_band", "prediction-error band where the curiosity drive is nonzero"),
    ("wellness_mode", "which wellness formula grades each model step"),
    ("rom_limit", "developmental range-of-motion limit right now"),
    ("strength_limit", "developmental strength limit right now"),
    ("delays", "(sensory, motor) delay in steps"),
]

# ---------------------------------------------------------------------------------
# EXPLANATIONS. Every chip and row on the page carries one, shown when it is clicked.
#
# Each entry states what the number IS, HOW it is computed and from which field, and
# WHAT WOULD FOOL IT. The last is not decoration: most of this project's wrong turns
# were a number that was correct and meant something other than its label.
#
# `use` and `clear` are a DESIGN review of the row on a 1-10 scale, not measurements.
# Anything below 8 on either was rewritten or moved to the secondary panel; the test
# suite asserts none is left below 8. They are deliberately not rendered - they say
# nothing about the run.
# ---------------------------------------------------------------------------------
DOC = {
 'chip:code': dict(t='Which code produced these weights',
   w="The commit the trainer loaded at launch, from the newest checkpoint's lineage.json,"
     ' compared with the current source HEAD.',
   h="lineage.json 'code_commit' vs `git rev-parse HEAD`.",
   f='A clean working tree does not prove an hours-old process loaded it. H28 ran commit'
     ' 7ac7ac3a while HEAD was 790fd7db - eight commits later - so every repair in between'
     ' was absent from the run being judged by it. If this chip is red, the results you are'
     ' reading came from different code than the one you are editing.',
   use=10, clear=9),
 'chip:ledger': dict(t='Did the executed action reach the model',
   w='The share of steps on which the model was shown the command the body actually carried'
     ' out, rather than the one that happened to arrive with the frame.',
   h='contract.exec_seen divided by the lineage step when the counter spans the lineage (it'
     ' exceeds this segment\'s own steps, as on H84), otherwise by the steps since the resume point.',
   f='100% here says the plumbing works and nothing more. H28 supplied the executed action'
     ' on 99.95% of 124,483 steps and the causal test still could not tell that action from'
     ' a shuffled one. Delivery is not use.',
   use=10, clear=8),
 'chip:causal': dict(t='Causal prediction error, and which way it is moving',
   w='How wrong the model is about the NEXT model step given the current state and the'
     ' executed action, averaged over the three real senses.',
   h='next_pe for vision, tactile and proprioception, averaged; first quarter of this'
     " process's records against the last quarter.",
   f="Three ways. A still body makes it easy - predicting 'nothing changes' scores well: on"
     " H28's nearly still body the model's proprioception error was 48 times the error of predicting"
     ' that nothing changes (causal test of 2026-08-26). It can rise simply'
     ' because the body started moving, which genuinely makes the world harder to predict.'
     ' And it averages three senses that move in opposite directions. Read it beside body'
     ' speed and the per-sense rows, never on its own.',
   use=9, clear=8),
 'chip:follow': dict(t='Does the body move the way it was told',
   w='Direction agreement between the command and the motion that followed, over the whole'
     ' life of each command. +1 exactly as told, 0 unrelated, negative means opposed.',
   h="cmd_response_cos_held of the newest record: the mean over the commands whose 250 ms ended since"
     " the record before it. Shown only when the newest 72 records carry no follow_bout (a run"
     " without movement bouts).",
   f='It ignores SIZE completely. A body creeping 5% of the way in the right direction'
     ' scores the same as one moving the whole way. The magnitude row is the other half,'
     " and reading either alone is how '5% of the turn happens' got reported while the"
     ' direction was negative.',
   use=9, clear=9),
 'chip:follow_bout': dict(t='Does the body turn the way it was told, in movement bouts',
   w='Direction agreement between the command in force and the turn the body made during ONE 10 ms'
     ' physics frame: the last frame of each 50 ms model step, so 5 readings across each 250 ms'
     ' command, taken only in movement bouts. 100% = exactly as commanded, 0% = the turn is unrelated'
     ' to the command, below 0% = against it. It is not the whole command\'s 250 ms reading: in 10 ms a'
     ' limb turns only part of the way, so the chip sets it against a posture the body really reached'
     ' on physics alone, read on these same frames.',
   h="follow_bout from each record: train_mimo.py takes env.py's one-frame gauge (cmd_response.cos)"
     " on the last frame of each model step's burst, averages it over the bodies, and keeps a running"
     " average over movement-bout steps in which each new step counts 2 in 100. The chip is the mean"
     " over the newest 72 records since the last restart. The watcher's stop rule on bout obedience"
     " is judged on this field. Reference: the physics-only one-frame reading of a reachable posture,"
     " lineage_series FOLLOW['frame_reachable'] from measurements/following_out_of_100.json.",
   f='It ignores SIZE: a body creeping 5% of the way in the right direction scores the same as one'
     ' moving the whole way. Its horizon is 10 ms, so it cannot be set against the 250 ms'
     ' reachable-posture goal line. Until 2026-09-23 this chip carried the 250 ms help text and the'
     ' over-time page drew the 250 ms goal line under this series, so the distance it printed was the'
     ' gap between two horizons, not a following gap.',
   use=9, clear=9),
 'chip:causaltest': dict(t='The five-control causal test',
   w='Whether the correct executed action predicts the next model step better than'
     ' persistence, state-only, a shuffled action, a cross-body action, and a matched'
     ' passive body - on every seed, not on the average.',
   h='mimo_bridge.tools.causal_test, read from causal_test.json in the run folder.',
   f='It can only resolve what the data contains. If the body moves less than the seed noise can'
     " resolve, or the encoder's command columns are zero, the answer is fixed before the test runs - which"
     ' is why it reports INCONCLUSIVE (the gap was smaller than the seed noise) and INERT'
     ' (the pathway cannot carry anything) separately from FAIL.',
   use=10, clear=8),
 'chip:reconfault': dict(t='Is the reconstruction head intact',
   w='Whether the decoders that re-describe the current state are still working. On a'
     ' run WITHOUT residual prediction, damage shows as reconstruction error above'
     ' next-state error (describing the present cannot be harder than predicting the'
     ' future). WITH residual prediction on, next-state error collapses by construction'
     ' - the heads predict the change and get the present for free - so reconstruction error above'
     ' next-state error is expected there (1.16 to 2.45 times next-state error on the undamaged runs'
     ' H36 and H43), and damage shows'
     " instead as a jump past 4 times the run's own recent value.",
   h='Windowed decoder reconstruction error (recon_dec) per real sense: against windowed'
     ' next-state error (residual off), or against its own first-window value (residual'
     ' on, read from args.json).',
   f="Nothing about the model's LEARNING - a run can pass this check and still"
     ' have learned nothing about actions. It only catches destruction. Measured on undamaged runs'
     ' with residual prediction off, reconstruction error per 100 parts of next-state error: H22'
     ' 36-39, H28 26-59, H31 26-99. With it on: 1.16 times next-state error on H36 and 2.45 times'
     ' next-state error on H43 (the decoders trained at their floor weight, 2 in 100 of the loss).'
     ' Measured broken, residual off: H30 at 5.3 to 169 times next-state error.',
   use=10, clear=9),
 'chip:drift': dict(t='Drift on the channels nothing corrects',
   w="How far the two fake senses' error has moved since the run started.",
   h="Reconstruction error for audio and intermind, newest against the run's first"
     ' record.',
   f='These are not learning signals and never improve - the point is the opposite. They'
     " carry no loss weight, so their error tracks the model's own output scale rather"
     ' than any skill, which makes them the FIRST place drift becomes visible. On H30'
     ' intermind hit 110 times its value before the damage while the three real senses had not'
     ' yet risen; they followed an hour later.',
   use=9, clear=8),
 'chip:gates': dict(t='How many switches are on',
   w="A count of the run's switch-type gates that are on versus off.",
   h='args.json plus any gates recorded in the log; the full list is the Gates panel'
     ' below.',
   f='A count is not a configuration. Two runs with the same count can differ on the one'
     ' switch that decides everything: with intention on the body moved at 28 to 43 cm/s (H20-H22),'
     ' with it off at 3.6 to 11 cm/s (H23-H28), and'
     ' imagination - which changes no behaviour whatsoever - destroyed a run through the'
     ' training loss. Open the panel and read the switches.',
   use=8, clear=8),
 'chip:cmdpath': dict(t='Can the model read each body part\'s command',
   w='One body part at a time: how far a realistic 20-degree turn of ONLY that part\'s command moves'
     ' the motor token (the vector through which the model reads its own command), as a share of the'
     ' token\'s size, against a line measured on ckpt_step_344400. Red when any part is under the'
     ' line: the model cannot read that part\'s command at all.',
   h='tools/probes.py per_handle_readability on the newest checkpoint\'s saved motor encoder, fed 16'
     ' lived rows recorded in segment 31\'s world (docs/gain_of_function_v3/measurements/'
     'motor_rows_seg31_ckpt_step_344400.npz), 3 turn axes per part. The line (29 millionths of the'
     ' token\'s size) sits in the middle of the gap measured on ckpt_step_344400, where 10 of 26 parts'
     ' moved the token by 6 to 40 parts in 10 million and the next part 211 millionths. A checkpoint of another'
     ' body layout has no lived rows, and the chip says per-body-part readability is unmeasured.',
   f='The earlier test changed all 26 body parts at once, with random full rotations, and read green'
     ' while the encoder\'s pool read one part per head and gave the eyes, the head and 14 other'
     ' parts zero weight (WORKLOG 2026-09-22 20:30): a few read parts move the token a lot. It says'
     ' the encoder CAN see a command; whether the forward model uses it is the causal test.',
   use=10, clear=9),
}


# the motor encoder input layout: 10 achieved columns then 9 command columns
ACHIEVED, MOTOR_IN = 10, 19

# Row explanations, matched by a distinctive fragment of the row's label. SECONDARY
# rows scored below 8 for usefulness and are moved to a collapsed panel rather than
# deleted - everything measured stays on the page, ranked rather than hidden.
ROW_DOC = {
 'Status': dict(t='Run state, and two different clocks',
   w='Whether the process is alive, how long THIS process has run, and how old the run'
     ' folder is.',
   h="Liveness from the log's mtime; process time from elapsed_s in the newest record;"
     ' folder age from its creation time.',
   f='They are not the same number. A folder reused across launches is older than the'
     ' process writing to it, and the old page showed only the folder age.',
   use=9, clear=9),
 'How many babies': dict(t='Parallel bodies and the scene',
   w='How many MIMo bodies run at once, and which scene they are in.',
   h='args.json n_envs and scene.',
   f='Body count changes what every rate on this page means, and the between-body ratio'
     ' depends on it entirely.',
   use=8, clear=9),
 'Learning steps (checkpoint lineage)': dict(t="The brain's whole life, not this run",
   w='The step counter carried forward through every run this checkpoint descends from.',
   h="The newest record's step field.",
   f="It looks like this run's progress and is not. H28 displayed 825,600 while having"
     ' produced 124,483 of them.',
   use=8, clear=9),
 'New steps in this run': dict(t='What this process actually produced',
   w='Steps taken since the checkpoint this run resumed from.',
   h='Newest step minus the step parsed out of the --resume path.',
   f='Nothing - this is the number to judge a run by. But it counts steps taken, not steps'
     ' worth keeping: a run that diverged halfway still reports all of them.',
   use=9, clear=9),
 'Model transitions per second': dict(t='How many model steps run per second',
   w='How many times per second the model observes a state and takes a learning step, per'
     ' body and across all bodies.',
   h='model_steps_per_s from the log; the across-bodies figure multiplies by n_envs.',
   f="Not the simulator's frame rate - the model observes once per action-repeat interval,"
     ' which is 5 physics frames.',
   use=8, clear=9),
 'Simulator physics frames': dict(t='The physics rate, which is NOT the model rate',
   w='How many 10 ms physics frames the simulator computes per second per body.',
   h="model_steps_per_s multiplied by the run's action_repeat.",
   f='This row exists to stop a 5x confusion: the page used to print this number and call'
     ' it sense-snapshots the model processed.',
   use=8, clear=9),
 'Speed vs real time': dict(t='Simulated life per second of wall clock',
   w='How fast the bodies live compared with real time, per body and in total.',
   h='Physics frames per second multiplied by the 10 ms timestep.',
   f='The aggregate counts parallel bodies, so 8 slow bodies can exceed real time while'
     ' each individual body is far slower than an infant.',
   use=8, clear=8),
 'Age of the body': dict(t='Developmental phase, which gates the babble',
   w='The curriculum phase and the corresponding MIMo age.',
   h='phase from the log, age_months alongside it.',
   f='Not cosmetic, and it arrives sooner than it looks. The mature phase sets exploration'
     ' noise to zero, so reaching it switches motor babbling off entirely - and at'
     ' steps_per_phase 3000 that happens by step 14,999. Seven consecutive runs trained a'
     ' frozen body this way.',
   use=9, clear=8),
 'Guessing the next instant of': dict(t='Causal prediction error, per sense',
   w='How wrong the model is about the NEXT model step of this sense, given the current state'
     ' and the action the body actually executed. Lower is better.',
   h="next_pe for the sense; first quarter of THIS process's records against the last"
     ' quarter - never single records, never across a restart.',
   f='A still body makes this easy, and a moving one genuinely makes it harder - so a rise'
     ' is not automatically bad news. What it cannot tell you is whether the ACTION helped;'
     ' only the five-control causal test separates predicting-the-future from knowing-what-'
     ' you-did.',
   use=10, clear=9),
 'all real senses': dict(t='Causal error across the three real senses',
   w='The three real senses averaged. Audio and intermind are excluded: they are'
     ' synthetic on MIMo and would flatter the average.',
   h='Mean of the per-sense next_pe window figures.',
   f='An average hides opposite movements. On H30 proprioception improved while vision and'
     ' tactile worsened; the average alone would not show that.',
   use=9, clear=8),
 'FAULT - re-describing': dict(t='The reconstruction decoders are damaged',
   w='The decoders that re-describe the state the model is already looking at have'
     ' failed their health rule. Residual off: their error exceeded next-state error,'
     ' an ordering a working model cannot produce - the present is right there in the'
     ' input. Residual on: their error jumped more than 4x against its own recent value,'
     ' the H30 damage signature. The row text carries the measured numbers.',
   h='Windowed DECODER reconstruction error (recon_dec per sense; on logs from before'
     ' 2026-08-27, the decoders\' mean loss from the training records) against windowed'
     ' next-state error for the same sense.',
   f='It says the decoders are broken, not why. On H30 the cause was imagination feeding its'
     ' own guesses about unpredictable channels back into itself. And before 2026-08-27 this'
     ' rule read `pe`, which is predicted change, not reconstruction: it fired on H34 the'
     ' model step the model began predicting movement, with next-state error the best on record.'
     ' That was the instrument, not the model.',
   use=10, clear=9),
 'Untrained-channel drift': dict(t='The earliest sign the model is coming apart',
   w='Reconstruction error on audio and intermind, which have no MIMo source and are'
     ' filled with fresh noise every step.',
   h="Their pe - the heads' output on channels that receive no gradient, so it tracks the"
     " trunk's output scale - newest record against the run's first.",
   f='Read it backwards from every other row. These channels carry no loss weight and'
     ' receive no gradient, so a rising number is not the model failing to learn them - it'
     " is the model's own output scale growing. On H30 intermind reached 110 times its value"
     ' before the damage while vision, tactile and proprioception had not yet risen, and they only'
     ' followed an hour later.',
   use=10, clear=8),
 'Re-describing': dict(t='Reconstruction error - a diagnostic, NOT evidence about actions',
   w='How well the reconstruction decoders re-describe the state the model is already'
     ' looking at.',
   h='recon_dec for the sense - the decoders\' error - over the same windows. The decoders'
     ' train at a small scheduled weight (a scaffold that fades to a smoke-detector floor),'
     ' so after a change to the trunk they re-adapt slowly.',
   f='Two ways. It says nothing about whether the model learned what actions do - a still'
     ' body drives it low, which reads as progress and is not. And a low value only means'
     ' something next to the next-state error: once it climbs ABOVE that, the decoders are'
     ' broken rather than merely worse, and this row is replaced by a FAULT row. Do not'
     ' confuse it with the predicted-change row, which is the heads, not the decoders.',
   use=8, clear=8),
 'Predicted change': dict(t='How far the live guess sits from the present',
   w="The distance between the model's guess of the NEXT instant and the instant it is"
     ' looking at now.',
   h='pe for the sense: the prediction heads applied to this model step, scored against this'
     ' model step. With future-only heads that is next-guess minus present.',
   f='Read it as a copy detector in reverse. Near zero means the model predicts that'
     ' nothing will change - a still body, or a model that has learned to copy the present.'
     ' Rising means it predicts movement. It is not an error and must never feed a fault'
     ' rule: on H34 it did, and the page reported reconstruction broken while next-state'
     ' error was the best on record.',
   use=8, clear=8),
 'Does the body move the way it was told': dict(t="Direction agreement, over a command's life",
   w='Whether the body turned the way it was told. +1 exactly as told, 0 unrelated,'
     ' negative means it moved against the command on average.',
   h='cmd_response_cos_held - each command scored across the whole time it is in force.',
   f='It ignores size entirely. The magnitude row below is the other half, and the two'
     ' were blended into one misleading sentence before.',
   use=10, clear=9),
 'How much of the asked turn': dict(t='Magnitude only - says nothing about direction',
   w='What fraction of the commanded rotation actually happens.',
   h='cmd_response_ratio, per 10 ms slice.',
   f='A body creeping the wrong way scores the same as one creeping the right way. Reading'
     " this without the direction row above is how '5% of the turn happens' got reported"
     ' while the direction was negative.',
   use=8, clear=9),
 'Mechanical probe': dict(t='Plumbing health at launch, not current performance',
   w='A single held command applied to a quiet body during the launch preflight.',
   h='preflight.json, the limb-follows-a-command check.',
   f='A fixed probe from launch. It was previously quoted inline as if it were the trained'
     " controller's live score. With no preflight the row disappears rather than falling"
     ' back to a constant.',
   use=8, clear=8),
 'Reach task': dict(t='Distance from hand to ball',
   w='How far the nearest hand is from the target, and how often it touches.',
   h='task_hand_ball_dist_m and task_ball_touch_rate.',
   f='Distance can fall because the caregiver replaced the ball nearby, not because the'
     ' hand reached for it. The touch rate is the harder number and it has been 0% on every'
     ' run so far.',
   use=9, clear=9),
 'Hand touching the ball': dict(t='Contact, which is not reaching',
   w='Share of recent checks where a HAND is in contact with the ball.',
   h='task_ball_touch_hand_rate.',
   f='A hand that flops onto the ball counts. Contact is not intent.',
   use=8, clear=8),
 'gaze': dict(t='Gaze/ball alignment - cause unknown',
   w='How often the ball is near the gaze, and how often after the placement lock cleared.',
   h='task_gaze_on_ball_rate and the earned variant.',
   f='It cannot tell WHY the eyes and ball aligned: a spontaneous saccade, the built-in'
     ' motion reflex, head movement, the ball drifting through a still gaze, or the model.'
     ' It was previously labelled as the eyes having located the ball unaided.',
   use=8, clear=8),
 'How much it is moving': dict(t='Body speed and the stillness drive',
   w='Body-local speed, the stillness debt that has built up, and the movement wellness.',
   h='movement.local_speed, stillness_debt, movement_wellness.',
   f='Speed near zero means no causal test can resolve anything, whatever the model has'
     ' learned - and the cause is usually upstream. H23-H28 sat at 3.6-11 cm/s against'
     ' 28-43 when the body was moving, because the planner was switched off and the babble'
     ' was already zero. Check the commanded-rotation figure before blaming the model.',
   use=10, clear=9),
 'Co-movement channel (body map)': dict(t='Whether the body map\'s co-movement number is measured or bypassed',
   w='The number the movement drive\'s bonus is multiplied by (stats key movement.self_factor). With the body map'
     ' on (the code calls it `body_schema`) it is the map\'s co-movement number averaged over the 26 handles and'
     ' the bodies: how often the size of a handle\'s command change and the size of proprioception\'s change are'
     ' large in the same model step. It does not measure whether the command caused the movement. With the map'
     ' off the multiplier is a constant 1.00.',
   h='Detected from the log: a self_factor pinned at exactly 1.00 across every record is'
     ' the neutral fallback used when body-schema updating is off. With the map on, per handle: a running'
     ' average of (size of the command\'s change x size of each proprioception number\'s change), each size'
     ' divided by its own running root-mean-square; the handle\'s mean over the 286 proprioception numbers goes'
     ' through sigmoid(5 x (mean - one half)), which reads 7.6% for a mean of 0 and 92.4% for a mean of 1'
     ' (lem/model/body_schema.py update and self_membership).',
   f='1.00 reads like a confident measurement. It is the opposite: it is'
     ' the neutral value fed in when the channel is switched off, so nothing was measured'
     ' at all. A measured value is not agency either: the statistic has no time-lag bank, no sign and no'
     ' comparison against movement without a command. On synthetic inputs two unrelated random signals read'
     ' 30 to 80 of 100 on it and a body that answers its commands one model step late reads 8 of 100'
     ' (tests/test_body_schema_co_movement.py). The spec ends Stream 1 at \'body schema >80%\', which'
     ' cannot be reported while this says BYPASSED and which this statistic cannot establish when it is on.',
   use=9, clear=9),
 'Command-to-outcome matching': dict(t='Can the model pick its own command from the lineup',
   w='At each command ending (every 250 ms), a projection of the command must pick out that'
     ' command\'s own sensory change among decoys: the other bodies\' changes from the same'
     ' instant plus a bank of recent ones. Rank-1 is the fraction of events where the true'
     ' pairing wins.',
   h='tenure_tok.rank1; chance is 1/(1+n_decoys). Since 2026-09-02 the field is a true'
     ' per-event window mean; older logs re-report the last event between events'
     ' (sample-and-hold), so their window means are tenure-length-weighted.',
   f='It does not say WHICH sense carries the discrimination - the probe reports carry the'
     ' per-sense slice advantages. A rank1 at chance after hour 2 was H62\'s declared kill'
     ' line; above chance and climbing is the gain-of-function signal this project exists'
     ' to produce.',
   use=9, clear=8),
 'Hand-regard gate': dict(t='Vision joins the matching only while watching the hand',
   w='Vision\'s slice of the matching participates only for bodies whose hand was in view'
     ' at BOTH ends of the command - on hand-out-of-view tenures there is physically nothing'
     ' for the command to have changed in the image.',
   h='tenure_tok.vis_gate_open: the fraction of tenure events where the gate opened.',
   f='Demanding visual command-traces from frames that contain none eroded vision\'s causal'
     ' error ~5x over six H55 hours; removing vision entirely would abandon visual'
     ' self-recognition. A low percentage in the crib is expected - it is the case for a'
     ' strikeable mobile, not a defect.',
   use=8, clear=8),
 'Tenure pose lesson': dict(t='Predict what a whole command did, not one 50 ms slice',
   w='When a command ends, the model predicts the 250 ms pose change from the state where'
     ' the command began - once WITH the command and once with do-nothing - and the with-command'
     ' branch must beat do-nothing by 4% (a scale-free ratio, so shrinking every error'
     ' cannot satisfy it).',
   h='tenure_pe.pose (with-command error) and tenure_gap.pose (do-nothing minus with-command).'
     ' The hinge is skipped when the do-nothing error sits below the measured still-body'
     ' pose noise (CPU probe 2026-09-02) - a windless window teaches nothing.',
   f='A NEGATIVE margin smaller than the hinge\'s 4% right after a fresh-optimizer resume is the hinge doing'
     ' its job (pressure applied, not yet satisfied), not damage. Watch the direction'
     ' across hours, not the sign on one window.',
   use=8, clear=8),
 'Agency drive payments': dict(t='Evidence of your own effect pays wellness',
   w='Winning the command-to-outcome matching pays a bounded per-body amount into the same'
     ' wellness the planner steers by; a hand-watching bootstrap pays extra for a MOVING'
     ' hand in the fovea and fades as vision\'s measured agency evidence approaches'
     ' tactile\'s (displacement, not a clock; the anchor floors at the measured'
     ' advantage-noise scale).',
   h='agency.pay / prior_w / motion / fovea / vis_adv / touch_adv - window means since'
     ' 2026-09-02 (snapshots before).',
   f='Before 2026-09-02 the motion gauge read exactly zero in every run - the gate compared'
     ' a cache to itself twice over - so a bootstrap payment in an older log is a bug'
     ' artifact, not behavior. fov_missing > 0 means the flag never reached the model and'
     ' the bootstrap was dark, not declined.',
   use=8, clear=8),
 'Mobile above the chest': dict(t='The strikeable pendulum - the Rovee-Collier lever',
   w='A real crib mobile: a four-arm frame hanging from the ceiling, free to swing'
     ' AND spin, with five bright shapes dangling to undersides of 15-19 cm - the'
     ' band the wrists cross a quarter of the time (measured from 3,552 live wrist'
     ' positions). Average flailing misses it; a deliberate extension strikes it, and'
     ' one strike sets the whole assembly swinging, spinning and dancing for seconds'
     ' (probe-tuned damping; ambient drift settles below 2 cm/s) - a large, lasting,'
     ' SELF-CAUSED event filling a supine view',
   h='task_mobile_hit (any contact), task_mobile_hand (hand contact), task_'
     'mobile_speed (bob speed, cm/s) - window means. A limp body measures 0 contact.',
   f='This is an ENVIRONMENT lever, not a scripted goal: no wellness channel pays for striking it'
     ' directly. The chain it exists to feed: accidental strike -> recognized as'
     ' self-caused (matching) -> feels good (agency wellness) -> steered toward'
     ' (planner) -> deliberate striking. Rising hand-strike rate with rising'
     ' hand-regard gate is the signature to watch for.',
   use=9, clear=8),
 'Situation dependence (controlled test)': dict(t='Do the commands read the world - measured, not assumed',
   w='The situation test teleports the ball between two positions while the body holds'
     ' still and measures how much the model’s DETERMINISTIC command proposal moves,'
     ' against a twin measurement with the ball held (same cadence, no dice anywhere).',
   h='situation_test.json in the run folder. ESTABLISHED needs three things: the command must change'
     ' more when the ball moves than it changes from frame to frame with nothing moved (the jitter),'
     ' with the whole bootstrap 95% interval above 1 time the jitter; zero slot effect; and zero'
     ' determinism floor. First established 2026-09-03 on H65: a moved ball changed the command 1.40'
     ' times as much as the jitter did (95% interval 1.15 to 1.72 times the jitter).',
   f='This replaces the old unconditional “not established” parenthesis and the'
     ' bypassed correlation channel: a controlled yes/no with its own floors. H65’s 1.40 times the'
     ' jitter is where it started; it should grow as vision’s credit compounds.',
   use=9, clear=9),
 'Plan horizon license': dict(t='How far ahead the planner may aim, earned',
   w='The planner used to be capped at one step by a hand-set constant. Now each'
     " imagination depth (1..3 steps ahead) unlocks while the model's prediction at"
     ' that depth BEATS a do-nothing baseline on the same senses - a win rate averaged'
     ' over about its last 100 tries, per depth. Horizon = 1 + contiguous licensed depths.',
   h='plan_license.win_emas per depth; licensed = contiguous depths above 50 wins in 100, the'
     ' canonical beat-do-nothing null (the same standard as the tenure hinge). The'
     ' EMAs persist across resumes in runtime_state.pt - skill is not re-earned.',
   f='This is what makes multi-second projects (rolling over, and later crawling)'
     ' reachable ORGANICALLY: no goal is scripted anywhere - the horizon grows only'
     ' as demonstrated multi-step prediction grows, and the goals themselves still'
     ' come from whatever model steps felt best. Mode "fixed" means the old constant is'
     ' still in force and this row is telemetry only.',
   use=8, clear=8),
 'Expectation gate': dict(t='How much the model perceives what it expects',
   w='A learned gate blends up to half of the model\'s own prediction into the tokens'
     ' BEHAVIOR consumes (acting, planning) - perception smoothed by expectation. Every'
     ' lesson still grades raw encodings; a sustained-surprise rule (fast error above 2x a'
     ' slow baseline, measured margin) forces raw reality back through.',
   h='td_gate per sense - per-sense window means since 2026-09-02.',
   f='Pinned at ~0 for several full runs means the gate is trained-shut and the question'
     ' is whether to retire it; opening means check causal error for divergence between'
     ' the acting and learning paths. Grading any lesson on the blend would cut measured'
     ' error by (1-g)^2 with zero learning - the self-grading class removed 2026-09-02;'
     ' never reintroduce it.',
   use=8, clear=8),
 'General chest rotation (legacy measurement)': dict(t='General rotation can include turning around the pen',
   w='The largest chest rotation from the reset orientation seen this run. Turning around the pen can raise it without rolling onto the side.',
   h='Maximum of roll_deg_max across records.',
   f='It cannot establish rolling over or intention. The development dashboard now measures rotation about the body length separately: back 0, side 90, belly 180 degrees.',
   use=8, clear=8),
 'How much experience': dict(t='Body-time produced, not calendar age',
   w='Minutes of simulated body time this run produced per body.',
   h='New steps x action_repeat x 10 ms.',
   f="Not comparable to an infant's age: bodies run in parallel, model steps are re-studied"
     ' from memory, and none of the biological clock applies.',
   use=8, clear=8),
 'Judge-B': dict(t='Scorer agreement on REAL states only',
   w='How far the learned wellness scorer sits from the grounded one, on real states.',
   h='wellness_align_err. Shown only when imagination is on - otherwise it is a claim'
     ' about a mechanism that was not running.',
   f='Agreement on real states says nothing about imagined rollouts, which is the domain'
     ' that matters and is not measured here.',
   use=8, clear=8),
 'seeing a new picture': dict(t='Vision freshness',
   w='Whether every model step got a newly rendered image.',
   h='vision_fresh_fraction.',
   f='Below 1.0 means some steps re-read the previous frame, which silently halves the'
     ' visual learning signal.',
   use=8, clear=9),
 'Eye pitch': dict(t='Which way the head is pointing',
   w='Eye pitch in degrees from level; positive is above the horizon.',
   h='task_eye_pitch, with the launch preflight value alongside.',
   f='Reported as a measured angle rather than classified - +35 degrees used to be'
     ' classified as level.',
   use=8, clear=9),
 'see its own hands': dict(t='Are the hands visible to learn from',
   w='How often a hand is in the field of view, and how far off-centre the nearest is.',
   h='task_hand_seen and task_hand_view.',
   f='Visually guided reaching cannot be learned from hands the eyes never see, so a low'
     ' value caps what the run can possibly demonstrate. On H28 this sat at 0% for eight'
     ' and a half hours.',
   use=9, clear=9),
 'crossing of the geometric midline': dict(t='Do the arms cross the body',
   w='Share of time a hand is past the body midline.',
   h="task_crossed - the share of sampled steps on which a hand's lateral position is past"
     " the body's centre line.",
   f='A geometric threshold. It does not mean the crossing was intended.',
   use=8, clear=8),
 'Curiosity drive': dict(t='A function of prediction error, not a mood',
   w='The curiosity value, a deterministic function of recent prediction error, nonzero'
     " only inside the run's configured band.",
   h='curiosity_drive, with the band read from args.json.',
   f="It was described as the model's own dial of interest. It measures no interest in"
     ' anything - it is a triangular function of a scalar error.',
   use=8, clear=9),
 'Broken numbers': dict(t='NaN count, which must stay zero',
   w='How many steps produced a non-finite number.',
   h='Sum of nan_steps. The row only appears when it is nonzero.',
   f='Nothing - any value above zero invalidates the run. But zero here does not mean'
     ' undamaged: H30 destroyed its reconstruction heads without producing a single NaN.',
   use=9, clear=9),
 'Saves that failed': dict(t='Failed checkpoint writes, which must stay zero',
   w='How many hourly saves raised an error since this segment started; the run continued'
     ' and tries again next hour.',
   h='checkpoint_failures and checkpoint_last_error in the newest record. The row only'
     ' appears when it is nonzero.',
   f='Nothing about learning. A disk that refuses writes (D: on 2026-09-18: 1-2 MB/s, port'
     ' resets) means a crash now loses everything since the newest complete save.',
   use=9, clear=9),
 'Peak allocation': dict(t='Peak GPU memory the trainer allocated in the last window',
   w="The largest CUDA allocation in the last report window against the card's total; the"
     " body workers' eye renderers hold about 100 MiB more per body outside this count.",
   h="peak_vram_mb in the newest record (the allocator's own count, reset each window).",
   f="Nothing about learning. Near the card's total the allocator thrashes and the step time"
     ' triples (measured 2026-09-18 at 5,270 of 8,188 MiB with eight renderers).',
   use=8, clear=9),
 'brain was saved': dict(t='Checkpoint recency',
   w='When weights were last written, and how many snapshots exist.',
   h='checkpoints.json in the run folder: the timestamp of the newest entry and the number'
     ' of entries.',
   f='Checkpoints are hourly by default, so a run under an hour old has none - which is'
     ' not a reason to lose the work. Write {"cmd": "snapshot_and_stop"} to control.json in'
     ' the run folder and it saves within seconds.',
   use=8, clear=9),
 'Built-in help it was born with': dict(t='Endowed mechanisms, present from step zero', secondary=True,
   w='Which reflexes the body was given rather than learned: the snap-to-motion eye'
     ' reflex, and whether a caregiver returns the ball after it goes unseen.',
   h='args.json gaze_reflex and caregiver_seconds.',
   f='These move the eyes and replace the toy without the model doing anything, so any'
     ' gaze or reach number has to be read against them.',
   use=7, clear=8),
 'weighted boxes': dict(t='Box contact by ANY body part', secondary=True,
   w='How often each weighted box is touched, by any part of the body, with hand-only'
     ' alongside.',
   h='task_box_touch_rate and task_box_touch_hand_rate.',
   f='A leg resting against a box counts. It shows exposure, not weight perception or'
     ' grasping.',
   use=6, clear=8),
 'hand pressure above threshold': dict(t='Contact exposure, not skill', secondary=True,
   w='Share of time a hand presses above 2 kPa.',
   h='task_pressing and task_press_kpa.',
   f='Random babble pressed on 90 of every 100 steps when this row was written (2026-08-26; the'
     ' run it was read from is not recorded), so this is an exposure rate rather than an achievement.',
   use=7, clear=8),
 'eyesight': dict(t='Fixed optical configuration', secondary=True,
   w='Angular resolution of the wide field and the sharp centre.',
   h='Computed from eye_size, fovea_size and fovea_fov in args.json.',
   f='Static for the whole run - configuration, not a result.',
   use=6, clear=8),
 'Spontaneous eye movements': dict(t='Endowed mechanism, not learned', secondary=True,
   w='Saccade rate per minute of body time.',
   h='task_saccades divided by body-minutes.',
   f='Built in, not learned. It is the DATA for learning what looking does, and must not'
     ' be counted as gain of function.',
   use=7, clear=8),
 'Eye flicks toward movement': dict(t='Endowed reflex, not learned', secondary=True,
   w='Reflex snap rate per minute of body time.',
   h='task_snaps divided by body-minutes.',
   f='Fires on peripheral motion, so a still crib keeps it low regardless of the model.',
   use=6, clear=8),
 'toy back': dict(t='Caregiver interventions', secondary=True,
   w='How often the caregiver returned the ball to view.',
   h='task_rescues and task_unseen_s.',
   f='A high count means the scene, not the model, is keeping the target visible.',
   use=7, clear=8),
 'action signatures': dict(t='Uncalibrated between-body ratio', secondary=True,
   w="Spread of each body's average command between bodies, over each body's own variation"
     ' across time.',
   h='motor_discrimination_ratio: the spread between bodies over each body\'s own variation.',
   f='It rises when a body simply moves LESS, and the bodies differ in worker identity,'
     ' random history and pose. It was labelled as awareness of the situation, which it is'
     ' not.',
   use=5, clear=7),
 'How fast the skin': dict(t='Tactile is a discrete-event channel',
   w='How many skin patches start touching, and how many stop, in one model step.',
   h="contact_turnover from the trainer: this step's active-patch mask against the"
     ' previous one, averaged over the bodies and over the report window.',
   f='Measured 2026-08-28 on H35: about two patches appear and two vanish every 50 ms'
     ' against 5.6 in contact, so the set changes on 95% of steps - and those steps'
     ' carry 98% of the tactile prediction error. Real rather than threshold flicker: a'
     ' birth has median area 1,323 mm2, LARGER than the median held contact. This is'
     ' why the executed command adds 0% to predicting tactile against 9.2% for'
     ' proprioception, and why the counts above can read perfectly flat while the set'
     ' underneath them turns over completely.',
   use=9, clear=8),
 'touching its skin': dict(t='Live contact count', secondary=True,
   w='How many physical contacts the body is in right now.',
   h='contacts_per_env - the raw MuJoCo contact count for the whole model (body-floor'
     ' and box-floor included), averaged across bodies. The skin-patch count is'
     ' touch_active_patches.',
   f='A crib body rests on the floor, so 15-20 is normal and says nothing about'
     ' interaction.',
   use=7, clear=8),
 # ---- gain-of-function build, 2026-09-22 (docs/gain_of_function_v3/DESIGN.md)
 'Deliberate plans: how close': dict(t='How close a plan gets to the model step it chose',
   w='The distance, in token space, between where the last plan ended and the remembered model step it'
     ' chose as its goal, converted to instants of world change when the anchors fit this model.',
   h='intention.final_mse against scale_anchors.json one_instant and half_minute_apart. The anchors'
     ' were measured on H67 (128 wide); a run of another width prints "unanchored for this model".',
   f='Token distances do not transfer between models of different widths: dividing H84 (192 wide)'
     ' distances by 128-wide anchors printed "229% of the distance between two unrelated model steps".',
   use=8, clear=8),
 'Action use in prediction (prediction audit)': dict(t='Does knowing its own command improve prediction',
   w='The swap test: the prediction error of the next model step with the command the body really executed,'
     ' against the error with a control command in its place (context, memory and every other sense unchanged),'
     ' as a share of the way from 0% (the control predicts as well: the command is not used) to 100% (exact).'
     ' Three controls: another body\'s command size-matched to this body (the other body\'s asked moves and turns,'
     ' measured from its own pose, applied to this body\'s pose; the eyes\' row likewise), this body\'s own command'
     ' shifted by one standard deviation of each of its 26 x 9 numbers, and another body\'s absolute command (its'
     ' targets as they stand: the original control, kept for one segment, then retired). Graded at two places in'
     ' a hold: model steps that are multiples of 20 (with --motor-hold 5, the last model step of a hold) and the'
     ' hold\'s first model step.',
   h='prediction_audit.modalities.<sense>.improvement_over_size_matched, improvement_over_shifted_own and'
     ' improvement_over_shuffled (the absolute command); prediction_audit_hold_first.senses.<sense>.all_bodies'
     ' carries the same three at the hold\'s first model step. Measured online before each update on paired raw'
     ' tokens (mimo_bridge/prediction_audit.py); sensitivity_<control> is how far that control moves the'
     ' prediction, 0% = not at all, 100% = as far as the whole predicted change.',
   f='Encoded senses, not physical skill. The absolute command asks this body for changes chosen for another'
     ' pose, so a prediction that only grows with the size of the asked change scores above 0% against it'
     ' without using the command\'s effect; the size-matched control removes that. Over H84\'s first 344,400 model'
     ' steps the absolute reading was 0% to 0.08% of the way for every sense (DESIGN.md section 0), graded only'
     ' on the last model step of each hold.',
   use=10, clear=8),
 'Better than a copy of the present (prediction audit)': dict(t='Does it beat predicting that nothing changes',
   w='How much better the next model step is predicted than by copying the present model step forward.'
     ' 0% = no better, 100% = exact, below 0% = worse than copying.',
   h='prediction_audit.modalities.<sense>.improvement_over_persistence (1 - prediction error /'
     ' persistence error), on the same transitions as the action audit.',
   f='A still body makes copying hard to beat, and a world that becomes easier moves it without any'
     ' learning. On segment 31 vision and proprioception read below 0.',
   use=9, clear=8),
 'Wellness channels (window means over the bodies)': dict(t='What each wellness channel read',
   w='The window mean of each channel the grounded wellness score sums, over all bodies. The coherence channel'
     ' carries the body map\'s co-movement statistic (the row "Body map co-movement"): how often command changes'
     ' and proprioception changes fall in the same model step, not whether the command caused them.',
   h='wellness.<channel> in the record, averaged over the report window and the bodies.',
   f='Window means understate the per-model-step spread that decides a goal, and the channels spread very'
     ' differently (from model step to model step the prediction-error channel moved 0.38% of its 0-to-1 range,'
     ' tactile comfort 2.1% of it, on segments 30-31), so'
     ' comparing raw values says which channel is larger, not which matters.',
   use=8, clear=8),
 'Consequence term in wellness': dict(t='How often a model step carried a consequence',
   w='The share of model steps of a body with at least one self-relevant consequence (a contact, a moved thing,'
     ' a gained view) and the share of chosen goals that carried one.',
   h='consequence.rate and consequence.goal_rate (lem_model last_consequence), window means.',
   f='Until --consequence-self-contact exclude, hand pressing on its OWN body counts: on segment 30 the'
     ' flag was on 74% of model steps, so the rate is mostly self-contact, not effect on the world.',
   use=9, clear=8),
 'Who authored the executed command': dict(t='Planner, reflex and exploration shares',
   w='Of the commands latched in the window, the share whose action came from the planner, from a reflex'
     ' with exploration noise, and from a reflex alone.',
   h='exec_source: counts of model.last_action_source at each latch, divided by their total.',
   f='Authorship is not success: a planner command can be smoothed into near-silence before it reaches'
     ' the body (the smoother blends 65% of an average of proposals).',
   use=8, clear=8),
 'Body map co-movement (incorporation)': dict(t='The body map\'s co-movement numbers, averaged',
   w='The input of the coherence wellness channel: the body map\'s extent, the mean of one co-movement number'
     ' per handle (26) and per outside thing it'
     ' watches (the toy and the mobile). A handle\'s number counts how often the size of its command change and'
     ' the size of proprioception\'s change are large in the same model step; it does not say the command caused'
     ' the change. Beside it, for the toy: how its movement matches a hand\'s against the'
     ' same match with the hand moving the opposite way (the control).',
   h='incorporation.extent, shown between its real low and high ends as 0% and 100% (a score of 0 reads 7.6% of'
     ' the 0-to-1 range and a score of 1 reads 92.4%), and incorporation.objects_evidence. Per handle the score'
     ' is a running average of (size of the command\'s change x size of each proprioception number\'s change),'
     ' each size divided by its own running root-mean-square (lem/model/body_schema.py).',
   f='Reading it as how much of its body the model owns or controls: the handle statistic has no time-lag bank,'
     ' no sign and no comparison against movement without a command, and on synthetic inputs two unrelated'
     ' random signals read 30 to 80 of 100 on its own output (tests/test_body_schema_co_movement.py). And a'
     ' thing the world moves by itself (the mobile is physics-driven) can match a hand by chance; the'
     ' reversed-direction control is what separates carried from coincident.',
   use=8, clear=8),
 'Self-contact (own body touching itself)': dict(t='Self-touch: how often, and whether predicted',
   w='How many pairs of its own body parts touch at an average model step, how much the executed command'
     ' helps predict what a self-touching body feels, and what the self-touch wellness term paid.',
   h='self_contact_pairs (window mean per body), self_touch_controlled.improvement_over_shuffled and'
     ' wellness.self_touch.',
   f='Contact is not control: the addendum pays predicted and controlled self-contact, never contact'
     ' itself. The improvement is measured on few contacts (23 on segment 31) and sat below 0.',
   use=9, clear=8),
 'Causal chain of each plan': dict(t='Did it choose what it predicted would feel best',
   w='For each planning event, whether the chosen plan was the one with the highest predicted wellness'
     ' among itself and the alternatives from the same state, how often it was tied (its predicted lead'
     ' over the best alternative inside the tie band, the evaluator\'s own error on a difference between'
     ' two candidates), and how far apart the candidates were predicted to feel.',
   h='causal_chain.chosen_best_share (chance 1/(1 + alternatives_n)), chosen_tied_share, tie_band,'
     ' candidate_spread_mean, gap_to_best_alt_mean and plan_exec_distance_cm, written only with'
     ' --causal-chain on.',
   f='The planner does not aim at predicted wellness (it descends on distance to a remembered model step),'
     ' so this share has no goal line; and an evaluator that gives every candidate the same value'
     ' would make every plan "the best" unless ties are counted apart. The tie band is wider than the'
     ' spread of the candidates\' predicted wellness in every window of H84 (1,174 of 1,174 since'
     ' Sep 23; median 12.6 times the spread, docs/gain_of_function_v4/agents/'
     'hold_loss_history_2026-10-02.json), so every choice reads tied whatever the forward model'
     ' predicts: the tied share is no evidence about the forward model. Its planner-side reading is'
     ' the planner recheck\'s captured share (Development page, planner candidates).',
   use=10, clear=8),
 'Efference heads (predicted body and eye movement)': dict(t='Predicted movement in centimetres and degrees',
   w='Given the executed command directly, how far off the predicted movement of each body handle and of the'
     ' eyes is, against predicting no movement and against another body\'s command: size-matched to this body'
     ' (its asked moves and turns, measured from its own pose, applied to this body\'s pose) and, for one'
     ' segment, absolute (its targets as they stand).',
   h='efference.body_cm_err_latch, body_cm_do_nothing_latch, body_cm_size_matched_latch, body_cm_shuffled_latch'
     ' (the absolute command) and the gaze errors in degrees at 50, 100 and 150 ms (gaze_deg_size_matched beside'
     ' them), written only with --efference-heads on (lem/model/efference.py window_summary).',
   f='A head can beat "no movement" by predicting the average drift of every body; only the gap to another'
     ' body\'s size-matched command shows that THIS command was used. The absolute command errs more for any head'
     ' whose output grows with the asked change, so its gap overstates command use.',
   use=10, clear=8),
 'Machine load (memory, card, step time)': dict(t='Memory, card and time per step',
   w='Committed memory against the machine\'s limit, card memory in use, and the time one model step'
     ' took.',
   h='machine.commit_gb / commit_limit_gb, gpu_used_mib / gpu_total_mib and step_ms, sampled each'
     ' report window.',
   f='Nothing about learning. Segment 31 fell from 1.25 to 0.40 model steps per second at 14:16 on'
     ' 2026-09-22 (its first-hour mean against the hourly mean the watcher stopped it on) with nothing'
     ' recording these; a spill of card memory into system memory would show here first.',
   use=9, clear=9),
 'Vestibular prediction and trunk posture': dict(t='Vestibular and trunk posture',
   w='How much better than "nothing changes" vestibular (the body\'s load and balance readings) is predicted, and'
     ' how far the trunk is turned from lying on its back.',
   h='vestibular.improvement_over_persistence and trunk_roll_deg, written only with --vestibular-sense on.',
   f='Posture alone is not control: a body can be rolled by the adult or the floor. The prediction gain'
     ' is the part the model earns.',
   use=9, clear=8),
 'Controllability channel': dict(t='Did the body do what it expected',
   w='Agreement between the effect the model expected from the command it sent and the effect achieved'
     ' over the command\'s life, beside the same body\'s result against another body\'s command (the'
     ' control), as shares of the way from "unrelated" to "exactly as expected".',
   h='controllability.agreement_mean against control_mean (the shuffled-command control of the field'
     ' contract, 2026-09-23), scored only on commands where an effect was expected and commanded above a'
     ' floor; measured_share is a share of COMMANDS (--controllability on).',
   f='Stillness is not control: commands with no expected effect are left out, so read the commands'
     ' scored beside it; and an expectation that ignores the command can agree with a body\'s drift,'
     ' which only the control exposes.',
   use=10, clear=8),
 'Controllability gates on the same commands': dict(t='The gate in force beside the shadow gate',
   w='Controllability enters the goal ranking only while its evidence gate is open. Two gates are judged on'
     ' the same graded commands (the bodies\' commands latched on one model step are graded together when the'
     ' next commands appear). Each takes one reading per graded command: the mean, over the bodies measured,'
     ' of the agreement between the body head\'s expectation from this body\'s own command and what the body'
     ' did, minus the same agreement for the expectation from a control command. The gate in force takes'
     ' another body\'s absolute command (its targets as they stand) as the control; the shadow gate, recorded'
     ' only, takes another body\'s size-matched command (its asked moves and turns, measured from its own pose,'
     ' applied to this body\'s pose). Each gate opens when the mean of its last 32 readings is above 3 of their'
     ' standard errors and shuts when it is under 1.',
   h='controllability.gate_commands, gate_open_commands, shadow_gate_commands, shadow_gate_open_commands (per'
     ' report window: the graded commands at which that gate took a reading, and those after which it stood'
     ' open), gates_same_commands, gates_both_open_commands, gate_only_open_commands,'
     ' shadow_gate_only_open_commands, gate_open and shadow_gate_open (the window\'s end); beside them'
     ' size_matched_control_mean, size_matched_difference_mean, shadow_gate_mean, shadow_gate_standard_error and'
     ' shadow_gate_readings_above_zero. gate_definitions in the block defines each'
     ' (lem/model/wellness_channels.py; written from the code of 2026-10-03 with --controllability on and'
     ' --efference-heads on).',
   f='The shadow gate changes nothing: the controllability weight in the goal ranking reads the gate in force'
     ' alone. A body head whose output grows with the asked change forms a larger, worse expectation from an'
     ' absolute command chosen for another pose, so the gate in force can open without command use; the'
     ' size-matched command asks this body for changes of a real command\'s size, and the two counts side by'
     ' side show on how many graded commands that difference changed the gate\'s state.',
   use=10, clear=8),
 'Learning-progress channel': dict(t='Are prediction errors falling',
   w='The fall in the ratio of model error to "nothing changes" error, slow average against fast.'
     ' Positive = getting better, negative = getting worse.',
   h='learning_progress.value_mean, written only with --learning-progress on.',
   f='Measured on the ratio so that the world simply becoming easier (the adult leaving) does not count'
     ' as learning; raw error falls in that case with nothing learned.',
   use=9, clear=8),
 'Normalised goal ranking': dict(t='Is the goal picked by one channel\'s size alone',
   w='With goals ranked on channels put on one scale, the leading channel\'s share of the winning'
     ' goal\'s lead beside its share of the ranking weights, where in the goal ring the chosen goals'
     ' sit, and how often the integrity gate barred a candidate.',
   h='wellness_rank.winner_margin_share against weight_share_by_channel (else the weights),'
     ' goal_age and integrity_gated_share (--wellness-rank normalized). The shares of the lead add'
     ' to 100 and can be below 0 or above 100.',
   f='A channel can carry the lead because it matters most by priority weight, so its lead share is'
     ' read against its weight share; and slowly drifting channels made 42 of 44 goals the oldest'
     ' remembered model step in the build\'s own run, which only goal age shows.',
   use=9, clear=8),
 'Hand pressing split into world and self': dict(t='Hand pressing on the world and on itself',
   w='The share of model steps a hand presses something in the world, and the share it presses its own body.',
   h='hand_press.world_share and self_share, from the world report\'s hand_pressing_world/self.',
   f='399 of 414 pressing steps in a random-command probe were self-contact; before this split every'
     ' one of them counted as a consequence.',
   use=8, clear=8),
 'Contact prediction and control': dict(t='Is a touch foreseen before it happens',
   w='On skin patches untouched the model step before, in how many (touched, untouched) pairs the contact'
     ' head gave the patch about to be touched by its own body the higher chance (50 in 100 = a coin'
     ' flip), beside a base-rate guess on the same patches and model steps; and the self-touch onsets foreseen'
     ' with its own command beside the same onsets with another body\'s command.',
   h='contact_control.onset_auc_self against control_onset_auc_self (field contract 2026-09-23);'
     ' until those exist, success_paired against success_control, which are the same onsets'
     ' (--contact-head on).',
   f='The any-contact score (head_auc) mostly measures whether a touch that is already there'
     ' continues, which is easy; success_self counts every onset while success_control counts only'
     ' the onsets where the control ran, so those two are never set side by side.',
   use=10, clear=8),
 'Self-touch onsets': dict(t='Self-touches that began',
   w='How many touches of its own body began in the window, and how many were within view.',
   h='self_contact_events.onsets and in_view_share, from the world report\'s self_contact_onsets.',
   f='A structural contact (the thumb resting in the palm) is not an event; the world report marks'
     ' those so they can be left out.',
   use=8, clear=8),
 'Motor synergy map': dict(t='Does the body follow commands the synergy map writes',
   w='How well the body follows the commands written through the synergy map, against a posture the'
     ' body really reached on physics alone; beside it, how much of the posture changes the map'
     ' reproduces and how many independent directions it uses.',
   h='synergy.following_by_author (else following_mean), explained_share and effective_dim'
     ' (--motor-synergy on).',
   f='A map that reproduces the average posture change explains a large share while controlling'
     ' nothing; effective_dim near 1 means every joint still moves together.',
   use=9, clear=8),
 'Next-model-step loss normalisation': dict(t='Each sense pushed to beat a copy of the present',
   w='The effective weight of each sense\'s next-model-step lesson after dividing its error by its own'
     ' "nothing changes" error.',
   h='next_frame_norm.weight_effective (--next-frame-norm persistence).',
   f='Before it, audio carried 67% of the next-model-step loss and proprioception 3.8% of it (DESIGN.md'
     ' section 0): the sense the body moves got under a twentieth of the push.',
   use=8, clear=8),
}


def doc_for(label):
    """The ROW_DOC key whose fragment appears in this label, or None.

    Case-insensitive: matching exactly missed "Hand pressure above threshold"
    against the key "hand pressure above threshold" and rendered that row with no
    explanation at all. A test now asserts every rendered row resolves.
    """
    low = label.lower()
    for frag in ROW_DOC:
        if frag.lower() in low:
            return frag
    return None


_CKPT_W_CACHE = {}


def _card_mib():
    """The card's total memory in MiB, or None when no CUDA device is reachable here."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory // 2 ** 20)
    except Exception:
        pass
    return None


def command_pathway_weight(run_dir):
    """|w| of the motor encoder's command columns in the newest checkpoint, or None.

    This is the most direct read available on whether the model is building the ability
    to use its own actions. It is cached on the checkpoint's mtime because the page
    re-renders every 30 s and the file only changes hourly.
    """
    cks = glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_step_*", "model.safetensors"))
    if not cks:
        return None
    # tie-break on mtime: ckpt_step_701642 and ckpt_step_701642_tc19 parse to the same
    # step, and picking the 10-wide one makes W[:, 10:19] an EMPTY slice whose mean is nan
    best = max(cks, key=lambda q: (int(re.search(r"ckpt_step_(\d+)", q).group(1)),
                                   os.path.getmtime(q)))
    try:
        key = (best, os.path.getmtime(best))
    except OSError:
        return None
    if key in _CKPT_W_CACHE:
        return _CKPT_W_CACHE[key]
    try:
        from safetensors.torch import load_file
        W = load_file(best)["encoder_registry.encoders.motor.patch_mlp.0.weight"]
        if W.shape[1] < MOTOR_IN:
            return None          # predates the 10 -> 19 widening: not applicable, not zero
        val = float(W[:, ACHIEVED:MOTOR_IN].float().abs().mean())
    except Exception:
        val = None
    if len(_CKPT_W_CACHE) >= 512:
        _CKPT_W_CACHE.clear()
    _CKPT_W_CACHE[key] = val
    return val


def _merge(a, b):
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def motor_size_overrides(args):
    """The motor-encoder part of train_mimo.size_overrides (d_model, encoder_layers,
    encoder_width), read from a run's args.json without importing the trainer."""
    ov = {}
    try:
        d_model = int(args.get("d_model") or 0)
        layers, width = int(args.get("encoder_layers") or 0), int(args.get("encoder_width") or 0)
    except (TypeError, ValueError):
        return ov
    if d_model:
        ov["model"] = {"d_model": d_model}
    if layers or width:
        enc = {}
        if layers:
            enc["n_cross_attn_layers"] = layers
        if width:
            enc["d_internal"] = width
            enc["per_patch_hidden"] = width
        ov["modalities"] = {"motor": {"encoder": enc}}
    return ov


_READ_CACHE: dict = {}

# Lived motor rows for the one-body-part-at-a-time readability test (reviews/dashboards.md M9):
# achieved handle states and the orders the model sent, recorded by b5-synergy on
# ckpt_step_344400 in segment 31's own 8-body world (trunk_v1 body layout, 26 handles).
READABILITY_ROWS = os.path.join(REPO, "docs", "gain_of_function_v3", "measurements",
                                "motor_rows_seg31_ckpt_step_344400.npz")
# The line a handle's 20-degree order must move the motor token by, as a share of the token's
# size. MEASURED 2026-09-23 00:40 on ckpt_step_344400 (docs/gain_of_function_v3/measurements/
# per_handle_readability_ckpt_step_344400.json, script per_handle_readability.py beside it; 16
# lived rows, 3 turn axes each): 10 of the 26 handles (the eyes, head, chest, both upper arms, the
# left wrist, thumb, index, ring and pinky) move the token 0.6 to 4.0 millionths of its size; the
# next handle (right thigh) moves it 211 millionths, 52 times more, and the best-read 8 in 100.
# The line sits at the gap's geometric middle, 29 millionths, so every handle is at least 7 times
# away from it and the count below it does not depend on where in the gap it is drawn. Every
# handle below it gets zero weight from all 4 heads of the encoder's pool (the cross-check); the
# converse does not hold (the second attention layer mixes some zero-weight handles into the rows
# the pool reads). Turning all 26 handles at once (the old test) moves the token 21 in 100.
HANDLE_READ_LINE = 2.9e-5


def readability_rows(layout, n_handles, limit=16):
    """[(achieved [1, n, 10], order [1, n, 9])] lived rows for this body layout, or None."""
    try:
        import numpy as np
        import torch
        d = np.load(READABILITY_ROWS, allow_pickle=False)
        if str(d["layout"]) != str(layout) or d["ach"].shape[1] != n_handles:
            return None
        return [(torch.from_numpy(d["ach"][i:i + 1].astype("float32")),
                 torch.from_numpy(d["order"][i:i + 1].astype("float32")))
                for i in range(min(int(limit), d["ach"].shape[0]))]
    except (OSError, KeyError, ValueError):
        return None


def load_motor_encoder(safetensors_path, run_dir):
    """(encoder, body layout, handle names) for the motor encoder saved in a checkpoint, built at
    the run's own size; None when the checkpoint predates the 10 -> 19 widening. Only the motor
    encoder's tensors are loaded; no whole model is built."""
    from safetensors.torch import load_file
    import torch
    from mimo_bridge.lem_model import build_config
    from mimo_bridge.body_layout import config_overrides, names_for
    from lem.model.encoder_registry import EncoderRegistry
    sd = load_file(safetensors_path)
    w = sd.get("encoder_registry.encoders.motor.patch_mlp.0.weight")
    if w is None or w.shape[1] < MOTOR_IN:
        return None
    receipt = load(os.path.join(os.path.dirname(safetensors_path), "model_receipt.json"), {})
    layout = receipt.get("body_layout", "legacy")
    # the run's size flags, as train_mimo.size_overrides applies them to the motor
    # encoder: without them H84's 192-wide, 2-layer encoder failed to load into the
    # 128-wide, 1-layer default and the chip read "input reading unavailable"
    run_args = load(os.path.join(run_dir, "args.json"), {}) or {}
    cfg = build_config(_merge(config_overrides(layout), motor_size_overrides(run_args)))
    cfg["_active_modalities"] = ["motor"]
    cfg["encoder_defaults"]["output_layernorm_affine"] = "encoder_registry.output_norm.weight" in sd
    weights = {k[len("encoder_registry."):]: v.float() for k, v in sd.items()
               if k.startswith(("encoder_registry.encoders.motor.", "encoder_registry.output_norm."))}
    with torch.random.fork_rng(devices=[]):
        encoder = EncoderRegistry(cfg).eval()
        encoder.load_state_dict(weights, strict=True)
    return encoder, layout, list(names_for(layout))


def action_readability_for(run_dir):
    """Can the motor encoder in the newest checkpoint READ the action? Token movement for
    two different orders vs for zeroing the command's position block (tools/probes.py).

    The weight reading (command_pathway_weight) said "alive" for every checkpoint of H28
    to H33 while the action was drowned: 99.6% of the row's input energy was a position
    block in centimetres, and two different orders moved the token 1.6 against 14.6 for
    the block. Weight x input is what reaches the network, so this measures the input.
    Cached per checkpoint and mtime. Only the saved motor encoder is evaluated,
    using synthetic rows at the saved body layout; no whole model is constructed.
    """
    cks = glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_step_*", "model.safetensors"))
    if not cks:
        return None
    best = max(cks, key=lambda q: (int(re.search(r"ckpt_step_(\d+)", q).group(1)),
                                   os.path.getmtime(q)))
    try:
        key = (best, os.path.getmtime(best))
    except OSError:
        return None
    if key in _READ_CACHE:
        return _READ_CACHE[key]
    try:
        from mimo_bridge.tools.probes import action_readability, synthetic_motor_rows, per_handle_readability
        loaded = load_motor_encoder(best, run_dir)
        if loaded is None:
            val = None              # predates the widening: not applicable
        else:
            encoder, layout, names = loaded
            val = action_readability(encoder, synthetic_motor_rows(n=len(names)))
            val["handles"] = len(names)
            val["scope"] = "synthetic input test of saved motor encoder through current preprocessing"
            val["checkpoint"] = os.path.basename(os.path.dirname(best))
            # one body part at a time (reviews/dashboards.md M9): realistic lived rows when the
            # checkpoint's body layout matches the recorded ones, else none (never synthetic
            # full rotations again)
            rows = readability_rows(layout, len(names))
            if rows is not None:
                ph = per_handle_readability(encoder, rows)
                ph["names"] = list(names)
                ph["line"] = HANDLE_READ_LINE
                ph["below"] = [nm for nm, s in zip(names, ph["share"]) if s < HANDLE_READ_LINE]
                val["per_handle"] = ph
    except Exception:
        val = None
    if len(_READ_CACHE) >= 512:
        _READ_CACHE.clear()
    _READ_CACHE[key] = val
    return val


_GIT_CACHE = {"t": 0.0, "head": None}


def git_head():
    """Short hash of the current source HEAD; None if git is unavailable."""
    if time.time() - _GIT_CACHE["t"] < 60 and _GIT_CACHE["head"]:
        return _GIT_CACHE["head"]
    try:
        h = subprocess.run(["git", "-C", REPO, "rev-parse", "--short=8", "HEAD"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        _GIT_CACHE.update(t=time.time(), head=h or None)
    except Exception:
        _GIT_CACHE.update(t=time.time(), head=None)
    return _GIT_CACHE["head"]


# The lite model's core width when --d-model is not given (lem_model.py LITE "d_model": 128);
# H67, where scale_anchors.json's token-distance anchors were measured, ran at this width.
DEFAULT_D_MODEL = 128


def model_width(args):
    try:
        return int((args or {}).get("d_model") or 0) or DEFAULT_D_MODEL
    except (TypeError, ValueError):
        return DEFAULT_D_MODEL


def anchor_width(anchors):
    """Core width of the model the token-distance anchors were measured on, from that
    checkpoint's run folder (args.json); the default width when it cannot be read."""
    ckpt = (anchors or {}).get("ckpt")
    if anchors and anchors.get("d_model"):
        return int(anchors["d_model"])
    if not ckpt:
        return DEFAULT_D_MODEL
    return model_width(load(os.path.join(os.path.dirname(os.path.dirname(str(ckpt))), "args.json"), {}))


def plan_distance_row(im, an, args):
    """The plan-distance row, or None. Token distances are converted to instants of world
    change only when the anchors were measured on a model of this run's width; otherwise the
    row says "unanchored for this model" (H67's 128-wide anchors against H84's 192)."""
    if im.get("final_mse") is None or not an.get("one_instant"):
        return None
    fm = float(im["final_mse"])
    one, far = float(an["one_instant"]), float(an["half_minute_apart"])
    gp = im.get("gap_closed_pct")
    gp_s = (f"; this plan closed {gp:.0f}% of its goal-gap "
            f"(0% = never moved toward it, 100% = arrived)" if gp is not None else "")
    aw, rw = anchor_width(an), model_width(args)
    label = "Deliberate plans: how close do they get to what they want?"
    # a token distance is not a lived unit: it is printed only as instants of world change or as a
    # share of the distance between two unrelated moments, and only when the anchors fit this width
    if aw != rw:
        return (label, f"the last plan ended some token distance from the state it chose, which cannot be "
                       f"stated here: unanchored for this model: the anchors in scale_anchors.json (one instant "
                       f"of world change, two unrelated model steps) were measured on a {aw}-wide model and this run "
                       f"is {rw}-wide, so the distance cannot be stated in instants or as a share of unrelated "
                       f"model steps{gp_s}")
    return (label, f"the last plan ended about {fm/one:.0f} instants of world-change short of the state it chose, "
                   f"{100*fm/far:.0f}% of the distance between two unrelated model steps (arrived = under "
                   f"{100 * float(an.get('goal_arrival', 0.01)) / far:.1f}% of that distance; anchors measured "
                   f"{an.get('measured')}){gp_s}")


def parse_resume_step(resume_path):
    """Step number embedded in a resume checkpoint path, or None for a fresh start."""
    if not resume_path:
        return None
    m = re.search(r"ckpt_step_(\d+)", str(resume_path))
    return int(m.group(1)) if m else None


def fmt_change(a, b, k=None):
    """A change with its verdict attached, because a bare percentage says nothing.

    Every quantity on this page that moves is an ERROR, so smaller is better and a
    POSITIVE percentage is bad news - the opposite of the usual reading. Some rows
    said "getting worse" and others printed the percentage alone, leaving the reader
    to guess the sign convention on the one thing they need.
    """
    if not a:
        return "no starting value to compare with"
    change = 100.0 * (b - a) / a
    verdict = ("ABOUT THE SAME" if abs(change) <= 2 else
               "IMPROVING" if change < 0 else "GETTING WORSE")
    win = f" from the first {k} reports to the last {k}" if k else ""
    moved = ("did not change" if change == 0 else
             f"{'rose' if change > 0 else 'fell'} {fixed(abs(change), 2)}%")
    # the error's own level is an encoded distance with no lived anchor yet, so only its change
    # is printed (DESIGN.md section 11: never a mean squared error)
    return (f"{verdict} - the error {moved}{win} "
            f"(this is an error, so smaller is better; its level is an encoded distance, not yet "
            f"measured against a lived reference)")


def change_words(a, b):
    """How a quantity moved from a to b, as a percentage of a, never the raw values."""
    if not a:
        return "no starting value to compare with"
    change = 100.0 * (b - a) / a
    return "did not change" if change == 0 else f"{'rose' if change > 0 else 'fell'} {fixed(abs(change), 2)}%"


# Reconstruction error as a multiple of next-state error on UNDAMAGED runs with residual
# prediction on: 1.16 times on H36 (the day the flag landed) to 2.45 times on H43 (measured, see
# reconstruction_health). A level above that range is not a fault by the jump rule, and it is not
# shown green either.
RESIDUAL_UNDAMAGED = (1.16, 2.45)


def reconstruction_health(recon_win, next_win, residual=False):
    """Per-sense reconstruction/next-state ratio, and whether it means damage.

    Re-describing the state you are already looking at is an EASIER task than predicting
    the one that comes next. So reconstruction error should sit below next-state error,
    and a model where it does not has a broken reconstruction head. This is a structural
    relationship, not a threshold picked here - measured on the last 30 reports of four
    runs:

        H22 healthy     vision 0.39   tactile 0.37   proprioception 0.36
        H28 healthy            0.59           0.26                  0.47
        H31 healthy            0.32           0.26                  0.99
        H30 diverged           5.30           5.53                168.72

    Proprioception sits closest to 1.0 in healthy runs, because where a limb will be in
    50 ms is nearly as predictable as where it is now; the other two have real headroom.

    RESIDUAL PREDICTION breaks that premise (2026-08-31). With --residual-prediction on
    the heads predict the CHANGE and the present is added back, so next-state error
    collapses by construction and reconstruction sits above it in a healthy model -
    measured 1.16x the day the flag landed (H36) drifting to 2.45x (H43), while the
    decoders trained at their 0.02 floor weight throughout. Against that yardstick the
    old rule fires on every healthy residual run. Under residual the fault is instead a
    BLOW-UP against the run's own history: the H30 damage signature was 0.007 -> 0.9
    within a window (128x), so a last-window value over 4x the first-window value and
    over 0.05 absolute is damage, and the slow 2x era drift stays quiet.
    """
    out = {}
    for m, (_, b_now) in (next_win or {}).items():
        r = (recon_win or {}).get(m)
        if not r or not b_now:
            continue
        if residual:
            inv = bool(r[1] > 4 * max(r[0], 1e-9) and r[1] > 0.05)
        else:
            inv = r[1] > b_now
        out[m] = {"recon": r[1], "next": b_now, "ratio": r[1] / b_now,
                  "baseline": r[0], "residual": bool(residual), "inverted": inv}
    return out


def drift_channels(recs, source_mask_keys=("audio", "intermind")):
    """Reconstruction error on the channels that carry NO loss weight.

    source_scales sets synthetic to 0.0, so audio and intermind heads receive no
    gradient at all. Their error therefore tracks the shared trunk's output scale rather
    than any learning, which makes them the earliest visible sign of drift: on H30
    intermind reached 110x its healthy value while vision, tactile and proprioception still
    looked fine, and the real senses only followed later.
    """
    if not recs:
        return {}
    last = recs[-1].get("pe") or {}
    first = next((r.get("pe") for r in recs if r.get("pe")), {}) or {}
    out = {}
    for m in source_mask_keys:
        if m in last and m in first and first[m]:
            out[m] = {"now": last[m], "start": first[m], "ratio": last[m] / first[m]}
    return out


def split_at_resume(recs):
    """Records since the last process restart, plus how many restarts the log holds.

    A resumed run writes into the SAME folder, so one log can hold several
    configurations. H30 was relaunched with imagination switched on at step 708475 and
    the window comparison then straddled the change: its "first window" was entirely
    imagination-off and its "last window" entirely imagination-on, so the trend it
    printed was partly a config difference, not progress.

    A restart is detectable without guessing: elapsed_s counts from the start of the
    PROCESS, so it drops when a new one begins. Measured on H30 - record 69, step
    708475 elapsed 1645s, then step 708500 elapsed 18s.
    """
    seams = [i for i in range(1, len(recs))
             if (recs[i].get("elapsed_s") or 0) < (recs[i - 1].get("elapsed_s") or 0)]
    if not seams:
        return recs, 0, None
    i = seams[-1]
    return recs[i:], len(seams), recs[i].get("step")


def tidy_html(page: str) -> str:
    """A missing value renders as a dash, never as the word None."""
    return page.replace("<td>None</td>", "<td>&mdash;</td>")


def window_means(recs, field, keys):
    """(k, {key: (first-k mean, last-k mean)}) over ONE configuration only.

    Two separate guards, both learned the hard way:

    Single-record endpoints flip signs. On H28 tactile next-state error read 48% BETTER
    comparing first and last records, and 36% WORSE over 100-record windows. So every
    now-vs-start figure on this page averages a window.

    And the window must not cross a restart - see split_at_resume. Callers pass records
    already trimmed to the current process.
    """
    k = max(1, min(100, len(recs) // 4))
    out = {}
    for key in keys:
        a = [r[field][key] for r in recs[:k] if isinstance(r.get(field), dict) and r[field].get(key) is not None]
        b = [r[field][key] for r in recs[-k:] if isinstance(r.get(field), dict) and r[field].get(key) is not None]
        if a and b:
            out[key] = (sum(a) / len(a), sum(b) / len(b))
    return k, out


def self_causation_state(vals, rec_gate=None):
    """Plain-language state of the body map's co-movement channel (movement.self_factor), computed from the log.
    (The function keeps its name of 2026-08; the page's name for the statistic is co-movement since 2026-10-03:
    it counts command changes and proprioception changes in the same model step, not causation.)

    When body-schema updating is off the model passes 1.0 into the movement drive as a
    neutral fallback, so a constant 1.00 is a bypass, not a measurement. Newer runs log
    the gate explicitly ('gates' in the record); older runs are detected by the
    constant. gates_from reads the first word of the returned text ("measured", "DISABLED", "no data").
    """
    if rec_gate is not None:
        return ("DISABLED by choice - the co-movement statistic cannot separate movement the command "
                "caused from movement it did not, so the movement drive is fed a neutral 1.00; the controlled "
                "situation test answers this question instead (see its row)"
                if str(rec_gate).lower() != "active" else
                "measured: the body map's co-movement statistic is on (--body-schema-updates on); it counts "
                "command changes and proprioception changes in the same model step, not causation")
    if not vals:
        return "no data"
    if max(abs(v - 1.0) for v in vals) < 1e-9:
        return ("DISABLED by choice - constant 1.00 is the neutral feed, not a "
                "measurement; the controlled situation test answers this question "
                "instead (see its row)")
    return (f"measured: the handles' mean co-movement number read {fixed(100 * min(vals), 2)}% to "
            f"{fixed(100 * max(vals), 2)}% over {len(vals)} reports (its output's two ends: 7.6% = a handle's "
            f"command change and proprioception's change never fell in the same model step, 92.4% = their sizes "
            f"matched on every model step; co-movement, not the share of movement the command caused)")


def gates_from(args, last, selfc):
    """(switches, settings): every gate and setting with its state, from args + records.

    Newer runs write a 'gates' dict into each record; anything not recorded falls back
    to args.json. Nothing here is asserted from memory - a gate with no source reads
    'not recorded'.
    """
    rg = last.get("gates") or {}

    def sw(name):
        if name in rg:
            v = rg[name]
        elif name == "babble":
            v = None if "no_babble" not in args else ("off" if args.get("no_babble") else "on")
        elif name == "body_schema":
            # match the states self_causation_state actually emits ("DISABLED by
            # choice...", "measured, 0.4..0.9...", "no data") - the old prefixes
            # ("BYPASSED", "measured by") matched nothing and every run read unknown
            return ("bypassed" if selfc.startswith(("BYPASSED", "DISABLED")) else
                    "active" if selfc.startswith("measured") else
                    "no data yet" if selfc.startswith("no data") else "unknown")
        else:
            v = args.get(name)
        if v is None:
            return "not recorded"
        if isinstance(v, bool):
            return "on" if v else "off"
        return str(v)

    switches = [(n, sw(n), d) for n, d in GATE_DEFS]
    # planner gates only mean something when intention is on
    intent_on = str(args.get("intention", "off")).lower() not in ("off", "0", "false", "none")
    for n, d in (("plan depth cap", "how many imagined steps a plan may chain"),
                 ("scale gate", "fail-closed: an out-of-scale imagined state aborts the plan"),
                 ("plan safety exit", "abandons a plan whose imagined wellness cannot be judged")):
        # "armed" was status shorthand (the style guide writing rules): the state says what makes it on
        switches.append((n, "on (the planner is on)" if intent_on else "inactive (intention off)", d))
    settings = []
    for n, d in SETTING_DEFS:
        v = rg.get(n, last.get(n) if n in ("rom_limit", "strength_limit", "delays") else args.get(n))
        settings.append((n, "not recorded" if v is None else str(v), d))
    return switches, settings


def world_model_for(run_dir):
    """Latest world-model battery result (three questions in the operator's terms)."""
    cands = sorted(glob.glob(os.path.join(run_dir, "world_model", "battery_*.json")),
                   key=os.path.getmtime)
    if not cands:
        return None
    try:
        return json.load(open(cands[-1], encoding="utf-8"))
    except Exception:
        return None


def provenance_for(run_dir):
    """Which code the trainer actually loaded, from the newest checkpoint's lineage.json.

    A clean working tree does not prove an hours-old process loaded it: H28's tree was
    green at 790fd7db while its checkpoints record code_commit 7ac7ac3a, eight commits
    earlier. This is the row that catches that.
    """
    best, best_step = None, -1
    for p in glob.glob(os.path.join(run_dir, "checkpoints", "*", "lineage.json")):
        m = re.search(r"ckpt_step_(\d+)", p)
        s = int(m.group(1)) if m else 0
        if s > best_step:
            best, best_step = p, s
    if not best:
        return None
    lin = load(best)
    if not lin:
        return None
    head = git_head()
    rc = (lin.get("code_commit") or "")[:8]
    return {"code_commit": rc or None, "head": head,
            "match": bool(rc and head and rc == head[:8]),
            "resumed_from": os.path.basename(str(lin.get("resumed_from") or "")) or None,
            "lineage_step": best_step}


def preflight_probe(run_dir):
    """Launch-time effect probes: total pass count plus the limb-servo cosine."""
    pf = load(os.path.join(run_dir, "preflight.json"))
    if not pf:
        return None
    checks = pf.get("checks") or []
    out = {"ok": pf.get("ok"), "n_ok": sum(1 for c in checks if c.get("ok")), "n": len(checks)}
    for c in checks:
        if "limb follows a command" in c.get("name", ""):
            out["limb_cos"] = c.get("value")
        if c.get("name", "").startswith("face up"):
            out["face_up_deg"] = c.get("value")
    return out


def causal_result_for(run_dir):
    """Newest five-control causal test result for this run's checkpoints, if any ran."""
    # causal_test writes beside the --ckpt directory, so the file lands in
    # checkpoints/ OR checkpoints/<ckpt>/ depending on what was passed to --ckpt
    cands = (glob.glob(os.path.join(run_dir, "causal_test.json"))
             + glob.glob(os.path.join(run_dir, "checkpoints", "causal_test.json"))
             + glob.glob(os.path.join(run_dir, "checkpoints", "*", "causal_test.json")))
    if not cands:
        return None
    p = max(cands, key=os.path.getmtime)
    rep = load(p)
    if not rep:
        return None
    # naming the controls it lost to is the difference between "FAIL" and a result:
    # losing only to persistence is a still body; losing to shuffled means the action
    # input carries no weight at all.
    lost = sorted({c for m in (rep.get("summary") or {}).values() if isinstance(m, dict)
                   for c, won in (m.get("beats") or {}).items() if not won})
    return {"pass": bool(rep.get("pass")), "where": os.path.relpath(p, run_dir),
            "lost_to": lost, "summary": rep.get("summary") or {}}


# The fields describe_run reads from EVERY record (the rest it reads from the newest one).
# Kept per record, read incrementally: before 2026-09-22 every 30-second cycle re-parsed every
# run's whole stats file (796 MB across 277 folders, 25.2 CPU-seconds per cycle for this page
# alone, measured 2026-09-22 in a separate process).
_COMPACT_KEYS = ("step", "elapsed_s", "pe", "next_pe", "recon_dec", "task_hand_ball_dist_m",
                 "roll_deg_max", "nan_steps", "follow_bout")


def _compact(r):
    out = {k: r[k] for k in _COMPACT_KEYS if k in r}
    mv = r.get("movement")
    if isinstance(mv, dict) and "self_factor" in mv:
        out["movement"] = {"self_factor": mv["self_factor"]}
    it = r.get("intention")
    if isinstance(it, dict) and it:
        out["intention"] = {"plan_length": it.get("plan_length")}
    tt = r.get("tenure_tok")
    if isinstance(tt, dict) and "rank1" in tt:
        out["tenure_tok"] = {"rank1": tt["rank1"]}
    return out


def _source_tag(fn):
    """A tag of one function's source text, taken at import: the compact cache (lineage_series
    read_rows, which adds its own reader's tag) is re-read when _compact changes, and only then
    (fix-readout-dashboards 2026-09-23: the cache names carried no code version)."""
    import hashlib
    import inspect
    try:
        return hashlib.sha1(inspect.getsource(fn).encode("utf-8")).hexdigest()[:16]
    except (OSError, TypeError):
        return ""


_COMPACT_SIG = "legacy_compact_v1:" + ",".join(_COMPACT_KEYS) + ":" + _source_tag(_compact)


def compact_records(stats):
    """Every `mimo` record of a stats file, reduced to the fields read across records,
    read incrementally (lineage_series.read_rows: byte offsets cached per file)."""
    from mimo_bridge.tools.lineage_series import read_rows
    rows, _meta, _stats = read_rows(stats, os.path.join(SAVED, "lineage_series", "legacy"),
                                    extractor=_compact, sig=_COMPACT_SIG)
    return [row[2] for row in rows]


def tail_record(stats):
    """The newest complete `mimo` record, from a bounded tail (the full record)."""
    from mimo_bridge.tools.development_dashboard import last_training_record
    return last_training_record(stats) or None


def read_jsonl(path, kind):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") == kind:
                    out.append(r)
    except FileNotFoundError:
        pass
    return out


def load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def saved_date(path):
    try:
        return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))
    except OSError:
        return 'unknown date'


def spark(xs, ys, label):
    """Tiny PNG line chart as a data URI (empty string if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    fig, ax = plt.subplots(figsize=(3.2, 1.1), dpi=100)
    ax.plot(xs, ys, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(label, fontsize=7, loc="left")
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stop_reason_text(run_dir):
    """The watcher program's stop reason for a run, in plain words, or None."""
    stop = load(os.path.join(run_dir, "stop_reason.json"))
    if not isinstance(stop, dict):
        return None
    from mimo_bridge.tools.lineage_series import plain_stop_reason, stop_writer
    when = (time.strftime("%Y-%m-%d %H:%M", time.localtime(stop["timestamp"]))
            if isinstance(stop.get("timestamp"), (int, float)) else "an unrecorded time")
    return (f"stopped by {stop_writer(stop)} at {when}: {plain_stop_reason(stop.get('reason'))}"
            + ("; it waits for a repair before any resume" if stop.get("resume_policy") == "repair_required" else ""))


def run_status(info, run_dir, stats, summ, last, fresh):
    """status, proc_min and steps. A stop by the watcher program's stop rule is named as that,
    never as "finished" (seg31 read "finished" on 2026-09-22 after a step-rate stop)."""
    stop = None if fresh else stop_reason_text(run_dir)
    if summ is not None and not fresh:
        info["status"] = ("stopped by a stop rule" if stop else "finished" if summ.get("ok") else "crashed")
        info["proc_min"] = (summ.get("elapsed_s") or 0) / 60
        info["steps"] = summ.get("steps")
        if not summ.get("ok") and summ.get("exception"):
            info["error"] = summ["exception"].strip().splitlines()[-1][:200]
    else:
        age_s = time.time() - os.path.getmtime(stats) if os.path.exists(stats) else 1e9
        info["status"] = ("running" if age_s < 180 else "stopped by a stop rule" if stop
                          else "stopped (no news for %d min)" % (age_s / 60))
        info["proc_min"] = (last.get("elapsed_s") / 60) if last and last.get("elapsed_s") else None
        info["steps"] = last["step"] if last else 0
    if stop:
        info["stop_reason"] = stop
    from mimo_bridge.tools.lineage_series import motor_watch_note
    watch = motor_watch_note(SAVED, run=os.path.basename(run_dir))
    if watch:
        info["motor_watch"] = watch


def describe_light(run_dir, role="ancestor"):
    """A one-line history entry for a run on the lineage: status, steps, phase, newest save.
    Reads summary.json, stop_reason.json, checkpoints.json and the stats file's tail only."""
    name = os.path.basename(run_dir)
    stats = os.path.join(run_dir, "replay_training_stats.jsonl")
    args = load(os.path.join(run_dir, "args.json"), {})
    info = {"name": name, "bodies": args.get("n_envs"), "scene": args.get("scene"), "args": args,
            "run_dir": run_dir, "role": role,
            "mtime": os.path.getmtime(stats) if os.path.exists(stats) else 0.0}
    last = tail_record(stats) if os.path.exists(stats) else None
    fresh = os.path.exists(stats) and (time.time() - os.path.getmtime(stats)) < 240
    run_status(info, run_dir, stats, load(os.path.join(run_dir, "summary.json")), last, fresh)
    kept = KEPT_STEP.get(_norm(run_dir))
    if kept is not None:
        info["kept_step"] = kept
    if last:
        info["phase"] = PHASE_WORDS.get(last.get("phase"), last.get("phase"))
    ck = load(os.path.join(run_dir, "checkpoints.json"), [])
    if ck:
        info["last_snapshot"] = ck[-1].get("time")
    return info


def describe_run(run_dir):
    name = os.path.basename(run_dir)
    stats = os.path.join(run_dir, "replay_training_stats.jsonl")
    # a real clock to sort runs by. The old key was `last_snapshot or name`, which mixes a
    # timestamp string with a directory name - and a name beats a timestamp in ASCII, so any
    # run that never snapshotted outranked every run that did. The full panel was showing
    # H8_ctrl_driveoff, dead for two days with 0 steps, as "the most recent finished run".
    info_mtime = os.path.getmtime(stats) if os.path.exists(stats) else 0.0
    args = load(os.path.join(run_dir, "args.json"), {})
    summ = load(os.path.join(run_dir, "summary.json"))
    recs = compact_records(stats) if os.path.exists(stats) else []
    ck = load(os.path.join(run_dir, "checkpoints.json"), [])
    info = {"name": name, "bodies": args.get("n_envs"), "minutes_planned": args.get("minutes"),
            "scene": args.get("scene"), "mtime": info_mtime, "args": args, "run_dir": run_dir}
    fresh = os.path.exists(stats) and (time.time() - os.path.getmtime(stats)) < 240
    # folder age and process runtime are different clocks: a reused folder is older than
    # the process that last wrote to it, and the old page mixed them into one number
    try:
        info["folder_min"] = (time.time() - os.path.getctime(run_dir)) / 60
    except OSError:
        info["folder_min"] = None
    run_status(info, run_dir, stats, summ, recs[-1] if recs else None, fresh)
    if recs:
        # the full newest record (every field); recs holds only the fields read across records
        last = tail_record(stats) or recs[-1]
        info["rec"] = last
        info["phase"] = PHASE_WORDS.get(last.get("phase"), last.get("phase"))
        info["age"] = last.get("age_months")
        try:
            ar = int(args.get("action_repeat") or 1)
        except Exception:
            ar = 1
        info["ar"] = ar
        info["model_sps"] = last.get("model_steps_per_s")
        info["dt"] = last.get("env_dt_s", 0.01)
        # lineage step vs this run's own steps: the counter is restored on resume, so
        # the big number is the brain's whole life, not this run
        rs = parse_resume_step(args.get("resume"))
        info["resume_step"] = rs
        last_step = last.get("step") or 0
        info["new_steps"] = last_step - (rs or 0)
        # windows must not straddle a restart: one folder can hold several
        # configurations, and H30 was relaunched with imagination on mid-log
        seg, n_seams, seam_step = split_at_resume(recs)
        info["n_restarts"], info["seam_step"] = n_seams, seam_step
        info["seg_records"] = len(seg)
        # `pe` is PREDICTED CHANGE (the heads' guess of the next instant against the
        # present), not reconstruction. Reconstruction is the decoders' error, logged per
        # sense as recon_dec since 2026-08-27; older logs carry only the decoders' mean in
        # their training records. Before this the fault rule read `pe`, and on H34 it
        # reported reconstruction broken while next-state error was the best on record.
        k, info["change_win"] = window_means(seg, "pe", ("vision", "tactile", "proprioception"))
        _, info["recon_win"] = window_means(seg, "recon_dec", ("vision", "tactile", "proprioception"))
        info["recon_source"] = "decoders, per sense" if info["recon_win"] else None
        if not info["recon_win"]:
            trs = [t for t in read_jsonl(stats, "training")
                   if isinstance((t.get("loss_breakdown") or {}).get("recon"), (int, float))]
            if trs:
                kk = max(1, min(100, len(trs) // 4))
                vals = [float(t["loss_breakdown"]["recon"]) for t in trs]
                info["recon_mean_win"] = (sum(vals[:kk]) / kk, sum(vals[-kk:]) / kk)
                info["recon_source"] = "decoders, mean over senses (training records)"
        _, info["next_win"] = window_means(seg, "next_pe", ("vision", "tactile", "proprioception"))
        info["win_k"] = k
        info["follow"] = last.get("cmd_response_window_cos", last.get("cmd_response_cos"))
        info["follow_slow"] = last.get("cmd_response_cos_held", last.get("cmd_response_cos_250ms"))
        # under rest-bouts the raw held cosine goes negative BY CONSTRUCTION during
        # rest windows (the hold-still command opposes body drift), so obedience is
        # judged on movement bouts only, window-meaned - never one record
        fb = [r.get("follow_bout") for r in seg[-72:]]
        fb = [v for v in fb if v is not None]
        info["follow_bout_win"] = (sum(fb) / len(fb)) if fb else None
        info["ratio"] = last.get("cmd_response_ratio")
        info["pos_cos"] = last.get("cmd_response_pos_cos")
        info["contract"] = last.get("contract")
        info["ball_hand"] = last.get("task_ball_touch_hand_rate")
        info["gaze_on"] = last.get("task_gaze_on_ball_rate")
        info["gaze_earn"] = last.get("task_gaze_on_ball_earned_rate")
        info["gaze_deg"] = last.get("task_gaze_ball_deg")
        info["movement"] = last.get("movement")
        sf_vals = [r["movement"]["self_factor"] for r in recs
                   if isinstance(r.get("movement"), dict) and r["movement"].get("self_factor") is not None]
        info["selfc"] = self_causation_state(sf_vals, (last.get("gates") or {}).get("body_schema"))
        info["gates"] = gates_from(args, last, info["selfc"])
        info["preflight"] = preflight_probe(run_dir)
        info["world_model"] = world_model_for(run_dir)
        info["provenance"] = provenance_for(run_dir)
        info["causal_test"] = causal_result_for(run_dir)
        info["cmd_path_w"] = command_pathway_weight(run_dir)
        info["action_read"] = action_readability_for(run_dir)
        info["residual"] = (args or {}).get("residual_prediction") == "on"
        info["recon_health"] = reconstruction_health(info["recon_win"], info["next_win"],
                                                     residual=info["residual"])
        rmw = info.get("recon_mean_win")
        nwv = list((info.get("next_win") or {}).values())
        if rmw and nwv:
            nb = sum(v[1] for v in nwv) / len(nwv)
            _res = info.get("residual")
            info["recon_health_mean"] = {
                "recon": rmw[1], "next": nb,
                "ratio": (rmw[1] / nb) if nb else float("nan"),
                "baseline": rmw[0], "residual": bool(_res),
                "inverted": (bool(rmw[1] > 4 * max(rmw[0], 1e-9) and rmw[1] > 0.05)
                             if _res else (bool(nb) and rmw[1] > nb))}
        info["drift"] = drift_channels(seg)
        ints = [r["intention"]["plan_length"] for r in recs
                if isinstance((r.get("intention") or {}).get("plan_length"), (int, float))]
        info["plan_len"] = (sum(ints[-20:]) / len(ints[-20:])) if ints else None
        info["ball_dist"] = last.get("task_hand_ball_dist_m")
        info["ball_rate"] = last.get("task_ball_touch_rate")
        first_bd = next((r.get("task_hand_ball_dist_m") for r in recs if r.get("task_hand_ball_dist_m") is not None), None)
        info["ball_dist_first"] = first_bd
        info["box_rates"] = last.get("task_box_touch_rate")
        info["box_hand"] = last.get("task_box_touch_hand_rate")
        info["discrim"] = last.get("motor_discrimination_ratio")
        info["discrim_between"] = last.get("motor_discrimination_between")
        info["discrim_within"] = last.get("motor_discrimination_within")
        info["fresh"] = last.get("vision_fresh_fraction")
        info["walign"] = last.get("wellness_align_err")
        # body time = this run's own steps x its action_repeat x 10 ms
        info["baby_min"] = max(info["new_steps"], 0) * ar * 0.01 / 60.0
        for k2 in ("hand_view", "hand_seen", "hand_fov", "eye_pitch", "unseen_s",
                   "rescues", "snaps", "saccades", "crossed", "pressing", "press_kpa",
                   "reflex_on", "mobile_speed", "mobile_hit", "mobile_hand",
                   "ring_snaps", "squeaker_hits", "squeaker_foot", "dragonfly_deg",
                   "dragonfly_darts", "dragonfly_hover", "bee_deg", "antic_ok",
                   "antic_speed", "squeaker_view", "head_tilt", "squeaker_seen",
                   "headlook"):
            info[k2] = last.get("task_" + k2)
        info["hself"] = last.get("hself")
        info["entity_self"] = last.get("entity_self")
        info["intention_meta"] = last.get("intention")
        info["eyes"] = (args.get("eye_size") or 256, args.get("fovea_size") or 0,
                        args.get("fovea_fov") or 0.0, args.get("caregiver_seconds") or 0.0,
                        args.get("gaze_reflex", "on")) if args else None
        info["roll"] = last.get("roll_deg_max")
        info["rolled_frac"] = last.get("rolled_over_fraction")
        rolls = [r.get("roll_deg_max") for r in recs if r.get("roll_deg_max") is not None]
        info["roll_best"] = max(rolls) if rolls else None
        info["curiosity"] = last.get("curiosity_drive")
        info["contacts"] = last.get("contacts_per_env")
        info["turnover"] = last.get("contact_turnover")
        info["vicreg"] = last.get("vicreg")
        # ---- action knowledge (2026-09-02): the tenure lessons and the agency drive
        info["tenure_pe_pose"] = (last.get("tenure_pe") or {}).get("pose")
        info["tenure_gap_pose"] = (last.get("tenure_gap") or {}).get("pose")
        _tt = last.get("tenure_tok") or {}
        info["rank1"] = _tt.get("rank1")
        info["n_decoys"] = _tt.get("n_decoys")
        info["vis_gate"] = _tt.get("vis_gate_open")
        info["tenure_events"] = last.get("tenure_events")
        info["agency_g"] = last.get("agency")
        info["td_gate_g"] = last.get("td_gate")
        info["plan_lic"] = last.get("plan_license")
        try:
            import json as _json
            _stp = os.path.join(run_dir, "situation_test.json")
            info["sit_test"] = _json.load(open(_stp)) if os.path.exists(_stp) else None
        except Exception:
            info["sit_test"] = None
        _r1 = [v for v in ((r.get("tenure_tok") or {}).get("rank1") for r in seg)
               if v is not None]
        if len(_r1) >= 8:
            _kk = max(1, min(100, len(_r1) // 4))
            info["rank1_win"] = (sum(_r1[:_kk]) / _kk, sum(_r1[-_kk:]) / _kk)
        info["nan"] = sum(r.get("nan_steps", 0) for r in recs)
        # 2026-09-18: hourly saves that raised in this segment, and the trainer's own peak
        # CUDA allocation in the newest window (nvidia-smi adds the renderers on top)
        info["ckpt_failures"] = last.get("checkpoint_failures")
        info["ckpt_last_error"] = last.get("checkpoint_last_error")
        info["peak_vram_mb"] = last.get("peak_vram_mb")
        xs = [r["step"] for r in recs]
        ys = []
        for r in recs:
            pe = r.get("pe", {})
            vals = [pe[m] for m in ("vision", "tactile", "proprioception") if m in pe]
            ys.append(max(1e-6, sum(vals) / len(vals)) if vals else 1e-6)
        info["spark"] = spark(xs, ys, "reconstruction error (real senses), lower = better")
    if ck:
        info["last_snapshot"] = ck[-1].get("time")
        info["n_snapshots"] = len(ck)
    vids = sorted([v for v in glob.glob(os.path.join(run_dir, "videos", "*.mp4")) if not v.endswith("_eye.mp4")],
                  key=os.path.getmtime)
    if vids:
        info["video"] = os.path.relpath(vids[-1], SAVED).replace("\\", "/")
        eye = vids[-1].replace(".mp4", "_eye.mp4")
        if os.path.exists(eye):
            info["video_eye"] = os.path.relpath(eye, SAVED).replace("\\", "/")
        info["n_videos"] = len(vids)
        info["video_gallery"] = [os.path.relpath(v, SAVED).replace("\\", "/") for v in vids[-8:]]
    views = sorted(glob.glob(os.path.join(run_dir, "views", "*.png")), key=os.path.getmtime)
    if views:
        info["views_img"] = os.path.relpath(views[-1], SAVED).replace("\\", "/")
        info["views_gallery"] = [os.path.relpath(v, SAVED).replace("\\", "/") for v in views[-8:]]
    ol = read_jsonl(os.path.join(run_dir, "overseer_log.jsonl"), "overseer")
    if ol:
        last_o = ol[-1]
        notes = []
        # the writing hat's gated copy when present (reading gate); raw findings otherwise
        src_notes = last_o.get("notes_plain") or last_o.get("findings", {})
        for hat, label in (("verdict", "VERDICT"), ("cto", "CTO"), ("researcher", "AI researcher"), ("designer", "Sim designer")):
            for x in (src_notes.get(hat) or []):
                notes.append(f"[{label}] {x}")
        for a in last_o.get("actions") or []:
            notes.append(f"[action] {a}")
        info["overseer"] = notes[:8]
        info["overseer_time"] = time.strftime("%H:%M", time.localtime(last_o.get("timestamp", 0)))
    return info


def stale_note(note, infos):
    """Run names mentioned in an operator note that are not live right now.

    The "H23b: transition contract live" banner sat on the page through H24-H28
    because --note is free text nobody re-checked. A note naming a run that is not
    running is flagged rather than displayed as current status.
    """
    live = {i["name"] for i in infos if i.get("status") == "running"}
    named = set(re.findall(r"\bH\d+[A-Za-z_]*\b", note))
    return sorted(n for n in named if not any(n == L or L.startswith(n + "_") for L in live))


def _chip(txt, cls, key=""):
    d = f' data-doc="{key}" role="button" tabindex="0"' if key in DOC else ""
    return f'<span class="chip {cls}{" ck" if key in DOC else ""}"{d}>{html.escape(txt)}</span>'


def chips_for(i):
    """The few computed numbers that decide anything.

    Colour states a FACT - a sign, a match, a pass - never a grade against a threshold
    invented here. Every chip carries an explanation keyed into DOC.
    """
    chips = []
    pv = i.get("provenance")
    if pv and pv.get("code_commit"):
        if pv.get("match"):
            chips.append(_chip(f"code: run == HEAD ({pv['code_commit']})", "g", "chip:code"))
        elif pv.get("head"):
            chips.append(_chip(f"code: run {pv['code_commit']} != HEAD {pv['head']}", "b", "chip:code"))
        else:
            chips.append(_chip(f"code: run {pv['code_commit']} (HEAD unknown)", "w", "chip:code"))
    ct = i.get("contract")
    if ct and i.get("new_steps"):
        seen, lineage_step = ct.get("exec_seen", 0) or 0, ct.get("step", 0) or 0
        new = lineage_step - (i.get("resume_step") or 0)
        # H84's counter is carried through resumes (344,399 seen at lineage step 344,400 on
        # seg31, whose own steps were 16,025): divided by this segment it read 2149.1%
        if seen > new:
            frac, span = seen / max(1, lineage_step), f"of the lineage's {lineage_step:,} steps"
        else:
            frac, span = seen / max(1, new), f"of this segment's {new:,} steps"
        chips.append(_chip(f"action ledger {frac:.1%} {span}", "g" if frac >= 0.99 else "b",
                           "chip:ledger"))
    cw = i.get("cmd_path_w")
    ar = i.get("action_read")
    if cw is not None and cw == 0.0:
        chips.append(_chip("action pathway INERT (weights exactly 0)", "b", "chip:cmdpath"))
    elif ar is not None:
        # the reading is on the INPUT, one body part at a time (reviews/dashboards.md M9): the
        # all-at-once test read green while 10 of 26 parts could not be read at all
        ph = ar.get("per_handle")
        if not ar.get("readable"):
            chips.append(_chip(f"action DROWNED: even changing every body part's command at once moves the "
                               f"motor token {fixed(100 * ar['diff_action'] / max(ar['pos_block'], 1e-9), 2)}% "
                               f"as far as zeroing the command's position block", "b", "chip:cmdpath"))
        elif ph:
            below = ph.get("below") or []
            n = len(ph.get("per_handle") or [])
            line = per(ph["line"], "parts of its size") if ph.get("line") else "the measured line"
            if below:
                from mimo_bridge.tools.development_dashboard import handle_words
                chips.append(_chip(f"action UNREAD for {len(below)} of {n} body parts: a 20-degree turn of one "
                                   f"of them moves the motor token under {line} ("
                                   + ", ".join(handle_words(x) for x in below) + ")", "b", "chip:cmdpath"))
            else:
                chips.append(_chip(f"action readable for all {n} body parts: each one's 20-degree turn moves "
                                   f"the motor token above {line}", "g", "chip:cmdpath"))
        else:
            chips.append(_chip(f"all {ar.get('handles') or 26} handles changed at once; per-body-part "
                               f"readability unmeasured", "w", "chip:cmdpath"))
    elif cw is not None:
        chips.append(_chip("action pathway weights are not zero; the input reading is unavailable", "w",
                           "chip:cmdpath"))
    rh = i.get("recon_health") or {}
    bad = [m for m, d in rh.items() if d["inverted"]]
    if bad:
        if any(rh[m].get("residual") for m in bad):
            worst = max(rh[m]["recon"] / max(rh[m].get("baseline") or 1e-9, 1e-9) for m in bad)
            chips.append(_chip(f"RECONSTRUCTION BROKEN on {len(bad)} sense(s), "
                               f"up to {worst:.0f}x its own recent value", "b", "chip:reconfault"))
        else:
            worst = max(rh[m]["ratio"] for m in bad)
            chips.append(_chip(f"RECONSTRUCTION BROKEN on {len(bad)} sense(s), "
                               f"up to {worst:.0f}x next-state error", "b", "chip:reconfault"))
    elif rh:
        worst = max(d["ratio"] for d in rh.values())
        if any(d.get("residual") for d in rh.values()):
            lo, hi = RESIDUAL_UNDAMAGED
            if worst > hi:
                # outside the range the undamaged residual runs measured: not a fault by the jump rule,
                # never green (verify-fix-dashboards 2026-09-23: seg31 read 47 times on proprioception
                # with green "steady" while the lead's sweep found that decoder 17 in 100 units live)
                name = max(rh, key=lambda m: rh[m]["ratio"])
                chips.append(_chip(f"reconstruction {fixed(worst, 2)} times next-state error on "
                                   f"{SENSE_NAMES.get(name, name)}: above the {fixed(lo, 3)} to {fixed(hi, 3)} times "
                                   f"of the undamaged residual runs; no jump, so the fault rule is silent", "w",
                                   "chip:reconfault"))
            else:
                chips.append(_chip(f"reconstruction steady (worst {fixed(worst, 3)} times next-state error; "
                                   f"undamaged residual runs measured {fixed(lo, 3)} to {fixed(hi, 3)} times)", "g",
                                   "chip:reconfault"))
        else:
            chips.append(_chip(f"reconstruction intact (worst {pct(worst, 2)} of next-state error; a working "
                               f"model stays under 100%)", "g", "chip:reconfault"))
    elif i.get("recon_health_mean"):
        hm = i["recon_health_mean"]
        if hm["inverted"]:
            chips.append(_chip(f"RECONSTRUCTION BROKEN (mean over senses), {hm['ratio']:.0f}x next-state error",
                               "b", "chip:reconfault"))
        else:
            chips.append(_chip(f"reconstruction intact ({pct(hm['ratio'], 2)} of next-state error, mean over "
                               f"senses)", "g", "chip:reconfault"))
    elif i.get("next_win"):
        chips.append(_chip("reconstruction not logged on this run", "w", "chip:reconfault"))
    dr = i.get("drift") or {}
    if dr:
        # a ratio with its anchor (the style guide): "x0.8" was unanchored, and a letter before it hid it
        # from the decimal scanner (verify-fix-dashboards 2026-09-23)
        w = max(d["ratio"] for d in dr.values())
        mult = fixed(w, 2)
        chips.append(_chip(f"untrained-channel drift: error {'the same as' if mult == '1' else mult + ' times'} its "
                           f"first value (over 3 times = the shared trunk is drifting)", "b" if w > 3 else "w",
                           "chip:drift"))
    nw = i.get("next_win") or {}
    if nw:
        a = sum(v[0] for v in nw.values()) / len(nw)
        b = sum(v[1] for v in nw.values()) / len(nw)
        chg = 100.0 * (b - a) / a if a else 0.0
        cls = "b" if chg > 2 else ("g" if chg < -2 else "w")
        word = "worse" if chg > 2 else ("better" if chg < -2 else "flat")
        chips.append(_chip(f"causal error {'+' if chg > 0 else ''}{fixed(chg, 2)}% ({word}) from the first "
                           f"reports to the last", cls, "chip:causal"))
    fbw = i.get("follow_bout_win")
    fs = i.get("follow_slow")
    if fbw is not None:
        # bout-only, hour-window mean; measured healthy band 0.006 +/- 0.015, line -0.038.
        # follow_bout is the one-frame (10 ms) gauge read on the last frame of each model step,
        # not the order's 250 ms: its own help key and its own physics reference on those frames
        # (until 2026-09-23 it wore chip:follow, the 250 ms held reading's help)
        from mimo_bridge.tools.lineage_series import FOLLOW
        ref = FOLLOW.get("frame_reachable")
        goal = (f"; a posture the body really reached, commanded on physics alone, reads {fixed(100 * ref, 2)}% "
                f"on the same frames ({fixed(abs(100 * (ref - fbw)), 2)} percentage points "
                f"{'short of' if fbw < ref else 'past'} it)" if ref is not None else
                "; the goal line on these frames is unmeasured, so the distance to it is unmeasured")
        chips.append(_chip(f"follows commands in movement bouts, one 10 ms frame per model step (hour mean): "
                           f"{fixed(100 * fbw, 2)}% of the way from unrelated to exact{goal}",
                           "g" if fbw > 0 else ("b" if fbw < -0.038 else "w"),
                           "chip:follow_bout"))
    elif fs is not None:
        chips.append(_chip(f"follows commands: {fixed(100 * fs, 2)}% of the way from unrelated to exact",
                           "g" if fs > 0 else ("b" if fs < 0 else "w"), "chip:follow"))
    cz = i.get("causal_test")
    if cz is None:
        chips.append(_chip("five-control causal test: not run on this checkpoint", "w", "chip:causaltest"))
    else:
        tail = ("" if cz["pass"] or not cz.get("lost_to") else
                " - loses to " + ", ".join(cz["lost_to"]))
        chips.append(_chip("five-control causal test: " + ("PASS" if cz["pass"] else "FAIL") + tail,
                           "g" if cz["pass"] else "b", "chip:causaltest"))
    gates = i.get("gates")
    if gates:
        sw = gates[0]
        n_on = sum(1 for _, st, _ in sw if st in ("on", "active", "armed") or st.startswith("on ("))
        n_off = sum(1 for _, st, _ in sw if st.startswith(("off", "bypassed", "inactive")))
        chips.append(_chip(f"gates: {n_on} on / {n_off} off", "w", "chip:gates"))
    return '<div class="chips">' + "".join(chips) + "</div>" if chips else ""


def _n(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def tie_meaning(cc):
    """What a planner tie is, beside a tied share (lineage_series.tie_meaning, shared by both pages)."""
    from mimo_bridge.tools.lineage_series import tie_meaning as meaning
    return meaning(cc)


def gof_rows(rec, args=None):
    """Rows for the fields the newest record carries beyond the older rows: the prediction
    audit, wellness channels, consequence, who authored each order, the body map, self-contact,
    and every DESIGN.md section 3 block. Each row appears only when its field is present;
    a missing field is never shown as 0. Every number is a count per 100, a real unit, or a
    percentage between two named ends (DESIGN.md section 11); each measured row names its control."""
    from mimo_bridge.tools.lineage_series import FOLLOW, LP_FROZEN_BAND, SERIES, lead_channel, say
    rec, args = rec or {}, args or {}
    rows = []
    senses = ("vision", "tactile", "proprioception")
    way = lambda v: fixed(100 * v, 2) + "%"                       # noqa: E731
    pa = (rec.get("prediction_audit") or {}).get("modalities") or {}

    def _by_sense(block, key):
        vals = [(SENSE_NAMES[m], _n((block.get(m) or {}).get(key))) for m in senses]
        return [(k, v) for k, v in vals if v is not None]

    def _listed(vals):
        return ", ".join(f"{way(v)} for {k}" for k, v in vals)
    use = _by_sense(pa, "improvement_over_shuffled")
    sized = _by_sense(pa, "improvement_over_size_matched")
    if use or sized:
        # the swap test's three controls and its second phase (mimo_bridge/prediction_audit.py DEFINITIONS)
        shifted = _by_sense(pa, "improvement_over_shifted_own")
        hf = {m: ((((rec.get("prediction_audit_hold_first") or {}).get("senses") or {}).get(m) or {})
                  .get("all_bodies") or {}) for m in senses}
        first_sized = _by_sense(hf, "improvement_over_size_matched")
        parts = []
        if sized:
            parts.append("against another body's command size-matched to this body " + _listed(sized))
        if shifted:
            parts.append("against its own command shifted by one standard deviation of each number "
                         + _listed(shifted))
        if use:
            parts.append(("against another body's absolute command (the original control, retired after one "
                          "segment) " if sized else "against another body's command (the control) ") + _listed(use))
        if first_sized:
            parts.append("at the hold's first model step, against the size-matched command " + _listed(first_sized))
        sens = _by_sense(pa, "sensitivity_size_matched") or _by_sense(pa, "sensitivity")
        rows.append(("Action use in prediction (prediction audit)",
                     "in the newest window, knowing its own executed command improves the next-model-step prediction, "
                     "of the way from 'the command is not used' (the control predicts as well) to 'exact prediction': "
                     + "; ".join(parts) + "; goal line unmeasured"
                     + ("; changing the command moves the prediction " + _listed(sens)
                        + " of the way from 'not at all' to 'as far as the predicted change itself'" if sens else "")))
    pers = [(SENSE_NAMES[m], _n((pa.get(m) or {}).get("improvement_over_persistence"))) for m in senses]
    pers = [(k, v) for k, v in pers if v is not None]
    if pers:
        rows.append(("Better than a copy of the present (prediction audit)",
                     "the next model step is predicted better than \"nothing changes\" by "
                     + ", ".join(f"{way(v)} for {k}" for k, v in pers)
                     + " of the way to exact (0% = no better, below 0% = worse than predicting that nothing "
                       "changes; goal line unmeasured)"))
    well = rec.get("wellness")
    if isinstance(well, dict) and well:
        rows.append(("Wellness channels (window means over the bodies)",
                     ", ".join(f"{k.replace('_', ' ')} {way(v)}" for k, v in well.items() if _n(v) is not None)
                     + " of the way from each channel's lowest to its highest. Their spreads differ, so in a raw "
                       "sum the widest channel's spread picks the goal (addendum item 4)"))
    cq = rec.get("consequence")
    if isinstance(cq, dict) and _n(cq.get("rate")) is not None:
        own = str(args.get("consequence_self_contact", "count")) != "exclude"
        rows.append(("Consequence term in wellness",
                     f"a consequence was present on {per(cq['rate'], 'model steps of a body')}"
                     + (f"; {per(cq['goal_rate'], 'chosen goals')} carried one"
                        if _n(cq.get("goal_rate")) is not None else "")
                     + (". Hand pressing on its OWN body counts as a consequence on this run"
                        if own else ". Only contact with the world counts on this run")))
    src = rec.get("exec_source")
    if isinstance(src, dict) and src:
        names = {"planner": "the planner", "reflex_noise": "reflex plus exploration noise", "reflex": "reflex alone"}
        rows.append(("Who authored the executed command",
                     "of every 100 commands latched in the window: "
                     + ", ".join(f"{names.get(k, k)} {100 * v:.0f}" for k, v in src.items() if _n(v) is not None)))
    inc = rec.get("incorporation")
    if isinstance(inc, dict) and _n(inc.get("extent")) is not None:
        ev = inc.get("objects_evidence") or {}
        toy = ev.get("toy") or {}
        rows.append(("Body map co-movement (incorporation)",
                     f"the mean of the body map's co-movement numbers reads {say(SERIES['extent'], inc['extent'])} "
                     f"(26 handles and the outside things it watches; a handle's number counts command changes "
                     f"and proprioception changes that fall in the same model step, not causation)"
                     + (f"; the toy's movement matches the hand's {way(toy['match'])} of the way from 'unrelated' "
                        f"to 'moves with the hand', against {way(toy['control'])} with the hand moving the opposite "
                        f"way (the control)" if _n(toy.get("match")) is not None
                        and _n(toy.get("control")) is not None else "")))
    pairs = _n(rec.get("self_contact_pairs"))
    if pairs is not None:
        stc = rec.get("self_touch_controlled") or {}
        rows.append(("Self-contact (own body touching itself)",
                     f"pairs of its own body parts touching at an average model step: {say(SERIES['self_pairs'], pairs)} (0 = never)"
                     + (f"; knowing its command improves the prediction of what a self-touching body feels by "
                        f"{way(stc['improvement_over_shuffled'])} of the way from 'another body's command does as well' "
                        f"to 'exact', on {stc.get('rows')} contacts" if _n(stc.get("improvement_over_shuffled")) is not None
                        else "")
                     + (f"; the self-touch wellness term paid {way(rec['wellness']['self_touch'])} of its 0-to-1 range"
                        if _n((rec.get("wellness") or {}).get("self_touch")) is not None else "")))
    cc = rec.get("causal_chain")
    if isinstance(cc, dict):
        alt = _n(cc.get("alternatives_n"))
        best = _n(cc.get("chosen_best_share"))
        tied = _n(cc.get("chosen_tied_share"))
        dist = _n(cc.get("plan_exec_distance_cm"))
        rows.append(("Causal chain of each plan",
                     (f"chose the plan it predicted would feel best in {per(best, 'plans')} "
                      f"({cc.get('planner_events')} plans; chance {per(1 / (1 + alt), 'plans')})"
                      if best is not None and alt is not None else "no plan was scored in the window")
                     + (f"; tied in {per(tied, 'plans')}" + tie_meaning(cc) if tied is not None
                        else "; ties are not reported (a tie counts as the best)")
                     + (f"; the body received a command {fixed(dist, 2)} cm from the one the planner imagined" if dist is not None
                        else "")
                     + "; goal line unmeasured: the planner does not aim at predicted wellness"))
    ef = rec.get("efference")
    if isinstance(ef, dict) and _n(ef.get("body_cm_err_latch")) is not None:
        gz, gd = ef.get("gaze_deg_err") or [], ef.get("gaze_deg_do_nothing") or []
        rows.append(("Efference heads (predicted body and eye movement)",
                     f"where each body part goes over a command: {fixed(ef['body_cm_err_latch'], 2)} cm off"
                     + (f", against {fixed(ef['body_cm_do_nothing_latch'], 2)} cm for predicting no movement"
                        if _n(ef.get("body_cm_do_nothing_latch")) is not None else "")
                     + (f", {fixed(ef['body_cm_size_matched_latch'], 2)} cm with another body's command size-matched "
                        f"to this body (the control)" if _n(ef.get("body_cm_size_matched_latch")) is not None else "")
                     + ((f" and {fixed(ef['body_cm_shuffled_latch'], 2)} cm with another body's absolute command "
                         f"(retired after one segment)" if _n(ef.get("body_cm_size_matched_latch")) is not None else
                         f" and {fixed(ef['body_cm_shuffled_latch'], 2)} cm with another body's command (the control)")
                        if _n(ef.get("body_cm_shuffled_latch")) is not None else "")
                     + (f"; the eyes' angle 150 ms ahead: {fixed(gz[2], 2)} degrees off against {fixed(gd[2], 2)} "
                        f"degrees for predicting the eyes stay (where the eyes point, not what they see: the gaze "
                        f"probe measures that)" if len(gz) > 2 and len(gd) > 2 and _n(gz[2]) is not None
                        and _n(gd[2]) is not None else "")))
    mc = rec.get("machine")
    if isinstance(mc, dict) and mc:
        bits = []
        if _n(mc.get("commit_gb")) is not None and _n(mc.get("commit_limit_gb")) is not None:
            bits.append(f"{mc['commit_gb']:.1f} of {mc['commit_limit_gb']:.1f} GB committed")
        if _n(mc.get("gpu_used_mib")) is not None and _n(mc.get("gpu_total_mib")) is not None:
            bits.append(f"card {mc['gpu_used_mib']:,.0f} of {mc['gpu_total_mib']:,.0f} MiB")
        if _n(mc.get("step_ms")) is not None:
            bits.append(f"{mc['step_ms']:.0f} ms per model step")
        if bits:
            rows.append(("Machine load (memory, card, step time)", "; ".join(bits)))
    ve = rec.get("vestibular")
    if isinstance(ve, dict) and _n(ve.get("improvement_over_persistence")) is not None:
        rows.append(("Vestibular prediction and trunk posture",
                     f"predicted better than \"nothing changes\" by {way(ve['improvement_over_persistence'])} of the "
                     f"way to exact (0% = no better; goal line unmeasured)"
                     + (f"; trunk turned {abs(ve['trunk_roll_deg']):.0f} degrees from lying on its back (90 = on "
                        f"its side, 180 = belly down)" if _n(ve.get("trunk_roll_deg")) is not None else "")))
    ct = rec.get("controllability")
    if isinstance(ct, dict) and _n(ct.get("agreement_mean")) is not None:
        rows.append(("Controllability channel",
                     f"what the body did matched what it expected {way(ct['agreement_mean'])} of the way from "
                     f"'unrelated' to 'exactly as expected'"
                     + (f", against {way(ct['control_mean'])} for the same body's result against another body's "
                        f"command (the control)" if _n(ct.get("control_mean")) is not None else
                        "; no control this window")
                     + (f"; scored on {per(ct['measured_share'], 'commands')}" if _n(ct.get("measured_share")) is not None
                        else "")
                     + ("; its evidence gate is " + ("open" if ct.get("gate_open") else "shut")
                        if ct.get("gate_open") is not None else "")))
    elif isinstance(ct, dict):
        rows.append(("Controllability channel",
                     "no command was scored this window"
                     + (f" ({int(ct['orders']):,} commands graded" if _n(ct.get("orders")) is not None else " (")
                     + (f", {int(ct['efference_missing']):,} without the body head's command-long expectation"
                        if _n(ct.get("efference_missing")) else "") + ")"))
    # the gate in force beside the recorded-only shadow gate, on the same graded commands (2026-10-03,
    # lem/model/wellness_channels.py SHADOW_CONTROL_KIND; the sentence is the Development page's too)
    from mimo_bridge.tools.lineage_series import controllability_gates_text
    gates = controllability_gates_text(ct)
    if gates:
        rows.append(("Controllability gates on the same commands", gates))
    lp = rec.get("learning_progress")
    if isinstance(lp, dict) and _n(lp.get("value_mean")) is not None:
        v = lp["value_mean"]
        ctl = _n(lp.get("control_mean"))
        lo, hi = LP_FROZEN_BAND
        rows.append(("Learning-progress channel",
                     f"prediction errors, measured against \"nothing changes\", moved {say(SERIES['lp.value'], v)} "
                     f"(above 0 = falling, getting better)"
                     + (f", against {say(SERIES['lp.value'], ctl)} on the same-model-step control" if ctl is not None else
                        f"; frozen weights, with no learning possible, read {say(SERIES['lp.value'], lo)} to "
                        f"{say(SERIES['lp.value'], hi)}, so a reading inside that band is not evidence of learning")))
    wr = rec.get("wellness_rank")
    if isinstance(wr, dict) and wr.get("mode"):
        lc = lead_channel(rec)
        ga = wr.get("goal_age") or {}
        rows.append(("Normalised goal ranking",
                     f"mode {wr['mode']}"
                     + (f"; the leading channel, {lc[0].replace('_', ' ')}, carries {100 * lc[1]:.0f} of every 100 parts "
                        f"of the winning goal's lead" + (f" and holds {100 * lc[2]:.0f} of every 100 parts of the weights"
                                                         if lc[2] is not None else "")
                        + " (the parts of the lead add to 100 and can be below 0 or above 100)" if lc else "")
                     + (f"; chosen goals sat on average {way(ga['mean_share_back'])} of the way back through the goal "
                        f"ring (0% = newest, 100% = oldest), {per(ga['oldest_tenth_share'], 'goals')} in the oldest "
                        f"tenth where a choice blind to age puts 10" if _n(ga.get("mean_share_back")) is not None
                        and _n(ga.get("oldest_tenth_share")) is not None else "; goal age not recorded")
                     + (f"; the integrity gate barred {per(wr['integrity_gated_share'], 'candidate model steps')}"
                        if _n(wr.get("integrity_gated_share")) is not None else "")))
    hp = rec.get("hand_press")
    if isinstance(hp, dict) and _n(hp.get("self_share")) is not None:
        rows.append(("Hand pressing split into world and self",
                     (f"a hand presses the world on {per(hp['world_share'], 'model steps')}, "
                      if _n(hp.get("world_share")) is not None else "")
                     + f"its own body on {per(hp['self_share'], 'model steps')}"))
    cx = rec.get("contact_control")
    if isinstance(cx, dict) and cx:
        onset, onset_ctl = _n(cx.get("onset_auc_self")), _n(cx.get("control_onset_auc_self"))
        paired, sctl = _n(cx.get("success_paired")), _n(cx.get("success_control"))
        if onset is not None:
            text = (f"on patches untouched the model step before, it gave the patch about to be touched by its own body "
                    f"the higher chance in {per(onset, 'pairs')} (a coin flip scores 50)"
                    + (f", against {per(onset_ctl, 'pairs')} for a base-rate guess on the same patches and model steps"
                       if onset_ctl is not None else "; the base-rate control is not in this record"))
        elif paired is not None:
            text = (f"it foresaw {per(paired, 'self-touch onsets')} with its own command"
                    + (f", against {per(sctl, 'self-touch onsets')} with another body's command on the same onsets"
                       if sctl is not None else "; no control this window"))
        else:
            text = "no self-touch onset was scored this window"
        if _n(cx.get("habituation_mean")) is not None:
            text += (f"; average pay factor {way(cx['habituation_mean'])} (100% = a new pair of touching parts, "
                     f"lower = repeated)")
        rows.append(("Contact prediction and control", text + "; goal line unmeasured"))
    se = rec.get("self_contact_events")
    if isinstance(se, dict) and _n(se.get("onsets")) is not None:
        rows.append(("Self-touch onsets",
                     f"{say(SERIES['sce.onsets'], se['onsets'])} began"
                     + (f"; {per(se['in_view_share'], 'of them')} within 30 degrees of an eye's axis"
                        if _n(se.get("in_view_share")) is not None else "")))
    sy = rec.get("synergy")
    if isinstance(sy, dict) and (_n(sy.get("following_mean")) is not None or _n(sy.get("explained_share")) is not None):
        # the synergy's own orders only when the record splits following by who wrote the order;
        # following_mean is every graded order, babble and reflex included (verify-fix-dashboards)
        fol = SERIES["syn.following"]["get"](rec)
        every = _n(sy.get("following_mean"))
        who = ("the synergy-written commands" if fol is not None else
               "every command graded this window, whoever wrote it (a stand-in: following is not split by author)")
        fol = fol if fol is not None else every
        rows.append(("Motor synergy map",
                     (f"the body followed {who} {way(fol)} of the way from 'unrelated' to "
                      f"'exactly as commanded'" + (f", against {way(FOLLOW['gauge_reachable'])} for a posture the body "
                                                 f"really reached, on physics alone" if FOLLOW.get('gauge_reachable')
                                                 is not None else "") if fol is not None else "following not recorded")
                     + (f"; beside it, the map reproduces {way(sy['explained_share'])} of the posture changes before "
                        f"learning from them (can be below 0)" if _n(sy.get("explained_share")) is not None else "")
                     + (f"; uses {fixed(sy['effective_dim'], 2)} of its {sy.get('dim')} directions"
                        if _n(sy.get("effective_dim")) is not None else "")))
    nf = rec.get("next_frame_norm")
    if isinstance(nf, dict) and isinstance(nf.get("weight_effective"), dict):
        vals = {k: _n(v) for k, v in nf["weight_effective"].items() if _n(v) is not None}
        total = sum(vals.values())
        if total > 0:
            rows.append(("Next-model-step loss normalisation",
                         "share of the next-model-step lesson each sense carries after normalising: "
                         + ", ".join(f"{SENSE_NAMES.get(k, k)} {100 * v / total:.0f} of every 100" for k, v in vals.items())))
    return rows


def render(infos, extra_notes):
    dyn_docs = {}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    running = [i for i in infos if i.get("status") == "running"]
    parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>MIMo training dashboard</title>
<style>body{{font:15px/1.45 Segoe UI,Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafbfd;color:#1c2430}}
h1{{margin:0 0 4px}} .sub{{color:#55627a;margin-bottom:16px}}
.card{{background:#fff;border:1px solid #d9e0ea;border-radius:8px;padding:12px 16px;margin:10px 0;display:grid;grid-template-columns:1fr 340px;gap:12px}}
.run{{border-left:6px solid #2f5bb0}} .fin{{border-left:6px solid #1a7f37}} .bad{{border-left:6px solid #b42318}}
.name{{font-weight:700;font-size:16px}} .tag{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-left:6px;border:1px solid #ccc}}
.t-run{{background:#e8f0ff;color:#2f5bb0}} .t-fin{{background:#e7f6ec;color:#1a7f37}} .t-bad{{background:#fdecea;color:#b42318}}
table{{border-collapse:collapse;font-size:13px;margin-top:6px}} td,th{{padding:2px 8px;text-align:left;border-bottom:1px solid #eee}}
.muted{{color:#6b7790}} img{{max-width:100%}} .legend{{background:#fff;border:1px solid #d9e0ea;border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:14px}}
.chips{{margin:6px 0 2px}}
.chip{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12.5px;font-weight:600;margin:2px 6px 2px 0;border:1px solid}}
.chip.g{{background:#e7f6ec;color:#15662c;border-color:#bfe3cb}}
.chip.b{{background:#fdecea;color:#8f1d12;border-color:#f2c7c2}}
.chip.w{{background:#f1f3f7;color:#4a566e;border-color:#d5dbe6}}
details.sect{{margin:6px 0 2px}} details.sect summary{{cursor:pointer;font-weight:600;font-size:13px;color:#3b4a63}}
tr.ck{{cursor:pointer}} tr.ck:hover{{background:#f2f6ff}} tr.ck th::after{{content:" ?";color:#8a97ad;font-weight:400}}
.chip.ck{{cursor:pointer}} .chip.ck:hover{{filter:brightness(0.96)}}
#ov{{display:none;position:fixed;inset:0;background:rgba(16,22,34,.55);z-index:99;padding:24px;overflow:auto}}
#ov.on{{display:block}}
#bx{{background:#fff;max-width:680px;margin:6vh auto;border-radius:10px;padding:20px 24px;box-shadow:0 12px 40px rgba(0,0,0,.3)}}
#bx h3{{margin:0 0 2px;font-size:18px}} #bx .lb{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#7a869c;margin:14px 0 3px}}
#bx p{{margin:0;font-size:14px;line-height:1.5}} #bx .fool{{background:#fdf3f2;border-left:3px solid #d97b70;padding:8px 12px;border-radius:0 4px 4px 0}}
#bx button{{margin-top:18px;font:inherit;padding:6px 14px;border:1px solid #ccd3e0;background:#f7f9fc;border-radius:6px;cursor:pointer}}
</style></head><body>
<h1>MIMo training dashboard</h1>
<p><a href="dashboard.html">Open the new development dashboard</a> — this is the older diagnostic view; use each result's date.</p>
<div style="background:#fff7e0;border:2px solid #d9a514;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:16px">
<a href="lifetime.html" style="font-weight:700">&#128200; Training history &mdash; dated observations on the current ancestry</a></div>
<div class="sub">Updated {now}. This page refreshes itself every 30 seconds. {len(running)} run(s) going right now.</div>
<div class="legend"><b>How to read this.</b> Each box is one training run. The chip strip at the top of a box holds
the numbers that decide anything: whether the running code matches the current source, whether the executed action
reached the model, which way causal prediction error moved over report windows, whether the body follows commands,
and the five-control causal test. Every value on this page is computed by the dashboard script from the run's own
files at render time; free text typed by an operator is labelled as such.
<b>Scientific status.</b> Contacts, rotation angles, body-discrimination ratios, and gaze heuristics are
observational proxies. They do not by themselves establish purposeful behaviour, situational awareness, a learned
self-model, or valid imagined rollouts; consult the causal-control and mechanical-milestone results.
"Guessing the next instant" = prediction error of the next observation given the state and the executed action;
<b>lower is better</b>, and comparisons use first-window vs last-window means, never single records.
"Simulator physics frames per second" = 10 ms physics frames.
The model observes and learns only once per action-repeat interval (normally every 5 physics frames = 50 ms),
so physics FPS is not model-update FPS.
"Curiosity drive" = a deterministic function of recent prediction error, nonzero only inside the run's configured
band. "Snapshot" = all model weights saved to disk.
{sense_names_legend()}</div>
"""]
    for n in (extra_notes or []):
        bad = stale_note(n, infos)
        warn = ("" if not bad else
                f' <b style="color:#8f1d12">STALE: names {", ".join(bad)}, which is not running now</b>')
        parts.append('<div class="legend" style="border-left:6px solid %s">'
                     '<b>Operator note</b> (free text typed at launch, not a measurement): %s%s</div>'
                     % ("#b42318" if bad else "#d9e0ea", html.escape(n), warn))
    meta = read_jsonl(os.path.join(SAVED, "meta_overseer_log.jsonl"), "meta_overseer")
    if meta:
        m = meta[-1]
        aud = m.get("audit", {})
        tr = m.get("goal_trend", {})
        bits = [f"<b>Saved overseer audit</b> (last pass {time.strftime('%Y-%m-%d %H:%M', time.localtime(m['timestamp']))}; not a current training status):",
                f"incidents where the overseer acted beforehand: {aud.get('incidents_with_prior_action', 0)}; missed: {aud.get('incidents_missed', 0)}"]
        if tr:
            # This line used to render the discrimination ratio as an awareness claim,
            # assembled at RUNTIME from an f-string - so a test grepping this source for
            # the rendered sentence passed while the page still printed it. The check
            # that catches it has to read the OUTPUT. Ratio = between-body spread over
            # within-body variance; it rises when a body moves LESS, and the runs it
            # compares differ in body count, scene, checkpoint and duration.
            words = ", ".join(f"{TREND_LABEL.get(k, k)} is {TREND_WORD.get(v, v)}"
                              for k, v in tr.items())
            bits.append("trend across UNMATCHED runs (direction only, not a cause): " + words)
        # free text written by the meta-overseer: shown only when it passes DESIGN.md section 11
        held = 0
        for label, rows_ in (("threshold retuned: ", m.get("directive_changes") or []), ("gap: ", m.get("gaps") or [])):
            for c in rows_[:3]:
                text = clean(str(c))
                if text is None:
                    held += 1
                else:
                    bits.append(label + text)
        if held:
            bits.append(f"{held} more line{'s' if held > 1 else ''} of the audit print scores in a form this page does "
                        f"not show; meta_overseer_log.jsonl keeps them")
        parts.append('<div class="legend" style="border-left:6px solid #7a3fb0">' + "<br>".join(html.escape(b) if not b.startswith("<b>") else b for b in bits) + "</div>")
    # the plan-gap clock (operator 2026-09-06) is an archived analysis of an earlier lineage: its
    # distances are token distances measured on a 128-wide model and its sentences predate DESIGN.md
    # section 11, so the page names the file instead of reprinting its numbers
    gp = load(os.path.join(SAVED, "gap_eta.json"))
    if gp and gp.get("rung_text"):
        parts.append('<div class="legend"><b>Archived planning analysis</b> (file saved '
                     + saved_date(os.path.join(SAVED, 'gap_eta.json'))
                     + '): not current training status, and its rung number is not a percentage of project '
                       'completion. Its distances were measured on an earlier, 128-wide lineage; the numbers stay in '
                       '<a href="gap_eta.json">gap_eta.json</a>.</div>')
    # then-vs-now: the same world lived by the day-one brain and the current brain,
    # side by side - the panel that answers "is it still just flailing?" with eyes
    tvn_p = os.path.join(SAVED, "then_vs_now", "latest.json")
    tvn = load(tvn_p)
    if tvn and tvn.get("now"):
        import math as _m
        d1, nw = tvn["day1"], tvn["now"]
        parts.append(
            '<div class="legend" style="border-left:6px solid #2e6e63"><p>Recorded comparison — file saved '
            + saved_date(tvn_p) + '. This does not evaluate the current running weights.</p><b>Same world, '
            f'{tvn.get("seconds", 30):.0f} seconds, two brains (then vs now)</b><br>'
            f'day-one brain: {d1.get("strikes_hand", 0)} hand strikes on the mobile, '
            f'{d1.get("fovea_on_hand_s", 0):.1f} s looking at its own hand, commands turning '
            f'{_m.degrees(d1.get("mean_order_rad", 0)):.0f} degrees (thrash)<br>'
            f'comparison brain: {nw.get("strikes_hand", 0)} hand strikes, '
            f'{nw.get("fovea_on_hand_s", 0):.1f} s on its own hand, commands turning '
            f'{_m.degrees(nw.get("mean_order_rad", 0)):.0f} degrees - '
            f'{fixed(nw.get("strikes_hand", 0) / max(d1.get("strikes_hand", 1), 1), 2)} times the strikes '
            f'with {fixed(d1.get("mean_order_rad", 1) / max(nw.get("mean_order_rad", 1e-9), 1e-9), 2)} times '
            f'less movement. Watch it: {html.escape(str(tvn.get("video", "")))}</div>')
    # full panels only for LIVE runs and the single most recent finished one; everything
    # older collapses to a one-line history table (the page is a status screen, not an
    # archive - the per-run folders and the report hold the history)
    if any(i.get("role") for i in infos):
        # scoped build: the active run and the newest other run get full panels; the
        # active run's lineage gets one line per segment
        shown = [i for i in infos if i.get("role") in ("live", "other")]
        older = sorted([i for i in infos if i.get("role") == "ancestor"],
                       key=lambda d: d.get("mtime") or 0.0, reverse=True)
        older_title = ("Earlier segments of the active run's lineage (%d)" % len(older))
    else:
        live = [i for i in infos if i.get("status") == "running"]
        done = sorted([i for i in infos if i.get("status") != "running"],
                      key=lambda d: d.get("mtime") or 0.0, reverse=True)
        shown = live + done[:1]
        older = done[1:]
        older_title = "Finished runs (%d)" % len(older)
    parts.append('<div class="sub">This page describes the supervisor\'s active run, the runs it resumed from, '
                 'and the newest run outside that chain. Other run folders are not read (their files are '
                 'unchanged); the development dashboard\'s Over-time tab charts the whole lineage.</div>')
    if older:
        rows_h = "".join(
            f"<tr><td>{html.escape(i2['name'])}</td><td>{html.escape(str(i2.get('status','?')))}</td>"
            f"<td>{'{:,}'.format(i2['kept_step']) if i2.get('kept_step') is not None else i2.get('steps','')}</td>"
            f"<td>{i2.get('phase','')}</td>"
            f"<td>{i2.get('last_snapshot','')}</td></tr>" for i2 in older)
        parts.append('<details class="legend"><summary><b>%s</b> - one line each; '
                     'full data in each run folder</summary>'
                     '<table><tr><th>run</th><th>status</th><th>lineage step kept (its child resumed there)</th>'
                     '<th>phase</th><th>last snapshot</th></tr>%s</table></details>' % (older_title, rows_h))
    for i in shown:
        st = i.get("status", "?")
        cls = "run" if st == "running" else ("fin" if st == "finished" else "bad")
        tcls = "t-run" if st == "running" else ("t-fin" if st == "finished" else "t-bad")
        a2 = i.get("args") or {}
        rows = []

        def ref_row(label, value, reference="-"):
            """label / what we measured / what it should be."""
            rows.append((label, f"{value}  [reference: {reference}]"
                         if reference != "-" else str(value)))

        # ---- time: two different clocks, shown as two different numbers
        status_bits = [i["stop_reason"] if i.get("stop_reason") else st]
        if i.get("motor_watch"):
            status_bits.append(i["motor_watch"])
        if i.get("proc_min") is not None:
            status_bits.append(f"trainer's own clock {i['proc_min']:.0f} min"
                               + (f" of {i['minutes_planned']:.0f} planned" if i.get("minutes_planned") else ""))
        if i.get("folder_min") is not None:
            status_bits.append(f"run folder created {i['folder_min']:.0f} min ago")
            if i.get("proc_min") is not None and i["folder_min"] - i["proc_min"] > 15:
                status_bits.append("folder predates this process - reused across launches")
        rows.append(("Status", "; ".join(status_bits)))
        rows.append(("How many babies, and where", f"{i.get('bodies')} / {i.get('scene')}"))
        # ---- steps: the lineage counter is the brain's whole life, not this run
        rows.append(("Learning steps (checkpoint lineage)",
                     f"{i.get('steps')} - carried over from every run this brain has lived through"))
        if i.get("new_steps") is not None:
            rs = i.get("resume_step")
            rows.append(("New steps in this run",
                         f"{i['new_steps']}" + (f" (resumed at step {rs})" if rs else " (fresh start)")))
        # ---- speed: model rate and physics rate are different numbers
        if i.get("model_sps"):
            n = i.get("bodies") or 1
            sps = i["model_sps"]; ar = i.get("ar") or 1; dt = i.get("dt") or 0.01
            phys = sps * ar
            rt_one = phys * dt         # seconds of baby life per real second, one body
            rt_all = rt_one * n
            rows.append(("Model transitions per second",
                         f"{sps:.2f} per body; {sps * n:.1f} observations/s across {n} bod{'ies' if n != 1 else 'y'} (one batch)"))
            rows.append(("Simulator physics frames per second",
                         f"{phys:.1f} per body (action repeat {ar}: the model observes once per {ar} frames)"))
            rows.append(("Speed vs real time",
                         f"each body at {rt_one*100:.0f}% of real time; all {n} together {rt_all*100:.0f}% "
                         f"({rt_all:.2f} s of baby life per wall-clock second)"))
        if i.get("phase"):
            rows.append(("Age of the body", f"{i['phase']}" + (f", MIMo age {i['age']:g} months" if i.get("age") is not None else "")))
        # ---- the causal lesson first: it is the number the project turns on
        k = i.get("win_k")
        nw = i.get("next_win") or {}
        if nw:
            for m in ("vision", "tactile", "proprioception"):
                if m in nw:
                    a, b = nw[m]
                    rows.append((f"Guessing the next instant of {SENSE_NAMES.get(m, m)}",
                                 fmt_change(a, b, k)))
            agg_a = sum(v[0] for v in nw.values()) / len(nw)
            agg_b = sum(v[1] for v in nw.values()) / len(nw)
            rows.append(("Guessing the next instant - all real senses",
                         fmt_change(agg_a, agg_b, k) + ". An average hides senses moving in "
                         "opposite directions - read the three rows above it."))
        elif i.get("recon_win"):
            rows.append(("Guessing the next instant",
                         "no causal next-state data in this run's records (next-frame lesson produced none)"))
        # RECONSTRUCTION. A percentage is a trend; an inversion is a fault, and the page
        # must not format the second like the first. H30 printed "+15889%" beside rows
        # that looked otherwise ordinary while its reconstruction heads were destroyed.
        for m in ("vision", "tactile", "proprioception"):
            rw = (i.get("recon_win") or {}).get(m)
            if not rw:
                continue
            a, b = rw
            hz = (i.get("recon_health") or {}).get(m) or {}
            nm = SENSE_NAMES.get(m, m)
            res = bool(hz.get("residual"))
            base = hz.get("baseline") or float("nan")
            # the errors themselves are encoded distances with no lived anchor: only their ratios
            # and changes are printed (DESIGN.md section 11)
            if hz.get("inverted") and res:
                lbl = f"FAULT - re-describing {nm} is broken"
                jump = b / max(base, 1e-9)
                rows.append((lbl,
                             f"its error is {fixed(jump, 2)} times its own value at the start of this window. "
                             f"A jump that size against its own history means the decoders are damaged. This is "
                             f"a fault, not a trend."))
                dyn_docs[lbl] = dict(
                    t="The reconstruction decoders are damaged",
                    w=(f"The decoders' error on re-describing {nm} became {fixed(jump, 2)} times its own "
                       f"start-of-window value within the visible window, on a run where nothing else restarted. "
                       f"The H30 damage signature had this shape (a 128-times jump)."),
                    h=("Windowed decoder reconstruction error (recon_dec), last window against "
                       "first. This run trains with --residual-prediction on, so next-state error "
                       "is NOT the yardstick: the heads predict the change and get the present "
                       "for free."),
                    f=("A deliberate change to the trunk or to the decoders' weight schedule also "
                       "jumps this number; check the run's flags before treating it as damage."))
            elif hz.get("inverted"):
                lbl = f"FAULT - re-describing {nm} is broken"
                rows.append((lbl,
                             f"its error is {fixed(hz['ratio'], 2)} times its own next-state error. Describing what "
                             f"it is looking at cannot be harder than predicting what comes next, so the "
                             f"reconstruction head is damaged. This is a fault, not a trend."))
                dyn_docs[lbl] = dict(
                    t="The reconstruction decoders are damaged",
                    w=(f"Re-describing {nm} - the state the model is already looking at - costs "
                       f"{fixed(hz['ratio'], 2)} times what predicting the NEXT state costs. "
                       f"Describing the present cannot be harder than predicting the future in a "
                       f"working model: the present is right there in the input."),
                    h=("Windowed decoder reconstruction error (recon_dec) against windowed "
                       "next-state error for the same sense, both over the last reports."),
                    f=("On H30 the cause was imagination feeding its own guesses back into "
                       "itself. This comparison is only valid with residual prediction OFF - "
                       "with it on, next-state error collapses by construction and the "
                       "own-history rule applies instead."))
            else:
                if hz and res:
                    lo_r, hi_r = RESIDUAL_UNDAMAGED
                    frac = (f" It sits at {fixed(hz['ratio'], 3)} times its next-state error - above 1 time is "
                            f"expected with residual prediction on, and undamaged residual runs measured "
                            f"{fixed(lo_r, 3)} to {fixed(hi_r, 3)} times"
                            + (" (this is above that range)" if hz["ratio"] > hi_r else "")
                            + f"; it becomes a FAULT only on a jump past 4 times its own recent value.")
                elif hz:
                    frac = (f" It sits at {pct(hz['ratio'], 2)} of its own next-state error;"
                            f" above 100% this row becomes a FAULT.")
                else:
                    frac = ""
                lbl = f"Re-describing {nm} now (diagnostic, not causal)"
                rows.append((lbl, fmt_change(a, b, k) + "." + frac))
                if hz:
                    dyn_docs[lbl] = dict(
                        t="Reconstruction error - a diagnostic, NOT evidence about actions",
                        w=(f"The decoders re-describe the {nm} state the model is already looking "
                           f"at: their error {change_words(a, b)} over the window and sits at "
                           + (f"{fixed(hz['ratio'], 3)} times" if res else pct(hz['ratio'], 2) + " of")
                           + " the next-state error."
                           + (" Above 1 time is expected here: this run predicts the CHANGE "
                              "(--residual-prediction on), so next-state error gets the present "
                              "for free and sits low by construction. The decoders also train at "
                              "their floor weight (2 in 100 of the loss) - a smoke detector, not a "
                              "competitive skill." if res else
                              " Undamaged runs measured 26% to 99% of their next-state error (H22, H28, "
                              "H31); above 100% the decoders are broken and this row becomes a FAULT.")),
                        h=("recon_dec per sense over first/last report windows; ratio against "
                           "windowed next-state error."),
                        f=("A still body drives it low, which reads as progress and is not. It "
                           "says nothing about whether the model learned what its actions do."))
        rmw = i.get("recon_mean_win")
        if rmw and not i.get("recon_win"):
            a, b = rmw
            hm = i.get("recon_health_mean") or {}
            if hm.get("inverted"):
                if hm.get("residual"):
                    rows.append(("FAULT - re-describing (all senses) is broken",
                                 f"decoder error {fixed(b / max(hm.get('baseline') or 1e-9, 1e-9), 2)} times its own "
                                 f"start-of-window value. This log predates per-sense recording."))
                else:
                    rows.append(("FAULT - re-describing (all senses) is broken",
                                 f"decoder error {fixed(hm['ratio'], 2)} times the mean next-state error. Describing "
                                 f"what it is looking at cannot be harder than predicting what comes next. This log "
                                 f"predates per-sense recording."))
            else:
                rows.append(("Re-describing, all senses (decoders' mean)",
                             fmt_change(a, b, k) + (f" It sits at {pct(hm['ratio'], 2)} of the mean next-state error."
                                                    if hm else "") + " This log predates per-sense recording."))
        # PREDICTED CHANGE. Not an error. The heads' guess of the next instant against the
        # present: near zero when the model predicts that nothing changes, rising when it
        # predicts movement. It fed the fault rule until 2026-08-27.
        cw = i.get("change_win") or {}
        for m in ("vision", "tactile", "proprioception"):
            if m in cw:
                a, b = cw[m]
                rows.append((f"Predicted change of {SENSE_NAMES.get(m, m)}",
                             f"{change_words(a, b)} from the first {k} reports to the last {k}. How far the model's "
                             f"guess of the NEXT instant sits from the present. Not an error: small means it "
                             f"predicts persistence, larger means it predicts movement."))
        # THE EARLY WARNING. audio and intermind carry no loss weight, so their heads
        # never receive gradient. Their error therefore tracks the shared trunk's output
        # scale rather than any learning - which makes them the first place drift shows.
        dr = i.get("drift") or {}
        if dr:
            bits = ", ".join(f"{SENSE_NAMES.get(m, m)} " + ("the same as" if fixed(d['ratio'], 2) == "1" else
                                                            f"{fixed(d['ratio'], 2)} times") + " its first value"
                             for m, d in dr.items())
            worst = max(d["ratio"] for d in dr.values())
            rows.append(("Untrained-channel drift (earliest warning sign)",
                         f"{bits}. These two senses are fake on MIMo and carry no loss "
                         f"weight, so nothing corrects them - their error tracks the model's "
                         f"own output scale. "
                         + ("Over 3 times its first value: the shared trunk is drifting."
                            if worst > 3 else "Under 3 times its first value: no trunk drift by this rule.")))

        # ---- command following: direction, magnitude and the mechanical probe are
        # three different measurements; the old row blended them into one story
        hold_ms = (a2.get("action_repeat") or 1) * (a2.get("motor_hold") or 1) * 10
        if i.get("follow_slow") is not None:
            from mimo_bridge.tools.lineage_series import FOLLOW
            counted = (f"; counted on ckpt_step_344400 ({FOLLOW.get('measured')}): {fixed(FOLLOW['live'], 2)} of every "
                       f"100 body-part commands ended closer to their commanded turn, against {fixed(FOLLOW['chance'], 2)} "
                       f"by chance and {fixed(FOLLOW['reachable'], 2)} for a posture the body really reached on "
                       f"physics alone" if FOLLOW.get("live") is not None else "")
            rows.append(("Does the body move the way it was told (direction)",
                         f"{fixed(100 * i['follow_slow'], 2)}% of the way from 'unrelated' to 'exactly as told' over "
                         f"a command's whole {hold_ms} ms life (below 0% = against the command on average; an averaged "
                         f"direction match, not yet measured as a count per window){counted}"))
        if i.get("ratio") is not None:
            rows.append(("How much of the asked turn happens (size only)",
                         f"{i['ratio']:.0%} of the asked turn per 10 ms slice - magnitude only, ignores direction"))
        pf = i.get("preflight")
        if pf and pf.get("limb_cos") is not None:
            rows.append(("Mechanical probe at launch (plumbing, not performance)",
                         f"{fixed(100 * pf['limb_cos'], 2)}% of the way from 'unrelated' to 'exactly as told' on a "
                         f"quiet body at launch - pathway health, not the trained controller's live performance "
                         f"({pf['n_ok']}/{pf['n']} preflight checks passed)"))
        # ---- behaviour telemetry, labelled as the proxies they are
        if i.get("ball_dist") is not None:
            arrow = ""
            if i.get("ball_dist_first"):
                arrow = " (was %.0f cm at the start)" % (i["ball_dist_first"] * 100)
            rows.append(("Reach task: hand to ball", f"{i['ball_dist']*100:.0f} cm{arrow}; ball touched in {100*(i.get('ball_rate') or 0):.0f}% of recent checks"))
        if i.get("ball_hand") is not None:
            rows.append(("Hand touching the ball (recent)", f"{100*i['ball_hand']:.1f}% of the time. NOTE: this counts CONTACT, not reaching - a hand that flops onto the ball counts"))
        if i.get("gaze_on") is not None:
            earn = (f"Earned-look heuristic (recent window): {100*i['gaze_earn']:.1f}%. "
                    if i.get("gaze_earn") is not None else "")
            rows.append(("Ball near the gaze (cause unknown)",
                         f"{earn}Ball simply near the gaze (including every time the grown-up "
                         f"or a reset PLACED it there): {100*i['gaze_on']:.0f}%. Average miss "
                         f"{i.get('gaze_deg', 0):.0f} degrees. The causal source of an alignment - "
                         f"saccade, reflex, head motion, ball drift, or model gaze - is not "
                         f"separable from these logs"))
        if i.get("plan_len") is not None:
            rows.append(("How far ahead it imagines before acting", f"{i['plan_len']:.1f} steps ({i['plan_len']*0.05:.2f} seconds)"))
        if i.get("movement"):
            mv2 = i["movement"]
            rows.append(("How much it is moving",
                         f"body-local speed {mv2.get('local_speed', 0):.1f} cm/s; stillness debt "
                         f"{fixed(100 * (mv2.get('stillness_debt') or 0), 2)}% of its 0-to-1 range; movement wellness "
                         f"{fixed(100 * (mv2.get('movement_wellness') or 0), 2)}% of the way from its lowest to its highest"))
            rows.append(("Co-movement channel (body map)", i.get("selfc", "no data")))
        _st = i.get("sit_test")
        if _st:
            verdict = clean(str(_st.get("verdict", "?")))
            rows.append(("Situation dependence (controlled test)",
                         (verdict if verdict is not None else
                          "the saved verdict prints scores in a form this page does not show; read situation_test.json")
                         + f" [{_st.get('when', '?')}]"))
        # ---- action knowledge: what the model knows about its own orders (2026-09-02)
        if i.get("rank1") is not None:
            nd = int(i.get("n_decoys") or 0)
            ch = (1.0 / (1 + nd)) if nd else None
            trend = (f"; from the first reports to the last it moved from {per(i['rank1_win'][0], 'lineups')} to "
                     f"{per(i['rank1_win'][1], 'lineups')}" if i.get("rank1_win") else "")
            ev = (f" ({i['tenure_events']:.0f} command endings in the last window)"
                  if i.get("tenure_events") is not None else "")
            body = (f"picks its own command's effect out of a lineup of {nd + 1} in {per(i['rank1'], 'lineups')}, "
                    f"against {per(ch, 'lineups')} by chance" if ch
                    else f"picks its own command's effect in {per(i['rank1'], 'lineups')} against {nd} decoys")
            rows.append(("Command-to-outcome matching (rank-1)", body + trend + ev))
        if i.get("vis_gate") is not None:
            rows.append(("Hand-regard gate on vision's matching share",
                         f"open on {100 * i['vis_gate']:.1f}% of tenure events - vision's "
                         f"slice joins the matching only when the hand was in view at "
                         f"both ends of the command"))
        if i.get("tenure_pe_pose") is not None:
            _g = i.get("tenure_gap_pose")
            margin = (100 * _g / max(i['tenure_pe_pose'], 1e-12)) if _g is not None else None
            rows.append(("Tenure pose lesson (250 ms command horizon)",
                         (f"the with-command prediction beats do-nothing by {'+' if margin > 0 else ''}{fixed(margin, 2)}% "
                          f"of do-nothing's error (the hinge demands +4%)" if margin is not None else
                          "the do-nothing comparison is not in this record")
                         + "; its error level is an encoded distance, not yet measured against a lived reference"))
        _ag = i.get("agency_g") or {}
        if _ag.get("pay") is not None:
            aw = _n(a2.get("agency_weight"))
            bits = [(f"payment {fixed(100 * _ag['pay'] / aw, 2)}% of its weight's most" if aw else
                     "payment not yet measured against its maximum (no --agency-weight in args.json)")]
            prior0 = _n(a2.get("agency_prior"))
            if _ag.get("prior_w") is not None:
                bits.append(f"hand-watching bootstrap at {fixed(100 * _ag['prior_w'] / prior0, 2)}% of its starting weight"
                            if prior0 else "hand-watching bootstrap weight not measured against its start")
            if _ag.get("motion") is not None:
                bits.append(f"hand-motion gauge {fixed(100 * _ag['motion'], 2)}% of its 0-to-1 range")
            if _ag.get("fovea") is not None:
                bits.append(f"hand in fovea {100 * _ag['fovea']:.0f}%")
            if _ag.get("vis_adv") is not None and _ag.get("touch_adv") is not None:
                va, ta = float(_ag["vis_adv"]), float(_ag["touch_adv"])
                bits.append(f"vision's evidence of its own effect is {fixed(va / ta, 2)} times tactile's" if ta > 0 else
                            "tactile's evidence of its own effect is at or below zero, so vision's cannot be compared with it")
            if _ag.get("fov_missing"):
                bits.append(f"WARNING: fovea flag absent on {_ag['fov_missing']:.0f} bodies")
            rows.append(("Agency drive payments", "; ".join(bits)))
        if i.get("mobile_speed") is not None or i.get("mobile_hit") is not None:
            rows.append(("Mobile above the chest",
                         f"struck in {100 * (i.get('mobile_hit') or 0):.1f}% of recent steps "
                         f"({100 * (i.get('mobile_hand') or 0):.1f}% by a hand); bob moving "
                         f"{i.get('mobile_speed') or 0:.1f} cm/s on average"))
        if i.get("ring_snaps") is not None:
            rows.append(("Corner-of-the-eye ring (150 degree motion field)",
                         f"{i['ring_snaps']:.0f} ring-triggered snaps so far (subset of the "
                         f"reflex's {i.get('snaps') or 0:.0f}); motion outside the 60-degree "
                         f"wide cone is visible for the first time"))
        if i.get("squeaker_hits") is not None:
            rows.append(("Kick squeaker at the feet",
                         f"pressed {i['squeaker_hits']:.0f} times "
                         f"({i.get('squeaker_foot') or 0:.0f} by foot or lower leg); it "
                         f"squashes, flashes and squeaks only when the body strikes it"))
        if i.get("dragonfly_deg") is not None:
            rows.append(("Dragonfly and bee (world-caused movers)",
                         f"dragonfly {i['dragonfly_deg']:.0f} degrees off gaze, "
                         f"{i.get('dragonfly_darts') or 0:.0f} darts (hovering "
                         f"{100 * (i.get('dragonfly_hover') or 0):.0f}% - the hover is the "
                         f"pre-dart tell); bee {i.get('bee_deg') or 0:.0f} degrees off gaze, "
                         f"buzzing in stereo"))
        if i.get("antic_ok") is not None:
            rows.append(("Anticipation target supply",
                         f"a real self-caused change existed to look at in "
                         f"{100 * (i.get('antic_ok') or 0):.0f}% of recent steps "
                         f"(moving {i.get('antic_speed') or 0:.0f} cm/s) - the "
                         f"reality-graded gaze lesson feeds on these"))
        if i.get("squeaker_view") is not None:
            # operator caught this row lying three ways (2026-09-09): the value is
            # DEGREES (env emits np.degrees) but was printed as radians;
            # squeaker_seen is a WINDOW FRACTION (bools window-meaned) but any
            # nonzero was printed as currently-in-view; and the row claimed the
            # look was deliberate from a 30-degree geometry threshold - the
            # banned angle-threshold-as-intent class. Exact old phrases live in
            # test_dashboard_metric_semantics as banned literals.
            _hl = i.get("headlook") or 0
            _sf = float(i.get("squeaker_seen") or 0)
            rows.append(("Head-lift-and-look milestone",
                         f"the squeaker averaged {i['squeaker_view']:.0f} degrees off the "
                         f"gaze axis this window, inside the 30-degree view cone in "
                         f"{100 * _sf:.0f}% of it; head tilt {fixed(i.get('head_tilt') or 0, 2)} "
                         f"degrees; {_hl:.0f} view-cone entries so far (counted when the "
                         f"cone is entered from beyond 45 degrees - geometry only: whether "
                         f"any entry was a deliberate look is unmeasured)"))
        _wm = i.get("world_model") or {}
        if _wm.get("A_see_own_action"):
            _a3 = _wm["A_see_own_action"]; _b3 = _wm.get("B_touch_anticipation") or {}
            rows.append(("Does moving its hand tell it what it will see?",
                         f"knowing its own command improves its guess of the next view by "
                         f"{_a3.get('own_order_advantage_pct') or 0:.1f}% over a stranger's "
                         f"command (0% = its hand teaches its eyes nothing; goal line unmeasured; "
                         f"{_a3.get('n')} model steps)"))
            if _b3.get("anticipation_ratio") is not None:
                _r = float(_b3["anticipation_ratio"]); _cl = float(_b3.get("actual_ratio_ceiling") or 1.0)
                _pct = 100.0 * (_r - 1.0) / max(_cl - 1.0, 1e-9)
                rows.append(("Does it feel a touch coming?",
                             f"it is {_pct:.0f}% of the way from a brain that never sees "
                             f"touch coming to one that always does (score {fixed(_r, 3)} times a "
                             f"never-knowing brain, where always-knowing scores {fixed(_cl, 3)} times)"))
            _c3 = _wm.get("C_look_back") or {}
            if _c3:
                bits = [f"after {v['seconds_away']:.1f} s away it "
                        + ("expects some of what was really back there" if (v.get("look_back_advantage") or 0) > 0
                           else "assumes the world is whatever it last saw")
                        for v in _c3.values()]
                rows.append(("Looking away and back: does it remember the room?",
                             "; ".join(bits) + " (the saved advantage is not yet measured as a count; "
                             "some of it comes from the eye already sweeping that way, so the trend across "
                             "away-times is what to trust)"))
        _hs = i.get("hself") or {}
        if _hs.get("slots_filled") is not None:
            # P222: never sum -1 as a count ((-1 or 0) is -1); say unmeasured and how far
            _parts = []
            for _lvl, _name in (("l2", "minutes"), ("l3", "hours")):
                _s = _hs.get(f"{_lvl}_sig")
                if _s is None:
                    continue
                if _s >= 0:
                    _parts.append(f"{_s} carry a wellness link above a line set for 13.5 false counts "
                                  f"in 10,000 linkless tables at the {_name} level")
                else:
                    _left = _hs.get(f"{_lvl}_steps_to_first_evidence")
                    _parts.append(f"unmeasured at the {_name} level"
                                  + (f": {int(_left):,} more model steps until any slot could be counted"
                                     if _left is not None else ""))
            # the two gauges were first written as moments_with_a_row / best_row_cos (H84 segments 6, 7)
            _share = _hs.get("moments_with_a_slot", _hs.get("moments_with_a_row"))
            _near = _hs.get("best_slot_cos", _hs.get("best_row_cos"))
            rows.append(("Hierarchical self (observing only, feeds nothing)",
                         (f"{_hs['slots_filled']:,} slots in use hold {_hs['slots_distinct']:,} "
                          f"different patterns ({_hs['slots_twinned']:,} slots are copies of another "
                          f"slot; a slot stores what all the senses reported at one model step); "
                          + (f"{100 * _share:.0f} of 100 of today's model steps belong to a slot (0 = none: "
                             f"the table is about other model steps and no stored pattern is being "
                             f"refined, 100 = every model step belongs to a slot), and a model step sits "
                             f"{fixed(100 * _near, 2)}% of the way from 'nothing in common' to 'the same "
                             f"pattern' with its nearest slot (a cosine, not yet measured as a count); "
                             if _share is not None and _near is not None else "")
                          if _hs.get("slots_distinct") is not None and _hs.get("slots_twinned") is not None
                          else f"{_hs['slots_filled']} slots in use; ")
                         + "; ".join(_parts) + ". This is the groundwork for "
                         f"valuing postures by what they unlock - it changes no behavior until "
                         f"separately certified"))
        _es = i.get("entity_self") or {}
        if _es.get("names"):
            _nm, _lk = _es["names"], (_es.get("l2_link") or [])
            _pr = _es.get("presence_mean") or []
            _best = sorted(range(len(_nm)), key=lambda j: -abs(_lk[j] if j < len(_lk) else 0.0))[:3]
            _rd = _es.get("reader") or {}
            _counted = _es.get("l2_counted_names") or []
            rows.append((
                "Named things the self-model watches (%s level, observing only)" % "minutes",
                "%d named things - the caregiver, each toy, the mobile, the critters, the baby's "
                "own hand in view - each with its own presence signal read from the world. It is "
                "learning which of them go with %s rising and falling: %s. Strongest so far: %s. "
                "A link counts only after it beats 8 fake signals for %s windows of %s model steps; "
                "%s windows lived. The reader that would turn a counted link into a goal the "
                "planner aims at is %s"
                % (len(_nm), "the body's wellness" if _es.get("teacher") == "wellness"
                   else "the count of consequences the body caused",
                   ("counted so far: " + ", ".join(_counted)) if _counted
                   else "nothing counted yet, which is what -1 means",
                   "; ".join("%s present %.0f of 100 model steps, link %s%% of the way from 'no link' to 'always "
                             "together'" % (_nm[j], 100 * (_pr[j] if j < len(_pr) else 0.0),
                                            fixed(100 * (_lk[j] if j < len(_lk) else 0.0), 2)) for j in _best),
                   _es.get("windows_needed"), "{:,}".format(_es.get("l2_window_moments") or 0),
                   _es.get("l2_ticks"),
                   ("ON: each unit of link times presence adds %s%% of a wellness unit to the goal's value"
                    % fixed(100 * (_rd.get("applied_weight") or 0.0), 2)) if _rd.get("gate_open")
                   else "off - " + (clean(str(_rd.get("why") or "not built for this run"))
                                    or "its reason prints scores in a form this page does not show"))))
        _plan = plan_distance_row(i.get("intention_meta") or {},
                                  load(os.path.join(SAVED, "scale_anchors.json")) or {}, a2)
        if _plan:
            rows.append(_plan)
        _pl = i.get("plan_lic") or {}
        if _pl.get("win_emas") is not None:
            _w = _pl["win_emas"]
            rows.append(("Plan horizon license (earned lookahead)",
                         f"{_pl.get('licensed', 0)} of {len(_w)} depths licensed "
                         f"(wins in 100 tries against doing nothing: "
                         + ", ".join(f"{100 * x:.0f}" for x in _w)
                         + f"; a depth unlocks above 50 in 100) - mode {_pl.get('mode')}"))
        _tg = i.get("td_gate_g") or {}
        if _tg:
            rows.append(("Expectation gate (top-down blend into perception)",
                         ", ".join(f"{SENSE_NAMES.get(k, k)} {fixed(100 * v, 2)}%" for k, v in sorted(_tg.items()))
                         + " - the share of its own prediction the model blends into what behavior consumes "
                           "(0% = acting on raw perception; the gate allows up to 50%)"))
        # gain-of-function build (2026-09-22): the fields the record carries beyond the rows above
        rows.extend(gof_rows(i.get("rec"), a2))
        if i.get("box_rates"):
            b = i["box_rates"]
            hand = (f"; hand only: {100*i['box_hand']:.1f}%" if i.get("box_hand") is not None else "")
            rows.append(("Contact with the weighted boxes (any body part)",
                         f"any body part - light {100*b.get('light',0):.0f}%, medium {100*b.get('medium',0):.0f}%, "
                         f"heavy {100*b.get('heavy',0):.0f}%{hand}. A leg lying against a box counts"))
        if i.get("roll") is not None:
            best = i.get("roll_best") or i["roll"]
            ref_row("General chest rotation (legacy measurement)",
                    f"best so far {best:.0f} degrees from reset. Turning around the pen also raises this number; "
                    f"it cannot establish rolling over or intention. The new dashboard reports body-axis roll separately",
                    "-")
        if i.get("eyes"):
            eye_px, fov_px, fov_deg, care, reflex = i["eyes"]
            wide = eye_px / 60.0
            bits = [f"wide {eye_px}px over 60 degrees = {wide:.1f} px per degree, about 20/{30/(wide/2)*20:.0f}"]
            if fov_px:
                f = fov_px / max(fov_deg, 1e-9)
                bits.append(f"sharp centre {fov_px}px over {fov_deg:g} degrees = {f:.1f} px per degree, about 20/{30/(f/2)*20:.0f}")
            else:
                bits.append("no sharp centre")
            rows.append(("How sharp its eyesight is", "; ".join(bits)))
            rows.append(("Built-in help it was born with",
                         ("snap-to-motion reflex on" if reflex != "off" else "snap-to-motion reflex OFF")
                         + ("; caregiver returns the toy after %g s unseen" % care if care else "; no caregiver")))
        if i.get("baby_min"):
            rows.append(("How much experience this run has produced",
                         f"{i['baby_min']:.0f} minutes of body-time per body across "
                         f"{i.get('bodies')} bodies (wall clock {i.get('proc_min') or 0:.0f} min). "
                         f"NOT comparable to an infant's calendar: {i.get('bodies')} bodies run at once, "
                         f"model steps are re-studied from memory, and none of the biological clock - muscle "
                         f"growth, myelination, cortex maturing - is in our way. Judge progress "
                         f"per learning step, not against a baby's age"))
        # Judge B's real-state agreement says nothing about imagined rollouts, so the
        # row only appears when imagination is on and the number could matter
        imag_on = str(a2.get("imagination", "off")).lower() not in ("off", "0", "false", "none")
        if i.get("walign") is not None and imag_on:
            rows.append(("Judge-B agreement on recent real states",
                         f"off by {pct(i['walign'], 2)} of its 0-to-1 scale against the grounded score, on real "
                         f"states only - imagined rollouts are not validated by this number"))
        if i.get("fresh") is not None:
            w = "every step" if i["fresh"] > 0.99 else "SOME STEPS RE-READ THE PREVIOUS IMAGE"
            rows.append(("Is it seeing a new picture every model step?", f"{w} ({100*i['fresh']:.0f}% of model steps)"))
        if i.get("eye_pitch") is not None:
            p = i["eye_pitch"]
            face_up = (f"; launch preflight measured {pf['face_up_deg']:+.0f}"
                       if pf and pf.get("face_up_deg") is not None else "")
            rows.append(("Eye pitch", f"{p:+.0f} degrees from level (positive = above the horizon{face_up})"))
        if i.get("hand_view") is not None:
            ref_row("Can it see its own hands?",
                    f"a hand is in view {100*(i.get('hand_seen') or 0):.0f}% of the time; "
                    f"the nearest one is {i['hand_view']:.0f} degrees off to the side",
                    "-")
        # rates are per minute of BABY life, not wall clock
        baby_min = max(i.get("baby_min") or 0.0, 1e-9)
        if i.get("saccades") is not None and i.get("baby_min"):
            ref_row("Spontaneous eye movements (saccades)",
                    f"{i['saccades']/baby_min:.0f} per minute of the baby's own time "
                    f"({i['saccades']:.0f} in total)",
                    "design rate ~120/min (a biology-informed construction number); these "
                    "movements are the data for learning what looking does")
        if i.get("snaps") is not None and i.get("baby_min"):
            ref_row("Eye flicks toward movement",
                    f"{i['snaps']/baby_min:.0f} per minute of the baby's own time "
                    f"({i['snaps']:.0f} in total)",
                    "fires on peripheral MOTION, so a mostly-still crib keeps this low; "
                    "the spontaneous saccades above are the main mover")
        if i.get("crossed") is not None:
            rows.append(("Hand crossing of the geometric midline",
                         f"{100*i['crossed']:.1f}% of the time a hand is past the midline"))
        if i.get("pressing") is not None:
            kpa = f" (firmest hand contact {i['press_kpa']:.1f} kPa)" if i.get("press_kpa") is not None else ""
            rows.append(("Hand pressure above threshold",
                         f"{100*i['pressing']:.1f}% of the time above 2 kPa{kpa}"))
        if i.get("rescues") is not None:
            extra = f"; out of sight {i['unseen_s']:.1f} s right now" if i.get("unseen_s") is not None else ""
            rows.append(("Grown-up put the toy back in view", f"{i['rescues']:.0f} times{extra}"))
        if i.get("discrim") is not None:
            # a share of each body's own variation with its two ends named, never "0.09 times each" (a
            # multiplier the readout scanner cannot place; fix-readout-dashboards 2026-09-23)
            rows.append(("Between-body action signatures (uncalibrated)",
                         f"the spread between the bodies' average commands is {pct(i['discrim'], 2)} of each body's own "
                         f"variation over time (0% = every body's average command the same, 100% = the bodies differ "
                         f"as much as each one varies). Uncalibrated proxy: the ratio rises when a body "
                         f"simply moves less, and there is no matched-scenario baseline"))
        if i.get("curiosity") is not None:
            band = str(a2.get("curiosity_band") or "")
            band_txt = f"; nonzero only inside the error band set by --curiosity-band {band}" if band else ""
            rows.append(("Curiosity drive",
                         f"{fixed(100 * i['curiosity'], 2)}% of its maximum - a deterministic function of recent "
                         f"prediction error{band_txt}"))
        if i.get("contacts") is not None:
            rows.append(("Things touching its skin", f"{i['contacts']:.0f} spots"))
        tv = i.get("turnover")
        if tv and tv.get("born") is not None:
            def _pp(n):
                n = float(n or 0)
                return f"{fixed(n, 2)} patches" if n >= 1 or n == 0 else f"{fixed(100 * n, 2)} patches per 100 steps"
            act = (f"{_pp(i['active'])} in contact at a time" if i.get("active") is not None
                   else "the number in contact at a time")
            rows.append(("How fast the skin's contacts turn over",
                         f"about {_pp(tv['born'])} start touching and "
                         f"{_pp(tv['died'])} stop, every model step, against {act}. Most "
                         "contacts do not survive one step, and predicting them is most "
                         "of what tactile prediction is."))
        if i.get("nan"):
            rows.append(("Broken numbers (should always be 0)", str(i["nan"])))
        if i.get("ckpt_failures"):
            rows.append(("Saves that failed this segment (should always be 0)",
                         f'{i["ckpt_failures"]} - last error: {i.get("ckpt_last_error")}'))
        if i.get("peak_vram_mb") is not None:
            _card = _card_mib()
            rows.append(("Peak allocation on the card, last window",
                         f'{i["peak_vram_mb"]:,.0f} MiB allocated by the trainer'
                         + (f' of {_card:,} on the card' if _card else '')
                         + '; the eye renderers hold about 100 MiB more per body outside this count'))
        if i.get("last_snapshot"):
            rows.append(("Last time the brain was saved", f"{i['last_snapshot']}  ({i['n_snapshots']} saves so far)"))
        if i.get("error"):
            rows.append(("Error", i["error"]))
        # rows split by the design review in ROW_DOC: anything that scored below 8
        # for usefulness goes to a collapsed panel rather than being deleted.
        def _tr(k2, v):
            # a row with a live popup (numbers baked in at render time) wins over the
            # static fragment match
            key = k2 if k2 in dyn_docs else doc_for(k2)
            at = f' data-doc="{html.escape(key)}" role="button" tabindex="0"' if key else ""
            cls = ' class="ck"' if key else ""
            return (f"<tr{cls}{at}><th>{html.escape(k2)}</th>"
                    f"<td>{html.escape(str(v))}</td></tr>")
        prim = [(a, b) for a, b in rows
                if not (doc_for(a) and ROW_DOC[doc_for(a)].get("secondary"))]
        secd = [(a, b) for a, b in rows
                if doc_for(a) and ROW_DOC[doc_for(a)].get("secondary")]
        tbl = "<table>" + "".join(_tr(a, b) for a, b in prim) + "</table>"
        if secd:
            tbl += ('<details class="sect"><summary>Secondary telemetry (%d) - measured and kept, '
                    'but each is an exposure or endowed-mechanism count rather than evidence '
                    'about what the model learned</summary><table>%s</table></details>'
                    % (len(secd), "".join(_tr(a, b) for a, b in secd)))
        # ---- provenance: which code the trainer actually loaded
        pv = i.get("provenance")
        if pv:
            match_txt = ("MATCHES the current source HEAD" if pv.get("match") else
                         f"does NOT match the current source HEAD {pv.get('head')}" if pv.get("head") else
                         "current source HEAD unknown (git unavailable)")
            prov_rows = [("code commit the trainer loaded", f"{pv.get('code_commit')} - {match_txt}"),
                         ("resumed from", pv.get("resumed_from") or "fresh start"),
                         ("newest checkpoint step", pv.get("lineage_step"))]
            tbl += ('<details class="sect"><summary>Provenance - which code this run actually ran</summary><table>'
                    + "".join(f"<tr><th>{html.escape(str(k2))}</th><td>{html.escape(str(v))}</td></tr>" for k2, v in prov_rows)
                    + "</table></details>")
        # ---- gates: every switch and setting, with its state
        if i.get("gates"):
            switches, settings = i["gates"]
            g_rows = "".join(f"<tr><th>{html.escape(n)}</th><td>{html.escape(s)}</td>"
                             f"<td class=\"muted\">{html.escape(d)}</td></tr>" for n, s, d in switches)
            s_rows = "".join(f"<tr><th>{html.escape(n)}</th><td><code>{html.escape(s)}</code></td>"
                             f"<td class=\"muted\">{html.escape(d)}</td></tr>" for n, s, d in settings)
            tbl += ('<details class="sect" open><summary>Gates - every switch this run was launched with</summary>'
                    '<table><tr><th>gate</th><th>state</th><th>what it does when on</th></tr>' + g_rows
                    + '</table><table><tr><th>setting</th><th>value</th><th>meaning</th></tr>' + s_rows
                    + "</table></details>")
        img = f'<img src="{i["spark"]}">' if i.get("spark") else '<div class="muted">no curve yet</div>'
        # The two clips are the SAME recording: both written from one session at fps 10,
        # every 10th env step, and measured frame-identical - 300 frames, 30.0 s each.
        # Watching them apart means holding one in your head while you watch the other,
        # so they are driven as one clip: a single set of controls on the outside view,
        # with the eye view following it frame for frame.
        if i.get("video"):
            img += (f'<video class="vpair" data-pair="{html.escape(i["name"])}" '
                    f'src="{html.escape(i["video"])}" controls loop width="330" '
                    f'style="margin-top:6px"></video>')
        if i.get("video_eye"):
            img += (f'<video class="vpair" data-pair="{html.escape(i["name"])}" '
                    f'src="{html.escape(i["video_eye"])}" muted loop width="330" '
                    f'style="margin-top:4px"></video>')
        if i.get("video"):
            pair = i.get("video_eye")
            img += (f'<div class="muted">newest of {i.get("n_videos")} recording(s). '
                    + ("Outside view above, the same 30 s through the baby’s own left "
                       "eye below - one clip, they play together. " if pair else
                       "The baby from outside. ")
                    + 'Sim time at real speed.</div>')
        if i.get("views_img"):
            img += (f'<a href="{html.escape(i["views_img"])}"><img src="{html.escape(i["views_img"])}" style="margin-top:6px"></a>'
                    "<div class=\"muted\">whole scene from above and all four sides + the baby's eye (newest; click to open)</div>")
        gal = []
        for v in reversed(i.get("video_gallery") or []):
            gal.append(f'<a href="{html.escape(v)}">{html.escape(os.path.basename(v))}</a>')
        for v in reversed(i.get("views_gallery") or []):
            gal.append(f'<a href="{html.escape(v)}">{html.escape(os.path.basename(v))}</a>')
        if gal:
            img += '<div class="muted" style="margin-top:4px">all recordings: ' + " | ".join(gal) + "</div>"
        if i.get("overseer"):
            # the overseer's free text is shown only when it passes DESIGN.md section 11
            kept_notes = [t for t in (clean(x) for x in i["overseer"]) if t is not None]
            held_notes = len(i["overseer"]) - len(kept_notes)
            items = "".join(f"<li>{html.escape(x)}</li>" for x in kept_notes)
            if held_notes:
                items += (f"<li>{held_notes} more line{'s' if held_notes > 1 else ''} print scores in a form this "
                          f"page does not show; overseer_log.jsonl keeps them</li>")
            img += f'<div class="muted" style="margin-top:6px"><b>Overseer at {i.get("overseer_time")}</b><ul style="margin:2px 0 0 16px;padding:0">{items}</ul></div>'
        parts.append(f'<div class="card {cls}"><div><span class="name">{html.escape(i["name"])}</span>'
                     f'<span class="tag {tcls}">{st}</span>{chips_for(i)}{tbl}</div><div>{img}</div></div>')
    docs = {k: {"t": v["t"], "w": v["w"], "h": v["h"], "f": v["f"]}
            for k, v in list(DOC.items()) + list(ROW_DOC.items())}
    docs.update({k: {"t": v["t"], "w": v["w"], "h": v["h"], "f": v["f"]}
                 for k, v in dyn_docs.items()})
    parts.append(
        '<div id="ov"><div id="bx"><h3 id="bt"></h3>'
        '<div class="lb">What this is</div><p id="bw"></p>'
        '<div class="lb">How it is computed</div><p id="bh"></p>'
        '<div class="lb">What would fool it</div><p class="fool" id="bf"></p>'
        '<button onclick="cl()">Close</button></div></div>'
        '<script>var D=' + json.dumps(docs) + ';'
        'function op(k){var d=D[k];if(!d)return;'
        'document.getElementById("bt").textContent=d.t;'
        'document.getElementById("bw").textContent=d.w;'
        'document.getElementById("bh").textContent=d.h;'
        'document.getElementById("bf").textContent=d.f;'
        'document.getElementById("ov").className="on";}'
        'function cl(){document.getElementById("ov").className="";}'
        'document.addEventListener("click",function(e){'
        'var t=e.target.closest("[data-doc]");if(t){op(t.getAttribute("data-doc"));return;}'
        'if(e.target.id==="ov")cl();});'
        'document.addEventListener("keydown",function(e){'
        'if(e.key==="Escape")cl();'
        'if((e.key==="Enter"||e.key===" ")&&document.activeElement'
        '&&document.activeElement.hasAttribute("data-doc")){e.preventDefault();'
        'op(document.activeElement.getAttribute("data-doc"));}});</script>')
    # One clip, two cameras. The outside view carries the controls; the eye view has
    # none of its own and follows. Re-seeking only past a 0.15 s gap keeps it from
    # fighting the browser on every timeupdate, which stutters both.
    parts.append('<script>(function(){var g={};document.querySelectorAll("video.vpair").forEach(function(v){(g[v.dataset.pair]=g[v.dataset.pair]||[]).push(v);});Object.keys(g).forEach(function(k){var vs=g[k];if(vs.length<2)return;var lead=vs[0],rest=vs.slice(1);function each(f){rest.forEach(function(v){try{f(v);}catch(e){}});}lead.addEventListener("play",function(){each(function(v){v.currentTime=lead.currentTime;v.play();});});lead.addEventListener("pause",function(){each(function(v){v.pause();});});lead.addEventListener("seeking",function(){each(function(v){v.currentTime=lead.currentTime;});});lead.addEventListener("timeupdate",function(){each(function(v){if(Math.abs(v.currentTime-lead.currentTime)>0.15)v.currentTime=lead.currentTime;});});});})();</script>')
    parts.append("</body></html>")
    return "\n".join(parts)


run_dirs_cache: dict = {}


def run_dirs():
    dirs = []
    for base in (os.path.join(SAVED, "experiments"), os.path.join(SAVED, "runs")):
        if os.path.isdir(base):
            for d in sorted(glob.glob(os.path.join(base, "*"))):
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "args.json")):
                    dirs.append(d)
                    run_dirs_cache[os.path.basename(d)] = d
    return dirs


def _norm(path):
    return os.path.normcase(os.path.abspath(str(path)))


KEPT_STEP: dict = {}


def scoped_runs():
    """[(run_dir, role)] for the page: the supervisor's active run ('live'), the runs it
    resumed from ('ancestor', one line each), and the newest run outside that chain
    ('other'). Every other folder is left unread: describing all 277 folders cost 25 CPU-
    seconds per 30-second cycle on 2026-09-22 (measured in a separate process)."""
    dirs = run_dirs()
    state = load(os.path.join(SAVED, "training_continuation.json")) or {}
    chain = []
    KEPT_STEP.clear()
    if state.get("active_run"):
        chain = [_norm(state["active_run"])]
        try:
            from pathlib import Path
            from mimo_bridge.tools.development_dashboard import ancestry_bounds
            root = Path(SAVED).resolve().parents[1]
            rel = Path(state["active_run"]).resolve().relative_to(root)
            bounds = ancestry_bounds(root, dict(run=rel.as_posix(), step=None))
            chain = [_norm(p) for p in bounds] or chain
            # the step each ancestor's records are kept up to: its child resumed from there
            # (reviews/dashboards.md m4: seg13_reach printed its folder's last step, 232,544,
            # while the lineage kept it only to 227,547)
            KEPT_STEP.update({_norm(p): limit for p, limit in bounds.items() if limit is not None})
        except (ValueError, OSError, TypeError):
            pass
    known = {_norm(d): d for d in dirs}
    out = []
    if chain and chain[0] in known:
        out.append((known[chain[0]], "live"))
    for c in chain[1:]:
        if c in known:
            out.append((known[c], "ancestor"))
    # a deploy proof, a smoke test or a folder named with a leading underscore is not a run to
    # describe in full (m4: a 25-step deploy proof took the "newest other run" panel)
    others = [d for d in dirs if _norm(d) not in chain and not os.path.basename(d).startswith("_")
              and not re.search(r"(?i)(proof|smoke)", os.path.basename(d))]

    def newest(d):
        for name in ("replay_training_stats.jsonl", "args.json"):
            try:
                return os.path.getmtime(os.path.join(d, name))
            except OSError:
                continue
        return 0.0
    if others:
        out.append((max(others, key=newest), "other"))
    return out


def build(extra_notes=None):
    # the whole-life page (operator 2026-09-03: windowed per-run readouts hide
    # whether the model beats its own past) - refreshed by this same loop
    try:
        from mimo_bridge.tools import lifetime as _lt
        _lt.build(max_age_s=1800)
    except Exception as _e:
        extra_notes = list(extra_notes or []) + [f"lifetime page unavailable: {_e}"]
    infos = []
    for d, role in scoped_runs():
        try:
            info = describe_light(d, role) if role == "ancestor" else describe_run(d)
            info["role"] = role
            infos.append(info)
        except Exception as ex:
            infos.append({"name": os.path.basename(d), "status": "unreadable", "error": str(ex), "role": role})
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(tidy_html(render(infos, extra_notes or [])))
    os.replace(tmp, OUT)
    return OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--note", action="append", default=[], help="plain-text line shown at the top")
    a = ap.parse_args()
    from mimo_bridge.tools.training_continuation import OwnerLock
    owner=OwnerLock(os.path.join(SAVED,'legacy_dashboard.lock'))
    last_arch = 0.0
    from mimo_bridge.tools.selfreload import stamp as _stamp, exec_if_changed as _reexec
    import sys as _sys
    _t0 = _stamp()
    while True:
        _reexec(_t0, "mimo_bridge.tools.dashboard", _sys.argv[1:], log=print)
        p = build(a.note)
        if a.once:
            print("wrote", p)
            return
        # hourly history snapshot: the live page overwrites itself by design (it is a
        # status screen); these copies let you look back at what it showed
        if time.time() - last_arch >= 3600:
            try:
                import shutil
                hist = os.path.join(SAVED, "dashboard_history")
                os.makedirs(hist, exist_ok=True)
                shutil.copy2(OUT, os.path.join(hist, time.strftime("dashboard_%Y-%m-%d_%H%M.html")))
                old_copies = sorted(glob.glob(os.path.join(hist, "dashboard_*.html")))
                for f in old_copies[:-48]:
                    os.remove(f)
                last_arch = time.time()
            except OSError:
                pass
        time.sleep(a.every)


if __name__ == "__main__":
    main()
