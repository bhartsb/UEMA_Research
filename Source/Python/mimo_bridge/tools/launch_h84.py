# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The 2026-09-18 launch command, H84: fresh weights on the recommended changes.

One source of truth for the argument list, so the smoke run, the launch and the
receipt cannot drift apart. Built from H83's recorded command
(Saved/mimo_bridge/h83_launch_command_20260914.json) with these changes, each traced to
the gain-of-function tree of 2026-09-17 or the audit rows applied on 2026-09-18:

  fresh start            no --resume; no --self-choice-receipt; no --retire-acting-snapshot
                         (operator, September 16-17: "I don't care about the old learning")
  --self-choice off      the frozen shadow actor cost 368 ms of a 1,707 ms step and its
                         orders were never applied (tree S4)
  --contact-profile mechanical_v4     the fall ladder rises at every rung (tree W5)
  --core-blocks 4 --d-model 192 --vision-swin-layers 4 --vision-width 192
  --encoder-layers 2 --encoder-width 192 --replay-batch 24 --replay-every 2
                         14.4 M parameters (4.3x H83). The tree's proposed 32 M model did not
                         fit the 8 GiB card at 24, 16 or 8 bodies (see SIZE below)
  --intention-every 10 --motor-hold 5        every SECOND latch is a planner order; the other
                         is the controller's, carrying babble during bouts. Segment 1
                         (03:38-23:00, intention-every 5) executed planner orders only:
                         exec_source planner 1.0, babble 0 in all 19 hours - the model adds
                         babble only when no planner order exists (lem_model.py, the
                         exploration_noise block), which train_mimo.py:1196 had recorded on
                         2026-09-14 as the reason H83 ran 7. Under it the body's direction
                         match fell from 0.113 to 0.02 and hand strikes faded to 0. With 10
                         the schedule's arithmetic gives about 32 babble orders in 100
                         (0.5 x the 0.64 bout duty) against H83's 55 and segment 1's 0.
  --decoder-kp 6.0       best cell of tools/command_following.py under its declared
                         rule (tree X6); re-measured 2026-09-23 on the repaired gauge
                         (scale_anchors.json command_following): at half strength one
                         20-degree order to one body part turns 75% of the way from
                         unrelated to exactly the ordered way (95% interval 65% to 85%)
                         and 28 of 30 parts end nearer the ordered orientation. The rule
                         picks gain 6 again, but every cell's interval reaches 65%, so 30
                         trials per cell do not separate the gains. The 2026-09-18 table
                         that chose it read 72%, counting each chest trial as unrelated:
                         the gauge never scored the chest. H83 ran gain 2 (67% of the way
                         on physics, re-measured; 7% live)
  --consequence-w 0.5    consequence access in the goal-ring value; 0.5 per kind against
                         H83's 0.42 ring spread                               (tree W2)
  --planner-effort-per-body on               audit P112 (the 8x effort charge)
  --start-phase mature --age-binding fixed:24 --delay-binding off
                         the growth ramp is not part of this run: it lasted 12.5
                         body-minutes and rebuilt 24 bodies at each of five boundaries
                         (tree K8, K9); the sensory delay is refused (P002)
  --n-envs 8             the card sets the body count: about 100 MiB per body for the eye
                         renderers, and the live pass, imagination chain and re-study pass
                         share the rest; 8 bodies at the chosen size leave 1.4 GB of headroom
  checkpoints on C: inside the run directory; caregiver records on C: under
                         Saved/mimo_bridge/caregiver_records/<run>. D: (SanDisk Extreme
                         USB, HFS Plus) was measured 2026-09-18 02:51-02:56: 1-2 MB/s, 13-57 s per
                         32 MB chunk, dozens of retried IO operations and two port resets in the
                         system log, then a permission error after 160 MB; the first smoke's crash
                         snapshot failed on it with "The device is not ready". A 2.6 GB hourly
                         checkpoint cannot land there until the disk is repaired (its file system is
                         served by Apple's hfsplus.sys; reformatting is the operator's call).

Run:  python -m mimo_bridge.tools.launch_h84 --print             (the argument list)
      python -m mimo_bridge.tools.launch_h84 --smoke --n-envs 8    (180 steps, report every 10, replay from step ~132)
      python -m mimo_bridge.tools.launch_h84 --launch --n-envs 8   (writes the receipt and criteria, starts the trainer)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PY = os.path.join(REPO, ".venv_mimo", "Scripts", "python.exe")
RUN_NAME = "H84_fresh_drivers_20260918"


# The model size at launch. The tree's proposed 8 x 256 core / 12 x 256 vision / 4 x 256
# encoders was measured on 2026-09-18 03:53-03:00 in three smokes (24, 16 and 8 bodies):
# the live pass alone filled the card at every body count (7,9xx of 8,188 MiB from the
# first step, steps per second falling to 0.08 by step 30 at 8 bodies), before the replay
# pass at batch 48 had even started. The capacity benchmark's "everything fits at 6,058
# MiB" left out the imagination chain, the prediction heads and the worker renderers
# (~100 MiB per body, measured with tools/pool_step_cost.py). SIZE holds what the launch
# smoke selected; the smoke records in the launch folder carry each candidate's peak.
# Chosen 2026-09-18 03:20-03:24 (smoke_h84_8b_c4x192_v4x192_e2x192_r24e2, 180 steps): the
# allocator's own peak 3,963 MiB, the card at most 6,748 of 8,188 MiB with the eight
# renderers, 1.15 steps per second with the re-study pass active against 1.10 before it -
# no spill. The step before it (core 4 x 256 / vision 4 x 256 / encoders 2 x 256, 24.1 M)
# ran 1.03 steps per second until replay began and 0.22-0.31 after, at both replay 48
# every 4 and 24 every 2, with the card at 7,9xx MiB: the allocator thrashing at the
# ceiling. Replay 24 every 2 keeps the re-study lesson's samples per step equal to 48
# every 4 at half the peak. 14,369,945 parameters, 4.3x H83's 3.3 M.
SIZE = dict(core_blocks=4, d_model=192, vision_swin_layers=4, vision_width=192,
            encoder_layers=2, encoder_width=192, replay_batch=24, replay_every=2)


def command(n_envs: int, out: str, checkpoint_storage: str, caregiver_record_dir: str,
            max_steps: int = 0, report_every: int = 100, fill_ring_first: bool = True,
            size: dict = None, intention_every: int = 10, resume: str = "",
            explain_every_min: float = 60.0, entity_self: str = "dark",
            entity_self_teacher: str = "wellness", self_value_weight: float = 0.1,
            wellness_coherence: str = "extent") -> list:
    size = dict(SIZE, **(size or {}))
    args = [
        PY, "-u", "-m", "mimo_bridge.train_mimo",
        "--action-repeat", "5", "--action-tenure-alpha", "0.04", "--action-tenure-loss", "on",
        "--action-tenure-token-alpha", "0.005", "--action-tenure-token-reach", "on",
        "--action-tenure-token-weight", "0.25", "--action-tenure-weight", "0.5",
        "--age-binding", "fixed:24", "--agency-drive", "on", "--agency-prior", "0.05", "--agency-weight", "0.1",
        "--anticipation-gaze", "on", "--anticipation-gaze-w", "0.1", "--audio", "real", "--audio-freq-bands", "on",
        "--babble-bouts", "real", "--babble-noise", "0.05", "--babble-scale", "1.0", "--body-schema-updates", "on",
        "--caregiver-seconds", "15.0", "--checkpoint-every-min", "60.0", "--context-length", "128",
        "--corrupt-vision", "0.0", "--critters", "on", "--curiosity-band", "0.02:0.4", "--curiosity-goal", "on",
        "--curriculum", "on", "--decoder-kp", "6.0", "--delay-binding", "off", "--device", "cuda:0",
        "--episode-steps", "0", "--explain-body", "0", "--explain-every-min", str(explain_every_min), "--explain-keep-mb", "500.0",
        "--explain-steps", "0", *(["--fill-ring-first"] if fill_ring_first else []),
        "--fovea-fov", "24.0", "--fovea-size", "288",
        "--gaze-kd", "3.0", "--gaze-kp", "80.0", "--gaze-reflex", "on", "--goal-gaze-hold", "off",
        "--goal-locality", "off", "--goal-revalue", "off",
        # THE SELF-MODEL'S SLOW LEVELS, 2026-09-20. The whole-moment table is OFF: measured on
        # this lineage, its 6,400 stored patterns were all filled in the run's first 218 s of
        # body time, no moment of 2026-09-19 belonged to any of them (nearest pattern at cosine
        # 0.17 to 0.31 where 1 is the same pattern), and two thirds of the movement of their
        # presence came from a placeholder sense. The named-thing levels replace it: nine things
        # the world reports a presence for, so a thing can be pointed at and survives the
        # encoders changing.
        "--hierarchical-self", "off",
        "--entity-self", entity_self, "--entity-self-teacher", entity_self_teacher,
        "--self-value-weight", str(self_value_weight), "--self-value-level", "l2",
        "--self-value-gate-windows", "3",
        # WHAT THE COHERENCE TERM OF WELLNESS CARRIES: the share of the body the body map calls
        # its own, which rises as the perceived self grows. It used to carry how SURE the map
        # was, which on this lineage equalled 1 - 2 x that share (correlation -1.0000 over 1,402
        # records) and so paid for the perceived self SHRINKING, while carrying 53 of every 100
        # parts of the movement of wellness.
        "--wellness-coherence", wellness_coherence,
        "--hinge-armature", "0.02",
        "--imagination", "on", "--imagination-steps", "3", "--intention", "on",
        "--intention-every", str(intention_every),
        "--joint-damping-scale", "3.0", "--max-steps", str(max_steps), "--mobile", "on", "--motor-hold", "5",
        "--motor-state-formula", "boundary", "--movement-bonus", "0.2", "--movement-drive", "on",
        "--movement-mref", "24.0", "--movement-param-mod", "off", "--movement-penalty", "0.25",
        "--movement-sensor-mod", "off", "--n-envs", str(n_envs), "--next-frame-loss", "on",
        "--placeholder-senses", "on", "--plan-horizon", "earned", "--planner-effort-per-body", "on",
        "--probe-every", "20", "--prospective-wellness", "off", "--prospective-wellness-w", "0.15",
        "--render-every", "2", "--replay", "on", "--replay-batch", str(size["replay_batch"]), "--replay-every", str(size["replay_every"]),
        "--replay-size", "6000", "--report-every", str(report_every), "--residual-prediction", "on",
        "--ring-fov", "150.0", "--ring-size", "128", "--scene", "crib", "--scene-binding", "phase",
        "--seed", "0", "--smooth-actions", "0.35", "--spontaneous-saccades", "on", "--start-phase", "mature",
        "--steps-per-phase", "3000", "--think-harder", "on", "--top-down-gate", "on", "--touch-scale-mult", "1.0",
        "--video-every-min", "20.0", "--video-seconds", "30.0", "--vision-gate", "self-motion",
        "--wellness-mode", "baseline", "--minutes", "0", "--hself-slots", "6400",
        "--out", out,
        "--caregiver-mode", "responsive",
        "--caregiver-speech-dir", os.path.join(REPO, "Saved", "mimo_bridge", "caregiver_speech_exchange_v1_final"),
        "--body-layout", "trunk_v1", "--caregiver-record-dir", caregiver_record_dir,
        "--persistent-memory", "on", "--contact-profile", "mechanical_v4",
        "--caregiver-present-seconds", "60", "--caregiver-absent-seconds", "15",
        "--body-function", "modulate", "--self-state", "learn", "--self-choice", "off",
        "--checkpoint-storage", checkpoint_storage, "--self-prediction", "grounded_residual",
        "--rest-control", "choice",
        # 2026-09-18
        "--consequence-w", "0.5",
        "--core-blocks", str(size["core_blocks"]), "--d-model", str(size["d_model"]),
        "--vision-swin-layers", str(size["vision_swin_layers"]), "--vision-width", str(size["vision_width"]),
        "--encoder-layers", str(size["encoder_layers"]), "--encoder-width", str(size["encoder_width"]),
        *(["--resume", resume] if resume else []),
    ]
    return args


# Declared at launch, judged hourly by docs/h84_launch_2026-09-18/hourly_watch.py as window
# means (entries: name, path = dot field of the stats record, op, line, from_run_hour, mode,
# note). "alert" prints; "stop" writes snapshot_and_stop. The stop rules that need a measured
# baseline (replay loss > 3x hour 1, steps per second < 1/3 of hour 1, causal erosion) live in
# the watcher, which measures the baseline at its first check. Reachability of each line is
# in its note.
CRITERIA = [
    dict(name="execution_direction_match", path="cmd_response_cos_held", op="<", line=0.069, from_run_hour=2, mode="alert",
         note="How closely the body's turn matches its order over the 250 ms an order stays latched (cmd_response_cos_held), "
              "as a percentage of the way from 'unrelated to the order' (0%) to 'every part turned exactly the ordered way' "
              "(100%); turning against the order reads below 0%. The line is H83's reading at decoder gain 2, taken on the "
              "old gauge described below: 7% of the way. "
              "The goal line, physics alone at this run's gain 6 and half strength (scale_anchors.json command_following, "
              "re-measured 2026-09-23 on the repaired gauge): one 20-degree order to one body part turns 75% of the way "
              "(95% interval 65% to 85%) and 93 of every 100 ordered parts (28 of 30) end nearer the ordered orientation; for a "
              "whole-body posture the body can reach, 87 of every 100 parts end nearer (scale_anchors.json "
              "following_out_of_100). Distance to it: the segment 31 checkpoint's own orders, replayed in its saved world and "
              "scored on the repaired gauge, read 32% of the way, 43 points short of 75% "
              "(docs/gain_of_function_v3/measurements/following_gauge_before_after.json). Every H84 "
              "reading before 2026-09-23 was taken on the old gauge, which scored the eyes' row in place of the chest, "
              "spanned 26 ticks and took 4 of its 5 readings per order across two orders "
              "(docs/gain_of_function_v3/agents/fix-following-gauge.md): 15% of the way in the first hour, 8% by hour 19 "
              "with planner orders only, 10% after babble returned, 14% over segment 31. The 2026-09-18 goal line (72% of "
              "the way) counted its chest trials as unrelated, because the chest was never scored. From the fix on, "
              "cmd_response_cos_held is one reading per order over exactly its 25 ticks. This criterion watched the "
              "one-tick field cmd_response_cos (10 ms of physics, mostly momentum per env.py) until 2026-09-19."),
    dict(name="consequence_moments_exist", path="consequence.rate", op="<", line=0.001, from_run_hour=2, mode="alert",
         note="Share of goal-ring moments carrying a consequence (a hand contact, an object moved, a view gained). "
              "A whole hour under 0.001 means the body touched, moved and viewed nothing: the term has nothing to value."),
    dict(name="consequence_reaches_goal_choice", path="consequence.goal_rate", op="<", line=0.001, from_run_hour=2, mode="alert",
         note="Share of plans whose chosen goal moment carried a consequence (0 never, 1 every plan). With any "
              "consequence moment in the ring, 0.5 per kind exceeds H83's 0.42 ring-value spread, so the chosen goal "
              "should carry one; 0 with consequence.rate > 0 means the spread widened past 0.5 or the term is not "
              "reaching the argmax. Read together with consequence_moments_exist: both at 0 is a still body, "
              "only this one at 0 is the term."),
    dict(name="hand_strikes_the_mobile", path="task_mobile_hand", op="<", line=0.0001, from_run_hour=3, mode="alert",
         note="Share of body-steps on which a hand touched the mobile (H83: 0.0 all run). The hand handles are "
              "position handles and the consequence term values the contact; a whole hour at 0 from run-hour 3 "
              "means reach is still not happening."),
    dict(name="planner_moves_toward_goal", path="intention.gap_closed_pct", op="<", line=0.5, from_run_hour=3, mode="alert",
         note="The imagined rollout's distance closed, 0 = never moved toward its goal, 100 = arrived; None at "
              "horizon 1 (absent from the window until the planner earns depth 2). H83 read exactly 0.0."),
    dict(name="checkpoint_saves_land", path="checkpoint_failures", op=">", line=0, from_run_hour=0, mode="alert",
         note="Hourly saves that raised since the segment started (0 = every save landed). The trainer continues "
              "after a failed save and tries again next hour; the alert says the storage disk needs looking at."),
    dict(name="non_finite_update", path="nan_steps", op=">", line=0, from_run_hour=0, mode="stop",
         note="Any non-finite training update stops the run for repair (also the watcher's 5-minute check)."),
    # 2026-09-20: the self-model's slow levels over named things, and the coherence term of
    # wellness. Both alert rather than stop: the levels are dark, and a wellness term that reads
    # the wrong way is a reason to look, not a reason to lose the segment.
    dict(name="coherence_rewards_a_growing_self", path="wellness.coherence", op=">", line=0.5, from_run_hour=1, mode="alert",
         note="The coherence term of wellness now carries the EXTENT of the perceived self: the share "
              "of the body's handles the body map calls its own (0 = none, 1 = all). It ran 0.078 to "
              "0.338 on this lineage under the reading it replaces, so a window mean above 0.5 means "
              "the body map has claimed most of the body - which is the direction the objective wants, "
              "and worth reading beside the movement gauges rather than passing over. A value that "
              "never moves means the body map is not updating."),
    dict(name="named_things_are_being_watched", path="entity_self.l2_ticks", op="<", line=1, from_run_hour=3, mode="alert",
         note="Windows the named-thing levels have lived at the minutes level (10,000 moments each). "
              "Still 0 from run-hour 3 means they are not observing at all - the presence signals or "
              "the wiring are broken, not that nothing has been learned yet. A link cannot count "
              "before 13 windows (2 to warm the controls, 11 to climb the pass mark for nine things "
              "judged at once), which is 130,000 moments, about 30 hours of running at 1.2 moments "
              "per second."),
    # 2026-09-20: the self-model's belief about the body coming unstuck. Measured on this
    # lineage, self_state.present_error (how far the model's belief about its own body is
    # from the body's measured function, 0 = exactly right) ran 4.0e-05 to 1.0e-04 over
    # seg9's 350 records and seg13's first 89; at step 229,400 it jumped to 0.094 in one
    # report and never came back, the self-state term of the loss went from -3.89 to +141.7
    # while every other term held, and 2.8 hours later one body's forecast turned
    # non-finite and stopped the run. Eighteen automatic resumes then re-entered the same
    # diverged weights and crashed the same way, 6.4 hours for 418 steps. The line 0.01 is
    # 100x the healthy top and 5x below the diverged floor, so neither noise nor a slow
    # drift can reach it; a window that crosses it is a run whose self-model has come
    # unstuck, and continuing costs the hours that were paid once.
    dict(name="self_model_belief_stays_attached", path="self_state.present_error", op=">",
         line=0.01, from_run_hour=1, mode="stop",
         note="How far the model's belief about its own body sits from the body's measured "
              "function (0 = exactly right). Healthy on this lineage: 0.00004 to 0.0001. "
              "After the divergence of 2026-09-20 it read 0.049 to 0.100 and never "
              "recovered, and the run crashed 2.8 hours later on non-finite forecasts. "
              "Crossing 0.01 stops the run for repair: resume from the last checkpoint "
              "whose self-state loss was still negative, never from the diverged one."),
    # 2026-09-19, audit P238: the hierarchical self (dark - it feeds no loss, goal or wellness term,
    # so these two alert and never stop training). A SLOT is one line of its table: it stores one
    # pattern, what all the senses reported at one moment.
    dict(name="hself_slots_are_not_copies", path="hself.slots_twinned", op=">", line=0, from_run_hour=0, mode="alert",
         note="Slots of the hierarchical self whose stored pattern is a copy of another slot's (cosine above 0.9999; "
              "0 = every slot stores its own pattern). H84 read 0 in 10 checkpoints up to step 120,702. Since commit "
              "c4a0cde4b a moment changes only the slot it belongs to, so copies have no path; the two-kinds-of-moment "
              "test that ended with 9 copies in 11 slots under the old rule ends with 0. Any value above 0 means the "
              "rule is not doing what its test says: go back to ckpt_step_125325 on commit 828de24f0 and diagnose."),
    dict(name="hself_slots_are_not_drawing_together", path="hself.nearest_other_cos_median", op=">", line=0.95805, from_run_hour=0, mode="alert",
         note="How alike the middle slot's pattern and its nearest other slot's pattern are, as a cosine (1 = the same "
              "pattern). Under the old rule, in which every moment pulled on every slot, it rose from 0.9573 with the "
              "table just full (step 17,424) to 0.9580 (step 120,702): +0.0007 in 103,000 moments, so the printed "
              "value gained 0.0001 about every 15,000 moments. It read 0.9580 when the new rule landed at step "
              "125,325. Measured on the run's recorded moments, 0 of them belong to a slot of this table, so no "
              "stored pattern changes and the value stays exactly 0.9580 while moments_with_a_slot reads 0; 0.9581 "
              "means slots are still being drawn together. (Once moments do belong to slots, patterns move toward "
              "their own moments and this line needs declaring again.)"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="a short run into a scratch run directory, reporting every 10 steps; replay starts at once "
                         "(no --fill-ring-first) so the peak memory includes the re-study lesson")
    ap.add_argument("--smoke-steps", type=int, default=150,
                    help="replay starts once the ring holds a full context (128 steps); 150 reaches it")
    for k, v in SIZE.items():
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    ap.add_argument("--smoke-dir", default="")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--prepare", action="store_true",
                    help="write the receipt, the criteria and <out>/launch_command.json (argv + environment) and start "
                         "nothing: the caller starts the trainer itself. Measured 2026-09-18 23:10: a trainer this "
                         "script started from a tool session died 20 s later when the session's process tree was torn "
                         "down, while processes PowerShell's Start-Process parented directly survived 19 hours.")
    ap.add_argument("--resume", default="", help="a complete checkpoint folder to continue from (a new segment of the lineage)")
    ap.add_argument("--run-name", default=RUN_NAME, help="the segment's run directory name under experiments/")
    ap.add_argument("--intention-every", type=int, default=10)
    ap.add_argument("--explain-every-min", type=float, default=60.0,
                    help="minutes between explainer snapshots written by the trainer (0 = off). Hourly since "
                         "2026-09-19 (operator): segments 1-3 ran with it off, so no H84 page could be built")
    ap.add_argument("--entity-self", choices=["off", "dark"], default="dark",
                    help="the self-model's slow levels over NINE NAMED THINGS (the caregiver, "
                         "three toys, the mobile, the squeaker, two critters, the baby's own hand "
                         "in view). dark: watching only")
    ap.add_argument("--entity-self-teacher", choices=["wellness", "ability"], default="wellness",
                    help="what a thing's presence is correlated WITH. wellness is defensible again "
                         "now that its coherence term rewards the perceived self growing rather "
                         "than shrinking; ability counts the kinds of consequence the body caused")
    ap.add_argument("--self-value-weight", type=float, default=0.1,
                    help="the reader's weight: how much one unit of proven link times presence is "
                         "worth when the planner ranks a remembered moment. 0.1 against a ring "
                         "whose stored values spanned 0.42 on H83, so three proven things present "
                         "at once can move a moment by 0.3 and outrank a still one without "
                         "swamping the five wellness terms. It adds nothing at all until the "
                         "reader's gate opens, which needs 13 windows of 10,000 moments")
    ap.add_argument("--wellness-coherence", choices=["extent", "certainty"], default="extent",
                    help="which reading of the body map wellness carries; extent is the share of "
                         "the body it calls its own")
    ap.add_argument("--hself-tap-steps", type=int, default=0,
                    help="record the first N frame batches the hierarchical self observes into "
                         "<out>/hself_tap, exactly as it sees them (0 = off). A measurement aid for audit P238: a "
                         "change to the table's centre update is replayed on these moments before a run depends on "
                         "it. The recorder changes no number the run computes (test_hself_capacity_repairs.py)")
    a = ap.parse_args(argv)
    size = {k: getattr(a, k) for k in SIZE}
    run_name = a.run_name
    if a.smoke:
        # short: the run name appears twice in the companion file's path (run directory and
        # caregiver storage) and Windows refuses paths past 260 characters
        tag = (f"smoke_h84_{a.n_envs}b_c{size['core_blocks']}x{size['d_model']}"
               f"_v{size['vision_swin_layers']}x{size['vision_width']}_e{size['encoder_layers']}x{size['encoder_width']}"
               f"_r{size['replay_batch']}e{size['replay_every']}")
        out = a.smoke_dir or os.path.join(REPO, "Saved", "mimo_bridge", "experiments", tag)
        cmd = command(a.n_envs, out, os.path.join(out, "checkpoints"),
                      os.path.join(REPO, "Saved", "mimo_bridge", "caregiver_records", tag),
                      max_steps=a.smoke_steps, report_every=10, fill_ring_first=False, size=size,
                      intention_every=a.intention_every, explain_every_min=a.explain_every_min,
                      entity_self=a.entity_self, entity_self_teacher=a.entity_self_teacher,
                      self_value_weight=a.self_value_weight, wellness_coherence=a.wellness_coherence)
    else:
        out = os.path.join(REPO, "Saved", "mimo_bridge", "experiments", run_name)
        # caregiver records stay with the lineage's original folder across segments, as the
        # supervisor's own resumes keep them
        cmd = command(a.n_envs, out, os.path.join(out, "checkpoints"),
                      os.path.join(REPO, "Saved", "mimo_bridge", "caregiver_records", RUN_NAME), size=size,
                      intention_every=a.intention_every, explain_every_min=a.explain_every_min, resume=a.resume,
                      entity_self=a.entity_self, entity_self_teacher=a.entity_self_teacher,
                      self_value_weight=a.self_value_weight, wellness_coherence=a.wellness_coherence)
    if a.print or not (a.smoke or a.launch or a.prepare):
        print(" ".join(cmd))
        return
    os.makedirs(out, exist_ok=True)
    os.makedirs(cmd[cmd.index("--checkpoint-storage") + 1], exist_ok=True)
    # expandable_segments: the caching allocator grows segments instead of reserving fixed
    # blocks, so less of the card is "reserved but unallocated" (505 MiB in the 24-body
    # smoke's out-of-memory report). An allocator setting, not a training setting. Measured
    # 2026-09-18 23:13 in the trainer's stderr: PyTorch reports it "not supported on this
    # platform" (Windows), so it was a no-op in every run today; kept so the receipt says
    # what was asked for.
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "Source", "Python"), PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env_keys = ["PYTHONPATH", "PYTHONIOENCODING", "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF"]
    if a.hself_tap_steps > 0:
        env.update(HSELF_TAP_DIR=os.path.join(out, "hself_tap"), HSELF_TAP_STEPS=str(a.hself_tap_steps))
        env_keys += ["HSELF_TAP_DIR", "HSELF_TAP_STEPS"]
    if a.launch or a.prepare:
        with open(os.path.join(out, "declared_criteria.json"), "w", encoding="utf-8") as f:
            json.dump(CRITERIA, f, indent=2)
        receipt = dict(command=cmd, created=datetime.datetime.now().astimezone().isoformat(),
                       reason="2026-09-18 launch on the recommended changes (docs/MIMo_Gain_of_Function_Tree_2026-09-17.html): "
                              "fresh weights, shadow actor off, mechanical_v4, the size the card allows (core 4 x 192 / vision "
                              "4 x 192 / encoders 2 x 192, replay 24 every 2; 14.4 M parameters), planner cadence = latch, decoder "
                              "gain 6 (measured), consequence access 0.5 per kind, per-body effort charge; growth ramp not run; "
                              "8 bodies; the smokes that chose the size are in docs/h84_launch_2026-09-18/README.md.",
                       declared_criteria=CRITERIA, size=size, n_envs=a.n_envs, intention_every=a.intention_every, explain_every_min=a.explain_every_min,
                       resume=a.resume, run_name=run_name, hself_tap_steps=a.hself_tap_steps,
                       entity_self=a.entity_self, entity_self_teacher=a.entity_self_teacher,
                       self_value_weight=a.self_value_weight, wellness_coherence=a.wellness_coherence,
                       environment=dict(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="1"),
                       watcher="docs/h84_launch_2026-09-18/hourly_watch.py (own-baseline stop rules; see its docstring)")
        rp = os.path.join(REPO, "Saved", "mimo_bridge", "h84_launch_command_20260918.json" if run_name == RUN_NAME
                          else f"{run_name}_launch_command.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print("receipt", rp)
        if a.prepare:
            with open(os.path.join(out, "launch_command.json"), "w", encoding="utf-8") as f:
                json.dump(dict(argv=cmd, cwd=os.path.join(REPO, "Source", "Python"),
                               env={k: env[k] for k in env_keys},
                               log=os.path.join(out, "launcher_stdout.log")), f, indent=2)
            print("prepared", os.path.join(out, "launch_command.json"))
            return
        log = open(os.path.join(out, "launcher_stdout.log"), "ab", buffering=0)
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        p = subprocess.Popen(cmd, cwd=os.path.join(REPO, "Source", "Python"), env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        print("trainer pid", p.pid, "out", out)
        return
    # smoke: foreground, stream to the log
    with open(os.path.join(out, "smoke_stdout.log"), "wb", buffering=0) as log:
        rc = subprocess.call(cmd, cwd=os.path.join(REPO, "Source", "Python"), env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
    print("smoke exit", rc, "log", os.path.join(out, "smoke_stdout.log"))


if __name__ == "__main__":
    main()
