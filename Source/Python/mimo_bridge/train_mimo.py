# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Phase 2: train the lite LEM model on N MIMo bodies at once, with the growth curriculum.

    python -m mimo_bridge.train_mimo --n-envs 4 --minutes 60 --scene reach --curriculum on \
        --age-binding curriculum --out Saved/mimo_bridge/runs/E2_curr_seed0 --seed 0

What it does, in plain words:
  * starts N MIMo bodies (N worker processes), one lite LEM model on the GPU;
  * every env step: all N frames go through the model as one batch of N, the model learns
    one step, and its N actions go back to the N bodies;
  * the developmental curriculum (lem/training/curriculum.py, 6 phases) sets the range of
    motion, strength, babbling noise and which handles may move; when the phase changes
    the bodies are rebuilt at the phase's MIMo AGE and sensory/motor DELAYS (age binding);
  * stats go to <out>/replay_training_stats.jsonl with the SAME record types and names the
    UE run writes (type "inference" and type "training" from lem/reporting.py) plus a
    type "mimo" record with MIMo-only numbers (per-modality PE, wellness, touch, age);
  * a full checkpoint of every module (safetensors + all_weights.pt) is saved at least
    every --checkpoint-every-min minutes and at the end; paths listed in checkpoints.json.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimo_bridge.datasource import SingleEnvSource, PoolSource
from mimo_bridge.lem_model import COMMAND_POSITION_RANGE_CM, LemLite
from mimo_bridge.prediction_audit import accumulate as accumulate_prediction_audit, summarize as summarize_prediction_audit
from mimo_bridge.prediction_audit import accumulate_rows as accumulate_prediction_audit_rows, summarize_rows as summarize_prediction_audit_rows
from mimo_bridge.prediction_audit import summarize_hold_first as summarize_prediction_audit_hold_first

# the rest-branch pose-hold uses the literal 200.0 so the review's isolated AST
# fixtures (docs/review_2026-09-08/model_contract_checks.py) can execute it
# without this module's imports; this assert keeps the literal tied to the one
# shared constant (the calibrate.py lesson: an untied copy passes a value
# nobody runs)
assert COMMAND_POSITION_RANGE_CM == 200.0
from mimo_bridge.transition import Pending
from mimo_bridge.latent_body_shadow import add_arguments as add_latent_body_arguments, observe_shadow
from mimo_bridge.comparison_records import snapshot_weights
from mimo_bridge.hand_target_control import DIGITS as HAND_DIGITS
from mimo_bridge.limb_target_control import (ledger_from_frames, check_position_rows, planner_calf_geometry,
                                             check_finger_servo,
                                             action_contract_parts as limb_action_contract_parts,
                                             window_add as limb_window_add,
                                             window_summary as limb_window_summary)
from lem.reporting import build_inference_stats_record, build_sync_training_stats_record
from lem.training.curriculum import CurriculumManager, PHASE_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mimo_bridge.train")

# curriculum phase -> MIMo age (months) and (sensory, motor) delay in env steps of 10 ms.
# Guess written down: the model's 6 phases spread over MIMo's 0..24 month range; the MOTOR
# delay shrinks as the infant matures (MIMo's delay feature; k means k steps).
#
# THE SENSORY ENTRY IS 0 AT EVERY PHASE and BridgeEnv now refuses anything else. MIMo
# applies sensory_delay inside _get_obs, and no sense path in the bridge consumes what
# _get_obs returns - env.py _collect reads touch.sensor_outputs, _contacts(), proprio.read()
# and get_vestibular_obs() straight off live MuJoCo. Measured 2026-09-16: sensory_delay 4
# against 0 leaves proprioception, touch and vestibular identical on 12 of 12 steps, maximum
# difference 0.000e+00; the same probe with motor_delay 3 against 0 differs on 11 of 12, so
# that zero is a real zero and not a blind probe. The old 4/3/2/1/1/0 column therefore never
# reached a sense, and keeping it cost something real: with --delay-binding on, two phase
# transitions (early_infant->late_infant and child->mature) differed ONLY in that column and
# would rebuild every body and drop the context window, the pending lessons and the goal ring
# for a variable nothing reads. Implementing it instead is a separate, operator-owned change:
# it would alter what the body feels in 4 of 6 phases (at neonate the next moment would show
# a measured mean 0.323 of an order's 50 ms proprioception effect instead of 1.0), and the
# 4/3/2/1 values have no measurement behind them.
PHASE_AGE = {"neonate": 1.0, "early_infant": 4.0, "late_infant": 8.0, "toddler": 13.0, "child": 18.0, "mature": 24.0}
# developmental scenes: like a real infant, the body is FREE at every stage - no welds, no
# pins. It starts on its back in the playpen and must learn everything itself: kick, roll to
# the side, roll over, reach the mobile, get to the boxes. (The old "reach"/"selfbody" scenes
# with their welds remain available via --scene for comparisons only.)
PHASE_SCENE = {"neonate": "crib", "early_infant": "crib", "late_infant": "crib",
               "toddler": "crib", "child": "crib", "mature": "crib"}
PHASE_DELAYS = {"neonate": (0, 3), "early_infant": (0, 2), "late_infant": (0, 2), "toddler": (0, 1), "child": (0, 0), "mature": (0, 0)}

# touch patch -> coarse body region (for E3 analysis)
def region_of(name: str) -> str:
    if "eye" in name or name == "head":
        return "head"
    if any(k in name for k in ("hand", "distal", "middle", "knuckle", "metacarpal", "thbase", "thhub")):
        return "hand_L" if name.startswith("left") else "hand_R"
    if "arm" in name:
        return "arm_L" if name.startswith("left") else "arm_R"
    if "leg" in name:
        return "leg_L" if name.startswith("left") else "leg_R"
    if any(k in name for k in ("foot", "toe")):
        return "foot_L" if name.startswith("left") else "foot_R"
    return "torso"


def skin_regions_window(skr: dict, skr_n: dict, n_bodies: int):
    """The window's skin-region report: (mean over bodies, per-body list), or (None, None).

    skr[body][region] = [patches, force_share, self_patches] summed over the steps that body
    reported; skr_n[body] = those steps. Each body's entry is the window MEAN of the world's
    per-step region numbers: patches = skin patches of that region in contact, force_share =
    that region's share of the body's touch force (0-1), self_patches = its patches touched by an
    own part. The first value is the same three numbers averaged over the bodies that reported.
    A body that never reported reads {} in the list."""
    if not skr_n:
        return None, None
    per_body = []
    for b in range(int(n_bodies)):
        n = int(skr_n.get(str(b), 0) or 0)
        regions = skr.get(str(b)) or {}
        per_body.append({} if n <= 0 else {
            name: {"patches": round(s[0] / n, 3), "force_share": round(s[1] / n, 4),
                   "self_patches": round(s[2] / n, 3)} for name, s in regions.items()})
    names = sorted({name for d in per_body for name in d})
    rep = [d for d in per_body if d]
    mean = {name: {k: round(float(np.mean([d[name][k] for d in rep if name in d])), 4)
                   for k in ("patches", "force_share", "self_patches")}
            for name in names} if rep else None
    return mean, per_body


def self_contact_region_key(onset: dict) -> str:
    """The window's region-pair key of one self-contact onset: its two regions in NAME order, so
    "left hand|trunk" and "trunk|left hand" are one entry (the world sorts the pair by BODY name,
    which split 23 region pairs in two; review of b4, minor 8)."""
    return "|".join(sorted(str(x) for x in (onset.get("regions") or ("?", "?"))))


def self_contact_events_window(acc: dict, top: int = 5) -> Optional[dict]:
    """The `self_contact_events` block (DESIGN section 3) from the window's world reports, or None
    when no body reported self_contact_onsets (absent = unmeasured).

    onsets: self-touches that began in the window, all bodies (a pair of the body's own parts
    apart at least 5 env frames, 50 ms, and touching now; one touch felt on both sides is one);
    onsets_per_body_step: the same per body per moment; pairs_top: the most frequent region pairs
    [regionA, regionB, count], pairs_top_bodies the same by body name; structural_share: share of
    onsets that are a pair touching at rest under zero command (the thumb hub in its palm);
    in_view_share: share whose contact point was within 30 degrees of either eye's axis."""
    steps = int(acc.get("sce_body_steps", 0) or 0)
    if steps <= 0:
        return None
    n = int(acc.get("sce_onsets", 0) or 0)

    def ranked(d):
        return [[*k.split("|", 1), int(v)] for k, v in
                sorted((d or {}).items(), key=lambda kv: (-kv[1], kv[0]))[:top]]
    return {"onsets": n, "onsets_per_body_step": round(n / steps, 4),
            "pairs_top": ranked(acc.get("sce_regions")),
            "pairs_top_bodies": ranked(acc.get("sce_bodies")),
            "structural_share": (round(int(acc.get("sce_structural", 0) or 0) / n, 4) if n else None),
            "in_view_share": (round(int(acc.get("sce_in_view", 0) or 0) / n, 4) if n else None),
            "body_steps": steps}


def record_planning_decision(acc, model, previous_inputs, step, *, latched, source,
                             externally_overridden=False):
    """Retain each decision in its report window, including decisions with no action."""
    current = getattr(model, "_last_plan_inputs", None)
    if current is None or current is previous_inputs:
        return
    window = acc.setdefault("planning", {})
    window.setdefault("first_decision_step", int(step))
    window["last_decision_step"] = int(step)
    window["decisions"] = window.get("decisions", 0) + 1
    selected = current.get("plan0") is not None
    reached = selected and latched and source == "planner" and not externally_overridden
    for key, value in (("selected_plans", selected), ("latched_plans", reached),
                       ("no_action_decisions", not selected)):
        window[key] = window.get(key, 0) + int(value)
    detail = dict(current.get("outcome") or {})
    detail.update(step=int(step), selected_plan=selected, command_selected_for_body=bool(reached))
    if selected:
        intent = getattr(model, "last_intention", None)
        if not isinstance(intent, dict) or intent.get("step") != step:
            raise RuntimeError("Selected plan has no matching decision record")
        detail = copy.deepcopy(intent) | detail
        window["latest_plan"] = copy.deepcopy(detail)
    window["latest_decision"] = detail


def planning_window_summary(acc, first_step, stop_step):
    """Window step bounds are inclusive/exclusive, independent of the decision schedule."""
    values = copy.deepcopy(acc.get("planning") or {})
    last_plan = values.pop("latest_plan", None)
    for name in ("decisions", "selected_plans", "latched_plans", "no_action_decisions"):
        values.setdefault(name, 0)
    values.update(first_step=int(first_step), stop_step=int(stop_step))
    return last_plan, values


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", default="4", help="number of bodies, or 'auto' = as many as this machine can run")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--scene", default="crib",
                    help="crib is the developmental scene: free body on its back in the "
                         "playpen. 'reach' WELDS the upper body and hangs it in the air, "
                         "which is a rig for isolating arm control, not a nursery. It used "
                         "to be the default and every posture, mobile, caregiver and pen-art "
                         "change is written under 'crib', so a run that forgot "
                         "--scene-binding phase silently got none of them: H16, H17 and the "
                         "first H18 all trained a suspended body (2026-08-24)")
    ap.add_argument("--scene-binding", choices=["fixed", "phase"], default="fixed",
                    help="phase = the scene follows the growth phase through PHASE_SCENE. That "
                         "table is 'crib' at all six phases today, so this flag changes no "
                         "scene: the scene never differs between two phases and so never enters "
                         "the rebuild. The crib -> sitting -> reach progression this help used "
                         "to promise was never implemented, and is where the explainer page's "
                         "invented 'sitter' phase box came from.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--isolated-native-proof",default="",
                    help="Operational role: exact declaration of a validated bounded CPU native proof")
    ap.add_argument("--allow-preflight-fail", action="store_true",
                    help="launch even when tools/preflight.py fails - for experiments "
                         "that deliberately violate a measured band; the violation "
                         "stays loud in the log and preflight.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-every", type=int, default=2)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--placeholder-senses", choices=["on", "off"], default="on",
                    help="off disables audio and intermind. Their loss is masked out either "
                         "way, but the placeholder tokens still enter their encoders, the "
                         "cross-modal attention and the hidden state - fresh Gaussian noise "
                         "in the representation, not a harmless zero-weight channel.")
    ap.add_argument("--residual-prediction", choices=["on", "off"], default="off",
                    help="the prediction heads output the CHANGE and the current token is "
                         "added, so 'nothing happens' is the zero output rather than a "
                         "mapping the network must keep re-learning. Measured on H34: "
                         "persistence nearly matches the model on every sense, so most of "
                         "the target is copyable and the action is worth a few percent of "
                         "the loss; an MLP with 1,547 examples gets 9.2%% from the order "
                         "where the trained model gets ~0. Every prediction site must pass "
                         "the current tokens or the heads raise - half-application is not "
                         "possible. Changes what the weights mean: do not resume a "
                         "non-residual checkpoint into it and expect continuity.")
    ap.add_argument("--reinit", default="",
                    help="comma-separated components to start FRESH on --resume, e.g. "
                         "encoder:motor. Their checkpoint weights are skipped and their "
                         "optimizer moments dropped; everything else loads.")
    ap.add_argument("--fresh-optimizer", action="store_true",
                    help="resume weights but not Adam moments. Use when the MEANING of an "
                         "input changed: the moments were accumulated against the old "
                         "semantics and would steer the new pathway with stale statistics.")
    ap.add_argument("--fill-ring-first", action="store_true",
                    help="hold REPLAY back until enough same-episode transitions carry their "
                         "outgoing action to build a context window and a batch. It does NOT "
                         "gate live training, which continues throughout - so this is not a "
                         "clean-data guarantee for the run, only for the replay stream. The "
                         "ring is in-memory and starts empty at every launch, so there is no "
                         "legacy history to purge and no reason to wait for a full 6000.")
    ap.add_argument("--gaze-reflex", choices=["on", "off"], default="on",
                    help="snap-to-motion orienting reflex in the env worker (endowed anatomy, "
                         "permanent; 'off' only for A/B baselines like H18)")
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--episode-steps", type=int, default=2000, help="reset every N observations; 0 keeps the physical episode continuous")
    ap.add_argument("--curriculum", choices=["on", "off"], default="on")
    ap.add_argument("--steps-per-phase", type=int, default=3000)
    ap.add_argument("--start-phase", default="neonate", help="curriculum start phase (e.g. child = strong, free body)")
    ap.add_argument("--age-binding", default="curriculum", help="'curriculum' or 'fixed:<months>' or 'none'")
    ap.add_argument("--delay-binding", choices=["on", "off"], default="on")
    ap.add_argument("--wellness-mode", default="baseline", choices=["baseline", "direct", "forward"])
    ap.add_argument("--motor-state-formula", default="boundary", choices=["preference", "boundary"],
                    help="preference = wellness prefers every joint at mid-range, and since "
                         "motor_state is the ONLY wellness term that steers the body, that one "
                         "choice pushed the arms outward forever: never across the body, never "
                         "holding a press (both are far-from-centre postures). boundary = zero "
                         "penalty in the interior, quadratic only near the mechanical stops - "
                         "posture is free, only real limits repel")
    ap.add_argument("--gaze-kp", type=float, default=80.0,
                    help="eye servo stiffness, from a main-sequence sweep: kp 2.0 gave 27 deg/s "
                         "and 700 ms to settle so the eye never arrived; 80 gives 110 deg/s and "
                         "130 ms with no overshoot. Speed saturates near 110 (actuator limit)")
    ap.add_argument("--gaze-kd", type=float, default=3.0, help="eye servo damping (see --gaze-kp)")
    ap.add_argument("--motor-hold", type=int, default=5,
                    help="how many model steps one movement ORDER stays in force. Orders used "
                         "to be replaced every 50 ms while a limb needs about 250 ms, so ~10%% "
                         "of each was done before the next arrived (direction match -0.06 live "
                         "against +0.87 for a held order). Raising action_repeat would fix that "
                         "too but would drop SENSING to 4 Hz, which the gaze work needs at 20 Hz "
                         "to resolve a 300-600 ms saccade. This holds the order and leaves "
                         "sensing and learning untouched. 1 = off (the old behaviour)")
    ap.add_argument("--movement-bonus", type=float, default=0.2,
                    help="wellness bonus scale for moving (was 0.35). The movement drive was "
                         "measured swinging ~0.85 while every other term swings at most ~0.4 - "
                         "it drowned them, and the result was ceaseless flailing")
    ap.add_argument("--movement-penalty", type=float, default=0.25,
                    help="stillness-debt penalty scale (was 0.5). Pressing on something IS "
                         "stillness; punishing it superlinearly forbade pressing")
    ap.add_argument("--imagination", choices=["on", "off", "config"], default="config")
    ap.add_argument("--random-actions", action="store_true", help="E4 baseline: body gets random actions")
    ap.add_argument("--checkpoint-every-min", type=float, default=60.0)
    ap.add_argument("--checkpoint-storage", default="",
                    help="Optional per-run checkpoint directory on another drive; its parent receives this run's args.json")
    ap.add_argument("--probe-touch", action="store_true", help="E3: save tactile per-patch embeddings + region labels")
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--touch-scale-mult", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-only", action="store_true", help="act but do not learn (weights frozen): for probes")
    ap.add_argument("--retire-acting-snapshot", action="store_true",
                    help="drop the episode companion's saved frozen actor and qualify a "
                         "fresh candidate under the CURRENT configuration and code - the "
                         "explicit operator action for changing acting flags or bound "
                         "code on a shadow lineage; the retired actor's recorded "
                         "decisions stay in the old run's stats")
    ap.add_argument("--episode-continuation", choices=["resume", "fresh"], default="resume",
                    help="fresh: skip the saved physical-episode companion and start new worlds "
                         "(eval-only probes only - a training resume must migrate, never restart "
                         "the episode); the saved companion is never read or written")
    ap.add_argument("--no-babble", action="store_true", help="force exploration noise to zero: every motion is the model's own")
    ap.add_argument("--babble-noise", type=float, default=None,
                    help="ABSOLUTE exploration noise, overriding the developmental phase. "
                         "The phase table sets neonate 0.010, early_infant 0.008, late_infant "
                         "0.005, toddler 0.003, child 0.001 and mature 0.000 - but every one of "
                         "those is below BABBLE_FLOOR (0.05, curriculum.py) and limits_for() "
                         "returns the floor, so what the trainer receives is 0.05 in ALL SIX "
                         "phases and --babble-scale multiplies 0.05, not zero. H28 predates the "
                         "floor: it ran mature at 0.0 with the planner off too, leaving an "
                         "untrained controller as the only action source: 0.07 rad orders.")
    ap.add_argument("--babble-scale", type=float, default=1.0,
                    help="multiply the developmental exploration noise. UEMA 19.2 makes motor "
                         "babbling the bootstrap: the null prediction has to be BROKEN before "
                         "self-vs-world can separate. With the planner off, H23-H28 commanded "
                         "0.03-0.08 rad against 0.42-0.68 with it on, and nothing moved.")
    ap.add_argument("--limp", action="store_true", help="force strength to zero (a limp body): measures what pure physics scores")
    ap.add_argument("--body-schema-updates", choices=["on", "off"], default="off",
                    help="the self-vs-world map (UEMA 11.1): per-body motor-sensor correlation "
                         "EMAs update every step and coherence feeds the stats; state rides "
                         "runtime_state.pt across resumes. Off = channel bypassed (the default "
                         "this flag exposes; the mechanism itself is built and tested)")
    ap.add_argument("--replay", choices=["on", "off"], default="on", help="the diary: re-study stored moments between live steps")
    ap.add_argument('--body-function',choices=['off','observe','modulate'],default='off',
                    help='versioned local sensing capability; observe preserves actual senses, modulate changes resolution')
    ap.add_argument('--self-state',choices=['off','learn','perceive'],default='off',
                    help='learn local self perception; perceive also contributes its learned state to the brain')
    ap.add_argument('--self-prediction',choices=['legacy','grounded_residual'],default='legacy',
                    help='preserve prior learning while predicting local changes from observed functioning')
    ap.add_argument('--self-percept-release',choices=['off','on'],default='off',
                    help='on: no stored self-percept carries more than 0.05 of a body\'s recall, a body '
                         'whose live self-percept had to be rescaled to the limit (8), or whose restored '
                         'stored percepts sit at it, starts its stored percepts again from zero, and the '
                         'saved inputs and rehearsal windows holding them are cleared. H84 held one body '
                         'at the limit from 7:41 AM on 2026-09-25, four by step 415,075 '
                         '(docs/gain_of_function_v4/agents/self_percept_runaway_2026-09-26.md)')
    ap.add_argument('--self-choice',choices=['off','shadow','on'],default='off')
    ap.add_argument('--self-choice-receipt',default='',help='located physical validation and calibration')
    ap.add_argument("--replay-every", type=int, default=4, help="one re-study update every N live steps")
    ap.add_argument("--replay-batch", type=int, default=48)
    ap.add_argument("--replay-size", type=int, default=6000, help="moments remembered per body")
    ap.add_argument("--think-harder", choices=["on", "off"], default="off",
                    help="adaptive motor controller: more thinking blocks when unsure")
    ap.add_argument("--intention", choices=["on", "off"], default="off",
                    help="plan toward the best-feeling remembered moment and act on it")
    ap.add_argument("--intention-every", type=int, default=5)
    ap.add_argument("--no-intention-execute", action="store_true",
                    help="plan and measure but never act on the plan (metadata only)")
    ap.add_argument("--action-tenure-loss", choices=["on", "off"], default="off",
                    help="grade the action's marginal contribution at the order's END: "
                         "predict the tenure's change with the order and with 'do nothing' "
                         "(stop-grad), and require the order branch to win by alpha")
    ap.add_argument("--action-tenure-weight", type=float, default=0.5)
    ap.add_argument("--action-tenure-token-reach", choices=["on", "off"], default="off",
                    help="order-to-outcome matching into the encoders: a projection of "
                         "the order must pick out its own tenure's re-encoded token "
                         "change among recent decoys (bench-validated collapse-proof)")
    ap.add_argument("--action-tenure-token-weight", type=float, default=0.25)
    ap.add_argument("--action-tenure-token-alpha", type=float, default=0.005,
                    help="UNUSED since the matching form replaced the token hinge; "
                         "kept so old launch commands parse (review R11)")
    ap.add_argument("--agency-drive", choices=["on", "off"], default="off",
                    help="evidence of your own effect pays wellness: bounded match "
                         "payment (vision+touch) plus a moving-hand-in-fovea bootstrap "
                         "that fades by touch-parity displacement")
    ap.add_argument("--agency-weight", type=float, default=0.1)
    ap.add_argument("--agency-prior", type=float, default=0.05)
    # CONSEQUENCE ACCESS in the goal-ring value (2026-09-18, gain-of-function tree W2/W1).
    # Per body, per moment: one unit per kind of self-relevant consequence the world
    # measured (contact by a hand; an object moved by the body; a view gained), times
    # this weight, added to the value the planner ranks remembered moments by. 0 = off
    # (every run before 2026-09-18). Measured against H83's ring, whose stored values
    # spanned 3.993-4.410: 0.5 per kind lets one consequence moment outrank every still
    # moment. The direct reading is the stats field cq_goal_rate - how often the
    # planner's chosen goal moment carried a consequence.
    ap.add_argument("--consequence-w", type=float, default=0.0,
                    help="weight per kind of self-relevant consequence in the goal-ring value (0 = off)")
    ap.add_argument("--refresh-action-calibration", action="store_true",
                    help="on resume: refresh the grounded self dynamics' stored command centre and scale "
                         "from the resumed run's own stored executed orders, rescaling the command "
                         "encoder so every predicted value is unchanged (audit P203)")
    ap.add_argument("--mobile", choices=["on", "off"], default="off",
                    help="hang a strikeable pendulum mobile over the chest (bob at "
                         "the wrists' measured p90 height, 0.21 m): the Rovee-"
                         "Collier lever - a large lasting self-caused visual event")
    ap.add_argument("--vision-gate", choices=["hand", "self-motion"], default="hand",
                    help="hand: vision's matching slice needs the hand in view at both "
                         "tenure ends. self-motion: or the body's own mobile moving "
                         "above the measured 2 cm/s ambient floor - each body is alone "
                         "in its world, so that motion is its own doing")
    ap.add_argument("--plan-horizon", choices=["fixed", "earned"], default="fixed",
                    help="fixed: max_plan_steps from config (the old constant). "
                         "earned: each imagination depth unlocks while its "
                         "prediction beats do-nothing (win EMA > 0.5); horizon = "
                         "1 + licensed depths. Demonstrated skill, no clock.")
    ap.add_argument("--action-tenure-alpha", type=float, default=0.04,
                    help="hinge margin for the pose branch (scale-free ratio form)")
    ap.add_argument("--movement-drive", choices=["on", "off"], default="off",
                    help="Stage 1: movement-hunger wellness (stillness debt, movement bonus)")
    ap.add_argument("--movement-sensor-mod", choices=["on", "off"], default="off",
                    help="Stage 2: stillness numbs tactile/proprio gain, movement sharpens")
    ap.add_argument("--decoder-kp", type=float, default=8.0, help="servo stiffness (8 is stable once armature is on)")
    ap.add_argument("--joint-damping-scale", type=float, default=3.0, help="multiply MIMo joint damping (physics-rate, unlike our PD)")
    ap.add_argument("--hinge-armature", type=float, default=0.02, help="rotor/tendon inertia added to hinge joints: the anti-buzz fix")
    ap.add_argument("--finger-armature", type=float, default=0.0004,
                    help="rotor inertia added to the finger and thumb hinges instead of --hinge-armature. "
                         "The limbs' 0.02 made each finger about 200 times harder to turn and the "
                         "fingers could not follow their commands (measured 2026-09-22, see env.py)")
    ap.add_argument("--curiosity-band", default=None,
                    help="lo:hi PE band for the curiosity gate, e.g. 0.02:0.4 (default: config 0.1:0.4 - too high for a competent brain)")
    ap.add_argument("--corrupt-vision", type=float, default=0.0,
                    help="each step, replace the eye patches with noise at this probability: the Act 3 recovery test")
    ap.add_argument("--audio", choices=["real", "placeholder"], default="placeholder",
                    help="real = the crib hears: spectrum-space synthesis of impacts, the "
                         "squeaker's squeak and the bee's buzz into the avatar's own "
                         "256-bin stereo hearing contract (audio_adapter). The audio "
                         "encoder is already in every checkpoint; this feeds it")
    ap.add_argument("--babble-bouts", choices=["on", "off", "real"], default="off",
                    help="babble in infant-like motion bouts (2-5 s on, 1-3 s rest; "
                         "scheduled duty 64%% vs the infant 40-75%% band) instead of "
                         "every step - queue attempt #3, measured basis: duty was "
                         "100%% with zero pauses, and a never-resting body makes its "
                         "own future unpredictable (capacity test 2026-09-07). "
                         "'on' gates NOISE only - measured 2026-09-07: the planner "
                         "fully replaces it (command size identical, 0.531 vs 0.536 "
                         "rad) and the body never rests. 'real' also nulls the MOTOR "
                         "command during rest windows (zero strength, body settles; "
                         "eyes stay live; senses and lessons run throughout; the "
                         "executed-action record carries the null truthfully) - "
                         "quiet alertness by schedule, like sleep is for an infant")
    ap.add_argument("--goal-locality", choices=["on", "off"], default="off",
                    help="goal choice becomes best-felt WITHIN SELF-MEASURED REACH, "
                         "held until arrival/escape/old-age (operator decision "
                         "2026-09-06; measured basis: goals were chosen 1.6-1.9x "
                         "beyond the 14-step motion radius and plans never arrived; "
                         "reach radius and arrival line are the model's own running "
                         "measurements - nothing hand-tuned)")
    ap.add_argument("--curiosity-goal", choices=["on", "off"], default="off",
                    help="the moderate-error drive reaches goal selection: each moment's "
                         "felt value AS A GOAL is scaled by (1 + curiosity drive), so "
                         "goldilocks moments (the swinging mobile, the acting hand) win "
                         "the felt-state argmax more often. Decision-time only, no reward")
    ap.add_argument("--imagination-steps", type=int, default=None,
                    help="imagination chain depth override (yaml default 3); the "
                         "hail-mary lever against the aiming blur")
    ap.add_argument("--imagination-weight", type=float, default=None,
                    help="imagination loss weight override (yaml default 0.5)")
    ap.add_argument("--hself-slots", type=int, default=64,
                    help="entity-slot capacity of the hierarchical self tracker "
                         "(observation-only). Changing it across a resume migrates "
                         "the overlapping slots and logs the migration; the shadow "
                         "bank's distinct-patterns count is the measured basis for "
                         "sizing (2026-09-10 capacity review)")
    ap.add_argument("--entity-self", choices=["off", "dark"], default="off",
                    help="UEMA 11.4 Levels 2/3 over NAMED things - the caregiver, each toy, the "
                         "mobile, the critters, the infant's own hand in view (mimo_bridge/"
                         "entity_presence.py). Each has its own presence signal read from the "
                         "world, so a thing can be pointed at and survives the encoders changing, "
                         "which --hierarchical-self cannot. dark: it "
                         "watches and feeds nothing")
    ap.add_argument("--entity-self-teacher", choices=["wellness", "ability"], default="wellness",
                    help="what the entity levels correlate a thing's presence WITH. wellness: the "
                         "five-formula sum the body feels. ability: how many kinds of consequence "
                         "the world measured that moment - a hand contact, an object moved, a view "
                         "gained (0 to 3), which is what the project's objective asks for")
    ap.add_argument("--self-value-weight", type=float, default=0.0,
                    help="THE READER (UEMA 11.4, behind a gate). How much one unit of proven-link "
                         "times presence is worth when the planner ranks a remembered moment, "
                         "against a wellness whose five terms each run 0 to 1. 0 = the reader "
                         "watches and adds nothing. Above 0 it still adds nothing until its gate "
                         "opens: at least one entity's link must have passed the evidence test, "
                         "and the real reader must track wellness better than a SHUFFLED copy of "
                         "itself (the same link values on the wrong entities) for "
                         "--self-value-gate-windows windows in a row. A resume never inherits an "
                         "open gate")
    ap.add_argument("--self-value-level", choices=["l2", "l3"], default="l2",
                    help="which level's links the reader reads: minutes (l2) or hours (l3)")
    ap.add_argument("--self-value-gate-windows", type=int, default=3,
                    help="windows in a row the reader must beat its shuffled control before any "
                         "value is added")
    ap.add_argument("--wellness-coherence", choices=["extent", "certainty"], default="extent",
                    help="which reading of the body map the coherence term of wellness carries. "
                         "extent: the share of everything the map watches that it calls its own "
                         "(0 = none, 1 = all) - its 26 body handles AND any outside thing taken "
                         "into the self, so 28 things on the crib runs - which is what the "
                         "objective asks wellness to reward. certainty: how SURE the map is, "
                         "mean(|membership - 0.5| * 2), measured on H84 to equal 1 - 2 x the "
                         "BODY's own share (correlation -1.0000, largest gap 1.0e-05) and so paid for "
                         "the perceived self SHRINKING, while carrying 53 of every 100 parts of the "
                         "movement of wellness. Either way the number is now PER BODY; it used to "
                         "be the mean over all bodies. Recorded in the snapshot as "
                         "last_coherence_source")
    ap.add_argument("--hierarchical-self", choices=["off", "dark"], default="off",
                    help="UEMA 11.4 Levels 2/3, observation only: 64 discovered entity "
                         "slots correlate presence with wellness at minute/hour "
                         "timescales; logs membership stats, feeds NOTHING (the wire "
                         "into wellness is a later, separately certified switch)")
    ap.add_argument("--anticipation-gaze", choices=["on", "off"], default="off",
                    help="reality-graded gaze: a supervised term pulls handle "
                         "23 toward the env-measured direction of real self-caused change "
                         "(struck mobile / moving hand). Trains dwell-on-consequence")
    ap.add_argument("--planner-effort-per-body", choices=["on", "off"], default="off",
                    help="planner strength-penalty scaling. off (legacy): the penalty "
                         "sums over ALL bodies while the prediction term averages, so "
                         "8 bodies = 8x the relative effort penalty (external review "
                         "2026-09-08 finding 4) - the planner is pushed toward weaker "
                         "actions purely by body count. on: per-body sum averaged over "
                         "bodies, invariant to duplication. Flip only as a declared "
                         "attempt with criteria: it changes planner behavior.")
    ap.add_argument("--prospective-wellness", choices=["off", "add"], default="off",
                    help="add: the planner's action loop also descends weight x "
                         "(1 - predicted wellness of the imagined next state), judge = "
                         "approximate evaluator, motor token detached (2026-09-09 plan; "
                         "gate on the offline A/B probe before training with it)")
    ap.add_argument("--prospective-wellness-w", type=float, default=0.15,
                    help="weight of the prospective-wellness term (measured-basis: see "
                         "the gradient-scale probe artifact before changing)")
    ap.add_argument("--goal-gaze-hold", choices=["off", "on"], default="off",
                    help="hold the planned gaze target for the whole action interval "
                         "(goal persistence; reflex/saccades still interrupt in the env)")
    ap.add_argument("--goal-revalue", choices=["off", "on"], default="off",
                    help="re-score goal candidates at selection: touch+posture by the "
                         "current judge on stored tokens; the PRESENT body's capacity "
                         "deficit multiplies each candidate's judged capability, so a "
                         "tired body seeks remembered fresh-body states")
    ap.add_argument("--audio-freq-bands", choices=["off", "on"], default="off",
                    help="absolute-frequency band path in the audio encoder (09-09 "
                         "feasibility review: conv+global-average provably erases "
                         "frequency position - three tones at bins 40/120/200 gave "
                         "identical tokens). on adds ZERO-INIT parameters: enable at a "
                         "segment LAUNCH (fresh optimizer state), never a hot resume")
    ap.add_argument("--anticipation-gaze-w", type=float, default=0.1,
                    help="weight of the anticipation-gaze term. Default 0.1 is the UEMA "
                         "6.4 spec value; measured over H67_z (8h, 615 records) the raw "
                         "loss sat flat at median 0.148 rad^2 (a 22-degree miss; holding "
                         "the 24-degree fovea on the hand needs under ~0.044, the run's "
                         "own best records touched 0.041) while eyes-on-hand stayed "
                         "0.020-0.027 vs the 0.03 line - the pull was outranked by "
                         "vicreg 0.635 / tenure_tok 0.375 / imagination 0.289")
    ap.add_argument("--critters", choices=["on", "off"], default="off",
                    help="attention critters in the crib: a kick squeaker at the feet "
                         "(contingent hop), a hover-then-dart dragonfly (the hover is the "
                         "anticipation tell), and a bee with 20 Hz wing flicker. All "
                         "kinematic; flyers do not collide; task flags log every event")
    ap.add_argument("--ring-fov", type=float, default=0.0,
                    help="far-periphery motion ring: field of view in degrees (0 = off). "
                         "A rod-like coarse camera on the left eye read ONLY by the snap "
                         "reflex - the model never sees its pixels. Measured need: the "
                         "hands average 58 deg off-axis, inside the wide cone 10%% of the time")
    ap.add_argument("--ring-size", type=int, default=128,
                    help="ring render size in pixels (128 = 8x8 detection cells, snap lands within ~10 deg)")
    ap.add_argument("--fovea-size", type=int, default=288,
                    help="pixels per side of the narrow-field camera added inside each eye "
                         "(0 = no fovea). 288 px over 24 deg is 12 px/deg = 6 cycles/deg, the "
                         "acuity of a six-month-old; the wide 256 px over 60 deg is 4.27 px/deg")
    ap.add_argument("--fovea-fov", type=float, default=24.0,
                    help="field of view of the fovea cameras, degrees. Narrower is sharper for "
                         "the same pixels; 24 deg keeps a near object inside BOTH foveae, which "
                         "9.4 deg of disparity at 30 cm would push out of a tighter one")
    ap.add_argument("--spontaneous-saccades", choices=["on", "off"], default="on",
                    help="the brainstem's own jump-and-hold eye movement, endowed from birth "
                         "(~2 fixations/s). It is the data supply for learning what looking "
                         "does; off only for ablations")
    ap.add_argument("--touch-geometry", choices=["body", "surface"], default="body")
    ap.add_argument("--skin-profile", default="")
    ap.add_argument('--contact-profile', choices=['legacy','mechanical_v2','mechanical_v2_local','mechanical_v3','mechanical_v4'], default='legacy')
    ap.add_argument('--contact-footprint-guard', choices=['off', 'rounded_edge_v1'], default='off',
                    help="bounds on the virtual skin's per-contact footprints (mechanical_v3/v4 only). "
                         "'off' keeps the v1 footprints exactly. 'rounded_edge_v1' takes every box edge as "
                         "rounded to 1 mm: two flat faces whose overlap cannot hold a 1 mm-radius disk are "
                         "scored as the rounded edge's Hertz line contact, and no footprint's mean pressure "
                         "exceeds 10 MPa. Body 2's 71 s false pain (docs/gain_of_function_v4/agents/"
                         "body2_pain_2026-09-25.md) came from a 0.0165 mm2 flat-face sliver under 4.8 N. "
                         "A resumed world takes it only with --resume-env-settings launch")
    ap.add_argument('--rubbing-calibration', choices=['v4', 'excess_shear_v1_shadow', 'excess_shear_v1'], default='v4',
                    help="which friction work the virtual skin's pain memory stores as rubbing (mechanical_v2_local/"
                         "v3/v4). 'v4' is today's: all friction power once the shear exceeds 21.9 kPa, which counts "
                         "the creep and sliding of ordinary support (rubbing is 87-95 of every 100 parts of every "
                         "body's load). 'excess_shear_v1' stores only the work of the shear stress above the "
                         "219.4 kPa first-discomfort pressure (operator decision 2026-09-25 9:09 AM PDT; "
                         "docs/gain_of_function_v4/agents/ordinary_pain_calibration_2026-09-25.md); the peak, "
                         "exposure and touch-impairment memories are unchanged. '_shadow' keeps pain on v4 and "
                         "reports the recalibrated pain beside it. A resumed world takes it only with "
                         "--resume-env-settings launch")
    ap.add_argument('--touch-inside-rays', choices=['process', 'fixed_v1'], default='process',
                    help="the ray directions MIMo's touch uses, while each world is built, to drop skin points that "
                         "lie inside a neighbouring shape of the same body part (mimo_bridge/touch_inside_rays.py). "
                         "'process' is today's: directions drawn once per process from an unseeded generator, so "
                         "skin points lying on a seam between two shapes are live in some worlds and not in others "
                         "(0.37 in 100 draws give a different layout; one made the skin proof's second run diverge "
                         "at model step 9: docs/gain_of_function_v4/agents/training_repeatability_2026-09-26.md). "
                         "'fixed_v1': one fixed set, giving every world the usual layout (4,683 skin points at age 24 "
                         "with trunk_v1). A resumed world takes it only with --resume-env-settings launch")
    ap.add_argument('--persistent-memory', choices=['off','on'], default='off')
    # THE PERSISTENT MEMORY'S CENTRED REPAIR (lem/model/persistent_memory.py module note;
    # docs/gain_of_function_v4/agents/persistent_memory_2026-09-25.md). "off" (default) = today's model.
    # "centred": at the first lived moment the saturated memory's constant moves into ordinary biases (no
    # head's input changes), the memory reads its events and states centred, and its recurrent candidate's
    # two parts are held to fixed sizes so it can neither saturate nor latch again. Needs --persistent-memory
    # on. The switched checkpoint refuses a build without the flag.
    ap.add_argument('--persistent-memory-repair', choices=['off', 'centred'], default='off',
                    help="'centred': function-preserving repair of a saturated persistent memory; default 'off'.")
    # the folded constant's return to its pre-deploy size per sense slot after the switch (lem/model/
    # persistent_memory.py, THE CONSTANT'S RETURN): its length in model updates, read at the launch that switches
    # (later launches keep the switched checkpoint's own). NO DEFAULT with the repair on: the length must be the one
    # a training measurement set (the switch trial's return arm), or 0 to keep the constant as folded.
    ap.add_argument('--persistent-memory-constant-return', type=int, default=None,
                    help="with --persistent-memory-repair centred: the constant's return length in updates (0 = none)")
    # THE MOTOR CONTROLLER'S INPUT FOLD (lem/model/motor_input_fold.py; docs/gain_of_function_v4/agents/
    # memory_motor_layer_fall_2026-09-26.md, sections 8 and 9). "off" (default) = today's model. "on": at the first
    # training step after the repaired memory's switch the controller reads the newest tokens minus the memory's
    # constant, whose image moves into its input projection's and confidence head's biases (exact at the fold), so its
    # weights on those tokens no longer act as a bias with a 110,400-fold lever. Needs --persistent-memory-repair
    # centred; a folded checkpoint refuses a build without it.
    ap.add_argument('--motor-input-fold', choices=['off', 'on'], default='off',
                    help="'on': the motor controller reads the newest tokens without the switched memory's constant, "
                         "its image folded into the controller's biases; default 'off'.")
    # THE ASKED CHANGE AS AN INPUT OF THE MOTOR ENCODER (lem/model/motor_asked_change.py; docs/gain_of_function_v4/
    # agents/motor_encoder_asked_change_2026-10-03.md). "off" (default) = today's model, nothing built. "on": each
    # handle's asked change (command position minus achieved position, the turn from the achieved to the commanded
    # orientation, strength, radius; the eyes' row as its azimuth and elevation) enters the motor token through a
    # path with its own weights per handle, added to the motor encoder's output in units of that output's standard
    # deviation; the path's output layer starts at zero, so the first tokens are today's. A checkpoint trained with
    # it refuses a build without it. Enters the acting fingerprint only when on (acting_configuration).
    ap.add_argument('--motor-asked-change', choices=['off', 'on'], default='off',
                    help="'on': the motor encoder also reads each handle's asked change (command minus achieved "
                         "pose) through a path with its own weights per handle, zero at the start; default 'off'.")
    # AN ALL-SLOT PATH BESIDE EVERY PREDICTION HEAD'S ATTENTION (lem/model/heads_all_slot_path.py; docs/
    # gain_of_function_v4/restart_candidate/RESTART_CANDIDATE_HEADS_2026-10-04.md). "off" (default) = today's model,
    # nothing built. "on": per prediction head, the 6 newest tokens minus the persistent memory's constant, centred by
    # a running mean and LayerNormed, pass through a 2-layer network (each slot through its own weights) whose output
    # is added to the head's prediction; the output layers start at zero, so the first predictions are today's. A
    # checkpoint trained with it refuses a build without it. Enters the acting fingerprint only when on.
    ap.add_argument('--heads-all-slot-path', choices=['off', 'on'], default='off',
                    help="'on': every prediction head also reads all 6 newest tokens (memory constant removed, "
                         "centred, LayerNormed) through a path of its own, zero at the start; default 'off'.")
    # ATTENTION_CAP (lem/model/attention_cap.py; docs/gain_of_function_v4/attention_cap/ATTENTION_CAP_2026-10-05.md).
    # "off" (default) = today's model: every attention module scores keys by the dot product over the square root of
    # the head size. "on": every head of the transformer's cross-modal attention (all blocks) and of the motor
    # encoder's 2 attention layers and its pool scores a key as the bound times the cosine of its query and key, so no
    # key can take more than 1 / (1 + (n - 1) exp(-2 x bound)) of a head's weight over n keys. No parameter is added: a
    # checkpoint loads with it on or off. Both flags enter the acting fingerprint only when the cap is on
    # (acting_configuration).
    ap.add_argument('--attention-cap', choices=['off', 'on'], default='off',
                    help="'on': the transformer's cross-modal attention and the motor encoder's attention score each "
                         "key as --attention-cap-bound times the cosine of query and key; default 'off'.")
    ap.add_argument('--attention-cap-bound', type=float, default=None,
                    help="the bound B of --attention-cap on (a fixed number above 0; default: the measured bound in "
                         "lem/model/attention_cap.py MEASURED_BOUND, refused while unmeasured).")
    # THE VESTIBULAR ENCODER'S LINEAR INPUT PATH (lem/model/vestibular_input_path.py; docs/gain_of_function_v4/
    # agents/vestibular_token_2026-10-04.md). "off" (default) = today's model, nothing built. "on": one Linear(28 ->
    # 192, no bias) from the vestibular encoder's own input row is added to the encoder's last-layer output in units
    # of that output's standard deviation; its weight starts at zero, so the first tokens are today's; it learns from
    # the live row only. Needs --vestibular-sense on. A checkpoint trained with it refuses a build without it. Enters
    # the acting fingerprint only when on (acting_configuration).
    ap.add_argument('--vestibular-input-path', choices=['off', 'on', 'on_delayed_encoder_only'], default='off',
                    help="'on': the vestibular token also carries a linear map of the encoder's own 28 inputs, "
                         "through a path whose weight is zero at the start; default 'off'.")
    ap.add_argument("--contact-learning", choices=["on", "off"], default="off")
    ap.add_argument("--consequence-horizon", type=int, default=5)
    ap.add_argument("--rest-control", choices=["scheduled", "choice"], default="scheduled")
    ap.add_argument("--caregiver-mode", choices=["off", "responsive", "replay"], default="off")
    ap.add_argument('--caregiver-present-seconds',type=float,default=60.,
                    help='Interaction time after the object arrives; simulated seconds, not a learning target')
    ap.add_argument('--caregiver-absent-seconds',type=float,default=15.,
                    help='Absence time after withdrawal is complete; simulated seconds')
    ap.add_argument('--toy-hand-off', choices=['off', 'follow'], default='off',
                    help="whether the adult lets go of the toy when the infant grips it. The toy "
                         "is a MuJoCo mocap body that contact can never move, so with 'off' "
                         "nothing the infant does can carry it and a grasped toy can never join "
                         "the self (UEMA 11.2). 'follow': once two parts of one hand have "
                         "touched it for 0.1 s the toy goes where that hand goes, turning with "
                         "it, until the hand has been off it for 0.1 s; then the adult carries "
                         "it again from wherever it was left. The physical state keeps its shape, "
                         "so a run can switch this on at a resume and continue its episode "
                         "with --resume-env-settings launch"),
    ap.add_argument('--resume-env-settings', choices=['saved', 'launch'], default='saved',
                    help='Resume with saved environment settings, or explicitly migrate supported '
                         'toy-hand-off and finger-armature settings from this launch')
    ap.add_argument("--caregiver-speech-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Saved/mimo_bridge/caregiver_speech")))
    ap.add_argument("--caregiver-replay", default="")
    ap.add_argument("--caregiver-record-dir", default="",
                    help="Chunked caregiver replay tapes; empty keeps the preview-compatible in-memory tape")
    ap.add_argument('--body-layout', choices=['legacy', 'trunk_v1'], default='legacy')
    ap.add_argument("--caregiver-seconds", type=float, default=15.0,
                    help="seconds the ball may stay outside the 60 deg field before an adult "
                         "puts it back in front of the eyes (0 = no caregiver). Measured need: "
                         "the ball was within 10 deg of an eye axis on 0.88%% of body-steps")
    ap.add_argument("--context-length", type=int, default=0,
                    help="moments of memory the forward model sees (0 = lite default 4). 64 = ~3.2 s "
                         "of body time, long enough to span a reach or a crawl cycle; measured free "
                         "up to 128 because the vision encoder dominates the step")
    # MODEL SIZE. Until 2026-09-17 every one of these was a literal in lem_model.py's
    # LITE_OVERRIDES: 2 core blocks, 128 wide, 2 vision swin blocks, one attention
    # layer per sense encoder. No run recorded its depth, and a checkpoint saved at one
    # depth loaded into a model built at another with 150 missing keys and no error
    # (now refused by core_shape_mismatch). 0 keeps the lite default so an unchanged
    # launch line builds an unchanged model. Measured cost on the RTX 4070 Laptop
    # (tools/capacity_plan.py): core 8 x 256 with vision 12 x 256 and the four real
    # sense encoders at 4 x 256 is 32.3 M parameters, 6.1 GB peak, and adds 334 ms to a
    # 1,540 ms step - 21.7% slower. These flags are part of the acting configuration
    # fingerprint, so changing one on a resume is refused rather than silently applied.
    ap.add_argument("--core-blocks", type=int, default=0,
                    help="three-stage attention blocks in the core (0 = lite default 2; the full design is 8)")
    ap.add_argument("--d-model", type=int, default=0,
                    help="token width shared by every encoder output, the core and every head (0 = 128)")
    ap.add_argument("--vision-swin-layers", type=int, default=0,
                    help="swin blocks in the vision encoder (0 = lite default 2; the full design is 4)")
    ap.add_argument("--vision-width", type=int, default=0,
                    help="vision encoder internal width and angular-position hidden size (0 = 128)")
    ap.add_argument("--encoder-layers", type=int, default=0,
                    help="attention / hidden / conv layers in the tactile, motor, proprioception and "
                         "audio encoders (0 = lite default 1); intermind is left alone - it is "
                         "placeholder noise with loss weight 0.0 and nothing learns from it")
    ap.add_argument("--encoder-width", type=int, default=0,
                    help="internal width of those same four encoders (0 = 128)")
    ap.add_argument("--top-down-gate", choices=["on", "off"], default="off",
                    help="Act 3, the hill: the forward model's prediction of NOW corrects noisy senses (bounded 0.5; learning always graded on raw reality)")
    ap.add_argument("--smooth-actions", type=float, default=0.0,
                    help="0=off; 0.3 = each executed command is 30%% new, 70%% previous - motion in strokes, not twitches")
    ap.add_argument("--next-frame-loss", choices=["on", "off"], default="off",
                    help="the flip: the main lesson becomes 'produce the hidden NEXT moment' (LLM-style); current-frame reconstruction stays as a background guard")
    ap.add_argument("--movement-mref", type=float, default=12.0,
                    help="local speed (cm/s) that counts as clearly moving; set ~2x ambient jiggle so only vigorous motion feels good")
    ap.add_argument("--movement-param-mod", choices=["on", "off"], default="off",
                    help="Stage 3: stillness raises babble noise, movement nudges lr (bounded)")
    ap.add_argument("--action-repeat", type=int, default=5,
                    help="hold each command K env steps (10 ms each): a command needs time to act before it is graded")
    ap.add_argument("--self-touch-wellness", type=float, default=0.0,
                    help="how much a fully self-touching moment may add to wellness once "
                         "its gate opens (0 = the term stays exactly zero). The gate needs "
                         "the self-touch prediction to have beaten its shuffled-action "
                         "control on the newest measured window, so the weight alone can "
                         "never pay the body for pressing one hand on the other. SUPERSEDED "
                         "by --contact-head on with --contact-control-wellness, which pays only "
                         "self-touches predicted on both parts before they began; kept working "
                         "for old launch lines")
    ap.add_argument("--live-view-every", type=int, default=0,
                    help="render ONE watch-camera frame every N model steps into "
                         "<out>/live/frame.jpg, replacing what is there (0 = off). Nothing is "
                         "stored: the same file is overwritten, so a viewer reads the newest "
                         "picture and no history piles up. The cost per frame is reported as "
                         "live_view_ms in the stats")
    ap.add_argument("--live-view-body", type=int, default=0,
                    help="which body the live view watches (0 to n_envs-1)")
    ap.add_argument("--live-view-size", type=int, default=480, help="live frame pixels per side")
    ap.add_argument("--live-state-every", type=int, default=0,
                    help="publish the watched body's physical state to <out>/live/state.npz "
                         "every N model steps (0 = off). A few kB, no pictures: a viewer "
                         "process owning the same scene draws it from any camera it likes, so "
                         "this run pays only the write. Cost is live_state_ms in the stats")
    ap.add_argument("--video-every-min", type=float, default=20.0, help="record a scene video this often (0 = off)")
    ap.add_argument("--video-seconds", type=float, default=30.0, help="length of each video in SIM seconds")
    ap.add_argument("--explain-every-min", type=float, default=0.0,
                    help="write an explainer snapshot (<out>/explain/step_N.json) this "
                         "often (0 = off). Measured on the H62 checkpoint 2026-09-02: "
                         "capture 22 ms median / 32 ms p90, ~125-450 KB per snapshot - "
                         "at the recommended 5 min this is ~0.0001%% of wall time")
    ap.add_argument("--explain-steps", type=int, default=0,
                    help="write an explainer snapshot every K model steps (0 = off; the "
                         "checkpoint probe's mode - tools/explain_probe.py sets it). At "
                         "the measured 22 ms capture, K=100 costs ~0.2%% of a ~130 ms step")
    ap.add_argument("--explain-body", type=int, default=0, help="which body a snapshot records")
    ap.add_argument("--explain-keep-mb", type=float, default=500.0,
                    help="disk ceiling for <out>/explain/: newest 50 kept outright, older "
                         "history thinned evenly past this (measured 125-450 KB/snapshot "
                         "-> >=1,100 snapshots; the checkpoint dir's 1.8 GB rule's cousin)")
    # GAIN-OF-FUNCTION BUILD (docs/gain_of_function_v3/DESIGN.md, 2026-09-22). Every flag here
    # defaults to today's behaviour, so the supervisor resuming an old command line on new code
    # changes nothing; none of them is a continuation flag.
    # DEVELOPMENTAL OBSERVABILITY (addendum v4 Part 6; docs/gain_of_function_v4/OBSERVER_DESIGN.md).
    # Measurement equipment only: nothing it writes is read by the learner or this trainer.
    ap.add_argument("--observer", choices=["off", "record"], default="off",
                    help="record the developmental observer's streams: every world worker writes "
                         "the simulator's full state after each burst (body, objects, contacts, "
                         "gaze rays, per-frame transients) to <out>/observer/world/b<i>/, and this "
                         "process writes what the learner predicted per step (per-body next-moment "
                         "error, contact probabilities, each order's expected movement) to "
                         "<out>/observer/model/. The analysis runs in its own process "
                         "(python -m mimo_bridge.tools.observer). Changes no lesson, no command and "
                         "no reply to the learner (tests/test_observer_record.py)")
    ap.add_argument("--observer-chunk-steps", type=int, default=100,
                    help="model steps per observer chunk file (a crash loses at most this many)")
    ap.add_argument("--observer-images-every-min", type=float, default=10.0,
                    help="with --observer record, keep one left-eye picture per body this often, "
                         "for a human to audit the gaze geometry (0 = none); never read by the learner")
    ap.add_argument("--observer-budget-gb", type=float, default=12.0,
                    help="the most raw observer data this run holds on disk before the analysis "
                         "digests it (one sixth for the model lane, the rest split over the bodies); "
                         "at the budget the recorders drop steps and list them in gaps.jsonl")
    ap.add_argument("--observer-reserve-gb", type=float, default=12.0,
                    help="free disk space the observer never uses: below it the recorders write "
                         "nothing, so checkpoints keep their room")
    ap.add_argument("--causal-chain", choices=["off", "on"], default="off",
                    help="record the causal chain (DESIGN D13): every planning event's goal, "
                         "predicted next state and predicted wellness beside four alternatives "
                         "from the same state; every latched order's prediction before it is "
                         "sent, graded one moment later and at the order's end against what the "
                         "body produced, beside persistence. Per event to <out>/causal_chain.jsonl, "
                         "window block causal_chain in the mimo record. Adds no-grad forward "
                         "passes (cost: the causal_chain block and the deploy proof)")
    ap.add_argument("--efference-heads", choices=["off", "on"], default="off",
                    help="small heads that take the core's latent AND the executed order directly "
                         "and predict each handle's movement (cm) and turn (degrees) over the next "
                         "moment and the order's tenure, and the eyes' turn (degrees) at "
                         "50/100/150 ms, plus a gaze copy of the eye command into vision's "
                         "next-moment prediction (DESIGN D5 as amended 2026-09-23, efference_v2: "
                         "the latent centred and normalised, the asked change per handle, the "
                         "head's orientation, the gaze head's own layers). Trainable; appended "
                         "after every existing module, so a resume keeps every saved weight and "
                         "Adam moment; the gaze copy starts at zero, so switching it on changes "
                         "no prediction")
    ap.add_argument("--efference-weight", type=float, default=0.1,
                    help="loss weight of EACH efference head; each head's loss is its error in "
                         "units of the measured spread (squared to one spread, linear beyond), "
                         "summed over its graded horizons (docs/gain_of_function_v3/agents/"
                         "lane-b1-efference.md has the measured loss share at 0.1)")
    ap.add_argument("--vestibular-sense", choices=["off", "on"], default="off",
                    help="the model's sixth input slot, today the intermind placeholder (an "
                         "encoder that learned to turn its noise input into one constant token: "
                         "it moved 0.0000049 against a token size of 13.86 on ckpt_step_344400), "
                         "carries the body's vestibular / internal-state vector instead (DESIGN "
                         "D4, mimo_bridge/vestibular_sense.py): gravity and turning in the head, "
                         "the neck, which way is up for the trunk, support, effort, joint-limit "
                         "load, fatigue - 28 numbers per moment, each centred and scaled by its "
                         "running mean and spread. Same token count. A resume from an intermind "
                         "checkpoint keeps every other weight and every other Adam moment, moves "
                         "the slot's head, gate and decoder to the new name, and starts the new "
                         "encoder at the old constant, the slot's head at 'no change' and the "
                         "memory's slot inputs at 0, so the first step computes what the old "
                         "model computed. Stats block 'vestibular'")
    ap.add_argument("--vestibular-lesson-weight", type=float, default=None,
                    help="with --vestibular-sense on: the vestibular sense's weight in the "
                         "next-moment lesson, where every other sense counts 1. Default: the "
                         "measured vestibular_sense.LESSON_WEIGHT (its measurement is written "
                         "beside it)")
    ap.add_argument("--vestibular-wellness-pe", choices=["exclude", "include"], default="exclude",
                    help="with --vestibular-sense on: whether wellness's grounded prediction-error "
                         "term counts the vestibular sense's error. 'exclude' keeps the raw "
                         "wellness sum the one the old model computed (DESIGN D8); 'include' "
                         "counts it, which scores moving moments lower than resting ones through "
                         "this sense alone")
    ap.add_argument("--vestibular-encoder-repair", choices=["off", "rescale", "rescale_restart",
                                                            "rescale_restart_fp32"],
                    default="off",
                    help="with --vestibular-sense on: at the load, when the vestibular encoder's "
                         "last-layer bias has a spread above 1,000, 'rescale' divides that layer, "
                         "weight and bias, by a power of two s (to a spread of 9.2-18.3; 4,096 on H84) "
                         "and moves its Adam moments with it (first x s, second x s squared); "
                         "'rescale_restart' (the deploy's mode) divides the same and restarts the "
                         "encoder's Adam state; 'rescale_restart_fp32' also computes this encoder in "
                         "float32 whatever the run's autocast (in bf16 the revived weights' part is "
                         "rounded away beside the bias). All then restart the sense's next-moment "
                         "normalisation average, hold it out of learning progress for 2,000 steps, and "
                         "hold its next-moment term to the other senses' mean while its weight is at its "
                         "cap. "
                         "The deploy's switch copied a constant of spread 74,425 into that bias, and "
                         "the LayerNorm after it divides the weights' part of the token and every "
                         "gradient by that spread, so the token has not changed since Sept 23 "
                         "(docs/gain_of_function_v4/agents/vestibular_encoder_2026-09-25.md). Every "
                         "token stays what it was in float64, float32 and bf16 (the LayerNorm reads "
                         "y / s as it read y; a power of two changes no rounding). 'off' (default) = "
                         "today's model")
    # WELLNESS CHANNELS (DESIGN D6-D9, lem/model/wellness_channels.py). Defaults = today's goal
    # value, unchanged.
    ap.add_argument("--controllability", choices=["off", "on"], default="off",
                    help="measure, for every order that ran its full tenure, how far the body did "
                         "what was expected of it: the efference body head's expectation (or the "
                         "order's own commanded change when the head is off) against the achieved "
                         "change, direction cosine x smaller/larger size, -1 to +1; orders asking "
                         "for less than the body drifts while holding still are not counted. Stats "
                         "block 'controllability'; enters the goal value only through "
                         "--wellness-rank normalized")
    ap.add_argument("--learning-progress", choices=["off", "on"], default="off",
                    help="per body and sense, whether the next-moment error RELATIVE to predicting "
                         "no change is falling: (slow ratio - fast ratio) / slow ratio, -1 to +1. "
                         "The ratio cancels the world getting easier or harder. Stats block "
                         "'learning_progress'; enters the goal value only through --wellness-rank "
                         "normalized")
    ap.add_argument("--wellness-rank", choices=["raw", "normalized"], default="raw",
                    help="raw: the goal is the remembered moment with the largest plain sum of "
                         "wellness and drives (today). normalized: each channel is centred and "
                         "scaled by its own running statistics, weighted by priority (integrity > "
                         "controllability > learning progress > the rest > efficiency) and summed, "
                         "each channel taken as its distance from the body's own running average so "
                         "a slow drift cannot make the oldest or newest moment the goal; a moment "
                         "whose harm (skin pain or a limb at the end of its reach) is over the "
                         "measured line is chosen only when every remembered moment is over it, and "
                         "then the least harmful one. The raw sum that feeds the motor controller is "
                         "unchanged. Stats block 'wellness_rank'")
    ap.add_argument("--wellness-weights", default="",
                    help="override channel weights, 'name=w,name=w' (channels: integrity, "
                         "controllability, learning_progress, coherence, contact_control, "
                         "consequence, self_value, agency, prediction_error, tactile_comfort, "
                         "capability, movement, curiosity, efficiency)")
    ap.add_argument("--curiosity-mode", choices=["multiply", "additive"], default="multiply",
                    help="multiply: a goal moment's value x (1 + curiosity drive) (today; it decided "
                         "the goal in 5 of 5 segment 31 snapshots). additive: + the curiosity "
                         "weight (0.25) x the drive; required by --wellness-rank normalized")
    ap.add_argument("--consequence-self-contact", choices=["count", "exclude"], default="count",
                    help="count: a hand pressing anything counts as a consequence (today): the "
                         "body's own thumb and fingers included, it fired on 69.6%% of babble "
                         "frames (env-senses.md). exclude: only the part of the press that came "
                         "from outside the body counts (hand_pressing_world, 11.8%% of the same "
                         "frames)")
    # SELF-CONTACT FIRST-CLASS (DESIGN D10, lem/model/contact_head.py). Defaults = today's model.
    ap.add_argument("--contact-head", choices=["off", "on"], default="off",
                    help="a head that predicts, for each of the 53 skin patches, whether it is "
                         "touched in the next moment and whether by an own part or an outside "
                         "thing, from the core's latent AND the executed order given directly; "
                         "graded on the world's per-patch contact report, its gradient trains the "
                         "core too. Scores self-touch and outside-touch ONSETS: foreseen when the "
                         "head's probability, above the patch's base rate of starting that touch, "
                         "beat that patch's own line (crossed on 5 in 100 of its untouched "
                         "moments), both touching parts for a self-touch, beside the same with "
                         "another body's order of the same size. Appended after every existing "
                         "module, so a resume keeps every saved weight and Adam moment. Stats block "
                         "'contact_control'")
    ap.add_argument("--contact-head-weight", type=float, default=None,
                    help="loss weight of the contact head (default: the measured "
                         "lem/model/contact_head.CONTACT_HEAD_WEIGHT)")
    ap.add_argument("--contact-control-wellness", type=float, default=0.0,
                    help="weight of the wellness channel contact_control in the normalised goal "
                         "ranking (0 = off; DESIGN's table gives 0.5). It pays onsets the contact "
                         "head foresaw: 2 per self-touch foreseen on both parts, 1 per moment with a "
                         "foreseen outside touch, times 1 / (1 + onsets of that pair in the last 200 "
                         "moments); never contact, force or duration, never a pair that touches at "
                         "rest. It PAYS NOTHING until evidence exists: self-touch and outside touch "
                         "each pay only while their own gate is open, which needs 32 control moments "
                         "on which the body's own order foresaw more onsets than another body's "
                         "order, by more than 3 times the readings' own noise (shut again under 1 "
                         "time, after 200 moments with no reading, and at every load; one body has "
                         "no control, so its gates never open). The goal ranking reads each body's "
                         "running average of the payments over its last 5 orders. Needs "
                         "--contact-head on; enters the goal value only under --wellness-rank "
                         "normalized, never the raw wellness sum")
    # MOTOR SYNERGY AND THE SMOOTHER'S SOURCE (DESIGN D11, lem/model/motor_synergy.py). Defaults =
    # today's orders, unchanged.
    ap.add_argument('--hand-target-mode', choices=['world', 'wrist_relative_v1'], default='world',
                    help='Use feasible finger joint goals that follow each moving wrist.')
    # THE MOTOR INTERFACE AUDIT'S LIMB FIXES (mimo_bridge/limb_target_control.py, F2a and F2b of
    # docs/gain_of_function_v4/agents/motor_interface_audit.md). Both default to off: a resume of
    # an old command changes nothing. A resumed world takes them only with --resume-env-settings
    # launch (continuation_state.ENV_KWARGS_MIGRATABLE); the trainer refuses worlds whose setting
    # differs from the launch, in either direction.
    ap.add_argument('--parent-relative-targets', choices=['off', 'calf', 'calf_chest'], default='off',
                    help="calf: every order's calf target becomes the closest knee angle relative to the "
                         "thigh's ordered row, kept as a joint goal and re-placed on the thigh where it "
                         "is before every 10 ms frame; the ledger records the goal on the ordered thigh. "
                         "calf_chest: the chest's target likewise becomes the closest pose of its two "
                         "joints on the upper body the pelvis's ordered row implies. Replay then studies "
                         "only windows recorded under the same contract")
    ap.add_argument('--reach-projection', choices=['off', 'arms'], default='off',
                    help="arms: at each new order the arm's upper-arm, forearm and hand rows and the hand's "
                         "position become the world poses of the joint solution the decoder's own arm "
                         "solve settles on inside the joint ranges, the chest where it is; they stay world "
                         "targets for the order and the ledger records them")
    # THE OPERATOR'S DECISION OF 2026-09-25 9:09 AM PDT (audit F3 option A, reversing the tree's X7):
    # remove the head and pelvis position targets, keep their orientation targets. Default 'all' keeps
    # today's rows until a deploy; like the limb fixes it reaches a resumed world only with
    # --resume-env-settings launch, and the trainer refuses worlds that disagree with the launch.
    ap.add_argument('--position-rows', choices=['all', 'no_head_pelvis'], default='all',
                    help="all: the decoder solves a position for the wrists, the ten fingers, the head and "
                         "the pelvis (today). no_head_pelvis: not for the head and the pelvis, whose position "
                         "asks performed 1.7%% and 10.5%% and turned them off their orientation targets; "
                         "their orientation rows stay. Replay then studies only windows recorded under the "
                         "same setting")
    # THE FINGER COMMAND'S MEANING (action_decoder.py; the first-finger investigation of Oct 8, 2026, finding C).
    # Default 'off' keeps today's servo; like the limb fixes it reaches a resumed world only with
    # --resume-env-settings launch, and the trainer refuses worlds that disagree with the launch.
    ap.add_argument('--finger-joint-servo', choices=['off', 'on'], default='off',
                    help="off: the decoder servos each finger handle's pose, the fingertip bone the hand's joint "
                         "goal places on the actual arm (today). on (needs --hand-target-mode wrist_relative_v1): "
                         "each finger hinge is servoed toward its own latched goal angle, the goal the planner "
                         "search and the hand stage chose; the hinge limits, the coupling tendons, the gains, the "
                         "strength gate and the chain's one rescale are unchanged. Replay then studies only windows "
                         "recorded under the same setting")
    ap.add_argument('--physical-goals', choices=['off', 'memory'], default='off',
                    help='Use measured poses stored beside remembered goals in body-prediction action search.')
    ap.add_argument('--hand-goals', choices=['off', 'memory'], default='off',
                    help='Choose wrist-relative remembered finger shapes using the calibrated 250 ms response.')
    ap.add_argument('--hand-response-calibration', default='',
                    help='Source-bound learned hand-response coefficients, retained in model receipts.')
    # THE DECLARED REPLACEMENT OF A SAVED HAND CALIBRATION (docs/gain_of_function_v4/finger_command_interface/
    # HAND_CALIBRATION_ADMISSION_DESIGN.md, part A). Default '' keeps today's rule: a resume refuses a
    # --hand-response-calibration that differs from the one the checkpoint saved (LemLite._restore_hand_response_calibration).
    ap.add_argument('--replace-saved-hand-calibration', default='',
                    help="the closed-loop admission record (hand_calibration_admission.py compare's result) under which "
                         "a resume replaces the hand calibration the checkpoint saved with --hand-response-calibration; "
                         "refused unless the record names that calibration's payload and file bytes, the resumed "
                         "checkpoint's model weights (which the calibration's source_model_sha256 also names), this "
                         "launch's trainer code, --finger-joint-servo rule and the calibration's hand geometry, and its "
                         "decisions pass the declared line. The replacement is written into the model receipt "
                         "(hand_calibration_replacements) and the stats stream (type hand_calibration_replacement). "
                         "Default '': no replacement")
    ap.add_argument('--vision-input-calibration', default='',
                    help='Verified, source-bound visual input conditioning receipt; preserves initial predictions.')
    ap.add_argument('--vision-activation-calibration', default='',
                    help='Measured function-preserving repair of inactive eye-to-vision units.')
    add_latent_body_arguments(ap)
    ap.add_argument("--motor-synergy", choices=["off", "on"], default="off",
                    help="a small network turns a compact intention z into a coordinated change of "
                         "every body handle's target relative to where the body is (turns for the 25 "
                         "body handles, moves for the 14 whose position the decoder reads; never the "
                         "eyes). Trained on hindsight: the changes the body really made over each "
                         "order's tenure, weighted by how well the order was followed and how much "
                         "the body moved. The planner descends on z instead of 234 separate numbers. "
                         "Appended after every existing module, so a resume keeps every saved weight "
                         "and Adam moment. Stats block 'synergy'")
    ap.add_argument("--synergy-dim", type=int, default=None,
                    help="the size of z (default: the measured lem/model/motor_synergy.SYNERGY_DIM, "
                         "the principal components carrying 90%% of the achieved 250 ms posture "
                         "changes on ckpt_step_344400)")
    ap.add_argument("--synergy-planner", choices=["on", "off"], default="on",
                    help="on: plan through the motor synergy when enabled; off: keep the existing "
                         "planner while the synergy learns. Babble in the synergy has its own switch.")
    ap.add_argument("--synergy-babble", choices=["off", "on"], default="off",
                    help="babble drawn in z: one coordinated whole-body change per babble order "
                         "instead of 25 independent turns, sized every babble moment so each body "
                         "handle turns as far on average as today's wobble (0.798 rad at babble "
                         "0.05). Needs --motor-synergy on")
    ap.add_argument("--synergy-residual-l2", type=float, default=None,
                    help="give the planner a per-handle residual beside the synergy, charged this "
                         "weight x its summed square in measured spreads per body. Default: no "
                         "residual. Measured on ckpt_step_344400 (lem/model/motor_synergy."
                         "RESIDUAL_MEASUREMENT): at every weight from 0.001 to 1 the residual moved "
                         "the planner's end distance to its goal by at most 0.00005 of 0.71, because "
                         "the forward model barely depends on the order; its weight is to be "
                         "measured once that is repaired")
    ap.add_argument("--smooth-source", choices=["proposal", "executed"], default="proposal",
                    help="what --smooth-actions blends a new order with. proposal (today): a running "
                         "average of the model's own proposals, four in five of which were never "
                         "executed. executed: the order the body was really given over the interval "
                         "that just ended; the eyes' row is not blended, and the planner imagines "
                         "the gaze the eyes are sent")
    # MOTOR INTERFACE AUDIT (2026-09-25, docs/gain_of_function_v4/agents/motor_interface_audit.md
    # F1 and F4; deploy plan docs/gain_of_function_v4/agents/motor_fixes_eye_fields_2026-09-25.md).
    # Defaults = today's orders and lessons, unchanged: a supervisor resume of an old command on this
    # code changes nothing. Neither is a continuation flag; each is stamped in the model receipt when
    # on, and a resume that changes one logs which way it changed.
    ap.add_argument("--eye-row-convention", choices=["legacy", "physical"], default="legacy",
                    help="how the eyes' row of an order carries the elevation sent to the eye servo. "
                         "legacy (today): the row's pitch = plus the elevation, while proprioception "
                         "reports the eye's physical rotation, whose pitch is minus the vertical eye "
                         "angle (MIMo's vertical eye joint turns about 0 -1 0), so the motor lesson "
                         "grades every executed elevation against its mirror image and agrees only "
                         "with level gaze (3,040 recorded poses: the two readings differ by a median "
                         "12.7 degrees, twice the vertical angle). physical: the row's pitch = minus "
                         "the elevation, so the order, the sense and the servo share the eye's "
                         "physical rotation. Flips the sign of every learned relation to the ordered "
                         "eye pitch until relearned (the controller's eye output, the ledger, the "
                         "planner's gaze, the efference gaze inputs)")
    ap.add_argument("--executed-fields-only", choices=["off", "on"], default="off",
                    help="on: the order the ledger records and the planner imagines holds only what "
                         "the decoder reads - the achieved position on the 11 orientation-only body "
                         "rows and the eyes' row, radius 0 on every row - instead of 36 position "
                         "numbers and 26 radii that reach no actuator (the decoder's control is "
                         "unchanged, exactly). The motor encoder, the efference heads and the memory "
                         "then see those fields as the body's own; the stats' cmd_exec.pos_step_cm "
                         "falls to the position rows' share (the hands' pos_step_cm_hands is "
                         "unchanged) and cmd_response_pos_cos scores only the rows whose position "
                         "the decoder reads")
    # LOSS BALANCE (DESIGN D12, lem/model/next_frame_norm.py). Default = today's lesson, unchanged.
    ap.add_argument("--next-frame-norm", choices=["off", "persistence"], default="off",
                    help="off: the next-moment lesson adds up each sense's error as it comes (today; on "
                         "segment 31 audio carried 67 of every 100 parts of it, proprioception 4). "
                         "persistence: each sense's error is divided by its own running error of "
                         "predicting that nothing changes (2,000-step average) and the sum multiplied "
                         "by the mean of those divisors, so every sense is pushed to beat 'nothing "
                         "changes' equally and the lesson keeps its size; replay's re-study of the same "
                         "lesson carries the same weights. The weights reach the prediction heads and "
                         "the core; the gradient into an encoder through a head's '+ current token' "
                         "keeps weight 1 (weighted, that path taught the proprioception encoder to "
                         "report the body 6 times less within 60 steps). Needs --next-frame-loss on. "
                         "Stats block 'next_frame_norm'")
    # PROPRIOCEPTION TOKEN (mimo_bridge/proprio_token.py). Default off = today's model.
    ap.add_argument("--proprio-token-repair", choices=["off", "within_body"], default="off",
                    help="within_body (version 6): a lesson that keeps every body's proprioception token moving "
                         "with its own posture. Each body's last 1,200 raw inputs (every 5th step: 300 s, the replay "
                         "ring's span) are encoded again; a body whose response per unit of its posture movement (the "
                         "token's standard deviation over the window over the posture's) reads under "
                         "--proprio-token-deaf is pushed until it reaches --proprio-token-line, through the same "
                         "window with its velocity and capacity inputs held at their window mean (the push answers "
                         "the posture alone); from its first push on, a body whose token answers a fatigue change "
                         "more than the live bodies' largest answer at the start is pulled back to it (the hold); a "
                         "body between the two lines is left alone; the bodies of --proprio-token-skip are never "
                         "pushed. Its gradient joins "
                         "the proprioception encoder AFTER the gradient clip, at --proprio-token-strength of each "
                         "encoder tensor's running gradient size (mimo_bridge/proprio_token.py; "
                         "docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md section 45). Nothing changes at "
                         "the switch. Stats block 'proprio_token_repair'. 'off' (default) = today's model")
    ap.add_argument("--proprio-token-strength", type=float, default=None,
                    help="with --proprio-token-repair within_body: the lesson's gradient as a share of each "
                         "proprioception encoder tensor's running gradient size (Adam's second moment), "
                         "added after the clip; default proprio_token.STRENGTH (0.02), at most 0.1")
    ap.add_argument("--proprio-token-line", type=float, default=None,
                    help="with --proprio-token-repair within_body: the response per unit of posture movement every "
                         "body's token must reach over the 300-s window: the live bodies' level, the calibration "
                         "job's RULE LINE (proprio_token_measurements.py calibrate, on the saved rings re-encoded by "
                         "each checkpoint's own encoder). Unmeasured until then: the lesson refuses to start without "
                         "it")
    ap.add_argument("--proprio-token-deaf", type=float, default=None,
                    help="with --proprio-token-repair within_body: the response per unit of posture movement under "
                         "which a body is engaged (pushed until it reaches --proprio-token-line): the operator's "
                         "deaf/live line, the calibration job's RULE DEAF in the lesson's units (the lines file's "
                         "DEAF). Unmeasured until then: the lesson refuses to start without it; it must lie under the "
                         "line")
    ap.add_argument("--proprio-token-skip", default=None,
                    help="with --proprio-token-repair within_body: the bodies, numbered from 1 as the dashboard "
                         "numbers them (\"3\" or \"3,5\"), the lesson never pushes; on H84 body 3, which the operator "
                         "excludes from conclusions about learning. Default: none")
    ap.add_argument("--proprio-heading", choices=["off", "free"], default="off",
                    help="free: every proprioceptive row is turned by minus the body's heading (the lower "
                         "trunk's rotation about the vertical) before the encoder reads it, so the direction "
                         "the body lies in the crib no longer changes the input. It changes what the encoder "
                         "reads: the token changes at the switch for every body whose heading is not 0 and is "
                         "relearned (mimo_bridge/proprio_heading.py; the cost in "
                         "docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md). A checkpoint trained "
                         "with it refuses a build without it. 'off' (default) = today's world-frame input")
    # LIVENESS (DESIGN section 10, lem/model/liveness.py). Default off = today's model.
    ap.add_argument("--liveness-repair", choices=["off", "report", "on"], default="off",
                    help="report: every few steps, record which units of every hidden layer respond "
                         "(input above -3) and how evenly every attention module spreads, into the "
                         "stats block 'liveness'; read-only. on: also revive, once, the units that "
                         "never responded in the nine layers the 2026-09-22 sweep found dead (their "
                         "output unchanged at the switch; they then read their input centred and "
                         "normalised), and give the motor and touch encoders' pools a path every "
                         "handle and every contact row reaches (starting at zero). Appended after "
                         "every existing module, so a resume keeps every saved weight and Adam moment")
    ap.add_argument("--liveness-rule", choices=["live_line", "varying", "none"], default="live_line",
                    help="which units --liveness-repair on revives at the end of each watching window. "
                         "live_line (default, the rule before 2026-09-26, bit for bit): units whose input "
                         "never rose above -3. varying: also units whose output did not vary over the "
                         "window (range at most one bf16 step at the unit's own size, the stats block's "
                         "layers_varying measure), so a unit held at one active value is revived. none: "
                         "no unit is revived; every saved repair part (the revived units with their "
                         "centres, the all-element paths) loads and trains as in mode on, and the stats "
                         "block keeps reporting (a checkpoint trained in mode on cannot resume under "
                         "--liveness-repair report); lem/model/liveness.py WHICH UNITS THE REVIVAL CHOOSES")
    ap.add_argument("--resume", default=None, help="snapshot folder (ckpt_step_N) to continue from: weights, optimizer, step count, curriculum phase")
    return ap.parse_args(argv)


def _discrimination(ring: List[np.ndarray]) -> Optional[float]:
    """Plan metric 3. Each body lives in its own situation (own target, own pose), so a model
    that reads its senses should command DIFFERENT things per body. Ratio = variance of the
    per-body mean action / mean variance of each body's action over time. > 1.5 = the model
    clearly tells situations apart; ~1 = it commands the same thing everywhere."""
    if len(ring) < 20:
        return None
    x = np.stack(ring)                       # [T, B, D]
    between = x.mean(axis=0).var(axis=0).mean()      # variance across bodies of the time-mean
    within = x.var(axis=0).mean()                    # variance over time, averaged
    return float(between / max(within, 1e-9))


def _discrimination_parts(ring: List[np.ndarray]):
    """The ratio's two halves, reported separately.

    The ratio alone is ambiguous: it fell 1.51 -> 0.17 between H15 and H17 at the same time
    as body speed rose 54%, and faster bodies inflate the denominator on their own. Without
    the halves there is no way to tell "the bodies stopped telling situations apart" from
    "the bodies are simply moving more"."""
    if len(ring) < 20:
        return None, None
    x = np.stack(ring)
    return float(x.mean(axis=0).var(axis=0).mean()), float(x.var(axis=0).mean())


def switch_stored_eye_rows(model, prev_executed):
    """At a launch that changed --eye-row-convention (model.motor_interface_change names it): rewrite
    every stored ledger eye row in the new convention - the learner's stores
    (LemLite.convert_stored_eye_rows) and the trainer's ledger of the transition that ended at the
    checkpoint. Returns (prev_executed, {store: rows rewritten}); (prev_executed, None) when the
    convention did not change, which leaves everything exactly as it was."""
    if "eye_row_convention" not in (getattr(model, "motor_interface_change", None) or {}):
        return prev_executed, None
    done = model.convert_stored_eye_rows()
    if prev_executed is not None:
        prev_executed = LemLite.mirror_eye_rows(prev_executed, model.GAZE_IDX)
        done["trainer.prev_executed"] = 1
    conv = getattr(model, "motor_interface_conversion", None)
    if isinstance(conv, dict):
        conv["stored_eye_rows_rewritten"] = done
        conv["stored_eye_rows_readers"] = {k: LemLite.STORED_EYE_ROW_READERS.get(k, "unlisted") for k in done}
        conv["encoded_eye_rows_not_rewritten"] = dict(LemLite.ENCODED_EYE_ROW_CARRIERS)
    return prev_executed, done


def eye_replay_contract_record(model):
    """What replay studies after an eye row convention switch (the fourth review's fix round, the coordinator's
    decision): the action contract its windows must carry (limb_target_control.replay_contract with the eye row's
    bit under 'physical'), and how many of the ring's stored moments carry it now (none right after the switch:
    replay studies nothing until a whole window of moments recorded after the switch exists)."""
    from mimo_bridge.limb_target_control import replay_contract, EYE_ROW_CONTRACT_BIT
    rp = getattr(model, "replay", None)
    need = replay_contract(getattr(model, "hand_target_mode", "world"),
                           getattr(model, "parent_relative_targets", "off"),
                           getattr(model, "position_rows", "all"),
                           getattr(model, "eye_row_convention", "legacy"),
                           getattr(model, "finger_joint_servo", "off"))
    out = dict(required=None if need is None else int(need), eye_row_bit=int(EYE_ROW_CONTRACT_BIT),
               eye_row_convention=getattr(model, "eye_row_convention", "legacy"))
    if rp is not None:
        n = int(rp.count if rp.count < rp.size else rp.size)
        carries = (rp.action_contract[:, :n].to(torch.int64) == int(need)) if need is not None else None
        out.update(stored_moments=n * int(rp.n_envs),
                   stored_moments_under_it=None if carries is None else int((carries & rp.has_executed[:, :n]).sum()),
                   window_moments=int(model.ctx_len) + 1,
                   meaning=("replay samples windows of %d consecutive moments whose %d step-from moments all carry "
                            "the required contract; until such a window exists it studies nothing (the live "
                            "lessons go on)" % (int(model.ctx_len) + 1, int(model.ctx_len))))
    return out


def same_hand_goal(canonical, requested):
    """Compare physical poses; opposite quaternion signs name the same rotation."""
    a,b=np.asarray(canonical),np.asarray(requested)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():return False
    qa,qb=a[...,3:7],b[...,3:7]
    na,nb=np.linalg.norm(qa,axis=-1),np.linalg.norm(qb,axis=-1)
    if (na<1e-8).any() or (nb<1e-8).any():return False
    agreement=np.abs((qa*qb).sum(-1)/(na*nb))
    return bool(np.allclose(a[...,:3],b[...,:3],rtol=1e-5,atol=2e-5)
                and np.allclose(agreement,1.,rtol=0,atol=2e-6)
                and np.allclose(a[...,7:],b[...,7:],rtol=1e-5,atol=2e-6))


def cmd_exec_sample(held, achieved, handle_names) -> dict:
    """The size of one latched order, for the stats' cmd_exec block.

    held [B, H, 9] is the order latched for the body; achieved [B, H, 10] is the pose the model
    last read (positions in units of COMMAND_POSITION_RANGE_CM), or None. Returns the mean ordered
    strength (0 = limp, 1 = full) and the mean distance in cm between each part's ordered position
    and where it is now, over the body parts and over the hands; the two distances are None when
    no pose is given.

    The eyes' row is left out by its NAME ('gaze'), as the stillness gauge in main() and the
    planner's plan0 record (lem_model.py) do: the decoder gives that row an empty joint chain, so
    its position and strength reach no actuator. It is row 23 in both layouts; the last row is the
    chest in trunk_v1. Until 2026-09-23 the eyes' row was averaged in (segment 31's 16 stored
    rows: strength 0.4864 with it against 0.4876 without it, position step 30.13 cm against
    30.23 cm). Hands = rows 0-11, both wrists and all ten fingers, in both layouts."""
    hm = np.asarray(held, dtype=np.float64)
    body = [i for i, n in enumerate(handle_names) if n != 'gaze' and i < hm.shape[1]]
    out = {"strength": float(hm[:, body, 8].mean()),
           "strength_hands": float(hm[:, :12, 8].mean()),
           "step_cm": None, "step_cm_hands": None}
    if achieved is not None:
        ac = np.asarray(achieved, dtype=np.float64)
        nh = min(hm.shape[1], ac.shape[1])
        d = np.linalg.norm(hm[:, :nh, 0:3] - ac[:, :nh, 0:3] * COMMAND_POSITION_RANGE_CM, axis=-1)
        out["step_cm"] = float(d[:, [i for i in body if i < nh]].mean())
        out["step_cm_hands"] = float(d[:, :min(12, nh)].mean())
    return out


ENV_DT_ASSUMED_S: float = 0.005 * 2
"""One env step in the body's own time, assumed before a body exists to measure.

The crib scene's MuJoCo timestep is 0.005 s and BridgeEnv builds the env at frame_skip
2, so one env step is 0.005 x 2 = 0.010000 s and one model step at --action-repeat 5 is
0.050000 s - exact, not approximate. Every seconds-denominated number in the run (the
bout and rest lengths, one transition's duration, the context window in seconds)
multiplies it. The authority is env.dt, which only exists once a body is running:
main() reads frames[0].dt, records it as env_dt_s in the model receipt and in the type
'mimo' record, and passes it to _transition_seconds. Callers that run before the first
body exists (the self-choice receipt check) keep this assumed value, and main() logs a
warning naming both numbers when the measured dt differs."""


def _transition_seconds(args, env_dt: float = ENV_DT_ASSUMED_S) -> float:
    """How long one model transition lasts in the body's own time.

    env_dt is the MEASURED env step (frames[0].dt) wherever a body exists; the default
    is the scene's own 0.005 x 2 and gives the identical number for the crib scene."""
    return float(env_dt) * max(1, int(args.action_repeat or 1))


def curiosity_used_fallback(res) -> bool:
    """True when this step's curiosity drive came from the CURRENT-frame reconstruction
    error instead of the delayed causal error.

    lem_model sets causal_pe to NaN on exactly the steps where the delayed lesson had no
    entry and the drive fell back to recon_mean_pe. The trainer's causal_pe window mean
    skips those steps while its curiosity_drive window mean keeps them, so the two
    numbers printed side by side are means over different sets of steps; counting the
    fallback steps is what makes the difference readable from the record."""
    cp = getattr(res, "causal_pe", None)
    return cp is None or cp != cp


def bind_schedule_to_config(model, cm, steps_per_phase: int, global_step: int) -> None:
    """Point model.dev_limits(step) at the phase the CurriculumManager is actually in.

    Two readers answer "which phase is this run in": the CurriculumManager the trainer
    reads every step, and harness._get_developmental_phase behind LemLite.dev_limits(),
    which run_mode_a and run_mode_b read every step. The manager counts inside its phase;
    the fallback divides a step count by steps_per_phase. Writing the LAUNCH FLAG and the
    absolute global step left them disagreeing after any resume - a checkpoint saved at
    'child' resumed at global step 3,500,000 with steps_per_phase 3000 gave the manager
    'child' (strength limit 0.90) against the fallback's 'mature' (strength limit 1.00),
    whether or not --start-phase matched the saved phase. Writing the manager's own phase
    and the global step its phase began at makes the fallback return the manager's answer
    at that step and at every step after it."""
    tr = model.config.setdefault("training", {})
    tr["steps_per_phase"] = int(steps_per_phase)
    tr["start_phase"] = str(cm.phase_name)
    tr["phase_start_global"] = int(global_step) - int(cm.phase_step)


def rebuild_changes(args, phase: Optional[str], out_dir: str, kw: dict):
    """Which env kwargs this phase asks for that differ from the bodies now running.

    An EMPTY dict means the bodies are NOT rebuilt, and therefore that the context
    buffer, the pending transitions, the goal ring and the replay episode boundary all
    survive the phase change untouched while rom_limit and strength_limit change under
    them. Only age, sensory_delay, motor_delay and scene enter this comparison, and
    env_kwargs_for sets age only under --age-binding curriculum or fixed:, the delays
    only under --delay-binding on, and the scene only under --scene-binding phase, so
    with all three off the dict is empty at every transition."""
    new_kw = env_kwargs_for(args, phase, out_dir)
    changes = {k: new_kw.get(k) for k in ("age", "sensory_delay", "motor_delay", "scene")
               if new_kw.get(k) != kw.get(k)}
    return changes, new_kw


def phase_transition_record(transition_log: Optional[dict], step: int,
                            changes: dict, timestamp: float) -> dict:
    """The type 'phase_transition' line for replay_training_stats.jsonl.

    `rebuilt` is the field the stream previously had no way to carry: the record was
    written before the rebuild decision was even computed, so a reader saw a clean
    transition whether the bodies were rebuilt and the carried state dropped or nothing
    happened at all. The only line that said which was log.info("rebuilding bodies with
    %s") in the text log."""
    tl = dict(transition_log or {})
    tl.update({"type": "phase_transition", "timestamp": timestamp, "step": int(step),
               "changes": dict(changes), "rebuilt": bool(changes)})
    return tl


def explain_rest_fields(build_snapshot_fn, order_sent, rest_window: bool) -> dict:
    """The rest-substitution fields for an explainer snapshot, or {} on an explainer
    that predates them.

    On a scheduled rest window the body executes send_motor, whose position and rotation
    columns were replaced by the body's own achieved pose, while held_order is the order
    the model proposed. The snapshot's own prev_executed row is already post-substitution,
    so without order_sent and rest_window the two rows of one table sit on opposite sides
    of the substitution with no field saying so. The signature check keeps the trainer
    working against an explainer module that has not yet grown the two parameters -
    build_snapshot is called inside a try/except that logs and continues, so an unexpected
    keyword would turn the whole snapshot off silently."""
    import inspect
    try:
        params = inspect.signature(build_snapshot_fn).parameters
    except (TypeError, ValueError):
        return {}
    if "order_sent" not in params or "rest_window" not in params:
        return {}
    return {"order_sent": order_sent, "rest_window": bool(rest_window)}


def env_kwargs_for(args, phase: Optional[str], out_dir: str) -> dict:
    scene = PHASE_SCENE.get(phase, args.scene) if (args.scene_binding == "phase" and phase) else args.scene
    kw = dict(scene=scene, run_dir=os.path.join(out_dir, "scenes"), vision=not args.no_vision,
              render_every=args.render_every, touch_scale_mult=args.touch_scale_mult, seed=args.seed, tag=args.tag,
              kp=args.decoder_kp, joint_damping_scale=args.joint_damping_scale, hinge_armature=args.hinge_armature,
              finger_armature=getattr(args, 'finger_armature', 0.0004),
              hand_target_mode=getattr(args, 'hand_target_mode', 'world'),
              fovea_size=args.fovea_size, fovea_fov=args.fovea_fov,
              caregiver_seconds=args.caregiver_seconds,
              skin_profile=getattr(args, "skin_profile", ""),
              contact_profile=getattr(args, 'contact_profile', 'legacy'),
              body_function=getattr(args, 'body_function', 'off'),
              touch_geometry=getattr(args, "touch_geometry", "body"),
              caregiver_mode=getattr(args, "caregiver_mode", "off"),
              caregiver_speech_dir=getattr(args, "caregiver_speech_dir", ""),
              caregiver_replay=getattr(args, "caregiver_replay", ""),
              caregiver_record_dir=getattr(args, "caregiver_record_dir", ""),
              caregiver_present_seconds=getattr(args,'caregiver_present_seconds',60.),
              caregiver_absent_seconds=getattr(args,'caregiver_absent_seconds',15.),
              toy_hand_off=getattr(args, 'toy_hand_off', 'off'),
              body_layout=getattr(args, 'body_layout', 'legacy'),
              gaze_reflex=(args.gaze_reflex == "on"),
              spontaneous_saccades=args.spontaneous_saccades != "off",
              cmd_horizon=max(1, args.action_repeat * max(1, args.motor_hold)),
              mobile=args.mobile == "on",
              ring_fov=args.ring_fov, ring_size=args.ring_size,
              critters=args.critters == "on",
              audio_real=args.audio == "real",
              gaze_kp=args.gaze_kp, gaze_kd=args.gaze_kd)
    if args.age_binding == "curriculum" and phase is not None:
        kw["age"] = PHASE_AGE[phase]
    elif args.age_binding.startswith("fixed:"):
        kw["age"] = float(args.age_binding.split(":", 1)[1])
    if args.delay_binding == "on" and phase is not None:
        kw["sensory_delay"], kw["motor_delay"] = PHASE_DELAYS[phase]
    # the motor interface fixes' settings only when set (limb_target_control.py, action_decoder.
    # position_handles): at their defaults, which are BridgeEnv's own, the env settings stay exactly the
    # dict every earlier launch built, so an episode saved on the defaults resumes on code that predates
    # the settings (continuation_state._ABSENT_IS_DEFAULT keeps a resume from writing them either)
    for name, default in LIMB_ENV_DEFAULTS.items():
        value = getattr(args, name, default) or default
        if value != default:
            kw[name] = value
    # the skin fixes' settings, named only when on, for the same reason (continuation_state._ABSENT_IS_DEFAULT)
    if getattr(args, 'contact_footprint_guard', 'off') != 'off':
        kw["contact_footprint_guard"] = args.contact_footprint_guard
    if getattr(args, 'rubbing_calibration', 'v4') != 'v4':
        kw["rubbing_calibration"] = args.rubbing_calibration
    # the touch layout's repeatability switch, named only when on, for the same reason
    if getattr(args, 'touch_inside_rays', 'process') != 'process':
        kw["touch_inside_rays"] = args.touch_inside_rays
    # the finger command's meaning (--finger-joint-servo, action_decoder.py), named only when on, for the same reason
    if (getattr(args, 'finger_joint_servo', 'off') or 'off') != 'off':
        kw["finger_joint_servo"] = args.finger_joint_servo
    return kw


# the motor interface fixes' world settings and their defaults (BridgeEnv's; today's behaviour)
LIMB_ENV_DEFAULTS = {'parent_relative_targets': 'off', 'reach_projection': 'off', 'position_rows': 'all'}


def _deep_merge_dicts(a: dict, b: dict) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = _deep_merge_dicts(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def size_overrides(args) -> dict:
    """Config overrides for the model-size flags; empty when every flag is 0.

    Pure, so it is tested without argparse or a GPU. Mirrors the shapes
    tools/capacity_plan.py measured: tactile and motor take n_cross_attn_layers plus
    d_internal and per_patch_hidden; audio takes d_internal and a conv stack of
    2 + layers channels; proprioception takes d_internal and 1 + layers hidden
    layers; intermind is never touched here.
    """
    ov: dict = {}
    model: dict = {}
    if getattr(args, "core_blocks", 0):
        model["forward_model"] = {"n_layers": int(args.core_blocks)}
    if getattr(args, "d_model", 0):
        model["d_model"] = int(args.d_model)
    if model:
        ov["model"] = model
    mods: dict = {}
    vl, vw = getattr(args, "vision_swin_layers", 0), getattr(args, "vision_width", 0)
    if vl or vw:
        enc = {}
        if vl:
            enc["n_swin_layers"] = int(vl)
        if vw:
            enc["d_internal"] = int(vw)
            enc["angular_pos_hidden"] = int(vw)
        mods["vision"] = {"encoder": enc}
    el, ew = getattr(args, "encoder_layers", 0), getattr(args, "encoder_width", 0)
    if el or ew:
        L = int(el) if el else 1
        for name in ("tactile", "motor"):
            enc = {}
            if el:
                enc["n_cross_attn_layers"] = L
            if ew:
                enc["d_internal"] = int(ew)
                enc["per_patch_hidden"] = int(ew)
            mods[name] = {"encoder": enc}
        audio = {}
        if ew:
            audio["d_internal"] = int(ew)
        if el or ew:
            audio["conv_channels"] = [int(ew) if ew else 128] * (2 + L)
        mods["audio"] = {"encoder": audio}
        prop = {}
        if ew:
            prop["d_internal"] = int(ew)
        prop["hidden_dims"] = [int(ew) if ew else 128] * (1 + L)
        mods["proprioception"] = {"encoder": prop}
    if mods:
        ov["modalities"] = mods
    return ov


def auto_n_envs() -> int:
    """As many bodies as the machine can carry: leave 4 CPU threads for the model process and
    the OS, and keep every body (about 0.7 GB each) plus the model (2 GB) inside 60 percent of
    free RAM. Capped at 24 (beyond that the pipes to the workers cost more than the bodies)."""
    import psutil
    cpus = os.cpu_count() or 4
    free_gb = psutil.virtual_memory().available / 1e9
    by_cpu = max(1, cpus - 4)
    by_ram = max(1, int((free_gb * 0.6 - 2.0) / 0.7))
    return int(max(1, min(24, by_cpu, by_ram)))


def self_state_report(model):
    """The stats record's self_state block, then the next report's per-step counts start at zero.

    The block is the last step's self-state stats; its window_* fields count every step since the
    previous record (self_state_integration.new_window says why)."""
    record = dict(model.last_self_state)
    if getattr(model, 'self_state', None) is not None:
        from .self_state_integration import close_window
        close_window(model)
    return record


def build_learner(args, ckpt_dir):
    """Canonical learner construction shared by production and copied experiments."""
    imagination = None if args.imagination == "config" else (args.imagination == "on")
    _ov = {}
    if args.context_length:
        _ov = {"model": {"context_length": args.context_length}}
    _ov = _deep_merge_dicts(_ov, size_overrides(args))
    if args.curiosity_band:
        lo, hi = (float(x) for x in args.curiosity_band.split(":"))
        _ov.setdefault("training", {})["curiosity"] = {"pe_min": lo, "pe_max": hi}
    if args.placeholder_senses == "off":
        _ov.setdefault("modalities", {})
        for _m in ("audio", "intermind"):
            _ov["modalities"].setdefault(_m, {})["enabled"] = False
    if args.audio == "real":
        # the env writes the avatar's hearing contract into the frame bytes; audio
        # becomes a real sense (no placeholder generator, lessons engage via the
        # source mask) at the yaml's own loss weight
        _ov.setdefault("modalities", {}).setdefault("audio", {})["data_source"] = "real"
    if args.audio_freq_bands == "on":
        _ov.setdefault("modalities", {}).setdefault("audio", {}) \
           .setdefault("encoder", {})["freq_band_path"] = "on"
    # hail-mary knobs (operator doctrine 2026-09-06): the imagination chain's depth
    # and training weight, config-routed like every other lesson setting. Defaults
    # None = the yaml stands (steps 3, weight 0.5).
    if args.imagination_steps is not None:
        _ov.setdefault("training", {}).setdefault("imagination", {})["steps"] = int(args.imagination_steps)
    if args.imagination_weight is not None:
        _ov.setdefault("training", {}).setdefault("imagination", {})["loss_weight"] = float(args.imagination_weight)
    if args.residual_prediction == "on":
        # the heads predict the CHANGE and the current token is added. Every site that
        # turns last_step into a prediction must pass the current tokens or the heads
        # raise, so this cannot be half-applied - the failure mode that cost runs twice.
        _ov = dict(_ov or {})
        _ov["model"] = dict(_ov.get("model") or {}, residual_prediction=True)
    learner = LemLite(contact_learning=args.contact_learning == "on",
                    contact_profile=args.contact_profile,
                    persistent_memory=args.persistent_memory,
                    self_state=args.self_state,
                    self_prediction=getattr(args,'self_prediction','legacy'),
                    self_percept_release=getattr(args,'self_percept_release','off'),
                    proprio_token_repair=getattr(args, "proprio_token_repair", "off") or "off",
                    proprio_token_strength=getattr(args, "proprio_token_strength", None),
                    proprio_token_line=getattr(args, "proprio_token_line", None),
                    proprio_token_skip=getattr(args, "proprio_token_skip", None),
                    proprio_token_deaf=getattr(args, "proprio_token_deaf", None),
                    proprio_heading=getattr(args, "proprio_heading", "off") or "off",
                    self_choice=args.self_choice,self_choice_receipt=args.self_choice_receipt,
                    body_layout=args.body_layout,
                    consequence_horizon=args.consequence_horizon, device=args.device, no_vision=args.no_vision, wellness_mode=args.wellness_mode,
                    motor_state_formula=args.motor_state_formula,
                    movement_bonus=args.movement_bonus, movement_penalty=args.movement_penalty,
                    body_schema_updates=args.body_schema_updates == "on",
                    curiosity_goal=args.curiosity_goal == "on",
                    goal_locality=args.goal_locality == "on",
                    anticipation_gaze=args.anticipation_gaze == "on",
                    anticipation_gaze_w=args.anticipation_gaze_w,
                    planner_effort_per_body=args.planner_effort_per_body == "on",
                    consequence_weight=args.consequence_w,
                    wellness_coherence=args.wellness_coherence,
                    self_touch_wellness=args.self_touch_wellness,
                    entity_self=args.entity_self, entity_self_teacher=args.entity_self_teacher,
                    self_value_weight=args.self_value_weight,
                    self_value_level=args.self_value_level,
                    self_value_gate_windows=args.self_value_gate_windows,
                    refresh_action_calibration=args.refresh_action_calibration,
                    prospective_wellness=args.prospective_wellness,
                    prospective_wellness_w=args.prospective_wellness_w,
                    goal_gaze_hold=args.goal_gaze_hold,
                    goal_revalue=args.goal_revalue,
                    hierarchical_self=args.hierarchical_self,
                    hself_slots=args.hself_slots,
                    overrides=_ov or None,
                    think_harder=args.think_harder == "on", intention=args.intention == "on",
                    intention_every=args.intention_every,
                    intention_execute=not args.no_intention_execute,
                    action_tenure_loss=args.action_tenure_loss == "on",
                    action_tenure_weight=args.action_tenure_weight,
                    action_tenure_alpha=args.action_tenure_alpha,
                    action_tenure_token_reach=args.action_tenure_token_reach == "on",
                    action_tenure_token_weight=args.action_tenure_token_weight,
                    action_tenure_token_alpha=args.action_tenure_token_alpha,
                    agency_drive=args.agency_drive == "on",
                    agency_weight=args.agency_weight,
                    agency_prior=args.agency_prior,
                    plan_horizon=args.plan_horizon,
                    vision_gate=args.vision_gate,
                    movement_drive=args.movement_drive == "on",
                    movement_sensor_mod=args.movement_sensor_mod == "on",
                    movement_param_mod=args.movement_param_mod == "on",
                    movement_mref=args.movement_mref,
                    next_frame_loss=args.next_frame_loss == "on",
                    smooth_actions=args.smooth_actions,
                    top_down_gate=args.top_down_gate == "on",
                    # getattr: tools build learners from args.json files written before
                    # these flags existed
                    efference_heads=getattr(args, "efference_heads", "off") == "on",
                    efference_weight=float(getattr(args, "efference_weight", 0.1)),
                    efference_tenure_steps=max(1, int(getattr(args, "motor_hold", 5) or 1)),
                    causal_chain=getattr(args, "causal_chain", "off") == "on",
                    vestibular_sense=getattr(args, "vestibular_sense", "off") == "on",
                    vestibular_lesson_weight=getattr(args, "vestibular_lesson_weight", None),
                    vestibular_wellness_pe=getattr(args, "vestibular_wellness_pe", "exclude"),
                    vestibular_encoder_repair=getattr(args, "vestibular_encoder_repair", "off") or "off",
                    controllability=getattr(args, "controllability", "off") == "on",
                    learning_progress=getattr(args, "learning_progress", "off") == "on",
                    wellness_rank=getattr(args, "wellness_rank", "raw"),
                    wellness_weights=getattr(args, "wellness_weights", "") or None,
                    curiosity_mode=getattr(args, "curiosity_mode", "multiply"),
                    consequence_self_contact=getattr(args, "consequence_self_contact", "count"),
                    contact_head=getattr(args, "contact_head", "off") == "on",
                    contact_head_weight=getattr(args, "contact_head_weight", None),
                    contact_control_wellness=float(getattr(args, "contact_control_wellness", 0.0) or 0.0),
                    motor_synergy=getattr(args, "motor_synergy", "off") == "on",
                    synergy_dim=getattr(args, "synergy_dim", None),
                    synergy_babble=getattr(args, "synergy_babble", "off") == "on",
                    synergy_planner=getattr(args, "synergy_planner", "on") == "on",
                    synergy_residual_l2=getattr(args, "synergy_residual_l2", None),
                    synergy_tenure_steps=max(1, int(getattr(args, "motor_hold", 5) or 1)),
                    smooth_source=getattr(args, "smooth_source", "proposal") or "proposal",
                    next_frame_norm=getattr(args, "next_frame_norm", "off") or "off",
                    liveness_repair=getattr(args, "liveness_repair", "off") or "off",
                    liveness_rule=getattr(args, "liveness_rule", "live_line") or "live_line",
                    physical_goals=getattr(args, 'physical_goals', 'off'),
                    hand_target_mode=getattr(args, 'hand_target_mode', 'world'),
                    hand_goals=getattr(args, 'hand_goals', 'off'),
                    # getattr: args.json files written before these two flags build the old model
                    eye_row_convention=getattr(args, 'eye_row_convention', 'legacy') or 'legacy',
                    executed_fields_only=getattr(args, 'executed_fields_only', 'off') or 'off',
                    persistent_memory_repair=getattr(args, 'persistent_memory_repair', 'off') or 'off',
                    persistent_memory_constant_return=int(getattr(args, 'persistent_memory_constant_return', 0) or 0),
                    motor_input_fold=getattr(args, 'motor_input_fold', 'off') or 'off',
                    # getattr: args.json files written before the flag build the model without the path
                    motor_asked_change=getattr(args, 'motor_asked_change', 'off') or 'off',
                    # getattr: args.json files written before the flag build the model without the paths
                    heads_all_slot_path=getattr(args, 'heads_all_slot_path', 'off') or 'off',
                    # getattr: args.json files written before the flags build the model without the cap
                    attention_cap=getattr(args, 'attention_cap', 'off') or 'off',
                    attention_cap_bound=getattr(args, 'attention_cap_bound', None),
                    vestibular_input_path=getattr(args, 'vestibular_input_path', 'off') or 'off',
                    imagination=imagination, seed=args.seed, checkpoint_dir=ckpt_dir)
    # which contracts replay studies (limb_target_control.replay_contract); 'off' and 'all' = today's
    learner.parent_relative_targets = getattr(args, 'parent_relative_targets', 'off')
    learner.position_rows = getattr(args, 'position_rows', 'all')
    learner.finger_joint_servo = getattr(args, 'finger_joint_servo', 'off') or 'off'
    hand_calibration = getattr(args, 'hand_response_calibration', '')
    if hand_calibration:
        learner.configure_hand_response_calibration(hand_calibration)
    # --replace-saved-hand-calibration: set on the learner only when given, so a launch without it builds the learner
    # production builds (a PUBLIC name: the episode companion never carries it across a resume)
    replacement = getattr(args, 'replace_saved_hand_calibration', '') or ''
    if replacement:
        learner.replace_saved_hand_calibration = str(replacement)
    calibration = getattr(args, 'vision_input_calibration', '')
    if calibration:
        learner.configure_visual_input_calibration(calibration)
    activation = getattr(args, 'vision_activation_calibration', '')
    if activation:
        learner.configure_visual_activation_calibration(activation)
    return learner


def acting_configuration(args):
    """Bind qualification to sensory, motor, planning and model configuration.

    Output locations, reporting cadence and learner-only replay scheduling can
    change without changing the frozen actor's behavior. New flags are included
    by default so an unreviewed execution change cannot silently inherit a pass.
    """
    operational={'out','resume','minutes','max_steps','tag','report_every','checkpoint_every_min',
        'video_every_min','video_seconds','explain_every_min','explain_steps','explain_body','explain_keep_mb',
        'probe_touch','probe_every','allow_preflight_fail','reinit','fresh_optimizer','fill_ring_first',
        'replay','replay_every','replay_batch','replay_size','eval_only','self_choice','self_choice_receipt',
        'caregiver_record_dir','checkpoint_storage','isolated_native_proof','episode_continuation',
        'retire_acting_snapshot','refresh_action_calibration',
        # instrumentation only: no-grad predictions and a record, no change to any order
        'causal_chain', 'latent_body_hierarchy', 'latent_body_levels', 'latent_planner',
        'latent_body_config', 'latent_body_output', 'latent_body_resume', 'latent_body_goal',
        'latent_body_candidates', 'latent_body_radius',
        # (Oct 1, 2026) how often the shadow's own latest.pt is written: no order changes
        'latent_body_save_every'}
    configuration={k:v for k,v in vars(args).items() if k not in operational}
    if str(configuration.get('n_envs','')).isdigit():configuration['n_envs']=int(configuration['n_envs'])
    # --motor-asked-change enters the fingerprint only when on: off builds nothing, so an acting snapshot
    # qualified before the flag existed still matches a launch that leaves it off
    if (configuration.get('motor_asked_change') or 'off') == 'off':configuration.pop('motor_asked_change',None)
    # --heads-all-slot-path likewise: in the fingerprint only when on
    if (configuration.get('heads_all_slot_path') or 'off') == 'off':configuration.pop('heads_all_slot_path',None)
    # --attention-cap and its bound likewise: in the fingerprint only when the cap is on (off changes no number)
    if (configuration.get('attention_cap') or 'off') == 'off':
        configuration.pop('attention_cap',None);configuration.pop('attention_cap_bound',None)
    # --vestibular-input-path: the same rule
    if (configuration.get('vestibular_input_path') or 'off') == 'off':configuration.pop('vestibular_input_path',None)
    # --finger-joint-servo: the same rule (off calls the decoder exactly as before)
    if (configuration.get('finger_joint_servo') or 'off') == 'off':configuration.pop('finger_joint_servo',None)
    # --replace-saved-hand-calibration: the same rule (absent replaces nothing)
    if not configuration.get('replace_saved_hand_calibration'):configuration.pop('replace_saved_hand_calibration',None)
    return configuration


def main(argv=None):
    args = parse_args(argv)
    from mimo_bridge.latent_body_shadow import validate_arguments as validate_latent_body_arguments
    validate_latent_body_arguments(args)
    if args.isolated_native_proof:
        from .tools.native_proof_role import validate
        validate(list(sys.argv[1:] if argv is None else argv))
    if args.self_choice != 'off':
        from .self_intention import SelfIntention
        validation=SelfIntention(args.self_choice,args.self_choice_receipt)
        if (validation.receipt['order_steps']!=args.motor_hold
                or abs(validation.receipt['step_seconds']-_transition_seconds(args))>1e-9):
            raise ValueError('self choice receipt does not match actual command cadence')
        if args.self_choice=='on' and (args.babble_bouts!='off' or not args.no_babble or args.random_actions or args.limp):
            raise ValueError('validated self choice requires the tested absence of scheduled motor overrides')
    if (getattr(args, 'persistent_memory_repair', 'off') or 'off') != 'off':
        if getattr(args, 'persistent_memory_constant_return', None) is None:
            raise ValueError('--persistent-memory-repair centred needs --persistent-memory-constant-return: the '
                             'length in updates of the folded constant\'s return to its pre-deploy size, as a training '
                             'measurement set it (the switch trial\'s return arm), or 0 to keep the constant as folded')
        if args.persistent_memory_constant_return < 0:
            raise ValueError('--persistent-memory-constant-return must be 0 or more')
    if args.self_state != 'off' and args.body_function == 'off':
        raise ValueError('--self-state requires an actual --body-function observation producer')
    if args.self_choice!='off' and args.body_function!='modulate':
        raise ValueError('frozen self-choice candidates require actual modulated experience')
    args.n_envs = auto_n_envs() if str(args.n_envs).lower() == "auto" else int(args.n_envs)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=2)
    # a resumed run reuses its folder: the previous segment's summary.json would make
    # every watcher (dashboard, overseer) treat this LIVE run as finished - so the run
    # would show stale and, worse, go unsupervised. Clear it at startup.
    try:
        os.replace(os.path.join(out, "summary.json"), os.path.join(out, "summary_prev.json"))
    except OSError:
        pass
    # code version for snapshot lineage: which commit produced these weights
    try:
        import subprocess
        _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_repo,
                                             text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = "unknown"
    stats_path = os.path.join(out, "replay_training_stats.jsonl")
    ckpt_dir = os.path.abspath(args.checkpoint_storage) if args.checkpoint_storage else os.path.join(out, "checkpoints")
    if args.checkpoint_storage:
        # Checkpoint ancestry remains self-contained for every existing copied
        # experiment: <run>/args.json and <run>/checkpoints/ckpt_step_... .
        checkpoint_run = Path(ckpt_dir).parent
        checkpoint_run.mkdir(parents=True,exist_ok=True)
        from .tools.training_continuation import atomic_json
        atomic_json(checkpoint_run/'args.json',vars(args))
    ckpt_list = []
    t_start = time.time()
    step = 0
    ok = False
    err = None
    src = None
    observer_lane = None       # --observer record: the model lane (mimo_bridge/observer_model_lane.py)
    comparison = None          # the comparison records (mimo_bridge/comparison_records.py), every run
    cm = None
    explain_writer = None
    actor = None
    try:
        model = build_learner(args, ckpt_dir)
        resumed = None
        if args.resume:
            # optimizer state loads when the snapshot's architecture matches (adaptive ->
            # adaptive); on a mode switch the group counts differ and CheckpointManager
            # falls back to a fresh optimizer with a warning - both paths are safe
            resumed = model.load_checkpoint(args.resume, load_optimizer=not args.fresh_optimizer,
                                            reinit=tuple(x.strip() for x in args.reinit.split(',') if x.strip()))
            from .tools.training_continuation import atomic_json
            atomic_json(Path(out) / 'load_report.json', {
                'source': args.resume, 'step': resumed.get('step'),
                'loaded_components': resumed.get('loaded_components'),
                'reinitialized_components': resumed.get('reinitialized_components'),
                'optimizer_requested': not args.fresh_optimizer,
            })
            if args.fresh_optimizer:
                log.info("fresh optimizer: Adam moments NOT restored "
                         "(the motor token's meaning changed; old moments describe the old one)")
            if args.think_harder == "on":
                log.info("think-harder resume: if the snapshot was fixed-mode, the motor "
                         "controller's thinking blocks start fresh (mismatched keys skip)")
            step = int(resumed.get("step", 0))
            # review T14: the constructor seeded every RNG with --seed, so each segment
            # re-ran the identical exploration-noise sequence from the top. Folding the
            # restored step in makes segments independent draws; a FRESH run with the
            # same seed is untouched.
            if step > 0:
                import torch as _torch
                _s2 = (int(args.seed) + step) % (2 ** 31)
                model.noise_rng.manual_seed(_s2)
                _torch.manual_seed(_s2)
                np.random.seed(_s2)
            if args.eval_only:
                # probe mode: take the WEIGHTS from the snapshot but start counting from 0
                # and use the requested phase, not the snapshot's
                step = 0
                model.step_count = 0
                resumed["curriculum_state"] = None
        if args.curriculum == "on":
            start_phase = args.start_phase
            cs = (resumed or {}).get("curriculum_state") or {}
            if cs.get("phase"):
                start_phase = cs["phase"]
            cm = CurriculumManager(steps_per_phase=args.steps_per_phase, mastery_advancement=False,
                                   start_phase=start_phase, report_every=max(1000, args.steps_per_phase // 2),
                                   handle_names=model.handle_names)   # its log lines count this body's handles
            if "phase_step" in cs:   # review T12: 0 is a value (snapshot on a boundary)
                cm._phase_step = int(cs["phase_step"]); cm._global_step = step
            # the same clock for every reader of the schedule: model.dev_limits(step) - the
            # planner's fallback, run_mode_a and run_mode_b - now answers what cm answers.
            # After the restore above, so phase_start_global matches the restored phase_step.
            bind_schedule_to_config(model, cm, args.steps_per_phase, step)
            phase = cm.phase_name
        else:
            phase = None
        if resumed:
            log.info("resume: step %d, phase %s (phase step %s)", step, phase, (resumed.get("curriculum_state") or {}).get("phase_step"))
        kw = env_kwargs_for(args, phase, out)
        from . import continuation_state
        model.receipt['episode_continuation']=continuation_state.VERSION
        if args.episode_continuation == "fresh":
            if not args.eval_only:
                raise ValueError('--episode-continuation fresh is probe-only: a training resume '
                                 'must migrate the saved physical episode, never restart it')
            episode_resume = None
            log.info('episode continuation: fresh eval-only probe worlds; the saved episode companion is untouched')
        else:
            episode_resume = continuation_state.load(args.resume)
        continuation_flags = ('n_envs','body_layout','action_repeat','motor_hold','body_function',
                              'self_state','persistent_memory','context_length','smooth_actions',
                              'babble_bouts','babble_noise','no_babble','random_actions','eval_only',
                              'self_choice','self_choice_receipt','contact_profile','caregiver_mode',
                              'caregiver_speech_dir','audio','no_vision','critters','mobile',
                              'gaze_reflex','spontaneous_saccades','episode_steps')
        continuation_config = {k:getattr(args,k) for k in continuation_flags}
        continuation_config['self_prediction']=args.self_prediction
        if episode_resume is not None:
            episode_resume['config'].setdefault('self_prediction','legacy')
        for key in ('caregiver_present_seconds','caregiver_absent_seconds'):
            continuation_config[key]=getattr(args,key)
        if episode_resume is not None and episode_resume['config'].get('caregiver_speech_dir')!=args.caregiver_speech_dir:
            episode_resume=continuation_state.extend_caregiver_speech(episode_resume,args.caregiver_speech_dir)
            log.info('Extended caregiver speech library; existing utterances, episode and learned state preserved')
        if episode_resume is not None:
            # a resumed world is rebuilt from its SAVED env settings; without this the resume's
            # --toy-hand-off / --finger-armature never reached it (segment 31, 2026-09-22)
            episode_resume,_env_changes=continuation_state.migrate_env_kwargs(
                episode_resume,args,launch_kwargs=kw,env_settings=args.resume_env_settings,log=log.info)
        if episode_resume is not None and episode_resume['config'] != continuation_config:
            raise ValueError('physical continuation configuration changed; explicit migration is required')
        # review 2026-09-06 (T14's own rationale, applied to the ENV side): without this
        # every resumed segment replays the identical saccade schedule, ball draws and
        # critter waypoints from the top - chained segments sample less world diversity
        # than the step count suggests. A fresh run (step 0) is untouched.
        if step > 0:
            kw["seed"] = (int(args.seed) + step) % (2 ** 31)
        # effect probes before hours land on a bad flag combination (tools/preflight.py).
        # Follow-up review 2026-09-08: a preflight that only logs is not enforcement -
        # failure now BLOCKS the launch unless --allow-preflight-fail is passed (the
        # legitimate band-violating experiment states that intent on its launch line).
        try:
            from mimo_bridge.tools.preflight import run_probes
            # the settings of the world this launch builds: the saved first world on a resumed
            # episode, else kw. Until 2026-09-23 the gate rebuilt its own and tested the 24-row
            # legacy body on every trunk_v1 run (fix-rows-r1-preflight).
            pf = run_probes(os.path.join(out, "args.json"),
                            env_kw=(episode_resume['worlds'][0]['env_kwargs']
                                    if episode_resume is not None else kw))
            json.dump(pf, open(os.path.join(out, "preflight.json"), "w"), indent=2)
            bad = [c["name"] for c in pf["checks"] if not c["ok"]]
            log.info("preflight %s%s", "PASS" if pf["ok"] else "FAIL",
                     "" if pf["ok"] else f" on {bad}")
            if not pf["ok"] and not args.allow_preflight_fail:
                raise SystemExit(f"preflight FAIL on {bad}: launch refused "
                                 "(pass --allow-preflight-fail to run anyway)")
        except SystemExit:
            raise
        except Exception:
            log.warning("preflight could not run: %s", traceback.format_exc())
            if not args.allow_preflight_fail:
                raise SystemExit("preflight could not run: launch refused "
                                 "(pass --allow-preflight-fail to run anyway)")
        # say the scene out loud. H16, H17 and the first H18 all trained a WELDED, suspended
        # body because --scene-binding phase was dropped from the launch line, and nothing in
        # any log said which world the bodies were in.
        log.info("scene: %s (--scene %s, binding %s), %d bodies",
                 kw.get("scene"), args.scene, args.scene_binding, args.n_envs)
        if episode_resume is not None:
            saved_worlds=episode_resume['worlds']
            if args.n_envs==1:src=SingleEnvSource(**saved_worlds[0]['env_kwargs'])
            else:src=PoolSource(args.n_envs,{},per_env_kwargs=[w['env_kwargs'] for w in saved_worlds])
            frames=src.restore_full_state(saved_worlds)
        elif args.n_envs == 1:
            src = SingleEnvSource(**kw)
        else:
            src = PoolSource(args.n_envs, kw)
        if episode_resume is None:frames = src.reset()
        if getattr(args, "observer", "off") == "record":
            # the developmental observer's streams (addendum v4 Part 6): the workers record the
            # world after every burst, this process the learner's predictions; nothing returns
            from .observer_model_lane import ModelLane
            _obs_budget = max(0.0, float(args.observer_budget_gb)) * 2 ** 30
            _obs_reserve = int(max(0.0, float(args.observer_reserve_gb)) * 2 ** 30)
            src.observer_open(os.path.join(out, "observer", "world"), run=os.path.basename(out),
                              chunk_steps=args.observer_chunk_steps,
                              images_every_s=60.0 * max(0.0, args.observer_images_every_min),
                              budget_bytes=int(_obs_budget * 5 / 6 / max(1, args.n_envs)),
                              reserve_bytes=_obs_reserve,
                              extra=dict(n_envs=args.n_envs, action_repeat=args.action_repeat,
                                         motor_hold=args.motor_hold))
            observer_lane = ModelLane(os.path.join(out, "observer", "model"), run=os.path.basename(out),
                                      n_envs=args.n_envs, handle_names=list(model.handle_names),
                                      chunk_steps=args.observer_chunk_steps,
                                      budget_bytes=int(_obs_budget / 6), reserve_bytes=_obs_reserve)
            model.observer_per_body_errors = True
            log.info("developmental observer recording to %s", os.path.join(out, "observer"))
        # the body's own clock, measured rather than assumed. env.dt is MuJoCo's timestep
        # x frame_skip; it was exported nowhere (not args.json, not the receipt, not the
        # snapshot), so a scene-timestep or frame_skip change would have moved every
        # seconds-denominated number in the run with nothing noticing.
        env_dt = float(frames[0].dt)
        model.receipt["env_dt_s"] = env_dt
        model.receipt["model_step_s"] = _transition_seconds(args, env_dt)
        log.info("one env step %.6f s (measured from the running body), one model step "
                 "%.6f s at --action-repeat %d; the eyes render on the last env step of "
                 "the %d", env_dt, _transition_seconds(args, env_dt), args.action_repeat,
                 max(1, args.action_repeat))
        if abs(env_dt - ENV_DT_ASSUMED_S) > 1e-12:
            log.warning("env step is %.6f s, not the %.6f s this module assumes before a "
                        "body exists: the self-choice receipt check at launch used the "
                        "assumed value while the bout schedule and the transition ledger "
                        "use the measured one", env_dt, ENV_DT_ASSUMED_S)
        model.reset_context(args.n_envs)
        if args.replay == "on" and not args.eval_only:
            model.enable_replay(args.n_envs, size_per_env=args.replay_size)
            # the replay batch the switch's ring check reads the stored states in (--persistent-memory-repair)
            model.replay_batch = int(args.replay_batch)
            if args.fill_ring_first:
                # Plan item 5. Every moment the ring held before today is labelled with an
                # order that was never executed. Re-studying those teaches the wrong
                # contract, so nothing is re-studied until the ring is full of transitions
                # carrying the order the body actually got. Live learning continues
                # throughout - only replay waits.
                model.replay.clear()
                # Enough to form one context window plus a target, with slack for the
                # rejection sampling in sample_windows. NOT the whole ring: the ring is
                # in-memory and was already empty, so a full fill buys nothing and cost
                # ~23 minutes of replay silence per launch.
                ring_need = max(4 * args.replay_batch, 8 * (args.context_length or 32))
                # NOT 'same-episode': labelled_count() (replay.py:168-172) reduces
                # has_executed with .any() over the bodies and never looks at the episode
                # column. The episode rule is enforced later, when sample_windows rejects
                # a window that would span a reset - not by this count.
                log.info("replay held back until %d stored moments carry their outgoing "
                         "action (%.0f s of body time; ring holds %d per body; live "
                         "training is NOT gated)",
                         ring_need, ring_need * _transition_seconds(args, env_dt),
                         args.replay_size)
        # review T8: lem's keep_last_n=10 deleted the earliest hourly checkpoints of
        # chained runs - the before/after evidence the working rules lean on. The
        # bridge keeps 48 (lite snapshot ~37 MB -> 1.8 GB ceiling on ~27 GB free).
        if model.checkpoints is not None:
            model.checkpoints._keep_last_n = max(int(getattr(model.checkpoints, "_keep_last_n", 10)), 48)
        patch_names = None
        probe_rows = []
        probe_hook = None
        if args.probe_touch:
            from mimo_bridge.env import BridgeEnv  # names only
            # get patch names from a throwaway env in this process (cheap, no vision)
            tmp = BridgeEnv(scene=args.scene, run_dir=os.path.join(out, "scenes"), vision=False, tag=args.tag + "_names")
            patch_names = tmp.patch_names(); tmp.close()
            regions = [region_of(n) for n in patch_names]
            captured = {}
            def _hook(mod, inputs, output):
                captured["x"] = inputs[1].detach()   # [B, N, d] per-patch embeddings fed to the pool
            probe_hook = model.encoder_registry.encoders["tactile"].pool_attn.register_forward_hook(_hook)

        # THE COMPARISON RECORDS (mimo_bridge/comparison_records.py, Oct 10, 2026), in every run, no launch flag: per
        # model step the gradient norms before the clip (record c), per replay pass its windows and losses (record c),
        # and the spike rule's inputs and weights (record d), in <out>/comparison_records. Records a and b ride the
        # observer's model lane above. A failure disables the record that failed and never stops the learner.
        try:
            from .comparison_records import ComparisonRecords
            comparison = ComparisonRecords(os.path.join(out, "comparison_records"), run=os.path.basename(out),
                                           n_envs=args.n_envs)
            comparison.attach(model)
        except Exception:
            comparison = None
            log.warning("comparison records not started (training unaffected): %s",
                        traceback.format_exc().strip().splitlines()[-1])
        ctrl_path = os.path.join(out, "control.json")
        open(os.path.join(out, "control_supported"), "w").write(
            "snapshot stop snapshot_and_stop video advance_phase explain stage_actor promote_actor")
        overseer_stop = None
        explain_pending = False

        ctrl_retries = {"n": 0}

        def poll_control():
            """Overseer commands, handled BETWEEN steps (never during a snapshot, because
            snapshots happen in this same loop): snapshot | stop | snapshot_and_stop |
            video | advance_phase | explain. `explain` only SETS A FLAG: poll_control
            runs before model.step, when `frames` already holds the NEXT step's inputs
            while the model caches describe the previous one - snapshotting here would
            pair eye images with tokens they did not produce (and on a fresh run's step
            0 there is no completed step at all). The snapshot is taken at the hook
            after the next model.step, where frames and caches agree."""
            nonlocal overseer_stop, explain_pending, actor
            if not os.path.exists(ctrl_path):
                return
            # review T4: the old read used the locale codepage (cp1252 here) and
            # consumed the file on ANY failure - a command written by PowerShell 5.1
            # (UTF-16LE BOM) or caught mid-write was eaten and logged as cmd "".
            # BOM-aware decode; a bad parse leaves the file for the next poll (25
            # steps later) and only the third consecutive failure consumes it.
            try:
                raw = open(ctrl_path, "rb").read()
                if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                    text = raw.decode("utf-16")
                else:
                    text = raw.decode("utf-8-sig")
                command_data = json.loads(text)
                cmd = command_data.get("cmd", "")
                ctrl_retries["n"] = 0
            except Exception as e:
                ctrl_retries["n"] += 1
                if ctrl_retries["n"] < 3:
                    log.warning("control.json unreadable (%s) - retry %d/3 next poll",
                                e, ctrl_retries["n"])
                    return
                log.error("control.json unreadable 3 polls running (%s) - consumed", e)
                cmd = ""
                ctrl_retries["n"] = 0
            try:
                os.replace(ctrl_path, os.path.join(out, f"control_done_{int(time.time())}.json"))
            except OSError:
                return
            log.info("overseer command: %s", cmd)
            sf.write(json.dumps({"type": "overseer_cmd", "timestamp": time.time(), "step": step, "cmd": cmd}) + "\n"); sf.flush()
            if cmd in ("snapshot", "snapshot_and_stop"):
                save_ckpt("overseer")
            if cmd == "video":
                try:
                    src.start_video(os.path.join(out, "videos", f"step_{step}_overseer.mp4"),
                                    n_frames=int(args.video_seconds / frames[0].dt))
                except Exception:
                    pass
            if cmd == "advance_phase" and cm is not None and not cm.is_final_phase:
                cm._phase_step = max(cm._phase_step, args.steps_per_phase)   # next cm.step() rolls the phase
                log.info("overseer advanced the curriculum (bored of this phase)")
            if cmd == "explain":
                explain_pending = True
            if cmd == 'promote_actor':
                try:
                    if actor is None:raise ValueError('no independently observing candidate exists')
                    if args.babble_bouts!='off' or not args.no_babble or args.random_actions or args.limp:
                        raise ValueError('candidate adoption requires the qualified absence of external motor overrides')
                    receipt_path=Path(command_data['receipt'])
                    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
                    if receipt['order_steps']!=args.motor_hold or abs(receipt['step_seconds']-_transition_seconds(args))>1e-9:
                        raise ValueError('candidate receipt differs from physical execution cadence')
                    from .tools.actor_runtime import require_outer_actor
                    require_outer_actor(actor)
                    actor=actor.promote(receipt)
                    event=dict(type='actor_promotion',timestamp=time.time(),step=step,accepted=True,
                               receipt=str(receipt_path),snapshot_step=actor.born_step)
                except (KeyError,OSError,TypeError,ValueError) as ex:
                    event=dict(type='actor_promotion',timestamp=time.time(),step=step,accepted=False,reason=str(ex))
                sf.write(json.dumps(event)+'\n');sf.flush()
                log.info('acting snapshot promotion: %s',event)
            if cmd == 'stage_actor':
                try:
                    if actor is None:raise ValueError('no qualified acting snapshot exists')
                    calibration=json.loads(Path(command_data['calibration']).read_text(encoding='utf-8'))
                    actor.stage(model,factory,calibration)
                    event=dict(type='actor_candidate',timestamp=time.time(),step=step,accepted=True,
                               snapshot_step=actor.pending.born_step)
                except (KeyError,OSError,TypeError,ValueError) as ex:
                    event=dict(type='actor_candidate',timestamp=time.time(),step=step,accepted=False,reason=str(ex))
                sf.write(json.dumps(event)+'\n');sf.flush()
                log.info('acting snapshot candidate: %s',event)
            if cmd in ("stop", "snapshot_and_stop"):
                overseer_stop = cmd

        # Checkpoint cadence is independent of world continuity. Zero means no
        # scheduled process exit; the supervisor still resumes unexpected exits.
        deadline = t_start + args.minutes * 60.0 if args.minutes > 0 else float('inf')
        last_ckpt = time.time()
        ckpt_failures = 0          # hourly saves that raised (2026-09-18); reported in the stats
        ckpt_last_error = None
        last_video = time.time() - max(0.0, args.video_every_min * 60.0 - 120.0)   # first video 2 min in
        last_moment_video = 0.0   # critter-event clips (plan D13), 3-min cooldown
        _rec = {"on": False, "checked": 0.0, "n": 0}
        acc = dict(loss=0.0, grad=0.0, n=0, pe={}, rd={}, wl={}, born=0.0, died=0.0, turn_n=0, t_model=0.0, t_env=0.0, conf=0.0, bd={}, coh=0.0,
                   contacts=0.0, active=0.0, pressure=0.0, nan=0, cmd_cos=0.0, cmd_ratio=0.0, wgrad=0.0, cmd_acc=0.0,
                   cmd_win=0.0, ball_d=0.0, ball_touch=0.0, ball_n=0,
                   mobile_gap_min=float("inf"),
                   box_l=0.0, box_m=0.0, box_h=0.0, roll=0.0, roll_max=0.0, rolled=0.0,
                   vis_fresh=0.0)
        _prev_active: List[np.ndarray] = []   # per-body active-patch mask, for contact turnover
        # who authored the orders the body actually executed (tallied at the
        # latch only): "planner" = intention output (babble structurally skipped),
        # "reflex_noise" = controller + babble, "reflex" = controller, no noise.
        # Added 2026-09-14 after measuring that motor_hold == intention_every put
        # every latched order on a planner step - executed babble fraction was 0.
        exec_src = {"planner": 0, "reflex_noise": 0, "reflex": 0}
        # executed-command size gauges (2026-09-14 diagnosis gaps 1 and 5: nothing
        # measured commanded strength or the positional step being asked for, and
        # the hand-specific decline was invisible in all-handle rotation means).
        # Hands = handles 0-11 (both wrists + all ten fingers). The overall means cover the
        # body parts: the eyes' row is left out by name (cmd_exec_sample). pos_step_cm is the
        # position step ASKED of the body parts, as plan0's step_cm is; it is not the step the
        # body receives, because only the handles in action_decoder.POSITION_HANDLES send a
        # position to the decoder's solve, and the arm and leg segments, the feet and the
        # chest are orientation-only rows that are still counted. Values recorded before
        # 2026-09-23 include the eyes' row; on segment 31's 16 stored rows it lowered strength
        # by 0.0012 on the 0-to-1 scale and the step by 0.10 cm.
        cmdg = {"n": 0, "strength": 0.0, "strength_hands": 0.0,
                "step_cm": 0.0, "step_cm_hands": 0.0}
        motor_ring: List[np.ndarray] = []   # last actions [B, 207] for the discrimination ratio
        held_motor = None                   # the order currently in force (see --motor-hold)
        held_author = "unknown"             # who wrote it (the observer's provenance); set at each latch
        held_hand_joint_goal = None
        prev_motor_substeps = None
        prev_hand_joint_goal = None
        prev_action_contract = 'world_v0'
        # TransitionV1: the order that ACTUALLY produced the frames about to be observed,
        # carried across the iteration boundary. held_motor with handle 23 replaced by the
        # eye command the env really issued after the saccade generator and the reflex had
        # their say. Without this the model is told it did what it proposed, which under
        # motor_hold=5 is wrong on every step: four of five proposals are never issued at
        # all, and the fifth reaches the body only after developmental limits and
        # exploration noise have changed it.
        prev_executed = None
        prev_order_id = None     # serial of the order prev_executed carries (item 1)
        # the model's gaze command over the same transition (res.gaze_out, before the world's
        # saccade generator and gaze reflex): the replay ring keeps it beside the ledger row, whose eye
        # row is written from the gaze the world SENT (motor fixes deploy, F1: the switch splits the row)
        prev_ctrl_gaze = None
        order_seq = 0
        order_age = 0            # steps since the held order changed (explainer's hold
                                 # phase; step % hold is wrong after resets)
        last_explain = time.time()
        prev_gaze_substeps = None
        prev_decoded_ctrl = None
        # One Pending per body, opened when S_t is observed and closed when S_t+1
        # arrives with the action that carried the body there. This is the contract as
        # an object rather than four loose variables that agree by convention.
        pendings = []
        # (review 2026-09-06: a `last_transitions` retention list sat here - assigned
        # every step, read by nothing anywhere. The validation and the periodic ledger
        # log below keep their jobs; only the dead storage is gone.)
        ring_clean = not args.fill_ring_first    # plan item 5 gate (replay only)
        # review T1: `ring_need = 0` used to sit here, overwriting the 512 computed at
        # launch - replay began at ~67 stored moments (about 2 valid window starts per
        # body) and took a full optimizer step every 4 live steps on near-duplicates.
        # ring_need is only computed inside the replay-enabled branch above; an
        # eval-only run (replay never enabled) with --fill-ring-first otherwise
        # reaches the episode companion writer with ring_need unbound.
        if not (args.fill_ring_first and args.replay == "on" and not args.eval_only):
            ring_need = 0
        sf = open(stats_path, "a")
        # review T13: exec_seen/nf_aligned are process-local; without the segment's
        # start step in the stream, any reader but the dashboard divides them by the
        # GLOBAL step (~0.001% instead of ~100% after a resume)
        sf.write(json.dumps({"type": "launch", "timestamp": time.time(), "step": int(step),
                             "resumed_from": args.resume, "code_commit": git_commit}) + "\n")
        if getattr(model, "vestibular_switch", None):
            # what the checkpoint load did to put the vestibular sense in the intermind slot
            sf.write(json.dumps({"type": "vestibular_switch", "timestamp": time.time(),
                                 "step": int(step), "part": "weights",
                                 "weights": model.vestibular_switch}) + "\n")
        if getattr(model, "vestibular_encoder_repair_record", None):
            # --vestibular-encoder-repair rescale divided the encoder's last layer at this load: the
            # deploy's own marker in the stream (what was divided, by what, what the check read)
            sf.write(json.dumps({"type": "vestibular_encoder_repair", "timestamp": time.time(),
                                 "step": int(step), "switch": model.vestibular_encoder_repair_record},
                                default=str) + "\n")
        if getattr(model, "motor_asked_change_switch", None):
            # --motor-asked-change on at a checkpoint trained without the path: the switch's own marker in the
            # stream (the probe rows' token check: 0.0 = the same token with and without the path)
            sf.write(json.dumps({"type": "motor_asked_change_switch", "timestamp": time.time(),
                                 "step": int(step), "switch": model.motor_asked_change_switch},
                                default=str) + "\n")
        if getattr(model, "heads_all_slot_path_switch", None):
            # --heads-all-slot-path on at a checkpoint trained without the paths: the switch's own marker in the
            # stream (the probe rows' prediction check: 0.0 = the same predictions with and without the paths)
            sf.write(json.dumps({"type": "heads_all_slot_path_switch", "timestamp": time.time(),
                                 "step": int(step), "switch": model.heads_all_slot_path_switch},
                                default=str) + "\n")
        if getattr(model, "attention_cap_change", None):
            # --attention-cap differs from the cap the checkpoint was saved under (ATTENTION_CAP): the switch's own
            # marker in the stream, {saved, now, checkpoint} (each the receipt's stamp, None = without the cap)
            sf.write(json.dumps({"type": "attention_cap_switch", "timestamp": time.time(),
                                 "step": int(step), "switch": model.attention_cap_change},
                                default=str) + "\n")
        if getattr(model, "vestibular_input_path_switch", None):
            # --vestibular-input-path on at a checkpoint trained without the path: the switch's own marker in the
            # stream (the probe rows' token check: 0.0 = the same token with and without the path)
            sf.write(json.dumps({"type": "vestibular_input_path_switch", "timestamp": time.time(),
                                 "step": int(step), "switch": model.vestibular_input_path_switch},
                                default=str) + "\n")
        if getattr(model, "motor_interface_change", None):
            # a resume that switched --eye-row-convention or --executed-fields-only (audit F1/F4):
            # the deploy's own marker in the stream, {setting: [saved, now]}
            # and what the load did to the saved averages over the order's fields (converted where
            # exact: LemLite._convert_order_averages, review finding 2)
            sf.write(json.dumps({"type": "motor_interface_change", "timestamp": time.time(),
                                 "converted": getattr(model, "motor_interface_conversion", None),
                                 "step": int(step), "changes": model.motor_interface_change}) + "\n")
        if getattr(model, "persistent_memory_switch", None):
            # --persistent-memory-repair: this segment resumed a SWITCHED memory; the switch's own record (its
            # step, its readings, the way back: pre_switch_checkpoint) in every later segment's stream
            sf.write(json.dumps({"type": "persistent_memory_switch", "timestamp": time.time(),
                                 "step": int(step), "part": "loaded",
                                 "switch": model.persistent_memory_switch}) + "\n")
        sf.flush()
        if getattr(model, "proprio_token_repair_switch", None):
            # --proprio-token-repair within_body on a checkpoint trained without it: the deploy's own marker in the
            # stream, from which the hourly watcher judges the repair's kill lines (proprio_repair_watch.py)
            sf.write(json.dumps({"type": "proprio_token_repair", "timestamp": time.time(), "step": int(step),
                                 "switch": model.proprio_token_repair_switch}, default=str) + "\n")
            sf.flush()
        # the vestibular block of the stats record (--vestibular-sense on)
        vest_win = None
        if getattr(model, "vestibular_sense", False):
            from .vestibular_sense import VestibularWindow
            vest_win = VestibularWindow()
        # the machine block (always on) and the causal-chain recorder (--causal-chain on);
        # see mimo_bridge/machine_stats.py and mimo_bridge/causal_chain.py
        from .machine_stats import MachineSampler
        from .env import merge_burst_contact_report     # the burst merge's contact events
        machine = MachineSampler()
        causal = None
        if getattr(args, "causal_chain", "off") == "on":
            from .causal_chain import CausalChainRecorder
            causal = CausalChainRecorder(out, hold=max(1, args.motor_hold))

        def flush_stats():
            if comparison is not None:
                # the comparison records' held lines and status file reach disk before this stats record is written
                # (ComparisonRecords.flush never raises): a stop read on this record loses only the lines added after
                # it (the replay pass that runs at this counter after the record, and the model steps that follow)
                comparison.flush()
            n = max(1, acc["n"])
            now = time.time()
            last_window_plan, planning_summary = planning_window_summary(acc, step - int(acc["n"]), step)
            # t_causal: the causal-chain recorder's latch predictions (absent, so 0, when the
            # flag is off); counted in the step rate the watcher judges, never as env time
            fps = n / max(1e-9, acc["t_model"] + acc["t_env"] + acc.get("t_causal", 0.0))
            rec_i = build_inference_stats_record(
                inference_step=step, frame_number=frames[0].frame, fps=fps,
                avg_ms=1000 * (acc["t_model"] + acc["t_env"] + acc.get("t_causal", 0.0)) / n,
                buffer_fill=0.0, buffer_size=0, phase_name=dev["phase_name"], motor_confidence=acc["conf"] / n,
                has_fresh_vision=bool(frames[0].vision_patches is not None), source_mask=res.source_mask,
                stall_count=0, timings={"model": 1000 * acc["t_model"] / n, "env": 1000 * acc["t_env"] / n},
                vram_mb=None, peak_vram_mb=None, alloc_retries=None, transfer_times=None, gpu_info=None)
            rec_t = build_sync_training_stats_record(
                training_step=step, avg_loss=acc["loss"] / n, avg_grad=acc["grad"] / n, avg_train_ms=1000 * acc["t_model"] / n,
                batch_size=args.n_envs, buffer_fill=0.0, phase_name=dev["phase_name"],
                # item 4: real sources only - the synthetic noise channels used to
                # inflate this mean by construction
                train_mean_pe=(lambda _pr: (sum(_pr) / len(_pr) / n) if _pr else None)(
                    [v for k, v in acc["pe"].items()
                     if (res.source_mask or {}).get(k) == "real"]),
                loss_breakdown={k: v / n for k, v in acc["bd"].items()}, step_timings=None, priority_beta=None,
                per_sample_mse_stats=None, grad_norms=None, encoder_activations=None, temporal_attention=None,
                coherence_score=acc["coh"] / n, peak_vram_mb=None)
            rec_m = {"type": "mimo", "step": step, "timestamp": now, "elapsed_s": now - t_start, "phase": dev["phase_name"],
                     'continuity': dict(scheduled_episode_steps=args.episode_steps,
                         scheduled_process_minutes=args.minutes, simulation_seconds=[float(f.sim_time) for f in frames]),
                     # dict(model.last_memory), and with --persistent-memory-repair the report window's means
                     # of the repaired memory's gauges (LemLite.memory_report)
                     'persistent_memory': model.memory_report(),
                     # the last step's self-state stats plus per-step counts over the report (window_*)
                     'self_state': self_state_report(model),
                     'acting_snapshot': None if actor is None else dict(actor.last),
                     'mechanical_contact': [None if not (f.meta or {}).get('mechanical_contact') else
                         {k:v for k,v in f.meta['mechanical_contact'].items() if k != 'history'} for f in frames],
                     "prediction_audit": summarize_prediction_audit(acc.get('prediction_audit', {})),
                     # the same per body, with the pooled readings without body 2 (the operator's decision of
                     # 2026-09-25 7:47 PM PDT; prediction_audit.EXCLUDED_BODIES)
                     "prediction_audit_per_body": summarize_prediction_audit_rows(acc.get('prediction_audit_rows', {})),
                     # the swap test on the hold's first model step, every arm, per body and pooled (2026-10-02,
                     # prediction_audit.audit_blocks; docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md)
                     "prediction_audit_hold_first": summarize_prediction_audit_hold_first(
                         acc.get('prediction_audit_hold_first_rows', {})),
                     "pe_definition": "predicted_change_from_current_tokens; not next-frame accuracy" if args.next_frame_loss == 'on' else 'same-frame reconstruction',
                     "age_months": kw.get("age"), "delays": (kw.get("sensory_delay", 0), kw.get("motor_delay", 0)),
                     "n_envs": args.n_envs,
                     "body_layout": model.body_layout,
                     "caregiver_storage": [f.meta.get('caregiver', {}).get('recording') for f in frames],
                     "caregiver": [{k:v for k,v in f.meta.get('caregiver',{}).items() if k!='recording'} for f in frames],
                     "body_orientation": [f.meta.get('task',{}).get('body_orientation') for f in frames],
                     "motor_contract": [f.meta.get('motor_contract') for f in frames],
                     # each model step drives action_repeat env frames per body
                     "env_steps_per_s": fps * args.n_envs * max(1, args.action_repeat),
                     "model_steps_per_s": fps,
                     # the largest CUDA allocation this window (MiB, the caching allocator's own
                     # count, reset after each record); None on CPU. nvidia-smi's figure adds the
                     # worker renderers (~100 MiB per body) and reserved-but-unallocated blocks.
                     "hand_control": ({"contract":"wrist_relative_joint_v1",
                        "joint_error_deg":acc.get('hand_joint_error_sum',0.)/max(1,acc.get('hand_joint_error_n',0)),
                        "body_observations":acc.get('hand_joint_error_n',0),
                        "canonical_readback_failures":0,
                        "physical_substeps_recorded":acc.get('hand_substeps',0)}
                        if model.hand_target_mode != 'world' else None),
                     "physical_goal_mode": model.physical_goals,
                     "hand_goal_mode": model.hand_goals,
                     "peak_vram_mb": ((torch.cuda.max_memory_allocated() / 2 ** 20)
                                      if torch.cuda.is_available() and "cuda" in str(args.device) else None),
                     "env_dt_s": frames[0].dt,
                     # share of recent steps with babble ON (EMA, ~100-step horizon);
                     # 1.0 = the old continuous regime, ~0.64 = the bout schedule
                     "babble_duty": round(bout_ema, 4),
                     # BODY-side stillness (never the schedule): achieved-pose speed
                     # per step under rest vs bout bursts, and their unit-free ratio
                     # (1.0 = rests still no rest; the 'real' mode's gate is < 0.5)
                     "rest_body_speed": (round(rest_speed_ema, 6)
                                         if rest_speed_ema is not None else None),
                     "bout_body_speed": (round(bout_speed_ema, 6)
                                         if bout_speed_ema is not None else None),
                     "rest_over_bout_speed": (
                         round(rest_speed_ema / bout_speed_ema, 4)
                         if rest_speed_ema is not None and bout_speed_ema else None),
                     # obedience judged where commands are meant to be followed:
                     # follow_bout carries the 0.02 kill line; follow_rest grades the
                     # pose-hold (asked ~0, cosine noise) and is report-only
                     "follow_bout": (round(follow_bout_ema, 4)
                                     if follow_bout_ema is not None else None),
                     "follow_rest": (round(follow_rest_ema, 4)
                                     if follow_rest_ema is not None else None),
                     "pe": {k: v / n for k, v in acc["pe"].items()}, "wellness": {k: v / n for k, v in acc["wl"].items()},
                     "contacts_per_env": acc["contacts"] / n, "touch_active_patches": acc["active"] / n,
                     "contact_turnover": ({"born": acc["born"] / acc["turn_n"],
                                           "died": acc["died"] / acc["turn_n"]}
                                          if acc["turn_n"] else None),
                     "touch_mean_pressure_pa": acc["pressure"] / n,
                     # the max over the window, NOT a mean: this is the statistic
                     # contact_wellness actually scores pain from
                     "touch_peak_pressure_pa": acc.get("pressure_peak", 0.0),
                     "sim_time": frames[0].sim_time,
                     "nan_steps": acc["nan"], "rom_limit": dev["rom_limit"], "strength_limit": dev["strength_limit"],
                     # hourly saves that raised since this segment started (0 = every save
                     # landed); the watcher alerts on any, the next hour tries again
                     "checkpoint_failures": ckpt_failures, "checkpoint_last_error": ckpt_last_error,
                     "curiosity_drive": (acc["cur"] / acc["cur_n"]) if acc.get("cur_n") else None,
                     # steps in this window whose drive was computed from the
                     # reconstruction error because the delayed lesson had no entry:
                     # they are inside curiosity_drive's mean and outside causal_pe's
                     "curiosity_drive_steps": int(acc.get("cur_n", 0)),
                     "curiosity_drive_fallback_steps": int(acc.get("cur_fallback_n", 0)),
                     "causal_pe_steps": int(acc.get("cpe_n", 0)),
                     "imagination_loss": (acc["im"] / acc["im_n"]) if acc.get("im_n") else None,
                     "hself": model.hself_stats(),
                     "entity_self": model.entity_self_stats(),
                     "incorporation": model.incorporation_stats(),
                     "token_temporal_std": model.token_temporal_std(),
                     # the collapse watch on every body, per sense (entry 0 = token_temporal_std's body 0): a deaf
                     # body shows here the report after it goes deaf (proprio_token_2026-09-25.md)
                     "token_temporal_std_bodies": model.token_temporal_std_bodies(),
                     # --proprio-token-repair within_body: every body's reading of the lesson (absent when off)
                     **({"proprio_token_repair": model.proprio_within.gauges()}
                        if getattr(model, "proprio_within", None) is not None else {}),
                     "cmd_response_cos": acc["cmd_cos"] / n, "cmd_response_ratio": acc["cmd_ratio"] / n,
                     "cmd_response_acc_cos": acc["cmd_acc"] / n,
                     "cmd_response_window_cos": acc["cmd_win"] / n,
                     "cmd_response_pos_cos": acc.get("cmd_pos", 0.0) / n,
                     # under --executed-fields-only on the gauge above scores only the 14 rows whose
                     # position the decoder reads; the marker says so (absent = every powered row)
                     **({"cmd_response_pos_cos_rows": "position_handles"}
                        if model.executed_fields_only_on else {}),
                     # the motor interface settings in force (audit 2026-09-25 F1/F4), written only when
                     # one differs from the old default, so a record says which side of a deploy it is
                     **({"motor_interface": {"eye_row_convention": model.eye_row_convention,
                                             "executed_fields_only": bool(model.executed_fields_only_on),
                                             # the decoder setting whose live position rows F4 keeps
                                             # (--position-rows, audit F3): F3 and F4 must agree
                                             "position_rows": str(getattr(model, "position_rows", "all")
                                                                  or "all")}}
                        if (model.eye_row_convention != "legacy" or model.executed_fields_only_on)
                        else {}),
                     "cmd_mag_rad": acc.get("cmd_mag", 0.0) / n,
                     "wellness_grad_norm": acc["wgrad"] / n,
                     "wellness_grad_zero_frac": acc.get("wgrad_zero", 0) / n,
                     "task_hand_ball_dist_m": (acc["ball_d"] / acc["ball_n"]) if acc["ball_n"] else None,
                     "task_ball_touch_rate": (acc["ball_touch"] / acc["ball_n"]) if acc["ball_n"] else None,
                     "task_ball_touch_hand_rate": (acc.get("ball_hand", 0.0) / acc["ball_n"]) if acc["ball_n"] else None,
                     "task_gaze_on_ball_rate": (acc.get("gaze_on", 0.0) / acc["ball_n"]) if acc["ball_n"] else None,
                     "task_gaze_on_ball_earned_rate": (acc.get("gaze_earn", 0.0) / acc["ball_n"]) if acc["ball_n"] else None,
                     "task_gaze_ball_deg": (acc.get("gaze_deg", 0.0) / acc["ball_n"]) if acc["ball_n"] else None,
                     "task_box_touch_rate": {"light": acc["box_l"] / n, "medium": acc["box_m"] / n, "heavy": acc["box_h"] / n},
                     "task_box_touch_hand_rate": acc.get("box_hand", 0.0) / n,
                     "replay_loss": (acc.get("replay_loss", 0.0) / acc["replay_n"]) if acc.get("replay_n") else None,
                     # a run in which the executed action never reached the model must not
                     # be indistinguishable from one in which it reached it every step
                     "contract": {"exec_seen": int(getattr(res, "exec_seen", 0)),
                                  "nf_aligned": int(getattr(res, "nf_aligned", 0)),
                                  "step": int(step)},
                     # `pe` above is CURRENT-frame reconstruction. The causal quantity is
                     # the delayed lesson's error. Both are recorded, named for what they
                     # are, so nobody reads reconstruction as evidence about the action.
                     "recon_pe": (acc["rpe"] / acc["rpe_n"]) if acc.get("rpe_n") else None,
                     "causal_pe": (acc["cpe"] / acc["cpe_n"]) if acc.get("cpe_n") else None,
                     "causal_pe_per_modality": ({k2[3:]: v2 / acc["cp_n"] for k2, v2 in acc.items()
                                                 if k2.startswith("cp_") and k2 != "cp_n"}
                                                if acc.get("cp_n") else None),
                     "replay_contract": (None if not acc.get("replay_n") else
                         dict({k2: acc.get(k2, 0.0) / acc["replay_n"] for k2 in
                               ("replay_graded_mods", "replay_clean_frac",
                                "replay_fill_frac", "replay_batch")},
                              **{k2: acc.get("last_" + k2) for k2 in
                                 ("replay_motor_rebuilt", "replay_motor_unlabeled",
                                  "replay_ctx_embedded")})),
                     "intention": last_window_plan,
                     "intention_window": planning_summary,
                     "movement": (None if model.movement is None else
                                  {k2[3:]: v2 / n for k2, v2 in acc.items() if k2.startswith("mv_")}),
                     "next_pe": ({k2[3:]: v2 / n for k2, v2 in acc.items() if k2.startswith("np_")} or None),
                     "recon_dec": ({k2: v2 / n for k2, v2 in acc.get("rd", {}).items()} or None),
                     "tenure_pe": ({k2[3:]: v2 / acc["tn_n"] for k2, v2 in acc.items()
                                    if k2.startswith("tn_") and k2 != "tn_n"}
                                   if acc.get("tn_n") else None),
                     "tenure_gap": ({k2[3:]: v2 / acc["tn_n"] for k2, v2 in acc.items()
                                     if k2.startswith("tg_")} if acc.get("tn_n") else None),
                     "tenure_tok": ({k2[3:]: v2 / acc["tt_n"] for k2, v2 in acc.items()
                                     if k2.startswith("tt_") and k2 != "tt_n"}
                                    if acc.get("tt_n") else None),
                     "agency": ({k2[3:]: v2 / max(1, acc.get("agN_" + k2[3:], 0))
                                 for k2, v2 in acc.items() if k2.startswith("ag_")}
                                if any(k2.startswith("agN_") and v2 for k2, v2 in acc.items())
                                else None),
                     # consequence access (2026-09-18): kinds_mean, rate, weight and the
                     # direct reading goal_rate = share of planning events whose chosen
                     # goal moment carried a consequence (0 = never, 1 = every plan)
                     "consequence": ({k2[3:]: v2 / max(1, acc.get("cqN_" + k2[3:], 0))
                                      for k2, v2 in acc.items() if k2.startswith("cq_")}
                                     if any(k2.startswith("cqN_") and v2 for k2, v2 in acc.items())
                                     else None),
                     "jitter": acc.get("jitter", 0.0) / n,
                     "roll_deg_mean": acc["roll"] / n, "roll_deg_max": acc["roll_max"],
                     "rolled_over_fraction": acc["rolled"] / n,
                     # SELF-TOUCH (2026-09-21). The three counts are window means; the
                     # audit is the newest one measured, and it is measured only on the
                     # steps that also compute the shuffled-action control.
                     "self_contact_patches": (acc["sc_patches"] / acc["sc_n"]) if acc.get("sc_n") else None,
                     "self_contact_pairs": (acc["sc_pairs"] / acc["sc_n"]) if acc.get("sc_n") else None,
                     "self_contact_force_share": (acc["sc_share"] / acc["sc_n"]) if acc.get("sc_n") else None,
                     "self_touch_audit": getattr(model, "last_self_touch_audit", None),
                     # the newest one that had its shuffled-action control beside it: this
                     # is what the wellness gate reads, so it is what the dashboard shows
                     "self_touch_controlled": getattr(model, "last_self_touch_controlled", None),
                     # where the bodies are being touched, by region: window means per body,
                     # and their mean over the bodies (it used to be the LAST body's report at
                     # the window's last step)
                     **(lambda _m, _pb: {"skin_regions": _m, "skin_regions_per_body": _pb})(
                         *skin_regions_window(acc.get("skr") or {}, acc.get("skr_n") or {},
                                              args.n_envs)),
                     # self-contact onsets from the world's report (absent = no report)
                     **(lambda _se: {"self_contact_events": _se} if _se is not None else {})(
                         self_contact_events_window(acc)),
                     # the contact head and the contact_control channel (--contact-head on)
                     **({"contact_control": model.contact_control_window_summary()}
                        if getattr(model, "contact_head", None) is not None else {}),
                     # what the live view is charging the run: milliseconds per rendered
                     # frame and how many frames this window paid for
                     # frames and state files published this window; their cost is inside
                     # the step the worker was taking anyway, so it shows up in steps/s
                     "live_view_frames": int(acc.get("live_n", 0)),
                     "live_state_writes": int(acc.get("lstate_n", 0)),
                     "task_mobile_hand_gap_min_m": (float(acc["mobile_gap_min"])
                                                    if np.isfinite(acc.get("mobile_gap_min", np.inf))
                                                    else None),
                     "vision_fresh_fraction": acc["vis_fresh"] / n,
                     "cmd_response_cos_held": (acc["cmd_slow"] / acc["cmd_slow_n"]) if acc.get("cmd_slow_n") else None,
                     "wellness_align_err": (acc["walign"] / acc["walign_n"]) if acc.get("walign_n") else None,
                     **{f"task_{k}": (acc[k] / acc[k + "_n"]) for k in
                        ("hand_view", "hand_seen", "hand_fov", "eye_pitch", "unseen_s",
                         "rescues", "snaps", "saccades", "crossed", "pressing",
                         "press_kpa", "reflex_on", "mobile_speed", "mobile_hit",
                         "mobile_hand", "mobile_hand_gap",
                         "ring_snaps", "squeaker_hits", "squeaker_foot",
                         "dragonfly_deg", "dragonfly_darts", "dragonfly_hover",
                         "bee_deg", "antic_ok", "antic_speed",
                         "squeaker_view", "head_tilt", "squeaker_seen",
                         "headlook") if acc.get(k + "_n")},
                     "motor_discrimination_ratio": _discrimination(motor_ring) if args.n_envs > 1 else None,
                     # review T18: _discrimination_parts re-stacked the 200-frame ring
                     # twice per flush; once now
                     **(lambda _dp: {"motor_discrimination_between": _dp[0],
                                     "motor_discrimination_within": _dp[1]})(
                         _discrimination_parts(motor_ring) if args.n_envs > 1 else (None, None)),
                     "vicreg": acc["bd"].get("vicreg", 0.0) / n,
                     # item 4: per-sense window mean of the gate's blend size (was a
                     # flush-step snapshot); tenure_events counts order endings
                     "td_gate": ({k2[3:]: v2 / max(1, acc.get("gtN_" + k2[3:], 0))
                                  for k2, v2 in acc.items() if k2.startswith("gt_")}
                                 if any(k2.startswith("gtN_") and v2 for k2, v2 in acc.items())
                                 else None),
                     "tenure_events": int(acc.get("tn_n", 0)),
                     # the horizon license: per-depth beat-do-nothing win EMAs (an
                     # EMA is already a smooth window quantity - snapshot is exact)
                     "plan_license": (dict(getattr(model, "last_plan_license", None) or {})
                                      or None),
                     # share of LATCHED (executed) orders by author - the direct
                     # gauge for whether babble reaches the body at all; None when
                     # the running model does not carry the authorship stash
                     "exec_source": ({k: round(v / sum(exec_src.values()), 4)
                                      for k, v in exec_src.items()}
                                     if sum(exec_src.values()) else None),
                     # size of the orders actually latched: mean strength (0-1)
                     # and mean positional step being asked for (cm), over the body
                     # parts (eyes' row left out) and hands-only (handles 0-11) - the
                     # reach question, measured
                     "cmd_exec": ({"strength_mean": round(cmdg["strength"] / cmdg["n"], 4),
                                   "strength_hands": round(cmdg["strength_hands"] / cmdg["n"], 4),
                                   "pos_step_cm": round(cmdg["step_cm"] / cmdg["n"], 3),
                                   "pos_step_cm_hands": round(cmdg["step_cm_hands"] / cmdg["n"], 3)}
                                  if cmdg["n"] else None),
                     # memory, commit charge, card memory (all processes) and step time,
                     # every window: the 14:16 slowdown of segment 31 had none of it
                     "machine": machine.sample(model_ms=1000.0 * acc["t_model"] / n,
                                               env_ms=1000.0 * acc["t_env"] / n,
                                               causal_ms=(1000.0 * acc.get("t_causal", 0.0) / n
                                                          if causal is not None else None),
                                               wellness=(model.wellness_channels_cost_window()
                                                         if hasattr(model, "wellness_channels_cost_window")
                                                         else None),
                                               efference_ms=(model.efference_lesson_ms()
                                                             if getattr(model, "efference", None)
                                                             is not None else None)),
                     # absent when the feature is off (absent = unmeasured, never 0)
                     **({"causal_chain": causal.window_summary()} if causal is not None else {}),
                     **({"efference": model.efference_window_summary()}
                        if getattr(model, "efference", None) is not None else {}),
                     # the motor synergy (--motor-synergy on): its hindsight lesson, babble in z,
                     # the planner's z and the turn every order asked at issue
                     **({"synergy": model.synergy_window_summary()}
                        if getattr(model, "synergy", None) is not None else {}),
                     # the whole model's liveness (--liveness-repair report|on, DESIGN section 10)
                     **({"liveness": model.liveness.window_summary(model)}
                        if getattr(model, "liveness", None) is not None else {}),
                     # the asked-change path beside the motor encoder (--motor-asked-change on): its output's
                     # size in token units, how evenly the handles share it, its hidden layer's live share
                     **({"motor_asked_change": model.motor_asked_change.window_summary()}
                        if getattr(model, "motor_asked_change", None) is not None else {}),
                     # the all-slot paths beside the prediction heads (--heads-all-slot-path on): each path's
                     # output size, its reach over the 6 slots, its hidden layer's live share, and head_read (how
                     # evenly each head's whole prediction answers to the 6 slots)
                     **({"heads_all_slot_path": model.heads_all_slot_path_report()}
                        if getattr(model, "heads_all_slot_path", None) is not None else {}),
                     # the linear input path beside the vestibular encoder (--vestibular-input-path on): its
                     # output's size in token units and each input field group's reach
                     **({"vestibular_input_path": model.vestibular_input_path.window_summary()}
                        if getattr(model, "vestibular_input_path", None) is not None else {}),
                     **({"vestibular": vest_win.summary()} if vest_win is not None else {}),
                     # wellness channels (DESIGN D6-D8): each only when its flag is on
                     **({"controllability": model.controllability_window_summary()}
                        if getattr(model, "controllability_on", False) else {}),
                     **({"learning_progress": model.learning_progress_window_summary()}
                        if getattr(model, "learning_progress_on", False) else {}),
                     **({"wellness_rank": model.wellness_rank_window_summary()}
                        if getattr(model, "wellness_rank_mode", "raw") == "normalized" else {}),
                     # the next-moment lesson's per-sense weights (--next-frame-norm persistence)
                     **({"next_frame_norm": model.next_frame_norm_window_summary()}
                        if getattr(model, "nf_norm", None) is not None else {}),
                     # the hand press split by what pressed it (the world's report, env.py
                     # split_hand_press): share of body-moments over the 2 kPa line and the mean
                     # firmest press, world part and own-body part. Absent when no report carried
                     # the split (absent = unmeasured)
                     **({"hand_press": {
                         "world_share": round(acc["hp_world"] / acc["hp_n"], 4),
                         "self_share": round(acc["hp_self"] / acc["hp_n"], 4),
                         "world_kpa": round(acc["hp_world_kpa"] / acc["hp_n"], 3),
                         "self_kpa": round(acc["hp_self_kpa"] / acc["hp_n"], 3)}}
                        if acc.get("hp_n") else {}),
                     # the motor interface audit's limb fixes (--parent-relative-targets,
                     # --reach-projection): how far the latched orders were moved to be performable;
                     # absent when both are off
                     **({"limb_targets": limb_window_summary(acc["limb"])} if acc.get("limb") else {})}
            for _k in list(exec_src):    # zero in place: rebinding would shadow
                exec_src[_k] = 0         # the outer name in this nested scope
            for _k in list(cmdg):
                cmdg[_k] = 0 if _k == "n" else 0.0
            for r in (rec_i, rec_t, rec_m):
                sf.write(json.dumps(r) + "\n")
            sf.flush()
            if rec_m.get("peak_vram_mb") is not None:
                torch.cuda.reset_peak_memory_stats()     # peak_vram_mb is per window
            # The log said "pe" and showed CURRENT-frame reconstruction error. That is not
            # evidence about what the action did, and reading it as such is how a body
            # that had learned nothing about its own actions looked like it was learning.
            # Both are shown, each named for what it is.
            _cz = rec_m.get("causal_pe")
            # `pe` is the heads' guess of the NEXT instant scored against the present:
            # PREDICTED CHANGE, not reconstruction. Labelling it recon_pe is what made the
            # watchdog, the dashboard and the overseer all report a broken reconstruction
            # on H34 while its next-state error was the best on record (2026-08-27). The
            # decoders' error is recon_dec, logged separately.
            log.info("step %d | %s | loss %.4f | predicted change %s | causal_pe %s | exec %d | "
                     "%.1f model steps/s x %d envs | age %s | contacts %.1f | fresh vision %.0f%%",
                     step, dev["phase_name"], rec_t["avg_loss"],
                     {k: round(v, 4) for k, v in rec_m["pe"].items()},
                     ("n/a" if _cz is None else round(_cz, 5)),
                     int(getattr(res, "exec_seen", 0)),
                     fps, args.n_envs, kw.get("age"), rec_m["contacts_per_env"],
                     100 * rec_m["vision_fresh_fraction"])
            for k in acc:
                acc[k] = {} if isinstance(acc[k], dict) else 0.0
            acc["nan"] = 0
            # a CLOSEST-APPROACH accumulator cannot start a window at 0.0: the generic
            # reset above zeroes every number, and min(0.0, anything) is 0.0, so the gauge
            # would report a touch in every window after the first one it saw
            acc["mobile_gap_min"] = float("inf")

        def save_ckpt(reason):
            # review T9: snapshot_and_stop saved the identical weights twice (overseer
            # then end, 0.08 s apart on H60) - same folder, two registry rows
            if ckpt_list and ckpt_list[-1]["step"] == step:
                log.info("checkpoint at step %d already saved (%s) - %s save skipped",
                         step, ckpt_list[-1]["reason"], reason)
                return
            def write_episode(staging_path,final_path):
                if not episode_boundary or reason in ('crash','repeated_physical_fault'):return
                continuation_state.save(staging_path,src,model,dict(step=step,held_motor=held_motor,
                    held_hand_joint_goal=held_hand_joint_goal,prev_motor_substeps=prev_motor_substeps,
                    prev_hand_joint_goal=prev_hand_joint_goal,prev_action_contract=prev_action_contract,
                    prev_executed=prev_executed,prev_order_id=prev_order_id,prev_ctrl_gaze=prev_ctrl_gaze,
                    order_seq=order_seq,
                    order_age=order_age,prev_gaze_substeps=prev_gaze_substeps,prev_decoded_ctrl=prev_decoded_ctrl,
                    bout_on=bout_on,bout_until=bout_until,bout_ema=bout_ema,bout_rng=bout_rng.bit_generator.state,
                    prev_ach=prev_ach,prev_bout=prev_bout,rest_speed_ema=rest_speed_ema,bout_speed_ema=bout_speed_ema,
                    follow_bout_ema=follow_bout_ema,follow_rest_ema=follow_rest_ema,
                    last_physical_times=last_physical_times,ring_clean=ring_clean,ring_need=ring_need,
                    pendings=pendings,frame_meta=[f.meta for f in frames],
                    acting_snapshot=None if actor is None else actor.state_dict()),continuation_config,
                    published_path=final_path,
                    storage_root=(str(Path(args.caregiver_record_dir)/'checkpoint_state'/Path(out).name)
                                  if args.caregiver_record_dir else None))
            p = model.save_checkpoint(step, curriculum_state=None if cm is None else {"phase": cm.phase_name, "phase_step": cm.phase_step},
                                      companion_writer=write_episode)
            ckpt_list.append({"step": step, "path": p, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": reason,
                              "elapsed_min": (time.time() - t_start) / 60.0})
            # review T8: the file used to be rewritten from this segment's list alone,
            # so it forgot earlier segments' saves and kept listing deleted ones
            try:
                _prior = json.load(open(os.path.join(out, "checkpoints.json")))
            except (OSError, ValueError):
                _prior = []
            _seen = {e["path"] for e in ckpt_list}
            _merged = [e for e in _prior if e.get("path") not in _seen] + ckpt_list
            _merged = [e for e in _merged if e.get("path") and os.path.isdir(e["path"])]
            json.dump(_merged, open(os.path.join(out, "checkpoints.json"), "w"), indent=2)
            # lineage: everything needed to know what this snapshot IS, kept inside the
            # snapshot folder itself so it stays meaningful even when copied elsewhere
            lineage = {
                "run": os.path.basename(out), "out": out, "step": step, "reason": reason,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "phase": cm.phase_name if cm is not None else "fixed",
                "code_commit": git_commit,
                "resumed_from": args.resume,
                "flags": {k: getattr(args, k, None) for k in
                          ("n_envs", "eval_only", "limp", "no_babble", "babble_scale", "babble_noise", "random_actions",
                           "replay", "scene", "start_phase", "minutes")},
            }
            try:
                json.dump(lineage, open(os.path.join(p, "lineage.json"), "w"), indent=2)
                reg = os.path.join(out, "..", "..", "weights_registry.jsonl")
                with open(os.path.abspath(reg), "a", encoding="utf-8") as rf:
                    rf.write(json.dumps({**lineage, "path": p}) + "\n")
            except OSError:
                log.warning("could not write lineage for %s", p)
            log.info("checkpoint saved (%s): %s", reason, p)

        # rest-bouts (queue attempt #3, launched 2026-09-07): babble arrives in motion
        # BOUTS with real rests, like an infant, instead of every step forever.
        # Bout 2-5 s, rest 1-3 s (infant_reference bands) -> scheduled duty
        # 3.5/(3.5+2) = 64%, inside the infant 40-75% active-alert band. Measured
        # basis, three independent lines: the motion census (duty 100%, zero pauses,
        # un-infant-like); the capacity test (a never-resting body makes its own
        # future unpredictable for ANY same-size forward model - rest creates
        # learnable stretches); the attribution test (aimed actions can only show
        # their effect against a quiet baseline). Batch-uniform: dev is per-batch.
        bout_on, bout_until, bout_ema = True, 0, 1.0
        bout_rng = np.random.default_rng(int(args.seed) + 4242)
        _tsec = _transition_seconds(args, env_dt)   # one model step, seconds (measured dt)
        # BODY-SIDE stillness accounting (the rest claim is judged on the body, never
        # on the schedule - operator 2026-09-07 after the schedule was reported as
        # rest): achieved-pose speed per step, split by whether the JUST-FINISHED
        # burst was a bout or a rest. The unit-free ratio rest/bout is the gate.
        prev_ach, prev_bout = None, True
        rest_speed_ema, bout_speed_ema = None, None
        # obedience split by burst class: during rest the EXECUTED command is the
        # pose-hold (asked displacement ~0 -> the follow cosine is noise/negative by
        # construction, measured -0.169 window-mean at 36% rest), so the obedience
        # kill line must judge BOUT bursts only - and stays at its original 0.02
        follow_bout_ema, follow_rest_ema = None, None
        from .episode_continuity import physical_fault, scheduled_reset, PhysicalFaultBudget
        last_physical_times = None
        physical_fault_budget = PhysicalFaultBudget()
        episode_boundary = True
        if episode_resume is not None:
            st=episode_resume['trainer']
            if st['step']!=step:raise ValueError('world and trained checkpoint steps disagree')
            held_motor=st['held_motor'];prev_executed=st['prev_executed'];prev_order_id=st['prev_order_id']
            prev_ctrl_gaze=st.get('prev_ctrl_gaze')     # companions saved before it was kept hold none
            held_hand_joint_goal=st.get('held_hand_joint_goal')
            prev_motor_substeps=st.get('prev_motor_substeps')
            prev_hand_joint_goal=st.get('prev_hand_joint_goal')
            prev_action_contract=st.get('prev_action_contract','world_v0')
            order_seq=st['order_seq'];order_age=st['order_age']
            prev_gaze_substeps=st['prev_gaze_substeps'];prev_decoded_ctrl=st['prev_decoded_ctrl']
            bout_on=st['bout_on'];bout_until=st['bout_until'];bout_ema=st['bout_ema']
            bout_rng.bit_generator.state=st['bout_rng'];prev_ach=st['prev_ach'];prev_bout=st['prev_bout']
            rest_speed_ema=st['rest_speed_ema'];bout_speed_ema=st['bout_speed_ema']
            follow_bout_ema=st['follow_bout_ema'];follow_rest_ema=st['follow_rest_ema']
            last_physical_times=st['last_physical_times'];pendings=st['pendings']
            for f,meta in zip(frames,st.get('frame_meta',[f.meta for f in frames])):
                f.meta=continuation_state.tree(meta)
            ring_clean=st['ring_clean'];ring_need=st['ring_need']
            # THE VESTIBULAR SLOT (DESIGN D4): a companion saved by the intermind build holds
            # state keyed by that sense's name; move it before it is restored (mimo_bridge/
            # vestibular_sense.migrate_learner_state says what is cleared and what is kept)
            _vest_episode = None
            if getattr(model, "vestibular_sense", False):
                from .vestibular_sense import migrate_learner_state
                _vest_episode = migrate_learner_state(episode_resume['learner'], model)
            continuation_state.restore_learner(model,episode_resume['learner'])
            log.info('restored physical episode, executed orders, memory, replay and RNG at step %d',step)
            if _vest_episode is not None:
                log.warning("vestibular sense: the episode companion was saved by the intermind "
                            "build; moved before restoring: %s", _vest_episode)
                sf.write(json.dumps({"type": "vestibular_switch", "timestamp": time.time(),
                                     "step": int(step), "part": "episode",
                                     "episode": _vest_episode}) + "\n")
                sf.flush()
        # AN EYE ROW CONVENTION SWITCH (--eye-row-convention; the motor fixes deploy review, finding 3b):
        # every ledger eye row stored before it - the replay ring, the short histories, and the ledger
        # of the transition that ended at the checkpoint - is rewritten as the new convention's row for
        # the same eye command, after everything is restored and before the first step reads any of it
        prev_executed, _eye_rows = switch_stored_eye_rows(model, prev_executed)
        if _eye_rows is not None:
            # the command beside the last transition before the switch was recorded under the old convention:
            # dropped with its ledger's rewrite, so that order (executed before the switch) is back-filled
            # without the eye row's replay contract bit (LemLite._eye_contract_bit; the fourth review's fix round)
            prev_ctrl_gaze = None
            sf.write(json.dumps({"type": "motor_interface_stored_rows", "timestamp": time.time(),
                                 "step": int(step), "eye_rows_rewritten": _eye_rows,
                                 "readers": {k: LemLite.STORED_EYE_ROW_READERS.get(k, "unlisted")
                                             for k in _eye_rows},
                                 "encoded_not_rewritten": LemLite.ENCODED_EYE_ROW_CARRIERS,
                                 # the order centre's eye entries and the repaired memory's running mean of
                                 # events, each left as saved (LemLite.CENTRE_LEFT_AS_SAVED)
                                 "order_centre_eye_shift": (getattr(model, "motor_interface_conversion", None)
                                                            or {}).get("order_centre_eye_shift"),
                                 "event_centre_eye_shift": (getattr(model, "motor_interface_conversion", None)
                                                            or {}).get("event_centre_eye_shift"),
                                 # what replay studies from here (the eye row's replay contract)
                                 "replay_contract": eye_replay_contract_record(model)}) + "\n")
            sf.flush()
        if args.self_choice in ('on','shadow'):
            from .tools.actor_runtime import actor_for_run
            configuration=acting_configuration(args)
            actor_args=argparse.Namespace(**vars(args));actor_args.self_choice='off';actor_args.self_choice_receipt=''
            factory=lambda:build_learner(actor_args,None)
            saved_actor=None if episode_resume is None else episode_resume['trainer'].get('acting_snapshot')
            if saved_actor is not None and args.retire_acting_snapshot:
                log.info('saved acting snapshot RETIRED by explicit launch flag: its recorded '
                         'decisions remain in the source run\'s stats; a fresh candidate now '
                         'qualifies under the current configuration and code')
                saved_actor=None
            actor=actor_for_run(model,factory,configuration,args.self_choice,validation.receipt,saved=saved_actor)
            model.self_intention=None
            log.info('frozen acting snapshot retained at step %d; qualified=%s; learner updates continue separately',
                     actor.born_step,not actor.qualification)
        elif episode_resume is not None and episode_resume['trainer'].get('acting_snapshot') is not None:
            raise ValueError('cannot discard an acting snapshot during exact episode continuation')
        if model.hand_target_mode != 'world':
            hand_states = src.hand_state()
            if not hand_states or any(s is None for s in hand_states):
                raise ValueError('Learner and worlds disagree on the hand-target contract')
            geometry = hand_states[0]['geometry']
            if any(s['geometry'] != geometry for s in hand_states):
                raise ValueError('All shared-learner bodies must use identical hand geometry')
            initial = np.stack([s['current'] if s['held'] is None else s['held'] for s in hand_states])
            for frame, hand_state in zip(frames, hand_states):
                frame.meta['hand_joint_angles'] = hand_state['current']
                frame.meta['hand_joint_velocities'] = hand_state['velocity']
            model.configure_hand_geometry(geometry,
                initial if model._hand_previous_joints is None else None)
            if actor is not None:
                actor.model.configure_hand_geometry(geometry,
                    initial if actor.model._hand_previous_joints is None else None)
            if getattr(model, "hand_calibration_replacement", None) is not None:
                # --replace-saved-hand-calibration replaced the checkpoint's saved hand calibration at this launch's
                # load, and configure_hand_geometry just checked the worlds' hand geometry against it: the record
                # (what replaced what, under which admission record, the launch, the time) into the stats stream
                sf.write(json.dumps({"type": "hand_calibration_replacement", "timestamp": time.time(),
                                     "step": int(step), "replacement": model.hand_calibration_replacement_report()},
                                    default=str) + "\n")
                sf.flush()
            if held_hand_joint_goal is None:
                # Issue a versioned hand goal at this boundary. Physical state,
                # learner, optimizer and past experience are retained.
                held_motor = None
        # the calf rows the worlds record (--parent-relative-targets calf or calf_chest): the planner
        # imagines them too (limb_target_control.calf_rows_torch; review of 2026-09-25, finding 5).
        # None with the calves off, set at every launch so no earlier setting lingers
        _calf_geo = planner_calf_geometry([f.meta for f in frames], getattr(args, 'parent_relative_targets', 'off'))
        model.configure_calf_projection(_calf_geo)
        if actor is not None:
            actor.model.configure_calf_projection(_calf_geo)
        while time.time() < deadline and (args.max_steps <= 0 or step < args.max_steps):
            if step % 25 == 0:
                poll_control()
            if overseer_stop:
                log.info("stopping on overseer command (%s)", overseer_stop)
                break
            fault=physical_fault(frames,last_physical_times)
            if fault:
                sf.write(json.dumps(dict(type='episode_reset',step=step,timestamp=time.time(),reason=fault))+'\n');sf.flush()
                if physical_fault_budget.record(step):
                    save_ckpt('repeated_physical_fault')
                    # writer: the dashboards name the program that wrote a stop (lineage_series.stop_writer)
                    Path(out,'stop_reason.json').write_text(json.dumps(dict(
                        resume_policy='repair_required',reason=fault,timestamp=time.time(),
                        writer='mimo_bridge/train_mimo.py: repeated physical faults')),encoding='utf-8')
                    raise RuntimeError('repeated physical faults require repair: '+fault)
                frames=src.reset();model.reset_transition_state()
                if actor is not None:actor.reset_transition_state()
                if causal is not None:causal.clear('physical fault reset')
                if model.movement is not None: model.movement.reset()
                held_motor=prev_executed=prev_order_id=prev_ctrl_gaze=None
                pendings=[];_prev_active=[];last_physical_times=None
                continue
            last_physical_times=[float(f.sim_time) for f in frames]
            episode_boundary = False
            dev = cm.get_dev_limits() if cm is not None else model.dev_limits(step)
            from .body_layout import extend_limits
            dev = extend_limits(dev, len(model.handle_names))
            if args.no_babble:
                dev = {**dev, "exploration_noise": 0.0}
            elif args.babble_noise is not None:
                dev = {**dev, "exploration_noise": float(args.babble_noise)}
            elif args.babble_scale != 1.0:
                dev = {**dev, "exploration_noise": dev.get("exploration_noise", 0.0) * args.babble_scale}
            if args.babble_bouts in ("on", "real") and not args.no_babble:
                if step >= bout_until:
                    bout_on = not bout_on
                    dur_s = (bout_rng.uniform(2.0, 5.0) if bout_on
                             else bout_rng.uniform(1.0, 3.0))
                    bout_until = step + max(1, int(dur_s / _tsec))
                if not bout_on:
                    dev = {**dev, "exploration_noise": 0.0}
            bout_ema = 0.99 * bout_ema + 0.01 * (1.0 if bout_on else 0.0)
            t0 = time.perf_counter()
            planning_model = model if actor is None or actor.qualification else actor.model
            planning_inputs_before = getattr(planning_model, "_last_plan_inputs", None)
            if args.corrupt_vision > 0 and np.random.random() < args.corrupt_vision:
                for f in frames:      # the Act 3 test: noise where the picture should be
                    if f.vision_patches is not None:
                        f.vision_patches = np.random.randint(0, 255, size=f.vision_patches.shape,
                                                             dtype=f.vision_patches.dtype)                             if f.vision_patches.dtype == np.uint8 else                             np.random.random(f.vision_patches.shape).astype(f.vision_patches.dtype)
            res = model.step(frames, dev, train=not args.eval_only, step_index=step,
                             random_actions=args.random_actions, executed_action=prev_executed,
                             executed_ctrl_gaze=prev_ctrl_gaze,
                             executed_order_id=prev_order_id)
            if comparison is not None:
                # record c's line of this model step, and record d's event when the step fired one: its inputs, then
                # the weights after this update (comparison_records.snapshot_weights, into this launch's own folder of
                # the records, vision_spike/checkpoints/<launch>, where no retention rule removes a snapshot)
                comparison.after_step(step, model, res, snapshot=lambda k: snapshot_weights(
                    model, k, comparison.weights_folder))
            if getattr(model, "persistent_memory_switch_unwritten", None) is not None:
                # --persistent-memory-repair switched the memory during this step: its record (the fold's
                # largest changes against the line, the ring check, the way back) in the stream at once
                sf.write(json.dumps({"type": "persistent_memory_switch", "timestamp": time.time(),
                                     "step": int(step), "part": "switch",
                                     "switch": model.persistent_memory_switch_unwritten}) + "\n")
                sf.flush()
                model.persistent_memory_switch_unwritten = None
            actor_result=None
            if actor is not None:
                actor_result=actor.step(frames,dev,step_index=step,executed_action=prev_executed,
                                        executed_order_id=prev_order_id)
                # Reports retain the learner's losses; action fields identify
                # the actor that actually proposes movement to these bodies.
                from dataclasses import replace
                if not actor.qualification:
                    res=replace(res,motor_out=actor_result.motor_out,motor_for_action=actor_result.motor_for_action,
                        hand_joint_goal=actor_result.hand_joint_goal,hand_target_contract=actor_result.hand_target_contract,
                                gaze_out=actor_result.gaze_out,motor_confidence=actor_result.motor_confidence)
                actor.last['applied_to_body']=not actor.qualification
                model.last_self_state['choice']=dict(actor.last)
            if causal is not None:
                # the moment the pending predictions were made for has arrived: grade them
                causal.after_step(model, step)
            if vest_win is not None:
                # the frames the model just read (the burst below rebinds `frames`)
                vest_win.add(frames, model)
            if args.babble_bouts != "off" and model.last_achieved is not None:
                # the pose change just measured happened during the PREVIOUS burst,
                # so it is credited to that burst's bout/rest state
                # BODY handles only - handle 23 is the GAZE, and eyes stay live
                # during rest BY DESIGN, so including it turned the stillness
                # ratio into an eye-movement ratio once the body truly held
                # (measured 2026-09-08: gauge 0.92 while a direct env replay of
                # the same hold read body speed 0.008-0.015 cm/step, ~100x
                # stiller than bouts, at every strength)
                _body_ids = [i for i, name in enumerate(model.handle_names) if name != 'gaze']
                _ach = model.last_achieved[:, _body_ids, :3].detach().float().cpu().numpy()
                if prev_ach is not None and prev_ach.shape == _ach.shape:
                    _spd = float(np.linalg.norm(_ach - prev_ach, axis=-1).mean())
                    if prev_bout:
                        bout_speed_ema = (_spd if bout_speed_ema is None
                                          else 0.98 * bout_speed_ema + 0.02 * _spd)
                    else:
                        rest_speed_ema = (_spd if rest_speed_ema is None
                                          else 0.98 * rest_speed_ema + 0.02 * _spd)
                prev_ach = _ach
            # close the PREVIOUS moment's record: its cause has just arrived
            if pendings and prev_executed is not None:
                ep = model.replay._episode_id if model.replay is not None else 0
                closed, ex = [], np.asarray(prev_executed)
                for bi, pend in enumerate(pendings):
                    if pend is None or bi >= len(frames):
                        continue
                    try:
                        closed.append(pend.complete(
                            executed_action=ex[bi], state_t1={"frame": int(step)},
                            frame_t1=int(step), duration_s=_tsec,
                            gaze_substeps=(None if prev_gaze_substeps is None
                                           else prev_gaze_substeps[bi]),
                            order_id=prev_order_id,
                            decoded_ctrl=(None if prev_decoded_ctrl is None
                                          else prev_decoded_ctrl[bi]),
                            motor_substeps=None if prev_motor_substeps is None else prev_motor_substeps[bi],
                            hand_joint_goal=None if prev_hand_joint_goal is None else prev_hand_joint_goal[bi],
                            action_contract=prev_action_contract,
                            episode_id=ep))
                    except (ValueError, TypeError) as e:
                        # a record that cannot be true is dropped, loudly, not stored
                        # (review T17: a mixed None/array decoded_ctrl batch raises
                        # TypeError inside validate; only ValueError was caught)
                        log.warning("dropped a transition for body %d: %s", bi, e)
                if closed and step % (args.report_every * 10) == 0:
                    g = closed[0].gaze_summary()
                    log.info("transition ledger: %d/%d bodies closed a record; "
                             "gaze path %.1f deg over %d substeps",
                             len(closed), len(frames), g["path_deg"], g["n"])
            ep_now = model.replay._episode_id if model.replay is not None else 0
            observe_shadow(args, model, frames,
                           closed if pendings and prev_executed is not None else (), step, ep_now)
            pendings = [Pending(env_id=bi, episode_id=ep_now, frame_t=int(step),
                                state_t={"frame": int(step)},
                                achieved_t=(model.last_achieved[bi].detach().float().cpu().numpy()
                                            if model.last_achieved is not None else None),
                                context_t=None,
                                proposed_action=res.motor_out[bi])
                        for bi in range(len(frames))]
            if args.limp:
                res.motor_for_action[..., 8] = 0.0    # zero strength -> zero torque everywhere
            t1 = time.perf_counter()
            if args.probe_touch and step % args.probe_every == 0 and "x" in captured:
                x = captured["x"].float().cpu().numpy()           # [B, 256, d]
                _W = {"light": 1, "medium": 2, "heavy": 3}
                for b in range(x.shape[0]):
                    bt = (frames[b].meta.get("task", {}) or {}).get("box_touch") or []
                    wcls = _W.get(bt[0], 0) if len(bt) == 1 else 0
                    area = frames[b].patches[:, 3]
                    for pi in np.nonzero(area > 0)[0]:
                        probe_rows.append((step, int(pi), x[b, pi].astype(np.float16), wcls))
            # Render on the LAST frame of the burst only - the one the model reads. On a
            # fixed modulus the two never lined up: render_every=2 with action_repeat=5 drew
            # the eyes 5 times per 10 frames, of which the model saw 2 and only 1 was fresh,
            # so every other model step re-used the previous vision token. Now every render
            # is used and every step gets a fresh image, for FEWER renders than before.
            reps = max(1, args.action_repeat)
            # One ORDER, held for motor_hold model steps. The model still senses, predicts
            # and learns every step - only the limb target is kept still long enough for a
            # limb to reach it. Gaze is NOT held: the eyes are fast and the saccade
            # generator is meant to move them freely.
            hold = max(1, args.motor_hold)
            latched_now = False          # the causal-chain recorder reads it below
            if step % hold == 0 or held_motor is None:
                latched_now = True
                held_motor = res.motor_for_action
                held_hand_joint_goal = res.hand_joint_goal
                order_seq += 1           # a NEW order, even if numerically identical
                order_age = 0
                # who wrote the order now in force, for the developmental observer's provenance
                # (the order is held for motor_hold steps; a later step's proposal is not it)
                held_author = ("acting_snapshot" if actor is not None and not actor.qualification
                               else getattr(model, "last_action_source", None) or "unknown")
                _src = getattr(model, "last_action_source", None)
                if _src is not None:      # absent on models without the stash: leave unrecorded
                    exec_src[_src] = exec_src.get(_src, 0) + 1
                _cx = cmd_exec_sample(held_motor,
                                      (model.last_achieved.detach().float().cpu().numpy()
                                       if model.last_achieved is not None else None),
                                      model.handle_names)
                cmdg["n"] += 1
                cmdg["strength"] += _cx["strength"]
                cmdg["strength_hands"] += _cx["strength_hands"]
                if _cx["step_cm"] is not None:
                    cmdg["step_cm"] += _cx["step_cm"]
                    cmdg["step_cm_hands"] += _cx["step_cm_hands"]
            else:
                order_age += 1
            send_motor = held_motor
            scheduled_hold = args.babble_bouts == "real" and not bout_on and args.rest_control == "scheduled"
            if scheduled_hold:
                # REAL rest ('babble-bouts real', 2026-09-07): during a rest window the
                # EXECUTED command is replaced by a postural hold (built just below), so
                # the body settles instead of the planner replacing the babble
                # (measured: with noise-only gating the planner drove 0.531 rad in
                # rest windows, identical to bouts - zero rest happened). Eyes stay
                # live (gaze passes through); senses and every lesson run; the
                # executed-action record carries the hold that was actually sent, so the
                # state->action->state contract stays truthful. held_motor itself is NOT
                # touched - the held order and its bookkeeping survive the rest untouched.
                # Stillness is judged BODY-SIDE (rest vs bout achieved-pose speed), never
                # off this schedule.
                # REST = HOLD, NOT LIMP (measured on first launch, 2026-09-07): a
                # zero-strength null made rest windows 9.4x FASTER than bouts - the
                # body collapsed under gravity every rest and got lifted every bout
                # (the limp probes knew: even a limp body rolls 124 deg). Real rest
                # keeps muscle tone: target = the CURRENT achieved pose in command
                # centimetres AND the CURRENT achieved rotation, with strength (column 8)
                # the only field left from the model's own held order - an active
                # postural hold.
                # UNIT FIX (external review 2026-09-08 finding 1): last_achieved
                # carries positions DIVIDED by position_range_cm (200) - the
                # tokenizer's normalized form - while motor commands carry RAW
                # CENTIMETRES (lem_model.py:64, motor_for_action contract). The
                # original copy sent a hand at 20 cm a target of 0.1 cm: the review's
                # isolated decoder replay read -0.606 actuator command (scale -1..1)
                # where a correct hold reads 0 - every rest window pulled limbs
                # toward near-origin instead of holding them.
                # FIELD SEMANTICS VERIFIED AT THE WRITER (2026-09-08 second pass):
                # datasource.motor_to_echo fills the motor slot with ACHIEVED state
                # since 2026-08-20 - echo[0:3] = bones[4:7] (achieved world position,
                # cm), echo[3:7] = bones[0:4] (achieved world quat) - so
                # last_achieved x200 IS the body's true pose in command centimetres
                # (the tokenizer's "Target position" comment names the wire field,
                # not today's contents; the wire NAME is the spec-20.6 hazard, the
                # data is achieved). A first post-fix window read rest/bout 1.56 -
                # judged PENDING: it straddled two relaunch seams, the measured
                # EMA warm-up class (10.06 -> 0.56 washout); the first clean window
                # decides, and above 1 there means a real convention hunt.
                send_motor = np.array(held_motor, copy=True)
                if model.last_achieved is not None:
                    _ach_now = model.last_achieved.detach().float().cpu().numpy()
                    n_h = min(send_motor.shape[1], _ach_now.shape[1])
                    send_motor[:, :n_h, 0:3] = (_ach_now[:, :n_h, 0:3]
                                                * 200.0)   # == COMMAND_POSITION_RANGE_CM, asserted at import
                    _q = _ach_now[:, :n_h, 3:7]
                    _qn = np.linalg.norm(_q, axis=-1, keepdims=True)
                    _ok = (_qn > 1e-6).squeeze(-1)
                    send_motor[:, :n_h, 3:7] = np.where(
                        _ok[..., None], _q / np.maximum(_qn, 1e-6),
                        send_motor[:, :n_h, 3:7])
            send_hand_joint_goal = held_hand_joint_goal
            if scheduled_hold and model.hand_target_mode != 'world':
                send_hand_joint_goal = np.stack([f.meta['hand_joint_angles'] for f in frames])
                send_hand_joint_goal = model.hand_kinematics.project_joints(
                    torch.as_tensor(send_hand_joint_goal)).numpy()
                send_motor = model.hand_kinematics.assemble_world(torch.as_tensor(send_motor),
                    torch.as_tensor(send_hand_joint_goal), position_scale=1.).numpy()
            prev_bout = bout_on          # the burst below runs under this state
            if getattr(model, "synergy", None) is not None:
                if latched_now:
                    model._syn_latched_source = ("acting_snapshot" if actor is not None and not actor.qualification
                                                 else getattr(model, "last_action_source", None) or "unknown")
                model._syn_execution = {
                    "order_id": order_seq,
                    "source": "external_rest" if not bout_on else model._syn_latched_source,
                }
            _t_causal = 0.0
            if causal is not None:
                # before the burst: the order about to be sent is predicted now, and a
                # planning event of this step is stamped with whether it reached the body.
                # Timed on its own so the env's time is not charged with it.
                _tc = time.perf_counter()
                causal.on_latch(model, step, latched=latched_now, order_seq=order_seq,
                                send_motor=send_motor, gaze_out=res.gaze_out,
                                source=getattr(model, "last_action_source", None))
                _t_causal = time.perf_counter() - _tc
                acc["t_causal"] = acc.get("t_causal", 0.0) + _t_causal
            record_planning_decision(acc, planning_model, planning_inputs_before, step,
                latched=latched_now, source=getattr(planning_model, "last_action_source", None),
                externally_overridden=scheduled_hold or args.random_actions or args.limp)
            # EXPLAINER SNAPSHOT HOOK. Exactly here and nowhere else: after this
            # step's hold decision (held_motor/order_seq describe the order about to
            # be issued) and before the burst rebinds `frames` - so the eye images,
            # the model caches (last_tokens_fed/last_context, stashed inside step())
            # and the StepResult all describe the SAME completed step. prev_executed
            # is the action executed during the PREVIOUS interval - the one the model
            # just consumed as executed_action - and is labeled as such in the schema.
            if (explain_pending
                    or (args.explain_steps > 0 and step % args.explain_steps == 0)
                    or (args.explain_every_min > 0
                        and (time.time() - last_explain) >= args.explain_every_min * 60.0)):
                explain_pending = False
                last_explain = time.time()
                try:
                    from mimo_bridge.explainer.capture import build_snapshot
                    from mimo_bridge.explainer.writer import ExplainWriter
                    if explain_writer is None:
                        explain_writer = ExplainWriter(out, keep_mb=args.explain_keep_mb)
                    _ri = None
                    if model.replay is not None:
                        _ri = {"fill_frac": model.replay.fill_fraction(),
                               "clean_frac": model.replay.clean_fraction(),
                               "window_replay_loss": ((acc.get("replay_loss", 0.0) / acc["replay_n"])
                                                      if acc.get("replay_n") else None)}
                    snap = build_snapshot(
                        model if actor is None or actor.qualification else actor.model,
                        frames, res if actor is None or actor.qualification else actor_result,
                        run=os.path.basename(out), step=step,
                        phase=dev.get("phase_name"), dev=dev,
                        weights="frozen" if args.eval_only or (actor is not None and not actor.qualification) else "post_update",
                        body=max(0, min(int(args.explain_body), len(frames) - 1)),
                        eye_layout=([("wide", 256)]
                                    + ([("fovea", int(args.fovea_size))]
                                       if args.fovea_size > 0 else [])),
                        prev_executed=prev_executed, held_order=held_motor,
                        order_seq=order_seq, order_age=order_age,
                        age_months=kw.get("age"),
                        flags={k: getattr(args, k, None) for k in
                               ("eval_only", "intention", "next_frame_loss",
                                "top_down_gate", "residual_prediction", "motor_hold",
                                "action_repeat", "n_envs", "scene")},
                        replay_info=_ri,
                        # held_order is the order the model PROPOSED; on a scheduled rest
                        # window the body executed send_motor instead, with its position
                        # and rotation columns replaced by the achieved pose
                        **explain_rest_fields(build_snapshot, send_motor, scheduled_hold))
                    explain_writer.submit(snap)
                except Exception:
                    log.warning("explain snapshot failed (training unaffected): %s",
                                traceback.format_exc())
            burst, gaze_burst = [], []
            from .video_control import body_control
            video_control = body_control(scheduled_hold, args.random_actions)
            # What the watcher asks the stepping worker to draw for us THIS step, if
            # anything. It rides the step's own round trip: asking separately cost 19% of
            # the run's speed for a 4.3 ms render (measured 2026-09-21, segment 20).
            _live = {}
            if args.live_view_every > 0 and step % args.live_view_every == 0:
                _live["frame"] = os.path.join(out, "live", "frame.jpg")
                _live["size"] = args.live_view_size
                # RECORDING is off unless someone asked for it. The request is one small
                # file the dashboard writes; it is read at most once a second, and while it
                # says "recording" each live frame is also kept under live/rec.
                _now = time.time()
                if _now - _rec["checked"] >= 1.0:
                    _rec["checked"] = _now
                    try:
                        with open(os.path.join(out, "live", "record.json"), encoding="utf-8") as _rf:
                            _rec["on"] = str(json.load(_rf).get("state")) == "recording"
                    except (OSError, ValueError):
                        _rec["on"] = False
                    if not _rec["on"]:
                        _rec["n"] = 0
                if _rec["on"]:
                    _rec["n"] += 1
                    _live["keep"] = os.path.join(out, "live", "rec", "frame_%06d.jpg" % _rec["n"])
            if args.live_state_every > 0 and step % args.live_state_every == 0:
                _live["state"] = os.path.join(out, "live", "state.npz")
            acc["live_n"] = acc.get("live_n", 0) + (1 if _live.get("frame") else 0)
            acc["lstate_n"] = acc.get("lstate_n", 0) + (1 if _live.get("state") else 0)
            _observe = None
            if observer_lane is not None:
                # provenance of the burst about to run: a scheduled rest executes a postural hold,
                # not the held order; random/limp runs replace the learner's order entirely
                _obs_author = ("random_actions" if args.random_actions else "limp" if args.limp
                               else "external_rest" if scheduled_hold else held_author)
                _observe = {"step": int(step), "reps": int(reps), "author": _obs_author,
                            "latched": bool(latched_now), "order_seq": int(order_seq),
                            "rest": bool(scheduled_hold), "bout": bool(prev_bout)}
                observer_lane.add(step, model, author=_obs_author, latched=latched_now,
                                  order_seq=order_seq, rest=scheduled_hold, bout=prev_bout)
            for rep in range(reps):
                frames = src.step(send_motor, res.gaze_out, dev.get("handle_mask"),
                                  render=(rep == reps - 1), video_control=video_control,
                                  live=(_live if (_live and rep == reps - 1) else None),
                                  live_body=args.live_view_body,
                                  **({} if send_hand_joint_goal is None else {'hand_joint_goal':send_hand_joint_goal}),
                                  **({} if _observe is None else {"observe": dict(_observe, rep=rep)}))
                burst.append([dict(f.meta.get("task") or {}) for f in frames])
                gaze_burst.append(frames)
            # Close the ledger for this transition: what the body was actually given.
            # Gaze is NOT held - the saccade generator and the reflex can move the eyes on
            # every one of the five substeps - so the single final value is not the action
            # for the interval. The mean over the burst is the angular position the eyes
            # actually occupied while the image was forming, which is what the visual
            # consequence depends on; the final value alone would attribute a whole
            # interval's image to wherever the eyes happened to stop.
            # [n_substeps, n_envs, 2] - the whole eye trajectory, not a summary of it.
            # The mean is what the motor token carries (one action per interval is what
            # the encoder takes), but a mean may correspond to no command ever issued, so
            # the sequence is kept and travels with the transition for validation.
            if args.babble_bouts != "off":
                _cosv = [(f.meta.get("cmd_response") or {}).get("cos")
                         for f in frames]
                _cosv = [c for c in _cosv if c is not None]
                if _cosv:
                    _cf = float(np.mean(_cosv))
                    if prev_bout:
                        follow_bout_ema = (_cf if follow_bout_ema is None
                                           else 0.98 * follow_bout_ema + 0.02 * _cf)
                    else:
                        follow_rest_ema = (_cf if follow_rest_ema is None
                                           else 0.98 * follow_rest_ema + 0.02 * _cf)
            gaze_seq = np.stack([np.stack(
                [np.asarray(f.meta.get("gaze_cmd") or (0.0, 0.0), dtype=np.float32)
                 for f in snap_frames]) for snap_frames in gaze_burst])
            prev_gaze_substeps = np.transpose(gaze_seq, (1, 0, 2))   # [n_envs, sub, 2]
            prev_decoded_ctrl = np.stack([np.asarray(f.meta.get("decoded_ctrl"))
                                          for f in frames])                 if frames and frames[0].meta.get("decoded_ctrl") is not None else None
            # the motor interface audit's limb fixes (limb_target_control.py): the worlds report the
            # order as the learner must record it - calf rows as the knee goal on the ordered thigh,
            # arm rows projected into reach. None (and today's ledger) when both flags are off.
            limb_ledger = ledger_from_frames(
                [f.meta for f in frames], send_motor, model.handle_names,
                parent_relative=getattr(args, 'parent_relative_targets', 'off'),
                reach_projection=getattr(args, 'reach_projection', 'off'),
                finger_rows=([i for i, n in enumerate(model.handle_names) if n.split('_')[0] in HAND_DIGITS]
                             if send_hand_joint_goal is not None else ()))
            # in the learner's own eye row convention (--eye-row-convention; motor interface audit F1);
            # gaze_to_handle writes the eye row either way, so F1 applies to the limb ledger as to today's
            prev_executed = LemLite.gaze_to_handle(send_motor if limb_ledger is None else limb_ledger,
                                                   gaze_seq.mean(axis=0),
                                                   convention=model.eye_row_convention)
            # the model's gaze command for this burst, before the world's saccade generator and
            # gaze reflex (what src.step was sent): kept beside the ledger in the replay ring
            prev_ctrl_gaze = (None if res.gaze_out is None
                              else np.array(res.gaze_out, dtype=np.float32).reshape(len(frames), 2))
            if causal is not None and limb_ledger is not None:
                # the body received the order the worlds recorded, not the order as sent: the causal
                # chain's plan-to-executed distance is taken against it (review of 2026-09-25, finding 4)
                causal.on_executed(model, step, limb_ledger)
            prev_motor_substeps = None
            prev_hand_joint_goal = None
            prev_action_contract = 'world_v0'
            if send_hand_joint_goal is not None:
                canonical = np.stack([f.meta['canonical_motor_targets'] for f in frames])
                if not same_hand_goal(canonical, send_motor):
                    raise RuntimeError('Hand execution changed the canonical goal selected by the learner')
                prev_action_contract = 'wrist_relative_joint_v1'
                prev_hand_joint_goal = np.array(send_hand_joint_goal, copy=True)
                prev_motor_substeps = np.stack([np.stack([f.meta['executed_motor_targets']
                    for f in row]) for row in gaze_burst]).transpose(1,0,2,3)
                for bi, frame in enumerate(frames):
                    frame.meta['motor_substeps'] = prev_motor_substeps[bi]
                    error = frame.meta.get('hand_joint_error_deg')
                    if error is not None:
                        acc['hand_joint_error_sum'] = acc.get('hand_joint_error_sum',0.)+float(np.asarray(error).mean())
                        acc['hand_joint_error_n'] = acc.get('hand_joint_error_n',0)+1
                acc['hand_substeps'] = acc.get('hand_substeps',0)+len(frames)*len(gaze_burst)
            # the head and pelvis position rows (--position-rows): the worlds must solve what the launch
            # asked for; a world on the default reports nothing, so today's run checks nothing new
            check_position_rows([f.meta for f in frames], getattr(args, 'position_rows', 'all'))
            # the finger servo (--finger-joint-servo): likewise; a world on the default reports nothing
            check_finger_servo([f.meta for f in frames], getattr(args, 'finger_joint_servo', 'off') or 'off')
            _parts = limb_action_contract_parts(frames[0].meta)
            if _parts:
                # a calf (chest) row is now a joint goal on its parent's ordered pose, and the head's and
                # pelvis's position numbers are read by nothing: their own action contract (replay never
                # mixes two meanings in one window)
                prev_action_contract = '+'.join(([] if prev_action_contract == 'world_v0'
                                                 else [prev_action_contract]) + _parts)
                if prev_motor_substeps is None and any(p.startswith('parent_relative_') for p in _parts):
                    # the rows the physics received, regenerated every 10 ms, travel with the transition
                    # as the fingers' do
                    prev_motor_substeps = np.stack([np.stack([f.meta['executed_motor_targets']
                        for f in row]) for row in gaze_burst]).transpose(1,0,2,3)
                    for bi, frame in enumerate(frames):
                        frame.meta['motor_substeps'] = prev_motor_substeps[bi]
            if limb_ledger is not None and latched_now:
                # per author too: how far the planner's own orders were moved (review of 2026-09-25)
                limb_window_add(acc.setdefault('limb', {}),
                                [f.meta.get('limb_target_report') for f in frames],
                                author=("external_rest" if scheduled_hold else held_author))
            prev_order_id = order_seq
            if actor is not None:
                actor.prepare_learning_audit(model,prev_executed,step)
            # AUDIT FIX: 4 of 5 frames were thrown away, subsampling every event metric
            # 1-in-5. Booleans OR across the burst, scalars average, maxima max.
            for e, f in enumerate(frames):
                merged = dict(burst[-1][e])
                for snap in burst[:-1]:
                    t = snap[e]
                    # review 2026-09-06: mobile contacts last 1-3 env steps, so reading
                    # only the burst's last substep undercounted strikes up to 5x - and
                    # the one-shot "moment" flag on substeps 1-4 was consumed unseen
                    # (4/5 of moment videos never started)
                    for k in ("ball_touched", "ball_touched_hand", "gaze_on_ball",
                              "gaze_on_ball_earned", "mobile_touched",
                              "mobile_touched_hand"):
                        if t.get(k):
                            merged[k] = True
                    if t.get("moment") and not merged.get("moment"):
                        merged["moment"] = t["moment"]
                    for k in ("box_touch", "box_touch_hand"):
                        if t.get(k):
                            merged[k] = sorted(set(merged.get(k) or []) | set(t[k]))
                    for k in ("hand_ball_dist_m", "roll_deg", "gaze_ball_deg"):
                        if t.get(k) is not None and merged.get(k) is not None:
                            merged[k] = (merged[k] + t[k]) / 2.0
                    if t.get("roll_max_deg") is not None:
                        merged["roll_max_deg"] = max(merged.get("roll_max_deg", 0.0), t["roll_max_deg"])
                # contact ONSETS are events on one env frame: every frame's onsets and the union
                # of the skin's contact states, or 4 of 5 frames' events are lost (env-senses)
                merged.update(merge_burst_contact_report([snap[e] for snap in burst]))
                f.meta["task"] = merged
            t2 = time.perf_counter()
            step += 1
            acc["n"] += 1; acc["loss"] += res.loss; acc["grad"] += res.grad_norm; acc["conf"] += res.motor_confidence
            acc["t_model"] += t1 - t0; acc["t_env"] += t2 - t1 - _t_causal
            acc["coh"] += res.wellness.get("coherence", 0.0)
            if not np.isfinite(res.loss):
                acc["nan"] += 1
            for k, v in res.pe_per_modality.items():
                acc["pe"][k] = acc["pe"].get(k, 0.0) + v
            for k, v in (getattr(res, "recon_dec_per_modality", None) or {}).items():
                acc["rd"][k] = acc["rd"].get(k, 0.0) + v
            for k, v in res.wellness.items():
                acc["wl"][k] = acc["wl"].get(k, 0.0) + v
            for k, v in res.breakdown.items():
                acc["bd"][k] = acc["bd"].get(k, 0.0) + v
            acc["contacts"] += float(np.mean([f.n_contacts for f in frames]))
            if res.wellness_align_err is not None:
                acc["walign"] = acc.get("walign", 0.0) + res.wellness_align_err
                acc["walign_n"] = acc.get("walign_n", 0) + 1
            # every step, not a sample of them: a sampled flag aliased to 100% while half
            # the steps were re-reading the previous vision token
            acc["vis_fresh"] += float(frames[0].vision_patches is not None)
            acc["active"] += float(np.mean([(f.patches[:, 3] > 0).sum() for f in frames]))
            # CONTACT TURNOVER, the dominant property of the touch signal and until now
            # invisible. Measured 2026-08-28: about two patches appear and two vanish
            # every 50 ms against 5.6 active, the contact set changes on 95% of steps,
            # and those steps carry 98% of the touch prediction error. The turnover is
            # real rather than threshold flicker - a birth has median area 1,323 mm2,
            # larger than the median held contact. Without this logged, "did the world
            # change?" could only be answered with counts that stay flat while the set
            # underneath them turns over completely.
            _act = [(f.patches[:, 3] > 0) for f in frames]
            if len(_prev_active) == len(_act):
                acc["born"] += float(np.mean([float((a & ~p_).sum()) for a, p_ in zip(_act, _prev_active)]))
                acc["died"] += float(np.mean([float((~a & p_).sum()) for a, p_ in zip(_act, _prev_active)]))
                acc["turn_n"] += 1
            _prev_active = _act
            acc["pressure"] += float(np.mean([f.patches[:, 0].mean() for f in frames]))
            # The mean above is over ALL patches, most of which touch nothing, so it
            # is 36x below the worst patch at rest (478 Pa against 17,024 Pa
            # measured 2026-09-16). Wellness does NOT read the mean: contact_wellness
            # takes amax over sites, so one patch sets the whole body's pain. Log the
            # peak too, or the row that is supposed to show what the body felt shows
            # something no part of the body felt.
            acc["pressure_peak"] = max(acc.get("pressure_peak", 0.0),
                                       float(max(f.patches[:, 0].max() for f in frames)))
            acc["cmd_cos"] += float(np.mean([f.meta.get("cmd_response", {}).get("cos", 0.0) for f in frames]))
            # The held-order gauge. cmd_response_slow scores the order entered cmd_horizon frames
            # ago (action_repeat x motor_hold) over exactly those frames, so only on the LAST step
            # of a held order is it that order's whole tenure, from its latch pose. Until
            # 2026-09-23 every step's reading went in: 4 in 5 began inside a tenure and ran into
            # the next order (docs/gain_of_function_v3/agents/fix-following-gauge.md).
            slow = ([f.meta.get("cmd_response_slow", {}).get("cos") for f in frames]
                    if order_age == hold - 1 else [])
            slow = [v for v in slow if v is not None]
            if slow:
                acc["cmd_slow"] = acc.get("cmd_slow", 0.0) + float(np.mean(slow))
                acc["cmd_slow_n"] = acc.get("cmd_slow_n", 0) + 1
            acc["cmd_ratio"] += float(np.mean([f.meta.get("cmd_response", {}).get("ratio", 0.0) for f in frames]))
            acc["cmd_acc"] += float(np.mean([f.meta.get("cmd_response", {}).get("acc_cos", 0.0) for f in frames]))
            acc["cmd_win"] += float(np.mean([f.meta.get("cmd_response", {}).get("window_cos", 0.0) for f in frames]))
            # --executed-fields-only on: the orientation-only rows' ordered position is their achieved
            # position at the latch, so the position check scores only the rows whose position the
            # decoder reads (env._command_response pos_cos_position_rows; motor interface audit F4)
            _pos_key = "pos_cos_position_rows" if model.executed_fields_only_on else "pos_cos"
            acc["cmd_pos"] = acc.get("cmd_pos", 0.0) + float(np.mean([f.meta.get("cmd_response", {}).get(_pos_key, 0.0) for f in frames]))
            acc["cmd_mag"] = acc.get("cmd_mag", 0.0) + float(np.mean([f.meta.get("cmd_response", {}).get("cmd_mag", 0.0) for f in frames]))
            tk = [f.meta.get("task", {}) for f in frames]
            # The instruments added over 2026-08-23/24 were computed in the env and then
            # dropped on the floor: nothing aggregated them, so hand-in-view, the reflex
            # snap count, caregiver rescues and the face-up check existed only for one-off
            # probe scripts and never appeared in a log or on the dashboard.
            def _mean(key, default=None):
                v = [t[key] for t in tk if t and t.get(key) is not None]
                return float(np.mean(v)) if v else default

            def _rate(key):
                # only bodies that REPORT the key vote: dragonfly_hovering exists only
                # where the dragonfly is the visitor - counting absent bodies as 0 made
                # the rate move with visitor scheduling, not behavior (review 2026-09-06)
                v = [1.0 if t.get(key) else 0.0 for t in tk if t and t.get(key) is not None]
                return float(np.mean(v)) if v else None
            for name, val in (("hand_view", _mean("hand_view_deg")),
                              ("hand_seen", _rate("hand_in_view")),
                              ("hand_fov", _rate("hand_in_fovea")),
                              ("eye_pitch", _mean("eye_pitch_deg")),
                              ("unseen_s", _mean("ball_unseen_s")),
                              ("rescues", _mean("caregiver_rescues")),
                              ("snaps", _mean("reflex_snaps")),
                              ("saccades", _mean("spont_saccades")),
                              ("crossed", _rate("hands_crossed")),
                              ("pressing", _rate("hand_pressing")),
                              ("press_kpa", _mean("hand_press_kpa")),
                              ("mobile_speed", _mean("mobile_speed_cms")),
                              ("mobile_hit", _rate("mobile_touched")),
                              ("mobile_hand", _rate("mobile_touched_hand")),
                              ("mobile_hand_gap", _mean("mobile_hand_gap_m")),
                              ("reflex_on", _rate("reflex_active")),
                              # gaze plan D14: ring + critter + anticipation observability
                              ("ring_snaps", _mean("ring_snaps")),
                              ("squeaker_hits", _mean("squeaker_hits")),
                              ("squeaker_foot", _mean("squeaker_hits_foot")),
                              ("dragonfly_deg", _mean("dragonfly_deg")),
                              ("dragonfly_darts", _mean("dragonfly_darts")),
                              ("dragonfly_hover", _rate("dragonfly_hovering")),
                              ("bee_deg", _mean("bee_deg")),
                              ("antic_ok", _rate("antic_ok")),
                              ("antic_speed", _mean("antic_speed_cms")),
                              # lift-head-and-look milestone (2026-09-06)
                              ("squeaker_view", _mean("squeaker_view_deg")),
                              ("head_tilt", _mean("head_tilt_deg")),
                              ("squeaker_seen", _rate("squeaker_seen")),
                              ("headlook", _mean("headlook_events"))):
                if val is not None:
                    # review 2026-09-06: these are LIFETIME counters env-side; a window
                    # MEAN of a rising counter lags its end-of-window value by half a
                    # window and mixes counter epochs across worker rebuilds. Keep the
                    # latest value instead - consumers read them as totals.
                    if name in ("snaps", "ring_snaps", "saccades", "squeaker_hits",
                                "squeaker_foot", "dragonfly_darts", "headlook",
                                "rescues"):
                        acc[name] = val
                        acc[name + "_n"] = 1
                    else:
                        acc[name] = acc.get(name, 0.0) + val
                        acc[name + "_n"] = acc.get(name + "_n", 0) + 1
            # the hand press split (world / own body), read from the last frame like the old key
            _hp = [t for t in tk if t and t.get("hand_pressing_world") is not None]
            if _hp:
                acc["hp_n"] = acc.get("hp_n", 0) + 1
                acc["hp_world"] = acc.get("hp_world", 0.0) + float(np.mean([bool(t["hand_pressing_world"]) for t in _hp]))
                acc["hp_self"] = acc.get("hp_self", 0.0) + float(np.mean([bool(t.get("hand_pressing_self")) for t in _hp]))
                acc["hp_world_kpa"] = acc.get("hp_world_kpa", 0.0) + float(np.mean([float(t.get("hand_press_world_kpa") or 0.0) for t in _hp]))
                acc["hp_self_kpa"] = acc.get("hp_self_kpa", 0.0) + float(np.mean([float(t.get("hand_press_self_kpa") or 0.0) for t in _hp]))
            for cls, key in (("light", "box_l"), ("medium", "box_m"), ("heavy", "box_h")):
                acc[key] += float(np.mean([1.0 if cls in (t.get("box_touch") or []) else 0.0 for t in tk]))
            acc["box_hand"] = acc.get("box_hand", 0.0) + float(np.mean([1.0 if (t.get("box_touch_hand") or []) else 0.0 for t in tk]))
            acc["jitter"] = acc.get("jitter", 0.0) + float(np.mean([t.get("jitter", 0.0) for t in tk]))
            # the CLOSEST the hands came in this window, beside their mean: a mean over a
            # window of a body that is mostly elsewhere hides the one moment it nearly
            # arrived, which is the moment the strike rate of 0 cannot describe
            _gap = [t.get("mobile_hand_gap_m") for t in tk if t and t.get("mobile_hand_gap_m") is not None]
            if _gap:
                acc["mobile_gap_min"] = min(acc.get("mobile_gap_min", float("inf")), min(_gap))
            _sc = [t for t in tk if t and t.get("self_contact_patches") is not None]
            if _sc:
                acc["sc_n"] = acc.get("sc_n", 0) + 1
                acc["sc_patches"] = acc.get("sc_patches", 0.0) + float(np.mean([t["self_contact_patches"] for t in _sc]))
                acc["sc_pairs"] = acc.get("sc_pairs", 0.0) + float(np.mean([t.get("self_contact_pairs", 0) for t in _sc]))
                acc["sc_share"] = acc.get("sc_share", 0.0) + float(np.mean([t.get("self_contact_force_share", 0.0) for t in _sc]))
            # WHERE EACH BODY IS TOUCHED, as window means per body (2026-09-22). The record used to
            # carry the LAST body's report at the window's last step under skin_regions. Now every
            # body's region numbers are summed over the window's steps: acc["skr"][body][region] =
            # [patches, force_share, self_patches] sums, acc["skr_n"][body] = steps it reported.
            _skr, _skr_n = acc.setdefault("skr", {}), acc.setdefault("skr_n", {})
            for _bi, _t in enumerate(tk):
                _rg = (_t or {}).get("skin_regions")
                if not isinstance(_rg, dict):
                    continue
                _bk = str(_bi)
                _skr_n[_bk] = _skr_n.get(_bk, 0) + 1
                _sb = _skr.setdefault(_bk, {})
                for _name, _v in _rg.items():
                    if isinstance(_v, dict):
                        _s = _sb.setdefault(_name, [0.0, 0.0, 0.0])
                        _s[0] += float(_v.get("patches", 0) or 0)
                        _s[1] += float(_v.get("force_share", 0.0) or 0.0)
                        _s[2] += float(_v.get("self_patches", 0) or 0)
            # SELF-CONTACT ONSETS from the world's report (env.py _contact_report, every env frame of
            # the burst merged): how many began, which region pairs and body pairs, how many are the
            # resting posture (structural) and how many happened within 30 degrees of an eye's axis
            _ons_bodies = [(_t or {}).get("self_contact_onsets") for _t in tk
                           if "self_contact_onsets" in (_t or {})]
            if _ons_bodies:
                acc["sce_body_steps"] = acc.get("sce_body_steps", 0) + len(_ons_bodies)
                _rp, _bp = acc.setdefault("sce_regions", {}), acc.setdefault("sce_bodies", {})
                for _lst in _ons_bodies:
                    for _o in (_lst or ()):
                        acc["sce_onsets"] = acc.get("sce_onsets", 0) + 1
                        acc["sce_structural"] = acc.get("sce_structural", 0) + int(bool(_o.get("structural")))
                        acc["sce_in_view"] = acc.get("sce_in_view", 0) + int(bool(_o.get("in_view")))
                        _r = self_contact_region_key(_o)
                        _rp[_r] = _rp.get(_r, 0) + 1
                        _p = "|".join(str(x) for x in (_o.get("pair") or ("?", "?")))
                        _bp[_p] = _bp.get(_p, 0) + 1
            acc["roll"] += float(np.mean([t.get("roll_deg", 0.0) for t in tk]))
            acc["roll_max"] = max(acc["roll_max"], max((t.get("roll_max_deg", 0.0) for t in tk), default=0.0))
            acc["rolled"] += float(np.mean([1.0 if t.get("roll_max_deg", 0) > 120 else 0.0 for t in tk]))
            motor_ring.append(res.motor_for_action.reshape(args.n_envs, -1))
            if len(motor_ring) > 200:
                motor_ring.pop(0)
            # (review T18: a `transitions` list accumulated phase logs here that no
            # reader ever consumed; the stats file already carries them)
            if any("hand_ball_dist_m" in t for t in tk):
                acc["ball_n"] += 1
                acc["ball_d"] += float(np.mean([t.get("hand_ball_dist_m", 0.0) for t in tk if t]))
                acc["ball_touch"] += float(np.mean([1.0 if t.get("ball_touched") else 0.0 for t in tk if t]))
                acc["ball_hand"] = acc.get("ball_hand", 0.0) + float(np.mean([1.0 if t.get("ball_touched_hand") else 0.0 for t in tk if t]))
                acc["gaze_on"] = acc.get("gaze_on", 0.0) + float(np.mean([1.0 if t.get("gaze_on_ball") else 0.0 for t in tk if t]))
                acc["gaze_earn"] = acc.get("gaze_earn", 0.0) + float(np.mean([1.0 if t.get("gaze_on_ball_earned") else 0.0 for t in tk if t]))
                acc["gaze_deg"] = acc.get("gaze_deg", 0.0) + float(np.mean([t.get("gaze_ball_deg", 180.0) for t in tk if t]))
            acc["wgrad"] += res.wellness_grad_norm
            # The wellness->motor slope is zero whenever every handle sits inside the
            # reach envelope - the boundary formula is flat there by construction - so
            # the mean alone cannot tell a small steady pull from a pathway that
            # delivered nothing on most steps. Measured on H81's 20 explainer
            # snapshots: exactly 0.0 on 11 of them, mean 0.0014.
            acc["wgrad_zero"] = acc.get("wgrad_zero", 0) + int(res.wellness_grad_norm == 0.0)
            accumulate_prediction_audit(acc.setdefault('prediction_audit', {}), model.last_prediction_audit)
            accumulate_prediction_audit_rows(acc.setdefault('prediction_audit_rows', {}),
                                             getattr(model, 'prediction_audit_rows', None))
            accumulate_prediction_audit_rows(acc.setdefault('prediction_audit_hold_first_rows', {}),
                                             getattr(model, 'prediction_audit_hold_first_rows', None))
            for k2, v2 in (model.last_next_pe or {}).items():
                acc.setdefault("np_" + k2, 0.0)
                acc["np_" + k2] += v2
            # item 4 (2026-09-02): the tenure dicts are SAMPLE-AND-HOLD (the last
            # event's value persists between events), so accumulating every step made
            # the emitted "means" tenure-length-weighted and reset-diluted. They
            # accumulate only on EVENT steps now - "tenure"/"tenure_tok" appear in
            # the loss breakdown only on the step an order actually ends - and the
            # flush divides by the event counts, giving true per-event means.
            if "tenure" in res.breakdown:
                acc["tn_n"] = acc.get("tn_n", 0) + 1
                for k2, v2 in (getattr(res, "tenure_pe_per_modality", None) or {}).items():
                    acc.setdefault("tn_" + k2, 0.0); acc["tn_" + k2] += v2
                for k2, v2 in (getattr(res, "tenure_gap_per_modality", None) or {}).items():
                    acc.setdefault("tg_" + k2, 0.0); acc["tg_" + k2] += v2
            if "tenure_tok" in res.breakdown:
                acc["tt_n"] = acc.get("tt_n", 0) + 1
                for k2, v2 in (getattr(res, "tenure_tok_per_modality", None) or {}).items():
                    acc.setdefault("tt_" + k2, 0.0); acc["tt_" + k2] += v2
            # item 4: window means for the gauges the record used to snapshot.
            # Per-KEY counters: a sense absent from some steps (vision stale, say)
            # must divide by its own count, not the dict's.
            for k2, v2 in (getattr(model, "last_agency", None) or {}).items():
                acc.setdefault("ag_" + k2, 0.0); acc["ag_" + k2] += float(v2)
                acc.setdefault("agN_" + k2, 0); acc["agN_" + k2] += 1
            for k2, v2 in (getattr(model, "last_consequence", None) or {}).items():
                acc.setdefault("cq_" + k2, 0.0); acc["cq_" + k2] += float(v2)
                acc.setdefault("cqN_" + k2, 0); acc["cqN_" + k2] += 1
            for k2, v2 in (getattr(model, "last_gate", None) or {}).items():
                acc.setdefault("gt_" + k2, 0.0); acc["gt_" + k2] += float(v2)
                acc.setdefault("gtN_" + k2, 0); acc["gtN_" + k2] += 1
            if res.curiosity_drive is not None:
                acc["cur"] = acc.get("cur", 0.0) + float(res.curiosity_drive)
                acc["cur_n"] = acc.get("cur_n", 0) + 1
            # the drive is a plain float on every step; causal_pe below skips NaN steps.
            # Count the steps where the two divisors part company, so the record can say
            # how many of its curiosity_drive steps the causal_pe mean does not contain.
            if curiosity_used_fallback(res):
                acc["cur_fallback_n"] = acc.get("cur_fallback_n", 0) + 1
            _rp = getattr(res, "recon_pe", None)
            if _rp is not None and _rp == _rp:
                acc["rpe"] = acc.get("rpe", 0.0) + float(_rp)
                acc["rpe_n"] = acc.get("rpe_n", 0) + 1
            _cp = getattr(res, "causal_pe", None)
            if _cp is not None and _cp == _cp:
                acc["cpe"] = acc.get("cpe", 0.0) + float(_cp)
                acc["cpe_n"] = acc.get("cpe_n", 0) + 1
            for k2, v2 in (getattr(res, "next_pe_per_modality", None) or {}).items():
                acc.setdefault("cp_" + k2, 0.0); acc["cp_" + k2] += float(v2)
                acc["cp_n"] = acc.get("cp_n", 0) + (1 if k2 == "proprioception" else 0)
            if res.imagination_loss is not None:
                acc["im"] = acc.get("im", 0.0) + float(res.imagination_loss)
                acc["im_n"] = acc.get("im_n", 0) + 1
            if model.movement is not None and model.movement.last:
                for k2, v3 in model.movement.last.items():      # every drive number, not a
                    if isinstance(v3, (int, float)):               # hardcoded subset (smoothness
                        acc.setdefault("mv_" + k2, 0.0)            # was silently dropped)
                        acc["mv_" + k2] += float(v3)
            if step % args.report_every == 0:
                flush_stats()
            if cm is not None:
                r = cm.step(mean_pe=res.mean_pe)
                if r.get("progress_report"):
                    sf.write(json.dumps({"type": "curriculum_progress", "timestamp": time.time(), "phase": cm.phase_name,
                                         "phase_step": cm.phase_step, "global_step": cm.global_step}) + "\n")
                if r.get("phase_changed"):
                    # the rebuild decision is taken BEFORE the record is written, so the
                    # stream carries what happened instead of a transition that reads the
                    # same whether the bodies were rebuilt or nothing was touched.
                    # env_kwargs_for builds a dict and has no side effect, so moving it
                    # above the write changes nothing but the order of two pure calls.
                    changes, new_kw = rebuild_changes(args, cm.phase_name, out, kw)
                    sf.write(json.dumps(phase_transition_record(
                        r.get("transition_log"), step, changes, time.time())) + "\n")
                    sf.flush()
                    # the limits change at every transition whatever this says: rom_limit
                    # and strength_limit come from the phase table, not from env kwargs
                    log.info("phase -> %s (rom %.2f, strength %.2f); bodies rebuilt: %s; "
                             "changes %s; context buffer, pending transitions, goal ring "
                             "and replay episode boundary dropped: %s",
                             cm.phase_name, cm.get_dev_limits()["rom_limit"],
                             cm.get_dev_limits()["strength_limit"], bool(changes),
                             changes or "{}", bool(changes))
                    # keep the two readers of the schedule in step after the roll, and
                    # after an overseer 'advance_phase' that rolls early
                    bind_schedule_to_config(model, cm, args.steps_per_phase, step)
                    if changes:
                        src.rebuild(**changes)
                        kw.update(new_kw)
                        frames = src.reset()
                        # reset_transition_state below is the single owner of the replay
                        # episode boundary (lem_model.py): a separate mark_reset() here
                        # advanced the episode id twice for one boundary
                        # Nothing may be carried across a teleport. Without this the first
                        # observation of the new episode is paired with the old episode's
                        # action, the old temporal context, and a next-frame target that
                        # spans the jump.
                        #
                        # INDENTATION MATTERS HERE. These three lines sat one level out
                        # until 2026-08-25, so they ran on EVERY step (--curriculum
                        # defaults to on, so `cm is not None` is nearly always true).
                        # That zeroed the context buffer every step, cleared prev_executed
                        # before it could ever be used, bumped the replay episode id every
                        # step so no window was ever same-episode, and defeated
                        # --motor-hold. Training diverged: proprioception error 0.169 ->
                        # 0.740 over 900 steps. It failed silently - the counters that
                        # would have shown it were not surfaced anywhere.
                        model.reset_transition_state()
                        if actor is not None:actor.reset_transition_state()
                        if causal is not None:causal.clear('phase rebuild')
                        last_physical_times=None
                        held_motor = None
                        prev_executed = None
                        prev_order_id = None
                        prev_ctrl_gaze = None
                        pendings = []
                        _prev_active = []   # contact-turnover must not span the rebuild
            if model.replay is not None and step % args.replay_every == 0:
                if args.fill_ring_first and not ring_clean:
                    n_lab = model.replay.labelled_count()
                    if n_lab >= ring_need:
                        ring_clean = True
                        log.info("replay begins at step %d: %d transitions carry their "
                                 "outgoing action", step, n_lab)
                    else:
                        if step % (args.replay_every * 50) == 0:
                            log.info("waiting for replay: %d/%d labelled transitions",
                                     n_lab, ring_need)
                        rr = None
                if not (args.fill_ring_first and not ring_clean):
                    rr = model.replay_update(batch=args.replay_batch)
                else:
                    rr = None
                if rr and comparison is not None:
                    comparison.replay_pass(step, rr)          # record c's line of this replay pass
                if rr:
                    acc["replay_loss"] = acc.get("replay_loss", 0.0) + rr["replay_loss"]
                    acc["replay_n"] = acc.get("replay_n", 0) + 1
                    # carry the contract diagnostics into the record: which modalities were
                    # graded, and what share of the ring carries an executed action. Without
                    # these the run cannot show from its own log that the mask and the fill
                    # gate did anything.
                    for _k in ("replay_graded_mods", "replay_clean_frac",
                               "replay_fill_frac", "replay_batch"):
                        if _k in rr:
                            acc[_k] = acc.get(_k, 0.0) + float(rr[_k])
                    # replay_batch above is the ACTUAL number of windows the pass got,
                    # which is what makes the page's "48 windows" an upper bound; without
                    # it a run cannot show whether short episodes shrank its re-study.
                    # The next three are LIFETIME counters inside the model, not per-pass
                    # values, so averaging them would report the mean of a number that
                    # only grows: keep the newest reading instead, and their growth
                    # between records is the per-window signal.
                    for _k in ("replay_motor_rebuilt", "replay_motor_unlabeled",
                               "replay_ctx_embedded"):
                        if _k in rr:
                            acc["last_" + _k] = float(rr[_k])
            if scheduled_reset(step,args.episode_steps):
                sf.write(json.dumps(dict(type='episode_reset',step=step,timestamp=time.time(),reason='scheduled physical episode boundary'))+'\n');sf.flush()
                last_physical_times=None
                frames = src.reset()
                if model.movement is not None:
                    model.movement.reset()
                # the model's transition reset below is the single owner of the replay
                # episode boundary; a separate mark_reset() here advanced the episode id
                # twice for one boundary, which the physical-fault reset never did
                # Nothing may be carried across a teleport. Without this the first
                # observation of the new episode is paired with the old episode's action,
                # the old temporal context, and a next-frame target that spans the jump -
                # a fabricated cause, which is exactly what the transition contract removes.
                model.reset_transition_state()
                if actor is not None:actor.reset_transition_state()
                if causal is not None:causal.clear('scheduled episode reset')
                held_motor = None
                prev_executed = None
                prev_order_id = None
                prev_ctrl_gaze = None
                pendings = []
                _prev_active = []           # contact-turnover must not span the teleport
                last_physical_times = None
            if (time.time() - last_ckpt) >= args.checkpoint_every_min * 60.0:
                episode_boundary = True
                # 2026-09-18: a failed write must not end the run. The H84 smoke's crash
                # snapshot failed with "The device is not ready" on a USB disk whose system
                # log showed eleven retried IO operations and a port reset in six minutes.
                # Before this guard the same failure in an hourly save raised out of the
                # loop, the crash snapshot then failed on the same disk, and the run lost
                # every step since its last complete checkpoint. Now the failure is logged,
                # counted in the stats (checkpoint_failures) and tried again next hour.
                try:
                    save_ckpt("hourly")
                except Exception:
                    ckpt_failures += 1
                    ckpt_last_error = traceback.format_exc().strip().splitlines()[-1]
                    log.error("hourly checkpoint FAILED (%d so far): %s - training continues; "
                              "next attempt in %.0f min", ckpt_failures, ckpt_last_error,
                              args.checkpoint_every_min)
                last_ckpt = time.time()
            # moment videos (gaze plan D13): a critter event (task "moment" flag, one-shot)
            # cuts a short clip named for the event, at most one per 3 minutes. Uses the
            # scheduled-video machinery; captures the aftermath (there is no pre-roll).
            _mev = next(((i2, str((f.meta.get("task") or {}).get("moment") or ""))
                         for i2, f in enumerate(frames)
                         if (f.meta.get("task") or {}).get("moment")), None)
            _moment = _mev[1] if _mev else ""
            if _moment and (time.time() - last_moment_video) >= 180.0:
                last_moment_video = time.time()
                mp = os.path.join(out, "videos", f"moment_{_moment}_step_{step}.mp4")
                try:
                    # review 2026-09-06: the clip must come from the BODY that had the
                    # moment - worker 0's camera showed a bystander 7/8 of the time
                    src.start_video(mp, n_frames=int(min(8.0, args.video_seconds) / frames[0].dt),
                                    env_idx=_mev[0])
                    sf.write(json.dumps({"type": "video", "timestamp": time.time(),
                                         "step": step, "path": mp, "moment": _moment}) + "\n")
                    log.info("moment video (%s) -> %s", _moment, mp)
                except Exception:
                    pass
            if args.video_every_min > 0 and (time.time() - last_video) >= args.video_every_min * 60.0:
                vp = os.path.join(out, "videos", f"step_{step}_{time.strftime('%H%M%S')}.mp4")
                try:
                    src.start_video(vp, n_frames=int(args.video_seconds / frames[0].dt))
                    src.save_views(os.path.join(out, "views", f"step_{step}.png"))
                    sf.write(json.dumps({"type": "video", "timestamp": time.time(), "step": step, "path": vp}) + "\n")
                    log.info("recording %s s video + scene views -> %s", args.video_seconds, vp)
                except Exception as ex:
                    log.warning("video failed: %s", ex)
                last_video = time.time()
            episode_boundary = True
        if acc["n"] > 0:
            flush_stats()
        save_ckpt("end")
        if args.probe_touch and probe_rows:
            steps_ = np.array([r[0] for r in probe_rows]); idx = np.array([r[1] for r in probe_rows])
            emb = np.stack([r[2] for r in probe_rows])
            wcls = np.array([r[3] for r in probe_rows], dtype=np.int8)
            np.savez_compressed(os.path.join(out, "touch_probe.npz"), steps=steps_, patch_index=idx, emb=emb,
                                weight_class=wcls,
                                patch_names=np.array(patch_names), regions=np.array([region_of(n) for n in patch_names]))
            log.info("touch probe saved: %d rows", len(probe_rows))
        ok = True
    except Exception:
        err = traceback.format_exc()
        log.error("training run FAILED:\n%s", err)
        try:
            # a crash BEFORE the loop starts (a bad resume, say) has no save_ckpt yet, and
            # the NameError for it used to bury the real error under an UnboundLocalError
            if "save_ckpt" in locals():
                try:
                    save_ckpt("crash")  # never lose training to a crash again (take 4 lost 53 min)
                except Exception:
                    # 2026-09-18: the configured storage can be the thing that failed (a
                    # USB disk answering "The device is not ready"); the run directory's
                    # own disk gets one more try. The supervisor does not resume from this
                    # folder - it reads the configured storage - so a resume from it is
                    # a deliberate act, with the folder named in the log.
                    from lem.training.checkpoint import CheckpointManager
                    fallback = os.path.join(out, "checkpoints_fallback")
                    log.error("crash snapshot failed on the configured storage: %s - trying %s",
                              traceback.format_exc().strip().splitlines()[-1], fallback)
                    model.checkpoints = CheckpointManager(model.config, fallback)
                    save_ckpt("crash_fallback")
                    log.error("crash snapshot written to the FALLBACK folder %s; the supervisor "
                              "will not resume from it - resume by hand with --resume", fallback)
            else:
                log.error("crashed before training began; nothing to snapshot")
        except Exception:
            log.error("crash snapshot also failed: %s", traceback.format_exc())
    finally:
        try:
            if explain_writer is not None:
                explain_writer.flush()
        except Exception:
            pass
        try:
            if observer_lane is not None:
                observer_lane.close()
        except Exception:
            pass
        try:
            if comparison is not None:
                comparison.close()
        except Exception:
            pass
        try:
            if src is not None:
                src.close()
        except Exception:
            pass
    summary = {"ok": ok, "exception": err, "steps": step, "elapsed_s": time.time() - t_start, "n_envs": args.n_envs,
               "checkpoints": ckpt_list, "out": out, "args": vars(args), "resumed_from": args.resume}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    # stage_done.json: the marker a chain can TRUST. summary.json says a run ended; this
    # says which brain it started from and which it produced, so a later stage can verify
    # it is building on the work it meant to (2026-08-23: a chain ran an "hour 8" battery
    # against the hour-4 brain because a filename cannot carry that fact).
    try:
        from .tools.training_continuation import write_stage_done
        write_stage_done(out,ok=ok,step=step,resumed_from=args.resume,started_at=t_start,exception=err)
    except OSError:
        log.warning("could not write stage_done.json")
    log.info("summary: ok=%s steps=%d elapsed=%.1f min", ok, step, (time.time() - t_start) / 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
