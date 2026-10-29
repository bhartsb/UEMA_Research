# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Does the latent body hierarchy choose better orders than the current planner from the same saved starts?

    python -m mimo_bridge.tools.latent_body_compare --checkpoint CK --hierarchy H.pt --holdout HOLDOUT.jsonl
        --witness-bank BANK.pt --output OUT [--step-hierarchy H1.pt --step-holdout HOLDOUT1.jsonl]
        [--goal-design recorded_v2|anchored_v1] [--bodies 0,...,7] [--held-out-bodies 7] [--candidates 256]
        [--witnesses 8] [--seed 0] [--code-sampling auto|box|encoded|gaussian]

A pause job (it loads the production checkpoint and builds one world per start). Nothing is trained, no production
file is written, and the hierarchy never commands a live body.

STARTS. For each body, a fresh world from the checkpoint's episode companion; the production policy acts for
W model steps (W from --warmups, one start per W; the held-out bodies get --held-out-warmups), then the decision
moment is read exactly as docs/gain_of_function_v3/compare_hand_body_goals_2026-09-24.py reads it (the planners'
recorded comparison; this module imports its execution helpers). The hierarchy sees the last two observations.

GOALS, design recorded_v2 (Oct 1, 2026; the default). Review of the Sep 30 comparison: every goal there was the
controller's own order plus a small change executed for one hold (anchored_v1 below), so the controller's own endpoint
sat 0.02-0.36 cm from the wrist goal (median 0.16 cm) while the goal moved 0.2-6.0 cm from the start; that scored "do
what the controller does", not reaching, and punished every arm that changed more rows. A recorded_v2 goal is the
endpoint of an order this same body actually issued at another time, from a similar start:
  1. CANDIDATES. The witness bank (latent_body_split --horizon-steps 5: every recorded hold of every body, its start,
     its order and its end 5 model steps later). Among the same body's recorded hold starts whose trunk "up" lies
     within UP_DEG of the current one, the --witnesses nearest in START_FEATURES: every body handle's position
     relative to the pelvis (lower_trunk) in the pelvis frame, cm, and its velocity in that frame times VELOCITY_S
     (the cm it would travel in one hold), each root-mean-squared over the handles and the two added.
  2. TRANSPLANT. The recorded order's position and orientation rows (every handle but the eyes) are carried from the
     recorded pelvis pose to the current one (the order relative to the body's own pelvis is kept); radius, strength
     and the eyes are the controller's, as the planners' MatchedControls fix them. Projected to a feasible hand goal.
  3. REACHED BY EVIDENCE. Each candidate order is executed from the saved start through the production execution
     transform for one hold (0.25 s), twice (an unequal replay stops the job). Its endpoint is a pose some executed
     order reached from THIS start: the witness arm scores 0 cm on it (the "perfect reach" reference).
  4. NOT ANCHORED. A goal is admitted only when the controller's own endpoint lies at least SEPARATION_CM from it at
     the goal's handles (half the handle's median movement over one recorded hold, in the goal's frame); the first
     candidate in start-similarity order that clears the line is the goal; a start with no such candidate is recorded
     as not admitted (every candidate's separation kept). The arms run once per distinct admitted witness.
  Goals (GOALS_V2; handles, frame): wrist_L (world); index_L, index_R (the index handle relative to its own wrist, in
  the wrist's coordinates: the finger's own reach, not the 98-99% of its world movement that is the arm carrying it);
  foot_L (world); upperarms (both upper arms, world: the shoulders a trunk movement carries). --goals defaults to the
  goals whose handles, and the handles that carry them (GOAL_CARRIERS_V2), the hierarchy owns. Every arm is given the
  same whole goal pose (the witness endpoint).
  Each trial records witness_request_change: the witness order's largest change from the controller's request after
  both are projected to feasible hand goals, measured as the production planners' search region bounds it
  (lem.model.physical_goal.PhysicalTrustRegion, EFFERENCE_SCALES): every body handle but the fingers and the eyes at
  most 1.8 cm per coordinate and 11 degrees of turn, every hand joint goal at most 11 degrees; inside_planner_region
  says whether the planners could have issued the witness order at all. A witness whose re-execution in the trial does
  not repeat its candidate execution (trajectory digest) makes that trial's goals not admitted, with the reason kept.
GOALS, design anchored_v1 (Sep 30, 2026; kept to reproduce that run): from the controller's own request a reference
request is executed for one hold and its endpoint is the goal.
  wrist_L_reach  the left wrist requested REACH_CM (1.8 cm) above the controller's request (that script's witness)
  index_L_curl   the left index finger's knuckle, middle and distal joints at 95% of their flexion limit, every
  index_R_curl   other joint at the controller's goal (the same for the right hand)
  knee_L_flex    the controller's left calf and foot requests turned TURN_DEG (11 degrees) about the knee hinge
  trunk_turn     that script's supported_trunk_turn_witness

ARMS, all executed through the same production execution transform (smoother, development limits, hand joint
smoothing: that script's filtered()) and the same 0.25 s physics (its physical()), each twice from one snapshot
(an unequal replay stops the job):
  controller   the production policy's own order
  direct       the production planner (feasible-hand direct search) given the goal pose
  synergy      the production planner through the learned synergy, given the goal pose
  latent       lem.planner.latent_candidates.search over --candidates root codes drawn in the box of the measured
               latent radius (95th percentile of |root code| over --radius-rows held-out examples) of --hierarchy,
               scored by the hierarchy's predicted world positions of every handle at the end of the hold against the
               goal pose. A hierarchy whose horizon_steps x duration_s is the 0.25 s hold (HOLD HORIZON, Oct 1, 2026;
               trajectory.hold_example) is scored on its OWN hold prediction. A one-step hierarchy's step is
               extrapolated over the hold (start + factor x predicted one-step displacement; HOLD_FACTOR 4.2 for the
               pilot, HOLD_FACTORS per handle for the whole body: over a 5-step hold a handle travels 3.09-4.24 first
               steps along its first step's direction, 40,000 recorded seg41 rows).
  CODE SAMPLING (--code-sampling, Oct 1, 2026; lem.latent_body.code_range): "box" is the search above. "encoded"
               (the default, "auto", for a whole-body hierarchy): the candidates are the root codes the encoder gave
               one of the --radius-rows held-out rows each (all roots from the same row), only rows inside every root's
               range (the 95th percentile of the encoder's own Mahalanobis distances), with the encoder's mean code as
               candidate 0. "gaussian": drawn from the fitted mean and covariance inside the range. The box reached
               corners the encoder never visits (whole-body fit: box codes a median 4.9-26.6 Mahalanobis units out,
               the encoder's own 2.2-2.5). The declaration records each arm's mode and range; each trial's receipt its
               candidates' distances against the range.
  latent_step  (reported, --step-hierarchy) the same search with a second hierarchy, e.g. the one-step fit scored by
               its extrapolated step, so the hold model and the stretched step are compared on the same goals.
  The chosen code is decoded to handle targets, placed into the controller's request (the eyes, radius and strength
  stay the controller's) and projected to a feasible hand goal.
  witness      (recorded_v2) the goal's own order: 0 cm by construction, replay-verified.

SCORES (physical, at 0.25 s): goal_error_cm = root mean square distance of the goal handles from the goal, in the
goal's frame; goal_error_world_cm the same in world coordinates; tree_error_cm over the hierarchy's handles; pose_error
= that script's full-body physical_goal_error. Per arm and goal: gain_over_controller_cm (controller's error minus the
arm's) and gap_closed (1 - arm / controller: 1 = reached the goal, 0 = no closer than the controller's own order,
negative = farther); stay_cm is the goal error if every handle stayed where it started.

TRAINING EXPOSURE (C, Oct 1, 2026). Every declaration and result records what each side learned from: each hierarchy's
updates, example presentations, distinct training examples, model steps per example and the body-moments they cover
(from its fit receipt and split receipt), and the production model's checkpoint step times its bodies per step (live
body-moments) with its replay settings. A run whose exposure cannot be read refuses to start.

DECLARED LINES (before the run; the arm named by --primary, the hold hierarchy by default):
  better_than_planner  per goal, over its admitted starts n (at least MIN_ADMITTED, else the goal is "not decided"):
                       the primary latent arm beats BOTH direct and synergy on goal_error_cm in at least
                       wins_needed(n) starts (one-sided sign test P < 0.05; ties = exact equality, the replay noise is 0
                       because every arm repeats exactly) with a positive mean gain. 18 admitted starts: 13 wins.
  held_out             on the held-out bodies (never in the hierarchy's training rows), one sign per held-out START (the
                       goals of one start usually share one goal pose and the same orders, so they are not independent):
                       the mean over that start's admitted goals of (planner error - latent error) / controller error,
                       a win when positive; at least wins_needed(n) of n starts against each planner arm, n >= 5
                       (8 held-out starts: 7).
  beats_controller     reported for every arm: the same sign test against the controller (does the arm reach beyond
                       what the controller's own order does).
  latent_vs_latent_step reported: the hold hierarchy against the one-step hierarchy whose step is stretched over the
                       hold, the same sign test (does predicting the whole hold choose better orders).
  Per-goal lines count starts of 8 bodies x 2 plus the held-out body's 8: 22 starts, wins_needed(22) = 16 (P = 0.026).
  Reachability: every admitted goal is reached by an executed order from the same start (the witness, 0 cm), and the
  controller ends at least SEPARATION_CM from it. The planners can request at most 1.8 cm per coordinate and 11
  degrees away from the controller's request; on Sep 30 that moved their wrist endpoint 0.37-0.97 cm from the
  controller's, so on a wrist goal 2.4 cm or more away they can close part of the gap only, and inside_planner_region
  counts the goals they could have issued exactly. The latent arm decodes orders from the real code range, so it can
  request orders like the recorded ones; its arm level moved the arm in the predicted direction on 8 of 8 arms in the
  Sep 30 causal-control test. Admission: on the recorded rows (goal_stand_in.py, bodies 0, 3 and 7, 600 starts each,
  8 nearest recorded neighbours) every start had a neighbour at or beyond each separation line; that stand-in includes
  the neighbours' start mismatch (median 5.96 cm in this tool's start distance on body 7). Holding the start difference
  fixed, the order difference alone moves the left wrist a median 6.0 cm over a hold (order_share.py, body 7). In the
  world admission is measured by this run, every candidate's separation recorded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

GOAL_DESIGNS = ("recorded_v2", "anchored_v1")
# anchored_v1 (Sep 30): goal -> goal handles (world)
GOALS = {"wrist_L_reach": ("wrist_L",), "index_L_curl": ("index_L",), "index_R_curl": ("index_R",),
         "knee_L_flex": ("foot_L",), "trunk_turn": ("upperarm_L", "upperarm_R")}
# the handles whose requests an anchored_v1 goal's recipe changes: the hierarchy must own them to have a say
GOAL_REQUESTS = {"knee_L_flex": ("calf_L", "foot_L"), "trunk_turn": ("lower_trunk", "chest")}
# recorded_v2 (Oct 1): goal -> (goal handles, frame); "wrist": the handle relative to its own side's wrist
GOALS_V2 = {"wrist_L": (("wrist_L",), "world"), "index_L": (("index_L",), "wrist"), "index_R": (("index_R",), "wrist"),
            "foot_L": (("foot_L",), "world"), "upperarms": (("upperarm_L", "upperarm_R"), "world")}
# the handles that carry a recorded_v2 goal handle (its position follows theirs): the hierarchy must own them too
GOAL_CARRIERS_V2 = {"foot_L": ("thigh_L", "calf_L"), "upperarms": ("lower_trunk", "chest")}
# Half of each goal handle's median movement over one recorded hold, in the goal's frame (5-row hold examples of the
# seg41 recording, bodies 0, 3, 6 and 7, 4,352 holds each; docs/gain_of_function_v4/agents/latent_compare_design_2026-10-01/
# goal_design_*.log): wrist 4.60-5.02 cm, index relative to its wrist 0.58-0.63 cm, foot 1.09-1.24 cm, upper arms
# 0.90-0.96 cm. The controller's endpoint must lie at least this far from an admitted goal.
SEPARATION_CM = {"wrist_L": 2.4, "index_L": 0.30, "index_R": 0.30, "foot_L": 0.60, "upperarms": 0.47}
UP_DEG = 30.0          # candidate starts: the pelvis "up" within this angle of the current one (lying the same way)
VELOCITY_S = 0.25      # velocities enter the start distance as the cm they would carry a handle over one hold
MIN_ADMITTED = 10      # a goal's line is decided only on at least this many admitted starts (10 -> 9 wins needed)
MIN_HELD_OUT_STARTS = 5  # the held-out line is decided only on at least this many held-out starts (5 -> 5 wins needed)
HOLD_S = 0.25          # the order hold the arms are executed for (25 physics steps, 5 model steps)
# over a 5-step order hold, each handle's displacement along its first step's direction, in first steps (median over
# 40,000 recorded seg41 rows, docs/gain_of_function_v4/agents/latent_full_body_2026-10-01/hold_ratio_body.log)
HOLD_FACTORS = dict(wrist=4.24, thumb=4.15, index=4.05, middle=3.88, ring=3.97, pinky=4.03, head=4.15, upperarm=3.40,
                    lowerarm=4.12, thigh=3.94, calf=4.04, foot=3.39, gaze=4.03, lower_trunk=3.09, chest=3.70)
ARMS = ("controller", "direct", "synergy", "latent")
PLANNERS = ("direct", "synergy")
HOLD_FACTOR = 4.2
LINES = dict(alpha=0.05, per_goal="wins_needed(admitted starts) against both planners, admitted >= min_admitted",
             min_admitted=MIN_ADMITTED, held_out="one sign per held-out start (mean gap gain over its admitted goals); "
             "wins_needed(held-out starts) against each planner, starts >= min_held_out_starts",
             min_held_out_starts=MIN_HELD_OUT_STARTS, separation_cm=SEPARATION_CM)


def measure_radius(learner, examples, quantile=0.95):
    """The box half-width holding `quantile` of the real root codes, pooled over the roots' dimensions."""
    if len(examples) < 50:
        raise ValueError("Measure the latent radius on at least 50 held-out examples")
    with torch.no_grad():
        codes = learner.model.encode(learner.batch(examples))
    values = torch.cat([codes[r].abs().flatten() for r in learner.tree.roots])
    return float(values.quantile(quantile))


def measure_code_range(learner, examples):
    """lem.latent_body.code_range.CodeRange of the roots' real codes on held-out examples (the encoded search's
    candidates and every mode's distance record)."""
    from lem.latent_body.code_range import CodeRange
    if len(examples) < 50:
        raise ValueError("Measure the code range on at least 50 held-out examples")
    with torch.no_grad():
        codes = learner.model.encode(learner.batch(examples))
    return CodeRange({r: codes[r] for r in learner.tree.roots})


def latent_arm(learner, holdout, code_sampling="auto"):
    """A latent arm's search settings measured on its held-out examples: the box radius (always recorded), the code
    sampling (resolve_sampling: "auto" = encoded for a whole-body hierarchy) and the roots' code range."""
    from lem.latent_body.code_range import resolve_sampling
    return dict(radius=measure_radius(learner, holdout), sampling=resolve_sampling(code_sampling, learner.config),
                code_range=measure_code_range(learner, holdout))


def search_settings(spec):
    """The keyword arguments latent_order takes from a latent arm's settings (latent_arm)."""
    return dict(radius=spec["radius"], sampling=spec["sampling"], code_range=spec["code_range"])


def hold_scoring(cfg, body, hold_factor=HOLD_FACTOR, hold_s=HOLD_S):
    """("own", None) when the hierarchy predicts the whole hold (horizon_steps x duration_s = hold_s); ("factor",
    hold_factor) for a one-step pilot; ("factors", "HOLD_FACTORS") for a one-step whole-body hierarchy."""
    horizon = int(cfg.get("horizon_steps", 1))
    if math.isclose(horizon * float(cfg["duration_s"]), hold_s, rel_tol=1e-6, abs_tol=1e-9):
        return "own", None
    if horizon != 1:
        raise ValueError(f"A hierarchy predicting {horizon} model steps is neither one step nor the {hold_s} s hold")
    return ("factor", hold_factor) if body is None else ("factors", "HOLD_FACTORS")


def latent_order(learner, history, current, target_cm, template_cm, *, candidates, radius, generator,
                 hold_factor=HOLD_FACTOR, sampling="box", code_range=None):
    """Search root codes for the order whose predicted hold endpoint is nearest the goal pose.

    target_cm [H,3] goal positions; template_cm [H,9] the controller's request in centimetres (only the tree's
    owned rows change). Returns (order [H,9] cm, predicted mean squared cm of the chosen code, receipt)."""
    from lem.latent_body.trajectory import (end_positions, predicted_rows, row_options, row_scales, state_inputs,
                                            tensor)
    from lem.planner.latent_candidates import search
    from lem.planner.latent_rollout import handle_targets
    tree, cfg = learner.tree, learner.config
    if not math.isclose(current.sim_time - history.sim_time, cfg["duration_s"], rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("The two observations are not one trained model step apart")
    rows = row_options(cfg)
    body = rows["body"]
    mode, factor = hold_scoring(cfg, body, hold_factor)
    state, recent = state_inputs(tree, history, current, cfg["scales"], **rows)
    target = tensor(target_cm)
    starts = {r: current.bones[list(tree.subtree[r]), 4:7] for r in tree.roots}
    if mode == "own":
        factors = {r: torch.ones(len(tree.subtree[r]), 1) for r in tree.roots}
    elif mode == "factor":
        factors = {r: torch.full((len(tree.subtree[r]), 1), float(hold_factor)) for r in tree.roots}
    else:
        factors = {r: torch.tensor([[HOLD_FACTORS[tree.handle_names[i][:-2] if tree.handle_names[i][-2:] in ("_L", "_R")
                                                  else tree.handle_names[i]]] for i in tree.subtree[r]])
                   for r in tree.roots}

    def score(predictions):
        errors = []
        for root in tree.roots:
            ids = list(tree.subtree[root])
            change = predicted_rows(tree, predictions, root, body).reshape(candidates, len(ids), 6) * row_scales(
                tree.handle_names, ids, cfg["scales"], "change", rows["digit_scales"], body)
            one = end_positions(current, ids, change, tree.handle_names, rows["digit_frame"], body)
            held = starts[root] + factors[root] * (one - starts[root])
            errors.append((held - target[ids]).square().sum(-1))
        return torch.cat(errors, -1).mean(-1)

    choice = search(learner.model, state, recent, score, candidates=candidates, radius=radius, generator=generator,
                    sampling=sampling, code_range=code_range)
    order = handle_targets(tree, choice.predictions, current, template_cm, cfg["scales"],
                           cfg.get("hand_action_frame", "recorded"), **rows)
    return order, float(choice.scores[choice.index]), dict(
        selected=choice.index, candidates=choice.candidates, radius=radius,
        hold_scoring=mode, horizon_steps=int(cfg.get("horizon_steps", 1)),
        hold_factor=(1.0 if mode == "own" else hold_factor if mode == "factor" else "HOLD_FACTORS"),
        predicted_scores_quartiles=[float(q) for q in choice.scores.quantile(torch.tensor([.25, .5, .75]))],
        zero_code_score=float(choice.scores[0]), digit_frame=rows["digit_frame"], code_sampling=choice.sampling,
        candidate_code_range=choice.code_range)


def goal_error_cm(position_cm, target_cm, handles):
    p, t = np.asarray(position_cm, dtype=float)[list(handles)], np.asarray(target_cm, dtype=float)[list(handles)]
    return float(np.sqrt(np.square(p - t).sum(-1).mean()))


def goal_points(position_cm, quat_xyzw, ids, frame, names):
    """[len(ids), 3] the goal handles' positions in the goal's frame: world cm, or ("wrist") relative to the handle's
    own side's wrist in the wrist's coordinates."""
    from scipy.spatial.transform import Rotation
    pos = np.asarray(position_cm, dtype=np.float64)
    if frame == "world":
        return pos[list(ids)]
    if frame != "wrist":
        raise ValueError(f"Unknown goal frame {frame!r}")
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    out = []
    for i in ids:
        w = list(names).index("wrist_" + names[i][-1])
        out.append(Rotation.from_quat(quat[w]).inv().apply(pos[i] - pos[w]))
    return np.asarray(out)


def frame_error_cm(pos_a, quat_a, pos_b, quat_b, ids, frame, names):
    """Root mean square distance (cm) of handles `ids` between two poses, in the goal frame."""
    a, b = goal_points(pos_a, quat_a, ids, frame, names), goal_points(pos_b, quat_b, ids, frame, names)
    return float(np.sqrt(np.square(a - b).sum(-1).mean()))


def start_features(bones, names):
    """(positions [n,3] cm, velocities x VELOCITY_S [n,3] cm, up [3]) of every body handle (not the eyes) in the
    pelvis frame; bones [..., H, 11] (xyzw, world cm, world cm/s, capacity). Leading dimensions are kept."""
    from scipy.spatial.transform import Rotation
    bones = np.asarray(bones, dtype=np.float64)
    names = list(names)
    t = names.index("lower_trunk")
    rows = [i for i, n in enumerate(names) if n != "gaze"]
    lead = bones.shape[:-2]
    flat = bones.reshape(-1, *bones.shape[-2:])
    inv = Rotation.from_quat(flat[:, t, :4]).inv()
    pos = np.stack([inv.apply(flat[:, i, 4:7] - flat[:, t, 4:7]) for i in rows], 1)
    vel = np.stack([inv.apply(flat[:, i, 7:10]) for i in rows], 1) * VELOCITY_S
    up = inv.apply(np.tile([0., 0., 1.], (len(flat), 1)))
    return pos.reshape(*lead, len(rows), 3), vel.reshape(*lead, len(rows), 3), up.reshape(*lead, 3)


def start_distance(a, b):
    """cm: RMS over handles of the position difference plus RMS of the velocity term's difference (start_features)."""
    rms = lambda x, y: np.sqrt(np.square(x - y).sum(-1).mean(-1))
    return rms(a[0], b[0]) + rms(a[1], b[1])


class WitnessBank:
    """Recorded holds (latent_body_split --horizon-steps): start and end bones, the order, frames, per body."""
    def __init__(self, path):
        data = torch.load(path, weights_only=True, map_location="cpu")
        self.path = str(Path(path).resolve())
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.names = tuple(data["handle_names"])
        self.env_id = data["env_id"].numpy()
        self.frame = data["frame"].numpy()
        self.start = data["start"].numpy().astype(np.float64)
        self.end = data["end"].numpy().astype(np.float64)
        self.executed = data["executed"].numpy().astype(np.float64)
        self.horizon_steps = int(data["horizon_steps"])
        self.duration_s = float(data["duration_s"])
        self.action_contract = str(data["action_contract"])
        if not math.isclose(self.horizon_steps * self.duration_s, HOLD_S, rel_tol=1e-6):
            raise ValueError("The witness bank's holds are not the comparison's 0.25 s order hold")
        self.features = start_features(self.start, self.names)

    def nearest(self, body, bones, k, up_deg=UP_DEG):
        """[(bank index, start distance cm)] of the k nearest recorded hold starts of `body` lying within up_deg."""
        pos, vel, up = start_features(bones, self.names)
        mask = (self.env_id == body) & ((self.features[2] @ up) >= math.cos(math.radians(up_deg)))
        if not mask.any():
            return []
        idx = np.flatnonzero(mask)
        d = start_distance((self.features[0][idx], self.features[1][idx]), (pos[None], vel[None]))
        order = np.argsort(d, kind="stable")[:k]
        return [(int(idx[j]), float(d[j])) for j in order]

    def receipt(self):
        return dict(path=self.path, sha256=self.sha256, holds=len(self.env_id), horizon_steps=self.horizon_steps,
                    bodies={str(b): int((self.env_id == b).sum()) for b in sorted(set(self.env_id.tolist()))})


def transplant(order_cm, recorded_start, current, template_cm, names):
    """The recorded order [H,9] (cm, xyzw) carried from the recorded pelvis pose (recorded_start bones [H,11]) to the
    current one (current bones): every handle's position and orientation but the eyes'; radius, strength and the eyes'
    row are the controller's (template_cm [H,9] cm)."""
    from scipy.spatial.transform import Rotation
    names = list(names)
    t = names.index("lower_trunk")
    rs, rc = np.asarray(recorded_start, dtype=np.float64), np.asarray(current, dtype=np.float64)
    turn = Rotation.from_quat(rc[t, :4]) * Rotation.from_quat(rs[t, :4]).inv()
    order = np.asarray(order_cm, dtype=np.float64)
    out = np.array(template_cm, dtype=np.float64, copy=True)
    for i, n in enumerate(names):
        if n == "gaze":
            continue
        out[i, :3] = turn.apply(order[i, :3] - rs[t, 4:7]) + rc[t, 4:7]
        out[i, 3:7] = (turn * Rotation.from_quat(order[i, 3:7])).as_quat()
    return out


def request_change(request_cm, controller_cm, names, fingers=(), joints=None, controller_joints=None):
    """The witness request's change from the controller's as the production planners' search region bounds it
    (PhysicalTrustRegion.assemble): every body handle but the fingers (`fingers`, the hand geometry's finger rows) and the
    eyes, its largest per-coordinate position change (cm) and its turn (degrees); the hand joint goals (`joints` against
    `controller_joints`, radians), their largest change (degrees). inside_planner_region: all within the region."""
    from scipy.spatial.transform import Rotation
    from lem.model.efference import EFFERENCE_SCALES
    rows = [i for i, n in enumerate(names) if n != "gaze" and i not in set(fingers)]
    a, b = np.asarray(request_cm, dtype=np.float64)[rows], np.asarray(controller_cm, dtype=np.float64)[rows]
    position = float(np.abs(a[:, :3] - b[:, :3]).max())
    turn = float(np.degrees((Rotation.from_quat(a[:, 3:7]) * Rotation.from_quat(b[:, 3:7]).inv()).magnitude()).max())
    joint = None if joints is None else float(np.degrees(np.abs(
        np.asarray(joints, dtype=np.float64) - np.asarray(controller_joints, dtype=np.float64)).max()))
    cm, deg = float(EFFERENCE_SCALES["pos_tenure_cm"]), float(EFFERENCE_SCALES["rot_tenure_deg"])
    return dict(max_position_cm=position, max_turn_deg=turn, max_joint_deg=joint, region_cm=cm, region_deg=deg,
                inside_planner_region=bool(position <= cm and turn <= deg and (joint is None or joint <= deg)))


def choose_witness(candidates, goal, line=None):
    """The first candidate (start-similarity order) whose controller separation at `goal` reaches its line, or None."""
    line = SEPARATION_CM[goal] if line is None else line
    for k, c in enumerate(candidates):
        if c["separation_cm"][goal] >= line:
            return k
    return None


def wins_needed(n, alpha=LINES["alpha"]):
    """The fewest wins of n paired comparisons whose one-sided sign test tail is below alpha (12 -> 10, 20 -> 15)."""
    for k in range(n + 1):
        if sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n < alpha:
            return k
    return n + 1


def summarize(rows, held_out_bodies, *, alpha=LINES["alpha"], held_out_wins=None, primary="latent",
              min_admitted=MIN_ADMITTED, min_held_out_starts=MIN_HELD_OUT_STARTS):
    """rows: one per (body, start, goal) with outcomes[arm]['goal_error_cm']; rows with admitted False (no candidate
    cleared the separation line) are counted and left out of every test. Paired sign tests over the admitted starts.
    The held-out line takes one sign per held-out start (body, warmup): the mean over its admitted goals of (base error -
    primary error) / controller error. held_out_wins None: wins_needed(held-out starts)."""
    from lem.latent_body.metrics import paired_gain
    admitted = [r for r in rows if r.get("admitted", True)]
    held = [r for r in admitted if r["body"] in set(held_out_bodies)]
    held_starts = {}
    for r in held:
        held_starts.setdefault((r["body"], r.get("warmup")), []).append(r)
    if held_out_wins is None:
        held_out_wins = wins_needed(len(held_starts), alpha)
    out = dict(by_goal={}, held_out={}, primary=primary,
               lines=dict(alpha=alpha, held_out_wins=held_out_wins, min_admitted=min_admitted,
                          min_held_out_starts=min_held_out_starts))
    errors = lambda sel, arm: [r["outcomes"][arm]["goal_error_cm"] for r in sel]
    for goal in sorted({r["goal"] for r in rows}):
        sel = [r for r in admitted if r["goal"] == goal]
        block = dict(starts=sum(r["goal"] == goal for r in rows), admitted=len(sel), wins_needed=wins_needed(len(sel), alpha),
                     decided=len(sel) >= min_admitted)
        if sel:
            arms = [a for a in sel[0]["outcomes"] if all(a in r["outcomes"] for r in sel)]
            for base in ("controller",) + PLANNERS + tuple(a for a in arms if a.startswith("latent") and a != primary):
                block[f"{primary}_vs_{base}"] = paired_gain(errors(sel, base), errors(sel, primary), noise=0.0)
            block["beats_controller"] = {a: paired_gain(errors(sel, "controller"), errors(sel, a), noise=0.0)
                                         for a in arms if a not in ("controller", "witness")}
            block["means_cm"] = {a: float(np.mean(errors(sel, a))) for a in arms}
            closed = lambda a: [1 - r["outcomes"][a]["goal_error_cm"] / r["outcomes"]["controller"]["goal_error_cm"]
                                for r in sel if r["outcomes"]["controller"]["goal_error_cm"] > 0]
            block["gap_closed_median"] = {a: (float(np.median(closed(a))) if closed(a) else None)
                                          for a in arms if a != "controller"}
            if all("stay_cm" in r for r in sel):
                block["stay_cm_mean"] = float(np.mean([r["stay_cm"] for r in sel]))
            if all(r.get("witness_request_change") for r in sel):
                block["inside_planner_region"] = sum(r["witness_request_change"]["inside_planner_region"] for r in sel)
            block["better_than_planner"] = bool(block["decided"]) and all(
                block[f"{primary}_vs_{b}"]["wins"] >= block["wins_needed"]
                and block[f"{primary}_vs_{b}"]["one_sided_sign_p"] < alpha
                and block[f"{primary}_vs_{b}"]["mean_gain"] > 0 for b in PLANNERS)
        else:
            block["better_than_planner"] = False
        out["by_goal"][goal] = block
    def gap_gain(sel, base):
        values = [(r["outcomes"][base]["goal_error_cm"] - r["outcomes"][primary]["goal_error_cm"])
                  / r["outcomes"]["controller"]["goal_error_cm"] for r in sel
                  if r["outcomes"]["controller"]["goal_error_cm"] > 0]
        return float(np.mean(values)) if values else 0.0
    for base in ("controller",) + PLANNERS:
        if held_starts:
            gains = [gap_gain(sel, base) for sel in held_starts.values()]
            block = paired_gain(gains, [0.0] * len(gains), noise=0.0)
            block["unit"] = "per start: mean over its goals of (base error - primary error) / controller error"
            out["held_out"][f"{primary}_vs_{base}"] = block
    out["held_out"]["starts"] = len(held_starts)
    out["held_out"]["comparisons"] = len(held)
    out["held_out"]["passes"] = len(held_starts) >= min_held_out_starts and all(
        out["held_out"][f"{primary}_vs_{b}"]["wins"] >= held_out_wins for b in PLANNERS)
    out["latent_better_than_planner_goals"] = [g for g, b in out["by_goal"].items() if b["better_than_planner"]]
    out["not_admitted"] = {g: sum(1 for r in rows if r["goal"] == g and not r.get("admitted", True))
                           for g in sorted({r["goal"] for r in rows})}
    return out


def index_curl_request(session, raw, joints, side):
    """The controller's request with one index finger's knuckle, middle and distal joints at 95% of flexion."""
    geometry = session.model._hand_geometry
    prefix = "robot:%s_ff_" % ("left" if side == "L" else "right")
    research = joints.clone()
    touched = 0
    for index, joint in enumerate(geometry["joints"]):
        if joint["name"].startswith(prefix) and joint["name"].endswith(("_knuckle", "_middle", "_distal")):
            research[:, index] = .95 * joint["limits"][1]
            touched += 1
    if touched != 3:
        raise ValueError("Expected the index finger's knuckle, middle and distal joints")
    return raw.clone(), session.model.hand_kinematics.project_joints(research), (
        f"{side} index knuckle, middle and distal joints at 95% of their flexion limits (projected onto the "
        "finger coupling limits); every other request is the controller's")


def declared_goals(tree, text="", design="recorded_v2"):
    """{goal: (handles, frame)} named in `text` (comma-separated), or every goal of the design whose handles (and,
    recorded_v2, the handles that carry them; anchored_v1, the requests its recipe changes) the tree owns."""
    if design not in GOAL_DESIGNS:
        raise ValueError(f"Unknown goal design {design!r}")
    owners = set(tree.owners)
    if design == "anchored_v1":
        table = {g: (h, "world") for g, h in GOALS.items()}
        extra = GOAL_REQUESTS
    else:
        table, extra = GOALS_V2, GOAL_CARRIERS_V2
    held = lambda g: g in table and set(table[g][0]) | set(extra.get(g, ())) <= owners
    goals = {g: table.get(g) for g in (text.split(",") if text else [g for g in table if held(g)])}
    if not goals or not all(held(g) for g in goals):
        raise ValueError("Declare goals of the design whose handles the hierarchy's tree owns")
    return goals


def knee_flex_request(session, raw, joints, side, turn_deg):
    """The controller's request with one calf and foot turned turn_deg about the knee hinge's current world axis,
    toward flexion (MIMo flexes the knee toward its lower limit: turning by minus the angle about the hinge axis, as
    compare_hand_body_goals.elbow_flex_request does for the elbow). Leg positions are read by no executor."""
    from scipy.spatial.transform import Rotation
    env = session.env
    knee = env.model.joint("robot:%s_knee" % ("left" if side == "L" else "right")).id
    axis = np.asarray(env.data.xaxis[knee], dtype=np.float64)
    turn = Rotation.from_rotvec(-axis / np.linalg.norm(axis) * np.deg2rad(turn_deg))
    command = raw.clone()
    for name in ("calf_" + side, "foot_" + side):
        i = session.handle_names.index(name)
        command[0, i, 3:7] = torch.as_tensor((turn * Rotation.from_quat(command[0, i, 3:7].numpy())).as_quat(),
                                             dtype=command.dtype)
    return command, session.model.hand_kinematics.project_joints(joints.clone()), (
        f"Flex the {side} knee: the controller's calf and foot requests turned {turn_deg:g} degrees about the knee "
        "hinge's current axis, toward flexion; every other request is the controller's")


def hierarchy_exposure(path, learner):
    """What a hierarchy learned from: its fit receipt (receipt.json beside it) and its split receipt."""
    path = Path(path).resolve()
    fit = path.parent / "receipt.json"
    if not fit.exists():
        raise ValueError(f"No fit receipt beside {path}: the comparison records each side's training exposure")
    receipt = json.loads(fit.read_text(encoding="utf-8"))
    split_path = Path(receipt["data"]["path"]).parent / "split_receipt.json"
    if not split_path.exists():
        raise ValueError(f"No split receipt {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    distinct = int(split.get("train_examples", split["train_rows"]))
    horizon = int(learner.config.get("horizon_steps", 1))
    return dict(hierarchy=str(path), fit_receipt=str(fit), split_receipt=str(split_path),
                updates=int(learner.updates), example_presentations=int(learner.examples),
                distinct_training_examples=distinct, model_steps_per_example=horizon,
                distinct_body_moments=distinct * horizon, body_moment_presentations=int(learner.examples) * horizon,
                epochs=receipt.get("order", {}).get("epochs"), training_bodies=sorted(
                    {b for b, _ in receipt.get("training_body_episodes", [])}),
                held_out_examples=receipt.get("held_out_examples"),
                held_out_future_loss=receipt.get("held_out_hindsight_losses", {}).get("future"),
                held_out_future_if_nothing_moves=receipt.get("held_out_future_if_nothing_moves"))


def production_exposure(frozen):
    """The production model's live experience: its checkpoint step times its bodies per step (each body-moment one
    observed model step of one body), and its replay settings (windows of remembered moments; not counted here)."""
    step = frozen.load.get("step") if isinstance(frozen.load, dict) else None
    step = step if step is not None else frozen.info.get("lineage_step")
    bodies = int(getattr(frozen.args, "n_envs"))
    if step is None:
        raise ValueError("The checkpoint's step is unknown; the comparison records each side's training exposure")
    return dict(checkpoint=frozen.info.get("checkpoint"), checkpoint_step=int(step), bodies_per_step=bodies,
                live_body_moments=int(step) * bodies,
                replay=dict(replay=getattr(frozen.args, "replay", None), every_steps=getattr(frozen.args, "replay_every", None),
                            windows_per_pass=getattr(frozen.args, "replay_batch", None)))


def exposure_ratios(exposure):
    """Production live body-moments over each hierarchy's distinct body-moments and body-moment presentations."""
    live = exposure["production"]["live_body_moments"]
    return {label: dict(production_over_distinct=live / max(1, e["distinct_body_moments"]),
                        production_over_presentations=live / max(1, e["body_moment_presentations"]))
            for label, e in exposure["hierarchies"].items()}


def check_exposure(record):
    """Refuse a declaration or result without each side's training exposure."""
    e = record.get("training_exposure") or {}
    if not e.get("production") or not e.get("hierarchies") or set(e["hierarchies"]) != set(record.get("latent_arms", {})):
        raise ValueError("Every comparison records the production model's and each latent arm's training exposure")
    return record


def _compare_module(root):
    """docs/gain_of_function_v3/compare_hand_body_goals_2026-09-24.py: the planners' recorded comparison."""
    import importlib.util
    folder = root / "docs" / "gain_of_function_v3"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("compare_hand_body_goals", folder / "compare_hand_body_goals_2026-09-24.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pose_cm(pose):
    p = pose[0].detach().double().numpy() if torch.is_tensor(pose) else np.asarray(pose, dtype=np.float64)[0]
    return p[:, :3] * 200., p[:, 3:7]


def run(args):
    from unittest.mock import patch
    from mimo_bridge.tools import gof_probe_harness as H
    from mimo_bridge.tools.latent_body import load_learner, read_examples
    from mimo_bridge.latent_body_runtime import observation, observed_contract
    from mimo_bridge.hand_target_control import HandActionSpace, HandTargetControl
    from lem.model.physical_goal import pose_error_per_body
    from lem.latent_body.trajectory import row_options
    root = Path(__file__).resolve().parents[4]
    C = _compare_module(root)
    out = Path(args.output).resolve()
    if "saved" in {p.lower() for p in out.parts}:
        raise SystemExit("Keep comparison outputs outside Saved")
    started = time.time()
    # the latent arms: the primary (--hierarchy) and an optional second one (--step-hierarchy), each with its radius
    specs = [("latent", args.hierarchy, args.holdout)]
    if args.step_hierarchy:
        specs.append(("latent_step", args.step_hierarchy, args.step_holdout or args.holdout))
    latent = {}
    for label, path, holdout_path in specs:
        learner = load_learner(path)
        learner.model.eval()
        holdout = list(itertools.islice(read_examples(holdout_path), args.radius_rows))
        latent[label] = dict(learner=learner, path=str(Path(path).resolve()), holdout=str(holdout_path),
                             held_bodies={ex.start.env_id for ex in holdout},
                             exposure=hierarchy_exposure(path, learner),
                             **latent_arm(learner, holdout, args.code_sampling))
    primary = latent["latent"]["learner"]
    goals = declared_goals(primary.tree, args.goals, args.goal_design)
    for label, spec in latent.items():
        declared_goals(spec["learner"].tree, ",".join(goals), args.goal_design)   # every latent arm owns every goal
    bank = WitnessBank(args.witness_bank) if args.goal_design == "recorded_v2" else None
    if bank is not None and bank.names != primary.tree.handle_names:
        raise ValueError("Witness bank and hierarchy use different body layouts")
    if bank is not None and bank.action_contract != str(primary.config["action_contract"]):
        raise ValueError("Witness bank and hierarchy were recorded under different command meanings")
    bodies = [int(b) for b in args.bodies.split(",")]
    held_out = [int(b) for b in args.held_out_bodies.split(",")] if args.held_out_bodies else []
    for label, spec in latent.items():
        if not set(held_out) <= spec["held_bodies"]:
            raise ValueError(f"Held-out bodies {held_out} are not {label}'s held-out rows' bodies {spec['held_bodies']}")
    warmups = [int(w) for w in args.warmups.split(",")]
    held_warmups = [int(w) for w in args.held_out_warmups.split(",")]
    frozen = H.load_frozen(args.checkpoint, intention=True, threads=4)
    exposure = dict(production=production_exposure(frozen),
                    hierarchies={label: spec["exposure"] for label, spec in latent.items()})
    exposure["ratios"] = exposure_ratios(exposure)
    out.mkdir(parents=True, exist_ok=False)
    declaration = dict(test="latent_vs_planner_v2", goal_design=args.goal_design, checkpoint=str(Path(args.checkpoint).resolve()),
                       latent_arms={label: dict(hierarchy=spec["path"], sha256=hashlib.sha256(
                           Path(spec["path"]).read_bytes()).hexdigest(), updates=spec["learner"].updates,
                           config=spec["learner"].config, radius=spec["radius"], radius_rule="95th percentile of "
                           f"|root code| over {args.radius_rows} rows of {spec['holdout']}",
                           code_sampling=spec["sampling"], code_sampling_requested=args.code_sampling,
                           code_range=spec["code_range"].receipt(),
                           hold_scoring=hold_scoring(spec["learner"].config, row_options(spec["learner"].config)["body"])[0])
                           for label, spec in latent.items()},
                       primary="latent", goals={g: dict(handles=h, frame=f) for g, (h, f) in goals.items()},
                       separation_cm={g: SEPARATION_CM.get(g) for g in goals} if bank is not None else None,
                       witness_bank=None if bank is None else bank.receipt(), witnesses_per_start=args.witnesses,
                       up_deg=UP_DEG, velocity_s=VELOCITY_S, arms=ARMS + tuple(l for l in latent if l != "latent")
                       + (("witness",) if bank is not None else ()),
                       bodies=bodies, held_out_bodies=held_out, warmups=warmups, held_out_warmups=held_warmups,
                       candidates=args.candidates, horizon_s=HOLD_S, seed=args.seed, lines=LINES,
                       training_exposure=exposure, doc=__doc__, authority="test_world_only", production_control=False)
    check_exposure(declaration)
    (out / "declaration.json").write_text(json.dumps(declaration, indent=1, default=str), encoding="utf-8")
    generators = {label: torch.Generator().manual_seed(args.seed + k) for k, label in enumerate(latent)}
    episode, note = H.load_episode(frozen)
    if episode is None:
        raise RuntimeError(note)
    model = frozen.model
    if model.synergy is None:
        raise RuntimeError("Learned synergy required for the synergy planner arm")
    names = primary.tree.handle_names
    if tuple(model.handle_names) != names:
        raise ValueError("Hierarchy and checkpoint use different body layouts")
    initial_weights = C.checksum(model)
    settings = {k: getattr(model, k) for k in ("hand_target_mode", "hand_goals", "synergy_planner", "physical_goals")
                if hasattr(model, k)}
    body_rows = [i for i, n in enumerate(names) if n != "gaze"]
    rows = []
    plan = [(b, w) for b in bodies for w in (held_warmups if b in held_out else warmups)]
    for body, warm in plan:
        for key, value in settings.items():
            setattr(model, key, value)
        session = H.Session(frozen, episode=copy.deepcopy(episode), body=body)
        try:
            if primary.config["action_contract"] != observed_contract(session.frames[0]):
                raise ValueError("Hierarchy and world use different command meanings")
            session.observe()
            for _ in range(warm - 1):
                session.policy_act()
                session.observe()
            history = {label: observation(session.frames[0], spec["learner"].tree, 0, 0) for label, spec in latent.items()}
            session.policy_act()
            # the decision moment, read as compare_hand_body_goals reads it
            model.hand_target_mode = "world"
            model.hand_goals = "off"
            model.synergy_planner = False
            previous_motor = None if model._prev_mfa is None else model._prev_mfa.detach().clone()
            previous_hand = getattr(model, "_hand_previous_joints", None)
            previous_hand = None if previous_hand is None else previous_hand.detach().clone()
            intention, model.intention_gen = model.intention_gen, None
            try:
                session.observe()
            finally:
                model.intention_gen = intention
            model._prev_mfa = previous_motor
            current = {label: observation(session.frames[0], spec["learner"].tree, 1, 0) for label, spec in latent.items()}
            bones = current["latent"].bones.numpy().astype(np.float64)
            env = session.env
            env.hand_target_mode = "wrist_relative_v1"
            env.hand_target_control = HandTargetControl(env.model, env.decoder)
            session.src.env_kwargs["hand_target_mode"] = "wrist_relative_v1"
            control = env.hand_target_control
            kin, geometry = control.kinematics, control.geometry
            previous = env.data.qpos[control._joint_addresses].copy()[None] if previous_hand is None else previous_hand
            model.configure_hand_geometry(geometry, previous)
            model.hand_target_mode = "wrist_relative_v1"
            model.physical_goals = "memory"
            raw = torch.from_numpy(session.res.motor_out).float()
            canonical, initial_joints, _ = kin.project_world_orders(raw.numpy())
            raw = torch.from_numpy(canonical).float()
            initial_joints = torch.from_numpy(initial_joints).float()
            decision = model._capture_plan_state()
            tokens = {k: v.squeeze(1).detach().float().clone() for k, v in model.last_tokens.items()}
            snapshot = session.snapshot()
            base_command, base_joints = C.filtered(model, raw, initial_joints, decision)
            gaze = np.asarray(model._gaze_from_handle(raw)).reshape(-1)[:2]
            controller_cm = raw[0].double().numpy().copy()
            controller_cm[:, :3] *= 200.
            fingers = sorted(i for h in geometry["hands"] for i in h["finger_indices"])

            def execute(command, joints, target=None, target_joints=None, label=""):
                result, pose, actual = C.physical(session, snapshot, command, joints, gaze, target, target_joints)
                repeat, _, _ = C.physical(session, snapshot, command, joints, gaze, target, target_joints)
                if result != repeat:
                    raise RuntimeError(f"{label} replay differs")
                return result, pose, actual

            # ---------------------------------------------------------------- the goals of this start
            trials = []      # (goal names, target pose, target joints, goal record, witness arm or None)
            start_record = dict(body=body, warmup=warm)
            if bank is not None:
                _, control_pose, _ = execute(base_command, base_joints, label="controller")
                control_pos, control_quat = _pose_cm(control_pose)
                candidates = []
                for index, distance in bank.nearest(body, bones, args.witnesses):
                    request_cm = transplant(bank.executed[index], bank.start[index], bones, controller_cm, names)
                    request = torch.from_numpy(request_cm).float()[None].clone()
                    request[..., :3] /= 200.
                    projected, joints, _ = kin.project_world_orders(request.numpy())
                    projected_cm = np.asarray(projected, dtype=np.float64)[0].copy()
                    projected_cm[:, :3] *= 200.
                    command, cjoints = C.filtered(model, torch.from_numpy(projected).float(),
                                                  torch.from_numpy(joints).float(), decision)
                    result, pose, actual = execute(command, cjoints, label="witness")
                    pos, quat = _pose_cm(pose)
                    candidates.append(dict(
                        bank_index=index, recorded_frame=int(bank.frame[index]), start_distance_cm=distance,
                        separation_cm={g: frame_error_cm(pos, quat, control_pos, control_quat,
                                                         [names.index(h) for h in hs], f, names)
                                       for g, (hs, f) in goals.items()},
                        request_change=request_change(projected_cm, controller_cm, names, fingers,
                                                      np.asarray(joints)[0], initial_joints.numpy()[0]),
                        trajectory_sha256=result["trajectory_sha256"], pose=pose, joints=actual,
                        command=command, hand_joints=cjoints))
                start_record["candidates"] = [{k: v for k, v in c.items() if k not in ("pose", "joints", "command",
                                                                                         "hand_joints")}
                                              for c in candidates]
                chosen = {}
                for goal in goals:
                    k = choose_witness(candidates, goal)
                    if k is None:
                        rows.append(dict(start_record, goal=goal, admitted=False, held_out=body in held_out,
                                         separation_line_cm=SEPARATION_CM[goal]))
                        with (out / "trials.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(rows[-1], default=str) + "\n")
                        continue
                    chosen.setdefault(k, []).append(goal)
                for k, goal_names in chosen.items():
                    c = candidates[k]
                    witness = (c["command"], c["hand_joints"], dict(bank_index=c["bank_index"],
                                                                    trajectory_sha256=c["trajectory_sha256"]))
                    trials.append((goal_names, c["pose"], c["joints"], dict(
                        witness=dict(bank_index=c["bank_index"], recorded_frame=c["recorded_frame"], candidate_rank=k,
                                     start_distance_cm=c["start_distance_cm"], separation_cm=c["separation_cm"],
                                     trajectory_sha256=c["trajectory_sha256"]),
                        witness_request_change=c["request_change"],
                        recipe="the recorded order of this body nearest in start (rank %d), carried to the current "
                               "pelvis; radius, strength and the eyes the controller's" % k), witness))
            else:
                for goal in goals:
                    session.restore(snapshot)
                    if goal == "wrist_L_reach":
                        request, research, recipe = C.witness_request(session, raw, initial_joints, "wrist_reach_witness")
                    elif goal == "trunk_turn":
                        request, research, recipe = C.witness_request(session, raw, initial_joints,
                                                                      "supported_trunk_turn_witness")
                    elif goal == "knee_L_flex":
                        request, research, recipe = knee_flex_request(session, raw, initial_joints, "L", C.TURN_DEG)
                    else:
                        request, research, recipe = index_curl_request(session, raw, initial_joints, goal[6])
                    ref_command, ref_joints = C.filtered(model, request, research, decision)
                    reference, target, target_joints = execute(ref_command, ref_joints, label="reference")
                    trials.append(([goal], target, target_joints, dict(recipe=recipe,
                                   reference_sha256=reference["trajectory_sha256"]), None))
            # ---------------------------------------------------------------- the arms, once per goal pose
            for goal_names, target, target_joints, goal_record, witness in trials:
                began = time.perf_counter()
                target_cm = target[0, :, :3].numpy() * 200.
                trial = dict(decision, physical_goal=dict(pose=target, valid=torch.ones(1, dtype=torch.bool),
                                                          horizon_ms=250., source="same_state_physical_reference"))
                arms = {"controller": (base_command, base_joints, {})}

                class MatchedControls(HandActionSpace):
                    def assemble(self):
                        command, strength, penalty = super().assemble()
                        command = command.clone()
                        command[..., 8:9] = raw[..., 8:9]
                        command[:, model.GAZE_IDX] = raw[:, model.GAZE_IDX]
                        return command, raw[..., 8:9], penalty
                for mode, synergy in (("direct", False), ("synergy", True)):
                    session.restore(snapshot)
                    model.synergy_planner = synergy

                    def held_gaze(cur, goal_state, motor, decision_, execution_transform):
                        return motor, {"stopped_by": "gaze_held_for_matched_body_comparison"}
                    with patch("mimo_bridge.hand_target_control.HandActionSpace", MatchedControls), \
                            patch.object(model, "_plan_visual_gaze", held_gaze), torch.enable_grad():
                        chain, meta = model._plan_chain(tokens, tokens, raw, max_steps=1, decision=trial)
                    if not chain:
                        raise RuntimeError(f"{mode} planning rejected: {meta}")
                    arms[mode] = (*C.filtered(model, chain[0], meta["hand_joint_goal_raw"], trial), meta["descent"])
                model.synergy_planner = False
                session.restore(snapshot)
                template = raw[0].clone()
                template[:, :3] *= 200.
                for label, spec in latent.items():
                    learner = spec["learner"]
                    order, predicted, receipt = latent_order(learner, history[label], current[label], target_cm, template,
                                                             candidates=args.candidates,
                                                             generator=generators[label], **search_settings(spec))
                    placed = sorted(i for ids in learner.tree.owned.values() for i in ids if names[i] != "gaze")
                    request = raw.clone()
                    request[0, placed, :7] = order[placed, :7]
                    request[0, placed, :3] /= 200.
                    projected, latent_joints, _ = kin.project_world_orders(request.numpy())
                    arms[label] = (*C.filtered(model, torch.from_numpy(projected).float(),
                                               torch.from_numpy(latent_joints).float(), trial),
                                   dict(receipt, latent_predicted_mean_sq_cm=predicted))
                if witness is not None:
                    arms["witness"] = witness
                outcomes = {}
                target_pos, target_quat = _pose_cm(target)
                for mode, (command, joints, notes) in arms.items():
                    result, pose, _ = execute(command, joints, target, target_joints, label=mode)
                    pos, quat = _pose_cm(pose)
                    forecast = C.forecast_pose(model, tokens, trial, command)
                    outcomes[mode] = dict(
                        goal_error_cm={g: frame_error_cm(pos, quat, target_pos, target_quat,
                                                         [names.index(h) for h in goals[g][0]], goals[g][1], names)
                                       for g in goal_names},
                        goal_error_world_cm={g: goal_error_cm(pos, target_pos, [names.index(h) for h in goals[g][0]])
                                             for g in goal_names},
                        tree_error_cm=goal_error_cm(pos, target_pos, sorted(
                            i for ids in primary.tree.owned.values() for i in ids)),
                        pose_error=result["physical_goal_error"],
                        production_forecast_pose_error=float(pose_error_per_body(
                            forecast[..., :3] * 200, forecast[..., 3:7], target, body_rows)[0]),
                        trajectory_sha256=result["trajectory_sha256"], replayed_exactly=True,
                        command=command.tolist(), notes=notes if mode.startswith("latent") or mode == "witness" else
                        (dict(descent_steps=len(notes)) if notes else {}))
                start_pos, start_quat = bones[:, 4:7], bones[:, :4]
                if witness is not None and outcomes["witness"]["trajectory_sha256"] != witness[2]["trajectory_sha256"]:
                    # the goal's own order did not repeat its candidate execution: these goals are not tested
                    for goal in goal_names:
                        rows.append(dict(body=body, warmup=warm, goal=goal, admitted=False, held_out=body in held_out,
                                         reason="the witness re-execution differed from its candidate execution",
                                         **goal_record))
                        with (out / "trials.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(rows[-1], default=str) + "\n")
                    continue
                for goal in goal_names:
                    ids = [names.index(h) for h in goals[goal][0]]
                    per_arm = {}
                    for mode, o in outcomes.items():
                        per_arm[mode] = dict(o, goal_error_cm=o["goal_error_cm"][goal],
                                             goal_error_world_cm=o["goal_error_world_cm"][goal])
                    control = per_arm["controller"]["goal_error_cm"]
                    for mode, o in per_arm.items():
                        o["gain_over_controller_cm"] = control - o["goal_error_cm"]
                        o["gap_closed"] = (1 - o["goal_error_cm"] / control) if control > 0 else None
                    if witness is not None and per_arm["witness"]["goal_error_cm"] != 0.0:
                        raise RuntimeError("The witness order did not end on its own goal")
                    row = dict(start_record if bank is None else dict(body=body, warmup=warm), goal=goal,
                               admitted=True, held_out=body in held_out, frame=goals[goal][1], **goal_record,
                               stay_cm=frame_error_cm(start_pos, start_quat, target_pos, target_quat, ids,
                                                      goals[goal][1], names),
                               goal_moved_cm=goal_error_cm(start_pos, target_pos, ids),
                               outcomes=per_arm, shared_goals=goal_names, elapsed_s=time.perf_counter() - began)
                    rows.append(row)
                    with (out / "trials.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, default=str) + "\n")
                    print(body, warm, goal, {k: round(v["goal_error_cm"], 3) for k, v in per_arm.items()}, flush=True)
            if bank is not None:
                with (out / "starts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(start_record, default=str) + "\n")
        finally:
            session.close()
    if C.checksum(model) != initial_weights:
        raise RuntimeError("Frozen learned weights changed")
    result = dict(declaration, summary=summarize(rows, held_out), trials=len(rows), elapsed_s=time.time() - started,
                  completed=True)
    check_exposure(result)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps(dict(summary=result["summary"], training_exposure=exposure), indent=1, default=str))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("checkpoint", "hierarchy", "holdout", "output"):
        ap.add_argument("--" + flag, required=True)
    ap.add_argument("--goal-design", choices=GOAL_DESIGNS, default="recorded_v2")
    ap.add_argument("--witness-bank", default="", help="recorded_v2: latent_body_split --horizon-steps 5's witness_bank.pt")
    ap.add_argument("--witnesses", type=int, default=8, help="recorded_v2: candidate recorded orders per start")
    ap.add_argument("--step-hierarchy", default="", help="A second latent arm (latent_step), reported")
    ap.add_argument("--step-holdout", default="", help="Its held-out rows (default --holdout)")
    ap.add_argument("--bodies", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--held-out-bodies", default="7")
    ap.add_argument("--warmups", default="1,21", help="Policy steps before each start, one start per value")
    ap.add_argument("--held-out-warmups", default="1,11,21,31,41,51,61,71",
                    help="8 held-out starts: the held-out line takes one sign per start and needs at least 5")
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--radius-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goals", default="", help="Comma-separated goals of the design; default: every goal the tree owns")
    ap.add_argument("--code-sampling", choices=("auto", "box", "encoded", "gaussian"), default="auto",
                    help="Latent candidates: auto (encoded for a whole-body hierarchy, box for the pilot), box, encoded, "
                         "gaussian")
    args = ap.parse_args(argv)
    if args.candidates < 2:
        raise SystemExit("Declare at least 2 latent candidates")
    if args.goal_design == "recorded_v2" and (not args.witness_bank or args.witnesses < 1):
        raise SystemExit("recorded_v2 goals need --witness-bank and at least one witness per start")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
