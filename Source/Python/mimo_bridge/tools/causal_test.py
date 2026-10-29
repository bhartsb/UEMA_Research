# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The five-control causal test (plan item 6). The gate on item 7.

    python -m mimo_bridge.tools.causal_test --ckpt <dir> [--n 400] [--seeds 3]

The question. Does the model predict the next moment BETTER when handed the order the body
actually executed than when handed something else? If not it has learned continuation -
"given where the body is and how it is moving, what tends to happen next" - and not
intervention - "from this state, what happens if I do A rather than B". Only the second
supports intention, counterfactual imagination or reaching.

REWRITTEN 2026-08-25. The first version was not a valid gate. Six defects, all fixed here:

  1. WRONG ACTION INDEX. It scored transition i with the action that ARRIVED with state i -
     an order that had already finished acting. The action carrying state i to state i+1
     arrives one step later. Motor holding hid this on most limb transitions but never at a
     hold boundary, and gaze is not held at all.
  2. UNRELATED CONTEXT. Every transition was scored against whatever temporal context the
     collection loop left behind - in practice the final context of a different body's run.
     Each transition now carries the context snapshotted when it was observed.
  3. BROKEN PREDICTION ROUTE. It called predict_from_action(), which encodes the motor token
     without the registry's output LayerNorm and splices it into post-transformer hidden
     states without rerunning the transformer. That is the representation mismatch behind
     the planner saturation; measuring with it measures the defect. Prediction now goes
     through the registry encode path and the full forward model, as the live step does.
  4. UNMATCHED PASSIVE CONTROL. It handed an action from a different limp-body run to an
     active-body target. The control now runs THIS body with the actuators disabled.
  5. SHUFFLED ACTIONS COULD BE IDENTICAL. Under motor_hold=5 another timestep often holds
     the same command. Substitutes must now differ materially, and skips are reported.
  6. HARDCODED CONFIG. It fixed context_length and think_harder instead of rebuilding the
     configuration the checkpoint was trained under, from the run's args.json.

PASS requires the correct action to beat all five controls on every real modality, with the
spread across seeds reported. Beating on the mean is not beating.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The four ACTION substitutions plus persistence. "passive" is deliberately NOT here:
# it is a physical branch measurement, not a prediction score. Listing it as a scored
# control produced a column equal to "correct" by construction, which is meaningless -
# and before that, an apples-to-oranges comparison against a different data stream.
# PERTURBED is the control that cannot be passed by learning which action BELONGS to a
# state. Measured on H35 with the planner on, 74 transitions: substituting another
# transition's action (distance 14.7) cost 20-68%, while perturbing the model's OWN
# chosen action by as much as 11.0 cost about 1%. Distance was not what the model
# responded to - kind was. shuffled and cross-body both take an action chosen for a
# DIFFERENT state, so both can be beaten by a model that only learned the state-to-action
# mapping and never learned what an action causes. A perturbation stays plausible for
# THIS state, so beating it requires predicting a consequence.
CONTROLS = ["correct", "persistence", "state-only", "shuffled", "cross-body", "perturbed"]
PERTURB_FRACTION = 0.5      # of each column's own standard deviation over the stream
MIN_ACTION_DIFF = 1e-3      # a substitute equal to the real action tests nothing
from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, MOTOR_INPUT_F  # 10, 19


def body_rows_differ(cand, a_true, gaze_idx, min_diff=MIN_ACTION_DIFF):
    """True when two orders differ by more than min_diff in some BODY row: every row but the eyes'
    (review E14: a substitute differing only in the eyes' row tests no limb order). The eyes' row
    is left out by its index in the model's layout; until 2026-09-23 this compared rows 0-22,
    which in trunk_v1 also left out the lower trunk and the chest (fix-following-gauge)."""
    keep = [i for i in range(a_true.shape[-2]) if i != int(gaze_idx)]
    return float((cand[..., keep, :] - a_true[..., keep, :]).abs().max()) > min_diff


def _null_action(like):
    """"No action information", as a VALID motor command.

    torch.zeros_like gives an all-zero quaternion, which is not a rotation at all. That
    is off-distribution rather than uninformative, and makes the state-only control look
    artificially bad. The identity quaternion with zero strength is the real "do nothing".
    """
    a = torch.zeros_like(like)
    a[..., 6] = 1.0                      # w = 1: identity rotation
    return a


def _limits_for_run(ckpt, model):
    """The developmental limits the checkpoint TRAINED under, from the run's own log.

    The trainer takes its limits from the CurriculumManager when --curriculum is on (every
    recent run), and logs rom_limit / strength_limit / phase in every mimo record. The
    defaults table in lem/config/defaults.py is a different schedule (four phases, child
    at the top) and is only the fallback when the log is missing.
    """
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    p = os.path.join(run_dir, "replay_training_stats.jsonl")
    saved_args={}
    # beside the checkpoint, else the run folder its lineage.json names (archived on D:)
    args_path,_=_run_args_path(ckpt)
    if os.path.isfile(args_path):
        with open(args_path,encoding='utf-8') as stream:saved_args=json.load(stream)
    # External checkpoint storage and the canonical live run directory need not
    # share a parent. Its args.out names the actual log owner; never silently
    # substitute the separate step-based defaults because the USB has no log.
    canonical=saved_args.get('out')
    if canonical:
        alternate=os.path.join(os.path.abspath(canonical),'replay_training_stats.jsonl')
        if os.path.isfile(alternate):p=alternate
    metadata_path=os.path.join(os.path.abspath(ckpt),'meta.json')
    metadata={}
    if os.path.isfile(metadata_path):
        with open(metadata_path,encoding='utf-8') as stream:metadata=json.load(stream)
    recorded_phase=(metadata.get('curriculum_state') or {}).get('phase')
    # review E17: the LAST record described end-of-run limits; an early checkpoint
    # of a phase-advancing run was tested under limits it never trained with. The
    # record nearest the checkpoint's own step wins.
    _m = re.search(r"ckpt_step_(\d+)", os.path.basename(os.path.normpath(ckpt)))
    _ck_step = int(_m.group(1)) if _m else None
    last, best_d = None, None
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"mimo"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") == "mimo" and r.get("rom_limit") is not None:
                    if recorded_phase and r.get('phase')!=recorded_phase:
                        continue
                    d = (abs(int(r.get("step", 0)) - _ck_step)
                         if _ck_step is not None else 0)
                    if best_d is None or d <= best_d:
                        last, best_d = r, d
    except OSError:
        last = None
    if last is not None:
        return ({"phase_name": str(last.get("phase")), "rom_limit": float(last["rom_limit"]),
                 "strength_limit": float(last.get("strength_limit", 1.0)),
                 "age_months": last.get("age_months"),
                 "exploration_noise": 0.0}, "run log: " + p)
    if saved_args.get('curriculum')=='on':
        phase=recorded_phase
        from lem.training.curriculum import PHASE_DEFINITIONS
        if phase not in PHASE_DEFINITIONS:
            raise ValueError('Curriculum checkpoint has no canonical limits log or recorded native phase; defaults are not evidence')
        definition=PHASE_DEFINITIONS[phase]
        from mimo_bridge.train_mimo import PHASE_AGE
        return (dict(phase_name=phase,rom_limit=definition.rom_limit,strength_limit=definition.strength_limit,
                     age_months=PHASE_AGE.get(phase) if saved_args.get('age_binding')=='curriculum' else None,
                     exploration_noise=0.0),
                'recorded curriculum phase in '+metadata_path+'; native CurriculumManager phase definition (no run log)')
    m = re.search(r"ckpt_step_(\d+)", os.path.basename(os.path.normpath(ckpt)))
    step = int(m.group(1)) if m else 10 ** 9
    return (dict(model.dev_limits(step)), f"defaults table at step {step} (no run log found)")


_SNAP_PLAIN = (int, float, bool, str, bytes, type(None))


def _snap_helper_state(o):
    """review E10: copy a helper's mutable plain state (numbers, arrays, containers,
    rng states); skip anything heavy or shared (MuJoCo handles, adapter backrefs)."""
    import copy as _c
    st = {}
    for k, v in o.__dict__.items():
        if isinstance(v, _SNAP_PLAIN):
            st[k] = v
        elif isinstance(v, np.ndarray):
            st[k] = ("__arr__", v.copy())
        elif isinstance(v, np.random.Generator):
            st[k] = ("__rng__", _c.deepcopy(v.bit_generator.state))
        elif isinstance(v, (list, dict, tuple, set)):
            try:
                st[k] = ("__obj__", _c.deepcopy(v))
            except Exception:
                pass
    return st


def _restore_helper_state(o, st):
    for k, v in st.items():
        if isinstance(v, tuple) and len(v) == 2 and v[0] in ("__arr__", "__obj__", "__rng__"):
            tag, payload = v
            if tag == "__rng__":
                cur = o.__dict__.get(k)
                if isinstance(cur, np.random.Generator):
                    cur.bit_generator.state = payload
            else:
                o.__dict__[k] = payload
        else:
            o.__dict__[k] = v


def _storage_locations_args(ckpt):
    """args.json through mimo_bridge.storage_locations, when that module exists.

    Its entry point is resolve_run_args(checkpoint) (2026-09-22; it also follows a
    checkpoint that a manifest says was moved). Any failure means "not found here", never an
    error: this is the last of three places looked, and a miss is reported by the caller as
    found=False."""
    try:
        from mimo_bridge import storage_locations as locations
    except Exception:
        return None
    for name in ("resolve_run_args",):
        fn = getattr(locations, name, None)
        if not callable(fn):
            continue
        try:
            found = fn(ckpt)
        except Exception:
            continue
        if found and os.path.isfile(str(found)):
            return os.path.abspath(str(found))
    return None


def _run_args_path(ckpt):
    """(path, found) of the args.json of the run that WROTE this checkpoint.

    Three places, in order: beside it (<run>/checkpoints/ckpt_step_N, the trainer's own
    layout); the run folder its lineage.json names in "out" (the checkpoint archiver moves
    checkpoints to D: without their run's args.json - measured 2026-09-22 on segment 31's
    archive folder, which holds checkpoints/ only); mimo_bridge.storage_locations when it
    exists. When none has it the nominal beside-path is returned with found=False."""
    ck = os.path.abspath(ckpt)
    beside = os.path.join(os.path.dirname(os.path.dirname(ck)), "args.json")
    if os.path.isfile(beside):
        return beside, True
    lineage = os.path.join(ck, "lineage.json")
    if os.path.isfile(lineage):
        try:
            with open(lineage, encoding="utf-8") as stream:
                out = json.load(stream).get("out")
        except Exception:
            out = None
        if out and os.path.isfile(os.path.join(out, "args.json")):
            return os.path.abspath(os.path.join(out, "args.json")), True
    other = _storage_locations_args(ck)
    if other:
        return other, True
    return beside, False


def _cfg_from_run(ckpt):
    """The configuration the checkpoint was trained under, not a guess."""
    p, found = _run_args_path(ckpt)
    if not found:
        return {}, p, False
    with open(p, encoding="utf-8") as stream:
        return json.load(stream), p, True


# Behaviour-shaping flags that act only through the planner. causal_test's rule since
# 2026-09-06 is that they are armed only when the caller asks for the planner (intention
# on): the agency drive refuses to run without it (R4 guard), and a goal picker or an
# effort charge with no planner is a different machine from the one the run trained.
_PLANNER_ONLY_FLAGS = ("agency_drive", "curiosity_goal", "goal_locality", "planner_effort_per_body")
# Settings whose construction requires the planner, with the value that builds without it. The
# gain-of-function build added --wellness-rank normalized, which ranks the goal ring and refuses
# to build with --intention off (lem_model._init_wellness_channels). The ranking has no learned
# weights (its running statistics sit in runtime_state.pt and a raw build ignores them), so a
# planner-off probe of an upgraded checkpoint builds with the raw ranking (review of the probes,
# 2026-09-23: the causal test and every tool on _build_model refused the upgraded lineage).
_PLANNER_ONLY_SETTINGS = {"wellness_rank": "raw"}


def _probe_configuration(saved, device, seed, intention, arg_overrides):
    """The trainer's own argument namespace for a frozen probe of this checkpoint.

    The same contract as self_recovery_experiment.configuration(): the trainer's parse_args
    defaults, overwritten by the run's saved args.json, with a single body. It differs in
    two places only, both needed by this module's callers: the args.json is found through
    _run_args_path (archived checkpoints), and the device and seed are the caller's.
    Then this module's two existing rules: intention is the caller's choice (default off -
    the planner is not under test) and the planner-only flags follow it; arg_overrides are
    applied last and recorded by the caller."""
    from mimo_bridge.train_mimo import parse_args
    a = parse_args(["--out", os.path.join(ROOT, "_causal_test_probe_unused_out")])
    vars(a).update(saved or {})
    if arg_overrides:
        # A/B probes force a flag the checkpoint's run never set (prospective-
        # wellness offline comparison, 2026-09-09). Overrides are RECORDED by
        # the caller; silent divergence from the run's args stays banned.
        vars(a).update(arg_overrides)
    a.n_envs = 1
    a.device = str(device)
    a.seed = int(seed)
    a.intention = "on" if intention else "off"
    if not intention:
        for name in _PLANNER_ONLY_FLAGS:
            setattr(a, name, "off")
        for name, value in _PLANNER_ONLY_SETTINGS.items():
            setattr(a, name, value)
    return a


def _build_model(ckpt, device, seed, intention=False, arg_overrides=None, load=True):
    """The checkpoint's model, built by the TRAINER's own builder.

    REPAIRED 2026-09-22 (gain-of-function build). This function used to hand-copy about
    forty LemLite keyword arguments from args.json; every flag added after the copy was
    silently dropped (the plumbing-hole class this file's comments count five instances
    of), and H84 could not be loaded at all: its checkpoint refused with "checkpoint has
    learned grounded self dynamics; retain --self-prediction grounded_residual", because
    self_prediction was never passed (measured on ckpt_step_344400 by the training-probes
    survey). It now builds through train_mimo.build_learner on the trainer's own argument
    namespace (_probe_configuration), so every flag the trainer passes reaches the probe.

    Returns (model, args, args_path, found). `args` is the dict of the namespace the model
    was built from - the saved args.json filled out with the trainer's defaults, the
    caller's overrides applied, and the run's own seed kept (the probe's seed is used for
    the build only)."""
    from mimo_bridge import train_mimo
    saved, src, found = _cfg_from_run(ckpt)
    ns = _probe_configuration(saved, device, seed, intention, arg_overrides)
    m = train_mimo.build_learner(ns, None)
    # The developmental schedule's fallback reader (model.dev_limits) as this function
    # always configured it; the live limits still come from _limits_for_run.
    training = m.config.setdefault("training", {})
    training["steps_per_phase"] = int(getattr(ns, "steps_per_phase", 0) or 50_000)
    training["start_phase"] = str(getattr(ns, "start_phase", "") or "neonate")
    a = dict(vars(ns))
    a["seed"] = (saved or {}).get("seed", a["seed"])
    if load:
        m.load_checkpoint(ckpt, load_optimizer=False)
    return m, a, src, found


def world_kwargs(a, limits, scene, run_dir, cmd_horizon):
    """The world the causal test collects in: the run's own world, with the probe's settings on top.

    REPAIRED 2026-09-23 (fix-rows-r1-causal_test.py). run() used to copy ten keys of args.json into
    a world of BridgeEnv defaults. body_layout was not one of them, so a trunk_v1 checkpoint (every
    H84 segment: 26 body rows, the lower trunk and the chest after the eyes' row 23) met a legacy
    24-row world, and its first 26-row order stopped the test ("cannot reshape array of size 234
    into shape (24,9)"). The copy also left out the contact profile, the body function, the touch
    geometry, the skin profile, real audio, the vision ring the gaze reflex reads, and the phase's
    motor delay. train_mimo.env_kwargs_for, the trainer's own world builder, carries every one of
    them from the run's namespace `a` (the dict _build_model returns).

    The probe's own, as before this repair: the scene and the folder its scene files go to (run_dir,
    never the run's folder), vision on, the order horizon (cmd_horizon, action_repeat x motor_hold as
    _collect holds orders), the age the run log recorded nearest the checkpoint's step when there is
    one, and the critters, the mobile and the caregiver off (the ball is still put back in view after
    caregiver_seconds out of sight, as before). The run's caregiver tape is never recorded to or
    replayed. seed and tag are left out: each stream sets its own."""
    from argparse import Namespace
    from mimo_bridge import train_mimo
    phase = limits.get("phase_name")
    if phase not in train_mimo.PHASE_AGE:
        phase = None            # a phase the trainer's tables do not name sets no age and no delay
    kw = train_mimo.env_kwargs_for(Namespace(**a), phase, run_dir)
    kw.pop("seed", None)
    kw.pop("tag", None)
    kw.update(scene=scene, run_dir=run_dir, vision=True, cmd_horizon=int(cmd_horizon),
              caregiver_mode="off", caregiver_record_dir="", caregiver_replay="",
              critters=False, mobile=False)
    if limits.get("age_months") is not None:
        kw["age"] = float(limits["age_months"])
    return kw


def _command_pathway_weight(model):
    """|w| of the motor encoder's COMMAND columns (10-18), or None if not found.

    Zero-growth checkpoint surgery pads new columns with zeros so a grown network
    computes identically. ckpt_step_701642_tc19 was grown that way and never trained
    them: its command columns measure EXACTLY 0.000000, so substituting the action
    cannot change any prediction. A test on that checkpoint returns gaps of exactly
    zero no matter what else is true, which is a different statement from "the
    difference was too small to resolve" and must not be reported as the same thing.
    """
    try:
        W = model.encoder_registry.encoders["motor"].patch_mlp[0].weight
    except Exception:
        return None
    if W.shape[1] < MOTOR_INPUT_F:
        return None
    return float(W[:, MOTOR_ACHIEVED_F:MOTOR_INPUT_F].abs().mean())


def _collect(model, src, n_steps, hold, limp=False, limits=None, babble=0.0):
    """Completed transitions: state_t with its own context, the OUTGOING action, state_t+1.

    The action is attached one step late because it does not exist until the model has
    chosen it and the body has run it. Transition i is emitted only once step i+1 reveals
    which order carried the body there. Attaching the arrival action instead is defect 1.
    """
    from mimo_bridge.lem_model import LemLite
    # each stream starts clean: without this the second body inherits the first body's
    # temporal context and the passive body inherits the second's
    model.reset_transition_state()
    frames = src.reset()
    # model.dev_limits() takes a STEP, not a phase index. This function used to receive
    # a phase index 0-3 and pass it straight in, which is step 0-3 of the defaults table:
    # the NEONATE row, rom 0.05, strength 0.02. Every causal test before 2026-08-27 drove
    # the body at 2% strength whatever the checkpoint had trained under.
    dev = dict(limits) if limits else dict(model.dev_limits(10 ** 9))
    if babble:
        # Match the movement regime the checkpoint was TRAINED under. With the planner
        # off and no babble the controller emits ~0.07 rad orders, and the test returns
        # INERT for want of motion rather than for want of a causal signal - which says
        # nothing about the model. H33 moved at 0.42 rad on exploration_noise 0.05.
        # motor_for_action is defined as "after developmental constraints + noise", so
        # the ledgered executed action already carries the babble and the transition
        # contract (state_t + executed_action_t -> state_t+1) is unaffected.
        dev["exploration_noise"] = float(babble)
    if limp:
        dev["strength_limit"] = 0.0          # actuators disabled: gravity and contact only
    held, prev_exec, pending, out = None, None, None, []
    order_seq, prev_oid = 0, None        # item 1: stamped like the trainer stamps
    reps = int((limits or {}).get("_action_repeat") or 5)
    for i in range(n_steps):
        with torch.no_grad():
            # review E12: the shift-and-prepend reconstruction zeroed the OLDEST
            # slot (where S_{t-64} belongs). The buffer as it stands right now IS
            # S_t's context - snapshot it before step() pushes S_t. On the very
            # first step of a fresh model the buffer is None (nothing has been
            # seen): S_0's context is zeros, which is exactly what step() builds.
            ctx = (model.context_buffer.detach().float().cpu().clone()
                   if model.context_buffer is not None
                   else torch.zeros(len(frames), (model.ctx_len - 1) * model.n_mod,
                                    model.d_model))
            res = model.step(frames, dev, train=False, step_index=i,
                             executed_action=prev_exec, executed_order_id=prev_oid)
            toks = {m: v.detach().float().cpu()
                    for m, v in (model.last_tokens or {}).items()}
            ach = model.last_achieved.detach().float().cpu()
            # THE CONTEXT AND MEMORY THE LIVE LESSON USES (repair 2026-09-22). step()
            # back-fills the newest context slot with the order that just arrived
            # (lem_model B9 fix) and the next-frame lesson predicts S_t+1 from the
            # context it cached AFTER that back-fill (_nf_ctx). The pre-step snapshot
            # above still holds that slot's "not yet known" zero command. The lesson
            # also adds the persistent-memory bias of this moment to the forward pass;
            # H84 runs with persistent memory on and its recalled bias measured 1.18x
            # the token it lands on (H84 README), so a prediction without it is not the
            # model's prediction.
            _nf_ctx = getattr(model, "_nf_ctx", None)
            if torch.is_tensor(_nf_ctx) and _nf_ctx.shape == model.context_buffer.shape:
                ctx = _nf_ctx.detach().float().cpu().clone()
            _mem = getattr(model, "_memory_bias_current", None)
            mem = _mem.detach().float().cpu().clone() if torch.is_tensor(_mem) else None
        if pending is not None and prev_exec is not None:
            p_toks, p_ach, p_ctx, p_passive, p_mag, p_str, p_mem = pending
            rec = {"t": p_toks, "ach": p_ach, "ctx": p_ctx, "t1": toks,
                   "act": torch.as_tensor(prev_exec, dtype=torch.float32)}
            if p_mem is not None:
                rec["mem"] = p_mem
            if p_passive is not None:
                # the branch measured on THIS transition's interval, not the previous one
                rec["passive_delta"] = p_passive
            if p_mag is not None:
                # review E13: these belong to the interval this record covers; they
                # used to be written into out[-1] at burst time - the transition
                # closed from the PREVIOUS interval
                rec["cmd_mag"] = p_mag
                rec["cmd_strength"] = p_str
            out.append(rec)
        pending = (toks, ach, ctx, None, None, None, mem)

        if i % hold == 0 or held is None:
            held = np.zeros_like(res.motor_for_action) if limp else res.motor_for_action
            order_seq += 1
            if limp:
                held[..., 6] = 1.0                       # unit quaternion, zero strength
        # MATCHED PASSIVE BRANCH.
        #
        # Five defects in the first attempt, all real:
        #   1. it wrote the passive future into out[-1], which is the PREVIOUS transition -
        #      the passive target was one transition ahead of the record it scored.
        #   2. the passive arm ran five null steps plus a sixth inside model.step(), so it
        #      covered six substeps against the active arm's five.
        #   3. it called model.step() on the branch, which mutates the context buffer, the
        #      next-frame caches and step_count - and only the ENVIRONMENT was restored, so
        #      the real trajectory continued with context from the passive branch.
        #   4. snapshot_state() does not carry the env RNG, the saccade/reflex state or the
        #      sensor histories, so the two arms did not start from an identical state.
        #   5. gz_burst came from the first active run while the environment was then
        #      re-run, so the recorded gaze could belong to a trajectory that was discarded.
        #
        # Fixed by branching the PHYSICS ONLY, at equal horizons, with the model untouched
        # and the RNG pinned, and by attaching the result to THIS transition rather than
        # the previous one. The passive outcome is recorded as a physical delta; scoring it
        # through the model would require a model-state snapshot the model does not offer.
        env = getattr(src, "env", None)
        branch = None
        if env is not None and not limp and hasattr(env, "snapshot_state"):
            import copy as _copy
            snap = env.snapshot_state()
            rng_state = getattr(env, "rng", None)
            rng_snap = rng_state.bit_generator.state if rng_state is not None else None
            # review E10: the branch's steps also advance the saccade generator (its
            # own rng), the reflex, the touch pooler's vibration/slip/contact-age
            # state, proprioceptive fatigue, the jitter ring, the cmd histories and
            # the caregiver timer - none covered by restore_state, so the real arm
            # continued from a different helper state than it entered with.
            _saved = {}
            for _name in ("saccades", "reflex", "pooler", "proprio"):
                _o = getattr(env, _name, None)
                if _o is not None:
                    _saved[_name] = _snap_helper_state(_o)
            # review 2026-09-06: this list predates critters (09-05) and hearing
            # (09-06). Without them the branch's phantom squeaks SOUNDED in the real
            # timeline, hit counters kept branch presses, and the visitor schedule
            # drifted 5 frames per probe - the matched arm was not matched.
            _saved_env = {k: _copy.deepcopy(getattr(env, k)) for k in
                          ("_cmd_hist", "_cmd_delay", "_jit_prev", "_jit_ring",
                           "_unseen_s", "_cr", "_audio", "_hand_prev", "_mobile_prev",
                           "_onball_lock", "_target_home", "_rescues",
                           "_mobile_touch_prev", "_chime_cool") if hasattr(env, k)}
            _saved_geom = (np.array(env.model.geom_rgba), np.array(env.model.geom_size))
            # 2026-09-23 (fix-rows-r1-world_model_battery.py): the lists above predate the
            # virtual skin (contact_profile mechanical_v2_local and later: H84 runs
            # mechanical_v4), which keeps its own clock, the body function and the
            # responsive caregiver. The branch put the physics clock back but not the
            # skin's, so the real arm's first step stopped on "virtual skin time reversed"
            # in every trainer-built H84 world. The env's own paired-branch state
            # (save_full_state: the skin, body function, caregiver, contact trackers,
            # solver state) is now saved too and put back last.
            _full = env.save_full_state() if hasattr(env, "save_full_state") else None
            null = np.zeros_like(held)
            null[..., 6] = 1.0                       # identity rotation, zero strength
            for _ in range(reps):                    # EQUAL horizon, like the active arm
                src.step(null, res.gaze_out, None, render=False)
            after_gravity = np.array(env.data.qpos)
            env.restore_state(snap)
            if rng_snap is not None:
                env.rng.bit_generator.state = rng_snap   # same draws on the real arm
            for _name, _st in _saved.items():
                _restore_helper_state(getattr(env, _name), _st)
            for _k, _v in _saved_env.items():
                setattr(env, _k, _v)
            env.model.geom_rgba[:] = _saved_geom[0]     # squash/flicker/hide mutations
            env.model.geom_size[:] = _saved_geom[1]
            if _full is not None:
                env.restore_full_state(_full)
            branch = after_gravity

        gz_burst = []
        for rep in range(reps):                      # review E9: the run's own repeat
            frames = src.step(held, res.gaze_out, None, render=(rep == reps - 1))
            gz_burst.append(np.stack([np.asarray(f.meta.get("gaze_cmd") or (0.0, 0.0),
                                                 dtype=np.float32) for f in frames]))
        if branch is not None:
            after_action = np.array(env.data.qpos)
            # attached to the transition being CLOSED next, not to the previous one
            pending_passive = float(np.abs(after_action - branch).max())
        else:
            pending_passive = None
        _cr = [(getattr(f, "meta", None) or {}).get("cmd_response") or {} for f in frames]
        _cm = [c.get("cmd_mag") for c in _cr if c.get("cmd_mag") is not None]
        pending = (pending[0], pending[1], pending[2], pending_passive,
                   (float(np.mean(_cm)) if _cm else 0.0),
                   float(np.asarray(held)[..., 8].mean()), pending[6])
        # The size of the order actually sent, taken from the env's own cmd_response so
        # it is the SAME quantity the dashboard and vitals band report: the geodesic
        # angle from the body's current orientation to the commanded one. My first
        # version measured the target quaternion's angle from identity and read 2.31 rad
        # against the env's 0.05 - an absolute pose, not the size of the ask.
        # gaze is not held: the eyes can move on every substep, so the interval's action is
        # the mean angular position the eyes occupied, not wherever they happened to stop
        prev_exec = LemLite.gaze_to_handle(held, np.mean(np.stack(gz_burst), axis=0))
        prev_oid = order_seq
    return out


def _predict(model, tr, action, device):
    """One canonical transition, taking the LIVE route.

    The motor token is built through the encoder registry - so it gets the same output
    LayerNorm real tokens get - and the FULL forward model runs on this transition's own
    context before the prediction heads. predict_from_action() is deliberately not used:
    it skips both, which is the representation mismatch this test must not measure.

    The persistent-memory bias of the transition's own moment (tr["mem"], recorded by
    _collect) is added exactly as the live next-frame lesson adds it (repair 2026-09-22:
    it was omitted, so on a persistent-memory run every arm was scored on a forward pass
    the model never makes). A record without "mem" (a run without persistent memory, or
    a caller's own record) is predicted without a bias, as before.
    """
    from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, command_for_encoder
    d = device
    frame = {"motor": torch.cat([tr["ach"][..., :MOTOR_ACHIEVED_F].to(d),
                                 command_for_encoder(action.to(d))], dim=-1)}
    mem = tr.get("mem")
    memory_kw = {} if mem is None else {"memory_bias": mem.to(d)}
    with torch.no_grad():
        enc = model.encoder_registry.encode(frame, {"motor": "real"})
        tokens = {}
        for m in model.active:
            if m == "motor" and "motor" in enc:
                tokens[m] = enc["motor"].reshape(-1, 1, model.d_model)
            elif m in tr["t"]:
                tokens[m] = tr["t"][m].to(d).unsqueeze(1)
        _l, last_step, _c = model.forward_model(tokens, tr["ctx"].to(d), **memory_kw)
        return model.prediction_heads(
            last_step,
            current={m: v.reshape(v.shape[0], -1) for m, v in tokens.items()
                     if m in model.prediction_heads.head_modalities},
            who="causal_test._predict")


def run(ckpt, n_steps, seeds, scene, run_dir, device, intention=False, babble=0.0):
    from mimo_bridge.datasource import SingleEnvSource

    rows, meta = [], {}
    for seed in range(seeds):
        model, a, src_path, found = _build_model(ckpt, device, seed, intention=intention)
        hold = int(a.get("motor_hold") or 5)
        meta = {"args_json": src_path, "args_found": found, "motor_hold": hold,
                "context_length": model.ctx_len, "intention": bool(intention),
                "motor_in": int(model.encoder_registry.encoders["motor"].patch_mlp[0].in_features),
                "babble": float(babble)}

        limits, limits_src = _limits_for_run(ckpt, model)
        meta["limits"] = {k: limits.get(k) for k in ("phase_name", "rom_limit", "strength_limit")}
        meta["limits_source"] = limits_src

        # review E9: the env used to be rebuilt with defaults - base-age body vs
        # the trained 24-month one, decoder kp 8 vs the run's 2 (4x stiffer servo),
        # caregiver off, cmd horizon 5 vs 25. Taken from args.json like the model,
        # since 2026-09-23 through the trainer's own world builder (world_kwargs): the
        # hand-copied keys left out the body layout, so no trunk_v1 run could be tested.
        _reps = int(a.get("action_repeat") or 5)
        limits["_action_repeat"] = _reps
        env_kw = world_kwargs(a, limits, scene, run_dir, _reps * hold)
        meta["env_kw"] = {k: v for k, v in env_kw.items() if k != "run_dir"}

        def stream(tag, sd, n, limp=False):
            s = SingleEnvSource(tag=f"_{tag}{seed}", seed=sd, **env_kw)
            try:
                return _collect(model, s, n, hold, limp=limp, limits=limits,
                                babble=(0.0 if limp else babble))
            finally:
                s.close()

        A = stream("act", 100 + seed, n_steps)                        # this body
        B = stream("oth", 300 + seed, n_steps)                        # a different body
        # review E11: a limp arm was collected here (a third of an arm of env+model
        # stepping) and used only in a length check - the matched physics branch
        # replaced its role long ago.

        if len(A) < 8 or not B:
            raise SystemExit(f"collected {len(A)} active, {len(B)} other")

        synth = {m for m in model.active
                 if str(model.config["modalities"].get(m, {}).get("data_source", "real")) != "real"}
        mods = [m for m in model.pred_mods if m in A[0]["t"] and m not in synth]
        if not mods:
            raise SystemExit("no real predicted modalities to score")
        # the scale of a real order in this stream, per column, for the perturbed control
        _acts = torch.stack([tr["act"].reshape(-1, tr["act"].shape[-1]) for tr in A])
        _sd = _acts.reshape(-1, _acts.shape[-1]).std(dim=0) * PERTURB_FRACTION
        per = {c: {m: [] for m in mods} for c in CONTROLS}
        skipped = {"shuffled": 0, "cross-body": 0}
        g = torch.Generator().manual_seed(seed)

        for tr in A:
            a_true = tr["act"]

            def pick(pool, tag):
                for _ in range(32):
                    j = int(torch.randint(len(pool), (1,), generator=g))
                    cand = pool[j]["act"]
                    # review E14: within a hold window the limb order repeats, so
                    # the all-columns diff could be satisfied by the mean-gaze row
                    # alone - a "shuffled" control differing only in gaze
                    if body_rows_differ(cand, a_true, model.GAZE_IDX):
                        return cand
                skipped[tag] += 1
                return None

            def _perturb(a9):
                """The same order, nudged - still a plausible order FOR THIS STATE."""
                v = a9 + torch.randn(a9.shape, generator=g) * _sd
                q = v[..., 3:7]
                v = v.clone()
                v[..., 3:7] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                return v

            for c, av in (("correct", a_true),
                          ("state-only", _null_action(a_true)),
                          ("shuffled", pick(A, "shuffled")),
                          ("cross-body", pick(B, "cross-body")),
                          ("perturbed", _perturb(a_true))):
                if av is None:
                    continue
                preds = _predict(model, tr, av, device)
                for m in mods:
                    if m in preds:
                        per[c][m].append(float(F.mse_loss(preds[m].float().cpu(),
                                                          tr["t1"][m].float())))
            for m in mods:                                  # no model at all
                per["persistence"][m].append(float(F.mse_loss(tr["t"][m].float(),
                                                              tr["t1"][m].float())))

        # MATCHED PASSIVE. The branch is a PHYSICAL measurement: from the same state, how
        # far apart do acting and not acting leave the body? Scoring it through the model
        # would need a model-state snapshot the model does not offer, and faking one by
        # calling model.step() on the branch corrupts the real trajectory's context - which
        # is what the first attempt did.
        #
        # A near-zero delta means the action changed nothing physical, and then NO
        # prediction test can show action-conditioning: the control reports that directly
        # instead of dressing it up as an MSE.
        deltas = [tr["passive_delta"] for tr in A if "passive_delta" in tr]
        n_branch = len(deltas)
        # (nothing to score here: the branch is reported as a physical delta)

        cmags = [tr["cmd_mag"] for tr in A if "cmd_mag" in tr]
        subs = []
        for tr in A:
            for pool in (A, B):
                j = int(torch.randint(len(pool), (1,), generator=g))
                subs.append(float((pool[j]["act"] - tr["act"]).abs().mean()))
        cmd_w = _command_pathway_weight(model)
        rows.append({"seed": seed, "n_active": len(A), "n_passive": n_branch,
                     "command_pathway_weight": cmd_w,
                     "cmd_mag_rad": (float(np.mean(cmags)) if cmags else None),
                     "substitute_action_distance": (float(np.mean(subs)) if subs else None),
                     "passive_branch_qpos_delta": (float(np.mean(deltas)) if deltas else None),
                     "skipped_identical": dict(skipped),
                     "mse": {c: {m: float(np.mean(v)) if v else float("nan")
                                 for m, v in d.items()} for c, d in per.items()}})
        del model
        torch.cuda.empty_cache()

    mods = sorted({m for r in rows for m in r["mse"]["correct"]})
    summary, verdict = {}, True
    for m in mods:
        summary[m] = {}
        for c in CONTROLS:
            vals = [r["mse"][c].get(m, float("nan")) for r in rows]
            vals = [v for v in vals if v == v]
            summary[m][c] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                             "std": float(np.std(vals)) if vals else float("nan"),
                             "n_seeds": len(vals)}
        # PER SEED, not on the aggregate mean. The docstring has always said "beating on
        # the mean is not beating"; the implementation compared means anyway, so one lucky
        # seed could carry a losing result.
        cm = summary[m]["correct"]["mean"]
        per_seed_wins = {}
        for c in CONTROLS:
            if c == "correct":
                continue
            wins = [r["mse"]["correct"].get(m, float("nan")) < r["mse"][c].get(m, float("nan"))
                    for r in rows
                    if r["mse"]["correct"].get(m) == r["mse"]["correct"].get(m)
                    and r["mse"][c].get(m) == r["mse"][c].get(m)]
            per_seed_wins[c] = {"wins": int(sum(wins)), "of": len(wins),
                                "every_seed": bool(wins) and all(wins)}
        summary[m]["per_seed"] = per_seed_wins
        summary[m]["beats"] = {c: v["every_seed"] for c, v in per_seed_wins.items()}
        summary[m]["beats_every_control"] = all(summary[m]["beats"].values())
        summary[m]["mean_only_would_have_said"] = {
            c: bool(cm < summary[m][c]["mean"]) for c in CONTROLS if c != "correct"}
        summary[m]["action_has_no_influence"] = all(
            summary[m][c]["mean"] == cm for c in ("state-only", "shuffled", "cross-body"))
        # RESOLUTION. A verdict is only meaningful if substituting the action moves the
        # error by more than the run-to-run noise of the correct arm itself. On H28 the
        # correct arm read 0.07202 +/- 0.00057 and the shuffled arm 0.07199 - a gap of
        # 0.000027, twenty times SMALLER than the seed spread. Calling that FAIL claims
        # a measurement the data cannot support; it is a null result, not a negative
        # one. The bar is the measured spread, not a number chosen here.
        noise = summary[m]["correct"]["std"]
        gaps = {c: abs(summary[m][c]["mean"] - cm)
                for c in ("state-only", "shuffled", "cross-body")}
        summary[m]["substitution_gaps"] = gaps
        summary[m]["seed_noise"] = float(noise)
        # A std from two seeds is not a noise floor - the 2-seed rerun produced
        # std 0.0000 for proprioception, which makes any gap look resolvable. Three is
        # the minimum for the spread to mean anything.
        n_ok = summary[m]["correct"]["n_seeds"]
        summary[m]["resolvable"] = bool(n_ok >= 3 and noise == noise
                                        and any(gv > noise for gv in gaps.values()))
        verdict = verdict and summary[m]["beats_every_control"]

    # Three outcomes, not two. INCONCLUSIVE means the transitions did not differ
    # enough for any verdict: either the action never diverged the body physically,
    # or swapping it moved the error less than the seed noise.
    resolvable = any(d.get("resolvable") for d in summary.values())
    pdl = [r.get("passive_branch_qpos_delta") for r in rows
           if r.get("passive_branch_qpos_delta") is not None]
    cmg = [r.get("cmd_mag_rad") for r in rows if r.get("cmd_mag_rad") is not None]
    movement = {"passive_branch_qpos_delta": float(np.mean(pdl)) if pdl else None,
                "cmd_mag_rad": float(np.mean(cmg)) if cmg else None,
                "substitute_action_distance":
                    float(np.mean([r["substitute_action_distance"] for r in rows
                                   if r.get("substitute_action_distance") is not None]))
                    if any(r.get("substitute_action_distance") is not None for r in rows) else None}
    cw = [r.get("command_pathway_weight") for r in rows
          if r.get("command_pathway_weight") is not None]
    inert = bool(cw) and max(cw) == 0.0
    movement["command_pathway_weight"] = float(np.mean(cw)) if cw else None
    movement["command_pathway_inert"] = inert
    outcome = "PASS" if verdict else ("FAIL" if resolvable else
                                      ("INERT" if inert else "INCONCLUSIVE"))
    return {"type": "causal_test", "version": 3, "timestamp": time.time(),
            "outcome": outcome, "resolvable": bool(resolvable), "movement": movement,
            "checkpoint": os.path.abspath(ckpt), "n_steps": n_steps, "seeds": seeds,
            "scene": scene, "config": meta, "pass": bool(verdict),
            "per_seed": rows, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--babble-noise", type=float, default=0.0,
                    help="exploration noise during collection, as train_mimo --babble-noise. "
                         "Set it to the value the checkpoint trained under: with the planner "
                         "off and this at 0 the body barely moves and the result is INERT for "
                         "lack of motion, not for lack of a causal signal. Never applied to "
                         "the passive control, whose actuators are off by definition.")
    ap.add_argument("--intention", action="store_true",
                    help="collect with the planner ON. Off by default (the planner is "
                         "not under test), but with it off the controller emits ~0.07 rad "
                         "orders and transitions may not differ enough to resolve anything.")
    a = ap.parse_args(argv)
    rd = a.run_dir or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "causal_scenes")
    rep = run(a.ckpt, a.n, a.seeds, a.scene, rd, a.device, intention=a.intention,
              babble=a.babble_noise)

    cfg = rep["config"]
    if not cfg.get("args_found"):
        print(f"\n  WARNING: no args.json at {cfg['args_json']}; model built from defaults")
    print(f"\n  context {cfg['context_length']}, motor encoder {cfg['motor_in']}-wide, "
          f"motor_hold {cfg['motor_hold']}, {a.seeds} seed(s), "
          f"{rep['per_seed'][0]['n_active']} transitions each")
    mv = rep["movement"]
    print("  planner during collection: " + ("ON" if cfg.get("intention") else "off"))
    print("")
    print("  WAS THERE ANYTHING TO DETECT?")
    if mv.get("command_pathway_weight") is not None:
        print("    motor encoder command-column |w|      %.6f%s"
              % (mv["command_pathway_weight"],
                 "   INERT: the action cannot reach the prediction"
                 if mv.get("command_pathway_inert") else ""))
    if mv.get("cmd_mag_rad") is not None:
        print("    commanded rotation per handle      %.4f rad" % mv["cmd_mag_rad"])
    if mv.get("substitute_action_distance") is not None:
        print("    distance to the substitute action  %.4f" % mv["substitute_action_distance"])
    if mv.get("passive_branch_qpos_delta") is not None:
        print("    acting vs not acting, same state   %.5f qpos apart" % mv["passive_branch_qpos_delta"])
    print("")
    print("  lower is better")
    head = f"  {'modality':16s}" + "".join(f"{c:>13s}" for c in CONTROLS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for m, d in rep["summary"].items():
        print(f"  {m:16s}" + "".join(f"{d[c]['mean']:13.5f}" for c in CONTROLS))
        print(f"  {'':16s}" + "".join(f"{'+/-' + format(d[c]['std'], '.4f'):>13s}"
                                      for c in CONTROLS))
        bad = [c for c, ok in d["beats"].items() if not ok]
        ps = d.get("per_seed", {})
        print(f"  {'':16s}  beats every control on EVERY seed: {d['beats_every_control']}"
              + ("" if not bad else f"   loses to: {bad}"))
        if ps:
            print(f"  {'':16s}  per-seed wins: "
                  + ", ".join(f"{c} {v['wins']}/{v['of']}" for c, v in ps.items()))
        if d.get("mean_only_would_have_said") != d["beats"]:
            print(f"  {'':16s}  (comparing means alone would have given a DIFFERENT "
                  f"verdict: {d['mean_only_would_have_said']})")
    sk = rep["per_seed"][0]["skipped_identical"]
    if any(sk.values()):
        print(f"\n  substitutes skipped as identical to the real action: {sk}")
    zero = [m for m, d in rep["summary"].items() if d.get("action_has_no_influence")]
    if zero:
        print("")
        print("  Substituting the action changes NOTHING for " + str(zero) + ":")
        print("  the command pathway carries zero weight, which is where the")
        print("  migration deliberately starts it.")
    print("")
    print("  CAUSAL TEST " + rep["outcome"])
    if rep["outcome"] == "INERT":
        print("  The motor encoder's command columns are EXACTLY zero, so no action can")
        print("  change any prediction. This is not a result about what the model learned -")
        print("  the pathway was zero-padded by checkpoint growth and never trained. Gaps of")
        print("  zero were guaranteed before the test ran.")
    if rep["outcome"] == "INCONCLUSIVE":
        print("  NOT a negative result. Swapping the action moved the error by LESS than the")
        print("  seed-to-seed noise of the correct arm, so this data cannot tell the correct")
        print("  action from a wrong one in either direction. Per modality:")
        for m, d in rep["summary"].items():
            g = d.get("substitution_gaps") or {}
            if g:
                print("    %-16s largest gap %.6f   seed noise %.6f"
                      % (m, max(g.values()), d["seed_noise"]))
        print("  Restore action-dependent movement, then re-run. Item 7 stays off.")
    elif rep["outcome"] == "FAIL":
        print("  The action is not yet carrying prediction. Item 7 stays off.")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "causal_test.json")
    json.dump(rep, open(out, "w"), indent=2)
    print(f"  -> {out}")
    return 0 if rep["pass"] else (2 if rep["outcome"] == "INCONCLUSIVE" else 1)


if __name__ == "__main__":
    sys.exit(main())
