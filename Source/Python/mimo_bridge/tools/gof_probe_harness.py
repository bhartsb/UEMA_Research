# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Shared harness for the gain-of-function probes (addendum v2 items 7-10, DESIGN.md D15 and
section 9).

A probe freezes one checkpoint, builds its model on the CPU the way the trainer builds it, and
takes ONE body out of the checkpoint's own episode companion - its world (caregiver, mobile and
critters as saved, nothing recorded), context, persistent memory and histories at the saved
moment (see "THE EPISODE" below for why) - with the gaze reflex and spontaneous saccades
switchable. It then runs ARMS from identical restored states: the same world state and the same
model state, differing only in the order sent to the body. What the model predicted before an
order is compared with what the body then did. A checkpoint without a companion gets a fresh
one-body world warmed under the model's own behaviour, and its results say so.

Nothing here trains: every model step is train=False and no optimizer exists (the checkpoint is
loaded without its optimizer). Nothing here writes a checkpoint. Results go to
Saved/mimo_bridge/gof_probes/<run>/<checkpoint>/<probe>.json with one line per probe appended to
Saved/mimo_bridge/gof_probes/index.jsonl - outside the experiments/ folders the checkpoint
archiver moves.

THE MOMENT. After the model has stepped on frame t (observe()), the state a prediction starts
from is exactly the one the live next-frame lesson starts from one step later: the raw tokens of
t, the context the step cached after back-filling the order that just arrived (_nf_ctx), the
achieved pose of t and the persistent-memory bias of t. predict() re-encodes only the motor token
with the candidate order in its command half, runs the full forward model and the prediction
heads - the causal test's route, repaired 2026-09-22 to add the memory bias - and also returns
the forward model's last-step output, which is what the tenure pose head and the efference heads
read.

OPTIONAL COMPONENTS (built by other agents at the same time): the efference heads
(model.efference), a contact head, the vestibular sense (a modality named "vestibular") and a
motor synergy. components() reports which exist; a probe that needs an absent one reports
"unmeasured: component absent" instead of failing.
"""
from __future__ import annotations

import os
if __name__ == "__main__":
    # RUN AS A PROGRAM, A PROBE WORKS ON THE CPU: the card is hidden here, before torch starts, as it always was.
    # Importing the module never changes the process's devices (2026-09-26): the hide used to run at import, so a
    # test process that collected any probe test lost the card and skipped the card continuation test. A program
    # that imports a probe and wants the CPU hides the card at its own entry point (record:
    # docs/gain_of_function_v4/agents/probe_gpu_hiding_2026-09-26.md).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import collections
import contextlib
import copy
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# v2 (2026-09-23, the review of the probes, docs/gain_of_function_v3/reviews/probes.md): the eye
# row of every scripted order is the command act() sends; the binding headline is the
# counterfactual against holding still; goal lines are null until a working-control reading
# exists; index lines carry this version and the code commit.
HARNESS_VERSION = "gof_probe_harness_v4"
REPO = Path(__file__).resolve().parents[4]
SAVED = REPO / "Saved" / "mimo_bridge"
PROBES_ROOT = SAVED / "gof_probes"
INDEX = PROBES_ROOT / "index.jsonl"

_CODE = {}


def pinned_commit() -> Optional[str]:
    """The commit of a pinned job's export, or None. docs/gain_of_function_v4/pinned_job.py runs a pause job on a
    `git archive` export of one commit, with no .git inside, after checking every exported file against that
    commit's blobs; it names the commit in PINNED_COMMIT and the export in PINNED_ROOT. Only a process whose REPO is
    that export takes the commit from there."""
    commit, root = os.environ.get("PINNED_COMMIT"), os.environ.get("PINNED_ROOT")
    if not commit or not root:
        return None
    try:
        same = os.path.normcase(os.path.realpath(root)) == os.path.normcase(os.path.realpath(str(REPO)))
    except (OSError, ValueError):
        return None
    return commit if same else None


def code_commit() -> dict:
    """The commit the probe code ran on, and how many tracked files under Source/Python differed
    from it (the build is edited by several agents at once, so a clean tree is not assumed). In a
    pinned job's export, where git reads nothing, the export's checked commit (pinned_commit) with no
    file differing: without it such a run recorded no commit, so a failure there never counted as this
    code's and a failing probe was run again at every pass (found 2026-09-26, when the proprioception
    branch's pinned unit tests ran test_suite_records_a_failing_probe_and_runs_the_rest there)."""
    if not _CODE:
        def git(*a):
            try:
                return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                                      timeout=30).stdout
            except Exception:
                return ""
        head = git("rev-parse", "HEAD").strip() or None
        if head is None and pinned_commit():
            _CODE.update(commit=pinned_commit(), uncommitted_tracked_files=0, probe_files_uncommitted=[],
                         pinned_export=True)
            return dict(_CODE)
        dirty = [ln for ln in git("status", "--porcelain", "--untracked-files=no", "--",
                                  "Source/Python").splitlines() if ln.strip()]
        _CODE.update(commit=head, uncommitted_tracked_files=len(dirty),
                     probe_files_uncommitted=sorted(ln[3:] for ln in dirty
                                                    if "gof_probe_" in ln or "copied_execution" in ln))
    return dict(_CODE)


def refresh_code_commit() -> dict:
    """code_commit() read again from git. A long-running process (the probe watcher) calls this
    before each pass: the commit is cached at a process's first call, and a watcher that kept the
    commit it started on would judge a failure written by a suite child on a newer commit as "other
    code" for ever, spawning a child that skips every probe on every pass (verify-fix-probes
    2026-09-23)."""
    _CODE.clear()
    return code_commit()

# Commit charge one probe process may add, for tools/commit_guard. The v4 full saved-upgrade
# suite measured 8.24 GB peak private commit on 2026-09-23, although its peak resident working
# set was only 4.88 GB. A working-set measurement cannot size a commit guard. Reserve 10 GiB
# (the guard's 1024-based units); suite.json records peak_commit_gb in decimal GB on Windows.
NEED_GB = 10.0


class RefusedCheckpoint(ValueError):
    """The checkpoint cannot be probed as trained (no run configuration, or a component the
    load would have reinitialised)."""


# ============================================================================ checkpoints
def _locate(path) -> Path:
    """The checkpoint where it is now: unchanged when it exists, else where a storage move
    recorded it (mimo_bridge.storage_locations, when that module exists)."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    try:
        from mimo_bridge import storage_locations
        found = storage_locations.locate(p, verify=False)
        return Path(found)
    except Exception:
        return p


def resolve_checkpoint(ckpt) -> dict:
    """Where the checkpoint is, which run wrote it, its step and its run's args.json."""
    from mimo_bridge.tools.causal_test import _run_args_path
    path = _locate(ckpt)
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt} (looked at {path})")
    lineage = {}
    if (path / "lineage.json").exists():
        try:
            lineage = json.loads((path / "lineage.json").read_text(encoding="utf-8"))
        except ValueError:
            lineage = {}
    m = re.search(r"ckpt_step_(\d+)", path.name)
    step = lineage.get("step", int(m.group(1)) if m else None)
    args_path, found = _run_args_path(str(path))
    run_dir = lineage.get("out") or str(path.parent.parent)
    run = lineage.get("run") or Path(run_dir).name
    return dict(checkpoint=str(path), name=path.name, run=run, run_dir=run_dir,
                args_path=args_path, args_found=bool(found), lineage_step=step,
                lineage=lineage, resumed_from=lineage.get("resumed_from"))


def output_dir(info: dict) -> Path:
    return PROBES_ROOT / info["run"] / info["name"]


# ============================================================================ frozen model
@dataclass
class Frozen:
    model: Any
    args: SimpleNamespace
    info: dict
    limits: dict
    limits_source: str
    load: dict
    build_s: float
    load_s: float


def load_frozen(ckpt, *, seed: int = 0, intention: Optional[bool] = None,
                arg_overrides: Optional[dict] = None, threads: int = 4) -> Frozen:
    """Build the checkpoint's model on the CPU through the trainer's builder and load it.

    intention None = the run's own setting (so the warm-up behaves as the run did).
    Refuses a checkpoint whose run configuration cannot be found (a model built from defaults
    is not the trained model) and one whose load reinitialised any component."""
    from mimo_bridge.tools import causal_test as CT
    torch.set_num_threads(int(threads))
    info = resolve_checkpoint(ckpt)
    if not info["args_found"]:
        raise RefusedCheckpoint(f"no args.json for {info['checkpoint']} (looked at {info['args_path']}, "
                                "lineage.json 'out', storage_locations): a model built from defaults "
                                "is not the trained model")
    saved, _, _ = CT._cfg_from_run(info["checkpoint"])
    if intention is None:
        intention = str((arg_overrides or {}).get("intention", saved.get("intention", "off"))) == "on"
    t0 = time.time()
    model, a, _src, _found = CT._build_model(info["checkpoint"], "cpu", seed, intention=intention,
                                             arg_overrides=arg_overrides, load=False)
    build_s = time.time() - t0
    t0 = time.time()
    loaded = model.load_checkpoint(info["checkpoint"], load_optimizer=False)
    load_s = time.time() - t0
    reinit = list(loaded.get("reinitialized_components") or [])
    if reinit:
        raise RefusedCheckpoint(f"{info['checkpoint']}: the load reinitialised {reinit}; a probe of "
                                "a partly fresh model says nothing about the checkpoint")
    limits, limits_source = CT._limits_for_run(info["checkpoint"], model)
    if not limits.get("phase_name"):
        raise RefusedCheckpoint("no developmental limits for this checkpoint")
    return Frozen(model=model, args=SimpleNamespace(**a), info=info, limits=dict(limits),
                  limits_source=limits_source, load=dict(step=loaded.get("step"), reinitialized=reinit),
                  build_s=build_s, load_s=load_s)


def _module_attr(model, names):
    """The first of `names` that holds a torch module on the model (a flag of the same name, such
    as LemLite.motor_synergy = False, is not the module)."""
    for name in names:
        if isinstance(getattr(model, name, None), torch.nn.Module):
            return name
    return None


def components(model) -> Dict[str, dict]:
    """Which optional model parts exist in this build. Probes read this instead of assuming.
    Each part counts only when the attribute holds the MODULE: LemLite.motor_synergy is the flag
    (a bool) and LemLite.synergy the network (review of the probes, 2026-09-23: a flags-off build
    was reported as carrying a synergy)."""
    eff = getattr(model, "efference", None)
    contact = _module_attr(model, ("contact_head", "contact_heads", "self_contact_head"))
    synergy = _module_attr(model, ("synergy", "motor_synergy"))
    tenure = getattr(model, "tenure_heads", None)
    return {
        "efference": dict(present=eff is not None, attr="efference" if eff is not None else None),
        "contact_head": dict(present=contact is not None, attr=contact),
        "vestibular": dict(present="vestibular" in list(getattr(model, "active", []) or []),
                           attr="active['vestibular']"),
        "synergy": dict(present=synergy is not None, attr=synergy),
        "tenure_pose_head": dict(present=tenure is not None and "pose" in tenure,
                                 attr="tenure_heads['pose']" if tenure is not None else None),
        "persistent_memory": dict(present=getattr(model, "persistent_memory", None) is not None,
                                  attr="persistent_memory"),
    }


def absent(component: str) -> dict:
    return dict(status="unmeasured: component absent", component=component)


# ============================================================================ the episode
# WHY PROBES START FROM THE CHECKPOINT'S OWN EPISODE (measured 2026-09-22 on ckpt_step_344400,
# CPU, learning off). With a fresh one-body world the model's persistent memory (a 64-second
# window, 128 remembered events) and its 128-moment context start empty; after 300 model steps
# its next-moment vision error still read 0.21-0.24 (squared error per token dimension) against
# 0.0081 in segment 31's first 100 logged steps, and its tenure pose head's error 0.0004-0.0006
# against 0.00019 live. Restoring the checkpoint's own 8-body episode companion instead gave
# vision 0.0057-0.0122 and tenure 0.00007-0.00011 on the first 12 steps - the live values. A
# prediction probed off that distribution says little about the trained model, so the harness
# slices ONE body out of the saved episode (world, context, memory, histories) and falls back to
# a fresh world only when the checkpoint has no companion, labelling the result as such.
_EPISODES: Dict[str, Any] = {}


def _is_tape(value) -> bool:
    from mimo_bridge.event_tape import EventTape
    return isinstance(value, EventTape)


def sanitize_episode(value) -> int:
    """Replace every caregiver recording tape in a loaded episode with a plain list, in place.

    The saved caregiver state holds its EventTape, whose root is the LIVE run's recording folder.
    Deep-copying one flushes it, and Caregiver.restore closes the current one, which writes a
    manifest and a status file into that folder (mimo_bridge/event_tape.py). A probe must never
    touch the live recordings, so no tape survives in the probe's private copy. Returns how
    many were replaced."""
    count = 0
    if isinstance(value, dict):
        for k in list(value.keys()):
            if _is_tape(value[k]):
                value[k] = []
                count += 1
            else:
                count += sanitize_episode(value[k])
    elif isinstance(value, list):
        for i, v in enumerate(value):
            if _is_tape(v):
                value[i] = []
                count += 1
            else:
                count += sanitize_episode(v)
    elif isinstance(value, tuple):
        for v in value:
            count += sanitize_episode(v)
    return count


def load_episode(frozen: "Frozen"):
    """(episode, note): the checkpoint's episode companion, sanitised, cached per process; or
    (None, reason) when the checkpoint has none or it cannot be read."""
    key = frozen.info["checkpoint"]
    if key in _EPISODES:
        return _EPISODES[key]
    from mimo_bridge import continuation_state
    try:
        ep = continuation_state.load(key)
    except Exception as error:
        result = (None, f"episode companion unreadable: {type(error).__name__}: {error}"[:300])
    else:
        if ep is None:
            result = (None, "the checkpoint has no episode companion")
        else:
            tapes = sanitize_episode(ep)
            result = (ep, f"episode companion restored ({len(ep['worlds'])} bodies saved; "
                          f"{tapes} caregiver recording tape(s) replaced by a plain list)")
    _EPISODES.clear()                      # one companion in memory at a time (about 1-3 GB)
    _EPISODES[key] = result
    return result


def slice_batch(value, n: int, b: int, key: str = ""):
    """One body (index b) out of an n-body learner or trainer state.

    Batch-first tensors and arrays ([n, ...]) keep [b:b+1]; the persistent memory's recurrent
    carry ([layers, n, hidden], key 'carry') keeps [:, b:b+1]; a tuple of n per-body tuples
    (the self history's site names) keeps its b-th; containers are walked; 'n_envs' becomes 1;
    everything else is kept as it is."""
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        # These histories are [channel, body], unlike the usual batch-first tensors.
        # Check their named axis first, including the ambiguous channel-count == body-count case.
        if key in ("_goal_ring_ch", "last_raw") and value.ndim == 2 and value.shape[1] == n:
            return value[:, b:b + 1].clone() if torch.is_tensor(value) else value[:, b:b + 1].copy()
        if value.ndim >= 1 and value.shape[0] == n:
            return value[b:b + 1].clone() if torch.is_tensor(value) else value[b:b + 1].copy()
        if key == "carry" and value.ndim >= 2 and value.shape[1] == n:
            return value[:, b:b + 1].clone() if torch.is_tensor(value) else value[:, b:b + 1].copy()
        return value
    if isinstance(value, dict):
        return {k: (1 if k == "n_envs" and value[k] == n else slice_batch(v, n, b, str(k)))
                for k, v in value.items()}
    from collections import deque
    if isinstance(value, deque):
        return deque((slice_batch(v, n, b, key) for v in value), maxlen=value.maxlen)
    if isinstance(value, tuple):
        if len(value) == n and value and all(isinstance(v, tuple) for v in value):
            return (value[b],)
        return tuple(slice_batch(v, n, b, key) for v in value)
    if isinstance(value, list):
        if key in ("_contact_habit", "_gl_goal_index", "_gl_goal_age") and len(value) == n:
            return [copy.deepcopy(value[b])]
        return [slice_batch(v, n, b, key) for v in value]
    return value


def _fresh_part(model, name: str):
    """The module of one component a load reported reinitialised (lem/training/checkpoint.py
    names): 'encoder:<sense>', a side module (LemLite._side_modules), or a model attribute."""
    name = str(name)
    if name.startswith("encoder:"):
        encoders = getattr(getattr(model, "encoder_registry", None), "encoders", None) or {}
        sense = name.split(":", 1)[1]
        module = encoders[sense] if sense in encoders else None
    else:
        side = model._side_modules() if hasattr(model, "_side_modules") else {}
        module = side[name] if name in side else getattr(model, name, None)
    if not isinstance(module, torch.nn.Module):
        raise RefusedCheckpoint("probe-body reload: the first load's fresh part %r is no module of this "
                                "model, so a reload cannot be shown to leave it unchanged" % name)
    return module


def _tensors_of(module) -> Dict[str, torch.Tensor]:
    return {**{"parameter " + k: v.detach().clone() for k, v in module.named_parameters()},
            **{"buffer " + k: v.detach().clone() for k, v in module.named_buffers()}}


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal shape, type and values, a NaN equal to a NaN in the same place (torch.equal calls
    two NaNs different, which would read an untouched running statistic as re-created)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if torch.equal(a, b):
        return True
    return bool(a.is_floating_point() and ((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


# FRESH PARTS AT A PER-BODY RELOAD (2026-09-26, docs/gain_of_function_v4/agents/
# probe_reload_contact_head_2026-09-26.md). A part the checkpoint does not hold - a head a flag
# builds, such as the contact head on checkpoint 344,400 - is reported reinitialised by every load,
# the first one included. The loader leaves such a part as it is (lem/training/checkpoint.py lists a
# side module the checkpoint lacks and touches none of its tensors), so a reload re-creates nothing.
# A caller that built its own Frozen declares the first load's fresh parts in
# Frozen.load["reinitialized"]; the reload accepts exactly those, and only while it leaves every
# tensor of each of them unchanged and the module the same object. load_frozen refuses any
# reinitialisation, so every probe it builds declares none and keeps the rule that a reload which
# reinitialises anything is refused.
def restore_runtime_body(frozen: "Frozen", n: int, b: int):
    """Restart each body's probe from the checkpoint's runtime, not the preceding probe body.

    Runtime is separate from the episode companion. In particular, schemas, entity averages,
    wellness normalization and progress trackers otherwise leaked across probe starts.

    A reload that reinitialises a part the first load did not (Frozen.load["reinitialized"]) is
    refused: a trained part would come back fresh. A declared fresh part has nothing saved to
    restore; the reload is refused if it changes any of that part's tensors, so each body is read by
    the same part (see FRESH PARTS AT A PER-BODY RELOAD).
    """
    model = frozen.model
    fresh = sorted({str(x) for x in ((frozen.load or {}).get("reinitialized") or [])})
    kept = {name: _fresh_part(model, name) for name in fresh}
    before = {name: _tensors_of(module) for name, module in kept.items()}
    model._entity_self = None  # its saved batch state is applied when the one-body tracker is built
    loaded = model.load_checkpoint(frozen.info['checkpoint'], load_optimizer=False)
    reinit = [str(x) for x in (loaded.get('reinitialized_components') or [])]
    beyond = [x for x in reinit if x not in fresh]
    if beyond:
        raise RefusedCheckpoint('probe-body reload reinitialized %s%s' % (
            beyond, '' if not fresh else ' beyond the first load\'s fresh parts %s' % fresh))
    for name, module in kept.items():
        now = _fresh_part(model, name)
        after = _tensors_of(now)
        changed = sorted(k for k in set(before[name]) | set(after)
                         if k not in before[name] or k not in after
                         or not _same_tensor(before[name][k], after[k]))
        if now is not module or changed:
            raise RefusedCheckpoint('probe-body reload re-created the first load\'s fresh part %r (%s)' % (
                name, 'another module object' if now is not module else 'changed ' + ', '.join(changed[:4])))
    saved = getattr(model, '_entity_self_saved', None)
    if saved is not None:
        model._entity_self_saved = slice_entity_body(saved, n, b)
    wc = getattr(model, 'wellness_channels', None)
    if wc is not None:
        wc.load_state_dict(slice_batch(wc.state_dict(), n, b))


def slice_entity_body(saved, n: int, b: int):
    # Null-control count can equal body count (both 8 on H84). Its ncov/nvvar buffers are
    # shared controls, never body rows. A shape-only slice would silently corrupt them.
    batched = {f'{level}_{part}' for level in ('l2', 'l3')
               for part in ('p_fast', 'p_slow', 'v_fast', 'v_slow', 'nv_fast', 'nv_slow')}
    return {k: slice_batch(v, n, b) if k in batched else v.clone() for k, v in saved.items()}


def episode_body(ep: dict, b: int) -> dict:
    """Body b's world, learner and trainer state, as a one-body episode."""
    n = len(ep["worlds"])
    if not 0 <= b < n:
        raise ValueError(f"body {b} not in a {n}-body episode")
    learner = dict(ep["learner"])
    learner["objects"] = {k: v for k, v in learner["objects"].items() if k != "replay"}   # training only
    learner = slice_batch(learner, n, b)
    learner["cuda_rng"] = None            # the card's random stream: irrelevant on the CPU
    trainer = {k: slice_batch(v, n, b) for k, v in ep["trainer"].items() if k != "fit_execution"}
    meta = (ep["trainer"].get("frame_meta") or [None] * n)[b]
    return dict(world=copy.deepcopy(ep["worlds"][b]), learner=learner, trainer=trainer, frame_meta=meta,
                n_bodies=n, body=b)


# ============================================================================ the world
def world_kwargs(frozen: Frozen, *, seed: int, run_dir: Optional[str] = None, caregiver: bool = False,
                 critters: bool = False, mobile: bool = False, gaze_reflex: bool = False,
                 saccades: bool = False, vision: bool = True, overrides: Optional[dict] = None) -> dict:
    """The run's own world for one body (train_mimo.env_kwargs_for), with the probe's switches.

    Never records the caregiver (caregiver_record_dir '') and never replays a tape."""
    from mimo_bridge.train_mimo import env_kwargs_for
    run_dir = run_dir or str(PROBES_ROOT / "_scenes" / f"p{os.getpid()}")
    kw = env_kwargs_for(frozen.args, frozen.limits["phase_name"], run_dir)
    kw.update(seed=int(seed), tag=f"_gof{os.getpid()}", caregiver_record_dir="", caregiver_replay="")
    if not caregiver:
        kw.update(caregiver_mode="off", caregiver_seconds=0.0)
    kw["critters"] = bool(critters)
    kw["mobile"] = bool(mobile)
    kw["gaze_reflex"] = bool(gaze_reflex)
    kw["spontaneous_saccades"] = bool(saccades)
    if not vision:
        kw["vision"] = False
    if frozen.limits.get("age_months") is not None and "age" not in kw:
        kw["age"] = float(frozen.limits["age_months"])
    kw.update(overrides or {})
    return kw


def quat_mul_xyzw(a, b):
    av, aw, bv, bw = np.asarray(a[:3], float), float(a[3]), np.asarray(b[:3], float), float(b[3])
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - np.dot(av, bv)]


def axis_angle_xyzw(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    h = math.radians(deg) / 2.0
    return np.r_[axis * math.sin(h), math.cos(h)]


def rotation_deg_xyzw(q0, q1):
    """Angle in degrees of the turn from orientation q0 to q1 (x y z w). Taken as
    2 atan2(|vector part|, |scalar part|) of q1 * conj(q0), which stays exact near zero where
    acos of the dot product loses about a microdegree to rounding."""
    q0 = np.asarray(q0, float)
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = np.asarray(q1, float)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    r = quat_mul_xyzw(q1, np.r_[-q0[:3], q0[3]])
    return float(np.degrees(2.0 * math.atan2(np.linalg.norm(r[:3]), abs(r[3]))))


@dataclass
class Moment:
    """The state a prediction starts from (see the module docstring)."""
    tokens: Dict[str, torch.Tensor]
    ctx: torch.Tensor
    ach: torch.Tensor
    mem: Optional[torch.Tensor]
    context_source: str


# THE PREDICTION ROUTE EVERY MOMENT USED. Session.moment() takes the lesson's own context
# (_nf_ctx) when the step cached one and otherwise falls back to the context before the step; the
# two are different predictions. A probe resets this at its start, and route_record() writes what
# was used into its result and refuses a mix (review of the probes, 2026-09-23: the fallback was
# labelled on the moment but never written anywhere).
ROUTES: "collections.Counter[str]" = collections.Counter()
LESSON_CONTEXT = "lesson context (_nf_ctx, after the back-fill)"


def reset_routes() -> None:
    ROUTES.clear()


def route_record() -> dict:
    """{context_source, moments}: the one route every moment of this probe used. Raises
    ValueError when moments used different routes: a probe whose arms were predicted two ways
    cannot pool them."""
    used = dict(ROUTES)
    if len(used) > 1:
        raise ValueError("the probe's moments used different prediction routes %s; their predictions "
                         "cannot be pooled" % json.dumps(used))
    return dict(context_source=next(iter(used), None), moments=int(sum(used.values())))


# THE TRAINER'S BURST MERGE (mimo_bridge/train_mimo.py, the loop after src.step; read 2026-09-23).
# The model is stepped on the LAST env frame of a model step, whose task dict the trainer first
# overlays with events from the four frames before it. A probe that skipped this would hand the
# model (the contact head, the consequence and hand-pressing readers) a different report than
# training did.
_BURST_ANY = ("ball_touched", "ball_touched_hand", "gaze_on_ball", "gaze_on_ball_earned",
              "mobile_touched", "mobile_touched_hand")
_BURST_MEAN = ("hand_ball_dist_m", "roll_deg", "gaze_ball_deg")


def merge_burst_tasks(tasks: List[dict]) -> dict:
    """One model step's env-frame task dicts (oldest first) -> the dict the trainer hands the
    model: the last frame's, with booleans OR-ed, the first 'moment' kept, box touches united,
    three distances averaged pairwise in the trainer's order, roll_max_deg maximised, and the
    contact events merged by env.merge_burst_contact_report."""
    from mimo_bridge.env import merge_burst_contact_report
    if not tasks:
        return {}
    merged = dict(tasks[-1])
    for t in tasks[:-1]:
        t = t or {}
        for k in _BURST_ANY:
            if t.get(k):
                merged[k] = True
        if t.get("moment") and not merged.get("moment"):
            merged["moment"] = t["moment"]
        for k in ("box_touch", "box_touch_hand"):
            if t.get(k):
                merged[k] = sorted(set(merged.get(k) or []) | set(t[k]))
        for k in _BURST_MEAN:
            if t.get(k) is not None and merged.get(k) is not None:
                merged[k] = (merged[k] + t[k]) / 2.0
        if t.get("roll_max_deg") is not None:
            merged["roll_max_deg"] = max(merged.get("roll_max_deg", 0.0), t["roll_max_deg"])
    merged.update(merge_burst_contact_report(list(tasks)))
    return merged


# ============================================================================ pictures
# ONE PICTURE PER SCENE (2026-09-24, docs/gain_of_function_v4/agents/gaze_probe_replay.md). The
# eyes are drawn by OpenGL, and this machine's driver (Intel RaptorLake-S integrated graphics,
# OpenGL 4.6, driver 32.0.101.6129) does not return the same picture for the same scene when
# semi-transparent shapes overlap in view. Measured on the crib world: one unchanged scene drawn
# 400 times gave 23 different wide-eye pictures with the dragonfly's wings in view (the most common
# one 257 times, the next two 52 and 49 times), 29 different foveal pictures, 16 different wide-eye
# pictures with the bee's wings in view, and 1 with no critter in view. A picture differs from the
# most common one in 1 to 112 pixels, each examined difference one brightness level (1 of 255), and
# MuJoCo handed OpenGL the same scene every time. The scheduled eye test (probe_saved_gaze.py) runs every
# command twice from one saved state and requires the two runs to match exactly; it stopped on 12
# of its 17 attempts of 2026-09-24 with physics, touch, hearing and body position identical and
# only the vision token different (by up to 0.00001 on values near 3.4). The caregiver cut-point
# proof measured the same variation on 2026-09-13 (tools/prove_caregiver_render_repeatability.py:
# 2 of 10 frames differed with the physics exact).
# So a session draws a scene the first time it meets it and hands that picture out again when the
# same scene comes back (the second run of an arm). "The same scene" is decided by a fingerprint of
# everything MuJoCo passes to OpenGL for the draw: every shape's kind, ids, size, place, turn,
# colour and material values, the lights, the two OpenGL cameras, the drawing switches and the
# picture's size and camera. A scene that differs in any of those by any amount is drawn afresh.
# The fingerprints are kept in `log`, so a replay check can require identical scenes as well as
# identical pictures (probe_saved_gaze.py does). `redrawn_differently` counts the draws in which a
# fresh picture of a stored scene came back different (a driver slip, or the floor placed around
# the previous draw's camera, see wrap): the measured size of what this removes.
SCENE_PICTURES_KEPT = 64      # an arm's first run draws 3 model steps x 4 cameras = 12 scenes
SCENE_LOG_KEPT = 4096         # fingerprints kept for drain(); a probe that never drains cannot grow it

_GEOM_FIELDS = ("type", "dataid", "objtype", "objid", "category", "matid", "texcoord", "segid",
                "size", "pos", "mat", "rgba", "emission", "specular", "shininess", "reflectance",
                "camdist", "modelrbound", "transparent")
_LIGHT_FIELDS = ("pos", "dir", "directional", "castshadow", "bulbradius", "attenuation", "cutoff",
                 "exponent", "ambient", "diffuse", "specular", "headlight")
_CAMERA_FIELDS = ("pos", "forward", "up", "frustum_center", "frustum_width", "frustum_bottom",
                  "frustum_top", "frustum_near", "frustum_far", "orthographic")
_SCENE_FIELDS = ("flags", "framewidth", "framergb", "enabletransform", "translate", "rotate", "scale",
                 "stereo", "nskin", "skinvert", "skinnormal", "nflex", "flexvert", "flexnormal",
                 "flextexcoord")


_ABSENT = np.array([np.nan])


def _numbers(obj, fields) -> np.ndarray:
    """The named fields of one MuJoCo struct as one float64 row (float32 values convert exactly);
    a field this MuJoCo version lacks enters as NaN, so versions never collide silently."""
    parts = []
    for name in fields:
        value = getattr(obj, name, None)
        parts.append(_ABSENT if value is None else np.asarray(value, dtype=np.float64).reshape(-1))
    return np.concatenate(parts)


def scene_fingerprint(scene, camera_name: str, width: int, height: int) -> str:
    """SHA-256 of everything MuJoCo hands OpenGL for one draw: the MjvScene as mjv_updateScene
    left it (every drawn shape, the lights, the two OpenGL cameras, the switches and frame), and
    the camera name and picture size. Equal fingerprints mean MuJoCo asked for the same picture."""
    digest = hashlib.sha256(repr((str(camera_name), int(width), int(height))).encode())
    geoms = scene.geoms
    n = int(scene.ngeom)
    rows = [_numbers(geoms[i], _GEOM_FIELDS) for i in range(n)]
    digest.update(b"geoms %d" % n)
    if rows:
        digest.update(np.stack(rows).tobytes())
    lights = [_numbers(scene.lights[i], _LIGHT_FIELDS) for i in range(int(scene.nlight))]
    digest.update(b"lights %d" % len(lights))
    if lights:
        digest.update(np.stack(lights).tobytes())
    digest.update(np.stack([_numbers(scene.camera[i], _CAMERA_FIELDS) for i in range(2)]).tobytes())
    digest.update(_numbers(scene, _SCENE_FIELDS).tobytes())
    return digest.hexdigest()


class ScenePictures:
    """The pictures one session has drawn, by scene fingerprint (see the note above)."""

    def __init__(self, kept: int = SCENE_PICTURES_KEPT):
        self.kept = int(kept)
        self._pictures: "collections.OrderedDict[str, np.ndarray]" = collections.OrderedDict()
        self.log = collections.deque(maxlen=SCENE_LOG_KEPT)   # fingerprints since the last drain()
        self.draws = 0
        self.reused = 0                   # draws answered with a stored picture
        self.redrawn_differently = 0      # of those, draws whose fresh picture differed from the stored one

    def wrap(self, renderer) -> None:
        """Send one EyeRenderer's draws through this store (a second call changes nothing)."""
        if getattr(renderer, "_scene_pictures", None) is self:
            return
        import mujoco
        draw = renderer.render
        # THE FINGERPRINT OF THE REQUEST, NOT OF THE RENDERER'S HISTORY (2026-09-24, second fix).
        # MuJoCo places an infinite plane (the crib floor, geom 0, size "0 0 .25") around the camera
        # its scene held BEFORE the update, because mjv_updateScene adds the shapes before it
        # updates the camera. So one update of a scene that last drew another camera leaves the
        # floor somewhere else: measured with two fixed cameras 3.9 m apart, the floor's drawn
        # position was (0, 0, 0) or (4.24, -2.12, 0) for the same camera depending on the previous
        # draw, and identical with a second update. On checkpoint 397050, body 6, the scheduled eye
        # test failed at the same command in two processes (arm 57), and a one-body run reproduced
        # it on arms 7 to 10: physics, every sense input, every token and every picture identical,
        # and only the first draw on each renderer (left eye, wide and foveal) fingerprinted
        # differently, the floor 1.0 m apart in x. The first fingerprint read the renderer's own
        # scene; a private scene updated once carried the same history. So the store updates its
        # private scene twice from the same model, data, drawing options and a fixed camera: the
        # first update sets the scene's camera, the second places the floor around it. The result
        # is a function of what MuJoCo is asked to draw now and of nothing drawn before.
        private = mujoco.MjvScene(renderer.model, maxgeom=int(renderer.scene.maxgeom))
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

        def fingerprint(data, camera_name):
            camera.fixedcamid = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            for _ in range(2):
                mujoco.mjv_updateScene(renderer.model, data, renderer.option, None, camera,
                                       mujoco.mjtCatBit.mjCAT_ALL.value, private)
            private.flags[:] = renderer.scene.flags          # the renderer's drawing switches
            return scene_fingerprint(private, camera_name, renderer.width, renderer.height)

        def render(data, camera_name):
            picture = draw(data, camera_name)
            key = fingerprint(data, camera_name)
            self.draws += 1
            self.log.append(key)
            kept = self._pictures.get(key)
            if kept is None:
                self._pictures[key] = picture.copy()
                while len(self._pictures) > self.kept:
                    self._pictures.popitem(last=False)
                return picture
            self._pictures.move_to_end(key)
            self.reused += 1
            if not np.array_equal(kept, picture):
                self.redrawn_differently += 1
            return kept.copy()

        renderer.render = render
        renderer._scene_pictures = self

    def drain(self) -> List[str]:
        """The fingerprints logged since the last call, oldest first; the log starts empty again."""
        out = list(self.log)
        self.log.clear()
        return out

    def counts(self) -> dict:
        return dict(draws=self.draws, reused=self.reused, redrawn_differently=self.redrawn_differently)


# THE WORLD'S PER-FRAME TRACKERS (the mobile's chime gate, the mobile's speed, the jitter report, the
# roll's starting orientation and the command window) come back with the world's own snapshot since
# 2026-09-26 (BridgeEnv._FRAME_TRACKERS, docs/gain_of_function_v4/agents/env_full_state_trackers_2026-09-26.md).
# From 2026-09-24 until then a session saved and restored them itself (ENV_TRACKERS here): a run
# that chimed had blocked the chime in the next run of the same arm (gaze_probe_replay.md).
# The roll maximum (_roll_max) stays out of the world's snapshot: the trainer's "rolled over" share
# counts it per segment (BridgeEnv._FRAME_TRACKERS gives the readers). A session still saves and
# restores it itself, as it has since 2026-09-24, so every run of an arm starts from the snapshot's
# maximum and not from the one the previous run left behind.


class Session:
    """One body, one frozen model, identical restorable starts."""

    def __init__(self, frozen: Frozen, *, seed: int = 0, babble: float = 0.0,
                 world_of: Optional[Frozen] = None, episode: Optional[dict] = None, body: int = 0,
                 **world):
        """episode: the checkpoint's own episode (load_episode); the session is then body `body`
        of it at the saved moment - its world (caregiver, mobile, critters as saved, nothing
        recorded), context, memory and histories - with only the gaze reflex and spontaneous
        saccades switchable (world objects cannot change without changing the saved scene).
        world_of: build the body, its world and its limits from ANOTHER checkpoint's run (the
        earlier-checkpoint control with a fresh world)."""
        from mimo_bridge.datasource import SingleEnvSource
        from mimo_bridge.tools.probe_clock import ProbeClock
        self.frozen = frozen
        self.model = frozen.model
        self.args = frozen.args
        owner = world_of if world_of is not None else frozen
        self.world_args = owner.args          # the world's own limb contract (the ledger reads it)
        self.world = dict(world)
        self.reps = max(1, int(owner.args.action_repeat))
        self.hold = max(1, int(owner.args.motor_hold))
        self.dev = dict(owner.limits, exploration_noise=float(babble))
        self.executed = None
        self.order_id = 0
        self.res = None
        self.pre_ctx = None
        self.template = None          # radius/strength columns of the model's own last order
        self._policy_i = 0
        self._held = None
        self._held_hand_goal = None
        self._probe_hand_cache = None
        self.last_motor_substeps = None
        self.last_hand_joint_goal = None
        self.last_physical_order = None       # the last act()'s order as sent to the world (canonical hand form)
        self.last_gaze_planned = None         # the last act()'s commanded eye angles (known before the step)
        self.episode_body = None
        self.pictures = ScenePictures()
        if episode is not None:
            self._from_episode(episode, int(body), SingleEnvSource)
        else:
            self.kw = world_kwargs(owner, seed=seed, **world)
            self.src = SingleEnvSource(**self.kw)
            self.env = self.src.env
            self.model.reset_transition_state()
            self.frames = self.src.reset()
            self.start = "fresh world (empty context and memory)"
        self._fix_pictures()
        self.clock = ProbeClock(float(self.env.dt), self.reps, self.hold)
        self.n_handles = len(self.env.proprio.body_ids)
        self.handle_names = list(self.env.proprio.handle_names)
        self.gaze_idx = self.handle_names.index("gaze")
        self._check_finger_rule()
        self._configure_hand_control()

    def _check_finger_rule(self):
        """--finger-joint-servo (action_decoder.py, THE FINGER COMMAND'S MEANING). With hand goals on, the learner's
        hand stage predicts with a calibration checked against the learner's own setting
        (LemLite.configure_hand_geometry); this session's world comes from overrides, another run's settings or a
        saved world, so it must run the same finger rule, or the hand stage would predict a command the world does not
        execute (the trainer binds its worlds the same way: limb_target_control.check_finger_servo)."""
        if (getattr(self.model, 'hand_goals', 'off') or 'off') == 'off':
            return
        world_rule = getattr(self.env, 'finger_joint_servo', 'off') or 'off'
        learner_rule = getattr(self.model, 'finger_joint_servo', 'off') or 'off'
        if world_rule != learner_rule:
            raise ValueError("probe world runs --finger-joint-servo %s but the learner's hand stage was set up for "
                             "--finger-joint-servo %s (--hand-goals %s): its fixed response would predict a command "
                             "the world does not execute" % (world_rule, learner_rule, self.model.hand_goals))

    def _fix_pictures(self):
        """Send every eye camera of this session's world through self.pictures (see ScenePictures:
        the same scene gets the same picture). The watch camera, used only for videos, is left out."""
        env = self.env
        if not getattr(env, "vision", False):
            return
        self.pictures.wrap(env._renderer_obj())
        if int(getattr(env, "fovea_size", 0) or 0) > 0:
            self.pictures.wrap(env._fovea_renderer_obj())

    def _configure_hand_control(self):
        if getattr(self.model, 'hand_target_mode', 'world') == 'world':
            return
        control = getattr(self.env, 'hand_target_control', None)
        if control is None:
            raise ValueError('Probe learner and restored world disagree on the hand contract')
        geometry = control.geometry
        saved = getattr(self.model, '_hand_geometry', None)
        if saved is not None and saved != geometry:
            raise ValueError('Probe hand geometry differs from its saved learner')
        previous = getattr(self.model, '_hand_previous_joints', None)
        if previous is None:
            previous = (self._held_hand_goal if self._held_hand_goal is not None else
                        np.array(self.env.data.qpos[control._joint_addresses])[None])
        self.model.configure_hand_geometry(geometry, previous)
        if self._held_hand_goal is None:
            self._held = None

    def _from_episode(self, episode: dict, body: int, SingleEnvSource):
        from mimo_bridge import continuation_state
        part = episode_body(episode, body)
        restore_runtime_body(self.frozen, part['n_bodies'], body)
        world = part["world"]
        kw = dict(world["env_kwargs"])
        kw.update(run_dir=str(PROBES_ROOT / "_scenes" / f"p{os.getpid()}"), tag=f"_gof{os.getpid()}b{body}",
                  caregiver_record_dir="", caregiver_replay="")
        if "gaze_reflex" in self.world:
            kw["gaze_reflex"] = bool(self.world["gaze_reflex"])
        if "saccades" in self.world:
            kw["spontaneous_saccades"] = bool(self.world["saccades"])
        if "vision" in self.world:
            # a measurement without a graphics card (proprio_token_measurements.py grade): the camera off, the
            # saved world otherwise as it was; the model then reads its vision sense as stale
            kw["vision"] = bool(self.world["vision"])
        world["env_kwargs"] = kw
        self.kw = kw
        self.src = SingleEnvSource(**kw)
        self.env = self.src.env
        self.frames = self.src.restore_full_state([world])
        if part["frame_meta"] is not None:
            self.frames[0].meta = continuation_state.tree(part["frame_meta"])
        continuation_state.restore_learner(self.model, part["learner"])
        schemas = getattr(self.model, "_schemas", None)
        if isinstance(schemas, list) and len(schemas) > body:
            self.model._schemas = [schemas[body]]
        st = part["trainer"]
        self.executed = np.array(st["prev_executed"], dtype=np.float32, copy=True)
        self.order_id = int(st["prev_order_id"])
        self._held = np.array(st["held_motor"], dtype=np.float32, copy=True)
        goal = st.get('held_hand_joint_goal')
        self._held_hand_goal = None if goal is None else np.array(goal, copy=True)
        self.template = self._held.copy()
        self._policy_i = int(st["step"])        # the trainer latches on the global step
        self.episode_body = dict(body=body, n_bodies=part["n_bodies"], step=int(st["step"]))
        self.start = f"episode companion, body {body} of {part['n_bodies']} at step {int(st['step'])}"

    def close(self):
        try:
            self.src.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ stepping
    def observe(self):
        """The model steps on the current frames (learning off). Afterwards it holds the moment."""
        m = self.model
        self.pre_ctx = None if m.context_buffer is None else m.context_buffer.detach().clone()
        self.res = m.step(self.frames, self.dev, train=False, step_index=m.step_count,
                          executed_action=self.executed,
                          executed_order_id=(self.order_id if self.executed is not None else None))
        return self.res

    def eyes_rad(self):
        (yl, pl), _ = self.env.eye_yaw_pitch()
        return np.array([yl, pl], dtype=np.float64)

    def _canonical_hand_order(self, order, hand_joint_goal=None):
        """One feasible canonical request, shared by prediction and execution."""
        if getattr(self.model, 'hand_target_mode', 'world') == 'world':
            if hand_joint_goal is not None:
                raise ValueError('Explicit hand goals require the wrist-relative contract')
            return order, None
        kin = self.model.hand_kinematics
        indices = sorted({i for h in kin.geometry['hands'] for i in [h['wrist_index']]+h['finger_indices']})
        supplied = None if hand_joint_goal is None else np.asarray(hand_joint_goal).reshape(1,-1)
        cache = self._probe_hand_cache
        same_goal = cache is not None and ((supplied is None and (cache['supplied'] is None or
            np.array_equal(order[:,indices,:7],cache['canonical'][:,indices,:7]))) or
            (supplied is not None and cache['supplied'] is not None and np.array_equal(supplied,cache['supplied'])))
        if same_goal and any(np.array_equal(order[:,indices,:7],cache[k][:,indices,:7])
                            for k in ['requested','canonical']):
            canonical = order.copy()
            fingers = [i for h in kin.geometry['hands'] for i in h['finger_indices']]
            canonical[:,fingers,:7] = cache['canonical'][:,fingers,:7]
            return canonical, cache['joints'].copy()
        if supplied is None:
            canonical, joints, _ = kin.project_world_orders(order, position_scale=1.)
        else:
            kin.validate_joints(supplied,tolerance=2e-6)
            joints = supplied.copy()
            canonical = kin.canonical_from_joints(order[0],joints[0],position_scale=1.)[None]
        canonical = np.asarray(canonical,dtype=np.float32)
        self._probe_hand_cache = dict(requested=order.copy(),canonical=canonical.copy(),
                                     supplied=copy.deepcopy(supplied),joints=np.array(joints,copy=True))
        return canonical, joints

    def _ledger_order(self, issued, physical, joint_goal):
        """The body rows the trainer records for the model after a burst (train_mimo.py: prev_executed =
        gaze_to_handle(send_motor if limb_ledger is None else limb_ledger, ...)): the command as issued - for the
        wrist-relative hand contract its finger rows are the planner's own (assemble_world) form, not the canonical
        hand form _canonical_hand_order sends the world (MuJoCo/SciPy quaternions, which can carry the opposite sign
        for the same rotation; the model reads raw numbers) - or, when the world runs limb projections
        (--parent-relative-targets, --reach-projection), the world's limb ledger (limb_target_control.ledger_from_frames,
        the trainer's own reader, given what this world received: `physical`). The world copies the rows it does not
        own from what it received, so the finger columns 0:7 (the only ones the canonical form changes) are put back to
        the issued ones, except under reach projection with a joint goal, where the trainer too records the world's
        finger placement."""
        from mimo_bridge.hand_target_control import DIGITS as HAND_DIGITS
        from mimo_bridge.limb_target_control import ledger_from_frames
        fingers = [i for i, n in enumerate(self.handle_names) if n.split('_')[0] in HAND_DIGITS]
        reach = getattr(self.world_args, 'reach_projection', 'off')
        limb = ledger_from_frames(
            [f.meta for f in self.frames], physical, self.handle_names,
            parent_relative=getattr(self.world_args, 'parent_relative_targets', 'off'),
            reach_projection=reach, finger_rows=(fingers if joint_goal is not None else ()))
        if limb is None:
            return issued
        led = np.asarray(limb, dtype=np.float32)
        if not (reach == 'arms' and joint_goal is not None):
            led[:, fingers, :7] = np.asarray(issued, dtype=np.float32)[:, fingers, :7]
        return led

    def act(self, order, gaze_rad=None, new_order: bool = True, render: bool = True,
            on_env_step=None, hand_joint_goal=None):
        """Send one order for one model step (action_repeat env steps). gaze_rad None holds the
        eyes where they are. Returns the new frames. The world receives the canonical hand form
        (_canonical_hand_order; last_physical_order) with the hand joint goal; the model is told
        the order the trainer's ledger writes (_ledger_order: the command as issued, or the world's
        limb ledger) with the ISSUED eye row (the mean eye command the world applied this model step,
        saccades and the gaze reflex included; last_gaze_sent), which becomes the next observe()'s
        action. The commanded eye angles stay apart (last_gaze_planned): the issued row exists only
        after the step. The last frame's task dict is overlaid with the burst's events exactly as the
        trainer does (merge_burst_tasks), so the model reads the report it trained on."""
        from mimo_bridge.lem_model import LemLite
        issued = np.asarray(order, dtype=np.float32).reshape(1, self.n_handles, 9)
        order, joint_goal = self._canonical_hand_order(issued, hand_joint_goal)
        gaze = self.eyes_rad() if gaze_rad is None else np.asarray(gaze_rad, dtype=np.float64)
        gz, tasks, motor_substeps = [], [], []
        for rep in range(self.reps):
            self.frames = self.src.step(order, gaze.reshape(1, 2), None,
                                        render=bool(render and rep == self.reps - 1),
                                        **({} if joint_goal is None else {'hand_joint_goal':joint_goal}))
            gz.append(np.asarray(self.frames[0].meta.get("gaze_cmd") or gaze, dtype=np.float64))
            tasks.append(dict(self.frames[0].meta.get("task") or {}))
            if joint_goal is not None:
                meta = self.frames[0].meta
                if meta.get('hand_target_contract') != 'wrist_relative_joint_v1':
                    raise RuntimeError('Probe world did not execute the versioned hand goal')
                if not np.allclose(np.asarray(meta['hand_joint_goal']),joint_goal[0],rtol=0,atol=2e-6):
                    raise RuntimeError('Probe world executed a different hand joint goal')
                motor_substeps.append(np.asarray(meta['executed_motor_targets']).copy())
            if on_env_step is not None:
                on_env_step(self, rep)
        if self.frames and self.frames[0].meta.get("task") is not None:
            self.frames[0].meta["task"] = merge_burst_tasks(tasks)
        if new_order:
            self.order_id += 1
        self.last_gaze_sent = np.mean(gz, axis=0)
        self.last_gaze_planned = np.array(gaze, dtype=np.float64, copy=True)
        self.last_physical_order = np.array(order, copy=True)
        # in the probed model's own eye row convention, as the trainer's ledger writes it (motor
        # interface audit F1; 'legacy' for every checkpoint trained before --eye-row-convention)
        self.executed = LemLite.gaze_to_handle(self._ledger_order(issued, order, joint_goal), self.last_gaze_sent[None],
                                               convention=getattr(self.model, "eye_row_convention", "legacy"))
        self.last_hand_joint_goal = None if joint_goal is None else np.array(joint_goal,copy=True)
        self.last_motor_substeps = None if joint_goal is None else np.stack(motor_substeps)[None]
        if joint_goal is not None:
            self.frames[0].meta['motor_substeps'] = self.last_motor_substeps[0].copy()
        return self.frames

    def exec_order(self, order, gaze_rad=None, hand_joint_goal=None):
        """`order` as the trainer's ledger records it (the command as issued, not the canonical hand
        form act() sends to the world) with the eye row of the PLANNED eye command gaze_rad
        (LemLite.gaze_to_handle; the eyes where they are now when gaze_rad is None): what is known
        before the step (review of the probes, 2026-09-23: the eye row used before was the eye
        BODY's world orientation, 161 degrees from the command row). After act() the model is told
        the ISSUED eye row instead (session.executed), which no planner can read beforehand; with
        limb projections on it is also told the world's limb ledger, known only after the step.
        hand_joint_goal is accepted for the callers' form; act() validates and executes it."""
        from mimo_bridge.lem_model import LemLite
        if hand_joint_goal is not None:       # the goal checks act() applies, without its hand cache
            if getattr(self.model, 'hand_target_mode', 'world') == 'world':
                raise ValueError('Explicit hand goals require the wrist-relative contract')
            self.model.hand_kinematics.validate_joints(np.asarray(hand_joint_goal).reshape(1, -1), tolerance=2e-6)
        g = self.eyes_rad() if gaze_rad is None else np.asarray(gaze_rad, dtype=np.float64)
        o = np.asarray(order, dtype=np.float32).reshape(1, self.n_handles, 9)
        return LemLite.gaze_to_handle(o, g.reshape(1, 2),
                                      convention=getattr(self.model, "eye_row_convention", "legacy"))

    def policy_step(self):
        """One model step of the model's OWN behaviour, latched like the trainer (a new order
        every motor_hold steps; the eyes follow the model's gaze output every step)."""
        self.observe()
        return self.policy_act()

    def policy_act(self):
        """The second half of policy_step: act on the output of the observe() just made."""
        res = self.res
        new = self._held is None or self._policy_i % self.hold == 0
        if new:
            self._held = np.array(res.motor_for_action, dtype=np.float32, copy=True)
            goal = getattr(res, 'hand_joint_goal', None)
            self._held_hand_goal = None if goal is None else np.array(goal,copy=True)
            self.template = self._held.copy()
        self.act(self._held, gaze_rad=np.asarray(res.gaze_out, dtype=np.float64).reshape(-1)[:2],
                 new_order=new, hand_joint_goal=self._held_hand_goal)
        self._policy_i += 1
        return res

    def warm(self, n_steps: int):
        """Let the model live in the world for n model steps (fills its context)."""
        for _ in range(int(n_steps)):
            self.policy_step()

    def collect_starts(self, n: int, spacing: int, *, want_plan: bool = False, max_steps: int = 400,
                       plan_fallback: bool = False) -> List[dict]:
        """Live on under the model's own behaviour and keep n snapshots at least `spacing` model
        steps apart, each taken right after an observation and before the action. want_plan: keep
        only moments where the planner made a plan this step (plan0 is then in the snapshot's
        'plan0'); ignored when the build has no planner.

        plan_fallback (with want_plan): the first planning event without a plan at least `spacing`
        steps after the last start is kept in reserve, and when the window ends short of n starts
        it becomes the next start, with plan0 None (see START_WITHOUT_A_PLAN). A plan found later
        in the window still wins.

        Every start records under 'planner' what the planner did at that moment (planner_decision,
        or consulted False), how the start was chosen ('rule') and the planning events seen so far
        ('search'); self.last_search keeps the whole search, so a caller that refuses a short one
        can say why."""
        m = self.model
        planner = getattr(m, "intention_gen", None) is not None
        wanted = bool(want_plan and planner)
        starts, last, i = [], -10 ** 9, 0
        reserve, seen = None, []
        while len(starts) < int(n) and i < int(max_steps):
            prior = getattr(m, "_last_plan_inputs", None)
            self.observe()
            lpi = fresh_plan_inputs(m, prior) if planner else None
            decision = planner_decision(m, lpi, i)
            if decision is not None:
                seen.append(decision)
            planned = bool(decision and decision["planned"])
            if i - last >= spacing and (planned or not wanted):
                snap = self.snapshot()
                snap["plan0"] = None if not planned else lpi["plan0"].detach().float().clone()
                snap["step_offset"] = i
                snap["planner"] = dict(decision or dict(consulted=False),
                                       rule="planned" if planned else "no_plan_required", search=list(seen))
                starts.append(snap)
                last, reserve = i, None
            elif plan_fallback and wanted and decision is not None and reserve is None and i - last >= spacing:
                reserve = self.snapshot()
                reserve["plan0"] = None
                reserve["step_offset"] = i
                reserve["planner"] = dict(decision, rule="no_plan_in_window")
            self.policy_act()
            i += 1
        if len(starts) < int(n) and reserve is not None:
            reserve["planner"]["search"] = list(seen)
            starts.append(reserve)
        self.last_search = dict(model_steps=i, max_steps=int(max_steps), plan_wanted=wanted,
                                plan_fallback=bool(plan_fallback), planning_events=seen)
        return starts

    # ------------------------------------------------------------------ snapshots
    def snapshot(self) -> dict:
        """Call right after observe(): world, complete model execution state, ledger."""
        from mimo_bridge import copied_execution
        return dict(world=self.src.save_full_state(), roll_max=getattr(self.env, "_roll_max", None),
                    model=copied_execution.capture(self.model),
                    executed=None if self.executed is None else self.executed.copy(),
                    order_id=self.order_id, pre_ctx=None if self.pre_ctx is None else self.pre_ctx.clone(),
                    res=SimpleNamespace(motor_for_action=np.array(self.res.motor_for_action, copy=True),
                                        gaze_out=np.array(self.res.gaze_out, copy=True),
                                        hand_joint_goal=copy.deepcopy(getattr(self.res,'hand_joint_goal',None))) if self.res is not None else None,
                    template=None if self.template is None else self.template.copy(),
                    policy_i=self._policy_i, held=None if self._held is None else self._held.copy(),
                    held_hand_goal=copy.deepcopy(self._held_hand_goal),hand_cache=copy.deepcopy(self._probe_hand_cache),
                    motor_substeps=copy.deepcopy(self.last_motor_substeps),hand_goal=copy.deepcopy(self.last_hand_joint_goal))

    def restore(self, snap: dict):
        from mimo_bridge import copied_execution
        self.frames = self.src.restore_full_state(snap["world"])
        if snap.get("roll_max") is not None:           # the world's snapshot leaves it out (see above)
            self.env._roll_max = snap["roll_max"]
        copied_execution.restore(self.model, snap["model"])
        self.executed = None if snap["executed"] is None else snap["executed"].copy()
        self.order_id = snap["order_id"]
        self.pre_ctx = None if snap["pre_ctx"] is None else snap["pre_ctx"].clone()
        self.res = snap["res"]
        self.template = None if snap["template"] is None else snap["template"].copy()
        self._policy_i = snap["policy_i"]
        self._held = None if snap["held"] is None else snap["held"].copy()
        self._held_hand_goal = copy.deepcopy(snap.get('held_hand_goal'))
        self._probe_hand_cache = copy.deepcopy(snap.get('hand_cache'))
        self.last_motor_substeps = copy.deepcopy(snap.get('motor_substeps'))
        self.last_hand_joint_goal = copy.deepcopy(snap.get('hand_goal'))
        self._configure_hand_control()
        return self.frames

    # ------------------------------------------------------------------ readouts
    def moment(self) -> Moment:
        m = self.model
        nf_ctx = getattr(m, "_nf_ctx", None)
        if torch.is_tensor(nf_ctx) and m.context_buffer is not None and nf_ctx.shape == m.context_buffer.shape:
            ctx, source = nf_ctx, LESSON_CONTEXT
        else:
            ctx, source = self.pre_ctx, "context before the step (next-frame lesson off)"
        ROUTES[source] += 1
        mem = getattr(m, "_memory_bias_current", None)
        return Moment(tokens={k: v.detach().float().clone() for k, v in (m.last_tokens or {}).items()},
                      ctx=ctx.detach().float().clone(), ach=m.last_achieved.detach().float().clone(),
                      mem=mem.detach().float().clone() if torch.is_tensor(mem) else None,
                      context_source=source)

    def pose(self):
        """(positions cm [n,3], orientations x y z w [n,4]) of every handle's body, world frame."""
        d, ids = self.env.data, self.env.proprio.body_ids
        return (np.array(d.xpos[ids], dtype=np.float64) * 100.0,
                np.array(d.xquat[ids], dtype=np.float64)[:, [1, 2, 3, 0]])

    def hold_order(self, strength: Optional[float] = None):
        """The order "stay where you are": every body handle's target is its current place and
        orientation, and the eyes' row is the command that keeps the eyes where they are, in the
        form act() sends it (exec_order). Radius and strength come from the model's own last
        order when one exists (so the command columns stay in the range the encoder has seen),
        else the strength limit."""
        pos, quat = self.pose()
        order = np.zeros((1, self.n_handles, 9), dtype=np.float32)
        order[0, :, 0:3] = pos
        order[0, :, 3:7] = quat
        if self.template is not None:
            order[0, :, 7:9] = self.template[0, :, 7:9]
        else:
            order[0, :, 8] = float(self.dev.get("strength_limit", 1.0))
        if strength is not None:
            order[0, :, 8] = float(strength)
        return self.exec_order(order)

    def readout(self) -> dict:
        """What happened, read after observe(): the model's raw tokens and the physical body."""
        pos, quat = self.pose()
        eyes = np.degrees(self.eyes_rad())
        f = self.frames[0]
        return dict(tokens={k: v.detach().float().clone() for k, v in (self.model.last_tokens or {}).items()},
                    pos_cm=pos, quat=quat, eyes_deg=eyes, task=copy.deepcopy(f.meta.get("task") or {}),
                    touch_self=None if f.touch_self is None else np.array(f.touch_self, copy=True),
                    sim_time=float(f.sim_time))

    def run_arm(self, snap: dict, order, *, gaze_rad=None, steps: int = 1, before_step=None,
                hand_joint_goal=None, on_env_step=None) -> List[dict]:
        """From the restored snapshot: send `order` (held, one order id) for `steps` model steps
        and observe after each. before_step(session, k) runs before each step's order is sent
        (a perturbation hook). Returns one readout per step."""
        self.restore(snap)
        out = []
        for k in range(int(steps)):
            if before_step is not None:
                before_step(self, k)
            self.act(order, gaze_rad=gaze_rad, new_order=(k == 0),
                     hand_joint_goal=hand_joint_goal, on_env_step=on_env_step)
            self.observe()
            out.append(self.readout())
        return out

    def physics_only(self, snap: dict, order, steps: int, *, gaze_rad=None, watch=None,
                     render: bool = False) -> List[dict]:
        """From the restored snapshot, send `order` for `steps` model steps WITHOUT the model
        observing (the body only; a reachability search). watch(session) is called after each
        model step and its return values are collected. render True renders the last env step of
        every model step as act() and run_arm do (the eye reflex observes rendered frames only)."""
        self.restore(snap)
        out = []
        for k in range(int(steps)):
            self.act(order, gaze_rad=gaze_rad, new_order=(k == 0), render=render)
            out.append(watch(self) if watch is not None else None)
        return out

    def body_ids_under(self, root_name: str) -> set:
        """Every body in the subtree of a named body (a hand, the head, ...)."""
        m = self.env.model
        root = m.body(root_name).id
        out = set()
        for b in range(m.nbody):
            p = b
            while p > 0 and p != root:
                p = int(m.body_parentid[p])
            if p == root:
                out.add(b)
        return out

    def contact_between(self, group_a: set, group_b: set) -> float:
        """Total normal force (N) of the contacts now touching group_a against group_b."""
        e = self.env
        total = 0.0
        f = np.zeros(6)
        for i in range(e.data.ncon):
            c = e.data.contact[i]
            b1, b2 = int(e.model.geom_bodyid[c.geom1]), int(e.model.geom_bodyid[c.geom2])
            if (b1 in group_a and b2 in group_b) or (b1 in group_b and b2 in group_a):
                e._mj.mj_contactForce(e.model, e.data, i, f)
                total += abs(float(f[0]))
        return total

    def null_order(self):
        """The tenure lesson's "do nothing" order: identity rotation, zero strength, zero place
        (lem_model's tenure lesson, the do-nothing control its pose head is judged against)."""
        o = np.zeros((1, self.n_handles, 9), dtype=np.float32)
        o[..., 6] = 1.0
        return o

    def rotated(self, order, handle: str, axis, deg: float):
        """A copy of `order` whose handle target is turned by `deg` about a world axis."""
        i = self.handle_names.index(handle)
        o = np.array(order, dtype=np.float32, copy=True)
        q = quat_mul_xyzw(axis_angle_xyzw(axis, deg), o[0, i, 3:7])
        o[0, i, 3:7] = q / max(np.linalg.norm(q), 1e-12)
        return o

    def joint_axes(self, handle: str) -> List[dict]:
        """The hinge joints the decoder moves for a handle, with world axis and range."""
        e, dec = self.env, self.env.decoder
        i = self.handle_names.index(handle)
        out = []
        for j in dec.handles[i].joint_ids:
            j = int(j)
            if int(e.model.jnt_type[j]) != int(e._mj.mjtJoint.mjJNT_HINGE):
                continue
            q = float(e.data.qpos[int(e.model.jnt_qposadr[j])])
            lo, hi = (float(x) for x in e.model.jnt_range[j])
            out.append(dict(joint=e.model.joint(j).name, axis=np.array(e.data.xaxis[j], dtype=np.float64),
                            q_deg=math.degrees(q), lo_deg=math.degrees(lo), hi_deg=math.degrees(hi)))
        return out


class NoStartState(RuntimeError):
    """A body gave no start state inside its search window. start_states raises this instead of
    moving on to the next body (see START_WITHOUT_A_PLAN)."""


def fresh_plan_inputs(model, prior):
    """The planning record the observe just made, or None when that observe did not plan.

    NEVER WIPE model._last_plan_inputs TO DETECT A PLAN (2026-09-29). The persistent memory reads the last chosen
    goal from it at every step (memory_integration.observe: goal_summary), and production never clears it, so a
    probe that set it to None before an observe fed the memory "no goal" where production feeds the real one. The
    remembered-hand comparison failed its own repeat check at 414,200 on exactly that: its first branch wiped the
    record, its repeat restored it, and 196 of 234 command numbers differed. LemLite._plan_chain's caller writes a
    new dict at every planning event, so a new object means this observe planned. Take `prior` right before the
    observe: prior = model._last_plan_inputs."""
    now = getattr(model, "_last_plan_inputs", None)
    return None if now is prior else now


def planner_decision(model, inputs, offset: int) -> Optional[dict]:
    """What the planner did at one moment, read from the record it writes only on a planning event
    (LemLite._last_plan_inputs); None when it was not consulted. planned: it returned an order
    (plan0). plan_length, stopped_by, converged: its own outcome (lem_model._plan_chain).
    goal_distance_now: the distance the plan's first check measures (lem_model._plan_chain: over
    the senses graded real, the motor token left out, the sum of each sense's mean squared
    difference between the present tokens and the goal's); 0 when the goal is the present moment,
    and under 0.01 the planner stops as converged without an order."""
    if not inputs:
        return None
    outcome = inputs.get("outcome") or {}
    out = dict(consulted=True, step_offset=int(offset), step=int(getattr(model, "step_count", 0)) - 1,
               planned=inputs.get("plan0") is not None, plan_length=outcome.get("plan_length"),
               stopped_by=outcome.get("stopped_by"), converged=outcome.get("converged"),
               goal_distance_now=None)
    cur, goal = inputs.get("cur") or {}, inputs.get("goal") or {}
    mask = (inputs.get("decision") or {}).get("source_mask") or {}
    senses = ([m for m in cur if m != "motor" and m in goal and str(mask.get(m, "real")) == "real"]
              or [m for m in cur if m in goal])
    if senses and all(torch.is_tensor(cur[m]) and torch.is_tensor(goal[m]) and cur[m].shape == goal[m].shape
                      for m in senses):
        out["goal_distance_now"] = float(sum(float(((cur[m].detach().float() - goal[m].detach().float()) ** 2)
                                                   .mean()) for m in senses))
    return out


def _no_start_text(who: str, search: Optional[dict], found: int = 0, wanted: int = 1) -> str:
    """Why a search kept fewer starts than asked, from Session.last_search: 'body 1 gave no start
    state in 30 model steps. The planner was consulted 3 times: 0 made a plan, 3 stopped without
    one (converged 3 times); its goal sat 0, 0, 0 from the present moment ...'."""
    search = search or {}
    events = list(search.get("planning_events") or [])
    planned = sum(1 for e in events if e.get("planned"))
    reasons = collections.Counter(str(e.get("stopped_by")) for e in events if not e.get("planned"))
    kept = "no start state" if int(wanted) == 1 else "%d of %d start states" % (int(found), int(wanted))
    text = "%s gave %s in %s model steps. " % (who, kept, search.get("model_steps", "?"))
    if not events:
        text += "The planner was never consulted"
    else:
        text += "The planner was consulted %d time%s: %d made a plan, %d stopped without one (%s)" % (
            len(events), "" if len(events) == 1 else "s", planned, len(events) - planned,
            ", ".join("%s %d times" % kv for kv in sorted(reasons.items())) or "none")
    distances = ["%.4g" % e["goal_distance_now"] for e in events
                 if not e.get("planned") and e.get("goal_distance_now") is not None]
    if distances:
        text += ("; its goal sat %s from the present moment (under 0.01 the planner stops as converged)"
                 % ", ".join(distances[:8] + (["and %d more" % (len(distances) - 8)] if len(distances) > 8 else [])))
    return text + ". start_states never moves on without a body (START_WITHOUT_A_PLAN)."


# START_WITHOUT_A_PLAN (2026-09-25, docs/gain_of_function_v4/agents/gaze_body2_2026-09-25.md). A
# probe that asks for starts at planning moments (want_plan) used to lose a body without a word
# when its planner made no plan inside the search window: start_states moved on to the next body,
# and the eye test stopped later with "All eight bodies must complete both gaze conditions".
# Measured on checkpoint 400,350 of H84 (the run's causal_chain.jsonl): body 2's harm reading sat
# over the integrity gate line (0.92) at every one of its 200 remembered moments (0.9998 easing to
# 0.9969 over steps 399,800 to 400,340). With every moment over the line the goal ranking keeps
# the least harmful one (wellness_channels.rank_values), and with the pain easing that is always
# the newest, so the goal was the present moment at all 55 planning events. Run alone, the
# planner's first check then measures a distance of 0 and stops as converged without an order, at
# every planning event. In the eight-body trainer the same check reads the mean over all bodies,
# so the live run still planned. So a body that gives no start is now an error naming the body and
# what the planner did (NoStartState), and a probe whose measurement does not need the plan may
# pass plan_fallback: the body then starts at its first planning event without a plan, and the
# start says so under 'planner'.
def start_states(frozen: Frozen, n: int, *, want_plan: bool = False, warm: int = 300, spacing: int = 10,
                 seed: int = 0, babble: Optional[float] = None, world_of: Optional[Frozen] = None,
                 use_episode: bool = True, plan_fallback: bool = False, **world):
    """Yield (session, snapshot) for n start states, each snapshot taken right after the model
    observed and before it acted.

    From the checkpoint's episode companion when it exists: body 0, 1, ... of the saved moment
    (the model's own lived states), each searched for 3 x `spacing` model steps. Otherwise one
    fresh world, warmed `warm` model steps (300: the measured point where the tenure head's own
    error had fallen to 3x its live value, while vision stayed 25x - see the episode note above),
    then n states `spacing` steps apart. A body or world that gives fewer starts than asked raises
    NoStartState (START_WITHOUT_A_PLAN); plan_fallback is Session.collect_starts'.
    The session is closed when the caller moves to the next state."""
    babble = float(getattr((world_of or frozen).args, "babble_noise", 0.0) or 0.0) if babble is None else babble
    ep, note = (load_episode(frozen) if (use_episode and world_of is None) else (None, "fresh world requested"))
    if ep is not None:
        for b in range(min(int(n), len(ep["worlds"]))):
            s = Session(frozen, episode=ep, body=b, babble=babble, **world)
            try:
                s.start_note = note
                snaps = s.collect_starts(1, 0, want_plan=want_plan, max_steps=max(1, 3 * int(spacing)),
                                         plan_fallback=plan_fallback)
                if not snaps:
                    raise NoStartState(_no_start_text("body %d" % b, getattr(s, "last_search", None)))
                for snap in snaps:
                    yield s, snap
            finally:
                s.close()
        return
    s = Session(frozen, seed=seed, babble=babble, world_of=world_of, **world)
    try:
        s.start_note = note
        s.warm(warm)
        snaps = s.collect_starts(n, spacing, want_plan=want_plan, plan_fallback=plan_fallback)
        if len(snaps) < int(n):
            raise NoStartState(_no_start_text("the fresh world", getattr(s, "last_search", None),
                                              found=len(snaps), wanted=int(n)))
        for snap in snaps:
            yield s, snap
    finally:
        s.close()


# ============================================================================ predictions
@torch.no_grad()
def predict(model, mom: Moment, order_exec, *, want_eff: bool = True) -> dict:
    """What the model expects from this moment if `order_exec` is sent (executed units: cm,
    world quaternions). The next-frame lesson's route with the memory bias.

    Returns preds {sense: [1, d]} (the heads' raw output, what the lesson grades), last (the
    forward model's last-step output), tenure_cm [n, 3] (the tenure pose head's predicted
    250 ms change per handle in cm) when that head exists, and eff (efference heads in cm and
    degrees) when they exist."""
    from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, command_for_encoder
    act = torch.as_tensor(np.asarray(order_exec, dtype=np.float32)).reshape(1, -1, 9)
    enc_act = command_for_encoder(act)
    motor_in = torch.cat([mom.ach[..., :MOTOR_ACHIEVED_F], enc_act], dim=-1)
    z = model.encoder_registry.encode({"motor": motor_in}, {"motor": "real"})["motor"].reshape(1, 1, model.d_model)
    tokens = {}
    for m in model.active:
        if m == "motor":
            tokens[m] = z
        elif m in mom.tokens:
            tokens[m] = mom.tokens[m].reshape(1, 1, model.d_model)
    kw = {} if mom.mem is None else {"memory_bias": mom.mem}
    _l, last, _c = model.forward_model(tokens, mom.ctx, **kw)
    heads = model.prediction_heads
    preds = heads(last, current={m: v.reshape(1, -1) for m, v in tokens.items()
                                 if m in heads.head_modalities}, who="gof_probe_harness.predict")
    from lem.model.efference import apply_gaze_copy, head_motion_of
    preds = apply_gaze_copy(heads, preds, last, enc_act, mom.ach)
    out = dict(preds={m: v.detach().float().reshape(1, -1) for m, v in preds.items()}, last=last.detach(),
               z_motor=z.reshape(1, -1).detach(), enc_act=enc_act)
    th = getattr(model, "tenure_heads", None)
    if th is not None and "pose" in th:
        # the tenure lesson stores the order with its GAZE ROW NEUTRALISED (identity rotation,
        # zero strength; lem_model review M3b: the gaze is rewritten every step, so it is not
        # held for the tenure) - the head is read on the order the way it was trained
        gi = int(getattr(model, "GAZE_IDX", -1))
        last_t = last
        if 0 <= gi < act.shape[1]:
            neutral = act.clone()
            neutral[:, gi] = 0.0
            neutral[:, gi, 6] = 1.0
            if not torch.equal(neutral, act):
                mi = torch.cat([mom.ach[..., :MOTOR_ACHIEVED_F], command_for_encoder(neutral)], dim=-1)
                zt = model.encoder_registry.encode({"motor": mi}, {"motor": "real"})["motor"].reshape(1, 1, model.d_model)
                _l2, last_t, _c2 = model.forward_model(dict(tokens, motor=zt), mom.ctx, **kw)
        out["tenure_cm"] = (th["pose"](last_t.reshape(1, -1)).reshape(-1, 3) * 200.0).float().numpy()
        out["tenure_hidden"] = tenure_liveness(th["pose"], last_t)
    eff = getattr(model, "efference", None)
    if want_eff and eff is not None:
        if hasattr(model, "predict_efference"):
            # the model's own entry point (lem_model.predict_efference, commit 2adadb74e): the
            # same latent route as above, kept in one place by the model that trains the heads
            from lem.model.predictive_state import PredictiveState
            state = PredictiveState(tokens={m: v.reshape(1, -1) for m, v in mom.tokens.items()},
                                    context=mom.ctx, achieved=mom.ach, domain="real", memory_bias=mom.mem,
                                    head_motion=head_motion_of(model, getattr(model, "_nf_raw", None)))
            body_mlp = getattr(eff, "body_mlp", None)
            with hidden_preactivations(body_mlp if isinstance(body_mlp, torch.nn.Module) else None) as pre:
                lived = model.predict_efference(act.numpy(), state=state)
            out["eff"] = {k: v.detach().float().numpy()[0] for k, v in lived.items() if torch.is_tensor(v)}
            # the body head's hidden inputs on this prediction (DESIGN section 10: a unit is live
            # when its input rises above -3); kept as the per-unit maximum, never written out
            out["eff_hidden_max"] = {k: torch.cat(v).max(dim=0).values for k, v in pre.items() if v}
        else:
            body = eff.body(last, enc_act)
            try:
                gaze = eff.gaze(last, enc_act, gaze_head=getattr(model, "gaze_head", None))
            except RuntimeError:
                gaze = None
            lived = eff.to_lived(body=body, gaze=gaze)
            out["eff"] = {k: v.detach().float().numpy()[0] for k, v in lived.items()}
    return out


# A GELU unit whose input sits below -3 passes less than 1.2% of a change in that input (the GELU
# slope is -0.012 at -3 and shrinks toward 0 below it) and outputs less than 0.004 in size.
GELU_LIVE_INPUT = -3.0
# DESIGN section 10: a layer with fewer than half its units live is reported as not responding
LIVE_SHARE_LINE = 0.5


def gelu_slope(x: float) -> float:
    """How far a GELU unit's output moves for a change in its input, at input x, as a share of that
    change: d/dx [x Phi(x)] = Phi(x) + x phi(x). About 1 for a large input (the output follows the
    input fully), 0 far below zero (the output never moves); the lesson's correction reaches the
    unit's weights scaled by the same number. Phi from erfc, which keeps its digits far below zero."""
    x = float(x)
    Phi = 0.5 * math.erfc(-x / math.sqrt(2.0))
    phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return Phi + x * phi


# THE LIVENESS LINE AS A PERSON READS IT (fix-readout-explainer 2026-09-23). A raw input value such as
# "-3" or "-7" is a number no reader can picture (DESIGN section 11; the extended page scanner flags
# it). What the line means: below it GELU's output moves by less than LIVE_FOLLOW of a change in the
# unit's input (1.2%), and below it the slope only shrinks, so "input above the line" and "output
# follows at least 1.2% of a change" are the same test there.
LIVE_FOLLOW = abs(gelu_slope(GELU_LIVE_INPUT))
LIVE_RULE = ("A unit responds when its input leaves the flat low end of its activation function (GELU, "
             "the curve that turns a unit's input into its output). In that flat end, a change in the "
             "input moves the output by less than %s of the change; for a large input the output moves "
             "as much as the input (100%%).")


def follow_text(x: float) -> str:
    """How far a GELU unit at input x moves its output for a change in its input, as a count a
    person can picture ('6 in every hundred billion parts') or a percentage from 1% up."""
    s = abs(gelu_slope(x))
    if s < 5e-13:
        return "less than 1 in every trillion parts"
    p = parts_in(s)
    return (p + " parts") if p else pct(s, 2)


def live_rule_text() -> str:
    """LIVE_RULE with its share filled in: two sentences, '... less than 1.2% of the change ...'."""
    return LIVE_RULE % pct(LIVE_FOLLOW, 2)


def nearest_unit_text(input_max: float) -> str:
    """How near the unit nearest to responding came, in the same terms, as one sentence: 'Even the
    unit nearest to responding moved its output by only 3 in every hundred billion parts of a change
    in its input.', or that at least one unit left the flat end."""
    if input_max is None:
        return "Where the units' inputs sat was not recorded."
    if float(input_max) > GELU_LIVE_INPUT:
        return "At least one unit left the flat end on some order."
    # "by only less than 1 in every trillion parts" does not read (verify-fix-readout-explainer
    # 2026-09-23: every nearest input below about -7.6 printed it); "only" goes with a count alone
    follow = follow_text(input_max)
    return ("Even the unit nearest to responding moved its output by %s%s of a change in its input."
            % ("" if follow.startswith("less than") else "only ", follow))


def gelu_layers(module) -> List[tuple]:
    """(name, Linear) for every Linear layer directly followed by a GELU inside an nn.Sequential
    of `module` (its output is the GELU's input: the number the -3 line is drawn on)."""
    out = []
    if module is None:
        return out
    for name, sub in module.named_modules():
        if isinstance(sub, torch.nn.Sequential):
            kids = list(sub.named_children())
            for (n1, m1), (_n2, m2) in zip(kids, kids[1:]):
                if isinstance(m1, torch.nn.Linear) and isinstance(m2, torch.nn.GELU):
                    out.append((f"{name}.{n1}" if name else n1, m1))
    return out


@contextlib.contextmanager
def hidden_preactivations(module):
    """Collect, per Linear-before-GELU layer of `module`, the inputs its GELU received while the
    block runs: {layer: [tensor [n, units], ...]}. An absent module collects nothing."""
    got: Dict[str, list] = {}
    handles = []
    for name, lin in gelu_layers(module):
        def hook(_m, _i, o, name=name):
            got.setdefault(name, []).append(o.detach().float().reshape(-1, o.shape[-1]).cpu())
        handles.append(lin.register_forward_hook(hook))
    try:
        yield got
    finally:
        for h in handles:
            h.remove()


def live_share_of(max_inputs: Dict[str, torch.Tensor]) -> Optional[dict]:
    """{layer: share of units whose largest input exceeded GELU_LIVE_INPUT} from per-unit maxima
    gathered over the probe's predictions; None when nothing was gathered."""
    if not max_inputs:
        return None
    return {k: float((v > GELU_LIVE_INPUT).float().mean()) for k, v in max_inputs.items()}


def merge_max(acc: Dict[str, torch.Tensor], new: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    for k, v in (new or {}).items():
        acc[k] = v.clone() if k not in acc or acc[k].shape != v.shape else torch.maximum(acc[k], v)
    return acc


@torch.no_grad()
def predict_contacts(model, mom: "Moment", order_exec, patch_state) -> Optional[dict]:
    """The contact head's next-moment probabilities per skin patch for `order_exec` from this
    moment (LemLite.predict_contacts, the model's own entry point, on the same lesson context and
    memory bias as predict()): {p_contact, p_self, p_world: [53] numpy, patch_names}. None when
    the build has no contact head or the moment has no skin report."""
    if not isinstance(getattr(model, "contact_head", None), torch.nn.Module) or not hasattr(model, "predict_contacts"):
        return None
    if patch_state is None:
        return None
    from lem.model.predictive_state import PredictiveState
    state = PredictiveState(tokens={m: v.reshape(1, -1) for m, v in mom.tokens.items()}, context=mom.ctx,
                            achieved=mom.ach, domain="real", memory_bias=mom.mem)
    out = model.predict_contacts(state=state, order_cm=np.asarray(order_exec, dtype=np.float32),
                                 patch_state=np.asarray(patch_state, dtype=np.int64).reshape(1, -1))
    if not out:
        return None
    return {k: (v.detach().float().numpy()[0] if torch.is_tensor(v) else v) for k, v in out.items()}


@torch.no_grad()
def tenure_liveness(head, last) -> dict:
    """Whether the tenure pose head's hidden layer responds at all on this latent.

    Measured 2026-09-22 on ckpt_step_344400, bodies 0 and 1 of its episode: every one of the 256
    hidden units had an input between -29.1 and -6.5, so GELU returned exactly 0 and the head
    returned its output bias for any state and any order (and no gradient reached it)."""
    try:
        first = head[0]
        h = first(last.reshape(1, -1).to(first.weight.dtype))
    except Exception:
        return dict(status="unmeasured: tenure head has an unexpected shape")
    return dict(units=int(h.numel()), live_share=float((h > GELU_LIVE_INPUT).float().mean()),
                input_min=float(h.min()), input_max=float(h.max()), live_input_line=GELU_LIVE_INPUT)


@torch.no_grad()
def rollout(model, mom: Moment, order_exec, depth: int) -> List[dict]:
    """The imagination route (predictive_state.transition, the planner's and the imagination
    chain's): the same order held for `depth` moments. Returns per depth the predicted tokens
    (normalised, as transition returns them)."""
    from lem.model.predictive_state import PredictiveState, transition
    from mimo_bridge.lem_model import command_for_encoder
    act = command_for_encoder(torch.as_tensor(np.asarray(order_exec, dtype=np.float32)).reshape(1, -1, 9))
    state = PredictiveState(tokens={m: v.reshape(1, -1) for m, v in mom.tokens.items()}, context=mom.ctx,
                            achieved=mom.ach, domain="real", memory_bias=mom.mem)
    out = []
    for _ in range(int(depth)):
        state = transition(model.forward_model, model.encoder_registry, model.prediction_heads,
                           state, act, advance_context=True)
        out.append({m: v.detach().float().reshape(1, -1) for m, v in state.tokens.items()})
    return out


@torch.no_grad()
def predicted_wellness(model, mom: Moment, order_exec) -> Optional[float]:
    """The approximate wellness evaluator's sum over its five terms for the state one transition
    after `order_exec` (the planner's scorer, on the planner's route). None if absent."""
    wl = getattr(model, "wellness", None)
    if wl is None or not hasattr(wl, "score_predicted_wellness"):
        return None
    nxt = rollout_states(model, mom, order_exec, 1)[0]
    preds = {m: v for m, v in nxt.tokens.items() if m != "motor"}
    return float(wl.score_predicted_wellness(preds, nxt.tokens["motor"]).sum(-1).mean())


@torch.no_grad()
def rollout_states(model, mom: Moment, order_exec, depth: int):
    from lem.model.predictive_state import PredictiveState, transition
    from mimo_bridge.lem_model import command_for_encoder
    act = command_for_encoder(torch.as_tensor(np.asarray(order_exec, dtype=np.float32)).reshape(1, -1, 9))
    state = PredictiveState(tokens={m: v.reshape(1, -1) for m, v in mom.tokens.items()}, context=mom.ctx,
                            achieved=mom.ach, domain="real", memory_bias=mom.mem)
    states = []
    for _ in range(int(depth)):
        state = transition(model.forward_model, model.encoder_registry, model.prediction_heads,
                           state, act, advance_context=True)
        states.append(state)
    return states


# ============================================================================ scoring
def mse(a, b) -> float:
    a = torch.as_tensor(a).float().reshape(-1)
    b = torch.as_tensor(b).float().reshape(-1)
    return float(((a - b) ** 2).mean())


def l2(a) -> float:
    return float(torch.as_tensor(a).float().reshape(-1).norm())


def cosine(a, b) -> Optional[float]:
    a = torch.as_tensor(a).float().reshape(-1)
    b = torch.as_tensor(b).float().reshape(-1)
    na, nb = float(a.norm()), float(b.norm())
    if na < 1e-12 or nb < 1e-12:
        return None
    return float((a @ b) / (na * nb))


def captured_share(pred_change, actual_change) -> Optional[float]:
    """How much of the real change the prediction foresaw, along the real change's direction:
    (predicted . actual) / (actual . actual). 0 = the prediction did not move (or moved at right
    angles to reality), 1 = it foresaw the whole change, above 1 = it expected more than
    happened, negative = the wrong way. None when nothing really changed."""
    p = torch.as_tensor(pred_change).double().reshape(-1)
    a = torch.as_tensor(actual_change).double().reshape(-1)
    den = float(a @ a)
    if den <= 0.0:
        return None
    return float((p @ a) / den)


def pooled_share(rows, dot_key: str, norm_key: str) -> Optional[float]:
    """captured_share pooled over many arms: sum of (predicted . actual) over sum of
    (actual . actual). One arm whose real difference is a hundredth of a centimetre cannot
    dominate, as it does in a mean of per-arm ratios (measured in the first body smoke run,
    2026-09-22: a finger arm read -343 and its twin +316 on differences of 0.011 and 0.004 cm)."""
    num = [r.get(dot_key) for r in rows if r and r.get(dot_key) is not None and r.get(norm_key) is not None]
    den = [r.get(norm_key) for r in rows if r and r.get(dot_key) is not None and r.get(norm_key) is not None]
    total = float(np.sum(den)) if den else 0.0
    return float(np.sum(num) / total) if total > 0 else None


# Two predicted wellness values closer than this share of their size are the same number to the
# evaluator: float32 carries 24 bits (relative resolution 1.19e-7) and the score is a sum of five
# terms, so four resolution steps is the smallest difference a comparison can trust.
_TIE_REL = 4 * float(np.finfo(np.float32).eps)


def better_of(name_a: str, a: float, name_b: str, b: float) -> str:
    """The name of the larger, or "tie" when float32 cannot tell them apart."""
    if abs(a - b) <= _TIE_REL * max(1.0, abs(a), abs(b)):
        return "tie"
    return name_a if a > b else name_b


def sig(value, figures: int = 3) -> Optional[float]:
    """`value` rounded to `figures` significant figures (None and non-finite pass as None)."""
    if value is None:
        return None
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    if v == 0.0:
        return 0.0
    return round(v, figures - 1 - int(math.floor(math.log10(abs(v)))))


def pct(value, figures: int = 3) -> str:
    """A share (1 = all of it) as a percentage with `figures` significant figures, written out
    in full, never in scientific notation (DESIGN section 11; review of the probes, 2026-09-23:
    "7.3e-06%" was unreadable). "unmeasured" for None."""
    if value is None:
        return "unmeasured"
    v = 100.0 * float(value)
    if v != v:
        return "unmeasured"
    if v == 0.0:
        return "0%"
    d = max(0, int(figures) - 1 - int(math.floor(math.log10(abs(v)))))
    s = "%.*f" % (d, round(v, d))
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s + "%"


_POWERS = {3: "thousand", 4: "ten thousand", 5: "hundred thousand", 6: "million", 7: "ten million",
           8: "hundred million", 9: "billion", 10: "ten billion", 11: "hundred billion", 12: "trillion"}


def parts_in(value) -> Optional[str]:
    """A share under one in a hundred as a count a person can picture: 7.3e-8 -> "7 in every
    hundred million". None for a share of one in a hundred or more, or none at all."""
    if value is None:
        return None
    v = abs(float(value))
    if v == 0.0 or v != v or v >= 0.01:
        return None
    k = int(math.ceil(-math.log10(v)))
    n = v * 10 ** k
    if n >= 9.5:                       # rounding up to 10 moves to the next power
        k -= 1
        n = v * 10 ** k
    k = min(max(k, 3), 12)
    return "%d in every %s" % (max(1, int(round(v * 10 ** k))), _POWERS[k])


def share_text(value) -> str:
    """pct(value), with the picture of a tiny share and the meaning of a negative one."""
    s = pct(value)
    if value is None or value != value:
        return s
    extra = []
    p = parts_in(value)
    if p:
        extra.append(p)
    if value < 0:
        extra.append("minus: the prediction moved the wrong way")
    return s + (" (%s)" % "; ".join(extra) if extra else "")


def pairing_correct(pred_a, pred_b, act_a, act_b) -> Optional[bool]:
    """Two-way discrimination: does each prediction sit closer to its own outcome than to the
    other arm's? None when the two outcomes are identical (nothing to tell apart)."""
    if mse(act_a, act_b) <= 0.0:
        return None
    own = mse(pred_a, act_a) + mse(pred_b, act_b)
    swapped = mse(pred_a, act_b) + mse(pred_b, act_a)
    if abs(own - swapped) <= _TIE_REL * max(own, swapped, 1e-30):
        return None                # identical predictions: no decision was made, not a wrong one
    return bool(own < swapped)


@torch.no_grad()
def command_pathway(model, mom: "Moment", orders: Dict[str, Any]) -> dict:
    """How far each order moves the motor token, against the token's own size, and how the
    motor encoder's first layer weighs the order's nine command columns against the ten
    achieved-state columns. This is the entry point of every action effect: an order that does
    not move this token cannot change any prediction."""
    from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, MOTOR_INPUT_F, command_for_encoder
    toks = {}
    for name, o in orders.items():
        act = torch.as_tensor(np.asarray(o, dtype=np.float32)).reshape(1, -1, 9)
        mi = torch.cat([mom.ach[..., :MOTOR_ACHIEVED_F], command_for_encoder(act)], dim=-1)
        toks[name] = model.encoder_registry.encode({"motor": mi}, {"motor": "real"})["motor"].reshape(-1).float()
    names = list(toks)
    ref = names[0]
    size = l2(toks[ref])
    diffs = {n: (l2(toks[n] - toks[ref]) / size if size > 0 else None) for n in names[1:]}
    weights = None
    try:
        W = model.encoder_registry.encoders["motor"].patch_mlp[0].weight
        if W.shape[1] >= MOTOR_INPUT_F:
            weights = dict(command_columns=float(W[:, MOTOR_ACHIEVED_F:MOTOR_INPUT_F].abs().mean()),
                           achieved_columns=float(W[:, :MOTOR_ACHIEVED_F].abs().mean()))
    except Exception:
        weights = None
    return dict(reference=ref, token_change_share=diffs, encoder_first_layer_mean_abs_weight=weights,
                note="token_change_share = how far the order moves the motor token, as a share of the token's "
                     "size (0 = not at all)")


def body_change(r0: dict, r1: dict) -> dict:
    """How far every handle moved (cm, world x y z) and turned (degrees) between two readouts."""
    cm = np.asarray(r1["pos_cm"]) - np.asarray(r0["pos_cm"])
    deg = np.array([rotation_deg_xyzw(a, b) for a, b in zip(r0["quat"], r1["quat"])])
    return dict(cm=cm, cm_norm=np.linalg.norm(cm, axis=-1), deg=deg)


def token_scores(pred: Dict[str, torch.Tensor], actual: Dict[str, torch.Tensor],
                 current: Dict[str, torch.Tensor], senses) -> Dict[str, dict]:
    """Per sense: the prediction's squared error, the "nothing changes" error (persistence),
    and improvement_over_persistence = 1 - error/persistence (0 = no better than assuming the
    world stays as it is, 1 = exact, negative = worse than assuming nothing changes)."""
    out = {}
    for m in senses:
        if m not in pred or m not in actual or m not in current:
            continue
        e, p = mse(pred[m], actual[m]), mse(current[m], actual[m])
        out[m] = dict(pred_err=e, persist_err=p, improvement=(1.0 - e / p) if p > 0 else None,
                      actual_change=l2(actual[m].reshape(-1) - current[m].reshape(-1)),
                      predicted_change=l2(pred[m].reshape(-1) - current[m].reshape(-1)))
    return out


def real_senses(model) -> List[str]:
    """The senses whose prediction is scored: predicted by a head and fed by the world (the
    audio placeholder and the intermind placeholder are generated noise, never scored)."""
    heads = set(getattr(model.prediction_heads, "head_modalities", ()) or ())
    mods = model.config.get("modalities", {}) if hasattr(model, "config") else {}
    out = []
    for m in model.active:
        if m not in heads or m == "motor":
            continue
        if str((mods.get(m) or {}).get("data_source", "real")) != "real":
            continue
        out.append(m)
    return out


def mean_or_none(values) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and v == v]
    return float(np.mean(vals)) if vals else None


# Why every probe's goal line is null (review of the probes, 2026-09-23): a share of 1 means
# foreseeing the WHOLE real change, including what no model can foresee (world noise, the
# caregiver), so it is the scale's end, not what working control reads. The goal is filled in only
# when a working-control reading exists for that probe - for example the share a small reference
# predictor fitted on this world's recorded order-and-outcome pairs reaches from other moments.
GOAL_UNMEASURED = ("distance to goal unmeasured: no working-control reading exists for this probe yet "
                   "(100% is the scale's end, foreseeing even what no model can)")


def headline(metric: str, value, unit: str, floor, goal, lived: str, *, higher_is_better: bool = True,
             goal_note: Optional[str] = None) -> dict:
    """DESIGN.md section 9 headline. distance_to_goal is the distance from value to goal (in the
    metric's own unit, 3 significant figures) when both are measured, else None with goal_note
    saying why ("distance to goal unmeasured"). `display` is the value as a percentage with 3
    significant figures (the headline values are shares, 1 = all of the real change)."""
    dist = None
    if isinstance(value, (int, float)) and isinstance(goal, (int, float)) and value == value:
        dist = float(goal - value) if higher_is_better else float(value - goal)
        dist = sig(max(0.0, dist))
    out = dict(metric=metric, value=value, unit=unit, floor=floor, goal=goal, distance_to_goal=dist,
               lived=lived, display=None if value is None else pct(value))
    if goal is None:
        out["goal_note"] = goal_note or GOAL_UNMEASURED
    return out


# ============================================================================ output
def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_result(info: dict, probe: str, item: int, body: dict, *, root: Optional[Path] = None) -> Path:
    """Write <root>/<run>/<ckpt>/<probe>.json and append its index line."""
    root = PROBES_ROOT if root is None else Path(root)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    code = code_commit()
    record = dict(probe=probe, item=int(item), run=info["run"], checkpoint=info["checkpoint"],
                  lineage_step=info["lineage_step"], time=now, harness=HARNESS_VERSION,
                  harness_version=HARNESS_VERSION, code_commit=code.get("commit"), code=code)
    record.update(body)
    path = root / info["run"] / info["name"] / f"{probe}.json"
    _atomic_json(path, record)
    line = dict(probe=probe, item=int(item), run=info["run"], checkpoint=info["checkpoint"],
                lineage_step=info["lineage_step"], time=now, path=str(path),
                headline=record.get("headline"), status=record.get("status"),
                harness_version=HARNESS_VERSION, code_commit=code.get("commit"),
                code_uncommitted_tracked_files=code.get("uncommitted_tracked_files"))
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "index.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(line, default=_json_default) + "\n")
    try:
        refresh_suite(root, info, probe, record, path)
    except (OSError, TypeError, ValueError) as error:
        # the result file and its index line are the contract (DESIGN section 9); the summary
        # failing to update must not turn a measured probe into a failure record
        print(f"gof_probe_harness: suite.json not refreshed for {probe}: {type(error).__name__}: {error}",
              flush=True)
    return path


def refresh_suite(root: Path, info: dict, probe: str, record: dict, path: Path) -> Path:
    """Keep <run>/<ckpt>/suite.json's entry for `probe` on its newest result, whether the probe ran
    inside the suite or on its own (2026-09-23: the v2 gaze and body results were written by the
    probes run one at a time, and suite.json kept listing the v1 headlines beside them). The
    suite's own write at its end keeps these entries and adds its timing and memory."""
    sp = Path(root) / info["run"] / info["name"] / "suite.json"
    try:
        earlier = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    except (OSError, ValueError):
        earlier = {}
    probes = dict(earlier.get("probes") or {})
    probes[probe] = dict(status=record.get("status"), headline=record.get("headline"), path=str(path),
                         elapsed_s=record.get("elapsed_s"), time=record.get("time"),
                         harness_version=record.get("harness_version"), code_commit=record.get("code_commit"))
    _atomic_json(sp, dict(earlier, checkpoint=info["checkpoint"], run=info["run"],
                          lineage_step=info["lineage_step"], probes=probes))
    return sp


def read_index(root: Optional[Path] = None) -> List[dict]:
    p = (PROBES_ROOT if root is None else Path(root)) / "index.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def latest_index(root: Optional[Path] = None) -> List[dict]:
    """The index keeps every run (append-only); a probe re-run on the same checkpoint adds a
    newer line. This returns the newest line per (checkpoint, probe), in index order."""
    newest = {}
    for i, r in enumerate(read_index(root)):
        newest[(os.path.normcase(os.path.abspath(r.get("checkpoint", ""))), r.get("probe"))] = (i, r)
    return [r for _, r in sorted(newest.values(), key=lambda x: x[0])]


def indexed(checkpoint: str, probe: str, root: Optional[Path] = None) -> bool:
    want = os.path.normcase(os.path.abspath(checkpoint))
    return any(os.path.normcase(os.path.abspath(r.get("checkpoint", ""))) == want and r.get("probe") == probe
               for r in read_index(root))


def newest_record(checkpoint: str, probe: str, root: Optional[Path] = None) -> Optional[dict]:
    want = os.path.normcase(os.path.abspath(checkpoint))
    rows = [r for r in read_index(root)
            if os.path.normcase(os.path.abspath(r.get("checkpoint", ""))) == want and r.get("probe") == probe]
    return rows[-1] if rows else None


def is_failure(record: Optional[dict]) -> bool:
    return bool(record) and str(record.get("status") or "").startswith("failed")


# How many times one probe is run on one checkpoint by the SAME code (commit and harness version)
# before its failure stands until other code is committed. Two = one retry: the failures the
# review of the probes named as transient (a memory shortage, a scene-file race) need not repeat,
# and a checkpoint that fails every time costs one more run, never a loop (verify-fix-probes
# 2026-09-23: a same-code failure was never retried, so a transient one ended that checkpoint's
# measurement until someone committed other code).
SAME_CODE_ATTEMPTS = 2


def _same_code(rec: dict) -> bool:
    return (rec.get("harness_version") == HARNESS_VERSION and rec.get("code_commit") is not None
            and rec.get("code_commit") == code_commit().get("commit"))


def needs_run(checkpoint: str, probe: str, root: Optional[Path] = None) -> bool:
    """Whether a probe must (re)run on this checkpoint: it has no record; or its newest record is
    a failure written by other code (an older commit or harness version); or the newest records are
    failures of THIS code fewer than SAME_CODE_ATTEMPTS times in a row. A checkpoint that always
    fails on this code is run SAME_CODE_ATTEMPTS times, then waits for other code (review of the
    probes, 2026-09-23: a failure record used to count as measured for good)."""
    want = os.path.normcase(os.path.abspath(checkpoint))
    rows = [r for r in read_index(root)
            if os.path.normcase(os.path.abspath(r.get("checkpoint", ""))) == want and r.get("probe") == probe]
    if not rows:
        return True
    if not is_failure(rows[-1]):
        # Prediction and state-restoration contracts can change even for a probe
        # that previously succeeded. A result from an older harness is historical.
        return rows[-1].get("harness_version") != HARNESS_VERSION
    if not _same_code(rows[-1]):
        return True
    streak = 0
    for r in reversed(rows):
        if not (is_failure(r) and _same_code(r)):
            break
        streak += 1
    return streak < SAME_CODE_ATTEMPTS


def frozen_config(frozen: Frozen, session: Optional[Session] = None) -> dict:
    """What every probe records about how it was run."""
    cfg = dict(harness=HARNESS_VERSION, limits=frozen.limits, limits_source=frozen.limits_source,
               args_json=frozen.info["args_path"], build_s=round(frozen.build_s, 2),
               load_s=round(frozen.load_s, 2), reinitialized=frozen.load["reinitialized"],
               intention=getattr(frozen.args, "intention", None),
               components=components(frozen.model), device="cpu",
               threads=torch.get_num_threads(), code=code_commit())
    if session is not None:
        cfg["world"] = {k: v for k, v in session.kw.items() if k not in ("run_dir",)}
        cfg["model_step_s"] = session.clock.model_step_seconds
        cfg["order_tenure_s"] = session.clock.held_order_seconds
        cfg["start"] = getattr(session, "start", None)
        cfg["start_note"] = getattr(session, "start_note", None)
    return cfg


def earlier_checkpoint(info: dict, max_hops: int = 60) -> dict:
    """The oldest ancestor of this checkpoint (following lineage.json 'resumed_from') that still
    exists here or where a storage move put it. Returns {checkpoint, step, hops, chain} or
    {checkpoint: None, reason}. The caller still has to load it (it may refuse)."""
    chain, cur, hops = [], info.get("resumed_from"), 0
    while cur and hops < max_hops:
        p = _locate(cur)
        exists = (p / "model.safetensors").exists()
        chain.append(dict(path=str(cur), found=str(p) if exists else None))
        if not exists:
            break
        try:
            lin = json.loads((p / "lineage.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lin = {}
        cur = lin.get("resumed_from")
        hops += 1
    found = [c for c in chain if c["found"]]
    if not found:
        return dict(checkpoint=None, reason="no ancestor checkpoint found on disk (lineage.json resumed_from "
                                            "chain: %d links, none present)" % len(chain), chain=chain)
    return dict(checkpoint=found[-1]["found"], hops=len(found), chain=chain,
                candidates=[c["found"] for c in reversed(found)])


def ancestors(info: dict, max_hops: int = 60) -> List[str]:
    """Existing ancestor checkpoints, oldest first."""
    e = earlier_checkpoint(info, max_hops)
    return list(e.get("candidates") or [])


def main(argv=None):
    """Smoke: load a checkpoint, build its world, warm, snapshot, and time the pieces."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    t0 = time.time()
    fz = load_frozen(a.ckpt, threads=a.threads)
    s = Session(fz, seed=1)
    try:
        t1 = time.time()
        s.warm(a.warm)
        t2 = time.time()
        s.observe()
        snap = s.snapshot()
        t3 = time.time()
        s.restore(snap)
        t4 = time.time()
        mom = s.moment()
        pr = predict(fz.model, mom, s.hold_order())
        t5 = time.time()
        print(json.dumps(dict(load_s=round(t1 - t0, 2), warm_step_s=round((t2 - t1) / max(1, a.warm), 3),
                              snapshot_s=round(t3 - t2, 3), restore_s=round(t4 - t3, 3),
                              predict_s=round(t5 - t4, 3), context_source=mom.context_source,
                              components=components(fz.model),
                              handles=s.handle_names, tenure=("tenure_cm" in pr)), indent=1))
    finally:
        s.close()


if __name__ == "__main__":
    main()
