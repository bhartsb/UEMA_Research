# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Chance rates: what each detector finds when no learner is choosing anything.

Addendum sections 25 and 27: a capability is observed only "above a random/baseline threshold",
and false-positive rates come from long motor-babble-only runs. This module runs the SAME world
as a training run (its args.json, through train_mimo.env_kwargs_for), with the body driven by one
of three chance processes instead of a learner, records it with the same world recorder, and
digests it with the same analysis:

  random_orders   the body's own controller given random orders: every handle gets a uniformly
                  random target orientation, a position target within ORDER_REACH_CM of where it
                  is (the hand controller needs a finite one) and a random strength between 0 and
                  ORDER_STRENGTH, held for the run's motor hold (5 steps = 250 ms on H84).
                  The chance process the tracker judges stages against: motor babble through
                  the same decoder the learner's orders pass.
  random_ctrl     uniformly random actuator commands in [-1, 1], held the same way: flailing
                  well beyond what any learned order produces (reported, not judged against)
  limp            no command at all: what pure physics and the world's own movers (the saccade
                  generator, the caregiver, critters) produce
  yoked           (2026-09-25) the LEARNER'S OWN ORDERS replayed in shuffled time order: each body
                  gets the orders production's body of the same number was handed (the world
                  recorder's v2 'orders' field group), one whole order at a time (its motor hold and
                  the eye command of each of its steps), in a random permutation, from saved worlds.
                  It keeps what the learner asks for - how far, which parts, how strongly, how often
                  - and breaks when and from where it asks it, AND the order-to-order sequence:
                  production's consecutive orders move each hand back the way it came (a mean cosine
                  of -0.18 to -0.19 between consecutive displacements, where a shuffle makes 0; the
                  independent review of 2026-09-25, finding 6), so beating it means "chosen for the
                  moment OR produced by the sequence". Reported, never what stages are judged against
                  (the operator's choice) (yoke_units, replay_orders)
  yoked_sequence  (2026-09-25 evening) the same orders in their RECORDED ORDER, re-anchored the same
                  way, starting at a random order of the body's history (another moment than the
                  saved world's) and running on from there: it keeps the sequence and breaks only when
                  and from where. Compared with yoked it tells the sequence's part from the moment's.
                  Reported beside yoked

VISION AS IN THE RUN (2026-09-24). The worlds render the eyes' images exactly when training does:
with the run's own vision setting (on for H84), once per burst, at its last frame. The eyes'
motion reflex runs on those images, so babble's gaze is judged against the same reflex production
has. The first baselines (2026-09-24 morning) forced vision off: no reflex in babble while
production had it, one of the review's findings (OBSERVER_DESIGN.md section 9); a baseline
recorded with the other vision setting is moved aside as <kind>_vision_<on|off>, never mixed.
`--vision off` still records the old way, for comparison. Output:
<Saved>/mimo_bridge/observer/baselines/<world signature>/<kind>.json with, per capability and
sub-row, the count, the body-steps observed, the rate per step and the main measure's 5th, 50th
and 95th percentiles, and for the novelty miner how many babble episodes contain each token run;
the recordings stay beside it, so a detector change re-digests them without simulating again
(resummarize). A summary names the detector version it was digested with; the report ignores a
summary of another version (daemon.load_baseline), because chance rates of a different detector
are no control.

FROM PRODUCTION'S SAVED WORLDS (2026-09-24 evening; OBSERVER_DESIGN.md section 10, priority 4).
A world reset starts every body at the crib's centre in the reset pose, with its actuators at full
force and the caregiver opening its first scene. Production is judged in its own worlds, which
differ in all three (checkpoint 396,050, read from its saved state): the eight bodies lay 0.59-0.65
m from the crib's centre, where the reset-world babble's bodies got no farther than 0.26-0.37 m in
their 2,500 steps; each body's actuators gave a median of 33-41% of their full force (motor
fatigue under body function 'modulate' scales every actuator's gain); and the caregiver was away
from four bodies and engaged with four, all three of its objects shown across the eight, 19,803 s
into their lives.
`--from-checkpoint <run>/checkpoints/ckpt_step_N` records babble from that checkpoint's
saved eight-world episode: loaded as train_mimo loads it on resume (continuation_state.load: the
companion found through the storage move records and past 260 characters, hash-checked), body i
built from production body i's saved settings and restored to its saved state
(BridgeEnv.restore_full_state), checked field by field before its first order (restore_receipt).
Two things differ from production's world, both outputs: the world writes its scene files into the
babble's scratch folder, and the caregiver's event tape, which would append to production's
caregiver recordings, is replaced by a counter (DiscardedEvents). Babble from saved worlds
REPLACES the reset-world babble as the comparison (<kind>.json, the file the tracker reads through
daemon.load_baseline): it is the chance process in the conditions production's acts are counted
in. The reset-world recording moves aside to <kind>_reset_world, as vision_off was kept, and a
reset-world recording made later goes beside the saved-world comparison instead of replacing it
(place). Every summary names where its bodies started: start, source, and for saved worlds the
checkpoint and its step.

RE-RECORDED FROM EACH NEWER CHECKPOINT (2026-09-25; OBSERVER_DESIGN.md section 9): production's
worlds keep changing (places, fatigue, the caregiver's phase), so a saved-world recording is made
again from the newest checkpoint. The one it replaces is never deleted: it moves aside as
<kind>_ckpt_<step> (place), and `baseline --compare SIG --kind K --other K_ckpt_<step>` tests every
capability's rate in the two. A summary names the recorder's field groups its recording carries
(recorded_fields) and every capability it cannot measure for want of one (unmeasured_caps, left
out of caps, so the tracker calls the stage unmeasured instead of judging it against a zero).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

KINDS = ("random_orders", "random_ctrl", "limp", "yoked", "yoked_sequence")
YOKED_KINDS = ("yoked", "yoked_sequence")          # the learner's own orders replayed (replay_orders)
ORDER_STRENGTH = 0.3
ORDER_REACH_CM = 3.0
# the authors whose orders are the learner's own (detectors.MODEL_SELECTED and SPONTANEOUS): the yoke
# replays these, never a scheduled rest's hold, a probe, a script or another baseline
LEARNER_AUTHORS = ("planner", "self_choice", "acting_snapshot", "reflex", "reflex_noise")
# where a recording's bodies start: a reset of the run's world, or production's saved worlds
RESET_WORLD = "reset_world"
SAVED_WORLDS = "saved_worlds"
# what a recording WAS, which digesting it again (resummarize) does not change
SOURCE_FIELDS = ("start", "source", "source_run", "checkpoint", "checkpoint_step", "companion",
                 "companion_sha256", "saved_sim_time_s", "saved_caregiver_phase", "yoke")


def world_kwargs_like(run_dir: str, scratch: str, seed: int, vision: str = "like") -> dict:
    """The run's world (its args.json through train_mimo.env_kwargs_for); vision 'like' keeps the
    run's own setting, 'on'/'off' force it."""
    from mimo_bridge import train_mimo as T
    args = T.parse_args(["--out", scratch])
    with open(os.path.join(run_dir, "args.json"), encoding="utf-8") as f:
        a = json.load(f)
    for k, v in a.items():
        if hasattr(args, k):
            setattr(args, k, v)
    kw = dict(T.env_kwargs_for(args, None, scratch))
    kw.update(caregiver_record_dir="", run_dir=os.path.join(scratch, "scene"), seed=int(seed))
    if vision in ("on", "off"):
        kw["vision"] = vision == "on"
    return kw, int(getattr(args, "action_repeat", 5)), int(getattr(args, "motor_hold", 5))


class DiscardedEvents:
    """The caregiver's event record inside a babble world: counts what the caregiver appends and
    keeps nothing.

    A saved production world carries the caregiver's event tape (mimo_bridge/event_tape.py), which
    appends chunk files to production's caregiver recording folder and rewrites a status file in
    production's run folder (checkpoint 396,050, body 0: caregiver_records/H84_fresh_drivers_20260918
    and experiments/H84_fresh_drivers_20260918/scenes/caregiver_storage_12204.json). Restored as
    the trainer restores it, babble would add its own caregiver's events to production's
    recordings. Nothing reads the events back while a world runs: the caregiver appends, and the
    world reports how many (BridgeEnv._collect)."""

    def __init__(self, count: int = 0):
        self.count = int(count)

    def append(self, event) -> None:
        self.count += 1

    def __len__(self) -> int:
        return self.count


def detach_recording(world: dict) -> dict:
    """A saved world snapshot (BridgeEnv.save_full_state) whose caregiver records nothing: its event
    tape replaced by DiscardedEvents counting on from the tape's length. Shallow: every other entry
    is the saved one, and the snapshot given is not changed."""
    cg = world.get("caregiver") if isinstance(world, dict) else None
    if not isinstance(cg, dict) or "events" not in cg or isinstance(cg["events"], DiscardedEvents):
        return world
    events = cg["events"]
    count = len(events) if hasattr(events, "__len__") else 0
    return dict(world, caregiver=dict(cg, events=DiscardedEvents(count)))


def babble_world_kwargs(env_kwargs: dict, scratch: str) -> dict:
    """A saved world's settings for its babble copy: every one as saved (DESIGN D2: a resumed world
    keeps its saved settings), except the two folders it writes to. The scene files go to the
    babble's scratch folder (the saved value is production's scene folder) and the caregiver
    records nothing (the saved value is production's caregiver recording folder)."""
    return dict(env_kwargs, run_dir=os.path.join(scratch, "scene"), caregiver_record_dir="")


def _max_abs(a, b) -> Optional[float]:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def restore_receipt(env, snap: dict) -> dict:
    """Whether a restored world is in its saved state, compared exactly: simulated time, frame,
    every joint's position and velocity, the moved (mocap) bodies, actuator states, MuJoCo's whole
    integration state (positions, velocities, actuator states, controls, the solver's warm start),
    the model fields the world edits while it runs (the objects' visibility and collision, the
    mobile's mount, the fatigue-scaled actuator gains), and the caregiver's scene phase, shown
    object and clock. Returns every reading, `differs` (the names not equal) and `ok`."""
    d, m = env.data, env.model
    differs: List[str] = []

    def same(name, a, b) -> bool:
        a, b = np.asarray(a), np.asarray(b)
        eq = a.shape == b.shape and bool(np.array_equal(a, b))
        if not eq:
            differs.append(name)
        return eq

    out = dict(sim_time=float(d.time), saved_sim_time=float(snap["time"]),
               frame=int(getattr(env, "frame", -1)), saved_frame=snap.get("frame"),
               qpos_max_abs=_max_abs(d.qpos, snap["qpos"]), qvel_max_abs=_max_abs(d.qvel, snap["qvel"]))
    if float(d.time) != float(snap["time"]):
        differs.append("time")
    if snap.get("frame") is not None and int(getattr(env, "frame", -1)) != int(snap["frame"]):
        differs.append("frame")
    for name in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
        if snap.get(name) is not None:
            same(name, getattr(d, name), snap[name])
    if snap.get("act") is not None:
        same("act", d.act, snap["act"])
    mj = getattr(env, "_mj", None)
    if snap.get("integration_state") is not None and mj is not None:
        spec = mj.mjtState.mjSTATE_INTEGRATION
        now = np.zeros(mj.mj_stateSize(m, spec))
        mj.mj_getState(m, d, now, spec)
        out["integration_state_equal"] = same("integration_state", now, snap["integration_state"])
    compared, migrated = [], []
    for name, value in (snap.get("mutable_geometry") or {}).items():
        current = np.asarray(getattr(m, name))
        if current.shape != np.asarray(value).shape:
            # a snapshot saved before the adult's current shapes: restore_full_state migrates them
            # (caregiver_visual.upgrade_legacy_snapshot), exactly as the trainer's resume does
            migrated.append(name)
            continue
        same(name, current, value)
        compared.append(name)
    out.update(geometry_compared=compared, geometry_migrated=migrated)
    cg, saved = getattr(env, "caregiver", None), snap.get("caregiver")
    if cg is not None and isinstance(saved, dict):
        state = cg.state
        out.update(caregiver_phase=state.get("scene_phase"), saved_caregiver_phase=saved.get("scene_phase"),
                   caregiver_object=state.get("style"))
        for name, key in (("caregiver_phase", "scene_phase"), ("caregiver_object", "style"),
                          ("caregiver_time", "time")):
            if state.get(key) != saved.get(key):
                differs.append(name)
    out["differs"] = differs
    out["ok"] = not differs
    return out


def extract_saved_worlds(checkpoint: str, out_dir: str) -> dict:
    """Production's saved eight-world episode, written one file per body for the babble workers.

    Loaded exactly as train_mimo loads it on resume: continuation_state.load finds a checkpoint the
    archiver moved (storage_locations.locate), reads the companion its receipt names through
    storage_locations.long_path (the H84 companions sit at 253-268 characters, where Windows'
    limit is 260), and refuses a companion whose SHA-256 or step differs from the receipt. Only
    the worlds are kept, each as saved (settings, full world snapshot) with the caregiver's event
    tape detached (DiscardedEvents). The companion also holds the learner's state (820 MB unpacked
    at step 396,050), so run() calls this in a process of its own that ends before the babble
    worlds are built. Returns the manifest, also written as saved_worlds.json."""
    from pathlib import Path
    from mimo_bridge import continuation_state as CS
    from mimo_bridge.storage_locations import locate, long_path, resolve_run_args
    t0 = time.time()
    ck = locate(Path(checkpoint), verify=False)
    with open(long_path(ck / "episode_state.json"), encoding="utf-8") as f:
        receipt = json.load(f)
    state = CS.load(str(ck))
    if state is None:
        raise ValueError("%s holds no saved episode (no episode_state.json)" % ck)
    companion = locate(Path(receipt["storage_path"]), expected_sha256=receipt.get("sha256"), verify=False)
    step = int(state["trainer"]["step"])
    config = dict(state.get("config") or {})
    worlds = state.pop("worlds")
    del state                        # the learner's state goes before the files are written
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, w in enumerate(worlds):
        snap = detach_recording(w["world"])
        kw = dict(w["env_kwargs"])
        cg = snap.get("caregiver") if isinstance(snap.get("caregiver"), dict) else {}
        path = os.path.join(out_dir, "b%d.pkl" % i)
        with open(path + ".part", "wb") as f:
            pickle.dump(dict(env_kwargs=kw, world=snap, body=i, step=step), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".part", path)
        rows.append(dict(body=i, file=path, sim_time=float(snap["time"]), frame=snap.get("frame"),
                         vision=bool(kw.get("vision", True)), seed=kw.get("seed"), tag=kw.get("tag"),
                         caregiver_phase=cg.get("scene_phase"), caregiver_object=cg.get("style"),
                         caregiver_events=len(cg["events"]) if cg.get("events") is not None else None))
    try:
        run_dir = str(resolve_run_args(ck).parent)
    except FileNotFoundError:
        run_dir = None
    peak = None
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_pagefile", None) or getattr(mi, "peak_wset", None)
    except Exception:
        pass
    manifest = dict(checkpoint=str(ck), step=step, companion=str(companion),
                    companion_sha256=receipt.get("sha256"), companion_bytes=receipt.get("bytes"),
                    action_repeat=config.get("action_repeat"), motor_hold=config.get("motor_hold"),
                    n_envs=config.get("n_envs"), no_vision=config.get("no_vision"), run_dir=run_dir,
                    worlds=rows, seconds=time.time() - t0,
                    peak_memory_mb=(peak / 2 ** 20) if peak else None)
    with open(os.path.join(out_dir, "saved_worlds.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, default=_plain)
    return manifest


def _plain(x):
    """json.dump's fallback: numpy numbers as Python numbers, anything else as its text."""
    return x.item() if hasattr(x, "item") else str(x)


def saved_world_plan(manifest: dict, run_dir: Optional[str], vision: str, bodies: int) -> List[dict]:
    """One job per babble body from the extracted saved worlds: body i restores production body i.

    Refused: more bodies than the checkpoint saved; a forced vision setting other than the saved
    worlds' own (a restored world keeps its saved settings, and the eyes' motion reflex exists only
    in a world that renders); a run folder whose timing (frames per model step, model steps per
    order) differs from the saved episode's."""
    worlds = list(manifest["worlds"])
    if not 1 <= int(bodies) <= len(worlds):
        raise ValueError("checkpoint %s saved %d worlds; %d bodies asked for"
                         % (manifest["checkpoint"], len(worlds), bodies))
    worlds = worlds[:int(bodies)]
    if vision in ("on", "off"):
        for w in worlds:
            if bool(w["vision"]) != (vision == "on"):
                raise ValueError("saved world %d renders the eyes' images %s; a restored world keeps its "
                                 "saved settings, so --vision %s cannot apply (use --vision like)"
                                 % (w["body"], "on" if w["vision"] else "off", vision))
    reps, hold = manifest.get("action_repeat"), manifest.get("motor_hold")
    if run_dir:
        with open(os.path.join(run_dir, "args.json"), encoding="utf-8") as f:
            a = json.load(f)
        mine = (a.get("action_repeat", reps), a.get("motor_hold", hold))
        if reps is None or hold is None:
            reps, hold = mine
        elif tuple(int(x) for x in mine) != (int(reps), int(hold)):
            raise ValueError("run %s steps its bodies with action_repeat %s and motor_hold %s, the saved "
                             "episode of %s with %s and %s: babble would not match the run it is judged "
                             "against" % (run_dir, mine[0], mine[1], manifest["checkpoint"], reps, hold))
    if reps is None or hold is None:
        raise ValueError("the saved episode of %s names no action_repeat / motor_hold and no run folder "
                         "was given (--like)" % manifest["checkpoint"])
    return [dict(file=w["file"], body=int(w["body"]), reps=int(reps), hold=int(hold),
                 checkpoint=manifest["checkpoint"], step=int(manifest["step"])) for w in worlds]


def source_fields(manifest: Optional[dict] = None, bodies: Optional[int] = None,
                  run_dir: Optional[str] = None) -> dict:
    """Where a recording's bodies started, for its summary (SOURCE_FIELDS)."""
    run = os.path.basename(os.path.abspath(run_dir)) if run_dir else None
    if not manifest:
        return dict(start=RESET_WORLD, source_run=run,
                    source="from a world reset: the run's world built fresh from its args.json, every "
                           "body starting from the world's reset state")
    worlds = manifest["worlds"][:int(bodies)] if bodies else manifest["worlds"]
    return dict(start=SAVED_WORLDS, source_run=run,
                source=("from saved worlds: production checkpoint %s at step %d; body i starts from "
                        "production body i's saved state, restored as train_mimo restores it on resume"
                        % (manifest["checkpoint"], int(manifest["step"]))),
                checkpoint=manifest["checkpoint"], checkpoint_step=int(manifest["step"]),
                companion=manifest.get("companion"), companion_sha256=manifest.get("companion_sha256"),
                saved_sim_time_s=[w["sim_time"] for w in worlds],
                saved_caregiver_phase=[w.get("caregiver_phase") for w in worlds])


# ====================================================================== the yoked control
# An order's motor targets are WORLD positions (cm) and orientations (quaternion x, y, z, w) of each
# handle (env.BridgeEnv.step; env._command_response reads them so). Replayed at another moment as
# they were, they would ask each part to go where it was then: a hand that has moved 10 cm since
# would be sent 10 cm back, whatever the order asked. The yoke keeps what the order ASKED FOR - each
# handle's displacement and turn from its pose where the order began - and asks it again from the
# handle's pose where the replayed order begins. The eyes' row is kept as recorded (a head-relative
# order the dedicated eye servo reads, not a world pose), as are radius and strength, the mask, the
# hand's joint goal (joint angles, relative by nature) and each step's eye command.
def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z), row by row."""
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def _unit(q: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return q / np.linalg.norm(q, axis=-1, keepdims=True)


def relative_order(motor: np.ndarray, pos_m: np.ndarray, quat_wxyz: np.ndarray, keep_rows=()) -> dict:
    """What an order asks each handle to do from its pose where the order began: dpos_cm [H, 3] (the
    target minus the handle's position, cm) and qrel [H, 4] (the target orientation times the
    inverse of the handle's, w x y z). A missing target (NaN) stays missing; rows in keep_rows (the
    eyes' row) keep their recorded values."""
    motor = np.asarray(motor, dtype=np.float64)
    dpos = motor[:, 0:3] - 100.0 * np.asarray(pos_m, dtype=np.float64)
    here = np.asarray(quat_wxyz, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])
    target = _xyzw_to_wxyz(motor[:, 3:7])
    # a zero quaternion is the decoder's "no rotation": the world-aligned orientation
    # (action_decoder.quat_xyzw_to_mat returns the identity below a length of 1e-12)
    zero = np.linalg.norm(target, axis=-1) < 1e-12
    target[zero] = (1.0, 0.0, 0.0, 0.0)
    qrel = _qmul(_unit(target), here)
    keep = np.zeros(len(motor), dtype=bool)
    keep[list(keep_rows)] = True
    return dict(dpos_cm=dpos, qrel=qrel, tail=motor[:, 7:9].copy(), keep=keep, recorded=motor.copy())


def anchor_order(rel: dict, pos_m: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """The order asking each handle for the same displacement and turn from its current pose
    (relative_order inverted at another pose): motor [H, 9]."""
    motor = np.array(rel["recorded"], dtype=np.float64)
    motor[:, 0:3] = 100.0 * np.asarray(pos_m, dtype=np.float64) + rel["dpos_cm"]
    motor[:, 3:7] = _wxyz_to_xyzw(_unit(_qmul(rel["qrel"], np.asarray(quat_wxyz, dtype=np.float64))))
    motor[:, 7:9] = rel["tail"]
    keep = np.asarray(rel["keep"], dtype=bool)
    motor[keep] = rel["recorded"][keep]
    return motor


def replay_orders(env, rec, units: List[dict], steps: int, reps: int, rng, render: bool,
                  keep_sequence: bool = False) -> dict:
    """The yoked controls' loop: whole orders, each re-anchored at the handles' current poses
    (anchor_order) and held for as many model steps as it was in force in production, with its
    recorded eye command at each step, the eyes rendered at each burst's last frame as training renders
    them. yoked: the units in a random permutation (a new one each pass through them). yoked_sequence
    (keep_sequence): the units in their recorded order from a random first unit, running on through
    the last to the first and round again (the one break in the sequence per pass). The recorder keeps
    the order each frame was handed. Returns how many distinct units were replayed, passes made, orders
    played and (keep_sequence) the first unit."""
    body_ids = np.asarray(env.proprio.body_ids, dtype=np.int64)
    first = int(rng.integers(len(units))) if keep_sequence else None
    perm = np.roll(np.arange(len(units)), -first) if keep_sequence else rng.permutation(len(units))
    pos, passes, used, step, seq = 0, 1, set(), 0, 0
    while step < steps:
        if pos >= len(perm):
            perm = perm if keep_sequence else rng.permutation(len(units))
            pos, passes = 0, passes + 1
        k = int(perm[pos])
        pos += 1
        used.add(k)
        u = units[k]
        motor = anchor_order(u["rel"], np.array(env.data.xpos[body_ids]), np.array(env.data.xquat[body_ids]))
        mask, goal = u["mask"], u["hand_goal"]
        for s in range(len(u["gaze"])):
            if step >= steps:
                break
            g = np.asarray(u["gaze"][s], dtype=np.float64)
            gaze = g if np.isfinite(g).all() else None
            for rep in range(reps):
                raw = env.step(motor, gaze, mask, render=bool(render and rep == reps - 1),
                               **({} if goal is None else {"hand_joint_goal": goal}))
                rec.on_frame(env, raw, dict(step=step, rep=rep, reps=reps, author="yoked", latched=s == 0,
                                            order_seq=seq, bout=True, rest=False),
                             (motor, gaze, mask, None, goal))
            step += 1
        seq += 1
    out = dict(units_replayed=len(used), passes=passes, orders=seq)
    if keep_sequence:
        out["first_unit"] = first
    return out


def yoke_units(world_dirs: List[str], body: int, before_step: Optional[int] = None,
               max_chunks: Optional[int] = None) -> dict:
    """The learner's own orders production body `body` was handed, from its world recordings (the v2
    'orders' field group, observer_record.orders_in_force), as replayable UNITS: one per order - the
    records with one order sequence number, continuous, written by the learner (LEARNER_AUTHORS), no
    scheduled rest - holding what the order asked for (relative_order, from each handle's pose at the
    record before the order began, placed by forward kinematics of the recording's own saved model),
    its mask and hand joint goal, and the eye command of each of its records. An order runs on across
    a chunk boundary (one header, one world episode, consecutive steps, the same order sequence
    number): a resume or an early flush that puts a chunk's first record inside an order does not cut
    it. Orders from steps at or after before_step are left out (the saved worlds start there).
    Read-only on the recordings. Returns dict(units=[...], source=...); raises ValueError when no
    recording carries orders."""
    from mimo_bridge.observer_record import FIELD_ORDERS, orders_in_force, read_chunk
    from .daemon import kinematics_for
    units, authors, lengths, runs, steps_seen = [], defaultdict(int), defaultdict(int), [], []
    handle_names = None
    for world in world_dirs:
        paths = sorted(glob.glob(os.path.join(world, "b%d" % int(body), "chunk_*.npz")),
                       key=lambda p: (int(os.path.basename(p).split("_")[1]), os.path.basename(p)))
        if max_chunks:
            paths = paths[-int(max_chunks):]
        prev = None              # the previous record: (header name, episode, step, index, chunk)
        unit = None              # the order being gathered: its first record's facts and its eye commands
        used = False

        def close(u):
            nonlocal used
            if u is None:
                return
            u["gaze"] = np.asarray(u["gaze"], dtype=np.float64)
            units.append(u)
            authors[u["author"]] += 1
            lengths[len(u["gaze"])] += 1
            steps_seen.append(u["step"])
            used = True
        for p in paths:
            z = read_chunk(p)
            h = z["header"]
            if FIELD_ORDERS not in z["fields"]:
                close(unit)
                unit, prev = None, None
                continue
            of = orders_in_force(z)
            names = list(h.get("handle_names") or [])
            if handle_names is None:
                handle_names = names
            elif names != handle_names:
                raise ValueError("body %d: the recordings name different handles (%s, then %s)"
                                 % (body, handle_names, names))
            keep_rows = [names.index("gaze")] if "gaze" in names else []
            hb = np.asarray(h.get("handle_bodies") or [], dtype=np.int64)
            kin = kinematics_for(h, world)
            author_of = {int(v): k for k, v in (h.get("author_codes") or {}).items()}
            for i in range(len(z["step"])):
                step, episode, seq = int(z["step"][i]), int(z["episode"][i]), int(z["order_seq"][i])
                cont = (prev is not None and prev[0] == z["header_name"] and prev[1] == episode
                        and prev[2] == step - 1)
                author = author_of.get(int(z["author"][i]), "unknown")
                ok = (of["index"][i] >= 0 and author in LEARNER_AUTHORS and not bool(z["rest"][i])
                      and (before_step is None or step < int(before_step)))
                if unit is not None and ok and cont and seq == unit["order_seq"] and not bool(z["latched"][i]):
                    unit["gaze"].append(np.asarray(of["gaze"][i], dtype=np.float64))      # the order runs on
                else:
                    close(unit)
                    unit = None
                    if ok and cont and bool(z["latched"][i]):
                        pz, pi = prev[4], prev[3]
                        kin.place(pz["qpos"][pi], pz["mocap"][pi] if np.asarray(pz["mocap"]).size else None,
                                  pz["object_pose"][pi] if np.asarray(pz["object_pose"]).size else None)
                        k = int(of["index"][i])
                        mask, goal = of["mask"][k], of["hand_goal"][k]
                        unit = dict(rel=relative_order(of["motor"][k], kin.d.xpos[hb], kin.d.xquat[hb], keep_rows),
                                    mask=None if (mask < 0).all() else mask.astype(bool),
                                    # a goal with any unset joint is not handed on: the hand controller
                                    # refuses a partly unset goal (hand_target_control.validate_joints)
                                    hand_goal=(None if (goal.size == 0 or not np.isfinite(goal).all())
                                               else np.array(goal)),
                                    gaze=[np.asarray(of["gaze"][i], dtype=np.float64)],
                                    step=step, order_seq=seq, author=author)
                prev = (z["header_name"], episode, step, i, z)
        close(unit)
        if used:
            runs.append(world)
    if not units:
        raise ValueError("body %d: none of %s recorded a learner's order (the recorder's 'orders' field group, "
                         "observer_world_v2); the yoked control replays orders recorded after that recorder "
                         "ran in production" % (body, world_dirs))
    return dict(units=units, handle_names=handle_names,
                source=dict(worlds=runs, units=len(units), authors=dict(authors),
                            unit_steps={str(k): v for k, v in sorted(lengths.items())},
                            first_step=min(steps_seen), last_step=max(steps_seen), before_step=before_step))


def _yoke_worker(a):
    worlds, body, before_step, folder = a
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    plan = yoke_units(worlds, body, before_step)
    path = os.path.join(folder, "b%d.pkl" % body)
    with open(path + ".part", "wb") as f:
        pickle.dump(plan, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".part", path)
    return dict(body=body, file=path, source=plan["source"])


def _yoke_in_child(ctx, worlds: List[str], bodies: int, folder: str, before_step: Optional[int]) -> List[dict]:
    """yoke_units for every body in a process of its own (it loads the recording's saved model and
    forward kinematics), ended before the babble worlds are built."""
    os.makedirs(folder, exist_ok=True)
    with ctx.Pool(1) as pool:
        return pool.map(_yoke_worker, [(worlds, b, before_step, folder) for b in range(bodies)])


def record_one(run_dir: str, kind: str, body: int, steps: int, out_dir: str, seed: int,
               vision: str = "like", saved: Optional[dict] = None) -> dict:
    """One world under one chance process, recorded as body `body` under out_dir/observer/world.

    saved None: the run's world (its args.json), reset. saved (one entry of saved_world_plan):
    production body saved['body']'s world, built from its saved settings and restored to its saved
    state as the trainer's world worker restores it on resume (datasource._worker_main,
    'restore_full_state'), then checked (restore_receipt): a world not in its saved state is not
    recorded."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.observer_record import WorldRecorder
    scratch = os.path.join(out_dir, "scratch", "b%d" % body)
    os.makedirs(scratch, exist_ok=True)
    restored = None
    if saved is None:
        kw, reps, hold = world_kwargs_like(run_dir, scratch, seed, vision)
        env = BridgeEnv(**kw)
        start = dict(start=RESET_WORLD)
    else:
        with open(saved["file"], "rb") as f:
            spec = pickle.load(f)
        kw = babble_world_kwargs(spec["env_kwargs"], scratch)
        reps, hold = int(saved["reps"]), int(saved["hold"])
        env = BridgeEnv(**kw)
        env.restore_full_state(detach_recording(spec["world"]))
        restored = restore_receipt(env, spec["world"])
        if not restored["ok"]:
            env.close()
            raise RuntimeError("babble body %d: production body %d's world did not restore to its saved "
                               "state (%s differ); not recorded"
                               % (body, saved["body"], ", ".join(restored["differs"])))
        start = dict(start=SAVED_WORLDS, checkpoint=saved["checkpoint"], checkpoint_step=int(saved["step"]),
                     saved_body=int(saved["body"]), saved_sim_time_s=restored["saved_sim_time"])
    seen = bool(kw.get("vision"))
    extra = dict(baseline=kind, seed=seed, source_run=os.path.basename(run_dir) if run_dir else None,
                 vision=seen, **start)
    if kind == "random_orders":
        extra.update(order_strength=ORDER_STRENGTH, order_reach_cm=ORDER_REACH_CM)
    yoke = None
    if kind in YOKED_KINDS:
        if saved is None or not saved.get("yoke"):
            env.close()
            raise ValueError("the yoked controls replay production's orders in production's saved worlds: "
                             "they need --from-checkpoint and the recordings the orders come from")
        with open(saved["yoke"], "rb") as f:
            yoke = pickle.load(f)
        names = list(env.proprio.handle_names)
        if list(yoke.get("handle_names") or names) != names:
            env.close()
            raise ValueError("body %d: the recorded orders name handles %s, this world %s"
                             % (body, yoke.get("handle_names"), names))
        extra.update(yoke=dict(yoke["source"], method="re-anchored: each order's displacement and turn from "
                               "each handle's pose where it began, asked again from the pose where the "
                               "replayed order begins; " + (
                                   "in their recorded order from a random first order" if kind == "yoked_sequence"
                                   else "in a random permutation")))
    rec = WorldRecorder(os.path.join(out_dir, "observer", "world"), body=body, run="baseline_" + kind,
                        chunk_steps=100, budget_bytes=4 * 2 ** 30, reserve_bytes=8 * 2 ** 30, extra=extra)
    rng = np.random.default_rng(seed)
    if saved is None:
        env.reset()
        rec.mark("reset")
    else:
        rec.mark("restore")            # a new world episode for the recorder, as the trainer's worker marks it
    t0 = time.time()
    if kind in YOKED_KINDS:
        replay = replay_orders(env, rec, yoke["units"], steps, reps, rng, seen,
                               keep_sequence=kind == "yoked_sequence")
        rec.close()
        env.close()
        out = dict(body=body, records=rec.records, seconds=time.time() - t0, error=rec.error, vision=seen,
                   yoke=dict(yoke["source"], **replay))
        if restored is not None:
            out["restored"] = restored
        return out
    H = len(env.proprio.body_ids)
    author = "limp" if kind == "limp" else "random_actions"
    motor = ctrl = None
    for step in range(steps):
        latched = step % hold == 0
        if latched:
            if kind == "random_orders":
                motor = np.zeros((H, 9))
                # the hand controller (wrist_relative_v1) needs a finite position target for every
                # handle: each asks for where it is now, moved by up to 3 cm on each axis
                here = 100.0 * np.asarray(env.data.xpos[np.asarray(env.proprio.body_ids)], dtype=np.float64)
                motor[:, 0:3] = here + rng.uniform(-ORDER_REACH_CM, ORDER_REACH_CM, (H, 3))
                q = rng.normal(size=(H, 4))
                motor[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
                motor[:, 7] = 1.0
                motor[:, 8] = rng.uniform(0.0, ORDER_STRENGTH, H)
                ctrl = None
            elif kind == "random_ctrl":
                ctrl = rng.uniform(-1.0, 1.0, env.model.nu)
            else:
                ctrl = np.zeros(env.model.nu)
        for rep in range(reps):
            # the eyes' images as training renders them: the burst's last frame (train_mimo)
            raw = env.step(motor if ctrl is None else None, None, None, ctrl=ctrl,
                           render=bool(seen and rep == reps - 1))
            # the recorder keeps the order each world was handed (its v2 'orders' field group)
            rec.on_frame(env, raw, dict(step=step, rep=rep, reps=reps, author=author, latched=latched,
                                        order_seq=step // hold, bout=True, rest=False),
                         (motor if ctrl is None else None, None, None, ctrl, None))
    rec.close()
    env.close()
    out = dict(body=body, records=rec.records, seconds=time.time() - t0, error=rec.error, vision=seen)
    if restored is not None:
        out["restored"] = restored
    return out


def recording_header(out_dir: str) -> Optional[dict]:
    """The first world header of a recording (<out_dir>/observer/world/b*/header_*.json), or None."""
    for path in sorted(glob.glob(os.path.join(out_dir, "observer", "world", "b*", "header_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            continue
    return None


def chance_row(ds: List[dict], q: Optional[list], steps: float) -> dict:
    """One capability's (or sub-row's) chance rates from babble's counted detections: the count, the
    occasions (per body and part, exactly as the tracker groups the learner's: tracker.occasion_ids),
    acts_sq (the sum over occasions of the squared acts per occasion, so the tracker divides babble's
    act count by its MEASURED burst spread: tracker.babble_dispersion), and for a capability with a
    main measure q = [name, direction] its 5th, 50th and 95th percentiles over every measured act
    (measures) and over each occasion's FIRST measured act (measures_occasion, with
    measured_occasions: the line the tracker's size test judges the learner's occasions against)."""
    from .tracker import first_in_occasion, occasion_ids
    subs = [d.get("sub") or "" for d in ds]
    code = {s: i for i, s in enumerate(sorted(set(subs)))}
    step0 = np.asarray([d["step0"] for d in ds], dtype=np.int64)
    oid = (occasion_ids(step0, np.asarray([int(d.get("body", 0)) for d in ds], dtype=np.int64),
                        np.asarray([code[s] for s in subs], dtype=np.int64)) if ds else np.zeros(0, dtype=np.int64))
    sizes = np.bincount(oid) if oid.size else np.zeros(0, dtype=np.int64)
    row = dict(count=len(ds), occasions=int(np.count_nonzero(sizes)),
               acts_sq=float(np.sum(sizes.astype(np.float64) ** 2)),
               steps=steps, rate_per_step=(len(ds) / steps if steps else None))
    if q:
        vals = np.asarray([np.nan if (d.get("measures") or {}).get(q[0]) is None else d["measures"][q[0]] for d in ds],
                          dtype=np.float64)
        fin = np.isfinite(vals)
        if fin.any():
            row["measures"] = {q[0]: [float(np.percentile(vals[fin], p)) for p in (5, 50, 95)]}
            first = vals[first_in_occasion(step0, oid, fin)]
            row["measured_occasions"] = int(first.size)
            row["measures_occasion"] = {q[0]: [float(np.percentile(first, p)) for p in (5, 50, 95)]}
    return row


def summarize(out_dir: str, kind: str, vision: bool = None) -> dict:
    """Digest a baseline recording and reduce it to chance rates per capability and sub-row."""
    from .daemon import DETECTOR_VERSION, RunDigest, read_jsonl, world_signature
    from . import catalog as CAT
    from . import novelty as NV
    dg = RunDigest(out_dir)
    dg.digest(final=True)
    dets = read_jsonl(os.path.join(dg.out, "detections.jsonl"))
    eps = read_jsonl(os.path.join(dg.out, "episodes.jsonl"))
    steps = 0
    for body, pids in dg.exposure.items():
        for pid, bins in pids.items():
            steps += sum(r["steps"] for r in bins.values())
    cat = CAT.load()
    quality = {c["id"]: c.get("quality") for c in cat["capabilities"]}
    caps: Dict[str, dict] = {}
    groups = defaultdict(list)
    for d in dets:
        # the same acts production never counts are left out of chance as well
        if d.get("scripted") or d.get("caregiver_touch") or d.get("world_moved"):
            continue
        groups[(d["cap"], None)].append(d)
        if d.get("sub"):
            groups[(d["cap"], d["sub"])].append(d)

    fields = recording_fields(out_dir)
    unmeasured = unmeasured_caps(fields)
    for c in cat["capabilities"]:
        cid = c["id"]
        if cid in unmeasured:
            continue                  # never a zero: the tracker then calls the stage unmeasured
        caps[cid] = chance_row(groups.get((cid, None), []), quality.get(cid), steps)
        subs = {sub: chance_row(ds, quality.get(cid), steps) for (cc, sub), ds in groups.items() if cc == cid and sub}
        if subs:
            caps[cid]["subs"] = subs
    header = next(iter(dg.headers_seen.values()), None)
    if header is None:
        # a digest sealed by an earlier pass reads no chunk again (a re-summary under the same
        # detector version), and the four kept summaries of 2026-09-24 lost their world signature
        # that way: the header comes from the recording itself
        header = recording_header(out_dir)
    sig = world_signature(header) if header else None
    if vision is None and header is not None:
        vision = (header.get("extra") or {}).get("vision")
    return dict(schema="observer_baseline_v1", kind=kind, world_signature=sig, steps=steps,
                occasions_per="body and part",
                detector_version=DETECTOR_VERSION, generated=time.time(), vision=vision,
                order_strength=ORDER_STRENGTH if kind == "random_orders" else None,
                order_reach_cm=ORDER_REACH_CM if kind == "random_orders" else None, caps=caps,
                recorded_fields=sorted(fields), unmeasured_caps=unmeasured,
                novelty_runs=NV.baseline_runs(eps), novelty_steps=steps, episodes=len(eps))


def recording_fields(out_dir: str) -> set:
    """The recorder's v2 field groups EVERY header of a recording carries (observer_record.FIELDS;
    none for observer_world_v1)."""
    from mimo_bridge.observer_record import recorded_fields
    sets = []
    for path in sorted(glob.glob(os.path.join(out_dir, "observer", "world", "b*", "header_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                sets.append(recorded_fields(json.load(f)))
        except (OSError, ValueError):
            continue
    return set.intersection(*sets) if sets else set()


def unmeasured_caps(fields: set) -> Dict[str, str]:
    """{capability: why} for every detector whose v9 definition reads a field group the recording
    lacks (detectors.NEEDS_FIELDS)."""
    from .detectors import NEEDS_FIELDS
    out = {}
    for cid, needs in sorted(NEEDS_FIELDS.items()):
        missing = [n for n in needs if n not in fields]
        if missing:
            out[cid] = ("the recording carries no %s (observer_world_v1): %s is measured only from recordings "
                        "that do" % (" or ".join(missing).replace("_", " "), cid))
    return out


def _write_json(path: str, value) -> None:
    """Written whole or not at all: the digester reads these files while they are replaced."""
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1, default=_plain)
    os.replace(path + ".part", path)


def resummarize(signature: str, kind: str) -> str:
    """Digest a kept baseline recording again with the current detectors and rewrite its
    summary (the simulation is not repeated; its recordings list is kept). `kind` is the summary's
    file name, so a recording kept aside (random_orders_reset_world) is digested the same way."""
    from .daemon import SAVED
    final_dir = os.path.join(SAVED, "observer", "baselines", signature)
    rec_dir = os.path.join(final_dir, kind)
    path = os.path.join(final_dir, kind + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = {}
    summary = summarize(rec_dir, kind, old.get("vision", False if old else None))
    summary["recordings"] = old.get("recordings")
    summary["recording_dir"] = rec_dir
    summary["resummarized_from"] = old.get("detector_version")
    # What the recording WAS does not change when it is digested again: where its bodies started,
    # and the order process's settings. summarize() names the settings only under the base kind's
    # name, so a recording kept aside lost them (random_orders_vision_off has read None since its
    # re-summary of 2026-09-24): they come from the old summary, else the recording's own header.
    for k in SOURCE_FIELDS:
        if k in old:
            summary[k] = old[k]
    if summary.get("order_strength") is None:
        extra = (recording_header(rec_dir) or {}).get("extra") or {}
        if old.get("order_strength") is not None:
            summary.update(order_strength=old["order_strength"], order_reach_cm=old.get("order_reach_cm"))
        elif extra.get("baseline") == "random_orders":
            summary.update(order_strength=extra.get("order_strength", ORDER_STRENGTH),
                           order_reach_cm=extra.get("order_reach_cm", ORDER_REACH_CM))
    _write_json(path, summary)
    return path


def _worker(a):
    run_dir, kind, body, steps, out_dir, seed, vision, saved = (tuple(a) + (None,))[:8]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return record_one(run_dir, kind, body, steps, out_dir, seed, vision, saved)


def _start_of(summary: Optional[dict]) -> str:
    """Where a summary's bodies started; every recording before 2026-09-24 evening began at a reset
    and carries no field."""
    return (summary or {}).get("start") or RESET_WORLD


def aside_name(kind: str, old: dict, new: dict) -> Optional[str]:
    """The name baseline `old` takes beside `new` of the same kind: the kind, then where old's
    bodies started when that differs from new's, then old's vision setting when that differs
    (random_orders_reset_world, random_orders_vision_off); two saved-world recordings of the same
    vision from different checkpoints keep the older beside the newer as <kind>_ckpt_<its step>
    (2026-09-25: re-recorded from each newer checkpoint and compared, never deleted), and a recording
    of the same variant made by another recorder format (its summary's recorded_fields) keeps the
    older as <kind>[_ckpt_<step>]_v1. None when both are the same variant (same start, same vision,
    same checkpoint, same recorded fields): the newer then replaces the older."""
    parts = []
    if _start_of(old) != _start_of(new):
        parts.append(_start_of(old))
    was, now = bool(old.get("vision", False)), bool(new.get("vision", False))   # no field: vision off
    if was != now:
        parts.append("vision_on" if was else "vision_off")
    if not parts:
        saved = _start_of(old) == SAVED_WORLDS and old.get("checkpoint_step") is not None
        old_f, new_f = sorted(old.get("recorded_fields") or []), sorted(new.get("recorded_fields") or [])
        if saved and old.get("checkpoint_step") != new.get("checkpoint_step"):
            parts.append("ckpt_%d" % int(old["checkpoint_step"]))
        elif "recorded_fields" in new and old_f != new_f:
            # the same variant recorded by another recorder format (observer_world_v1 before v2): kept
            if saved:
                parts.append("ckpt_%d" % int(old["checkpoint_step"]))
            parts.append("v1" if not old_f else "earlier_fields")
    return "_".join([kind] + parts) if parts else None


def set_aside(final_dir: str, kind: str, vision: bool, start: str = RESET_WORLD,
              remove: bool = True, checkpoint_step: Optional[int] = None,
              recorded_fields: Optional[list] = None) -> Optional[str]:
    """A kept baseline of another variant than a new one (another vision setting, or bodies that
    started elsewhere) moves to its aside name (aside_name), summary and recording, so a new one
    never mixes with it and the old stays for comparison. Returns the name it took, or None.
    remove=False leaves <kind>.json for the caller to replace in one step (place): the digester
    reads it every cycle, and a missing file reads as "no babble" (a spurious stage-history reset)."""
    path = os.path.join(final_dir, kind + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        return None
    new = dict(vision=bool(vision), start=start, checkpoint_step=checkpoint_step)
    if recorded_fields is not None:
        new["recorded_fields"] = list(recorded_fields)
    tag = aside_name(kind, old, new)
    if tag is None:
        return None
    rec, aside = os.path.join(final_dir, kind), os.path.join(final_dir, tag)
    if os.path.isdir(aside):
        shutil.rmtree(aside)
    if os.path.isdir(rec):
        shutil.move(rec, aside)
    old["kind"] = tag
    old["recording_dir"] = aside
    old["vision"] = bool(old.get("vision", False))
    if "start" not in old:
        reset = source_fields(None)
        old.update(start=reset["start"], source=reset["source"])
    _write_json(os.path.join(final_dir, tag + ".json"), old)
    if remove:
        os.remove(path)
    return tag


def place(final_dir: str, kind: str, summary: dict, stage: str) -> str:
    """Move a finished recording (stage) and its summary to where the tracker reads it, or beside it.

    The tracker reads <kind>.json (daemon.load_baseline). Babble from saved worlds is that
    comparison: it replaces a reset-world one, which moves aside (set_aside) to
    <kind>_reset_world. A reset-world recording made while a saved-world comparison exists goes
    beside it, under its aside name, instead of replacing it. A saved-world recording from another
    checkpoint moves the previous comparison aside as <kind>_ckpt_<step> (aside_name), so the two
    can be compared (compare). Of two recordings of the same variant (same start, same vision, same
    checkpoint) the newer replaces the older, as before."""
    os.makedirs(final_dir, exist_ok=True)
    try:
        with open(os.path.join(final_dir, kind + ".json"), encoding="utf-8") as f:
            current = json.load(f)
    except (OSError, ValueError):
        current = None
    name = kind
    if current is not None and _start_of(current) == SAVED_WORLDS and _start_of(summary) != SAVED_WORLDS:
        name = aside_name(kind, summary, current)
    else:
        # <kind>.json itself is replaced below in one step, never left missing
        set_aside(final_dir, kind, bool(summary.get("vision")), _start_of(summary), remove=False,
                  checkpoint_step=summary.get("checkpoint_step"), recorded_fields=summary.get("recorded_fields"))
    rec_dir = os.path.join(final_dir, name)
    if os.path.isdir(rec_dir):
        shutil.rmtree(rec_dir)
    shutil.move(stage, rec_dir)
    summary["kind"] = name
    summary["recording_dir"] = rec_dir
    path = os.path.join(final_dir, name + ".json")
    _write_json(path, summary)
    return path


def compare(signature: str, kind: str = "random_orders", other: Optional[str] = None) -> dict:
    """Each capability's chance rate in two kept baselines of one world: by default the comparison
    (<kind>.json) against the reset-world recording kept aside (<kind>_reset_world.json), the test
    OBSERVER_DESIGN.md section 9 names for babble from saved worlds. Per row: acts and occasions,
    each per 1,000 body-steps, and the two-sided exact conditional test of the two counts (given
    both counts, how unlikely is a split this far from the one the two recordings' body-steps
    predict); rows come in order of the occasions' p-value. `comparable` is False when the two
    were digested with different detector versions (re-summarize the older one first). A capability
    one summary could not measure (its unmeasured_caps: the recording lacks a field group the
    detector reads) is listed with `measured` and no test, never compared against a zero. `other`
    names any kept baseline of the world: the previous checkpoint's <kind>_ckpt_<step> after a
    re-recording. Reads the two summaries only: no world."""
    from scipy import stats
    from .daemon import SAVED
    folder = os.path.join(SAVED, "observer", "baselines", signature)
    other = other or "%s_%s" % (kind, RESET_WORLD)
    loaded = []
    for name in (kind, other):
        with open(os.path.join(folder, name + ".json"), encoding="utf-8") as f:
            loaded.append(json.load(f))
    a, b = loaded
    ea, eb = float(a.get("steps") or 0), float(b.get("steps") or 0)

    def test(ka: int, kb: int) -> Optional[float]:
        n = ka + kb
        if not n or not ea or not eb:
            return None
        share = ea / (ea + eb)
        return float(min(1.0, 2.0 * min(stats.binom.sf(ka - 1, n, share), stats.binom.cdf(ka, n, share))))

    def per_1000(k: int, e: float) -> Optional[float]:
        return 1000.0 * k / e if e else None

    rows = []
    ca, cb = a.get("caps") or {}, b.get("caps") or {}
    ua, ub = a.get("unmeasured_caps") or {}, b.get("unmeasured_caps") or {}
    for cid in sorted(set(ca) | set(cb) | set(ua) | set(ub)):
        if cid not in ca or cid not in cb:
            rows.append(dict(cap=cid, measured=[cid in ca, cid in cb], acts=[(ca.get(cid) or {}).get("count"),
                                                                         (cb.get(cid) or {}).get("count")],
                             p_acts=None, p_occasions=None,
                             note=ua.get(cid) or ub.get(cid) or "absent from one summary"))
            continue
        ra, rb = ca[cid] or {}, cb[cid] or {}
        ka, kb = int(ra.get("count") or 0), int(rb.get("count") or 0)
        oa, ob = int(ra.get("occasions") or 0), int(rb.get("occasions") or 0)
        rows.append(dict(cap=cid, acts=[ka, kb], occasions=[oa, ob],
                         acts_per_1000_steps=[per_1000(ka, ea), per_1000(kb, eb)],
                         occasions_per_1000_steps=[per_1000(oa, ea), per_1000(ob, eb)],
                         p_acts=test(ka, kb), p_occasions=test(oa, ob)))
    rows.sort(key=lambda r: 2.0 if r["p_occasions"] is None else r["p_occasions"])
    return dict(signature=signature, baselines=[kind, other], starts=[_start_of(a), _start_of(b)],
                steps=[a.get("steps"), b.get("steps")], vision=[a.get("vision"), b.get("vision")],
                detector_versions=[a.get("detector_version"), b.get("detector_version")],
                comparable=a.get("detector_version") == b.get("detector_version"),
                checkpoint_step=a.get("checkpoint_step"), checkpoint_steps=[a.get("checkpoint_step"),
                                                                            b.get("checkpoint_step")],
                recorded_fields=[a.get("recorded_fields"), b.get("recorded_fields")], rows=rows)


def _record_all(ctx, jobs: list, workers: int) -> list:
    """Every body's recording, `workers` worlds at a time, each world in a process of its own."""
    with ctx.Pool(max(1, min(workers, len(jobs)))) as pool:
        return pool.map(_worker, jobs)


def _extract_worker(a):
    checkpoint, folder = a
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return extract_saved_worlds(checkpoint, folder)


def _extract_in_child(ctx, checkpoint: str, folder: str) -> dict:
    """extract_saved_worlds in a process that ends before the babble worlds are built, so the
    learner's state the companion also holds is never in memory beside them."""
    with ctx.Pool(1) as pool:
        return pool.apply(_extract_worker, ((checkpoint, folder),))


def run(run_dir: Optional[str], kind: str, bodies: int, steps: int, workers: int, root: str = None,
        vision: str = "like", from_checkpoint: Optional[str] = None,
        yoke_from: Optional[List[str]] = None) -> str:
    """Record `bodies` worlds under chance process `kind` for `steps` model steps each, digest them
    and place the summary (place). from_checkpoint: body i starts from production body i's saved
    world in that checkpoint (extract_saved_worlds); run_dir may then be None, and the run folder
    that wrote the checkpoint is named instead. kinds 'yoked' and 'yoked_sequence' need from_checkpoint:
    body i replays production body i's own orders recorded before the checkpoint's step in the run
    folders yoke_from (default: the checkpoint's own run folder), built in a process of its own
    (yoke_units)."""
    from .daemon import SAVED
    import multiprocessing as mp
    if not from_checkpoint and not run_dir:
        raise ValueError("a reset-world baseline needs the run folder whose args.json sets the world (--like)")
    if kind in YOKED_KINDS and not from_checkpoint:
        raise ValueError("the yoked controls replay production's own orders in production's saved worlds: "
                         "they need --from-checkpoint")
    stage = os.path.join(root or os.path.join(SAVED, "observer", "baselines", "_recording"),
                         "%s_%s" % (kind, SAVED_WORLDS) if from_checkpoint else kind)
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)
    ctx = mp.get_context("spawn")
    manifest, plan = None, [None] * bodies
    yokes = None
    if from_checkpoint:
        manifest = _extract_in_child(ctx, from_checkpoint, os.path.join(stage, "scratch", "saved_worlds"))
        plan = saved_world_plan(manifest, run_dir, vision, bodies)
        run_dir = run_dir or manifest.get("run_dir")
        if kind in YOKED_KINDS:
            sources = [os.path.join(r, "observer", "world") for r in (yoke_from or [manifest.get("run_dir")]) if r]
            yokes = _yoke_in_child(ctx, sources, bodies, os.path.join(stage, "scratch", "yoke"),
                                   int(manifest["step"]))
            for p, y in zip(plan, yokes):
                p["yoke"] = y["file"]
    jobs = [(run_dir, kind, b, steps, stage, 1000 + b, vision, plan[b]) for b in range(bodies)]
    results = _record_all(ctx, jobs, workers)
    if manifest is not None:
        # the worlds stay in the checkpoint; the per-body copies go, saved_worlds.json stays
        for w in manifest["worlds"]:
            try:
                os.remove(w["file"])
            except OSError:
                pass
    seen = bool(results and all(r.get("vision") for r in results))
    summary = summarize(stage, kind, seen)
    summary["recordings"] = results
    summary.update(source_fields(manifest, bodies, run_dir))
    if yokes is not None:
        summary["yoke"] = dict(source_runs=sorted({w for y in yokes for w in y["source"]["worlds"]}),
                               before_step=int(manifest["step"]),
                               units_per_body=[y["source"]["units"] for y in yokes],
                               replayed=[r.get("yoke", {}).get("units_replayed") for r in results],
                               passes=[r.get("yoke", {}).get("passes") for r in results],
                               first_unit=[r.get("yoke", {}).get("first_unit") for r in results],
                               method="each learner order (planner, self choice, acting snapshot, reflex, babble "
                                      "noise; never a rest hold) replayed whole %s, its displacement and turn "
                                      "from each handle's pose where it began asked again from the pose where "
                                      "the replayed order begins" % (
                                          "in its recorded order from a random first order (the sequence kept)"
                                          if kind == "yoked_sequence" else "in a random permutation"))
    sig = summary["world_signature"] or "unknown"
    return place(os.path.join(SAVED, "observer", "baselines", sig), kind, summary, stage)
