# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Every saved movement recording the viewer can read, each with an id the dashboard passes back (never a file path).

A RECORDING is one run's observer world records: per body, the chunks (chunk_<first step>_<process>_<sequence>.npz,
100 model steps each), the recorder headers and the saved model. The same run is often copied (a preserved copy on
D:, a reading copy, the archive of tools/observer/archive.py); copies of one run share the recorder's header file names,
so they are merged into one recording: every chunk name found in any copy, each read from the copy whose chunk file times
span the widest stretch (a copy made without file times shows one time for every chunk, and the viewer dates each
command by its chunk's file time).

Only the folders matched by ROOTS are read (the registered project records of the specification); the dashboard selects
a recording by its id.

KIND: 'training' for a run of the production lineage (its recorder names it H84_fresh_drivers_...), 'test' for every
other run (comparisons, repairs, rehearsals; the test-registry rows that name the run are listed with it).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
from typing import Dict, List, Optional

from . import HOME, REPO, part_path, replace_file

SOURCES_FILE = HOME / "sources.json"
TRAINING_PREFIX = "H84_fresh_drivers_"
_CHUNK = re.compile(r"chunk_(\d+)_(\d+)_(\d+)\.npz$")
_HEADER = re.compile(r"header_(\d+)_(\d+)\.json$")

# (where, pattern): every world folder the viewer reads. The scratch patterns name the job folders' own run folders,
# not the reading and rehearsal copies beside them (those are merged anyway, by header name).
ROOTS = (
    ("production", str(REPO / "Saved" / "mimo_bridge" / "experiments" / "*" / "observer" / "world")),
    ("preserved", "D:/UEMA_Research/external/D/UEMA_preserved/*/run/observer/world"),
    ("archive", "D:/UEMA_archive/observer_raw/*/observer/world"),
    ("archive", "D:/UEMA_archive/observer_preserved_*/*/observer/world"),
    ("scratch", "D:/UEMA_Research/external/D/UEMA_scratch/*/run/observer/world"),
    ("scratch", "D:/UEMA_Research/external/D/UEMA_scratch/*/run/*/observer/world"),
    ("scratch", "D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/learning_on/measure/run/*/run/observer/world"),
    ("scratch", "D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/probe_path/training_records/*/observer/world"),
)


def _body_folders(world: str) -> Dict[int, str]:
    out = {}
    for p in glob.glob(os.path.join(world, "b*")):
        m = re.fullmatch(r"b(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            out[int(m.group(1))] = p
    return out


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _registry_rows(root=REPO) -> List[tuple]:
    """(row id, the row as text) of every test-registry row, for finding the rows that name a run."""
    reg = _read_json(str(root / "docs" / "gain_of_function_v4" / "test_registry.json")) or {}
    return [(str(r.get("id")), json.dumps(r, ensure_ascii=False)) for r in reg.get("rows") or [] if isinstance(r, dict)]


def scan(roots=ROOTS, registry_rows: Optional[List[tuple]] = None) -> List[dict]:
    """Every recording under `roots`, copies merged (module doc). Each: id, run, kind, label, rows, args, bodies
    {body: {chunks: [[name, path, mtime, first step]], headers: [paths], model: path or None}}, first_step, last_step,
    first_time, last_time, chunks, copies [world folders]."""
    found: Dict[tuple, dict] = {}
    for where, pattern in roots:
        for world in sorted(glob.glob(pattern)):
            bodies = _body_folders(world)
            if not bodies:
                continue
            first_body = min(bodies)
            heads = sorted(glob.glob(os.path.join(bodies[first_body], "header_*.json")))
            if not heads:
                continue
            header = _read_json(heads[0]) or {}
            run = str(header.get("run") or os.path.basename(os.path.dirname(os.path.dirname(world))))
            key = (run, os.path.basename(heads[0]))
            rec = found.setdefault(key, dict(run=run, copies=[], bodies={}, _where=[]))
            rec["copies"].append(world.replace("\\", "/"))
            rec["_where"].append(where)
            run_dir = os.path.dirname(os.path.dirname(world))
            if "args" not in rec:
                args = _read_json(os.path.join(run_dir, "args.json"))
                if args is not None:
                    rec["args"] = {k: args.get(k) for k in ARG_KEYS}
            for b, folder in bodies.items():
                chunks = []
                for p in glob.glob(os.path.join(folder, "chunk_*.npz")):
                    m = _CHUNK.search(os.path.basename(p))
                    if not m or p.endswith(".part.npz"):
                        continue
                    try:
                        mt = os.path.getmtime(p)
                    except OSError:
                        continue
                    chunks.append((os.path.basename(p), p.replace("\\", "/"), mt, int(m.group(1))))
                if not chunks:
                    continue
                span = max(c[2] for c in chunks) - min(c[2] for c in chunks)
                slot = rec["bodies"].setdefault(b, dict(chunks={}, headers=set(), model=None))
                for name, path, mt, first in chunks:
                    have = slot["chunks"].get(name)
                    # the copy whose chunk times span the widest stretch dates commands best
                    if have is None or span > have[4]:
                        slot["chunks"][name] = (name, path, mt, first, span)
                for h in glob.glob(os.path.join(folder, "header_*.json")):
                    slot["headers"].add(h.replace("\\", "/"))
                if slot["model"] is None:
                    hd = _read_json(sorted(glob.glob(os.path.join(folder, "header_*.json")))[0]) or {}
                    mf = hd.get("model_file")
                    if mf and os.path.isfile(os.path.join(world, mf)):
                        slot["model"] = os.path.join(world, mf).replace("\\", "/")
    rows = _registry_rows() if registry_rows is None else registry_rows
    out = []
    for (run, head), rec in found.items():
        bodies = {}
        firsts, lasts, t0s, t1s, n = [], [], [], [], 0
        for b, slot in sorted(rec["bodies"].items()):
            ch = sorted(slot["chunks"].values(), key=lambda c: (c[3], c[0]))
            bodies[str(b)] = dict(chunks=[[c[0], c[1], c[2], c[3]] for c in ch], headers=sorted(slot["headers"]),
                                  model=slot["model"])
            firsts.append(ch[0][3])
            lasts.append(ch[-1][3])
            t0s.append(min(c[2] for c in ch))
            t1s.append(max(c[2] for c in ch))
            n += len(ch)
        if not bodies:
            continue
        rid = hashlib.sha1(("%s|%s" % (run, head)).encode("utf-8")).hexdigest()[:12]
        kind = "training" if run.startswith(TRAINING_PREFIX) else "test"
        label = _label(run, kind)
        if len(run) < 6:
            # a recorder named only "run" (the finger investigation's jobs): the job folders above it name it, and a
            # row search for "run" would match every row
            parts = rec["copies"][0].split("/")
            label = " ".join(p for p in parts[-6:-3] if p not in ("run", "measure", "observer"))
            named = []
        else:
            named = [rid_ for rid_, text in rows if run in text]
        out.append(dict(id=rid, run=run, kind=kind, label=label, rows=named, rows_total=len(named),
                        settings=settings_of(rec.get("args") or {}),
                        args=rec.get("args") or {}, bodies=bodies, copies=rec["copies"], where=sorted(set(rec["_where"])),
                        first_step=min(firsts), last_step=max(lasts) + 99, first_time=min(t0s), last_time=max(t1s),
                        chunks=n))
    out.sort(key=lambda r: r["first_time"])
    return out


# THE CONTROL SETTINGS THAT SPLIT THE RECORDINGS (the independent review of Oct 18, 2026, finding 3): the finger "gap
# closed" of the 6 runs with the finger joint servo off or unset reads 24-26%, of the runs with it on 36-39%. Runs with
# different settings are separate examples, never a before/after pair (the specification), so each recording carries
# them and the page compares runs only within one settings group.
SETTING_KEYS = ("finger_joint_servo", "arm_stage")
# the second review of Oct 18, 2026: two more control flags change the commands before the decoder reads them (env.py:
# they replace the calf, chest and arm rows), so a run with either on is its own group; four learner settings differ
# inside the groups and are named beside a comparison, without splitting it
ROW_FLAGS = (("parent_relative_targets", "parent-relative targets"), ("reach_projection", "reach projection"))
LEARNER_KEYS = ("proprio_token_skip", "proprio_pose", "attention_cap", "hand_response_calibration")
# a learner setting a run's arguments do not name reads as the trainer's default (train_mimo.py: --attention-cap off,
# --proprio-token-skip none, --hand-response-calibration none); proprio_pose is no argument of this checkout's trainer,
# and a run that does not name it ran without it, which the pose comparison's control arm names "off"
LEARNER_DEFAULTS = {"proprio_token_skip": "none", "proprio_pose": "off", "attention_cap": "off",
                    "hand_response_calibration": "none"}
ARG_KEYS = ("position_rows", "hand_target_mode", "resume", "body_layout", "finger_joint_servo", "intention",
            "babble_noise", "arm_stage", "smooth_actions", "motor_hold") + tuple(f for f, _ in ROW_FLAGS) + LEARNER_KEYS


def _on(value) -> bool:
    return str(value if value is not None else "off").strip().lower() not in ("off", "none", "false", "0", "")


def settings_of(args: dict) -> dict:
    """The recording's control settings that differ between the recordings: finger joint servo (unset = off, the
    trainer's default), arm stage (unset = off), and parent-relative targets and reach projection when on; `key` names
    the group, `label` says it in words. `other`: the learner settings of LEARNER_KEYS (LEARNER_DEFAULTS where the run's
    arguments do not name one; a calibration file by its file name), shown where two runs of one group differ in them."""
    servo = str(args.get("finger_joint_servo") or "off")
    stage = str(args.get("arm_stage") or "off")
    flags = [(flag, words) for flag, words in ROW_FLAGS if _on(args.get(flag))]
    label = ("finger joint servo %s" % servo + ("" if stage == "off" else ", arm stage %s" % stage)
             + "".join(", %s on" % words for _, words in flags))
    other = {}
    for k in LEARNER_KEYS:
        v = args.get(k)
        if k == "hand_response_calibration" and v:
            v = os.path.basename(str(v).replace("\\", "/"))
        other[k] = LEARNER_DEFAULTS[k] if v is None or v == "" else str(v)
    return dict(finger_joint_servo=servo, arm_stage=stage,
                key="servo_%s|arm_%s" % (servo, stage) + "".join("|%s" % flag for flag, _ in flags), label=label,
                smooth_actions=args.get("smooth_actions"), motor_hold=args.get("motor_hold"), other=other)


# EACH RUN IN WORDS, shown under its name on the page (the independent review of Oct 19, 2026: the run names are codes):
# a test run's words come from the title of its own row in docs/gain_of_function_v4/test_registry.json (named beside
# it; tests/test_movement_viewer.py checks that each row exists); the three finger-investigation runs of Oct 10 have no
# row and are described from their folder (D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/learning_on) and their settings
RUN_WORDS = {
    "arm_control": ("finger investigation from 481,800: the finger joint servo off", None),
    "arm_finger": ("finger investigation from 481,800: the finger joint servo on", None),
    "arm_control_again": ("finger investigation from 481,800: the servo off, run again", None),
    "B1LO_skip13_481800": ("body 1 left out of a training lesson, 481,800 to 484,675", "body1_lesson_off_skip13_481800"),
    "B1LO_pushon_481800": ("comparison run: the same span with body 1 kept in", "body1_lesson_off_pushon_481800"),
    "B1C_skip13_484675": ("body 1 left out: a continuation, 484,675 to 489,350", "body1_continuation_B1C_skip13_484675"),
    "B1E_skip13_485975": ("body 1 left out: the extension, 485,975 to 489,350", "body1_extension_B1E_skip13_485975"),
    "B1A_skip13_489350": ("the adopted body-1 line of training, 489,350 to 494,025", "b1_adopt_B1A_skip13_489350"),
    "B1R_skip13_489625": ("recovery run: that line resumed at 489,625, to 494,025", "b1_recover_B1R_skip13_489625"),
    "B1K_skip13_492800": ("continuation: 492,800 to 494,025, after the recovery run", "b1_continue_B1K_skip13_492800"),
    "ARM1_skip13_494025": ("arm stage on: a learning window from 494,025 (one command per body saved)",
                           "arm_stage_learning_494025_run"),
    "ARM2_skip13_494025": ("arm stage on: the learning window from 494,025, the replacement run",
                           "arm_stage_learning_494025_run2"),
    "ARMOFF_skip13_494025": ("arm stage off: the same window from 494,025, for comparison", "arm_stage_onoff_off_run_494025"),
    "POSE_B_on_494025": ("pose comparison B: the pose read-out term on, 494,025 to 496,500",
                         "pose_readout_comparison_B_on_494025_run"),
    "POSE_A_off_494025": ("pose comparison A: the pose read-out term off, 494,025 to 496,500",
                          "pose_readout_comparison_A_off_494025_run"),
}


def _clock(t) -> str:
    """A time of day in the dashboard computer's time zone (Pacific): '8:43 AM'."""
    lt = time.localtime(float(t))
    return "%d:%02d %s" % (lt.tm_hour % 12 or 12, lt.tm_min, "AM" if lt.tm_hour < 12 else "PM")


def about_of(rec: dict) -> dict:
    """(name, words, registry row) for the page: a training run is named by its segment and the clock time of its first
    saved record ("training segment 41c from 8:43 AM": three runs of Oct 6 share segment 41c, the independent review of
    Oct 19, 2026), a test run by its own name; the words come from RUN_WORDS (empty when a run has none)."""
    label = str(rec.get("label") or "")
    if rec.get("kind") == "training" and label.startswith("training "):
        seg = label.split("_seg", 1)[-1].split("_")[0] if "_seg" in label else label[len("training "):]
        start = rec.get("first_time")
        return dict(name="training segment " + seg + (" from " + _clock(start) if start else ""), words="", row=None)
    words, row = RUN_WORDS.get(label, ("", None))
    return dict(name=label, words=words, row=row)


def _label(run: str, kind: str) -> str:
    """A short name for the run: the segment name of a production run, the run's own name otherwise."""
    if run.startswith(TRAINING_PREFIX):
        return "training " + run[len(TRAINING_PREFIX):]
    return run


def write(recordings: List[dict], path=SOURCES_FILE) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    tmp = part_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(scanned_at=time.time(), roots=[p for _, p in ROOTS], recordings=recordings), f)
    replace_file(tmp, path)


def load(path=SOURCES_FILE) -> dict:
    """The last scan ({scanned_at, roots, recordings}); an empty one when none was written."""
    return _read_json(str(path)) or dict(scanned_at=None, roots=[p for _, p in ROOTS], recordings=[])


def by_id(rid: str, path=SOURCES_FILE) -> Optional[dict]:
    """The recording with this id from the last scan, or None (an unknown id is refused by the caller)."""
    if not isinstance(rid, str) or not re.fullmatch(r"[0-9a-f]{12}", rid):
        return None
    for r in load(path).get("recordings") or []:
        if r.get("id") == rid:
            return r
    return None
