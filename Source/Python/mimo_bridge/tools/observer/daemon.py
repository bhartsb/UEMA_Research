# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The observer's digestion service: new recordings in, development report out.

Runs in its own process beside training (never inside it). Each cycle, for every run of the
lineage that has an observer folder and is not yet SEALED:
  1. reads the world chunks it has not finalized, per body, in trainer-process order, at most
     MAX_WINDOW_CHUNKS at a time (a backlog after an outage is worked through window by window,
     so the digester's memory stays bounded);
  2. builds the track (features.py), primitives, episodes and detections over a window that
     reads LOOKBACK_CHUNKS chunks before its first unfinalized one and holds back the last MARGIN
     steps (an act near the live end is judged when its future has been recorded), never cutting
     through an episode;
  3. joins the learner's own records to each finalized detection (evidence.py) - it waits until
     the model lane covers those steps, so no act is judged without its learner record;
  4. appends detections.jsonl.gz and episodes.jsonl.gz (one gzip member per window), adds the
     finalized steps to the exposure (steps per author class and per caregiver presence) and to
     the fast lane's summaries, and COMMITS: progress.json holds the digest state, exposure, fast
     lane and the record files' sizes in one atomic write. A digester stopped between an append
     and its commit cuts the files back to the committed sizes when it starts again, so a window
     is never counted twice;
  5. marks a chunk digested once every record in it is finalized; digested chunks older than
     RAW_KEEP_S, or beyond four fifths of the recorder's own budget, are removed (the steps a
     replay needs are cut into a snippet first), each only after its copy on the archive disk
     checks out (archive.py, Oct 18, 2026: a chunk whose copy cannot be made stays).
A run that is no longer the active one and has nothing left to digest is SEALED: it is never
digested again, and only its raw-data housekeeping continues (hourly).
Then the LINEAGE report: the lineage store (store.py: every run's detections as columns, read
incrementally) feeds the tracker's stages for every capability with and without the caregiver,
the timeline, the novelty candidates (against babble's rate of the same runs), the horizon
report, the fast lanes and the recorder's health, written to
<Saved>/mimo_bridge/observer/<lineage>/development.json for the dashboard.

A trainer relaunch that resumed from an earlier checkpoint abandons the previous process's later
steps (stream.abandoned_from); the report reads every detection's liveness from the current
cuts, so an abandoned future stops counting the moment the relaunch appears.

A change of DETECTOR_VERSION or ANALYSIS_VERSION moves a run's old analysis folder aside
(analysis_<old version>_<time>) and digests the raw chunks still on disk again: detections of two
definitions are never mixed in one count.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import time
import traceback
import uuid
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np

from mimo_bridge.observer_record import read_chunk
from . import archive as AR
from . import catalog as CAT
from . import detectors as D
from . import evidence as EV
from . import fastlane as FL
from . import history as HI
from . import horizon as HZ
from . import novelty as NV
from . import pins as PN
from . import primitives as P
from . import replay as RP
from . import segmenter as S
from . import store as ST
from . import tracker as TR
from .features import Track
from .kinematics import Kinematics
from .stream import ModelLaneIndex, RunFiles, abandoned_from

# v4 (2026-09-24): the evidence join reads the learner's lane one record later (evidence.py,
# TRAINER_LANE_LAG), credits each act to the order at its peak motion, judges a touch only where the
# world shows it starting (its own bit, its untouched patches) and scores that verdict at matched
# no-touch moments; the tracker's rules are v3 (tracker.RULES_VERSION)
# v5 (2026-09-25, observer statistics track, docs/gain_of_function_v4/agents/observer_statistics_2026-09-25.md):
# the fast lanes in the exposure's 100-step bins with each spike found over the window's whole track
# (fastlane.binned), cut at the lineage cuts like the exposure (cut_fast); the learner's fast lane and
# the finalization's wait for the lane read lane step = world step + TRAINER_LANE_LAG; a window reads
# LOOKBACK_CHUNKS = 10 chunks before its new ones; a hand act is credited to the order one record
# before its peak motion (the lag rule; replaced on 2026-10-08 by observer_attribution_v1, which credits an act of
# the hands alone to no single order: evidence.command_intervals); each act's wellness change gets its reference
# stretches (evidence.wellness_references); the matched moments of both controls are placed by their
# distance from the act, not by the window, and an act waits until its moments are recorded (evidence
# module doc, MATCHED MOMENTS; the independent review of 2026-09-25, finding 1)
ANALYSIS_VERSION = "observer_analysis_v5"
DETECTOR_VERSION = "observer_detectors_v11"     # Oct 12: contact attribution stops at onset; two-hand frame overlap and transfer continuity
MARGIN = 60              # steps held back at the live end (3 s of the act's future)
# Chunks read before a window's first unfinalized chunk (1 until analysis v5; the constant CONTEXT = 40
# steps it replaces was never read). THE LIVE DIGEST COUNTED FEWER TURN ACTS (2026-09-25, observer
# statistics track): finger, wrist and closing turns are a zigzag scan that begins where the read record
# begins (primitives._turns), so where a window starts moves the turns after it. Segment 40's 9:33 AM run
# (2,000 steps x 8 bodies), digested in 1-chunk windows as the live digester sees chunks arrive, against
# one pass over the whole run: with 1 chunk of look-back 265 of 7,733 acts differed (3.4%: palm shaping,
# finger isolation, spreading, the wrist), and production's live digest of that run matched the 1-chunk
# count; with 3 chunks 126, 5 chunks 47, 10 chunks 22 (0.28%), 20 and 40 chunks 21. The remaining 21 are
# turns at the window's END, which the zigzag confirms only when a later reversal comes (the detector's
# definition, not the window's). The cost of 10: that run's 1-chunk digest took 85 s against 42 s.
LOOKBACK_CHUNKS = 10
BIN = 100                # exposure bins, in steps
FAST_BIN = BIN           # fast-lane bins (1,000 steps before analysis v5): cut exactly where the exposure is
MAX_WINDOW_CHUNKS = 20   # chunks decoded at once per body (2,000 steps)
MAX_SPAN_CHUNKS = 4 * MAX_WINDOW_CHUNKS   # a window widened for a long episode stops here
LANE_STALE_S = 600       # a learner lane with no new chunk for this long is not waited for
SEAL_QUIET_S = 600       # a run is sealed only after its recorders wrote nothing for this long
WINDOW_FAILURES = 3      # a window failing this many passes in a row is set aside (quarantined)
READ_FAILURES = 3        # an unreadable chunk is given up after this many attempts
RAW_KEEP_S = 72 * 3600
REPLAY_MAX_PER_REPORT = 12
ROOT = CAT.ROOT
SAVED = os.path.join(str(ROOT), "Saved", "mimo_bridge")


def _atomic_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, default=_jsonable)
    for wait in (0.005, 0.02, 0.08, 0.32, 1.0):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(wait)
    os.replace(tmp, path)


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def _clean(v):
    """JSON-safe copy (NaN -> None)."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


def _dumps(r) -> str:
    return json.dumps(_clean(r), default=_jsonable)


def read_jsonl(path: str) -> List[dict]:
    """Every record of a record file (plain or gzip-compressed JSON lines)."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    rows, _ = ST.read_records(path)
    return rows


def world_signature(header: dict) -> str:
    """What makes two worlds' chance rates comparable: scene, body, objects, skin."""
    key = dict(scene=header.get("scene"), nq=header.get("nq"), nu=header.get("nu"),
               objects=sorted(o["name"] for o in header.get("objects", [])),
               patches=len(header.get("patch_names", [])), handles=header.get("handle_names"))
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


_KIN: Dict[str, Kinematics] = {}
KIN_CACHE = 1


def kinematics_for(header: dict, world_dir: str) -> Kinematics:
    """One Kinematics per kinematic structure, shared by every body that recorded it.

    Each holds the compiled model's physics arena (380 MB private for H84's world, measured
    2026-09-24); keyed per header, the first production pass built one per body, eight copies,
    4.8 GB. The eight worlds' headers name the same saved model (its SHA-256 in the header) and
    the same objects and joints; place() writes each step's own mount poses before any
    measurement, so sharing is exact."""
    shape = json.dumps([[(o.get("name"), o.get("root"), o.get("free"), o.get("mocap")) for o in header.get("objects", [])],
                        [(j.get("name"), j.get("qpos")) for j in header.get("joints", [])]], sort_keys=True)
    # the saved model's own hash when the header has it (every run of a lineage then shares one
    # copy: the four run folders of 2026-09-24 held 0.96 GB with one per folder); else its place
    model = header.get("model_sha256") or "|".join((os.path.abspath(world_dir), header["model_file"]))
    key = "|".join((model, hashlib.sha256(shape.encode("utf-8")).hexdigest()[:16]))
    k = _KIN.pop(key, None)
    if k is None:
        k = Kinematics(header, world_dir)
    _KIN[key] = k                  # most recently used last
    while len(_KIN) > KIN_CACHE:
        _KIN.pop(next(iter(_KIN)))
    return k


class ChainReader:
    """causal_chain.jsonl read incrementally, keeping only what the evidence join needs."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "causal_chain.jsonl")
        self.offset = 0
        self.latches: List[dict] = []

    def update(self) -> int:
        n = 0
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return 0
        end = data.rfind(b"\n")
        if end < 0:
            return 0
        self.offset += end + 1
        for line in data[:end + 1].splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") != "latch":
                continue
            self.latches.append(latch_summary(r))
            n += 1
        return n


def latch_summary(r: dict) -> dict:
    """What the evidence join keeps of one planner latch record (causal_chain.jsonl).

    THE SELECTION THE PLANNER RECORDED BEFORE EXECUTION (observer_attribution_v1; the operator's addition of Oct 8,
    2026, relayed by the coordinator). The planner search picks its order by each candidate's PREDICTED distance to the
    goal entry's tokens (goal_distance: the chosen order and the alternatives it is recorded against, holding still,
    the motor controller's order and two perturbed orders), and the hand stage then makes its own calculation
    (hand_optimizer: the frozen fit's predicted finger-pose error at its first and its selected joint goal). Both are
    kept per body beside the approximate wellness evaluator's outcome, which the decisive test still reads with its tie
    band unchanged (register entry OBSERVER_SELECTED_STAGE_READS_UNOPTIMIZED_EVALUATOR)."""
    plan = r.get("plan") or {}
    gaps = plan.get("pred_wellness_gaps") or {}
    hand = plan.get("hand_optimizer") or {}
    return dict(type="latch", step=int(r.get("step", -1)), order_seq=r.get("order_seq"),
                source=r.get("source"), reached_body=r.get("reached_body"),
                plan=dict(horizon=plan.get("horizon"),
                          pred_wellness_gaps=dict(outcome=gaps.get("outcome"),
                                                  gap_to_best_alt=gaps.get("gap_to_best_alt"),
                                                  tie_band=gaps.get("tie_band")),
                          goal_distance=plan.get("goal_distance"), alternatives=plan.get("alternatives"),
                          pred_change=plan.get("pred_change"),
                          hand_optimizer=dict(first=hand.get("first"), selected=hand.get("selected"),
                                              valid=hand.get("valid"), stopped_by=hand.get("stopped_by"))))


RECORD_FILES = ("detections.jsonl.gz", "episodes.jsonl.gz")


def final_cut(cut_i: int, eps: List[dict], dets: List[dict], closed: bool) -> int:
    """The last track index a window finalizes, moved back from cut_i until it lies inside no
    episode and, unless the record is closed, before every act built on a turn that is still open
    where the window's record ends (detections carrying open_end; primitives.turns_confirmed, detector
    v9): such an act's extreme and start may still move with the records to come, so the window
    after, which reads it again from its look-back, judges it once its turn is confirmed. A turn is
    confirmed within primitives.TURN_HOLD_RECORDS of the series' last progress, so the wait is
    bounded."""
    while True:
        new = cut_i
        for e in eps:
            if e["i0"] <= new < e["i1"]:
                new = min(new, e["i0"] - 1)
        if not closed:
            for d in dets:
                if d.get("open_end") and d["i0"] <= new:
                    new = min(new, d["i0"] - 1)
        if new == cut_i:
            return cut_i
        cut_i = new


def make_joiner(lane, latches: List[dict], header: dict) -> EV.Joiner:
    """The evidence join for a production run: the trainer writes its model lane one record behind
    the world recorder (evidence.TRAINER_LANE_LAG, measured on segment 39)."""
    return EV.Joiner(lane, latches, (lane.handles if lane is not None else []), header,
                     lane_lag=EV.TRAINER_LANE_LAG)


class RunDigest:
    def __init__(self, run_dir: str):
        self.files = RunFiles(run_dir)
        self.run = os.path.basename(os.path.abspath(run_dir))
        self.out = self.files.analysis
        self.beat: Optional[Callable[[str, int], None]] = None
        self._archive_other_versions()
        os.makedirs(self.out, exist_ok=True)
        self.progress_path = os.path.join(self.out, "progress.json")
        self.state_path = os.path.join(self.out, "state.json")
        prog = self._load("progress.json", None)
        if prog is not None:
            self.state, self.exposure, self.fast = prog["state"], prog["exposure"], prog["fast"]
            self.committed = {k: int(v) for k, v in (prog.get("sizes") or {}).items()}
        else:
            self.state, self.exposure, self.fast = {}, {}, {}
            self.committed = {}
        for name in RECORD_FILES:
            self.committed.setdefault(name, 0)
            ST.truncate(os.path.join(self.out, name), self.committed[name])
        self.state.setdefault("version", ANALYSIS_VERSION)
        self.state.setdefault("detector_version", DETECTOR_VERSION)
        self.state.setdefault("bodies", {})
        self.state.setdefault("cuts", {})
        # one id per analysis folder: the lineage store notices a run digested again from scratch
        self.state.setdefault("analysis_id", uuid.uuid4().hex)
        self.lane: Optional[ModelLaneIndex] = None
        self.chain = ChainReader(run_dir)
        self.headers_seen: Dict[str, dict] = {}

    def _archive_other_versions(self) -> None:
        """Move an analysis folder written by another analysis or detector version aside."""
        if not os.path.isdir(self.out):
            return
        versions = None
        for name in ("progress.json", "state.json"):
            p = os.path.join(self.out, name)
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    v = json.load(f)
                st = v.get("state", v) if name == "progress.json" else v
                versions = (st.get("version"), st.get("detector_version"))
            except (OSError, ValueError, AttributeError):
                versions = (None, None)
            break
        if versions is None:
            versions = (None, None) if os.listdir(self.out) else (ANALYSIS_VERSION, DETECTOR_VERSION)
        if versions != (ANALYSIS_VERSION, DETECTOR_VERSION):
            dest = "%s_%s_%d" % (self.out, versions[1] or "unknown", int(time.time()))
            os.replace(self.out, dest)
            _atomic_json(os.path.join(dest, "moved_aside.json"),
                         dict(reason="analysis or detector version changed", old=list(versions),
                              new=[ANALYSIS_VERSION, DETECTOR_VERSION], at=time.time()))

    def _load(self, name, default):
        try:
            with open(os.path.join(self.out, name), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return default

    def kin_for(self, header: dict) -> Kinematics:
        return kinematics_for(header, self.files.world)

    @property
    def sealed(self) -> bool:
        return bool(self.state.get("sealed"))

    def newest_write(self) -> float:
        """When the run's recorders last wrote anything: world chunks, headers and status files
        of every body, and the learner lane's files (0.0 when there are none)."""
        newest = 0.0
        for pattern in (os.path.join(self.files.world, "b*", "*"), os.path.join(self.files.model, "*")):
            for p in glob.glob(pattern):
                try:
                    newest = max(newest, os.path.getmtime(p))
                except OSError:
                    continue
        return newest

    def reopen_if_written(self) -> bool:
        """UNSEAL A RUN ITS RECORDERS WROTE AFTER ITS SEAL (2026-09-24): segment 39's first run was
        sealed at 3:38 PM PDT with 0 chunks, two minutes after it started and before its first chunk
        arrived (another run was still the active one), and its steps 390,650-392,949 never reached
        the report. A seal is a promise that nothing more will come; new files break it."""
        sealed_at = self.state.get("sealed")
        if not sealed_at or self.newest_write() <= float(sealed_at):
            return False
        self.state.pop("sealed", None)
        self.state.setdefault("reopened", []).append(time.time())
        self.commit()
        return True

    # ------------------------------------------------------------------ commit
    def commit(self) -> None:
        """One atomic write of everything a restart resumes from (module doc, step 4), then the
        readers' copies."""
        self.state["updated"] = time.time()
        self.committed = {name: ST.size(os.path.join(self.out, name)) for name in RECORD_FILES}
        _atomic_json(self.progress_path, dict(state=self.state, exposure=self.exposure, fast=self.fast,
                                              sizes=self.committed))
        _atomic_json(self.state_path, self.state)
        _atomic_json(os.path.join(self.out, "exposure.json"), self.exposure)
        _atomic_json(os.path.join(self.out, "fastlane.json"), self.fast)

    # ------------------------------------------------------------------ one cycle
    def digest(self, final: bool = False) -> dict:
        summary = dict(run=self.run, detections=0, episodes=0, chunks=0, errors=[], pending=0)
        if not self.files.exists() or self.sealed:
            return summary
        if os.path.isdir(self.files.model):
            if self.lane is None:
                self.lane = ModelLaneIndex(self.files)
            else:
                self.lane.update()
        self.chain.update()
        for body in self.files.bodies():
            try:
                r = self._digest_body(body, final)
                for k in ("detections", "episodes", "chunks", "pending"):
                    summary[k] += r.get(k, 0)
            except Exception:
                summary["errors"].append(dict(body=body, error=traceback.format_exc()[-2000:]))
        if self.lane is not None:
            finals = [min(int(v) for v in b["final"].values())
                      for b in self.state["bodies"].values() if b.get("final")]
            if finals:
                self.lane.prune(min(finals) - 2000)
        # and quiet: a run its recorders wrote within SEAL_QUIET_S may still be starting or running
        if (final and not summary["errors"] and summary["pending"] == 0
                and time.time() - self.newest_write() > SEAL_QUIET_S):
            self.state["sealed"] = time.time()
            self.lane = None
            self.chain.latches = []
        self.commit()
        return summary

    def _digest_body(self, body: int, final: bool) -> dict:
        res = dict(detections=0, episodes=0, chunks=0, pending=0)
        bst = self.state["bodies"].setdefault(str(body), dict(done={}, final={}))
        bst.setdefault("read_failures", {})
        bst.setdefault("window_failures", {})
        bst.setdefault("last_end", {})
        segs = self.files.world_segments(body)
        cuts = abandoned_from(segs)
        self.state["cuts"][str(body)] = {str(p): int(c) for p, c in cuts.items()}
        for seg in segs:                                  # where and when this run's record begins
            if seg.chunks:
                self.state["first_step"] = min(int(self.state.get("first_step", seg.first_step)), int(seg.first_step))
                self.state["start_ms"] = min(int(self.state.get("start_ms", seg.start_ms) or seg.start_ms), int(seg.start_ms))
        for si, seg in enumerate(segs):
            closed_seg = final or si < len(segs) - 1
            span = MAX_WINDOW_CHUNKS
            while True:
                todo = [c for c in seg.chunks if c.name not in bst["done"]]
                if not todo:
                    break
                idx0 = seg.chunks.index(todo[0])
                window = seg.chunks[max(0, idx0 - LOOKBACK_CHUNKS):idx0 + span]
                whole = idx0 + span >= len(seg.chunks)
                decoded = []
                for c in window:
                    try:
                        decoded.append((c, read_chunk(c.path)))
                        bst["read_failures"].pop(c.name, None)
                    except Exception:
                        if c in todo:
                            n_fail = bst["read_failures"][c.name] = bst["read_failures"].get(c.name, 0) + 1
                            if n_fail >= READ_FAILURES:
                                bst["done"][c.name] = dict(pid=seg.pid, unreadable=True, at=time.time())
                groups: List[list] = []
                for c, d in decoded:
                    if groups and groups[-1][-1][1]["header_name"] == d["header_name"]:
                        groups[-1].append((c, d))
                    else:
                        groups.append([(c, d)])
                before = len(bst["done"])
                key = todo[0].name
                try:
                    for gi, group in enumerate(groups):
                        closed = (closed_seg and whole) or gi < len(groups) - 1
                        r = self._process(body, seg.pid, group, closed, cuts.get(seg.pid))
                        for k in r:
                            res[k] += r[k]
                    bst["window_failures"].pop(key, None)
                except Exception:
                    # a window that fails again and again is set aside with its reason, never
                    # retried forever: the body's later record keeps being digested
                    n_fail = bst["window_failures"][key] = bst["window_failures"].get(key, 0) + 1
                    self.state.setdefault("window_errors", {})[key] = traceback.format_exc()[-1500:]
                    if n_fail < WINDOW_FAILURES:
                        self.commit()
                        raise
                    last = None
                    for c in window:
                        if c in todo:
                            bst["done"][c.name] = dict(pid=seg.pid, quarantined=True, at=time.time())
                            last = c
                    if last is not None:
                        # its steps are skipped, not counted: the finalized line moves past them
                        z = read_chunk(last.path) if os.path.exists(last.path) else None
                        if z is not None:
                            bst["final"][str(seg.pid)] = max(int(bst["final"].get(str(seg.pid), -1)), int(z["step"][-1]))
                        self.state["quarantined_windows"] = int(self.state.get("quarantined_windows", 0)) + 1
                self.commit()
                if self.beat is not None:
                    self.beat(self.run, body)
                if whole:
                    break
                if len(bst["done"]) == before:
                    if span >= MAX_SPAN_CHUNKS:
                        break                # nothing can be finalized yet: the next pass tries again
                    span = min(2 * span, MAX_SPAN_CHUNKS)   # one episode longer than the window: widen it
            res["pending"] += sum(1 for c in seg.chunks if c.name not in bst["done"])
        return res

    def _process(self, body: int, pid: int, group: list, closed: bool, cut: Optional[int]) -> dict:
        bst = self.state["bodies"][str(body)]
        fin = bst["final"].get(str(pid))
        lo = int(fin) if fin is not None else -(1 << 62)
        chunks = [d for _, d in group]
        header = chunks[0]["header"]
        self.headers_seen[world_signature(header)] = header
        kin = self.kin_for(header)
        steps = np.concatenate([d["step"] for d in chunks])
        live = np.ones(len(steps), dtype=bool) if cut is None else steps < cut
        tr = Track(chunks, kin, pid=pid, live=live)
        prims = P.all_primitives(tr)
        # episodes resume after the finalized line: the last pass never cut through an episode,
        # so everything that starts after it belongs to episodes not yet written
        eps = S.segment(tr, [p for p in prims if int(tr.step[p["i0"]]) > lo], self.run, body)
        dets = D.run_all(tr, prims)
        S.attach(eps, dets)
        cut_i = tr.n - 1 if closed else tr.n - 1 - MARGIN
        # the production lane is one record behind the world's (evidence.TRAINER_LANE_LAG)
        joiner = make_joiner(self.lane, self.chain.latches, header)
        ahead = None                     # the last world step whose record and learner record are in
        if not closed:
            ahead = int(tr.step[-1])
        if self.lane is not None and not closed and self._lane_alive():
            # the learner's record of world record s is lane step s + TRAINER_LANE_LAG, and an act
            # finalized at the cut reads the lane up to MARGIN steps past it (its end, the records
            # after it, its order's efference record): the lane is held back as the world is (until
            # analysis v5 the cut waited only for lane step = world step)
            while cut_i >= 0 and int(tr.step[cut_i]) + EV.TRAINER_LANE_LAG + MARGIN > self.lane.max_step:
                cut_i -= 1
            ahead = min(ahead, int(self.lane.max_step) - EV.TRAINER_LANE_LAG)
        # never cut through an episode or before an act whose turn is still open (final_cut, detectors v9),
        # and finalize an act only once its matched moments are recorded (evidence.Joiner.matched_horizon,
        # analysis v5, independent review of 2026-09-25, finding 1): one loop until the cut stops moving
        while True:
            new = final_cut(cut_i, eps, dets, closed)
            if ahead is not None:
                for d in dets:
                    if d["i0"] <= new and int(d["step0"]) > lo:
                        h = joiner.matched_horizon(d, tr)
                        if h is not None and h > ahead:
                            new = min(new, d["i0"] - 1)
            if new == cut_i:
                break
            cut_i = new
        if cut_i < 0 or int(tr.step[cut_i]) <= lo:
            return dict(detections=0, episodes=0, chunks=0)
        cut_step = int(tr.step[cut_i])
        new_eps = [e for e in eps if e["i1"] <= cut_i and e["step0"] > lo]
        # an act overlapping one already finalized for the same capability, sub-row and key is
        # the same act found again from a window that began elsewhere: it is not emitted twice
        last_end = bst.setdefault("last_end", {})
        new_dets = []
        for d in sorted((d for d in dets if d["i0"] <= cut_i and d["step0"] > lo), key=lambda d: d["step0"]):
            k = "%s|%s|%s|%s" % (pid, d["cap"], d.get("sub", ""), d.get("key", ""))
            if k in last_end and int(d["step0"]) < int(last_end[k]):
                continue
            last_end[k] = max(int(last_end.get(k, -1)), int(d["step1"]))
            new_dets.append(d)
        # each act's matched moments are fixed by its distance from the act, whichever window
        # finalizes it; every act the window found decides whose moment is whose (evidence.py)
        joiner.attach(new_dets, body, tr, all_dets=dets)
        for d in new_dets:
            d.update(body=int(body), pid=int(pid), run=self.run, detector_version=DETECTOR_VERSION)
            for k in ("i0", "i1", "onset_index", "entry_index", "contact_entities", "open_end"):
                d.pop(k, None)
        for e in new_eps:
            e.update(run=self.run)
            e.pop("i0", None)
            e.pop("i1", None)
        ST.append_records(os.path.join(self.out, "detections.jsonl.gz"), new_dets, _dumps)
        ST.append_records(os.path.join(self.out, "episodes.jsonl.gz"), new_eps, _dumps)
        sel = np.nonzero((tr.step > lo) & (tr.step <= cut_step))[0]
        self._expose(body, pid, tr, sel)
        self._fast(body, pid, tr, sel)
        for c, d in group:
            if int(d["step"][-1]) <= cut_step:
                bst["done"][c.name] = dict(pid=int(pid), last=int(d["step"][-1]), at=time.time())
        bst["final"][str(pid)] = cut_step
        return dict(detections=len(new_dets), episodes=len(new_eps), chunks=len(group))

    def _expose(self, body: int, pid: int, tr: Track, sel: np.ndarray) -> None:
        """Each finalized record's body-step in its BIN, per author class and caregiver presence; each bin
        also keeps its first and last recorded step (lo, hi, since analysis v5), so a lineage cut inside
        the bin counts only the records before it (_before_cut)."""
        ex = self.exposure.setdefault(str(body), {}).setdefault(str(pid), {})
        for i in sel:
            s = int(tr.step[i])
            b = str(s // BIN)
            row = ex.setdefault(b, dict(steps=0, by={}, abs=0, abs_by={}))
            row["lo"] = min(int(row.get("lo", s)), s)
            row["hi"] = max(int(row.get("hi", s)), s)
            cls = D.provenance_class(D.AUTHOR_NAMES.get(int(tr.author[i]), "unknown"))
            row["steps"] += 1
            row["by"][cls] = row["by"].get(cls, 0) + 1
            if not bool(tr.caregiver_present[i]):
                row["abs"] += 1
                row["abs_by"][cls] = row["abs_by"].get(cls, 0) + 1

    def _fast(self, body: int, pid: int, tr: Track, sel: np.ndarray) -> None:
        """The fast lanes of the records this window finalizes (sel), per FAST_BIN bin of steps: each
        lane's spikes found over the window's whole track and counted in the bin of their peak
        (fastlane.binned). The learner's lane is read one record later than the world's
        (evidence.TRAINER_LANE_LAG: lane step t describes world record t - 1); until analysis v5 it
        was read at lane step = world step, one record early."""
        if not len(sel):
            return
        self.state["fast_bin"] = FAST_BIN
        fb = self.fast.setdefault(str(body), {}).setdefault(str(pid), {})
        w = float(tr.header.get("weight_n", 30.0))
        keep = np.zeros(tr.n, dtype=bool)
        keep[np.asarray(sel, dtype=np.int64)] = True
        bins = np.asarray(tr.step, dtype=np.int64) // FAST_BIN
        lanes: Dict[int, dict] = defaultdict(dict)
        series = FL.world_series(tr.fast, tr.frames)
        # each frame's record, as world_series lays the frames out (at most the recorded slots a record)
        rec = np.repeat(np.arange(tr.n), np.clip(np.asarray(tr.frames, dtype=np.int64), 0, tr.fast.shape[2]))
        for name, floor in FL.world_floors(w):
            for b, v in FL.binned(series[name], FL.FRAME_S, rec, keep, bins, min_excess=floor).items():
                lanes[b][name] = v
        if self.lane is not None:
            lag = EV.TRAINER_LANE_LAG
            rows = [self.lane.rows.get(int(s) + lag) for s in tr.step]
            for s_i, sense in enumerate(self.lane.senses or []):
                x = np.asarray([np.nan if r is None else float(r["next_pe_body"][body, s_i]) for r in rows])
                if np.count_nonzero(np.isfinite(x)) < 10:
                    continue
                for b, v in FL.binned(x, FL.STEP_S, np.arange(tr.n), keep, bins, ratio=True).items():
                    if v["samples"]:
                        lanes[b]["learner_" + sense] = v
        for b, per in lanes.items():
            row = fb.setdefault(str(b), {})
            for k, v in per.items():
                prev = row.get(k)
                if prev is None:
                    row[k] = v
                else:
                    _merge_fast(prev, v)

    def _lane_alive(self) -> bool:
        """The learner lane is still writing: its latest process left no error file, its status
        shows no refusal (a budget or disk-reserve stop writes only gaps) and it closed a chunk
        within LANE_STALE_S. Only then does finalizing a world step wait for the lane."""
        hdrs = sorted(glob.glob(os.path.join(self.files.model, "header_*.json")), key=os.path.getmtime)
        if not hdrs:
            return False
        pid = os.path.basename(hdrs[-1]).split("_")[1]
        if os.path.exists(os.path.join(self.files.model, "lane_error_%s.json" % pid)):
            return False
        try:
            with open(os.path.join(self.files.model, "status_%s.json" % pid), encoding="utf-8") as f:
                st = json.load(f)
            if st.get("refusal") or st.get("enabled") is False or st.get("error"):
                return False
        except (OSError, ValueError):
            pass
        chunks = glob.glob(os.path.join(self.files.model, "chunk_*.npz"))
        newest = max((os.path.getmtime(p) for p in chunks), default=os.path.getmtime(hdrs[-1]))
        return time.time() - newest < LANE_STALE_S

    # ------------------------------------------------------------------ raw data housekeeping
    def housekeeping(self, keep_s: float = RAW_KEEP_S, pinned: Optional[set] = None,
                     pins: Optional[PN.Pins] = None, archive: Optional[AR.Archive] = None) -> dict:
        """Remove digested raw chunks past their keep time, and the oldest digested ones while a
        body's folder is beyond four fifths of its recorder budget; never a chunk `pinned` names, nor one
        an evidence pin keeps (pins.py: the coordinator's pin list), nothing in this run while an invalid pin names
        it, and nothing at all while the pin list cannot be read or an invalid pin names no usable run. With `archive`
        (archive.py; the digester's own pass always gives it) a chunk is removed only after its archived copy has the
        original's size and SHA-256; a chunk whose copy cannot be made stays. Returns what was removed and what the
        pins kept (`pinned_kept`: world chunks per body, lane chunks; `pins`: the ids of the pins naming this run;
        `command_history`: whether its causal_chain.jsonl exists; `blocked`: why nothing was removed; `archive_kept`:
        chunks kept because their archived copy could not be made, with `archive_why`)."""
        removed, freed = 0, 0
        pinned = set(pinned or set())
        now = time.time()
        run_pins = pins.for_run(self.files.run) if pins is not None else []
        report = dict(removed=0, freed_bytes=0, pins=[p["id"] for p in run_pins],
                      pinned_kept=dict(world={}, model=0), blocked=None, archive_kept=0, archive_why=None)

        def archived(p: str) -> bool:
            """True when `p` may be removed: no archive was given, or its archived copy checks out."""
            if archive is None:
                return True
            ok, why = archive.ensure(self.files.run, p)
            if not ok:
                report["archive_kept"] += 1
                report["archive_why"] = report["archive_why"] or why
            return ok
        if run_pins:
            report["command_history"] = os.path.exists(os.path.join(self.files.run, "causal_chain.jsonl"))
        if pins is not None and pins.blocks_all:
            report["blocked"] = pins.error or "; ".join(pins.block_all)
            return report
        if pins is not None and pins.blocks_run(self.files.run):
            report["blocked"] = "; ".join(pins.blocks_run(self.files.run)) + ": no raw record of this run is deleted"
            return report
        for body in self.files.bodies():
            bst = self.state["bodies"].get(str(body), {})
            done = bst.get("done", {})
            folder = self.files.body_dir(body)
            kept = set()
            if run_pins:
                kept = set(PN.pinned_names(run_pins, PN.chunk_ranges(folder), body=body))
                if kept:
                    report["pinned_kept"]["world"][str(body)] = len(kept)
            budget = None
            hdrs = sorted(glob.glob(os.path.join(folder, "header_*.json")))
            if hdrs:
                try:
                    with open(hdrs[-1], encoding="utf-8") as f:
                        budget = json.load(f).get("budget_bytes")
                except (OSError, ValueError):
                    budget = None
            files = []
            for name, info in done.items():
                p = os.path.join(folder, name)
                if os.path.exists(p) and name not in pinned and name not in kept:
                    files.append((os.path.getmtime(p), p, os.path.getsize(p)))
            files.sort()
            used = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)
                       if os.path.isfile(os.path.join(folder, f)))
            for mt, p, sz in files:
                over = budget is not None and used > 0.8 * float(budget)
                if now - mt > keep_s or over:
                    if not archived(p):
                        continue
                    try:
                        os.remove(p)
                        removed += 1
                        freed += sz
                        used -= sz
                    except OSError:
                        pass
        # the learner lane: chunks wholly before every body's finalized steps
        finals = [min(int(v) for v in b["final"].values())
                  for b in self.state["bodies"].values() if b.get("final")]
        if finals and os.path.isdir(self.files.model):
            low = min(finals) - 2000
            lane_kept = set()
            if run_pins:
                # the lane numbers its records one step after the world's (evidence.TRAINER_LANE_LAG), and the
                # efference record of the command in force at a pin's last step comes when the next one begins
                lane_kept = set(PN.pinned_names(run_pins, PN.chunk_ranges(self.files.model),
                                                shift_lo=EV.TRAINER_LANE_LAG,
                                                shift_hi=EV.TRAINER_LANE_LAG + PN.LANE_REACH_STEPS))
                report["pinned_kept"]["model"] = len(lane_kept)
            for p in glob.glob(os.path.join(self.files.model, "chunk_*.npz")):
                if os.path.basename(p) in lane_kept:
                    continue
                try:
                    first = int(os.path.basename(p).split("_")[1])
                    if (first + 100 < low or self.sealed) and now - os.path.getmtime(p) > keep_s:
                        if not archived(p):
                            continue
                        sz = os.path.getsize(p)
                        os.remove(p)
                        removed += 1
                        freed += sz
                except (OSError, ValueError, IndexError):
                    pass
        report.update(removed=removed, freed_bytes=freed)
        return report


# ====================================================================== lineage report
def _before_cut(start: int, size: int, cut: Optional[int], lo: Optional[int] = None, hi: Optional[int] = None,
                records: Optional[float] = None) -> float:
    """The share of a bin's records [start, start + size) that come before a cut: 1 without a cut or with
    the bin wholly before it, 0 with the bin at or past it. A cut INSIDE the bin (a crash resume restarts at
    a multiple of 25 steps, so its first step can fall inside a 100-step bin) counts the bin's records
    before it: (cut - lo) / (hi - lo + 1) of them, the records being consecutive from the bin's first
    recorded step lo to its last hi (analysis v5; independent review of 2026-09-25, finding 6: the whole bin
    was counted, up to 99 steps a body twice). A bin written before lo and hi were kept is taken to start
    at its first step with its `records` consecutive records (a run that ended before the cut, mid-bin,
    keeps them all), or without them to fill the bin."""
    if cut is None or int(cut) >= start + size:
        return 1.0
    if int(cut) <= start:
        return 0.0
    if lo is None or hi is None:
        lo = start
        hi = start + (int(records) if records else size) - 1
    if int(hi) < int(lo):
        return 1.0
    return float(min(1.0, max(0.0, (int(cut) - int(lo)) / float(int(hi) - int(lo) + 1))))


def exposure_for_tracker(exposures: List[tuple]) -> Dict[str, dict]:
    """[(exposure dict, cuts[, lineage cut])] of every run -> per body {steps, by_author, windows,
    absent...}; the records at or past a process's cut or the run's lineage cut are an abandoned future
    and not counted (a bin straddling a cut counts its records before it: _before_cut)."""
    out: Dict[str, dict] = {}
    for item in exposures:
        ex, cuts = item[0], item[1]
        line_cut = item[2] if len(item) > 2 else None
        for body, pids in ex.items():
            row = out.setdefault(body, dict(steps=0, by_author=defaultdict(float), windows=defaultdict(float),
                                            absent_steps=0, absent_by_author=defaultdict(float),
                                            absent_windows=defaultdict(float)))
            for pid, bins in pids.items():
                cut = (cuts.get(body) or {}).get(pid)
                for b, r in bins.items():
                    start = int(b) * BIN
                    f = min(_before_cut(start, BIN, cut, r.get("lo"), r.get("hi"), r.get("steps")),
                            _before_cut(start, BIN, line_cut, r.get("lo"), r.get("hi"), r.get("steps")))
                    if f <= 0.0:
                        continue
                    w = str(start // TR.RET_WINDOW)
                    counted = f * (r["steps"] - sum(v for k, v in r["by"].items() if k in TR.EXCLUDED))
                    row["steps"] += counted
                    row["windows"][w] += counted
                    for k, v in r["by"].items():
                        row["by_author"][k] += f * v
                    ca = f * (r["abs"] - sum(v for k, v in r["abs_by"].items() if k in TR.EXCLUDED))
                    row["absent_steps"] += ca
                    row["absent_windows"][w] += ca
                    for k, v in r["abs_by"].items():
                        row["absent_by_author"][k] += f * v
    return {b: {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in r.items()} for b, r in out.items()}


def load_baseline(signature: str, kind: str = "random_orders") -> dict:
    """A baseline summary of this world (baseline.py), or {}; random orders are the babble that
    stages and novelty are judged against."""
    path = os.path.join(SAVED, "observer", "baselines", signature, kind + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            b = json.load(f)
    except (OSError, ValueError):
        return {}
    if b.get("detector_version") not in (None, DETECTOR_VERSION):
        return {}                    # chance rates of other detector definitions are no control
    return b


def load_baselines(signature: str) -> Dict[str, dict]:
    return load_baseline(signature).get("caps", {})


def recorder_health(run_dir: str) -> dict:
    files = RunFiles(run_dir)
    out = dict(bodies={}, model=None)
    for body in files.bodies():
        folder = files.body_dir(body)
        st = sorted(glob.glob(os.path.join(folder, "status_*.json")), key=os.path.getmtime)
        s = None
        if st:
            try:
                with open(st[-1], encoding="utf-8") as f:
                    s = json.load(f)
            except (OSError, ValueError):
                s = None
        err = os.path.join(folder, "recorder_error.json")
        out["bodies"][str(body)] = dict(status=s, error=os.path.exists(err), gaps=len(files.gaps(body)),
                                        bytes=sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)
                                                  if os.path.isfile(os.path.join(folder, f))))
    if os.path.isdir(files.model):
        st = sorted(glob.glob(os.path.join(files.model, "status_*.json")), key=os.path.getmtime)
        if st:
            try:
                with open(st[-1], encoding="utf-8") as f:
                    out["model"] = json.load(f)
            except (OSError, ValueError):
                pass
    try:
        out["disk_free_gib"] = shutil.disk_usage(run_dir).free / 2 ** 30
    except OSError:
        out["disk_free_gib"] = None
    hdrs = sorted(glob.glob(os.path.join(files.world, "b*", "header_*.json")), key=os.path.getmtime)
    try:
        with open(hdrs[-1], encoding="utf-8") as f:
            out["reserve_gib"] = float(json.load(f).get("reserve_bytes")) / 2 ** 30
    except (IndexError, OSError, ValueError, TypeError):
        out["reserve_gib"] = None
    return out


def _run_state(run: str, digests: Optional[Dict[str, RunDigest]]) -> tuple:
    """(state, exposure, fast lane) of a run: the digest in memory, else its files."""
    dg = (digests or {}).get(run)
    if dg is not None:
        return dg.state, dg.exposure, dg.fast
    a = RunFiles(run).analysis
    out = []
    for name in ("state.json", "exposure.json", "fastlane.json"):
        try:
            with open(os.path.join(a, name), encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            out.append({})
    return tuple(out)


def quality_map(cat: dict) -> Dict[str, str]:
    return {c["id"]: c["quality"][0] for c in cat["capabilities"] if c.get("quality")}


def lineage_cuts(states: Dict[str, dict]) -> Dict[str, int]:
    """run -> the step from which its records are an abandoned future of the LINEAGE: the first
    recorded step of any run started after it. The supervisor resumes each continuation into a new
    run folder; after a crash it resumes from the last hourly checkpoint, and the new run records
    the same step numbers again, so the old run's later steps must stop counting (a cut inside
    one folder, stream.abandoned_from, never sees the next folder)."""
    known = [(int(st.get("start_ms") or 0), int(st["first_step"]), run) for run, st in states.items()
             if st.get("first_step") is not None]
    out = {}
    for start, first, run in known:
        later = [f for s, f, r in known if s > start]
        if later:
            out[run] = min(later)
    return out


def _cut_of(cuts_by_run: Dict[str, dict], line: Dict[str, int], run, body, pid) -> Optional[int]:
    c = ((cuts_by_run.get(run) or {}).get(str(body)) or {}).get(str(pid))
    l = line.get(run)
    vals = [int(v) for v in (c, l) if v is not None]
    return min(vals) if vals else None


def live_rows(store: ST.LineageStore, cuts_by_run: Dict[str, dict], line: Optional[Dict[str, int]] = None) -> np.ndarray:
    """One flag per stored detection: not in an abandoned future, inside its run folder (a later
    process of the same run) or across the lineage (a later run folder)."""
    line = line or {}
    t = store.dets.freeze()
    live = np.ones(len(t), dtype=bool)
    if not len(t):
        return live
    runs, bodies, pids, step0 = t["run"], t["body"], t["pid"], t["step0"]
    for run, cuts in cuts_by_run.items():
        rid = store.strings.index.get(run)
        if rid is None:
            continue
        for body, per in (cuts or {}).items():
            for pid, cut in per.items():
                m = (runs == rid) & (bodies == int(body)) & (pids == int(pid)) & (step0 >= int(cut))
                live[m] = False
    for run, cut in line.items():
        rid = store.strings.index.get(run)
        if rid is not None:
            live[(runs == rid) & (step0 >= int(cut))] = False
    return live


def _episode_live(e: dict, cuts_by_run: Dict[str, dict], line: Optional[Dict[str, int]] = None) -> bool:
    c = _cut_of(cuts_by_run, line or {}, e.get("run"), e.get("body"), e.get("pid"))
    return c is None or int(e.get("step0", 0)) < int(c)


def build_report(run_dirs: List[str], out_path: str, digests: Optional[Dict[str, RunDigest]] = None,
                 render_replays: bool = True, store: Optional[ST.LineageStore] = None,
                 *, current_run: Optional[str] = None) -> dict:
    cat = CAT.load()
    if store is None:
        store = ST.LineageStore(None, quality_map(cat))
        for run in run_dirs:
            dg = (digests or {}).get(run)
            store.update(os.path.basename(run), RunFiles(run).analysis, dg.committed if dg else None)
    exposures, fast, cuts_by_run, states = [], {}, {}, {}
    header = None
    for run in run_dirs:
        name = os.path.basename(run)
        st, ex, fl = _run_state(run, digests)
        states[name] = st
        cuts_by_run[name] = st.get("cuts", {})
        fast[name] = fl
    line = lineage_cuts(states)
    for run in run_dirs:
        name = os.path.basename(run)
        st, ex, fl = _run_state(run, digests)
        exposures.append((ex, cuts_by_run[name], line.get(name)))
        # the fast lanes are cut where the exposure is (cut_fast)
        fast[name] = cut_fast(fl, cuts_by_run[name], line.get(name), st.get("fast_bin", 10 * BIN), ex)
        hs = sorted(glob.glob(os.path.join(run, "observer", "world", "b*", "header_*.json")), key=os.path.getmtime)
        if hs:
            try:
                with open(hs[-1], encoding="utf-8") as f:
                    header = json.load(f)
            except (OSError, ValueError):
                pass
    signature = world_signature(header) if header else None
    base_full = load_baseline(signature) if signature else {}
    baselines = base_full.get("caps", {})
    exposure = exposure_for_tracker(exposures)
    counted_steps = float(sum(v["steps"] for v in exposure.values()))
    t = store.dets.freeze()
    live = live_rows(store, cuts_by_run, line)
    # novelty: the counted episodes' runs against the babble episodes' runs; an episode counts when
    # it is live, untouched by the caregiver and mostly written by a counted author (the exposure
    # it is compared over leaves probe, baseline and rest steps out the same way)
    ep_rows = store.eps.rows
    use = np.asarray([_episode_live(e, cuts_by_run, line) and not e["caregiver_touch"]
                      and D.provenance_class(e.get("author", "unknown")) not in TR.EXCLUDED
                      and not e.get("toy_held") for e in ep_rows], dtype=bool)
    nov_base = dict(runs=base_full.get("novelty_runs") or {}, steps=base_full.get("novelty_steps") or 0)
    nov_tests: dict = {}
    cands = NV.mine_table(store.eps, use, store.eps.tokens, nov_base if nov_base["steps"] else None, counted_steps,
                          tests=nov_tests)
    by_id = {e["id"]: dict(e, live=bool(u)) for e, u in zip(ep_rows, use)}
    o12 = NV.o12_detections(cands, by_id)
    tracked = TR.track_all(t, exposure, baselines, cat, live=live)
    if o12:
        tracked["O12"].update(TR.track_all(o12, exposure, baselines, dict(capabilities=[cat["by_id"]["O12"]]))["O12"])
    tl = TR.timeline(tracked)
    step_now = int(t["step0"].max()) if len(t) else None
    control = "babble %s" % base_full.get("generated") if baselines else "no babble"
    for lost in stage_history(os.path.dirname(out_path), tracked, step_now, control):
        tl.append(dict(step=lost["step"] if lost["step"] is not None else (step_now or 0), cap=lost["cap"],
                       name=(tracked.get(lost["cap"]) or {}).get("name"), stage="lost:" + lost["stage"]))
    tl.sort(key=lambda e: e["step"])
    # Empty failed launches sort after recorded runs. The selected run is supplied
    # by the supervisor/CLI; directory ordering cannot identify current health.
    latest = current_run or (run_dirs[-1] if run_dirs else None)
    lane = None
    chain = []
    if latest:
        dg = (digests or {}).get(latest)
        if dg is not None:
            lane, chain = dg.lane, dg.chain.latches
        elif os.path.isdir(RunFiles(latest).model):
            lane = ModelLaneIndex(RunFiles(latest))
    horizon = HZ.report(lane, chain)
    replays = _replays(run_dirs, tracked, cands, by_id, out_path) if render_replays else {}
    bodies = sorted(int(b) for b in np.unique(t["body"])) if len(t) else []
    fast_by_body = _fast_summary(fast)
    report = dict(
        version=ANALYSIS_VERSION, detector_version=DETECTOR_VERSION, generated=time.time(),
        # the counting rules and the last step, so a dated copy (history.py) says what it was judged under
        rules_version=TR.RULES_VERSION, last_step=step_now,
        runs=[os.path.basename(r) for r in run_dirs], world_signature=signature,
        current_run=os.path.basename(latest) if latest else None,
        baseline=dict(available=bool(baselines), signature=signature, generated=base_full.get("generated"),
                      kind="random_orders" if baselines else None, caps=len(baselines),
                      steps=base_full.get("steps"), order_reach_cm=base_full.get("order_reach_cm"),
                      order_strength=base_full.get("order_strength"),
                      novelty_measured=bool(nov_base["steps"]),
                      # the first baselines carry no field: they were recorded with vision off
                      vision=bool(base_full.get("vision", False)) if baselines else None,
                      start=base_full.get("start"), checkpoint_step=base_full.get("checkpoint_step"),
                      training_vision=_run_vision(latest)),
        counts=dict(detections=int(len(t)), episodes=len(ep_rows), live_detections=int(live.sum()),
                    bodies=bodies),
        recording=dict(first_step=min((int(st["first_step"]) for st in states.values() if st.get("first_step") is not None), default=None),
                       lineage_cuts=line),
        exposure={b: dict(steps=v["steps"], absent_steps=v["absent_steps"], by_author=v["by_author"])
                  for b, v in exposure.items()},
        capabilities=tracked, timeline=tl, horizon=horizon,
        novelty=[{k: v for k, v in c.items() if k != "occurrences"} for c in cands], novelty_tests=nov_tests,
        fast_lane=fast_by_body, fast_lane_total=_fast_total(fast_by_body),
        fast_lane_chance=baseline_fast(signature),
        health={os.path.basename(latest): recorder_health(latest)} if latest else {},
        replays=replays, catalog_rows=len(cat["capabilities"]),
        roles=_roles())
    _atomic_json(out_path, _clean(report))
    return report


def cut_fast(fl: dict, cuts: dict, line_cut: Optional[int], bin_steps: int = None, ex: Optional[dict] = None) -> dict:
    """A run's fast lanes without its abandoned future (OBSERVER_DESIGN.md section 9): the records at or
    past its process's cut inside the run folder (stream.abandoned_from) or the run's lineage cut
    (lineage_cuts: the next run's first step) are left out, as exposure_for_tracker leaves them out of
    the exposure, so a crash resume's steps recorded twice count once in the fast lanes too (until
    analysis v5 the fast lanes of both copies were summed). A bin straddling a cut keeps the share of
    its records before it (_before_cut, from the same bin's first and last step in the exposure `ex`
    when the bins are the exposure's): its samples, spikes, areas and recovery times scaled and rounded,
    its largest values as they were. bin_steps: the run's fast-lane bin (state 'fast_bin'; 1,000 steps
    before v5)."""
    size = int(bin_steps or FAST_BIN)
    out: dict = {}
    for body, pids in (fl or {}).items():
        for pid, bins in pids.items():
            cut = (cuts.get(body) or {}).get(pid)
            for b, lanes in bins.items():
                start = int(b) * size
                r = ((((ex or {}).get(body) or {}).get(pid) or {}).get(b) or {}) if size == BIN else {}
                f = min(_before_cut(start, size, cut, r.get("lo"), r.get("hi"), r.get("steps")),
                        _before_cut(start, size, line_cut, r.get("lo"), r.get("hi"), r.get("steps")))
                if f <= 0.0:
                    continue
                if f < 1.0:
                    lanes = {k: _scaled_fast(v, f) for k, v in lanes.items()}
                out.setdefault(body, {}).setdefault(pid, {})[b] = lanes
    return out


def _scaled_fast(v: dict, f: float) -> dict:
    """One fast-lane bin summary kept to the share f of its records (cut_fast)."""
    out = dict(v)
    for k in ("samples", "spikes"):
        out[k] = int(round(f * int(v.get(k, 0) or 0)))
    for k in ("area_sum", "recovery_sum_s"):
        out[k] = f * float(v.get(k, 0.0) or 0.0)
    return out


def _merge_fast(prev: dict, v: dict) -> None:
    """Add one fast-lane summary into another: counts and sums add, the largest is the larger."""
    for k in ("samples", "spikes"):
        prev[k] = int(prev.get(k, 0)) + int(v.get(k, 0))
    for k in ("area_sum", "recovery_sum_s"):
        prev[k] = float(prev.get(k, 0.0) or 0.0) + float(v.get(k, 0.0) or 0.0)
    for k in ("largest", "largest_ratio", "short_peak"):
        vals = [x for x in (prev.get(k), v.get(k)) if x is not None]
        if vals:
            prev[k] = float(max(vals))


def _fast_summary(fast: dict) -> dict:
    """Per body and lane over every run: spikes per 1000 samples, the largest, the mean excess
    area of a spike and its mean recovery time."""
    out = {}
    for run, bodies in fast.items():
        for body, pids in bodies.items():
            row = out.setdefault(body, {})
            for pid, bins in pids.items():
                for b, lanes in bins.items():
                    for k, v in lanes.items():
                        r = row.setdefault(k, dict(samples=0, spikes=0))
                        _merge_fast(r, v)
    for body, row in out.items():
        for k, r in row.items():
            n = r.get("spikes", 0)
            r["spikes_per_1000"] = 1000.0 * n / r["samples"] if r.get("samples") else None
            r["mean_area"] = r.get("area_sum", 0.0) / n if n else None
            r["mean_recovery_s"] = r.get("recovery_sum_s", 0.0) / n if n else None
    return out


# STAGE HISTORY (section 32: "first seen, repeated, chosen, lost"). Every report recomputes each stage
# from all the evidence, so a stage that passed and later failed left no trace (OBSERVER_DESIGN.md
# section 9). stage_state.json keeps, per capability row and stage, whether it was ever reached and the
# step at which it was last found lost; stage_history.jsonl appends every loss and every return, with
# the p-value and the line at the time. A new detector or analysis version, new counting rules
# (tracker.RULES_VERSION) or a new babble recording (the control every stage is judged against) starts
# the state afresh, with a 'reset' line in the history: a changed definition or control is not a loss.
def stage_history(folder: str, tracked: dict, step_now: Optional[int], control: str = "") -> List[dict]:
    """Record this report's stage changes; return the stages lost now: [{step, cap, stage}]."""
    path = os.path.join(folder, "stage_state.json")
    try:
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
    except (OSError, ValueError):
        prev = {}
    version = "%s/%s/%s/%s" % (ANALYSIS_VERSION, DETECTOR_VERSION, TR.RULES_VERSION, control)
    fresh = prev.get("version") != version
    rows = {} if fresh else dict(prev.get("rows") or {})
    now = time.time()
    flips = []
    if fresh and prev.get("rows"):
        # say so in the history, so a loss recorded before the change is read against it; and name every stage that
        # stood before the change and does not stand under the new version (independent review of 4342f9194: tracker
        # rules v6 withdrew H05's Predicted pass with only a 'reset' line), with what it reads now
        withdrawn = []
        for key, old in sorted((prev.get("rows") or {}).items()):
            if not old.get("ever") or old.get("lost_at") is not None:
                continue
            cid, stage = key.split("|", 1)
            st = ((tracked.get(cid) or {}).get("stages") or {}).get(stage) or {}
            if st.get("reached") is True:
                continue
            withdrawn.append(dict(cap=cid, stage=stage, first_step=old.get("first_step"),
                                  now=("not judged" if st.get("not_judged") else
                                       "not measurable" if st.get("reached") is None else "not reached"),
                                  why=st.get("note")))
        flips.append(dict(time=now, change="reset", step=step_now,
                          reason="definitions, counting rules or babble changed: %s -> %s" % (prev.get("version"), version),
                          withdrawn=withdrawn))
    for cid, row in tracked.items():
        for stage in TR.STAGES:
            st = (row.get("stages") or {}).get(stage) or {}
            key = "%s|%s" % (cid, stage)
            old = rows.get(key) or {}
            if st.get("reached") is True:
                if old.get("lost_at") is not None:
                    flips.append(dict(time=now, cap=cid, stage=stage, change="back", step=step_now,
                                      p_value=st.get("p_value"), alpha=st.get("alpha")))
                rows[key] = dict(ever=True, lost_at=None, first_step=st.get("first_step"))
            elif old.get("ever") and old.get("lost_at") is None:
                flips.append(dict(time=now, cap=cid, stage=stage, change="lost", step=step_now,
                                  p_value=st.get("p_value"), alpha=st.get("alpha")))
                rows[key] = dict(old, lost_at=step_now)
    if flips:
        with open(os.path.join(folder, "stage_history.jsonl"), "a", encoding="utf-8") as f:
            for e in flips:
                f.write(json.dumps(_clean(e)) + "\n")
    _atomic_json(path, dict(version=version, updated=now, rows=rows))
    return [dict(step=v["lost_at"], cap=k.split("|")[0], stage=k.split("|")[1])
            for k, v in sorted(rows.items()) if v.get("lost_at") is not None]


def _run_vision(run_dir: Optional[str]) -> Optional[bool]:
    """Whether a training run renders the eyes' images (its args.json), or None if unreadable."""
    try:
        with open(os.path.join(run_dir, "args.json"), encoding="utf-8") as f:
            return not bool(json.load(f).get("no_vision", False))
    except (OSError, ValueError, TypeError):
        return None


def _fast_total(per_body: dict) -> dict:
    """One fast-lane summary per lane, summed over the bodies of a _fast_summary."""
    tot: Dict[str, dict] = {}
    for lanes in per_body.values():
        for k, v in lanes.items():
            _merge_fast(tot.setdefault(k, dict(samples=0, spikes=0)), v)
    for r in tot.values():
        n = r.get("spikes", 0)
        r["spikes_per_10000"] = 1e4 * n / r["samples"] if r.get("samples") else None
        r["mean_recovery_s"] = r.get("recovery_sum_s", 0.0) / n if n else None
    return tot


# THE FAST LANE'S ANCHORS (section 24; the style guide "anchors are overseer work"). A count of sudden
# rises in head acceleration or falls in support means nothing alone: measured 2026-09-24 on 32,000
# frames each, a limp body in this world makes 0 support falls of a quarter body weight held 30 ms,
# random-order babble 823 per 10,000 frames, the learner's bodies 722. So every fast-lane count is
# shown beside the same lanes of the babble and limp recordings (their digests' fastlane.json).
def baseline_fast(signature: Optional[str]) -> Dict[str, dict]:
    """The random-order babble and limp recordings' fast lanes summed over their bodies, or {} for
    a kind whose digest is missing or of another detector version."""
    out: Dict[str, dict] = {}
    if not signature:
        return out
    for kind in ("random_orders", "limp"):
        folder = os.path.join(SAVED, "observer", "baselines", signature, kind, "observer", "analysis")
        try:
            with open(os.path.join(folder, "state.json"), encoding="utf-8") as f:
                st = json.load(f)
            with open(os.path.join(folder, "fastlane.json"), encoding="utf-8") as f:
                fl = json.load(f)
        except (OSError, ValueError):
            continue
        # the same detectors AND the same analysis: the fast lanes' bins and spike rules are the
        # analysis's (analysis v5 changed both), so another version's lanes are no anchor
        if st.get("detector_version") != DETECTOR_VERSION or st.get("version") != ANALYSIS_VERSION:
            continue
        out[kind] = _fast_total(_fast_summary({kind: fl}))
    return out


def _roles() -> dict:
    path = os.path.join(str(ROOT), "docs", "gain_of_function_v4", "measurement_roles.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def replay_key(e: dict) -> str:
    """The replay index key of an act or episode: its episode id when it has one."""
    if e.get("episode"):
        return str(e["episode"])
    if e.get("id"):
        return str(e["id"])
    return "%s:b%s:p%s:s%s" % (e.get("run"), e.get("body"), e.get("pid"), e.get("step0"))


def _replays(run_dirs: List[str], tracked: dict, cands: List[dict], by_id: Dict[str, dict], out_path: str) -> dict:
    """Snippets and GIFs for the first observation of each capability and each novelty example,
    cut while the raw chunks still exist. Returns replay_key -> relative GIF path / status; an
    act outside every episode is replayed from its own steps."""
    folder = os.path.join(os.path.dirname(out_path), "replays")
    os.makedirs(folder, exist_ok=True)
    want = []
    for cid, row in tracked.items():
        st = (row.get("stages") or {}).get("observed_once") or {}
        ex = st.get("example")
        if ex and ex in by_id:
            want.append(by_id[ex])
        elif st.get("act"):
            want.append(st["act"])
    for c in cands:
        for eid in c.get("examples", [])[:1]:
            if eid in by_id:
                want.append(by_id[eid])
    out = {}
    made = 0
    index_path = os.path.join(folder, "index.json")
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        index = {}
    for e in want:
        key = replay_key(e)
        if key in index:
            out[key] = index[key]
            continue
        if made >= REPLAY_MAX_PER_REPORT:
            continue                         # no more new ones this report; cached ones still listed
        run = next((r for r in run_dirs if os.path.basename(r) == e.get("run")), None)
        if run is None:
            continue
        files = RunFiles(run)
        seg = [s for s in files.world_segments(int(e["body"])) if s.pid == int(e["pid"])]
        if not seg:
            index[key] = dict(ok=False, error="raw record already removed")
            continue
        chunks = []
        for c in seg[0].chunks:
            if c.first_step <= e["step1"] + RP.PAD_STEPS and c.first_step + 200 >= e["step0"] - RP.PAD_STEPS:
                try:
                    chunks.append(read_chunk(c.path))
                except Exception:
                    pass
        snip = RP.cut_snippet(chunks, int(e["step0"]), int(e["step1"]))
        if snip is None:
            index[key] = dict(ok=False, error="raw record already removed")
            continue
        safe = hashlib.sha256(key.encode()).hexdigest()[:16]
        RP.save_snippet(os.path.join(folder, safe + ".npz"), snip)
        r = RP.render(snip, files.world, os.path.join(folder, safe + ".gif"), focus_step=int(e["step0"]))
        index[key] = dict(ok=r["ok"], gif=("replays/%s.gif" % safe) if r["ok"] else None,
                          snippet="replays/%s.npz" % safe, error=r.get("error"), run=e.get("run"),
                          body=e.get("body"), step0=e.get("step0"), step1=e.get("step1"))
        out[key] = index[key]
        made += 1
    _atomic_json(index_path, index)
    return out


# ====================================================================== the service
def lineage_runs(active_run: str) -> List[str]:
    """Every run folder of the active run's lineage that has an observer folder, oldest first."""
    exp = os.path.dirname(os.path.abspath(active_run))
    name = os.path.basename(os.path.abspath(active_run))
    lineage = name.split("_seg")[0] if "_seg" in name else name
    runs = [p for p in glob.glob(os.path.join(exp, lineage + "*"))
            if os.path.isdir(os.path.join(p, "observer", "world"))]
    if os.path.isdir(os.path.join(active_run, "observer", "world")) and os.path.abspath(active_run) not in map(os.path.abspath, runs):
        runs.append(active_run)
    return sorted(runs, key=run_start_ms)


def run_start_ms(run_dir: str) -> float:
    """When a run began recording: its earliest world header's time (header_<pid>_<ms>.json);
    headers are never removed, unlike chunks. A folder's time stamp changes whenever a file in it
    does, so it cannot order runs."""
    best = None
    for p in glob.glob(os.path.join(run_dir, "observer", "world", "b*", "header_*.json")):
        try:
            ms = int(os.path.basename(p).split("_")[2].split(".")[0])
        except (IndexError, ValueError):
            continue
        best = ms if best is None else min(best, ms)
    # a run without a world header yet: its observer folder's time, in MILLISECONDS like the headers'
    # (until 2026-09-25 in seconds, which sorted such a run before every other: the digester's status
    # of 2:09 PM listed seg40_resume_20260925_030344 first, and a new run before its first header
    # would have lost 'latest' to the run before it)
    return float(best) if best is not None else 1000.0 * os.path.getmtime(os.path.join(run_dir, "observer"))


def active_run() -> Optional[str]:
    try:
        with open(os.path.join(SAVED, "training_continuation.json"), encoding="utf-8") as f:
            return json.load(f).get("active_run")
    except (OSError, ValueError):
        return None


def report_path_for(run_dir: str) -> str:
    name = os.path.basename(os.path.abspath(run_dir))
    lineage = name.split("_seg")[0] if "_seg" in name else name
    return os.path.join(SAVED, "observer", lineage, "development.json")


def heartbeat_writer(path: str, every_s: float = 10.0) -> Callable[[str, int], None]:
    """A callback the digest calls after each window: the supervisor reads the file to tell a
    long backlog (beating) from a stuck digester (silent)."""
    last = [0.0]

    def beat(run: str, body: int) -> None:
        now = time.time()
        if now - last[0] >= every_s:
            last[0] = now
            try:
                _atomic_json(path, dict(pid=os.getpid(), time=now, run=run, body=int(body)))
            except OSError:
                pass
    return beat


def serve(run: Optional[str], interval_s: float, once: bool, render: bool = True) -> None:
    from mimo_bridge.tools.training_continuation import OwnerLock
    os.makedirs(os.path.join(SAVED, "observer"), exist_ok=True)
    status_path = os.path.join(SAVED, "observer", "daemon_status.json")
    try:
        lock = OwnerLock(os.path.join(SAVED, "observer", "owner.lock"))   # one digester at a time
    except OSError:
        print("another observer digester holds Saved/mimo_bridge/observer/owner.lock; exiting", flush=True)
        return
    try:
        _serve(status_path, run, interval_s, once, render)
    finally:
        lock.close()


def _serve(status_path, run, interval_s, once, render) -> None:
    try:
        import psutil
        me = psutil.Process()
    except Exception:
        me = None
    digests: Dict[str, RunDigest] = {}
    stores: Dict[str, ST.LineageStore] = {}
    beat = heartbeat_writer(os.path.join(SAVED, "observer", "daemon_heartbeat.json"))
    last_housekeeping: Dict[str, float] = {}
    # the pins of the last pass: a pin file read before and missing now keeps them in force (pins.py, FAILING CLOSED)
    last_pins: Optional[PN.Pins] = None
    # THE ARCHIVE (archive.py; the movement viewer, Oct 18, 2026): housekeeping removes a raw chunk only after its copy
    # on the archive disk has the original's size and SHA-256
    archive = AR.Archive.configured()
    # the report's dated copy every 4 hours on the history disk (history.py); never fails the cycle
    snapshots = HI.SnapshotWriter()
    while True:
        t0 = time.time()
        status = dict(pid=os.getpid(), time=t0, ok=True)
        try:
            active = run or active_run()
            if not active:
                status.update(ok=False, note="no active run")
            else:
                runs = lineage_runs(active)
                out = report_path_for(active)
                folder = os.path.join(os.path.dirname(out), "store")
                store = stores.get(out)
                if store is None:
                    store = stores[out] = ST.LineageStore(folder, quality_map(CAT.load()))
                summaries = []
                for r in runs:
                    dg = digests.get(r)
                    if dg is None:
                        dg = digests[r] = RunDigest(r)
                        dg.beat = beat
                    if dg.sealed and dg.reopen_if_written():
                        status.setdefault("reopened", []).append(dg.run)
                    if not dg.sealed:
                        # a lineage's earlier runs are finished: their last segment is complete
                        summaries.append(dg.digest(final=os.path.abspath(r) != os.path.abspath(active)))
                    beat(dg.run, -1)
                try:
                    for r in runs:
                        store.update(digests[r].run, digests[r].out, digests[r].committed,
                                     digests[r].state.get("analysis_id"))
                except ST.StoreMismatch as why:
                    # a run's analysis was rebuilt since the store read it: the store is rebuilt
                    # from every run's committed records
                    status["store_rebuilt"] = str(why)
                    store = stores[out] = ST.LineageStore(None, quality_map(CAT.load()))
                    store.folder = folder
                    for r in runs:
                        beat(digests[r].run, -1)
                        store.update(digests[r].run, digests[r].out, digests[r].committed,
                                     digests[r].state.get("analysis_id"))
                store.save()
                rep = build_report(runs, out, digests, render_replays=render, store=store, current_run=active)
                beat("report", -1)
                status["history"] = snapshots.maybe(out, os.path.basename(os.path.dirname(out)))
                now = time.time()
                # THE EVIDENCE PINS (pins.py), read every pass: the coordinator's list of the run folders and model-step
                # ranges whose raw records housekeeping never deletes; reported every pass, an expired pin and an
                # unreadable file included (an unreadable file stops all deletion; an invalid pin stops its run's)
                pins = last_pins = PN.Pins(PN.pin_file(SAVED), previous=last_pins)
                kept = {}
                for r in runs:
                    dg = digests[r]
                    if not dg.sealed or now - last_housekeeping.get(r, 0) > 3600:
                        hk = dg.housekeeping(pins=pins, archive=archive)
                        last_housekeeping[r] = now
                        if hk.get("pins") or hk.get("blocked") or hk.get("archive_kept"):
                            kept[dg.run] = {k: hk.get(k) for k in ("pins", "pinned_kept", "command_history", "blocked",
                                                                   "removed", "archive_kept", "archive_why")}
                status["pins"] = dict(pins.summary(known_runs=runs), housekeeping=kept)
                status["archive"] = dict(root=archive.root, unavailable=archive.available())
                status.update(active=active, runs=[os.path.basename(r) for r in runs], report=out,
                              digested=summaries, detections=rep["counts"]["detections"],
                              sealed=sum(1 for r in runs if digests[r].sealed),
                              seconds=time.time() - t0)
        except Exception:
            status.update(ok=False, error=traceback.format_exc()[-4000:])
        if me is not None:
            try:
                status["memory_mb"] = me.memory_info().private / 2 ** 20 if hasattr(me.memory_info(), "private") \
                    else me.memory_info().rss / 2 ** 20
            except Exception:
                pass
        status["finished"] = time.time()          # `time` is when the pass began
        _atomic_json(status_path, _clean(status))
        if once:
            return
        time.sleep(max(5.0, interval_s - (time.time() - t0)))
