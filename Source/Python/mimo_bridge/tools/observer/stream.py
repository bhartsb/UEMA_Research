# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Layer 1, the raw world stream: which recorded chunks exist, in what order, and which are live.

A run folder holds <run>/observer/world/b<body>/chunk_<first step>_<pid>_<seq>.npz (one per 100
model steps by default) with the header JSON each chunk names, and <run>/observer/model/ the
learner's lane. One trainer PROCESS is one SEGMENT; a relaunch starts a new segment in the same
folder. When a relaunch resumed from an earlier checkpoint than where the previous process got
to (a crash resumes from the last hourly save), the previous process's steps from that point on
are an ABANDONED FUTURE: the learner that produced them was rolled back with the checkpoint, so
nothing it did then belongs to the developing model. They are kept on disk but never counted.

Order of segments: by the creation time of the process's first header (header_<pid>_<ms>.json).
Within a segment: by the chunk's sequence number.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_CHUNK = re.compile(r"chunk_(\d+)_(\d+)_(\d+)\.npz$")
_HEADER = re.compile(r"header_(\d+)_(\d+)\.json$")


@dataclass
class ChunkRef:
    path: str
    first_step: int
    pid: int
    seq: int
    start_ms: int = 0              # the segment's (process's) first header time, for ordering

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class Segment:
    pid: int
    start_ms: int
    chunks: List[ChunkRef] = field(default_factory=list)

    @property
    def first_step(self) -> int:
        return min(c.first_step for c in self.chunks) if self.chunks else 1 << 62


class RunFiles:
    def __init__(self, run_dir: str):
        self.run = os.path.abspath(run_dir)
        self.root = os.path.join(self.run, "observer")
        self.world = os.path.join(self.root, "world")
        self.model = os.path.join(self.root, "model")
        self.analysis = os.path.join(self.root, "analysis")

    def exists(self) -> bool:
        return os.path.isdir(self.world)

    def bodies(self) -> List[int]:
        out = []
        for p in glob.glob(os.path.join(self.world, "b*")):
            m = re.fullmatch(r"b(\d+)", os.path.basename(p))
            if m and os.path.isdir(p):
                out.append(int(m.group(1)))
        return sorted(out)

    def body_dir(self, body: int) -> str:
        return os.path.join(self.world, "b%d" % int(body))

    @staticmethod
    def _starts(folder: str) -> Dict[int, int]:
        """pid -> the millisecond time of that process's first header in the folder."""
        starts: Dict[int, int] = {}
        for p in glob.glob(os.path.join(folder, "header_*.json")):
            m = _HEADER.search(os.path.basename(p))
            if m:
                pid, ms = int(m.group(1)), int(m.group(2))
                starts[pid] = min(starts.get(pid, ms), ms)
        return starts

    def _segments(self, folder: str) -> List[Segment]:
        starts = self._starts(folder)
        segs: Dict[int, Segment] = {}
        for p in glob.glob(os.path.join(folder, "chunk_*.npz")):
            m = _CHUNK.search(os.path.basename(p))
            if not m or p.endswith(".part.npz"):
                continue
            first, pid, seq = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start = starts.get(pid, 0)
            seg = segs.setdefault(pid, Segment(pid=pid, start_ms=start))
            seg.chunks.append(ChunkRef(p, first, pid, seq, start))
        for s in segs.values():
            s.chunks.sort(key=lambda c: c.seq)
        return sorted(segs.values(), key=lambda s: (s.start_ms, s.pid))

    def world_segments(self, body: int) -> List[Segment]:
        return self._segments(self.body_dir(body))

    def model_segments(self) -> List[Segment]:
        return self._segments(self.model)

    def gaps(self, body: Optional[int] = None) -> List[dict]:
        path = os.path.join(self.body_dir(body) if body is not None else self.model, "gaps.jsonl")
        out = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out


def abandoned_from(segments: List[Segment]) -> Dict[int, int]:
    """pid -> the step from which that segment's records are an abandoned future: the first step
    of any LATER segment. A resume from checkpoint N starts its records at step N (train_mimo
    sets step = N and the first burst carries it), so an earlier process's records at N and
    beyond were rolled back with the learner. A later segment that starts past this one's last
    step abandons nothing, and the cut then removes nothing."""
    out: Dict[int, int] = {}
    for i, s in enumerate(segments):
        later = [t.first_step for t in segments[i + 1:] if t.chunks]
        if later:
            out[s.pid] = min(later)
    return out


def live_mask(steps: np.ndarray, pid: int, cuts: Dict[int, int]) -> np.ndarray:
    """True for the records of segment `pid` that were not abandoned by a later resume."""
    cut = cuts.get(int(pid))
    steps = np.asarray(steps)
    return np.ones(len(steps), dtype=bool) if cut is None else steps < cut


# ---------------------------------------------------------------------- model lane by step
class ModelLaneIndex:
    """The learner's per-step lane, indexed by step, for one run (live records only).

    Incremental: update() reads only chunks it has not read; rows from an earlier process past
    a later process's first step are removed (an abandoned future); prune() forgets old steps."""

    def __init__(self, files: RunFiles):
        self.files = files
        self.rows: Dict[int, dict] = {}
        self.contact: Dict[int, dict] = {}
        self.eff: Dict[int, dict] = {}
        self.licenses: List[tuple] = []          # (step, the planner's per-depth licence)
        self.wellness_names: List[str] = []
        self.senses: List[str] = []
        self.handles: List[str] = []
        self.read: set = set()
        self.max_step = -1
        self.update()

    def update(self) -> int:
        from mimo_bridge.observer_model_lane import read_model_chunk
        files = self.files
        segs = files.model_segments()
        cuts = abandoned_from(segs)
        hdr = sorted(glob.glob(os.path.join(files.model, "header_*.json")))
        if hdr and not self.senses:
            try:
                with open(hdr[-1], encoding="utf-8") as f:
                    h = json.load(f)
                self.senses = h.get("senses", [])
                self.handles = h.get("handle_names", [])
            except (OSError, ValueError):
                pass
        new = 0
        for seg in segs:
            cut = cuts.get(seg.pid, 1 << 62)
            for c in seg.chunks:
                if c.path in self.read:
                    continue
                try:
                    z = read_model_chunk(c.path)
                except Exception:
                    continue
                self.read.add(c.path)
                new += 1
                names = [str(x) for x in z.get("wellness_names", [])]
                for w in names:
                    if w not in self.wellness_names:
                        self.wellness_names.append(w)
                for i, st in enumerate(z["step"]):
                    if st >= cut:
                        continue
                    row = dict(pid=seg.pid, author=int(z["author"][i]), latched=bool(z["latched"][i]),
                               next_pe=z["next_pe"][i], next_pe_body=z["next_pe_body"][i])
                    if names:
                        row["wellness"] = {w: z["wellness"][i, j] for j, w in enumerate(names)}
                    self.rows[int(st)] = row
                    self.max_step = max(self.max_step, int(st))
                if "contact_step" in z:
                    for i, st in enumerate(z["contact_step"]):
                        if st < cut:
                            self.contact[int(st)] = dict(pid=seg.pid, q=z["contact_q"][i],
                                                         base=z["contact_base"][i], valid=z["contact_valid"][i])
                if "license_json" in z:
                    for txt in z["license_json"]:
                        try:
                            rec = json.loads(str(txt))
                        except ValueError:
                            continue
                        if int(rec.get("step", -1)) < cut:
                            self.licenses.append((int(rec.get("step", -1)), rec.get("license") or {}))
                if "eff_step" in z:
                    for i, st in enumerate(z["eff_step"]):
                        if st < cut:
                            rec = dict(pid=seg.pid, began=int(z["eff_began"][i]), exp_cm=z["eff_exp_cm"][i],
                                       exp_deg=z["eff_exp_deg"][i], ach_cm=z["eff_ach_cm"][i],
                                       ach_deg=z["eff_ach_deg"][i])
                            if "eff_sh_cm" in z:
                                rec.update(sh_cm=z["eff_sh_cm"][i], sh_deg=z["eff_sh_deg"][i])
                            self.eff[int(st)] = rec
        # a later process abandons an earlier one's steps from its own first step on
        for seg in segs:
            cut = cuts.get(seg.pid)
            if cut is None:
                continue
            for table in (self.rows, self.contact, self.eff):
                for st in [s for s, r in table.items() if r.get("pid") == seg.pid and s >= cut]:
                    del table[st]
        self.licenses.sort(key=lambda t: t[0])
        return new

    def prune(self, before_step: int) -> None:
        for table in (self.rows, self.contact, self.eff):
            for st in [s for s in table if s < before_step]:
                del table[st]


def read_causal_chain(run_dir: str, max_records: int = 200000) -> List[dict]:
    """The planner's latch records (<run>/causal_chain.jsonl, --causal-chain on)."""
    path = os.path.join(run_dir, "causal_chain.jsonl")
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
                if len(out) >= max_records:
                    break
    except OSError:
        pass
    return out
