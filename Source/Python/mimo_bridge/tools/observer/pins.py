# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Evidence pins: the raw records of open investigations, which the digester's housekeeping never deletes.

The operator's instruction of 7:37 AM PDT, Oct 8, 2026 (relayed by the coordinator): the digester deletes a digested raw
chunk 72 hours after its file time (daemon.RAW_KEEP_S) and called its housekeeping with no pins (observer source review
F8, D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/observer/REVIEW.md); production's per-model-step records of every index_L
movement from checkpoint 481,800 were deleted that way at 5:00 PM PDT, Oct 7. The coordinator writes a pin list; the
digester reads it at every housekeeping pass. Nothing in the observer writes it.

THE FILE, PIN_FILE (Saved/mimo_bridge/observer/pins.json):

    {"pins": [{"id": "finger_index_L_v2",
               "run": "H84_fresh_drivers_20260918_seg41c_0841",
               "steps": [483773, 483822],
               "bodies": [5],
               "until": "2026-11-08T00:00:00-07:00",
               "reason": "first-finger investigation, shared episode version 2",
               "owner": "the coordinator"}]}

  id      a name, unique in the file
  run     a run folder's name (as under Saved/mimo_bridge/experiments) or its absolute path
  steps   [first, last] model steps, both included; several ranges are several pins
  bodies  the bodies whose world chunks the pin keeps (left out: every body)
  until   optional: when the pin EXPIRES (ISO 8601 with its UTC offset, or seconds since 1970). An expired pin is
          reported in every housekeeping result and in the digester's status file (daemon_status.json, `pins`), and
          KEEPS PROTECTING its records until the coordinator removes or renews it: expiry never releases evidence
  reason, owner   free text, carried into the reports
Write the file by an atomic replace (write a new file, then rename it over the old one), and remove every pin by
writing {"pins": []}: a file read once and then missing keeps the pins last read in force (FAILING CLOSED below).

WHAT A PIN KEEPS, in its run: every world chunk of its bodies whose records reach into the range, and every learner-lane
chunk whose records reach into the range as the lane numbers it. A chunk's range is read from its own `step` column
(its first and last model step): a process id the system gave a second process in the same folder restarts the
sequence numbers in the chunk names, so a neighbouring name does not bound a chunk (independent review of 4342f9194,
9:35 AM PDT, Oct 8, 2026: the chunk of 483,700-483,799 was given 483,700-483,700 and removed). An unreadable chunk is
kept from its first step on. The learner lane is one record behind the world (evidence.TRAINER_LANE_LAG: lane step t
holds the learner's records of world model step t - 1), and the efference record of the command in force at a pin's
last model step is written when the next command begins, at most one hold of 5 model steps later; so a pin [first,
last] keeps the lane chunks reaching into [first + lag, last + lag + LANE_REACH_STEPS]. The run's causal_chain.jsonl,
the command history, is never deleted by housekeeping; the pin records that the investigation needs it, and its presence
is reported. Pinned chunks still count toward their body folder's recorder budget: while a folder holds more than four
fifths of it, housekeeping removes the unpinned digested chunks, oldest first (the shared episode's 50 model steps,
483,773-483,822 of seg41c_0841, lie in two world chunks of 127-131 KB on body 5 and two learner-lane chunks of 422 KB,
against a body folder's budget of 1.25 GiB).

FAILING CLOSED. Deletion cannot be undone, so every doubt keeps records:
  - a pin file that exists and cannot be read, or whose content is not a list of pins, stops ALL deletion for that
    pass (a UTF-8 byte-order mark, as PowerShell writes, is read);
  - an INVALID entry protects instead of releasing (the same review: an `until` without its UTC offset, a second pin
    with the same id, steps written as text were reported and their chunks deleted): an entry that names a run (a
    non-empty text) stops every deletion in that run for the pass, one without a usable run stops all deletion; each
    is reported with why;
  - a pin file that was read before and is missing now keeps the pins last read in force for the digester's life,
    and says so;
  - a pin naming a run the digester does not list is reported every pass.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

PIN_FILE_NAME = "pins.json"
# The lane records a pin's last model step needs beyond the lag: the efference record of the command in force there is
# written when the next command begins, at most one hold of 5 model steps later (train_mimo --motor-hold 5 on H84);
# two holds, so a longer hold of up to 10 model steps is covered too
LANE_REACH_STEPS = 10
OPEN_END = 1 << 62
_CHUNK = re.compile(r"chunk_(\d+)_(\d+)_(\d+)\.npz$")


def pin_file(saved_mimo_bridge: str) -> str:
    """Where the digester reads the pins: <Saved>/mimo_bridge/observer/pins.json."""
    return os.path.join(saved_mimo_bridge, "observer", PIN_FILE_NAME)


def _when(v) -> Optional[float]:
    """An expiry as seconds since 1970 (epoch number, or ISO 8601 text with its offset); raises on anything else."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    t = _dt.datetime.fromisoformat(str(v))
    if t.tzinfo is None:
        raise ValueError("an expiry needs its UTC offset: %r" % v)
    return t.timestamp()


def _usable_run(e) -> Optional[str]:
    """The run an entry names, when it names one as a non-empty text; None otherwise."""
    r = e.get("run") if isinstance(e, dict) else None
    return r.strip() if isinstance(r, str) and r.strip() else None


def _same_run(named: str, run_dir: str) -> bool:
    name = os.path.basename(os.path.abspath(run_dir))
    return named == name or os.path.normcase(os.path.abspath(named)) == os.path.normcase(os.path.abspath(run_dir))


class Pins:
    """The pin list as read at one moment: `pins` (the valid ones), `invalid` [(entry, why)], `error` (the file could
    not be read: every deletion stops), `blocked_runs` {run: [why]} (an invalid entry naming the run: that run's
    deletion stops), `block_all` [why] (an invalid entry naming no usable run: every deletion stops), `missing_file`
    (the file was read before and is missing now: the pins last read stand), and the clock it was judged against
    (`now`). `previous`: the Pins of the digester's last pass (FAILING CLOSED in the module doc)."""

    def __init__(self, path: Optional[str] = None, entries: Optional[List[dict]] = None, now: Optional[float] = None,
                 previous: Optional["Pins"] = None):
        self.path = path
        self.now = time.time() if now is None else float(now)
        self.pins: List[dict] = []
        self.invalid: List[Tuple[dict, str]] = []
        self.error: Optional[str] = None
        self.blocked_runs: Dict[str, List[str]] = defaultdict(list)
        self.block_all: List[str] = []
        self.missing_file = False
        self._absent = False
        if entries is None and path is not None:
            entries = self._read(path)
            if self._absent and previous is not None and previous.entries:
                entries = list(previous.entries)
                self.missing_file = True
        self.entries: List = list(entries or [])
        seen = set()
        for e in self.entries:
            why = self._check(e, seen)
            if why:
                self.invalid.append((e, why))
                run = _usable_run(e)
                if run is not None:
                    self.blocked_runs[run].append("invalid pin %s: %s" % (
                        (e.get("id") if isinstance(e, dict) else None) or "(no id)", why))
                else:
                    self.block_all.append("invalid pin with no usable run (%s): %s" % (str(e)[:120], why))
            else:
                seen.add(str(e["id"]))
                p = dict(e, steps=[int(e["steps"][0]), int(e["steps"][1])], expires=_when(e.get("until")))
                p["expired"] = bool(p["expires"] is not None and p["expires"] < self.now)
                self.pins.append(p)

    def _read(self, path: str) -> Optional[List[dict]]:
        if not os.path.exists(path):
            self._absent = True
            return []
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
        except (OSError, ValueError) as ex:
            self.error = "the pin file %s cannot be read (%s): no raw record is deleted this pass" % (path, ex)
            return None
        pins = data.get("pins") if isinstance(data, dict) else None
        if not isinstance(pins, list):
            self.error = "the pin file %s holds no list of pins: no raw record is deleted this pass" % path
            return None
        return pins

    @staticmethod
    def _check(e, seen: set) -> Optional[str]:
        if not isinstance(e, dict):
            return "not an object"
        if not e.get("id"):
            return "no id"
        if str(e["id"]) in seen:
            return "a second pin with id %s" % e["id"]
        if not _usable_run(e):
            return "no run"
        st = e.get("steps")
        if not (isinstance(st, (list, tuple)) and len(st) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                                       for x in st) and st[0] <= st[1]):
            return "steps is not [first, last] model steps with first <= last"
        b = e.get("bodies")
        if b is not None and not (isinstance(b, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in b)):
            return "bodies is not a list of body numbers"
        try:
            _when(e.get("until"))
        except (TypeError, ValueError) as ex:
            return "until is not a time with its offset (%s)" % ex
        return None

    @property
    def blocks_all(self) -> bool:
        """The file could not be read, or an invalid entry names no usable run: housekeeping deletes nothing."""
        return self.error is not None or bool(self.block_all)

    def blocks_run(self, run_dir: str) -> List[str]:
        """Why this run's housekeeping deletes nothing this pass: the reasons of the invalid entries naming it."""
        out: List[str] = []
        for named, whys in self.blocked_runs.items():
            if _same_run(named, run_dir):
                out += whys
        return out

    def for_run(self, run_dir: str) -> List[dict]:
        """The valid pins naming this run folder (by name or by absolute path)."""
        return [p for p in self.pins if _same_run(str(p["run"]), run_dir)]

    def summary(self, known_runs: Optional[List[str]] = None) -> dict:
        """What the digester reports every pass: the file, its error, every valid pin with its state (active or
        EXPIRED, and whether its run is known), every invalid entry with why and what it blocks, and whether the pins
        last read stand in for a missing file."""
        names = {os.path.basename(os.path.abspath(r)) for r in (known_runs or [])}
        fulls = {os.path.normcase(os.path.abspath(r)) for r in (known_runs or [])}
        rows = []
        for p in self.pins:
            r = str(p["run"])
            known = None if known_runs is None else (r in names or os.path.normcase(os.path.abspath(r)) in fulls)
            rows.append(dict(id=p["id"], run=r, steps=p["steps"], bodies=p.get("bodies"), until=p.get("until"),
                             state="EXPIRED (still protecting until removed)" if p["expired"] else "active",
                             run_known=known, reason=p.get("reason"), owner=p.get("owner")))
        return dict(file=self.path, error=self.error, pins=rows,
                    expired=[r["id"] for r in rows if r["state"].startswith("EXPIRED")],
                    unknown_runs=[r["id"] for r in rows if r["run_known"] is False],
                    invalid=[dict(entry=e if isinstance(e, dict) else str(e), why=w) for e, w in self.invalid],
                    blocked_runs={k: list(v) for k, v in self.blocked_runs.items()}, block_all=list(self.block_all),
                    missing_file=("the pin file is missing; the %d entries last read stand until a file with an empty "
                                  "list is written" % len(self.entries)) if self.missing_file else None,
                    judged_at=self.now)


def _read_steps(path: str) -> Optional[Tuple[int, int]]:
    """A chunk's first and last model step, from its own `step` column (numpy's lazy npz member read)."""
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        s = np.asarray(z["step"])
    return (int(s.min()), int(s.max())) if s.size else None


def chunk_ranges(folder: str, read_steps=None) -> Dict[str, Tuple[int, int]]:
    """{chunk file name: (first step, last step)} of every chunk in a folder, each read from the chunk's own `step`
    column (`read_steps(path)` -> (first, last), default _read_steps); a chunk whose steps cannot be read keeps the first
    step of its name and an open end (it is kept by any pin from its first step on)."""
    read_steps = read_steps or _read_steps
    out: Dict[str, Tuple[int, int]] = {}
    for p in glob.glob(os.path.join(folder, "chunk_*.npz")):
        m = _CHUNK.search(os.path.basename(p))
        if not m or p.endswith(".part.npz"):
            continue
        first = int(m.group(1))
        try:
            got = read_steps(p)
        except Exception:
            got = None
        if got is None:
            out[os.path.basename(p)] = (first, OPEN_END)
        else:
            lo, hi = got
            out[os.path.basename(p)] = (min(first, int(lo)), int(hi))
    return out


def pinned_names(pins: List[dict], ranges: Dict[str, Tuple[int, int]], body: Optional[int] = None,
                 shift_lo: int = 0, shift_hi: int = 0) -> Dict[str, List[str]]:
    """{chunk name: [ids of the pins that keep it]}: a chunk is kept when its records reach into a pin's step range,
    moved by shift_lo at its first step and shift_hi at its last (the learner lane: the lag and LANE_REACH_STEPS), and
    (for a world chunk, body given) the pin names its body or every body."""
    out: Dict[str, List[str]] = {}
    for name, (a, b) in ranges.items():
        for p in pins:
            if body is not None and p.get("bodies") is not None and int(body) not in [int(x) for x in p["bodies"]]:
                continue
            lo, hi = p["steps"]
            lo, hi = int(lo) + int(shift_lo), int(hi) + int(shift_hi)
            if a <= hi and b >= lo:
                out.setdefault(name, []).append(str(p["id"]))
    return out
