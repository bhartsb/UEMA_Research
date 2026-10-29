# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Dated copies of the Development report, and what the page reads from them (operator, 2026-09-30).

The digester overwrites development.json every cycle and keeps no history, so the page could not show
change. SnapshotWriter, called by the daemon right after each report, copies the report every EVERY_S
(4 hours) to

    <HISTORY_ROOT>\\<lineage>\\development_<YYYYmmdd_HHMM>_step<N>.json     (local time, PDT here)

and deletes copies older than KEEP_S (5 days). Only names this module writes (NAME_RE) are ever
deleted, and only in that folder. HISTORY_ROOT is D:\\UEMA_Research\\external\\D\\UEMA_observer_history (environment variable
UEMA_OBSERVER_HISTORY overrides it, for tests); D: is a USB disk that has vanished before, so a missing
disk is a logged warning and a retry, never an observer failure. The writing runs on its own thread:
the disk's first access after an idle minute has taken 10 s, and the digester must not wait for it.

index.json beside the copies keeps each copy's few page numbers (per row: occasions, babble's count in
the same time, the stages passed) so the date list and the small charts never read the whole reports.
A copy missing from the index is summarized from its own file.

A copy rebuilt later from the observer's store (rebuild.py) is named ..._rebuilt.json and says so in
its "rebuilt" field.

COUNTING RULES. Two reports compare like for like only under the same definitions, counting rules and
babble recording: the daemon's stage history logs every change as a "reset" line (daemon.stage_history),
and reports since 2026-09-30 carry the same version string themselves (rules_key). A comparison across
a reset is flagged as not like for like.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from typing import Callable, Dict, List, Optional

EVERY_S = 4 * 3600
KEEP_S = 5 * 86400
MISSING_RETRY_S = 600            # a missing disk is looked for again after this long
WARN_EVERY_S = 3600              # and its warning is logged at most this often
DEFAULT_ROOT = r"D:\UEMA_Research\external\D\UEMA_observer_history"
NAME_RE = re.compile(r"^development_(\d{8})_(\d{4})_step(\d+)(_rebuilt)?\.json$")
INDEX = "index.json"
STAGES = ["observed_once", "repeated", "predicted_before_action", "selected_intentionally", "generalized",
          "retained"]


def history_root() -> str:
    return os.environ.get("UEMA_OBSERVER_HISTORY") or DEFAULT_ROOT


def drive_present(path: str) -> bool:
    """Whether the disk holding `path` is there (a path without a drive letter: always)."""
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    if not drive:
        return True
    try:
        return os.path.isdir(drive + os.sep)
    except OSError:
        return False


def name_for(when: float, step: Optional[int], rebuilt: bool = False) -> str:
    return "development_%s_step%d%s.json" % (time.strftime("%Y%m%d_%H%M", time.localtime(when)), int(step or 0),
                                            "_rebuilt" if rebuilt else "")


def parse_name(name: str) -> Optional[dict]:
    m = NAME_RE.match(os.path.basename(name or ""))
    if not m:
        return None
    try:
        t = time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M"))
    except ValueError:
        return None
    return dict(file=os.path.basename(name), time=t, step=int(m.group(3)), rebuilt=bool(m.group(4)))


def report_step(rep: dict) -> Optional[int]:
    """The training step a report reaches: its last_step (reports since 2026-09-30), else the latest step
    its timeline, stage dates or retention blocks name (a lower bound: such a label says "or later")."""
    if rep.get("last_step") is not None:
        return int(rep["last_step"])
    steps = [int(e["step"]) for e in rep.get("timeline") or [] if e.get("step") is not None]
    for row in (rep.get("capabilities") or {}).values():
        for st in (row.get("stages") or {}).values():
            if isinstance(st, dict) and st.get("first_step") is not None:
                steps.append(int(st["first_step"]))
            for h in (st.get("history") or []) if isinstance(st, dict) else []:
                if h.get("window") is not None:
                    steps.append(3000 * int(h["window"]))
    first = (rep.get("recording") or {}).get("first_step")
    if first is not None:
        steps.append(int(first))
    return max(steps) if steps else None


def rules_key(rep: dict) -> Optional[str]:
    """The definitions, counting rules and babble a report was judged under, in the stage history's own
    form ('observer_analysis_v5/observer_detectors_v10/tracker_rules_v5/babble 1790441286.6060493');
    None for a report written before reports carried their counting rules."""
    if not rep.get("rules_version"):
        return None
    base = rep.get("baseline") or {}
    control = ("babble %s" % base.get("generated")) if base.get("available") else "no babble"
    return "%s/%s/%s/%s" % (rep.get("version"), rep.get("detector_version"), rep["rules_version"], control)


def _occasions(st: dict):
    return st.get("occasions")


def row_numbers(stages: dict) -> list:
    """[occasions, babble's occasions in the same time, stages passed as bits in STAGES order]."""
    rep, seen = stages.get("repeated") or {}, stages.get("observed_once") or {}
    occ = rep.get("occasions") if rep.get("occasions") is not None else seen.get("occasions")
    exp = rep.get("expected_by_chance") if rep.get("expected_by_chance") is not None else seen.get("expected_by_chance")
    mask = 0
    for i, k in enumerate(STAGES):
        if (stages.get(k) or {}).get("reached") is True:
            mask |= 1 << i
    return [occ, None if exp is None else round(float(exp), 2), mask]


def summary(rep: dict, name: Optional[str] = None) -> dict:
    """What the date list and the charts need of one report."""
    rows = {}
    for cid, row in (rep.get("capabilities") or {}).items():
        rows[cid] = row_numbers(row.get("stages") or {})
        for sk, sr in (row.get("subs") or {}).items():
            rows["%s|%s" % (cid, sk)] = row_numbers(sr.get("stages") or {})
    out = dict(generated=rep.get("generated"), step=report_step(rep), rules=rules_key(rep),
               rebuilt=bool(rep.get("rebuilt")), rows=rows)
    if name:
        out.update(file=os.path.basename(name))
    return out


# ====================================================================== the folder
def list_snapshots(folder: str) -> List[dict]:
    """Every copy this module wrote in `folder`, oldest first (names only: nothing is read)."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    out = [p for p in (parse_name(n) for n in names) if p]
    return sorted(out, key=lambda p: (p["time"], p["step"]))


def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def read_index(folder: str) -> dict:
    try:
        with open(os.path.join(folder, INDEX), encoding="utf-8") as f:
            idx = json.load(f)
        return idx if isinstance(idx.get("entries"), dict) else dict(schema=1, entries={})
    except (OSError, ValueError, AttributeError):
        return dict(schema=1, entries={})


def write_index(folder: str, idx: dict) -> None:
    _atomic_write(os.path.join(folder, INDEX), json.dumps(idx).encode("utf-8"))


def prune(folder: str, now: Optional[float] = None, keep_s: float = KEEP_S) -> List[str]:
    """Delete this module's copies older than keep_s (by the time in their names) and their index
    lines; nothing else in the folder is touched. Returns the names deleted."""
    now = time.time() if now is None else now
    gone = []
    try:
        for n in os.listdir(folder):          # a copy cut off while writing leaves its .part behind
            if n.endswith(".part") and (NAME_RE.match(n[:-5]) or n[:-5] == INDEX):
                full = os.path.join(folder, n)
                if now - os.path.getmtime(full) > 3600:
                    os.remove(full)
    except OSError:
        pass
    for p in list_snapshots(folder):
        if now - p["time"] > keep_s:
            try:
                os.remove(os.path.join(folder, p["file"]))
                gone.append(p["file"])
            except OSError:
                pass
    if gone:
        idx = read_index(folder)
        for n in gone:
            idx["entries"].pop(n, None)
        write_index(folder, idx)
    return gone


def write_snapshot(data: bytes, folder: str, when: Optional[float] = None, rebuilt: bool = False,
                   rep: Optional[dict] = None) -> str:
    """Write one dated copy (the report's own bytes) and its index line; returns its path."""
    rep = json.loads(data.decode("utf-8")) if rep is None else rep
    when = float(rep.get("generated") or time.time()) if when is None else when
    os.makedirs(folder, exist_ok=True)
    name = name_for(when, report_step(rep), rebuilt)
    path = os.path.join(folder, name)
    _atomic_write(path, data)
    idx = read_index(folder)
    idx["entries"][name] = summary(rep, name)
    write_index(folder, idx)
    return path


class SnapshotWriter:
    """The daemon's every-4-hours copy (module doc). maybe() is called after each report; it never
    raises and never waits for the disk: the work runs on a thread, one at a time."""

    def __init__(self, root: Optional[str] = None, every_s: float = EVERY_S, keep_s: float = KEEP_S,
                 log: Callable[[str], None] = None):
        self.root = root
        self.every_s, self.keep_s = every_s, keep_s
        self.log = log or (lambda msg: print(msg, flush=True))
        self.next_try: Dict[str, float] = {}     # lineage -> the earliest time worth trying again
        self.last_warning = float("-inf")
        self.status: dict = {}
        self.thread: Optional[threading.Thread] = None

    def maybe(self, report_path: str, lineage: str, now: Optional[float] = None, wait: bool = False) -> dict:
        try:
            now = time.time() if now is None else now
            if self.thread is not None and self.thread.is_alive():
                return self.status
            if now < self.next_try.get(lineage, 0.0):
                return self.status
            with open(report_path, "rb") as f:          # read here: the next cycle replaces the file
                data = f.read()
            self.thread = threading.Thread(target=self._work, args=(data, lineage, now), daemon=True,
                                           name="observer-history")
            self.thread.start()
            if wait:
                self.thread.join()
        except Exception:
            self.status = dict(ok=False, error=traceback.format_exc()[-800:], time=now)
        return self.status

    def _work(self, data: bytes, lineage: str, now: float) -> None:
        try:
            root = self.root or history_root()
            if not drive_present(root):
                self.next_try[lineage] = now + MISSING_RETRY_S
                self.status = dict(ok=False, time=now, note="history disk missing: %s; no copy written" % root)
                if now - self.last_warning >= WARN_EVERY_S:
                    self.last_warning = now
                    self.log("observer history: WARNING the disk for %s is not present; the 4-hourly copy of the "
                             "Development report is skipped and tried again in %d minutes" % (root, MISSING_RETRY_S // 60))
                return
            folder = os.path.join(root, lineage)
            os.makedirs(folder, exist_ok=True)
            newest = max((p["time"] for p in list_snapshots(folder) if not p["rebuilt"]), default=None)
            if newest is not None and now - newest < self.every_s:
                self.next_try[lineage] = newest + self.every_s
                self.status = dict(ok=True, time=now, note="newest copy is recent", newest=newest)
                return
            path = write_snapshot(data, folder, when=now)
            gone = prune(folder, now, self.keep_s)
            self.next_try[lineage] = now + self.every_s
            self.status = dict(ok=True, time=now, wrote=path, pruned=gone)
            self.log("observer history: wrote %s%s" % (path, "; deleted %d older than 5 days" % len(gone) if gone else ""))
        except Exception:
            self.next_try[lineage] = now + MISSING_RETRY_S
            self.status = dict(ok=False, time=now, error=traceback.format_exc()[-800:])
            if now - self.last_warning >= WARN_EVERY_S:
                self.last_warning = now
                self.log("observer history: WARNING the copy failed and is tried again in %d minutes: %s"
                         % (MISSING_RETRY_S // 60, self.status["error"].strip().splitlines()[-1]))


# ====================================================================== what the page reads
def read_resets(lineage_folder: str) -> List[dict]:
    """The stage history's 'reset' lines (a change of definitions, counting rules or babble), oldest
    first: [{time, step, before, after, withdrawn}] (withdrawn: the stages that stood before the change and not after
    it, with what each reads now; daemon.stage_history, since tracker rules v6)."""
    out = []
    try:
        with open(os.path.join(lineage_folder, "stage_history.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("change") != "reset":
                    continue
                reason = e.get("reason") or ""
                before, _, after = reason.partition(": ")[2].partition(" -> ")
                out.append(dict(time=float(e.get("time") or 0), step=e.get("step"), before=before or None,
                                after=after or None, withdrawn=list(e.get("withdrawn") or [])))
    except OSError:
        pass
    return sorted(out, key=lambda r: r["time"])


def rules_at(resets: List[dict], t: Optional[float]) -> Optional[str]:
    """The counting rules in force at time t from the stage history's resets."""
    if t is None or not resets:
        return None
    before = [r for r in resets if r["time"] <= t]
    return before[-1]["after"] if before else resets[0]["before"]


def compare_rules(a: dict, b: dict, resets: List[dict]) -> dict:
    """Whether two reports (each {time, rules}) compare like for like: the same counting rules and no
    reset between them. Returns {like, resets: [the resets between], note}."""
    lo, hi = sorted([float(a.get("time") or 0), float(b.get("time") or 0)])
    if a.get("rules") and b.get("rules"):
        # both carry their own rules (a rebuilt copy is judged under today's, whatever reset came since)
        return dict(like=a["rules"] == b["rules"], resets=[r for r in resets if lo < r["time"] <= hi]
                    if a["rules"] != b["rules"] else [], rules=(a["rules"], b["rules"]))
    between = [r for r in resets if lo < r["time"] <= hi]
    ra, rb = a.get("rules") or rules_at(resets, a.get("time")), b.get("rules") or rules_at(resets, b.get("time"))
    like = not between and (ra is None or rb is None or ra == rb)
    return dict(like=like, resets=between, rules=(ra, rb))


_SUMMARY_CACHE: Dict[tuple, dict] = {}


def load_entries(folder: str) -> List[dict]:
    """Every copy in the folder with its summary, oldest first: the index's, or the file's own (cached by
    name and size) for a copy the index lacks. Never writes."""
    idx = read_index(folder).get("entries") or {}
    out = []
    for p in list_snapshots(folder):
        s = idx.get(p["file"])
        if s is None:
            try:
                size = os.path.getsize(os.path.join(folder, p["file"]))
                key = (folder, p["file"], size)
                s = _SUMMARY_CACHE.get(key)
                if s is None:
                    with open(os.path.join(folder, p["file"]), encoding="utf-8") as f:
                        s = summary(json.load(f), p["file"])
                    _SUMMARY_CACHE[key] = s
            except (OSError, ValueError):
                continue
        out.append(dict(s, **p))
    return out


def load_report(folder: str, name: str) -> Optional[dict]:
    """One copy by its name (only names this module writes: no other path can be read)."""
    if not parse_name(name) or os.path.basename(name) != name:
        return None
    try:
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
