# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The lineage's detections and episodes as the report reads them, small enough to hold for weeks.

Each run's analysis folder keeps the full records as gzip-compressed JSON lines
(detections.jsonl.gz, episodes.jsonl.gz), one gzip member per digestion window, so the byte
offset after any member is a safe place to resume reading. On H84 the observer finds about three
detections a second (measured 2026-09-24: 0.7 per body-step at 4.2 body-steps a second), about
1.5 KB each as JSON: held as dicts, a lineage report would outgrow memory within days and the
plain files would add about 480 MB a day to disk C:. So:

  DetTable      the few fields of each detection the tracker reads, as numpy columns (76 bytes
                a detection since store v3, 63 before): capability, sub-row, steps, body, process, run, the
                capability's main measure, provenance, the prediction and intention verdicts,
                episode and context; since store v3 the provenance of the order the act is
                credited to (its peak motion's, evidence.py), whether that is a later order than
                the one at its first step, and the touch verdict's matched no-touch moments and
                how many of them passed
  EpisodeTable  each episode's identity, place in time and its novelty runs (the distinct
                2-3 token runs the miner counts), not its full token list
  LineageStore  both, for every run of a lineage, extended from the files by byte offset and
                saved beside the lineage report, so a restarted digester parses only what is new

Nothing here decides anything: tracker.py judges the columns, novelty.py mines the runs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import zlib
from typing import Dict, Iterable, List, Optional

import numpy as np

# v3 (2026-09-24): columns cprov, credit_moved, null_n, null_k (a v2 save is rebuilt from the records)
# v4 (2026-09-25, observer statistics track): each unit's token counts and capabilities (EpisodeTable:
# bag, bag_n, n_tok, caps) for the novelty miner's order control; the wellness references' columns
# wref_n, wref_sum, wref_sq (stretches without the act) and wall_n, wall_sum, wall_sq (every stretch)
# v5 (2026-10-08, observer_attribution_v1): columns attr (how the act's writer is attributed: ATTR_*), alone (H05's
# other digits inside the act: 1 within the line, 0 beyond it, -1 not measured), sel and ach (two of an act of the
# hands alone's separate stage readings) and pred_n, pred_k (its commands judged and predicted: the Predicted stage's
# units). A v4 save is rebuilt from the record files, which lose nothing: every record keeps what it was written with
STORE_VERSION = "observer_store_v5"
# attr: an act of other parts (or of hands with other parts) credited to the order at its peak motion (evidence.credit),
# an act of the hands alone written before observer_attribution_v1 (credited to one order then: its writer is NOT
# MEASURED now), and an act of the hands alone with its orders listed (command_intervals), all of one writer class or
# of several (mixed)
ATTR_PEAK, ATTR_HAND_UNMEASURED, ATTR_ONE_CLASS, ATTR_MIXED = -2, -1, 0, 1
# The store is a cache of the record files: saving it every pass would rewrite all of it (about
# 11 MB more for each day of H84 production at the measured rate), so it is saved at most this
# often; a digester restarted in between reads the records after the saved offsets again.
SAVE_EVERY_S = 1800.0

F_SCRIPTED = 1
F_CARE_TOUCH = 2
F_CARE_PRESENT = 4
F_PRED_JUDGED = 8
F_PRED_TRUE = 16
F_INTENT_TRUE = 32
F_LIVE_RECORDED = 64
F_WORLD_MOVED = 128

FIELDS = (("cap", np.int32), ("sub", np.int32), ("step0", np.int64), ("step1", np.int64),
          ("body", np.int16), ("pid", np.int64), ("run", np.int32), ("measure", np.float32),
          ("flags", np.uint8), ("prov", np.int32), ("pred_kind", np.int32), ("episode", np.int32),
          ("context", np.int32), ("wellness", np.float32),
          # the credited order's provenance (evidence.py credit; the first step's when unrecorded),
          # whether it is another order than the first step's (-1 unmeasured, 0 the same, 1 later, 2
          # earlier: a hand act peaking on its first record since analysis v5),
          # and the touch verdict's matched no-touch moments and passes (evidence.contact_nulls)
          ("cprov", np.int32), ("credit_moved", np.int8), ("null_n", np.int32), ("null_k", np.int32),
          # since store v4: the wellness change's reference stretches given to the act (evidence.
          # wellness_references): how many, the sum and the sum of squares of their summed changes
          ("wref_n", np.int32), ("wref_sum", np.float32), ("wref_sq", np.float32),
          # ... and over every stretch of the act's length, act or not (the unselected reference)
          ("wall_n", np.int32), ("wall_sum", np.float32), ("wall_sq", np.float32),
          # since store v5: how the act's writer is attributed (ATTR_*) and H05's other digits inside the act; for an
          # act of the hands alone, two of its four separate stages (evidence.Joiner.four_stages): planner selected (0
          # a command was not the planner's; -1 not judged, as no noise band of the goal distance is measured, or not
          # measured) and achieved (1 yes, 0 no, -1 not measured); and its movement prediction's units, the commands
          # judged and the commands predicted (evidence.movement_prediction_intervals; 0 and 0 for every other act,
          # whose one verdict is in `flags`)
          ("attr", np.int8), ("alone", np.int8), ("sel", np.int8), ("ach", np.int8),
          ("pred_n", np.int16), ("pred_k", np.int16))


def _tri(v) -> int:
    return -1 if v is None else int(bool(v))


def attribution_code(d: dict) -> int:
    """ATTR_* of one detection (store v5): from its evidence.attribution when the join listed its orders, else by
    whether it is an act of the hands alone (evidence.Joiner.is_hand_act)."""
    att = (d.get("evidence") or {}).get("attribution")
    if att and att.get("intervals"):
        return ATTR_MIXED if att.get("writer_class") == "mixed" else ATTR_ONE_CLASS
    from .evidence import Joiner
    try:
        hand = Joiner.is_hand_act(d)
    except (KeyError, TypeError):
        hand = False
    return ATTR_HAND_UNMEASURED if hand else ATTR_PEAK


class Strings:
    """An append-only table of the strings the columns index (-1 is None)."""

    def __init__(self, items: Iterable[str] = ()):
        self.items: List[str] = list(items)
        self.index: Dict[str, int] = {s: i for i, s in enumerate(self.items)}

    def id(self, s) -> int:
        if s is None:
            return -1
        s = str(s)
        i = self.index.get(s)
        if i is None:
            i = self.index[s] = len(self.items)
            self.items.append(s)
        return i

    def get(self, i: int):
        i = int(i)
        return None if i < 0 else self.items[i]

    def ids(self, values) -> np.ndarray:
        return np.asarray([self.index.get(str(v), -2) for v in values], dtype=np.int64)


def context_key(d: dict) -> str:
    """What makes two acts' starting situations different (tracker: generalized): the posture at
    the start and the objects involved. Not the act's own repetition key: bend and straighten of
    one finger are two keys but one situation, and counted as two they made every repeated row
    "generalized" at once (2026-09-24 review)."""
    return json.dumps([d.get("posture"), sorted(d.get("objects") or [])])


class DetTable:
    """Detections as columns. `quality` maps a capability id to the name of its main measure
    (catalog 'quality'); a detection's `measure` is that measure's value (NaN when absent)."""

    def __init__(self, strings: Optional[Strings] = None, quality: Optional[Dict[str, str]] = None):
        self.s = strings if strings is not None else Strings()
        self.quality = dict(quality or {})
        self.cols = {k: np.zeros(0, dtype=t) for k, t in FIELDS}
        self._rows: List[tuple] = []

    def __len__(self) -> int:
        return len(self.cols["cap"]) + len(self._rows)

    def add(self, d: dict) -> None:
        s = self.s
        ev = d.get("evidence") or {}
        pred = ev.get("prediction") or {}
        intent = ev.get("intention") or {}
        flags = 0
        flags |= F_SCRIPTED if d.get("scripted") else 0
        flags |= F_CARE_TOUCH if d.get("caregiver_touch") else 0
        flags |= F_CARE_PRESENT if d.get("caregiver_present") else 0
        if pred.get("verdict") is not None:
            flags |= F_PRED_JUDGED
            flags |= F_PRED_TRUE if pred.get("verdict") else 0
        flags |= F_INTENT_TRUE if intent.get("verdict") else 0
        flags |= F_LIVE_RECORDED if d.get("live", True) else 0
        flags |= F_WORLD_MOVED if d.get("world_moved") else 0
        well = (ev.get("wellness") or {}).get("total_change")
        try:
            well = float(well) if well is not None else float("nan")
        except (TypeError, ValueError):
            well = float("nan")
        name = self.quality.get(d["cap"])
        v = (d.get("measures") or {}).get(name) if name else None
        try:
            measure = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            measure = float("nan")
        credit = ev.get("credit") or {}
        moved = credit.get("moved") if credit.get("measured", True) else None
        seq, first = credit.get("order_seq"), credit.get("first_order_seq")
        if moved and seq is not None and first is not None and int(seq) < int(first):
            moved = 2                                   # an earlier order: the hands' and arms' lag
        wl = ev.get("wellness") or {}
        alone = d.get("alone_inside")
        stg = ev.get("stages") or {}
        self._rows.append((s.id(d["cap"]), s.id(d.get("sub") or None), int(d["step0"]),
                           int(d.get("step1", d["step0"])), int(d.get("body", 0)), int(d.get("pid", 0)),
                           s.id(d.get("run")), measure, flags, s.id(d.get("provenance")),
                           s.id(pred.get("kind")), s.id(d.get("episode")), s.id(context_key(d)), well,
                           s.id(credit.get("provenance") or d.get("provenance")),
                           -1 if moved is None else (2 if moved == 2 else int(bool(moved))),
                           int(pred.get("null_moments") or 0), int(pred.get("null_passed") or 0),
                           int(wl.get("reference_n") or 0), float(wl.get("reference_sum") or 0.0),
                           float(wl.get("reference_sq") or 0.0),
                           int(wl.get("all_n") or 0), float(wl.get("all_sum") or 0.0),
                           float(wl.get("all_sq") or 0.0),
                           attribution_code(d), -1 if alone is None else int(bool(alone)),
                           _tri((stg.get("planner_selected") or {}).get("verdict")),
                           _tri((stg.get("achieved") or {}).get("verdict")),
                           min(int(pred.get("commands_judged") or 0), 32767),
                           min(int(pred.get("commands_predicted") or 0), 32767)))

    def extend(self, dets: Iterable[dict]) -> None:
        for d in dets:
            self.add(d)

    def freeze(self) -> "DetTable":
        if self._rows:
            block = list(zip(*self._rows))
            for (k, t), vals in zip(FIELDS, block):
                self.cols[k] = np.concatenate([self.cols[k], np.asarray(vals, dtype=t)])
            self._rows = []
        return self

    def __getitem__(self, k: str) -> np.ndarray:
        self.freeze()
        return self.cols[k]

    @classmethod
    def from_dicts(cls, dets: List[dict], quality=None) -> "DetTable":
        """A table of plain detection dicts; `quality` is a capability->measure map, or one
        measure name used for every capability (the tests' single-capability lists)."""
        if isinstance(quality, str):
            names = {d["cap"]: quality for d in dets}
        else:
            names = dict(quality or {})
        t = cls(quality=names)
        t.extend(dets)
        return t.freeze()

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        self.freeze()
        tmp = path + ".part.npz"
        np.savez(tmp, **self.cols)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with np.load(path) as z:
            for k, t in FIELDS:
                self.cols[k] = z[k].astype(t) if k in z else np.zeros(0, dtype=t)
        self._rows = []


class EpisodeTable:
    """Each episode's identity and novelty runs; the miner's input (novelty.mine_table).
    `tokens` maps a run id to the run's tokens, for naming a candidate. Since store v4 each unit also
    keeps what a shuffle of its tokens keeps, for the miner's order control (novelty.order_test):
    `bag` (its tokens' string ids), `bag_n` (how many times each), `n_tok` (their total) and `caps`
    (the capabilities detected in it, as string ids: a run they explain never counts in it)."""

    def __init__(self, strings: Optional[Strings] = None):
        self.s = strings if strings is not None else Strings()
        self.rows: List[dict] = []          # id, run, body, pid, step0, step1, t0, duration_s,
        #                                     caregiver_touch, caregiver_present, author, runs,
        #                                     bag, bag_n, n_tok, caps
        self.tokens: Dict[str, List[str]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, e: dict, runs: Optional[List[tuple]] = None) -> None:
        from .novelty import episode_runs, candidate_id
        from .detectors import AUTHOR_NAMES
        if runs is None:
            runs = episode_runs(e.get("tokens") or [], e.get("caps") or [])
        ids = []
        for r in runs:
            rid = candidate_id(r)
            if rid not in self.tokens:
                self.tokens[rid] = list(r)
            ids.append(self.s.id(rid))
        from .novelty import token_bag
        bag = token_bag(e.get("tokens") or [])
        authors = e.get("authors") or {}
        top = max(authors.items(), key=lambda kv: kv[1])[0] if authors else 0
        self.rows.append(dict(id=e["id"], run=e.get("run"), body=int(e["body"]), pid=int(e.get("pid", 0)),
                              step0=int(e["step0"]), step1=int(e["step1"]), t0=float(e.get("t0", 0.0)),
                              duration_s=float(e.get("duration_s", 0.0)),
                              caregiver_touch=bool(e.get("caregiver_touch")),
                              caregiver_present=bool(e.get("caregiver_present")),
                              toy_held=bool(e.get("toy_held")),
                              author=AUTHOR_NAMES.get(int(top), "unknown"),
                              runs=np.asarray(ids, dtype=np.int32),
                              bag=np.asarray([self.s.id(t) for t in bag], dtype=np.int32),
                              bag_n=np.asarray(list(bag.values()), dtype=np.int32),
                              n_tok=int(sum(bag.values())),
                              caps=np.asarray([self.s.id(c) for c in (e.get("caps") or [])], dtype=np.int32)))

    _ARRAYS = ("runs", "bag", "bag_n", "caps")

    def save(self, path: str) -> None:
        rows = [dict(r, **{k: r[k].tolist() for k in self._ARRAYS}) for r in self.rows]
        _atomic_text(path, json.dumps(dict(rows=rows, tokens=self.tokens)))

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.rows = [dict(r, **{k: np.asarray(r.get(k) or [], dtype=np.int32) for k in self._ARRAYS})
                     for r in data["rows"]]
        self.tokens = data.get("tokens") or {}


# ====================================================================== the record files
def _atomic_text(path: str, text: str) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def append_records(path: str, rows: List[dict], dumps) -> None:
    """One gzip member of JSON lines appended to `path` (a no-op for no rows)."""
    if not rows:
        return
    data = "".join(dumps(r) + "\n" for r in rows).encode("utf-8")
    with open(path, "ab") as f:
        f.write(gzip.compress(data, compresslevel=6))
        f.flush()
        os.fsync(f.fileno())


def read_records(path: str, start: int = 0, end: Optional[int] = None):
    """(the JSON lines of a record file from byte `start` - a member boundary - up to `end`,
    the bytes of complete members read). A torn last member (a pass cut off while writing) is
    left unread; plain .jsonl files (written before 2026-09-24) are read up to their last full
    line."""
    try:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read() if end is None else f.read(max(0, end - start))
    except OSError:
        return [], 0
    if not data:
        return [], 0
    if path.endswith(".gz"):
        try:
            text = gzip.decompress(data)             # every member whole: the usual case, at C speed
            used = len(data)
        except (OSError, EOFError, zlib.error):
            text, used = _whole_members(data)
    else:
        cut = data.rfind(b"\n") + 1
        text, used = data[:cut], cut
    out = []
    for line in text.decode("utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out, used


def _whole_members(data: bytes):
    """The complete gzip members at the start of data (a torn last member is left)."""
    mv = memoryview(data)
    parts, pos = [], 0
    while pos < len(data):
        d = zlib.decompressobj(wbits=31)
        try:
            chunk = d.decompress(mv[pos:])
        except zlib.error:
            break
        if not d.eof:
            break
        parts.append(chunk)
        pos = len(data) - len(d.unused_data)
    return b"".join(parts), pos


def record_file(folder: str, name: str) -> str:
    """The run's record file for `name` ('detections' or 'episodes'): the compressed file, or the
    plain one an older digest left."""
    gz = os.path.join(folder, name + ".jsonl.gz")
    plain = os.path.join(folder, name + ".jsonl")
    if os.path.exists(gz) or not os.path.exists(plain):
        return gz
    return plain


def size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def truncate(path: str, length: int) -> bool:
    """Cut a record file back to `length` bytes (the members after the last saved progress were
    written by a pass that did not finish; that pass's windows are digested again)."""
    try:
        if os.path.getsize(path) > length:
            with open(path, "r+b") as f:
                f.truncate(length)
            return True
    except OSError:
        pass
    return False


# ====================================================================== the lineage store
class LineageStore:
    """Every run's detections and episodes of one lineage, read incrementally from each run's
    record files up to the size its digest has committed."""

    def __init__(self, folder: Optional[str], quality: Dict[str, str]):
        self.folder = folder
        self.strings = Strings()
        self.quality = dict(quality)
        self.dets = DetTable(self.strings, self.quality)
        self.eps = EpisodeTable(self.strings)
        self.offsets: Dict[str, Dict[str, int]] = {}
        self._saved_at = 0.0
        self.qhash = hashlib.sha256(json.dumps(sorted(self.quality.items())).encode()).hexdigest()[:16]
        if folder:
            self._load()

    def _meta_path(self) -> str:
        return os.path.join(self.folder, "store_meta.json")

    def _data_paths(self, gen: int):
        return (os.path.join(self.folder, "store_dets_%d.npz" % gen),
                os.path.join(self.folder, "store_episodes_%d.json" % gen))

    def _load(self) -> None:
        """The last complete save: store_meta.json names the generation of the data files it
        was written with, so a save cut off before its meta leaves the previous one in force."""
        try:
            with open(self._meta_path(), encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("version") != STORE_VERSION or meta.get("quality_hash") != self.qhash:
                return                       # rebuilt from the record files
            dets_p, eps_p = self._data_paths(int(meta["generation"]))
            self.strings.items[:] = meta["strings"]
            self.strings.index = {s: i for i, s in enumerate(self.strings.items)}
            self.dets.load(dets_p)
            self.eps.load(eps_p)
            self.offsets = meta["offsets"]
            self.generation = int(meta["generation"])
        except Exception:                        # any damaged save: rebuilt from the record files
            self.strings.__init__()
            self.dets = DetTable(self.strings, self.quality)
            self.eps = EpisodeTable(self.strings)
            self.offsets = {}

    def save(self, force: bool = False) -> bool:
        if not self.folder or (not force and time.time() - self._saved_at < SAVE_EVERY_S):
            return False
        self._saved_at = time.time()
        os.makedirs(self.folder, exist_ok=True)
        # after the highest generation on disk: a store rebuilt in memory starts at none, and
        # must not overwrite the data files the meta on disk still names
        on_disk = [int(n.split("_")[-1].split(".")[0]) for n in os.listdir(self.folder)
                   if n.startswith(("store_dets_", "store_episodes_")) and n.split("_")[-1].split(".")[0].isdigit()]
        gen = max([getattr(self, "generation", 0)] + on_disk) + 1
        dets_p, eps_p = self._data_paths(gen)
        self.dets.save(dets_p)
        self.eps.save(eps_p)
        _atomic_text(self._meta_path(), json.dumps(dict(version=STORE_VERSION, quality_hash=self.qhash,
                                                        generation=gen, strings=self.strings.items,
                                                        offsets=self.offsets, saved=time.time())))
        self.generation = gen
        for name in os.listdir(self.folder):
            if name.startswith(("store_dets_", "store_episodes_")) and not name.startswith(
                    ("store_dets_%d." % gen, "store_episodes_%d." % gen)):
                try:
                    os.remove(os.path.join(self.folder, name))
                except OSError:
                    pass
        return True

    def update(self, run: str, analysis_dir: str, committed: Optional[Dict[str, int]] = None,
               analysis_id: Optional[str] = None) -> int:
        """Read what run `run` committed since the last update; returns the records read. The
        store is inconsistent, and says so (StoreMismatch: the caller rebuilds it), when the run's
        analysis was rebuilt since it read it: another analysis id, a committed size below the
        stored offset, or bytes at the stored offset that are no member start."""
        n = 0
        off = self.offsets.setdefault(run, {})
        if analysis_id is not None:
            known = off.get("analysis_id")
            if known is not None and known != analysis_id:
                raise StoreMismatch("run %s was digested again (analysis %s, the store read %s)"
                                    % (run, analysis_id, known))
            off["analysis_id"] = analysis_id
        for name in ("detections", "episodes"):
            path = record_file(analysis_dir, name)
            end = size(path) if committed is None else int(committed.get(os.path.basename(path), size(path)))
            start = int(off.get(os.path.basename(path), 0))
            if end < start:
                raise StoreMismatch("%s shrank below what the store read (%d < %d)" % (path, end, start))
            if end == start:
                continue
            rows, used = read_records(path, start, end)
            if used == 0 and committed is not None:
                raise StoreMismatch("%s holds no whole record at the store's offset %d" % (path, start))
            if name == "detections":
                for d in rows:
                    d.setdefault("run", run)
                    self.dets.add(d)
            else:
                for e in rows:
                    e.setdefault("run", run)
                    self.eps.add(e)
            off[os.path.basename(path)] = start + used
            n += len(rows)
        self.dets.freeze()
        return n


class StoreMismatch(RuntimeError):
    pass
