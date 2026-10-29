# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The stored per-command measures of every recording body (commands.py), and what the dashboard's chart reads.

STORAGE: Saved/mimo_bridge/movement_viewer/index/<recording id>/b<body>.npz (the command rows) and b<body>.json (the
chunk names they were computed from, the index version, the model fingerprint, the handle names). A recording that grew
(a live run) is updated from three chunks before its first new chunk on, so the commands at the old end get their
after-next values; any other change of its chunk list rebuilds that body. A chunk that disappeared from every copy keeps
its commands' rows (their measures were computed from it) and is listed as no longer saved.

QUERIES (series, command_at, pick): the commands of the chosen recordings and body (or every body), author (the
planner's commands, author code 1, or every author) and time range, with a value at the command's end for the chosen
measure, sorted by time.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from . import INDEX_DIR, part_path, replace_file
from . import commands as C

AUTHOR_NAMES = {0: "unknown", 1: "planner", 2: "reflex", 3: "babble noise", 4: "self choice", 5: "acting snapshot",
                6: "scheduled rest", 7: "probe", 8: "script", 9: "random actions", 10: "limp", 11: "yoked replay"}
TAIL_CHUNKS = 3


def _paths(rid: str, body: int):
    folder = INDEX_DIR / rid
    return folder / ("b%d.npz" % int(body)), folder / ("b%d.json" % int(body))


def read_meta(rid: str, body: int) -> Optional[dict]:
    try:
        with open(_paths(rid, body)[1], encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(rid: str, body: int, cols: Dict[str, np.ndarray], meta: dict) -> None:
    npz, js = _paths(rid, body)
    os.makedirs(npz.parent, exist_ok=True)
    tmp = str(npz)[:-4] + ".%d.part.npz" % os.getpid()
    np.savez_compressed(tmp, **cols)
    replace_file(tmp, npz)
    tmpj = part_path(js)
    with open(tmpj, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    replace_file(tmpj, js)


def _load_cols(rid: str, body: int) -> Optional[Dict[str, np.ndarray]]:
    npz = _paths(rid, body)[0]
    if not npz.is_file():
        return None
    with np.load(npz, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def update_body(rec: dict, body: int) -> dict:
    """Bring one recording body's index up to date with its chunk list. Returns what was done."""
    b = rec["bodies"].get(str(body))
    if not b:
        return dict(body=body, done="no chunks")
    chunks = b["chunks"]
    names = [c[0] for c in chunks]
    meta = read_meta(rec["id"], body)
    old = None
    if meta and meta.get("version") == C.INDEX_VERSION:
        known = meta.get("chunks") or []
        if known == names:
            return dict(body=body, done="unchanged", commands=meta.get("commands"))
        if set(names) < set(known):
            # chunks gone from every copy: the rows stay (computed when the chunks existed)
            meta["no_longer_saved"] = sorted(set(known) - set(names))
            meta["checked_at"] = time.time()
            js = _paths(rec["id"], body)[1]
            tmpj = part_path(js)
            with open(tmpj, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            replace_file(tmpj, js)
            return dict(body=body, done="chunks no longer saved", missing=len(meta["no_longer_saved"]))
        if names[:len(known)] == known and len(known) >= TAIL_CHUNKS:
            old = _load_cols(rec["id"], body)
    t0 = time.time()
    if old is not None:
        start = len(meta["chunks"]) - TAIL_CHUNKS
        load = chunks[start:]
        keep_from = int(load[1][3]) if len(load) > 1 else int(load[0][3])
    else:
        start, load, keep_from = 0, chunks, None
    rec_data = C.load_records(load)
    header = rec_data["headers"][0]
    model = C.BodyModel(b["model"], header)
    other = [i for i, h in enumerate(rec_data["headers"]) if h.get("model_fingerprint") != header.get("model_fingerprint")]
    skipped = sorted(str(load[i][0]) for i in other)
    if other:
        # a chunk recorded with another model is not scored with this one: its records start no command
        rec_data["order_on"] = np.asarray(rec_data["order_on"]) & ~np.isin(np.asarray(rec_data["chunk"]), other)
    cols = C.measure(rec_data, model, (rec.get("args") or {}).get("position_rows"), keep_from_step=keep_from)
    cols["chunk"] = (cols["chunk"] + start).astype(np.int32)
    if old is not None:
        keep = old["step"] < keep_from
        cols = {k: np.concatenate([old[k][keep], cols[k]]) for k in cols}
    new_meta = dict(version=C.INDEX_VERSION, recording=rec["id"], run=rec["run"], body=int(body), chunks=names,
                    handle_names=model.names, model_fingerprint=model.fingerprint,
                    position_rows=(rec.get("args") or {}).get("position_rows"), commands=int(len(cols["step"])),
                    first_step=int(cols["step"].min()) if len(cols["step"]) else None,
                    last_step=int(cols["step"].max()) if len(cols["step"]) else None,
                    first_time=float(cols["time"].min()) if len(cols["time"]) else None,
                    last_time=float(cols["time"].max()) if len(cols["time"]) else None,
                    computed_at=time.time(), seconds=round(time.time() - t0, 3), other_model_chunks_not_scored=skipped,
                    joint_names=model.joint_names)
    _write(rec["id"], body, cols, new_meta)
    _CACHE.pop((rec["id"], int(body)), None)
    return dict(body=body, done="tail" if old is not None else "full", commands=new_meta["commands"],
                seconds=new_meta["seconds"])


def update_recording(rec: dict) -> List[dict]:
    out = []
    for body in sorted(int(k) for k in rec["bodies"]):
        try:
            out.append(update_body(rec, body))
        except C.Refused as why:
            out.append(dict(body=body, done="refused", why=str(why)))
    return out


# ------------------------------------------------------------------ reading (the dashboard process)
_CACHE: Dict[tuple, tuple] = {}
_LOCK = threading.Lock()


def body_cols(rid: str, body: int) -> Optional[Dict[str, np.ndarray]]:
    """One recording body's rows, cached in this process until its file changes."""
    npz = _paths(rid, body)[0]
    try:
        mt = os.path.getmtime(npz)
    except OSError:
        return None
    key = (rid, int(body))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and hit[0] == mt:
            return hit[1]
    cols = _load_cols(rid, body)
    with _LOCK:
        _CACHE[key] = (mt, cols)
    return cols


def selection(recordings: List[dict], body, measure: str, author: str, t0: Optional[float] = None,
              t1: Optional[float] = None, touch: Optional[str] = None) -> Dict[str, np.ndarray]:
    """The matching commands, sorted by time: time, step, body, recording number (into `recordings`), author, and the
    measure's start, end and after-next values. Commands without an end value for the measure are left out. touch
    'hand': only commands during which a hand part touched an object (saved contacts)."""
    if measure not in C.MEASURES:
        raise ValueError("unknown measure %r" % measure)
    parts = []
    for ri, rec in enumerate(recordings):
        bodies = sorted(int(k) for k in rec["bodies"]) if body in (None, "all") else [int(body)]
        for b in bodies:
            cols = body_cols(rec["id"], b)
            if cols is None or not len(cols["step"]):
                continue
            e = cols["m_%s_e" % measure]
            keep = np.isfinite(e)
            if author == "planner":
                keep &= cols["author"] == 1
            elif author not in (None, "all"):
                keep &= cols["author"] == int(author)
            if t0 is not None:
                keep &= cols["time"] >= float(t0)
            if t1 is not None:
                keep &= cols["time"] <= float(t1)
            if touch == "hand" and "hand_touch_L" in cols:
                keep &= cols["hand_touch_L"] | cols["hand_touch_R"]
            if not keep.any():
                continue
            parts.append(dict(time=cols["time"][keep], step=cols["step"][keep],
                              body=np.full(int(keep.sum()), b, dtype=np.int16),
                              rec=np.full(int(keep.sum()), ri, dtype=np.int16), author=cols["author"][keep],
                              s=cols["m_%s_s" % measure][keep].astype(np.float64), e=e[keep].astype(np.float64),
                              n=cols["m_%s_n" % measure][keep].astype(np.float64),
                              lift=(cols["lift_cm"][keep].astype(np.float64) if "lift_cm" in cols
                                    else np.full(int(keep.sum()), np.nan)),
                              hand=((cols["hand_touch_L"] | cols["hand_touch_R"])[keep] if "hand_touch_L" in cols
                                    else np.zeros(int(keep.sum()), bool))))
    if not parts:
        return dict(time=np.zeros(0), step=np.zeros(0, np.int64), body=np.zeros(0, np.int16),
                    rec=np.zeros(0, np.int16), author=np.zeros(0, np.int8), s=np.zeros(0), e=np.zeros(0),
                    n=np.zeros(0), lift=np.zeros(0), hand=np.zeros(0, bool))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(out["time"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def _num(x) -> Optional[float]:
    x = float(x)
    return round(x, 4) if np.isfinite(x) else None


def summary(sel: Dict[str, np.ndarray]) -> dict:
    """Medians at start, end and after next; the gap closed, (sum of start values - sum of end values) / sum of start
    values over commands with both; the gain kept after the next command, (sum start - sum after next) /
    (sum start - sum end) over commands with all three. The start and end medians are taken over the same commands, the
    ones with both (every command where none has a start), so a page's "typical command" pair describes the commands it
    counts (the independent review of Oct 19, 2026, round 7)."""
    s, e, n = sel["s"], sel["e"], sel["n"]
    both = np.isfinite(s) & np.isfinite(e)
    three = both & np.isfinite(n)
    closed = (float(s[both].sum() - e[both].sum()) / float(s[both].sum())) if both.any() and s[both].sum() > 0 else None
    gain = float(s[three].sum() - e[three].sum()) if three.any() else 0.0
    kept = (float(s[three].sum() - n[three].sum()) / gain) if three.any() and gain > 0 else None
    return dict(commands=int(len(e)), with_start=int(both.sum()), with_next=int(three.sum()),
                median_start=_num(np.median(s[both])) if both.any() else None,
                median_end=_num(np.median(e[both] if both.any() else e)) if len(e) else None,
                median_next=_num(np.median(n[three])) if three.any() else None,
                gap_closed=_num(closed) if closed is not None else None,
                gain_kept=_num(kept) if kept is not None else None)


def series(recordings: List[dict], body, measure: str, author: str, t0=None, t1=None,
           per_bin: int = 40, max_bins: int = 60, touch: Optional[str] = None) -> dict:
    """Per recording, bins of consecutive commands (at least `per_bin` commands, at most `max_bins` bins): each with its
    time span, model-step span, count, median start, end and after-next values, the middle half of end values (25th and
    75th percentiles), the share of the gap its commands closed (closed: (sum of start values - sum of end values) / sum
    of start values over its commands with both, as summary()), and the smallest-error and largest-error command."""
    sel = selection(recordings, body, measure, author, t0, t1, touch)
    out = []
    for ri, rec in enumerate(recordings):
        m = sel["rec"] == ri
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        k = int(min(max_bins, max(1, len(idx) // per_bin)))
        bins = []
        for part in np.array_split(idx, k):
            if not len(part):
                continue
            e = sel["e"][part]
            s = sel["s"][part]
            nn = sel["n"][part]
            lo, hi = int(part[np.argmin(e)]), int(part[np.argmax(e)])
            both = np.isfinite(s) & np.isfinite(e)
            closed = (float(s[both].sum() - e[both].sum()) / float(s[both].sum())
                      if both.any() and s[both].sum() > 0 else None)
            bins.append(dict(t=float(np.median(sel["time"][part])), t0=float(sel["time"][part[0]]),
                             closed=_num(closed) if closed is not None else None, with_start=int(both.sum()),
                             t1=float(sel["time"][part[-1]]), step0=int(sel["step"][part[0]]),
                             step1=int(sel["step"][part[-1]]), n=int(len(part)),
                             start=_num(np.nanmedian(s)) if np.isfinite(s).any() else None,
                             end=_num(np.median(e)), p25=_num(np.percentile(e, 25)), p75=_num(np.percentile(e, 75)),
                             next=_num(np.nanmedian(nn)) if np.isfinite(nn).any() else None,
                             best=_ref(sel, lo, recordings), worst=_ref(sel, hi, recordings)))
        sub = {kk: v[idx] for kk, v in sel.items()}
        st = rec.get("settings") or {}
        out.append(dict(recording=rec["id"], label=rec["label"], kind=rec["kind"], bins=bins, summary=summary(sub),
                        settings=st.get("label"), settings_key=st.get("key"), other=st.get("other") or {}))
    name, unit, how = C.MEASURES[measure]
    groups = {}
    for x in sorted(out, key=lambda x: x["bins"][0]["t"]):
        groups.setdefault(x["settings_key"], []).append(x)
    # one summary per settings group (the second review of Oct 18, 2026, finding 1): a summary pooled over runs with
    # different control settings mixes separate examples, so the page shows the pooled one only for a single group
    key_of = [(rec.get("settings") or {}).get("key") for rec in recordings]
    summary_by_settings = []
    for key, g in groups.items():
        m = np.isin(sel["rec"], [ri for ri, k in enumerate(key_of) if k == key])
        summary_by_settings.append(dict(settings=g[0]["settings"], settings_key=key, runs=len(g),
                                        summary=summary({kk: v[m] for kk, v in sel.items()})))
    # earliest and latest run WITHIN each settings group (runs with other settings are separate examples), with the
    # learner settings that still differ between the two (sources.settings_of: other)
    by_settings = [dict(settings=g[0]["settings"], runs=len(g),
                        earliest=dict(label=g[0]["label"], t=g[0]["bins"][0]["t"], gap_closed=g[0]["summary"]["gap_closed"]),
                        latest=dict(label=g[-1]["label"], t=g[-1]["bins"][0]["t"], gap_closed=g[-1]["summary"]["gap_closed"]),
                        other_differs=[dict(name=k, earliest=g[0]["other"].get(k), latest=g[-1]["other"].get(k))
                                       for k in sorted(set(g[0]["other"]) | set(g[-1]["other"]))
                                       if g[0]["other"].get(k) != g[-1]["other"].get(k)])
                   for g in groups.values()]
    return dict(measure=measure, name=name, unit=unit, definition=how, author=author, body=body,
                series=out, summary=summary(sel), summary_by_settings=summary_by_settings, by_settings=by_settings,
                t0=t0, t1=t1, time_range=[float(sel["time"][0]), float(sel["time"][-1])] if len(sel["time"]) else None)


def _ref(sel: Dict[str, np.ndarray], i: int, recordings: List[dict]) -> dict:
    rec = recordings[int(sel["rec"][i])]
    return dict(recording=rec["id"], label=rec["label"], kind=rec["kind"], body=int(sel["body"][i]),
                step=int(sel["step"][i]), time=float(sel["time"][i]), author=AUTHOR_NAMES.get(int(sel["author"][i])),
                start=_num(sel["s"][i]), end=_num(sel["e"][i]), next=_num(sel["n"][i]),
                settings=(rec.get("settings") or {}).get("label"), hand_touch=bool(sel["hand"][i]),
                lift_cm=_num(sel["lift"][i]))


def command_at(recordings: List[dict], body, measure: str, author: str, t0=None, t1=None, index: Optional[int] = None,
               pick: Optional[str] = None, at_time: Optional[float] = None, touch: Optional[str] = None) -> dict:
    """One command of the selection: by position (index, 0 = earliest), by pick ('best' = smallest error at its end,
    'median', 'worst' = largest) or nearest to a time. Returns it with the selection's size and its position."""
    sel = selection(recordings, body, measure, author, t0, t1, touch)
    n = len(sel["e"])
    if not n:
        return dict(count=0, command=None)
    if pick == "best":
        i = int(np.argmin(sel["e"]))
    elif pick == "worst":
        i = int(np.argmax(sel["e"]))
    elif pick == "median":
        i = int(np.argsort(sel["e"], kind="stable")[n // 2])
    elif at_time is not None:
        i = int(np.argmin(np.abs(sel["time"] - float(at_time))))
    else:
        i = int(min(max(0, int(index or 0)), n - 1))
    return dict(count=n, index=i, command=_ref(sel, i, recordings))
